#!/usr/bin/env python3
"""PE fusion 2단계: PE 4개 안에서 묶기 vs 나누기 처리량 비교 (compile만)

구성 (전체 문장 수를 맞춤):
  A: fusion 4 x 인스턴스 1개     B: fusion 2 x 인스턴스 2개     C: fusion 1 x 인스턴스 4개
  전체 16: A b16 / B b8x2 / C b4x4      전체 64: A b64 / B b32x2 / C b16x4

인스턴스마다 별도 프로세스 + 서로 다른 장치(furiosa:0 ~ n-1) + 새 전용 캐시.
각 프로세스는 set_fusion(해당 값)을 장치 사용 전에 한 번 호출한다 (bert_worker.py:42).
설정은 기존 측정과 같음: BF16, 길이 128, int32 id, int32 position_ids 명시 전달,
CPU 4D 마스크(passthrough 패치), cpu_fallback=False. 입력 문장은 전역 위치
p = 인스턴스 번호 x batch + i 로 SENTENCES[p % 16] (기존 run의 순환 규칙과 같음).

동시 시작: 모든 인스턴스가 컴파일(첫 호출)과 웜업 10회를 끝내고 ready 파일을
만들면 부모가 go 파일을 만들고, 각 인스턴스는 반복 200회를 돌린다. 반복 중에
furiosa-smi ps를 찍어 실제 사용 PE를 기록한다.

기록: 전체 처리량 = 전체 처리 문장 수(인스턴스 수 x batch x 반복) ÷
(가장 이른 시작 ~ 가장 늦은 종료; CLOCK_MONOTONIC이라 프로세스 간 비교 가능),
인스턴스별 반복 평균 ± SD와 첫 호출 시간, effective device_configuration.
인스턴스가 하나라도 실패하면 그 구성은 fail로 기록하고 다음 구성으로 넘어간다.

결과: runs/pe_throughput_<시각>/t<전체>_<구성>/inst*.json, results.jsonl
보고서 표는 bert_pe_throughput_report.py가 만든다.

사용법:
  python bert_pe_throughput.py                    # 전체 16, 64 x 구성 A, B, C
  python bert_pe_throughput.py --totals 16        # 전체 16만
  python bert_pe_throughput.py --configs A,C      # 일부 구성만
  python bert_pe_throughput.py --repeats 200      # 기본 200
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

WARMUP = 10
READY_TIMEOUT_S = 1800   # 전 인스턴스의 컴파일(첫 호출)+웜업 제한
RUN_TIMEOUT_S = 3600
TOTALS = [16, 64]
CONFIGS = {'A': (4, 1), 'B': (2, 2), 'C': (1, 4)}   # 이름 -> (fusion, 인스턴스 수); batch = 전체 // 인스턴스 수

EXTRA_ENV = {'HF_HUB_OFFLINE': '1'}
UNSET_ENV = ['TORCH_LOGS', 'FURIOSA_TORCH_LOG', 'FURIOSA_TORCH_PROFILE']


def command(args):
    # bert_run.py:51과 같음
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as e:
        return dict(command=args, error=repr(e))


def npu_busy(check):
    # bert_run.py:60과 같음
    return check.get('returncode') != 0 or any(
        line.strip().startswith('|') and line.strip().split('|')[1].strip().isdigit()
        for line in check.get('stdout', '').splitlines())


# ---------------------------------------------------------------- 자식(인스턴스)
def instance(a):
    import torch
    import furiosa.torch as ft
    rec = dict(device=a.device, instance=a.index, fusion_requested=a.fusion, batch=a.batch,
               repeats=a.repeats, status='running')
    try:
        # bert_worker.py:42와 같음: set_fusion은 장치/스트림/모델 사용 전에 프로세스당 한 번만.
        fusion = ft.set_fusion(a.fusion)
        rec['fusion'] = {k: getattr(fusion, k) for k in ['num_fusion', 'num_pe_per_cluster', 'num_chips', 'num_clusters_per_chip']}
        if fusion.num_fusion != a.fusion:
            raise RuntimeError(f'set_fusion returned num_fusion={fusion.num_fusion}, requested {a.fusion}')
        from bert_model import load_model, encode, SENTENCES
        if not ft.is_available() or a.device not in ft.get_available_devices():
            raise RuntimeError(f'{a.device} unavailable; available={ft.get_available_devices()}')
        cache = Path(a.cache)
        if cache.exists():
            raise RuntimeError('Never reuse an existing cache dir')   # bert_worker.py:54와 같음
        tok, model = load_model()
        sents = [SENTENCES[(a.index * a.batch + i) % len(SENTENCES)] for i in range(a.batch)]
        rec['sentence_indices'] = [(a.index * a.batch + i) % len(SENTENCES) for i in range(a.batch)]
        inp_cpu = encode(tok, sents, posid_int32=True)
        keys = ['input_ids', 'token_type_ids', 'attention_mask', 'position_ids']
        dev = torch.device(a.device)
        with ft.config.cpu_fallback(False), ft.config.cache_context(str(cache), summary=False), torch.inference_mode():
            rec['device_configuration'] = str(ft.get_device_configuration())
            rec['available_devices'] = ft.get_available_devices()
            model = model.to(dev)
            inp = {k: inp_cpu[k].to(dev) for k in keys}
            stream = ft.current_stream(a.device); stream.synchronize()
            runner = torch.compile(model, backend=ft.backend)   # bert_worker.py:86과 같음

            t0 = time.perf_counter_ns(); out = runner(**inp); stream.synchronize()
            rec['first_call_s'] = (time.perf_counter_ns() - t0) / 1e9
            t0 = time.perf_counter_ns()
            for _ in range(WARMUP):
                out = runner(**inp); stream.synchronize()
            rec['warmup_sum_ms'] = (time.perf_counter_ns() - t0) / 1e6

            Path(a.ready).touch()                       # 준비 완료 신호
            while not Path(a.go).exists():              # 동시 시작 신호 대기
                time.sleep(0.01)
            times = []
            rec['t_begin_mono_ns'] = time.monotonic_ns()   # CLOCK_MONOTONIC: 프로세스 간 비교 가능
            for _ in range(a.repeats):
                t0 = time.perf_counter_ns(); out = runner(**inp); stream.synchronize()
                times.append((time.perf_counter_ns() - t0) / 1e6)
            rec['t_end_mono_ns'] = time.monotonic_ns()
            y = out.logits.detach().to('cpu').float()
            if list(y.shape) != [a.batch, 2] or bool(y.isnan().any()) or bool(y.isinf().any()):
                raise RuntimeError(f'Invalid output: shape={list(y.shape)}')
        rec.update(repeat_mean_ms=statistics.mean(times), repeat_sd_ms=statistics.stdev(times),
                   repeat_ms=[round(t, 3) for t in times], status='ok')
    except BaseException as e:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        msg = str(e).strip().splitlines()
        rec.update(status='fail', error_type=type(e).__name__, error_head=msg[0][:300] if msg else '')
    Path(a.result).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0 if rec['status'] == 'ok' else 1


# ---------------------------------------------------------------- 부모
def run_config(run_dir, total, label, repeats, env):
    fusion, n_inst = CONFIGS[label]
    batch = total // n_inst
    d = run_dir / f't{total}_{label}'
    d.mkdir()
    go = d / 'go'
    rec = dict(config=label, total=total, fusion=fusion, n_instances=n_inst, batch=batch, repeats=repeats,
               devices=[f'furiosa:{i}' for i in range(n_inst)])
    before = command(['furiosa-smi', 'ps'])
    (d / 'smi_before.json').write_text(json.dumps(before))
    if npu_busy(before):
        rec.update(status='held', reason='NPU in use by another process (or furiosa-smi failed)')
        return rec
    procs = []
    for i in range(n_inst):
        cmd = [sys.executable, '-u', __file__, '--instance', '--index', str(i), '--device', f'furiosa:{i}',
               '--fusion', str(fusion), '--batch', str(batch), '--repeats', str(repeats),
               '--cache', str(d / f'cache_f{fusion}_inst{i}'), '--ready', str(d / f'ready{i}'),
               '--go', str(go), '--result', str(d / f'inst{i}.json')]
        log = (d / f'inst{i}.log').open('w')
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                          start_new_session=True, cwd=str(Path(__file__).resolve().parent)), log))
    try:
        t0 = time.monotonic()
        while not all((d / f'ready{i}').exists() for i in range(n_inst)):   # 전원 컴파일+웜업 완료 대기
            if any(p.poll() is not None for _, p, _ in procs):
                raise RuntimeError('an instance exited before ready (see inst*.log)')
            if time.monotonic() - t0 > READY_TIMEOUT_S:
                raise TimeoutError(f'ready timeout {READY_TIMEOUT_S}s')
            time.sleep(0.1)
        rec['go_mono_ns'] = time.monotonic_ns(); go.touch()                 # 동시 시작
        time.sleep(1.0)                                                     # 반복 실행 중 스냅샷
        (d / 'smi_during.json').write_text(json.dumps(command(['furiosa-smi', 'ps'])))
        for _, p, _ in procs:
            p.wait(timeout=RUN_TIMEOUT_S)
        insts = {str(i): json.loads((d / f'inst{i}.json').read_text()) for i in range(n_inst)}
        rec['instances'] = insts
        if all(v['status'] == 'ok' for v in insts.values()):
            begin = min(v['t_begin_mono_ns'] for v in insts.values())
            end = max(v['t_end_mono_ns'] for v in insts.values())
            rec['wall_s'] = (end - begin) / 1e9
            rec['sentences'] = n_inst * batch * repeats
            rec['throughput_sents_per_s'] = rec['sentences'] / rec['wall_s']
            rec['status'] = 'ok'
        else:
            rec['status'] = 'fail'
            rec['failed_instances'] = [i for i, v in insts.items() if v['status'] != 'ok']
    except BaseException as e:  # noqa: BLE001
        rec.update(status='fail', error=repr(e))
        for _, p, _ in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
        if isinstance(e, KeyboardInterrupt):
            raise
    finally:
        for _, _, log in procs:
            log.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--totals', type=lambda s: [int(x) for x in s.split(',')], default=TOTALS)
    ap.add_argument('--configs', type=lambda s: [x for x in CONFIGS if x in s.split(',')], default=list(CONFIGS))
    ap.add_argument('--repeats', type=int, default=200)
    ap.add_argument('--instance', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--index', type=int); ap.add_argument('--device'); ap.add_argument('--fusion', type=int)
    ap.add_argument('--batch', type=int); ap.add_argument('--cache'); ap.add_argument('--ready')
    ap.add_argument('--go'); ap.add_argument('--result')
    a = ap.parse_args()
    if a.instance:
        return instance(a)
    bad = [t for t in a.totals if any(t % CONFIGS[c][1] for c in a.configs)]
    if bad:
        ap.error(f'totals {bad} not divisible by instance counts of {a.configs}')

    run_dir = Path(__file__).resolve().parent / 'runs' / time.strftime('pe_throughput_%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True)
    env = dict(os.environ)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update(EXTRA_ENV)

    for total in a.totals:
        for label in a.configs:
            fusion, n_inst = CONFIGS[label]
            print(f'[t{total} {label}] fusion {fusion} x {n_inst} inst, b{total // n_inst} ... ', end='', flush=True)
            t0 = time.perf_counter()
            rec = run_config(run_dir, total, label, a.repeats, env)
            with (run_dir / 'results.jsonl').open('a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if rec.get('status') == 'ok':
                print(f"ok {rec['throughput_sents_per_s']:.1f} 문장/s ({time.perf_counter() - t0:.0f}s)")
            else:
                print(f"{rec.get('status')} ({time.perf_counter() - t0:.0f}s)")   # 실패해도 다음 구성으로
            if rec.get('status') == 'held':
                print('NPU_BUSY: 다른 프로세스가 NPU 사용 중; 중단'); return 3
    print(f'\n결과: {run_dir}/results.jsonl')
    print(f'보고서: python bert_pe_throughput_report.py --run-dir {run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

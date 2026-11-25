#!/usr/bin/env python3
"""동시 실행 확인: BERT compile 인스턴스 2개를 furiosa:0/furiosa:1에서 같이 돌리기

PE fusion 2단계(묶기 vs 나누기 처리량 비교, bert_pe_throughput.py)의 사전 확인.
세 단계를 차례로 돌린다:
  solo0: furiosa:0 혼자        solo1: furiosa:1 혼자        pair: 둘이 동시에
각 인스턴스는 별도 프로세스에서 set_fusion(1)을 장치 사용 전에 한 번 호출하고,
기존 측정과 같은 설정으로 compile을 만든다: BF16, 길이 128, int32 id,
int32 position_ids 명시 전달, CPU 4D 마스크(passthrough 패치), cpu_fallback=False,
프로세스마다 새 전용 캐시, batch 1. 입력은 세 단계 모두 같은 문장(SENTENCES[0])이다.

동시 시작: 자식이 컴파일(첫 호출)과 웜업 10회를 끝내면 ready 파일을 만들고
go 파일이 생길 때까지 기다린다. 부모는 모든 ready를 확인한 뒤 go를 만들고,
반복 실행 중에 furiosa-smi ps를 찍어 실제 사용 PE를 기록한다. 자식도
effective get_device_configuration()을 기록한다.

마지막에 pair와 solo의 반복 평균을 비교해 간섭(느려짐 배수)을 출력한다.
결과: runs/pair_check_<시각>/{solo0,solo1,pair}/inst*.json, results.jsonl, smi_*.json

사용법:
  python bert_pair_check.py                # repeats 50
  python bert_pair_check.py --repeats 200
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

FUSION = 1
WARMUP = 10
READY_TIMEOUT_S = 1800   # 컴파일(첫 호출)+웜업 제한
RUN_TIMEOUT_S = 1800
PHASES = [('solo0', [0]), ('solo1', [1]), ('pair', [0, 1])]   # 값은 furiosa:<n>의 n

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
    rec = dict(device=a.device, fusion_requested=FUSION, batch=a.batch, repeats=a.repeats, status='running')
    try:
        # bert_worker.py:42와 같음: set_fusion은 장치/스트림/모델 사용 전에 프로세스당 한 번만.
        fusion = ft.set_fusion(FUSION)
        rec['fusion'] = {k: getattr(fusion, k) for k in ['num_fusion', 'num_pe_per_cluster', 'num_chips', 'num_clusters_per_chip']}
        if fusion.num_fusion != FUSION:
            raise RuntimeError(f'set_fusion returned num_fusion={fusion.num_fusion}, requested {FUSION}')
        from bert_model import load_model, encode, SENTENCES
        if not ft.is_available() or a.device not in ft.get_available_devices():
            raise RuntimeError(f'{a.device} unavailable; available={ft.get_available_devices()}')
        cache = Path(a.cache)
        if cache.exists():
            raise RuntimeError('Never reuse an existing cache dir')   # bert_worker.py:54와 같음
        tok, model = load_model()
        inp_cpu = encode(tok, [SENTENCES[0]] * a.batch, posid_int32=True)   # 세 단계 모두 같은 입력
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
def run_phase(run_dir, name, dev_ids, repeats, env):
    d = run_dir / name
    d.mkdir()
    go = d / 'go'
    before = command(['furiosa-smi', 'ps'])
    (d / 'smi_before.json').write_text(json.dumps(before))
    if npu_busy(before):
        return dict(phase=name, status='held', reason='NPU in use by another process (or furiosa-smi failed)')
    procs = []
    for i in dev_ids:
        cmd = [sys.executable, '-u', __file__, '--instance', '--device', f'furiosa:{i}',
               '--cache', str(d / f'cache_f{FUSION}_inst{i}'), '--ready', str(d / f'ready{i}'),
               '--go', str(go), '--result', str(d / f'inst{i}.json'), '--batch', '1', '--repeats', str(repeats)]
        log = (d / f'inst{i}.log').open('w')
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                          start_new_session=True, cwd=str(Path(__file__).resolve().parent)), log))
    rec = dict(phase=name, devices=[f'furiosa:{i}' for i in dev_ids], repeats=repeats)
    try:
        t0 = time.monotonic()
        while not all((d / f'ready{i}').exists() for i in dev_ids):   # 전원 컴파일+웜업 완료 대기
            if any(p.poll() is not None for _, p, _ in procs):
                raise RuntimeError('an instance exited before ready (see inst*.log)')
            if time.monotonic() - t0 > READY_TIMEOUT_S:
                raise TimeoutError(f'ready timeout {READY_TIMEOUT_S}s')
            time.sleep(0.1)
        go_mono_ns = time.monotonic_ns(); go.touch()                  # 동시 시작
        time.sleep(1.0)                                               # 반복 실행 중 스냅샷
        (d / 'smi_during.json').write_text(json.dumps(command(['furiosa-smi', 'ps'])))
        for _, p, _ in procs:
            p.wait(timeout=RUN_TIMEOUT_S)
        insts = {str(i): json.loads((d / f'inst{i}.json').read_text()) for i in dev_ids}
        rec['instances'] = insts
        rec['go_mono_ns'] = go_mono_ns
        if all(v['status'] == 'ok' for v in insts.values()):
            rec['status'] = 'ok'
            rec['wall_s'] = (max(v['t_end_mono_ns'] for v in insts.values()) - go_mono_ns) / 1e9
        else:
            rec['status'] = 'fail'
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
    ap.add_argument('--repeats', type=int, default=50)
    ap.add_argument('--instance', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--device'); ap.add_argument('--cache'); ap.add_argument('--ready')
    ap.add_argument('--go'); ap.add_argument('--result'); ap.add_argument('--batch', type=int, default=1)
    a = ap.parse_args()
    if a.instance:
        return instance(a)

    run_dir = Path(__file__).resolve().parent / 'runs' / time.strftime('pair_check_%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True)
    env = dict(os.environ)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update(EXTRA_ENV)

    results = []
    for name, dev_ids in PHASES:
        print(f'[{name}] furiosa:{dev_ids} ... ', end='', flush=True)
        t0 = time.perf_counter()
        rec = run_phase(run_dir, name, dev_ids, a.repeats, env)
        results.append(rec)
        with (run_dir / 'results.jsonl').open('a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"{rec.get('status')} {time.perf_counter() - t0:.0f}s")
        if rec.get('status') == 'held':
            print('NPU_BUSY: 다른 프로세스가 NPU 사용 중; 중단'); return 3

    print(f'\n결과: {run_dir}/results.jsonl')
    by = {r['phase']: r for r in results}
    solo = {}
    for name, dev_ids in PHASES:
        r = by[name]
        if r['status'] != 'ok':
            print(f"{name}: {r['status']} {r.get('error', '')}"); continue
        for i in dev_ids:
            v = r['instances'][str(i)]
            m, s = v['repeat_mean_ms'], v['repeat_sd_ms']
            pes = v['device_configuration'].split(', Single')[0]
            line = f"{name} furiosa:{i}: repeat {m:.3f} ± {s:.3f} ms, first {v['first_call_s']:.1f} s, PE {pes}"
            if name.startswith('solo'):
                solo[i] = m
            elif i in solo:
                line += f'  → solo 대비 x{m / solo[i]:.3f}'
            print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

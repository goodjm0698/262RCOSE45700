"""Build BERT/RESULTS.md from a finished run.
Usage: python bert_report.py --run-dir runs/ID [--notes notes.md]
"""
import argparse
from pathlib import Path
from common import read_json

def f(x, d=2):
    if x is None: return '—'
    if isinstance(x, (int,)) and not isinstance(x, bool): return f'{x:,}'
    return f'{x:,.{d}f}'


def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n|'+'---|'*len(headers)+'\n'+''.join('| '+' | '.join(str(c) for c in r)+' |\n' for r in rows)+'\n'


def rng(v, d=2, scale=1.0):
    if not v or v['median'] is None: return '—'
    return f"{f(v['median']*scale, d)} ({f(v['minimum']*scale, d)}~{f(v['maximum']*scale, d)})"


def tget(d, i):
    return d.get(str(i), d.get(i)) if d else None     # JSON turns int trial keys into strings


def msd(p, d=3):
    return '—' if not p or p.get('mean_ms') is None else f"{f(p['mean_ms'], d)} ± {f(p.get('stdev_ms'), d)}"


def report(run, notes=None):
    run = Path(run); cfg = read_json(run/'config.json'); m = read_json(run/'manifest.json')
    be = read_json(run/'break_even_summary.json'); oc = read_json(run/'output_comparison.json')['pairs']
    cp = run/'cumulative_validation_summary.json'
    cum = read_json(cp) if cp.exists() else dict(batches=[])
    cum_ran = any(w['stage'] == 'cumulative' and w.get('started_utc') for w in m['workers'])
    cumb = lambda b: next((x for x in cum['batches'] if x['batch'] == b), {}) if cum_ran else {}   # {} -> '—'
    batches = {b['batch']: b for b in be['batches']}
    bl = cfg.get('batches', [1, 4])
    devcfg = next((s for s in [read_json(p).get('effective', {}).get('device_configuration') for p in sorted((run/'cases').glob('*/result.json'))] if s), '—')
    attn = next((s for s in [read_json(p).get('attn_implementation') for p in sorted((run/'cases').glob('*/result.json'))] if s), '—')
    t = f"# BERT-base NPU 측정 결과 (ViT 방식)\n\n실행 ID `{cfg['run_id']}`. 원본은 `BERT/vitstyle/runs/{cfg['run_id']}/`에 있음.\n\n"
    t += "## 실험 조건\n\n"
    t += (f"- 모델: `{cfg['model']}`, BF16, 길이 {cfg['seq_len']}, `cpu_fallback=False`, attention 구현 `{attn}`\n"
          f"- 마스크: `create_bidirectional_mask`를 passthrough로 패치하고 4D additive mask를 CPU에서 만들어 NPU로 넘김 (step4 방식)\n"
          f"- 입력: step4의 실제 문장을 128로 패딩함. batch B는 1~B번 문장 (batch {bl}), id는 int32\n"
          f"- PE: 별도 설정 없이 SDK 기본값 사용, `furiosa:0`. 실제 적용값은 `{devcfg.split(', Single')[0]})`라서 1 PE임\n"
          f"- 한 번 실행할 때마다 첫 호출 1회 + 워밍업 {cfg['warmup']}회 + 반복 {cfg['repeats']}회. 조건마다 새 프로세스로 {cfg['trials']}세트 실행함. 실행 순서는 회차마다 바꿈\n"
          f"- 대표값: 반복 {cfg['trials']*cfg['repeats']}회(세트 {cfg['trials']}개 × {cfg['repeats']}회)를 모두 모은 평균 ± 표준편차. 중앙값과 세트별 평균도 같이 적음. 정상 종료한 세트만 포함함\n"
          f"- 첫 호출은 성능 비교에서 빼고 기록만 함. 첫 호출·워밍업 합·A는 세트 3개의 중앙값 (min~max)\n"
          f"- 캐시: 실행마다 `runs/<id>/caches/<stage>/<case>`를 새로 만들어 씀. 팀 공용 캐시는 사용하지 않음\n"
          f"- 타이머: `stream.synchronize()` → `perf_counter_ns` → `runner(**inputs)` → `stream.synchronize()`. 입력 전송과 출력 CPU 복사는 타이머 밖임\n"
          f"- A = runner 준비(S) + 첫 호출(F) + 워밍업 합(W). 모델 로드와 NPU 전송 시간은 제외함. T(N) = A + N × 반복 평균\n"
          f"- NPU 사용 시간 합계 {m['npu_budget_used_s']:.0f}초 (예산 {cfg['total_budget_s']}초)\n\n")
    bad = [w for w in m['workers'] if w['stage'] == 'performance' and w['status'] != 'ok']
    if bad:
        t += "## 정상 종료하지 않은 성능 측정 worker\n\n"+table(['case', '상태', '사유'], [
            [w['case_id'], w['status'], (w.get('reason') or '')+(f" ({w['skipped_because']})" if w.get('skipped_because') else '')] for w in bad])
    trials = list(range(1, cfg['trials']+1))
    t += f"## 표 1. 반복 시간 (ms, 대표값은 {cfg['trials']*cfg['repeats']}회 전체 평균 ± 표준편차)\n\n"
    rows = []
    for b in bl:
        for mode in ['eager', 'compile']:
            p = batches[b].get('pooled_repeat_by_mode', {}).get(mode)
            rows.append([b, mode, msd(p), f(p and p['median_ms'], 3), f(p and p['n'])]
                        +[f(tget(p and p['trial_means_ms'], i), 3) for i in trials])
    t += table(['batch', '모드', '평균 ± 표준편차', '중앙값', 'n']+[f'세트{i} 평균' for i in trials], rows)
    t += "## 표 1-2. 첫 호출·워밍업·A (기록용, 세트 3개의 중앙값, 괄호는 min~max)\n\n"
    rows = [[b, mode, rng(batches[b]['first_call_ms_by_mode'][mode], 2, 1e-3)+' s', rng(batches[b]['warmup_sum_ms_by_mode'][mode]),
             rng(batches[b]['A_ms_by_mode'][mode], 2, 1e-3)+' s'] for b in bl for mode in ['eager', 'compile']]
    t += table(['batch', '모드', '첫 호출', f"워밍업 {cfg['warmup']}회 합 (ms)", 'A'], rows)
    t += "## 표 2. 배속 (eager 반복 평균 / compile 반복 평균)\n\n"
    t += table(['batch', '대표 (전체 평균끼리)']+[f'세트{i}' for i in trials],
               [[b, f(batches[b].get('pooled_speedup'), 3)]+[f(tget(batches[b]['per_trial_speedup'], i), 3) for i in trials] for b in bl])
    t += "대표 배속은 전체 평균끼리 나눈 값임. 세트별 값은 같은 세트의 eager와 compile 평균끼리 나눈 값임.\n\n"
    t += "## 표 3. 손익분기점\n\n"
    rows = []
    for b in bl:
        c = cumb(b)
        keep = {True: '유지됨', False: '유지 안 됨'}.get(c.get('maintained_after_crossing'), '—')
        rows.append([b, rng(batches[b]['N_win'], 0), f(c.get('first_observed_compile_advantage_N')),
                     f"{keep} (N={f(c.get('completed_common_calls'))}까지 관측)" if c else '—', c.get('status', '미실행')])
    t += table(['batch', '예측 N_win 중앙값 (min~max)', '실측 교차 N', '교차 후 우위', '상태'], rows)
    if not cum_ran: t += "이번 실행에서는 cumulative 단계를 돌리지 않았음. 실측 칸은 비어 있고 예측값만 있음.\n\n"
    t += f"N은 첫 호출과 워밍업 {cfg['warmup']}회를 마친 뒤 추가로 호출한 횟수임. 실측은 별도 프로세스와 새 캐시에서 대표 회차(N_win이 중앙값인 회차)의 예측값과 비교함.\n\n"
    t += "## 표 4. 출력 비교 (eager vs compile, BF16 logits를 CPU FP32로 바꿔 비교)\n\n"
    rows = []
    for st_, label in [('outcmp', '16문장 (출력 비교 worker)'), ('performance', '성능 측정 출력 (마지막 반복, 3회 중 최악값)')]:
        for b in bl:
            ps = [p for p in oc if (p['stage'], p['batch']) == (st_, b) and p.get('structural_finite_ok')]
            if not ps: rows.append([label, b, '—', '—', '—', '—', '—']); continue
            n = ps[0]['compared_samples']
            rows.append([label, b, f(max(p['max_abs_error'] for p in ps), 4), f(max(p['mean_abs_error'] for p in ps), 4),
                         f(min(p['cosine_mean'] for p in ps), 6), f(min(p['cosine_min'] for p in ps), 6),
                         f"{min(p['top1_matches'] for p in ps)}/{n}"])
    t += table(['입력', 'batch', '최대 절대오차', '평균 절대오차', 'cosine 평균', 'cosine 최소', 'Top-1 일치'], rows)
    t += "허용 오차 기준은 아직 정하지 않았으므로 합격 판정은 하지 않음(pending). BERT logits는 2차원이라 cosine이 높게 나오기 쉬움.\n\n"
    if notes and Path(notes).exists(): t += Path(notes).read_text()
    out = run.parents[2]/'RESULTS.md'    # BERT/RESULTS.md
    out.write_text(t)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--notes', type=Path)
    a = p.parse_args(); print(report(a.run_dir.resolve(), a.notes))

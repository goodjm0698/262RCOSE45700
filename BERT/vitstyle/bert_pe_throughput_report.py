#!/usr/bin/env python3
"""PE fusion 2단계 결과 표: 구성별 처리량, 인스턴스별 반복 시간, 혼자 대비 느려짐

bert_pe_throughput.py의 run(results.jsonl)을 읽어 전체 문장 수별로 표를 만든다.
'혼자 돌 때' 기준은 --baseline-runs로 준 기존 fusion run들(읽기만 함)에서
같은 (fusion, batch)의 compile 반복 평균을 찾아 쓴다. 없으면 '—'.
구성 B(fusion 2, b8/b32)의 기준은 아직 측정된 run이 없으므로, 필요하면
  python bert_run.py run --fusion 2 --modes compile --batches 8,32 --trials 1
로 만든 run을 --baseline-runs에 추가하면 된다.

사용법:
  python bert_pe_throughput_report.py --run-dir runs/pe_throughput_ID \\
    --baseline-runs runs/20261110T140237Z-61d998 runs/20261113T051434Z-fe0cc4 \\
                    runs/20261119T030722Z-d74488 runs/20261119T032413Z-616241 \\
    [--out ../RESULTS_pe_throughput.md]
--out이 없으면 표를 표준 출력으로만 낸다.
"""
import argparse
import json
from pathlib import Path

from common import read_json
from bert_report import f, table, mode_stats, run_fusion


def baselines(runs):
    """{(fusion, batch): pooled compile repeat stats}; 같은 키가 여러 run에 있으면 나중 run이 이김. 읽기만 함."""
    cells = {}; src = {}
    for r in runs:
        rc = read_json(Path(r) / 'config.json'); fu = run_fusion(rc)
        for (mode, b), p in mode_stats(r).items():
            if mode == 'compile':
                cells[(fu, b)] = p; src[(fu, b)] = rc['run_id']
    return cells, src


def report(run_dir, base_runs, out=None):
    run_dir = Path(run_dir)
    recs = [json.loads(line) for line in (run_dir / 'results.jsonl').read_text().splitlines()]
    cells, src = baselines(base_runs)
    used = {}
    t = f"# PE fusion 2단계: 묶기 vs 나누기 처리량 비교\n\n원본은 `BERT/vitstyle/runs/{run_dir.name}/`에 있음. compile만 측정함.\n\n"
    for total in sorted({r['total'] for r in recs}):
        sub = [r for r in recs if r['total'] == total]
        rep = sub[0]['repeats']
        t += f"## 전체 {total}문장 (인스턴스당 반복 {rep}회, 전체 처리 문장 = 인스턴스 수 x batch x {rep})\n\n"
        tput_a = next((r.get('throughput_sents_per_s') for r in sub if r['config'] == 'A' and r.get('status') == 'ok'), None)
        rows = []
        for r in sub:
            head = [r['config'], f"{r['fusion']} x {r['n_instances']}", r['batch']]
            if r.get('status') != 'ok':
                rows.append(head + [r.get('status', '—'), '—', '—', '—', '—', '—']); continue
            insts = [r['instances'][str(i)] for i in range(r['n_instances'])]
            per = '; '.join(f"{v['repeat_mean_ms']:.3f} ± {v['repeat_sd_ms']:.3f}" for v in insts)
            mean_of_means = sum(v['repeat_mean_ms'] for v in insts) / len(insts)
            base = cells.get((r['fusion'], r['batch']))
            if base:
                used[(r['fusion'], r['batch'])] = src[(r['fusion'], r['batch'])]
            rows.append(head + [f"{r['throughput_sents_per_s']:.1f}",
                                f(tput_a and r['throughput_sents_per_s'] / tput_a, 3),
                                per,
                                '—' if not base else f"{base['mean_ms']:.3f} → x{mean_of_means / base['mean_ms']:.3f}",
                                '; '.join(f"{v['first_call_s']:.1f}" for v in insts),
                                ', '.join(r['devices'])])
        t += table(['구성', 'fusion x 인스턴스', 'batch', '처리량 (문장/s)', 'A 대비',
                    '인스턴스별 반복 평균 ± SD (ms)', '혼자 평균 (ms) → 느려짐', '첫 호출 (s)', '장치'], rows)
    t += ("처리량 = 전체 처리 문장 수 ÷ (가장 이른 인스턴스 시작 ~ 가장 늦은 인스턴스 종료). "
          "느려짐 = 동시 실행의 인스턴스 반복 평균(인스턴스 간 평균) ÷ 같은 (fusion, batch)를 혼자 돌린 기존 run의 compile 반복 평균. "
          "실제 사용 PE는 run 폴더의 `t*_*/smi_during.json`과 인스턴스별 `device_configuration`에 있음.\n\n")
    if used:
        t += "혼자 기준 run (읽기만 함): " + '; '.join(
            f"`{rid}` (fusion {fu} batch {b})" for (fu, b), rid in sorted(used.items())) + ".\n"
    missing = sorted({(r['fusion'], r['batch']) for r in recs if r.get('status') == 'ok'} - set(cells))
    if missing:
        t += "혼자 기준이 없는 조건: " + ', '.join(f"fusion {fu} batch {b}" for fu, b in missing) + \
             " — `bert_run.py run --fusion <n> --modes compile --batches <b> --trials 1`로 만든 run을 --baseline-runs에 추가.\n"
    print(t)
    if out:
        Path(out).write_text(t)
        print(f'저장: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--baseline-runs', type=Path, nargs='*', default=[],
                   help='기존 fusion run들 (혼자 기준; 읽기만 함)')
    p.add_argument('--out', type=Path, help='markdown 저장 경로 (없으면 표준 출력만)')
    a = p.parse_args()
    report(a.run_dir.resolve(), [r.resolve() for r in a.baseline_runs], a.out)

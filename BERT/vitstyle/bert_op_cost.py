#!/usr/bin/env python3
"""NPU eager cost of ONE aten op call: dropout (eval) and linear, batch 1 and 16.

Follows bert_op_count.py: B1..B8 eager forward ~18.7 ms / 179 NPU ops ~ 105 us per op. This
measures what a single eager op call costs on the NPU.

Each case runs once in its own fresh process with a new private cache
(runs/op_cost_<time>/cache/<case>) under the measurement settings of bert_worker.py:47
(cpu_fallback(False), cache_context, inference_mode), eager only, BF16.
  input   : x = randn(B, 128, 768) BF16 (seed 0), already on the NPU before timing
  dropout : nn.Dropout(p=0.1).eval()  -> aten.dropout(train=False), a PrivateUse1 op in furiosa
  linear  : nn.Linear(768, 768) BF16 (seed 0), the shape of BERT's Q/K/V/output projections
Schedule: first call 1 (recorded only) + warmup 10 + measured 200, each call timed like the
measurement code (common.timed_call: synchronize -> perf_counter_ns -> op -> synchronize).
Extra, after that: 200 calls back to back with one synchronize at the end (total / 200), which
is closer to how ops run inside a forward (no per-op synchronize).
Outputs are checked outside the timer: dropout(eval) == input, linear vs CPU FP32 reference.

Usage: python bert_op_cost.py [--cases dropout:1,dropout:16,linear:1,linear:16]
Before each case `furiosa-smi ps` is checked; if another process uses the NPU it stops.
"""
import argparse
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEVICE = 'furiosa:0'
SEQ, HIDDEN = 128, 768
WARMUP, CALLS = 10, 200
TIMEOUT_S = 1800   # same per-worker limit as bert_run.py
DEFAULT_CASES = 'dropout:1,dropout:16,linear:1,linear:16'


def npu_busy():
    # Same rule as bert_run.npu_busy
    p = subprocess.run(['furiosa-smi', 'ps'], capture_output=True, text=True, timeout=30)
    busy = p.returncode != 0 or any(l.strip().startswith('|') and l.strip().split('|')[1].strip().isdigit()
                                    for l in p.stdout.splitlines())
    return busy, p.stdout+p.stderr


def stats_us(ms):
    us = [v*1000 for v in ms]
    return dict(n=len(us), mean_us=st.mean(us), stdev_us=st.stdev(us) if len(us) > 1 else None,
                median_us=st.median(us), min_us=min(us), max_us=max(us))


def child(op, B, run_dir):
    rec = dict(op=op, batch=B, input_shape=[B, SEQ, HIDDEN], dtype='torch.bfloat16', warmup=WARMUP, calls=CALLS, status='running')
    try:
        import torch
        import furiosa.torch as ft
        from common import timed_call
        cache = Path(run_dir)/'cache'/f'{op}_b{B}'
        if cache.exists(): raise RuntimeError('Never reuse an existing cache dir')
        rec['cache_dir'] = str(cache)
        torch.manual_seed(0)
        x_cpu = torch.randn(B, SEQ, HIDDEN).to(torch.bfloat16)
        if op == 'dropout':
            mod = torch.nn.Dropout(p=0.1).eval()
        else:
            mod = torch.nn.Linear(HIDDEN, HIDDEN).to(torch.bfloat16).eval()
        with torch.inference_mode():
            ref = (x_cpu.float() if op == 'dropout' else torch.nn.functional.linear(x_cpu.float(), mod.weight.float(), mod.bias.float()))
        dev = torch.device(DEVICE)
        with ft.config.cpu_fallback(False), ft.config.cache_context(str(cache), summary=False), torch.inference_mode():
            rec['cpu_fallback'] = ft.config.get_cpu_fallback()
            rec['device_configuration'] = str(ft.get_device_configuration())
            if rec['cpu_fallback']: raise RuntimeError('cpu_fallback must be False')
            mod = mod.to(dev); x = x_cpu.to(dev)
            stream = ft.current_stream(DEVICE); stream.synchronize()
            phases = {'first': [], 'warmup': [], 'measure': []}
            for phase, n in [('first', 1), ('warmup', WARMUP), ('measure', CALLS)]:
                for _ in range(n):
                    y, ms, ok = timed_call(mod, x, stream.synchronize)
                    if not ok: raise RuntimeError('invalid timing')
                    phases[phase].append(ms)
            # extra: back to back, one synchronize at the end
            stream.synchronize(); t0 = time.perf_counter_ns()
            for _ in range(CALLS): y = mod(x)
            stream.synchronize(); b2b_ms = (time.perf_counter_ns()-t0)/1e6
            out = y.detach().to('cpu').float(); stream.synchronize()
        rec.update(first_call_ms=phases['first'][0], warmup_ms=phases['warmup'], measure_ms=phases['measure'],
                   per_call=stats_us(phases['measure']), back_to_back_total_ms=b2b_ms, back_to_back_us_per_call=b2b_ms*1000/CALLS,
                   out_shape=list(out.shape), max_abs_diff_vs_cpu=float((out-ref).abs().max()), status='ok')
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        msg = str(e).strip().splitlines()
        rec.update(status='fail', error_type=type(e).__name__, error_head=msg[0][:300] if msg else '')
    with open(Path(run_dir)/'results.jsonl', 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False)+'\n')


def parent(cases):
    run_dir = HERE/'runs'/time.strftime('op_cost_%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True)
    env = dict(os.environ)
    for k in ['TORCH_LOGS', 'FURIOSA_TORCH_LOG', 'FURIOSA_TORCH_PROFILE']: env.pop(k, None)   # bert_run.py launch()
    for op, B in cases:
        busy, smi = npu_busy()
        if busy:
            print(f'NPU in use (or furiosa-smi failed); stopped before {op} B{B}:\n{smi}'); break
        print(f'[{op} B{B}] ... ', end='', flush=True); t0 = time.perf_counter()
        with open(run_dir/f'{op}_b{B}.log', 'w') as lf:
            try:
                subprocess.run([sys.executable, '-u', __file__, '--child', f'{op}:{B}', '--run-dir', str(run_dir)],
                               stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=TIMEOUT_S, start_new_session=True)
            except subprocess.TimeoutExpired:
                with open(run_dir/'results.jsonl', 'a') as f:
                    f.write(json.dumps(dict(op=op, batch=B, status='timeout', timeout_s=TIMEOUT_S))+'\n')
        print(f'{time.perf_counter()-t0:.0f}s (wall, includes compile)')
    rp = run_dir/'results.jsonl'
    print(f'\nresults: {rp}\n')
    print(f'| op | batch | first call ms | per call (sync each) mean ± SD µs | median µs | back-to-back µs/call | max|diff| vs CPU FP32 |')
    print('|---|---|---|---|---|---|---|')
    for line in (rp.read_text().splitlines() if rp.exists() else []):
        r = json.loads(line)
        if r['status'] == 'ok':
            p = r['per_call']
            print(f"| {r['op']} | {r['batch']} | {r['first_call_ms']:.1f} | {p['mean_us']:.1f} ± {p['stdev_us']:.1f} | {p['median_us']:.1f} "
                  f"| {r['back_to_back_us_per_call']:.1f} | {r['max_abs_diff_vs_cpu']:.4f} |")
        elif r['status'] == 'fail':
            print(f"| {r['op']} | {r['batch']} | FAIL {r['error_type']}: {r['error_head'][:80]} (log {run_dir/(r['op']+'_b'+str(r['batch'])+'.log')}) | | | | |")
        else:
            print(f"| {r['op']} | {r['batch']} | TIMEOUT after {r['timeout_s']} s | | | | |")


def parse_cases(s):
    out = []
    for item in s.split(','):
        op, b = item.split(':')
        if op not in ('dropout', 'linear') or int(b) < 1: raise SystemExit(f'bad case {item}')
        out.append((op, int(b)))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=DEFAULT_CASES)
    ap.add_argument('--child', help=argparse.SUPPRESS); ap.add_argument('--run-dir', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        op, b = a.child.split(':'); child(op, int(b), a.run_dir)
    else:
        parent(parse_cases(a.cases))

#!/usr/bin/env python3
"""Full BERT batch 4: does it run end to end on NPU in eager and compile? (no timing)

Each mode runs once in its own fresh process with a new private cache
(runs/b4_check_<time>/cache/<mode>), using the same NPU settings as bert_worker.py:47/74
(cpu_fallback(False), cache_context, inference_mode, torch.compile(backend=ft.backend))
and the same inputs as the measurement (bert_model.encode: step4 sentences 1-4, length 128,
int32 ids, CPU 4D additive BF16 mask).
Reference: CPU FP32 model with the ORIGINAL (unpatched) HF mask path and 2D mask, as in
step4_compare.py. Prints max |logit diff| vs CPU FP32 and label agreement.

Usage (from anywhere):
  python bert_b4_check.py                 # eager then compile
  python bert_b4_check.py --modes compile
Before each mode `furiosa-smi ps` is checked; if another process uses the NPU it stops.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
DEVICE = 'furiosa:0'
BATCH = 4
TIMEOUT_S = 1800   # same per-worker limit as bert_run.py


def npu_busy():
    # Same rule as bert_run.npu_busy
    p = subprocess.run(['furiosa-smi', 'ps'], capture_output=True, text=True, timeout=30)
    busy = p.returncode != 0 or any(l.strip().startswith('|') and l.strip().split('|')[1].strip().isdigit()
                                    for l in p.stdout.splitlines())
    return busy, p.stdout+p.stderr


def child(mode, run_dir):
    sys.path.insert(0, str(HERE))
    rec = dict(mode=mode, batch=BATCH, status='running')
    try:
        import torch
        import furiosa.torch as ft
        from transformers import AutoModelForSequenceClassification
        from bert_model import MODEL, SENTENCES, SEQ_LEN, load_model, encode
        cache = Path(run_dir)/'cache'/mode
        if cache.exists(): raise RuntimeError('Never reuse an existing cache dir')
        rec['cache_dir'] = str(cache)
        sents = SENTENCES[:BATCH]

        # 1) CPU FP32 reference, original mask path (before load_model() patches it)
        ref_model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
        tok, model = load_model()            # patches create_bidirectional_mask; BF16 eval
        x2d = tok(sents, padding='max_length', max_length=SEQ_LEN, truncation=True, return_tensors='pt')
        with torch.inference_mode():
            ref = ref_model(**x2d).logits.float()
        del ref_model

        # 2) NPU run, same settings as bert_worker.py
        x = encode(tok, sents)
        rec['input_dtypes'] = {k: str(x[k].dtype) for k in ['input_ids', 'token_type_ids', 'attention_mask']}
        dev = torch.device(DEVICE)
        with ft.config.cpu_fallback(False), ft.config.cache_context(str(cache), summary=False), torch.inference_mode():
            rec['cpu_fallback'] = ft.config.get_cpu_fallback()
            rec['device_configuration'] = str(ft.get_device_configuration())
            model = model.to(dev)
            inp = {k: x[k].to(dev) for k in ['input_ids', 'token_type_ids', 'attention_mask']}
            runner = model if mode == 'eager' else torch.compile(model, backend=ft.backend)
            y = runner(**inp).logits
            out = y.detach().to('cpu').float()
            ft.current_stream(DEVICE).synchronize()
        rec.update(out_shape=list(y.shape), out_dtype=str(y.dtype), finite=bool(torch.isfinite(out).all()),
                   max_abs_diff_vs_cpu_fp32=float((out-ref).abs().max()),
                   label_match=f'{int((out.argmax(1) == ref.argmax(1)).sum())}/{BATCH}',
                   npu_logits=out.tolist(), cpu_fp32_logits=ref.tolist(), status='ok')
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        msg = str(e).strip().splitlines()
        rec.update(status='fail', error_type=type(e).__name__, error_head=msg[0][:300] if msg else '')
    with open(Path(run_dir)/'results.jsonl', 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False)+'\n')


def parent(modes):
    run_dir = HERE/'runs'/time.strftime('b4_check_%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True)
    env = dict(os.environ)
    for k in ['TORCH_LOGS', 'FURIOSA_TORCH_LOG', 'FURIOSA_TORCH_PROFILE']: env.pop(k, None)   # bert_run.py:112
    env['HF_HUB_OFFLINE'] = '1'                                                              # bert_run.py:113
    for mode in modes:
        busy, smi = npu_busy()
        if busy:
            print(f'NPU in use (or furiosa-smi failed); stopped before {mode}:\n{smi}'); break
        print(f'[{mode}] B{BATCH} ... ', end='', flush=True); t0 = time.perf_counter()
        with open(run_dir/f'{mode}.log', 'w') as lf:
            try:
                subprocess.run([sys.executable, '-u', __file__, '--child', mode, '--run-dir', str(run_dir)],
                               stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=TIMEOUT_S, start_new_session=True)
            except subprocess.TimeoutExpired:
                with open(run_dir/'results.jsonl', 'a') as f:
                    f.write(json.dumps(dict(mode=mode, status='timeout', timeout_s=TIMEOUT_S))+'\n')
        print(f'{time.perf_counter()-t0:.0f}s (wall, includes load/compile)')
    rp = run_dir/'results.jsonl'
    print(f'\nresults: {rp}')
    for line in (rp.read_text().splitlines() if rp.exists() else []):
        r = json.loads(line)
        if r['status'] == 'ok':
            print(f"[{r['mode']:7s}] OK   shape={r['out_shape']} {r['out_dtype']} ids={r['input_dtypes']['input_ids']} "
                  f"max|diff| vs CPU FP32={r['max_abs_diff_vs_cpu_fp32']:.4f} labels={r['label_match']}")
        elif r['status'] == 'fail':
            print(f"[{r['mode']:7s}] FAIL {r['error_type']}: {r['error_head'][:100]}  (log: {run_dir/(r['mode']+'.log')})")
        else:
            print(f"[{r['mode']:7s}] TIMEOUT after {r['timeout_s']}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='eager,compile')
    ap.add_argument('--child', help=argparse.SUPPRESS); ap.add_argument('--run-dir', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child: child(a.child, a.run_dir)
    else: parent([m for m in a.modes.split(',') if m in ('eager', 'compile')])

#!/usr/bin/env python3
"""Count aten ops dispatched by ONE BERT forward on CPU (no NPU), per batch.

Purpose: check whether eager NPU time (~18.7 ms flat for B1..B8, run 20261110T140237Z-61d998)
is explained by a fixed cost per dispatched op.

Model/input setup is the measurement code's (bert_model.py): load_model() = mask-function
passthrough patch + BF16 eval + attention implementation from HF config (sdpa); encode() =
step4 sentences 1..B padded to 128, int32 ids, CPU 4D additive BF16 mask. Forward runs under
torch.inference_mode() as in bert_worker.py:47. Under inference_mode the autograd keys are
skipped, so the TorchDispatchMode sees ops at the same granularity the furiosa PrivateUse1
kernels receive them (e.g. aten.linear arrives whole, not as t+addmm).

Each counted op is classified with the installed furiosa-torch registration (read only; no NPU
tensor is created and the script checks that no NPU device node is opened):
  view    - NATIVE_IMPLS / view_ops (furiosa/torch/_db/aten_config.py:12-74,
            _custom_ops/view_ops.py:121-127): metadata only, no NPU program
  npu     - has a PrivateUse1 kernel; in eager each call goes through run_aten_op and runs one
            compiled program (furiosa/torch/_backend/eager.py:16-39)
  no_pu1  - no PrivateUse1 kernel; torch's composite kernel runs and re-dispatches (column 'route')
It also replays the per-forward sequence of eager cache keys (same key as
furiosa/torch/_backend/_utils.py:7-26: op + shape/stride/dtype + non-tensor args) through an LRU
of the default load-cache size (32, _backend/aten.py:126) to estimate steady-state misses.

Per-op time = eager repeat mean (150-call pooled, from the run's break_even_summary.json) / count.

Usage: python bert_op_count.py [--run-dir runs/20261110T140237Z-61d998] [--batches 1,2,4,8,16]
Writes runs/op_count_<time>/op_count.json (new directory; the measurement run is only read).
"""
import argparse
from collections import Counter, OrderedDict
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LOAD_CACHE_DEFAULT = 32   # furiosa/torch/_backend/aten.py:126, unless set_eager_cache_size() is called


def npu_fds():
    """Open file descriptors pointing at device nodes other than the usual ones (should be empty)."""
    out = []
    for fd in os.listdir('/proc/self/fd'):
        try:
            target = os.readlink(f'/proc/self/fd/{fd}')
        except OSError:
            continue
        if target.startswith('/dev/') and not target.startswith(('/dev/null', '/dev/pts', '/dev/urandom', '/dev/random', '/dev/tty', '/dev/shm')):
            out.append(target)
    return out


def furiosa_classifier():
    import torch
    import furiosa.torch  # noqa: F401  registers PrivateUse1 kernels (furiosa/torch/_register.py:63-179)
    from furiosa.torch._db.aten_config import NATIVE_IMPLS
    from furiosa.torch._custom_ops import view_ops
    view = {str(op) for op in NATIVE_IMPLS} | {f'aten.{n}' if '.' in n else f'aten.{n}.default' for n, _ in view_ops.ops()}

    def qual(op_name):  # 'aten.addmm.default' -> 'aten::addmm'
        _, name, overload = op_name.split('.')
        return f'aten::{name}' + ('' if overload == 'default' else f'.{overload}')

    def has(op_name, key):
        return torch._C._dispatch_has_kernel_for_dispatch_key(qual(op_name), key)

    def classify(op_name):
        if op_name in view: return 'view', 'furiosa native view'
        if has(op_name, 'PrivateUse1'): return 'npu', 'run_aten_op (1 compiled program)'
        for key in ['CompositeImplicitAutograd', 'CompositeExplicitAutograd']:
            if has(op_name, key): return 'no_pu1', key
        return 'no_pu1', 'no kernel'
    return classify


def cache_key(func, args, kwargs):
    import torch
    from torch.utils import _pytree as pytree
    flat, spec = pytree.tree_flatten((args, kwargs))
    return (str(func), str(spec)) + tuple((tuple(t.shape), t.stride(), str(t.dtype)) if isinstance(t, torch.Tensor) else repr(t) for t in flat)


def lru_misses(seq, size, passes=3):
    """Misses in the last pass of `passes` identical forwards (steady state)."""
    cache = OrderedDict(); miss = 0
    for p in range(passes):
        miss = 0
        for k in seq:
            if k in cache: cache.move_to_end(k)
            else:
                miss += 1; cache[k] = None
                if len(cache) > size: cache.popitem(last=False)
    return miss


def count_one(model, x):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    seq = []

    class Dispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seq.append((str(func), cache_key(func, args, kwargs or {})))
            return func(*args, **(kwargs or {}))

    with torch.inference_mode(), Dispatch():
        out = model(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
    return seq, out.logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=HERE/'runs'/'20261110T140237Z-61d998')
    ap.add_argument('--batches', default='1,2,4,8,16')
    a = ap.parse_args()
    batches = [int(b) for b in a.batches.split(',')]

    import torch
    from bert_model import load_model, encode, SENTENCES, SEQ_LEN, ID_DTYPE
    classify = furiosa_classifier()
    fds = npu_fds()

    summary = json.loads((a.run_dir/'break_even_summary.json').read_text())
    eager_ms = {b['batch']: b['pooled_repeat_by_mode']['eager']['mean_ms'] for b in summary['batches']}

    tok, model = load_model()
    attn = model.config._attn_implementation
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    rows, kinds = [], {}
    for B in batches:
        x = encode(tok, SENTENCES[:B])
        assert str(x['input_ids'].dtype) == ID_DTYPE and x['attention_mask'].dtype == torch.bfloat16
        assert list(x['attention_mask'].shape) == [B, 1, SEQ_LEN, SEQ_LEN]
        seq, logits = count_one(model, x)
        assert list(logits.shape) == [B, 2]
        ops = Counter(n for n, _ in seq)
        cls = Counter()
        for n, c in ops.items(): cls[classify(n)[0]] += c
        npu_keys = [k for n, k in seq if classify(n)[0] == 'npu']
        e = eager_ms.get(B)
        kinds[B] = dict(ops.most_common())
        rows.append(dict(batch=B, eager_mean_ms=e, total=sum(ops.values()), by_class=dict(cls),
                         npu_distinct_cache_keys=len(set(npu_keys)),
                         npu_load_cache_misses_per_forward=lru_misses(npu_keys, LOAD_CACHE_DEFAULT),
                         us_per_op=e*1000/sum(ops.values()) if e else None,
                         us_per_npu_op=e*1000/cls['npu'] if e and cls['npu'] else None))

    f = lambda v, d=3: '—' if v is None else f'{v:.{d}f}'
    print(f'torch {torch.__version__}, attn_implementation={attn}, dtype=bf16, ids={ID_DTYPE}, seq_len={SEQ_LEN}, inference_mode')
    print(f'device nodes opened by this process: {fds or "none"}')
    print('\n| batch | eager mean ms | aten ops (total) | view | npu | no_pu1 | µs/op (total) | µs/op (npu only) '
          f'| distinct npu cache keys | load-cache misses/forward (LRU {LOAD_CACHE_DEFAULT}) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        c = r['by_class']
        print(f"| {r['batch']} | {f(r['eager_mean_ms'])} | {r['total']} | {c.get('view', 0)} | {c.get('npu', 0)} | {c.get('no_pu1', 0)} "
              f"| {f(r['us_per_op'], 1)} | {f(r['us_per_npu_op'], 1)} | {r['npu_distinct_cache_keys']} | {r['npu_load_cache_misses_per_forward']} |")
    names = sorted({n for k in kinds.values() for n in k}, key=lambda n: -kinds[batches[0]].get(n, 0))
    print('\naten ops by kind (count per forward)')
    print('| op | class | route | '+' | '.join(f'B{b}' for b in batches)+' |')
    print('|---|---|---|'+'---|'*len(batches))
    for n in names:
        c, route = classify(n)
        print(f'| {n} | {c} | {route} | '+' | '.join(str(kinds[b].get(n, 0)) for b in batches)+' |')

    out_dir = HERE/'runs'/time.strftime('op_count_%Y%m%d_%H%M%S'); out_dir.mkdir(parents=True)
    (out_dir/'op_count.json').write_text(json.dumps(dict(
        source_run=str(a.run_dir), torch_version=torch.__version__, attn_implementation=attn, id_dtype=ID_DTYPE,
        device_nodes_opened=fds, load_cache_size_assumed=LOAD_CACHE_DEFAULT, rows=rows,
        kinds={str(k): v for k, v in kinds.items()}, classes={n: classify(n) for n in names}), indent=1, ensure_ascii=False))
    print(f'\nwrote {out_dir/"op_count.json"}')


if __name__ == '__main__':
    main()

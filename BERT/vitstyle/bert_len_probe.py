#!/usr/bin/env python3
"""길이 48·64 재시도 프로브: batch 1, eager/compile, id dtype int64 vs int32

지난 기록(BERT/embedding_length_test.py 커밋 ecb22f9, step3_eager_len64_fail.log)에서
길이 48·64는 임베딩 조회(aten.index)가 NPU 컴파일에 실패했다. 당시에는 int64 인덱스였고,
그 뒤 int32 전환으로 batch>1(ids)과 eager+fusion2(position_ids)의 같은 실패가 풀렸으므로
길이 48·64도 int32로 재시도한다.

케이스: 길이 {48, 64} x dtype {int64, int32} x 대상 {embedding, full} x 모드 {eager, compile}.
input_ids/token_type_ids/position_ids 세 개를 모두 같은 dtype으로 명시해 넘긴다.
PE fusion은 호출하지 않음(SDK 기본값 = PE 1개; 지난 실패와 같은 조건).

bert_b4_probe.py와 같은 방식: 케이스마다 별도 프로세스 + 새 전용 캐시,
결과는 runs/len_probe_<시각>/ 아래 케이스별 로그와 results.jsonl.

사용법:
  python bert_len_probe.py               # 전체 16케이스
  python bert_len_probe.py --only 1,2,3  # 일부 케이스만
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODEL = "textattack/bert-base-uncased-SST-2"
DEVICE = "furiosa:0"
TIMEOUT_S = 600  # 케이스당 제한(compile 포함). 넘으면 SIGKILL
BATCH = 1
SENTENCE = "a stirring , funny and finally transporting re-imagining of a classic story"  # bert_model.SENTENCES[0]


def npu_context(cache_dir):
    """bert_worker.py:57과 같은 컨텍스트: cpu_fallback(False) + 전용 cache_context + inference_mode."""
    import contextlib
    import torch
    import furiosa.torch as ft
    stack = contextlib.ExitStack()
    stack.enter_context(ft.config.cpu_fallback(False))
    stack.enter_context(ft.config.cache_context(str(cache_dir), summary=False))
    stack.enter_context(torch.inference_mode())
    return stack


EXTRA_ENV = {"HF_HUB_OFFLINE": "1"}
UNSET_ENV = ["TORCH_LOGS", "FURIOSA_TORCH_LOG", "FURIOSA_TORCH_PROFILE"]

# (id, 길이, dtype, 대상, 모드). 전부 batch 1.
CASES = [(i + 1, L, dt, target, mode)
         for i, (L, dt, target, mode) in enumerate(
             (L, dt, target, mode)
             for L in [48, 64]
             for target in ["embedding", "full"]
             for dt in ["int64", "int32"]
             for mode in ["eager", "compile"])]
CASE_BY_ID = {c[0]: c for c in CASES}


def additive_mask(mask2d, L, dtype):
    """bert_model.additive_mask와 같되 길이가 L: [B,1,L,L] additive BF16 마스크."""
    import torch
    neg = torch.finfo(dtype).min
    m = (1.0 - mask2d[:, None, None, :].to(torch.float32)) * neg
    return m.expand(-1, 1, L, -1).contiguous().to(dtype)


def run_case(cid, cache_dir):
    import torch
    import furiosa.torch as ft

    _, L, dt, target, mode = CASE_BY_ID[cid]
    dtype = getattr(torch, dt)
    torch.manual_seed(0)
    if Path(cache_dir).exists():
        raise RuntimeError("Never reuse an existing cache dir")  # bert_worker.py:54와 같음

    from bert_model import patch_mask
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    patch_mask()  # 측정과 같은 passthrough 마스크 (full 모델에서만 의미 있음)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(torch.bfloat16).eval()

    x = tok([SENTENCE] * BATCH, padding="max_length", max_length=L, truncation=True, return_tensors="pt")
    ids = x["input_ids"].to(dtype)
    tt = x["token_type_ids"].to(dtype)
    pos = torch.arange(L, dtype=dtype).unsqueeze(0)          # [1,L], 세 id 모두 같은 dtype
    mask = additive_mask(x["attention_mask"], L, torch.bfloat16)

    if target == "embedding":
        mod = model.bert.embeddings
        call = lambda m, mv: m(input_ids=mv(ids), token_type_ids=mv(tt), position_ids=mv(pos))
        shape_of = lambda out: list(out.shape)
    else:
        mod = model
        call = lambda m, mv: m(input_ids=mv(ids), token_type_ids=mv(tt), attention_mask=mv(mask), position_ids=mv(pos))
        shape_of = lambda out: list(out.logits.shape)

    with torch.inference_mode():
        ref = call(mod, lambda t: t)  # CPU 기준값
    ref = ref if target == "embedding" else ref.logits

    dev = torch.device(DEVICE)
    with npu_context(cache_dir):
        m = mod.to(dev)
        if mode == "compile":
            m = torch.compile(m, backend=ft.backend)  # bert_worker.py:86과 같음: backend만 지정
        t0 = time.perf_counter()
        out = call(m, lambda t: t.to(dev))
        out = (out if target == "embedding" else out.logits).cpu()
        elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": round(elapsed, 1),
        "out_shape": list(out.shape),
        "max_abs_diff_vs_cpu": float((out.float() - ref.float()).abs().max()),
    }


def child(cid, result_path):
    rec = {"case": cid}
    _, L, dt, target, mode = CASE_BY_ID[cid]
    rec.update(seq_len=L, batch=BATCH, dtype=dt, target=target, mode=mode)
    try:
        cache_dir = Path(result_path).parent / "cache" / f"case{cid:02d}"
        rec.update(cache_dir=str(cache_dir))
        rec.update(status="ok", **run_case(cid, cache_dir))
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        msg = str(e).strip().splitlines()
        rec.update(
            status="fail",
            error_type=type(e).__name__,
            error_head=msg[0][:300] if msg else "",
            aten_ops=sorted(set(re.findall(r"aten\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?", tb))),
        )
    with open(result_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parent(ids):
    run_dir = Path(__file__).resolve().parent / "runs" / time.strftime("len_probe_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    result_path = run_dir / "results.jsonl"
    env = dict(os.environ)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update(EXTRA_ENV)

    for cid in ids:
        _, L, dt, target, mode = CASE_BY_ID[cid]
        print(f"[{cid:2d}] L{L} {target:9s} {dt} {mode:7s} ... ", end="", flush=True)
        log = run_dir / f"case{cid:02d}.log"
        t0 = time.perf_counter()
        try:
            with open(log, "w") as lf:
                subprocess.run(
                    [sys.executable, __file__, "--case", str(cid), "--result", str(result_path)],
                    stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=TIMEOUT_S,
                )
        except subprocess.TimeoutExpired:  # run()이 자식을 SIGKILL함
            with open(result_path, "a") as f:
                f.write(json.dumps({"case": cid, "status": "timeout", "timeout_s": TIMEOUT_S}) + "\n")
        print(f"{time.perf_counter() - t0:.0f}s")

    print(f"\n결과: {result_path}")
    for line in result_path.read_text().splitlines():
        r = json.loads(line)
        _, L, dt, target, mode = CASE_BY_ID[r["case"]]
        head = f"[{r['case']:2d}] L{L} {target:9s} {dt} {mode:7s}"
        if r["status"] == "ok":
            print(f"{head}  OK    diff={r['max_abs_diff_vs_cpu']:.4f}")
        elif r["status"] == "fail":
            print(f"{head}  FAIL  {r['error_type']}: {r['error_head'][:80]}")
            if r.get("aten_ops"):
                print(f"{'':34s}ops: {', '.join(r['aten_ops'])}")
        else:
            print(f"{head}  TIMEOUT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="쉼표로 구분한 케이스 번호")
    ap.add_argument("--case", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.case is not None:
        child(a.case, a.result)
    else:
        parent([int(x) for x in a.only.split(",")] if a.only else [c[0] for c in CASES])

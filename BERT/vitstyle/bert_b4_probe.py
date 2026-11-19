#!/usr/bin/env python3
"""BERT batch 4 임베딩 실패 원인 확인 프로브

케이스마다 별도 프로세스로 실행한다 (한 케이스가 실패해도 다음 케이스에 영향 없음).
결과는 runs/b4_probe_<시각>/ 아래에 케이스별 로그와 results.jsonl로 남는다.

사용법:
  python bert_b4_probe.py              # 전체 케이스
  python bert_b4_probe.py --only 2,3,4 # 일부 케이스만
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
SEQ = 128
DEVICE = "furiosa:0"
TIMEOUT_S = 300  # 케이스당 제한. 넘으면 SIGKILL


# ---- 기존 코드(bert-vitstyle-timing 브랜치)와 맞춰야 하는 부분 -------------
def compile_module(m):
    """bert_worker.py:74와 같음: backend만 지정하고 다른 options는 주지 않음.
    cpu_fallback=False는 compile 인자가 아니라 컨텍스트로 걸리므로 run_case의
    npu_context()에서 eager/compile 모두에 적용함 (bert_worker.py:47)."""
    import torch
    import furiosa.torch as ft
    return torch.compile(m, backend=ft.backend)


def npu_context(cache_dir):
    """bert_worker.py:47과 같은 컨텍스트: cpu_fallback(False) + 전용 cache_context + inference_mode.
    SDK에는 캐시 경로용 환경 변수가 없고, 기존 코드도 환경 변수 대신 cache_context를 씀."""
    import contextlib
    import torch
    import furiosa.torch as ft
    stack = contextlib.ExitStack()
    stack.enter_context(ft.config.cpu_fallback(False))
    stack.enter_context(ft.config.cache_context(str(cache_dir), summary=False))
    stack.enter_context(torch.inference_mode())
    return stack


# bert_run.py:112-113과 같음: HF 오프라인 로딩. 캐시 경로는 환경 변수가 아니라
# child()에서 {run_dir}/cache/caseNN 을 cache_context로 지정함 (케이스마다 새 경로).
EXTRA_ENV = {"HF_HUB_OFFLINE": "1"}
UNSET_ENV = ["TORCH_LOGS", "FURIOSA_TORCH_LOG", "FURIOSA_TORCH_PROFILE"]
# ---------------------------------------------------------------------------

# (id, 대상, batch, index dtype, 모드, index 모양, 목적)
CASES = [
    (1,  "word",       1, "int64", "eager",   "BxL",    "대조군: 성공해야 정상"),
    (2,  "word",       4, "int64", "eager",   "BxL",    "eager 실패 재현"),
    (3,  "word",       4, "int32", "eager",   "BxL",    "int32로 해결되는지"),
    (4,  "word",       4, "int64", "eager",   "1x(BL)", "[1,512]로 펴면 되는지"),
    (5,  "word",       4, "int64", "eager",   "(BL)",   "1차원 [512]로 펴면 되는지"),
    (6,  "token_type", 4, "int64", "eager",   "BxL",    "token_type도 같은 실패인지"),
    (7,  "token_type", 4, "int32", "eager",   "BxL",    "token_type int32"),
    (8,  "full",       4, "int64", "compile", "BxL",    "compile 실패 재현, 남는 연산 2개 확인"),
    (9,  "full",       4, "int32", "compile", "BxL",    "compile + int32"),
    (10, "word",       4, "int64", "compile", "BxL",    "word만 compile"),
    (11, "token_type", 4, "int64", "compile", "BxL",    "token_type만 compile"),
]
CASE_BY_ID = {c[0]: c for c in CASES}


def run_case(cid, cache_dir):
    import torch
    from transformers import AutoModelForSequenceClassification

    _, target, B, dt, mode, shape, _ = CASE_BY_ID[cid]
    torch.manual_seed(0)
    if Path(cache_dir).exists():
        raise RuntimeError("Never reuse an existing cache dir")  # bert_worker.py:46과 같음
    # bert_model.py:73과 같은 로딩 순서
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(torch.bfloat16).eval()
    emb = model.bert.embeddings
    dtype = getattr(torch, dt)

    ids = torch.randint(1000, model.config.vocab_size, (B, SEQ), dtype=dtype)
    tt = torch.zeros(B, SEQ, dtype=dtype)

    if target == "word":
        mod, inp = emb.word_embeddings, ids
    elif target == "token_type":
        mod, inp = emb.token_type_embeddings, tt
    else:  # full: word + token_type + position + LayerNorm
        mod, inp = emb, None
    if inp is not None:
        if shape == "1x(BL)":
            inp = inp.reshape(1, -1)
        elif shape == "(BL)":
            inp = inp.reshape(-1)

    def call(m, dev=None):
        mv = (lambda t: t.to(dev)) if dev is not None else (lambda t: t)
        if target == "full":
            return m(input_ids=mv(ids), token_type_ids=mv(tt))
        return m(mv(inp))

    with torch.inference_mode():
        ref = call(mod)  # CPU 기준값

    dev = torch.device(DEVICE)
    with npu_context(cache_dir):
        m = mod.to(dev)
        if mode == "compile":
            m = compile_module(m)

        t0 = time.perf_counter()
        out = call(m, dev).cpu()
        elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": round(elapsed, 1),
        "out_shape": list(out.shape),
        "max_abs_diff_vs_cpu": float((out.float() - ref.float()).abs().max()),
    }


def child(cid, result_path):
    rec = {"case": cid}
    _, target, B, dt, mode, shape, purpose = CASE_BY_ID[cid]
    rec.update(target=target, batch=B, dtype=dt, mode=mode, shape=shape, purpose=purpose)
    try:
        cache_dir = Path(result_path).parent / "cache" / f"case{cid:02d}"   # {run_dir}/cache/caseNN
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
    # bert_run.py:54와 같은 위치(BERT/vitstyle/runs). 실행한 폴더와 상관없이 고정함.
    run_dir = Path(__file__).resolve().parent / "runs" / time.strftime("b4_probe_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    result_path = run_dir / "results.jsonl"
    env = dict(os.environ)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update({k: v.format(run_dir=run_dir.resolve()) for k, v in EXTRA_ENV.items()})

    for cid in ids:
        c = CASE_BY_ID[cid]
        print(f"[{cid:2d}] {c[1]:10s} B{c[2]} {c[3]} {c[4]:7s} {c[5]:7s} ... ", end="", flush=True)
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
        c = CASE_BY_ID[r["case"]]
        head = f"[{r['case']:2d}] {c[1]:10s} B{c[2]} {c[3]} {c[4]:7s} {c[5]:7s}"
        if r["status"] == "ok":
            print(f"{head}  OK    diff={r['max_abs_diff_vs_cpu']:.4f}")
        elif r["status"] == "fail":
            print(f"{head}  FAIL  {r['error_type']}: {r['error_head'][:80]}")
            if r.get("aten_ops"):
                print(f"{'':45s}ops: {', '.join(r['aten_ops'])}")
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

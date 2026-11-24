#!/usr/bin/env python3
"""position_ids dtype 프로브: fusion 2 + eager, 위치 임베딩만

run 20261124T065313Z-748779 (eager, fusion 2)에서 batch 1/16/64 전부
position_embeddings(position_ids)의 aten.index.Tensor가 eager 연산별 컴파일에서
UnsupportedOpError로 실패함 (global-compiler lib.rs:60, count_valid_operators_if (1) != 0).
input_ids/token_type_ids는 bert_model.encode에서 int32로 바꾸지만 position_ids는
HF BertEmbeddings의 버퍼(arange(512).expand(1,-1) → int64)를 모델이 직접 슬라이스해
쓰므로 int32 전환에 포함되지 않았음. 이 프로브는 위치 임베딩 하나만 떼어
int64 vs int32, batch 1/16을 fusion 2 + eager로 확인한다.

bert_b4_probe.py와 같은 방식: 케이스마다 별도 프로세스(한 케이스가 실패해도 다음
케이스에 영향 없음), 케이스마다 새 전용 캐시. set_fusion은 bert_worker.py:42와
같이 프로세스마다 장치/모델 사용 전에 한 번만 호출하고 effective 값을 기록한다.
결과는 runs/posid_probe_<시각>/ 아래 케이스별 로그와 results.jsonl로 남는다.

사용법:
  python bert_posid_probe.py              # 전체 케이스, fusion 2
  python bert_posid_probe.py --only 1,2   # 일부 케이스만
  python bert_posid_probe.py --fusion 4   # 다른 fusion 값과 비교할 때
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
DEFAULT_FUSION = 2


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


# bert_run.py:135-136과 같음: HF 오프라인 로딩, 로그/프로파일 환경 변수 제거.
EXTRA_ENV = {"HF_HUB_OFFLINE": "1"}
UNSET_ENV = ["TORCH_LOGS", "FURIOSA_TORCH_LOG", "FURIOSA_TORCH_PROFILE"]

# (id, batch, index dtype, 목적). 모드는 전부 eager, 대상은 위치 임베딩만.
# HF는 batch와 무관하게 [1,L] 버퍼 슬라이스를 쓰므로 B1이 실제 실패 경로의 재현이고,
# B16은 position_ids를 배치별로 넘길 때의 [B,L] 모양을 확인한다.
CASES = [
    (1,  1, "int64", "실패 재현: HF 버퍼와 같은 [1,128] int64"),
    (2,  1, "int32", "[1,128] int32로 해결되는지"),
    (3, 16, "int64", "[16,128] int64"),
    (4, 16, "int32", "[16,128] int32"),
]
CASE_BY_ID = {c[0]: c for c in CASES}


def run_case(cid, cache_dir, fusion_n):
    import torch
    import furiosa.torch as ft

    # bert_worker.py:42와 같음: set_fusion은 장치/스트림/모델 사용 전에 프로세스당 한 번만.
    fusion = ft.set_fusion(fusion_n)
    fusion_rec = {k: getattr(fusion, k) for k in ["num_fusion", "num_pe_per_cluster", "num_chips", "num_clusters_per_chip"]}
    if fusion.num_fusion != fusion_n:
        raise RuntimeError(f"set_fusion returned num_fusion={fusion.num_fusion}, requested {fusion_n}")

    from transformers import AutoModelForSequenceClassification

    _, B, dt, _ = CASE_BY_ID[cid]
    torch.manual_seed(0)
    if Path(cache_dir).exists():
        raise RuntimeError("Never reuse an existing cache dir")  # bert_worker.py:54와 같음
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(torch.bfloat16).eval()
    pe = model.bert.embeddings.position_embeddings
    buffer_dtype = str(model.bert.embeddings.position_ids.dtype)  # 모델이 실제로 쓰는 버퍼의 dtype 기록

    ids = torch.arange(SEQ, dtype=getattr(torch, dt)).unsqueeze(0)  # [1,L]: HF 버퍼 슬라이스와 같은 값/모양
    if B > 1:
        ids = ids.repeat(B, 1)  # [B,L]

    with torch.inference_mode():
        ref = pe(ids)  # CPU 기준값

    dev = torch.device(DEVICE)
    with npu_context(cache_dir):
        m = pe.to(dev)
        t0 = time.perf_counter()
        out = m(ids.to(dev)).cpu()
        elapsed = time.perf_counter() - t0

    return {
        "fusion": fusion_rec,
        "model_buffer_dtype": buffer_dtype,
        "elapsed_s": round(elapsed, 1),
        "out_shape": list(out.shape),
        "max_abs_diff_vs_cpu": float((out.float() - ref.float()).abs().max()),
    }


def child(cid, result_path, fusion_n):
    rec = {"case": cid, "fusion_requested": fusion_n}
    _, B, dt, purpose = CASE_BY_ID[cid]
    rec.update(target="position", batch=B, dtype=dt, mode="eager", purpose=purpose)
    try:
        cache_dir = Path(result_path).parent / "cache" / f"case{cid:02d}_f{fusion_n}"  # 캐시 경로에 fusion 태그 (bert_run.py:41)
        rec.update(cache_dir=str(cache_dir))
        rec.update(status="ok", **run_case(cid, cache_dir, fusion_n))
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


def parent(ids, fusion_n):
    run_dir = Path(__file__).resolve().parent / "runs" / time.strftime("posid_probe_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    result_path = run_dir / "results.jsonl"
    env = dict(os.environ)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update(EXTRA_ENV)

    for cid in ids:
        c = CASE_BY_ID[cid]
        print(f"[{cid:2d}] position B{c[1]:<2d} {c[2]} eager f{fusion_n} ... ", end="", flush=True)
        log = run_dir / f"case{cid:02d}.log"
        t0 = time.perf_counter()
        try:
            with open(log, "w") as lf:
                subprocess.run(
                    [sys.executable, __file__, "--case", str(cid), "--result", str(result_path),
                     "--fusion", str(fusion_n)],
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
        head = f"[{r['case']:2d}] position B{c[1]:<2d} {c[2]} eager f{fusion_n}"
        if r["status"] == "ok":
            print(f"{head}  OK    diff={r['max_abs_diff_vs_cpu']:.4f}  buffer={r['model_buffer_dtype']}")
        elif r["status"] == "fail":
            print(f"{head}  FAIL  {r['error_type']}: {r['error_head'][:80]}")
            if r.get("aten_ops"):
                print(f"{'':38s}ops: {', '.join(r['aten_ops'])}")
        else:
            print(f"{head}  TIMEOUT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="쉼표로 구분한 케이스 번호")
    ap.add_argument("--fusion", type=int, default=DEFAULT_FUSION, help="set_fusion 값 (기본 2)")
    ap.add_argument("--case", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.case is not None:
        child(a.case, a.result, a.fusion)
    else:
        parent([int(x) for x in a.only.split(",")] if a.only else [c[0] for c in CASES], a.fusion)

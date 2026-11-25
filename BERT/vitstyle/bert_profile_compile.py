#!/usr/bin/env python3
"""compile 실행 프로파일링: batch 16·64, PE 1개(SDK 기본값), 연산별 시간

furiosa-torch 2026.4.0의 공식 프로파일러 사용 (furiosa/torch/profiler/__init__.py,
_C/profiler/__init__.pyi):
  - `furiosa.torch.profiler.Profiler(path)` 컨텍스트에 들어가면 수집을 시작하고,
    나올 때 Chrome trace JSON을 path에 쓴다 (chrome://tracing, https://ui.perfetto.dev에서 열람).
  - NPU 상세 수준은 환경 변수 FURIOSA_TORCH_PROFILE이 정한다:
      trace = 연산(operator)·DMA 수준(오버헤드 최대), info(기본) = task 수준, warn = 끔.
    로깅 초기화 시점에 고정되므로 **프로파일 세션 시작 전에** 설정해야 한다.
    이 스크립트는 부모가 자식 프로세스 환경에 trace를 넣어 준다
    (bert_run.py:135는 측정에서 이 변수를 지우지만, 여기서는 프로파일링이 목적).
  - `RecordFunction(이름)`으로 trace에 구간 라벨을 넣을 수 있다.

측정과 같은 설정(bert_worker.py): BF16, 길이 128, int32 id, CPU 4D 마스크,
cpu_fallback(False), 새 전용 캐시, torch.compile(backend=ft.backend), inference_mode.
set_fusion은 호출하지 않음(PE 1개). 첫 호출 + 워밍업은 프로파일 밖, 반복 호출만
Profiler 안에서 실행한다. 프로파일 오버헤드가 있으므로 여기서 잰 시간은 성능
수치로 쓰지 않는다(연산별 비중 파악용).

batch마다: runs/profile_<시각>/b<batch>.trace.json (원본 trace),
b<batch>.ops.txt (이름별 합계 시간 상위 집계), results.jsonl.

사용법:
  python bert_profile_compile.py                  # batch 16, 64
  python bert_profile_compile.py --batches 16     # 일부만
  python bert_profile_compile.py --posid-int32    # 측정 옵션과 맞출 때
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

DEVICE = "furiosa:0"
TIMEOUT_S = 1800   # batch 64 compile 첫 호출이 길어서 worker_timeout_s와 같게
WARMUP = 10        # 측정(config warmup)과 같음; 프로파일 밖
REPEATS = 10       # 프로파일 안에서 재는 반복 호출 수 (trace 크기 때문에 50 대신 10)
PROFILE_ENV = "FURIOSA_TORCH_PROFILE"
PROFILE_LEVEL = "trace"   # 연산·DMA 수준. task 수준만 보려면 info

EXTRA_ENV = {"HF_HUB_OFFLINE": "1"}
UNSET_ENV = ["TORCH_LOGS", "FURIOSA_TORCH_LOG"]   # PROFILE_ENV는 지우지 않고 trace로 설정


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


def summarize_trace(trace_path, out_path, top=40):
    """Chrome trace JSON에서 complete 이벤트('ph'=='X')의 dur(us)를 이름별로 합산."""
    raw = json.loads(Path(trace_path).read_text())
    events = raw["traceEvents"] if isinstance(raw, dict) else raw
    total = defaultdict(float); count = defaultdict(int)
    for e in events:
        if e.get("ph") == "X" and e.get("dur") is not None:
            total[e["name"]] += e["dur"]; count[e["name"]] += 1
    rows = sorted(total.items(), key=lambda kv: -kv[1])[:top]
    width = max((len(n) for n, _ in rows), default=10)
    lines = [f"{'name':{width}s}  {'sum_ms':>10s}  {'calls':>6s}  {'mean_us':>10s}"]
    lines += [f"{n:{width}s}  {t/1000:10.3f}  {count[n]:6d}  {t/count[n]:10.1f}" for n, t in rows]
    Path(out_path).write_text("\n".join(lines) + "\n")
    return len(events)


def run_case(B, run_dir, posid_int32):
    import torch
    import furiosa.torch as ft
    if os.environ.get(PROFILE_ENV) != PROFILE_LEVEL:
        raise RuntimeError(f"{PROFILE_ENV}={PROFILE_LEVEL} must be set before the session starts")
    from bert_model import load_model, encode, SENTENCES, sentence_indices

    cache_dir = run_dir / "cache" / f"b{B}"
    if cache_dir.exists():
        raise RuntimeError("Never reuse an existing cache dir")
    tok, model = load_model()
    idx = sentence_indices("performance", B)
    inp_cpu = encode(tok, [SENTENCES[i] for i in idx], posid_int32=posid_int32)
    keys = ["input_ids", "token_type_ids", "attention_mask"] + (["position_ids"] if posid_int32 else [])

    dev = torch.device(DEVICE)
    trace_path = run_dir / f"b{B}.trace.json"
    with npu_context(cache_dir):
        model = model.to(dev)
        inp = {k: inp_cpu[k].to(dev) for k in keys}
        stream = ft.current_stream(DEVICE)
        runner = torch.compile(model, backend=ft.backend)   # bert_worker.py:86과 같음

        # 첫 호출 + 워밍업은 프로파일 밖 (컴파일·캐시 적재가 섞이지 않게)
        for _ in range(1 + WARMUP):
            runner(**inp); stream.synchronize()

        timings = []
        with ft.profiler.Profiler(str(trace_path)):
            for i in range(1, REPEATS + 1):
                with ft.profiler.RecordFunction(f"repeat_{i}"):
                    t0 = time.perf_counter_ns()
                    out = runner(**inp)
                    stream.synchronize()
                    timings.append((time.perf_counter_ns() - t0) / 1e6)
        _ = out.logits.detach().to("cpu").float()   # 출력 모양 확인용

    n_events = summarize_trace(trace_path, run_dir / f"b{B}.ops.txt")
    return dict(trace=str(trace_path), ops_summary=str(run_dir / f"b{B}.ops.txt"),
                trace_events=n_events, profiled_calls=REPEATS,
                profiled_mean_ms=sum(timings) / len(timings), profiled_calls_ms=[round(t, 3) for t in timings])


def child(B, run_dir, posid_int32):
    rec = {"batch": B, "posid_int32": posid_int32, "profile_level": PROFILE_LEVEL,
           "warmup_outside_profiler": 1 + WARMUP}
    try:
        rec.update(status="ok", **run_case(B, Path(run_dir), posid_int32))
    except BaseException as e:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        msg = str(e).strip().splitlines()
        rec.update(status="fail", error_type=type(e).__name__, error_head=msg[0][:300] if msg else "")
    with open(Path(run_dir) / "results.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parent(batches, posid_int32):
    run_dir = Path(__file__).resolve().parent / "runs" / time.strftime("profile_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    env = dict(os.environ)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update(EXTRA_ENV)
    env[PROFILE_ENV] = PROFILE_LEVEL   # 자식 프로세스 시작 전에 고정해야 연산 수준이 켜짐

    for B in batches:
        print(f"[b{B}] compile PE1 {PROFILE_LEVEL} ... ", end="", flush=True)
        log = run_dir / f"b{B}.log"
        t0 = time.perf_counter()
        try:
            with open(log, "w") as lf:
                cmd = [sys.executable, __file__, "--child", str(B), "--run-dir", str(run_dir)]
                if posid_int32:
                    cmd.append("--posid-int32")
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with open(run_dir / "results.jsonl", "a") as f:
                f.write(json.dumps({"batch": B, "status": "timeout", "timeout_s": TIMEOUT_S}) + "\n")
        print(f"{time.perf_counter() - t0:.0f}s")

    print(f"\n결과: {run_dir}/results.jsonl")
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            print(f"b{r['batch']}: 프로파일 반복 평균 {r['profiled_mean_ms']:.3f} ms (오버헤드 포함), "
                  f"trace {r['trace']}, 연산별 합계 {r['ops_summary']}")
        else:
            print(f"b{r['batch']}: {r['status']} {r.get('error_type', '')} {r.get('error_head', '')[:80]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=[16, 64])
    ap.add_argument("--posid-int32", action="store_true", help="측정의 --posid-int32와 같게 position_ids int32 전달")
    ap.add_argument("--child", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child is not None:
        child(a.child, a.run_dir, a.posid_int32)
    else:
        parent(a.batches, a.posid_int32)

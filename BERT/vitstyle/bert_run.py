"""BERT ViT-style comparison orchestrator. Adapted from /root/ViT/vit_pilot/fastpath_compare.py.

Stages (each worker = new process + new private cache):
  performance : 3 trials x {eager,compile} x BATCHES, rotated order (reduces to ViT's for B1/B4)
  outcmp      : 16 step4 sentences (B > 16: reused cyclically to B), {eager,compile} x BATCHES (replaces ViT diagnostic gate)
A failed/timed-out worker is recorded and the next worker starts; only a busy NPU stops the run.
After one failure/timeout, that (mode, batch) is not launched in later trials: status 'skipped',
reason '이전 세트 실패로 건너뜀', skipped_because=<failed case_id>.
  cumulative  : break-even validation, target = ceil(1.2*median N_win)
Usage:
  python bert_run.py run                                  # new run dir, performance stage (B1..B16, 3 trials)
  python bert_run.py run --batches 32,64 --trials 1       # same settings, other batches / trial count
  python bert_run.py run --fusion 2 --batches 1,16,64 --trials 1
                                                          # PE fusion: worker calls furiosa.torch.set_fusion(2) once
  python bert_run.py resume --run-dir runs/ID --stage outcmp|cumulative
  python bert_run.py analyze --run-dir runs/ID
If another process is on the NPU the worker is held and the orchestrator stops (exit 3).
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import Events, now, read_json, write_json, sha256

HERE = Path(__file__).resolve().parent
SOURCES = ['bert_run.py', 'bert_worker.py', 'bert_model.py', 'bert_math.py', 'bert_analyze.py', 'common.py']
WORKER_SOURCES = ['bert_worker.py', 'bert_model.py', 'common.py']  # must not change mid-run
STAGES = ['performance', 'outcmp', 'cumulative']
FAILED = ['error', 'timeout']   # a (mode, batch) with one of these is skipped in later trials
SKIP_REASON = '이전 세트 실패로 건너뜀'
BATCHES = [1, 2, 4, 8, 16]   # default for `run`; length 128, int32 ids, CPU 4D mask; batch B uses step4 sentences 1..B
FUSIONS = [1, 2, 4, 8]       # furiosa.torch.Fusion: 1/2/4 hardware-fused PEs in one cluster, 8 = software fusion (one chip)


class NPUBusy(Exception):
    pass


def command(args):
    # Copied from /root/ViT/vit_pilot/environment.py: command
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as e:
        return dict(command=args, error=repr(e))


def npu_busy(check):
    return check.get('returncode') != 0 or any(
        line.strip().startswith('|') and line.strip().split('|')[1].strip().isdigit()
        for line in check.get('stdout', '').splitlines())


def find_case(m, stage, mode, batch, trial=1):
    return next((w for w in reversed(m['workers']) if (w['stage'], w['mode'], w['batch'], w['trial']) == (stage, mode, batch, trial)), None)


def fresh(a):
    run = HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+os.urandom(3).hex())
    run.mkdir(parents=True, exist_ok=False)
    for name in ['logs', 'cases', 'outputs', 'code']: (run/name).mkdir()
    from bert_math import NUMERIC_RULE
    cfg = dict(run_id=run.name, model='textattack/bert-base-uncased-SST-2', seq_len=128, device=a.device,
               dtype='torch.bfloat16', id_dtype='torch.int32', cpu_fallback=False, mask='create_bidirectional_mask passthrough + CPU 4D additive BF16 mask (step4_compare.py)',
               pe=('SDK default (not set); recorded per worker as effective.device_configuration' if a.fusion is None else
                   f'furiosa.torch.set_fusion({a.fusion}) once per worker before any device use; effective value recorded per worker'),
               fusion=a.fusion,
               cache_policy='fresh_process_private_sdk_cache_initially_absent_system_cold_unverified',
               timer='current_stream.synchronize(); perf_counter_ns; runner(**inputs); current_stream.synchronize()',
               numeric_rule=NUMERIC_RULE, atol=None, rtol=None,
               cumulative_target_rule='ceil(1.2 * median paired N_win); N_win=1 -> at least 20; cap/budget may shorten',
               cumulative_wall_bound_rule='1.5*max(new-trial prepare+A+N*t)/1000 +60 s +N*0.001 s per mode',
               batches=a.batches, worker_timeout_s=1800, total_budget_s=14400, warmup=10, repeats=50, trials=a.trials, max_cumulative_calls=10000,
               hf_hub_offline=True)
    write_json(run/'config.json', cfg)
    workers = []
    def add(stage, mode, b, trial=1, status='not_run'):
        cid = f'{stage}_{mode}_b{b}_t{trial}'
        workers.append(dict(case_id=cid, root_case_id=cid, stage=stage, mode=mode, batch=b, trial=trial,
                            status=status, target_calls=None, cache_path=str(run/'caches'/stage/cid)))
    # Rotation generalised from ViT: t1 batches ascending, eager first; t2 ascending, compile first;
    # t3 descending, eager first; mode order alternates between consecutive batches.
    # With batches [1,4] this is exactly ViT's [[e1,c1,c4,e4],[c1,e1,e4,c4],[e4,c4,c1,e1]].
    def order(trial):
        bs = list(cfg['batches']) if trial % 3 != 0 else list(reversed(cfg['batches']))
        first = ['eager', 'compile'] if trial % 3 != 2 else ['compile', 'eager']
        return [(mode, b) for i, b in enumerate(bs) for mode in (first if i % 2 == 0 else first[::-1])]
    for trial in range(1, cfg['trials']+1):
        for mode, b in order(trial): add('performance', mode, b, trial)
    for b in cfg['batches']:
        for mode in ['eager', 'compile']: add('outcmp', mode, b)
    for b in cfg['batches']:
        for mode in ['eager', 'compile']: add('cumulative', mode, b, status='not_planned')
    m = dict(run_id=run.name, created_utc=now(), command=[sys.executable, *sys.argv], status='ready', workers=workers,
             npu_budget_used_s=0, execution_order=[], resume_history=[], code_sha256={n: sha256(HERE/n) for n in SOURCES},
             expected_performance_rows=2*len(cfg['batches'])*cfg['trials']*(1+cfg['warmup']+cfg['repeats']))
    for n in SOURCES: shutil.copyfile(HERE/n, run/'code'/n)
    write_json(run/'manifest.json', m)
    tests = [sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE/'tests'), '-p', 'test_math.py', '-v']
    t = subprocess.run(tests, capture_output=True, text=True)
    (run/'logs/math_tests.log').write_text(t.stdout+t.stderr)
    if t.returncode: raise RuntimeError('Math tests failed; no NPU worker launched')
    return run


def stop_own(proc):
    if proc.poll() is not None: return
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait()


def launch(run, cfg, m, w, ev):
    check = command(['furiosa-smi', 'ps']); write_json(run/'logs'/f"{w['case_id']}.device_before.json", check)
    if npu_busy(check):
        w.update(status='held', reason='NPU in use by another process (or furiosa-smi failed)', held_utc=now())
        write_json(run/'manifest.json', m); raise NPUBusy(check.get('stdout', '')+check.get('stderr', ''))
    if any(sha256(HERE/n) != m['code_sha256'][n] for n in WORKER_SOURCES): raise RuntimeError('Measurement code changed mid-run')
    limit = min(cfg['worker_timeout_s'], cfg['total_budget_s']-m['npu_budget_used_s'])
    if limit <= 0: w.update(status='held', reason='Overall NPU budget exhausted'); return
    env = os.environ.copy()
    for k in ['TORCH_LOGS', 'FURIOSA_TORCH_LOG', 'FURIOSA_TORCH_PROFILE']: env.pop(k, None)
    env['HF_HUB_OFFLINE'] = '1'
    cmd = [sys.executable, '-u', str(HERE/'bert_worker.py'), '--run', str(run), '--case', w['case_id']]
    w.update(status='running', started_utc=now(), command=cmd, limit_s=limit)
    m['execution_order'].append(w['case_id']); write_json(run/'manifest.json', m)
    ev.emit('worker_start', case_id=w['case_id'], limit_s=limit)
    proc = None; started = time.perf_counter()
    try:
        with (run/'logs'/f"{w['case_id']}.stdout.log").open('w') as out, (run/'logs'/f"{w['case_id']}.stderr.log").open('w') as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, start_new_session=True, cwd=str(HERE))
            w['pid'] = proc.pid; write_json(run/'manifest.json', m)
            try:
                code = proc.wait(timeout=limit)
                rp = run/'cases'/w['case_id']/'result.json'
                rs = read_json(rp).get('status', 'error') if rp.exists() else 'error'
                w.update(returncode=code, status=rs if code == 0 else 'error')
            except subprocess.TimeoutExpired:
                stop_own(proc); w.update(status='timeout', reason='Operational time limit; not an unsupported-operation verdict')
    except BaseException:
        if proc is not None: stop_own(proc)
        w.update(status='interrupted'); raise
    finally:
        el = time.perf_counter()-started; m['npu_budget_used_s'] += el
        w.update(ended_utc=now(), wall_elapsed_s=el); write_json(run/'manifest.json', m)
        ev.emit('worker_end', case_id=w['case_id'], status=w['status'], wall_elapsed_s=el, npu_budget_used_s=m['npu_budget_used_s'])


def advance(run, stage):
    cfg, m = read_json(run/'config.json'), read_json(run/'manifest.json'); ev = Events(run/'logs/orchestrator.jsonl')
    from bert_analyze import analyze, plan_cumulative
    if stage == 'cumulative':
        analyze(run); plan_cumulative(run); m = read_json(run/'manifest.json')
    m['status'] = 'running_'+stage; write_json(run/'manifest.json', m)
    code = 0
    try:
        for w in [w for w in m['workers'] if w['stage'] == stage]:
            if w['status'] not in ['not_run', 'held']: continue
            if stage == 'cumulative' and not w.get('target_calls'): continue
            if w['status'] == 'held' and w.get('reason', '').startswith('Overall'): continue
            if stage == 'performance':
                prev = next((p for p in m['workers'] if p['stage'] == 'performance' and (p['mode'], p['batch']) == (w['mode'], w['batch'])
                             and p['trial'] < w['trial'] and p['status'] in FAILED), None)
                if prev:
                    w.update(status='skipped', reason=SKIP_REASON, skipped_because=prev['case_id'], skipped_utc=now())
                    write_json(run/'manifest.json', m); ev.emit('worker_skipped', case_id=w['case_id'], because=prev['case_id'])
                    continue
            launch(run, cfg, m, w, ev)
        m['status'] = 'finished_'+stage
    except NPUBusy as e:
        m['status'] = 'held_npu_busy'; code = 3
        print('NPU_BUSY: another process is using the NPU; stopped before next worker\n'+str(e), flush=True)
    finally:
        m['last_update_utc'] = now(); write_json(run/'manifest.json', m)
        write_json(run/'logs'/f'device_after_{stage}.json', command(['furiosa-smi', 'ps']))
        analyze(run)
        ev.emit('stage_end', stage=stage, status=m['status'], npu_budget_used_s=m['npu_budget_used_s'])
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'resume', 'analyze'])
    p.add_argument('--run-dir', type=Path); p.add_argument('--stage', choices=STAGES, default='performance')
    p.add_argument('--device', default='furiosa:0')
    p.add_argument('--batches', type=lambda s: sorted({int(b) for b in s.split(',')}), default=BATCHES,
                   help='run only: comma-separated batch sizes (default 1,2,4,8,16)')
    p.add_argument('--trials', type=int, default=3, help='run only: independent sets per (mode, batch) (default 3)')
    p.add_argument('--fusion', type=int, choices=FUSIONS, default=None,
                   help='run only: PEs fused into furiosa:0 via furiosa.torch.set_fusion (default: not set = SDK default)')
    a = p.parse_args()
    if a.action != 'run' and (a.batches != BATCHES or a.trials != 3 or a.fusion is not None):
        p.error('--batches/--trials/--fusion apply only to `run`; resume/analyze use the run\'s config.json')
    if a.trials < 1 or min(a.batches) < 1: p.error('--trials and batch sizes must be >= 1')
    run = fresh(a) if a.action == 'run' else a.run_dir.resolve()
    print('BERT_RUN '+str(run), flush=True)
    if a.action == 'analyze':
        from bert_analyze import analyze
        analyze(run); return 0
    if a.action == 'resume':
        m = read_json(run/'manifest.json')
        m['resume_history'].append(dict(utc=now(), command=[sys.executable, *sys.argv], stage=a.stage)); write_json(run/'manifest.json', m)
    return advance(run, a.stage if a.action == 'resume' else 'performance')


if __name__ == '__main__': raise SystemExit(main())

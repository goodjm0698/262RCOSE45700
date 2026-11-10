"""One fresh-process BERT worker. Adapted from /root/ViT/vit_pilot/fastpath_compare_worker.py.

Kept from ViT: strict cpu_fallback(False), per-worker fresh cache_context, inference_mode,
current-stream synchronize timer (common.timed_call), runner_setup_ms measured separately,
first/warmup/repeat|work schedule, outputs saved for first/last warmup/last repeat,
every cumulative work output checked on CPU outside the timer.
Changed for BERT: HF model + CPU 4D mask (step4_compare.py), logits [B,2], no MHA fastpath
control (HF BERT does not use nn.MultiheadAttention), no diagnostic stage, new 'outcmp' stage
that runs the 16 step4 sentences in chunks of B.
"""
import argparse
from pathlib import Path
import signal
import time
import traceback
from common import Events, now, read_json, write_json, sha256, timed_call

CACHE_POLICY = 'fresh_process_private_sdk_cache_initially_absent_system_cold_unverified'
TIMING_SCOPE = 'cpu_dispatch_runner_and_current_stream_completion_excludes_external_prepost_transfers'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--case', required=True)
    a = p.parse_args(); cfg = read_json(a.run/'config.json')
    spec = next(w for w in read_json(a.run/'manifest.json')['workers'] if w['case_id'] == a.case)
    folder = a.run/'cases'/a.case; folder.mkdir(parents=True, exist_ok=False)
    ev = Events(folder/'events.jsonl'); stage = spec['stage']; B = spec['batch']
    r = dict(**{k: spec[k] for k in ['case_id', 'stage', 'mode', 'batch', 'trial']}, status='running', started_utc=now(),
             timings=[], outputs={}, output_checks={}, output_valid=False, dtype='torch.bfloat16',
             cache_state=CACHE_POLICY, timing_scope=TIMING_SCOPE, compiler_time_ms=None,
             script_sha256=sha256(Path(__file__)), inference_mode=True, active_call=None)
    def save(): write_json(folder/'result.json', r)
    def terminate(signum, frame): raise TimeoutError('Own worker received termination signal')
    signal.signal(signal.SIGTERM, terminate)
    save()
    try:
        import torch
        import furiosa.torch as ft
        import transformers
        from bert_model import load_model, encode, inputs_hash, SENTENCES, SEQ_LEN
        dev = torch.device(cfg['device'])
        if not ft.is_available() or cfg['device'] not in ft.get_available_devices(): raise RuntimeError('Allowed device unavailable')
        if torch._dynamo.config.suppress_errors or ft.backend.eager_fallback: raise RuntimeError('Error suppression/backend eager fallback must be disabled')
        cache = Path(spec['cache_path']); r.update(cache_path=str(cache), cache_existed_at_entry=cache.exists())
        if cache.exists(): raise RuntimeError('Never reuse or delete an existing worker cache')
        with ft.config.cpu_fallback(False), ft.config.cache_context(str(cache), summary=False), torch.inference_mode():
            r['effective'] = dict(cpu_fallback=ft.config.get_cpu_fallback(), device_configuration=str(ft.get_device_configuration()),
                compiler_config=str(ft.config.get_compiler_config()), cache_config=str(ft.config.get_cache_config()),
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                backend_eager_fallback=ft.backend.eager_fallback, suppress_errors=torch._dynamo.config.suppress_errors,
                torch_version=torch.__version__, transformers_version=transformers.__version__,
                compile_kwargs='backend only; all other compile options omitted')
            if r['effective']['cpu_fallback']: raise RuntimeError('cpu_fallback must be False')
            ev.emit('model_input_prepare_start'); start = time.perf_counter_ns()
            tok, model = load_model()
            r['attn_implementation'] = model.config._attn_implementation
            if any(p.dtype != torch.bfloat16 for p in model.parameters()): raise RuntimeError('All parameters must be BF16')
            sents = SENTENCES if stage == 'outcmp' else SENTENCES[:B]
            chunks_cpu = [encode(tok, sents[i:i+B]) for i in range(0, len(sents), B)]
            r['input_hash'] = inputs_hash(chunks_cpu[0]) if stage != 'outcmp' else None
            r['input_hashes'] = [inputs_hash(c) for c in chunks_cpu]
            r['valid_tokens'] = [c['valid_tokens'] for c in chunks_cpu]
            r['sentence_indices'] = list(range(len(sents)))
            model = model.to(dev)
            chunks = [{k: c[k].to(dev) for k in ['input_ids', 'token_type_ids', 'attention_mask']} for c in chunks_cpu]
            stream = ft.current_stream(cfg['device']); stream.synchronize()
            if not stream.query(): raise RuntimeError('Official completion query failed')
            r.update(model_input_prepare_ms=(time.perf_counter_ns()-start)/1e6, sync_checked=True,
                     input_metadata={k: dict(shape=list(v.shape), dtype=str(v.dtype), device=str(v.device)) for k, v in chunks[0].items()})
            ev.emit('model_input_prepare_end', elapsed_ms=r['model_input_prepare_ms'])

            ev.emit('runner_setup_start'); start = time.perf_counter_ns()
            runner = model if spec['mode'] == 'eager' else torch.compile(model, backend=ft.backend)
            r['runner_setup_ms'] = (time.perf_counter_ns()-start)/1e6
            ev.emit('runner_setup_end', elapsed_ms=r['runner_setup_ms']); save()
            call = lambda inp: runner(**inp)

            schedule = [('first', 1)]
            if stage == 'outcmp':
                schedule += [('chunk', i) for i in range(2, len(chunks)+1)]   # chunk 1 is the first call
                n = len(chunks)
            else:
                schedule += [('warmup', i) for i in range(1, cfg['warmup']+1)]
                phase = 'work' if stage == 'cumulative' else 'repeat'
                n = spec['target_calls'] if stage == 'cumulative' else cfg['repeats']
                schedule += [(phase, i) for i in range(1, n+1)]

            def check_output(y, key, store=True):
                cpu = y.detach().to('cpu', non_blocking=False).float(); stream.synchronize()
                m = dict(shape=list(y.shape), dtype=str(y.dtype), device=str(y.device),
                         nan_count=int(torch.isnan(cpu).sum()), inf_count=int(torch.isinf(cpu).sum()))
                ok = m['shape'] == [B, 2] and m['dtype'] == 'torch.bfloat16' and not (m['nan_count'] or m['inf_count'])
                if not ok: raise RuntimeError('Invalid output '+key+': '+str(m))
                if store:
                    target = a.run/'outputs'/f'{a.case}_{key}.pt'; torch.save(cpu, target)
                    r['outputs'][key] = str(target); r['output_checks'][key] = dict(**m, valid=True); r['output_file'] = str(target)
                return cpu

            outcmp_parts = []; prev = None; phase_started = None; phase_times = []
            for call_index, (phase, iteration) in enumerate(schedule, 1):
                active = dict(call_index=call_index, phase=phase, iteration=iteration)
                if phase != prev:
                    if prev is not None:
                        ev.emit('phase_end', phase=prev, wall_ms=(time.perf_counter_ns()-phase_started)/1e6, measured_sum_ms=sum(phase_times))
                    phase_started = time.perf_counter_ns(); phase_times = []; prev = phase
                    r['active_call'] = active; save(); ev.emit('phase_start', phase=phase, requested_calls=sum(p == phase for p, _ in schedule))
                x = chunks[call_index-1] if stage == 'outcmp' else chunks[0]
                row = dict(run_id=cfg['run_id'], case_id=a.case, stage=stage, mode=spec['mode'], batch=B, trial=spec['trial'],
                           phase=phase, iteration=iteration, elapsed_ms=None, timing_valid=False, status='running')
                try:
                    out, elapsed, valid = timed_call(call, x, stream.synchronize)
                except Exception as e:
                    row.update(status='error', error=repr(e)); r['timings'].append(row); raise
                y = out.logits
                row.update(elapsed_ms=elapsed, elapsed_ns=round(elapsed*1000000), timing_valid=valid, status='ok')
                r['timings'].append(row); phase_times.append(elapsed)
                if stage == 'outcmp':
                    outcmp_parts.append(check_output(y, f'chunk{call_index}', store=False))
                elif phase == 'first' or (phase == 'warmup' and iteration == cfg['warmup']) or (phase in ['repeat', 'work'] and iteration == n):
                    check_output(y, phase); save()
                elif stage == 'cumulative' and phase == 'work':
                    check_output(y, 'work_check', store=False)   # every work output checked, outside timer
                if stage == 'cumulative' and phase == 'work' and iteration % 500 == 0:
                    r['active_call'] = active; save(); ev.emit('work_progress', completed=iteration, target=n, measured_sum_ms=sum(phase_times))
            ev.emit('phase_end', phase=prev, wall_ms=(time.perf_counter_ns()-phase_started)/1e6, measured_sum_ms=sum(phase_times))
            if stage == 'outcmp':
                allout = torch.cat(outcmp_parts)
                target = a.run/'outputs'/f'{a.case}_all16.pt'; torch.save(allout, target)
                r['outputs']['all16'] = str(target); r['output_file'] = str(target)
            r.update(active_call=None, output_valid=True, cumulative_work_outputs_checked=n if stage == 'cumulative' else None)
            r['status'] = 'ok'
        r['cpu_fallback_after_context'] = ft.config.get_cpu_fallback()
    except Exception:
        r.update(status='error', error=traceback.format_exc()); traceback.print_exc(); ev.emit('worker_error', error=r['error'])
    finally:
        r['ended_utc'] = now(); save(); ev.emit('worker_end', status=r['status'])
    return 0 if r['status'] == 'ok' else 1


if __name__ == '__main__': raise SystemExit(main())

"""Aggregation, break-even, cumulative validation, output comparison.
Adapted from /root/ViT/vit_pilot/fastpath_compare_analysis.py, fastpath_support.compare_pair,
attention_analysis.compare_files and analyze.warm_state. Only current-run measurements are used.
"""
from fractions import Fraction
import math
from pathlib import Path
import statistics as st
from common import read_json, write_json, write_csv
from bert_math import costs, solve, G, T, duration_ms, NUMERIC_RULE

LABELS = ['negative', 'positive']


def find_case(m, stage, mode, batch, trial=1):
    return next((w for w in reversed(m['workers']) if (w['stage'], w['mode'], w['batch'], w['trial']) == (stage, mode, batch, trial)), None)


def result_for(run, case):
    if not case: return {}
    p = Path(run)/'cases'/case['case_id']/'result.json'
    return read_json(p) if p.exists() else {}


def valid(row):
    v = row.get('elapsed_ms')
    return row.get('status') == 'ok' and row.get('timing_valid') and isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def mean_range(values):
    return dict(values=values, median=st.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None, count=len(values))


def warm_state(values):
    # Copied from /root/ViT/vit_pilot/analyze.py
    if len(values) < 10:
        return 'unknown', 'Fewer than 10 valid repeat calls; no steady-state claim'
    cv = st.stdev(values)/st.mean(values); n = len(values)//2
    drift = abs(st.mean(values[:n])-st.mean(values[n:]))/st.mean(values)
    return ('preliminary_stable' if cv <= 0.10 and drift <= 0.10 else 'not_stable',
            f'Heuristic only: repeat CV={cv:.4f}, half-mean relative drift={drift:.4f}; thresholds 0.10')


def compare_tensors(a, b, sentence_ids):
    """candidate a (compile) vs reference b (eager); CPU FP32. Adapted from ViT compare_files."""
    import torch
    a, b = a.float(), b.float()
    r = dict(tolerance_status='pending', allclose=None, accuracy_computed=False, structural_finite_ok=False,
             top1_matches=None, compared_samples=None)
    if a.shape != b.shape or list(a.shape) != [len(sentence_ids), 2]:
        r['reason'] = 'Shape mismatch'; return r
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        r['reason'] = 'NaN/Inf present'; return r
    d = (a-b).abs(); ad, bd = a.double(), b.double(); norms = ad.norm(dim=1)*bd.norm(dim=1)
    cos = [float((ad[i]*bd[i]).sum()/norms[i]) if float(norms[i]) > 0 else None for i in range(len(a))]
    ai, bi = a.argmax(1), b.argmax(1)
    r.update(structural_finite_ok=True, max_abs_error=float(d.max()), mean_abs_error=float(d.mean()),
             cosine_per_sample=cos, cosine_mean=st.mean(cos) if None not in cos else None,
             cosine_min=min(cos) if None not in cos else None,
             top1_matches=int((ai == bi).sum()), compared_samples=len(a), top1_ratio=float((ai == bi).float().mean()),
             samples=[dict(sentence_id=sentence_ids[i], compile_label=LABELS[int(ai[i])], eager_label=LABELS[int(bi[i])],
                           compile_logits=[float(v) for v in a[i]], eager_logits=[float(v) for v in b[i]],
                           top1_match=bool(ai[i] == bi[i]), cosine=cos[i]) for i in range(len(a))])
    r['mismatches'] = [x for x in r['samples'] if not x['top1_match']]
    return r


def compare_pair(run, m, stage, batch, trial=1):
    import torch
    ws = [find_case(m, stage, mode, batch, trial) for mode in ['eager', 'compile']]
    rs = [result_for(run, w) for w in ws]
    out = dict(stage=stage, batch=batch, trial=trial, eligible=False, eager_case_id=ws[0] and ws[0]['case_id'],
               compile_case_id=ws[1] and ws[1]['case_id'], tolerance_status='pending')
    if any(not w or w['status'] != 'ok' or r.get('status') != 'ok' or not r.get('output_valid') for w, r in zip(ws, rs)):
        out['reason'] = 'Both workers and outputs must complete normally'; return out
    if any(rs[0].get(k) != rs[1].get(k) for k in ['input_hashes', 'dtype', 'timing_scope', 'cache_state', 'attn_implementation']):
        out['reason'] = 'Paired input/dtype/timing/cache/attention policy mismatch'; return out
    keys = ['device_configuration', 'compiler_config', 'torch_threads', 'cpu_fallback', 'backend_eager_fallback', 'suppress_errors']
    if any(rs[0]['effective'].get(k) != rs[1]['effective'].get(k) for k in keys) or rs[0]['effective']['cpu_fallback']:
        out['reason'] = 'Effective settings differ or cpu_fallback not False'; return out
    ids = rs[0]['sentence_indices']
    phases = ['all16'] if stage == 'outcmp' else [ph for ph in ['first', 'warmup', 'repeat', 'work'] if ph in rs[0]['outputs'] and ph in rs[1]['outputs']]
    out['phase_comparisons'] = []
    for ph in phases:
        e, c = [torch.load(r['outputs'][ph], map_location='cpu', weights_only=True) for r in rs]
        v = compare_tensors(c, e, ids); out['phase_comparisons'].append(dict(phase=ph, **v))
    main = out['phase_comparisons'][-1]   # all16, or last repeat/work output
    out.update({k: v for k, v in main.items() if k != 'phase'}, primary_phase=main['phase'])
    n = len(ids)
    out['eligible'] = all(p['structural_finite_ok'] and p['top1_matches'] == n for p in out['phase_comparisons'])
    out['reason'] = 'Output tolerance pending; Top-1 consistency only' if out['eligible'] else 'Structure/finite/Top-1 check failed'
    return out


def worker_summary(run, w, cfg):
    r = result_for(run, w); rows = r.get('timings', [])
    vals = [x['elapsed_ms'] for x in rows if x['phase'] == 'repeat' and valid(x)]
    state, why = warm_state(vals); c = costs(r, cfg['warmup'], cfg['repeats'])
    first = [x['elapsed_ms'] for x in rows if x['phase'] == 'first' and valid(x)]
    warm = [x['elapsed_ms'] for x in rows if x['phase'] == 'warmup' and valid(x)]
    s = dict(run_id=cfg['run_id'], case_id=w['case_id'], mode=w['mode'], batch=w['batch'], trial=w['trial'], status=w['status'],
             model_id=cfg['model'], seq_len=cfg['seq_len'], dtype='torch.bfloat16', cpu_fallback=False,
             requested_repeats=cfg['repeats'], valid_repeats=len(vals),
             model_input_prepare_ms=r.get('model_input_prepare_ms'), runner_setup_ms=r.get('runner_setup_ms'),
             first_call_ms=first[0] if len(first) == 1 else None, warmup_ms=warm, valid_warmups=len(warm),
             warmup_sum_ms=sum(warm) if len(warm) == cfg['warmup'] else None, A_ms=float(c['A']) if c['status'] == 'valid' else None,
             mean_ms=st.mean(vals) if vals else None, median_ms=st.median(vals) if vals else None,
             stdev_ms=st.stdev(vals) if len(vals) > 1 else None, CV=st.stdev(vals)/st.mean(vals) if len(vals) > 1 else None,
             min_ms=min(vals) if vals else None, max_ms=max(vals) if vals else None,
             throughput_sentences_s=w['batch']*len(vals)*1000/sum(vals) if vals and c['status'] == 'valid' else None,
             cost_status=c['status'], cost_reason=c.get('reason'), warm_state_assessment=state, warm_state_reason=why,
             device_configuration=r.get('effective', {}).get('device_configuration'), attn_implementation=r.get('attn_implementation'),
             timing_scope=r.get('timing_scope'), cache_policy=r.get('cache_state'), input_hash=r.get('input_hash'), compiler_time_ms=None)
    return s, c


def initial_and_work(result, warmups):
    """Contiguous observed prefix only (copied from ViT)."""
    try:
        s = duration_ms(result.get('runner_setup_ms')); rows = result.get('timings', [])
        first = [x for x in rows if x['phase'] == 'first']
        warm = sorted([x for x in rows if x['phase'] == 'warmup'], key=lambda x: x['iteration'])
        if len(first) != 1 or len(warm) != warmups or any(not valid(x) for x in first+warm): raise LookupError('Initial calls incomplete/invalid')
        f = duration_ms(first[0]['elapsed_ms'], True); wsum = sum((duration_ms(x['elapsed_ms'], True) for x in warm), Fraction())
        work = []
        for x in sorted([x for x in rows if x['phase'] == 'work'], key=lambda x: x['iteration']):
            if x['iteration'] != len(work)+1 or not valid(x): break
            work.append(duration_ms(x['elapsed_ms'], True))
        return dict(status='valid_prefix', S=s, F=f, W=wsum, A=s+f+wsum, work=work)
    except (LookupError, ValueError, KeyError) as e:
        return dict(status='insufficient_data', reason=str(e))


def analyze_cumulative(run, cfg, m, by_batch, outputs):
    rows = []; summaries = []
    for b in by_batch:
        batch = b['batch']; trial = b['representative_trial']
        ws = [find_case(m, 'cumulative', mode, batch) for mode in ['eager', 'compile']]
        rs = [result_for(run, w) for w in ws]; cs = [initial_and_work(r, cfg['warmup']) for r in rs]
        output = next(x for x in outputs if (x['stage'], x['batch']) == ('cumulative', batch))
        summary = dict(batch=batch, status='predicted_only', prediction_source_trial=trial, planned_N=ws[0].get('target_calls'),
                       first_observed_compile_advantage_N=None, final_winning_suffix_start_N=None, maintained_after_crossing=None,
                       transitions=[], completed_common_calls=0, primary_median_N=b['N_win']['median'],
                       primary_trial_N_range=[b['N_win']['minimum'], b['N_win']['maximum']],
                       validation_output_eligible=output['eligible'], reason='Cumulative workers not completed')
        if trial and all(c['status'] == 'valid_prefix' for c in cs):
            e, c = cs; count = min(len(e['work']), len(c['work']))
            pc = [costs(result_for(run, find_case(m, 'performance', mode, batch, trial)), cfg['warmup'], cfg['repeats']) for mode in ['eager', 'compile']]
            obs = [e['A'], c['A']]; before = None; trans = []; first_win = None; last_entry = None
            for n in range(0, count+1):
                if n: obs[0] += e['work'][n-1]; obs[1] += c['work'][n-1]
                pred = [q['A']+n*q['t'] for q in pc]; diff = obs[1]-obs[0]
                rows.append(dict(run_id=cfg['run_id'], batch=batch, N=n, K=n+cfg['warmup']+1, effective_sentences=batch*n,
                                 prediction_source_trial=trial, observed_eager_ms=float(obs[0]), observed_compile_ms=float(obs[1]),
                                 observed_difference_ms=float(diff), predicted_eager_ms=float(pred[0]), predicted_compile_ms=float(pred[1]),
                                 predicted_difference_ms=float(pred[1]-pred[0]),
                                 output_validation_status='eligible' if output['eligible'] else 'incomplete_or_failed'))
                if n:
                    state = 'compile_faster' if diff < 0 else 'tie' if diff == 0 else 'eager_faster'
                    if state != before: trans.append(dict(N=n, state=state, difference_ms=float(diff)))
                    if state == 'compile_faster' and first_win is None: first_win = n
                    if state == 'compile_faster' and before != 'compile_faster': last_entry = n
                    before = state
            normal = all(w['status'] == 'ok' and r.get('status') == 'ok' for w, r in zip(ws, rs)) and ws[0].get('target_calls') == count and output['eligible']
            status = 'observed_crossing_reached' if normal and first_win else 'prediction_not_confirmed_within_observed_range' if normal else 'incomplete'
            summary.update(status=status, completed_common_calls=count, first_observed_compile_advantage_N=first_win if output['eligible'] else None,
                           final_winning_suffix_start_N=last_entry if normal and before == 'compile_faster' else None,
                           maintained_after_crossing=(normal and first_win is not None and last_entry == first_win and before == 'compile_faster'),
                           observed_end_difference_ms=float(obs[1]-obs[0]), transitions=trans, reason=None if normal else 'Incomplete/invalid paired execution',
                           observed_startup={mode: {k+'_ms': float(q[k]) for k in ['S', 'F', 'W', 'A']} for mode, q in zip(['eager', 'compile'], cs)},
                           observed_work_mean_ms={mode: float(sum(q['work'], Fraction())/len(q['work'])) if q['work'] else None for mode, q in zip(['eager', 'compile'], cs)})
            if first_win and b['N_win']['median']:
                summary['observed_minus_predicted_median_N'] = first_win-b['N_win']['median']
                summary['relative_crossing_difference'] = first_win/b['N_win']['median']-1
            summary['posthoc_observed_startup_with_primary_rates'] = solve(e['A'], c['A'], pc[0]['t'], pc[1]['t'], batch, cfg['warmup'])
        summaries.append(summary)
    write_csv(Path(run)/'cumulative_validation.csv', rows, ['run_id', 'batch', 'N', 'K', 'effective_sentences', 'prediction_source_trial',
              'observed_eager_ms', 'observed_compile_ms', 'observed_difference_ms', 'predicted_eager_ms', 'predicted_compile_ms',
              'predicted_difference_ms', 'output_validation_status'])
    res = dict(batches=summaries, folded_into_independent_trials=False, primary_predictions_refitted=False,
               scope='sum of S/F/W/work measured intervals; outside-timer checks and process wall excluded')
    write_json(Path(run)/'cumulative_validation_summary.json', res)
    return res


def plan_cumulative(run):
    """Copied from ViT plan_cumulative; gate is performance + outcmp output eligibility."""
    run = Path(run); cfg, m, be = [read_json(run/n) for n in ['config.json', 'manifest.json', 'break_even_summary.json']]
    plans = []; remaining = max(0, cfg['total_budget_s']-m['npu_budget_used_s'])
    oc = read_json(run/'output_comparison.json')['pairs']
    for batch in [1, 4]:
        ws = [find_case(m, 'cumulative', mode, batch) for mode in ['eager', 'compile']]
        if any(w.get('started_utc') for w in ws):
            plans.append(dict(batch=batch, status='existing_plan_preserved', target_calls=ws[0].get('target_calls'))); continue
        paired = [v for v in be['trials'] if v['batch'] == batch]
        good = [v for v in paired if v['status'] in ['finite_break_even', 'already_beneficial', 'always_beneficial'] and v['N_win'] is not None]
        gate = next(p for p in oc if (p['stage'], p['batch']) == ('outcmp', batch))
        reason = None
        if len(good) != len(paired) or not good: reason = 'No uniformly valid finite paired prediction'
        elif not gate['eligible']: reason = 'outcmp gate failed: '+str(gate.get('reason'))
        if reason:
            for w in ws: w.update(status='held', reason=reason)
            plans.append(dict(batch=batch, status='predicted_only', reason=reason)); continue
        median = st.median([v['N_win'] for v in good]); desired = math.ceil(Fraction(6, 5)*Fraction(str(median)))
        if median == 1: desired = max(20, desired)
        requested = min(desired, cfg['max_cumulative_calls'])
        def bound(mode, n):
            vals = []
            for trial in range(1, cfg['trials']+1):
                r = result_for(run, find_case(m, 'performance', mode, batch, trial)); c = costs(r, cfg['warmup'], cfg['repeats'])
                if c['status'] == 'valid': vals.append((r['model_input_prepare_ms']+float(c['A']+n*c['t']))/1000)
            return 1.5*max(vals)+60+n*.001
        def fits(n):
            bs = [bound(mode, n) for mode in ['eager', 'compile']]
            return max(bs) <= cfg['worker_timeout_s'] and sum(bs) <= remaining
        low, high = 0, requested
        while low < high:
            mid = (low+high+1)//2
            if fits(mid): low = mid
            else: high = mid-1
        n = low
        if n == 0:
            reason = 'Conservative estimate cannot fit both workers in remaining budget'
            for w in ws: w.update(status='held', reason=reason)
            plans.append(dict(batch=batch, status='predicted_only', reason=reason, desired_calls=desired)); continue
        bs = {mode: bound(mode, n) for mode in ['eager', 'compile']}; remaining -= sum(bs.values())
        for w in ws: w.update(status='not_run', target_calls=n)
        plans.append(dict(batch=batch, status='planned_full_target' if n >= desired else 'shortened_by_budget_or_cap',
                          median_N_win=median, desired_calls=desired, target_calls=n, conservative_worker_wall_bounds_s=bs,
                          remaining_reserved_budget_s=remaining))
    write_json(run/'cumulative_plan.json', dict(plans=plans, budget_used_at_planning_s=m['npu_budget_used_s'],
               total_budget_s=cfg['total_budget_s'], rule=cfg['cumulative_target_rule']))
    write_json(run/'manifest.json', m)


def analyze(run):
    run = Path(run); cfg, m = read_json(run/'config.json'), read_json(run/'manifest.json')
    timings = []; summary = []; comp = {}; be = []; prefix = []
    for w in m['workers']:
        if w['stage'] != 'performance': continue
        r = result_for(run, w)
        for row in r.get('timings', []):
            timings.append(dict(**row, dtype='torch.bfloat16', cpu_fallback=False, input_hash=r.get('input_hash'), cache_policy=r.get('cache_state')))
        if w is find_case(m, 'performance', w['mode'], w['batch'], w['trial']) and r:
            s, c = worker_summary(run, w, cfg); summary.append(s); comp[w['case_id']] = c
    outputs = [compare_pair(run, m, st_, b, t) for st_ in ['performance', 'outcmp', 'cumulative'] for b in [1, 4]
               for t in (range(1, cfg['trials']+1) if st_ == 'performance' else [1])]
    for b in [1, 4]:
        for trial in range(1, cfg['trials']+1):
            ws = [find_case(m, 'performance', mode, b, trial) for mode in ['eager', 'compile']]
            cs = [comp.get(w['case_id'], dict(status='insufficient_data', reason='not run')) for w in ws]
            pair = next(p for p in outputs if (p['stage'], p['batch'], p['trial']) == ('performance', b, trial))
            if pair['eligible'] and all(c['status'] == 'valid' for c in cs):
                e, c = cs; v = solve(e['A'], c['A'], e['t'], c['t'], b, cfg['warmup'])
                for mode, q in zip(['eager', 'compile'], cs):
                    v.update({f'S_{mode}_ms': float(q['S']), f'F_{mode}_ms': float(q['F']), f'W_{mode}_ms': float(q['W'])})
                v['compile_speedup'] = float(e['t']/c['t'])
                ss = [next(s for s in summary if s['case_id'] == w['case_id']) for w in ws]
                pooled = math.sqrt(sum(s['stdev_ms']**2 for s in ss)); v['combined_repeat_SD_ms'] = pooled
                v['variation_assessment'] = 'uncertain' if abs(v['delta_t_ms']) <= pooled else 'mean_gap_exceeds_descriptive_combined_SD'
                v['warm_state_assessments'] = {s['mode']: s['warm_state_assessment'] for s in ss}
                for k in range(1, 2+cfg['warmup']):
                    prefix.append(dict(batch=b, trial=trial, K=k, G_eager_ms=float(G(e, k)), G_compile_ms=float(G(c, k)), difference_ms=float(G(c, k)-G(e, k))))
                if v['N_win'] is not None:
                    n = v['N_win']; assert G(e, n+1+cfg['warmup']) == T(e, n) and G(c, n+1+cfg['warmup']) == T(c, n)
            else:
                v = solve(None, None, None, None, b, cfg['warmup']); v['reason'] = pair.get('reason') if not pair['eligible'] else str([c.get('reason') for c in cs])
            v.update(run_id=cfg['run_id'], batch=b, trial=trial, eager_case_id=ws[0]['case_id'], compile_case_id=ws[1]['case_id'],
                     output_validation_status='eligible_tolerance_pending' if pair['eligible'] else 'held')
            be.append(v)
    by_batch = []
    for b in [1, 4]:
        paired = [v for v in be if v['batch'] == b]
        good = [v for v in paired if v['N_win'] is not None and v['status'] in ['finite_break_even', 'already_beneficial', 'always_beneficial']]
        rep = sorted(good, key=lambda x: (x['N_win'], x['trial']))[len(good)//2] if good else None
        S = lambda mode, k: mean_range([s[k] for s in summary if s['batch'] == b and s['mode'] == mode and s['cost_status'] == 'valid'])
        by_batch.append(dict(batch=b, statuses=[dict(trial=v['trial'], status=v['status'], N_win=v['N_win']) for v in paired],
                             valid_finite_paired_trials=len(good), requested_paired_trials=cfg['trials'],
                             N_win=mean_range([v['N_win'] for v in good]), representative_trial=rep['trial'] if rep else None,
                             compile_speedup=mean_range([v['compile_speedup'] for v in paired if v.get('compile_speedup')]),
                             per_trial_speedup={v['trial']: v.get('compile_speedup') for v in paired},
                             **{f'{k}_by_mode': {mode: S(mode, k) for mode in ['eager', 'compile']}
                                for k in ['first_call_ms', 'warmup_sum_ms', 'A_ms', 'mean_ms', 'CV', 'runner_setup_ms', 'model_input_prepare_ms']}))
    write_csv(run/'timings.csv', timings, ['run_id', 'case_id', 'stage', 'mode', 'batch', 'trial', 'phase', 'iteration', 'elapsed_ns', 'elapsed_ms',
              'timing_valid', 'status', 'dtype', 'cpu_fallback', 'input_hash', 'cache_policy'])
    write_csv(run/'trial_summary.csv', summary, list(summary[0]) if summary else ['case_id', 'status'])
    write_csv(run/'break_even_trials.csv', be, ['run_id', 'batch', 'trial', 'eager_case_id', 'compile_case_id', 'status', 'A_e_ms', 'A_c_ms', 't_e_ms', 't_c_ms',
              'delta_A_ms', 'delta_t_ms', 'real_crossing_N', 'N_win', 'K_win', 'effective_images', 'forward_count', 'compile_speedup',
              'last_advantage_N', 'tie_N', 'strict_at_N', 'predecessor_N', 'difference_at_predecessor_ms', 'difference_at_N_ms',
              'variation_assessment', 'output_validation_status', 'reason'])
    write_json(run/'break_even_summary.json', dict(numeric_rule=NUMERIC_RULE, scenario='T(N)=S+F+sum(W)+N*mean_repeat; N>=1 calls after first+warmups',
               note='effective_images column = batch*N sentences (name kept from ViT)', prepare_excluded=True, trials=be, batches=by_batch))
    write_json(run/'output_comparison.json', dict(accuracy_computed=False, tolerance_status='pending', pairs=outputs))
    write_json(run/'initial_prefix.json', dict(rows=prefix))
    cum = analyze_cumulative(run, cfg, m, by_batch, outputs)
    return dict(summary=summary, break_even=be, batches=by_batch, outputs=outputs, cumulative=cum)

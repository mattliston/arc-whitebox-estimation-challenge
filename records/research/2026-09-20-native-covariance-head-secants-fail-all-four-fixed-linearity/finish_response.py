"""Preserve the target-blind response certificate, positive or negative."""
import datetime,shutil
from common import HERE,D,read,write,pin,verify,seal_check
seal_check();q=read(HERE/'response_qualified.json');u=read(HERE/'unit.json')
guards=[read(p) for p in HERE.glob('*_watch.json')]
assert len(guards)==16 and all(g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0 for g in guards)
resources=dict(guarded_groups=len(guards),all_groups_retired=True,max_swap_growth_bytes=0,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),elapsed_guarded_s=sum(g['elapsed_s'] for g in guards),
    full_native_calls=q['full_native_calls'],full_native_flops=q['full_native_flops'],
    small_source_calls=sum(len(r['bills']) for r in u['rows']),small_source_flops=sum(sum(r['bills']) for r in u['rows']),
    new_target_arrays_accessed=0,fit_steps=0,new_samples=0,new_dataset_downloads=0,GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE/'response_resources.json',resources)
status='passed; fixed acquisition is eligible' if q['passed'] else 'failed; no acquisition, target scoring, fit or upload follows'
lines=['# Covariance-head actual native response certificate','',status+'.','',
    'Parent is the exact packed covariance source submitted as331684. Only one',
    'FP32 output coefficient per variant changes; all source inverses and other',
    'parameter bytes are exact. Actual FNP/independent formulas, symmetry and',
    'local head linearity pass. Every full run preserves first8 output rows,',
    'returns finite/nonnegative16x1024 FP32, records its own complete cold bill,',
    'and retires caches. Repeated36/head0+epsilon output and bill are exact.','']
for cell in q['cells']:
    lines.append(f"- Rank{cell['rank']},head{cell['coordinate']}: spacing SNR{cell['spacing_snr']:.9g}, forward/central discrepancy{cell['forward_vs_centered_relative']:.9g}, double-step curvature{cell['double_curvature_relative']:.9g}; pass={cell['passed']}.")
lines+=['','These are actual native secants in two sampled coordinates at epsilon1/16.',
    'They are not a full exact Jacobian, large-step certificate or accuracy result.',
    'No truth arrays were decoded and no fitting occurred. Failed gates, if any,',
    'are preserved without changing epsilon, directions, thresholds or weights.',
    f"All16 guards retired,zero added swap,peak{resources['peak_group_rss_bytes']}B.",
    'Official bank remains331672 at3.6316074532884024e-9; goal is active.','']
with (HERE/'RESPONSE_RESULT.md').open('x') as f:f.write('\n'.join(lines))
files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
record=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),response_stage_complete=True,response_gate_passed=q['passed'],
    study_closed=not q['passed'],fit_steps=0,truth_arrays_accessed=False,uploaded=False,goal_achieved=False,
    artifacts=[pin(p) for p in files],external_pins=[pin(D/'full_lower_covariance_edge_packed_v1/candidate.py'),pin(D/'official_best_331672.json')],resources=resources)
for p in record['artifacts']+record['external_pins']:verify(p)
write(HERE/'response_closeout.json',record)
print(dict(response_gate_passed=q['passed'],artifact_pins=len(files),resources=resources),flush=True)

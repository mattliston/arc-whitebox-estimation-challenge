import datetime
import shutil
from common import HERE,D,OLD,PARENT,read,write,pin,verify,seal_check

seal_check()
assert read(HERE/'completed.json')['completed'] and read(HERE/'capture_qualified.json')['passed']
result=read(HERE/'response_result.json');outcome=read(HERE/'outcome.json')
comparison=read(HERE/'selector_differences.json')
rank38=[r for r in comparison['within_canonical'] if r['rank']==38]
assert len(rank38)==6
rank38_selected_indices_fixed=all(r['delta']['membership_change_cells']==0 and
    r['delta']['count_change_cells']==0 and r['delta']['pure_order_cells']==0 for r in rank38)
guards={p.name:read(p) for p in HERE.glob('*_watch.json')}
assert len(guards)==23
assert all(g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and
           g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<8_000_000_000 for g in guards.values())
calls=[read(p) for p in HERE.glob('*/result.json')]
assert len(calls)==19 and all(not r['truth_accessed'] for r in calls)
assert all(r['max_rss_bytes']<8_000_000_000 for r in calls)
leaves=[g for k,g in guards.items() if k!='controller_watch.json']
resources=dict(guarded_groups=len(guards),all_groups_retired=True,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards.values()),max_swap_growth_bytes=0,
    leaf_elapsed_s=sum(g['elapsed_s'] for g in leaves),full_native_calls=len(calls),
    full_native_flops=sum(r['flops'] for r in calls),peak_native_self_rss_bytes=max(r['max_rss_bytes'] for r in calls),
    selector_trace_calls=45*len(calls),fit_steps=0,truth_arrays_decoded=0,new_samples=0,new_downloads=0,GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free)
assert resources['artifact_bytes_before_closeout']<75_000_000
write(HERE/'resources.json',resources)
lines=['# Retained-index ordering response diagnostic','',
    'Exact official331684 remains the adjusted3.5844144212297072e-9 bank,100/100 completed.',
    'One source edit sorts retained selector indices ascending after original top-count selection.',
    'All existing thresholds, retained sets on identical inputs, FP32 kernels and head coefficients remain fixed.',
    'The original16-head response study remains closed negative; the same epsilon1/16 and coordinates0/8',
    'were tested on this separately pinned ordering intervention, with no targets or fitting.',
    'Thirty selector fixtures and independent matrix/CP/QR permutation identities pass.',
    'Four original rank36/head0 replays match all16 old output bytes and complete bills.',
    'All canonical perturbations preserve their own base first8 rows; repeat output/bill/trace is exact.','',
    '| Rank | Head | Original asymmetry | Canonical asymmetry | Original curvature | Canonical curvature | Pass |',
    '| --- | --- | --- | --- | --- | --- | --- |']
for c in result['cells']:
    lines.append(f"|{c['rank']}|{c['coordinate']}|{c['original_asymmetry']:.9g}|{c['forward_vs_centered_relative']:.9g}|{c['original_curvature']:.9g}|{c['double_curvature_relative']:.9g}|{c['passed']}|")
lines+=['',f"Overall fixed continuation gate passed: {result['passed']}.",
    'Each cell requires spacing SNR>=20 and both response ratios<=.10.',
    'The saved trace comparison distinguishes count/membership changes from pure order changes.',
    f'Across all six rank38 probes, selected indices and masks equal the canonical base: {rank38_selected_indices_fixed}.',
    'Input scores do change. Both rank38 response cells still fail, so differing selector sets/orders',
    'at the sampled endpoints are not required for the observed instability. This does not identify',
    'another internal cause, establish fixed branches between endpoints, or qualify a precision change.',
    'It is a finite intervention on these reused cases, not a complete derivative, accuracy result,',
    'universal floating-point bound or proof about every source of response nonlinearity.',
    'No CHECK targets, model fit, native accuracy screen, client qualification, package or upload occurred.',
    ('A separately declared fit/accuracy design is licensed; this diagnostic alone does not license deployment.'
     if result['passed'] else 'The fixed ordering diagnostic is closed; no epsilon/coordinate/scope/precision rescue follows.'),'',
    f"All{len(guards)} guarded groups retired;{len(calls)} full native calls;zero added swap;",
    f"peak sampled group RSS{resources['peak_group_rss_bytes']} bytes; no new samples,datasets or GPU.",
    'The approximately1e-9 goal remains active and unachieved.','']
with (HERE/'RESULT.md').open('x') as f:f.write('\n'.join(lines))
files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
record=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outcome=outcome,
    response_cells=result['cells'],rank38_selected_indices_fixed=rank38_selected_indices_fixed,
    resources=resources,uploaded=False,goal_achieved=False,
    artifact_pins=[pin(p) for p in files],external_pins=[pin(PARENT),pin(D/'official_best_331684.json'),
        pin(OLD/'closeout_after_replay_v1.json')])
for p in record['artifact_pins']+record['external_pins']:verify(p)
write(HERE/'closeout.json',record)
print(dict(pins=len(files),outcome=outcome,resources=resources),flush=True)

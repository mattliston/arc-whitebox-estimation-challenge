from common import *
from datetime import datetime,timezone
seal_check()
for p in read(HERE/'repair_sources_sealed.json')['pins']:verify(p)
r=read(HERE/'audit_result.json');v=read(HERE/'replay_result.json');end=read(HERE/'completed.json')
assert r['completed'] and v['completed'] and end['completed'] and not r['opportunity_gate_passed']
assert not r['native_fixtures'] and not list(HERE.glob('*_plan.json'))
assert r['certificates']=={'signed_executable_0':162,'signed_executable_1':324,'signed_executable_2':162}
assert v['calls']==590 and v['eligible_calls']==192 and v['unique_shapes']==129 and v['lifetime_intervals']==90633
guards=[read(HERE/(name+'_watch.json')) for name in ['audit','audit_v2','replay']]
assert [g['exit_code'] for g in guards]==[1,0,0]
assert all(g['whole_group_retired'] and g['max_swap_growth_bytes']==0 and g['error'] is None for g in guards)
resources=dict(guarded_groups=len(guards),groups=[g['pid'] for g in guards],failed_initial_ast_replay_preserved=True,
    whole_groups_retired=True,controller_terminal_sessions=[16558],new_estimator_calls=0,new_native_kernel_calls=0,
    measured_production_flops=0,research_exact_integer_and_fraction_operations_unmetered=True,
    peak_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_reported_self_rss_bytes=max(r['peak_rss_bytes'],v['peak_rss_bytes']),
    initial_failed_worker_self_rss_unavailable=True,max_added_swap_bytes=0,
    summed_guarded_wall_s=sum(g['elapsed_s'] for g in guards),
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    no_weights_truths_fit_samples_gpu_downloads_or_upload=True)
assert resources['artifact_bytes_before_closeout']<20_000_000
write(HERE/'resources.json',resources)
summary=dict(completed=True,verdict='Closed at fixed 3% component cost opportunity gate; no numerical qualification.',
    networks=[{k:n[k] for k in ('rank','complete_cold_flops','saved','saving_fraction','new_total','selected_count','passes')} for n in r['networks']],
    full_census_calls=590,eligible_calls=192,unique_shapes=129,placement_intervals=90633,
    exact_rational_basis_products=648,old_representative_prices_and_memory_reproduced=True,
    independent_arithmetic_memory_and_census_replay_passed=True,native_runtime_or_accuracy_qualified=False,
    target_score_measured=False,candidate=None,official_changed=False,goal_achieved=False,
    official_adjusted=3.5844144212297072e-9,official_raw=1.949715441540434e-8,
    milestone_4e9_banked=True,milestones_3e9_2e9_approximately_1e9_unmet=True,
    upload_review_question_still_pending=True,resources=resources)
write(HERE/'summary.json',summary)
report='''# Normalized Smirnov schedule: concrete component envelope is too small

Padding once to even terminal tiles makes this fixed three-Smirnov plus
one-Winograd program algebraically applicable to every selected shape.
After cold bindings, copies and complete component workspace are counted,
it fails the prospective 3% saving gate on both current-bank censuses.

| Network | Saved F in favorable census replacement | Fraction of complete bank bill | Cheaper eligible calls |
| --- | ---: | ---: | ---: |
| 35 | 4,816,968,272 | 1.1650703722% | 16 |
| 100 | 7,335,251,652 | 1.7621599966% | 20 |

The corresponding symbolic new totals are 408,631,728,637 and
408,929,540,760 F, against original complete cold bills 413,448,696,909
and 416,264,792,412 F. They are census calculations, not newly measured
estimator bills or score forecasts. Every changed product was priced cold;
no initialization saving or reused numerical result was taken for free.
Ineligible, over-workspace or more expensive products keep their old bill.
Replacing an entire old call remains favorable: changed cache ownership,
dispatcher costs and autonomous states have not been qualified.

All 590 calls were reconciled. The fixed single-long-axis rule admits192
calls with129 unique shapes. Three existing orientations pass648 exact
rational basis products. Closed-form stage and leaf arithmetic agrees at
every eligible shape. The independent replay checks90,633 allocation
lifetimes, nonoverlapping concrete addresses, high-water marks, copies,
all cold view charges and complete totals. The1,168,900,096-byte cap
includes the arena, padded inputs and full padded output; it is not a
whole-estimator memory certificate. The preserved old representative also
replays its exact cold/warm7,258,335,196/6,611,249,012 F, arena1,115,801,600 B
and padding46,283,536 B.

Preparation first encountered a wrong report filename before writing a
seal or starting a worker; prepare_v2 records both corrected paths and an
exact source inverse. The first guarded audit then caught a separate AST
loader error: extraction omitted Block's dataclass decorator. It stopped
during the old representative replay, before any new schedule price.
Additive loader_v2/audit_v2 preserve decorators; exact textual inverses,
original sources/logs/failed guard, and the repair seal are retained.
The schedule, source circuits, 3% gate and all resource limits are unchanged.

This closes one fixed component schedule. It does not strengthen the old
catalog's negative into a bound on every circuit, streaming order or
accuracy/cost composition. Old Smirnov native timing failures remain
failures; this metadata result neither repairs nor repeats them. No native
fixture, estimator, target score, fit, panel, package or upload followed.
The reviewed L3L4/L3-cubed shell also already has adverse executed evidence
and a missing-fifth-cumulant cancellation issue; it was not started anew.

Three guarded groups retired, including the initial failed replay;
controller16558 is terminal. Peak sampled and maximum reported RSS are
41,435,136 B; the failed worker has no separate self-RSS report. All guards
record zero added swap. All arithmetic here is research metadata or exact
algebra, not metered production inference. resources.json records wall time
and retained bytes. No network weights or truths were decoded and no new
samples, GPU work or downloads occurred.

Official331684 remains adjusted3.5844144212297072e-9 and
raw1.949715441540434e-8,100/100 complete,zero failures. The4e-9 milestone
is banked;3e-9,2e-9 and approximately1e-9 remain unmet and the goal active.
The current organizer announcement confirms Phase2 submissions close
17 October2026 at23:59UTC; the older September schedule is superseded:
https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197

The separately qualified covariance/readout release remains unchanged.
Automatic approval review rejected uploading that private archive to
AIcrowd because trusted authorization for the exact payload and destination
was not visible. The existing exact-upload question remains unanswered.
No retry, bypass, new submission receipt or upload occurred.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
externals={p['path']:p for seal in ['sources_sealed.json','repair_sources_sealed.json'] for p in read(HERE/seal)['pins'] if not Path(p['path']).is_relative_to(HERE)}
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='closeout.json']
for p in artifacts+list(externals.values()):verify(p)
write(HERE/'closeout.json',dict(completed=True,utc=datetime.now(timezone.utc).isoformat(),opportunity_gate_passed=False,
    independent_replay_passed=True,native_qualified=False,candidate=None,official_changed=False,goal_achieved=False,
    artifacts=artifacts,external_pins=list(externals.values()),resources=resources,summary=pin(HERE/'summary.json'),report=pin(HERE/'RESULT.md')))
print(dict(closeout=pin(HERE/'closeout.json'),artifacts=len(artifacts),external_pins=len(externals),resources=resources))

"""Preserve the qualified actual-read diagnostic and its strict scope."""
from datetime import datetime,timezone
import ast,math,os,shutil,subprocess
from common import *
seal_check()
for name in ('repair_sources_sealed.json','finalizer_sources_sealed.json'):
    for p in read(HERE/name)['pins']:verify(p)
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
repair=read(HERE/'response_repair.json')
for old_name,changes in repair['changes'].items():
    new_name=Path(old_name).stem+'_v2.py';t=(HERE/new_name).read_text()
    for old,new in reversed(changes):
        assert t.count(new)==1;t=t.replace(new,old,1)
    assert t==(HERE/old_name).read_text(),old_name
result=read(HERE/'result.json');completed=read(HERE/'completed.json');unit=read(HERE/'unit_v2.json')
assert result['completed'] and result['diagnostic_qualified'] and result['both_networks_flagged']
assert completed['completed'] and unit['passed'] and len(unit['fixtures'])==6
assert unit['runtime_pair_binding_confirmed'] and unit['formal_fp64_series_response_only']
assert unit['fp32_consumer_gap_norm']<=unit['fp32_consumer_allowance']
assert not result['targets_read'] and not result['candidate_created']
assert not (HERE/'unit.json').exists()
guards=[done(n) for n in completed['guards']]
failed=read(HERE/'unit_watch.json')
assert failed['exit_code']==1 and not failed['completed'] and failed['whole_group_retired'] and failed['max_swap_growth_bytes']==0
guards.append(failed);assert len(guards)==5 and len(list(HERE.glob('*_watch.json')))==5
records=[];aggregate=[]
for row in result['rows']:
    rank=row['rank'];record=read(HERE/f'rank{rank:04d}/result.json');records.append(record)
    for key in ('source','prediction','trace'):verify(record[key])
    assert record['all16_bytes_and_complete_bill_exact'] and record['observers_restored'] and record['numeric_cache_retired']
    trace=read(record['trace']['path']);calls=trace['calls']
    assert [x['layer'] for x in calls]==list(range(1,15))
    assert all(x['local_covariance_is_formal_fp64_series_response'] for x in calls)
    # A differently grouped scalar replay checks the saved pooled ratios.
    er=math.hypot(*(x['effective_error_norm'] for x in calls))/math.hypot(*(x['full_K21_offdiag_norm'] for x in calls))
    cr=math.hypot(*(x['local_covariance_error_norm'] for x in calls))/math.hypot(*(x['original_covariance_offdiag_norm'] for x in calls))
    assert math.isclose(er,row['pooled_effective_error_over_full_K21'],rel_tol=2e-15)
    assert math.isclose(cr,row['pooled_local_covariance_relative'],rel_tol=2e-15)
    assert row['flagged_layers']==[x['layer'] for x in calls if x['effective_error_over_full_K21']>3e-5]
    aggregate.append(dict(rank=rank,pooled_read=er,pooled_local_covariance=cr,scalar_replay_passed=True))
write(HERE/'aggregate_replay.json',dict(passed=True,rows=aggregate,metadata_only=True,no_new_arrays_or_targets=True))
groups=sorted(g['pid'] for g in guards);active=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)==4 and (int(t[0]) in groups or int(t[1]) in groups or
        (int(t[0])!=os.getpid() and any(m in t[3] for m in ('openblas_runtime_env_v1/bin/python',str(D/'run_guarded.py'),str(HERE/'run.py'),str(HERE/'run_v2.py'),str(HERE/'admit_repaired.py'))))):active.append(line)
assert not active,active
prior=read(D/'continuation_checkpoint_v89.json')
for key in ('study','workers','verification','official_bank','pending_release','pending_candidate','pending_archive','rejection'):verify(prior[key])
for name in ('submission_intent.json','submission.json','submission_attempt.json'):
    assert not (D/'frozen_covariance_readout_composition_v1'/name).exists()
resources=dict(guarded_groups=5,groups=groups,all_groups_retired=True,controller_terminal_sessions=[93145,67613],
    successful_guards=completed['guards'],preserved_failed_guards=['unit'],
    full_native_calls=2,full_native_flops=result['full_native_flops'],observed_complete_reads=28,
    peak_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_reported_self_rss_bytes=max(x['peak_rss_bytes'] for x in records+[unit,result]),
    failed_unit_self_rss_unavailable=True,max_swap_growth_bytes=max(g['max_swap_growth_bytes'] for g in guards),
    total_guarded_wall_s=math.fsum(g['elapsed_s'] for g in guards),
    max_snapshot_bytes=max(r['max_snapshot_bytes'] for r in result['rows']),
    max_estimated_reference_workspace_bytes=max(r['max_reference_workspace_upper_bytes'] for r in result['rows']),
    reference_arithmetic_unmetered_research_only=True,diagnostic_wall_is_not_production_wall=True,
    targets_read=False,new_Monte_Carlo_truth_samples=0,new_fit_steps=0,new_downloads=0,GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free,
    swap=subprocess.check_output(['/usr/sbin/sysctl','-n','vm.swapusage'],text=True).strip())
assert resources['max_swap_growth_bytes']==0 and resources['artifact_bytes_before_closeout']<10_000_000
assert resources['max_snapshot_bytes']<=250_000_000 and resources['max_estimated_reference_workspace_bytes']<=400_000_000
write(HERE/'resources.json',resources)
report='''# Actual signed-pair read roundoff is measurable and locally small

Both unchanged bank runs trigger the prospective numerical flag: some
effective read discrepancies exceed 3e-5 of the complete off-diagonal K21
norm. They are much smaller than the prior synthetic fixture's percent-scale
error. Their pooled formal covariance responses are about one part per
million of the original off-diagonal covariance. This is numerical evidence
on current states, not a measured improvement in final-layer accuracy.

| Network | Maximum error / full K21 | Pooled error / full K21 | Pooled formal covariance response | Maximum formal covariance response |
| --- | ---: | ---: | ---: | ---: |
'''
for r in result['rows']:
    report+=f"| {r['rank']} | {r['maximum_effective_error_over_full_K21']:.12g} | {r['pooled_effective_error_over_full_K21']:.12g} | {r['pooled_local_covariance_relative']:.12g} | {r['maximum_local_covariance_relative']:.12g} |\n"
report+='''
Each network flags layers 2 and 4 through 14; layers 1 and 3 remain below
the fixed threshold. All 14 complete reads per network were observed,
including young-only, materialized-old and lazy-old paths. Relative to the
CP read itself, the maximum effective discrepancies are 7.85736010596e-5
and 7.51850321734e-5. The reference preserves the declared old factorization
and the supplied stored operands; it does not substitute rounded reconstructed
right factors, change a tensor, or isolate matrix-kernel error from the other
FP32 arithmetic within a read. Diagonal entries are excluded because the
actual consumer overwrites them through its independent FP64 scalar path.

The FP64 reference uses streamed 64-row products. A second reference sums
257-column contractions before the same old maps. Maximum relative reference
disagreements are 2.63105946172e-15 and 2.41320554950e-15, at most
3.71209273476e-11 of the measured read-error norms. Six tiny fixtures also
agree with independent reconstructed-right scalar math.fsum evaluation.
They cover all ownership modes, zero coefficients, no young columns, empty
width, strides, input preservation, exact repetition and snapshot lifetime
after caller mutation. Actual FNP parent-read output bytes and bills match
with and without the observer, and the observer returns the original object.

The first unit failed its ancillary covariance-affinity check. Source review
had selected an obsolete FunctionDef named nonlinear, overlooking subsequent
module assignments. The effective binding is _nonlinear_pair_batch, with
edge-correlation terms through degree2 and FP32 series arithmetic. The old
degree0 response was incorrect. RESPONSE_CORRECTION.md documents the exact
cause and additive repair; original files and the failure are preserved.
Textual inverse checks recover all four original scripts. Primary read
references, fixtures, native controls and the 3e-5 numerical flag did not
change, and no full-network call occurred before repair.

The corrected local covariance response uses the actual series' formal FP64
algebra with original scalar inputs, including correlation degrees0..2.
Its independent unbatched FP64 finite-difference certificate has maximum
absolute error 1.27393755267e-18, below the original 1e-12 algebra tolerance.
The actual rounded FP32 consumer's unit response differs by norm
2.19303139269e-9, below its separately declared 2.15110744451e-8 rounding
allowance. The recorded network response is the formal series quantity;
it is not an exact derivative of FP32 rounding or the learned autonomous
estimator. Scalar means and marginal diagonals are fixed in that calculation.

All 16 predictions and complete cold bills equal the original bank controls
exactly: 413448696909 and 416264792412 FLOPs. Original functions each run
once in the observed inference path; NumPy reference arithmetic is research
instrumentation, never feeds back, and does not change the paid bank bill.
It is not a free proposed production operation. Both wrappers restore,
both numeric caches retire, and no wide trace arrays were saved. An independent
scalar aggregate replay verifies the pooled ratios and layer flags.

The positive numerical flag alone does not supply an economically useful
candidate. It neither measures a final MSE gain nor bounds accumulated
autonomous effects; downstream states, basis construction, learned heads
and rounding can change under an intervention. No next candidate, precision
or kernel change is selected from this diagnostic. The failed polarized
read and the previous negative FP64 shared-sketch endpoint remain closed.
The receiver-direction result reviewed this turn was already executed and
yielded only 0.5570% old-host TRAIN raw improvement, failing its own gate;
it is not an unrun raw-error lead.
'''
report+=f'''
Five guarded groups retired; controllers 93145 and 67613 are terminal.
Two original native calls account for {resources['full_native_flops']:,}
measured FLOPs, excluding offline reference arithmetic. Peak sampled group
RSS was {resources['peak_sampled_group_rss_bytes']:,} bytes and maximum
reported process RSS {resources['peak_reported_self_rss_bytes']:,} bytes.
All guards recorded zero added swap. Maximum snapshot size was
{resources['max_snapshot_bytes']:,} bytes, estimated reference workspace
{resources['max_estimated_reference_workspace_bytes']:,} bytes. About
{resources['artifact_bytes_before_closeout']/1e6:.3f} MB was preserved before
closeout. No truths, fitting, new Monte Carlo samples, GPU work or downloads.

Official 331684 remains adjusted 3.5844144212297072e-9 and raw
1.949715441540434e-8, 100/100 complete with zero failures. The 4e-9 milestone
is banked; 3e-9, 2e-9 and approximately1e-9 remain unmet. The full Phase2
goal remains active. The separate qualified covariance/readout archive is
unchanged. Automatic approval review rejected its upload because it could
not verify trusted authorization for that exact private archive and AIcrowd
destination. The existing exact-upload question remains unanswered; no retry,
bypass, submission receipt or upload occurred in this study.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for p in external:verify(p)
write(HERE/'summary.json',dict(completed=True,diagnostic_qualified=True,both_networks_flagged=True,
    candidate=None,official_changed=False,goal_achieved=False,resources=resources,
    report=pin(HERE/'RESULT.md'),result=pin(HERE/'result.json'),aggregate_replay=pin(HERE/'aggregate_replay.json'),
    pending_archive=prior['pending_archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False))
write(HERE/'closeout.json',dict(completed=True,diagnostic_qualified=True,both_networks_flagged=True,
    created_at_utc=datetime.now(timezone.utc).isoformat(),candidate=None,official_changed=False,goal_achieved=False,
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,
    resources=resources,pending_archive=prior['pending_archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False))
print(dict(completed=True,diagnostic_qualified=True,both_networks_flagged=True,resources=resources,
    artifacts=len(read(HERE/'closeout.json')['artifacts'])),flush=True)

"""Preserve the failed fixed joint head; no native candidate is admitted."""
from datetime import datetime, timezone
import ast
import os
import shutil
from common import *

SEALS = ('sources_sealed.json', 'fit_sources_sealed.json',
         'replay_sources_sealed.json', 'closeout_source_sealed.json')
for name in SEALS:
    verify(name)
result = read(HERE / 'information_result.json')
replay = read(HERE / 'replay_result.json')
training = read(HERE / 'training_result.json')
unit = read(HERE / 'unit_result.json')
frozen = read(HERE / 'model_frozen.json')
assert result['completed'] and not result['passed']
assert replay['passed'] and not replay['information_gate_passed'] and replay['no_refit']
assert [r['passed'] for r in result['records']] == [False, False]
assert [r['passed'] for r in replay['records']] == [False, False]
assert training['completed'] and training['updates'] == 1344 and len(training['epochs']) == 32
assert frozen['final_epoch'] == 32 and frozen['reused_validation']
assert not frozen['new_validation_fourth_fields_decoded']
assert unit['passed'] and unit['conservative_seven_layer_flops'] == 9_403_993_088
assert read(HERE / 'fit_completed.json')['completed']
for key in ('model', 'normalizers', 'training'):
    check(frozen[key])
for name in ('candidate.py', 'build.json', 'native_sources_sealed.json',
             'submission_intent.json', 'submission.json', 'submission_attempt.json'):
    assert not (HERE / name).exists(), name
ast.parse((HERE / 'build.py').read_text())

targets = [read(HERE / f'targets_mini{o:04d}.json') for o in TRAIN + VALID]
assert all(t['passed'] and not t['public1B_labels_decoded'] and not t['late_MC_noise_removed'] for t in targets)
checks = [v for t in targets for v in t['checks'].values()]
assert all(v['passed'] for v in checks)
for t in targets:
    check(t['source'])
    check(t['previous_qualification'])
    for record in t['records']:
        check(record['target'])
frozen_time = datetime.fromisoformat(frozen['utc']).timestamp()
for ordinal in VALID:
    assert (HERE / f'targets_mini{ordinal:04d}.json').stat().st_mtime > frozen_time

guards = [read(p) for p in sorted(HERE.glob('*_watch.json'))]
assert len(guards) == 12, len(guards)
assert all(g['completed'] and g['exit_code'] == 0 and g['error'] is None
           and g['whole_group_retired'] and g['max_swap_growth_bytes'] == 0 for g in guards)
groups = sorted(g['pid'] for g in guards)
rows = []
for line in subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True).splitlines():
    a = line.strip().split(None, 3)
    if len(a) == 4:
        rows.append(dict(pid=int(a[0]), pgid=int(a[1]), rss_kib=int(a[2]), command=a[3]))
remaining = [r for r in rows if r['pid'] in groups or r['pgid'] in groups]
assert not remaining, remaining
resource_floor()
now = datetime.now(timezone.utc).isoformat()
resources = dict(
    utc=now, guard_count=len(guards), completed_guards=len(guards), failed_guards=0,
    groups=groups, all_groups_retired=True, remaining_owned_rows=[], nice=10, blas_threads=1,
    max_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    max_successful_process_rss_bytes=max(v['peak_rss_bytes'] for v in [unit, training, result, replay] + targets),
    max_added_swap_bytes=0, total_guard_wall_s=sum(g['elapsed_s'] for g in guards),
    full_native_parent_calls=0, full_native_changed_calls=0, offline_training_not_flop_metered=True,
    new_download_bytes=0, artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free,
    system_swap=subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True).strip())
write(HERE / 'resources.json', resources)
qual = dict(
    maximum_gaussian_fourth_standardized_rms=max(v['standardized_rms'] for v in checks if 'standardized_rms' in v),
    maximum_gaussian_fourth_standardized_entry=max(v['standardized_max'] for v in checks if 'standardized_max' in v),
    maximum_raw_diagonal_float32_ulps=max(v['max_raw_float32_ulps'] for v in checks if 'max_raw_float32_ulps' in v),
    maximum_fourth_rounding_bound_to_missing_rms=max(v['bound_residual_rms_ratio'] for v in checks if 'bound_residual_rms_ratio' in v),
    maximum_replayed_target_relative_error=max(v for r in replay['records'] for l in r['layers'] for v in l['target_relative_errors']),
    maximum_replayed_layer_loss_relative_error=max(l['loss_relative_error'] for r in replay['records'] for l in r['layers']),
    maximum_replayed_ratio_relative_error=max(r['relative'] for r in replay['records']))
summary = dict(
    completed=True, information_gate_passed=False,
    validation=[{k: r[k] for k in ('ordinal', 'ratios', 'passed')} for r in result['records']],
    metric_order=result['metric_order'], joint_gates=result['joint_gates'], reused_validation=True,
    final_training=training['final_training'], model=frozen['model'], normalizers=frozen['normalizers'],
    component_flops=unit['conservative_seven_layer_flops'], full_candidate_created=False,
    native_adapter_draft='build.py is preserved preparatory source only; it was never sealed or executed.',
    independent_replay_passed=True, numerical_qualification=qual, resources=resources,
    official_changed=False, goal_achieved=False, new_download_bytes=0, public1B_labels_decoded=False)
write(HERE / 'summary.json', summary)
table = '\n'.join('| {} | {} |'.format(r['ordinal'], ' | '.join(f'{v:.9f}' for v in r['ratios'])) for r in result['records'])
report = f'''# Joint incoming third/fourth head fails its fixed information gate

The single model finished its prescribed 32 epochs and 1,344 updates. Both
validation networks fail the next-layer mean response gate. The fixed
experiment is closed, with no extra epochs, model selection, amplitude
change, native candidate, final-score evaluation, package or submission.

| MINI ordinal | Weighted T21 | Weighted T31 | Weighted T22 | Joint covariance | Joint next mean |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

Each network had to attain joint covariance AND next-mean ratios at most
0.90. The three plane ratios are auxiliary reports, not acceptance gates.
Ratios use frozen TRAIN per-layer energy normalizers. They are squared
errors in real-arithmetic responses on unchanged parent trajectories;
they are not actual final-layer MSE or FLOP-adjusted scores. Both covariance
gates pass, while next-mean ratios 0.917358747 and 0.931169619 fail. The
independent binomial centering and full matrix congruence replay preserves
the verdict. Its maximum target, layer-loss and aggregate-ratio relative
differences are {qual['maximum_replayed_target_relative_error']:.12g},
{qual['maximum_replayed_layer_loss_relative_error']:.12g} and
{qual['maximum_replayed_ratio_relative_error']:.12g}, respectively.

This is a new three-output model trained from scratch. It jointly predicts
normalized offdiagonal K21, K31 and K22 corrections at layers 8 through 14.
Its 140 causal node features include previous-parent H64 transported by W
and W squared; pair inputs are current R, S and S transpose. Only predicted
K22 is symmetrized. The exact directed response uses the same rho_ij for
both orientations and was certified on asymmetric and active-floor fixtures
before fitting. The earlier single-third model and its closed verdict are
unchanged. No complete fourth tensor or exact all-distinct emission was
implemented or tested by this experiment.

TRAIN MINI ordinals were 57, 68, 45, 59, 3 and 12. Validation ordinals 14
and 71 had already been used by the previous experiment and are explicitly
reused validation, not pristine holdouts. Their new fourth-moment fields
were admitted only after this final model was hash-frozen. All eight
existing 100M public payloads were reused without downloads or sampling.
Newly decoded fields were pre_m4, pre_M31 and pre_M22, alongside previously
admitted preactivation moments. No 1B target field was decoded.

All eight source checks passed. Maximum Gaussian first-layer standardized
fourth RMS was {qual['maximum_gaussian_fourth_standardized_rms']:.12g}, maximum entry
{qual['maximum_gaussian_fourth_standardized_entry']:.12g}; limits were 1.5 and 12.
Maximum raw fourth diagonal discrepancy was
{qual['maximum_raw_diagonal_float32_ulps']:.12g} FP32 ULPs (limit 2).
The maximum late-layer FP32 rounding bound was
{qual['maximum_fourth_rounding_bound_to_missing_rms']:.12g} of the missing-plane RMS
(limit 0.05). These checks do not remove late-layer Monte Carlo noise or
shared numerical bias. Source credits and pinned revisions are recorded
in DATA_PROVENANCE.md and the source seals.

Centering, Wick influence variance, directed source response, analytic
gradient, FNP equation, permutation, zero-output and repeat checks passed.
One complete standalone head and physical correction with cold array
parameter ingress measured 1,343,427,584 weighted FLOPs; the conservative
seven-layer sum was 9,403,993,088, below the 15-billion component cap. There
is no full changed-estimator bill. The unexecuted build.py adapter draft
is preserved as preparatory work; no native source seal or candidate exists.

All {len(guards)} guarded groups retired successfully. Maximum sampled group
and process peak RSS were {resources['max_sampled_group_rss_bytes']:,} and
{resources['max_successful_process_rss_bytes']:,} bytes, respectively. Added
swap was zero. New artifacts before closeout total
{resources['artifact_bytes_before_closeout']:,} bytes, within the 2.5GB cap;
free disk was {resources['disk_free_bytes']/1e9:.3f}GB. No new parent or changed
native trajectory was run. This is a scoped negative for this model,
features, fixed loss and optimizer, not an impossibility result for joint
moment prediction.

Official submission 331684 remains adjusted 3.5844144212297072e-9 and raw
1.949715441540434e-8, with 100/100 cases and zero failures. The 4e-9 milestone
is banked; 3e-9, 2e-9 and approximately 1e-9 remain unmet. The goal is active.

Automatic approval review previously rejected uploading the separate
qualified covariance/readout archive because it could not verify trusted
authorization for the exact private payload and AIcrowd destination. The
existing approval question is unanswered; this experiment made no retry.
'''
with (HERE / 'RESULT.md').open('x') as f:
    f.write(report)
external = {}
for seal in SEALS:
    for p in read(HERE / seal)['pins']:
        if not Path(p['path']).is_relative_to(HERE):
            check(p)
            external[p['path']] = p
for t in targets:
    for p in (t['source'], t['previous_qualification']):
        if not Path(p['path']).is_relative_to(HERE):
            external[p['path']] = p
artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
assert sum(p['bytes'] for p in artifacts) < 2_500_000_000
write(HERE / 'closeout.json', dict(
    completed=True, utc=now, information_gate_passed=False, native_qualified=False, candidate=None,
    independent_replay_passed=True, official_changed=False, goal_achieved=False,
    summary=pin(HERE / 'summary.json'), model=frozen['model'], resources=resources,
    artifacts=artifacts, external_pins=list(external.values())))
print(dict(completed=True, closeout=pin(HERE / 'closeout.json'),
           validation=summary['validation'], resources=resources, numerical_qualification=qual), flush=True)

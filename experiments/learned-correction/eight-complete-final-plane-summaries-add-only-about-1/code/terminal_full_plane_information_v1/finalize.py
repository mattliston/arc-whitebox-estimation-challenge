"""Preserve and verify the completed fixed information screen; no numerical work."""
from datetime import datetime, timezone
import os
import shutil
import subprocess
from common import *

seal_check()
seal_check('features_sealed.json')
seal_check('finalize_sources_sealed.json')
completed = read(HERE / 'completed.json')
score = read(HERE / 'result.json')
features = read(HERE / 'features_result.json')
unit = read(HERE / 'unit.json')
audit = read(HERE / 'audit_result.json')
inputs = read(HERE / 'inputs.json')
assert completed['completed'] and not completed['passed'] and completed['audit_passed']
assert not score['passed'] and score['oracle_not_deployable']
assert score['extra_inference_cost_assumed'] == 0 and score['new_native_calls'] == 0
assert not score['held_out'] and not score['official_score']
assert features['passed'] and features['repeat_bytes_exact'] and features['targets_not_decoded']
assert features['new_native_calls'] == 0 and not features['production_cost_measured']
assert unit['passed'] and len(unit['records']) == 6
assert audit['passed'] and audit['gate_decisions_reproduced'] and audit['full_rank_QR_replay']
assert audit['all_served_bytes_exact'] and len(audit['checks']) == 4
assert features['rows'][0]['array_digests'] == features['rows'][2]['array_digests']
check(score['payload'])
check(audit['score'])
for p in completed['guards']:
    check(p)
for row in inputs['rows']:
    for key in ['capture', 'weights', 'parent_prediction', 'truth']:
        check(row[key])
for row in features['rows']:
    check(row['file'])
    check(row['original_capture'])
    assert row['input_preserved']
    assert row['diagonal_relative'] <= 1e-3 and row['variance_relative'] <= 1e-3
    assert row['slice_relative'] <= 2e-10 and row['covariance_slice_relative'] <= 2e-10

compact_rows = []
for row in score['rows']:
    assert not row['passed']
    assert row['arms']['affine']['rank'] == row['arms']['affine']['columns'] == 65
    assert row['arms']['extended']['rank'] == row['arms']['extended']['columns'] == 73
    assert all(v['normal_residual'] <= 1e-8 for v in row['arms'].values())
    # Both controls fail in both renderings; preserve the exact declared gates.
    assert all(row['ratios'][k + '_parent'] > .85 and row['ratios'][k + '_affine'] > .90
               for k in ['ideal', 'served'])
    compact_rows.append(dict(rank=row['rank'], parent_raw=row['parent_raw'],
        parent_cold_flops=row['parent_cold_flops'], multiplier=row['multiplier'],
        ratios=row['ratios'], passed=row['passed'],
        arm_raw={name: {k: arm[k] for k in ['ideal_raw', 'served_raw']}
                 for name, arm in row['arms'].items()}))
assert [r['rank'] for r in compact_rows] == [35, 100]
assert [r['parent_cold_flops'] for r in compact_rows] == [413448696909, 416264792412]

names = ['unit', 'features', 'oracle', 'audit']
guards = [done(n) for n in names]
groups = sorted(g['pid'] for g in guards)
live = []
workers = []
markers = [str(RUNTIME), str(D / 'run_guarded.py'), str(HERE / 'run.py')]
for line in subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True).splitlines():
    row = line.strip().split(None, 3)
    if len(row) != 4:
        continue
    if int(row[0]) in groups or int(row[1]) in groups:
        live.append(line)
    if int(row[0]) != os.getpid() and any(m in row[3] for m in markers):
        workers.append(line)
assert not live and not workers, (live, workers)

bank = read(D / 'official_best_331684.json')
assert bank['submission_id'] == 331684 and bank['completed'] == 100 and bank['failed'] == 0
assert bank['adjusted'] == 3.5844144212297072e-9
bank_pins = [v for v in bank.values() if isinstance(v, dict) and {'path', 'sha256', 'bytes'} <= v.keys()]
assert len(bank_pins) == 5
for p in bank_pins:
    check(p)
previous = read(D / 'continuation_checkpoint_v74.json')
check(previous['study'])
check(previous['official_bank'])

artifact_bytes = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert artifact_bytes < 15_000_000
resources = dict(guarded_groups=len(guards), all_groups_retired=True,
    controller_terminal_session=56034, full_native_calls=0, full_native_flops=0,
    reused_native_capture_files=2,
    reused_capture_bytes=sum(r['capture']['bytes'] for r in inputs['rows']),
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max(r['peak_rss_bytes'] for r in [unit, features, score, audit]),
    max_swap_growth_bytes=0, reference_networks_decoded=2,
    same_network_oracle_projections=4, independent_QR_replays=4, cross_network_fits=0,
    new_samples=0, new_downloads=0, GPU_calls=0, production_cost_measured=False,
    artifact_bytes_before_closeout=artifact_bytes, disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE / 'resources.json', resources)
checks = dict(unit_cases=len(unit['records']),
    unit_max_error=max(v for r in unit['records'] for v in r['errors'].values()),
    formal_native_diagonal_max_relative=max(r['diagonal_relative'] for r in features['rows']),
    variance_diagonal_max_relative=max(r['variance_relative'] for r in features['rows']),
    independent_plane_slice_max_relative=max(r['slice_relative'] for r in features['rows']),
    independent_covariance_slice_max_relative=max(r['covariance_slice_relative'] for r in features['rows']),
    independent_QR_coefficient_max_relative=max(r['coefficient_relative'] for r in audit['checks']),
    independent_QR_prediction_max_relative=max(r['prediction_relative'] for r in audit['checks']),
    all_served_bytes_exact=True, repeat_feature_bytes_exact=True,
    source_captures_parent_predictions_and_official_pins_unchanged=True)
write(HERE / 'summary.json', dict(completed=True, negative=True, rows=compact_rows,
    extra_inference_cost_assumed=0, oracle_not_deployable=True, held_out=False,
    scope='Eight complete pre15 covariance/third-plane summaries added to the 65-column affine control; fixed linear same-network oracle question only.',
    checks=checks, resources=resources, official_changed=False, goal_achieved=False,
    next_numerical_study=None))

report = '''# Eight complete final-plane summaries: insufficient additional information

The fixed information screen fails on both existing development networks,
even with separate privileged coefficients for each network and zero added
inference cost. The added summaries make a small positive contribution, but
do not satisfy either advancement threshold. No producer, cross-network fit,
candidate or upload is selected from this study.

| Existing 1B network | Ideal / parent | Ideal / affine control | Served / parent | Served / affine control |
| --- | ---: | ---: | ---: | ---: |
'''
for row in compact_rows:
    r = row['ratios']
    report += f"| {row['rank']} | {r['ideal_parent']:.12f} | {r['ideal_affine']:.12f} | {r['served_parent']:.12f} | {r['served_affine']:.12f} |\n"
report += f'''
Each network required both ideal and served ratios <=0.85 against the
unchanged parent and <=0.90 against the matching affine oracle. All eight
comparisons fail. Ideal extra improvement beyond affine is 0.790% / 1.207%;
served extra improvement is 0.878% / 1.345%. The roughly 12% reduction against
the parent includes the privileged refit of its existing 65-column readout.
It cannot be credited to the eight added summaries or called a deployable gain.
These reused references are not held-out validation or official scores.

The eight summaries are row mean, RMS and mean absolute covariance
correlation; row/column norms and sums of the standardized repeated-third
plane; and correlation times alpha. Every retained signed factor, full star,
trace term, row and diagonal is included. The frozen feature definitions are
in PROTOCOL.md. Postactivation covariance summaries are excluded. This study
asks a different question from the previous 65 hidden-direction queries,
while reusing their authenticated exact-331684 captures. No new physical
cumulant source or universal representation limit is established.

Both controls have full rank (65 and 73). Independent QR solves using
separately accumulated column scales and losses reproduce the SVD solutions,
all clamped FP32 prediction bytes and every gate decision. Maximum coefficient
relative difference is {checks['independent_QR_coefficient_max_relative']:.12g};
maximum ideal prediction relative difference is
{checks['independent_QR_prediction_max_relative']:.12g}. Privileged coefficients
remain explicitly labeled ORACLE_DO_NOT_DEPLOY.npz. Ideal predictions are
unclipped FP64; served predictions are nonnegative FP32.

Six small explicit-tensor/indexed-congruence fixtures verify all eight
statistics, including nonsymmetric inputs, no factors, signed cancellation,
zero terms and permutation. Nonzero references use true relative errors;
exact-zero references use absolute errors. Maximum error is
{checks['unit_max_error']:.12g} against 2e-10. Wide independent 8-by-8 plane
and covariance slices agree within {checks['independent_plane_slice_max_relative']:.12g}
and {checks['independent_covariance_slice_max_relative']:.12g}, respectively.
Formal FP64 diagonals differ from native k3 by at most
{checks['formal_native_diagonal_max_relative']:.12g}; covariance diagonals differ
from the original sigma squared by at most {checks['variance_diagonal_max_relative']:.12g}.
Those checks meet their fixed 1e-3 tolerances; they do not assert exact equality
to the native packed arithmetic. Inputs are unchanged and 35/100/35 feature
bytes repeat exactly. Features were sealed before reference decoding.

The optimistic screen retains each parent's independent cold multiplier:
413448696909 F for 35 and 416264792412 F for 100. The old warm capture's
100 bill is not used as its cold cost. Research FP64 feature contraction
has no measured deployment bill or runtime qualification; zero extra cost is
an explicit screening assumption, not a native implementation claim.

All four guarded groups retired; controller 56034 is terminal. Zero new
native estimator calls, new samples, downloads or GPU calls. Two existing
capture files ({resources['reused_capture_bytes']} B) were reused, not copied.
There were four same-network least-squares projections and four independent
QR replays, with zero cross-network fits. Peak sampled group RSS was
{resources['peak_group_rss_bytes']} B, self-reported peak
{resources['peak_self_reported_rss_bytes']} B, with zero added swap. Fast-stage
sampling can understate memory; use self-report or the whole-study maximum.
Artifacts before closeout total about {artifact_bytes / 1e6:.3f} MB.

This closes these eight summaries and the declared linear information
question, without a feature, normalization, rank, SVD or nonlinear rescue.
All earlier negative scopes remain intact. Official submission 331684 is
unchanged at adjusted 3.5844144212297072e-9, 100/100 complete, zero failures.
The 4e-9 crossing is banked; 3e-9, 2e-9 and approximately 1e-9 remain unmet.
The goal remains active.
'''
with (HERE / 'RESULT.md').open('x') as f:
    f.write(report)
external = [p for p in read(HERE / 'sources_sealed.json')['pins']
            if not Path(p['path']).is_relative_to(HERE)]
for r in inputs['rows']:
    external += [r[k] for k in ['capture', 'weights', 'parent_prediction', 'truth']]
external += bank_pins + [previous['study']]
external = list({p['path']: p for p in external}.values())
for p in external:
    check(p)
artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
write(HERE / 'closeout.json', dict(completed=True, negative=True,
    created_at_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts,
    external_pins=external, resources=resources, official_changed=False,
    goal_achieved=False, candidate=None, oracle_not_deployable=True,
    source_audit=pin(HERE / 'SOURCE_REVIEW.md'), next_numerical_study=None))
print(dict(closeout=pin(HERE / 'closeout.json'), artifacts=len(artifacts),
           external=len(external), resources=resources), flush=True)

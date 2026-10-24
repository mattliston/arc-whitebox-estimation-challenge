"""Close the fixed signed-pair read after its numerical qualification failure."""
from datetime import datetime, timezone
import ast, math, os, shutil, subprocess
from common import *

seal_check()
for p in read(HERE/'finalizer_sources_sealed.json')['pins']:
    verify(p)
for p in HERE.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
build = read(HERE/'build.json')
verify(build['candidate']); verify(build['parent'])
assert (HERE/'candidate.py').read_bytes() == PARENT.read_bytes()+b'\n\n'+(HERE/'production.py').read_bytes()
assert build['exact_parent_prefix_inverse'] and build['no_parameter_changes']
repair = read(HERE/'unit_reference_repair.json')
restored = (HERE/'unit_v2.py').read_text()
for old, new in reversed(repair['changes']):
    assert restored.count(new) == 1
    restored = restored.replace(new, old, 1)
assert restored == (HERE/'unit.py').read_text()
assert (HERE/'native_v2.py').read_text().replace("done('unit_v2')", "done('unit')").replace('unit_v2.json', 'unit.json') == (HERE/'native.py').read_text()
zero = read(HERE/'zero_reference_diagnostic.json')
wide = read(HERE/'wide_diagnostic.json')
assert zero['passed'] and zero['actual_exactly_zero'] and zero['independent_scalar_exactly_zero']
assert wide['completed'] and wide['candidate_unchanged'] and wide['gates_unchanged']
assert wide['fp64_pair_identity_relative'] <= 3e-11 and len(wide['small_fixtures']) == 5
assert all(r['inputs_preserved'] and r['repeat_exact'] and r['permutation_passed'] for r in wide['small_fixtures'])
errors = wide['results']
actual = errors['polarized_existing_kernel']['relative']
assert actual == 0.0035854277148522043 and actual > 3e-5
assert actual < errors['original_existing_kernel']['relative']
assert str(actual) in (HERE/'unit_v2.log').read_text()
assert not wide['targets_read'] and wide['full_native_calls'] == 0
for key in ('candidate', 'unit_failure'):
    verify(wide[key])
names = ('unit', 'zero_reference', 'unit_v2', 'wide_diagnostic')
guards = [read(HERE/f'{n}_watch.json') for n in names]
assert len(list(HERE.glob('*_watch.json'))) == 4
for name, g in zip(names, guards):
    expected = 1 if name in ('unit', 'unit_v2') else 0
    assert g['exit_code'] == expected and g['completed'] == (expected == 0)
    assert g['whole_group_retired'] and g['error'] is None and g['max_swap_growth_bytes'] == 0
for name in ('unit.json', 'unit_v2.json', 'disabled', 'enabled', 'cold38', 'completed.json', 'qualification.json', 'score.json'):
    assert not (HERE/name).exists(), name
groups = sorted(g['pid'] for g in guards)
active = []
markers = ('openblas_runtime_env_v1/bin/python', str(D/'run_guarded.py'), str(HERE/'run.py'), str(HERE/'run_v2.py'))
for line in subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True).splitlines():
    t = line.strip().split(None, 3)
    if len(t) == 4 and (int(t[0]) in groups or int(t[1]) in groups or
            (int(t[0]) != os.getpid() and any(m in t[3] for m in markers))):
        active.append(line)
assert not active, active
prior = read(D/'continuation_checkpoint_v88.json')
for key in ('study', 'pending_release', 'pending_candidate', 'pending_archive', 'rejection', 'official_bank'):
    verify(prior[key])
pending = D/'frozen_covariance_readout_composition_v1'
for name in ('submission_intent.json', 'submission.json', 'submission_attempt.json'):
    assert not (pending/name).exists()
resources = dict(
    guarded_groups=4, groups=groups, all_groups_retired=True,
    controller_terminal_sessions=[5738, 89525, 17099, 51028],
    guards=[dict(name=n, exit_code=g['exit_code'], completed=g['completed'], pid=g['pid']) for n, g in zip(names, guards)],
    full_native_calls=0, full_native_flops=0, numerical_work='synthetic unit and diagnostic kernels only',
    peak_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_reported_self_rss_bytes=wide['peak_rss_bytes'], failed_unit_self_rss_unavailable=True,
    max_swap_growth_bytes=0, total_guarded_wall_s=math.fsum(g['elapsed_s'] for g in guards),
    targets_decoded=False, weights_decoded=False, target_and_weight_files_hash_verified_only=True,
    new_fit_steps=0, new_samples=0, new_downloads=0, GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free,
    swap=subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True).strip())
assert resources['artifact_bytes_before_closeout'] < 30_000_000
write(HERE/'resources.json', resources)
result = dict(
    completed=True, status='closed_numerical_qualification_failed', implementation_qualified=False,
    candidate=pin(HERE/'candidate.py'), exact_parent_prefix_inverse=True, no_parameter_changes=True,
    initial_failure='invalid relative reference for mathematically exact cancellation',
    reference_repair_inverse_verified=True, original_failures_preserved=True,
    candidate_fixtures_rng_precision_tolerances_gates_unchanged=True,
    small_fixture_checks_passed=5, fixed_wide_fixture=[512, 2048, 512],
    relative_error_limit=3e-5, actual_relative_error=actual, multiple_of_limit=actual/3e-5,
    wide_numerical_gate_passed=False, wide_improved_over_original=True,
    fp64_pair_identity_relative=wide['fp64_pair_identity_relative'], diagnostics=errors,
    direct_products_are_diagnostics_not_candidates=True,
    materialized_lazy_and_complete_ownership_checks_run=False, malformed_rejection_checks_run=False,
    unit_complete=False, full_native_calls=0, native_qualification=None, score=None,
    targets_decoded=False, weights_decoded=False, parameter_fit=False, official_changed=False,
    goal_achieved=False, resources=resources)
write(HERE/'result.json', result)
report = '''# Fixed signed-pair read fails numerical qualification

The midpoint/difference read using the unchanged matrix kernel fails its
prospective wide-fixture tolerance. Relative error is **0.0035854277148522043**
against **0.00003**, about **119.5 times** the limit. This closes the fixed
candidate before any full-network call or score. The official bank remains
331684: adjusted 3.5844144212297072e-9, raw 1.949715441540434e-8,
100/100 cases complete, zero failures. The 4e-9 milestone is banked;
3e-9, 2e-9 and approximately 1e-9 remain unmet. The full goal stays active.

For each original signed pair, the candidate evaluates the exact real
identity a*(x*x*u-y*y*v) = (h*h+d*d/4)*(a*D)+(2*h*d)*(a*H),
where h=(x+y)/2, d=x-y, H=(u+v)/2 and D=u-v. It packs the same
contracted width and uses the original product function. All new arithmetic
is FP32. The common old basis, lazy map and addition order are retained.
Marginal cubic diagonals remain on their existing FP64 path. Factor
construction, compression, rank, cohort schedule, covariance, heads and
learned parameters are unchanged. The candidate is an exact parent prefix
plus an append-only override; its bytes did not change after qualification
started. No dominant product is removed by this identity.

The first unit attempt exposed an oracle defect: an exactly cancelling
fixture had exactly zero candidate and independent scalar outputs, while
the dense FP64 reference left a 5.679343126843894e-16 roundoff norm.
Dividing by that artifact produced an invalid relative failure. The
preserved diagnostic proves the exact pair cancellation. An additive
unit_v2 handles that declared exact-zero case, bounds the dense roundoff,
and requires exact zero from both candidate and scalar oracle. Reversing
the recorded edits recovers unit.py exactly. The candidate, fixture order,
random stream, precision, all nonzero tolerances and objective gates were
unchanged. Both failed source versions and logs remain preserved.

The repaired unit then failed on the fixed 512x2048 by 2048x512 close-pair
fixture. Five preceding small fixtures passed, including exact cancellation,
zero coefficients, multiple signed cohorts, repetition, permutation and
input preservation. The materialized/lazy/complete ownership checks and
malformed-input rejection checks occur after the wide fixture and were
never reached. There is no successful complete unit receipt.

A bounded diagnostic replayed the same fixture and compared the four
arithmetic paths below. Its FP64 polarization identity agrees with direct
FP64 evaluation to relative 5.478158045314982e-13.

| Synthetic fixture path | Relative error | Measured FLOPs |
| --- | ---: | ---: |
'''
for name, label in (
        ('polarized_existing_kernel', 'Pair rewrite, existing kernel'),
        ('original_existing_kernel', 'Original signed read, existing kernel'),
        ('polarized_direct_matmul_diagnostic_only', 'Pair rewrite, direct matmul diagnostic'),
        ('original_direct_matmul_diagnostic_only', 'Original signed read, direct matmul diagnostic')):
    row = errors[name]
    report += f"| {label} | {row['relative']:.16g} | {row['flops']:,} |\n"
report += '''
The rewrite improves this synthetic error relative to the original path
but still fails its absolute gate. Direct matmul with the same pair helper
greatly reduces the error, identifying a substantial contribution from the
existing fast-kernel arithmetic on this fixture. These are synthetic kernel
measurements, not full-estimator accuracy or cost evidence. Direct products
are diagnostic comparisons only: no replacement candidate, new packing,
precision, layer choice, fit, or gate relaxation was introduced.

No planned disabled/enabled/cold full-network sequence ran; no truth or
weight arrays were decoded. Previously recorded input files were hash
verified only. No target score, panel, confirmation, package or upload was
produced. The prepared native and score scripts remain as unexecuted history.
'''
report += f'''
Four guarded groups retired; controllers 5738, 89525, 17099 and 51028 are
terminal. Peak sampled group RSS was {resources['peak_sampled_group_rss_bytes']:,}
bytes, and the wide diagnostic reported a process high-water mark of
{resources['peak_reported_self_rss_bytes']:,} bytes. Failed unit processes did
not write their own high-water marks; the sampled values are not exact peak
bounds. All four guards recorded zero added swap. Numerical work used
nice10, one BLAS thread and the existing 8GB group limit. About
{resources['artifact_bytes_before_closeout']/1e6:.3f} MB was preserved before
closeout. No new samples, training, GPU work or downloads occurred.

The separate qualified covariance/readout archive remains unchanged.
Automatic approval review rejected its AIcrowd upload because it could not
verify trusted authorization for that exact private payload and destination.
The existing exact-upload approval question remains unanswered. No retry,
bypass, upload process or submission receipt occurred in this study.
'''
with (HERE/'RESULT.md').open('x') as f:
    f.write(report)
external = [p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for p in external:
    verify(p)
write(HERE/'summary.json', dict(
    completed=True, status=result['status'], implementation_qualified=False,
    candidate=result['candidate'], actual_relative_error=actual, relative_error_limit=3e-5,
    full_native_calls=0, targets_decoded=False, official_changed=False, goal_achieved=False,
    resources=resources, report=pin(HERE/'RESULT.md'), result=pin(HERE/'result.json'),
    pending_archive=prior['pending_archive'], upload_approval_still_pending=True, upload_attempted_this_turn=False))
write(HERE/'closeout.json', dict(
    completed=True, status=result['status'], implementation_qualified=False,
    created_at_utc=datetime.now(timezone.utc).isoformat(), candidate=result['candidate'],
    official_changed=False, goal_achieved=False,
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()], external_pins=external,
    resources=resources, pending_archive=prior['pending_archive'],
    upload_approval_still_pending=True, upload_attempted_this_turn=False))
print(dict(completed=True, status=result['status'], resources=resources,
           artifacts=len(read(HERE/'closeout.json')['artifacts'])), flush=True)

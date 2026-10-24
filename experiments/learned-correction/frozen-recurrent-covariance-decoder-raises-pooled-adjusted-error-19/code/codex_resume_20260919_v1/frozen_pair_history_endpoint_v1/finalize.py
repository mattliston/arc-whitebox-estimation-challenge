"""Close the predeclared fixed endpoint; this script only reads receipts and hashes files."""
from datetime import datetime, timezone
import math
import os
import shutil
import subprocess
from common import *

seal_check()
for p in read(HERE / 'finalizer_sources_sealed.json')['pins']:
    verify(p)
completed = read(HERE / 'completed.json')
build = read(HERE / 'build.json')
legacy = read(HERE / 'legacy_replay.json')
unit = read(HERE / 'unit.json')
qualification = read(HERE / 'qualification.json')
score = read(HERE / 'score.json')
assert completed['completed'] and not completed['passed']
assert build['exact_source_inverse'] and build['all_tensor_bytes_qualified']
assert legacy['passed'] and unit['passed'] and qualification['passed']
assert score['completed'] and not score['passed'] and score['old_proxy_gate_still_failed']
assert [r['rank'] for r in score['rows']] == [36, 38]
assert all(r['adjusted_ratio'] > 1.05 for r in score['rows'])
pooled = math.fsum(r['candidate']['adjusted'] for r in score['rows']) / math.fsum(r['parent']['adjusted'] for r in score['rows'])
assert math.isclose(pooled, score['pooled_adjusted_ratio'], rel_tol=2e-15)
assert completed['guards'] == ['build', 'legacy_replay', 'unit', 'disabled', 'enabled', 'cold38', 'qualify', 'score']
guards = [done(n) for n in completed['guards']]
native = [read(HERE / n / 'result.json') for n in ('disabled', 'enabled', 'cold38')]
records = [r for n in native for r in n['records']]
assert len(records) == qualification['full_native_calls'] == 7
assert sum(r['flops'] for r in records) == qualification['full_native_flops']
assert all(n['passive_observers_restored'] and n['capture_additional_fnp_operations'] == 0 for n in native)
for r in records:
    for k in ('prediction', 'capture', 'weights'):
        verify(r[k])
    assert r['numeric_cache_retired'] and len(r['unfloored_variance_minima']) == 16
for r in score['rows']:
    for name in ('candidate', 'parent'):
        x = r[name]
        assert x['multiplier'] == max(.1, x['flops'] / 2**41)
        assert x['adjusted'] == x['raw'] * x['multiplier']
        assert x['fsum_relative'] <= 2e-12

groups = sorted(g['pid'] for g in guards)
active = []
markers = ('openblas_runtime_env_v1/bin/python', str(D / 'run_guarded.py'), str(HERE / 'run_native.py'), str(HERE / 'run_prepare.py'))
for line in subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True).splitlines():
    t = line.strip().split(None, 3)
    if len(t) == 4 and (int(t[0]) in groups or int(t[1]) in groups or (int(t[0]) != os.getpid() and any(m in t[3] for m in markers))):
        active.append(line)
assert not active, active
prior = read(D / 'continuation_checkpoint_v86.json')
for k in ('study', 'pending_release', 'pending_candidate', 'pending_archive', 'rejection', 'official_bank'):
    verify(prior[k])
pending = D / 'frozen_covariance_readout_composition_v1'
for name in ('submission_intent.json', 'submission.json', 'submission_attempt.json'):
    assert not (pending / name).exists()

external = {}
def collect(x):
    if isinstance(x, dict):
        if {'path', 'sha256', 'bytes'} <= x.keys() and not Path(x['path']).is_relative_to(HERE):
            p = {k: x[k] for k in ('path', 'sha256', 'bytes')}
            if p['path'] in external:
                assert p == external[p['path']]
            external[p['path']] = p
        for y in x.values():
            collect(y)
    elif isinstance(x, list):
        for y in x:
            collect(y)
for name in ('sources_sealed.json', 'native_sources_sealed.json', 'inputs.json', 'predictions_sealed.json'):
    collect(read(HERE / name))
for p in external.values():
    verify(p)

resources = dict(
    guarded_groups=len(guards), groups=groups, all_groups_retired=True,
    controller_terminal_sessions=[31070, 63223], active_numerical_sessions=[],
    full_native_calls=len(records), full_native_flops=qualification['full_native_flops'],
    peak_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_rss_bytes=max(x['peak_rss_bytes'] for x in [build, legacy, unit, qualification, score] + records),
    max_swap_growth_bytes=max(g['max_swap_growth_bytes'] for g in guards),
    total_guarded_wall_s=math.fsum(g['elapsed_s'] for g in guards),
    existing_8M_validation_labels_replayed=[400, 500], ordinary_1B_development_labels_read=[36, 38],
    new_fit_steps=0, new_samples=0, new_downloads=0, GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free,
    swap=subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True).strip())
assert resources['artifact_bytes_before_closeout'] < 250_000_000
assert resources['peak_sampled_group_rss_bytes'] < 8_000_000_000 and resources['max_swap_growth_bytes'] == 0
write(HERE / 'resources.json', resources)

report = '''# Frozen history covariance decoder fails the native development gate

Replacing the bank's covariance edge at layers 8 through 14 with the stored
history decoder increases pooled adjusted error by **19.7306%** on networks
36 and 38. Both networks regress by more than the permitted 5%. This fixed
endpoint stops at its predeclared gate; it does not advance to a panel,
confirmation, package, or upload. No scale, layer, precision, or model tuning
was performed after the result.

| Network | Parent raw MSE | Candidate raw MSE | Parent cold FLOPs | Candidate cold FLOPs | Adjusted ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
for r in score['rows']:
    report += f"| {r['rank']} | {r['parent']['raw']:.12e} | {r['candidate']['raw']:.12e} | {r['parent']['flops']} | {r['candidate']['flops']} | {r['adjusted_ratio']:.12f} |\n"
report += f'''
The pooled adjusted ratio is **{pooled:.15f}**. Candidate cold FLOPs increase
by 5,004,517,698 on each network, about 1.2%. The score uses each arm's own
complete cold bill and max(0.1, FLOPs / 2^41). Independent math.fsum checks
agree with the raw MSE calculations within 1.75e-16 relative error. These
networks are outside the decoder's six original training and two validation
networks, but were reused in earlier research. They are not pristine holdouts
and this is not an official score.

The endpoint computes a separate recurrent state with the original frozen
P0 encoder on its current physical inputs. Its three saved symmetric 65 by
65 matrices form a covariance correction modulated by 1, R, and R squared.
The old edge is replaced at its existing location. The parent mean encoder,
other parameters, physical backbone, and readout remain unchanged. The
source inverse reconstructs the exact bank and all exported tensor bytes
match. This result concerns this frozen autonomous transfer, not all history
models or covariance methods. The input-state shift from the original
oracle-state training law is present; this experiment does not isolate it
as the cause of the regression.

Formula and implementation qualification passed before final-mean scoring:

- Three deterministic kernel fixtures passed NumPy FP32/FP64, permutation,
  symmetry, repeat, zero-coefficient, and charged parameter-load checks.
- Disabled calls on 35, 100, 35 match all 16 original output rows and each
  corresponding complete cold or warm bill exactly.
- Enabled calls preserve the first eight parent output rows exactly. The
  repeated 36 and separate cold 38 reproduce output and capture bytes.
- All 60 saved recurrent states match the original NumPy equations exactly.
  All 28 source subsets match NumPy FP32 exactly, and FP64 within
  {max(x['fp64_relative'] for x in qualification['kernel_fidelity']):.12e} relative error.
- All outputs and sources are finite; every unfloored variance exceeds
  1e-10. Maximum full-source relative asymmetry is
  {max(x['symmetry_relative'] for r in records for x in r['source_checks']):.12e},
  within the predeclared 3e-5 tolerance. Native numeric caches retired.

The archived validation losses on 400 and 500 replay within
{max(e for r in legacy['rows'] for e in r['relative_errors'].values()):.12e}
relative error. This corroborates the stored model, whose current bytes
were frozen before decoding; the old verdict did not hash-pin the model
and loss equality is not unique parameter-identity proof. The old joint
covariance/response information gate remains failed. This native endpoint
explicitly revised that older stopping decision and has now also failed.

All eight guarded groups retired. The seven full native calls consumed
{resources['full_native_flops']:,} measured FLOPs. Peak sampled group RSS was
{resources['peak_sampled_group_rss_bytes']:,} bytes and peak process RSS was
{resources['peak_self_rss_bytes']:,} bytes. Added swap was zero. About
{resources['artifact_bytes_before_closeout']/1e6:.3f} MB of artifacts were
preserved before closeout. There was no new fitting, sampling, GPU work,
download, or upload.

Official submission 331684 remains adjusted 3.5844144212297072e-9, raw
1.949715441540434e-8, with 100/100 cases and zero failures. The 4e-9 milestone
is banked; 3e-9, 2e-9, and approximately 1e-9 remain unmet. The full Phase 2
goal remains active.

The separate qualified covariance/readout archive is unchanged. Automatic
approval review rejected its upload before process creation because it could
not verify trusted authorization for the exact private payload and AIcrowd
destination. The existing exact-upload approval question remains unanswered;
this study did not retry or bypass that rejection.
'''
with (HERE / 'RESULT.md').open('x') as f:
    f.write(report)
write(HERE / 'summary.json', dict(
    completed=True, implementation_qualified=True, development_gate_passed=False,
    pooled_adjusted_ratio=pooled, per_network_adjusted_ratios=[r['adjusted_ratio'] for r in score['rows']],
    candidate=build['candidate'], promoted=False, official_changed=False, goal_achieved=False,
    original_proxy_gate_still_failed=True, report=pin(HERE / 'RESULT.md'), score=pin(HERE / 'score.json'),
    qualification=pin(HERE / 'qualification.json'), resources=resources,
    pending_archive=prior['pending_archive'], upload_approval_still_pending=True, upload_attempted_this_turn=False))
write(HERE / 'closeout.json', dict(
    completed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
    implementation_qualified=True, development_gate_passed=False, candidate=build['candidate'],
    promoted=False, official_changed=False, goal_achieved=False, pooled_adjusted_ratio=pooled,
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],
    external_pins=[external[k] for k in sorted(external)], resources=resources,
    pending_archive=prior['pending_archive'], upload_approval_still_pending=True, upload_attempted_this_turn=False))
print(dict(completed=True, development_gate_passed=False, pooled_adjusted_ratio=pooled,
           artifacts=len(read(HERE / 'closeout.json')['artifacts']), external_pins=len(external), resources=resources), flush=True)

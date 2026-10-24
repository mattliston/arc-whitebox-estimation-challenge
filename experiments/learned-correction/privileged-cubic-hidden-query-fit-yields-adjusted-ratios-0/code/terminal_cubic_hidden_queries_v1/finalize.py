"""Preserve the fixed experiment and its independent reconstruction."""
import datetime
import shutil
import subprocess

from common import *

assert read(HERE / 'completed.json')['completed']
audit = read(HERE / 'audit_result.json')
assert audit['passed']
for name in ('sources_sealed.json', 'captures_sealed.json', 'features_sealed.json',
             'audit_sources_sealed.json', 'source_review_pins.json'):
    verify(name)
result = read(HERE / 'result.json')
assert result['passed'] is False
watch = [(p, read(p)) for p in sorted(HERE.glob('*_watch.json'))]
assert len(watch) == 5
assert all(r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes'] == 0 for _, r in watch)
ps = subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True)
groups = {r['pid'] for _, r in watch}
live = [line for line in ps.splitlines() if len(line.split()) >= 3 and int(line.split()[1]) in groups]
assert not live, live
capture = read(HERE / 'capture/result.json')
features = read(HERE / 'features_result.json')
self_peaks = [read(HERE / name)['peak_rss_bytes'] for name in
              ('unit.json', 'capture/result.json', 'features_result.json', 'result.json', 'audit_result.json')]
resources = dict(guarded_groups=len(watch), all_groups_retired=True,
                 full_native_calls=len(capture['records']),
                 full_native_flops=sum(r['flops'] for r in capture['records']),
                 measured_feature_calls=len(features['rows']),
                 measured_feature_flops=sum(r['helper_flops'] for r in features['rows']),
                 peak_group_rss_bytes=max(r['max_rss_bytes'] for _, r in watch),
                 peak_self_reported_rss_bytes=max(self_peaks),
                 max_swap_growth_bytes=max(r['max_swap_growth_bytes'] for _, r in watch),
                 new_samples=0, new_downloads=0, GPU_calls=0,
                 fitted_deployable_models=0, diagnostic_networks=2, privileged_oracle_fits=4,
                 artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
                 disk_free_bytes=shutil.disk_usage(HERE).free)
assert resources['artifact_bytes_before_closeout'] < 300_000_000
write(HERE / 'resources.json', resources)
table = '\n'.join(
    f"|{r['rank']}|{r['ratios']['ideal_adjusted_parent']:.12f}|"
    f"{r['ratios']['served_adjusted_parent']:.12f}|{r['ratios']['ideal_raw_affine']:.12f}|"
    f"{r['ratios']['served_raw_affine']:.12f}|"
    for r in result['rows'])
coef_error = max(a['coefficient_relative'] for r in audit['rows'] for a in r['arms'].values())
report = f'''# Terminal cubic hidden queries: closed negative

The complete retained formal pre15 third tensor was queried along all 65
fixed normalized directions from [1,H15]/sigma15. The parent trajectory
and its existing hidden state were unchanged. This adds deterministic
readout features, not a new physical cumulant source or native gradient.

|Research rank|Ideal adjusted / parent|Served adjusted / parent|Ideal raw / affine oracle|Served raw / affine oracle|
|---|---:|---:|---:|---:|
{table}

Every value had to be <=0.80 on each network. Both networks fail both gates.
Even these privileged per-network fits yield only about 10–12% extra raw
error reduction beyond the affine hidden-state oracle. All 65 affine and
130 extended directions survive the fixed numerical cutoff. There is no
column, normalization, precision, SVD, nonlinear-feature or coefficient
rescue. No cross-network fit, native candidate, panel, package or upload
follows this screen.

These are optimistic information tests on reused research networks and
ordinary existing 1B reference means. They are not held-out validation or
official scores. The saved ORACLE_DO_NOT_DEPLOY.npz coefficients depend
on each network's reference answers and must never be deployed. Complete
candidate cost has not been measured. Oracle scoring charges each
unchanged parent's authenticated cold bill, plus the measured helper and
1MF readout allowance; the affine control receives the same 1MF allowance.
The rank100 warm capture matches its independent cold control at all 16
layers. No changed parent bill or achieved competition score is claimed.

Independent dense symmetric-tensor and actual-FNP tests passed, including
signed cancellation, zero factors, nonsymmetric stored matrices, neuron
permutations, preservation and repeated bytes/bills. Passive35/100/35
captures reproduce all 16 original outputs and complete cold/warm bills;
all repeated capture arrays are exact. The formal-diagonal/native-k3
relative differences are about 7.02e-7, below the declared 1e-3 gate.
The formal FP64 tensor does not claim exact packed FP32 association.

The helper costs 14,413,775,998 and 14,345,339,518 FLOPs on ranks35/100,
under the 20GF limit. Independent wide matrix reconstruction agrees to
9.14e-16 relative or better. All features and bills were sealed before
reference decoding. Independent QR solves reproduce the primary SVD
coefficients within {coef_error:.6g} relative and all clamped FP32 payloads
exactly. Scalar math.fsum reproduces all losses and gates within
{audit['max_metric_relative']:.6g} relative. No new fit choice was made.

Three full native calls used {resources['full_native_flops']} FLOPs and
three feature calls used {resources['measured_feature_flops']} FLOPs.
All five guarded process groups retired. Maximum sampled group RSS was
{resources['peak_group_rss_bytes']} bytes; maximum self-reported process
RSS was {resources['peak_self_reported_rss_bytes']} bytes. The short oracle
stage's first sampled RSS understates its actual peak; use the self-report
or the study-wide peak. Added swap was zero. Artifacts before closeout
total {resources['artifact_bytes_before_closeout']} bytes, below 300MB.
No new samples, downloads or GPU calls.

SOURCE_REVIEW.md preserves the correction of stale historical language:
canonical fourth-trace reconstruction already ran and failed; it is not
an unfinished lead. Earlier physical-source and representation failures
remain closed. This result closes only the exact declared query/readout
family and is not a universal bound on the parent's stored information.

Official submission331684 remains adjusted3.5844144212297072e-9,
100/100 complete, zero failures. The 4e-9 bank is retained. The 3e-9,
2e-9 and approximately1e-9 targets remain unmet; the goal stays active.
'''
with (HERE / 'RESULT.md').open('x') as f:
    f.write(report)
artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'closeout.json']
external = []
for seal in ('sources_sealed.json', 'source_review_pins.json'):
    external.extend(p for p in read(HERE / seal)['pins'] if not Path(p['path']).is_relative_to(HERE))
for p in artifacts + external:
    assert sha(p['path']) == p['sha256']
write(HERE / 'closeout.json', dict(completed=True, closed_negative=True,
      information_gate_passed=False, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      artifacts=artifacts, external_pins=external, resources=resources,
      goal_achieved=False, official_changed=False, next_numeric_candidate=None))
print(json.dumps(dict(closed_negative=True, artifacts=len(artifacts), external_pins=len(external),
                      resources=resources)), flush=True)

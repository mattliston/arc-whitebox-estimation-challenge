import datetime
import shutil
import subprocess

from common import *

verify();verify('geometry_sealed.json');verify('audit_sources_sealed.json')
completed = read(HERE/'completed.json');assert completed['completed'] and not completed['passed']
assert completed['targets_opened'] is False and not (HERE/'score_result.json').exists()
audit = read(HERE/'audit_result.json');assert audit['passed'] and not audit['target_arrays_opened']
spectrum = read(HERE/'spectrum_result.json');watch = [read(p) for p in sorted(HERE.glob('*_watch.json'))]
assert len(watch) == 3 and all(r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes'] == 0 for r in watch)
groups = {r['pid'] for r in watch}
ps = subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='], text=True)
assert not [r for r in ps.splitlines() if len(r.split()) > 2 and int(r.split()[1]) in groups]
resources = dict(guarded_groups=3, all_groups_retired=True, new_full_predictions=0,
                 reference_arrays_decoded=0, ordinary_1B_targets_decoded=0, fits=0, new_samples=0,
                 new_downloads=0, GPU_calls=0, peak_group_rss_bytes=max(r['max_rss_bytes'] for r in watch),
                 peak_self_reported_rss_bytes=max(read(HERE/n)['peak_rss_bytes'] for n in ('unit.json','spectrum_result.json','audit_result.json')),
                 max_swap_growth_bytes=0, artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
                 disk_free_bytes=shutil.disk_usage(HERE).free)
assert resources['artifact_bytes_before_closeout'] < 20_000_000
write(HERE/'resources.json', resources)
failed = [(r['rank'], x['layer'], x['retained_fraction']) for r in spectrum['rows'] for x in r['layers'] if not x['spectral_gate_passed']]
table = '\n'.join(f'|{rank}|{layer}|{fraction:.12f}|' for rank, layer, fraction in failed)
cells = [x for r in audit['rows'] for x in r['layers']]
max_eigen = max(x['eigenvalue_relative'] for x in cells)
max_fraction = max(x['energy_fraction_delta'] for x in cells)
relative_norms = [x['full_negative_relative_norm'] for x in cells]
bounds = [x['full_negative_gaussian_response_rms_bound'] for x in cells]
report = f'''# Current negative covariance modes: closed before accuracy scoring

The fixed rank16 action fails its spectral-energy prerequisite on both
research networks. These are the three failed cells out of fourteen:

|Rank|Post layer|Negative energy retained by16 modes|
|---|---:|---:|
{table}

Every cell required at least0.99. No rank, threshold, diagonal scaling,
gain, layer or fit rescue follows. score.py remains unexecuted, and no
8M covariance labels or ordinary1B means were decoded. No native candidate,
complete candidate bill, accuracy score, package or upload was produced.
This is not evidence of a failed accuracy gate, because that stage did not run.

The negative component is small: its full Frobenius norm is
{min(relative_norms):.6g} to {max(relative_norms):.6g} of the actual covariance
norm over these cells. This describes the current saved331684 trajectory;
it does not establish a general PSD or accuracy guarantee. Original
negative counts range32..129; signs very near zero can be sensitive to
numerical precision. The fixed squared-energy gate is independently stable.

The audit uses an eigenvalue-only solver on the opposite matrix triangle.
Spectra agree within {max_eigen:.6g} relative, and retained-energy fractions
within {max_fraction:.6g} absolute. Selected factors satisfy the original
eigen-equations and orthogonality. Dense and factored covariance-to-mean
responses agree. Small PSD/zero/indefinite, permutation, orthogonal-change
and rank-selection tests pass; Gaussian derivative finite differences agree
within8.13e-11 relative.

A separately labeled post-result, target-free geometry bound gives
{min(bounds):.6g} to {max(bounds):.6g} RMS for the immediate Gaussian mean
response of the FULL negative spectral part. It follows from its spectral
norm and the next weight-column norms. This is not a measured true-error
reduction or bound on learned-head/recurrent effects, and it does not
change the failed gate. No additional estimator action was selected.

For context, full negative spectral clipping of a symmetric matrix is a
standard positive-semidefinite projection construction ([Higham,1988](https://eprints.maths.manchester.ac.uk/363/)).
Our proposed C+D preserves the parent's antisymmetric part, changes the
diagonal and keeps at most16 modes; it is not that full projection or the
earlier diagonal-preserving repair. Offline eigendecompositions do not
establish an affordable inference algorithm. The existing symmetry-only,
high-order, covariance-fit, FMM and compression results remain unchanged.

Three guarded groups retired, zero added swap. Peak sampled group RSS
{resources['peak_group_rss_bytes']} bytes; self-reported peak
{resources['peak_self_reported_rss_bytes']} bytes. Artifacts before closeout
total {resources['artifact_bytes_before_closeout']} bytes. No new forward
prediction, samples, download, fit or GPU work. Existing captures/weights
were reused by reference without duplicating the datasets.

Official331684 remains adjusted3.5844144212297072e-9,100/100 complete,
zero failures.4e-9 is banked;3e-9,2e-9 and approximately1e-9 remain unmet.
The original goal stays active. This closes only the declared spectral
action and does not prove every covariance correction ineffective.
'''
with (HERE/'RESULT.md').open('x') as f:
    f.write(report)
artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'closeout.json']
external = [x for x in read(HERE/'sources_sealed.json')['pins'] if not Path(x['path']).is_relative_to(HERE)]
for item in artifacts+external:
    assert sha(item['path']) == item['sha256']
write(HERE/'closeout.json', dict(completed=True, closed_negative=True, stage='rank16_spectrum',
      created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), artifacts=artifacts,
      external_pins=external, resources=resources, targets_decoded=False, goal_achieved=False,
      official_changed=False, next_numeric_candidate=None))
print(json.dumps(dict(closed_negative=True, failed_cells=failed, artifacts=len(artifacts),
                     external_pins=len(external), resources=resources)), flush=True)

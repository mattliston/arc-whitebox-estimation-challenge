"""Close only retired local work; keep runtime failures and pending grades explicit."""
import datetime
import shutil
from common import HERE, D, PARENT, read, write, pin, verify, seal_check

seal_check()
seal_check('boundary_sources_sealed.json')
for p in read(HERE/'prospective_release_sources.json')['pins']:
    verify(p)
assert read(HERE/'native_qualified.json')['passed']
pairs = read(HERE/'pairs_qualified.json')
outcome = read(HERE/'boundary_completed.json')['outcome']
guards = {p.name:read(p) for p in HERE.glob('*_watch.json')}
assert all(g['completed'] and g['whole_group_retired'] and g['exit_code']==0
           and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<8_000_000_000
           for g in guards.values())
native = []
clients = []
for p in HERE.glob('*/result.json'):
    native.extend(read(p).get('records',[]))
for p in HERE.glob('*/completed.json'):
    clients.extend(read(p).get('records',[]))
assert len(native)+len(clients)<=24
uploaded = (HERE/'submission_attempt.json').exists() and read(HERE/'submission_attempt.json')['exit_code']==0
if not outcome['passed']:
    for name in ['package.json','submission_intent.json','release_completed.json']:
        assert not (HERE/name).exists(),name
if uploaded:
    seal_check('release_sources_sealed.json')
    assert read(HERE/'boundary_qualified.json')['passed']
    assert read(HERE/'release_completed.json')['completed']
leaves = [g for n,g in guards.items() if not n.endswith('controller_watch.json')]
resources = dict(guarded_groups=len(guards),leaf_guarded_groups=len(leaves),all_groups_retired=True,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards.values()),max_swap_growth_bytes=0,
    leaf_elapsed_s=sum(g['elapsed_s'] for g in leaves),native_full_calls=len(native),
    native_full_flops=sum(r['flops'] for r in native),client_full_calls=len(clients),
    client_full_flops=sum(r['summary']['flops_used'] for r in clients),
    peak_native_self_rss_bytes=max(r['max_rss_bytes'] for r in native),
    fits=0,truth_arrays_decoded=0,new_samples=0,new_downloads=0,GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free)
assert resources['artifact_bytes_before_closeout']<150_000_000
write(HERE/'resources.json',resources)
native_result = read(HERE/'native_qualified.json')
lines = [
    '# Current-parent Gaussian gate reuse', '',
    'Parent #331684 is adjusted 3.5844144212297072e-9 with 100/100 completed.',
    'The candidate shares only the current layer clipped variance, sigma, alpha and gate.',
    'The scalar final layer retains its default path; no model coefficients or physical equations change.',
    'There is no network-dependent global cache and no prediction work outside the timed call.',
    'Gaussian/pair-map unit fixtures and full disabled/enabled native checks pass byte-for-byte.',
    'All sixteen output layers, own repeat and warm/cold behavior remain exact.',
    f"Each complete prediction removes {native_result['expected_removed_flops']} counted operations.", '',
    f"- Five-pair prediction-only residual ratio: {pairs['residual_ratio']:.15g}; required <=0.98.",
    f"- Residual wins: {pairs['residual_wins']}/5; required >=4.",
    f"- Maximum candidate residual: {pairs['maximum_candidate_residual_s']:.15g}s; required <=0.25s.",
    f"- Worst residual ratio: {pairs['worst_residual_ratio']:.15g}; required <=1.10.",
    f"- Summed total-wall ratio: {pairs['wall_ratio']:.15g}; required <=1.05.",
    f"- Worst total-wall ratio: {pairs['worst_wall_ratio']:.15g}; required <=1.10.", '',
    'All paired outputs are byte exact and the ordinary client confirms exactly sixty fewer requests:',
    'fifteen each of maximum, square root, division and Gaussian CDF. Other operation counts are unchanged.',
    'The fixed alternating five-pair measurement uses module initialization outside the prediction budget,',
    'matching the installed SDK timing scope. These local results do not prove a platform runtime gain.', '',
    f"Final local outcome: {outcome}", f"Uploaded: {uploaded}.",
    'No model fit, new samples, label decoding, dataset download or GPU work occurred.',
    f"All {len(guards)} guarded groups are retired; zero added swap; peak sampled RSS {resources['peak_group_rss_bytes']} bytes.",
    'The approximately 1e-9 goal remains unachieved; official grading governs any new bank.', '']
if not outcome['passed']:
    lines += ['The declared runtime gate failed. No repetition, threshold change, variant, package or upload follows.',
              'Conditional release scripts are preserved but ineligible.', '']
with (HERE/'RESULT.md').open('x') as f:
    f.write('\n'.join(lines))
files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
record = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outcome=outcome,
    uploaded=uploaded,official_result_pending=uploaded,goal_achieved=False,resources=resources,
    artifact_pins=[pin(p) for p in files],
    external_pins=[pin(PARENT),pin(D/'official_best_331684.json'),
                   pin(D/'full_lower_multilayer_mean_observer_v1/closeout.json')])
for p in record['artifact_pins']+record['external_pins']:
    verify(p)
write(HERE/'closeout.json',record)
print(dict(artifact_pins=len(files),outcome=outcome,uploaded=uploaded,resources=resources),flush=True)

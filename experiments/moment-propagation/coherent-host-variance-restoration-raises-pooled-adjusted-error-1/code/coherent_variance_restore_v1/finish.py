"""Post-result stdlib verification and immutable closeout, not a numerical run."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
ROOT = P2.parents[3]
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)


seal = read(HERE / 'sources_sealed.json')
for r in seal['sources'] + seal['external_pins']:
    assert sha(r['path']) == r['sha256'], r['path']
guards = [read(HERE / n) for n in ('unit_watch.json', 'null_watch.json',
          'null_verify_watch.json', 'dev_watch.json', 'cold100_watch.json', 'score_watch.json')]
assert all(g['completed'] and g['whole_group_retired'] and not g['error']
           and g['max_swap_growth_bytes'] == 0 and g['max_rss_bytes'] < 8_000_000_000 for g in guards)
calls = []
for sub in ('null', 'dev', 'cold100'):
    r = read(HERE / sub / 'result.json')
    assert r['runner_sha256'] == sha(D / 'weight5_reuse_v1/native.py')
    for c in r['records']:
        assert sha(c['output']) == c['output_sha256']
        assert c['weight5_cache']['completed'] and c['weight5_cache']['numeric_cache_retired']
        calls.append(c)
score = read(HERE / 'dev_score.json')
assert not score['development_gate_passed']
assert all(r['adjusted_ratio'] > 1 for r in score['rows'])
for r in score['rows']:
    for k in ('parent', 'candidate'):
        v = r[k]
        assert abs(v['adjusted'] - v['raw']*max(.1, v['flops']/2**41)) < 1e-23
assert score['repeat_byte_exact'] and score['warm_cold100_byte_exact'] and score['first_two_layers_byte_exact']
assert read(HERE / 'unit.json')['passed'] and read(HERE / 'null_verdict.json')['passed']
peak = max([g['max_rss_bytes'] for g in guards] + [c['max_rss_bytes'] for c in calls]
           + [read(HERE / 'unit.json')['peak_rss_bytes']])
resources = dict(guarded_stages=len(guards), successful_guarded_stages=len(guards),
                 full_prediction_calls=len(calls), null_calls=2, candidate_calls=4,
                 billed_prediction_flops=sum(c['flops'] for c in calls),
                 total_guarded_wall_s=sum(g['elapsed_s'] for g in guards),
                 peak_rss_bytes=peak, peak_includes_worker_self_reports=True,
                 additional_swap_bytes=0, all_owned_groups_retired=True,
                 terminal_sessions=[68312,59673,53768], new_samples=False,
                 new_training=False, new_archive=False, new_upload=False,
                 remaining_disk_GiB=shutil.disk_usage(HERE).free/1024**3)
assert peak < 8_000_000_000 and len(calls) == 6
write(HERE / 'resources.json', resources)
external = {r['path']: r['sha256'] for r in seal['external_pins']}
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = read(store / 'manifest.json')['roles']['observer_fit']['row_manifests']
external[str(store / 'manifest.json')] = sha(store / 'manifest.json')
for row in score['rows']:
    rank = row['rank']
    truth = P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
    assert sha(truth) == row['truth_sha256']
    external[str(truth)] = sha(truth)
    weight = store / Path(manifest[rank]['path']).parent / 'weights.npy'
    assert sha(weight) == next(c['weights_sha256'] for c in calls if c['rank'] == rank)
    external[str(weight)] = sha(weight)
for sub in ('dev', 'cold100'):
    for c in read(D / 'lazy_rotation_v1' / sub / 'result.json')['records']:
        assert sha(c['output']) == c['output_sha256']
        external[c['output']] = c['output_sha256']
archive = D / 'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz'
assert sha(archive) == '594aa6e9debf9f0e370c29afea9ccb33b475af6d27f7454bbd85abfeb8a5c8f4'
external[str(archive)] = sha(archive)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
files = [p for p in sorted(HERE.rglob('*')) if p.is_file()]
assert sum(p.stat().st_size for p in files) < 10_000_000
closeout = dict(created_at_utc=now,
                status='completed negative at fixed native development gate',
                source_seal_sha256=sha(HERE / 'sources_sealed.json'),
                artifacts=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
                external_pins=[dict(path=p,sha256=h) for p,h in sorted(external.items())],
                resources=resources, no_active_worker=True,
                goal_status='active', goal_achieved=False)
write(HERE / 'closeout.json', closeout)
print(json.dumps(dict(closeout_sha256=sha(HERE/'closeout.json'),
                     artifact_count=len(files), external_pin_count=len(external),
                     resources=resources)), flush=True)

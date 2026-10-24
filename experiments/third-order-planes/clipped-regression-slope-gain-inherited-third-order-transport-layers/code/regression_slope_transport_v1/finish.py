"""Post-score stdlib audit and immutable closeout for this fixed failed rule."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
D=HERE.parent;P2=D.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2)

parser=argparse.ArgumentParser()
parser.add_argument('--sessions',required=True)
args=parser.parse_args()
seal=read(HERE/'sources_sealed.json')
for item in seal['sources']+seal['external_pins']:
    assert sha(item['path'])==item['sha256'],item['path']
guards=[read(HERE/name) for name in ('unit_watch.json','null_watch.json','dev_watch.json','cold100_watch.json','score_watch.json')]
assert all(g['completed'] and g['whole_group_retired'] and g['exit_code']==0
           and g['error'] is None and g['max_swap_growth_bytes']==0 for g in guards)
unit=read(HERE/'unit.json');null=read(HERE/'null_verdict.json');build=read(HERE/'build.json')
assert unit['passed'] and null['passed'] and null['cold_repeat_bills_exact']
assert unit['mixed_coefficient_counterexample_retained'] and not unit['derivative_correctness_claim']
assert build['source_inverse_byte_exact'] and build['aux_raw2_only_inverse_exact'] and build['source_through_newborn_exact']
score=read(HERE/'dev_score.json')
assert not score['development_gate_passed']
assert score['repeat_byte_exact'] and score['warm_cold100_byte_exact'] and score['first_two_layers_byte_exact']
for row in score['rows']:
    for key in ('parent','candidate'):
        r=row[key];assert abs(r['adjusted']-r['raw']*max(.1,r['flops']/2**41))<1e-23
    assert abs(row['raw_ratio']-row['candidate']['raw']/row['parent']['raw'])<1e-14
    assert abs(row['adjusted_ratio']-row['candidate']['adjusted']/row['parent']['adjusted'])<1e-14
assert score['development_gate_passed']==all(r['adjusted_ratio']<=.95 for r in score['rows'])
calls=[]
for sub in ('null','dev','cold100'):
    rec=read(HERE/sub/'result.json')
    assert rec['runner_sha256']==sha(D/'weight5_reuse_v1/native.py')
    assert rec['candidate_sha256']==build['null_sha256' if sub=='null' else 'candidate_sha256']
    for call in rec['records']:
        assert sha(call['output'])==call['output_sha256']
        assert call['weight5_cache']['completed'] and call['weight5_cache']['numeric_cache_retired']
        calls.append(call)
assert len(calls)==6
peak=max([g['max_rss_bytes'] for g in guards]+[c['max_rss_bytes'] for c in calls]+[unit['peak_rss_bytes']])
assert peak<8_000_000_000
resources=dict(guarded_stages=len(guards),successful_guarded_stages=len(guards),
               full_prediction_calls=len(calls),null_calls=2,candidate_calls=4,
               billed_prediction_flops=sum(c['flops'] for c in calls),
               unit_native_billed_flops=2*unit['native_complete_context_flops'],
               total_guarded_wall_s=sum(g['elapsed_s'] for g in guards),
               peak_rss_bytes=peak,peak_includes_worker_self_reports=True,
               additional_swap_bytes=0,all_owned_groups_retired=True,
               terminal_sessions=list(map(int,args.sessions.split(','))),
               new_samples=False,new_training=False,new_archive=False,new_upload=False,
               remaining_disk_GiB=shutil.disk_usage(HERE).free/2**30)
write(HERE/'resources.json',resources)
external={r['path']:r['sha256'] for r in seal['external_pins']}
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=read(store/'manifest.json')['roles']['observer_fit']['row_manifests']
external[str(store/'manifest.json')]=sha(store/'manifest.json')
for row in score['rows']:
    rank=row['rank'];truth=P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
    assert sha(truth)==row['truth_sha256'];external[str(truth)]=sha(truth)
    weight=store/Path(manifest[rank]['path']).parent/'weights.npy'
    assert sha(weight)==next(c['weights_sha256'] for c in calls if c['rank']==rank)
    external[str(weight)]=sha(weight)
for sub in ('dev','cold100'):
    for call in read(D/'lazy_rotation_v1'/sub/'result.json')['records']:
        assert sha(call['output'])==call['output_sha256']
        external[call['output']]=call['output_sha256']
archive=D/'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz'
assert sha(archive)=='594aa6e9debf9f0e370c29afea9ccb33b475af6d27f7454bbd85abfeb8a5c8f4'
external[str(archive)]=sha(archive)
assert (HERE/'NEGATIVE_RESULT.md').exists()
files=sorted(p for p in HERE.rglob('*') if p.is_file());size=sum(p.stat().st_size for p in files)
assert size<12_000_000
closeout=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              status='completed fixed empirical regression-slope native test; development gate failed',
              source_seal_sha256=sha(HERE/'sources_sealed.json'),
              artifacts=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
              external_pins=[dict(path=p,sha256=h) for p,h in sorted(external.items())],
              artifact_bytes_before_closeout=size,resources=resources,no_active_worker=True,
              goal_status='active',goal_achieved=False)
write(HERE/'closeout.json',closeout)
print(json.dumps(dict(closeout_sha256=sha(HERE/'closeout.json'),artifacts=len(files),
                     external_pins=len(external),artifact_bytes_before_closeout=size,resources=resources)),flush=True)

from common import (HERE,D,CANDIDATE_SHA,PARENT_SHA,read,write,pin,verify,verify_pin)
import numpy as np

verify()
for item in read(HERE/'matched_runtime_sources_sealed.json')['pins']:
    verify_pin(item)
candidate=read(HERE/'flms_panel_boundary/completed.json')
parent=read(HERE/'flms_parent_boundary/completed.json')
old=read(D/'lazy_rotation_objective_v1/lazy_objective_boundary/completed.json')
assert candidate['candidate_sha256']==CANDIDATE_SHA
assert parent['candidate_sha256']==old['candidate_sha256']==PARENT_SHA
for name in ('boundary_watch.json','parent_boundary_watch.json'):
    guard=read(HERE/name)
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
rows=[]
for i,rank in enumerate((35,100,35)):
    a,b,c=candidate['records'][i],parent['records'][i],old['records'][i]
    assert a['rank']==b['rank']==c['rank']==rank
    path=HERE/'flms_parent_boundary'/f'call_{i:02d}_rank{rank:04d}.npy'
    old_path=D/'lazy_rotation_objective_v1/lazy_objective_boundary'/path.name
    assert np.load(path,allow_pickle=False).tobytes()==np.load(old_path,allow_pickle=False).tobytes()
    assert b['summary']['flops_used']==c['summary']['flops_used']
    assert b['requests']==c['requests']
    rows.append(dict(rank=rank,index=i,parent_output=pin(path),parent_bytes_exact=True,
        parent_flops_exact=True,parent_requests_exact=True,
        candidate_wall_s=a['wall_s'],parent_current_wall_s=b['wall_s'],parent_old_wall_s=c['wall_s'],
        candidate_to_current_wall=a['wall_s']/b['wall_s'],current_to_old_parent_wall=b['wall_s']/c['wall_s'],
        candidate_requests=a['requests'],parent_requests=b['requests'],request_ratio=a['requests']/b['requests'],
        candidate_flops=a['summary']['flops_used'],parent_flops=b['summary']['flops_used'],
        flop_ratio=a['summary']['flops_used']/b['summary']['flops_used']))
pooled=sum(r['candidate_wall_s'] for r in rows)/sum(r['parent_current_wall_s'] for r in rows)
slowdown=sum(r['parent_current_wall_s'] for r in rows)/sum(r['parent_old_wall_s'] for r in rows)
passed=(pooled<=1.075 and all(r['candidate_to_current_wall']<=1.10 and
        r['request_ratio']<=1.02 and r['flop_ratio']<=1.05 for r in rows))
result=dict(completed=True,rows=rows,pooled_wall_ratio=pooled,parent_current_to_old_wall_ratio=slowdown,
    calibrated_runtime_gate_passed=passed,original_absolute_runtime_gate_passed=False,
    original_historical_ratio_gate_passed=False,platform_trial_not_guaranteed=True,
    source_seal=pin(HERE/'matched_runtime_sources_sealed.json'))
write(HERE/'matched_runtime_comparison.json',result)
print(result,flush=True)

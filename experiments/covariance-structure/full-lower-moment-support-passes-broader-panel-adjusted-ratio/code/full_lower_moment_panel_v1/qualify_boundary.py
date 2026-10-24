from common import (HERE,D,CANDIDATE,CANDIDATE_SHA,read,write,pin,sha,verify,verify_pin)
import numpy as np

verify()
for item in read(HERE/'boundary_sources_sealed.json')['pins']:
    verify_pin(item)
assert read(HERE/'panel_score.json')['panel_advancement_passed']
assert read(HERE/'confirmation_score.json')['confirmation_advancement_passed']
new=read(HERE/'flms_panel_boundary/completed.json')
old=read(D/'lazy_rotation_objective_v1/lazy_objective_boundary/completed.json')
native=read(D/'full_lower_moment_support_v1/dev/result.json')
watch=read(HERE/'boundary_watch.json')
assert new['completed'] and new['candidate_sha256']==CANDIDATE_SHA
assert native['candidate_sha256']==CANDIDATE_SHA and sha(CANDIDATE)==CANDIDATE_SHA
assert watch['completed'] and watch['whole_group_retired']
assert watch['max_swap_growth_bytes']==0
assert not new['truth_accessed']
rows=[]
for i,rank in enumerate((35,100,35)):
    call,parent,reference=new['records'][i],old['records'][i],native['records'][i]
    assert call['rank']==parent['rank']==reference['rank']==rank
    path=HERE/'flms_panel_boundary'/f'call_{i:02d}_rank{rank:04d}.npy'
    prediction=np.load(path,allow_pickle=False)
    assert prediction.shape==(16,1024) and prediction.dtype==np.float32
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    assert prediction.tobytes()==np.load(reference['output'],allow_pickle=False).tobytes()
    assert call['summary']['flops_used']==reference['flops']
    rows.append(dict(rank=rank,index=i,output=pin(path),native_bytes_exact=True,
        native_flops_exact=True,wall_s=call['wall_s'],parent_wall_s=parent['wall_s'],
        wall_ratio=call['wall_s']/parent['wall_s'],requests=call['requests'],
        parent_requests=parent['requests'],flops=call['summary']['flops_used'],
        residual_s=call['summary']['residual_wall_time_s']))
assert rows[0]['output']['sha256']==rows[2]['output']['sha256']
ratio=sum(r['wall_s'] for r in rows)/sum(r['parent_wall_s'] for r in rows)
passed=all(r['wall_s']<45 for r in rows) and ratio<=1.10
write(HERE/'boundary_comparison.json',dict(completed=True,rows=rows,
    candidate_sha256=CANDIDATE_SHA,native_byte_exact=True,native_flops_exact=True,
    repeat_byte_exact=True,numpy_free_client=True,pooled_wall_ratio=ratio,
    boundary_gate_passed=passed,watch=pin(HERE/'boundary_watch.json'),
    caveat='Saved same-machine Lazy wall comparison; not a guarantee of platform runtime.'))
print(dict(passed=passed,pooled_wall_ratio=ratio,rows=rows),flush=True)

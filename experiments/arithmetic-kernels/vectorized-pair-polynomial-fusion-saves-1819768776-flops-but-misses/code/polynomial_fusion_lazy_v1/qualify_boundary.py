"""Actual client qualification for the frozen objective-qualified candidate."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
SOURCE=HERE
PARENT=D/'lazy_rotation_objective_v1'
read=lambda p:json.loads(Path(p).read_text())
build=read(HERE/'build.json')
assert read(HERE/'dev_score.json')['development_gate_passed']
assert read(HERE/'panel_score.json')['panel_gate_passed']
assert read(HERE/'confirmation_score.json')['confirmation_gate_passed']
new=read(HERE/'fusion_boundary/completed.json')
old=read(PARENT/'lazy_objective_boundary/completed.json')
native=read(SOURCE/'dev/result.json')
watch=read(HERE/'boundary_watch.json')
assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes']==0
assert new['completed'] and new['candidate_sha256']==native['candidate_sha256']==build['candidate_sha256']
assert hashlib.sha256((HERE/'candidate.py').read_bytes()).hexdigest()==build['candidate_sha256']
assert old['candidate_sha256']==build['parent_sha256']
rows=[]
for i,rank in enumerate((35,100,35)):
    a,b,n=new['records'][i],old['records'][i],native['records'][i]
    assert a['rank']==b['rank']==n['rank']==rank
    result=np.load(HERE/'fusion_boundary'/f'call_{i:02d}_rank{rank:04d}.npy')
    parent=np.load(PARENT/'lazy_objective_boundary'/f'call_{i:02d}_rank{rank:04d}.npy')
    assert np.array_equal(result,np.load(n['output']))
    assert np.array_equal(result[:1],parent[:1])
    assert a['summary']['flops_used']==n['flops']
    context=a['summary']
    rows.append(dict(index=i,rank=rank,native_byte_exact=True,native_flops_exact=True,
                     new_wall_s=a['wall_s'],parent_wall_s=b['wall_s'],wall_ratio=a['wall_s']/b['wall_s'],
                     new_requests=a['requests'],parent_requests=b['requests'],
                     request_delta=a['requests']-b['requests'],flops=context['flops_used'],
                     all_layer_parent_maxabs=float(np.max(np.abs(result.astype(np.float64)-parent.astype(np.float64)))),
                     backend_s=context['flopscope_backend_time_s'],overhead_s=context['flopscope_overhead_time_s'],
                     residual_s=context['residual_wall_time_s']))
assert np.array_equal(np.load(HERE/'fusion_boundary/call_00_rank0035.npy'),
                      np.load(HERE/'fusion_boundary/call_02_rank0035.npy'))
pooled=sum(r['new_wall_s'] for r in rows)/sum(r['parent_wall_s'] for r in rows)
passed=all(r['new_wall_s']<45 for r in rows) and pooled<=1.05
result=dict(candidate_sha256=build['candidate_sha256'],rows=rows,native_byte_exact=True,
            native_flops_exact=True,repeat_byte_exact=True,first_layer_byte_exact=True,
            numpy_free_client=True,pooled_wall_ratio=pooled,runtime_gate_passed=passed,boundary_gate_passed=passed,
            old_component_gate_remains_failed=True,watch=watch,truth_accessed=False,
            caveat='Saved Lazy comparison; local complete-context timing, not a platform guarantee.')
with (HERE/'boundary_comparison.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)

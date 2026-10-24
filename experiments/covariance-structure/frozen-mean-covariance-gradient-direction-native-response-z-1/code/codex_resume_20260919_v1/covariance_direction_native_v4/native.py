"""One immutable native probe; no truth access or direction adjustment."""
import resource,sys,time,types
from common import HERE,FIT,check,meta,module,read,sha,write
check()
rank=int(sys.argv[1]);label=sys.argv[2];assert rank in FIT and label in ('zero','plus','minus','halfplus','halfminus','repeat')
assert read(HERE/'unit_result.json')['passed']
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
assert all(x['num_threads']==1 for x in threadpool_info())
if label=='zero':
    assert rank==35;candidate=read(HERE/'unit_result.json')['zero_source']
else:
    probes=read(HERE/'probes/manifest.json');assert probes['completed']
    entry=next(x for x in probes['probes'] if x['name']==('halfplus' if label=='repeat' else label))
    candidate=entry['source']
assert sha(candidate['path'])==candidate['sha256']
item=next(x for x in read(HERE/'inputs.json')['fit'] if x['rank']==rank)
out=HERE/f'native_rank{rank:04d}_{label}';out.mkdir(exist_ok=False)
weights=np.load(item['weights'],allow_pickle=False)
assert weights.dtype==np.float32 and weights.shape==(16,1024,1024)
with np.load(item['capture'],allow_pickle=False) as z:parent=z['prediction']
assert parent.shape==(16,1024) and np.isfinite(parent).all()
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:
    ww=[fnp.asarray(weights[l]) for l in range(16)]
assert ingress.summary_dict()['flops_used']==0
mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
start=time.monotonic()
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    mod=module(candidate['path'],'cdn_native_candidate')
    output=mod.Estimator().predict(mlp,2**41)
value=np.asarray(output,dtype=np.float32).copy();seconds=time.monotonic()-start
assert value.shape==(16,1024) and np.isfinite(value).all() and (value>=0).all()
assert value[:2].tobytes()==parent[:2].tobytes()
assert mod._CDN_LAST==list(range(1,15))
assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
if label=='zero':assert value.tobytes()==parent.tobytes()
np.save(out/'prediction.npy',value)
if label=='repeat':
    assert rank==35 and sha(out/'prediction.npy')==sha(HERE/'native_rank0035_halfplus/prediction.npy')
record=dict(completed=True,rank=rank,label=label,candidate=candidate,prediction=meta(out/'prediction.npy'),
    first2_parent_bytes_exact=True,zero_all16_bytes_exact=(True if label=='zero' else None),
    repeat_byte_exact=(True if label=='repeat' else None),complete_flops=ctx.summary_dict()['flops_used'],
    parent_flops=item['parent_flops'],all14_covariance_writes=True,trust_envelope_passed=True,
    numeric_cache_retired=True,truth_accessed=False,wall_s=seconds,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    source_seal=meta(HERE/'sources_sealed.json'))
write(out/'result.json',record);print(record,flush=True)

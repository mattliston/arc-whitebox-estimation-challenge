"""Only retain selector references during the full counted call."""
import argparse
import importlib.util
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import HERE,read,write,pin,sha,verify,seal_check,weights_path

ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('out');ap.add_argument('rank',type=int)
args=ap.parse_args();seal_check();assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
out=HERE/args.out;out.mkdir(exist_ok=False)
source=__import__('pathlib').Path(args.source).resolve()
allowed=[p[k] for p in read(HERE/'build.json')['variants'] for k in ['original','canonical']]
verify(next(p for p in allowed if p['path']==str(source)))
wp=weights_path(args.rank);weights=np.load(wp,allow_pickle=False)
assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
assert ingress.summary_dict()['flops_used']==0
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('selector_traced_estimator',source)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
held=[]
mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{args.rank}')
began=time.monotonic()
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    spec.loader.exec_module(mod);original=mod.select
    def select(score,floor):
        result=original(score,floor)
        held.append((score,floor,result[0],result[1]))
        return result
    mod.select=select
    prediction=mod.Estimator().predict(mlp,2**41)
wall=time.monotonic()-began;summary=ctx.summary_dict()
assert len(held)==45 and mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['numeric_cache_retired']
output=np.asarray(prediction).copy()
assert output.shape==(16,1024) and output.dtype==np.float32 and np.isfinite(output).all() and (output>=0).all()
scores=[];indices=[];masks=[];counts=[]
for score,floor,idx,mask in held:
    score=np.asarray(score).copy();idx=np.asarray(idx).copy();mask=np.asarray(mask).copy()
    assert score.shape==mask.shape==(1024,) and mask.dtype==np.bool_ and idx.ndim==1
    assert np.isfinite(score).all() and floor==.01
    count=min(1024,((np.count_nonzero(score>=floor)+31)//32)*32)
    expected=np.argsort(-score,kind='stable')[:count]
    assert len(idx)==count and np.array_equal(np.sort(idx),np.sort(expected))
    check=np.zeros(1024,dtype=bool);check[idx]=True;assert check.tobytes()==mask.tobytes()
    padded=np.full(1024,-1,dtype=np.int64);padded[:len(idx)]=idx
    scores.append(score);indices.append(padded);masks.append(mask);counts.append(count)
path=out/'trace.npz'
np.savez(path,scores=np.stack(scores),indices=np.stack(indices),masks=np.stack(masks),counts=np.asarray(counts,dtype=np.int64))
output_path=out/'prediction.npy';np.save(output_path,output)
record=dict(rank=args.rank,source=pin(source),trace=pin(path),prediction=pin(output_path),weights=pin(wp),
    flops=summary['flops_used'],multiplier=max(.1,summary['flops_used']/2**41),wall_s=wall,
    includes_initialization=True,selector_calls=45,extra_capture_arithmetic=0,numeric_cache_retired=True,
    max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threadpools=threadpool_info(),truth_accessed=False)
held.clear();del output,prediction,scores,indices,masks,counts,weights,ww,mlp
write(out/'result.json',record)
print(dict(rank=args.rank,flops=record['flops'],wall_s=wall,selector_calls=45),flush=True)

"""Reference-only gnn_step taps; all copying happens after the counted parent."""
import argparse,importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import HERE,PARENT,PARENT_SHA,read,write,pin,sha,seal_check,weights_path
ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('ranks');args=ap.parse_args()
seal_check();assert all(t['num_threads']==1 for t in threadpool_info())
out=HERE/args.out;out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('multilayer_exact_parent',PARENT);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
held={};records=[]
for index,rank in enumerate(map(int,args.ranks.split(','))):
    wp=weights_path(rank);weights=np.load(wp,allow_pickle=False);assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
    assert ingress.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    held.clear();started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if index==0:
            spec.loader.exec_module(mod);original=mod.gnn_step
            def step(l,x,sigma,w,state,p):
                answer=original(l,x,sigma,w,state,p)
                assert l not in held;held[l]=(x,sigma,answer[1])
                return answer
            mod.gnn_step=step;estimator=mod.Estimator()
        prediction=estimator.predict(mlp,2**41)
    wall=time.monotonic()-started;bill=ctx.summary_dict()['flops_used']
    assert sorted(held)==list(range(16)) and mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['numeric_cache_retired']
    arrays=dict(features=np.stack([np.asarray(held[l][0]).copy() for l in range(16)]),
        sigma=np.stack([np.asarray(held[l][1]).copy() for l in range(16)]),
        hidden=np.stack([np.asarray(held[l][2]).copy() for l in range(16)]),prediction=np.asarray(prediction).copy())
    assert arrays['features'].shape==(16,1024,30) and arrays['features'].dtype==np.float32
    assert arrays['hidden'].shape==(16,1024,64) and arrays['hidden'].dtype==np.float32
    assert arrays['sigma'].shape==(16,1024) and arrays['sigma'].dtype==np.float64
    assert arrays['prediction'].shape==(16,1024) and arrays['prediction'].dtype==np.float32
    assert all(np.isfinite(a).all() for a in arrays.values()) and (arrays['prediction']>=0).all() and (arrays['sigma']>0).all()
    path=out/f'call{index:02d}_rank{rank:04d}.npz';np.savez(path,**arrays)
    record=dict(index=index,rank=rank,capture=pin(path),flops=bill,multiplier=max(.1,bill/2**41),includes_initialization=index==0,
        weights=pin(wp),source_sha256=PARENT_SHA,extra_parent_arithmetic=0,numeric_cache_retired=True,wall_s=wall,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    records.append(record);write(out/f'call{index:02d}.json',record);print(dict(rank=rank,index=index,flops=bill,wall_s=wall),flush=True)
    held.clear();del arrays,prediction,weights,ww,mlp
write(out/'result.json',dict(completed=True,records=records,source_sha256=sha(PARENT),truth_accessed=False,capture_source=pin(__file__)))

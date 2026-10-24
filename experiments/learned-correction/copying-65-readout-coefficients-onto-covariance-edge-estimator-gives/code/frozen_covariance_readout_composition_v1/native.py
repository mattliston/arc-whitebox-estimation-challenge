"""Meter a cold or repeated actual estimator; save small passive references."""
import argparse,importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--ranks',required=True);ap.add_argument('--bank',action='store_true');args=ap.parse_args()
seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
b=read(HERE/'build.json');verify(b['candidate']);source=PARENT if args.bank else HERE/'candidate.py'
meta=read(HERE/'inputs.json');ranks=list(map(int,args.ranks.split(',')))
if set(ranks)&set(CONFIRM):
    done('qualify');assert read(HERE/'qualification.json')['passed'] and len(ranks)==1 and not args.bank
out=HERE/args.out;out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('frozen_composition_native',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
held={};variances=[];records=[]
for index,rank in enumerate(ranks):
    wp=meta['weights'][str(rank)];verify(wp);w=np.load(wp['path'],allow_pickle=False)
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(w[l]) for l in range(16)]
    assert ingress.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    held.clear();variances.clear();started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if index==0:
            spec.loader.exec_module(mod)
            original_scalar=mod.nonlinear_scalar;original_step=mod.gnn_step;original_gaussian=mod.gaussian_raw
            def scalar(mu,v,k3,k4):
                answer=original_scalar(mu,v,k3,k4);held['host']=answer[0];return answer
            def step(l,x,sigma,W,state,P):
                answer=original_step(l,x,sigma,W,state,P)
                if l==15:held.update(hidden=answer[1],delta=answer[0],sigma=sigma,P=P)
                return answer
            def gaussian(mu,v):
                variances.append(v);return original_gaussian(mu,v)
            mod.nonlinear_scalar=scalar;mod.gnn_step=step;mod.gaussian_raw=gaussian;estimator=mod.Estimator()
        prediction=estimator.predict(mlp,2**41)
    wall=time.monotonic()-started;bill=int(ctx.summary_dict()['flops_used'])
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    assert set(held)=={'host','hidden','delta','sigma','P'} and len(variances)==16
    minima=[]
    for v in variances:
        x=np.asarray(v);assert x.shape==(1024,) and np.isfinite(x).all() and np.min(x)>=1e-10
        minima.append(float(np.min(x)))
    a={k:np.asarray(v).copy() for k,v in held.items() if k!='P'}
    for k in ('head','head_b','rms','gain'):a[k]=np.asarray(held['P'][k]).copy()[15]
    a['prediction']=np.asarray(prediction).copy()
    assert a['prediction'].shape==(16,1024) and a['prediction'].dtype==np.float32
    assert a['hidden'].shape==(1024,64) and (a['prediction']>=0).all() and all(np.isfinite(v).all() for v in a.values())
    reconstructed=np.maximum(a['host'].astype(float)+a['delta'].astype(float),0).astype(np.float32)
    assert np.array_equal(reconstructed,a['prediction'][-1])
    pp=out/f'call{index:02d}_rank{rank:04d}.npy';np.save(pp,a['prediction'])
    cp=out/f'capture{index:02d}_rank{rank:04d}.npz';np.savez(cp,**a)
    row=dict(index=index,rank=rank,prediction=pin(pp),capture=pin(cp),weights=wp,flops=bill,
        multiplier=max(.1,bill/2**41),wall_s=wall,includes_initialization=index==0,bank=args.bank,
        numeric_cache_retired=True,truth_accessed=False,passive_extra_fnp_ops=0,unfloored_variance_minima=minima,
        final_host_plus_delta_exact=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/f'call{index:02d}.json',row);records.append(row)
    print(dict(index=index,rank=rank,flops=bill,wall_s=wall,bank=args.bank),flush=True)
    held.clear();variances.clear();del prediction,a,w,ww,mlp,reconstructed,v,x
write(out/'result.json',dict(completed=True,records=records,source=pin(source),truth_accessed=False,
    capture_additional_fnp_operations=0,full_estimator_calls=len(records)))

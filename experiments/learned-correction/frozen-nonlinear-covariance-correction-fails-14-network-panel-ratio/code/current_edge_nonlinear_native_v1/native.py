"""Native measurement with passive source/variance references only."""
import argparse,importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--ranks',required=True);ap.add_argument('--bank-parameters',action='store_true');args=ap.parse_args()
seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
b=read(HERE/'build.json');verify(b['candidate']);verify(b['bank_literal']);meta=read(HERE/'inputs.json')
bank_literal=read(b['bank_literal']['path'])
out=HERE/args.out;out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('adapted_edge_candidate',b['candidate']['path']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
held={};variances=[];records=[];indices=np.arange(16)*67;ix=np.ix_(indices,indices)
for index,rank in enumerate(map(int,args.ranks.split(','))):
    wp=meta['weights'][str(rank)];verify(wp);w=np.load(wp['path'],allow_pickle=False)
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(w[l]) for l in range(16)]
    assert ingress.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    held.clear();variances.clear();started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if index==0:
            spec.loader.exec_module(mod)
            if args.bank_parameters:mod._EDGE_LITERAL=bank_literal
            assert mod._EDGE_ENABLED and mod._EDGE_PARAMETERS is None
            original=mod._edge_source;raw=mod.gaussian_raw
            def source(layer,m,K21,aux):
                assert layer not in held
                answer=original(layer,m,K21,aux)
                held[layer]=dict(m=m,K21=K21,source=answer,**{k:aux[k] for k in ['sigma','alpha','pdf','gate','pre_k3','pre_k4','pre_cov']})
                return answer
            def gaussian(mu,v):
                variances.append(v)
                return raw(mu,v)
            mod._edge_source=source;mod.gaussian_raw=gaussian;estimator=mod.Estimator()
        prediction=estimator.predict(mlp,2**41)
    wall=time.monotonic()-started;bill=int(ctx.summary_dict()['flops_used'])
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    assert set(held)==set(range(8,15)) and len(variances)==16
    variance_minima=[]
    for v in variances:
        x=np.asarray(v);assert x.shape==(1024,) and np.isfinite(x).all() and np.min(x)>=1e-10
        variance_minima.append(float(np.min(x)))
    a=np.asarray(prediction).copy();assert a.shape==(16,1024) and a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
    pp=out/f'call{index:02d}_rank{rank:04d}.npy';np.save(pp,a)
    arrays={'indices':indices};checks=[]
    for layer,fields in held.items():
        for k,value in fields.items():
            aa=np.asarray(value);assert np.isfinite(aa).all()
            arrays[f'{k}_{layer:02d}']=(aa[ix] if aa.ndim==2 else aa[indices]).copy()
        full=np.asarray(fields['source']);assert full.shape==(1024,1024) and full.dtype==np.float64 and np.array_equal(full,full.T)
        checks.append(dict(layer=layer,full_source_finite_and_symmetric=True,source_mse=float(np.mean(full*full))))
    cp=out/f'local{index:02d}_rank{rank:04d}.npz';np.savez(cp,**arrays)
    row=dict(index=index,rank=rank,prediction=pin(pp),capture=pin(cp),weights=wp,flops=bill,
        multiplier=max(.1,bill/2**41),wall_s=wall,includes_initialization=index==0,bank_parameters=args.bank_parameters,
        numeric_cache_retired=True,truth_accessed=False,source_checks=checks,unfloored_variance_minima=variance_minima,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/f'call{index:02d}.json',row);records.append(row)
    print(dict(index=index,rank=rank,flops=bill,wall_s=wall,bank_parameters=args.bank_parameters),flush=True)
    held.clear();variances.clear();del prediction,a,w,ww,mlp,arrays,full,fields,aa,value,v,x
write(out/'result.json',dict(completed=True,records=records,candidate=b['candidate'],truth_accessed=False,
    capture_additional_fnp_operations=0,full_estimator_calls=len(records),saved_source_indices=indices.tolist()))

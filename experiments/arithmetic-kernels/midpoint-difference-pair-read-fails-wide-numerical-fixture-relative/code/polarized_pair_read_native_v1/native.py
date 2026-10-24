import argparse,importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--ranks',required=True);ap.add_argument('--disabled',action='store_true');args=ap.parse_args()
seal_check();done('unit');assert read(HERE/'unit.json')['passed'];assert all(t['num_threads']==1 for t in threadpool_info())
build=read(HERE/'build.json');verify(build['candidate']);meta=read(HERE/'inputs.json');out=HERE/args.out;out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
sp=importlib.util.spec_from_file_location('polar_native',HERE/'candidate.py');mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod
original_helper=original_gaussian=None;variances=[];calls=[];records=[]
try:
    for index,rank in enumerate(map(int,args.ranks.split(','))):
        wp=meta['weights'][str(rank)];verify(wp);weights=np.load(wp['path'],allow_pickle=False)
        assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(w) for w in weights]
        assert ingress.summary_dict()['flops_used']==0
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}');calls.clear();variances.clear();start=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if index==0:
                sp.loader.exec_module(mod);mod._POLAR_ENABLED=not args.disabled
                original_helper=mod._polar_pair_product;original_gaussian=mod.gaussian_raw
                def observe_helper(left,right,lam,cohorts,start,stop,product):
                    calls.append(dict(left=list(left.shape),right=list(right.shape),start=start,stop=stop,cohorts=[list(c) for c in cohorts],kernel=product.__name__))
                    return original_helper(left,right,lam,cohorts,start,stop,product)
                def observe_gaussian(mu,v):variances.append(v);return original_gaussian(mu,v)
                mod._polar_pair_product=observe_helper;mod.gaussian_raw=observe_gaussian;estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        wall=time.monotonic()-start;bill=int(ctx.summary_dict()['flops_used'])
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert len(variances)==16 and (not calls if args.disabled else len(calls)>14)
        minima=[]
        for v in variances:
            x=np.asarray(v);assert x.shape==(1024,) and np.isfinite(x).all() and np.min(x)>=1e-10;minima.append(float(np.min(x)))
        a=np.asarray(prediction).copy();assert a.shape==(16,1024) and a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
        pp=out/f'call{index:02d}_rank{rank:04d}.npy';np.save(pp,a)
        record=dict(index=index,rank=rank,weights=wp,prediction=pin(pp),flops=bill,multiplier=max(.1,bill/2**41),
            includes_initialization=index==0,disabled=args.disabled,wall_s=wall,numeric_cache_retired=True,
            unfloored_variance_minima=minima,helper_calls=list(calls),targets_read=False,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(out/f'call{index:02d}.json',record);records.append(record)
        print(dict(rank=rank,index=index,flops=bill,wall_s=wall,helper_calls=len(calls)),flush=True)
        variances.clear();del prediction,a,weights,ww,mlp
finally:
    if original_helper is not None:mod._polar_pair_product=original_helper;mod.gaussian_raw=original_gaussian
write(out/'result.json',dict(completed=True,records=records,candidate=build['candidate'],passive_observers_restored=True,
    capture_additional_fnp_operations=0,full_native_calls=len(records),targets_read=False))

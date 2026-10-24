"""Actual native calls, with passive history/source references and no new FNP arithmetic."""
import argparse,importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--ranks',required=True);ap.add_argument('--disabled',action='store_true');args=ap.parse_args()
seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
b=read(HERE/'build.json');verify(b['candidate']);meta=read(HERE/'inputs.json');out=HERE/args.out;out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('history_candidate_native',b['candidate']['path']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
hist={};sources={};variances=[];records=[];indices=np.arange(16)*67;ix=np.ix_(indices,indices);current_layer=None
original_gnn=original_source=original_gaussian=None
try:
    for index,rank in enumerate(map(int,args.ranks.split(','))):
        wp=meta['weights'][str(rank)];verify(wp);w=np.load(wp['path'],allow_pickle=False)
        assert w.shape==(16,1024,1024) and w.dtype==np.float32
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(v) for v in w]
        assert ingress.summary_dict()['flops_used']==0
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
        hist.clear();sources.clear();variances.clear();current_layer=None;started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if index==0:
                spec.loader.exec_module(mod);mod._HISTORY_ENABLED=not args.disabled
                assert mod._HISTORY_PARAMETERS is None
                original_gnn=mod.gnn_step;original_source=mod._history_covariance;original_gaussian=mod.gaussian_raw
                def observe_gnn(l,x,sigma,W,state,P):
                    global current_layer
                    answer=original_gnn(l,x,sigma,W,state,P)
                    if mod._HISTORY_PARAMETERS is not None and P is mod._HISTORY_PARAMETERS['P']:
                        assert l not in hist
                        hist[l]=dict(x=x,sigma=sigma,before=state,after=answer[1]);current_layer=l
                    return answer
                def observe_source(state,aux):
                    answer=original_source(state,aux);assert current_layer not in sources
                    sources[current_layer]=dict(state=state,sigma=aux['sigma'],pre_cov=aux['pre_cov'],source=answer)
                    return answer
                def observe_gaussian(mu,v):variances.append(v);return original_gaussian(mu,v)
                mod.gnn_step=observe_gnn;mod._history_covariance=observe_source;mod.gaussian_raw=observe_gaussian
                estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        wall=time.monotonic()-started;bill=int(ctx.summary_dict()['flops_used'])
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert len(variances)==16
        if args.disabled:assert not hist and not sources and mod._HISTORY_PARAMETERS is None
        else:assert set(hist)==set(range(15)) and set(sources)==set(range(8,15))
        minima=[]
        for v in variances:
            a=np.asarray(v);assert a.shape==(1024,) and np.isfinite(a).all() and np.min(a)>=1e-10
            minima.append(float(np.min(a)))
        a=np.asarray(prediction).copy();assert a.shape==(16,1024) and a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
        pp=out/f'call{index:02d}_rank{rank:04d}.npy';np.save(pp,a)
        arrays={'indices':indices};checks=[]
        for l,fields in hist.items():
            for k,value in fields.items():
                z=np.asarray(value);assert np.isfinite(z).all();arrays[f'{k}_{l:02d}']=z.copy()
        for l,fields in sources.items():
            for k,value in fields.items():
                z=np.asarray(value);assert np.isfinite(z).all()
                arrays[f'kernel_{k}_{l:02d}']=(z[ix] if k in ('source','pre_cov') else z[indices]).copy()
            full=np.asarray(fields['source']);assert full.shape==(1024,1024) and full.dtype==np.float64
            asym=relative(full,full.T);assert asym<=3e-5
            checks.append(dict(layer=l,full_source_finite=True,symmetry_relative=asym,source_mse=float(np.mean(full*full))))
        cp=out/f'capture{index:02d}_rank{rank:04d}.npz';np.savez(cp,**arrays)
        row=dict(index=index,rank=rank,prediction=pin(pp),capture=pin(cp),weights=wp,flops=bill,
            multiplier=max(.1,bill/2**41),includes_initialization=index==0,disabled=args.disabled,
            wall_s=wall,numeric_cache_retired=True,ordinary_1B_targets_read=False,source_checks=checks,
            unfloored_variance_minima=minima,history_layers=sorted(hist),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(out/f'call{index:02d}.json',row);records.append(row)
        print(dict(index=index,rank=rank,flops=bill,wall_s=wall,disabled=args.disabled),flush=True)
        hist.clear();sources.clear();variances.clear();del prediction,a,w,ww,mlp,arrays
finally:
    if original_gnn is not None:
        mod.gnn_step=original_gnn;mod._history_covariance=original_source;mod.gaussian_raw=original_gaussian
write(out/'result.json',dict(completed=True,records=records,candidate=b['candidate'],ordinary_1B_targets_read=False,
    passive_observers_restored=True,capture_additional_fnp_operations=0,full_estimator_calls=len(records)))

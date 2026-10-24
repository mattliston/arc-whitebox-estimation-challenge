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
from common import *

ap=argparse.ArgumentParser()
ap.add_argument('--out',required=True);ap.add_argument('--ranks',required=True)
ap.add_argument('--disabled',action='store_true')
args=ap.parse_args()
seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
inputs=read(HERE/'inputs.json');meta={r['rank']:r for r in inputs['rows']}
for v in inputs['matching_contexts'].values():check(v['pin'])
build=read(HERE/'build.json');check(build['candidate'])
out=HERE/args.out;out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('radial_plane_observer_candidate',build['candidate']['path'])
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
records=[]


def rel(a,b):
    error=float(np.linalg.norm(a-b));norm=float(np.linalg.norm(b))
    return error/norm if norm else error


def digest(a):
    return __import__('hashlib').sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


for index,rank in enumerate(map(int,args.ranks.split(','))):
    row=meta[rank];check(row['weights']);check(row['prediction'])
    refs=inputs['matching_contexts']['cold100' if args.out=='cold100' else 'triple']['records']
    reference=refs[index]
    assert reference['rank']==rank and reference['includes_initialization']==(index==0)
    assert reference['weights_sha256']==row['weights']['sha256']
    w=np.load(row['weights']['path'],allow_pickle=False)
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(x) for x in w]
    assert ingress.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    costs=dict(source=0,advance=0,read=0)
    base_means=[];shifts=[];rays=[];source_checks=[];ray_checks=[];advance_checks=[]
    replay=dict(mean=None,ray=None)
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if index==0:
            spec.loader.exec_module(mod);mod._R22_ENABLED=not args.disabled
            original_source,original_ray=mod._r22_source,mod._r22_ray
            original_advance,original_read=mod._r22_advance,mod._r22_read
            def source(c,q):
                before=int(ctx.summary_dict()['flops_used']);answer=original_source(c,q)
                costs['source']+=int(ctx.summary_dict()['flops_used'])-before
                cn,qn=np.asarray(c),np.asarray(q)
                ss=(cn+cn.T)/2;qq=(qn+qn.T)/2;v=np.diag(ss)
                den=float(np.sum(v)**2+2*np.sum(ss*ss));assert den>0 and np.isfinite(den)
                radial=3*float(np.sum(qq))/den
                expected=qq-(radial/3)*(np.outer(v,v)+2*ss*ss)
                actual=np.asarray(answer)
                error=rel(actual.astype(float),expected);assert error<=2e-6,error
                assert actual.dtype==np.float32 and np.isfinite(actual).all()
                source_checks.append(dict(relative=error,source_sha256=digest(actual),
                    unrounded_double_trace=float(np.sum(expected)),reference_norm=float(np.linalg.norm(expected))))
                return answer
            def ray(d,w32):
                answer=original_ray(d,w32)
                dn=np.asarray(d).astype(float);wn=np.asarray(w32).astype(float);square=wn*wn
                expected=3*np.sum(square*(dn@square),axis=0)-2*np.sum(np.diag(dn)[:,None]*square**2,axis=0)
                actual=np.asarray(answer).copy();error=rel(actual,expected)
                assert error<=3e-5 and np.isfinite(actual).all(),error
                replay['ray']=expected
                rays.append(actual);ray_checks.append(dict(relative=error,sha256=digest(actual)))
                return answer
            def advance(state,w32,w64,aux):
                before=int(ctx.summary_dict()['flops_used'])
                answer=original_advance(state,w32,w64,aux)
                costs['advance']+=int(ctx.summary_dict()['flops_used'])-before
                if state['plane'] is None:
                    assert state['mean'] is None and replay['mean'] is None
                    return answer
                a,p,s,g=(np.asarray(aux[k]) for k in ['alpha','pdf','sigma','gate'])
                expected=(a*a-1)*p*replay['ray']/(24*s**3)
                if replay['mean'] is not None:expected+=g*(np.asarray(w64).T@replay['mean'])
                replay['mean']=expected
                actual=np.asarray(state['mean']);error=rel(actual,expected)
                assert error<=3e-5 and np.isfinite(actual).all(),error
                advance_checks.append(dict(relative=error,shift_sha256=digest(actual)))
                return answer
            def output(mean,state):
                before=int(ctx.summary_dict()['flops_used']);answer=original_read(mean,state)
                costs['read']+=int(ctx.summary_dict()['flops_used'])-before
                base_means.append(np.asarray(mean).copy())
                shifts.append(np.zeros_like(base_means[-1]) if state['mean'] is None else np.asarray(state['mean']).copy())
                return answer
            mod._r22_source,mod._r22_ray=source,ray
            mod._r22_advance,mod._r22_read=advance,output
            estimator=mod.Estimator()
        result=estimator.predict(mlp,2**41)
    elapsed=time.monotonic()-started;bill=int(ctx.summary_dict()['flops_used'])
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    pred=np.asarray(result).copy();base=np.maximum(np.stack(base_means),0).astype(np.float32)
    parent=np.load(row['prediction']['path'],allow_pickle=False)
    assert len(base_means)==16 and base.tobytes()==parent.tobytes()
    assert pred.shape==(16,1024) and pred.dtype==np.float32 and np.isfinite(pred).all() and (pred>=0).all()
    assert pred[0].tobytes()==parent[0].tobytes()
    shift=np.stack(shifts);assert shift.shape==(16,1024) and np.isfinite(shift).all()
    if args.disabled:
        assert pred.tobytes()==parent.tobytes() and bill==reference['flops']
        assert costs['source']==costs['advance']==0 and not np.any(shift)
        extra=0;baseline_read=costs['read']
    else:
        baseline_read=read(HERE/'disabled/result.json')['records'][0]['helper_costs']['read']
        extra=costs['source']+costs['advance']+costs['read']-baseline_read
        assert bill-extra==reference['flops'],(bill,extra,reference['flops'],costs,baseline_read)
        assert 0<extra<=.10*reference['flops']
        assert len(source_checks)==len(ray_checks)==len(advance_checks)==15
    pp=out/f'call{index:02d}_rank{rank:04d}.npy';sp=out/f'call{index:02d}_shifts.npy'
    np.save(pp,pred);np.save(sp,shift)
    if rays:
        rp=out/f'call{index:02d}_rays.npy';np.save(rp,np.stack(rays));ray_pin=pin(rp)
    else:ray_pin=None
    record=dict(index=index,rank=rank,prediction=pin(pp),shifts=pin(sp),rays=ray_pin,weights=row['weights'],
        flops=bill,multiplier=max(.1,bill/2**41),wall_s_with_research_checks=elapsed,
        includes_initialization=index==0,input_flops=0,enabled=not args.disabled,
        underlying_all16_parent_bytes_exact=True,underlying_context_bill_exact=True,
        baseline_context_flops=reference['flops'],helper_costs=costs,baseline_read_cost=baseline_read,
        extra_flops=extra,extra_fraction=extra/reference['flops'],first_parent_row_exact=True,
        source_checks=source_checks,ray_checks=ray_checks,advance_checks=advance_checks,
        numeric_cache_retired=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/f'call{index:02d}.json',record);records.append(record)
    print(dict(rank=rank,index=index,enabled=not args.disabled,flops=bill,extra_flops=extra,
        wall_s=elapsed,peak_rss=record['peak_rss_bytes']),flush=True)
    del w,ww,mlp,result,pred,base,parent,shift
write(out/'result.json',dict(completed=True,records=records,candidate=build['candidate'],targets_decoded=False,
    research_comparisons_feed_candidate=False,production_runtime_qualified=False))


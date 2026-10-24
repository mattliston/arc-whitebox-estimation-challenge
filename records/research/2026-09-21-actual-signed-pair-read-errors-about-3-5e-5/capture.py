"""Observe one unchanged cold bank call; offline references never feed back."""
import importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
from reference import audit_read,norm,ratio,covariance_response
rank=int(sys.argv[1]);assert rank in (35,100)
seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
ref=next(x for x in read(HERE/'inputs.json')['rows'] if x['rank']==rank)
for key in ('weights','prediction','control','old_result','attribution'):verify(ref[key])
weights=np.load(ref['weights']['path'],allow_pickle=False);assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(x) for x in weights]
assert ingress.summary_dict()['flops_used']==0
mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('observed_pair_read',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
pending=None;calls=[];original_read=None;original_nonlinear=None;started=time.monotonic()
try:
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        spec.loader.exec_module(mod)
        original_read=mod._shared_old384_read;original_nonlinear=mod.nonlinear
        def observed_read(left,right,lam,live,state):
            global pending
            assert pending is None
            frame=sys._getframe(1);assert frame.f_code is mod._predict_binary.__code__
            layer=int(frame.f_locals['l']);response=np.asarray(frame.f_locals['response_indices']).copy()
            width=int(frame.f_locals['n']);del frame
            assert layer==1+len(calls)
            answer,row,delta=audit_read(original_read,left,right,lam,live,state,response,width,ctx)
            pending=dict(layer=layer,row=row,delta=delta)
            return answer
        def observed_nonlinear(*args,**kwargs):
            global pending
            assert pending is not None
            frame=sys._getframe(1);assert frame.f_code is mod._predict_binary.__code__
            layer=int(frame.f_locals['l']);del frame
            assert layer==pending['layer']
            original=original_nonlinear(*args,**kwargs)
            after=int(ctx._flops_used)
            total=np.array(args[2],dtype=np.float64,copy=True);np.fill_diagonal(total,0)
            fields,unused=covariance_response(pending['delta'],original[-1],original[1]);del unused
            total_norm=norm(total);del total
            row=pending['row'];row.update(layer=layer,full_K21_offdiag_norm=total_norm,
                effective_error_over_full_K21=ratio(row['effective_error_norm'],total_norm),**fields)
            assert row['effective_error_over_full_K21'] is not None
            row['exceeds_numerical_tolerance']=row['effective_error_over_full_K21']>3e-5
            assert int(ctx._flops_used)==after
            calls.append(row);pending=None
            print(dict(rank=rank,layer=layer,read_error=row['effective_read_relative_error'],
                       full_K21_error=row['effective_error_over_full_K21'],covariance_error=row['local_covariance_relative']),flush=True)
            return original
        mod._shared_old384_read=observed_read;mod.nonlinear=observed_nonlinear
        result=mod.Estimator().predict(mlp,2**41)
finally:
    if original_read is not None:mod._shared_old384_read=original_read
    if original_nonlinear is not None:mod.nonlinear=original_nonlinear
assert pending is None and [r['layer'] for r in calls]==list(range(1,15))
assert mod._shared_old384_read is original_read and mod.nonlinear is original_nonlinear
bill=int(ctx.summary_dict()['flops_used']);assert bill==ref['flops'],(bill,ref['flops'])
assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
prediction=np.asarray(result).copy();assert prediction.shape==(16,1024) and prediction.dtype==np.float32
assert prediction.tobytes()==np.load(ref['prediction']['path'],allow_pickle=False).tobytes()
assert np.isfinite(prediction).all() and (prediction>=0).all()
out=HERE/f'rank{rank:04d}';out.mkdir(exist_ok=False);np.save(out/'prediction.npy',prediction)
assert sha(out/'prediction.npy')==ref['prediction']['sha256']
write(out/'trace.json',dict(rank=rank,calls=calls,full_flops=bill,reference_arithmetic_unmetered_research_only=True,
    all16_bytes_and_complete_bill_exact=True,targets_read=False,wide_arrays_saved=False))
write(out/'result.json',dict(completed=True,rank=rank,source=pin(SOURCE),reference=ref,
    prediction=pin(out/'prediction.npy'),trace=pin(out/'trace.json'),full_flops=bill,read_calls=len(calls),
    all16_bytes_and_complete_bill_exact=True,observers_restored=True,numeric_cache_retired=True,
    targets_read=False,candidate_created=False,diagnostic_wall_is_not_production_wall=True,
    wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=threadpool_info()))
print(dict(completed=True,rank=rank,read_calls=len(calls),bill=bill,wall_s=time.monotonic()-started),flush=True)

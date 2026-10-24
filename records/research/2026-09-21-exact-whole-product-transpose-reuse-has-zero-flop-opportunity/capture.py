"""Read-only operand fingerprints around the exact native _mm calls."""
import importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
from identity import fingerprint
rank=int(sys.argv[1]);assert rank in (35,100);seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
ref=next(x for x in read(HERE/'inputs.json')['rows'] if x['rank']==rank)
for k in ('weights','prediction','control','old_result','attribution'):verify(ref[k])
expected=read(ref['attribution']['path'])['outermost_mm_calls']
weights=np.load(ref['weights']['path'],allow_pickle=False);assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(x) for x in weights]
assert ingress.summary_dict()['flops_used']==0
mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('observed_product_identities',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
calls=[];depth=0;original_mm=None;started=time.monotonic()
try:
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        spec.loader.exec_module(mod);original_mm=mod._mm
        def observed(left,right):
            global depth
            outer=depth==0;before=ctx._flops_used
            if outer:
                caller=sys._getframe(1);origin=[caller.f_code.co_name,caller.f_lineno];frame=caller;layer=-1
                while frame is not None:
                    if frame.f_code.co_filename==str(SOURCE) and frame.f_code.co_name=='_predict_binary':layer=int(frame.f_locals.get('l',-1))
                    frame=frame.f_back
                del caller,frame
                a=np.asarray(left);b=np.asarray(right)
                fields=dict(a=fingerprint(a),b=fingerprint(b),at=fingerprint(a.T),bt=fingerprint(b.T))
                del a,b
            depth+=1
            try:answer=original_mm(left,right)
            finally:depth-=1
            if outer:
                bill=int(ctx._flops_used-before);a=np.asarray(answer)
                fields.update(c=fingerprint(a),ct=fingerprint(a.T));del a
                old=dict(left=list(left.shape),right=list(right.shape),flops=bill,input_dtypes=[str(left.dtype),str(right.dtype)],caller=origin)
                index=len(calls);assert old==expected[index],(rank,index,old,expected[index])
                assert fields['a']['shape'][1]==fields['b']['shape'][0]
                assert fields['c']['shape']==[fields['a']['shape'][0],fields['b']['shape'][1]]
                calls.append(dict(index=index,layer=layer,**old,**fields))
                if len(calls)%50==0:print(dict(rank=rank,observed_calls=len(calls),layer=layer),flush=True)
            return answer
        mod._mm=observed
        result=mod.Estimator().predict(mlp,2**41)
finally:
    if original_mm is not None:mod._mm=original_mm
assert depth==0 and len(calls)==295 and mod._mm is original_mm
bill=int(ctx.summary_dict()['flops_used']);assert bill==ref['flops']
assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
prediction=np.asarray(result).copy();assert prediction.shape==(16,1024) and prediction.dtype==np.float32
assert prediction.tobytes()==np.load(ref['prediction']['path'],allow_pickle=False).tobytes()
assert np.isfinite(prediction).all() and (prediction>=0).all()
out=HERE/f'rank{rank:04d}';out.mkdir(exist_ok=False);np.save(out/'prediction.npy',prediction)
assert sha(out/'prediction.npy')==ref['prediction']['sha256']
write(out/'trace.json',dict(rank=rank,calls=calls,full_flops=bill,outer_mm_flops=sum(x['flops'] for x in calls),
    matches_old_census_row_for_row=True,targets_read=False,wide_arrays_saved=False,
    hashed_payload_bytes=sum(x[k]['bytes'] for x in calls for k in ('a','b','at','bt','c','ct'))))
write(out/'result.json',dict(completed=True,rank=rank,source=pin(SOURCE),weights=ref['weights'],reference=ref,
    prediction=pin(out/'prediction.npy'),trace=pin(out/'trace.json'),full_flops=bill,outer_mm_calls=len(calls),
    all16_bytes_and_complete_bill_exact=True,all295_shape_dtype_caller_bills_exact=True,all_operands_finite=True,
    observer_restored=True,numeric_cache_retired=True,targets_read=False,candidate_created=False,
    observer_wall_is_not_production_wall=True,wall_s=time.monotonic()-started,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=threadpool_info()))
print(dict(completed=True,rank=rank,calls=len(calls),bill=bill,wall_s=time.monotonic()-started),flush=True)

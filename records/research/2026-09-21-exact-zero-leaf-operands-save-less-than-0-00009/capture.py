"""Passive exact-zero census of original charged leaf operands."""
import collections,importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget
from threadpoolctl import threadpool_info
from common import *
from counts import count_terms
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
spec=importlib.util.spec_from_file_location('observed_leaf_zeros',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
outer=[];leaves=[];depth=0;active_index=None;original_mm=None;original_charge=_budget.BudgetContext._charge_op
started=time.monotonic()
def charged(ctx,*args,**kwargs):
    before=ctx._flops_used;answer=original_charge(ctx,*args,**kwargs);delta=int(ctx._flops_used-before)
    if kwargs.get('op_name',args[0] if args else None)!='matmul':return answer
    frame=sys._getframe(1);chosen=None;site=None;inside=False;layer=-1
    while frame is not None:
        if frame.f_code.co_filename==str(SOURCE):
            key=(frame.f_code.co_name,frame.f_lineno)
            if key in SITES:
                assert chosen is None
                names=SITES[key];chosen=(frame.f_locals[names[0]],frame.f_locals[names[1]]);site=key
            if frame.f_code.co_name=='_mm':inside=True
            if frame.f_code.co_name=='_predict_binary':layer=int(frame.f_locals.get('l',-1))
        frame=frame.f_back
    del frame
    if chosen is not None:
        assert inside and depth>0 and active_index is not None
        counts=count_terms(*chosen);del chosen
        assert counts['dense_flops']==delta,(site,counts,delta)
        leaves.append(dict(index=len(leaves),outer_index=active_index,layer=layer,function=site[0],line=site[1],actual_flops=delta,**counts))
    return answer
try:
    _budget.BudgetContext._charge_op=charged
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        spec.loader.exec_module(mod);original_mm=mod._mm
        def observed(left,right):
            global depth,active_index
            top=depth==0;before=ctx._flops_used
            if top:
                assert active_index is None;active_index=len(outer)
                caller=sys._getframe(1);origin=[caller.f_code.co_name,caller.f_lineno];del caller
            depth+=1
            try:answer=original_mm(left,right)
            finally:depth-=1
            if top:
                row=dict(left=list(left.shape),right=list(right.shape),flops=int(ctx._flops_used-before),
                    input_dtypes=[str(left.dtype),str(right.dtype)],caller=origin)
                assert row==expected[active_index],(rank,active_index,row,expected[active_index])
                outer.append(row);active_index=None
                if len(outer)%50==0:print(dict(rank=rank,outer_calls=len(outer),leaf_calls=len(leaves)),flush=True)
            return answer
        mod._mm=observed
        result=mod.Estimator().predict(mlp,2**41)
finally:
    _budget.BudgetContext._charge_op=original_charge
    if original_mm is not None:mod._mm=original_mm
assert depth==0 and active_index is None and len(outer)==295
assert mod._mm is original_mm and _budget.BudgetContext._charge_op is original_charge
bill=int(ctx.summary_dict()['flops_used']);assert bill==ref['flops']
assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
prediction=np.asarray(result).copy();assert prediction.shape==(16,1024) and prediction.dtype==np.float32
assert prediction.tobytes()==np.load(ref['prediction']['path'],allow_pickle=False).tobytes()
assert np.isfinite(prediction).all() and (prediction>=0).all()
site_bills=collections.Counter()
for row in leaves:site_bills[(row['function'],row['line'])]+=row['actual_flops']
old=next(x for x in read(read(HERE/'inputs.json')['previous_bound']['path'])['rows'] if x['rank']==rank)
assert dict(site_bills)=={(x['function'],x['line']):x['flops'] for x in old['site_bills']}
assert sum(site_bills.values())==old['retained']
out=HERE/f'rank{rank:04d}';out.mkdir(exist_ok=False);np.save(out/'prediction.npy',prediction)
assert sha(out/'prediction.npy')==ref['prediction']['sha256']
write(out/'trace.json',dict(rank=rank,leaves=leaves,outer_calls=outer,full_flops=bill,
    site_bills=[dict(function=k[0],line=k[1],flops=v) for k,v in sorted(site_bills.items())],
    all_leaf_bills_match_previous_census=True,all_outer_records_exact=True,targets_read=False,wide_arrays_saved=False))
write(out/'result.json',dict(completed=True,rank=rank,source=pin(SOURCE),reference=ref,prediction=pin(out/'prediction.npy'),
    trace=pin(out/'trace.json'),full_flops=bill,leaf_calls=len(leaves),outer_calls=len(outer),all16_bytes_and_complete_bill_exact=True,
    observer_restored=True,numeric_cache_retired=True,all_leaf_operands_finite=True,targets_read=False,candidate_created=False,
    observer_wall_is_not_production_wall=True,wall_s=time.monotonic()-started,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=threadpool_info()))
print(dict(completed=True,rank=rank,leaf_calls=len(leaves),bill=bill,wall_s=time.monotonic()-started),flush=True)

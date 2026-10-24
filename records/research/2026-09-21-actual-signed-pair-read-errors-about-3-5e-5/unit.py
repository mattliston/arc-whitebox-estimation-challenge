import importlib.util,resource,sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
from reference import snapshot,reference,scalar_reference,ratio,norm,measure,covariance_response,audit_read
seal_check();assert all(x['num_threads']==1 for x in threadpool_info())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('pair_read_units',SOURCE);mod=importlib.util.module_from_spec(spec)
with flops.BudgetContext(flop_budget=2**41,quiet=True):spec.loader.exec_module(mod)
rng=np.random.default_rng(90890);rows=[]
for mode,young,zero,strided in [('young_only',12,False,False),('materialized',6,False,False),('lazy',6,False,False),
                               ('materialized',0,False,True),('lazy',0,True,True),('young_only',0,True,False)]:
    old=0 if mode=='young_only' else 10;width=old+young;m,n,rank=7,9,384
    left=rng.standard_normal((m,width)).astype(np.float32)*.1
    right=rng.standard_normal((n,width)).astype(np.float32)*.1
    if strided:left=left[::-1];right=right[::-1]
    lam=rng.standard_normal(width//2).astype(np.float32);lam=np.concatenate([lam,-lam])
    if zero:lam[:]=0
    live=np.arange(n,dtype=np.int64);response=np.arange(m,dtype=np.int64)
    state=dict(columns=old,lazy=None,F=None)
    if old:
        state['pre_basis']=(rng.standard_normal((n,rank))*.03).astype(np.float32)
        if mode=='materialized':state['F']=(rng.standard_normal((rank,old))*.1).astype(np.float32)
        else:state['lazy']=dict(layer=8,prefix=4,joined=6,base=(rng.standard_normal((rank,4))*.1).astype(np.float32),
            join=(rng.standard_normal((rank,6))*.1).astype(np.float32),map=(rng.standard_normal((rank,rank))*.02).astype(np.float32))
    owned=snapshot(left,right,lam,live,state)
    before=[v.tobytes() for v in owned['arrays'].values()]
    ref=reference(owned);blocked=reference(owned,3);scalar=scalar_reference(owned)
    for candidate in (blocked,scalar):
        rel=ratio(norm(candidate-ref),norm(ref));assert rel is not None and rel<=3e-11,(mode,rel)
    assert np.array_equal(ref,reference(owned)) and before==[v.tobytes() for v in owned['arrays'].values()]
    # Mutating caller storage cannot change the saved reference.
    lcopy=left.copy();lcopy[:]=8
    assert not np.shares_memory(owned['arrays']['left'],left)
    original_left=left.copy();left[:]=1;assert np.array_equal(ref,reference(owned));left[:]=original_left
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        args=[fnp.asarray(x) for x in (left,right,lam,live)]
        converted=dict(state)
        for name in ('F','pre_basis'):
            if isinstance(state.get(name),np.ndarray):converted[name]=fnp.asarray(state[name])
        if state['lazy'] is not None:converted['lazy']={k:fnp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in state['lazy'].items()}
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as plain_ctx:
        plain=mod._shared_old384_read(*args,converted)
    plain_bytes=np.asarray(plain).tobytes();calls=[]
    def original(*x):
        output=mod._shared_old384_read(*x);calls.append(output);return output
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as watched_ctx:
        answer,row,delta=audit_read(original,*args,converted,response,n,watched_ctx)
    assert len(calls)==1 and answer is calls[0]
    assert np.asarray(answer).tobytes()==plain_bytes
    assert watched_ctx.summary_dict()['flops_used']==plain_ctx.summary_dict()['flops_used']==row['original_read_flops']
    assert not np.any(np.diag(delta))
    assert before==[v.tobytes() for v in owned['arrays'].values()]
    rows.append(dict(young_columns=young,zero=zero,strided=strided,independent_scalar_passed=True,
        original_bytes_and_bill_exact=True,snapshot_independent=True,**row))
# Real parent nonlinear map: fixed-diagonal K21 perturbations are affine in cov.
n=7;mu=(rng.standard_normal(n)*.1).astype(np.float64)
C=np.eye(n)*.2+np.ones((n,n))*.03
K=rng.standard_normal((n,n))*.001
delta=rng.standard_normal((n,n))*.0003;np.fill_diagonal(delta,0)
k4=np.zeros(n);k22=np.zeros((n,n))
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    args=[fnp.asarray(x) for x in (mu,C,K,k4,k22)]
    original=mod.nonlinear(*args)
    perturbed=mod.nonlinear(args[0],args[1],fnp.asarray(K+delta),args[3],args[4])
fields,dc=covariance_response(delta,original[-1],original[1])
actual=np.asarray(perturbed[1])-np.asarray(original[1])
assert norm(actual-dc)<=1e-12 and np.array_equal(np.asarray(original[0]),np.asarray(perturbed[0]))
assert np.array_equal(np.diag(np.asarray(original[1])),np.diag(np.asarray(perturbed[1])))
write(HERE/'unit.json',dict(passed=True,fixtures=rows,independent_scalar_fsum_reference=True,
    snapshot_mutation_lifetime_passed=True,parent_return_identity_and_bill_passed=True,
    nonlinear_affinity_maxabs=float(np.max(np.abs(actual-dc))),nonlinear_affinity_passed=True,
    targets_read=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,fixtures=len(rows),affinity_maxabs=float(np.max(np.abs(actual-dc)))),flush=True)

import importlib.util,resource,sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
seal_check();assert all(x['num_threads']==1 for x in threadpool_info())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
sp=importlib.util.spec_from_file_location('polar_unit',HERE/'candidate.py');mod=importlib.util.module_from_spec(sp)
with flops.BudgetContext(flop_budget=2**41,quiet=True):sp.loader.exec_module(mod)
rng=np.random.default_rng(90889)
def fixture(m,n,pairs,epsilon):
    ll=[];rr=[];coeff=[];cohorts=[];start=0
    for p in pairs:
        x=rng.standard_normal((m,p)).astype(np.float32);y=(x+epsilon*rng.standard_normal((m,p))).astype(np.float32)
        u=rng.standard_normal((n,p)).astype(np.float32);v=(u+epsilon*rng.standard_normal((n,p))).astype(np.float32)
        a=rng.uniform(-1.5,1.5,p).astype(np.float32)
        ll += [x,y];rr += [u,v];coeff += [a,-a];cohorts.append((start,2*p));start+=2*p
    return np.concatenate(ll,axis=1),np.concatenate(rr,axis=1),np.concatenate(coeff),cohorts
def direct64(left,right,lam):
    l=left.astype(np.float64);r=right.astype(np.float64);a=lam.astype(np.float64)
    return (l*l)@(r*a).T
def scalar64(left,right,lam,cohorts):
    result=np.zeros((len(left),len(right)),np.float64)
    for i in range(len(left)):
        for j in range(len(right)):
            terms=[]
            for start,width in cohorts:
                p=width//2
                for k in range(start,start+p):
                    x,y=float(left[i,k]),float(left[i,k+p]);u,v=float(right[j,k]),float(right[j,k+p])
                    terms.append(float(lam[k])*(x*x*u-y*y*v))
            import math
            result[i,j]=math.fsum(terms)
    return result
rows=[]
def check(left,right,lam,cohorts,wide=False):
    before=[x.tobytes() for x in (left,right,lam)]
    with flops.BudgetContext(flop_budget=2**41,quiet=True):args=[fnp.asarray(x) for x in (left,right,lam)]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as c:
        out=mod._polar_pair_product(*args,cohorts,0,len(lam),mod._mm)
    actual=np.asarray(out).copy();dense=direct64(left,right,lam)
    cancelling=True
    for start,width in cohorts:
        p=width//2
        cancelling=cancelling and (not np.any(lam[start:start+p]) or
            (np.array_equal(left[:,start:start+p],left[:,start+p:start+width]) and
             np.array_equal(right[:,start:start+p],right[:,start+p:start+width])))
    if cancelling:
        assert not np.any(actual) and not np.any(scalar64(left,right,lam,cohorts))
        scale=(np.abs(left.astype(np.float64))**2)@(np.abs(right.astype(np.float64))*np.abs(lam.astype(np.float64))).T
        assert np.max(np.abs(dense))<=128*np.finfo(np.float64).eps*max(1.,float(np.max(scale)))
        ref=np.zeros_like(dense);er=None
    else:
        ref=dense;er=relative(actual,ref)
        assert er<=3e-5,(left.shape,right.shape,er)
        if not wide:assert relative(ref,scalar64(left,right,lam,cohorts))<=3e-11
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as repeat_c:
        repeated=mod._polar_pair_product(*args,cohorts,0,len(lam),mod._mm)
    assert actual.tobytes()==np.asarray(repeated).tobytes()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as old_c:
        original=mod._mm(args[0]*args[0],(args[1]*args[2]).T)
    old_error=relative(np.asarray(original),ref) if np.any(ref) else None
    assert before==[x.tobytes() for x in (left,right,lam)]
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        perm=mod._polar_pair_product(fnp.asarray(left[::-1].copy()),fnp.asarray(right[::-1].copy()),args[2],cohorts,0,len(lam),mod._mm)
    assert relative(np.asarray(perm),actual[::-1,::-1])<=3e-5 or not np.any(actual)
    row=dict(left=list(left.shape),right=list(right.shape),relative=er,original_relative=old_error,
        reference_exact_zero=cancelling,dense64_roundoff_norm=float(np.linalg.norm(dense)) if cancelling else None,
        polarized_flops=c.summary_dict()['flops_used'],repeat_flops=repeat_c.summary_dict()['flops_used'],original_flops=old_c.summary_dict()['flops_used'],
        inputs_preserved=True,repeat_exact=True,permutation_passed=True)
    rows.append(row);return row
for m,n,pp,eps in ((7,9,[3],.7),(17,11,[2,5],.1),(9,5,[4,3],2**-12),(7,7,[3],0.)):
    check(*fixture(m,n,pp,eps))
z=fixture(5,7,[3,2],.2);z[2][:]=0;check(*z)
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    arena=fnp.empty((mod._PACKED_ELEMENTS,),dtype=fnp.float32)
    mod._SCRATCH=[arena,mod._cp3_bind_arena_views(arena,mod._PACKED_PUSH),mod._cp3_bind_arena_views(arena,mod._PACKED_READ),None,None,None,None]
    mod._bind_push6(arena)
wide=check(*fixture(512,512,[512,512],2**-10),wide=True)
assert wide['relative']<wide['original_relative'],wide
ownership=[]
left,right,lam,cohorts=fixture(7,384,[2,4],.2)
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    l=fnp.asarray(left);r=fnp.asarray(right);a=fnp.asarray(lam)
    state=dict(columns=12,F=r,lazy=None,cohorts=cohorts)
    materialized=mod._slr_inner(l,a,state)
ref=direct64(left,right,lam);er=relative(np.asarray(materialized),ref);assert er<=3e-5
ownership.append(dict(mode='materialized',relative=er))
matrix=(rng.standard_normal((384,384))*.03).astype(np.float32)
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    M=fnp.asarray(matrix)
    state=dict(columns=12,F=None,lazy=dict(layer=min(mod._SLR_LAZY_LAYERS),prefix=4,joined=8,base=r[:,:4],join=r[:,4:],map=M),cohorts=cohorts)
    lazy=mod._slr_inner(l,a,state)
ref=direct64(left[:,:4],right[:,:4],lam[:4])@matrix.astype(np.float64).T+direct64(left[:,4:],right[:,4:],lam[4:])
er=relative(np.asarray(lazy),ref);assert er<=3e-5;ownership.append(dict(mode='lazy',relative=er))
left,unused,lam,cohorts=fixture(7,9,[2,4,3],.1)
right=rng.standard_normal((9,18)).astype(np.float32);factor=rng.standard_normal((384,12)).astype(np.float32);basis=(rng.standard_normal((9,384))*.02).astype(np.float32)
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    state=dict(columns=12,F=fnp.asarray(factor),lazy=None,cohorts=cohorts,pre_basis=fnp.asarray(basis))
    out=mod._shared_old384_read(fnp.asarray(left),fnp.asarray(right),fnp.asarray(lam),fnp.arange(9),state)
ref=direct64(left[:,:12],factor,lam[:12])@basis.astype(np.float64).T+direct64(left[:,12:],right[:,12:],lam[12:])
er=relative(np.asarray(out),ref);assert er<=3e-5;ownership.append(dict(mode='complete_old_young',relative=er))
left,right,lam,cohorts=fixture(7,9,[3],.2);bad=lam.copy();bad[-1]+=np.float32(.1)
rejected=0
for lv,rv,cv,co,start,stop in [(left,right,bad,cohorts,0,6),(left,right,lam,[(1,6)],0,6),(left,right,lam,[],0,6),
 (left.astype(np.float64),right,lam,cohorts,0,6),(left[:,:4],right,lam,cohorts,0,6)]:
    try:
        with flops.BudgetContext(flop_budget=2**41,quiet=True):mod._polar_pair_product(fnp.asarray(lv),fnp.asarray(rv),fnp.asarray(cv),co,start,stop,mod._mm)
    except RuntimeError:rejected+=1
    else:raise AssertionError('Malformed pair accepted')
assert rejected==5
write(HERE/'unit_v2.json',dict(passed=True,kernels=rows,ownership=ownership,rejections=rejected,wide_close_pair_improved=True,
    sources_unmodified=True,targets_read=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,wide=wide,ownership=ownership),flush=True)

import importlib.util
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,PARENT,write,pin,seal_check

seal_check()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    with flops.BudgetContext(flop_budget=2**41,quiet=True):spec.loader.exec_module(m)
    return m
parent=load(PARENT,'selector_original');candidate=load(HERE/'candidate.py','selector_canonical')
rows=[]
for dtype in [np.float32,np.float64]:
    for n in [0,7,32,33,1024]:
        for role in ['empty','ties','ramp']:
            score=(np.full(n,-1.) if role=='empty' else np.resize(np.array([.5,.01,.01,0.,-.1]),n)
                   if role=='ties' else np.linspace(-.5,.5,n)).astype(dtype)
            with flops.BudgetContext(flop_budget=2**41,quiet=True):x=fnp.asarray(score)
            before=np.asarray(x).copy();answers=[];bills=[]
            for mod in [parent,candidate,candidate]:
                with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:answer=mod.select(x,.01)
                answers.append(tuple(np.asarray(a).copy() for a in answer));bills.append(ctx.summary_dict()['flops_used'])
            old,new,again=answers
            assert before.tobytes()==np.asarray(x).tobytes()
            assert old[1].tobytes()==new[1].tobytes()==again[1].tobytes()
            assert np.array_equal(np.sort(old[0]),new[0]) and new[0].tobytes()==again[0].tobytes()
            assert len(new[0])==min(n,((np.count_nonzero(score>=.01)+31)//32)*32)
            assert bills[1]==bills[2] and bills[1]>=bills[0]
            rows.append(dict(dtype=str(np.dtype(dtype)),width=n,fixture=role,retained=len(new[0]),
                parent_flops=bills[0],canonical_flops=bills[1],mask_exact=True,ascending=True,repeat_exact=True))
rng=np.random.default_rng(2026092068)
n,k,rank=9,5,4
perm=rng.permutation(n);inverse=np.argsort(perm)
w=rng.normal(size=(n,k));a=rng.normal(size=(n,n));c=a@a.T
left=w.T@c@w;wp=w[perm];right=wp.T@c[np.ix_(perm,perm)]@wp
matrix=float(np.linalg.norm(left-right)/np.linalg.norm(left))
base=rng.normal(size=(n,rank));shift=rng.normal(size=(n,rank))*.1
factors=np.concatenate([base+shift,base-shift],axis=1)
weights=np.r_[rng.normal(size=rank),np.zeros(rank)];weights[rank:]=-weights[:rank]
order=rng.permutation(rank);columns=np.r_[order,order+rank]
tensor=np.einsum('ir,jr,kr,r->ijk',factors,factors,factors,weights,optimize=True)
other=np.einsum('ir,jr,kr,r->ijk',factors[:,columns],factors[:,columns],factors[:,columns],weights[columns],optimize=True)
cubic=float(np.linalg.norm(tensor-other)/np.linalg.norm(tensor))
y=rng.normal(size=(n,rank));q,_=np.linalg.qr(y,mode='reduced');qp,_=np.linalg.qr(y[perm],mode='reduced');qp=qp[inverse]
projection=float(np.linalg.norm(q@q.T-qp@qp.T)/np.linalg.norm(q@q.T))
assert max(matrix,cubic,projection)<=2e-12
write(HERE/'unit.json',dict(passed=True,selector_fixtures=rows,matrix_congruence_relative=matrix,
    signed_paired_tensor_relative=cubic,qr_range_projector_relative=projection,
    full_estimator_bitwise_equivalence_claimed=False,targets_accessed=False,source=pin(__file__)))
print(dict(passed=True,fixtures=len(rows),matrix=matrix,cubic=cubic,projection=projection),flush=True)

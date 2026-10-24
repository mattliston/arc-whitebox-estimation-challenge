import ast,resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import *
seal_check()
tree=ast.parse((HERE/'fragment.py').read_text())
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_nested_move','_nested_layout','_shared_old384_read'}]
def finite(x,name):
    if str(x.dtype)!='float32' or not bool(fnp.all(fnp.isfinite(x))):raise RuntimeError(name)
    return x
env=dict(fnp=fnp,_shared_old384_mm=lambda a,b:a@b,_shared_old384_finite=finite,_mm=lambda a,b:a@b,_NESTED_RANK=256)
exec(compile(ast.Module(body=selected,type_ignores=[]),'nested_unit_functions','exec'),env)
def error(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1.0))
rng=np.random.default_rng(735256);records=[]
def moves(rank,zero=False):
    r=8;old_basis=old_factors=old_core=None;old_coeff=np.zeros(0,dtype=np.float32)
    checks=[];saved=[]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        for step in range(3):
            J=(rng.normal(size=(r,6))*.2).astype(np.float32)
            lam=np.array([.5,-.5,1.,-1.,.7,-.7],dtype=np.float32)
            if zero:lam*=0
            if step and old_basis is not None:
                # A first-tier overlap is generally not orthogonal.
                rotation=(np.eye(r)+rng.normal(size=(r,r))*.025).astype(np.float32)
                old_basis=fnp.asarray(rotation)@old_basis
            J0=J.astype(np.float64)
            if old_basis is None:
                full=J0;G=(J0*lam.astype(np.float64)**2)@J0.T
            else:
                U=np.asarray(old_basis,dtype=np.float64);F=np.asarray(old_factors,dtype=np.float64)
                full=np.concatenate([U@F,J0],axis=1)
                G=(J0*lam.astype(np.float64)**2)@J0.T+U@np.asarray(old_core,dtype=np.float64)@U.T
            Q,F,S,removed,restored=env['_nested_move'](fnp.asarray(J),fnp.asarray(lam),old_basis,old_factors,old_core,rank)
            q,f,s=[np.asarray(v,dtype=np.float64) for v in (Q,F,S)]
            coeff=np.concatenate([old_coeff,lam]);expected=q@q.T@full
            errors=dict(gram=error(removed,G),factors=error(q@f,expected),core=error(s,q.T@G@q),
                restored=error(restored,q@q.T@G@q@q.T),orthogonality=error(q.T@q,np.eye(rank)))
            # The removed/replaced component yields exactly the projected
            # weighted factor Gram in real arithmetic, without a PSD clamp.
            middle=rng.normal(size=(r,4))*.1
            complete=G+middle@middle.T
            repaired=complete-np.asarray(removed)+np.asarray(restored)
            errors['repair']=error(repaired,(expected*coeff.astype(np.float64)**2)@expected.T+middle@middle.T)
            if rank==r:errors['full_rank_identity']=error(q@f,full)
            assert max(errors.values())<=3e-5,errors
            checks.append(dict(step=step,rank=rank,zero_coefficients=zero,errors=errors))
            saved += [np.asarray(x).copy() for x in (Q,F,S,removed,restored)]
            old_basis,old_factors,old_core,old_coeff=Q,F,S,coeff
        bill=ctx.summary_dict()['flops_used']
    return checks,saved,bill
for rank,zero in [(4,False),(8,False),(4,True)]:
    state=rng.bit_generator.state
    a,x,bill=moves(rank,zero);rng.bit_generator.state=state
    b,y,bill2=moves(rank,zero)
    assert bill==bill2 and all(u.tobytes()==v.tobytes() for u,v in zip(x,y))
    records.extend(a)
# Execute the actual production-size layout and read helper; its input row
# dimension is small, allowing independent explicit third tensors.
for cancelling in [False,True]:
    r,t,n,b,c,young=384,256,10,6,8,4
    U=(rng.normal(size=(n,r))*.05).astype(np.float32)
    Q=(rng.normal(size=(r,t))*.05).astype(np.float32)
    F2=(rng.normal(size=(t,b))*.05).astype(np.float32)
    F1=(rng.normal(size=(r,c))*.05).astype(np.float32)
    Y=(rng.normal(size=(n,young))*.05).astype(np.float32)
    coeff=np.resize(np.array([1,-1],dtype=np.float32),b+c+young)
    if cancelling:
        F2[:,1::2]=F2[:,::2];F1[:,1::2]=F1[:,::2];Y[:,1::2]=Y[:,::2]
    full=np.concatenate([U.astype(float)@Q.astype(float)@F2.astype(float),U.astype(float)@F1.astype(float),Y],axis=1)
    pp=full.astype(np.float32);index=np.array([1,3,5,7,9]);response=np.array([0,2,4,6])
    state=dict(columns=b+c,nested_columns=b,lazy=None,F=fnp.asarray(F1),nested_basis=fnp.asarray(Q),
        nested_factors=fnp.asarray(F2),nested_core=fnp.asarray(np.eye(t,dtype=np.float32)),
        pre_basis=fnp.asarray(U),nested_started=True)
    outputs=[];bills=[]
    for _ in range(2):
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            ans=env['_shared_old384_read'](fnp.asarray(pp[response]),fnp.asarray(pp[index]),fnp.asarray(coeff),index,state)
        outputs.append(np.asarray(ans).copy());bills.append(ctx.summary_dict()['flops_used'])
    tensor=np.zeros((n,n,n))
    for j,weight in enumerate(coeff):tensor+=weight*np.einsum('i,j,k->ijk',full[:,j],full[:,j],full[:,j])
    expected=np.array([[tensor[i,i,j] for j in index] for i in response])
    err=error(outputs[0],expected);assert err<=3e-5,err
    assert outputs[0].tobytes()==outputs[1].tobytes() and bills[0]==bills[1]
    records.append(dict(production_read=True,cancelling=cancelling,error=err,flops=bills[0]))
write(HERE/'unit.json',dict(passed=True,records=records,repeat_bytes_and_bills_exact=True,
    error_normalization='Frobenius difference / max(1, reference Frobenius norm)',
    tiny_helpers_use_actual_FNP_classical_matmul=True,production_kernels_tested_by_full_calls=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,cases=len(records)),flush=True)

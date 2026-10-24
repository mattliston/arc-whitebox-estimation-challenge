import math
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,write,seal_check
from observer_math import raw_design,normalize_design,direct,local_design,solve_gram,fnp_step

seal_check('fit_sources_sealed.json');rng=np.random.Generator(np.random.PCG64(2026092066))
raw=rng.normal(size=(3,5,4));sigma=np.exp(.2*rng.normal(size=(3,5)));gate=rng.uniform(.1,.9,size=(3,5));w=rng.normal(size=(3,5,5))/3
mean=rng.normal(size=4);scale=rng.uniform(.5,2,size=4);beta=rng.normal(size=7)
b=raw_design(raw,sigma,gate,w);z=normalize_design(b,mean,scale);y=direct(raw,sigma,gate,w,mean,scale,beta)
explicit=[];previous=[0.]*5
for l in range(3):
    out=[]
    for j in range(5):
        s=sigma[l,j]*(math.fsum(((raw[l,j,k]-mean[k])/scale[k])*beta[k] for k in range(4))+beta[4+l])
        t=gate[l,j]*math.fsum(w[l,k,j]*previous[k] for k in range(5)) if l else 0.
        out.append(t+s)
    explicit.append(out);previous=out
explicit=np.asarray(explicit);assert np.allclose(y,explicit,rtol=2e-13,atol=2e-13)
assert np.allclose(z@beta,y,rtol=2e-13,atol=2e-13)
assert np.allclose(normalize_design(local_design(raw,sigma),mean,scale)@beta,direct(raw,sigma,gate,w,mean,scale,beta,False),rtol=2e-13,atol=2e-13)
fnp_rows=[];bills=[]
for coefficients in [beta,np.zeros_like(beta),beta]:
    with flops.BudgetContext(flop_budget=1e8,quiet=True) as ctx:
        fm=fnp.asarray(mean);fs=fnp.asarray(scale);fb=fnp.asarray(coefficients);previous=None;rows=[]
        for l in range(3):
            previous=fnp_step(fnp,fnp.asarray(raw[l]),fnp.asarray(sigma[l]),fnp.asarray(gate[l]),fnp.asarray(w[l]),previous,fm,fs,fb,l);rows.append(previous)
        answer=fnp.stack(rows)
    fnp_rows.append(np.asarray(answer).copy());bills.append(ctx.summary_dict()['flops_used'])
assert np.allclose(fnp_rows[0],explicit,rtol=2e-13,atol=2e-13)
assert np.count_nonzero(fnp_rows[1])==0 and fnp_rows[0].tobytes()==fnp_rows[2].tobytes() and bills[0]==bills[2]
A=rng.normal(size=(79,9));A[:,-1]=0;target=rng.normal(size=79);weight=rng.uniform(.1,1,size=79);weight/=weight.sum()
gram=A.T@(weight[:,None]*A);rhs=A.T@(weight*target);coef,cert=solve_gram(gram,rhs)
s=np.asarray(cert['design_rms']);independent=np.linalg.lstsq(np.vstack([np.sqrt(weight)[:,None]*A/s,np.sqrt(1e-3)*np.eye(9)]),np.r_[np.sqrt(weight)*target,np.zeros(9)],rcond=None)[0]/s
assert np.allclose(coef,independent,rtol=1e-10,atol=1e-12)
write(HERE/'unit.json',dict(passed=True,independent_scalar_recurrence=True,normalized_design_equivalence=True,
    fnp_reference_max_absolute=float(np.max(np.abs(fnp_rows[0]-explicit))),fnp_repeat_exact=True,zero_beta_exact=True,
    fnp_bills=bills,ridge_augmented_least_squares_max_absolute=float(np.max(np.abs(coef-independent))),ridge_certificate=cert,
    targets_accessed=False))
print(dict(passed=True,fnp_bills=bills),flush=True)

import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import *
from solver import fit,serve
seal_check()
for repair_pin in read(HERE/'repair_sources_sealed.json')['pins']:check(repair_pin)
rng=np.random.default_rng(2026092179);rows=[]
for case in range(4):
    n=31+case;d=7;X=rng.normal(size=(n,d));y=rng.normal(size=n);w=np.exp(rng.normal(size=n))
    X[:,0]=1;X[:,1]=0;X[:,2]=X[:,3]
    if case==3:y[:]=0
    beta,cert=fit(np,X,y,w);wn=w/w.sum();xn=X/cert['scale']
    aug=np.vstack([np.sqrt(wn)[:,None]*xn,np.sqrt(1e-3)*np.eye(d)])
    yy=np.concatenate([np.sqrt(wn)*y,np.zeros(d)])
    scaled_alt=np.linalg.lstsq(aug,yy,rcond=None)[0];alt=scaled_alt/cert['scale']
    error=float(np.linalg.norm(cert['scaled_beta']-scaled_alt)/max(np.linalg.norm(cert['scaled_beta']),1e-30));assert error<=2e-10
    prediction_error=float(np.linalg.norm(X@beta-X@alt)/max(np.linalg.norm(X@beta),1e-30));assert prediction_error<=2e-10
    rows.append(dict(case=case,augmented_scaled_lstsq_relative=error,prediction_relative=prediction_error,normal_residual=cert['residual']))
n=1024;H=rng.normal(size=(n,64)).astype(np.float32);Z=rng.normal(size=(n,65))*1e-5
sigma=np.exp(rng.normal(size=n)*.1);parent=np.abs(rng.normal(size=n)).astype(np.float32)
bill=[]
for d in [65,130]:
    coef=rng.normal(size=d)*.01;reference=np.maximum(parent.astype(float)+np.column_stack([sigma[:,None]*np.column_stack([np.ones(n),H]),Z])[:,:d]@coef,0).astype(np.float32)
    outputs=[];costs=[]
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
            out=serve(fnp,fnp.asarray(parent),fnp.asarray(sigma),fnp.asarray(H),fnp.asarray(Z),fnp.asarray(coef))
        outputs.append(np.asarray(out).copy());costs.append(ctx.summary_dict()['flops_used'])
    assert outputs[0].tobytes()==outputs[1].tobytes()==reference.tobytes() and costs[0]==costs[1]
    assert costs[0]<=2_000_000,costs
    with flops.BudgetContext(flop_budget=2**30,quiet=True):
        out=serve(fnp,fnp.asarray(parent),fnp.asarray(sigma),fnp.asarray(H),fnp.asarray(Z),fnp.zeros(d,dtype=fnp.float64))
    assert np.asarray(out).tobytes()==parent.tobytes()
    bill.append(dict(columns=d,flops=costs[0]))
write(HERE/'solver_unit_v2.json',dict(passed=True,rows=rows,wide_readout_bills=bill,
    repeated_bytes_and_bill=True,zero_coefficients_exact=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,wide_readout_bills=bill),flush=True)

"""Full-span quadratic-form and input-permutation certificate at width8."""
import hashlib
import json
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from feature import correlated_energy

HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(19092026)
n=8
a=rng.standard_normal((n,n)).astype(np.float32)
c=a@a.T+np.eye(n,dtype=np.float32)
w=rng.standard_normal((n,n)).astype(np.float32)
state=rng.standard_normal((n,64)).astype(np.float32)
parameters=dict(msg_w=rng.standard_normal((48,64)).astype(np.float32)*.1,
                msg_b=rng.standard_normal(48).astype(np.float32)*.1)
pdf=rng.uniform(.01,.4,n).astype(np.float32)
scale=np.sqrt(c.diagonal().astype(np.float64))
corr=c.astype(np.float64)/scale[:,None]/scale[None,:]
source=state.astype(np.float64)@parameters['msg_w'][:16].astype(np.float64).T+parameters['msg_b'][:16]
target=np.zeros((n,16),dtype=np.float64)
for j in range(n):
    for k in range(16):
        v=w[:,j].astype(np.float64)*source[:,k]
        target[j,k]=.5*pdf[j]*np.clip((v@(corr-np.eye(n))@v)/max(v@v,1e-12),-8,8)
outputs=[]
bills=[]
for permutation in (np.arange(n),np.array([6,2,0,7,1,5,3,4])):
    with flops.BudgetContext(flop_budget=1e8,quiet=True) as ctx:
        cc=fnp.asarray(c[np.ix_(permutation,permutation)])
        ww=fnp.asarray(w[permutation])
        hh=fnp.asarray(state[permutation])
        pp={k:fnp.asarray(v) for k,v in parameters.items()}
        y=correlated_energy(fnp,cc,hh,ww,pp,fnp.asarray(pdf))
    outputs.append(np.asarray(y).copy())
    bills.append(ctx.summary_dict()['flops_used'])
errors=[float(np.max(np.abs(y-target))) for y in outputs]
permutation_error=float(np.max(np.abs(outputs[0]-outputs[1])))
assert max(errors)<2e-5 and permutation_error<2e-5
rec=dict(passed=True,full_span_max_absolute_errors=errors,permutation_max_absolute_error=permutation_error,
    all_inclusive_test_flops=bills,feature_sha256=hashlib.sha256((HERE/'feature.py').read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),truth_accessed=False,
    scope='Exact full-span quadratic form at width8; does not certify rank8 approximation accuracy at width1024.')
with (HERE/'certificate.json').open('x') as f:json.dump(rec,f,indent=2)
print(json.dumps(rec),flush=True)

"""Small native arithmetic and CP-gauge qualification, no network arrays."""
import json
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
HERE=Path(__file__).resolve().parent
out=HERE/'unit.json';assert not out.exists()
rng=np.random.default_rng(706196)
P=rng.normal(size=(12,18)).astype(np.float32)
lam=rng.normal(size=18).astype(np.float32)
scale=np.ldexp(np.ones(18,np.float32),np.arange(18)%5-2)
scale[::3]*=-1
records=[];ys=[]
for tag,p,l in [('original',P,lam),('equivalent_scale',P*scale,lam/(scale**3))]:
 sq=p*p;norm=np.sum(sq,axis=0);weight=(l*l)*(norm*norm)
 expected_diag=sq@weight
 rows=np.sort(np.argsort(-expected_diag.astype(np.float64),kind='stable')[:6])
 expected_Y=p@(weight[:,None]*p[rows,:].T)
 with flops.BudgetContext(flop_budget=10**8,quiet=True) as ctx:
  pf=fnp.array(p);lf=fnp.array(l)
  sf=pf*pf;nf=fnp.sum(sf,axis=0);wf=(lf*lf)*(nf*nf)
  df=fnp.matmul(sf,wf)
  rf=fnp.sort(fnp.argsort(-df.astype(fnp.float64),kind='stable')[:6])
  yf=fnp.matmul(pf,wf[:,None]*pf[rf,:].T)
  actual=np.asarray(yf).copy();actual_d=np.asarray(df).copy();actual_r=np.asarray(rf).copy()
 error=float(np.max(np.abs(actual-expected_Y))/max(1.,float(np.max(np.abs(expected_Y)))))
 diag_error=float(np.max(np.abs(actual_d-expected_diag))/max(1.,float(np.max(np.abs(expected_diag)))))
 assert error<2e-6 and diag_error<2e-6 and np.array_equal(actual_r,rows)
 ys.append(actual)
 records.append(dict(case=tag,metric_relative_max=error,diagonal_relative_max=diag_error,rows_exact=True,flops=int(ctx.summary_dict()['flops_used'])))
gauge=float(np.max(np.abs(ys[0]-ys[1]))/max(1.,float(np.max(np.abs(ys[0])))))
assert gauge<2e-6
record=dict(passed=True,records=records,power_two_cp_gauge_relative=gauge,network_arrays_accessed=False)
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record,indent=2),flush=True)

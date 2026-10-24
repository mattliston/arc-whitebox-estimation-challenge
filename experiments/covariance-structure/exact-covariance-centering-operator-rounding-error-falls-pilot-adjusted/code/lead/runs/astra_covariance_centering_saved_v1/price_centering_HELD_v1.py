from pathlib import Path
import json,hashlib,time,resource,os
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
def center(C):
 r=fnp.mean(C,axis=1);c=fnp.mean(r)
 C0=((C-r[:,None])-r[None,:])+c
 return C0,r,c
def restore(Cov,W,r,c):
 u=fnp.sum(W,axis=0);v=r@W;t=v-(np.float32(.5)*c)*u
 R=t[:,None]*u[None,:]
 return (Cov+R)+R.T
rng=np.random.default_rng(20261015)
with fs.BudgetContext(flop_budget=2**50,quiet=True):
 C=fnp.asarray(rng.normal(size=(1024,1024)).astype('f'))
 W=fnp.asarray(rng.normal(size=(1024,1024)).astype('f'))
 Cov=fnp.zeros((1024,1024),dtype=fnp.float32)
with fs.BudgetContext(flop_budget=2**50,quiet=True) as budget:
 C0,r,c=center(C);out=restore(Cov,W,r,c)
extra=int(budget.flops_used)
result=dict(status='BELOW_POINT2_PERCENT_GATE',shape=[1024,1024],extra_flops_per_full_hidden=extra,extra_ops_per_full_hidden=len(budget._op_log),fifteen_full_layers_upper_flops=15*extra,fifteen_full_layers_upper_percent=1500*extra/242136236203,whole_bill_reference=242136236203,formula='r=mean(C,1); c=mean(r); C0=C-r[:,None]-r[None,:]+c; u=sum(W,0); v=r@W; t=v-.5*c*u; result=Cov+t[:,None]*u[None,:]+u[:,None]*t[None,:]',dtype='float32',no_dense_product_in_tariff=True,new_einsum_signatures=0,notes=['Current kernels unchanged; price only incremental centering and rank2 restore.','Fifteen full1024 sandwiches is a conservative shape upper bound, not measured whole-model bill.','Basis/order differ in floating point; native prediction equivalence is not claimed.'])
assert result['fifteen_full_layers_upper_percent']<.2
p=D/'CENTERING_TARIFF_v1.json'
with p.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result));print('sha256',hashlib.sha256(p.read_bytes()).hexdigest())


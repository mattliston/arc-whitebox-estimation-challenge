"""Prepared follow-up: compare ordinary batched leaf-product implementations.

Run in the OpenBLAS runtime through guard.py, after the current qualification.
All three methods perform the same scalar matrix products. Do not deploy an
einsum path with suspicious undercounting: require its bill to match arithmetic.
No source candidate is modified or imported by this microbenchmark.
"""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE=Path(__file__).resolve().parent
output=HERE/sys.argv[1]
assert not output.exists()
rng=np.random.default_rng(20260926)
results=[]
for depth,dim in [(5,32),(6,16),(7,8)]:
    shape=(7,)*depth+(dim,dim)
    a=rng.standard_normal(shape,dtype=np.float32)
    b=rng.standard_normal(shape,dtype=np.float32)
    ref=a.reshape(-1,dim,dim)[:64].astype(np.float64) @ b.reshape(-1,dim,dim)[:64].astype(np.float64)
    A=fnp.asarray(a);B=fnp.asarray(b)
    C=fnp.empty(shape,dtype=fnp.float32)
    tmp=fnp.empty(shape,dtype=fnp.float32)
    del a,b
    for rep in range(3):
        methods=('blas','einsum','outer') if rep%2==0 else ('outer','einsum','blas')
        for method in methods:
            t0=time.process_time()
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                if method=='blas':
                    fnp.matmul(A,B,out=C)
                elif method=='einsum':
                    fnp.einsum('...ik,...kj->...ij',A,B,out=C,optimize=False)
                else:
                    fnp.multiply(A[..., :, 0:1],B[..., 0:1, :],out=C)
                    for k in range(1,dim):
                        fnp.multiply(A[..., :, k:k+1],B[..., k:k+1, :],out=tmp)
                        fnp.add(C,tmp,out=C)
            cpu=time.process_time()-t0
            sample=np.asarray(C.reshape((-1,dim,dim))[:64],dtype=np.float64)
            record=dict(depth=depth,dim=dim,method=method,rep=rep,cpu_s=cpu,
                flops=int(ctx.summary_dict()['flops_used']),scalar_multiply_add_count=(7**depth)*dim*dim*(2*dim-1),
                relative_error=float(np.linalg.norm(sample-ref)/np.linalg.norm(ref)),max_abs_error=float(np.abs(sample-ref).max()))
            results.append(record);print(json.dumps(record),flush=True)
    del A,B,C,tmp,ref,sample
    gc.collect()
with output.open('x') as f:json.dump(results,f,indent=2)

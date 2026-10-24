"""Explicit tensor certificate for the new star residual trace read."""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(20260927)
n,k,m=7,3,5
A=rng.normal(size=(n,n))
u,s,vt=np.linalg.svd(A,full_matrices=False)
Q,B=u[:,:k],s[:k,None]*vt[:k]
D=A-Q@B
W=rng.normal(size=(n,m))
T=np.zeros((n,n,n))
for a in range(n):
    for b in range(n):
        T[a,a,b]+=D[b,a]/3
        T[a,b,a]+=D[b,a]/3
        T[b,a,a]+=D[b,a]/3
v=np.einsum('aab->b',T)
cheap=D.sum(axis=1)+2*np.diag(D)
no_D=A.sum(axis=1)-Q@B.sum(axis=1)+2*(np.diag(A)-np.sum(Q*B.T,axis=1))
I=np.eye(n)
trace_T=(np.einsum('a,bc->abc',v,I)+np.einsum('b,ac->abc',v,I)+np.einsum('c,ab->abc',v,I))/(n+2)
reference=np.einsum('abc,ai,bi,ci->i',trace_T,W,W,W)
formula=np.sum(W*W,axis=0)/(n+2)*(W.T@no_D)
exact_star=np.sum((W.T@D)*(W.T*W.T),axis=1)
direct_star=np.einsum('abc,ai,bi,ci->i',T,W,W,W)
errors=dict(star_tensor_read=float(np.max(np.abs(exact_star-direct_star))),
            tensor_trace=float(np.max(np.abs(3*v-cheap))),
            no_residual_matrix=float(np.max(np.abs(cheap-no_D))),
            trace_preservation=float(np.max(np.abs(np.einsum('aab->b',trace_T)-v))),
            orthogonality=float(abs(np.sum((T-trace_T)*trace_T))),
            scalar_read=float(np.max(np.abs(reference-formula))))
assert max(errors.values())<1e-10
record=dict(errors=errors,passed=True,endpoint_truth_accessed=False,
            scope='Small algebra identity, not a Phase1 network or an endpoint/cost result.',
            protocol_sha256=hashlib.sha256((HERE/'star_residual_trace_protocol_v1.json').read_bytes()).hexdigest())
with (HERE/'star_residual_trace_certificate_v1.json').open('x') as fp:
    json.dump(record,fp,indent=2)
print(json.dumps(record))

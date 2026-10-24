"""Check the stated trace projection identity, not approximation accuracy."""
import json
from pathlib import Path
import numpy as np
from radial_cold_cp_trace_v1 import trace_vector,trace_fit

out=Path(__file__).resolve().parent/'cold_cp_trace_certificate_v1.json'
assert not out.exists()
rng=np.random.default_rng(8748)
p=rng.normal(size=(8,13));lam=rng.normal(size=13)
A=rng.normal(size=(8,8));C=A@A.T
T=np.einsum('ir,jr,kr,r->ijk',p,p,p,lam)
t=trace_vector(p,lam)
exact=np.linalg.solve(np.trace(C)*np.eye(8)+2*C,t)
b=trace_fit(t,C)
def sym(a,B):
    return np.einsum('i,jk->ijk',a,B)+np.einsum('j,ik->ijk',a,B)+np.einsum('k,ij->ijk',a,B)
S=sym(b,C)
w=rng.normal(size=(8,8))
P=np.einsum('ijk,ia,jb,kc->abc',S,w,w,w)
ids=np.arange(8)
read=3*(w.T@b)*np.diag(w.T@C@w)
result=dict(trace_max_abs=float(np.max(np.abs(np.einsum('ijj->i',T)-t))),
    fixed_solve_relative=float(np.linalg.norm(b-exact)/np.linalg.norm(exact)),
    push_diagonal_max_abs=float(np.max(np.abs(P[ids,ids,ids]-read))))
assert result['trace_max_abs']<1e-10
assert result['fixed_solve_relative']<1e-6
assert result['push_diagonal_max_abs']<1e-10
out.write_text(json.dumps(dict(pass_all=True,**result),indent=2)+'\n')
print(json.dumps(result),flush=True)

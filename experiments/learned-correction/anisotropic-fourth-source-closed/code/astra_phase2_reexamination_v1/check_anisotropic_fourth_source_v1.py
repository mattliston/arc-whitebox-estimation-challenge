"""Certify represented planes, trace fitting, and first-order birth formula."""
import json
from pathlib import Path
import numpy as np
from radial_anisotropic_fourth_source_v1 import fit_diagonal

out=Path(__file__).resolve().parent/'anisotropic_fourth_source_certificate_v1.json'
assert not out.exists()
rng=np.random.default_rng(91839);n=6
A=rng.normal(size=(n,n));C=A@A.T
E=rng.normal(size=(n,n));E=(E+E.T)*.0005
alpha=.03;D=alpha*C+E
K=(np.einsum('ij,kl->ijkl',C,D)+np.einsum('ik,jl->ijkl',C,D)+np.einsum('il,jk->ijkl',C,D)
   +np.einsum('jk,il->ijkl',C,D)+np.einsum('jl,ik->ijkl',C,D)+np.einsum('kl,ij->ijkl',C,D))/6
ids=np.arange(n);v=np.diag(C);dv=np.diag(D)
k22=(np.outer(v,dv)+np.outer(dv,v)+4*C*D)/6
k31=(v[:,None]*D+dv[:,None]*C)/2
g=rng.uniform(.1,.9,size=n);d=rng.uniform(.1,.5,size=n)
a=1+alpha/3
base=g[:,None]*C;delta=g[:,None]*E/(6*a)
exact_first=a*base[:,:,None]*base[:,None,:]
exact_first+=a*(base[:,:,None]*delta[:,None,:]+delta[:,:,None]*base[:,None,:])
shifted=a*(base+delta)[:,:,None]*(base+delta)[:,None,:]
quadratic=a*delta[:,:,None]*delta[:,None,:]
target=rng.normal(size=n)
Dfit=fit_diagonal(C,D,target)
fit22=(np.outer(v,np.diag(Dfit))+np.outer(np.diag(Dfit),v)+4*C*Dfit)/6
result=dict(k22_max_abs=float(np.max(np.abs(K[ids[:,None],ids[:,None],ids[None,:],ids[None,:]]-k22))),
    k31_max_abs=float(np.max(np.abs(K[ids[:,None],ids[:,None],ids[:,None],ids[None,:]]-k31))),
    fitted_trace_max_abs=float(np.max(np.abs(fit22.sum(axis=1)-target))),
    birth_remainder_identity_max_abs=float(np.max(np.abs(shifted-exact_first-quadratic))))
assert max(result.values())<1e-10,result
out.write_text(json.dumps(dict(pass_all=True,**result),indent=2)+'\n')
print(json.dumps(result),flush=True)

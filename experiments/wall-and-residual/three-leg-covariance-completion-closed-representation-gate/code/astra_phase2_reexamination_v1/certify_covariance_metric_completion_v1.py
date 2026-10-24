"""Independent dense tensor certificates, no score labels or wide nets."""
from pathlib import Path
import hashlib
import json
import numpy as np
from covariance_metric_completion_v1 import (
    operator,factors,plane,push_plane,star_from_plane,star_query,complete)

HERE=Path(__file__).resolve().parent
OUT=HERE/'covariance_metric_completion_certificate_v1.json'
assert not OUT.exists()
rng=np.random.default_rng(953265);records=[]
for n in (6,9):
    B=rng.normal(size=(n,n));C=B@B.T/n+np.eye(n)
    sigma=np.sqrt(np.diag(C));R=C/np.outer(sigma,sigma)
    H=rng.normal(size=(n,n));J=rng.normal(size=(n,n))
    U,V=factors(R,H,sigma)
    T=(np.einsum('ia,ja,ka->ijk',U,U,V)+
       np.einsum('ia,ja,ka->ijk',U,V,U)+
       np.einsum('ia,ja,ka->ijk',V,U,U))
    ix=np.arange(n);K=T[ix,ix,:]
    expected=operator(R,H)*(sigma[:,None]**2*sigma[None,:])
    error=float(np.linalg.norm(expected-K)/np.linalg.norm(K))
    adjoint=abs(float(np.sum(J*operator(R,H))-np.sum(H*operator(R,J))))
    adjoint/=max(np.linalg.norm(J)*np.linalg.norm(operator(R,H)),1e-30)
    quadratic=float(np.sum(H*operator(R,H)))
    W=rng.normal(size=(n,n))/np.sqrt(n)
    pushed=np.einsum('ia,jb,kc,ijk->abc',W,W,W,T,optimize=True)
    queried=push_plane(U,V,W,ix)
    push_error=float(np.linalg.norm(queried-pushed[ix,ix,:])/np.linalg.norm(pushed[ix,ix,:]))
    target=rng.normal(size=(n,n))
    S=star_from_plane(target-K)
    addition=np.zeros_like(T)
    for a in range(n):
        for r in range(n):
            addition[a,r,r]+=S[a,r]/3
            addition[r,a,r]+=S[a,r]/3
            addition[r,r,a]+=S[a,r]/3
    restoration=float(np.linalg.norm((T+addition)[ix,ix,:]-target)/np.linalg.norm(target))
    addpush=np.einsum('ia,jb,kc,ijk->abc',W,W,W,addition,optimize=True)[ix,ix,:]
    star_error=float(np.linalg.norm(star_query(S,W,ix)-addpush)/np.linalg.norm(addpush))
    recovered,_,convergence=complete(R,operator(R,H),steps=100)
    inverse_error=float(np.linalg.norm(operator(R,recovered)-operator(R,H))/np.linalg.norm(operator(R,H)))
    row=dict(n=n,plane_relative=error,adjoint_relative=adjoint,positive_quadratic=quadratic,
             push_relative=push_error,restoration_relative=restoration,star_query_relative=star_error,
             inverse_relative=inverse_error,iterations=len(convergence))
    row['passed']=max(error,adjoint,push_error,restoration,star_error,inverse_error)<1e-10 and quadratic>0
    records.append(row)
result=dict(records=records,passed=all(r['passed'] for r in records),
            source_sha256=hashlib.sha256((HERE/'covariance_metric_completion_v1.py').read_bytes()).hexdigest(),
            qualification='Synthetic dense tensor identities only; no wide representation or accuracy claim.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)

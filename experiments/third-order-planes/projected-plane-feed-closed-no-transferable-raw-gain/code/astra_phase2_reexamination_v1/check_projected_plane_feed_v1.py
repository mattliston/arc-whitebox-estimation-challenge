"""Small dense identity and Frobenius projection checks, no empirical labels."""
import json
from pathlib import Path
import numpy as np
from projected_plane_feed_v1 import projected_birth

HERE=Path(__file__).resolve().parent
target=HERE/"projected_plane_feed_certificate_v1.json"
assert not target.exists()
rng=np.random.default_rng(399851);n=12
A=rng.normal(size=(n,n));C=A@A.T/n+np.eye(n)
g=rng.uniform(.1,.9,n);alpha=rng.normal(size=n);pdf=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
c=rng.normal(scale=.1,size=n);K=c[:,None]*C
aux=dict(gate=g,alpha=alpha,pdf=pdf,pre_v=np.diag(C))
extra,profile=projected_birth(C,K,aux,np.ones(n,dtype=bool))
e=-alpha*pdf/np.diag(C)
expected=e*c
assert np.max(np.abs(extra-expected))<1e-14
dense=np.zeros((n,n,n));projected=np.zeros_like(dense)
for i in range(n):
    for j in range(n):
        for k in range(n):
            if len({i,j,k})<3:continue
            for a,b,d in ((i,j,k),(j,i,k),(k,i,j)):
                dense[i,j,k]+=.5*e[a]*g[b]*g[d]*(K[a,b]*C[a,d]+K[a,d]*C[a,b])
                projected[i,j,k]+=extra[a]*g[b]*g[d]*C[a,b]*C[a,d]
error=float(np.max(np.abs(dense-projected)))
assert error<1e-14
u=rng.normal(size=17);v=rng.normal(size=17)
coef=float(u@v/(v@v));matrix=(np.outer(u,v)+np.outer(v,u))/2
residual=matrix-coef*np.outer(v,v)
orthogonality=float(np.sum(residual*np.outer(v,v)))
assert abs(orthogonality)<1e-12
record=dict(all_distinct_max_abs=error,birth_max_abs=float(np.max(np.abs(extra-expected))),
    frobenius_normal_equation=orthogonality,proportional_plane_coverage=profile["projected_fraction"])
target.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)

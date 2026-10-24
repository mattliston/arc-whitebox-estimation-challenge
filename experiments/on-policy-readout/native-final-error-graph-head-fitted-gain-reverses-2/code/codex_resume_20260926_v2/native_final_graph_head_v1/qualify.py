from pathlib import Path
import hashlib
import json
import numpy as np
from graph import graph_fields, transport

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rng=np.random.default_rng(2026092705)
records=[]
for n in (17,31,49):
    L=rng.normal(size=(n,n));C=L@L.T/n+np.eye(n)*.3
    C[0,1]+=1e-7  # Preserve a directed, asymmetric fixture.
    a=rng.normal(size=n)
    aux=dict(alpha=a,gate=1/(1+np.exp(-a)),pdf=np.exp(-a*a/2)/np.sqrt(2*np.pi))
    m=rng.normal(size=n)
    oldC=C.copy();oldm=m.copy()
    x=graph_fields(m,C,aux)
    assert np.array_equal(C,oldC) and np.array_equal(m,oldm)
    perm=rng.permutation(n)
    xp=graph_fields(m[perm],C[perm][:,perm],{k:v[perm] for k,v in aux.items()})
    discrepancy=float(np.max(np.abs(xp-x[perm])))
    assert discrepancy<1e-9,(n,discrepancy)
    sd=np.sqrt(np.diag(C));R=C/(sd[:,None]*sd[None,:]);np.fill_diagonal(R,0)
    phi=np.column_stack([aux[k] for k in ('alpha','gate','pdf')])
    ref=np.empty((n,8))
    for i in range(n):
        ind=sorted(range(n),key=lambda j:-abs(R[i,j]))[:8]
        for j in range(3):
            ref[i,j]=sum(R[i,k]*phi[k,j] for k in ind)/8
            ref[i,j+3]=sum(abs(R[i,k])*phi[k,j] for k in ind)/8
        ref[i,6]=sum(R[i,k]**2 for k in ind)
        ref[i,7]=max(abs(R[i,k]) for k in ind)
    err=float(np.max(np.abs(ref-x[:,:8])));assert err<1e-12
    H=rng.normal(size=(n,19));W=rng.normal(size=(n,n));g=aux['gate']
    direct=np.stack([sum(W[j,i]*H[j] for j in range(n))*g[i] for i in range(n)])
    e=float(np.max(np.abs(transport(H,g,W)-direct)));assert e<1e-12
    records.append(dict(width=n,permutation_max_abs=discrepancy,neighbor_max_abs=err,transport_max_abs=e))
result=dict(passed=True,records=records,targets_accessed=False,
            graph_sha256=sha(HERE/'graph.py'),protocol_sha256=sha(HERE/'protocol.json'))
with (HERE/'qualification.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)

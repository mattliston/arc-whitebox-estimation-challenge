"""Independent explicit labeled-tree checks; no network or truth arrays."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from tree import q31, diagonal, post_q22, source

HERE=Path(__file__).resolve().parent
assert not (HERE/'units.json').exists()
rng=np.random.default_rng(603271)


def labeled_trees():
    for sequence in itertools.product(range(4), repeat=2):
        degree=[1]*4
        for x in sequence: degree[x]+=1
        remaining=list(sequence); edges=[]
        for x in sequence:
            leaf=next(i for i in range(4) if degree[i]==1)
            edges.append((leaf,x));degree[leaf]-=1;degree[x]-=1
        last=[i for i in range(4) if degree[i]==1]
        edges.append(tuple(last))
        yield edges


trees=list(labeled_trees()); assert len(trees)==16
assert len(set(tuple(sorted(tuple(sorted(e)) for e in t)) for t in trees))==16


def explicit(C,g,d,e):
    n=len(C);out=np.zeros((n,)*4);deriv={1:g,2:d,3:e}
    for indices in itertools.product(range(n),repeat=4):
        value=0.
        for edges in trees:
            degree=[0]*4;product=1.
            for a,b in edges:
                degree[a]+=1;degree[b]+=1;product*=C[indices[a],indices[b]]
            for vertex in range(4):product*=deriv[degree[vertex]][indices[vertex]]
            value+=product
        out[indices]=value
    return out


records=[]
for n,m in ((3,2),(4,5)):
    A=rng.normal(size=(n,n));C=A@A.T/n+np.eye(n)
    g,d,e=(rng.normal(size=n) for _ in range(3));W=rng.normal(size=(n,m))
    T=explicit(C,g,d,e)
    expected31=np.einsum('abcd,ai,bi,ci,dj->ij',T,W,W,W,W,optimize=True)
    expected22=np.array([[T[i,i,j,j] for j in range(n)] for i in range(n)])
    expected4=np.diag(expected31)
    checks={
        'q31':float(np.max(np.abs(q31(C,g,d,e,W)-expected31))),
        'q22':float(np.max(np.abs(post_q22(C,g,d,e)-expected22))),
        'k4':float(np.max(np.abs(diagonal(C,g,d,e,W)-expected4))),
        'scale':float(np.max(np.abs(q31(1.7*C,g,d,e,W)-1.7**3*expected31))),
    }
    scale=max(1.,float(np.max(np.abs(expected31))),float(np.max(np.abs(expected22))))
    assert max(checks.values())<1e-10*scale,checks
    zero=q31(C,g,np.zeros(n),np.zeros(n),W);assert np.array_equal(zero,np.zeros_like(zero))
    h=1e-5;direction=rng.normal(size=W.shape)
    fd=(diagonal(C,g,d,e,W+h*direction)-diagonal(C,g,d,e,W-h*direction))/(2*h)
    analytic=4*np.diag(q31(C,g,d,e,np.column_stack((W,direction)))[:m,m:])
    fd_error=float(np.max(np.abs(fd-analytic))/max(1.,np.max(np.abs(analytic))))
    assert fd_error<1e-8,fd_error
    Cd=np.diag(np.diag(C));v=np.diag(Cd)
    marginal=(12*d*d*g*g+4*e*g**3)*v**3
    independent=(marginal[:,None]*W**3).T@W
    diag_error=float(np.max(np.abs(q31(Cd,g,d,e,W)-independent)))
    assert diag_error<1e-10*max(1.,np.max(np.abs(independent)))
    Cpost=.3*C+np.eye(n); _,_,info,_,_=source(C,g,d,e,W,Cpost)
    radial22=info['beta']*(np.outer(np.diag(Cpost),np.diag(Cpost))+2*Cpost*Cpost)/3
    trace_error=float(abs(np.sum(post_q22(C,g,d,e)-radial22)))
    assert trace_error<1e-10*max(1.,abs(info['trace']))
    records.append(dict(n=n,m=m,absolute_errors=checks,fd_relative_error=fd_error,
                        diagonal_covariance_error=diag_error,trace_error=trace_error))

out=dict(passed=True,records=records,tree_count=len(trees),
         tree_sha256=hashlib.sha256((HERE/'tree.py').read_bytes()).hexdigest(),
         utc_finished=datetime.now(timezone.utc).isoformat(),truth_accessed=False)
with (HERE/'units.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2),flush=True)

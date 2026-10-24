import itertools
import numpy as np
from common import *
from transport import carry,faces,query
verify();rng=np.random.default_rng(2026092741);records=[]
def dense(A,B):
    n=len(A);Q=np.zeros((n,n,n,n))
    for i,j,k,l in itertools.product(range(n),repeat=4):
        Q[i,j,k,l]=(A[i,j]*B[k,l]+A[i,k]*B[j,l]+A[i,l]*B[j,k]+B[i,j]*A[k,l]+B[i,k]*A[j,l]+B[i,l]*A[j,k])/6
    return Q
for n,m in ((3,4),(5,3),(4,4)):
    for kind in ('zero','diagonal','signed'):
        Z=rng.normal(size=(n,n));A=Z@Z.T/n+np.eye(n)
        B=rng.normal(size=(n,n));B=(B+B.T)/2
        if kind=='zero':B*=0
        if kind=='diagonal':B=np.diag(np.diag(B))
        W=rng.normal(size=(n,m))/np.sqrt(n);g=rng.uniform(.05,.95,n);d=rng.normal(size=n)
        Q=dense(A,B);T=np.zeros((n,n,n))
        for i,j,k in itertools.product(range(n),repeat=3):
            T[i,j,k]=.5*(d[i]*g[j]*g[k]*Q[i,i,j,k]+d[j]*g[i]*g[k]*Q[j,j,i,k]+d[k]*g[i]*g[j]*Q[k,k,i,j])
        L=g[:,None]*W
        qt=np.einsum('abcd,ai,bj,ck,dl->ijkl',Q,L,L,L,L,optimize=True)
        tt=np.einsum('abc,ai,bj,ck->ijk',T,W,W,W,optimize=True)
        A1,B1,t21=carry(A,B,g,d,W);q4,q31,q22=faces(A1,B1)
        errs=[]
        for i,j in itertools.product(range(m),repeat=2):
            errs.extend((abs(q4[i]-qt[i,i,i,i]),abs(q31[i,j]-qt[i,i,i,j]),abs(q22[i,j]-qt[i,i,j,j]),abs(t21[i,j]-tt[i,i,j])))
            ref=query(A,B,g,d,W,i,j);errs.extend(abs(x-y) for x,y in zip(ref,(q31[i,j],q22[i,j],t21[i,j])))
        perm=rng.permutation(n);out=rng.permutation(m)
        ap,bp,tp=carry(A[np.ix_(perm,perm)],B[np.ix_(perm,perm)],g[perm],d[perm],W[np.ix_(perm,out)])
        errs.extend((float(np.max(np.abs(ap-A1[np.ix_(out,out)]))),float(np.max(np.abs(bp-B1[np.ix_(out,out)]))),float(np.max(np.abs(tp-t21[np.ix_(out,out)])))))
        oldscale=np.exp(rng.uniform(-1,1,n));newscale=np.exp(rng.uniform(-1,1,m))
        aa,bb,t=carry(A*np.outer(oldscale,oldscale),B*np.outer(oldscale,oldscale),g,d/oldscale,W/oldscale[:,None]*newscale[None,:])
        errs.append(float(np.max(np.abs(t-t21*(newscale**2)[:,None]*newscale[None,:]))))
        qa=faces(aa,bb)
        errs.extend((float(np.max(np.abs(qa[0]-q4*newscale**4))),float(np.max(np.abs(qa[1]-q31*newscale[:,None]**3*newscale[None,:]))),float(np.max(np.abs(qa[2]-q22*np.outer(newscale**2,newscale**2))))))
        error=max(errs);bound=2e-11*(1+float(np.max(np.abs(qt)))+float(np.max(np.abs(tt))))
        assert error<=bound,(n,m,kind,error,bound)
        records.append(dict(n=n,m=m,kind=kind,maximum_error=error,bound=bound))
once(HERE/'unit_result.json',dict(passed=True,records=records,maximum_error=max(r['maximum_error'] for r in records),targets_accessed=False,seal_sha256=sha(HERE/'seal.json')))
print(json.dumps(dict(passed=True,cases=len(records),maximum_error=max(r['maximum_error'] for r in records))),flush=True)

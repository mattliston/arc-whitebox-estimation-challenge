import numpy as np
from common import *
from basis import matrices,actions,NAMES
verify();rng=np.random.default_rng(2026092742);records=[]
for n in (3,7,11):
    z=rng.normal(size=(n,n));C=z@z.T+.1*np.eye(n);K=rng.normal(size=(n,n));s=np.exp(rng.normal(size=n));g=rng.uniform(0,1,n);pdf=rng.uniform(.01,.4,n)
    R,S,products=matrices(C,K,s);refs=[]
    for A,B in ((R,R),(R,S),(R,S.T),(S,S.T)):
        X=np.zeros((n,n))
        for i in range(n):
            for j in range(n):X[i,j]=sum(A[i,k]*B[k,j] for k in range(n))/n
        refs.append(X)
    errors=[float(np.max(np.abs(x-y))) for x,y in zip(refs,products)]
    pairs=[(R,False),(S,False),(S.T,False),(R,True),(S,True)]
    pairs += [(M,fourth) for M in refs for fourth in (False,True)]
    pairs += [(refs[1].T,False),(refs[2].T,False)]
    built=list(actions(C,K,s,g,pdf));assert [name for name,D in built]==list(NAMES)
    perm=rng.permutation(n);permuted=list(actions(C[np.ix_(perm,perm)],K[np.ix_(perm,perm)],s[perm],g[perm],pdf[perm]))
    W=rng.normal(size=(n,n+2))
    for j,((name,D),(M,fourth)) in enumerate(zip(built,pairs)):
        ref=np.zeros_like(D)
        for i in range(n):
            for k in range(n):
                if i!=k:ref[i,k]=.5*s[i]*s[k]*(M[i,k]*pdf[i]*(pdf[k] if fourth else g[k])+M[k,i]*pdf[k]*(pdf[i] if fourth else g[i]))
        errors.append(float(np.max(np.abs(ref-D))))
        errors.append(float(np.max(np.abs(D[np.ix_(perm,perm)]-permuted[j][1]))))
        push=np.sum(W*(D@W),axis=0)
        exact=np.array([sum(W[a,i]*ref[a,b]*W[b,i] for a in range(n) for b in range(n)) for i in range(n+2)])
        errors.append(float(np.max(np.abs(push-exact))))
        assert np.all(np.diag(D)==0) and np.array_equal(D,D.T)
    bound=1e-11*(1+max(float(np.max(np.abs(x))) for x in refs));assert max(errors)<bound
    records.append(dict(n=n,maximum_error=max(errors),bound=bound))
once(HERE/'unit_result.json',dict(passed=True,records=records,targets_accessed=False,seal_sha256=sha(HERE/'seal.json')))
print(json.dumps(dict(passed=True,records=records)))

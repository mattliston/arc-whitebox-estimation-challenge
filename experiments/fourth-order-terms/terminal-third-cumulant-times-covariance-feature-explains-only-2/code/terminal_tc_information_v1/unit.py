"""Independent small dense tensor verification of the specified TC sector."""
from pathlib import Path
import hashlib,json,resource,time
import numpy as np
from algebra import Products, owners, query_and_plane

H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=json.loads((H/'source_sealed.json').read_text())
for name,digest in seal['files_sha256'].items():assert sha(H/name)==digest
t0=time.monotonic();rng=np.random.default_rng(2026092017);records=[]
for n in (3,5,7):
    P=rng.normal(size=(n,8));lam=rng.normal(size=8)
    W=rng.normal(size=(n,n));A=rng.normal(size=(n,n));previous=np.arange(n-1)
    B=rng.normal(size=(n,n));B=(B+B.T)/2
    C=rng.normal(size=(n,n));C=C@C.T+np.eye(n)
    b=rng.normal(size=n)
    packet=dict(P=P,**{'lambda':lam},W14=W,star=A,previous=previous,
                previous_alive=np.arange(n)<n-1,Bpre=B,Cpre=C,bpre=b)
    mm=Products();pp,ll,V,U,bb,BB,CC=owners(packet,mm)
    Ae=A.copy();Ae[-1,:]=0;Ae[:,-1]=0;Ae[-1,-1]=A[-1,-1]
    S=np.zeros((n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                S[i,j,k]=((Ae[i,k] if j==k else 0)
                          +(Ae[j,i] if k==i else 0)
                          +(Ae[k,j] if i==j else 0))/3
    # The off-diagonal input star uses A_(single,repeated).
    T=np.einsum('ia,jb,kc,ijk->abc',W,W,W,S)
    T+=np.einsum('ir,jr,kr,r->ijk',P,P,P,lam)
    T+=b[:,None,None]*B[None,:,:]+b[None,:,None]*B[:,None,:]+b[None,None,:]*B[:,:,None]
    g=rng.uniform(.15,.9,size=n);d=rng.normal(size=n);rays=rng.normal(size=(n,4));Z=g[:,None]*rays
    Q,K=query_and_plane(pp,ll,V,U,bb,BB,Z,mm)
    qref=np.einsum('ijk,ja,ka->ia',T,Z,Z)
    kref=np.einsum('iij->ij',T)
    F=np.zeros((n,n,n,n))
    for a in range(n):
        for c in range(n):
            for e in range(n):
                for f in range(n):
                    ix=[a,c,e,f]
                    for h in range(4):
                        i=ix[h];other=[ix[j] for j in range(4) if j!=h]
                        j,k,l=other
                        F[a,c,e,f]+=d[i]*g[j]*g[k]*g[l]*(C[i,j]*T[i,k,l]+C[i,k]*T[i,j,l]+C[i,l]*T[i,j,k])
    actual=12*np.sum(d[:,None]*rays*(C@Z)*Q,axis=0)
    expected=np.einsum('ijkl,ia,ja,ka,la->a',F,rays,rays,rays,rays)
    trace=4*((d*np.diag(C)*g)@(K.T@(g*g))+2*(d*g)@((C*K)@(g*g)))
    trace_ref=np.einsum('iijj->',F)
    errors={name:float(np.max(np.abs(x-y))/max(1.,np.max(np.abs(y)))) for name,x,y in [('query',Q,qref),('plane',K,kref),('ray',actual,expected),('trace',np.asarray(trace),np.asarray(trace_ref))]}
    assert max(errors.values())<=1e-11,errors
    records.append(dict(n=n,errors=errors))
# Exact cancellation checks the signed sum, not per-atom energy.
P=np.array([[1.,2.],[3.,4.],[5.,6.]]);P=np.concatenate([P,P],axis=1)
z=np.arange(6,dtype=float).reshape(3,2);mm=Products()
q,k=query_and_plane(P,np.array([1.,2.,-1.,-2.]),np.eye(3),np.zeros((3,3)),np.zeros(3),np.zeros((3,3)),z,mm)
assert np.array_equal(q,np.zeros_like(q)) and np.array_equal(k,np.zeros_like(k))
record=dict(passed=True,records=records,exact_cp_cancellation=True,source_seal_sha256=sha(H/'source_sealed.json'),elapsed_s=time.monotonic()-t0,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (H/'unit.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)

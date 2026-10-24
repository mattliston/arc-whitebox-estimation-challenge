"""Independent explicit third and12-term fourth tensors, no network data."""
from datetime import datetime,timezone
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from source import Third,post_q22,read_q31,source

H=Path(__file__).resolve().parent;assert not (H/'units.json').exists()
rng=np.random.default_rng(6092734)


def dense(third):
    p,l,U,V,b,B=third.p,third.lam,third.U,third.V,third.b,third.B
    T=np.einsum('ia,ja,ka,a->ijk',p,p,p,l)
    T+=(np.einsum('ia,ja,ka->ijk',U,V,V)+np.einsum('ia,ja,ka->ijk',V,U,V)+np.einsum('ia,ja,ka->ijk',V,V,U))/3
    T+=np.einsum('i,jk->ijk',b,B)+np.einsum('j,ik->ijk',b,B)+np.einsum('k,ij->ijk',b,B)
    if third.repair is not None:
        A=third.repair
        for i,j,k in itertools.product(range(len(b)),repeat=3):
            T[i,j,k]+=(A[i,j]*(j==k)+A[j,i]*(i==k)+A[k,i]*(i==j))/3
    return T


def tensor4(C,T,g,d):
    out=np.zeros((len(g),)*4)
    for idx in itertools.product(range(len(g)),repeat=4):
        value=0.
        for root in range(4):
            other=[k for k in range(4) if k!=root]
            factor=d[idx[root]]*np.prod([g[idx[k]] for k in other])
            for neighbor in other:
                leaves=[k for k in other if k!=neighbor]
                value+=factor*C[idx[root],idx[neighbor]]*T[idx[root],idx[leaves[0]],idx[leaves[1]]]
        out[idx]=value
    return out


records=[]
for n,m in ((3,2),(4,5)):
    p=rng.normal(size=(n,6));l=rng.normal(size=6)
    U=rng.normal(size=(n,3));V=rng.normal(size=(n,3));b=rng.normal(size=n)
    A=rng.normal(size=(n,n));B=A@A.T/n;C=B+np.eye(n)
    third=Third(p,l,U,V,b,B);T=dense(third)
    X=rng.normal(size=(n,m));Y=rng.normal(size=(n,m))
    errors={}
    errors['bilinear']=float(np.max(np.abs(third.bilinear(X,Y)-np.einsum('ijk,jm,km->im',T,X,Y))))
    errors['plane']=float(np.max(np.abs(third.plane()-np.einsum('iij->ij',T))))
    target=third.plane()+rng.normal(size=(n,n))
    third.restore(target);T=dense(third)
    errors['restored_plane']=float(np.max(np.abs(third.plane()-target)))
    errors['restored_tensor']=float(np.max(np.abs(np.einsum('iij->ij',T)-target)))
    errors['restored_bilinear']=float(np.max(np.abs(third.bilinear(X,Y)-np.einsum('ijk,jm,km->im',T,X,Y))))
    g=rng.normal(size=n);d=rng.normal(size=n);W=rng.normal(size=(n,m))
    Q=tensor4(C,T,g,d)
    want=np.einsum('abcd,ai,bi,ci,dj->ij',Q,W,W,W,W,optimize=True)
    got,k4=read_q31(C,third,g,d,W)
    errors['q31']=float(np.max(np.abs(got-want)))
    errors['k4']=float(np.max(np.abs(k4-np.diag(want))))
    errors['q22']=float(np.max(np.abs(post_q22(C,target,g,d)-np.einsum('iijj->ij',Q))))
    scale=max(1.,float(np.max(np.abs(Q))),float(np.max(np.abs(want))))
    assert max(errors.values())<=2e-11*scale,errors
    h=1e-5;direction=rng.normal(size=W.shape)
    _,plus=read_q31(C,third,g,d,W+h*direction);_,minus=read_q31(C,third,g,d,W-h*direction)
    analytic=4*np.diag(read_q31(C,third,g,d,np.column_stack((W,direction)))[0][:m,m:])
    fd=float(np.max(np.abs((plus-minus)/(2*h)-analytic))/max(1.,np.max(np.abs(analytic))))
    assert fd<1e-8,fd
    zero=Third(p,np.zeros_like(l),np.zeros_like(U),V,np.zeros_like(b),B)
    assert np.array_equal(read_q31(C,zero,g,d,W)[0],np.zeros((m,m)))
    _,_,info,_,_=source(C,third,target,g,d,W,C)
    radial=info['beta']*(np.outer(np.diag(C),np.diag(C))+2*C*C)/3
    trace_error=float(abs(np.sum(post_q22(C,target,g,d)-radial)))
    assert trace_error<1e-11*max(1.,abs(info['trace']))
    records.append(dict(n=n,m=m,errors=errors,fd_relative_error=fd,trace_error=trace_error))
out=dict(passed=True,records=records,source_sha256=hashlib.sha256((H/'source.py').read_bytes()).hexdigest(),
         utc_finished=datetime.now(timezone.utc).isoformat(),truth_accessed=False)
with (H/'units.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2),flush=True)

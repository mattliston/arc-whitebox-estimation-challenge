"""Exact one-high projection queries for an offline residual information test."""
import itertools
import numpy as np


def plane(slices, U, W=None, indices=None):
    """K21 rows of (I-Q^3)T from M[a,j,k]=T(U[:,a],e_j,e_k)."""
    q,n,_=slices.shape
    indices=np.arange(n) if indices is None else np.asarray(indices)
    H=U.T if W is None else U.T@W
    # Double/triple projected slices. Keep contraction order fixed.
    N=np.einsum('jb,ajk->abk',U,slices,optimize=True)
    core=np.einsum('kc,abk->abc',U,N,optimize=True)
    NW=N if W is None else N@W
    out=np.zeros((len(indices),H.shape[1]))
    for a,M in enumerate(slices):
        A=M[indices] if W is None else W[:,indices].T@M@W
        diagonal=A[np.arange(len(indices)),indices]
        out+=2*H[a,indices,None]*A+diagonal[:,None]*H[a,None,:]
    out-=np.einsum('ai,bi,abj->ij',H[:,indices],H[:,indices],NW,optimize=True)
    out-=2*np.einsum('ai,bj,abi->ij',H[:,indices],H,NW[:,:,indices],optimize=True)
    out+=np.einsum('abc,ai,bi,cj->ij',core,H[:,indices],H[:,indices],H,optimize=True)
    return out


def repeated_push(K,W,indices):
    """Exact push of the coordinate-star tensor with repeated plane K."""
    V=W[:,indices]
    square=V*V
    quadratic=K.T@square+2*V*(K@V)-2*np.diag(K)[:,None]*square
    return quadratic.T@W


def distinct_plane(slices,U,W,indices):
    coordinate=plane(slices,U)
    return plane(slices,U,W,indices)-repeated_push(coordinate,W,indices)


def certificate():
    rng=np.random.default_rng(294823)
    records=[]
    for n,q in ((8,2),(12,4)):
        raw=rng.normal(size=(n,n,n))
        T=sum(raw.transpose(p) for p in itertools.permutations(range(3)))/6
        U=np.linalg.qr(rng.normal(size=(n,q)))[0]
        W=rng.normal(size=(n,n))/np.sqrt(n)
        M=np.einsum('ia,ijk->ajk',U,T)
        Q=np.eye(n)-U@U.T
        dense=T-np.einsum('ia,jb,kc,abc->ijk',Q,Q,Q,T,optimize=True)
        keep=np.ones((n,n,n),bool)
        ii,jj,kk=np.indices((n,n,n))
        keep&=(ii!=jj)&(ii!=kk)&(jj!=kk)
        repeated=dense[range(n),range(n),:]
        pushed=np.einsum('ia,jb,kc,ijk->abc',W,W,W,dense,optimize=True)
        distinct=np.einsum('ia,jb,kc,ijk->abc',W,W,W,dense*keep,optimize=True)
        indices=np.arange(0,n,2)
        errors=dict(coordinate=float(np.max(np.abs(plane(M,U)-repeated))),
                    full_push=float(np.max(np.abs(plane(M,U,W,indices)-pushed[indices,indices,:]))),
                    distinct_push=float(np.max(np.abs(distinct_plane(M,U,W,indices)-distinct[indices,indices,:]))))
        assert max(errors.values())<2e-13,errors
        records.append(dict(width=n,q=q,errors=errors))
    return dict(passed=True,cells=records)


if __name__=='__main__':
    import json
    from pathlib import Path
    out=Path(__file__).resolve().with_name('residual_one_high_oracle_certificate_v1.json')
    assert not out.exists()
    result=certificate()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

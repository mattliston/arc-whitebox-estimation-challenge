"""CP-sector one-high matrices and exact paired-birth slice compiler."""
import numpy as np
from linear_transport_residual_source_v1 import coord_slice


def plane(U,L):
    n=len(U);out=np.zeros((n,n))
    for u,M in zip(U.T,L):out+=2*u[:,None]*M+np.diag(M)[:,None]*u[None,:]
    return out


def slices(U,L,H):
    out=np.zeros((H.shape[1],len(U),len(U)))
    for u,M in zip(U.T,L):
        R=M@H;v=u@H
        for a,h in enumerate(H.T):out[a]+=v[a]*M+np.outer(u,R[:,a])+np.outer(R[:,a],u)
    return out


def coefficients(M,U):
    N=np.stack([m@U for m in M]) # a,k,b
    core=np.stack([U.T@n for n in N]) # a,b,c
    result=[]
    for a,m in enumerate(M):
        A=U@N[a].T
        result.append(m-.5*(A+A.T)+(U@core[a]@U.T)/3)
    return np.stack(result)


def birth_slices(B,birth,u,delta,H,births,alive,mat32=False):
    B=B*alive[:,None];b=birth*births*alive
    linear=b*u;diagonal=b*(3*u*u+delta*delta)/4
    K=linear[:,None]*B.T;np.fill_diagonal(K,diagonal)
    A=3*K.T-2*np.diag(np.diag(K));out=[]
    for h in H.T:
        left=B*(b*h)[None,:]
        X=(left.astype(np.float32)@B.astype(np.float32).T).astype(np.float64) if mat32 else left@B.T
        Y=(b*(B.T@h))[:,None]*B.T
        out.append(X+Y+Y.T+coord_slice(A,h))
    return np.stack(out)


def dense(U,L):
    T=np.einsum('ia,ajk->ijk',U,L)
    return T+T.transpose(1,0,2)+T.transpose(2,1,0)


def certificate():
    rng=np.random.default_rng(26952002);records=[]
    for n,q in ((8,3),(12,4)):
        from itertools import permutations
        raw=rng.normal(size=(n,n,n));T=sum(raw.transpose(p) for p in permutations(range(3)))/6
        U=np.linalg.qr(rng.normal(size=(n,q)))[0];Q=np.eye(n)-U@U.T
        M=np.einsum('ia,ijk->ajk',U,T);L=coefficients(M,U)
        exact=T-np.einsum('ia,jb,kc,ijk->abc',Q,Q,Q,T,optimize=True)
        W=rng.normal(size=(n,n))/np.sqrt(n);V=W.T@U;R=np.stack([W.T@m@W for m in L])
        pushed=np.einsum('ia,jb,kc,ijk->abc',W,W,W,exact,optimize=True)
        errors=dict(projection=float(np.max(np.abs(dense(U,L)-exact))),
                    plane=float(np.max(np.abs(plane(V,R)-pushed[np.arange(n),np.arange(n),:]))))
        B=rng.normal(size=(n,n))*.1;np.fill_diagonal(B,0)
        birth=rng.uniform(.1,.3,n);u=rng.normal(size=n)*.1;delta=rng.uniform(.1,.5,n)
        births=rng.integers(0,2,n).astype(bool);alive=rng.integers(0,2,n).astype(bool)
        p1=B.copy();p2=B.copy();np.fill_diagonal(p1,(u+delta)/2);np.fill_diagonal(p2,(u-delta)/2)
        p1*=alive[:,None];p2*=alive[:,None];weights=birth/delta*births
        born=np.einsum('ia,ja,ka,a->ijk',p1,p1,p1,weights)-np.einsum('ia,ja,ka,a->ijk',p2,p2,p2,weights)
        H=rng.normal(size=(n,q));actual=birth_slices(B,birth,u,delta,H,births,alive)
        errors['birth_slices']=float(np.max(np.abs(actual-np.einsum('ia,ijk->ajk',H,born))))
        target=rng.normal(size=(n,n));rest=target-plane(U,L);star=3*rest.T-2*np.diag(np.diag(rest))
        coordinate=star.T/3;np.fill_diagonal(coordinate,np.diag(star))
        errors['coordinate_restoration']=float(np.max(np.abs(plane(U,L)+coordinate-target)))
        assert max(errors.values())<2e-12,errors
        records.append(dict(width=n,q=q,errors=errors))
    return dict(passed=True,records=records)


if __name__=='__main__':
    import json
    from pathlib import Path
    out=Path(__file__).resolve().with_name('cp_only_one_high_certificate_v1.json');assert not out.exists()
    record=certificate();out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

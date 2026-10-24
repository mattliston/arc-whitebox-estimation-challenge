"""Specified fixed source sum queried along arbitrary backward directions."""
import numpy as np
from linear_transport_residual_source_v1 import coord_slice,trace_plane
from full_edge_gaussian_bond3_v1 import slices as gaussian_slices,dense as gaussian_dense
from residual_one_high_oracle_v1 import repeated_push


def trace_slice(b,B,h):
    bh=B@h
    return (b@h)*B+np.outer(b,bh)+np.outer(bh,b)


def source_slices(star,W,g,d,e,C,bpre,Bpre,Bpost,radial,star_pre,H):
    J=W*g[None,:];c=g*bpre;D=g[:,None]*Bpre*g[None,:]-Bpost
    bnew=radial*d*np.diag(C)/6;Dnew=g[:,None]*C*g[None,:]-Bpost
    K=(g*g)[:,None]*g[None,:]*star_pre+trace_plane(c,D)+trace_plane(bnew,Dnew)
    A=3*K.T-2*np.diag(np.diag(K))
    answer=gaussian_slices(C,g,d,e,H)
    for a,h in enumerate(H.T):
        answer[a]+=J.T@coord_slice(star,J@h)@J+trace_slice(c,D,h)+trace_slice(bnew,Dnew,h)-coord_slice(A,h)
    return answer


def dense_source(star,W,g,d,e,C,bpre,Bpre,Bpost,radial):
    n=len(g);I=np.eye(n);S=np.einsum('ij,jk->ijk',star,I)
    S=(S+S.transpose(1,0,2)+S.transpose(2,1,0))/3
    J=W*g[None,:]
    T=np.einsum('ia,jb,kc,ijk->abc',J,J,J,S,optimize=True)
    c=g*bpre;D=g[:,None]*Bpre*g[None,:]-Bpost
    b=radial*d*np.diag(C)/6;Dnew=g[:,None]*C*g[None,:]-Bpost
    for z,B in ((c,D),(b,Dnew)):
        T+=np.einsum('i,jk->ijk',z,B)+np.einsum('j,ik->ijk',z,B)+np.einsum('k,ij->ijk',z,B)
    ii,jj,kk=np.indices(T.shape);T*=((ii!=jj)&(ii!=kk)&(jj!=kk))
    return T+gaussian_dense(C,g,d,e)


def certificate():
    rng=np.random.default_rng(261935);n=8;depth=4
    U=np.linalg.qr(rng.normal(size=(n,3)))[0];rows=[]
    for l in range(depth):
        star=rng.normal(size=(n,n))*.03;W=rng.normal(size=(n,n))/np.sqrt(n);g=rng.uniform(.1,.9,n)
        d=rng.uniform(.1,.5,n);e=rng.normal(size=n)*.2
        x=rng.normal(size=(n,n));C=x@x.T/n+np.eye(n)*.3
        x=rng.normal(size=(n,n));Bpre=x@x.T/n
        x=rng.normal(size=(n,n));Bpost=x@x.T/n
        bpre=rng.normal(size=n)*.01;radial=.03
        Kold=star.T.copy()/3;np.fill_diagonal(Kold,np.diag(star))
        row=dict(star=star,W=W,g=g,d=d,e=e,C=C,bpre=bpre,Bpre=Bpre,Bpost=Bpost,radial=radial)
        row['star_pre']=repeated_push(Kold,W,np.arange(n));rows.append(row)
    A=np.eye(n);total=np.zeros((len(U.T),n,n));dense=np.zeros((n,n,n));maximum=0.
    for row in reversed(rows):
        H=A@U;M=source_slices(**row,H=H)
        T=dense_source(**{k:v for k,v in row.items() if k!='star_pre'})
        maximum=max(maximum,float(np.max(np.abs(M-np.einsum('ia,ijk->ajk',H,T)))))
        total+=np.stack([A.T@m@A for m in M])
        dense+=np.einsum('ia,jb,kc,ijk->abc',A,A,A,T,optimize=True)
        A=(row['W']*row['g'][None,:])@A
    target=np.einsum('ia,ijk->ajk',U,dense)
    error=float(np.max(np.abs(total-target)));assert max(error,maximum)<1e-12
    return dict(passed=True,width=n,depth=depth,source_query_max_error=maximum,multistep_max_error=error)

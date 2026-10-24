"""Exact missing star plus trace-metric transport, before q4 projection."""
import numpy as np
from residual_one_high_oracle_v1 import repeated_push,distinct_plane


def coord_slice(A,h):
    return (A*h[None,:]+h[:,None]*A.T+np.diag(A.T@h))/3


def trace_plane(b,B):
    return np.diag(B)[:,None]*b[None,:]+2*B*b[:,None]


def source(star,W,g,bpre,Bpre,Bpost,U,next_W,indices):
    J=W*g[None,:];c=g*bpre;D=g[:,None]*Bpre*g[None,:]-Bpost
    Kold=star.T.copy()/3;np.fill_diagonal(Kold,np.diag(star))
    Ksource=repeated_push(Kold,J,np.arange(len(g)))+trace_plane(c,D)
    A=3*Ksource.T-2*np.diag(np.diag(Ksource))
    matrices=[];star_norm=0.;metric_norm=0.
    for u in U.T:
        Ms=J.T@coord_slice(star,J@u)@J
        Du=D@u;Mt=(c@u)*D+np.outer(c,Du)+np.outer(Du,c)
        matrices.append(Ms+Mt-coord_slice(A,u))
        star_norm+=np.sum(Ms*Ms);metric_norm+=np.sum(Mt*Mt)
    matrices=np.stack(matrices)
    full=repeated_push(Kold,J@next_W,indices)
    dn=next_W.T@D@next_W;cn=next_W.T@c
    full+=trace_plane(cn,dn)[indices]
    full-=repeated_push(Ksource,next_W,indices)
    projected=distinct_plane(matrices,U,next_W,indices)
    return dict(slices=matrices,full=full,projected=projected,
                star_slice_norm=np.sqrt(star_norm),metric_slice_norm=np.sqrt(metric_norm))


def certificate():
    rng=np.random.default_rng(18352026);records=[]
    for n,q in ((8,2),(12,4)):
        star=rng.normal(size=(n,n))*.03;W=rng.normal(size=(n,n))/np.sqrt(n)
        g=rng.uniform(.05,.95,n);b=rng.normal(size=n)*.01
        x=rng.normal(size=(n,n));B=x@x.T/n
        x=rng.normal(size=(n,n));C=x@x.T/n
        U=np.linalg.qr(rng.normal(size=(n,q)))[0];wn=rng.normal(size=(n,n))/np.sqrt(n)
        I=np.eye(n);T=np.einsum('ij,jk->ijk',star,I)
        T=(T+T.transpose(1,0,2)+T.transpose(2,1,0))/3
        J=W*g[None,:];dense=np.einsum('ia,jb,kc,ijk->abc',J,J,J,T,optimize=True)
        c=g*b;D=g[:,None]*B*g[None,:]-C
        dense+=np.einsum('i,jk->ijk',c,D)+np.einsum('j,ik->ijk',c,D)+np.einsum('k,ij->ijk',c,D)
        ii,jj,kk=np.indices((n,n,n));dense*=((ii!=jj)&(ii!=kk)&(jj!=kk))
        idx=np.arange(0,n,2);r=source(star,W,g,b,B,C,U,wn,idx)
        expected=np.einsum('ia,ijk->ajk',U,dense)
        pushed=np.einsum('ia,jb,kc,ijk->abc',wn,wn,wn,dense,optimize=True)
        e=dict(slices=float(np.max(np.abs(r['slices']-expected))),full_query=float(np.max(np.abs(r['full']-pushed[idx,idx,:]))))
        assert max(e.values())<1e-12,e
        records.append(dict(width=n,q=q,errors=e))
    return dict(passed=True,records=records)

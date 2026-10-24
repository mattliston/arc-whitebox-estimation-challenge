"""Four-matrix CP replacement with separate exact-coordinate star/trace."""
import numpy as np
from scipy.special import ndtr
import radial_compact_feed_fullpair_v1 as parent
from radial_binary_compact import select
from compact_scalar_final_v1 import nonlinear_scalar
from cp_only_one_high_math_v1 import plane,slices,coefficients,birth_slices

nonlinear=parent.env['nonlinear'];star_push=parent.env['star_push']


def next_basis(C,V,indices):
    A=C[np.ix_(indices,indices)]
    Q=V[indices].copy()
    for _ in range(4):
        Q,R=np.linalg.qr(A@Q,mode='reduced')
        Q*=np.where(np.diag(R)<0,-1.,1.)[None,:]
    U=np.zeros_like(V);U[indices]=Q
    assert np.max(np.abs(U.T@U-np.eye(4)))<1e-12
    return U


def predict(W,hook=None,final_hook=None,cov32=True,metric='covariance',collect_states=False,gate_floor=.01):
    depth,n,_=W.shape;assert n==1024 and depth==16
    m=np.zeros(n);C=np.eye(n);radial=0.;b=np.zeros(n)
    star=np.zeros((n,n),np.float32);previous_alive=np.ones(n,dtype=bool)
    U=np.eye(n)[:,:4].copy();L=np.zeros((4,n,n));means=[];states=[]
    for l in range(depth):
        w32=W[l].astype(np.float32);w=W[l].astype(np.float64)
        mu=w.T@m;V=w.T@U
        if l==depth-1:
            pushed=C.astype(np.float32)@w32
            v=np.sum(w32.astype(np.float64)*pushed.astype(np.float64),axis=0)
            metric_cov=C.copy();np.fill_diagonal(metric_cov,np.diag(C)*previous_alive)
            metric_push=metric_cov.astype(np.float32)@w32
            bv=np.sum(w32.astype(np.float64)*metric_push.astype(np.float64),axis=0)
            k3=np.zeros(n)
            for a,M in enumerate(L):
                pushed=M.astype(np.float32)@w32
                diagonal=np.sum(w32.astype(np.float64)*pushed.astype(np.float64),axis=0)
                k3+=3*V[:,a]*diagonal
            k3+=star_push(W[l],star,diagonal_only=True)+3*bv*(w.T@b)
            m,aux=nonlinear_scalar(mu,v,k3,radial*v*v)
            if final_hook is not None:m=m+np.asarray(final_hook(l,m,aux,W[l]),np.float64)
            means.append(m.copy());break
        Cpre=(w32.T@C.astype(np.float32)@w32).astype(np.float64) if cov32 else w.T@C@w
        metric_cov=C.copy();np.fill_diagonal(metric_cov,np.diag(C)*previous_alive)
        Bpre=(w32.T@metric_cov.astype(np.float32)@w32).astype(np.float64);bpre=w.T@b
        vv=np.diag(Cpre);indices,alive=select(ndtr(mu/np.sqrt(np.maximum(vv,1e-10))),gate_floor)
        live=alive[:,None]*alive[None,:]
        R=np.zeros_like(L) if l==0 else np.stack([(w32.T@M.astype(np.float32)@w32).astype(np.float64) for M in L])
        cp_pre=plane(V,R);diagonal=np.diag(cp_pre).copy();cp_pre*=live;np.fill_diagonal(cp_pre,diagonal)
        star_pre=star_push(W[l],star);diagonal=np.diag(star_pre).copy();star_pre*=live;np.fill_diagonal(star_pre,diagonal)
        K21=cp_pre+star_pre+np.diag(Bpre)[:,None]*bpre[None,:]+2*Bpre*bpre[:,None]
        k4=radial*vv*vv;k22=radial*(np.outer(vv,vv)+2*Cpre*Cpre)/3
        m,C,target,radial_new,aux=nonlinear(mu,Cpre,K21,k4,k22,metric);host=m.copy()
        if hook is not None:
            dm=np.asarray(hook(l,m,C,K21,aux,W[l]),np.float64)
            C=C-np.outer(m,dm)-np.outer(dm,m)-np.outer(dm,dm);m=m+dm
        means.append(m.copy())
        g=aux['gate'];d=aux['pdf']/aux['sigma'];birth=d*(1+radial/3)
        b=(g*bpre+radial*d*vv/6)*alive
        selected,births=select(np.minimum(g,1-g),gate_floor)
        diagonal=np.diag(C).copy();C*=live;np.fill_diagonal(C,diagonal)
        diagonal=np.diag(target).copy();target*=live;np.fill_diagonal(target,diagonal)
        inherited=g[:,None]**2*g[None,:]*cp_pre
        Bpost=C.copy();np.fill_diagonal(Bpost,np.diag(Bpost)*alive)
        trace_post=np.diag(Bpost)[:,None]*b[None,:]+2*Bpost*b[:,None]
        residual=target-inherited-trace_post
        beta=2*(aux['gaussian_mean']-host*g);innovation=np.diag(residual).copy()
        safe_birth=np.maximum(birth,1e-12);u=np.clip(beta/safe_birth,-8*vv,8*vv)
        disc=4*innovation/safe_birth-3*u*u
        delta=np.minimum(np.sqrt(np.maximum(disc,.01*vv*vv)),16*vv)
        bp=g[:,None]*Cpre;np.fill_diagonal(bp,0)
        Unew=next_basis(C,V,indices);ga=g*alive
        Rpost=R*ga[None,:,None]*ga[None,None,:]
        M=slices(V*ga[:,None],Rpost,Unew)
        M+=birth_slices(bp,birth,u,delta,Unew,births,alive,mat32=True)
        U=Unew;L=coefficients(M,U)
        cp_post=plane(U,L);rest=target-cp_post-trace_post
        star=(3*rest.T-2*np.diag(np.diag(rest))).astype(np.float32)
        if collect_states:
            stats=dict(layer=l,active=int(alive.sum()),births=len(selected),matrix_channels=4,
                       coordinate_plane_max_error=float(np.max(np.abs(cp_post+trace_post+star.T/3+np.diag(2*np.diag(star)/3)-target))),
                       covariance_subspace_trace_fraction=float(np.trace(U.T@C@U)/np.trace(C)))
            states.append(stats)
        previous_alive=alive;radial=radial_new
    return np.stack(means),states

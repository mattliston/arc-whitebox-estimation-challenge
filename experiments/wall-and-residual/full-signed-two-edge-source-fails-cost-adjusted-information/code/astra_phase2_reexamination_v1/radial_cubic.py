"""One-factor cubic births, preserving current repeated planes exactly.

Approximation, not an algebraic equality: replace mixed p,p,e births by
one cube with its diagonal fixed by the local third-cumulant innovation.
Unlike a large-shift polarization, this creates no huge canceling star.
One factor push and one repeated-plane GEMM replace two of each.
"""
import numpy as np
from k3_radial_numpy import nonlinear,star_push

def plane(p,lam):
    if p.shape[1]==0:return np.zeros((p.shape[0],p.shape[0]),np.float64)
    return ((p*p)@(p*lam).T).astype(np.float64)

def predict(W,hook=None,cov32=True,metric="covariance",collect_states=False,birth_mode="distinct"):
    depth,n,_=W.shape;m=np.zeros(n);C=np.eye(n);radial=0.0
    p=np.zeros((n,0),np.float32);lam=np.zeros(0,np.float32);star=np.zeros((n,n),np.float32)
    means=[];states=[]
    for l in range(depth):
        w32=W[l].astype(np.float32);w=W[l].astype(np.float64)
        mu=w.T@m
        Cpre=(w32.T@C.astype(np.float32)@w32).astype(np.float64) if cov32 else w.T@C@w
        pp=w32.T@p;cp_pre=plane(pp,lam);K21=cp_pre+star_push(W[l],star)
        vv=np.diag(Cpre);k4=radial*vv*vv;k22=radial*(np.outer(vv,vv)+2*Cpre*Cpre)/3
        m,C,target,radial_new,aux=nonlinear(mu,Cpre,K21,k4,k22,metric)
        if collect_states:states.append(dict(mean=m.copy(),cov=C.copy(),k21=target.copy(),radial=radial_new,**aux))
        if hook is not None:
            dm=np.asarray(hook(l,m,C,K21,aux,W[l]),np.float64)
            C=C-np.outer(m,dm)-np.outer(dm,m)-np.outer(dm,dm);m=m+dm
        means.append(m.copy())
        if l==depth-1:break
        g=aux["gate"];birth=aux["pdf"]/aux["sigma"]
        inherited=g[:,None]**2*g[None,:]*cp_pre
        bp=g[:,None]*Cpre
        diag_innovation=np.maximum(np.diag(target-inherited),1e-12)
        if birth_mode=="distinct":
            bd=np.sqrt(diag_innovation/np.maximum(birth,1e-12))
        elif birth_mode=="mixed":
            off=bp.T.copy();np.fill_diagonal(off,0)
            resid=target-inherited
            b=np.sum(resid*off,axis=1)/np.maximum(np.sum(off*off,axis=1),1e-24)
            b=np.maximum(b,1e-10)
            bd=diag_innovation/b
        else:raise ValueError(birth_mode)
        # Avoid extreme inactive-node factors; their innovation is negligible.
        bd=np.clip(bd,1e-5*vv,10*vv)
        np.fill_diagonal(bp,bd)
        blam=diag_innovation/bd**3
        born=plane(bp.astype(np.float32),blam.astype(np.float32))
        cp_post=inherited+born
        p=np.concatenate([g.astype(np.float32)[:,None]*pp,bp.astype(np.float32)],axis=1)
        lam=np.concatenate([lam,blam.astype(np.float32)])
        residual=target-cp_post
        star=(3*residual.T-2*np.diag(np.diag(residual))).astype(np.float32)
        radial=radial_new
    return np.stack(means),states

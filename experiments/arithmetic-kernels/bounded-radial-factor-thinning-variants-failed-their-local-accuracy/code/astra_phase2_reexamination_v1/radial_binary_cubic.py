"""Two signed cubes capture the binary Gaussian birth plus leading star.

a[(x+t*y)^3-(x+s*y)^3] has mixed coefficients lambda=a(t-s),
beta=lambda(t+s), and diagonal D=lambda(t^2+t*s+s^2).
Here x is an off-diagonal covariance column and y a coordinate vector.
Unlike one cube, the spurious x^3 term cancels algebraically at all depths.
Unrepresented repeated entries remain in the one-step star.
"""
import numpy as np
from k3_radial_numpy import nonlinear,star_push
from radial_cubic import plane

def predict(W,hook=None,cov32=True,metric="covariance",collect_states=False,birth_mode="analytic"):
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
        host=m.copy()
        if hook is not None:
            dm=np.asarray(hook(l,m,C,K21,aux,W[l]),np.float64)
            C=C-np.outer(m,dm)-np.outer(dm,m)-np.outer(dm,dm);m=m+dm
        means.append(m.copy())
        if l==depth-1:break
        g=aux["gate"];birth=aux["pdf"]/aux["sigma"]
        inherited=g[:,None]**2*g[None,:]*cp_pre
        bp=g[:,None]*Cpre;np.fill_diagonal(bp,0)
        residual=target-inherited
        if birth_mode=="analytic":
            beta=2*(aux["gaussian_mean"]-host*g)
        elif birth_mode=="fitted":
            remainder=residual-bp*bp*birth[None,:]
            beta=np.sum(remainder*bp.T,axis=1)/np.maximum(np.sum(bp.T**2,axis=1),1e-24)
        else:raise ValueError(birth_mode)
        innovation=np.diag(residual).copy()
        safe_birth=np.maximum(birth,1e-12)
        u=beta/safe_birth
        # Bounds only regularize nearly deterministic nodes where all birth
        # coefficients are negligible; never clip a covariance eigenvalue.
        u=np.clip(u,-8*vv,8*vv)
        disc=4*innovation/safe_birth-3*u*u
        delta=np.sqrt(np.maximum(disc,.01*vv*vv))
        delta=np.minimum(delta,16*vv)
        t=(u+delta)/2;s=(u-delta)/2;a=birth/delta
        p1=bp.copy();p2=bp.copy();np.fill_diagonal(p1,t);np.fill_diagonal(p2,s)
        born=bp*bp*birth[None,:]+beta[:,None]*bp.T
        np.fill_diagonal(born,birth*(3*u*u+delta*delta)/4)
        cp_post=inherited+born
        p=np.concatenate([g.astype(np.float32)[:,None]*pp,p1.astype(np.float32),p2.astype(np.float32)],axis=1)
        lam=np.concatenate([lam,a.astype(np.float32),-a.astype(np.float32)])
        rest=target-cp_post
        star=(3*rest.T-2*np.diag(np.diag(rest))).astype(np.float32)
        if collect_states:states.append(dict(layer=l,clipped_disc=int(np.sum(disc<.01*vv*vv)),star_rms=float(np.sqrt(np.mean(star*star)))))
        radial=radial_new
    return np.stack(means),states

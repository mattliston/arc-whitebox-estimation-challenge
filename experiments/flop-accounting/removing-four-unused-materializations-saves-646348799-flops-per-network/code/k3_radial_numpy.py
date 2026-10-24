"""Independent NumPy implementation of the SIMPLE K3 power-cumulant map.

Offline reference only; public estimator will use FlopScope exclusively.
Full covariance-path CP births, one-step coordinate-star residual, scalar
radial fourth cumulant. Mean/power-cumulant arithmetic is float64; carrier
matrix multiplications are float32. Weights use (input,output) convention.
"""
import numpy as np
from scipy.special import ndtr

def cp_plane(p,q,lam):
    if p.shape[1]==0:
        return np.zeros((p.shape[0],p.shape[0]),dtype=np.float64)
    return ((p*p)@(q*lam).T+2*(p*q)@(p*lam).T).astype(np.float64)

def cp_diagonal(p,q,lam):
    return (3*np.sum(p*p*q*lam,axis=1)).astype(np.float64)

def star_push(W,star,diagonal_only=False):
    w = W.T.astype(np.float32)
    A = star.astype(np.float32)
    if diagonal_only:
        return np.sum((w@A)*w*w,axis=1).astype(np.float64)
    wa = w@A
    return (2*(wa*w)@w.T+(w*w)@wa.T).astype(np.float64)/3

def gaussian_raw(mu,v):
    v = np.maximum(v,1e-10)
    s = np.sqrt(v); a = mu/s
    pdf = np.exp(-a*a/2)/np.sqrt(2*np.pi); g = ndtr(a)
    m = s*pdf+mu*g
    r2 = (mu*mu+v)*g+mu*s*pdf
    r3 = mu*(mu*mu+3*v)*g+(mu*mu+2*v)*s*pdf
    r4 = (mu**4+6*mu*mu*v+3*v*v)*g+mu*(mu*mu+5*v)*s*pdf
    return s,a,pdf,g,m,r2,r3,r4

def nonlinear(mu,C,K21,k4,k22_override=None,metric="identity"):
    n = len(mu); v = np.diag(C).copy()
    s,a,pdf,g,mg,r2g,r3g,r4g = gaussian_raw(mu,v)
    d = pdf/s; e = -a*pdf/v; f = (a*a-1)*pdf/(v*s)
    k3 = np.diag(K21).copy()
    m = mg + k3*e/6+k4*f/24
    r2 = r2g+k3*d/3+k4*e/12
    r3 = r3g+k3*g+k4*d/4
    r4 = r4g+4*k3*mg+k4*g
    var = r2-m*m
    post_k3 = r3-3*m*r2+2*m**3
    post_k4 = r4-4*m*r3+6*m*m*r2-3*m**4-3*var*var
    Co = C.copy(); np.fill_diagonal(Co,0)
    Ko = K21.copy(); np.fill_diagonal(Ko,0)
    k22 = k4/3 if k22_override is None else k22_override
    mix2 = Co*Co+k22/2
    cov = np.outer(g,g)*Co + (np.outer(d,g)*Ko+np.outer(g,d)*Ko.T)/2 + np.outer(d,d)*mix2/2
    pk21 = 2*np.outer(mg,g)*Co+np.outer(mg,d)*Ko.T+np.outer(g,g)*Ko+np.outer(g,d)*mix2
    pk22 = 4*np.outer(mg,mg)*Co+2*(np.outer(mg,g)*Ko.T+np.outer(g,mg)*Ko)+2*np.outer(g,g)*mix2
    post21 = pk21-2*m[:,None]*cov
    post22 = pk22-2*m[:,None]*pk21.T-2*m[None,:]*pk21+4*np.outer(m,m)*cov-2*cov*cov
    np.fill_diagonal(post22,post_k4)
    radial = 3*np.sum(post22)/(n*(n+2))
    np.fill_diagonal(cov,var)
    np.fill_diagonal(post21,post_k3)
    if metric=="covariance":
        radial = 3*np.sum(post22)/(np.trace(cov)**2+2*np.sum(cov*cov))
    aux = dict(sigma=s,alpha=a,pdf=pdf,gate=g,gaussian_mean=mg,pre_v=v,pre_cov=C,pre_k3=k3,
               pre_k4=np.broadcast_to(k4,(n,)).copy(),post_v=var,post_k3=post_k3,post_k4=post_k4)
    return m,cov,post21,radial,aux

def predict(W,hook=None,cov32=False,cap=0,metric="identity",collect_states=True):
    depth,n,_ = W.shape
    m = np.zeros(n); C = np.eye(n); radial = 0.0
    p = np.zeros((n,0),np.float32); q = p.copy(); lam = np.zeros(0,np.float32)
    star = np.zeros((n,n),np.float32)
    means,states = [],[]
    for l in range(depth):
        w32 = W[l].astype(np.float32)
        w = W[l].astype(np.float64)
        mu = w.T@m
        Cpre = (w32.T@C.astype(np.float32)@w32).astype(np.float64) if cov32 else w.T@C@w
        pp = w32.T@p; pq = w32.T@q
        cp_pre = cp_plane(pp,pq,lam)
        K21 = cp_pre+star_push(W[l],star)
        if metric=="covariance":
            vv = np.diag(Cpre)
            k4 = radial*vv*vv
            k22 = radial*(np.outer(vv,vv)+2*Cpre*Cpre)/3
        else:
            k4,k22 = 4*radial,None
        m,C,target,radial_new,aux = nonlinear(mu,Cpre,K21,k4,k22,metric)
        if collect_states:
            states.append(dict(mean=m.copy(),cov=C.copy(),k21=target.copy(),radial=radial_new,**aux))
        if hook is not None:
            dm = np.asarray(hook(l,m,C,K21,aux,W[l]),dtype=np.float64)
            C = C - np.outer(m,dm)-np.outer(dm,m)-np.outer(dm,dm)
            m = m+dm
        means.append(m.copy())
        if l==depth-1:
            break
        g = aux["gate"]; birth = aux["pdf"]/aux["sigma"]
        inherited = g[:,None]**2*g[None,:]*cp_pre
        bp = g[:,None]*Cpre
        born_plane = bp*bp*birth[None,:]+2*(g*np.diag(Cpre)*birth)[:,None]*bp.T
        cp_post = inherited+born_plane
        p = np.concatenate([g.astype(np.float32)[:,None]*pp,bp.astype(np.float32)],axis=1)
        q = np.concatenate([g.astype(np.float32)[:,None]*pq,np.eye(n,dtype=np.float32)],axis=1)
        lam = np.concatenate([lam,birth.astype(np.float32)])
        if cap and len(lam)>cap:
            ppn = np.sum(p*p,axis=0); qqn = np.sum(q*q,axis=0); pqn = np.sum(p*q,axis=0)
            energy = lam*lam*3*(ppn*ppn*qqn+2*ppn*pqn*pqn)
            keep = np.argsort(-energy,kind="stable")[:cap]
            p,q,lam = p[:,keep],q[:,keep],lam[keep]
            cp_post = cp_plane(p,q,lam)
        residual = target-cp_post
        star = (3*residual.T-2*np.diag(np.diag(residual))).astype(np.float32)
        radial = radial_new
    return np.stack(means),states

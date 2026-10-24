"""Higher Gaussian/Edgeworth correlation orders for the radial K3 host.

All pair kernels are analytic derivatives of Gaussian ReLU moments.
Terms are dimensionless: rho^r/r! times scaled moment derivatives.
The reference (gauss_order=2, edge_order=0, local=False, k31=False)
is the existing SIMPLE map exactly. No training labels enter this map.
"""
import inspect
import math
import numpy as np
import k3_radial_numpy as base


def derivatives(a,pdf,g,mg_over_s,order):
    # A[r] = sigma^(r-1) d^r E[relu]/dmu^r.
    A=[mg_over_s,g]
    h0=np.ones_like(a);h1=a
    for r in range(2,order+1):
        if r==2: h=h0
        elif r==3: h=h1
        else:
            h=a*h1-(r-3)*h0;h0,h1=h1,h
        A.append((-1)**(r-2)*h*pdf)
    B=[(a*a+1)*g+a*pdf,2*mg_over_s,2*g]
    for r in range(3,order+1): B.append(2*A[r-1])
    return A,B


def make_nonlinear(gauss_order=6,edge_order=0,local=False,k31=False):
    def nonlinear(mu,C,K21,k4,k22_override=None,metric="identity"):
        n=len(mu);v=np.diag(C).copy()
        s,a,pdf,g,mg,r2g,r3g,r4g=base.gaussian_raw(mu,v)
        d=pdf/s;e=-a*pdf/v;f=(a*a-1)*pdf/(v*s)
        k3=np.diag(K21).copy()
        m=mg+k3*e/6+k4*f/24
        r2=r2g+k3*d/3+k4*e/12
        r3=r3g+k3*g+k4*d/4
        r4=r4g+4*k3*mg+k4*g
        var=r2-m*m
        post_k3=r3-3*m*r2+2*m**3
        post_k4=r4-4*m*r3+6*m*m*r2-3*m**4-3*var*var
        rho=C/np.outer(s,s);np.fill_diagonal(rho,0)
        T21=K21/(v[:,None]*s[None,:]);np.fill_diagonal(T21,0)
        K22=k4/3 if k22_override is None else k22_override
        T22=K22/np.outer(v,v)
        t3=k3/s**3;t4=np.broadcast_to(k4,(n,))/v**2
        # The covariance-pairing model gives kappa31/(s_i^3 s_j)
        # = alpha*rho. Its standardized diagonal equals alpha.
        T31=t4[:,None]*rho if k31 else None
        maxorder=max(gauss_order,edge_order+4)
        A,B=derivatives(a,pdf,g,mg/s,maxorder)
        def pair(F,G):
            ans=np.zeros_like(C)
            rr=np.ones_like(C)
            for r in range(max(gauss_order,edge_order)+1):
                if r and r<=gauss_order:
                    ans+=rr*np.outer(F[r],G[r])
                if r<=edge_order:
                    ec=(T21*np.outer(F[r+2],G[r+1])+T21.T*np.outer(F[r+1],G[r+2]))/2
                    ec+=T22*np.outer(F[r+2],G[r+2])/4
                    if k31:
                        ec+=(T31*np.outer(F[r+3],G[r+1])+T31.T*np.outer(F[r+1],G[r+3]))/6
                    # Pure local cumulants cancel the product of marginal
                    # expectations at r=0; only correlated terms remain.
                    if local and r:
                        ec+=(np.outer(t3*F[r+3],G[r])+np.outer(F[r],t3*G[r+3]))/6
                        ec+=(np.outer(t4*F[r+4],G[r])+np.outer(F[r],t4*G[r+4]))/24
                    ans+=rr*ec
                rr=rr*rho/(r+1)
            return ans
        cov=pair(A,A)*np.outer(s,s)
        pk21=pair(B,A)*np.outer(v,s)
        pk22=pair(B,B)*np.outer(v,v)
        post21=pk21-2*m[:,None]*cov
        post22=pk22-2*m[:,None]*pk21.T-2*m[None,:]*pk21+4*np.outer(m,m)*cov-2*cov*cov
        np.fill_diagonal(post22,post_k4)
        radial=3*np.sum(post22)/(n*(n+2))
        np.fill_diagonal(cov,var);np.fill_diagonal(post21,post_k3)
        if metric=="covariance": radial=3*np.sum(post22)/(np.trace(cov)**2+2*np.sum(cov*cov))
        aux=dict(sigma=s,alpha=a,pdf=pdf,gate=g,gaussian_mean=mg,pre_v=v,pre_cov=C,pre_k3=k3,
                 pre_k4=np.broadcast_to(k4,(n,)).copy(),post_v=var,post_k3=post_k3,post_k4=post_k4)
        return m,cov,post21,radial,aux
    return nonlinear


def make_predict(**options):
    env=dict(vars(base));env["nonlinear"]=make_nonlinear(**options)
    exec(inspect.getsource(base.predict),env)
    return env["predict"]

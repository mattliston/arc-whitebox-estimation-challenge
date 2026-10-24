"""Offline exact queries for the all-distinct K3 times covariance edge."""
import numpy as np


class ThirdCore:
    def __init__(self,p,lam,U,V,b,B,K):
        self.p,self.lam,self.U,self.V,self.b,self.B,self.K=p,lam,U,V,b,B,K

    def matrix(self,h):
        p,lam,U,V,b,B,K=self.p,self.lam,self.U,self.V,self.b,self.B,self.K
        M=(p*((p.T@h)*lam))@p.T
        cross=(U*(V.T@h))@V.T
        M+=(cross+cross.T+(V*(U.T@h))@V.T)/3
        bh=B@h
        M+=(b@h)*B+np.outer(b,bh)+np.outer(bh,b)
        M-=h[:,None]*K+K.T*h[None,:]
        np.fill_diagonal(M,0)
        return M

    def quadratic(self,H):
        p,lam,U,V,b,B,K=self.p,self.lam,self.U,self.V,self.b,self.B,self.K
        hp=H@p;hv=H@V
        q=p@(lam*np.sum(p*hp,axis=0))
        q+=(U@np.sum(V*hv,axis=0)+2*V@np.sum(U*hv,axis=0))/3
        q+=b*np.sum(H*B)+2*B@(H@b)
        dh=np.diag(H)
        q-=K.T@dh+2*np.sum(H*K,axis=1)-2*dh*np.diag(K)
        return q

    def quadratic_factor(self,X):
        p,lam,U,V,b,B,K=self.p,self.lam,self.U,self.V,self.b,self.B,self.K
        px=p.T@X;vx=V.T@X;ux=U.T@X;bx=B@X
        q=p@(lam*np.sum(px*px,axis=1))
        q+=(U@np.sum(vx*vx,axis=1)+2*V@np.sum(ux*vx,axis=1))/3
        q+=b*np.sum(X*bx)+2*bx@(b@X)
        dh=np.sum(X*X,axis=1)
        q-=K.T@dh+2*np.sum(X*(K@X),axis=1)-2*dh*np.diag(K)
        return q


def row(core,C,g,d,W,index,modes=None):
    v=W[:,index];t=d*v;h=g*v
    M=core.matrix(h)
    H=C*np.outer(t,t)
    q=core.quadratic(H)
    answer=W.T@(g*q+2*d*((C*M)@t))
    approx=None
    if modes is not None:
        X=t[:,None]*modes
        qlow=core.quadratic_factor(X)
        # M is all-distinct, so diagonal removal of the covariance edge is
        # algebraically irrelevant. The factor quadratic removes repeated T.
        low=modes@modes.T
        approx=W.T@(g*qlow+2*d*((low*M)@t))
    return answer,approx

"""Leading covariance-times-third fourth source, explicit retained tensor."""
import numpy as np


class Third:
    def __init__(self,p,lam,U,V,b,B):
        self.p,self.lam,self.U,self.V,self.b,self.B=p,lam,U,V,b,B
        self.repair=None

    def plane(self):
        p,l,U,V,b,B=self.p,self.lam,self.U,self.V,self.b,self.B
        value=(p*p*l)@p.T
        value+=(2*(U*V)@V.T+(V*V)@U.T)/3
        value+=np.diag(B)[:,None]*b[None,:]+2*b[:,None]*B
        if self.repair is not None:
            A=self.repair
            value+=A.T/3
            value[np.diag_indices_from(value)]+=2*np.diag(A)/3
        return value

    def restore(self,target):
        assert self.repair is None
        residual=target-self.plane()
        self.repair=3*residual.T-2*np.diag(np.diag(residual))

    def bilinear(self,X,Y):
        p,l,U,V,b,B=self.p,self.lam,self.U,self.V,self.b,self.B
        px=p.T@X;py=px if Y is X else p.T@Y
        out=p@(l[:,None]*px*py)
        del px,py
        ux=U.T@X;vx=V.T@X
        if Y is X:uy,vy=ux,vx
        else:uy,vy=U.T@Y,V.T@Y
        out+=(U@(vx*vy)+V@(ux*vy+vx*uy))/3
        bx=B@X;by=bx if Y is X else B@Y
        out+=b[:,None]*np.sum(X*by,axis=0)+bx*(b@Y)[None,:]+by*(b@X)[None,:]
        if self.repair is not None:
            A=self.repair
            ax=A.T@X;ay=ax if Y is X else A.T@Y
            out+=(A@(X*Y)+X*ay+Y*ax)/3
        return out


def post_q22(C,K,g,d):
    v=np.diag(C)
    left=2*(d*g)[:,None]*(g*g)[None,:]*(v[:,None]*K.T+2*C*K)
    return left+left.T


def read_q31(C,T,g,d,W):
    q=g[:,None]*W;B=C@q;D=d[:,None]*W*B
    K=T.bilinear(q,q);S=T.bilinear(D,q)
    value=3*(d[:,None]*B*K).T@W+3*(d[:,None]*W*K).T@B+6*S.T@q
    marginal=12*np.sum(D*K,axis=0)
    return value,marginal


def source(C,T,K,g,d,W,Cpost):
    raw31,raw4=read_q31(C,T,g,d,W)
    trace_faces=post_q22(C,K,g,d)
    denominator=np.trace(Cpost)**2+2*np.sum(Cpost*Cpost)
    assert denominator>0 and np.isfinite(denominator)
    beta=3*np.sum(trace_faces)/denominator
    metric=W.T@Cpost@W;v=np.diag(metric)
    return (raw4-beta*v*v,raw31-beta*v[:,None]*metric,
            dict(beta=float(beta),trace=float(np.sum(trace_faces)),denominator=float(denominator)),raw4,raw31)

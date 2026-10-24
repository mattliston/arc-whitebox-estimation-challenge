"""OFFLINE full-edge reference for thin-query truncation diagnostics.

Materializes several full matrices and expensive CP products. Deliberately
not a deployment method or a reopening of the cost-failed dense card.
"""
import numpy as np
from fourth_trace_full_queries_v1 import fused_query


class FullEdgeThird:
    def __init__(self,p,lam,star,wstar,b,B,K,C):
        self.p,self.lam=p,lam
        self.U,self.V=wstar.T@star,wstar.T
        self.b,self.B,self.K=b,B,K
        self.edge=C.copy();np.fill_diagonal(self.edge,0)

    def fourth_third_part(self,h1,h2,g,d,X):
        p,lam,U,V,b,B,K,C=self.p,self.lam,self.U,self.V,self.b,self.B,self.K,self.edge
        ph=p.T@h1;vh=V.T@h1;uh=U.T@h1;bh=B@h1
        F=(p*(lam*ph))@p.T
        cross=(U*vh)@V.T
        F+=(cross+cross.T+(V*uh)@V.T)/3
        F+=(b@h1)*B+np.outer(b,bh)+np.outer(bh,b)
        F-=h1[:,None]*K+K.T*h1[None,:]
        np.fill_diagonal(F,0)
        E=((C.T@(h2[:,None]*p))*p*lam)@p.T
        CV=C.T@(h2[:,None]*V);CU=C.T@(h2[:,None]*U)
        E+=((CU*V)@V.T+(CV*U)@V.T+(CV*V)@U.T)/3
        E+=(C.T@(h2*b))[:,None]*B+b[:,None]*(C.T@(h2[:,None]*B))
        E+=((C*B).T@h2)[:,None]*b[None,:]
        E-=C*K.T*h2[None,:]
        np.fill_diagonal(E,0)
        return (F*(np.outer(g,g)+C*np.outer(d,d))+E*np.outer(d,g)+E.T*np.outer(g,d))@X


def full_edge_query(C,K21,K22,A,B,h,wick,modes,third,repeated,X):
    # Reuse all already-certified terms; restore exactly the one Gaussian
    # triangle edge deliberately truncated by the thin production query.
    out=fused_query(C,K21,K22,A,B,h,wick,modes,third,repeated,X)
    edge=modes@modes.T;np.fill_diagonal(edge,0)
    exact=C.copy();np.fill_diagonal(exact,0)
    gram=C.T@(h[2][:,None]*C)
    spectator=h[2]*np.diag(C)
    repair_delta=(spectator[:,None]*C+C*spectator[None,:])*(edge-C)*np.outer(wick[2],wick[2])
    np.fill_diagonal(repair_delta,0)
    out+=((exact-edge)*gram*np.outer(wick[2],wick[2])+repair_delta)@X
    return out


def quadratic_source_diagonal(A,C,g,d,W):
    """Contract the FULL-C source T with every column of next W.

    T=(d_i g_j g_k Q_iijk+cyclic)/2, Q=Sym(A,C).
    Source repeated entries are included. Full-vs-low difference after
    coordinate restoration subtracts its repeated-plane star below.
    """
    z=g[:,None]*W;t=d[:,None]*W
    Cz=C@z;Az=A@z
    return ((t*np.diag(A)[:,None]).sum(0)*(z*Cz).sum(0)
            +(t*np.diag(C)[:,None]).sum(0)*(z*Az).sum(0)
            +4*(t*Az*Cz).sum(0))/4


def quadratic_source_plane(A,C,g,d):
    v=np.diag(C);ad=np.diag(A)
    q31=(ad[:,None]*C+v[:,None]*A)/2
    q22=(np.outer(ad,v)+np.outer(v,ad)+4*A*C)/6
    return (d*g)[:,None]*g[None,:]*q31+(g*g)[:,None]*d[None,:]*q22/2


def zero_repeated_source_diagonal(A,C,g,d,W):
    plane=quadratic_source_plane(A,C,g,d)
    star=3*plane.T-2*np.diag(np.diag(plane))
    return quadratic_source_diagonal(A,C,g,d,W)-((star.T@W)*(W*W)).sum(0)

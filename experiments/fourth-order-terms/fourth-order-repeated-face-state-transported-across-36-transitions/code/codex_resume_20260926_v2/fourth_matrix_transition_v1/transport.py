"""Exact affine carry and specified leading quadratic fourth-to-third source."""
import numpy as np

def faces(A,B):
    a=np.diag(A);b=np.diag(B)
    return a*b,(a[:,None]*B+b[:,None]*A)/2,(np.outer(a,b)+np.outer(b,a)+4*A*B)/6

def carry(A,B,g,d,W):
    L=g[:,None]*W
    Ag=L.T@A@L;Bg=L.T@B@L
    Ag=(Ag+Ag.T)/2;Bg=(Bg+Bg.T)/2
    # Two vector/matrix terms and the complete six-permutation mixed columns.
    av=W.T@(d*np.diag(A));bv=W.T@(d*np.diag(B))
    X=W.T*d[None,:];U=L.T@A;V=L.T@B
    T=(2*av[:,None]*Bg+np.diag(Bg)[:,None]*av[None,:]
       +2*bv[:,None]*Ag+np.diag(Ag)[:,None]*bv[None,:])/12
    T=T+((X*U)@V.T+(X*V)@U.T+(U*V)@X.T)/3
    return Ag,Bg,T

def query(A,B,g,d,W,i,j):
    """Independent original-coordinate vector contractions for (iiij,iijj,iij)."""
    u=W[:,i];v=W[:,j];x=g*u;y=g*v
    Ax=A@x;Ay=A@y;Bx=B@x;By=B@y
    aa=x@Ax;ab=x@Ay;bb=y@Ay;da=x@Bx;db=x@By;dd=y@By
    q31=(aa*db+da*ab)/2
    q22=(aa*dd+da*bb+4*ab*db)/6
    Qxy=(np.diag(A)*(x@By)+np.diag(B)*(x@Ay)+2*Ax*By+2*Ay*Bx)/6
    Qxx=(np.diag(A)*(x@Bx)+np.diag(B)*(x@Ax)+4*Ax*Bx)/6
    t21=.5*(2*np.dot(d*u,Qxy)+np.dot(d*v,Qxx))
    return q31,q22,t21

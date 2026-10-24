"""Contracted fourth-emission identities; NumPy or CPU Torch batch algebra."""
import numpy as np


def coordinate_slice(A,h,xp=np):
    return (A*h[None,:]+h[:,None]*A.T+xp.diag(A.T@h))/3


def remove_repeated(M,K,U,xp=np):
    A=3*K.T-2*xp.diag(xp.diag(K))
    return xp.stack([m-coordinate_slice(A,u,xp) for m,u in zip(M,U.T)])


def batch_source(C,rawA,rawB,raw31,raw22,U,g,d,count,xp=np):
    """Unbiased per-batch fourth k-statistics; moments are centered means."""
    f=count*count/((count-1)*(count-2)*(count-3))
    ac=f*(count+1);wc=f*(count-1);v=xp.diag(C)
    Q31=ac*raw31-wc*3*v[:,None]*C
    Q22=ac*raw22-wc*(v[:,None]*v[None,:]+2*C*C)
    K=(d*g)[:,None]*g[None,:]*Q31+.5*(g*g)[:,None]*d[None,:]*Q22
    matrices=[]
    for a,u in enumerate(U.T):
        h=g*u;t=d*u;ch=C@h
        A=ac*rawA[a]-wc*(v[:,None]*ch[None,:]+2*C*ch[:,None])
        B=ac*rawB[a]-wc*(C*(v@t)+2*(C*t[None,:])@C)
        matrices.append(.5*(d[:,None]*g[None,:]*A+g[:,None]*d[None,:]*A.T+g[:,None]*g[None,:]*B))
    return xp.stack(matrices),K,Q31,Q22


def radial_source(C,g,d,birth_d,radial,Bpost,U):
    B=g[:,None]*C;np.fill_diagonal(B,0)
    b=radial*d*np.diag(C)/6
    K=(radial/3)*(B*B)*birth_d[None,:]
    K+=np.diag(Bpost)[:,None]*b[None,:]+2*Bpost*b[:,None]
    out=[]
    for u in U.T:
        z=birth_d*(B.T@u)
        M=(radial/3)*((B*(birth_d*u)[None,:])@B.T+z[:,None]*B.T+B*z[None,:])
        cu=Bpost@u
        M+=(b@u)*Bpost+np.outer(b,cu)+np.outer(cu,b)
        out.append(M)
    return remove_repeated(np.stack(out),K,U)

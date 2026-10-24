"""Offline exact algebra for a three-leg covariance-metric completion."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def operator(R,H):
    return ((R*R)@H+2*R*(R@H.T))@R


def factors(R,H,sigma):
    return sigma[:,None]*R,sigma[:,None]*(R@H.T)


def plane(U,V):
    return (U*U)@V.T+2*(U*V)@U.T


def push_plane(U,V,W,indices):
    A=W.T@U;B=W.T@V
    return (A[indices]*A[indices])@B.T+2*(A[indices]*B[indices])@A.T


def star_from_plane(K):
    return 3*K.T-2*np.diag(np.diag(K))


def star_query(S,W,indices):
    A=W.T@S;B=W.T
    return ((2*A[indices]*B[indices])@B.T+(B[indices]*B[indices])@A.T)/3


def complete(R,K,steps=30,tolerance=1e-10,checkpoints=(2,)):
    """PCG with symmetric Kronecker preconditioning; no regularization."""
    fleft=cho_factor(R*R,lower=True,check_finite=False)
    fright=cho_factor(R,lower=True,check_finite=False)
    def precondition(Y):
        Z=cho_solve(fleft,Y,check_finite=False)
        return cho_solve(fright,Z.T,check_finite=False).T
    H=np.zeros_like(K);residual=K.copy()
    norm=max(float(np.linalg.norm(K)),1e-300)
    Z=precondition(residual);direction=Z.copy()
    rz=float(np.sum(residual*Z));records=[];saved={}
    assert rz>0
    for iteration in range(1,steps+1):
        action=operator(R,direction)
        curvature=float(np.sum(direction*action))
        assert curvature>0 and np.isfinite(curvature),curvature
        alpha=rz/curvature
        H+=alpha*direction
        residual-=alpha*action
        relative=float(np.linalg.norm(residual)/norm)
        records.append(dict(iteration=iteration,residual_relative=relative))
        if iteration in checkpoints:saved[iteration]=H.copy()
        if relative<tolerance:break
        Z=precondition(residual)
        rz_new=float(np.sum(residual*Z))
        assert rz_new>0 and np.isfinite(rz_new),rz_new
        direction=Z+(rz_new/rz)*direction
        rz=rz_new
    return H,saved,records

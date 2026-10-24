"""Static one-high CP+star sector and exact coordinate-plane restoration."""
import numpy as np


def star_plane(star,W):
    wt=W.T;wa=wt@star
    return (2*(wa*wt)@wt.T+(wt*wt)@wa.T)/3


def star_diagonal(star,W):
    wt=W.T
    return ((wt@star)*wt*wt).sum(1)


def restore_plane_diagonal(plane,W):
    return star_diagonal(3*plane.T-2*np.diag(np.diag(plane)),W)


def compare(p,lam,star,U,W,b=None,B=None):
    Q=np.eye(len(U))-U@U.T
    lowp=Q@p
    pw=W.T@p;lw=W.T@lowp
    full=(pw*pw*pw*lam).sum(1)+star_diagonal(star,W)
    low=(lw*lw*lw*lam).sum(1)+star_diagonal(star,Q@W)
    low_plane=(lowp*lowp)@(lowp*lam).T+star_plane(star,Q)
    if b is not None:
        full+=3*(W.T@b)*(W*(B@W)).sum(0)
    restored=full-low+restore_plane_diagonal(low_plane,W)
    return dict(full=full,low=low,low_plane=low_plane,restored=restored)


def matrix_coefficients(T,U):
    """Dense small-test-only constructor, q full matrices, no large use."""
    R=np.einsum('ijk,ia->ajk',T,U)
    D=np.einsum('ajk,jb->abk',R,U)
    E=np.einsum('abk,kc->abc',D,U)
    return np.stack([3*R[a]-1.5*(U@D[a]+D[a].T@U.T)+U@E[a]@U.T for a in range(U.shape[1])])

"""Complete Gaussian bond-order3 distinct source, exact full-edge slices."""
import numpy as np
from linear_transport_residual_source_v1 import coord_slice


def slices(C,g,d,e,U):
    C=C.copy();np.fill_diagonal(C,0)
    A=d[:,None]*(C*C);B=g[:,None]*C
    K=(d*g)[:,None]*(C*C*C)*e[None,:]
    star=3*K.T-2*np.diag(np.diag(K))
    matrices=[]
    for h in U.T:
        triangle=(d[:,None]*d[None,:])*C*((C*(d*h)[None,:])@C)
        X=(A*(e*h)[None,:])@B.T
        Y=(e*(B.T@h))[:,None]*A.T+(e*(A.T@h))[:,None]*B.T
        matrices.append(triangle+.5*(X+X.T+Y+Y.T)-coord_slice(star,h))
    return np.stack(matrices)


def dense(C,g,d,e):
    C=C.copy();np.fill_diagonal(C,0)
    T=np.einsum('i,j,k,ij,jk,ki->ijk',d,d,d,C,C,C)
    from itertools import permutations
    base=.5*np.einsum('i,j,k,ij,ik->ijk',e,d,g,C*C,C)
    T+=sum(base.transpose(p) for p in permutations(range(3)))
    ii,jj,kk=np.indices(T.shape);T*=((ii!=jj)&(ii!=kk)&(jj!=kk))
    return T

"""Exact non-Gaussian two-edge source; no completed-square remainder."""
import numpy as np
from linear_transport_residual_source_v1 import coord_slice


def factors(C,g,d,eta,K21,K22):
    B=g[:,None]*C
    D=.5*g[:,None]*K21.T*eta[None,:]+.5*d[:,None]*K21+.25*d[:,None]*K22*eta[None,:]
    np.fill_diagonal(B,0);np.fill_diagonal(D,0)
    return B,D


def slices(C,g,d,eta,K21,K22,H):
    B,D=factors(C,g,d,eta,K21,K22)
    K=2*B*D*d[None,:];star=3*K.T-2*np.diag(np.diag(K));out=[]
    for h in H.T:
        X=(B*(d*h)[None,:])@D.T
        Y=(d*(D.T@h))[:,None]*B.T+(d*(B.T@h))[:,None]*D.T
        out.append(X+X.T+Y+Y.T-coord_slice(star,h))
    return np.stack(out)


def dense(C,g,d,eta,K21,K22):
    B,D=factors(C,g,d,eta,K21,K22);I=np.eye(len(g))
    T=np.einsum('ia,ja,ka,a->ijk',B,D,I,d)
    T=T+T.transpose(1,0,2);T=T+T.transpose(0,2,1)+T.transpose(2,1,0)
    ii,jj,kk=np.indices(T.shape);T*=((ii!=jj)&(ii!=kk)&(jj!=kk))
    return T

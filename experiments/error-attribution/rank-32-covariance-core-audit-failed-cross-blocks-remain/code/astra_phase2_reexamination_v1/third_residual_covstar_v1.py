"""Offline residual coordinate/covariance completion, with no clipping or fit."""
import itertools
import numpy as np


def factors(C,K):
    C=(C+C.T)/2
    sigma=np.sqrt(np.diag(C))
    assert np.isfinite(sigma).all() and sigma.min()>0
    R=C/np.outer(sigma,sigma)
    np.fill_diagonal(R,1)
    rho=R.copy();np.fill_diagonal(rho,0)
    assert np.max(np.abs(rho))<.98
    S=K/(sigma[:,None]**2*sigma[None,:])
    assert np.max(np.abs(np.diag(S)))==0
    A=(S.T-rho*S)/(2*(1-rho*rho))
    np.fill_diagonal(A,0)
    P=sigma[:,None]*R
    E=np.diag(sigma)
    B=sigma[:,None]*A
    return P,E,B


def plane(P,E,B):
    return 2*((P*E)@B.T+(P*B)@E.T+(E*B)@P.T)


def carry(C,K,gate,W):
    P,E,B=factors(C,K)
    J=gate[:,None]*W
    return plane(J.T@P,J.T@E,J.T@B)


def small_tensor(P,E,B):
    unsymmetric=np.einsum('ia,ja,ka->ijk',P,E,B)
    return sum(unsymmetric.transpose(p) for p in itertools.permutations(range(3)))

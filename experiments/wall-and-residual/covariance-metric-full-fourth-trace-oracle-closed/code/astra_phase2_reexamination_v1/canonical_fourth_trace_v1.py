"""Canonical Euclidean trace-generated harmonic component, offline only."""
import numpy as np


def canonical_trace_core(R):
    R = (R+R.T)/2
    n = len(R)
    return 6*R/(n+4)-3*np.trace(R)*np.eye(n)/((n+2)*(n+4))


def small_tensor(A):
    I = np.eye(len(A))
    return (np.einsum('ij,kl->ijkl', A, I)+np.einsum('ik,jl->ijkl', A, I)+
            np.einsum('il,jk->ijkl', A, I)+np.einsum('ij,kl->ijkl', I, A)+
            np.einsum('ik,jl->ijkl', I, A)+np.einsum('il,jk->ijkl', I, A))/6

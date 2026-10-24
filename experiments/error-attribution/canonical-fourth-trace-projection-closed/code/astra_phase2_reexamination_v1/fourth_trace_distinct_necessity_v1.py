"""Offline full-rank core inversion and a precisely zeroed diagram class."""
import numpy as np


class ZeroDistinctThird:
    def fourth_third_part(self, h1, h2, g, d, X):
        return np.zeros_like(X)


def core_trace(A, C):
    return (np.trace(C)*A+np.trace(A)*C+2*(A@C+C@A))/6


def inverse_core(C, R):
    C = (C+C.T)/2
    R = (R+R.T)/2
    values, vectors = np.linalg.eigh(C)
    assert values.min() > -1e-9, values.min()
    t = np.trace(C)
    rotated = vectors.T@R@vectors
    diagonal_den = t+4*values
    trace_A = np.sum(6*np.diag(rotated)/diagonal_den)/(1+np.sum(values/diagonal_den))
    diagonal = (6*np.diag(rotated)-trace_A*values)/diagonal_den
    ar = 6*rotated/(t+2*(values[:, None]+values[None, :]))
    np.fill_diagonal(ar, diagonal)
    A = vectors@ar@vectors.T
    return (A+A.T)/2


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))

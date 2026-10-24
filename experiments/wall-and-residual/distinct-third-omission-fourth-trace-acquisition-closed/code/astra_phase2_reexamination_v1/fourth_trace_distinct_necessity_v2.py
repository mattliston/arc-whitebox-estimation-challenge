"""Exact Sylvester/rank-one inverse on its actual algebraic domain."""
import numpy as np
from fourth_trace_distinct_necessity_v1 import ZeroDistinctThird, core_trace, relative


def inverse_core(C, R):
    C = (C+C.T)/2
    R = (R+R.T)/2
    values, vectors = np.linalg.eigh(C)
    t = np.trace(C)
    diagonal_den = t+4*values
    assert np.min(diagonal_den) > 0, (t, values.min())
    consistency = 1+np.sum(values/diagonal_den)
    assert np.isfinite(consistency) and abs(consistency) > 1e-12, consistency
    rotated = vectors.T@R@vectors
    trace_A = np.sum(6*np.diag(rotated)/diagonal_den)/consistency
    diagonal = (6*np.diag(rotated)-trace_A*values)/diagonal_den
    ar = 6*rotated/(t+2*(values[:, None]+values[None, :] ))
    np.fill_diagonal(ar, diagonal)
    A = vectors@ar@vectors.T
    return (A+A.T)/2

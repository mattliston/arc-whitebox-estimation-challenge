"""Trace-augmented risk projection retaining both covariance cross blocks."""
import numpy as np


def project_tangent(Z, Q):
    assert np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1]))) <= 1e-10
    U = Z @ Q
    core = Q.T @ U
    core = (core + core.T) / 2
    shift = (np.trace(Z) - np.trace(core)) / (Q.shape[1] + 2)
    V = U - .5 * Q @ core + .5 * shift * Q
    prediction = Q @ V.T + V @ Q.T
    return prediction, V, float(shift)

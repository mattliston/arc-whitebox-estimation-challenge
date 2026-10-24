"""Exact trace-augmented Gaussian risk projection into a fixed subspace."""
import numpy as np


def tilt(matrix, u, coefficient):
    """(I+coefficient*u*u.T) matrix (I+coefficient*u*u.T)."""
    matrix = (matrix + matrix.T) / 2
    v = matrix @ u
    return matrix + coefficient * (np.outer(u, v) + np.outer(v, u)) + coefficient * coefficient * float(u @ v) * np.outer(u, u)


def project(Z, Q):
    assert np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1]))) <= 1e-10
    core = Q.T @ Z @ Q
    core = (core + core.T) / 2
    shift = (np.trace(Z) - np.trace(core)) / (Q.shape[1] + 2)
    core = core + shift * np.eye(Q.shape[1])
    return (Q @ core) @ Q.T, core, float(shift)

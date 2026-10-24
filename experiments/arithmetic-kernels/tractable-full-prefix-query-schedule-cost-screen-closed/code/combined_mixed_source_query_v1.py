"""Exact full repeated query of combined heavy and Gaussian path sources."""
import numpy as np
from exact_heavy_source_slices_v1 import factors
from residual_one_high_oracle_v1 import repeated_push


def combined_factors(C, g, d, e, eta, K21, K22):
    B, D = factors(C, g, d, eta, K21, K22)
    off = C.copy(); np.fill_diagonal(off, 0)
    F = D * d[None, :] + .5 * (d[:, None] * off * off) * e[None, :]
    np.fill_diagonal(F, 0)
    return B, F


def query(B, F, V):
    p, q, r = V.T @ B, V.T @ F, V.T
    raw = 2 * ((p * q) @ r.T + (p * r) @ q.T + (q * r) @ p.T)
    local = 2 * B * F
    return raw - repeated_push(local, V, np.arange(V.shape[1]))

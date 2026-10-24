"""Offline observer only: no values return to the parent estimator."""
import numpy as np


def source(C, sigma, H):
    C, sigma, H = (np.asarray(x, dtype=np.float64) for x in (C, sigma, H))
    n = len(sigma)
    assert C.shape == (n, n) and H.shape[0] == n
    assert np.isfinite(C).all() and np.isfinite(H).all() and np.all(sigma > 0)
    R = C / (sigma[:, None] * sigma[None, :])
    np.fill_diagonal(R, 0.)
    centered = H - H.mean(axis=0, keepdims=True)
    U = (R @ centered) / np.sqrt(n)
    V = (R * R) @ centered
    return U, V


def terminal(U, V, sigma14, W, sigma15, gate15, pdf15):
    U, V, sigma14, W, sigma15, gate15, pdf15 = (
        np.asarray(x, dtype=np.float64)
        for x in (U, V, sigma14, W, sigma15, gate15, pdf15))
    assert np.all(sigma14 > 0) and np.all(sigma15 > 0)
    mean = (W.T @ (sigma14[:, None] * U)) * (gate15 / sigma15)[:, None]
    variance = ((W * W).T @ (sigma14[:, None] ** 2 * V)) * (pdf15 / (2 * sigma15 ** 2))[:, None]
    result = np.concatenate((mean, variance), axis=1)
    assert np.isfinite(result).all()
    return result

"""Exact repeated faces of the leading connected Gaussian fourth tree."""
import numpy as np


def q31(C, g, d, e, W):
    B = C @ (g[:, None] * W)
    T = d[:, None] * W * B
    Y = C @ T
    first = 6 * d[:, None] * B * Y + e[:, None] * B**3
    second = 6 * d[:, None] * W * Y + 3 * e[:, None] * W * B**2
    return first.T @ W + second.T @ B


def diagonal(C, g, d, e, W):
    B = C @ (g[:, None] * W)
    T = d[:, None] * W * B
    return 12 * np.sum(T * (C @ T), axis=0) + 4 * np.sum(e[:, None] * W * B**3, axis=0)


def post_q22(C, g, d, e):
    v = np.diag(C)
    a = d * g * v
    value = 4 * np.outer(a, a) * C
    value += 2 * C**2 * (np.outer(d*d*v, g*g) + np.outer(g*g, d*d*v)
                        + 2 * np.outer(d*g, d*g) * C)
    value += 2 * C**2 * (np.outer(e*g*v, g*g) + np.outer(g*g, e*g*v))
    return value


def source(C, g, d, e, W, Cpost):
    tree31 = q31(C, g, d, e, W)
    tree4 = np.diag(tree31).copy()
    trace_faces = post_q22(C, g, d, e)
    denominator = np.trace(Cpost)**2 + 2 * np.sum(Cpost*Cpost)
    assert denominator > 0 and np.isfinite(denominator)
    beta = 3 * np.sum(trace_faces) / denominator
    metric = W.T @ Cpost @ W
    v = np.diag(metric)
    delta31 = tree31 - beta * v[:, None] * metric
    delta4 = tree4 - beta * v*v
    return delta4, delta31, dict(beta=float(beta), trace=float(np.sum(trace_faces)),
                                denominator=float(denominator)), tree4, tree31

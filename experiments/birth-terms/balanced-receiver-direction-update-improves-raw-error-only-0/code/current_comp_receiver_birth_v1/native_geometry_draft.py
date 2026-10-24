"""Unexecuted FNP spelling of the frozen RB operator; not a candidate.

No model, parameter loader, hook, fitted literal or score. A later native
port requires separate admission and measured complete accounting.
"""
import flopscope.numpy as fnp


def _rb_diagonal(x, value):
    # Preserve the qualified shadow's subtract/add grouping. Do not replace
    # it with an in-place diagonal assignment without numerical admission.
    return x - fnp.diag(fnp.diag(x)) + fnp.diag(value)


def _rb_geometry(B0, H, Theta):
    centered = H - fnp.mean(H, axis=0)
    scale = fnp.sqrt(fnp.mean(centered * centered, axis=0) + 1e-6)
    Z = fnp.clip(centered / scale, -8, 8)
    R = fnp.tanh(((Z @ Theta) @ Z.T) / 64)
    B = B0.astype(fnp.float64)
    V = B * R.astype(fnp.float64)
    norm2 = fnp.sum(B * B, axis=0)
    denominator = fnp.where(norm2 == 0, fnp.ones_like(norm2), norm2)
    F = V - B * (fnp.sum(B * V, axis=0) / denominator)[None, :]
    F = _rb_diagonal(F, fnp.zeros_like(norm2))
    rounded = (B + .1 * F).astype(fnp.float32)
    new = fnp.where(B0 == 0, B0, rounded)
    D = new.astype(fnp.float64) - B
    return new, F, D


def _rb_extra_plane(B0, D, t, s, a, alive, births):
    B = B0.astype(fnp.float64)
    aa = a.astype(fnp.float32).astype(fnp.float64)
    tt = (t * alive).astype(fnp.float32).astype(fnp.float64)
    ss = (s * alive).astype(fnp.float32).astype(fnp.float64)
    w = aa * (tt - ss) * births
    z = aa * (tt * tt - ss * ss) * births
    extra = (2 * B * D + D * D) * w[None, :] + z[:, None] * D.T
    return _rb_diagonal(extra, fnp.zeros_like(t))

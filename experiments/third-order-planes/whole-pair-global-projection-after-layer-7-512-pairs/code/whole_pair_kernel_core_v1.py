"""Inert FNP-only complete-pair projection primitives; not a candidate.

Original physical cubes remain unchanged. The fixed policy uses pair energy,
normalized pair coordinates, an FP64 selected Gram and ridge1e-3. No model,
labels, fitted head, state acquisition, file I/O or execution occurs on import.
"""
import flopscope.numpy as fnp


def _gpk_features(positive, negative):
    """Derive in FP64 from the ACTUAL stored FP32 physical cubes."""
    p = positive.astype(fnp.float64)
    q = negative.astype(fnp.float64)
    h = (p+q)/2
    d = p-q
    h2, d2 = h*h, d*d
    u = h2+d2/4
    v = 2*h*d
    t = 3*h2*d+d2*d/4
    return h, d, u, v, t


def _gpk_norms(features):
    h, d, u, v, t = features
    hh = fnp.sum(h*h, axis=0)
    dd = fnp.sum(d*d, axis=0)
    hd = fnp.sum(h*d, axis=0)
    full = 3*hh*hh*dd+6*hh*hd*hd+1.5*hd*hd*dd+dd*dd*dd/16
    plane = (fnp.sum(u*u, axis=0)*dd
             + 2*fnp.sum(u*v, axis=0)*hd
             + fnp.sum(v*v, axis=0)*hh)
    diagonal = fnp.sum(t*t, axis=0)
    distinct = full-3*plane+2*diagonal
    # The exact quantity is nonnegative. Large negative cancellation is a
    # qualification failure, never a data-dependent alternative solver.
    if bool(fnp.any(distinct < -1e-10*fnp.maximum(full, 1e-300))):
        raise ValueError("Negative all-distinct pair norm beyond rounding allowance")
    return fnp.maximum(distinct, 0), full


def _gpk_kernel(left, right, dtype):
    """Nine literal ordinary products; FP64 polynomial in either mode."""
    h, d, u, v, t = (a.astype(dtype) for a in left)
    hp, dp, up, vp, tp = (a.astype(dtype) for a in right)
    x = (h.T @ hp).astype(fnp.float64)
    y = (h.T @ dp).astype(fnp.float64)
    z = (d.T @ hp).astype(fnp.float64)
    w = (d.T @ dp).astype(fnp.float64)
    uu = (u.T @ up).astype(fnp.float64)
    uv = (u.T @ vp).astype(fnp.float64)
    vu = (v.T @ up).astype(fnp.float64)
    vv = (v.T @ vp).astype(fnp.float64)
    tt = (t.T @ tp).astype(fnp.float64)
    full = 3*x*x*w+6*x*y*z+0.75*(y*y+z*z)*w+w*w*w/16
    plane = uu*w+uv*z+vu*y+vv*x
    return full-3*plane+2*tt


def _gpk_solve(positive, negative, weights, keep):
    """Fixed energy-selected pair dictionary, no coefficient/head training.

The on-input linear solve is the deterministic approximation algorithm.
It does not use final or intermediate network error targets.
"""
    features = _gpk_features(positive, negative)
    norm2, _ = _gpk_norms(features)
    weights64 = weights.astype(fnp.float64)
    energy = weights64*weights64*norm2
    selected = fnp.argsort(-energy, kind='stable')[:keep]
    if int(selected.shape[0]) != keep or bool(fnp.any(norm2[selected] <= 0)):
        raise ValueError("Insufficient nonzero pair dictionary atoms")
    scale = fnp.sqrt(norm2[selected])
    landmarks = tuple(a[:, selected] for a in features)
    cross = _gpk_kernel(landmarks, features, fnp.float32)
    rhs = (cross @ weights64)/scale
    del cross
    gram = _gpk_kernel(landmarks, landmarks, fnp.float64)
    gram = gram/(scale[:, None]*scale[None, :])
    gram = (gram+gram.T)/2
    gram = gram+1e-3*fnp.eye(keep, dtype=fnp.float64)
    coefficients = fnp.linalg.solve(gram, rhs)
    stored_weights = (coefficients/scale).astype(fnp.float32)
    if not bool(fnp.all(fnp.isfinite(stored_weights))):
        raise ValueError("Nonfinite complete-pair projection coefficients")
    return selected, stored_weights


def _gpk_plane(p, weights, product):
    """Actual stored-factor plane, with independent FP64 cubic diagonal."""
    value = product(p*p, (p*weights).T).astype(fnp.float64)
    p64 = p.astype(fnp.float64)
    diagonal = fnp.sum(p64*p64*p64*weights.astype(fnp.float64), axis=1)
    fnp.fill_diagonal(value, diagonal)
    return value


def _gpk_compress(p, weights, star, newborn, counts, active, product, keep=512):
    """One post7 operation: old cohorts0..6 only, newest sharing preserved.

Caller must invoke only after the ORIGINAL post7 star and newborn are built.
The all16 response readers/heads remain caller-owned and unchanged.
"""
    if len(counts) != 8:
        raise ValueError("Complete-pair compression is fixed at post7")
    cursor = 0
    plus, minus = [], []
    for count in counts[:-1]:
        plus.append(fnp.arange(cursor, cursor+count, dtype=fnp.int64))
        minus.append(fnp.arange(cursor+count, cursor+2*count, dtype=fnp.int64))
        cursor += 2*count
    if cursor != newborn[0]:
        raise ValueError("Old-cohort/newborn prefix ownership mismatch")
    plus = fnp.concatenate(plus)
    minus = fnp.concatenate(minus)
    if not bool(fnp.all(weights[plus] == -weights[minus])):
        raise ValueError("Original signed-pair weights no longer match")
    old = p[active, :cursor]
    pp = p[fnp.ix_(active, plus)]
    qq = p[fnp.ix_(active, minus)]
    selected, paired_weights = _gpk_solve(pp, qq, weights[plus], keep)
    indices = fnp.concatenate((plus[selected], minus[selected]))
    replacement = p[:, indices]
    replacement_weights = fnp.concatenate((paired_weights, -paired_weights))
    before = _gpk_plane(old, weights[:cursor], product)
    after = _gpk_plane(replacement[active], replacement_weights, product)
    rest = before-after
    correction = 3*rest.T
    fnp.fill_diagonal(correction, fnp.diag(rest))
    n = int(p.shape[0])
    full = fnp.zeros((n,n), dtype=fnp.float64)
    fnp.put(full, active[:, None]*n+active[None, :], correction)
    next_star = (star.astype(fnp.float64)+full).astype(fnp.float32)
    next_p = fnp.concatenate((replacement, p[:, cursor:]), axis=1)
    next_weights = fnp.concatenate((replacement_weights, weights[cursor:]))
    next_newborn = (2*keep, newborn[1], newborn[2], newborn[3], newborn[4])
    return next_p, next_weights, next_star, next_newborn

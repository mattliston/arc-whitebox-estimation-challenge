"""Finite piecewise-quadratic scalar oracle; not a deployable predictor."""
import numpy as np


def clipped_scalar_oracle(parent, feature, truth):
    p, z, y = [np.asarray(v, dtype=np.float64) for v in (parent, feature, truth)]
    assert p.ndim == 1 and p.shape == z.shape == y.shape and p.size
    assert all(np.isfinite(v).all() for v in (p, z, y))
    assert (p >= 0).all() and (y >= 0).all()
    moving = z != 0
    thresholds = np.zeros_like(z)
    thresholds[moving] = -p[moving]/z[moving]
    assert np.isfinite(thresholds).all()
    breaks = np.unique(thresholds[moving])
    edges = np.concatenate(([-np.inf], breaks, [np.inf]))
    candidates = [0., *map(float, breaks)]
    # Recompute each quadratic directly, avoiding an accumulated sweep error.
    # At a boundary either convention gives the same continuous prediction.
    for left, right in zip(edges[:-1], edges[1:]):
        active = ((z > 0) & (thresholds <= left)) | ((z < 0) & (thresholds >= right))
        zz = z[active]
        denom = float(np.dot(zz, zz))
        if denom:
            root = -float(np.dot(zz, (p-y)[active]))/denom
            if left <= root <= right and np.isfinite(root):
                candidates.append(root)
    coefficients = np.asarray(candidates)
    errors = np.empty(len(coefficients))
    for start in range(0, len(coefficients), 64):
        stop = min(start+64, len(coefficients))
        predictions = np.maximum(p[None, :]+coefficients[start:stop, None]*z[None, :], 0.)
        errors[start:stop] = np.mean((predictions-y[None, :])**2, axis=1)
    best = int(np.argmin(errors))
    coefficient = float(coefficients[best])
    ideal = np.maximum(p+coefficient*z, 0.)
    served = ideal.astype(np.float32)
    error = float(np.mean((ideal-y)**2))
    unit_roundoff = 2.**-24
    lower = max(0., (1-unit_roundoff)*np.sqrt(error)-unit_roundoff*np.sqrt(np.mean(y*y)))**2
    zz = float(np.dot(z, z))
    no_clip_coefficient = float(np.dot(z, y-p)/zz) if zz else 0.
    return dict(coefficient=coefficient, ideal_raw=error,
                served_raw=float(np.mean((served.astype(np.float64)-y)**2)),
                rounding_allowance_lower_raw=float(lower),
                breakpoints=int(len(breaks)), candidates=int(len(coefficients)),
                no_clip_coefficient=no_clip_coefficient,
                no_clip_raw=float(np.mean((p+no_clip_coefficient*z-y)**2))), ideal, served

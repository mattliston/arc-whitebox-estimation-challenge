"""Input-only FP64 diagnostic. Not a deployable FlopScope estimator."""

import numpy as np
from scipy.special import erfcx, log_ndtr, ndtr


LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2)


def _log_d(u):
    value = 1 - u * SQRT_PI_OVER_2 * erfcx(u / np.sqrt(2))
    if np.any(value <= 0) or not np.isfinite(value).all():
        raise FloatingPointError("Mills derivative lost positivity")
    return np.log(value)


def _log_c(a):
    out = np.empty_like(a)
    negative = a < 0
    out[negative] = -0.5*a[negative]**2-LOG_SQRT_2PI+_log_d(-a[negative])
    b = a[~negative]
    out[~negative] = np.log(b*ndtr(b)+np.exp(-0.5*b*b-LOG_SQRT_2PI))
    return out


def _log_r(t):
    out = np.empty_like(t)
    negative = t <= 0
    out[negative] = np.log(SQRT_PI_OVER_2 * erfcx(-t[negative]/np.sqrt(2)))
    b = t[~negative]
    out[~negative] = log_ndtr(b)+0.5*b*b+LOG_SQRT_2PI
    return out


def stein(s, mu, sigma):
    s, mu, sigma = np.broadcast_arrays(np.asarray(s, dtype=np.float64),
                                      np.asarray(mu, dtype=np.float64),
                                      np.asarray(sigma, dtype=np.float64))
    if not (np.isfinite(s).all() and np.isfinite(mu).all()
            and np.isfinite(sigma).all() and np.all(sigma > 0)):
        raise FloatingPointError("invalid Stein domain")
    alpha = mu/sigma
    z = (s-mu)/sigma
    if np.max(np.abs(alpha)) > 64 or np.max(np.abs(z)) > 256:
        raise FloatingPointError("outside preregistered Stein domain")
    left = s <= 0
    a = np.where(left, alpha, -alpha)
    t = np.where(left, z, -z)
    lc, lr = _log_c(a), _log_r(t)
    lh = np.empty_like(t)
    negative = t < 0
    lh[negative] = _log_d(-t[negative])
    nonnegative = ~negative
    b = t[nonnegative]
    lb = np.full_like(b, -np.inf)
    positive = b > 0
    lb[positive] = np.log(b[positive])
    lh[nonnegative] = np.logaddexp(0., lb+lr[nonnegative])
    q = np.exp(lc+lr)
    g = np.where(left, -q, q-1)
    gp = -np.exp(lc+lh-np.log(sigma))
    if not (np.isfinite(g).all() and np.isfinite(gp).all()
            and np.all(q <= 1+2e-10) and np.all(gp <= 0)):
        raise FloatingPointError("invalid Stein output")
    return g, gp


def evaluate(weights, mu, sigma, x0, z0, t, probes, *, detailed=False, guard=None):
    """Two probe replicas, fixed actual rows, one difference recurrence."""
    depth, n, _ = weights.shape
    rows = len(x0)
    if probes.shape != (2, rows, n):
        raise ValueError("expected exactly two complete probe replicas")
    x = np.array(x0, dtype=np.float64, copy=True)
    y = t[:, None]*x0 + np.sqrt(1-t*t)[:, None]*z0
    slopes = ndtr(mu/sigma)
    difference = np.zeros_like(x)
    tx = ty = tc = None
    mask_x = mask_y = None
    histories = []
    max_z = 0.
    for layer in range(depth):
        w = weights[layer]
        ax, ay = x@w, y@w
        if layer == 0:
            shared = probes@w
            tx, ty, tc = shared, shared, shared
        else:
            tx = (tx*mask_x[None, :, :])@w
            ty = (ty*mask_y[None, :, :])@w
            tc = (tc*slopes[layer-1][None, None, :])@w
        g, gp = stein(ax, mu[layer], sigma[layer])
        control_products = tx*ty-tc*tc
        ds = -gp*(control_products[0]-control_products[1])
        if layer == 0:
            difference = ds
        else:
            difference = ds-g*np.roll(difference@w, 1, axis=0)
        max_z = max(max_z, float(np.max(np.abs((ax-mu[layer])/sigma[layer]))))
        mask_x, mask_y = ax > 0, ay > 0
        if detailed:
            histories.append(dict(ax=ax.copy(), ay=ay.copy(), tx=tx.copy(),
                                  ty=ty.copy(), tc=tc.copy(), g=g.copy(), gp=gp.copy(),
                                  difference=difference.copy()))
        x, y = np.maximum(ax, 0), np.maximum(ay, 0)
        if not np.isfinite(difference).all():
            raise FloatingPointError("nonfinite difference recurrence")
        if guard is not None:
            guard(layer)
    mean_difference = difference.mean(axis=0)
    statistic = float(rows*np.dot(mean_difference, mean_difference)/(2*n))
    return dict(mean_difference=mean_difference, statistic=statistic,
                max_abs_standardized_deviation=max_z, histories=histories)

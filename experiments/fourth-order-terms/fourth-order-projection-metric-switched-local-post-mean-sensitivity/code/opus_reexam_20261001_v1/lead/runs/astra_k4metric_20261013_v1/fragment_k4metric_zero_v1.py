# Astra K4L projection metric v1; applies consistently to all existing _k4l_dk callers.
# New fragment only; native no-op, route parity, bill and final seal remain required.
_K4METRIC_LAMBDA = 0.0
_K4METRIC_PARENT = _k4l_dk

def _k4l_dk(loc, a, vv):
    """c (loc - beta vv^2) on the pushed rows (loc != 0), 0 elsewhere. a = mu / sigma, vv = pre-activation variances (n,)."""
    if _K4METRIC_LAMBDA == 0.0:
        return _K4METRIC_PARENT(loc, a, vv)
    _e2_H = _E2Hold()   # temporaries of this call, released together
    loc = loc.astype(fnp.float64)
    a2 = a * a
    q = _e2_H(a2 - 1) * _e2_H(fnp.exp(_e2_H(a2 * -0.5)))
    pushed = fnp.not_equal(loc, 0.0)
    iso = vv * vv
    wi = _e2_H(_e2_H(_e2_H(q * q) + 1e-12) * pushed) * iso
    beta = _e2_H(fnp.sum(_e2_H(wi * loc))) / _e2_H(fnp.sum(_e2_H(wi * iso)))
    wi_mean = _e2_H(_e2_H(_e2_H(q * q) + 1e-12) * pushed) / _e2_H(fnp.maximum(vv, 1e-12))
    beta_mean = _e2_H(fnp.sum(_e2_H(wi_mean * loc))) / _e2_H(fnp.maximum(_e2_H(fnp.sum(_e2_H(wi_mean * iso))), 1e-300))
    beta = beta_mean if _K4METRIC_LAMBDA == 1.0 else _e2_H(beta + _e2_H(_K4METRIC_LAMBDA * _e2_H(beta_mean - beta)))
    dk = _e2_H(loc - _e2_H(beta * iso)) * pushed
    return dk if isinstance(_K4L_C, tuple) or _K4L_C == 1.0 else dk * _K4L_C

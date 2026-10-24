"""One fixed post-birth L14 residual redistribution, paid FP64 arithmetic."""

def _bcls14(Bpost, b, rest, rest_diag):
    _bcls_hold = _E2Hold()
    C = _bcls_hold(Bpost.astype(fnp.float64))
    R = _bcls_hold(rest.astype(fnp.float64))
    fnp.fill_diagonal(R, rest_diag)
    v = _bcls_hold(fnp.diag(C))
    vv = _bcls_hold(v * v)
    normv = _bcls_hold(fnp.sum(vv))
    rownorm = _bcls_hold(fnp.sum(_bcls_hold(C * C), axis=1))
    rhs = _bcls_hold(_bcls_hold(_bcls_hold(R.T) @ v) + _bcls_hold(2 * _bcls_hold(fnp.sum(_bcls_hold(C * R), axis=1))))
    diagonal = _bcls_hold(_bcls_hold(normv + _bcls_hold(4 * rownorm)) + _bcls_hold(4 * vv))
    direction = _bcls_hold(rhs / _bcls_hold(fnp.maximum(diagonal, 1e-30)))
    normal_direction = _bcls_hold(_bcls_hold(_bcls_hold(_bcls_hold(normv * direction) + _bcls_hold(_bcls_hold(4 * rownorm) * direction)) + _bcls_hold(_bcls_hold(2 * v) * _bcls_hold(C @ direction))) + _bcls_hold(2 * _bcls_hold(_bcls_hold(C.T) @ _bcls_hold(v * direction))))
    step = _bcls_hold(_bcls_hold(fnp.sum(_bcls_hold(rhs * direction))) / _bcls_hold(fnp.maximum(_bcls_hold(fnp.sum(_bcls_hold(direction * normal_direction))), 1e-30)))
    delta = _bcls_hold(_bcls_hold(1.0 * step) * direction)
    plane = _bcls_hold(_bcls_hold(_bcls_hold(v[:, None]) * _bcls_hold(delta[None, :])) + _bcls_hold(_bcls_hold(2 * C) * _bcls_hold(delta[:, None])))
    return (_bcls_hold(b + delta), _bcls_hold(_bcls_hold(R - plane).astype(rest.dtype)), _bcls_hold(rest_diag - _bcls_hold(_bcls_hold(3 * v) * delta)))

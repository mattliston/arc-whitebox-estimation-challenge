"""One fixed post-birth L14 residual redistribution, paid FP64 arithmetic."""

def _bcls14(Bpost, b, rest, rest_diag):
    C = Bpost.astype(fnp.float64)
    R = rest.astype(fnp.float64)
    fnp.fill_diagonal(R, rest_diag)
    v = fnp.diag(C)
    vv = v * v
    normv = fnp.sum(vv)
    rownorm = fnp.sum(C * C, axis=1)
    rhs = R.T @ v + 2 * fnp.sum(C * R, axis=1)
    diagonal = normv + 4 * rownorm + 4 * vv
    direction = rhs / fnp.maximum(diagonal, 1e-30)
    normal_direction = (normv * direction + 4 * rownorm * direction
                        + 2 * v * (C @ direction)
                        + 2 * (C.T @ (v * direction)))
    step = fnp.sum(rhs * direction) / fnp.maximum(fnp.sum(direction * normal_direction), 1e-30)
    delta = 0.125 * step * direction
    plane = v[:, None] * delta[None, :] + 2 * C * delta[:, None]
    return b + delta, (R - plane).astype(rest.dtype), rest_diag - 3 * v * delta

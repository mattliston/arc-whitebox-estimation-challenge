"""Fixed K4 repeated-face row/diagonal projection; prediction-local vectors only."""
_K4R_ALPHA = 1.0
_K4R_LAYERS = (2,3,4,5,6,7,8,9,10,11,12,13,14,15)


def _k4r_residual(post_k4, post22, C, radial):
    n = int(C.shape[0])
    if n <= 2 or C.shape != (n,n) or post22.shape != (n,n) or post_k4.shape != (n,):
        raise RuntimeError('K4 row/diagonal full owned source shape')
    if any(str(x.dtype) != 'float64' for x in (post_k4,post22,C)):
        raise TypeError('K4 row/diagonal source must be float64')
    diagonal = fnp.diag(C)
    residual = 3 * (post22 - (radial / 3) * (fnp.outer(diagonal,diagonal) + 2 * C * C))
    fnp.fill_diagonal(residual, post_k4 - radial * diagonal * diagonal)
    return residual


def _k4r_project(residual):
    n = int(residual.shape[0])
    if n <= 2 or residual.shape != (n,n) or str(residual.dtype) != 'float64':
        raise RuntimeError('K4 row/diagonal projection source')
    diagonal = fnp.diag(residual)
    rows = 0.5 * (fnp.sum(residual,axis=0) + fnp.sum(residual,axis=1))
    constant = (fnp.sum(rows) - fnp.sum(diagonal)) / (n * (n - 1))
    u = (rows - diagonal - (n - 1) * constant) / (n - 2)
    v = diagonal - 2 * u - constant
    return u, v, constant


def _k4r_state(post_k4, post22, C, radial):
    return _k4r_project(_k4r_residual(post_k4,post22,C,radial))


def _k4r_delta(w32, state):
    u, v, constant = state
    n = int(w32.shape[0])
    if w32.shape != (n,n) or u.shape != (n,) or v.shape != (n,):
        raise RuntimeError('K4 row/diagonal read shape')
    if str(w32.dtype) != 'float32' or str(u.dtype) != 'float64' or str(v.dtype) != 'float64':
        raise TypeError('K4 row/diagonal read precision')
    w64 = w32.astype(fnp.float64)
    Q = w64 * w64
    total = fnp.sum(Q,axis=0)
    return (2 * total * (u @ Q) + constant * total * total) + v @ (Q * Q)

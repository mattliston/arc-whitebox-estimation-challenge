# Fixed GN-only projection, layers 10-12; unchanged scalar coefficients.
def _gn_perp_host(layer, host, increment):
    if layer not in (10, 11, 12):
        return increment
    norm2 = fnp.sum(host * host)
    if not bool(fnp.isfinite(norm2)) or float(norm2) < 0.0:
        raise ValueError('GN host projection requires finite nonnegative norm')
    if float(norm2) <= 1e-30:
        return increment
    h = host / fnp.sqrt(norm2)
    d = 1.0 - 2.0 * h * h
    if float(fnp.min(d)) <= 1e-6:
        return increment
    D = increment.astype(fnp.float64)
    A = (D + D.T) * 0.5
    fnp.fill_diagonal(A, 0.0)
    b = A @ h
    v = h / d
    z = b / d
    # d > 1e-6 implies denominator >= 1. Zero D needs no normalization.
    u = 2.0 * (z - v * fnp.sum(h * z) / (1.0 + fnp.sum(h * v)))
    Q = A - (fnp.outer(u, h) + fnp.outer(h, u)) * 0.5
    fnp.fill_diagonal(Q, 0.0)
    return Q

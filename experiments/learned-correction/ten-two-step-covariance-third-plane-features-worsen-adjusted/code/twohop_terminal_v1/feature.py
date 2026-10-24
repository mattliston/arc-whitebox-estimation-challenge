"""Fixed full two-step pair operators; no truth or learned coefficients."""


def twohop(fnp, covariance, third_plane, weights, sigma, alpha, pdf):
    variance = fnp.maximum(fnp.diag(covariance), 1e-10)
    scale = fnp.sqrt(variance)
    correlation = covariance / (scale[:, None] * scale[None, :])
    correlation = .5 * (correlation + correlation.T)
    fnp.fill_diagonal(correlation, 0)
    third = third_plane / (variance[:, None] * scale[None, :])
    fnp.fill_diagonal(third, 0)
    u = (scale[:, None] * weights.astype(fnp.float64)).astype(fnp.float32)
    r = correlation.astype(fnp.float32)
    t = third.astype(fnp.float32)
    a = (r @ u).astype(fnp.float64)
    b = (t @ u).astype(fnp.float64)
    d = (t.T @ u).astype(fnp.float64)
    squared_sigma = sigma.astype(fnp.float64) ** 2
    q = fnp.stack([fnp.sum(a*b, axis=0), fnp.sum(a*d, axis=0),
                   fnp.sum(b*b, axis=0), fnp.sum(d*d, axis=0),
                   fnp.sum(b*d, axis=0)], axis=1) / squared_sigma[:, None]
    return fnp.concatenate([q * pdf[:, None], q * (alpha*pdf)[:, None]], axis=1).astype(fnp.float32)

"""Prospective covariance remainder; synthetic/source use only, no deployed hook."""

def _astra_cfmr_kernel(mu, covariance, variance, k3, radial):
    n = int(mu.shape[0])
    q = radial / 3.0
    if not (0.0 < float(q) < 1.0 / (5.0 + 10.0 ** 0.5)):
        return None
    cmu = covariance @ mu
    norm2 = fnp.sum(mu * cmu)
    if not float(norm2) > 0.0:
        return None
    u = cmu / fnp.sqrt(norm2)
    sigma = fnp.sqrt(variance)
    us = u / sigma
    denominator = 3.0 * fnp.sum(us * us)
    if not float(denominator) > 0.0:
        return None
    beta = fnp.sum((k3 / (variance * sigma)) * us) / denominator
    if not float(beta * beta) < float(q):
        return None
    c = (beta / q) * u
    v0 = variance - q * c * c
    if not float(fnp.min(v0)) > 0.0:
        return None
    s0 = fnp.sqrt(v0)
    a0 = mu / s0
    p0 = fnp.exp(-0.5 * a0 * a0) / (2.0 * 3.141592653589793) ** 0.5
    g0 = flops.stats.norm.cdf(a0)
    J = c * g0 + 0.5 * s0 * p0
    r10 = 10.0 ** 0.5
    nodes = fnp.asarray((0.0, (5.0-r10)**0.5, -(5.0-r10)**0.5,
                         (5.0+r10)**0.5, -(5.0+r10)**0.5), dtype=fnp.float64) * fnp.sqrt(q)
    weights = (8.0 / 15.0, (7.0 + 2.0 * r10) / 60.0,
               (7.0 + 2.0 * r10) / 60.0, (7.0 - 2.0 * r10) / 60.0,
               (7.0 - 2.0 * r10) / 60.0)
    m = mu[None, :] + c[None, :] * nodes[:, None]
    s = s0[None, :] * fnp.sqrt(1.0 + nodes[:, None])
    a = m / s
    p = fnp.exp(-0.5 * a * a) / (2.0 * 3.141592653589793) ** 0.5
    M = m * flops.stats.norm.cdf(a) + s * p
    w = fnp.asarray(weights, dtype=fnp.float64)
    average = fnp.sum(M * w[:, None], axis=0)
    centered = M - average[None, :]
    weighted = centered * w[:, None]
    result = fnp.einsum('ai,aj->ij', weighted, centered, optimize=False) - q * fnp.outer(J, J)
    fnp.fill_diagonal(result, 0.0)
    return result

_ASTRA_CFMR_PARENT = nonlinear
_ASTRA_CFMR_STATS = dict(seen=[], applied=[])

def _astra_cfmr_nonlinear(*args, **kwargs):
    out = _ASTRA_CFMR_PARENT(*args, **kwargs)
    layer = int(_PHYS_LAYER[0])
    if not 8 <= layer <= 14:
        return out
    _ASTRA_CFMR_STATS['seen'].append(layer)
    mu, pre_cov = args[:2]
    pre_v = fnp.diag(pre_cov)
    if not float(fnp.min(pre_v)) > 0.0:
        return out
    k22 = args[4] if len(args) > 4 else kwargs.get('k22_override')
    if k22 is None:
        return out
    # Recover the actual shared scale from the restored diagonal of K22.
    # Its diagonal is radial * v^2 even when the off-diagonal has a trained scale.
    radial = fnp.mean(fnp.diag(k22) / (pre_v * pre_v))
    m, post_cov, target, radial_new, aux = out
    correction = _astra_cfmr_kernel(mu, pre_cov, pre_v, aux['pre_k3'], radial)
    if correction is None:
        return out
    post_cov = post_cov + correction
    _ASTRA_CFMR_STATS['applied'].append(layer)
    return m, post_cov, target, radial_new, aux

nonlinear = _astra_cfmr_nonlinear

# Register exactly the intended global wrapper in the existing dense identity snapshot.
# Keep every other source identity and the original dense guard unchanged.
assert sum(name == 'nonlinear' for name, value in _ASTRA_DENSE_CONTEXT) == 1
assert all((value is _ASTRA_CFMR_PARENT) if name == 'nonlinear'
           else (globals().get(name) is value) for name, value in _ASTRA_DENSE_CONTEXT)
_ASTRA_DENSE_CONTEXT = tuple((name, _astra_cfmr_nonlinear if name == 'nonlinear' else value)
                             for name, value in _ASTRA_DENSE_CONTEXT)

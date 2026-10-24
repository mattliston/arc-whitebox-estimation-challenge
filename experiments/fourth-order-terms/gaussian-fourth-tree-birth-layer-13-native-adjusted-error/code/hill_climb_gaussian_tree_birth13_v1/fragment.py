# One post12 Gaussian tree birth, consumed as k4/Q31 at pre13. No Q22 carry.
_GT13_STATS = dict(captures=0, consumptions=0, products=0)


def _gt13_prepare(Cpre, aux, Cpost):
    # Symmetry is required by the connected-tree contraction, not positivity.
    C = (Cpre + Cpre.T) * 0.5
    g = fnp.copy(aux['gate'])
    d = aux['pdf'] / aux['sigma']
    e = (-aux['alpha'] * aux['pdf']) / (aux['sigma'] ** 2)
    v = fnp.diag(C)
    h = d * g
    a = h * v
    C2 = C * C
    x = (d * d + e * g) * v
    # Sum of the complete leading-tree post-coordinate Q22, in O(n^2).
    trace = 4 * (a @ (C @ a) + x @ (C2 @ (g * g)) + h @ ((C2 * C) @ h))
    metric = (Cpost + Cpost.T) * 0.5
    den = fnp.trace(metric) ** 2 + 2 * fnp.sum(metric * metric, dtype=fnp.float64)
    if not bool(fnp.isfinite(den)) or not bool(den > 0):
        raise ValueError('GT13 source trace metric')
    beta = 3 * trace / den
    if not bool(fnp.isfinite(beta)):
        raise ValueError('GT13 nonfinite source radial coefficient')
    _GT13_STATS['captures'] += 1
    return C.astype(fnp.float32), g, d, e, beta


def _gt13_faces(source, W32, W64, Cnext, mu, K21):
    C32, g, d, e, beta = source
    # Only the four matrix contractions are FP32. Existing _mm selects its
    # paid D7 square route at full width. All pointwise source algebra is FP64.
    B32 = _mm(C32, (g[:, None] * W64).astype(fnp.float32))
    B = B32.astype(fnp.float64)
    T32 = (d[:, None] * W64 * B).astype(fnp.float32)
    Y = _mm(C32, T32).astype(fnp.float64)
    first = 6 * d[:, None] * B * Y + e[:, None] * (B ** 3)
    second = 6 * d[:, None] * W64 * Y + 3 * e[:, None] * W64 * (B ** 2)
    tree31 = (_mm(first.astype(fnp.float32).T, W32).astype(fnp.float64)
              + _mm(second.astype(fnp.float32).T, B32).astype(fnp.float64))
    # Reuse the actual native Cpre13 instead of two new metric-transport GEMMs.
    v = fnp.diag(Cnext)
    delta4 = fnp.diag(tree31) - beta * v * v
    delta31 = tree31 - (beta * v)[:, None] * Cnext
    row_scale = _k31_row_scale(mu, fnp.diag(K21), v)
    induced = delta4 if row_scale is None else delta4 * row_scale
    extra31 = delta31 - (induced / v)[:, None] * Cnext
    if not bool(fnp.all(fnp.isfinite(delta4))) or not bool(fnp.all(fnp.isfinite(extra31))):
        raise ValueError('GT13 nonfinite source faces')
    _GT13_STATS['products'] += 4
    _GT13_STATS['consumptions'] += 1
    return delta4, extra31

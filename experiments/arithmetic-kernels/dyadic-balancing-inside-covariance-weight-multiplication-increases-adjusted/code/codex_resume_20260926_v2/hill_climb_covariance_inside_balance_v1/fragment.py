def _dyadic_covariance_first(a, b):
    global _W5_PORT_SCOPE
    if int(b.shape[1]) == 0 or int(a.shape[1]) == 0:
        return _mm(a, b)
    if str(a.dtype) != 'float32' or str(b.dtype) != 'float32':
        raise TypeError('Dyadic covariance first product requires float32')
    if int(a.shape[1]) != int(b.shape[0]):
        raise ValueError('Dyadic covariance first product inner shape')
    h = _E2Hold()
    an = h(fnp.max(h(fnp.abs(a)), axis=0))
    bn = h(fnp.max(h(fnp.abs(b)), axis=1))
    valid = h(h(an > 0) & h(bn > 0))
    safe_a = h(fnp.where(valid, an, 1.0))
    safe_b = h(fnp.where(valid, bn, 1.0))
    exponent = h(fnp.clip(h(fnp.rint(h(0.5 * h(fnp.log2(h(safe_b / safe_a)))))), -32, 32))
    d = h(h(fnp.exp2(exponent)).astype(fnp.float32))
    aa = h(a * h(d[None, :]))
    bb = h(b / h(d[:, None]))
    saved_scope = _W5_PORT_SCOPE
    _W5_PORT_SCOPE = None  # scaled operands cannot borrow an original-W grant
    _TR_LAST[0] = None
    try:
        return _mm(aa, bb)
    finally:
        _TR_LAST[0] = None  # scaled A overwrote the shared A transform arena
        _W5_PORT_SCOPE = saved_scope

# Immutable-base full transpose ownership; all other products fall back.
def _ubc_transpose_product(left, factor, uv, leg):
    right = factor.T
    run = _UBC_RUN[0]
    assert leg in ('ut', 'vt')
    slot = 0 if leg == 'ut' else 1
    if (run is None or run['uv'] is not uv or factor is not uv[slot]
            or _UBC_CONTEXT[0] is not None):
        return _shared_old384_mm(left, right)
    k, n = map(int, right.shape)
    context = dict(leg=leg, uv=uv, generation=run['generation'],
                   ancestry=[(right, (0, n, n, k))], usable=True)
    _UBC_CONTEXT[0] = context
    try:
        return _shared_old384_mm(left, right)
    finally:
        context['ancestry'].clear()
        _UBC_CONTEXT[0] = None

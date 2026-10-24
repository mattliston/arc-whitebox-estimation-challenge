# Fixed L10--14 old-carried polarization; prediction-local ownership only.
_OSP_CONTEXT = [None]
_OSP_PARENT_INNER = _uvp_inner
_OSP_PARENT_PREDICT = _predict_binary


def _osp_begin(U, state):
    ctx = _OTT_CONTEXT[0]
    _OSP_CONTEXT[0] = dict(layer=_UVP_CUR[0], state=state,
                           n=int(U.shape[0]) if ctx is None else ctx['n'],
                           rows=None if ctx is None else ctx['rows'],
                           count=int(U.shape[0]), packets=[], response=None)


def _osp_rows(rows):
    ctx = _OSP_CONTEXT[0]
    ctx['rows'] = rows if ctx['rows'] is None else ctx['rows'][rows]
    ctx['count'] = int(rows.shape[0])


def _osp_capture(uv, pu, pv):
    ctx = _OSP_CONTEXT[0]
    if ctx is None:
        return
    assert ctx['layer'] == _UVP_CUR[0] and 10 <= ctx['layer'] <= 14
    assert tuple(pu.shape) == tuple(pv.shape) == (ctx['count'], int(uv[0].shape[1]))
    assert str(pu.dtype) == str(pv.dtype) == 'float32'
    assert not any(packet[0] is uv for packet in ctx['packets'])
    ctx['packets'].append((uv, pu, pv))


def _osp_read(left, right, lam, live, state, response):
    ctx = _OSP_CONTEXT[0]
    if ctx is None:
        return _cp_read_young_sq(left, right, lam, live, state)
    assert ctx['layer'] == _UVP_CUR[0] and ctx['state'] is state
    assert int(left.shape[0]) == int(response.shape[0])
    uv = _uvp_state(state)
    expected = uv[1:]
    assert len(expected) == len(ctx['packets'])
    assert all(any(p[0] is item for p in ctx['packets']) for item in expected)
    if ctx['rows'] is None:
        assert ctx['count'] == ctx['n']
        ctx['response'] = response
    else:
        inverse = fnp.full(ctx['n'], ctx['count'], dtype=fnp.int64)
        fnp.put(inverse, ctx['rows'], fnp.arange(ctx['count'], dtype=fnp.int64))
        ctx['response'] = inverse[response]
    try:
        result = _cp_read_young_sq(left, right, lam, live, state)
        assert not ctx['packets'], 'old polarization packets not consumed'
        return result
    finally:
        _OSP_CONTEXT[0] = None


def _uvp_inner(left_sq, lam_cols, uv):
    ctx = _OSP_CONTEXT[0]
    if ctx is None:
        return _OSP_PARENT_INNER(left_sq, lam_cols, uv)
    assert ctx['layer'] == _UVP_CUR[0] and ctx['response'] is not None
    matches = [i for i, packet in enumerate(ctx['packets']) if packet[0] is uv]
    assert len(matches) == 1
    _, pu, pv = ctx['packets'].pop(matches[0])
    Fu, Fv2, blocks, ru = uv
    P = int(Fu.shape[1]); rows = int(left_sq.shape[0])
    assert int(lam_cols.shape[0]) == int(left_sq.shape[1]) == 2 * P
    assert sum(h for _, h in blocks) == P
    hold = _E2Hold()
    # Append a zero row: padded response selection may include omitted rows.
    zero = hold(fnp.zeros((1, P), dtype=fnp.float32))
    u = hold(hold(fnp.concatenate((pu, zero), axis=0))[ctx['response']])
    v = hold(hold(fnp.concatenate((pv, zero), axis=0))[ctx['response']])
    Su = hold(u * v)
    Dv = hold(hold(u * u) + hold(v * v))
    b = 0
    for a, width in blocks:
        assert width > 0 and 0 <= a and a + 2 * width <= 2 * P
        weight2 = hold(hold(lam_cols[a:a + width]) * 2.0)
        weight4 = hold(weight2 * 2.0)
        hold(fnp.multiply(Su[:, b:b + width], weight4, out=Su[:, b:b + width]))
        hold(fnp.multiply(Dv[:, b:b + width], weight2, out=Dv[:, b:b + width]))
        b += width
    inner = _shared_old384_mm(Dv, Fv2.T)
    inner_u = _shared_old384_mm(Su, Fu.T)
    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))
    _BH_HOLD.clear()
    return inner


def _predict_binary(*args, **kwargs):
    _OSP_CONTEXT[0] = None
    try:
        return _OSP_PARENT_PREDICT(*args, **kwargs)
    finally:
        _OSP_CONTEXT[0] = None

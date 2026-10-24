# Fixed half-pair PPS approximation of the COMPLETE folded old-base read.
# Source prefix and all original callsites stay unchanged. No saved indices.
_PCS18_LAYERS = (11, 12, 13, 14)
_PCS18_ACTIVE = [False]
_PCS18_CAPTURE = [None]
_PCS18_INNER_CONTEXT = [None]
_PCS18_STATS = dict(reads=[], retired=True)
_PCS18_PARENT_LEFT = _np_left
_PCS18_PARENT_INNER = _RTT_PARENT_INNER
_PCS18_PARENT_UVP = _uvp_inner
_PCS18_PARENT_PREDICT = Estimator.predict


def _pcs18_select(score, count):
    """Capped inclusion probabilities; fixed systematic phase one half."""
    hold = _E2Hold()
    w = hold(score.astype(fnp.float64))
    floor = hold(fnp.maximum(hold(fnp.max(w) * 1e-15), 1e-30))
    w = hold(fnp.maximum(w, floor))
    ordered = hold(-hold(fnp.sort(hold(-w))))
    # Reverse cumulative sums avoid catastrophic total-minus-prefix cancellation.
    reverse = hold(ordered[::-1])
    tails = hold(hold(fnp.cumsum(reverse))[::-1])
    grid = hold(fnp.arange(count, dtype=fnp.float64))
    scale_candidates = hold(hold(count - grid) / hold(tails[:count]))
    admitted = hold(hold(scale_candidates * hold(ordered[:count])) <= (1.0 + 1e-14))
    eligible = hold(hold(fnp.nonzero(admitted)[0])[:1])
    scale = hold(fnp.take(scale_candidates, eligible))
    probability = hold(fnp.minimum(1.0, hold(scale * w)))
    cumulative = hold(fnp.cumsum(probability))
    selected = hold(fnp.searchsorted(cumulative, hold(grid + 0.5), side='right'))
    inverse = hold(1.0 / hold(fnp.take(probability, selected)))
    return selected, inverse.astype(fnp.float32)


def _pcs18_score_chunk(left, lam, pp, live, start, width):
    """Exact-real pair Frobenius norm, no pair-by-pair output matrix."""
    hold = _E2Hold()
    ap = hold(hold(left[:, start:start + width]).astype(fnp.float64))
    am = hold(hold(left[:, start + width:start + 2 * width]).astype(fnp.float64))
    lp = hold(hold(lam[start:start + width]).astype(fnp.float64))
    lm = hold(hold(lam[start + width:start + 2 * width]).astype(fnp.float64))
    xp = hold(hold(fnp.take(hold(pp[:, start:start + width]), live, axis=0)).astype(fnp.float64))
    xm = hold(hold(fnp.take(hold(pp[:, start + width:start + 2 * width]), live, axis=0)).astype(fnp.float64))
    aa = hold(fnp.sum(hold(ap * ap), axis=0))
    bb = hold(fnp.sum(hold(am * am), axis=0))
    ab = hold(fnp.sum(hold(ap * am), axis=0))
    xx = hold(fnp.sum(hold(xp * xp), axis=0))
    yy = hold(fnp.sum(hold(xm * xm), axis=0))
    xy = hold(fnp.sum(hold(xp * xm), axis=0))
    first = hold(hold(aa * xx) * hold(lp * lp))
    second = hold(hold(bb * yy) * hold(lm * lm))
    cross = hold(hold(hold(2.0 * ab) * xy) * hold(lp * lm))
    norm2 = hold(hold(first + second) + cross)
    return fnp.sqrt(hold(fnp.maximum(norm2, 0.0)))


def _pcs18_mm(left, right):
    """Zero-pad only K to64 for the incumbent ordinary fast dispatcher."""
    m, k, n = int(left.shape[0]), int(left.shape[1]), int(right.shape[1])
    padded = (k + 63) // 64 * 64
    if padded == k:
        return _shared_old384_mm(left, right)
    hold = _E2Hold()
    a = hold(fnp.zeros((m, padded), dtype=fnp.float32))
    b = hold(fnp.zeros((padded, n), dtype=fnp.float32))
    hold(fnp.copyto(hold(a[:, :k]), left))
    hold(fnp.copyto(hold(b[:k]), right))
    return _shared_old384_mm(a, b)


def _pcs18_sample_inner(left, lam, uv, pp, live):
    """Sample paired COMPLETE left factors, including the existing NP fold."""
    Fu, Fv, blocks, ru = uv
    hold = _E2Hold()
    score = hold(fnp.concatenate([hold(_pcs18_score_chunk(left, lam, pp, live, int(a), int(h)))
                                  for a, h in blocks]))
    pairs = int(Fu.shape[1])
    selected, inverse = _pcs18_select(score, pairs // 2)
    hold((selected, inverse))
    # Pair coordinates are fixed roster metadata, not data-selected host values.
    plus = hold(fnp.concatenate([hold(fnp.arange(int(a), int(a) + int(h))) for a, h in blocks]))
    minus = hold(fnp.concatenate([hold(fnp.arange(int(a) + int(h), int(a) + 2 * int(h))) for a, h in blocks]))
    ip = hold(fnp.take(plus, selected)); im = hold(fnp.take(minus, selected))
    wp = hold(hold(fnp.take(lam, ip)) * inverse)
    wm = hold(hold(fnp.take(lam, im)) * inverse)
    ap = hold(hold(fnp.take(left, ip, axis=1)) * wp)
    am = hold(hold(fnp.take(left, im, axis=1)) * wm)
    Su = hold(ap + am); Dv = hold(ap - am)
    sampled_u = hold(fnp.take(Fu, selected, axis=1))
    sampled_v = hold(fnp.take(Fv, selected, axis=1))
    # No UBX coefficient-transform cache: sampled columns change every layer.
    # Existing ordinary shared matrix dispatcher still selects its fast kernel.
    inner = _pcs18_mm(Dv, hold(sampled_v.T))
    inner_u = hold(_pcs18_mm(Su, hold(sampled_u.T)))
    common = hold(inner[:, :ru])
    hold(fnp.add(common, inner_u, out=common))
    return inner


def _np_left(l, *args, **kwargs):
    _PCS18_CAPTURE[0] = None
    result = _PCS18_PARENT_LEFT(l, *args, **kwargs)
    if (_PCS18_ACTIVE[0] and int(l) in _PCS18_LAYERS and len(args) >= 3
            and _E2_LAYER.get('ecut_cols') is not None):
        pp = args[1]
        if len(pp.shape) == 2 and int(pp.shape[0]) == 1024 and pp.dtype == 'float32':
            _PCS18_CAPTURE[0] = dict(layer=int(l), pp=pp, live=_E2_LAYER['ecut_cols'])
    return result


def _pcs18_inner(left, lam, state):
    capture = _PCS18_CAPTURE[0]
    if (not _PCS18_ACTIVE[0] or capture is None or _PCS18_INNER_CONTEXT[0] is not None
            or int(_RS_LAYER[0]) != capture['layer']):
        return _PCS18_PARENT_INNER(left, lam, state)
    context = dict(state=state, capture=capture, consumed=False)
    _PCS18_INNER_CONTEXT[0] = context
    try:
        return _PCS18_PARENT_INNER(left, lam, state)
    finally:
        _PCS18_INNER_CONTEXT[0] = None
        _PCS18_CAPTURE[0] = None


def _uvp_inner(left, lam, uv):
    context = _PCS18_INNER_CONTEXT[0]
    if context is None or context['consumed']:
        return _PCS18_PARENT_UVP(left, lam, uv)
    state = context['state']; cached = state.get('uvp')
    if cached is None or len(cached) < 2 or uv is not cached[1]:
        return _PCS18_PARENT_UVP(left, lam, uv)
    Fu, Fv, blocks, ru = uv
    pairs = int(Fu.shape[1]); end = 0
    tiled = bool(blocks)
    for start, width in blocks:
        tiled = tiled and int(start) == end and int(width) > 0
        end = int(start) + 2 * int(width)
    capture = context['capture']
    if (not tiled or pairs % 2 or pairs < 2 or end != 2 * pairs
            or int(left.shape[1]) != end or tuple(lam.shape) != (end,)
            or int(Fv.shape[1]) != pairs or int(Fu.shape[0]) != int(ru)
            or left.dtype != 'float32' or lam.dtype != 'float32'
            or int(capture['pp'].shape[1]) < end):
        return _PCS18_PARENT_UVP(left, lam, uv)
    context['consumed'] = True
    result = _pcs18_sample_inner(left, lam, uv, capture['pp'], capture['live'])
    _PCS18_STATS['reads'].append((capture['layer'], int(left.shape[0]), pairs, pairs // 2))
    return result


def _pcs18_predict(self, *args, **kwargs):
    _PCS18_CAPTURE[0] = None
    _PCS18_INNER_CONTEXT[0] = None
    _PCS18_STATS.update(reads=[], retired=False)
    _PCS18_ACTIVE[0] = True
    try:
        return _PCS18_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _PCS18_ACTIVE[0] = False
        _PCS18_CAPTURE[0] = None
        _PCS18_INNER_CONTEXT[0] = None
        _PCS18_STATS['retired'] = True
        _ASTRA_COMPUTE_STATS['pcs18'] = dict(reads=list(_PCS18_STATS['reads']), retired=True)


_RTT_PARENT_INNER = _pcs18_inner
Estimator.predict = _pcs18_predict

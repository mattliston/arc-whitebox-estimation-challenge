# Fixed OLD read-only pair pruning. Full roster, reconstruction and NP state remain original.
_OP75_ON = True
_OP75_ACTIVE = False
_OP75_CACHE = {}
_OP75_STATS = dict(builds=[], reads=[], retired=True)
_OP75_PARENT_INNER = _uvp_inner
_OP75_PARENT_PREDICT = _predict_binary


def _op75_select(uv, lam_cols):
    Fu, Fv, blocks, ru = uv
    P = int(Fu.shape[1]); K = min(P, ((3 * P + 255) // 256) * 64)
    assert ru == 256 and tuple(Fv.shape) == (320, P)
    assert sum(h for _, h in blocks) == P and tuple(lam_cols.shape) == (2 * P,)
    plus, minus = [], []
    for a, h in blocks:
        plus.extend(range(a, a + h)); minus.extend(range(a + h, a + 2 * h))
    assert sorted(plus + minus) == list(range(2 * P))
    hold = _E2Hold()
    ip = hold(fnp.asarray(plus, dtype=fnp.int64)); im = hold(fnp.asarray(minus, dtype=fnp.int64))
    # Stable nonnegative expansion of ||(u+v)^3-(u-v)^3||_F^2.
    u = hold(Fu.astype(fnp.float64)); v = hold(Fv.astype(fnp.float64))
    uu = hold(fnp.sum(hold(u * u), axis=0)); vv = hold(fnp.sum(hold(v * v), axis=0))
    uv0 = hold(fnp.sum(hold(u * hold(v[:ru])), axis=0))
    norm = hold(hold(4.0 * vv) * hold(hold(3.0 * hold(uu * uu)) + hold(vv * vv)))
    norm = hold(norm + hold(hold(24.0 * hold(uu + vv)) * hold(uv0 * uv0)))
    weight = hold(lam_cols[ip].astype(fnp.float64))
    score = hold(hold(weight * weight) * norm)
    keep = hold(fnp.sort(hold(fnp.argsort(hold(-score), kind='stable'))[:K]))
    columns = hold(fnp.concatenate([hold(ip[keep]), hold(im[keep])]))
    compact = (hold(Fu[:, keep]), hold(Fv[:, keep]), ((0, K),), ru)
    return keep, columns, compact


def _uvp_inner(left_sq, lam_cols, uv):
    if not _OP75_ON or not _OP75_ACTIVE or _UVP_CUR[0] not in (10, 11, 12):
        return _OP75_PARENT_INNER(left_sq, lam_cols, uv)
    run = _UBC_RUN[0]
    assert run is not None and run['uv'] is uv and run['owner'] is not None
    assert _UBX_CTX[0] is None and _UBC_CONTEXT[0] is None
    key = id(uv); entry = _OP75_CACHE.get(key)
    if entry is None:
        keep, columns, compact = _op75_select(uv, lam_cols)
        entry = (uv, run['owner'], run['generation'], keep, columns, compact)
        _OP75_CACHE[key] = entry
        _OP75_STATS['builds'].append([int(_UVP_CUR[0]), int(uv[0].shape[1]), int(compact[0].shape[1])])
    assert entry[0] is uv and entry[1] is run['owner'] and entry[2] == run['generation']
    _, _, generation, keep, columns, compact = entry
    hold = _E2Hold()
    selected_left = hold(left_sq[:, columns]); selected_lam = hold(lam_cols[columns])
    saved_uv, saved_generation = run['uv'], run['generation']
    run['uv'], run['generation'] = compact, (generation, 'op75-read-only', int(compact[0].shape[1]))
    try:
        result = _OP75_PARENT_INNER(selected_left, selected_lam, compact)
    finally:
        run['uv'], run['generation'] = saved_uv, saved_generation
    _OP75_STATS['reads'].append([int(_UVP_CUR[0]), int(left_sq.shape[0]), int(uv[0].shape[1]), int(compact[0].shape[1])])
    return result


def _predict_binary(mlp, *args, **kwargs):
    global _OP75_ACTIVE
    assert not _OP75_ACTIVE
    _OP75_CACHE.clear(); _OP75_STATS.update(builds=[], reads=[], retired=False)
    _OP75_ACTIVE = bool(_OP75_ON and int(mlp.width) == 1024 and int(mlp.depth) == 16 and kwargs.get('final_hook') is not None)
    try:
        return _OP75_PARENT_PREDICT(mlp, *args, **kwargs)
    finally:
        _OP75_ACTIVE = False; _OP75_CACHE.clear(); _OP75_STATS['retired'] = True

# Young-only negative-tail spherical trace; original old-row path is untouched.
_YNT_PARENT_PUSH = _push_bank_shared_newborn
_YNT_PARENT_DIAGONAL = _ott_diagonal
_YNT_PARENT_PREDICT = _predict_binary
_YNT_REPAIR = [None]
_YNT_STATS = dict(calls=0, crops=0, rows_dropped=0, fallback=0, retired=True)


def _ynt_trace(w, p, previous, newborn, coefficient, tail):
    q, selected, base, t, s = newborn
    q, births = int(q), int(selected.shape[0])
    weights = coefficient.astype(fnp.float64)
    B = base.astype(fnp.float64)
    t64, s64 = t.astype(fnp.float64), s.astype(fnp.float64)
    normB = fnp.sum(B * B, axis=0)
    ct = weights[q:q + births] * (normB + t64 * t64)
    cs = weights[q + births:] * (normB + s64 * s64)
    tau = B @ (ct + cs)
    if q:
        P = p[:, :q].astype(fnp.float64)
        carried = P @ (weights[:q] * fnp.sum(P * P, axis=0))
        fnp.put(tau, previous, tau[previous] + carried)
    fnp.put(tau, selected, tau[selected] + t64 * ct + s64 * cs)
    # Ambient weight dimension is 1024, including rows outside previous.
    wt = w[:, tail].astype(fnp.float64)
    normw = fnp.sum(wt * wt, axis=0)
    return (3 * normw / (int(w.shape[0]) + 2)) * (wt.T @ tau)


def _push_bank_shared_newborn(w, p, previous, newborn):
    global _W5_PORT_SCOPE
    _YNT_REPAIR[0] = None
    _YNT_STATS['calls'] = _YNT_STATS['calls'] + 1
    ctx = _OTT_CONTEXT[0]
    if ctx is None or not 7 <= _UVP_CUR[0] <= 14 or newborn is None:
        _YNT_STATS['fallback'] = _YNT_STATS['fallback'] + 1
        return _YNT_PARENT_PUSH(w, p, previous, newborn)
    n, m = int(w.shape[0]), int(w.shape[1])
    q, selected, base, t, s = newborn
    q, births = int(q), int(selected.shape[0])
    if (n != 1024 or ctx['n'] != n or tuple(ctx['a'].shape) != (m,)
            or tuple(ctx['live'].shape) != (m,) or tuple(p.shape) != (int(previous.shape[0]), q)
            or tuple(base.shape) != (n, births) or tuple(t.shape) != (births,)
            or tuple(s.shape) != (births,) or q + 2 * births > int(ctx['lam'].shape[0])):
        raise RuntimeError('young negative trace ownership')
    negative = (ctx['a'] < -2.5) & (~ctx['live'])
    kept = min(m, (int(fnp.nonzero(~negative)[0].shape[0]) + 63) // 64 * 64)
    if m - kept < 64:
        _YNT_STATS['fallback'] = _YNT_STATS['fallback'] + 1
        return _YNT_PARENT_PUSH(w, p, previous, newborn)
    score = fnp.where(negative, ctx['a'] + 2.5, 0.0)
    order = fnp.argsort(-score, kind='stable')
    rows, tail = order[:kept], order[kept:]
    if kept == 0 or bool(fnp.any(ctx['live'][tail])):
        raise RuntimeError('young negative trace padded alive roster')
    saved_scope = _W5_PORT_SCOPE
    _W5_PORT_SCOPE = None  # A cropped young wx cannot inherit a full-W grant.
    try:
        pushed = _YNT_PARENT_PUSH(w[:, rows], p, previous, newborn)
        width = q + 2 * births
        if tuple(pushed.shape) != (kept, width):
            raise RuntimeError('young negative trace pushed shape')
        inverse = fnp.full(m, kept, dtype=fnp.int64)
        fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))
        answer = fnp.concatenate((pushed, fnp.zeros((1, width), dtype=pushed.dtype)), axis=0)[inverse]
        coefficients = ctx['lam'][-width:]
        delta = _ynt_trace(w, p, previous, newborn, coefficients, tail)
        full_tail = tail if ctx['rows'] is None else ctx['rows'][tail]
        repair = fnp.zeros(ctx['n'], dtype=fnp.float64)
        fnp.put(repair, full_tail, delta)
        _YNT_REPAIR[0] = repair
        _YNT_STATS['crops'] = _YNT_STATS['crops'] + 1
        _YNT_STATS['rows_dropped'] = _YNT_STATS['rows_dropped'] + m - kept
        return answer
    except BaseException:
        _YNT_REPAIR[0] = None
        raise
    finally:
        _W5_PORT_SCOPE = saved_scope


def _ott_diagonal(diagonal, lam):
    try:
        result = _YNT_PARENT_DIAGONAL(diagonal, lam)
        return result if _YNT_REPAIR[0] is None else result + _YNT_REPAIR[0]
    finally:
        _YNT_REPAIR[0] = None


def _predict_binary(*args, **kwargs):
    _YNT_STATS.clear()
    _YNT_STATS.update(calls=0, crops=0, rows_dropped=0, fallback=0, retired=False)
    _YNT_REPAIR[0] = None
    try:
        return _YNT_PARENT_PREDICT(*args, **kwargs)
    finally:
        _YNT_REPAIR[0] = None
        _YNT_STATS['retired'] = True

# Inherited-young-only negative-tail trace; newborn and old paths stay exact.
_YNT_PARENT_PUSH = _push_bank_shared_newborn
_YNT_PARENT_DIAGONAL = _ott_diagonal
_YNT_PARENT_PREDICT = _predict_binary
_YNT_REPAIR = [None]
_YNT_STATS = dict(calls=0, crops=0, rows_dropped=0, fallback=0, retired=True)


def _ynt_trace(w, p, previous, coefficient, tail):
    P = p.astype(fnp.float64)
    weights = coefficient.astype(fnp.float64)
    tau = P @ (weights * fnp.sum(P * P, axis=0))
    # Trace is compact on previous; harmonic scale uses ALL 1024 weight rows.
    wt = w[:, tail].astype(fnp.float64)
    normw = fnp.sum(wt * wt, axis=0)
    return (3 * normw / (int(w.shape[0]) + 2)) * (wt[previous, :].T @ tau)


def _push_bank_shared_newborn(w, p, previous, newborn):
    global _W5_PORT_SCOPE
    _YNT_REPAIR[0] = None
    _YNT_STATS['calls'] = _YNT_STATS['calls'] + 1
    ctx = _OTT_CONTEXT[0]
    if ctx is None or not 7 <= _UVP_CUR[0] <= 14 or newborn is None or int(newborn[0]) == 0:
        _YNT_STATS['fallback'] = _YNT_STATS['fallback'] + 1
        return _YNT_PARENT_PUSH(w, p, previous, newborn)
    n, m = int(w.shape[0]), int(w.shape[1])
    q, selected, base, t, s = newborn
    q, births = int(q), int(selected.shape[0])
    if (n != 1024 or ctx['n'] != n or tuple(ctx['a'].shape) != (m,)
            or tuple(ctx['live'].shape) != (m,) or tuple(p.shape) != (int(previous.shape[0]), q)
            or tuple(base.shape) != (n, births) or tuple(t.shape) != (births,)
            or tuple(s.shape) != (births,) or q + 2 * births > int(ctx['lam'].shape[0])):
        raise RuntimeError('inherited young trace ownership')
    negative = (ctx['a'] < -2.5) & (~ctx['live'])
    kept = min(m, (int(fnp.nonzero(~negative)[0].shape[0]) + 63) // 64 * 64)
    if m - kept < 64:
        _YNT_STATS['fallback'] = _YNT_STATS['fallback'] + 1
        return _YNT_PARENT_PUSH(w, p, previous, newborn)
    score = fnp.where(negative, ctx['a'] + 2.5, 0.0)
    order = fnp.argsort(-score, kind='stable')
    rows, tail = order[:kept], order[kept:]
    if kept == 0 or bool(fnp.any(ctx['live'][tail])):
        raise RuntimeError('inherited young trace padded alive roster')
    _e2_H = _E2Hold()
    wp = _e2_H(w[previous, :]).T  # Same full-outer operand as the parent newborn call.
    saved_scope = _W5_PORT_SCOPE
    try:
        _W5_PORT_SCOPE = None  # Only cropped inherited work loses a full-W grant.
        try:
            wc = _e2_H(w[:, rows])
            inherited_rows = _mm(_e2_H(wc[previous, :]).T, _e2_H(p[:, :q]))
        finally:
            _W5_PORT_SCOPE = saved_scope
        if tuple(inherited_rows.shape) != (kept, q):
            raise RuntimeError('inherited young trace pushed shape')
        inverse = fnp.full(m, kept, dtype=fnp.int64)
        fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))
        inherited = fnp.concatenate([inherited_rows, fnp.zeros((1, q), dtype=inherited_rows.dtype)], axis=0)[inverse]
        # These newborn expressions and full operands are literally the parent.
        common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])
        _PHYS_HANDLES['newborn_common'] = common
        directions = _e2_H(w[selected, :]).T
        answer = fnp.concatenate([inherited, _e2_H(common + _e2_H(directions * t)), _e2_H(common + _e2_H(directions * s))], axis=1)
        coefficients = ctx['lam'][-(q + 2 * births):][:q]
        delta = _ynt_trace(w, p, previous, coefficients, tail)
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

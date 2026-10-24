# Fixed L10 residual-tail Ritz compression; no other join or rank changes.
_LTR_ON = True
_LTR_ACTIVE = [False]
_LTR_STATS = {}
_LTR_PARENT_PUSH_IMPL = _astra_bcc_push_impl
assert _R320_RULE == {7: 384, 10: 320, 13: 384}
assert _ASTRA_BCC_ON and _LCR_RU == 256 and _UVP_RU == 256 and 10 in _UVP_ONLY_LAYERS

def _ltr_extend_qr(q, r, extra):
    # Original q/r are read-only; q's first320 columns are copied unchanged.
    n, base = map(int, q.shape)
    add = int(extra.shape[1])
    p1 = fnp.matmul(q.T, extra)
    residual = extra - fnp.matmul(q, p1)
    p2 = fnp.matmul(q.T, residual)
    residual = residual - fnp.matmul(q, p2)
    qe, re = fnp.linalg.qr(residual, mode='reduced')
    fullq = fnp.concatenate([q, qe], axis=1)
    fullr = fnp.zeros((base + add, base + add), dtype=fnp.float32)
    fnp.copyto(fullr[:base, :base], r)
    fnp.copyto(fullr[:base, base:], p1 + p2)
    fnp.copyto(fullr[base:, base:], re)
    return fullq, fullr

def _ltr_compress(q, core, common=256, kept_tail=64):
    assert core is not None and str(q.dtype) == str(core.dtype) == 'float32'
    tail = core[common:, common:]
    values, vectors = fnp.linalg.eigh(tail)
    v = vectors[:, -kept_tail:]
    newtail = fnp.matmul(q[:, common:], v)
    result = fnp.concatenate([q[:, :common], newtail], axis=1)
    cross = fnp.matmul(core[:common, common:], v)
    small = fnp.matmul(v.T, fnp.matmul(tail, v))
    small = (small + small.T) * 0.5
    top = fnp.concatenate([core[:common, :common], cross], axis=1)
    bottom = fnp.concatenate([cross.T, small], axis=1)
    newcore = fnp.concatenate([top, bottom], axis=0)
    _shared_old384_finite(result, 'L10 fixed tail basis')
    _shared_old384_finite(newcore, 'L10 fixed tail core')
    return result, newcore

def _ltr_push_impl(layer, w32, p, previous, newborn, lam, state, rows=None):
    _UVP_CUR[0] = layer
    """One fixed old tier; four recent cohorts remain exact, no final-layer join.

    Projection precedes the inherited read. It intentionally approximates that
    inherited tensor; it does NOT secretly restore the old incumbent plane.
    The original next nonlinear star assembly and BD repair remain untouched.
    """
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    joins_here = _JOIN_SCHEDULE.get(layer)
    if not state['columns'] and joins_here is None:
        if state['joins']:
            raise RuntimeError('Shared-old384 premature confinement')
        return _push_bank_shared_newborn(w32 if rows is None else w32[:, rows], p, previous, newborn)
    rank = _r320_rank(layer) if joins_here is not None else (int(state['rank']) if state.get('rank') else 384)   # PT-60 per-join rank
    if int(w32.shape[0]) != 1024 or int(previous.shape[0]) < rank:
        raise RuntimeError('Shared-old384 fixed rank requires at least384 live rows')
    if newborn is None or len(state['cohorts']) != layer:
        raise RuntimeError('Shared-old384 cohort history')
    # PT-60: rank chosen above (per join)
    old_columns = int(state['columns'])
    _cp_old0 = old_columns   # E-48 --colpack E4: p holds the roster columns from here on, on the rows previous
    Qp = _xsg_rows(8412, state['Q'], previous) if old_columns else None   # opus-engineer a62 XSG
    if joins_here is not None:
        joining = int(_e2_H(joins_here[0]))
        start = int(_e2_H(_e2_H(state['cohorts'][joins_here[0]])[0]))
        stop = int(_e2_H(_e2_H(state['cohorts'][joins_here[1]])[0])) + int(_e2_H(_e2_H(state['cohorts'][joins_here[1]])[1]))
        width = stop - start
        if start != old_columns or width <= 0 or stop - _cp_old0 > int(p.shape[1]):
            raise RuntimeError('Shared-old384 intact signed cohort ownership')
        J = p[:, start - _cp_old0:stop - _cp_old0]
        if width % 2 or int(J.shape[1]) != width:
            raise RuntimeError('Shared-old384 complete two-leg cohort')
        # Frozen weighting rule: squared ACTUAL stored signed-cube coefficients.
        # Both equal-and-opposite pair members receive equal nonnegative weight.
        # No tensor-error, cancellation, spectrum, or endpoint certificate follows.
        coefficient = lam[start:stop]
        weight = coefficient * coefficient
        Omega = w32[previous, :rank]
        if _JOIN_NYSTROM:   # PT-42 Nystrom join: Omega = E_rows (the 384 live rows of largest Gram diagonal); gathers replace J^T Omega, Qp^T Omega
            _jn_d = fnp.matmul(_e2_H(J * J), weight)
            if old_columns:
                _jn_d = _jn_d + fnp.sum(_e2_H(_e2_H(_shared_old384_mm(Qp, state['core'])) * Qp), axis=1)
            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
            assert layer == 10 and rank == 320 and int(previous.shape[0]) >= 384
            _ltr_order = fnp.argsort(-_jn_d.astype(fnp.float64), kind='stable')
            _ltr_extra_rows = fnp.sort(_ltr_order[320:384])
            _ltr_extra_y = _shared_old384_mm(J, weight[:, None] * J[_ltr_extra_rows, :].T)
            if old_columns:
                _ltr_extra_y = _ltr_extra_y + _shared_old384_mm(Qp, _shared_old384_mm(state['core'], Qp[_ltr_extra_rows, :].T))
            _jn_rows = fnp.concatenate([_jn_rows, _ltr_extra_rows])
            del _jn_d, _ltr_order, _ltr_extra_rows
        else:
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_join_rows_mm(_e2_H(J.T), Omega))))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_shared_old384_mm(_e2_H(Qp.T), Omega))))))
        _slr_require(_JOIN_NYSTROM, 'Nystrom core requires Nystrom columns')
        _need_core = layer < max(_JOIN_SCHEDULE)
        K = None
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        # Keep the original QR320 call and its common256 bytes. Fixed two-pass
        # residual QR adds64 directions without perturbing the original basis.
        Qn, unused_R = _ltr_extend_qr(Qn, unused_R, _ltr_extra_y)
        del _ltr_extra_y
        core = None
        if _need_core:   # opus-engineer a08: R K^-1 R' with K = Y[rows] = Qn[rows] R is Qn[rows]^-1 R' (one solve, no product)
            R64 = unused_R.astype(fnp.float32)
            core = fnp.linalg.solve(Qn[_jn_rows, :].astype(fnp.float32), R64.T).astype(fnp.float32)
            core = (core + core.T) * 0.5
            del R64
        Qn, core = _ltr_compress(Qn, core)
        _LTR_STATS['calls'] += 1
        _LTR_STATS['layers'].append(layer)
        _LTR_STATS['ranks'].append((384, 320, 256))
        del K, unused_R, Y, Omega, _jn_rows
        _shared_old384_finite(Qn, 'QR basis')
        if tuple(Qn.shape) != (int(previous.shape[0]), rank):
            raise RuntimeError('Shared-old384 QR shape')
        state['rank_new'] = rank   # PT-60: the new frame's rank
        Fj = _shared_old384_mm(_e2_H(Qn.T), J)
        joined_core = None
        if old_columns:
            overlap = _shared_old384_mm(_e2_H(Qn.T), Qp)
            factors = _slr_join_factors(layer, overlap, state, Fj)
            del overlap
        else:
            factors = Fj
        old_columns = stop
        state['columns'] = old_columns
        state['rank'] = state.pop('rank_new')   # PT-60
        _slr_finish_factors(state, factors)
        state['core'] = _shared_old384_finite(core, 'Nystrom approximate Gram core') if _need_core else None
        state['joins'].append(dict(layer=layer, cohort=joining, start=start, stop=stop, previous_active=int(_e2_H(previous.shape[0]))))
        _slr_layout(state, old_columns)
        active_basis = Qn
        del J, weight, coefficient, Fj, joined_core, factors, core
    else:
        if not old_columns or (layer >= max(_JOIN_SCHEDULE) and len(state['joins']) != len(_JOIN_SCHEDULE)):
            raise RuntimeError('Shared-old384 terminal join history')
        active_basis = Qp
    # Full1024 rows are mandatory for the original diagonal and omitted-radial
    # consumers. No response-only or active-only reconstruction discount.
    wx = w32 if rows is None else _xsg_cols(8480, w32, rows)   # PT-15 deadcut: projection rows only; the join above keeps the full W (opus-engineer a62 XSG)
    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm, _e2_H(_e2_H(_xsg_rows(8481, wx, previous)).T), active_basis)   # opus-engineer a62 XSG
    old = _astra_bcc_capture('old', lambda: _slr_reconstruct(U, state))
    _shared_old384_finite(old, 'all-row reconstructed old bank')
    state['pre_basis'] = U
    # Preserve the exact existing young/newborn implementation and column order.
    # The original full p remains present and paid; no post-gating work is removed.
    inherited_columns, selected, base, t, s = newborn
    if old_columns > int(inherited_columns):
        raise RuntimeError('Shared-old384 attempted newborn confinement')
    young_newborn = (_e2_H(int(inherited_columns) - old_columns), selected, base, t, s)
    young = _astra_bcc_capture('young', lambda: _push_bank_shared_newborn(wx, _e2_H(p[:, old_columns - _cp_old0:]), previous, young_newborn))
    answer = _astra_bcc_join(old, young, state)
    if int(answer.shape[1]) != int(lam.shape[0]):
        raise RuntimeError('Shared-old384 complete original signed column roster')
    return answer

def _ltr_dispatch(layer, *args, **kwargs):
    if not _LTR_ON or layer != 10:
        return _LTR_PARENT_PUSH_IMPL(layer, *args, **kwargs)
    assert _LTR_ACTIVE[0] and _ASTRA_BCC_CTX[0] is not None
    return _ltr_push_impl(layer, *args, **kwargs)

_astra_bcc_push_impl = _ltr_dispatch
_LTR_PARENT_PREDICT = Estimator.predict
def _ltr_predict(self, *args, **kwargs):
    assert not _LTR_ACTIVE[0]
    _LTR_STATS.clear()
    _LTR_STATS.update(calls=0, layers=[], ranks=[], retired=False, value_cache_entries=0)
    _LTR_ACTIVE[0] = bool(_LTR_ON)
    try:
        return _LTR_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _LTR_ACTIVE[0] = False
        _LTR_STATS['retired'] = True
Estimator.predict = _ltr_predict

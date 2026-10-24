

# Fixed balanced JSK operands; 2026-10-21T01:35:58Z. Numerical-order hypothesis, no fitted targets.
assert _ASTRA_BCC_ON and _JOIN_NYSTROM
assert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14
assert _GNN_PARAMS is None and _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}
_O21_BALANCED_JSK = True

def _o21_jbl_young_gram(J1, selected_weight, rows):
    hold = _E2Hold()
    scale = hold(fnp.sqrt(selected_weight))
    A = hold(J1 * hold(scale[None, :]))
    selected = hold(A[rows, :])
    return _shared_old384_mm(A, hold(selected.T))

def _astra_bcc_push_impl(layer, w32, p, previous, newborn, lam, state, rows=None):
    _UVP_CUR[0] = layer
    """One fixed old tier; four recent cohorts remain exact, no final-layer join.

    Projection precedes the inherited read. It intentionally approximates that
    inherited tensor; it does NOT secretly restore the old incumbent plane.
    The original next nonlinear star assembly and BD repair remain untouched.
    """
    _e2_H = _E2Hold(); _e2_append_inline = _e2_H.append; _e2_value_inline = None   # E2: temporaries of this call, released together (one hidden free trip)
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
        joining = int((_e2_append_inline((_e2_value_inline := joins_here[0])) or _e2_value_inline))
        start = int((_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := state['cohorts'][joins_here[0]])) or _e2_value_inline)[0])) or _e2_value_inline))
        stop = int((_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := state['cohorts'][joins_here[1]])) or _e2_value_inline)[0])) or _e2_value_inline)) + int((_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := state['cohorts'][joins_here[1]])) or _e2_value_inline)[1])) or _e2_value_inline))
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
        if _JOIN_NYSTROM:   # [opus-lead] JSK-lite v6: today's Nystrom sketch on the kc heaviest joined columns (jsk/JOINSKETCH_v1.md); the old-tier term and the QR are unchanged
            _jn_JJ = J * J
            _jn_d = fnp.matmul(_jn_JJ, weight)
            _jn_c = fnp.sum(_jn_JJ, axis=0) * weight
            del _jn_JJ
            if old_columns:
                _jn_Qcore = _shared_old384_mm(Qp, state['core']); _jn_d = _jn_d + fnp.sum(_jn_Qcore * Qp, axis=1)  # [astra] Retain owned Qcore through young sketch.
            _jn_rows = fnp.sort(fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank])
            _jn_cols1 = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer]]
            J1 = J[:, _jn_cols1]
            Y = _o21_jbl_young_gram(J1, weight[_jn_cols1], _jn_rows)
            del J1, _jn_cols1, _jn_c
            if old_columns:
                Y = Y + _shared_old384_mm(_jn_Qcore, (Qp[_jn_rows, :]).T); del _jn_Qcore  # [astra] Same real product, one rank cube removed.
            del _jn_d
        else:
            Y = _shared_old384_mm(J, (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := weight[:, None])) or _e2_value_inline) * (_e2_append_inline((_e2_value_inline := _join_rows_mm(_e2_append_inline((_e2_value_inline := J.T)) or _e2_value_inline, Omega))) or _e2_value_inline)))) or _e2_value_inline))
            if old_columns:
                Y = Y + (_e2_append_inline((_e2_value_inline := _shared_old384_mm(Qp, _e2_append_inline((_e2_value_inline := _shared_old384_mm(state['core'], _e2_append_inline((_e2_value_inline := _shared_old384_mm(_e2_append_inline((_e2_value_inline := Qp.T)) or _e2_value_inline, Omega))) or _e2_value_inline))) or _e2_value_inline))) or _e2_value_inline)
        _slr_require(_JOIN_NYSTROM, 'Nystrom core requires Nystrom columns')
        _need_core = layer < max(_JOIN_SCHEDULE)
        K = None
        _jsym18_landmark(Y, _jn_rows); Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:   # opus-engineer a08: R K^-1 R' with K = Y[rows] = Qn[rows] R is Qn[rows]^-1 R' (one solve, no product)
            R64 = unused_R.astype(fnp.float32)
            core = fnp.linalg.solve(Qn[_jn_rows, :].astype(fnp.float32), R64.T).astype(fnp.float32)
            core = (core + core.T) * 0.5
            del R64
        del K, unused_R, Y, Omega, _jn_rows
        _shared_old384_finite(Qn, 'QR basis')
        if tuple(Qn.shape) != (int(previous.shape[0]), rank):
            raise RuntimeError('Shared-old384 QR shape')
        state['rank_new'] = rank   # PT-60: the new frame's rank
        Fj = _shared_old384_mm((_e2_append_inline((_e2_value_inline := Qn.T)) or _e2_value_inline), J)
        if _need_core:   # [opus-lead] JSK-lite v6: the Nystrom core is the subsample's Gram in the frame; rescale it to the full joined trace
            core = core * (fnp.sum(fnp.matmul(Fj * Fj, weight)) / fnp.trace(core))
        joined_core = None
        if old_columns:
            overlap = _shared_old384_mm((_e2_append_inline((_e2_value_inline := Qn.T)) or _e2_value_inline), Qp)
            factors = _slr_join_factors(layer, overlap, state, Fj)
            del overlap
        else:
            factors = Fj
        old_columns = stop
        state['columns'] = old_columns
        state['rank'] = state.pop('rank_new')   # PT-60
        _slr_finish_factors(state, factors)
        state['core'] = _shared_old384_finite(core, 'Nystrom approximate Gram core') if _need_core else None
        state['joins'].append(dict(layer=layer, cohort=joining, start=start, stop=stop, previous_active=int((_e2_append_inline((_e2_value_inline := previous.shape[0])) or _e2_value_inline))))
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
    inherited_columns, selected, base, t, ss = newborn
    if old_columns > int(inherited_columns):
        raise RuntimeError('Shared-old384 attempted newborn confinement')
    young_newborn = (int(inherited_columns) - old_columns, selected, base, t, ss)
    young_p = p[:, old_columns - _cp_old0:]
    fusion = _o18_fusion_prepare(layer, wx, young_p, previous, young_newborn, active_basis)
    saved_fusion = _O18_FUSION_CONTEXT[0]
    if saved_fusion is not None:
        raise RuntimeError('nested U/young fusion context')
    _O18_FUSION_CONTEXT[0] = fusion
    try:
        if fusion is not None and fusion['eligible']:
            young = _astra_bcc_capture('young', lambda: _push_bank_shared_newborn(wx, young_p, previous, young_newborn))
            U = fusion['U']
            if U is None or tuple(U.shape) != (int(wx.shape[1]), rank):
                raise RuntimeError('U/young fusion failed to preserve full basis')
            old = _astra_bcc_capture('old', lambda: _slr_reconstruct(U, state))
            _shared_old384_finite(old, 'all-row reconstructed old bank')
            state['pre_basis'] = U
        else:
            U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm, (_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := _xsg_rows(8481, wx, previous))) or _e2_value_inline).T)) or _e2_value_inline), active_basis)   # opus-engineer a62 XSG
            old = _astra_bcc_capture('old', lambda: _slr_reconstruct(U, state))
            _shared_old384_finite(old, 'all-row reconstructed old bank')
            state['pre_basis'] = U
            young = _astra_bcc_capture('young', lambda: _push_bank_shared_newborn(wx, young_p, previous, young_newborn))
    finally:
        _O18_FUSION_CONTEXT[0] = saved_fusion
    answer = _astra_bcc_join(old, young, state)
    if int(answer.shape[1]) != int(lam.shape[0]):
        raise RuntimeError('Shared-old384 complete original signed column roster')
    return answer

_FAST_RESCUE_FRAC = {}

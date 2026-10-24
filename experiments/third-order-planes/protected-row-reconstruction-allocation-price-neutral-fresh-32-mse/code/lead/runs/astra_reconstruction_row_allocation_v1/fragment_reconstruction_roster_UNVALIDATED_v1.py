# HELD single protected-hub reconstruction-roster arm; not scientifically validated.
# c149 source only. No rank, basis, read, tail-trace or tensor-state change.
_AROW_ON = True
_AROW_LAYERS = (10, 11, 12)
_AROW_ACTIVE = [False]
_AROW_GATE = [None]  # scalar id/floor only; never retains an array
_AROW_STATS = dict(calls=0, changed_calls=0, events=[], retired=True)
_AROW_PARENT_SELECT = select
_AROW_PARENT_RECON = _astra_bcc_recon_impl
_AROW_PARENT_PREDICT = _predict_binary


def select(score, floor):
    # Capture the exact active floor at the existing live-roster call. The
    # response and birth calls use different score objects and do not replace it.
    if _AROW_ON and _AROW_ACTIVE[0] and score is _E2_LAYER.get('g'):
        _AROW_GATE[0] = (id(score), floor)
    return _AROW_PARENT_SELECT(score, floor)


def _arow_event(m, kept, protected, selected_protected, swapped, reason):
    assert 0 <= kept <= m and 0 <= protected <= m and swapped >= 0
    _AROW_STATS['calls'] += 1
    _AROW_STATS['changed_calls'] += int(swapped > 0)
    _AROW_STATS['events'].append(dict(layer=int(_UVP_CUR[0]), rows=m,
        baseline_kept=kept, selected_kept=kept, tail_rows=m-kept,
        protected=protected, selected_protected=selected_protected,
        swapped=swapped, reason=reason, rank=320, materialized=True))


def _arow_choose(U, state, ctx, rows, tail, kept):
    m = int(U.shape[0])
    assert _AROW_ACTIVE[0] and _AROW_ON and _UVP_CUR[0] in _AROW_LAYERS
    assert state['lazy'] is None and int(state['rank']) == 320 and tuple(U.shape) == (m, 320)
    gate = _E2_LAYER['g']
    captured = _AROW_GATE[0]
    assert captured is not None and captured[0] == id(gate), 'Missing exact active gate floor'
    # Recompute the actual unchanged birth selection, including its stable
    # 32-row padding. Both response and birth rosters are full-network indices.
    uncertainty = fnp.minimum(gate, 1.0 - gate)
    unused_birth_rows, protected = _AROW_PARENT_SELECT(uncertainty, captured[1])
    fnp.put(protected, _E2_LAYER['ecut_rows'], True)
    variance = _E2_LAYER['vc']
    if ctx['rows'] is not None:
        protected = protected[ctx['rows']]
        uncertainty = uncertainty[ctx['rows']]
        variance = variance[ctx['rows']]
    assert tuple(protected.shape) == tuple(uncertainty.shape) == tuple(variance.shape) == (m,)
    count = int(fnp.nonzero(protected)[0].shape[0])
    if count >= kept:
        # No new deletion on this branch. Parent is not claimed to preserve
        # every hub when the protected union itself exhausts the budget.
        _arow_event(m, kept, count, int(fnp.nonzero(protected[rows])[0].shape[0]), 0, 'protected_parent')
        return rows, tail
    energy = fnp.sum(U * U, axis=1).astype(fnp.float64)
    importance = (energy * fnp.sqrt(energy)) * uncertainty / variance
    if not bool(fnp.all(fnp.isfinite(importance))):
        _arow_event(m, kept, count, int(fnp.nonzero(protected[rows])[0].shape[0]), 0, 'nonfinite_parent')
        return rows, tail
    priority = fnp.where(protected, float('inf'), importance)
    order = fnp.argsort(-priority, kind='stable')
    selected, omitted = order[:kept], order[kept:]
    selected_protected = int(fnp.nonzero(protected[selected])[0].shape[0])
    assert int(selected.shape[0]) == kept and int(omitted.shape[0]) == m-kept
    assert selected_protected == count, 'New deletion of a protected hub'
    # Count set changes without imposing the old numerical row order.
    old_mask = fnp.zeros(m, dtype=fnp.bool_)
    fnp.put(old_mask, rows, True)
    swapped = kept - int(fnp.nonzero(old_mask[selected])[0].shape[0])
    _arow_event(m, kept, count, selected_protected, swapped, 'allocated')
    return selected, omitted


def _arow_recon_impl(U, state):
    if _UVP_CUR[0] not in _LCR_LAYERS:
        return _LCR_PARENT_RECON(U, state)
    ctx = _OTT_CONTEXT[0]
    if ctx is None:
        return _OTT_ORIGINAL_RECON(U, state)
    a = ctx['a']
    m = int(U.shape[0])
    score = _OTT_Z - a
    negative_dead = (a < -_OTT_NEG_Z) & (~ctx['live'])
    score = fnp.where(negative_dead, a + _OTT_NEG_Z, score)
    kept = min(m, (int(fnp.nonzero(score >= 0.0)[0].shape[0]) + 63) // 64 * 64)
    if kept == m:
        _arow_event(m, kept, 0, 0, 0, "full_parent")
        return _OTT_ORIGINAL_RECON(U, state)
    ordered = fnp.argsort(-score, kind='stable')
    rows, tail = ordered[:kept], ordered[kept:]
    rows, tail = _arow_choose(U, state, ctx, rows, tail, kept)
    columns = int(state['columns'])
    lam = ctx['lam'][:columns]
    group = state['lazy']
    if group is None:
        old = _OTT_ORIGINAL_RECON(U[rows], state)
        delta = _ott_trace(U, state['F'], lam, tail)
    else:
        # One paid full mapping feeds both kept reconstruction and tail trace.
        mapped = _shared_old384_mm(U, group['map'])
        _shared_old384_finite(mapped, 'tail-trace lazy mapped full basis')
        uv = _uvp_state(state)
        _slr_require(uv[0] == 'l', 'local common lazy split')
        old = _astra_bcc_lazy(mapped, U, rows, uv, columns)
        split = int(group['prefix'])
        delta = (_ott_trace(mapped, group['base'], lam[:split], tail)
                 + _ott_trace(U, group['join'], lam[split:], tail))
    full_tail = tail if ctx['rows'] is None else ctx['rows'][tail]
    repair = fnp.zeros(ctx['n'], dtype=fnp.float64)
    fnp.put(repair, full_tail, delta)
    _OTT_DATA['repair'] = repair
    _OTT_DATA['full_tail'] = full_tail
    _OTT_STATS['calls'] += 1
    _OTT_STATS['rows_omitted'] += m-kept
    return _exact_expand_rows(old, rows, m)


def _astra_bcc_recon_impl(U, state):
    if (not _AROW_ON or not _AROW_ACTIVE[0] or _UVP_CUR[0] not in _AROW_LAYERS
            or state.get('lazy') is not None):
        return _AROW_PARENT_RECON(U, state)
    assert int(state['rank']) == 320 and _OTT_CONTEXT[0] is not None
    return _arow_recon_impl(U, state)


def _predict_binary(*args, **kwargs):
    assert not _AROW_ACTIVE[0] and _AROW_GATE[0] is None
    _AROW_STATS.clear()
    _AROW_STATS.update(calls=0, changed_calls=0, events=[], retired=False)
    _AROW_ACTIVE[0] = bool(_AROW_ON)
    try:
        return _AROW_PARENT_PREDICT(*args, **kwargs)
    finally:
        _AROW_ACTIVE[0] = False
        _AROW_GATE[0] = None
        _AROW_STATS['retired'] = True

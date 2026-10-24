# Fixed L7 rank320 plus one signed omitted-tensor-trace direction.
import math
assert _R320_RULE == {7:384,10:320,13:384}
_R320_RULE = dict(_R320_RULE, **{})
_R320_RULE[7] = 320
_TRACE321_STATS = dict(calls=0,layers=[],base_rank=0,stored_rank=0,source_rows=0,join_columns=0,augmented=False,fallback_reason=None,relative_trace_residual=0.0,residual_gram_diagonal=0.0)

def _trace321_enrich(Q, F, core, J, coefficient):
    hold = _E2Hold()
    n,r,columns = int(Q.shape[0]),int(Q.shape[1]),int(J.shape[1])
    assert r==320 and Q.shape==(n,r) and F.shape==(r,columns) and core.shape==(r,r)
    _TRACE321_STATS.update(calls=_TRACE321_STATS['calls']+1,layers=[7],base_rank=r,stored_rank=r,source_rows=n,join_columns=columns)
    # Trace of T - (Q Q^T)^tensor3 T is t - Q F(lam*||F_j||^2).
    # The subtracted term lies in span(Q), so its orthogonal residual is
    # exactly (I-Q Q^T)t in real arithmetic; no QF materialization is needed.
    column_norm2 = hold(fnp.sum(hold(J*J),axis=0))
    t = hold(J @ hold(coefficient*column_norm2)).astype(fnp.float64)
    Q64 = hold(Q.astype(fnp.float64))
    z = t - hold(Q64 @ hold(Q64.T @ t))
    z = z - hold(Q64 @ hold(Q64.T @ z))
    total2 = float(hold(fnp.sum(hold(t*t))))
    residual2 = float(hold(fnp.sum(hold(z*z))))
    if not (math.isfinite(total2) and math.isfinite(residual2)):
        raise RuntimeError('TRACE321 nonfinite trace direction')
    _TRACE321_STATS['relative_trace_residual'] = (max(residual2,0.0)/max(total2,1e-300))**0.5
    if residual2 <= 1e-24*max(total2,1e-300):
        _TRACE321_STATS['fallback_reason']='degenerate_trace'
        return Q,F,core
    z = hold(z / residual2**0.5).astype(fnp.float32)
    f = hold(z[None,:] @ J)
    weight = hold(coefficient*coefficient)
    weighted_f = hold(weight*f[0])
    cross = hold(F @ weighted_f)
    diagonal = float(hold(fnp.sum(hold(f[0]*weighted_f))))
    core_scale = abs(float(hold(fnp.sum(hold(fnp.diag(core))))))
    if not (math.isfinite(diagonal) and math.isfinite(core_scale)):
        raise RuntimeError('TRACE321 nonfinite residual Gram scalar')
    _TRACE321_STATS['residual_gram_diagonal']=diagonal
    if diagonal <= 1e-12*max(core_scale,1e-30):
        _TRACE321_STATS['fallback_reason']='degenerate_gram'
        return Q,F,core
    # Exact-arithmetic Nyström residual update in the augmented frame:
    # diag(core,0) + [cross;diagonal][cross;diagonal]^T / diagonal.
    # Its added outer product is PSD. Baseline floating core is NOT claimed
    # to be a certified PSD lower bound; no eigenvalue clipping/repair occurs.
    augmented_core = fnp.zeros((r+1,r+1),dtype=fnp.float64)
    hold(fnp.copyto(augmented_core[:r,:r],core))
    direction = hold(fnp.concatenate([hold(cross.astype(fnp.float64)),hold(fnp.asarray([diagonal],dtype=fnp.float64))]))
    direction = hold(direction / diagonal**0.5)
    hold(fnp.add(augmented_core,hold(fnp.outer(direction,direction)),out=augmented_core))
    answer=(fnp.concatenate([Q,z[:,None]],axis=1),fnp.concatenate([F,f],axis=0),augmented_core.astype(fnp.float32))
    _TRACE321_STATS.update(stored_rank=r+1,augmented=True,fallback_reason=None)
    return answer

def _astra_bcc_push_impl(layer, w32, p, previous, newborn, lam, state, rows=None):
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
            del _jn_d
        else:
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_join_rows_mm(_e2_H(J.T), Omega))))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_shared_old384_mm(_e2_H(Qp.T), Omega))))))
        _slr_require(_JOIN_NYSTROM, 'Nystrom core requires Nystrom columns')
        _need_core = layer < max(_JOIN_SCHEDULE)
        K = None
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
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
        Fj = _shared_old384_mm(_e2_H(Qn.T), J)
        if layer == 7:
            if old_columns or rank != 320:
                raise RuntimeError('TRACE321 fixed first-join geometry changed')
            Qn, Fj, core = _trace321_enrich(Qn, Fj, core, J, coefficient)
            rank = int(Qn.shape[1])
            state['rank_new'] = rank
            _shared_old384_finite(Qn, 'TRACE321 enriched basis')
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

_TRACE321_PARENT_RAW = _UBC_PARENT_PREDICT
def _trace321_predict(*args,**kwargs):
    _TRACE321_STATS.update(calls=0,layers=[],base_rank=0,stored_rank=0,source_rows=0,join_columns=0,augmented=False,fallback_reason=None,relative_trace_residual=0.0,residual_gram_diagonal=0.0)
    return _TRACE321_PARENT_RAW(*args,**kwargs)
_UBC_PARENT_PREDICT = _trace321_predict

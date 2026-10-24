

# BEGIN FABLE-ENGINEER2 SPLIT u/v OLD-TIER COORDINATES (uvp v0, 2026-09-25; BILL PROTOTYPE, NO ACCURACY CLAIM; lead: native bill of the
# split u/v old-tier frames before anyone invests). Every old-tier cohort holds birth PAIRS: column i of the t-leg and column i of the
# s-leg (the two halves of the cohort's columns). With F_t = F_u + F_v/2 and F_s = F_u - F_v/2 in the join frame, the pair mean F_u is
# kept only in the frame's leading r_u = 256 coordinates and the half difference F_v/2 in all of them (nested frame: one basis, one
# basis push, one trailing read). The per-layer products that scale with the coordinates then split in two:
#   rebuild  U F               -> U[:, :r_u] F_u  +  U (F_v/2), recombined per pair into the t and s columns (2 ops per cohort),
#   read     (L^2 lam) F^T     -> (a_t + a_s) F_u^T into the leading r_u columns  +  (a_t - a_s) (F_v/2)^T, a = L^2 lam (per cohort),
# for the materialized frame and for both blocks of the lazy L13 group. The split coordinates are formed once per join (per cohort
# slices, 2 ops each + 2 scales). The join itself (Nystrom Y, QR, Fj, overlap, core) is UNCHANGED here, so its share of the saving is
# not counted (conservative). One change inside the join: the Nystrom rows are taken in ENERGY order (they were index-sorted), so the
# QR frame's leading coordinates carry the most energy and the r_u truncation of the pair means drops the least.
_UVP_ON = True
_UVP_RU = 256
_UVP_STATS = {'recon': 0, 'read': 0, 'split': 0}


def _uvp_blocks(state, lo, hi):
    """(offset from lo, half width) of every two-leg cohort inside roster columns [lo, hi), in roster order."""
    out, covered = [], 0
    for start, width in state['cohorts']:
        s, w = int(start), int(width)
        if w > 0 and s >= lo and s + w <= hi:
            out.append((s - lo, w // 2))
            covered += w
    _slr_require(covered == hi - lo, 'u/v pair blocks tile the roster')
    return tuple(out)


def _uvp_split(F, blocks):
    """F (rank x cols, roster order) -> (F_u leading r_u rows x P, F_v/2 rank x P), pairs in roster order."""
    rank = int(F.shape[0])
    ru = min(_UVP_RU, rank)
    P = sum(h for _, h in blocks)
    Fu = fnp.empty((ru, P), dtype=fnp.float32)
    Fv2 = fnp.empty((rank, P), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(F[:ru, a:a + h], F[:ru, a + h:a + 2 * h], out=Fu[:, b:b + h]))
        h_(fnp.subtract(F[:, a:a + h], F[:, a + h:a + 2 * h], out=Fv2[:, b:b + h]))
        b += h
    h_(fnp.multiply(Fu, 0.5, out=Fu))
    h_(fnp.multiply(Fv2, 0.5, out=Fv2))
    _BH_HOLD.clear()
    _UVP_STATS['split'] += 1
    return (Fu, Fv2, blocks, ru)


def _uvp_recon(U, uv):
    """U (rows x rank) -> the t and s columns of every pair, roster order: U[:, :r_u] F_u +- U (F_v/2)."""
    Fu, Fv2, blocks, ru = uv
    rows = int(U.shape[0])
    Pu = _shared_old384_mm(U[:, :ru], Fu)
    Pv = _shared_old384_mm(U, Fv2)
    cols = 2 * int(Fu.shape[1])
    out = fnp.empty((rows, cols), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(Pu[:, b:b + h], Pv[:, b:b + h], out=out[:, a:a + h]))
        h_(fnp.subtract(Pu[:, b:b + h], Pv[:, b:b + h], out=out[:, a + h:a + 2 * h]))
        b += h
    _BH_HOLD.clear()
    del Pu, Pv
    return out


def _uvp_inner(left_sq, lam_cols, uv):
    """(L^2 lam) F^T with F in u/v form: rows x rank."""
    Fu, Fv2, blocks, ru = uv
    rows = int(left_sq.shape[0])
    P = int(Fu.shape[1])
    a_ = left_sq * lam_cols[None, :]
    Su = fnp.empty((rows, P), dtype=fnp.float32)
    Dv = fnp.empty((rows, P), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Su[:, b:b + h]))
        h_(fnp.subtract(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Dv[:, b:b + h]))
        b += h
    _BH_HOLD.clear()
    del a_
    inner = _shared_old384_mm(Dv, Fv2.T)
    inner_u = _shared_old384_mm(Su, Fu.T)
    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))
    _BH_HOLD.clear()
    del Su, Dv, inner_u
    return inner


_UVP_OLD_FINISH = _slr_finish_factors


def _slr_finish_factors(state, factors):
    _UVP_OLD_FINISH(state, factors)
    state['uvp'] = None
    if not _UVP_ON:
        return
    group = state['lazy']
    cols = int(state['columns'])
    if group is None:
        state['uvp'] = ('m', _uvp_split(state['F'], _uvp_blocks(state, 0, cols)))
    else:
        old = int(group['prefix'])
        state['uvp'] = ('l', _uvp_split(group['base'], _uvp_blocks(state, 0, old)), _uvp_split(group['join'], _uvp_blocks(state, old, cols)))


def _slr_reconstruct(U, state):
    """Full-row reconstruction in u/v form (uvp); the caller retains its original finite guard."""
    _e2_H = _E2Hold()
    _slr_layout(state, int(state['columns']))
    _slr_require(len(U.shape) == 2 and int(_e2_H(U.shape[0])) > 0 and (int(_e2_H(U.shape[1])) == int(state['rank'])) and (str(U.dtype) == 'float32'), 'current common basis')
    uv = state.get('uvp')
    _slr_require(uv is not None, 'u/v coordinates formed at the join')
    _UVP_STATS['recon'] += 1
    if uv[0] == 'm':
        return _uvp_recon(U, uv[1])
    group = state['lazy']
    mapped = _shared_old384_mm(U, group['map'])
    _shared_old384_finite(mapped, 'lazy mapped all-row basis')
    prefix = _uvp_recon(mapped, uv[1])
    recent = _uvp_recon(U, uv[2])
    old = fnp.concatenate([prefix, recent], axis=1)
    del mapped, prefix, recent
    return old


def _cp_slr_inner_sq(old_left, lam_old, state):
    """= _slr_inner with old_left already squared, in u/v form (uvp)."""
    _e2_H = _E2Hold()
    columns = int(state['columns'])
    _slr_layout(state, columns)
    _slr_require(len(old_left.shape) == 2 and int(_e2_H(old_left.shape[1])) == columns and (tuple(lam_old.shape) == (columns,)) and (str(old_left.dtype) == str(lam_old.dtype) == 'float32'), 'complete original weighted repeated-index read')
    uv = state.get('uvp')
    _slr_require(uv is not None, 'u/v coordinates formed at the join')
    _UVP_STATS['read'] += 1
    if uv[0] == 'm':
        return _uvp_inner(old_left, lam_old, uv[1])
    group = state['lazy']
    split = int(group['prefix'])
    inner0 = _uvp_inner(_e2_H(old_left[:, :split]), _e2_H(lam_old[:split]), uv[1])
    inner1 = _uvp_inner(_e2_H(old_left[:, split:]), _e2_H(lam_old[split:]), uv[2])
    mapped = _shared_old384_mm(inner0, _e2_H(group['map'].T))
    _shared_old384_finite(mapped, 'lazy mapped response inner')
    inner = mapped + inner1
    del inner0, inner1, mapped
    return inner


def _slr_inner(old_left, lam_old, state):
    """_slr_inner in u/v form (uvp): squares the left operand, then as _cp_slr_inner_sq."""
    return _cp_slr_inner_sq(old_left * old_left, lam_old, state)


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):
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
    Qp = state['Q'][previous, :] if old_columns else None
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
            _jn_rows = _e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]   # uvp: ENERGY order (was index-sorted), so the QR frame's leading columns carry the most energy
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
            del _jn_d, _jn_rows
        else:
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_join_rows_mm(_e2_H(J.T), Omega))))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_shared_old384_mm(_e2_H(Qp.T), Omega))))))
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        del unused_R, Y, Omega
        _shared_old384_finite(Qn, 'QR basis')
        if tuple(Qn.shape) != (int(previous.shape[0]), rank):
            raise RuntimeError('Shared-old384 QR shape')
        state['rank_new'] = rank   # PT-60: the new frame's rank
        Fj = _shared_old384_mm(_e2_H(Qn.T), J)
        joined_core = _shared_old384_mm(Fj, _e2_H(_e2_H(weight[:, None]) * _e2_H(Fj.T)))
        if old_columns:
            overlap = _shared_old384_mm(_e2_H(Qn.T), Qp)
            factors = _slr_join_factors(layer, overlap, state, Fj)
            core = _shared_old384_mm(overlap, _e2_H(_shared_old384_mm(state['core'], _e2_H(overlap.T))))
            core = core + joined_core
            del overlap
        else:
            factors, core = Fj, joined_core
        old_columns = stop
        state['columns'] = old_columns
        state['rank'] = state.pop('rank_new')   # PT-60
        _slr_finish_factors(state, factors)
        state['core'] = _shared_old384_finite(core, 'weighted Gram core')
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
    wx = w32 if rows is None else w32[:, rows]   # PT-15 deadcut: projection rows only; the join above keeps the full W
    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm, _e2_H(_e2_H(wx[previous, :]).T), active_basis)
    old = _slr_reconstruct(U, state)
    _shared_old384_finite(old, 'all-row reconstructed old bank')
    state['pre_basis'] = U
    # Preserve the exact existing young/newborn implementation and column order.
    # The original full p remains present and paid; no post-gating work is removed.
    inherited_columns, selected, base, t, s = newborn
    if old_columns > int(inherited_columns):
        raise RuntimeError('Shared-old384 attempted newborn confinement')
    young_newborn = (_e2_H(int(inherited_columns) - old_columns), selected, base, t, s)
    young = _push_bank_shared_newborn(wx, _e2_H(p[:, old_columns - _cp_old0:]), previous, young_newborn)
    answer = fnp.concatenate([old, young], axis=1)
    if int(answer.shape[1]) != int(lam.shape[0]):
        raise RuntimeError('Shared-old384 complete original signed column roster')
    return answer

# END FABLE-ENGINEER2 SPLIT u/v OLD-TIER COORDINATES

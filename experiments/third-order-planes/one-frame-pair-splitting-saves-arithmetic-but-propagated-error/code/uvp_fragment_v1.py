

# BEGIN FABLE-ENGINEER2 SPLIT u/v OLD-TIER FRAMES (uvp v1, 2026-09-25; BILL PROTOTYPE, NO ACCURACY CLAIM). v1 = v0 (rebuild and read in
# u/v form) plus the JOIN in u/v form, so the pair means get their OWN frame: at each join the pair means u = (x_t + x_s)/2 and the
# half differences v2 = (x_t - x_s)/2 of the joining columns are formed per cohort; Q_u = QR of the u Nystrom sample (energy-ordered
# rows, rank r_u = 256, plus the old u Gram), Q_c = QR of the v Nystrom sample (rank - r_u columns) with Q_u projected out, and the
# frame is Qn = [Q_u | Q_c] (nested: one basis, one basis push, one trailing read at rank 384). Coordinates: F_u = Q_u^T u (r_u rows),
# F_v/2 = Qn^T v2 (all rows); the weighted Gram core splits into core_u (r_u^2) and core_v (rank^2); a materialized join rotates the
# old u coordinates by the overlap's leading r_u x r_u block (old u outside Q_u is dropped) and the old v by the full overlap; the lazy
# L13 join keeps the old blocks and the map, as before. No F, no lazy group: the u/v state replaces them (_slr_layout checks it).
# Not ported (prototype): the virtual-rotation overflow envelope of the lazy map; unequal pair weights (averaged).
_UVP_ON = True
_UVP_RU = 256
_UVP_STATS = {'recon': 0, 'read': 0, 'join': 0}


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




def _slr_layout(state, columns):
    """u/v form (uvp v1): exact ordered pair roster over the materialized or lazy u/v coordinates."""
    uv = state.get('uvp')
    _slr_require(uv is not None and columns > 0 and columns == int(state['columns']), 'u/v state with a complete positive old column count')
    if uv['kind'] == 'm':
        P = 2 * int(uv['Fu'].shape[1])
        _slr_require(P == columns and int(uv['Fv2'].shape[1]) * 2 == columns and int(uv['Fv2'].shape[0]) == int(state['rank']), 'materialized u/v factors')
        return
    base, join = uv['base'], uv['join']
    old = int(uv['prefix'])
    _slr_require(2 * int(base['Fu'].shape[1]) == old and 2 * int(join['Fu'].shape[1]) == columns - old
                 and int(uv['map'].shape[0]) == int(state['rank']) and int(uv['map'].shape[1]) == int(base['Fv2'].shape[0]), 'lazy u/v blocks and map')


def _slr_reconstruct(U, state):
    """Full-row reconstruction in u/v form (uvp v1)."""
    _slr_layout(state, int(state['columns']))
    _slr_require(len(U.shape) == 2 and int(U.shape[1]) == int(state['rank']) and str(U.dtype) == 'float32', 'current common basis')
    uv = state['uvp']
    _UVP_STATS['recon'] += 1
    if uv['kind'] == 'm':
        return _uvp_recon(U, (uv['Fu'], uv['Fv2'], uv['blocks'], uv['ru']))
    mapped = _shared_old384_mm(U, uv['map'])
    _shared_old384_finite(mapped, 'lazy mapped all-row basis')
    b = uv['base']; j = uv['join']
    prefix = _uvp_recon(mapped, (b['Fu'], b['Fv2'], b['blocks'], b['ru']))
    recent = _uvp_recon(U, (j['Fu'], j['Fv2'], j['blocks'], j['ru']))
    old = fnp.concatenate([prefix, recent], axis=1)
    del mapped, prefix, recent
    return old


def _cp_slr_inner_sq(old_left, lam_old, state):
    """= _slr_inner with old_left already squared, in u/v form (uvp v1)."""
    columns = int(state['columns'])
    _slr_layout(state, columns)
    _slr_require(len(old_left.shape) == 2 and int(old_left.shape[1]) == columns and tuple(lam_old.shape) == (columns,)
                 and str(old_left.dtype) == str(lam_old.dtype) == 'float32', 'complete original weighted repeated-index read')
    uv = state['uvp']
    _UVP_STATS['read'] += 1
    if uv['kind'] == 'm':
        return _uvp_inner(old_left, lam_old, (uv['Fu'], uv['Fv2'], uv['blocks'], uv['ru']))
    split = int(uv['prefix'])
    b = uv['base']; j = uv['join']
    inner0 = _uvp_inner(old_left[:, :split], lam_old[:split], (b['Fu'], b['Fv2'], b['blocks'], b['ru']))
    inner1 = _uvp_inner(old_left[:, split:], lam_old[split:], (j['Fu'], j['Fv2'], j['blocks'], j['ru']))
    mapped = _shared_old384_mm(inner0, uv['map'].T)
    _shared_old384_finite(mapped, 'lazy mapped response inner')
    inner = mapped + inner1
    del inner0, inner1, mapped
    return inner


def _slr_inner(old_left, lam_old, state):
    """_slr_inner in u/v form (uvp v1): squares the left operand, then as _cp_slr_inner_sq."""
    return _cp_slr_inner_sq(old_left * old_left, lam_old, state)


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    _UVP_STATS['join'] += 1 if _JOIN_SCHEDULE.get(layer) is not None else 0
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
        # uvp v1: the join in u/v form (nested frame [Q_u | Q_c]); pair means u = (x_t + x_s)/2, half differences v2 = (x_t - x_s)/2
        nrow = int(J.shape[0])
        ru = min(_UVP_RU, rank)
        rc = rank - ru
        blocksJ = _uvp_blocks(state, start, stop)
        P = sum(h for _, h in blocksJ)
        Ju = fnp.empty((nrow, P), dtype=fnp.float32)
        Jv2 = fnp.empty((nrow, P), dtype=fnp.float32)
        w2 = fnp.empty((P,), dtype=fnp.float32)
        hh = _BH_HOLD.append
        bo = 0
        for ao, h in blocksJ:
            hh(fnp.add(J[:, ao:ao + h], J[:, ao + h:ao + 2 * h], out=Ju[:, bo:bo + h]))
            hh(fnp.subtract(J[:, ao:ao + h], J[:, ao + h:ao + 2 * h], out=Jv2[:, bo:bo + h]))
            hh(fnp.add(weight[ao:ao + h], weight[ao + h:ao + 2 * h], out=w2[bo:bo + h]))
            bo += h
        hh(fnp.multiply(Ju, 0.5, out=Ju))
        hh(fnp.multiply(Jv2, 0.5, out=Jv2))
        _BH_HOLD.clear()
        d_u = fnp.matmul(Ju * Ju, w2)
        d_v = fnp.matmul(Jv2 * Jv2, w2)
        if old_columns:
            uvo = state['uvp']
            ru_old = int(uvo['ru'])
            Qpu = Qp[:, :ru_old]
            d_u = d_u + fnp.sum(_shared_old384_mm(Qpu, state['core_u']) * Qpu, axis=1)
            d_v = d_v + fnp.sum(_shared_old384_mm(Qp, state['core_v']) * Qp, axis=1)
        rows_u = fnp.argsort(-(d_u.astype(fnp.float64)), kind='stable')[:ru]
        rows_v = fnp.argsort(-(d_v.astype(fnp.float64)), kind='stable')[:rc]
        Y_u = _shared_old384_mm(Ju, w2[:, None] * Ju[rows_u, :].T)
        if old_columns:
            Y_u = Y_u + _shared_old384_mm(Qpu, _shared_old384_mm(state['core_u'], Qpu[rows_u, :].T))
        Q_u, unused_R = fnp.linalg.qr(Y_u, mode='reduced')
        Y_v = _shared_old384_mm(Jv2, w2[:, None] * Jv2[rows_v, :].T)
        if old_columns:
            Y_v = Y_v + _shared_old384_mm(Qp, _shared_old384_mm(state['core_v'], Qp[rows_v, :].T))
        Y_c = Y_v - _shared_old384_mm(Q_u, _shared_old384_mm(Q_u.T, Y_v))
        Q_c, unused_R = fnp.linalg.qr(Y_c, mode='reduced')
        Qn = fnp.concatenate([Q_u, Q_c], axis=1)
        del unused_R, Y_u, Y_v, Y_c, d_u, d_v, rows_u, rows_v
        _shared_old384_finite(Qn, 'QR basis')
        if tuple(Qn.shape) != (int(previous.shape[0]), rank):
            raise RuntimeError('Shared-old384 QR shape')
        Fj_u = _shared_old384_mm(Q_u.T, Ju)
        Fj_v2 = _shared_old384_mm(Qn.T, Jv2)
        core_u = _shared_old384_mm(Fj_u, w2[:, None] * Fj_u.T)
        core_v = _shared_old384_mm(Fj_v2, w2[:, None] * Fj_v2.T)
        new_uv = dict(Fu=Fj_u, Fv2=Fj_v2, blocks=blocksJ, ru=ru)
        if old_columns:
            overlap = _shared_old384_mm(Qn.T, Qp)
            ov_uu = overlap[:ru, :ru_old]
            if uvo['kind'] != 'm':
                raise RuntimeError('uvp: one lazy group at most')
            if layer in _SLR_LAZY_LAYERS:
                state['uvp'] = dict(kind='l', base=uvo, join=dict(new_uv, blocks=_uvp_blocks(state, old_columns, stop)), map=overlap, prefix=old_columns, ru=ru)
            else:
                Fu = fnp.concatenate([_shared_old384_mm(ov_uu, uvo['Fu']), Fj_u], axis=1)
                Fv2 = fnp.concatenate([_shared_old384_mm(overlap, uvo['Fv2']), Fj_v2], axis=1)
                state['uvp'] = dict(kind='m', Fu=_shared_old384_finite(Fu, 'uvp u factors'), Fv2=_shared_old384_finite(Fv2, 'uvp v factors'), blocks=_uvp_blocks(state, 0, stop), ru=ru)
            core_u = core_u + _shared_old384_mm(ov_uu, _shared_old384_mm(state['core_u'], ov_uu.T))
            core_v = core_v + _shared_old384_mm(overlap, _shared_old384_mm(state['core_v'], overlap.T))
            del overlap, ov_uu
        else:
            state['uvp'] = dict(kind='m', Fu=_shared_old384_finite(Fj_u, 'uvp u factors'), Fv2=_shared_old384_finite(Fj_v2, 'uvp v factors'), blocks=_uvp_blocks(state, 0, stop), ru=ru)
        state['rank_new'] = rank
        old_columns = stop
        state['columns'] = old_columns
        state['rank'] = state.pop('rank_new')
        state['F'] = None
        state['lazy'] = None
        state['core_u'] = _shared_old384_finite(core_u, 'uvp weighted Gram core (u)')
        state['core_v'] = _shared_old384_finite(core_v, 'uvp weighted Gram core (v)')
        state['joins'].append(dict(layer=layer, cohort=joining, start=start, stop=stop, previous_active=int(_e2_H(previous.shape[0]))))
        _slr_layout(state, old_columns)
        active_basis = Qn
        del J, weight, coefficient, Ju, Jv2, w2, Fj_u, Fj_v2, core_u, core_v, new_uv
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


# END FABLE-ENGINEER2 SPLIT u/v OLD-TIER FRAMES

"""Fixed age7/rank224 tier inside the incumbent age4/rank384 tier.

Append to Cost150. All numerical work executes in the paid model call.
No labels, new learned parameters, or new scratch arena.
"""

_NESTED224_PARENT_STATE = _shared_old384_state
_NESTED224_PARENT_PUSH = _shared_old384_push
_NESTED224_PARENT_READ = _shared_old384_read
_NESTED224_RANK = 224


def _shared_old384_state():
    state = _NESTED224_PARENT_STATE()
    state.update(older_columns=0, older_joins=[], U2=None, F2=None, core2=None)
    return state


def _nested224_move(state, layer, w32, lam):
    mm = _shared_old384_mm
    finite = _shared_old384_finite
    previous_columns = int(state['older_columns'])
    cohort = layer - 8
    start, width = state['cohorts'][cohort]
    if start != previous_columns or len(state['older_joins']) != cohort or width % 2:
        raise RuntimeError('Nested224 complete chronological signed-cohort ownership')
    factors = state['F']
    mover = factors[:, :width]
    rank = int(factors.shape[0])
    thin = _NESTED224_RANK
    coefficients = lam[start:start + width]
    weight = coefficients * coefficients
    G = mm(mover, weight[:, None] * mover.T)
    if previous_columns:
        G = G + mm(state['U2'], mm(state['core2'], state['U2'].T))
    finite(G, 'nested combined Gram')
    sketch = w32[:rank, :thin].copy()
    for _pass in range(2):
        Un, unused_R = fnp.linalg.qr(mm(G, sketch), mode='reduced')
        finite(Un, 'nested QR basis')
        sketch = Un
        del unused_R
    moved = mm(Un.T, mover)
    if previous_columns:
        overlap = mm(Un.T, state['U2'])
        moved = fnp.concatenate([mm(overlap, state['F2']), moved], axis=1)
    core2 = mm(Un.T, mm(G, Un))
    state['core'] = finite(state['core'] - G + mm(Un, mm(core2, Un.T)),
                           'nested replaced full core')
    state['F'] = factors[:, width:]
    state['U2'] = finite(Un, 'nested map')
    state['F2'] = finite(moved, 'nested factors')
    state['core2'] = finite(core2, 'nested core')
    state['older_columns'] = start + width
    state['older_joins'].append(cohort)
    if int(moved.shape[1]) != int(state['older_columns']):
        raise RuntimeError('Nested224 complete nested factor roster')
    # Bound virtual factor magnitudes without materializing the full r1 x B bank.
    umax = fnp.max(fnp.abs(Un)).astype(fnp.float64)
    fmax = fnp.max(fnp.abs(moved)).astype(fnp.float64)
    if not bool((umax < 2.0**80) & (fmax < 2.0**80)
                & (thin * umax * fmax < 2.0**80)):
        raise RuntimeError('Nested224 virtual factor overflow guard')


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):
    if layer < 8:
        return _NESTED224_PARENT_PUSH(layer, w32, p, previous, newborn, lam, state)
    mm = _shared_old384_mm
    finite = _shared_old384_finite
    if int(w32.shape[0]) != 1024 or int(previous.shape[0]) < 384:
        raise RuntimeError('Nested224 first-tier geometry')
    if newborn is None or len(state['cohorts']) != layer:
        raise RuntimeError('Nested224 cohort history')
    old_columns = int(state['columns'])
    Qp = state['Q'][previous, :]
    if layer < 15:
        joining = layer - 5
        if len(state['joins']) != joining:
            raise RuntimeError('Nested224 first-tier join order')
        start, width = state['cohorts'][joining]
        stop = start + width
        if start != old_columns or width % 2 or stop > int(p.shape[1]):
            raise RuntimeError('Nested224 first-tier whole signed cohort')
        J = p[previous, start:stop]
        coefficient = lam[start:stop]
        weight = coefficient * coefficient
        Omega = w32[previous, :384]
        Y = mm(J, weight[:, None] * mm(J.T, Omega))
        Y = Y + mm(Qp, mm(state['core'], mm(Qp.T, Omega)))
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        del unused_R, Y, Omega
        finite(Qn, 'first-tier QR basis')
        Fj = mm(Qn.T, J)
        joined_core = mm(Fj, weight[:, None] * Fj.T)
        overlap = mm(Qn.T, Qp)
        rotated = mm(overlap, state['F'])
        core = mm(overlap, mm(state['core'], overlap.T))
        state['F'] = finite(fnp.concatenate([rotated, Fj], axis=1), 'middle factors')
        state['core'] = finite(core + joined_core, 'first-tier weighted core')
        if state['older_columns']:
            state['U2'] = finite(mm(overlap, state['U2']), 'rotated nested map')
        old_columns = stop
        state['columns'] = old_columns
        state['joins'].append(dict(layer=layer, cohort=joining, start=start,
                                   stop=stop, previous_active=int(previous.shape[0])))
        _nested224_move(state, layer, w32, lam)
        active_basis = Qn
        del J, coefficient, weight, Fj, joined_core, overlap, rotated, core
    else:
        if len(state['joins']) != 10 or len(state['older_joins']) != 7:
            raise RuntimeError('Nested224 terminal join history')
        active_basis = Qp
    U = mm(w32[previous, :].T, active_basis)
    nested_basis = finite(mm(U, state['U2']), 'nested prebasis')
    oldest = mm(nested_basis, state['F2'])
    middle = mm(U, state['F'])
    old = finite(fnp.concatenate([oldest, middle], axis=1), 'all-row old bank')
    if int(old.shape[1]) != old_columns:
        raise RuntimeError('Nested224 reconstructed old roster')
    state['pre_basis'] = U
    inherited_columns, selected, base, t, s = newborn
    if old_columns > int(inherited_columns):
        raise RuntimeError('Nested224 attempted newborn confinement')
    young_newborn = (int(inherited_columns) - old_columns, selected, base, t, s)
    young = _push_bank_shared_newborn(w32, p[:, old_columns:], previous, young_newborn)
    answer = fnp.concatenate([old, young], axis=1)
    if int(answer.shape[1]) != int(lam.shape[0]):
        raise RuntimeError('Nested224 original signed-column roster')
    return answer


def _shared_old384_read(left, right, lam, live_indices, state):
    older = int(state['older_columns'])
    if not older:
        return _NESTED224_PARENT_READ(left, right, lam, live_indices, state)
    mm = _shared_old384_mm
    old_columns = int(state['columns'])
    x2 = left[:, :older]
    x1 = left[:, older:old_columns]
    inner2 = _shared_old384_finite(
        mm(x2 * x2, (state['F2'] * lam[:older][None, :]).T), 'nested read inner')
    lifted = _shared_old384_finite(mm(inner2, state['U2'].T), 'nested lifted read')
    inner1 = mm(x1 * x1, (state['F'] * lam[older:old_columns][None, :]).T)
    old_plane = mm(lifted + inner1, state['pre_basis'][live_indices, :].T)
    young_left, young_right = left[:, old_columns:], right[:, old_columns:]
    young_plane = _mm(young_left * young_left, (young_right * lam[old_columns:]).T)
    return young_plane + old_plane

# Fixed older-only second tier. The complete original source precedes this.
_NESTED_ENABLED = True
_NESTED_RANK = 256
_nested_parent_state = _shared_old384_state
_nested_parent_push = _shared_old384_push
_nested_parent_read = _shared_old384_read


def _nested_move(J, coefficient, old_basis, old_factors, old_core, rank):
    """Project one intact moving cohort plus the previously nested prefix."""
    weight = coefficient * coefficient
    G = _shared_old384_mm(J, weight[:, None] * J.T)
    if old_basis is not None:
        G = G + _shared_old384_mm(old_basis, _shared_old384_mm(old_core, old_basis.T))
    _shared_old384_finite(G, 'nested entering weighted core')
    Q0, R0 = fnp.linalg.qr(G[:, :rank].copy(), mode='reduced')
    del R0
    _shared_old384_finite(Q0, 'nested first QR')
    Q2, R2 = fnp.linalg.qr(_shared_old384_mm(G, Q0), mode='reduced')
    del R2, Q0
    _shared_old384_finite(Q2, 'nested second QR')
    joining = _shared_old384_mm(Q2.T, J)
    if old_basis is None:
        factors = joining
    else:
        overlap = _shared_old384_mm(Q2.T, old_basis)
        prefix = _shared_old384_mm(overlap, old_factors)
        factors = fnp.concatenate([prefix, joining], axis=1)
        del overlap, prefix
    core = _shared_old384_mm(Q2.T, _shared_old384_mm(G, Q2))
    restored = _shared_old384_mm(Q2, _shared_old384_mm(core, Q2.T))
    _shared_old384_finite(factors, 'nested complete factors')
    _shared_old384_finite(core, 'nested projected core')
    _shared_old384_finite(restored, 'nested restored first-tier core')
    return Q2, factors, core, G, restored


def _shared_old384_state():
    state = _nested_parent_state()
    state.update(nested_columns=0, nested_basis=None, nested_factors=None,
                 nested_core=None, nested_joins=[], nested_started=False)
    return state


def _nested_layout(state):
    old = int(state['columns'])
    nested = int(state['nested_columns'])
    rank = _NESTED_RANK
    if (not 0 < nested < old or state['lazy'] is not None
            or tuple(state['F'].shape) != (384, old - nested)
            or tuple(state['nested_basis'].shape) != (384, rank)
            or tuple(state['nested_factors'].shape) != (rank, nested)
            or tuple(state['nested_core'].shape) != (rank, rank)):
        raise RuntimeError('Nested256 ordered factor layout')


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):
    if not _NESTED_ENABLED or layer < 8:
        return _nested_parent_push(layer, w32, p, previous, newborn, lam, state)
    if int(w32.shape[0]) != 1024 or int(previous.shape[0]) < 384:
        raise RuntimeError('Nested256 first-tier geometry')
    if newborn is None or len(state['cohorts']) != layer or state['lazy'] is not None:
        raise RuntimeError('Nested256 cohort or canonical middle state')
    old_columns = int(state['columns'])
    nested_columns = int(state['nested_columns'])
    Qp = state['Q'][previous, :]
    if layer < 15:
        joining = layer - 5
        if len(state['joins']) != joining:
            raise RuntimeError('Nested256 first-tier join order')
        start, width = state['cohorts'][joining]
        stop = start + width
        if start != old_columns or width <= 0 or width % 2:
            raise RuntimeError('Nested256 first-tier intact cohort')
        J = p[previous, start:stop]
        coefficient = lam[start:stop]
        weight = coefficient * coefficient
        Omega = w32[previous, :384]
        Y = _shared_old384_mm(J, weight[:, None] * _shared_old384_mm(J.T, Omega))
        Y = Y + _shared_old384_mm(Qp, _shared_old384_mm(state['core'], _shared_old384_mm(Qp.T, Omega)))
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        del unused_R, Y, Omega
        _shared_old384_finite(Qn, 'first-tier QR basis')
        Fj = _shared_old384_mm(Qn.T, J)
        joined_core = _shared_old384_mm(Fj, weight[:, None] * Fj.T)
        overlap = _shared_old384_mm(Qn.T, Qp)
        middle = fnp.concatenate([_shared_old384_mm(overlap, state['F']), Fj], axis=1)
        core = _shared_old384_mm(overlap, _shared_old384_mm(state['core'], overlap.T)) + joined_core
        basis = (_shared_old384_mm(overlap, state['nested_basis']) if nested_columns else None)
        mover = layer - 8
        begin, move_width = state['cohorts'][mover]
        if begin != nested_columns or len(state['nested_joins']) != mover or move_width % 2:
            raise RuntimeError('Nested256 second-tier intact cohort order')
        Q2, F2, S2, removed, restored = _nested_move(
            middle[:, :move_width], lam[begin:begin + move_width], basis,
            state['nested_factors'], state['nested_core'], _NESTED_RANK)
        state['core'] = _shared_old384_finite(core - removed + restored, 'repaired first-tier core')
        state['F'] = _shared_old384_finite(middle[:, move_width:].copy(), 'middle factor roster')
        state['nested_basis'], state['nested_factors'], state['nested_core'] = Q2, F2, S2
        state['nested_columns'] = begin + move_width
        state['nested_started'] = True
        state['nested_joins'].append(dict(layer=layer, cohort=mover, start=begin, width=move_width))
        state['columns'] = stop
        state['joins'].append(dict(layer=layer, cohort=joining, start=start, stop=stop,
                                   previous_active=int(previous.shape[0])))
        active_basis = Qn
        del J, coefficient, weight, Fj, joined_core, overlap, middle, core, basis, removed, restored
    else:
        if len(state['joins']) != 10 or len(state['nested_joins']) != 7:
            raise RuntimeError('Nested256 terminal cohort history')
        active_basis = Qp
    _nested_layout(state)
    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis', _shared_old384_mm,
                      w32[previous, :].T, active_basis)
    lifted = _shared_old384_mm(U, state['nested_basis'])
    _shared_old384_finite(lifted, 'nested all-row basis')
    prefix = _shared_old384_mm(lifted, state['nested_factors'])
    middle = _shared_old384_mm(U, state['F'])
    old = fnp.concatenate([prefix, middle], axis=1)
    _shared_old384_finite(old, 'all-row reconstructed old bank')
    state['pre_basis'] = U
    old_columns = int(state['columns'])
    inherited_columns, selected, base, t, s = newborn
    if old_columns > int(inherited_columns):
        raise RuntimeError('Nested256 newborn confinement')
    young_newborn = (int(inherited_columns) - old_columns, selected, base, t, s)
    young = _push_bank_shared_newborn(w32, p[:, old_columns:], previous, young_newborn)
    answer = fnp.concatenate([old, young], axis=1)
    if int(answer.shape[1]) != int(lam.shape[0]):
        raise RuntimeError('Nested256 complete signed roster')
    return answer


def _shared_old384_read(left, right, lam, live_indices, state):
    if not state['nested_started']:
        return _nested_parent_read(left, right, lam, live_indices, state)
    _nested_layout(state)
    old = int(state['columns'])
    split = int(state['nested_columns'])
    prefix = _shared_old384_mm(left[:, :split] * left[:, :split],
                               (state['nested_factors'] * lam[:split][None, :]).T)
    inner = _shared_old384_mm(prefix, state['nested_basis'].T)
    middle = left[:, split:old]
    inner = inner + _shared_old384_mm(middle * middle, (state['F'] * lam[split:old][None, :]).T)
    _shared_old384_finite(inner, 'nested complete response inner')
    old_plane = _shared_old384_mm(inner, state['pre_basis'][live_indices, :].T)
    young_left, young_right = left[:, old:], right[:, old:]
    young_plane = _mm(young_left * young_left, (young_right * lam[old:]).T)
    return young_plane + old_plane

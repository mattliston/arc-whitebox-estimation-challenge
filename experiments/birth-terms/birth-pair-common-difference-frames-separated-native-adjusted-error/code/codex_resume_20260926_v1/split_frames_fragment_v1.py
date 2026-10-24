

# Separate frames for the mean and half-difference of each signed birth pair.
# Research implementation, 2026-09-26. All model-dependent work uses flopscope.
# The column roster and signed weights are preserved; only the old-tier
# approximation changes. L10 is materialized and L13 retains one lazy map.
_UV2_RANK_U = {7: 384, 10: 256, 13: 192}
_UV2_STATS = {'joins': 0, 'recon': 0, 'read': 0}
_UV2_PARENT_PUSH = _shared_old384_push
_UV2_PARENT_INNER = _cp_slr_inner_sq


def _uv2_blocks(state, lo, hi):
    blocks = []
    covered = 0
    for start, width in state['cohorts']:
        if start >= lo and start + width <= hi:
            _slr_require(width > 0 and width % 2 == 0, 'uv2 intact pair cohort')
            blocks.append((start - lo, width // 2))
            covered += width
    _slr_require(covered == hi - lo, 'uv2 complete pair roster')
    return tuple(blocks)


def _uv2_pair_parts(J, lam, blocks):
    n = int(J.shape[0])
    pairs = int(J.shape[1]) // 2
    u = fnp.empty((n, pairs), dtype=fnp.float32)
    v = fnp.empty((n, pairs), dtype=fnp.float32)
    coeff = fnp.empty((pairs,), dtype=fnp.float32)
    hold = []
    b = 0
    for a, h in blocks:
        jp, jm = J[:, a:a+h], J[:, a+h:a+2*h]
        hold.append(fnp.add(jp, jm, out=u[:, b:b+h]))
        hold.append(fnp.subtract(jp, jm, out=v[:, b:b+h]))
        hold.append(fnp.copyto(coeff[b:b+h], lam[a:a+h]))
        # Algebra requires exactly opposite weights, which the source maintains.
        _slr_require(bool(fnp.all(lam[a:a+h] == -lam[a+h:a+2*h])), 'uv2 opposite pair weights')
        b += h
    hold.append(fnp.multiply(u, 0.5, out=u))
    hold.append(fnp.multiply(v, 0.5, out=v))
    return u, v, (coeff * coeff) * 2.0


def _uv2_join_part(J, weight, rank, old, Qp, layer):
    d = fnp.matmul(J * J, weight)
    if old is not None:
        d = d + fnp.sum(_shared_old384_mm(Qp, old['core']) * Qp, axis=1)
    rows = fnp.argsort(-d.astype(fnp.float64), kind='stable')[:rank]
    Y = _shared_old384_mm(J, weight[:, None] * J[rows, :].T)
    if old is not None:
        Y = Y + _shared_old384_mm(Qp, _shared_old384_mm(old['core'], Qp[rows, :].T))
    Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
    del unused_R, Y, d, rows
    _shared_old384_finite(Qn, 'uv2 new basis')
    Fn = _shared_old384_mm(Qn.T, J)
    core = _shared_old384_mm(Fn, weight[:, None] * Fn.T)
    if old is None:
        part = dict(F=Fn, lazy=None)
    else:
        _slr_require(old['lazy'] is None, 'uv2 at most one lazy map')
        overlap = _shared_old384_mm(Qn.T, Qp)
        core = core + _shared_old384_mm(overlap, _shared_old384_mm(old['core'], overlap.T))
        if layer == 13:
            part = dict(F=None, lazy=dict(base=old['F'], join=Fn, map=overlap))
            _slr_virtual_rotation_envelope(overlap, old['F'])
        else:
            part = dict(F=fnp.concatenate([_shared_old384_mm(overlap, old['F']), Fn], axis=1), lazy=None)
    part['rank'] = rank
    part['core'] = _shared_old384_finite(core, 'uv2 Gram core')
    if part['F'] is not None:
        _shared_old384_finite(part['F'], 'uv2 materialized factors')
    else:
        for name in ('base', 'join', 'map'):
            _shared_old384_finite(part['lazy'][name], 'uv2 lazy ' + name)
    return Qn, part


def _uv2_reconstruct_part(U, part):
    group = part['lazy']
    if group is None:
        return _shared_old384_mm(U, part['F'])
    mapped = _shared_old384_mm(U, group['map'])
    return fnp.concatenate([_shared_old384_mm(mapped, group['base']),
                            _shared_old384_mm(U, group['join'])], axis=1)


def _uv2_reconstruct(U, state):
    uv = state['uv2']
    ru = uv['u']['rank']
    pu = _uv2_reconstruct_part(U[:, :ru], uv['u'])
    pv = _uv2_reconstruct_part(U[:, ru:], uv['v'])
    out = fnp.empty((int(U.shape[0]), state['columns']), dtype=fnp.float32)
    hold = []
    b = 0
    for a, h in _uv2_blocks(state, 0, state['columns']):
        hold.append(fnp.add(pu[:, b:b+h], pv[:, b:b+h], out=out[:, a:a+h]))
        hold.append(fnp.subtract(pu[:, b:b+h], pv[:, b:b+h], out=out[:, a+h:a+2*h]))
        b += h
    _UV2_STATS['recon'] += 1
    return out


def _uv2_inner_part(left, part):
    group = part['lazy']
    if group is None:
        return _shared_old384_mm(left, part['F'].T)
    split = int(group['base'].shape[1])
    inner0 = _shared_old384_mm(left[:, :split], group['base'].T)
    inner1 = _shared_old384_mm(left[:, split:], group['join'].T)
    return _shared_old384_mm(inner0, group['map'].T) + inner1


def _cp_slr_inner_sq(old_left, lam_old, state):
    if 'uv2' not in state:
        return _UV2_PARENT_INNER(old_left, lam_old, state)
    rows = int(old_left.shape[0])
    pairs = state['columns'] // 2
    weighted = old_left * lam_old[None, :]
    su = fnp.empty((rows, pairs), dtype=fnp.float32)
    dv = fnp.empty((rows, pairs), dtype=fnp.float32)
    hold = []
    b = 0
    for a, h in _uv2_blocks(state, 0, state['columns']):
        ap, am = weighted[:, a:a+h], weighted[:, a+h:a+2*h]
        hold.append(fnp.add(ap, am, out=su[:, b:b+h]))
        hold.append(fnp.subtract(ap, am, out=dv[:, b:b+h]))
        b += h
    del weighted
    inner_u = _uv2_inner_part(su, state['uv2']['u'])
    inner_v = _uv2_inner_part(dv, state['uv2']['v'])
    _UV2_STATS['read'] += 1
    return fnp.concatenate([inner_u, inner_v], axis=1)


def _slr_inner(old_left, lam_old, state):
    return _cp_slr_inner_sq(old_left * old_left, lam_old, state)


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    joins = _JOIN_SCHEDULE.get(layer)
    if not state['columns'] and joins is None:
        return _UV2_PARENT_PUSH(layer, w32, p, previous, newborn, lam, state, rows=rows)
    old_columns = int(state['columns'])
    incoming_columns = old_columns
    if joins is not None:
        start = state['cohorts'][joins[0]][0]
        last_start, last_width = state['cohorts'][joins[1]]
        stop = last_start + last_width
        _slr_require(start == old_columns and stop > start, 'uv2 contiguous join')
        J = p[:, :stop-start]
        blocks = _uv2_blocks(state, start, stop)
        Ju, Jv, weight = _uv2_pair_parts(J, lam[start:stop], blocks)
        previous_uv = state.get('uv2')
        if previous_uv is None:
            oldu = oldv = Qpu = Qpv = None
        else:
            oldu, oldv = previous_uv['u'], previous_uv['v']
            ru = oldu['rank']
            Qpu = state['Q'][previous, :ru]
            Qpv = state['Q'][previous, ru:]
        Qu, partu = _uv2_join_part(Ju, weight, _UV2_RANK_U[layer], oldu, Qpu, layer)
        Qv, partv = _uv2_join_part(Jv, weight, _r320_rank(layer), oldv, Qpv, layer)
        active_basis = fnp.concatenate([Qu, Qv], axis=1)
        state['uv2'] = dict(u=partu, v=partv)
        state['columns'] = old_columns = stop
        state['rank'] = partu['rank'] + partv['rank']
        state['joins'].append(dict(layer=layer, cohort=joins[0], start=start, stop=stop,
                                   previous_active=int(previous.shape[0])))
        _UV2_STATS['joins'] += 1
        del Ju, Jv, weight, J, Qu, Qv, previous_uv, Qpu, Qpv, oldu, oldv
    else:
        active_basis = state['Q'][previous, :]
    wx = w32 if rows is None else w32[:, rows]
    U = _w5_port_call(_W5_PORT_SCOPE, w32, previous, 'shared_basis',
                      _shared_old384_mm, wx[previous, :].T, active_basis)
    old = _uv2_reconstruct(U, state)
    _shared_old384_finite(old, 'uv2 all-row bank')
    state['pre_basis'] = U
    inherited_columns, selected, base, t, s = newborn
    _slr_require(old_columns <= inherited_columns, 'uv2 newborn exclusion')
    young_newborn = (int(inherited_columns)-old_columns, selected, base, t, s)
    young = _push_bank_shared_newborn(wx, p[:, old_columns-incoming_columns:], previous, young_newborn)
    answer = fnp.concatenate([old, young], axis=1)
    _slr_require(int(answer.shape[1]) == int(lam.shape[0]), 'uv2 complete signed roster')
    return answer


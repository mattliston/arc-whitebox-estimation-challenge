# Exact operand loading through the public flat-scatter API.
# The existing quad has the original operand's C logical order. put flattens
# its input values by definition, so the separately paid source reshape is
# unnecessary. The quad owner, following add/sub and all tree layouts remain.
_O20_FLATLOAD_ON = True
_O20_FLATLOAD_PARENT_A = _rs_load_a
_O20_FLATLOAD_PARENT_B = _rs_load_b
_O20_FLATLOAD_INDEX = [None, 0, {}]
_O20_FLATLOAD_STATS = dict(a=0, b=0, avoided_reshape_elements=0,
                          index_elements_created=0, index_capacity=0)
_O20_FLATLOAD_CAP = 1 << 23


def _o20_flatload_indices(size):
    owner, capacity, views = _O20_FLATLOAD_INDEX
    if capacity < size:
        capacity = 1 << (size - 1).bit_length()
        owner = fnp.arange(capacity, dtype=fnp.int32)
        views = {}
        _O20_FLATLOAD_INDEX[:] = [owner, capacity, views]
        _O20_FLATLOAD_STATS['index_elements_created'] += capacity
        _O20_FLATLOAD_STATS['index_capacity'] = capacity
    answer = views.get(size)
    if answer is None:
        answer = owner if size == capacity else owner[:size]
        # Retain only a bounded number of zero-copy prefix views.
        if len(views) >= 256:
            views.clear()
        views[size] = answer
    return answer


def _rs_load_a(a, X, slot, L, m, k):
    size = int(m) * int(k)
    if (not _O20_FLATLOAD_ON or _ASTRA_QL_ON or size <= 0
            or size > _O20_FLATLOAD_CAP or a.dtype != 'float32'):
        return _O20_FLATLOAD_PARENT_A(a, X, slot, L, m, k)
    assert _rc_math.prod(a.shape) == size
    quad = slot[1] if slot[1] is not None else _l3_quad(slot, 0, L)
    assert _rc_math.prod(quad.shape) == size
    h = _BH_HOLD.append
    h(fnp.put(quad, _o20_flatload_indices(size), a))
    h(fnp.add(X[4], X[6], out=X[4]))
    _O20_FLATLOAD_STATS['a'] += 1
    _O20_FLATLOAD_STATS['avoided_reshape_elements'] += size


def _rs_load_b(b, Y, slot, L, k, n):
    size = int(k) * int(n)
    if (not _O20_FLATLOAD_ON or _ASTRA_QL_ON or size <= 0
            or size > _O20_FLATLOAD_CAP or b.dtype != 'float32'):
        return _O20_FLATLOAD_PARENT_B(b, Y, slot, L, k, n)
    assert _rc_math.prod(b.shape) == size
    quad = slot[1] if slot[1] is not None else _l3_quad(slot, 1, L)
    assert _rc_math.prod(quad.shape) == size
    h = _BH_HOLD.append
    h(fnp.put(quad, _o20_flatload_indices(size), b))
    h(fnp.subtract(Y[3], Y[1], out=Y[1]))
    _O20_FLATLOAD_STATS['b'] += 1
    _O20_FLATLOAD_STATS['avoided_reshape_elements'] += size

# Exact Smirnov input-circuit output ownership.
# L/R outputs write directly to the existing separate LA/RB owner. A node may
# also feed later groups; those consumers read its new destination. Nodes in
# a group have the same depth/operator and cannot depend on each other.
# Each scalar arithmetic operation and operand order remains unchanged.
_O20_SMDIRECT_ON = True
_O20_SMDIRECT_PARENT = _astra_sm40_linear_grouped
_O20_SMDIRECT_KEY_PARENT = _sm40_program_key
_O20_SMDIRECT_PLANS = {
    id(record['prepared'][kind]): record['prepared'][kind]
    for record in _SMX_RECORDS.values() for kind in ('L', 'R')
}
_O20_SMDIRECT_STATS = dict(calls=0, compiled=0, avoided_stack_elements=0,
                          old_arithmetic_calls=0, new_arithmetic_calls=0)


def _o20_smdirect_eligible(plan, out, prepared):
    return (_O20_SMDIRECT_ON and out is not None
            and isinstance(prepared, _Sm40Prepared)
            and _O20_SMDIRECT_PLANS.get(id(plan)) is plan)


def _sm40_program_key(plan, inputs, out, prepared):
    if _o20_smdirect_eligible(plan, out, prepared):
        return ('o20_direct_v1', id(plan), id(inputs), id(out))
    return _O20_SMDIRECT_KEY_PARENT(plan, inputs, out, prepared)


def _o20_smdirect_runs(positions):
    """Partition sorted plane indices into exact basic strided slices."""
    positions = sorted(positions)
    at = 0
    while at < len(positions):
        if at + 1 == len(positions):
            yield tuple(positions[at:at + 1])
            return
        end = at + 2
        step = positions[at + 1] - positions[at]
        while end < len(positions) and positions[end] - positions[end - 1] == step:
            end += 1
        yield tuple(positions[at:end])
        at = end


def _astra_sm40_linear_grouped(fnp, plan, inputs, shape, stats,
                                out=None, prepared=None):
    if not _o20_smdirect_eligible(plan, out, prepared):
        return _O20_SMDIRECT_PARENT(fnp, plan, inputs, shape, stats,
                                    out=out, prepared=prepared)
    key = _sm40_program_key(plan, inputs, out, prepared)
    program = prepared.programs.get(key)
    if program is None:
        input_refs, groups, output_refs, uses = plan
        batches, node_views = prepared
        output_at = {ref: i for i, ref in enumerate(output_refs)}
        assert len(output_at) == len(output_refs) == int(out.shape[0])
        produced = {ref for _, nodes in groups for ref, args in nodes}
        assert set(output_at) <= produced
        output_views = tuple(fnp.unstack(out, axis=0))
        values = {ref: inputs[i] for ref, i in input_refs.items()}
        operations = {'add': fnp.add, 'sub': fnp.subtract,
                      'scale': fnp.multiply, 'neg': fnp.multiply}
        ops = []
        node_at = 0
        peak = 0
        scalar_bytes = _rc_math.prod(shape) * 4
        for group_index, ((depth, op, lit), nodes) in enumerate(groups):
            node_map = dict(nodes)
            output_nodes = {output_at[ref]: ref for ref, args in nodes
                            if ref in output_at}
            internal_nodes = {i: ref for i, (ref, args) in enumerate(nodes)
                              if ref not in output_at}
            # No reference created by this group is available until all its
            # operations have been compiled. This asserts same-group DAG
            # independence rather than relying only on its depth label.
            assert all(arg in values for ref, args in nodes for arg in args)
            for is_output, mapping in ((False, internal_nodes), (True, output_nodes)):
                owner = out if is_output else batches[group_index]
                for positions in _o20_smdirect_runs(mapping):
                    refs = tuple(mapping[i] for i in positions)
                    left = [values[node_map[ref][0]] for ref in refs]
                    right = ([values[node_map[ref][1]] for ref in refs]
                             if op in ('add', 'sub') else None)
                    if len(positions) == 1:
                        i = positions[0]
                        destination = output_views[i] if is_output else node_views[node_at + i]
                        lhs = left[0]
                        rhs = right[0] if right else -1 if op == 'neg' else lit
                    else:
                        step = positions[1] - positions[0]
                        destination = owner[positions[0]:positions[-1] + 1:step]
                        lhs = left
                        rhs = right if right else -1 if op == 'neg' else lit
                    ops.append((operations[op], lhs, rhs, destination))
            for i, (ref, args) in enumerate(nodes):
                values[ref] = (output_views[output_at[ref]] if ref in output_at
                               else node_views[node_at + i])
            node_at += len(nodes)
            peak = max(peak, 2 * len(nodes) * scalar_bytes)
        program = (inputs, out, tuple(ops), output_views, peak)
        if len(prepared.programs) >= 16:
            prepared.programs.clear()
        # Existing setup precompiler intercepts this exact insertion before
        # any arithmetic. Its key helper above follows the same runtime flag.
        prepared.programs[key] = program
        _O20_SMDIRECT_STATS['compiled'] += 1
    _, _, ops, output_views, peak = program
    held = []
    for operation, lhs, rhs, destination in ops:
        held.append(operation(lhs, rhs, out=destination))
    stats['temp_peak'] = max(stats['temp_peak'], peak)
    stats['temp_live'] = 0
    _O20_SMDIRECT_STATS['calls'] += 1
    _O20_SMDIRECT_STATS['avoided_stack_elements'] += _rc_math.prod(out.shape)
    _O20_SMDIRECT_STATS['old_arithmetic_calls'] += len(plan[1])
    _O20_SMDIRECT_STATS['new_arithmetic_calls'] += len(ops)
    return out


# Append after tail_materialize_complete_v1.py. Exact coverage extension only.
# The owned-B route previously reached its captured old loader; when QL is off
# it can use the same flat loader without retaining any fresh tree owner.
_O20_FLATEXTRA_ON = True
_O20_FLATEXTRA_OWNED_PARENT = _rs_load_b_owned_ql_v1
_O20_FLATEXTRA_HZD_PARENT = _hzd_load
_O20_FLATEXTRA_STATS = dict(owned_b=0, hzd=0, hzd_avoided_reshape_elements=0)


def _rs_load_b_owned_ql_v1(b, Y, slot, L, k, n):
    if _O20_FLATEXTRA_ON and _O20_FLATLOAD_ON and not _ASTRA_QL_ON:
        _O20_FLATEXTRA_STATS['owned_b'] += 1
        return _rs_load_b(b, Y, slot, L, k, n)
    return _O20_FLATEXTRA_OWNED_PARENT(b, Y, slot, L, k, n)


def _hzd_load(quad, x, L, rm, cm, hr, hc):
    if (not _O20_FLATEXTRA_ON or not _O20_FLATLOAD_ON
            or x.dtype != 'float32' or rm <= 0 or cm <= 0
            or _rc_math.prod(x.shape) > _O20_FLATLOAD_CAP):
        return _O20_FLATEXTRA_HZD_PARENT(quad, x, L, rm, cm, hr, hc)
    main, rs, cs, cr, rz, cz = _hzd_parts(quad, L, rm, cm, hr, hc)
    h = _BH_HOLD.append
    u = 1 << L
    R, C = u * rm, u * cm
    xm = x[:R, :C]
    size = _rc_math.prod(xm.shape)
    h(xm)
    h(fnp.put(main, _o20_flatload_indices(size), xm))
    _O20_FLATEXTRA_STATS['hzd_avoided_reshape_elements'] += size
    if hr:
        xr = x[R:, :C]
        size = _rc_math.prod(xr.shape)
        h(xr)
        h(fnp.put(rs, _o20_flatload_indices(size), xr))
        _O20_FLATEXTRA_STATS['hzd_avoided_reshape_elements'] += size
    if hc:
        xc = x[:R, C:]
        size = _rc_math.prod(xc.shape)
        h(xc)
        h(fnp.put(cs, _o20_flatload_indices(size), xc))
        _O20_FLATEXTRA_STATS['hzd_avoided_reshape_elements'] += size
    if hr and hc:
        xx = x[R:, C:]
        size = _rc_math.prod(xx.shape)
        h(xx)
        h(fnp.put(cr, _o20_flatload_indices(size), xx))
        _O20_FLATEXTRA_STATS['hzd_avoided_reshape_elements'] += size
    z = _f32z()
    if hr:
        h(fnp.copyto(rz, z))
    if hc:
        h(fnp.copyto(cz, z))
    _O20_FLATEXTRA_STATS['hzd'] += 1


# [astra-bill] 2026-10-20T19:11:25Z: NP copy-only alternative.
# Original _npx_left polynomial and reduction expressions are byte-preserved.
# Do not combine with the K31 reassociation tail, which replaces the same fold.
assert not globals().get('_ASTRA_K31PAIR_ON', False)
_O20_NPX_COPY_EXACT_ON = True
_O20_NPX_COPY_EXACT_PARENT = _K3LB_PARENT_LEFT
_O20_NPX_COPY_EXACT_STATS = dict(calls=0, delta_blocks=0,
                               weighted_blocks=0, avoided_concat_elements=0)

def _o20_npx_copy_exact_left_v1(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov):
    """The read's left factor: the response rows of pp^2, with the pair law folded in when the layer has one. Stores diagB (scattered to
    the units, 0 outside the response rows) and the vectors of the forms for _np_forms and for the law."""
    if not _O20_NPX_COPY_EXACT_ON:
        return _O20_NPX_COPY_EXACT_PARENT(l, pp2, pp, response_indices,
            response_mask, alive, lam, mu, vc, vv, cov)
    S = _NP_STATE
    S['diag'] = S['pp2'] = S['fm'] = S['aa'] = S['ab'] = S['Ub'] = S['Rb'] = None
    if S['QX'] is None:
        return _unique_rows_via_scatter(pp2, response_indices)
    n, R = int(pp.shape[0]), int(pp.shape[1])
    C = _np_const(n)
    if l > C['last']:
        return _unique_rows_via_scatter(pp2, response_indices)
    sizes = S['sizes']
    if sum(sizes) != R or int(S['QX'].shape[1]) != R:
        raise RuntimeError('pair law: weight roster')
    hold = _E2Hold()
    q, xq = hold(fnp.unstack(S['QX']))
    rows = hold(_unique_rows_via_scatter(pp, response_indices))                               # the pushed columns on the response rows
    bounds, at = [], 0
    for width in sizes[:-1]:
        at += width
        bounds.append(at)
    blocks = hold(fnp.split(rows, bounds, axis=1))                                            # the roster's blocks, (+, -) of every cohort: views
    # Partners remain views until their original elementwise consumers.
    local = None
    ages = _np_ages()
    young, at, y0, y1 = [], 0, 0, 0
    for b, K in (S['births'] if ages else []):   # the roster's cohorts, oldest first: the run from the first to the last one with a strength
        strength = ages[l - b - 1] if 1 <= l - b <= len(ages) else 0.0
        if strength:
            if not young:
                y0 = at
            y1 = at + 2 * K
        if young or strength:
            young.append((b, K, strength))
        at += 2 * K
    while young and not young[-1][2]:
        young.pop()
    if young:   # the fourth-chaos plane of the young pairs' local quartic: a polynomial in (own entry, difference to the partner) per column
        W, rows_kept = y1 - y0, int(rows.shape[0])
        own = hold(rows[:, y0:y1])
        # Preserve own-minus-partner in the original cubic arithmetic.
        # All boundaries are the already declared paired cohort roster.
        delta = hold(fnp.empty(own.shape, dtype=own.dtype))
        young_indices, delta_cuts, column = [], [], 0
        for block_index, width in enumerate(sizes):
            if y0 <= column < y1:
                young_indices.append(block_index)
                delta_cuts.append(column + width - y0)
            column += width
        assert young_indices and delta_cuts[-1] == y1 - y0
        assert sum(sizes[i] for i in young_indices) == y1 - y0
        delta_parts = hold(fnp.split(delta, delta_cuts[:-1], axis=1))
        for block_index, destination in zip(young_indices, delta_parts):
            hold(fnp.subtract(blocks[block_index], blocks[block_index ^ 1],
                              out=destination))
        _O20_NPX_COPY_EXACT_STATS['delta_blocks'] += len(young_indices)
        if _NP_K3L_ALL:   # the pair's means over all units, from the full columns: sums of o^2 (the squares are there), of the partner's, of o x partner
            inner, at = [], 0
            for _b, K, _s in young:
                inner += [at + K, at + 2 * K]
                at += 2 * K
            full = hold(pp[:, y0:y1])
            pieces = hold(fnp.split(full, inner[:-1], axis=1))
            squares = hold(fnp.sum(hold(pp2[:, y0:y1]), axis=0))
            halves = hold(fnp.split(squares, inner[:-1]))
            pair = [hold(fnp.einsum('ik,ik->k', pieces[i], pieces[i + 1], optimize=False)) for i in range(0, len(pieces), 2)]   # scout-native NPX: sum_i p_i q_i once per pair
            cross = hold(fnp.concatenate([pair[i // 2] for i in range(len(pieces))]))           # the same number for the pair's plus and minus column
            means = hold(C['Cmean'] @ hold(fnp.stack([squares, hold(fnp.concatenate([halves[i ^ 1] for i in range(len(halves))])), cross])))
            dd = hold(delta * delta)
        else:             # over the response rows
            dd = hold(delta * delta)
            means = hold(hold(fnp.stack([hold(fnp.sum(hold(own * own), axis=0)), hold(fnp.sum(hold(own * delta), axis=0)), hold(fnp.sum(dd, axis=0))])) * (1.0 / rows_kept))
        lam_young = hold(lam[None, y0:y1])
        quartic = [(S['k4e'][b] if strength == 1.0 else hold(S['k4e'][b] * strength)) for b, _K, strength in young]   # the plane is linear in e1..e4
        quartic = quartic[0] if len(quartic) == 1 else hold(fnp.concatenate(quartic, axis=1))
        # one and 1 / lam (lam^0, lam^-1), the plus flag, e1..e4, the three means (float32)
        table = hold(fnp.concatenate([hold(fnp.power(lam_young, C['p01'])), hold(fnp.greater(lam_young, 0.0)), quartic, means]))
        c3, c2, c1, c0, c_o, c_d = hold(fnp.unstack(hold(C['coefK'] @ hold(fnp.prod(hold(table[C['idxK']]), axis=0)))))
        # o (o (c3 o + c2 Delta) + c1 Delta^2 + c_o) + Delta (c0 Delta^2 + c_d), per column
        value = hold(own * c3)
        other = hold(delta * c2)
        hold(fnp.add(value, other, out=value)); hold(fnp.multiply(value, own, out=value))
        hold(fnp.multiply(dd, c1, out=other))
        hold(fnp.add(value, other, out=value)); hold(fnp.add(value, c_o, out=value)); hold(fnp.multiply(value, own, out=value))
        hold(fnp.multiply(dd, c0, out=other))
        hold(fnp.add(other, c_d, out=other)); hold(fnp.multiply(other, delta, out=other))
        hold(fnp.add(value, other, out=value))
        local = (y0, y1, value)                                                               # (k31 / 3) x (the plane's left factor) / lam, per column
    swapped = hold(fnp.empty(rows.shape, dtype=rows.dtype))
    swapped_parts = hold(fnp.split(swapped, bounds, axis=1))
    xq_parts = hold(fnp.split(xq, bounds))
    for block_index, (destination, weight) in enumerate(zip(swapped_parts, xq_parts)):
        hold(fnp.multiply(blocks[block_index ^ 1], weight, out=destination))
    _O20_NPX_COPY_EXACT_STATS['calls'] += 1
    _O20_NPX_COPY_EXACT_STATS['weighted_blocks'] += len(blocks)
    _O20_NPX_COPY_EXACT_STATS['avoided_concat_elements'] += int(rows.shape[0]) * R
    r2 = hold(rows * q)
    hold(fnp.add(r2, swapped, out=r2))                                                        # kap (q_r pp_ir + xq_r P_ir)
    diagonal = fnp.zeros(n, dtype=fnp.float32)
    fnp.put(diagonal, response_indices, hold(hold(fnp.multiply(rows, r2, out=swapped)) @ lam))   # kap diagB, 0 outside the response rows
    S['diag'] = diagonal
    if _NP_DIAG:
        S['pp2'] = pp2                                                                        # for (pp o pp) c after the pair map
    if local is not None:   # after B's diagonal: the local quartic's plane is no part of B
        r2_young = hold(r2[:, local[0]:local[1]])
        hold(fnp.add(r2_young, local[2], out=r2_young))                                       # the read's plane gets - f_i (k31 / 3) K31L_ij, as the reference's fold
    Y = hold(fnp.prod(hold(hold(fnp.stack([C['one'], C['two'], response_mask, alive, vv, mu, hold(cov @ mu), diagonal, hold(mu / vc)]))[C['idx_reg']]), axis=0))
    solution = hold(fnp.linalg.solve(hold(hold(Y @ hold(Y.T)) * C['Mreg'][l]), C['unit']))
    solution = hold(solution / hold(fnp.unstack(solution))[C['target']])                      # kap x (multiple x c0, .., c1, c2), 1, 0, ..
    _hc_column(l, Y, C['target'], vc, response_mask, int(response_indices.shape[0]))   # opus-engineer hc: the head's new input column from this layer's diagonal
    vectors = hold(hold(hold(solution[C['idxM']]) @ Y).astype(fnp.float32))                   # f / kap, kap aB alive, kap aB, then the mean forms'
    nf = C['nf']
    parts = hold(fnp.split(vectors, [3, 3 + nf])) if nf else [vectors]
    fm, S['aa'], S['ab'] = hold(fnp.unstack(parts[0]))
    left2 = rows * rows
    if not _NP_STRENGTH[l]:
        return left2
    S['fm'] = fm                                                                              # mu_i / v_i on the response rows (f_i = kap x this)
    if nf:
        S['Rb'], S['Ub'] = parts[1], parts[2]
    hold(fnp.multiply(r2, hold(hold(fm[response_indices])[:, None]), out=r2))                # f_i R2_ir, in place
    hold(fnp.subtract(left2, r2, out=left2))
    _NP_STATS['folds'] += 1
    return left2

_K3LB_PARENT_LEFT = _o20_npx_copy_exact_left_v1

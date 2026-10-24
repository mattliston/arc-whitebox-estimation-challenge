

# Direct K31 coefficient64 vectors, shared cubic32; 2026-10-21T01:54:27Z.
assert _NP_K3L_ALL and _O20_NPX_COPY_EXACT_ON
assert _K3LB_PARENT_LEFT is _o20_npx_copy_exact_left_v1
assert _GNN_PARAMS is None and _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}
_O21_K31C_STATS = dict(calls=0,row_pairs=0,pairs=0)

"""Same direct common coefficients, with small matrix calls to limit requests."""
_O21_K31_COEFF64_CONST = None
def _o21_k31_common_matrix64(lam_plus, quartic, square_plus, square_minus, cross, n, hold):
    global _O21_K31_COEFF64_CONST
    if _O21_K31_COEFF64_CONST is None:
        base = [[-1/12,0,0,0],[.5,1/6,0,0],[-.75,-.5,-.25,0],[1/3,1/3,1/3,1/3],
                [1/6,0,0,0],[-.5,0,0,0],[.75,.5,.5,0],[-1/3,-1/3,-1/3,0]]
        means = [[1,0,0],[1,0,-1],[1,1,-2]]
        lower = [[0,-6,0,0,0,-1,0,0,0,0,0,0],[-3,0,0,0,-2,0,0,0,-1,0,0,0],
                 [0,0,0,-1,0,0,0,-2,0,0,0,-3]]
        _O21_K31_COEFF64_CONST = tuple(fnp.asarray(x,dtype=fnp.float64)for x in(base,means,lower))
    basis, meanmap, lowermap = _O21_K31_COEFF64_CONST
    inv = hold(1.0 / hold(lam_plus.astype(fnp.float64)))
    g = hold(hold(lam_plus > 0.0).astype(fnp.float64))
    both = hold(basis @ hold(quartic.astype(fnp.float64)))
    coeff = hold(hold(hold(both[:4] + hold(both[4:] * g)) * inv))
    raw = hold(hold(fnp.stack((square_plus,square_minus,cross))).astype(fnp.float64))
    means = hold(hold(meanmap @ raw) * (1.0 / n))
    products = hold(hold(coeff[:,None,:] * means[None,:,:]).reshape((12,int(lam_plus.shape[0]))))
    lower = hold(lowermap @ products)
    return hold(hold(fnp.concatenate((coeff,lower),axis=0)).astype(fnp.float32))


def _o21_k31_common_polynomial32(own, coefficients, young, hold):
    cuts, at = [], 0
    for _born, width, _strength in young:
        cuts += [at + width, at + 2 * width]
        at += 2 * width
    own_parts = hold(fnp.split(own, cuts[:-1], axis=1))
    pair_cuts = []
    pair_at = 0
    for _born, width, _strength in young:
        pair_at += width
        pair_cuts.append(pair_at)
    coeff_parts = hold(fnp.split(coefficients, pair_cuts[:-1], axis=1))
    value = hold(fnp.empty(own.shape, dtype=own.dtype))
    out_parts = hold(fnp.split(value, cuts[:-1], axis=1))
    for j, (_born, width, _strength) in enumerate(young):
        p, q = own_parts[2 * j], own_parts[2 * j + 1]
        plus, minus = out_parts[2 * j], out_parts[2 * j + 1]
        aa, bb, cc, ddcoef, ee, ff, gg = hold(fnp.unstack(coeff_parts[j], axis=0))

        delta = hold(fnp.subtract(p, q))
        square = hold(fnp.multiply(delta, delta))
        a = hold(fnp.multiply(p, aa))
        b = hold(fnp.multiply(delta, bb))
        c = hold(fnp.multiply(square, cc))
        common = hold(fnp.add(a, b))
        hold(fnp.multiply(common, p, out=common))
        hold(fnp.add(common, ff, out=common))

        # minus = p * (p*(A*p+B*D)+F+C*D^2) + D*(Dcoef*D^2+G).
        tmp = hold(fnp.add(common, c))
        hold(fnp.multiply(tmp, p, out=minus))
        tail = hold(fnp.multiply(square, ddcoef))
        hold(fnp.add(tail, gg, out=tail))
        hold(fnp.multiply(tail, delta, out=tail))
        hold(fnp.add(minus, tail, out=minus))

        # plus-minus = D*(3*A*p^2+B*p*D+F+(C/3)*D^2) + E*p.
        hold(fnp.multiply(p, a, out=tmp))
        hold(fnp.multiply(tmp, 2.0, out=tmp))
        hold(fnp.add(tmp, common, out=tmp))
        hold(fnp.multiply(c, 1.0 / 3.0, out=c))
        hold(fnp.add(tmp, c, out=tmp))
        hold(fnp.multiply(tmp, delta, out=tmp))
        hold(fnp.multiply(p, ee, out=tail))
        hold(fnp.add(tmp, tail, out=tmp))
        hold(fnp.add(minus, tmp, out=plus))
    return value


def _o21_k31c_left_v1(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov):
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
        assert _NP_K3L_ALL
        own = hold(rows[:, y0:y1])
        cuts, pair_cuts, at, pair_at = [], [], 0, 0
        for born, width, strength in young:
            cuts += [at + width, at + 2 * width]
            at += 2 * width
            pair_at += width
            pair_cuts.append(pair_at)
        full = hold(pp[:, y0:y1])
        pieces = hold(fnp.split(full, cuts[:-1], axis=1))
        squares = hold(fnp.sum(hold(pp2[:, y0:y1]), axis=0))
        halves = hold(fnp.split(squares, cuts[:-1]))
        pair = [hold(fnp.einsum('ik,ik->k', pieces[i], pieces[i + 1], optimize=False)) for i in range(0, len(pieces), 2)]
        lp = hold(fnp.split(hold(lam[y0:y1]), cuts[:-1]))
        quartic = [(hold(S['k4e'][born][:, :width]) if strength == 1.0 else hold(hold(S['k4e'][born][:, :width]) * strength)) for born, width, strength in young]
        def joined(parts, axis=0):
            return parts[0] if len(parts) == 1 else hold(fnp.concatenate(parts, axis=axis))
        common = hold(_o21_k31_common_matrix64(joined(lp[::2]), joined(quartic, 1), joined(halves[::2]), joined(halves[1::2]), joined(pair), n, hold))
        value = hold(_o21_k31_common_polynomial32(own, common, young, hold))
        _O21_K31C_STATS['calls'] += 1
        _O21_K31C_STATS['row_pairs'] += int(rows.shape[0]) * pair_at
        _O21_K31C_STATS['pairs'] += pair_at
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

_K3LB_PARENT_LEFT = _o21_k31c_left_v1
_FAST_RESCUE_FRAC = {}

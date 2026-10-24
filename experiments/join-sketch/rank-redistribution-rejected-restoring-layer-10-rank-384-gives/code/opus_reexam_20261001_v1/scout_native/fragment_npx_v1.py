

# BEGIN SCOUT-NATIVE NPX v1 (2026-10-12; byte-identical predictions expected, certified by runs): in the pair law's fourth-chaos fold
# (_np_left with _NP_K3L_ALL), the cross sum of a pair's two columns over all units, sum_i p_i q_i, is the same number for the pair's plus
# and minus column (the same products, summed in the same order). The pair law v9 forms it per COLUMN: a copy of the young run with every
# pair's two columns exchanged (n x W, billed n W) and one einsum over n x W (billed 2 n W). Here it is formed per PAIR: one einsum per
# cohort on the cohort's plus and minus blocks (views, billed n W in all), each value used for both columns (one small concatenate):
# -2 n W per read layer, one request more per cohort beyond the first. The pair law's _np_left (v9, the first definition in the file) is
# copied with only that change and installed under the K3L-by-layer wrapper (_K3LB_PARENT_LEFT), so the later wrappers stay as they are.

def _npx_left(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov):
    """The read's left factor: the response rows of pp^2, with the pair law folded in when the layer has one. Stores diagB (scattered to
    the units, 0 outside the response rows) and the vectors of the forms for _np_forms and for the law."""
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
    swapped = hold(fnp.concatenate([blocks[i ^ 1] for i in range(len(blocks))], axis=1))      # every pair's two columns exchanged (block copies)
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
        delta = hold(own - hold(swapped[:, y0:y1]))                                            # Delta = own - partner
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
    hold(fnp.multiply(swapped, xq, out=swapped))
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


_K3LB_PARENT_LEFT = _npx_left
# END SCOUT-NATIVE NPX v1

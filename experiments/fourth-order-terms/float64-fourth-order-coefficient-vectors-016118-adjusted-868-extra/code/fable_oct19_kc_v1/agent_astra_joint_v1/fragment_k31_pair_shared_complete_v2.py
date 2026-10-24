# Experimental K31L4 pair polynomial reuse. Appended after the current physics.
# p and q are the +/- columns, with equal and opposite lambda. D = p - q.
# Q=e1*p^3*D+e2*p^2*D^2+e3*p*D^3+e4*D^4, and H is its second-chaos part.
# The original signed-column left factors are:
#   minus = Q_D/(12*lambda_plus) - H_D/(24*lambda_plus)
#   plus  = minus + Q_p/(12*lambda_plus) - H_p/(24*lambda_plus).
# Both cubics can share their intermediates. Existing coefficient tables and
# fitted strengths are retained; only their evaluation order changes.
# No estimator prediction has been run for this fragment.

_ASTRA_K31PAIR_ON = True
_ASTRA_K31PAIR_STATS = dict(calls=0, pairs=0, row_pairs=0, ideal_poly_saved=0)


def _astra_k31pair_polynomial(own, coefficients, young, hold):
    cuts, at = [], 0
    for _born, width, _strength in young:
        cuts += [at + width, at + 2 * width]
        at += 2 * width
    own_parts = hold(fnp.split(own, cuts[:-1], axis=1))
    coeff_parts = hold(fnp.split(coefficients, cuts[:-1], axis=1))
    value = hold(fnp.empty(own.shape, dtype=own.dtype))
    out_parts = hold(fnp.split(value, cuts[:-1], axis=1))
    for j, (_born, width, _strength) in enumerate(young):
        p, q = own_parts[2 * j], own_parts[2 * j + 1]
        plus, minus = out_parts[2 * j], out_parts[2 * j + 1]
        aa, c2, c1, c0, co, cd = hold(fnp.unstack(coeff_parts[2 * j], axis=0))
        ff = hold(coeff_parts[2 * j + 1][4])
        # Coefficients of the common (p,D) polynomials. All operations here
        # are O(cohort width), not O(response rows * cohort width).
        bb = hold(fnp.subtract(c2, hold(fnp.multiply(aa, 3.0))))
        cc = hold(fnp.subtract(c1, bb))
        ddcoef = hold(fnp.subtract(c0, hold(fnp.multiply(cc, 1.0 / 3.0))))
        ee = hold(fnp.subtract(co, ff))
        gg = hold(fnp.subtract(cd, ff))

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
        row_pairs = int(p.shape[0]) * int(width)
        _ASTRA_K31PAIR_STATS['pairs'] += int(width)
        _ASTRA_K31PAIR_STATS['row_pairs'] += row_pairs
        # Parent: 2 subtract + 2 square + 24 Horner operations per pair/row.
        # This form: 23 per pair/row; seven O(width) coefficient operations.
        _ASTRA_K31PAIR_STATS['ideal_poly_saved'] += 5 * row_pairs - 7 * int(width)
    _ASTRA_K31PAIR_STATS['calls'] += 1
    return value

_ASTRA_K31PAIR_PARENT_LEFT = _K3LB_PARENT_LEFT

def _astra_k31pair_left_v2(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov):
    """The read's left factor: the response rows of pp^2, with the pair law folded in when the layer has one. Stores diagB (scattered to
    the units, 0 outside the response rows) and the vectors of the forms for _np_forms and for the law."""
    if not _ASTRA_K31PAIR_ON or not _NP_K3L_ALL:
        return _ASTRA_K31PAIR_PARENT_LEFT(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov)
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
        inner, at = [], 0
        for _b, K, _s in young:
            inner += [at + K, at + 2 * K]
            at += 2 * K
        full = hold(pp[:, y0:y1])
        pieces = hold(fnp.split(full, inner[:-1], axis=1))
        squares = hold(fnp.sum(hold(pp2[:, y0:y1]), axis=0))
        halves = hold(fnp.split(squares, inner[:-1]))
        pair = [hold(fnp.einsum('ik,ik->k', pieces[i], pieces[i + 1], optimize=False)) for i in range(0, len(pieces), 2)]
        cross = hold(fnp.concatenate([pair[i // 2] for i in range(len(pieces))]))
        means = hold(C['Cmean'] @ hold(fnp.stack([squares, hold(fnp.concatenate([halves[i ^ 1] for i in range(len(halves))])), cross])))
        lam_young = hold(lam[None, y0:y1])
        quartic = [(S['k4e'][b] if strength == 1.0 else hold(S['k4e'][b] * strength)) for b, _K, strength in young]   # the plane is linear in e1..e4
        quartic = quartic[0] if len(quartic) == 1 else hold(fnp.concatenate(quartic, axis=1))
        # one and 1 / lam (lam^0, lam^-1), the plus flag, e1..e4, the three means (float32)
        table = hold(fnp.concatenate([hold(fnp.power(lam_young, C['p01'])), hold(fnp.greater(lam_young, 0.0)), quartic, means]))
        coefficients = hold(C['coefK'] @ hold(fnp.prod(hold(table[C['idxK']]), axis=0)))
        value = _astra_k31pair_polynomial(own, coefficients, young, hold)
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

_K3LB_PARENT_LEFT = _astra_k31pair_left_v2

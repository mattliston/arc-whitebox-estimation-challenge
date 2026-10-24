# BEGIN OPUS-ENGINEER FINAL-LAYER HEAD COLUMNS (a42 hcf v3, 2026-10-10; a change of the final head's INPUT, OFF unless weights are given).
# v3 = v2 with column 18's cohorts pinned to the trainer's definition (10-10 03:3xZ): the young pairs born at layers 13 and 14 (ages 1-2)
# at strength 1 (_HCF_AGES), whatever strengths the file's own plane uses (v2 took the plane's ages and strengths). v2 text follows.
# The final layer's learned correction is gnn_step at layer 15 (_predict_learned_compact's scalar_terminal_hook: the same enc0 as
# layers 1-14, with layer 15's centres, scales and head); its first layer gets
#     enc0_b (+ the head columns of fragment_hc_v2, none at layer 15) + sum_k z_k u_k',
#     z_k = clip((x_k - mean) / sd, -8, 8) over the KEPT ROWS of the final layer's row cut (|mu / sigma| <= _L15_ROWCUT_Z: the rows the
#     bank is pushed on into layer 15), 0 elsewhere; u_k = 128 weights of the final layer (the trainer's columns 17-19, O/trainer/otr_m.py).
#   'f15_diag'       (17) the law matrix's diagonal at layer 15, B_ii = sum_r pp_ir (w_r pp_ir + x_r pp_i,partner(r)) on the kept rows with
#                    the law's weights advanced through layer 14 (the fold's update at layer 14, skipped without this column), the
#                    residual of its least-squares fit on (v, 1, mu^2) over the kept rows, over v (any constant factor drops out)
#   'f15_k31l_diag'  (18) the diagonal of the young pairs' fourth-chaos plane at layer 15 on the kept rows, its means over the kept rows
#   'f15_k4_local'   (19) K4L's per-unit value at layer 15 over v^2 (the k4_15 term the file already computes)
# Made in a wrapper of _k4l_local (called by _l15_rowcut_k3 with the kept rows' pushed columns; the roster and lam come from a wrapper
# of _l15_rowcut_k3), finished in a wrapper of _k4l_final (mu, v, the kept rows), added in a wrapper of _hc_bias at l = 15.
_HCF_COLUMNS = ()        # the columns, in the order of the rows of _HCF_U
_HCF_U = None            # K x 128 weights (a tuple of K tuples of float32 values); None = no column
_HCF_STATE = dict(raw={}, ctx=None, rows15=None)
_HCF_CONST = {}
_HCF_STATS = dict(columns=0, used=0)
_HCF_AGES = (1.0, 1.0)   # column 18: the strengths of the cohorts aged 1, 2, .. (born at 14, 13, ..)


def _hcf_put(name, vector, mask, count):
    """A final-layer column's per-unit vector (float64), its unit mask (float64 0 / 1) and the number of its units."""
    if _HCF_U is None or name not in _HCF_COLUMNS:
        return
    _HCF_STATE['raw'][name] = (vector, mask, int(count))
    _HCF_STATS['columns'] += 1


_HCF_PARENT_L15 = _l15_rowcut_k3


def _l15_rowcut_k3(layer, mu, v, w32, p, previous, newborn, lam, state, star):
    _HCF_STATE['ctx'] = (layer, lam)
    try:
        return _HCF_PARENT_L15(layer, mu, v, w32, p, previous, newborn, lam, state, star)
    finally:
        _HCF_STATE['ctx'] = None


def _hcf_rows15(pp_rows, lam):
    """Columns 17 and 18 on the kept rows (vectors of the kept rows' length), from the pushed columns on those rows."""
    hold = _E2Hold()
    S = _NP_STATE
    out = {}
    want17, want18 = 'f15_diag' in _HCF_COLUMNS, 'f15_k31l_diag' in _HCF_COLUMNS
    if S['QX'] is None or not (want17 or want18):
        return out
    R = int(pp_rows.shape[1])
    sizes = S['sizes']
    if sum(sizes) != R or int(S['QX'].shape[1]) != R:
        raise RuntimeError('final-layer columns: weight roster')
    n = int(_HCF_CONST.get('n15') or 1024)
    C = _np_const(n)
    bounds, at = [], 0
    for width in sizes[:-1]:
        at += width
        bounds.append(at)
    blocks = hold(fnp.split(pp_rows, bounds, axis=1))
    swapped = hold(fnp.concatenate([blocks[i ^ 1] for i in range(len(blocks))], axis=1))           # every pair's two columns exchanged
    if want17:
        q, xq = hold(fnp.unstack(S['QX']))
        r2 = hold(pp_rows * q)
        hold(fnp.add(r2, hold(swapped * xq), out=r2))                                                # kap (q_r pp_ir + xq_r P_ir)
        out['f15_diag'] = hold(hold(hold(pp_rows * r2) @ lam).astype(fnp.float64))                     # kap B_ii on the kept rows
    if want18:
        ages = _HCF_AGES
        young, at, y0, y1 = [], 0, 0, 0
        for b, K in (S['births'] if ages else []):
            strength = ages[15 - b - 1] if 1 <= 15 - b <= len(ages) else 0.0
            if strength:
                if not young:
                    y0 = at
                y1 = at + 2 * K
            if young or strength:
                young.append((b, K, strength))
            at += 2 * K
        while young and not young[-1][2]:
            young.pop()
        if young:
            kept = int(pp_rows.shape[0])
            own = hold(pp_rows[:, y0:y1])
            delta = hold(own - hold(swapped[:, y0:y1]))
            dd = hold(delta * delta)
            means = hold(hold(fnp.stack([hold(fnp.sum(hold(own * own), axis=0)), hold(fnp.sum(hold(own * delta), axis=0)), hold(fnp.sum(dd, axis=0))])) * (1.0 / kept))
            lam_young = hold(lam[None, y0:y1])
            quartic = [(S['k4e'][b] if strength == 1.0 else hold(S['k4e'][b] * strength)) for b, _K, strength in young]
            quartic = quartic[0] if len(quartic) == 1 else hold(fnp.concatenate(quartic, axis=1))
            table = hold(fnp.concatenate([hold(fnp.power(lam_young, C['p01'])), hold(fnp.greater(lam_young, 0.0)), quartic, means]))
            c3, c2, c1, c0, c_o, c_d = hold(fnp.unstack(hold(C['coefK'] @ hold(fnp.prod(hold(table[C['idxK']]), axis=0)))))
            value = hold(own * c3)
            other = hold(delta * c2)
            hold(fnp.add(value, other, out=value)); hold(fnp.multiply(value, own, out=value))
            hold(fnp.multiply(dd, c1, out=other))
            hold(fnp.add(value, other, out=value)); hold(fnp.add(value, c_o, out=value)); hold(fnp.multiply(value, own, out=value))
            hold(fnp.multiply(dd, c0, out=other))
            hold(fnp.add(other, c_d, out=other)); hold(fnp.multiply(other, delta, out=other))
            hold(fnp.add(value, other, out=value))                                                   # (k31 / 3) x the plane's left factor / lam
            out['f15_k31l_diag'] = hold(hold(hold(value * own) @ hold(lam[y0:y1])).astype(fnp.float64))
    return out


_HCF_PARENT_K4L_LOCAL = _k4l_local


def _k4l_local(layer, pp, pp2, t3, state):
    if _HCF_U is not None and layer == 15 and _HCF_STATE['ctx'] is not None:
        _HCF_STATE['rows15'] = _hcf_rows15(pp, _HCF_STATE['ctx'][1])
    return _HCF_PARENT_K4L_LOCAL(layer, pp, pp2, t3, state)


_HCF_PARENT_K4L_FINAL = _k4l_final


def _k4l_final(mu, v):
    found = _K4L_FINAL[0]
    dk = _HCF_PARENT_K4L_FINAL(mu, v)
    found_rows15, _HCF_STATE['rows15'] = _HCF_STATE['rows15'], None
    if _HCF_U is None or found is None or found[1] is None:
        return dk
    hold = _E2Hold()
    rows = found[0]
    n, kept = int(v.shape[0]), int(rows.shape[0])
    mask = fnp.zeros(n, dtype=fnp.float64)
    fnp.put(mask, rows, 1.0)
    if 'f15_k4_local' in _HCF_COLUMNS and not isinstance(dk, float):
        _hcf_put('f15_k4_local', hold(dk / hold(v * v)), mask, kept)
    got = found_rows15 or {}
    if 'f15_diag' in got and kept >= 16:            # residual of the fit on (v, 1, mu^2) over the kept rows: G^-1 e_target times the rows
        K = _HCF_CONST
        if 'unit4' not in K:
            K['unit4'] = fnp.asarray([0.0, 0.0, 0.0, 1.0])
        vr, mr = hold(v[rows]), hold(mu[rows])
        X = hold(fnp.stack([vr, hold(fnp.ones(kept, dtype=fnp.float64)), hold(mr * mr), got['f15_diag']]))
        weights = hold(fnp.linalg.solve(hold(X @ hold(X.T)), K['unit4']))
        full = fnp.zeros(n, dtype=fnp.float64)
        fnp.put(full, rows, hold(hold(weights @ X) / vr))
        _hcf_put('f15_diag', hold(full), mask, kept)
    if 'f15_k31l_diag' in got:
        full = fnp.zeros(n, dtype=fnp.float64)
        fnp.put(full, rows, hold(got['f15_k31l_diag'] / hold(hold(v[rows]) * hold(v[rows]))))
        _hcf_put('f15_k31l_diag', hold(full), mask, kept)
    return dk


def _hcf_bias(bias):
    """enc0_b, or enc0_b + sum_k z_k u_k' with the final layer's columns (a column the network did not make counts as 0)."""
    present = dict(_HCF_STATE['raw'])
    _HCF_STATE['raw'] = {}
    if _HCF_U is None or not present:
        return bias
    hold = _E2Hold()
    K = _HCF_CONST
    n = int(next(iter(present.values()))[0].shape[0])
    if K.get('n') != n:
        K.update(n=n, u=fnp.asarray(_HCF_U, dtype=fnp.float32), zeros=fnp.zeros(n, dtype=fnp.float64))
    raw = [present.get(name, (K['zeros'], K['zeros'], n)) for name in _HCF_COLUMNS]
    M = hold(fnp.stack([mask for _, mask, _ in raw]))
    X = hold(hold(fnp.stack([hold(x) for x, _, _ in raw])) * M)
    inverse = hold(fnp.asarray([[1.0 / max(count, 1)] for _, _, count in raw]))
    root = hold(fnp.asarray([[max(count - 1.0, 0.0) ** 0.5] for _, _, count in raw]))
    centred = hold(hold(X - hold(hold(fnp.sum(X, axis=1, keepdims=True)) * inverse)) * M)
    scale = hold(hold(fnp.power(hold(fnp.maximum(hold(fnp.sum(hold(centred * centred), axis=1, keepdims=True)), 1e-300)), -0.5)) * root)
    Z = hold(hold(fnp.clip(hold(centred * scale), -8.0, 8.0)).astype(fnp.float32))
    _HCF_STATS['used'] += 1
    return bias + hold(fnp.einsum('kn,kc->nc', Z, K['u'], optimize=False))


_HCF_PARENT_HC_BIAS = _hc_bias


def _hc_bias(l, bias):
    out = _HCF_PARENT_HC_BIAS(l, bias)
    return _hcf_bias(out) if l == 15 else out
# END OPUS-ENGINEER FINAL-LAYER HEAD COLUMNS

# BEGIN OPUS-ENGINEER HEAD COLUMNS (a42 hc v2, 2026-10-02; a change of the learned mean correction's INPUT, OFF unless weights are
# given). v2 = v1 for SEVERAL columns: every column is a per-unit vector that some place of the layer leaves in _HC_STATE (the fold
# leaves the law diagonal's residual, as v1; the loop leaves the others through _hc_put), and the first layer of the correction gets
#     enc0_b + sum_k z_k u_k',     z_k = clip((x_k - mean) / sd, -8, 8) over the column's own units (0 elsewhere), u_k = 128 weights.
# With ONE column ('diag') the arithmetic is v1's, operation for operation. With K columns the standardisation runs once on the
# stacked rows (the same operations per row) and the bias term is one product Z' U: about 18 requests a layer for any K, plus what
# each column's own vector costs.
# A column that a layer does not put counts as 0 for that layer (the trainer's rule).
#   'diag'            the trainer's column 6: (B_ii - fit_i) / (tr C v_i) on the response rows of the layer's bank read (see v1)
#   'k31l_diag_rows'  column 9: the diagonal of the young pairs' fourth-chaos plane over v^2, response rows (_hc_pdiag, in the fold)
#   'k3_young', 'k3_old'   columns 12, 13: the bank's per-unit third cumulant from the young columns / from the old tier, over v sigma
#   'k3_star', 'k3_bform', 'k4_local'   columns 14-16: the star's diagonal over v sigma, 3 b / sigma, K4L's per-unit value over v^2
#   'k4_local_rows'   column 16 on the response rows only (natively K4L's value is exactly 0 on the rows the push drops, which are
#                     never response rows: on the rows both sides hold the same numbers)
# (names of O/trainer/otr_m.py HEAD_COLUMNS; constant factors are left out, the standardisation removes them).
_HC_COLUMNS = ('diag',)  # the columns, in the order of the rows of _HC_U
_HC_U = None             # K x 128 weights (a tuple of K tuples of float32 values); None = no column
_HC_STATE = dict(l=None, raw={})
_HC_CONST = {}
_HC_STATS = dict(columns=0, used=0)


def _hc_put(l, name, vector, mask=None, count=None):
    """A column's per-unit vector for layer l (float64; 0 outside `mask`, which is None for all units; count = number of its units)."""
    if _HC_U is None or name not in _HC_COLUMNS or not (1 <= l <= 14):
        return
    S = _HC_STATE
    if S['l'] != l:
        S['l'], S['raw'] = l, {}
    S['raw'][name] = (vector, mask, int(vector.shape[0]) if count is None else int(count))


def _hc_column(l, Y, target, vc, response_mask, nrows):
    """Called by the fold after its regression rows are made: Y rows 0..target = the masked regressors and, at `target`, the masked
    diagonal. Leaves the residual over v (times a positive number) as the column 'diag' of layer l."""
    if _HC_U is None or 'diag' not in _HC_COLUMNS or not (1 <= l <= 14) or nrows < 16:
        return
    hold = _E2Hold()   # temporaries of this call, released together
    K = _HC_CONST
    if 'unit' not in K:
        K['unit'] = {4: fnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0]), 3: fnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0])}
    if target == 4:      # the fold's regression has the constant: rows (v, 1, mu^2, 2 mu C mu, diagonal)
        X = hold(Y[:5])
    else:                # the table regression, rows (v, mu^2, 2 mu C mu, diagonal): the constant is the fifth row
        X = hold(fnp.concatenate((hold(Y[:4]), hold(response_mask[None, :]))))
    weights = hold(fnp.linalg.solve(hold(X @ hold(X.T)), K['unit'][target]))
    _hc_put(l, 'diag', hold(weights @ X) / vc, response_mask, nrows)             # (residual / v) x a positive number; 0 outside the rows
    _HC_STATS['columns'] += 1


def _hc_k3(l, weighted_cubes, old, before, after, vc, s):
    """Columns 12, 13. weighted_cubes = lam_r pp_ir^3 (float32, all pushed columns, the first `old` are the old tier's), before /
    after = its row sum before / after the old tier's tail correction. The old tier's stored part is summed once more (its columns
    are few); the young part is the rest."""
    if _HC_U is None or not ('k3_young' in _HC_COLUMNS or 'k3_old' in _HC_COLUMNS) or not (1 <= l <= 14):
        return
    hold = _E2Hold()
    scale = hold(vc * s)
    young = before - hold(hold(fnp.sum(hold(weighted_cubes[:, :old]), axis=1)).astype(fnp.float64)) if old else before
    _hc_put(l, 'k3_young', hold(young) / scale)
    _hc_put(l, 'k3_old', hold(after - young) / scale)


def _hc_pdiag(l, value, own, lam_young, response_indices, response_mask, vc, n):
    """Column 9. value = the fold's added left-factor term on the young columns (per column, over lam), own = the columns' response
    rows, lam_young = their lam (a slice, one entry per column): sum_r lam_r value_ir own_ir is the plane's diagonal on the response
    rows (times k31 / 3)."""
    if _HC_U is None or 'k31l_diag_rows' not in _HC_COLUMNS or not (1 <= l <= 14):
        return
    hold = _E2Hold()
    full = fnp.zeros(n, dtype=fnp.float64)
    fnp.put(full, response_indices, hold(hold(value * own) @ lam_young))
    _hc_put(l, 'k31l_diag_rows', hold(full) / hold(vc * vc), response_mask, int(response_indices.shape[0]))


def _hc_bias(l, bias):
    """enc0_b, or enc0_b + sum_k z_k u_k' with the columns this layer has (used once: a column never outlives its layer; a column
    the layer did not put counts as 0)."""
    S = _HC_STATE
    if _HC_U is None or S['l'] != l:
        return bias
    present = dict(S['raw'])
    S['l'], S['raw'] = None, {}
    if not present:
        return bias
    hold = _E2Hold()
    K = _HC_CONST
    if 'u' not in K:
        K['u'] = fnp.asarray(_HC_U if len(_HC_COLUMNS) > 1 else _HC_U[0], dtype=fnp.float32)
    if len(_HC_COLUMNS) > 1:
        n = int(next(iter(present.values()))[0].shape[0])
        if K.get('ones_n') != n:
            K.update(ones_n=n, ones=fnp.ones(n, dtype=fnp.float64), zeros=fnp.zeros(n, dtype=fnp.float64))
        raw = [present.get(name, (K['zeros'], None, n)) for name in _HC_COLUMNS]
    else:
        raw = [present[_HC_COLUMNS[0]]]
    _HC_STATS['used'] += 1
    if len(raw) == 1:    # v1's operations
        x, mask, count = raw[0]
        hold(x)
        centred = hold(x - hold(hold(fnp.sum(x)) * (1.0 / count)))
        if mask is not None:
            centred = hold(centred * mask)
        scale = hold(hold(fnp.power(hold(fnp.maximum(hold(fnp.sum(hold(centred * centred))), 1e-300)), -0.5)) * ((count - 1.0) ** 0.5))
        z = hold(hold(fnp.clip(hold(centred * scale), -8.0, 8.0)).astype(fnp.float32))
        return bias + hold(fnp.outer(z, K['u']))
    M = hold(fnp.stack([(K['ones'] if mask is None else mask) for _, mask, _ in raw]))
    X = hold(hold(fnp.stack([hold(x) for x, _, _ in raw])) * M)          # a column counts on its own units only
    inverse = hold(fnp.asarray([[1.0 / count] for _, _, count in raw]))
    root = hold(fnp.asarray([[(count - 1.0) ** 0.5] for _, _, count in raw]))
    centred = hold(hold(X - hold(hold(fnp.sum(X, axis=1, keepdims=True)) * inverse)) * M)
    scale = hold(hold(fnp.power(hold(fnp.maximum(hold(fnp.sum(hold(centred * centred), axis=1, keepdims=True)), 1e-300)), -0.5)) * root)
    Z = hold(hold(fnp.clip(hold(centred * scale), -8.0, 8.0)).astype(fnp.float32))
    return bias + hold(fnp.einsum('kn,kc->nc', Z, K['u'], optimize=False))
# END OPUS-ENGINEER HEAD COLUMNS

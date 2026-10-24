# BEGIN OPUS-ENGINEER HEAD COLUMN (a42 hc v1, 2026-10-02; a change of the learned mean correction's INPUT, OFF unless the weights are
# given). The trainer's new input column of the correction's first layer (O/trainer/candidates/head_col_s1/column.json; torch code
# O/trainer/otr_m.py, function columns, column 6 'diag_resid_rows'), layers 1-14:
#     rows  = the response rows of the layer's bank read
#     B_ii  = the diagonal of the pair law's matrix on the pushed columns (the fold holds kap B_ii on the rows: _np_left's `diagonal`)
#     x_i   = (B_ii - fit_i) / (tr C v_i) on the rows, fit = least squares of B_ii over the rows on (tr C v_i, 1, mu_i^2, 2 mu_i (C mu)_i)
#     z_i   = clip((x_i - mean) / sd, -8, 8) over the rows (sd with rows - 1), 0 outside the rows
#     first layer of the correction: pre-activation = [features, messages] enc0_w' + (enc0_b + z_i u'),   u = 128 weights for all layers.
# Native form: the fold's regression rows Y hold the masked regressors and the masked diagonal already. The residual is one 5 x 5
# solve: with G the Gram matrix of (regressors, target) on the rows, G^-1 e_target is (- coefficients, 1) / (residual sum of squares),
# so its product with the rows is a positive multiple of the residual; every positive per-layer factor (kap, tr C, that sum) drops out
# in the standardisation. With the table regression (three regressors, no constant) the constant is added as a fifth row. No bill
# worth naming (per-unit rows); 19 requests a layer with the constant regressor, 21 with the table.
_HC_U = None             # the 128 weights (a tuple of float32 values); None = no column
_HC_STATE = dict(l=None, z=None)
_HC_CONST = {}
_HC_STATS = dict(columns=0, used=0)


def _hc_column(l, Y, target, vc, response_mask, nrows):
    """Called by the fold after its regression rows are made: Y rows 0..target = the masked regressors and, at `target`, the masked
    diagonal. Leaves the column of layer l in _HC_STATE."""
    S = _HC_STATE
    S['l'] = S['z'] = None
    if _HC_U is None or not (1 <= l <= 14) or nrows < 3:
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
    x = hold(hold(weights @ X) / vc)                                             # (residual / v) x a positive number; 0 outside the rows
    centred = hold(hold(x - hold(hold(fnp.sum(x)) * (1.0 / nrows))) * response_mask)
    scale = hold(hold(fnp.power(hold(fnp.maximum(hold(fnp.sum(hold(centred * centred))), 1e-300)), -0.5)) * ((nrows - 1.0) ** 0.5))
    S['z'] = hold(fnp.clip(hold(centred * scale), -8.0, 8.0)).astype(fnp.float32)
    S['l'] = l
    _HC_STATS['columns'] += 1


def _hc_bias(l, bias):
    """enc0_b, or enc0_b + z u' when this layer's read made a column (used once: a column never outlives its layer)."""
    S = _HC_STATE
    if S['z'] is None or S['l'] != l:
        return bias
    hold = _E2Hold()
    K = _HC_CONST
    if 'u' not in K:
        K['u'] = fnp.asarray(_HC_U, dtype=fnp.float32)
    z = S['z']
    S['l'] = S['z'] = None
    _HC_STATS['used'] += 1
    return bias + hold(fnp.outer(hold(z), K['u']))
# END OPUS-ENGINEER HEAD COLUMN

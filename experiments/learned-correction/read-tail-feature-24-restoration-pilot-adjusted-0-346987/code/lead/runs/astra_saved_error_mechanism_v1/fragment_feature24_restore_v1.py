
# Restore the linear standardized K21 row moment consumed by GNN feature 24.
# The incumbent covariance/K21 read remains untouched. Only its omitted young
# tail moment is recovered; the old-tail contribution already remains exact.
_F24_PARENT_FEATURES = features
_F24_PARENT_PREDICT = Estimator.predict
_F24_STATS = dict(layers=[], retired=True)


def _f24_tail_restore(left_young, right_tail, lam_young, current_young_tail,
                      rows, tail_cols, hold):
    layer = int(_RS_LAYER[0])
    if layer < 2:
        return
    assert 2 <= layer <= 14 and 'f24_tail_delta' not in _E2_LAYER
    n = int(_E2_LAYER['s'].shape[0])
    m, k = int(left_young.shape[0]), int(left_young.shape[1])
    nt = int(right_tail.shape[0])
    assert tuple(current_young_tail.shape) == (m, nt)
    inverse_s = hold(1.0 / hold(_E2_LAYER['s'][tail_cols]))
    # Two thin matvecs recover the exact mathematical young-tail row sum.
    exact = hold(left_young @ hold(lam_young * hold(right_tail.T @ inverse_s)))
    change = hold(exact - hold(current_young_tail @ inverse_s))

    # The core replaces every K21 diagonal after the read, so its tail-diagonal
    # difference must not be included in the learned feature correction.
    inverse = hold(fnp.full(n, -1, dtype=fnp.int64))
    hold(fnp.put(inverse, tail_cols, hold(fnp.arange(nt, dtype=fnp.int64))))
    positions = hold(inverse[rows])
    diag_rows = hold(fnp.nonzero(hold(positions >= 0))[0])
    nd = int(diag_rows.shape[0])
    if nd:
        diag_cols = hold(positions[diag_rows])
        a = hold(hold(left_young[diag_rows]).astype(fnp.float64))
        b = hold(hold(right_tail[diag_cols]).astype(fnp.float64))
        exact_diagonal = hold(fnp.einsum('ij,ij,j->i', a, b, lam_young, optimize=False))
        current_diagonal = hold(fnp.take(current_young_tail,
            hold(hold(diag_rows * nt) + diag_cols)))
        removal = hold(hold(exact_diagonal - current_diagonal) /
                       hold(_E2_LAYER['s'][hold(rows[diag_rows])]))
        hold(fnp.put(change, diag_rows, hold(change[diag_rows] - removal)))
    values = hold(change / hold(hold(_E2_LAYER['vc'][rows]) * (float(n) ** 0.5)))
    delta = fnp.zeros(n, dtype=fnp.float64)
    hold(fnp.put(delta, rows, values))
    _E2_LAYER['f24_tail_delta'] = delta
    _F24_STATS['layers'].append(dict(layer=layer, rows=m, young=k,
                                   tail=nt, diagonal_rows=nd))


def features(l, m, C, K21, aux, W):
    answer = _F24_PARENT_FEATURES(l, m, C, K21, aux, W)
    delta = _E2_LAYER.pop('f24_tail_delta', None)
    if delta is not None:
        assert 2 <= l <= 14
        hold = _E2Hold()
        # Match the existing float32 feature interface after adding the analytic
        # change to its row-sum coordinate. All other coordinates are unchanged.
        corrected = hold(hold(answer[:, 24].astype(fnp.float64)) + delta)
        idx = hold(hold(fnp.arange(int(answer.shape[0]), dtype=fnp.int64) * 30) + 24)
        hold(fnp.put(answer, idx, hold(corrected.astype(fnp.float32))))
    return answer


def _f24_predict(self, *args, **kwargs):
    assert 'f24_tail_delta' not in _E2_LAYER
    _F24_STATS['layers'].clear()
    _F24_STATS['retired'] = False
    try:
        return _F24_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _E2_LAYER.pop('f24_tail_delta', None)
        _F24_STATS['retired'] = True


Estimator.predict = _f24_predict

# Fixed L10 Nyström192+128 selection; source-only until explicit root grant.
# The rank stays320. No eigenvector rotation, rank increase or reused mean claim.
_RNS_RIDGE = 1e-6
_RNS_STATS = dict(calls=0, layer=10, rank=320, first=192, second=128,
                  ridge_relative=_RNS_RIDGE, negative_residuals=0)


def _rns_residual_diagonal(Y0, rows0, diagonal, hold):
    # G is symmetric in exact arithmetic. The carried approximate core can
    # introduce small asymmetry/indefiniteness, so symmetrize this192x192
    # principal block and add ONE fixed ridge. No retry or ridge escalation.
    Y64 = hold(Y0.astype(fnp.float64))
    K = hold(Y64[rows0, :])
    K = hold((K + hold(K.T)) * 0.5)
    scale = hold(fnp.maximum(hold(fnp.max(hold(fnp.abs(hold(fnp.diag(K)))))), 1e-30))
    Kr = hold(K + hold(hold(fnp.eye(int(K.shape[0]), dtype=fnp.float64)) * hold(_RNS_RIDGE * scale)))
    # A failed positive-definiteness check closes the smoke; do not silently
    # fall back or tune the regularizer using its accuracy result.
    hold(fnp.linalg.cholesky(Kr))
    X = hold(fnp.linalg.solve(Kr, hold(Y64.T)))
    removed = hold(fnp.sum(hold(Y64 * hold(X.T)), axis=1))
    residual = hold(diagonal.astype(fnp.float64) - removed)
    _shared_old384_finite(residual, 'RNS fixed Schur residual diagonal')
    _RNS_STATS['negative_residuals'] += int(fnp.sum(residual < 0.0))
    return hold(fnp.maximum(residual, 0.0))


def _rns_join_columns(J, weight, Qp, core, diagonal):
    # Reuse the incumbent exact diagonal and first192 paid Gram columns.
    # G = J diag(weight) J' + Qp core Qp'. Every column still uses the original
    # expression and matrix helpers; only the second block's row IDs change.
    hold = _E2Hold()
    m = int(J.shape[0])
    if m < 320 or tuple(diagonal.shape) != (m,):
        raise RuntimeError('RNS fixed rank320 L10 geometry')
    rows0 = hold(fnp.sort(hold(fnp.argsort(hold(-hold(diagonal.astype(fnp.float64))), kind='stable'))[:192]))

    def columns(rows):
        Y = hold(_shared_old384_mm(J, hold(hold(weight[:, None]) * hold(hold(J[rows, :]).T))))
        if Qp is not None:
            Y = hold(Y + hold(_shared_old384_mm(Qp, hold(_shared_old384_mm(core, hold(hold(Qp[rows, :]).T))))))
        return Y

    Y0 = columns(rows0)
    score = _rns_residual_diagonal(Y0, rows0, diagonal, hold)
    # Exclude selected rows after clipping so exactly128 distinct remaining
    # rows are taken even when the residual is rank-deficient/all zero.
    hold(fnp.put(score, rows0, -1.0))
    rows1 = hold(fnp.sort(hold(fnp.argsort(hold(-score), kind='stable'))[:128]))
    Y1 = columns(rows1)
    chosen = hold(fnp.concatenate((rows0, rows1)))
    order = hold(fnp.argsort(chosen, kind='stable'))
    rows = hold(chosen[order])
    Y = hold(hold(fnp.concatenate((Y0, Y1), axis=1))[:, order])
    _shared_old384_finite(Y, 'RNS split Gram columns')
    _RNS_STATS['calls'] += 1
    return Y, rows

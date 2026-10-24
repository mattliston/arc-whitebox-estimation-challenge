# Fixed GOP2 constant-output sum restoration. No source/input mean change.
_GSM_PARENT_CONTRACT = _gop2_contract
_GSM_PARENT_PREDICT = Estimator.predict
_GSM_STATS = dict(calls=0, planes=0, retired=True)
assert _GOP2_ON is True
assert _LVFS_LIMIT == 10**15

def _gop2_contract(T, X):
    # Fixed two-dimensional output range; exact full mean component.
    mu = fnp.mean(X, axis=1)
    centered = fnp.subtract(X, mu[:, None, :])
    wide = centered.astype(fnp.float64)
    gram = fnp.matmul(wide.transpose(0, 2, 1), wide)
    _, vectors = fnp.linalg.eigh(gram)
    V = vectors[:, :, -2:].astype(fnp.float32)
    coefficients = fnp.matmul(centered, V)
    projected = fnp.matmul(T, coefficients)
    Q, _ = fnp.linalg.qr(projected.astype(fnp.float64), mode='reduced')
    Q = Q.astype(fnp.float32)
    back = fnp.matmul(Q.transpose(0, 2, 1), T)
    fitted = fnp.matmul(back, centered)
    residual = fnp.matmul(Q, fitted)
    # Fixed constant-output direction orthogonal to the existing rank-two Q.
    # Float64 bookkeeping only; the original Q, fitted and residual stay unchanged.
    Q64 = Q.astype(fnp.float64)
    q = fnp.sum(Q64, axis=1)
    u = fnp.subtract(1.0, fnp.matmul(Q64, q[:, :, None]))
    denom = fnp.sum(fnp.multiply(u, u), axis=1)
    column_mass = fnp.sum(T, axis=1, dtype=fnp.float64)
    exact_sum = fnp.matmul(column_mass[:, None, :], wide)[:, 0, :]
    fitted_sum = fnp.matmul(q[:, None, :], fitted.astype(fnp.float64))[:, 0, :]
    missed_sum = fnp.subtract(exact_sum, fitted_sum)
    # Fixed numerical degeneracy rule: 64 float32 eps times output dimension.
    valid = fnp.greater(denom, (64.0 * 2.0**-23) * int(T.shape[1]))
    safe_denom = fnp.where(valid, denom, 1.0)
    factor = fnp.where(valid, fnp.divide(missed_sum, safe_denom), 0.0)
    correction = fnp.multiply(u, factor[:, None, :]).astype(fnp.float32)
    residual = fnp.add(residual, correction)
    _GSM_STATS['calls'] += 1
    _GSM_STATS['planes'] += int(T.shape[0])
    mass = fnp.sum(T, axis=2)
    return fnp.add(residual, fnp.multiply(mass[:, :, None], mu[:, None, :]))

def _gsm_predict(self, *args, **kwargs):
    global _GSM_STATS
    _GSM_STATS = dict(calls=0, planes=0, retired=False)
    try:
        return _GSM_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _GSM_STATS['retired'] = True

Estimator.predict = _gsm_predict

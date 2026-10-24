# One fixed GOP2 output-subspace iteration; rank2, all maps and full mean retained.
_GSI_ON = True
_GSI_ACTIVE = [False]
_GSI_STATS = {}
_GSI_PARENT_CONTRACT = _gop2_contract
_GSI_PARENT_PREDICT = Estimator.predict
assert _GOP2_ON is True and _GSI_PARENT_CONTRACT.__name__ == '_gop2_contract'

def _gsi_contract(T, X):
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
    # A=T*centered. The already-paid fitted is Q.T*A (batch,2,16).
    # One fixed output-covariance power step A*(A.T*Q), without forming A.
    coefficients = fnp.matmul(centered, fitted.transpose(0, 2, 1))
    projected = fnp.matmul(T, coefficients)
    Q, _ = fnp.linalg.qr(projected.astype(fnp.float64), mode='reduced')
    Q = Q.astype(fnp.float32)
    back = fnp.matmul(Q.transpose(0, 2, 1), T)
    fitted = fnp.matmul(back, centered)
    residual = fnp.matmul(Q, fitted)
    mass = fnp.sum(T, axis=2)
    return fnp.add(residual, fnp.multiply(mass[:, :, None], mu[:, None, :]))

def _gop2_contract(T, X):
    if not _GSI_ON:
        return _GSI_PARENT_CONTRACT(T, X)
    assert _GSI_ACTIVE[0] and _GOP2_ACTIVE[0]
    assert len(X.shape) == 3 and int(X.shape[2]) == 16 and int(X.shape[0]) in (1,3)
    _GSI_STATS['calls'] += 1
    _GSI_STATS['blocks'] += int(X.shape[0])
    return _gsi_contract(T, X)

def _gsi_predict(self, *args, **kwargs):
    assert not _GSI_ACTIVE[0]
    _GSI_STATS.clear()
    _GSI_STATS.update(calls=0, blocks=0, fixed_iterations=1, retired=False, value_cache_entries=0)
    _GSI_ACTIVE[0] = bool(_GSI_ON)
    try:
        return _GSI_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _GSI_ACTIVE[0] = False
        _GSI_STATS['retired'] = True
Estimator.predict = _gsi_predict

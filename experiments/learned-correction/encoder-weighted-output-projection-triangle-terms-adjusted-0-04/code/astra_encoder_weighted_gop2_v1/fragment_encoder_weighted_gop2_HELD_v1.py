# Fixed encoder-weighted source range for the existing rank-two GOP projection.
# Exact mean and output least-squares fit remain unchanged. No fitted parameters.
_GOPW_PARENT_CONTRACT = _gop2_contract
_GOPW_PARENT_MESSAGES = _gop2_messages
_GOPW_PARENT_GNN = _gop2_gnn_step
_GOPW_PARENT_PREDICT = Estimator.predict
_GOPW_METRIC = [None]
_GOPW_CACHE = []

def _gopw_metrics(P):
    if _GOPW_CACHE and _GOPW_CACHE[0] is P:
        return _GOPW_CACHE[1]
    nx = int(P['centers'].shape[1])
    cm = int(P['msg_w'].shape[0]) // 3
    encoder = P['enc0_w'][:, nx:].astype(fnp.float64)
    norms = fnp.sqrt(fnp.sum(encoder * encoder, axis=0))
    metrics = fnp.reshape(norms, (6, cm))
    _GOPW_CACHE[:] = [P, metrics]
    return metrics

def _gop2_contract(T, X):
    metric = _GOPW_METRIC[0]
    if metric is None:
        return _GOPW_PARENT_CONTRACT(T, X)
    assert tuple(metric.shape) == (int(X.shape[0]), int(X.shape[2]))
    mu = fnp.mean(X, axis=1)
    centered = fnp.subtract(X, mu[:, None, :])
    weighted = centered.astype(fnp.float64) * metric[:, None, :]
    gram = fnp.matmul(weighted.transpose(0, 2, 1), weighted)
    _, vectors = fnp.linalg.eigh(gram)
    # Form the weighted PCA coefficients, retaining the original skinny products.
    V = (vectors[:, :, -2:] * metric[:, :, None]).astype(fnp.float32)
    coefficients = fnp.matmul(centered, V)
    projected = fnp.matmul(T, coefficients)
    Q, _ = fnp.linalg.qr(projected.astype(fnp.float64), mode='reduced')
    Q = Q.astype(fnp.float32)
    back = fnp.matmul(Q.transpose(0, 2, 1), T)
    fitted = fnp.matmul(back, centered)
    residual = fnp.matmul(Q, fitted)
    mass = fnp.sum(T, axis=2)
    return fnp.add(residual, fnp.multiply(mass[:, :, None], mu[:, None, :]))

def _gop2_messages(k, W, state):
    previous = _GOPW_METRIC[0]
    _GOPW_METRIC[0] = _gopw_metrics(k['P'])[1:2]
    try:
        return _GOPW_PARENT_MESSAGES(k, W, state)
    finally:
        _GOPW_METRIC[0] = previous

def _gop2_gnn_step(l, x, sigma, W, state, P):
    previous = _GOPW_METRIC[0]
    _GOPW_METRIC[0] = _gopw_metrics(P)[3:6]
    try:
        return _GOPW_PARENT_GNN(l, x, sigma, W, state, P)
    finally:
        _GOPW_METRIC[0] = previous

def _gopw_predict(self, *args, **kwargs):
    assert _GOPW_METRIC[0] is None and not _GOPW_CACHE
    try:
        return _GOPW_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _GOPW_METRIC[0] = None
        _GOPW_CACHE.clear()

Estimator.predict = _gopw_predict

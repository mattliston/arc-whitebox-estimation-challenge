# Fixed current-CM8 consumer-weighted energy seed; no changes to M2/W3.
assert _VMFR_CM == 8 and _GNN_PARAMS is None and not _E2_GNN_STATE
assert _gop2_gnn_step.__code__.co_firstlineno == 17745
_ECM8_ROOT = None


def _ecm8_energy_contract(T, X):
    global _ECM8_ROOT
    assert X.shape[0] == 3 and X.shape[2] == 8
    if _ECM8_ROOT is None:
        # The build supplies exact current enc0 W weights, not captured data.
        enc = fnp.asarray(_ECM8_ENCODER_LITERAL, dtype=fnp.float64)
        gram_metric = fnp.matmul(enc.transpose(0, 2, 1), enc)
        values, vectors = fnp.linalg.eigh(gram_metric)
        scales = fnp.sqrt(fnp.maximum(values, 1e-30))
        _ECM8_ROOT = fnp.matmul(vectors * scales[:, None, :], vectors.transpose(0, 2, 1))
    root = _ECM8_ROOT
    mu = fnp.mean(X, axis=1)
    mu_column = mu[:, None, :]
    centered = fnp.subtract(X, mu_column)
    wide = centered.astype(fnp.float64)
    gram = fnp.matmul(wide.transpose(0, 2, 1), wide)
    weighted = fnp.matmul(fnp.matmul(root, gram), root)
    _, vectors = fnp.linalg.eigh(weighted)
    V = fnp.matmul(root, vectors[:, :, -2:]).astype(fnp.float32)
    coefficients = fnp.matmul(centered, V)
    projected = fnp.matmul(T, coefficients)
    Q, _ = fnp.linalg.qr(projected.astype(fnp.float64), mode='reduced')
    Q = Q.astype(fnp.float32)
    back = fnp.matmul(Q.transpose(0, 2, 1), T)
    fitted = fnp.matmul(back, centered)
    residual = fnp.matmul(Q, fitted)
    mass = fnp.sum(T, axis=2)
    return fnp.add(residual, fnp.multiply(mass[:, :, None], mu_column))

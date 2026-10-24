

# Fixed accuracy-for-compute single, 2026-10-21T01:02:16Z. No fitted weights change.
assert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14
assert _GNN_PARAMS is None and _W3R_METRIC is None and not _E2_MSG
assert _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}
_O21_MESSAGE_RANK_VARIANT = 'gop2rank3'

def _gop2_contract(T, X):
    mu = fnp.mean(X, axis=1)
    mu_column = mu[:, None, :]
    centered = fnp.subtract(X, mu_column)
    wide = centered.astype(fnp.float64)
    gram = fnp.matmul(wide.transpose(0, 2, 1), wide)
    _, vectors = fnp.linalg.eigh(gram)
    V = vectors[:, :, -3:].astype(fnp.float32)
    coefficients = fnp.matmul(centered, V)
    projected = fnp.matmul(T, coefficients)
    Q, _ = fnp.linalg.qr(projected.astype(fnp.float64), mode='reduced')
    Q = Q.astype(fnp.float32)
    back = fnp.matmul(Q.transpose(0, 2, 1), T)
    fitted = fnp.matmul(back, centered)
    residual = fnp.matmul(Q, fitted)
    mass = fnp.sum(T, axis=2)
    return fnp.add(residual, fnp.multiply(mass[:, :, None], mu_column))

_FAST_RESCUE_FRAC = {}

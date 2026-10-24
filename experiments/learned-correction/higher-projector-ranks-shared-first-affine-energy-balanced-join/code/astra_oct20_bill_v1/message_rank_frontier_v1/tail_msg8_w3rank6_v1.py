

# Fixed accuracy-for-compute single, 2026-10-21T01:02:15Z. No fitted weights change.
assert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14
assert _GNN_PARAMS is None and _W3R_METRIC is None and not _E2_MSG
assert _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}
_O21_MESSAGE_RANK_VARIANT = 'w3rank6'

def _w3r_product(T, X):
    root, inverse = _w3r_metric()
    wide = X.astype(fnp.float64)
    gram = fnp.matmul(wide.T, wide)
    weighted = fnp.matmul(fnp.matmul(root, gram), root)
    _, vectors = fnp.linalg.eigh(weighted)
    V = vectors[:, -6:]
    L = fnp.matmul(root, V).astype(fnp.float32)
    B = fnp.matmul(V.T, inverse).astype(fnp.float32)
    coeff = fnp.matmul(X, L)
    projected = fnp.matmul(T, coeff)
    _W3R_STATS['calls'] += 1
    return fnp.matmul(projected, B)

_FAST_RESCUE_FRAC = {}

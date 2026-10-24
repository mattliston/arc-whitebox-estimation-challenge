def _lr_factor(A32):
    # Fixed two-block star subspace, with the downstream rank held at 32.
    cn = fnp.sum(fnp.multiply(A32, A32, dtype=fnp.float64), axis=0)
    cols = fnp.sort(fnp.argsort(-cn, kind='stable')[:_LR_K])
    Q0 = fnp.linalg.qr(A32[:, cols])[0]
    Z = fnp.linalg.qr(_tsr_mm(A32.T, Q0))[0]
    Y = _tsr_mm(A32, Z)
    Q = fnp.linalg.qr(fnp.concatenate((Q0, Y), axis=1))[0]
    # Two existing rank-32 products retain the incumbent's routed kernel.
    B = fnp.concatenate((_tsr_mm(Q[:, :32].T, A32),
                         _tsr_mm(Q[:, 32:].T, A32)), axis=0)
    U, S, Vt = fnp.linalg.svd(B, full_matrices=False)
    return Q @ U[:, :32], S[:32, None] * Vt[:32]

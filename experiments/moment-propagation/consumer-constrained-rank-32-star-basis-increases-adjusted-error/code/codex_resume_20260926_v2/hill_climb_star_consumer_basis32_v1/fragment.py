def _scb32_factor(A32, w):
    """Fixed rank32 basis containing the squared-weight consumer response."""
    s = fnp.sum(w * w, axis=1, dtype=fnp.float32)
    y = A32 @ s
    norm2 = fnp.sum(y * y, dtype=fnp.float32)
    if norm2 == 0.0:
        return _lr_factor(A32)
    q = (y / fnp.sqrt(norm2))[:, None]
    cn = fnp.sum(fnp.multiply(A32, A32, dtype=fnp.float64), axis=0)
    cols = fnp.sort(fnp.argsort(-cn, kind='stable')[:31])
    Q = fnp.linalg.qr(fnp.concatenate([q, A32[:, cols]], axis=1))[0]
    Z = fnp.linalg.qr(_tsr_mm(A32.T, Q))[0]
    Y = _tsr_mm(A32, Z)
    Q = fnp.linalg.qr(fnp.concatenate([q, Y[:, 1:]], axis=1))[0]
    return Q, _tsr_mm(Q.T, A32)

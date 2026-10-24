# New immutable covariance-head payload; all arrays are materialized in predict.
_PCC_PARAMETERS = None


def _pcc_parameters():
    global _PCC_PARAMETERS
    if _PCC_PARAMETERS is None:
        _PCC_PARAMETERS = fnp.asarray(_PCC_LITERAL, dtype=fnp.float32)
    return _PCC_PARAMETERS


def _pcc_covariance(layer, C, H, Q):
    n = int(C.shape[0])
    if not 1 <= layer <= 14 or C.shape != (n, n) or H.shape != (n, 64) or Q.shape != (14, 65):
        raise ValueError('Typed current-layer covariance action')
    if str(H.dtype) != 'float32' or str(Q.dtype) != 'float32' or str(C.dtype) != 'float64':
        raise ValueError('Original FP32 hidden/head and FP64 physical covariance')
    row = Q[layer-1]
    z = H @ row[:64] + row[64]
    scale = fnp.exp(.005 * fnp.tanh(z)).astype(fnp.float64)
    updated = (C * scale[:, None]) * scale[None, :]
    if not bool(fnp.all(fnp.isfinite(updated))):
        raise ValueError('Nonfinite covariance congruence')
    return updated

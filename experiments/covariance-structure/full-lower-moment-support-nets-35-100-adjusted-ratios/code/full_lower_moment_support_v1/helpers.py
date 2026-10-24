def _flms_covariance(w, C):
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    return _mm(w.T, _mm(C.astype(fnp.float32), w)).astype(fnp.float64)

def _flms_covariance_diagonal(w, C):
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance diagonal shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    pushed = _mm(C.astype(fnp.float32), w)
    v = fnp.sum(w.astype(fnp.float64)*pushed.astype(fnp.float64), axis=0)
    return v, v

def _flms_star(w, A):
    if w.ndim != 2 or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    pushed = _mm(wt, A.astype(fnp.float32))
    sub = (2*_mm(pushed*wt, w) + _mm(wt*wt, pushed.T))/3
    diagonal = fnp.sum(pushed*wt*wt, axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    return result

def _flms_star_diagonal(w, A):
    if w.ndim != 2 or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS star diagonal shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    pushed = _mm(wt, A.astype(fnp.float32))
    return fnp.sum(pushed*wt*wt, axis=1, dtype=fnp.float64)

# Restore the full stored star only for the existing terminal diagonal consumer.
def _flms_star_diagonal(w, A):
    if w.ndim != 2 or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS star diagonal shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    pushed = _mm(wt, A if str(A.dtype) == 'float32' else A.astype(fnp.float32))
    return fnp.sum((pushed * wt) * wt, axis=1, dtype=fnp.float64)

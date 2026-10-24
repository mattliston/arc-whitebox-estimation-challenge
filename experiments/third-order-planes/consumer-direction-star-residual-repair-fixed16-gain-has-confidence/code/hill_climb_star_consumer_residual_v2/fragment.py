"""One fixed consumer-direction residual repair on S32's unchanged K32/Q1 star."""
_CSR_STATS = dict(covariance_calls=0, diagonal_calls=0, zero_directions=0,
                  max_input_width=0, max_output_width=0)


def _csr_vectors(A32, Q, B, wt, wt2):
    # wt2 has output rows and input columns. All numeric work is paid FNP.
    s = fnp.sum(wt2, axis=0, dtype=fnp.float32)
    norm2 = fnp.sum(s * s, dtype=fnp.float32)
    _CSR_STATS['max_input_width'] = max(_CSR_STATS['max_input_width'], int(wt.shape[1]))
    _CSR_STATS['max_output_width'] = max(_CSR_STATS['max_output_width'], int(wt.shape[0]))
    if norm2 == 0.0:
        _CSR_STATS['zero_directions'] += 1
        return None
    r = A32 @ s - Q @ (B @ s)
    a = wt @ (r / norm2)
    b = wt2 @ s
    return a, b


def _flms_cov_star(w, C, A):
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    C32 = C.astype(fnp.float32)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    Q, B = _lr_factor(A32)
    P1 = _tsr_mm(wt, Q)
    wt2 = wt * wt
    first = _mm(C32, w)
    T2 = _tsr_mm(wt2, B.T)
    s2 = _tsr_mm(T2, P1.T)
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    sub = s2 / 3
    diagonal = fnp.sum(P1 * T2, axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    # Above: exact original active S32 operation sequence, including s1zero.
    correction = _csr_vectors(A32, Q, B, wt, wt2)
    _CSR_STATS['covariance_calls'] += 1
    if correction is not None:
        a, b = correction
        outer = b[:, None] * a[None, :]
        outer = outer / 3
        result = result + outer
        # The represented third-tensor diagonal has no factor 1/3.
        fnp.fill_diagonal(result, diagonal + (a * b).astype(fnp.float64))
    return Cpre, result


def _flms_star_diagonal(w, A):
    if w.ndim != 2 or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS star diagonal shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    A32 = A.astype(fnp.float32)
    Q, B = _lr_factor(A32)
    pushed = (wt @ Q) @ B
    diagonal = fnp.sum((pushed * wt) * wt, axis=1, dtype=fnp.float64)
    # Keep the original terminal projection/products/reduction before repair.
    correction = _csr_vectors(A32, Q, B, wt, wt * wt)
    _CSR_STATS['diagonal_calls'] += 1
    if correction is not None:
        a, b = correction
        diagonal = diagonal + (a * b).astype(fnp.float64)
    return diagonal

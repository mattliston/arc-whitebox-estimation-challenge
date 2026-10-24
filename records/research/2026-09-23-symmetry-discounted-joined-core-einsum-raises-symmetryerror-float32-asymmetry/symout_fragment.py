

# BEGIN FABLE-ENGINEER SYMMETRIC-OUTPUT PRODUCTS (E-43, 2026-09-23; rounding-level, not bit-exact): flopscope bills a symmetric-output
# einsum (Gram X^T X, weighted Gram F diag(g) F^T) at half the dense cost, and a product whose output is symmetric only needs its upper
# blocks. (1) The covariance push w^T (C w) of every layer computes its two 512-wide diagonal blocks and ONE off-diagonal block through
# _mm and mirrors the other (0.435 instead of 0.551 dense units); the layer's two star products stay batched (P=2). (2) The layer-0
# Gram w^T w takes its diagonal blocks as symmetric einsums (0.396 units instead of 0.435). (3) The old tier's joined core
# Fj diag(weight) Fj^T is one weighted-Gram einsum (0.50 instead of about 0.6-0.7). The call sites of (2) and (3) are switched by the
# builder (count-asserted); _resident_gram512 itself is unchanged (its other caller, _cov_compact, passes a weighted pair).
def _sym_upper(A, B):
    # A @ B for a product whose output is symmetric (w^T C w): diagonal blocks and the upper block through _mm, the lower block mirrored
    m = int(A.shape[0]); h = m // 2
    out = fnp.empty((m, m), dtype=fnp.float32)
    part = _mm(A[:h], B[:, :h]); fnp.copyto(out[:h, :h], part); del part
    part = _mm(A[:h], B[:, h:]); fnp.copyto(out[:h, h:], part); fnp.copyto(out[h:, :h], part.T); del part
    part = _mm(A[h:], B[:, h:]); fnp.copyto(out[h:, h:], part); del part
    return out


def _sym_gram(X):
    # X^T X: symmetric einsums on the diagonal blocks, the upper block through _mm, the lower block mirrored
    m = int(X.shape[1]); h = m // 2
    X1, X2 = X[:, :h], X[:, h:]
    out = fnp.empty((m, m), dtype=fnp.float32)
    fnp.copyto(out[:h, :h], fnp.einsum('ai,aj->ij', X1, X1))
    part = _mm(X1.T, X2); fnp.copyto(out[:h, h:], part); fnp.copyto(out[h:, :h], part.T); del part
    fnp.copyto(out[h:, h:], fnp.einsum('ai,aj->ij', X2, X2))
    return out


def _flms_cov_star(w, C, A):
    # = (_flms_covariance(w, C), _flms_star(w, A)): the stage-1 products and the two star products stay batched; the covariance's
    # second product w^T (C w) is symmetric, so only its upper blocks are computed (_sym_upper)
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    C32 = C.astype(fnp.float32)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    first, pushed = _mm_batch([(C32, w), (wt, A32)])
    pw = pushed*wt
    wt2 = wt*wt
    s1, s2 = _mm_batch([(pw, w), (wt2, pushed.T)])
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    sub = (2*s1 + s2)/3
    diagonal = fnp.sum(pw*wt, axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    return Cpre, result
# END FABLE-ENGINEER SYMMETRIC-OUTPUT PRODUCTS

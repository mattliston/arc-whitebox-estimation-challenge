


# BEGIN FABLE-PORTER --lrnys (PT-48, lead 23:5x; TEST ONLY, not for firing): Nystrom range finder for the low-rank star. The range finder's
# Y = A Omega (Omega the fixed seeded 1024 x 48 Gaussian) becomes a gather of the 48 columns of A with the largest norm (float64 sums of
# squares, stable argsort, ascending column order), so there is no A Omega product. _LRN_POWER = False also drops the power step (A^T Q and
# A Q'); True keeps it. The projection B = Q^T A and the exact rank-32 truncation are unchanged.
_LRN_POWER = False


def _lr_factor(A32):
    A64 = A32.astype(fnp.float64)
    cn = fnp.sum(A64 * A64, axis=0)
    cols = fnp.sort(fnp.argsort(-cn, kind='stable')[:_LR_K])
    Q = fnp.linalg.qr(A32[:, cols])[0]
    for _ in range(_LR_Q if _LRN_POWER else 0):
        Q = fnp.linalg.qr(A32 @ fnp.linalg.qr(A32.T @ Q)[0])[0]
    B = Q.T @ A32
    if _LR_R is None:
        return Q, B
    U, S, Vt = fnp.linalg.svd(B, full_matrices=False)
    return Q @ U[:, :_LR_R], S[:_LR_R, None] * Vt[:_LR_R]
# END FABLE-PORTER --lrnys

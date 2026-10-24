# Research approximation: stop subspace iteration on the right after one
# half-step and truncate A @ V. This omits the final full Q.T @ A projection.
# For orthonormal V, the returned product is the rank-r truncation of A V V.T.
# It retains the A.T @ Q0 and A @ V passes; it is not the rejected no-power arm.
def _lr_factor(A32):
    A64 = A32.astype(fnp.float64)
    cn = fnp.sum(A64 * A64, axis=0)
    cols = fnp.sort(fnp.argsort(-cn, kind='stable')[:_LR_K])
    Q0 = fnp.linalg.qr(A32[:, cols])[0]
    V = fnp.linalg.qr(A32.T @ Q0)[0]
    Z = A32 @ V
    U, S, Rt = fnp.linalg.svd(Z, full_matrices=False)
    r = _LR_K if _LR_R is None else _LR_R
    right = V @ Rt[:r].T
    return U[:, :r], S[:r, None] * right.T

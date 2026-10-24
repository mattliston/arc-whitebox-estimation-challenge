# Fixed rank32 residual-column start; the original QR, Q1 and B arithmetic follows.
_SRP_STATS = dict(calls=0, updates=0, fallbacks=0, zero_inputs=0, negative_scores=0)


def _srp_columns(A32, A64, cn, k):
    m, n = map(int, A32.shape)
    if not (1 <= k <= min(m, n) and tuple(A64.shape) == (m, n)
            and tuple(cn.shape) == (n,) and str(A32.dtype) == 'float32'
            and str(A64.dtype) == str(cn.dtype) == 'float64'):
        raise ValueError('SRP column-selection shape/dtype')
    if not bool(fnp.all(fnp.isfinite(cn))):
        raise ValueError('SRP nonfinite input column norm')
    _SRP_STATS['calls'] += 1
    energy_order = fnp.argsort(-cn, kind='stable')
    maximum = fnp.max(cn)
    if bool(maximum == 0):
        _SRP_STATS['zero_inputs'] += 1
        _SRP_STATS['fallbacks'] += 1
        return fnp.sort(energy_order[:k])
    # u64 is unit roundoff, not epsilon. This fixed breakdown convention is
    # distinct from the accuracy of the mixed-precision residual scores.
    u64 = 1.1102230246251565e-16
    gamma = (m * u64) / (1.0 - m * u64)
    tolerance = (gamma * gamma) * maximum
    basis = fnp.empty((m, k - 1), dtype=fnp.float64)
    chosen = fnp.empty(k, dtype=fnp.int64)
    used = fnp.zeros(n, dtype=fnp.bool_)
    residual = cn
    for t in range(k):
        # Argmax chooses the first original index in an exact tie. Reapply
        # exclusion after each nonnegative clamp, so selected entries stay out.
        j = int(fnp.argmax(fnp.where(used, -1.0, residual)))
        if t == k - 1:
            fnp.put(chosen, t, j)
            break
        v = A64[:, j]
        if t:
            previous = basis[:, :t]
            for _ in range(2):
                v = v - previous @ (previous.T @ v)
        norm2 = fnp.sum(v * v, dtype=fnp.float64)
        if not bool(fnp.isfinite(norm2)):
            raise ValueError('SRP nonfinite pivot residual')
        if bool(norm2 <= tolerance):
            remaining = energy_order[fnp.logical_not(used[energy_order])]
            fnp.copyto(chosen[t:], remaining[:k - t])
            _SRP_STATS['fallbacks'] += 1
            break
        q = v / fnp.sqrt(norm2)
        fnp.copyto(basis[:, t], q)
        fnp.put(chosen, t, j)
        fnp.put(used, j, True)
        projection = (q.astype(fnp.float32) @ A32).astype(fnp.float64)
        residual = residual - projection * projection
        if not bool(fnp.all(fnp.isfinite(residual))):
            raise ValueError('SRP nonfinite score downdate')
        _SRP_STATS['negative_scores'] += int(fnp.count_nonzero(residual < 0))
        residual = fnp.maximum(residual, 0.0)
        _SRP_STATS['updates'] += 1
    return fnp.sort(chosen)


def _lr_factor(A32):
    A64 = A32.astype(fnp.float64)
    cn = fnp.sum(A64 * A64, axis=0)
    cols = _srp_columns(A32, A64, cn, _LR_K)
    Q = fnp.linalg.qr(A32[:, cols])[0]
    for _ in range(_LR_Q if _LRN_POWER else 0):
        Q = fnp.linalg.qr(_tsr_mm(A32, fnp.linalg.qr(_tsr_mm(A32.T, Q))[0]))[0]
    B = _tsr_mm(Q.T, A32)
    if _LR_R is None:
        return Q, B
    U, S, Vt = fnp.linalg.svd(B, full_matrices=False)
    return Q @ U[:, :_LR_R], S[:_LR_R, None] * Vt[:_LR_R]

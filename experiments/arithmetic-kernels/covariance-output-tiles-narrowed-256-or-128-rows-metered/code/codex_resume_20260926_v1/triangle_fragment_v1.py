

# Compute upper block rows of an analytically symmetric product, then reflect.
# Same numerical operation as S29's two-block covariance product, with four
# 256-row blocks. No symmetry flags or inferred-symmetric einsums are used.
_TRI_BLOCK = 256


def _sym_upper(A, B):
    m = int(A.shape[0])
    block = min(_TRI_BLOCK, m)
    out = fnp.empty((m, m), dtype=fnp.float32)
    hold = []
    for i in range(0, m, block):
        j = min(i + block, m)
        top = _mm(A[i:j, :], B[:, i:])
        hold.append(fnp.copyto(out[i:j, i:], top))
        if j < m:
            hold.append(fnp.copyto(out[j:, i:j], top[:, j-i:].T))
    return out


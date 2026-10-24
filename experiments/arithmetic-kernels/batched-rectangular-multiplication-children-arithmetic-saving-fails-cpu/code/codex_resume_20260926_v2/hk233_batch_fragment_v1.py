def _hkb233_product(a, b, mm_many, quantum=64):
    m, k, n = (int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))
    assert int(b.shape[0]) == k and a.ndim == b.ndim == 2
    M = (m + 2 * quantum - 1) // (2 * quantum) * 2 * quantum
    K = (k + 3 * quantum - 1) // (3 * quantum) * 3 * quantum
    N = (n + 3 * quantum - 1) // (3 * quantum) * 3 * quantum
    h, v, w = (M // 2, K // 3, N // 3)
    aa = fnp.zeros((M, K), dtype=a.dtype)
    bb = fnp.zeros((K, N), dtype=b.dtype)
    fnp.copyto(aa[:m, :k], a)
    fnp.copyto(bb[:k, :n], b)
    A_1_1 = aa[0 * h:1 * h, 0 * v:1 * v]
    A_1_2 = aa[0 * h:1 * h, 1 * v:2 * v]
    A_1_3 = aa[0 * h:1 * h, 2 * v:3 * v]
    A_2_1 = aa[1 * h:2 * h, 0 * v:1 * v]
    A_2_2 = aa[1 * h:2 * h, 1 * v:2 * v]
    A_2_3 = aa[1 * h:2 * h, 2 * v:3 * v]
    B_1_1 = bb[0 * v:1 * v, 0 * w:1 * w]
    B_1_2 = bb[0 * v:1 * v, 1 * w:2 * w]
    B_1_3 = bb[0 * v:1 * v, 2 * w:3 * w]
    B_2_1 = bb[1 * v:2 * v, 0 * w:1 * w]
    B_2_2 = bb[1 * v:2 * v, 1 * w:2 * w]
    B_2_3 = bb[1 * v:2 * v, 2 * w:3 * w]
    B_3_1 = bb[2 * v:3 * v, 0 * w:1 * w]
    B_3_2 = bb[2 * v:3 * v, 1 * w:2 * w]
    B_3_3 = bb[2 * v:3 * v, 2 * w:3 * w]
    m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9, m_10, m_11, m_12, m_13, m_14, m_15 = mm_many([(A_2_1, B_1_1 - B_2_1 - B_1_3), (A_1_2, B_2_2 - B_1_2 - B_2_3), (A_1_3, -B_3_2 - B_1_3 + B_3_3), (A_2_3, -B_3_1 - B_2_3 + B_3_3), (A_1_2 + A_1_1, B_1_2), (A_1_1 + A_1_3, B_1_3), (A_1_1 - A_2_1, B_1_1 - B_3_1 - B_1_2), (A_2_1 + A_1_2, B_2_1 - B_1_2), (A_2_2 + A_2_1, B_2_1), (A_1_2 - A_2_2, B_2_1 - B_2_2 + B_3_2), (A_2_2 + A_2_3, B_2_3), (A_1_1 - A_2_1 + A_1_3, B_3_1 - B_1_3), (A_1_2 - A_2_2 - A_2_3, -B_3_2 + B_2_3), (A_1_2 - A_2_2 + A_1_3 - A_2_3, B_3_2), (A_2_1 - A_1_1 + A_2_3 - A_1_3, B_3_1)])
    out = fnp.empty((M, N), dtype=a.dtype)
    fnp.copyto(out[0 * h:1 * h, 0 * w:1 * w], m_1 + m_5 + m_6 + m_7 + m_8 + m_12)
    fnp.copyto(out[1 * h:2 * h, 0 * w:1 * w], m_1 + m_6 + m_9 + m_12 + m_15)
    fnp.copyto(out[0 * h:1 * h, 1 * w:2 * w], m_2 + m_5 + m_11 + m_13 + m_14)
    fnp.copyto(out[1 * h:2 * h, 1 * w:2 * w], m_2 - m_8 + m_9 + m_10 + m_11 + m_13)
    fnp.copyto(out[0 * h:1 * h, 2 * w:3 * w], m_3 + m_6 + m_11 + m_13 + m_14)
    fnp.copyto(out[1 * h:2 * h, 2 * w:3 * w], m_4 + m_6 + m_11 + m_12 + m_15)
    return out[:m, :n]


def _hkb6_many(pairs):
    """Largest complete batch that fits the unchanged depth-six arena."""
    m, k, n = int(pairs[0][0].shape[0]), int(pairs[0][0].shape[1]), int(pairs[0][1].shape[1])
    assert m % 64 == k % 64 == n % 64 == 0
    assert all(tuple(a.shape) == (m, k) and tuple(b.shape) == (k, n)
               and str(a.dtype) == str(b.dtype) == 'float32' for a, b in pairs)
    counts = (7**6 * (m//64) * (k//64),
              7**6 * (k//64) * (n//64),
              7**6 * (m//64) * (n//64))
    cap = min(15, *(r // q for r, q in zip(_IPT_REGION[6], counts)))
    assert cap >= 1 and _AB_ON and _SCRATCH is not None
    outs = [fnp.empty((m, n), dtype=fnp.float32) for _ in pairs]
    _TR_LAST[0] = None
    for start in range(0, len(pairs), cap):
        block = pairs[start:start + cap]
        target = outs[start:start + cap]
        _bb_product([a for a, b in block], [b for a, b in block], target, 6)
    _TR_LAST[0] = None
    return outs

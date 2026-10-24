# Hopcroft-Kerr <2,3,3:15>, coefficients from fmm.univ-lille.fr/2x3x3.html.
# Research helper only. All model arithmetic uses metered fnp operations.
def _hk233_product(a, b, mm, quantum=64):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    assert int(b.shape[0]) == k and a.ndim == b.ndim == 2
    M = ((m + 2 * quantum - 1) // (2 * quantum)) * 2 * quantum
    K = ((k + 3 * quantum - 1) // (3 * quantum)) * 3 * quantum
    N = ((n + 3 * quantum - 1) // (3 * quantum)) * 3 * quantum
    h, v, w = M // 2, K // 3, N // 3
    aa = fnp.zeros((M, K), dtype=a.dtype)
    bb = fnp.zeros((K, N), dtype=b.dtype)
    fnp.copyto(aa[:m, :k], a)
    fnp.copyto(bb[:k, :n], b)
    A_1_1 = aa[0*h:1*h, 0*v:1*v]
    A_1_2 = aa[0*h:1*h, 1*v:2*v]
    A_1_3 = aa[0*h:1*h, 2*v:3*v]
    A_2_1 = aa[1*h:2*h, 0*v:1*v]
    A_2_2 = aa[1*h:2*h, 1*v:2*v]
    A_2_3 = aa[1*h:2*h, 2*v:3*v]
    B_1_1 = bb[0*v:1*v, 0*w:1*w]
    B_1_2 = bb[0*v:1*v, 1*w:2*w]
    B_1_3 = bb[0*v:1*v, 2*w:3*w]
    B_2_1 = bb[1*v:2*v, 0*w:1*w]
    B_2_2 = bb[1*v:2*v, 1*w:2*w]
    B_2_3 = bb[1*v:2*v, 2*w:3*w]
    B_3_1 = bb[2*v:3*v, 0*w:1*w]
    B_3_2 = bb[2*v:3*v, 1*w:2*w]
    B_3_3 = bb[2*v:3*v, 2*w:3*w]
    m_1 = mm(A_2_1, B_1_1 - B_2_1 - B_1_3).copy()
    m_2 = mm(A_1_2, B_2_2 - B_1_2 - B_2_3).copy()
    m_3 = mm(A_1_3, -B_3_2 - B_1_3 + B_3_3).copy()
    m_4 = mm(A_2_3, -B_3_1 - B_2_3 + B_3_3).copy()
    m_5 = mm(A_1_2 + A_1_1, B_1_2).copy()
    m_6 = mm(A_1_1 + A_1_3, B_1_3).copy()
    m_7 = mm(A_1_1 - A_2_1, B_1_1 - B_3_1 - B_1_2).copy()
    m_8 = mm(A_2_1 + A_1_2, B_2_1 - B_1_2).copy()
    m_9 = mm(A_2_2 + A_2_1, B_2_1).copy()
    m_10 = mm(A_1_2 - A_2_2, B_2_1 - B_2_2 + B_3_2).copy()
    m_11 = mm(A_2_2 + A_2_3, B_2_3).copy()
    m_12 = mm(A_1_1 - A_2_1 + A_1_3, B_3_1 - B_1_3).copy()
    m_13 = mm(A_1_2 - A_2_2 - A_2_3, -B_3_2 + B_2_3).copy()
    m_14 = mm(A_1_2 - A_2_2 + A_1_3 - A_2_3, B_3_2).copy()
    m_15 = mm(A_2_1 - A_1_1 + A_2_3 - A_1_3, B_3_1).copy()
    out = fnp.empty((M, N), dtype=a.dtype)
    fnp.copyto(out[0*h:1*h, 0*w:1*w], m_1 + m_5 + m_6 + m_7 + m_8 + m_12)
    fnp.copyto(out[1*h:2*h, 0*w:1*w], m_1 + m_6 + m_9 + m_12 + m_15)
    fnp.copyto(out[0*h:1*h, 1*w:2*w], m_2 + m_5 + m_11 + m_13 + m_14)
    fnp.copyto(out[1*h:2*h, 1*w:2*w], m_2 - m_8 + m_9 + m_10 + m_11 + m_13)
    fnp.copyto(out[0*h:1*h, 2*w:3*w], m_3 + m_6 + m_11 + m_13 + m_14)
    fnp.copyto(out[1*h:2*h, 2*w:3*w], m_4 + m_6 + m_11 + m_12 + m_15)
    return out[:m, :n]

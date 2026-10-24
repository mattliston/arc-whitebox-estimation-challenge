# Catalog rank-23 3x3x3 scheme; source fmm.univ-lille.fr/3x3x3.html.
# Research only: every model operation is metered by fnp.
def _ld333_product(a, b, mm, quantum=64):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    assert int(b.shape[0]) == k and a.ndim == b.ndim == 2
    M = ((m + 3*quantum - 1) // (3*quantum)) * 3*quantum
    K = ((k + 3*quantum - 1) // (3*quantum)) * 3*quantum
    N = ((n + 3*quantum - 1) // (3*quantum)) * 3*quantum
    h, v, w = M // 3, K // 3, N // 3
    aa = fnp.zeros((M, K), dtype=a.dtype)
    bb = fnp.zeros((K, N), dtype=b.dtype)
    fnp.copyto(aa[:m, :k], a)
    fnp.copyto(bb[:k, :n], b)
    out = fnp.empty((M, N), dtype=a.dtype)
    A_1_1 = aa[0*h:1*h, 0*v:1*v]
    A_1_2 = aa[0*h:1*h, 1*v:2*v]
    A_1_3 = aa[0*h:1*h, 2*v:3*v]
    A_2_1 = aa[1*h:2*h, 0*v:1*v]
    A_2_2 = aa[1*h:2*h, 1*v:2*v]
    A_2_3 = aa[1*h:2*h, 2*v:3*v]
    A_3_1 = aa[2*h:3*h, 0*v:1*v]
    A_3_2 = aa[2*h:3*h, 1*v:2*v]
    A_3_3 = aa[2*h:3*h, 2*v:3*v]
    B_1_1 = bb[0*v:1*v, 0*w:1*w]
    B_1_2 = bb[0*v:1*v, 1*w:2*w]
    B_1_3 = bb[0*v:1*v, 2*w:3*w]
    B_2_1 = bb[1*v:2*v, 0*w:1*w]
    B_2_2 = bb[1*v:2*v, 1*w:2*w]
    B_2_3 = bb[1*v:2*v, 2*w:3*w]
    B_3_1 = bb[2*v:3*v, 0*w:1*w]
    B_3_2 = bb[2*v:3*v, 1*w:2*w]
    B_3_3 = bb[2*v:3*v, 2*w:3*w]
    r9 = 1 / 2 * A_3_1
    oA8 = A_1_1 + r9
    r10 = 1 / 2 * A_3_2
    oA1 = A_2_2 + r10
    oA13 = A_2_1 + r9
    r12 = 2 * A_1_3
    oA17 = A_1_2 - r12
    r13 = 2 * A_2_3
    oA6 = r13 - oA1
    oA15 = r13 - oA13
    t15 = oA17 + r10
    oA11 = r12 - oA8
    oA0 = A_2_2 - r13
    oA3 = -A_2_3
    oA4 = A_3_3 + oA6 + oA11
    oA7 = A_1_1 - r12
    oA10 = -A_3_1
    oA12 = A_2_1 - r13
    oA14 = A_3_3 + oA15 - t15
    oA16 = -t15
    oA18 = A_1_2 + r10
    oA2 = A_3_3
    oA5 = A_3_2
    oA9 = A_1_3
    oA19 = A_2_2
    oA20 = A_2_1
    oA21 = A_1_1
    oA22 = A_1_2
    r9 = 2 * B_1_3
    oB12 = B_1_1 - r9
    r10 = 1 / 2 * B_3_2
    oB8 = B_1_2 + r10
    r11 = 2 * B_2_3
    oB0 = B_2_2 - r11
    r12 = 1 / 2 * B_3_1
    oB18 = B_2_1 + r12
    oB15 = oB12 + r12
    oB16 = oB18 - r11
    oB11 = oB8 - r9
    oB6 = oB0 + r10
    oB1 = B_2_2 + r10
    oB3 = oB15 + oB6 - B_3_3
    oB7 = B_1_2 - r9
    oB9 = B_3_3 - oB16 - oB11
    oB13 = B_1_1 + r12
    oB17 = B_2_1 - r11
    oB2 = B_3_3
    oB4 = B_3_2
    oB5 = B_2_3
    oB10 = B_1_3
    oB14 = B_3_1
    oB19 = B_2_1
    oB20 = B_1_2
    oB21 = B_1_1
    oB22 = B_2_2
    iC0 = mm(oA0, oB0).copy()
    iC1 = mm(oA1, oB1).copy()
    iC2 = mm(oA2, oB2).copy()
    iC3 = mm(oA3, oB3).copy()
    iC4 = mm(oA4, oB4).copy()
    iC5 = mm(oA5, oB5).copy()
    iC6 = mm(oA6, oB6).copy()
    iC7 = mm(oA7, oB7).copy()
    iC8 = mm(oA8, oB8).copy()
    iC9 = mm(oA9, oB9).copy()
    iC10 = mm(oA10, oB10).copy()
    iC11 = mm(oA11, oB11).copy()
    iC12 = mm(oA12, oB12).copy()
    iC13 = mm(oA13, oB13).copy()
    iC14 = mm(oA14, oB14).copy()
    iC15 = mm(oA15, oB15).copy()
    iC16 = mm(oA16, oB16).copy()
    iC17 = mm(oA17, oB17).copy()
    iC18 = mm(oA18, oB18).copy()
    iC19 = mm(oA19, oB19).copy()
    iC20 = mm(oA20, oB20).copy()
    iC21 = mm(oA21, oB21).copy()
    iC22 = mm(oA22, oB22).copy()
    t16 = iC15 + iC10
    t13 = iC11 + iC10
    t15 = iC6 - iC5
    t14 = iC16 - iC5
    fnp.copyto(out[2*h:3*h, 2*w:3*w], iC5 - iC10 + iC2)
    t12 = iC13 + t16
    t10 = iC1 + t15
    t9 = iC17 + t14
    t11 = iC8 + t13
    fnp.copyto(out[2*h:3*h, 1*w:2*w], iC4 - 2 * iC7 - 2 * iC0 - 2 * t15 - 2 * t13)
    fnp.copyto(out[1*h:2*h, 0*w:1*w], iC19 + iC12 + t12)
    fnp.copyto(out[0*h:1*h, 1*w:2*w], iC22 + iC7 + t11)
    fnp.copyto(out[0*h:1*h, 2*w:3*w], iC9 + 1 / 2 * iC18 + 1 / 2 * t14 + 1 / 2 * t11)
    fnp.copyto(out[1*h:2*h, 1*w:2*w], iC20 + iC0 + t10)
    fnp.copyto(out[1*h:2*h, 2*w:3*w], iC3 + 1 / 2 * t12 + 1 / 2 * t10)
    fnp.copyto(out[0*h:1*h, 0*w:1*w], iC21 + iC18 + t9)
    fnp.copyto(out[2*h:3*h, 0*w:1*w], iC14 - 2 * iC12 - 2 * t16 - 2 * t9)
    return out[:m, :n]

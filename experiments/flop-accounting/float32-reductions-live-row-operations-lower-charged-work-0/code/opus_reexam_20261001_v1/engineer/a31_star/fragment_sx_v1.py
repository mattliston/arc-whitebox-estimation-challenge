# BEGIN OPUS-ENGINEER STAR FACTORIZATION WITH ONE PRODUCT LESS (a31 SX v1, 2026-10-02; an approximation, NOT rounding-level): the
# parent factors the star plane A as Q B with Q0 = orth(32 columns of A), Q1 = orth(A' Q0), Q = orth(A Q1), B = Q' A: three thin
# products with A. Z = A Q1 already gives a rank-32 factorization, A ~ Z Q1' (A's rows projected on span(Q1)); from layer
# _SX_FIRST on it is returned in place of (Q, B), which drops the product Q' A and the second QR (67 Mflop per layer as plain
# matmuls). Probe a31_star/probe_star.py (network 651): the transported star's error rises by 6-19% at layers 4-14 and by 36-61% at
# layers 2-3, hence the first layer.
_SX_FIRST = 4
_SX_OLD_LR_FACTOR = _lr_factor
_SX_STATS = dict(short=0, full=0)


def _lr_factor(A32):
    if int(_PHYS_LAYER[0]) + 1 < _SX_FIRST or _LR_R is not None or not _LRN_POWER or _LR_Q != 1:   # _PHYS_LAYER holds the layer before the one this star is moved into
        _SX_STATS['full'] += 1
        return _SX_OLD_LR_FACTOR(A32)
    cn = fnp.sum(fnp.multiply(A32, A32, dtype=fnp.float64), axis=0)
    cols = fnp.sort(fnp.argsort(-cn, kind='stable')[:_LR_K])
    Q1 = fnp.linalg.qr(_tsr_mm(A32.T, fnp.linalg.qr(A32[:, cols])[0]))[0]
    _SX_STATS['short'] += 1
    return _tsr_mm(A32, Q1), Q1.T
# END OPUS-ENGINEER STAR FACTORIZATION WITH ONE PRODUCT LESS

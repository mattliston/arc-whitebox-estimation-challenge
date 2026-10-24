# BEGIN OPUS-ENGINEER STAR WITH EXACT ROW SUMS (a33 SS v3, 2026-10-02; a physics change, NOT rounding-level). What every layer takes
# from the star plane A is s2 = (W o W)' A' W. The squared weights are positive: (W o W)' = m 1' + R, m_i = the mean of unit i's
# squared weights, so
#     s2 = m (W' A 1)' + R A' W.
# The first part needs only the ROW SUMS of A (the third-order coupling of every unit to the squared norm of the layer below) and
# holds 99.4-99.8% of s2's power at layers 4-14 (97-99.6% at layers 1-3; network 651, a31_star/probe_star3.py). The parent's
# rank-32 factors Q B miss 1-2% of the row sums, which is 15-45% of the power of the whole error they make in s2 and (by the
# native sessions) the part of it that costs MSE. Three forms, chosen per layer by _SS_RANK (one entry for the star moved into
# each of the layers 1..15), all with the first part exact:
#   k > 0  the parent's factors at rank k plus one more column in each factor of s2 (T2 gets m, P1 gets W' u, u = A 1 - Q (B 1))
#   0      the first part alone: s2 = m (W' A 1)'; no factorization
#   -1     one direction, no factorization: a = A 1 (the row sums), s2 = ((W o W)' A' a) (W' a)' / a'a. It holds the first part
#          exactly (A' a sums to a'a) and, because the star plane is 60-75% one rank-one piece at layers 4-12, 45% of the rest that
#          rank 32 holds (a33_starsum/forms_offline.py), for 8 n^2 flops a layer.
# (v1: ranks > 0 only; v2: + rank 0; the first sessions ran on those.)
_SS_RANK = (32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32)
_SS_STATS = dict(planes=0, final=0)


def _ss_factors(wt, wt2, A32, hold):
    # the two factors of s2 = T2 P1' for the layer this star is moved into (or of its rows, for the final layer)
    global _LR_K
    n = int(A32.shape[0])
    K = _SS_RANK[int(_PHYS_LAYER[0])]   # _PHYS_LAYER holds the layer before the one this star is moved into
    rows = hold(fnp.sum(A32, axis=1))
    if K < 0:
        return hold(wt @ rows)[:, None], hold(hold(wt2 @ hold(rows @ A32)) / hold(fnp.sum(hold(rows * rows))))[:, None]
    mean = hold(hold(fnp.sum(wt2, axis=1)) / n)
    if K == 0:
        return hold(wt @ rows)[:, None], mean[:, None]
    _LR_K = K
    Q, B = _lr_factor(A32)
    miss = hold(rows - hold(Q @ hold(fnp.sum(B, axis=1))))
    P1 = fnp.concatenate([hold(_tsr_mm(wt, Q)), hold(hold(wt @ miss)[:, None])], axis=1)
    T2 = fnp.concatenate([hold(_tsr_mm(wt2, B.T)), hold(mean[:, None])], axis=1)
    return P1, T2


def _flms_cov_star(w, C, A, retain32=False):
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    hold = _E2Hold()
    wt = w.T
    C32 = _ps32(C)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    wt2 = wt * wt
    P1, T2 = _ss_factors(wt, wt2, A32, hold)
    first = _mm(C32, w)
    s2 = _tsr_mm(T2, P1.T)
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    sub = s2 / 3
    diagonal = fnp.sum(hold(P1 * T2), axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    _SS_STATS['planes'] += 1
    if retain32:
        fnp.fill_diagonal(sub, diagonal)
        return Cpre, result, cov, sub
    return Cpre, result


def _flms_star_diagonal(w, A):
    # the L15 star diagonal (w = the kept rows' weight columns), same form of the plane leaving L14, same exact row sums
    if w.ndim != 2 or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS star diagonal shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    hold = _E2Hold()
    wt = w.T
    P1, T2 = _ss_factors(wt, hold(wt * wt), A.astype(fnp.float32), hold)
    _SS_STATS['final'] += 1
    return fnp.sum(hold(P1 * T2), axis=1, dtype=fnp.float64)
# END OPUS-ENGINEER STAR WITH EXACT ROW SUMS

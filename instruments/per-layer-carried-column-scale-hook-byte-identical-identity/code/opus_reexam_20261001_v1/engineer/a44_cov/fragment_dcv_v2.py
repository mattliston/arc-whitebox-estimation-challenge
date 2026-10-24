# BEGIN OPUS-ENGINEER DEAD-CUT COVARIANCE SANDWICH (a44 dcv v2, 2026-10-10; an APPROXIMATION; lead 10-10 02:29Z / 03:0xZ). v2 = v1 with
# the floor of the roster its own constant: _DCV_FLOOR (None = the live roster's floor, 0.01, v1). At 0.01 (v1) the cut units'
# covariances with the live ones carry 0.6-2e-3 of C_pre and the final-layer error rose 7-16% (RESULT.md section 32); at 1e-3
# (alpha -3.09) the cut drops 0.6-1.5e-4 (651: 75-151 units at layers 9-14, m 960 / 896), at Phi(-3.5) 0.6-1.0e-4 (95-121 units
# at layers 11-14, m 960; a44_cov/probe_deadcov_v3.py). v1 text follows.
# Every layer 1-15 pushes the post covariance C of the layer below through its weights: C_pre = W' C W (layers 1-14: the product
# C W, 1.01 Gflop, and the symmetric W' (C W), 0.71 Gflop, 9% of the bill together; layer 15: the diagonal alone, one C W). The
# units of the layer below outside its live roster (gate under the floor, 0.01) hold 1e-4 of C's trace; here their rows and columns
# of C are dropped. With a = the roster of the layer below (its units by gate as select() orders them, the count at or above the
# floor rounded up to a multiple of _DCV_ROUND, the size the products' kernels pad to anyway):
#     layers 1-14   C_pre = W_a' C_aa W_a          (the star part as the parent: _flms_cov_star's own lines)
#     layer 15      v     = the column sums of W_a o (C_aa W_a)
# A roster of the whole width runs the parent. What it drops on network 651 (a44_cov/probe_deadcov.py; live 992 .. 800 of 1,024 at
# layers 5-14, all 1,024 at layers 1-4): 0.6-2.0e-3 of C_pre's Frobenius norm, its diagonal 0.7-3.6e-4 rms, nearly all of it the
# cross covariances of the cut units with the live ones (the file's own float32 product is 1e-4 off).
_DCV_ROUND = 64
_DCV_FLOOR = None
_DCV_STATS = dict(cut=0, full=0, rows=0)


def _dcv_roster(gate, floor):
    """The units of the layer below the products keep, in select()'s order; None = all of them."""
    n = int(gate.shape[0])
    floor = floor if _DCV_FLOOR is None else _DCV_FLOOR
    m = min(n, (int(fnp.nonzero(gate >= floor)[0].shape[0]) + _DCV_ROUND - 1) // _DCV_ROUND * _DCV_ROUND)
    if m >= n:
        _DCV_STATS['full'] += 1
        return None
    _DCV_STATS['cut'] += 1
    _DCV_STATS['rows'] += m
    return fnp.argsort(-gate, kind='stable')[:m]


def _dc_cov_star(w, C, A, gate, floor, retain32=False):
    roster = _dcv_roster(gate, floor)
    if roster is None:
        return _flms_cov_star(w, C, A, retain32)
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
    wa = hold(w[roster])
    first = _mm(hold(C32[fnp.ix_(roster, roster)]), wa)
    s2 = _tsr_mm(T2, P1.T)
    cov = _sym_upper(hold(wa.T), first)
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


def _dc_cov_diagonal(w, C, gate, floor):
    roster = _dcv_roster(gate, floor)
    if roster is None:
        return _flms_covariance_diagonal(w, C)
    hold = _E2Hold()
    wa = hold(w[roster])
    pushed = hold(_mm(hold(hold(_ps32(C))[fnp.ix_(roster, roster)]), wa))
    v = fnp.sum(hold(fnp.multiply(wa, pushed, dtype=fnp.float64)), axis=0)
    return v, v
# END OPUS-ENGINEER DEAD-CUT COVARIANCE SANDWICH

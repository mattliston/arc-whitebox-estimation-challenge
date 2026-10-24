

# BEGIN FABLE-ENGINEER2 --s1lr (E2-S1LR v2, 2026-09-24; an APPROXIMATION, gated on VALID40): the star's dense product s1 = pw w
# (pw = pushed o w^T; until now the second half of the layer's batched depth-6 pair with the covariance's C w) is computed from pw's
# rank-32 factors by the same E-62 recipe as the star plane (range finder with the fixed seeded 48-column Omega, one power step, exact
# truncation to rank 32), using the ORIGINAL unrouted _lr_factor so the request budget holds: s1 = L (R w). In-process screen
# (fable_engineer2_v1/lowrank/, S21, nets 37/39/35/100, no refit, learned ON): raw +0.43% (per net +0.53/-0.52/+0.58/+1.14; full-rank
# control -0.02%), native bill ~-2.1%. C w now runs as one product through _mm (depth 6 via --d6x), the same per-product arithmetic
# as its half of the batched pair. v2: stacks directly on S21 (lead 21:47: the --lrroute subset is parked on the residual budget);
# with --lrroute present it would still take the unrouted factor. _flms_star (unbatched fallback) and _flms_star_diagonal are unchanged.
_S1LR_ON = True
_S1LR_OLD_COV_STAR = _flms_cov_star
_S1LR_FACTOR = _LRR_OLD_LR_FACTOR if '_LRR_OLD_LR_FACTOR' in globals() else _lr_factor


def _flms_cov_star(w, C, A):
    if not _S1LR_ON:
        return _S1LR_OLD_COV_STAR(w, C, A)
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    C32 = C.astype(fnp.float32)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    Q, B = _lr_factor(A32)
    P1 = wt @ Q
    pushed = P1 @ B
    pw = pushed * wt
    wt2 = wt * wt
    first = _mm(C32, w)
    L1, R1 = _S1LR_FACTOR(pw)
    s1 = L1 @ (R1 @ w)
    s2 = (wt2 @ B.T) @ P1.T
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    sub = (2 * s1 + s2) / 3
    diagonal = fnp.sum(pw * wt, axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    return Cpre, result
# END FABLE-ENGINEER2 --s1lr

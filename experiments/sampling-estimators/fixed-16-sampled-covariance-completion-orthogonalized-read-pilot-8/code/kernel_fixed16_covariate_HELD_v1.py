# Held fixed16 covariance-shaped residual correction; no hook/activation.
# Caller supplies already-paid RTT2 prefix, tail, Q and scores=Q.T@left_young.
def _rsc_fixed16(prefix, tail, Q, scores, young_right, young_lam,
                 live_indices, response_rows, state, old_inner, cov32, fm, hold):
    if fm is None:
        return tail
    m, c15 = int(prefix.shape[0]), int(prefix.shape[1])
    c2 = int(young_right.shape[0]); count = min(16, c15)
    if not count or c15 == c2 or not int(Q.shape[1]):
        return tail
    # Immutable, deterministic midpoint columns; no data-dependent sampling.
    ids = hold(fnp.asarray(tuple(((2*j+1)*c15)//(2*count) for j in range(count)), dtype=fnp.int64))
    sampled_live = hold(live_indices[ids])
    P = hold(prefix[:, ids])
    if int(state['columns']):
        assert old_inner is not None
        U = hold(_xsg_rows(5119, state['pre_basis'], sampled_live))
        P = hold(P - hold(old_inner @ hold(U.T)))
    Y = hold(hold(young_right[ids]) * young_lam)
    residual = hold(P - hold(Q @ hold(scores @ hold(Y.T))))
    row_factor = hold(fm[response_rows])
    rr = hold(response_rows[:, None])
    # A direct two-dimensional gather creates only m x sample / m x tail.
    Ds = hold(hold(row_factor[:, None]) * hold(cov32[rr, sampled_live]))
    Ds = hold(Ds - hold(Q @ hold(hold(Q.T) @ Ds)))
    numerator = hold(fnp.sum(hold(residual * Ds), dtype=fnp.float64))
    denominator = hold(fnp.sum(hold(Ds * Ds), dtype=fnp.float64))
    alpha = hold(numerator / hold(fnp.maximum(denominator, 1e-30)))
    # Keep alpha remote. All-zero fm gives numerator zero and therefore alpha zero.
    alpha = hold(fnp.where(hold(fnp.isfinite(alpha)), alpha, 0.0))
    alpha = hold(alpha.astype(fnp.float32))
    tail_live = hold(live_indices[c15:c2])
    Dt = hold(hold(row_factor[:, None]) * hold(cov32[rr, tail_live]))
    Dt = hold(Dt - hold(Q @ hold(hold(Q.T) @ Dt)))
    return hold(tail + hold(alpha * Dt))

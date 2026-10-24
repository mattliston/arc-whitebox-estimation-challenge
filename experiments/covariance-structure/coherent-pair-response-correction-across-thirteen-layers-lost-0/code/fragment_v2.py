# Fixed complete span of existing initial and power-refined tail directions.
# Prefix, old-tier content, rosters, and all physical diagonal terms are unchanged.
_RTP4_PARENT_PREDICT = Estimator.predict
_RTP4_STATS = dict(reads=[], retired=True)

def _rtp_refine(prefix, Q, inner, old_right, hold):
    rank = int(Q.shape[1]); assert rank in (1, 2)
    # Exactly the existing power product and old-tier subtraction order.
    V = hold(prefix.T @ Q)
    if inner is not None:
        assert old_right is not None and tuple(inner.shape) == (int(prefix.shape[0]), int(old_right.shape[1]))
        V = hold(V - hold(old_right @ hold(inner.T @ Q)))
    else:
        assert old_right is None
    U = hold(prefix @ V)
    if inner is not None:
        U = hold(U - hold(inner @ hold(old_right.T @ V)))
    # Normalize before the fixed rank test: every input column is unit-scale.
    m = int(Q.shape[0]); wide = hold(U.astype(fnp.float64))
    wideT = hold(wide.T)
    norms = hold(fnp.sqrt(hold(fnp.einsum('ij,ij->i', wideT, wideT, optimize=False))))
    normalized = hold(wide / hold(fnp.maximum(norms, 1e-300)))
    joint = hold(fnp.concatenate((hold(Q.astype(fnp.float64)), normalized), axis=1))
    qjoint, rjoint = fnp.linalg.qr(joint, mode='reduced')
    qjoint, rjoint = hold(qjoint), hold(rjoint)
    rvalues = rjoint.tolist()
    width = 2 * rank
    diagonals = [abs(float(rvalues[j][j])) for j in range(width)]
    threshold = 100.0 * (2.0 ** -23) * _rc_math.sqrt(m)
    expanded = all(_rc_math.isfinite(x) and x > threshold for x in diagonals)
    if expanded:
        answer = hold(qjoint.astype(fnp.float32))
        qr_rank, fallback = width, False
    else:
        # Any near dependence uses the original refinement, never an arbitrary
        # QR nullspace column. Reuses alreadycomputed U; no repeated power step.
        second = hold(U[:, 1]) if rank == 2 else hold(fnp.zeros(int(U.shape[0]), dtype=fnp.float32))
        fresh = hold(_rtp_qr_basis(hold(U[:, 0]), second, hold))
        qr_rank = int(fresh.shape[1]); fallback = qr_rank < rank
        answer = Q if fallback else fresh
    _RTP4_STATS['reads'].append(dict(layer=int(_RS_LAYER[0]), initial_rank=rank,
        output_rank=int(answer.shape[1]), expanded=expanded,
        min_joint_rdiag=min(diagonals), threshold=threshold))
    return answer, qr_rank, fallback

def _rtp4_predict(self, mlp, budget):
    _RTP4_STATS.update(reads=[], retired=False)
    try:
        return _RTP4_PARENT_PREDICT(self, mlp, budget)
    finally:
        _RTP4_STATS['retired'] = True

Estimator.predict = _rtp4_predict

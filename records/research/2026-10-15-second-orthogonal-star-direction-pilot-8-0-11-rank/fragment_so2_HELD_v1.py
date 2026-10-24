# Fixed second row-sum-orthogonal star direction; no learned or fitted scale.
_SO2_PARENT_FACTORS = _ss_factors
_SO2_PARENT_PRODUCT = _astra_star_scaled_product
_SO2_ACTIVE = False
_SO2_STATS = dict(layers=[], fallback=[], retired=True)

def _so2_extra(A, rows, z, norm, wt, wt2, beta, hold):
    n = int(A.shape[0])
    # Two projections suppress FP32 loss of orthogonality; threshold is the
    # dimension-scaled machine-roundoff bound, not an outcome-selected dial.
    v = hold(A @ z)
    scale = float(hold(fnp.sum(hold(v * v), dtype=fnp.float64)))
    q = hold(v - hold(rows * hold(hold(fnp.sum(hold(rows * v))) / norm)))
    q = hold(q - hold(rows * hold(hold(fnp.sum(hold(rows * q))) / norm)))
    qnorm = hold(fnp.sum(hold(q * q), dtype=fnp.float64))
    size = float(qnorm)
    if not (scale > 0 and size > scale * (n * 1.1920928955078125e-7)**2
            and float('inf') > scale and float('inf') > size):
        return None
    q = hold(q / (size**0.5))
    r = hold(q @ A)
    # q^T A 1=q^T rows=0. Remove only its roundoff residual explicitly.
    r = hold(r - hold(hold(fnp.sum(r)) / n))
    return hold(wt @ q)[:, None], hold(hold(wt2 @ r) * beta)[:, None]

def _ss_factors(wt, wt2, A32, hold):
    if not _SO2_ACTIVE:
        return _SO2_PARENT_FACTORS(wt, wt2, A32, hold)
    index = int(_PHYS_LAYER[0])
    assert 0 <= index < 15 and _SS_RANK[index] == -1
    beta = float(_SS_BETA[index])
    assert beta == 0.9105267079535775
    # Original final d401 factor body, unchanged through P1/T2.
    rows = hold(fnp.sum(A32, axis=1))
    P1 = hold(wt @ rows)[:, None]
    z = hold(rows @ A32)
    norm = hold(fnp.sum(hold(rows * rows)))
    mixed = hold(hold(hold(z / norm) * beta) + (1.0 - beta) / int(A32.shape[0]))
    T2 = hold(wt2 @ mixed)[:, None]
    extra = _so2_extra(A32, rows, z, norm, wt, wt2, beta, hold)
    _SO2_STATS['layers'].append(index + 1)
    _SO2_STATS['fallback'].append(extra is None)
    if extra is None:
        return P1, T2
    return hold(fnp.concatenate((P1, extra[0]), axis=1)), hold(fnp.concatenate((T2, extra[1]), axis=1))

def _astra_star_scaled_product(T2, P1, hold):
    if not _SO2_ACTIVE or int(T2.shape[1]) == 1:
        return _SO2_PARENT_PRODUCT(T2, P1, hold)
    assert int(T2.shape[1]) == int(P1.shape[1]) == 2
    # Preserve parent's original rank-one multiplication and /3 placement.
    base = hold(_tsr_mm(hold(T2[:, :1] / 3), P1[:, :1].T))
    correction = hold(_tsr_mm(hold(T2[:, 1:] / 3), P1[:, 1:].T))
    return base + correction

_SO2_PARENT_PREDICT = _predict_binary
def _predict_binary(*args, **kwargs):
    global _SO2_ACTIVE
    assert not _SO2_ACTIVE
    _SO2_ACTIVE = True
    _SO2_STATS['layers'].clear()
    _SO2_STATS['fallback'].clear()
    _SO2_STATS['retired'] = False
    try:
        return _SO2_PARENT_PREDICT(*args, **kwargs)
    finally:
        _SO2_ACTIVE = False
        _SO2_STATS['retired'] = True

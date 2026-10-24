# [astra] Fixed L13-only selection membership by signed-pair tensor energy.
# Keep both original columns, original lambda-squared weights and kc1536.
# No matrix is retained in global state or in the caller's E2 lifetime owner.
_O18_JST_STATS = dict(calls=0, layers=[], selectedcols=[])

def _o18_jst_columns(layer, J, weight, column_score, state, start):
    if layer != 13:
        raise RuntimeError('JST selector is restricted to L13')
    first, last = _JOIN_SCHEDULE[layer]
    plus, minus = [], []
    expected = 0
    for birth in range(first, last + 1):
        absolute, width = state['cohorts'][birth]
        offset = int(absolute) - int(start)
        width = int(width)
        if offset != expected or width <= 0 or width % 2:
            raise RuntimeError('JST requires contiguous complete signed cohorts')
        half = width // 2
        plus.extend(range(offset, offset + half))
        minus.extend(range(offset + half, offset + width))
        expected += width
    if expected != int(J.shape[1]) or int(weight.shape[0]) != expected:
        raise RuntimeError('JST signed cohort ownership mismatch')
    ip = fnp.asarray(plus, dtype=fnp.int64)
    im = fnp.asarray(minus, dtype=fnp.int64)
    # Scalar-vector norms from the parent are float32 and can make nearly
    # cancelling pair scores negative. Recompute these three reductions in
    # float64. The two matrix temporaries die before selection and return.
    P = J[:, ip].astype(fnp.float64)
    M = J[:, im].astype(fnp.float64)
    p2 = fnp.sum(P * P, axis=0)
    m2 = fnp.sum(M * M, axis=0)
    cross = fnp.sum(P * M, axis=0)
    del P, M
    score = weight[ip].astype(fnp.float64) * (p2 * p2 * p2 + m2 * m2 * m2 - 2.0 * cross * cross * cross)
    score = fnp.maximum(score, 0.0)
    keep = fnp.argsort(-score, kind='stable')[:_JSK_KC[layer] // 2]
    selected = fnp.sort(fnp.concatenate([ip[keep], im[keep]]))
    _O18_JST_STATS['calls'] += 1
    _O18_JST_STATS['layers'].append(int(layer))
    _O18_JST_STATS['selectedcols'].append(int(selected.shape[0]))
    return selected

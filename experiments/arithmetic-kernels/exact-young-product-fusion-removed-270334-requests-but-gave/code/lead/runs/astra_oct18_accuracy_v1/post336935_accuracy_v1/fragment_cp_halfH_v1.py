# [astra] Fixed CP-content paired sketch. Basis selection only; full signed
# factors, old contribution, output rank and exact-diagonal row choice unchanged.
# Keep kc/2 pair centers and kc/4 correlated difference directions. Select the
# latter by lambda^2 ||center||^4 ||difference||^2, without endpoint labels.
def _o18_cph_columns(layer, J, weight, column_score, state, start):
    first, last = _JOIN_SCHEDULE[layer]
    plus, minus = [], []
    expected = 0
    for birth in range(first, last + 1):
        absolute, width = state['cohorts'][birth]
        offset, width = int(absolute) - int(start), int(width)
        if offset != expected or width <= 0 or width % 2:
            raise RuntimeError('CP-H sketch requires contiguous complete cohorts')
        half = width // 2
        plus.extend(range(offset, offset + half))
        minus.extend(range(offset + half, offset + width))
        expected += width
    if expected != int(J.shape[1]) or int(weight.shape[0]) != expected:
        raise RuntimeError('CP-H sketch cohort ownership mismatch')
    ip, im = fnp.asarray(plus, dtype=fnp.int64), fnp.asarray(minus, dtype=fnp.int64)
    pairscore = column_score[ip] + column_score[im]
    cm = fnp.argsort(-pairscore.astype(fnp.float64), kind='stable')[:_JSK_KC[layer] // 2]
    P, N = J[:, ip], J[:, im]
    common, difference = (P + N) * 0.5, (P - N) * 0.5
    pw = weight[ip]
    hs = fnp.sum(difference * difference, axis=0) * pw
    md = common.astype(fnp.float64)
    mn = fnp.sum(md * md, axis=0)
    cp_score = hs.astype(fnp.float64) * (mn * mn)
    hh = fnp.argsort(-cp_score, kind='stable')[:_JSK_KC[layer] // 4]
    # Parent complete-pair H trace is preserved. A zero difference span needs
    # no correction; maximum only guards degenerate all-identical cohorts.
    target = fnp.sum(hs[cm]).astype(fnp.float64)
    selected = fnp.sum(hs[hh]).astype(fnp.float64)
    multiplier = (2.0 * target / fnp.maximum(selected, 1e-30)).astype(fnp.float32)
    columns = fnp.concatenate((common[:, cm], difference[:, hh]), axis=1)
    weights = fnp.concatenate((2.0 * pw[cm], multiplier * pw[hh]), axis=0)
    return columns, weights

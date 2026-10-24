# [astra] Selection-only signed-pair mean sketch. Full Fj, signed factors,
# reconstruction, pair law, read and exact diagonal row selection are unchanged.
# For an equal-weight pair P,Q: w(PP'+QQ') = 2w MM' + (w/2) DD'.
# Only the second term is omitted from the basis-selection sketch.
_ASTRA_JSM_STATS = dict(calls=0, parent_columns=0, mean_columns=0)

def _astra_jsm_columns(layer, J, weight, column_score, state, start):
    first, last = _JOIN_SCHEDULE[layer]
    plus, minus = [], []
    expected = 0
    for birth in range(first, last + 1):
        absolute, width = state['cohorts'][birth]
        offset = int(absolute) - int(start)
        width = int(width)
        if offset != expected or width <= 0 or width % 2:
            raise RuntimeError('pair-mean join sketch requires contiguous complete cohorts')
        half = width // 2
        plus.extend(range(offset, offset + half))
        minus.extend(range(offset + half, offset + width))
        expected += width
    if expected != int(J.shape[1]) or int(weight.shape[0]) != expected:
        raise RuntimeError('pair-mean join sketch ownership mismatch')
    ip = fnp.asarray(plus, dtype=fnp.int64)
    im = fnp.asarray(minus, dtype=fnp.int64)
    # Select complete pairs by the sum of the parent's two weighted energies.
    score = column_score[ip] + column_score[im]
    keep = fnp.argsort(-(score.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer] // 2]
    ip, im = ip[keep], im[keep]
    common = (J[:, ip] + J[:, im]) * 0.5
    common_weight = weight[ip] * 2.0
    _ASTRA_JSM_STATS['calls'] += 1
    _ASTRA_JSM_STATS['parent_columns'] += min(_JSK_KC[layer], expected)
    _ASTRA_JSM_STATS['mean_columns'] += int(common.shape[1])
    return common, common_weight

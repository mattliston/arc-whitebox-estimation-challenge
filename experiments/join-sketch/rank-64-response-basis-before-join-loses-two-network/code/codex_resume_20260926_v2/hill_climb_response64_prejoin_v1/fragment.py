

# Fill only response padding already paid by the same prejoin KS read.
def _response64_prejoin(layer, score, floor, live_indices, lam, state):
    indices, mask = select(score, floor)
    if not (4 <= layer <= 6) or int(state['columns']) != 0:
        return indices, mask
    m, k, n = int(indices.shape[0]), int(lam.shape[0]), int(live_indices.shape[0])
    filled = (m + 63) // 64 * 64
    if not (0 < m < filled <= n <= 1024 and k > 1024):
        return indices, mask
    if not (_KS_ON and _AB_ON and _WINOGRAD and _SCRATCH is not None):
        return indices, mask
    # Either READ7 shape would bypass the ordinary multi-chunk KS dispatcher.
    if (m, k, n) in _R7_SHAPES or (filled, k, n) in _R7_SHAPES:
        return indices, mask
    current = _ks_plan(m, k, n)
    expanded = _ks_plan(filled, k, n)
    if current is None or len(current[2]) < 2 or current != expanded:
        return indices, mask
    keep = []
    negative = -score
    order = fnp.argsort(negative, kind='stable')
    keep.extend((negative, order))
    indices = order[:filled]
    mask = fnp.zeros(len(score), dtype=fnp.bool_)
    keep.append(fnp.put(mask, indices, True))
    return indices, mask

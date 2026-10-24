"""Same-count physical distinct-third birth ranking, fixed on exact S32."""
_BDS_STATS = dict(calls=0, full_or_empty=0, reranked=0, changed_units=0,
                  selected_total=0, roundoff_clamps=0)


def _birth_distinct_select(score, floor, birth, g, alive, Cpre):
    old_indices, old_mask = select(score, floor)
    count, n = int(old_indices.shape[0]), int(score.shape[0])
    _BDS_STATS['calls'] += 1
    _BDS_STATS['selected_total'] += count
    if count == 0 or count == n:
        _BDS_STATS['full_or_empty'] += 1
        return old_indices, old_mask
    if _E2_LAYER.get('VV_for') is not Cpre:
        raise RuntimeError('birth strength: cached covariance owner changed')
    CC = _E2_LAYER['CC']
    if CC.shape != (n, n) or str(CC.dtype) != 'float64':
        raise RuntimeError('birth strength: cached squared covariance layout')
    gate = g * alive
    p2 = CC * (gate * gate)[:, None]
    fnp.fill_diagonal(p2, 0)
    q = fnp.sum(p2, axis=0, dtype=fnp.float64)
    fourth = fnp.sum(p2 * p2, axis=0, dtype=fnp.float64)
    qq = q * q
    distinct = qq - fourth
    # Exact nonnegative identity for squared entries; only roundoff may clamp.
    tolerance = (32 * n * 2.220446049250313e-16) * (qq + fourth)
    if not fnp.all(fnp.isfinite(distinct)) or not fnp.all(distinct >= -tolerance):
        raise RuntimeError('birth strength: invalid distinct-pair norm')
    _BDS_STATS['roundoff_clamps'] += int(fnp.nonzero(distinct < 0)[0].shape[0])
    coefficient = birth * alive
    strength = (coefficient * coefficient) * fnp.maximum(distinct, 0.0)
    if not fnp.all(fnp.isfinite(strength)):
        raise RuntimeError('birth strength: nonfinite score')
    order = fnp.argsort(-strength, kind='stable')
    chosen = fnp.sort(order[:count])
    # Ascending original indices restore the parent's stable tie convention.
    indices = chosen[fnp.argsort(-score[chosen], kind='stable')]
    mask = fnp.zeros(n, dtype=fnp.bool_)
    fnp.put(mask, indices, True)
    _BDS_STATS['reranked'] += 1
    _BDS_STATS['changed_units'] += int(fnp.nonzero(mask != old_mask)[0].shape[0])
    return indices, mask

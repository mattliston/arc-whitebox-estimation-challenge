# Fixed firstjoin common-mode residual row selector; all later algebra unchanged.
_JRS_ON = True
_JRS_ACTIVE = [False]
_JRS_STATS = dict(calls=0, records=[], retired=True)

def _jrs_select(J, weight, diagonal, rank):
    hold = _E2Hold()
    m, k = int(J.shape[0]), int(J.shape[1])
    assert 0 < rank <= m and tuple(weight.shape) == (k,) and tuple(diagonal.shape) == (m,)
    order = hold(fnp.argsort(hold(-hold(diagonal.astype(fnp.float64))), kind='stable'))
    original = hold(fnp.sort(hold(order[:rank])))
    record = dict(layer=7, m=m, columns=k, rank=int(rank), original_anchor=True,
                  fallback=False, reason=None)
    _JRS_STATS['calls'] += 1
    _JRS_STATS['records'].append(record)
    c = hold(fnp.sum(J, axis=0))
    y = hold(J @ hold(weight*c))
    yy = hold(y.astype(fnp.float64))
    s = hold(fnp.sum(yy))
    if not bool(hold(fnp.isfinite(s))) or not bool(hold(s > 0)):
        record.update(fallback=True, reason='nonpositive_or_nonfinite_common_energy')
        return original
    score = hold(hold(diagonal.astype(fnp.float64)) - hold(hold(yy*yy)/s))
    if not bool(hold(fnp.all(hold(fnp.isfinite(score))))):
        record.update(fallback=True, reason='nonfinite_residual_score')
        return original
    anchor = hold(order[:1])
    hold(fnp.put(score, anchor, -fnp.inf))
    other = hold(hold(fnp.argsort(hold(-score), kind='stable'))[:rank-1])
    return fnp.sort(hold(fnp.concatenate((anchor, other))))

_JRS_PARENT_PREDICT = Estimator.predict
def _jrs_predict(self, *args, **kwargs):
    _JRS_STATS.clear()
    _JRS_STATS.update(calls=0, records=[], retired=False)
    _JRS_ACTIVE[0] = bool(_JRS_ON)
    try:
        return _JRS_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _JRS_ACTIVE[0] = False
        _JRS_STATS['retired'] = True
Estimator.predict = _jrs_predict


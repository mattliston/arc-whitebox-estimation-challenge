# Fixed F32 covariance centering for hidden layers1..14; original dispatch and STAR.
_CCT_ON = True
_CCT_ACTIVE = [False]
_CCT_STATS = dict(calls=0, layers=[], rows=[], retired=True)

def _cct_center(C32, w, hold):
    scope = _W5_PORT_SCOPE
    layer = None if scope is None else scope['layer']
    if not (_CCT_ON and _CCT_ACTIVE[0] and type(layer) is int and 1 <= layer <= 14 and int(C32.shape[0]) > 0):
        return C32, None
    r = hold(fnp.mean(C32, axis=1))
    c = hold(fnp.mean(r))
    row, col = hold(r[:, None]), hold(r[None, :])
    C0 = hold(hold(hold(C32 - row) - col) + c)
    return C0, (w, r, c, layer, int(C32.shape[0]))

def _cct_restore(cov, aux, hold):
    if aux is None:
        return cov
    w, r, c, layer, rows = aux
    u = hold(fnp.sum(w, axis=0))
    v = hold(r @ w)
    t = hold(v - hold(hold(c * .5) * u))
    R = hold(hold(t[:, None]) * hold(u[None, :]))
    # Form the symmetric correction first, preserving exact matrix symmetry.
    correction = hold(R + hold(R.T))
    result = cov + correction
    _CCT_STATS['calls'] += 1
    _CCT_STATS['layers'].append(layer)
    _CCT_STATS['rows'].append(rows)
    return result

_CCT_PARENT_PREDICT = Estimator.predict
def _cct_predict(self, *args, **kwargs):
    if _CCT_ACTIVE[0]:
        raise RuntimeError('CCT does not permit nested prediction')
    _CCT_STATS.clear()
    _CCT_STATS.update(calls=0, layers=[], rows=[], retired=False)
    _CCT_ACTIVE[0] = bool(_CCT_ON)
    try:
        return _CCT_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _CCT_ACTIVE[0] = False
        _CCT_STATS['retired'] = True
Estimator.predict = _cct_predict


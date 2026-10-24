# Fixed L1--3 exact-real CP read centering. No old-factor or tail change.
_O18_CPC_PARENT_READ = _RTT_PARENT_READ
_O18_CPC_STATS = dict(calls=0, shapes=[])

def _o18_cpc_read(left, young_right, lam, live_indices, state):
    layer = _RS_LAYER[0]
    if layer not in (1, 2, 3):
        return _O18_CPC_PARENT_READ(left, young_right, lam, live_indices, state)
    m, k = map(int, left.shape)
    n = int(young_right.shape[0])
    assert int(state['columns']) == 0
    assert tuple(young_right.shape) == (n, k) and tuple(lam.shape) == (k,)
    assert left.dtype == young_right.dtype == lam.dtype == 'float32'
    hold = _E2Hold()
    # Average the unweighted operand, before the parent's signed-lambda placement.
    u = hold(hold(fnp.mean(left, axis=1, keepdims=True,
                           dtype=fnp.float64)).astype(fnp.float32))
    centered = hold(left - u)
    product = hold(_O18_CPC_PARENT_READ(centered, young_right, lam,
                                       live_indices, state))
    # Matvec is outside all tree scratch; fresh add owns the returned plane.
    columns = hold(fnp.matmul(young_right, lam))
    correction = hold(u * hold(columns[None, :]))
    result = fnp.add(product, correction)
    _O18_CPC_STATS['calls'] += 1
    _O18_CPC_STATS['shapes'].append((int(layer), m, k, n))
    return result

_RTT_PARENT_READ = _o18_cpc_read
_O18_CPC_PARENT_PREDICT = Estimator.predict

def _o18_cpc_predict(self, *args, **kwargs):
    _O18_CPC_STATS.update(calls=0, shapes=[])
    return _O18_CPC_PARENT_PREDICT(self, *args, **kwargs)

Estimator.predict = _o18_cpc_predict

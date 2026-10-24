# Fixed L2 diagnostic: preserve the current exact prefix, restore only its tail.
_O18_R2_PARENT_READ = _RC_PARENT_READ
_O18_R2_PARENT_PREDICT = Estimator.predict
_O18_R2_STATS = dict(restored=[], retired=True)

def _o18_r2_read(left, young_right, lam, live_indices, state):
    if int(_RS_LAYER[0]) != 2 or not (_RTT_ACTIVE and _RTP_ON):
        return _O18_R2_PARENT_READ(left, young_right, lam, live_indices, state)
    assert _RE22_PENDING[0] is None and int(state['columns']) == 0
    m, k, c2 = int(left.shape[0]), int(young_right.shape[1]), int(young_right.shape[0])
    assert tuple(left.shape) == (m, k) and tuple(lam.shape) == (k,)
    assert tuple(live_indices.shape) == (c2,)
    floor = .5 * _rc_math.erfc(1.5 / _rc_math.sqrt(2.0))
    c15 = min(c2, (int(fnp.nonzero(_E2_LAYER['g'] >= floor)[0].shape[0]) + 31) // 32 * 32)
    _O18_R2_STATS['restored'].append(dict(layer=2, m=m, k=k, c15=c15, c2=c2, old_columns=0))
    if c15 == c2:
        return _RTT_PARENT_READ(left, young_right, lam, live_indices, state)
    assert 0 < c15 < c2
    hold = _E2Hold()
    prefix = hold(_RTT_PARENT_READ(left, hold(young_right[:c15]), lam,
                                  hold(live_indices[:c15]), state))
    # A copy makes ownership independent of any parent's scratch implementation.
    prefix = hold(fnp.copy(prefix))
    weighted_tail = hold(hold(young_right[c15:c2]) * lam)
    tail = hold(fnp.matmul(left, hold(weighted_tail.T)))
    assert tuple(prefix.shape) == (m, c15) and tuple(tail.shape) == (m, c2-c15)
    assert _RE22_PENDING[0] is None
    # Complete plane energy reaches features directly; no RE22 extrapolation.
    return fnp.concatenate((prefix, tail), axis=1)

_RC_PARENT_READ = _o18_r2_read

def _o18_r2_predict(self, mlp, budget):
    _O18_R2_STATS.update(restored=[], retired=False)
    try:
        result = _O18_R2_PARENT_PREDICT(self, mlp, budget)
        assert len(_O18_R2_STATS['restored']) == 1
        return result
    finally:
        _O18_R2_STATS['retired'] = True

Estimator.predict = _o18_r2_predict

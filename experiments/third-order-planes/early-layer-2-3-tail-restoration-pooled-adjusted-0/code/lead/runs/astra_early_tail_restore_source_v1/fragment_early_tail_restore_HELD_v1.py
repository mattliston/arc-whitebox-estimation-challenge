# Fixed zero-based L2/L3 full young-read restoration inside the unchanged c2 mask.
# All other layers retain the submitted QR/refined RTT2 computation.
assert _RC_PARENT_READ is _rtt_read
assert _RTT_PARENT_READ is _yr7_read and _RC_COLZ == 2.0
_ETR_PARENT_READ = _rtt_read
_ETR_PARENT_PREDICT = Estimator.predict
_ETR_ON = True
_ETR_ACTIVE = False
_ETR_LAYERS = (2, 3)
_ETR_STATS = dict(restored=[], shapes=[], retired=True)


def _rtt_read(left, young_right, lam, live_indices, state):
    if not (_ETR_ON and _ETR_ACTIVE and _RTT_ACTIVE
            and _RS_LAYER[0] in _ETR_LAYERS):
        return _ETR_PARENT_READ(left, young_right, lam, live_indices, state)
    layer = int(_RS_LAYER[0])
    assert _RTP_ON and _RTT_CONTEXT[0] is None
    assert int(state['columns']) == 0  # The first old-bank join is L7.
    m, k, c2 = int(left.shape[0]), int(young_right.shape[1]), int(young_right.shape[0])
    assert tuple(left.shape) == (m, k) and tuple(lam.shape) == (k,)
    assert tuple(live_indices.shape) == (c2,)
    # Use the exact existing full-c2 route, including its natural D6/D7 policy.
    # The outer c2 roster/padding and subsequent _np_forms mask are untouched.
    result = _RTT_PARENT_READ(left, young_right, lam, live_indices, state)
    assert tuple(result.shape) == (m, c2)
    _RTT_STATS['reads'].append(dict(layer=layer, m=m, k=k, c15=c2, c2=c2,
        old_rank=0, changed=False, basis_rank=None, old_inner_calls=0,
        forced_d7=False, refine_rank_in=None, refine_rank_out=None,
        refine_qr_rank=None, refine_fallback=None, refine_old_rank=None))
    _ETR_STATS['restored'].append(layer)
    _ETR_STATS['shapes'].append([layer, m, k, c2])
    return result


_RC_PARENT_READ = _rtt_read


def _etr_predict(self, *args, **kwargs):
    global _ETR_ACTIVE
    assert not _ETR_ACTIVE
    _ETR_STATS.clear()
    _ETR_STATS.update(restored=[], shapes=[], retired=False)
    _ETR_ACTIVE = bool(_ETR_ON)
    try:
        return _ETR_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _ETR_ACTIVE = False
        _ETR_STATS['retired'] = True


Estimator.predict = _etr_predict

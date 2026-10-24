
# Observational L7--14 tap: Python references only, no FNP operations.
_NLR_CAPTURE = {}
_NLR_ACTIVE = False
_NLR_PARENT_FEATURES = features
_NLR_PARENT_GNN = gnn_step
_NLR_PARENT_FHM = _astra_fhm_effective
_NLR_PARENT_PREDICT = Estimator.predict

def features(l, m, C, K21, aux, W):
    answer = _NLR_PARENT_FEATURES(l, m, C, K21, aux, W)
    if _NLR_ACTIVE and 7 <= l <= 14:
        assert l not in _NLR_CAPTURE
        _NLR_CAPTURE[l] = dict(host=m)
    return answer

def gnn_step(l, x, sigma, W, state, P):
    answer = _NLR_PARENT_GNN(l, x, sigma, W, state, P)
    if _NLR_ACTIVE and 7 <= l <= 14:
        entry = _NLR_CAPTURE[l]
        assert set(entry) == {'host'}
        cache = _E2_GNN_STATE[int(sigma.shape[0])]['layers'][l]
        entry.update(features30=x, state64=answer[1], sigma=sigma,
                     delta=answer[0], head=cache['head'], head_b=cache['head_b'],
                     rms=cache['rms'], gain=cache['gain'])
    return answer

def _astra_fhm_effective(layer, mean, delta):
    answer = _NLR_PARENT_FHM(layer, mean, delta)
    if _NLR_ACTIVE and 7 <= layer <= 14:
        entry = _NLR_CAPTURE[layer]
        assert entry['host'] is mean and 'effective_delta' not in entry
        entry['effective_delta'] = answer
    return answer

def _nlr_predict(self, *args, **kwargs):
    global _NLR_ACTIVE
    assert not _NLR_ACTIVE
    _NLR_CAPTURE.clear()
    _NLR_ACTIVE = True
    try:
        return _NLR_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _NLR_ACTIVE = False
Estimator.predict = _nlr_predict

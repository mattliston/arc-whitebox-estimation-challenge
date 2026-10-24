
# Diagnostic-only synchronous host observer. No added FNP arithmetic.
_SH_OBSERVER = None
_SH_READ = _RC_PARENT_READ
_SH_FEATURES = features
_SH_GNN = gnn_step
_SH_FHM = _astra_fhm_effective

def _sh_read(left, right, lam, indices, state):
    answer = _SH_READ(left, right, lam, indices, state)
    layer = int(_RS_LAYER[0])
    if _SH_OBSERVER is not None and layer in (4, 5, 6):
        assert int(state['columns']) == 0
        _SH_OBSERVER('read', layer, dict(left=left, right=right, lam=lam,
            rows=_E2_LAYER['ecut_rows'], cols=indices, answer=answer,
            c15=_RTT_STATS['reads'][-1]['c15']))
    return answer

def features(l, m, C, K21, aux, W):
    answer = _SH_FEATURES(l, m, C, K21, aux, W)
    if _SH_OBSERVER is not None and l in (4, 5, 6):
        _SH_OBSERVER('features', l, dict(features30=answer, host=m, C=C))
    return answer

def gnn_step(l, x, sigma, W, state, P):
    answer = _SH_GNN(l, x, sigma, W, state, P)
    if _SH_OBSERVER is not None and l in (4, 5, 6):
        _SH_OBSERVER('gnn', l, dict(delta=answer[0], state64=answer[1], sigma=sigma))
    return answer

def _astra_fhm_effective(layer, mean, delta):
    answer = _SH_FHM(layer, mean, delta)
    if _SH_OBSERVER is not None and layer in (4, 5, 6):
        _SH_OBSERVER('fhm', layer, dict(host=mean, effective_delta=answer))
    return answer

_RC_PARENT_READ = _sh_read

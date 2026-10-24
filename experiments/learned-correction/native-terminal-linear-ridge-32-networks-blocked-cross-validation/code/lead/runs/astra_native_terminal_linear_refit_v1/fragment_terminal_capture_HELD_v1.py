
# Observational terminal-only tap: Python strong references, zero FNP operations.
_TLR_CAPTURE = {}
_TLR_ACTIVE = False
_TLR_PARENT_GNN = gnn_step
_TLR_PARENT_FEATURES = final_scalar_features
_TLR_PARENT_PREDICT = Estimator.predict

def final_scalar_features(l,m,aux,W):
    answer = _TLR_PARENT_FEATURES(l,m,aux,W)
    if _TLR_ACTIVE and l == 15:
        assert not _TLR_CAPTURE
        _TLR_CAPTURE['host'] = m
    return answer

def gnn_step(l,x,sigma,W,state,P):
    answer = _TLR_PARENT_GNN(l,x,sigma,W,state,P)
    if _TLR_ACTIVE and l == 15:
        assert set(_TLR_CAPTURE) == {'host'}
        cache = _E2_GNN_STATE[int(sigma.shape[0])]['layers'][15]
        _TLR_CAPTURE.update(features30=x, state64=answer[1], sigma=sigma,
                            delta=answer[0], head=cache['head'], head_b=cache['head_b'],
                            rms=cache['rms'], gain=cache['gain'])
    return answer

def _tlr_predict(self,*args,**kwargs):
    global _TLR_ACTIVE
    assert not _TLR_ACTIVE
    _TLR_CAPTURE.clear(); _TLR_ACTIVE = True
    try:
        return _TLR_PARENT_PREDICT(self,*args,**kwargs)
    finally:
        _TLR_ACTIVE = False
Estimator.predict = _tlr_predict

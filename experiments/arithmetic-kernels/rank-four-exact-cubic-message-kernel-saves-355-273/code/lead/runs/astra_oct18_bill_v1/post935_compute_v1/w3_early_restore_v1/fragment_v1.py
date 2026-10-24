# Fixed diagnostic: restore exact W^3 messages at L1--3 only.
_O18_W3E_PARENT_MESSAGE = _gop2_message_product
_O18_W3E_PARENT_GNN = gnn_step
_O18_W3E_PARENT_PREDICT = Estimator.predict
_O18_W3E_LAYER = [None]
_O18_W3E_STATS = dict(restored_layers=[], retired=True)

def _gop2_message_product(T, X):
    layer = _O18_W3E_LAYER[0]
    if layer not in (1, 2, 3):
        return _O18_W3E_PARENT_MESSAGE(T, X)
    n = int(X.shape[1]); assert X.shape == (3, n, 16)
    out = fnp.empty((3, n, 16), dtype=fnp.float32)
    fnp.matmul(T[0:1], X[0:1], out=out[0:1])
    fnp.copyto(out[1:2], _gop2_contract(T[1:2], X[1:2]))
    fnp.copyto(out[2], fnp.matmul(T[2], X[2]))
    _GOP2_STATS['message_misses'] += 1
    _O18_W3E_STATS['restored_layers'].append(layer)
    return out

def gnn_step(l, x, sigma, W, state, P):
    assert _O18_W3E_LAYER[0] is None
    _O18_W3E_LAYER[0] = int(l)
    try:
        return _O18_W3E_PARENT_GNN(l, x, sigma, W, state, P)
    finally:
        _O18_W3E_LAYER[0] = None

def _o18_w3e_predict(self, mlp, budget):
    assert _O18_W3E_LAYER[0] is None
    _O18_W3E_STATS.update(restored_layers=[], retired=False)
    try:
        answer = _O18_W3E_PARENT_PREDICT(self, mlp, budget)
        assert _O18_W3E_STATS['restored_layers'] == [1, 2, 3]
        return answer
    finally:
        _O18_W3E_LAYER[0] = None
        _O18_W3E_STATS['retired'] = True

Estimator.predict = _o18_w3e_predict

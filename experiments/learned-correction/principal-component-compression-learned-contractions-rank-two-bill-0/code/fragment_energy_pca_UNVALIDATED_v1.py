# HELD fixed current-source rank2 E2/E3 energy; new hypothesis, old E23 FAIL unchanged.
_AEP2_ON = True
_AEP2_PARENT_GNN = gnn_step
_AEP2_PARENT_PREDICT = Estimator.predict
assert _AEP2_PARENT_GNN.__code__.co_firstlineno == 729
assert 'gnn_step' not in _TR_CLEAR
_AEP2_ACTIVE = [False]
_AEP2_STATS = dict(calls=0, layers=[], energy_blocks=0, retired=True, value_cache_entries=0)

def _aep2_energy(T,X):
    # E1 original product, E2/E3 fixed rank2 current-source approximation.
    exact=fnp.matmul(T[:1],X[:1])
    Y=X[1:]
    mu=fnp.mean(Y,axis=1)
    Yc=fnp.subtract(Y,mu[:,None,:])
    Y64=Yc.astype(fnp.float64)
    G=fnp.matmul(Y64.transpose(0,2,1),Y64)
    _,V=fnp.linalg.eigh(G)
    V2=V[:,:,-2:].astype(fnp.float32)
    Z=fnp.matmul(Yc,V2)
    ZW=fnp.matmul(T[1:],Z)
    residual=fnp.matmul(ZW,V2.transpose(0,2,1))
    mass=fnp.sum(T[1:],axis=2)
    approx=fnp.add(residual,fnp.multiply(mass[:,:,None],mu[:,None,:]))
    return fnp.concatenate((exact,approx),axis=0)


def _aep2_gnn_step(l,x,sigma,W,state,P):
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)
    if l<1:
        return (_e2_H(fnp.zeros_like(sigma)), _e2_H(state * 0))
    k = _e2_gnn_consts(P, int(_e2_H(sigma.shape[0])))
    c = _e2_gnn_layer(k, l)
    Hx = k['Hx']
    _e2_hold.append(fnp.clip(((x if str(x.dtype) == 'float32' else x.astype(fnp.float32)) - c['centers']) / c['scales'], -8, 8, out=Hx))
    if l == 15:
        _e2_hold.append(fnp.multiply(Hx, k['mask'], out=Hx))
    M = _e2_messages(k, W, state)
    _e2_hold.append(fnp.power(k['SRC3'], 2, out=k['SQ']))
    E = _aep2_energy(_e2_H(_e2_H(_e2_wpow(W, need6=True))['T135']), k['SQ'])
    _e2_hold.append(fnp.multiply(k['Hg'], M, out=k['HM03']))
    _e2_hold.append(fnp.maximum(E, 0, out=E))
    _e2_hold.append(fnp.log1p(E, out=k['HM36']))
    _e2_hold.append(fnp.clip(k['Hm'], -8, 8, out=k['Hm']))
    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + _e2_H(_hc_bias(l, P['enc0_b']))))   # opus-engineer hc: enc0_b + z u' when the layer's read made the new column
    h = silu(_e2_H(_e2_H(h @ k['enc2_wT']) + P['enc2_b']))
    state = _e2_H(h @ k['enc4_wT']) + P['enc4_b']
    delta = _e2_H(_e2_H(_e2_H(sigma.astype(fnp.float32)) * c['rms']) * _e2_H(_e2_H(state @ c['head']) + c['head_b'])) * c['gain']
    return delta,state


def gnn_step(l, x, sigma, W, state, P):
    if not _AEP2_ON or l < 1:
        return _AEP2_PARENT_GNN(l, x, sigma, W, state, P)
    assert _AEP2_ACTIVE[0] and 1 <= l <= 15
    _AEP2_STATS['calls'] += 1
    _AEP2_STATS['layers'].append(int(l))
    _AEP2_STATS['energy_blocks'] += 2
    return _aep2_gnn_step(l, x, sigma, W, state, P)


def _aep2_predict(self, *args, **kwargs):
    global _AEP2_STATS
    assert not _AEP2_ACTIVE[0]
    _AEP2_STATS = dict(calls=0, layers=[], energy_blocks=0, retired=False, value_cache_entries=0)
    if not _AEP2_ON:
        _AEP2_STATS['retired'] = True
        return _AEP2_PARENT_PREDICT(self, *args, **kwargs)
    _AEP2_ACTIVE[0] = True
    try:
        return _AEP2_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _AEP2_ACTIVE[0] = False
        _AEP2_STATS['retired'] = True

Estimator.predict = _aep2_predict

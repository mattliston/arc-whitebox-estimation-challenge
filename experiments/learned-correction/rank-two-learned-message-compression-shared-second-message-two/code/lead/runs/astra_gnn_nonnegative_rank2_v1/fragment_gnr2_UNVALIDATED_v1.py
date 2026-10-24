# HELD: fixed current-source rank2 E1/E2/E3 and shared M2. No selected dials.
_GNR2_ON = True
_GNR2_PARENT_GNN = gnn_step
_GNR2_PARENT_MESSAGES = _e2_messages
_GNR2_PARENT_PREDICT = Estimator.predict
assert _GNR2_PARENT_GNN.__code__.co_firstlineno == 729
assert _GNR2_PARENT_MESSAGES.__code__.co_firstlineno == 715
assert 'gnn_step' not in _TR_CLEAR and '_e2_messages' not in _TR_CLEAR
_GNR2_ACTIVE = [False]
_GNR2_STATS = dict(calls=0,layers=[],energy_blocks=0,message_misses=0,retired=True,value_cache_entries=0)

def _gnr2_contract(T,X):
    # Exact mean component; rank2 centered current-source component, no value cache.
    mu=fnp.mean(X,axis=1)
    centered=fnp.subtract(X,mu[:,None,:])
    wide=centered.astype(fnp.float64)
    gram=fnp.matmul(wide.transpose(0,2,1),wide)
    _,vectors=fnp.linalg.eigh(gram)
    V=vectors[:,:,-2:].astype(fnp.float32)
    coefficients=fnp.matmul(centered,V)
    projected=fnp.matmul(T,coefficients)
    residual=fnp.matmul(projected,V.transpose(0,2,1))
    mass=fnp.sum(T,axis=2)
    return fnp.add(residual,fnp.multiply(mass[:,:,None],mu[:,None,:]))

def _gnr2_message_product(T,X):
    n=int(X.shape[1]);assert X.shape==(3,n,16)
    out=fnp.empty((3,n,16),dtype=fnp.float32)
    # Original W/W3 products with their original per-item operand layouts.
    fnp.matmul(T[0:3:2],X[0:3:2],out=out[0:3:2])
    fnp.copyto(out[1:2],_gnr2_contract(T[1:2],X[1:2]))
    _GNR2_STATS['message_misses']+=1
    return out

def _gnr2_messages(k, W, state):
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)
    """src = state@msg_w.T + msg_b into the persistent SRC and M = [w.T@src1, w2.T@src2, w3.T@src3] as one batched matmul whose
    per-item operands have the original layouts; the edge source and gnn_step of one layer share the result."""
    got = _E2_MSG.get('v')
    if got is not None and got[0] is state and got[1] is W:
        return got[2]
    _e2_hold.append(fnp.add(state @ k['msg_wT'], k['P']['msg_b'], out=k['SRC']))
    M = _gnr2_message_product(_e2_H(_e2_H(_e2_wpow(W))['T013']), k['SRC3'])
    _E2_MSG['v'] = (state, W, M)
    return M


def _gnr2_gnn_step(l,x,sigma,W,state,P):
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
    E = _gnr2_contract(_e2_H(_e2_H(_e2_wpow(W, need6=True))['T135']), k['SQ'])
    _e2_hold.append(fnp.multiply(k['Hg'], M, out=k['HM03']))
    _e2_hold.append(fnp.maximum(E, 0, out=E))
    _e2_hold.append(fnp.log1p(E, out=k['HM36']))
    _e2_hold.append(fnp.clip(k['Hm'], -8, 8, out=k['Hm']))
    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + _e2_H(_hc_bias(l, P['enc0_b']))))   # opus-engineer hc: enc0_b + z u' when the layer's read made the new column
    h = silu(_e2_H(_e2_H(h @ k['enc2_wT']) + P['enc2_b']))
    state = _e2_H(h @ k['enc4_wT']) + P['enc4_b']
    delta = _e2_H(_e2_H(_e2_H(sigma.astype(fnp.float32)) * c['rms']) * _e2_H(_e2_H(state @ c['head']) + c['head_b'])) * c['gain']
    return delta,state

def _e2_messages(k,W,state):
    if not _GNR2_ON:
        return _GNR2_PARENT_MESSAGES(k,W,state)
    assert _GNR2_ACTIVE[0]
    return _gnr2_messages(k,W,state)

def gnn_step(l,x,sigma,W,state,P):
    if not _GNR2_ON or l<1:
        return _GNR2_PARENT_GNN(l,x,sigma,W,state,P)
    assert _GNR2_ACTIVE[0] and 1<=l<=15
    _GNR2_STATS['calls']+=1
    _GNR2_STATS['layers'].append(int(l))
    _GNR2_STATS['energy_blocks']+=3
    return _gnr2_gnn_step(l,x,sigma,W,state,P)

def _gnr2_predict(self,*args,**kwargs):
    global _GNR2_STATS
    assert not _GNR2_ACTIVE[0]
    _GNR2_STATS=dict(calls=0,layers=[],energy_blocks=0,message_misses=0,retired=False,value_cache_entries=0)
    if not _GNR2_ON:
        _GNR2_STATS['retired']=True
        return _GNR2_PARENT_PREDICT(self,*args,**kwargs)
    _GNR2_ACTIVE[0]=True
    try:
        return _GNR2_PARENT_PREDICT(self,*args,**kwargs)
    finally:
        _GNR2_ACTIVE[0]=False
        _GNR2_STATS['retired']=True

Estimator.predict=_gnr2_predict

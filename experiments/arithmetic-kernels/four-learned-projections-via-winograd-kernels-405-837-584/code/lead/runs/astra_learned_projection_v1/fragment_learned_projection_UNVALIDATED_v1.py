# HELD learned projection fast kernels; real algebra unchanged, float32 rounding changes.
_ALP_ON = True
_ALP_ACTIVE = [False]
_ALP_CACHE = dict(owner=None, values={}, bytes=0)
_ALP_LIMIT = 4 * 1024 * 1024
_ALP_STATS = dict(calls=0, stores=0, hits=0, fallback=0, max_cache_bytes=0, events=[], retired=True)
_ALP_PARENT_MESSAGES = _e2_messages
_ALP_PARENT_GNN = gnn_step
_ALP_PARENT_PREDICT = _predict_binary


def _alp_reset():
    _ALP_CACHE.update(owner=None, values={}, bytes=0)


def _alp_project(a, b, owner, name):
    L = 3 if name == 'msg' else 4
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    K = 128 if name == 'enc0' else k
    expected = {'msg': (64, 48), 'enc0': (126, 128), 'enc2': (128, 128), 'enc4': (128, 64)}[name]
    if (not _ALP_ON or not _ALP_ACTIVE[0] or (k, n) != expected
            or str(a.dtype) != 'float32' or str(b.dtype) != 'float32'
            or tuple(b.shape) != (k, n) or not _ipt_fits(L, m, K, n)):
        _ALP_STATS['fallback'] += 1
        return fnp.matmul(a, b)
    if _ALP_CACHE['owner'] is not owner:
        _alp_reset()
        _ALP_CACHE['owner'] = owner
    key = (name, L, K, n)
    entry = _ALP_CACHE['values'].get(key)
    if entry is not None and entry['b'] is not b:
        # Learned-factor identity is part of cache ownership, not merely shape.
        _alp_reset()
        _ALP_CACHE['owner'] = owner
        entry = None
    hold = []
    _TR_LAST[0] = None
    try:
        if entry is None:
            count = 7 ** L * (K // (1 << L)) * (n // (1 << L))
            assert _ALP_CACHE['bytes'] + 4 * count <= _ALP_LIMIT
            bb = b
            if K != k:
                bb = fnp.zeros((K, n), dtype=fnp.float32)
                hold.append(fnp.copyto(bb[:k], b))
            # Owned B has no shared-arena alias and is never changed after forward.
            side = _astra_ubown_bind_v1(L, K, n, False)
            TB = _astra_ubown_forward_v1(bb, L, K, n, side)
            entry = dict(b=b, tree=TB)
            _ALP_CACHE['values'][key] = entry
            _ALP_CACHE['bytes'] += 4 * count
            _ALP_STATS['stores'] += 1
            _ALP_STATS['max_cache_bytes'] = max(_ALP_STATS['max_cache_bytes'], _ALP_CACHE['bytes'])
        else:
            _ALP_STATS['hits'] += 1
        aa = a
        if K != k:
            aa = fnp.zeros((m, K), dtype=fnp.float32)
            hold.append(fnp.copyto(aa[:, :k], a))
        out = fnp.empty((m, n), dtype=fnp.float32)
        TA = _ipt_forward_a_ab(aa, L, m, K)
        _ipt_leaf_unwind_ab(TA, entry['tree'], out, L, m, n)
        _ALP_STATS['calls'] += 1
        _ALP_STATS['events'].append(dict(name=name, m=m, k=k, padded_k=K, n=n, L=L))
        return out
    finally:
        # The fast products overwrite the shared A arena; never restore a stale identity.
        _TR_LAST[0] = None
        _BH_HOLD.clear()
        del hold

def _alp__e2_messages(k, W, state):
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)
    """src = state@msg_w.T + msg_b into the persistent SRC and M = [w.T@src1, w2.T@src2, w3.T@src3] as one batched matmul whose
    per-item operands have the original layouts; the edge source and gnn_step of one layer share the result."""
    got = _E2_MSG.get('v')
    if got is not None and got[0] is state and got[1] is W:
        return got[2]
    _e2_hold.append(fnp.add(_alp_project(state, k['msg_wT'], k['P'], 'msg'), k['P']['msg_b'], out=k['SRC']))
    M = fnp.matmul(_e2_H(_e2_H(_e2_wpow(W))['T013']), k['SRC3'])
    _E2_MSG['v'] = (state, W, M)
    return M

def _alp_gnn_step(l,x,sigma,W,state,P):
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
    E = fnp.matmul(_e2_H(_e2_H(_e2_wpow(W, need6=True))['T135']), k['SQ'])
    _e2_hold.append(fnp.multiply(k['Hg'], M, out=k['HM03']))
    _e2_hold.append(fnp.maximum(E, 0, out=E))
    _e2_hold.append(fnp.log1p(E, out=k['HM36']))
    _e2_hold.append(fnp.clip(k['Hm'], -8, 8, out=k['Hm']))
    h = silu(_e2_H(_e2_H(_alp_project(k['H'], k['enc0_wT'], P, 'enc0')) + _e2_H(_hc_bias(l, P['enc0_b']))))   # opus-engineer hc: enc0_b + z u' when the layer's read made the new column
    h = silu(_e2_H(_e2_H(_alp_project(h, k['enc2_wT'], P, 'enc2')) + P['enc2_b']))
    state = _e2_H(_alp_project(h, k['enc4_wT'], P, 'enc4')) + P['enc4_b']
    delta = _e2_H(_e2_H(_e2_H(sigma.astype(fnp.float32)) * c['rms']) * _e2_H(_e2_H(state @ c['head']) + c['head_b'])) * c['gain']
    return delta,state

def _e2_messages(k, W, state):
    if not _ALP_ON or not _ALP_ACTIVE[0]:
        return _ALP_PARENT_MESSAGES(k, W, state)
    return _alp__e2_messages(k, W, state)


def gnn_step(l, x, sigma, W, state, P):
    if not _ALP_ON or not _ALP_ACTIVE[0]:
        return _ALP_PARENT_GNN(l, x, sigma, W, state, P)
    return _alp_gnn_step(l, x, sigma, W, state, P)


def _predict_binary(*args, **kwargs):
    if not _ALP_ON:
        return _ALP_PARENT_PREDICT(*args, **kwargs)
    assert not _ALP_ACTIVE[0]
    _alp_reset()
    _ALP_STATS.update(calls=0, stores=0, hits=0, fallback=0, max_cache_bytes=0, events=[], retired=False)
    _ALP_ACTIVE[0] = True
    try:
        return _ALP_PARENT_PREDICT(*args, **kwargs)
    finally:
        _ALP_ACTIVE[0] = False
        _alp_reset()
        _TR_LAST[0] = None
        _ALP_STATS['retired'] = True

"""Prospective metered append-only port; not an admitted candidate or result.

The eventual builder must set _NF_LITERAL to a frozen FP32 payload and
require the information gate. This draft has not been executed or qualified.
"""
_NF_STATE = {}
_NF_PARAMETERS = None


def _nf_graph_fields(m, C, aux):
    variance = fnp.maximum(fnp.diag(C), 1e-12)
    sd = fnp.sqrt(variance)
    R = C / (sd[:, None] * sd[None, :])
    fnp.fill_diagonal(R, 0)
    a, g, p = aux['alpha'], aux['gate'], aux['pdf']
    phi = fnp.stack((a, g, p), axis=1)
    n = int(a.shape[0])
    idx = fnp.argsort(-fnp.abs(R), axis=1, kind='stable')[:, :8]
    rn = fnp.take_along_axis(R, idx, axis=1)
    neigh = phi[idx]
    signed = fnp.mean(rn[:, :, None] * neigh, axis=1)
    absolute = fnp.mean(fnp.abs(rn)[:, :, None] * neigh, axis=1)
    energy = fnp.stack((fnp.sum(rn*rn, axis=1), fnp.max(fnp.abs(rn), axis=1)), axis=1)
    soft = []
    for gamma in (8., 32.):
        z = gamma * R * R
        fnp.fill_diagonal(z, float('-inf'))
        z = z - fnp.max(z, axis=1, keepdims=True)
        weight = fnp.exp(z)
        weight = weight / fnp.sum(weight, axis=1, keepdims=True)
        soft.append(weight @ phi)
    lv = fnp.log(variance)
    seed = fnp.stack((fnp.ones(n), a, g, p, a*p, (a*a-1)*p,
                      m/sd, lv-fnp.mean(lv)), axis=1)
    seed = seed / fnp.maximum(fnp.sqrt(fnp.sum(seed*seed, axis=0)), 1e-12)
    Q = fnp.linalg.qr(seed, mode='reduced')[0]
    vmean = fnp.mean(variance)
    S = (C+C.T)*.5/vmean
    for _ in range(8):
        Q = fnp.linalg.qr(S @ Q, mode='reduced')[0]
    projected = Q @ (Q.T @ phi)
    leverage = fnp.sum(Q*Q, axis=1)
    fraction = fnp.sum(Q*(S @ Q), axis=1)/(variance/vmean)
    return fnp.concatenate((signed, absolute, energy, *soft, projected,
                            leverage[:, None], fraction[:, None]), axis=1)


def _nf_transport(history, gate, w):
    return (w.astype(fnp.float64).T @ history)*gate[:, None]


_NF_OLD_FEATURES = features
_NF_OLD_FINAL = final_scalar_features
_NF_OLD_STEP = gnn_step
_NF_OLD_PREDICT = Estimator.predict


def features(l, m, C, K21, aux, w):
    x = _NF_OLD_FEATURES(l, m, C, K21, aux, w)
    if _NF_STATE.get('active') and 12 <= l <= 14:
        if l == 12:
            history = None
        else:
            history = _nf_transport(_NF_STATE.pop('history'), aux['gate'], w)
        born = _nf_graph_fields(m, C, aux)*aux['sigma'][:, None]
        _NF_STATE['history'] = born if history is None else fnp.concatenate((history, born), axis=1)
    return x


def final_scalar_features(l, m, aux, w):
    x = _NF_OLD_FINAL(l, m, aux, w)
    if _NF_STATE.get('active') and l == 15:
        _NF_STATE['history'] = _nf_transport(_NF_STATE.pop('history'), aux['gate'], w)
    return x


def gnn_step(l, x, sigma, w, state, P):
    value = _NF_OLD_STEP(l, x, sigma, w, state, P)
    if _NF_STATE.get('active') and l == 15:
        graph = (_NF_STATE.pop('history')/sigma[:, None]).astype(fnp.float32)
        control = fnp.concatenate((x.astype(fnp.float32), value[1]), axis=1)
        X = fnp.concatenate((control, graph), axis=1)
        t = _NF_PARAMETERS
        X = fnp.clip((X-t['mean'])/t['scale'], -8, 8)
        h = silu(X @ t['layers.0.weight'].T+t['layers.0.bias'])
        h = silu(h @ t['layers.2.weight'].T+t['layers.2.bias'])
        result = (h @ t['layers.4.weight'].T+t['layers.4.bias'])[:, 0]
        _NF_STATE['delta'] = result*sigma.astype(fnp.float32)*t['output_scale']
    return value


def _nf_predict(self, mlp, flop_budget):
    global _NF_PARAMETERS
    assert not _NF_STATE
    if int(mlp.width) != 1024 or int(mlp.depth) != 16:
        return _NF_OLD_PREDICT(self, mlp, flop_budget)
    if _NF_PARAMETERS is None:
        _NF_PARAMETERS = _unpack_static_fp32(_NF_LITERAL)
    _NF_STATE['active'] = True
    try:
        base = _NF_OLD_PREDICT(self, mlp, flop_budget)
        delta = _NF_STATE.pop('delta')
        last = (base[15].astype(fnp.float64)+delta.astype(fnp.float64)).astype(fnp.float32)
        return fnp.concatenate((base[:15], last[None]), axis=0)
    finally:
        _NF_STATE.clear()


Estimator.predict = _nf_predict

_E2_GNN_STATE = {}   # E2: per-process GNN work buffers, transposed weights, per-layer parameter views and derived constants
_E2_MSG = {}         # E2: the current layer's message products M=[w.T@src1, w2.T@src2, w3.T@src3] for one (state, W) pair


def _e2_gnn_consts(P, n):
    k = _E2_GNN_STATE.get(n)
    if k is not None and k['P'] is P:
        return k
    f32 = fnp.float32
    cm = int(P['msg_w'].shape[0]) // 3
    nx = int(P['centers'].shape[1])
    H = fnp.empty((n, nx + 6 * cm), dtype=f32)                   # h = [x | clip(msgs)] written in place
    HMt = H[:, nx:].reshape(n, 6, cm).transpose(1, 0, 2)
    SRC = fnp.empty((n, 3 * cm), dtype=f32)
    MSG = fnp.empty((n, 3 * cm), dtype=f32)
    k = dict(P=P, cm=cm, H=H, Hx=H[:, :nx], Hg=H[:, 5:6], Hm=H[:, nx:], HM03=HMt[0:3], HM36=HMt[3:6],
             SRC=SRC, SRC3=SRC.reshape(n, 3, cm).transpose(1, 0, 2), SQ=fnp.empty((3, n, cm), dtype=f32),
             MSG=MSG, MSG3=MSG.reshape(n, 3, cm).transpose(1, 0, 2),
             msg_wT=P['msg_w'].T, enc0_wT=P['enc0_w'].T, enc2_wT=P['enc2_w'].T, enc4_wT=P['enc4_w'].T,
             mask=fnp.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=f32), layers={})
    _E2_GNN_STATE[n] = k
    return k


def _e2_gnn_layer(k, l):
    c = k['layers'].get(l)
    if c is None:
        P = k['P']
        c = k['layers'][l] = dict(centers=P['centers'][l], scales=P['scales'][l], rms=P['rms'][l], head=P['head'][l],
                                  head_b=P['head_b'][l], gain=1 + 0.25 * fnp.tanh(P['gain'][l:l + 1]))
    return c


def _e2_messages(k, W, state):
    """src = state@msg_w.T + msg_b into the persistent SRC and M = [w.T@src1, w2.T@src2, w3.T@src3] as one batched matmul whose
    per-item operands have the original layouts; the edge source and gnn_step of one layer share the result."""
    got = _E2_MSG.get('v')
    if got is not None and got[0] is state and got[1] is W:
        return got[2]
    fnp.add(state @ k['msg_wT'], k['P']['msg_b'], out=k['SRC'])
    M = fnp.matmul(_e2_wpow(W)['T013'], k['SRC3'])
    _E2_MSG['v'] = (state, W, M)
    return M


def gnn_step(l,x,sigma,W,state,P):
    if l<1:
        return fnp.zeros_like(sigma),state*0
    k = _e2_gnn_consts(P, int(sigma.shape[0]))
    c = _e2_gnn_layer(k, l)
    Hx = k['Hx']
    fnp.clip(((x if str(x.dtype) == 'float32' else x.astype(fnp.float32)) - c['centers']) / c['scales'], -8, 8, out=Hx)
    if l == 15:
        fnp.multiply(Hx, k['mask'], out=Hx)
    M = _e2_messages(k, W, state)
    fnp.power(k['SRC3'], 2, out=k['SQ'])
    E = fnp.matmul(_e2_wpow(W, need6=True)['T135'], k['SQ'])
    fnp.multiply(k['Hg'], M, out=k['HM03'])
    fnp.maximum(E, 0, out=E)
    fnp.log1p(E, out=k['HM36'])
    fnp.clip(k['Hm'], -8, 8, out=k['Hm'])
    h=silu(k['H']@k['enc0_wT']+P["enc0_b"])
    h=silu(h@k['enc2_wT']+P["enc2_b"])
    state=h@k['enc4_wT']+P["enc4_b"]
    delta=(sigma.astype(fnp.float32)*c['rms'])*(state@c['head']+c['head_b'])*c['gain']
    return delta,state

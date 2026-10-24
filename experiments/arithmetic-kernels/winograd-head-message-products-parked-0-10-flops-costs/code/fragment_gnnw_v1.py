

# BEGIN SCOUT-NATIVE GNN-W v1 (2026-10-12; rounding-level, NOT bit-exact: the GNN head's message inputs only). The six message products
# M = [w'src1, w2'src2, w3'src3] and E = [w2'sq1, w4'sq2, w6'sq3] ((n x n)' @ (n x 16) each, 1.17% of the S50 bill) ran as plain
# batched matmuls (m = 16 is below the tree kernel's range). Here each runs as ONE level of Strassen-Winograd on the transposed weight
# powers: the four A-side sums are made once per layer on the existing (6, n, n) power stack (both batches select their powers from it
# by a basic slice: [0:3] and [1::2]), the seven half-size products of a batch go into one stack, and the four output quadrants come from
# one 7-term combination (an einsum with the fixed Winograd coefficients) written straight into the (3, n, 16) layout.
_GW_ON = True
_GW_STATS = dict(products=0, sums=0)
_GW_G = [None]
_GW_SUMS = {'W': None, 'v': None}
_GW_BUF = {}


def _gw_gamma():
    g = _GW_G[0]
    if g is None:
        g = _GW_G[0] = fnp.asarray([[[1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0]],
                                    [[1, 0, 0, -1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1]]], dtype=fnp.float32)
    return g


def _gw_sums(st):
    # A = the transposed powers (slots of WB^T); blocks A11, A12, A21, A22 of every slot; S1..S4 once per layer (slot 4 is unused)
    if _GW_SUMS['W'] is st['W'] and _GW_SUMS['v'] is not None:
        return _GW_SUMS['v']
    WBT = st['WB'].transpose(0, 2, 1)
    n = int(WBT.shape[1])
    h = n // 2
    A11, A12, A21, A22 = WBT[:, :h, :h], WBT[:, :h, h:], WBT[:, h:, :h], WBT[:, h:, h:]
    S1 = A21 + A22
    S2 = S1 - A11
    S3 = A11 - A21
    S4 = A12 - S2
    _GW_SUMS['W'], _GW_SUMS['v'] = st['W'], (A11, A12, A22, S1, S2, S3, S4)
    _GW_STATS['sums'] += 1
    return _GW_SUMS['v']


def _gw_product(st, sl, X):
    # (3, n, cm) = A[sl] @ X, A the transposed powers, X (3, n, cm): one Winograd level
    A11, A12, A22, S1, S2, S3, S4 = _gw_sums(st)
    n, cm = int(X.shape[1]), int(X.shape[2])
    h, c = n // 2, cm // 2
    key = (n, cm)
    P = _GW_BUF.get(key)
    if P is None:
        P = _GW_BUF[key] = fnp.empty((7, 3, h, c), dtype=fnp.float32)
    B11, B12, B21, B22 = X[:, :h, :c], X[:, :h, c:], X[:, h:, :c], X[:, h:, c:]
    T1 = B12 - B11
    T2 = B22 - T1
    T3 = B22 - B12
    T4 = T2 - B21
    hold = [T1, T2, T3, T4]
    for j, (a, b) in enumerate(((A11[sl], B11), (A12[sl], B21), (S4[sl], B22), (A22[sl], T4), (S1[sl], T1), (S2[sl], T2), (S3[sl], T3))):
        hold.append(fnp.matmul(a, b, out=P[j]))
    C = fnp.einsum('pqj,jzab->zpaqb', _gw_gamma(), P, optimize=False)
    _GW_STATS['products'] += 1
    del hold
    return fnp.reshape(C, (3, n, cm))


def _e2_messages(k, W, state):
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)
    """= the parent with M from _gw_product (the powers stack is made with w6 here, as gnn_step makes it right after)."""
    got = _E2_MSG.get('v')
    if got is not None and got[0] is state and got[1] is W:
        return got[2]
    _e2_hold.append(fnp.add(state @ k['msg_wT'], k['P']['msg_b'], out=k['SRC']))
    if _GW_ON:
        M = _gw_product(_e2_H(_e2_wpow(W, need6=True)), slice(0, 3), k['SRC3'])
    else:
        M = fnp.matmul(_e2_H(_e2_H(_e2_wpow(W))['T013']), k['SRC3'])
    _E2_MSG['v'] = (state, W, M)
    return M


def gnn_step(l,x,sigma,W,state,P):
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
    if _GW_ON:
        E = _gw_product(_e2_H(_e2_wpow(W, need6=True)), slice(1, 6, 2), k['SQ'])
    else:
        E = fnp.matmul(_e2_H(_e2_H(_e2_wpow(W, need6=True))['T135']), k['SQ'])
    _e2_hold.append(fnp.multiply(k['Hg'], M, out=k['HM03']))
    _e2_hold.append(fnp.maximum(E, 0, out=E))
    _e2_hold.append(fnp.log1p(E, out=k['HM36']))
    _e2_hold.append(fnp.clip(k['Hm'], -8, 8, out=k['Hm']))
    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + _e2_H(_hc_bias(l, P['enc0_b']))))   # opus-engineer hc: enc0_b + z u' when the layer's read made the new column
    h = silu(_e2_H(_e2_H(h @ k['enc2_wT']) + P['enc2_b']))
    state = _e2_H(h @ k['enc4_wT']) + P['enc4_b']
    delta = _e2_H(_e2_H(_e2_H(sigma.astype(fnp.float32)) * c['rms']) * _e2_H(_e2_H(state @ c['head']) + c['head_b'])) * c['gain']
    return delta,state
# END SCOUT-NATIVE GNN-W v1

_E2_WPOW = {}   # E2: the current layer's weight powers [w, w2, w3, w4, (unused), w6] in one persistent (6,n,n) float32 stack
_E2_FEAT = {}   # E2: per-process constant feature columns (ones, ones*(2l/15-1), ones*(l/16))


def _e2_wpow(W, need6=False):
    """w = W.astype(float32), w2 = w*w, w3 = w2*w, w4 = w2*w2, w6 = w3*w3 computed once per layer (same float32 operations the
    features, GNN and message-edge code each recomputed), stacked so the GNN/message products can run as batched matmuls."""
    st = _E2_WPOW.get('st')
    n = int(W.shape[0])
    if st is None or int(st['WB'].shape[1]) != n:
        WB = fnp.empty((6, n, n), dtype=fnp.float32)
        WBT = WB.transpose(0, 2, 1)
        S4 = fnp.empty((4, n), dtype=fnp.float32)
        st = dict(W=None, WB=WB, rows=fnp.unstack(WB), T013=WBT[0:3], T135=WBT[1::2], R02=WB[0:3:2], R13=WB[1:4:2],
                  C13=fnp.asarray([0.001953125, 1.1444091796875e-05], dtype=fnp.float32)[:, None, None],
                  C4=fnp.asarray([0.7071067811865475, 11.313708498984761, 93.47798314754834, 836.0924988699916], dtype=fnp.float32)[:, None],
                  S4=S4, S4e=S4[0::2], S4o=S4[1::2], S4v=fnp.unstack(S4))
        _E2_WPOW['st'] = st
    w, w2, w3, w4, _, w6 = st['rows']
    if st['W'] is not W:
        fnp.copyto(w, W)
        fnp.multiply(w, w, out=w2)
        fnp.multiply(w2, w, out=w3)
        fnp.multiply(w2, w2, out=w4)
        st['W'], st['sums'], st['have6'] = W, None, False
    if need6 and not st['have6']:
        fnp.multiply(w3, w3, out=w6)
        st['have6'] = True
    return st


def _e2_wsums(st):
    """(sum(w)*c1, sum(w2-2^-9)*c2, sum(w3)*c3, sum(w4-3*2^-18)*c4) column sums, two batched reductions (same per-column order)."""
    if st['sums'] is None:
        fnp.sum(st['R02'], axis=1, out=st['S4e'])
        fnp.sum(st['R13'] - st['C13'], axis=1, out=st['S4o'])
        fnp.multiply(st['S4'], st['C4'], out=st['S4'])
        st['sums'] = st['S4v']
    return st['sums']


def _e2_feature_columns(n, l):
    c = _E2_FEAT.get((n, l))
    if c is None:
        ones = _E2_FEAT.get(n)
        if ones is None:
            ones = _E2_FEAT[n] = fnp.ones(n)
        c = _E2_FEAT[(n, l)] = (ones * (2 * l / 15 - 1), ones * (l / 16), ones)
    return c


def features(l, m, C, K21, aux, W):
    n = len(m)
    c = _e2_feature_columns(n, l)
    s = aux['sigma']
    v = s * s
    rho = aux['pre_cov'] / fnp.outer(s, s)
    skplane = K21 / fnp.outer(v, s)
    skplane_square = skplane * skplane
    postscale = fnp.sqrt(fnp.maximum(aux['post_v'], 1e-12))
    postrho = C / fnp.outer(postscale, postscale)
    postrho = fnp.clip(postrho, -1, 1)
    ws = _e2_wsums(_e2_wpow(W))
    ls = fnp.log(s)
    ms = m / s
    vs = v * s
    vsq = v * v
    rootn = float(n) ** 0.5
    vals = (c[0], c[1], c[2], aux['alpha'], ls - fnp.mean(ls), aux['gate'], aux['pdf'], aux['gaussian_mean'] / s, ms, (m - aux['gaussian_mean']) / s, aux['pre_k3'] / vs, aux['post_v'] / v, aux['pre_k4'] / vsq, fnp.mean(rho, axis=1), fnp.sqrt(fnp.mean(rho * rho, axis=1)), fnp.mean(fnp.abs(rho), axis=1), ws[0], ws[1], ws[2], ws[3], aux['post_k3'] / vs, aux['post_k4'] / vsq, fnp.sqrt(fnp.sum(skplane_square, axis=1)), fnp.sqrt(fnp.sum(skplane_square, axis=0)), fnp.sum(skplane, axis=1) / rootn, fnp.sum(skplane, axis=0) / rootn, rho @ aux['alpha'] / n, fnp.sqrt(fnp.mean(postrho * postrho, axis=1)), fnp.mean(postrho, axis=1), c[2] * fnp.mean(ms))
    return fnp.stack(vals, axis=1, dtype=fnp.float32)

# HELD: local translation tangent of the learned correction, L12--14 only.
# Context held fixed: Cpre, pre cumulants, previous state, HC bias, postrho summaries.
_LG18_LAYERS = (12, 13, 14)
_LG18_COLUMNS = (3, 5, 6, 7, 8, 9, 11, 20, 21, 26, 29)
_LG18_PARENT_SILU = silu
_LG18_PARENT_MESSAGES = _e2_messages
_LG18_PARENT_GNN = _gop2_gnn_step
_LG18_PARENT_PREDICT = Estimator.predict
_LG18_ACTIVE = None
_LG18_FORWARD = []
_LG18_MESSAGE = None
_LG18_PENDING = None
_LG18_CACHE = {}
_LG18_STATS = dict(layers=[], retired=True)


def _lg18_local_columns(aux, host, xp):
    """Unnormalized d feature / d mu_i; other marginal means are fixed.

    Raw ReLU moments obey r_j' = j r_(j-1), j>=2.  The corrected
    mean derivative is the matching first-order Edgeworth gate.
    Columns26/29 include only their own-row diagonal dependence (1/n).
    """
    hold = _E2Hold()
    s = aux['sigma']; a = aux['alpha']; p = aux['pdf']; g = aux['gate']
    v = hold(s * s); vs = hold(v * s); vv = hold(v * v)
    t3 = hold(aux['pre_k3'] / vs)
    t4 = hold(aux['pre_k4'] / vv)
    a2 = hold(a * a); a3 = hold(a2 * a)
    sk = hold(hold(hold(t3 * hold(a2 - 1)) * p) / 6)
    ku = hold(hold(hold(t4 * hold(hold(3 * a) - a3)) * p) / 24)
    gate = hold(hold(g + sk) + ku)
    one_minus = hold(1 - gate)
    post_v = aux['post_v']; post_k3 = aux['post_k3']
    m2 = hold(host * host); m3 = hold(m2 * host)
    dv = hold(hold(2 * host) * one_minus)
    dk3 = hold(hold(3 * one_minus) * hold(post_v - m2))
    dk4 = hold(hold(4 * one_minus) * hold(hold(post_k3 - hold(hold(3 * host) * post_v)) + m3))
    inv = hold(1 / s)
    n = int(s.shape[0])
    gate_inv = hold(gate * inv)
    columns = xp.stack((inv, hold(p * inv), hold(hold(hold(-a) * p) * inv), hold(g * inv),
                        gate_inv, hold(hold(gate - g) * inv), hold(dv / v),
                        hold(dk3 / vs), hold(dk4 / vv),
                        hold(inv / n), hold(gate_inv / n)), axis=1)
    return columns, gate


def _lg18_head_jvp(dinput, weight0, weight2, head2, forward, scale, xp):
    """Exact directional derivative of the two-SiLU frozen-context head.

    forward contains (preactivation, SiLU output, 1+exp(-preactivation)).
    Capturing these values adds no forward arithmetic or rounding changes.
    """
    _z0, y0, den0 = forward[0]
    _z2, y2, den2 = forward[1]
    sig0 = 1 / den0
    sig2 = 1 / den2
    d0 = (dinput @ weight0) * (sig0 + y0 * (1 - sig0))
    d2 = (d0 @ weight2) * (sig2 + y2 * (1 - sig2))
    return (d2 @ head2) * scale


def silu(x):
    if _LG18_ACTIVE is None:
        return _LG18_PARENT_SILU(x)
    hold = _E2Hold()
    denominator = hold(1 + hold(fnp.exp(hold(-x))))
    value = x / denominator
    _LG18_FORWARD.append((x, value, denominator))
    return value


def _e2_messages(k, W, state):
    global _LG18_MESSAGE
    result = _LG18_PARENT_MESSAGES(k, W, state)
    if _LG18_ACTIVE is not None:
        if _LG18_MESSAGE is not None:
            raise RuntimeError('LG18 duplicate message capture')
        _LG18_MESSAGE = result
    return result


def _lg18_tangent(l, x, sigma, P, result):
    hold = _E2Hold()
    k = _e2_gnn_consts(P, int(sigma.shape[0])); c = _e2_gnn_layer(k, l)
    aux = _E2_LAYER['pair']['aux']; host = _E2_LAYER['beta_host']
    key = int(l)
    cached = _LG18_CACHE.get(key)
    if cached is None or cached['P'] is not P:
        columns = fnp.asarray(_LG18_COLUMNS)
        w0local = hold(fnp.take(k['enc0_wT'], columns, axis=0))
        # H = [30 local | 48 gate-scaled messages | 48 message energies].
        cm = int(k['cm'])
        if cm != 16:
            raise RuntimeError('LG18 requires the current sixteen message channels')
        w0 = fnp.concatenate((w0local, k['enc0_wT'][30:78]), axis=0)
        cached = dict(P=P, columns=columns, w0=w0,
                      scales=fnp.take(c['scales'], columns),
                      head2=k['enc4_wT'] @ c['head'])
        _LG18_CACHE[key] = cached
    raw, host_gate = _lg18_local_columns(aux, host, fnp)
    hold(raw); hold(host_gate)
    local = hold(hold(raw.astype(fnp.float32)) / cached['scales'])
    normalized = hold(fnp.take(k['Hx'], cached['columns'], axis=1))
    local = hold(local * hold(fnp.less(hold(fnp.abs(normalized)), 8)))
    # Index1 in the eleven-column local table is normalized feature5 (gate).
    dg = hold(local[:, 1])
    messages = hold(_LG18_MESSAGE * hold(dg[None, :, None]))
    messages = hold(messages * hold(fnp.less(hold(fnp.abs(k['HM03'])), 8)))
    messages = hold(messages.transpose(1, 0, 2).reshape(int(sigma.shape[0]), 48))
    dinput = hold(fnp.concatenate((local, messages), axis=1))
    scale = hold(hold(sigma.astype(fnp.float32) * c['rms']) * c['gain'])
    # Explicit held intermediates reproduce the helper's JVP while grouping frees.
    _z0, y0, den0 = _LG18_FORWARD[0]
    _z2, y2, den2 = _LG18_FORWARD[1]
    sig0 = hold(1 / den0); sig2 = hold(1 / den2)
    prime0 = hold(sig0 + hold(y0 * hold(1 - sig0)))
    prime2 = hold(sig2 + hold(y2 * hold(1 - sig2)))
    d0 = hold(hold(dinput @ cached['w0']) * prime0)
    d2 = hold(hold(d0 @ k['enc2_wT']) * prime2)
    tangent = hold(hold(d2 @ cached['head2']) * scale)
    # Match effective dm=max(dm,-host), including the rare clipped branch.
    active = hold(fnp.greater(hold(host + result[0]), 0))
    return fnp.where(active, tangent, hold(-host_gate)).astype(fnp.float32)


def _gop2_gnn_step(l, x, sigma, W, state, P):
    global _LG18_ACTIVE, _LG18_MESSAGE, _LG18_PENDING
    if l not in _LG18_LAYERS:
        return _LG18_PARENT_GNN(l, x, sigma, W, state, P)
    if _LG18_ACTIVE is not None or _LG18_FORWARD or _LG18_MESSAGE is not None or _LG18_PENDING is not None:
        raise RuntimeError('LG18 stale or concurrent layer state')
    _LG18_ACTIVE = int(l)
    try:
        result = _LG18_PARENT_GNN(l, x, sigma, W, state, P)
        if len(_LG18_FORWARD) != 2 or _LG18_MESSAGE is None:
            raise RuntimeError('LG18 forward ownership capture failed')
        _LG18_PENDING = (int(l), _lg18_tangent(l, x, sigma, P, result))
        return result
    finally:
        _LG18_ACTIVE = None
        _LG18_FORWARD.clear()
        _LG18_MESSAGE = None


def _lg18_apply(l, C):
    global _LG18_PENDING
    pending, _LG18_PENDING = _LG18_PENDING, None
    if pending is None or pending[0] != int(l):
        raise RuntimeError('LG18 missing current-layer tangent')
    hold = _E2Hold()
    tangent = hold(pending[1])
    gate = hold(_E2_LAYER['g'].astype(fnp.float32))
    row = hold(fnp.outer(tangent, gate))
    symmetric = hold(row + row.T)
    correction = hold(_ps_pre32(_E2_LAYER['Cpre']) * symmetric)
    fnp.fill_diagonal(correction, 0)
    _LG18_STATS['layers'].append(int(l))
    return C + correction


def _lg18_predict(self, *args, **kwargs):
    global _LG18_ACTIVE, _LG18_MESSAGE, _LG18_PENDING
    if _LG18_ACTIVE is not None or _LG18_FORWARD or _LG18_MESSAGE is not None or _LG18_PENDING is not None:
        raise RuntimeError('LG18 stale prediction state')
    _LG18_STATS.clear(); _LG18_STATS.update(layers=[], retired=False)
    try:
        result = _LG18_PARENT_PREDICT(self, *args, **kwargs)
        if _LG18_STATS['layers'] != list(_LG18_LAYERS) or _LG18_PENDING is not None:
            raise RuntimeError('LG18 did not consume all fixed layer tangents')
        return result
    finally:
        _LG18_ACTIVE = None
        _LG18_FORWARD.clear()
        _LG18_MESSAGE = None
        _LG18_PENDING = None
        _LG18_STATS['retired'] = True


Estimator.predict = _lg18_predict

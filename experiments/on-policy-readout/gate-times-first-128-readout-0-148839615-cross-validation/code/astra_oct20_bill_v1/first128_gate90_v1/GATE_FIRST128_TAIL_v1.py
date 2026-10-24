def _fg90_gate_feature(k, h, hold):
    gate = hold(k['H'][:, 5:6])
    feature = hold(h * gate)
    return feature

def _fg90_gate_head(k, l):
    rows = k.get('_fg90_gate_rows')
    if rows is None:
        rows = k['_fg90_gate_rows'] = fnp.unstack(k['P']['gate_first_head'])
    return rows[l]

def _gop2_gnn_step(l, x, sigma, W, state, P):
    _e2_H = _E2Hold()
    _e2_append_inline = _e2_H.append
    _e2_value_inline = None
    _e2_hold = []
    if l < 1:
        return (_e2_append_inline((_e2_value_inline := fnp.zeros_like(sigma))) or _e2_value_inline, _e2_append_inline((_e2_value_inline := (state * 0))) or _e2_value_inline)
    k = _e2_gnn_consts(P, int(_e2_append_inline((_e2_value_inline := sigma.shape[0])) or _e2_value_inline))
    c = _e2_gnn_layer(k, l)
    Hx = k['Hx']
    _e2_hold.append(fnp.clip(((x if x.dtype == 'float32' else x.astype(fnp.float32)) - c['centers']) / c['scales'], -8, 8, out=Hx))
    if l == 15:
        _e2_hold.append(fnp.multiply(Hx, k['mask'], out=Hx))
    M = _e2_messages(k, W, state)
    _e2_hold.append(fnp.power(k['SRC3'], 2, out=k['SQ']))
    E = _gop2_contract(_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := _e2_wpow(W, need6=True))) or _e2_value_inline)['T135'])) or _e2_value_inline, k['SQ'])
    _e2_hold.append(fnp.multiply(k['Hg'], M, out=k['HM03']))
    _e2_hold.append(fnp.maximum(E, 0, out=E))
    _e2_hold.append(fnp.log1p(E, out=k['HM36']))
    _e2_hold.append(fnp.clip(k['Hm'], -8, 8, out=k['Hm']))
    h = silu(_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (k['H'] @ k['enc0_wT']))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := _hc_bias(l, P['enc0_b']))) or _e2_value_inline)))) or _e2_value_inline)
    first_scalar = _e2_H(h @ _fh90_first_head(k, l))
    gate_scalar = _e2_H(_fg90_gate_feature(k, h, _e2_H) @ _fg90_gate_head(k, l))
    h = silu(_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (h @ k['enc2_wT']))) or _e2_value_inline) + P['enc2_b']))) or _e2_value_inline)
    state = (_e2_append_inline((_e2_value_inline := (h @ k['enc4_wT']))) or _e2_value_inline) + P['enc4_b']
    delta = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := sigma.astype(fnp.float32))) or _e2_value_inline) * c['rms']))) or _e2_value_inline) * (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (_e2_H(state @ c['head']) + _e2_H(h @ _hc90_hidden_head(k, l)) + first_scalar + gate_scalar))) or _e2_value_inline) + c['head_b']))) or _e2_value_inline)))) or _e2_value_inline) * c['gain']
    return (delta, state)

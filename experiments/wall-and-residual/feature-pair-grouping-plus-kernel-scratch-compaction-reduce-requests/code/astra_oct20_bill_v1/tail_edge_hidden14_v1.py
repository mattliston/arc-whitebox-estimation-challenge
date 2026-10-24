# [astra-bill 2026-10-20T21:06:14Z] Static learned-edge hidden width 14.
# Approximation: omit frozen output channels (9, 14); no fit and no runtime gathers.
# GNN message width remains 16 (48 transported message features).
assert _EDGE_PARAMETERS is None and _MESSAGE_EDGE_PARAMETERS is None
assert not _E2_EDGE_STATE
assert _gn_message_edge_source_base is _g5_message_edge_base
assert len(_EDGE_LITERAL['output']) == 16
assert _EDGE_LITERAL['output'][9] == -0.0002476689696777612
assert _EDGE_LITERAL['output'][14] == 0.009856822900474072
_O20_EDGE_HIDDEN_KEEP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15)
_O20_EDGE_HIDDEN = 14
_EDGE_LITERAL = {
    key: ([[row[i] for i in _O20_EDGE_HIDDEN_KEEP] for row in value]
          if key in ('left', 'right', 'edge') else
          [value[i] for i in _O20_EDGE_HIDDEN_KEEP]
          if key in ('bias', 'output') else value)
    for key, value in _EDGE_LITERAL.items()
}
_MESSAGE_EDGE_LITERAL = {
    key: [[row[i] for i in _O20_EDGE_HIDDEN_KEEP] for row in value]
    for key, value in _MESSAGE_EDGE_LITERAL.items()
}

def _g5_message_edge_base(layer,m,K21,aux,w,state,P,context):
    _e2_H = _E2Hold(); _e2_append_inline = _e2_H.append; _e2_value_inline = None   # E2: temporaries of this call, released together (one hidden free trip)
    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)
    global _EDGE_PARAMETERS
    if not 8<=layer<=14:
        raise ValueError('Frozen covariance edge layer')
    if _EDGE_PARAMETERS is None:
        _EDGE_PARAMETERS={k:fnp.asarray(v,dtype=fnp.float32) for k,v in _EDGE_LITERAL.items()}
    t=_EDGE_PARAMETERS
    n = int((_e2_append_inline((_e2_value_inline := m.shape[0])) or _e2_value_inline))
    k=_e2_edge_consts(t,n)
    a=aux['alpha'];p=aux['pdf'];g=aux['gate'];s=aux['sigma']
    sk=aux['pre_k3']/s**3;ku=aux['pre_k4']/s**4;ms=_e2_ms(m,s)
    F = fnp.stack((g, p, (_e2_append_inline((_e2_value_inline := (a * p))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (a * a))) or _e2_value_inline) - 1))) or _e2_value_inline) * p))) or _e2_value_inline), ms, (_e2_append_inline((_e2_value_inline := (a * g))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (sk * g))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (sk * p))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := _g5_chain(sk, a, p))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (sk * ms))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (ku * g))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (ku * p))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := _g5_chain(ku, a, p))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (ku * ms))) or _e2_value_inline)), axis=1, dtype=fnp.float32)
    R=k['R'];S=k['S']
    pr=_E2_LAYER.get('pair')
    _e2_hold.append(fnp.divide(_ps_pre32(aux['pre_cov']),pr['ss32'] if pr is not None and pr['aux'] is aux else fnp.outer(_ps32(s),_ps32(s)),out=R))
    _e2_hold.append(fnp.divide(_ps_k2132(K21),fnp.outer(_ps32(s**2),_ps32(s)),out=S))
    F=F/t['feature_scale'];_e2_hold.append(fnp.divide(R,t['rho_scale'],out=R));_e2_hold.append(fnp.divide(S,t['third_scale'],out=S))
    # Draft: the previous layer's state is transported into current coordinates.
    # Both new48x16 matrices must begin at positive zero. All old weights freeze.
    cm = int((_e2_append_inline((_e2_value_inline := P['msg_w'].shape[0])) or _e2_value_inline)) // 3
    if cm != 16 or context['left'].shape != (48, _O20_EDGE_HIDDEN) or context['right'].shape != (48, _O20_EDGE_HIDDEN):
        raise ValueError('Message covariance draft shape mismatch')
    gk=_e2_gnn_consts(P,n)
    _e2_hold.append(fnp.clip(_e2_messages(gk,w,state),-8,8,out=gk['MSG3']))
    messages=gk['MSG']
    left = (_e2_append_inline((_e2_value_inline := (F @ t['left']))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := (messages @ context['left']))) or _e2_value_inline)
    right = (_e2_append_inline((_e2_value_inline := (F @ t['right']))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := (messages @ context['right']))) or _e2_value_inline)
    sf=s.astype(fnp.float32);pf=p.astype(fnp.float32);gf=g.astype(fnp.float32)
    raw = _ecut_raw(left, right, t, k, pf, gf, _e2_hold)   # opus-engineer Ecut: the edge on the response rows x alive columns, 0 elsewhere
    return (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (0.5 * (_e2_append_inline((_e2_value_inline := (raw + (_e2_append_inline((_e2_value_inline := raw.T)) or _e2_value_inline)))) or _e2_value_inline)))) or _e2_value_inline) * (_e2_append_inline((_e2_value_inline := fnp.outer(sf, sf))) or _e2_value_inline)))) or _e2_value_inline) * t['cov_factor']))) or _e2_value_inline)

_gn_message_edge_source_base = _g5_message_edge_base

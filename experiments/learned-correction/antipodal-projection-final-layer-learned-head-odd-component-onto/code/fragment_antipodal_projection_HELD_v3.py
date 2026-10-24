"""Fixed terminal antipodal head projection. Source-only until root grant.

For a fixed prefix, truth(w)-host(w) has an odd part linear in w.
Keep the head's even part and project its odd part onto the 65-column
span W.T@[state64, previous_mean]. No truth, fitted dials, or bank replay.
The source-derived complement literal is appended by the builder.
"""
_AP_PARENT_PREDICT = _predict_binary
_AP_PARENT_FEATURES = final_scalar_features
_AP_PARENT_GNN = gnn_step
_AP_PARENT_HC_BIAS = _hc_bias
_AP_SCALAR = _K415_PARENT_SCALAR
_AP_SCOPE = {}
_AP_CONST = {}
_AP_STATS = dict(calls=0, layers=[], span_columns=65, retained_rank=-1,
                 reused_signed_channels=16, new_signed_channels=48,
                 opposite_encoder_calls=0, hcf_bias_reuses=0)


def _predict_binary(*args, **kwargs):
    _AP_SCOPE.clear()
    _AP_STATS.update(calls=0, layers=[], retained_rank=-1,
                     opposite_encoder_calls=0, hcf_bias_reuses=0)
    try:
        return _AP_PARENT_PREDICT(*args, **kwargs)
    finally:
        _AP_SCOPE.clear()


def final_scalar_features(l, m, aux, W):
    out = _AP_PARENT_FEATURES(l, m, aux, W)
    if l == 15:
        _AP_SCOPE.update(aux=aux, W=W)
    return out


def _hc_bias(l, bias):
    out = _AP_PARENT_HC_BIAS(l, bias)
    if l == 15 and _AP_SCOPE.get('capture_bias', False):
        _AP_SCOPE['bias'] = out
    return out


def _ap_opposite_features(x, aux):
    """Exact sign laws, recomputing scalar post moments in float64."""
    hold = _E2Hold()
    s = aux['sigma']; v = hold(s * s)
    mu = hold(aux['alpha'] * s)
    m, other = _AP_SCALAR(hold(-mu), aux['pre_v'], hold(-aux['pre_k3']), aux['pre_k4'])
    ms = hold(m / s); vs = hold(v * s); vsq = hold(v * v)
    columns = list(hold(fnp.unstack(x.T)))
    columns[3] = other['alpha']
    columns[5] = other['gate']
    columns[7] = hold(other['gaussian_mean'] / s)
    columns[8] = ms
    columns[9] = hold(hold(m - other['gaussian_mean']) / s)
    columns[10] = hold(-x[:, 10])
    columns[11] = hold(other['post_v'] / v)
    columns[16] = hold(-x[:, 16])
    columns[18] = hold(-x[:, 18])
    columns[20] = hold(other['post_k3'] / vs)
    columns[21] = hold(other['post_k4'] / vsq)
    columns[29] = hold(x[:, 2] * hold(fnp.mean(ms)))
    return hold(fnp.stack(columns, axis=1, dtype=fnp.float32)), mu


def _ap_project_odd(B, odd):
    """Numerically defined span projection; fixed relative SVD cutoff 1e-6."""
    hold = _E2Hold()
    B64 = hold(B.astype(fnp.float64))
    norm = hold(fnp.sqrt(hold(fnp.sum(hold(B64 * B64), axis=0))))
    B64 = hold(B64 / hold(fnp.maximum(norm, 1e-300)))
    U, singular, _ = hold(fnp.linalg.svd(B64, full_matrices=False))
    keep = hold(singular > hold(singular[0] * 1e-6))
    coeff = hold(U.T @ hold(odd.astype(fnp.float64)))
    projected = hold(U @ hold(coeff * keep))
    _AP_STATS['retained_rank'] = int(fnp.sum(keep))
    return projected


def gnn_step(l, x, sigma, W, state, P):
    if l != 15:
        return _AP_PARENT_GNN(l, x, sigma, W, state, P)
    if _AP_SCOPE.get('W') is not W or 'aux' not in _AP_SCOPE:
        raise RuntimeError('AP terminal feature custody')
    if _HCF_COLUMNS != ('f15_diag',):
        raise RuntimeError('AP requires the pinned even final HCF column')
    hold = _E2Hold()
    _AP_SCOPE['capture_bias'] = True
    try:
        plus, parent_state = _AP_PARENT_GNN(l, x, sigma, W, state, P)
    finally:
        _AP_SCOPE['capture_bias'] = False
    k = _e2_gnn_consts(P, int(sigma.shape[0])); c = _e2_gnn_layer(k, l)
    if int(k['cm']) != 16 or int(state.shape[1]) != 64:
        raise RuntimeError('AP pinned head dimensions')
    cached = _E2_MSG.get('v')
    if cached is None or cached[0] is not state or cached[1] is not W:
        raise RuntimeError('AP signed-message custody')
    M = cached[2]
    if 'complement' not in _AP_CONST:
        _AP_CONST['complement'] = fnp.asarray(_AP_COMPLEMENT_LITERAL, dtype=fnp.float32)
    xminus, mu = _ap_opposite_features(x, _AP_SCOPE['aux'])
    hx = hold(fnp.clip(hold(hold(xminus - c['centers']) / c['scales']), -8, 8))
    hx = hold(hx * k['mask'])
    gate_feature = hx[:, 5:6]
    # All W-even energy messages and the quadratic HCF column are unchanged.
    hm = hold(fnp.concatenate([hold(-gate_feature * M[0]),
                               hold(gate_feature * M[1]),
                               hold(-gate_feature * M[2])], axis=1))
    hm = hold(fnp.clip(hm, -8, 8))
    H = hold(fnp.concatenate([hx, hm, k['H'][:, 78:]], axis=1))
    h = hold(silu(hold(hold(H @ k['enc0_wT']) + _AP_SCOPE['bias'])))
    h = hold(silu(hold(hold(h @ k['enc2_wT']) + P['enc2_b'])))
    state_minus = hold(hold(h @ k['enc4_wT']) + P['enc4_b'])
    minus = hold(hold(hold(sigma.astype(fnp.float32) * c['rms']) *
                      hold(hold(state_minus @ c['head']) + c['head_b'])) * c['gain'])
    # Existing M1 supplies state@msg_w[:16].T; subtract its source bias.
    wsum = hold(_e2_wsums(_e2_wpow(W))[0] * (2.0 ** 0.5))
    reused = hold(M[0] - hold(fnp.outer(wsum, P['msg_b'][:16])))
    missing_source = hold(state @ _AP_CONST['complement'])
    missing = hold(_e2_wpow(W)['T013'][0] @ missing_source)
    B = hold(fnp.concatenate([reused, missing, hold(mu.astype(fnp.float32))[:, None]], axis=1))
    odd = hold(hold(plus - minus) * 0.5)
    even = hold(hold(plus + minus) * 0.5)
    projected = _ap_project_odd(B, odd)
    delta = hold(even.astype(fnp.float64)) + projected
    _AP_STATS['calls'] += 1; _AP_STATS['layers'].append(15)
    _AP_STATS['opposite_encoder_calls'] += 1
    _AP_STATS['hcf_bias_reuses'] += 1
    _AP_SCOPE.pop('bias', None)
    return delta, parent_state

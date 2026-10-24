# Algebraic regrouping of the active Gaussian/local-cumulant/K31 series.
# T21/T22 and all pair-map diagonals are unchanged. FP32 association changes.
_ASTRA_HF_PARENT_SERIES = _psp_series
_ASTRA_HF_ON = True
_ASTRA_HF_STATS = dict(calls=0, ba=0, packed=0, elements=0, layers=[], retired=True)

def _astra_hf_grouped(F, G, F5, G5, P, t21, t12, t22, a, hold, qr, qc):
    Z = a['Z']
    z0, z1, z2, work, other = _rqd_get('HF_Z5', Z, lambda: fnp.unstack(Z))
    # Relative to rho^r/r!: local skew/kurtosis use 1/6 and 1/24;
    # K31 uses 1/6 at r=1 and 1/3 at r=2.
    df = hold(hold(hold(F5[1, 4] / 6.0) + hold(F5[2, 5] / 24.0)) + hold(hold(qr * F[3]) / 6.0))
    dg = hold(hold(hold(G5[1, 4] / 6.0) + hold(G5[2, 5] / 24.0)) + hold(hold(qc * G[3]) / 6.0))
    hold(fnp.multiply(hold(F[1] + df), G[1], out=work))
    hold(fnp.multiply(F[1], dg, out=other))
    hold(fnp.add(work, other, out=work))
    hold(fnp.multiply(work, P['P6'][0], out=z0))
    df = hold(hold(hold(F5[1, 5] / 6.0) + hold(F5[2, 6] / 24.0)) + hold(hold(qr * F[4]) / 3.0))
    dg = hold(hold(hold(G5[1, 5] / 6.0) + hold(G5[2, 6] / 24.0)) + hold(hold(qc * G[4]) / 3.0))
    hold(fnp.multiply(hold(F[2] + df), G[2], out=work))
    hold(fnp.multiply(F[2], dg, out=other))
    hold(fnp.add(work, other, out=work))
    hold(fnp.multiply(work, P['P6'][1], out=work))
    hold(fnp.add(z0, work, out=z0))
    hold(fnp.multiply(F[3], G[3], out=work))
    hold(fnp.multiply(work, P['P6'][2], out=work))
    hold(fnp.add(z0, work, out=z0))
    # Original E21 and E22 operations and rounding order, on original tensors.
    X = a['X2'][:3]
    hold(fnp.multiply(F[[[2, 3], [1, 2], [2, 3]]],
                      G[[[1, 2], [2, 3], [2, 3]]], out=X))
    hold(fnp.multiply(a['X2r1'][:3], a['RR1'], out=a['X2r1'][:3]))
    T = a['T'][:3]
    hold(fnp.sum(X, axis=1, out=T))
    t1, t2, t3 = fnp.unstack(T)
    hold(fnp.multiply(t21, t1, out=z1))
    hold(fnp.add(z1, hold(t12 * t2), out=z1))
    hold(fnp.multiply(t22, t3, out=z2))
    hold(fnp.divide(Z[1:3], a['div'][:2], out=Z[1:3]))
    return fnp.sum(Z[:3], axis=0)

def _psp_series(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=False):
    if not _ASTRA_HF_ON:
        return _ASTRA_HF_PARENT_SERIES(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=persistent_fg)
    if _T31_LAYERS:
        raise RuntimeError('Hermite fusion requires the modeled K31 plane')
    q = _E2_LAYER['astra_hf31_row']
    if persistent_fg:
        qr, qc = q[None, :, None], q[None, None, :]
        _ASTRA_HF_STATS['ba'] += 1
    else:
        v = _psp_views(_e2_pair_state(1024), 1024, 128)
        qr = hold(fnp.take(q, v['idx']['rows']))[None, :, :, None]
        qc = hold(fnp.take(q, v['idx']['cols']))[None, :, None, :]
        _ASTRA_HF_STATS['packed'] += 1
    result = _astra_hf_grouped(F, G, F5, G5, P, t21, t12, t22, a, hold, qr, qc)
    _ASTRA_HF_STATS['calls'] += 1
    _ASTRA_HF_STATS['layers'].append([int(_PHYS_LAYER[0]), bool(persistent_fg)])
    elements = 1
    for dim in result.shape:
        elements *= int(dim)
    _ASTRA_HF_STATS['elements'] += elements
    if not persistent_fg:
        _E2_LAYER.pop('astra_hf31_row', None)
    return result

_ASTRA_HF_PARENT_PREDICT = Estimator.predict
def _astra_hf_predict(self, *args, **kwargs):
    _ASTRA_HF_STATS.clear()
    _ASTRA_HF_STATS.update(calls=0, ba=0, packed=0, elements=0, layers=[], retired=False)
    try:
        return _ASTRA_HF_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _E2_LAYER.pop('astra_hf31_row', None)
        _ASTRA_HF_STATS['retired'] = True
Estimator.predict = _astra_hf_predict

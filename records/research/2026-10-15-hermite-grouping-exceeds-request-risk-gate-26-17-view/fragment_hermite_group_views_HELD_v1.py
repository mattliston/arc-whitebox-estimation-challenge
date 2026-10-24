# Algebraic regrouping of the active Gaussian/local-cumulant/K31 series.
# T21/T22 and all pair-map diagonals are unchanged. FP32 association changes.
_ASTRA_HF_PARENT_SERIES = _psp_series
_ASTRA_HF_ON = True
_ASTRA_HF_STATS = dict(calls=0, ba=0, packed=0, elements=0, layers=[], retired=True)

def _astra_hf_grouped(F, G, F5, G5, P, t21, t12, t22, a, hold, qr, qc, persistent_fg=False):
    Z = a['Z']
    z0, z1, z2, work, unused = _rqd_get('HFV_Z5', Z, lambda: fnp.unstack(Z))
    # Both rho orders share the same small vector operations. Constant arrays
    # and scratch views retain their exact owner; no value cache is added.
    def views():
        d = len(F.shape) - 1
        div6 = fnp.asarray(6.0, dtype=F.dtype)
        div24 = fnp.asarray(24.0, dtype=F.dtype)
        k31 = fnp.reshape(fnp.asarray([6.0, 3.0], dtype=F.dtype), (2,) + (1,)*d)
        batch, other, powers = a['X1'][:2], a['T'][:2], P['P6'][:2]
        T = a['T'][:3]
        return (batch, other, powers, div6, div24, k31,
                batch[0], batch[1], P['P6'][2], a['X2'][:3],
                a['X2r1'][:3], T, fnp.unstack(T), Z[1:3], a['div'][:2], Z[:3])
    # Cache only views; all owners are immutable members of the same arena/state.
    # Include every independent owner in the key so replacement rebuilds views.
    owners = (a['X1'], a['T'], P['P6'], a['X2'], a['X2r1'], a['div'])
    tag = ('HFV_WORK_R1',) + tuple(id(owner) for owner in owners)
    cached = _rqd_get(tag, Z, lambda: (owners, views()))
    if not all(old is current for old, current in zip(cached[0], owners)):
        raise RuntimeError('HF view owner identity')
    (batch, other, powers, div6, div24, k31, batch0, batch1, power3,
     X, Xr1, T, ts, Z12, div12, Z3) = cached[1]
    def input_views(F, F5):
        f12, f34, f3 = _rqd_get('HFV_F_R1', F, lambda: (F[1:3], F[3:5], F[3]))
        f54, f65 = _rqd_get('HFV_F5_R1', F5, lambda: (F5[1,4:6], F5[2,5:7]))
        return f12, f34, f3, f54, f65
    if persistent_fg:
        f12, f34, f3, f54, f65 = input_views(F, F5)
        g12, g34, g3, g54, g65 = input_views(G, G5)
    else:
        f12, f34, f3, f54, f65 = F[1:3], F[3:5], F[3], F5[1,4:6], F5[2,5:7]
        g12, g34, g3, g54, g65 = G[1:3], G[3:5], G[3], G5[1,4:6], G5[2,5:7]
    df = hold(hold(hold(f54 / div6) + hold(f65 / div24)) + hold(hold(qr * f34) / k31))
    dg = hold(hold(hold(g54 / div6) + hold(g65 / div24)) + hold(hold(qc * g34) / k31))
    hold(fnp.multiply(hold(f12 + df), g12, out=batch))
    hold(fnp.multiply(f12, dg, out=other))
    hold(fnp.add(batch, other, out=batch))
    hold(fnp.multiply(batch, powers, out=batch))
    hold(fnp.add(batch0, batch1, out=z0))
    hold(fnp.multiply(f3, g3, out=work))
    hold(fnp.multiply(work, power3, out=work))
    hold(fnp.add(z0, work, out=z0))
    # Original E21 and E22 operations and rounding order, on original tensors.
    hold(fnp.multiply(F[[[2, 3], [1, 2], [2, 3]]],
                      G[[[1, 2], [2, 3], [2, 3]]], out=X))
    hold(fnp.multiply(Xr1, a['RR1'], out=Xr1))
    hold(fnp.sum(X, axis=1, out=T))
    t1, t2, t3 = ts
    hold(fnp.multiply(t21, t1, out=z1))
    hold(fnp.add(z1, hold(t12 * t2), out=z1))
    hold(fnp.multiply(t22, t3, out=z2))
    hold(fnp.divide(Z12, div12, out=Z12))
    return fnp.sum(Z3, axis=0)

def _psp_series(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=False):
    if not _ASTRA_HF_ON or 'astra_hf31_row' not in _E2_LAYER:
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
    result = _astra_hf_grouped(F, G, F5, G5, P, t21, t12, t22, a, hold, qr, qc, persistent_fg=persistent_fg)
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

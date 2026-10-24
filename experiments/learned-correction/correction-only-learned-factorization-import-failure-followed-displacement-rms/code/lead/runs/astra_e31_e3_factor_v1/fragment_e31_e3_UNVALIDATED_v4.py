# HELD, NON-BITEXACT: root grant and full qualification required.
# HELD correction-only kernel. No model integration or native qualification.
# Requires source-pinned original _psp_series, _rqd_get and fnp in globals.
_ASTRA_E31_ENABLED = True
_ASTRA_E31_PARENT_SERIES = _psp_series
_ASTRA_E31_CONST = {}
_ASTRA_E31_STATS = dict(on_calls=0, off_calls=0, Q=0, H=0, gaussian_reuse_calls=0, constant_flops=0)


def _astra_e31_series(F, G, F5, G5, P, t21, t12, t22, t31, t13,
                      a, hold, wf, wg, persistent_fg=False):
    if not _ASTRA_E31_ENABLED:
        _ASTRA_E31_STATS['off_calls'] += 1
        return _ASTRA_E31_PARENT_SERIES(F, G, F5, G5, P, t21, t12,
                                       t22, t31, t13, a, hold,
                                       persistent_fg=persistent_fg)
    # wf/wg are CURRENT, unscaled-by-pe4 T31 row multipliers. Values are
    # never retained here; only owner-checked shape views may persist.
    q = 1
    for dim in a['Z'].shape[1:]:
        q *= int(dim)
    h = 1
    for dim in F.shape[1:]:
        h *= int(dim)
    _ASTRA_E31_STATS['on_calls'] += 1
    _ASTRA_E31_STATS['Q'] += q
    _ASTRA_E31_STATS['H'] += h
    Z = a['Z']
    z0, z1, z2, z3, z4 = _rqd_get('Z5', Z, lambda: fnp.unstack(Z))
    # Gaussian channel: same products, order and reduction as the parent.
    X = a['X1']
    if persistent_fg:
        F13 = _rqd_get('F13', F, lambda: F[1:4])
        G13 = _rqd_get('G13', G, lambda: G[1:4])
    else:
        F13, G13 = F[1:4], G[1:4]
    # Exact Gaussian/T22 reuse: raw F2G2/F3G3 occupy X2 row2,
    # offset4Q..6Q, disjoint from Gaussian X1 at0..3Q in both arenas.
    # New basic views are call-local; no retained raw values/cache entries.
    raw22, x21 = a['X2'][2], a['X2'][:2]
    x10, x112 = X[:1], X[1:3]
    p10, p112 = P['P6'][:1], P['P6'][1:3]
    f10, f112, g10, g112 = F13[:1], F13[1:3], G13[:1], G13[1:3]
    hold(fnp.multiply(f112, g112, out=raw22))
    hold(fnp.multiply(f10, g10, out=x10))
    hold(fnp.multiply(x10, p10, out=x10))
    hold(fnp.multiply(raw22, p112, out=x112))
    hold(fnp.sum(X, axis=0, out=z0))
    # Gaussian fully consumed before overlapping first two X2 rows change.
    X = _rqd_get('E31_X2_3', a['X2'], lambda: a['X2'][:3])
    hold(fnp.multiply(F[[[2, 3], [1, 2]]],
                      G[[[1, 2], [2, 3]]], out=x21))
    _ASTRA_E31_STATS['gaussian_reuse_calls'] += 1
    Xr1 = _rqd_get('E31_X2_r1', a['X2r1'], lambda: a['X2r1'][:3])
    hold(fnp.multiply(Xr1, a['RR1'], out=Xr1))
    T = _rqd_get('E31_T3', a['T'], lambda: a['T'][:3])
    hold(fnp.sum(X, axis=1, out=T))
    t1, t2, t3 = _rqd_get('E31_T3_rows', T, lambda: fnp.unstack(T))
    hold(fnp.multiply(t21, t1, out=z1))
    hold(fnp.add(z1, hold(t12 * t2), out=z1))
    hold(fnp.multiply(t22, t3, out=z2))
    # e31+e3 ONLY. r=2 requires factor2 because rho^2 = 2*P2.
    if persistent_fg:
        f12, f34 = _rqd_get('E31_Frows', F, lambda: (F[1:3], F[3:5]))
        g12, g34 = _rqd_get('E31_Grows', G, lambda: (G[1:3], G[3:5]))
        s12 = _rqd_get('E31_Fskew', F5, lambda: F5[1, 4:6])
        t12r = _rqd_get('E31_Gskew', G5, lambda: G5[1, 4:6])
    else:
        f12, f34, g12, g34 = F[1:3], F[3:5], G[1:3], G[3:5]
        s12, t12r = F5[1, 4:6], G5[1, 4:6]
    ndim = len(F.shape)
    r12 = _ASTRA_E31_CONST.get(ndim)
    if r12 is None:
        r12 = fnp.asarray([1.0, 2.0], dtype=fnp.float32).reshape((2,) + (1,) * (ndim - 1))
        _ASTRA_E31_CONST[ndim] = r12
        _ASTRA_E31_STATS['constant_flops'] += 6
    A = hold(hold(hold(f34 * wf) * r12) + s12)
    B = hold(hold(hold(g34 * wg) * r12) + t12r)
    x0, x1 = _rqd_get('E31_X3_rows', a['X3'], lambda: (a['X3'][0], a['X3'][1]))
    hold(fnp.multiply(A, g12, out=x0))
    hold(fnp.multiply(f12, B, out=x1))
    hold(fnp.add(x0, x1, out=x0))
    hold(fnp.multiply(x0, P['P12'], out=x0))
    hold(fnp.sum(x0, axis=0, out=z3))
    # z3 is now owned by Z. X3 scratch can be reused for the unchanged e4.
    X = _rqd_get('E31_X3_2', a['X3'], lambda: a['X3'][:2])
    hold(fnp.multiply(F5[[[2, 2], [0, 0]], [[5, 6], [1, 2]]],
                      G5[[[0, 0], [2, 2]], [[1, 2], [5, 6]]], out=X))
    hold(fnp.multiply(X, P['P12'], out=X))
    U = _rqd_get('E31_U4', a['U'], lambda: a['U'][2:4])
    hold(fnp.sum(X, axis=1, out=U))
    t8, t9 = _rqd_get('E31_U2_rows', U, lambda: fnp.unstack(U))
    hold(fnp.add(t8, t9, out=z4))
    Z15 = _rqd_get('Z15', Z, lambda: Z[1:5])
    hold(fnp.divide(Z15, a['div'], out=Z15))
    return fnp.sum(Z, axis=0)


def _astra_e31_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity', extra31=None):
    assert _ASTRA_E31_ENABLED and _ASTRA_E31_RUNTIME['active'] and extra31 is None
    assert len(mu)==1024 and _E2_LAYER.get('psp_blocks')==128
    _ASTRA_E31_META['elided_formation_elements'] += len(mu)*len(mu)
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)
    """E2 request diet of the W-G pair map: bit-identical arithmetic (same float64/float32 operations per element, same order),
    shared subexpressions computed once, float32 casts folded into out= stores, the Hermite-series products batched over
    persistent buffers (views built once per process), the series terms stacked so one ordered axis-0 sum reproduces
    ans + e21/2 + e22/4 + (e31+e3)/6 + e4/24."""
    n = len(mu)
    st = _e2_pair_state(n)
    v = fnp.diag(C)
    k3 = fnp.diag(K21)
    L = _E2_LAYER
    if L.get('mu') is mu and L.get('Cpre') is C:     # the live loop already ran maximum/sqrt/divide/cdf on this (mu, diag C)
        vc, s, a, g = L['vc'], L['s'], L['a'], L['g']
    else:
        vc = fnp.maximum(v, 1e-10)
        s = fnp.sqrt(vc)
        a = mu / s
        g = flops.stats.norm.cdf(a)
    na = -a
    pdf = _e2_H(fnp.exp(_e2_H(_e2_H(na * a) / 2))) / 2.5066282746310002
    mu2 = mu * mu
    KV = st['K1325'] * vc                                   # rows vc (1*vc == vc), 3vc, 2vc, 5vc
    Q = mu2 + KV
    MQ = mu * Q
    v3 = _e2_H(fnp.unstack(KV))[1]
    q1, _, q2, _ = fnp.unstack(Q)
    _, mq3, _, mq5 = fnp.unstack(MQ)
    P4 = _e2_H(_e2_H(mu ** 4) + _e2_H(_e2_H(_e2_H(6 * mu) * mu) * vc)) + _e2_H(v3 * vc)
    # rows: mg = mu*g + (1*s)*pdf, r2g = (mu2+vc)*g + (mu*s)*pdf, r3g = (mu*(mu2+3vc))*g + ((mu2+2vc)*s)*pdf, r4g = P4*g + ((mu*(mu2+5vc))*s)*pdf
    G4 = _e2_H(_e2_H(fnp.stack((mu, q1, mq3, P4))) * g) + _e2_H(_e2_H(_e2_H(fnp.stack((st['one'], mu, q2, mq5))) * s) * pdf)
    mg = _e2_H(fnp.unstack(G4))[0]
    d = pdf / s
    A3 = na * pdf
    e = A3 / v
    a2 = a * a
    h4 = a2 - 1
    A4 = h4 * pdf
    f = A4 / _e2_H(v * s)
    # rows: m, r2, r3, r4 (k3*g/1 == k3*g, k3*(4*mg) == (4*k3)*mg exactly, k4*g/1 == k4*g)
    m, r2, r3, r4 = fnp.unstack(_e2_H(_e2_H(G4 + _e2_H(_e2_H(k3 * _e2_H(fnp.stack((e, d, g, _e2_H(4 * mg))))) / st['D3'])) + _e2_H(_e2_H(k4 * _e2_H(fnp.stack((f, e, d, g)))) / st['D4'])))
    var = r2 - _e2_H(m * m)
    post_k3 = _e2_H(r3 - _e2_H(_e2_H(3 * m) * r2)) + _e2_H(2 * _e2_H(m ** 3))
    post_k4 = _e2_H(_e2_H(_e2_H(r4 - _e2_H(_e2_H(4 * m) * r3)) + _e2_H(_e2_H(_e2_H(6 * m) * m) * r2)) - _e2_H(3 * _e2_H(m ** 4))) - _e2_H(_e2_H(3 * var) * var)
    # Hermite derivative rows (orders 0..6) exactly as derivatives(a, pdf, g, mg / s, 6) builds them.
    h5 = _e2_H(a * h4) - _e2_H(2 * a)
    h6 = _e2_H(a * h5) - _e2_H(3 * h4)
    mgs = mg / s
    aa = fnp.stack((mgs, g, pdf, A3, A4, _e2_H(_e2_H(-h5) * pdf), _e2_H(h6 * pdf)))
    bb = fnp.concatenate((_e2_H(_e2_H(_e2_H(_e2_H(a2 + 1) * g) + _e2_H(a * pdf))[None]), _e2_H(2 * _e2_H(aa[:6]))))
    FT, GT = st['FT'], st['GT']
    _e2_hold.append(fnp.stack((aa, bb, bb), axis=1, out=FT))
    _e2_hold.append(fnp.stack((aa, aa, bb), axis=1, out=GT))
    _e2_hold.append(fnp.divide(k3, s ** 3, out=st['t3']))
    t4 = k4 / _e2_H(v ** 2)
    fnp.copyto(st['t4'], t4)
    _pe34 = _PE34_BY_LAYER.get(_PHYS_LAYER[0])   # opus-engineer2 a62 pe3 / pe4 by layer (make_pe34_v1.py): torch's e3 x (1 + pe3[l]), e4 x (1 + pe4[l])
    if _pe34 is not None:
        if _pe34[0] != 1.0:
            _e2_hold.append(fnp.multiply(st['t3'], _pe34[0], out=st['t3']))
        if _pe34[1] != 1.0:
            _e2_hold.append(fnp.multiply(st['t4'], _pe34[1], out=st['t4']))
    _e2_hold.append(fnp.multiply(FT, st['T34c'], out=st['FS12'])); _e2_hold.append(fnp.multiply(GT, st['T34c'], out=st['GS12']))   # [t3*F, t4*F], [t3*G, t4*G]
    SVV = st['SVV']; ss, vs, vv = st['SVVv']
    _e2_hold.append(fnp.outer(s, s, out=ss))
    _ps_ss32 = _ps32(ss)
    rho = _ps_pre32(C) / _ps_ss32
    fnp.fill_diagonal(rho, 0)
    P16 = st['P16']
    _e2_hold.append(fnp.divide(rho, st['DIV6'], out=P16))
    _e2_hold.append(fnp.cumprod(P16, axis=0, out=P16))   # fixed Gaussian R3: rho^r/r!, r=1..3
    _e2_hold.append(fnp.outer(v, s, out=vs))
    _ps_vs32 = _ps32(vs)
    T21 = st['T21']
    _e2_hold.append(fnp.divide(_ps_k2132(K21), _ps_vs32, out=T21))
    fnp.fill_diagonal(T21, 0)
    K22 = k4 / 3 if k22_override is None else k22_override
    if L.get('VV_for') is not C:          # the live loop already wrote outer(diag C, diag C) into vv for this C
        _e2_hold.append(fnp.outer(v, v, out=vv))
    _ps_vv32 = L['ps_vv32'] if L.get('VV_for') is C and 'ps_vv32' in L else _ps32(vv)
    _e2_hold.append(fnp.divide(_ps32(K22), _ps_vv32, out=st['T22']))
    _r31 = _k31_row_scale(mu, k3, v)
    if extra31 is None:
        _astra_e31_w = _ps32(t4 * _r31 if _r31 is not None else t4)
        # ON-only: no consumer observes T31; retain exact w conversion.
    else:
        x31 = _e2_H(extra31 / _e2_H(_e2_H(v * s)[:, None])) / _e2_H(s[None, :])
        fnp.fill_diagonal(x31, 0)
        _e2_hold.append(fnp.add(_ps32(t4 * _r31 if _r31 is not None else t4)[:, None] * rho, _ps32(x31), out=st['T31']))
    if n == 1024 and L.get('psp_blocks') == 128:
        cov, pk21, pk22 = fnp.unstack(_e2_H(_astra_e31_pair_series(st, n, _e2_H, _astra_e31_w)) * fnp.stack((_ps_ss32, _ps_vs32, _ps_vv32)))
    else:
        Z0, Z1, Z2, Z3, Z4 = st['Zv']
        _e2_A = _e2_pair_arena(n)   # E2-A: the series products and tri sums live in the idle kernel arena (no fresh allocation)
        X = _e2_A['X1']
        _e2_hold.append(fnp.multiply(st['Fr16'], st['Gr16'], out=X))
        _e2_hold.append(fnp.multiply(X, st['P6'], out=X))
        _e2_hold.append(fnp.sum(X, axis=0, out=Z0))   # ans
        del X
        X = _e2_A['X2']
        _e2_hold.append(fnp.multiply(st['Fr'][[[2, 3], [1, 2], [2, 3], [3, 4], [1, 2]]], st['Gr'][[[1, 2], [2, 3], [2, 3], [1, 2], [3, 4]]], out=X))
        _e2_hold.append(fnp.multiply(X, st['RR'], out=X))
        T1, T2, T3, T4, T5 = fnp.unstack(_e2_H(fnp.sum(X, axis=1, out=_e2_A['T'])))
        del X
        X = _e2_A['X3']
        _e2_hold.append(fnp.multiply(st['F5'][[[1, 1], [0, 0], [2, 2], [0, 0]], [[4, 5], [1, 2], [5, 6], [1, 2]]],
                                     st['G5'][[[0, 0], [1, 1], [0, 0], [2, 2]], [[1, 2], [4, 5], [1, 2], [5, 6]]], out=X))
        _e2_hold.append(fnp.multiply(X, st['P12'], out=X))
        T6, T7, T8, T9 = fnp.unstack(_e2_H(fnp.sum(X, axis=1, out=_e2_A['U'])))
        del X
        _e2_hold.append(fnp.multiply(T21, T1, out=Z1)); _e2_hold.append(fnp.add(Z1, st['T21T'] * T2, out=Z1))   # e21
        _e2_hold.append(fnp.multiply(st['T22'], T3, out=Z2))   # e22
        _e2_hold.append(fnp.multiply(st['T31'], T4, out=Z3)); _e2_hold.append(fnp.add(Z3, st['T31T'] * T5, out=Z3))
        _e2_hold.append(fnp.add(Z3, T6 + T7, out=Z3))   # e31 + e3
        _e2_hold.append(fnp.add(T8, T9, out=Z4))   # e4
        _e2_hold.append(fnp.divide(st['Z14'], st['DIVZ'], out=st['Z14']))
        cov, pk21, pk22 = fnp.unstack(_e2_H(_e2_H(fnp.sum(st['Z'], axis=0)) * fnp.stack((_ps_ss32, _ps_vs32, _ps_vv32))))
    m2 = _ps32(2 * m)
    post21 = pk21 - _e2_H(fnp.einsum('i,ij->ij', m2, cov, optimize=False))
    post22 = _e2_H(_e2_H(_e2_H(pk22 - _e2_H(fnp.einsum('i,ji->ij', m2, pk21, optimize=False))) - _e2_H(m2 * pk21)) + _e2_H(fnp.einsum('i,j,ij->ij', m2, m2, cov, optimize=False))) - _e2_H(_e2_H(2 * cov) * cov)
    _ps_target = post21
    cov = cov.astype(fnp.float64)
    post21 = post21.astype(fnp.float64)
    post22 = post22.astype(fnp.float64)
    fnp.fill_diagonal(post22, post_k4)
    fnp.fill_diagonal(cov, var)
    fnp.fill_diagonal(post21, post_k3)
    fnp.fill_diagonal(_ps_target, _ps32(post_k3))
    _E2_LAYER.update(ps_target64=post21, ps_target32=_ps_target)
    den = None
    if metric == 'covariance':
        den = _e2_H(_e2_H(fnp.trace(cov)) ** 2) + _e2_H(2 * _e2_H(fnp.sum(_e2_H(cov * cov))))
        radial = _e2_H(3 * _e2_H(fnp.sum(post22))) / den
    else:
        radial = _e2_H(3 * _e2_H(fnp.sum(post22))) / _e2_H(n * _e2_H(n + 2))
    aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg, pre_v=v, pre_cov=C, pre_k3=k3, pre_k4=k4, post_v=var, post_k3=post_k3, post_k4=post_k4)
    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, ss32=_ps_ss32, mgs=mgs)
    return (m, cov, post21, radial, aux)

def _astra_e31_pair_series(st, n, hold, w, block=128):
    assert _ASTRA_E31_ENABLED and _ASTRA_E31_RUNTIME['active']
    v = _psp_views(st, n, block)
    idx = v['idx']
    wf = hold(w[None, :, None])
    wg = hold(w[None, None, :])
    _ASTRA_E31_META['dispatch_calls'] += 1
    # BA remains full, with the original reduction axes and arithmetic order.
    ba = hold(_astra_e31_series(v['Fba'], v['Gba'], v['F5ba'], v['G5ba'], st,
                         st['T21'], st['T21T'], st['T22'], None, None, v['ba'], hold, wf, wg, persistent_fg=True))
    Zv = st['Zv']
    raw = _rqd_get('Zv0', Zv, lambda: Zv[0])
    raw0, raw1, raw2 = _rqd_get('raw3', raw, lambda: (raw[0], raw[1], raw[2]))
    hold(fnp.copyto(raw1, ba[0]))
    # Gather current values every invocation; only shape views/indices persist.
    inverse, transposed = _ppi_get(st, idx, n, block)
    _ppi_copy_p16(st['P16'], inverse, v['p16out'], idx['count'], block, hold)
    f5 = hold(fnp.take(v['Fsrc'], idx['rows'], axis=3))[:, :, :, :, :, None]
    g5 = hold(fnp.take(v['Gsrc'], idx['cols'], axis=3))[:, :, :, :, None, :]
    wr = hold(fnp.take(w, idx['rows']))
    wc = hold(fnp.take(w, idx['cols']))
    pwf = hold(wr[None, :, :, None])
    pwg = hold(wc[None, :, None, :])
    _ASTRA_E31_META['row_gather_elements'] += int(wr.size) + int(wc.size)
    t21 = hold(_ppi_take_slot(st['T21'], inverse, idx['count'], block, 't21'))
    t12 = hold(_ppi_take_slot(st['T21'], transposed, idx['count'], block, 't12'))
    t22 = hold(_ppi_take_slot(st['T22'], inverse, idx['count'], block, 't22'))
    t31 = t13 = None
    _ASTRA_E31_META['elided_gather_elements'] += 2*n*n
    packed = hold(_astra_e31_series(f5[0], g5[0], f5, g5, v['P'], t21, t12, t22, t31, t13, v['packed'], hold, pwf, pwg))
    for target, values in zip((raw0, raw2), fnp.unstack(packed)):
        fnp.put(target, idx['upper'], values)
        fnp.put(target, idx['lower'], values[idx['diagonal_blocks']:])
    return raw


# Integration preserves parent/fallback T31 writes; only ON private clone omits dead writes.
# TR_CLEAR wraps the named core after nonlinear captured its raw function.
# Preserve that exact active raw alias; do not add a new TR-clear wrapper.
assert _nonlinear_pair_batch.__wrapped__ is nonlinear
assert _T31_LAYERS == ()
_ASTRA_E31_PARENT_NONLINEAR = nonlinear
_ASTRA_E31_PARENT_PREDICT = _predict_binary
_ASTRA_E31_RUNTIME = dict(active=False, layer=None, retired=True)
_ASTRA_E31_META = dict(dispatch_calls=0, row_gather_elements=0, delegated=0,
                       elided_formation_elements=0, elided_gather_elements=0)
_PB_CACHES = tuple(_PB_CACHES) + ('_ASTRA_E31_CONST',)


def _astra_e31_nonlinear(mu, C, K21, k4, k22_override=None,
                         metric='identity', extra31=None):
    if (not _ASTRA_E31_ENABLED or extra31 is not None or len(mu) != 1024
            or _E2_LAYER.get('psp_blocks') != 128):
        _ASTRA_E31_META['delegated'] += 1
        return _ASTRA_E31_PARENT_NONLINEAR(mu, C, K21, k4, k22_override,
                                         metric, extra31=extra31)
    if _ASTRA_E31_RUNTIME['active']:
        raise RuntimeError('e31 factor reentrant pair call')
    _ASTRA_E31_RUNTIME.update(active=True, layer=_PHYS_LAYER[0], retired=False)
    try:
        return _astra_e31_pair_batch(mu, C, K21, k4, k22_override, metric)
    finally:
        _ASTRA_E31_RUNTIME.update(active=False, layer=None, retired=True)


nonlinear = _astra_e31_nonlinear


def _predict_binary(*args, **kwargs):
    if _ASTRA_E31_RUNTIME['active']:
        raise RuntimeError('e31 factor retained active context before prediction')
    for key in _ASTRA_E31_STATS:
        _ASTRA_E31_STATS[key] = 0
    for key in _ASTRA_E31_META:
        _ASTRA_E31_META[key] = 0
    try:
        return _ASTRA_E31_PARENT_PREDICT(*args, **kwargs)
    finally:
        # We retain no mutable row-weight owner, only booleans/layer metadata.
        _ASTRA_E31_RUNTIME.update(active=False, layer=None, retired=True)

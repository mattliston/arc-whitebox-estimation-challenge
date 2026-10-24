# Current-card exact elementwise request grouping only.
# Full suffix keeps current centered Hermite, K31, pair, covariance and billing logic.
# Cache creation is charged inside the first actual pair call; no setup/model import here.
assert nonlinear is _nonlinear_pair_batch
assert all(globals().get(name) is value for name, value in _ASTRA_DENSE_CONTEXT)
_O20_PAIR_GROUP_CONST = {}
def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity', extra31=None):
    _e2_H = _E2Hold()
    _e2_append_inline = _e2_H.append
    _e2_value_inline = None
    _e2_hold = []
    'E2 request diet of the W-G pair map: bit-identical arithmetic (same float64/float32 operations per element, same order),\n    shared subexpressions computed once, float32 casts folded into out= stores, the Hermite-series products batched over\n    persistent buffers (views built once per process), the series terms stacked so one ordered axis-0 sum reproduces\n    ans + e21/2 + e22/4 + (e31+e3)/6 + e4/24.'
    n = len(mu)
    st = _e2_pair_state(n)
    v = _E2_LAYER['rqs2_vv'] if _E2_LAYER.get('Cpre') is C and _E2_LAYER.get('rqs2_vv') is not None else fnp.diag(C)
    k3 = _E2_LAYER['rqs2_kdiag'] if _E2_LAYER.get('ps_K21') is K21 and _E2_LAYER.get('rqs2_kdiag') is not None else fnp.diag(K21)
    L = _E2_LAYER
    if L.get('mu') is mu and L.get('Cpre') is C:
        vc, s, a, g = (L['vc'], L['s'], L['a'], L['g'])
    else:
        vc = fnp.maximum(v, 1e-10)
        s = fnp.sqrt(vc)
        a = mu / s
        g = flops.stats.norm.cdf(a)
    na = -a
    pdf = (_e2_append_inline((_e2_value_inline := fnp.exp(_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (na * a))) or _e2_value_inline) / 2))) or _e2_value_inline))) or _e2_value_inline) / 2.5066282746310002
    mu2 = mu * mu
    KV = st['K1325'] * vc
    Q = mu2 + KV
    MQ = mu * Q
    v3 = (_e2_append_inline((_e2_value_inline := fnp.unstack(KV))) or _e2_value_inline)[1]
    q1, _, q2, _ = fnp.unstack(Q)
    _, mq3, _, mq5 = fnp.unstack(MQ)
    P4 = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (mu ** 4))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (6 * mu))) or _e2_value_inline) * mu))) or _e2_value_inline) * vc))) or _e2_value_inline)))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := (v3 * vc))) or _e2_value_inline)
    G4 = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := fnp.stack((mu, q1, mq3, P4)))) or _e2_value_inline) * g))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := fnp.stack((st['one'], mu, q2, mq5)))) or _e2_value_inline) * s))) or _e2_value_inline) * pdf))) or _e2_value_inline)
    mg = (_e2_append_inline((_e2_value_inline := fnp.unstack(G4))) or _e2_value_inline)[0]
    d = pdf / s
    a2 = a * a
    h4 = a2 - 1
    h5 = (_e2_append_inline((_e2_value_inline := (a * h4))) or _e2_value_inline) - (_e2_append_inline((_e2_value_inline := (2 * a))) or _e2_value_inline)
    h6 = (_e2_append_inline((_e2_value_inline := (a * h5))) or _e2_value_inline) - (_e2_append_inline((_e2_value_inline := (3 * h4))) or _e2_value_inline)
    _o20_pg_nh5 = -h5
    _e2_H.append(_o20_pg_nh5)
    _o20_pg_pdf_inputs = fnp.stack((na, h4, _o20_pg_nh5, h6, a))
    _e2_H.append(_o20_pg_pdf_inputs)
    _o20_pg_pdf_products = _o20_pg_pdf_inputs * pdf
    _e2_H.append(_o20_pg_pdf_products)
    A3, A4, _o20_pg_nh5pdf, _o20_pg_h6pdf, _o20_pg_apdf = fnp.unstack(_o20_pg_pdf_products)
    e = A3 / v
    f = A4 / (_e2_append_inline((_e2_value_inline := (v * s))) or _e2_value_inline)
    m, r2, r3, r4 = fnp.unstack(_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (G4 + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (k3 * (_e2_append_inline((_e2_value_inline := fnp.stack((e, d, g, _e2_append_inline((_e2_value_inline := (4 * mg))) or _e2_value_inline)))) or _e2_value_inline)))) or _e2_value_inline) / st['D3']))) or _e2_value_inline)))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (k4 * (_e2_append_inline((_e2_value_inline := fnp.stack((f, e, d, g)))) or _e2_value_inline)))) or _e2_value_inline) / st['D4']))) or _e2_value_inline)))) or _e2_value_inline)
    _o20_pg_mcol = _O20_PAIR_GROUP_CONST.get('mcol')
    if _o20_pg_mcol is None:
        _o20_pg_mcol = _O20_PAIR_GROUP_CONST['mcol'] = fnp.asarray([[3.0], [4.0], [6.0], [2.0]])
    _o20_pg_multiples = _o20_pg_mcol * m
    _e2_H.append(_o20_pg_multiples)
    _o20_pg_m3, _o20_pg_m4, _o20_pg_m6, _o20_pg_m2 = fnp.unstack(_o20_pg_multiples)
    var = r2 - (_e2_append_inline((_e2_value_inline := (m * m))) or _e2_value_inline)
    post_k3 = (_e2_append_inline((_e2_value_inline := (r3 - (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := _o20_pg_m3)) or _e2_value_inline) * r2))) or _e2_value_inline)))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := (2 * (_e2_append_inline((_e2_value_inline := (m ** 3))) or _e2_value_inline)))) or _e2_value_inline)
    post_k4 = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (r4 - (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := _o20_pg_m4)) or _e2_value_inline) * r3))) or _e2_value_inline)))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := _o20_pg_m6)) or _e2_value_inline) * m))) or _e2_value_inline) * r2))) or _e2_value_inline)))) or _e2_value_inline) - (_e2_append_inline((_e2_value_inline := (3 * (_e2_append_inline((_e2_value_inline := (m ** 4))) or _e2_value_inline)))) or _e2_value_inline)))) or _e2_value_inline) - (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (3 * var))) or _e2_value_inline) * var))) or _e2_value_inline)
    mgs = mg / s
    aa = fnp.stack((mgs, g, pdf, A3, A4, _e2_append_inline((_e2_value_inline := _o20_pg_nh5pdf)) or _e2_value_inline, _e2_append_inline((_e2_value_inline := _o20_pg_h6pdf)) or _e2_value_inline))
    bb = fnp.concatenate((_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (a2 + 1))) or _e2_value_inline) * g))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := _o20_pg_apdf)) or _e2_value_inline)))) or _e2_value_inline)[None])) or _e2_value_inline, _e2_append_inline((_e2_value_inline := (2 * (_e2_append_inline((_e2_value_inline := aa[:6])) or _e2_value_inline)))) or _e2_value_inline))
    FT, GT = (st['FT'], st['GT'])
    _o18_scalar_m2 = _o20_pg_m2
    _e2_H.append(_o18_scalar_m2)
    _o18_scalar_ratio = _o18_scalar_m2 / s
    _e2_H.append(_o18_scalar_ratio)
    _o18_scalar_product = _o18_scalar_ratio * aa
    _e2_H.append(_o18_scalar_product)
    _o18_scalar_bc = bb - _o18_scalar_product
    _e2_H.append(_o18_scalar_bc)
    _e2_hold.append(fnp.stack((aa, bb, _o18_scalar_bc), axis=1, out=FT))
    _e2_hold.append(fnp.stack((aa, aa, _o18_scalar_bc), axis=1, out=GT))
    _e2_hold.append(fnp.divide(k3, s ** 3, out=st['t3']))
    t4 = k4 / (_e2_append_inline((_e2_value_inline := (v ** 2))) or _e2_value_inline)
    fnp.copyto(st['t4'], t4)
    _pe34 = _PE34_BY_LAYER.get(_PHYS_LAYER[0])
    if _pe34 is not None:
        if _pe34[0] != 1.0:
            _e2_hold.append(fnp.multiply(st['t3'], _pe34[0], out=st['t3']))
        if _pe34[1] != 1.0:
            _e2_hold.append(fnp.multiply(st['t4'], _pe34[1], out=st['t4']))
    _e2_hold.append(fnp.multiply(FT, st['T34c'], out=st['FS12']))
    _e2_hold.append(fnp.multiply(GT, st['T34c'], out=st['GS12']))
    SVV = st['SVV']
    ss, vs, vv = st['SVVv']
    _e2_hold.append(fnp.outer(s, s, out=ss))
    _ps_ss32 = _ps32(ss)
    rho = _ps_pre32(C) / _ps_ss32
    fnp.fill_diagonal(rho, 0)
    P16 = st['P16']
    _e2_hold.append(fnp.divide(rho, st['DIV6'], out=P16))
    _e2_hold.append(fnp.cumprod(P16, axis=0, out=P16))
    _e2_hold.append(fnp.outer(v, s, out=vs))
    _ps_vs32 = _ps32(vs)
    T21 = st['T21']
    _e2_hold.append(fnp.divide(_ps_k2132(K21), _ps_vs32, out=T21))
    fnp.fill_diagonal(T21, 0)
    K22 = k4 / 3 if k22_override is None else k22_override
    if L.get('VV_for') is not C:
        _e2_hold.append(fnp.outer(v, v, out=vv))
    _ps_vv32 = L['ps_vv32'] if L.get('VV_for') is C and 'ps_vv32' in L else _ps32(vv)
    _e2_hold.append(fnp.divide(_ps32(K22), _ps_vv32, out=st['T22']))
    _r31 = _k31_row_scale(mu, k3, v)
    if extra31 is None:
        _astra_hf31_row = _ps32(t4 * _r31 if _r31 is not None else t4)
        _E2_LAYER['astra_hf31_row'] = _astra_hf31_row
        _e2_hold.append(fnp.multiply(_astra_hf31_row[:, None], rho, out=st['T31'])) if not (_ASTRA_HF_ON and n == 1024 and (L.get('psp_blocks') == 128)) else None
    else:
        x31 = (_e2_append_inline((_e2_value_inline := (extra31 / (_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := (v * s))) or _e2_value_inline)[:, None])) or _e2_value_inline)))) or _e2_value_inline) / (_e2_append_inline((_e2_value_inline := s[None, :])) or _e2_value_inline)
        fnp.fill_diagonal(x31, 0)
        _e2_hold.append(fnp.add(_ps32(t4 * _r31 if _r31 is not None else t4)[:, None] * rho, _ps32(x31), out=st['T31']))
    if n == 1024 and L.get('psp_blocks') == 128:
        _dm2_raw = _e2_append_inline((_e2_value_inline := _psp_pair_series(st, n, _e2_H))) or _e2_value_inline
        _dm2_x0, _dm2_x1, _dm2_x2 = _rqd_get('raw3', _dm2_raw, lambda: fnp.unstack(_dm2_raw))
        cov, pk21, pk22 = (_e2_append_inline((_e2_value_inline := (_dm2_x0 * _ps_ss32))) or _e2_value_inline, _e2_append_inline((_e2_value_inline := (_dm2_x1 * _ps_vs32))) or _e2_value_inline, _e2_append_inline((_e2_value_inline := (_dm2_x2 * _ps_vv32))) or _e2_value_inline)
    else:
        Z0, Z1, Z2, Z3, Z4 = st['Zv']
        _e2_A = _e2_pair_arena(n)
        X = _e2_A['X1']
        _e2_hold.append(fnp.multiply(st['Fr16'], st['Gr16'], out=X))
        _e2_hold.append(fnp.multiply(X, st['P6'], out=X))
        _e2_hold.append(fnp.sum(X, axis=0, out=Z0))
        del X
        X = _e2_A['X2']
        _e2_hold.append(fnp.multiply(st['Fr'][[[2, 3], [1, 2], [2, 3], [3, 4], [1, 2]]], st['Gr'][[[1, 2], [2, 3], [2, 3], [1, 2], [3, 4]]], out=X))
        _e2_hold.append(fnp.multiply(X, st['RR'], out=X))
        T1, T2, T3, T4, T5 = fnp.unstack(_e2_append_inline((_e2_value_inline := fnp.sum(X, axis=1, out=_e2_A['T']))) or _e2_value_inline)
        del X
        X = _e2_A['X3']
        _e2_hold.append(fnp.multiply(st['F5'][[[1, 1], [0, 0], [2, 2], [0, 0]], [[4, 5], [1, 2], [5, 6], [1, 2]]], st['G5'][[[0, 0], [1, 1], [0, 0], [2, 2]], [[1, 2], [4, 5], [1, 2], [5, 6]]], out=X))
        _e2_hold.append(fnp.multiply(X, st['P12'], out=X))
        T6, T7, T8, T9 = fnp.unstack(_e2_append_inline((_e2_value_inline := fnp.sum(X, axis=1, out=_e2_A['U']))) or _e2_value_inline)
        del X
        _e2_hold.append(fnp.multiply(T21, T1, out=Z1))
        _e2_hold.append(fnp.add(Z1, st['T21T'] * T2, out=Z1))
        _e2_hold.append(fnp.multiply(st['T22'], T3, out=Z2))
        _e2_hold.append(fnp.multiply(st['T31'], T4, out=Z3))
        _e2_hold.append(fnp.add(Z3, st['T31T'] * T5, out=Z3))
        _e2_hold.append(fnp.add(Z3, T6 + T7, out=Z3))
        _e2_hold.append(fnp.add(T8, T9, out=Z4))
        _e2_hold.append(fnp.divide(st['Z14'], st['DIVZ'], out=st['Z14']))
        cov, pk21, pk22 = fnp.unstack(_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := fnp.sum(st['Z'], axis=0))) or _e2_value_inline) * fnp.stack((_ps_ss32, _ps_vs32, _ps_vv32))))) or _e2_value_inline)
    m2 = _ps32(_o20_pg_m2)
    post21 = pk21 - (_e2_append_inline((_e2_value_inline := fnp.einsum('i,ij->ij', m2, cov, optimize=False))) or _e2_value_inline)
    post22 = pk22 - (_e2_append_inline((_e2_value_inline := (2 * cov))) or _e2_value_inline) * cov
    _ps_target = post21
    cov = cov.astype(fnp.float64)
    post21 = post21.astype(fnp.float64)
    post22 = post22.astype(fnp.float64)
    fnp.fill_diagonal(post22, post_k4)
    fnp.fill_diagonal(cov, var)
    fnp.fill_diagonal(post21, post_k3)
    fnp.fill_diagonal(_ps_target, post_k3)
    _E2_LAYER.update(ps_target64=post21, ps_target32=_ps_target)
    den = None
    if metric == 'covariance':
        den = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := fnp.trace(cov))) or _e2_value_inline) ** 2))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := (2 * (_e2_append_inline((_e2_value_inline := fnp.sum(_e2_append_inline((_e2_value_inline := (cov * cov))) or _e2_value_inline))) or _e2_value_inline)))) or _e2_value_inline)
        radial = (_e2_append_inline((_e2_value_inline := (3 * (_e2_append_inline((_e2_value_inline := fnp.sum(post22))) or _e2_value_inline)))) or _e2_value_inline) / den
    else:
        radial = (_e2_append_inline((_e2_value_inline := (3 * (_e2_append_inline((_e2_value_inline := fnp.sum(post22))) or _e2_value_inline)))) or _e2_value_inline) / (_e2_append_inline((_e2_value_inline := (n * (_e2_append_inline((_e2_value_inline := (n + 2))) or _e2_value_inline)))) or _e2_value_inline)
    aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg, pre_v=v, pre_cov=C, pre_k3=k3, pre_k4=k4, post_v=var, post_k3=post_k3, post_k4=post_k4)
    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, ss32=_ps_ss32, mgs=mgs)
    return (m, cov, post21, radial, aux)

nonlinear = _nonlinear_pair_batch
_ASTRA_DENSE_CONTEXT = tuple((name, globals()[name] if name in ('_nonlinear_pair_batch', 'nonlinear') else value)
                             for name, value in _ASTRA_DENSE_CONTEXT)

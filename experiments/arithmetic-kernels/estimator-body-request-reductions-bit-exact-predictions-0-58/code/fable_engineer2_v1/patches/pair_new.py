_E2_LAYER = {}        # E2: per-layer intermediates published by the live loop / pair map, each consumer checks object identity
_E2_PAIR_STATE = {}   # E2: persistent float32 pair-series buffers and their views (built once per process; n is fixed)


def _e2_pair_state(n):
    st = _E2_PAIR_STATE.get(n)
    if st is not None:
        return st
    f32 = fnp.float32
    PB = fnp.empty((7, n, n), dtype=f32)                 # [1, rho, rho^2/2!, ..., rho^6/6!]
    fnp.copyto(PB[0], 1.0)
    FS = fnp.empty((3, 7, 3, n), dtype=f32)              # [F, t3*F, t4*F] in r-major (r, channel, i) layout
    GS = fnp.empty((3, 7, 3, n), dtype=f32)
    Z = fnp.empty((5, 3, n, n), dtype=f32)               # [ans, e21, e22, e31+e3, e4] before the final ordered sum
    FT, FS1, FS2 = fnp.unstack(FS)
    GT, GS1, GS2 = fnp.unstack(GS)
    Fr, Gr = FT[:, :, :, None], GT[:, :, None, :]
    T21 = fnp.empty((n, n), dtype=f32); T22 = fnp.empty((n, n), dtype=f32); T31 = fnp.empty((n, n), dtype=f32)
    st = dict(FT=FT, GT=GT, FS1=FS1, FS2=FS2, GS1=GS1, GS2=GS2, F5=FS[:, :, :, :, None], G5=GS[:, :, :, None, :], Fr=Fr, Gr=Gr,
              Fr16=Fr[1:7], Gr16=Gr[1:7], P16=PB[1:7], P6=PB[1:7][:, None], RR=PB[0:3][:, None], P12=PB[1:3][:, None],
              DIV6=fnp.arange(1, 7, dtype=f32)[:, None, None], DIVZ=fnp.asarray([2, 4, 6, 24], dtype=f32)[:, None, None, None],
              Z=Z, Zv=fnp.unstack(Z), Z14=Z[1:5], T21=T21, T21T=T21.T, T22=T22, T31=T31, T31T=T31.T,
              t3=fnp.empty((n,), dtype=f32), t4=fnp.empty((n,), dtype=f32))
    _E2_PAIR_STATE[n] = st
    return st


def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):
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
    pdf = fnp.exp(na * a / 2) / 2.5066282746310002
    mg = s * pdf + mu * g
    mu2 = mu * mu
    v3 = 3 * vc
    r2g = (mu2 + vc) * g + mu * s * pdf
    r3g = mu * (mu2 + v3) * g + (mu2 + 2 * vc) * s * pdf
    r4g = (mu ** 4 + 6 * mu * mu * vc + v3 * vc) * g + mu * (mu2 + 5 * vc) * s * pdf
    d = pdf / s
    A3 = na * pdf
    e = A3 / v
    a2 = a * a
    h4 = a2 - 1
    A4 = h4 * pdf
    f = A4 / (v * s)
    m = mg + k3 * e / 6 + k4 * f / 24
    r2 = r2g + k3 * d / 3 + k4 * e / 12
    r3 = r3g + k3 * g + k4 * d / 4
    r4 = r4g + 4 * k3 * mg + k4 * g
    var = r2 - m * m
    post_k3 = r3 - 3 * m * r2 + 2 * m ** 3
    post_k4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * var * var
    # Hermite derivative rows (orders 0..6) exactly as derivatives(a, pdf, g, mg / s, 6) builds them.
    h5 = a * h4 - 2 * a
    h6 = a * h5 - 3 * h4
    aa = fnp.stack((mg / s, g, pdf, A3, A4, -h5 * pdf, h6 * pdf))
    bb = fnp.concatenate((((a2 + 1) * g + a * pdf)[None], 2 * aa[:6]))
    FT, GT = st['FT'], st['GT']
    fnp.stack((aa, bb, bb), axis=1, out=FT)
    fnp.stack((aa, aa, bb), axis=1, out=GT)
    t3, t4_32 = st['t3'], st['t4']
    fnp.divide(k3, s ** 3, out=t3)
    t4 = k4 / v ** 2
    fnp.copyto(t4_32, t4)
    fnp.multiply(FT, t3, out=st['FS1']); fnp.multiply(FT, t4_32, out=st['FS2'])
    fnp.multiply(GT, t3, out=st['GS1']); fnp.multiply(GT, t4_32, out=st['GS2'])
    ss = fnp.outer(s, s)
    rho = C / ss
    fnp.fill_diagonal(rho, 0)
    P16 = st['P16']
    fnp.divide(rho.astype(fnp.float32), st['DIV6'], out=P16)
    fnp.cumprod(P16, axis=0, out=P16)                                     # rho^r/r!, r=1..6
    vs = fnp.outer(v, s)
    T21 = st['T21']
    fnp.divide(K21, vs, out=T21)
    fnp.fill_diagonal(T21, 0)
    K22 = k4 / 3 if k22_override is None else k22_override
    vv = L['VV'] if L.get('Cpre') is C and 'VV' in L else fnp.outer(v, v)
    fnp.divide(K22, vv, out=st['T22'])
    fnp.multiply(t4[:, None], rho, out=st['T31'])
    Z0, Z1, Z2, Z3, Z4 = st['Zv']
    X = st['Fr16'] * st['Gr16']
    fnp.multiply(X, st['P6'], out=X)
    fnp.sum(X, axis=0, out=Z0)                                            # ans
    X = st['Fr'][[[2, 3, 4], [1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 2, 3]]] * st['Gr'][[[1, 2, 3], [2, 3, 4], [2, 3, 4], [1, 2, 3], [3, 4, 5]]]
    fnp.multiply(X, st['RR'], out=X)
    T1, T2, T3, T4, T5 = fnp.unstack(fnp.sum(X, axis=1))                  # tri(F2,G1) tri(F1,G2) tri(F2,G2) tri(F3,G1) tri(F1,G3)
    X = (st['F5'][[[1, 1], [0, 0], [2, 2], [0, 0]], [[4, 5], [1, 2], [5, 6], [1, 2]]]
         * st['G5'][[[0, 0], [1, 1], [0, 0], [2, 2]], [[1, 2], [4, 5], [1, 2], [5, 6]]])
    fnp.multiply(X, st['P12'], out=X)
    T6, T7, T8, T9 = fnp.unstack(fnp.sum(X, axis=1))                      # the local t3/t4 corrections
    del X
    fnp.multiply(T21, T1, out=Z1); fnp.add(Z1, st['T21T'] * T2, out=Z1)   # e21
    fnp.multiply(st['T22'], T3, out=Z2)                                   # e22
    fnp.multiply(st['T31'], T4, out=Z3); fnp.add(Z3, st['T31T'] * T5, out=Z3)
    fnp.add(Z3, T6 + T7, out=Z3)                                          # e31 + e3
    fnp.add(T8, T9, out=Z4)                                               # e4
    fnp.divide(st['Z14'], st['DIVZ'], out=st['Z14'])
    c0, c1, c2 = fnp.unstack(fnp.sum(st['Z'], axis=0).astype(fnp.float64))
    cov = c0 * ss
    pk21 = c1 * vs
    pk22 = c2 * vv
    m2 = 2 * m
    m2c = m2[:, None]
    post21 = pk21 - m2c * cov
    post22 = pk22 - m2c * pk21.T - m2 * pk21 + fnp.outer(m2, m2) * cov - 2 * cov * cov
    fnp.fill_diagonal(post22, post_k4)
    fnp.fill_diagonal(cov, var)
    fnp.fill_diagonal(post21, post_k3)
    den = None
    if metric == 'covariance':
        den = fnp.trace(cov) ** 2 + 2 * fnp.sum(cov * cov)
        radial = 3 * fnp.sum(post22) / den
    else:
        radial = 3 * fnp.sum(post22) / (n * (n + 2))
    aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg, pre_v=v, pre_cov=C, pre_k3=k3, pre_k4=k4, post_v=var, post_k3=post_k3, post_k4=post_k4)
    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d)
    return (m, cov, post21, radial, aux)

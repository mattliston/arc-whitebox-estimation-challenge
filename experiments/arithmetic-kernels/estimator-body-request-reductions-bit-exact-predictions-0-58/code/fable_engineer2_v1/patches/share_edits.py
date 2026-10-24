# E2 share edits: outer products (s s^T, v s^T, v v^T) of the pair map written into one persistent (3,n,n) float64 stack so the
# float32 Hermite sum is promoted and scaled in ONE multiply (float32->float64 promotion is exact, as astype was); features and the
# message edge source reuse the pair map's s s^T, mg/s and m/s. Applied after pair, bd, learned, loop, fuse, vec.
OTHER = [
("""              T34=fnp.empty((2, n), dtype=f32), FS12=FS[1:3], GS12=GS[1:3], one=fnp.ones(n),
""", """              T34=fnp.empty((2, n), dtype=f32), FS12=FS[1:3], GS12=GS[1:3], one=fnp.ones(n), SVV=fnp.empty((3, n, n)),
"""),
("""    st['t3'], st['t4'] = fnp.unstack(st['T34'])
""", """    st['t3'], st['t4'] = fnp.unstack(st['T34'])
    st['SVVv'] = fnp.unstack(st['SVV'])
"""),
("""    aa = fnp.stack((mg / s, g, pdf, A3, A4, -h5 * pdf, h6 * pdf))
""", """    mgs = mg / s
    aa = fnp.stack((mgs, g, pdf, A3, A4, -h5 * pdf, h6 * pdf))
"""),
("""    ss = fnp.outer(s, s)
    rho = C / ss
""", """    SVV = st['SVV']; ss, vs, vv = st['SVVv']
    fnp.outer(s, s, out=ss)
    rho = C / ss
"""),
("""    vs = fnp.outer(v, s)
    T21 = st['T21']
""", """    fnp.outer(v, s, out=vs)
    T21 = st['T21']
"""),
("""    vv = L['VV'] if L.get('Cpre') is C and 'VV' in L else fnp.outer(v, v)
""", """    if L.get('VV_for') is not C:          # the live loop already wrote outer(diag C, diag C) into vv for this C
        fnp.outer(v, v, out=vv)
"""),
("""    c0, c1, c2 = fnp.unstack(fnp.sum(st['Z'], axis=0).astype(fnp.float64))
    cov = c0 * ss
    pk21 = c1 * vs
    pk22 = c2 * vv
""", """    cov, pk21, pk22 = fnp.unstack(fnp.sum(st['Z'], axis=0) * SVV)   # float32 series promoted exactly, times (ss, vs, vv)
"""),
("""    post22 = pk22 - fnp.einsum('i,ji->ij', m2, pk21, optimize=False) - m2 * pk21 + fnp.outer(m2, m2) * cov - 2 * cov * cov
""", """    post22 = pk22 - fnp.einsum('i,ji->ij', m2, pk21, optimize=False) - m2 * pk21 + fnp.einsum('i,j,ij->ij', m2, m2, cov, optimize=False) - 2 * cov * cov
"""),
("""    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d)
""", """    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, mgs=mgs)
"""),
# features: pair-map s s^T and mg/s; m/s shared with the message edge source
("""    rho = aux['pre_cov'] / fnp.outer(s, s)
    skplane = K21 / fnp.outer(v, s)
""", """    pr = _E2_LAYER.get('pair')
    pr = pr if pr is not None and pr['aux'] is aux else None
    rho = aux['pre_cov'] / (pr['ss'] if pr else fnp.outer(s, s))
    skplane = K21 / fnp.outer(v, s)
"""),
("""    ms = m / s
    vs = v * s
""", """    ms = _e2_ms(m, s)
    vs = v * s
"""),
("""    vals = (c[0], c[1], c[2], aux['alpha'], ls - fnp.mean(ls), aux['gate'], aux['pdf'], aux['gaussian_mean'] / s, ms,""",
 """    vals = (c[0], c[1], c[2], aux['alpha'], ls - fnp.mean(ls), aux['gate'], aux['pdf'], pr['mgs'] if pr else aux['gaussian_mean'] / s, ms,"""),
("""def _e2_feature_columns(n, l):
""", """def _e2_ms(m, s):
    \"\"\"m/s computed once per layer for the message edge source and features (same m and sigma objects).\"\"\"
    c = _E2_LAYER.get('ms')
    if c is not None and c[0] is m and c[1] is s:
        return c[2]
    ms = m / s
    _E2_LAYER['ms'] = (m, s, ms)
    return ms


def _e2_feature_columns(n, l):
"""),
]
EDGE = [
("""    sk=aux['pre_k3']/s**3;ku=aux['pre_k4']/s**4;ms=m/s
""", """    sk=aux['pre_k3']/s**3;ku=aux['pre_k4']/s**4;ms=_e2_ms(m,s)
"""),
("""    fnp.divide(aux['pre_cov'],fnp.outer(s,s),out=R)
""", """    pr=_E2_LAYER.get('pair')
    fnp.divide(aux['pre_cov'],pr['ss'] if pr is not None and pr['aux'] is aux else fnp.outer(s,s),out=R)
"""),
]
LIVE = [
("""            _e2_VV = fnp.outer(vv, vv)
            _e2_CC = Cpre * Cpre
            _E2_LAYER.update(VV=_e2_VV, CC=_e2_CC)
""", """            _e2_VV = _e2_pair_state(n)['SVVv'][2]
            fnp.outer(vv, vv, out=_e2_VV)
            _e2_CC = Cpre * Cpre
            _E2_LAYER.update(VV_for=Cpre, CC=_e2_CC)
"""),
]

# E2 vec edits: per-unit vector algebra batched into stacked rows (each row = the original expression's operations in the
# original order), constants hoisted, and gathers shared. Applied after pair, bd, learned, loop, fuse.
OTHER = [
# pair map state: one (2,n) float32 buffer for (t3, t4) and the constant row multipliers of the batched moment algebra
("""              t3=fnp.empty((n,), dtype=f32), t4=fnp.empty((n,), dtype=f32))
    _E2_PAIR_STATE[n] = st
""", """              T34=fnp.empty((2, n), dtype=f32), FS12=FS[1:3], GS12=GS[1:3], one=fnp.ones(n),
              K1325=fnp.asarray([[1.0], [3.0], [2.0], [5.0]]), D3=fnp.asarray([[6.0], [3.0], [1.0], [1.0]]),
              D4=fnp.asarray([[24.0], [12.0], [4.0], [1.0]]))
    st['t3'], st['t4'] = fnp.unstack(st['T34'])
    st['T34c'] = st['T34'][:, None, None, :]
    _E2_PAIR_STATE[n] = st
"""),
# pair map: [mg, r2g, r3g, r4g] = X*g + (Y*s)*pdf row-wise and [m, r2, r3, r4] = (G4 + k3*E3/D3) + k4*E4/D4 row-wise
("""    mg = s * pdf + mu * g
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
""", """    mu2 = mu * mu
    KV = st['K1325'] * vc                                   # rows vc (1*vc == vc), 3vc, 2vc, 5vc
    Q = mu2 + KV
    MQ = mu * Q
    v3 = fnp.unstack(KV)[1]
    q1, _, q2, _ = fnp.unstack(Q)
    _, mq3, _, mq5 = fnp.unstack(MQ)
    P4 = mu ** 4 + 6 * mu * mu * vc + v3 * vc
    # rows: mg = mu*g + (1*s)*pdf, r2g = (mu2+vc)*g + (mu*s)*pdf, r3g = (mu*(mu2+3vc))*g + ((mu2+2vc)*s)*pdf, r4g = P4*g + ((mu*(mu2+5vc))*s)*pdf
    G4 = fnp.stack((mu, q1, mq3, P4)) * g + fnp.stack((st['one'], mu, q2, mq5)) * s * pdf
    mg = fnp.unstack(G4)[0]
    d = pdf / s
    A3 = na * pdf
    e = A3 / v
    a2 = a * a
    h4 = a2 - 1
    A4 = h4 * pdf
    f = A4 / (v * s)
    # rows: m, r2, r3, r4 (k3*g/1 == k3*g, k3*(4*mg) == (4*k3)*mg exactly, k4*g/1 == k4*g)
    m, r2, r3, r4 = fnp.unstack(G4 + k3 * fnp.stack((e, d, g, 4 * mg)) / st['D3'] + k4 * fnp.stack((f, e, d, g)) / st['D4'])
"""),
("""    t3, t4_32 = st['t3'], st['t4']
    fnp.divide(k3, s ** 3, out=t3)
    t4 = k4 / v ** 2
    fnp.copyto(t4_32, t4)
    fnp.multiply(FT, t3, out=st['FS1']); fnp.multiply(FT, t4_32, out=st['FS2'])
    fnp.multiply(GT, t3, out=st['GS1']); fnp.multiply(GT, t4_32, out=st['GS2'])
""", """    fnp.divide(k3, s ** 3, out=st['t3'])
    t4 = k4 / v ** 2
    fnp.copyto(st['t4'], t4)
    fnp.multiply(FT, st['T34c'], out=st['FS12']); fnp.multiply(GT, st['T34c'], out=st['GS12'])   # [t3*F, t4*F], [t3*G, t4*G]
"""),
# _bd_update publishes its float32 (a, t*alive, s*alive) rows for the live loop's newborn/lam gathers
("""    a64, t64, s64 = fnp.unstack(fnp.stack((a, t * alive, s * alive), dtype=fnp.float32).astype(fnp.float64))
""", """    ats32 = fnp.stack((a, t * alive, s * alive), dtype=fnp.float32)
    _E2_LAYER.update(ats_a=a, ats_t=t, ats_s=s, ats_alive=alive, ats32=ats32)
    a64, t64, s64 = fnp.unstack(ats32.astype(fnp.float64))
"""),
# birth-node head: per-layer constant rows cstar and 1-|cstar| cached
("""            _bdnh_pending = _bdnh_coefficients(state, _bdnh_A, _bdnh_cstar[l])
""", """            _bdnh_pending = _e2_bdnh_coefficients(state, _bdnh_A, _bdnh_cstar, l)
"""),
("""def _bdnh_coefficients(H, A, cstar):
""", """_E2_BDNH = {}   # E2: per-layer (cstar, 1-|cstar|) rows of the birth-node head, built once per process


def _e2_bdnh_coefficients(H, A, cstar_all, l):
    \"\"\"_bdnh_coefficients(H, A, cstar_all[l]) with the constant rows cstar and (1-|cstar|) cached per layer (same arithmetic).\"\"\"
    c = _E2_BDNH.get(l)
    if c is None or c[0] is not cstar_all:
        cstar = cstar_all[l]
        c = _E2_BDNH[l] = (cstar_all, cstar, 1 - fnp.abs(cstar))
    cstar, onem = c[1], c[2]
    n = int(H.shape[0])
    if H.shape != (n, 64) or A.shape != (64, 4) or cstar.shape != (4,):
        raise ValueError('Birth node-head shape mismatch')
    if any(str(x.dtype) != 'float32' for x in (H, A, cstar)):
        raise TypeError('Birth node-head operands must already be FP32')
    centered = H - fnp.mean(H, axis=0)
    scale = fnp.sqrt(fnp.mean(centered * centered, axis=0) + 1e-6)
    standardized = fnp.clip(centered / scale, -8, 8)
    response = fnp.tanh(standardized @ A)
    coefficients = cstar + onem * response
    return coefficients.T


def _bdnh_coefficients(H, A, cstar):
"""),
]
LIVE = [
("""            omitted = alive & ~response_mask
""", """            omitted = fnp.greater(alive, response_mask)   # E2: == alive & ~response_mask for booleans
"""),
("""                C = C - fnp.outer(m, dm) - fnp.outer(dm, m) - fnp.outer(dm, dm)
""", """                _e2_mdm = fnp.outer(m, dm)
                C = C - _e2_mdm - _e2_mdm.T - fnp.outer(dm, dm)   # E2: outer(dm, m) == outer(m, dm).T exactly
"""),
("""            _e2_alive_sel = alive[selected]
            newborn = (int(pp.shape[1]), selected, _bd_base32[:, selected], (t[selected] * _e2_alive_sel).astype(fnp.float32), (s[selected] * _e2_alive_sel).astype(fnp.float32))
""", """            if (_E2_LAYER.get('ats_a') is a and _E2_LAYER.get('ats_t') is t and _E2_LAYER.get('ats_s') is s
                    and _E2_LAYER.get('ats_alive') is alive):
                _e2_asel, _e2_tsel, _e2_ssel = fnp.unstack(_E2_LAYER['ats32'][:, selected])   # float32 (a, t*alive, s*alive) gathered once
            else:
                _e2_alive_sel = alive[selected]
                _e2_asel = a[selected].astype(fnp.float32)
                _e2_tsel = (t[selected] * _e2_alive_sel).astype(fnp.float32)
                _e2_ssel = (s[selected] * _e2_alive_sel).astype(fnp.float32)
            newborn = (int(pp.shape[1]), selected, _bd_base32[:, selected], _e2_tsel, _e2_ssel)
"""),
("""            _e2_asel = a[selected].astype(fnp.float32)
            lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])
""", """            lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])
"""),
]

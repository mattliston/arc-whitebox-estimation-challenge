# Held common arithmetic; no model imports, arrays or state at module import.
_ASTRA_LMT_LOCAL = [3, 5, 6, 7, 8, 9, 11, 20, 21]
_ASTRA_LMT_COLS = _ASTRA_LMT_LOCAL + list(range(30, 78))

def _astra_lmt_local(m, aux):
    # Conditional d/d alpha at fixed pre variance/cumulants and relational context.
    # Use raw pre_v as the original Edgeworth denominators do, while sigma
    # contains max(pre_v,1e-10). This also honors the active small-v branch.
    s = aux['sigma']; a = aux['alpha']; g = aux['gate']; ph = aux['pdf']
    v = s * s; ratio = v / aux['pre_v']
    t3 = aux['pre_k3'] / (v * s); t4 = aux['pre_k4'] / (v * v)
    u = m / s; pv = aux['post_v'] / v; pk3 = aux['post_k3'] / (v * s)
    a2 = a * a
    du = g + ratio * (t3 * (a2 - 1) * ph / 6 - t4 * (a * a2 - 3 * a) * ph / 24)
    rg2 = (a2 + 1) * g + a * ph
    rg3 = a * (a2 + 3) * g + (a2 + 2) * ph
    r2 = pv + u * u; r3 = pk3 + 3 * u * r2 - 2 * u * u * u
    dr2 = 2 * (aux['gaussian_mean'] / s) - t3 * a * ph / 3 + ratio * t4 * (a2 - 1) * ph / 12
    dr3 = 3 * rg2 + t3 * ph - t4 * a * ph / 4
    dr4 = 4 * rg3 + 4 * t3 * g + t4 * ph
    dv = dr2 - 2 * u * du
    dk3 = dr3 - 3 * du * r2 - 3 * u * dr2 + 6 * u * u * du
    dk4 = dr4 - 4 * du * r3 - 4 * u * dr3 + 12 * u * du * r2 + 6 * u * u * dr2 - 12 * u * u * u * du - 6 * pv * dv
    return fnp.stack((fnp.ones_like(a), ph, -a * ph, g, du, du - g, dv, dk3, dk4), axis=1).astype(fnp.float32)

def _astra_lmt_sigmoid(z):
    # Same mathematical SiLU derivative, without exp(+large) overflow.
    e = fnp.exp(-fnp.abs(z)); den = 1 + e
    return (z >= 0) * (1 / den) + (z < 0) * (e / den)

def _astra_lmt_jvp(dx, H, M, z0, z1, W0, W2, W4, head, rms, gain, scales):
    local = H[:, _ASTRA_LMT_LOCAL]
    dhx = (dx / scales[_ASTRA_LMT_LOCAL]) * ((local > -8) & (local < 8))
    signed = H[:, 30:78]
    dsm = dhx[:, 1:2] * M * ((signed > -8) & (signed < 8))
    dh = fnp.concatenate((dhx, dsm), axis=1)
    dz0 = dh @ W0[_ASTRA_LMT_COLS]
    sg0 = _astra_lmt_sigmoid(z0)
    dh0 = dz0 * (sg0 * (1 + z0 * (1 - sg0)))
    dz1 = dh0 @ W2
    sg1 = _astra_lmt_sigmoid(z1)
    dh1 = dz1 * (sg1 * (1 + z1 * (1 - sg1)))
    return (dh1 @ (W4 @ head)) * rms * gain

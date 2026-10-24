# BEGIN OPUS-ENGINEER BIRTH UPDATE IN FLOAT32 (bd32 v1, 2026-10-01; rounding-level, NOT bit-exact; needs the builder's two loop edits:
# the call passes the loop's float32 copy of bp, and born takes the plane delta without widening): the parent's _bd_update (three
# directions, physical j1 leg, S19 raw legs) runs its (3, n, n) direction stack and the plane delta in float64 and so pays the
# float64 rate on about 27 n x n passes per layer (850 Mflop per network), although both results are stored in float32: the new base
# as it is returned, the delta when born takes it. Here the same operations in the same order run in float32. bp is the loop's
# float32 copy, K21 and Cpre are the layer's retained float32 planes (no conversion is paid), per-unit vectors are formed in float64
# exactly as before and narrowed once. Only the parent's live branch (physical j1 leg set) is carried; anything else falls back.
_BD32_STATE = {}
_BD32_PARENT = _bd_update


def _bd32_state(n):
    st = _BD32_STATE.get(n)
    if st is None:
        D = fnp.empty((3, n, n), dtype=fnp.float32)
        st = dict(D=D, Dv=fnp.unstack(D), D12=D[1:3], diag=fnp.arange(3 * n) % n * (n + 1) + fnp.arange(3 * n) // n * (n * n),
                  old32=fnp.empty((n, n), dtype=fnp.float32))
        _BD32_STATE[n] = st
    return st


def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
    if _BD_J1_PHYS is None or str(bp.dtype) != 'float32':
        return _BD32_PARENT(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)
    _e2_H = _E2Hold()
    _e2_hold = []
    f32 = fnp.float32
    n = int(_e2_H(bp.shape[0]))
    st = _bd32_state(n)
    D = st['D']; D0, D1, D2 = st['Dv']
    c = coefficients.astype(fnp.float64)[:3]
    eta = _e2_H(aux['alpha'] / aux['sigma']).astype(f32)
    half_d = _e2_H(0.5 * d).astype(f32)
    negative_half_g = _e2_H(_e2_H(-0.5) * aux['gate']).astype(f32)
    negative_half_d = -half_d
    K32 = _ps_k2132(K21)
    C32 = _ps_pre32(Cpre)
    denominator = fnp.maximum(_e2_H(fnp.sum(_e2_H(bp * bp), axis=0)), 1e-20)
    _e2_hold.append(fnp.multiply(half_d[:, None], K32, out=D0))
    _e2_hold.append(fnp.multiply(negative_half_g[:, None], K32.T, out=D1))
    _e2_hold.append(fnp.multiply(negative_half_d[:, None], _e2_H(C32 * C32), out=D2))
    _e2_hold.append(fnp.multiply(st['D12'], eta, out=st['D12']))
    fnp.put(D, st['diag'], 0.0)
    # Preserve the unweighted directions until both original projections exist.
    _bl_p0 = _e2_H(fnp.sum(_e2_H(bp * D0), axis=0)) / denominator
    _bl_p2 = _e2_H(fnp.sum(_e2_H(bp * D2), axis=0)) / denominator
    _j1m, _g5_j1tail, _g5_j1col = _g5_j1((1.0, 0.0, 1.0), _BD_J1_PHYS)
    c = (c * _j1m + _g5_j1tail) if c.ndim == 1 else (_g5_mulrow(c, _j1m) + _g5_j1col)
    _bl_effective = c
    if _S19_BD_ADD is not None:
        _bl_raw = fnp.asarray(_S19_BD_ADD[:3], dtype=fnp.float64)
        _bl_effective = c + (_bl_raw if c.ndim == 1 else _e2_H(_bl_raw[:, None]))
    c32 = c.astype(f32)
    _bl_effective = _bl_effective.astype(f32)
    _bl_projection = _e2_H(_e2_H(c32[0]) * _bl_p0) + _e2_H(_e2_H(c32[2]) * _bl_p2)
    _e2_hold.append(fnp.multiply(D, _bl_effective[:, None, None] if c.ndim == 1 else _bl_effective[:, None, :], out=D))
    change = fnp.sum(D, axis=0)
    _e2_hold.append(fnp.subtract(change, _e2_H(bp * _bl_projection), out=change))
    fnp.fill_diagonal(change, 0.0)

    alive_column = alive[:, None]
    old32 = st['old32']
    _e2_hold.append(fnp.multiply(bp, alive_column, out=old32))
    new32 = fnp.empty((n, n), dtype=f32)
    _e2_hold.append(fnp.multiply(bp + change, alive_column, out=new32))
    del change
    difference = new32 - old32

    ats32 = fnp.stack((a, _e2_H(t * alive), _e2_H(s * alive)), dtype=f32)
    _E2_LAYER.update(ats_a=a, ats_t=t, ats_s=s, ats_alive=alive, ats32=ats32)
    a64, t64, s64 = fnp.unstack(_e2_H(ats32.astype(fnp.float64)))
    weight = _e2_H(_g5_chain(a64, _e2_H(t64 - s64), births)).astype(f32)
    quadratic_weight = _e2_H(_g5_chain(a64, _e2_H(_e2_H(t64 * t64) - _e2_H(s64 * s64)), births)).astype(f32)
    delta = _e2_H(_e2_H(_e2_H(new32 + old32) * difference) * weight) + _e2_H(fnp.einsum('i,ji->ij', quadratic_weight, difference, optimize=False))
    fnp.fill_diagonal(delta, 0.0)
    return new32, delta
# END OPUS-ENGINEER BIRTH UPDATE IN FLOAT32

# Broadcast-only request diet. Each einsum has two operands and no reduction.
# Public 0.12.3 out= computes then copies: the five persistent output uses add
# 7*n*n metered copy elements/layer; new32 is separately owned without out=.
# Arithmetic order, dtypes, and every persistent D/old32 owner are retained.
def _o18_bd_literal_column(values):
    key = ('o18_bd_literal_column', tuple(float(x) for x in values))
    value = _G5_CONST.get(key)
    if value is None:
        value = _G5_CONST[key] = _g5_arr(values)[:, None]
    return value

def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
    if _BD_J1_PHYS is None or (bp.dtype) != 'float32':
        return _BD32_PARENT(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)
    _e2_H = _E2Hold(); _e2_append_inline = _e2_H.append; _e2_value_inline = None
    _e2_hold = []
    f32 = fnp.float32
    n = int((_e2_append_inline((_e2_value_inline := bp.shape[0])) or _e2_value_inline))
    st = _bd32_state(n)
    D = st['D']; D0, D1, D2 = st['Dv']
    c = coefficients.astype(fnp.float64)[:3]
    eta = (_e2_append_inline((_e2_value_inline := (aux['alpha'] / aux['sigma']))) or _e2_value_inline).astype(f32)
    half_d = (_e2_append_inline((_e2_value_inline := (0.5 * d))) or _e2_value_inline).astype(f32)
    negative_half_g = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (-0.5))) or _e2_value_inline) * aux['gate']))) or _e2_value_inline).astype(f32)
    negative_half_d = -half_d
    K32 = _ps_k2132(K21)
    C32 = _ps_pre32(Cpre)
    denominator = fnp.maximum((_e2_append_inline((_e2_value_inline := fnp.sum(_e2_append_inline((_e2_value_inline := (bp * bp))) or _e2_value_inline, axis=0))) or _e2_value_inline), 1e-20)
    _e2_hold.append(fnp.einsum('i,ij->ij', half_d, K32, out=D0, optimize=False))
    _e2_hold.append(fnp.einsum('i,ji->ij', negative_half_g, K32, out=D1, optimize=False))
    _e2_hold.append(fnp.einsum('i,ij->ij', negative_half_d, _idc_get('cc32', C32, 1, lambda: C32 * C32), out=D2, optimize=False))
    _e2_hold.append(fnp.multiply(st['D12'], eta, out=st['D12']))
    fnp.put(D, st['diag'], 0.0)
    # Preserve the unweighted directions until both original projections exist.
    _bl_p0 = (_e2_append_inline((_e2_value_inline := fnp.sum(_e2_append_inline((_e2_value_inline := (bp * D0))) or _e2_value_inline, axis=0))) or _e2_value_inline) / denominator
    _bl_p2 = (_e2_append_inline((_e2_value_inline := fnp.sum(_e2_append_inline((_e2_value_inline := (bp * D2))) or _e2_value_inline, axis=0))) or _e2_value_inline) / denominator
    _j1m, _g5_j1tail, _g5_j1col = _g5_j1((1.0, 0.0, 1.0), _BD_J1_PHYS)
    c = (c * _j1m + _g5_j1tail) if c.ndim == 1 else (_g5_mulrow(c, _j1m) + _g5_j1col)
    _bl_effective = c
    if _S19_BD_ADD is not None:
        _bl_raw = _g5_arr(_S19_BD_ADD[:3])
        _bl_effective = c + (_bl_raw if c.ndim == 1 else (_e2_append_inline((_e2_value_inline := _o18_bd_literal_column(_S19_BD_ADD[:3]))) or _e2_value_inline))
    c32 = c.astype(f32)
    _o18_cparts = fnp.unstack(c32)
    _bl_effective = _bl_effective.astype(f32)
    _bl_projection = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := _o18_cparts[0])) or _e2_value_inline) * _bl_p0))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := _o18_cparts[2])) or _e2_value_inline) * _bl_p2))) or _e2_value_inline)
    _e2_hold.append(fnp.einsum('aij,a->aij' if c.ndim == 1 else 'aij,aj->aij', D, _bl_effective, out=D, optimize=False))
    change = fnp.sum(D, axis=0)
    _e2_hold.append(fnp.subtract(change, (_e2_append_inline((_e2_value_inline := (bp * _bl_projection))) or _e2_value_inline), out=change))
    fnp.fill_diagonal(change, 0.0)

    old32 = st['old32']
    _e2_hold.append(fnp.einsum('ij,i->ij', bp, alive, out=old32, optimize=False))
    # The public no-reduction result is owned float32; no persistent owner needs filling.
    new32 = fnp.einsum('ij,i->ij', bp + change, alive, optimize=False)
    del change
    difference = new32 - old32

    ats32 = fnp.stack((a, (_e2_append_inline((_e2_value_inline := (t * alive))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (s * alive))) or _e2_value_inline)), dtype=f32)
    _E2_LAYER.update(ats_a=a, ats_t=t, ats_s=s, ats_alive=alive, ats32=ats32)
    a64, t64, s64 = fnp.unstack((_e2_append_inline((_e2_value_inline := ats32.astype(fnp.float64))) or _e2_value_inline))
    weight = (_e2_append_inline((_e2_value_inline := _g5_chain(a64, _e2_append_inline((_e2_value_inline := (t64 - s64))) or _e2_value_inline, births))) or _e2_value_inline).astype(f32)
    quadratic_weight = (_e2_append_inline((_e2_value_inline := _g5_chain(a64, _e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (t64 * t64))) or _e2_value_inline) - (_e2_append_inline((_e2_value_inline := (s64 * s64))) or _e2_value_inline)))) or _e2_value_inline, births))) or _e2_value_inline).astype(f32)
    delta = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (new32 + old32))) or _e2_value_inline) * difference))) or _e2_value_inline) * weight))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := fnp.einsum('i,ji->ij', quadratic_weight, difference, optimize=False))) or _e2_value_inline)
    fnp.fill_diagonal(delta, 0.0)
    return new32, delta

_ASTRA_DENSE_CONTEXT = tuple((name, _bd_update if name == '_bd_update' else value)
                             for name, value in _ASTRA_DENSE_CONTEXT)

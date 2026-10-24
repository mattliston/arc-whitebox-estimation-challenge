_CVR_RESTORE = True


def _cvr_apply(C, p, star, b, newborn, state, host_v, alive):
    """One centered affine action on every persistent moment owner."""
    current_v = fnp.diag(C)
    valid = (~alive) | ((host_v > 0) & (current_v > 0)
                        & fnp.isfinite(host_v) & fnp.isfinite(current_v))
    if not bool(fnp.all(valid)):
        raise RuntimeError('CVR strictly positive finite active variances')
    # where protects inactive coordinates, including exact zero variances.
    numerator = fnp.where(alive, host_v, 1.0)
    denominator = fnp.where(alive, current_v, 1.0)
    scale = fnp.sqrt(numerator / denominator)
    if not _CVR_RESTORE:
        scale = fnp.ones(len(host_v), dtype=fnp.float64)
    if not bool(fnp.all(fnp.isfinite(scale))):
        raise RuntimeError('CVR finite scale')
    scale32 = scale.astype(fnp.float32)
    C = (scale[:, None] * C) * scale[None, :]
    p = scale32[:, None] * p
    star = (scale32[:, None] * star) * (scale32 * scale32)[None, :]
    b = scale * b
    inherited, selected, base, t, s = newborn
    base = scale32[:, None] * base
    t = scale32[selected] * t
    s = scale32[selected] * s
    newborn = (inherited, selected, base, t, s)
    if state['columns']:
        if state['pre_basis'] is not None:
            raise RuntimeError('CVR applied only after shared commit')
        state['Q'] = scale32[:, None] * state['Q']
    return C, p, star, b, newborn, scale

# Fixed L10 within-span energy ordering. No sketch/rank/capacity increase.
# The existing native Nystrom core represents total lambda^2 column energy,
# a proxy for common-u energy, NOT the exact Gram of the pair means alone.
_CUO_STATS = dict(calls=0, layer=10, rank=320, common_u_rank=256)


def _cuo_order(Q, core):
    if tuple(core.shape) != (320, 320) or int(Q.shape[1]) != 320 or _UVP_RU != 256:
        raise RuntimeError('common-u ordering fixed L10 scope')
    values, vectors = fnp.linalg.eigh(core.astype(fnp.float64))
    order = fnp.argsort(-values, kind='stable')
    V = vectors[:, order]
    # Each sign is fixed by the first maximum-magnitude component.
    pivots = fnp.argmax(fnp.abs(V), axis=0)
    pivot_values = fnp.take_along_axis(V, pivots[None, :], axis=0)[0]
    signs = fnp.where(pivot_values >= 0, 1.0, -1.0)
    V = (V * signs[None, :]).astype(fnp.float32)
    rotated = Q @ V
    # Retain every signed eigenvalue; there is no positivity repair or cutoff.
    ordered_core = fnp.diag(values[order]).astype(fnp.float32)
    _CUO_STATS['calls'] += 1
    return rotated, ordered_core

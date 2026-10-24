# Coordinate permutation only; all320 directions and full signed core retained.
_CUD_STATS = dict(calls=0, layer=10, rank=320, common_u_rank=256)

def _cud_order(Q, core):
    if tuple(core.shape)!=(320,320) or int(Q.shape[1])!=320 or _UVP_RU!=256:
        raise RuntimeError('common-u diagonal ordering fixed L10 scope')
    # Ties preserve the incumbent QR coordinate order. No sign flips or clipping.
    order=fnp.argsort(-fnp.diag(core),kind='stable')
    ordered_Q=Q[:,order]
    ordered_core=core[order,:][:,order]
    _CUD_STATS['calls']+=1
    return ordered_Q,ordered_core

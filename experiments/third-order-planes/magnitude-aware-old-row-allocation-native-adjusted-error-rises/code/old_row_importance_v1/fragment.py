# Fixed old-local-row allocation. The complete carried tensor is unchanged.
_ORI_STATS = []


def _ori_select(U, ctx):
    a = ctx['a']
    m = int(U.shape[0])
    base_score = _OTT_Z - a
    negative_dead = (a < -_OTT_NEG_Z) & (~ctx['live'])
    base_score = fnp.where(negative_dead, a + _OTT_NEG_Z, base_score)
    base_kept = min(m, (int(fnp.nonzero(base_score >= 0.0)[0].shape[0]) + 63) // 64 * 64)
    mandatory = (a < 0.0) & ctx['live']
    mandatory_count = int(fnp.nonzero(mandatory)[0].shape[0])
    requested = max(0, base_kept - 128)
    kept = min(m, max(64, ((requested + 63) // 64) * 64,
                      ((mandatory_count + 63) // 64) * 64))
    v = _E2_LAYER['vc']
    if ctx['rows'] is not None:
        v = v[ctx['rows']]
    wide = U.astype(fnp.float64)
    leverage = fnp.sum(wide * wide, axis=1) / fnp.maximum(v, 1e-10)
    relative = leverage / (fnp.mean(leverage) + 1e-30)
    priority = -0.5 * a * a + fnp.log(fnp.maximum(relative, 1e-30))
    priority = fnp.where(mandatory, fnp.max(priority) + 1.0, priority)
    ordered = fnp.argsort(-priority, kind='stable')
    _ORI_STATS.append(dict(rows=m, base_kept=base_kept, kept=kept,
                           mandatory=mandatory_count, old_rank=int(U.shape[1])))
    return kept, ordered[:kept], ordered[kept:]

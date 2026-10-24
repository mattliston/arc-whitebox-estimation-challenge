_RGT_ENABLED = True


def _rgt_rebase_history(layer, mu, host, aux, cp_pre, inherited):
    if layer == 0 or not _RGT_ENABLED:
        return aux['gate'], inherited
    variance = aux['pre_v']
    raw2 = aux['raw2']
    if not bool(fnp.all(fnp.isfinite(variance))) or bool(fnp.any(variance <= 0)):
        raise RuntimeError('RGT requires finite positive pre-variance')
    if not bool(fnp.all(fnp.isfinite(raw2))) or not bool(fnp.all(fnp.isfinite(mu))) or not bool(fnp.all(fnp.isfinite(host))):
        raise RuntimeError('RGT finite pre-head scalar moments')
    raw_gain = (raw2 - mu * host) / variance
    if not bool(fnp.all(fnp.isfinite(raw_gain))):
        raise RuntimeError('RGT nonfinite regression gain')
    q = fnp.clip(raw_gain, 0.0, 1.0)
    return q, q[:, None] ** 2 * q[None, :] * cp_pre

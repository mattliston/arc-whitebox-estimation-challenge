# Frozen current-card port of the proven elementwise feature grouping.
# Append before the permanent no-governor tail; no cache or dense identity guard changes.
assert _RE22_PARENT_FEATURES.__name__ == 'features'
assert features is _re22_features


# BEGIN [opus-vecB] FEATURES REQUEST GROUPS v1 (2026-10-17; append LAST): the per-layer feature vector with (a) the six float64 vector
# divisions ((m-gm)/s, pre_k3/vs, post_v/v, pre_k4/vsq, post_k3/vs, post_k4/vsq) as ONE stacked divide and (b) the four float32 square
# roots (rms of rho, the two skplane column/row norms, rms of postrho) as ONE stacked sqrt. Same IEEE operation per element on the same
# operands and dtypes (asserted), so the feature matrix is bit-identical; -3 requests per layer. The RE22 wrapper keeps calling its parent.
def _vecb_features(l, m, C, K21, aux, W):
    _e2_H = _E2Hold(); _h = _e2_H.append
    def keep(x):
        _h(x); return x
    n = len(m)
    c = _e2_feature_columns(n, l)
    s = aux['sigma']
    v = s * s
    pr = _E2_LAYER.get('pair')
    pr = pr if pr is not None and pr['aux'] is aux else None
    _ps_s32 = _ps32(s); _ps_v32 = _ps32(v)
    rho = _ps_pre32(aux['pre_cov']) / (pr['ss32'] if pr else fnp.outer(_ps_s32, _ps_s32))
    skplane = _ps_k2132(K21) / keep(fnp.outer(_ps_v32, _ps_s32))
    skplane_square = skplane * skplane
    postscale = fnp.sqrt(keep(fnp.maximum(aux['post_v'], 1e-12)))
    _ps_postscale32 = _ps32(postscale)
    postrho = _ps32(C) / keep(fnp.outer(_ps_postscale32, _ps_postscale32))
    postrho = fnp.clip(postrho, keep(-1), 1)
    ws = _e2_wsums(keep(_e2_wpow(W)))
    ls = fnp.log(s)
    ms = _e2_ms(m, s)
    vs = v * s
    vsq = v * v
    rootn = float(n) ** 0.5
    gm = aux['gaussian_mean']
    num = keep(fnp.stack((keep(m - gm), aux['pre_k3'], aux['post_v'], aux['pre_k4'], aux['post_k3'], aux['post_k4'])))
    den = keep(fnp.stack((s, vs, v, vsq, vs, vsq)))
    assert (num.dtype) == 'float64' and (den.dtype) == 'float64'
    q_mgm_s, q_k3_vs, q_pv_v, q_k4_vsq, q_pk3_vs, q_pk4_vsq = fnp.unstack(keep(num / den))
    r_rho = keep(fnp.mean(keep(rho * rho), axis=1)); r_sk1 = keep(fnp.sum(skplane_square, axis=1)); r_sk0 = keep(fnp.sum(skplane_square, axis=0))
    r_post = keep(fnp.mean(keep(postrho * postrho), axis=1))
    roots = keep(fnp.stack((r_rho, r_sk1, r_sk0, r_post)))
    assert (roots.dtype) == 'float32'
    rms_rho, nrm_sk1, nrm_sk0, rms_post = fnp.unstack(keep(fnp.sqrt(roots)))
    vals = (c[0], c[1], c[2], aux['alpha'], keep(ls - keep(fnp.mean(ls))), aux['gate'], aux['pdf'], pr['mgs'] if pr else gm / s, ms, q_mgm_s,
            q_k3_vs, q_pv_v, q_k4_vsq, keep(fnp.mean(rho, axis=1)), rms_rho, keep(fnp.mean(keep(fnp.abs(rho)), axis=1)), keep(ws[0]), keep(ws[1]), keep(ws[2]),
            keep(ws[3]), q_pk3_vs, q_pk4_vsq, nrm_sk1, nrm_sk0, keep(keep(fnp.sum(skplane, axis=1)) / rootn), keep(keep(fnp.sum(skplane, axis=0)) / rootn),
            keep(keep(rho @ aux['alpha']) / n), rms_post, keep(fnp.mean(postrho, axis=1)), keep(c[2] * keep(fnp.mean(ms))))
    return fnp.stack(vals, axis=1, dtype=fnp.float32)


_RE22_PARENT_FEATURES = _vecb_features
# END [opus-vecB] FEATURES REQUEST GROUPS v1

def _gt_cp_gate(aux):
    alpha = aux['alpha']
    sigma = aux['sigma']
    variance = aux['pre_v']
    skew = aux['pre_k3'] / (sigma * sigma * sigma)
    kurt = aux['pre_k4'] / (variance * variance)
    alpha2 = alpha * alpha
    correction = aux['pdf'] * (skew * (alpha2 - 1) / 6
                    - kurt * (alpha2 * alpha - 3 * alpha) / 24)
    return fnp.clip(aux['gate'] + correction, 0.0, 1.0)


def _igt_rebase_history(layer, aux, cp_pre, inherited):
    if layer == 0:
        return aux['gate'], inherited
    q = _gt_cp_gate(aux)
    return q, q[:, None] ** 2 * q[None, :] * cp_pre

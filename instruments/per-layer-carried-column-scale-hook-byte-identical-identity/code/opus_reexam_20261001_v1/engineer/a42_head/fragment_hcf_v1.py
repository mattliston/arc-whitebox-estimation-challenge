# BEGIN OPUS-ENGINEER FINAL-LAYER HEAD COLUMNS (a42 hcf v1, 2026-10-10; a change of the final head's INPUT, OFF unless weights are given).
# The final layer's learned correction is gnn_step at layer 15 (_predict_learned_compact's scalar_terminal_hook: the same enc0 as
# layers 1-14, with layer 15's centres, scales and head); its first layer gets
#     enc0_b (+ the head columns of fragment_hc_v2, none at layer 15) + sum_k z_k u_k',
#     z_k = clip((x_k - mean) / sd, -8, 8) over the column's own units (0 elsewhere), u_k = 128 weights of the FINAL layer,
# through a wrapper of _hc_bias (so the file needs the HC lever). Columns, on the KEPT ROWS of the final layer's row cut
# (|mu / sigma| <= _L15_ROWCUT_Z: the rows the bank is pushed into layer 15 on; the convention asked of the trainer at 2026-10-10T02:14Z):
#   'k4_local15'   K4L's per-unit value at layer 15 over v^2 (the k4_15 term the file already computes: no bill)
# (written next, when the trainer's definitions are fixed: 'diag15', the law diagonal with the weights advanced through layer 14;
#  'k31l_diag15', the young pairs' plane diagonal at layer 15.)
_HCF_COLUMNS = ()        # the columns, in the order of the rows of _HCF_U
_HCF_U = None            # K x 128 weights (a tuple of K tuples of float32 values); None = no column
_HCF_STATE = dict(raw={})
_HCF_CONST = {}
_HCF_STATS = dict(columns=0, used=0)


def _hcf_put(name, vector, mask, count):
    """A final-layer column's per-unit vector (float64), its unit mask (float64 0 / 1) and the number of its units."""
    if _HCF_U is None or name not in _HCF_COLUMNS:
        return
    _HCF_STATE['raw'][name] = (vector, mask, int(count))
    _HCF_STATS['columns'] += 1


def _hcf_k4(dk, rows, v):
    """Column 'k4_local15' from the final layer's K4L value dk (0 off the kept rows) and the kept rows' indices."""
    if _HCF_U is None or 'k4_local15' not in _HCF_COLUMNS:
        return
    hold = _E2Hold()
    n = int(v.shape[0])
    mask = fnp.zeros(n, dtype=fnp.float64)
    fnp.put(mask, rows, 1.0)
    _hcf_put('k4_local15', hold(dk / hold(v * v)), mask, int(rows.shape[0]))


def _hcf_bias(bias):
    """enc0_b, or enc0_b + sum_k z_k u_k' with the final layer's columns (a column the network did not make counts as 0)."""
    present = dict(_HCF_STATE['raw'])
    _HCF_STATE['raw'] = {}
    if _HCF_U is None or not present:
        return bias
    hold = _E2Hold()
    K = _HCF_CONST
    n = int(next(iter(present.values()))[0].shape[0])
    if K.get('n') != n:
        K.update(n=n, u=fnp.asarray(_HCF_U, dtype=fnp.float32), zeros=fnp.zeros(n, dtype=fnp.float64))
    raw = [present.get(name, (K['zeros'], K['zeros'], n)) for name in _HCF_COLUMNS]
    M = hold(fnp.stack([mask for _, mask, _ in raw]))
    X = hold(hold(fnp.stack([hold(x) for x, _, _ in raw])) * M)
    inverse = hold(fnp.asarray([[1.0 / max(count, 1)] for _, _, count in raw]))
    root = hold(fnp.asarray([[max(count - 1.0, 0.0) ** 0.5] for _, _, count in raw]))
    centred = hold(hold(X - hold(hold(fnp.sum(X, axis=1, keepdims=True)) * inverse)) * M)
    scale = hold(hold(fnp.power(hold(fnp.maximum(hold(fnp.sum(hold(centred * centred), axis=1, keepdims=True)), 1e-300)), -0.5)) * root)
    Z = hold(hold(fnp.clip(hold(centred * scale), -8.0, 8.0)).astype(fnp.float32))
    _HCF_STATS['used'] += 1
    return bias + hold(fnp.einsum('kn,kc->nc', Z, K['u'], optimize=False))
# END OPUS-ENGINEER FINAL-LAYER HEAD COLUMNS


# The final layer's K4L value is made by _k4l_final (K4L fragment); wrapped here so that its kept rows are known (the parent consumes
# _K4L_FINAL, so the rows are read first). Appended after the K4L fragment.
_HCF_PARENT_K4L_FINAL = _k4l_final


def _k4l_final(mu, v):
    found = _K4L_FINAL[0]
    dk = _HCF_PARENT_K4L_FINAL(mu, v)
    if _HCF_U is not None and found is not None and found[1] is not None and not isinstance(dk, float):
        _hcf_k4(dk, found[0], v)
    return dk


_HCF_PARENT_HC_BIAS = _hc_bias


def _hc_bias(l, bias):
    out = _HCF_PARENT_HC_BIAS(l, bias)
    return _hcf_bias(out) if l == 15 else out

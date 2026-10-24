"""One-hop, triple-distinct Gaussian H2-H2-H2 cycle, fixed births L8..14.

The exact Gaussian cycle is d_i*d_j*d_k*C_ij*C_jk*C_ki. Retain the
single covariance direction C@1/sqrt(1.T@C@1), then remove every repeated
index before its next-layer diagonal contraction. This is a rank-one
surrogate, not the full cycle or a full third-tensor transport.
"""
_TRI_ACTIVE = [False]
_TRI_STATE = {}
_TRI_STATS = dict(birth_layers=[], push_layers=[], fallback_layers=[],
                  sum_abs_delta=0.0, max_abs_delta=0.0, retired=True)
_TRI_PARENT_BINARY = _predict_binary


def _tri_birth(l, Cpre, d):
    if not _TRI_ACTIVE[0] or not 8 <= l <= 14:
        return
    assert not _TRI_STATE
    _TRI_STATS['birth_layers'].append(int(l))
    hold = _E2Hold()
    row = fnp.sum(Cpre, axis=1)
    energy = float(hold(fnp.sum(row)))
    if int(row.shape[0]) < 3 or not 0.0 < energy < float('inf'):
        _TRI_STATE.update(layer=int(l), a=None)
        _TRI_STATS['fallback_layers'].append(int(l))
        return
    # a=d*u^2; evaluate without an unnecessary square root.
    a = hold(d * hold(row * row)) / energy
    _TRI_STATE.update(layer=int(l), a=a)


def _tri_push(l, w, k3):
    if not _TRI_ACTIVE[0] or not 9 <= l <= 15:
        return k3
    assert _TRI_STATE.get('layer') == l - 1
    a = _TRI_STATE.pop('a')
    _TRI_STATE.clear()
    _TRI_STATS['push_layers'].append(int(l))
    if a is None:
        return k3
    assert str(w.dtype) == str(a.dtype) == str(k3.dtype) == 'float64'
    hold = _E2Hold()
    a2 = hold(a * a)
    w2 = hold(w * w)
    p = hold(w.T @ a)
    q = hold(w2.T @ a2)
    r = hold(hold(w2 * w).T @ hold(a2 * a))
    # Inclusion-exclusion: all triples minus three equal-pair sets plus
    # twice the all-equal intersection. Signed weights are retained.
    delta = hold(hold(hold(p * p) * p) - hold(3.0 * hold(p * q))) + hold(2.0 * r)
    magnitude = hold(fnp.abs(delta))
    _TRI_STATS['sum_abs_delta'] += float(hold(fnp.sum(magnitude)))
    _TRI_STATS['max_abs_delta'] = max(_TRI_STATS['max_abs_delta'], float(hold(fnp.max(magnitude))))
    return k3 + delta


def _predict_binary(mlp, *args, **kwargs):
    if int(mlp.width) != 1024 or int(mlp.depth) != 16 or kwargs.get('final_hook') is None:
        return _TRI_PARENT_BINARY(mlp, *args, **kwargs)
    if _TRI_ACTIVE[0] or _TRI_STATE:
        raise RuntimeError('Triangle one-hop scope leaked')
    _TRI_STATS.clear()
    _TRI_STATS.update(birth_layers=[], push_layers=[], fallback_layers=[],
                      sum_abs_delta=0.0, max_abs_delta=0.0, retired=False)
    _TRI_ACTIVE[0] = True
    try:
        answer = _TRI_PARENT_BINARY(mlp, *args, **kwargs)
        assert not _TRI_STATE
        return answer
    finally:
        _TRI_ACTIVE[0] = False
        _TRI_STATE.clear()
        _TRI_STATS['retired'] = True

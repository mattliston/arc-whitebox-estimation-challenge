# Fixed c15-minus128, rank8 young-tail projection; exact old contribution.
_Y8_ON = True
_Y8_ACTIVE = [False]
_Y8_CONST = {}
_Y8_LITERAL = __PROBE_LITERAL__
_Y8_STATS = dict(reads=[], retired=True)
_Y8_PARENT_READ = _RC_PARENT_READ
_Y8_PARENT_PREDICT = Estimator.predict
_Y8_PARENT_CS_BUILD = _cs_build
_CS_DICTS = tuple(_CS_DICTS) + ('_Y8_CONST',)

def _y8_probes():
    if not _Y8_CONST:
        _Y8_CONST.update(_unpack_static_fp32(_Y8_LITERAL))
    probes = _Y8_CONST['probes']
    assert tuple(probes.shape) == (1024, 6) and str(probes.dtype) == 'float32'
    return probes

def _cs_build(n=1024, depth=16):
    result = _Y8_PARENT_CS_BUILD(n, depth)
    if _Y8_ON:
        _y8_probes()
    return result

def _y8_basis(prefix, inner, old_right, vc, mu, hold):
    m, c = map(int, prefix.shape)
    assert tuple(vc.shape) == tuple(mu.shape) == (m,)
    probes = hold(_y8_probes()[:c])
    young_action = hold(prefix @ probes)
    if inner is not None:
        assert tuple(old_right.shape) == (c, int(inner.shape[1]))
        young_action = hold(young_action - hold(inner @ hold(old_right.T @ probes)))
    else:
        assert old_right is None
    columns = hold(fnp.concatenate((hold(vc[:, None]), hold(mu[:, None]), young_action), axis=1))
    columns64 = hold(columns.astype(fnp.float64))
    assert bool(hold(fnp.all(hold(fnp.isfinite(columns64)))))
    norms = hold(fnp.linalg.norm(columns64, axis=0))
    denominator = hold(fnp.where(hold(norms > 0.0), norms, 1.0))
    normalized = hold(columns64 / denominator)
    q, r = fnp.linalg.qr(normalized, mode='reduced')
    q, r = hold(q), hold(r)
    threshold = 32.0 * 2.220446049250313e-16 * max(m, 8)
    full_rank = int(q.shape[1]) == 8 and bool(hold(fnp.all(hold(fnp.abs(hold(fnp.diag(r))) > threshold))))
    if not full_rank:
        return hold(_rtp_qr_basis(vc, mu, hold)), True
    return hold(q.astype(fnp.float32)), False

def _y8_read(left, young_right, lam, live_indices, state):
    if not _Y8_ON or not _Y8_ACTIVE[0] or not _RTT_ACTIVE:
        return _Y8_PARENT_READ(left, young_right, lam, live_indices, state)
    layer = int(_RS_LAYER[0]); assert 1 <= layer <= 14
    m, k, c2 = int(left.shape[0]), int(young_right.shape[1]), int(young_right.shape[0])
    assert int(live_indices.shape[0]) == c2
    old = int(state['columns'])
    rank = int(state['pre_basis'].shape[1]) if old else 0
    assert int(left.shape[1]) == old + k and tuple(lam.shape) == (old + k,)
    floor = .5 * _rc_math.erfc(1.5 / _rc_math.sqrt(2.0))
    original = min(c2, (int(fnp.nonzero(_E2_LAYER['g'] >= floor)[0].shape[0]) + 31) // 32 * 32)
    record = dict(layer=layer, m=m, k=k, c15=original, c15_parent=original, c2=c2,
                  old_rank=rank, changed=original<c2, basis_rank=None, old_inner_calls=0,
                  forced_d7=False, refine_rank_in=None, refine_rank_out=None,
                  refine_qr_rank=None, refine_fallback=None, refine_old_rank=None)
    if original == c2:
        _RTT_STATS['reads'].append(record)
        return _RTT_PARENT_READ(left, young_right, lam, live_indices, state)
    c = max(32, original - 128)
    assert 0 < c < c2 and c % 32 == 0 and _RTT_CONTEXT[0] is None
    assert _YR7_SHAPES == _RTT_FIXED_YR7_SHAPES
    context = dict(state=state, m=m, inner=None, inner_calls=0)
    hold = _E2Hold(); _RTT_CONTEXT[0] = context
    try:
        # Preserve submitted D6/natural-D7 dispatch; no whitelist promotion.
        prefix = hold(_RTT_PARENT_READ(left, hold(young_right[:c]), lam, hold(live_indices[:c]), state))
        assert tuple(prefix.shape) == (m, c)
        rows = _E2_LAYER['ecut_rows']; assert tuple(rows.shape) == (m,)
        vc, mu = hold(_E2_LAYER['vc'][rows]), hold(_E2_LAYER['mu'][rows])
        inner = context['inner']
        if old:
            assert context['inner_calls'] == 1 and inner is not None
            prefix_old = hold(_xsg_rows(5119, state['pre_basis'], hold(live_indices[:c])))
        else:
            assert context['inner_calls'] == 0 and inner is None
            prefix_old = None
        Q, fallback = _y8_basis(prefix, inner, prefix_old, vc, mu, hold)
        qrank = int(Q.shape[1]); assert 0 <= qrank <= 8
        if qrank:
            scores = hold(Q.T @ hold(left[:, old:]))
            weighted = hold(hold(young_right[c:c2]) * hold(lam[old:]))
            tail = hold(Q @ hold(scores @ hold(weighted.T)))
        else:
            tail = hold(fnp.zeros((m, c2-c), dtype=fnp.float32))
        if old:
            tail_old = hold(_xsg_rows(5119, state['pre_basis'], hold(live_indices[c:c2])))
            tail = hold(tail + hold(inner @ hold(tail_old.T)))
        record.update(c15=c, basis_rank=qrank, old_inner_calls=context['inner_calls'],
                      refine_rank_in=8, refine_rank_out=qrank, refine_qr_rank=qrank,
                      refine_fallback=bool(fallback), refine_old_rank=rank)
        _RTT_STATS['reads'].append(record)
        _Y8_STATS['reads'].append(dict(layer=layer, m=m, k=k, original=original,
                                     prefix=c, c2=c2, rank=qrank, fallback=bool(fallback),
                                     old_rank=rank, old_inner_calls=context['inner_calls']))
        return fnp.concatenate((prefix, tail), axis=1)
    finally:
        _RTT_CONTEXT[0] = None
        assert _YR7_SHAPES == _RTT_FIXED_YR7_SHAPES

_RC_PARENT_READ = _y8_read

def _y8_predict(self, *args, **kwargs):
    assert not _Y8_ACTIVE[0]
    _Y8_ACTIVE[0] = bool(_Y8_ON)
    _Y8_STATS['reads'].clear(); _Y8_STATS['retired'] = False
    try:
        return _Y8_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _Y8_ACTIVE[0] = False
        _Y8_STATS['retired'] = True

Estimator.predict = _y8_predict

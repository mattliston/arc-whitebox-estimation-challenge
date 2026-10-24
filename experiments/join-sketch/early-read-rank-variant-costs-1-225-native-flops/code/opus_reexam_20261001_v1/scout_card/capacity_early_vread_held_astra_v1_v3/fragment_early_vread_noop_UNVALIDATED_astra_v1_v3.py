# HELD NEW PHYSICS: full L10 rank384 state, prefix v-read320 only L10--13.
# The omitted coordinates320:384 belong to the CURRENT index-sorted384 QR
# frame. They are neither the old320 Nystrom span nor certified weak axes.
# Common256, reconstruction, Q/F/core/pre_basis, tail trace and signed roster
# are unchanged. Only early response reads are approximated. No gain claim.
# v3: preserve the actual active plain/RLS/RLF reader chain. Original full
# rank determines fused dispatch; only old-inner/right contractions change.
# All young scaling, holds, kernel plan and unwind code remain exact.
# Read-column cropping, read scales and form restoration remain outside.
_ASTRA_EVR_ON = False
_ASTRA_EVR_LAYERS = (10, 11, 12, 13)
_ASTRA_EVR_PARENT_READ = _RC_PARENT_READ
_ASTRA_EVR_STATS = dict(calls=0, events=[])


def _astra_evr_inner(old_left, lam_cols, uv):
    Fu, Fv2, blocks, ru = uv
    assert ru == 256 and int(Fv2.shape[0]) == 384
    rows, P = int(old_left.shape[0]), int(Fu.shape[1])
    a_ = old_left * lam_cols[None, :]
    Su = fnp.empty((rows, P), dtype=fnp.float32)
    Dv = fnp.empty((rows, P), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Su[:, b:b + h]))
        h_(fnp.subtract(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Dv[:, b:b + h]))
        b += h
    _BH_HOLD.clear()
    del a_
    # Distinct layout tag: cannot pretend the320 and384 B trees are identical.
    inner = _ubx_product(Dv, Fv2[:320].T, uv, 'vt_early320')
    inner_u = _ubx_product(Su, Fu.T, uv, 'ut')
    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))
    _BH_HOLD.clear()
    del Su, Dv, inner_u
    return inner


def _astra_evr_record(inner, state, uv):
    group = state['lazy']
    layer = _UVP_CUR[0]
    assert int(state['rank']) == 384 and int(state['pre_basis'].shape[1]) == 384
    assert (layer in (10, 11, 12) and group is None) or (layer == 13 and group['layer'] == 13)
    assert int(inner.shape[1]) == (320 if group is None else 384)
    _ASTRA_EVR_STATS['calls'] += 1
    _ASTRA_EVR_STATS['events'].append(dict(layer=layer, prefix_v_read_rank=320,
        full_state_rank=int(state['rank']), prefix_rank=384, common_rank=256,
        lazy_prefix=group is not None, rows=int(inner.shape[0]),
        prefix_pair_columns=int(uv[1][0].shape[1]), trailing_rank=int(inner.shape[1])))
    return inner


def _astra_evr_slr_inner(old_left, lam_old, state):
    """= _slr_inner with old_left already squared, in u/v form (uvp)."""
    _e2_H = _E2Hold()
    columns = int(state['columns'])
    _slr_layout(state, columns)
    _slr_require(len(old_left.shape) == 2 and int(_e2_H(old_left.shape[1])) == columns and (tuple(lam_old.shape) == (columns,)) and (str(old_left.dtype) == str(lam_old.dtype) == 'float32'), 'complete original weighted repeated-index read')
    if _UVP_CUR[0] not in _UVP_ONLY_LAYERS:
        return _UVP_ORIG_READ(old_left, lam_old, state)
    uv = _uvp_state(state)
    _UVP_STATS['read'] += 1
    if uv[0] == 'm':
        return _astra_evr_record(_astra_evr_inner(old_left, lam_old, uv[1]), state, uv)
    group = state['lazy']
    split = int(group['prefix'])
    inner0 = _astra_evr_inner(_e2_H(old_left[:, :split]), _e2_H(lam_old[:split]), uv[1])
    inner1 = _uvp_inner(_e2_H(old_left[:, split:]), _e2_H(lam_old[split:]), uv[2])
    mapped = _shared_old384_mm(inner0, _e2_H(group['map'][:, :320].T))
    _shared_old384_finite(mapped, 'lazy mapped response inner')
    inner = mapped + inner1
    del inner0, inner1, mapped
    return _astra_evr_record(inner, state, uv)


def _astra_evr_basis_rows(state, live_indices):
    right = _xsg_rows(5119, state['pre_basis'], live_indices)
    return right[:, :320] if state['lazy'] is None else right


def _astra_evr_plain_read(left, young_right, lam, live_indices, state):
    # E-48 --colpack E3: = _e2_shared_old384_read_young with left already squared ((pp*pp)[response rows]); the same products
    old_columns = int(state['columns'])
    if not old_columns:
        return _mm(left, (young_right * lam).T)
    old_left = left[:, :old_columns]
    inner = _astra_evr_slr_inner(old_left, lam[:old_columns], state)
    old_plane = _shared_old384_mm(inner, _astra_evr_basis_rows(state, live_indices).T)   # opus-engineer a62 XSG
    young_left = left[:, old_columns:]
    young_plane = _mm(young_left, (young_right * lam[old_columns:]).T)
    return young_plane + old_plane


def _astra_evr_lambda_read(left, young_right, lam, live_indices, state):
    m, n = int(left.shape[0]), int(young_right.shape[0])
    if m >= n:
        _ASTRA_RLS_STATS['parent'] += 1
        return _astra_evr_plain_read(left, young_right, lam, live_indices, state)
    old_columns = int(state['columns'])
    width = int(young_right.shape[1])
    _ASTRA_RLS_STATS['switched'] += 1
    _ASTRA_RLS_STATS['scalar_products_saved'] += (n - m) * width
    if not old_columns:
        return _mm(left * lam, young_right.T)
    old_left = left[:, :old_columns]
    inner = _astra_evr_slr_inner(old_left, lam[:old_columns], state)
    old_plane = _shared_old384_mm(inner, _astra_evr_basis_rows(state, live_indices).T)
    young_left = left[:, old_columns:]
    young_plane = _mm(young_left * lam[old_columns:], young_right.T)
    return young_plane + old_plane


def _astra_evr_fused_read(left, young_right, lam, live_indices, state):
    old_columns = int(state['columns'])
    m, n, k = int(left.shape[0]), int(young_right.shape[0]), int(young_right.shape[1])
    rank = int(state['pre_basis'].shape[1]) if old_columns else 0
    plan = None
    if (_ASTRA_RLF_ON and old_columns and rank > 0 and rank % 64 == 0
            and _WINOGRAD and _AB_ON and _SCRATCH is not None
            # The incumbent dispatches these shapes to private depth-7
            # read trees before its generic K-split path. Keep that route.
            and not ((m, k, n) in _R7_SHAPES
                     and int(_SCRATCH[0].shape[0]) == 391602176)
            and 0 < m <= 1024 and 0 < n <= 1024 and k >= 2048
            and str(left.dtype) == str(young_right.dtype) == 'float32'
            and _ASTRA_RLF_CONTEXT[0] is None and _UBX_CTX[0] is None
            and _UBC_CONTEXT[0] is None):
        plan = _ks_plan(m, k, n)
        # This is exactly the incumbent _mm's multi-chunk dispatch domain.
        if plan is not None and (len(plan[2]) < 2 or not _ipt_fits(6, plan[0], rank, plan[1])):
            plan = None
    if plan is None:
        _ASTRA_RLF_STATS['parent'] += 1
        return _astra_evr_lambda_read(left, young_right, lam, live_indices, state)
    hold = _E2Hold()
    inner = hold(_astra_evr_slr_inner(hold(left[:, :old_columns]), hold(lam[:old_columns]), state))
    right = hold(hold(_astra_evr_basis_rows(state, live_indices)).T)
    # Original full-state rank decided dispatch; actual old-read rank now follows inner.
    rank = int(inner.shape[1])
    if tuple(inner.shape) != (m, rank) or tuple(right.shape) != (rank, n):
        raise RuntimeError('ASTRA read fusion: old trailing factor shapes')
    young_left = hold(left[:, old_columns:])
    young_lam = hold(lam[old_columns:])
    if m < n:
        a, b = hold(young_left * young_lam), hold(young_right.T)
    else:
        a, b = young_left, hold(hold(young_right * young_lam).T)
    M, N = plan[:2]
    hzr, hzc = (bool(plan[3]), bool(plan[4])) if len(plan) == 6 else (False, False)
    context = dict(inner=inner, right=right, M=M, N=N, hzr=hzr, hzc=hzc, done=False)
    _ASTRA_RLF_CONTEXT[0] = context
    try:
        # Original young chunk sizes and strip layout are retained; the
        # callback adds the old product just before this one final unwind.
        result = _ks_product(a, b, fnp.empty((m, n), dtype=fnp.float32), m, k, n, plan)
        if not context['done']:
            raise RuntimeError('ASTRA read fusion: final unwind was not reached')
        _ASTRA_RLF_STATS['fused'] += 1
        _ASTRA_RLF_STATS['shapes'].append((m, k, n, rank, M, N, hzr, hzc))
        return result
    finally:
        _ASTRA_RLF_CONTEXT[0] = None


def _astra_evr_read(left, young_right, lam, live_indices, state):
    if not _ASTRA_EVR_ON:
        return _ASTRA_EVR_PARENT_READ(left, young_right, lam, live_indices, state)
    if _UVP_CUR[0] not in _ASTRA_EVR_LAYERS or not int(state['columns']):
        return _ASTRA_EVR_PARENT_READ(left, young_right, lam, live_indices, state)
    group = state['lazy']
    rank = int(state['rank']) if group is None else int(group['base_rank'])
    if rank != 384:
        return _ASTRA_EVR_PARENT_READ(left, young_right, lam, live_indices, state)
    return _astra_evr_fused_read(left, young_right, lam, live_indices, state)


_RC_PARENT_READ = _astra_evr_read
_ASTRA_EVR_PARENT_PREDICT = _predict_binary


def _predict_binary(*args, **kwargs):
    _ASTRA_EVR_STATS.clear()
    _ASTRA_EVR_STATS.update(calls=0, events=[])
    return _ASTRA_EVR_PARENT_PREDICT(*args, **kwargs)

# ASTRA READ-LEAF-FUSE v1, append to s51k2tw_x (optionally after read-lambda).
# Exact real arithmetic: sum the old trailing read and young read in the
# existing depth-6 leaf basis, then perform one common output unwind.
# No concatenated operands, rank reduction, changed young chunk plan, or
# underpriced operation. Float32 arithmetic changes; native certification
# is required. Do not also append the exploratory read-merge fragment.
_ASTRA_RLF_PARENT_READ = _RC_PARENT_READ
_ASTRA_RLF_PARENT_UNWIND = _ks_unwind
_ASTRA_RLF_CONTEXT = [None]
_ASTRA_RLF_ON = True
_ASTRA_RLF_STATS = dict(fused=0, parent=0, leaf_add_elements=0, shapes=[])


def _ks_unwind(TC, Z1, ops, basis, out, L, m, n):
    context = _ASTRA_RLF_CONTEXT[0]
    if context is not None and not context['done']:
        if L != 6 or (m, n) != (context['M'], context['N']):
            raise RuntimeError('ASTRA read fusion: unexpected output tree')
        context['done'] = True
        inner, right = context['inner'], context['right']
        rank = int(inner.shape[1])
        hzr, hzc = context['hzr'], context['hzc']
        h = _BH_HOLD.append
        # A/B side regions are disjoint from TC and its leaf accumulator.
        # Rank is a multiple of 64, so the extra product has no inner strip.
        if hzr:
            TA = _hzd_forward_a(inner, m, rank, True, False)
        else:
            aa = inner
            if int(inner.shape[0]) != m:
                aa = fnp.zeros((m, rank), dtype=fnp.float32)
                h((aa, fnp.copyto(aa[:int(inner.shape[0])], inner)))
            TA = _ipt_forward_a_ab(aa, 6, m, rank)
        if hzc:
            TB = _hzd_forward_b(right, rank, n, False, True)
        else:
            bb = right
            if int(right.shape[1]) != n:
                bb = fnp.zeros((rank, n), dtype=fnp.float32)
                h((bb, fnp.copyto(bb[:, :int(right.shape[1])], right)))
            TB = _ipt_forward_b_ab(bb, 6, rank, n)
        acc = _ks_acc(m, n)
        if hzr or hzc:
            # Entry 1 writes into the existing separate accumulator, with
            # the same strip groups/zero fills used by later young chunks.
            _hz_leaf(_hz_views(m, rank, n, hzr, hzc)[1])
        else:
            h(fnp.matmul(TA, TB, out=acc))
        h(fnp.add(TC, acc, out=TC))
        _ASTRA_RLF_STATS['leaf_add_elements'] += 7 ** 6 * (m // 64) * (n // 64)
    return _ASTRA_RLF_PARENT_UNWIND(TC, Z1, ops, basis, out, L, m, n)


def _astra_read_leaf_fuse(left, young_right, lam, live_indices, state):
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
        return _ASTRA_RLF_PARENT_READ(left, young_right, lam, live_indices, state)
    hold = _E2Hold()
    inner = hold(_cp_slr_inner_sq(hold(left[:, :old_columns]), hold(lam[:old_columns]), state))
    right = hold(hold(_xsg_rows(5119, state['pre_basis'], live_indices)).T)
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


_RC_PARENT_READ = _astra_read_leaf_fuse

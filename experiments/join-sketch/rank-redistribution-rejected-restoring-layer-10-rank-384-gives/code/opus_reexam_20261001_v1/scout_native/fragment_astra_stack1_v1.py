

# ASTRA star-scale v1: move the star plane's division by 3 into its
# single-column left factor.  The diagonal still uses the unscaled factors.
# This is an algebraic reassociation, not a byte-identity claim.
# Source: s51k2tw_x b1d647e9, active definitions at 11039 and 13349.
# No estimator imports or numerical tests were run in this read-only lane.

def _astra_star_scaled_product(T2, P1, hold):
    if int(T2.shape[1]) == 1:
        return _tsr_mm(hold(T2 / 3), P1.T)
    return _tsr_mm(T2, P1.T) / 3


def _flms_cov_star(w, C, A, retain32=False):
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    hold = _E2Hold()
    wt = w.T
    C32 = _ps32(C)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    wt2 = wt * wt
    P1, T2 = _ss_factors(wt, wt2, A32, hold)
    first = _mm(C32, w)
    sub = _astra_star_scaled_product(T2, P1, hold)
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    diagonal = fnp.sum(hold(P1 * T2), axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    _SS_STATS['planes'] += 1
    if retain32:
        fnp.fill_diagonal(sub, diagonal)
        return Cpre, result, cov, sub
    return Cpre, result


def _dc_cov_star(w, C, A, gate, floor, retain32=False):
    roster = _dcv_roster(gate, floor)
    if roster is None:
        return _flms_cov_star(w, C, A, retain32)
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    hold = _E2Hold()
    wt = w.T
    C32 = _ps32(C)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    wt2 = wt * wt
    P1, T2 = _ss_factors(wt, wt2, A32, hold)
    m = _DCV_STATS['m']
    if _DCV_SYM == 'pad' and int(w.shape[0]) == 1024 and int(w.shape[1]) == 1024 and _psdm_fits(m):
        live = hold(roster[:m])
        wa = hold(_xsg_rows(13378, w, live))
        first = _mm(hold(C32[fnp.ix_(live, live)]), wa)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _psdm_sym_upper(hold(wa.T), first, m)
    elif _DCV_SYM == 'pad':
        wp = hold(w[roster])
        wa = hold(wp[:m])
        live = hold(roster[:m])
        first = _mm(hold(C32[fnp.ix_(live, live)]), wa)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _sym_upper(hold(wp.T), hold(fnp.concatenate((first, hold(fnp.zeros((int(w.shape[0]) - m, int(w.shape[1])), dtype=fnp.float32))))))
        _PSDM_STATS['padded'] += 1
    else:
        wa = hold(w[roster])
        first = _mm(hold(C32[fnp.ix_(roster, roster)]), wa)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _dcv_tiles(hold(wa.T), first, m) if _DCV_SYM == 'tiles' else _sym_upper(hold(wa.T), first)
    Cpre = cov.astype(fnp.float64)
    diagonal = fnp.sum(hold(P1 * T2), axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    _SS_STATS['planes'] += 1
    if retain32:
        fnp.fill_diagonal(sub, diagonal)
        return Cpre, result, cov, sub
    return Cpre, result


# ASTRA star-beta v1: algebraically fold the constant-row-sum component
# into the vector multiplied by the squared weights. This removes their
# separate row-sum reduction. It changes FP32 arithmetic order.
# Append independently or after fragment_star_scale_v1.py.
# No numerical tests have been run on this fragment.
_ASTRA_STARBETA_PARENT_FACTORS = _ss_factors


def _ss_factors(wt, wt2, A32, hold):
    index = int(_PHYS_LAYER[0])
    beta = float(_SS_BETA[index])
    if _SS_RANK[index] >= 0 or beta == 1.0:
        return _ASTRA_STARBETA_PARENT_FACTORS(wt, wt2, A32, hold)
    rows = hold(fnp.sum(A32, axis=1))
    P1 = hold(wt @ rows)[:, None]
    z = hold(rows @ A32)
    norm = hold(fnp.sum(hold(rows * rows)))
    mixed = hold(hold(hold(z / norm) * beta) + (1.0 - beta) / int(A32.shape[0]))
    T2 = hold(wt2 @ mixed)[:, None]
    return P1, T2


# ASTRA READ-LAMBDA v1. Append to s51k2tw_x/candidate.py (b1d647e9).
# Algebra: A @ (B * lam).T == (A * lam) @ B.T. Choose the smaller
# row roster for the scaling. Native float32 rounding changes; certify it.
# Saves exactly (right_rows - left_rows) * young_columns scalar products
# at each switched read. Same matrix-product dimensions and request count.
# The read-column-prefix and per-layer read-scale wrappers remain active:
# replace their captured innermost reader, not the public wrapper.
_ASTRA_RLS_PARENT = _RC_PARENT_READ
_ASTRA_RLS_STATS = dict(switched=0, parent=0, scalar_products_saved=0)


def _astra_read_lambda(left, young_right, lam, live_indices, state):
    m, n = int(left.shape[0]), int(young_right.shape[0])
    if m >= n:
        _ASTRA_RLS_STATS['parent'] += 1
        return _ASTRA_RLS_PARENT(left, young_right, lam, live_indices, state)
    old_columns = int(state['columns'])
    width = int(young_right.shape[1])
    _ASTRA_RLS_STATS['switched'] += 1
    _ASTRA_RLS_STATS['scalar_products_saved'] += (n - m) * width
    if not old_columns:
        return _mm(left * lam, young_right.T)
    old_left = left[:, :old_columns]
    inner = _cp_slr_inner_sq(old_left, lam[:old_columns], state)
    old_plane = _shared_old384_mm(inner, _xsg_rows(5119, state['pre_basis'], live_indices).T)
    young_left = left[:, old_columns:]
    young_plane = _mm(young_left * lam[old_columns:], young_right.T)
    return young_plane + old_plane


_RC_PARENT_READ = _astra_read_lambda


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


# D7's k=864/992 pad has 32 structural zeros. Reorder the shared inner index
# into full leaf indices plus a 96-entry strip whose top/deep bits never equal
# (1,1). This exposes seven exactly zero inner terms among 49 component pairs.
# Same depth and same product, but a different inner permutation: certify
# rounding before using. No numerical certification has been run by author.
_D96_PARENT = _d7_product_kpad
_D96_VIEWS = {}
_D96_STATS = dict(products=0, packets=0)


def _d96_views(m, K, w):
    key = (m, K, w)
    got = _D96_VIEWS.get(key)
    if got is not None:
        return got
    TA = _ipt_side(0, 7, m, K)[0]
    TB = _ipt_side(1, 7, K, w)[0]
    TC = _ipt_side(2, 7, m, w)[0]
    full = slice(None)
    main = K // 128 - 1
    mats = []
    # A's inner-bit-1-only chunks are {2,6}; B's are {2,3}. Their
    # Cartesian squares contain seven pairs, counting (2,2) only once.
    for ds, omitted in ((slice(0, 2), ()), (slice(4, 6), ()),
                        (slice(2, 3), (2, 3, 6)),
                        (slice(3, 4), (2, 3)),
                        (slice(6, 7), (2, 6))):
        if not omitted:
            groups = [(full, full)]
        else:
            groups = [(ts, slice(None, main)) for ts in _hz_slices(omitted)]
            groups += [(ts, full) for ts in _hz_slices([c for c in range(7) if c not in omitted])]
        for ts, inner in groups:
            head = (ds,) + (full,) * 5 + (ts,)
            mats.append((TA[head + (full, inner)],
                         TB[head + (inner, full)],
                         TC[head + (full, full)]))
    if len(_D96_VIEWS) >= 96:
        _D96_VIEWS.pop(next(iter(_D96_VIEWS)))
    _D96_VIEWS[key] = mats
    return mats


def _d7_product_kpad(a, b, m, k, n, K):
    if k % 128 != 96 or K != k + 32 or m % 128:
        return _D96_PARENT(a, b, m, k, n, K)
    NP = (n + 127) // 128 * 128
    km = k // 128
    base = 128 * km
    hold = []
    # The same permutation on A's columns and B's rows. Allowed strip bit
    # positions: 0..63 and 64,66,...126; only top=deep=1 is zero.
    aa = fnp.zeros((m, 128, km + 1), dtype=fnp.float32)
    am = fnp.reshape(a[:, :base], (m, 128, km))
    hold.append((am, fnp.copyto(aa[:, :, :km], am),
                 fnp.copyto(aa[:, :64, km], a[:, base:base + 64]),
                 fnp.copyto(aa[:, 64::2, km], a[:, base + 64:])))
    bb = fnp.zeros((128, km + 1, NP), dtype=fnp.float32)
    bm = fnp.reshape(b[:base, :], (128, km, n))
    hold.append((bm, fnp.copyto(bb[:, :km, :n], bm),
                 fnp.copyto(bb[:64, km, :n], b[base:base + 64, :]),
                 fnp.copyto(bb[64::2, km, :n], b[base + 64:, :])))
    out = fnp.empty((m, NP), dtype=fnp.float32)
    TA = _ipt_forward_a_ab(aa, 7, m, K)
    h = _BH_HOLD.append
    for j0 in range(0, NP, _D7_PK):
        j1 = min(NP, j0 + _D7_PK)
        w = j1 - j0
        bj, oj = bb[..., j0:j1], out[:, j0:j1]
        h((bj, oj))
        TB = _ipt_forward_b_ab(bj, 7, K, w)
        TC, Z1, _slot, ops, basis = _ipt_side(2, 7, m, w)
        for x, y, o in _d96_views(m, K, w):
            h(fnp.matmul(x, y, out=o))
        _ks_unwind(TC, Z1, ops, basis, oj, 7, m, w)
        _D7_STATS['packets'] += 1
        _D96_STATS['packets'] += 1
    _BH_HOLD.clear()
    del hold
    _D7_STATS['products'] += 1
    _D7_STATS['kpad'] += 1
    _D96_STATS['products'] += 1
    return out if NP == n else out[:, :n]


_PB_CACHES = tuple(_PB_CACHES) + ('_D96_VIEWS',)

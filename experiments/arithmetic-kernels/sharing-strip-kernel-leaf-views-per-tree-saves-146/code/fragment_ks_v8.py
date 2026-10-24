# BEGIN OPUS-ENGINEER HALF-ZERO STRIP K-SPLIT (hzs-ks v8, 2026-10-02 = v7 with the leaf views cached per TREE instead of per product: a product
# (M, q, N) takes its views of the left tree (M, q), of the right tree (q, N), of the output tree and of the leaf-sum buffer (M, N) from
# four entries that every other product with the same tree shares; the SAME views as v7 (hzs/unit_views_v8.py), built once per tree.
# v7 = v6 with the leaf views built by one index expression each and the
# operand views shared between the output tree and the leaf-sum buffer: the SAME views (hzs/unit_views_v7.py: same memory regions in
# all eight strip combinations), 346 -> 156 index requests over those combinations; nothing else changes. v6, 2026-10-01;
# rounding-level, NOT bit-exact): a k-split read (m x k x n, k >= 2048)
# whose m or n is an odd multiple of 32 runs today either at depth 5 or zero-padded to the next multiple of 64 at depth 6, where the
# 32 pad rows cost as much as 64 real rows. Here the 32 real rows beyond the last multiple of 64 and 32 zero rows are INTERLEAVED on
# the deepest row bit: operand row (block b = 0..63, leaf row r) holds a main row for r < a, and for r = a the strip row b/2 when b
# is even, zero when b is odd. Every depth-5 node of the A tree then has a zero bottom half in its last leaf row, so at the deepest
# level that leaf row needs only 4 of the 7 chunk products (chunks 0, 2, 3, 5; the A tree is exactly zero there in chunks 4 and 6 and
# chunk 1 feeds only the discarded zero-row outputs). The same holds for a half-zero last leaf COLUMN of B (chunks 0, 2, 5, 6; the B
# tree is exactly zero in chunks 1 and 3 and chunk 4 cancels in the decode), and for a half-zero last INNER index the strip term
# vanishes exactly in chunks 2, 3, 6 (used by the push fragment). v6 leaf stage: the 7 chunks are grouped by which strips they need
# and each group is ONE batched matmul on the leaf rows / inner indices / columns it needs (4-5 matmuls per k chunk, no per-row
# matrix-vector calls, no buffers); the strip slots a group does not compute are zero-filled. A strip so costs the depth-5 rate
# instead of twice the depth-6 rate, and the leaf work is strictly below the zero-padded product's. Trees, keys, forward transforms,
# accumulation and unwind are the parent's (same keys as its padded k-split). Certified against float64 in hzs/certify_ks.py.
# Also admits k = odd multiple of 32 (32 zero inner columns).
_HZ_ON = True
_HZ_PRICE = 750000.0                        # flops per request, as _KS_PRICE
_HZ_D5 = True                               # also move reads the parent keeps at depth 5 (more bill saved, more kernel time); False = only
                                            # the reads the parent already zero-pads at depth 6
_HZ_STATS = {'ks': 0, 'row': 0, 'col': 0}
_HZ_PLANS = {}
_HZ_VIEWS = {}
_HZ_VIEW_CAP = 96
_HZ_SIDES = {}                              # (kind, tree shape, strip flags) -> (views per leaf group, zero-fill views)
_HZ_SIDE_CAP = 384
_HZ_ROW_FULL = (0, 2, 3, 5)                 # chunks whose product the strip leaf ROW needs
_HZ_COL_FULL = (0, 2, 5, 6)                 # chunks whose product the strip leaf COLUMN needs
_HZ_K_FULL = (0, 1, 4, 5)                   # chunks in which the inner-strip term is not exactly zero
_HZ_OLD_KS_PLAN = _ks_plan
_HZ_OLD_KS_PRODUCT = _ks_product


def _hz_slices(chunks):
    # sorted chunk indices -> basic slices (arithmetic progressions) covering exactly them
    out, rest = [], list(chunks)
    while rest:
        run = [rest[0]]
        if len(rest) > 1:
            step = rest[1] - rest[0]
            for c in rest[1:]:
                if c == run[-1] + step:
                    run.append(c)
                else:
                    break
        else:
            step = 1
        out.append(slice(run[0], run[-1] + 1, step))
        rest = rest[len(run):]
    return out


def _hz_groups(hzr, hzk, hzc):
    # -> (matmul groups [(slice, row full, inner full, col full)], row-strip fill slices, col-strip fill slices)
    groups = {}
    for c in range(7):
        key = ((not hzr) or c in _HZ_ROW_FULL, (not hzk) or c in _HZ_K_FULL, (not hzc) or c in _HZ_COL_FULL)
        groups.setdefault(key, []).append(c)
    mats = [(s,) + key for key, chunks in groups.items() for s in _hz_slices(chunks)]
    rowfill = _hz_slices([c for c in range(7) if c not in _HZ_ROW_FULL]) if hzr else []
    colfill = _hz_slices([c for c in range(7) if c not in _HZ_COL_FULL]) if hzc else []
    return mats, rowfill, colfill


def _hz_side(kind, T, lo0, lo1, hzr, hzk, hzc):
    # the leaf-stage views of ONE tree: kind 0 = left operand (rows x inner), 1 = right operand (inner x columns), 2 = output tree,
    # 3 = leaf-sum buffer (rows x columns). lo0, lo1 = the main part of its two leaf dimensions (one less than the tree's where that
    # dimension carries a strip). A tree of a given kind and shape is always the same region of the arena, so its views serve every
    # product that uses it. Every view is ONE index expression, exactly v7's.
    key = (kind, tuple(int(x) for x in T.shape), hzr, hzk, hzc)
    got = _HZ_SIDES.get(key)
    if got is not None:
        return got
    mats, rowfill, colfill = _hz_groups(hzr, hzk, hzc)
    full = slice(None)
    views, fills = [], []
    for s, rf, kf, cf in mats:
        if kind == 0:
            first, second = (slice(None, lo0) if (hzr and not rf) else full), (slice(None, lo1) if (hzk and not kf) else full)
        elif kind == 1:
            first, second = (slice(None, lo0) if (hzk and not kf) else full), (slice(None, lo1) if (hzc and not cf) else full)
        else:
            first, second = (slice(None, lo0) if (hzr and not rf) else full), (slice(None, lo1) if (hzc and not cf) else full)
        views.append(T[s] if (first is full and second is full) else T[s, ..., first, second])
    if kind >= 2:
        for s in rowfill:
            fills.append(T[s, ..., lo0:, :])
        for s in colfill:
            fills.append(T[s, ..., :, lo1:])
    if len(_HZ_SIDES) >= _HZ_SIDE_CAP:
        _HZ_SIDES.pop(next(iter(_HZ_SIDES)))
    got = _HZ_SIDES[key] = (views, fills)
    return got


def _hz_leaf_views(TA, TB, T, ra, kq, cb, hzr, hzk, hzc, kind_out=2):
    # ra, kq, cb = main leaf rows / inner indices / columns (one less than the tree's when that dimension carries a strip).
    # v8: assembled from the per-tree entries; kind_out = 3 when T is the leaf-sum buffer of the output tree's shape.
    A = _hz_side(0, TA, ra, kq, hzr, hzk, hzc)[0]
    B = _hz_side(1, TB, kq, cb, hzr, hzk, hzc)[0]
    O, fills = _hz_side(kind_out, T, ra, cb, hzr, hzk, hzc)
    return dict(mats=list(zip(A, B, O)), fills=fills)


def _hz_leaf(v):
    h = _BH_HOLD.append
    for x, y, o in v['mats']:
        h(fnp.matmul(x, y, out=o))
    for f in v['fills']:
        h(fnp.copyto(f, 0.0))


def _hz_plan(m, k, n, padded=False):
    # padded = the parent's own rule already runs this read zero-padded at depth 6: the strips then only save (no depth change)
    key = (m, k, n, padded)
    if key in _HZ_PLANS:
        return _HZ_PLANS[key]
    plan = None
    hzr, hzc = m % 64 == 32, n % 64 == 32
    if _HZ_ON and k >= 2048 and m % 32 == 0 and n % 32 == 0 and k % 32 == 0 and m >= 96 and n >= 96 and (hzr or hzc or k % 64):
        M, N, K = (m + 63) // 64 * 64, (n + 63) // 64 * 64, (k + 63) // 64 * 64
        region = _IPT_REGION[6]
        cmax = min(_KS_CAP, 64 * min(region[0] // (7 ** 6 * (M // 64)), region[1] // (7 ** 6 * (N // 64))))
        if cmax >= 64:
            nb = K // 64
            nch = -(-nb // (cmax // 64))
            base, r = divmod(nb, nch)
            sizes = tuple([(base + 1) * 64] * r + [base * 64] * (nch - r))
            if all(_ipt_fits(6, M, q, N) for q in sizes):
                b5 = 2 * (7 / 8) ** 5 * m * k * n + _KS_C5 * (m * k + k * n + m * n)
                b6 = (sum(2 * (7 / 8) ** 6 * M * q * N + _KS_C6 * (M * q + q * N + M * N) for q in sizes)
                      + (nch - 1) * m * n + m * n - (nch - 1) * (58.88 * M * N + m * n - 7 ** 6 * (M // 64) * (N // 64)))
                b6 += (2.0 * m * k if (hzr or K != k) else 0) + (2.0 * k * n if (hzc or K != k) else 0) + 2.0 * m * n
                b6 -= (3 * 7 ** 5 * 2 * (K // 64) * (N // 64) if hzr else 0) + (3 * 7 ** 5 * 2 * (M // 64 - (1 if hzr else 0)) * (K // 64) if hzc else 0)
                r6 = 47 * nch + (nch - 1) + 36 + 12 + nch * (8 if (hzr and hzc) else 5)
                if padded or b6 + _HZ_PRICE * r6 < b5 + _HZ_PRICE * _KS_REQ5:
                    plan = (M, N, sizes, hzr, hzc, K)
    _HZ_PLANS[key] = plan
    return plan


def _ks_plan(m, k, n):
    old = _HZ_OLD_KS_PLAN(m, k, n)
    if not _HZ_ON or (m % 64 == 0 and n % 64 == 0 and k % 64 == 0) or (old is None and not _HZ_D5):
        return old
    plan = _hz_plan(m, k, n, old is not None)
    return plan if plan is not None else old


def _hz_views(M, q, N, hzr, hzc):
    # leaf-stage views of the (M, q), (q, N), (M, N) trees and of the leaf-sum buffer, bound once per shape
    key = (M, q, N, hzr, hzc)
    got = _HZ_VIEWS.get(key)
    if got is not None:
        return got
    TA = _ipt_side(0, 6, M, q)[0]
    TB = _ipt_side(1, 6, q, N)[0]
    TC = _ipt_side(2, 6, M, N)[0]
    ra = M // 64 - (1 if hzr else 0)
    cb = N // 64 - (1 if hzc else 0)
    entry = [_hz_leaf_views(TA, TB, TC, ra, q // 64, cb, hzr, False, hzc), _hz_leaf_views(TA, TB, _ks_acc(M, N), ra, q // 64, cb, hzr, False, hzc, kind_out=3)]
    if len(_HZ_VIEWS) >= _HZ_VIEW_CAP:
        _HZ_VIEWS.pop(next(iter(_HZ_VIEWS)))
    _HZ_VIEWS[key] = entry
    return entry


def _hz_ks_product(a, b, out, m, k, n, plan):
    M, N, sizes, hzr, hzc, K = plan
    hold = []
    al, bl = M // 64 - 1, N // 64 - 1
    me, ne = (64 * al if hzr else m), (64 * bl if hzc else n)
    if hzr:
        aa = fnp.zeros((64, al + 1, K), dtype=fnp.float32)
        am = fnp.reshape(a[:me], (64, al, k))
        hold.append((am, fnp.copyto(aa[:, :al, :k], am), fnp.copyto(aa[0::2, al, :k], a[me:])))
        _HZ_STATS['row'] += 1
    elif K != k:
        aa = fnp.zeros((M, K), dtype=fnp.float32)
        hold.append(fnp.copyto(aa[:, :k], a))
    else:
        aa = a
    if hzc:   # packed in the SOURCE's memory order (the reads pass a transposed C array), so the pack is a contiguous copy and the
        # tree load reads a transposed view exactly as the parent's load does
        bbt = fnp.zeros((64, bl + 1, K), dtype=fnp.float32)
        bt = b.T
        bm = fnp.reshape(bt[:ne], (64, bl, k))
        hold.append((bt, bm, fnp.copyto(bbt[:, :bl, :k], bm), fnp.copyto(bbt[0::2, bl, :k], bt[ne:])))
        bb = fnp.transpose(bbt, (2, 0, 1))
        hold.append(bbt)
        _HZ_STATS['col'] += 1
    elif K != k:
        bb = fnp.zeros((K, N), dtype=fnp.float32)
        hold.append(fnp.copyto(bb[:k], b))
    else:
        bb = b
    TC, Z1, _pair, ops, basis = _ipt_side(2, 6, M, N)
    h = _BH_HOLD.append
    k0 = 0
    for j, q in enumerate(sizes):
        ak, bk = aa[..., k0:k0 + q], bb[k0:k0 + q]
        h((ak, bk))
        TA = _ipt_forward_a_ab(ak, 6, M, q)
        TB = _ipt_forward_b_ab(bk, 6, q, N)
        if hzr or hzc:
            _hz_leaf(_hz_views(M, q, N, hzr, hzc)[1 if j else 0])
        else:
            h(fnp.matmul(TA, TB, out=(_ks_acc(M, N) if j else TC)))
        if j:
            h(fnp.add(TC, _ks_acc(M, N), out=TC))
        k0 += q
    if M == m and N == n:
        _ks_unwind(TC, Z1, ops, basis, out, 6, M, N)
    else:
        o = fnp.empty((M, N), dtype=fnp.float32)
        _ks_unwind(TC, Z1, ops, basis, o, 6, M, N)
        if hzr and hzc:
            o4 = fnp.reshape(o, (64, al + 1, 64, bl + 1))
            p0 = fnp.reshape(o4[:, :al, :, :bl], (me, ne)); p1 = fnp.reshape(o4[:, :al, 0::2, bl], (me, 32))
            p2 = fnp.reshape(o4[0::2, al, :, :bl], (32, ne))
            hold.append((o4, p0, p1, p2, fnp.copyto(out[:me, :ne], p0), fnp.copyto(out[:me, ne:], p1),
                         fnp.copyto(out[me:, :ne], p2), fnp.copyto(out[me:, ne:], o4[0::2, al, 0::2, bl])))
        elif hzr:
            o3 = fnp.reshape(o, (64, al + 1, N))
            p0 = fnp.reshape(o3[:, :al, :n], (me, n))
            hold.append((o3, p0, fnp.copyto(out[:me], p0), fnp.copyto(out[me:], o3[0::2, al, :n])))
        else:
            o3 = fnp.reshape(o, (M, 64, bl + 1))
            p0 = fnp.reshape(o3[:m, :, :bl], (m, ne))
            hold.append((o3, p0, fnp.copyto(out[:, :ne], p0), fnp.copyto(out[:, ne:], o3[:m, 0::2, bl])))
    _BH_HOLD.clear()
    del hold
    _HZ_STATS['ks'] += 1
    return out


def _ks_product(a, b, out, m, k, n, plan):
    if len(plan) == 3:
        return _HZ_OLD_KS_PRODUCT(a, b, out, m, k, n, plan)
    return _hz_ks_product(a, b, out, m, k, n, plan)


_PB_CACHES = _PB_CACHES + ('_HZ_VIEWS', '_HZ_SIDES')
# END OPUS-ENGINEER HALF-ZERO STRIP K-SPLIT

# BEGIN FABLE-ENGINEER DEPTH-5 READ K-SPLIT v2 (E-74b, 2026-09-25; rounding-level, not bit-exact). v2 = v1 with the chunks summed in
# LEAF space: every chunk's depth-6 leaf product is added into the output tree (the second chunk on uses a leaf-sized buffer placed
# right after the output tree in the depth-6 C region), and ONE unwind forms the output. The unwind is linear, so the result is the
# same sum in another rounding order. Per extra chunk this drops an unwind (36 requests with fused keys), an output reshape and an
# output add, and adds one leaf-sized add. Same selection rule as v1 (conservative for v2). v1 text follows.
# (v1) The bank reads (_packed_read5) with
# inner dim >= 4096 run at depth 5 (engineer2's --d6x keeps reads there: their dims rarely divide by 64 and their depth-6 A trees do not
# fit the region). A read runs as a k-split at depth 6 when S24's own cost rule says it pays (bill + 750k flops per request, as
# _rule_depth): m and n zero-padded to multiples of 64 (fnp.zeros + fnp.copyto, as _push6_padded), k cut into balanced multiples of
# 64 that fit the depth-6 regions (at most 4096), one alt-basis depth-6 tree per chunk into its own (M, N) output, then the chunk
# outputs' leading m x n windows added into out in chunk order. Bill models fitted on S24's kernel (E-72d, 24 read shapes, max error
# 3 Mflop): depth 5 = 2 (7/8)^5 mkn + 18.36 (mk + kn + mn); split = sum over chunks of 2 (7/8)^6 M q N + 30.60 (Mq + qN + MN)
# + (nch - 1) mn + 1.35 x padded-operand elements. Requests: a fused tree product is 72 at depth 5, 83 at depth 6.
# E-72d (OpenBLAS, S24 + E-73, net 100): the rule splits 13 shapes (14 products per net), bill -0.86%, +1.5k requests with fused
# chunk keys, backend x1.6-1.8 on those products. Setup: the prebind's phases 1-2 are S24's minus the 19 depth-5 keys that only
# split reads used (ksplit_plan_census.json); phase 3 binds the chunk keys FUSED with the guard's leftover time, largest saving first.
# A chunk key that misses phase 3 binds lazily in predict, unfused. Compose BEFORE the transform-reuse fragment (it wraps this read).
_KS_ON = True
_KS_PRICE = 750000.0                      # flops per request (_rule_depth's price)
_KS_C5, _KS_C6, _KS_PAD = 18.36, 30.60, 1.35
_KS_REQ5, _KS_REQ6 = 72, 83
_KS_CAP = 4096
_KS_PLANS = {}
_KS_STATS = {'split': 0}
_KS_SHAPES = ((992, 6144, 1024), (832, 8192, 928), (768, 7296, 928), (704, 8192, 864), (640, 7296, 864), (928, 8128, 992),
              (640, 8192, 384), (768, 6144, 384), (576, 8192, 384), (704, 6144, 384), (832, 4608, 384), (576, 5120, 800),
              (928, 6144, 384))  # the reads the rule splits in S23's net-35/100 census, largest modelled saving first
_KS_DROP = frozenset((
    (0, 5, 576, 8192), (0, 5, 640, 8192), (0, 5, 704, 6144), (0, 5, 704, 8192), (0, 5, 768, 6144), (0, 5, 832, 8192), (0, 5, 928, 6144),
    (0, 5, 928, 8128), (0, 5, 992, 6144), (1, 5, 6144, 1024), (1, 5, 8128, 992), (1, 5, 8192, 928), (2, 5, 576, 800), (2, 5, 704, 384),
    (2, 5, 768, 384), (2, 5, 832, 384), (2, 5, 928, 384), (2, 5, 928, 992), (2, 5, 992, 1024)))
_PB_SINGLE = tuple(x for x in _PB_SINGLE if x not in _KS_DROP)
_KS_ACC = {}                              # (M, N) -> the second leaf buffer's view; dropped with the tree caches (_pb_reset)
_PB_CACHES = _PB_CACHES + ('_KS_ACC',)


def _ks_acc(M, N):
    view = _KS_ACC.get((M, N))
    if view is None:
        region = _IPT_REGION[6]
        start = region[0] + region[1] + 7 ** 6 * (M // 64) * (N // 64)
        view = _ipt_view(_SCRATCH[0], start, (7,) * 6 + (M // 64, N // 64))
        _KS_ACC[(M, N)] = view
    return view


def _ks_unwind(TC, Z1, ops, basis, out, L, m, n):
    # = S24's _ipt_leaf_unwind_ab after its leaf matmul: same ops in the same order
    h = _BH_HOLD.append
    if ops is not None:
        for f, x, y, o in ops:
            h(f(x, y, out=o))
    else:
        for j in range(L, 1, -1):
            Z = Z1[j]
            h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=Z[0])); h(fnp.add(Z[5], Z[4], out=Z[5]))
            h(fnp.add(Z[5], Z[3], out=Z[2])); h(fnp.add(Z[5], Z[1], out=Z[3])); h(fnp.add(Z[4], Z[6], out=Z[1]))
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    tl, tr, bl, br = _rs_out_quadrants(out, L, m, n)
    Z = Z1[1]
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=tl)); h(fnp.add(Z[5], Z[4], out=Z[5]))
    h(fnp.add(Z[5], Z[3], out=tr)); h(fnp.add(Z[5], Z[1], out=br)); h(fnp.add(Z[4], Z[6], out=Z[4])); h(fnp.subtract(br, Z[4], out=bl))
    return out


def _ks_plan(m, k, n):
    key = (m, k, n)
    if key in _KS_PLANS:
        return _KS_PLANS[key]
    plan = None
    if k >= 4096 and m % 32 == 0 and n % 32 == 0 and k % 64 == 0:
        M, N = (m + 63) // 64 * 64, (n + 63) // 64 * 64
        region = _IPT_REGION[6]
        cmax = min(_KS_CAP, 64 * min(region[0] // (7 ** 6 * (M // 64)), region[1] // (7 ** 6 * (N // 64))))
        if cmax >= 64:
            nb = k // 64
            nch = -(-nb // (cmax // 64))
            base, r = divmod(nb, nch)
            sizes = tuple([(base + 1) * 64] * r + [base * 64] * (nch - r))
            if nch >= 2 and all(_ipt_fits(6, M, q, N) for q in sizes):
                padn = (M * k if M != m else 0) + (k * N if N != n else 0)
                b5 = 2 * (7 / 8) ** 5 * m * k * n + _KS_C5 * (m * k + k * n + m * n)
                b6 = (sum(2 * (7 / 8) ** 6 * M * q * N + _KS_C6 * (M * q + q * N + M * N) for q in sizes)
                      + (nch - 1) * m * n + _KS_PAD * padn)
                r6 = nch * _KS_REQ6 + (nch - 1) + nch + 2 * ((M != m) + (N != n))
                if b6 + _KS_PRICE * r6 < b5 + _KS_PRICE * _KS_REQ5:
                    plan = (M, N, sizes)
    _KS_PLANS[key] = plan
    return plan


def _ks_product(a, b, out, m, k, n, plan):
    M, N, sizes = plan
    hold = []
    if M != m:
        aa = fnp.zeros((M, k), dtype=fnp.float32)
        hold.append(fnp.copyto(aa[:m, :], a))
    else:
        aa = a
    if N != n:
        bb = fnp.zeros((k, N), dtype=fnp.float32)
        hold.append(fnp.copyto(bb[:, :n], b))
    else:
        bb = b
    TC, Z1, _pair, ops, basis = _ipt_side(2, 6, M, N)
    acc = _ks_acc(M, N)
    h = _BH_HOLD.append
    k0 = 0
    for j, q in enumerate(sizes):
        TA = _ipt_forward_a_ab(aa[:, k0:k0 + q], 6, M, q)
        TB = _ipt_forward_b_ab(bb[k0:k0 + q, :], 6, q, N)
        if j == 0:
            h(fnp.matmul(TA, TB, out=TC))
        else:
            h(fnp.matmul(TA, TB, out=acc))
            h(fnp.add(TC, acc, out=TC))
        k0 += q
    if M == m and N == n:
        _ks_unwind(TC, Z1, ops, basis, out, 6, M, N)
    else:
        o = fnp.empty((M, N), dtype=fnp.float32)
        _ks_unwind(TC, Z1, ops, basis, o, 6, M, N)
        hold.append(fnp.copyto(out, o[:m, :n]))
    _BH_HOLD.clear()
    del hold
    return out


_KS_OLD_READ5 = _packed_read5


def _packed_read5(a, b, out, views):
    if _KS_ON and _AB_ON and len(a.shape) == 2 and len(b.shape) == 2:
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        plan = _ks_plan(m, k, n)
        if plan is not None and _ipt_ok(a, b, out, 5):
            _KS_STATS['split'] += 1
            return _ks_product(a, b, out, m, k, n, plan)
    return _KS_OLD_READ5(a, b, out, views)


_KS_OLD_PB_BIND = _pb_bind


def _pb_bind(t0):
    # phases 1-2: S24's prebind and level fusion (on the filtered key list); phase 3: the split reads' chunk keys, bound fused
    _KS_OLD_PB_BIND(t0)
    done = 0
    try:
        for m, k, n in _KS_SHAPES:
            plan = _ks_plan(m, k, n)
            if plan is None:
                continue
            M, N, sizes = plan
            qs = sorted(set(sizes))
            for kind, rows, cols in [(0, M, q) for q in qs] + [(1, q, N) for q in qs] + [(2, M, N)]:
                if _pb_time.monotonic() - t0 > _PB_WALL_S:
                    return
                if dict.get((_IPT_A, _IPT_B, _IPT_C)[kind], (6, rows, cols)) is None:
                    _LVF_EAGER[0] = True
                    try:
                        _ipt_side(kind, 6, rows, cols)
                    finally:
                        _LVF_EAGER[0] = False
                    done += 1
            if (M, N) not in _KS_ACC:
                _ks_acc(M, N)
                done += 1
    finally:
        _PB_LOG.append(('ksplit', done, round(_pb_time.monotonic() - t0, 3)))
        _BH_HOLD.clear()
# END FABLE-ENGINEER DEPTH-5 READ K-SPLIT v2

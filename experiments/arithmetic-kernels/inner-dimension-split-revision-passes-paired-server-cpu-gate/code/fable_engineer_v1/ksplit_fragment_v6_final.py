# BEGIN FABLE-ENGINEER K-SPLIT v6 (E-77, 2026-09-25; rounding-level; FINAL: partition agreed with engineer2 12:1x, read6 products included). v6 = v5w with (a) no 4096 chunk
# cap (the depth-6 regions alone set the chunk size, so a read that fits whole runs as ONE depth-6 tree: engineer2's coordination point)
# and (b) an _mm wrapper on top of the chain: a big-inner-dim product in the base _mm's read regime (k > 1024, m and n <= 1024) that
# needs >= 2 depth-6 chunks and passes the v2-priced rule goes WHOLE to the leaf-sum split, before the base _mm's k-tiling (8192) and
# engineer2's six-level read (per-chunk unwinds). Bench (e77_sqsplit_bench_s27.log, OpenBLAS): vs read6 -1.7..-2.8% bill, -23..-130
# requests, backend x0.92-1.21 per product. Single-chunk products fall through (v5w's read wrapper / engineer2's whole moves).
# (v5w) DEPTH-5 READ K-SPLIT v5w (E-74e, 2026-09-25; rounding-level). v5w = v4w with the rule also allowed ONE chunk
# (a padded depth-6 read, no k-split) and k >= 2048: the depth-6 cost model is E-72's fit on single depth-6 products for nch = 1 (the
# v2 correction applies from the second chunk on). Over nets 300..580 + 35/100/37/39 the model adds ~6 reads per net, bill ~-0.12%,
# ~+94 requests per net (e74_shape_census_sel8plus4.json). The leaf buffer is touched only when there are >= 2 chunks.
# (v4w) DEPTH-5 READ K-SPLIT v4w (E-74d wide, for S26 WITH group 5 only (lead); rounding-level). v4w = v4
# with the selection rule priced at v2's own costs (still S24's 750k flops per request): one unwind instead of nch (58.88 flops per
# M x N element, measured exactly on 13 shapes), a leaf-sized add per extra chunk, an output copy when padded; requests 47 per chunk
# + 36 for the unwind (+2 per padded operand, +2 for a padded output), exact on 23 shapes. The bill model is within 3 Mflop of 23
# measured v2 bills (e74_ksprod_v2_price0.log). On S23's census it adds 5 shapes to v4's 13; on nets 300..580 + 35/100/37/39 it
# splits 15-26 reads per net (v4: 11-19). Leaf buffers: the 24 (M, N) this rule produced there.
# (v4) DEPTH-5 READ K-SPLIT v4 (E-74d, for S26; rounding-level, not bit-exact). v4 = v3 with the prebind
# lists taken off S23's census: bank-read m, k and n vary per net and moved between S23 and S24 (porter's ks_shape_check: 9 of 13
# split shapes on net 35 were outside v3's list). The chunk keys (kind, 6, rows, cols) now come from the S26 prebind list the porter
# re-derives with the standard rule (keys used by >= 2 of the 8 selection nets 300..580), bound in phase 1 and fused in phase 2 like
# every other key; v3's shape list and drop filter are gone. Phase 3 binds only the leaf buffers, which are not tree keys, for the 22
# (M, N) the rule produced over nets 300..580 + 35/100/37/39 (e74_shape_census_sel8plus4.json), most selection nets first.
# (v3) DEPTH-5 READ K-SPLIT v3 (E-74c, for S26). v3 = v2 plus two S26-only
# setup settings (lead and porter, 2026-09-25): the prebind guard 3.0 -> 3.5 s and S24's dead batched prebinds dropped. v2 text follows.
# (v2) DEPTH-5 READ K-SPLIT v2 (E-74b; rounding-level, not bit-exact). v2 = v1 with the chunks summed in
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
_PB_WALL_S = 3.5   # S26 only (lead 2026-09-25): the prebind guard goes from S24's 3.0 s to 3.5 s. The platform's setup limit is 5 s; S24's
                   # platform setup max was 2.88 s, and phase 3 below adds 38 chunk keys + 13 leaf buffers (~+0.1 s locally), so at 3.0 s the
                   # slowest workers would leave chunk keys unfused and bind them lazily (billed) on their cold first MLP. 3.5 s still caps setup
                   # ~1.4 s under the platform limit. Gate: local setup wall under load <= 3.6 s.
_PB_BATCHED = ()   # S26 only (porter 2026-09-25): S24's 3 batched depth-6 prebinds (P = 2, 1024^3) serve no census net; dropped for setup time.
_KS_ON = True
_KS_PRICE = 750000.0                      # flops per request (_rule_depth's price)
_KS_C5, _KS_C6, _KS_PAD = 18.36, 30.60, 1.35
_KS_REQ5, _KS_REQ6 = 72, 83
_KS_CAP = 1 << 20                         # v6: no chunk cap; the depth-6 regions decide
_KS_PLANS = {}
_KS_STATS = {'split': 0, 'mm': 0}
_KS_ACC_MN = ((384, 384), (768, 384), (832, 384), (960, 384), (960, 1024), (704, 384), (704, 896), (896, 384), (896, 960), (576, 384),
              (576, 832), (640, 384), (832, 960), (512, 384), (768, 896), (1024, 1024), (512, 768), (640, 832), (640, 896), (512, 832),
              (704, 832), (832, 896), (576, 768), (768, 960))   # the wide rule's leaf-buffer (M, N) on the same nets, most selection nets first
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
    if k >= 2048 and m % 32 == 0 and n % 32 == 0 and k % 64 == 0:
        M, N = (m + 63) // 64 * 64, (n + 63) // 64 * 64
        region = _IPT_REGION[6]
        cmax = min(_KS_CAP, 64 * min(region[0] // (7 ** 6 * (M // 64)), region[1] // (7 ** 6 * (N // 64))))
        if cmax >= 64:
            nb = k // 64
            nch = -(-nb // (cmax // 64))
            base, r = divmod(nb, nch)
            sizes = tuple([(base + 1) * 64] * r + [base * 64] * (nch - r))
            if nch >= 1 and all(_ipt_fits(6, M, q, N) for q in sizes):
                padn = (M * k if M != m else 0) + (k * N if N != n else 0)
                b5 = 2 * (7 / 8) ** 5 * m * k * n + _KS_C5 * (m * k + k * n + m * n)
                b6 = (sum(2 * (7 / 8) ** 6 * M * q * N + _KS_C6 * (M * q + q * N + M * N) for q in sizes)
                      + (nch - 1) * m * n + _KS_PAD * padn)
                pout = M != m or N != n     # v2 costs (v4w): one unwind, leaf-sized adds, an output copy when padded
                b6 += (m * n if pout else 0) - (nch - 1) * (58.88 * M * N + m * n - 7 ** 6 * (M // 64) * (N // 64))
                r6 = 2 * ((M != m) + (N != n)) + 47 * nch + (nch - 1) + 36 + 2 * pout
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
    acc = _ks_acc(M, N) if len(sizes) > 1 else None
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
    # phases 1-2: S26's prebind list and level fusion (the chunk keys are in that list); phase 3: the leaf buffers
    _KS_OLD_PB_BIND(t0)
    done = 0
    try:
        for M, N in _KS_ACC_MN:
            if _pb_time.monotonic() - t0 > _PB_WALL_S:
                return
            if (M, N) not in _KS_ACC:
                _ks_acc(M, N)
                done += 1
    finally:
        _PB_LOG.append(('ksplit', done, round(_pb_time.monotonic() - t0, 3)))
        _BH_HOLD.clear()
_KS_OLD_MM = _mm


def _mm(a, b):
    # v6 (b): big-inner-dim products that need >= 2 depth-6 chunks go whole to the leaf-sum split, ahead of the base _mm's k-tiling
    if (_KS_ON and _AB_ON and _WINOGRAD and len(a.shape) == 2 and len(b.shape) == 2
            and str(a.dtype) == 'float32' and str(b.dtype) == 'float32'):
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        if k > 1024 and 0 < m <= 1024 and 0 < n <= 1024:
            plan = _ks_plan(m, k, n)
            if plan is not None and len(plan[2]) >= 2 and _SCRATCH is not None:
                _KS_STATS['mm'] += 1
                return _ks_product(a, b, fnp.empty((m, n), dtype=fnp.float32), m, k, n, plan)
    return _KS_OLD_MM(a, b)
# END FABLE-ENGINEER K-SPLIT v6

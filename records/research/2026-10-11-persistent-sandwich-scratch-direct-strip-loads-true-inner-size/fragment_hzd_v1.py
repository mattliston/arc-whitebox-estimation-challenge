# BEGIN OPUS-ENGINEER DIRECT STRIP LOADS OF THE K-SPLIT READS (a59 hzd v1, 2026-10-11; predictions byte-identical, bill down; compose after
# hzs-ks). A k-split read whose rows (or columns) carry a half-zero strip first copies its operand into a zero-interleaved pack, and every
# chunk's tree load then copies a slice of the pack into the tree's level-1 quadrant view: each element is reshaped and copied twice.
# Here the chunk's rows of the ORIGINAL operand go straight into the quadrant view: the main rows into its first al block rows, the
# strip's 32 rows into the deepest row bit's 0 half of block row al, zeros into its 1 half (the pack's zero rows); then the same S1
# addition, basis additions and level ops as the parent forward. The tree holds exactly the values the pack's load wrote. Bill: about
# 2 x the packed elements less (0.10% of a network's bill in S47s, a59_direct/probe_packs.py); requests: 8 per chunk against 4 per chunk
# + 8 for the pack. Only for K == k (no zero-padded inner columns); otherwise the parent runs. The transform-reuse key is cleared after
# a direct forward (the A region no longer holds the transform of the object it names).
_HZD_ON = True
_HZD_STATS = dict(a=0, b=0, parent=0)
_HZD_VIEWS = {}                             # id(quadrant view) -> (view, {(kind, al): (main, strip, zero)})
_HZD_PARENT_KS = _hz_ks_product


def _hzd_parts(quad, kind, L, s):
    # the quadrant view's main block rows (A) / columns (B) 0..s-1 and block s's halves on the deepest bit
    got = _HZD_VIEWS.get(id(quad))
    if got is None or got[0] is not quad:
        got = _HZD_VIEWS[id(quad)] = (quad, {})
    parts = got[1].get((kind, s))
    if parts is None:
        full = slice(None)
        if kind == 0:   # axes (row bits x L, block row, column bits x L, block column)
            parts = (quad[(full,) * L + (slice(0, s),)], quad[(full,) * (L - 1) + (0, s)], quad[(full,) * (L - 1) + (1, s)])
        else:           # axes (row bits x L, block row, column bits x L, block column)
            parts = (quad[(full,) * (2 * L + 1) + (slice(0, s),)], quad[(full,) * (2 * L) + (0, s)], quad[(full,) * (2 * L) + (1, s)])
        got[1][(kind, s)] = parts
    return parts


def _hzd_forward_a(a, M, q, k0, al, me):
    # = _ipt_forward_a_ab(pack[..., k0:k0 + q], 6, M, q) for the row-strip pack of a, without the pack
    L = 6
    TA, X1, A02, ops, basis = _ipt_side(0, L, M, q)
    h = _BH_HOLD.append
    quad = A02[1] if A02[1] is not None else _l3_quad(A02, 0, L)
    main, strip, zero = _hzd_parts(quad, 0, L, al)
    c = q >> L
    am = a[:me, k0:k0 + q]
    sm = fnp.reshape(am, (2,) * L + (al,) + (2,) * L + (c,))
    a2 = a[me:, k0:k0 + q]
    s2 = fnp.reshape(a2, (2,) * (L - 1) + (2,) * L + (c,))
    h((am, sm, a2, s2)); h(fnp.copyto(main, sm)); h(fnp.copyto(strip, s2)); h(fnp.copyto(zero, 0.0))
    X = X1[1]
    h(fnp.add(X[4], X[6], out=X[4]))
    for lo, hi in basis:
        h(fnp.add(lo, hi, out=lo))
    if ops is not None:
        for f, x, y, o in ops:
            h(f(x, y, out=o))
    else:
        for j in range(1, L + 1):
            X = X1[j]
            h(fnp.subtract(X[4], X[0], out=X[5])); h(fnp.subtract(X[2], X[5], out=X[3])); h(fnp.subtract(X[6], X[5], out=X[1]))
    _TR_LAST[0] = None
    _HZD_STATS['a'] += 1
    return TA


def _hzd_forward_b(b, N, q, k0, bl, ne):
    # = _ipt_forward_b_ab(pack[k0:k0 + q], 6, q, N) for the column-strip pack of b, without the pack
    L = 6
    TB, Y1, B02, ops, basis = _ipt_side(1, L, q, N)
    h = _BH_HOLD.append
    quad = B02[1] if B02[1] is not None else _l3_quad(B02, 1, L)
    main, strip, zero = _hzd_parts(quad, 1, L, bl)
    c = q >> L
    bm = b[k0:k0 + q, :ne]
    sm = fnp.reshape(bm, (2,) * L + (c,) + (2,) * L + (bl,))
    b2 = b[k0:k0 + q, ne:]
    s2 = fnp.reshape(b2, (2,) * L + (c,) + (2,) * (L - 1))
    h((bm, sm, b2, s2)); h(fnp.copyto(main, sm)); h(fnp.copyto(strip, s2)); h(fnp.copyto(zero, 0.0))
    Y = Y1[1]
    h(fnp.subtract(Y[3], Y[1], out=Y[1]))
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    if ops is not None:
        for f, x, y, o in ops:
            h(f(x, y, out=o))
    else:
        for j in range(1, L + 1):
            Y = Y1[j]
            h(fnp.add(Y[1], Y[0], out=Y[5])); h(fnp.subtract(Y[5], Y[2], out=Y[6])); h(fnp.subtract(Y[3], Y[5], out=Y[4]))
    _HZD_STATS['b'] += 1
    return TB


def _hz_ks_product(a, b, out, m, k, n, plan):
    M, N, sizes, hzr, hzc, K = plan
    if not (_HZD_ON and (hzr or hzc) and K == k):
        _HZD_STATS['parent'] += 1
        return _HZD_PARENT_KS(a, b, out, m, k, n, plan)
    hold = []
    al, bl = M // 64 - 1, N // 64 - 1
    me, ne = (64 * al if hzr else m), (64 * bl if hzc else n)
    if hzr:
        _HZ_STATS['row'] += 1
    if hzc:
        _HZ_STATS['col'] += 1
    TC, Z1, _pair, ops, basis = _ipt_side(2, 6, M, N)
    h = _BH_HOLD.append
    k0 = 0
    for j, q in enumerate(sizes):
        if hzr:
            TA = _hzd_forward_a(a, M, q, k0, al, me)
        else:
            ak = a[..., k0:k0 + q]
            h(ak)
            TA = _ipt_forward_a_ab(ak, 6, M, q)
        if hzc:
            TB = _hzd_forward_b(b, N, q, k0, bl, ne)
        else:
            bk = b[k0:k0 + q]
            h(bk)
            TB = _ipt_forward_b_ab(bk, 6, q, N)
        _hz_leaf(_hz_views(M, q, N, hzr, hzc)[1 if j else 0])
        if j:
            h(fnp.add(TC, _ks_acc(M, N), out=TC))
        k0 += q
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


_PB_CACHES = tuple(_PB_CACHES) + ('_HZD_VIEWS',)
# END OPUS-ENGINEER DIRECT STRIP LOADS OF THE K-SPLIT READS

# BEGIN OPUS-ENGINEER DIRECT STRIP LOADS (a59 hzd v3 = v2 with the view cache bounded (_HZD_VIEW_CAP quadrant views, oldest dropped first:
# a long worker session re-binds evicted tree keys, each with a new quadrant view); v2 = v1 (k-split reads) + the depth-6 strip products _hzg_product, with one loader for
# every strip layout; 2026-10-11; predictions byte-identical, bill down; compose after hzs-ks and hzs-gen). A strip product first copies an
# operand whose rows and / or columns carry a half-zero strip into a zero-interleaved pack, and the tree load then copies the pack (or a
# slice of it) into the tree's level-1 quadrant view: each element is reshaped and copied twice. Here the ORIGINAL operand goes straight
# into the quadrant view (axes: row bits x L, row block, column bits x L, column block): the main blocks, the strip's 32 rows / columns on
# the deepest bit's 0 half of the last block, the corner, and zeros into that bit's 1 half (the pack's zero rows / columns); then the same
# first addition, basis additions and level ops as the parent forward. The tree holds exactly the values the pack's load wrote. Bill:
# about 2 x the packed elements less (k-split 0.10%, strip products 0.035% of a network's bill in S47s, a59_direct/probe_packs.py).
# K-split reads only for K == k (no zero-padded inner columns), otherwise the parent runs. The transform-reuse key is cleared after a
# direct A forward (the A region no longer holds the transform of the object it names); unpacked operands take the parent's forwards.
_HZD_ON = True
_HZD_STATS = dict(a=0, b=0, parent=0, g=0)
_HZD_VIEWS = {}                             # id(quadrant view) -> (view, {(L, rows main, cols main, hr, hc): parts})
_HZD_VIEW_CAP = 512
_HZD_PARENT_KS = _hz_ks_product
_HZD_PARENT_G = _hzg_product


def _hzd_parts(quad, L, rm, cm, hr, hc):
    # the quadrant view's destinations: main blocks [:rm, :cm]; row strip (deepest row bit 0 of block rm) over the main columns; column
    # strip (deepest column bit 0 of block cm) over the main rows; corner; the zero halves (deepest bit 1 of the strip block, whole width)
    got = _HZD_VIEWS.get(id(quad))
    if got is None or got[0] is not quad:
        if len(_HZD_VIEWS) >= _HZD_VIEW_CAP:
            _HZD_VIEWS.pop(next(iter(_HZD_VIEWS)))
        got = _HZD_VIEWS[id(quad)] = (quad, {})
    key = (L, rm, cm, hr, hc)
    parts = got[1].get(key)
    if parts is None:
        f = slice(None)
        main = quad[(f,) * L + (slice(0, rm),) + (f,) * L + (slice(0, cm),)]
        rs = quad[(f,) * (L - 1) + (0, rm) + (f,) * L + (slice(0, cm),)] if hr else None
        cs = quad[(f,) * L + (slice(0, rm),) + (f,) * (L - 1) + (0, cm)] if hc else None
        cr = quad[(f,) * (L - 1) + (0, rm) + (f,) * (L - 1) + (0, cm)] if (hr and hc) else None
        rz = quad[(f,) * (L - 1) + (1, rm)] if hr else None
        cz = quad[(f,) * (2 * L) + (1, cm)] if hc else None
        parts = got[1][key] = (main, rs, cs, cr, rz, cz)
    return parts


def _hzd_load(quad, x, L, rm, cm, hr, hc):
    # x (rows, cols) in natural order: rows = 2^L rm (+ 2^(L-1) strip rows when hr), cols = 2^L cm (+ 2^(L-1) when hc)
    main, rs, cs, cr, rz, cz = _hzd_parts(quad, L, rm, cm, hr, hc)
    h = _BH_HOLD.append
    u, s = 1 << L, 1 << (L - 1)
    R, C = u * rm, u * cm
    xm = x[:R, :C]
    vm = fnp.reshape(xm, (2,) * L + (rm,) + (2,) * L + (cm,))
    h((xm, vm)); h(fnp.copyto(main, vm))
    if hr:
        xr = x[R:, :C]
        vr = fnp.reshape(xr, (2,) * (L - 1) + (2,) * L + (cm,))
        h((xr, vr)); h(fnp.copyto(rs, vr))
    if hc:
        xc = x[:R, C:]
        vc = fnp.reshape(xc, (2,) * L + (rm,) + (2,) * (L - 1))
        h((xc, vc)); h(fnp.copyto(cs, vc))
    if hr and hc:
        xx = x[R:, C:]
        vx = fnp.reshape(xx, (2,) * (L - 1) + (2,) * (L - 1))
        h((xx, vx)); h(fnp.copyto(cr, vx))
    if hr:
        h(fnp.copyto(rz, 0.0))
    if hc:
        h(fnp.copyto(cz, 0.0))


def _hzd_forward_a(x, M, K, hr, hc):
    # = _ipt_forward_a_ab(pack, 6, M, K) for the strip pack of x (rows M, inner K), without the pack
    L = 6
    TA, X1, A02, ops, basis = _ipt_side(0, L, M, K)
    h = _BH_HOLD.append
    quad = A02[1] if A02[1] is not None else _l3_quad(A02, 0, L)
    _hzd_load(quad, x, L, (M >> L) - (1 if hr else 0), (K >> L) - (1 if hc else 0), hr, hc)
    X = X1[1]
    h(fnp.add(X[4], X[6], out=X[4]))
    for lo, hi in basis:
        h(fnp.add(lo, hi, out=lo))
    if ops is not None:
        for f, y, z, o in ops:
            h(f(y, z, out=o))
    else:
        for j in range(1, L + 1):
            X = X1[j]
            h(fnp.subtract(X[4], X[0], out=X[5])); h(fnp.subtract(X[2], X[5], out=X[3])); h(fnp.subtract(X[6], X[5], out=X[1]))
    _TR_LAST[0] = None
    _HZD_STATS['a'] += 1
    return TA


def _hzd_forward_b(x, K, N, hr, hc):
    # = _ipt_forward_b_ab(pack, 6, K, N) for the strip pack of x (rows = inner K, columns N), without the pack
    L = 6
    TB, Y1, B02, ops, basis = _ipt_side(1, L, K, N)
    h = _BH_HOLD.append
    quad = B02[1] if B02[1] is not None else _l3_quad(B02, 1, L)
    _hzd_load(quad, x, L, (K >> L) - (1 if hr else 0), (N >> L) - (1 if hc else 0), hr, hc)
    Y = Y1[1]
    h(fnp.subtract(Y[3], Y[1], out=Y[1]))
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    if ops is not None:
        for f, y, z, o in ops:
            h(f(y, z, out=o))
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
        ak = a[..., k0:k0 + q]
        h(ak)
        TA = _hzd_forward_a(ak, M, q, True, False) if hzr else _ipt_forward_a_ab(ak, 6, M, q)
        bk = b[k0:k0 + q]
        h(bk)
        TB = _hzd_forward_b(bk, q, N, False, True) if hzc else _ipt_forward_b_ab(bk, 6, q, N)
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


def _hzg_product(a, b, m, k, n):
    # = the parent's strip product with the packed operands loaded directly (A: rows m and / or inner k carry strips; B: inner k and,
    # for a single product, columns n)
    hzr, hzk, hzc = m % 64 == 32, k % 64 == 32, n % 64 == 32
    if not (_HZD_ON and (hzr or hzk or hzc)):
        return _HZD_PARENT_G(a, b, m, k, n)
    M, K, N = (m + 63) // 64 * 64, (k + 63) // 64 * 64, (n + 63) // 64 * 64
    hold = []
    out = fnp.empty((m, n), dtype=fnp.float32)
    TA = _hzd_forward_a(a, M, K, hzr, hzk) if (hzr or hzk) else _ipt_forward_a_ab(a, 6, M, K)
    al, bl = M // 64 - 1, N // 64 - 1
    me, ne = (64 * al if hzr else m), (64 * bl if hzc else n)
    for j0 in range(0, N, 4096):
        j1 = min(N, j0 + 4096)
        w = j1 - j0
        bj = b if (j0 == 0 and j1 == N) else b[:, j0:j1]
        _BH_HOLD.append(bj)
        if hzk or hzc:
            TB = _hzd_forward_b(bj, K, w, hzk, hzc)
        else:
            TB = _ipt_forward_b_ab(bj, 6, K, w)
        TC, Z1, _pair, ops, basis = _ipt_side(2, 6, M, w)
        _hz_leaf(_hzg_views(M, K, w, hzr, hzk, hzc))
        if not (hzr or hzc):
            oj = out if (j0 == 0 and j1 == N) else out[:, j0:j1]
            _BH_HOLD.append(oj)
            _ks_unwind(TC, Z1, ops, basis, oj, 6, M, w)
            continue
        o = fnp.empty((M, w), dtype=fnp.float32)
        _ks_unwind(TC, Z1, ops, basis, o, 6, M, w)
        if hzr and hzc:
            o4 = fnp.reshape(o, (64, al + 1, 64, bl + 1))
            p0 = fnp.reshape(o4[:, :al, :, :bl], (me, ne)); p1 = fnp.reshape(o4[:, :al, 0::2, bl], (me, 32))
            p2 = fnp.reshape(o4[0::2, al, :, :bl], (32, ne))
            hold.append((o4, p0, p1, p2, fnp.copyto(out[:me, :ne], p0), fnp.copyto(out[:me, ne:], p1),
                         fnp.copyto(out[me:, :ne], p2), fnp.copyto(out[me:, ne:], o4[0::2, al, 0::2, bl])))
        elif hzr:
            o3 = fnp.reshape(o, (64, al + 1, w))
            p0 = fnp.reshape(o3[:, :al, :], (me, w))
            hold.append((o3, p0, fnp.copyto(out[:me, j0:j1], p0), fnp.copyto(out[me:, j0:j1], o3[0::2, al, :])))
        else:
            o3 = fnp.reshape(o, (M, 64, bl + 1))
            p0 = fnp.reshape(o3[:, :, :bl], (m, ne))
            hold.append((o3, p0, fnp.copyto(out[:, :ne], p0), fnp.copyto(out[:, ne:], o3[:, 0::2, bl])))
    _BH_HOLD.clear()
    del hold
    _HZG_STATS['push' if n > 1024 else 'single'] += 1
    _HZD_STATS['g'] += 1
    return out


_PB_CACHES = tuple(_PB_CACHES) + ('_HZD_VIEWS',)
# END OPUS-ENGINEER DIRECT STRIP LOADS

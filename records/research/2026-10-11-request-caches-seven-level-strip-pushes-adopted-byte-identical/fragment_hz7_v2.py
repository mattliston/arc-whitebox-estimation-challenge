# BEGIN OPUS-ENGINEER SEVEN-LEVEL STRIP PUSHES AND READS (a57 hz7 v2 = v1 + the plane reads, 2026-10-11; lead 03:5xZ twin; rounding-level, NOT bit-exact). A packet push
# (n > 1024 columns, m, k <= 1024) that runs at 6 levels because m or k is not a multiple of 128 runs at 7 levels: a dimension whose
# rest beyond its last multiple of 128 is 32 or 64 becomes a half-zero strip (the rest interleaved with zeros on the deepest index bit,
# exactly the 6-level strip products' layout one level deeper: 64 strip slots instead of 32), a multiple of 128 stays as it is; the
# columns run in 2048-wide packets sharing one 7-level A tree (4096-wide 7-level B / C trees do not fit the arena), the last packet
# zero-padded to a multiple of 128. The leaf stage takes the strip products' chunk groups (_hz_groups: the strip row needs 4 of the 7
# deepest chunk products, the inner strip term vanishes exactly in chunks 2, 3, 6), the trees are _ipt_side's 7-level ones
# (_IPT_REGION[7], d7cp's region). Rule: taken when the fitted bill model (a30_hz7/price_hz7.py, calibrated on S38's census) says 7
# levels with strips bill at least _HZ7_GAIN below 6 levels with strips, for products the parent runs at 6 levels and d7cp does not take.
_HZ7_ON = True
_HZ7_GAIN = 0.03
_HZ7_PK = 2048
_HZ7_STATS = {'products': 0, 'packets': 0}
_HZ7_PLANS = {}
_HZ7_VIEWS = {}
_HZ7_VIEW_CAP = 128
_HZ7_OLD_MM = _mm


def _hz7_dim(x):
    """(main 128-blocks, strip flag) of a dimension, or None when its rest beyond 128 x blocks is not 0, 32 or 64."""
    full, rest = divmod(int(x), 128)
    if rest == 0:
        return full, False
    if rest in (32, 64):
        return full, True
    return None


def _hz7_bill(L, m, k, n, pk):
    # price_hz7.push_bill: a packet push at L levels with strips (dimensions padded to multiples of 2^(L-1), the odd half as a strip)
    S = lambda q: ((7 / 4) ** q - 1) / 0.75
    U = 1 << L; H = U // 2
    M, K = (m + H - 1) // H * H, (k + H - 1) // H * H
    hzr, hzk = M % U == H, K % U == H
    M, K = (M + U - 1) // U * U, (K + U - 1) // U * U
    N = (n + U - 1) // U * U
    packets = [min(N, j0 + pk) - j0 for j0 in range(0, N, pk)]
    pb = lambda w: (2 * M * K * w * (7 / 8) ** L - M * w * (7 / 4) ** L + 0.7668 * K * w * S(L) + 1.5213 * M * w * S(L)
                    + 2.6307 * (K * w + M * w))
    bill = 0.7669 * M * K * S(L) + 2.6307 * M * K + sum(pb(w) for w in packets)
    bill += 0.98 * ((M * K if (hzr or hzk or (M, K) != (m, k)) else 0) + (K * N if (hzk or K != k or N != n) else 0)
                    + (m * n if (hzr or N != n) else 0))
    for w in packets:
        cbt = w // U
        ra = M // U - (1 if hzr else 0)
        if hzr:
            bill -= 3 * 7 ** (L - 1) * (2 * (K // U) - 1) * cbt
        if hzk:
            bill -= 3 * 7 ** (L - 1) * 2 * ra * cbt
    return bill


def _hz7_plan(m, k, n):
    key = (m, k, n)
    if key in _HZ7_PLANS:
        return _HZ7_PLANS[key]
    take = False
    dm, dk = _hz7_dim(m), _hz7_dim(k)
    if (_HZ7_ON and 7 in _IPT_REGION and dm is not None and dk is not None and n > 1024 and n % 32 == 0
            and 256 <= m <= 1024 and 256 <= k <= 1024 and (dm[1] or dk[1] or m % 128 or k % 128)
            and not (_D7_ON and m in _D7_DIMS and (k in _D7_DIMS or k in _D7_KPAD))):
        M, K = 128 * (dm[0] + (1 if dm[1] else 0)), 128 * (dk[0] + (1 if dk[1] else 0))
        if _ipt_fits(7, M, K, _HZ7_PK) and _rule_depth('push', m, k, n) == 6:
            take = _hz7_bill(7, m, k, n, _HZ7_PK) < (1.0 - _HZ7_GAIN) * _hz7_bill(6, m, k, n, 4096)
    _HZ7_PLANS[key] = take
    return take


def _hz7_pack_a(x, m, k, hold):
    """A (m x k) -> (128, rl, 128, kl): strip rows / inner strip columns interleaved with zeros on the deepest bit."""
    (rm, hzr), (km, hzk) = _hz7_dim(m), _hz7_dim(k)
    rl, kl = rm + (1 if hzr else 0), km + (1 if hzk else 0)
    y = fnp.zeros((128, rl, 128, kl), dtype=fnp.float32)
    xm = fnp.reshape(x[:128 * rm, :128 * km], (128, rm, 128, km))
    hold.append((xm, fnp.copyto(y[:, :rm, :, :km], xm)))
    sr, sk = m - 128 * rm, k - 128 * km
    if hzr:
        xr = fnp.reshape(x[128 * rm:, :128 * km], (sr, 128, km))
        hold.append((xr, fnp.copyto(y[0:2 * sr:2, rm, :, :km], xr)))
    if hzk:
        xc = fnp.reshape(x[:128 * rm, 128 * km:], (128, rm, sk))
        hold.append((xc, fnp.copyto(y[:, :rm, 0:2 * sk:2, km], xc)))
    if hzr and hzk:
        hold.append(fnp.copyto(y[0:2 * sr:2, rm, 0:2 * sk:2, km], x[128 * rm:, 128 * km:]))
    return y


def _hz7_pack_b(x, k, n, NP, hold):
    """B (k x n) -> (128, kl, NP): inner strip rows interleaved, columns natural and zero-padded to NP."""
    km, hzk = _hz7_dim(k)
    kl = km + (1 if hzk else 0)
    if not hzk and NP == n:
        return x
    y = fnp.zeros((128, kl, NP), dtype=fnp.float32)
    xm = fnp.reshape(x[:128 * km], (128, km, n))
    hold.append((xm, fnp.copyto(y[:, :km, :n], xm)))
    if hzk:
        sk = k - 128 * km
        hold.append(fnp.copyto(y[0:2 * sk:2, km, :n], x[128 * km:]))
    return y


def _hz7_views(M, K, w, hzr, hzk):
    key = (M, K, w, hzr, hzk)
    got = _HZ7_VIEWS.get(key)
    if got is not None:
        return got
    TA = _ipt_side(0, 7, M, K)[0]
    TB = _ipt_side(1, 7, K, w)[0]
    TC = _ipt_side(2, 7, M, w)[0]
    v = _hz_leaf_views(TA, TB, TC, M // 128 - (1 if hzr else 0), K // 128 - (1 if hzk else 0), w // 128, hzr, hzk, False)
    if len(_HZ7_VIEWS) >= _HZ7_VIEW_CAP:
        _HZ7_VIEWS.pop(next(iter(_HZ7_VIEWS)))
    _HZ7_VIEWS[key] = v
    return v


def _hz7_product(a, b, m, k, n):
    (rm, hzr), (km, hzk) = _hz7_dim(m), _hz7_dim(k)
    M, K = 128 * (rm + (1 if hzr else 0)), 128 * (km + (1 if hzk else 0))
    NP = (n + 127) // 128 * 128
    hold = []
    aa = _hz7_pack_a(a, m, k, hold) if (hzr or hzk) else a
    bb = _hz7_pack_b(b, k, n, NP, hold)
    out = fnp.empty((m, n), dtype=fnp.float32)
    TA = _ipt_forward_a_ab(aa, 7, M, K)
    sr = m - 128 * rm
    for j0 in range(0, NP, _HZ7_PK):
        j1 = min(NP, j0 + _HZ7_PK)
        w = j1 - j0
        wn = min(n, j1) - j0                       # the packet's real columns
        bj = bb if (j0 == 0 and j1 == NP and bb is not b) else bb[..., j0:j1]
        _BH_HOLD.append(bj)
        TB = _ipt_forward_b_ab(bj, 7, K, w)
        TC, Z1, _pair, ops, basis = _ipt_side(2, 7, M, w)
        _hz_leaf(_hz7_views(M, K, w, hzr, hzk))
        if not hzr and wn == w:
            oj = out if (j0 == 0 and j1 == n) else out[:, j0:j1]
            _BH_HOLD.append(oj)
            _ks_unwind(TC, Z1, ops, basis, oj, 7, M, w)
        else:
            o = fnp.empty((M, w), dtype=fnp.float32)
            _ks_unwind(TC, Z1, ops, basis, o, 7, M, w)
            if hzr:
                o3 = fnp.reshape(o, (128, rm + 1, w))
                p0 = fnp.reshape(o3[:, :rm, :wn], (128 * rm, wn))
                hold.append((o3, p0, fnp.copyto(out[:128 * rm, j0:j0 + wn], p0), fnp.copyto(out[128 * rm:, j0:j0 + wn], o3[0:2 * sr:2, rm, :wn])))
            else:
                hold.append(fnp.copyto(out[:, j0:j0 + wn], o[:, :wn]))
            hold.append(o)
        _HZ7_STATS['packets'] += 1
    _BH_HOLD.clear()
    del hold
    _HZ7_STATS['products'] += 1
    return out


def _mm(a, b):
    if (_HZ7_ON and _SCRATCH is not None and _AB_ON and _WINOGRAD and len(a.shape) == len(b.shape) == 2
            and str(a.dtype) == str(b.dtype) == 'float32' and int(a.shape[1]) == int(b.shape[0]) and _UBC_CONTEXT[0] is None):
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        if n > 1024 and m <= 1024 and k <= 1024:
            scope = _W5_PORT_SCOPE
            port = scope is not None and scope['grant'] is not None and scope['grant']['left'] is a
            if not port and _hz7_plan(m, k, n):
                return _hz7_product(a, b, m, k, n)
    return _HZ7_OLD_MM(a, b)


# v2: the k-split plane reads (k >= 2048, m, n <= 1024) at 7 levels on the read kernel's private regions (_r7_enter: _R7_REGION and
# its own tree caches), with a row / column rest of 32 or 64 as a half-zero strip, chunks of the inner dimension (multiples of 128) that
# fit the regions, the leaf products of every chunk after the first summed in a leaf-sum buffer behind the C tree (as _r7_product).
# Its leaf views have their own cache (the R7 trees are other arena regions than the push trees of the same shape). Rule: the fitted
# read model (a30_hz7/price_hz7.py read_bill, within 0.1-2% of the measured 6-level reads) gains more than _HZ7R_GAIN.
_HZ7R_ON = True
_HZ7R_GAIN = 0.015
_HZ7R_PLANS = {}
_HZ7R_SIDES = {}
_HZ7R_VIEWS = {}
_HZ7R_ACCS = {}
_HZ7R_STATS = {'products': 0, 'chunks': 0}


def _hz7r_bill(L, m, k, n):
    # price_hz7.read_bill (7 levels: the read kernel's private regions)
    S = lambda q: ((7 / 4) ** q - 1) / 0.75
    U = 1 << L; H = U // 2
    M, N = (m + H - 1) // H * H, (n + H - 1) // H * H
    hzr, hzc = M % U == H, N % U == H
    M, N, K = (M + U - 1) // U * U, (N + U - 1) // U * U, (k + U - 1) // U * U
    region = {6: (90354432, 120472576), 7: (7 ** 7 * 128, 7 ** 7 * 128)}[L]
    cmax = U * min(region[0] // (7 ** L * (M // U)), region[1] // (7 ** L * (N // U)))
    if cmax < U:
        return None
    nb = K // U; nch = -(-nb // (cmax // U)); base, r = divmod(nb, nch)
    sizes = [(base + 1) * U] * r + [base * U] * (nch - r)
    CL, CC = 0.7669 * S(L) + 2.6307, 1.5213 * S(L) + 2.6307
    b = (sum(2 * (7 / 8) ** L * M * q * N + CL * (M * q + q * N) + (CC - (7 / 4) ** L) * M * N for q in sizes)
         + (nch - 1) * m * n + m * n - (nch - 1) * (CC * M * N + m * n - 7 ** L * (M // U) * (N // U)))
    b += (2.0 * m * k if (hzr or K != k or M != m) else 0) + (2.0 * k * n if (hzc or K != k or N != n) else 0) + 2.0 * m * n
    b -= (3 * 7 ** (L - 1) * 2 * (K // U) * (N // U) if hzr else 0) + (3 * 7 ** (L - 1) * 2 * (M // U - (1 if hzr else 0)) * (K // U) if hzc else 0)
    return b


def _hz7r_sizes(M, K, N):
    cmax = 128 * min((7 ** 7 * 128) // (7 ** 7 * (M // 128)), (7 ** 7 * 128) // (7 ** 7 * (N // 128)))
    nb = K // 128
    nch = -(-nb // (cmax // 128))
    base, r = divmod(nb, nch)
    return [(base + 1) * 128] * r + [base * 128] * (nch - r)


def _hz7r_plan(m, k, n):
    key = (m, k, n)
    if key in _HZ7R_PLANS:
        return _HZ7R_PLANS[key]
    take = False
    dm, dn = _hz7_dim(m), _hz7_dim(n)
    if (_HZ7R_ON and dm is not None and dn is not None and k >= 2048 and k % 32 == 0 and 256 <= m <= 1024 and 256 <= n <= 1024
            and (dm[1] or dn[1]) and (m, k, n) not in _R7_SHAPES):
        b7, b6 = _hz7r_bill(7, m, k, n), _hz7r_bill(6, m, k, n)
        take = b7 is not None and b6 is not None and b7 < (1.0 - _HZ7R_GAIN) * b6
    _HZ7R_PLANS[key] = take
    return take


def _hz7r_side(kind, T, lo0, lo1, hzr, hzk, hzc):
    # = _hz_side for the read kernel's private trees, in its own cache
    key = (kind, tuple(int(x) for x in T.shape), hzr, hzk, hzc)
    got = _HZ7R_SIDES.get(key)
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
    got = _HZ7R_SIDES[key] = (views, fills)
    return got


def _hz7r_acc(M, N):
    got = _HZ7R_ACCS.get((M, N))
    if got is None:
        blocks = 7 ** 7 * (M // 128) * (N // 128)
        got = _HZ7R_ACCS[(M, N)] = _ipt_view(_SCRATCH[0], _R7_REGION[0] + _R7_REGION[1] + blocks, (7,) * 7 + (M // 128, N // 128))
    return got


def _hz7r_views(M, q, N, hzr, hzc):
    key = (M, q, N, hzr, hzc)
    got = _HZ7R_VIEWS.get(key)
    if got is not None:
        return got
    TA = _ipt_side(0, 7, M, q)[0]
    TB = _ipt_side(1, 7, q, N)[0]
    TC = _ipt_side(2, 7, M, N)[0]
    ra, kq, cb = M // 128 - (1 if hzr else 0), q // 128, N // 128 - (1 if hzc else 0)
    entry = []
    for T, kind in ((TC, 2), (_hz7r_acc(M, N), 3)):
        A = _hz7r_side(0, TA, ra, kq, hzr, False, hzc)[0]
        B = _hz7r_side(1, TB, kq, cb, hzr, False, hzc)[0]
        O, fills = _hz7r_side(kind, T, ra, cb, hzr, False, hzc)
        entry.append(dict(mats=list(zip(A, B, O)), fills=fills))
    _HZ7R_VIEWS[key] = entry
    return entry


def _hz7r_product(a, b, m, k, n):
    (rm, hzr), (cm, hzc) = _hz7_dim(m), _hz7_dim(n)
    M, N = 128 * (rm + (1 if hzr else 0)), 128 * (cm + (1 if hzc else 0))
    K = (k + 127) // 128 * 128
    sizes = _hz7r_sizes(M, K, N)
    sr, sc = m - 128 * rm, n - 128 * cm
    out = fnp.empty((m, n), dtype=fnp.float32)
    saved = _r7_enter()
    try:
        hold = []
        if hzr:
            aa = fnp.zeros((128, rm + 1, K), dtype=fnp.float32)
            am = fnp.reshape(a[:128 * rm], (128, rm, k))
            hold.append((am, fnp.copyto(aa[:, :rm, :k], am), fnp.copyto(aa[0:2 * sr:2, rm, :k], a[128 * rm:])))
        elif K != k:
            aa = fnp.zeros((M, K), dtype=fnp.float32)
            hold.append(fnp.copyto(aa[:, :k], a))
        else:
            aa = a
        if hzc:   # packed in the source's memory order, as the 6-level strip reads
            bbt = fnp.zeros((128, cm + 1, K), dtype=fnp.float32)
            bt = b.T
            bm = fnp.reshape(bt[:128 * cm], (128, cm, k))
            hold.append((bt, bm, fnp.copyto(bbt[:, :cm, :k], bm), fnp.copyto(bbt[0:2 * sc:2, cm, :k], bt[128 * cm:])))
            bb = fnp.transpose(bbt, (2, 0, 1))
            hold.append(bbt)
        elif K != k:
            bb = fnp.zeros((K, N), dtype=fnp.float32)
            hold.append(fnp.copyto(bb[:k], b))
        else:
            bb = b
        TC, Z1, _pair, ops, basis = _ipt_side(2, 7, M, N)
        h = _BH_HOLD.append
        k0 = 0
        for j, q in enumerate(sizes):
            ak, bk = aa[..., k0:k0 + q], bb[k0:k0 + q]
            h((ak, bk))
            _ipt_forward_a_ab(ak, 7, M, q)
            _ipt_forward_b_ab(bk, 7, q, N)
            _hz_leaf(_hz7r_views(M, q, N, hzr, hzc)[1 if j else 0])
            if j:
                h(fnp.add(TC, _hz7r_acc(M, N), out=TC))
            k0 += q
            _HZ7R_STATS['chunks'] += 1
        if M == m and N == n:
            _ks_unwind(TC, Z1, ops, basis, out, 7, M, N)
        else:
            o = fnp.empty((M, N), dtype=fnp.float32)
            _ks_unwind(TC, Z1, ops, basis, o, 7, M, N)
            me, ne = 128 * rm if hzr else m, 128 * cm if hzc else n
            if hzr and hzc:
                o4 = fnp.reshape(o, (128, rm + 1, 128, cm + 1))
                p0 = fnp.reshape(o4[:, :rm, :, :cm], (me, ne)); p1 = fnp.reshape(o4[:, :rm, 0:2 * sc:2, cm], (me, sc))
                p2 = fnp.reshape(o4[0:2 * sr:2, rm, :, :cm], (sr, ne))
                hold.append((o4, p0, p1, p2, fnp.copyto(out[:me, :ne], p0), fnp.copyto(out[:me, ne:], p1),
                             fnp.copyto(out[me:, :ne], p2), fnp.copyto(out[me:, ne:], o4[0:2 * sr:2, rm, 0:2 * sc:2, cm])))
            elif hzr:
                o3 = fnp.reshape(o, (128, rm + 1, N))
                p0 = fnp.reshape(o3[:, :rm, :n], (me, n))
                hold.append((o3, p0, fnp.copyto(out[:me], p0), fnp.copyto(out[me:], o3[0:2 * sr:2, rm, :n])))
            else:
                o3 = fnp.reshape(o, (M, 128, cm + 1))
                p0 = fnp.reshape(o3[:m, :, :cm], (m, ne))
                hold.append((o3, p0, fnp.copyto(out[:, :ne], p0), fnp.copyto(out[:, ne:], o3[:m, 0:2 * sc:2, cm])))
            hold.append(o)
        del hold
    finally:
        _r7_leave(saved)
    _HZ7R_STATS['products'] += 1
    return out


_HZ7R_OLD_MM = _mm


def _mm(a, b):
    if (_HZ7R_ON and _SCRATCH is not None and _AB_ON and _WINOGRAD and len(a.shape) == len(b.shape) == 2
            and str(a.dtype) == str(b.dtype) == 'float32' and int(a.shape[1]) == int(b.shape[0]) and _UBC_CONTEXT[0] is None):
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        if k >= 2048 and m <= 1024 and n <= 1024 and _hz7r_plan(m, k, n):
            return _hz7r_product(a, b, m, k, n)
    return _HZ7R_OLD_MM(a, b)


_PB_CACHES = tuple(_PB_CACHES) + ('_HZ7_VIEWS', '_HZ7R_VIEWS', '_HZ7R_SIDES', '_HZ7R_ACCS')
# END OPUS-ENGINEER SEVEN-LEVEL STRIP PUSHES AND READS

# BEGIN OPUS-ENGINEER SEVEN-LEVEL STRIP PUSHES (a57 hz7 v1, 2026-10-11; lead 03:5xZ twin; rounding-level, NOT bit-exact). A packet push
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


_PB_CACHES = tuple(_PB_CACHES) + ('_HZ7_VIEWS',)
# END OPUS-ENGINEER SEVEN-LEVEL STRIP PUSHES

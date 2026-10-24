# BEGIN OPUS-ENGINEER HALF-ZERO STRIP PRODUCTS (hzs-gen v7 = v5 with the seven-level exclusion tied to _D7_ON, 2026-10-01; rounding-level, NOT bit-exact; compose AFTER hzs-ks): the
# half-zero strip of hzs-ks for every other depth-6 tree product whose m, k or n is an odd multiple of 32: packet pushes (n > 1024,
# one A tree, 4096-column packets exactly as _cp_d6_shared) and single products (all dims <= 1024). A dimension's 32 rows/columns
# beyond its last multiple of 64 and 32 zeros are interleaved on the deepest index bit, so:
#   rows of A / columns of B (m, n): the strip leaf row / column needs 4 of the 7 deepest chunk products (as hzs-ks);
#   the inner index (k): the last leaf column of A is zero in chunks 2, 6 and the last leaf row of B in chunks 2, 3, so the
#   inner-strip term vanishes EXACTLY in chunks 2, 3, 6: those three run their leaf matmul on the leading inner indices only.
# A pad therefore costs the depth-5 rate on 32 rows/columns instead of the depth-6 rate on 64. Keys are those of the padded
# depth-6 product. Rule: taken when the fitted bill model says the bill falls by more than _HZG_PRICE flops per extra request
# against the cheaper of the parent's depth-5 and zero-padded depth-6 products.
_HZG_ON = True
_HZG_PRICE = 750000.0
_HZG_D5 = True                              # also take products the parent keeps at depth 5 when the rule says they pay (v4 switch)
_HZG_STATS = {'push': 0, 'single': 0}
_HZG_VIEWS = {}
_HZG_VIEW_CAP = 128
_HZG_PLANS = {}
_HZG_OLD_MM = _mm


def _hzg_pack(x, r, c, hz_r, hz_c, hold):
    # (r, c) -> (64, R/64, 64, C/64) with the odd 32 rows / columns interleaved with zeros on the deepest bit; R, C = r, c rounded to 64
    R, C = (r + 63) // 64 * 64, (c + 63) // 64 * 64
    rl, cl = R // 64, C // 64
    rm, cm = (rl - 1 if hz_r else rl), (cl - 1 if hz_c else cl)
    y = fnp.zeros((64, rl, 64, cl), dtype=fnp.float32)
    xm = fnp.reshape(x[:64 * rm, :64 * cm], (64, rm, 64, cm))
    hold.append((xm, fnp.copyto(y[:, :rm, :, :cm], xm)))
    if hz_r:
        xr = fnp.reshape(x[64 * rm:, :64 * cm], (32, 64, cm))
        hold.append((xr, fnp.copyto(y[0::2, rm, :, :cm], xr)))
    if hz_c:
        xc = fnp.reshape(x[:64 * rm, 64 * cm:], (64, rm, 32))
        hold.append((xc, fnp.copyto(y[:, :rm, 0::2, cm], xc)))
    if hz_r and hz_c:
        hold.append(fnp.copyto(y[0::2, rm, 0::2, cm], x[64 * rm:, 64 * cm:]))
    return y


def _hzg_pack_rows(x, r, c, hold):
    # (r, c), r = 64 rm + 32 -> (64, rm + 1, c): rows interleaved, columns natural (the packets slice them)
    rm = r // 64
    y = fnp.zeros((64, rm + 1, c), dtype=fnp.float32)
    xm = fnp.reshape(x[:64 * rm], (64, rm, c))
    hold.append((xm, fnp.copyto(y[:, :rm, :], xm), fnp.copyto(y[0::2, rm, :], x[64 * rm:])))
    return y


def _hzg_views(M, K, w, hzr, hzk, hzc):
    key = (M, K, w, hzr, hzk, hzc)
    got = _HZG_VIEWS.get(key)
    if got is not None:
        return got
    TA = _ipt_side(0, 6, M, K)[0]
    TB = _ipt_side(1, 6, K, w)[0]
    TC = _ipt_side(2, 6, M, w)[0]
    v = _hz_leaf_views(TA, TB, TC, M // 64 - (1 if hzr else 0), K // 64 - (1 if hzk else 0), w // 64 - (1 if hzc else 0), hzr, hzk, hzc)
    if len(_HZG_VIEWS) >= _HZG_VIEW_CAP:
        _HZG_VIEWS.pop(next(iter(_HZG_VIEWS)))
    _HZG_VIEWS[key] = v
    return v


def _hzg_product(a, b, m, k, n):
    hzr, hzk, hzc = m % 64 == 32, k % 64 == 32, n % 64 == 32
    M, K, N = (m + 63) // 64 * 64, (k + 63) // 64 * 64, (n + 63) // 64 * 64
    hold = []
    aa = _hzg_pack(a, m, k, hzr, hzk, hold) if (hzr or hzk) else a
    if hzc:                                   # single product only (n <= 1024): B rows (inner) and columns interleaved
        bb = _hzg_pack(b, k, n, hzk, hzc, hold)
    elif hzk:
        bb = _hzg_pack_rows(b, k, n, hold)
    else:
        bb = b
    out = fnp.empty((m, n), dtype=fnp.float32)
    TA = _ipt_forward_a_ab(aa, 6, M, K)
    al, bl = M // 64 - 1, N // 64 - 1
    me, ne = (64 * al if hzr else m), (64 * bl if hzc else n)
    for j0 in range(0, N, 4096):
        j1 = min(N, j0 + 4096)
        w = j1 - j0
        bj = bb if (j0 == 0 and j1 == N) else bb[..., j0:j1]
        _BH_HOLD.append(bj)
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
    return out


def _hzg_bill6(M, K, w):
    # the parent's fitted alt-basis tree bill beyond the shared A side (per packet)
    S = ((7 / 4) ** 6 - 1) / 0.75
    return 2 * M * K * w * (7 / 8) ** 6 - M * w * (7 / 4) ** 6 + 0.7668 * K * w * S + 1.5213 * M * w * S + 2.6307 * (K * w + M * w)


def _hzg_plan(m, k, n, port):
    # port = the W5 port holds a grant for this product (the parent then runs it at depth 5 with the port's A leaf)
    key = (m, k, n, port)
    if key in _HZG_PLANS:
        return _HZG_PLANS[key]
    take = False
    hzr, hzk, hzc = m % 64 == 32, k % 64 == 32, n % 64 == 32
    push = n > 1024
    if (_HZG_ON and (hzr or hzk or hzc) and m % 32 == 0 and k % 32 == 0 and n % 32 == 0 and 96 <= m <= 1024 and 96 <= k <= 1024
            and 96 <= n <= 8192 and not (push and hzc)):
        M, K, N = (m + 63) // 64 * 64, (k + 63) // 64 * 64, (n + 63) // 64 * 64
        packets = [min(N, j0 + 4096) - j0 for j0 in range(0, N, 4096)]
        if all(_ipt_fits(6, M, K, w) for w in packets):
            S = ((7 / 4) ** 6 - 1) / 0.75
            pad6 = (0.7669 * M * K * S + 2.6307 * M * K + sum(_hzg_bill6(M, K, w) for w in packets)
                    + 0.98 * ((M * K if (hzr or hzk) else 0) + (K * N if (hzk or hzc) else 0) + (m * n if (hzr or hzc) else 0)))
            save = 0.0
            for w in packets:
                cbt = w // 64
                ra, cb = M // 64 - (1 if hzr else 0), cbt - (1 if hzc else 0)
                if hzr:
                    save += 3 * 7 ** 5 * (2 * (K // 64) - 1) * cbt
                if hzc:
                    save += 3 * 7 ** 5 * (2 * (K // 64) - 1) * ra
                if hzk:
                    save += 3 * 7 ** 5 * 2 * ra * cb
            # model checked against measured bills in hzs/model_check.py: within 2% (the depth-5 model reads 2.5% high)
            hz = pad6 - save + 0.02 * ((M * K if (hzr or hzk) else 0) + (K * N if (hzk or hzc) else 0)) + (m * n if (hzr or hzc) else 0)
            b5 = 0.975 * _p6_bill(5, m, k, n, 0)
            npk = len(packets)
            if push:
                old6 = m % 64 == 0 and _rule_depth('push', m, k, n) == 6
            else:
                old6 = (not port) and _p6_plan(m, k, n) is not None
            if old6:
                take = hz + _HZG_PRICE * (10 + (6 if (hzr or hzc) else 3) * npk) < pad6
            elif _HZG_D5:
                take = hz + _HZG_PRICE * 60 * npk < b5
    _HZG_PLANS[key] = take
    return take


def _mm(a, b):
    if (_HZG_ON and _SCRATCH is not None and _AB_ON and _WINOGRAD and len(a.shape) == len(b.shape) == 2
            and str(a.dtype) == str(b.dtype) == 'float32' and int(a.shape[1]) == int(b.shape[0])):
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        if ((m % 64 == 32 or k % 64 == 32 or n % 64 == 32) and k <= 1024 and m <= 1024 and _UBC_CONTEXT[0] is None
                and not (_D7_ON and n > 1024 and m in _D7_DIMS and (k in _D7_DIMS or k in _D7_KPAD))):
            scope = _W5_PORT_SCOPE
            port = scope is not None and scope['grant'] is not None and scope['grant']['left'] is a
            if _hzg_plan(m, k, n, port):
                return _hzg_product(a, b, m, k, n)
        if k >= 2048 and m <= 1024 and n <= 1024 and (m % 64 == 32 or n % 64 == 32 or k % 64 == 32):
            plan = _ks_plan(m, k, n)          # single-chunk reads: the parent's k-split entry takes only two chunks or more
            if plan is not None and len(plan) == 6 and len(plan[2]) == 1:
                return _hz_ks_product(a, b, fnp.empty((m, n), dtype=fnp.float32), m, k, n, plan)
    return _HZG_OLD_MM(a, b)


_PB_CACHES = _PB_CACHES + ('_HZG_VIEWS',)
# END OPUS-ENGINEER HALF-ZERO STRIP PRODUCTS

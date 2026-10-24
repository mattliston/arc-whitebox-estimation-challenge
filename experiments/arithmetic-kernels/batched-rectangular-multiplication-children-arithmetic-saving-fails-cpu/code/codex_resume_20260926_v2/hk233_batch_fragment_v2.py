def _hkb233_product(a, b, mm_many, quantum=64):
    m, k, n = (int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))
    assert int(b.shape[0]) == k and a.ndim == b.ndim == 2
    M = (m + 2 * quantum - 1) // (2 * quantum) * 2 * quantum
    K = (k + 3 * quantum - 1) // (3 * quantum) * 3 * quantum
    N = (n + 3 * quantum - 1) // (3 * quantum) * 3 * quantum
    h, v, w = (M // 2, K // 3, N // 3)
    aa = fnp.zeros((M, K), dtype=a.dtype)
    bb = fnp.zeros((K, N), dtype=b.dtype)
    fnp.copyto(aa[:m, :k], a)
    fnp.copyto(bb[:k, :n], b)
    A_1_1 = aa[0 * h:1 * h, 0 * v:1 * v]
    A_1_2 = aa[0 * h:1 * h, 1 * v:2 * v]
    A_1_3 = aa[0 * h:1 * h, 2 * v:3 * v]
    A_2_1 = aa[1 * h:2 * h, 0 * v:1 * v]
    A_2_2 = aa[1 * h:2 * h, 1 * v:2 * v]
    A_2_3 = aa[1 * h:2 * h, 2 * v:3 * v]
    B_1_1 = bb[0 * v:1 * v, 0 * w:1 * w]
    B_1_2 = bb[0 * v:1 * v, 1 * w:2 * w]
    B_1_3 = bb[0 * v:1 * v, 2 * w:3 * w]
    B_2_1 = bb[1 * v:2 * v, 0 * w:1 * w]
    B_2_2 = bb[1 * v:2 * v, 1 * w:2 * w]
    B_2_3 = bb[1 * v:2 * v, 2 * w:3 * w]
    B_3_1 = bb[2 * v:3 * v, 0 * w:1 * w]
    B_3_2 = bb[2 * v:3 * v, 1 * w:2 * w]
    B_3_3 = bb[2 * v:3 * v, 2 * w:3 * w]
    m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9, m_10, m_11, m_12, m_13, m_14, m_15 = mm_many([(A_2_1, B_1_1 - B_2_1 - B_1_3), (A_1_2, B_2_2 - B_1_2 - B_2_3), (A_1_3, -B_3_2 - B_1_3 + B_3_3), (A_2_3, -B_3_1 - B_2_3 + B_3_3), (A_1_2 + A_1_1, B_1_2), (A_1_1 + A_1_3, B_1_3), (A_1_1 - A_2_1, B_1_1 - B_3_1 - B_1_2), (A_2_1 + A_1_2, B_2_1 - B_1_2), (A_2_2 + A_2_1, B_2_1), (A_1_2 - A_2_2, B_2_1 - B_2_2 + B_3_2), (A_2_2 + A_2_3, B_2_3), (A_1_1 - A_2_1 + A_1_3, B_3_1 - B_1_3), (A_1_2 - A_2_2 - A_2_3, -B_3_2 + B_2_3), (A_1_2 - A_2_2 + A_1_3 - A_2_3, B_3_2), (A_2_1 - A_1_1 + A_2_3 - A_1_3, B_3_1)])
    out = fnp.empty((M, N), dtype=a.dtype)
    fnp.copyto(out[0 * h:1 * h, 0 * w:1 * w], m_1 + m_5 + m_6 + m_7 + m_8 + m_12)
    fnp.copyto(out[1 * h:2 * h, 0 * w:1 * w], m_1 + m_6 + m_9 + m_12 + m_15)
    fnp.copyto(out[0 * h:1 * h, 1 * w:2 * w], m_2 + m_5 + m_11 + m_13 + m_14)
    fnp.copyto(out[1 * h:2 * h, 1 * w:2 * w], m_2 - m_8 + m_9 + m_10 + m_11 + m_13)
    fnp.copyto(out[0 * h:1 * h, 2 * w:3 * w], m_3 + m_6 + m_11 + m_13 + m_14)
    fnp.copyto(out[1 * h:2 * h, 2 * w:3 * w], m_4 + m_6 + m_11 + m_12 + m_15)
    return out[:m, :n]



# Dedicated view caches include the complete arena layout in each key.
# No extra arithmetic arena is allocated; old parent views are left intact.
_HKB_CACHES = (_CapLRU(), _CapLRU(), _CapLRU())
for _hkb_cache in _HKB_CACHES:
    _CAP_BY_ID[id(_hkb_cache)] = 8
_HKB_REGION = None

def _hkb_side_p(kind, L, P, rows, cols):
    # entry = (T, chunks, per, None, basis); per[p] = (natural-order level-1 chunk views, pair view) of product p; v2 keeps no intermediates
    cache = _HKB_CACHES[kind]
    key = (L, P, rows, cols, _HKB_REGION)
    found = cache.get(key)
    if found is not None:
        return found
    arena = _SCRATCH[0]
    region = _HKB_REGION
    start = (0, region[0], region[0] + region[1])[kind]
    u = 1 << L
    r, c = rows // u, cols // u
    T = _ipt_view(arena, start, (7,) * L + (P, r, c))
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    chunks = [None, None]
    for j in range(2, L + 1):
        nr = L - j
        chunks.append(fnp.unstack(T[(data,) * nr] if nr else T, axis=nr))
    found = (T, chunks, _rs_level1(kind, L, T, (P, r, c)), None, _rs_basis_views(kind, L, T))
    return _cache_put(cache, key, found)

def _hkb_bb_product(As, Bs, outs, L):
    P = len(As)
    m, k, n = int(As[0].shape[0]), int(As[0].shape[1]), int(Bs[0].shape[1])
    region = _HKB_REGION
    u = 1 << L
    fits = (_AB_ON and region is not None and _SCRATCH is not None and not (m % u or k % u or n % u)
            and 7 ** L * P * (m // u) * (k // u) <= region[0] and 7 ** L * P * (k // u) * (n // u) <= region[1]
            and 7 ** L * P * (m // u) * (n // u) <= region[2]
            and all(tuple(x.shape) == (m, k) and str(x.dtype) == 'float32' for x in As)
            and all(tuple(x.shape) == (k, n) and str(x.dtype) == 'float32' for x in Bs)
            and all(tuple(x.shape) == (m, n) and str(x.dtype) == 'float32' for x in outs))
    if not fits:
        raise RuntimeError('Rectangular batch did not fit the planned arena')
    TA, XA, perA, _ka, basA = _hkb_side_p(0, L, P, m, k)
    TB, XB, perB, _kb, basB = _hkb_side_p(1, L, P, k, n)
    TC, XC, perC, _kc, basC = _hkb_side_p(2, L, P, m, n)
    h = _BH_HOLD.append
    for p in range(P):
        X, A02 = perA[p]
        _rs_load_a(As[p], X, A02, L, m, k)
        Y, B02 = perB[p]
        _rs_load_b(Bs[p], Y, B02, L, k, n)
    for lo, hi in basA:
        h(fnp.add(lo, hi, out=lo))
    for lo, hi in basB:
        h(fnp.subtract(hi, lo, out=lo))
    for p in range(P):
        X, _A02 = perA[p]
        h(fnp.subtract(X[4], X[0], out=X[5])); h(fnp.subtract(X[2], X[5], out=X[3])); h(fnp.subtract(X[6], X[5], out=X[1]))
        Y, _B02 = perB[p]
        h(fnp.add(Y[1], Y[0], out=Y[5])); h(fnp.subtract(Y[5], Y[2], out=Y[6])); h(fnp.subtract(Y[3], Y[5], out=Y[4]))
    for j in range(2, L + 1):
        X = XA[j]
        h(fnp.subtract(X[4], X[0], out=X[5])); h(fnp.subtract(X[2], X[5], out=X[3])); h(fnp.subtract(X[6], X[5], out=X[1]))
        Y = XB[j]
        h(fnp.add(Y[1], Y[0], out=Y[5])); h(fnp.subtract(Y[5], Y[2], out=Y[6])); h(fnp.subtract(Y[3], Y[5], out=Y[4]))
    h(fnp.matmul(TA, TB, out=TC))
    for j in range(L, 1, -1):
        Z = XC[j]
        h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=Z[0])); h(fnp.add(Z[5], Z[4], out=Z[5]))
        h(fnp.add(Z[5], Z[3], out=Z[2])); h(fnp.add(Z[5], Z[1], out=Z[3])); h(fnp.add(Z[4], Z[6], out=Z[1]))
    for lo, hi in basC:
        h(fnp.subtract(hi, lo, out=lo))
    for p in range(P):
        Z, _none = perC[p]
        tl, tr, bl, br = _rs_out_quadrants(outs[p], L, m, n)
        h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=tl)); h(fnp.add(Z[5], Z[4], out=Z[5]))
        h(fnp.add(Z[5], Z[3], out=tr)); h(fnp.add(Z[5], Z[1], out=br)); h(fnp.add(Z[4], Z[6], out=Z[4])); h(fnp.subtract(br, Z[4], out=bl))
    _BH_HOLD.clear()
    return outs


def _hkb6_many(pairs):
    global _HKB_REGION
    m, k, n = int(pairs[0][0].shape[0]), int(pairs[0][0].shape[1]), int(pairs[0][1].shape[1])
    assert m % 64 == k % 64 == n % 64 == 0
    assert all(tuple(a.shape) == (m, k) and tuple(b.shape) == (k, n)
               and str(a.dtype) == str(b.dtype) == 'float32' for a, b in pairs)
    counts = (7**6 * (m//64) * (k//64),
              7**6 * (k//64) * (n//64),
              7**6 * (m//64) * (n//64))
    space = min(sum(_IPT_REGION[6]), int(_SCRATCH[0].shape[0]))
    cap = min(15, space // sum(counts))
    assert cap >= 1 and _AB_ON and _SCRATCH is not None
    _HKB_REGION = tuple(cap*q for q in counts)
    assert sum(_HKB_REGION) <= space
    outs = [fnp.empty((m, n), dtype=fnp.float32) for _ in pairs]
    _TR_LAST[0] = None
    for start in range(0, len(pairs), cap):
        block = pairs[start:start + cap]
        _hkb_bb_product([a for a, b in block], [b for a, b in block], outs[start:start+cap], 6)
    _TR_LAST[0] = None
    return outs

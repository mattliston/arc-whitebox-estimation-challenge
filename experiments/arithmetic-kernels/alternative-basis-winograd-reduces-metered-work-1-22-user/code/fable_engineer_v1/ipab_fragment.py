

# BEGIN FABLE-ENGINEER ALTERNATIVE-BASIS TREE KERNEL (E-58 --ipab, 2026-09-24; rounding-level): the in-place tree kernel with the
# Winograd recursion written in an alternative basis (after Karstadt & Schwartz): A's quadrant a21 is replaced by s3 = a11 - a21,
# B's b12 by t1 = b12 - b11, and C is produced with c12' = c12 - c22 in place of c12, at EVERY recursion level. In that basis each
# level encodes A with 3 subtractions (s2 = a22 - s3, s1 = s2 + a11, s4 = a12 - s2) and B with 3 (t2 = b22 - t1, t3 = t2 - b11,
# t4 = t2 - b21) instead of 4 each, and decodes C with 6 additions (u = m1 + m6, c11 = m1 + m2, v = u + m7, c12' = m3 - m7,
# c21 = v - m4, c22 = v + m5) instead of 7. The basis changes are one operation per level on the ORIGINAL-size operands (A and B
# before the encoding, C after the decoding), so the growing levels pay 12 operations instead of 15. Products by chunk: m1 0, m7 1,
# m2 2, m4 3, m5 4, m6 5, m3 6; data chunks A [0:4] (quadrant (i,j) at i + 2j), B [0:7:2] (at 2i + 4j), C [0:4] (at 2i + j): every
# view is a slice, all four A and B quadrants are plain slots (no in-place encoding step), and the decoding runs in place.
# Certified against A @ B in float64 (fable_engineer_v1/ipab_cert.py); float32 error as the Winograd recursion's.
_IPAB = _CapLRU()
_CAP_BY_ID[id(_IPAB)] = 480
_IPAB_OLD_BB = _bb_product
_IPAB_OLD_W5 = _w5_port_product_body
_IPAB_OLD_CP = _cp_d6_shared
_IPAB_OLD_PRODUCT = _ipt_product


def _ipab_side(kind, L, P, rows, cols):
    # kind 0 = A, 1 = B, 2 = C; P = None for one product, else the product count of a batched tree (7,)*L + (P, r, c)
    key = (kind, L, P, rows, cols)
    found = _IPAB.get(key)
    if found is not None:
        return found
    arena = _SCRATCH[0]
    region = _IPT_REGION[L]
    start = (0, region[0], region[0] + region[1])[kind]
    u = 1 << L
    r, c = rows // u, cols // u
    tail = (r, c) if P is None else (P, r, c)
    T = _ipt_view(arena, start, (7,) * L + tail)
    D = slice(0, 7, 2) if kind == 1 else slice(0, 4)
    keep = [T]
    chunks = [None, None]
    for j in range(2, L + 1):                    # level j: levels L..j+1 restricted to the data chunks, level j's chunk axis split
        nr = L - j
        V = T[(D,) * nr] if nr else T
        keep.append(V)
        chunks.append(fnp.unstack(V, axis=nr))
    R = T[(D,) * (L - 1)]                        # level 1 (root) chunks, deeper levels restricted to the data
    keep.append(R)
    chunks[1] = fnp.unstack(R, axis=L - 1)
    basis = [None, None]
    if kind < 2:                                 # A: a21 (1) -> a11 (0) - a21 ; B: b12 (4) -> b12 - b11 (0), at levels 2..L on the data
        pa, pb = (0, 1) if kind == 0 else (0, 4)
        for j in range(2, L + 1):
            nr = L - j
            X = T[(D,) * nr + (slice(None),) + (D,) * (j - 1)]
            keep.append(X)
            basis.append((X[(slice(None),) * nr + (pa,)], X[(slice(None),) * nr + (pb,)]))
    nr = L - 1
    V1 = fnp.reshape(T[(D,) * nr], (2, 2) * nr + (7,) + tail)
    keep.append(V1)
    if kind < 2:                                 # A and B data chunks are (j, i) pairs: swap each pair to (i, j)
        V1 = fnp.transpose(V1, [x for p in range(nr) for x in (2 * p + 1, 2 * p)] + list(range(2 * nr, 2 * nr + 1 + len(tail))))
        keep.append(V1)
    if P is None:
        per = [fnp.unstack(V1, axis=2 * nr)]
    else:
        per = [fnp.unstack(Vp, axis=2 * nr) for Vp in fnp.unstack(V1, axis=2 * nr + 1)]
    if kind < 2:
        pairs = [V1[(slice(None),) * (2 * nr) + (slice(0, 3, 2),) + ((() if P is None else (p,)))] for p in range(1 if P is None else P)]
    else:
        pairs = None
    found = dict(T=T, chunks=chunks, basis=basis, per=per, pairs=pairs, keep=keep)
    return _cache_put(_IPAB, key, found)


def _ipab_load_a(a, S, p, L, m, k):
    # level-1 load of A into the root data chunks (a11 0, s3 = a11 - a21 at 1, a12 2, a22 3), from the input's bit-split view
    h = _BH_HOLD.append
    u = 1 << L
    Aq = _ipt_bits(a, L, m // u, k // u)
    A0, A1 = fnp.unstack(Aq, axis=0)
    (a11, a12), (a21, a22) = fnp.unstack(A0, axis=0), fnp.unstack(A1, axis=0)
    h((A0, A1, a11, a12, a21, a22))
    X = S['per'][p]
    h(fnp.subtract(a11, a21, out=X[1])); h(fnp.copyto(X[3], a22))
    first = fnp.moveaxis(A0, 0, 2 * (L - 1)); h(first); h(fnp.copyto(S['pairs'][p], first))


def _ipab_load_b(b, S, p, L, k, n):
    # level-1 load of B into the root data chunks (b11 0, b21 2, t1 = b12 - b11 at 4, b22 6)
    h = _BH_HOLD.append
    u = 1 << L
    Bq = _ipt_bits(b, L, k // u, n // u)
    Bj0, Bj1 = fnp.unstack(Bq, axis=1)
    (b11, b21), (b12, b22) = fnp.unstack(Bj0, axis=0), fnp.unstack(Bj1, axis=0)
    h((Bj0, Bj1, b11, b21, b12, b22))
    Y = S['per'][p]
    h(fnp.subtract(b12, b11, out=Y[4])); h(fnp.copyto(Y[6], b22))
    first = fnp.moveaxis(Bj0, 0, 2 * (L - 1)); h(first); h(fnp.copyto(S['pairs'][p], first))


def _ipab_encode(S, kind, L):
    # basis change at levels 2..L on the data, then the 3-operation encoding at every level
    h = _BH_HOLD.append
    for j in range(2, L + 1):
        x0, x1 = S['basis'][j]
        if kind == 0:
            h(fnp.subtract(x0, x1, out=x1))          # s3 = a11 - a21
        else:
            h(fnp.subtract(x1, x0, out=x1))          # t1 = b12 - b11
    for j in range(1, L + 1):
        X = S['chunks'][j]
        if kind == 0:
            h(fnp.subtract(X[3], X[1], out=X[5])); h(fnp.add(X[5], X[0], out=X[4])); h(fnp.subtract(X[2], X[5], out=X[6]))
        else:
            h(fnp.subtract(X[6], X[4], out=X[5])); h(fnp.subtract(X[5], X[0], out=X[1])); h(fnp.subtract(X[5], X[2], out=X[3]))


def _ipab_decode(S, L):
    # decoding at levels L..2 in place (c11 0, c12' 1, c21 2, c22 3)
    h = _BH_HOLD.append
    for j in range(L, 1, -1):
        Z = S['chunks'][j]
        h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=Z[0])); h(fnp.add(Z[5], Z[1], out=Z[5]))
        h(fnp.subtract(Z[6], Z[1], out=Z[1])); h(fnp.subtract(Z[5], Z[3], out=Z[2])); h(fnp.add(Z[5], Z[4], out=Z[3]))


def _ipab_finish(Z, out, L, m, n):
    # level-1 decoding into the output, then c12 = c12' + c22 at every level on the output's bit-split view
    h = _BH_HOLD.append
    u = 1 << L
    Cq = _ipt_bits(out, L, m // u, n // u)
    C0, C1 = fnp.unstack(Cq, axis=0)
    (tl, tr), (bl, br) = fnp.unstack(C0, axis=0), fnp.unstack(C1, axis=0)
    h((C0, C1, tl, tr, bl, br))
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=tl)); h(fnp.add(Z[5], Z[1], out=Z[5]))
    h(fnp.subtract(Z[6], Z[1], out=tr)); h(fnp.subtract(Z[5], Z[3], out=bl)); h(fnp.add(Z[5], Z[4], out=br))
    h(fnp.add(tr, br, out=tr))
    for j in range(2, L + 1):
        ax = 2 + 2 * (L - j)
        top = Cq[(slice(None),) * ax + (0, 1)]; bot = Cq[(slice(None),) * ax + (1, 1)]
        h((top, bot)); h(fnp.add(top, bot, out=top))


def _ipab_product(a, b, out, L):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    SA = _ipab_side(0, L, None, m, k); SB = _ipab_side(1, L, None, k, n); SC = _ipab_side(2, L, None, m, n)
    _ipab_load_a(a, SA, 0, L, m, k); _ipab_encode(SA, 0, L)
    _ipab_load_b(b, SB, 0, L, k, n); _ipab_encode(SB, 1, L)
    _BH_HOLD.append(fnp.matmul(SA['T'], SB['T'], out=SC['T']))
    _ipab_decode(SC, L)
    _ipab_finish(SC['per'][0], out, L, m, n)
    _BH_HOLD.clear()
    return out


def _ipt_product(a, b, out, L):
    return _ipab_product(a, b, out, L)


def _cp_d6_shared(a, b, out):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if not (4096 < n and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _ipt_fits(6, m, k, 4096) and _ipt_fits(6, m, k, n - 4096)):
        return _IPAB_OLD_CP(a, b, out)
    L = 6
    SA = _ipab_side(0, L, None, m, k)
    _ipab_load_a(a, SA, 0, L, m, k); _ipab_encode(SA, 0, L)
    for j0, j1 in ((0, 4096), (4096, n)):
        bj, oj = b[:, j0:j1], out[:, j0:j1]
        _BH_HOLD.append((bj, oj))
        nc = j1 - j0
        SB = _ipab_side(1, L, None, k, nc); SC = _ipab_side(2, L, None, m, nc)
        _ipab_load_b(bj, SB, 0, L, k, nc); _ipab_encode(SB, 1, L)
        _BH_HOLD.append(fnp.matmul(SA['T'], SB['T'], out=SC['T']))
        _ipab_decode(SC, L)
        _ipab_finish(SC['per'][0], oj, L, m, nc)
    _BH_HOLD.clear()
    return out


def _bb_product(As, Bs, outs, L):
    P = len(As)
    m, k, n = int(As[0].shape[0]), int(As[0].shape[1]), int(Bs[0].shape[1])
    region = _IPT_REGION.get(L)
    u = 1 << L
    fits = (region is not None and _SCRATCH is not None and not (m % u or k % u or n % u)
            and 7 ** L * P * (m // u) * (k // u) <= region[0] and 7 ** L * P * (k // u) * (n // u) <= region[1]
            and 7 ** L * P * (m // u) * (n // u) <= region[2]
            and all(tuple(x.shape) == (m, k) and str(x.dtype) == 'float32' for x in As)
            and all(tuple(x.shape) == (k, n) and str(x.dtype) == 'float32' for x in Bs)
            and all(tuple(x.shape) == (m, n) and str(x.dtype) == 'float32' for x in outs))
    if not fits:
        return _IPAB_OLD_BB(As, Bs, outs, L)
    SA = _ipab_side(0, L, P, m, k); SB = _ipab_side(1, L, P, k, n); SC = _ipab_side(2, L, P, m, n)
    for p in range(P):
        _ipab_load_a(As[p], SA, p, L, m, k)
        _ipab_load_b(Bs[p], SB, p, L, k, n)
    _ipab_encode(SA, 0, L); _ipab_encode(SB, 1, L)
    _BH_HOLD.append(fnp.matmul(SA['T'], SB['T'], out=SC['T']))
    _ipab_decode(SC, L)
    for p in range(P):
        _ipab_finish(SC['per'][p], outs[p], L, m, n)
    _BH_HOLD.clear()
    return outs


def _w5_port_product_body(a, b, out, views, levels, cache):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    L = levels
    if not (_ipt_ok(a, b, out, L) and tuple(cache._shape) == (7,) * L + (m >> L, k >> L)):
        return _IPAB_OLD_W5(a, b, out, views, levels, cache)
    if cache._leaf is None:
        SA = _ipab_side(0, L, None, m, k)
        _ipab_load_a(a, SA, 0, L, m, k); _ipab_encode(SA, 0, L)
        cached = fnp.empty(cache._shape, dtype=fnp.float32)
        cache._own_leaf(cached, cached)
        _BH_HOLD.append(fnp.copyto(cached, SA['T']))
    SB = _ipab_side(1, L, None, k, n); SC = _ipab_side(2, L, None, m, n)
    _ipab_load_b(b, SB, 0, L, k, n); _ipab_encode(SB, 1, L)
    _BH_HOLD.append(fnp.matmul(cache._leaf, SB['T'], out=SC['T']))
    _ipab_decode(SC, L)
    _ipab_finish(SC['per'][0], out, L, m, n)
    _BH_HOLD.clear()
    return out
# END FABLE-ENGINEER ALTERNATIVE-BASIS TREE KERNEL

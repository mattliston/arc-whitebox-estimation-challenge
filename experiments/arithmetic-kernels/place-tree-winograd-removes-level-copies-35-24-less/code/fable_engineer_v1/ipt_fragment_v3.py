

# BEGIN FABLE-ENGINEER IN-PLACE TREE WINOGRAD KERNEL (E-53 --ipt v3, 2026-09-24; bit-exact; v3 = v2 with fewer requests: chunk views by one
# unstack per level, binding temporaries held, per-call quadrants by unstack, caps sized to two nets of shapes): every Winograd product that went through
# _packed_product_cached, colpack's shared-A pair (_cp_d6_shared) or the depth-5 read (_packed_read5) now runs on ONE tree array per
# operand instead of one array per level. Tree shape (7,)*L + leaf with the DEEPEST level outermost: an op at level j is 4**(L-j)
# contiguous runs of 7**(j-1) leaf blocks. Every block sits in a room of 7 chunks and its four quadrants are four of them (A: quadrant
# (i,j) at chunk 4i+2j, B at 2i+j, C at i+2j), so the next level's plain slots are already in place: no level copies below level 1,
# the other slots are written into the free chunks (one of them in place over the quadrant it replaces), the leaf is ONE batched matmul
# over all 7**L chunk pairs, and the unwind runs in place. Winograd products by chunk: m1 0, m5 1, m2 2, m3 3, m7 4, m6 5, m4 6 (the
# old kernel's p0..p6 = m1..m7). Same formulas on the same operands as the old kernels, in the same operand order: bit-identical.
# Trees live in fixed per-depth regions of the arena (A, then B, then C), bound once per shape (A keyed by (L, m, k), B by (L, k, n),
# C by (L, m, n)); products that do not fit fall back to the old kernels (e.g. engineer2's read6). Per call the inputs and the output
# are reshaped into bit-split views (billed at numel); nothing else new is billed.
_IPT_REGION = {2: (1 << 20, 1 << 22, 1 << 22), 3: (5619712, 22478848, 22478848), 4: (78675968, 78675968, 78675968),
               5: (137682944, 137682944, 68841472), 6: (30118144, 120472576, 120472576)}
_IPT_A = _CapLRU(); _IPT_B = _CapLRU(); _IPT_C = _CapLRU()
_CAP_BY_ID[id(_IPT_A)] = 160; _CAP_BY_ID[id(_IPT_B)] = 160; _CAP_BY_ID[id(_IPT_C)] = 160
_IPT_OLD_PPC = _packed_product_cached
_IPT_OLD_CP = _cp_d6_shared
_IPT_OLD_READ5 = _packed_read5


def _ipt_view(arena, start, shape):
    count = 1
    for x in shape:
        count *= x
    piece = arena[start:start + count]
    view = fnp.reshape(piece, shape)
    _BH_HOLD.append((piece, view))
    return view


def _ipt_fits(L, m, k, n):
    region = _IPT_REGION.get(L)
    if region is None or _SCRATCH is None:
        return False
    u = 1 << L
    if m % u or k % u or n % u or m <= 0 or k <= 0 or n <= 0:
        return False
    blocks = 7 ** L
    ra, ka, nb = m // u, k // u, n // u
    return (blocks * ra * ka <= region[0] and blocks * ka * nb <= region[1] and blocks * ra * nb <= region[2]
            and region[0] + region[1] + region[2] <= int(_SCRATCH[0].shape[0]))


def _ipt_side(kind, L, rows, cols):
    # kind 0 = A (data chunks 0,2,4,6), 1 = B (0..3), 2 = C (0..3); returns (tree, per-level chunk views, level-1 pair slot view)
    cache = (_IPT_A, _IPT_B, _IPT_C)[kind]
    key = (L, rows, cols)
    found = cache.get(key)
    if found is not None:
        return found
    arena = _SCRATCH[0]
    region = _IPT_REGION[L]
    start = (0, region[0], region[0] + region[1])[kind]
    u = 1 << L
    r, c = rows // u, cols // u
    T = _ipt_view(arena, start, (7,) * L + (r, c))
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    chunks = [None, None]
    keep = [T]
    for j in range(2, L + 1):
        nr = L - j
        V = T[(data,) * nr] if nr else T
        keep.append(V)
        chunks.append(fnp.unstack(V, axis=nr))
    nr = L - 1
    R = T[(data,) * nr]
    V1 = fnp.reshape(R, (2, 2) * nr + (7, r, c))
    keep += [R, V1]
    if kind == 2:   # C's data chunk is i + 2j: swap each pair to (i, j)
        V1 = fnp.transpose(V1, [x for p in range(nr) for x in (2 * p + 1, 2 * p)] + [2 * nr, 2 * nr + 1, 2 * nr + 2])
        keep.append(V1)
    chunks[1] = fnp.unstack(V1, axis=2 * nr)
    pair = V1[(slice(None),) * (2 * nr) + (slice(0, 3, 2),)] if kind < 2 else None
    found = (T, chunks, pair, keep)
    return _cache_put(cache, key, found)


def _ipt_bits(X, L, r, c):
    # (2^L r, 2^L c) matrix -> view with axes (i1, j1, iL, jL, ..., i2, j2, row, col)
    V = fnp.reshape(X, (2,) * L + (r,) + (2,) * L + (c,))
    view = fnp.transpose(V, [0, L + 1] + [x for p in range(L, 1, -1) for x in (p - 1, L + p)] + [L, 2 * L + 1])
    _BH_HOLD.append((V, view))
    return view


def _ipt_forward_a(a, L, m, k):
    TA, X1, A02, _keep = _ipt_side(0, L, m, k)
    h = _BH_HOLD.append
    u = 1 << L
    Aq = _ipt_bits(a, L, m // u, k // u)
    A0, A1 = fnp.unstack(Aq, axis=0)
    (a11, a12), (a21, a22) = fnp.unstack(A0, axis=0), fnp.unstack(A1, axis=0)
    h((A0, A1, a11, a12, a21, a22))
    X = X1[1]
    h(fnp.add(a21, a22, out=X[1])); h(fnp.subtract(X[1], a11, out=X[5])); h(fnp.subtract(a11, a21, out=X[4])); h(fnp.subtract(a12, X[5], out=X[3]))
    first = fnp.moveaxis(A0, 0, 2 * (L - 1))
    h(first); h(fnp.copyto(A02, first)); h(fnp.copyto(X[6], a22))
    for j in range(2, L + 1):
        X = X1[j]
        h(fnp.add(X[4], X[6], out=X[1])); h(fnp.subtract(X[1], X[0], out=X[5])); h(fnp.subtract(X[0], X[4], out=X[4])); h(fnp.subtract(X[2], X[5], out=X[3]))
    return TA


def _ipt_forward_b(b, L, k, n):
    TB, Y1, B02, _keep = _ipt_side(1, L, k, n)
    h = _BH_HOLD.append
    u = 1 << L
    Bq = _ipt_bits(b, L, k // u, n // u)
    Bj0, Bj1 = fnp.unstack(Bq, axis=1)
    (b11, b21), (b12, b22) = fnp.unstack(Bj0, axis=0), fnp.unstack(Bj1, axis=0)
    h((Bj0, Bj1, b11, b12, b21, b22))
    Y = Y1[1]
    h(fnp.subtract(b12, b11, out=Y[1])); h(fnp.subtract(b22, Y[1], out=Y[5])); h(fnp.subtract(b22, b12, out=Y[4])); h(fnp.subtract(Y[5], b21, out=Y[6]))
    first = fnp.moveaxis(Bj0, 0, 2 * (L - 1))
    h(first); h(fnp.copyto(B02, first)); h(fnp.copyto(Y[3], b22))
    for j in range(2, L + 1):
        Y = Y1[j]
        h(fnp.subtract(Y[3], Y[1], out=Y[4])); h(fnp.subtract(Y[1], Y[0], out=Y[1])); h(fnp.subtract(Y[3], Y[1], out=Y[5])); h(fnp.subtract(Y[5], Y[2], out=Y[6]))
    return TB


def _ipt_leaf_unwind(TA, TB, out, L, m, n):
    TC, Z1, _pair, _keep = _ipt_side(2, L, m, n)
    h = _BH_HOLD.append
    h(fnp.matmul(TA, TB, out=TC))
    for j in range(L, 1, -1):
        Z = Z1[j]
        h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=Z[0])); h(fnp.add(Z[5], Z[4], out=Z[4])); h(fnp.add(Z[5], Z[1], out=Z[5]))
        h(fnp.add(Z[5], Z[3], out=Z[2])); h(fnp.add(Z[4], Z[1], out=Z[3])); h(fnp.subtract(Z[4], Z[6], out=Z[1]))
    u = 1 << L
    Cq = _ipt_bits(out, L, m // u, n // u)
    Z = Z1[1]
    C0, C1 = fnp.unstack(Cq, axis=0)
    (tl, tr), (bl, br) = fnp.unstack(C0, axis=0), fnp.unstack(C1, axis=0)
    h((C0, C1, tl, tr, bl, br))
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=tl)); h(fnp.add(Z[5], Z[4], out=Z[4])); h(fnp.add(Z[5], Z[1], out=Z[5]))
    h(fnp.add(Z[5], Z[3], out=tr)); h(fnp.add(Z[4], Z[1], out=br)); h(fnp.subtract(Z[4], Z[6], out=bl))
    return out


def _ipt_product(a, b, out, L):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    TA = _ipt_forward_a(a, L, m, k)
    TB = _ipt_forward_b(b, L, k, n)
    _ipt_leaf_unwind(TA, TB, out, L, m, n)
    _BH_HOLD.clear()
    return out


def _ipt_ok(a, b, out, L):
    return (L >= 2 and len(a.shape) == len(b.shape) == len(out.shape) == 2 and int(b.shape[0]) == int(a.shape[1])
            and tuple(out.shape) == (int(a.shape[0]), int(b.shape[1]))
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _ipt_fits(L, int(a.shape[0]), int(a.shape[1]), int(b.shape[1])))


def _packed_product_cached(a, b, out, views, levels):
    if _ipt_ok(a, b, out, levels):
        return _ipt_product(a, b, out, levels)
    return _IPT_OLD_PPC(a, b, out, views, levels)


def _cp_d6_shared(a, b, out):
    # colpack's two six-level packets with ONE A transform: the A tree is formed once and read by both leaves
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if not (4096 < n and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _ipt_fits(6, m, k, 4096) and _ipt_fits(6, m, k, n - 4096)):
        return _IPT_OLD_CP(a, b, out)
    TA = _ipt_forward_a(a, 6, m, k)
    for j0, j1 in ((0, 4096), (4096, n)):
        bj, oj = b[:, j0:j1], out[:, j0:j1]
        _BH_HOLD.append((bj, oj))
        TB = _ipt_forward_b(bj, 6, k, j1 - j0)
        _ipt_leaf_unwind(TA, TB, oj, 6, m, j1 - j0)
    _BH_HOLD.clear()
    return out


def _packed_read5(a, b, out, views):
    if _ipt_ok(a, b, out, 5):
        return _ipt_product(a, b, out, 5)
    return _IPT_OLD_READ5(a, b, out, views)
# END FABLE-ENGINEER IN-PLACE TREE WINOGRAD KERNEL


# BEGIN FABLE-ENGINEER IN-PLACE TREE KERNEL: BATCHED PRODUCTS AND THE W5 PORT (E-53 --ipt v2, 2026-09-24; bit-exact)
# (1) _bb_product (the layer's P = 2 independent 1024^3 depth-5 products): trees with a product axis before the leaf dims, shape
#     (7,)*L + (P, r, c). Level 1 reads each product's own operands and the level-1 unwind writes each product's own output; levels
#     2..L, the leaf matmul and unwind levels L..2 run once for all P products. Same per-element arithmetic as one IPT product each.
# (2) _w5_port_product_body: the port's cached A leaf is now the A tree (same shape (7,)*5 + (m/32, k/32)); the creator forms and
#     copies it, later sites skip the A forward exactly as before.
_IPT_OLD_BB = _bb_product
_IPT_OLD_W5 = _w5_port_product_body
_IPT_BA = _CapLRU(); _IPT_BB = _CapLRU(); _IPT_BC = _CapLRU()
_CAP_BY_ID[id(_IPT_BA)] = 8; _CAP_BY_ID[id(_IPT_BB)] = 8; _CAP_BY_ID[id(_IPT_BC)] = 8


def _ipt_side_p(kind, L, P, rows, cols):
    cache = (_IPT_BA, _IPT_BB, _IPT_BC)[kind]
    key = (L, P, rows, cols)
    found = cache.get(key)
    if found is not None:
        return found
    arena = _SCRATCH[0]
    region = _IPT_REGION[L]
    start = (0, region[0], region[0] + region[1])[kind]
    u = 1 << L
    r, c = rows // u, cols // u
    T = _ipt_view(arena, start, (7,) * L + (P, r, c))
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    chunks = [None, None]
    keep = [T]
    for j in range(2, L + 1):
        nr = L - j
        V = T[(data,) * nr] if nr else T
        keep.append(V)
        chunks.append(fnp.unstack(V, axis=nr))
    nr = L - 1
    R = T[(data,) * nr]
    V1 = fnp.reshape(R, (2, 2) * nr + (7, P, r, c))
    keep += [R, V1]
    if kind == 2:
        V1 = fnp.transpose(V1, [x for p in range(nr) for x in (2 * p + 1, 2 * p)] + [2 * nr, 2 * nr + 1, 2 * nr + 2, 2 * nr + 3])
        keep.append(V1)
    per = []
    for p, Vp in enumerate(fnp.unstack(V1, axis=2 * nr + 1)):
        keep.append(Vp)
        per.append((fnp.unstack(Vp, axis=2 * nr), Vp[(slice(None),) * (2 * nr) + (slice(0, 3, 2),)] if kind < 2 else None))
    found = (T, chunks, per, keep)
    return _cache_put(cache, key, found)


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
        return _IPT_OLD_BB(As, Bs, outs, L)
    TA, XA, perA, _ka = _ipt_side_p(0, L, P, m, k)
    TB, XB, perB, _kb = _ipt_side_p(1, L, P, k, n)
    TC, XC, perC, _kc = _ipt_side_p(2, L, P, m, n)
    h = _BH_HOLD.append
    for p in range(P):
        X, A02 = perA[p]
        Aq = _ipt_bits(As[p], L, m // u, k // u)
        A0, A1 = fnp.unstack(Aq, axis=0)
        (a11, a12), (a21, a22) = fnp.unstack(A0, axis=0), fnp.unstack(A1, axis=0)
        h((A0, A1, a11, a12, a21, a22))
        h(fnp.add(a21, a22, out=X[1])); h(fnp.subtract(X[1], a11, out=X[5])); h(fnp.subtract(a11, a21, out=X[4])); h(fnp.subtract(a12, X[5], out=X[3]))
        first = fnp.moveaxis(A0, 0, 2 * (L - 1))
        h(first); h(fnp.copyto(A02, first)); h(fnp.copyto(X[6], a22))
        Y, B02 = perB[p]
        Bq = _ipt_bits(Bs[p], L, k // u, n // u)
        Bj0, Bj1 = fnp.unstack(Bq, axis=1)
        (b11, b21), (b12, b22) = fnp.unstack(Bj0, axis=0), fnp.unstack(Bj1, axis=0)
        h((Bj0, Bj1, b11, b12, b21, b22))
        h(fnp.subtract(b12, b11, out=Y[1])); h(fnp.subtract(b22, Y[1], out=Y[5])); h(fnp.subtract(b22, b12, out=Y[4])); h(fnp.subtract(Y[5], b21, out=Y[6]))
        first = fnp.moveaxis(Bj0, 0, 2 * (L - 1))
        h(first); h(fnp.copyto(B02, first)); h(fnp.copyto(Y[3], b22))
    for j in range(2, L + 1):
        X = XA[j]
        h(fnp.add(X[4], X[6], out=X[1])); h(fnp.subtract(X[1], X[0], out=X[5])); h(fnp.subtract(X[0], X[4], out=X[4])); h(fnp.subtract(X[2], X[5], out=X[3]))
        Y = XB[j]
        h(fnp.subtract(Y[3], Y[1], out=Y[4])); h(fnp.subtract(Y[1], Y[0], out=Y[1])); h(fnp.subtract(Y[3], Y[1], out=Y[5])); h(fnp.subtract(Y[5], Y[2], out=Y[6]))
    h(fnp.matmul(TA, TB, out=TC))
    for j in range(L, 1, -1):
        Z = XC[j]
        h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=Z[0])); h(fnp.add(Z[5], Z[4], out=Z[4])); h(fnp.add(Z[5], Z[1], out=Z[5]))
        h(fnp.add(Z[5], Z[3], out=Z[2])); h(fnp.add(Z[4], Z[1], out=Z[3])); h(fnp.subtract(Z[4], Z[6], out=Z[1]))
    for p in range(P):
        Z, _none = perC[p]
        Cq = _ipt_bits(outs[p], L, m // u, n // u)
        C0, C1 = fnp.unstack(Cq, axis=0)
        (tl, tr), (bl, br) = fnp.unstack(C0, axis=0), fnp.unstack(C1, axis=0)
        h((C0, C1, tl, tr, bl, br))
        h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=tl)); h(fnp.add(Z[5], Z[4], out=Z[4])); h(fnp.add(Z[5], Z[1], out=Z[5]))
        h(fnp.add(Z[5], Z[3], out=tr)); h(fnp.add(Z[4], Z[1], out=br)); h(fnp.subtract(Z[4], Z[6], out=bl))
    _BH_HOLD.clear()
    return outs


def _w5_port_product_body(a, b, out, views, levels, cache):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    if not (_ipt_ok(a, b, out, levels) and tuple(cache._shape) == (7,) * levels + (m >> levels, k >> levels)):
        return _IPT_OLD_W5(a, b, out, views, levels, cache)
    if cache._leaf is None:
        TA = _ipt_forward_a(a, levels, m, k)
        cached = fnp.empty(cache._shape, dtype=fnp.float32)
        cache._own_leaf(cached, cached)
        _BH_HOLD.append(fnp.copyto(cached, TA))
    TB = _ipt_forward_b(b, levels, k, n)
    _ipt_leaf_unwind(cache._leaf, TB, out, levels, m, n)
    _BH_HOLD.clear()
    return out
# END FABLE-ENGINEER IN-PLACE TREE KERNEL: BATCHED PRODUCTS AND THE W5 PORT



# BEGIN FABLE-ENGINEER GROUPED-REQUEST WINOGRAD KERNEL (E-15 v7c, 2026-09-22)
# Same Winograd recursion, same operand pairs and same operation order per element as the bank's _packed_product_cached (every
# product bit-identical). Requests: each level runs 4 A-side, 4 B-side and 4 backward ufunc requests (bank: 6/5/7), each ufunc
# applied to a strided PAIR of quadrants/slots. Every arena region is viewed once as a fully blocked array
# (7,)*l + (2,)*(L-l) + (mL,) + (2,)*(L-l) + (kL,) (newest slot axis first), so quadrant/pair/slot views are cached free views
# (built with unstack where possible). Slot order is permuted (new 0..6 = old 3,0,1,2,4,5,6) so each side's copies are one stack.
# A and B are transformed level by level (the arena layouts reuse regions across levels). The backward keeps s1/s2 in the u-scratch
# (s0 in the bl corner), so no ufunc reads and writes overlapping memory. The top operands are copied into fixed blocked buffers
# (one copyto each, billed m*k, k*n, m*n). The LAST level materializes no copies where possible: the leaf products read the
# previous level's quadrants directly; the leaf products are relocated to the free arena tail when it fits (square and D4 push
# layouts), which removes the region overlap that otherwise forces a side to keep its copies. The W5 port body uses the same kernel.
_GP_A_FRAMES = {}; _GP_B_FRAMES = {}; _GP_O_FRAMES = {}
_GP_TOP = None
_GP_TOP_ELEMENTS = 1024 * 8192
_GP_EXTENT = {}
_GP_HOLD = []   # every handle a kernel call creates and does not keep (out= aliases, temporaries), released together at the
                # end of the call so the client frees them in ONE batched round trip instead of one hidden trip per op


def _gp_top_buffers():
    global _GP_TOP
    if _GP_TOP is None:
        _GP_TOP = (fnp.empty((_GP_TOP_ELEMENTS,), dtype=fnp.float32), fnp.empty((_GP_TOP_ELEMENTS,), dtype=fnp.float32),
                   fnp.empty((_GP_TOP_ELEMENTS,), dtype=fnp.float32))
    return _GP_TOP


def _gp_region(arena, start, stop, lead, L_rest, rows, cols):
    exact = tuple(lead) + (2,) * L_rest + (rows,) + (2,) * L_rest + (cols,)
    numel = 1
    for x in exact:
        numel *= x
    assert start + numel <= stop, (start, stop, exact)
    flat = arena[start:start + numel]
    _GP_HOLD.append(flat)
    return fnp.reshape(flat, exact)


def _gp_top(buffer, L, rows, cols):
    flat = buffer[:(rows << L) * (cols << L)]
    _GP_HOLD.append(flat)
    return (fnp.reshape(flat, (rows << L, cols << L)),
            fnp.reshape(flat, (2,) * L + (rows,) + (2,) * L + (cols,)))


_GP_TOPS = {}


def _gp_top_cached(index, L, rows, cols):
    # top-buffer views shared by every layout: (2-D view for the copy, blocked view)
    key = (index, L, rows, cols)
    found = _GP_TOPS.get(key)
    if found is None:
        found = _gp_put(_GP_TOPS, key, _gp_top(_gp_top_buffers()[index], L, rows, cols))
    return found


def _gp_q(l, L, i, j):
    return (slice(None),) * (l - 1) + (i,) + (slice(None),) * (L - l + 1) + (j,)


def _gp_front(view, axis):
    return view if axis == 0 else fnp.moveaxis(view, axis, 0)


def _gp_put(cache, key, value):
    if len(cache) >= 4096:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def _gp_leaf_place(views, L, mL, nL):
    # the leaf products' (start, stop): relocated to the free arena tail when it fits, else the layout's own leaf region
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    tail = _GP_EXTENT.get(id(views))
    if tail is None:
        tail = _GP_EXTENT[id(views)] = max(stop for _start, stop, _shape in specs)
    numel = 7 ** L * mL * nL
    if tail + numel <= int(arena.shape[0]):
        return tail, tail + numel
    return specs[4 * L][0], specs[4 * L][1]


def _gp_a_frame(views, L, m, k):
    key = (id(views), L, m, k)
    found = _GP_A_FRAMES.get(key)
    if found is not None:
        return found
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    mL, kL = m >> L, k >> L
    top2, X = _gp_top_cached(0, L, mL, kL)
    levels = []
    for l in range(1, L + 1):
        start, stop, _shape = specs[2 * (l - 1)]
        Y = _gp_region(arena, start, stop, (7,) * l, L - l, mL, kL)
        ys = fnp.unstack(Y, axis=0)
        p0, p1 = X[_gp_q(l, L, slice(None), 0)], X[_gp_q(l, L, slice(None), 1)]
        _GP_HOLD.extend((p0, p1))
        a11, a21 = fnp.unstack(p0, axis=l - 1)
        a12, a22 = fnp.unstack(p1, axis=l - 1)
        a_i0 = _gp_front(p0, l - 1)
        full = ((a22, a11, a12), a21, a_i0, Y[0:3], ys[4], Y[4::-3], Y[5:7], ys[5], ys[3])
        if l < L:
            levels.append(full)
        else:
            a_j0 = _gp_front(X[_gp_q(l, L, 0, slice(None))], L)
            last = (full, (a11, a12, a21, a22, ys[3], ys[4], ys[5], ys[6]), (a22, a_j0, ys[3], Y[4:7], Y[3:7]), (ys[0], Y[1:3], ys[3], Y[4:7]), Y)
            used = (specs[2 * (L - 2)][0], specs[2 * (L - 2)][0] + 7 ** (L - 1) * 4 * mL * kL) if L > 1 else (-1, -1)
        X = Y
    return _gp_put(_GP_A_FRAMES, key, (top2, tuple(levels), last, used))


def _gp_b_frame(views, L, k, n):
    key = (id(views), L, k, n)
    found = _GP_B_FRAMES.get(key)
    if found is not None:
        return found
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    kL, nL = k >> L, n >> L
    top2, X = _gp_top_cached(1, L, kL, nL)
    levels = []
    for l in range(1, L + 1):
        start, stop, _shape = specs[2 * (l - 1) + 1]
        Z = _gp_region(arena, start, stop, (7,) * l, L - l, kL, nL)
        zs = fnp.unstack(Z, axis=0)
        p0, p1 = X[_gp_q(l, L, slice(None), 0)], X[_gp_q(l, L, slice(None), 1)]
        _GP_HOLD.extend((p0, p1))
        b11, b21 = fnp.unstack(p0, axis=l - 1)
        _b12, b22 = fnp.unstack(p1, axis=l - 1)
        b_i1 = _gp_front(p1, l - 1)
        b_j0 = _gp_front(X[_gp_q(l, L, 0, slice(None))], L)
        full = ((b11, b21, b22), b_i1, b_j0, b22, b21, Z[1:4], Z[4:7:2], zs[4], zs[5], zs[0])
        if l < L:
            levels.append(full)
        else:
            b_i0 = _gp_front(p0, l - 1)
            last = (full, (zs[0], b_i0, b22, Z[4:7]), (zs[0], Z[1:3], Z[3:7]), Z)
            used = (specs[2 * (L - 2) + 1][0], specs[2 * (L - 2) + 1][0] + 7 ** (L - 1) * 4 * kL * nL) if L > 1 else (-1, -1)
        X = Z
    return _gp_put(_GP_B_FRAMES, key, (top2, tuple(levels), last, used))


def _gp_o_frame(views, L, m, n):
    key = (id(views), L, m, n)
    found = _GP_O_FRAMES.get(key)
    if found is not None:
        return found
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    assert len(specs) == 6 * L, (len(specs), L)
    mL, nL = m >> L, n >> L
    inter_specs = specs[4 * L + 1:4 * L + L]; u_specs = specs[5 * L:6 * L]
    top2, D0 = _gp_top_cached(2, L, mL, nL)
    lstart, lstop = _gp_leaf_place(views, L, mL, nL)
    Q = _gp_region(arena, lstart, lstop, (7,) * L, 0, mL, nL)
    qs = fnp.unstack(Q, axis=0)
    leaf = (Q, qs[0], Q[1:3], qs[3], Q[4:7], Q[3:7], (lstart, lstop))
    backward = []
    for unwind, l in enumerate(range(L, 0, -1)):
        if l > 1:
            start, stop, _shape = inter_specs[unwind]
            D = _gp_region(arena, start, stop, (7,) * (l - 1), L - l + 1, mL, nL)
        else:
            D = D0
        start, stop, _shape = u_specs[unwind]
        S = _gp_region(arena, start, stop, (3,) + (7,) * (l - 1), L - l, mL, nL)
        pd0, pd1 = D[_gp_q(l, L, slice(None), 0)], D[_gp_q(l, L, slice(None), 1)]
        _GP_HOLD.extend((pd0, pd1))
        _d00, d10 = fnp.unstack(pd0, axis=l - 1)
        backward.append((qs[1], Q[2:6:3], _gp_front(pd0, l - 1), d10, Q[6:3:-2], S[0:2], S[1::-1], Q[3:5], _gp_front(pd1, l - 1), S[0], qs[0]))
        Q = D
        qs = fnp.unstack(Q, axis=0) if l > 1 else None
    return _gp_put(_GP_O_FRAMES, key, (top2, leaf, tuple(backward)))


def _gp_levels(alevels, blevels):
    for (acopies, a21, a_i0, y03, y4, y41, y56, y5, y3), (bcopies, b_i1, b_j0, b22, b21, z13, z46, z4, z5, z0) in zip(alevels, blevels):
        _GP_HOLD.append(fnp.stack(acopies, axis=0, out=y03))           # A slots 0,1,2 = a22, a11, a12
        _GP_HOLD.append(fnp.add(a21, acopies[0], out=y4))              # T4 = a21 + a22
        _GP_HOLD.append(fnp.subtract(y41, a_i0, out=y56))              # (T5, T6) = (T4 - a11, a11 - a21)
        _GP_HOLD.append(fnp.subtract(acopies[2], y5, out=y3))          # T2 = a12 - T5
        _GP_HOLD.append(fnp.stack(bcopies, axis=0, out=z13))           # B slots 1,2,3 = b11, b21, b22
        _GP_HOLD.append(fnp.subtract(b_i1, b_j0, out=z46))             # (bb4, bb6) = (b12 - b11, b22 - b12)
        _GP_HOLD.append(fnp.subtract(b22, z4, out=z5))                 # bb5 = b22 - bb4
        _GP_HOLD.append(fnp.subtract(z5, b21, out=z0))                 # bb3 = bb5 - b21


def _gp_a_one(full):
    acopies, a21, a_i0, y03, y4, y41, y56, y5, y3 = full
    _GP_HOLD.append(fnp.stack(acopies, axis=0, out=y03))
    _GP_HOLD.append(fnp.add(a21, acopies[0], out=y4))
    _GP_HOLD.append(fnp.subtract(y41, a_i0, out=y56))
    _GP_HOLD.append(fnp.subtract(acopies[2], y5, out=y3))


def _gp_b_one(full, elim_b):
    bcopies, b_i1, b_j0, b22, b21, z13, z46, z4, z5, z0 = full
    if not elim_b:
        _GP_HOLD.append(fnp.stack(bcopies, axis=0, out=z13))
    _GP_HOLD.append(fnp.subtract(b_i1, b_j0, out=z46))
    _GP_HOLD.append(fnp.subtract(b22, z4, out=z5))
    _GP_HOLD.append(fnp.subtract(z5, b21, out=z0))


def _gp_a_last(alast, elim_a):
    afull, (a11, a12, a21, a22, y3, y4, y5, y6) = alast[0], alast[1]
    if elim_a:
        _GP_HOLD.append(fnp.add(a21, a22, out=y4))                     # T4 = a21 + a22
        _GP_HOLD.append(fnp.subtract(y4, a11, out=y5))                 # T5 = T4 - a11
        _GP_HOLD.append(fnp.subtract(a11, a21, out=y6))                # T6 = a11 - a21
        _GP_HOLD.append(fnp.subtract(a12, y5, out=y3))                 # T2 = a12 - T5
    else:
        _gp_a_one(afull)


def _gp_leaf(alast, blast, leaf, elim_a, elim_b, aleaf=None):
    _afull, _alevel, (va22, a_j0, vy3, y47, y37), (y0, y12, cy3, cy47), Y = alast
    _bfull, (bz0, b_i0, b22v, bz47), (cz0, z12, z37), Z = blast
    Q, q0, q12, q3, q47, q37, _place = leaf
    if aleaf is not None:                             # W5 cached A leaf (all seven slots materialized)
        if elim_b:
            _GP_HOLD.append(fnp.matmul(aleaf[0], bz0, out=q0))
            _GP_HOLD.append(fnp.matmul(aleaf[1:3], b_i0, out=q12))
            _GP_HOLD.append(fnp.matmul(aleaf[3], b22v, out=q3))
            _GP_HOLD.append(fnp.matmul(aleaf[4:7], bz47, out=q47))
        else:
            _GP_HOLD.append(fnp.matmul(aleaf, Z, out=Q))
    elif elim_a and elim_b:
        _GP_HOLD.append(fnp.matmul(va22, bz0, out=q0))                 # p0 = a22 @ bb3
        _GP_HOLD.append(fnp.matmul(a_j0, b_i0, out=q12))               # (p1, p2) = (a11 @ b11, a12 @ b21)
        _GP_HOLD.append(fnp.matmul(vy3, b22v, out=q3))                 # p3 = T2 @ b22
        _GP_HOLD.append(fnp.matmul(y47, bz47, out=q47))                # p4..p6
    elif elim_a:
        _GP_HOLD.append(fnp.matmul(va22, cz0, out=q0))
        _GP_HOLD.append(fnp.matmul(a_j0, z12, out=q12))
        _GP_HOLD.append(fnp.matmul(y37, z37, out=q37))
    elif elim_b:
        _GP_HOLD.append(fnp.matmul(y0, bz0, out=q0))
        _GP_HOLD.append(fnp.matmul(y12, b_i0, out=q12))
        _GP_HOLD.append(fnp.matmul(cy3, b22v, out=q3))
        _GP_HOLD.append(fnp.matmul(cy47, bz47, out=q47))
    else:
        _GP_HOLD.append(fnp.matmul(Y, Z, out=Q))


def _gp_backward(backward):
    for p1, p25, d_i0, s0, p64, s12, s21, p34, d_i1, s1, p0 in backward:
        _GP_HOLD.append(fnp.add(p1, p25, out=d_i0))                    # tl = p1 + p2 ; s0 = p1 + p5 (held in the bl corner)
        _GP_HOLD.append(fnp.add(s0, p64, out=s12))                     # s1 = s0 + p6 ; s2 = s0 + p4 (u-scratch)
        _GP_HOLD.append(fnp.add(s21, p34, out=d_i1))                   # tr = s2 + p3 ; br = s1 + p4
        _GP_HOLD.append(fnp.subtract(s1, p0, out=s0))                  # bl = s1 - p0


def _gp_disjoint(used, place):
    return used[1] <= place[0] or place[1] <= used[0]


def _packed_product_cached(a, b, out, views, levels):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    assert int(b.shape[-2]) == k and tuple(out.shape) == (m, n)
    assert m % 2 ** levels == k % 2 ** levels == n % 2 ** levels == 0
    assert m * k <= _GP_TOP_ELEMENTS and k * n <= _GP_TOP_ELEMENTS and m * n <= _GP_TOP_ELEMENTS
    atop, alevels, alast, aused = _gp_a_frame(views, levels, m, k)
    btop, blevels, blast, bused = _gp_b_frame(views, levels, k, n)
    otop, leaf, backward = _gp_o_frame(views, levels, m, n)
    elim_a = levels > 1 and _gp_disjoint(aused, leaf[6])
    elim_b = levels > 1 and _gp_disjoint(bused, leaf[6])
    _GP_HOLD.append(fnp.copyto(atop, a))
    _GP_HOLD.append(fnp.copyto(btop, b))
    _gp_levels(alevels, blevels)
    _gp_a_last(alast, elim_a)
    _gp_b_one(blast[0], elim_b)
    _gp_leaf(alast, blast, leaf, elim_a, elim_b)
    _gp_backward(backward)
    _GP_HOLD.append(fnp.copyto(out, otop))
    _GP_HOLD.clear()
    return out


def _w5_port_product_body(a, b, out, views, levels, cache):
    # grouped form of the bank's W-left D5 reuse body: the A side (all levels, copies kept) runs only for the creating call and its
    # full level-L leaf is copied into the owner's cache, exactly as the bank does; every call runs the B side and the backward.
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    assert int(b.shape[-2]) == k and tuple(out.shape) == (m, n)
    assert m % 2 ** levels == k % 2 ** levels == n % 2 ** levels == 0
    atop, alevels, alast, _aused = _gp_a_frame(views, levels, m, k)
    btop, blevels, blast, bused = _gp_b_frame(views, levels, k, n)
    otop, leaf, backward = _gp_o_frame(views, levels, m, n)
    elim_b = levels > 1 and _gp_disjoint(bused, leaf[6])
    if cache._leaf is None:
        _GP_HOLD.append(fnp.copyto(atop, a))
        for full in alevels:
            _gp_a_one(full)
        _gp_a_one(alast[0])
        cached = fnp.empty(cache._shape, dtype=fnp.float32)
        cache._own_leaf(cached, cached)
        _GP_HOLD.append(fnp.copyto(cached, alast[4]))
    _GP_HOLD.append(fnp.copyto(btop, b))
    for full in blevels:
        _gp_b_one(full, False)
    _gp_b_one(blast[0], elim_b)
    _gp_leaf(alast, blast, leaf, False, elim_b, aleaf=cache._leaf)
    _gp_backward(backward)
    _GP_HOLD.append(fnp.copyto(out, otop))
    _GP_HOLD.clear()
    return out
# END FABLE-ENGINEER GROUPED-REQUEST WINOGRAD KERNEL v7c

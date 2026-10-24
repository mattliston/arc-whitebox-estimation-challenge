

# BEGIN FABLE-ENGINEER GROUPED-REQUEST WINOGRAD KERNEL (E-3 v5, 2026-09-22)
# Same Winograd recursion, same operand pairs and same operation order per element as _packed_product_cached (bit-exact products);
# only the request structure changes: every level runs 4 A-side, 4 B-side and 4 backward requests (was 6/5/7) by applying each
# ufunc to strided PAIRS of quadrants/slots at once. Every arena region is viewed ONCE as a fully blocked array
# (7,)*l + (2,)*(L-l) + (mL,) + (2,)*(L-l) + (kL,) (newest slot axis FIRST, same memory order as the original frames), so all
# quadrant/pair views are free cached getitems/moveaxes. Slot order is permuted (new 0..6 = old 3,0,1,2,4,5,6) so that each side's
# three copies are one stack. A and B are transformed level by level (the arena layouts reuse regions across levels, exactly as
# the original kernel's lifetimes allow). The backward writes s1/s2 into the original u-scratch region (s0 into the bl corner), so
# no op reads and writes overlapping memory. The top operands are copied into fixed blocked buffers (one copyto each, billed
# m*k, k*n, m*n) instead of 12 per-call quadrant getitems. v3: the LAST level materializes no copies: its copy slots are read
# by the leaf products straight from the previous level's quadrants (saves 3/7 of the last level's A and/or B slot writes) - but
# only for a side whose level-(L-1) array's used arena extent does not overlap the leaf products' used extent (the arena layouts
# reuse dead regions); such a side keeps its copies. The check is static per (layout, shape).
_GP_A_FRAMES = {}; _GP_B_FRAMES = {}; _GP_O_FRAMES = {}
_GP_TOP = None
_GP_TOP_ELEMENTS = 1024 * 8192


def _gp_top_buffers():
    global _GP_TOP
    if _GP_TOP is None:
        _GP_TOP = (fnp.empty((_GP_TOP_ELEMENTS,), dtype=fnp.float32), fnp.empty((_GP_TOP_ELEMENTS,), dtype=fnp.float32),
                   fnp.empty((_GP_TOP_ELEMENTS,), dtype=fnp.float32))
    return _GP_TOP


def _gp_region(arena, spec, lead, L_rest, rows, cols):
    start, stop, _shape = spec
    exact = tuple(lead) + (2,) * L_rest + (rows,) + (2,) * L_rest + (cols,)
    numel = 1
    for x in exact:
        numel *= x
    assert start + numel <= stop, (spec, exact)
    return fnp.reshape(arena[start:start + numel], exact)


def _gp_top(buffer, L, rows, cols):
    flat = buffer[:(rows << L) * (cols << L)]
    return (fnp.reshape(flat, (rows << L, cols << L)),
            fnp.reshape(flat, (2,) * L + (rows,) + (2,) * L + (cols,)))


def _gp_q(l, L, i, j):
    # quadrant (i,j) of a level-(l-1) blocked array (lead (7,)*(l-1)); i or j may be a slice (pair)
    return (slice(None),) * (l - 1) + (i,) + (slice(None),) * (L - l + 1) + (j,)


def _gp_front(view, axis):
    return view if axis == 0 else fnp.moveaxis(view, axis, 0)


def _gp_put(cache, key, value):
    if len(cache) >= 4096:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def _gp_a_frame(views, L, m, k):
    key = (id(views), L, m, k)
    found = _GP_A_FRAMES.get(key)
    if found is not None:
        return found
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    mL, kL = m >> L, k >> L
    top2, X = _gp_top(_gp_top_buffers()[0], L, mL, kL)
    levels = []
    for l in range(1, L + 1):
        Y = _gp_region(arena, specs[2 * (l - 1)], (7,) * l, L - l, mL, kL)
        a11, a12, a21, a22 = (X[_gp_q(l, L, 0, 0)], X[_gp_q(l, L, 0, 1)], X[_gp_q(l, L, 1, 0)], X[_gp_q(l, L, 1, 1)])
        a_i0 = _gp_front(X[_gp_q(l, L, slice(None), 0)], l - 1)
        full = ((a22, a11, a12), a21, a_i0, Y[0:3], Y[4], Y[4::-3], Y[5:7], Y[5], Y[3])
        if l < L:
            levels.append(full)
        else:
            a_j0 = _gp_front(X[_gp_q(l, L, 0, slice(None))], L)
            last = (full, (a11, a12, a21, a22, Y[3], Y[4], Y[5], Y[6]), (a22, a_j0, Y[3], Y[4:7], Y[3:7]), (Y[0], Y[1:3], Y[3], Y[4:7]), Y)
            start = specs[2 * (L - 2)][0] if L > 1 else -1
            used = (start, start + 7 ** (L - 1) * 4 * mL * kL) if L > 1 else (-1, -1)
        X = Y
    return _gp_put(_GP_A_FRAMES, key, (top2, tuple(levels), last, used))


def _gp_b_frame(views, L, k, n):
    key = (id(views), L, k, n)
    found = _GP_B_FRAMES.get(key)
    if found is not None:
        return found
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    kL, nL = k >> L, n >> L
    top2, X = _gp_top(_gp_top_buffers()[1], L, kL, nL)
    levels = []
    for l in range(1, L + 1):
        Z = _gp_region(arena, specs[2 * (l - 1) + 1], (7,) * l, L - l, kL, nL)
        b11, b21, b22 = X[_gp_q(l, L, 0, 0)], X[_gp_q(l, L, 1, 0)], X[_gp_q(l, L, 1, 1)]
        b_i1 = _gp_front(X[_gp_q(l, L, slice(None), 1)], l - 1)
        b_j0 = _gp_front(X[_gp_q(l, L, 0, slice(None))], L)
        full = ((b11, b21, b22), b_i1, b_j0, b22, b21, Z[1:4], Z[4:7:2], Z[4], Z[5], Z[0])
        if l < L:
            levels.append(full)
        else:
            b_i0 = _gp_front(X[_gp_q(l, L, slice(None), 0)], l - 1)
            last = (full, (Z[0], b_i0, b22, Z[4:7]), (Z[0], Z[1:3], Z[3:7]), Z)
            start = specs[2 * (L - 2) + 1][0] if L > 1 else -1
            used = (start, start + 7 ** (L - 1) * 4 * kL * nL) if L > 1 else (-1, -1)
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
    leaf_spec = specs[4 * L]; inter_specs = specs[4 * L + 1:4 * L + L]; u_specs = specs[5 * L:6 * L]
    top2, D0 = _gp_top(_gp_top_buffers()[2], L, mL, nL)
    Q = _gp_region(arena, leaf_spec, (7,) * L, 0, mL, nL)
    leaf = (Q, Q[0], Q[1:3], Q[3], Q[4:7], Q[3:7], (leaf_spec[0], leaf_spec[0] + 7 ** L * mL * nL))
    backward = []
    for unwind, l in enumerate(range(L, 0, -1)):
        D = _gp_region(arena, inter_specs[unwind], (7,) * (l - 1), L - l + 1, mL, nL) if l > 1 else D0
        S = _gp_region(arena, u_specs[unwind], (3,) + (7,) * (l - 1), L - l, mL, nL)
        d_i0 = _gp_front(D[_gp_q(l, L, slice(None), 0)], l - 1)
        d_i1 = _gp_front(D[_gp_q(l, L, slice(None), 1)], l - 1)
        backward.append((Q[1], Q[2:6:3], d_i0, D[_gp_q(l, L, 1, 0)], Q[6:3:-2], S[0:2], S[1::-1], Q[3:5], d_i1, S[0], Q[0]))
        Q = D
    return _gp_put(_GP_O_FRAMES, key, (top2, leaf, tuple(backward)))


def _gp_levels(alevels, blevels):
    for (acopies, a21, a_i0, y03, y4, y41, y56, y5, y3), (bcopies, b_i1, b_j0, b22, b21, z13, z46, z4, z5, z0) in zip(alevels, blevels):
        fnp.stack(acopies, axis=0, out=y03)           # A slots 0,1,2 = a22, a11, a12
        fnp.add(a21, acopies[0], out=y4)              # T4 = a21 + a22
        fnp.subtract(y41, a_i0, out=y56)              # (T5, T6) = (T4 - a11, a11 - a21)
        fnp.subtract(acopies[2], y5, out=y3)          # T2 = a12 - T5
        fnp.stack(bcopies, axis=0, out=z13)           # B slots 1,2,3 = b11, b21, b22
        fnp.subtract(b_i1, b_j0, out=z46)             # (bb4, bb6) = (b12 - b11, b22 - b12)
        fnp.subtract(b22, z4, out=z5)                 # bb5 = b22 - bb4
        fnp.subtract(z5, b21, out=z0)                 # bb3 = bb5 - b21


def _gp_last_and_leaf(alast, blast, leaf, elim_a, elim_b):
    afull, (a11, a12, a21, a22, y3, y4, y5, y6), (va22, a_j0, vy3, y47, y37), (y0, y12, cy3, cy47), Y = alast
    bfull, (bz0, b_i0, b22v, bz47), (cz0, z12, z37), Z = blast
    Q, q0, q12, q3, q47, q37, _used = leaf
    if elim_a:
        fnp.add(a21, a22, out=y4)                     # T4 = a21 + a22
        fnp.subtract(y4, a11, out=y5)                 # T5 = T4 - a11
        fnp.subtract(a11, a21, out=y6)                # T6 = a11 - a21
        fnp.subtract(a12, y5, out=y3)                 # T2 = a12 - T5
    else:
        _gp_a_one(afull)
    _gp_b_one(bfull, elim_b)
    if elim_a and elim_b:
        fnp.matmul(va22, bz0, out=q0)                 # p0 = a22 @ bb3
        fnp.matmul(a_j0, b_i0, out=q12)               # (p1, p2) = (a11 @ b11, a12 @ b21)
        fnp.matmul(vy3, b22v, out=q3)                 # p3 = T2 @ b22
        fnp.matmul(y47, bz47, out=q47)                # p4..p6
    elif elim_a:
        fnp.matmul(va22, cz0, out=q0)
        fnp.matmul(a_j0, z12, out=q12)
        fnp.matmul(y37, z37, out=q37)
    elif elim_b:
        fnp.matmul(y0, bz0, out=q0)
        fnp.matmul(y12, b_i0, out=q12)
        fnp.matmul(cy3, b22v, out=q3)
        fnp.matmul(cy47, bz47, out=q47)
    else:
        fnp.matmul(Y, Z, out=Q)


def _gp_a_one(full):
    acopies, a21, a_i0, y03, y4, y41, y56, y5, y3 = full
    fnp.stack(acopies, axis=0, out=y03)
    fnp.add(a21, acopies[0], out=y4)
    fnp.subtract(y41, a_i0, out=y56)
    fnp.subtract(acopies[2], y5, out=y3)


def _gp_b_one(full, elim_b):
    bcopies, b_i1, b_j0, b22, b21, z13, z46, z4, z5, z0 = full
    if not elim_b:
        fnp.stack(bcopies, axis=0, out=z13)
    fnp.subtract(b_i1, b_j0, out=z46)
    fnp.subtract(b22, z4, out=z5)
    fnp.subtract(z5, b21, out=z0)


def _gp_backward(backward):
    for p1, p25, d_i0, s0, p64, s12, s21, p34, d_i1, s1, p0 in backward:
        fnp.add(p1, p25, out=d_i0)                    # tl = p1 + p2 ; s0 = p1 + p5 (held in the bl corner)
        fnp.add(s0, p64, out=s12)                     # s1 = s0 + p6 ; s2 = s0 + p4 (u-scratch)
        fnp.add(s21, p34, out=d_i1)                   # tr = s2 + p3 ; br = s1 + p4
        fnp.subtract(s1, p0, out=s0)                  # bl = s1 - p0


def _packed_product_cached(a, b, out, views, levels):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    assert int(b.shape[-2]) == k and tuple(out.shape) == (m, n)
    assert m % 2 ** levels == k % 2 ** levels == n % 2 ** levels == 0
    assert m * k <= _GP_TOP_ELEMENTS and k * n <= _GP_TOP_ELEMENTS and m * n <= _GP_TOP_ELEMENTS
    atop, alevels, alast, aused = _gp_a_frame(views, levels, m, k)
    btop, blevels, blast, bused = _gp_b_frame(views, levels, k, n)
    otop, leaf, backward = _gp_o_frame(views, levels, m, n)
    lused = leaf[6]
    elim_a = levels > 1 and (aused[1] <= lused[0] or lused[1] <= aused[0])
    elim_b = levels > 1 and (bused[1] <= lused[0] or lused[1] <= bused[0])
    fnp.copyto(atop, a)
    fnp.copyto(btop, b)
    _gp_levels(alevels, blevels)
    _gp_last_and_leaf(alast, blast, leaf, elim_a, elim_b)
    _gp_backward(backward)
    fnp.copyto(out, otop)
    return out
# END FABLE-ENGINEER GROUPED-REQUEST WINOGRAD KERNEL

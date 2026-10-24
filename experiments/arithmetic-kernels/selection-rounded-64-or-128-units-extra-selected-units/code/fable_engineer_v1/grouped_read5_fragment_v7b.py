

# BEGIN FABLE-ENGINEER GROUPED READ5 (E-14 v7b, 2026-09-22): the fixed three-packet D5 read on the grouped-request structure, with the
# bank's frame sharing: A-side views keyed by (m, k), B-side by (k, n), backward by (m, n), child regions by (packet size, dims)
# (the two 2-slot packets share them), packet views by (m, k, n). Levels 1-4 through the READ arena regions blocked five levels
# deep, level 5 per packet (slots 0:2, 2:4, 4:7 of the level-4 arrays) copy-free with 4 leaf matmuls per packet, then the grouped
# backward. Same operand pairs and order per element as the bank (bit-exact products).
_GP_R5 = {}


def _gp_r5_get(key, build):
    found = _GP_R5.get(key)
    if found is None:
        found = _gp_put(_GP_R5, key, build())
    return found


def _gp_r5_side(views, rows, cols, which, top_index):
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    L = 5
    rL, cL = rows >> L, cols >> L
    top2, X = _gp_top_cached(top_index, L, rL, cL)
    levels = []
    for l in range(1, 5):
        start, stop, _shape = specs[2 * (l - 1) + which]
        Y = _gp_region(arena, start, stop, (7,) * l, L - l, rL, cL)
        ys = fnp.unstack(Y, axis=0)
        p0, p1 = X[_gp_q(l, L, slice(None), 0)], X[_gp_q(l, L, slice(None), 1)]
        q00, q10 = fnp.unstack(p0, axis=l - 1)
        q01, q11 = fnp.unstack(p1, axis=l - 1)
        if which == 0:
            levels.append(((q11, q00, q01), q10, _gp_front(p0, l - 1), Y[0:3], ys[4], Y[4::-3], Y[5:7], ys[5], ys[3]))
        else:
            levels.append(((q00, q10, q11), _gp_front(p1, l - 1), _gp_front(X[_gp_q(l, L, 0, slice(None))], L), q11, q10,
                           Y[1:4], Y[4:7:2], ys[4], ys[5], ys[0]))
        X = Y
    return (top2, tuple(levels), X)


def _gp_r5_back(views, m, n):
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    L = 5
    mL, nL = m >> L, n >> L
    otop, D0 = _gp_top_cached(2, L, mL, nL)
    P4 = _gp_region(arena, specs[16][0], specs[16][1], (7,) * 4, 1, mL, nL)
    Q = P4; qs = fnp.unstack(Q, axis=0)
    backward = []
    for unwind, l in enumerate(range(4, 0, -1)):
        if l > 1:
            D = _gp_region(arena, specs[17 + unwind][0], specs[17 + unwind][1], (7,) * (l - 1), L - l + 1, mL, nL)
        else:
            D = D0
        S = _gp_region(arena, specs[20 + unwind][0], specs[20 + unwind][1], (3,) + (7,) * (l - 1), L - l, mL, nL)
        pd0, pd1 = D[_gp_q(l, L, slice(None), 0)], D[_gp_q(l, L, slice(None), 1)]
        _d00, d10 = fnp.unstack(pd0, axis=l - 1)
        backward.append((qs[1], Q[2:6:3], _gp_front(pd0, l - 1), d10, Q[6:3:-2], S[0:2], S[1::-1], Q[3:5], _gp_front(pd1, l - 1), S[0], qs[0]))
        Q = D
        qs = fnp.unstack(Q, axis=0) if l > 1 else None
    return (otop, P4, tuple(backward))


def _gp_r5_child(views, which, q, rows, cols):
    arena = _VIEW_SPECS[id(views)][0]
    lead = (q, 7, 7, 7)
    if which == 'a':
        v = _read5_view(arena, 157351936, 216358912, (7,) + lead + (rows >> 5, cols >> 5))
    elif which == 'b':
        v = _read5_view(arena, 216358912, 275365888, (7,) + lead + (rows >> 5, cols >> 5))
    elif which == 'c':
        v = _read5_view(arena, 275365888, 282741760, (7,) + lead + (rows >> 5, cols >> 5))
    else:
        v = _read5_view(arena, 157351936, 216358912, (3,) + lead + (rows >> 5, cols >> 5))
    return (v, fnp.unstack(v, axis=0))


def _gp_r5_packets(views, m, k, n, XA, XB, P4):
    L = 5
    packets = []
    for begin, end in ((0, 2), (2, 4), (4, 7)):
        q = end - begin
        ca, cas = _gp_r5_get(('ca', id(views), q, m, k), lambda: _gp_r5_child(views, 'a', q, m, k))
        cb, cbs = _gp_r5_get(('cb', id(views), q, k, n), lambda: _gp_r5_child(views, 'b', q, k, n))
        cc, ccs = _gp_r5_get(('cc', id(views), q, m, n), lambda: _gp_r5_child(views, 'c', q, m, n))
        ss, _sss = _gp_r5_get(('ss', id(views), q, m, n), lambda: _gp_r5_child(views, 's', q, m, n))
        pa, pb, D = XA[begin:end], XB[begin:end], P4[begin:end]
        pa0, pa1 = pa[_gp_q(L, L, slice(None), 0)], pa[_gp_q(L, L, slice(None), 1)]
        a11, a21 = fnp.unstack(pa0, axis=L - 1)
        a12, a22 = fnp.unstack(pa1, axis=L - 1)
        alast = (a11, a12, a21, a22, cas[3], cas[4], cas[5], cas[6], _gp_front(pa[_gp_q(L, L, 0, slice(None))], L), ca[4:7])
        pb0, pb1 = pb[_gp_q(L, L, slice(None), 0)], pb[_gp_q(L, L, slice(None), 1)]
        _b11, b21 = fnp.unstack(pb0, axis=L - 1)
        _b12, b22 = fnp.unstack(pb1, axis=L - 1)
        blast = (_gp_front(pb1, L - 1), _gp_front(pb[_gp_q(L, L, 0, slice(None))], L), b22, b21,
                 cb[4:7:2], cbs[4], cbs[5], cbs[0], _gp_front(pb0, L - 1), cb[4:7])
        leaf = (ccs[0], cc[1:3], ccs[3], cc[4:7])
        pd0, pd1 = D[_gp_q(L, L, slice(None), 0)], D[_gp_q(L, L, slice(None), 1)]
        _d00, d10 = fnp.unstack(pd0, axis=L - 1)
        back = ((ccs[1], cc[2:6:3], _gp_front(pd0, L - 1), d10, cc[6:3:-2], ss[0:2], ss[1::-1], cc[3:5], _gp_front(pd1, L - 1), ss[0], ccs[0]),)
        packets.append((alast, blast, leaf, back))
    return tuple(packets)


def _gp_r5_last_and_leaf(alast, blast, leaf):
    a11, a12, a21, a22, y3, y4, y5, y6, a_j0, y47 = alast
    b_i1, b_j0, b22, b21, z46, z4, z5, z0, b_i0, z47 = blast
    q0, q12, q3, q47 = leaf
    fnp.add(a21, a22, out=y4)
    fnp.subtract(y4, a11, out=y5)
    fnp.subtract(a11, a21, out=y6)
    fnp.subtract(a12, y5, out=y3)
    fnp.subtract(b_i1, b_j0, out=z46)
    fnp.subtract(b22, z4, out=z5)
    fnp.subtract(z5, b21, out=z0)
    fnp.matmul(a22, z0, out=q0)
    fnp.matmul(a_j0, b_i0, out=q12)
    fnp.matmul(y3, b22, out=q3)
    fnp.matmul(y47, z47, out=q47)


def _packed_read5(a, b, out, views):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    assert int(b.shape[-2]) == k and tuple(out.shape) == (m, n)
    assert 0 < m <= 1024 and 0 < k <= 8192 and 0 < n <= 1024 and m % 32 == k % 32 == n % 32 == 0
    assert len(_VIEW_SPECS[id(views)][1]) == 24
    atop, levels_a, XA = _gp_r5_get(('A', id(views), m, k), lambda: _gp_r5_side(views, m, k, 0, 0))
    btop, levels_b, XB = _gp_r5_get(('B', id(views), k, n), lambda: _gp_r5_side(views, k, n, 1, 1))
    otop, P4, backward = _gp_r5_get(('O', id(views), m, n), lambda: _gp_r5_back(views, m, n))
    packets = _gp_r5_get(('P', id(views), m, k, n), lambda: _gp_r5_packets(views, m, k, n, XA, XB, P4))
    fnp.copyto(atop, a)
    fnp.copyto(btop, b)
    _gp_levels(levels_a, levels_b)
    for alast, blast, leaf, back in packets:
        _gp_r5_last_and_leaf(alast, blast, leaf)
        _gp_backward(back)
    _gp_backward(backward)
    fnp.copyto(out, otop)
    return out
# END FABLE-ENGINEER GROUPED READ5 v7b

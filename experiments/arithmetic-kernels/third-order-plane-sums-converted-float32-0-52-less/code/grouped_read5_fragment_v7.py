

# BEGIN FABLE-ENGINEER GROUPED READ5 (E-5 for v7, 2026-09-22): the fixed three-packet D5 read (_packed_read5) on the grouped-request
# structure: levels 1-4 through the READ arena regions blocked five levels deep, level 5 per packet (slots 0:2, 2:4, 4:7 of the
# level-4 arrays) through the same child regions, copy-free (4 leaf matmuls per packet), then the grouped backward. Same operand
# pairs and order as the original per element (bit-exact products).
_GP_R5_FRAMES = {}


def _gp_r5_frame(views, m, k, n):
    key = (id(views), m, k, n)
    found = _GP_R5_FRAMES.get(key)
    if found is not None:
        return found
    assert 0 < m <= 1024 and 0 < k <= 8192 and 0 < n <= 1024 and m % 32 == k % 32 == n % 32 == 0
    arena, specs, _bound = _VIEW_SPECS[id(views)]
    assert len(specs) == 24
    L = 5
    mL, kL, nL = m >> L, k >> L, n >> L
    atop, XA = _gp_top(_gp_top_buffers()[0], L, mL, kL)
    btop, XB = _gp_top(_gp_top_buffers()[1], L, kL, nL)
    levels_a = []; levels_b = []
    for l in range(1, 5):
        Y = _gp_region(arena, specs[2 * (l - 1)][0], specs[2 * (l - 1)][1], (7,) * l, L - l, mL, kL)
        Z = _gp_region(arena, specs[2 * (l - 1) + 1][0], specs[2 * (l - 1) + 1][1], (7,) * l, L - l, kL, nL)
        a11, a12, a21, a22 = (XA[_gp_q(l, L, 0, 0)], XA[_gp_q(l, L, 0, 1)], XA[_gp_q(l, L, 1, 0)], XA[_gp_q(l, L, 1, 1)])
        a_i0 = _gp_front(XA[_gp_q(l, L, slice(None), 0)], l - 1)
        levels_a.append(((a22, a11, a12), a21, a_i0, Y[0:3], Y[4], Y[4::-3], Y[5:7], Y[5], Y[3]))
        b11, b21, b22 = XB[_gp_q(l, L, 0, 0)], XB[_gp_q(l, L, 1, 0)], XB[_gp_q(l, L, 1, 1)]
        b_i1 = _gp_front(XB[_gp_q(l, L, slice(None), 1)], l - 1)
        b_j0 = _gp_front(XB[_gp_q(l, L, 0, slice(None))], L)
        levels_b.append(((b11, b21, b22), b_i1, b_j0, b22, b21, Z[1:4], Z[4:7:2], Z[4], Z[5], Z[0]))
        XA, XB = Y, Z
    P4 = _gp_region(arena, specs[16][0], specs[16][1], (7,) * 4, 1, mL, nL)
    packets = []
    for begin, end in ((0, 2), (2, 4), (4, 7)):
        q = end - begin
        lead = (q, 7, 7, 7)
        ca = _read5_view(arena, 157351936, 216358912, (7,) + lead + (mL, kL))
        cb = _read5_view(arena, 216358912, 275365888, (7,) + lead + (kL, nL))
        cc = _read5_view(arena, 275365888, 282741760, (7,) + lead + (mL, nL))
        ss = _read5_view(arena, 157351936, 216358912, (3,) + lead + (mL, nL))
        pa, pb, D = XA[begin:end], XB[begin:end], P4[begin:end]
        a11, a12, a21, a22 = (pa[_gp_q(L, L, 0, 0)], pa[_gp_q(L, L, 0, 1)], pa[_gp_q(L, L, 1, 0)], pa[_gp_q(L, L, 1, 1)])
        alast = (a11, a12, a21, a22, ca[3], ca[4], ca[5], ca[6], _gp_front(pa[_gp_q(L, L, 0, slice(None))], L), ca[4:7])
        b21, b22 = pb[_gp_q(L, L, 1, 0)], pb[_gp_q(L, L, 1, 1)]
        blast = (_gp_front(pb[_gp_q(L, L, slice(None), 1)], L - 1), _gp_front(pb[_gp_q(L, L, 0, slice(None))], L), b22, b21,
                 cb[4:7:2], cb[4], cb[5], cb[0], _gp_front(pb[_gp_q(L, L, slice(None), 0)], L - 1), cb[4:7])
        leaf = (cc[0], cc[1:3], cc[3], cc[4:7])
        back = ((cc[1], cc[2:6:3], _gp_front(D[_gp_q(L, L, slice(None), 0)], L - 1), D[_gp_q(L, L, 1, 0)], cc[6:3:-2], ss[0:2],
                 ss[1::-1], cc[3:5], _gp_front(D[_gp_q(L, L, slice(None), 1)], L - 1), ss[0], cc[0]),)
        packets.append((alast, blast, leaf, back))
    otop, D0 = _gp_top(_gp_top_buffers()[2], L, mL, nL)
    inter_specs = specs[17:20]; u_specs = specs[20:24]
    Q = P4
    backward = []
    for unwind, l in enumerate(range(4, 0, -1)):
        D = _gp_region(arena, inter_specs[unwind][0], inter_specs[unwind][1], (7,) * (l - 1), L - l + 1, mL, nL) if l > 1 else D0
        S = _gp_region(arena, u_specs[unwind][0], u_specs[unwind][1], (3,) + (7,) * (l - 1), L - l, mL, nL)
        d_i0 = _gp_front(D[_gp_q(l, L, slice(None), 0)], l - 1)
        d_i1 = _gp_front(D[_gp_q(l, L, slice(None), 1)], l - 1)
        backward.append((Q[1], Q[2:6:3], d_i0, D[_gp_q(l, L, 1, 0)], Q[6:3:-2], S[0:2], S[1::-1], Q[3:5], d_i1, S[0], Q[0]))
        Q = D
    found = (atop, btop, otop, tuple(levels_a), tuple(levels_b), tuple(packets), tuple(backward))
    return _gp_put(_GP_R5_FRAMES, key, found)


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
    atop, btop, otop, levels_a, levels_b, packets, backward = _gp_r5_frame(views, m, k, n)
    fnp.copyto(atop, a)
    fnp.copyto(btop, b)
    _gp_levels(levels_a, levels_b)
    for alast, blast, leaf, back in packets:
        _gp_r5_last_and_leaf(alast, blast, leaf)
        _gp_backward(back)
    _gp_backward(backward)
    fnp.copyto(out, otop)
    return out
# END FABLE-ENGINEER GROUPED READ5

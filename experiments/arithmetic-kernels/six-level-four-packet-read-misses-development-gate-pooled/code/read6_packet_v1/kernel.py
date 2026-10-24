"""Two extra read levels in four bounded packets, using explicitly dead regions."""
_READ6_CHILD_CACHE = {}
_READ6_FRAME_CACHE = {}


def _read6_child_frame(arena, q, m, k, n):
    key = (id(arena), q, m, k, n)
    found = _READ6_CHILD_CACHE.get(key)
    if found is not None:
        return found
    r, t, c = m // 32, k // 32, n // 32
    rr, tt, cc = m // 64, k // 64, n // 64
    lead = (q, 7, 7, 7)
    a5 = _read5_view(arena, 157351936, 196689920, (7,) + lead + (r, t))
    b5 = _read5_view(arena, 196689920, 236027904, (7,) + lead + (t, c))
    a6 = _read5_view(arena, 236027904, 304869376, (7, 7) + lead + (rr, tt))
    b6 = _read5_view(arena, 304869376, 373710848, (7, 7) + lead + (tt, cc))
    c6 = _read5_view(arena, 373710848, 382316032, (7, 7) + lead + (rr, cc))
    # Child inputs are dead before reconstruction starts.
    c5 = _read5_view(arena, 157351936, 196689920, (7,) + lead + (r, c))
    s6 = _read5_view(arena, 196689920, 236027904, (3, 7) + lead + (rr, cc))
    s5 = _read5_view(arena, 196689920, 236027904, (3,) + lead + (r, c))
    aq6 = (a5[..., :rr, :tt], a5[..., :rr, tt:],
           a5[..., rr:, :tt], a5[..., rr:, tt:])
    bq6 = (b5[..., :tt, :cc], b5[..., :tt, cc:],
           b5[..., tt:, :cc], b5[..., tt:, cc:])
    corners6 = (c5[..., :rr, :cc], c5[..., :rr, cc:],
                c5[..., rr:, :cc], c5[..., rr:, cc:])
    found = (a6, b6, c6,
             fnp.unstack(a5, axis=0), fnp.unstack(b5, axis=0), a5[:2], b5[:3],
             aq6, bq6, fnp.unstack(a6, axis=0), fnp.unstack(b6, axis=0), a6[:2], b6[:3],
             fnp.unstack(c6, axis=0), fnp.unstack(s6, axis=0), corners6,
             fnp.unstack(c5, axis=0), fnp.unstack(s5, axis=0))
    return _cache_put(_READ6_CHILD_CACHE, key, found)


def _read6_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
    key = (id(views), m, k, n)
    found = _READ6_FRAME_CACHE.get(key)
    if found is not None:
        return found
    assert 0 < m <= 1024 and 0 < k <= 8192 and 0 < n <= 1024
    assert m % 64 == k % 64 == n % 64 == 0
    arena = _VIEW_SPECS[id(views)][0]
    r, t, c = m // 32, k // 32, n // 32
    parts = []
    for begin, end in ((0, 2), (2, 4), (4, 6), (6, 7)):
        a, b, out = leaf_a[begin:end], leaf_b[begin:end], leaf[begin:end]
        aq = (a[..., :r, :t], a[..., :r, t:], a[..., r:, :t], a[..., r:, t:])
        bq = (b[..., :t, :c], b[..., :t, c:], b[..., t:, :c], b[..., t:, c:])
        corners = (out[..., :r, :c], out[..., :r, c:], out[..., r:, :c], out[..., r:, c:])
        parts.append((aq, bq, corners, _read6_child_frame(arena, end - begin, m, k, n)))
    return _cache_put(_READ6_FRAME_CACHE, key, tuple(parts))


def _read6_forward(aq, bq, aa, bb, acopies, bcopies):
    a11, a12, a21, a22 = aq
    b11, b12, b21, b22 = bq
    fnp.add(a21, a22, out=aa[4])
    fnp.subtract(aa[4], a11, out=aa[5])
    fnp.subtract(a11, a21, out=aa[6])
    fnp.subtract(a12, aa[5], out=aa[2])
    fnp.stack((a11, a12), axis=0, out=acopies)
    fnp.copyto(aa[3], a22)
    fnp.subtract(b12, b11, out=bb[4])
    fnp.subtract(b22, bb[4], out=bb[5])
    fnp.subtract(b22, b12, out=bb[6])
    fnp.subtract(bb[5], b21, out=bb[3])
    fnp.stack((b11, b21, b22), axis=0, out=bcopies)


def _read6_backward(products, scratch, corners):
    p0, p1, p2, p3, p4, p5, p6 = products
    s0, s1, s2 = scratch
    tl, tr, bl, br = corners
    fnp.add(p0, p1, out=tl)
    fnp.add(p0, p5, out=s0)
    fnp.add(s0, p6, out=s1)
    fnp.add(s0, p4, out=s2)
    fnp.add(s2, p2, out=tr)
    fnp.subtract(s1, p3, out=bl)
    fnp.add(s1, p4, out=br)


def _read6_leaf_packets(m, k, n, views, leaf_a, leaf_b, leaf):
    for aq, bq, corners, child in _read6_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
        (a6, b6, c6, aa5, bb5, ac5, bc5, aq6, bq6, aa6, bb6, ac6, bc6,
         products6, scratch6, corners6, products5, scratch5) = child
        _read6_forward(aq, bq, aa5, bb5, ac5, bc5)
        _read6_forward(aq6, bq6, aa6, bb6, ac6, bc6)
        fnp.matmul(a6, b6, out=c6)
        _read6_backward(products6, scratch6, corners6)
        _read6_backward(products5, scratch5, corners)


_PACKED_ELEMENTS = 392150528
_PACKED_READ = (_PACKED_READ[:16]
                + ((382316032, 392150528, (7, 7, 7, 7, 64, 64)),)
                + _PACKED_READ[17:])

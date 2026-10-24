# Appended to the frozen Cost75 source by build.py. Only paid fnp arithmetic.
_READ5_CHILD_CACHE = {}
_READ5_FRAME_CACHE = {}


def _read5_view(arena, start, stop, shape):
    count = 1
    for dimension in shape:
        count *= dimension
    assert start + count <= stop
    return fnp.reshape(arena[start:start + count], shape)


def _read5_child_frame(arena, q, m, k, n):
    key = (id(arena), q, m, k, n)
    found = _READ5_CHILD_CACHE.get(key)
    if found is not None:
        return found
    rows, inner, columns = m // 32, k // 32, n // 32
    lead = (q, 7, 7, 7)
    ca = _read5_view(arena, 157351936, 216358912, (7,) + lead + (rows, inner))
    cb = _read5_view(arena, 216358912, 275365888, (7,) + lead + (inner, columns))
    cc = _read5_view(arena, 275365888, 282741760, (7,) + lead + (rows, columns))
    # The scratch lifetime starts only after the child matmul has completed.
    ss = _read5_view(arena, 157351936, 216358912, (3,) + lead + (rows, columns))
    found = (ca, cb, cc, fnp.unstack(ca, axis=0), fnp.unstack(cb, axis=0),
             ca[:2], cb[:3], fnp.unstack(cc, axis=0), fnp.unstack(ss, axis=0))
    return _cache_put(_READ5_CHILD_CACHE, key, found)


def _read5_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
    key = (id(views), m, k, n)
    found = _READ5_FRAME_CACHE.get(key)
    if found is not None:
        return found
    assert 0 < m <= 1024 and 0 < k <= 8192 and 0 < n <= 1024
    assert m % 32 == k % 32 == n % 32 == 0
    arena = _VIEW_SPECS[id(views)][0]
    rows, inner, columns = m // 32, k // 32, n // 32
    parts = []
    for begin, end in ((0, 2), (2, 4), (4, 7)):
        a, b, out = leaf_a[begin:end], leaf_b[begin:end], leaf[begin:end]
        aq = (a[..., :rows, :inner], a[..., :rows, inner:],
              a[..., rows:, :inner], a[..., rows:, inner:])
        bq = (b[..., :inner, :columns], b[..., :inner, columns:],
              b[..., inner:, :columns], b[..., inner:, columns:])
        corners = (out[..., :rows, :columns], out[..., :rows, columns:],
                   out[..., rows:, :columns], out[..., rows:, columns:])
        child = _read5_child_frame(arena, end - begin, m, k, n)
        parts.append((aq, bq, corners, child))
    return _cache_put(_READ5_FRAME_CACHE, key, tuple(parts))


def _read5_leaf_packets(m, k, n, views, leaf_a, leaf_b, leaf):
    for aq, bq, corners, child in _read5_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
        ca, cb, cc, aa, bb, acopies, bcopies, products, scratch = child
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
        fnp.matmul(ca, cb, out=cc)
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


_PACKED_ELEMENTS = 292576256
_PACKED_READ = (_PACKED_READ[:16]
                + ((282741760, 292576256, (7, 7, 7, 7, 64, 64)),)
                + _PACKED_READ[17:])

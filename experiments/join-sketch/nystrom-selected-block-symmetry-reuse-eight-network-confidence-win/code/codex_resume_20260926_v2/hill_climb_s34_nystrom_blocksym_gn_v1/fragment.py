# Selected Nyström rows form a weighted Gram matrix in real arithmetic.
# D5 changes rounding; only off-diagonal blocks are mirrored.
def _ng_d5(a, b):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    assert _ipt_fits(5, m, k, n)
    out = fnp.empty((m, n), dtype=fnp.float32)
    _TR_LAST[0] = None
    try:
        ta = _ipt_forward_a_ab(a, 5, m, k)
        tb = _ipt_forward_b_ab(b, 5, k, n)
        tc, z1, _pair, ops, basis = _ipt_side(2, 5, m, n)
        _BH_HOLD.append(fnp.matmul(ta, tb, out=tc))
        _ks_unwind(tc, z1, ops, basis, out, 5, m, n)
    finally:
        _TR_LAST[0] = None
        _BH_HOLD.clear()
    return out


def _ng_selected_gram(J, weight, rows):
    h = _E2Hold()
    m, k, r = int(J.shape[0]), int(J.shape[1]), int(rows.shape[0])
    selected = h(J[rows, :])
    rhs = h(h(weight[:, None]) * h(selected.T))
    strips = [(start, min(128, r - start)) for start in range(0, r, 128)]
    if not (_AB_ON and _WINOGRAD and r in (320, 384) and m >= r
            and str(J.dtype) == str(weight.dtype) == 'float32'
            and all(_ipt_fits(5, b, k, r-start) for start, b in strips)):
        return _shared_old384_mm(J, rhs)
    mask = h(fnp.ones(m, dtype=fnp.bool_))
    h(fnp.put(mask, rows, False))
    other = h(fnp.nonzero(mask)[0])
    Y = fnp.empty((m, r), dtype=fnp.float32)
    cols = h(fnp.arange(r, dtype=fnp.int64))
    if m > r:
        rest = h(_shared_old384_mm(h(J[other, :]), rhs))
        h(fnp.put(Y, h(h(other[:, None] * r) + cols), rest))
    gram = h(fnp.empty((r, r), dtype=fnp.float32))
    for start, b in strips:
        end = start + b
        part = h(_ng_d5(h(selected[start:end, :]), h(rhs[:, start:])))
        h(fnp.copyto(h(gram[start:end, start:]), part))
        if end < r:
            h(fnp.copyto(h(gram[end:, start:end]), h(h(part[:, b:]).T)))
    h(fnp.put(Y, h(h(rows[:, None] * r) + cols), gram))
    return Y


# Twenty-three new view keys for three saved geometries; same arena/caps.
_NG_PREBIND = (
    (0, 5, 64, 5568), (0, 5, 128, 4672), (0, 5, 128, 5568), (0, 5, 128, 6144),
    (1, 5, 4672, 128), (1, 5, 4672, 256), (1, 5, 4672, 384),
    (1, 5, 5568, 64), (1, 5, 5568, 192), (1, 5, 5568, 320),
    (1, 5, 6144, 128), (1, 5, 6144, 256),
    (2, 5, 64, 64), (2, 5, 128, 128), (2, 5, 128, 192),
    (2, 5, 128, 256), (2, 5, 128, 320), (2, 5, 128, 384),
    (0, 6, 448, 4672), (0, 6, 512, 5568), (1, 6, 5568, 320),
    (2, 6, 448, 384), (2, 6, 512, 320),
)
_PB_SINGLE = _NG_PREBIND + tuple(_PB_SINGLE)

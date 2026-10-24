# Quarter-zero HZ7 strip pruning, source-derived and not yet numerically certified.
# HZ7's remainder-32 pack sets both the deepest and topmost input bit to zero.
# The parent already prunes the deepest bit.  Apply the same known-zero support
# at the topmost bit, retaining the parent's pack, transforms and unwind.
_QZ7_PARENT_PRODUCT = _hz7_product
_QZ7_PARENT_VIEWS = _hz7_views
_QZ7_CONTEXT = []
_QZ7_VIEWS = {}
_QZ7_STATS = dict(products=0, packets=0)


def _qz7_views(M, K, w, hzr, hzk, qzr, qzk):
    key = (M, K, w, hzr, hzk, qzr, qzk)
    got = _QZ7_VIEWS.get(key)
    if got is not None:
        return got
    TA = _ipt_side(0, 7, M, K)[0]
    TB = _ipt_side(1, 7, K, w)[0]
    TC = _ipt_side(2, 7, M, w)[0]
    ra = M // 128 - int(hzr)
    ka = K // 128 - int(hzk)
    full = slice(None)
    mats, fills = [], []
    # T's component axes are deepest first and topmost last.  Splitting the
    # topmost axis is unnecessary when the deepest group already omits a strip.
    for ds, dr, dk, _dc in _hz_groups(hzr, hzk, False)[0]:
        for ts, tr, tk, _tc in _hz_groups(qzr and dr, qzk and dk, False)[0]:
            rf, kf = dr and tr, dk and tk
            rr = slice(None, ra) if hzr and not rf else full
            kk = slice(None, ka) if hzk and not kf else full
            head = (ds,) + (full,) * 5 + (ts,)
            mats.append((TA[head + (rr, kk)],
                         TB[head + (kk, full)],
                         TC[head + (rr, full)]))
    if hzr:
        for ds in _hz_slices([c for c in range(7) if c not in _HZ_ROW_FULL]):
            fills.append(TC[(ds,) + (full,) * 6 + (slice(ra, None), full)])
    if qzr:
        # Disjoint from the parent's deepest-bit fills: only the four deepest
        # row-support chunks need these additional top-bit zero writes.
        for ds in _hz_slices(_HZ_ROW_FULL):
            for ts in _hz_slices([c for c in range(7) if c not in _HZ_ROW_FULL]):
                fills.append(TC[(ds,) + (full,) * 5 + (ts, slice(ra, None), full)])
    got = dict(mats=mats, fills=fills)
    if len(_QZ7_VIEWS) >= 128:
        _QZ7_VIEWS.pop(next(iter(_QZ7_VIEWS)))
    _QZ7_VIEWS[key] = got
    return got


def _hz7_views(M, K, w, hzr, hzk):
    if _QZ7_CONTEXT:
        qzr, qzk = _QZ7_CONTEXT[-1]
        if qzr or qzk:
            _QZ7_STATS['packets'] += 1
            return _qz7_views(M, K, w, hzr, hzk, qzr, qzk)
    return _QZ7_PARENT_VIEWS(M, K, w, hzr, hzk)


def _hz7_product(a, b, m, k, n, dest=None):
    qzr, qzk = m % 128 == 32, k % 128 == 32
    _QZ7_CONTEXT.append((qzr, qzk))
    if qzr or qzk:
        _QZ7_STATS['products'] += 1
    try:
        return _QZ7_PARENT_PRODUCT(a, b, m, k, n, dest=dest)
    finally:
        _QZ7_CONTEXT.pop()


_PB_CACHES = tuple(_PB_CACHES) + ('_QZ7_VIEWS',)

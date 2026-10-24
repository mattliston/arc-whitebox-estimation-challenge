# ASTRA HZT v1: omit unused deepest-level HZD forward entries.
# Append to the pinned s51k2tw_x candidate. No layout or route changes.
# The retained arithmetic has exactly the parent's inputs and operation order.
# The existing _hz_leaf masks never read the omitted cells; keep its fills.
_HZT_PARENT_A = _HZDA_PARENT
_HZT_PARENT_B = _hzd_forward_b
_HZT_VIEWS = {}
_HZT_STATS = dict(a=0, b=0, saved_elements=0)


def _hzt_last_views(kind, rows, cols, hr, hc):
    L = 6
    entry = _ipt_side(kind, L, rows, cols)
    T, chunks = entry[:2]
    key = (kind, rows, cols, hr, hc)
    got = _HZT_VIEWS.get(key)
    if got is not None and got[0] is T:
        return got[1]
    X = chunks[L]
    full = slice(None)
    if kind == 0:
        # Component 3 is unused on the inner strip; component 1 feeds
        # only the discarded output-row half on the row strip.
        r3, c3 = full, (slice(None, cols // 64 - 1) if hc else full)
        r1, c1 = (slice(None, rows // 64 - 1) if hr else full), full
        op3 = ((fnp.subtract, X[2][..., r3, c3], X[5][..., r3, c3], X[3][..., r3, c3])
               if hc else (fnp.subtract, X[2], X[5], X[3]))
        op1 = ((fnp.subtract, X[6][..., r1, c1], X[5][..., r1, c1], X[1][..., r1, c1])
               if hr else (fnp.subtract, X[6], X[5], X[1]))
        ops = ((fnp.subtract, X[4], X[0], X[5]), op3, op1)
    else:
        # Component 6 is unused on the inner strip; component 4 feeds
        # only the discarded output-column half on the column strip.
        r6, c6 = (slice(None, rows // 64 - 1) if hr else full), full
        r4, c4 = full, (slice(None, cols // 64 - 1) if hc else full)
        op6 = ((fnp.subtract, X[5][..., r6, c6], X[2][..., r6, c6], X[6][..., r6, c6])
               if hr else (fnp.subtract, X[5], X[2], X[6]))
        op4 = ((fnp.subtract, X[3][..., r4, c4], X[5][..., r4, c4], X[4][..., r4, c4])
               if hc else (fnp.subtract, X[3], X[5], X[4]))
        ops = ((fnp.add, X[1], X[0], X[5]), op6, op4)
    if len(_HZT_VIEWS) >= 256:
        _HZT_VIEWS.pop(next(iter(_HZT_VIEWS)))
    _HZT_VIEWS[key] = (T, ops)
    return ops


def _hzt_forward_a(x, M, K, hr, hc):
    if not (hr or hc):
        return _HZT_PARENT_A(x, M, K, hr, hc)
    L = 6
    TA, X1, A02, ops, basis = _ipt_side(0, L, M, K)
    h = _BH_HOLD.append
    quad = A02[1] if A02[1] is not None else _l3_quad(A02, 0, L)
    _hzd_load(quad, x, L, M // 64 - int(hr), K // 64 - int(hc), hr, hc)
    X = X1[1]
    h(fnp.add(X[4], X[6], out=X[4]))
    for lo, hi in basis:
        h(fnp.add(lo, hi, out=lo))
    if ops is not None:
        for f, y, z, o in ops[:-2]:
            h(f(y, z, out=o))
    else:
        for j in range(1, L):
            X = X1[j]
            h(fnp.subtract(X[4], X[0], out=X[5]))
            h(fnp.subtract(X[2], X[5], out=X[3]))
            h(fnp.subtract(X[6], X[5], out=X[1]))
    for f, y, z, o in _hzt_last_views(0, M, K, hr, hc):
        h(f(y, z, out=o))
    _TR_LAST[0] = None
    _HZD_STATS['a'] += 1
    _HZT_STATS['a'] += 1
    _HZT_STATS['saved_elements'] += 7 ** 5 * (int(hr) * (K // 64) + int(hc) * (M // 64))
    return TA


# Preserve the existing final HZD-A resident-tree wrapper and its exact key.
_HZDA_PARENT = _hzt_forward_a


def _hzd_forward_b(x, K, N, hr, hc):
    if not (hr or hc):
        return _HZT_PARENT_B(x, K, N, hr, hc)
    L = 6
    TB, Y1, B02, ops, basis = _ipt_side(1, L, K, N)
    h = _BH_HOLD.append
    quad = B02[1] if B02[1] is not None else _l3_quad(B02, 1, L)
    _hzd_load(quad, x, L, K // 64 - int(hr), N // 64 - int(hc), hr, hc)
    Y = Y1[1]
    h(fnp.subtract(Y[3], Y[1], out=Y[1]))
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    if ops is not None:
        for f, y, z, o in ops[:-2]:
            h(f(y, z, out=o))
    else:
        for j in range(1, L):
            Y = Y1[j]
            h(fnp.add(Y[1], Y[0], out=Y[5]))
            h(fnp.subtract(Y[5], Y[2], out=Y[6]))
            h(fnp.subtract(Y[3], Y[5], out=Y[4]))
    for f, y, z, o in _hzt_last_views(1, K, N, hr, hc):
        h(f(y, z, out=o))
    _HZD_STATS['b'] += 1
    _HZT_STATS['b'] += 1
    _HZT_STATS['saved_elements'] += 7 ** 5 * (int(hr) * (N // 64) + int(hc) * (K // 64))
    return TB


_PB_CACHES = tuple(_PB_CACHES) + ('_HZT_VIEWS',)

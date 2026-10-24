

# BEGIN FABLE-ENGINEER SQ6 (E-24, 2026-09-22): square products (all dims <= 1024) may take a sixth Winograd level when every dim
# divides 64 and the depth rule's own value (bill + 750000 per request) says so (in practice the 1024^3 products: -5.5% bill each).
# Depth 6 runs through _packed_product_cached with the SQUARE6 arena layout. Changes rounding only (a deeper recursion), not physics.
def _rule_depth(kind, m, k, n, maximum=6):
    key = (kind, m, k, n, maximum, 'sq6')
    if key in _COST_RULE_CACHE:
        return _COST_RULE_CACHE[key]
    options = (0, 3, 4, 5, 6) if kind == 'sq' else ((0, 4, 5) if kind == 'read' else (0, 4))
    if kind == 'push' and m == 1024 and m % 64 == k % 64 == n % 64 == 0:
        options = (0, 4, 6)
    best_depth = 0
    best_value = 2 * m * k * n - m * n + 750000
    for depth in options[1:]:
        if depth > maximum or m % 2**depth or k % 2**depth or n % 2**depth:
            continue
        packets = (4096, n - 4096) if depth == 6 and n > 4096 else (n,)
        growth = (7 / 4)**depth
        bill = sum(2*m*k*nc*(7/8)**depth - m*nc*growth
                   + (7/3)*(m*k+k*nc+m*nc)*(growth-1) for nc in packets)
        calls = (18*depth+14)*len(packets)
        if kind == 'read' and depth == 5:
            calls = 142
        value = bill + 750000*calls
        if value < best_value:
            best_value, best_depth = value, depth
    _COST_RULE_CACHE[key] = best_depth
    return best_depth


def _square_schedule(m, k, n):
    if m == n:
        depth = _SQUARE_EQUAL_DEPTHS[m//32-1][k//32-1]
    elif (m == k and n == 1024) or (k == n and m == 1024):
        depth = _SQUARE_CROSS_DEPTHS[k//32-1]
    else:
        depth, _p = _square_schedule_reference(m, k, n)
    if depth >= 5 and m % 64 == 0 and k % 64 == 0 and n % 64 == 0:
        depth = 6
    depth = _rule_depth('sq', m, k, n, min(depth, _SQUARE_DEPTH_CAP))
    unit = 2**depth
    return depth, ((m+unit-1)//unit*unit, (k+unit-1)//unit*unit, (n+unit-1)//unit*unit)


def _square_product(a, b, m, k, n):
    depth, (mp, kp, npad) = _square_schedule(m, k, n)
    if depth == 0:
        return a@b
    if _SCRATCH[depth] is None:
        specs = {3:_PACKED_SQUARE, 4:_PACKED_SQUARE4, 5:_PACKED_SQUARE5, 6:_PACKED_SQUARE6}[depth]
        _SCRATCH[depth] = _cp3_bind_arena_views(_SCRATCH[0], specs)
    if mp != m or kp != k:
        aa = fnp.zeros((mp, kp), dtype=fnp.float32)
        fnp.copyto(aa[:m, :k], a)
    else:
        aa = a
    if kp != k or npad != n:
        bb = fnp.zeros((kp, npad), dtype=fnp.float32)
        fnp.copyto(bb[:k, :n], b)
    else:
        bb = b
    out = fnp.empty((mp, npad), dtype=fnp.float32)
    if depth == 3:
        result = _packed_product_3(aa, bb, out, _SCRATCH[3])
    elif depth == 4:
        result = _packed_product_4(aa, bb, out, _SCRATCH[4])
    elif depth == 5:
        result = _packed_product_5(aa, bb, out, _SCRATCH[5])
    else:
        result = _packed_product_cached(aa, bb, out, _SCRATCH[6], 6)
    return result if (mp == m and npad == n) else result[:m, :n]
# END FABLE-ENGINEER SQ6

def _peel_prebind_batch_io(ker, key):
    """Populate the existing bb_batch cache; public view APIs only.

    Requires the existing side and outer caches. No arithmetic or new buffer.
    `key` is exactly (family, L, cm, ck, cn, P), as in outer_cache.
    """
    env = ker.M
    fnp = ker.fnp
    cache = env._sm40_io_cache
    LA, RB, PC, _, parts = env._sm40_outer_cache[key]
    if all((key, i) in cache for i in range(len(parts))):
        return
    family, L, m, k, n, P = key
    u = 1 << L
    TA, XA, GA, basA, opsA = ker.side(0, L, P, m, k)
    TB, XB, GB, basB, opsB = ker.side(1, L, P, k, n)
    TC, XC, GC, basC, opsC = ker.side(2, L, P, m, n)
    # These two destination views depend on side geometry, not chunk index.
    dcache = getattr(env, '_peel_batch_destination_cache', None)
    if dcache is None:
        dcache = env._peel_batch_destination_cache = {}
    ka, kb = (family, L, P, m, k, 0), (family, L, P, k, n, 1)
    if ka not in dcache:
        dcache[ka] = fnp.moveaxis(GA[0:3:2], 0, L + 1)
    if kb not in dcache:
        dcache[kb] = fnp.moveaxis(GB[0:3:2], 0, 1)
    dA, dB = dcache[ka], dcache[kb]
    total = int(LA.shape[0])
    assert total % P == 0 and len(parts) == total // P
    cuts = tuple(range(P, total, P))
    ocache = getattr(env, '_peel_batch_operand_cache', None)
    if ocache is None:
        ocache = env._peel_batch_operand_cache = {}
    chunks = ()
    # For a fixed family, each operand always comes from the same outer
    # owner; a change in the third matrix dimension cannot change these views.
    for side, owner, rows, cols in ((0, LA, m, k), (1, RB, k, n), (2, PC, m, n)):
        operand_key = (family, L, P, rows, cols, side)
        if operand_key not in ocache:
            V = fnp.reshape(owner, (total,) + (2,) * L + (rows // u,) + (2,) * L + (cols // u,))
            if side == 0:
                first, lower = fnp.unstack(V, axis=1)
                v21, v22 = fnp.unstack(lower, axis=L + 1)
                arrays = (first, v21, v22)
            elif side == 1:
                first, right = fnp.unstack(V, axis=L + 2)
                v12, v22 = fnp.unstack(right, axis=1)
                arrays = (first, v12, v22)
            else:
                upper, lower = fnp.unstack(V, axis=1)
                v11, v12 = fnp.unstack(upper, axis=L + 1)
                v21, v22 = fnp.unstack(lower, axis=L + 1)
                arrays = (v11, v12, v21, v22)
            ocache[operand_key] = tuple(tuple(fnp.split(x, cuts, axis=0)) for x in arrays) if cuts else tuple((x,) for x in arrays)
        chunks += ocache[operand_key]
    common = (TA, TB, TC, XA[1], XB[1], XC[1], basA, basB, basC, opsA, opsB, opsC)
    for i, vs in enumerate(zip(*chunks)):
        cache[(key, i)] = common + (dA, vs[0], vs[1], vs[2], dB, vs[3], vs[4], vs[5], tuple(vs[6:]))

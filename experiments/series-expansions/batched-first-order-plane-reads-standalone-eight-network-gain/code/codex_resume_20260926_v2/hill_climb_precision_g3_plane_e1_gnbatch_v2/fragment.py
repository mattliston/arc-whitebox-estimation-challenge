# E1 GN channel batching: cached basic views; no contraction/reduction fusion.
_E1GN_CACHE = {}


def _e1gn_state(st, n):
    arena = _SCRATCH[0]
    pair = _E2_PAIR_ARENA.get(n)
    assert pair is not None and pair[0] is arena, 'GN requires completed current pair scratch'
    FT, PB = st['FT'], st['P16']
    assert tuple(FT.shape) == (7, 3, n) and tuple(PB.shape) == (3, n, n)
    assert str(FT.dtype) == str(PB.dtype) == str(arena.dtype) == 'float32'
    assert int(arena.shape[0]) >= 72 * n * n
    got = _E1GN_CACHE.get(n)
    if got is not None and got[0] is arena and got[1] is FT and got[2] is PB:
        return got[3]
    nn = n * n
    D = fnp.reshape(arena[:2 * nn], (2, n, n))
    B = fnp.reshape(arena[2 * nn:4 * nn], (2, n, n))
    H = fnp.reshape(arena[4 * nn:6 * nn], (2, n, n))
    D0, D1 = fnp.unstack(D)
    g = dict(D=D, D0=D0, D1=D1, D1T=D1.T, B=B, B0=B[0], H=H,
             left0=FT[2:4, 0, :, None], right0=FT[1, 0, None, :],
             left1=FT[3:5, 0, :, None], right1=FT[2, 0, None, :],
             rhoT=PB[0].T,
             divisors=fnp.asarray([2.0, 6.0], dtype=fnp.float32)[:, None, None])
    _E1GN_CACHE[n] = (arena, FT, PB, g)
    return g


def _e1gn_contract(g, rho, hold):
    hold(fnp.multiply(g['left0'], g['right0'], out=g['B']))
    hold(fnp.multiply(g['left1'], g['right1'], out=g['H']))
    hold(fnp.multiply(rho, g['H'], out=g['H']))
    hold(fnp.add(g['B'], g['H'], out=g['B']))
    hold(fnp.multiply(g['D'], g['B'], out=g['D']))
    hold(fnp.divide(g['D'], g['divisors'], out=g['D']))
    hold(fnp.add(g['D0'], g['D1'], out=g['B0']))
    return g['B0']


_E1GN_PARENT_RESET = _pb_reset


def _pb_reset(saved):
    try:
        return _E1GN_PARENT_RESET(saved)
    finally:
        _E1GN_CACHE.clear()

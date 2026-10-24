# UNVALIDATED: source-only first-order pair-local factorization.
# Mathematical equivalence does not imply byte-identical float32 outputs.
# Must be appended only to a source whose last _psp_series matches bc96f4d6.
# No network evaluation, integration or submission is authorized by this file.
_PLF_ENABLED = True
_PLF_PARENT_SERIES = _psp_series
_PLF_STATS = dict(off_calls=0, on_calls=0, Q=0, H=0, local_flop_credit=0,
                  persistent_calls=0, gathered_calls=0)



def _psp_series(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=False):
    if not _PLF_ENABLED:
        _PLF_STATS['off_calls'] += 1
        return _PLF_PARENT_SERIES(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold,
                                  persistent_fg=persistent_fg)
    # Scalar shape accounting only: no tensor/view/context retained.
    q = 1
    for dim in a['Z'].shape[1:]:
        q *= int(dim)
    h = 1
    for dim in F.shape[1:]:
        h *= int(dim)
    _PLF_STATS['on_calls'] += 1
    _PLF_STATS['Q'] += q
    _PLF_STATS['H'] += h
    _PLF_STATS['local_flop_credit'] += 21*q + 50*h
    _PLF_STATS['persistent_calls' if persistent_fg else 'gathered_calls'] += 1
    Z = a['Z']
    z0, z1, z2, z3, _ = _rqd_get('Z5', Z, lambda: fnp.unstack(Z))
    # Cache ONLY shape views of persistent owners. Their values change every
    # layer. The derived dF, dG and corrected f are always recomputed.
    if persistent_fg:
        f12, f3 = _rqd_get('PLF_F123', F, lambda: (F[1:3], F[3]))
        g12, g3 = _rqd_get('PLF_G123', G, lambda: (G[1:3], G[3]))
        f34 = _rqd_get('PLF_F34', F5, lambda: (F5[1, 4:6], F5[2, 5:7]))
        g34 = _rqd_get('PLF_G34', G5, lambda: (G5[1, 4:6], G5[2, 5:7]))
    else:
        f12, f3, g12, g3 = F[1:3], F[3], G[1:3], G[3]
        f34 = F5[1, 4:6], F5[2, 5:7]
        g34 = G5[1, 4:6], G5[2, 5:7]
    dF = hold(hold(f34[0] / 6) + hold(f34[1] / 24))
    dG = hold(hold(g34[0] / 6) + hold(g34[1] / 24))
    f = hold(f12 + dF)
    X = a['X3']
    x0, x1, r3 = _rqd_get('PLF_X3', X, lambda: (X[0], X[1], X[1, 0]))
    hold(fnp.multiply(f, g12, out=x0))
    hold(fnp.multiply(f12, dG, out=x1))
    hold(fnp.add(x0, x1, out=x0))
    hold(fnp.multiply(x0, P['P12'], out=x0))
    hold(fnp.sum(x0, axis=0, out=z0))
    # x1 is dead after the directional sum, so its first plane is free scratch.
    hold(fnp.multiply(f3, g3, out=r3))
    p3 = _rqd_get('PLF_P3', P['P6'], lambda: P['P6'][2])
    hold(fnp.multiply(r3, p3, out=r3))
    hold(fnp.add(z0, r3, out=z0))
    # Exact original mixed-cumulant products, order and divisors.
    X = a['X2']
    hold(fnp.multiply(F[[[2, 3], [1, 2], [2, 3], [3, 4], [1, 2]]],
                      G[[[1, 2], [2, 3], [2, 3], [1, 2], [3, 4]]], out=X))
    hold(fnp.multiply(a['X2r1'], a['RR1'], out=a['X2r1']))
    T = a['T']
    hold(fnp.sum(X, axis=1, out=T))
    t1, t2, t3, t4, t5 = _rqd_get('T5', T, lambda: fnp.unstack(T))
    hold(fnp.multiply(t21, t1, out=z1)); hold(fnp.add(z1, hold(t12 * t2), out=z1))
    hold(fnp.multiply(t22, t3, out=z2))
    hold(fnp.multiply(t31, t4, out=z3)); hold(fnp.add(z3, hold(t13 * t5), out=z3))
    z13, z04 = _rqd_get('PLF_Z', Z, lambda: (Z[1:4], Z[:4]))
    div = _rqd_get('PLF_DIV', a['div'], lambda: a['div'][:3])
    hold(fnp.divide(z13, div, out=z13))
    return fnp.sum(z04, axis=0)

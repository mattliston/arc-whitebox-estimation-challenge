# HELD prospective fixed Gaussian R3 -> R4 restoration on current4e.
# Existing mean/scalar moments/diagonal overrides, mixed terms and GTR intact.
_G4R_ON=True
_G4R_PARENT=_psp_series
_G4R_STATS=dict(calls=0,Q=0,layers=[])
assert _GTR_ENABLED and _ASTRA_IOP_ON is False

def _g4r_series(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=False):
    if not _G4R_ON:
        return _G4R_PARENT(F,G,F5,G5,P,t21,t12,t22,t31,t13,a,hold,persistent_fg=persistent_fg)
    if not _GTR_ENABLED:
        return _GTR_PARENT_SERIES(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=persistent_fg)
    Z = a['Z']
    z0, z1, z2, z3, z4 = _rqd_get('Z5', Z, lambda: fnp.unstack(Z))
    X = a['X1']
    if persistent_fg:
        F13, G13 = _rqd_get('F13', F, lambda: F[1:4]), _rqd_get('G13', G, lambda: G[1:4])
    else:
        F13, G13 = F[1:4], G[1:4]
    # Both consumers originally round the same F2*G2 and F3*G3.
    # X2[2] is disjoint from X1 in both actual BA and packed arenas.
    X2 = a['X2']
    # Basic slices are free, call-local views; add no global cache entries.
    raw22, x21, x23 = X2[2], X2[:2], X2[3:]
    x10, x112 = X[:1], X[1:3]
    p10, p112 = P['P6'][:1], P['P6'][1:3]
    f10, f112, g10, g112 = F13[:1], F13[1:3], G13[:1], G13[1:3]
    hold(fnp.multiply(f112, g112, out=raw22))
    hold(fnp.multiply(f10, g10, out=x10))
    hold(fnp.multiply(x10, p10, out=x10))
    hold(fnp.multiply(raw22, p112, out=x112))
    hold(fnp.sum(X, axis=0, out=z0))
    # First omitted Gaussian term, same normalized derivative tables.
    # Original R3 sum is complete; X1[0] is dead and disjoint from raw22.
    r4 = a['X1'][0]
    hold(fnp.multiply(F[4], G[4], out=r4))
    hold(fnp.multiply(r4, P['P6'][2], out=r4))
    hold(fnp.multiply(r4, P['P6'][0], out=r4))
    hold(fnp.divide(r4, 4.0, out=r4))
    hold(fnp.add(z0, r4, out=z0))
    _G4R_STATS['layers'].append(int(_PHYS_LAYER[0]))
    _G4R_STATS['Q'] += int(r4.size)
    _G4R_STATS['calls'] += 1
    # Gaussian is fully consumed before the overlapping X2 rows are written.
    X = X2
    hold(fnp.multiply(F[[[2, 3], [1, 2]]], G[[[1, 2], [2, 3]]], out=x21))
    hold(fnp.multiply(F[[[3, 4], [1, 2]]], G[[[1, 2], [3, 4]]], out=x23))
    hold(fnp.multiply(a['X2r1'], a['RR1'], out=a['X2r1']))
    T = a['T']
    hold(fnp.sum(X, axis=1, out=T))
    t1, t2, t3, t4, t5 = _rqd_get('T5', T, lambda: fnp.unstack(T))
    X = a['X3']
    hold(fnp.multiply(F5[[[1, 1], [0, 0], [2, 2], [0, 0]], [[4, 5], [1, 2], [5, 6], [1, 2]]],
                      G5[[[0, 0], [1, 1], [0, 0], [2, 2]], [[1, 2], [4, 5], [1, 2], [5, 6]]], out=X))
    hold(fnp.multiply(X, P['P12'], out=X))
    U = a['U']
    hold(fnp.sum(X, axis=1, out=U))
    t6, t7, t8, t9 = _rqd_get('U4', U, lambda: fnp.unstack(U))
    hold(fnp.multiply(t21, t1, out=z1)); hold(fnp.add(z1, hold(t12 * t2), out=z1))
    hold(fnp.multiply(t22, t3, out=z2))
    hold(fnp.multiply(t31, t4, out=z3)); hold(fnp.add(z3, hold(t13 * t5), out=z3))
    hold(fnp.add(z3, hold(t6 + t7), out=z3))
    hold(fnp.add(t8, t9, out=z4))
    Z15 = _rqd_get('Z15', Z, lambda: Z[1:5])
    hold(fnp.divide(Z15, a['div'], out=Z15))
    q = 1
    for dimension in Z.shape[1:]:q *= int(dimension)
    h = 1
    for dimension in F.shape[1:]:h *= int(dimension)
    _GTR_STATS['calls'] += 1
    _GTR_STATS['Q'] += q
    _GTR_STATS['H'] += h
    _GTR_STATS['credit'] += 2*q + 16*h
    _GTR_STATS['persistent_calls' if persistent_fg else 'gathered_calls'] += 1
    return fnp.sum(Z, axis=0)

_psp_series=_g4r_series

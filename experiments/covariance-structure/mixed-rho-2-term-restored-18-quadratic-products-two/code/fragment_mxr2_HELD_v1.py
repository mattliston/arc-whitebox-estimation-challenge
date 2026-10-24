# Fixed first-order mixed-cumulant rho^2 restoration, strength1, L1..14.
# Not Gaussian R4, not second-order-in-cumulant Edgeworth, no learned changes.
_MXR2_PARENT_SERIES = _psp_series
_MXR2_PARENT_PREDICT = _predict_binary
_MXR2_ON = True
_MXR2_STATS = dict(calls=0, Q=0, layers=[], retired=False)
def _mxr2_series(F, G, F5, G5, P, t21, t12, t22, t31, t13, a, hold, persistent_fg=False):
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
    # All original Gaussian/mixed/local products have been consumed into Z.
    # T's first two rows are now dead, disjoint from every Z row in BA/packed.
    # P12[1] already owns rho^2/2; the original /2,/4,/6 below remain intact.
    u, v = T[0], T[1]
    p2 = P['P12'][1]
    f3, f4, f5 = F[3], F[4], F[5]
    g3, g4, g5 = G[3], G[4], G[5]
    hold(fnp.multiply(f4, g3, out=u)); hold(fnp.multiply(u, t21, out=u))
    hold(fnp.multiply(f3, g4, out=v)); hold(fnp.multiply(v, t12, out=v))
    hold(fnp.add(u, v, out=u)); hold(fnp.multiply(u, p2, out=u)); hold(fnp.add(z1, u, out=z1))
    hold(fnp.multiply(f4, g4, out=u)); hold(fnp.multiply(u, t22, out=u))
    hold(fnp.multiply(u, p2, out=u)); hold(fnp.add(z2, u, out=z2))
    hold(fnp.multiply(f5, g3, out=u)); hold(fnp.multiply(u, t31, out=u))
    hold(fnp.multiply(f3, g5, out=v)); hold(fnp.multiply(v, t13, out=v))
    hold(fnp.add(u, v, out=u)); hold(fnp.multiply(u, p2, out=u)); hold(fnp.add(z3, u, out=z3))
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

def _psp_series(F,G,F5,G5,P,t21,t12,t22,t31,t13,a,hold,persistent_fg=False):
    layer = int(_PHYS_LAYER[0])
    if not _MXR2_ON or not 1 <= layer <= 14:
        return _MXR2_PARENT_SERIES(F,G,F5,G5,P,t21,t12,t22,t31,t13,a,hold,persistent_fg=persistent_fg)
    assert _GTR_ENABLED
    out = _mxr2_series(F,G,F5,G5,P,t21,t12,t22,t31,t13,a,hold,persistent_fg=persistent_fg)
    q=1
    for size in a['Z'].shape[1:]:q*=int(size)
    _MXR2_STATS['calls']+=1;_MXR2_STATS['Q']+=q
    if layer not in _MXR2_STATS['layers']:_MXR2_STATS['layers'].append(layer)
    return out

def _predict_binary(*args,**kwargs):
    global _MXR2_STATS
    _MXR2_STATS=dict(calls=0,Q=0,layers=[],retired=False)
    try:return _MXR2_PARENT_PREDICT(*args,**kwargs)
    finally:_MXR2_STATS['retired']=True

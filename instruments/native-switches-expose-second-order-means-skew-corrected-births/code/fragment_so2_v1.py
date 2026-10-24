# BEGIN OPUS-ENGINEER SECOND-ORDER MEAN TERM (a41 so2 v1, 2026-10-02; a physics switch, OFF by default: at strength 0 the pair
# map's outputs are returned untouched). Native form of the per-unit part of the physicist's O/physicist/so2.py (install(mean=, raw2=)):
# at the order of the fourth cumulants the mean of a rectified unit also has the product of two third cumulants,
#     mean += c t3^2 sigma He4(alpha) phi / 72,      E relu^2 += r (- t3^2 v He3(alpha) phi / 36),      t3 = k3 / sigma^3,
# INSIDE the pair map at layers _SO2_LO..min(_SO2_HI, 14): the pair map's mean becomes m + c m2 (the host of everything after it:
# beta, the omitted radial source, the learned correction's features), and the per-unit variance follows consistently,
#     var = (r2 + r q2) - (m + c m2)^2,   r2 = (mu^2 + v) g + mu sigma phi + k3 phi / (3 sigma) - k4 alpha phi / (12 v)   (the card's own raw moment),
# written on the diagonal of the pair map's covariance and into aux['post_v']. Nothing else of the pair map changes (the third-cumulant
# target and the off-diagonal covariance keep the first-order mean, as in the reference).
# One polynomial evaluation on nine per-unit rows (a gather by a constant index table, a product over the degree axis, a product with
# a constant coefficient matrix that holds the two strengths), float64: 10 requests a layer, about 1 Mflop.
_SO2_MEAN = 0.0          # c: strength of the mean term
_SO2_RAW2 = None         # r: strength of the E relu^2 term; None = c (the consistent per-unit variance)
_SO2_LO, _SO2_HI = 1, 15
_SO2_CONST = {}
_SO2_STATS = dict(layers=0)
_SO2_PARENT = nonlinear


def _so2_tables():
    """(index table: degree x terms, coefficient matrix: 2 x terms) for c m2 and r2 + r q2 on the rows
    (one, mu, s, 1/s, alpha, pdf, gate, k3, k4). Plain Python."""
    names = ['one', 'mu', 's', 'is', 'a', 'ph', 'g', 'k3', 'k4']
    c = float(_SO2_MEAN)
    r = c if _SO2_RAW2 is None else float(_SO2_RAW2)
    cm2 = {('k3', 'k3') + ('is',) * 5 + ('a',) * 4 + ('ph',): c / 72.0, ('k3', 'k3') + ('is',) * 5 + ('a',) * 2 + ('ph',): -6.0 * c / 72.0,
           ('k3', 'k3') + ('is',) * 5 + ('ph',): 3.0 * c / 72.0}                                                    # c k3^2 He4(a) phi / (72 s^5)
    r2c = {('mu', 'mu', 'g'): 1.0, ('s', 's', 'g'): 1.0, ('mu', 's', 'ph'): 1.0, ('k3', 'ph', 'is'): 1.0 / 3.0,
           ('k4', 'a', 'ph', 'is', 'is'): -1.0 / 12.0,                                                             # the card's r2
           ('k3', 'k3') + ('is',) * 4 + ('a',) * 3 + ('ph',): -r / 36.0, ('k3', 'k3') + ('is',) * 4 + ('a',) + ('ph',): 3.0 * r / 36.0}   # r q2
    terms = sorted(set(cm2) | set(r2c), key=lambda k: (len(k), k))
    degree = max(len(k) for k in terms)
    pos = {name: i for i, name in enumerate(names)}
    index = [[(pos[k[j]] if j < len(k) else pos['one']) for k in terms] for j in range(degree)]
    return index, [[p.get(k, 0.0) for k in terms] for p in (cm2, r2c)]


def _so2_pair(mu, C, K21, k4, k22_override=None, metric='identity', extra31=None):
    out = _SO2_PARENT(mu, C, K21, k4, k22_override, metric) if extra31 is None else _SO2_PARENT(mu, C, K21, k4, k22_override, metric, extra31=extra31)
    l = _PHYS_LAYER[0]
    if not (_SO2_MEAN or _SO2_RAW2) or l is None or not (_SO2_LO <= l <= min(_SO2_HI, 14)):
        return out
    hold = _E2Hold()   # temporaries of this call, released together
    m, cov, post21, radial, aux = out
    S = _SO2_CONST
    n = int(mu.shape[0])
    if S.get('n') != n:
        index, coefficient = _so2_tables()
        S.update(n=n, index=fnp.asarray(index), coefficient=fnp.asarray(coefficient, dtype=fnp.float64), one=fnp.ones(n, dtype=fnp.float64))
    s = aux['sigma']
    rows = hold(fnp.stack([S['one'], mu, s, hold(fnp.power(s, -1.0)), aux['alpha'], aux['pdf'], aux['gate'], aux['pre_k3'], aux['pre_k4']]))
    cm2, r2c = hold(fnp.unstack(hold(S['coefficient'] @ hold(fnp.prod(hold(rows[S['index']]), axis=0)))))
    mean = m + cm2
    var = r2c - hold(mean * mean)
    fnp.fill_diagonal(cov, var)
    aux['post_v'] = var
    pair = _E2_LAYER.get('pair')
    if pair is not None and pair.get('cov') is cov:   # the cached denominator was made with the old diagonal: the live loop recomputes it from cov
        _E2_LAYER['pair'] = dict(pair, cov=None)
    _SO2_STATS['layers'] += 1
    return (mean, cov, post21, radial, aux)


nonlinear = _so2_pair
# END OPUS-ENGINEER SECOND-ORDER MEAN TERM

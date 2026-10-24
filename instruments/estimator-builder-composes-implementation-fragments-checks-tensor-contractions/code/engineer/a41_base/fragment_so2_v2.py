# BEGIN OPUS-ENGINEER SECOND-ORDER TERMS (a41 so2 v2, 2026-10-02; physics switches, ALL OFF by default: at strength 0 the pair map's
# outputs are returned untouched). Native form of the so2 keywords of the physicist's base module O/physicist/pbase_frozen_s4a.py
# (so2m, so2t, so2k, so2p, so2td, so2f; so2.py before it): at the order of the fourth cumulants the moments of a rectified unit also
# hold the products of two third cumulants.
#   MEAN:   mean += c t3^2 sigma He4(alpha) phi / 72,   E relu^2 += r (- t3^2 v He3(alpha) phi / 36),   t3 = k3 / sigma^3,
#     INSIDE the pair map at layers _SO2_LO..min(_SO2_HI, 14); the per-unit variance follows: var = (r2 + r q2) - (m + c m2)^2, written on
#     the diagonal of the covariance and into aux['post_v']. With the target path off this is v1's code line for line.
#   TARGET:   the pair map's third-cumulant target cum(f_i, f_i, f_j) += t * dt, its diagonal (post_k3) += t * (pk3 - pk3_0), with
#       dt = dp1 v_i sigma_j - 2 (m1_i (off + dc1) - m_i off),   m1 = m + m2 (always the full second-order mean, as in the reference),
#     dc1 / dp1 = the reference's second-order lists of the covariance / of E f_i^2 f_j (parts a b q c d e, powers of rho up to kmax).
#     _SO2_TDIAG: the diagonal alone (so2td), no plane and no row term.
#     LEAN FORM: the target needs only the combination, and the two lists differ only in their left rows, so it is ONE list with
#       U_r = v B_r - 2 m1 sigma A_r = 2 (v A_(r-1) - m1 sigma A_r)   (left),     W_q = sigma A_q   (right),
#       dt = T o [(M_g0 + rho o M_g1) + T o (M_gq0 + rho o M_gq1)] + T' o [(M_h0 + rho o M_h1) + T' o (M_hq0 + rho o M_hq1)]
#            + (T o T') o (M_c0 + rho o M_c1) + rho o M_de - 2 m2_i off_ij,
#     T = K21 / (v_i sigma_j), its transposed view T' and rho = the pair map's own float32 planes of this call (st['T21'], st['T21T'],
#     st['P16'][0]); every M = an outer-product sum of rank 1-3 of per-unit rows. The rows come from two small float32 monomial tables
#     (a gather from the base rows, a product over the degree axis, a sum over two slots; the coefficients are constant base rows) on
#     the pair map's own derivative rows A0..A6 (st['FT'][:, 0]) and t3 (st['t3']). With all parts and kmax 1, per layer: 45 float32
#     plane passes, 5 for the row term and the write, 2 for the tables; 48 requests (53 with the mean term).
#     WRITTEN: the pair map's stored float32 copy of the target (the plane the live loop reads when _FULL_LOWER_MOMENTS is on) and the
#     DIAGONAL of the float64 target; the float64 plane's off-diagonal part is not read by that loop and stays as the pair map made it
#     (asserted below).
#   FINAL:   at layer 15 the mean term f t3^2 sigma He4(alpha) phi / 72 is added to the learned correction (one loop line).
_SO2_MEAN = 0.0          # c: strength of the mean term
_SO2_RAW2 = None         # r: strength of the E relu^2 term; None = c (the consistent per-unit variance)
_SO2_TGT = 0.0           # t: strength of the second-order part of the third-cumulant target
_SO2_KMAX = 1            # highest power of rho kept in the target's list (0 or 1)
_SO2_PARTS = 'abqcde'    # the parts of the target's list
_SO2_TDIAG = False       # True: the target's diagonal alone
_SO2_FINAL = 0.0         # f: strength of the mean term at the final layer
_SO2_LO, _SO2_HI = 1, 15
_SO2_CONST = {}
_SO2_STATS = dict(layers=0, target_layers=0, final=0)
_SO2_PARENT = nonlinear
assert _FULL_LOWER_MOMENTS or _SO2_TDIAG or not _SO2_TGT, 'so2 target path: this loop would read the float64 target off the diagonal, which the path does not write'


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


def _so2_target_tables():
    """The two monomial tables of the target path and the layout of their rows. Plain Python.
    Base rows: one, s, t3, m, var0, alpha | A0..A6 (A0 = Gaussian mean / s, A1 = gate, A_r = (-1)^r He_(r-2)(alpha) pdf) | the constants
    (zero, then every coefficient other than 1) | the first table's outputs m1, m2, q2, q3, r2, t3^2, v, m1 s, A7.
    A row = at most two (coefficient, factors); an unused second slot is the zero row."""
    t, kmax, parts = float(_SO2_TGT), int(_SO2_KMAX), ('' if _SO2_TDIAG else _SO2_PARTS)
    c = float(_SO2_MEAN)
    r = c if _SO2_RAW2 is None else float(_SO2_RAW2)
    assert kmax in (0, 1) and set(parts) <= set('abqcde')
    out1 = [('m1', [(1.0, ('m',)), (1.0 / 72.0, ('t3', 't3', 's', 'A6'))]),
            ('m2', [(1.0 / 72.0, ('t3', 't3', 's', 'A6'))]),
            ('q2', [(1.0 / 36.0, ('t3', 't3', 's', 's', 'A5'))]),
            ('q3', [(1.0 / 12.0, ('t3', 't3', 's', 's', 's', 'A4'))]),
            ('r2', [(1.0, ('var0',)), (1.0, ('m', 'm'))]),
            ('t3t3', [(1.0, ('t3', 't3'))]),
            ('v', [(1.0, ('s', 's'))]),
            ('m1s', [(1.0, ('m', 's')), (1.0 / 72.0, ('t3', 't3', 's', 's', 'A6'))]),
            ('A7', [(-1.0, ('a', 'A6')), (-4.0, ('A5',))])]                              # - He5 pdf = - alpha A6 - 4 A5
    tp = ((), ('t3',), ('t3t3',))
    U = lambda q, p=0: [(2.0, tp[p] + ('v', 'A%d' % (q - 1))), (-2.0, tp[p] + ('m1s', 'A%d' % q))]   # t3^p (v B_q - 2 m1 s A_q)
    Wq = lambda q, p=0: [(1.0, tp[p] + ('s', 'A%d' % q))]                                            # t3^p s A_q
    scale = lambda x, row: [(x * y, f) for y, f in row]
    c6, c9, c18, c2, c1 = t / 12.0, t / 8.0, t / 4.0, t / 36.0, t / 72.0
    mats = {}
    add = lambda name, left, right: mats.setdefault(name, []).append((left, right))
    if 'a' in parts:
        add('g0', scale(c6, U(5, 1)), Wq(1)); add('h0', scale(c6, U(1)), Wq(5, 1))
        if kmax:
            add('g1', scale(c6, U(6, 1)), Wq(2)); add('h1', scale(c6, U(2)), Wq(6, 1))
    if 'b' in parts:
        add('g0', scale(c6, U(2)), Wq(4, 1)); add('h0', scale(c6, U(4, 1)), Wq(2))
        if kmax:
            add('g1', scale(c6, U(3)), Wq(5, 1)); add('h1', scale(c6, U(5, 1)), Wq(3))
    if 'q' in parts:
        add('gq0', scale(c9, U(4)), Wq(2)); add('hq0', scale(c9, U(2)), Wq(4))
        if kmax:
            add('gq1', scale(c9, U(5)), Wq(3)); add('hq1', scale(c9, U(3)), Wq(5))
    if 'c' in parts:
        add('c0', scale(c18, U(3)), Wq(3))
        if kmax:
            add('c1', scale(c18, U(4)), Wq(4))
    if kmax and 'd' in parts:
        add('de', scale(c2, U(4, 1)), Wq(4, 1))
    if kmax and 'e' in parts:
        add('de', scale(c1, U(7, 2)), Wq(1)); add('de', scale(c1, U(1)), Wq(7, 2))
    order = [k for k in ('g0', 'g1', 'gq0', 'gq1', 'h0', 'h1', 'hq0', 'hq1', 'c0', 'c1', 'de') if k in mats]
    ranks = [len(mats[k]) for k in order]
    out2 = [left for k in order for left, _ in mats[k]] + [right for k in order for _, right in mats[k]]
    cuts, at = [], 0
    for x in ranks + ranks:
        at += x
        cuts.append(at)
    out2 += [[(t, ('q3',)), (-3.0 * t, ('m2', 'r2'))], [(-3.0 * t, ('m1', 'q2')), (2.0 * t, ('m2', 'm1', 'm1'))],
             [(2.0 * t, ('m2', 'm1', 'm')), (2.0 * t, ('m2', 'm', 'm'))]]          # t (pk3 - pk3_0) = t (q3 - 3 (m2 r2 + m1 q2) + 2 m2 (m1^2 + m1 m + m^2))
    cuts.append(at + 3)
    blocks = ['dk3']
    if not _SO2_TDIAG:
        out2.append([(-2.0 * t, ('m2',))]); cuts.append(cuts[-1] + 1); blocks.append('cm')       # the row factor of - 2 t m2_i off_ij
    if c or r:
        out2 += [[(c, ('m2',))], [(r, ('q2',)), (-2.0 * c, ('m', 'm2'))], [(-c * c, ('m2', 'm2'))]]   # c m2;  var - var0 = r q2 - 2 c m m2 - c^2 m2^2
        cuts += [cuts[-1] + 1, cuts[-1] + 3]; blocks += ['cmean', 'dvar']
    consts = [0.0] + sorted(({float(x) for _, row in out1 for x, _ in row} | {float(x) for row in out2 for x, _ in row}) - {1.0, 0.0})
    names = ['one', 's', 't3', 'm', 'var0', 'a'] + ['A%d' % q for q in range(7)] + [('c', x) for x in consts]
    names2 = names + [k for k, _ in out1]

    def table(rows, names_):
        pos = {name: i for i, name in enumerate(names_)}
        full = [[((('c', float(x) + 0.0),) if float(x) != 1.0 else ()) + tuple(f) for x, f in row] for row in rows]   # the coefficient as one more factor
        degree = max(len(f) for row in full for f in row)
        index = [[[pos['one'], pos['one']] for _ in rows] for _ in range(degree)]
        for i, row in enumerate(full):
            assert 1 <= len(row) <= 2
            for k in (0, 1):
                factors = row[k] if k < len(row) else (('c', 0.0),)      # an unused slot is the zero row
                for j, name in enumerate(factors):
                    index[j][i][k] = pos[name]
        return index
    return dict(i1=table([row for _, row in out1], names), i2=table(out2, names2), consts=consts, order=order, ranks=dict(zip(order, ranks)),
                cuts=cuts[:-1], blocks=blocks, mean=bool(c or r))


def _so2_target(m, cov, post21, aux, n):
    """The stored float32 copy of the target takes t * dt off the diagonal; both copies take t * (pk3 - pk3_0) on the diagonal;
    aux['post_k3'] follows. Returns (mean, variance) of the mean term when that is on, else (None, None)."""
    hold = _E2Hold()   # temporaries of this call, released together
    S = _SO2_CONST
    if S.get('tn') != n:
        tab = _so2_target_tables()
        S.update(tn=n, tone=fnp.ones(n, dtype=fnp.float32), crow=fnp.asarray([[x] * n for x in tab['consts']], dtype=fnp.float32),
                 i1=fnp.asarray(tab['i1']), i2=fnp.asarray(tab['i2']), order=tab['order'], ranks=tab['ranks'], cuts=tab['cuts'], blocks=tab['blocks'], tmean=tab['mean'])
    st = _e2_pair_state(n)
    if S.get('ft') is not st['FT']:      # views of the pair map's persistent buffers, taken once per process
        S.update(ft=st['FT'], A=st['FT'][:, 0], rhos=fnp.unstack(st['P16']))
    base = hold(fnp.concatenate((hold(fnp.stack((S['tone'], aux['sigma'], st['t3'], m, aux['post_v'], aux['alpha']), dtype=fnp.float32)), S['A'], S['crow'])))
    first = hold(fnp.sum(hold(fnp.prod(hold(base[S['i1']]), axis=0)), axis=1))                 # m1, m2, q2, q3, r2, t3^2, v, m1 s, A7
    second = hold(fnp.sum(hold(fnp.prod(hold(hold(fnp.concatenate((base, first)))[S['i2']]), axis=0)), axis=1))
    order = S['order']
    blocks = hold(fnp.split(second, S['cuts']))
    named = dict(zip(S['blocks'], blocks[2 * len(order):]))
    B = {k: (blocks[i], blocks[len(order) + i]) for i, k in enumerate(order)}
    if not _SO2_TDIAG:
        T, Tt, rho = st['T21'], st['T21T'], S['rhos'][0]
        outer = lambda k: fnp.einsum('ri,rj->ij', B[k][0], B[k][1], optimize=False)
        weighted = lambda P, k: fnp.einsum('ij,ri,rj->ij', P, B[k][0], B[k][1], optimize=False)

        def plus(x, y):
            if x is None:
                return y
            hold(fnp.add(x, hold(y), out=x))
            return x

        def times(x, P):
            hold(fnp.multiply(x, P, out=x))
            return x

        def nested(P, k0, k1, q0, q1):
            """P o [(M_k0 + rho o M_k1) + P o (M_q0 + rho o M_q1)]"""
            x = weighted(rho, q1) if q1 in B else None
            if q0 in B:
                x = plus(x, outer(q0))
            if x is not None:
                times(x, P)
            if k1 in B:
                x = plus(x, weighted(rho, k1) if S['ranks'][k1] == 1 else times(outer(k1), rho))
            if k0 in B:
                x = plus(x, outer(k0))
            if x is not None:
                times(x, P)
            return x

        dt = nested(T, 'g0', 'g1', 'gq0', 'gq1')
        h = nested(Tt, 'h0', 'h1', 'hq0', 'hq1')
        if h is not None:
            dt = plus(dt, h)
        if 'c0' in B:
            TTt = hold(T * Tt)
            dt = plus(dt, times(plus(weighted(rho, 'c1'), outer('c0')), TTt) if 'c1' in B else weighted(TTt, 'c0'))
        if 'de' in B:
            dt = plus(dt, times(outer('de'), rho))
        t32 = _E2_LAYER['ps_target32']
        assert _E2_LAYER.get('ps_target64') is post21 and str(t32.dtype) == 'float32'
        if dt is not None:
            hold(fnp.add(t32, hold(dt), out=t32))
        hold(fnp.add(t32, hold(fnp.einsum('ri,ij->ij', named['cm'], cov, optimize=False)), out=t32))      # - 2 t m2_i cov_ij (the diagonal is rewritten below)
    post_k3 = aux['post_k3'] + hold(fnp.sum(named['dk3'], axis=0))
    fnp.fill_diagonal(post21, post_k3)
    if _E2_LAYER.get('ps_target64') is post21:
        fnp.fill_diagonal(_E2_LAYER['ps_target32'], hold(post_k3.astype(fnp.float32)))
    aux['post_k3'] = post_k3
    _SO2_STATS['target_layers'] += 1
    if not S['tmean']:
        return None, None
    return m + hold(fnp.sum(named['cmean'], axis=0)), aux['post_v'] + hold(fnp.sum(named['dvar'], axis=0))


def _so2_pair(mu, C, K21, k4, k22_override=None, metric='identity', extra31=None):
    out = _SO2_PARENT(mu, C, K21, k4, k22_override, metric) if extra31 is None else _SO2_PARENT(mu, C, K21, k4, k22_override, metric, extra31=extra31)
    l = _PHYS_LAYER[0]
    if not (_SO2_MEAN or _SO2_RAW2 or _SO2_TGT) or l is None or not (_SO2_LO <= l <= min(_SO2_HI, 14)):
        return out
    m, cov, post21, radial, aux = out
    n = int(mu.shape[0])
    if _SO2_TGT:      # the target path also makes the mean term's rows (the same per-unit quantities)
        mean, var = _so2_target(m, cov, post21, aux, n)
        if mean is None:
            return out
    else:
        hold = _E2Hold()   # temporaries of this call, released together
        S = _SO2_CONST
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


def _so2_final(delta, aux):
    """final layer: the learned correction plus f t3^2 sigma He4(alpha) phi / 72 = f k3^2 He4(alpha) phi / (72 sigma^5)."""
    if not _SO2_FINAL:
        return delta
    hold = _E2Hold()
    S = _SO2_CONST
    s = aux['sigma']
    n = int(s.shape[0])
    if S.get('fn') != n:
        c = float(_SO2_FINAL) / 72.0
        head = [3, 3] + [1] * 5 + [2]          # k3 k3 (1/s)^5 pdf on the rows (one, 1/s, pdf, k3, alpha)
        S.update(fn=n, fone=fnp.ones(n, dtype=fnp.float64), fc=fnp.asarray([[c], [-6.0 * c], [3.0 * c]], dtype=fnp.float64),
                 fi=fnp.asarray([[x, x, x] for x in head] + [[4, 4, 0], [4, 4, 0], [4, 0, 0], [4, 0, 0]]))
    rows = hold(fnp.stack((S['fone'], hold(fnp.power(s, -1.0)), aux['pdf'], aux['pre_k3'], aux['alpha'])))
    m2 = hold(fnp.sum(hold(hold(fnp.prod(hold(rows[S['fi']]), axis=0)) * S['fc']), axis=0))
    _SO2_STATS['final'] += 1
    return delta + m2


nonlinear = _so2_pair
# END OPUS-ENGINEER SECOND-ORDER TERMS

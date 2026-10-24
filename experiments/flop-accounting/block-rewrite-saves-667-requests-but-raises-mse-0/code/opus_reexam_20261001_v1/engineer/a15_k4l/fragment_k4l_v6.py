# BEGIN OPUS-ENGINEER K4L v6 (a15, 2026-10-02; v6 = v5 with the hub terms of ALL young cohorts in one block of operations: the +
# columns of the cohorts side by side, the - columns side by side (two block copies), then one Horner evaluation, 18 requests a
# layer instead of 37; a strength per age is folded into the cohort's coefficients at its birth (one stack per age, same
# operations for any strengths). Rounding level against v5; billed 16 flops per element of (rows x pairs) instead of 12.
# v5 = v4 with the birth coefficients from two polynomial tables: the same arithmetic
# in 15 requests a layer instead of 33, rounding level (float64, cast to float32 as before). v4 = v3 with the per-age multiplication ALWAYS made when _K4L_C is a tuple, also for
# a strength of exactly 1 (v3 skipped it): the operations then do not depend on the strengths' values, so two files that differ in
# them differ in float literals only (the release lane's fast path). v3 = v2 with a strength per pair age: _K4L_C may be a tuple (age 1, age 2, ...), each
# cohort's terms are multiplied by its strength before the level is projected out (the projection is linear, so this equals a
# strength per age on that age's own correction); a single number gives v2's arithmetic. v2 = v1's arithmetic with every temporary
# held to the end of the call and lower-case einsum subscripts): native port of the lead's K4L rule (O/K4L_SPEC.md, torch reference
# O/e16b_k4_closed_loop.py). An ERROR lever, not a rounding-level change: it moves the predictions by design.
# The card's per-unit fourth cumulant of the pre-activations is one number per layer, k4_i = radial v_i^2 (times the band dial).
# Every column pair of the bank was born at a hub h as p+- = s_h + t+- e_h, and the ReLU at the hub gives the post-activations a
# fourth cumulant with m legs on the hub and 4 - m on its covariance column, with exactly known coefficients S1..S4. At a later
# layer the pair's pushed columns p (+) and q (-) hold the two transported legs, Wt = (p - q) / (t+ - t-) and Y = p - t+ Wt, and the
# hub adds  4 S1 Y^3 Wt + 6 S2 Y^2 Wt^2 + 4 S3 Y Wt^3 + S4 Wt^4  to the fourth cumulant of every unit. Summed over the pairs born
# at most _K4L_AGE layers earlier this is loc_i; the level the card was trained with is kept:
#     beta = sum_i w_i loc_i v_i^2 / sum_i w_i v_i^4,   w_i = ((a_i^2 - 1) exp(-a_i^2 / 2))^2,  a = mu / sigma,
#     k4_i <- k4_i + c (loc_i - beta v_i^2)
# for the k4 the pair map receives at layers 1-14 and for the final layer's k4.
# Native form: the quartic is written on p and D = p - q,  D (e1 p^3 + D (e2 p^2 + D (e3 p + e4 D))), with four coefficients per
# pair fixed at its birth (the shift from Y to p and the powers of the gap are folded into them), and p^2, p^3 are the squares and
# cubes the live loop already forms: 12 flops per element of (rows x pairs), float32. Rows the native row cuts do not push have
# loc = 0 exactly; they get no correction and are left out of the beta sums (the torch reference has no row cuts).
_K4L_C = 1.0
_K4L_AGE = 2
class _K4LUnstacked(dict):
    def __missing__(self, l):                     # made only when somebody asks (the pair law's fragment up to v8 does)
        self[l] = fnp.unstack(_K4L_RAW[l])
        return self[l]


_K4L_B = _K4LUnstacked()                          # birth layer -> (e1, e2, e3, e4), float32, one entry per pair; this prediction
_K4L_RAW = {}                                     # birth layer -> the same four as one array (4, pairs)
_K4L_E = {}                                       # birth layer -> per age the array (4, pairs) times that age's strength (the same array when _K4L_C is one number)
_K4L_FINAL = [None]                               # (rows, local terms on those rows) left by the final layer's row-cut push
_K4L_CONST = {}
_K4L_STATS = dict(blocks=0, pairs=0, elements=0)


def _k4l_tables():
    """S1..S4 at the hub and the four birth coefficients as two polynomial tables (index: degree x terms, coefficient: outputs x
    terms; nested lists). Plain Python, at import: no array."""
    def add(*ps):
        out = {}
        for p in ps:
            for k, c in p.items():
                out[k] = out.get(k, 0.0) + c
        return {k: c for k, c in out.items() if c != 0.0}

    def mul(*ps):
        out = {(): 1.0}
        for p in ps:
            if not isinstance(p, dict):
                p = {(): float(p)}
            nxt = {}
            for k1, c1 in out.items():
                for k2, c2 in p.items():
                    k = tuple(sorted(k1 + k2))
                    nxt[k] = nxt.get(k, 0.0) + c1 * c2
            out = nxt
        return {k: c for k, c in out.items() if c != 0.0}

    def table(outputs, names):
        pos = {name: i for i, name in enumerate(names)}
        terms = sorted({k for p in outputs for k in p}, key=lambda k: (len(k), k))
        degree = max(len(k) for k in terms)
        index = [[(pos[k[j]] if j < len(k) else pos['one']) for k in terms] for j in range(degree)]
        return index, [[p.get(k, 0.0) for k in terms] for p in outputs]

    one = {(): 1.0}
    a, g, ph, m, tp, sg, isg, igap = ({(x,): 1.0} for x in ('a', 'g', 'ph', 'm', 'tp', 'sg', 'isg', 'igap'))
    # Gaussian moments of the rectified hub in units of sigma, and the host mean in units of sigma
    a2 = mul(a, a)
    rho2 = add(mul(add(one, a2), g), mul(a, ph))
    rho3 = add(mul(add(mul(3.0, a), mul(a2, a)), g), mul(add(mul(2.0, one), a2), ph))
    rho4 = add(mul(add(mul(3.0, one), mul(6.0, a2), mul(a2, a2)), g), mul(add(mul(5.0, a), mul(a2, a)), ph))
    mt = mul(m, isg)
    mt2 = mul(mt, mt)
    S = [mul(-1.0, a, ph),                                                                    # S_m / sigma^(2m-4), m = 1..4
         add(mul(2.0, g), mul(-2.0, g, g), mul(-2.0, mt, ph)),
         add(mul(3.0, rho2), mul(-6.0, mt2), mul(-3.0, g, rho2), mul(6.0, g, mt2)),
         add(rho4, mul(-4.0, mt, rho3), mul(12.0, mt2, rho2), mul(-6.0, mt2, mt2), mul(-3.0, rho2, rho2))]
    first = table([one, mul(-1.0, tp, isg, isg), mul(sg, sg, igap), mul(isg, isg, isg, isg)] + S,
                  ['one', 'a', 'g', 'ph', 'm', 'tp', 'sg', 'gap', 'isg', 'igap'])            # one, u = - t+ / v, w = v / gap, 1 / v^2, S1..S4
    # the shift Y = p - t+ Wt: coefficient of p^(4-M) Wt^M = binom(4, M) sum_m binom(M, m) (-t+ / v)^(M-m) S_m, times (v / gap)^M / v^2
    u, w, iv2 = ({(x,): 1.0} for x in ('u', 'w', 'iv2'))
    binom = ((1, 0, 0, 0), (2, 1, 0, 0), (3, 3, 1, 0), (4, 6, 4, 1))
    outputs = []
    for M in range(1, 5):
        poly = {}
        for mm in range(1, M + 1):
            poly = add(poly, mul(float((4, 6, 4, 1)[M - 1] * binom[M - 1][mm - 1]), *([u] * (M - mm) + [{('S%d' % mm,): 1.0}])))
        outputs.append(mul(poly, *([w] * M + [iv2])))
    return first + table(outputs, ['one', 'u', 'w', 'iv2', 'S1', 'S2', 'S3', 'S4'])


_K4L_TABLES = _k4l_tables()   # at import: plain Python


def _k4l_const():
    C = _K4L_CONST
    if not C:
        idxA, coefA, idxB, coefB = _K4L_TABLES
        if isinstance(_K4L_C, tuple):             # e1..e4, then e1..e4 times each age's strength
            coefB = coefB + [[c * x for x in row] for c in _K4L_C for row in coefB]
        C.update(zip(('idxA', 'coefA', 'idxB', 'coefB'), (fnp.asarray(x) for x in (idxA, coefA, idxB, coefB))))
    return C


def _k4l_birth(l, aux, host, selected, t, s):
    """The four coefficients of the pairs born at layer l. host: the pair map's mean (before the learned correction); t, s: the
    + and - hub entries of the newborn columns (n,). Two polynomial evaluations (a gather by a constant index table, a product
    over the degree axis, a product with a constant coefficient matrix each): v4's arithmetic in 15 requests instead of 33."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    C = _k4l_const()
    if l == 0:
        _K4L_B.clear(); _K4L_RAW.clear(); _K4L_E.clear()
        _K4L_STATS.update(blocks=0, pairs=0, elements=0)
    sigma = aux['sigma']
    if 'one' not in C or int(C['one'].shape[0]) != int(sigma.shape[0]):
        C['one'] = fnp.ones(int(sigma.shape[0]), dtype=fnp.float64)
    H = _e2_H(_e2_H(fnp.stack([C['one'], aux['alpha'], aux['gate'], aux['pdf'], host, t, sigma, _e2_H(t - s)]))[:, selected])
    H = _e2_H(fnp.concatenate([H, _e2_H(fnp.power(_e2_H(H[6:]), -1.0))]))                    # + 1 / sigma, 1 / gap
    A = _e2_H(C['coefA'] @ _e2_H(fnp.prod(_e2_H(H[C['idxA']]), axis=0)))                     # one, u, w, 1 / v^2, S1..S4
    E = _e2_H(C['coefB'] @ _e2_H(fnp.prod(_e2_H(A[C['idxB']]), axis=0))).astype(fnp.float32)
    if isinstance(_K4L_C, tuple):
        parts = fnp.split(E, 1 + len(_K4L_C))
        _K4L_RAW[l], _K4L_E[l] = parts[0], parts[1:]
    else:
        _K4L_RAW[l], _K4L_E[l] = E, [E] * _K4L_AGE


def _k4l_local(l, pp, pp2, pp3, state):
    """Sum over the pairs born at most _K4L_AGE layers before layer l of their hub terms, per row of pp (float32).
    pp: pushed bank columns (rows, all columns); pp2, pp3: its squares and cubes, or None (not used since v6: the squares and
    cubes of the + columns are formed on the block)."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    cohorts = []
    for j in range(max(l - _K4L_AGE, 0), l):
        start, width = state['cohorts'][j]
        start, K = int(start), int(width) // 2
        if K:
            cohorts.append((start, K, j))
    if not cohorts:
        return None
    cuts = sorted({c for start, K, _j in cohorts for c in (start, start + K, start + 2 * K)})
    pieces = _e2_H(fnp.split(pp, cuts, axis=1))      # views; pieces[i + 1] starts at cuts[i]
    at = {c: i + 1 for i, c in enumerate(cuts)}
    plus = [pieces[at[start]] for start, K, _j in cohorts]
    minus = [pieces[at[start + K]] for start, K, _j in cohorts]
    tables = [_K4L_E[j][l - j - 1] for _start, _K, j in cohorts]   # the cohort born at j is l - j layers old
    if len(cohorts) == 1:
        P, Q, table = plus[0], minus[0], tables[0]
    else:
        P, Q, table = _e2_H(fnp.concatenate(plus, axis=1)), _e2_H(fnp.concatenate(minus, axis=1)), _e2_H(fnp.concatenate(tables, axis=1))
    e1, e2, e3, e4 = _e2_H(fnp.unstack(table))
    D = _e2_H(P - Q)
    h = _e2_H(D * e4)                                # D (e1 p^3 + D (e2 p^2 + D (e3 p + e4 D))), summed over the pairs
    x = _e2_H(P * e3)
    _e2_H(fnp.add(h, x, out=h))
    _e2_H(fnp.multiply(h, D, out=h))
    p2 = _e2_H(P * P)
    _e2_H(fnp.multiply(p2, e2, out=x))
    _e2_H(fnp.add(h, x, out=h))
    _e2_H(fnp.multiply(h, D, out=h))
    _e2_H(fnp.multiply(p2, P, out=p2))               # p^3
    _e2_H(fnp.multiply(p2, e1, out=p2))
    _e2_H(fnp.add(h, p2, out=h))
    for _start, K, _j in cohorts:
        _K4L_STATS.update(blocks=_K4L_STATS['blocks'] + 1, pairs=_K4L_STATS['pairs'] + K,
                          elements=_K4L_STATS['elements'] + int(pp.shape[0]) * K)
    return fnp.einsum('ij,ij->i', h, D, optimize=False)


def _k4l_dk(loc, a, vv):
    """c (loc - beta vv^2) on the pushed rows (loc != 0), 0 elsewhere. a = mu / sigma, vv = pre-activation variances (n,)."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    loc = loc.astype(fnp.float64)
    a2 = a * a
    q = _e2_H(a2 - 1) * _e2_H(fnp.exp(_e2_H(a2 * -0.5)))
    pushed = fnp.not_equal(loc, 0.0)
    iso = vv * vv
    wi = _e2_H(_e2_H(_e2_H(q * q) + 1e-12) * pushed) * iso
    beta = _e2_H(fnp.sum(_e2_H(wi * loc))) / _e2_H(fnp.sum(_e2_H(wi * iso)))
    dk = _e2_H(loc - _e2_H(beta * iso)) * pushed
    return dk if isinstance(_K4L_C, tuple) or _K4L_C == 1.0 else dk * _K4L_C


def _k4l_final(mu, v):
    """The final layer's correction, from the local terms the row-cut push left in _K4L_FINAL."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    found, _K4L_FINAL[0] = _K4L_FINAL[0], None
    if found is None or found[1] is None:
        return 0.0
    loc = fnp.zeros(int(mu.shape[0]), dtype=fnp.float32)
    fnp.put(loc, found[0], found[1])
    return _k4l_dk(loc, mu / _e2_H(fnp.sqrt(_e2_H(fnp.maximum(v, 1e-10)))), v)
# END OPUS-ENGINEER K4L v6

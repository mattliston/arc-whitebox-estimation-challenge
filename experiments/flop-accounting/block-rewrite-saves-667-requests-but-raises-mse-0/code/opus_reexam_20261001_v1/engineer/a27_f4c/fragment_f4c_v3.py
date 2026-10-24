# BEGIN OPUS-ENGINEER F4c v3 (a27, 2026-10-02; v3 = v1's arithmetic bit for bit in fewer requests: the cohorts' column blocks of r, r^2 and the
# shift block by three splits instead of five views per cohort, the row factor by one two-operand einsum, the two scattered vectors
# written in float32 and widened by the scatter itself, the division of beta after it; 74 -> 62 requests a step): native port of the physicist lane's exact-term MIXTURE (O/physicist/f4b.py mode 'mix',
# metric 'birth'; plain-array reference O/physicist/f4c_native_ref.py; RESULT.md T13 / T14). An ERROR lever; REPLACES F4.
# F4 shifts every young gated column along its own square, theta A p^2. The mixture keeps that rule and adds the hub's exact
# fourth-cumulant terms S1 = -alpha pdf / sigma^2 and S2 = 2 (g - g^2 - m pdf / sigma) at strengths c1, c2. For a pair born at hub h
# (columns s + t+ e_h and s + t- e_h, weights +lam and -lam) and its pushed columns p, q at a later layer, with D = p - q:
#     shift_p = (d/2) [theta A p^2 + (n_p0 p^2 + n_p1 p D + n_p2 D^2) / lam],  shift_q = (d/2) [theta A q^2 + (n_m0 p^2 + n_m1 p D + n_m2 D^2) / (-lam)]
# The six n are the least-squares solution of a 2 x 2 system whose metric is fixed at the pair's birth (|s|^2 : 1, s = the pair's
# common leg), so they are constants of the pair. Here they are folded, AT THE BIRTH, together with theta A (q^2 = p^2 - 2 p D + D^2)
# and the weight the columns carry at each age into six float32 coefficients per pair and age:
#     shift_p = (d/2) (a_p p^2 + b_p p D + c_p D^2),   shift_q = (d/2) (a_m p^2 + b_m p D + c_m D^2).
# Birth (_f4c_birth, once per layer, small vectors only): in units of the leg, x = t+ / |s|, y = t- / |s|, every n is
#     [ S1 / |s| * P1(x, y) + S2 / |s|^2 * P2(x, y) ] / ((x - y)^(j+1) Delta(x, y)),   Delta = 2 x^2 y^2 + (x + y)^2 + 2,
# with integer polynomials P that _f4c_const derives once per process from the reference's own matrices (B' G M, the adjugate of
# B' G B, the change from (Y^2, Y Wt, Wt^2) to (p^2, p D, D^2)) by exact polynomial arithmetic; one division by x - y is taken
# exactly there. Two small einsums evaluate all of them on power tables of x and y. The reference's two guards are kept: a pair
# whose gap is below 1e-12 or whose system has det <= 1e-8 G00 G11 gets no exact part (the rule part stays).
# Step (_f4c_step, end of steps 1..14, cohorts born 1 or 2 layers earlier, the cohort born at layer 0 included: its rule part is 0
# but its S2 is not): live rows only, float32; per cohort D, p D, D^2 and the two quadratic forms written into one shift block, then
# F4's anchor on the whole block with X = lam o shift:  beta = [X @ (v @ (u o u)) + 2 u @ (v @ (u o X))] / sum v^2,  diagonal = 3 sum_r u_r^2 X_r.
# Billed 21.5 flops per element of (live rows x shifted columns) (F4: 16): r^2 (1), per pair D, p D, D^2 and ten combine passes
# (13 per pair), five matrix-vector passes (10), two products for them (2), the row scaling and the addition of the shift (2).
_F4C_THETA = 0.378                                # strength of F4's rule; a tuple gives one strength per age (age 1 first)
_F4C_C = (0.48, 0.48)                             # strengths of the exact S1 and S2; a tuple of pairs gives one pair per age
_F4C_AGE = 2
_F4C_LAST = 14                                    # the last step; pairs born there are never shifted
_F4C_COEF = {}                                    # birth layer -> per age the six coefficient vectors (float32, one entry per pair); this prediction
_F4C_CONST = {}
_F4C_STATS = dict(layers=0, pairs=0, elements=0)


def _f4c_const():
    C = _F4C_CONST
    if C:
        return C
    f64 = fnp.float64

    def mul(p, q):                                # polynomials in (x, y) as {(power of x, power of y): coefficient}
        out = {}
        for (a, b), u in p.items():
            for (c, e), w in q.items():
                out[(a + c, b + e)] = out.get((a + c, b + e), 0) + u * w
        return out

    def lin(*terms):                              # sum of factor * polynomial
        out = {}
        for factor, p in terms:
            for key, u in p.items():
                out[key] = out.get(key, 0) + factor * u
        return out

    def over_gap(p):                              # p / (x - y), exact: synthetic division in x
        carry, out = {}, {}
        for a in range(max([key[0] for key in p], default=-1), -1, -1):
            new = {b + 1: u for b, u in carry.items()}
            for (a2, b), u in p.items():
                if a2 == a:
                    new[b] = new.get(b, 0) + u
            if a == 0:
                if any(new.values()):
                    raise RuntimeError('F4c: a numerator is not divisible by the gap')
            else:
                out.update({(a - 1, b): u for b, u in new.items() if u})
                carry = new
        return out

    X, Y, ONE = {(1, 0): 1}, {(0, 1): 1}, {(0, 0): 1}
    T = (X, Y)
    a2, c2, bq = lin((1, ONE), (1, mul(X, X))), lin((1, ONE), (1, mul(Y, Y))), lin((1, ONE), (1, mul(X, Y)))   # 1 + x^2, 1 + y^2, 1 + x y
    other = (mul(c2, c2), mul(a2, a2))            # the diagonal entry of B' G B of the OTHER sign
    cross = mul(bq, bq)                           # its off-diagonal entry
    # rows of B' G M at |s| = 1 for the source S1 (l = 0) and S2 (l = 1): [l][r][sign], r = coefficient of Y^2, Y Wt, Wt^2
    R = [[[lin((2, T[s])) for s in (0, 1)], [lin((2, ONE)) for s in (0, 1)], [{} for s in (0, 1)]],
         [[mul(T[s], T[s]) for s in (0, 1)], [lin((4, T[s])) for s in (0, 1)], [ONE for s in (0, 1)]]]
    numer = [[[[[0.0] * 4 for _a in range(6)] for _j in range(3)] for _s in range(2)] for _l in range(2)]
    for l in (0, 1):
        for s in (0, 1):
            N = [lin((1, mul(other[s], R[l][r][s])), (-1, mul(cross, R[l][r][1 - s]))) for r in (0, 1, 2)]   # adjugate times B' G M
            # (Y^2, Y Wt, Wt^2) -> (p^2, p D, D^2) with Y = p - x D / gap, Wt = D / gap; every form carries gap^-(j+2) Delta^-1
            forms = (N[0], lin((1, N[1]), (-2, mul(X, N[0]))), lin((1, mul(mul(X, X), N[0])), (-1, mul(X, N[1])), (1, N[2])))
            for j, form in enumerate(forms):
                for (a, b), u in over_gap(form).items():
                    numer[l][s][j][a][b] = float(u)   # an IndexError here = a degree the power tables do not hold
    gap = lin((1, X), (-1, Y))
    ac = mul(a2, c2)
    delta = lin((1, ac), (1, cross))
    guard = lin((1.0, mul(mul(gap, gap), delta)), (-1e-8, mul(ac, ac)))   # det - 1e-8 G00 G11 in units of the leg
    extra = [[[0.0] * 5 for _a in range(6)] for _o in range(2)]
    for o, p in enumerate((delta, guard)):
        for (a, b), u in p.items():
            extra[o][a][b] = float(u)
    theta = _F4C_THETA if isinstance(_F4C_THETA, tuple) else (_F4C_THETA,) * _F4C_AGE
    strength = _F4C_C if isinstance(_F4C_C[0], tuple) else (_F4C_C,) * _F4C_AGE
    if len(theta) != _F4C_AGE or len(strength) != _F4C_AGE:
        raise RuntimeError('F4c: one strength per age')
    carry, new = (1.0, 1.0) if _S19_LAM is None else (float(_S19_LAM[0]), float(_S19_LAM[1]))
    # [age][source][sign]: strength, S1 = -(alpha pdf / sigma^2) and S2 = 2 (g - g^2 - m pdf / sigma), the sign of the column's weight,
    # and the dials the weight has collected when the cohort is `age` layers old (newborn x s_new, then x s_carry per step)
    mix = [[[strength[i][0] * -1.0 * sign / (new * carry ** (i + 1)) for sign in (1.0, -1.0)],
            [strength[i][1] * 2.0 * sign / (new * carry ** (i + 1)) for sign in (1.0, -1.0)]] for i in range(_F4C_AGE)]
    C.update(numer=fnp.asarray(numer, dtype=f64), extra=fnp.asarray(extra, dtype=f64), mix=fnp.asarray(mix, dtype=f64),
             theta=fnp.asarray(theta, dtype=f64), rule=fnp.asarray([[-1.0, 0.0, 0.0], [-1.0, 2.0, -1.0]], dtype=f64),   # theta A p^2 and theta A q^2 on (p^2, p D, D^2), A = -(alpha / (sigma (1 + radial / 3)))
             e6=fnp.arange(6, dtype=f64)[:, None], e5=fnp.arange(5, dtype=f64)[:, None], e4=fnp.arange(4, dtype=f64)[:, None],
             m123=fnp.asarray([[-1.0], [-2.0], [-3.0]], dtype=f64), leg=fnp.asarray([[-0.5], [-1.0]], dtype=f64))
    return C


def _f4c_birth(l, aux, d, host, selected, t, s, base, weight, radial):
    """The coefficients of the pairs born at layer l. d: pdf / sigma (n,); host: the pair map's mean (before the learned
    correction); t, s: the + and - hub entries of the newborn columns (n,); base: their common legs (n, K) float32, the hub's own
    entry 0; weight: their third-order weights before the dials (K,), positive; radial: this layer's fourth-order scalar."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    C = _f4c_const()
    if l == 0:
        _F4C_COEF.clear()
        _F4C_STATS.update(layers=0, pairs=0, elements=0)
    if l >= _F4C_LAST:
        return
    a, sg, g, dd, m, tp, tm = fnp.unstack(_e2_H(fnp.stack([aux['alpha'], aux['sigma'], aux['gate'], d, host, t, s]))[:, selected])
    w = fnp.maximum(_e2_H(fnp.sum(_e2_H(base * base), axis=0)), 1e-30)        # |s|^2 of the common leg
    leg = fnp.power(w, C['leg'])                                              # 1 / |s|, 1 / |s|^2 (float64)
    rw, _unused = fnp.unstack(leg)
    x = tp * rw
    y = tm * rw
    px = fnp.power(x, C['e6'])
    numer = fnp.einsum('lsjab,ak,bk->lsjk', C['numer'], px, _e2_H(fnp.power(y, C['e4'])), optimize=False)
    delta, guard = fnp.unstack(_e2_H(fnp.einsum('oab,ak,bk->ok', C['extra'], px, _e2_H(fnp.power(y, C['e5'])), optimize=False)))
    S1 = _e2_H(a * dd) / sg                                                   # alpha pdf / sigma^2
    S2 = _e2_H(g - _e2_H(g * g)) - _e2_H(m * dd)                              # g - g^2 - m pdf / sigma
    source = _e2_H(fnp.stack([S1, S2])) * leg                                 # in units of the leg
    ok = _e2_H(guard > 0.0) & _e2_H(_e2_H(fnp.abs(_e2_H(tp - tm))) > 1e-12)   # the reference's two guards
    gaps = fnp.power(_e2_H(fnp.where(ok, _e2_H(x - y), 1.0)), C['m123'])      # (x - y)^-1, ^-2, ^-3
    scale = _e2_H(gaps / delta) / _e2_H(fnp.maximum(weight, 1e-30))
    exact = _e2_H(fnp.einsum('als,lsjk,lk->asjk', C['mix'], numer, source, optimize=False)) * scale
    A = a / _e2_H(sg * _e2_H(1 + _e2_H(radial / 3)))                          # minus F4's birth scalar (the sign is in C['rule'])
    rule = fnp.einsum('a,sj,k->asjk', C['theta'], C['rule'], A, optimize=False)
    coef = _e2_H(fnp.where(ok, exact, 0.0)) + rule
    vectors = fnp.unstack(fnp.reshape(_e2_H(coef.astype(fnp.float32)), (6 * _F4C_AGE, int(tp.shape[0]))))
    _F4C_COEF[l] = [vectors[6 * i:6 * i + 6] for i in range(_F4C_AGE)]


def _f4c_step(l, right, g, d, live_indices, v, lam, state):
    """End of step l. right: pushed young columns on the live rows (float32); g, d: gate and pdf / sigma (n,); v: diagonal of the
    carried post covariance (n,); lam: carried third-order weights before this layer's dials; state: old-tier state (cohort roster).
    Returns None, or (shift, beta, diagonal): shift (live rows, k) float32 for the last k gated young columns, beta (n,) and
    diagonal (n,) float64 for the star."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    f32, f64 = fnp.float32, fnp.float64
    old = int(state['columns'])
    cohorts = [(int(state['cohorts'][j][0]), int(state['cohorts'][j][1]) // 2, j) for j in range(max(l - _F4C_AGE, 0), l)]
    cohorts = [c for c in cohorts if c[1] > 0 and c[0] >= old]
    if not cohorts:
        return None
    n = int(g.shape[0])
    nc = int(lam.shape[0])
    if int(right.shape[1]) != nc - old:
        raise RuntimeError('F4c: young column roster')
    y0 = cohorts[0][0]                               # the shifted columns: the young cohorts first .. l-1, the last nc - y0 columns
    r = right[:, y0 - old:]
    nl, ky = int(r.shape[0]), nc - y0
    hd = d * 0.5
    vg = v * g
    rows = _e2_H(_e2_H(fnp.stack([_e2_H(vg * g), vg, hd, g]))[:, live_indices]).astype(f32)   # row weights on the live rows
    vg2, vgl, hl, gl = fnp.unstack(rows)
    lam2 = _e2_H(lam[y0:])
    if _S19_LAM is not None:
        lam2 = lam2 * _S19_LAM[0]                    # the weights these columns carry out of the step
    RR = r * r
    s1 = vg2 @ RR                                    # v . u_r^2
    E = fnp.empty((nl, ky), dtype=f32)               # the shifts, in the column layout of r
    bounds = []
    for start, K, j in cohorts:                      # the block is the cohorts' (+, -) column blocks, one after the other
        if start - y0 != (bounds[-1] if bounds else 0):
            raise RuntimeError('F4c: young column roster')
        bounds += [start - y0 + K, start - y0 + 2 * K]
    if bounds[-1] != ky:
        raise RuntimeError('F4c: young column roster')
    rb, RRb, Eb = (_e2_H(fnp.split(x, bounds[:-1], axis=1)) for x in (r, RR, E))   # the blocks as views, one request per array
    for i, (start, K, j) in enumerate(cohorts):
        ap, bp_, cp, am, bm_, cm = _F4C_COEF[j][l - j - 1]   # the cohort born at j is l - j layers old
        P = rb[2 * i]
        PP = RRb[2 * i]
        Dm = P - rb[2 * i + 1]
        PD = P * Dm
        _e2_H(fnp.multiply(Dm, Dm, out=Dm))          # D^2, in place
        dp = Eb[2 * i]
        dm = Eb[2 * i + 1]
        _e2_H(fnp.multiply(PP, ap, out=dp))          # the + columns: a_p p^2 + b_p p D + c_p D^2
        x = PD * bp_
        _e2_H(fnp.add(dp, x, out=dp))
        _e2_H(fnp.multiply(Dm, cp, out=x))
        _e2_H(fnp.add(dp, x, out=dp))
        _e2_H(fnp.multiply(PP, am, out=dm))          # the - columns
        _e2_H(fnp.multiply(PD, bm_, out=x))
        _e2_H(fnp.add(dm, x, out=dm))
        _e2_H(fnp.multiply(Dm, cm, out=x))
        _e2_H(fnp.add(dm, x, out=dm))
        _e2_H((P, PP, Dm, PD, x, dp, dm))
        _F4C_STATS.update(pairs=_F4C_STATS['pairs'] + K, elements=_F4C_STATS['elements'] + nl * 2 * K)
    E = fnp.einsum('ij,i->ij', _e2_H(E), hl, optimize=False)   # the row factor d / 2 for all shifted columns at once
    # anchor: the added tensor is sum_r lam_r Sym(u_r, u_r, shift_r), u_r = g o r_r
    t1 = E @ _e2_H(lam2 * s1)
    X = r * E
    s2 = vgl @ X
    t2 = r @ _e2_H(lam2 * s2)
    _e2_H(fnp.multiply(RR, E, out=X))
    dg = X @ lam2
    beta = fnp.zeros(n, dtype=f64)
    _e2_H(fnp.put(beta, live_indices, _e2_H(t1 + _e2_H(_e2_H(2 * gl) * t2))))   # float32 values, widened by the scatter; the division is in float64 as before
    _e2_H(fnp.divide(beta, _e2_H(fnp.sum(_e2_H(v * v))), out=beta))
    diagonal = fnp.zeros(n, dtype=f64)
    _e2_H(fnp.put(diagonal, live_indices, _e2_H(_e2_H(_e2_H(3 * gl) * gl) * dg)))
    _F4C_STATS['layers'] = _F4C_STATS['layers'] + 1
    return E, beta, diagonal
# END OPUS-ENGINEER F4c v3

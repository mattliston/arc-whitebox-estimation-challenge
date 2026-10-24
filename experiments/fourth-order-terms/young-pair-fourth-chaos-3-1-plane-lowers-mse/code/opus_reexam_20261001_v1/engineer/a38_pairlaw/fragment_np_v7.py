# BEGIN OPUS-ENGINEER PAIR LAW OF THE NORM COUPLING, FOLDED INTO THE BANK READ (a38 NP v7, 2026-10-02; a physics change). Native form of the
# lead's O/ncar6.py at level 'read' (spec O/NCAR6_SPEC.md); it contains the physicist's bank channel (a37 NB) as its first half.
# WITH ITS DEFAULT LITERALS THE ARITHMETIC IS v1's (a38_pairlaw/fragment_np_v1.py, the version the first sessions ran) in a quarter of
# its requests (v2); v3 and v4 add three switches: _NP_FORMS (which forms the fold removes), _NP_DFIT (the regression with a constant
# in place of the table), _NP_BETA (the carried weights' source from the card's beta). What keeps the requests low:
#   - nothing is computed that no read uses: no fold work above the last layer with a strength, no update from that layer on;
#   - the response rows of the pushed columns are taken once (in place of the parent's rows of the squares, which are their squares);
#   - every per-unit and per-hub formula of the law is ONE polynomial evaluation (a gather by a constant index table, a product over
#     the degree axis, one product with a constant coefficient matrix); the tables are the formulas below expanded by exact
#     polynomial arithmetic in plain Python when the file is imported (_np_tables; no array, a few milliseconds outside every
#     prediction), and become arrays at the first prediction of the process (_np_const);
#   - every per-layer literal (strength, the multiple of C, the two carry factors) sits inside those constant tables.
# v5 = v4's arithmetic with three passes over the columns done by faster kernels (same bill): the exchange of every pair's two
# columns is a split into the roster's blocks and one concatenate (block copies; v4's put_along_axis took 6 ns per element, more
# than all other passes of the fold together), B's diagonal is a product and a matrix-vector product instead of a three-operand
# einsum, and the f-scaling is done in place. The roster's block sizes are plain numbers: no index array, three requests a layer fewer.
# v6 = v5 plus a second term in the same left factor (switch _NP_K3L, 0 = off = v5): THE FOURTH-CHAOS PART OF THE YOUNG PAIRS' LOCAL
# QUARTIC AS A (3,1) PLANE (the lead's O/k31l.py with chaos4, level 'read'; addendum of O/NCAR6_SPEC.md). K4L uses the local quartic
# of a pair, Q = e1 p^3 D + e2 p^2 D^2 + e3 p D^3 + e4 D^4 on its pushed columns p (+), q (-), D = p - q, on the diagonal only. The
# plane is K31L_ij = (1/4) sum_h [dQ/dp_i p_j + dQ/dD_i D_j] minus its second-chaos projection (two of the three i-slots replaced by
# the pair's means of p^2, p D, D^2), which the pair law already carries: (1/8) sum_h [dH/dp_i p_j + dH/dD_i D_j] with
# H = m_pp Q_pp + 2 m_pD Q_pD + m_DD Q_DD. Written per COLUMN r of the cohorts at most _NP_K3L_AGE layers old, with o = the column's
# own entry and Delta = o - its partner's: the left factor gets (k31 / 3)(1 / lam_r)(a cubic + linear polynomial in o, Delta) with six
# coefficients per column from K4L's own e1..e4 of the pair, the column's 1 / lam, the three means and whether it is the plus
# column: one more polynomial table (a39_k31l/identity_check.py and roster_form_check.py: equal to the reference's formulas to
# 1e-15). v7 = v6's arithmetic with the per-column polynomial in Horner form, o (o (c3 o + c2 Delta) + c1 Delta^2 + c_o) + Delta (c0
# Delta^2 + c_d): 14 passes over (response rows x young columns) where v6's six monomial slabs and their einsum were billed 20, and
# the coefficient table in float32 (its gathers are billed 4 per element, 8 in float64).
# STATE next to lam, one entry per bank column (same roster): w (second weight) and x (the cross weight of the column's pair), carried as
# QX = kap x (w / lam, x / lam), kap = strength / n of the layer that reads them (so the read needs no scaling), and the sizes of
# the roster's blocks (the plus columns and the minus columns of every cohort, in order: plain numbers). Matrix of a layer on its pushed columns:
#     B = sum_r w_r pp_r pp_r' + sum_pairs x (pp+ pp-' + pp- pp+').
# READ of layer l (response rows i; P = the pushed columns with every pair's two columns swapped; R2_ir = q_r pp_ir + xq_r P_ir):
#     diagB_i = sum_r lam_r pp_ir R2_ir
#     regression of diagB over the response rows on (v_i, mu_i^2, 2 mu_i (C mu)_i), coefficients (c0, c1, c2); with _NP_DFIT 3 also on 1
#     the multiple of C in B: aB = _NP_DSCALE x _NP_RHO[l] x c0   (_NP_DFIT 3: _NP_DSCALE x c0, no table)
#     left factor: (pp^2)_ir - f_i R2_ir,  f_i = strength mu_i / (n v_i)                    (the read multiplies by lam afterwards)
#     after the read, on the block (response rows x alive columns), the forms the fold must not contain are put back:
#       plane_ij += f_i (aB C_ij  [+ c1 mu_i mu_j with _NP_FORMS >= 2]  [+ c2 (mu_i (C mu)_j + (C mu)_i mu_j) with _NP_FORMS 3])
# AFTER THE PAIR MAP of layer l (the lean law, per unit; only the birth hubs h are needed):
#     gam = g - m d, V = 2 m (1 - g), gb = mean(gam), S2 = 2 (gam - g^2), S3 = 3 (1 - g)(r2 - 2 m^2), e3 = - alpha pdf / v
#     Ntd = diagB + a_rest v, a_rest = ca (radial / 3) tr C - aB;   Pv = gb Ntd + 2 (Co o Co) gam
#     rv = d (Co V) + (e3 / 2) Pv + S3 + d (Ko' gam) + (e3 / 2) (Ko V)                      (Ko = the pair map's K21 without its diagonal, l >= 1)
#     D  = S4 + V (Co V) + 2 gam (Co o Co) gam - 2 g^2 (Co o Co) g^2 + gam gb Ntd + V (Ko' gam) + gam (Ko V) - 2 g d ((Ko o Co) g^2)
#          - 2 g^2 ((Ko' o Co)(g d)) - g^2 ((Co o Co) S2 + gb diagB + (pp o pp) c)         [- gb a_rest VarG with _NP_AREST]
#     carried: q <- (2 / s_carry)(gb q + conv V' pp_r),  xq <- (2 gb / s_carry) xq          (c_r = conv lam_r V' pp_r; with _NP_BETA the
#       card's beta = 2 (gaussian mean - host gate) on the alive units stands for V here: its products with the columns are the
#       radial source's, so the update makes no pass over the columns for them)
#     newborn pair of hub h (columns p+ = s_h + t+ e_h with lam = +a s_new, p- = s_h + t- e_h with lam = -a s_new; gap = t+ - t-):
#       w+ = 2 (S2 t-^2 - 2 rv t- + D) / gap^2,  w- = 2 (S2 t+^2 - 2 rv t+ + D) / gap^2,  x = 2 (- S2 t+ t- + rv (t+ + t-) - D) / gap^2
# with the three source strengths _NP_BIRTHS (S2), _NP_ROWS (rv), _NP_DIAG (D). As in v1, diagB and (pp o pp) c exist on the response
# rows of the layer's read and are 0 at the few hubs outside them; t+ and t- are the loop's unmasked ones (gap = delta, a gap = birth).
_NP_STRENGTH = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)   # of the fold, per layer
_NP_RHO = (0.0, 1.040, 0.929, 0.936, 0.874, 0.821, 0.775, 0.739, 0.712, 0.671, 0.643, 0.613, 0.596, 0.579, 0.555, 0.0)   # off-diagonal multiple of C over the diagonal fit's
_NP_BIRTHS = 1.0
_NP_ROWS = 1.0
_NP_DIAG = 1.0
_NP_CONV = 1.0
_NP_CA = 0.75
_NP_AREST = False
_NP_K21 = True
_NP_FORMS = 1
_NP_DFIT = 2
_NP_DSCALE = 1.0
_NP_BETA = False
_NP_K3L = 0.0
_NP_K3L_AGE = 2
_NP_K3L_ALL = True
_NP_RIDGE = 1e-9
_NP_STATE = dict(QX=None, sizes=None, births=None, k4e=None, diag=None, pp2=None, fm=None, aa=None, ab=None, Ub=None, Rb=None)
_NP_CACHE = {}
_NP_STATS = dict(folds=0, updates=0, columns=0, pairs=0)


def _np_key(n):
    return (n, _S19_LAM, _NP_STRENGTH, _NP_RHO, _NP_BIRTHS, _NP_ROWS, _NP_DIAG, _NP_CONV, _NP_CA, _NP_AREST, _NP_K21, _NP_RIDGE, _NP_FORMS, _NP_DFIT, _NP_DSCALE,
            _NP_K3L, _NP_K3L_AGE, _NP_K3L_ALL)


def _np_tables(n):
    """The law's formulas as index and coefficient tables (nested lists): plain Python polynomial arithmetic, no array."""
    C = dict(key=_np_key(n))
    carry, s_new = (1.0, 1.0) if _S19_LAM is None else _S19_LAM
    last = max(l for l in range(16) if _NP_STRENGTH[l])                 # the last layer whose read is changed
    kap = [(_NP_STRENGTH[l] or 1.0) / n for l in range(16)]
    kap[0] = kap[1]

    # polynomials as {sorted tuple of symbol names: coefficient}; names starting with '#' are per-layer constants
    def sym(name):
        return {(name,): 1.0}

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

    def table(outputs, names, constants):
        """outputs: polynomials in the rows `names`; -> (index table degree x terms, per-layer coefficient matrices outputs x terms)."""
        pos = {name: i for i, name in enumerate(names)}
        split = [[(tuple(s for s in k if s[0] != '#'), tuple(s for s in k if s[0] == '#'), c) for k, c in p.items()] for p in outputs]
        terms = sorted({rows for p in split for rows, _, _ in p}, key=lambda k: (len(k), k))
        where = {rows: i for i, rows in enumerate(terms)}
        degree = max(len(k) for k in terms)
        index = [[(pos[k[j]] if j < len(k) else pos['one']) for k in terms] for j in range(degree)]
        coefficient = []
        for values in constants:
            layer = [[0.0] * len(terms) for _ in outputs]
            for o, p in enumerate(split):
                for rows, cons, c in p:
                    for s in cons:
                        c = c * values[s]
                    layer[o][where[rows]] += c
            coefficient.append(layer)
        return index, coefficient

    one = {(): 1.0}
    # ---- table 1: per-unit vectors from the loop's own (rows of `raw`)
    raw = ['one', 'g', 'm', 'd', 'al', 'ph', 'v', 'tp', 'tm', 'kd', 'sg', 'mu', 'delta', 'sb', 'vc', 'mask']
    g, m, d, al, ph, v, tp, tm, kd, sg, mu = (sym(x) for x in ('g', 'm', 'd', 'al', 'ph', 'v', 'tp', 'tm', 'kd', 'sg', 'mu'))
    V = mul(2.0, m, add(one, mul(-1.0, g)))
    gam = add(g, mul(-1.0, m, d))
    g2 = mul(g, g)
    S2 = mul(2.0, add(gam, mul(-1.0, g2)))
    al2 = mul(al, al)
    r2 = mul(v, add(mul(add(al2, one), g), mul(al, ph)))
    m2 = mul(m, m)
    S3v = mul(3.0, v, add(one, mul(-1.0, g)), add(r2, mul(-2.0, m2)))
    first = [mul(sym('#aV'), V), mul(sym('#aG'), gam), g2, S2, mul(g, d), add(mul(sym('delta'), sym('sb'), sym('vc')), mul(1e-30, one)),   # the last: gap^2 a v (floored: units that are no hub may have 0)
             V, gam, S3v, mul(0.5, al, ph)]
    t1 = ['Vt', 'gt', 'g2', 'S2', 'gd', 'den', 'V', 'gam', 'S3v', 'alph2']
    if _NP_DIAG:
        r3 = mul(v, sg, add(mul(add(al2, mul(3.0, one)), al, g), mul(add(al2, mul(2.0, one)), ph)))
        r4 = mul(v, v, add(mul(add(mul(add(al2, mul(6.0, one)), al2), mul(3.0, one)), g), mul(add(al2, mul(5.0, one)), al, ph)))
        excess = add(r2, mul(-1.0, m2))
        first.append(add(r4, mul(-4.0, m, r3), mul(6.0, m2, r2), mul(-3.0, m2, m2), mul(-3.0, excess, excess)))
        t1.append('S4')
        if _NP_AREST:
            gm = add(mul(sg, ph), mul(mu, g))
            first.append(add(r2, mul(-1.0, gm, gm)))
            t1.append('varG')
    layers = range(last)                                                # the layers with an update: 0 .. last - 1
    C['idx1'], C['coef1'] = table(first, raw, [{'#aV': 2.0 * kap[l + 1] / carry, '#aG': 2.0 * kap[l + 1] / (carry * kap[l])} for l in layers])
    C['aV'] = [2.0 * kap[l + 1] / carry for l in layers]
    # ---- table 2: the newborn pairs' four weights at the hubs (rows of F: raw, table 1's, the products, the scalars)
    products = ['cv', 'kv', 'ktg', 'kcg2', 'kctgd', 'k3', 'diag', 'iden', 'aB'] + ([] if _NP_DIAG else ['ccg'])
    names = raw + t1 + products + (['ccg', 'ccg2', 'ccs2'] if _NP_DIAG else []) + ['a2', 'trC', 'rad']
    C['products'] = products
    V, gam, g2, S2, gd, S3v, alph2 = (sym(x) for x in ('V', 'gam', 'g2', 'S2', 'gd', 'S3v', 'alph2'))
    iaV, iaG, ik = sym('#iaV'), sym('#iaG'), sym('#ik')
    mask = sym('mask')
    v2 = mul(v, v)
    cov_v = add(mul(iaV, sym('cv')), mul(-1.0, v, V))                   # (Co V)
    co2gam = add(mul(iaG, sym('ccg')), mul(-1.0, v2, gam))              # ((Co o Co) gam)
    gb = mul(iaG, sym('a2'))
    diag = mul(ik, sym('diag'))                                        # 0 outside the response rows by construction
    a_rest = add(mul(_NP_CA / 3.0, sym('rad'), sym('trC')), mul(-1.0, ik, sym('aB')))
    ntd = add(diag, mul(a_rest, v))
    pv = add(mul(gb, ntd), mul(2.0, co2gam))
    kov = add(mul(iaV, sym('kv')), mul(-1.0, kd, V)) if _NP_K21 else {}
    kotg = add(mul(iaG, sym('ktg')), mul(-1.0, kd, gam)) if _NP_K21 else {}
    v_rv = add(mul(d, v, cov_v), S3v, mul(-1.0, alph2, pv), mul(d, v, kotg), mul(-1.0, alph2, kov))   # v x rv  (v e3 / 2 = - alpha pdf / 2)
    a11, v_a12, a22 = mul(_NP_BIRTHS, S2), mul(_NP_ROWS, v_rv), {}
    if _NP_DIAG:
        co2g2 = add(sym('ccg2'), mul(-1.0, v2, g2))
        co2s2 = add(sym('ccs2'), mul(-1.0, v2, S2))
        dgn = add(sym('S4'), mul(V, cov_v), mul(2.0, gam, co2gam), mul(-2.0, g2, co2g2), mul(gam, gb, ntd))
        if _NP_K21:
            kcg2 = add(sym('kcg2'), mul(-1.0, kd, v, g2))
            kctgd = add(sym('kctgd'), mul(-1.0, kd, v, gd))
            dgn = add(dgn, mul(V, kotg), mul(gam, kov), mul(-2.0, gd, kcg2), mul(-2.0, g2, kctgd))
        D = add(dgn, mul(-1.0, g2, add(mul(_NP_BIRTHS, co2s2), mul(gb, diag), mul(_NP_CONV, iaV, mask, sym('k3')))))
        if _NP_AREST:
            D = add(D, mul(-1.0, gb, a_rest, sym('varG')))
        a22 = mul(_NP_DIAG, D)
    scale = mul(sym('#kn'), sym('iden'))                                # kap' 2 / (s_new gap^2 a v), with gap^2 a = delta birth
    q_plus = mul(scale, add(mul(v, a11, tm, tm), mul(-2.0, v_a12, tm), mul(v, a22)))
    q_minus = mul(-1.0, scale, add(mul(v, a11, tp, tp), mul(-2.0, v_a12, tp), mul(v, a22)))
    x_plus = mul(scale, add(mul(v_a12, add(tp, tm)), mul(-1.0, v, a11, tp, tm), mul(-1.0, v, a22)))
    C['idxH'], C['coefH'] = table([q_plus, q_minus, x_plus, mul(-1.0, x_plus)], names,
                                  [{'#iaV': carry / (2.0 * kap[l + 1]), '#iaG': carry * kap[l] / (2.0 * kap[l + 1]), '#ik': 1.0 / kap[l],
                                    '#kn': 2.0 * kap[l + 1] / s_new} for l in layers])
    # ---- the diagonal regression and the forms: monomials of the per-unit stack (one, two, mask, alive, v, mu, C mu, diagB, mu / v):
    # the regressors and the target on the response rows, then the rows the forms are made of. One Gram matrix; times a constant
    # matrix that scales the system (so that the solution over its entry at the target is: multiple x c0, c1, c2, .., 1) and cuts
    # the other rows off (their entries of the solution are 0); the forms' vectors are (the solution gathered by a constant index
    # table) x (the monomials): f, aB x alive, aB, c1 alive mu + c2 alive C mu, c2 alive mu, f mu, f C mu.
    stack = ['one', 'two', 'mask', 'alive', 'v', 'mu', 'cmu', 'diag', 'fr']
    regressors = [('mask', 'v')] + ([('mask',)] if _NP_DFIT == 3 else []) + [('mask', 'mu', 'mu'), ('mask', 'mu', 'cmu', 'two')]
    target = len(regressors)
    nf = _NP_FORMS - 1
    rows = regressors + [('mask', 'diag'), ('mask', 'fr'), ('alive',), ('one',)] + [('alive', 'mu'), ('alive', 'cmu')][:nf] + [('mask', 'fr', 'mu'), ('mask', 'fr', 'cmu')][:nf]
    size = len(rows)
    place = {name: i for i, name in enumerate(stack)}
    C['idx_reg'] = [[(place[k[j]] if j < len(k) else place['one']) for k in rows] for j in range(max(len(k) for k in rows))]
    matrices = []
    for l in range(16):
        multiple = _NP_DSCALE * (_NP_RHO[l] if _NP_DFIT == 2 else 1.0)
        if 1 <= l <= last and not multiple > 0.0:
            raise ValueError('pair law: the multiple of C needs a positive factor at every layer of the fold')
        scale = [1.0 / (multiple or 1.0)] + [1.0] * (target - 1) + [-1.0]
        matrices.append([[(scale[a] * scale[b] * (1.0 + _NP_RIDGE * (a == b)) if a <= target and b <= target else float(a == b)) for b in range(size)] for a in range(size)])
    C['Mreg'] = matrices
    zero_at, c1, c2 = target + 1, target - 2, target - 1          # where the normalised solution holds 0, c1 and c2 (1 at the target)
    f_row, alive_row, one_row = target + 1, target + 2, target + 3
    outputs = [{f_row: target}, {alive_row: 0}, {one_row: 0}]                                 # f, aB alive, aB
    if nf >= 1:
        outputs.append({target + 4: c1, **({target + 5: c2} if nf == 2 else {})})             # c1 alive mu (+ c2 alive C mu)
    if nf == 2:
        outputs.append({target + 4: c2})                                                      # c2 alive mu
    outputs += [{target + 4 + nf + k: target} for k in range(nf)]                             # f mu (, f C mu)
    C['idxM'] = [[o.get(y, zero_at) for y in range(size)] for o in outputs]
    C.update(target=target, nf=nf, size=size)
    C['last'] = last
    if _NP_K3L:
        # ---- the young pairs' fourth-chaos plane: per column, the coefficients of o^3, o^2 Delta, o Delta^2, Delta^3, o, Delta from
        # (one, plus-column flag, 1 / lam, e1..e4, the means of p^2, p D, D^2 written on the column's own o^2, o Delta, Delta^2)
        o, dl, wp, Li = sym('o'), sym('dl'), sym('wp'), sym('Li')
        e1, e2, e3, e4 = (sym(x) for x in ('e1', 'e2', 'e3', 'e4'))
        s1, s2, s3 = sym('s1'), sym('s2'), sym('s3')
        total = {}
        for plus in (True, False):
            if plus:   # the column is p: D = Delta
                p, D, mpp, mpD, mDD = o, dl, s1, s2, s3
            else:      # the column is q: p = o - Delta, D = - Delta
                p, D = add(o, mul(-1.0, dl)), mul(-1.0, dl)
                mpp, mpD, mDD = add(s1, mul(-2.0, s2), s3), add(mul(-1.0, s2), s3), s3
            Qp = add(mul(3.0, e1, p, p, D), mul(2.0, e2, p, D, D), mul(e3, D, D, D))
            QD = add(mul(e1, p, p, p), mul(2.0, e2, p, p, D), mul(3.0, e3, p, D, D), mul(4.0, e4, D, D, D))
            h1 = add(mul(12.0, e1, mpD), mul(4.0, e2, mDD))
            h2 = add(mul(6.0, e1, mpp), mul(8.0, e2, mpD), mul(6.0, e3, mDD))
            h3 = add(mul(4.0, e2, mpp), mul(12.0, e3, mpD), mul(24.0, e4, mDD))
            Hp, HD = add(mul(p, h1), mul(D, h2)), add(mul(p, h2), mul(D, h3))
            if plus:
                total = add(total, mul(wp, Li, _NP_K3L / 3.0, add(mul(0.25, add(Qp, QD)), mul(-0.125, add(Hp, HD)))))
            else:
                total = add(total, mul(add(one, mul(-1.0, wp)), Li, _NP_K3L / 3.0, add(mul(-0.25, QD), mul(0.125, HD))))
        parts = {}
        for k, c in total.items():
            key = (k.count('o'), k.count('dl'))
            rest = tuple(x for x in k if x not in ('o', 'dl'))
            parts.setdefault(key, {})
            parts[key][rest] = parts[key].get(rest, 0.0) + c
        index, coefficient = table([parts.get(key, {}) for key in ((3, 0), (2, 1), (1, 2), (0, 3), (1, 0), (0, 1))],
                                   ['one', 'wp', 'Li', 'e1', 'e2', 'e3', 'e4', 's1', 's2', 's3'], [{}])
        C['idxK'], C['coefK'] = index, coefficient[0]
        C['Cmean'] = [[1.0 / n, 0.0, 0.0], [1.0 / n, 0.0, -1.0 / n], [1.0 / n, 1.0 / n, -2.0 / n]]   # sums of o^2, the partner's, o x partner over all units -> the three means
    return C


_NP_TABLES = _np_tables(1024)   # at import: plain Python


def _np_const(n):
    """The tables as arrays, once per process."""
    C = _NP_CACHE
    if C:
        return C
    T = _NP_TABLES if _NP_TABLES['key'] == _np_key(n) else _np_tables(n)
    C.update(last=T['last'], idx1=fnp.asarray(T['idx1']), coef1=fnp.unstack(fnp.asarray(T['coef1'])), idxH=fnp.asarray(T['idxH']),
             coefH=fnp.unstack(fnp.asarray(T['coefH'])), idx_reg=fnp.asarray(T['idx_reg']), Mreg=fnp.unstack(fnp.asarray(T['Mreg'])),
             idxM=fnp.asarray(T['idxM']), unit=fnp.asarray([float(k == T['target']) for k in range(T['size'])]),
             one=fnp.ones(n, dtype=fnp.float64), two=fnp.full(n, 2.0, dtype=fnp.float64), zero=fnp.zeros(n, dtype=fnp.float64),
             zero0=fnp.asarray(0.0), target=T['target'], nf=T['nf'], aV=T['aV'], products=T['products'])
    if _NP_K3L:
        C.update(idxK=fnp.asarray(T['idxK']), coefK=fnp.asarray(T['coefK'], dtype=fnp.float32), Cmean=fnp.asarray(T['Cmean'], dtype=fnp.float32),
                 ones4=fnp.ones(4 * n, dtype=fnp.float32))
    return C


def _np_left(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov):
    """The read's left factor: the response rows of pp^2, with the pair law folded in when the layer has one. Stores diagB (scattered to
    the units, 0 outside the response rows) and the vectors of the forms for _np_forms and for the law."""
    S = _NP_STATE
    S['diag'] = S['pp2'] = S['fm'] = S['aa'] = S['ab'] = S['Ub'] = S['Rb'] = None
    if S['QX'] is None:
        return _unique_rows_via_scatter(pp2, response_indices)
    n, R = int(pp.shape[0]), int(pp.shape[1])
    C = _np_const(n)
    if l > C['last']:
        return _unique_rows_via_scatter(pp2, response_indices)
    sizes = S['sizes']
    if sum(sizes) != R or int(S['QX'].shape[1]) != R:
        raise RuntimeError('pair law: weight roster')
    hold = _E2Hold()
    q, xq = hold(fnp.unstack(S['QX']))
    rows = hold(_unique_rows_via_scatter(pp, response_indices))                               # the pushed columns on the response rows
    bounds, at = [], 0
    for width in sizes[:-1]:
        at += width
        bounds.append(at)
    blocks = hold(fnp.split(rows, bounds, axis=1))                                            # the roster's blocks, (+, -) of every cohort: views
    swapped = hold(fnp.concatenate([blocks[i ^ 1] for i in range(len(blocks))], axis=1))      # every pair's two columns exchanged (block copies)
    local = None
    young = [(b, K) for b, K in S['births'] if l - b <= _NP_K3L_AGE] if _NP_K3L else []
    if young:   # the fourth-chaos plane of the young pairs' local quartic: six monomials of (own entry, difference to the partner) per column
        W = 2 * sum(K for _b, K in young)
        y0, rows_kept = R - W, int(rows.shape[0])
        own = hold(rows[:, y0:])
        delta = hold(own - hold(swapped[:, y0:]))                                              # Delta = own - partner
        if _NP_K3L_ALL:   # the pair's means over all units, from the full columns: sums of o^2 (the squares are there), of the partner's, of o x partner
            inner, at = [], 0
            for _b, K in young:
                inner += [at + K, at + 2 * K]
                at += 2 * K
            full = hold(pp[:, y0:])
            pieces = hold(fnp.split(full, inner[:-1], axis=1))
            squares = hold(fnp.sum(hold(pp2[:, y0:]), axis=0))
            halves = hold(fnp.split(squares, inner[:-1]))
            cross = hold(fnp.einsum('ik,ik->k', full, hold(fnp.concatenate([pieces[i ^ 1] for i in range(len(pieces))], axis=1)), optimize=False))
            means = hold(C['Cmean'] @ hold(fnp.stack([squares, hold(fnp.concatenate([halves[i ^ 1] for i in range(len(halves))])), cross])))
            dd = hold(delta * delta)
        else:             # over the response rows
            dd = hold(delta * delta)
            means = hold(hold(fnp.stack([hold(fnp.sum(hold(own * own), axis=0)), hold(fnp.sum(hold(own * delta), axis=0)), hold(fnp.sum(dd, axis=0))])) * (1.0 / rows_kept))
        lam_young = hold(lam[y0:])
        head = hold(fnp.stack([hold(C['ones4'][:W]), hold(fnp.greater(lam_young, 0.0)), hold(fnp.power(lam_young, -1.0))], dtype=fnp.float32))
        quartic = S['k4e'][young[0][0]] if len(young) == 1 else hold(fnp.concatenate([S['k4e'][b] for b, _K in young], axis=1))
        table = hold(fnp.concatenate([head, quartic, means]))                                 # one, plus flag, 1 / lam, e1..e4, the three means (float32)
        c3, c2, c1, c0, c_o, c_d = hold(fnp.unstack(hold(C['coefK'] @ hold(fnp.prod(hold(table[C['idxK']]), axis=0)))))
        # o (o (c3 o + c2 Delta) + c1 Delta^2 + c_o) + Delta (c0 Delta^2 + c_d), per column
        value = hold(own * c3)
        other = hold(delta * c2)
        hold(fnp.add(value, other, out=value)); hold(fnp.multiply(value, own, out=value))
        hold(fnp.multiply(dd, c1, out=other))
        hold(fnp.add(value, other, out=value)); hold(fnp.add(value, c_o, out=value)); hold(fnp.multiply(value, own, out=value))
        hold(fnp.multiply(dd, c0, out=other))
        hold(fnp.add(other, c_d, out=other)); hold(fnp.multiply(other, delta, out=other))
        hold(fnp.add(value, other, out=value))
        local = (y0, value)                                                                   # (k31 / 3) x (the plane's left factor) / lam, per column
    hold(fnp.multiply(swapped, xq, out=swapped))
    r2 = hold(rows * q)
    hold(fnp.add(r2, swapped, out=r2))                                                        # kap (q_r pp_ir + xq_r P_ir)
    diagonal = fnp.zeros(n, dtype=fnp.float32)
    fnp.put(diagonal, response_indices, hold(hold(fnp.multiply(rows, r2, out=swapped)) @ lam))   # kap diagB, 0 outside the response rows
    S['diag'] = diagonal
    if _NP_DIAG:
        S['pp2'] = pp2                                                                        # for (pp o pp) c after the pair map
    if local is not None:   # after B's diagonal: the local quartic's plane is no part of B
        r2_young = hold(r2[:, local[0]:])
        hold(fnp.add(r2_young, local[1], out=r2_young))                                       # the read's plane gets - f_i (k31 / 3) K31L_ij, as the reference's fold
    Y = hold(fnp.prod(hold(hold(fnp.stack([C['one'], C['two'], response_mask, alive, vv, mu, hold(cov @ mu), diagonal, hold(mu / vc)]))[C['idx_reg']]), axis=0))
    solution = hold(fnp.linalg.solve(hold(hold(Y @ hold(Y.T)) * C['Mreg'][l]), C['unit']))
    solution = hold(solution / hold(fnp.unstack(solution))[C['target']])                      # kap x (multiple x c0, .., c1, c2), 1, 0, ..
    vectors = hold(hold(hold(solution[C['idxM']]) @ Y).astype(fnp.float32))                   # f / kap, kap aB alive, kap aB, then the mean forms'
    fm, S['aa'], S['ab'] = hold(fnp.unstack(vectors))[:3]
    left2 = rows * rows
    if not _NP_STRENGTH[l]:
        return left2
    S['fm'] = fm                                                                              # mu_i / v_i on the response rows (f_i = kap x this)
    nf = C['nf']
    if nf:
        S['Rb'], S['Ub'] = vectors[3:3 + nf], vectors[3 + nf:3 + 2 * nf]
    hold(fnp.multiply(r2, hold(hold(fm[response_indices])[:, None]), out=r2))                # f_i R2_ir, in place
    hold(fnp.subtract(left2, r2, out=left2))
    _NP_STATS['folds'] += 1
    return left2


def _np_forms(l, plane, cov32):
    """plane = the n x n third-order plane with the read's block in place; puts the removed forms back on that block (rows response,
    columns alive): f_i (aB C_ij + c1 mu_i mu_j + c2 (mu_i (C mu)_j + (C mu)_i mu_j))."""
    S = _NP_STATE
    if S['fm'] is None:
        return
    hold = _E2Hold()
    if cov32 is None:
        cov32 = hold(_E2_LAYER['Cpre'].astype(fnp.float32))
    forms = hold(fnp.einsum('ij,i,j->ij', cov32, S['fm'], S['aa'], optimize=False))
    if S['Ub'] is not None:
        hold(fnp.add(forms, hold(fnp.einsum('ai,aj->ij', S['Ub'], S['Rb'], optimize=False)), out=forms))
    hold(fnp.add(plane, forms, out=plane))


def _np_update(l, lam, pp, mu, vv, m, d, aux, t_plus, t_minus, delta, safe_birth, vc, response_mask, hubs, radial, K32, kdiag):
    """The weights after layer l, called at the births: the carried columns' and the newborn pairs' from the law. t_plus, t_minus: the
    loop's unmasked t and s of every unit; K32, kdiag: the pair map's K21 (float32, diagonal filled) and its diagonal."""
    S = _NP_STATE
    hold = _E2Hold()
    if l == 0:
        for key in S:
            S[key] = None
    if _NP_K3L and S['k4e'] is None:
        S['k4e'] = {}
    n = int(mu.shape[0])
    C = _np_const(n)
    K = int(hubs.shape[0])
    if l >= C['last'] or not (K or S['QX'] is not None):                                      # no later read uses the weights
        for key in S:
            hold(S[key])
            S[key] = None
        return
    R = int(lam.shape[0])
    k21 = _NP_K21 and l >= 1
    zero = C['zero']
    raw = hold(fnp.stack([C['one'], aux['gate'], m, d, aux['alpha'], aux['pdf'], vv, t_plus, t_minus,
                          (kdiag if k21 else zero), aux['sigma'], mu, delta, safe_birth, vc, response_mask]))
    T1 = hold(C['coef1'][l] @ hold(fnp.prod(hold(raw[C['idx1']]), axis=0)))
    Vt, gt, g2, S2, gd, den = hold(fnp.unstack(hold(T1.astype(fnp.float32))))[:6]
    a2 = hold(fnp.mean(gt))                                                                   # (2 / s_carry)(kap' / kap) gb
    cq = None
    if R:
        if _NP_BETA:
            cq = hold(_E2_LAYER['np_column_sum'] * C['aV'][l])                                # (2 kap' / s_carry) beta' pp_r, from the radial source (float64)
        else:
            cq = hold(Vt @ pp)                                                                # (2 kap' / s_carry) V' pp_r
        QX = S['QX']
        hold(fnp.multiply(QX, a2, out=QX))
        q_row = hold(QX[:1])
        hold(fnp.add(q_row, (cq if _NP_CONV == 1.0 else hold(cq * _NP_CONV)), out=q_row))
    if K:
        cov32 = _E2_LAYER.get('ps_Cpre32')                                                    # the pair map's float32 covariance of this layer
        if cov32 is None:
            cov32 = hold(_E2_LAYER['Cpre'].astype(fnp.float32))
        cc32 = hold(cov32 * cov32)
        vectors = dict(cv=hold(Vt @ cov32), iden=hold(fnp.power(den, -1.0)))
        if _NP_DIAG:
            squares = [hold(hold(fnp.stack([gt, g2, S2])) @ cc32)]                            # (C o C)[gam, g^2, S2]
        else:
            squares = []
            vectors['ccg'] = hold(gt @ cc32)
        if k21:
            vectors.update(kv=hold(K32 @ Vt), ktg=hold(gt @ K32))
            if _NP_DIAG:   # (K21 o C) g^2 and (K21 o C)' (g d); the diagonal is taken out in the polynomial
                vectors.update(kcg2=hold(fnp.einsum('ij,ij,j->i', K32, cov32, g2, optimize=False)),
                               kctgd=hold(fnp.einsum('ij,ij,i->j', K32, cov32, gd, optimize=False)))
        if S['diag'] is not None:
            vectors.update(diag=S['diag'], aB=S['ab'])
            if _NP_DIAG and R:
                weights = hold(cq * lam)
                vectors['k3'] = hold(S['pp2'] @ (hold(weights.astype(fnp.float32)) if _NP_BETA else weights))   # (pp o pp)(lam x V' pp), every unit
        if isinstance(radial, (int, float)):
            radial = C['zero0'] if radial == 0 else hold(fnp.asarray(float(radial)))
        scalars = hold(fnp.outer(hold(fnp.stack([a2, hold(fnp.sum(vv)), radial])), C['one']))
        F = hold(fnp.concatenate([raw, T1, hold(fnp.stack([vectors.get(k, zero) for k in C['products']]))] + squares + [scalars]))
        H = hold(F[:, hubs])
        born = fnp.reshape(hold(hold(C['coefH'][l] @ hold(fnp.prod(hold(H[C['idxH']]), axis=0))).astype(fnp.float32)), (2, 2 * K))
        if R:
            hold(S['QX']); hold(born)
            S['QX'] = fnp.concatenate([S['QX'], born], axis=1)
            S['sizes'], S['births'] = S['sizes'] + [K, K], S['births'] + [(l, K)]
        else:
            S['QX'], S['sizes'], S['births'] = born, [K, K], [(l, K)]
        if _NP_K3L:   # K4L's four birth coefficients of this layer's pairs, once per column (plus block, minus block)
            quartic = hold(fnp.stack(_K4L_B[l]))
            S['k4e'][l] = fnp.concatenate([quartic, quartic], axis=1)
            for b in [b for b in S['k4e'] if b <= l - _NP_K3L_AGE]:
                hold(S['k4e'].pop(b))
        _NP_STATS['pairs'] += K
    for key in ('diag', 'pp2', 'fm', 'aa', 'ab', 'Ub', 'Rb'):
        hold(S[key])
        S[key] = None
    _NP_STATS['updates'] += 1
    _NP_STATS['columns'] += R
# END OPUS-ENGINEER PAIR LAW OF THE NORM COUPLING

# BEGIN OPUS-ENGINEER PAIR LAW OF THE NORM COUPLING, FOLDED INTO THE BANK READ (a38 NP v1, 2026-10-02; a physics change). Native form of the
# lead's O/ncar6.py at level 'read' (spec O/NCAR6_SPEC.md); it contains the physicist's bank channel (a37 NB) as its first half.
# STATE next to lam, one entry per bank column (same roster): w (second weight), x (the cross weight of the column's pair), both carried
# divided by the column's lam (q = w / lam, xq = x / lam: the carried columns' lam only changes by s_carry per layer, so nothing is
# ever divided by a carried lam), and the roster position of each column's pair partner. Matrix of a layer on its pushed columns:
#     B = sum_r w_r pp_r pp_r' + sum_pairs x (pp+ pp-' + pp- pp+').
# READ of layer l (response rows i; P = the pushed columns with every pair's two columns swapped):
#     diagB_i = sum_r lam_r (q_r pp_ir^2 + xq_r pp_ir P_ir)
#     the multiple of C in B: aB = _NP_RHO[l] x (coefficient of v_i in the regression of diagB on (v_i, mu_i^2, mu_i (C mu)_i))
#     left factor: left2_ir <- left2_ir - f_i (q_r pp_ir + xq_r P_ir),  f_i = strength mu_i / (n v_i)       (the read multiplies by lam)
#     after the read, on the block (response rows x alive columns): plane_ij += f_i aB C_ij
# AFTER THE PAIR MAP of layer l = 0..14 (the lean law, per unit; only the birth hubs h are needed):
#     gam = g - m d, V = 2 m (1 - g), gb = mean(gam), S2 = 2 (gam - g^2), S3 = 3 (1 - g)(r2 - 2 m^2), e3 = - alpha pdf / v
#     Ntd = diagB + a_rest v, a_rest = ca (radial / 3) tr C - aB;   Pv = gb Ntd + 2 (Co o Co) gam
#     rv = d (Co V) + (e3 / 2) Pv + S3 + d (Ko' gam) + (e3 / 2) (Ko V)                      (Ko = the pair map's K21 without its diagonal, l >= 1)
#     D  = S4 + V (Co V) + 2 gam (Co o Co) gam - 2 g^2 (Co o Co) g^2 + gam gb Ntd + V (Ko' gam) + gam (Ko V) - 2 g d ((Ko o Co) g^2)
#          - 2 g^2 ((Ko' o Co)(g d)) - g^2 ((Co o Co) S2 + gb diagB + (pp o pp) c)         [- gb a_rest VarG with _NP_AREST]
#     carried: q <- (2 / s_carry)(gb q + conv V' pp_r),  xq <- (2 gb / s_carry) xq
#     newborn pair of hub h (columns p+ = s_h + t+ e_h with lam = +a s_new, p- = s_h + t- e_h with lam = -a s_new; gap = t+ - t-):
#       w+ = 2 (S2 t-^2 - 2 rv t- + D) / gap^2,  w- = 2 (S2 t+^2 - 2 rv t+ + D) / gap^2,  x = 2 (- S2 t+ t- + rv (t+ + t-) - D) / gap^2
# with the three source strengths _NP_BIRTHS (S2), _NP_ROWS (rv), _NP_DIAG (D). This first version is written for parity, not for few
# requests: the per-hub algebra is plain vector arithmetic.
_NP_STRENGTH = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)   # of the fold, per layer
_NP_RHO = (0.0, 1.040, 0.929, 0.936, 0.874, 0.821, 0.775, 0.739, 0.712, 0.671, 0.643, 0.613, 0.596, 0.579, 0.555, 0.0)   # off-diagonal multiple of C over the diagonal fit's
_NP_BIRTHS = 1.0
_NP_ROWS = 1.0
_NP_DIAG = 1.0
_NP_CONV = 1.0
_NP_CA = 0.75
_NP_AREST = False
_NP_K21 = True
_NP_RIDGE = ((1.000000001, 1.0, 1.0), (1.0, 1.000000001, 1.0), (1.0, 1.0, 1.000000001))
_NP_STATE = dict(q=None, xq=None, partner=None, diag=None, aB=None, rows=None, resp=None)
_NP_STATS = dict(folds=0, updates=0, columns=0, pairs=0)


def _np_fold(l, left2, pp, response_indices, lam, mu, vc, vv, cov):
    """left2 = (pp^2)[response rows], changed in place; stores diagB on the response rows and aB for the law; returns what the form
    removal needs, or None when there is no state. vc = the floored variances, vv = the variances, cov = the pre-activation covariance."""
    S = _NP_STATE
    S['diag'] = S['aB'] = S['rows'] = S['resp'] = None
    if S['q'] is None:
        return None
    if int(S['q'].shape[0]) != int(lam.shape[0]):
        raise RuntimeError('pair law: weight roster')
    hold = _E2Hold()
    rows = _unique_rows_via_scatter(pp, response_indices)                                     # the pushed columns on the response rows
    swapped = fnp.zeros((int(rows.shape[0]), int(rows.shape[1])), dtype=fnp.float32)
    fnp.put_along_axis(swapped, hold(S['partner'][None, :]), rows, axis=1)                    # every pair's two columns exchanged
    diagonal = hold(hold(left2 @ hold(S['q'] * lam)) + hold(hold(rows * swapped) @ hold(S['xq'] * lam))).astype(fnp.float64)   # diagB on the response rows
    cmu = cov @ mu
    picked = fnp.unstack(hold(hold(fnp.stack([vv, mu, cmu]))[:, response_indices]))
    X = fnp.stack([picked[0], hold(picked[1] * picked[1]), hold(picked[1] * picked[2])], axis=1)
    Xt = X.T
    coefficients = fnp.linalg.solve(hold(hold(Xt @ X) * hold(fnp.asarray(_NP_RIDGE))), hold(Xt @ diagonal))
    aB = hold(fnp.unstack(coefficients))[0] * _NP_RHO[l]
    S['diag'], S['aB'], S['rows'], S['resp'] = diagonal, aB, rows, response_indices
    if not _NP_STRENGTH[l]:
        return None
    f = mu / hold(vc * (int(pp.shape[0]) / _NP_STRENGTH[l]))                                  # f_i = strength mu_i / (n v_i), every unit
    f32 = hold(hold(f[response_indices]).astype(fnp.float32))
    hold(fnp.subtract(left2, hold(fnp.einsum('ir,i,r->ir', rows, f32, S['q'], optimize=False)), out=left2))
    hold(fnp.subtract(left2, hold(fnp.einsum('ir,i,r->ir', swapped, f32, S['xq'], optimize=False)), out=left2))
    _NP_STATS['folds'] += 1
    return (f, aB)


def _np_forms(l, plane, context, response_mask, alive, cov32):
    """plane = the n x n third-order plane with the read's block in place; adds f_i aB C_ij on that block (rows response, columns alive)."""
    if context is None:
        return
    hold = _E2Hold()
    f, aB = context
    lead = hold(hold(hold(f * response_mask) * aB).astype(fnp.float32))
    hold(fnp.add(plane, hold(fnp.einsum('ij,i,j->ij', cov32, lead, hold(alive.astype(fnp.float32)), optimize=False)), out=plane))


def _np_update(l, c):
    """The weights after layer l (c = the live loop's locals at the births): the carried columns' q and xq, and the newborn pairs' from the law."""
    S = _NP_STATE
    if l == 0:
        S.update(q=None, xq=None, partner=None, diag=None, aB=None, rows=None, resp=None)
    hold = _E2Hold()
    lam, pp, m, mu, vv, aux, d = c['lam'], c['pp'], c['m'], c['mu'], c['vv'], c['aux'], c['d']
    carry, s_new = (1.0, 1.0) if _S19_LAM is None else _S19_LAM
    columns = int(lam.shape[0])
    g, al, ph = aux['gate'], aux['alpha'], aux['pdf']
    gam = g - hold(m * d)
    V = hold(m - hold(m * g)) * 2.0
    gb = fnp.mean(gam)
    cov = c['Cpre']
    cov32 = c['_plane_cov32'] if c['_plane_cov32'] is not None else cov.astype(fnp.float32)
    V32 = hold(V.astype(fnp.float32))
    q_carried = xq_carried = cq = None
    if columns:
        cq = hold(V32 @ pp) * _NP_CONV                                                       # conv x V' pp_r  (c_r = lam_r x this)
        if S['q'] is not None:
            gb32 = hold(hold(gb * (2.0 / carry)).astype(fnp.float32))
            q_carried = hold(cq * (2.0 / carry)) + hold(S['q'] * gb32)
            xq_carried = S['xq'] * gb32
        else:
            q_carried = cq * (2.0 / carry)
            xq_carried = fnp.zeros(columns, dtype=fnp.float32)
    hubs = c['selected']
    K = int(hubs.shape[0])
    if K:
        # per-unit vectors of the law, then their values at the hubs (one gather)
        n = int(mu.shape[0])
        g2 = g * g
        S2 = hold(gam - g2) * 2.0
        gam32 = hold(gam.astype(fnp.float32))
        cc32 = cov32 * cov32
        full = dict(g=g, m=m, d=d, al=al, ph=ph, v=vv, mu=mu, V=V, gam=gam, S2=S2, tp=c['t'], tm=c['s'],   # t+ and t- of every unit, NOT masked by alive: a hub's gap is never 0
                    cv=hold(cov32 @ V32).astype(fnp.float64), ccg=hold(cc32 @ gam32).astype(fnp.float64))
        if S['diag'] is not None:          # diagB lives on the response rows of this layer's read; the few hubs outside them get 0
            diag_full = fnp.zeros(n, dtype=fnp.float64)
            fnp.put(diag_full, S['resp'], S['diag'])
            full['diag'] = diag_full
        k21 = _NP_K21 and l >= 1
        if k21:
            K32 = c['_ps_K2132']
            full.update(kv=hold(K32 @ V32).astype(fnp.float64), ktg=hold(hold(K32.T) @ gam32).astype(fnp.float64), kd=c['_ps_kdiag'])
        if _NP_DIAG:
            g232 = hold(g2.astype(fnp.float32))
            full.update(ccg2=hold(cc32 @ g232).astype(fnp.float64), ccs2=hold(cc32 @ hold(S2.astype(fnp.float32))).astype(fnp.float64))
            if k21:
                kc = hold(K32 * cov32)                                                        # its diagonal is taken out per unit below
                full.update(kcg2=hold(kc @ g232).astype(fnp.float64), kctgd=hold(hold(kc.T) @ hold(hold(g * d).astype(fnp.float32))).astype(fnp.float64))
            if columns and S['rows'] is not None:
                k3_full = fnp.zeros(n, dtype=fnp.float64)
                fnp.put(k3_full, S['resp'], hold(hold(hold(S['rows'] * S['rows']) @ hold(cq * lam)).astype(fnp.float64)))   # (pp o pp) c on the response rows
                full['k3vd'] = k3_full
        names = list(full)
        at = dict(zip(names, fnp.unstack(hold(hold(fnp.stack([full[k] for k in names]))[:, hubs]))))
        hg, hm, hd, hal, hph, hv, hmu, hV, hgam, hS2 = (at[k] for k in ('g', 'm', 'd', 'al', 'ph', 'v', 'mu', 'V', 'gam', 'S2'))
        trace = fnp.sum(vv)
        a_rest = hold(trace * (_NP_CA / 3.0 * c['radial'])) - (S['aB'] if S['aB'] is not None else 0.0)
        hdiag = at['diag'] if 'diag' in at else fnp.zeros(K, dtype=fnp.float64)
        ntd = hdiag + hold(hv * a_rest)
        hv2 = hv * hv
        cov_v = at['cv'] - hold(hv * hV)                                                      # (Co V) at the hubs
        co2gam = at['ccg'] - hold(hv2 * hgam)                                                 # ((Co o Co) gam) at the hubs
        pv = hold(ntd * gb) + hold(co2gam * 2.0)
        alph = hal * hph
        e3h = hold(alph / hv) * -0.5                                                          # e3 / 2
        hal2 = hal * hal
        r2 = hv * hold(hold(hold(hal2 + 1.0) * hg) + alph)
        hm2 = hm * hm
        one_g = 1.0 - hg
        S3 = hold(hold(one_g * 3.0) * hold(r2 - hold(hm2 * 2.0)))
        rv = hold(hold(hd * cov_v) + hold(e3h * pv)) + S3
        if k21:
            kov = at['kv'] - hold(at['kd'] * hV)
            kotg = at['ktg'] - hold(at['kd'] * hgam)
            rv = hold(rv) + hold(hold(hd * kotg) + hold(e3h * kov))
        D = None
        if _NP_DIAG:
            hs = fnp.sqrt(hv)
            r3 = hold(hv * hs) * hold(hold(hold(hold(hal2 + 3.0) * hal) * hg) + hold(hold(hal2 + 2.0) * hph))
            r4 = hv2 * hold(hold(hold(hold(hold(hal2 + 6.0) * hal2) + 3.0) * hg) + hold(hold(hold(hal2 + 5.0) * hal) * hph))
            excess = r2 - hm2
            S4 = hold(hold(hold(r4 - hold(hold(hm * 4.0) * r3)) + hold(hold(hm2 * 6.0) * r2)) - hold(hold(hm2 * hm2) * 3.0)) - hold(hold(excess * excess) * 3.0)
            hg2 = hg * hg
            hco2g2 = at['ccg2'] - hold(hv2 * hg2)
            hco2s2 = at['ccs2'] - hold(hv2 * hS2)
            dgn = hold(hold(hold(S4 + hold(hV * cov_v)) + hold(hold(hgam * co2gam) * 2.0)) - hold(hold(hg2 * hco2g2) * 2.0)) + hold(hold(hgam * gb) * ntd)
            if k21:
                kd_v = at['kd'] * hv
                kcg2 = at['kcg2'] - hold(kd_v * hg2)
                kctgd = at['kctgd'] - hold(kd_v * hold(hg * hd))
                dgn = hold(hold(hold(dgn + hold(hV * kotg)) + hold(hgam * kov)) - hold(hold(hold(hg * hd) * kcg2) * 2.0)) - hold(hold(hg2 * kctgd) * 2.0)
            k3vd = at['k3vd'] if 'k3vd' in at else 0.0
            D = dgn - hold(hg2 * hold(hold(hold(hco2s2 * _NP_BIRTHS) + hold(hdiag * gb)) + k3vd))
            if _NP_AREST:
                gm = hold(hold(hs * hph) + hold(hmu * hg))
                D = hold(D) - hold(hold(hold(r2 - hold(gm * gm)) * a_rest) * gb)
        tp, tm = at['tp'], at['tm']
        gap = tp - tm
        scale = hold(fnp.power(hold(hold(gap * gap) * hold(c['_e2_asel'].astype(fnp.float64) * s_new)), -1.0)) * 2.0   # 2 / (gap^2 lam of the plus column)
        a11 = hS2 * _NP_BIRTHS; a12 = rv * _NP_ROWS
        a22 = (D * _NP_DIAG) if D is not None else 0.0
        q_plus = hold(hold(hold(hold(a11 * tm) * tm) - hold(hold(a12 * tm) * 2.0)) + a22) * scale
        q_minus = hold(hold(hold(hold(a11 * tp) * tp) - hold(hold(a12 * tp) * 2.0)) + a22) * hold(-scale)
        x_plus = hold(hold(hold(a12 * hold(tp + tm)) - hold(hold(a11 * tp) * tm)) - a22) * scale
        born_q = [hold(q_plus.astype(fnp.float32)), hold(q_minus.astype(fnp.float32))]
        born_x = hold(x_plus.astype(fnp.float32))
        born_x = [born_x, hold(-born_x)]
    else:
        born_q = born_x = []
    S['q'] = fnp.concatenate(([q_carried] if columns else []) + born_q) if (columns or K) else None
    S['xq'] = fnp.concatenate(([xq_carried] if columns else []) + born_x) if (columns or K) else None
    pieces = ([S['partner']] if columns else []) + ([fnp.arange(columns + K, columns + 2 * K), fnp.arange(columns, columns + K)] if K else [])
    S['partner'] = fnp.concatenate(pieces) if pieces else None
    S['rows'] = S['resp'] = None
    _NP_STATS['updates'] += 1
    _NP_STATS['columns'] += columns
    _NP_STATS['pairs'] += K
# END OPUS-ENGINEER PAIR LAW OF THE NORM COUPLING

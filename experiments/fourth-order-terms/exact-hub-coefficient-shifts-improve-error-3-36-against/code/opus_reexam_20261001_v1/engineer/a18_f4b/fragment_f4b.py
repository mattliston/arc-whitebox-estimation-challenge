# BEGIN OPUS-ENGINEER F4b (a18 DRAFT, 2026-10-02): native form of the physicist lane's F4b (O/physicist/f4b.py), the two in-place
# column shifts per pair that carry the hub's exact fourth cumulant into the next births. An ERROR lever; generalises F4.
# The hub h of a pair left two columns p+- = s + t+- e_h; at a later layer their pushed columns are P and Q. On the pair's own two
# directions the hub's fourth cumulant (m legs on the hub, 4 - m on its covariance column, coefficients S0..S4 with S0 = 0) is a
# binary quartic with five coefficients fixed at birth,
#     Psi_m = (t+ - t-)^-4 sum_{i <= 4-m, j <= m} binom(4-m, i) binom(m, j) (-1)^j (-t-)^(4-m-i) t+^(m-j) S_(i+j),
# (F4's rule A is the case Psi = (A lam+, 0, 0, 0, A lam-)), and its feed into the births at row k is
#     (d_k / 2) [ tau++ u+ u+' + tau+- (u+ u-' + u- u+') + tau-- u- u-' ],   u+- = g o P, g o Q the pair's gated columns,
#     tau++ = Psi0 P_k^2 + 2 Psi1 P_k Q_k + Psi2 Q_k^2,  tau+- = Psi1 P_k^2 + 2 Psi2 P_k Q_k + Psi3 Q_k^2,  tau-- = Psi2 P_k^2 + 2 Psi3 P_k Q_k + Psi4 Q_k^2.
# Two shifts can carry the u+ u+' and u- u-' parts; the cross part is projected on them in the least-squares sense, which needs only
# the Gram of the two gated columns (spp = u+.u+, spq = u+.u-, sqq = u-.u-):
#     x+ = (d/2) (tau++ + k+ tau+-),  k+ = 2 spq sqq / (spp sqq + spq^2);    x- = (d/2) (tau-- + k- tau+-),  k- = 2 spq spp / (spp sqq + spq^2).
# (a18_f4b/check_projection.py: equal to f4b.py's 3 x 3 Gram, 3 x 2 design and batched 2 x 2 solve to 1e-15.)
# The columns move by x+- / lam+-. The plane this adds to the bank is handled as in F4: beta_k = sum_i v_i P_ik / sum v^2 with the exact
# diagonal; anchor 'star' takes v_i beta_k + 2 C_ik beta_i out of the star, anchor 'b' lowers the carried b by beta instead.
# Native form: live rows only, float32 planes, per cohort three squares (P^2, P Q, Q^2), three column sums, two quadratic forms, and
# the anchor's matrix-vector passes: 47 flops per element of (live rows x pairs) = 23.5 per column element (F4: 16).
_F4B_C = 0.5
_F4B_AGE = 2
_F4B_ANCHOR = 'star'
_F4B_TERMS = (1, 2, 3, 4)
_F4B_LAYERS = (1, 14)
_F4B_PSI = {}                                     # birth layer -> Psi (5, K) float64; this prediction
_F4B_CONST = {}
_F4B_STATS = dict(layers=0, pairs=0, elements=0)


def _f4b_const():
    C = _F4B_CONST
    if not C:
        f64 = fnp.float64
        binom = ((1,), (1, 1), (1, 2, 1), (1, 3, 3, 1), (1, 4, 6, 4, 1))
        basis = [[[[0.0] * 5 for _b in range(5)] for _a in range(5)] for _m in range(5)]
        for m in range(5):
            for i in range(5 - m):
                for j in range(m + 1):
                    basis[m][4 - m - i][m - j][i + j] += float(binom[4 - m][i] * binom[m][j] * (-1) ** j)
        C['basis'] = fnp.asarray(basis, dtype=f64)                       # [m, power of -t-, power of t+, index of S]
        C['expo'] = fnp.arange(5, dtype=f64)[:, None]
        # Gaussian moments of the rectified hub in units of sigma (as in K4L): rows . powers of alpha, times gate / pdf
        C['moments'] = fnp.asarray([[1, 0, 1, 0, 0], [0, 3, 0, 1, 0], [3, 0, 6, 0, 1],
                                    [0, 1, 0, 0, 0], [2, 0, 1, 0, 0], [0, 5, 0, 1, 0]], dtype=f64)
        keep = [1.0 if m in _F4B_TERMS else 0.0 for m in (1, 2, 3, 4)]
        C['hub'] = fnp.asarray([[0.0] * 13,                                                        # S0 = 0
                                [-keep[0]] + [0.0] * 12,
                                [0, 2 * keep[1], -2 * keep[1], -2 * keep[1]] + [0.0] * 9,
                                [0.0] * 4 + [3 * keep[2], -6 * keep[2], -3 * keep[2], 6 * keep[2]] + [0.0] * 5,
                                [0.0] * 8 + [keep[3], -4 * keep[3], 12 * keep[3], -6 * keep[3], -3 * keep[3]]], dtype=f64)
        C['sigma_power'] = fnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=f64)[:, None]             # S_m = sigma^(2m-4) x (S_m in units of sigma)
    return C


def _f4b_birth(l, aux, host, selected, t, s):
    """Psi of the pairs born at layer l (exact hub coefficients S1..S4 of O/K4L_SPEC.md, host = the pair map's mean)."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    C = _f4b_const()
    if l == 0:
        _F4B_PSI.clear()
        _F4B_STATS.update(layers=0, pairs=0, elements=0)
    a, sg, g, ph, m, tp, tm = fnp.unstack(_e2_H(fnp.stack([aux['alpha'], aux['sigma'], aux['gate'], aux['pdf'], host, t, s]))[:, selected])
    G = C['moments'] @ _e2_H(fnp.power(a, C['expo']))
    Gp = _e2_H(G[3:]) * ph
    rho2, rho3, rho4 = fnp.unstack(_e2_H(_e2_H(G[:3]) * g) + Gp)
    one, mt, mt2, _unused, mt4 = fnp.unstack(fnp.power(_e2_H(m / sg), C['expo']))
    L = fnp.stack([_e2_H(Gp[0]), g, g, mt, rho2, mt2, g, g, rho4, rho3, rho2, mt4, rho2])
    R = fnp.stack([one, one, g, ph, one, one, rho2, mt2, one, mt, mt2, one, rho2])
    S = _e2_H(fnp.einsum('qk,kj,kj->qj', C['hub'], L, R, optimize=False)) * _e2_H(fnp.power(sg, _e2_H(C['sigma_power'] * 2)))   # S0..S4 (5, K)
    U = fnp.einsum('mabn,ak,bk->mnk', C['basis'], _e2_H(fnp.power(_e2_H(-tm), C['expo'])), _e2_H(fnp.power(tp, C['expo'])), optimize=False)
    gap = tp - tm
    g2 = gap * gap
    g4 = g2 * g2
    usable = fnp.greater(g4, 1e-48)             # the parent keeps the gap above a tenth of the hub's variance; a zero gap gets Psi = 0
    _F4B_PSI[l] = fnp.where(usable, _e2_H(_e2_H(fnp.einsum('mnk,nk->mk', U, S, optimize=False)) / _e2_H(fnp.maximum(g4, 1e-48))), 0.0)


def _f4b_step(l, right, g, d, live_indices, v, lam, state):
    """End of step l. right: pushed young columns on the live rows (float32); g, d: gate and pdf / sigma (n,); v: diagonal of the
    carried post covariance (n,); lam: carried third-order weights before this layer's dials; state: old-tier state (cohort roster).
    Returns None, or (shifts, beta, diagonal): shifts = [(first column counted from the end of the young roster, block), ...] to add to
    the gated young columns, beta (n,) and diagonal (n,) float64 for the anchor."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    f32 = fnp.float32
    if not (_F4B_LAYERS[0] <= l <= _F4B_LAYERS[1]):
        return None
    young = [j for j in range(max(l - _F4B_AGE, 0), l)]
    if not young:
        return None
    n = int(g.shape[0])
    nc = int(lam.shape[0])
    old = int(state['columns'])
    hd = d * 0.5
    vg = v * g
    rows = _e2_H(_e2_H(fnp.stack([_e2_H(g * g), _e2_H(vg * g), vg, hd, g]))[:, live_indices]).astype(f32)
    om, vg2, vgl, hl, gl = fnp.unstack(rows)
    hcol = hl[:, None]
    shifts, t1_parts, t2_parts, dg_parts = [], [], [], []
    for j in young:
        start, width = state['cohorts'][j]
        start, K = int(start), int(width) // 2
        if K == 0:
            continue
        P = right[:, start - old:start - old + K]
        Q = right[:, start - old + K:start - old + 2 * K]
        PP = P * P
        QQ = Q * Q
        PQ = P * Q
        spp = _e2_H(om @ PP).astype(fnp.float64)
        sqq = _e2_H(om @ QQ).astype(fnp.float64)
        spq = _e2_H(om @ PQ).astype(fnp.float64)
        cross = spp * sqq
        sq2 = spq * spq
        den = fnp.maximum(cross + sq2, 1e-300)       # a dead column gives 0 / 1e-300 = 0 below, not 0 / 0
        ok = fnp.greater(_e2_H(_e2_H(cross - sq2) * den), _e2_H(_e2_H(cross * cross) * 1e-8))        # the two columns must differ
        kp = _e2_H(_e2_H(2 * spq) * sqq) / den
        km = _e2_H(_e2_H(2 * spq) * spp) / den
        psi0, psi1, psi2, psi3, psi4 = fnp.unstack(_F4B_PSI[j])
        lp = lam[start:start + K]
        lm = lam[start + K:start + 2 * K]
        if _S19_LAM is not None:
            lp = lp * _S19_LAM[0]
            lm = lm * _S19_LAM[0]
        cp = _e2_H(ok * _F4B_C) / lp                 # strength over the column's weight; 0 for a degenerate pair
        cm = _e2_H(ok * _F4B_C) / lm
        coef = _e2_H(fnp.stack([_e2_H(_e2_H(psi0 + _e2_H(kp * psi1)) * cp), _e2_H(_e2_H(_e2_H(psi1 + _e2_H(kp * psi2)) * cp) * 2),
                                _e2_H(_e2_H(psi2 + _e2_H(kp * psi3)) * cp), _e2_H(_e2_H(psi2 + _e2_H(km * psi1)) * cm),
                                _e2_H(_e2_H(_e2_H(psi3 + _e2_H(km * psi2)) * cm) * 2), _e2_H(_e2_H(psi4 + _e2_H(km * psi3)) * cm)])).astype(f32)
        ap, bp_, gp, am, bm_, gm = fnp.unstack(coef)
        dp = PP * ap                                  # the shift of the + columns: (d/2) (ap P^2 + bp P Q + gp Q^2)
        x = PQ * bp_
        _e2_H(fnp.add(dp, x, out=dp))
        _e2_H(fnp.multiply(QQ, gp, out=x))
        _e2_H(fnp.add(dp, x, out=dp))
        _e2_H(fnp.multiply(dp, hcol, out=dp))
        dm = PP * am
        _e2_H(fnp.multiply(PQ, bm_, out=x))
        _e2_H(fnp.add(dm, x, out=dm))
        _e2_H(fnp.multiply(QQ, gm, out=x))
        _e2_H(fnp.add(dm, x, out=dm))
        _e2_H(fnp.multiply(dm, hcol, out=dm))
        # anchor sums; the added tensor is sum_r lam_r Sym(u_r, u_r, delta_r)
        lp32 = lp.astype(f32) if str(lp.dtype) != 'float32' else lp
        lm32 = lm.astype(f32) if str(lm.dtype) != 'float32' else lm
        t1 = _e2_H(dp @ _e2_H(lp32 * _e2_H(vg2 @ PP))) + _e2_H(dm @ _e2_H(lm32 * _e2_H(vg2 @ QQ)))
        _e2_H(fnp.multiply(P, dp, out=x))
        s2p = vgl @ x
        _e2_H(fnp.multiply(Q, dm, out=x))
        s2m = vgl @ x
        t2 = _e2_H(P @ _e2_H(lp32 * s2p)) + _e2_H(Q @ _e2_H(lm32 * s2m))
        _e2_H(fnp.multiply(PP, dp, out=x))
        dgp = x @ lp32
        _e2_H(fnp.multiply(QQ, dm, out=x))
        dgm = x @ lm32
        t1_parts.append(t1); t2_parts.append(t2); dg_parts.append(_e2_H(dgp + dgm))
        shifts.append((nc - start, dp))
        shifts.append((nc - start - K, dm))
        _F4B_STATS.update(pairs=_F4B_STATS['pairs'] + K, elements=_F4B_STATS['elements'] + int(P.shape[0]) * K)
    if not shifts:
        return None
    t1, t2, dg = t1_parts[0], t2_parts[0], dg_parts[0]
    for i in range(1, len(t1_parts)):
        t1 = _e2_H(t1) + t1_parts[i]
        t2 = _e2_H(t2) + t2_parts[i]
        dg = _e2_H(dg) + dg_parts[i]
    beta_l = _e2_H(_e2_H(t1 + _e2_H(_e2_H(2 * gl) * t2)).astype(fnp.float64)) / _e2_H(fnp.sum(_e2_H(v * v)))
    diag_l = _e2_H(_e2_H(_e2_H(3 * gl) * gl) * dg).astype(fnp.float64)
    beta = fnp.zeros(n, dtype=fnp.float64)
    _e2_H(fnp.put(beta, live_indices, beta_l))
    diagonal = fnp.zeros(n, dtype=fnp.float64)
    _e2_H(fnp.put(diagonal, live_indices, diag_l))
    _F4B_STATS['layers'] = _F4B_STATS['layers'] + 1
    return shifts, beta, diagonal
# END OPUS-ENGINEER F4b

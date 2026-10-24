# PHYSICIST COPY (O/physicist/f4b_t138.py, 2026-10-13; lead 07:0xZ, Astra's F4 nonlinear-remainder lead) of f4b.py (sha 3d7a2d12) with ONE
# read-only diagnostic block after the shift (env T138_DUMP = a JSONL path; unset = f4b bit for bit): per layer, for the shifted young columns u
# (before) and the ACTUAL stored shift e = after - before (float64), weights lam: the first-chaos vectors cL = 3 sum lam (a e + 2 c u) of the
# intended linear term L = 3 sum lam (u.x)^2 (e.x) and cN = 3 sum lam (d u + (2c + d) e) of the remainder N = sum lam [3 (u.x)(e.x)^2 + (e.x)^3]
# (a = |u|^2, c = u.e, d = |e|^2; Astra F4_NONLINEAR_SIGNED_CROSS_v1.txt), and the repeated-index (K21) slices RL = sum lam [2 u (u o e)' + e (u^2)'],
# RN = sum lam [(u + e)(e^2)' + 2 e (u o e)'], RU = sum lam u (u^2)' (the shifted columns' own plane), all rows and kink rows (|alpha| < 1.5).
"""TAG opus-physicist F4b: drop-in for the torch research card (same hook points as f4.py).
THE EXACT LOCAL FOURTH CUMULANT OF EVERY YOUNG HUB, FED INTO BIRTHS BY TWO IN-PLACE COLUMN SHIFTS PER PAIR ("RIDE" of the lead's E15).

Physics. A hub h born at layer j left the pair p+- = s + t+- e_h (weights lam+-). The ReLU at h also makes a fourth cumulant with
m legs on the hub and 4-m legs on the covariance column s, with exactly known coefficients (Gaussian z_h; lead, K4L_SPEC.md):
     S0 = 0,  S1 = E f''' = -alpha pdf / v,  S2 = 2 (g - g^2 - m d),  S3 = 3 (1 - g)(E x^2 - 2 m^2),  S4 = kappa4(x).
At a later layer the pair's pushed columns give the transported legs  Wt = (pp+ - pp-)/(t+ - t-),  Y = (t+ pp- - t- pp+)/(t+ - t-),
and the feed of that fourth cumulant into the births with hub k is
     (d_k/2) g_i g_j Q_ijkk = (gY)(gY) h1_k + [(gY)(gWt) + (gWt)(gY)] h2_k + (gWt)(gWt) h3_k,
     h_q = (d/2) (S_q Y^2 + 2 S_{q+1} Y Wt + S_{q+2} Wt^2),   q = 0, 1, 2.
Shifting the two carried columns u+- = g o pp+- by delta+- adds  sum+- lam+- [(gY)(gY) + t+- (..) + t+-^2 (gWt)(gWt)] delta+-, so the
pair can carry x+- = lam+- delta+- with  (h1, h2, h3) = x+ (1, t+, t+^2) + x- (1, t-, t-^2)  in the least-squares sense; the metric is
the Gram matrix of the three (i, j) matrix types. f4.py is the special case S_m = A sum+- lam+- t+-^m (mode 'ruleA': S1 right, S2..S4
set by the pair's geometry), which lies exactly in the span; that identity is this module's certificate (f4b_cert.py).

The plane of the added tensor (entries with a repeated index) is modelled as in f4.py by the b-carrier form v_i beta_k + 2 C_ik beta_i,
beta_k = sum_i v_i P_ik / sum_i v_i^2, with the exact diagonal 3 sum_r u_ri^2 x_ri. Two ways to keep the card's plane where it was:
   anchor='star' : that plane model is removed from the star (f4.py's cheap anchor); the DISTINCT content of the added tensor keeps
                   its part of the form Sym(C x beta)
   anchor='b'    : the whole tensor Sym(C x beta) is removed, by lowering the carried b vector by beta (plane and distinct content);
                   the star only gets the diagonal difference. This removes the part of the local feed that the card's isotropic
                   b source (radial d v / 6, the sum over ALL hubs of the same vertex) already holds: no double counting.
   anchor=<int L>: 'b' at the end of steps 1..L, 'star' afterwards (the double counting is large only at layers 1-4).

install(cd, torch, c=1.0, age=2, mode='exact', terms=(1, 2, 3, 4), anchor='star', layers=(1, 14))   c: strength on S1..S4 (physical
   value 1; in mode 'ruleA' it is f4.py's theta); terms: which S_m are kept (exact mode only); layers: the shift is applied at the end
   of steps lo..hi only (a shift at the end of step l feeds the births of layer l and the third cumulant of layer l+1).
   mode='mix': S1..S4 = cs[m-1] * exact S_m + ths[m-2] * (what F4's rule implies for S_m, m = 2, 3, 4); c and terms are ignored. The
   seven numbers may be floats or 0-dim tensors (trainable dials). F4 at theta = mix with cs = (theta, 0, 0, 0), ths = (theta,) * 3.
   ths with FOUR entries includes the rule's own S1 (then F4 at theta is cs = (0, 0, 0, 0), ths = (theta,) * 4, and 'F4 at theta plus the
   exact S1, S2 at c' is cs = (c, c, 0, 0), ths = (theta,) * 4: the literal mixture of the held-out fit).
   cs and ths may also be dicts {age: tuple} (age = 1 .. `age`) for strengths that depend on the age of the cohort.
   even2: the exact S2 is split into its part odd in alpha, -alpha pdf (the same as the odd part of F4's rule), and the rest (even in
   alpha: +0.18 at alpha = 0, +0.05 at |alpha| = 0.75); even2 multiplies the even part (exact and mix modes).
   iso2: the card's isotropic fourth cumulant radial (C C + C C + C C) / 3 already holds, for every hub, a term of the S2 form with
   coefficient (2/3) radial_post g_h^2 (0.002 at layer 0, 0.012-0.014 at layers 9-13 on network 35; the hub mean of the exact even
   part is 0.18 at layer 0 and 0.015-0.02 at layers 8-13); iso2 times that share is subtracted from S2.
   metric='birth': the least-squares metric of the two-shift solve uses |gY|^2 : |gWt|^2 = |s|^2 : 1 (s = the pair's common leg at
   birth) and no cross term, instead of the three dot products of the current legs; then the six numbers that turn (Y^2, Y Wt, Wt^2)
   into the two shifts are constants of the pair, fixed at its birth (the native form needs no per-pair algebra at later layers).
   coll=x: the lead's COLL term of E15b at strength x, on every unjoined carried column at most coll_age layers old: the column shifts
   along the common vector d/2 by its own scalar c_r = (2/n) V21 . (the column's leg left ungated one layer earlier), V21 = 2 (m_G - m g);
   it goes through the same anchor as the other shifts (use anchor='b' to take its isotropic part out of the carried b).
   form='own': instead of the general two shifts, each column shifts along its own square only, x+- = a+- (d/2) pp+-^2 (F4's form:
   the native F4 step with two birth-like scalars per pair); (a+, a-) = weighted least squares of the same target.
Returns cd; cd.f4b_info[l] holds per-layer diagnostics (rho = projection of beta on the card's b source, masked = pairs skipped
because their two columns are degenerate). Checkpoint-safe (birth data in a per-layer dict, detached).
Held-out results and the port form: RESULT.md T7, T13, T14; plain-array references f4b_native_ref.py (per-layer metric) and
f4c_native_ref.py (pair constants fixed at birth, bit-identical to mode='mix', metric='birth')."""


import os as _os138
_T138 = _os138.environ.get('T138_DUMP')
_T138_RANK = [_os138.environ.get('T138_RANK')]


def install(cd, torch, c=1.0, age=2, mode='exact', terms=(1, 2, 3, 4), anchor='star', layers=(1, 14), cs=(1.0, 1.0, 0.0, 0.0), ths=(0.0, 0.0, 0.0), even2=1.0, iso2=0.0, form='ride', metric='gram', coll=0.0, coll_age=99):
    t = torch; side = dict(B={}); info = {}; cd.f4b_info = info; cd.f4b_side = side; f64 = t.float64
    push0 = cd.shared['push']

    def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        r = push0(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
        side['pp'], side['oc'] = r[0], int(r[4])
        return r
    cd.shared['push'] = push
    bd0 = cd.env['_bd_update']

    def bd(bp, Cpre, K21, aux, d, alive, births, a, tplus, tminus, coefficients):
        side['tpm'] = (tplus.detach(), tminus.detach())
        return bd0(bp, Cpre, K21, aux, d, alive, births, a, tplus, tminus, coefficients)
    cd.env['_bd_update'] = bd
    sel0 = cd.select

    def select(score):
        r = sel0(score); side['sel'] = r[0]
        return r
    cd.select = select
    prev = cd.hooks.get('state_out')

    def state_out(l, out, ctx):
        if prev is not None: out = prev(l, out, ctx)
        if l > 14: return out
        aux = ctx['aux']; C, p, lam, star, b, alive, stops = out[1], out[2], out[3], out[4], out[5], out[6], out[12]
        pp = side['pp']; nc = int(pp.shape[1])
        young = [j for j in range(max(l - age, 0), l)] if layers[0] <= l <= layers[1] else []
        if young:
            dh = (aux['pdf'] / aux['sigma'] / 2) * alive
            us, xs, cols, masked = [], [], [], 0
            for j in young:
                B = side['B'][j]; k0 = int(stops[j - 1]) if j else 0; K = (int(stops[j]) - k0) // 2
                assert K == B['tp'].numel() and k0 >= side['oc'] and k0 + 2 * K <= nc, (l, j, K, k0, side['oc'], nc)
                tp, tm = B['tp'][None, :], B['tm'][None, :]; gap = tp - tm
                ok = gap[0].abs() > 1e-12; gap = t.where(ok[None, :], gap, t.ones_like(gap))
                pplus, pminus = pp[:, k0:k0 + K].to(f64), pp[:, k0 + K:k0 + 2 * K].to(f64)
                uplus, uminus = p[:, k0:k0 + K].to(f64), p[:, k0 + K:k0 + 2 * K].to(f64)
                Wt = (pplus - pminus) / gap; Y = (tp * pminus - tm * pplus) / gap
                a = (tp * uminus - tm * uplus) / gap; bb_ = (uplus - uminus) / gap
                aa, bb, ab = (a * a).sum(0), (bb_ * bb_).sum(0), (a * bb_).sum(0)
                if metric == 'birth':                                                                                # a metric known at the pair's birth: |gY|^2 / |gWt|^2 ~ |s|^2, no cross term
                    aa, bb, ab = B['s2'], t.ones_like(aa), t.zeros_like(aa)
                Gam = t.stack((t.stack((aa * aa, 2 * aa * ab, ab * ab)), t.stack((2 * aa * ab, 2 * (aa * bb + ab * ab), 2 * bb * ab)),
                               t.stack((ab * ab, 2 * bb * ab, bb * bb)))).permute(2, 0, 1)                          # K x 3 x 3
                one = t.ones(K, dtype=f64)
                Bm = t.stack((t.stack((one, one)), t.stack((tp[0], tm[0])), t.stack((tp[0] ** 2, tm[0] ** 2)))).permute(2, 0, 1)   # K x 3 x 2
                BtG = Bm.transpose(1, 2) @ Gam; G2 = BtG @ Bm                                                        # K x 2 x 3, K x 2 x 2
                det = G2[:, 0, 0] * G2[:, 1, 1] - G2[:, 0, 1] * G2[:, 1, 0]
                ok = ok & (G2[:, 0, 0] > 0) & (G2[:, 1, 1] > 0) & (det > 1e-8 * G2[:, 0, 0] * G2[:, 1, 1])              # the two columns must differ
                G2 = t.where(ok[:, None, None], G2, t.eye(2, dtype=f64)[None]); BtG = BtG * ok[:, None, None]
                R = t.linalg.solve(G2, BtG); masked += int((~ok).sum())                                               # K x 2 x 3
                lp, lm = lam[k0:k0 + K].to(f64), lam[k0 + K:k0 + 2 * K].to(f64)
                Sr = [B['A'] * (lp * tp[0] ** m + lm * tm[0] ** m) for m in range(5)]                               # what F4's rule implies
                if mode == 'exact': ce, cr = [c if m in terms else 0.0 for m in (1, 2, 3, 4)], [0.0] * 5
                elif mode == 'ruleA': ce, cr = [0.0] * 4, [c] * 5
                elif mode == 'mix':
                    ce, th_ = list(cs[l - j] if isinstance(cs, dict) else cs), list(ths[l - j] if isinstance(ths, dict) else ths)
                    cr = [0.0] + th_ if len(th_) == 4 else [0.0, 0.0] + th_                                         # four entries: the rule's own S1 too (= F4 itself at that strength)
                else: raise ValueError(mode)
                Se = (B['S'][0], B['S2o'] + even2 * (B['S'][1] - B['S2o']) - iso2 * B['S2i'], B['S'][2], B['S'][3])  # S2 = odd + even2 x even - iso2 x isotropic share
                S = [cr[0] * Sr[0]] + [ce[m - 1] * Se[m - 1] + cr[m] * Sr[m] for m in (1, 2, 3, 4)]
                M = t.stack((t.stack((S[0], 2 * S[1], S[2])), t.stack((S[1], 2 * S[2], S[3])), t.stack((S[2], 2 * S[3], S[4])))).permute(2, 0, 1)
                Nm = R @ M                                                                                           # K x 2 x 3 on (Y^2, Y Wt, Wt^2)
                m1, m2, m3 = Y * Y, Y * Wt, Wt * Wt
                if form == 'own':                                                                                    # each column shifts along its OWN square only (F4's form), two scalars per pair
                    tt = t.stack((tp[0], tm[0]))                                                                     # 2 x K
                    cvec = t.stack((t.ones_like(tt), 2 * tt, tt * tt), 1)                                            # 2 x 3 x K: (1, 2t, t^2) on (Y^2, Y Wt, Wt^2)
                    tq = t.stack((t.ones_like(tt), tt, tt * tt), 1)                                                  # 2 x 3 x K: t^q, q = 0, 1, 2
                    wq = t.stack((aa * aa, 2 * (aa * bb + ab * ab), bb * bb))                                        # 3 x K   weight of the three (i, j) matrix types
                    wr = t.stack(((dh[:, None] * m1).pow(2).sum(0), (dh[:, None] * m2).pow(2).sum(0), (dh[:, None] * m3).pow(2).sum(0)))   # 3 x K   weight of the three monomials
                    basis = tq[:, :, None, :] * cvec[:, None, :, :]                                                  # 2 x 3 x 3 x K: basis[s, q, r] = t_s^q c_s,r
                    wgt = wq[:, None, :] * wr[None, :, :]                                                            # 3 x 3 x K
                    Gm = t.einsum('sqrk,qrk,uqrk->suk', basis, wgt, basis); rhs = t.einsum('sqrk,qrk,kqr->sk', basis, wgt, M)
                    det_ = Gm[0, 0] * Gm[1, 1] - Gm[0, 1] * Gm[1, 0]; good = ok & (det_ > 1e-10 * Gm[0, 0] * Gm[1, 1])
                    det_ = t.where(good, det_, t.ones_like(det_))
                    a_p = t.where(good, (Gm[1, 1] * rhs[0] - Gm[0, 1] * rhs[1]) / det_, t.zeros_like(det_))
                    a_m = t.where(good, (Gm[0, 0] * rhs[1] - Gm[1, 0] * rhs[0]) / det_, t.zeros_like(det_))
                    Nm = t.stack((t.stack((a_p, 2 * a_p * tp[0], a_p * tp[0] ** 2), 1), t.stack((a_m, 2 * a_m * tm[0], a_m * tm[0] ** 2), 1)), 1)   # K x 2 x 3
                x = [dh[:, None] * (Nm[:, s, 0][None, :] * m1 + Nm[:, s, 1][None, :] * m2 + Nm[:, s, 2][None, :] * m3) for s in (0, 1)]
                us += [uplus, uminus]; xs += x; cols.append((k0, K))
            u64 = t.cat(us, 1); x64 = t.cat(xs, 1); k_lo = cols[0][0]; k_hi = cols[-1][0] + 2 * cols[-1][1]; assert k_hi == nc, (l, k_hi, nc)
            if coll and 'coll' in side:                                                                             # the lead's COLL term on every unjoined carried column of age <= coll_age
                k_c = max(side['oc'], int(stops[l - coll_age - 1]) if l - coll_age >= 1 else 0)
                if k_c < k_lo:                                                                                       # widen the shifted block; the S-term shift is zero on the older columns
                    u_old = p[:, k_c:k_lo].to(f64); u64 = t.cat((u_old, u64), 1); x64 = t.cat((t.zeros_like(u_old), x64), 1); k_lo = k_c
                cc = side['coll'][k_lo:k_hi]; assert cc.numel() == k_hi - k_lo, (l, cc.numel(), k_hi - k_lo)
                x64 = x64 + coll * dh[:, None] * (lam[k_lo:k_hi].to(f64) * cc)[None, :]
            ly = lam[k_lo:k_hi].to(f64); ly = t.where(ly.abs() > 0, ly, t.ones_like(ly))
            new = (u64 + x64 / ly[None, :]).to(p.dtype)
            if _T138:                                                                                     # PHYSICIST T138: read-only diagnostic
                with t.no_grad():
                    U = u64.detach(); E = new.detach().to(f64) - U; lm = lam[k_lo:k_hi].detach().to(f64)
                    aa_, cc_, dd_ = (U * U).sum(0), (U * E).sum(0), (E * E).sum(0)
                    cL = 3 * (E @ (lm * aa_) + 2 * (U @ (lm * cc_))); cN = 3 * (U @ (lm * dd_) + E @ (lm * (2 * cc_ + dd_)))
                    RL = 2 * ((U * lm[None, :]) @ (U * E).T) + (E * lm[None, :]) @ (U * U).T
                    RN = ((U + E) * lm[None, :]) @ (E * E).T + 2 * ((E * lm[None, :]) @ (U * E).T)
                    RU = (U * lm[None, :]) @ (U * U).T
                    kr = (aux['alpha'].detach().abs() < 1.5)
                    fro = lambda M, rows=None: float((M if rows is None else M[rows]).pow(2).sum().sqrt())
                    rec = dict(layer=int(l), K=int(U.shape[1]), e_over_u_median=float((dd_.sqrt() / aa_.sqrt().clamp_min(1e-300)).median()),
                               cL=float(cL.norm()), cN=float(cN.norm()), cos_cL_cN=float((cL @ cN) / (cL.norm() * cN.norm()).clamp_min(1e-300)),
                               RL=fro(RL), RN=fro(RN), RU=fro(RU), RL_kink=fro(RL, kr), RN_kink=fro(RN, kr), RU_kink=fro(RU, kr),
                               cos_RL_RN=float((RL * RN).sum() / (RL.norm() * RN.norm()).clamp_min(1e-300)), rank=_T138_RANK[0])
                    with open(_T138, 'a') as fh: fh.write(__import__('json').dumps(rec) + '\n')
            v = t.diagonal(C); uu = u64 * u64
            beta = ((x64 * (v @ uu)[None, :]).sum(1) + 2 * (u64 @ (v @ (u64 * x64)))) / (v * v).sum()
            dg = 3 * (uu * x64).sum(1)
            how = anchor if isinstance(anchor, str) else ('b' if l <= anchor else 'star')
            if how == 'star':
                dP = t.diagonal_scatter(v[:, None] * beta[None, :] + 2 * C * beta[:, None], dg)
                star = star - (3 * dP.T - 2 * t.diag(t.diagonal(dP))).to(star.dtype)
            elif how == 'b':
                star = star - t.diag(dg - 3 * v * beta).to(star.dtype); b = b - beta.to(b.dtype)
            else: raise ValueError(anchor)
            p = t.cat((p[:, :k_lo], new, p[:, k_hi:]), dim=1)
            src = ctx['radial'] * aux['pdf'] * aux['sigma'] * alive / 6                                             # the card's b source radial d v / 6
            info[l] = dict(rho=float((beta * src).sum() / (src * src).sum()), beta_norm_over_b=float(beta.norm() / b.norm().clamp_min(1e-300)),
                           cos_beta_b=float((beta * b).sum() / (beta.norm() * b.norm()).clamp_min(1e-300)), masked=masked)
        sel = side['sel']; f = lambda x: x.detach().to(f64)[sel]
        al, s, gt, ph, m = f(aux['alpha']), f(aux['sigma']), f(aux['gate']), f(aux['pdf']), f(ctx['host']); vv = s * s
        r2 = vv * ((1 + al * al) * gt + al * ph); r3 = vv * s * ((al ** 3 + 3 * al) * gt + (al * al + 2) * ph)
        r4 = vv * vv * ((al ** 4 + 6 * al * al + 3) * gt + (al ** 3 + 5 * al) * ph)
        S = (-al * ph / vv, 2 * (gt - gt * gt - m * ph / s), 3 * (1 - gt) * (r2 - 2 * m * m), r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2)
        k0n = int(stops[l - 1]) if l else 0; Kn = (int(stops[l]) - k0n) // 2; tpn = f(side['tpm'][0])
        s2 = ((out[2][:, k0n:k0n + Kn].detach().to(f64) ** 2).sum(0) - tpn * tpn).clamp_min(1e-30)                  # |s|^2 of the newborn pair's common leg
        side['B'][l] = dict(tp=tpn, tm=f(side['tpm'][1]), S=S, s2=s2, S2o=-al * ph, S2i=(2.0 / 3.0) * out[7].detach().to(f64) * gt * gt, A=(-al / (s * (1 + ctx['radial'] / 3))).detach().to(t.float32).to(f64))
        if coll:                                                                                                    # c_r = (2/n) V21 . (the column's leg left ungated at this layer), for the next step
            V21 = (2 * (aux['gaussian_mean'] - ctx['host'] * aux['gate']) * alive).detach().to(f64)
            c_car = (2.0 / V21.numel()) * (V21 @ pp.to(f64)) if nc else t.zeros(0, dtype=f64)
            c_new = (2.0 / V21.numel()) * (V21 @ ctx['Cpre'].detach().to(f64)[:, sel])
            side['coll'] = t.cat((c_car, c_new, c_new))
        return out[:2] + (p, lam, star, b) + out[6:]
    cd.hooks['state_out'] = state_out
    return cd

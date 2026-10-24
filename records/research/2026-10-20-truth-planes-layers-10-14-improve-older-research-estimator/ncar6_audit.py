"""TAG opus-physicist NCAR6_AUDIT: a COPY of the lead's O/ncar6.py (version of 2026-10-02 14:00Z by its file name) with two changes for the audit: (1) at level r31 the
network-specific matrix R, the whole matrix B and the coefficient diagnostics are dumped at every layer even when the strength is 0 (passive run);
(3) option gain: the law's homogeneity term carried on the pairs (1: kappa by the Frobenius projection, 2: kappa from the diagonals), which also supplies the multiple of C that
is not on the pairs (a_iso = 2 kappa of the layer before) in place of 0.75 radial / 3; (4) dfit 5 / 6: the diagonal regression with a constant regressor (no per-layer table);
(8) options k4 / k4last: the diagonal's per-unit part in the per-unit fourth cumulant, k4_i += k4 (6 / n) r_i at layers 1-14 (pair map input) and at the final layer (k4last);
(7) dfit 7 / 8: a stand-in for the C^2 form, Z Z' with Z = C Q (Q = pk Krylov vectors of the mean direction / the top-pk covariance eigenvectors), its coefficient fitted on the
diagonal together with tr C x C, the two mean forms and a constant; (6) option pmproj = K: after the forms, the one-sided form on the mean direction and its K - 1 next Krylov vectors is removed from the plane;
(5) option k22: the diagonal's per-unit part (residual of the diagonal regression, response rows) in the pair map's K22 plane, K22_ij += k22 (r_i + r_j) / n;
(2) dump[l]['coef'] holds the multiple of C by the exact off-diagonal fits and by diagonal regressions with and without a constant regressor.
Everything else is the lead's text, unchanged. Original header: TAG opus-reexam NCAR6 (lead, 2026-10-02): THE WHOLE LEAN LAW ON THE BANK'S OWN COLUMN PAIRS: no new columns, no eigenvectors, no probes, no rank cut.

The fresh sources of the second-chaos matrix (physicist's certified lean law, nlaw.py) are, in the post space of layer l,
    births      g g' o (Co S2 Co)        = sum_h S2_h c_h c_h'                 c_h = g o Co[:, h]   (the hub's coupling vector)
    row forms   R Co G + G Co R          = sum_h rv_h (e_h c_h' + c_h e_h')    e_h = unit vector of hub h
    diagonal                             = sum_h D_h e_h e_h'
and the card's third-order bank is born on exactly those two vectors per hub: its newborn columns are p+- = s_h + t+- e_h with s_h = c_h plus the card's
(small) birth-direction changes. So each hub's contribution is a symmetric 2 x 2 form on its own column pair,
    [s e] M_h [s e]',  M_h = [[S2_h, rv_h], [rv_h, D_h]],     s = (t+ p- - t- p+) / gap,  e = (p+ - p-) / gap,  gap = t+ - t-,
and the bank's own transport (columns pushed through W and the gates, joins included) carries it for nothing:
    one second weight per column   w_r <- 2 (gbar w_r + lam_r (V . pp_r))        (the physicist's bank channel: transport + conversion)
    one cross weight per pair      x_h <- 2 gbar x_h
    newborn pair of layer l        w+ = 2 tau1 M tau1',  w- = 2 tau2 M tau2',  x = 2 tau1 M tau2',   tau1 = (-t-, 1) / gap,  tau2 = (t+, -1) / gap
    matrix of a layer              B = sum_r w_r pp_r pp_r' + sum_h x_h (pp+ pp-' + pp- pp+')   on the layer's pushed columns.
Dropped: the gain term (2 kappa C_next - kappa g g' o C: cancels to leading order), sources at units that are not birth hubs (gate near 0 or 1: their
coefficients vanish), the complete law's K21 x covariance products. The part of the true matrix that is not on the pairs is the isotropic rest
a_rest C, a_rest = ca (radial / 3) tr C - (the fitted multiple of C in B); it enters the diagonal (variance tilt, law's diagonal) and, with `arest`, its
diagonal sandwich gbar a_rest (gamma v - VarG) rides on the hub's e e' form.
Use here (research form): the K31 extra plane k31 (3/n) R_ij / (sigma_i^3 sigma_j), R = B with its forms removed by least squares (physicist's `specific`),
through remainder31. Native form: the fold inside the bank read (physicist's ncar_fold.py) with the left factor
    pp_ir^2 lam_r  ->  pp_ir^2 lam_r + f_i (w_r pp_ir + x_r pp_i,partner(r)),   elementwise work only.

install(cd, torch, st, k31=1.0, lo=1, hi=14, forms=4, births=1.0, rows=1.0, diag=1.0, conv=1.0, ca=0.75, arest=True, k21=True, start=0, dump=None, level='r31', dfit=0, dscale=1.0)
  level: 'r31' the plane through remainder31 (reference; a native card cannot afford the separate read); 'read' the fold inside the bank read (native form).
  dfit: 1 = the form coefficients from a regression of the DIAGONAL of B on (tr C v, mu^2, 2 mu (C mu)) (native), times dscale for the multiple of C; 0 = exact least squares.
  st: the harness side table: st['pp'] pushed columns, st['lam_in'] their weights, st['births'][l] = dict(tp, tm, sel) of the layer's selected hubs.
  births, rows, diag, conv: strengths of the four sources (ablation dials);  start: first layer whose fresh sources are taken.
cd.ncar_info[l] = dict(a_B_over_trC, a_true_over_trC, plane_rms, cross_share)."""
import math, os, sys
from ncar_bank_frozen_s2s import specific

# off-diagonal multiple of C over the diagonal regression's coefficient, per layer: mean of networks 651 and 35 (e45_651/rank0651_e50.err, rank0035_e50.err; the two differ by <= 0.04)
RHO1 = {1: 1.040, 2: 0.929, 3: 0.936, 4: 0.874, 5: 0.821, 6: 0.775, 7: 0.739, 8: 0.712, 9: 0.671, 10: 0.643, 11: 0.613, 12: 0.596, 13: 0.579, 14: 0.555}


def install(cd, torch, st, k31=1.0, lo=1, hi=14, forms=4, births=1.0, rows=1.0, diag=1.0, conv=1.0, ca=0.75, arest=True, k21=True, start=0, dump=None, level='r31', dfit=0, dscale=1.0, gain=0, k22=0.0, pmproj=0, pk=4, k4=0.0, k4last=0.0, k4mo=0):
    k22s = k22; k4s = k4
    t = torch; f64 = t.float64; n = 1024; S = dict(w=None, x=None, partner=None, diagB=None, aB=0.0, mu=None, C=None, l=None, a_iso=0.0, Bm=None); info = {}; cd.ncar_info = info
    old_pin = cd.hooks.get('pair_in')

    def build(pp32, C, mu):          # the matrix of the layer on its pushed columns, its network-specific part (forms removed) and its multiple of C
        pp = pp32.detach().to(f64); ncol = pp.shape[1]; w, x, partner = S['w'][:ncol], S['x'][:ncol], S['partner'][:ncol]
        B = (pp * w[None, :] + pp[:, partner] * x[None, :]) @ pp.T; B = (B + B.T) / 2
        S['Bm'] = B
        if dfit and forms:           # form coefficients from the DIAGONAL of B alone (the native way, physicist's ncar_fold dfit): no product over pairs
            v = t.diagonal(C); Cmu = C @ mu; trC = t.trace(C)
            Zs = None
            if dfit in (7, 8):      # PHYSICIST: a stand-in for the C^2 form without an n^3 product: Z Z' with Z = C Q, Q = pk Krylov vectors of the mean direction (dfit 7: pk products of C with a
                if dfit == 7:       # vector) or the top-pk covariance eigenvectors (dfit 8); the rest of C^2 is close to a multiple of C and goes into the fitted multiple
                    ks = [mu]
                    for _ in range(int(pk) - 1): ks.append(C @ ks[-1])
                    Q_ = t.linalg.qr(t.stack(ks, 1))[0]
                else: Q_ = t.linalg.eigh(C)[1][:, -int(pk):]
                Zs = C @ Q_
            cols_ = [trC * v] + ([] if dfit == 5 else [mu * mu, 2 * mu * Cmu]) + ([t.ones(n, dtype=f64)] if dfit in (5, 6, 7, 8, 9) else []) + ([(Zs * Zs).sum(1)] if Zs is not None else [])
            if dfit == 9: cols_ = cols_ + [(C * C).sum(1)]      # PHYSICIST dfit 9: the C^2 form's coefficient from the diagonal (regressor (C^2)_ii = row sums of C o C, the lead's 14:14Z finding)
            co = t.linalg.lstsq(t.stack(cols_, 1), t.diagonal(B)[:, None]).solution[:, 0]      # PHYSICIST: dfit 5 / 6 / 7 / 8 add a CONSTANT regressor
            S['dres'], S['dres_l'] = t.diagonal(B) - t.stack(cols_, 1) @ co, S.get('l_dbg')      # PHYSICIST: the diagonal minus its fitted part = the per-unit part (K22 companion)
            aB = dscale * (RHO1[S['l_dbg']] if dfit == 2 else 1.0) * co[0] * trC; R = B - aB * C          # dfit 2: the off-diagonal multiple of C = a per-layer literal x the diagonal's
            if forms >= 2 and dfit != 5: R = R - co[1] * t.outer(mu, mu)
            if forms >= 3 and dfit != 5: R = R - co[2] * (t.outer(mu, Cmu) + t.outer(Cmu, mu))
            if dfit == 9:      # what is removed for the C^2 form: pk 0 the exact C @ C (reference); pk 1 only its part through the two units themselves, (v_i + v_j) C_ij (elementwise, free);
                Co_ = C - t.diag(v)      # pk 2: + the top mode of Co by two power steps from C mu; pk 10 + K: + the top-K exact eigenpairs of Co^2
                if int(pk) == 0: F2 = C @ C
                else:
                    F2 = (v[:, None] + v[None, :]) * C
                    if int(pk) == 2:
                        q_ = Cmu / Cmu.norm()
                        for _ in range(2): q_ = Co_ @ q_; q_ = q_ / q_.norm()
                        lam_ = float(q_ @ (Co_ @ q_)); F2 = F2 + lam_ * lam_ * t.outer(q_, q_)
                    elif int(pk) > 10:
                        ev_, U_ = t.linalg.eigh(Co_); o_ = ev_.abs().argsort(descending=True)[:int(pk) - 10]; F2 = F2 + (U_[:, o_] * (ev_[o_] ** 2)[None, :]) @ U_[:, o_].T
                R = R - co[4] * F2
            if Zs is not None:
                R = R - co[4] * (Zs @ Zs.T)
                if S.get('l_dbg') in (5, 9, 13):      # the same five coefficients by least squares on the off-diagonal (what an exact removal of these forms would use)
                    off_ = ~t.eye(n, dtype=t.bool); ZZ = Zs @ Zs.T; Xo = t.stack([(trC * C)[off_], t.outer(mu, mu)[off_], (t.outer(mu, Cmu) + t.outer(Cmu, mu))[off_], ZZ[off_]], 1)
                    ce = t.linalg.lstsq(Xo, B[off_][:, None]).solution[:, 0]; info.setdefault('zcheck', {})[S['l_dbg']] = dict(diag=[float(co[0]), float(co[1]), float(co[2]), float(co[4])], offdiag=[float(x) for x in ce])
        elif forms: R, aB = specific(t, B, C, mu, forms)
        else: R, aB = B, 0.0
        if os.environ.get('NCAR6_INFO'):      # diagnostic: the multiple of C by the exact off-diagonal fits (one form, four forms) against the diagonal regression
            v = t.diagonal(C); Cmu = C @ mu; trC = t.trace(C); off = ~t.eye(n, dtype=t.bool)
            cd_ = t.linalg.lstsq(t.stack([trC * v, mu * mu, 2 * mu * Cmu], 1), t.diagonal(B)[:, None]).solution[:, 0]
            a1 = float((B[off] * C[off]).sum() / (C[off] * C[off]).sum()); a4 = float(specific(t, B, C, mu, 4)[1])
            print('NCAR6_INFO layer %2d  a_off1/trC %+.5f  a_off4/trC %+.5f  a_diag/trC %+.5f  off1/diag %.3f  off4/diag %.3f  |R|/|B| %.3f  radial/3 %.5f' % (
                S.get('l_dbg', -1), a1 / float(trC), a4 / float(trC), float(cd_[0]), a1 / float(trC) / float(cd_[0]), a4 / float(trC) / float(cd_[0]),
                float((R - t.diag(t.diagonal(R))).norm() / (B - t.diag(t.diagonal(B))).norm()), S.get('rad_dbg', float('nan')) / 3), file=sys.stderr, flush=True)
        if pmproj:      # PHYSICIST: everything with a leg on the mean direction (pmproj 1) or on span{mu, C mu, ...} (pmproj K) is taken out, R <- (I - Q Q') R (I - Q Q'): the ONE-SIDED form,
            R0 = R - t.diag(t.diagonal(R)); ks = [mu]      # which is where the missing gain term's content sits (80% of it at layers 5-14, t21_gain_spectrum); natively: B applied to K vectors
            for _ in range(int(pmproj) - 1): ks.append(C @ ks[-1])
            Qm = t.linalg.qr(t.stack(ks, 1))[0]; RQ = R0 @ Qm; R = R0 - Qm @ RQ.T - RQ @ Qm.T + Qm @ (Qm.T @ RQ) @ Qm.T
        return B, R - t.diag(t.diagonal(R)), float(aB)

    if level == 'read':              # the fold inside the bank read (what a native card does): K21_eff[i, j] = K21[i, j] - (mu_i / (n v_i)) R_ij on the response rows
        old_mu, old_c = cd.hooks.get('mu'), cd.hooks.get('cpre')

        def mu_hook(l, mu, ctx):
            if old_mu is not None: mu = old_mu(l, mu, ctx)
            S['mu'], S['l'] = mu.detach().to(f64), l; S['l_dbg'] = l; S['diagB'] = t.zeros(n, dtype=f64); S['aB'] = 0.0; return mu

        def c_hook(l, C, ctx):
            if old_c is not None: C = old_c(l, C, ctx)
            S['C'] = C.detach().to(f64); return C
        cd.hooks['mu'], cd.hooks['cpre'] = mu_hook, c_hook
        read0 = cd.shared['read']

        def read(tt, pp, lam, response_indices, indices, U, F, old_columns):
            plane = read0(tt, pp, lam, response_indices, indices, U, F, old_columns)
            if S['w'] is not None and S['l'] is not None and S['l'] <= 14 and pp.shape[1]:
                B, R, aB = build(pp, S['C'], S['mu']); S['diagB'] = t.diagonal(B).clone(); S['aB'] = aB
                info[S['l']] = dict(a_B_over_trC=aB / float(t.trace(S['C'])), plane_rms=float((R * R).mean().sqrt()))
                if k31 and lo <= S['l'] <= hi:
                    Fd = -(S['mu'] / (n * t.diagonal(S['C']).clamp_min(1e-12)))[:, None] * R
                    plane = plane + k31 * Fd[response_indices][:, indices].to(plane.dtype)
            return plane
        cd.shared['read'] = read

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if l > 14: return pin
        mu, Cpre, k21_, k4v, k22, rem = pin
        if k22s and S.get('dres') is not None and S.get('dres_l') == l and lo <= l <= hi:      # PHYSICIST: K22 companion, K22_ij += (r_i + r_j) / n, r = the per-unit part on the response rows
            dr = S['dres'] * ctx['response_mask'].detach().to(f64); k22 = k22 + (k22s * (dr[:, None] + dr[None, :]) / n).to(k22.dtype); pin = (mu, Cpre, k21_, k4v, k22, rem)
        if k4s and S.get('dres') is not None and S.get('dres_l') == l and lo <= l <= hi:      # PHYSICIST: per-unit kurtosis companion, k4_i += k4 (6 / n) r_i (second chaos of the unit's fourth cumulant)
            dr = S['dres'] * ctx['response_mask'].detach().to(f64)
            if k4mo:      # PHYSICIST k4mo=1: for the MEAN ONLY: registered in cd.mk4[layer]; the harness adds the first-order kurtosis terms to the per-unit mean and post variance (e16b: MEAN-ONLY KURTOSIS)
                cd.mk4[l] = cd.mk4.get(l, 0) + (k4s * 6.0 / n) * dr
            else:
                k4v = k4v + (k4s * 6.0 / n) * dr.to(k4v.dtype) if t.is_tensor(k4v) and k4v.ndim else k4v + (k4s * 6.0 / n) * dr
            pin = (mu, Cpre, k21_, k4v, pin[4], rem)
        if level == 'read': return pin          # the fold was made inside the read of this layer
        S['l_dbg'] = l; S['rad_dbg'] = float(ctx['radial'])
        S['diagB'] = t.zeros(n, dtype=f64); S['aB'] = 0.0
        if S['w'] is not None and st.get('pp') is not None and st['pp'].shape[1]:
            C = Cpre.detach().to(f64); B, R, aB = build(st['pp'], C, mu.detach().to(f64))
            S['diagB'] = t.diagonal(B).clone(); S['aB'] = aB
            info[l] = dict(a_B_over_trC=aB / float(t.trace(C)), a_true_over_trC=ca * float(ctx['radial']) / 3, plane_rms=float((R * R).mean().sqrt()))
            if dump is not None and lo <= l <= hi:          # AUDIT: recorded whatever the strength
                m64 = mu.detach().to(f64); v_ = t.diagonal(C); Cmu_ = C @ m64; trC_ = t.trace(C); off_ = ~t.eye(n, dtype=t.bool); dB = t.diagonal(B); one_ = t.ones(n, dtype=f64)
                reg = lambda cols: float(t.linalg.lstsq(t.stack(cols, 1), dB[:, None]).solution[0, 0])
                dump[l] = dict(R=R.clone(), B=B.clone(), coef=dict(off1=float((B[off_] * C[off_]).sum() / (C[off_] * C[off_]).sum() / trC_), off4=float(specific(t, B, C, m64, 4)[1] / trC_),
                               diag3=reg([trC_ * v_, m64 * m64, 2 * m64 * Cmu_]), diag3c=reg([trC_ * v_, m64 * m64, 2 * m64 * Cmu_, one_]), diag1=reg([trC_ * v_]), diag1c=reg([trC_ * v_, one_]),
                               radial_over_3=float(ctx['radial']) / 3, a_rest_over_trC=ca * float(ctx['radial']) / 3 - aB / float(trC_)))
            if k31 and lo <= l <= hi:
                s = t.diagonal(C).clamp_min(1e-10).sqrt(); rm = k31 * (3.0 / n) * R / (s ** 3)[:, None] / s[None, :]
                rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook

    old_out = cd.hooks.get('state_out')

    def out_hook(l, out, ctx):
        if old_out is not None: out = old_out(l, out, ctx)
        if not 0 <= l <= 14: return out
        C = ctx['Cpre'].detach().to(f64); mu = ctx['mu'].detach().to(f64); m = out[0].detach().to(f64)
        v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); al = mu / s; ph = t.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = t.erfc(-al / math.sqrt(2)) / 2; d = ph / s
        r2 = v * ((1 + al * al) * g + al * ph); r3 = v * s * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
        gam = g - m * d; V = 2 * m * (1 - g); S2 = 2 * (gam - g * g); S3 = 3 * (1 - g) * (r2 - 2 * m * m)
        S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2; e3 = -al * ph / v; varG = r2 - (s * ph + mu * g) ** 2
        gb = float(gam.mean()); Co = C - t.diag(v); Co2 = Co * Co; g2 = g * g
        ncol = 0 if st.get('pp') is None else int(st['pp'].shape[1]); total = int(out[3].numel()); K3Vd = t.zeros(n, dtype=f64)
        if ncol:
            pp = st['pp'].detach().to(f64); c = conv * st['lam_in'].detach().to(f64)[:ncol] * (V @ pp); K3Vd = (pp * pp) @ c
            w_old = S['w'][:ncol] if S['w'] is not None else t.zeros(ncol, dtype=f64); x_old = S['x'][:ncol] if S['x'] is not None else t.zeros(ncol, dtype=f64)
            w_car, x_car = 2 * (gb * w_old + c), 2 * gb * x_old
            partner = S['partner'][:ncol] if S['partner'] is not None else t.arange(ncol)
        else:
            w_car = x_car = t.zeros(0, dtype=f64); partner = t.zeros(0, dtype=t.long)
        # the layer's own quantities of the law
        diagB = S['diagB'] if S['diagB'] is not None else t.zeros(n, dtype=f64)
        a_rest = ca * float(ctx['radial']) / 3 * float(t.trace(C)) - S['aB']                         # the isotropic part of the true matrix that is not on the pairs
        if gain: a_rest = S['a_iso']                 # PHYSICIST: with the gain term the multiple of C that is not on the pairs is known: Nt = B + 2 kappa(l-1) C
        Ntd = diagB + a_rest * v
        Pv = gb * Ntd + 2 * (Co2 @ gam)
        rv = d * (Co @ V) + (e3 / 2) * Pv + S3
        dgn = S4 + V * (Co @ V) + 2 * gam * (Co2 @ gam) - 2 * g2 * (Co2 @ g2) + gam * gb * Ntd
        if k21 and l >= 1:
            Kf = ctx['k21'].detach().to(f64); Ko = Kf - t.diag(t.diagonal(Kf)); KoV = Ko @ V; KoTg = Ko.T @ gam; KC = Ko * Co
            rv = rv + d * KoTg + (e3 / 2) * KoV
            dgn = dgn + V * KoTg + gam * KoV - 2 * g * d * (KC @ g2) - 2 * g2 * (KC.T @ (g * d))
        D = dgn - g2 * (births * (Co2 @ S2) + gb * diagB + K3Vd)                                      # the law's diagonal minus what the pair forms put on the diagonal themselves
        if gain:      # PHYSICIST: the homogeneity (gain) term of the law on the pairs. Tilt P = gbar Nt + 2 C Gamma C + K3V; its pure-gain part kappa C acts EXACTLY as kappa x post covariance,
            # i.e. 2 kappa C_next at the next layer (an explicit multiple of C, kept as a_iso) and, in post space, - kappa (g g' o Co + diag VarG): the first is the row form
            # - (kappa / 2) g_h (e_h c_h' + c_h e_h'), the second goes to D. The carried a_iso C itself transports as gbar a_iso g g' o Co, also a row form: net row coefficient - (kappa - gbar a_iso) g / 2.
            CC = float((C * C).sum())
            if gain == 1:      # kappa by the Frobenius projection on C, as in nlaw.law
                K3m = (pp * c[None, :]) @ pp.T if ncol else t.zeros((n, n), dtype=f64)
                PC = gb * (float((S['Bm'] * C).sum()) if (S['Bm'] is not None and ncol) else 0.0) + gb * a_rest * CC + 2 * float((((C * gam[None, :]) @ C) * C).sum()) + float((K3m * C).sum())
                kap = PC / CC
            else:              # kappa from the diagonals only (what a native card has): <diag P, v> / <v, v>
                kap = float(((gb * Ntd + 2 * ((C * C) @ gam) + K3Vd) * v).sum() / (v * v).sum())
            rv = rv - (kap - gb * a_rest) * g / 2; D = D - kap * varG; S['a_iso'] = 2 * kap; info.setdefault('kap', {})[l] = kap / float(t.trace(C))
        elif arest: D = D - gb * a_rest * varG                                                          # isotropic rest: 2 gbar a_rest W'[G Co G + diag(gamma v)]W = multiple of C_next + this sandwich
        new = total - ncol; K = new // 2
        if new and l >= start and 'births' in st and l in st['births'] and 'sel' in st['births'][l]:
            Bh = st['births'][l]; sel = Bh['sel']; tp, tm = Bh['tp'], Bh['tm']; assert K == int(sel.numel()) == int(tp.numel()), (l, K, sel.numel())
            gap = tp - tm; a11, a12, a22 = births * S2[sel], rows * rv[sel], diag * D[sel]
            wp = 2 * (a11 * tm * tm - 2 * a12 * tm + a22) / (gap * gap); wm = 2 * (a11 * tp * tp - 2 * a12 * tp + a22) / (gap * gap)
            xx = 2 * (-a11 * tp * tm + a12 * (tp + tm) - a22) / (gap * gap)
            w_new, x_new = t.cat((wp, wm)), t.cat((xx, xx))
        else:
            w_new = x_new = t.zeros(new, dtype=f64)
        S['w'] = t.cat((w_car, w_new)); S['x'] = t.cat((x_car, x_new))
        S['partner'] = t.cat((partner, t.arange(ncol + K, ncol + 2 * K), t.arange(ncol, ncol + K), t.arange(ncol + 2 * K, total)))
        return out
    cd.hooks['state_out'] = out_hook
    if k4last:      # PHYSICIST: the same per-unit term at the FINAL layer (no read there: the diagonal of the pair matrix from the layer-15 pushed columns, regression on (tr C v_i, 1))
        old15 = cd.hooks.get('k4_15')
        def k4_15(l, v4, ctx):
            if old15 is not None: v4 = old15(l, v4, ctx)
            if S['w'] is None or st.get('pp') is None or not st['pp'].shape[1]: return v4
            pp = st['pp'].detach().to(f64); ncol = pp.shape[1]; w, x, partner = S['w'][:ncol], S['x'][:ncol], S['partner'][:ncol]
            dB = (pp * pp) @ w + (pp * pp[:, partner]) @ x; vv = ctx['vv'].detach().to(f64); X = t.stack([vv.sum() * vv, t.ones(n, dtype=f64)], 1)
            return v4 + (k4last * 6.0 / n) * (dB - X @ t.linalg.lstsq(X, dB[:, None]).solution[:, 0])
        cd.hooks['k4_15'] = k4_15
    return cd

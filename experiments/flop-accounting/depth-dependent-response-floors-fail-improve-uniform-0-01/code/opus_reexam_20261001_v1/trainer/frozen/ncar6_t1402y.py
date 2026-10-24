"""TRAINER'S COPY (O/trainer/frozen/ncar6_t1402y.py) of the lead's O/ncar6.py as of 14:02Z (sha 1897e1d4) with FOUR ADDED LINES, all marked TRAINER
(the two of ncar6_t1402x.py, plus: the two parts of the matrix's diagonal, carried columns and pair cross terms, kept in S['dparts'], and the layer's
per-unit sources rv and d o Pv kept in S['rv_prev'], S['dPv_prev'] for the next layer's head inputs):
the module's state dict is exposed as cd.ncar6_state, and the diagonal regression's fitted diagonal is kept in it (S['fit']). Nothing else differs and no
number the module computes changes (certified against the unchanged copy in the pool). They feed the head-input features of otr_m.py. Original header:
TAG opus-reexam NCAR6 (lead, 2026-10-02): THE WHOLE LEAN LAW ON THE BANK'S OWN COLUMN PAIRS: no new columns, no eigenvectors, no probes, no rank cut.

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
  c2sw = s > 0 (with dfit 4 and c2k = K): the C^2 term from a warm-started subspace iteration with s sweeps instead of exact eigenpairs (the native form).
  dfit 3 / 4: diagonal regression with a constant regressor (no table) / also with (C^2)_ii and b C^2 removed (c2k = K: C^2 from its top-K eigenpairs; 0 = exact).
  pmu = K: after the forms, the Rayleigh quotients of what is left on K orthonormalised Krylov vectors mu, C mu, ... are removed (no eigenvectors).
  pdiag = K: after the forms, only the Rayleigh quotients of what is left on the top-K covariance eigenvectors are removed (natively: K eigenvectors, ONE pass over the columns per vector).
  ptop = K: after the forms, the two-sided block P R P on the top-K eigenvectors of the covariance is removed (natively: K eigenvectors and two passes over the columns per vector).
  cenb, cenr: source centering of the births (S2 minus cenb x its weighted mean: no C^2-like part is made) and of the row forms (rv minus cenr x rho g: no multiple of C is made).
  level: 'r31' the plane through remainder31 (reference; a native card cannot afford the separate read); 'read' the fold inside the bank read (native form).
  dfit: 1 = the form coefficients from a regression of the DIAGONAL of B on (tr C v, mu^2, 2 mu (C mu)) (native), times dscale for the multiple of C; 0 = exact least squares.
  st: the harness side table: st['pp'] pushed columns, st['lam_in'] their weights, st['births'][l] = dict(tp, tm, sel) of the layer's selected hubs.
  births, rows, diag, conv: strengths of the four sources (ablation dials);  start: first layer whose fresh sources are taken.
cd.ncar_info[l] = dict(a_B_over_trC, a_true_over_trC, plane_rms, cross_share)."""
import math, os, sys
from ncar_bank_frozen_s2s import specific

# off-diagonal multiple of C over the diagonal regression's coefficient, per layer: mean of networks 651 and 35 (e45_651/rank0651_e50.err, rank0035_e50.err; the two differ by <= 0.04)
RHO1 = {1: 1.040, 2: 0.929, 3: 0.936, 4: 0.874, 5: 0.821, 6: 0.775, 7: 0.739, 8: 0.712, 9: 0.671, 10: 0.643, 11: 0.613, 12: 0.596, 13: 0.579, 14: 0.555}


def install(cd, torch, st, k31=1.0, lo=1, hi=14, forms=4, births=1.0, rows=1.0, diag=1.0, conv=1.0, ca=0.75, arest=True, k21=True, start=0, dump=None, level='r31', dfit=0, dscale=1.0, cenb=0.0, cenr=0.0, ptop=0, pdiag=0, pmu=0, c2k=0, c2sw=0):
    t = torch; f64 = t.float64; n = 1024; S = dict(w=None, x=None, partner=None, diagB=None, aB=0.0, mu=None, C=None, l=None); info = {}; cd.ncar_info = info
    cd.ncar6_state = S          # TRAINER: the per-layer state, read by the head-input features
    old_pin = cd.hooks.get('pair_in')

    def build(pp32, C, mu):          # the matrix of the layer on its pushed columns, its network-specific part (forms removed) and its multiple of C
        pp = pp32.detach().to(f64); ncol = pp.shape[1]; w, x, partner = S['w'][:ncol], S['x'][:ncol], S['partner'][:ncol]
        B = (pp * w[None, :] + pp[:, partner] * x[None, :]) @ pp.T; B = (B + B.T) / 2
        S['dparts'], S['dparts_l'] = ((pp * pp) @ w, (pp * pp[:, partner]) @ x), S.get('l_dbg')      # TRAINER: carried-column part and pair cross part of the diagonal
        if dfit >= 3 and forms:      # dfit 3: diagonal regression WITH A CONSTANT (physicist 13:46Z: the coefficient of tr C v_i is then the off-diagonal multiple of C itself;
            v = t.diagonal(C); Cmu = C @ mu; trC = t.trace(C)             # no table); dfit 4: the same with the regressor (C^2)_ii, and b C^2 removed too (c2k > 0: C^2 from its top-c2k eigenpairs)
            cols = [trC * v, t.ones_like(v), mu * mu, 2 * mu * Cmu] + ([(C * C).sum(1)] if dfit == 4 else [])
            co = t.linalg.lstsq(t.stack(cols, 1), t.diagonal(B)[:, None]).solution[:, 0]
            S['fit'], S['fit_l'] = t.stack(cols, 1) @ co, S.get('l_dbg')          # TRAINER: the regression's fitted diagonal of this layer
            aB = dscale * co[0] * trC; R = B - aB * C
            if forms >= 2: R = R - co[2] * t.outer(mu, mu)
            if forms >= 3: R = R - co[3] * (t.outer(mu, Cmu) + t.outer(Cmu, mu))
            if dfit == 4:
                if c2k and c2sw:     # NATIVE FORM of the C^2 term: its compression P C^2 P on a K-dimensional subspace X0 found by subspace iteration: the previous layer's
                    Xp = S.get('X')  # block pushed through the gates and W (K products with W), then c2sw sweeps with C (K products each); X1 = C X0 gives X0 (X1' X1) X0' for nothing
                    if Xp is None or S.get('Xg') is None:
                        idx = t.diagonal(C).argsort(descending=True)[:int(c2k)]; X0 = t.linalg.qr(C[:, idx])[0]            # first layer: the K columns of C with the largest variance
                    else:
                        X0 = t.linalg.qr(cd.ws[S['l_dbg']].to(f64).T @ (S['Xg'][:, None] * Xp))[0]
                    X1 = C @ X0
                    for _ in range(int(c2sw) - 1): X0 = t.linalg.qr(X1)[0]; X1 = C @ X0
                    S['X_new'] = X0; C2 = X0 @ (X1.T @ X1) @ X0.T
                elif c2k:
                    lamC, VC = t.linalg.eigh(C); Uk = VC[:, -int(c2k):]; C2 = (Uk * (lamC[-int(c2k):] ** 2)[None, :]) @ Uk.T
                else: C2 = C @ C
                R = R - co[4] * C2
        elif dfit and forms:         # form coefficients from the DIAGONAL of B alone (the native way, physicist's ncar_fold dfit): no product over pairs
            v = t.diagonal(C); Cmu = C @ mu; trC = t.trace(C)
            co = t.linalg.lstsq(t.stack([trC * v, mu * mu, 2 * mu * Cmu], 1), t.diagonal(B)[:, None]).solution[:, 0]
            aB = dscale * (RHO1[S['l_dbg']] if dfit == 2 else 1.0) * co[0] * trC; R = B - aB * C          # dfit 2: the off-diagonal multiple of C = a per-layer literal x the diagonal's
            if forms >= 2: R = R - co[1] * t.outer(mu, mu)
            if forms >= 3: R = R - co[2] * (t.outer(mu, Cmu) + t.outer(Cmu, mu))
        elif forms: R, aB = specific(t, B, C, mu, forms)
        else: R, aB = B, 0.0
        if ptop:                     # the two-sided block of what is left on the top-K covariance eigenvectors taken out (where the C^2-like content sits: C^2 is nearly lambda_1^2 u_1 u_1')
            lamC, VC = t.linalg.eigh(C); Ut = VC[:, -int(ptop):]; R = R - Ut @ (Ut.T @ R @ Ut) @ Ut.T
        if pmu:                      # the cheap top-mode repair: Rayleigh quotients of what is left on the mean direction (pmu 1) and on the next Krylov vectors C mu, C^2 mu ...
            vs = []; x_ = mu.clone()                                                                     # (pmu 2, 3 ...), orthonormalised: no eigenvector needed, one pass over the columns each
            for _ in range(int(pmu)):
                for y_ in vs: x_ = x_ - (y_ @ x_) * y_
                x_ = x_ / x_.norm().clamp_min(1e-300); vs.append(x_); x_ = C @ x_
            Vm = t.stack(vs, 1); qd = ((Vm.T @ R) * Vm.T).sum(1); R = R - (Vm * qd[None, :]) @ Vm.T
        if pdiag:                    # only the DIAGONAL of what is left in the eigenbasis of the covariance, on the top-K modes: R -= sum_m (u_m' R u_m) u_m u_m' (the spectral
            lamC, VC = t.linalg.eigh(C); Ut = VC[:, -int(pdiag):]; qd = ((Ut.T @ R) * Ut.T).sum(1); R = R - (Ut * qd[None, :]) @ Ut.T      # function of C; cross terms between modes stay)
        if os.environ.get('NCAR6_SAVEB') and S.get('l_dbg') in (5, 9, 13):      # diagnostic: the matrix, the covariance and the mean of the layer to a file
            import numpy as _np
            _np.savez(os.environ['NCAR6_SAVEB'] + '_L%d.npz' % S['l_dbg'], B=B.numpy().astype('float32'), C=C.numpy(), mu=mu.numpy())
        if os.environ.get('NCAR6_INFO'):      # diagnostic: the multiple of C by the exact off-diagonal fits (one form, four forms) against the diagonal regression
            v = t.diagonal(C); Cmu = C @ mu; trC = t.trace(C); off = ~t.eye(n, dtype=t.bool)
            cd_ = t.linalg.lstsq(t.stack([trC * v, mu * mu, 2 * mu * Cmu], 1), t.diagonal(B)[:, None]).solution[:, 0]
            a1 = float((B[off] * C[off]).sum() / (C[off] * C[off]).sum()); a4 = float(specific(t, B, C, mu, 4)[1])
            print('NCAR6_INFO layer %2d  a_off1/trC %+.5f  a_off4/trC %+.5f  a_diag/trC %+.5f  off1/diag %.3f  off4/diag %.3f  |R|/|B| %.3f  radial/3 %.5f' % (
                S.get('l_dbg', -1), a1 / float(trC), a4 / float(trC), float(cd_[0]), a1 / float(trC) / float(cd_[0]), a4 / float(trC) / float(cd_[0]),
                float((R - t.diag(t.diagonal(R))).norm() / (B - t.diag(t.diagonal(B))).norm()), S.get('rad_dbg', float('nan')) / 3), file=sys.stderr, flush=True)
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
        if level == 'read': return pin          # the fold was made inside the read of this layer
        S['l_dbg'] = l; S['rad_dbg'] = float(ctx['radial'])
        S['diagB'] = t.zeros(n, dtype=f64); S['aB'] = 0.0
        if S['w'] is not None and st.get('pp') is not None and st['pp'].shape[1]:
            C = Cpre.detach().to(f64); B, R, aB = build(st['pp'], C, mu.detach().to(f64))
            S['diagB'] = t.diagonal(B).clone(); S['aB'] = aB
            info[l] = dict(a_B_over_trC=aB / float(t.trace(C)), a_true_over_trC=ca * float(ctx['radial']) / 3, plane_rms=float((R * R).mean().sqrt()))
            if k31 and lo <= l <= hi:
                if dump is not None: dump[l] = dict(R=R.to(t.float32).clone())
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
        Ntd = diagB + a_rest * v
        Pv = gb * Ntd + 2 * (Co2 @ gam)
        rv = d * (Co @ V) + (e3 / 2) * Pv + S3
        dgn = S4 + V * (Co @ V) + 2 * gam * (Co2 @ gam) - 2 * g2 * (Co2 @ g2) + gam * gb * Ntd
        if k21 and l >= 1:
            Kf = ctx['k21'].detach().to(f64); Ko = Kf - t.diag(t.diagonal(Kf)); KoV = Ko @ V; KoTg = Ko.T @ gam; KC = Ko * Co
            rv = rv + d * KoTg + (e3 / 2) * KoV
            dgn = dgn + V * KoTg + gam * KoV - 2 * g * d * (KC @ g2) - 2 * g2 * (KC.T @ (g * d))
        # SOURCE CENTERING (removes forms where they are made, for nothing): the births sum_h S2_h c_h c_h' with S2 -> S2 - cenb x (its weighted mean) lose their C^2-like
        # part s G Co^2 G; the row forms with rv -> rv - cenr x rho g lose their multiple of the covariance 2 rho G Co G (rho = least squares of rv on g; weights
        # omega_h = |c_h|^2 = (Co2 g^2)_h in both)
        om = Co2 @ g2
        if cenb: S2 = S2 - cenb * float((S2 * om).sum() / om.sum().clamp_min(1e-300))
        if cenr: rv = rv - cenr * float((rv * g * om).sum() / (g2 * om).sum().clamp_min(1e-300)) * g
        D = dgn - g2 * (births * (Co2 @ S2) + gb * diagB + K3Vd)                                      # the law's diagonal minus what the pair forms put on the diagonal themselves
        if arest: D = D - gb * a_rest * varG                                                          # isotropic rest: 2 gbar a_rest W'[G Co G + diag(gamma v)]W = multiple of C_next + this sandwich
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
        S['rv_prev'], S['dPv_prev'], S['src_l'] = rv, d * Pv, l                                               # TRAINER: the layer's per-unit sources (row factor, variance tilt)
        if S.get('X_new') is not None: S['X'], S['Xg'] = S.pop('X_new'), g                                    # the C^2 subspace of this layer and its gates, for the next layer's warm start
        S['partner'] = t.cat((partner, t.arange(ncol + K, ncol + 2 * K), t.arange(ncol, ncol + K), t.arange(ncol + 2 * K, total)))
        return out
    cd.hooks['state_out'] = out_hook
    return cd

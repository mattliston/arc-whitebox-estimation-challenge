"""TAG opus-physicist K31L_AUDIT: my COPY of the lead's O/k31l.py (sha 2ccd70d0e195, 2026-10-02 15:25Z) with audit options added: dump (dict: the raw plane of every layer),
agemin (cohorts of age agemin..age), rowrm (1: the part of the plane proportional row by row to rho is taken out before use, so that the pair map's own isotropic K31 term
keeps the row coefficients; 2: only that row part is used). Everything else is the lead's text.

TAG opus-reexam K31L (lead, 2026-10-02): the HUB-LOCAL fourth cumulant of the young bank pairs as a (3,1) PLANE (the plane companion of K4L).

K4L uses the young pairs' local quartic form, per hub h on its two pushed vectors Y (the pushed coupling vector s_h) and E (the pushed unit vector e_h),
    Q_h = S1 Sym(E Y Y Y) x 4 + S2 Sym(E E Y Y) x 6 + S3 Sym(Y E E E) x 4 + S4 E E E E,      (S1, S2, S3, S4) = (e3, S2, S3, S4) of the hub at its birth,
only on the diagonal: k4_i += Q(i, i, i, i). The pair-carried law (ncar6.py) carries the SECOND-CHAOS projection of the same sources (and their transport)
into the K31 plane. This module puts the exact local form itself into the plane, for the cohorts of age <= `age`:
    K31L_ij = sum_h [ S1 (Y_i^3 E_j + 3 Y_i^2 E_i Y_j) + S2 (3 Y_i E_i^2 Y_j + 3 Y_i^2 E_i E_j) + S3 (E_i^3 Y_j + 3 Y_i E_i^2 E_j) + S4 E_i^3 E_j ],
through remainder31 (standardised by sigma_i^3 sigma_j), with `forms` removed by least squares (physicist's `specific`). `chaos4`: the plane's second-chaos
projection is taken out first, 3 (1/n) sum_k Q(k, k, i, j) -> only the fourth-chaos part is added (to sit on top of ncar6 without counting twice).
Native form, if it pays: more terms in the left factor of the K21 bank read (cubic in the pair's two columns), no new product.

install(cd, torch, st, k31=1.0, age=2, lo=1, hi=14, forms=4, chaos4=False, level='r31')
  level 'read': the fold inside the K21 bank read (native form; nothing removed), 'r31': through remainder31.
  means: 'all' column means of Y^2, Y E, E^2 over all units; 'resp' over the read's response rows (level 'read'); 'iid' without the cross term.
  st: the harness side table (st['pp'], st['stops'], st['births'][l] = dict(tp, tm, S, sel)), as for K4L in e16b_k4_closed_loop.py."""
from ncar_bank_frozen_s2s import specific


def install(cd, torch, st, k31=1.0, age=2, lo=1, hi=14, forms=4, chaos4=False, level='r31', means='all', dump=None, agemin=1, rowrm=0):
    t = torch; f64 = t.float64; n = 1024; info = {}; cd.k31l_info = info; S = dict(mu=None, C=None, l=None)
    old_pin = cd.hooks.get('pair_in')

    def plane_of(l, pp32, rows=None):
        pp = pp32.detach().to(f64); stops = st['stops']; P = t.zeros((n, n), dtype=f64)
        for j in range(l):
            if l - j > age or l - j < agemin or j not in st['births']: continue
            B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2
            if not K: continue
            gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
            E = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap; S1, S2, S3, S4 = (x[None, :] for x in B['S'])
            AY = 3 * S1 * Y * Y * E + 3 * S2 * Y * E * E + S3 * E ** 3
            AE = S1 * Y ** 3 + 3 * S2 * Y * Y * E + 3 * S3 * Y * E * E + S4 * E ** 3
            if chaos4:      # second-chaos projection of Q(i, i, i, j): 3 x (1/n-type mean of the squared slot) -> replace two of the three i-slots by their column mean
                if means == 'resp' and rows is not None:      # column means over the read's response rows only (what a native read has at hand)
                    Yr, Er = Y[rows], E[rows]; mY2, mYE, mE2 = (Yr * Yr).mean(0)[None, :], (Yr * Er).mean(0)[None, :], (Er * Er).mean(0)[None, :]
                elif means == 'iid':                           # the weights' own expectation: mean of E^2 = 2/n x (gated norm), from the column norms; cross term 0
                    mY2, mYE, mE2 = (Y * Y).sum(0)[None, :] / n, t.zeros((1, Y.shape[1]), dtype=f64), (E * E).sum(0)[None, :] / n
                else:
                    mY2, mYE, mE2 = (Y * Y).mean(0)[None, :], (Y * E).mean(0)[None, :], (E * E).mean(0)[None, :]
                # Q(k, k, i, j) summed over k / n, times 3: S1 (3 mY2 E_i Y_j ... ) written slot by slot
                AY = AY - 3 * (S1 * (mY2 * E + 2 * mYE * Y) + S2 * (mE2 * Y + 2 * mYE * E) + S3 * mE2 * E)
                AE = AE - 3 * (S1 * mY2 * Y + S2 * (mY2 * E + 2 * mYE * Y) + S3 * (mE2 * Y + 2 * mYE * E) + S4 * mE2 * E)
            P = P + AY @ Y.T + AE @ E.T
        return P

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if level == 'read' or not (lo <= l <= hi) or (not k31 and dump is None) or st.get('pp') is None or not st['pp'].shape[1]: return pin
        mu, Cpre, k21_, k4v, k22, rem = pin
        P = plane_of(l, st['pp'])
        if dump is not None: dump[l] = P.detach().to(t.float32).clone()
        if not k31: return pin
        C = Cpre.detach().to(f64)          # forms (symmetric) are removed from the symmetric part; the antisymmetric part of the (3,1) plane is kept as it is
        R = (specific(t, (P + P.T) / 2, C, mu.detach().to(f64), forms)[0] + (P - P.T) / 2) if forms else P
        s = t.diagonal(C).clamp_min(1e-10).sqrt(); rm = k31 * (R - t.diag(t.diagonal(R))) / (s ** 3)[:, None] / s[None, :]
        if rowrm:
            rho = C / t.outer(s, s); rho = rho - t.diag(t.diagonal(rho)); rowpart = ((rm * rho).sum(1) / (rho * rho).sum(1))[:, None] * rho; rm = (rm - rowpart) if rowrm == 1 else rowpart
        info[l] = float((rm * rm).mean().sqrt())
        rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook

    if level == 'read':     # the native form: folded into the K21 bank read, K21_eff[i, j] = K21[i, j] - (mu_i / (3 v_i)) K31L_ij on the response rows (no forms removed)
        old_mu, old_c = cd.hooks.get('mu'), cd.hooks.get('cpre')

        def mu_hook(l, mu, ctx):
            if old_mu is not None: mu = old_mu(l, mu, ctx)
            S['mu'], S['l'] = mu.detach().to(f64), l; return mu

        def c_hook(l, C, ctx):
            if old_c is not None: C = old_c(l, C, ctx)
            S['C'] = C.detach().to(f64); return C
        cd.hooks['mu'], cd.hooks['cpre'] = mu_hook, c_hook
        read0 = cd.shared['read']

        def read(tt, pp, lam, response_indices, indices, U, F, old_columns):
            plane = read0(tt, pp, lam, response_indices, indices, U, F, old_columns)
            if S['l'] is not None and lo <= S['l'] <= hi and k31 and pp.shape[1]:
                P = plane_of(S['l'], pp, response_indices)
                if dump is not None: dump[S['l']] = P.detach().to(t.float32).clone()
                if rowrm:
                    s_ = t.diagonal(S['C']).clamp_min(1e-10).sqrt(); rho = S['C'] / t.outer(s_, s_); rho = rho - t.diag(t.diagonal(rho)); T_ = (P - t.diag(t.diagonal(P))) / (s_ ** 3)[:, None] / s_[None, :]
                    rowpart = ((T_ * rho).sum(1) / (rho * rho).sum(1))[:, None] * rho; P = ((T_ - rowpart) if rowrm == 1 else rowpart) * (s_ ** 3)[:, None] * s_[None, :]
                Fd = -(S['mu'] / (3 * t.diagonal(S['C']).clamp_min(1e-12)))[:, None] * P
                plane = plane + k31 * Fd[response_indices][:, indices].to(plane.dtype)
            return plane
        cd.shared['read'] = read
    return cd

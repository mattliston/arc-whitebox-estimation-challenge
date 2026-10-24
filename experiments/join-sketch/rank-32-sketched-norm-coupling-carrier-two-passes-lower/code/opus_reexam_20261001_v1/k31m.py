"""TAG opus-reexam K31M (lead, 2026-10-02): the MIXED terms of the fresh fourth cumulant (first order in the carried third cumulant) as a (3,1) plane. RESEARCH FORM
(two products over hubs x columns a layer: not a native form; see k31diag.py for the derivation and e66_k31_truth.py for the comparison with the truth planes:
on network 35 they add 17-19 points of the true plane's energy to the law + the local quartic, with coefficients near 1).
    kappa4^x(a, b, c, d) contains d_a kappa3^z(a, b, c) g_b g_c (g_d C_ad)      (hub a: second derivative; two legs on a bank column r; one Gaussian edge)
    MX1_ij = 6 sum_r lam_r P_ri P_rj sum_a d_a pp_ra E_ai Y_ai
    MX2_ij = 3 sum_a d_a E_ai Y_aj sum_r lam_r pp_ra P_ri^2
    MX3_ij = 3 sum_a d_a Y_ai E_aj sum_r lam_r pp_ra P_ri^2
install(cd, torch, st, k31=1.0, terms=(1, 1, 1), lo=2, hi=14, forms=0)     terms: strengths of MX1, MX2, MX3;  forms: least-squares form removal on the symmetric part.
st['births'][l] must hold d (harness option k3dg=1 or k3c != 0)."""
from ncar_bank_frozen_s2s import specific


def install(cd, torch, st, k31=1.0, terms=(1.0, 1.0, 1.0), lo=2, hi=14, forms=0):
    t = torch; f64 = t.float64; n = 1024; S = dict(prev=None); info = {}; cd.k31m_info = info
    old_pin = cd.hooks.get('pair_in')

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if not (lo <= l <= hi) or not k31 or st.get('pp') is None or not st['pp'].shape[1]: return pin
        mu, Cpre, k21_, k4v, k22, rem = pin
        j = l - 1; Pv = S['prev']
        if j not in st['births'] or 'd' not in st['births'][j] or Pv is None or not Pv['ncol']: return pin
        pp = st['pp'].detach().to(f64); stops = st['stops']; B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2
        if not K or Pv['ncol'] != k0: return pin
        gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
        E = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap; d = B['d']
        Pold = pp[:, :k0]; lam = st['lam_in'].detach().to(f64)[:k0]; PPh = Pv['pp'][B['sel'], :]
        P = t.zeros((n, n), dtype=f64)
        if terms[0]:
            theta = 6 * (((E * Y) * d[None, :]) @ PPh); P = P + terms[0] * ((theta * Pold * lam[None, :]) @ Pold.T)
        if terms[1] or terms[2]:
            Xi = (Pold * Pold * lam[None, :]) @ PPh.T
            if terms[1]: P = P + terms[1] * 3 * ((Xi * E * d[None, :]) @ Y.T)
            if terms[2]: P = P + terms[2] * 3 * ((Xi * Y * d[None, :]) @ E.T)
        C = Cpre.detach().to(f64)
        R = (specific(t, (P + P.T) / 2, C, mu.detach().to(f64), forms)[0] + (P - P.T) / 2) if forms else P
        s = t.diagonal(C).clamp_min(1e-10).sqrt(); rm = k31 * (R - t.diag(t.diagonal(R))) / (s ** 3)[:, None] / s[None, :]
        info[l] = float((rm * rm).mean().sqrt())
        rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook
    old_out = cd.hooks.get('state_out')

    def out_hook(l, o_, ctx):
        if old_out is not None: o_ = old_out(l, o_, ctx)
        if l <= 14: S['prev'] = dict(pp=st['pp'].detach().to(f64).clone(), ncol=int(st['pp'].shape[1])) if st.get('pp') is not None and st['pp'].shape[1] else dict(pp=None, ncol=0)
        return o_
    cd.hooks['state_out'] = out_hook
    return cd

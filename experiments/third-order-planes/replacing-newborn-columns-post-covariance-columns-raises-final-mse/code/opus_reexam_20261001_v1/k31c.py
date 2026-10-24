"""TAG opus-reexam K31C (lead, 2026-10-02): the TWO-HUB (chain-tree) part of the fresh fourth cumulant as a (3,1) plane.

For Gaussian pre-activations the fourth cumulant of four post-activations is, at tree level, a sum over the 16 trees on the four arguments: 4 stars
(one unit with the third derivative: K4L's / K31L's S1 term on the hub's own pair) and 12 CHAINS a - b - c - d with second derivatives at the two inner units,
    kappa4(x_a, x_b, x_c, x_d) contains d_b d_c C_bc (g_a C_ab)(g_d C_cd),     d = pdf / sigma.
The chains have two hubs, so they are in no hub-local form: not in K4L, not in K31L4; the pair-carried law holds their SECOND-CHAOS projections (its row forms
and variance tilt are chains with one contracted pair). Pushed through the weights, with Y_b = pushed coupling vector and E_b = pushed unit vector of hub b
(the bank's newborn pair, as in k31l.py),
    K31C_ij = 6 sum_{b != c} d_b d_c C_bc (Y_b E_b)_i [ E_ci Y_cj + Y_ci E_cj ]
            = 6 [ ((A Cq) o E) Y' + ((A Cq) o Y) E' ]_ij,   A_ib = d_b Y_bi E_bi,   Cq = C of the birth layer on the hubs, zero diagonal, times d_c on the columns.
chaos4: its second-chaos projection is taken out (each of the three pairs of i-slots replaced by the mean over the units of the slot product; the Gram
matrices of the cohort's columns), so that it sits on top of ncar6 + k31l without counting twice.
Native cost if it pays: one product (response rows x hubs) x (hubs x hubs) a layer (the rest is the left factor of the bank read on the newborn pair).

install(cd, torch, st, k31=1.0, lo=1, hi=14, chaos4=True, forms=0, age=1, level='r31')
  st['births'][l] must hold tp, tm, sel and d (the hubs' pdf / sigma) and Ch (the layer's pre-activation covariance on the selected hubs); the harness
  e16b_k4_closed_loop.py stores them when k3c is asked for.  age 2: the cohort born two layers back too (same coefficients, its columns pushed twice)."""
from ncar_bank_frozen_s2s import specific


def install(cd, torch, st, k31=1.0, lo=1, hi=14, chaos4=True, forms=0, age=1, level='r31'):
    t = torch; f64 = t.float64; n = 1024; info = {}; cd.k31c_info = info; S = dict(mu=None, C=None, l=None)
    old_pin = cd.hooks.get('pair_in')

    def plane_of(l, pp32):
        pp = pp32.detach().to(f64); stops = st['stops']; P = t.zeros((n, n), dtype=f64)
        for j in range(l):
            if l - j > age or j not in st['births'] or 'Ch' not in st['births'][j]: continue
            B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2
            if not K: continue
            gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
            E = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap
            d = B['d']; kap = 6 * d[:, None] * B['Ch'] * d[None, :]; kap = kap - t.diag(t.diagonal(kap))            # 6 d_b d_c C_bc, b != c
            M = (Y * E) @ kap                                                                                      # sum_b (Y_b E_b)_i kappa_bc
            PY = M * E; PE = M * Y                                                                                 # left factors of Y_cj and of E_cj
            if chaos4:
                GYE = Y.T @ E / n; GYY = Y.T @ Y / n; GEE = E.T @ E / n; mYE = t.diagonal(GYE)
                v1 = kap.T @ mYE                                                                                   # sum_b kappa_bc m(Y_b E_b)
                # type 1 (j-slot = Y_c): pairs (Y_b E_b), (Y_b E_c), (E_b E_c);  type 2 (j-slot = E_c): pairs (Y_b E_b), (Y_b Y_c), (E_b Y_c)
                PY = PY - (E * v1[None, :] + E @ (kap * GYE) + Y @ (kap * GEE))
                PE = PE - (Y * v1[None, :] + E @ (kap * GYY) + Y @ (kap * GYE.T))
            P = P + PY @ Y.T + PE @ E.T
        return P

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if level == 'read' or not (lo <= l <= hi) or not k31 or st.get('pp') is None or not st['pp'].shape[1]: return pin
        mu, Cpre, k21_, k4v, k22, rem = pin
        P = plane_of(l, st['pp']); C = Cpre.detach().to(f64)
        R = (specific(t, (P + P.T) / 2, C, mu.detach().to(f64), forms)[0] + (P - P.T) / 2) if forms else P
        s = t.diagonal(C).clamp_min(1e-10).sqrt(); rm = k31 * (R - t.diag(t.diagonal(R))) / (s ** 3)[:, None] / s[None, :]
        info[l] = float((rm * rm).mean().sqrt())
        rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook
    return cd

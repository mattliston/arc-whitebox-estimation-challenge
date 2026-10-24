"""TAG opus-reexam K31DIAG (lead, 2026-10-02): PASSIVE dump of candidate (3,1) planes for comparison with the Monte Carlo truth planes (8M samples, networks 35 and 100).
Nothing is added to the card. At the layers in `layers` it stores (raw units of kappa4(z_i, z_i, z_i, z_j), float32):
  mu, v, C of the layer;
  QLf_a / QL4_a   the young pairs' local quartic plane of cohort age a = 1..4: full form / fourth-chaos part (k31l.py);
  CHf / CH4       the two-hub chain trees of the newborn cohort: full / fourth-chaos part (k31c.py);
  MX1, MX2, MX3   the MIXED terms, first order in the carried third cumulant: kappa4^x(a,b,c,d) contains d_a kappa3^z(a,b,c) g_b g_c (g_d C_ad) (hub a with the second
                  derivative, two legs on a bank column r, one Gaussian edge):
                    MX1_ij = 6 sum_r lam_r P_ri P_rj sum_a d_a pp_ra E_ai Y_ai        (the edge and the hub both on i-slots)
                    MX2_ij = 3 sum_a d_a E_ai Y_aj sum_r lam_r pp_ra P_ri^2           (the edge's far end on the j-slot)
                    MX3_ij = 3 sum_a d_a Y_ai E_aj sum_r lam_r pp_ra P_ri^2           (the hub on the j-slot)
                  pp_r = bank column at the layer before (on the hub rows), P_r = the same column pushed into this layer, (E_a, Y_a) = the newborn pair of hub a.
install(cd, torch, st, layers=(3, 5, 9, 12, 14), out=<dict>)   -> out[l] = dict of arrays.  The harness must store d and Ch at the births (option k3c != 0 or k3dg=1)."""
import k31l as _k31l


def install(cd, torch, st, layers=(3, 5, 9, 12, 14), out=None):
    t = torch; f64 = t.float64; n = 1024; S = dict(prev=None)
    old_pin = cd.hooks.get('pair_in')

    def quartic(pp, B, k0, K, chaos4):
        gap = (B['tp'] - B['tm'])[None, :]; pplus = pp[:, k0:k0 + K]; pminus = pp[:, k0 + K:k0 + 2 * K]
        E = (pplus - pminus) / gap; Y = (B['tp'][None, :] * pminus - B['tm'][None, :] * pplus) / gap; S1, S2, S3, S4 = (x[None, :] for x in B['S'])
        AY = 3 * S1 * Y * Y * E + 3 * S2 * Y * E * E + S3 * E ** 3; AE = S1 * Y ** 3 + 3 * S2 * Y * Y * E + 3 * S3 * Y * E * E + S4 * E ** 3
        if chaos4:
            mY2, mE2 = (Y * Y).mean(0)[None, :], (E * E).mean(0)[None, :]
            AY = AY - 3 * (S1 * mY2 * E + S2 * mE2 * Y + S3 * mE2 * E); AE = AE - 3 * (S1 * mY2 * Y + S2 * mY2 * E + S3 * mE2 * Y + S4 * mE2 * E)
        return AY @ Y.T + AE @ E.T, E, Y

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if l not in layers or st.get('pp') is None or not st['pp'].shape[1]: return pin
        mu, Cpre = pin[0], pin[1]; pp = st['pp'].detach().to(f64); stops = st['stops']; d32 = lambda x: x.to(t.float32).numpy()
        rec = dict(mu=d32(mu.detach()), v=d32(t.diagonal(Cpre.detach())), C=d32(Cpre.detach()))
        for a in (1, 2, 3, 4):
            j = l - a
            if j < 0 or j not in st['births']: continue
            k0 = stops[j]; K = (stops[j + 1] - k0) // 2
            if not K: continue
            rec['QLf_%d' % a] = d32(quartic(pp, st['births'][j], k0, K, False)[0]); rec['QL4_%d' % a] = d32(quartic(pp, st['births'][j], k0, K, True)[0])
        j = l - 1
        if j in st['births'] and 'Ch' in st['births'][j]:
            B = st['births'][j]; k0 = stops[j]; K = (stops[j + 1] - k0) // 2; _, E, Y = quartic(pp, B, k0, K, False)
            d = B['d']; kap = 6 * d[:, None] * B['Ch'] * d[None, :]; kap = kap - t.diag(t.diagonal(kap)); M = (Y * E) @ kap; PY = M * E; PE = M * Y
            rec['CHf'] = d32(PY @ Y.T + PE @ E.T)
            GYE = Y.T @ E / n; GYY = Y.T @ Y / n; GEE = E.T @ E / n; v1 = kap.T @ t.diagonal(GYE)
            PY4 = PY - (E * v1[None, :] + E @ (kap * GYE) + Y @ (kap * GEE)); PE4 = PE - (Y * v1[None, :] + E @ (kap * GYY) + Y @ (kap * GYE.T))
            rec['CH4'] = d32(PY4 @ Y.T + PE4 @ E.T)
            P = S['prev']
            if P is not None and P['ncol'] and P['ncol'] == k0:
                Pold = pp[:, :k0]; lam = st['lam_in'].detach().to(f64)[:k0]; PPh = P['pp'][B['sel'], :]                       # hubs x old columns, at the layer before
                A = (E * Y) * d[None, :]; theta = 6 * (A @ PPh)
                rec['MX1'] = d32((theta * Pold * lam[None, :]) @ Pold.T)
                Xi = (Pold * Pold * lam[None, :]) @ PPh.T
                rec['MX2'] = d32(3 * (Xi * E * d[None, :]) @ Y.T); rec['MX3'] = d32(3 * (Xi * Y * d[None, :]) @ E.T)
        out[l] = rec
        return pin
    cd.hooks['pair_in'] = pin_hook
    old_out = cd.hooks.get('state_out')

    def out_hook(l, o_, ctx):
        if old_out is not None: o_ = old_out(l, o_, ctx)
        if l <= 14 and st.get('pp') is not None: S['prev'] = dict(pp=st['pp'].detach().to(f64).clone(), ncol=int(st['pp'].shape[1]))
        elif l <= 14: S['prev'] = dict(pp=None, ncol=0)
        return o_
    cd.hooks['state_out'] = out_hook
    return cd

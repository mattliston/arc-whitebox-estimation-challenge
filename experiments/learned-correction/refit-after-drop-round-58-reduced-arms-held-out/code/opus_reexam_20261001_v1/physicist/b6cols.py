"""TAG opus-physicist B6COLS (2026-10-10): PER-UNIT COLUMNS FOR THE LEARNED MEAN CORRECTION, batch 6 of the lead (10-10 02:13Z: the physicist's right-but-rejected
terms as head inputs: 'what this term would do to unit i') and the best candidates of the open-loop screen against truth (physicist T61).
install(cd, torch) adds read-only hooks (chained on top of whatever is installed; no number of the card changes) that fill
    cd.b6cols[layer] = dict(name -> tensor (1024,) float64, raw, NOT standardised) and cd.b6cols[layer]['resp'] = the read's response rows (1.0 / 0.0)
for layers 1-14 (from what the pair map receives, after every other pair_in hook) and layer 15 (from the k3_15 / k4_15 hooks). The head-input code standardises each over the
response rows (layers 1-14) or the kept rows (layer 15), clips at 8 and puts 0 elsewhere, like column 6. Install it AFTER the card's modules (otr_m.install), so that its
'pair_in' wrapper sees the final inputs of the pair map; the side table cd.m_side (pushed columns, their weights, cohort starts, births) is read for the cohort sums.
Notation (layer l, pre-activations): mu, v = diag C, s = sqrt v, a = mu / s, ph = pdf(a), g = gate = Phi(a), k3 = diag of the K21 plane, k4 = the per-unit kurtosis the pair map
receives (radial + K4L), t3 = k3 / (v s), t4 = k4 / v^2, He2 = a^2 - 1, He3 = a^3 - 3 a, He4 = a^4 - 6 a^2 + 3.
  b6.1 'm2'    the second-order mean term over sigma: t3^2 He4 ph / 72                                       (so2.single, mean part; pbase so2m adds sigma x this to the host)
  b6.2 'k3d2'  the second-order change of the post third cumulant on the diagonal, over v s: (pk3 - pk3_0) / (v s), pk3 = (r3 + q3) - 3 m1 (r2 + q2) + 2 m1^3,
               pk3_0 = r3 - 3 m r2 + 2 m^3 with the card's own first-order raw moments r2, r3 and mean m, m1 = m + s m2, q2, q3 of so2.single   (so2.py, line of p21n, diagonal)
  b6.3 'f15'   at layer 15: 'm2' from the final layer's vv, k3, k4                                            (pbase so2f)
  b6.4 'gk'    the gate correction of gk over the gate: (t3 He2 ph / 6 - t4 He3 ph / 24) / g             'gb' the b-mode gate correction over the gate: (ph / s) bpre / g
  b6.5 'xk4'   the cross form's per-unit kurtosis part over v^2: 3.2 k3s b / v^2, k3s = the bank's + the star's per-unit k3 (the specific skew), b = the pushed b vector
  b6.6 'rg'    the row-gain law of the bank read (physicist T55b, sigma term + mean): 0.004 + 0.0022 (l - 3) - (0.068 + 0.033 (l - 3)) (log s - <log s>), <> over the response rows
  T61 screen:
  'qold'       the local quartic (K4L's hub terms) of the bank pairs aged 3 and more, per unit, over v^2 (raw: the isotropic part is NOT projected out)
  'eold'       sum over the columns of the cohorts aged 3 and more of |lam_r| pp_ir^2, over v
  'qb'         the bank read's plane (response rows x live columns, off the diagonal) row-summed against c_j = (g_j / s_j)^2 mu_j, over ((c . mu - c_i mu_i) v_i)
At layer 15 only 'f15', 'qold', 'eold' (no plane, no gate transport there).
Cost natively: m2, k3d2, f15, gk, gb, xk4, rg: elementwise on per-unit vectors the step holds; qb: one matrix-vector product of the read's plane; qold, eold: one pass over
(units x columns of the old cohorts), the K4L arithmetic extended to older pairs."""
import math

AGE_OLD = 3


def _he(a):
    return a * a - 1, a ** 3 - 3 * a, a ** 4 - 6 * a * a + 3


def _old_sums(t, side, l, v):
    """local quartic and |lam| energy of the cohorts aged >= AGE_OLD, from the columns pushed into layer l"""
    f64 = t.float64; n = v.shape[0]; pp = side['pp'].detach().to(f64); lam = side['lam_in'].detach().to(f64)[:pp.shape[1]]
    starts = side['starts']; births = side['births']; q = t.zeros(n, dtype=f64); e = t.zeros(n, dtype=f64)
    for j in range(l):
        if l - j < AGE_OLD or j + 1 >= len(starts): continue
        k0, k1 = starts[j], starts[j + 1]
        if k1 <= k0: continue
        e = e + (pp[:, k0:k1] ** 2) @ lam[k0:k1].abs()
        B = births.get(j); K = (k1 - k0) // 2
        if B is None or not K or len(B['tp']) != K: continue
        gap = (B['tp'] - B['tm']).to(f64)[None, :]; pl, pm = pp[:, k0:k0 + K], pp[:, k0 + K:k0 + 2 * K]
        E = (pl - pm) / gap; Y = (B['tp'].to(f64)[None, :] * pm - B['tm'].to(f64)[None, :] * pl) / gap; S1, S2, S3, S4 = (x.to(f64)[None, :] for x in B['S'])
        q = q + (4 * S1 * Y ** 3 * E + 6 * S2 * Y ** 2 * E ** 2 + 4 * S3 * Y * E ** 3 + S4 * E ** 4).sum(1)
    return q / v ** 2, e / v


def _so2_parts(t, mu, v, k3, k4):
    """m2 / s and the diagonal change of the post third cumulant / (v s), as so2.py computes them"""
    s = v.sqrt(); a = mu / s; ph = t.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = t.erfc(-a / math.sqrt(2)) / 2; he2, he3, he4 = _he(a)
    t3 = k3 / (v * s); q = t3 * t3
    m2s, q2, q3 = q * he4 * ph / 72, -q * v * he3 * ph / 36, q * v * s * he2 * ph / 12
    d, e = ph / s, -a * ph / v
    r2g = (mu * mu + v) * g + mu * s * ph; r3g = mu * (mu * mu + 3 * v) * g + (mu * mu + 2 * v) * s * ph
    r2 = r2g + k3 * d / 3 + k4 * e / 12; r3 = r3g + k3 * g + k4 * d / 4
    m = s * ((ph + a * g) + (t3 / 6) * (-a * ph) + (k4 / (v * v) / 24) * (he2 * ph))      # the card's first-order mean
    m1 = m + s * m2s; pk3 = (r3 + q3) - 3 * m1 * (r2 + q2) + 2 * m1 ** 3; pk3_0 = r3 - 3 * m * r2 + 2 * m ** 3
    return m2s, (pk3 - pk3_0) / (v * s)


def install(cd, torch):
    t = torch; f64 = t.float64; cd.b6cols = {}; side = getattr(cd, 'm_side', None)
    pin0 = cd.hooks.get('pair_in')

    def pin_hook(l, pin, ctx):
        if pin0 is not None: pin = pin0(l, pin, ctx)
        if not 1 <= l <= 14 or not t.is_tensor(pin[2]) or pin[2].ndim != 2: return pin
        with t.no_grad():
            mu, C, K21 = pin[0].detach().to(f64), pin[1].detach().to(f64), pin[2].detach().to(f64); n = mu.shape[0]
            v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); a = mu / s; ph = t.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = t.erfc(-a / math.sqrt(2)) / 2
            k3 = t.diagonal(K21); k4 = pin[3].detach().to(f64) * t.ones(n, dtype=f64) if t.is_tensor(pin[3]) else float(pin[3]) * t.ones(n, dtype=f64)
            t3, t4 = k3 / (v * s), k4 / (v * v); he2, he3, _ = _he(a); gs = g.clamp_min(1e-6); bp = ctx['bpre'].detach().to(f64)
            resp = ctx['response_mask'].detach().to(f64); rm = resp > 0.5; eye = t.eye(n, dtype=t.bool)
            cp = ctx['cp'].detach().to(f64); sp = ctx['sp'].detach().to(f64)
            m2s, k3d2 = _so2_parts(t, mu, v, k3, k4)
            c = (g / s) ** 2 * mu; den = (c * mu).sum() - c * mu
            ls = s.log(); lsc = ls - (ls[rm].mean() if bool(rm.any()) else ls.mean())
            out = dict(m2=m2s, k3d2=k3d2, gk=(t3 * he2 * ph / 6 - t4 * he3 * ph / 24) / gs, gb=(ph / s) * bp / gs,
                       xk4=3.2 * (t.diagonal(cp) + t.diagonal(sp)) * bp / (v * v), rg=0.004 + 0.0022 * (l - 3) - (0.068 + 0.033 * (l - 3)) * lsc,
                       qb=((cp.masked_fill(eye, 0.) @ c) / den) / v, resp=resp)
            if side is not None and side.get('pp') is not None:
                out['qold'], out['eold'] = _old_sums(t, side, l, v)
            cd.b6cols[l] = out
        return pin
    cd.hooks['pair_in'] = pin_hook

    k3h0, k4h0 = cd.hooks.get('k3_15'), cd.hooks.get('k4_15'); S15 = {}

    def k3_15(l, k3, ctx):
        k3 = k3 if k3h0 is None else k3h0(l, k3, ctx); S15['k3'] = k3.detach().to(f64); return k3

    def k4_15(l, k4, ctx):
        k4 = k4 if k4h0 is None else k4h0(l, k4, ctx)
        with t.no_grad():
            vv, mu = ctx['vv'].detach().to(f64).clamp_min(1e-12), ctx['mu'].detach().to(f64); n = vv.shape[0]
            k4v = k4.detach().to(f64) * t.ones(n, dtype=f64) if t.is_tensor(k4) else float(k4) * t.ones(n, dtype=f64)
            out = dict(f15=_so2_parts(t, mu, vv, S15['k3'], k4v)[0])
            if side is not None and side.get('pp') is not None: out['qold'], out['eold'] = _old_sums(t, side, 15, vv)
            cd.b6cols[15] = out
        return k4
    cd.hooks['k3_15'], cd.hooks['k4_15'] = k3_15, k4_15
    return cd

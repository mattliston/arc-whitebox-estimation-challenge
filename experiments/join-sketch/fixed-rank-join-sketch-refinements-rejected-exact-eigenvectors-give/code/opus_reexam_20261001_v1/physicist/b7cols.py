"""TAG opus-physicist B7COLS (2026-10-10): PER-UNIT COLUMNS FOR THE LEARNED MEAN CORRECTION, batch 7 = the passing NEW columns of the adjoint screen T64
(exact zero-weight gradients of the final-layer MSE on the S42 torch card, 36 of the trainer's gradient ranks; PARTIAL = the part of each column that the head's
126 inputs and column 6 do not already hold; table physicist/t64_screen_36_full.txt). Layers 1-14, from what the pair map receives (like b6cols).
install(cd, torch) adds read-only hooks (chained on whatever is installed; no number of the card changes) that fill
    cd.b7cols[layer] = dict(name -> tensor (1024,) float64, raw, NOT standardised) and cd.b7cols[layer]['resp'] = the read's response rows (1.0 / 0.0).
Standardise each over ALL units (convention 'a' of T64; kmu / kg / qk are 4-5x stronger in it than on the response rows), clip at 8.
Notation (layer l, pre-activations): mu, v = diag C, s = sqrt v, a = mu / s, ph = pdf(a), g = gate = Phi(a), K21 = the K21 plane the pair map receives, Ko = K21 with its
diagonal set to 0, t3 = diag(K21) / (v s), t4 = k4 / v^2 (k4 = what the pair map receives).
  'fold'  the diagonal of the fourth-chaos plane (k31l's chaos4 form, physicist T60 per_age 'f') of the bank's pairs aged 3 and more, summed over those cohorts, over v^2
          (pairs of cohort j: columns k0:k0+K (plus) and k0+K:k0+2K (minus) of the pushed bank pp, births B = (tp, tm, S1..S4) of that cohort)
          partial: 102 / 128 weights beyond 2.5 se, mean t^2 27.7, halves 0.99, first order -0.455% at step 0.05
  'kmu'   sum_j Ko_ij mu_j / (v_i s_i)                                     partial: 109 / 128, mean t^2 48.4, halves 0.98, first order -0.083%
  'kg'    sum_j Ko_ij g_j / (v_i s_i)                                      partial: 109 / 128, mean t^2 41.7, halves 0.97, first order -0.058%
  'qk'    (sum_j Ko_ij c_j / (c . mu - c_i mu_i)) / v_i, c_j = (g_j / s_j)^2 mu_j   partial: 105 / 128, mean t^2 41.6, halves 0.98, first order -0.073%
  'e34'   the next Edgeworth mean term over sigma: -t3 t4 He5(a) ph / 144, He5 = a^5 - 10 a^3 + 15 a (elementwise, free)   partial: 40 / 128, 4.43, -0.137%
  's1'    sum_r lam_r pp_ir^2 over the columns of the cohort born one layer back (age 1), over v   partial: 37 / 128, 4.11, -0.139%
(qold / eold / qb / m2 are in b6cols; columns 9 / 16 / 20 in the trainer's module.)
Cost natively: kmu / kg / qk: one matrix-vector product each with the K21 plane (or one with three right-hand sides); e34 elementwise; s1 one pass over the youngest cohort's
columns; fold: one pass over (units x columns of the cohorts aged 3+) with the births' four moments, the K4L-style arithmetic of qold (about the same price as qold)."""
import math

AGE_OLD = 3


def _fold(t, side, l, v):
    """the fourth-chaos plane's diagonal of the cohorts aged >= AGE_OLD (T60 per_age 'f', summed), over v^2; and the age-1 signed energy s1 over v"""
    f64 = t.float64; n = v.shape[0]; pp = side['pp'].detach().to(f64); lam = side['lam_in'].detach().to(f64)[:pp.shape[1]]
    starts = side['starts']; births = side['births']; f = t.zeros(n, dtype=f64); s1 = t.zeros(n, dtype=f64)
    for j in range(l):
        if j + 1 >= len(starts): continue
        k0, k1 = starts[j], starts[j + 1]
        if k1 <= k0: continue
        if l - j == 1: s1 = s1 + (pp[:, k0:k1] ** 2) @ lam[k0:k1]
        if l - j < AGE_OLD: continue
        B = births.get(j); K = (k1 - k0) // 2
        if B is None or not K or len(B['tp']) != K: continue
        gap = (B['tp'] - B['tm']).to(f64)[None, :]; pl, pm = pp[:, k0:k0 + K], pp[:, k0 + K:k0 + 2 * K]
        Ev = (pl - pm) / gap; Y = (B['tp'].to(f64)[None, :] * pm - B['tm'].to(f64)[None, :] * pl) / gap; S1, S2, S3, S4 = (x.to(f64)[None, :] for x in B['S'])
        AY = 3 * S1 * Y * Y * Ev + 3 * S2 * Y * Ev * Ev + S3 * Ev ** 3; AE = S1 * Y ** 3 + 3 * S2 * Y * Y * Ev + 3 * S3 * Y * Ev * Ev + S4 * Ev ** 3
        mY2, mYE, mE2 = (Y * Y).mean(0)[None, :], (Y * Ev).mean(0)[None, :], (Ev * Ev).mean(0)[None, :]
        AY = AY - 3 * (S1 * (mY2 * Ev + 2 * mYE * Y) + S2 * (mE2 * Y + 2 * mYE * Ev) + S3 * mE2 * Ev)
        AE = AE - 3 * (S1 * mY2 * Y + S2 * (mY2 * Ev + 2 * mYE * Y) + S3 * (mE2 * Y + 2 * mYE * Ev) + S4 * mE2 * Ev)
        f = f + (AY * Y + AE * Ev).sum(1)
    return f / v ** 2, s1 / v


def install(cd, torch):
    t = torch; f64 = t.float64; cd.b7cols = {}; side = getattr(cd, 'm_side', None)
    pin0 = cd.hooks.get('pair_in')

    def pin_hook(l, pin, ctx):
        if pin0 is not None: pin = pin0(l, pin, ctx)
        if not 1 <= l <= 14 or not t.is_tensor(pin[2]) or pin[2].ndim != 2 or (side is not None and side.get('again')): return pin
        with t.no_grad():
            mu, C, K21 = pin[0].detach().to(f64), pin[1].detach().to(f64), pin[2].detach().to(f64); n = mu.shape[0]
            v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); a = mu / s; ph = t.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = ctx['gate'].detach().to(f64)
            k4 = pin[3].detach().to(f64) * t.ones(n, dtype=f64) if t.is_tensor(pin[3]) else float(pin[3]) * t.ones(n, dtype=f64)
            Ko = K21.masked_fill(t.eye(n, dtype=t.bool), 0.); t3, t4 = t.diagonal(K21) / (v * s), k4 / (v * v)
            c = (g / s) ** 2 * mu; den = (c * mu).sum() - c * mu
            out = dict(kmu=(Ko @ mu) / (v * s), kg=(Ko @ g) / (v * s), qk=((Ko @ c) / den) / v, e34=-t3 * t4 * (a ** 5 - 10 * a ** 3 + 15 * a) * ph / 144,
                       resp=ctx['response_mask'].detach().to(f64))
            if side is not None and side.get('pp') is not None: out['fold'], out['s1'] = _fold(t, side, l, v)
            cd.b7cols[l] = out
        return pin
    cd.hooks['pair_in'] = pin_hook
    return cd

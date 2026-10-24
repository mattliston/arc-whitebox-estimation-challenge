"""TAG opus-physicist: the per-age cohort sums of T60 (t60_s42_dump.py), as an importable function (torch passed in); code copied unchanged."""


def per_age(l, side, v, mu, t, f64, n, BANDS):
    """per-unit sums over the bank's cohorts by pair age, from the columns pushed into layer l"""
    pp = side['pp'].detach().to(f64); lam = side['lam_in'].detach().to(f64)[:pp.shape[1]]; starts = side['starts']; births = side['births']; s = v.sqrt(); out = {}
    for nm, a0, a1 in BANDS:
        q = t.zeros(n, dtype=f64); f = t.zeros(n, dtype=f64); c = t.zeros(n, dtype=f64); e = t.zeros(n, dtype=f64); sg = t.zeros(n, dtype=f64)
        for j in range(l):
            if not a0 <= l - j <= a1 or j + 1 >= len(starts): continue
            k0, k1 = starts[j], starts[j + 1]
            if k1 <= k0: continue
            P_ = pp[:, k0:k1]; L_ = lam[k0:k1]; c = c + (P_ ** 3) @ L_; e = e + (P_ ** 2) @ L_.abs(); sg = sg + (P_ ** 2) @ L_
            B = births.get(j); K = (k1 - k0) // 2
            if B is None or not K or len(B['tp']) != K: continue
            gap = (B['tp'] - B['tm']).to(f64)[None, :]; pl, pm = pp[:, k0:k0 + K], pp[:, k0 + K:k0 + 2 * K]
            Ev = (pl - pm) / gap; Y = (B['tp'].to(f64)[None, :] * pm - B['tm'].to(f64)[None, :] * pl) / gap; S1, S2, S3, S4 = (x.to(f64)[None, :] for x in B['S'])
            q = q + (4 * S1 * Y ** 3 * Ev + 6 * S2 * Y ** 2 * Ev ** 2 + 4 * S3 * Y * Ev ** 3 + S4 * Ev ** 4).sum(1)
            AY = 3 * S1 * Y * Y * Ev + 3 * S2 * Y * Ev * Ev + S3 * Ev ** 3; AE = S1 * Y ** 3 + 3 * S2 * Y * Y * Ev + 3 * S3 * Y * Ev * Ev + S4 * Ev ** 3
            mY2, mYE, mE2 = (Y * Y).mean(0)[None, :], (Y * Ev).mean(0)[None, :], (Ev * Ev).mean(0)[None, :]
            AY = AY - 3 * (S1 * (mY2 * Ev + 2 * mYE * Y) + S2 * (mE2 * Y + 2 * mYE * Ev) + S3 * mE2 * Ev)
            AE = AE - 3 * (S1 * mY2 * Y + S2 * (mY2 * Ev + 2 * mYE * Y) + S3 * (mE2 * Y + 2 * mYE * Ev) + S4 * mE2 * Ev)
            f = f + (AY * Y + AE * Ev).sum(1)
        out.update({f'q{nm}': q / v ** 2, f'f{nm}': f / v ** 2, f'c{nm}': c / (v * s), f'e{nm}': e / v, f's{nm}': sg / v})
    return out

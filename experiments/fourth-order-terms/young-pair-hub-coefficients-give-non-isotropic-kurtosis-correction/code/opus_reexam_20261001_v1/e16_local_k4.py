#!/usr/bin/env python3
"""TAG opus-reexam E16: per-unit fourth cumulant of the pre-activations from the exact LOCAL hub coefficients (free: elementwise on the
bank's own column pairs), against Monte Carlo truth (32M per-unit, two independent draws) and against the card's own per-unit k4.
   k4_local(i) = sum over pairs (all ages)  4 S1 Y_i^3 Wt_i + 6 S2 Y_i^2 Wt_i^2 + 4 S3 Y_i Wt_i^3 + S4 Wt_i^4      (see e15_local_feed.py)
Reported per layer: the card's k4 against truth (relative rms error, coefficient), the share of the truth-minus-card residual explained by
k4_local (fitted on draw A, scored on draw B), by its parts, and the same with the isotropic part refitted.
usage: e16_local_k4.py <rank> <layers>"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE / 'physicist'; P2 = HERE.parent
import numpy as np
rank = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(',')]
BI = np.load(PH / f't4births_rank{rank:04d}.npz'); A2 = np.load(HERE / f'analyst/a2_card_rank{rank:04d}.npz')
PU = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz'); PUi = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz')


def hub_data(j):
    b = lambda k: BI[f'{k}_{j}']
    s_ = b('sel'); a, s, gt, ph, m = (b(k)[s_].astype(np.float64) for k in ('alpha', 'sigma', 'gate', 'pdf', 'host'))
    tp, tm = (b(k)[s_].astype(np.float64) for k in ('plus', 'minus')); vv = s * s; dj = ph / s
    r2 = vv * ((1 + a * a) * gt + a * ph); r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph); r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
    return tp, tm, (-a * ph / vv, 2 * (gt - gt * gt - m * dj), 3 * (1 - gt) * (r2 - 2 * m * m), r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2)


for l in layers:
    src = PH / f't4cols_rank{rank:04d}_L{l}.npz'
    if not src.exists(): src = HERE / f'e15cols_rank{rank:04d}_L{l}.npz'
    COL = np.load(src); pp = COL['pp'].astype(np.float64); stops = [0] + [int(x) for x in COL['stops']]
    parts = {f'S{m}_{g}': np.zeros(1024) for m in (1, 2, 3, 4) for g in ('age1', 'age2', 'age3p')}
    for j in range(l):
        tp, tm, S = hub_data(j); k0 = stops[j]; K = (stops[j + 1] - k0) // 2; gap = (tp - tm)[None, :]
        Wt = (pp[:, k0:k0 + K] - pp[:, k0 + K:k0 + 2 * K]) / gap; Y = (tp[None, :] * pp[:, k0 + K:k0 + 2 * K] - tm[None, :] * pp[:, k0:k0 + K]) / gap
        g_ = 'age1' if l - j == 1 else ('age2' if l - j == 2 else 'age3p')
        parts[f'S1_{g_}'] += (4 * S[0][None, :] * Y ** 3 * Wt).sum(1); parts[f'S2_{g_}'] += (6 * S[1][None, :] * Y ** 2 * Wt ** 2).sum(1)
        parts[f'S3_{g_}'] += (4 * S[2][None, :] * Y * Wt ** 3).sum(1); parts[f'S4_{g_}'] += (S[3][None, :] * Wt ** 4).sum(1)
    loc = sum(parts.values()); loc1 = sum(v for k, v in parts.items() if k.endswith('age1')); loc12 = loc1 + sum(v for k, v in parts.items() if k.endswith('age2'))
    ta = PU['mc_pre_k4'][l].astype(np.float64); tb = PUi['mc_pre_k4'][l].astype(np.float64); va = PU['mc_pre_v'][l].astype(np.float64)
    kc = A2[f'k4_{l}'].astype(np.float64); vc = A2[f'v_{l}'].astype(np.float64); radial = float(A2[f'radial_{l}'])
    mu = PU['mc_pre_m'][l].astype(np.float64); al = mu / np.sqrt(va); w4 = ((al * al - 1) * np.exp(-al * al / 2)) ** 2 + 1e-12     # weight of k4 in the mean formula
    noise = float(((ta - tb) ** 2).mean() / 2)
    def stats(pred, w=None):
        w = np.ones(1024) if w is None else w
        ea, eb = ta - pred, tb - pred
        return dict(rel_rms_err=round(math.sqrt(max(float((w * ea * eb).sum()), 0.0) / float((w * ta * tb).sum())), 4))
    def expl(base, feats, w=None):
        """share of the noise-free energy of (truth - base) explained by feats, fitted on draw A, scored on draw B and the reverse."""
        w = np.ones(1024) if w is None else w; X = np.stack(feats, 1) * np.sqrt(w)[:, None]; out = []
        for t1, t2 in ((ta, tb), (tb, ta)):
            r1, r2_ = (t1 - base) * np.sqrt(w), (t2 - base) * np.sqrt(w); co = np.linalg.lstsq(X, r1, rcond=None)[0]; fit = X @ co
            out.append(((2 * float(fit @ r2_) - float(fit @ fit)) / float(r1 @ r2_), co))
        return round(float(np.mean([o[0] for o in out])), 4), [round(float(x), 3) for x in (out[0][1] + out[1][1]) / 2]
    iso = va * va
    rec = dict(layer=l, radial_card=round(radial, 5), truth_k4_over_v2_mean=round(float((ta / iso).mean()), 5), card_k4_over_v2_mean=round(float((kc / (vc * vc)).mean()), 5),
               truth_noise_rel=round(math.sqrt(noise / float((ta * tb).mean())), 4),
               card=stats(kc), card_mean_weighted=stats(kc, w4), card_is_isotropic_corr=round(float(np.corrcoef(kc, vc * vc)[0, 1]), 5),
               iso_fit=stats(iso * float((iso * ta).sum() / (iso * iso).sum())),
               local_norm_over_truth=round(float(np.linalg.norm(loc) / math.sqrt(float((ta * tb).sum()))), 4),
               corr_local_truth_minus_card=round(float(np.corrcoef(loc, (ta + tb) / 2 - kc)[0, 1]), 4),
               residual_of_card_explained=dict(local_all=expl(kc, [loc]), local_age1=expl(kc, [loc1]), local_age12=expl(kc, [loc12]), iso_only=expl(kc, [iso]), iso_plus_local=expl(kc, [iso, loc]),
                                               iso_plus_parts=expl(kc, [iso] + [sum(v for k, v in parts.items() if k.startswith(f'S{m}_')) for m in (1, 2, 3, 4)])),
               residual_of_card_explained_mean_weighted=dict(local_all=expl(kc, [loc], w4), iso_plus_local=expl(kc, [iso, loc], w4)),
               card_plus_local_at_1=stats(kc + loc), card_plus_local_at_1_mean_weighted=stats(kc + loc, w4))
    print(json.dumps(rec), flush=True)

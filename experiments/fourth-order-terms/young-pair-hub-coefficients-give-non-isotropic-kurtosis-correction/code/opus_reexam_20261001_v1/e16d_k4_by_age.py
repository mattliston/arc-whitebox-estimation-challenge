#!/usr/bin/env python3
"""TAG opus-reexam E16d: E16's open-loop test by column age, weighted by the kurtosis term of the mean formula (network 35).
For each layer: share of the card's per-unit k4 residual energy (truth - radial v^2, weighted) explained by isotropic + hub terms of
ages <= A (fitted on draw A, scored on draw B), with the fitted hub coefficient; and at coefficient 1 after removing the weighted
isotropic projection (what K4L applies). usage: e16d_k4_by_age.py <rank> <layers>"""
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
    COL = np.load(src); pp = COL['pp'].astype(np.float64); stops = [0] + [int(x) for x in COL['stops']]; by_age = {}
    for j in range(l):
        tp, tm, S = hub_data(j); k0 = stops[j]; K = (stops[j + 1] - k0) // 2; gap = (tp - tm)[None, :]
        Wt = (pp[:, k0:k0 + K] - pp[:, k0 + K:k0 + 2 * K]) / gap; Y = (tp[None, :] * pp[:, k0 + K:k0 + 2 * K] - tm[None, :] * pp[:, k0:k0 + K]) / gap
        by_age[l - j] = (Wt * (4 * S[0][None, :] * Y ** 3 + Wt * (6 * S[1][None, :] * Y ** 2 + Wt * (4 * S[2][None, :] * Y + S[3][None, :] * Wt)))).sum(1)
    ta = PU['mc_pre_k4'][l].astype(np.float64); tb = PUi['mc_pre_k4'][l].astype(np.float64); va = PU['mc_pre_v'][l].astype(np.float64)
    kc = A2[f'k4_{l}'].astype(np.float64); mu = PU['mc_pre_m'][l].astype(np.float64); al = mu / np.sqrt(va)
    w = ((al * al - 1) * np.exp(-al * al / 2)) ** 2 + 1e-12; sw = np.sqrt(w); iso = va * va
    ra, rb = (ta - kc) * sw, (tb - kc) * sw; E = float(ra @ rb); out = dict(layer=l, weighted_residual_rel=round(math.sqrt(E / float((w * ta * tb).sum())), 4))
    for A in (1, 2, 4, 6, 99):
        loc = sum(v for a_, v in by_age.items() if a_ <= A); X = np.stack([iso, loc], 1) * sw[:, None]; sc = []; cs = []
        for r1, r2_ in ((ra, rb), (rb, ra)):
            co = np.linalg.lstsq(X, r1, rcond=None)[0]; fit = X @ co; sc.append((2 * float(fit @ r2_) - float(fit @ fit)) / E); cs.append(co)
        beta = (w * loc * iso).sum() / (w * iso * iso).sum(); dk = (loc - beta * iso) * sw
        at1 = (2 * float(dk @ ra) + 2 * float(dk @ rb)) / 2 - float(dk @ dk)
        # with the isotropic level refitted afterwards (what a retuned band dial could do)
        r1 = ra - dk; r2_ = rb - dk; c_iso = float((iso * sw) @ (r1 + r2_) / 2) / float((iso * sw) @ (iso * sw)); f2 = dk + c_iso * iso * sw
        at1_iso = (float(f2 @ ra) + float(f2 @ rb)) - float(f2 @ f2)
        out[f'age<={A}'] = dict(fitted=round(float(np.mean(sc)), 4), hub_coef=round(float(np.mean([c[1] for c in cs])), 3), iso_coef=round(float(np.mean([c[0] for c in cs])), 4),
                               K4L_at_1=round(at1 / E, 4), K4L_at_1_iso_refit=round(at1_iso / E, 4))
    print(json.dumps(out), flush=True)

#!/usr/bin/env python3
"""TAG opus-reexam E20: does the COLLECTIVE term add to K4L's per-unit kurtosis? (open loop, network 35, layers 9 and 13)
The same-unit part of z_i^2 responds linearly to the previous layer (d Var(x_a)/d mu_a = V21_a), which gives the per-unit fourth cumulant
   k4_coll(i) = 6 kappa(z_i, z_i, zeta),  zeta = (2/n) sum_a V21_a (z_a - mu_a) of the previous layer
              = 6 sum_r lam_r c_r pp_r(l)_i^2,   c_r = (2/n) V21 . (ungated leg of column r one layer earlier)        (E15b's scalars)
(for a pair born at the previous layer the ungated leg is the covariance column of its hub). Elementwise on the bank: free.
Also the exact-weight version (W_ai^2 instead of 2/n): k4_collx(i) = 6 sum_r lam_r pp_ir^2 [(W o W)' (V21 o leg_r)]_i  (one more push; reference).
Scored as in E16d: share of the card's weighted per-unit k4 residual explained by isotropic + K4L's hub terms (age <= 2) + the new term.
usage: e20_k4_collective.py <rank> <layer>"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE / 'physicist'; P2 = HERE.parent
import numpy as np
rank = int(sys.argv[1]); l = int(sys.argv[2]); n = 1024
BI = np.load(PH / f't4births_rank{rank:04d}.npz'); A2 = np.load(HERE / f'analyst/a2_card_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
PU = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz'); PUi = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
Wl = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')[l].astype(np.float64)
COL = np.load(PH / f't4cols_rank{rank:04d}_L{l}.npz'); pp = COL['pp'].astype(np.float64); lam = COL['lam'].astype(np.float64); stops = [0] + [int(x) for x in COL['stops']]
PRE = np.load(HERE / f'e15cols_rank{rank:04d}_L{l - 1}.npz'); ppm = PRE['pp'].astype(np.float64)
Cm = MC[f'cov_pre_{l - 1}'].astype(np.float64); mm = BI[f'host_{l - 1}'].astype(np.float64); gm = BI[f'gate_{l - 1}'].astype(np.float64); V21 = 2 * mm * (1 - gm)
sel = BI[f'sel_{l - 1}']
ung = np.concatenate((ppm, Cm[:, sel], Cm[:, sel]), 1); assert ung.shape[1] == pp.shape[1]
c = (2.0 / n) * (V21[:, None] * ung).sum(0)
coll = 6 * (pp * pp) @ (lam * c)
X = (Wl * Wl).T @ (V21[:, None] * ung); collx = 6 * ((pp * pp) * X * lam[None, :]).sum(1)
ages = np.zeros(pp.shape[1], dtype=int)
for j in range(l): ages[stops[j]:stops[j + 1]] = l - j
coll_y = 6 * (pp[:, ages <= 2] ** 2) @ (lam[ages <= 2] * c[ages <= 2])
def hub_data(j):
    b = lambda k: BI[f'{k}_{j}']
    s_ = b('sel'); a, s, gt, ph, m = (b(k)[s_].astype(np.float64) for k in ('alpha', 'sigma', 'gate', 'pdf', 'host'))
    tp, tm = (b(k)[s_].astype(np.float64) for k in ('plus', 'minus')); vv = s * s; dj = ph / s
    r2 = vv * ((1 + a * a) * gt + a * ph); r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph); r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
    return tp, tm, (-a * ph / vv, 2 * (gt - gt * gt - m * dj), 3 * (1 - gt) * (r2 - 2 * m * m), r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2)
loc = np.zeros(n)
for j in (l - 2, l - 1):
    tp, tm, S = hub_data(j); k0 = stops[j]; K = (stops[j + 1] - k0) // 2; gap = (tp - tm)[None, :]
    Wt = (pp[:, k0:k0 + K] - pp[:, k0 + K:k0 + 2 * K]) / gap; Y = (tp[None, :] * pp[:, k0 + K:k0 + 2 * K] - tm[None, :] * pp[:, k0:k0 + K]) / gap
    loc += (Wt * (4 * S[0][None, :] * Y ** 3 + Wt * (6 * S[1][None, :] * Y ** 2 + Wt * (4 * S[2][None, :] * Y + S[3][None, :] * Wt)))).sum(1)
ta = PU['mc_pre_k4'][l].astype(np.float64); tb = PUi['mc_pre_k4'][l].astype(np.float64); va = PU['mc_pre_v'][l].astype(np.float64)
kc = A2[f'k4_{l}'].astype(np.float64); mu = PU['mc_pre_m'][l].astype(np.float64); al = mu / np.sqrt(va)
w = ((al * al - 1) * np.exp(-al * al / 2)) ** 2 + 1e-12; sw = np.sqrt(w); iso = va * va
ra, rb = (ta - kc) * sw, (tb - kc) * sw; E = float(ra @ rb)
def fit(feats):
    Xm = np.stack(feats, 1) * sw[:, None]; sc = []; cs = []
    for r1, r2_ in ((ra, rb), (rb, ra)):
        co = np.linalg.lstsq(Xm, r1, rcond=None)[0]; f = Xm @ co; sc.append((2 * float(f @ r2_) - float(f @ f)) / E); cs.append(co)
    return round(float(np.mean(sc)), 4), [round(float(x), 3) for x in np.mean(cs, 0)]
proj = lambda x: x - ((w * x * iso).sum() / (w * iso * iso).sum()) * iso
print(json.dumps(dict(layer=l, weighted_size_vs_iso=dict(loc=round(float(np.sqrt((w * proj(loc) ** 2).sum() / (w * (kc) ** 2).sum())), 4), coll=round(float(np.sqrt((w * proj(coll) ** 2).sum() / (w * kc ** 2).sum())), 4),
                                                         collx=round(float(np.sqrt((w * proj(collx) ** 2).sum() / (w * kc ** 2).sum())), 4)),
                      corr=dict(loc_coll=round(float(np.corrcoef(proj(loc) * sw, proj(coll) * sw)[0, 1]), 3), coll_collx=round(float(np.corrcoef(proj(coll) * sw, proj(collx) * sw)[0, 1]), 3)),
                      explained=dict(iso_loc=fit([iso, loc]), iso_coll=fit([iso, coll]), iso_collx=fit([iso, collx]), iso_coll_young=fit([iso, coll_y]),
                                     iso_loc_coll=fit([iso, loc, coll]), iso_loc_collx=fit([iso, loc, collx])))))

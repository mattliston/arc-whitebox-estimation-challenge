#!/usr/bin/env python3
"""TAG opus-reexam E17: how much of the true K31 plane, K31[i,j] = kappa(z_i,z_i,z_i,z_j), do the exact hub coefficients explain?
(open loop, network <rank>; truth = 8M-sample planes with a second independent draw at layers 2, 5, 9, 12, 15)
  iso    v_i C_ij                                  (the card: radial v_i C_ij)
  row    (dk_i / v_i) C_ij, dk = K4L's per-unit correction (what K4L already gives the pair map through its row factor)
  hub    sum over pairs  A_ih Y_jh + B_ih Wt_jh,   A = 3 S1 Y^2 Wt + 3 S2 Y Wt^2 + S3 Wt^3,  B = S1 Y^3 + 3 S2 Y^2 Wt + 3 S3 Y Wt^2 + S4 Wt^3
         (two plane-sized products per cohort; by age group)
Rows = kink rows (|alpha| < 1.5), off-diagonal; coefficients fitted on draw 1, scored on draw 2 (R2 = 1 - |y2 - fit|^2 / |y2|^2; the
second draw's own noise bounds it: see noise_ceiling).
usage: e17_k31_local.py <rank> <layers>"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE / 'physicist'; P2 = HERE.parent
import numpy as np
rank = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(',')]
BI = np.load(PH / f't4births_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
PL = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz'); PL2 = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}_noise2.npz')
eye = np.eye(1024, dtype=bool)


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
    mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); al = mu / np.sqrt(v)
    hub = {'age1': 0.0, 'age2': 0.0, 'age34': 0.0, 'age5p': 0.0}; loc = {k: np.zeros(1024) for k in hub}
    for j in range(l):
        tp, tm, S = hub_data(j); k0 = stops[j]; K = (stops[j + 1] - k0) // 2; gap = (tp - tm)[None, :]
        Wt = (pp[:, k0:k0 + K] - pp[:, k0 + K:k0 + 2 * K]) / gap; Y = (tp[None, :] * pp[:, k0 + K:k0 + 2 * K] - tm[None, :] * pp[:, k0:k0 + K]) / gap
        S1, S2, S3, S4 = (x[None, :] for x in S)
        A = 3 * S1 * Y ** 2 * Wt + 3 * S2 * Y * Wt ** 2 + S3 * Wt ** 3; B = S1 * Y ** 3 + 3 * S2 * Y ** 2 * Wt + 3 * S3 * Y * Wt ** 2 + S4 * Wt ** 3
        g_ = 'age1' if l - j == 1 else ('age2' if l - j == 2 else ('age34' if l - j <= 4 else 'age5p'))
        hub[g_] = hub[g_] + A @ Y.T + B @ Wt.T; loc[g_] += (A * Y + B * Wt).sum(1)
    kink = np.abs(al) < 1.5; m = kink[:, None] & ~eye
    Kt = PL[f'K31_{l}'].astype(np.float64); Kt2 = PL2[f'K31_{l}'].astype(np.float64) if f'K31_{l}' in PL2.files else None
    iso = v[:, None] * C
    ori = (float(np.corrcoef(Kt[m], iso[m])[0, 1]), float(np.corrcoef(Kt.T[m], iso[m])[0, 1]))
    if ori[1] > ori[0]:
        Kt = Kt.T; Kt2 = Kt2.T if Kt2 is not None else None
    y1 = Kt[m]; y2 = Kt2[m] if Kt2 is not None else Kt[m]
    w4 = ((al * al - 1) * np.exp(-al * al / 2)) ** 2 + 1e-12; iso2 = v * v
    def dk(names):
        lc = sum(loc[n] for n in names); beta = (w4 * lc * iso2).sum() / (w4 * iso2 * iso2).sum(); return lc - beta * iso2
    feats = {'iso': iso[m]}
    for nm, names in (('a1', ['age1']), ('a12', ['age1', 'age2']), ('all', list(hub))):
        feats[f'row_{nm}'] = ((dk(names) / v)[:, None] * C)[m]; feats[f'hub_{nm}'] = sum(hub[n] for n in names)[m]
    def r2(keys, fixed=()):
        base1 = sum(feats[k] for k in fixed) if fixed else 0.0
        X = np.stack([feats[k] for k in keys], 1); co = np.linalg.lstsq(X, y1 - base1, rcond=None)[0]; fit = X @ co + base1
        return round(float(1 - ((y2 - fit) ** 2).sum() / (y2 ** 2).sum()), 4), [round(float(x), 3) for x in co]
    out = dict(layer=l, kink_rows=int(kink.sum()), second_draw=Kt2 is not None, orientation=[round(x, 3) for x in ori],
               noise_ceiling=round(float(1 - ((y2 - y1) ** 2).sum() / 2 / (y2 ** 2).sum()), 4) if Kt2 is not None else None,
               iso_coef_over_card_radial=None,
               R2=dict(iso=r2(['iso']), iso_row_a12=r2(['iso', 'row_a12']), iso_row_all=r2(['iso', 'row_all']),
                       iso_hub_a1=r2(['iso', 'hub_a1']), iso_hub_a12=r2(['iso', 'hub_a12']), iso_hub_all=r2(['iso', 'hub_all']),
                       iso_hub_a12_at_1=r2(['iso'], fixed=['hub_a12']), iso_hub_all_at_1=r2(['iso'], fixed=['hub_all']),
                       iso_row_a12_hub_a12=r2(['iso', 'row_a12', 'hub_a12'])),
               norms_vs_truth={k: round(float(np.linalg.norm(f_) / np.linalg.norm(y1)), 3) for k, f_ in feats.items()})
    print(json.dumps(out), flush=True)

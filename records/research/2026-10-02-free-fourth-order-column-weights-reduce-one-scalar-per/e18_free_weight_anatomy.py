#!/usr/bin/env python3
"""TAG opus-reexam E18: what are the physicist's FREE fourth-order column weights (t4_cp4_ceiling.py: one fitted weight per bank
column explains 0.49 / 0.60 of the true feed at layers 9 / 13, against 0.33 for the exact hub-local terms)? Are they predictable
from cheap per-column quantities?  ratio_r = lam4_free_r / lam_r regressed, with weights (lam_r |pp_r|^3)^2 (a proxy of the column's
plane energy), on: rule A; rule A by age; the pair's exact local coefficients turned into the two CP weights that reproduce S1 and S2
(c+ t+ + c- t- = S1, c+ t+^2 + c- t-^2 = S2) and S2 and S3; 1/t; hub alpha, alpha^2; the collective scalar c_r of E15b.
Also the sign-consistency of the free weights inside a pair (do + and - columns of one hub get related weights?).
usage: e18_free_weight_anatomy.py <rank> <layer>"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE / 'physicist'
import numpy as np
rank = int(sys.argv[1]); l = int(sys.argv[2])
BI = np.load(PH / f't4births_rank{rank:04d}.npz'); COL = np.load(PH / f't4cols_rank{rank:04d}_L{l}.npz'); FW = np.load(PH / f't4_lam4_rank{rank:04d}_L{l}_none.npz')
pp = COL['pp'].astype(np.float64); lam = COL['lam'].astype(np.float64); stops = [0] + [int(x) for x in COL['stops']]; lam4 = FW['lam4'].astype(np.float64)
n_cols = pp.shape[1]; feats = {k: np.zeros(n_cols) for k in ('A', 'cpS12', 'cpS23', 'inv_t', 'alpha', 'alpha2', 'S2overL', 'S3overL')}; age = np.zeros(n_cols, dtype=int); pair = np.full(n_cols, -1)
for j in range(l):
    b = lambda k: BI[f'{k}_{j}']
    s_ = b('sel'); a, s, gt, ph, m = (b(k)[s_].astype(np.float64) for k in ('alpha', 'sigma', 'gate', 'pdf', 'host'))
    tp, tm, coef = (b(k)[s_].astype(np.float64) for k in ('plus', 'minus', 'coef')); vv = s * s; dj = ph / s; gap = tp - tm
    r2 = vv * ((1 + a * a) * gt + a * ph)
    S1 = -a * ph / vv; S2 = 2 * (gt - gt * gt - m * dj); S3 = 3 * (1 - gt) * (r2 - 2 * m * m)
    k0 = stops[j]; K = (stops[j + 1] - k0) // 2; L = coef            # lam = +-coef (x s_new)
    # CP weights of the pair matching (S1, S2):  c+ t+ + c- t- = S1,  c+ t+^2 + c- t-^2 = S2
    cp12 = (S2 - tm * S1) / (tp * gap); cm12 = (S2 - tp * S1) / (tm * -gap)
    # matching (S2, S3):  c+ t+^2 + c- t-^2 = S2,  c+ t+^3 + c- t-^3 = S3
    cp23 = (S3 - tm * S2) / (tp * tp * gap); cm23 = (S3 - tp * S2) / (tm * tm * -gap)
    for sl, sgn, c12, c23, t in ((slice(k0, k0 + K), +1, cp12, cp23, tp), (slice(k0 + K, k0 + 2 * K), -1, cm12, cm23, tm)):
        feats['A'][sl] = S1 / (coef * gap); feats['cpS12'][sl] = c12 / (sgn * L); feats['cpS23'][sl] = c23 / (sgn * L)
        feats['inv_t'][sl] = 1 / np.where(np.abs(t) < 1e-6 * s, 1e-6 * s, t); feats['alpha'][sl] = a; feats['alpha2'][sl] = a * a
        feats['S2overL'][sl] = S2 / (sgn * L); feats['S3overL'][sl] = S3 / (sgn * L)
        age[sl] = l - j
    pair[k0:k0 + K] = np.arange(k0 + K, k0 + 2 * K); pair[k0 + K:k0 + 2 * K] = np.arange(k0, k0 + K)
ratio = lam4 / np.where(np.abs(lam) < 1e-30, 1e-30, lam)
wgt = (lam * (pp ** 2).sum(0) ** 1.5) ** 2; wgt = wgt / wgt.sum()
def r2(keys, mask=None):
    mk = np.ones(n_cols, bool) if mask is None else mask
    X = np.stack([feats[k][mk] for k in keys] + [np.ones(mk.sum())], 1) * np.sqrt(wgt[mk])[:, None]; y = ratio[mk] * np.sqrt(wgt[mk])
    co = np.linalg.lstsq(X, y, rcond=None)[0]; return round(float(1 - ((y - X @ co) ** 2).sum() / (y ** 2).sum()), 4), [round(float(x), 3) for x in co]
out = dict(layer=l, columns=n_cols, cv_of_free_fit=float(FW['cv']), ratio_weighted_rms=float(np.sqrt((wgt * ratio ** 2).sum())),
           A_weighted_rms=float(np.sqrt((wgt * feats['A'] ** 2).sum())),
           corr_ratio_A=float(np.corrcoef(ratio * np.sqrt(wgt), feats['A'] * np.sqrt(wgt))[0, 1]),
           pair_corr_of_ratio=float(np.corrcoef(ratio * np.sqrt(wgt), ratio[pair] * np.sqrt(wgt[pair]))[0, 1]),
           R2=dict(A=r2(['A']), cpS12=r2(['cpS12']), cpS23=r2(['cpS23']), A_cpS12_cpS23=r2(['A', 'cpS12', 'cpS23']),
                   all=r2(['A', 'cpS12', 'cpS23', 'inv_t', 'alpha', 'alpha2', 'S2overL', 'S3overL'])),
           R2_by_age={f'age{a_}': r2(['A', 'cpS12', 'cpS23', 'inv_t', 'alpha', 'alpha2', 'S2overL', 'S3overL'], age == a_)[0] for a_ in (1, 2, 3, 4) if (age == a_).any()},
           weight_share_by_age={f'age{a_}': round(float(wgt[age == a_].sum()), 3) for a_ in range(1, l + 1)},
           ratio_rms_by_age={f'age{a_}': round(float(np.sqrt((wgt[age == a_] * ratio[age == a_] ** 2).sum() / wgt[age == a_].sum())), 3) for a_ in range(1, l + 1)})
print(json.dumps(out))

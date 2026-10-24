#!/usr/bin/env python3
"""TAG opus-reexam E31b: is the second-chaos part of the unit-specific kurtosis a function of FREE per-unit quantities?
Gain-mode model z = (1 + eps)(mu + h): kappa4(z_i) = 12 s2 v_i^2 + 12 kappa3(eps) v_i mu_i^2 + kappa4(eps) mu_i^4, and
N = 4 s2 tr C C + 2 s2 (m Cm' + Cm m') + ... gives Q_i = a v_i + b mu_i (C mu)_i + c mu_i^2. Regress (cross-draw, weighted as E16/E20):
the card's per-unit k4 residual on those forms with and without Q, and Q's unit-specific part on them. usage: e31b_alpha_forms.py <rank>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; rank = int(sys.argv[1]); n = 1024
D = np.load(HERE / f'e31_rank{rank:04d}.npz'); A2 = np.load(HERE / f'analyst/a2_card_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
PU = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz'); PUi = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz')
for l in [int(x) for x in D['layers']]:
    ta = PU['mc_pre_k4'][l].astype(np.float64); tb = PUi['mc_pre_k4'][l].astype(np.float64); va = PU['mc_pre_v'][l].astype(np.float64); mu = PU['mc_pre_m'][l].astype(np.float64)
    C = MC[f'cov_pre_{l}'].astype(np.float64); Cmu = C @ mu; C2 = (C * C).sum(1)
    kc = A2[f'k4_{l}'].astype(np.float64); al = mu / np.sqrt(va); w = ((al * al - 1) * np.exp(-al * al / 2)) ** 2 + 1e-12; sw = np.sqrt(w); iso = va * va
    Q = (D[f'Q_{l}_0'] + D[f'Q_{l}_1']) / 2; Q0, Q1 = D[f'Q_{l}_0'], D[f'Q_{l}_1']
    ra, rb = (ta - kc) * sw, (tb - kc) * sw; E = float(ra @ rb)
    def fit(feats):
        Xm = np.stack(feats, 1) * sw[:, None]; sc = []
        for r1, r2_ in ((ra, rb), (rb, ra)):
            co = np.linalg.lstsq(Xm, r1, rcond=None)[0]; f = Xm @ co; sc.append((2 * float(f @ r2_) - float(f @ f)) / E)
        return round(float(np.mean(sc)), 4)
    def r2q(feats):        # share of Q's weighted energy beyond `v` explained by the forms, fitted on half 0 and scored on half 1
        Xm = np.stack(feats, 1) * sw[:, None]; base = np.stack([va], 1) * sw[:, None]
        res = lambda y: y * sw - base @ np.linalg.lstsq(base, y * sw, rcond=None)[0]
        y0, y1 = res(Q0), res(Q1); Xr = Xm - base @ np.linalg.lstsq(base, Xm, rcond=None)[0]
        f = Xr @ np.linalg.lstsq(Xr, y0, rcond=None)[0]
        return round(float((2 * f @ y1 - f @ f) / (y0 @ y1)), 4)
    free = [iso, va, va * mu * mu, mu ** 4, mu * Cmu, mu * mu, C2, va * mu, mu ** 3, mu]
    print(json.dumps(dict(layer=l, explained=dict(iso_v=fit([iso, va]), alpha_forms=fit([iso, va, va * mu * mu, mu ** 4]), free_forms=fit(free),
                                                  iso_v_Q=fit([iso, va, Q]), free_forms_Q=fit(free + [Q])),
                          Q_unit_specific_explained_by=dict(mu2=r2q([mu * mu]), mu_Cmu=r2q([mu * Cmu]), mu2_muCmu_C2=r2q([mu * mu, mu * Cmu, C2]), all_free=r2q(free[2:])))))

#!/usr/bin/env python3
"""TAG opus-reexam E31 analysis: share of the unit-specific per-unit fourth cumulant explained by the second-chaos proxy Q_i (and Qm_i).
Truth: two independent 32M-sample draws of the per-unit pre-activation fourth cumulant (cross-draw scoring as in E16/E20, weights = the
kurtosis weight of the mean formula). usage: e31_analyze.py <rank>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; rank = int(sys.argv[1]); n = 1024
D = np.load(HERE / f'e31_rank{rank:04d}.npz'); A2 = np.load(HERE / f'analyst/a2_card_rank{rank:04d}.npz')
PU = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz'); PUi = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz')
print('samples per half', int(D['N']))
for l in [int(x) for x in D['layers']]:
    ta = PU['mc_pre_k4'][l].astype(np.float64); tb = PUi['mc_pre_k4'][l].astype(np.float64); va = PU['mc_pre_v'][l].astype(np.float64); mu = PU['mc_pre_m'][l].astype(np.float64)
    kc = A2[f'k4_{l}'].astype(np.float64); al = mu / np.sqrt(va); w = ((al * al - 1) * np.exp(-al * al / 2)) ** 2 + 1e-12; sw = np.sqrt(w); iso = va * va
    Q0, Q1 = D[f'Q_{l}_0'], D[f'Q_{l}_1']; Qm0, Qm1 = D[f'Qm_{l}_0'], D[f'Qm_{l}_1']; Q = (Q0 + Q1) / 2; Qm = (Qm0 + Qm1) / 2
    # same-sample sanity: the MC's own k4 against the 32M truth
    k4mc = (D[f'm4_{l}_0'] + D[f'm4_{l}_1']) / 2 - 3 * ((D[f'v_{l}_0'] + D[f'v_{l}_1']) / 2) ** 2
    proj = lambda x, basis: x - basis @ np.linalg.lstsq(basis * sw[:, None], x * sw, rcond=None)[0]
    Bv = np.stack([iso], 1); Bq = np.stack([va * np.mean(va) * n, va], 1)      # isotropic forms: k4 ~ v^2 ; Q ~ v tr C (and v)
    rQ0, rQ1 = proj(Q0, np.stack([va], 1)), proj(Q1, np.stack([va], 1))
    rel = float(np.corrcoef(rQ0 * sw, rQ1 * sw)[0, 1])
    ra, rb = (ta - kc) * sw, (tb - kc) * sw; E = float(ra @ rb)
    def fit(feats):
        Xm = np.stack(feats, 1) * sw[:, None]; sc = []; cs = []
        for r1, r2_ in ((ra, rb), (rb, ra)):
            co = np.linalg.lstsq(Xm, r1, rcond=None)[0]; f = Xm @ co; sc.append((2 * float(f @ r2_) - float(f @ f)) / E); cs.append(co)
        return round(float(np.mean(sc)), 4), [float('%.4g' % x) for x in np.mean(cs, 0)]
    # unit-specific parts, plain correlation
    tk = proj((ta + tb) / 2, Bv); tq = proj(Q, np.stack([va], 1)); tqm = proj(Qm, np.stack([va], 1))
    print(json.dumps(dict(layer=l, mean_excess_kurtosis=round(float(np.mean((ta + tb) / 2 / iso)), 4), spread=round(float(np.std((ta + tb) / 2 / iso)), 4),
                          Q_over_v_trC=round(float(np.mean(Q / (va * va.sum()))), 5), Q_unit_specific_reliability_between_halves=round(rel, 3),
                          corr_unit_specific_k4_vs_Q=round(float(np.corrcoef(tk * sw, tq * sw)[0, 1]), 3), corr_vs_Qm=round(float(np.corrcoef(tk * sw, tqm * sw)[0, 1]), 3),
                          slope_k4_on_Q_x_n=round(float((tk * w * tq).sum() / (w * tq * tq).sum()) * n, 3),
                          explained=dict(iso=fit([iso]), iso_Q=fit([iso, Q]), iso_v_Q=fit([iso, va, Q]), iso_v_Q_Qm=fit([iso, va, Q, Qm])),
                          mc_k4_vs_truth_corr=round(float(np.corrcoef(proj(k4mc, Bv) * sw, tk * sw)[0, 1]), 3))))

#!/usr/bin/env python3
"""TAG opus-reexam E32b (network 35, layers 12 -> 13, numpy): what is the third of the network-specific norm-coupling matrix that the transport
law leaves unexplained? Candidate sources in post space of layer 12, pushed with W13 (factor 2 from S = 2 R):
  gauss   G C diag(S2) C G                              (Gaussian births, hub with two legs)
  diag    diag(S4)                                      (each unit's own fourth cumulant)
  conv    sum_r lam_r (pp_r(12) . beta) (g pp_r(12))(g pp_r(12))',  beta = 2 m (1 - g)    (conversion from the card's third-order bank)
  hubS1   Sym[ S1_a (sum_c g_c^2 C_ac^2) g_b C_ab ]     (hub with three legs, two of them closed on the norm)
Scores as in e32_analyze.py (c): cross-half, network-specific parts (isotropic and mean forms projected out)."""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import pcard as pc
rank = 35; n = 1024; off = ~np.eye(n, dtype=bool)
D = np.load(HERE / f'e32_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); BI = np.load(HERE / f'physicist/t4births_rank{rank:04d}.npz')
C12 = MC['cov_pre_12'].astype(np.float64); C13 = MC['cov_pre_13'].astype(np.float64); mu13 = MC['mean_pre_13'].astype(np.float64); mu12 = MC['mean_pre_12'].astype(np.float64)
N12 = [(lambda x: (x + x.T) / 2)(D[f'M_12_{h}'] - 2 * C12 @ C12) for h in (0, 1)]; N13 = [(lambda x: (x + x.T) / 2)(D[f'M_13_{h}'] - 2 * C13 @ C13) for h in (0, 1)]
W = pc.weights(rank)[13].astype(np.float64)
v = np.diag(C12); sd = np.sqrt(v); al = mu12 / sd; pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd
m = MC['mean_post_12'].astype(np.float64)
r2 = v * ((1 + al * al) * g + al * pdf); r3 = v * sd * ((al ** 3 + 3 * al) * g + (al * al + 2) * pdf); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * pdf)
S1 = -al * pdf / v; S2 = 2 * (g - g * g - m * d); S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2
GC = g[:, None] * C12
push = lambda A: 2 * W.T @ A @ W
F = dict(gauss=push((GC * S2[None, :]) @ GC.T), diag=2 * W.T @ (S4[:, None] * W))
row = S1 * ((g * g)[None, :] * C12 * C12).sum(1); H = (row[:, None] * C12) * g[None, :]; F['hubS1'] = push(H + H.T)
beta_t = 2 * m * (1 - g); Cb = C12 @ beta_t
rowform = lambda phi: push((lambda A: A + A.T)((phi[:, None] * C12) * g[None, :]))
F['R1'] = rowform(d * Cb)                      # quadratic vertex at a, its second leg on the linear response of the norm: d_a (C beta)_a g_b C_ab + sym
F['Rmu'] = rowform(d * mu12)                   # the same with C beta replaced by the mean (top covariance mode)
F['Rg'] = rowform(g * (1 - g))                 # generic row forms of the gate
F['Rd'] = rowform(d * sd)
c12 = np.load(HERE / f'e15cols_rank{rank:04d}_L12.npz'); c13 = np.load(HERE / f'physicist/t4cols_rank{rank:04d}_L13.npz')
pp12 = c12['pp'].astype(np.float64); lam12 = c12['lam'].astype(np.float64); nold = pp12.shape[1]; pp13 = c13['pp'][:, :nold].astype(np.float64)
gc, mc_ = BI['gate_12'].astype(np.float64), BI['host_12'].astype(np.float64); beta = 2 * mc_ * (1 - gc)
cw = lam12 * (pp12.T @ beta)
F['conv'] = 2 * (pp13 * cw[None, :]) @ pp13.T
ages = np.zeros(nold, dtype=int); stops = [0] + [int(x) for x in c12['stops']]
for j in range(len(stops) - 1): ages[stops[j]:stops[j + 1]] = 12 - j
young = ages <= 4; F['conv_young'] = 2 * (pp13[:, young] * cw[young][None, :]) @ pp13[:, young].T
iso = dict(trC_C=np.trace(C13) * C13, C2=C13 @ C13, mumu=np.outer(mu13, mu13)); Cm = C13 @ mu13; iso['muCmu'] = np.outer(mu13, Cm) + np.outer(Cm, mu13)
Xi = np.stack([iso[k][off] for k in iso], 1); pr = lambda a: a - Xi @ np.linalg.lstsq(Xi, a, rcond=None)[0]
src = {k: pr(F[k][off]) for k in F}
print(json.dumps(dict(columns=int(nold), young=int(young.sum()), sizes_rms={k: float('%.3g' % np.sqrt((x ** 2).mean())) for k, x in src.items()})))
for h, hp in ((0, 1), (1, 0)):
    T = pr((W.T @ ((g[:, None] * N12[hp]) * g[None, :]) @ W)[off]); y, y2 = pr(N13[h][off]), pr(N13[hp][off]); out = dict(target_half=h, target_rms=float('%.3g' % np.sqrt((y ** 2).mean())))
    for name, cols in (('T', ['T']), ('T_gauss_diag', ['T', 'gauss', 'diag']), ('T_gauss_diag_conv', ['T', 'gauss', 'diag', 'conv']), ('T_gauss_diag_convyoung', ['T', 'gauss', 'diag', 'conv_young']),
                       ('all', ['T', 'gauss', 'diag', 'conv', 'hubS1']), ('all_R1', ['T', 'gauss', 'diag', 'conv', 'hubS1', 'R1']), ('all_rowforms', ['T', 'gauss', 'diag', 'conv', 'hubS1', 'R1', 'Rmu', 'Rg', 'Rd']), ('sources_only', ['gauss', 'diag', 'conv', 'hubS1', 'R1'])):
        X = np.stack([T if k == 'T' else src[k] for k in cols], 1); co = np.linalg.lstsq(X, y, rcond=None)[0]; f = X @ co
        out[name] = dict(share=round(float((2 * f @ y2 - f @ f) / (y @ y2)), 4), coef=[float('%.3g' % x) for x in co])
    res = y - T * out['T']['coef'][0]; res2 = y2 - T * out['T']['coef'][0]
    out['corr_of_unexplained_with'] = {k: round(float(np.corrcoef(src[k], res)[0, 1]), 3) for k in src}
    print(json.dumps(out))

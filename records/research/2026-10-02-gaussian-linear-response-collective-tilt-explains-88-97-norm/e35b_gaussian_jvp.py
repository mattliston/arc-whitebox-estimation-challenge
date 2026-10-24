#!/usr/bin/env python3
"""TAG opus-reexam E35b (network 35, layer 12): is the ReLU step of the norm-coupling matrix the EXACT derivative of the Gaussian pair map?
Tilt of the layer's distribution by its own post norm R: pre mean shifts by eps t (t = E[z~ R~]), pre covariance by eps P (P = E[z~ z~' R~], Mmix).
Prediction: Mpost = E[x~ x~' R~] = d/d eps PairMapGauss(mu + eps t, C + eps P), by central differences with the card's own Gaussian pair function
(series to order 6 in the correlation). Scored as E35: cross-half share of the network-specific off-diagonal energy of N_post = Mpost - 2 Cpost^2
explained (the same Gaussian pairing removed from the prediction), next to the gate product G (Mmix - 2 C gamma C) G of E35 (0.70)."""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
pc.set_threads(3); rank = 35; l = 12; n = 1024; off = ~np.eye(n, dtype=bool)
D = np.load(HERE / f'e35_rank{rank:04d}_L{l}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
f = lambda k: MC[k].astype(np.float64); sym = lambda x: (x + x.T) / 2
C, mu, Cp, mp = f(f'cov_pre_{l}'), f(f'mean_pre_{l}'), f(f'cov_post_{l}'), f(f'mean_post_{l}')
cd = pc.build(rank, learned=True); pair = cd.env['_nonlinear_first_gaussian']; T = lambda x: torch.from_numpy(np.ascontiguousarray(x))
Z = torch.zeros((n, n), dtype=torch.float64); z1 = torch.zeros(n, dtype=torch.float64)
def PM(mu_, C_):
    out = pair(T(mu_), T(C_), Z, z1, None, 'covariance')
    return out[0].numpy().astype(np.float64), out[1].numpy().astype(np.float64)       # host mean, post covariance
m0, C0 = PM(mu, C)
print(json.dumps(dict(gaussian_pair_map_vs_truth=dict(mean_rel=float(np.linalg.norm(m0 - mp) / np.linalg.norm(mp)), cov_offdiag_rel=float(np.linalg.norm((C0 - Cp)[off]) / np.linalg.norm(Cp[off]))))))
v = np.diag(C); sd = np.sqrt(v); al = mu / sd; pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd; gam = g - mp * d
Cm = Cp @ mp; Fpost = np.stack([(np.trace(Cp) * Cp)[off], (Cp @ Cp)[off], np.outer(mp, mp)[off], (np.outer(mp, Cm) + np.outer(Cm, mp))[off]], 1)
ns = lambda A: A[off] - Fpost @ np.linalg.lstsq(Fpost, A[off], rcond=None)[0]
eps = 2e-6; pred = []; gate = []
for h in (0, 1):
    P = sym(D[f'Mmix_{h}']); t = D[f'tz_{h}']
    mp_, Cplus = PM(mu + eps * t, C + eps * P); mm_, Cminus = PM(mu - eps * t, C - eps * P)
    dC = (Cplus - Cminus) / (2 * eps); dm = (mp_ - mm_) / (2 * eps)
    # E[x~ x~' R~] = d/d eps E_eps[(x - m)(x - m)'] at fixed centre m = d/d eps Cov_eps (the mean-shift term is second order)
    pred.append(dC - 2 * Cp @ Cp)
    gate.append((g[:, None] * (P - 2 * (C * gam[None, :]) @ C)) * g[None, :])
Npost = [sym(D[f'Mpost_{h}']) - 2 * Cp @ Cp for h in (0, 1)]
def score(preds):
    out = []
    for h, hp in ((0, 1), (1, 0)):
        y, y2 = ns(Npost[h]), ns(Npost[hp]); x = ns(preds[hp]); c = float(x @ y / (x @ x)); out.append((round(float((2 * c * x @ y2 - c * c * x @ x) / (y @ y2)), 4), round(c, 3), round(float((2 * x @ y2 - x @ x) / (y @ y2)), 4)))
    return dict(share_fitted=[o[0] for o in out], coef=[o[1] for o in out], share_at_coefficient_1=[o[2] for o in out])
print(json.dumps(dict(gaussian_jvp=score(pred), gate_product=score(gate))))

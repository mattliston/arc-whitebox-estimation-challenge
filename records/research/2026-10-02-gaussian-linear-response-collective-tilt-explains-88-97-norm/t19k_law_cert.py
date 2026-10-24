#!/usr/bin/env python3
"""TAG opus-physicist T19k (numpy): CERTIFICATE of the one-layer law for the second-chaos matrix at layer 0 -> 1, where the inputs of the
ReLU are exactly Gaussian (z0 = x W0, x standard normal: mean 0, C0 = W0' W0), so tree level with the Gaussian terms must be all there is.
   Ntilde(1) = 2 W1' N W1,   N_ab = sum_c kappa4(x_a, x_b, x_c, x_c)   (x = relu z0),  tree level:
   a != b:  N_ab = g_a g_b (C S2 C)_ab [c != a, b]                     Gaussian births,  S2 = 2 (gamma - g^2), gamma = g - m d
                 + C_ab [d_a g_b (C V)_a + g_a d_b (C V)_b]            row forms,        V = E phi' = 2 m (1 - g)
                 + C_ab (g_b S3_a + g_a S3_b)                          the units' own (3,1) cumulant: S3 = d kappa3_post / d mu
   a == b:  N_aa = S4_a + V_a (C V)_a - V_a^2 v_a + sum_{c != a} (2 gamma_a gamma_c - 2 g_a^2 g_c^2) C_ac^2
Compared with the Monte Carlo Ntilde(1) (t19g, two 4M blocks): share of its energy explained (cross-block), for the whole matrix and for
the network-specific part (the lead's forms removed), term by term. usage: t19k_law_cert.py <rank>"""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; n = 1024; off = ~np.eye(n, dtype=bool); dg = np.eye(n, dtype=bool)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rank = int(sys.argv[1]); man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r'); Z = np.load(HERE / f't19g_rank{rank:04d}.npz')
w0 = W[0].astype(np.float64); W1 = W[1].astype(np.float64); C = w0.T @ w0; v = np.diag(C).copy(); s = np.sqrt(v); ph = 1 / math.sqrt(2 * math.pi)
g = np.full(n, 0.5); d = ph / s; m = s * ph; gam = g - m * d; S2 = 2 * (gam - g * g); V = 2 * m * (1 - g)
r2 = v * 0.5; r3 = v * s * 2 * ph; r4 = 1.5 * v * v; S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2; S3 = 3 * (1 - g) * (r2 - 2 * m * m)
Co = C - np.diag(v); CV = Co @ V + v * V
terms = {}
G_ = (g[:, None] * Co * S2[None, :]) @ (Co * g[None, :]); np.fill_diagonal(G_, 0.0)           # sum over c != a, b (Co has zero diagonal)
terms['Gaussian births'] = G_
terms['row forms'] = Co * (d * (Co @ V + 0 * v))[:, None] * g[None, :] + Co * g[:, None] * (d * (Co @ V))[None, :]
terms['own (3,1)'] = Co * (S3[:, None] * g[None, :] + g[:, None] * S3[None, :])
terms['diagonal: own S4'] = np.diag(S4)
terms['diagonal: V (C V) and (2,2) sums'] = np.diag(V * (Co @ V) + ((2 * gam[:, None] * gam[None, :] - 2 * (g * g)[:, None] * (g * g)[None, :]) * Co * Co).sum(1))
push = lambda X: 2 * (W1.T @ X @ W1)
Nb = []
for b in (0, 1):
    M11 = Z[f'M11_L1_b{b}'].astype(np.float64); x = Z[f'T_L1_b{b}'].astype(np.float64) - M11 * float(Z[f'S_L1_b{b}']) - 2 * M11 @ M11; Nb.append((x + x.T) / 2)
C1 = (Z['M11_L1_b0'].astype(np.float64) + Z['M11_L1_b1'].astype(np.float64)) / 2; mu1 = Z['mu_L1'].astype(np.float64); C2 = C1 @ C1; Cmu = C1 @ mu1
forms = dict(trC_C=np.trace(C1) * C1, C2=C2, mumu=np.outer(mu1, mu1), muCmu=np.outer(mu1, Cmu) + np.outer(Cmu, mu1), eye=np.eye(n))
def specific(N):
    R = N.copy()
    for mask, names in ((off, ('trC_C', 'C2', 'mumu', 'muCmu')), (dg, ('trC_C', 'C2', 'mumu', 'eye'))):
        X = np.stack([forms[k][mask] for k in names], 1); R[mask] = N[mask] - X @ np.linalg.lstsq(X, N[mask], rcond=None)[0]
    return R
share = lambda P, T0, T1, mask: round(float((2 * (P[mask] * T1[mask]).sum() - (P[mask] * P[mask]).sum()) / (T0[mask] * T1[mask]).sum()), 4)
full = np.ones((n, n), dtype=bool); Rb = [specific(x) for x in Nb]
print(json.dumps(dict(rank=rank, monte_carlo=dict(block_agreement_full=round(float((Nb[0] * Nb[1]).sum() / np.sqrt((Nb[0] ** 2).sum() * (Nb[1] ** 2).sum())), 4),
                                                  block_agreement_network_specific=round(float((Rb[0] * Rb[1]).sum() / np.sqrt((Rb[0] ** 2).sum() * (Rb[1] ** 2).sum())), 4),
                                                  network_specific_share_of_energy=round(float((Rb[0] * Rb[1]).sum() / (Nb[0] * Nb[1]).sum()), 4)))))
acc = np.zeros((n, n))
for name, X in terms.items():
    acc = acc + X; P = push(acc); Pr = specific(P); P1 = push(X); P1r = specific(P1)
    print(json.dumps({'cumulative up to': name, 'whole matrix: share explained (nothing fitted)': share(P, Nb[0], Nb[1], full), 'network-specific part': share(Pr, Rb[0], Rb[1], full),
                      'off-diagonal of the network-specific part': share(Pr, Rb[0], Rb[1], off), 'this term alone, network-specific': share(P1r, Rb[0], Rb[1], full),
                      'fitted coefficient of this term alone': round(float(((P1r * (Rb[0] + Rb[1]) / 2).sum()) / (P1r * P1r).sum()), 3)}), flush=True)

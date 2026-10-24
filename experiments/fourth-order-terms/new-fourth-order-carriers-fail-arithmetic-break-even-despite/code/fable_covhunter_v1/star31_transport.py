#!/usr/bin/env python3
"""TAG fable-covhunter (CH-41): TRANSPORT TEST of a plane-only fourth-order carrier (the (3,1)+(2,2) analogue of the card's (2,1) star),
with TRUE inputs, r35. The 8M post-activation planes at layer l (public_g19 mps_planes_rank0035_post*.npz: K31, K22, per-unit k4 on the
K31 diagonal) are pushed through W_{l+1} assuming kappa4(y) lives only on the index patterns (a,a,a,a), (a,a,a,c), (a,a,c,c):
  K31'[b,d] = sum_a k4_a W_ab^3 W_ad + sum_{a!=c} K31[a,c] (W_ab^3 W_cd + 3 W_ab^2 W_cb W_ad) + 3 sum_{a!=c} K22[a,c] W_ab^2 W_cb W_cd
(about 5-6 dense products per layer). Graded against the TRUE pre-activation K31 at l+1 (8M, mps_planes_rank0035*.npz) in the
covariance-read metric at l+1: weight |alpha_b| pdf_b g_d / v_b on the off-diagonal (the physicist's r=0 K31 read weight), true gates from
the 32M mcall truth. Compared with the card-form isotropic K31 = t4 rho (k4_b/v_b C_bd with the true k4, C) and with the noise floor.
usage: star31_transport.py"""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; G19 = P2 / 'public_g19_attribution_v1'
import sys; sys.path.insert(0, str(HERE)); import covstep as cs
W = cs.weights(35).astype(np.float64); mc = np.load(HERE / 'mcall_rank0035_P32M.npz'); TR = np.load(G19 / 'mps_truth_rank0035_P32M.npz')
post = {}
for f in ('mps_planes_rank0035_post.npz',):
    d = np.load(G19 / f)
    for k in d.files:
        if k != 'layers': post[k] = d[k]
pre = {}
for f in ('mps_planes_rank0035.npz', 'mps_planes_rank0035_new.npz'):
    d = np.load(G19 / f)
    for k in d.files:
        if k != 'layers' and k not in pre: pre[k] = d[k]
n = 1024; off = ~np.eye(n, dtype=bool)
for l in (5, 8, 12, 13):
    if f'K31_{l}' not in post or f'K31_{l + 1}' not in pre: continue
    K31y, K22y = post[f'K31_{l}'].astype(np.float64), post[f'K22_{l}'].astype(np.float64)
    k4y = np.diag(K31y).copy(); K31o = K31y.copy(); np.fill_diagonal(K31o, 0.); K22o = K22y.copy(); np.fill_diagonal(K22o, 0.)
    w = W[l + 1]; w2 = w * w; w3 = w2 * w
    M = K31o @ w
    pred = w3.T @ (k4y[:, None] * w) + w3.T @ M + 3 * ((w2 * M).T @ w) + 3 * (((w2.T @ K22o) * w.T) @ w)
    true = pre[f'K31_{l + 1}'].astype(np.float64)
    mu = mc[f'mean_pre_{l + 1}']; v = np.diag(mc[f'cov_pre_{l + 1}']); s = np.sqrt(v); al = mu / s
    pdf = np.exp(-al ** 2 / 2) / math.sqrt(2 * math.pi); g = 0.5 * np.array([math.erfc(-x / math.sqrt(2)) for x in al])
    wt = (np.abs(al) * pdf / v)[:, None] * g[None, :]
    iso = (TR['mc_pre_k4'][l + 1] / v)[:, None] * mc[f'cov_pre_{l + 1}']          # card form K31_bd = k4_b C_bd / v_b with the TRUE k4, C
    err = lambda X: float(np.linalg.norm((wt * (X - true))[off]) / np.linalg.norm((wt * true)[off]))
    cosr = lambda X: float(((wt * X)[off] @ (wt * true)[off]) / np.linalg.norm((wt * X)[off]) / np.linalg.norm((wt * true)[off]))
    # best scale of the transported plane (read metric) and its residual
    a_ = float(((wt * pred)[off] @ (wt * true)[off]) / ((wt * pred)[off] @ (wt * pred)[off]))
    print(f'L{l}->L{l + 1}: read-weighted rel error: plane-star transport {err(pred):.3f} (cos {cosr(pred):.3f}, best scale {a_:.2f}, scaled err {err(a_ * pred):.3f})'
          f' | isotropic card form (true k4, C) {err(iso):.3f} (cos {cosr(iso):.3f}) | unweighted star err {np.linalg.norm((pred - true)[off]) / np.linalg.norm(true[off]):.3f}', flush=True)

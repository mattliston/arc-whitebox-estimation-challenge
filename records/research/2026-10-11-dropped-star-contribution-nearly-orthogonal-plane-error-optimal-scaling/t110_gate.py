#!/usr/bin/env python3
"""TAG opus-physicist T110 gate (lead 10-11): HOW MUCH OF THE K21 ERROR TWO LAYERS LATER IS THE STAR'S DROPPED HUB PART?
The star made at layer l carries the post plane's residual P (pair form: T(a,a,b) = P_ab; P = star' / 3 off the diagonal, star's diagonal on it). Pushed
into layer l+1 (weights W1) the pair tensor becomes T' = sum_a Sym(w_a, w_a, u_a) - 2 sum_a P_aa w_a^3 (w_a = row a of W1, u_a = (P W1)_a): a HUB form whose
plane the card keeps (by its star push) and whose triple-distinct part it drops. Transported through layer l+1 gate-linearly (Phi = gate) and pushed
into l+2 (W2) the full tensor gives the plane
  H_exact = (X o X) Y' + 2 (X o Y) X' - 2 (X o X) diag(Pd) X',  X = W2' diag(Phi) W1', Y = W2' diag(Phi) U' (U = P W1);
the pair closure of the l+1 plane Q = Phi^2 Phi T'_plane (T'_plane = (W1 o W1)' U + 2 (W1 o U)' W1 - 2 (W1 o W1)' diag(Pd) W1) pushed with both pair terms gives
  H_pair = (W2 o W2)' Q W2 + 2 (W2 o (Q W2))' W2 - 2 (W2 o W2)' diag(Qd) W2.
D = H_exact - H_pair is the dropped hub part's contribution to the plane at l+2. Compared, covariance-weighted (pdf_i g_j / s_i)^2 at l+2, with the card's
K21 error E = K21_true - K21_card at l+2 (8M-sample planes, networks 35 / 100): |D| / |E| and the share of |E|^2 the best multiple of D removes.
usage: t110_gate.py"""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
erf = np.vectorize(math.erf)
for r in (35, 100):
    D = np.load(HERE / 't110_stardump' / 'dumps' / f'D_{r}.npz'); T = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{r:04d}.npz')
    for l in (10, 11, 12):
        A = D[f'star_into_{l + 1}'].astype(np.float64); W1 = D[f'W_{l + 1}'].astype(np.float64); W2 = D[f'W_{l + 2}'].astype(np.float64)
        Phi = D[f'gate_{l + 1}'].astype(np.float64)
        P = A.T / 3.0; np.fill_diagonal(P, np.diag(A)); Pd = np.diag(P)
        U = P @ W1
        X = W2.T @ (Phi[:, None] * W1.T); Y = W2.T @ (Phi[:, None] * U.T)
        H_exact = (X * X) @ Y.T + 2 * (X * Y) @ X.T - 2 * ((X * X) * Pd[None, :]) @ X.T
        Tpl = (W1 * W1).T @ U + 2 * (W1 * U).T @ W1 - 2 * ((W1 * W1) * Pd[:, None]).T @ W1
        Q = (Phi ** 2)[:, None] * Phi[None, :] * Tpl; Qd = np.diag(Q)
        H_pair = (W2 * W2).T @ Q @ W2 + 2 * (W2 * (Q @ W2)).T @ W2 - 2 * ((W2 * W2) * Qd[:, None]).T @ W2
        Dm = H_exact - H_pair; np.fill_diagonal(Dm, 0.0)
        mu, C = D[f'mu_{l + 2}'].astype(np.float64), D[f'C_{l + 2}'].astype(np.float64); s = np.sqrt(np.diag(C).clip(1e-12)); a = mu / s
        pdf = np.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = 0.5 * (1 + erf(a / math.sqrt(2)))
        w = ((pdf / s)[:, None] * g[None, :]) ** 2; np.fill_diagonal(w, 0.0)
        E = T[f'K21_{l + 2}'].astype(np.float64) - D[f'K21_{l + 2}'].astype(np.float64); np.fill_diagonal(E, 0.0)
        nE, nD, ip = math.sqrt((E * E * w).sum()), math.sqrt((Dm * Dm * w).sum()), (E * Dm * w).sum()
        Hn = math.sqrt((H_exact * H_exact * w).sum())
        print('net %d, star of layer %d -> plane of layer %d: |dropped hub part| / |K21 error| = %.3f; best multiple %.2f removes %.1f%% of the error energy; '
              '(|exact push of the star through two layers| / |error| = %.3f)' % (r, l, l + 2, nD / nE, ip / max(nD * nD, 1e-300), 100 * ip * ip / max(nD * nD * nE * nE, 1e-300), Hn / nE), flush=True)

#!/usr/bin/env python3
"""Certificate: the card's plane-only residual star pushed by _push_cubic
equals the CP read of the appended (p_i=e_i, q_i=R_off[i]+R_ii/3 e_i,
lambda=1) terms after one W-push (T_iij = R_ij, T_iii = R_ii). n=96, f64.
Both formulas transcribed from estimator_cap.py:1134-1156 and :871-891."""
import numpy as np
rng = np.random.default_rng(7)
n = 96
W = rng.standard_normal((n, n)) / np.sqrt(n)     # weight (in, out) as the card uses W[l]
R = rng.standard_normal((n, n)); R = 0.5 * (R + R.T)
diag = np.diag(R).copy()
D = np.eye(n, dtype=bool)
# --- _push_cubic (card) -------------------------------------------------
W2 = W * W
directed = np.where(D, 0.0, R)
transported = directed @ W
pushed_diag = np.sum(diag[:, None] * W2 * W + 3.0 * W2 * transported, axis=0)
pushed_k21 = (W2.T @ (diag[:, None] * W) + 2.0 * (W * transported).T @ W + W2.T @ transported)
star = pushed_k21.copy(); star[D] = pushed_diag
# --- CP representation + push + _cp_read ---------------------------------
q_rows = np.where(D, (diag / 3.0)[:, None], R)         # q_i = row i
P = np.eye(n); Q = q_rows.T                            # columns = terms
Pp = W.T @ P; Qp = W.T @ Q                             # pushed p, q
lam = np.ones(n)
first = (Pp * Pp) @ (Qp * lam[None, :]).T              # sum_r p_a^2 q_b lam
second = (Pp * Qp) @ (Pp * lam[None, :]).T             # sum_r p_a q_a p_b lam
cp = first + 2.0 * second
rel = np.linalg.norm(cp - star) / np.linalg.norm(star)
rel_diag = np.linalg.norm(np.diag(cp) - np.diag(star)) / np.linalg.norm(np.diag(star))
print({"rel_offdiag_and_full": rel, "rel_diag": rel_diag, "ok": bool(rel < 1e-12 and rel_diag < 1e-12)})

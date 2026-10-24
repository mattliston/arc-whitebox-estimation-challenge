#!/usr/bin/env python3
"""TAG fable-physicist (OT-1): is the old tier's third-order content a Sym(C x x) (gain-type) family? From the bank dump
(fable_oldtier_economics_v1/bank_r35/layer_LL.npz pp, lam, old_columns; state_LL.npz Cpre) form the OLD plane
K21_old[a,c] = sum_{j<old} lam_j pp_aj^2 pp_cj (full 1024 x 1024, diagonal = old per-unit k3) and fit it with
  (i) rank-1 v x^T, (ii) Sym(Cpre x x): v_a x_c + 2 C_ac x_a (exact normal equations, n unknowns), (iii) (ii) + a second free vector
  on the gated-covariance family Sym(Cg x y), Cg = W^T diag(g^2 v) W style is NOT available -> use Cpre only.
Reports relative Frobenius error on response x live (what the card reads) and on the diagonal. numpy only. usage: old_sym_fit.py <layers>"""
import sys, json
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parents[2] / 'fable_oldtier_economics_v1' / 'bank_r35'
for l in [int(x) for x in sys.argv[1].split(',')]:
    z = np.load(D / f'layer_{l:02d}.npz'); s = np.load(D / f'state_{l:02d}.npz')
    no = int(z['old_columns']); P = z['pp'][:, :no].astype(np.float64); lam = z['lam'][:no].astype(np.float64)
    resp, live = z['response'], z['live']
    K = (P * P) @ (lam[:, None] * P.T)                       # full old plane incl. diagonal
    C = s['Cpre'].astype(np.float64); v = np.diag(C).copy()
    def err(M):
        R = (K - M)[np.ix_(resp, live)]; T = K[np.ix_(resp, live)]
        return round(float(np.linalg.norm(R) / np.linalg.norm(T)), 4), round(float(np.linalg.norm(np.diag(K - M)) / np.linalg.norm(np.diag(K))), 4)
    # (i) rank-1 best: SVD
    U, S, Vt = np.linalg.svd(K); r1 = S[0] * np.outer(U[:, 0], Vt[0])
    # (ii) Sym(C x x): normal equations N x = L^T K
    N = (v @ v) * np.eye(1024) + 2 * C * v[None, :] + 2 * v[:, None] * C + 4 * np.diag((C * C).sum(1))
    rhs = K.T @ v + 2 * (K * C).sum(1)
    x = np.linalg.solve(N, rhs); M2 = np.outer(v, x) + 2 * x[:, None] * C
    rec = dict(layer=l, old_columns=no, rank1=err(r1), sym_Cx=err(M2), rank1_share=round(float(S[0] ** 2 / (S ** 2).sum()), 3),
               x_cos_mu=round(float(abs(x @ s['mu']) / np.linalg.norm(x) / np.linalg.norm(s['mu'])), 3),
               x_cos_bpre=round(float(abs(x @ s['bpre']) / np.linalg.norm(x) / np.linalg.norm(s['bpre'])), 3))
    print(json.dumps(rec), flush=True)

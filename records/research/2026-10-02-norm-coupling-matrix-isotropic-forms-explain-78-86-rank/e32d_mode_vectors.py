#!/usr/bin/env python3
"""TAG opus-reexam E32d (network 35, layers 12 -> 13, numpy): the network-specific norm-coupling matrix is mostly 'collective mode x unit vector':
Ntilde ~ F(C) + sum_m (u_m q_m' + q_m u_m'), u_m = top eigenvectors of C (E32c: one-sided top-8 + spectral part = 93-94% of the energy).
The partner vectors q_m = (network-specific Ntilde) u_m are VECTORS: can they be built from free objects?
  T      (W' G Ntilde(12) G W) u_m                        transported matrix (other half of layer 12), i.e. the carried vector
  conv   sum_r lam_r pp_r(13) (pp_r(13) . u_m)(pp_r(12) . beta)   double contraction of the card's third-order bank       [free: one pass over the columns]
  gauss  (W' G C diag(S2) C G W) u_m, hubS1, R1 (row forms)       [matrix-vector products: free]
Scores: cross-half share of |q_m|^2 explained, per mode and pooled over m <= k, after projecting isotropic and mean forms out of the matrices."""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import pcard as pc
rank = 35; n = 1024; off = ~np.eye(n, dtype=bool)
D = np.load(HERE / f'e32_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); BI = np.load(HERE / f'physicist/t4births_rank{rank:04d}.npz')
C12 = MC['cov_pre_12'].astype(np.float64); C13 = MC['cov_pre_13'].astype(np.float64); mu13 = MC['mean_pre_13'].astype(np.float64); mu12 = MC['mean_pre_12'].astype(np.float64)
sym = lambda x: (x + x.T) / 2
N12 = [sym(D[f'M_12_{h}'] - 2 * C12 @ C12) for h in (0, 1)]; N13 = [sym(D[f'M_13_{h}'] - 2 * C13 @ C13) for h in (0, 1)]
W = pc.weights(rank)[13].astype(np.float64)
v = np.diag(C12); sd = np.sqrt(v); al = mu12 / sd; pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd
m = MC['mean_post_12'].astype(np.float64); S1 = -al * pdf / v; S2 = 2 * (g - g * g - m * d); GC = g[:, None] * C12
push = lambda A: 2 * W.T @ A @ W
F = dict(gauss=push((GC * S2[None, :]) @ GC.T))
row = S1 * ((g * g)[None, :] * C12 * C12).sum(1); H = (row[:, None] * C12) * g[None, :]; F['hubS1'] = push(H + H.T)
beta_t = 2 * m * (1 - g); Cb = C12 @ beta_t; R = ((d * Cb)[:, None] * C12) * g[None, :]; F['R1'] = push(R + R.T)
c12 = np.load(HERE / f'e15cols_rank{rank:04d}_L12.npz'); c13 = np.load(HERE / f'physicist/t4cols_rank{rank:04d}_L13.npz')
pp12 = c12['pp'].astype(np.float64); lam12 = c12['lam'].astype(np.float64); nold = pp12.shape[1]; pp13 = c13['pp'][:, :nold].astype(np.float64)
gc, mc_ = BI['gate_12'].astype(np.float64), BI['host_12'].astype(np.float64); cw = lam12 * (pp12.T @ (2 * mc_ * (1 - gc)))
iso = dict(trC_C=np.trace(C13) * C13, C2=C13 @ C13, mumu=np.outer(mu13, mu13)); Cm = C13 @ mu13; iso['muCmu'] = np.outer(mu13, Cm) + np.outer(Cm, mu13)
Xi = np.stack([iso[k][off] for k in iso], 1)
def ns(A):                      # network-specific part of a matrix: off-diagonal residual after the isotropic and mean forms, diagonal zeroed
    r = np.zeros((n, n)); r[off] = A[off] - Xi @ np.linalg.lstsq(Xi, A[off], rcond=None)[0]; return r
lamC, V = np.linalg.eigh(C13); V = V[:, ::-1]; lamC = lamC[::-1]
Y = [ns(x) for x in N13]; Tm = [ns(W.T @ ((g[:, None] * N12[h]) * g[None, :]) @ W) for h in (0, 1)]; S = {k: ns(F[k]) for k in F}
K = 8; U = V[:, :K]
convU = 2 * (pp13 * cw[None, :]) @ (pp13.T @ U)            # the bank's double contraction, one column pass (n x R x K)
convM = ns(2 * (pp13 * cw[None, :]) @ pp13.T) @ U           # the same after projecting the forms out of the full matrix (reference)
print(json.dumps(dict(top_eigenvalues=[float('%.3g' % x) for x in lamC[:K]], cos_top_eigvec_with_mean=round(float(abs(V[:, 0] @ mu13) / np.linalg.norm(mu13)), 3))))
for h, hp in ((0, 1), (1, 0)):
    q, q2 = Y[h] @ U, Y[hp] @ U; t = Tm[hp] @ U; feats = dict(T=t, conv=convM, gauss=S['gauss'] @ U, hubS1=S['hubS1'] @ U, R1=S['R1'] @ U, conv_raw=convU)
    out = dict(target_half=h, q_energy_share_of_matrix=round(float(2 * (q * q2).sum() / (Y[h] * Y[hp]).sum()), 3), per_mode={}, pooled={})
    for name, cols in (('T', ['T']), ('T_conv', ['T', 'conv']), ('T_conv_gauss_hubS1_R1', ['T', 'conv', 'gauss', 'hubS1', 'R1']), ('conv_only', ['conv']), ('sources_only', ['conv', 'gauss', 'hubS1', 'R1']), ('T_convraw', ['T', 'conv_raw'])):
        X = np.stack([feats[k].ravel() for k in cols], 1); co = np.linalg.lstsq(X, q.ravel(), rcond=None)[0]; f = X @ co
        out['pooled'][name] = dict(share=round(float((2 * f @ q2.ravel() - f @ f) / (q.ravel() @ q2.ravel())), 4), coef=[float('%.3g' % x) for x in co])
    for mm in range(K):
        X = np.stack([feats[k][:, mm] for k in ('T', 'conv', 'gauss', 'hubS1', 'R1')], 1); co = np.linalg.lstsq(X, q[:, mm], rcond=None)[0]; f = X @ co
        out['per_mode'][mm] = dict(rel=round(float(np.corrcoef(q[:, mm], q2[:, mm])[0, 1]), 3), share=round(float((2 * f @ q2[:, mm] - f @ f) / (q[:, mm] @ q2[:, mm])), 3),
                                  share_T_only=round(float((lambda a, b: (2 * (a * (a @ q[:, mm]) / (a @ a)) @ b - ((a * (a @ q[:, mm]) / (a @ a)) ** 2).sum()) / (q[:, mm] @ b))(t[:, mm], q2[:, mm])), 3),
                                  corr_conv=round(float(np.corrcoef(feats['conv'][:, mm], q[:, mm])[0, 1]), 3))
    print(json.dumps(out))

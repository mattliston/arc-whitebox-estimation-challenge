"""[opus-kinkoracle] stop-rule part 2: can cheap runtime features predict the L14 kink-unit relative variance error dv/v (truth - ours)?
features per kink unit (|a14| < 1): a, |a|, log v14, diag share s = sum_j W^2 C13_jj / v14, mu14^2 / v14, and their products with s;
leave-one-rank-out linear regression; report R^2 out of sample and the MSE-weighted share (weights = phi(a)^2 * v, the mean's sensitivity)."""
import numpy as np
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
K = O / 'lead/runs/opus_kinkoracle_v1'
STORE = O.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
X, Y, G, Wt = [], [], [], []
for r in (16, 35, 150, 200, 250, 259, 277, 287):
    d = np.load(K / f'res_dump_{r}_dump.npz'); A = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_A32M.npz'); B = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_B32M.npz')
    W = np.load(STORE / f'rows/observer_fit/{r:04d}/weights.npy')[14].astype(np.float64)
    tv14 = 0.5 * (A['mc_pre_v'][14] + B['mc_pre_v'][14]); v = d['pre_v14']; mu = d['mu14']; a = mu / np.sqrt(v)
    k = np.abs(a) < 1.0
    s = ((W * W).T @ d['post_v13']) / v
    f = np.stack([a, np.abs(a), np.log(v) - np.log(v).mean(), s, mu * mu / v, a * s, np.abs(a) * s], axis=1)[k]
    X.append(f); Y.append(((tv14 - v) / v)[k]); G.append(np.full(int(k.sum()), r)); Wt.append((np.exp(-a * a / 2) ** 2 * v)[k])
X, Y, G, Wt = np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(Wt)
pred = np.zeros_like(Y)
for r in np.unique(G):
    tr, te = G != r, G == r
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xt = np.c_[np.ones(tr.sum()), (X[tr] - mx) / sx]; Xe = np.c_[np.ones(te.sum()), (X[te] - mx) / sx]
    b = np.linalg.lstsq(Xt, Y[tr], rcond=None)[0]; pred[te] = Xe @ b
r2 = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
wr2 = 1 - np.sum(Wt * (Y - pred) ** 2) / np.sum(Wt * Y ** 2)
print(f'n {len(Y)} kink units over 8 ranks; dv/v sd {100*Y.std():.3f}%; leave-one-rank-out R^2 {r2:+.3f}; sensitivity-weighted share removed {wr2:+.3f}')

"""dissect337259 agent: ORIGIN budget of the final-layer MEAN error on the shipped card, many store nets (state dumps y/truth + store weights).
e_l = y_l - truth_l (post means); inherited_l = g_l o (W_l^T e_{l-1}) (first order); born_l = e_l - inherited_l; linear propagation to L15:
P_{l->15} born_l with J_k = diag(g_k) W_k^T; origin share of layer l = <P born_l, e_15> / |e_15|^2 (signed, sums to ~1 if linear).
usage: mean_origin_v1.py <n nets>"""
import sys, glob, json, numpy as np
from pathlib import Path
P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
files = sorted(glob.glob(K + 'stateprobe/sp_*.npz'))[:n]
shares = []; inh_ratio = []; lin_check = []
for f in files:
    rank = int(f.split('_')[-1][:-4]); d = np.load(f)
    W = np.load(STORE / 'rows/observer_fit' / ('%04d' % rank) / 'weights.npy').astype(np.float64)
    y = d['y'].astype(np.float64); t = d['truth'].astype(np.float64); e = y - t
    g = {l: d['gate_%d' % l].astype(np.float64) for l in range(1, 16)}
    born = {0: e[0]}
    for l in range(1, 16):
        inh = g[l] * (W[l].T @ e[l - 1]); born[l] = e[l] - inh
    # propagate each born_l to 15
    contrib = {}
    for l in range(0, 16):
        v = born[l].copy()
        for k in range(l + 1, 16): v = g[k] * (W[k].T @ v)
        contrib[l] = v
    tot = sum(contrib.values()); e15 = e[15]
    lin_check.append(float(np.linalg.norm(tot - e15) / np.linalg.norm(e15)))
    shares.append([float(contrib[l] @ e15 / (e15 @ e15)) for l in range(16)])
    inh_ratio.append([float(np.linalg.norm(e[l] - born[l]) ** 2 / max(np.linalg.norm(e[l]) ** 2, 1e-30)) for l in range(1, 16)])
S = np.array(shares); print('n nets', len(files), 'linearization residual |sum - e15|/|e15| median %.3f' % np.median(lin_check))
print('origin share of the final MSE by layer (mean over nets, %):'); print(' '.join('L%d %.1f' % (l, 100 * S[:, l].mean()) for l in range(16)))
print('L0-7 %.1f  L8-12 %.1f  L13-15 %.1f' % (100 * S[:, :8].sum(1).mean(), 100 * S[:, 8:13].sum(1).mean(), 100 * S[:, 13:].sum(1).mean()))
print('inherited share of |e_l|^2 per layer (mean):', ' '.join('L%d %.2f' % (l + 1, v) for l, v in enumerate(np.array(inh_ratio).mean(0))))
# born error energy per layer (not propagated) for reference

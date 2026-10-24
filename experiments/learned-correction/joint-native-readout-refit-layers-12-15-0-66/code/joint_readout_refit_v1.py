"""TAG fable-lead 2026-10-20: JOINT native refit of the learned mean correction's READOUTS at layers L (delta_l = sigma_l rms_l gain_l
(state_l @ head_l + head_b_l)): the change d(head_l) moves the final mean by J_l (sigma_l * state_l) d(head_l) (mean-chain Jacobian,
J_l = D15 W15^T ... D_{l+1} W_{l+1}^T), so the final residual r15 is fitted jointly over all layers in L by ridge on
X = [J_l (sigma_l * state_l[:, p]), J_l sigma_l] with LEAVE-ONE-NETWORK-OUT and the ridge chosen by LOO. Reports the LOO gain (% of the
final MSE) per ridge and the size of the move (|d head| / |head| per layer needs the literal; here relative to the fitted scale).
usage: joint_readout_refit_v1.py <probe dir> <layers> [max nets]"""
import sys, glob, json
import numpy as np
from pathlib import Path
P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
pdir, layers = Path(sys.argv[1]), [int(x) for x in sys.argv[2].split(',')]
maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
files = sorted(glob.glob(str(pdir / 'sp_*.npz')), key=lambda f: int(Path(f).stem[3:]))[:maxn]
XtX = None; Xty = None; per = []   # per-net (XtX_n, Xty_n, yy_n)
for f in files:
    d = np.load(f); rank = int(Path(f).stem[3:])
    Wn = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'); W = [np.asarray(Wn[l], dtype=np.float64) for l in range(16)]
    y, t = d['y'].astype(np.float64), d['truth'].astype(np.float64); r = y[15] - t[15]
    gates = {k: np.asarray(d['gate_%d' % k], dtype=np.float64) for k in range(1, 16) if ('gate_%d' % k) in d.files}
    cols = []
    for l in layers:
        S = d['state_in_%d' % l].astype(np.float64); sig = np.asarray(d['sigma_%d' % l], dtype=np.float64)
        B = np.column_stack([S * sig[:, None], sig])      # (1024, 65): the readout's basis times sigma
        for k in range(l + 1, 16):
            B = gates[k][:, None] * (W[k].T @ B)
        cols.append(B)
    X = np.concatenate(cols, 1)
    per.append((X.T @ X, X.T @ r, float(r @ r)))
n = len(per); p = per[0][0].shape[0]
XtX = sum(a for a, b, c in per); Xty = sum(b for a, b, c in per); T = sum(c for a, b, c in per)
sc = np.sqrt(np.diag(XtX) / n).clip(1e-300)
print('nets %d, params %d (%d layers x %d), final MSE pooled %.4f e-9' % (n, p, len(layers), p // len(layers), T / n / 1024 * 1e9))
for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
    R = lam * n * np.diag(sc ** 2)
    beta = np.linalg.solve(XtX + R, -Xty); gain_in = (2 * beta @ Xty + beta @ XtX @ beta) / T * 100
    loo = 0.0
    for a, b, c in per:
        bb = np.linalg.solve(XtX - a + R, -(Xty - b)); loo += 2 * bb @ b + bb @ a @ bb
    print('ridge %7.1e: in-sample %+7.3f%%  LOO %+7.3f%%  |beta|/scale per layer %s' % (lam, gain_in, loo / T * 100, ' '.join('%.2e' % np.linalg.norm((beta * sc)[i * (p // len(layers)):(i + 1) * (p // len(layers))]) for i in range(len(layers)))))

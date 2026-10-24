"""TAG fable-lead 2026-10-20: does the GNN mean correction's DIRECTION (cos(delta_l, mu_l), cos(delta_l, m_l), its projection on
the mean pattern) predict the per-network sign of the common-mode optimum (slope of U(c) at 0 from the store screen)?
Features from stateprobe/sp_<rank>.npz: delta_l, mu_l, m0_l (pre-correction post mean), gaussian_mean_l, gate_l; L8-14."""
import glob, json
import numpy as np
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
res = {}
for tag in 'AB':
    d = json.load(open(K + 'cmode_store_%s.json' % tag))
    for r, v in d['results'].items():
        if len(v['U']) >= 4: res[int(r)] = v['U']
rows = []
for rank, U in sorted(res.items()):
    slope = 100 * (U[2] - U[3]) / (0.2 * U[1])
    d = np.load(K + 'stateprobe/sp_%d.npz' % rank)
    f = {}
    for l in range(8, 15):
        dl = d['delta_%d' % l].astype(np.float64); mu = d['mu_%d' % l].astype(np.float64); m0 = d['m0_%d' % l].astype(np.float64)
        g = d['gate_%d' % l].astype(np.float64); gm = d['gaussian_mean_%d' % l].astype(np.float64)
        f.setdefault('cos_delta_mu', []).append(float(dl @ mu / (np.linalg.norm(dl) * np.linalg.norm(mu) + 1e-30)))
        f.setdefault('cos_delta_m0', []).append(float(dl @ m0 / (np.linalg.norm(dl) * np.linalg.norm(m0) + 1e-30)))
        f.setdefault('proj_delta_m0_rel', []).append(float(dl @ m0 / (m0 @ m0)))   # relative common-mode shift of the means by the GNN
        f.setdefault('cos_delta_gate', []).append(float(dl @ g / (np.linalg.norm(dl) * np.linalg.norm(g) + 1e-30)))
        f.setdefault('delta_mean_rel', []).append(float(dl.mean() / m0.mean()))
        f.setdefault('gm_minus_m0_rel', []).append(float((gm - m0).mean() / m0.mean()))
    feats = {k + '_late': float(np.mean(v[2:])) for k, v in f.items()}
    feats.update({k + '_mid': float(np.mean(v[:2])) for k, v in f.items()})
    rows.append((rank, slope, feats))
names = sorted(rows[0][2].keys())
y = np.array([r[1] for r in rows]); X = np.array([[r[2][n] for n in names] for r in rows])
print('n =', len(rows)); print('corr(slope, feature):'); 
for j, n in enumerate(names):
    print('  %-28s %+.2f' % (n, np.corrcoef(y, X[:, j])[0, 1]))
# LOO sign rule for the two best single features
dp = np.array([100 * (res[r][2] / res[r][1] - 1) for r, _, _ in rows]); dm = np.array([100 * (res[r][3] / res[r][1] - 1) for r, _, _ in rows])
for j in np.argsort(-np.abs([np.corrcoef(y, X[:, k])[0, 1] for k in range(len(names))]))[:3]:
    Xd = np.c_[np.ones(len(y)), X[:, j]]; pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        beta = np.linalg.lstsq(Xd[m], y[m], rcond=None)[0]; pred[i] = Xd[i] @ beta
    gain = np.where(pred < 0, dp, dm)
    print('LOO sign rule on %-26s mean %+.3f%% (lower %d/%d)' % (names[j], gain.mean(), int((gain < 0).sum()), len(y)))
print('oracle %+.3f%%' % np.minimum(dp, dm).mean())

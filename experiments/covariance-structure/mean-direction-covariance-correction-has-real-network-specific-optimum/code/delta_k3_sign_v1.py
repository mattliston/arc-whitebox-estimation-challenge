"""TAG fable-lead 2026-10-20: does the GNN correction oppose or reinforce the card's own third-order MEAN term per network, and does
that predict the per-network common-mode sign? k3 mean term per unit ~ k3 * pdf(a) * (a^2 - 1) / (6 v) (Edgeworth first order);
features: cos(delta, k3term), the LS coefficient of delta on k3term, same for the k4 term (k4 pdf (a^3-3a) /(24 v^1.5)); L8-14."""
import json
import numpy as np
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
res = {}
for tag in 'AB':
    d = json.load(open(K + 'cmode_store_%s.json' % tag))
    for r, v in d['results'].items():
        if len(v['U']) >= 4: res[int(r)] = v['U']
rows = []
for rank, U in sorted(res.items()):
    slope = 100 * (U[2] - U[3]) / (0.2 * U[1]); d = np.load(K + 'stateprobe/sp_%d.npz' % rank); f = {}
    for l in range(8, 15):
        dl = d['delta_%d' % l].astype(np.float64); a = d['alpha_%d' % l].astype(np.float64); v = d['pre_v_%d' % l].astype(np.float64)
        k3 = d['pre_k3_%d' % l].astype(np.float64); k4 = d['pre_k4_%d' % l].astype(np.float64); pdf = d['pdf_%d' % l].astype(np.float64); s = np.sqrt(v)
        t3 = k3 * pdf * (a * a - 1) / (6 * v); t4 = k4 * pdf * (a ** 3 - 3 * a) / (24 * v * s)
        for name, t in (('k3term', t3), ('k4term', t4)):
            f.setdefault('cos_delta_' + name, []).append(float(dl @ t / (np.linalg.norm(dl) * np.linalg.norm(t) + 1e-30)))
            f.setdefault('beta_delta_' + name, []).append(float(dl @ t / (t @ t + 1e-30)))
        f.setdefault('delta_over_k3term', []).append(float(np.linalg.norm(dl) / (np.linalg.norm(t3) + 1e-30)))
    feats = {k + '_late': float(np.mean(x[2:])) for k, x in f.items()}; feats.update({k + '_mid': float(np.mean(x[:2])) for k, x in f.items()})
    rows.append((rank, slope, feats))
names = sorted(rows[0][2].keys()); y = np.array([r[1] for r in rows]); X = np.array([[r[2][n] for n in names] for r in rows])
dp = np.array([100 * (res[r][2] / res[r][1] - 1) for r, _, _ in rows]); dm = np.array([100 * (res[r][3] / res[r][1] - 1) for r, _, _ in rows])
print('n =', len(rows))
for j, n in enumerate(names):
    print('  %-26s corr %+.2f' % (n, np.corrcoef(y, X[:, j])[0, 1]))
for j in np.argsort(-np.abs([np.corrcoef(y, X[:, k])[0, 1] for k in range(len(names))]))[:2]:
    Xd = np.c_[np.ones(len(y)), X[:, j]]; pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        beta = np.linalg.lstsq(Xd[m], y[m], rcond=None)[0]; pred[i] = Xd[i] @ beta
    gain = np.where(pred < 0, dp, dm); print('LOO sign rule on %-24s mean %+.3f%% (lower %d/%d)' % (names[j], gain.mean(), int((gain < 0).sum()), len(y)))

"""TAG fable-lead 2026-10-20: can card-known per-unit features predict the pre-variance error dv_i = v_card - v_true at L9-14?
Fit a universal linear model (per layer, and pooled) on one net, test on the other (nets 35/100; k21dump card quantities + mcall
truth). Target: dv_i / vscale (vscale = rms truth variance of the layer). Reports out-of-sample R^2 on sum dv^2 (the oracle metric)."""
import numpy as np, json
from scipy.stats import norm
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
STORE = P2 + '/learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.load(open(STORE + '/manifest.json'))['roles']['observer_fit']['row_manifests']
def weights(rank):
    p = STORE + '/' + MAN[rank]['path']; p = p[:-len('/row_manifest.json')]
    return np.load(p + '/weights.npy')
def gauss_post(mu, v):
    s = np.sqrt(v); a = mu / s; phi, Phi = norm.pdf(a), norm.cdf(a)
    m = mu * Phi + s * phi; e2 = (mu ** 2 + v) * Phi + mu * s * phi
    return m, e2 - m ** 2, phi, Phi, s
data = {}
for rank in (35, 100):
    T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank); Wl = weights(rank)
    for l in range(9, 15):
        d, dp = np.load('k21dump_rank%d_L%d.npz' % (rank, l)), np.load('k21dump_rank%d_L%d.npz' % (rank, l - 1))
        W = np.asarray(Wl[l], dtype=np.float64)
        if W.shape != (1024, 1024): W = W.T
        mu, v, k3, k4 = d['mu'], d['v'], d['k3'], d['k4']
        mp, vp, phip, Phip, sp = gauss_post(dp['mu'], dp['v'])
        vt = np.diag(T['cov_pre_%d' % l]).astype(np.float64); dv = v - vt
        # orientation check: mu should equal W @ m_post(l-1) + b approximately -> pick orientation by correlation
        c1 = np.corrcoef(W @ mp, mu)[0, 1]; c2 = np.corrcoef(W.T @ mp, mu)[0, 1]
        if c2 > c1: W = W.T
        s = np.sqrt(v); a = mu / s
        F = dict(one=np.ones(1024), v=v, mu=mu, mu2=mu ** 2, k3=k3, k4=k4, gate=d['gate'], alpha=d['alpha'],
                 a=a, phi=norm.pdf(a), Phi=norm.cdf(a), Wphi=W @ phip, WPhi=W @ Phip, W1=W @ np.ones(1024), Wsphi=W @ (sp * phip),
                 Wm=W @ mp, Wvp=(W ** 2) @ vp, rown=(W ** 2).sum(1), Wk3=(W ** 3) @ dp['k3'], Wk4=(W ** 4) @ dp['k4'],
                 Cpre_rowsum=d['Cpre'].astype(np.float64).sum(1), K21c_rowsum=d['K21_card'].astype(np.float64).sum(1),
                 K21c_diag=np.diag(d['K21_card']).astype(np.float64))
        # products with v (relative structure)
        F2 = dict(F); 
        for k in ('mu2', 'Wphi', 'WPhi', 'Wsphi', 'Wm', 'Wvp', 'k4', 'k3'):
            F2[k + '_x_v'] = F[k] * v
        data[(rank, l)] = (F2, dv, vt, np.sqrt((vt ** 2).mean()), c1, c2)
names = list(next(iter(data.values()))[0].keys())
def design(F, vs):
    X = np.stack([F[n] for n in names], 1)
    return X / vs
def fit(keys, lam=1e-3):
    Xs, ys = [], []
    for k in keys:
        F, dv, vt, vs, _, _ = data[k]; X = design(F, vs); X = X / np.abs(X).mean(0).clip(1e-12); Xs.append(X); ys.append(dv / vs)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    sc = np.abs(X).mean(0).clip(1e-12)
    return sc, np.linalg.solve((X / sc).T @ (X / sc) + lam * np.eye(X.shape[1]), (X / sc).T @ y)
def r2(key, sc, beta):
    F, dv, vt, vs, _, _ = data[key]; X = design(F, vs); X = X / np.abs(X).mean(0).clip(1e-12)
    pred = (X / sc) @ beta * vs
    return 1 - ((dv - pred) ** 2).sum() / (dv ** 2).sum(), np.corrcoef(pred, dv)[0, 1]
print('orientation corr', {k: (round(v[4], 3), round(v[5], 3)) for k, v in data.items()})
for l in range(9, 15):
    sc, b = fit([(35, l)]); r_in = r2((35, l), sc, b); r_out = r2((100, l), sc, b)
    sc2, b2 = fit([(100, l)]); r_in2 = r2((100, l), sc2, b2); r_out2 = r2((35, l), sc2, b2)
    print('L%2d per-layer: fit35 R2 in %.3f out(100) %.3f corr %.3f | fit100 R2 in %.3f out(35) %.3f corr %.3f' % (l, r_in[0], r_out[0], r_out[1], r_in2[0], r_out2[0], r_out2[1]))
sc, b = fit([(35, l) for l in range(9, 15)])
print('pooled L9-14 fit on 35 -> R2 on 100 per layer', ' '.join('%d:%.3f' % (l, r2((100, l), sc, b)[0]) for l in range(9, 15)))
sc, b = fit([(100, l) for l in range(9, 15)])
print('pooled L9-14 fit on 100 -> R2 on 35 per layer', ' '.join('%d:%.3f' % (l, r2((35, l), sc, b)[0]) for l in range(9, 15)))
# single-feature screens (pooled, both nets): corr of each feature with dv/vs
print('single-feature |corr| with dv (pooled both nets, L10-14):')
rows = []
for n in names:
    xs, ys = [], []
    for k, (F, dv, vt, vs, _, _) in data.items():
        if k[1] >= 10: xs.append(F[n] / vs); ys.append(dv / vs)
    x, y = np.concatenate(xs), np.concatenate(ys)
    if np.std(x) > 0: rows.append((abs(np.corrcoef(x, y)[0, 1]), n))
print(' '.join('%s:%.3f' % (n, c) for c, n in sorted(rows, reverse=True)))

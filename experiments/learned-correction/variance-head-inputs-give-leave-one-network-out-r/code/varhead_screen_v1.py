"""TAG fable-lead 2026-10-20: VARIANCE-HEAD screen. Does the learned correction's GNN state (carried into layer l before its pair map)
predict the per-unit PRE-variance error dv_l = v_card - v_true (the -18/-26% non-co-adapted oracle channel)? Nets 35/37/39/100 (the only
ones with pre-covariance truths, mcall P32M), layers 9-14, ridge on [1, state_in_l (64), pre mu/v/k3/k4, x_{l-1} (30)] with
LEAVE-ONE-NET-OUT; target dv/vscale. Reports out-of-sample R^2 per layer and per net, and the implied share of the oracle."""
import numpy as np, sys
P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
ranks = [35, 37, 39, 100]; LAYERS = list(range(9, 15))
data = {}
for r in ranks:
    d = np.load('stateprobe_var/sp_%d.npz' % r); T = np.load(P2 + '/fable_covhunter_v1/mcall_rank%04d_P32M.npz' % r)
    for l in LAYERS:
        v = d['pre_v_%d' % l].astype(np.float64); vt = np.diag(T['cov_pre_%d' % l]).astype(np.float64)
        vsA, vsB = np.diag(T['cov_pre_A_%d' % l]), np.diag(T['cov_pre_B_%d' % l])
        vs = np.sqrt((vt ** 2).mean()); y = (v - vt) / vs; noise = ((vsA - vsB) / 2) / vs
        S = d['state_in_%d' % l].astype(np.float64); xprev = d['x_%d' % (l - 1)].astype(np.float64) if ('x_%d' % (l - 1)) in d.files else np.zeros((1024, 0))
        mu, k3, k4 = d['mu_%d' % l], d['pre_k3_%d' % l], d['pre_k4_%d' % l]
        s = np.sqrt(v); a = mu / s
        base = np.column_stack([np.ones(1024), v / vs, mu, a, k3 / s ** 3, k4 / v ** 2, (v - vt * 0) / vs])
        data[(r, l)] = dict(y=y, noise=noise, S=S, X=xprev, base=base)
def design(dd, what):
    cols = [dd['base']]
    if 'S' in what: cols.append(dd['S'])
    if 'X' in what: cols.append(dd['X'])
    return np.concatenate(cols, 1)
for what in ('base', 'base+S', 'base+S+X'):
    for lam in (1e-3, 1e-1):
        res = []
        for l in LAYERS:
            r2s = []
            for hold in ranks:
                Xtr = np.concatenate([design(data[(r, l)], what) for r in ranks if r != hold]); ytr = np.concatenate([data[(r, l)]['y'] for r in ranks if r != hold])
                sc = np.sqrt((Xtr ** 2).mean(0)).clip(1e-12); Xs = Xtr / sc
                th = np.linalg.solve(Xs.T @ Xs + lam * len(ytr) * np.eye(Xs.shape[1]), Xs.T @ ytr) / sc
                Xte = design(data[(hold, l)], what); yte = data[(hold, l)]['y']; p = Xte @ th
                r2s.append(1 - ((yte - p) ** 2).sum() / (yte ** 2).sum())
            res.append('L%d:%+.3f' % (l, np.mean(r2s)))
        print('%-9s ridge %.0e  LOO R^2 by layer: %s' % (what, lam, ' '.join(res)))
# noise ceiling: the truth's own MC noise share of dv^2
print('truth MC noise share of dv^2 by layer:', ' '.join('L%d:%.2f' % (l, np.mean([(data[(r, l)]['noise'] ** 2).sum() / (data[(r, l)]['y'] ** 2).sum() for r in ranks])) for l in LAYERS))
# pooled-layer fit (one model for all layers 10-14), LOO by net
for what in ('base+S', 'base+S+X'):
    r2s = []
    for hold in ranks:
        Xtr = np.concatenate([design(data[(r, l)], what) for r in ranks if r != hold for l in LAYERS[1:]]); ytr = np.concatenate([data[(r, l)]['y'] for r in ranks if r != hold for l in LAYERS[1:]])
        sc = np.sqrt((Xtr ** 2).mean(0)).clip(1e-12); Xs = Xtr / sc
        th = np.linalg.solve(Xs.T @ Xs + 1e-2 * len(ytr) * np.eye(Xs.shape[1]), Xs.T @ ytr) / sc
        for l in LAYERS[1:]:
            Xte = design(data[(hold, l)], what); yte = data[(hold, l)]['y']; p = Xte @ th
            r2s.append((l, 1 - ((yte - p) ** 2).sum() / (yte ** 2).sum()))
    print('%-9s pooled L10-14 model, LOO R^2 per (net,layer) mean %.3f, min %.3f, max %.3f' % (what, np.mean([x[1] for x in r2s]), min(x[1] for x in r2s), max(x[1] for x in r2s)))

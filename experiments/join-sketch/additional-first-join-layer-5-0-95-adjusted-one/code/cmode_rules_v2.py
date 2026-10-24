"""dissect337259 agent: rule families beyond the linear one, LOO on the 26 nets (uses cmode_magnitude_v1.npz). usage: cmode_rules_v2.py [CMAX]"""
import sys, numpy as np
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
z = np.load(K + 'agent_ideas_v1/cmode_magnitude_v1.npz', allow_pickle=True); names = list(z['names']); X = z['X']; A = z['A']; B = z['B']
CMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
rng = np.random.default_rng(1)
def gain(a, b, c): return a * c + b * c * c
def zs(x): return (x - x.mean(0)) / (x.std(0) + 1e-12)
def fit_a(xtr, atr, lam):
    Xd = np.column_stack([np.ones(len(atr)), xtr]); R = lam * np.eye(Xd.shape[1]); R[0, 0] = 0
    return np.linalg.solve(Xd.T @ Xd + R, Xd.T @ atr)
def loo(xcols, rule, lam=1.0, **kw):
    n = len(A); g = np.zeros(n); cs = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        xs = zs(xcols[tr]); mu, sd = xcols[tr].mean(0), xcols[tr].std(0) + 1e-12
        w = fit_a(xs, A[tr], lam); a_hat_tr = w[0] + xs @ w[1:]; a_hat = w[0] + ((xcols[i] - mu) / sd) @ w[1:]
        c = rule(a_hat, a_hat_tr, A[tr], B[tr], **kw); cs[i] = c; g[i] = gain(A[i], B[i], c)
    return g, cs
def r_linear(a_hat, a_tr, A_tr, B_tr):
    bbar = max(np.mean(np.maximum(B_tr, 5.0)), 5.0); return float(np.clip(-a_hat / (2 * bbar), -CMAX, CMAX))
def r_scaled(a_hat, a_tr, A_tr, B_tr):
    # c = -k a_hat with k chosen on the training fold to maximise the in-fold gain (closed form: k = sum(a a_hat) / (2 sum(b a_hat^2)))
    k = np.sum(A_tr * a_tr) / max(2 * np.sum(np.maximum(B_tr, 0.0) * a_tr ** 2), 1e-9); return float(np.clip(-k * a_hat, -CMAX, CMAX))
def r_bang(a_hat, a_tr, A_tr, B_tr, frac=0.5):
    # apply +-CMAX against the predicted sign only when |a_hat| is in the top 'frac' of the training fold's |a_hat| (confidence gate)
    thr = np.quantile(np.abs(a_tr), 1 - frac); return float(-CMAX * np.sign(a_hat)) if abs(a_hat) >= thr else 0.0
def perm(xcols, rule, stat, nperm=400, lam=1.0, **kw):
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(len(A)); g, _ = loo(xcols[p], rule, lam, **kw); cnt += (g.mean() <= stat)
    return cnt / nperm
sets = {'gnn_eff_slope': ['slp_gnn_eff'], 'diff': ['slp_diff'], 'gnn+diff': ['slp_gnn_eff', 'slp_diff'], 'gnn+diff+kink15': ['slp_gnn_eff', 'slp_diff', 'slp_kink15'],
        'gnn+mu2': ['slp_gnn_eff', 'slp_mu2'], 'obs-only: diff+mu2+kink1': ['slp_diff', 'slp_mu2', 'slp_kink1'], 'all5': ['slp_gnn_eff', 'slp_diff', 'slp_mu2', 'slp_kink15', 'slp_mumode']}
print('oracle |c|<=%.2f: %.3f%%' % (CMAX, gain(A, B, np.clip(np.where(B > 0, -A / (2 * np.maximum(B, 1e-9)), np.sign(-A) * CMAX), -CMAX, CMAX)).mean()))
for sname, cols in sets.items():
    x = X[:, [names.index(c) for c in cols]]
    if not np.isfinite(x).all(): continue
    for lam in (1.0, 5.0):
        for rname, rule, kw in (('linear', r_linear, {}), ('scaled', r_scaled, {}), ('bang50', r_bang, dict(frac=0.5)), ('bang30', r_bang, dict(frac=0.3))):
            g, cs = loo(x, rule, lam, **kw)
            if g.mean() < -0.1 or (sname in ('gnn_eff_slope', 'diff') and lam == 1.0):
                pr = perm(x, rule, g.mean(), nperm=300, lam=lam, **kw)
                print('  %-26s lam %.0f %-7s gain %+.3f%%  lower %.2f  applied %2d/26  chance %.3f' % (sname, lam, rname, g.mean(), (g < 0).mean(), int((cs != 0).sum()), pr))
            else:
                print('  %-26s lam %.0f %-7s gain %+.3f%%  lower %.2f  applied %2d/26' % (sname, lam, rname, g.mean(), (g < 0).mean(), int((cs != 0).sum())))
# how much of the oracle a perfect-SIGN rule with fixed |c| captures
for cm in (0.05, 0.1, 0.15, 0.2):
    print('perfect sign, |c| = %.2f: mean gain %+.3f%%' % (cm, gain(A, B, -cm * np.sign(A)).mean()))

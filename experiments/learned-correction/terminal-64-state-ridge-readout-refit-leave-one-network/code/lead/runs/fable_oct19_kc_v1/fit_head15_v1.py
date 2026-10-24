"""TAG fable-lead 2026-10-20: refit of the learned correction's TERMINAL readout (gnn_step at l = 15: delta = sigma rms15 (state15 @ head15 + head_b15) gain15)
on the shipped card's own state, from native_head_probe_v1.py dumps (headfit/hp_<net>.npz: state15, sigma, m0 = output - delta, A, B truths).
Ridge least squares on the residual (truth - m0) with leave-one-net-out validation (cross-half U on the held-out net); prints the pooled
held-out U change against the current head, and writes the fitted head15 / head_b15 to headfit/head15_fit.npz.
usage: fit_head15_v1.py <glob of dumps> [ridge lambda relative, default 1e-3] [shrink toward current head 0..1, default 0]"""
import sys, glob
import numpy as np
files = sorted(glob.glob(sys.argv[1])); lam_rel = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3; shrink = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
nets = []
for f in files:
    z = np.load(f); S = z['state15']; sig = z['sigma']; m0 = z['m0']; A = z['A']; B = z['B']; y15 = z['y15']
    rms = float(np.asarray(z['P15_rms']).ravel()[0]) if 'P15_rms' in z.files else 1.0; gain = float(np.asarray(z['P15_gain']).ravel()[0]) if 'P15_gain' in z.files else 1.0
    head = np.asarray(z['P15_head'], dtype=np.float64).ravel(); head_b = float(np.asarray(z['P15_head_b']).ravel()[0]) if 'P15_head_b' in z.files else 0.0
    X = np.concatenate([S * sig[:, None], sig[:, None]], 1); target = (A + B) / 2 - m0      # delta = sigma (S @ h_eff + b_eff): exact (LS residual 1e-11); P15 constants do not reproduce it
    nets.append(dict(name=f, X=X, t=target, A=A, B=B, m0=m0, y15=y15, delta=y15 - m0, P15=np.concatenate([head, [head_b]]), rg=rms * gain))
Xall = np.concatenate([n['X'] for n in nets]); dall = np.concatenate([n['delta'] for n in nets]); h_eff, *_ = np.linalg.lstsq(Xall, dall, rcond=None)
for n in nets: n['head'] = h_eff; n['check'] = float(np.abs(n['m0'] + n['X'] @ h_eff - n['y15']).max())
ratio = h_eff[:-1] / (nets[0]['P15'][:-1] * nets[0]['rg']); print('effective head vs P15 head x rms x gain: ratio median %.4f spread %.2e; b ratio %.4f' % (np.median(ratio), np.std(ratio), h_eff[-1] / (nets[0]['P15'][-1] * nets[0]['rg'])))
np.save('headfit/h_eff.npy', h_eff)
print('%d nets; head dim %d; reconstruction check with the effective head (max |m0 + X h - y15|) %.2e' % (len(nets), nets[0]['X'].shape[1] - 1, max(n['check'] for n in nets)))
U = lambda p, n: float(np.mean((p - n['A']) * (p - n['B'])))
def fit(train):
    X = np.concatenate([n['X'] for n in train]); t = np.concatenate([n['t'] for n in train]); h0 = train[0]['head']
    G = X.T @ X; lam = lam_rel * np.trace(G) / G.shape[0]; h = np.linalg.solve(G + lam * np.eye(G.shape[0]), X.T @ t + lam * h0)   # ridge toward the current head
    return (1 - shrink) * h + shrink * h0
base = []; refit = []; insample = []
for i, n in enumerate(nets):
    h = fit([m for j, m in enumerate(nets) if j != i]); base.append(U(n['y15'], n)); refit.append(U(n['m0'] + n['X'] @ h, n)); insample.append(U(n['m0'] + n['X'] @ fit(nets), n))
base, refit, insample = map(np.array, (base, refit, insample))
print('leave-one-net-out: pooled U base %.4e -> refit %.4e (%+.3f%%, lower on %d/%d, mean-of-ratios %+.3f%% se %.3f) | in-sample %+.3f%% | head-off U %+.2f%%'
      % (base.mean(), refit.mean(), 100 * (refit.mean() / base.mean() - 1), (refit < base).sum(), len(nets), 100 * (refit / base - 1).mean(), 100 * (refit / base - 1).std(ddof=1) / np.sqrt(len(nets)), 100 * (insample.mean() / base.mean() - 1), 100 * (np.mean([U(n['m0'], n) for n in nets]) / base.mean() - 1)))
hfull = fit(nets); np.savez('headfit/head15_fit.npz', head=hfull[:-1], head_b=hfull[-1], head_old=nets[0]['head'][:-1], head_b_old=nets[0]['head'][-1], lam_rel=lam_rel, shrink=shrink, n_nets=len(nets))
print('fitted head15 saved: |h_new - h_old| / |h_old| = %.3f, head_b %.4g -> %.4g' % (np.linalg.norm(hfull[:-1] - nets[0]['head'][:-1]) / np.linalg.norm(nets[0]['head'][:-1]), nets[0]['head'][-1], hfull[-1]))

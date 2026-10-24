"""TAG fable-lead 2026-10-20: screen of candidate NEW inputs for the learned mean correction, natively, on the final residual.
For layer l and candidate per-unit feature f (from the state probe stateprobe/sp_<rank>.npz), a linear bypass delta_l += c sigma_l f
moves the final mean by c J_l (sigma_l f), J_l = D15 W15^T D14 W14^T ... D_{l+1} W_{l+1}^T (D_k = diag(Phi(a_k)) = the pair map's gate;
the mean-chain linearisation validated 99% at L14 on 10-16). c is fitted by least squares on the final residual r15 = y15 - truth15 with
LEAVE-ONE-NETWORK-OUT; the LOO gain is the change of sum r15^2 across nets (% of the final MSE). 'newness' = 1 - R^2 of f regressed on
[1, state_l, x_l] (what the head already sees, linearly), pooled over the first 24 nets.
usage: head_input_screen_v1.py <probe dir> <layers e.g. 10,11,12,13,14,15> [max nets]"""
import sys, glob, json
import numpy as np
from pathlib import Path
P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
pdir, layers = Path(sys.argv[1]), [int(x) for x in sys.argv[2].split(',')]
maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
files = sorted(glob.glob(str(pdir / 'sp_*.npz')), key=lambda f: int(Path(f).stem[3:]))[:maxn]
def feats(d, l, W):
    g = lambda k: np.asarray(d['%s_%d' % (k, l)], dtype=np.float64)
    v, k3, k4, mu, s = g('pre_v'), g('pre_k3'), g('pre_k4'), g('mu'), g('sigma')
    a = g('alpha'); phi = g('pdf'); gate = g('gate')
    e = -a * phi / v; fv = phi / (2 * s)      # d m / d k3 (x6) and d m / d v
    F = dict(one=np.ones_like(v), v=v, k3=k3, k4=k4, mu=mu, gate=gate, pdf=phi, m0=g('m0'), delta=g('delta'),
             k3std=k3 / s ** 3, k4std=k4 / v ** 2, k3e=k3 * e / 6, k4f=k4 * (a * a - 1) * phi / (v * s) / 24,
             crow=np.sqrt(g('crow2')) / v if ('crow2_%d' % l) in d.files else np.zeros_like(v),
             k21row=np.sqrt(g('k21row2')) / s ** 3 if ('k21row2_%d' % l) in d.files else np.zeros_like(v),
             k21col=np.sqrt(g('k21col2')) / s ** 3 if ('k21col2_%d' % l) in d.files else np.zeros_like(v),
             k21rowsum=g('k21rowsum') / s ** 3 if ('k21rowsum_%d' % l) in d.files else np.zeros_like(v),
             k21colsum=g('k21colsum') / s ** 3 if ('k21colsum_%d' % l) in d.files else np.zeros_like(v),
             dcovrow=np.sqrt(g('dcovrow2')) / v if ('dcovrow2_%d' % l) in d.files else np.zeros_like(v),
             dcovrowsum=g('dcovrowsum') / v if ('dcovrowsum_%d' % l) in d.files else np.zeros_like(v),
             cpostrow=np.sqrt(g('cpostrow2')) if ('cpostrow2_%d' % l) in d.files else np.zeros_like(v))
    if l >= 2 and ('post_k3_%d' % (l - 1)) in d.files:
        Wl = W[l]; pk3, pk4, pv = (np.asarray(d['post_k3_%d' % (l - 1)], dtype=np.float64), np.asarray(d['post_k4_%d' % (l - 1)], dtype=np.float64), np.asarray(d['post_v_%d' % (l - 1)], dtype=np.float64))
        k3hat = (Wl ** 3).T @ pk3; k4hat = (Wl ** 4).T @ pk4; vhat = (Wl ** 2).T @ pv
        F.update(k3hat=k3hat, k4hat=k4hat, vhat=vhat, dk3hat=(k3hat - k3), dvhat=(vhat - v), k3hat_e=k3hat * e / 6, dk3hat_e=(k3hat - k3) * e / 6, dvhat_fv=(vhat - v) * fv, k4hat_f=k4hat * (a * a - 1) * phi / (v * s) / 24)
    # products with the sensitivities (what a mean correction from an error proxy would look like)
    for k in ('crow', 'k21row', 'k21col', 'dcovrow', 'k3std', 'k4std'):
        F[k + '_e'] = F[k] * e * s ** 3 / 6; F[k + '_fv'] = F[k] * fv * v
    return F, e, fv
stats = {}   # (l, name) -> list of (a_n, b_n)
tot = []     # per-net sum r15^2
newness = {}
nets = 0
for f in files:
    d = np.load(f); rank = int(Path(f).stem[3:])
    Wn = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'); W = [np.asarray(Wn[l], dtype=np.float64) for l in range(16)]
    y, t = d['y'].astype(np.float64), d['truth'].astype(np.float64); r = y[15] - t[15]; tot.append(float((r ** 2).sum()))
    gates = {k: np.asarray(d['gate_%d' % k], dtype=np.float64) for k in range(1, 16) if ('gate_%d' % k) in d.files}
    for l in layers:
        F, e, fv = feats(d, l, W); sig = np.asarray(d['sigma_%d' % l], dtype=np.float64)
        for name, fe in F.items():
            u = sig * fe
            for k in range(l + 1, 16):
                u = gates[k] * (W[k].T @ u)
            stats.setdefault((l, name), []).append((float(u @ r), float(u @ u)))
        if nets < 24:
            X = np.column_stack([np.ones(1024), d['state_in_%d' % l].astype(np.float64), d['x_%d' % l].astype(np.float64)])
            Xs = X / np.maximum(np.abs(X).mean(0), 1e-12)
            G = Xs.T @ Xs + 1e-6 * np.eye(Xs.shape[1])
            for name, fe in F.items():
                if np.std(fe) == 0: continue
                beta = np.linalg.solve(G, Xs.T @ fe); res = fe - Xs @ beta
                newness.setdefault((l, name), []).append(float(1 - (res ** 2).sum() / ((fe - fe.mean()) ** 2).sum()))
    nets += 1
T = sum(tot)
print('nets %d, final MSE pooled %.4f e-9' % (nets, T / nets / 1024 * 1e9))
rows = []
for (l, name), ab in stats.items():
    A = np.array([x[0] for x in ab]); B = np.array([x[1] for x in ab])
    if B.sum() == 0: continue
    c_all = -A.sum() / B.sum()
    gain_in = (2 * c_all * A.sum() + c_all ** 2 * B.sum()) / T * 100
    loo = 0.0
    for n in range(len(A)):
        c = -(A.sum() - A[n]) / max(B.sum() - B[n], 1e-300); loo += 2 * c * A[n] + c ** 2 * B[n]
    nw = 1 - np.mean(newness.get((l, name), [0.0]))
    rows.append((loo / T * 100, gain_in, l, name, c_all, nw))
rows.sort()
print('%8s %8s %3s %-14s %12s %7s' % ('LOO%', 'in%', 'L', 'feature', 'c', 'newness'))
for loo, gin, l, name, c, nw in rows[:40]:
    print('%+8.3f %+8.3f %3d %-14s %12.4e %7.3f' % (loo, gin, l, name, c, nw))
print('... worst 5:'); [print('%+8.3f %+8.3f %3d %-14s %12.4e %7.3f' % (loo, gin, l, name, c, nw)) for loo, gin, l, name, c, nw in rows[-5:]]

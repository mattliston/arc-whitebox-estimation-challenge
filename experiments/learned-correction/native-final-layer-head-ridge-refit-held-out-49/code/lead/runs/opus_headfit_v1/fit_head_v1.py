"""[opus-headfit] ridge LS refit of (head[15], head_b[15]) on captured L15 inputs. y_nod = y15 - delta; delta = s (state@head + b) with
s = sigma*rms*gain (checked). Minimizes sum over units of (truth - y_nod - s (state@h + b))^2 = weighted LS with weights s^2, ridge toward
the CURRENT head. lambda by network-grouped 5-fold CV inside the TRAIN 2/3 only, scored by the noise-free cross-half U (mean (y-A)(y-B)).
Reports CV gain, held-out 1/3 gain and the 24 control seeds (offline-exact: the head acts only at the last step)."""
import json, glob, math, statistics as st, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; O = H.parents[2]; F = O/'analyst/fresh'
seeds = json.load(open(H/'SEEDS_headfit_v1.json'))['seeds']; CTRL = list(range(7103248, 7103256)) + list(range(7103080, 7103096))
def load(s):
    p = H/f'cap/head_{s}.npz'
    if not p.exists(): return None
    z = np.load(p); t = np.load(F/f'truth_seed{s}.npz')
    sc = z['sigma'] * float(np.ravel(z['rms'])[0] if np.size(z['rms']) == 1 else 1) * float(np.ravel(z['gain'])[0])
    if np.size(z['rms']) != 1: sc = z['sigma'] * np.ravel(z['rms']) * float(np.ravel(z['gain'])[0])
    head, b = np.ravel(z['head']), float(np.ravel(z['head_b'])[0])
    X = np.concatenate([z['state'], np.ones((z['state'].shape[0], 1))], axis=1)
    pred = sc * (X @ np.append(head, b))
    chk = float(np.max(np.abs(pred - z['delta'])) / (np.max(np.abs(z['delta'])) + 1e-30))
    return dict(seed=s, X=X, sc=sc, ynod=z['y15'] - z['delta'], y15=z['y15'], A=t['mc_post_A'][-1].astype(np.float64), B=t['mc_post_B'][-1].astype(np.float64),
                T=t['mc_post'][-1].astype(np.float64), theta0=np.append(head, b), chk=chk)
D = {s: d for s in seeds + CTRL if (d := load(s)) is not None}
fresh = [s for s in seeds if s in D]; ctrl = [s for s in CTRL if s in D]
print('captured fresh', len(fresh), 'controls', len(ctrl), '| delta reproduction max rel err', max(D[s]['chk'] for s in D))
theta0 = D[fresh[0]]['theta0']; assert all(np.allclose(D[s]['theta0'], theta0) for s in D)
rng = np.random.default_rng(7); order = list(fresh); rng.shuffle(order)
ntr = (2 * len(order)) // 3; train, held = sorted(order[:ntr]), sorted(order[ntr:])
def normal(ss):
    G = np.zeros((len(theta0),) * 2); g = np.zeros(len(theta0))
    for s in ss:
        d = D[s]; Xw = d['X'] * d['sc'][:, None]; r = d['T'] - d['ynod']
        G += Xw.T @ Xw; g += Xw.T @ r
    return G, g
def solve(G, g, lam):
    n = G.shape[0]; scale = np.trace(G) / n
    return np.linalg.solve(G + lam * scale * np.eye(n), g + lam * scale * theta0)
def gain(ss, th):   # mean per-network log cross-half ratio (new vs current head), and aggregate
    lr, Un, Ub = [], 0.0, 0.0
    for s in ss:
        d = D[s]; y = d['ynod'] + d['sc'] * (d['X'] @ th); y0 = d['y15']
        un, ub = float(np.mean((y - d['A']) * (y - d['B']))), float(np.mean((y0 - d['A']) * (y0 - d['B'])))
        lr.append(100 * math.log(un / ub)); Un += un; Ub += ub
    return st.mean(lr), (st.stdev(lr) / math.sqrt(len(lr)) if len(lr) > 1 else float('nan')), 100 * math.log(Un / Ub), sum(x < 0 for x in lr), lr
folds = [train[i::5] for i in range(5)]; lams = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0]; cv = {}
for lam in lams:
    lr = []
    for f in folds:
        G, g = normal([s for s in train if s not in f]); th = solve(G, g, lam); lr += gain(f, th)[4]
    cv[lam] = st.mean(lr)
best = min(cv, key=cv.get)
print('CV (train 2/3, %d nets, grouped 5-fold): mean per-net xMSE vs current head by lambda:' % len(train), {k: round(v, 3) for k, v in cv.items()}, '-> lambda', best)
G, g = normal(train); th = solve(G, g, best)
for name, ss in (('HELD-OUT 1/3', held), ('CONTROLS 24', ctrl), ('ctl_s2tr 7103248-55', [s for s in ctrl if s >= 7103248]), ('ctl_s2tr_a80 7103080-095', [s for s in ctrl if s < 7103200])):
    if ss:
        m, se, agg, lower, lr = gain(ss, th); print(f"{name}: n {len(ss)} xnet {m:+.3f}% (SE {se:.3f}) aggregate {agg:+.3f}% lower {lower}/{len(ss)}")
dth = th - theta0; print('refit change: |d head| / |head| = %.3f, head_b %+.5f -> %+.5f' % (np.linalg.norm(dth[:-1]) / np.linalg.norm(theta0[:-1]), theta0[-1], th[-1]))
np.savez(H/'REFIT_head15_v1.npz', theta=th, theta0=theta0, lam=best, train=np.array(train), held=np.array(held))

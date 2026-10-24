"""[opus-headfit] low-dimensional follow-up to fit_head_v1 (same captures, same train/held split rng 7): in-sample gain of the full
65-parameter refit (overfit check), and 1-2 parameter refits that cannot overfit: y = ynod + a*delta (scale) and y = ynod + a*delta + c*s
(scale + head_b shift). Scored by the noise-free cross-half U per network (mean per-net log ratio, SE, aggregate)."""
import json, math, statistics as st
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; O = H.parents[2]; F = O/'analyst/fresh'
seeds = json.load(open(H/'SEEDS_headfit_v1.json'))['seeds']; CTRL = list(range(7103248, 7103256)) + list(range(7103080, 7103096))
def load(s):
    p = H/f'cap/head_{s}.npz'
    if not p.exists(): return None
    z = np.load(p); t = np.load(F/f'truth_seed{s}.npz'); sc = z['sigma'] * float(np.ravel(z['rms'])[0]) * float(np.ravel(z['gain'])[0])
    X = np.concatenate([z['state'], np.ones((z['state'].shape[0], 1))], axis=1)
    return dict(X=X, sc=sc, ynod=z['y15'] - z['delta'], y15=z['y15'].astype(np.float64), d=z['delta'], A=t['mc_post_A'][-1].astype(np.float64),
                B=t['mc_post_B'][-1].astype(np.float64), T=t['mc_post'][-1].astype(np.float64))
D = {s: d for s in seeds + CTRL if (d := load(s)) is not None}; fresh = [s for s in seeds if s in D]; ctrl = [s for s in CTRL if s in D]
rng = np.random.default_rng(7); order = list(fresh); rng.shuffle(order); ntr = (2 * len(order)) // 3; train, held = sorted(order[:ntr]), sorted(order[ntr:])
z = np.load(H/'REFIT_head15_v1.npz'); assert sorted(z['train'].tolist()) == train, 'split differs from fit_head_v1'
def score(ss, f):
    lr, Un, Ub = [], 0.0, 0.0
    for s in ss:
        d = D[s]; y = f(d); un = float(np.mean((y - d['A']) * (y - d['B']))); ub = float(np.mean((d['y15'] - d['A']) * (d['y15'] - d['B'])))
        lr.append(100 * math.log(un / ub)); Un += un; Ub += ub
    return f"xnet {st.mean(lr):+.3f}% (SE {st.stdev(lr) / math.sqrt(len(lr)):.3f}) agg {100 * math.log(Un / Ub):+.3f}% lower {sum(x < 0 for x in lr)}/{len(lr)}"
full = lambda d: d['ynod'] + d['sc'] * (d['X'] @ z['theta'])
print('full 65-param refit (lambda %g): TRAIN in-sample %s' % (float(z['lam']), score(train, full)))
def lowfit(cols):   # LS on the train 2/3 of y - ynod = sum_k coef_k * col_k
    G = sum(np.stack([c(D[s]) for c in cols], 1).T @ np.stack([c(D[s]) for c in cols], 1) for s in train)
    g = sum(np.stack([c(D[s]) for c in cols], 1).T @ (D[s]['T'] - D[s]['ynod']) for s in train); return np.linalg.solve(G, g)
for name, cols in (('scale a*delta', [lambda d: d['d']]), ('scale + bias a*delta + c*s', [lambda d: d['d'], lambda d: d['sc']])):
    co = lowfit(cols); f = lambda d, co=co, cols=cols: d['ynod'] + sum(k * c(d) for k, c in zip(co, cols))
    print(f'{name}: coef {np.round(co, 4).tolist()} | train {score(train, f)} | HELD {score(held, f)} | CONTROLS24 {score(ctrl, f)}')

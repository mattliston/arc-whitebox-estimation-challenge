"""[opus-l14fit] ridge refit of (head[14], head_b[14]) for the FINAL-layer error. Validated linear propagation: dy15 = Phi(z15) * W15^T dm14,
dm14 = mask * s14 * ([state14, 1] @ dtheta) (mask = unclipped units, s14 = sigma14 rms14 gain14). Target truth15 - y15. Ridge toward the
current head; lambda by network-grouped 5-fold CV inside the TRAIN 2/3 (noise-free cross-half score). Frozen: held-out 1/3 + 24 controls."""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
F = Path(__file__).resolve().parent; O = F.parents[2]; sys.path.insert(0, str(O/'analyst/fresh')); import loader
fresh = json.load(open(O/'lead/runs/opus_headfit_v1/SEEDS_headfit_v1.json'))['seeds'][:100]
CTRL = list(range(7103248, 7103256)) + list(range(7103080, 7103096))
def load(s):
    p = F/f'cap/l14_{s}.npz'
    if not p.exists(): return None
    z = np.load(p); y = z['y']; W = np.asarray(loader.weights(s)[15], dtype=np.float64)
    with np.load(O/f'analyst/fresh/truth_seed{s}.npz') as t: A, B, T = (t[k][-1].astype(np.float64) for k in ('mc_post_A', 'mc_post_B', 'mc_post'))
    s14 = z['sigma14'] * float(np.ravel(z['rms14'])[0]) * float(np.ravel(z['gain14'])[0]) * (z['dm14_raw'] >= -z['mean14_pre'])
    X = np.concatenate([z['state14'], np.ones((1024, 1))], axis=1)
    phi = norm.cdf((y[14] @ W) / np.sqrt(np.maximum(z['v15'], 1e-30)))
    M = phi[:, None] * (W.T @ (s14[:, None] * X))
    return dict(M=M, y15=y[15], A=A, B=B, T=T, theta0=np.append(np.ravel(z['head14']), float(np.ravel(z['head_b14'])[0])))
D = {s: d for s in fresh + CTRL if (d := load(s)) is not None}
fr = [s for s in fresh if s in D]; ct = [s for s in CTRL if s in D]; print('captured fresh', len(fr), 'controls', len(ct))
rng = np.random.default_rng(7); order = list(fr); rng.shuffle(order); ntr = (2 * len(order)) // 3; train, held = sorted(order[:ntr]), sorted(order[ntr:])
def normal(ss):
    G = np.zeros((65, 65)); g = np.zeros(65)
    for s in ss: d = D[s]; G += d['M'].T @ d['M']; g += d['M'].T @ (d['T'] - d['y15'])
    return G, g
def solve(G, g, lam): sc = np.trace(G) / 65; return np.linalg.solve(G + lam * sc * np.eye(65), g)
def score(ss, dth):
    lr, Un, Ub = [], 0.0, 0.0
    for s in ss:
        d = D[s]; y = d['y15'] + d['M'] @ dth; y0 = d['y15']
        un, ub = float(np.mean((y - d['A']) * (y - d['B']))), float(np.mean((y0 - d['A']) * (y0 - d['B'])))
        lr.append(100 * math.log(un / ub)); Un += un; Ub += ub
    return st.mean(lr), (st.stdev(lr) / math.sqrt(len(lr)) if len(lr) > 1 else float('nan')), 100 * math.log(Un / Ub), sum(x < 0 for x in lr), lr
folds = [train[i::5] for i in range(5)]; lams = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0]; cv = {}
for lam in lams:
    lr = []
    for f in folds:
        G, g = normal([s for s in train if s not in f]); lr += score(f, solve(G, g, lam))[4]
    cv[lam] = st.mean(lr)
best = min(cv, key=cv.get); print('CV (train %d nets, grouped 5-fold) per-net xMSE by lambda:' % len(train), {k: round(v, 3) for k, v in cv.items()}, '-> lambda', best)
G, g = normal(train); dth = solve(G, g, best)
for name, ss in (('HELD-OUT 1/3', held), ('CONTROLS 24', ct), ('ctl_s2tr 8', [s for s in ct if s >= 7103248]), ('ctl_s2tr_a80 16', [s for s in ct if s < 7103200])):
    if ss: m, se, agg, lower, _ = score(ss, dth); print(f'{name}: n {len(ss)} xnet {m:+.3f}% (SE {se:.3f}) aggregate {agg:+.3f}% lower {lower}/{len(ss)}')
th0 = D[fr[0]]['theta0']; print('refit change |d head14|/|head14| = %.3f, head_b14 %+.5f -> %+.5f' % (np.linalg.norm(dth[:-1]) / np.linalg.norm(th0[:-1]), th0[-1], th0[-1] + dth[-1]))
np.savez(F/'REFIT_head14_v1.npz', dtheta=dth, theta0=th0, lam=best, train=np.array(train), held=np.array(held))

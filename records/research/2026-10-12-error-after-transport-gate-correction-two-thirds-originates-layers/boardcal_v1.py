"""radlab boardcal_v1 (10-12, lead 22:1xZ): BOARD vs PREDICTION calibration from the graded eval JSONs (public 50 per-network scores,
read-only; numpy single-thread). For each pair of submissions: per-network final-layer MSE and adjusted-score changes (pooled = ratio of sums,
mean of per-network %, sd, se, lower on k of 50; bootstrap se of the pooled change over networks). Same-physics pairs (twin vs safe of one
round) measure the native per-network numerics spread that every twin-vs-twin change also carries.
usage: python -B radlab/boardcal_v1.py"""
import json
from pathlib import Path

import numpy as np

O = Path(__file__).resolve().parents[1]
EV = {335411: 'fire/s48s_20261011/eval_335411.json', 335483: 'fire/s48twin2_20261012/eval_335483.json',
      335505: 'fire/s50twin_20261012/eval_335505.json', 335508: 'fire/s50s_20261012/eval_335508.json',
      335621: 'fire/s51twin_20261012/eval_335621.json', 335618: 'fire/s51s_20261012/eval_335618.json'}
NAME = {335411: 'S48 safe', 335483: 'S48 twin2 (twin v2)', 335505: 'S50 twin', 335508: 'S50 safe', 335621: 'S51 twin', 335618: 'S51 safe'}


def load(sid):
    r = json.loads((O / EV[sid]).read_text())['evaluation']['results']
    pub = {p['mlp_index']: p['scores'] for p in r['public_scores']}
    idx = sorted(pub)
    mse = np.array([pub[i]['final_layer_mse'] for i in idx]); adj = np.array([pub[i]['adjusted_final_layer_score'] for i in idx])
    return idx, mse, adj


def compare(a, b, label):
    ia, ma, aa = load(a); ib, mb, ab = load(b)
    assert ia == ib
    rng = np.random.default_rng(0); out = []
    for name, x, y in (('final MSE', ma, mb), ('adjusted', aa, ab)):
        ch = 100 * (x / y - 1); pooled = 100 * (x.sum() / y.sum() - 1)
        boot = [100 * (x[s].sum() / y[s].sum() - 1) for s in (rng.integers(0, len(x), len(x)) for _ in range(4000))]
        out.append('%s pooled %+.3f%% (bootstrap se %.3f), per-network mean %+.3f%% sd %.3f, lower on %d / ties %d of %d' % (
            name, pooled, np.std(boot), ch.mean(), ch.std(ddof=1), int((x < y).sum()), int((x == y).sum()), len(x)))
    print('%-46s %s\n%-46s %s' % (label, out[0], '', out[1]))
    return 100 * (ma / mb - 1)


print('SAME PHYSICS (numerics only):')
d_tw_s50 = compare(335505, 335508, 'S50 twin vs S50 safe')
d_tw_s51 = compare(335621, 335618, 'S51 twin vs S51 safe')
print('ROUND STEPS (twin vs twin, safe vs safe):')
s50t = compare(335505, 335483, 'S50 twin vs S48 twin2')
s50s = compare(335508, 335411, 'S50 safe vs S48 safe')
s51t = compare(335621, 335505, 'S51 twin vs S50 twin')
s51s = compare(335618, 335508, 'S51 safe vs S50 safe')
print('correlation of per-network changes: S51 twin-step vs S51 safe-step %.2f; S50 twin-step vs S50 safe-step %.2f; S50 step vs S51 step (safes) %.2f' % (
    np.corrcoef(s51t, s51s)[0, 1], np.corrcoef(s50t, s50s)[0, 1], np.corrcoef(s50s, s51s)[0, 1]))
both51 = (s51t + s51s) / 2; both50 = (s50t + s50s) / 2
print('averaged twin+safe step: S51 mean %+.3f%% (se %.3f), S50 mean %+.3f%% (se %.3f)' % (
    both51.mean(), both51.std(ddof=1) / np.sqrt(50), both50.mean(), both50.std(ddof=1) / np.sqrt(50)))

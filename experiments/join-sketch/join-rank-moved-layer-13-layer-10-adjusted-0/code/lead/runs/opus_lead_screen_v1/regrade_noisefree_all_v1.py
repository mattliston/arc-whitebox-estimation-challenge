"""Re-grade every paired native experiment on disk with the truth-noise-free cross-half metric (no model calls).
Pairs, per experiment dir and seed, the ON arm with an in-directory control arm (parent/OFF/reference/C/zero/f043).
Prints per-experiment n, plain MSE log, noise-free MSE log, bill log, plain net, noise-free net (means, %), sorted by noise-free net."""
import glob, json, math, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

O = Path(__file__).resolve().parents[3]
CONTROLS = ('parent', 'OFF', 'reference', 'C', 'zero', 'f043')
bill = lambda f: max(f / 2**41, .1)
truth = {}


def tr(s):
    if s not in truth:
        p = O / f'analyst/fresh/truth_seed{s}.npz'
        if not p.exists():
            truth[s] = None
        else:
            z = np.load(p)
            truth[s] = (z['mc_post'][-1].astype(np.float64), z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64))
    return truth[s]


def arm(pred):
    m = re.match(r'.*/native_(\d+)_(.+)_pred\.npz$', pred)
    return int(m.group(1)), m.group(2)


def load(pred, seed):
    j = Path(pred.replace('_pred.npz', '.json'))
    if not j.exists():
        return None
    try:
        rows = json.loads(j.read_text())['rows']
        row = next(r for r in rows if r['seed'] == seed)
        with np.load(pred) as z:
            y = z[str(seed)].astype(np.float64)[-1]
        return row['flops'], y
    except Exception:
        return None


out = []
for exp in sorted(glob.glob(str(O / 'lead/runs/*'))):
    arms = defaultdict(dict)
    for p in glob.glob(exp + '/run_*/native_71*_*_pred.npz'):
        s, a = arm(p)
        arms[s].setdefault(a, p)
    d = []
    for s, a in arms.items():
        ctl = next((c for c in CONTROLS if c in a), None)
        if 'ON' not in a or ctl is None or tr(s) is None:
            continue
        on, cf = load(a['ON'], s), load(a[ctl], s)
        if not on or not cf:
            continue
        M, A, B = tr(s)
        pm = lambda y: np.mean((y - M) ** 2)
        xm = lambda y: np.mean((y - A) * (y - B))
        if xm(on[1]) <= 0 or xm(cf[1]) <= 0:
            continue
        d.append((100 * math.log(pm(on[1]) / pm(cf[1])), 100 * math.log(xm(on[1]) / xm(cf[1])), 100 * math.log(bill(on[0]) / bill(cf[0]))))
    if len(d) >= 2:
        m, x, b = (st.mean(v[i] for v in d) for i in range(3))
        se = st.stdev([v[1] + v[2] for v in d]) / len(d) ** .5
        out.append((x + b, m + b, len(d), m, x, b, se, Path(exp).name))
print('%-52s %3s %8s %8s %8s %9s %9s %6s' % ('experiment', 'n', 'MSE', 'xMSE', 'bill', 'net', 'xnet', 'xSE'))
for xn, pn, n, m, x, b, se, name in sorted(out):
    flag = ' FLIP' if pn >= 0 > xn else ''
    print('%-52s %3d %+8.3f %+8.3f %+8.3f %+9.3f %+9.3f %6.3f%s' % (name[:52], n, m, x, b, pn, xn, se, flag))

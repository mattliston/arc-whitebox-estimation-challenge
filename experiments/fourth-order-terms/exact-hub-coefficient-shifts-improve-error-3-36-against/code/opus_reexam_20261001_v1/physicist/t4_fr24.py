#!/usr/bin/env python3
"""TAG opus-physicist: the held-out table. Arms from k4l_pred_rank*_{tag}.npz (my copy of the lead's e16b) against the base and F4 arms of
t3_pred_rank*_fresh24.npz (on_0, on_0.4) and, where present, the lead's logs O/e16c_fresh24 (F4 + K4L) and O/e16f_fresh24_grid.
Final-layer MSE against the organizers' truth; pooled change, paired bootstrap over networks.
usage: t4_fr24.py <tag> [<tag> ...]"""
import sys, glob, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; O = HERE.parent; rng = np.random.default_rng(0)
m15 = lambda z, k: float(((z[k][15] - z['truth'][15]) ** 2).mean()); mall = lambda z, k: float(((z[k] - z['truth']) ** 2).mean())
base, f4, arms, alls = {}, {}, {}, {}
for f in glob.glob(str(HERE / 't3_pred_rank*_fresh24.npz')):
    z = np.load(f); r = int(Path(f).name.split('_')[2][4:]); base[r] = m15(z, 'on_0'); f4[r] = m15(z, 'on_0.4'); alls[('base', r)] = mall(z, 'on_0'); alls[('F4 0.4', r)] = mall(z, 'on_0.4')
arms['F4 0.4'] = f4
for tag in sys.argv[1:]:
    for f in glob.glob(str(HERE / f'k4l_pred_rank*_{tag}.npz')):
        z = np.load(f); r = int(Path(f).name.split('_')[2][4:])
        for k in z.files:
            if k != 'truth': arms.setdefault(f'{tag}:{k}', {})[r] = m15(z, k); alls[(f'{tag}:{k}', r)] = mall(z, k)
for d in ('e16c_fresh24', 'e16f_fresh24_grid', 'e16e_fresh24_theta'):
    for f in glob.glob(str(O / d / 'rank*.log')):
        r = int(Path(f).stem[4:])
        for line in open(f):
            if line.startswith('{'):
                j = json.loads(line)
                if j['arm'] != 'base': arms.setdefault(f"lead {d[:4]}:{j['arm']}", {})[r] = j['final_x1e9'] * 1e-9
def report(name, a, ref, refname):
    ranks = sorted(set(a) & set(ref))
    if len(ranks) < 2: return
    b = np.array([ref[r] for r in ranks]); c = np.array([a[r] for r in ranks]); per = c / b - 1
    idx = rng.integers(0, len(b), size=(4000, len(b))); bs = c[idx].sum(1) / b[idx].sum(1) - 1
    al = ''
    if all((name, r) in alls for r in ranks) and all((refname, r) in alls for r in ranks): al = f"  all-layer {100 * (sum(alls[(name, r)] for r in ranks) / sum(alls[(refname, r)] for r in ranks) - 1):+.2f}%"
    print(f"  {name:24s} vs {refname:10s}: {len(ranks):2d} nets  pooled {100 * (c.sum() / b.sum() - 1):+.2f}% [95% {100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}]"
          f"  better {int((per < 0).sum())}/{len(b)}  mean {100 * per.mean():+.2f}% (se {100 * per.std(ddof=1) / len(b) ** 0.5:.2f})  median {100 * np.median(per):+.2f}%  worst {100 * per.max():+.2f}%{al}")
print('against the unchanged card:')
for name, a in sorted(arms.items()): report(name, a, base, 'base')
print('against F4 at strength 0.4 on the same networks:')
for name, a in sorted(arms.items()):
    if name != 'F4 0.4': report(name, a, f4, 'F4 0.4')
for ref in [k for k in arms if k.endswith(':F6K4') or k.endswith(':F4K4a2')]:
    print(f'against {ref}:')
    for name, a in sorted(arms.items()):
        if ':SK' in name or name.endswith(':S'): report(name, a, arms[ref], ref)

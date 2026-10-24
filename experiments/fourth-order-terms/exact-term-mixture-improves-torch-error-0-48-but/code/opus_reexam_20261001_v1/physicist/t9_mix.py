#!/usr/bin/env python3
"""TAG opus-physicist T9: which MIXTURE of the measured changes would be best? Linear model of the final-layer prediction,
   pred(a) = base + sum_k a_k (pred_k - base),
with the coefficients a fitted on all networks (pooled least squares of the final-layer error) and scored LEAVE-ONE-NETWORK-OUT.
The model is only a guide for which closed-loop arm to run next (changes do not add exactly); a mixture counts when its own
closed-loop run confirms it.
usage: t9_mix.py <base spec> <name>=<spec> [<name>=<spec> ...] [-- <subset> ...]
   spec = <tag>:<arm> for t3_pred_rank*_{tag}.npz, k4l/<tag>:<arm> for k4l_pred_rank*_{tag}.npz
   a spec may be a difference 'specA~specB' (e.g. K=k4l/fr24a:SK~k4l/fr24a:S: the change K4L makes on top of S)
   subsets after '--' are comma lists of names to fit together (default: every single name, and all names together)"""
import sys, glob
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent


def load(spec):
    tag, arm = spec.split(':'); pre = 't3_pred'
    if tag.startswith('k4l/'): pre, tag = 'k4l_pred', tag[4:]
    out = {}
    for f in glob.glob(str(HERE / f'{pre}_rank*_{tag}.npz')):
        z = np.load(f)
        if arm in z.files: out[int(Path(f).name.split('_')[2][4:])] = (z[arm][15].astype(np.float64), z['truth'][15].astype(np.float64))
    return out


args = sys.argv[1:]; subsets = []
if '--' in args: i = args.index('--'); subsets = [s.split(',') for s in args[i + 1:]]; args = args[:i]
base = load(args[0]); names, P = [], {}
for a in args[1:]:
    name, spec = a.split('=', 1)
    if '~' in spec:
        sa, sb = spec.split('~'); A, B = load(sa), load(sb); P[name] = {r: A[r][0] - B[r][0] for r in set(A) & set(B)}
    else:
        A = load(spec); P[name] = {r: A[r][0] - base[r][0] for r in set(A) & set(base)}
    names.append(name)
ranks = sorted(set(base).intersection(*[set(P[n]) for n in names])); E = {r: base[r][0] - base[r][1] for r in ranks}
m0 = np.array([float(E[r] @ E[r]) for r in ranks]); print(f"{len(ranks)} networks; base pooled final-layer MSE {m0.sum() / 1024 / len(ranks):.4e}")
if not subsets: subsets = [[n] for n in names] + ([names] if len(names) > 1 else [])
rng = np.random.default_rng(0)


def solve(idx, ns):
    A = sum(np.array([[P[a][ranks[i]] @ P[b][ranks[i]] for b in ns] for a in ns]) for i in idx)
    b = sum(np.array([P[a][ranks[i]] @ E[ranks[i]] for a in ns]) for i in idx)
    return -np.linalg.solve(A, b)


def mse(i, ns, a):
    e = E[ranks[i]] + sum(x * P[n][ranks[i]] for x, n in zip(a, ns)); return float(e @ e)


allidx = list(range(len(ranks)))
for ns in subsets:
    one = np.array([mse(i, ns, np.ones(len(ns))) for i in allidx]); a = solve(allidx, ns); fit = np.array([mse(i, ns, a) for i in allidx])
    loo = np.array([mse(i, ns, solve([j for j in allidx if j != i], ns)) for i in allidx])
    idx = rng.integers(0, len(ranks), size=(4000, len(ranks))); bs = loo[idx].sum(1) / m0[idx].sum(1) - 1
    print(f"{'+'.join(ns):28s} all at 1: {100 * (one.sum() / m0.sum() - 1):+.2f}%   fitted {', '.join(f'{n} {x:.3f}' for n, x in zip(ns, a))}: in sample {100 * (fit.sum() / m0.sum() - 1):+.2f}%, "
          f"left out {100 * (loo.sum() / m0.sum() - 1):+.2f}% [95% {100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}], better on {int((loo < m0).sum())}/{len(ranks)}")
print('cosines between the changes (pooled over networks): ' + '  '.join(
    f"{a}.{b} {sum(P[a][r] @ P[b][r] for r in ranks) / np.sqrt(sum(P[a][r] @ P[a][r] for r in ranks) * sum(P[b][r] @ P[b][r] for r in ranks)):+.2f}" for i, a in enumerate(names) for b in names[i + 1:]))
print('size of each change, |P|^2 / |e0|^2 pooled (%): ' + '  '.join(f"{n} {100 * sum(P[n][r] @ P[n][r] for r in ranks) / m0.sum():.2f}" for n in names))

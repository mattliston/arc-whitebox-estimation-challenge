"""TAG opus-trainer (2026-10-02): the joint optimum of many dials from single-dial FORWARD runs (the physicist's mixture idea,
O/physicist/t9_mix.py, through this lane's pool). The final-layer prediction is close to linear in small dial changes: with P0 the
base prediction and Pk the prediction with arm k applied, P(a) = P0 + sum_k a_k (Pk - P0), so the pooled final-layer MSE is a
quadratic in a that is minimised exactly: no gradient, no per-dial line search, interactions between the dials included.
  queue <run dir> <label> <spec.json> <rank,rank,...>   writes the parameter files of the base and of every arm into the spec's
        directory and queues forward requests that keep the final predictions (ids a_<label>_<arm>_r<rank>, served first)
  solve <run dir> <label> <spec.json> [ridge, default 0.02] [--arms a,b,c --tag name]   pooled least squares (over the named arms only;
        the optimum file then carries the tag) over the networks that have the base and every
        arm; what each arm does alone; the weights at the optimum; the gain the model predicts and its leave-one-network-out
        version (weights fitted without the network they are scored on); writes <spec dir>/<label>_optimum.pt = the base plus the
        weighted arms (to be run for real before anything is claimed)
spec.json: {"request": {extra request fields, e.g. "f4c": 1},
            "base": {"pl": {family: shift | [16 shifts]}, "c_scale": {"<leg>": factor}, "file": "<parameter file>"},
            "arms": {"<name>": {"pl": {...}, "c_scale": {...}}, ...}}     an arm = the base plus its changes; parameters = the parent's,
            or with base "file" the arrays and shifts stored in that file (a candidate), to which the base's own pl / c_scale add
The weights are in units of the arm's own step (1 = the arm as run). A ridge of r adds r x the arm's own energy to its diagonal
(arms whose prediction change is not far above the card's per-network precision noise, about 4% of the error's rms, would
otherwise be fitted to noise)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from otr_common import data_paths, enqueue, read_source, result_path, save_params

MODE, RUN, LABEL, SPEC = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3], Path(sys.argv[4]).resolve()
spec = json.loads(SPEC.read_text())
OUT = SPEC.parent
arrays, dials = read_source(np)
BASE_PL = {}
if 'file' in spec['base']:   # the base is a stored candidate: its arrays and its shifts
    stored = torch.load(spec['base']['file'], map_location='cpu', weights_only=True)
    assert all(float(stored['dials'][key]) == float(value) for key, value in dials.items())
    arrays = {key: value.numpy() for key, value in stored['arrays'].items()}
    BASE_PL = {family: [float(x) for x in values] for family, values in stored.get('pl', {}).items()}


def expand(pl):
    return {family: ([float(x) for x in value] if isinstance(value, list) else [float(value)] * 16) for family, value in pl.items()}


def parameters(weights):
    """The base plus weights[name] x each arm: arrays (cstar legs scaled) and the 16-layer shifts."""
    pl = dict(BASE_PL)
    for family, values in expand(spec['base'].get('pl', {})).items():
        pl[family] = [a + b for a, b in zip(pl.get(family, [0.0] * 16), values)]
    scale = {int(leg): float(f) for leg, f in spec['base'].get('c_scale', {}).items()}
    for name, weight in weights.items():
        arm = spec['arms'][name]
        for family, values in expand(arm.get('pl', {})).items():
            pl[family] = [a + weight * b for a, b in zip(pl.get(family, [0.0] * 16), values)]
        for leg, f in arm.get('c_scale', {}).items():
            scale[int(leg)] = scale.get(int(leg), 1.0) * (1 + weight * (float(f) - 1))
    A = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
    for leg, f in scale.items():
        A['c'][:, leg] *= f
    return A, {family: values for family, values in pl.items() if any(values)}


names = list(spec['arms'])
TAG = ''
if '--arms' in sys.argv:   # solve over a subset of the arms that were run (the others get weight 0)
    names = sys.argv[sys.argv.index('--arms') + 1].split(',')
    TAG = '_' + sys.argv[sys.argv.index('--tag') + 1]
    assert set(names) <= set(spec['arms'])
    sys.argv = [x for i, x in enumerate(sys.argv) if x not in ('--arms', '--tag') and sys.argv[i - 1] not in ('--arms', '--tag')]
if MODE == 'queue':
    ranks = [int(x) for x in sys.argv[5].split(',')]
    for arm in ['base'] + names:
        path = OUT / ('%s_%s.pt' % (LABEL, arm))
        if not path.exists():
            A, pl = parameters({} if arm == 'base' else {arm: 1.0})
            save_params(torch, path, A, dials, pl)
        for rank in ranks:
            enqueue(RUN, dict(id='a_%s_%s_r%04d' % (LABEL, arm, rank), mode='eval', rank=rank, params=str(path), keep_final=1,
                              **spec.get('request', {})))
    print('queued %d forward runs (%d networks x base + %d arms)' % (len(ranks) * (len(names) + 1), len(ranks), len(names)))
    sys.exit()

ridge = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
R = RUN / 'results'
load = lambda arm, rank: json.loads((R / ('a_%s_%s_r%04d.json' % (LABEL, arm, rank))).read_text())
ranks = sorted(int(f.stem.rsplit('_r', 1)[1]) for f in R.glob('a_%s_base_r*.json' % LABEL))
ranks = [r for r in ranks if all((R / ('a_%s_%s_r%04d.json' % (LABEL, arm, r))).exists() for arm in names)]
assert len(ranks) >= 3, 'networks with the base and every arm: %d' % len(ranks)
D, res, alone = [], [], {name: [] for name in names}
for rank in ranks:
    with np.load(data_paths(rank)[1], allow_pickle=False) as saved:
        truth = np.asarray(saved['mc_post'], dtype=np.float64)[15]
    base = load('base', rank)
    p0 = np.array(base['final_prediction'])
    assert abs(((p0 - truth) ** 2).mean() / base['mse'] - 1) < 1e-9
    cols = []
    for name in names:
        row = load(name, rank)
        cols.append(np.array(row['final_prediction']) - p0)
        alone[name].append((row['mse'], base['mse']))
    D.append(np.stack(cols, 1))
    res.append(truth - p0)
base_energy = sum((r ** 2).sum() for r in res)
print('%d networks; base final-layer MSE %.4e; rms of the error %.3e' % (len(ranks), base_energy / (1024 * len(ranks)),
                                                                        (base_energy / (1024 * len(ranks))) ** .5))


def fit(indices):
    G = sum(D[i].T @ D[i] for i in indices)
    h = sum(D[i].T @ res[i] for i in indices)
    return np.linalg.solve(G + ridge * np.diag(np.diag(G)), h)


weights = fit(range(len(ranks)))
model = sum(((res[i] - D[i] @ weights) ** 2).sum() for i in range(len(ranks))) / base_energy - 1
loo = sum(((res[i] - D[i] @ fit([j for j in range(len(ranks)) if j != i])) ** 2).sum() for i in range(len(ranks))) / base_energy - 1
print('%-34s %10s %12s %9s %9s' % ('arm', 'alone %', 'size / error', 'weight', 'spread'))
loo_weights = np.stack([fit([j for j in range(len(ranks)) if j != i]) for i in range(len(ranks))])
for k, name in enumerate(names):
    a = np.array(alone[name])
    size = (sum((D[i][:, k] ** 2).sum() for i in range(len(ranks))) / base_energy) ** .5
    print('%-34s %+10.3f %12.3f %+9.3f %9.3f' % (name, 100 * (a[:, 0].mean() / a[:, 1].mean() - 1), size, weights[k],
                                                loo_weights[:, k].std() * (len(ranks) - 1) ** .5))
print('the model at its optimum: %+.3f%% of the pooled error; leave-one-network-out: %+.3f%% (ridge %g)' % (100 * model, 100 * loo, ridge))
A, pl = parameters(dict(zip(names, (float(w) for w in weights))))
path = OUT / ('%s_optimum%s.pt' % (LABEL, TAG))
if not path.exists():
    save_params(torch, path, A, dials, pl)
    (OUT / ('%s_optimum%s.json' % (LABEL, TAG))).write_text(json.dumps(dict(
        networks=ranks, ridge=ridge, weights=dict(zip(names, (float(w) for w in weights))), model_pct=100 * model, leave_one_out_pct=100 * loo,
        shifts={family: values for family, values in pl.items()}, c=A['c'].tolist()), indent=1))
print('optimum written to', path)

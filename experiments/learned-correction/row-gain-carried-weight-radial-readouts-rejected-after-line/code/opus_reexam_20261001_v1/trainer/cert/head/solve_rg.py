"""TAG opus-trainer (2026-10-02): solve of the weights of one zero-weight READOUT (batch 4: 'row' = row gain of the bank read, 'b' = carried b,
'rad' = radial; otr_m.py comment READOUTS) by measured steps along minus the mean of exact network gradients, on the S42 arithmetic in torch.
  direction <gradient label> <readout[,readout]> <out.json> [tmin, default 0]
        mean of 'rg_grads' over the training networks; weights whose mean is within tmin standard errors of zero do not move; unit length
  queue <name> <direction.json> <eta,eta,...> <ranks file> [<current weights.json>]      forward runs, ids a_0r<name>_eta<k>_r<rank> (eta 0 = given label in read)
  judge <name> <weights.json> <ranks file>                                               forward runs at the weights, ids a_0r<name>j_r<rank>
  read <name> <direction.json> <eta,eta,...> <ranks file> <base label> [<current weights.json>]
        pooled change against the base label, parabola, minimum; writes cert/head/rg_<name>_weights.json (dict by readout)
Card of every run: S39 card, law with the table and three forms (module n6y, dfit 2), first dials (cert/fold/c3q4zd6_optimum.pt), k31l ages 1-2, the head's
diagonal column at step 2's weights (cert/head/v2r_weights.json), readouts on.
usage (from O/trainer): python -B cert/head/solve_rg.py <direction|queue|judge|read> ..."""
import json
import sys
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(T))
R = T / 'run_M/results'
SHAPES = dict(row=(2, 52), b=(16, 64), rad=(16, 64))
BASE = str(T / 'cert/fold/c3q4zd6_optimum.pt')
K = [dict(module='k31ly', age=1, k31=0.0, leaf='nkl1'), dict(module='k31ly', age=2, k31=1.0, leaf='nkl2'), dict(module='k31ly', age=3, k31=0.0, leaf='nkl3')]
REQUEST = dict(f4c=1, ss=0, hf=2, rg=1, ncar=[dict(module='n6y', dfit=2)] + K, hu=json.loads((T / 'cert/head/v2r_weights.json').read_text()))
mode = sys.argv[1]


def weights(path):
    W = {name: np.zeros(shape) for name, shape in SHAPES.items()}
    if path:
        for name, value in json.loads(Path(path).read_text()).items():
            W[name] = np.array(value)
    return W


def moved(W0, direction, eta):
    W = {name: value.copy() for name, value in W0.items()}
    for name, d in direction['direction'].items():
        W[name] = W[name] + eta * np.array(d).reshape(SHAPES[name])
    return {name: value.tolist() for name, value in W.items()}


if mode == 'direction':
    import torch
    label, names, out = sys.argv[2], sys.argv[3].split(','), Path(sys.argv[4])
    tmin = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    rows = [torch.load(f, map_location='cpu', weights_only=False) for f in sorted(R.glob('a_%s_r*.pt' % label))]
    rows = [r for r in rows if 'rg_grads' in r]
    N = len(rows); mse = float(np.mean([r['mse'] for r in rows]))
    G = np.stack([np.concatenate([r['rg_grads'][n].double().numpy().reshape(-1) / r['scale'] for n in names]) for r in rows])
    m, se = G.mean(0), G.std(0, ddof=1) / N ** .5
    t = np.where(se > 0, m / np.maximum(se, 1e-300), 0.0)
    keep = np.abs(t) >= tmin
    d = -m * keep
    length = float(np.sqrt((d * d).sum())); left = max(float(((m * m - se * se) * keep).sum()), 0.0) ** .5
    a, b = G[0::2].mean(0) * keep, G[1::2].mean(0) * keep
    direction, at = {}, 0
    for n in names:
        size = int(np.prod(SHAPES[n])); direction[n] = (d[at:at + size] / max(length, 1e-300)).tolist(); at += size
    note = dict(networks=N, readouts=names, tmin=tmin, moving=int(keep.sum()), of=int(keep.size), pooled_mse=mse, direction=direction, beyond_3=int((np.abs(t) > 3).sum()),
                halves_cosine=float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-300)),
                first_order_pct_per_unit_step=-100 * length / mse, first_order_pct_per_unit_step_noise_out=-100 * left / mse)
    out.write_text(json.dumps(note, indent=1))
    print(json.dumps({k: v for k, v in note.items() if k != 'direction'}))
elif mode in ('queue', 'judge'):
    from otr_common import enqueue
    name = sys.argv[2]
    if mode == 'judge':
        W, ranks = weights(sys.argv[3]), [int(x) for x in Path(sys.argv[4]).read_text().split(',')]
        print('queued %d forward runs at the weights' % sum(enqueue(T / 'run_M', dict(id='a_0r%sj_r%04d' % (name, r), mode='eval', rank=r, params=BASE,
                                                                                    rgw={k: v.tolist() for k, v in W.items()}, **REQUEST)) for r in ranks))
    else:
        direction, etas, ranks = json.loads(Path(sys.argv[3]).read_text()), [float(x) for x in sys.argv[4].split(',')], [int(x) for x in Path(sys.argv[5]).read_text().split(',')]
        W0 = weights(sys.argv[6] if len(sys.argv) > 6 else None); n = 0
        for k, eta in enumerate(etas, 1):
            n += sum(enqueue(T / 'run_M', dict(id='a_0r%s_eta%d_r%04d' % (name, k, r), mode='eval', rank=r, params=BASE, rgw=moved(W0, direction, eta), **REQUEST)) for r in ranks)
        print('queued %d forward runs (%d step sizes x %d ranks)' % (n, len(etas), len(ranks)))
else:
    name, direction, etas, ranks, base_label = sys.argv[2], json.loads(Path(sys.argv[3]).read_text()), [float(x) for x in sys.argv[4].split(',')], \
        [int(x) for x in Path(sys.argv[5]).read_text().split(',')], sys.argv[6]
    W0 = weights(sys.argv[7] if len(sys.argv) > 7 else None)
    L = lambda label, r: json.loads((R / ('a_%s_r%04d.json' % (label, r))).read_text())
    xs, ys = [0.0], [0.0]
    for k, eta in enumerate(etas, 1):
        rs = [r for r in ranks if (R / ('a_0r%s_eta%d_r%04d.json' % (name, k, r))).exists() and (R / ('a_%s_r%04d.json' % (base_label, r))).exists()]
        if len(rs) < 2:
            continue
        a = np.array([L('0r%s_eta%d' % (name, k), r)['mse'] for r in rs]); b = np.array([L(base_label, r)['mse'] for r in rs]); ch = 100 * (a / b - 1)
        la = np.array([np.mean(L('0r%s_eta%d' % (name, k), r)['mse_layers']) for r in rs]); lb = np.array([np.mean(L(base_label, r)['mse_layers']) for r in rs])
        pooled = 100 * (a.mean() / b.mean() - 1)
        print('eta %.4f (first order %+.2f%%): %2d ranks, pooled %+.3f%% (se %.3f, better on %d/%d; all layers %+.3f%%)' % (
            eta, eta * direction['first_order_pct_per_unit_step_noise_out'], len(rs), pooled, ch.std(ddof=1) / len(rs) ** .5, int((a < b).sum()), len(rs), 100 * (la.mean() / lb.mean() - 1)))
        xs.append(eta); ys.append(pooled)
    if len(xs) >= 3:
        c = np.polyfit(xs, ys, 2)
        best = -c[1] / (2 * c[0]) if c[0] > 0 else max(xs)
        best = float(min(max(best, 0.0), 1.5 * max(xs)))
        print('parabola: minimum at eta %.4g, %+.3f%% (curvature %+.3g)' % (best, np.polyval(c, best), c[0]))
        (T / 'cert/head' / ('rg_%s_weights.json' % name)).write_text(json.dumps(moved(W0, direction, best)))
        print('weights written to cert/head/rg_%s_weights.json' % name)

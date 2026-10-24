"""TAG opus-trainer (2026-10-02): solve of the 128 first-layer weights of ONE new head-input column (lead 18:42Z: column 0, the unit-specific part of the
law matrix's diagonal) by steepest descent with measured steps: the mean of exact network gradients gives the direction, forward runs on the 16
line-search ranks give the step, a fresh mean gradient at the new point gives the next direction.
  direction <gradient label[,label]> <column[,column...]> <out.json>
        (several columns: ONE direction over all their weights, minus the mean gradient of each, unit length over all of them together)
        mean gradient of the column's 128 weights over the training networks whose results (run_M/results/a_<label>_r*.pt) carry 'hu_grads'; writes the
        unit direction d = -mean / |mean|, its t statistics and the first-order change per unit step (noise share of the length taken out)
  queue <name> <base params.pt> <direction.json> <eta,eta,...> <ranks file> [<current weights.json>]
        forward runs with the weights = current + eta x d (ids a_0h<name>_eta<k>_r<rank>, served first); eta 0 is not queued (give the base label to read)
  grads <name> <base params.pt> <weights.json | zero> [judge <ranks file>]
        exact gradient runs at the given weights on the 36 training networks of the step tests (ids a_0h<name>g_r<rank>, mode grad, scale 1e8);
        with `judge <ranks file>` forward runs at those weights on those ranks instead (ids a_0h<name>j_r<rank>)
  read <name> <direction.json> <eta,eta,...> <ranks file> <base label> [<current weights.json>]
        pooled change of the final-layer MSE of each eta against the base label on the same ranks, the parabola through them and its minimum;
        writes cert/head/<name>_weights.json = current + eta* x d (6 x 128; the other columns' rows stay as they are)
The request fields of the forward runs are those of REQUEST below (S39 card, law with the constant regressor and three forms through the state-exposing
copy, k31l as three instances, head inputs on); with the environment variable HEAD_CARD=s42 the law takes the TABLE form (the fired files' arithmetic).
usage (from O/trainer): python -B cert/head/solve.py <direction|queue|read> ..."""
import json
import sys
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(T))
R = T / 'run_M/results'
# since 19:10Z the state-exposing copies n6y / k31ly (certified bit-identical): the gradient runs then also carry the second batch of head inputs at zero weights
REQUEST = dict(f4c=1, ss=0, hf=2, ncar=[dict(module='n6y'), dict(module='k31ly', age=1, k31=0.0, leaf='nkl1'), dict(module='k31ly', age=2, k31=1.0, leaf='nkl2'),
                                        dict(module='k31ly', age=3, k31=0.0, leaf='nkl3')])
import os
if os.environ.get('HEAD_CARD') == 's42':      # the S41 / S42 arithmetic in torch: the law with the TABLE (dfit 2); the base parameter file is then the FIRST dials (cert/fold/c3q4zd6_optimum.pt)
    REQUEST = dict(REQUEST, ncar=[dict(module='n6y', dfit=2)] + REQUEST['ncar'][1:])
mode = sys.argv[1]
ROWS = 17


def weights(path):
    U = np.zeros((ROWS, 128))
    if path:
        given = np.array(json.loads(Path(path).read_text())); U[:given.shape[0]] = given
    return U


def step(direction):      # the direction as a ROWS x 128 array (one or several columns)
    D = np.zeros((ROWS, 128))
    if 'direction_rows' in direction:
        for column, row in direction['direction_rows'].items():
            D[int(column)] = row
    else:
        D[direction['column']] = direction['direction']
    return D

if mode == 'direction':
    import torch
    labels, columns, out = sys.argv[2].split(','), [int(x) for x in sys.argv[3].split(',')], Path(sys.argv[4])
    column = columns[0]
    rows = [torch.load(f, map_location='cpu', weights_only=False) for label in labels for f in sorted(R.glob('a_%s_r*.pt' % label))]
    rows = [r for r in rows if 'hu_grads' in r and r['hu_grads'].shape[0] > max(columns)]
    N = len(rows); mse = float(np.mean([r['mse'] for r in rows]))
    G = np.stack([np.concatenate([r['hu_grads'][c].double().numpy() / r['scale'] for c in columns]) for r in rows])
    m, se = G.mean(0), G.std(0, ddof=1) / N ** .5
    t = m / np.maximum(se, 1e-300)
    a, b = G[0::2].mean(0), G[1::2].mean(0)
    length = float(np.sqrt((m * m).sum())); left = max(float((m * m).sum() - (se * se).sum()), 0.0) ** .5
    note = dict(networks=N, column=column, columns=columns, pooled_mse=mse, direction=(-m / length)[:128].tolist(),
                direction_rows={str(c): (-m / length)[128 * i:128 * (i + 1)].tolist() for i, c in enumerate(columns)}, beyond_3=int((np.abs(t) > 3).sum()), mean_t2=float((t * t).mean()),
                halves_cosine=float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())), first_order_pct_per_unit_step=-100 * length / mse,
                first_order_pct_per_unit_step_noise_out=-100 * left / mse, gradient_files=[len(rows)])
    out.write_text(json.dumps(note, indent=1))
    print(json.dumps({k: v for k, v in note.items() if k not in ('direction', 'direction_rows')}))
elif mode == 'queue':
    from otr_common import enqueue
    name, base, direction, etas, ranks = sys.argv[2], str(Path(sys.argv[3]).resolve()), json.loads(Path(sys.argv[4]).read_text()), [float(x) for x in sys.argv[5].split(',')], \
        [int(x) for x in Path(sys.argv[6]).read_text().split(',')]
    U0 = weights(sys.argv[7] if len(sys.argv) > 7 else None); D = step(direction); n = 0
    for k, eta in enumerate(etas, 1):
        U = U0 + eta * D
        n += sum(enqueue(T / 'run_M', dict(id='a_0h%s_eta%d_r%04d' % (name, k, r), mode='eval', rank=r, params=base, hu=U.tolist(), **REQUEST)) for r in ranks)
    print('queued %d forward runs (%d step sizes x %d ranks)' % (n, len(etas), len(ranks)))
elif mode == 'grads':
    from otr_common import enqueue
    name, base, U = sys.argv[2], str(Path(sys.argv[3]).resolve()), weights(None if sys.argv[4] == 'zero' else sys.argv[4])
    if len(sys.argv) > 5 and sys.argv[5] == 'judge':
        ranks = [int(x) for x in Path(sys.argv[6]).read_text().split(',')]
        print('queued %d forward runs at the weights' % sum(enqueue(T / 'run_M', dict(id='a_0h%sj_r%04d' % (name, r), mode='eval', rank=r, params=base, hu=U.tolist(), **REQUEST)) for r in ranks))
    else:
        TRAIN = [597, 165, 667, 25, 309, 65, 642, 125, 164, 288, 297, 330, 124, 662, 603, 50, 182, 323, 381, 90, 201, 208, 17, 563, 421, 154, 85, 594, 411, 115, 605, 678, 384, 647, 91, 60]
        print('queued %d gradient runs at the weights' % sum(enqueue(T / 'run_M', dict(id='a_0h%sg_r%04d' % (name, r), mode='grad', rank=r, scale=1e8, params=base, hu=U.tolist(), **REQUEST)) for r in TRAIN))
else:
    name, direction, etas, ranks, base_label = sys.argv[2], json.loads(Path(sys.argv[3]).read_text()), [float(x) for x in sys.argv[4].split(',')], \
        [int(x) for x in Path(sys.argv[5]).read_text().split(',')], sys.argv[6]
    U0 = weights(sys.argv[7] if len(sys.argv) > 7 else None)
    L = lambda label, r: json.loads((R / ('a_%s_r%04d.json' % (label, r))).read_text())
    xs, ys = [0.0], [0.0]
    for k, eta in enumerate(etas, 1):
        rs = [r for r in ranks if (R / ('a_0h%s_eta%d_r%04d.json' % (name, k, r))).exists() and (R / ('a_%s_r%04d.json' % (base_label, r))).exists()]
        if len(rs) < 2:
            continue
        a = np.array([L('0h%s_eta%d' % (name, k), r)['mse'] for r in rs]); b = np.array([L(base_label, r)['mse'] for r in rs]); ch = 100 * (a / b - 1)
        la = np.array([np.mean(L('0h%s_eta%d' % (name, k), r)['mse_layers']) for r in rs]); lb = np.array([np.mean(L(base_label, r)['mse_layers']) for r in rs])
        pooled = 100 * (a.mean() / b.mean() - 1)
        print('eta %.3f (first order %+.2f%%): %2d ranks, pooled %+.3f%% (se %.3f, better on %d/%d; all layers %+.3f%%)' % (
            eta, eta * direction['first_order_pct_per_unit_step_noise_out'], len(rs), pooled, ch.std(ddof=1) / len(rs) ** .5, int((a < b).sum()), len(rs), 100 * (la.mean() / lb.mean() - 1)))
        xs.append(eta); ys.append(pooled)
    if len(xs) >= 3:
        c = np.polyfit(xs, ys, 2)
        best = -c[1] / (2 * c[0]) if c[0] > 0 else max(xs)
        best = float(min(max(best, 0.0), 1.5 * max(xs)))
        print('parabola: minimum at eta %.4g, %+.3f%% (curvature %+.3g)' % (best, np.polyval(c, best), c[0]))
        U = U0 + best * step(direction)
        (T / 'cert/head' / ('%s_weights.json' % name)).write_text(json.dumps(U.tolist()))
        print('weights written to cert/head/%s_weights.json (length per column: %s)' % (name, ', '.join('%d: %.4f' % (c, float(np.sqrt((U[c] ** 2).sum()))) for c in range(ROWS) if U[c].any())))

"""TAG opus-trainer (2026-10-10): ONE JOINT SOLVE FOR S43 over small real items (lead 10-10 02:11Z item 3), on the S42 arithmetic in torch, by the mixture
method: one forward run per item at a trial size with the final predictions kept, on the 16 line-search ranks; the final-layer prediction is close to
linear in each item's size, so P(a) = P0 + sum_k a_k (P_k - P0) and the pooled final-layer MSE is a quadratic in a, minimised exactly (ridge 0.02 of each
arm's own energy). Then the combined candidate is run for real on the 16 ranks and judged ONCE on the 50 judging ranks against S42's arithmetic.
BASE = S42's arithmetic: parameters cert/fold/c3q4zd6_optimum.pt (first dials), law with the table and three forms (module n6y, dfit 2), k31l three instances
(ages 1-2 at strength 1), the diagonal head column at step 2's weights (cert/head/v2r_weights.json), head inputs on (hf 2).
ARMS (each a change of the base; a = 1 is the arm as run):
  bsk    the physicist's birth weight with the skew-corrected kink density (module pbase, bsk 1.0)
  arr    the learned arrays one step along the mean of 36 exact gradients at S42's point (cert/law_ls3: elements beyond 2.5 standard errors), eta 0.0025
  age3   k31l on the cohorts of age 3 at strength 1 (leaves nkl3 +1, nkl2 -1: ages 1-2 stay at 1)
  f15    the final-layer head columns at a step along minus their mean gradient (cert/s43/f15_weights.json), if they pass their table
usage (from O/trainer): python -B cert/s43/mix.py queue <arm,arm,...>  |  solve <arm,arm,...> [ridge]  |  candidate <weights json> <name>  |  judge <name> <ranks file> <label>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

T = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(T))
from otr_common import data_paths, enqueue, load_params, load_pl, save_params

RUN, OUT = T / 'run_M', T / 'cert/s43'
BASE = T / 'cert/fold/c3q4zd6_optimum.pt'
K = [dict(module='k31ly', age=1, k31=0.0, leaf='nkl1'), dict(module='k31ly', age=2, k31=1.0, leaf='nkl2'), dict(module='k31ly', age=3, k31=0.0, leaf='nkl3')]
U2 = np.array(json.loads((T / 'cert/head/v2r_weights.json').read_text()))
REQUEST = dict(f4c=1, ss=0, hf=2, ncar=[dict(module='n6y', dfit=2)] + K)
LS16 = [int(x) for x in (T / 'cert/ls16_ranks.txt').read_text().split(',')]
ETA_ARR = 0.0025


def full_u(U):
    out = np.zeros((21, 128)); out[:U.shape[0]] = U
    return out


def build(a):
    """(parameter file, request fields) for item sizes a (dict arm -> size); the base for a = {}."""
    arrays, dials = load_params(torch, BASE); pl = {k: list(v) for k, v in load_pl(torch, BASE).items()}
    if a.get('arr'):
        e0, _ = load_params(torch, T / 'cert/law_ls3/eta_0.pt'); e1, _ = load_params(torch, T / 'cert/law_ls3/eta_1.pt')
        eta1 = json.loads((T / 'cert/law_ls3/prepare.json').read_text())['etas'][1]
        arrays = {k: (v.double() + a['arr'] * ETA_ARR / eta1 * (e1[k].double() - e0[k].double())).to(torch.float32) for k, v in arrays.items()}
    if a.get('age3'):
        for fam, sign in (('nkl3', 1.0), ('nkl2', -1.0)):
            pl[fam] = [x + sign * a['age3'] for x in pl.get(fam, [0.0] * 16)]
    request = dict(REQUEST, ncar=list(REQUEST['ncar']))
    if a.get('bsk'):
        request['ncar'] = request['ncar'] + [dict(module='pbase', bsk=float(a['bsk']))]
    U = full_u(U2)
    if a.get('f15'):
        D = np.array(json.loads((OUT / 'f15_step.json').read_text()))
        U = U + a['f15'] * full_u(D)
    request['hu'] = U.tolist()
    return arrays, dials, pl, request


def params_file(name, a):
    arrays, dials, pl, request = build(a)
    path = OUT / ('%s.pt' % name)
    if not path.exists():
        save_params(torch, path, arrays, dials, {k: v for k, v in pl.items() if any(v)})
    return str(path), request


mode = sys.argv[1]
if mode == 'queue':
    arms = sys.argv[2].split(',')
    n = 0
    for arm in ['base'] + arms:
        path, request = params_file('mix_%s' % arm, {} if arm == 'base' else {arm: 1.0})
        n += sum(enqueue(RUN, dict(id='a_0s43_%s_r%04d' % (arm, r), mode='eval', rank=r, params=path, keep_final=1, **request)) for r in LS16)
    print('queued %d forward runs' % n)
elif mode == 'solve':
    arms = sys.argv[2].split(','); ridge = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    L = lambda arm, r: json.loads((RUN / 'results' / ('a_0s43_%s_r%04d.json' % (arm, r))).read_text())
    D, res, alone = [], [], {arm: [] for arm in arms}
    for r in LS16:
        with np.load(data_paths(r)[1], allow_pickle=False) as saved:
            truth = np.asarray(saved['mc_post'], dtype=np.float64)[15]
        b = L('base', r); p0 = np.array(b['final_prediction'])
        cols = []
        for arm in arms:
            row = L(arm, r); cols.append(np.array(row['final_prediction']) - p0); alone[arm].append((row['mse'], b['mse']))
        D.append(np.stack(cols, 1)); res.append(truth - p0)
    E0 = sum((x ** 2).sum() for x in res)
    fit = lambda idx: np.linalg.solve(sum(D[i].T @ D[i] for i in idx) + ridge * np.diag(np.diag(sum(D[i].T @ D[i] for i in idx))), sum(D[i].T @ res[i] for i in idx))
    w = fit(range(len(LS16)))
    model = sum(((res[i] - D[i] @ w) ** 2).sum() for i in range(len(LS16))) / E0 - 1
    loo = sum(((res[i] - D[i] @ fit([j for j in range(len(LS16)) if j != i])) ** 2).sum() for i in range(len(LS16))) / E0 - 1
    for k, arm in enumerate(arms):
        x = np.array(alone[arm]); ch = 100 * (x[:, 0] / x[:, 1] - 1)
        print('%-6s alone %+.3f%% (se %.3f, better on %d/16)   weight %+.3f' % (arm, 100 * (x[:, 0].mean() / x[:, 1].mean() - 1), ch.std(ddof=1) / 4, int((x[:, 0] < x[:, 1]).sum()), w[k]))
    print('model at its optimum %+.3f%%, leave-one-network-out %+.3f%% (ridge %g)' % (100 * model, 100 * loo, ridge))
    (OUT / 'solve.json').write_text(json.dumps(dict(arms=arms, weights=dict(zip(arms, map(float, w))), model_pct=100 * model, loo_pct=100 * loo, ridge=ridge), indent=1))
elif mode == 'candidate':
    a = json.loads(Path(sys.argv[2]).read_text()); name = sys.argv[3]
    a = a.get('weights', a)
    path, request = params_file(name, a)
    (OUT / ('%s_request.json' % name)).write_text(json.dumps(dict(params=path, **request)))
    print('candidate', name, 'sizes', a, '->', path)
else:      # judge <name> <ranks file> <label>: forward runs of a candidate (keep_final off)
    name, ranks, label = sys.argv[2], [int(x) for x in Path(sys.argv[3]).read_text().split(',')], sys.argv[4]
    req = json.loads((OUT / ('%s_request.json' % name)).read_text())
    print('queued %d' % sum(enqueue(RUN, dict(id='a_%s_r%04d' % (label, r), mode='eval', rank=r, **req)) for r in ranks))

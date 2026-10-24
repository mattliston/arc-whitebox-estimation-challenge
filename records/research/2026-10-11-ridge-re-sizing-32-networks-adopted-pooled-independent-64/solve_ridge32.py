"""TAG opus-trainer Nd (lead 10-11 03:5xZ): re-size Nc's 12 arms (same 96 gradients, same directions: cert/refit/jmNc_d1) on 32 generated networks,
7100136-7100151 (Nc's sizing evaluations, run_M/results a_0mNc1_<arm>) + 7100152-7100167 (run_Nd/results a_0mNd1_<arm>), by ridge least squares on
the pooled final-layer residuals: minimise sum_i |truth_i - p0_i - D_i w|^2 + lam sum_k A_kk w_k^2 (A = sum_i D_i' D_i, so lam is relative to each
arm's own energy, as Nc's fixed 0.02). lam is chosen by leave-one-network-out over the 32 on a fixed grid; the point at that lam is the ONE frozen
point (written with publish: never overwritten). Also written, as diagnostics only: the half step, and the leave-one-out of each arm group alone and of
the full set without each group. Every record is checked against the truth plan (A27), Nc's pinned cut simulator and the arm's own parameter file.
usage (from O/trainer): python -B cert/Nd/solve_ridge32.py <out json>"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
from otr_common import publish
from otr_generated_manifest_v1 import TruthPlan
from otr_truth_records_v1 import validate_record

A27 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/a27_fast_truth_20261011T0023Z/plan.json'
A27_SHA = '71f455d1d7f5f7239503468cacd5710f4db9d595c7f1be389c8d937e4f64a979'
RANKS = list(range(7100136, 7100168))
PLAN = TruthPlan(A27, A27_SHA, RANKS)
NC = json.loads(Path('cert/refit/jmNc_d1/arms.json').read_text())
ARMS = NC['live']
CUTSIM = json.loads(json.loads(Path('cert/cut_refit_Nc_launch_v1/driver_handle.json').read_text())['environment']['JM_REQ'])['cutsim']
GROUPS = dict(arrays=['arrp', 'arre'], head=['h6', 'hb3', 'h20', 'h17'], kappa=['kce', 'kcl'], dials=['dX1', 'dbscale', 'dc0', 'dk31'])
GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT = Path(sys.argv[1])
assert not OUT.exists(), OUT


def record(arm, r):
    run, label = ('run_M', '0mNc1') if r <= 7100151 else ('run_Nd', '0mNd1')
    path = Path(run) / 'results' / ('a_%s_%s_r%d.json' % (label, arm, r))
    value = json.loads(path.read_text())
    assert value.get('cutsim') == CUTSIM, ('simulator mismatch', path)
    validate_record(PLAN, value, r)
    expect = 'cert/refit/jmI_p2_noAS.pt' if arm == 'base' else 'cert/refit/jmNc_d1/arm_%s.pt' % arm
    assert Path(value['params']).resolve() == Path(expect).resolve(), ('params mismatch', path, value['params'])
    assert 'final_prediction' in value, path
    return value, str(path), sha(path)


D, res, inputs, alone = [], [], [], {a: [] for a in ARMS}
for r in RANKS:
    entry, expected, _ = PLAN.resolve(r)
    assert sha(entry['path']) == expected
    with np.load(entry['path'], allow_pickle=False) as z:
        truth = np.asarray(z['mc_post'], dtype=np.float64)[15]
    b, bpath, bsha = record('base', r); p0 = np.array(b['final_prediction'])
    cols = []
    inputs.append(dict(rank=r, arm='base', path=bpath, sha256=bsha))
    for a in ARMS:
        v, vpath, vsha = record(a, r); inputs.append(dict(rank=r, arm=a, path=vpath, sha256=vsha))
        cols.append(np.array(v['final_prediction']) - p0); alone[a].append((v['mse'], b['mse']))
    D.append(np.stack(cols, 1)); res.append(truth - p0)
N = len(RANKS); E0 = sum(float((x ** 2).sum()) for x in res)
AA = [d.T @ d for d in D]; BB = [d.T @ x for d, x in zip(D, res)]
A_all, B_all = sum(AA), sum(BB)


def fit(idx_cols, lam, leave=None):
    A = A_all - (AA[leave] if leave is not None else 0); B = B_all - (BB[leave] if leave is not None else 0)
    A = A[np.ix_(idx_cols, idx_cols)]; B = B[idx_cols]
    return np.linalg.solve(A + lam * np.diag(np.diag(A)), B)


def err(idx_cols, w, i):
    return float(((res[i] - D[i][:, idx_cols] @ w) ** 2).sum())


def loo(idx_cols, lam):
    return sum(err(idx_cols, fit(idx_cols, lam, i), i) for i in range(N)) / E0 - 1


allc = list(range(len(ARMS)))
table = {str(lam): 100 * loo(allc, lam) for lam in GRID}
lam = min(GRID, key=lambda x: table[str(x)])
w = fit(allc, lam)
model = sum(err(allc, w, i) for i in range(N)) / E0 - 1
half = sum(err(allc, w / 2, i) for i in range(N)) / E0 - 1
groups = {}
for g, members in GROUPS.items():
    cols = [ARMS.index(a) for a in members]
    rest = [k for k in allc if k not in cols]
    groups[g] = dict(alone_loo_pct=100 * min(loo(cols, x) for x in GRID), without_loo_pct=100 * min(loo(rest, x) for x in GRID))
per_arm = {a: dict(alone_pct=100 * (np.mean([x[0] for x in alone[a]]) / np.mean([x[1] for x in alone[a]]) - 1),
                   loo_alone_pct=100 * min(loo([ARMS.index(a)], x) for x in GRID)) for a in ARMS}
nc = json.loads(Path('cert/refit/jmNc_d1/solve.json').read_text())['weights']
result = dict(schema='trainer_Nd_ridge32_v1', utc=os.popen('date -u +%FT%TZ').read().strip(), networks=RANKS, arms=ARMS, ridge_grid_loo_pct=table,
              ridge=lam, weights=dict(zip(ARMS, map(float, w))), model_pct=100 * model, loo_pct=table[str(lam)], half_step_model_pct=100 * half,
              nc_weights_on16=nc, groups=groups, per_arm=per_arm, cutsim=CUTSIM, truth_plan=PLAN.spec(), inputs=inputs,
              direction=dict(path='cert/refit/jmNc_d1/direction.pt', sha256=sha('cert/refit/jmNc_d1/direction.pt')))
publish(OUT, lambda stream: json.dump(result, stream, indent=1))
print('ridge LOO by lam (%%): %s' % ', '.join('%s: %+.3f' % (k, v) for k, v in table.items()))
print('chosen lam %s: model %+.3f%%, LOO %+.3f%%, half step %+.3f%%' % (lam, 100 * model, table[str(lam)], 100 * half))
print('weights:', ', '.join('%s %+.3f (Nc %+.3f)' % (a, x, nc.get(a, 0.0)) for a, x in zip(ARMS, w)))
for g, v in groups.items():
    print('group %-7s alone LOO %+.3f%%, full set without it LOO %+.3f%%' % (g, v['alone_loo_pct'], v['without_loo_pct']))

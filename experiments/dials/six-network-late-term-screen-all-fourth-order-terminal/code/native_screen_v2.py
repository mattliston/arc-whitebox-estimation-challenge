"""TAG fable-lead 2026-10-20: native_screen_v1 with settings from a JSON FILE (large literals) and PE34 keys restored to int; dict settings may
carry '_GNN_LITERAL' (tuples restored) and '_GNN_PARAMS': None (forces re-unpack). usage: native_screen_v2.py <candidate> <ranks> <settings.json>"""
import json, sys, time, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'); P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
candidate, ranks, settings = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], json.load(open(sys.argv[3]))
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
BASE = {k: getattr(mod, k) for s in settings for k in s}
def coerce(k, v):
    if k == '_PE34_BY_LAYER': return {int(a): tuple(b) for a, b in v.items()}
    if k == '_GNN_LITERAL': return {a: (tuple(b[0]), b[1]) for a, b in v.items()}
    if k == '_K31R_TABLE': return {int(a): tuple(b) for a, b in v.items()}
    return v
class MLP:
    def __init__(self, weights, seed): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'store-%d' % seed
rows = []
for rank in ranks:
    W = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'); W = [np.asarray(W[l]) for l in range(16)]
    tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    res = []
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:      # warm-up: the first predict of a process pays the lazy binds (+0.25% bill); discard it
        est.predict(MLP(W, rank), 2**41)
    for st in settings:
        for k, v in BASE.items(): setattr(mod, k, v)
        if any(k == '_GNN_LITERAL' for k in st): mod._GNN_PARAMS = None
        for k, v in st.items(): setattr(mod, k, coerce(k, v))
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP(W, rank), 2**41), dtype=np.float64)
        if any(k == '_GNN_LITERAL' for k in st): mod._GNN_PARAMS = None
        y15 = y[15] if y.ndim == 2 else y; mse = float(np.mean((y15 - tp[15]) ** 2)); util = float(ctx.flops_used) / 2 ** 41; res.append((mse, util, mse * max(0.1, util)))
    rows.append(res); print('rank %4d base %.4e util %.5f | %s' % (rank, res[0][0], res[0][1], ' | '.join('S%d %+.2f/%+.2f/%+.2f' % (i, 100 * (r[0] / res[0][0] - 1), 100 * (r[1] / res[0][1] - 1), 100 * (r[2] / res[0][2] - 1)) for i, r in enumerate(res[1:], 1))), flush=True)
A = np.array(rows)
for i in range(1, len(settings)):
    d = 100 * (A[:, i, 2] / A[:, 0, 2] - 1); print('POOLED S%-2d %-38s adjusted %+.3f%% (se %.3f, lower %d/%d) MSE %+.3f%% bill %+.3f%%' % (i, json.dumps(settings[i])[:38], d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0, (d < 0).sum(), len(d), (100 * (A[:, i, 0] / A[:, 0, 0] - 1)).mean(), (100 * (A[:, i, 1] / A[:, 0, 1] - 1)).mean()), flush=True)
print('SCREEN_DONE', flush=True)

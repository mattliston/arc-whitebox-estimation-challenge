"""TAG scout-native run_cert_v2 (2026-10-12; = v1 + PB_ALL=1: the setup prebind guard _PB_WALL_S lifted so every listed key binds whatever the load): one process, one candidate, fresh networks in the given order: per network the exact bill,
charged-op count, final-layer MSE vs the analyst's MC truth, prediction sha (float32 bytes of all 16 layers, as gate_rdz_v1) and the
candidate's stats dicts named in STATS (env, comma list). Predictions saved to <out>_pred.npz.
usage: run_cert_v1.py <candidate.py> <comma seeds> <out.json>"""
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(O / 'analyst/fresh'))
import loader

candidate, seeds, out = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], Path(sys.argv[3]).resolve()
assert not out.exists()
stats_names = [x for x in os.environ.get('STATS', '').split(',') if x]
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_cert'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
if os.environ.get('PB_ALL') == '1':
    mod._PB_WALL_S = 1e9
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
ops = [0]
ORIG = _budget.BudgetContext._charge_op


def counted(self, *a, **k):
    ops[0] += 1
    return ORIG(self, *a, **k)


_budget.BudgetContext._charge_op = counted
rows, preds = [], {}
for seed in seeds:
    w = loader.weights(seed)
    truth = loader.truth(seed, check_weights=False)['mc_post']
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
    ops[0] = 0
    t0 = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    mse = ((y - truth) ** 2).mean(axis=1)
    row = dict(seed=seed, flops=int(ctx.summary_dict()['flops_used']), ops=ops[0], wall_s=round(time.monotonic() - t0, 1),
               mse_final=float(mse[15]), mse_layers=[float(x) for x in mse],
               pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16],
               stats={n: json.loads(json.dumps(mod.__dict__.get(n), default=str)) for n in stats_names})
    rows.append(row); preds[str(seed)] = y.astype(np.float32)
    print(json.dumps(row), flush=True)
_budget.BudgetContext._charge_op = ORIG
np.savez(str(out).replace('.json', '_pred.npz'), **preds)
with out.open('x') as f:
    json.dump(dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), seeds=seeds, rows=rows), f)
print(json.dumps(dict(done=True, n=len(rows))), flush=True)

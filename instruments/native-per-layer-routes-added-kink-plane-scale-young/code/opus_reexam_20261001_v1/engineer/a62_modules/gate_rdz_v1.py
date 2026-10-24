"""TAG opus_engineer a62 RDZ gate runner: one arm of the read-cut error test on fresh generated networks with Monte Carlo truths
(analyst/fresh), one process per arm, the same seed order in every arm.

usage: gate_rdz_v1.py <candidate.py> <rowz|none> <colz|none> <comma seeds> <out.json>
Per network: bill, charged-op count, final-layer MSE against the truth's pooled mean (and all 16 layers' MSE), the RDZ counters, and
the prediction's sha. Predictions are saved next to out.json as <out>_pred.npz (one (16, 1024) array per seed).
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import types
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader

candidate = Path(sys.argv[1]).resolve()
rowz = None if sys.argv[2] == 'none' else float(sys.argv[2])
colz = None if sys.argv[3] == 'none' else float(sys.argv[3])
seeds = [int(x) for x in sys.argv[4].split(',')]
out = Path(sys.argv[5]).resolve()
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_rdz'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
if rowz is not None or colz is not None:
    mod._RDZ_ROWZ, mod._RDZ_COLZ = rowz, colz
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2',
                                scratch_dir=None, submission_dir=None, seed=0))
ops = [0]
original = _budget.BudgetContext._charge_op


def counted(self, *args, **kwargs):
    ops[0] += 1
    return original(self, *args, **kwargs)


_budget.BudgetContext._charge_op = counted
rows, preds = [], {}
for seed in seeds:
    w = loader.weights(seed)
    truth = loader.truth(seed, check_weights=False)['mc_post']
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
    stats0 = dict(getattr(mod, '_RDZ_STATS', {}))
    ops[0] = 0; t0 = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    stats = {k: v - stats0.get(k, 0) for k, v in getattr(mod, '_RDZ_STATS', {}).items()}
    mse = ((y - truth) ** 2).mean(axis=1)
    row = dict(seed=seed, rowz=rowz, colz=colz, flops=int(ctx.summary_dict()['flops_used']), ops=ops[0],
               wall_s=round(time.monotonic() - t0, 2), mse_final=float(mse[15]), mse_layers=[float(x) for x in mse],
               rdz=stats, pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16])
    rows.append(row); preds[str(seed)] = y.astype(np.float32)
    print(json.dumps(row), flush=True)
    del w, y
_budget.BudgetContext._charge_op = original
np.savez(str(out).replace('.json', '_pred.npz'), **preds)
with out.open('x') as stream:
    json.dump(dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), rowz=rowz, colz=colz,
                   seeds=seeds, rows=rows), stream)
print(json.dumps(dict(done=True, n=len(rows))), flush=True)

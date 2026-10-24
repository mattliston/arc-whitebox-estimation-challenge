"""[opus-viewbill] native N-network worker-like session in ONE process (warm_judge_v1's setup/predict emulation: PB_ALL prebind,
unlimited setup wall, BudgetContext 2**41): predicts the given seeds in order and records flops / charged ops per network.
usage: multi_judge_v1.py <candidate.py> <out.json> <seed> [<seed> ...]   env: CANDIDATE_SHA256, PB_ALL=1, BLAS threads=1"""
import hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
candidate, out, seeds = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), [int(s) for s in sys.argv[3:]]
assert not out.exists()
assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')) and os.environ['PB_ALL'] == '1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == os.environ['CANDIDATE_SHA256']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_multi'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
ops = [0]; original = _budget.BudgetContext._charge_op
def counted(self, *a, **kw):
    ops[0] += 1
    return original(self, *a, **kw)
_budget.BudgetContext._charge_op = counted
rows, preds = [], {}
for s in seeds:
    w = loader.weights(s)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{s}')
    ops[0] = 0; t0 = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    assert y.shape == (16, 1024) and np.isfinite(y).all()
    rows.append(dict(seed=s, flops=int(ctx.summary_dict()['flops_used']), ops=ops[0], wall=time.monotonic() - t0)); preds[str(s)] = y.astype(np.float32)
    print(json.dumps(rows[-1]), flush=True)
np.savez(str(out).replace('.json', '_pred.npz'), **preds)
with out.open('x') as f:
    json.dump(dict(candidate=str(candidate), sha256=os.environ['CANDIDATE_SHA256'], rows=rows), f)

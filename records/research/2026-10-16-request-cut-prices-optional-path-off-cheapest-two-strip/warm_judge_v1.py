"""[opus-pushsm] native FIRST + WARM bills in one process: the diag judge's setup/predict emulation (PB_ALL prebind, unlimited setup
wall, BudgetContext 2**41), then predict on a warm-up seed (the worker's first network: lazy binds land here) and on the target seed
(warm). Records flops / charged ops / process instructions / cycles of both networks and the target's predictions + MSE.
usage: warm_judge_v1.py <candidate.py> <warm-up seed> <seed> <out.json>   env: CANDIDATE_SHA256, PB_ALL=1, BLAS threads=1"""
import hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
sys.path.insert(0, str(O/'release')); import rusage
candidate, warm_seed, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).resolve()
assert not out.exists()
assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')) and os.environ['PB_ALL'] == '1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == os.environ['CANDIDATE_SHA256']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_warm'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
ops = [0]; original = _budget.BudgetContext._charge_op
def counted(self, *a, **kw):
    ops[0] += 1
    return original(self, *a, **kw)
_budget.BudgetContext._charge_op = counted


def network(s):
    w = loader.weights(s)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{s}')
    ops[0] = 0; c0, i0 = rusage.raw(); t0 = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    c1, i1 = rusage.raw()
    assert y.shape == (16, 1024) and np.isfinite(y).all()
    return y, dict(seed=s, flops=int(ctx.summary_dict()['flops_used']), ops=ops[0], instructions=i1 - i0, cycles=c1 - c0,
                   wall=time.monotonic() - t0, pushsm=dict(mod.__dict__.get('_PUSHSM_STATS', {})), smx=dict(mod.__dict__.get('_SMX_STATS', {})))


_, first = network(warm_seed)
y, warm = network(seed)
truth = loader.truth(seed, check_weights=False)['mc_post']
warm['mse_final'] = float(((y[-1] - truth[-1]) ** 2).mean())
np.savez(str(out).replace('.json', '_pred.npz'), **{str(seed): y.astype(np.float32)})
with out.open('x') as f:
    json.dump(dict(candidate=str(candidate), sha256=os.environ['CANDIDATE_SHA256'], first=first, rows=[warm]), f)
print(json.dumps(dict(first=first, warm=warm)), flush=True)

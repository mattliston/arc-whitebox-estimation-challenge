"""Generic fresh-process native screen judge (lead). Same setup/predict emulation as the opus/astra native judges
(PB_ALL prebind, unlimited setup wall, flopscope BudgetContext 2**41, MC-truth MSE) without protocol coupling.
usage: screen_judge_opus_v1.py <candidate.py> <seed> <out.json>   env: CANDIDATE_SHA256, PB_ALL=1, BLAS threads=1"""
import hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(O / 'analyst/fresh'))
import loader

candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
assert not out.exists()
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
assert all(os.environ.get(k) == '1' for k in THREADS) and os.environ['PB_ALL'] == '1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == os.environ['CANDIDATE_SHA256']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_cert'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); start = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
setup_wall = time.monotonic() - start
ops = [0]; original = _budget.BudgetContext._charge_op
def counted(self, *a, **kw):
    ops[0] += 1
    return original(self, *a, **kw)
_budget.BudgetContext._charge_op = counted
w = loader.weights(seed)
truth = loader.truth(seed, check_weights=False)['mc_post']
before = hashlib.sha256(w.tobytes()).hexdigest()
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
ops[0] = 0; start = time.monotonic()
try:
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
finally:
    _budget.BudgetContext._charge_op = original
assert y.shape == (16, 1024) and np.isfinite(y).all()
assert hashlib.sha256(w.tobytes()).hexdigest() == before
mse = ((y - truth) ** 2).mean(axis=1)
row = dict(seed=seed, weights_sha256=before, setup_wall_s=setup_wall, predict_wall_unrounded_s=time.monotonic() - start,
           flops=int(ctx.summary_dict()['flops_used']), ops=ops[0], mse_final=float(mse[-1]), mse_layers=[float(x) for x in mse])
np.savez(str(out).replace('.json', '_pred.npz'), **{str(seed): y.astype(np.float32)})
with out.open('x') as f:
    json.dump(dict(candidate=str(candidate), sha256=os.environ['CANDIDATE_SHA256'], rows=[row],
                   judge_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   loader_sha256=hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(), python_version=sys.version), f)
print(json.dumps(dict(done=True, seed=seed, flops=row['flops'], mse_final=row['mse_final'])), flush=True)

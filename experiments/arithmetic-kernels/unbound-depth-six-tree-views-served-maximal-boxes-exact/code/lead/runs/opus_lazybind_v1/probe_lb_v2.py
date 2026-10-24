"""[opus-lazybind] probe v2 = v1 + saves the prediction (<out>_pred.npz) and reports the noise-free cross-half xMSE vs the control.
[v1] probe = the lead's screen judge (PB_ALL setup, unlimited setup wall, one fresh network, BudgetContext 2**41) that
also prints the candidate's _LB_STATS and compares its prediction bytes with a stored control prediction (same seed).
usage (OpenBLAS env, BLAS 1, PB_ALL=1, under the guard): probe_lb_v1.py <candidate.py> <seed> <control_pred.npz> <out.json>"""
import hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh'))
import loader
cand = Path(sys.argv[1]).resolve(); seed = int(sys.argv[2]); ctl = np.load(sys.argv[3])[str(seed)]; out = Path(sys.argv[4])
assert not out.exists() and os.environ.get('PB_ALL') == '1'
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); t0 = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
setup_s = time.monotonic() - t0
boxes = dict(getattr(mod, '_LB_STATS', {}))
ops = [0]; original = _budget.BudgetContext._charge_op
def counted(self, *a, **kw):
    ops[0] += 1
    return original(self, *a, **kw)
_budget.BudgetContext._charge_op = counted
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
t0 = time.monotonic()
try:
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
finally:
    _budget.BudgetContext._charge_op = original
y32 = y.astype(np.float32)
row = dict(seed=seed, setup_s=setup_s, predict_s=time.monotonic() - t0, flops=int(ctx.summary_dict()['flops_used']), ops=ops[0],
           lb_after_setup=boxes, lb_after_predict=dict(getattr(mod, '_LB_STATS', {})), bytes_equal=bool(y32.tobytes() == ctl.tobytes()),
           max_abs_diff=float(np.max(np.abs(y32.astype(np.float64) - ctl.astype(np.float64)))), first_layer_diff=next((l for l in range(16) if y32[l].tobytes() != ctl[l].tobytes()), None))
np.savez(str(out).replace('.json', '_pred.npz'), **{str(seed): y32})
t = loader.truth(seed, check_weights=False) if False else np.load(O / f'analyst/fresh/truth_seed{seed}.npz')
A, B = t['mc_post_A'][-1], t['mc_post_B'][-1]
ya, yb = y32[-1].astype(np.float64), ctl[-1].astype(np.float64)
row['xmse_pct'] = float(100 * np.log(np.mean((ya - A) * (ya - B)) / np.mean((yb - A) * (yb - B))))
row['rel_rms_final'] = float(np.sqrt(np.mean((ya - yb) ** 2)) / np.sqrt(np.mean(yb ** 2)))
out.write_text(json.dumps(row, indent=1) + '\n'); print(json.dumps(row), flush=True)

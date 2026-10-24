"""[opus-k4unit] native capture on an observer_fit rank (the networks with 2 x 32M per-unit truths): same setup/predict emulation as
screen_judge_opus_v1 (PB_ALL prebind, unlimited setup wall, BudgetContext 2**41); saves the candidate's _K4U_CAP stash + predictions.
usage: k4u_capture_v1.py <capture candidate.py> <rank> <out.npz>   env: CANDIDATE_SHA256, PB_ALL=1, BLAS threads=1"""
import hashlib, importlib.util, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
candidate, rank, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
assert not out.exists()
assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')) and os.environ['PB_ALL'] == '1'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == os.environ['CANDIDATE_SHA256']
spec = importlib.util.spec_from_file_location('oc', O/'trainer/otr_common.py'); oc = importlib.util.module_from_spec(spec); spec.loader.exec_module(oc)
wpath = oc.data_paths(rank)[0]
w = np.load(wpath)
assert w.shape == (16, 1024, 1024), w.shape
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_k4u'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'rank-{rank}')
t0 = time.monotonic()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
cap = mod._K4U_CAP
assert 'L14' in cap and 'L15' in cap, list(cap)
flat = {f'{k}_{kk}': vv for k, d in cap.items() for kk, vv in d.items()}
np.savez(out, y=y, flops=np.array(int(ctx.flops_used)), wall=np.array(time.monotonic() - t0), weights_path=np.array(str(wpath)), **flat)
print('saved', out, 'flops', int(ctx.flops_used), {k: v.shape for k, v in flat.items()})

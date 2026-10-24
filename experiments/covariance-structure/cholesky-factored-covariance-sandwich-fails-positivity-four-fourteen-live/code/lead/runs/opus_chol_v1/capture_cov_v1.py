"""[opus-pass-release] Cholesky-sandwich PD census, capture step: run a candidate natively on one seed and record, at every call of the
live covariance sandwich (_dc_cov_star / _flms_cov_star: Cpre = W^T C W), the input covariance C (as float64), the weight W and the dcv
roster the call uses (from _dcv_roster's return, read with sys.setprofile; no monkeypatching). usage: capture_cov_v1.py <cand.py> <seed> <out.npz>"""
import sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, seed, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_cov'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
A = lambda x: np.array(np.asarray(x))
codes = {mod._dc_cov_star.__code__: 'dc', mod._flms_cov_star.__code__: 'flms'}
RC = mod._dcv_roster.__code__
REC = []; pending = {}
def prof(frame, event, arg):
    co = frame.f_code
    if event == 'call' and co in codes and not pending.get('busy'):
        L = frame.f_locals
        pending.update(busy=True, kind=codes[co], C=A(L['C']).astype(np.float64), w=A(L['w']).astype(np.float32), roster=None, m=None)
    elif event == 'return' and co is RC and pending.get('busy') and pending.get('roster') is None:
        pending['roster'] = None if arg is None else A(arg).astype(np.int64)
        pending['m'] = int(mod._DCV_STATS.get('m', -1)) if isinstance(getattr(mod, '_DCV_STATS', None), dict) else -1
    elif event == 'return' and co in codes and pending.get('busy') and codes[co] == pending['kind']:
        REC.append(dict(pending)); pending.clear()
w = loader.weights(seed); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
sys.setprofile(prof)
try:
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        y = est.predict(mlp, 2**41)
finally:
    sys.setprofile(None)
print('sandwich calls captured', len(REC), [(r['kind'], None if r['roster'] is None else len(r['roster']), r['m']) for r in REC])
np.savez_compressed(out, **{f'C{i}': r['C'].astype(np.float32) for i, r in enumerate(REC)}, **{f'W{i}': r['w'] for i, r in enumerate(REC)},
                    **{f'R{i}': (r['roster'] if r['roster'] is not None else np.arange(0)) for i, r in enumerate(REC)},
                    meta=np.array([[0 if r['kind'] == 'dc' else 1, -1 if r['roster'] is None else len(r['roster']), -1 if r['m'] is None else r['m']] for r in REC]))

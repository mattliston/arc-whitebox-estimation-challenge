"""[opus-exactkernel] FULL pre-activation covariance at L12-14 (from the variance capture v2) via sys.setprofile (no monkeypatching: the module checks function identity). Per layer l: the estimator's
POST variance of layer l-1 (diag of C entering the l-th _dc_cov_star / _dc_cov_diagonal call), its PRE variance at l (diag of the returned
C_pre; at L15 the returned v), the OTT tail rows (trace-only old-tier rows) per layer, and the predicted means y."""
import sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, seeds, outdir = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(',')], Path(sys.argv[3])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_var2'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
STAR, DIAG, OTT = mod._dc_cov_star.__code__, mod._dc_cov_diagonal.__code__, getattr(mod, '_ott_trace', None)
OTT = OTT.__code__ if OTT is not None else None
REC = {}; A = lambda x: np.array(np.asarray(x), dtype=np.float64)
def prof(frame, event, arg):
    co = frame.f_code
    if co is STAR or co is DIAG:
        li = int(mod._PHYS_LAYER[0]) + 1
        if event == 'call':
            REC.setdefault(li, {})['post_prev'] = np.diagonal(A(frame.f_locals['C'])).copy()
        elif event == 'return' and arg is not None:
            r = arg[0] if isinstance(arg, tuple) else arg
            r = A(r)
            if r.ndim == 2 and li in (12, 13, 14): REC.setdefault(li, {})['Cpre_full'] = r.astype(np.float32)
            REC.setdefault(li, {})['pre'] = np.diagonal(r).copy() if r.ndim == 2 else r
    elif OTT is not None and co is OTT and event == 'call':
        REC.setdefault(('ott', int(mod._UVP_CUR[0])), A(frame.f_locals['tail']).astype(np.int64))
for s in seeds:
    w = loader.weights(s); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{s}')
    REC.clear(); sys.setprofile(prof)
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True): y = A(est.predict(mlp, 2**41))
    finally:
        sys.setprofile(None)
    arr = {}
    for k, v in REC.items():
        if isinstance(k, tuple): arr[f'ott_tail_L{k[1]}'] = v
        else:
            for kk, vv in v.items(): arr[f'{kk}_L{k}'] = vv
    np.savez_compressed(outdir/f'var_{s}.npz', y=y, **arr); print('captured', s, sorted(arr), flush=True)

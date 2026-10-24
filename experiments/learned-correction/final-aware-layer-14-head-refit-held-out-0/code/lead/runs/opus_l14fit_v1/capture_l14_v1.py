"""[opus-l14fit] capture (sys.setprofile; no monkeypatching): at the L14 return of the live _gop2_gnn_step: state, sigma, rms/gain/head/
head_b, raw delta; at the L14 call of _astra_fhm_effective: the pre-correction mean and the raw dm (clipping = max(dm, -mean)); at the L15
return of _dc_cov_diagonal: the L15 pre-variance v15; and the predicted means y (16 x 1024). One predict per seed."""
import sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, seeds, outdir, tag = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(',')], Path(sys.argv[3]), sys.argv[4]
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_l14'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
GS, EF, DG = mod._gop2_gnn_step.__code__, mod._astra_fhm_effective.__code__, mod._dc_cov_diagonal.__code__; REC = {}
A = lambda x: np.array(np.asarray(x), dtype=np.float64)
def prof(frame, event, arg):
    co = frame.f_code
    if co is GS and event == 'return' and arg is not None and int(frame.f_locals.get('l', -1)) == 14:
        c = frame.f_locals['c']; REC.update(delta14=A(arg[0]).ravel(), state14=A(arg[1]), sigma14=A(frame.f_locals['sigma']).ravel(),
                                              rms14=A(c['rms']), gain14=A(c['gain']), head14=A(c['head']), head_b14=A(c['head_b']))
    elif co is EF and event == 'call' and int(frame.f_locals.get('layer', -1)) == 14:
        REC.update(mean14_pre=A(frame.f_locals['mean']), dm14_raw=A(frame.f_locals['delta']))
    elif co is DG and event == 'return' and arg is not None and int(mod._PHYS_LAYER[0]) + 1 == 15:
        REC['v15'] = A(arg[0] if isinstance(arg, tuple) else arg).ravel()
for s in seeds:
    p = outdir/f'{tag}_{s}.npz'
    if p.exists(): continue
    w = loader.weights(s); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{s}')
    REC.clear(); sys.setprofile(prof)
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True): y = A(est.predict(mlp, 2**41))
    finally:
        sys.setprofile(None)
    missing = [k for k in ('state14', 'mean14_pre', 'v15') if k not in REC]; assert not missing, missing
    np.savez_compressed(p, y=y, **REC); print('captured', tag, s, flush=True)

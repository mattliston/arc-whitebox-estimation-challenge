"""[opus-headfit] capture the final-layer learned mean correction inputs natively (S2tr 46e9315d): at the L15 return of the live
_gop2_gnn_step (sys.setprofile, no monkeypatching): state (head input), sigma, delta, and the layer constants rms / gain / head / head_b;
plus the final prediction y15 (= m + delta in _astra_fhm_core). One predict per seed."""
import json, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, seeds, outdir = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(',')], Path(sys.argv[3])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_head'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
CO = mod._gop2_gnn_step.__code__; REC = {}
A = lambda x: np.array(np.asarray(x), dtype=np.float64)
def prof(frame, event, arg):
    if frame.f_code is CO and event == 'return' and arg is not None and int(frame.f_locals.get('l', -1)) == 15:
        c = frame.f_locals['c']; REC.update(delta=A(arg[0]).ravel(), state=A(arg[1]), sigma=A(frame.f_locals['sigma']).ravel(),
                                              rms=A(c['rms']), gain=A(c['gain']), head=A(c['head']), head_b=A(c['head_b']))
for s in seeds:
    p = outdir/f'head_{s}.npz'
    if p.exists(): continue
    w = loader.weights(s); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{s}')
    REC.clear(); sys.setprofile(prof)
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True): y = A(est.predict(mlp, 2**41))
    finally:
        sys.setprofile(None)
    assert 'state' in REC, 'L15 head not captured'
    np.savez_compressed(p, y15=y[-1], **REC); print('captured', s, REC['state'].shape, flush=True)

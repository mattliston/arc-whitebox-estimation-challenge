"""[opus-acc] fresh-process native dial screen: exec candidate, apply in-process dial overrides (OPUS_DIAL_SPEC json list), setup
with PB_ALL=1, predict one generated network under flopscope, write per-layer MSE vs cached MC truth + FLOPs. Research only."""
import hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh'))
import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
spec = json.loads(os.environ.get('OPUS_DIAL_SPEC', '[]'))
assert not out.exists()
assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
assert os.environ['PB_ALL'] == '1'
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_dial'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
applied = []
for s in spec:
    k = s['kind']
    if k == 'head_scale':          # scale the learned mean correction delta at the listed layers
        parent = mod.gnn_step; layers = {int(l): float(s['s']) for l in s['layers']}
        def gnn_step(l, x, sigma, W, state, P, _parent=parent, _layers=layers):
            d, st = _parent(l, x, sigma, W, state, P)
            f = _layers.get(int(l))
            return (d if f is None else d * f), st
        mod.gnn_step = gnn_step
    elif k == 'dict':              # set one key of a module dict literal
        mod.__dict__[s['name']][s['key']] = s['value']
    elif k == 'set':               # replace a module literal
        v = s['value']
        if isinstance(v, dict): v = {(int(a) if str(a).lstrip('-').isdigit() else a): b for a, b in v.items()}
        mod.__dict__[s['name']] = (tuple(v) if isinstance(v, list) else v)
    elif k == 'uvp15':             # r_u of the L15 old-tier pair-mean read (source hard-codes 256 at L15; min(r_u, rank) applies)
        parent15 = mod._LCR_PARENT_UV_STATE; ru15 = int(s['value'])
        def _uvp_state(state, _parent=parent15, _ru15=ru15):
            ru = mod._LCR_RU if mod._UVP_CUR[0] in mod._LCR_LAYERS else _ru15
            if state.get('lcr_ru') != ru:
                state['uvp'] = None
            saved = mod._UVP_RU; mod._UVP_RU = ru
            try:
                uv = _parent(state)
            finally:
                mod._UVP_RU = saved
            state['lcr_ru'] = ru
            return uv
        mod._uvp_state = _uvp_state
    else:
        raise ValueError(k)
    applied.append(s)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); t0 = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
setup_s = time.monotonic() - t0
w = loader.weights(seed); truth = loader.truth(seed, check_weights=False)['mc_post']
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
t0 = time.monotonic()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
wall = time.monotonic() - t0
assert y.shape == (16, 1024) and np.isfinite(y).all()
mse = ((y - truth)**2).mean(axis=1)
np.savez(str(out).replace('.json', '_pred.npz'), y=y.astype(np.float32))
with out.open('x') as f:
    json.dump(dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), seed=seed, spec=applied,
                   flops=int(ctx.summary_dict()['flops_used']), mse_final=float(mse[-1]), mse_layers=[float(x) for x in mse],
                   setup_s=setup_s, predict_wall_s=wall, pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16]), f)
print(json.dumps(dict(done=True, seed=seed, mse_final=float(mse[-1]), wall=wall)), flush=True)

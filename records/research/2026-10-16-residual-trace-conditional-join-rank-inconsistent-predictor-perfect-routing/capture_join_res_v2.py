"""[opus-condrank] v2 (records only QR calls inside the Nystrom join branch) capture, at every join's QR call inside _astra_bcc_push_impl (sys.setprofile; no monkeypatching), the inputs that fix
the Nystrom residual exactly: J, weight, Qp and the OLD core (if any), the pivot rows and Y. Residual = tr G - tr(Y G_PP^-1 Y') offline."""
import sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, tag, seeds, outdir = Path(sys.argv[1]), sys.argv[2], [int(s) for s in sys.argv[3].split(',')], Path(sys.argv[4])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_jres'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
QR = fnp.linalg.qr.__code__; REC = []
A = lambda x: np.array(np.asarray(x), dtype=np.float64)
def prof(frame, event, arg):
    if event == 'call' and frame.f_code is QR:
        c = frame.f_back
        if c is not None and c.f_code.co_name == '_astra_bcc_push_impl' and '_jn_rows' in c.f_locals and 'J' in c.f_locals:
            L = c.f_locals; st = L['state']; old = int(L.get('old_columns', 0) or 0)
            REC.append(dict(layer=int(L['layer']), J=A(L['J']), w=A(L['weight']), rows=A(L['_jn_rows']).astype(np.int64), Y=A(frame.f_locals['a']),
                            Qp=A(L['Qp']) if old and L.get('Qp') is not None else np.zeros((0, 0)), core=A(st['core']) if old and st.get('core') is not None else np.zeros((0, 0))))
for s in seeds:
    w = loader.weights(s); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{s}')
    REC.clear(); sys.setprofile(prof)
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True): est.predict(mlp, 2**41)
    finally:
        sys.setprofile(None)
    out = {}
    for r in REC:
        for k, v in r.items():
            if k != 'layer': out[f"L{r['layer']}_{k}"] = v
    np.savez_compressed(outdir/f'jres_{tag}_{s}.npz', **out); print('captured', tag, s, [r['layer'] for r in REC], flush=True)

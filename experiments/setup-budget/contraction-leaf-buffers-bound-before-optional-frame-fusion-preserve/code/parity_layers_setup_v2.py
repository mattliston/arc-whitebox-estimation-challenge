#!/usr/bin/env python3
"""TAG fable-porter (PT-52a: versioned copy of parity_layers_setup.py; the ONLY change: the per-rank stats line also prints _TR_STATS
(engineer E-73 transform reuse). Original: TAG fable-porter (PT-50a: versioned copy of parity_layers.py; the ONLY change: each Estimator's setup(context) runs once before its first
predict, as the platform worker and census_boundary_setup_v2.py call it (inside a BudgetContext here, in process), and the module's
_PB_LOG bound count and _LVF_STATS fused-key count are printed, so a setup-only kernel (lvfuse2) is actually exercised). Original:
TAG fable-porter (PT-33b): in-process (.venv, 1 BLAS thread) native predictions of <candA> and <candB> on each rank: all 16 layer outputs
array-equal? flops equal? plus the final-layer MSE ratio B/A against truthpost. usage: parity_layers.py <candA.py> <candB.py> <ranks e.g. 35,100>"""
import sys, json, types
from pathlib import Path
import numpy as np, flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
def load(path, tag):
    mod = types.ModuleType(tag); mod.__file__ = str(path); sys.modules[tag] = mod
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__); return mod
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
A, B = load(sys.argv[1], 'candA'), load(sys.argv[2], 'candB'); ranks = [int(x) for x in sys.argv[3].split(',')]
print(f"TAG fable-porter PT-33b: {Path(sys.argv[1]).name} vs {Path(sys.argv[2]).name}, in-process, ranks {ranks} (first cold)")
ea, eb = A.Estimator(), B.Estimator()
for tag, mod, est in (('A', A, ea), ('B', B, eb)):
    if hasattr(type(est), 'setup'):
        sctx = types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0)
        import time as _t; t0 = _t.time()
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            est.setup(sctx)
        pb = [x for x in getattr(mod, '_PB_LOG', []) if x[0] == 'bound']
        print(f"setup {tag}: wall {_t.time() - t0:.2f} s; bound {pb[-1][1] if pb else None}; guard {[list(x) for x in getattr(mod, '_PB_LOG', []) if x[0] != 'bound']}; lvf {getattr(mod, '_LVF_STATS', None)}", flush=True)
    else:
        print(f'setup {tag}: none', flush=True)
for r in ranks:
    W = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy"); out = {}
    for tag, est in (('A', ea), ('B', eb)):
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"par-{r}"), 2**41), dtype=np.float64)
        out[tag] = (y, ctx.summary_dict()["flops_used"])
    truth = np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    mse = {k: float(((v[0][15] - truth) ** 2).mean()) for k, v in out.items()}
    same = bool(np.array_equal(out['A'][0], out['B'][0])); d = float(np.abs(out['A'][0] - out['B'][0]).max())
    print(f"lvf after r{r}: A {getattr(A, '_LVF_STATS', None)} B {getattr(B, '_LVF_STATS', None)}; tr A {getattr(A, '_TR_STATS', None)} B {getattr(B, '_TR_STATS', None)}", flush=True)
    print(f"r{r}: all-16-layer predictions identical {same} (max abs diff {d:.3e}); flops A {out['A'][1]:,} B {out['B'][1]:,} identical {out['A'][1] == out['B'][1]} "
          f"(B/A {100*(out['B'][1]/out['A'][1]-1):+.3f}%); final MSE B/A {100*(mse['B']/mse['A']-1):+.3f}%", flush=True)

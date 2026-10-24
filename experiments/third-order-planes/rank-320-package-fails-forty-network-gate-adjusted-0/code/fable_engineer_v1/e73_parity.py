#!/usr/bin/env python3
"""TAG fable-engineer (E-73 gate 1 = e71_parity.py + per-net transform-reuse counts of candB (_TR_STATS)): in-process (.venv flopscope, 1 BLAS thread) parity of <candA> vs <candB> WITH Estimator.setup first
(the platform order; parity_layers.py skips setup, which would leave a setup-bound feature like --lvfuse off): per rank, all 16 layer
outputs array-equal, flops identical, final MSE ratio; plus each candidate's setup log (_PB_LOG) and --lvfuse stats.
usage: .venv/bin/python e71_parity.py <candA.py> <candB.py> <ranks>"""
import sys, json, types, time
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
ranks = [int(x) for x in sys.argv[3].split(',')]
res = {}
for tag, path in (('A', sys.argv[1]), ('B', sys.argv[2])):
    mod = load(path, 'cand' + tag); est = mod.Estimator()
    t0 = time.time(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
    print(json.dumps(dict(cand=tag, file=Path(path).name, setup_s=round(time.time() - t0, 2), pb_log=[list(x) for x in getattr(mod, '_PB_LOG', [])], lvf=getattr(mod, '_LVF_STATS', None))), flush=True)
    res[tag] = {}
    for r in ranks:
        W = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"e71-{r}"), 2**41), dtype=np.float64)
        res[tag][r] = (y, ctx.summary_dict()["flops_used"])
        if hasattr(mod, '_TR_STATS'):
            print(json.dumps(dict(cand=tag, rank=r, tr_stats=dict(mod._TR_STATS))), flush=True); mod._TR_STATS.update(reused=0, computed=0)
    del mod, est
for r in ranks:
    (ya, fa), (yb, fb) = res['A'][r], res['B'][r]
    truth = np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    ma, mb = float(((ya[15] - truth) ** 2).mean()), float(((yb[15] - truth) ** 2).mean())
    print(json.dumps(dict(rank=r, all16_identical=bool(np.array_equal(ya, yb)), max_abs_diff=float(np.abs(ya - yb).max()), flops_A=fa, flops_B=fb, flops_identical=fa == fb, mse_ratio=mb / ma)), flush=True)

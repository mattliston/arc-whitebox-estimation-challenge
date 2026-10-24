#!/usr/bin/env python3
"""TAG fable-engineer2 (d7cp parity shortfall probe): load a candidate with the d7cp fragment, setup() first, run nets in order and log
every _mm call with m, k >= 896 and n > 1024 (the colpack regime): shape, whether d7cp took it, and the flops the call billed
(budget delta around the call). usage: probe_d7cp_calls.py <cand.py> <nets>"""
import sys, json, types
from pathlib import Path
import numpy as np, flopscope as flops
P2 = Path(__file__).resolve().parents[2]
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("cand"); mod.__file__ = sys.argv[1]; sys.modules["cand"] = mod
exec(compile(Path(sys.argv[1]).read_text(), sys.argv[1], "exec"), mod.__dict__)
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
log = []; CTX = [None]
live_mm = mod._mm
def spy(a, b):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if m >= 896 and k >= 896 and n > 1024:
        before = CTX[0].summary_dict()['flops_used']; ST = getattr(mod, '_D7_STATS', {'products': 0}); took0 = ST['products']
        out = live_mm(a, b)
        log.append(dict(m=m, k=k, n=n, d7=ST['products'] > took0, flops=int(CTX[0].summary_dict()['flops_used'] - before)))
        return out
    return live_mm(a, b)
mod._mm = spy
for r in [int(x) for x in sys.argv[2].split(',')]:
    W = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy"); log.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        CTX[0] = ctx
        est.predict(MLP([W[l] for l in range(16)], f"probe-{r}"), 2**41)
    print(json.dumps(dict(rank=r, flops=int(ctx.summary_dict()['flops_used']), calls=log)), flush=True)

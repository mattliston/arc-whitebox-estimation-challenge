#!/usr/bin/env python3
"""TAG fable-engineer2 (split u/v old-tier BILL prototype): setup() first, then nets in order under their own BudgetContext; prints the
bill (flops_used), util and the uvp call counters; saves outputs + flops to <out.npz> for an MSE glance (no accuracy claim).
usage: uvp_bill_probe.py <cand.py> <nets> <out.npz>"""
import sys, json, types, time, traceback
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
out = {}
for r in [int(x) for x in sys.argv[2].split(',')]:
    W = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
    t0 = time.time()
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"uvp-{r}"), 2**41), dtype=np.float64)
    except Exception as e:
        print(json.dumps(dict(rank=r, error=f"{type(e).__name__}: {str(e)[:300]}")), flush=True); traceback.print_exc(); continue
    f = int(ctx.summary_dict()["flops_used"]); out[f"y{r}"] = y; out[f"f{r}"] = np.int64(f)
    print(json.dumps(dict(rank=r, flops=f, util=f / 2**41, wall_s=round(time.time() - t0, 1), finite=bool(np.isfinite(y).all()), stats=dict(getattr(mod, '_UVP_STATS', {})))), flush=True)
np.savez(sys.argv[3], **out)
print("SAVED", sys.argv[3], flush=True)

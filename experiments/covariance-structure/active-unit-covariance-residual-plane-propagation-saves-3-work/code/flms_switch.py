#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-9): the FLMS switch (_FULL_LOWER_MOMENTS) as a bill-vs-raw dial on the deployed 331824 card:
True = full 1024-row covariance/star pushes (deployed); False = alive-compact pushes (cheaper at depth where ~25% of units are dead).
usage: flms_switch.py <candidate.py> <ranks> -> prints util/mse for both settings"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text(); assert src0.count("_FULL_LOWER_MOMENTS = True\n") == 1
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
out = {}
for flag in (True, False):
    src = src0.replace("_FULL_LOWER_MOMENTS = True\n", f"_FULL_LOWER_MOMENTS = {flag}\n")
    mod = types.ModuleType(f"flms_{flag}"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
    exec(compile(src, str(cand), "exec"), mod.__dict__); est = mod.Estimator()
    for rank in ranks:
        W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
        truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
        t0 = time.time()
        try:
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"flms-{rank}"), 2**41), dtype=np.float64)
            util = ctx.summary_dict()["flops_used"] / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
            rec = dict(flms=flag, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))
        except Exception as ex:
            import traceback; rec = dict(flms=flag, rank=rank, error=repr(ex)[:300], tb=traceback.format_exc()[-1000:])
        out[f"{flag}:{rank}"] = rec; print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f"flms_switch_{cand.stem}.json", "w"), indent=1)

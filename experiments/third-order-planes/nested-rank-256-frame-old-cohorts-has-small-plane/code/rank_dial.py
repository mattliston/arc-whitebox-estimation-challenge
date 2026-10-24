#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-15): the shared old-tier RANK as a native dial on the 331824 card (only the LAST
_shared_old384_push definition is patched; also _SLR_RANK).  usage: rank_dial.py <cand> <ranks e.g. 35,100> <tier ranks e.g. 352,320>"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tiers = [int(x) for x in sys.argv[3].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
def patched(r):
    i = src0.rfind("def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):\n"); j = src0.index("\n\n\ndef ", i)
    body = src0[i:j]; assert body.count("    rank = 384\n") == 1 and body.count("< 384:") == 1
    body = body.replace("    rank = 384\n", f"    rank = {r}\n").replace("< 384:", f"< {r}:")
    s = src0[:i] + body + src0[j:]
    assert s.count("_SLR_RANK = 384\n") == 1; s = s.replace("_SLR_RANK = 384\n", f"_SLR_RANK = {r}\n")
    assert s.count("width = fnp.full((1, 1), 384.0, dtype=fnp.float64)") == 1; s = s.replace("width = fnp.full((1, 1), 384.0, dtype=fnp.float64)", f"width = fnp.full((1, 1), {r}.0, dtype=fnp.float64)")
    return s
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
for r in tiers:
    mod = types.ModuleType(f"rank_{r}"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
    exec(compile(patched(r), str(cand), "exec"), mod.__dict__); est = mod.Estimator()
    for rank in ranks:
        W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
        truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
        t0 = time.time()
        try:
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"rd-{rank}"), 2**41), dtype=np.float64)
            util = ctx.summary_dict()["flops_used"] / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
            rec = dict(tier_rank=r, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))
        except Exception as ex:
            import traceback; rec = dict(tier_rank=r, rank=rank, error=repr(ex)[:300], tb=traceback.format_exc()[-800:])
        print(json.dumps(rec), flush=True)

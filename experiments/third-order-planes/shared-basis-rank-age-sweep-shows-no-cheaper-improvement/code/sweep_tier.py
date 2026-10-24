#!/usr/bin/env python3
"""TAG fable-wall-repair (W-N): the shared old-tier RANK (384) and confinement AGE (5 layers) of the c2 third-order carrier as
bill-vs-raw dials. Builds a patched module in memory per (rank, age), predicts the given nets natively, reports util, final MSE vs the
1e9 truth and adjusted(1e9).  usage: sweep_tier.py <cand> <ranks e.g. 35,100> <configs e.g. 384:5,512:5,384:4,384:6>"""
import sys, json, types, importlib.util, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; configs = [tuple(int(v) for v in c.split(":")) for c in sys.argv[3].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
src0 = cand.read_text()
def patched(rank, age):
    s = src0
    for old, new in (("    rank = 384\n", f"    rank = {rank}\n"), ("    if layer < 5:\n", f"    if layer < {age}:\n"), ("        joining = layer - 5\n", f"        joining = layer - {age}\n"),
                     ("        if len(state['joins']) != 10 or not old_columns:\n", f"        if len(state['joins']) != {15 - age} or not old_columns:\n")):
        assert s.count(old) == 1, (old, s.count(old)); s = s.replace(old, new)
    return s
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
out = {}
for rank_, age in configs:
    mod = types.ModuleType(f"tier_{rank_}_{age}"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
    exec(compile(patched(rank_, age), str(cand), "exec"), mod.__dict__)
    est = mod.Estimator()
    for rank in ranks:
        W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
        t0 = time.time()
        try:
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"tier-{rank}"), 2**41), dtype=np.float64)
            bill = ctx.summary_dict()["flops_used"]; util = bill / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
            rec = dict(tier_rank=rank_, age=age, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))
        except Exception as ex:
            rec = dict(tier_rank=rank_, age=age, rank=rank, error=repr(ex)[:200])
        out[f"{rank_}:{age}:{rank}"] = rec; print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f"sweep_tier_{cand.stem}.json", "w"), indent=1)

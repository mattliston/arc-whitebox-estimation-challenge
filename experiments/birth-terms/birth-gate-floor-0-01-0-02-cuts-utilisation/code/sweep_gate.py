#!/usr/bin/env python3
"""TAG fable-wall-repair (W-K): the birth/alive/response gate floor as a bill-vs-raw dial. For each floor, patch the candidate's
_predict_binary default (the active caller passes no gate_floor), predict the given nets natively, and report bill (util), final MSE
vs the 1e9 truth and adjusted(1e9) = MSE x max(0.1, util).  usage: sweep_gate.py <cand> <ranks e.g. 35,100> <floors e.g. 0.01,0.02,0.05>"""
import sys, json, types, importlib.util, functools, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; floors = [float(x) for x in sys.argv[3].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
out = {}
for floor in floors:
    spec = importlib.util.spec_from_file_location(f"gate_cand_{floor}", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    mod._predict_binary = functools.partial(mod._predict_binary, gate_floor=floor)
    est = mod.Estimator()
    for rank in ranks:
        W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
        t0 = time.time()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"gate-{rank}"), 2**41), dtype=np.float64)
        bill = ctx.summary_dict()["flops_used"]; util = bill / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
        rec = dict(floor=floor, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))
        out[f"{floor}:{rank}"] = rec; print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f"sweep_gate_{cand.stem}.json", "w"), indent=1)

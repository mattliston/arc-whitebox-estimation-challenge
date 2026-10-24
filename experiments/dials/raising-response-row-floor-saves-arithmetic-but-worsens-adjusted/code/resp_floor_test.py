#!/usr/bin/env python3
"""TAG fable-refit (RF-19): the RESPONSE-ROW floor as a native bill dial on the 331824 card. The plane read (26% of the bill) costs
~ |response rows| x columns x |live|; response rows = alive units with min(g,1-g) >= floor (bank 0.01). Rows below the floor are treated
as 'omitted' by the card (radial source). usage: resp_floor_test.py <cand> <ranks> <floors e.g. 0.02,0.03,0.05>"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; floors = [float(x) for x in sys.argv[3].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
H = "            response_indices, response_mask = select(response_score, gate_floor)\n"
i = src0.rfind(H); assert i > 0
src = src0[:i] + "            response_indices, response_mask = select(response_score, _RESP_FLOOR)\n" + src0[i + len(H):]
src += "\n_RESP_FLOOR = 0.01\n"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mod = types.ModuleType("resp_floor_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(src, str(cand), "exec"), mod.__dict__); est = mod.Estimator()
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    for fl in [0.01] + floors:
        mod._RESP_FLOOR = fl; t0 = time.time()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"rf-{rank}"), 2**41), dtype=np.float64)
        util = ctx.summary_dict()["flops_used"] / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
        print(json.dumps(dict(floor=fl, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))), flush=True)

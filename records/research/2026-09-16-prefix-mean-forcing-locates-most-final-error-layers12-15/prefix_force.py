#!/usr/bin/env python3
"""TAG fable-wall-repair (W-O): downstream impact of each layer's mean error. For L = -1..14 the state mean is forced to the 1e9 truth
at every layer <= L (through the head hook; the card's consistent covariance update absorbs it) and the card runs freely afterwards;
the FINAL-layer MSE as a function of L shows how much of the final error originates at or before each layer.
usage: prefix_force.py <cand> <rank>"""
import sys, json, types, importlib.util
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("prefix_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mlp = MLP([W[l] for l in range(16)], f"prefix-{rank}")
orig_pb = mod._predict_binary; LIMIT = [-1]
def forced_pb(mlp_, hook=None, **kw):
    def fhook(l, m, C, K21, aux, w):
        delta = hook(l, m, C, K21, aux, w)
        if l <= LIMIT[0]:
            return fnp.asarray(truth[l] - np.asarray(m, dtype=np.float64), dtype=fnp.float64)
        return delta
    return orig_pb(mlp_, hook=fhook, **kw)
mod._predict_binary = forced_pb
res = {}
for L in range(-1, 15):
    LIMIT[0] = L
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        out = np.asarray(mod.Estimator().predict(mlp, 2**41), dtype=np.float64)
    res[L] = float(((out[15] - truth[15]) ** 2).mean()); print(f"r{rank}: mean forced through L{L:2d} -> final MSE {res[L]:.3e}", flush=True)
json.dump(res, open(HERE / f"prefix_force_{cand.stem}_r{rank}.json", "w"), indent=1)

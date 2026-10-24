#!/usr/bin/env python3
"""TAG fable-wall-repair (W-P): oracle supplies at chosen layers using the stored 8M-sample per-unit truths (mps_truth_rank*_P8M.npz):
replace the card's per-unit pre third cumulant (diag K21) and/or per-unit pre fourth cumulant (k4, plus the diagonal of the K22 plane)
by the MC truth in the pair nonlinearity at layers in LAYERS, run the rest of the card freely, report the final MSE.
usage: oracle_diag.py <cand> <rank> <layers e.g. 12,13,14,15> <mode k3|k4|both>"""
import sys, json, types, importlib.util
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2]); LAYERS = {int(x) for x in sys.argv[3].split(",")}; MODE = sys.argv[4]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("oracle_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
tr = np.load(P2 / "public_g19_attribution_v1" / f"mps_truth_rank{rank:04d}_P8M.npz"); k3_true, k4_true, v_true = tr["mc_pre_k3"], tr["mc_pre_k4"], tr["mc_pre_v"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mlp = MLP([W[l] for l in range(16)], f"oracle-{rank}")
layer = [0]; stats = []
def wrap(fn):
    def w(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = layer[0]; layer[0] += 1
        if l in LAYERS:
            K21 = K21.copy(); k4 = k4.copy(); k22 = k22_override.copy()
            if MODE in ("k3", "both"):
                d_card = np.asarray(fnp.diag(K21)); stats.append((l, "k3", float(np.sqrt(np.mean((d_card - k3_true[l]) ** 2) / np.mean(k3_true[l] ** 2)))))
                fnp.fill_diagonal(K21, fnp.asarray(k3_true[l]))
            if MODE in ("k4", "both"):
                d_card = np.asarray(k4); stats.append((l, "k4", float(np.sqrt(np.mean((d_card - k4_true[l]) ** 2) / np.mean(k4_true[l] ** 2)))))
                k4 = fnp.asarray(k4_true[l]); fnp.fill_diagonal(k22, k4)
            return fn(mu, C, K21, k4, k22, metric)
        return fn(mu, C, K21, k4, k22_override, metric)
    return w
for name in ("_nonlinear_first_gaussian", "_nonlinear_pair_batch", "_nonlinear_terminal_live", "nonlinear"):
    setattr(mod, name, wrap(getattr(mod, name)))
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    out = np.asarray(mod.Estimator().predict(mlp, 2**41), dtype=np.float64)
mse = ((out - truth) ** 2).mean(axis=1)
print(f"r{rank} oracle {MODE} at layers {sorted(LAYERS)}: final MSE {mse[15]:.3e} | per-layer " + " ".join(f"L{l}:{mse[l]:.2e}" for l in range(10, 16)))
for l, k, rel in stats: print(f"   L{l}: card {k} rel rms error vs 8M truth {rel:.3f}")

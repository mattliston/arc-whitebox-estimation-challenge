#!/usr/bin/env python3
"""TAG fable-wall-repair (W-Q): oracle PLANE supplies at chosen layers from MPS 8M-sample pre-activation plane truths (mc_planes_mps.py
output: K21_l = cum(h_i,h_i,h_j), K22_l = cum(h_i,h_i,h_j,h_j), K31_l = cum(h_i,h_i,h_i,h_j)). Modes: k21 (full third-order plane incl.
diagonal), k22 (fourth-order plane incl. diagonal k4), both. The card runs freely elsewhere. Reports the final MSE and the card's plane
errors vs truth (off-diagonal rel rms).  usage: oracle_planes.py <cand> <rank> <planes.npz> <layers> <mode>"""
import sys, json, types, importlib.util
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, planes = Path(sys.argv[1]).resolve(), int(sys.argv[2]), np.load(sys.argv[3]); LAYERS = {int(x) for x in sys.argv[4].split(",")}; MODE = sys.argv[5]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("oraclep_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mlp = MLP([W[l] for l in range(16)], f"oraclep-{rank}")
layer = [0]; stats = []
def offrel(card, true):
    c = np.asarray(card, dtype=np.float64).copy(); t = np.asarray(true, dtype=np.float64).copy(); np.fill_diagonal(c, 0); np.fill_diagonal(t, 0)
    return float(np.sqrt(np.sum((c - t) ** 2) / np.sum(t ** 2)))
def wrap(fn):
    def w(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = layer[0]; layer[0] += 1
        if l in LAYERS:
            if MODE in ("k21", "both"):
                T = planes[f"K21_{l}"].astype(np.float64); stats.append((l, "K21 off-diag", offrel(K21, T), "diag", float(np.sqrt(np.mean((np.diag(np.asarray(K21)) - np.diag(T)) ** 2) / np.mean(np.diag(T) ** 2)))))
                K21 = fnp.asarray(T)
            if MODE in ("k22", "both"):
                T = planes[f"K22_{l}"].astype(np.float64); stats.append((l, "K22 off-diag", offrel(k22_override, T), "diag(k4)", float(np.sqrt(np.mean((np.asarray(k4) - np.diag(T)) ** 2) / np.mean(np.diag(T) ** 2)))))
                k22_override = fnp.asarray(T); k4 = fnp.asarray(np.diag(T).copy())
            return fn(mu, C, K21, k4, k22_override, metric)
        return fn(mu, C, K21, k4, k22_override, metric)
    return w
for name in ("_nonlinear_first_gaussian", "_nonlinear_pair_batch", "_nonlinear_terminal_live", "nonlinear"):
    setattr(mod, name, wrap(getattr(mod, name)))
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    out = np.asarray(mod.Estimator().predict(mlp, 2**41), dtype=np.float64)
mse = ((out - truth) ** 2).mean(axis=1)
print(f"r{rank} oracle planes {MODE} at layers {sorted(LAYERS)}: final MSE {mse[15]:.3e} | per-layer " + " ".join(f"L{l}:{mse[l]:.2e}" for l in range(10, 16)))
for l, k, rel, k2, rel2 in stats: print(f"   L{l}: card {k} rel error {rel:.3f} | {k2} rel error {rel2:.3f}")

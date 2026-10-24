#!/usr/bin/env python3
"""TAG fable-wall-repair (W-M): closed-loop training rows for the c2 card's learned mean head. The card runs exactly as deployed
(current head in the loop); at every layer the hook records the RAW per-unit features (before the head's normalisation), the incoming
recurrent state, sigma, the card mean (host), the head's current delta, and the 1e9 truth. Target for a new head = truth - host
(closed-loop error at that layer). No leak: everything recorded is computed before the head's correction is applied.
usage: rows_collect.py <cand> <rank_lo> <rank_hi>  -> rows_c2/rows_rank<r>.npz"""
import sys, json, types, importlib.util, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, lo, hi = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3])
OUT = HERE / "rows_c2"; OUT.mkdir(exist_ok=True)
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("rows_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
orig_features, orig_gnn = mod.features, mod.gnn_step
for rank in range(lo, hi + 1):
    if (OUT / f"rows_rank{rank:04d}.npz").exists():
        continue
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    rec = {k: [] for k in ("x", "state_in", "sigma", "host", "delta", "truth")}
    def features_rec(l, m, C, K21, aux, Wl):
        x = orig_features(l, m, C, K21, aux, Wl)
        rec["x"].append(np.asarray(x, dtype=np.float32)); rec["sigma"].append(np.asarray(aux["sigma"], dtype=np.float64)); rec["host"].append(np.asarray(m, dtype=np.float64)); rec["truth"].append(truth[l])
        return x
    def gnn_rec(l, x, sigma, Wl, state, P):
        rec["state_in"].append(np.asarray(state, dtype=np.float32))
        delta, state2 = orig_gnn(l, x, sigma, Wl, state, P)
        rec["delta"].append(np.asarray(delta, dtype=np.float64))
        return delta, state2
    mod.features, mod.gnn_step = features_rec, gnn_rec
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        out = np.asarray(mod.Estimator().predict(MLP([W[l] for l in range(16)], f"rows-{rank}"), 2**41), dtype=np.float64)
    mod.features, mod.gnn_step = orig_features, orig_gnn
    arr = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(OUT / f"rows_rank{rank:04d}.npz", rank=rank, final_mse=float(((out[15] - truth[15]) ** 2).mean()), **arr)
    print(f"rank {rank}: layers {arr['x'].shape[0]} x {arr['x'].shape[1:]} | final mse {float(((out[15]-truth[15])**2).mean()):.3e} | {time.time()-t0:.0f}s", flush=True)

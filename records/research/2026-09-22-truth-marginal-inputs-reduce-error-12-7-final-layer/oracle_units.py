#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-11): per-unit ORACLE supplies on the deployed 331824 card from the 8M per-unit truths
(mps_truth_rank*_P8M.npz: mc_pre_v / mc_pre_k3 / mc_pre_k4 per layer). Modes (any combination of letters): v = pre-activation
variance (diag of C, at the pair map and at the final scalar map), 3 = diag k3, 4 = per-unit k4 (+ K22 diagonal). Applied at the
listed layers (the final layer's nonlinear_scalar is wrapped too).  usage: oracle_units.py <cand> <rank> <layers> <modes e.g. v,34,v34>"""
import sys, json, types, importlib.util, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2]); LAYERS = {int(x) for x in sys.argv[3].split(",")}; MODES = sys.argv[4].split(",")
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
tr = np.load(P2 / "public_g19_attribution_v1" / f"mps_truth_rank{rank:04d}_P8M.npz"); k3_true, k4_true, v_true = tr["mc_pre_k3"], tr["mc_pre_k4"], tr["mc_pre_v"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mlp = MLP([W[l] for l in range(16)], f"oracle-{rank}")
for MODE in MODES:
    spec = importlib.util.spec_from_file_location(f"oracle_cand_{MODE}", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    layer = [0]
    def wrap(fn):
        def w(mu, C, K21, k4, k22_override=None, metric='identity'):
            l = layer[0]; layer[0] += 1
            if l in LAYERS:
                K21 = K21.copy(); k4 = k4.copy(); k22 = k22_override.copy() if k22_override is not None else None
                if "v" in MODE:
                    C = C.copy(); fnp.fill_diagonal(C, fnp.asarray(v_true[l]))
                if "3" in MODE:
                    fnp.fill_diagonal(K21, fnp.asarray(k3_true[l]))
                if "4" in MODE:
                    k4 = fnp.asarray(k4_true[l])
                    if k22 is not None: fnp.fill_diagonal(k22, k4)
                return fn(mu, C, K21, k4, k22, metric)
            return fn(mu, C, K21, k4, k22_override, metric)
        return w
    def wrap_scalar(fn):
        def w(mu, v, k3, k4):
            l = 15
            if l in LAYERS:
                if "v" in MODE: v = fnp.asarray(v_true[l])
                if "3" in MODE: k3 = fnp.asarray(k3_true[l])
                if "4" in MODE: k4 = fnp.asarray(k4_true[l])
            return fn(mu, v, k3, k4)
        return w
    for name in ("_nonlinear_first_gaussian", "_nonlinear_pair_batch", "_nonlinear_terminal_live", "nonlinear"):
        setattr(mod, name, wrap(getattr(mod, name)))
    mod.nonlinear_scalar = wrap_scalar(mod.nonlinear_scalar)
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        out = np.asarray(mod.Estimator().predict(mlp, 2**41), dtype=np.float64)
    mse = ((out - truth) ** 2).mean(axis=1)
    print(json.dumps(dict(rank=rank, mode=MODE, layers=sorted(LAYERS), final_mse=float(mse[15]), per_layer={l: float(mse[l]) for l in range(8, 16)}, wall=round(time.time() - t0, 1))), flush=True)

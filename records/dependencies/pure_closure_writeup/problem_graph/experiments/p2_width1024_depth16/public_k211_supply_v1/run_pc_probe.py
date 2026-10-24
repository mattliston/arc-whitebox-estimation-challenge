#!/usr/bin/env python3
"""Predictor-corrector supply: double-pass K estimation. Truth on 3 rows."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"

def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec); sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m

def instrument(mod, pc_gain):
    """pc_gain: 0 = plain supply; g>0 = amplify K by (1+g) as the
    predictor-corrector's closed-form limit (the cascade underestimates K
    itself by roughly the same fraction the planes were short)."""
    state = {"K22resid": None, "W": None}
    orig_stars = mod._reset_stars
    orig_assemble = mod._assemble_quartic_provider
    orig_push = mod._push_quartic
    def stars_wrap(*args, **kwargs):
        result = orig_stars(*args, **kwargs)
        k4, k31, k22 = args[2], args[3], args[4]
        qpp = kwargs.get("quartic_provider_planes", args[9] if len(args) > 9 else None)
        diag_ctx = kwargs.get("diagonal_context", args[10] if len(args) > 10 else None)
        k4r, k31r, k22r = mod._quartic_residual(k4, k31, k22, qpp)
        arr = np.asarray(k22r, dtype=np.float64).view(np.ndarray)
        sym = 0.5 * (arr + arr.T)
        var = np.asarray(args[6], dtype=np.float64).view(np.ndarray)
        cov = np.asarray(args[5], dtype=np.float64).view(np.ndarray)
        pairing = np.outer(var, var) + 2.0 * cov * cov
        np.fill_diagonal(pairing, 0.0); np.fill_diagonal(sym, 0.0)
        state["K22resid"] = (1.0 + pc_gain) * (sym - float(result[2]) * pairing)
        return result
    def push_wrap(star_h, pairing_alpha, weight_context, pre_covariance, pre_variance,
                  diagonal_context, fusion_context):
        state["W"] = np.asarray(weight_context[0], dtype=np.float64).view(np.ndarray)
        return orig_push(star_h, pairing_alpha, weight_context, pre_covariance, pre_variance,
                         diagonal_context, fusion_context)
    def assemble_wrap(host_planes, provider_planes):
        k4, k31, k22 = orig_assemble(host_planes, provider_planes)
        if state["K22resid"] is not None and state["W"] is not None:
            W = state["W"]; W2 = W * W
            K = state["K22resid"]
            KW2 = K @ W2
            S = 3.0 * np.einsum("ia,ia->a", W2, KW2)
            k4 = k4 + mod.fnp.asarray(S, dtype=mod.fnp.float64)
            W2TK = W2.T @ K
            k31_add = 3.0 * ((W2TK * W.T) @ W)
            k22_add = W2.T @ KW2
            np.fill_diagonal(k31_add, 0.0); np.fill_diagonal(k22_add, 0.0)
            k31 = k31 + mod.fnp.asarray(k31_add, dtype=mod.fnp.float64)
            k22 = k22 + mod.fnp.asarray(k22_add, dtype=mod.fnp.float64)
        return k4, k31, k22
    mod._reset_stars = stars_wrap
    mod._push_quartic = push_wrap
    mod._assemble_quartic_provider = assemble_wrap

def main():
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)
    for rank in (35, 100, 201):
        line = {"rank": rank}
        for tag, g in (("g0", 0.0), ("g10", 0.10), ("g20", 0.20)):
            mod = load(f"pc_{tag}_{rank}", BASE)
            instrument(mod, g)
            entry = entries[rank]
            rmp = STORE / entry["path"]
            row = json.loads(rmp.read_text())
            weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
            mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                      seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"pc-{rank}")
            mlp.validate()
            with threadpool_limits(limits=2):
                with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
                    out = np.asarray(mod._predict(mlp), dtype=np.float64)
            line[tag] = round(float(np.mean((out[15] - targets[rank, 15]) ** 2)) * 1e8, 4)
        print(json.dumps(line), flush=True)

if __name__ == "__main__":
    main()

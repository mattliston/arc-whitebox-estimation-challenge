#!/usr/bin/env python3
"""Truth: base vs full-rank K22 supply host raw on 8 rows."""
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
ROWS = [35, 65, 100, 130, 160, 190, 220, 234]

def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec); sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m

def instrument(mod, supply_on):
    state = {"K22resid": None, "W": None}
    orig_reset_stars = mod._reset_stars
    orig_assemble = mod._assemble_quartic_provider
    orig_push = mod._push_quartic

    def stars_wrap(k21_target, cp_post_k21, k4, k31, k22, covariance, variance,
                   covariance2, variance_outer, quartic_provider_planes=None,
                   diagonal_context=None):
        result = orig_reset_stars(k21_target, cp_post_k21, k4, k31, k22, covariance,
                                  variance, covariance2, variance_outer,
                                  quartic_provider_planes, diagonal_context)
        if supply_on:
            k4r, k31r, k22r = mod._quartic_residual(k4, k31, k22, quartic_provider_planes)
            arr = np.asarray(k22r, dtype=np.float64).view(np.ndarray)
            sym = 0.5 * (arr + arr.T)
            var = np.asarray(variance, dtype=np.float64).view(np.ndarray)
            cov = np.asarray(covariance, dtype=np.float64).view(np.ndarray)
            pairing = np.outer(var, var) + 2.0 * cov * cov
            np.fill_diagonal(pairing, 0.0)
            np.fill_diagonal(sym, 0.0)
            state["K22resid"] = sym - float(result[2]) * pairing
        return result

    def push_wrap(star_h, pairing_alpha, weight_context, pre_covariance, pre_variance,
                  diagonal_context, fusion_context):
        state["W"] = np.asarray(weight_context[0], dtype=np.float64).view(np.ndarray)
        return orig_push(star_h, pairing_alpha, weight_context, pre_covariance,
                         pre_variance, diagonal_context, fusion_context)

    def assemble_wrap(host_planes, provider_planes):
        k4, k31, k22 = orig_assemble(host_planes, provider_planes)
        if supply_on and state["K22resid"] is not None and state["W"] is not None:
            W = state["W"]; W2 = W * W
            K = state["K22resid"]
            KW2 = K @ W2
            S = 3.0 * np.einsum("ia,ia->a", W2, KW2)
            k4 = k4 + mod.fnp.asarray(S, dtype=mod.fnp.float64)
            W2TK = W2.T @ K
            k31_add = 3.0 * ((W2TK * W.T) @ W)
            k22_add = W2.T @ KW2
            np.fill_diagonal(k31_add, 0.0)
            np.fill_diagonal(k22_add, 0.0)
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
    mods = {}
    for tag, on in (("base", False), ("k22full", True)):
        m = load(f"kft_{tag}", BASE)
        instrument(m, on)
        mods[tag] = m
    base_mse, full_mse = [], []
    for rank in ROWS:
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"kft-{rank}")
        mlp.validate()
        truth_final = targets[rank, 15]
        for tag, store in (("base", base_mse), ("k22full", full_mse)):
            with threadpool_limits(limits=2):
                with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
                    out = np.asarray(mods[tag]._predict(mlp), dtype=np.float64)
            store.append(float(np.mean((out[15] - truth_final) ** 2)))
        print(json.dumps({"rank": rank, "base": round(base_mse[-1]*1e8, 4),
                          "k22full": round(full_mse[-1]*1e8, 4)}), flush=True)
    b, f = np.array(base_mse), np.array(full_mse)
    print(json.dumps({"mean_rel_change_HOST_RAW": float(f.mean()/b.mean() - 1.0),
                      "wins": int((f < b).sum()), "n": len(ROWS)}))

if __name__ == "__main__":
    main()

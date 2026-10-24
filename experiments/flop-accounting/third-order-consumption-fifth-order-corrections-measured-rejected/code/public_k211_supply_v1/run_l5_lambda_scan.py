#!/usr/bin/env python3
"""TAG l5-lambda-scan-v1: shared amplitude lam=0.5 on BOTH order-3 sides.
Prior cells: pair-only +5-11 percent; univariate-only +186 percent; both +2.85."""
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
SHELL = P2 / "public_lambda3_shell_probe_v1/estimator_x.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROWS = [35, 100, 160]

sys.path.insert(0, str(HERE))
from run_ue5_probe import patch_ue5, load
LAM = 0.5


def instrument_supply(mod):
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
        if state["K22resid"] is not None and state["W"] is not None:
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
    base = {}
    for part in (0, 1):
        base.update(json.loads(
            (P2 / f"public_supply_card_v1/card_raw200_part{part}.json").read_text()))
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)
    rel = []
    for rank in ROWS:
        mod = load(f"shue5_{rank}", SHELL)
        instrument_supply(mod)
        patch_ue5(mod, lam=LAM)
        orig_w = mod._edgeworth_weights
        fnpm = mod.fnp
        def w_wrap(k3, k21, k4, k31, k22, fusion_context):
            weights = orig_w(k3, k21, k4, k31, k22, fusion_context)
            for i in range(17, 35):
                fnpm.copyto(weights[i], weights[i] * LAM)
            return weights
        mod._edgeworth_weights = w_wrap

        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"shue5-{rank}")
        mlp.validate()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**44, wall_time_limit_s=900.0, quiet=True):
                host_v, _, _ = mod.predict_with_native_tape(mlp)
        pred = np.asarray(host_v, dtype=np.float64)
        assert pred.shape == (16, 1024), pred.shape
        mse = float(np.mean((targets[rank][15] - pred[15]) ** 2))
        b = base[str(rank)]["mse"]
        rel.append(mse / b - 1.0)
        print(json.dumps({"rank": rank, "card": round(b * 1e8, 4),
                          "lam05": round(mse * 1e8, 4),
                          "rel": round(mse / b - 1.0, 4)}), flush=True)
    print(json.dumps({"mean_rel_change": round(float(np.mean(rel)), 4),
                      "wins": int(sum(r < 0 for r in rel)), "n": len(rel)}))


if __name__ == "__main__":
    main()

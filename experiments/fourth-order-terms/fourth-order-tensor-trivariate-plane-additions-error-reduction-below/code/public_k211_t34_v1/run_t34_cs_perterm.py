#!/usr/bin/env python3
"""Staged chain+star probe: flag arm, per-term ranking, top-3 subset pricing.

TAG=k211-t34-chainstar  Stage 1: the affordable class only (T2 + T3/T4
chain+star groups, triangles OFF) on ranks 35/100/201; supply baselines
reused from K211_T34_RESULT.json (deterministic - the fresh supply arm
matched the banked card numbers exactly, twice).  Bar: mean effect <= -0.5%
to continue.  Stage 2 (bar met): single-group arms on rank 35 -> raw-effect
ranking of all 10 canonical chain+star groups.  Stage 3: the top-3 subset
on all 3 rows = the priced deployable candidate (~6.4e9/layer, ~90e9/card
per retained group).  Writes K211_T34_CS_RESULT.json.  No submission.
"""
import importlib.util, json, os, sys, time
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(P2 / "public_k211_supply_v1"))
from t34_supply_core import (T2_GROUPS, T34_CHAINSTAR_GROUPS,
                             kappa211_t34_supply)

BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
PRIOR = HERE / "K211_T34_RESULT.json"
RESULT = HERE / "K211_T34_CS_RESULT.json"
ROWS = (35, 100, 201)
CHAINSTAR = T2_GROUPS + T34_CHAINSTAR_GROUPS
BAR_MEAN_EFFECT = -0.005
COST_PER_GROUP_PER_LAYER = 6.4e9
ASSEMBLES = 14


def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec); sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def instrument(mod, groups):
    state = {"K22resid": None, "W": None, "stash": None}
    orig_reset_stars = mod._reset_stars
    orig_full_reset = mod._full_relu_reset
    orig_assemble = mod._assemble_quartic_provider
    orig_push = mod._push_quartic

    def full_reset_wrap(pre_mean, pre_covariance, pre_k21, pre_k4, pre_k31,
                        pre_k22, coefficients, gaussian_raw, sigma, sigma2,
                        sigma3, alpha, pdf, diagonal_context, fusion_context):
        c = np.asarray(coefficients, dtype=np.float64).view(np.ndarray)
        sig = np.asarray(sigma, dtype=np.float64).view(np.ndarray)
        cov = np.asarray(pre_covariance, dtype=np.float64).view(np.ndarray)
        rho = np.clip(cov / np.maximum(sig[:, None] * sig[None, :], 1e-30),
                      -1.0, 1.0)
        state["stash"] = {"c1": c[0, 0:5].copy(), "c2": c[1, 0:5].copy(),
                          "rho": rho}
        return orig_full_reset(pre_mean, pre_covariance, pre_k21, pre_k4,
                               pre_k31, pre_k22, coefficients, gaussian_raw,
                               sigma, sigma2, sigma3, alpha, pdf,
                               diagonal_context, fusion_context)

    def stars_wrap(k21_target, cp_post_k21, k4, k31, k22, covariance, variance,
                   covariance2, variance_outer, quartic_provider_planes=None,
                   diagonal_context=None):
        result = orig_reset_stars(k21_target, cp_post_k21, k4, k31, k22,
                                  covariance, variance, covariance2,
                                  variance_outer, quartic_provider_planes,
                                  diagonal_context)
        k4r, k31r, k22r = mod._quartic_residual(k4, k31, k22,
                                                quartic_provider_planes)
        arr = np.asarray(k22r, dtype=np.float64).view(np.ndarray)
        sym = 0.5 * (arr + arr.T)
        var = np.asarray(variance, dtype=np.float64).view(np.ndarray)
        cov = np.asarray(covariance, dtype=np.float64).view(np.ndarray)
        pairing = np.outer(var, var) + 2.0 * cov * cov
        np.fill_diagonal(pairing, 0.0)
        np.fill_diagonal(sym, 0.0)
        state["K22resid"] = sym - float(result[2]) * pairing
        return result

    def push_wrap(star_h, pairing_alpha, weight_context, pre_covariance,
                  pre_variance, diagonal_context, fusion_context):
        state["W"] = np.asarray(weight_context[0],
                                dtype=np.float64).view(np.ndarray)
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
            if groups and state["stash"] is not None:
                st = state["stash"]
                S211 = kappa211_t34_supply(W, st["rho"], st["c1"], st["c2"],
                                           groups=groups)
                k4 = k4 + mod.fnp.asarray(S211, dtype=mod.fnp.float64)
        return k4, k31, k22

    mod._full_relu_reset = full_reset_wrap
    mod._reset_stars = stars_wrap
    mod._push_quartic = push_wrap
    mod._assemble_quartic_provider = assemble_wrap


def forward(rank, groups, entries, targets, tag):
    entry = entries[rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                      allow_pickle=False)
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"cs-{tag}-{rank}")
    mlp.validate()
    mod = load(f"cs_{tag}_{rank}_{os.getpid()}", BASE)
    instrument(mod, groups)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=3600.0,
                                 quiet=True):
            out = np.asarray(mod._predict(mlp), dtype=np.float64)
    mse = float(np.mean((out[15] - targets[rank, 15]) ** 2))
    wall = time.perf_counter() - started
    print(json.dumps({"tag": tag, "rank": rank, "mse_e8": round(mse * 1e8, 4),
                      "wall": round(wall, 1)}), flush=True)
    return mse


def main():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    prior = json.loads(PRIOR.read_text())
    supply_mse = {r: prior["truth_rows"][str(r)]["supply"]["final_mse"]
                  for r in ROWS}
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)

    # Stage 1: full chain+star arm on 3 rows.
    stage1 = {}
    for rank in ROWS:
        mse = forward(rank, CHAINSTAR, entries, targets, "chainstar")
        stage1[rank] = {"supply": supply_mse[rank], "chainstar": mse,
                        "effect_fraction": mse / supply_mse[rank] - 1.0}
    mean_effect = float(np.mean([stage1[r]["effect_fraction"] for r in ROWS]))
    bar_met = mean_effect <= BAR_MEAN_EFFECT
    print(json.dumps({"stage1_mean_effect": mean_effect,
                      "bar_met": bar_met}), flush=True)

    stage2 = None
    stage3 = None
    if bar_met:
        # Stage 2: single-group arms on rank 35.
        singles = {}
        for tag in CHAINSTAR:
            mse = forward(35, (tag,), entries, targets, f"only-{tag}")
            singles[tag] = {"mse": mse,
                            "effect_fraction": mse / supply_mse[35] - 1.0}
        ranking = sorted(singles, key=lambda t: singles[t]["effect_fraction"])
        stage2 = {"row": 35, "singles": singles, "ranking": ranking,
                  "sum_of_single_effects":
                      float(sum(s["effect_fraction"] for s in singles.values())),
                  "combined_chainstar_effect_rank35":
                      stage1[35]["effect_fraction"]}
        # Stage 3: top-3 subset on 3 rows.
        top3 = tuple(ranking[:3])
        subset_rows = {}
        for rank in ROWS:
            mse = forward(rank, top3, entries, targets, "top3")
            subset_rows[rank] = {"supply": supply_mse[rank], "top3": mse,
                                 "effect_fraction":
                                     mse / supply_mse[rank] - 1.0}
        stage3 = {
            "groups": list(top3),
            "rows": {str(k): v for k, v in subset_rows.items()},
            "mean_effect_fraction": float(np.mean(
                [subset_rows[r]["effect_fraction"] for r in ROWS])),
            "cost_per_layer_flops": COST_PER_GROUP_PER_LAYER * 3,
            "cost_per_card_flops": COST_PER_GROUP_PER_LAYER * 3 * ASSEMBLES,
        }

    verdict = "BAR_FAILED_LANE_CLOSES" if not bar_met else (
        "TOP3_PRICED" if stage3 else "INCOMPLETE")
    result = {
        "schema": "k211-t34-chainstar-result-v1",
        "verdict": verdict,
        "bar_mean_effect_to_continue": BAR_MEAN_EFFECT,
        "supply_baseline_source": "K211_T34_RESULT.json (deterministic "
        "forwards; the fresh supply arm matched banked card_raw200 exactly)",
        "stage1_chainstar_3rows": {str(k): v for k, v in stage1.items()},
        "stage1_mean_effect_fraction": mean_effect,
        "full_t34_mean_effect_for_reference":
            prior["mean_t34_effect_fraction"],
        "stage2_per_term_rank35": stage2,
        "stage3_top3_subset": stage3,
        "cost_model": {
            "per_group_per_layer_flops": COST_PER_GROUP_PER_LAYER,
            "per_group_per_card_flops": COST_PER_GROUP_PER_LAYER * ASSEMBLES,
            "assembles": ASSEMBLES,
            "note": "3 GEMMs of 2*1024^3 per group per assemble; shared "
            "R-power elementwise maps negligible; whole-card margin 288e9",
        },
    }
    tmp = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULT)
    print(json.dumps({"verdict": verdict,
                      "stage1_mean": mean_effect,
                      "stage3": stage3 and stage3["mean_effect_fraction"]},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Truth: supply vs supply+F-form vs supply+kappa211-planes vs both, 3 rows.

TAG=missing-routes-truth  Monkeypatch harness (run_t34_truth.py pattern):
the full-rank K22 supply verbatim in every arm, plus at every assemble
  fform:     k22 += 2F (the never-carried ij|ij route, diagonal zeroed)
  planes211: k31 += 1.5(R1+R2), k22 += 0.5(P1+P1^T)+2P2 (kappa211 <=4,
             all 11 canonical groups incl. explicit mirrors, diag zeroed;
             the kappa211 k4-diagonal supply stays OFF - banked separately)
  both:      fform + planes211
Raw host (_predict) final-layer MSE vs truth on ranks 35/100/201; supply
baselines reused from ../public_k211_t34_v1/K211_T34_RESULT.json
(deterministic, cross-checked against the banked card twice).  Per-layer
RMS of every added plane recorded.  Writes MISSING_ROUTES_RESULT.json.
Decision bar (team lead): mean <= -2 percent on any arm keeps the lane
open; otherwise it closes with receipts.  No submission, no store writes.
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
sys.path.insert(0, str(P2 / "public_k211_t34_v1"))
from missing_routes_core import f_form_k22_missing, kappa211_plane_supplies

BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
PRIOR = P2 / "public_k211_t34_v1/K211_T34_RESULT.json"
RESULT = HERE / "MISSING_ROUTES_RESULT.json"
# Trimmed matrix (team-lead speed directive 2026-08-26): rows 35/100 only,
# planes211-only arm dropped - fform and both are the decision cells; the
# first (killed) launch already measured rank-35 fform at +0.019 percent,
# reproduced here deterministically.
ROWS = (35, 100)
ARMS = ("fform", "both")


def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec); sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def instrument(mod, use_fform, use_planes, norms):
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
            record = {}
            if use_fform:
                fadd = f_form_k22_missing(W, K)
                k22 = k22 + mod.fnp.asarray(fadd, dtype=mod.fnp.float64)
                record["fform_k22_rms"] = float(np.sqrt(np.mean(fadd * fadd)))
                record["carried_k22_rms"] = float(
                    np.sqrt(np.mean(k22_add * k22_add)))
            if use_planes and state["stash"] is not None:
                st = state["stash"]
                p31, p22 = kappa211_plane_supplies(W, st["rho"], st["c1"],
                                                   st["c2"])
                k31 = k31 + mod.fnp.asarray(p31, dtype=mod.fnp.float64)
                k22 = k22 + mod.fnp.asarray(p22, dtype=mod.fnp.float64)
                record["k211_k31_rms"] = float(np.sqrt(np.mean(p31 * p31)))
                record["k211_k22_rms"] = float(np.sqrt(np.mean(p22 * p22)))
                record["carried_k31_rms"] = float(
                    np.sqrt(np.mean(k31_add * k31_add)))
            if record:
                norms.append(record)
        return k4, k31, k22

    mod._full_relu_reset = full_reset_wrap
    mod._reset_stars = stars_wrap
    mod._push_quartic = push_wrap
    mod._assemble_quartic_provider = assemble_wrap


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

    rows_out = {}
    all_norms = {}
    for rank in ROWS:
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                          allow_pickle=False)
        mlp = MLP(width=1024, depth=16,
                  weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"mr-{rank}")
        mlp.validate()
        truth_final = targets[rank, 15]
        arms = {"supply": {"final_mse": supply_mse[rank],
                           "source": "reused-prior-deterministic"}}
        for tag in ARMS:
            use_f = tag in ("fform", "both")
            use_p = tag in ("planes211", "both")
            norms = []
            mod = load(f"mr_{tag}_{rank}", BASE)
            instrument(mod, use_f, use_p, norms)
            started = time.perf_counter()
            with threadpool_limits(limits=4):
                with flops.BudgetContext(flop_budget=2**43,
                                         wall_time_limit_s=14400.0,
                                         quiet=True):
                    out = np.asarray(mod._predict(mlp), dtype=np.float64)
            mse = float(np.mean((out[15] - truth_final) ** 2))
            arms[tag] = {"final_mse": mse,
                         "effect_fraction": mse / supply_mse[rank] - 1.0,
                         "finite": bool(np.isfinite(out).all()),
                         "wall_seconds": time.perf_counter() - started}
            if tag == "both":
                all_norms[rank] = norms
            print(json.dumps({"rank": rank, "arm": tag,
                              "mse_e8": round(mse * 1e8, 4),
                              "effect_pct": round(arms[tag]["effect_fraction"] * 100, 3),
                              "wall": round(arms[tag]["wall_seconds"], 1)}),
                  flush=True)
        rows_out[rank] = arms

    means = {tag: float(np.mean([rows_out[r][tag]["effect_fraction"]
                                 for r in ROWS])) for tag in ARMS}
    bar = -0.02
    verdict = ("CONTINUE_" + max(means, key=lambda t: -means[t]).upper()
               if any(v <= bar for v in means.values())
               else "SUB_MATERIAL_LANE_CLOSES")
    verify = json.loads((HERE / "VERIFY_MISSING_ROUTES.json").read_text())
    result = {
        "schema": "missing-routes-result-v1",
        "verdict": verdict,
        "bar_mean_effect_to_continue": bar,
        "arm_definition": {
            "fform": "supply + 2F on k22 (the never-carried ij|ij route)",
            "planes211": "supply + kappa211<=4 k31/k22 planes (all 11 "
            "canonical groups, explicit mirrors; k4-diag OFF, banked "
            "separately at -0.66 percent)",
            "both": "supply + fform + planes211",
        },
        "brute_check": {"worst_rel_err": verify["worst_rel_err"],
                        "pass": verify["pass"]},
        "truth_rows": {str(k): v for k, v in rows_out.items()},
        "mean_effect_fractions": means,
        "per_layer_rms_both_arm": {str(k): v for k, v in all_norms.items()},
        "supply_baseline_source": str(PRIOR),
    }
    tmp = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULT)
    print(json.dumps({"verdict": verdict, "means": means}, indent=1),
          flush=True)


if __name__ == "__main__":
    main()

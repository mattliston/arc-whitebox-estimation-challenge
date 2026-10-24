#!/usr/bin/env python3
"""Truth: k22full supply vs k22full + full kappa211(<=4) on 3 rows.

TAG=k211-t34-truth  Monkeypatch harness combining public_k211_supply_v1/
run_k22full_raw.py (the full-rank K22 residual supply, verbatim) with the
kappa211 supply: c-table channels q=0..4 + rho stashed at _full_relu_reset
(run_supply_probe.py pattern), W at _push_quartic, and S = 3*sum W2 W W
kappa211(i;j,k) added to k4 at _assemble_quartic_provider.  The t34 arm
adds the COMPLETE order<=4 kappa211 (T2 included; T2 alone was measured
tiny, and the per-layer norm split recorded here shows the T2 vs T3+T4
decomposition).  Raw host (_predict) final-layer MSE vs truth on ranks
35/100/201, both arms fresh and paired.  Writes K211_T34_RESULT.json.
No submission, no store writes.
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
from t34_supply_core import FORMULA_TERMS, kappa211_t34_supply
from supply_core import kappa211_supply as banked_t2_supply

BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
RESULT = HERE / "K211_T34_RESULT.json"
ROWS = (35, 100, 201)


def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec); sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def instrument(mod, t34_on, norms):
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
            if t34_on and state["stash"] is not None:
                st = state["stash"]
                S34 = kappa211_t34_supply(W, st["rho"], st["c1"], st["c2"],
                                          include_t2=False)
                S2 = banked_t2_supply(W, st["rho"], st["c1"][0], st["c1"][1],
                                      st["c1"][2], st["c2"][1], st["c2"][2])
                k4 = k4 + mod.fnp.asarray(S2 + S34, dtype=mod.fnp.float64)
                norms.append({
                    "k22full_S_rms": float(np.sqrt(np.mean(S * S))),
                    "k211_t2_rms": float(np.sqrt(np.mean(S2 * S2))),
                    "k211_t34_rms": float(np.sqrt(np.mean(S34 * S34))),
                })
        return k4, k31, k22

    mod._full_relu_reset = full_reset_wrap
    mod._reset_stars = stars_wrap
    mod._push_quartic = push_wrap
    mod._assemble_quartic_provider = assemble_wrap


def main():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)

    rows_out = {}
    layer_norms = {}
    for rank in ROWS:
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                          allow_pickle=False)
        mlp = MLP(width=1024, depth=16,
                  weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"t34-{rank}")
        mlp.validate()
        truth_final = targets[rank, 15]
        arms = {}
        for tag, on in (("supply", False), ("supply_t34", True)):
            norms = []
            mod = load(f"t34_{tag}_{rank}", BASE)
            instrument(mod, on, norms)
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                with flops.BudgetContext(flop_budget=2**43,
                                         wall_time_limit_s=7200.0,
                                         quiet=True):
                    out = np.asarray(mod._predict(mlp), dtype=np.float64)
            mse = float(np.mean((out[15] - truth_final) ** 2))
            arms[tag] = {"final_mse": mse,
                         "finite": bool(np.isfinite(out).all()),
                         "wall_seconds": time.perf_counter() - started}
            if on:
                layer_norms[rank] = norms
            print(json.dumps({"rank": rank, "arm": tag,
                              "mse_e8": round(mse * 1e8, 4),
                              "wall": round(arms[tag]["wall_seconds"], 1)}),
                  flush=True)
        arms["t34_effect_fraction"] = (
            arms["supply_t34"]["final_mse"] / arms["supply"]["final_mse"] - 1.0
        )
        rows_out[rank] = arms

    effects = [rows_out[r]["t34_effect_fraction"] for r in ROWS]
    verify = json.loads((HERE / "VERIFY_T34.json").read_text())
    result = {
        "schema": "k211-t34-result-v1",
        "term_list": {
            "source": "public_trivariate_derivation_v1/FORMULA.json kappa211",
            "convention": "(p, q, t, coefficient) = exponents of "
            "(r_ij, r_ik, r_jk), coef 1/(p! q! t!); i-functional B[p+q] "
            "(minus 2 A[p] A[q] when t=0), j A[p+t], k A[q+t]; "
            "S(a) = 3 * sum_distinct W2 W W kappa211",
            "orders_implemented": [3, 4],
            "terms": [list(t) for t in FORMULA_TERMS
                      if t[0] + t[1] + t[2] in (3, 4)],
            "grouping": "mirror pairs (p,q,t)<->(q,p,t) collapsed 2x; "
            "classes: star t=0, chain q=0 (doubled), triangles per-column "
            "GEMM with the (1,1,1)/(1,2,1) chain shared",
        },
        "brute_check": {"worst_rel_err": verify["worst"],
                        "tolerance": verify["tolerance_rel"],
                        "pass": verify["pass"]},
        "arm_definition": {
            "supply": "workspace_v4 + full-rank K22 residual supply "
            "(run_k22full_raw.py instrument, verbatim)",
            "supply_t34": "same + complete kappa211 order<=4 (T2+T3+T4) "
            "added to k4 at every assemble",
        },
        "truth_rows": {str(k): v for k, v in rows_out.items()},
        "mean_t34_effect_fraction": float(np.mean(effects)),
        "per_layer_supply_rms": {str(k): v for k, v in layer_norms.items()},
    }
    tmp = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULT)
    print(json.dumps({
        "six_mses": {str(r): {a: rows_out[r][a]["final_mse"]
                              for a in ("supply", "supply_t34")} for r in ROWS},
        "mean_t34_effect": result["mean_t34_effect_fraction"],
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()

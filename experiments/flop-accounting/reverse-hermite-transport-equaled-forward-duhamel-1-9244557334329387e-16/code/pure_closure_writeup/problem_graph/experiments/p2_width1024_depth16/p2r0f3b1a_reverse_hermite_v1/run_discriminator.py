#!/usr/bin/env python3
"""Frozen target-free discriminator for P2R0F3B1A.

This program deliberately has no estimator, observer, selection, or challenge-data
import.  It uses synthetic weights, the small-n exact-ledger implementation, and
an antithetic Monte Carlo reference.  RESULT.json is terminal for this receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parent / "public_exact_ledger_v1"
sys.path.insert(0, str(LEDGER))
import ledger_rig as rig  # noqa: E402
import ledger_verify as verify  # noqa: E402

PREREG = HERE / "PREREGISTRATION.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
RESULT = HERE / "RESULT.json"
RUNGS = ("K4", "K5", "K7", "K9", "K12")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relmax(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def geometric_mean(xs) -> float:
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.exp(np.mean(np.log(xs))))


def frozen_weights(family_seed: int, width: int, depth: int):
    if width not in (8, 16) or depth > 5:
        raise ValueError((width, depth))
    rng = np.random.default_rng(family_seed)
    master = [rng.standard_normal((16, 16)) for _ in range(5)]
    scale = np.sqrt(2.0 / width)
    return [np.asarray(M[:width, :width] * scale, dtype=np.float64)
            for M in master[:depth]]


def full_state(post, n: int):
    zero = np.zeros(n)
    zmat = np.zeros((n, n))
    out = {"m": post["m"], "Sigma": post["Sigma"],
           "k3d": zero.copy(), "K21": zmat.copy(),
           "k4d": zero.copy(), "K31": zmat.copy(), "K22": zmat.copy(),
           "k5d": zero.copy(), "K41": zmat.copy(), "K32": zmat.copy()}
    for key in ("k3d", "K21", "k4d", "K31", "K22", "k5d", "K41", "K32"):
        if key in post:
            out[key] = post[key]
    return out


def local_sources(pre, layer_index: int):
    """Independent c-table implementation, checked against mean-map helpers."""
    mu = np.asarray(pre["m"], dtype=np.float64)
    var = np.maximum(np.diag(pre["Sigma"]).copy(), 1e-12)
    sd = np.sqrt(var)
    if layer_index == 0:
        z = np.zeros_like(mu)
        return {r: z.copy() for r in RUNGS}, 0.0, {
            "g3_rms": 0.0, "g4_rms": 0.0, "g5_hat_rms": 0.0}

    g3 = np.asarray(pre["k3d"], dtype=np.float64) / sd ** 3
    g4 = np.asarray(pre["k4d"], dtype=np.float64) / sd ** 4
    g5 = 5.0 * g3 * g4 - (40.0 / 9.0) * g3 ** 3
    table, _, _, _ = rig.c_table(mu, var, pmax=1, qmax=12)
    C = table[1]
    s5 = (g5 / 120.0) * C[5]
    s7 = s5 + (g3 * g4 / 144.0) * C[7]
    s9 = s7 + (g3 ** 3 / 1296.0) * C[9]
    s12 = (s9 + (g3 * g5 / 720.0 + g4 ** 2 / 1152.0) * C[8]
           + (g3 ** 2 * g4 / 1728.0) * C[10]
           + (g3 ** 4 / 31104.0) * C[12])
    got = {"K4": np.zeros_like(mu), "K5": s5, "K7": s7,
           "K9": s9, "K12": s12}

    incumbent = verify.model_mean(mu, var, g3, g4, "incumbent")
    _, col = verify.mean_columns(mu, var, qmax=12)
    ref5 = incumbent + (g5 / 120.0) * col[5] - incumbent
    ref7 = ref5 + (g3 * g4 / 144.0) * col[7]
    order3 = verify.model_mean(mu, var, g3, g4, "order3", g5=g5)
    ref9 = order3 - incumbent
    order4 = verify.model_mean(mu, var, g3, g4, "order4", g5=g5,
                               g6=np.zeros_like(g3))
    ref12 = order4 - incumbent
    refs = {"K4": np.zeros_like(mu), "K5": ref5, "K7": ref7,
            "K9": ref9, "K12": ref12}
    identity = max(relmax(got[r], refs[r]) for r in RUNGS)
    stats = {"g3_rms": rms(g3), "g4_rms": rms(g4),
             "g5_hat_rms": rms(g5),
             **{f"source_{r}_rms": rms(got[r]) for r in RUNGS}}
    return got, identity, stats


def capture_level_c(Ws):
    n = Ws[0].shape[0]
    zero = np.zeros(n)
    zmat = np.zeros((n, n))
    state = {"m": zero.copy(), "Sigma": np.eye(n),
             "k3d": zero.copy(), "K21": zmat.copy(),
             "k4d": zero.copy(), "K31": zmat.copy(), "K22": zmat.copy(),
             "k5d": zero.copy(), "K41": zmat.copy(), "K32": zmat.copy()}
    posts, phis, sources, source_stats = [], [], [], []
    source_identity_max_rel = 0.0
    for li, W in enumerate(Ws):
        pre = rig.push_state(W, state, "C")
        for key in ("k3d", "k4d", "k5d"):
            pre.setdefault(key, zero.copy())
        for key in ("K21", "K31", "K22", "K41", "K32"):
            pre.setdefault(key, zmat.copy())
        var = np.maximum(np.diag(pre["Sigma"]).copy(), 1e-12)
        phis.append(rig.norm_cdf(pre["m"] / np.sqrt(var)))
        src, ident, stats = local_sources(pre, li)
        sources.append(src)
        source_stats.append({"layer": li + 1, **stats})
        source_identity_max_rel = max(source_identity_max_rel, ident)
        post = rig.reset_state(pre, "A" if li == 0 else "C", n)
        posts.append(post)
        state = full_state(post, n)

    canonical = rig.run_ledger(Ws, "C")
    baseline_max_rel = 0.0
    for a, b in zip(posts, canonical):
        if set(a) != set(b):
            raise AssertionError((set(a), set(b)))
        for key in a:
            baseline_max_rel = max(baseline_max_rel, relmax(a[key], b[key]))
    return {"posts": posts, "phis": phis, "sources": sources,
            "source_stats": source_stats,
            "source_identity_max_rel": source_identity_max_rel,
            "baseline_reproduction_max_rel": baseline_max_rel}


def all_endpoint_deltas(Ws, capture):
    """Both equivalent implementations, retained to prove the equivalence."""
    depth = len(Ws)
    n = Ws[0].shape[0]
    forward = {r: [] for r in RUNGS}
    reverse = {r: [] for r in RUNGS}
    parity = 0.0
    for endpoint in range(depth):
        for rung in RUNGS:
            f = np.zeros(n)
            b = np.zeros(n)
            for li in range(endpoint + 1):
                v = capture["sources"][li][rung].copy()
                for k in range(li + 1, endpoint + 1):
                    v = (v @ Ws[k]) * capture["phis"][k]
                f += v

                A = np.eye(n)
                for k in range(li + 1, endpoint + 1):
                    J = Ws[k] * capture["phis"][k][None, :]
                    A = A @ J
                b += capture["sources"][li][rung] @ A
            forward[rung].append(f)
            reverse[rung].append(b)
            parity = max(parity, relmax(f, b))
    return forward, reverse, parity


def antithetic_layer_means(Ws, n_base: int, seed: int, batch: int = 200_000):
    """Target-free synthetic reference and an SE for each endpoint mean."""
    n = Ws[0].shape[0]
    depth = len(Ws)
    sums = [np.zeros(n) for _ in range(depth)]
    sums2 = [np.zeros(n) for _ in range(depth)]
    rng = np.random.default_rng(seed)
    W32 = [np.asarray(W, dtype=np.float32) for W in Ws]
    done = 0
    while done < n_base:
        b = min(batch, n_base - done)
        z = rng.standard_normal((b, n)).astype(np.float32)
        xp, xm = z, -z
        for li, W in enumerate(W32):
            xp = np.maximum(xp @ W, np.float32(0.0))
            xm = np.maximum(xm @ W, np.float32(0.0))
            pair = (xp + xm) * np.float32(0.5)
            sums[li] += pair.sum(axis=0, dtype=np.float64)
            p64 = np.asarray(pair, dtype=np.float64)
            sums2[li] += np.sum(p64 * p64, axis=0, dtype=np.float64)
        done += b
    means, ses = [], []
    for s, s2 in zip(sums, sums2):
        mean = s / done
        var_pair = np.maximum((s2 - done * mean * mean) / max(done - 1, 1), 0.0)
        means.append(mean)
        ses.append(np.sqrt(var_pair / done))
    return means, ses


def endpoint_metrics(baseline, truth, truth_se, deltas):
    base_err = np.asarray(baseline) - np.asarray(truth)
    base_mse = float(np.mean(base_err ** 2))
    truth_scale = max(rms(truth), 1e-300)
    out = {"baseline_mse": base_mse, "baseline_rms": math.sqrt(base_mse),
           "baseline_relative_rms": math.sqrt(base_mse) / truth_scale,
           "truth_rms": truth_scale, "reference_se_rms": rms(truth_se),
           "rungs": {}}
    for rung in RUNGS:
        delta = np.asarray(deltas[rung])
        corrected_err = base_err + delta
        corrected_mse = float(np.mean(corrected_err ** 2))
        dn, en = float(np.linalg.norm(delta)), float(np.linalg.norm(base_err))
        target = -base_err
        tn = float(np.linalg.norm(target))
        cosine = None if dn == 0.0 or tn == 0.0 else float(np.dot(delta, target) / (dn * tn))
        corrected_rms = math.sqrt(corrected_mse)
        out["rungs"][rung] = {
            "corrected_mse": corrected_mse,
            "corrected_rms": corrected_rms,
            "corrected_relative_rms": corrected_rms / truth_scale,
            "improvement_factor": (base_mse / corrected_mse
                                   if corrected_mse > 0.0 else math.inf),
            "correction_rms": rms(delta),
            "correction_relative_rms": rms(delta) / truth_scale,
            "correction_over_baseline_error_rms": (dn / en if en > 0.0 else math.inf),
            "alignment_cosine_to_truth_minus_baseline": cosine,
            "reference_se_over_corrected_rms": (rms(truth_se) / corrected_rms
                                                 if corrected_rms > 0.0 else math.inf),
            "finite": bool(np.all(np.isfinite(delta)) and np.isfinite(corrected_mse))
        }
    return out


def main():
    prereg = json.loads(PREREG.read_text())
    seal = json.loads(SEAL.read_text())
    bindings = {
        "preregistration_sha256": sha256(PREREG),
        "runner_sha256": sha256(Path(__file__)),
        "ledger_rig_sha256": sha256(LEDGER / "ledger_rig.py"),
        "ledger_verify_sha256": sha256(LEDGER / "ledger_verify.py")
    }
    if bindings != seal["bindings"]:
        raise RuntimeError({"seal": seal["bindings"], "actual": bindings})
    if RESULT.exists():
        raise RuntimeError("RESULT.json already exists; frozen receipt is single-use")

    n_base = int(prereg["frozen_networks"]["antithetic_base_samples_per_network"])
    family_seeds = prereg["frozen_networks"]["family_weight_seeds"]
    ref8 = prereg["frozen_networks"]["n8_d4"]["reference_seeds"]
    ref16 = prereg["frozen_networks"]["n16_d5"]["reference_seeds"]
    cases, captures = [], {}
    algebra = {"baseline_reproduction_max_rel": 0.0,
               "source_identity_max_rel": 0.0,
               "adjoint_parity_max_rel": 0.0,
               "K4_bitwise_identity": True}
    t0 = time.time()
    for fi, family_seed in enumerate(family_seeds):
        for width, depth, reference_seed in ((8, 4, ref8[fi]), (16, 5, ref16[fi])):
            label = f"family{fi + 1}_n{width}_d{depth}"
            print(f"[{label}] ledger + reverse/forward algebra", flush=True)
            Ws = frozen_weights(family_seed, width, depth)
            cap = capture_level_c(Ws)
            fwd, rev, parity = all_endpoint_deltas(Ws, cap)
            algebra["baseline_reproduction_max_rel"] = max(
                algebra["baseline_reproduction_max_rel"], cap["baseline_reproduction_max_rel"])
            algebra["source_identity_max_rel"] = max(
                algebra["source_identity_max_rel"], cap["source_identity_max_rel"])
            algebra["adjoint_parity_max_rel"] = max(
                algebra["adjoint_parity_max_rel"], parity)
            for endpoint in range(depth):
                algebra["K4_bitwise_identity"] = bool(
                    algebra["K4_bitwise_identity"]
                    and np.array_equal(cap["posts"][endpoint]["m"],
                                       cap["posts"][endpoint]["m"] + fwd["K4"][endpoint]))

            print(f"[{label}] {2*n_base:,} antithetic forward samples", flush=True)
            truth, truth_se = antithetic_layer_means(Ws, n_base, reference_seed)
            endpoints = []
            for endpoint in range(depth):
                endpoints.append(endpoint_metrics(
                    cap["posts"][endpoint]["m"], truth[endpoint], truth_se[endpoint],
                    {r: fwd[r][endpoint] for r in RUNGS}))
            case = {"label": label, "family_index": fi + 1,
                    "family_weight_seed": family_seed, "reference_seed": reference_seed,
                    "width": width, "depth": depth,
                    "antithetic_base_samples": n_base,
                    "total_forward_samples": 2 * n_base,
                    "final": endpoints[-1],
                    "matched_depth4": endpoints[3],
                    "source_stats": cap["source_stats"]}
            cases.append(case)
            captures[label] = {"case": case, "endpoints": endpoints}

    final_cases = [c["final"] for c in cases]
    aggregates = {}
    for rung in RUNGS:
        factors = [c["rungs"][rung]["improvement_factor"] for c in final_cases]
        aggregates[rung] = {"geometric_mean_improvement": geometric_mean(factors),
                            "minimum_improvement": float(min(factors)),
                            "maximum_improvement": float(max(factors))}

    scaling_pairs = []
    for fi in range(2):
        c8 = captures[f"family{fi + 1}_n8_d4"]["case"]["matched_depth4"]
        c16 = captures[f"family{fi + 1}_n16_d5"]["case"]["matched_depth4"]
        a8, a16 = c8["rungs"]["K12"], c16["rungs"]["K12"]
        scaling_pairs.append({
            "family_index": fi + 1,
            "delta_relative_rms_n8_over_n16": (a8["correction_relative_rms"]
                                                / a16["correction_relative_rms"]),
            "corrected_relative_rms_n8_over_n16": (a8["corrected_relative_rms"]
                                                    / a16["corrected_relative_rms"]),
            "corrected_relative_mse_n8_over_n16": (
                a8["corrected_relative_rms"] / a16["corrected_relative_rms"]) ** 2,
            "baseline_relative_rms_n8_over_n16": (c8["baseline_relative_rms"]
                                                   / c16["baseline_relative_rms"])
        })
    delta_ratio_med = float(np.median(
        [p["delta_relative_rms_n8_over_n16"] for p in scaling_pairs]))
    residual_ratio_med = float(np.median(
        [p["corrected_relative_rms_n8_over_n16"] for p in scaling_pairs]))
    scaling = {
        "matched_depth": 4, "pairs": scaling_pairs,
        "median_delta_relative_rms_n8_over_n16": delta_ratio_med,
        "median_corrected_relative_rms_n8_over_n16": residual_ratio_med,
        "one_over_n_source_supported": bool(1.4 <= delta_ratio_med <= 2.8),
        "one_over_n_squared_residual_supported": bool(2.8 <= residual_ratio_med <= 5.6),
        "caution": "Diagnostic only: prior low-degree coherence was observed to scale much more slowly than 1/n."
    }

    k12 = [c["rungs"]["K12"] for c in final_cases]
    algebra_pass = bool(
        algebra["baseline_reproduction_max_rel"] <= 1e-12
        and algebra["source_identity_max_rel"] <= 1e-13
        and algebra["adjoint_parity_max_rel"] <= 1e-12
        and algebra["K4_bitwise_identity"])
    accuracy_pass = bool(all(x["improvement_factor"] >= 3.0 for x in k12)
                         and aggregates["K12"]["geometric_mean_improvement"] >= 3.0)
    alignment_pass = bool(all(x["alignment_cosine_to_truth_minus_baseline"] is not None
                              and x["alignment_cosine_to_truth_minus_baseline"] >= 0.5
                              for x in k12))
    nontriviality_pass = bool(sum(x["correction_over_baseline_error_rms"] >= 0.05
                                  for x in k12) >= 3)
    stability_pass = bool(all(
        x["finite"]
        and x["correction_over_baseline_error_rms"] <= 4.0
        and x["corrected_mse"] <= 10.0 * c["baseline_mse"]
        and x["reference_se_over_corrected_rms"] <= 0.10
        for x, c in zip(k12, final_cases)))
    degree_pass = bool(aggregates["K12"]["geometric_mean_improvement"]
                       >= max(aggregates[r]["geometric_mean_improvement"]
                              for r in RUNGS if r != "K12"))

    budget = 2_200_000_000_000
    rank5 = math.comb(1028, 5)
    cheap_extra = 222_167_040
    conservative_host_upper = 1_800_000_000_000
    cost_rank = {
        "compressed_extra_flops_upper_bound": cheap_extra,
        "conservative_current_host_upper_bound": conservative_host_upper,
        "compressed_host_plus_extra_upper_bound": conservative_host_upper + cheap_extra,
        "board_budget": budget,
        "compressed_cost_pass": bool(conservative_host_upper + cheap_extra <= budget),
        "exact_scalar_coefficients_by_total_degree": {
            str(n): {str(d): math.comb(n + d - 1, d) for d in range(1, 13)}
            for n in (8, 16, 1024)},
        "full_degree5_scalar_coefficient_lower_bound": rank5,
        "full_degree5_rank_cost_pass": bool(rank5 <= budget),
        "note": "The full degree-5 lower bound is for one scalar endpoint before evaluation arithmetic, outputs, or layers."
    }

    gates = {"algebra_pass": algebra_pass, "accuracy_3x_every_seed_pass": accuracy_pass,
             "alignment_pass": alignment_pass, "nontriviality_pass": nontriviality_pass,
             "stability_pass": stability_pass, "degree_ladder_pass": degree_pass,
             "compressed_cost_pass": cost_rank["compressed_cost_pass"],
             "full_degree5_rank_cost_pass": cost_rank["full_degree5_rank_cost_pass"],
             "reverse_direction_is_non_equivalent_new_operator": False,
             "material_new_source_pass": bool(accuracy_pass and alignment_pass
                                               and nontriviality_pass and degree_pass)}
    if not stability_pass and all(x["reference_se_over_corrected_rms"] > 0.10 for x in k12):
        verdict = "INCONCLUSIVE_NOISE"
    elif (algebra_pass and accuracy_pass and alignment_pass and nontriviality_pass
          and stability_pass and degree_pass and cost_rank["compressed_cost_pass"]):
        verdict = "PASS_SMALL_N_ONLY"
    elif not cost_rank["full_degree5_rank_cost_pass"]:
        verdict = "CLOSE_RANK_COST_DICHOTOMY"
    else:
        verdict = "CLOSE_EQUIVALENT_EFFECT_CAPPED"

    payload = {
        "schema": "p2r0f3b1a.reverse_hermite.result.v1",
        "node": "P2R0F3B1A", "verdict": verdict,
        "target_access": {"development_targets_opened": 0, "target_fits": 0,
                          "observer_or_selection_artifacts_opened": 0,
                          "submissions_authorized": 0},
        "bindings": bindings, "elapsed_wall_s": time.time() - t0,
        "operator_classification": {
            "compressed_reverse_equals_forward_duhamel": True,
            "proof_witness_max_relative_disagreement": algebra["adjoint_parity_max_rel"],
            "perturbative_onsager_description": "sum of frozen-state local predicted mean-map defects transported to the final mean; no forward feedback",
            "exact_full_hermite_expansion_claimed": False},
        "algebra": algebra, "cases": cases, "degree_aggregates": aggregates,
        "finite_width_scaling": scaling, "cost_rank": cost_rank, "gates": gates,
        "terminal_reason": (
            "The compressed target-free source passed every frozen gate; only a new receipt may authorize further work."
            if verdict == "PASS_SMALL_N_ONLY" else
            "The cheap arm is algebraically first-order Duhamel and failed at least one material-source gate; the first exact non-equivalent degree-5 scalar functional exceeds the complete board budget in coefficient count alone. Stop without development targets."),
        "preregistered_terminal_stop_obeyed": True
    }
    tmp = RESULT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULT)
    print(json.dumps({"verdict": verdict, "gates": gates,
                      "K12_geomean_improvement": aggregates["K12"],
                      "finite_width_scaling": scaling,
                      "result_sha256": sha256(RESULT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TAG quadverdict-v1: assemble QUADRATURE_VERDICT.json from measured JSONs.

Reads theory_branching.json, realnet_measurements.json, curve_v2.json,
nnls_v3.json, realnet_rules.json, quad_verify.json. Pure arithmetic, no
forwards. The budget frame:
  B = 2^41 flops; forward = 2*16*1024^2 = 2^25 (mult+add counted separately);
  N_plain = 2^16 = 65536 forwards.
  CV hybrid: precompose 15*2*1024^3 + calibration 2048 forwards + per-pair
  (forward + affine eval 2*1024^2+2048) -> N_pairs.
Feasibility: MSE_hybrid = vbar * VR / N_pairs; required capture for target T
is 1 - T*N_pairs/vbar. All capture ceilings are MEASURED (rank 35) with
cross-row spread from ranks 36/120/234.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

B = 2**41
FWD = 2 * 16 * 1024**2
AFF = 2 * 1024**2 + 2 * 1024
PRE = 15 * 2 * 1024**3 + 16 * 2 * 1024**2 + 16 * 2 * 1024**2
CAL = 2048


def main():
    th = json.loads((HERE / "theory_branching.json").read_text())
    rm = json.loads((HERE / "realnet_measurements.json").read_text())
    cv2 = json.loads((HERE / "curve_v2.json").read_text())
    nn = json.loads((HERE / "nnls_v3.json").read_text())
    ru = json.loads((HERE / "realnet_rules.json").read_text())
    qv = json.loads((HERE / "quad_verify.json").read_text())

    vbar = ru["vbar"]
    n_plain = B // FWD
    n_pairs = int((B - PRE - CAL * FWD) / (FWD + AFF))
    sj = rm["p2"]
    trC = sj["trace_C"]
    trV = sj["trVar"]
    lin_stein_raw = sj["best_linear_capture_stein"]
    SJ = 160
    lin_stein_debiased = (lin_stein_raw * trV - (trC - lin_stein_raw * trV) / SJ) / trV / (1 - 1 / SJ)
    cap_affine = rm["p4"]["capture_affine"]
    vr_affine = rm["p4"]["vr_affine_paired"]

    def hybrid_mse(vr):
        return vbar * vr / n_pairs

    def req_capture(T, n=None):
        return 1.0 - T * (n or n_pairs) / vbar

    mass = {k: nn[k] for k in ("mass_le_1", "mass_le_2", "mass_le_4", "mass_le_8",
                               "mass_le_16", "mass_le_32", "mass_le_64")}
    cond = {k: v["capture"] for k, v in rm["p3"].items()}

    ladder = {
        "plain_MC_at_65536": vbar / n_plain,
        "antithetic_at_65536_pred": vbar * (1.0 + cv2["chat_final"]["-0.95"]) / n_plain,
        "cv_affine_hybrid_at_budget": hybrid_mse(vr_affine),
        "oracle_deg2_exact_surrogate": hybrid_mse(1.0 - mass["mass_le_2"]),
        "oracle_deg3_stroud_scrambled": vbar * (1.0 - mass["mass_le_4"] + nn["masses"].get("4", 0.0)) / n_plain,
        "oracle_deg16_exact_rule": vbar * (1.0 - mass["mass_le_16"]) / n_plain,
        "oracle_deg64_exact_rule": vbar * (1.0 - mass["mass_le_64"]) / n_plain,
        "oracle_perfect_512dim_subspace": vbar * (1.0 - cond["512"]) / n_plain,
        "gh2_top2_hybrid_at_4096_measured": ru["rules"]["GH2"]["4096"]["mse_mean"],
        "lattice_4093x8_measured": ru["rules"]["LAT"]["4093"]["mse_mean"],
        "mc_16384_measured": ru["rules"]["MC"]["16384"]["mse_mean"],
        "cv_16384_measured": ru["rules"]["CV"]["16384"]["mse_mean"],
    }

    xrows = ru["cross_rows"]
    verdict = {
        "tag": "quadverdict-v1",
        "question": ("can a deterministic quadrature/cubature (or exact-mean control-variate "
                     "hybrid) compute m_16 = E[x_16] of the width-1024 depth-16 relu MLP to "
                     "1e-9-class MSE inside 2^41 flops?"),
        "feasible": False,
        "route": ("NONE reaches 1e-9. Best sampling-class rule at budget = Gaussian-equivalent "
                  "affine control variate (Stein slope a_l = Phi(mu_l/s_l)), exact mean c_eff, "
                  "MSE ~ 0.89x plain MC. All deterministic rules measured/bounded WORSE than "
                  "plain MC at matched counts. The analytic closure (3.3e-8 fleet) dominates "
                  "everything the quadrature class can offer; its bias cannot be repaired by "
                  "sampling inside 2^41 (would need capture >= 99.9% vs measured <= 43% "
                  "for any exactly-integrable surrogate family)."),
        "predicted_mse_at_budget": {
            "row35_scale (vbar=%.4e)" % vbar: ladder,
            "required_vr_for_1e-9": 1e-9 * n_pairs / vbar,
            "required_capture_for_1e-9": req_capture(1e-9),
            "required_capture_to_match_closure_3.3e-8_row35": req_capture(3.3e-8),
            "note_fleet_scale": ("rank-35 vbar is ~6.8x the fleet-average implied by the known "
                                 "plain-MC 1.5e-7@65k figure; fractions (captures, masses, VRs) "
                                 "are the transferable objects; cross-row spread below"),
            "cross_row_vbar": {r: x["vbar"] for r, x in xrows.items()},
            "cross_row_capture_affine": {r: x["capture_affine"] for r, x in xrows.items()},
            "cross_row_linear_mass": {r: x["linear_mass_richardson"] for r, x in xrows.items()},
        },
        "measured_smalln_exponent": {
            "n64_d6": qv["e2_exponents"],
            "real_1024_d16_MC": ru["rules"]["MC"]["fitted_exponent"],
            "real_1024_d16_CV": ru["rules"]["CV"]["fitted_exponent"],
            "fail_fast_best_rmse_gain": qv["fail_fast"]["best_rmse_gain_at_matched_N"],
            "fail_fast_beats_3x": qv["fail_fast"]["beats_3x"],
        },
        "active_spectrum_summary": {
            "gradient_spectrum_k_at": {k: sj[k] for k in
                                       ("k_at_50pct", "k_at_90pct", "k_at_99pct", "k_at_99.9pct")},
            "conditional_capture_topk": cond,
            "hermite_mass_measured_rank35": mass,
            "hermite_mass_above_64": 1.0 - mass["mass_le_64"],
            "linear_mass_three_ways": {
                "richardson_curve": cv2["per_layer_linear_mass"]["richardson"][15],
                "stein_debiased": lin_stein_debiased,
                "affine_capture_paired": cap_affine,
                "optimal_beta_capture": rm["p4"]["capture_optimal_beta"],
            },
            "annealed_vs_real": {
                "annealed_M1_16": th["degree_table"]["16"]["P_deg_eq_1"],
                "real_M1_16": nn["mass_le_1"],
                "annealed_median_degree_16": th["degree_table"]["16"]["quantiles"]["0.5"],
                "per_layer_linear_mass_real": cv2["per_layer_linear_mass"]["richardson"],
                "layer1_exact_agreement": cv2["per_layer_linear_mass"]["richardson"][0],
            },
        },
        "budget_frame": {
            "B": B, "forward_flops": FWD, "n_plain": n_plain,
            "affine_eval_flops": AFF, "precompose_flops": PRE,
            "calibration_forwards": CAL, "n_pairs": n_pairs,
        },
        "verification": {
            "ensemble_annealed_law_n64": {
                "measured_mean": qv["e1"]["ensemble_mean_linear_mass"],
                "predicted": qv["e1"]["annealed_prediction"],
                "fixed_net_sd": qv["e1"]["ensemble_sd"],
            },
            "n64_truth_floor": qv["e2_truth_floor_mse"],
            "n64_predictions": qv["e2_predictions"],
            "n64_ratios_at_4096": qv["e2_ratio_vs_mc_at_4096"],
            "orientation_check": rm["orientation"],
            "mc_scaling_check_2048": {"measured": rm["p0"]["mc2048_vs_truth_mse"],
                                      "predicted": rm["p0"]["mc2048_predicted_mse"]},
            "curve_null_rho0": {"value": nn.get("null_rho0", cv2["nnls_v2"]["null_rho0"]),
                                "se": cv2["nnls_v2"]["null_rho0_se"]},
        },
    }
    (HERE / "QUADRATURE_VERDICT.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: verdict[k] for k in ("feasible",)}, indent=0))
    print("req capture 1e-9: %.5f | affine capture %.4f | deg2 oracle %.4f | 512dim oracle %.4f"
          % (verdict["predicted_mse_at_budget"]["required_capture_for_1e-9"],
             cap_affine, mass["mass_le_2"], cond["512"]))
    print("ladder:", json.dumps({k: "%.3e" % v for k, v in ladder.items()}, indent=1))
    print("DONE quadverdict-v1")


if __name__ == "__main__":
    main()

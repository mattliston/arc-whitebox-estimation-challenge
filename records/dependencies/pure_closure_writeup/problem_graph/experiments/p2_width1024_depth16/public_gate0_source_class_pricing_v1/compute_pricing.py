#!/usr/bin/env python3
"""TAG gate0-source-class-pricing-v1: exact arithmetic pricing every known
target-blind source super-class against the ~83.805% late-q capture bar.
All constants are measured receipts, cited by package; no new experiment."""
import json

# --- measured inputs (provenance in OUTPUT) ---
Q_MSE_L10_15 = [4.943317939687495e-09, 5.2047399740178475e-09,
                5.5494274979631525e-09, 5.788652346161188e-09,
                6.0741317687577325e-09, 6.068341723033198e-09]  # p2r0g6 oracle
CAPTURE_REQUIRED = 0.83805
VBAR_PER_ROW = 0.075          # quadrature receipt vbar 0.064-0.086, midpoint
BUDGET = 2 ** 41
PATH_FLOPS = 15 * 2 * 1024 * 1024   # one path, 15 transitions, fp32 GEMM
K6_MEDIAN_N_OVER_S = 35.95    # screen result, worst-role median 29-46
K6_GATE = 1.0 / 9.0
CV_ORACLE_FLOOR = 2.85e-8     # quadrature receipt: free degree-64 rule floor
DET_TAPE_BEST_ACTUAL = 0.005  # <=0.5% endpoint, enriched-J decomposition audit
DET_TAPE_ORACLE = 0.105       # N truth-projection oracle ceiling

q_rms = (sum(Q_MSE_L10_15) / len(Q_MSE_L10_15)) ** 0.5
resid_frac = (1.0 - CAPTURE_REQUIRED) ** 0.5
se_required = resid_frac * q_rms          # all residual budget as estimator noise
sigma_unit = VBAR_PER_ROW ** 0.5
p_max = BUDGET / PATH_FLOPS               # entire budget as paths, zero host
se_order1_at_full_budget = sigma_unit / (p_max ** 0.5)
se_deficit = se_order1_at_full_budget / se_required
paths_deficit = se_deficit ** 2
vr_measured_best = 1.62                   # quadrature receipt rules race
vr_oracle_deg2 = 1.0 / (1.0 - 0.432)
vr_oracle_deg64 = 1.0 / 0.028
k6_paths_deficit = K6_MEDIAN_N_OVER_S / K6_GATE

out = {
 "schema": "gate0-source-class-pricing-v1",
 "bar": {"endpoint_capture_required": CAPTURE_REQUIRED,
         "q_innovation_rms_per_unit_l10_15": q_rms,
         "max_allowed_source_noise_se": se_required,
         "note": "most generous read: ALL residual error budget spent as estimator noise; learning assumed perfect"},
 "class_1_sampled_order1": {
   "per_unit_activation_sd_measured": sigma_unit,
   "paths_at_full_budget_zero_host": p_max,
   "se_at_full_budget": se_order1_at_full_budget,
   "se_deficit_x": se_deficit,
   "flops_deficit_x_at_1_over_P": paths_deficit,
   "with_measured_vr_1p62x": paths_deficit / vr_measured_best,
   "with_oracle_deg2_surrogate": paths_deficit / (vr_measured_best * vr_oracle_deg2),
   "with_free_oracle_deg64_surrogate": paths_deficit / vr_oracle_deg64,
   "verdict": "CLOSED: order-1 (minimum-variance sampled observable) needs >=33x full budget even under the unbuildable free degree-64 oracle surrogate, with the entire budget given to the source and none to the host"},
 "class_2_sampled_higher_order": {
   "k6_measured_paths_deficit_x": k6_paths_deficit,
   "verdict": "CLOSED: per-path variance grows with observable order; measured anchor K6 at 324x gate deficit (0/315 cells); order 2-5 sits between order-1 (>=1193x raw) and K6"},
 "class_3_sampled_cv_hybrids": {
   "measured_oracle_floor_mse": CV_ORACLE_FLOOR,
   "verdict": "CLOSED: floor 2.85e-8 >> 1e-8 with oracle surrogates (quadrature receipt, matched predictions)"},
 "class_4_deterministic_tapes": {
   "best_actual_endpoint_fraction": DET_TAPE_BEST_ACTUAL,
   "best_truth_oracle_fraction": DET_TAPE_ORACLE,
   "verdict": "CLOSED: two independent estimator lineages measure <=0.5% actual, 10.5% oracle ceiling vs 83.8% required"},
 "provenance": {
   "q_innovation": "p2r0g6_native_j_physical_local_oracle_v1/RESULT.json (per_layer_local_q_minus_qJ_mse)",
   "vbar_sigma": "public_quadrature_derivation_v1/QUADRATURE_VERDICT.json (measured vbar 6.4-8.6e-2, VR race, oracle surrogate captures, CV floor)",
   "k6_screen": "public_support3_screen_science_v1/SCREEN_RESULT.json",
   "det_tapes": "public_q24_signed_pair_v4r2_enriched_j_pair_factorial_decomposition_audit_v1 (054d09bb) + public_full688_rank16_cross_unit_covariance_cv_v1 N_VS_J decomposition (10.535% oracle)"},
 "external_constraint": (
   "Raw <1e-8 through the enriched-J lane requires a target-blind source "
   "carrying >=83.805% of the late-q innovation variance at the endpoint. "
   "Every known source super-class is measured out: deterministic tapes "
   "(<=0.5% actual, 10.5% oracle), sampled path statistics at any order "
   "(>=33x full budget at oracle variance-reduction ceilings; order-1 is "
   "the minimum-variance sampled observable), sampled control-variate "
   "hybrids (2.85e-8 floor), and the specific support>=3 K6 replica source "
   "(0/315). Reopening requires either an exact analytic late-layer "
   "propagation advance (the blocked P2R0F3B family's named external "
   "theorems) or a certified sampled/deterministic representation with "
   ">=1000x better per-flop information about the activation law than "
   "independent path sampling. No such object is known.")
}
open("GATE0_PRICING.json", "w").write(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("bar", "class_1_sampled_order1")}, indent=1))

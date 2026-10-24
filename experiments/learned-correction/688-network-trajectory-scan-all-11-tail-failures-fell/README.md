# 688-network trajectory scan: all 11 tail failures fell below 5e-7 after the variance-floor repair

2026-08-24 · experiment · informational · measured

The scan completed role ranks 0..687 with 688/688 finite base outputs. Exactly 11 trajectories meet max-layer or final-layer mean squared error (MSE) >5e-7 at ranks 55, 71, 106, 317, 339, 458, 485, 507, 540, 575, and 602. All 11 frozen near-floor-min repairs are finite NO_HIT and below 5e-7; the effective repaired-population final-layer mean is 9.115992600417338e-8.

Recorded measurement:

    The scan completed role ranks 0..687 with 688/688 finite base outputs. Exactly 11 trajectories meet max-layer or final-layer MSE >5e-7 at ranks 55, 71, 106, 317, 339, 458, 485, 507, 540, 575, and 602. All 11 frozen near-floor-min repairs are finite NO_HIT and below 5e-7; the effective repaired-population final-layer mean is 9.115992600417338e-8. No leaderboard-name or private-board identity join was used.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/data_gate/sealed_role_targets_v2.py](code/data_gate/sealed_role_targets_v2.py)
- [code/fullbudget_k21_k4_closure_a1_no_cp4_v1/estimator.py](code/fullbudget_k21_k4_closure_a1_no_cp4_v1/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py](code/fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/correlation_shrink/estimator.py](code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/correlation_shrink/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/hybrid_exact_var_floor_trigger/estimator.py](code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/hybrid_exact_var_floor_trigger/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/raw_diag_load_rescale/estimator.py](code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/raw_diag_load_rescale/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/spectral_negative_buffer_1_16/estimator.py](code/fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1/spectral_negative_buffer_1_16/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_final_ec34_product_probe_v1/estimator.py](code/fullbudget_k21_k4_closure_q24_final_ec34_product_probe_v1/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/gaussian_m11/estimator.py](code/fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/gaussian_m11/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/m11_no_l3sq/estimator.py](code/fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/m11_no_l3sq/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/m11_no_l4/estimator.py](code/fullbudget_k21_k4_closure_q24_m11_channel_ablations_v1/m11_no_l4/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py](code/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py](code/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py)
- [code/fullbudget_k21_k4_closure_q24_v1/estimator.py](code/fullbudget_k21_k4_closure_q24_v1/estimator.py)
- [code/public_observer_fit_candidate_truth_score_v1/run_candidate_truth_row_v1.py](code/public_observer_fit_candidate_truth_score_v1/run_candidate_truth_row_v1.py)
- [code/public_observer_fit_candidate_truth_score_v1/run_panel_v1.py](code/public_observer_fit_candidate_truth_score_v1/run_panel_v1.py)
- [code/public_observer_fit_truth_scan_v1/run_truth_row_v1.py](code/public_observer_fit_truth_scan_v1/run_truth_row_v1.py)
- [code/public_observer_fit_truth_scan_v1/run_until_hits_v1.py](code/public_observer_fit_truth_scan_v1/run_until_hits_v1.py)

</details>

## Evidence

- [PUBLIC_SCAN_RESULT_688.json#L53](PUBLIC_SCAN_RESULT_688.json#L53)
- [TARGET_ACCESS_AUTHORIZATION.json#L1](TARGET_ACCESS_AUTHORIZATION.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [PUBLIC_SCAN_RESULT_688.json](PUBLIC_SCAN_RESULT_688.json) · result
- [TARGET_ACCESS_AUTHORIZATION.json](TARGET_ACCESS_AUTHORIZATION.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-688-network-trajectory-scan-all-11-tail-failures-fell) · [Copy provenance](../../../PROVENANCE.json)


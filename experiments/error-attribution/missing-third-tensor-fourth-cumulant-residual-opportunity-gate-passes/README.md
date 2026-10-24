# Missing third tensor: fourth-cumulant residual opportunity gate passes at one cell

2026-09-06T01:06Z · experiment · informational · measured

Exact dense width8/12 projection and coordinate-star certificates pass below9e-16. Source-v2 rank35 postL12 parent CP21568 remains unmodified; its next32 third-cumulant repeated-index plane (K21) rows match the previous receipt BYTE EXACTLY. The fixed <=.7, cosine>=.6, both cross-stream<1 gate PASSES.

Recorded measurement:

    Exact dense width8/12 projection and coordinate-star certificates pass below9e-16. Source-v2 rank35 postL12 parent CP21568 remains unmodified; its next32 K21 rows match the previous receipt BYTE EXACTLY. The q4 basis is frozen from the pre-target host covariance (14.8067% covariance trace). Two NEW independent8M streams, with seeds distinct from each other AND the archived host/truth seeds, collect only four full post-third slices and32 next-K21 rows. Centered float32 MPS products, float64 accumulation; 71.28s/75.98s and less than0.95GB host peak each, sequential GPU jobs. Slice diagonal consistency4.11e-7/4.30e-7, double-slice symmetry5.77e-8/ 5.70e-8. Shared float32 arithmetic bias is not independently measured. - Baseline observed K21 response MSE1.55864301e-9; cross-stream signal point estimate1.51909224e-9; sampling-noise fraction2.5375%. - Signal natural-strength error ratio .315424912; correction/error cosine .827756638; correction norm .803140295 times error norm. - Independent cross-stream ratios .343131581 and .336631922. - Unweighted signal error ratio .128959464. Keeping projected repeated entries too changes weighted ratio only to .309826898.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/acquire_residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/acquire_residual_one_high_oracle_v1.py)
- [code/astra_phase2_reexamination_v1/analyze_residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/analyze_residual_one_high_oracle_v1.py)
- [code/astra_phase2_reexamination_v1/analyze_third_error_innovation_v1.py](code/astra_phase2_reexamination_v1/analyze_third_error_innovation_v1.py)
- [code/astra_phase2_reexamination_v1/check_cumulant_tilt_response_v1.py](code/astra_phase2_reexamination_v1/check_cumulant_tilt_response_v1.py)
- [code/astra_phase2_reexamination_v1/check_cumulative_backward_source_v1.py](code/astra_phase2_reexamination_v1/check_cumulative_backward_source_v1.py)
- [code/astra_phase2_reexamination_v1/check_full_edge_gaussian_bond3_v1.py](code/astra_phase2_reexamination_v1/check_full_edge_gaussian_bond3_v1.py)
- [code/astra_phase2_reexamination_v1/check_linear_transport_residual_source_v1.py](code/astra_phase2_reexamination_v1/check_linear_transport_residual_source_v1.py)
- [code/astra_phase2_reexamination_v1/check_one_high_actual_value_v1.py](code/astra_phase2_reexamination_v1/check_one_high_actual_value_v1.py)
- [code/astra_phase2_reexamination_v1/check_positive_path_recursive_v1.py](code/astra_phase2_reexamination_v1/check_positive_path_recursive_v1.py)
- [code/astra_phase2_reexamination_v1/check_positive_path_value_transfer_v1.py](code/astra_phase2_reexamination_v1/check_positive_path_value_transfer_v1.py)
- [code/astra_phase2_reexamination_v1/cross_depth_energy_statistics_v1.py](code/astra_phase2_reexamination_v1/cross_depth_energy_statistics_v1.py)
- [code/astra_phase2_reexamination_v1/cumulant_tilt_response_v1.py](code/astra_phase2_reexamination_v1/cumulant_tilt_response_v1.py)

<details>
<summary>Additional programs and dependencies (20)</summary>

- [code/astra_phase2_reexamination_v1/cumulative_backward_source_v1.py](code/astra_phase2_reexamination_v1/cumulative_backward_source_v1.py)
- [code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py](code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/fit_residual_trace_representation_v1.py](code/astra_phase2_reexamination_v1/fit_residual_trace_representation_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_emission_oracle_math_v1.py](code/astra_phase2_reexamination_v1/fourth_emission_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/freeze_fourth_emission_inputs_v1.py](code/astra_phase2_reexamination_v1/freeze_fourth_emission_inputs_v1.py)
- [code/astra_phase2_reexamination_v1/freeze_residual_one_high_inputs_v1.py](code/astra_phase2_reexamination_v1/freeze_residual_one_high_inputs_v1.py)
- [code/astra_phase2_reexamination_v1/full_edge_gaussian_bond3_v1.py](code/astra_phase2_reexamination_v1/full_edge_gaussian_bond3_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py](code/astra_phase2_reexamination_v1/linear_transport_residual_source_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_birth_reference_v1.py](code/astra_phase2_reexamination_v1/positive_path_birth_reference_v1.py)
- [code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py](code/astra_phase2_reexamination_v1/positive_path_resummation_v1.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py](code/astra_phase2_reexamination_v1/residual_one_high_oracle_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md#L1](RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md#L1)
- [RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md#L8](RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md#L8)
- [RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md#L13](RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md#L13)

<details>
<summary>All result files and records (6)</summary>

- [RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md](RESIDUAL_ONE_HIGH_ORACLE_V1_VERDICT.md) · record
- [residual_one_high_oracle_certificate_v1.json](residual_one_high_oracle_certificate_v1.json) · result
- [residual_one_high_inputs_r35_l12_v1.json](residual_one_high_inputs_r35_l12_v1.json) · result
- [residual_one_high_truth_r35_l12_A_v1.json](residual_one_high_truth_r35_l12_A_v1.json) · result
- [residual_one_high_oracle_r35_l12_v1.json](residual_one_high_oracle_r35_l12_v1.json) · result
- [RESIDUAL_ONE_HIGH_ORACLE_DESIGN_V1.md](RESIDUAL_ONE_HIGH_ORACLE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-missing-third-tensor-fourth-cumulant-residual-opportunity-gate-passes) · [Copy provenance](../../../PROVENANCE.json)


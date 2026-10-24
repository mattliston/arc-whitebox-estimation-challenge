# Grouped rank-40 matrix kernels: corrected all-24 adjusted -0.706639% versus the row-energy parent

2026-10-15T23:37:39Z · experiment · adopted · measured

A fixed 105-shape grouped strided-read kernel dispatch passed paired science, then exact request helpers and a corrected first-view bill transferred the results to the released source. The initial larger forecast was superseded after all ten native bills were reconciled.

Recorded measurement:

    initial24 vs336294 -1.043867%,OOS8-.990817%;corrected24 vs336294-.869110%,vs336334-.706639%,OOS8 incremental-.669127%;omittedfirstview276903104FLOPs

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_smirnov_rpe_science_v1/grade_sm40_retired_phase_v1.py](code/lead/runs/astra_smirnov_rpe_science_v1/grade_sm40_retired_phase_v1.py)
- [code/lead/runs/astra_smirnov_rpe_science_v1/sm40_rpe_source_contract_v1.py](code/lead/runs/astra_smirnov_rpe_science_v1/sm40_rpe_source_contract_v1.py)
- [code/lead/runs/astra_smirnov_rpe_science_v1/finite_sm40_astra_v1.py](code/lead/runs/astra_smirnov_rpe_science_v1/finite_sm40_astra_v1.py)
- [code/lead/runs/astra_smirnov_rpe_science_v1/native_sm40_judge_astra_v1.py](code/lead/runs/astra_smirnov_rpe_science_v1/native_sm40_judge_astra_v1.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/build_alias_hold_v5.py](code/lead/runs/astra_smirnov_current24_virtual_v1/build_alias_hold_v5.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/build_candidate_v1.py](code/lead/runs/astra_smirnov_current24_virtual_v1/build_candidate_v1.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/build_candidate_v2.py](code/lead/runs/astra_smirnov_current24_virtual_v1/build_candidate_v2.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/build_combined_arena_v14.py](code/lead/runs/astra_smirnov_current24_virtual_v1/build_combined_arena_v14.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/build_combined_arena_v15.py](code/lead/runs/astra_smirnov_current24_virtual_v1/build_combined_arena_v15.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/build_combined_arena_v4.py](code/lead/runs/astra_smirnov_current24_virtual_v1/build_combined_arena_v4.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/grade_native_ten_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/grade_native_ten_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/grade_native_ten_v2.py](code/lead/runs/astra_smirnov_final_native_ten_v1/grade_native_ten_v2.py)

<details>
<summary>Additional programs and dependencies (46)</summary>

- [code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_TEMPLATE_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_TEMPLATE_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/finite_final_astra_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/finite_final_astra_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/native_final_judge_astra_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/native_final_judge_astra_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/prepare_parity_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/prepare_parity_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_cov_output_views_v1/candidate_cov_output_views_HELD_v1.py](code/lead/runs/astra_cov_output_views_v1/candidate_cov_output_views_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py](code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_prefix22_uvp_split_native_ten_v1/finite_uvp_astra_v1.py](code/lead/runs/astra_prefix22_uvp_split_native_ten_v1/finite_uvp_astra_v1.py)
- [code/lead/runs/astra_prefix22_uvp_split_native_ten_v1/native_uvp_judge_astra_v1.py](code/lead/runs/astra_prefix22_uvp_split_native_ten_v1/native_uvp_judge_astra_v1.py)
- [code/lead/runs/astra_prefix22_uvp_split_native_ten_v1/rpe_source_contract_v1.py](code/lead/runs/astra_prefix22_uvp_split_native_ten_v1/rpe_source_contract_v1.py)
- [code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py](code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_HELD_v4.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_HELD_v4.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v14.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v14.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_owned8_HELD_v1.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_owned8_HELD_v1.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_owned8_HELD_v2.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_owned8_HELD_v2.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/fragment_smirnov_arena_HELD_v14.py](code/lead/runs/astra_smirnov_current24_virtual_v1/fragment_smirnov_arena_HELD_v14.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/fragment_smirnov_arena_HELD_v15.py](code/lead/runs/astra_smirnov_current24_virtual_v1/fragment_smirnov_arena_HELD_v15.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/fragment_smirnov_arena_HELD_v4.py](code/lead/runs/astra_smirnov_current24_virtual_v1/fragment_smirnov_arena_HELD_v4.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_arena_fused_v4.py](code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_arena_fused_v4.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_arena_fused_v5.py](code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_arena_fused_v5.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_batched_v1.py](code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_batched_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_uvp_split_views_v1/candidate_prefix22_uvp_split_HELD_v1.py](code/lead/runs/astra_uvp_split_views_v1/candidate_prefix22_uvp_split_HELD_v1.py)
- [code/lead/runs/astra_uvp_view_hold_v1/candidate_prefix22_uvp_views_HELD_v1.py](code/lead/runs/astra_uvp_view_hold_v1/candidate_prefix22_uvp_views_HELD_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3573](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3573)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3563](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3563)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3565](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3565)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3569](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3569)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3575](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3575)

<details>
<summary>All result files and records (20)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ALL24_DECISION_astra_v1.json](ALL24_DECISION_astra_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PROPOSED_ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json](PROPOSED_ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json) · result
- [PROPOSED_ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v2.json](PROPOSED_ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v2.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [RELEASE_HANDOFF_astra_v1.json](RELEASE_HANDOFF_astra_v1.json) · result
- [REMAINING_RETIREMENT_astra_v1.json](REMAINING_RETIREMENT_astra_v1.json) · result
- [SANITY_RESULT_v1.json](SANITY_RESULT_v1.json) · result
- [ALIAS_HOLD_SOURCE_v5.json](ALIAS_HOLD_SOURCE_v5.json) · result
- [BUILD_COMBINED_ARENA_v14.json](BUILD_COMBINED_ARENA_v14.json) · result
- [BUILD_COMBINED_ARENA_v15.json](BUILD_COMBINED_ARENA_v15.json) · result
- [BUILD_COMBINED_ARENA_v4.json](BUILD_COMBINED_ARENA_v4.json) · result
- [BUILD_COMBINED_ARENA_v7.json](BUILD_COMBINED_ARENA_v7.json) · result
- [FINAL_SOURCE_CONTRACT_REVIEW_v1.json](FINAL_SOURCE_CONTRACT_REVIEW_v1.json) · result
- [FIXED_LAYER1_R7_ROUTE_v1.json](FIXED_LAYER1_R7_ROUTE_v1.json) · result
- [NATIVE_TEN_IMPLEMENTATION_PASS_COST_MISMATCH_v1.json](NATIVE_TEN_IMPLEMENTATION_PASS_COST_MISMATCH_v1.json) · result
- [NATIVE_TEN_PARITY_v1.json](NATIVE_TEN_PARITY_v1.json) · result
- [PARITY_RETIREMENT_astra_v1.json](PARITY_RETIREMENT_astra_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336342: grouped rank-40 read kernels with corrected whole-call billing](../../../submissions/336342/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-grouped-rank-40-matrix-kernels-corrected-all-24-adjusted) · [Copy provenance](../../../PROVENANCE.json)


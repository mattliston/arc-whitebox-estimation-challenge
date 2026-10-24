# Hermite covariance grouping into two products per order: all-24 adjusted -0.570899%

2026-10-15T18:17:36Z · experiment · adopted · measured

Gaussian, local third/fourth-order and modeled third-order cross contributions were regrouped into two outer products at each of the first two correlation orders, retaining the other pair terms. The request-lean final source passed the full 24 and out-of-sample sign gates and was submitted after view-only repair.

Recorded measurement:

    tiny maxfloat64error1.67e-16,FP32relative~1e-7; initial forecast1005469696FLOPs/.410396%; pilot vs QR MSE-.195069%,bill-.411051%,adjusted-.606121%; final24 vs336178 adjusted-.570899%,OOS8-.470520%,MSE-.186120%,bill-.384778%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_hermite_group_fusion_v1/build_source_v1.py](code/lead/runs/astra_hermite_group_fusion_v1/build_source_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_v1/build_source_v2.py](code/lead/runs/astra_hermite_group_fusion_v1/build_source_v2.py)
- [code/lead/runs/astra_hermite_group_fusion_v1/candidate_hermite_group_QR_HELD_v1.py](code/lead/runs/astra_hermite_group_fusion_v1/candidate_hermite_group_QR_HELD_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_v1/fragment_hermite_group_HELD_v1.py](code/lead/runs/astra_hermite_group_fusion_v1/fragment_hermite_group_HELD_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_v1/check_tiny_tariff_v1.py](code/lead/runs/astra_hermite_group_fusion_v1/check_tiny_tariff_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_science_v1/grade_hf_retired_phase_v1.py](code/lead/runs/astra_hermite_group_fusion_science_v1/grade_hf_retired_phase_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_science_v1/finite_hf_astra_v1.py](code/lead/runs/astra_hermite_group_fusion_science_v1/finite_hf_astra_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_science_v1/native_hf_judge_astra_v1.py](code/lead/runs/astra_hermite_group_fusion_science_v1/native_hf_judge_astra_v1.py)
- [code/lead/runs/astra_hermite_group_request_lean_v1/candidate_hermite_group_lean_QR_HELD_v2.py](code/lead/runs/astra_hermite_group_request_lean_v1/candidate_hermite_group_lean_QR_HELD_v2.py)
- [code/lead/runs/astra_hermite_group_request_lean_v1/fragment_hermite_group_lean_HELD_v1.py](code/lead/runs/astra_hermite_group_request_lean_v1/fragment_hermite_group_lean_HELD_v1.py)
- [code/lead/runs/astra_hermite_group_request_lean_v1/fragment_hermite_group_lean_HELD_v2.py](code/lead/runs/astra_hermite_group_request_lean_v1/fragment_hermite_group_lean_HELD_v2.py)
- [code/lead/runs/astra_hermite_group_request_lean_v1/check_alias_v2.py](code/lead/runs/astra_hermite_group_request_lean_v1/check_alias_v2.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

- [code/lead/runs/astra_hermite_group_request_lean_v1/check_alias_v3.py](code/lead/runs/astra_hermite_group_request_lean_v1/check_alias_v3.py)
- [code/lead/runs/astra_hermite_group_request_lean_v1/check_tiny_tariff_lean_v1.py](code/lead/runs/astra_hermite_group_request_lean_v1/check_tiny_tariff_lean_v1.py)
- [code/lead/runs/astra_hermite_group_request_lean_v1/check_tiny_tariff_lean_v2.py](code/lead/runs/astra_hermite_group_request_lean_v1/check_tiny_tariff_lean_v2.py)
- [code/lead/runs/astra_hermite_group_fusion_final_science_v1/grade_hf_retired_phase_v1.py](code/lead/runs/astra_hermite_group_fusion_final_science_v1/grade_hf_retired_phase_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_final_science_v1/check_cached_hf89_parity_v1.py](code/lead/runs/astra_hermite_group_fusion_final_science_v1/check_cached_hf89_parity_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_final_science_v1/finite_hf_astra_v1.py](code/lead/runs/astra_hermite_group_fusion_final_science_v1/finite_hf_astra_v1.py)
- [code/lead/runs/astra_hermite_group_fusion_final_science_v1/native_hf_judge_astra_v1.py](code/lead/runs/astra_hermite_group_fusion_final_science_v1/native_hf_judge_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3464](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3464)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3457](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3457)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3461](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3461)

<details>
<summary>All result files and records (25)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SOURCE_BUILD_v1.json](SOURCE_BUILD_v1.json) · result
- [TINY_AND_TARIFF_v1.json](TINY_AND_TARIFF_v1.json) · result
- [TINY_GUARD_v2.json](TINY_GUARD_v2.json) · result
- [TINY_RETIREMENT_v1.json](TINY_RETIREMENT_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [ACTIVE_CALLCHAIN_PEER_v1.json](ACTIVE_CALLCHAIN_PEER_v1.json) · result
- [IMPORT_PREFLIGHT_v1.json](IMPORT_PREFLIGHT_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json](ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json) · result
- [ROOT_PROSPECTIVE_REMAINING_DEFER_v1.json](ROOT_PROSPECTIVE_REMAINING_DEFER_v1.json) · result
- [SOURCE_BUILD_v2.json](SOURCE_BUILD_v2.json) · result
- [SOURCE_READY_v2.json](SOURCE_READY_v2.json) · result
- [SOURCE_SCOPE_CLARIFICATION_v1.json](SOURCE_SCOPE_CLARIFICATION_v1.json) · result
- [REAL_ARENA_ALIAS_CHECK_v3.json](REAL_ARENA_ALIAS_CHECK_v3.json) · result
- [TINY_GUARD_v1.json](TINY_GUARD_v1.json) · result
- [ALL24_DECISION_astra_v1.json](ALL24_DECISION_astra_v1.json) · result
- [PILOT_DECISION_astra_v1-4417f953.json](PILOT_DECISION_astra_v1-4417f953.json) · result
- [HF89_CACHED_PARITY_v1.json](HF89_CACHED_PARITY_v1.json) · result
- [IMPORT_PREFLIGHT_v1-ecfab82f.json](IMPORT_PREFLIGHT_v1-ecfab82f.json) · result
- [PILOT_RETIREMENT_astra_v1-b863f1db.json](PILOT_RETIREMENT_astra_v1-b863f1db.json) · result
- [PROTOCOL_REGISTERED_astra_v1-497c218a.json](PROTOCOL_REGISTERED_astra_v1-497c218a.json) · result
- [REMAINING_RETIREMENT_astra_v1.json](REMAINING_RETIREMENT_astra_v1.json) · result
- [ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1-eb292f48.json](ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1-eb292f48.json) · result

</details>

## Related work

- shipped in: [Submission #336294: Hermite covariance grouping and QR read with exact view caching](../../../submissions/336294/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-hermite-covariance-grouping-into-two-products-per-order-all) · [Copy provenance](../../../PROVENANCE.json)


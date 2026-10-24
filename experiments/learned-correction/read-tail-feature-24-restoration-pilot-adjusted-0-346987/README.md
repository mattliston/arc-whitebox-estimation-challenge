# Read-tail feature 24 restoration: pilot adjusted +0.346987%

2026-10-15T21:39:50Z · experiment · rejected · measured

Restoring the omitted-tail contribution to feature 24 increased both raw error and metered arithmetic and was closed after eight networks.

Recorded measurement:

    MSE+.162643%,bill+.184344%,net+.346987%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_saved_error_mechanism_v1/build_feature24_restore_v1.py](code/lead/runs/astra_saved_error_mechanism_v1/build_feature24_restore_v1.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/build_shadow_v1.py](code/lead/runs/astra_saved_error_mechanism_v1/build_shadow_v1.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/candidate_feature24_tail_restore_all_v1.py](code/lead/runs/astra_saved_error_mechanism_v1/candidate_feature24_tail_restore_all_v1.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/candidate_shadow_capture_v1.py](code/lead/runs/astra_saved_error_mechanism_v1/candidate_shadow_capture_v1.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/candidate_shadow_replay_capture_v2.py](code/lead/runs/astra_saved_error_mechanism_v1/candidate_shadow_replay_capture_v2.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/fragment_feature24_restore_v1.py](code/lead/runs/astra_saved_error_mechanism_v1/fragment_feature24_restore_v1.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/fragment_shadow_capture_v1.py](code/lead/runs/astra_saved_error_mechanism_v1/fragment_shadow_capture_v1.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/analyze_shadow_v1.py](code/lead/runs/astra_saved_error_mechanism_v1/analyze_shadow_v1.py)
- [code/lead/runs/astra_saved_error_mechanism_v1/analyze_shadow_v2.py](code/lead/runs/astra_saved_error_mechanism_v1/analyze_shadow_v2.py)
- [code/lead/runs/astra_feature24_tail_restore_science_v1/grade_f24_retired_phase_v1.py](code/lead/runs/astra_feature24_tail_restore_science_v1/grade_f24_retired_phase_v1.py)
- [code/lead/runs/astra_feature24_tail_restore_science_v1/f24_source_contract_v1.py](code/lead/runs/astra_feature24_tail_restore_science_v1/f24_source_contract_v1.py)
- [code/lead/runs/astra_feature24_tail_restore_science_v1/finite_f24_astra_v1.py](code/lead/runs/astra_feature24_tail_restore_science_v1/finite_f24_astra_v1.py)

<details>
<summary>Additional programs and dependencies (24)</summary>

- [code/lead/runs/astra_feature24_tail_restore_science_v1/native_f24_judge_astra_v1.py](code/lead/runs/astra_feature24_tail_restore_science_v1/native_f24_judge_astra_v1.py)
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
- [code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py](code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3539](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3539)

<details>
<summary>All result files and records (10)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [FEATURE24_CANDIDATE_MANIFEST_v1.json](FEATURE24_CANDIDATE_MANIFEST_v1.json) · result
- [ACTUAL_FNP_TINY_v1.json](ACTUAL_FNP_TINY_v1.json) · result
- [ACTUAL_FNP_TINY_v2.json](ACTUAL_FNP_TINY_v2.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [ROOT_SOURCE_PEER_v1.json](ROOT_SOURCE_PEER_v1.json) · result
- [HARNESS_PREPARATION_v1.json](HARNESS_PREPARATION_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json](ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-read-tail-feature-24-restoration-pilot-adjusted-0-346987) · [Copy provenance](../../../PROVENANCE.json)


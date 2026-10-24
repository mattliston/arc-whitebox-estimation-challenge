# Paid-prefix row-energy feature 22: all-24 adjusted -0.162470%, out-of-sample -0.145096%

2026-10-15T22:24:19Z · experiment · adopted · measured

A fixed correction to row-energy feature 22 using the paid prefix passed the paired 24 and out-of-sample rules on the current best parent and was submitted after request-only repair.

Recorded measurement:

    MSE-.203213%,bill+.040743%,net-.162470%,SE.061628;OOS8-.145096%;fresh16-.171158%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_prefix_row_energy22_v1/candidate_prefix_energy22_HELD_v1.py](code/lead/runs/astra_prefix_row_energy22_v1/candidate_prefix_energy22_HELD_v1.py)
- [code/lead/runs/astra_prefix_row_energy22_v1/fragment_prefix_energy22_HELD_v1.py](code/lead/runs/astra_prefix_row_energy22_v1/fragment_prefix_energy22_HELD_v1.py)
- [code/lead/runs/astra_prefix_row_energy22_v1/check_actual_tiny_v1.py](code/lead/runs/astra_prefix_row_energy22_v1/check_actual_tiny_v1.py)
- [code/lead/runs/astra_prefix_row_energy22_v1/check_actual_tiny_v2.py](code/lead/runs/astra_prefix_row_energy22_v1/check_actual_tiny_v2.py)
- [code/lead/runs/astra_prefix_row_energy22_science_v1/grade_rpe_retired_phase_v1.py](code/lead/runs/astra_prefix_row_energy22_science_v1/grade_rpe_retired_phase_v1.py)
- [code/lead/runs/astra_prefix_row_energy22_science_v1/rpe_source_contract_v1.py](code/lead/runs/astra_prefix_row_energy22_science_v1/rpe_source_contract_v1.py)
- [code/lead/runs/astra_prefix_row_energy22_science_v1/finite_rpe_astra_v1.py](code/lead/runs/astra_prefix_row_energy22_science_v1/finite_rpe_astra_v1.py)
- [code/lead/runs/astra_prefix_row_energy22_science_v1/native_rpe_judge_astra_v1.py](code/lead/runs/astra_prefix_row_energy22_science_v1/native_rpe_judge_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3551](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3551)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SOURCE_HELD_v1.json](SOURCE_HELD_v1.json) · result
- [ACTUAL_FNP_TINY_v2.json](ACTUAL_FNP_TINY_v2.json) · result
- [LARGER_ACCURACY_PEER_PASS_v1.json](LARGER_ACCURACY_PEER_PASS_v1.json) · result
- [ALL24_DECISION_astra_v1.json](ALL24_DECISION_astra_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PREPARED_FOR_ROOT_REVIEW_v1.json](PREPARED_FOR_ROOT_REVIEW_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [RELEASE_HANDOFF_astra_v1.json](RELEASE_HANDOFF_astra_v1.json) · result
- [REMAINING_RETIREMENT_astra_v1.json](REMAINING_RETIREMENT_astra_v1.json) · result
- [ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json](ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336334: paid-prefix row-energy correction with exact split-view batching](../../../submissions/336334/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-paid-prefix-row-energy-feature-22-all-24-adjusted) · [Copy provenance](../../../PROVENANCE.json)


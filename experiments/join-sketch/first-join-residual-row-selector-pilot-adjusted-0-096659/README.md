# First-join residual-row selector: pilot adjusted +0.096659%

2026-10-15T20:53:48Z · experiment · rejected · measured

Selecting first-join rows by the residual diagonal increased raw and adjusted errors and did not advance beyond eight networks.

Recorded measurement:

    MSE+.089544%,bill+.007116%,net+.096659%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_firstjoin_commonmode_selector_v1/build_jrs_HELD_v1.py](code/lead/runs/astra_firstjoin_commonmode_selector_v1/build_jrs_HELD_v1.py)
- [code/lead/runs/astra_firstjoin_commonmode_selector_v1/candidate_jrs_HELD_v1.py](code/lead/runs/astra_firstjoin_commonmode_selector_v1/candidate_jrs_HELD_v1.py)
- [code/lead/runs/astra_firstjoin_commonmode_selector_v1/fragment_jrs_HELD_v1.py](code/lead/runs/astra_firstjoin_commonmode_selector_v1/fragment_jrs_HELD_v1.py)
- [code/lead/runs/astra_firstjoin_commonmode_selector_v1/check_jrs_HELD_v1.py](code/lead/runs/astra_firstjoin_commonmode_selector_v1/check_jrs_HELD_v1.py)
- [code/lead/runs/astra_firstjoin_commonmode_selector_v1/check_saved_selector_v1.py](code/lead/runs/astra_firstjoin_commonmode_selector_v1/check_saved_selector_v1.py)
- [code/lead/runs/astra_join_residual_selector_science_v1/grade_jrs_retired_phase_v1.py](code/lead/runs/astra_join_residual_selector_science_v1/grade_jrs_retired_phase_v1.py)
- [code/lead/runs/astra_join_residual_selector_science_v1/bind_source_ready_v1.py](code/lead/runs/astra_join_residual_selector_science_v1/bind_source_ready_v1.py)
- [code/lead/runs/astra_join_residual_selector_science_v1/jrs_source_contract_v1.py](code/lead/runs/astra_join_residual_selector_science_v1/jrs_source_contract_v1.py)
- [code/lead/runs/astra_join_residual_selector_science_v1/finite_jrs_astra_v1.py](code/lead/runs/astra_join_residual_selector_science_v1/finite_jrs_astra_v1.py)
- [code/lead/runs/astra_join_residual_selector_science_v1/native_jrs_judge_astra_v1.py](code/lead/runs/astra_join_residual_selector_science_v1/native_jrs_judge_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3531](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3531)

<details>
<summary>All result files and records (15)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [COMPUTE_SOURCE_PEER_PASS_v1.json](COMPUTE_SOURCE_PEER_PASS_v1.json) · result
- [SOURCE_MANIFEST_HELD_v1.json](SOURCE_MANIFEST_HELD_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result
- [ACTUAL_RETIREMENT_v1.json](ACTUAL_RETIREMENT_v1.json) · result
- [GUARD_v1.json](GUARD_v1.json) · result
- [SAVED_SELECTOR_GEOMETRY_v1.json](SAVED_SELECTOR_GEOMETRY_v1.json) · result
- [TINY_GUARD_v1.json](TINY_GUARD_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [SOURCE_PEER_PASS_v1.json](SOURCE_PEER_PASS_v1.json) · result
- [HARNESS_PREP_HELD_v1.json](HARNESS_PREP_HELD_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [ROOT_DIRECT_PILOT_AUTHORITY_v1.json](ROOT_DIRECT_PILOT_AUTHORITY_v1.json) · result
- [ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json](ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-first-join-residual-row-selector-pilot-adjusted-0-096659) · [Copy provenance](../../../PROVENANCE.json)


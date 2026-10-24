# Young-tail rank eight with 128 fewer prefix rows: pilot adjusted +11.201516%

2026-10-15T19:48:34Z · experiment · rejected · measured

The fixed larger low-rank tail was evaluated despite an unfavorable saved covariance proxy and failed decisively. No basis, rank or band tuning followed.

Recorded measurement:

    saved covariance error20.2315x;L7-14absolute1.124657e-3 vs3.599435e-4;pilot MSE+14.256688%,bill-3.055172%,net+11.201516%,SE2.689788%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_young_tail_rank8_source_v1/build_y8_HELD_v1.py](code/lead/runs/astra_young_tail_rank8_source_v1/build_y8_HELD_v1.py)
- [code/lead/runs/astra_young_tail_rank8_source_v1/build_y8_HELD_v2.py](code/lead/runs/astra_young_tail_rank8_source_v1/build_y8_HELD_v2.py)
- [code/lead/runs/astra_young_tail_rank8_source_v1/candidate_y8_HELD_v1.py](code/lead/runs/astra_young_tail_rank8_source_v1/candidate_y8_HELD_v1.py)
- [code/lead/runs/astra_young_tail_rank8_source_v1/candidate_y8_HELD_v2.py](code/lead/runs/astra_young_tail_rank8_source_v1/candidate_y8_HELD_v2.py)
- [code/lead/runs/astra_young_tail_rank8_source_v1/fragment_y8_HELD_v1.py](code/lead/runs/astra_young_tail_rank8_source_v1/fragment_y8_HELD_v1.py)
- [code/lead/runs/astra_young_tail_rank8_source_v1/fragment_y8_HELD_v2.py](code/lead/runs/astra_young_tail_rank8_source_v1/fragment_y8_HELD_v2.py)
- [code/lead/runs/astra_young_tail_rank8_source_v1/fragment_y8_template_HELD_v1.py](code/lead/runs/astra_young_tail_rank8_source_v1/fragment_y8_template_HELD_v1.py)
- [code/lead/runs/astra_young_tail_rank8_science_v1/grade_ytr8_retired_phase_v1.py](code/lead/runs/astra_young_tail_rank8_science_v1/grade_ytr8_retired_phase_v1.py)
- [code/lead/runs/astra_young_tail_rank8_science_v1/bind_source_ready_v1.py](code/lead/runs/astra_young_tail_rank8_science_v1/bind_source_ready_v1.py)
- [code/lead/runs/astra_young_tail_rank8_science_v1/finite_ytr8_astra_v1.py](code/lead/runs/astra_young_tail_rank8_science_v1/finite_ytr8_astra_v1.py)
- [code/lead/runs/astra_young_tail_rank8_science_v1/native_ytr8_judge_astra_v1.py](code/lead/runs/astra_young_tail_rank8_science_v1/native_ytr8_judge_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

<details>
<summary>Additional programs and dependencies (22)</summary>

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3499](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3499)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3495](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3495)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3497](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3497)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [FRONTIER_SOURCE_PEER_PASS_v2.json](FRONTIER_SOURCE_PEER_PASS_v2.json) · result
- [SOURCE_BUILD_HELD_v1.json](SOURCE_BUILD_HELD_v1.json) · result
- [SOURCE_BUILD_HELD_v2.json](SOURCE_BUILD_HELD_v2.json) · result
- [SOURCE_CHECK_ACTUAL_EXITS_v1.json](SOURCE_CHECK_ACTUAL_EXITS_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [SOURCE_PEER_PASS_v1.json](SOURCE_PEER_PASS_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json](ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-young-tail-rank-eight-128-fewer-prefix-rows-pilot) · [Copy provenance](../../../PROVENANCE.json)


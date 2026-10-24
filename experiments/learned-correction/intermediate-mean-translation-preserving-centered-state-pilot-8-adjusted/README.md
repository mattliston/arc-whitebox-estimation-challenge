# Intermediate mean translation preserving centered state: pilot-8 adjusted +1.130840%

2026-10-15T17:37:20Z · experiment · rejected · measured

The same frozen all-eight-layer mean map was applied as a deterministic translation while preserving the original centered covariance, births, nonlinear-pair state and trace. It failed without continuation or tuning.

Recorded measurement:

    QR MSE +1.128928%,bill+.001912%,net+1.130840%,SE1.36460%; e406+1.096730%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_native_intermediate_mean_translation_v1/grade_lmt_retired_phase_v1.py](code/lead/runs/astra_native_intermediate_mean_translation_v1/grade_lmt_retired_phase_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_translation_v1/candidate_QR_mean_translation_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_translation_v1/candidate_QR_mean_translation_HELD_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_translation_v1/fragment_mean_translation_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_translation_v1/fragment_mean_translation_HELD_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_translation_v1/check_mean_translation_tiny_v1.py](code/lead/runs/astra_native_intermediate_mean_translation_v1/check_mean_translation_tiny_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_translation_v1/finite_lmt_astra_v1.py](code/lead/runs/astra_native_intermediate_mean_translation_v1/finite_lmt_astra_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_translation_v1/native_lmt_judge_astra_v1.py](code/lead/runs/astra_native_intermediate_mean_translation_v1/native_lmt_judge_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (17)</summary>

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3449](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3449)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [TINY_MEAN_TRANSLATION_RESULT_v1.json](TINY_MEAN_TRANSLATION_RESULT_v1.json) · result
- [SOURCE_PLAN_HELD_v1.json](SOURCE_PLAN_HELD_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result
- [LIFETIME_ALIAS_CHECK_v1.json](LIFETIME_ALIAS_CHECK_v1.json) · result
- [PILOT_RETIREMENT_astra_v1.json](PILOT_RETIREMENT_astra_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-intermediate-mean-translation-preserving-centered-state-pilot-8-adjusted) · [Copy provenance](../../../PROVENANCE.json)


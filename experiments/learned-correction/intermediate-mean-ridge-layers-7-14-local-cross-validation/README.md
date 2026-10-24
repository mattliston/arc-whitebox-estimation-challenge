# Intermediate mean ridge at layers 7-14: local cross-validation -0.395022%, rollout adjusted +0.900108%

2026-10-15T17:28:37Z · experiment · rejected · measured

A fixed ridge 10/shrink 0.5 repair on 94 local features was permitted one unchanged rollout despite missing the local half-percent criterion. The rollout failed on final accuracy even though all implementation checks passed.

Recorded measurement:

    local CV -.395022%, SE.06217%,27/32 lower,all8 folds negative; pilot QR MSE +.897849%,bill+.002259%,net+.900108%,SE1.31796%; e406 +.865999%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_native_intermediate_mean_refit_v1/grade_lmr_retired_phase_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/grade_lmr_retired_phase_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/candidate_QR_intermediate_capture_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/candidate_QR_intermediate_capture_HELD_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/candidate_QR_local_mean_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/candidate_QR_local_mean_HELD_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/check_capture_source_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/check_capture_source_HELD_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_intermediate_capture_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_intermediate_capture_HELD_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_local_mean_FROZEN_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_local_mean_FROZEN_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_local_mean_TEMPLATE_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_local_mean_TEMPLATE_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_local_mean_TEMPLATE_v2.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/fragment_local_mean_TEMPLATE_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact8_v1/build_uc8_source_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/build_uc8_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/candidate_uc8_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/candidate_uc8_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/finite_uc8_diagnostic_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/finite_uc8_diagnostic_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (22)</summary>

- [code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py)
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
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/accept_capture_retirement_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/accept_capture_retirement_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/finite_intermediate_capture_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/finite_intermediate_capture_HELD_v1.py)
- [code/lead/runs/astra_native_intermediate_mean_refit_v1/native_intermediate_capture_HELD_v1.py](code/lead/runs/astra_native_intermediate_mean_refit_v1/native_intermediate_capture_HELD_v1.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_young_read7_v1/candidate_ON_HELD_v2.py](code/lead/runs/astra_young_read7_v1/candidate_ON_HELD_v2.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3446](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3446)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3445](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3445)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [LOCAL_NETWORK_BLOCK_CV_RESULT_v1.json](LOCAL_NETWORK_BLOCK_CV_RESULT_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [TINY_LOCAL_MEAN_RESULT_v1.json](TINY_LOCAL_MEAN_RESULT_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-intermediate-mean-ridge-layers-7-14-local-cross-validation) · [Copy provenance](../../../PROVENANCE.json)


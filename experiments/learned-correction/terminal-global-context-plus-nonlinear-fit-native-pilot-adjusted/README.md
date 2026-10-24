# Terminal global context plus nonlinear fit: native pilot adjusted +0.541490% despite cross-validation -0.373141%

2026-10-15T15:46:36Z · experiment · rejected · measured

Adding fixed global-context ridge features to the terminal nonlinear correction improved training cross-validation but failed the native pilot. The combined archive was held without continuation or delivery.

Recorded measurement:

    CV MSE -.373141% vs QR; incremental -.203890% vs MLP; native8 vs QR MSE +.538245%, bill +.003245%, net +.541490%; vs e406 +.507380%; added cost 7.956M FLOPs/35 ops

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_terminal_context_mlp_stack_v1/grade_retired_phase_v1.py](code/lead/runs/astra_terminal_context_mlp_stack_v1/grade_retired_phase_v1.py)
- [code/lead/runs/astra_terminal_context_mlp_stack_v1/candidate_QR_MLP_context_ON_HELD_v1.py](code/lead/runs/astra_terminal_context_mlp_stack_v1/candidate_QR_MLP_context_ON_HELD_v1.py)
- [code/lead/runs/astra_terminal_context_mlp_stack_v1/fragment_terminal_context_HELD_v1.py](code/lead/runs/astra_terminal_context_mlp_stack_v1/fragment_terminal_context_HELD_v1.py)
- [code/lead/runs/astra_terminal_context_mlp_stack_v1/check_native_export_v1.py](code/lead/runs/astra_terminal_context_mlp_stack_v1/check_native_export_v1.py)
- [code/lead/runs/astra_terminal_context_mlp_stack_v1/fit_stack_context_HELD_v1.py](code/lead/runs/astra_terminal_context_mlp_stack_v1/fit_stack_context_HELD_v1.py)
- [code/lead/runs/astra_terminal_context_mlp_stack_v1/finite_tcx_astra_v1.py](code/lead/runs/astra_terminal_context_mlp_stack_v1/finite_tcx_astra_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (18)</summary>

- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_native_terminal_mlp16_v1/fragment_terminal_mlp_HELD_v1.py](code/lead/runs/astra_native_terminal_mlp16_v1/fragment_terminal_mlp_HELD_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_terminal_context_mlp_stack_v1/native_tcx_judge_astra_v1.py](code/lead/runs/astra_terminal_context_mlp_stack_v1/native_tcx_judge_astra_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3408](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3408)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3404](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3404)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3406](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3406)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [ROOT_STAGE_A_RECEIPT_v1.json](ROOT_STAGE_A_RECEIPT_v1.json) · result
- [STACK_CONTEXT_CV_RESULT_v1.json](STACK_CONTEXT_CV_RESULT_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result
- [ACTUAL_FIT_RETIREMENT_v1.json](ACTUAL_FIT_RETIREMENT_v1.json) · result
- [FIT_GUARD_v1.json](FIT_GUARD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-terminal-global-context-plus-nonlinear-fit-native-pilot-adjusted) · [Copy provenance](../../../PROVENANCE.json)


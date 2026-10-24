# Native terminal linear ridge on 32 networks: blocked cross-validation selects zero

2026-10-15T15:03:08Z · experiment · rejected · measured

The fixed linear terminal correction captured all 32 native trajectories exactly, then network-block cross-validation selected no correction.

Recorded measurement:

    all32 captured; blocked CV selected zero

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_native_terminal_linear_refit_v1/candidate_QR_terminal_capture_HELD_v1.py](code/lead/runs/astra_native_terminal_linear_refit_v1/candidate_QR_terminal_capture_HELD_v1.py)
- [code/lead/runs/astra_native_terminal_linear_refit_v1/fragment_terminal_capture_HELD_v1.py](code/lead/runs/astra_native_terminal_linear_refit_v1/fragment_terminal_capture_HELD_v1.py)
- [code/lead/runs/astra_native_terminal_linear_refit_v1/fit_network_block_ridge_HELD_v1.py](code/lead/runs/astra_native_terminal_linear_refit_v1/fit_network_block_ridge_HELD_v1.py)
- [code/lead/runs/astra_native_terminal_linear_refit_v1/accept_capture_retirement_v1.py](code/lead/runs/astra_native_terminal_linear_refit_v1/accept_capture_retirement_v1.py)
- [code/lead/runs/astra_native_terminal_linear_refit_v1/finite_terminal_capture_HELD_v1.py](code/lead/runs/astra_native_terminal_linear_refit_v1/finite_terminal_capture_HELD_v1.py)
- [code/lead/runs/astra_native_terminal_linear_refit_v1/native_terminal_capture_HELD_v1.py](code/lead/runs/astra_native_terminal_linear_refit_v1/native_terminal_capture_HELD_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact8_v1/build_uc8_source_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/build_uc8_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/candidate_uc8_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/candidate_uc8_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/finite_uc8_diagnostic_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/finite_uc8_diagnostic_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py)
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
- [code/lead/runs/astra_read_column_oracle_v1/observer_stream_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/observer_stream_HELD_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_young_read7_v1/candidate_ON_HELD_v2.py](code/lead/runs/astra_young_read7_v1/candidate_ON_HELD_v2.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3397](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3397)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [NETWORK_BLOCK_CV_RESULT_v1.json](../linear-94-feature-ridge-nonlinear-94-16-1-fits/NETWORK_BLOCK_CV_RESULT_v1.json) · result
- [SOURCE_TINY_READY_v1.json](SOURCE_TINY_READY_v1.json) · result
- [FIT_RULE_FROZEN_v1.json](FIT_RULE_FROZEN_v1.json) · result
- [CAPTURE_PROTOCOL_PRELAUNCH_IDENTITY_CORRECTION_v0.json](CAPTURE_PROTOCOL_PRELAUNCH_IDENTITY_CORRECTION_v0.json) · result
- [CAPTURE_PROTOCOL_REGISTERED_v1.json](CAPTURE_PROTOCOL_REGISTERED_v1.json) · result
- [MANAGED_SESSIONS_v1.json](MANAGED_SESSIONS_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-native-terminal-linear-ridge-32-networks-blocked-cross-validation) · [Copy provenance](../../../PROVENANCE.json)


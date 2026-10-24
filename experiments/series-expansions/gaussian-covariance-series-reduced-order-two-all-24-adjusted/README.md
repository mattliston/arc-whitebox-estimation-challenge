# Gaussian covariance series reduced to order two: all-24 adjusted +0.142775% on the QR parent

2026-10-15T16:05:48Z · experiment · rejected · measured

The Gaussian-only reduction preserved shared second-order/non-Gaussian work and exact diagonals but failed the full 24 and out-of-sample tests on the QR-plus-nonlinear stack.

Recorded measurement:

    vs QR MSE +0.152246%, bill -0.009471%, net +0.142775%; vs e406 +0.073575%; OOS8 +0.226489%/+0.219045%; pilot8 -0.165790%/-0.199899%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_gaussian_r2_stack_v1/grade_retired_phase_v1.py](code/lead/runs/astra_gaussian_r2_stack_v1/grade_retired_phase_v1.py)
- [code/lead/runs/astra_gaussian_r2_stack_v1/build_source_v1.py](code/lead/runs/astra_gaussian_r2_stack_v1/build_source_v1.py)
- [code/lead/runs/astra_gaussian_r2_stack_v1/candidate_QR_MLP_gaussian_R2_HELD_v1.py](code/lead/runs/astra_gaussian_r2_stack_v1/candidate_QR_MLP_gaussian_R2_HELD_v1.py)
- [code/lead/runs/astra_gaussian_r2_stack_v1/finite_gr2_astra_v1.py](code/lead/runs/astra_gaussian_r2_stack_v1/finite_gr2_astra_v1.py)
- [code/lead/runs/astra_gaussian_r2_stack_v1/native_gr2_judge_astra_v1.py](code/lead/runs/astra_gaussian_r2_stack_v1/native_gr2_judge_astra_v1.py)
- [code/lead/runs/astra_gaussian_r2_stack_v1/parent_psp_series_v1.py](code/lead/runs/astra_gaussian_r2_stack_v1/parent_psp_series_v1.py)
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
- [code/lead/runs/astra_native_terminal_mlp16_v1/candidate_QR_terminal_mlp_ON_HELD_v1.py](code/lead/runs/astra_native_terminal_mlp16_v1/candidate_QR_terminal_mlp_ON_HELD_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3415](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3415)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3412](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3412)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3417](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3417)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ALL24_DECISION_astra_v1.json](ALL24_DECISION_astra_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [ROOT_STAGE_A_RECEIPT_v1.json](ROOT_STAGE_A_RECEIPT_v1.json) · result
- [TINY_FNP_RESULT_v3.json](TINY_FNP_RESULT_v3.json) · result
- [SOURCE_CONTRACT_READY_v1.json](SOURCE_CONTRACT_READY_v1.json) · result
- [SOURCE_PROOF_v1.json](SOURCE_PROOF_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-gaussian-covariance-series-reduced-order-two-all-24-adjusted) · [Copy provenance](../../../PROVENANCE.json)


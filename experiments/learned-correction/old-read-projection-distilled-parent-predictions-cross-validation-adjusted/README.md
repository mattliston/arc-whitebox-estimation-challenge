# Old-read projection distilled to the parent predictions: cross-validation adjusted +1.30503%

2026-10-15T16:43:16Z · experiment · rejected · measured

A fixed teacher-only fit tried to recover the discarded old-read information from 32 native captures. It worsened teacher-error recovery and was closed without an all 32 refit, export or native pilot.

Recorded measurement:

    whole CV adjusted +1.30503% vs QR; MSE +2.45485%, bill -1.12227%; teacher-error recovery -.52396%; 10/32 wins; unrefitted cheap +1.25518%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_protected_old_coadapt_v1/candidate_POP_terminal_capture_HELD_v1.py](code/lead/runs/astra_protected_old_coadapt_v1/candidate_POP_terminal_capture_HELD_v1.py)
- [code/lead/runs/astra_protected_old_coadapt_v1/fit_warmstart_distillation_HELD_v1.py](code/lead/runs/astra_protected_old_coadapt_v1/fit_warmstart_distillation_HELD_v1.py)
- [code/lead/runs/astra_protected_old_coadapt_v1/fit_warmstart_distillation_HELD_v2.py](code/lead/runs/astra_protected_old_coadapt_v1/fit_warmstart_distillation_HELD_v2.py)
- [code/lead/runs/astra_protected_old_coadapt_v1/accept_capture_retirement_v1.py](code/lead/runs/astra_protected_old_coadapt_v1/accept_capture_retirement_v1.py)
- [code/lead/runs/astra_protected_old_coadapt_v1/finite_terminal_capture_HELD_v1.py](code/lead/runs/astra_protected_old_coadapt_v1/finite_terminal_capture_HELD_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact8_v1/build_uc8_source_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/build_uc8_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/candidate_uc8_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/candidate_uc8_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/finite_uc8_diagnostic_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/finite_uc8_diagnostic_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py](code/lead/runs/astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (16)</summary>

- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_protected_old_coadapt_v1/native_terminal_capture_HELD_v1.py](code/lead/runs/astra_protected_old_coadapt_v1/native_terminal_capture_HELD_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_young_read7_v1/candidate_ON_HELD_v2.py](code/lead/runs/astra_young_read7_v1/candidate_ON_HELD_v2.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3429](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3429)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BLOCKED_CV_RESULT_v1.json](BLOCKED_CV_RESULT_v1.json) · result
- [SOURCE_TINY_READY_v1.json](SOURCE_TINY_READY_v1.json) · result
- [ACTUAL_FIT_RETIREMENT_v1.json](ACTUAL_FIT_RETIREMENT_v1.json) · result
- [FIT_RULE_FROZEN_v1.json](FIT_RULE_FROZEN_v1.json) · result
- [BRANCH_TARIFF_v1.json](BRANCH_TARIFF_v1.json) · result
- [CAPTURE_PROTOCOL_REGISTERED_v1.json](CAPTURE_PROTOCOL_REGISTERED_v1.json) · result
- [fold0_FREEZE_v1.json](fold0_FREEZE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-old-read-projection-distilled-parent-predictions-cross-validation-adjusted) · [Copy provenance](../../../PROVENANCE.json)


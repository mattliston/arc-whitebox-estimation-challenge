# Old-pair pruning at layers 10-12: native pilot adjusted +2.855901% for bill -0.297149%

2026-10-15T17:07:48Z · experiment · rejected · measured

The fixed read-aware selector dropped the bottom quarter of old pairs while retaining the original weights and full carried state. It failed the pilot and did not advance to release.

Recorded measurement:

    MSE +3.15305%; bill -.297149%; adjusted +2.855901% vs QR/+2.821792% vs e406

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_old_pair_prune75_v1/grade_retired_phase_v2.py](code/lead/runs/astra_old_pair_prune75_v1/grade_retired_phase_v2.py)
- [code/lead/runs/astra_old_pair_prune75_v1/candidate_QR_OP75_ON_HELD_v1.py](code/lead/runs/astra_old_pair_prune75_v1/candidate_QR_OP75_ON_HELD_v1.py)
- [code/lead/runs/astra_old_pair_prune75_v1/candidate_QR_OP75_ON_HELD_v2.py](code/lead/runs/astra_old_pair_prune75_v1/candidate_QR_OP75_ON_HELD_v2.py)
- [code/lead/runs/astra_old_pair_prune75_v1/fragment_op75_HELD_v1.py](code/lead/runs/astra_old_pair_prune75_v1/fragment_op75_HELD_v1.py)
- [code/lead/runs/astra_old_pair_prune75_v1/fragment_op75_HELD_v2.py](code/lead/runs/astra_old_pair_prune75_v1/fragment_op75_HELD_v2.py)
- [code/lead/runs/astra_old_pair_prune75_v1/price_selector_HELD_v2.py](code/lead/runs/astra_old_pair_prune75_v1/price_selector_HELD_v2.py)
- [code/lead/runs/astra_old_pair_prune75_v1/finite_op75_astra_v2.py](code/lead/runs/astra_old_pair_prune75_v1/finite_op75_astra_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (18)</summary>

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
- [code/lead/runs/astra_old_pair_prune75_v1/native_op75_judge_astra_v1.py](code/lead/runs/astra_old_pair_prune75_v1/native_op75_judge_astra_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3438](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3438)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [TINY_FNP_RESULT_v2.json](TINY_FNP_RESULT_v2.json) · result
- [SOURCE_PROOF_v1.json](SOURCE_PROOF_v1.json) · result
- [SOURCE_PROOF_v2.json](SOURCE_PROOF_v2.json) · result
- [SOURCE_READY_v2.json](SOURCE_READY_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-old-pair-pruning-layers-10-12-native-pilot-adjusted) · [Copy provenance](../../../PROVENANCE.json)


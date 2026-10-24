# Layer-10 join rank 384: initial single-network adjusted +2.64766%, later evidence requires reconsideration

2026-10-15T23:48:20Z · experiment · rejected · measured

The fixed rank 384 candidate initially closed on one network for cost and raw-error loss. Subsequent eight- and twenty-four-network noise-corrected tests evaluated the same capacity change separately.

Recorded measurement:

    single7103248 bill+1.61418%,rawlog+1.04637%,net+2.64766%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_full384_smirnov_v1/build_fixed_dispatch_v2.py](code/lead/runs/astra_full384_smirnov_v1/build_fixed_dispatch_v2.py)
- [code/lead/runs/astra_full384_smirnov_v1/build_rank_base_v1.py](code/lead/runs/astra_full384_smirnov_v1/build_rank_base_v1.py)
- [code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py](code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py)
- [code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_RANK_BASE_HELD_v1.py](code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_RANK_BASE_HELD_v1.py)
- [code/lead/runs/astra_full384_smirnov_v1/tiny_fixed384_v1.py](code/lead/runs/astra_full384_smirnov_v1/tiny_fixed384_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/grade_bill1_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/grade_bill1_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/full384_source_contract_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/full384_source_contract_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/finite_full384_bill1_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/finite_full384_bill1_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/native_bill1_judge_astra_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/native_bill1_judge_astra_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/prepare_bill1_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/prepare_bill1_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/prepare_bill1_v2.py](code/lead/runs/astra_full384_smirnov_bill1_v1/prepare_bill1_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

<details>
<summary>Additional programs and dependencies (28)</summary>

- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_full384_current_feasibility_v1/candidate_full384_current_UNVALIDATED_v1.py](code/lead/runs/astra_full384_current_feasibility_v1/candidate_full384_current_UNVALIDATED_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py](code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_batched_slots_v2.py](code/lead/runs/astra_smirnov_current24_virtual_v1/smirnov_batched_slots_v2.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/finite_final_astra_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/finite_final_astra_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/native_final_judge_astra_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/native_final_judge_astra_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3579](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3579)

<details>
<summary>All result files and records (10)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [FIXED_SOURCE_READY_v2.json](FIXED_SOURCE_READY_v2.json) · result
- [LARGER_ACCURACY_SOURCE_PEER_v1.json](LARGER_ACCURACY_SOURCE_PEER_v1.json) · result
- [RANK_BASE_SOURCE_READY_v1.json](RANK_BASE_SOURCE_READY_v1.json) · result
- [ROOT_ONE_MODEL_GRANT_v1.json](ROOT_ONE_MODEL_GRANT_v1.json) · result
- [TINY_FIXED384_v1.json](TINY_FIXED384_v1.json) · result
- [BILL1_DECISION_v1.json](BILL1_DECISION_v1.json) · result
- [BILL1_RETIREMENT_v1.json](BILL1_RETIREMENT_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [PROTOCOL_REGISTERED_v1.json](PROTOCOL_REGISTERED_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-layer-10-join-rank-384-initial-single-network-adjusted) · [Copy provenance](../../../PROVENANCE.json)


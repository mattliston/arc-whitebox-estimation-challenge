# Energy-ordered 64-coordinate read-cut oracle: three-network MSE +0.2229%, fails +0.15% criterion

2026-10-16T00:36:47Z · experiment · rejected · measured

An unmetered singular-value decomposition removed far less product energy than the QR-tail cut, yet its final error still failed the pre-registered ceiling. A deployable rotation would add arithmetic comparable to the saving.

Recorded measurement:

    MSE-.1455/+.6273/+.1868%;mean+.2229%;QR-tail+1.6645%;energy1.1e-4..8.8e-4vs2.5e-3..9.3e-3;~1.18G saving

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_u64_oracle_v1/grade_oracle3_v1.py](code/lead/runs/opus_u64_oracle_v1/grade_oracle3_v1.py)
- [code/lead/runs/opus_u64_oracle_v1/candidate_u64_oracle_svd_NOTDEPLOYABLE_v1.py](code/lead/runs/opus_u64_oracle_v1/candidate_u64_oracle_svd_NOTDEPLOYABLE_v1.py)
- [code/lead/runs/opus_u64_oracle_v1/finite_u64_oracle_v1.py](code/lead/runs/opus_u64_oracle_v1/finite_u64_oracle_v1.py)
- [code/lead/runs/opus_u64_oracle_v1/native_u64_oracle_judge_v1.py](code/lead/runs/opus_u64_oracle_v1/native_u64_oracle_judge_v1.py)
- [code/lead/runs/opus_u64_oracle_v1/prepare_oracle_v1.py](code/lead/runs/opus_u64_oracle_v1/prepare_oracle_v1.py)
- [code/lead/runs/opus_u64_oracle_v1/u64_screen_contract_v1.py](code/lead/runs/opus_u64_oracle_v1/u64_screen_contract_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_read256_inner_cost_v1/candidate_uniform64_read_HELD_v1.py](code/lead/runs/astra_read256_inner_cost_v1/candidate_uniform64_read_HELD_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/finite_uniform64_bill1_v1.py](code/lead/runs/astra_uniform64_bill1_v1/finite_uniform64_bill1_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/native_bill1_judge_astra_v1.py](code/lead/runs/astra_uniform64_bill1_v1/native_bill1_judge_astra_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/uniform64_source_contract_v1.py](code/lead/runs/astra_uniform64_bill1_v1/uniform64_source_contract_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/lead/runs/opus_u64_screen_v1/finite_u64_screen_v1.py](code/lead/runs/opus_u64_screen_v1/finite_u64_screen_v1.py)
- [code/lead/runs/opus_u64_screen_v1/finite_u64_screen_v2.py](code/lead/runs/opus_u64_screen_v1/finite_u64_screen_v2.py)
- [code/lead/runs/opus_u64_screen_v1/native_u64_screen_judge_v1.py](code/lead/runs/opus_u64_screen_v1/native_u64_screen_judge_v1.py)
- [code/lead/runs/opus_u64_screen_v1/prepare_u64_screen_v1.py](code/lead/runs/opus_u64_screen_v1/prepare_u64_screen_v1.py)
- [code/lead/runs/opus_u64_screen_v1/prepare_u64_screen_v2.py](code/lead/runs/opus_u64_screen_v1/prepare_u64_screen_v2.py)
- [code/lead/runs/opus_u64_screen_v1/u64_screen_contract_v1.py](code/lead/runs/opus_u64_screen_v1/u64_screen_contract_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3601](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3601)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ORACLE3_RESULT_v1.json](ORACLE3_RESULT_v1.json) · result
- [OPUS_LEAD_GRANT_v1.json](OPUS_LEAD_GRANT_v1.json) · result
- [PROTOCOL_REGISTERED_oracle_v1.json](PROTOCOL_REGISTERED_oracle_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-energy-ordered-64-coordinate-read-cut-oracle-three-network) · [Copy provenance](../../../PROVENANCE.json)


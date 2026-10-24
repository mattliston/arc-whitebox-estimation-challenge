# Dropping 64 trailing read coordinates: ten-network adjusted +0.2033%

2026-10-16T00:31:46Z · experiment · rejected · measured

The uniform 64-coordinate read cut did not reproduce its favorable first network. Porting it to the expanded table also failed, so the conditional 24-network test did not run.

Recorded measurement:

    ten MSE+.6950%,SE.3148,3/10lower;bill-.4916%,SE.0039;net+.2033%,SE.3169,5/10;port bill-.4911%,MSE+.9522%,net+.4611%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_read256_inner_cost_v1/build_uniform64_source_v1.py](code/lead/runs/astra_read256_inner_cost_v1/build_uniform64_source_v1.py)
- [code/lead/runs/astra_read256_inner_cost_v1/candidate_uniform64_read_HELD_v1.py](code/lead/runs/astra_read256_inner_cost_v1/candidate_uniform64_read_HELD_v1.py)
- [code/lead/runs/astra_read256_inner_cost_v1/price_read256_v1.py](code/lead/runs/astra_read256_inner_cost_v1/price_read256_v1.py)
- [code/lead/runs/astra_read256_inner_cost_v1/price_uniform64_v2.py](code/lead/runs/astra_read256_inner_cost_v1/price_uniform64_v2.py)
- [code/lead/runs/astra_read256_inner_cost_v1/cache_uniform64_v1.py](code/lead/runs/astra_read256_inner_cost_v1/cache_uniform64_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/grade_actual_scope_v2.py](code/lead/runs/astra_uniform64_bill1_v1/grade_actual_scope_v2.py)
- [code/lead/runs/astra_uniform64_bill1_v1/grade_bill1_v1.py](code/lead/runs/astra_uniform64_bill1_v1/grade_bill1_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/uniform64_source_contract_v1.py](code/lead/runs/astra_uniform64_bill1_v1/uniform64_source_contract_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/finite_uniform64_bill1_v1.py](code/lead/runs/astra_uniform64_bill1_v1/finite_uniform64_bill1_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/native_bill1_judge_astra_v1.py](code/lead/runs/astra_uniform64_bill1_v1/native_bill1_judge_astra_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/prepare_bill1_v1.py](code/lead/runs/astra_uniform64_bill1_v1/prepare_bill1_v1.py)
- [code/lead/runs/astra_uniform64_bill1_v1/prepare_bill1_v2.py](code/lead/runs/astra_uniform64_bill1_v1/prepare_bill1_v2.py)

<details>
<summary>Additional programs and dependencies (32)</summary>

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/finite_full384_bill1_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/finite_full384_bill1_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/full384_source_contract_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/full384_source_contract_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/grade_bill1_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/grade_bill1_v1.py)
- [code/lead/runs/astra_full384_smirnov_bill1_v1/native_bill1_judge_astra_v1.py](code/lead/runs/astra_full384_smirnov_bill1_v1/native_bill1_judge_astra_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py](code/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/price_v1.py](code/lead/runs/astra_smirnov_current24_virtual_v1/price_v1.py)
- [code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py](code/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3598](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3598)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3591](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3591)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [UNIFORM64_SOURCE_HELD_v1.json](UNIFORM64_SOURCE_HELD_v1.json) · result
- [ACTIVE_SCOPE_CORRECTION_v1.json](ACTIVE_SCOPE_CORRECTION_v1.json) · result
- [CACHE_UNIFORM64_v1.json](CACHE_UNIFORM64_v1.json) · result
- [CURRENT24_WARM_TARIFF_v1.json](CURRENT24_WARM_TARIFF_v1.json) · result
- [LANE_HANDOFF_v1.json](LANE_HANDOFF_v1.json) · result
- [LANE_HANDOFF_v2.json](LANE_HANDOFF_v2.json) · result
- [LANE_HANDOFF_v3.json](LANE_HANDOFF_v3.json) · result
- [BILL1_DECISION_v2.json](BILL1_DECISION_v2.json) · result
- [ACTUAL_COVERAGE_ACCEPTANCE_v2.json](ACTUAL_COVERAGE_ACCEPTANCE_v2.json) · result
- [BILL1_RETIREMENT_v2.json](BILL1_RETIREMENT_v2.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-dropping-64-trailing-read-coordinates-ten-network-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)


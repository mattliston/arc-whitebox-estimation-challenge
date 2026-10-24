# Kernel dispatch expanded to 167 shapes with layer-one read reallocation: all-24 adjusted -0.2096793%

2026-10-16T00:11:06Z · experiment · adopted · measured

The fixed table expansion and first-read reallocation passed a prospectively assigned 24-network paired test and the official 120-second release gates. The resulting submission later failed on platform wall and residual limits.

Recorded measurement:

    all24 net-.2096793%,SE.0485719%;OOS8-.1720799%;risk19.79%,worst113.1s,setup4.441s,slow4.585s

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/grade_table_r7_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/grade_table_r7_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/table_r7_source_contract_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/table_r7_source_contract_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/finite_table_r7_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/finite_table_r7_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/native_table_r7_judge_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/native_table_r7_judge_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/prepare_science_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_table_r7_science_v1/prepare_science_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/census_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/census_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/build_table_r7_stack_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/build_table_r7_stack_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/build_unseen_table_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/build_unseen_table_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen32_table_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen32_table_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/price_unseen_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/price_unseen_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_r7_firstread_reallocation_v1/candidate_r7_firstread_reallocation_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_r7_firstread_reallocation_v1/candidate_r7_firstread_reallocation_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (61)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_r7_firstread_reallocation_v1/candidate_r7_firstread_reallocation_HELD_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_r7_firstread_reallocation_v1/candidate_r7_firstread_reallocation_HELD_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/grade_captured_setup_parity_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/grade_captured_setup_parity_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/prepare_boundary_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/prepare_boundary_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/finalize_metadata_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/finalize_metadata_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/finish_table_r7_release_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/finish_table_r7_release_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/launch_native35_100_replays_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/launch_native35_100_replays_v1.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/quick.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/quick.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/build_uc32_source_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/candidate_uc32_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/fragment_uc32_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_iop_rescue_v1/candidate_fhm_dense_IOP_OFF_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_fhm_iop_rescue_v1/fragment_IOP_OFF_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_boundary_v1/early_count_gate_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_boundary_v1/early_count_gate_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_boundary_v1/finite_gop2_current_boundary_HELD_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_boundary_v1/finite_gop2_current_boundary_HELD_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_delivery_v1/candidate_current4e_GOP2_ON_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_own20_v1/finite_gop2_current_census_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_current_own20_v1/finite_gop2_current_census_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_iop_view_dedup_v1/native_ivd_qualification_judge_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_iop_view_dedup_v1/owned_native_runner_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_iop_view_dedup_v1/qualification_lifecycle_helpers_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_current24_virtual_v1/price_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_current24_virtual_v1/price_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_final_native_ten_v1/final_source_contract_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_final_native_ten_v1/native_final_judge_astra_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_final_native_ten_v1/native_final_judge_astra_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/early_count_gate_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/early_count_gate_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/release_gate_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/release_gate_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/boundary/early_count_gate_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/boundary/early_count_gate_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/boundary/finite_table_r7_boundary_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_table_r7_delivery_v1/boundary/finite_table_r7_boundary_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/fastgate_pin_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/fastgate_pin_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/scout_native/certify_astra_bank_copy_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/scout_native/certify_astra_bank_copy_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/scout_native/run_cert_astra_bank_copy_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3587](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3587)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3583](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3583)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3595](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3595)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3597](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3597)

<details>
<summary>All result files and records (25)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [FINAL24_DECISION_v1.json](FINAL24_DECISION_v1.json) · result
- [PILOT_DECISION_v1.json](PILOT_DECISION_v1.json) · result
- [COMPLETED_HANDOFF_v2.json](COMPLETED_HANDOFF_v2.json) · result
- [LIVE_HANDOFF_v1.json](LIVE_HANDOFF_v1.json) · result
- [PILOT_RETIREMENT_v1.json](PILOT_RETIREMENT_v1.json) · result
- [PROTOCOL_REGISTERED_astra_v1.json](PROTOCOL_REGISTERED_astra_v1.json) · result
- [REMAINING_RETIREMENT_v1.json](REMAINING_RETIREMENT_v1.json) · result
- [CENSUS_v1.json](CENSUS_v1.json) · result
- [REQUEST_AND_WARM_CEILING_v1.json](REQUEST_AND_WARM_CEILING_v1.json) · result
- [SOURCE_HELD_v1.json](SOURCE_HELD_v1.json) · result
- [STACK_SOURCE_HELD_v2.json](STACK_SOURCE_HELD_v2.json) · result
- [UNSEEN32_PRICE_v1.json](UNSEEN32_PRICE_v1.json) · result
- [LARGER_ACCURACY_STACK_PEER_v1.json](LARGER_ACCURACY_STACK_PEER_v1.json) · result
- [SOURCE_HELD_v1-6016327f.json](SOURCE_HELD_v1-6016327f.json) · result
- [SOURCE_HELD_v2.json](SOURCE_HELD_v2.json) · result
- [SOURCE_HELD_v3.json](SOURCE_HELD_v3.json) · result
- [TINY_SOURCE_TARIFF_v2.json](TINY_SOURCE_TARIFF_v2.json) · result
- [RELEASE_CONFIRMED_v1.json](RELEASE_CONFIRMED_v1.json) · result
- [DELIVERY_READY_v1.json](DELIVERY_READY_v1.json) · result
- [FINALIZATION_COPY_01_v1.json](FINALIZATION_COPY_01_v1.json) · result
- [FINALIZATION_COPY_03_v1.json](FINALIZATION_COPY_03_v1.json) · result
- [FINALIZATION_SEQUENCE_HELD_v1.json](FINALIZATION_SEQUENCE_HELD_v1.json) · result
- [NATIVE35_100_CAPTURED_SETUP_PARITY_v1.json](NATIVE35_100_CAPTURED_SETUP_PARITY_v1.json) · result
- [NATIVE35_100_CAPTURED_SETUP_RETIREMENT_v1.json](NATIVE35_100_CAPTURED_SETUP_RETIREMENT_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336351: expanded 167-shape read dispatch and first-read reallocation; three runtime failures](../../../submissions/336351/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation) · [Copy provenance](../../../PROVENANCE.json)


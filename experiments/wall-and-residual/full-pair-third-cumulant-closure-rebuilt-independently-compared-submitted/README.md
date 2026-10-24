# Full-pair third-cumulant closure rebuilt independently and compared with the submitted estimator

2026-09-05T00:08Z · experiment · informational · measured

Unlearned identity-radial host: 4.0824e-8 / 4.3204e-8. Covariance-metric fourth cumulant: 3.2346e-8 / 3.2068e-8. candidate_radial_gnn_splitread_v1.py uses only FlopScope numerical ops.

Recorded measurement:

    Unlearned identity-radial host: 4.0824e-8 / 4.3204e-8. Covariance-metric fourth cumulant: 3.2346e-8 / 3.2068e-8. Full covariance host learned on 200 honest rows, fixed160/40 net split, 40epochs, C64, mean head only: 2.7374e-8 / 2.6161e-8 in NumPy. Held-out final one-step MSE 7.554e-9 versus host9.229e-9 (18.15% removal). Feature arrays are captured before current-layer mean correction; no v-head. candidate_radial_gnn_splitread_v1.py uses only FlopScope numerical ops. The final oversized CP read is split algebraically into D4-compatible blocks. Rank35 raw2.73668e-8, local utilization0.340816 (0.352039 without splitting). Official-version local 35/100/35 gate passed. NumPy output maximum deviation 2.19e-6, learned correction maximum deviation1.35e-8; repeated rank35 output byte-identical. PeakRSS7.22GB. Residual time0.394s cold with diagnostic copying, 0.322s second call with copying, 0.330s hot clean call; the cold margin is small. | Panel | Raw | Utilization | Adjusted | Versus recovered329790 card | | --- | ---: | ---: | ---: | ---: | | Public8 | 3.38836e-8 | .3408 | 1.15481e-8 | -28.04% | | Virgin6 | 3.01877e-8 | .3408 | 1.02885e-8 | -20.95% |

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/candidate_radial_gnn_splitread_v1.py](code/astra_phase2_reexamination_v1/candidate_radial_gnn_splitread_v1.py)
- [code/astra_phase2_reexamination_v1/attribute_radial_rows.py](code/astra_phase2_reexamination_v1/attribute_radial_rows.py)
- [code/astra_phase2_reexamination_v1/build_binary_fnp.py](code/astra_phase2_reexamination_v1/build_binary_fnp.py)
- [code/astra_phase2_reexamination_v1/build_radial_fnp.py](code/astra_phase2_reexamination_v1/build_radial_fnp.py)
- [code/astra_phase2_reexamination_v1/build_radial_splitread.py](code/astra_phase2_reexamination_v1/build_radial_splitread.py)
- [code/astra_phase2_reexamination_v1/build_remote_dtype_fix.py](code/astra_phase2_reexamination_v1/build_remote_dtype_fix.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_learned_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_radial_gnn_remote_dtype_v2.py](code/astra_phase2_reexamination_v1/candidate_radial_gnn_remote_dtype_v2.py)
- [code/astra_phase2_reexamination_v1/candidate_radial_gnn_v1.py](code/astra_phase2_reexamination_v1/candidate_radial_gnn_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_recovered_cp3_8k_v1.py](code/astra_phase2_reexamination_v1/candidate_recovered_cp3_8k_v1.py)

<details>
<summary>Additional programs and dependencies (75)</summary>

- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/certify_pair_memory_response_v1.py](code/astra_phase2_reexamination_v1/certify_pair_memory_response_v1.py)
- [code/astra_phase2_reexamination_v1/certify_pair_memory_response_v2.py](code/astra_phase2_reexamination_v1/certify_pair_memory_response_v2.py)
- [code/astra_phase2_reexamination_v1/chain_binary_cap_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_cap_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_binary_diag_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_diag_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_binary_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_centered_feedback_training_v1.py](code/astra_phase2_reexamination_v1/chain_centered_feedback_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_feed_fullpair_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_feed_fullpair_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_secondorder_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_secondorder_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_sphere_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_sphere_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_current_graph_training_v1.py](code/astra_phase2_reexamination_v1/chain_current_graph_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py](code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py)
- [code/astra_phase2_reexamination_v1/check_radial_candidate_v1.py](code/astra_phase2_reexamination_v1/check_radial_candidate_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py](code/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/generate_all_compact_tensorfeat_rows_v1.py](code/astra_phase2_reexamination_v1/generate_all_compact_tensorfeat_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_binary_cap_rows.py](code/astra_phase2_reexamination_v1/generate_binary_cap_rows.py)
- [code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py](code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py)
- [code/astra_phase2_reexamination_v1/generate_binary_diag_rows.py](code/astra_phase2_reexamination_v1/generate_binary_diag_rows.py)
- [code/astra_phase2_reexamination_v1/generate_binary_rows.py](code/astra_phase2_reexamination_v1/generate_binary_rows.py)
- [code/astra_phase2_reexamination_v1/generate_centered_feedback_rows_v1.py](code/astra_phase2_reexamination_v1/generate_centered_feedback_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_feed_fullpair_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_feed_fullpair_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_radialfeed_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_radialfeed_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_secondorder_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_secondorder_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_sphere_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_sphere_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_tensorfeat_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_tensorfeat_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_current_graph_rows_v1.py](code/astra_phase2_reexamination_v1/generate_current_graph_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/generate_response_rows_v1.py](code/astra_phase2_reexamination_v1/generate_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py)
- [code/astra_phase2_reexamination_v1/judge_nonlinear_pair_memory_information_v1.py](code/astra_phase2_reexamination_v1/judge_nonlinear_pair_memory_information_v1.py)
- [code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py](code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py)
- [code/astra_phase2_reexamination_v1/pair_memory_action_physics_v2.py](code/astra_phase2_reexamination_v1/pair_memory_action_physics_v2.py)
- [code/astra_phase2_reexamination_v1/qualify_two_memory_capture_replay_v2.py](code/astra_phase2_reexamination_v1/qualify_two_memory_capture_replay_v2.py)
- [code/astra_phase2_reexamination_v1/radial_binary_cap.py](code/astra_phase2_reexamination_v1/radial_binary_cap.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_binary_cubic.py](code/astra_phase2_reexamination_v1/radial_binary_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_binary_diagfeedback.py](code/astra_phase2_reexamination_v1/radial_binary_diagfeedback.py)
- [code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py](code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py)
- [code/astra_phase2_reexamination_v1/radial_centered_feedback_v1.py](code/astra_phase2_reexamination_v1/radial_centered_feedback_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_secondorder_v1.py](code/astra_phase2_reexamination_v1/radial_compact_secondorder_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_sphere_v1.py](code/astra_phase2_reexamination_v1/radial_compact_sphere_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_tensorfeat_v1.py](code/astra_phase2_reexamination_v1/radial_compact_tensorfeat_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_fnp_tail.py](code/astra_phase2_reexamination_v1/radial_fnp_tail.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_graph_features_v1.py](code/astra_phase2_reexamination_v1/radial_graph_features_v1.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/radial_secondorder_v1.py](code/astra_phase2_reexamination_v1/radial_secondorder_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py](code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py)
- [code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py](code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESEARCH_RECEIPTS_2026_09_04.md#L8](../exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L8)
- [RESEARCH_RECEIPTS_2026_09_04.md#L21](../exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L21)
- [RESEARCH_RECEIPTS_2026_09_04.md#L30](../exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L30)
- [RESEARCH_RECEIPTS_2026_09_04.md#L37](../exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md#L37)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04.md](../exact-kernel-resource-tradeoff/RESEARCH_RECEIPTS_2026_09_04.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted) · [Copy provenance](../../../PROVENANCE.json)


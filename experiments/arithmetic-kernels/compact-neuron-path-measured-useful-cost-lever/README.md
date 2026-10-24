# Compact-neuron path: measured useful cost lever

2026-09-05T00:52Z · experiment · informational · measured

radial_binary_gateprune.py first tested dropping cold cross-neuron state while keeping all means and all univariate diagonal moments. Gate<.001 gave2.79629e-8 /2.94772e-8, effectively unchanged. Cold-state diagonal-shadow experiment: preserve the next-layer diagonal reads of omitted covariance and cubic-star matrices.

Recorded measurement:

    radial_binary_gateprune.py first tested dropping cold cross-neuron state while keeping all means and all univariate diagonal moments. Gate<.001 gave2.79629e-8 /2.94772e-8, effectively unchanged. Gate<.01 gave 3.25649e-8 /3.42971e-8. These first dense emulations were not cost claims. radial_binary_compact.py is the actual compact reference: active and birth counts round UP to32, only active-active CP plane entries are read, full factor pushes permit exact scalar third-moment reads for every neuron, and cold covariance/star diagonals remain. At .001 raw2.79925e-8 /2.94442e-8; at .01 raw2.93709e-8 /3.32566e-8. FlopScope port at .01: candidate_binary_compact_g01_unlearned_v2.py. Rank35 raw2.93826e-8, utilization.233989, wall23.85s, matching its reference. It uses active input blocks plus exact cold diagonals in covariance/star pushes, and padded compact matrix products. Initial v1 tried unsupported indexed writes and failed before a full run; v2 uses documented, metered fnp.put scatter operations. That is a normal public API, not a bypass. Keeping full covariance/star state but pruning only the multi-step cubic carrier (radial_binary_sparsecarry.py) gives at gate.01, birthfloor.001: 2.75715e-8 /2.90177e-8. Larger gates .05 or .1 lose too much raw before coadaptation. This variant is not metered or coadapted yet. Cold-state diagonal-shadow experiment: preserve the next-layer diagonal reads of omitted covariance and cubic-star matrices. At gate=birthfloor.01, 2.94334e-8 /3.16415e-8 is a small mixed gain versus compact. At .05 both thresholds, raw1.455e-7 /1.276e-7. This stronger arm also removes many nearly linear POSITIVE births; it is not a clean test of stronger negative-gate pruning alone. No further shadow implementation undertaken yet.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/radial_binary_gateprune.py](code/astra_phase2_reexamination_v1/radial_binary_gateprune.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_v2.py](code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_v2.py)
- [code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py](code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py)
- [code/astra_phase2_reexamination_v1/build_binary_compact_fnp.py](code/astra_phase2_reexamination_v1/build_binary_compact_fnp.py)
- [code/astra_phase2_reexamination_v1/build_direct_transform_v1.py](code/astra_phase2_reexamination_v1/build_direct_transform_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_direct_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_direct_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_learned_d5_tile8192_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_d5_tile8192_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_direct_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_direct_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (65)</summary>

- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_centered_feedback_training_v1.py](code/astra_phase2_reexamination_v1/chain_centered_feedback_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_feed_fullpair_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_feed_fullpair_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py](code/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_secondorder_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_secondorder_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_sphere_training_v1.py](code/astra_phase2_reexamination_v1/chain_compact_sphere_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_current_graph_training_v1.py](code/astra_phase2_reexamination_v1/chain_current_graph_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py](code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py](code/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/generate_all_compact_tensorfeat_rows_v1.py](code/astra_phase2_reexamination_v1/generate_all_compact_tensorfeat_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_all_fullpair_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_all_fullpair_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py](code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py)
- [code/astra_phase2_reexamination_v1/generate_centered_feedback_rows_v1.py](code/astra_phase2_reexamination_v1/generate_centered_feedback_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_feed_fullpair_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_feed_fullpair_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_radialfeed_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_radialfeed_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_secondorder_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_secondorder_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_sphere_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_sphere_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_compact_tensorfeat_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_tensorfeat_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_current_graph_rows_v1.py](code/astra_phase2_reexamination_v1/generate_current_graph_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_fullpair_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_fullpair_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/generate_response_rows_v1.py](code/astra_phase2_reexamination_v1/generate_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py](code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_cubic.py](code/astra_phase2_reexamination_v1/radial_binary_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_centered_feedback_v1.py](code/astra_phase2_reexamination_v1/radial_centered_feedback_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_secondorder_v1.py](code/astra_phase2_reexamination_v1/radial_compact_secondorder_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_sphere_v1.py](code/astra_phase2_reexamination_v1/radial_compact_sphere_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_tensorfeat_v1.py](code/astra_phase2_reexamination_v1/radial_compact_tensorfeat_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_graph_features_v1.py](code/astra_phase2_reexamination_v1/radial_graph_features_v1.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/radial_secondorder_v1.py](code/astra_phase2_reexamination_v1/radial_secondorder_v1.py)
- [code/astra_phase2_reexamination_v1/screen_compact_final_v1.py](code/astra_phase2_reexamination_v1/screen_compact_final_v1.py)
- [code/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py](code/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/train_compact_final_head_v1.py](code/astra_phase2_reexamination_v1/train_compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py](code/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py](code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py)
- [code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py](code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L33](../bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md#L33)
- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L38](../bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md#L38)
- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L44](../bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md#L44)
- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L1](../bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_LATE.md](../bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-compact-neuron-path-measured-useful-cost-lever) · [Copy provenance](../../../PROVENANCE.json)


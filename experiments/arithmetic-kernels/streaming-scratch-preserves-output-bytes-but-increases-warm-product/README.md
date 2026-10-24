# Streaming scratch preserves output bytes but increases warm product wall to ratio 1.4943203934180924

2026-09-13 · experiment · rejected · measured

The fixed streaming allocation and release policy passed numerical, ownership and billing checks. Both component timing gates failed, showing that bounded live storage and exact bytes did not imply faster execution.

Recorded measurement:

    First.15282954182475805s to.20504137501120567s,ratio1.3416344285473047;warm.11378770787268877s to.17003529239445925s,ratio1.4943203934180924;timed1521986879F.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_streamed_scratch_v1/client.py](code/current_comp_streamed_scratch_v1/client.py)
- [code/current_comp_streamed_scratch_v1/kernel.py](code/current_comp_streamed_scratch_v1/kernel.py)
- [code/current_comp_streamed_scratch_v1/watch.py](code/current_comp_streamed_scratch_v1/watch.py)
- [code/current_comp_streamed_scratch_v1/worker.py](code/current_comp_streamed_scratch_v1/worker.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)

<details>
<summary>Additional programs and dependencies (117)</summary>

- [code/birth_direction_fit_v1/fnp/core.py](code/birth_direction_fit_v1/fnp/core.py)
- [code/birth_direction_fit_v1/fnp/core_v2.py](code/birth_direction_fit_v1/fnp/core_v2.py)
- [code/birth_direction_fit_v1/fnp/reference_numpy.py](code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/birth_direction_fit_v1/judge_terminal.py](code/birth_direction_fit_v1/judge_terminal.py)
- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)
- [code/birth_direction_fit_v1/native.py](code/birth_direction_fit_v1/native.py)
- [code/birth_direction_fit_v1/native_v2.py](code/birth_direction_fit_v1/native_v2.py)
- [code/birth_direction_fit_v1/qualify_model.py](code/birth_direction_fit_v1/qualify_model.py)
- [code/birth_direction_fit_v1/qualify_native_v2.py](code/birth_direction_fit_v1/qualify_native_v2.py)
- [code/birth_direction_fit_v1/qualify_wide_v2.py](code/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/birth_direction_fit_v1/train.py](code/birth_direction_fit_v1/train.py)
- [code/birth_direction_node_head_v1/build.py](code/birth_direction_node_head_v1/build.py)
- [code/birth_direction_node_head_v1/build_terminal_v1/estimator.py](code/birth_direction_node_head_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_node_head_v1/build_zero_v1/estimator.py](code/birth_direction_node_head_v1/build_zero_v1/estimator.py)
- [code/birth_direction_node_head_v1/core.py](code/birth_direction_node_head_v1/core.py)
- [code/birth_direction_node_head_v1/model.py](code/birth_direction_node_head_v1/model.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/current_bd_p_coadaptation_v1/build_terminal.py](code/current_bd_p_coadaptation_v1/build_terminal.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_comp_aligned_push_d6_v1/build_v1/estimator.py](code/current_comp_aligned_push_d6_v1/build_v1/estimator.py)
- [code/current_comp_compact_boxes_v2/rpc37_v1/estimator.py](code/current_comp_compact_boxes_v2/rpc37_v1/estimator.py)
- [code/current_comp_compact_scratch_v1/client.py](code/current_comp_compact_scratch_v1/client.py)
- [code/current_comp_compact_scratch_v1/kernel.py](code/current_comp_compact_scratch_v1/kernel.py)
- [code/current_comp_compact_scratch_v1/watch.py](code/current_comp_compact_scratch_v1/watch.py)
- [code/current_comp_compact_scratch_v1/worker.py](code/current_comp_compact_scratch_v1/worker.py)
- [code/current_comp_cstar_information_v1/information.py](code/current_comp_cstar_information_v1/information.py)
- [code/current_comp_cstar_information_v1/model.py](code/current_comp_cstar_information_v1/model.py)
- [code/current_comp_cstar_information_v1/qualification.py](code/current_comp_cstar_information_v1/qualification.py)
- [code/current_comp_cstar_information_v1/watch_qualification.py](code/current_comp_cstar_information_v1/watch_qualification.py)
- [code/current_comp_cstar_proposal_v1/common.py](code/current_comp_cstar_proposal_v1/common.py)
- [code/current_comp_cstar_proposal_v1/model.py](code/current_comp_cstar_proposal_v1/model.py)
- [code/current_comp_cstar_proposal_v1/watch.py](code/current_comp_cstar_proposal_v1/watch.py)
- [code/current_comp_cstar_proposal_v1/worker.py](code/current_comp_cstar_proposal_v1/worker.py)
- [code/current_comp_joint_p_node_v1/build.py](code/current_comp_joint_p_node_v1/build.py)
- [code/current_comp_joint_p_node_v1/model.py](code/current_comp_joint_p_node_v1/model.py)
- [code/current_comp_joint_p_node_v1/qualification.py](code/current_comp_joint_p_node_v1/qualification.py)
- [code/current_comp_joint_p_node_v1/train.py](code/current_comp_joint_p_node_v1/train.py)
- [code/current_comp_joint_p_node_v1/watch_qualification.py](code/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/current_comp_joint_p_node_v1/watch_train.py](code/current_comp_joint_p_node_v1/watch_train.py)
- [code/current_comp_push_frontier4_v1/kernel.py](code/current_comp_push_frontier4_v1/kernel.py)
- [code/current_comp_push_frontier_boxes_v1/compose.py](code/current_comp_push_frontier_boxes_v1/compose.py)
- [code/current_comp_push_frontier_boxes_v1/kernel.py](code/current_comp_push_frontier_boxes_v1/kernel.py)
- [code/current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py](code/current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py)
- [code/current_comp_push_microbatch3_v1/kernel.py](code/current_comp_push_microbatch3_v1/kernel.py)
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/probe_ad_scalar.py](code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/qualification_v2.py](code/current_comp_receiver_birth_v1/qualification_v2.py)
- [code/current_comp_receiver_birth_v1/qualification_v3.py](code/current_comp_receiver_birth_v1/qualification_v3.py)
- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_root_batch_gram_v1/client.py](code/current_comp_root_batch_gram_v1/client.py)
- [code/current_comp_root_batch_gram_v1/kernel.py](code/current_comp_root_batch_gram_v1/kernel.py)
- [code/current_comp_root_batch_gram_v1/watch.py](code/current_comp_root_batch_gram_v1/watch.py)
- [code/current_comp_root_batch_gram_v1/worker.py](code/current_comp_root_batch_gram_v1/worker.py)
- [code/current_comp_root_batch_gram_v2/client.py](code/current_comp_root_batch_gram_v2/client.py)
- [code/current_comp_root_batch_gram_v2/kernel.py](code/current_comp_root_batch_gram_v2/kernel.py)
- [code/current_comp_root_batch_gram_v2/watch.py](code/current_comp_root_batch_gram_v2/watch.py)
- [code/current_comp_root_batch_gram_v2/worker.py](code/current_comp_root_batch_gram_v2/worker.py)
- [code/current_comp_shared_gain_slice_v1/build_v1/estimator.py](code/current_comp_shared_gain_slice_v1/build_v1/estimator.py)
- [code/current_comp_shared_lazy_rotation_v1/compose.py](code/current_comp_shared_lazy_rotation_v1/compose.py)
- [code/current_comp_shared_lazy_rotation_v1/fnp.py](code/current_comp_shared_lazy_rotation_v1/fnp.py)
- [code/current_comp_shared_lazy_rotation_v1/price.py](code/current_comp_shared_lazy_rotation_v1/price.py)
- [code/current_comp_shared_lazy_rotation_v1/tiny_client.py](code/current_comp_shared_lazy_rotation_v1/tiny_client.py)
- [code/current_comp_shared_lazy_rotation_v1/tiny_worker.py](code/current_comp_shared_lazy_rotation_v1/tiny_worker.py)
- [code/current_comp_shared_lazy_rotation_v1/tiny_worker_v2.py](code/current_comp_shared_lazy_rotation_v1/tiny_worker_v2.py)
- [code/current_comp_shared_lazy_rotation_v1/watch_tiny.py](code/current_comp_shared_lazy_rotation_v1/watch_tiny.py)
- [code/current_comp_shared_lazy_rotation_v1/watch_tiny_v2.py](code/current_comp_shared_lazy_rotation_v1/watch_tiny_v2.py)
- [code/current_comp_shared_old384_v1/build_v1/estimator.py](code/current_comp_shared_old384_v1/build_v1/estimator.py)
- [code/current_comp_shared_old384_v1/fnp.py](code/current_comp_shared_old384_v1/fnp.py)
- [code/current_comp_shared_old384_v1/price_metadata.py](code/current_comp_shared_old384_v1/price_metadata.py)
- [code/current_comp_shared_old384_v1/scalar_semantics_probe.py](code/current_comp_shared_old384_v1/scalar_semantics_probe.py)
- [code/current_comp_shared_old384_v1/scalar_semantics_probe_v2.py](code/current_comp_shared_old384_v1/scalar_semantics_probe_v2.py)
- [code/current_comp_shared_old384_v1/scalar_semantics_probe_v3.py](code/current_comp_shared_old384_v1/scalar_semantics_probe_v3.py)
- [code/current_comp_shared_old384_v1/watch_scalar_semantics.py](code/current_comp_shared_old384_v1/watch_scalar_semantics.py)
- [code/current_comp_shared_old384_v1/watch_scalar_semantics_v2.py](code/current_comp_shared_old384_v1/watch_scalar_semantics_v2.py)
- [code/current_comp_shared_old384_v1/watch_scalar_semantics_v3.py](code/current_comp_shared_old384_v1/watch_scalar_semantics_v3.py)
- [code/current_comp_shared_pair256_v1/compose.py](code/current_comp_shared_pair256_v1/compose.py)
- [code/current_comp_shared_pair256_v1/metric_fnp.py](code/current_comp_shared_pair256_v1/metric_fnp.py)
- [code/current_comp_shared_pair256_v1/tiny_client.py](code/current_comp_shared_pair256_v1/tiny_client.py)
- [code/current_comp_shared_pair256_v1/tiny_worker.py](code/current_comp_shared_pair256_v1/tiny_worker.py)
- [code/current_comp_shared_pair256_v1/watch_tiny.py](code/current_comp_shared_pair256_v1/watch_tiny.py)
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_native_objective_p1_v1/build.py](code/current_d50_native_objective_p1_v1/build.py)
- [code/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py](code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py)
- [code/current_d50_native_objective_p1_v1/common.py](code/current_d50_native_objective_p1_v1/common.py)
- [code/current_d50_native_objective_p1_v1/native.py](code/current_d50_native_objective_p1_v1/native.py)
- [code/current_d50_native_objective_p1_v1/native_v2.py](code/current_d50_native_objective_p1_v1/native_v2.py)
- [code/current_d50_native_objective_p1_v1/watch.py](code/current_d50_native_objective_p1_v1/watch.py)
- [code/current_d50_native_objective_p1_v1/worker.py](code/current_d50_native_objective_p1_v1/worker.py)
- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor.py](code/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py](code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py)
- [code/current_p1_node_composition_v1/build.py](code/current_p1_node_composition_v1/build.py)
- [code/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/resident_gram512_v1/check_synthetic.py](code/resident_gram512_v1/check_synthetic.py)
- [code/resident_gram512_v1/gram.py](code/resident_gram512_v1/gram.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/terminal_dead_consumers_v1/estimator.py](code/terminal_dead_consumers_v1/estimator.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [STREAMED_SCRATCH_NEGATIVE_20260913_UTC.md#L8](STREAMED_SCRATCH_NEGATIVE_20260913_UTC.md#L8)

<details>
<summary>All result files and records (1)</summary>

- [STREAMED_SCRATCH_NEGATIVE_20260913_UTC.md](STREAMED_SCRATCH_NEGATIVE_20260913_UTC.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-streaming-scratch-preserves-output-bytes-but-increases-warm-product) · [Copy provenance](../../../PROVENANCE.json)


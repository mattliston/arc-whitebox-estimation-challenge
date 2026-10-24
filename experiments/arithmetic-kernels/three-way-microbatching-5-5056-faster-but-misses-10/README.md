# Three-way microbatching is 5.5056% faster but misses the 10% synthetic kernel timing gate

2026-09-13 · experiment · rejected · measured

The fixed nine-product geometry preserved output bytes and complete arithmetic but provided insufficient real-client speedup.

Recorded measurement:

    RESULT_20260912_PDT.md:
    # Fixed micro3 kernel: operational pass, advancement fail
    Worst of two warm core RPC wall times, same synthetic1024x384x8192 operands:
    Micro3/serial ratio0.944944097750027: 5.5056% faster, FAIL against the unchanged
    <=0.90 component usefulness gate. No full-card build, shape variant, retuning,
    Synthetic FP64-reference RMS error: D4 4.88619554130107e-6; serial/micro
    2.2146435445315837e-5. These are signed matrix-product errors, not final-layer
    at #330390 adjusted4.53886349950395e-9; failed #330840 remains failed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_push_microbatch3_v1/client.py](code/current_comp_push_microbatch3_v1/client.py)
- [code/current_comp_push_microbatch3_v1/kernel.py](code/current_comp_push_microbatch3_v1/kernel.py)
- [code/current_comp_push_microbatch3_v1/watch.py](code/current_comp_push_microbatch3_v1/watch.py)
- [code/current_comp_push_microbatch3_v1/worker.py](code/current_comp_push_microbatch3_v1/worker.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)

<details>
<summary>Additional programs and dependencies (89)</summary>

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
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/probe_ad_scalar.py](code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/qualification_v2.py](code/current_comp_receiver_birth_v1/qualification_v2.py)
- [code/current_comp_receiver_birth_v1/qualification_v3.py](code/current_comp_receiver_birth_v1/qualification_v3.py)
- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_shared_gain_slice_v1/build_v1/estimator.py](code/current_comp_shared_gain_slice_v1/build_v1/estimator.py)
- [code/current_comp_shared_old384_v1/build_v1/estimator.py](code/current_comp_shared_old384_v1/build_v1/estimator.py)
- [code/current_comp_shared_old384_v1/fnp.py](code/current_comp_shared_old384_v1/fnp.py)
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
- [code/current_d50_serial_d6_kernel_v1/client.py](code/current_d50_serial_d6_kernel_v1/client.py)
- [code/current_d50_serial_d6_kernel_v1/kernel.py](code/current_d50_serial_d6_kernel_v1/kernel.py)
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
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [RESULT_20260912_PDT.md#L3](RESULT_20260912_PDT.md#L3)
- [RESULT_20260912_PDT.md#L1](RESULT_20260912_PDT.md#L1)
- [RESULT_20260912_PDT.md#L11](RESULT_20260912_PDT.md#L11)
- [RESULT_20260912_PDT.md#L17](RESULT_20260912_PDT.md#L17)
- [RESULT_20260912_PDT.md#L18](RESULT_20260912_PDT.md#L18)

<details>
<summary>All result files and records (2)</summary>

- [RESULT_20260912_PDT.md](RESULT_20260912_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-three-way-microbatching-5-5056-faster-but-misses-10) · [Copy provenance](../../../PROVENANCE.json)


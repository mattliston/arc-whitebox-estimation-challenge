# Padded six-level factor propagation cuts cold cost to ratio 0.954146799147358 but takes 54.854506958s wall

2026-09-12 · experiment · inconclusive · measured

The fixed cold-and-repeat check established numerical fidelity and complete cost without an accuracy panel or hosted reliability claim.

Recorded measurement:

    RUNTIME_RESULT_20260912.md:
    Phase 2 only. This is a runtime/fidelity/complete-cost component, not a raw
    | Observation | Cold37 | Repeat37 |
    | Complete participant FLOPs | 440083035306 | 438128627824 |
    | Same-fixture APUSH complete FLOPs | 461231998786 | 459287028488 |
    | Own/control complete-cost ratio | 0.954146799147358 | 0.9539320743856958 |
    | Own scoring multiplier | 0.20012659447547776 | 0.19923783285048557 |
    | Actual NumPy-free wall seconds | 54.854506958 | 51.742047875 |
    | Actual residual seconds | 0.2703627230000052 | 0.2323816209999947 |
    | All16 maximum absolute drift vs APUSH | 3.4570693969726562e-6 | 3.4570693969726562e-6 |
    All16 fidelity stays below the fixed1e-5 gate. This does not assert
    2.72233356% below APUSH450955876627.575F; the measured37 saving is NOT the

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_padded_push_d6_v1/build.py](code/current_comp_padded_push_d6_v1/build.py)
- [code/current_comp_padded_push_d6_v1/client.py](code/current_comp_padded_push_d6_v1/client.py)
- [code/current_comp_padded_push_d6_v1/runner.py](code/current_comp_padded_push_d6_v1/runner.py)
- [code/current_comp_padded_push_d6_v1/watch.py](code/current_comp_padded_push_d6_v1/watch.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)

<details>
<summary>Additional programs and dependencies (142)</summary>

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
- [code/check_remote_series_v1.py](code/check_remote_series_v1.py)
- [code/current_bd_p_coadaptation_v1/build_terminal.py](code/current_bd_p_coadaptation_v1/build_terminal.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_comp_aligned_push_d6_v1/build.py](code/current_comp_aligned_push_d6_v1/build.py)
- [code/current_comp_aligned_push_d6_v1/build_v1/estimator.py](code/current_comp_aligned_push_d6_v1/build_v1/estimator.py)
- [code/current_comp_aligned_push_d6_v1/client.py](code/current_comp_aligned_push_d6_v1/client.py)
- [code/current_comp_aligned_push_d6_v1/native.py](code/current_comp_aligned_push_d6_v1/native.py)
- [code/current_comp_aligned_push_d6_v1/runner.py](code/current_comp_aligned_push_d6_v1/runner.py)
- [code/current_comp_aligned_push_d6_v1/score.py](code/current_comp_aligned_push_d6_v1/score.py)
- [code/current_comp_aligned_push_d6_v1/score_v2.py](code/current_comp_aligned_push_d6_v1/score_v2.py)
- [code/current_comp_aligned_push_d6_v1/watch.py](code/current_comp_aligned_push_d6_v1/watch.py)
- [code/current_comp_aligned_push_d6_v1/watch_native.py](code/current_comp_aligned_push_d6_v1/watch_native.py)
- [code/current_comp_aligned_push_d6_v1/watch_score.py](code/current_comp_aligned_push_d6_v1/watch_score.py)
- [code/current_comp_cstar_information_v1/information.py](code/current_comp_cstar_information_v1/information.py)
- [code/current_comp_cstar_information_v1/model.py](code/current_comp_cstar_information_v1/model.py)
- [code/current_comp_cstar_information_v1/qualification.py](code/current_comp_cstar_information_v1/qualification.py)
- [code/current_comp_cstar_information_v1/watch_qualification.py](code/current_comp_cstar_information_v1/watch_qualification.py)
- [code/current_comp_cstar_proposal_v1/common.py](code/current_comp_cstar_proposal_v1/common.py)
- [code/current_comp_cstar_proposal_v1/model.py](code/current_comp_cstar_proposal_v1/model.py)
- [code/current_comp_cstar_proposal_v1/watch.py](code/current_comp_cstar_proposal_v1/watch.py)
- [code/current_comp_cstar_proposal_v1/worker.py](code/current_comp_cstar_proposal_v1/worker.py)
- [code/current_comp_joint_p_covariance_v1/build.py](code/current_comp_joint_p_covariance_v1/build.py)
- [code/current_comp_joint_p_covariance_v1/fnp.py](code/current_comp_joint_p_covariance_v1/fnp.py)
- [code/current_comp_joint_p_covariance_v1/model.py](code/current_comp_joint_p_covariance_v1/model.py)
- [code/current_comp_joint_p_covariance_v1/qualification.py](code/current_comp_joint_p_covariance_v1/qualification.py)
- [code/current_comp_joint_p_covariance_v1/train.py](code/current_comp_joint_p_covariance_v1/train.py)
- [code/current_comp_joint_p_covariance_v1/watch_build.py](code/current_comp_joint_p_covariance_v1/watch_build.py)
- [code/current_comp_joint_p_covariance_v1/watch_qualification.py](code/current_comp_joint_p_covariance_v1/watch_qualification.py)
- [code/current_comp_joint_p_covariance_v1/watch_train.py](code/current_comp_joint_p_covariance_v1/watch_train.py)
- [code/current_comp_joint_p_node_v1/build.py](code/current_comp_joint_p_node_v1/build.py)
- [code/current_comp_joint_p_node_v1/model.py](code/current_comp_joint_p_node_v1/model.py)
- [code/current_comp_joint_p_node_v1/native.py](code/current_comp_joint_p_node_v1/native.py)
- [code/current_comp_joint_p_node_v1/qualification.py](code/current_comp_joint_p_node_v1/qualification.py)
- [code/current_comp_joint_p_node_v1/score.py](code/current_comp_joint_p_node_v1/score.py)
- [code/current_comp_joint_p_node_v1/train.py](code/current_comp_joint_p_node_v1/train.py)
- [code/current_comp_joint_p_node_v1/watch_native.py](code/current_comp_joint_p_node_v1/watch_native.py)
- [code/current_comp_joint_p_node_v1/watch_qualification.py](code/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/current_comp_joint_p_node_v1/watch_train.py](code/current_comp_joint_p_node_v1/watch_train.py)
- [code/current_comp_padded_push_d6_v1/build_v1/estimator.py](code/current_comp_padded_push_d6_v1/build_v1/estimator.py)
- [code/current_comp_partial_d6_v1/build.py](code/current_comp_partial_d6_v1/build.py)
- [code/current_comp_partial_d6_v1/build_v1/estimator.py](code/current_comp_partial_d6_v1/build_v1/estimator.py)
- [code/current_comp_partial_d6_v1/build_v1/null_estimator.py](code/current_comp_partial_d6_v1/build_v1/null_estimator.py)
- [code/current_comp_partial_d6_v1/native.py](code/current_comp_partial_d6_v1/native.py)
- [code/current_comp_partial_d6_v1/watch_build.py](code/current_comp_partial_d6_v1/watch_build.py)
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/probe_ad_scalar.py](code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/qualification_v2.py](code/current_comp_receiver_birth_v1/qualification_v2.py)
- [code/current_comp_receiver_birth_v1/qualification_v3.py](code/current_comp_receiver_birth_v1/qualification_v3.py)
- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_receiver_directions_v1/build.py](code/current_comp_receiver_directions_v1/build.py)
- [code/current_comp_receiver_directions_v1/build_v2.py](code/current_comp_receiver_directions_v1/build_v2.py)
- [code/current_comp_receiver_directions_v1/fnp.py](code/current_comp_receiver_directions_v1/fnp.py)
- [code/current_comp_receiver_directions_v1/model.py](code/current_comp_receiver_directions_v1/model.py)
- [code/current_comp_receiver_directions_v1/native.py](code/current_comp_receiver_directions_v1/native.py)
- [code/current_comp_receiver_directions_v1/qualification.py](code/current_comp_receiver_directions_v1/qualification.py)
- [code/current_comp_receiver_directions_v1/train.py](code/current_comp_receiver_directions_v1/train.py)
- [code/current_comp_receiver_directions_v1/watch_native.py](code/current_comp_receiver_directions_v1/watch_native.py)
- [code/current_comp_receiver_directions_v1/watch_qualification.py](code/current_comp_receiver_directions_v1/watch_qualification.py)
- [code/current_comp_receiver_directions_v1/watch_train.py](code/current_comp_receiver_directions_v1/watch_train.py)
- [code/current_d50_d6_cost_census_v1/run.py](code/current_d50_d6_cost_census_v1/run.py)
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
- [code/current_d50_response10_candidate_v1/build.py](code/current_d50_response10_candidate_v1/build.py)
- [code/current_d50_response10_candidate_v1/build_v1/estimator.py](code/current_d50_response10_candidate_v1/build_v1/estimator.py)
- [code/current_d50_response10_candidate_v1/build_v1/null_estimator.py](code/current_d50_response10_candidate_v1/build_v1/null_estimator.py)
- [code/current_d50_response10_candidate_v1/native.py](code/current_d50_response10_candidate_v1/native.py)
- [code/current_d50_response10_candidate_v1/score.py](code/current_d50_response10_candidate_v1/score.py)
- [code/current_d50_response10_candidate_v1/score_v2.py](code/current_d50_response10_candidate_v1/score_v2.py)
- [code/current_d50_response10_candidate_v1/watch_native.py](code/current_d50_response10_candidate_v1/watch_native.py)
- [code/current_d50_response10_census_v1/capture.py](code/current_d50_response10_census_v1/capture.py)
- [code/current_d50_response10_census_v1/watch.py](code/current_d50_response10_census_v1/watch.py)
- [code/current_d50_serial_d6_kernel_v1/client.py](code/current_d50_serial_d6_kernel_v1/client.py)
- [code/current_d50_serial_d6_kernel_v1/kernel.py](code/current_d50_serial_d6_kernel_v1/kernel.py)
- [code/current_d50_serial_d6_kernel_v1/run.py](code/current_d50_serial_d6_kernel_v1/run.py)
- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_p1_d6_push832_probe_v1/client.py](code/current_p1_d6_push832_probe_v1/client.py)
- [code/current_p1_d6_push832_probe_v1/run.py](code/current_p1_d6_push832_probe_v1/run.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor.py](code/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py](code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py)
- [code/current_p1_node_composition_v1/admit_v2.py](code/current_p1_node_composition_v1/admit_v2.py)
- [code/current_p1_node_composition_v1/build.py](code/current_p1_node_composition_v1/build.py)
- [code/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/current_p1_node_composition_v1/endpoint_v2.py](code/current_p1_node_composition_v1/endpoint_v2.py)
- [code/current_p1_node_composition_v1/native.py](code/current_p1_node_composition_v1/native.py)
- [code/current_p1_node_composition_v1/native_v2.py](code/current_p1_node_composition_v1/native_v2.py)
- [code/current_p1_node_composition_v1/score.py](code/current_p1_node_composition_v1/score.py)
- [code/current_p1_node_composition_v1/score_v2.py](code/current_p1_node_composition_v1/score_v2.py)
- [code/current_p1_node_composition_v1/watch_native.py](code/current_p1_node_composition_v1/watch_native.py)
- [code/current_p1_node_composition_v1/watch_native_v2.py](code/current_p1_node_composition_v1/watch_native_v2.py)
- [code/current_p1_partial_d6_v1/build.py](code/current_p1_partial_d6_v1/build.py)
- [code/current_p1_partial_d6_v1/build_v2.py](code/current_p1_partial_d6_v1/build_v2.py)
- [code/current_p1_partial_d6_v1/build_v2/estimator.py](code/current_p1_partial_d6_v1/build_v2/estimator.py)
- [code/current_p1_partial_d6_v1/native.py](code/current_p1_partial_d6_v1/native.py)
- [code/current_p1_partial_d6_v1/watch_native.py](code/current_p1_partial_d6_v1/watch_native.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/port_joint_p_covariance_working_v1.py](code/port_joint_p_covariance_working_v1.py)
- [code/port_receiver_directions_working_v1.py](code/port_receiver_directions_working_v1.py)
- [code/run_remote_series_v1.py](code/run_remote_series_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [RUNTIME_RESULT_20260912.md#L3](RUNTIME_RESULT_20260912.md#L3)
- [RUNTIME_RESULT_20260912.md#L4](RUNTIME_RESULT_20260912.md#L4)
- [RUNTIME_RESULT_20260912.md#L14](RUNTIME_RESULT_20260912.md#L14)
- [RUNTIME_RESULT_20260912.md#L16](RUNTIME_RESULT_20260912.md#L16)
- [RUNTIME_RESULT_20260912.md#L17](RUNTIME_RESULT_20260912.md#L17)

<details>
<summary>All result files and records (2)</summary>

- [RUNTIME_RESULT_20260912.md](RUNTIME_RESULT_20260912.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-12-padded-six-level-factor-propagation-cuts-cold-cost-ratio) · [Copy provenance](../../../PROVENANCE.json)


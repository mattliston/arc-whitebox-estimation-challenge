# Shared-rank384 approximate training model passes checkpointed output and gradient replay without native fidelity

2026-09-13 · tool · informational · measured

The repaired owned-input path preserved every served output, selector, variance record and mean-head gradient across the plain and checkpointed approximate model. This qualified a fixed anchored proposal workflow while leaving earlier native-fidelity and native-secant failures unchanged.

Recorded measurement:

    Baseline10.509986793s;checkpointed forward/backward30.524448583s;fresh repeat30.118070708s;all16outputs,47selectors,11gradient keys exact;peak7081279488B,zero added swap.

## Programs

- [code/current_shared384_anchored_p_proposal_v1/build_native.py](code/current_shared384_anchored_p_proposal_v1/build_native.py)
- [code/current_shared384_anchored_p_proposal_v1/proposal_admission.py](code/current_shared384_anchored_p_proposal_v1/proposal_admission.py)
- [code/current_shared384_anchored_p_proposal_v1/qualification.py](code/current_shared384_anchored_p_proposal_v1/qualification.py)
- [code/current_shared384_anchored_p_proposal_v1/run_qualification.py](code/current_shared384_anchored_p_proposal_v1/run_qualification.py)
- [code/current_shared384_anchored_p_proposal_v1/run_qualification_v2.py](code/current_shared384_anchored_p_proposal_v1/run_qualification_v2.py)
- [code/current_shared384_anchored_p_proposal_v1/run_training.py](code/current_shared384_anchored_p_proposal_v1/run_training.py)
- [code/current_shared384_anchored_p_proposal_v1/supervise.py](code/current_shared384_anchored_p_proposal_v1/supervise.py)
- [code/current_shared384_anchored_p_proposal_v1/supervise_v2.py](code/current_shared384_anchored_p_proposal_v1/supervise_v2.py)
- [code/current_shared384_anchored_p_proposal_v1/train_engine.py](code/current_shared384_anchored_p_proposal_v1/train_engine.py)
- [code/current_shared384_anchored_p_proposal_v1/watch_build_native.py](code/current_shared384_anchored_p_proposal_v1/watch_build_native.py)
- [code/current_shared384_anchored_p_proposal_v1/watch_training.py](code/current_shared384_anchored_p_proposal_v1/watch_training.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (76)</summary>

- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)
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
- [code/current_comp_joint_p_node_v1/build.py](code/current_comp_joint_p_node_v1/build.py)
- [code/current_comp_joint_p_node_v1/model.py](code/current_comp_joint_p_node_v1/model.py)
- [code/current_comp_joint_p_node_v1/qualification.py](code/current_comp_joint_p_node_v1/qualification.py)
- [code/current_comp_joint_p_node_v1/train.py](code/current_comp_joint_p_node_v1/train.py)
- [code/current_comp_joint_p_node_v1/watch_qualification.py](code/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/current_comp_joint_p_node_v1/watch_train.py](code/current_comp_joint_p_node_v1/watch_train.py)
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_shared_gain_slice_v1/build_v1/estimator.py](code/current_comp_shared_gain_slice_v1/build_v1/estimator.py)
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
- [code/current_native_trajectory_adjoint_v1/supervise.py](code/current_native_trajectory_adjoint_v1/supervise.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor.py](code/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/current_p1_node_composition_v1/build.py](code/current_p1_node_composition_v1/build.py)
- [code/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/current_shared384_causal_p_v1/differentiable_rangefinder.py](code/current_shared384_causal_p_v1/differentiable_rangefinder.py)
- [code/current_shared384_causal_p_v1/model.py](code/current_shared384_causal_p_v1/model.py)
- [code/current_shared384_causal_p_v1/native_rangefinder.py](code/current_shared384_causal_p_v1/native_rangefinder.py)
- [code/current_shared384_causal_p_v1/native_rangefinder_v2.py](code/current_shared384_causal_p_v1/native_rangefinder_v2.py)
- [code/current_shared384_causal_p_v1/physical_alignment.py](code/current_shared384_causal_p_v1/physical_alignment.py)
- [code/current_shared384_causal_p_v1/rangefinder_alignment.py](code/current_shared384_causal_p_v1/rangefinder_alignment.py)
- [code/current_shared384_causal_p_v1/shared_operator.py](code/current_shared384_causal_p_v1/shared_operator.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [EXECUTION_POSITIVE_20260913_UTC.md#L8](EXECUTION_POSITIVE_20260913_UTC.md#L8)

<details>
<summary>All result files and records (1)</summary>

- [EXECUTION_POSITIVE_20260913_UTC.md](EXECUTION_POSITIVE_20260913_UTC.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-13-shared-rank384-approximate-training-model-passes-checkpointed-output-gradient) · [Copy provenance](../../PROVENANCE.json)


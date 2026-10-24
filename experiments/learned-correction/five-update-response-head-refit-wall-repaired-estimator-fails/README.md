# Five-update response-head refit on the wall-repaired estimator fails: adjusted ratio 1.0195443375 on network 37

2026-09-19 · experiment · rejected · measured

The September 13 TRAIN20 fit of the learned response heads (P, the per-layer learned mean-response coefficients) had saved five updates but was never judged natively; it changes only 11 existing P literal fields of the I95 wall-repaired card (recovered_p5_I95.py). On held-out rank 37 with stored 1e9-sample means the recovered heads raise raw error and the adjusted objective, so the endpoint was rejected without further training.

Code availability: The exact candidate, recovery builder and native acquisition runner survive. The standalone rank-37 scoring command was not found as a saved Python program; its result JSON is preserved.

Recorded measurement:

    I95 raw 2.1635469517e-8, complete cold FLOPs 442570886172, adjusted 4.3543099841e-9; recovered P5 raw 2.2058320435e-8, same FLOPs, adjusted 4.4394120880e-9; adjusted ratio 1.0195443375

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/recovered_p5_I95.py](code/codex_resume_20260919_v1/recovered_p5_I95.py)
- [code/codex_resume_20260919_v1/build_recovered.py](code/codex_resume_20260919_v1/build_recovered.py)
- [code/codex_resume_20260919_v1/native_screen.py](code/codex_resume_20260919_v1/native_screen.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_zero_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/compare_ready.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/compare_ready.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)

<details>
<summary>Additional programs and dependencies (76)</summary>

- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/judge_terminal.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/judge_terminal.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/model.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/model.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_model.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_model.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_native_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_wide_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/train.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/train.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_terminal_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_terminal_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_zero_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_zero_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/core.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/model.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/model.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/build_terminal.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/build_terminal.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/model.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/qualify.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/qualify.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/train.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/train.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/build.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/model.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/qualification.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/qualification.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/train.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/train.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_qualification.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_train.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_train.py)
- [code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/model.py](code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/qualification.py](code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/qualification.py)
- [code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/watch_qualification.py](code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/astra_phase2_reexamination_v1/current_comp_shared_gain_slice_v1/build_v1/estimator.py](code/astra_phase2_reexamination_v1/current_comp_shared_gain_slice_v1/build_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/common.py](code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/common.py)
- [code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/native.py](code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/native.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/null_estimator.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/null_estimator.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/common.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/common.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native_v2.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/watch.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/watch.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/worker.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/worker.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/common.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/common.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native_validation.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native_validation.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/watch_fit.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/watch_fit.py)
- [code/astra_phase2_reexamination_v1/current_native_trajectory_adjoint_v1/supervise.py](code/astra_phase2_reexamination_v1/current_native_trajectory_adjoint_v1/supervise.py)
- [code/astra_phase2_reexamination_v1/current_p1_native_objective_p2_monitor_v2/monitor.py](code/astra_phase2_reexamination_v1/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build.py](code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/build_native.py](code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/build_native.py)
- [code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/proposal_admission.py](code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/proposal_admission.py)
- [code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/qualification.py](code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/qualification.py)
- [code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/run_training.py](code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/run_training.py)
- [code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/train_engine.py](code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/train_engine.py)
- [code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/watch_build_native.py](code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/watch_build_native.py)
- [code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/watch_training.py](code/astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/watch_training.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/differentiable_rangefinder.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/differentiable_rangefinder.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/model.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/native_rangefinder.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/native_rangefinder.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/native_rangefinder_v2.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/native_rangefinder_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/physical_alignment.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/physical_alignment.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/rangefinder_alignment.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/rangefinder_alignment.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py](code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py)

</details>

## Evidence

- [RECEIPTS.md#L17](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L17)
- [RECEIPTS.md#L31](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L31)
- [RECEIPTS.md#L39](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L39)

<details>
<summary>All result files and records (3)</summary>

- [recovered37_score.json](recovered37_score.json) · result
- [recovered_build.json](recovered_build.json) · config
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

## Related work

- measured by: [Five-update response-head refit on holdout networks 37/39/42/48: pooled adjusted ratio 1.0002074896](../five-update-response-head-refit-holdout-networks-37-39/README.md)

[Research graph](../../../graph/index.html#node=2026-09-19-five-update-response-head-refit-wall-repaired-estimator-fails) · [Copy provenance](../../../PROVENANCE.json)


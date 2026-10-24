# Scalar final readout distilled from the head's predictions: 14-net noise-corrected adjusted ratio .9870967012 (held)

2026-09-19 · experiment · inconclusive · measured

A final-only ridge fit on ranks 100/150/200 replaces the terminal head with a scalar read of the covariance and third-cumulant diagonals (65 readout coefficients; ten pair-summary inputs masked), fitted to the original head's predictions with no truth labels. It trades a small raw-error increase for a 2.6% lower bill: on the 8 public + 6 virgin paired panel it wins 11 of 14 nets but does not cross 4e-9, so it was held as a cost component rather than submitted.

Recorded measurement:

    scalar37 raw ratio 1.00225327, cold F ratio .97457166, adjusted ratio .97676763 vs I95; paired14 NC adjusted: public8 4.6220625306e-9 -> 4.5546569598e-9 (ratio .9854165602, 7/8), virgin6 3.5724345057e-9 -> 3.5366926056e-9 (.9899950860, 4/6), all14 4.1722219485e-9 -> 4.1183865223e-9 (.9870967012, 11/14); mean cold F 442.128B -> 430.439B

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/scalar_distill_I95.py](code/codex_resume_20260919_v1/scalar_distill_I95.py)
- [code/codex_resume_20260919_v1/distill_scalar_screen.py](code/codex_resume_20260919_v1/distill_scalar_screen.py)
- [code/codex_resume_20260919_v1/build_scalar_final.py](code/codex_resume_20260919_v1/build_scalar_final.py)
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

- [RECEIPTS.md#L44](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L44)
- [RECEIPTS.md#L53](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L53)
- [RECEIPTS.md#L68](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L68)

<details>
<summary>All result files and records (4)</summary>

- [scalar37_score.json](scalar37_score.json) · result
- [paired14_score.json](paired14_score.json) · result
- [scalar_distill_pilot.json](scalar_distill_pilot.json) · result
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

## Related work

- enables: [Exact-cost Winograd depth selection plus 1.5 MF per request improves adjusted error: ratio .957179078](../../arithmetic-kernels/exact-cost-winograd-depth-selection-plus-1-5-mf/README.md)

[Research graph](../../../graph/index.html#node=2026-09-19-scalar-final-readout-distilled-head-s-predictions-14-net) · [Copy provenance](../../../PROVENANCE.json)


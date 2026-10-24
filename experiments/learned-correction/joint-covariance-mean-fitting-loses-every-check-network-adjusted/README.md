# Joint covariance and mean fitting loses every check network: adjusted ratio 1.772556056561821

2026-09-10 · experiment · rejected · measured

An independently fitted covariance endpoint was compared with a matched mean-only control using the same four-network native scoring frame.

Recorded measurement:

    CHECK4_RESULT_20260909.md:
    scientific advancement gates failed. No fit extension, sweep, native40,
    CHECK ranks 37 (cold), 39, 42, 48 (warm); repeated 37 is excluded. Adjusted
    | Original COMP | 1.9933680058595472e-8 | 475126242316.25 | 4.2964338254612944e-9 |
    | Matched P-only control | 1.986607659226598e-8 | 475126242316.25 | 4.281556597677963e-9 |
    | Sole P+covariance terminal | 3.504779533236127e-8 | 475231339973.25 | 7.589299078726298e-9 |
    Terminal/P-only raw ratio 1.764203171652205, adjusted ratio
    1.772556056561821, complete-F ratio 1.0002211994363597, **0/4 wins**.
    Terminal/COMP adjusted ratio 1.7664182405768718, also 0/4 wins.
    adjusted ratio .9965373078260472 against COMP, same compute, 3/4 wins.
    | Qualification | 5 | 23.630925458 | 1659322368 | .355496792 |
    | Fixed training + CHECK forwards | 64 | 247.258133458 | 4269375488 | .378123292 |
    | Zero build | 0 | 6.864233834 | 352354304 | .325268417 |
    | Terminal build | 0 | 8.683585 | 347291648 | .332440625 |
    | Native zero | 3 | 91.916457958 | 6740967424 | .332640166 |
    | Native terminal | 5 | 152.590286291 | 7028686848 | .331850292 |
    | Saved-output scoring | 0 | 16.968877792 | 405536768 | .324592167 |

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_joint_p_covariance_v1/build.py](code/current_comp_joint_p_covariance_v1/build.py)
- [code/current_comp_joint_p_covariance_v1/fnp.py](code/current_comp_joint_p_covariance_v1/fnp.py)
- [code/current_comp_joint_p_covariance_v1/model.py](code/current_comp_joint_p_covariance_v1/model.py)
- [code/current_comp_joint_p_covariance_v1/native.py](code/current_comp_joint_p_covariance_v1/native.py)
- [code/current_comp_joint_p_covariance_v1/qualification.py](code/current_comp_joint_p_covariance_v1/qualification.py)
- [code/current_comp_joint_p_covariance_v1/score.py](code/current_comp_joint_p_covariance_v1/score.py)
- [code/current_comp_joint_p_covariance_v1/train.py](code/current_comp_joint_p_covariance_v1/train.py)
- [code/current_comp_joint_p_covariance_v1/watch_build.py](code/current_comp_joint_p_covariance_v1/watch_build.py)
- [code/current_comp_joint_p_covariance_v1/watch_native.py](code/current_comp_joint_p_covariance_v1/watch_native.py)
- [code/current_comp_joint_p_covariance_v1/watch_qualification.py](code/current_comp_joint_p_covariance_v1/watch_qualification.py)
- [code/current_comp_joint_p_covariance_v1/watch_score.py](code/current_comp_joint_p_covariance_v1/watch_score.py)
- [code/current_comp_joint_p_covariance_v1/watch_train.py](code/current_comp_joint_p_covariance_v1/watch_train.py)

<details>
<summary>Additional programs and dependencies (104)</summary>

- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
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
- [code/current_comp_joint_p_node_v1/native.py](code/current_comp_joint_p_node_v1/native.py)
- [code/current_comp_joint_p_node_v1/qualification.py](code/current_comp_joint_p_node_v1/qualification.py)
- [code/current_comp_joint_p_node_v1/score.py](code/current_comp_joint_p_node_v1/score.py)
- [code/current_comp_joint_p_node_v1/train.py](code/current_comp_joint_p_node_v1/train.py)
- [code/current_comp_joint_p_node_v1/watch_native.py](code/current_comp_joint_p_node_v1/watch_native.py)
- [code/current_comp_joint_p_node_v1/watch_qualification.py](code/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/current_comp_joint_p_node_v1/watch_score.py](code/current_comp_joint_p_node_v1/watch_score.py)
- [code/current_comp_joint_p_node_v1/watch_train.py](code/current_comp_joint_p_node_v1/watch_train.py)
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_receiver_directions_v1/build.py](code/current_comp_receiver_directions_v1/build.py)
- [code/current_comp_receiver_directions_v1/build_v2.py](code/current_comp_receiver_directions_v1/build_v2.py)
- [code/current_comp_receiver_directions_v1/fnp.py](code/current_comp_receiver_directions_v1/fnp.py)
- [code/current_comp_receiver_directions_v1/model.py](code/current_comp_receiver_directions_v1/model.py)
- [code/current_comp_receiver_directions_v1/native.py](code/current_comp_receiver_directions_v1/native.py)
- [code/current_comp_receiver_directions_v1/qualification.py](code/current_comp_receiver_directions_v1/qualification.py)
- [code/current_comp_receiver_directions_v1/score.py](code/current_comp_receiver_directions_v1/score.py)
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
- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor.py](code/current_p1_native_objective_p2_monitor_v2/monitor.py)
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
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/port_joint_p_covariance_working_v1.py](code/port_joint_p_covariance_working_v1.py)
- [code/port_receiver_directions_working_v1.py](code/port_receiver_directions_working_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [CHECK4_RESULT_20260909.md#L3](CHECK4_RESULT_20260909.md#L3)
- [CHECK4_RESULT_20260909.md#L12](CHECK4_RESULT_20260909.md#L12)
- [CHECK4_RESULT_20260909.md#L15](CHECK4_RESULT_20260909.md#L15)
- [CHECK4_RESULT_20260909.md#L22](CHECK4_RESULT_20260909.md#L22)
- [CHECK4_RESULT_20260909.md#L23](CHECK4_RESULT_20260909.md#L23)

<details>
<summary>All result files and records (1)</summary>

- [CHECK4_RESULT_20260909.md](CHECK4_RESULT_20260909.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-10-joint-covariance-mean-fitting-loses-every-check-network-adjusted) · [Copy provenance](../../../PROVENANCE.json)


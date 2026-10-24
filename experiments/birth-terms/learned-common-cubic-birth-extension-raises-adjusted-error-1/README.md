# Learned common-cubic birth extension raises adjusted error to 1.0288877816834323, with 0/4 wins

2026-09-10 · experiment · rejected · measured

A single 65-coefficient extension modified common cubic slots at births10-14 while the existing learned coefficients remained fixed.

Recorded measurement:

    CHECK4_RESULT_20260910.md:
    completed. Phase 2 only. Operational PASS; scientific gate FAIL. This arm is
    | Frozen current COMP | 1.9933680058595472e-8 | 475126242316.25 | 4.2964338254612944e-9 |
    | Fixed terminal CC | 1.9958315243117663e-8 | 488236228163.75 | 4.420548267828534e-9 |
    CC/COMP ratios: raw **1.0012358573253797**, complete compute
    **1.0275926368191926**, adjusted **1.0288877816834323**. Raw wins 2/4;
    adjusted wins **0/4**. Neither the primary <=0.95 adjusted / >=3 wins gate nor
    the smaller strict-gain / >=3 wins component gate passed. The nominal 4.421e-9
    | 37 | 4.51604278831124e-9 | 4.632823669716281e-9 | 1.025859117567996 |
    | 39 | 3.798008499495254e-9 | 3.929228143501814e-9 | 1.034549591983272 |
    | 42 | 4.6492506142600086e-9 | 4.793391393402326e-9 | 1.0310030134104222 |
    | 48 | 4.222433399778674e-9 | 4.326749864693714e-9 | 1.0247052955105242 |
    factor slots at births 10..14, did not earn its complete 2.7593% cost increase.
    It changed raw error by only +0.1236%. This is the fixed 65-parameter fit described
    target-blind CHECK forwards. Wall through finalization 141.755589042 s; peak
    #330390, adjusted 4.53886349950395e-9, raw 2.120838555441651e-8. The <1e-9 goal
    
    QUALIFICATION_RESULT_20260909.md:
    error2.0816681711721685e-17 versus1e-12 limit; MPS32 error
    
    TRAINING_AND_BUILD_RESULT_20260910.md:
    hyperparameter or gate changed after results. The sole Q65 fit completed
    raw5aed3ca9cea22e3b6ead5bc13eb61430234f941b6af86f8a9df3193f0f5dd6dc.
    | zero |3.991828959|334446592|.32799662499999993|211d59968a8014f4ca176355d4d9f6fee7bfb8603191cd695b9bef429c8b1a70|
    | terminal |5.559052416999999|312098816|.32666354200000036|7da49c1c67c99e4f002d2d52dcb5e973532bb9908ec183cf33dacee6b919bb58|

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_common_cubic_v1/build.py](code/current_comp_common_cubic_v1/build.py)
- [code/current_comp_common_cubic_v1/fnp.py](code/current_comp_common_cubic_v1/fnp.py)
- [code/current_comp_common_cubic_v1/model.py](code/current_comp_common_cubic_v1/model.py)
- [code/current_comp_common_cubic_v1/native.py](code/current_comp_common_cubic_v1/native.py)
- [code/current_comp_common_cubic_v1/native_source_preflat_review_v1.py](code/current_comp_common_cubic_v1/native_source_preflat_review_v1.py)
- [code/current_comp_common_cubic_v1/qualification.py](code/current_comp_common_cubic_v1/qualification.py)
- [code/current_comp_common_cubic_v1/qualification_source_pre_review_v1.py](code/current_comp_common_cubic_v1/qualification_source_pre_review_v1.py)
- [code/current_comp_common_cubic_v1/score.py](code/current_comp_common_cubic_v1/score.py)
- [code/current_comp_common_cubic_v1/score_source_preflight_draft_v1.py](code/current_comp_common_cubic_v1/score_source_preflight_draft_v1.py)
- [code/current_comp_common_cubic_v1/train.py](code/current_comp_common_cubic_v1/train.py)
- [code/current_comp_common_cubic_v1/watch_build.py](code/current_comp_common_cubic_v1/watch_build.py)
- [code/current_comp_common_cubic_v1/watch_native.py](code/current_comp_common_cubic_v1/watch_native.py)

<details>
<summary>Additional programs and dependencies (117)</summary>

- [code/current_comp_common_cubic_v1/watch_qualification.py](code/current_comp_common_cubic_v1/watch_qualification.py)
- [code/current_comp_common_cubic_v1/watch_score.py](code/current_comp_common_cubic_v1/watch_score.py)
- [code/current_comp_common_cubic_v1/watch_score_source_preflight_draft_v1.py](code/current_comp_common_cubic_v1/watch_score_source_preflight_draft_v1.py)
- [code/current_comp_common_cubic_v1/watch_train.py](code/current_comp_common_cubic_v1/watch_train.py)
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
- [code/current_comp_receiver_directions_v1/watch_score.py](code/current_comp_receiver_directions_v1/watch_score.py)
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

- [CHECK4_RESULT_20260910.md#L3](CHECK4_RESULT_20260910.md#L3)
- [CHECK4_RESULT_20260910.md#L4](CHECK4_RESULT_20260910.md#L4)
- [CHECK4_RESULT_20260910.md#L19](CHECK4_RESULT_20260910.md#L19)
- [CHECK4_RESULT_20260910.md#L20](CHECK4_RESULT_20260910.md#L20)
- [CHECK4_RESULT_20260910.md#L22](CHECK4_RESULT_20260910.md#L22)
- [QUALIFICATION_RESULT_20260909.md#L3](QUALIFICATION_RESULT_20260909.md#L3)
- [QUALIFICATION_RESULT_20260909.md#L12](QUALIFICATION_RESULT_20260909.md#L12)
- [TRAINING_AND_BUILD_RESULT_20260910.md#L3](TRAINING_AND_BUILD_RESULT_20260910.md#L3)
- [TRAINING_AND_BUILD_RESULT_20260910.md#L4](TRAINING_AND_BUILD_RESULT_20260910.md#L4)
- [TRAINING_AND_BUILD_RESULT_20260910.md#L17](TRAINING_AND_BUILD_RESULT_20260910.md#L17)
- [TRAINING_AND_BUILD_RESULT_20260910.md#L25](TRAINING_AND_BUILD_RESULT_20260910.md#L25)
- [TRAINING_AND_BUILD_RESULT_20260910.md#L26](TRAINING_AND_BUILD_RESULT_20260910.md#L26)

<details>
<summary>All result files and records (3)</summary>

- [CHECK4_RESULT_20260910.md](CHECK4_RESULT_20260910.md) · record
- [QUALIFICATION_RESULT_20260909.md](QUALIFICATION_RESULT_20260909.md) · record
- [TRAINING_AND_BUILD_RESULT_20260910.md](TRAINING_AND_BUILD_RESULT_20260910.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-10-learned-common-cubic-birth-extension-raises-adjusted-error-1) · [Copy provenance](../../../PROVENANCE.json)


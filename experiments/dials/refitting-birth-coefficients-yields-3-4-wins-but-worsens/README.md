# Refitting birth coefficients yields 3/4 wins but worsens mean adjusted score by 0.1128042679%

2026-09-10 · experiment · rejected · measured

Relative-headroom parameterization passed execution checks, but the sole fitted terminal failed its fixed mean-improvement criterion.

Recorded measurement:

    QUALIFICATION_RESULT_20260910.md:
    
    
    RESULT_CLOSED_20260910.md:
    The predeclared native CHECK4 gate required both strictly lower mean adjusted
    and at least3/4 individual adjusted wins. Actual result:3/4 wins, but the mean
    worsened. Scientific FAIL, operational PASS. No native40, independent14,
    COMP control's four score rows replayed exactly in FP64. Reported adjusted
    | Original COMP2e13 | 1.9933680058595472e-8 | 475126242316.25 | 4.2964338254612944e-9 |
    | Sole cstar terminald41a | 1.995711833068142e-8 | 475126242316.25 | 4.301280386185403e-9 |
    Terminal/control: raw1.0011758125954189, completeF1.0,
    adjusted1.0011280426793465 (+0.1128042679%). Individual adjusted ratios in
    1.0118669747153812. Neither the three wins nor a different aggregation
    rescues the failed mean gate. This is not an official score or sub3 lead.
    board poll. The separate CD component PASSED native40 adjusted4.3954880086e-9

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_cstar_proposal_v1/build.py](code/current_comp_cstar_proposal_v1/build.py)
- [code/current_comp_cstar_proposal_v1/common.py](code/current_comp_cstar_proposal_v1/common.py)
- [code/current_comp_cstar_proposal_v1/model.py](code/current_comp_cstar_proposal_v1/model.py)
- [code/current_comp_cstar_proposal_v1/native.py](code/current_comp_cstar_proposal_v1/native.py)
- [code/current_comp_cstar_proposal_v1/score.py](code/current_comp_cstar_proposal_v1/score.py)
- [code/current_comp_cstar_proposal_v1/watch.py](code/current_comp_cstar_proposal_v1/watch.py)
- [code/current_comp_cstar_proposal_v1/watch_build.py](code/current_comp_cstar_proposal_v1/watch_build.py)
- [code/current_comp_cstar_proposal_v1/watch_native.py](code/current_comp_cstar_proposal_v1/watch_native.py)
- [code/current_comp_cstar_proposal_v1/watch_score.py](code/current_comp_cstar_proposal_v1/watch_score.py)
- [code/current_comp_cstar_proposal_v1/worker.py](code/current_comp_cstar_proposal_v1/worker.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (105)</summary>

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
- [code/current_comp_cstar_information_v1/information.py](code/current_comp_cstar_information_v1/information.py)
- [code/current_comp_cstar_information_v1/model.py](code/current_comp_cstar_information_v1/model.py)
- [code/current_comp_cstar_information_v1/qualification.py](code/current_comp_cstar_information_v1/qualification.py)
- [code/current_comp_cstar_information_v1/watch_qualification.py](code/current_comp_cstar_information_v1/watch_qualification.py)
- [code/current_comp_cstar_proposal_v1/build_v1/estimator.py](code/current_comp_cstar_proposal_v1/build_v1/estimator.py)
- [code/current_comp_joint_p_node_v1/build.py](code/current_comp_joint_p_node_v1/build.py)
- [code/current_comp_joint_p_node_v1/model.py](code/current_comp_joint_p_node_v1/model.py)
- [code/current_comp_joint_p_node_v1/native.py](code/current_comp_joint_p_node_v1/native.py)
- [code/current_comp_joint_p_node_v1/qualification.py](code/current_comp_joint_p_node_v1/qualification.py)
- [code/current_comp_joint_p_node_v1/score.py](code/current_comp_joint_p_node_v1/score.py)
- [code/current_comp_joint_p_node_v1/train.py](code/current_comp_joint_p_node_v1/train.py)
- [code/current_comp_joint_p_node_v1/watch_native.py](code/current_comp_joint_p_node_v1/watch_native.py)
- [code/current_comp_joint_p_node_v1/watch_qualification.py](code/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/current_comp_joint_p_node_v1/watch_train.py](code/current_comp_joint_p_node_v1/watch_train.py)
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/probe_ad_scalar.py](code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/qualification_v2.py](code/current_comp_receiver_birth_v1/qualification_v2.py)
- [code/current_comp_receiver_birth_v1/qualification_v3.py](code/current_comp_receiver_birth_v1/qualification_v3.py)
- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_receiver_directions_v1/build.py](code/current_comp_receiver_directions_v1/build.py)
- [code/current_comp_receiver_directions_v1/fnp.py](code/current_comp_receiver_directions_v1/fnp.py)
- [code/current_comp_receiver_directions_v1/model.py](code/current_comp_receiver_directions_v1/model.py)
- [code/current_comp_receiver_directions_v1/qualification.py](code/current_comp_receiver_directions_v1/qualification.py)
- [code/current_comp_receiver_directions_v1/train.py](code/current_comp_receiver_directions_v1/train.py)
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
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/port_receiver_directions_working_v1.py](code/port_receiver_directions_working_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [QUALIFICATION_RESULT_20260910.md#L3](QUALIFICATION_RESULT_20260910.md#L3)
- [RESULT_CLOSED_20260910.md#L3](RESULT_CLOSED_20260910.md#L3)
- [RESULT_CLOSED_20260910.md#L18](RESULT_CLOSED_20260910.md#L18)
- [RESULT_CLOSED_20260910.md#L19](RESULT_CLOSED_20260910.md#L19)
- [RESULT_CLOSED_20260910.md#L20](RESULT_CLOSED_20260910.md#L20)
- [RESULT_CLOSED_20260910.md#L28](RESULT_CLOSED_20260910.md#L28)

<details>
<summary>All result files and records (3)</summary>

- [QUALIFICATION_RESULT_20260910.md](QUALIFICATION_RESULT_20260910.md) · record
- [RESULT_CLOSED_20260910.md](RESULT_CLOSED_20260910.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-10-refitting-birth-coefficients-yields-3-4-wins-but-worsens) · [Copy provenance](../../../PROVENANCE.json)


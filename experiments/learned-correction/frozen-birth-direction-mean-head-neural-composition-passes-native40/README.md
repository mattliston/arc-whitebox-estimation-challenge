# Frozen birth-direction and mean-head neural composition passes native40 and independent14 at ratio 0.9741291234422259

2026-09-09 · experiment · adopted · measured

Composition retained both earlier payloads without refitting and supplied fresh native output and cost evidence on the combined trajectory.

Recorded measurement:

    NATIVE40_SCORE_V3_RESULT.md:
    per-network FLOPs acquired before truth access. Repeat37 excluded.
    | Quantity, mean over40 | Retained P1 | Frozen NODE+P1 |
    | Final raw MSE |2.112546667623291e-8|2.092035131674627e-8|
    | Complete FLOPs |468321676751.725|468313646101.175|
    | Adjusted, mean per-net product |4.495952499840845e-9|4.451951498340901e-9|
    Adjusted ratio0.9902131969807285,26/40 per-net adjusted wins. Both mean
    The scorer exactly reproduced all40 archived P1 raw/bill/adjusted records.
    Prospective strict adjusted mean and>=24wins gates passed. Select one
    
    NATIVE_TERMINAL_V2_RESULT.md:
    mean improvement plus>=24wins admits the subsequent independent14 stage.
    
    PANEL_NATIVE_RESULT.md:
    repeat difference: 1,383,914,298 FLOPs, including NODE-A materialization.
    Require exact archived P1 replay and strict noise-corrected adjusted gain
    on BOTH public8 and virgin6. No raw, adjusted, retained-component,
    
    PANEL_SCORE_RESULT.md:
    Both predeclared strict group gates pass against the exact archived P1 baseline.
    | Independent 4M truth group | P1 noise-corrected adjusted | NODE/P1 noise-corrected adjusted | Candidate / P1 |
    | Public 8 | 4.896376273904718e-9 | 4.77415874629813e-9 | 0.9750391880097231 |
    | Virgin 6 | 3.839813677062407e-9 | 3.7345329733435845e-9 | 0.9725818196992919 |
    | Combined 14 | 4.4435637324008696e-9 | 4.328604843603325e-9 | 0.9741291234422259 |
    Combined mean noise-corrected raw MSE is 2.0305056229588224e-8 versus
    2.0847972531261027e-8 for P1. Complete mean candidate FLOPs are
    467697788399.9286 versus 467682934475.3571 for P1. Each reported adjusted
    mean is the mean of per-net raw times its own max(0.1, FLOPs/2^41), not a
    The native40 fixed-1e9-truth result remains 4.451951498340901e-9 adjusted,
    2.092035131674627e-8 raw, 468313646101.175 mean complete FLOPs.
    
    SOURCE_BUILD_RESULT.md:
    P1/NODE/cstar proofs passed. The old NODE pilot gate is required to remain
    
    ZERO_RESULT.md:
    pilot's failed3% gate remains failed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_p1_node_composition_v1/admission_memory_probe.py](code/current_p1_node_composition_v1/admission_memory_probe.py)
- [code/current_p1_node_composition_v1/admit_v2.py](code/current_p1_node_composition_v1/admit_v2.py)
- [code/current_p1_node_composition_v1/build.py](code/current_p1_node_composition_v1/build.py)
- [code/current_p1_node_composition_v1/endpoint_v2.py](code/current_p1_node_composition_v1/endpoint_v2.py)
- [code/current_p1_node_composition_v1/native.py](code/current_p1_node_composition_v1/native.py)
- [code/current_p1_node_composition_v1/native_v2.py](code/current_p1_node_composition_v1/native_v2.py)
- [code/current_p1_node_composition_v1/panel_admit.py](code/current_p1_node_composition_v1/panel_admit.py)
- [code/current_p1_node_composition_v1/panel_admit_draft_v1.py](code/current_p1_node_composition_v1/panel_admit_draft_v1.py)
- [code/current_p1_node_composition_v1/panel_common.py](code/current_p1_node_composition_v1/panel_common.py)
- [code/current_p1_node_composition_v1/panel_common_draft_v1.py](code/current_p1_node_composition_v1/panel_common_draft_v1.py)
- [code/current_p1_node_composition_v1/panel_native.py](code/current_p1_node_composition_v1/panel_native.py)
- [code/current_p1_node_composition_v1/panel_score.py](code/current_p1_node_composition_v1/panel_score.py)

<details>
<summary>Additional programs and dependencies (105)</summary>

- [code/current_p1_node_composition_v1/panel_watch.py](code/current_p1_node_composition_v1/panel_watch.py)
- [code/current_p1_node_composition_v1/panel_watch_draft_v1.py](code/current_p1_node_composition_v1/panel_watch_draft_v1.py)
- [code/current_p1_node_composition_v1/score.py](code/current_p1_node_composition_v1/score.py)
- [code/current_p1_node_composition_v1/score_v2.py](code/current_p1_node_composition_v1/score_v2.py)
- [code/current_p1_node_composition_v1/score_v3.py](code/current_p1_node_composition_v1/score_v3.py)
- [code/current_p1_node_composition_v1/watch_native.py](code/current_p1_node_composition_v1/watch_native.py)
- [code/current_p1_node_composition_v1/watch_native_v2.py](code/current_p1_node_composition_v1/watch_native_v2.py)
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
- [code/birth_direction_fit_v1/judge_terminal_v2.py](code/birth_direction_fit_v1/judge_terminal_v2.py)
- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)
- [code/birth_direction_fit_v1/native.py](code/birth_direction_fit_v1/native.py)
- [code/birth_direction_fit_v1/native_v2.py](code/birth_direction_fit_v1/native_v2.py)
- [code/birth_direction_fit_v1/panel.py](code/birth_direction_fit_v1/panel.py)
- [code/birth_direction_fit_v1/panel_v2.py](code/birth_direction_fit_v1/panel_v2.py)
- [code/birth_direction_fit_v1/qualify_model.py](code/birth_direction_fit_v1/qualify_model.py)
- [code/birth_direction_fit_v1/qualify_native_v2.py](code/birth_direction_fit_v1/qualify_native_v2.py)
- [code/birth_direction_fit_v1/qualify_wide_v2.py](code/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/birth_direction_fit_v1/train.py](code/birth_direction_fit_v1/train.py)
- [code/birth_direction_node_head_v1/build_terminal_v1/estimator.py](code/birth_direction_node_head_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_node_head_v1/build_zero_v1/estimator.py](code/birth_direction_node_head_v1/build_zero_v1/estimator.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/check_response_rows_panel_v1.py](code/check_response_rows_panel_v1.py)
- [code/current_bd_p_coadaptation_v1/build_terminal.py](code/current_bd_p_coadaptation_v1/build_terminal.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_d50_d6_cost_census_v1/run.py](code/current_d50_d6_cost_census_v1/run.py)
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_native_objective_p1_v1/build.py](code/current_d50_native_objective_p1_v1/build.py)
- [code/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py](code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py)
- [code/current_d50_native_objective_p1_v1/common.py](code/current_d50_native_objective_p1_v1/common.py)
- [code/current_d50_native_objective_p1_v1/judge.py](code/current_d50_native_objective_p1_v1/judge.py)
- [code/current_d50_native_objective_p1_v1/judge_v2.py](code/current_d50_native_objective_p1_v1/judge_v2.py)
- [code/current_d50_native_objective_p1_v1/native.py](code/current_d50_native_objective_p1_v1/native.py)
- [code/current_d50_native_objective_p1_v1/native_v2.py](code/current_d50_native_objective_p1_v1/native_v2.py)
- [code/current_d50_native_objective_p1_v1/native_watch.py](code/current_d50_native_objective_p1_v1/native_watch.py)
- [code/current_d50_native_objective_p1_v1/native_watch_v2.py](code/current_d50_native_objective_p1_v1/native_watch_v2.py)
- [code/current_d50_native_objective_p1_v1/panel.py](code/current_d50_native_objective_p1_v1/panel.py)
- [code/current_d50_native_objective_p1_v1/watch.py](code/current_d50_native_objective_p1_v1/watch.py)
- [code/current_d50_native_objective_p1_v1/watch_panel.py](code/current_d50_native_objective_p1_v1/watch_panel.py)
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
- [code/current_d50_terminal_linear_v2/judge_terminal.py](code/current_d50_terminal_linear_v2/judge_terminal.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/panel.py](code/current_d50_terminal_linear_v2/panel.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_d50_terminal_linear_v2/watch_panel.py](code/current_d50_terminal_linear_v2/watch_panel.py)
- [code/current_d50_terminal_linear_v2/watch_validation.py](code/current_d50_terminal_linear_v2/watch_validation.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor.py](code/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py](code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py)
- [code/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [NATIVE40_SCORE_V3_RESULT.md#L3](NATIVE40_SCORE_V3_RESULT.md#L3)
- [NATIVE40_SCORE_V3_RESULT.md#L5](NATIVE40_SCORE_V3_RESULT.md#L5)
- [NATIVE40_SCORE_V3_RESULT.md#L7](NATIVE40_SCORE_V3_RESULT.md#L7)
- [NATIVE40_SCORE_V3_RESULT.md#L9](NATIVE40_SCORE_V3_RESULT.md#L9)
- [NATIVE40_SCORE_V3_RESULT.md#L10](NATIVE40_SCORE_V3_RESULT.md#L10)
- [NATIVE_TERMINAL_V2_RESULT.md#L3](NATIVE_TERMINAL_V2_RESULT.md#L3)
- [NATIVE_TERMINAL_V2_RESULT.md#L36](NATIVE_TERMINAL_V2_RESULT.md#L36)
- [PANEL_NATIVE_RESULT.md#L3](PANEL_NATIVE_RESULT.md#L3)
- [PANEL_NATIVE_RESULT.md#L10](PANEL_NATIVE_RESULT.md#L10)
- [PANEL_NATIVE_RESULT.md#L36](PANEL_NATIVE_RESULT.md#L36)
- [PANEL_NATIVE_RESULT.md#L37](PANEL_NATIVE_RESULT.md#L37)
- [PANEL_SCORE_RESULT.md#L3](PANEL_SCORE_RESULT.md#L3)
- [PANEL_SCORE_RESULT.md#L6](PANEL_SCORE_RESULT.md#L6)
- [PANEL_SCORE_RESULT.md#L11](PANEL_SCORE_RESULT.md#L11)
- [PANEL_SCORE_RESULT.md#L13](PANEL_SCORE_RESULT.md#L13)
- [PANEL_SCORE_RESULT.md#L14](PANEL_SCORE_RESULT.md#L14)
- [SOURCE_BUILD_RESULT.md#L3](SOURCE_BUILD_RESULT.md#L3)
- [SOURCE_BUILD_RESULT.md#L5](SOURCE_BUILD_RESULT.md#L5)
- [ZERO_RESULT.md#L3](ZERO_RESULT.md#L3)
- [ZERO_RESULT.md#L35](ZERO_RESULT.md#L35)

<details>
<summary>All result files and records (7)</summary>

- [NATIVE40_SCORE_V3_RESULT.md](NATIVE40_SCORE_V3_RESULT.md) · record
- [NATIVE_TERMINAL_V2_RESULT.md](NATIVE_TERMINAL_V2_RESULT.md) · record
- [PANEL_NATIVE_RESULT.md](PANEL_NATIVE_RESULT.md) · record
- [PANEL_SCORE_RESULT.md](PANEL_SCORE_RESULT.md) · record
- [SOURCE_BUILD_RESULT.md](SOURCE_BUILD_RESULT.md) · record
- [ZERO_RESULT.md](ZERO_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-frozen-birth-direction-mean-head-neural-composition-passes-native40) · [Copy provenance](../../../PROVENANCE.json)


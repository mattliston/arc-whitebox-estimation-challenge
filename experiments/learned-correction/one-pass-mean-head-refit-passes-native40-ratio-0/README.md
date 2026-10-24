# One-pass mean-head refit passes native40 at ratio 0.9918044807401466 and independent14 at .991026496598678

2026-09-09 · experiment · adopted · measured

All response-head parameters were adapted once on the birth-direction estimator and scored with their own complete bills.

Recorded measurement:

    INDEPENDENT14_RESULT.md:
    The sole P1 terminal passed the predeclared strict BOTH-group NC-adjusted
    | Independent4M group | Current d50 NC adjusted | P1 NC adjusted | P1 / d50 |
    | public8 | 4.9147243456412085e-9 | 4.896376273904718e-9 | .9962667139708937 |
    | virgin6 | 3.909232150253503e-9 | 3.839813677062407e-9 | .9822424275348819 |
    | combined14 | 4.4837991190464774e-9 | 4.4435637324008696e-9 | .991026496598678 |
    Combined NC raw2.103192289821446e-8→2.0847972531261027e-8.
    Observed raw3.93228031332489e-8→3.9138852766295464e-8.
    Mean complete FLOPs467682934475.3571 for BOTH roles. Means of per-network
    
    NATIVE40_RESULT.md:
    The sole one-pass all-P payload passed its predeclared native40 gate:
    | Current d50 | 2.12989435468549e-8 | 468321676751.725 | 4.533103638013092e-9 |
    | P1 terminal | 2.112546667623291e-8 | 468321676751.725 | 4.495952499840845e-9 |
    Adjusted ratio0.9918044807401466;29/40 per-network wins. All40 primary
    both-group independent14 gate. No production qualification, submission,

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_d50_native_objective_p1_v1/build.py](code/current_d50_native_objective_p1_v1/build.py)
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

<details>
<summary>Additional programs and dependencies (70)</summary>

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
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py](code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py)
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
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
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

- [INDEPENDENT14_RESULT.md#L3](INDEPENDENT14_RESULT.md#L3)
- [INDEPENDENT14_RESULT.md#L5](INDEPENDENT14_RESULT.md#L5)
- [INDEPENDENT14_RESULT.md#L8](INDEPENDENT14_RESULT.md#L8)
- [INDEPENDENT14_RESULT.md#L10](INDEPENDENT14_RESULT.md#L10)
- [INDEPENDENT14_RESULT.md#L11](INDEPENDENT14_RESULT.md#L11)
- [NATIVE40_RESULT.md#L3](NATIVE40_RESULT.md#L3)
- [NATIVE40_RESULT.md#L5](NATIVE40_RESULT.md#L5)
- [NATIVE40_RESULT.md#L9](NATIVE40_RESULT.md#L9)
- [NATIVE40_RESULT.md#L10](NATIVE40_RESULT.md#L10)
- [NATIVE40_RESULT.md#L12](NATIVE40_RESULT.md#L12)

<details>
<summary>All result files and records (3)</summary>

- [INDEPENDENT14_RESULT.md](INDEPENDENT14_RESULT.md) · record
- [NATIVE40_RESULT.md](NATIVE40_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-one-pass-mean-head-refit-passes-native40-ratio-0) · [Copy provenance](../../../PROVENANCE.json)


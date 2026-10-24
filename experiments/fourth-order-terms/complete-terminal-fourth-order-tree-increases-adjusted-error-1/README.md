# Complete terminal fourth-order tree increases adjusted error to 1.0222155980099077 of the parent

2026-09-09 · experiment · rejected · measured

A fixed coefficient-one post-layer14 source and pre-layer15 readout failed both development networks after all numerical checks passed.

Recorded measurement:

    RESULT.md:
    | TRAIN net | d50 raw | Tree raw | Raw ratio | Own-adjusted ratio |
    |35|1.912339176377929e-8|1.9101621001483257e-8|.9988615637557942|1.0171549707058867|
    |100|1.851847332688951e-8|1.8683402038498447e-8|1.0089061721610417|1.0274644056646505|
    Mean adjusted4.030701439709112e-9 ->4.120245882591645e-9, ratio
    1.0222155980099077. Zero of two adjusted wins: the predeclared BOTH-net gate
    All five cold native calls passed: zero35/100 match every original raw and

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_full_fourth_tree_v1/build.py](code/terminal_full_fourth_tree_v1/build.py)
- [code/terminal_full_fourth_tree_v1/check_tiny.py](code/terminal_full_fourth_tree_v1/check_tiny.py)
- [code/terminal_full_fourth_tree_v1/judge.py](code/terminal_full_fourth_tree_v1/judge.py)
- [code/terminal_full_fourth_tree_v1/native.py](code/terminal_full_fourth_tree_v1/native.py)
- [code/terminal_full_fourth_tree_v1/watch.py](code/terminal_full_fourth_tree_v1/watch.py)
- [code/terminal_full_fourth_tree_v1/watch_judge.py](code/terminal_full_fourth_tree_v1/watch_judge.py)
- [code/terminal_full_fourth_tree_v1/watch_tiny.py](code/terminal_full_fourth_tree_v1/watch_tiny.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)

<details>
<summary>Additional programs and dependencies (41)</summary>

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
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_return_quantization_v1/analyze.py](code/current_d50_return_quantization_v1/analyze.py)
- [code/current_d50_return_quantization_v1/watch.py](code/current_d50_return_quantization_v1/watch.py)
- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/current_d50_terminal_linear_v2/judge_terminal.py](code/current_d50_terminal_linear_v2/judge_terminal.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/terminal_full_fourth_tree_v1/build_v1/estimator.py](code/terminal_full_fourth_tree_v1/build_v1/estimator.py)
- [code/terminal_full_fourth_tree_v1/build_v1/zero_estimator.py](code/terminal_full_fourth_tree_v1/build_v1/zero_estimator.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L11](RESULT.md#L11)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-complete-terminal-fourth-order-tree-increases-adjusted-error-1) · [Copy provenance](../../../PROVENANCE.json)


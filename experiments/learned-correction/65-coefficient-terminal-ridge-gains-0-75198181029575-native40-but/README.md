# 65-coefficient terminal ridge gains 0.75198181029575% on native40 but fails the independent public group

2026-09-09 · experiment · rejected · measured

A single TRAIN160 ridge solve changed only existing terminal coefficients, and the independent-truth public group worsened despite a small pooled gain.

Recorded measurement:

    INDEPENDENT14_RESULT.md:
    The sole frozen P65 candidate passed its native40 prospective gate: adjusted
    4.499015523213379e-9 versus d50 4.533103638013092e-9, 31/40 wins, unchanged
    | Independent 4M reference group | d50 NC adjusted | P65 NC adjusted | P65/d50 |
    | Public8 | 4.9147243456412085e-9 | 4.94069662320205e-9 | 1.0052845847974925 |
    | Virgin6 | 3.909232150253503e-9 | 3.87124165286998e-9 | 0.9902818518002162 |
    | Combined14 | 4.4837991190464774e-9 | 4.48235877877402e-9 | 0.9996787678853989 |
    Combined NC raw is 2.1029186173599835e-8 versus 2.103192289821446e-8;
    
    NATIVE40_RESULT.md:
    | Actual reused VALID40 | Current d50 | Sole terminal candidate |
    | Mean final observed MSE | 2.12989435468549e-8 | 2.113816435457673e-8 |
    | Mean complete FLOPs | 468321676751.725 | 468321676751.725 |
    | Mean own-adjusted score | 4.533103638013092e-9 | 4.499015523213379e-9 |
    Adjusted ratio .9924801818970425: a .75198181029575% improvement, with
    31/40 individual wins. Both prospective gates pass. Each score is the mean
    of its own per-network final MSE times max(.1, complete F/2^41), not a
    product of aggregate means. The references are the fixed observed 1e9-sample
    relative residual is1.9329680903046935e-15, condition43368.89219011511;
    65 existing terminal coefficients changed and no inference operations added.
    public8/virgin6 reference panel. NumPy-free production, residual/runtime/RSS,
    
    QUALIFICATION_RESULT.md:
    
    
    TRAIN_CAPTURE_RESULT.md:
    one-shot ridge fit and actual native40 own-adjusted judgment remain the next

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/judge_terminal.py](code/current_d50_terminal_linear_v2/judge_terminal.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/panel.py](code/current_d50_terminal_linear_v2/panel.py)
- [code/current_d50_terminal_linear_v2/watch_capture.py](code/current_d50_terminal_linear_v2/watch_capture.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_d50_terminal_linear_v2/watch_panel.py](code/current_d50_terminal_linear_v2/watch_panel.py)
- [code/current_d50_terminal_linear_v2/watch_validation.py](code/current_d50_terminal_linear_v2/watch_validation.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (57)</summary>

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
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_return_quantization_v1/analyze.py](code/current_d50_return_quantization_v1/analyze.py)
- [code/current_d50_return_quantization_v1/watch.py](code/current_d50_return_quantization_v1/watch.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
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
- [INDEPENDENT14_RESULT.md#L6](INDEPENDENT14_RESULT.md#L6)
- [INDEPENDENT14_RESULT.md#L7](INDEPENDENT14_RESULT.md#L7)
- [INDEPENDENT14_RESULT.md#L11](INDEPENDENT14_RESULT.md#L11)
- [INDEPENDENT14_RESULT.md#L13](INDEPENDENT14_RESULT.md#L13)
- [NATIVE40_RESULT.md#L3](NATIVE40_RESULT.md#L3)
- [NATIVE40_RESULT.md#L7](NATIVE40_RESULT.md#L7)
- [NATIVE40_RESULT.md#L9](NATIVE40_RESULT.md#L9)
- [NATIVE40_RESULT.md#L10](NATIVE40_RESULT.md#L10)
- [NATIVE40_RESULT.md#L11](NATIVE40_RESULT.md#L11)
- [QUALIFICATION_RESULT.md#L3](QUALIFICATION_RESULT.md#L3)
- [TRAIN_CAPTURE_RESULT.md#L3](TRAIN_CAPTURE_RESULT.md#L3)
- [TRAIN_CAPTURE_RESULT.md#L17](TRAIN_CAPTURE_RESULT.md#L17)

<details>
<summary>All result files and records (5)</summary>

- [INDEPENDENT14_RESULT.md](INDEPENDENT14_RESULT.md) · record
- [NATIVE40_RESULT.md](NATIVE40_RESULT.md) · record
- [QUALIFICATION_RESULT.md](QUALIFICATION_RESULT.md) · record
- [TRAIN_CAPTURE_RESULT.md](TRAIN_CAPTURE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but) · [Copy provenance](../../../PROVENANCE.json)


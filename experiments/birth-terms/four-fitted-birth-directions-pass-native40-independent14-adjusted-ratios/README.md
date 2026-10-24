# Four fitted birth directions pass native40 and independent14: adjusted ratios 0.9510628328263212 and .9395927214016986

2026-09-08 · experiment · adopted · measured

The single preset six-epoch terminal fit improved both independent-truth groups with frozen learned response heads.

Recorded measurement:

    INDEPENDENT14_RESULT_2026_09_08.md:
    | Group | Current409f NC raw | Terminal NC raw | Current409f NC adjusted | Terminal NC adjusted | Adjusted ratio |
    | public8 | 2.43238993902554e-8 | 2.29783886896747e-8 | 5.1858949910628045e-9 | 4.9147243456412085e-9 | .9477099621398192 |
    | virgin6 | 1.9971264872362867e-8 | 1.8436635176267465e-8 | 4.2202955997622905e-9 | 3.909232150253503e-9 | .9262934450548184 |
    | combined14 | 2.2458484596872894e-8 | 2.103192289821446e-8 | 4.772066680505441e-9 | 4.4837991190464774e-9 | .9395927214016986 |
    (observed final MSE - mean(mc_se15**2)) * max(.1, own complete FLOPs/2^41).
    The actual observed combined14 raw means are4.074936483190734e-8 and
    their own complete participant bills. Combined14 mean FLOPs:
    
    NATIVE40_RESULT_2026_09_08.md:
    | Current409f bank | 2.246383679135106e-8 | 4.766355577729645e-9 | 466876867515.125 |
    | Zero birth-direction action | 2.246383679135106e-8 | 4.781115190719337e-9 | 468321676751.725 |
    | Frozen terminal action | 2.12989435468549e-8 | 4.533103638013092e-9 | 468321676751.725 |
    Adjusted is the mean per-net raw MSE times max(0.1, that net's complete
    FLOPs/2^41), not the product of the two means. No independent4M noise
    Terminal/bank raw ratio0.9481436205526269, adjusted ratio0.9510628328263212:
    5.1856% raw and4.8937% adjusted reduction. Terminal/zero adjusted ratio
    0.9481268401172048. There are38/40 raw and adjusted wins against BOTH
    controls, passing the prospective24/40 and mean-adjusted-gain gates.
    terminal7.314GB, zero7.556GB. Native residual timing is diagnostic only;
    production0.4s residual qualification remains required before any bank.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/input_ownership_probe_v1.py](code/birth_direction_fit_v1/input_ownership_probe_v1.py)
- [code/birth_direction_fit_v1/judge_terminal.py](code/birth_direction_fit_v1/judge_terminal.py)
- [code/birth_direction_fit_v1/judge_terminal_v2.py](code/birth_direction_fit_v1/judge_terminal_v2.py)
- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)
- [code/birth_direction_fit_v1/native.py](code/birth_direction_fit_v1/native.py)
- [code/birth_direction_fit_v1/native_v2.py](code/birth_direction_fit_v1/native_v2.py)
- [code/birth_direction_fit_v1/panel.py](code/birth_direction_fit_v1/panel.py)
- [code/birth_direction_fit_v1/panel_v2.py](code/birth_direction_fit_v1/panel_v2.py)
- [code/birth_direction_fit_v1/prepare_fixture_data.py](code/birth_direction_fit_v1/prepare_fixture_data.py)
- [code/birth_direction_fit_v1/qualify_model.py](code/birth_direction_fit_v1/qualify_model.py)

<details>
<summary>Additional programs and dependencies (42)</summary>

- [code/birth_direction_fit_v1/qualify_native_v2.py](code/birth_direction_fit_v1/qualify_native_v2.py)
- [code/birth_direction_fit_v1/qualify_wide_v2.py](code/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/birth_direction_fit_v1/train.py](code/birth_direction_fit_v1/train.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)
- [code/birth_direction_fit_v1/fnp/core.py](code/birth_direction_fit_v1/fnp/core.py)
- [code/birth_direction_fit_v1/fnp/core_v2.py](code/birth_direction_fit_v1/fnp/core_v2.py)
- [code/birth_direction_fit_v1/fnp/reference_numpy.py](code/birth_direction_fit_v1/fnp/reference_numpy.py)
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

- [INDEPENDENT14_RESULT_2026_09_08.md#L3](INDEPENDENT14_RESULT_2026_09_08.md#L3)
- [INDEPENDENT14_RESULT_2026_09_08.md#L8](INDEPENDENT14_RESULT_2026_09_08.md#L8)
- [INDEPENDENT14_RESULT_2026_09_08.md#L10](INDEPENDENT14_RESULT_2026_09_08.md#L10)
- [INDEPENDENT14_RESULT_2026_09_08.md#L11](INDEPENDENT14_RESULT_2026_09_08.md#L11)
- [INDEPENDENT14_RESULT_2026_09_08.md#L12](INDEPENDENT14_RESULT_2026_09_08.md#L12)
- [NATIVE40_RESULT_2026_09_08.md#L3](NATIVE40_RESULT_2026_09_08.md#L3)
- [NATIVE40_RESULT_2026_09_08.md#L15](NATIVE40_RESULT_2026_09_08.md#L15)
- [NATIVE40_RESULT_2026_09_08.md#L16](NATIVE40_RESULT_2026_09_08.md#L16)
- [NATIVE40_RESULT_2026_09_08.md#L17](NATIVE40_RESULT_2026_09_08.md#L17)
- [NATIVE40_RESULT_2026_09_08.md#L19](NATIVE40_RESULT_2026_09_08.md#L19)

<details>
<summary>All result files and records (3)</summary>

- [INDEPENDENT14_RESULT_2026_09_08.md](INDEPENDENT14_RESULT_2026_09_08.md) · record
- [NATIVE40_RESULT_2026_09_08.md](NATIVE40_RESULT_2026_09_08.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios) · [Copy provenance](../../../PROVENANCE.json)


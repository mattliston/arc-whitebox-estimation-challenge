# Exact native initial Gram reproduces layer0 output bytes on both development networks

2026-09-09 · tool · informational · measured

Three fixed weights-only calls isolated the initial Gram implementation without qualifying the complete estimator response.

Recorded measurement:

    RESULT.md:
    the bank's complete inference cost. Paid context walls0.19748/0.16959/0.16418s.
     raw bytes6e9d9c7b7cf31b16e70eaddf32fdf93a7888c90a64a4d0833a7c4014caa28b72.
     raw bytes622f3b98a1f7092af81d14f458d8557b1912707e3175645a1d9022311e628a80.
    Bank330390 remains official4.53886349950395e-9 adjusted.

## Programs

- [code/current_d50_initial_gram_witness_v1/acquire.py](code/current_d50_initial_gram_witness_v1/acquire.py)
- [code/current_d50_initial_gram_witness_v1/watch.py](code/current_d50_initial_gram_witness_v1/watch.py)
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

<details>
<summary>Additional programs and dependencies (32)</summary>

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
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_bd_p_coadaptation_v1/watch.py](code/current_bd_p_coadaptation_v1/watch.py)
- [code/current_d50_hybrid_cp_qualification_v1/model.py](code/current_d50_hybrid_cp_qualification_v1/model.py)
- [code/current_d50_hybrid_cp_qualification_v1/qualify.py](code/current_d50_hybrid_cp_qualification_v1/qualify.py)
- [code/current_d50_hybrid_cp_qualification_v1/watch.py](code/current_d50_hybrid_cp_qualification_v1/watch.py)
- [code/current_d50_native_p_bridge_v1/build_probes.py](code/current_d50_native_p_bridge_v1/build_probes.py)
- [code/current_d50_native_p_bridge_v1/build_probes_v1/null/estimator.py](code/current_d50_native_p_bridge_v1/build_probes_v1/null/estimator.py)
- [code/current_d50_native_p_bridge_v1/native.py](code/current_d50_native_p_bridge_v1/native.py)
- [code/current_d50_native_p_bridge_v1/watch.py](code/current_d50_native_p_bridge_v1/watch.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L16](RESULT.md#L16)
- [RESULT.md#L29](RESULT.md#L29)
- [RESULT.md#L31](RESULT.md#L31)
- [RESULT.md#L37](RESULT.md#L37)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../graph/index.html#node=2026-09-09-exact-native-initial-gram-reproduces-layer0-output-bytes-both) · [Copy provenance](../../PROVENANCE.json)


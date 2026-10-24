# 4096-column autonomous fit passes cost readiness but diverges at a physical retirement boundary

2026-09-09 · finding · inconclusive · measured

The capped candidate reduced compute, yet native versus accelerator state diverged at network67 and final-epoch training error remained substantially higher.

Recorded measurement:

    READINESS_RESULT_2026_09_08.md:
    the actual matched two-net bank cost. Both bills pass the prospective360B
    fit-admission ceiling. Native peak RSS3.766GB and wall19.79–21.02s.
    The fixed unadapted native raw MSE is5.5411150298101864e-8 on35 and
    approximately2.1167e-8 raw would give3e-9 adjusted (planning approximation;
    
    TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md:
    FLOPs were exactly303943076245, cold305326989911, difference1383913666: all
    equal their own original complete bills. Native wall17.009/16.870s and process
    peak3.275GB passed the existing120s/8GB harness gates. These are instrumented
    hashed/read, and no raw or adjusted score was computed. The old1e-5 fidelity
    Final-epoch online raw means were5.4611131994342085e-8 for the capped P fit and
    2.1017176554671835e-8 for the current bank's birth-direction fit: ratio2.5984.
    The cap's own complete native40 mean cost is309716647338.85F, roughly34% below
    the current bank. That saving would not offset a2.60x raw increase if the
    bill would give about7.69e-9, but is explicitly NOT a measured adjusted score:
    raw will be near the approximately2.13e-8 needed for3e-9 at its cost.
    raw/adjusted closeout. The next efficiency information need is a CURRENT-d50
    complete tile/cost census for serial-topD6, before kernel-only savings are

## Programs

- [build.py](build.py)
- [build_terminal.py](build_terminal.py)
- [check_tiny.py](check_tiny.py)
- [check_wide.py](check_wide.py)
- [check_wide_v2.py](check_wide_v2.py)
- [check_wide_v3.py](check_wide_v3.py)
- [check_wide_v4.py](check_wide_v4.py)
- [common.py](common.py)
- [diagnose_terminal_rank67_v1.py](diagnose_terminal_rank67_v1.py)
- [estimator.py](estimator.py)
- [identity_certificate_v3.py](identity_certificate_v3.py)
- [judge_terminal.py](judge_terminal.py)

<details>
<summary>Additional programs and dependencies (40)</summary>

- [model.py](model.py)
- [model_v2.py](model_v2.py)
- [native.py](native.py)
- [native40.py](native40.py)
- [train.py](train.py)
- [transfer_certificate_v4.py](transfer_certificate_v4.py)
- [../../../submissions/330390/estimator.py](../../../submissions/330390/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/build_zero_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/compare_ready.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/compare_ready.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/judge_terminal.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/judge_terminal.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_model.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_native_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_wide_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_wide_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/train.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/train.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [estimator-604d23a2.py](estimator-604d23a2.py)
- [build-60da4a0c.py](build-60da4a0c.py)
- [core.py](core.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [READINESS_RESULT_2026_09_08.md#L3](READINESS_RESULT_2026_09_08.md#L3)
- [READINESS_RESULT_2026_09_08.md#L11](READINESS_RESULT_2026_09_08.md#L11)
- [READINESS_RESULT_2026_09_08.md#L12](READINESS_RESULT_2026_09_08.md#L12)
- [READINESS_RESULT_2026_09_08.md#L24](READINESS_RESULT_2026_09_08.md#L24)
- [READINESS_RESULT_2026_09_08.md#L27](READINESS_RESULT_2026_09_08.md#L27)
- [TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L3](TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L3)
- [TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L15](TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L15)
- [TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L16](TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L16)
- [TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L17](TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L17)
- [TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L40](TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md#L40)

<details>
<summary>All result files and records (4)</summary>

- [READINESS_RESULT_2026_09_08.md](READINESS_RESULT_2026_09_08.md) · record
- [TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md](TERMINAL_RANK67_DIAGNOSIS_RESULT_V1.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [build.json](build.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges) · [Copy provenance](../../../PROVENANCE.json)


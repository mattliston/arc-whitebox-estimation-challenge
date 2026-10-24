# Trained 4096-column cap fails native/accelerator fidelity on 7 of 40 networks before target scoring

2026-09-09 · incident · failed · measured

The fixed terminal candidate completed its native acquisition and resource checks but failed the unchanged all-layer fidelity gate before validation truth access. Different float32 operation orders before hard factor selection were a source-backed hypothesis, not a demonstrated cause at this point.

Recorded measurement:

    Mean complete40 cost309,716,647,338.85F;initialization1,383,913,666F;maximum41 wall38.1984921s;RSS4,191,535,104B;7/40all-layer differences exceed1e-5;rank67 maximum3.4943223e-5.

## Programs

- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/build.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/build.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/build_terminal.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/build_terminal.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_tiny.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_tiny.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide_v2.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide_v2.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide_v3.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide_v3.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide_v4.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/check_wide_v4.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/common.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/common.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/diagnose_terminal_rank67_v1.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/diagnose_terminal_rank67_v1.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/estimator.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/estimator.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/identity_certificate_v3.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/identity_certificate_v3.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/judge_terminal.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/judge_terminal.py)

<details>
<summary>Additional programs and dependencies (40)</summary>

- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/model.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/model.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/model_v2.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/model_v2.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/native.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/native.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/native40.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/native40.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/train.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/train.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/transfer_certificate_v4.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/transfer_certificate_v4.py)
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
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/estimator-604d23a2.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/estimator-604d23a2.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/build-60da4a0c.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/build-60da4a0c.py)
- [../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/core.py](../2026-09-09-4096-column-autonomous-fit-passes-cost-readiness-but-diverges/core.py)
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

- [TERMINAL_FIDELITY_FAILURE_2026_09_09.md#L3](TERMINAL_FIDELITY_FAILURE_2026_09_09.md#L3)

<details>
<summary>All result files and records (1)</summary>

- [TERMINAL_FIDELITY_FAILURE_2026_09_09.md](TERMINAL_FIDELITY_FAILURE_2026_09_09.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-trained-4096-column-cap-fails-native-accelerator-fidelity-7) · [Copy provenance](../../../PROVENANCE.json)


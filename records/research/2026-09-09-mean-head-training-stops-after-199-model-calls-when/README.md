# Mean-head training stops after 199 model calls when process inventory exceeds its 0.4-second timeout

2026-09-09 · incident · failed · measured

The independent monitor lost its required process observation and stopped training before any terminal payload or validation score. This was an operational observation failure, with no evidence of a numerical, variance or out-of-memory rejection.

Recorded measurement:

    199completed calls;979.950054750s;peak7,670,022,144B;zero swap growth;maximum successfully observed interval0.520809666s.

## Programs

- [../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/build-dce1494d.py](../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/build-dce1494d.py)
- [../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/common-f443ec27.py](../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/common-f443ec27.py)
- [native_common.py](native_common.py)
- [../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/watch-fe227690.py](../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/watch-fe227690.py)
- [../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/worker-41cdc599.py](../2026-09-09-mean-head-restart-completes-80-updates-but-only-13/worker-41cdc599.py)
- [../../../submissions/330390/estimator.py](../../../submissions/330390/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/build_zero_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/compare_ready.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/compare_ready.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small.py)

<details>
<summary>Additional programs and dependencies (48)</summary>

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
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/run.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/run.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/common.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/common.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/native.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/native.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/build-57f87081.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/build-57f87081.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/estimator-0deb2373.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/estimator-0deb2373.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/null_estimator.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/null_estimator.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/common.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/common.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/native-ebab249a.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/native-ebab249a.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/native_v2-fad59d76.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/native_v2-fad59d76.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/watch.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/watch.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/worker.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/worker.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/capture.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/capture.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/watch-7f868ba0.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/watch-7f868ba0.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/common.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/common.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit_v1/estimator.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native_validation.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native_validation.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_fit.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_fit.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [../../../experiments/wall-and-residual/exact-winograd-optimization/code/astra_phase2_reexamination_v1/packed_arena_generic.py](../../../experiments/wall-and-residual/exact-winograd-optimization/code/astra_phase2_reexamination_v1/packed_arena_generic.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [TRAINING_MONITOR_FAILURE.md#L3](TRAINING_MONITOR_FAILURE.md#L3)

<details>
<summary>All result files and records (1)</summary>

- [TRAINING_MONITOR_FAILURE.md](TRAINING_MONITOR_FAILURE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-mean-head-training-stops-after-199-model-calls-when) · [Copy provenance](../../../PROVENANCE.json)


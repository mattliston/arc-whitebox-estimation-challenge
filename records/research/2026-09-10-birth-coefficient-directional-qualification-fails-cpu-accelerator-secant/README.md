# Birth-coefficient directional qualification fails: CPU/accelerator secant disagreement .2999523038318002

2026-09-10 · incident · failed · measured

The original finite-step criterion failed before any target-reading information stage or fit.

Recorded measurement:

    QUALIFICATION_FAILURE_20260910.md:
    The fixed original CPU/MPS scalar finite-step gate failed on rank35,

## Programs

- [information.py](information.py)
- [model.py](model.py)
- [qualification.py](qualification.py)
- [watch_qualification.py](watch_qualification.py)
- [../../../submissions/330390/estimator.py](../../../submissions/330390/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/build_zero_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/compare_ready.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/compare_ready.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/check_small_v2.py)

<details>
<summary>Additional programs and dependencies (63)</summary>

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
- [../../../experiments/learned-correction/birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/code/birth_direction_node_head_v1/build.py](../../../experiments/learned-correction/birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/code/birth_direction_node_head_v1/build.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/estimator.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/estimator.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/estimator-1a99b49f.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/estimator-1a99b49f.py)
- [../../../experiments/learned-correction/birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/code/birth_direction_node_head_v1/core.py](../../../experiments/learned-correction/birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/code/birth_direction_node_head_v1/core.py)
- [../../../experiments/learned-correction/birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/code/birth_direction_node_head_v1/model.py](../../../experiments/learned-correction/birth-direction-neural-correction-improves-adjusted-error-2-093318322829274/code/birth_direction_node_head_v1/model.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py)
- [build.py](build.py)
- [model-cb49d7d5.py](model-cb49d7d5.py)
- [qualification-6d161c22.py](qualification-6d161c22.py)
- [train.py](train.py)
- [watch_qualification-096102b0.py](watch_qualification-096102b0.py)
- [watch_train.py](watch_train.py)
- [../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/model.py](../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/model.py)
- [../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/probe_ad_scalar.py](../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/qualification.py](../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/qualification.py)
- [../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v2.py](../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v2.py)
- [../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v3.py](../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v3.py)
- [../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/watch_qualification.py](../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/watch_qualification.py)
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
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/common.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/common.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit_v1/estimator.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native_validation.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native_validation.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_fit.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_fit.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/monitor.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/monitor.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/build.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/build.py)
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/terminal_estimator.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/terminal_estimator.py)
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

- [QUALIFICATION_FAILURE_20260910.md#L3](QUALIFICATION_FAILURE_20260910.md#L3)
- [QUALIFICATION_FAILURE_20260910.md#L8](QUALIFICATION_FAILURE_20260910.md#L8)

<details>
<summary>All result files and records (2)</summary>

- [QUALIFICATION_FAILURE_20260910.md](QUALIFICATION_FAILURE_20260910.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant) · [Copy provenance](../../../PROVENANCE.json)


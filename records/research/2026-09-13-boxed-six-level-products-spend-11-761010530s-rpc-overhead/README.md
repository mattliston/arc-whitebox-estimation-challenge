# Boxed six-level products spend 11.761010530s in RPC overhead within 51.040018208s total wall

2026-09-13 · finding · informational · measured

An unchanged cold prediction reconciled operation backend time, operation overhead, RPC overhead and participant residual exactly.

Recorded measurement:

    RESULT_20260912_PDT.md:
    Actual context51.040018208s. Native operation backend34.962000845s;
    native operation overhead3.622005700s; signed remaining RPC-compute gap
    0.470171730s; reported RPC overhead11.761010530s; participant residual
    0.224829403s. These sum to wall exactly at recorded precision. Native
    | subtract |10.702534650|14712|13684907853|
    | add |9.168836600|13737|16765395469|
    | matmul |7.658592069|653|346059966563|
    | stack |3.020096193|3590|7875370254|
    | multiply |2.566753210|4966|7233291747|
    | copyto |0.500006972|1807|1276720986|
    | astype |0.365096527|660|1354044808|
    | QR |0.177730168|10|4454350848|
    alone cannot account for the6.04s gap to the prior45s gate. No hidden serial-D6

## Programs

- [client.py](client.py)
- [watch.py](watch.py)
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

<details>
<summary>Additional programs and dependencies (84)</summary>

- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/judge_terminal.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/judge_terminal.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_model.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_native_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_wide_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_wide_v2.py)
- [estimator.py](estimator.py)
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
- [../../../experiments/arithmetic-kernels/float32-pair-polynomial-component-passes-independent14-4-83424098592912e-9/code/check_remote_series_v1.py](../../../experiments/arithmetic-kernels/float32-pair-polynomial-component-passes-independent14-4-83424098592912e-9/code/check_remote_series_v1.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/information.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/information.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/model.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/model.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/qualification.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/qualification.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/watch_qualification.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/watch_qualification.py)
- [../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/common.py](../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/common.py)
- [../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/model.py](../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/model.py)
- [../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/watch.py](../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/watch.py)
- [../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/worker.py](../../../experiments/arithmetic-kernels/broader-aligned-six-level-factor-propagation-improves-native40-adjusted/code/current_comp_cstar_proposal_v1/worker.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/build.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/build.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/model-cb49d7d5.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/model-cb49d7d5.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/qualification-6d161c22.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/qualification-6d161c22.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/train.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/train.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/watch_qualification-096102b0.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/watch_qualification-096102b0.py)
- [../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/watch_train.py](../2026-09-10-birth-coefficient-directional-qualification-fails-cpu-accelerator-secant/watch_train.py)
- [../../../experiments/learned-correction/fixed-temporal-observer-gains-1-95462-training-error-but/code/astra_phase2_reexamination_v1/current_comp_padded_push_d6_v1/client.py](../../../experiments/learned-correction/fixed-temporal-observer-gains-1-95462-training-error-but/code/astra_phase2_reexamination_v1/current_comp_padded_push_d6_v1/client.py)
- [../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier4_v1/kernel.py](../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier4_v1/kernel.py)
- [../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier_boxes_v1/compose.py](../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier_boxes_v1/compose.py)
- [../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier_boxes_v1/kernel.py](../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier_boxes_v1/kernel.py)
- [../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py](../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py)
- [watch_rpc37.py](watch_rpc37.py)
- [../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_microbatch3_v1/kernel.py](../../../experiments/third-order-planes/32-row-cubic-innovation-probe-has-tangent-variance-000780328087107739/code/astra_phase2_reexamination_v1/current_comp_push_microbatch3_v1/kernel.py)
- [compose.py](compose.py)
- [estimator-27f67658.py](estimator-27f67658.py)
- [watch_d4_only.py](watch_d4_only.py)
- [watch_rpc37-4c636a55.py](watch_rpc37-4c636a55.py)
- [../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/model.py](../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/model.py)
- [../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/probe_ad_scalar.py](../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/qualification.py](../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/qualification.py)
- [../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v2.py](../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v2.py)
- [../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v3.py](../../../experiments/birth-terms/balanced-receiver-direction-update-improves-raw-error-only-0/code/current_comp_receiver_birth_v1/qualification_v3.py)
- [../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/watch_qualification.py](../../../experiments/third-order-planes/64-endpoint-probes-explain-no-useful-gradient-energy-cross/code/current_comp_receiver_birth_v1/watch_qualification.py)
- [../../../submissions/330840/estimator.py](../../../submissions/330840/estimator.py)
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
- [../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/monitor_tests.py](../2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls/monitor_tests.py)
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

- [RESULT_20260912_PDT.md#L3](RESULT_20260912_PDT.md#L3)
- [RESULT_20260912_PDT.md#L9](RESULT_20260912_PDT.md#L9)
- [RESULT_20260912_PDT.md#L10](RESULT_20260912_PDT.md#L10)
- [RESULT_20260912_PDT.md#L11](RESULT_20260912_PDT.md#L11)
- [RESULT_20260912_PDT.md#L12](RESULT_20260912_PDT.md#L12)

<details>
<summary>All result files and records (2)</summary>

- [RESULT_20260912_PDT.md](RESULT_20260912_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-boxed-six-level-products-spend-11-761010530s-rpc-overhead) · [Copy provenance](../../../PROVENANCE.json)


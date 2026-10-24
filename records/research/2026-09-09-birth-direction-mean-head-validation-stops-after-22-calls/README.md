# Birth-direction and mean-head validation stops after 22 calls at 8020918272 bytes against an 8000000000-byte cap

2026-09-09 · incident · failed · measured

The original terminal native acquisition exceeded its whole-process memory guard while beginning the next network. No validation truth was opened; extra supervisor memory was identified as a diagnostic lead without assuming a repaired estimator would pass.

Recorded measurement:

    22completed primary calls;RSS8020918272B against8000000000B;worker7795539968B,supervisor223166464B;961.994564625s;zero added swap.

## Programs

- [admission_memory_probe.py](admission_memory_probe.py)
- [admit_v2.py](admit_v2.py)
- [build.py](build.py)
- [endpoint_v2.py](endpoint_v2.py)
- [native.py](native.py)
- [native_v2.py](native_v2.py)
- [panel_admit.py](panel_admit.py)
- [panel_admit_draft_v1.py](panel_admit_draft_v1.py)
- [panel_common.py](panel_common.py)
- [panel_common_draft_v1.py](panel_common_draft_v1.py)
- [panel_native.py](panel_native.py)
- [panel_score.py](panel_score.py)

<details>
<summary>Additional programs and dependencies (105)</summary>

- [panel_watch.py](panel_watch.py)
- [panel_watch_draft_v1.py](panel_watch_draft_v1.py)
- [score.py](score.py)
- [score_v2.py](score_v2.py)
- [score_v3.py](score_v3.py)
- [watch_native.py](watch_native.py)
- [watch_native_v2.py](watch_native_v2.py)
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
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/judge_terminal_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/judge_terminal_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/panel.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/panel.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/panel_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/panel_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_model.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_native_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_wide_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/qualify_wide_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/train.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/train.py)
- [estimator.py](estimator.py)
- [estimator-1a99b49f.py](estimator-1a99b49f.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py)
- [../../../submissions/329851/estimator.py](../../../submissions/329851/estimator.py)
- [../../../submissions/329926/estimator.py](../../../submissions/329926/estimator.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py](../../../experiments/flop-accounting/response-row-matched-round-closed-declared-gate/code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py)
- [../../../experiments/arithmetic-kernels/float32-pair-polynomial-component-passes-independent14-4-83424098592912e-9/code/check_response_rows_panel_v1.py](../../../experiments/arithmetic-kernels/float32-pair-polynomial-component-passes-independent14-4-83424098592912e-9/code/check_response_rows_panel_v1.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/build_terminal.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py)
- [run.py](run.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/common.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/common.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/native.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_full_return_capture_v1/native.py)
- [build-57f87081.py](build-57f87081.py)
- [estimator-0deb2373.py](estimator-0deb2373.py)
- [null_estimator.py](null_estimator.py)
- [common.py](common.py)
- [judge.py](judge.py)
- [judge_v2.py](judge_v2.py)
- [native-ebab249a.py](native-ebab249a.py)
- [native_v2-fad59d76.py](native_v2-fad59d76.py)
- [native_watch.py](native_watch.py)
- [native_watch_v2.py](native_watch_v2.py)
- [panel.py](panel.py)
- [watch.py](watch.py)
- [watch_panel.py](watch_panel.py)
- [worker.py](worker.py)
- [build-61c9ff6a.py](build-61c9ff6a.py)
- [estimator-9d32027a.py](estimator-9d32027a.py)
- [null_estimator-0deb2373.py](null_estimator-0deb2373.py)
- [native-cccd8b50.py](native-cccd8b50.py)
- [score-3bc2aeb4.py](score-3bc2aeb4.py)
- [score_v2-e4a9ef56.py](score_v2-e4a9ef56.py)
- [watch_native-68f3371b.py](watch_native-68f3371b.py)
- [capture.py](capture.py)
- [watch-7f868ba0.py](watch-7f868ba0.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/common.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/common.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit_v1/estimator.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/judge_terminal.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/judge_terminal.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native_validation.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/native_validation.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/panel.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/panel.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_fit.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_fit.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_panel.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_panel.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_validation.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_terminal_linear_v2/watch_validation.py)
- [monitor.py](monitor.py)
- [monitor_tests.py](monitor_tests.py)
- [terminal_estimator.py](terminal_estimator.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/exact-winograd-optimization/code/astra_phase2_reexamination_v1/packed_arena_generic.py](../../../experiments/wall-and-residual/exact-winograd-optimization/code/astra_phase2_reexamination_v1/packed_arena_generic.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [NATIVE40_MEMORY_FAILURE.md#L3](NATIVE40_MEMORY_FAILURE.md#L3)

<details>
<summary>All result files and records (1)</summary>

- [NATIVE40_MEMORY_FAILURE.md](NATIVE40_MEMORY_FAILURE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-birth-direction-mean-head-validation-stops-after-22-calls) · [Copy provenance](../../../PROVENANCE.json)


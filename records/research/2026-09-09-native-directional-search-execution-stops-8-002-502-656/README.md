# Native directional-search execution stops at 8,002,502,656-byte memory guard before fitting

2026-09-09 · incident · failed · measured

CPU function checks passed, but the attempted native acquisition exceeded its resource bound; the scientific proposal remained untested in this attempt.

Recorded measurement:

    RESULT.md:
    raw2.120838555441651e-8. No new score, fitting, check-label access or board
    peak8,789,786,624B, below its11GB gate. No MPS or targets.
    gate at8,002,502,656B. Wall19.6941828327s and complete468,515,344,292F were
    |35 cold|471955793128|6250872832|
    |36|474178309330|6878150656|
    |38|461185316515|6945275904|
    |40|435265791019|6957121536|
    |41|456347351486|7163232256|
    |43|454218445035|7456079872|
    |44|486434011516|7866843136|
    |45|460274838877|7933968384|
    |100|468515344292|8002502656|
    rank100 failed resource admission before its post-resource observation gate;

## Programs

- [../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/common.py](../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/common.py)
- [../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/cpu.py](../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/cpu.py)
- [../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/interpolate.py](../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/interpolate.py)
- [../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/judge.py](../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/judge.py)
- [../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/native.py](../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/native.py)
- [../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/watch.py](../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_native_directional_search_v1/watch.py)
- [../../../submissions/330390/estimator.py](../../../submissions/330390/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/train.py)
- [../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/watch.py](../2026-09-09-birth-direction-mean-head-coadaptation-stops-numerical-gate-secant/watch.py)
- [../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_cpu_pilot_v1/worker.py](../../../experiments/learned-correction/native-directional-search-gains-0-8624937-fit-networks-but/code/current_d50_cpu_pilot_v1/worker.py)
- [../2026-09-09-exact-initial-gram-injection-still-fails-directional-response-discrepancy/model-c8cb5cfe.py](../2026-09-09-exact-initial-gram-injection-still-fails-directional-response-discrepancy/model-c8cb5cfe.py)
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

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L13](RESULT.md#L13)
- [RESULT.md#L17](RESULT.md#L17)
- [RESULT.md#L26](RESULT.md#L26)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-native-directional-search-execution-stops-8-002-502-656) · [Copy provenance](../../../PROVENANCE.json)


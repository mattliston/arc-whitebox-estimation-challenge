# First-head response discrepancy is overwhelmingly output rounding; conclusion does not extend to all layers

2026-09-09 · finding · informational · measured

Forty exact endpoint reconstructions isolated return rounding on the native-initial-Gram CPU trajectory and explicitly limited the causal claim to layer1.

Recorded measurement:

    RESULT.md:
    raw2.120838555441651e-8. No new score or submission.
    |35|.025|.000342698868|.000373578936|.999626421064|
    |35|.05|.000709490273|.000756660329|.999243339671|
    |100|.025|.000493315515|.000208578605|.999791421395|
    |100|.05|.000971049682|.002173446011|.997826553989|
    ratios are below the fixed.25 local-dominance threshold. Vector/scalar
    
    RESULT_SCOPE_ADDENDUM.md:
    

## Programs

- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_return_quantization_v1/analyze.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_return_quantization_v1/analyze.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_return_quantization_v1/watch.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_d50_return_quantization_v1/watch.py)
- [../../../submissions/330390/estimator.py](../../../submissions/330390/estimator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/model.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/model.py)
- [../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py](../../../experiments/learned-correction/65-coefficient-terminal-ridge-gains-0-75198181029575-native40-but/code/current_bd_p_coadaptation_v1/qualify.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L18](RESULT.md#L18)
- [RESULT.md#L19](RESULT.md#L19)
- [RESULT.md#L20](RESULT.md#L20)
- [RESULT_SCOPE_ADDENDUM.md#L3](RESULT_SCOPE_ADDENDUM.md#L3)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · record
- [RESULT_SCOPE_ADDENDUM.md](RESULT_SCOPE_ADDENDUM.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-first-head-response-discrepancy-overwhelmingly-output-rounding-conclusion-does) · [Copy provenance](../../../PROVENANCE.json)


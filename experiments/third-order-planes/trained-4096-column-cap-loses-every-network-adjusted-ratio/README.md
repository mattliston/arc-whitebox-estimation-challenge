# Trained 4096-column cap loses every network: adjusted ratio 1.617820646864215 versus uncapped reference

2026-09-19 · experiment · rejected · measured

The preserved capped candidate was finally scored against the uncapped reference and the current delayed-rotation estimator using its actual saved bills.

Recorded measurement:

    # Frozen cap4096: measured objective failure
    | Original 40, uncapped old bank | 4.766355577729645e-9 | 7.711108463947432e-9 | 1.617820646864215 | 0/40 |
    | Original 40, untrained cap | 8.26458112816147e-9 | 7.711108463947432e-9 | .9330307663956391 | 37/40 |
    | Common 8, current Lazy | 3.8588427268781516e-9 | 8.052305190410887e-9 | 2.0867150491321773 | 0/8 |
    Training improved the cap's own adjusted error by 6.70%, but the resulting
    Original-40 terminal raw MSE is 5.47484085348415e-8 at mean measured sequence
    cost 309716647338.85 FLOPs. All scores are means of per-network raw MSE times
    max(.1, that call's own FLOPs/2^41), not products of unrelated means.
    fixed candidate. These are reused ordinary1B research references; no claim
    sealed before reference access. The old native/MPS <=1e-5 fidelity gate still
    fails on 7/40 networks, maximum 3.4943222999572754e-5. Actual shadow/native final
    MSE ratios range .9984798265621299 to 1.002038062621613. Thus numerical drift is
    The serial guarded scorer completed in 3.7016 seconds, with peak RSS
    69779456 bytes, zero added swap and the process group retired. Existing model,
    The strongest qualified local model remains Lazy at noise-corrected14
    AIcrowd #331599; the latest poll is still submitted/queued with no score.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/cap4096_objective_v1/score.py](code/codex_resume_20260919_v1/cap4096_objective_v1/score.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_zero_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/compare_ready.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/compare_ready.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)

<details>
<summary>Additional programs and dependencies (32)</summary>

- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/judge_terminal.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/judge_terminal.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/model.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/model.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_model.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_model.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_native_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_wide_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/train.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/train.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/build.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/build_terminal.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/build_terminal.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/build_terminal_v1/estimator.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/build_terminal_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/common.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/common.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/estimator.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/judge_terminal.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/judge_terminal.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/model.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/model_v2.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/model_v2.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/native40.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/native40.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/train.py](code/astra_phase2_reexamination_v1/current_paircap4096_e2e_v1/train.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_v1/build.py](code/astra_phase2_reexamination_v1/current_paircap4096_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_paircap4096_v1/core.py](code/astra_phase2_reexamination_v1/current_paircap4096_v1/core.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L1](NEGATIVE_RESULT.md#L1)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)
- [REEXAMINATION_INDEX_20260919.md#L1333](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1333)
- [REEXAMINATION_INDEX_20260919.md#L1336](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1336)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-trained-4096-column-cap-loses-every-network-adjusted-ratio) · [Copy provenance](../../../PROVENANCE.json)


# Trained 4096-column cap has native40 adjusted ratio 1.7010659979808653 and 0/40 wins

2026-09-09 · experiment · rejected · measured

The preserved terminal predictions were scored against current birth-direction and untrained capped controls using their actual saved costs.

Recorded measurement:

    RESULT.md:
    | Current d50 |2.12989435468549e-8|468321676751.725|4.533103638013092e-9|
    | CAP4096 null |5.8676205281886746e-8|309736222733.85|8.26458112816147e-9|
    | CAP4096 terminal |5.47484085348415e-8|309716647338.85|7.711108463947432e-9|
    Terminal/d50 adjusted ratio1.7010659979808653, raw ratio2.5704753108717404,
    0/40 adjusted wins. Six-epoch adaptation improved the compressed null by
    6.6969% adjusted with37/40 wins, but did not approach the current bank.
    the newly reached per-net MSE1% diagnostic passes. Neither changes the

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/strategic_reset_20260909_v1/cap_native_objective_v1/score.py](code/strategic_reset_20260909_v1/cap_native_objective_v1/score.py)
- [code/strategic_reset_20260909_v1/cap_native_objective_v1/score_v2.py](code/strategic_reset_20260909_v1/cap_native_objective_v1/score_v2.py)
- [code/strategic_reset_20260909_v1/cap_native_objective_v1/watch.py](code/strategic_reset_20260909_v1/cap_native_objective_v1/watch.py)
- [code/strategic_reset_20260909_v1/cap_native_objective_v1/watch_v2.py](code/strategic_reset_20260909_v1/cap_native_objective_v1/watch_v2.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)

<details>
<summary>Additional programs and dependencies (35)</summary>

- [code/birth_direction_fit_v1/fnp/core.py](code/birth_direction_fit_v1/fnp/core.py)
- [code/birth_direction_fit_v1/fnp/core_v2.py](code/birth_direction_fit_v1/fnp/core_v2.py)
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
- [code/current_paircap4096_e2e_v1/build.py](code/current_paircap4096_e2e_v1/build.py)
- [code/current_paircap4096_e2e_v1/build_terminal.py](code/current_paircap4096_e2e_v1/build_terminal.py)
- [code/current_paircap4096_e2e_v1/build_terminal_v1/estimator.py](code/current_paircap4096_e2e_v1/build_terminal_v1/estimator.py)
- [code/current_paircap4096_e2e_v1/common.py](code/current_paircap4096_e2e_v1/common.py)
- [code/current_paircap4096_e2e_v1/estimator.py](code/current_paircap4096_e2e_v1/estimator.py)
- [code/current_paircap4096_e2e_v1/judge_terminal.py](code/current_paircap4096_e2e_v1/judge_terminal.py)
- [code/current_paircap4096_e2e_v1/model.py](code/current_paircap4096_e2e_v1/model.py)
- [code/current_paircap4096_e2e_v1/model_v2.py](code/current_paircap4096_e2e_v1/model_v2.py)
- [code/current_paircap4096_e2e_v1/native40.py](code/current_paircap4096_e2e_v1/native40.py)
- [code/current_paircap4096_e2e_v1/train.py](code/current_paircap4096_e2e_v1/train.py)
- [code/current_paircap4096_v1/build.py](code/current_paircap4096_v1/build.py)
- [code/current_paircap4096_v1/core.py](code/current_paircap4096_v1/core.py)
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
- [RESULT.md#L13](RESULT.md#L13)
- [RESULT.md#L14](RESULT.md#L14)
- [RESULT.md#L15](RESULT.md#L15)
- [RESULT.md#L17](RESULT.md#L17)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-trained-4096-column-cap-has-native40-adjusted-ratio-1) · [Copy provenance](../../../PROVENANCE.json)


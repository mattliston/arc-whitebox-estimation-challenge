# Birth-direction neural correction improves adjusted error 2.093318322829274% but misses its 3% gate

2026-09-09 · experiment · rejected · measured

A fixed 20-training and four-validation-network pilot trained the birth-node head for six epochs and thirty updates.

Recorded measurement:

    FIDELITY_RESULT_2026_09_09.md:
    - Added paid head work: 14,853,888 FLOPs per warm call, plus 512 FLOPs cold parameter setup. This is measured prototype overhead, not a future fitted total bill.
    - Maximum native/shadow mean difference: 2.86102294921875e-6, below the prospective 1e-5 gate.
    - CPU/MPS fixed-cotangent A-gradient relative errors: rank35 0.0012942266263578383; rank100 0.0002691941419957581, below the 0.05 gate. Non-negligible CPU norms: 2.057291652745952e-5 and 1.9324722656663348e-5.
    - Existing 1e9-truth final-MSE ratios: 0.9995490635595055–1.0004896947443824 across fixed prototypes and both shadows, within the prospective 1% gate. These truths serve numerical fidelity only, not an improvement verdict.
    
    TERMINAL_PILOT_RESULT_2026_09_09.md:
    Frame: fixed reused-development VALID37/39/42/48, observed existing1e9 `mc_post` final-layer MSE multiplied by each candidate's own complete native scoring multiplier. Parent is exact d50; its original first-four/cold37 prefix is reused after verifying all41 archived records. New zero and terminal roles each ran all four plus exact repeat with their own input/lifecycle/bills. No estimated-cost substitution or repeat averaging.
    | Parent d50 |2.026176015404889e-8|4.367945309309118e-9|475111388300.25|
    | Zero head |2.026176015404889e-8|4.3680821740128575e-9|475126242316.25|
    | Terminal head |1.984349169123722e-8|4.276510309818188e-9|475126242316.25|
    Terminal adjusted improves2.093318322829274% against parent and2.096386023583985% against zero. Three of the same four nets beat BOTH controls; rank37 is slightly worse. The prospective mean-gain threshold was3% against each control, so **pilot_gate_passed=false**. The score itself completed validly: passed/completed/integrity/numerical_fidelity true.
    All16 zero outputs are byte-identical to parent. Terminal native/MPS maximum differences2.3842e-6–2.6226e-6 are below1e-5; final-MSE ratios0.9992619094580636–1.0003822472662278 are within1%. All four predictions/bills and terminal A were sealed before validation truths. No production or independent14 claim is made.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/birth_direction_node_head_v1/build.py](code/birth_direction_node_head_v1/build.py)
- [code/birth_direction_node_head_v1/build_terminal.py](code/birth_direction_node_head_v1/build_terminal.py)
- [code/birth_direction_node_head_v1/check_tiny.py](code/birth_direction_node_head_v1/check_tiny.py)
- [code/birth_direction_node_head_v1/check_wide.py](code/birth_direction_node_head_v1/check_wide.py)
- [code/birth_direction_node_head_v1/core.py](code/birth_direction_node_head_v1/core.py)
- [code/birth_direction_node_head_v1/judge_terminal.py](code/birth_direction_node_head_v1/judge_terminal.py)
- [code/birth_direction_node_head_v1/judge_terminal_v2.py](code/birth_direction_node_head_v1/judge_terminal_v2.py)
- [code/birth_direction_node_head_v1/model.py](code/birth_direction_node_head_v1/model.py)
- [code/birth_direction_node_head_v1/native_validation.py](code/birth_direction_node_head_v1/native_validation.py)
- [code/birth_direction_node_head_v1/native_wide.py](code/birth_direction_node_head_v1/native_wide.py)
- [code/birth_direction_node_head_v1/train.py](code/birth_direction_node_head_v1/train.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (41)</summary>

- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)
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
- [code/birth_direction_node_head_v1/build_nonzero_v1/estimator.py](code/birth_direction_node_head_v1/build_nonzero_v1/estimator.py)
- [code/birth_direction_node_head_v1/build_terminal_v1/estimator.py](code/birth_direction_node_head_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_node_head_v1/build_zero_v1/estimator.py](code/birth_direction_node_head_v1/build_zero_v1/estimator.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/current_paircap4096_e2e_v1/build.py](code/current_paircap4096_e2e_v1/build.py)
- [code/current_paircap4096_e2e_v1/common.py](code/current_paircap4096_e2e_v1/common.py)
- [code/current_paircap4096_e2e_v1/estimator.py](code/current_paircap4096_e2e_v1/estimator.py)
- [code/current_paircap4096_e2e_v1/model.py](code/current_paircap4096_e2e_v1/model.py)
- [code/current_paircap4096_e2e_v1/model_v2.py](code/current_paircap4096_e2e_v1/model_v2.py)
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

- [FIDELITY_RESULT_2026_09_09.md#L3](FIDELITY_RESULT_2026_09_09.md#L3)
- [FIDELITY_RESULT_2026_09_09.md#L10](FIDELITY_RESULT_2026_09_09.md#L10)
- [FIDELITY_RESULT_2026_09_09.md#L11](FIDELITY_RESULT_2026_09_09.md#L11)
- [FIDELITY_RESULT_2026_09_09.md#L12](FIDELITY_RESULT_2026_09_09.md#L12)
- [FIDELITY_RESULT_2026_09_09.md#L13](FIDELITY_RESULT_2026_09_09.md#L13)
- [TERMINAL_PILOT_RESULT_2026_09_09.md#L3](TERMINAL_PILOT_RESULT_2026_09_09.md#L3)
- [TERMINAL_PILOT_RESULT_2026_09_09.md#L9](TERMINAL_PILOT_RESULT_2026_09_09.md#L9)
- [TERMINAL_PILOT_RESULT_2026_09_09.md#L13](TERMINAL_PILOT_RESULT_2026_09_09.md#L13)
- [TERMINAL_PILOT_RESULT_2026_09_09.md#L14](TERMINAL_PILOT_RESULT_2026_09_09.md#L14)
- [TERMINAL_PILOT_RESULT_2026_09_09.md#L15](TERMINAL_PILOT_RESULT_2026_09_09.md#L15)

<details>
<summary>All result files and records (3)</summary>

- [FIDELITY_RESULT_2026_09_09.md](FIDELITY_RESULT_2026_09_09.md) · record
- [TERMINAL_PILOT_RESULT_2026_09_09.md](TERMINAL_PILOT_RESULT_2026_09_09.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-birth-direction-neural-correction-improves-adjusted-error-2-093318322829274) · [Copy provenance](../../../PROVENANCE.json)


# First-head covariance attribution fails two of four fixed vector/scalar response cells

2026-09-09 · experiment · rejected · measured

Native covariance boundary substitutions explained some directional discrepancy but did not satisfy the complete fixed local-response gate.

Recorded measurement:

    RESULT.md:
    native-N minus native-L on the same native C0/W1. Gate: both norm(D-E)/norm(D)
    layer1 cotangent contribution; it is not final-layer MSE.
    | 35 | .025 | .4974263464 | .6208350235 | FAIL |
    | 35 | .05 | .4228644706 | .0061162675 | PASS |
    | 100 | .025 | .4924508346 | .1394479381 | PASS |
    | 100 | .05 | .4927801134 | .9738723037 | FAIL |
    pass the vector gate in any case (remaining ratios .9422,.9759,1.0410,.9756).
    Actual FNP diagnostic-context FLOPs: N cold13,399,308,660; each remaining

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_d50_covariance_boundary_v1/cpu.py](code/current_d50_covariance_boundary_v1/cpu.py)
- [code/current_d50_covariance_boundary_v1/judge.py](code/current_d50_covariance_boundary_v1/judge.py)
- [code/current_d50_covariance_boundary_v1/native.py](code/current_d50_covariance_boundary_v1/native.py)
- [code/current_d50_covariance_boundary_v1/watch.py](code/current_d50_covariance_boundary_v1/watch.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)

<details>
<summary>Additional programs and dependencies (40)</summary>

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
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_bd_p_coadaptation_v1/watch.py](code/current_bd_p_coadaptation_v1/watch.py)
- [code/current_d50_hybrid_cp_qualification_v1/model.py](code/current_d50_hybrid_cp_qualification_v1/model.py)
- [code/current_d50_hybrid_cp_qualification_v1/qualify.py](code/current_d50_hybrid_cp_qualification_v1/qualify.py)
- [code/current_d50_hybrid_cp_qualification_v1/watch.py](code/current_d50_hybrid_cp_qualification_v1/watch.py)
- [code/current_d50_initial_gram_witness_v1/acquire.py](code/current_d50_initial_gram_witness_v1/acquire.py)
- [code/current_d50_initial_gram_witness_v1/watch.py](code/current_d50_initial_gram_witness_v1/watch.py)
- [code/current_d50_native_g0_forward_v1/model.py](code/current_d50_native_g0_forward_v1/model.py)
- [code/current_d50_native_g0_forward_v1/qualify.py](code/current_d50_native_g0_forward_v1/qualify.py)
- [code/current_d50_native_g0_forward_v1/watch.py](code/current_d50_native_g0_forward_v1/watch.py)
- [code/current_d50_native_p_bridge_v1/build_probes.py](code/current_d50_native_p_bridge_v1/build_probes.py)
- [code/current_d50_native_p_bridge_v1/build_probes_v1/null/estimator.py](code/current_d50_native_p_bridge_v1/build_probes_v1/null/estimator.py)
- [code/current_d50_native_p_bridge_v1/judge.py](code/current_d50_native_p_bridge_v1/judge.py)
- [code/current_d50_native_p_bridge_v1/native.py](code/current_d50_native_p_bridge_v1/native.py)
- [code/current_d50_native_p_bridge_v1/watch.py](code/current_d50_native_p_bridge_v1/watch.py)
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
- [RESULT.md#L21](RESULT.md#L21)
- [RESULT.md#L24](RESULT.md#L24)
- [RESULT.md#L28](RESULT.md#L28)
- [RESULT.md#L29](RESULT.md#L29)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-first-head-covariance-attribution-fails-two-four-fixed-vector) · [Copy provenance](../../../PROVENANCE.json)


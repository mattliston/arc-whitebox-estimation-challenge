# Response-row floor 0.1 reduces work but multiplies adjusted error by 4.8666953591

2026-09-09 · experiment · rejected · measured

The fixed threshold change retained the fitted response payload and was rejected on actual final-layer error times its own complete cost.

Recorded measurement:

    RESULT.md:
    | d50 |1.88209325453344e-8|470235568710|4.024874313854929e-9|
    | P1 |1.8762623409467065e-8|470235568710|4.01240370696628e-9|
    | response.1 with frozen P1 |1.0356745998995807e-7|415869748378|1.958783714424208e-8|
    Candidate/d50 raw ratio5.5027804675, adjusted ratio4.8666953591;
    individual adjusted ratios4.3306885504/5.4242758251, zero wins.
    It misses the predeclared5% mean adjusted gain and neither-net-worse gate.
    exact repeat and source-owned startup. Actual startup1383913786 FLOPs
    before reference access. Raw computation uses float64 differences to

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_d50_response10_candidate_v1/build.py](code/current_d50_response10_candidate_v1/build.py)
- [code/current_d50_response10_candidate_v1/native.py](code/current_d50_response10_candidate_v1/native.py)
- [code/current_d50_response10_candidate_v1/score.py](code/current_d50_response10_candidate_v1/score.py)
- [code/current_d50_response10_candidate_v1/score_v2.py](code/current_d50_response10_candidate_v1/score_v2.py)
- [code/current_d50_response10_candidate_v1/watch_native.py](code/current_d50_response10_candidate_v1/watch_native.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/core.py](code/birth_direction_fit_v1/fnp/core.py)
- [code/birth_direction_fit_v1/fnp/core_v2.py](code/birth_direction_fit_v1/fnp/core_v2.py)
- [code/birth_direction_fit_v1/fnp/reference_numpy.py](code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

- [code/birth_direction_fit_v1/native.py](code/birth_direction_fit_v1/native.py)
- [code/birth_direction_fit_v1/native_v2.py](code/birth_direction_fit_v1/native_v2.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_d50_d6_cost_census_v1/run.py](code/current_d50_d6_cost_census_v1/run.py)
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/current_d50_response10_candidate_v1/build_v1/estimator.py](code/current_d50_response10_candidate_v1/build_v1/estimator.py)
- [code/current_d50_response10_candidate_v1/build_v1/null_estimator.py](code/current_d50_response10_candidate_v1/build_v1/null_estimator.py)
- [code/current_d50_response10_census_v1/capture.py](code/current_d50_response10_census_v1/capture.py)
- [code/current_d50_response10_census_v1/watch.py](code/current_d50_response10_census_v1/watch.py)
- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L13](RESULT.md#L13)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-09-response-row-floor-0-1-reduces-work-but-multiplies) · [Copy provenance](../../../PROVENANCE.json)


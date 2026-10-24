# Float32 pair-polynomial component passes independent14: 4.83424098592912e-9 to 4.788089831892398e-9

2026-09-07 · experiment · adopted · measured

Only the three-channel pair-polynomial arithmetic moved to float32, returning to float64 before scaling and centering. Every network saved the same work, and the small development improvement survived independent-truth and persistent-client checks.

Recorded measurement:

    Combined14 adjusted4.83424098592912e-9 to4.788089831892398e-9,gain0.9546722%;every network4497408000F saved;public8 gain0.9565366%;virgin6 gain0.9516179%;maximum served drift3.6954879760742188e-6.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_pair_polynomial_f32_panel_v1.py](code/check_pair_polynomial_f32_panel_v1.py)
- [code/qualify_pair_polynomial_f32_remote_v1.py](code/qualify_pair_polynomial_f32_remote_v1.py)
- [code/prepare_pair_polynomial_f32_remote_launch_v1.py](code/prepare_pair_polynomial_f32_remote_launch_v1.py)
- [code/candidate_response_pair_polynomial_f32_v1.py](code/candidate_response_pair_polynomial_f32_v1.py)
- [code/build_pair_polynomial_f32_candidate_v1.py](code/build_pair_polynomial_f32_candidate_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)

<details>
<summary>Additional programs and dependencies (15)</summary>

- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_pair_polynomial_f32_dev_v1.py](code/check_pair_polynomial_f32_dev_v1.py)
- [code/check_remote_series_v1.py](code/check_remote_series_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/check_response_rows_panel_v1.py](code/check_response_rows_panel_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/pair_polynomial_f32_source_v1.py](code/pair_polynomial_f32_source_v1.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)
- [code/run_remote_series_v1.py](code/run_remote_series_v1.py)

</details>

## Evidence

- [RETAINED_COMPONENT_PAIR_POLYNOMIAL_F32_V1.md#L15](RETAINED_COMPONENT_PAIR_POLYNOMIAL_F32_V1.md#L15)

<details>
<summary>All result files and records (1)</summary>

- [RETAINED_COMPONENT_PAIR_POLYNOMIAL_F32_V1.md](RETAINED_COMPONENT_PAIR_POLYNOMIAL_F32_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-float32-pair-polynomial-component-passes-independent14-4-83424098592912e-9) · [Copy provenance](../../../PROVENANCE.json)


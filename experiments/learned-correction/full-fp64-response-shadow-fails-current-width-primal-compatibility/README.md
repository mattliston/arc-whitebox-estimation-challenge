# Full-FP64 response shadow fails current-width primal compatibility at the correlation-row RMS feature

2026-09-08 · experiment · rejected · measured

Observation-only capture preserved the native estimator, and the analytical shadow agreed with an independent algebra implementation. The first current-width replay still failed its feature-value gate despite passing the fixed derivative tests, stopping before any target verdict.

Recorded measurement:

    Feature27 relative6.097862296908888e-6,maxabs1.7599077491060097e-5 versus1e-6/1e-8;served correction/sigma discrepancy8.387076404732103e-10;firstwide3.130358s,peak805,027,840B.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_response_quadratic_v1/adjudicate_affine_zero_curvature.py](code/current_response_quadratic_v1/adjudicate_affine_zero_curvature.py)
- [code/current_response_quadratic_v1/capture_current.py](code/current_response_quadratic_v1/capture_current.py)
- [code/current_response_quadratic_v1/certify_tiny.py](code/current_response_quadratic_v1/certify_tiny.py)
- [code/current_response_quadratic_v1/feature_second_jet_reference.py](code/current_response_quadratic_v1/feature_second_jet_reference.py)
- [code/current_response_quadratic_v1/gnn_second_jet_reference.py](code/current_response_quadratic_v1/gnn_second_jet_reference.py)
- [code/current_response_quadratic_v1/quadratic_covariance_reference.py](code/current_response_quadratic_v1/quadratic_covariance_reference.py)
- [code/current_response_quadratic_v1/qualify_wide.py](code/current_response_quadratic_v1/qualify_wide.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [FP64_WIDE_CLOSEOUT_2026_09_08_PDT.md#L20](FP64_WIDE_CLOSEOUT_2026_09_08_PDT.md#L20)

<details>
<summary>All result files and records (1)</summary>

- [FP64_WIDE_CLOSEOUT_2026_09_08_PDT.md](FP64_WIDE_CLOSEOUT_2026_09_08_PDT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-full-fp64-response-shadow-fails-current-width-primal-compatibility) · [Copy provenance](../../../PROVENANCE.json)


# Nonlinear pair memory worsens held-out plane error: pooled memory/control ratio 1.010987454

2026-09-08 · experiment · rejected · measured

Matched current-state models were compared on plane error and mean/variance consumers; the additional memory failed the required improvement.

Recorded measurement:

    VERDICT_2026_09_07_PDT.md:
    | Quantity | rank400 control | rank400 memory | rank500 control | rank500 memory |
    | Plane MSE |9.221599507e-5|9.323458288e-5|8.855818896e-5|8.952584915e-5|
    | Immediate post-head mean discrepancy |6.731541796e-8|6.967474055e-8|6.259395949e-8|6.476906494e-8|
    | Next physical variance discrepancy |1.106560880e-6|1.178090560e-6|1.046178668e-6|1.135685558e-6|
    Memory/control plane ratios1.011045674 and1.010926829; mean-consumer
    ratios1.035048770 and1.034749447; variance-consumer ratios1.064641432
    and1.085556026. Pooled held-out plane MSE9.138021601e-5 versus control
    9.038709202e-5 (ratio1.010987454). Memory/zero-decoder ratio.368283964;
    The prospectively fixed gate required at least20% plane improvement from
    The estimated full-student279.846B cost was never a measured deployment

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/nonlinear_memory_capacity_current_v1/capture_current.py](code/nonlinear_memory_capacity_current_v1/capture_current.py)
- [code/nonlinear_memory_capacity_current_v1/evaluate_consumers.py](code/nonlinear_memory_capacity_current_v1/evaluate_consumers.py)
- [code/nonlinear_memory_capacity_current_v1/evaluate_predictions.py](code/nonlinear_memory_capacity_current_v1/evaluate_predictions.py)
- [code/nonlinear_memory_capacity_current_v1/memory_model.py](code/nonlinear_memory_capacity_current_v1/memory_model.py)
- [code/nonlinear_memory_capacity_current_v1/prepare_rows.py](code/nonlinear_memory_capacity_current_v1/prepare_rows.py)
- [code/nonlinear_memory_capacity_current_v1/qualify_model.py](code/nonlinear_memory_capacity_current_v1/qualify_model.py)
- [code/nonlinear_memory_capacity_current_v1/train_capacity.py](code/nonlinear_memory_capacity_current_v1/train_capacity.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [code/candidate_response_pair_polynomial_f32_v1.py](code/candidate_response_pair_polynomial_f32_v1.py)
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

</details>

## Evidence

- [VERDICT_2026_09_07_PDT.md#L3](VERDICT_2026_09_07_PDT.md#L3)
- [VERDICT_2026_09_07_PDT.md#L36](VERDICT_2026_09_07_PDT.md#L36)
- [VERDICT_2026_09_07_PDT.md#L38](VERDICT_2026_09_07_PDT.md#L38)
- [VERDICT_2026_09_07_PDT.md#L39](VERDICT_2026_09_07_PDT.md#L39)
- [VERDICT_2026_09_07_PDT.md#L40](VERDICT_2026_09_07_PDT.md#L40)

<details>
<summary>All result files and records (1)</summary>

- [VERDICT_2026_09_07_PDT.md](VERDICT_2026_09_07_PDT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-nonlinear-pair-memory-worsens-held-out-plane-error-pooled) · [Copy provenance](../../../PROVENANCE.json)


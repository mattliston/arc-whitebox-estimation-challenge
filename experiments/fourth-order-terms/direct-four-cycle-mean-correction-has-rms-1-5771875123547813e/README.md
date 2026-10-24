# Direct four-cycle mean correction has RMS 1.5771875123547813e-11 and no useful signal

2026-09-07 · experiment · rejected · measured

The fixed source-derived correction was compared with existing layer13 mean error without running a new estimator.

Recorded measurement:

    CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md:
    | Bank served mean13 MSE |1.8242360939270932e-8|
    | Squared RMS of the source-derived correction |2.487520449127864e-22|
    | Correction RMS |1.5771875123547813e-11|
    | Correction RMS / original error RMS |1.1677309945771646e-7|
    | Mean(error*correction) |5.408428159457152e-18|
    | Natural direct-vector MSE |1.8242360950088036e-8|
    | Natural direct-vector MSE / bank mean13 MSE |1.0000000005929661|
    | Cosine with the needed correction |−0.002538909457390117|
    | Maximum relative direct-vector gain from this magnitude |2.335461852798204e-7|
    for this SAME direct-vector comparison. The observed ratio differs from1
    Actual diagnostic charge6340616 FLOPs, including conversions and vector/
    .0003926260s residual; high-water RSS64503808bytes. Score peak45154304B.
    FNP.12/Whest.16, noMPS. Native source cap2^27FLOPs/20s and8GiB RSS gates
    omega to9.039261238838492e-17 relative error. Source arrays and bill were
      Four-mode full unlearned rollout raw ratios35/100=.9988421436495236 /
      1.000107648992099. These were NOT complete adjusted-score measurements,
      but missed fixed raw gates: raw35/100=2.61823343950e-6/4.25983219808e-6.
    - RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md already preserves

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_cycle_direct_mean_witness_v1.py](code/check_cycle_direct_mean_witness_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_gaussian_fourth_r1_source_v1.py](code/check_gaussian_fourth_r1_source_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)

</details>

## Evidence

- [CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L3](CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L3)
- [CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L16](CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L16)
- [CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L17](CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L17)
- [CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L18](CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L18)
- [CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L19](CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md#L19)

<details>
<summary>All result files and records (1)</summary>

- [CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md](CYCLE_DIRECT_MEAN_WITNESS_V1_RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-direct-four-cycle-mean-correction-has-rms-1-5771875123547813e) · [Copy provenance](../../../PROVENANCE.json)


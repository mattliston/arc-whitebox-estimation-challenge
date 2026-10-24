# Float32 pair-polynomial evaluation improves two-network adjusted score by 0.9786264148%

2026-09-07 · experiment · adopted · measured

The precision partition reduced metered polynomial work while retaining the response-row parent and fixed scoring configuration.

Recorded measurement:

    PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md:
    once equally in call0. Adjusted is mean per-net MSE*max(.1,F/2^41).
    | Quantity | Retained response parent | FP32 polynomial candidate |
    |35 raw MSE|2.1099028082828634e-8|2.1088806425006294e-8|
    |100 raw MSE|1.997488978344223e-8|1.9971951803183636e-8|
    |35 completeF|476670806147|472173398147|
    |100 completeF|473227916215|468730508215|
    |35 adjusted|4.573526314361569e-9|4.528180121523978e-9|
    |100 adjusted|4.298579128245664e-9|4.2571005536736955e-9|
    |Mean raw|2.053695893313543e-8|2.0530379114094965e-8|
    |Mean completeF|474949361181|470451953181|
    |Mean adjusted|4.436052721303617e-9|4.392640337598837e-9|
    Mean adjusted ratio0.9902137358520792, gain0.9786264148%; per-net ratios
    0.9900850700923711 and0.9903506313750476. Raw MSE decreases0.0320389161%;
    completeF decreases0.9469236865%. The prospective BOTH-wins/mean<=.995 gate
    2.7418137e-6 on100(L13), below the fixed5e-6 gate. This is served-mean
    Native wall is18.7925..21.1510s across both roles; peakRSS6736920576B<8GiB.
    Candidate native residual0.61990..0.67603s is NOT a passing production residual
    strictly<1e-9, with validated banks<4e-9/<3e-9/<2e-9. The raw-accuracy gap

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_pair_polynomial_f32_dev_v1.py](code/check_pair_polynomial_f32_dev_v1.py)
- [code/build_pair_polynomial_f32_candidate_v1.py](code/build_pair_polynomial_f32_candidate_v1.py)
- [code/check_pair_polynomial_f32_v1.py](code/check_pair_polynomial_f32_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_pair_polynomial_f32_v1.py](code/candidate_response_pair_polynomial_f32_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/pair_polynomial_f32_source_v1.py](code/pair_polynomial_f32_source_v1.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)

</details>

## Evidence

- [PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L3](PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L3)
- [PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L11](PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L11)
- [PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L13](PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L13)
- [PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L15](PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L15)
- [PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L16](PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md#L16)

<details>
<summary>All result files and records (1)</summary>

- [PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md](PAIR_POLYNOMIAL_F32_DEV_RESULT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-float32-pair-polynomial-evaluation-improves-two-network-adjusted-score) · [Copy provenance](../../../PROVENANCE.json)


# Frozen response-row reduction improves two-network adjusted score by 3.63221%

2026-09-07T02:36Z · experiment · adopted · measured

The no-refit port of the learned response-row reduction (frozen payload P with 16 full-feature heads; it replaces the bank's scalar-final Q head and selects response rows at threshold .01) was acquired natively on ranks 35 and 100 against the unchanged bank with the same stored 1e9 truth means and complete per-call bills. The mean of per-net raw x max(.1, F/2^41) fell 3.63221%, with a raw worsening on rank 35 that the old raw-only closeout had used to reject the method, so the development gate passed and the candidate advanced unchanged to the independent 14-net panel.

Recorded measurement:

    Rank 35: bank raw 2.0731001663431382e-8 / 490466018197 F / adjusted 4.623803688036139e-9 vs response 2.1099028082828634e-8 / 476670806147 F / 4.573526314361569e-9; rank 100: bank 2.0592545939233167e-8 / 489374517341 F / 4.582701526412439e-9 vs response 1.997488978344223e-8 / 473227916215 F / 4.298579128245664e-9; two-net mean adjusted 4.603252607224289e-9 vs 4.436052721303617e-9 (ratio .9636778816660484); mean raw ratio .9939591407109105; output max errors 2.14281e-6/2.55457e-6 vs the archived reference.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)

</details>

## Evidence

- [RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md#L12](RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md#L12)
- [RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md#L17](RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md#L17)
- [RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md#L55](RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md#L55)

<details>
<summary>All result files and records (2)</summary>

- [RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md](RESPONSE_ROWS_ADJUSTED_DEVELOPMENT_RESULT_V1.md) · result
- [result.json](result.json) · result

</details>

## Related work

- enables: [Frozen response-row reduction authorized for one score re-evaluation with four-level Winograd products](../../../records/research/2026-09-07-frozen-response-row-reduction-authorized-one-score-re-evaluation/README.md)
- enables: [Frozen response-row reduction improves independent14 adjusted score by 3.25073% and is retained](../frozen-response-row-reduction-improves-independent14-adjusted-score-3/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-frozen-response-row-reduction-improves-two-network-adjusted-score) · [Copy provenance](../../../PROVENANCE.json)


# One Gaussian fourth-cumulant mode after layer 12 fails the fixed .5 next-mean response gate

2026-09-07T03:47:22Z · experiment · rejected · measured

A target-free information test on rank 35 compared the exact ray contraction of the masked Gaussian two-quadratic/two-linear fourth source born at post-layer-12 (consumer layer 13, actual W13 rays) against a same-source radial control and against that control plus one fixed signed 16-power full-matrix mode (its literal bank-Q scalar overlap removed). The extra mode reduced the isolated source-local mean-response error by only .6447068751124108%, failing the preregistered E1/E0 <= .5 gate, so the fixed supplier was closed without a carrier, fit, extra mode, column, layer or strength test. Bank output bytes matched the sealed rank-35 output and the bank bill was identical to its prior first call.

Recorded measurement:

    E0 1.0399728226818358e-10, E1 1.0332680463947054e-10, E1/E0 .9935529312488759; correction/residual cosine .08537515238819657; correction norm 3.7329154294891325e-5 vs residual norm 3.2633298491360015e-4; leading Rayleigh value 14.075188395070171; source partial trace 37.52352999955553, mode partial trace 1.6200257934122577; bank bill 490,466,018,197 F; proxy 13,040,273,442 F (cap 20B); native capture wall 21.0457798336 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/check_gaussian_fourth_r1_source_v1.py](code/check_gaussian_fourth_r1_source_v1.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)

</details>

## Evidence

- [GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md#L18](GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md#L18)
- [GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md#L28](GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md#L28)
- [GAUSSIAN_FOURTH_R1_SOURCE_TEST_PROTOCOL_V1.md#L118](GAUSSIAN_FOURTH_R1_SOURCE_TEST_PROTOCOL_V1.md#L118)

<details>
<summary>All result files and records (7)</summary>

- [GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md](GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md) · result
- [GAUSSIAN_FOURTH_R1_SOURCE_TEST_PROTOCOL_V1.md](GAUSSIAN_FOURTH_R1_SOURCE_TEST_PROTOCOL_V1.md) · record
- [GAUSSIAN_FOURTH_R1_SOURCE_TEST_ACCEPTANCE_V1.md](GAUSSIAN_FOURTH_R1_SOURCE_TEST_ACCEPTANCE_V1.md) · record
- [GAUSSIAN_FOURTH_R1_SAME_SOURCE_PROXY_V1.md](GAUSSIAN_FOURTH_R1_SAME_SOURCE_PROXY_V1.md) · record
- [GAUSSIAN_FOURTH_R1_SOURCE_CLOSURE_AUDIT_V1.md](GAUSSIAN_FOURTH_R1_SOURCE_CLOSURE_AUDIT_V1.md) · record
- [GAUSSIAN_FOURTH_R1_PLANNING_COST_V1.md](GAUSSIAN_FOURTH_R1_PLANNING_COST_V1.md) · record
- [result.json](result.json) · result

</details>

## Related work

- supersedes: [Full-matrix fourth-cumulant modulation modes sum_r lambda_r S(V_r): 64.42450944B transport per mode, no tested source](../../../records/research/2026-09-07-full-matrix-fourth-cumulant-modulation-modes-sum-r-lambda/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-one-gaussian-fourth-cumulant-mode-after-layer-12-fails) · [Copy provenance](../../../PROVENANCE.json)


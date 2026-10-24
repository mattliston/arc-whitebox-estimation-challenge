# Quadratic response-covariance action improves next-mean cross-risk only .301445% and fails its gate

2026-09-08 · experiment · rejected · measured

A fixed amplitude-one layer12 correction was checked against two independent truth streams without fitting or a full estimator intervention.

Recorded measurement:

    CLOSED_2026_09_08_PDT.md:
    | Next-mean cross-stream risk |2.4633411085508702e-9|2.455915481776116e-9|.9969855466833326|
    | Stream A next-mean squared error |2.953289721514421e-9|2.9455285372230347e-9|.9973720206876938|
    | Stream B next-mean squared error |2.9525055822168254e-9|2.9454155129587035e-9|.9975986262986848|
    | Secondary covariance cross-risk |1.3617925504782897e-8|1.3673111843420251e-8|1.004052477641912|
    The primary gate required cross-risk ratio <=.75 and each individual
    stream ratio <=.85. All fail: the primary improvement is only .301445%.
    All 47 numerical/frame/provenance guards passed; error=null. Therefore
    repair and rerun. Wall1.09019483346s, peakRSS622886912B; one nice10 OFFPY
    Alive count832. Signal energy1.6067520494084327e-12, signal/error products
    development nets: learned/host cross-MSE ratios .866590867724 and
    threshold were only about1.04/.52 paired jackknife standard errors. This
    is neither strong-confidence10% evidence nor final raw improvement.
    | PROTOCOL.md |aed9a97cdf4a297f61b9500e12c3f29b5a7aed37f69b9eb28ba707efee1adaca|
    | check.py |b6b94734dae577e3049edd421131c04e51201bf21534602863ffe9cc7bc54e44|
    | witness/result.json |e41ec3f2f1f5535568dd5f3cb62c4ac83cb6f2407d78ef16ccd4fb7ee7e86c9f|
    | witness/prediction.npz |cdb8668301f893ad6d01ca57cff3aa5363955d1d6b0361f7a38ec60b4ff742a6|
    | S/diagnosis/result.json |14b3dbabcd594587f9898acb1f8e3287932e00e9e25bf17347ff5c14a2d92227|
    | T/qualification/result.json |52938606d3f116f5f41021fd19f5f3b03f8b9d7c2713741d0d19cbada4221494|

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/current_response_covariance_witness_v1/check.py](code/astra_phase2_reexamination_v1/current_response_covariance_witness_v1/check.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py)
- [code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py](code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py)
- [code/astra_phase2_reexamination_v1/current_response_quadratic_v1/feature_second_jet_reference.py](code/astra_phase2_reexamination_v1/current_response_quadratic_v1/feature_second_jet_reference.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/astra_phase2_reexamination_v1/current_response_quadratic_v1/quadratic_covariance_reference.py](code/astra_phase2_reexamination_v1/current_response_quadratic_v1/quadratic_covariance_reference.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [code/public_g19_attribution_v1/mc_truth_mps_indep.py](code/public_g19_attribution_v1/mc_truth_mps_indep.py)

</details>

## Evidence

- [CLOSED_2026_09_08_PDT.md#L3](CLOSED_2026_09_08_PDT.md#L3)
- [CLOSED_2026_09_08_PDT.md#L29](CLOSED_2026_09_08_PDT.md#L29)
- [CLOSED_2026_09_08_PDT.md#L30](CLOSED_2026_09_08_PDT.md#L30)
- [CLOSED_2026_09_08_PDT.md#L31](CLOSED_2026_09_08_PDT.md#L31)
- [CLOSED_2026_09_08_PDT.md#L32](CLOSED_2026_09_08_PDT.md#L32)

<details>
<summary>All result files and records (2)</summary>

- [CLOSED_2026_09_08_PDT.md](CLOSED_2026_09_08_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-quadratic-response-covariance-action-improves-next-mean-cross-risk) · [Copy provenance](../../../PROVENANCE.json)


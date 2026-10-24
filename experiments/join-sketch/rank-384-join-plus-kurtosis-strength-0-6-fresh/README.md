# Rank-384 join plus kurtosis strength 0.6: fresh-24 noise-free adjusted -0.199%, aggregate -0.096%

2026-10-16T01:34:04Z · experiment · adopted · measured

An independent 24-network test of the combined capacity and dial change gave a small mean advantage with wide uncertainty. The result supported a modest expected-value stack rather than a large gain.

Recorded measurement:

    xMSE-1.801%,SE.464,19/24;plain-1.063%;bill+1.603%,SE.006;xnet-.199%,SE.465,11/24;aggregate-.096%;last8+.089%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_acc_dials_v1/grade_x24_v1.py](code/lead/runs/opus_acc_dials_v1/grade_x24_v1.py)
- [code/lead/runs/opus_acc_dials_v1/candidate_full384_gk4_060_v1.py](code/lead/runs/opus_acc_dials_v1/candidate_full384_gk4_060_v1.py)
- [code/lead/runs/opus_acc_dials_v1/analyze_v1.py](code/lead/runs/opus_acc_dials_v1/analyze_v1.py)
- [code/lead/runs/opus_acc_dials_v1/analyze_v2.py](code/lead/runs/opus_acc_dials_v1/analyze_v2.py)
- [code/lead/runs/opus_acc_dials_v1/dial_judge_v1.py](code/lead/runs/opus_acc_dials_v1/dial_judge_v1.py)
- [code/lead/runs/opus_acc_dials_v1/dial_judge_v2.py](code/lead/runs/opus_acc_dials_v1/dial_judge_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3638](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3638)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3643](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3643)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3664](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3664)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [X24_DECISION_v1.json](../../dials/transport-gate-kurtosis-strength-0-6-fresh-eight-noise/X24_DECISION_v1.json) · result
- [analysis_latest.json](../../dials/transport-gate-kurtosis-strength-0-6-fresh-eight-noise/analysis_latest.json) · result
- [jobs_b1.json](../../dials/transport-gate-kurtosis-strength-0-6-fresh-eight-noise/jobs_b1.json) · result
- [jobs_b2.json](../../dials/transport-gate-kurtosis-strength-0-6-fresh-eight-noise/jobs_b2.json) · result
- [jobs_b3.json](../../dials/transport-gate-kurtosis-strength-0-6-fresh-eight-noise/jobs_b3.json) · result
- [jobs_b4.json](../../dials/transport-gate-kurtosis-strength-0-6-fresh-eight-noise/jobs_b4.json) · result

</details>

## Related work

- shipped in: [Submission #336367: layer-10 rank 384, compact kernels and exact request cuts; seven runtime failures](../../../submissions/336367/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-rank-384-join-plus-kurtosis-strength-0-6-fresh) · [Copy provenance](../../../PROVENANCE.json)


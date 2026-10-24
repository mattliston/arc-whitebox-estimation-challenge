# Layer-10 join rank 384: 24-network noise-free adjusted -0.461%, final eight +0.173%

2026-10-16T01:07:36Z · experiment · adopted · measured

Broader noise-corrected tests found consistent raw accuracy gains from full rank 384 but did not satisfy the original final-eight rule. The expanded stack was eventually submitted after additional evidence and exact request repair.

Recorded measurement:

    24 xMSE-2.081%,SE.412,21/24lower;plain-1.552%;bill+1.621%,SE.016;xnet-.461%,SE.410,12/24;last8+.173%,2/8;ops+384;initialrisk22.44%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_f384_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/census_v1.py](code/census_v1.py)
- [code/grade_sci24_v1.py](code/grade_sci24_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3615](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3615)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3613](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3613)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3617](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3617)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [F384_DECISION_v1.json](F384_DECISION_v1.json) · result
- [F384_STACK_DECISION_v1.json](F384_STACK_DECISION_v1.json) · result
- [SCI24_DECISION_v1.json](SCI24_DECISION_v1.json) · result
- [JUDGE_OFFICIAL120_f384_v1.txt](JUDGE_OFFICIAL120_f384_v1.txt) · result

</details>

## Related work

- shipped in: [Submission #336367: layer-10 rank 384, compact kernels and exact request cuts; seven runtime failures](../../../submissions/336367/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-layer-10-join-rank-384-24-network-noise-free) · [Copy provenance](../../../PROVENANCE.json)


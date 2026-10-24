# Usage-ranked prebind list reduced requests by 167–438 with byte-identical heavy-column-sketch predictions

2026-10-18T02:49:25Z · experiment · adopted · measured

A census on eight fit and five holdout networks identified unused setup keys and reordered the prebind list by usage. Swapping thirty keys reduced lazy binding without changing predictions and entered submission #336867.

Recorded measurement:

    361 of 649 unused; 30 swapped; simulated first/later OLD 1,394/975, NEW 503/466; verified requests -167/-438; FLOPs -0.06/-0.08%; normal residual -2.2/-1.5%, slow -2.7/-0.6%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_fix336687_v1/jsk/fragment_prebind_v53j_v1.py](code/lead/runs/opus_fix336687_v1/jsk/fragment_prebind_v53j_v1.py)
- [code/engineer/a62_modules/prebind_targeted_v2.py](code/engineer/a62_modules/prebind_targeted_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5077](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5077)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5084](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5084)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5086](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5086)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

## Related work

- shipped in: [Submission #336867: usage-ranked prebind list: 100 networks passed at 1.4515052706908603e-09](../../../submissions/336867/README.md)

[Research graph](../../../graph/index.html#node=2026-10-18-usage-ranked-prebind-list-reduced-requests-167438-byte-identical) · [Copy provenance](../../../PROVENANCE.json)


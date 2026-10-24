# Generation-0 threshold 7000 reduced young collections tenfold but platform time saving did not reproduce

2026-10-17T16:48:57Z · experiment · adopted · measured

Increasing the generation-0 threshold to 7000 during prediction kept prediction bytes, FLOPs and requests identical. Local paired residual savings were not reproduced per unit of host speed in the graded union-product submission.

Recorded measurement:

    Young GC 15–20 per network versus ~220; SAFE 16–18 versus 204–233; local residual -3.3%; platform residual x1.018 median with kernel x1.015, ~+0.3% per unit host speed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_v34y.py](code/candidate_v34y.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4887](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4887)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4888](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4888)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4909](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4909)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

## Related work

- shipped in: [Submission #336779: union of inherited and newborn push products: 1.4688380752067338e-09](../../../submissions/336779/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-generation-0-threshold-7000-reduced-young-collections-tenfold-but) · [Copy provenance](../../../PROVENANCE.json)


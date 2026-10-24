# Six-worker experiment fails throughput threshold: 40 evaluations versus 46 at four workers

2026-10-13T17:44Z · experiment · rejected · measured

With about 10 heavy processes on the box, the refit round's worker cap was raised from 4 to 6 for a timed window; throughput fell rather than rose, so the two extra workers were retired after their current evals and the cap went back to 4.

Recorded measurement:

    46 evals in the 20 min before the raise at 4 workers; 40 evals in the 20 min after at 6 workers; pass bar >= 1.15 x 46; memory_pressure free 74%, swap 1525M

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cap6_s56_v3.sh](code/cap6_s56_v3.sh)
- [code/retire_top_v2.sh](code/retire_top_v2.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2194](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2194)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2203](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2203)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-six-worker-experiment-fails-throughput-threshold-40-evaluations-versus) · [Copy provenance](../../../PROVENANCE.json)


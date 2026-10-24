# Small refit rounds rejected for native insignificance or independent-check failure

2026-10-13T08:06Z · experiment · rejected · measured

candidate generation 53 (S53) (sizing n = 64, 6 arms) froze at leave-one-out (LOO) -0.023% and passed its check at -0.030% (se 0.013), but every component moves native predictions below the 2e-6 rounding floor, so s53k1 read +0.003% on list a and was shelved; candidate generation 55 (S55) froze at -0.060% and failed its check (+0.027%).

Recorded measurement:

    S53 LOO -0.023%, check -0.030% (24/32); components arrp 7.6e-7, h17 5.8e-7, dx2a1 3.7e-7 of the prediction; S55 LOO -0.060%, check +0.027% (se 0.054, 14/32)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/jmS53_v1_pbs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1763](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1763)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1848](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1848)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1963](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1963)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1977](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1977)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1998](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1998)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1999](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1999)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2001](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2001)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2004](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2004)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2006](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2006)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2007](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2007)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2111](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2111)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2118](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2118)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2124](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2124)

<details>
<summary>All result files and records (6)</summary>

- [FROZEN_POINT_v1.json](FROZEN_POINT_v1.json) · record
- [CHECK_VERDICT_v1.json](CHECK_VERDICT_v1.json) · record
- [RULES_v1.json](RULES_v1.json) · record
- [FROZEN_POINT_v1-a2b41b69.json](FROZEN_POINT_v1-a2b41b69.json) · record
- [CHECK_VERDICT_v1-7b91c67b.json](CHECK_VERDICT_v1-7b91c67b.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Re-sizing previously fitted arms gives no fresh-network signal; two-stage pruning fails its stability rule](../re-sizing-previously-fitted-arms-gives-no-fresh-network/README.md)
- enables: [Four-arm refit accepted: -0.137% independent and -0.231% sealed MSE change](../../../records/research/2026-10-13-four-arm-refit-accepted-0-137-independent-0-231/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-small-refit-rounds-rejected-native-insignificance-or-independent-check) · [Copy provenance](../../../PROVENANCE.json)


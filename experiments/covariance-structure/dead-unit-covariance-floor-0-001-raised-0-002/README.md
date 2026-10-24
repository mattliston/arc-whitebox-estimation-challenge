# Dead-unit covariance floor 0.001 raised to 0.002 / 0.0015: +0.015% / -0.015% adjusted on 32 ranks (bill repaid in raw)

2026-10-19T09:52:37Z · experiment · rejected · measured

Raising the dead-unit covariance floor (_DCV_FLOOR) removes small covariance products: on seven ranks 0.002 read -0.29% vs the base, but on 32 paired ranks it is +0.015% because the uniform bill saving (every rank negative) is paid back by raw +0.148% with a positive held-out mean, a 1:1 trade; 0.0015 is a wash. The family was closed at every level and the 0.003/0.004 arms cancelled.

Recorded measurement:

    0.002 (dcv002): 7 ranks raw -0.277% (se .19, 4/7), bill -0.374%, adjusted -0.649% (se .18) = -0.29% vs base; 32 ranks all32 +0.015% (se .087, 15/32 lower) = bill -0.132% (-0.05..-0.28 every rank) + raw +0.148%; heldout25 +0.101% (raw +0.236%), seven -0.291%; 0.0015 (dcv0015): -0.015% (se .058; bill -0.075%, raw +0.060%; held-out +0.003%); dcv002 peak RSS 7.02 GB

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_dials_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_dcv002_sci_v1.py](code/candidate_dcv002_sci_v1.py)
- [code/candidate_dcv0015_sci_v1.py](code/candidate_dcv0015_sci_v1.py)
- [code/candidate_rel_ab47_dcv002_v1.py](code/candidate_rel_ab47_dcv002_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5914](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5914)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5917](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5917)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5930](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5930)

<details>
<summary>All result files and records (2)</summary>

- [science32_dcv002.log](science32_dcv002.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-dead-unit-covariance-floor-0-001-raised-0-002) · [Copy provenance](../../../PROVENANCE.json)


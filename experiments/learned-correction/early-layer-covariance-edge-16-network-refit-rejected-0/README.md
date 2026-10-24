# Early-layer covariance edge and a 16-network refit rejected: +0.831% FLOPs and +0.197% independent MSE

2026-10-11T23:03Z · experiment · rejected · measured

The trained covariance edge applied at layers 1-7 passed its gradient gate on the candidate generation 48 (S48) moment-closure estimator but costs 0.115% of metered FLOPs per layer natively; the refit-after-drop (RAD)-S48 round sized on 16 networks overfit (model -0.60%, leave-one-out -0.045%) and failed its fresh check.

Recorded measurement:

    edx 0.1 bill +0.831 / +0.798%, +551 requests; parity corr 0.62-0.85, native/torch 1.18-1.71; RAD-S48 check A38 7100392-423 +0.197% (se 0.183, 15 of 32)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_early_edge_v1.py](code/fragment_early_edge_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1126](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1126)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1139](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1139)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1206](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1206)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1211](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1211)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1217](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1217)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1219](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1219)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1223](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1223)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1227](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1227)

<details>
<summary>All result files and records (4)</summary>

- [FROZEN_POINT_v1.json](FROZEN_POINT_v1.json) · record
- [CHECK_VERDICT_v1.json](CHECK_VERDICT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [edgex_table_v1.out](edgex_table_v1.out) · result

</details>

## Related work

- enables: [Refit-after-drop strength round accepted: -0.203% pooled check and -0.379% sealed MSE](../../../records/research/2026-10-12-refit-after-drop-strength-round-accepted-0-203-pooled/README.md)
- ← enables: [First refit-after-drop strength round adopted: -0.223% on fresh networks at zero added FLOPs](../../../records/research/2026-10-11-first-refit-after-drop-strength-round-adopted-0-223/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-early-layer-covariance-edge-16-network-refit-rejected-0) · [Copy provenance](../../../PROVENANCE.json)


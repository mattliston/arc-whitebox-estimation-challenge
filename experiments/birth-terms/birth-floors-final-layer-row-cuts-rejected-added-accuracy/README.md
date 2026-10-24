# Birth floors and final-layer row cuts rejected: added accuracy is outweighed by FLOPs or error

2026-10-12T17:04Z · experiment · rejected · measured

Adding births at a lower floor buys -0.8..-1.0% for +6-14% births (several percent of metered FLOPs); raising the floor late costs +4.6..+38.6% error for -1.6..-4.3% metered FLOPs; the final-layer row cut at z 3.0 is neutral natively (-0.29% metered FLOPs, +0.25% MSE).

Recorded measurement:

    floor 0.005 -0.770% (se 0.484), 0.003 -0.985%; b02_8_14 +4.93% for -1.61%, b05 +37.7% for -4.27%; z 3.0 native score -0.042% (se 0.172), z 2.5 +3.44%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/physicist/t113_table.py](code/physicist/t113_table.py)
- [code/physicist/t129_table.py](code/physicist/t129_table.py)
- [code/engineer/a62_modules/s49/rz30/candidate.py](code/engineer/a62_modules/s49/rz30/candidate.py)
- [code/engineer/a62_modules/make_floor_v1.py](code/engineer/a62_modules/make_floor_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1160](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1160)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1207](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1207)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1208](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1208)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1215](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1215)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1216](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1216)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1534](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1535](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1535)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1542](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1542)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1549](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1549)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1559](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1559)

<details>
<summary>All result files and records (3)</summary>

- [sel_counts.jsonl](sel_counts.jsonl) · result
- [sel_counts-b2332fb7.jsonl](sel_counts-b2332fb7.jsonl) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Capacity redistribution rejected: early birth reduction adds 18.0% MSE; late capacity costs exceed gains](../../third-order-planes/capacity-redistribution-rejected-early-birth-reduction-adds-18-0/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-birth-floors-final-layer-row-cuts-rejected-added-accuracy) · [Copy provenance](../../../PROVENANCE.json)


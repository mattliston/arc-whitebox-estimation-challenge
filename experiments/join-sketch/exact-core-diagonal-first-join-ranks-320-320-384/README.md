# Exact core diagonal at the first join with ranks 320/320/384: first network adjusted +3.20% (bill -1.20%), closed early

2026-10-14T20:48:18Z · experiment · rejected · measured

Repairing the first-join Nystrom core with its exact diagonal while funding it by lowering the layer-7 and layer-10 join ranks to 320 (one extra 320 x 6144 product) saved 1.2% of bill but raised the MSE by 4.4% on the first network, tripping the pre-registered +2% gross rule; the rank-384 variant was never run.

Recorded measurement:

    7100136: adjusted +3.202107998%, log MSE +4.352289091%, log bill -1.200379772%, 3,060,055,068 FLOPs saved; core trace 442.0027466 -> 443.0115051

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_join_core_diagonal_rank320_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_jcd320_HELD_v1.py](code/fragment_jcd320_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3052](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3052)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3060](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3060)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3063](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3063)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3066](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3066)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3068](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3068)

<details>
<summary>All result files and records (3)</summary>

- [ROOT_RETIREMENT_ACCEPTANCE_v1.json](ROOT_RETIREMENT_ACCEPTANCE_v1.json) · record
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-exact-core-diagonal-first-join-ranks-320-320-384) · [Copy provenance](../../../PROVENANCE.json)


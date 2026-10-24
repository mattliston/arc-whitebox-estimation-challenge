# Dead-cut tiles rejected on the current implementation: -0.0088% FLOPs costs 5,869 charged operations

2026-10-12T03:15:53Z · experiment · rejected · measured

Release's max-aggressive twin table priced dead-cut tiles at -0.12% of metered FLOPs from an candidate generation 43 (S43)-skeleton measurement taken against the padded product at inner size 1024; measured on twin v2 (xsgTw + _DCV_SYM 'tiles') on 7100136 with every listed key bound, the tiles cut only 0.0088% of metered FLOPs for +5,869 charged operations (final MSE +0.05%, rounding level), because true-inner-size covariance sandwich (XSDM) (the inner-size sandwich, in every file since S47e) already runs that product at the true inner size m. With the tiles gone, the strip lever seven-level strip push (HZ7) alone (+0.1..+0.5 podium points) was below a daily slot, so no third twin was built on the candidate generation 48 (S48) physics; the table also showed that dropping the three-request loads (L3off) helps at 100 networks only and loses at 300 (P(loss) 27-28%).

Recorded measurement:

    tiles on twin v2: bill -0.0088%, +5,869 charged ops, final MSE +0.05%; table P(podium) at 100 / 300 networks (x1.00): fired twin 15.2 / 14.9, +XSG 16.0 / 15.7, +XSG+HZ7 16.2 / 15.9, +XSG+L3off 16.4 / 15.6 (P(loss) 10.1% / 27.3%), +XSG+tiles BOUND 17.3 / 16.9 (stale price); twin v2 + HZ7 +0.1..+0.5 points

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/release/max_twin_table.py](code/release/max_twin_table.py)
- [code/engineer/a62_modules/s49/xsgTw_tiles/candidate.py](code/engineer/a62_modules/s49/xsgTw_tiles/candidate.py)
- [code/engineer/a62_modules/s49/xsgTw/candidate.py](code/engineer/a62_modules/s49/xsgTw/candidate.py)
- [code/release/pair_mc_twin.py](code/release/pair_mc_twin.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1280](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1280)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1282](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1282)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1285](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1285)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1286](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1286)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1287](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1287)

<details>
<summary>All result files and records (3)</summary>

- [max_twin_table.json](max_twin_table.json) · result
- [build.json](build.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-dead-cut-tiles-rejected-current-implementation-0-0088-flops) · [Copy provenance](../../../PROVENANCE.json)


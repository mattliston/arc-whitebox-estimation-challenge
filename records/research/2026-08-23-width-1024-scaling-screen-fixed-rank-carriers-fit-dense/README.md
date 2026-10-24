# Width-1024 scaling screen: fixed-rank carriers fit; dense third-order transport exceeds the budget

2026-08-23T17:16Z · finding · informational · derived

An algebraic screen compared the two geometries before any flopscope receipt: the budget grows 8.0846x, weight and forward geometry 8.0x, and a fixed-rank O(depth x width^2 x rank) carrier is almost budget-neutral, while dense cubic transitions grow 30.97x (3.83x relative to budget). The Phase 1 rank-22 host's dense PSD certificate costs 21,543,026,688 FLOPs per transition at width 1024, so an eight-times port of that host lands 212,087,161,104 FLOPs (9.64%) over the Phase 2 budget. The dense Cholesky/PSD repair was therefore named the first redesign target.

Recorded measurement:

    ratios: budget 8.084644321882353, weight_and_forward_geometry 8.0, trajectory_cells 2.0, fixed_rank_relative_to_budget 0.9895302355289459, dense_cubic_transition_geometry 30.967741935483872, dense_cubic_relative_to_budget 3.8304396214023715; phase2 dense PSD total for 15 transitions 323,145,400,320 FLOPs (0.14694951474666595 of budget); eight_x_screen_over_budget 212,087,161,104 (fraction 0.09644604815859868)

## Programs

- [verify_phase2_static.py](verify_phase2_static.py)

## Evidence

- [STATIC_SCALING_GATE.json#L22](STATIC_SCALING_GATE.json#L22)
- [STATIC_SCALING_GATE.json#L31](STATIC_SCALING_GATE.json#L31)
- [source_README.md#L21](source_README.md#L21)

<details>
<summary>All result files and records (3)</summary>

- [STATIC_SCALING_GATE.json](STATIC_SCALING_GATE.json) · result
- [MIGRATION_MATRIX.md](MIGRATION_MATRIX.md) · record
- [source_README.md](source_README.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-23-width-1024-scaling-screen-fixed-rank-carriers-fit-dense) · [Copy provenance](../../../PROVENANCE.json)


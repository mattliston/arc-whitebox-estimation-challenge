# Request and FLOP cost map separates vector overhead, matrix work and fast arithmetic kernels

2026-10-17T17:00:43Z · tool · informational · measured

The request cost map measured physics vector operations, physics matrix operations and fast arithmetic kernels separately to price runtime changes. It identified vector round trips as large in request count and small in billed arithmetic, while early layers consumed disproportionate bill relative to final error.

Recorded measurement:

    Physics vector: 10,160 charges = 24.5% requests / 0.40% bill; physics matrix: 8,871 = 21.4% / 6.2%; kernel plus grouped strided reads: 22,370 = 54% / 94.3%; layers 0–6 = 42% bill / 7% final error; old tier 63 GF (24.5%).

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4889](../../TEAM_LOG_2026-10-01_to_10-22.md#L4889)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4891](../../TEAM_LOG_2026-10-01_to_10-22.md#L4891)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log
- [COST_MAP_v1.md](COST_MAP_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-request-flop-cost-map-separates-vector-overhead-matrix-work) · [Copy provenance](../../../PROVENANCE.json)


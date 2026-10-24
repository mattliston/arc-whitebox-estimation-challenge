# Platform kernel and participant-time tails cause repeated failures despite stable request counts

2026-10-16T02:25:53Z · finding · informational · measured

Paired runs showed large random host slowdowns rather than accumulation by worker position. The observed residual and wall tails motivated leaner kernels and later runtime routing.

Recorded measurement:

    kernel320-420ms/GFLOP;noise~+-10%,tails+20%;heavynets~100smedian,max100draws~115-120s;residual.413vs.372onmatchednet

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3709](../../TEAM_LOG_2026-10-01_to_10-22.md#L3709)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3704](../../TEAM_LOG_2026-10-01_to_10-22.md#L3704)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3713](../../TEAM_LOG_2026-10-01_to_10-22.md#L3713)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3744](../../TEAM_LOG_2026-10-01_to_10-22.md#L3744)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-platform-kernel-participant-time-tails-cause-repeated-failures-despite) · [Copy provenance](../../../PROVENANCE.json)


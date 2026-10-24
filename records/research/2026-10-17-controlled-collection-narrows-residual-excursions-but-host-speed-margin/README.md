# Controlled collection narrows residual excursions, but the host-speed margin still requires median at most .320 s

2026-10-17T15:35:21Z · finding · informational · measured

Cross-run telemetry showed stochastic residual excursions on a narrow runtime margin. Per-prediction heap freezing reduced their spread, after which the host-speed episode defined the nomination margin rather than a single zero-failure run.

Recorded measurement:

    Controlled 300 networks: p99 1.057, max 1.127; default-GC 1097 networks: p99 1.179, max 1.365; median x (1 + 22% + 2 percentage points) <= .400 implies median <= .320.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4878](../../TEAM_LOG_2026-10-01_to_10-22.md#L4878)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4896](../../TEAM_LOG_2026-10-01_to_10-22.md#L4896)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log
- [GUARANTEE_MARGIN_v1.md](GUARANTEE_MARGIN_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-controlled-collection-narrows-residual-excursions-but-host-speed-margin) · [Copy provenance](../../../PROVENANCE.json)


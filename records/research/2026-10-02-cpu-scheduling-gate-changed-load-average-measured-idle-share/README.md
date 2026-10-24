# CPU scheduling gate changed from load average to measured idle share

2026-10-02T13:15Z · decision · informational · measured

A load average of 51 coexisted with 58% idle CPU, leaving experiments blocked unnecessarily. Every lane changed to a two-second idle-share gate while preserving memory limits.

Recorded measurement:

    1-minute load 51; 58% of 16 cores idle; idle_gate.sh 30

## Programs

- [../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/idle_gate.sh](../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/idle_gate.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L326](../../TEAM_LOG_2026-10-01_to_10-22.md#L326)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-cpu-scheduling-gate-changed-load-average-measured-idle-share) · [Copy provenance](../../../PROVENANCE.json)


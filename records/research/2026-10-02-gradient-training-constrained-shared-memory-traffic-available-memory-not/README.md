# Gradient training is constrained by shared memory traffic and available memory, not only CPU cores

2026-10-02T00:17Z · finding · informational · measured

Concurrent forward work slows gradient training through memory traffic, and two simultaneous gradient workers exceed the available memory margin. The pool therefore uses one worker during memory pressure and grants training priority over other heavy work.

Recorded measurement:

    idle forward8s vs15-20s beside other jobs; gradient73-74s,8updates/hour; two workers159-190s/gradient vs one101-115s; gradient peak7.5GB,forward3.5GB; swap175MiB ->3.9GB of5.1GB when both start

## Programs

- [../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/autoscale.sh](../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/autoscale.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L65](../../TEAM_LOG_2026-10-01_to_10-22.md#L65)
- [TEAM_LOG_2026-10-01_to_10-22.md#L151](../../TEAM_LOG_2026-10-01_to_10-22.md#L151)
- [TEAM_LOG_2026-10-01_to_10-22.md#L391](../../TEAM_LOG_2026-10-01_to_10-22.md#L391)
- [TEAM_LOG_2026-10-01_to_10-22.md#L395](../../TEAM_LOG_2026-10-01_to_10-22.md#L395)
- [TEAM_LOG_2026-10-01_to_10-22.md#L398](../../TEAM_LOG_2026-10-01_to_10-22.md#L398)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-gradient-training-constrained-shared-memory-traffic-available-memory-not) · [Copy provenance](../../../PROVENANCE.json)


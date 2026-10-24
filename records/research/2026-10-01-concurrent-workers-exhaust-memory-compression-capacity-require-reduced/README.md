# Concurrent workers exhaust memory compression capacity and require reduced concurrency

2026-10-01T22:35Z · incident · informational · measured

With three trainer workers (6.4-6.7 GB each), native jobs, torch screens and a browser on the 16-core 48 GB box, swap grew in bursts (the native guard kills a job at +256 MiB), the compressor held 24 GB with 10-18 thousand pages compressed per second and load reached 52 with nobody's job moving. The trainer's supervisor shed a worker per 128 MiB of swap growth and was capped at 2 workers; the lead paused lanes by signal, set one process of at most 2 threads per lane while the refit pool trained, and on 13:16Z replaced the load-average gate (which read 51-90 with the machine 55% idle) by an idle-share gate.

Recorded measurement:

    swap bursts +135 MiB (19:30Z) and +261 MiB (19:54Z); compressor 24 GB, free pages 0.1 GB, load 17-30 then 46-52; forward 34 s instead of 10 s; trainer workers 35% user + 29% system of a core each; gradient 100-145 s under load vs 73-74 s quiet; idle_gate.sh 30

## Programs

- [gate.sh](gate.sh)
- [idle_gate.sh](idle_gate.sh)
- [autoscale.sh](autoscale.sh)
- [own_gate.sh](own_gate.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6](../../TEAM_LOG_2026-10-01_to_10-22.md#L6)
- [TEAM_LOG_2026-10-01_to_10-22.md#L21](../../TEAM_LOG_2026-10-01_to_10-22.md#L21)
- [TEAM_LOG_2026-10-01_to_10-22.md#L22](../../TEAM_LOG_2026-10-01_to_10-22.md#L22)
- [TEAM_LOG_2026-10-01_to_10-22.md#L24](../../TEAM_LOG_2026-10-01_to_10-22.md#L24)
- [TEAM_LOG_2026-10-01_to_10-22.md#L48](../../TEAM_LOG_2026-10-01_to_10-22.md#L48)
- [TEAM_LOG_2026-10-01_to_10-22.md#L126](../../TEAM_LOG_2026-10-01_to_10-22.md#L126)
- [TEAM_LOG_2026-10-01_to_10-22.md#L127](../../TEAM_LOG_2026-10-01_to_10-22.md#L127)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L79](../../SESSION_NOTES_2026-10-01_to_10-03.md#L79)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L414](../../SESSION_NOTES_2026-10-01_to_10-03.md#L414)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L570](../../SESSION_NOTES_2026-10-01_to_10-03.md#L570)

<details>
<summary>All result files and records (3)</summary>

- [lead_paused_pids_0512.txt](lead_paused_pids_0512.txt) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced) · [Copy provenance](../../../PROVENANCE.json)


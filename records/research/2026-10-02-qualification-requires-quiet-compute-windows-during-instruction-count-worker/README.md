# Qualification requires quiet compute windows during instruction-count and worker stages

2026-10-02T02:27Z · decision · adopted · measured

Because a boundary session took 6x longer under the shared machine's load and projections depend on load, the lead declared quiet windows during qualification chains: no torch screens, trainer workers or native sessions until the chain's log says done. The first window was not kept (trainer workers and a physicist screen kept running; two agents only read the log between their own turns), so the lead paused processes by SIGSTOP; later windows were announced with lines beginning QUIET WINDOW OPEN / CLOSED that watchers parse, after two lanes' watchers misread the first such line.

Recorded measurement:

    boundary session 6x slower under load; S37 chain about 60 minutes; breaches: trainer pool 5 min 50 s into the 07:25Z window, physicist one 3-thread forward 07:25-07:28Z and 03:00-03:17Z

## Programs

- [quiet_watch.sh](quiet_watch.sh)
- [../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/idle_gate.sh](../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/idle_gate.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L83](../../TEAM_LOG_2026-10-01_to_10-22.md#L83)
- [TEAM_LOG_2026-10-01_to_10-22.md#L89](../../TEAM_LOG_2026-10-01_to_10-22.md#L89)
- [TEAM_LOG_2026-10-01_to_10-22.md#L92](../../TEAM_LOG_2026-10-01_to_10-22.md#L92)
- [TEAM_LOG_2026-10-01_to_10-22.md#L129](../../TEAM_LOG_2026-10-01_to_10-22.md#L129)
- [TEAM_LOG_2026-10-01_to_10-22.md#L172](../../TEAM_LOG_2026-10-01_to_10-22.md#L172)
- [TEAM_LOG_2026-10-01_to_10-22.md#L174](../../TEAM_LOG_2026-10-01_to_10-22.md#L174)
- [TEAM_LOG_2026-10-01_to_10-22.md#L175](../../TEAM_LOG_2026-10-01_to_10-22.md#L175)
- [TEAM_LOG_2026-10-01_to_10-22.md#L180](../../TEAM_LOG_2026-10-01_to_10-22.md#L180)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two](../../../submissions/333435/README.md)
- shipped in: [Submission #333481: fitted dials, masked covariance correction and three-request operand loads](../../../submissions/333481/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-qualification-requires-quiet-compute-windows-during-instruction-count-worker) · [Copy provenance](../../../PROVENANCE.json)


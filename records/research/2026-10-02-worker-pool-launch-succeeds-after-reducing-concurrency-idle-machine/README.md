# Worker-pool launch succeeds after reducing concurrency on an idle machine

2026-10-02T05:33Z · lesson · informational · measured

The trainer's attempt to start a three-worker, four-thread refit pool was refused by its session's permission system as interfering with other workloads (load average 46-52 on 16 cores at the time). The lead paused every other lane's job by signal, the trainer waited for the load to fall below about 8 and launched a two-worker pool, which was allowed. Rule: clear the machine first, then launch the smallest pool that does the job, and verify the machine state first-hand before retrying a refused launch.

Recorded measurement:

    refusal at 03:27Z and conditions at 05:07Z: load average 46, six heavy processes of other lanes; pool allowed at 05:33Z with the box at load 5

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L101](../../TEAM_LOG_2026-10-01_to_10-22.md#L101)
- [TEAM_LOG_2026-10-01_to_10-22.md#L126](../../TEAM_LOG_2026-10-01_to_10-22.md#L126)
- [TEAM_LOG_2026-10-01_to_10-22.md#L127](../../TEAM_LOG_2026-10-01_to_10-22.md#L127)
- [TEAM_LOG_2026-10-01_to_10-22.md#L138](../../TEAM_LOG_2026-10-01_to_10-22.md#L138)
- [TEAM_LOG_2026-10-01_to_10-22.md#L140](../../TEAM_LOG_2026-10-01_to_10-22.md#L140)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-worker-pool-launch-succeeds-after-reducing-concurrency-idle-machine) · [Copy provenance](../../../PROVENANCE.json)


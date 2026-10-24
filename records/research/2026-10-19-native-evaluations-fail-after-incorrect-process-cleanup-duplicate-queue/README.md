# Native evaluations fail after incorrect process cleanup, duplicate queue control and excess concurrency

2026-10-19T18:10:00Z · incident · informational · measured

Killing 'orphan' flopscope servers by walking parent PIDs mis-attributed the servers of three running lanes (transient PIDs) and crashed them at their current part (salvaged at 22-25 ranks); a second runner started while four lanes were still running and had to be killed; a third lane beside an in-process probe tripped the guard's hard 50% free rule; a fifth lane tripped the swap rule and died at 16 ranks. Rule written: never kill servers by heuristics while lanes run, only the specific driver, or let runs finish.

Recorded measurement:

    min free 47% with three judges + probe; five judges at 43% free; fifth lane: free 34% < 35% with +300 MB swap growth; k4band_b3p killed at 16 ranks

## Programs

- [../../../instruments/native-evaluation-queue-replaces-dependent-shell-chains-fixes-shell/code/kill_tree_v1.sh](../../../instruments/native-evaluation-queue-replaces-dependent-shell-chains-fixes-shell/code/kill_tree_v1.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5976](../../TEAM_LOG_2026-10-01_to_10-22.md#L5976)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5984](../../TEAM_LOG_2026-10-01_to_10-22.md#L5984)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5987](../../TEAM_LOG_2026-10-01_to_10-22.md#L5987)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6003](../../TEAM_LOG_2026-10-01_to_10-22.md#L6003)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-native-evaluations-fail-after-incorrect-process-cleanup-duplicate-queue) · [Copy provenance](../../../PROVENANCE.json)


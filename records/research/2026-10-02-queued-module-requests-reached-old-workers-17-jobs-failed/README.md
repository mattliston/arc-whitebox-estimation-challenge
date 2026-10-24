# Queued module requests reached old workers: 17 jobs failed before the worker reload

2026-10-02T13:56Z · incident · failed · measured

Requests for a new carrier module were queued 40 seconds before workers reloaded it, producing KeyError for the module entry. The error files were preserved and the requests repeated after reload; no reported result depended on the failures.

Recorded measurement:

    17 requests; queued 40 seconds before reload

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L338](../../TEAM_LOG_2026-10-01_to_10-22.md#L338)
- [TEAM_LOG_2026-10-01_to_10-22.md#L339](../../TEAM_LOG_2026-10-01_to_10-22.md#L339)
- [TEAM_LOG_2026-10-01_to_10-22.md#L340](../../TEAM_LOG_2026-10-01_to_10-22.md#L340)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RESULT.md](../2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-queued-module-requests-reached-old-workers-17-jobs-failed) · [Copy provenance](../../../PROVENANCE.json)


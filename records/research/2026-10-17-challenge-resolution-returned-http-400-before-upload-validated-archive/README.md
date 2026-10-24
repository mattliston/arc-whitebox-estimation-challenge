# Challenge resolution returned HTTP 400 before upload; the validated archive was retried deliberately

2026-10-17T03:38:15Z · incident · failed · measured

A challenge-resolution GET returned a database error after identity lookup and before registration, upload or submission creation. The wrapper counted the attempt, but inspection confirmed no POST or submission ID; a deliberate fresh-tag retry created #336640.

Recorded measurement:

    HTTP 400; wrapper attempt 2/10 for UTC 17; no POST or submission ID.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4787](../../TEAM_LOG_2026-10-01_to_10-22.md#L4787)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4789](../../TEAM_LOG_2026-10-01_to_10-22.md#L4789)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-challenge-resolution-returned-http-400-before-upload-validated-archive) · [Copy provenance](../../../PROVENANCE.json)


# Public leaderboard scanner merges results and alerts on changes exceeding 2%

2026-10-01T18:33Z · tool · informational · measured

A scanner reads every new public submission id into a merged JSONL record with telemetry (walls, residuals, flops, per-network errors), writes a top-12 leader table and a waiter re-scans at fixed hours, writing a log line only when a top-six score moves by more than 2% or a team enters the top six. The merged record fed the leader forensics, the timeout-risk model and the leader-series forecasts.

Recorded measurement:

    ids 333376-333384 read at 18:30Z; re-scan at 02:00 and 08:00 UTC; merged record ids 332296-333451 = 1,010 runs with telemetry

## Programs

- [code/scan_submissions.py](code/scan_submissions.py)
- [code/a5_leaders.py](code/a5_leaders.py)
- [code/a11_waiter.sh](code/a11_waiter.sh)
- [code/a11_rescan.py](code/a11_rescan.py)
- [code/a5_scan_driver.py](code/a5_scan_driver.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4)
- [TEAM_LOG_2026-10-01_to_10-22.md#L13](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L13)
- [TEAM_LOG_2026-10-01_to_10-22.md#L27](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L27)
- [TEAM_LOG_2026-10-01_to_10-22.md#L112](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L112)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-01-public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2) · [Copy provenance](../../PROVENANCE.json)


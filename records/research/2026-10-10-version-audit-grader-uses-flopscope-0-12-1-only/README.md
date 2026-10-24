# Version audit: grader uses flopscope 0.12.1; the only 0.12.1 repricing concerns full/full_like

2026-10-10T03:13Z · finding · informational · measured

The newest flopscope on PyPI and GitHub is 0.12.1 (client, server; whestbench 0.16.1, all 08-29); the grader ran exactly these with numpy 2.2.6 and python 3.10.21 on our candidate generation 42 (S42) page (#333607) and on J2W's #335075, while our native build environments carry 0.12.0 and the repo .venv 0.10.0. The 0.12.1 changelog's only price change is full / full_like with a non-scalar fill value (+50-55% on that op); no disclosure-fix release existed yet. An hourly watcher (a11_waiter_v3.sh running a11_watch.py) was armed to report new PyPI/GitHub releases, organizer forum posts and the challenge page's selection text, alongside the 4-hourly board scans.

Recorded measurement:

    flopscope 0.12.1 (grader) vs 0.12.0 (native builds) vs 0.10.0 (.venv); full/full_like +50-55%; grader numpy 2.2.6, python 3.10.21

## Programs

- [a11_waiter_v3.sh](a11_waiter_v3.sh)
- [a11_watch.py](a11_watch.py)
- [a11_waiter_v2.sh](a11_waiter_v2.sh)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a11_rescan.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a11_rescan.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_leaders.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_leaders.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L524](../../TEAM_LOG_2026-10-01_to_10-22.md#L524)
- [TEAM_LOG_2026-10-01_to_10-22.md#L481](../../TEAM_LOG_2026-10-01_to_10-22.md#L481)
- [TEAM_LOG_2026-10-01_to_10-22.md#L530](../../TEAM_LOG_2026-10-01_to_10-22.md#L530)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-version-audit-grader-uses-flopscope-0-12-1-only) · [Copy provenance](../../../PROVENANCE.json)


# Platform watcher stops reading the forum and checks releases and selection text only

2026-10-13T02:14Z · decision · informational · measured

After the owner's instruction that nothing on the forum is worth reading, the hourly board watch was replaced by a version that no longer fetches forum topics and checks only PyPI / GitHub releases of flopscope and whestbench and the challenge page's selection text, with board scans every four hours.

Recorded measurement:

    scans at 02 / 06 / 10 / 14 / 18 / 22:00Z until 2026-10-18T02:00Z; test run 02:14Z: flopscope 0.12.1, whestbench 0.16.1, no change

## Programs

- [a11_waiter_v5.sh](a11_waiter_v5.sh)
- [a11_watch_v5.py](a11_watch_v5.py)
- [a11_rescan_v4.py](a11_rescan_v4.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_leaders.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_leaders.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1882](../../TEAM_LOG_2026-10-01_to_10-22.md#L1882)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-platform-watcher-stops-reading-forum-checks-releases-selection-text) · [Copy provenance](../../../PROVENANCE.json)


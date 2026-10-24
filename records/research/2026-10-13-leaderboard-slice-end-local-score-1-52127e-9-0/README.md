# Leaderboard at the slice end: local score 1.52127e-9 is 0.092% behind fourth place

2026-10-13T19:57Z · finding · informational · measured

The four-hourly board scans over 10-13 recorded the leaders moving while our five stack is submitted took us from 1.5388e-9 to 1.52127e-9: marius_binner improved three times, oliver_chen and Luna once each, J2W once late, and we stayed fifth, 0.0919% behind Luna and 1.4695% behind third at the 19:55Z scan; the pair Monte Carlo's podium probability at our current score fell from 3.4% to 2.4-2.5% as the leaders moved.

Recorded measurement:

    02:00Z: marius 1.4019e-09 -> 1.3992e-09 (-0.19%, final MSE 1.1339e-08, share 0.1234), oliver 1.5045e-09 -> 1.5043e-09 (final MSE 1.1520e-08, share 0.1305), Luna 1.5301e-09 -> 1.5199e-09 (-0.67%, final MSE 1.3076e-08, share 0.1162); podium at par 3.4% (5.9% if J2W's two files fail together), P(ahead of Luna) 14.8%; 06:00Z: marius 1.3924e-09 (-0.49%, final MSE 1.1284e-08), oliver 1.4989e-09 (-0.36%), ours #335696 1.5322e-09; podium 2.4%; 10:00Z: marius 1.3882e-09 (-0.30%, final MSE 1.1283e-08, share 0.1230), ours #335709 1.5288e-09; podium 2.5%, P(1st) at most 0.2%; 19:55Z: ours #335782 1.52126817409e-09, Luna 1.51987169106e-09 (0.0919% ahead), oliver 1.49891348086e-09 (required reduction 1.4695%), J2W 1.43426793216e-09

## Programs

- [a11_rescan_astra_v2.py](a11_rescan_astra_v2.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1876](../../TEAM_LOG_2026-10-01_to_10-22.md#L1876)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1878](../../TEAM_LOG_2026-10-01_to_10-22.md#L1878)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1958](../../TEAM_LOG_2026-10-01_to_10-22.md#L1958)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1959](../../TEAM_LOG_2026-10-01_to_10-22.md#L1959)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2047](../../TEAM_LOG_2026-10-01_to_10-22.md#L2047)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2048](../../TEAM_LOG_2026-10-01_to_10-22.md#L2048)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2261](../../TEAM_LOG_2026-10-01_to_10-22.md#L2261)

<details>
<summary>All result files and records (5)</summary>

- [a5_leaders_20261013T0200Z.md](a5_leaders_20261013T0200Z.md) · result
- [a5_leaders_20261013T0600Z.md](a5_leaders_20261013T0600Z.md) · result
- [a5_leaders_20261013T1000Z.md](a5_leaders_20261013T1000Z.md) · result
- [report.json](report.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-leaderboard-slice-end-local-score-1-52127e-9-0) · [Copy provenance](../../../PROVENANCE.json)


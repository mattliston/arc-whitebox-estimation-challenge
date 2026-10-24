# Leaderboard and pair model: 5.6% podium probability at current score, 19.0% at a 0.99 score multiplier

2026-10-12T23:11Z · finding · informational · measured

The automatic board re-scan after #335633 found marius_binner at 1.4019e-9 (final-layer MSE 1.1339e-8, budget share 0.1236), J2W 1.4347e-9, oliver_chen 1.5045e-9 and Luna 1.5301e-9 ahead of our 1.5388e-9 (final MSE 1.3394e-8, share 0.1149). The pair Monte Carlo (a16_pair_mc_v4.py, primary #335633 at 4.41% loss risk per 100 networks plus fallback #335618 at 2.5%) gave P(podium) by our score fraction x and showed a same-physics safe fallback is worth under one podium point.

Recorded measurement:

    marius 1.4172e-09 -> 1.4019e-09 (-1.08%), final-layer MSE 1.1339e-08, share 0.1236, residual 0.340 / 0.355 s, wall 86 / 92 s; Luna 1.5344e-09 -> 1.5301e-09 (-0.28%), final MSE 1.3198e-08; ours #335633 1.5443e-09 -> 1.5388e-09 (-0.36%), final MSE 1.3394e-08, share 0.1149, residual 0.323 / 0.342 s, wall 80 / 85 s; P(ahead of Luna) x1.000 31.5%, x0.995 47.8%, x0.99 64.5%; podium x1.000 5.6% (11.0% if J2W's two files fail together), x0.995 10.8%, x0.99 19.0%, x0.985 30.9%, x0.98 45.6%, x0.975 61.0%, x0.970 75.1%; P(1st) about 0; expected place 4.58; same-physics safe at x1.0017: podium +0.1 point at x1.000, +0.2 at x0.99, +0.4 at x0.98

## Programs

- [a16_odds_waiter_v10.sh](a16_odds_waiter_v10.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1779](../../TEAM_LOG_2026-10-01_to_10-22.md#L1779)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1788](../../TEAM_LOG_2026-10-01_to_10-22.md#L1788)

<details>
<summary>All result files and records (2)</summary>

- [a5_leaders_20261012T2304Z.md](a5_leaders_20261012T2304Z.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-leaderboard-pair-model-5-6-podium-probability-current-score) · [Copy provenance](../../../PROVENANCE.json)


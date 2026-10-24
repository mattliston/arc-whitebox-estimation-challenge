# Public runs quantify lost-network frequency against mean residual and wall time

2026-10-02T04:06Z · finding · informational · measured

From 1,010 public runs with telemetry the share of runs that lost at least one network rises with the mean residual (1.9% at 0.30-0.32 s, 13% at 0.36-0.37 s, 56% at 0.37-0.38 s) and with the mean wall (0 of 592 below 85 s, 5% at 95-100 s, 44% above 105 s). The estimate for S37 and S36 was about 2%. A later analyst model from 79 same-file pairs found that the worst residual rose more than 5% in 13% of pairs and never more than 11%; it also estimated a failure risk of about 2% for our current file.

Recorded measurement:

    residual: 0.30-0.32 s 1 of 53, 0.32-0.35 s 2 of 107, 0.35-0.36 s 3 of 44, 0.36-0.37 s 5 of 38 (13%, 95% 6-27%), 0.37-0.38 s 9 of 16, >0.38 s 14 of 15; wall: <85 s 0 of 592, 85-90 s 1 of 120, 90-95 s 3 of 94, 95-100 s 4 of 75, 100-105 s 10 of 68, >105 s 27 of 61; A16: same-file pairs 79, worst residual >5% in 13%, >10% in 4%, never >11%; a lost network = 0.95 per network = 0.019 for one of 50

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L112](../../TEAM_LOG_2026-10-01_to_10-22.md#L112)
- [TEAM_LOG_2026-10-01_to_10-22.md#L113](../../TEAM_LOG_2026-10-01_to_10-22.md#L113)
- [TEAM_LOG_2026-10-01_to_10-22.md#L156](../../TEAM_LOG_2026-10-01_to_10-22.md#L156)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L86](../../SESSION_NOTES_2026-10-01_to_10-03.md#L86)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L421](../../SESSION_NOTES_2026-10-01_to_10-03.md#L421)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-public-runs-quantify-lost-network-frequency-against-mean-residual) · [Copy provenance](../../../PROVENANCE.json)


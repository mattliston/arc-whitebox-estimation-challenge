# Incomplete setup under load raises later-network charged work by 1.2% and requests by 9%

2026-10-01T20:44Z · finding · informational · derived

When machine load hit 27 during a boundary session the estimator's setup ran out of its 4.0 s guard after binding 146 keys and fusing none (setup wall 4.51 s against the 5 s platform limit); the next two networks then billed +1.2% / +0.35% and made +9% / +5% requests. This is the mechanism by which the worst worker of submission #333099, set up a second time while slow, ran at 1.30x residual.

Recorded measurement:

    load 27: 146 keys bound, 0 fused, setup wall 4.51 s; next networks bill +1.2% / +0.35%, requests +9% / +5%; #333099 worst worker residual x1.30

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L31](../../TEAM_LOG_2026-10-01_to_10-22.md#L31)
- [TEAM_LOG_2026-10-01_to_10-22.md#L103](../../TEAM_LOG_2026-10-01_to_10-22.md#L103)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-incomplete-setup-under-load-raises-later-network-charged-work) · [Copy provenance](../../../PROVENANCE.json)


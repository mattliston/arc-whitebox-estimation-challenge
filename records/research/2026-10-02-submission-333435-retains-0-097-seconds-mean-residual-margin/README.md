# Submission #333435 retains 0.097 seconds of mean residual margin

2026-10-02T02:19Z · finding · informational · measured

From the public record of 328 runs with the same bill on every network and mean residual 0.25-0.35 s, the worst network of a run exceeds the mean residual by 0.02 s at the median, 0.04-0.07 s in the worst tenth and 0.10-0.14 s in the worst hundredth. S36 left 0.112-0.117 s of room (exceeded in 1.5% of runs), S37's 2,000 extra requests for the two physics terms gave that back (0.097 s, 1.8%, the same as S35); the wall is not the exposed limit. S37 loses its first network on a worker 1.20x slower in everything, S35 at 1.17x, S36 at 1.26x; each 1,000 requests removed buys 0.007 s of room.

Recorded measurement:

    room: S35 0.096 s / 37.5 s, S36 0.112-0.117 s / 45.8 s, S37 0.097 s / 43 s; worst-network excess over mean residual: median 0.02 s, worst tenth 0.04-0.07 s, worst hundredth 0.10-0.14 s; exceeded S37's room in 1.8% of 328 runs, S36's 1.5%; wall >1.25x mean in 0.7% of 704 runs; first loss on a slow worker: S37 1.20x (residual) / 1.37x (wall), S35 1.17x / 1.26x, S36 1.26x / 1.42x; 1,000 requests = 0.007 s of room

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L80](../../TEAM_LOG_2026-10-01_to_10-22.md#L80)
- [TEAM_LOG_2026-10-01_to_10-22.md#L81](../../TEAM_LOG_2026-10-01_to_10-22.md#L81)
- [TEAM_LOG_2026-10-01_to_10-22.md#L106](../../TEAM_LOG_2026-10-01_to_10-22.md#L106)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-submission-333435-retains-0-097-seconds-mean-residual-margin) · [Copy provenance](../../../PROVENANCE.json)


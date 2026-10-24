# Charged end scans grew with worker age; per-prediction freezing kept scans at 0.8–2.9 ms

2026-10-17T16:15:34Z · finding · adopted · measured

The charged end-of-prediction full scan grew with session age in the repaired controller lineage. Per-prediction heap freezing retained young collections and stabilized the scan; platform residual spread and age effects then narrowed.

Recorded measurement:

    Late/early residual +12.8%, +5.1%, +2.8%, -2.9%, +4.0% in #336741; local 20 end scan 0.8–2.9 ms versus 3→16 ms; #336759 standard deviation / median 2.0% versus 4.6%.

## Programs

- [../../../submissions/336759/estimator.py](../../../submissions/336759/estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4882](../../TEAM_LOG_2026-10-01_to_10-22.md#L4882)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4883](../../TEAM_LOG_2026-10-01_to_10-22.md#L4883)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4890](../../TEAM_LOG_2026-10-01_to_10-22.md#L4890)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

## Related work

- shipped in: [Submission #336741: native residual accessor with inactive fallback: two late-session residual failures](../../../submissions/336741/README.md)
- shipped in: [Submission #336759: per-prediction heap freeze and dtype comparisons: 100 networks passed](../../../submissions/336759/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-charged-end-scans-grew-worker-age-per-prediction-freezing) · [Copy provenance](../../../PROVENANCE.json)


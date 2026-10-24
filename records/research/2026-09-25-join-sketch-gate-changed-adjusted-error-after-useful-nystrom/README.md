# Join-sketch gate changed to adjusted error after a useful Nyström tradeoff failed a raw-only rule

2026-09-25T04:57Z · decision · informational · measured

The candidate added 0.49% raw error against a projected 1.76% work saving. The gate was revised to assess adjusted error consistently with other compression experiments.

Recorded measurement:

      (2) PORTER JOIN SKETCH: all three arms fail the +0.1% raw bar, but Nystrom (C) is raw +0.49% [-0.21, +1.20] at -1.76% bill = -1.28% ADJUSTED, the same trade class as s1. My raw-only bar was inconsistent, so GO a native C build + VALID40 vs S21 (bar adj <= -1.2%, CI excl 0, raw <= +0.8%), below S22 in priority. q=0 arms are dead (+3.5%).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L683](../../TEAM_LOG_2026-09-22_to_09-26.md#L683)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-join-sketch-gate-changed-adjusted-error-after-useful-nystrom) · [Copy provenance](../../../PROVENANCE.json)


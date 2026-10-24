# Local FLOP savings transfer fully to hosted scoring; raw-error gains transfer inconsistently

2026-10-19T15:38:44Z · lesson · informational · measured

Across the day's six fires the bill side of every local 32-rank read landed on the board in full or more (the final-layer row cut -0.295% -> util -0.40%) while raw gains landed at 0 (column squeeze -0.09% -> +0.04%), 3/4 (layer-14 exactness -0.35% -> -0.27%) or 1/4 to 1/2 (the layer-13 crop rung -0.16% -> -0.045% early, -0.087% final). Rule adopted: a piece fires only if its local read is <= -0.2% adjusted with held-out negative and its bill side is neutral or negative.

Recorded measurement:

    kc raw -0.09% -> +0.04%; rc30 bill -0.295% -> util -0.40%, raw +0.119% -> +0.091%; l14x raw -0.35% -> -0.268% paired (31/50 lower), bill +0.20% -> util +0.18%; l1314x raw -0.158% -> -0.045% (35 nets) / -0.087% final (23/50 lower)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5936](../../TEAM_LOG_2026-10-01_to_10-22.md#L5936)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5951](../../TEAM_LOG_2026-10-01_to_10-22.md#L5951)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5956](../../TEAM_LOG_2026-10-01_to_10-22.md#L5956)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5960](../../TEAM_LOG_2026-10-01_to_10-22.md#L5960)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5971](../../TEAM_LOG_2026-10-01_to_10-22.md#L5971)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5978](../../TEAM_LOG_2026-10-01_to_10-22.md#L5978)

<details>
<summary>All result files and records (3)</summary>

- [eval_337142.json](../../../submissions/337142/eval_337142.json) · result
- [eval_337126.json](../../../submissions/337126/eval_337126.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-local-flop-savings-transfer-fully-hosted-scoring-raw-error) · [Copy provenance](../../../PROVENANCE.json)


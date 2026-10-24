# Dead-tail and positive-tail cut thresholds are near the score optimum in both directions

2026-10-11T07:53Z · experiment · rejected · measured

Cutting more rows (2.0) costs far more error than it saves, un-cutting (3.0) gives back error gains 3-6x smaller than their metered FLOPs; the three un-cuts sum to -0.76% of the -1.06% all-tails-exact ceiling.

Recorded measurement:

    old dead tail 2.0 +0.49% (se 0.21) for ~0.1% bill; young dead tail 2.0 +1.49% (se 0.45) for ~0.6%; 3.0: old -0.16%, young -0.50% for ~+1.4% bill, positive -0.10% for +0.5-0.6%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/t94_table.py](code/t94_table.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1039](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1039)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1040](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1040)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1041](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1041)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1045](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1045)

<details>
<summary>All result files and records (2)</summary>

- [cut_counts.jsonl](cut_counts.jsonl) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Tail-repair alternatives rejected: exact tails gain about 1.06% MSE for 2.9% + 1.65% additional FLOPs](../tail-repair-alternatives-rejected-exact-tails-gain-about-1/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-dead-tail-positive-tail-cut-thresholds-near-score-optimum) · [Copy provenance](../../../PROVENANCE.json)


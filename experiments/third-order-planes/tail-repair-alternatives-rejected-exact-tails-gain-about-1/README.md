# Tail-repair alternatives rejected: exact tails gain about 1.06% MSE for 2.9% + 1.65% additional FLOPs

2026-10-11T06:21Z · experiment · rejected · measured

The native old-tier and young dead-tail cuts put back a cheap k3 estimate on the cut rows; computing the tails exactly is worth -1.06% but costs 2.9% + 1.65% of metered FLOPs; the estimate's common scale is at its optimum (per-network best scales scatter from -4.9 to 1.6) and head columns can only carry that estimate.

Recorded measurement:

    tails exact 16 seeds -1.08% pooled (se 0.36, 13 of 16); estimate removed +5.9%; scale x1.25 on 16 +0.94% (6 of 16), x1.5 +2.64%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/t90_table.py](code/t90_table.py)
- [code/t91_table.py](code/t91_table.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L998](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L998)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1018](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1018)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1021](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1021)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Three native approximations explain 91% of the native-versus-research error gap](../../../records/research/2026-10-10-three-native-approximations-explain-91-native-versus-research-error/README.md)
- enables: [Dead-tail and positive-tail cut thresholds are near the score optimum in both directions](../dead-tail-positive-tail-cut-thresholds-near-score-optimum/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-tail-repair-alternatives-rejected-exact-tails-gain-about-1) · [Copy provenance](../../../PROVENANCE.json)


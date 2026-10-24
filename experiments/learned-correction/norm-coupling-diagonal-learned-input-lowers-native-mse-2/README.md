# Norm-coupling diagonal as a learned input lowers native MSE 2.15% on ten networks, all ten improved

2026-10-02T18:41Z · experiment · adopted · measured

The unit-specific residual of the norm-coupling matrix diagonal becomes one standardized input of the learned mean correction, with 128 weights shared across layers 1-14. Its first two line-search steps transfer to the submitted arithmetic, lowering native final MSE 2.15% on all ten fresh networks at 0.002% additional bill; the request reductions make the full candidate bill 0.03% lower.

Recorded measurement:

    zero weights: 105/128 beyond 3 standard errors, mean t squared 54.89, halves cosine +0.98; research 25 ranks step 1 -1.787% (se 0.286, 23/25), step 2 -2.144% (se 0.304, 24/25); submitted-arithmetic research -1.920% / -2.180%; native step 1 -1.55% (8/10), step 2 -2.15% (10/10), 14.0376e-9 vs 14.3459e-9, se 0.28; step 2 vs step 1 -0.61%; 294 more requests, 5.8 Mflop (0.002%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a42_head`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_hc_v1.py](code/fragment_hc_v1.py)
- [code/unit_hc.py](code/unit_hc.py)
- [code/parity_hc.py](code/parity_hc.py)
- [code/judge_ten.py](code/judge_ten.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L420](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L420)
- [TEAM_LOG_2026-10-01_to_10-22.md#L421](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L421)
- [TEAM_LOG_2026-10-01_to_10-22.md#L424](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L424)
- [TEAM_LOG_2026-10-01_to_10-22.md#L426](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L426)
- [TEAM_LOG_2026-10-01_to_10-22.md#L433](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L433)
- [TEAM_LOG_2026-10-01_to_10-22.md#L436](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L436)
- [TEAM_LOG_2026-10-01_to_10-22.md#L437](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L437)
- [TEAM_LOG_2026-10-01_to_10-22.md#L443](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L443)
- [TEAM_LOG_2026-10-01_to_10-22.md#L446](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L446)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [v2r_weights.json](v2r_weights.json) · config
- [column.json](column.json) · config

</details>

## Related work

- shipped in: [Submission #333607: learned norm-coupling diagonal input with two fitted steps and fewer requests](../../../submissions/333607/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-norm-coupling-diagonal-learned-input-lowers-native-mse-2) · [Copy provenance](../../../PROVENANCE.json)


# Nyström join from largest Gram-diagonal rows improves native adjusted error 1.816%

2026-09-25T05:34Z · experiment · adopted · measured

Native forty-network validation found neutral raw error and reduced join work. The row-selected Nyström join was integrated into the combined candidate.

Recorded measurement:

    - 2026-09-24 22:34 lead: S21jn (Nystrom join, 384 top-Gram-diagonal rows) VALID40 vs S21 PASS: adjusted -1.816% [CI -2.212, -1.414], 35/40, raw -0.026% (torch VALID16 had said +0.49%). Not fired alone. S23 = S21s1 + loopR1 params + round-1 dials + s1=0 + Nystrom join, prepared by the porter now (partial byte-rebuilds of each tested file), gated after the S22 and s1=0 verdicts. Bar vs the S21s1 rows adj <= -3.0% if all parts pass. Expected ~-5% -> ~2.40e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/build_join_nystrom.py](code/fable_porter_v1/build_join_nystrom.py)
- [code/fable_porter_v1/stack/stack_S21jn_mem.py](code/fable_porter_v1/stack/stack_S21jn_mem.py)
- [code/fable_porter_v1/valid40_vs_ref_ci.py](code/fable_porter_v1/valid40_vs_ref_ci.py)
- [code/fable_refit_v1/paired_eval.py](code/fable_refit_v1/paired_eval.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L712](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L712)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332246: cross-term removal and Nyström old-cohort joins](../../../submissions/332246/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-nystrom-join-largest-gram-diagonal-rows-improves-native-adjusted) · [Copy provenance](../../../PROVENANCE.json)


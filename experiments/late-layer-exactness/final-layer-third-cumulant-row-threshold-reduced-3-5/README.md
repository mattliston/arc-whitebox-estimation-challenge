# Final-layer third-cumulant row threshold reduced from 3.5 to 3.0: -0.177% adjusted on 32 networks

2026-10-19T09:52:37Z · experiment · adopted · derived

At the final layer the third-cumulant push and the star diagonal run only on rows with |mu/sigma| <= z and other rows get k3 = 0; lowering z from 3.5 to 3.0 saves the k3 push on the dropped rows at a small raw cost because their missing third cumulant is not propagated. On 32 paired ranks the cut is a step at rate 0.4 (a 10-16 11-rank read of raw +0.4% had been pessimistic); hosted, the bill cut transferred in full and more and the raw cost at three quarters.

Recorded measurement:

    l15rc30 e3cb750e vs the base: adjusted -0.177% (se .073, 24/32 lower) = bill -0.295% (every rank) + raw +0.119%; held-out 25 adjusted -0.247% (se .074, 20/25; raw +0.046%); stack stk1 (kc + dcv0015 + rc30) on 28 ranks vs the kc arm: -0.176% (se .073, 20/28; raw +0.205%, bill -0.378%); hosted 49 public nets paired vs #337065: adjusted -0.309% (se .059), MSE +0.091% (se .055, 18/49 lower), flops -0.399%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_l15rc30_sci_v1.py](code/candidate_l15rc30_sci_v1.py)
- [code/candidate_stk1_sci_v1.py](code/candidate_stk1_sci_v1.py)
- [code/candidate_rel_stk7_v1.py](code/candidate_rel_stk7_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5930](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5930)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5931](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5931)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5932](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5932)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5936](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5936)

<details>
<summary>All result files and records (2)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337090: final-layer third-cumulant row threshold 3.0 and own prebinding list](../../../submissions/337090/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-final-layer-third-cumulant-row-threshold-reduced-3-5) · [Copy provenance](../../../PROVENANCE.json)


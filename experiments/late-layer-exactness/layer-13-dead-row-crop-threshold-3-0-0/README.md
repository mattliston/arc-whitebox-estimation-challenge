# Layer-13 dead-row crop at threshold 3.0: -0.095% adjusted; extending to layer 12 gives +0.013%

2026-10-19T14:57:54Z · experiment · adopted · measured

Extending the young-push dead-row crop restore (YNT 2.5 -> 3.0) from layer 14 to layers 13+14 doubles the single-arm raw gain for +0.06% more bill (held-out 21: -0.328%), unlike the old-tail threshold whose layer-13 extension added nothing. Stacked with the layer-14 pieces (l1314x) it is worth -0.095% vs the fired #337126 physics on 32 common ranks, less than the single arms suggested; the layer-12 rung is a wash alone and inside the stack, so the ladder ends at layers 13+14 and the layer-11 arms were dropped unrun.

Recorded measurement:

    ynt1314_30 2034f5b8 (28 ranks vs stk1): adjusted -0.165% (se .115, 19/28 lower) = raw -0.417% + bill +0.249%; held-out 21 -0.328% (se .10, 17/21; raw -0.586%); l1314x 27bae294 vs stk1: adjusted -0.190% (se .12, 18/28) = raw -0.526% + bill +0.333%, held-out 21 -0.321% (raw -0.659%); l1314x vs l14x (32 ranks): -0.095% (se .074, 17/32 lower; raw -0.158%, bill +0.062%); ynt1214_30 vs ynt1314_30 (32 ranks): the L12 rung raw -0.091% for bill +0.105% (adjusted +0.013%); l1214x vs l1314x: +0.013% (raw -0.092%, bill +0.105%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_ynt1314_30_sci_v1.py](code/candidate_ynt1314_30_sci_v1.py)
- [code/candidate_ynt1214_30_sci_v1.py](code/candidate_ynt1214_30_sci_v1.py)
- [code/candidate_ynt1114_30_sci_v1.py](code/candidate_ynt1114_30_sci_v1.py)
- [code/candidate_l1314x_sci_v1.py](code/candidate_l1314x_sci_v1.py)
- [code/candidate_rel_l1314x_v1.py](code/candidate_rel_l1314x_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5959](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5959)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5962](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5962)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5963](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5963)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5966](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5966)

<details>
<summary>All result files and records (2)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337142: young dead-row crop z 3.0 extended to layer 13 on #337126, graded 1.4328308e-9 (-0.027%)](../../../submissions/337142/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-layer-13-dead-row-crop-threshold-3-0-0) · [Copy provenance](../../../PROVENANCE.json)


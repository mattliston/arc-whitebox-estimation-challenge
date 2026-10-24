# Last-join-only rank cut is harmful natively: raw +1.047%, work only -0.688%

2026-09-25T18:07Z · experiment · rejected · measured

The native work saving was much smaller than the research unit-cost model predicted. The schedule choice was reversed and all rank schedules were priced natively.

Recorded measurement:

    2026-09-25 11:07 porter: S27s VALID40 vs S27 rows: adjusted (pooled): +0.351%  95% CI [+0.025%, +0.669%]  wins 15/40; per-net min -1.85% median +0.61% max +2.65% | raw raw (pooled MSE ratio): +1.047%  95% CI [+0.719%, +1.367%]  wins 8/40 (ties 0) | util util: mean 0.140638 vs 0.141611 (-0.688%); pooled util (= pooled flops) -0.688%  | bar adj <= -0.3%: "verdict": "FAIL"
    - 2026-09-25 11:22 lead: SCHEDULE PICK REVERSED BY NATIVE BILLS. S27s (384/384/320) native VALID40 vs S27: raw +1.047% [+0.72, +1.37] (torch +1.07%, fine), but util only -0.688% (the unit-cost model said -1.87%), so adjusted +0.351% [+0.03, +0.67]: HARMFUL. S27r (384/320/320): util -3.167% (model -3.38%). By subtraction, if additive, the L10 cut alone saves about -2.48% of the bill (model -1.51%). Torch RAW is reliable; the torch BILL split is not. Price bills natively from now on.
    2026-09-25 11:23 porter: S27s VALID40 vs S27 rows FAIL: adjusted +0.351% [+0.025, +0.669] 15/40, raw +1.047% [+0.719, +1.367], util only -0.688% (L13 cut) -> S27s OUT (lead). Its chain STOPPED before its residual gate (was waiting on memory, no lock). S27r chain DONE: residual PASS (+0.006 s), setup under concurrent load 2.08-2.62 s with all 410 keys bound. S27s smoke with the leaf-buffers-first fix: 22 leaf buffers first (0.028 s), 414 bound, 2.651 s at load 7-8. Held-out runner stopped to re-plan (S27s arm dropped, S27t arm added; existing rows reused).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/build_r320_router_v2.py](code/fable_porter_v1/build_r320_router_v2.py)
- [code/fable_porter_v1/stack/stack_S27s_mem.py](code/fable_porter_v1/stack/stack_S27s_mem.py)
- [code/fable_porter_v1/valid40_vs_ref_ci.py](code/fable_porter_v1/valid40_vs_ref_ci.py)
- [code/fable_refit_v1/paired_eval.py](code/fable_refit_v1/paired_eval.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1105](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1105)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1107](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1107)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1110](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1110)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-last-join-only-rank-cut-harmful-natively-raw-1) · [Copy provenance](../../../PROVENANCE.json)


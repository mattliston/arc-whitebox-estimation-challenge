# Native two-join rank cut fails the single-panel gate: adjusted -0.343% with confidence interval crossing zero

2026-09-25T17:41Z · experiment · inconclusive · measured

Native raw loss matched the research estimate, but the work saving was smaller. The final schedule choice remained dependent on the independent forty-network evaluation.

Recorded measurement:

    2026-09-25 10:41 porter: S27r VALID40 vs S27 rows: adjusted (pooled): -0.343%  95% CI [-1.014%, +0.319%]  wins 23/40; per-net min -4.81% median -0.61% max +4.33% | raw raw (pooled MSE ratio): +2.917%  95% CI [+2.228%, +3.596%]  wins 3/40 (ties 0) | util util: mean 0.137126 vs 0.141611 (-3.167%); pooled util (= pooled flops) -3.167%  | bar adj <= -0.3%: "verdict": "FAIL"
    - 2026-09-25 10:46 lead: S27r (384/320/320) NATIVE VALID40 vs S27 rows: raw +2.917% [+2.23, +3.60], util -3.167% [-3.185, -3.147], adjusted -0.343% [-1.01, +0.32], 23/40 -> FAIL on VALID40 alone. The raw penalty matches torch (+2.88%); the native bill cut is 6% smaller than the analyst's unit-cost model (-3.17% vs -3.38%). The 80-net native pool (held-out arms) still decides. For L13-only (S27s), scaling that bill shortfall suggests about -1.75% bill and about -0.7% adjusted on VALID40.
    2026-09-25 10:46 porter: S27r VALID40 vs S27 rows: adjusted -0.343% [-1.014, +0.319] 23/40, raw +2.917% [+2.228, +3.596], util -3.17% -> FAIL on VALID40 alone (bar -0.3%); the 80-net native pool decides. S27s control PASS (router all-384 == S27), census running. S28 chain: k-split leaf-buffers-first step added after the key list; S28 params now loopR6 ep1 (ef2e4e73 / dials 507410cc).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/build_r320_router_v2.py](code/fable_porter_v1/build_r320_router_v2.py)
- [code/fable_porter_v1/stack/stack_S27r_mem.py](code/fable_porter_v1/stack/stack_S27r_mem.py)
- [code/fable_porter_v1/valid40_vs_ref_ci.py](code/fable_porter_v1/valid40_vs_ref_ci.py)
- [code/fable_refit_v1/paired_eval.py](code/fable_refit_v1/paired_eval.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1092](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1092)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1098](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1098)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1100](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1100)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-native-two-join-rank-cut-fails-single-panel-gate) · [Copy provenance](../../../PROVENANCE.json)


# Upper-block covariance and reduced requests improve adjusted error 1.094% on 40 networks

2026-09-23T17:00Z · experiment · adopted · measured

The covariance block computation composed with smaller request reductions and the next trunk-refit epoch. It passed accuracy checks and later entered the submitted estimator.

Recorded measurement:

    2026-09-23 10:00 porter: S10L (da7e8f65) VALID40 vs S9 rows: raw -0.072% [CI -0.137, -0.005] (25/40), util -1.022%, adjusted -1.094% [CI -1.156, -1.030] 40/40, 0 errors: PASS. Vs S7 -2.08%. Fire waits on S9 telemetry + engineer2's wall projection.
    2026-09-24 10:03 porter: (recorded) S10' VALID40 vs S9 rows 09-23 10:48: raw -0.082% [CI -0.142, -0.019] 23/40, util -0.700%, adjusted -0.785% 40/40. Next: S11 = S10L recipe + B2 epoch-1 refit (lead GO).
    2026-09-24 10:23 porter: S11 (acf67b9d = S10L recipe + B2 ep1) VALID40 vs S9 rows: adjusted -1.289% [CI -1.382, -1.200] 40/40, raw -0.271% 30/40, util -1.022%: PASSES all bars; ep1 vs ep0 native -0.199% (torch -0.23). Ready to fire.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/stack/stack_S10L_so3_lean_B2_mem.py](code/fable_porter_v1/stack/stack_S10L_so3_lean_B2_mem.py)
- [code/fable_porter_v1/valid40_vs_ref_ci.py](code/fable_porter_v1/valid40_vs_ref_ci.py)
- [code/fable_refit_v1/paired_eval.py](code/fable_refit_v1/paired_eval.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L301](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L301)
- [TEAM_LOG_2026-09-22_to_09-26.md#L311](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L311)
- [TEAM_LOG_2026-09-22_to_09-26.md#L317](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L317)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332182: upper-block covariance products, lean views, and an epoch-1 refit](../../../submissions/332182/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-upper-block-covariance-reduced-requests-improve-adjusted-error-1) · [Copy provenance](../../../PROVENANCE.json)


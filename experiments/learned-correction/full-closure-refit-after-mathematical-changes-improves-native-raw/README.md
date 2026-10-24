# Full closure refit after mathematical changes improves native raw error 0.50%

2026-09-23T11:12Z · experiment · adopted · measured

A completed joint closure refit produced a small held-out gain that transferred to the native estimator.

Recorded measurement:

    - 2026-09-23 04:12 [physicist] REFIT LINE END STATE (leadergap/RESULT.md sec F-G): (a) S2 closure refit final VALID16 -0.58% (-0.37/-0.57/-0.58 by epoch, 12/16), saturated; params_epoch2 into S6. (b) tier-cut refit dead. S5e-native refit DESIGNED, NOT STARTED (gated on S6 VALID40: skip if S6 keeps >= 0.4%; else a 15-min torch VALID16 mismatch-vs-overlap check; if mismatch: warm start, 'full' groups, lr 1e-4, 2 epochs, ~8-10 thread-hours).
    2026-09-23 04:24 porter: S6 (stack_S6_mem.py, sha 2b4372a7...) = S5e + refit epoch2 + diet mem,yright,arena: refit r35/r100 MSE -0.43/-0.82% at identical flops; diet bit-exact -0.04/-0.08% flops; VALID40 vs S5e raw 0.99497 (33/40), adj 0.99454 (34/40); vs bank adj 0.89591. Passes the bar.
    2026-09-23 04:30 porter: S6 panel beats S5e on both halves (panel8 0.9963, virgin6 0.9877). S6b (stack_S6b_mem.py, a514c21b...) = S6 + --batch: MSE bit-identical, flops +0.38% cold r35 / identical warm r100; validate OK.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1/stack`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/stack_S6_mem.py](code/stack_S6_mem.py)
- [code/stack_S6b_mem.py](code/stack_S6b_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L187](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L187)
- [TEAM_LOG_2026-09-22_to_09-26.md#L189](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L189)
- [TEAM_LOG_2026-09-22_to_09-26.md#L192](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L192)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/RESULT.md) · record

</details>

## Related work

- shipped in: [Submission #332005: joint closure refit, batched products, and memory-layout changes](../../../submissions/332005/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-full-closure-refit-after-mathematical-changes-improves-native-raw) · [Copy provenance](../../../PROVENANCE.json)


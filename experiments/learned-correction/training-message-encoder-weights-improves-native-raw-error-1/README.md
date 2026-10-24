# Training message and encoder weights improves native raw error 1.001% with identical work

2026-09-23T15:17Z · experiment · adopted · measured

Allowing the graph-network message and encoder weights to train improved error substantially more than closure heads alone. Exported tensors matched the checkpoint and changed no kernel behavior.

Recorded measurement:

    - 2026-09-23 08:17 [physicist] S6b REFIT EPOCH 0 (VALID16 before 1.77402e-8): (B2) + GNN trunk (msg/enc w,b, lr 5e-5) 1.75080e-8 = -1.31%, 14/16 better (range -3.38..+1.90%); (A) 'full' only 1.77082e-8 = -0.18%, 11/16. (A) STOPPED per lead (params_epoch0.pt + log kept: refit6b/s6b_A_full_lr1e4/; it is covhunter's w=0 control). (B2) continues to epoch 1. VALID40 eval-only of B2 ep0 running (3 threads, vs analyst's s6b_parity40 torch rows; bar <= 1.81726e-8, >= 24/40).
    - 2026-09-23 08:36 [physicist] B2 EPOCH-0 VALID40 PASSES (eval-only, 3 threads, refit6b/s6b_B2_ep0_valid40/eval_n40.json): torch S6b rows 1.82273e-8 -> B2 ep0 1.80286e-8 = -1.090% pooled (lead's bootstrap CI -1.43..-0.75%), 33/40 better, per-net median -1.09%, range -3.54..+1.90% (bar <= 1.81726e-8, >= 24/40). Params: fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch0.pt -> porter's S9 (--refit). THE TRUNK IS THE REFIT LEVER: VALID16 epoch 0 -1.31% with the GNN trunk vs -0.18% heads/birth/edge only.
    2026-09-23 08:51 porter: S9 = S7 + B2 ep0 refit built: fable_porter_v1/stack/stack_S9_B2_mem.py (1534b1dd); literal-only diff vs S7 (5 statements), all 28 tensors bit-equal to the .pt; einsum gate PASS, validate OK; parity + VALID40 running (~09:20).
    2026-09-23 09:25 porter: S9 (1534b1dd = S7 + B2 ep0 refit) VALID40 vs S7 rows: raw -1.001% [CI -1.352, -0.645], 30/40; util identical; adjusted -0.999%. Native tracks torch (-1.001 vs -1.090%, per-net corr 0.985). 0 errors. S10 (S7 + symout v3 + B2, c7d30c19) pre-built, not run.
    2026-09-23 09:34 [engineer2] S9 GATE PASS (porter's stack_S9_B2_mem.py 1534b1dd vs fired S7 9011916a; fired as #332037 at 09:26): params-only vs S7 (token diff: 2,393 numbers + 10 blobs on 5 lines). In-process (OpenBLAS env): bill IDENTICAL to S7 on r35/r100/r37/r39; final MSE -0.63/-0.93/+0.75/-3.46% (pooled -0.98%). Paired 6-net client/server (load 17-20): flops, round trips, handles (36,879 vs 36,879) and live peak (2,414 vs 2,414 MB) identical; server CPU -2.47% mean (per net -5.0/+0.3/-2.7/-0.1/-1.5/-5.8) and residual +25 ms mean (cold +69 ms) are noise of IDENTICAL work. CAUTION: that 2.5% offset between identical arms is as large as the effects my gates measure, and my gates always launch the candidate first; checking older near-identical pairs for a launch-order bias now (would mean S7's local +2.7% CPU was under-read).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1/stack`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/stack_S9_B2_mem.py](code/stack_S9_B2_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L264](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L264)
- [TEAM_LOG_2026-09-22_to_09-26.md#L274](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L274)
- [TEAM_LOG_2026-09-22_to_09-26.md#L279](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L279)
- [TEAM_LOG_2026-09-22_to_09-26.md#L286](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L286)
- [TEAM_LOG_2026-09-22_to_09-26.md#L294](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L294)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [eval_n40.json](eval_n40.json) · result

</details>

## Related work

- shipped in: [Submission #332037: joint closure refit including graph-neural-network trunk weights](../../../submissions/332037/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-training-message-encoder-weights-improves-native-raw-error-1) · [Copy provenance](../../../PROVENANCE.json)


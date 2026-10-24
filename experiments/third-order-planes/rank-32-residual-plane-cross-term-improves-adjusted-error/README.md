# Rank-32 residual-plane cross term improves adjusted error 1.782% on forty networks

2026-09-25T04:34Z · experiment · adopted · measured

The small star cross term tolerated strong low-rank approximation despite its broad spectrum. Native validation and paired runtime checks supported submission.

Recorded measurement:

      - [LR] STAR s1 = (push o w^T) w (~4.4%): content = 15% (median) of the star term, which is ~6% (1.6-7.1%) of K21; pw spectrum flat (E(128) .60) -> TESTED: r32 raw +0.43% (per net +0.53/-0.52/+0.58/+1.14) at native bill ~-2.06% -> PASSES the bar narrowly; r64 +0.29% at -0.1% and r128 +0.40% at +1.9% fail the bill bar; full-rank control (r1024) -0.02%. Raw is flat in r (flat spectrum, small content). Estimated adjusted ~-1.6%; 4-net SE on raw ~0.35%. Asking the lead before any native build.
    - 2026-09-24 21:35 lead: ENGINEER2 LOW-RANK SCREEN of S21 (E-62 recipe, 37/39/35/100, no refit, learned ON; full-rank controls unbiased).
      - Old tier at r128: +90% raw. KILLED (the old plane carries 23-62% of the read plane).
      - Newborn push and W5 base: not star-like, not tested.
      - Star s1 = (push o w^T) w at r32: raw +0.43% (se ~0.35%) at ~-2.06% bill, a narrow pass. r64/r128 don't pay (flat spectrum).
    2026-09-24 21:48 [engineer2] s1 r32 ON S21 ALONE (lead): fable_engineer2_v1/lrroute/S21_s1lr_mem.py 30aea66e = S21 eddbcfcd + s1lr_fragment_v2.py (s1 = L (R w) from pw's rank-32 factors, unrouted _lr_factor; C w single at depth 6). validate/einsum PASS, no new imports. Parity (in-situ bill) -> VALID40 vs S21 rows (porter harness, yielding to S22) -> paired residual gate queued (lrroute/run_s1only.sh).
    - 2026-09-24 21:51 lead: s1 r32 on plain S21 (engineer2 lrroute/S21_s1lr_mem.py 30aea66e): 4-net parity bill -2.00% (in-process), MSE -0.006%, adjusted -2.00%. VALID40 vs S21 rows running (bar adj <= -1.2%, CI excl 0, raw <= +0.8%), then residual gate. If it passes: fire S21+s1 as a bank; S22 is re-based to S21+s1+round-2 (porter preparing the recipe variant with a byte-check vs 30aea66e).
    - 2026-09-24 22:02 lead: s1 r32 VALID40 PASS (engineer2 S21_s1lr_mem.py 30aea66e vs S21 rows): adjusted -1.782% [CI -2.052, -1.502], 39/40; raw +0.249% [-0.023, +0.529] (15/40); util -2.030%. Residual gate running since 22:01. S22 flag -> S21s1 (porter). If the residual gate passes, fire S21+s1 as a bank (expected ~2.537e-9); the porter stages fire_candS21s1.
    2026-09-24 22:12 [engineer2] S21s1 (fired #332236, 30aea66e) gate record: VALID40 vs S21 rows adjusted -1.782% [-2.052, -1.502] 39/40, raw +0.249% [-0.023, +0.529], util -2.030% -> PASS (bar -1.2%, raw <= +0.8%); paired residual gate cold +0.003 s (warm +0.006/+0.002), requests -182/net, server CPU -0.6..-0.8 s/net, bill -2.02..-2.04% -> PASS; parity (in process) bill -1.997%, MSE -0.006%. TORCH s1 COUNTERPART sent to the physicist for loop3: in lrstar_torch_hook.star() replace pw @ w by Qs @ ((Qs.T @ pw) @ w) with Qs = basis(pw) (detached, same seeded Omega/k48/q1/r32; star_diagonal unchanged). Port certificate: lrroute/s1_cert_r37.npz (pw, w, native s1_r32, exact s1 at the first 3 star layers of net 37); a plain port of the recipe reproduces native s1_r32 to ~1e-6; truncation there: ||s1_r32 - s1||/||s1|| ~0.90, star off-diagonal change 8.8-12.6%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/lrroute`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/S21_s1lr_mem.py](code/S21_s1lr_mem.py)
- [code/s1lr_fragment_v2.py](code/s1lr_fragment_v2.py)
- [code/run_s1only.sh](code/run_s1only.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L659](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L659)
- [TEAM_LOG_2026-09-22_to_09-26.md#L660](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L660)
- [TEAM_LOG_2026-09-22_to_09-26.md#L661](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L661)
- [TEAM_LOG_2026-09-22_to_09-26.md#L662](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L662)
- [TEAM_LOG_2026-09-22_to_09-26.md#L663](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L663)
- [TEAM_LOG_2026-09-22_to_09-26.md#L670](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L670)
- [TEAM_LOG_2026-09-22_to_09-26.md#L676](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L676)
- [TEAM_LOG_2026-09-22_to_09-26.md#L686](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L686)
- [TEAM_LOG_2026-09-22_to_09-26.md#L696](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L696)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332236: rank-32 evaluation of the residual-plane cross term](../../../submissions/332236/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-rank-32-residual-plane-cross-term-improves-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)


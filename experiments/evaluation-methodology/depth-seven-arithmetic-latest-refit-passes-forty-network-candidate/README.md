# Depth-seven arithmetic with the latest refit passes a forty-network candidate gate

2026-09-25T20:26Z · experiment · informational · measured

The combined depth-seven packing, square-product rule, and epoch-1 refit pass the forty-network adjusted gate, exact-parameter check, residual check, and setup cap. This source is staged as a candidate; the record does not show it being submitted.

Recorded measurement:

    2026-09-25 13:26 porter: S29 BUILT stack/stack_S29_mem.py 224f870aa9acf32f (loopR7s ep1 literals ALL 28 BIT-EQUAL; einsum vs S16/S23/S24/S27/S28 PASS; validate PASS; depth-7 gate: 465 single (460 list + 5 prepended) + 24 leaf = 489 intended, no duplicates). Re-census list vs S28: kept 445, swapped 15 (14 of 15 dropped depth-6 keys unused under d7cp+d7sq on all 12 census nets; added 11 depth-7 packet/896 keys + 4 L4/L5). engineer2 froze a copy for the S28-vs-S29 server-CPU wall pair; asked the lead to slot it between S29's smoke and VALID40.
    2026-09-25 13:29 porter: S29 (224f870a) built = S24 + j3cut + ksplit_fragment_v5w + capraise480_fragment + refit s6b_loopR7s_joint_3ep/params_epoch1.pt (0a8fbdad7733a076) + schedule {"7":384,"10":320,"13":384} + group 5 (b769aa28266eb4a3) + k-split leaf buffers first (setup) + guard 4.0 s + d7cp d7cp_fragment_v2 (64210b6451085693) + d7sq d7sq_fragment (8b863bbd31e3e560) + key list + treuse (last); literals bit-equal to the refit .pt; k-split settings effective; einsum/validate + treuse identity PASS; KS CHUNK-KEY COVERAGE: 51/58 chunk keys of the 19 split reads are in _PB_SINGLE (7 bind lazily); smoke next.
    2026-09-25 13:31 porter: S29 smoke OK (wall 3.168 s at load { 11.00 9.00 8.67 }; bound 489 entries); VALID40 vs S28 rows STARTED (bar adj <= -0.3%, CI excl 0).
    2026-09-25 13:34 porter: S29 smoke PASS (setup 3.168 s at load 11/9/8.7; bound 489/489 intended; 465 singles fused, guard cut nothing; util r35 0.133728 / r100 0.133476 = -0.94% / -0.79% vs S28 smoke; all outputs finite). VALID40 vs S28 rows running since 13:31 (~14:10). engineer2's wall pair takes the post-VALID40 gap (before the residual). Fallback S29n (no d7sq) chain + args ready.
    2026-09-25 13:43 porter: S29 PASSES the VALID40 bar vs S28: adjusted (pooled): -0.978%  95% CI [-1.142%, -0.820%]  wins 38/40; per-net min -2.68% median -0.99% max +0.25% | raw raw (pooled MSE ratio): -0.134%  95% CI [-0.288%, +0.014%]  wins 27/40 (ties 0); residual gate next.
    2026-09-25 13:53 porter: S29 residual gate vs S28: S29 cold delta vs S28 +0.006 s (rule <= +0.010 s) -> RESIDUAL GATE PASS | setup under concurrent load: S29 walls [3.04, 3.013, 2.945] s, bound [489, 489, 489] (gate <= 4.1 s: PASS) | S28 walls [2.988, 3.063, 2.849] s, bound 
    2026-09-25 13:53 porter: S29 STAGED: fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py (224f870a; FIRE_CHECKS.txt; VALID40 adjusted (pooled): -0.978%  95% CI [-1.142%, -0.820%]  wins 38/40; per-net min -2.68% medi; residual PASS).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/stack/stack_S29_mem.py](code/fable_porter_v1/stack/stack_S29_mem.py)
- [code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py](code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1288](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1288)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1291](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1291)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1295](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1295)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1296](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1296)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1297](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1297)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1298](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1298)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1299](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1299)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-depth-seven-arithmetic-latest-refit-passes-forty-network-candidate) · [Copy provenance](../../../PROVENANCE.json)


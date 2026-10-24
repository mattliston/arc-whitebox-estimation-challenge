# Inner-dimension split revision passes the paired server-CPU gate

2026-09-25T19:57Z · experiment · adopted · measured

The final split revision saves arithmetic and server operations while a full paired run shows a small user-CPU improvement. This reverses the earlier noisy warm-test decision and qualifies the change for the next bundle.

Recorded measurement:

    - 2026-09-25 12:57 lead: S29 CONTENTS FIXED: S28 + d7cp v2 (64210b64) + d7sq (8b863bbd) + loopR7s best-epoch literals (if the physicist's check clears 0.15 pt; the analyst's VALID40 has epoch 1 at -0.179% raw vs the S28 params, se .072), else loopR7 epoch 1. OUT: k-split v6 (parity mean -0.11%, net 35 +0.19%: under the bar and mixed); the 7 chunk keys (net-35-specific, about 0.09% of one net). Chain: fable_porter_v1/fire_chain_S29.sh (key census on the pre-list file with depth-7 applied, value rule capped at 460, ksfirst -> guard 4.0 -> d7 -> treuse). Gates: VALID40 vs S28 rows adjusted <= -0.3%; paired residual vs S28 with per-net server CPU for the wall ledger (projected slowest at most 90 s).
    - 2026-09-25 13:05 [engineer] E-77 BIG-K SPLIT (v6) SIZED AND KEPT OUT OF S29 (lead): bench (e77_sqsplit_bench_s27.log, OpenBLAS, warm) v2 leaf-sum split vs read6 on 4 read6 shapes -1.7..-2.8% bill, -23..-130 requests, backend x0.92-1.21; the -1.3% model was wrong (read6 already depth 6; census bills included first-time binds). Warm path census S27t net 100 (e77b_path_census.py): read6 4 calls 19.0 GF; wide pushes 48 GF = engineer2's frontier (partition agreed: chunk-needing products mine, whole-fit p6 theirs). ksplit_fragment_v6_final.py (bda9a0ab222ba4a5) parity vs v5w: bill +0.19/-0.32/-0.20/-0.12% (mean -0.11%, net 35 pays first-time binds), requests -75..-552, MSE mean +0.09%: under the 0.2% bar, OUT. E-78 SPLIT u/v OLD-TIER FRAMES, native scope (paper; S28 net 100 census): frame costs 74.5 GF = reconstruct 35.0 + inner 28.3 + rank-rank 11.2 (the 87 GF _shared_old384_push is the exact recent cohorts, untouched); price r_u 256 ~-2.0..-2.5% (best -2.73%), r_u 192 ~-4.0..-4.5%; the inner MUST use the per-frame form lambda[2uv F_u + (u^2+v^2/4) F_v] (naive both-frame projection = +5.6% loss); cores split cleanly (equal pair weights). Build waits for covhunter's torch raw.
    2026-09-25 13:06 porter: S30 queue item (engineer, lead's condition): k-split v6 = fable_engineer_v1/ksplit_fragment_v6_final.py bda9a0ab222ba4a5 REPLACES v5w (k-split arg) if its user-CPU pair lands <= +2%; key census on the v6 base (the S29 chain does this automatically). engineer2 closed p6; partition agreed.
    2026-09-25 13:29 porter: S30 INPUT CLEARED (engineer): k-split v6_final bda9a0ab222ba4a5 replaces v5w (user-CPU pair vs v5w on 6 nets -0.57% (se .16), bill -0.158%, -305 ops/net; wall projection on S27 73.7 -> 73.3 s; files fable_engineer_v1/optime_e77b_ucpu_v6.json vs _v5w.json). S30 = S29 (or its winner) with the k-split arg = v6_final; the chain's key census runs on the v6 base automatically.
    - 2026-09-25 13:29 lead: k-split v6 CLEARS the S30 bar (engineer; OpenBLAS user-CPU pair vs v5w, 6 nets): user CPU -0.57% (se .16), bill -0.158%, -305 op calls/net; projected expected slowest on #332295 73.7 -> 73.3 s. S30 candidates so far: v6 (ksplit_fragment_v6_final.py bda9a0ab, key census on the v6 base) + the L15-only u/v split read (engineer2 uvp_fragment_v0c.py 287ffe5f, -0.15%, L0-14 bit-identical) + loopR8s if it clears 0.15 pt + two-frame split if its native bill pays.
    - 2026-09-25 13:30 [engineer] v6 CLEARED FOR S30 (lead's <= +2% user CPU condition): OpenBLAS user-CPU pair v6 vs v5w (6 nets): user CPU -0.57% (se 0.16%), bill -0.158%, op calls -305/net; wall on S27 #332295 73.7 -> 73.3 s central. ksplit_fragment_v6_final.py bda9a0ab222ba4a5 replaces v5w in S30, key census on the v6 base. PER-JOIN u-RANK BILL SPLIT (for covhunter's split-frame targeting; S28 net 100, rank-linear old-tier products, best case before overheads): L7-9 (rank 384) 16.7 GF, -0.47% per 64 of r_u cut; L10-12 (320) 22.0 GF, -0.74% per 64; L13-15 (384) 24.6 GF, -0.69% per 64. With covhunter's missed-u-energy (L7 worst), candidates su7/su10/su13 = 384/256/192 (-2.81% best, ~-2.3..-2.5% net) or 320/256/192 (-3.28% best, ~-2.7..-3.0% net). engineer2 builds; I review kernel fit.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/fire_chain_S29.sh](code/fable_porter_v1/fire_chain_S29.sh)
- [code/fable_engineer_v1/e77b_path_census.py](code/fable_engineer_v1/e77b_path_census.py)
- [code/fable_engineer_v1/ksplit_fragment_v6_final.py](code/fable_engineer_v1/ksplit_fragment_v6_final.py)
- [code/fable_engineer2_v1/uvsplit/uvp_fragment_v0c.py](code/fable_engineer2_v1/uvsplit/uvp_fragment_v0c.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1240](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1240)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1247](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1247)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1249](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1249)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1292](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1292)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1293](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1293)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1294](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1294)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [e77_sqsplit_bench_s27.log](e77_sqsplit_bench_s27.log) · log
- [optime_e77b_ucpu_v6.json](optime_e77b_ucpu_v6.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-inner-dimension-split-revision-passes-paired-server-cpu-gate) · [Copy provenance](../../../PROVENANCE.json)


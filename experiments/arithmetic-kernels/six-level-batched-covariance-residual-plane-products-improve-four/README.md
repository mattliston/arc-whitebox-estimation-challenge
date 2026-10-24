# Six-level batched covariance and residual-plane products improve four-network adjusted error 1.557%

2026-09-24T21:47Z · experiment · adopted · measured

The enlarged batched tree region enabled one more multiplication level. An independent wrapper verified exact agreement, and measured runtime remained within the newly available wall margin.

Recorded measurement:

    - 2026-09-24 14:47 [engineer] DEPTH LEAD for engineer2's re-pricing: the batched covariance/star products (P=2, 1024^3, ~31 calls/net) are pinned at depth 5; on the tree kernel depth 6 bills -7.4% per call (2.340 -> 2.167 G), +15 ops, user CPU 279 -> 355 ms, max rel error vs float64 3.2e-5 -> 6.7e-5 (one more level). Per net ~-5.3 G = -1.45% of the bill for ~+2.4 s local user CPU; needs a depth-6 batched region (P=2 trees 60.2M floats per side), _mm_batch to accept depth 6, and a VALID40. Sent to engineer2 (depth owner).
    2026-09-24 15:15 [engineer2] DEPTH-6 UNDER THE ALT BASIS (in-situ shadow on S15, r35 cold + r100 warm; native bit-identical and bill-equal to parity; altbasis/depth6_ab_shadow_v2.py, depth6_price.py): only the batched P=2 1024^3 covariance/star products pay. They give bill -1.504%/net for local user +2.61 s, +422 op calls = platform wall +3.83 s (band +2.9..+4.7) and residual +2.44 ms (0.39% bill per wall-s). Per call: -0.196 G (-8.5%), +93 ms (+76%), +15 ops. Next five classes (single 1024^3, 384x832x4096, 896x384x4096, W5 squares, 512x1024x1024 x14) add -0.35% for +1.5 s; everything eligible -2.12% for +8.2 s / +8.8 ms. Reads with k 3520-6144, n=384 are bill-positive at L6. MSE of the batched move alone (depth6_batched_parity.py, 35/100/37/39): bill -1.502%, MSE -0.056%, adjusted -1.557%. Region: P=2 trees need 60,236,288 floats/side; suggest _IPT_REGION[6] = (60236288, 120472576, 120472576), not 60.2M per side (keeps colpack's packets in the tree kernel). Table handed to the engineer (depth rule owner).
    - 2026-09-24 15:15 [engineer] --bb6 BUILT on S15 (fable_engineer_v1/bb6_fragment.py appended to fable_porter_v1/stack/stack_S14_ab_mem.py e313ab77 -> candS16bb6_mem.py a4a1e0f7): the batched 1024^3 covariance/star products run at six levels through engineer2's depth-generic alt-basis _bb_product; the depth-6 A region grows 30.1M -> 90.4M floats (P<=3 trees; 331.3M in all). In process vs S15: bill -1.482% (r35 cold) / -1.525% (r100 warm), ops +423/+420, predictions max rel change 2.0/2.1e-6 (rounding level), user CPU +2.6/+3.0 s unpaired at load 16. Einsum gate PASS, validate PASS. Queued under the lock: cs_probe identity, porter's paired user-CPU vs S15 (6 nets), residual_probe cold (S15 vs S16bb6).
    2026-09-24 15:31 [engineer2] S16 VERIFIED (engineer's candS16bb6_mem.py a4a1e0f7 = S15 e313ab77 + bb6_fragment.py, byte-checked): in-process parity on 35/100/37/39 BIT-EXACT on all layers (max abs 0.0) and bill-equal to the flop vs my independent batched-depth-6 wrapper run (par4_d6bb_S15), so the enlarged depth-6 A region (90.4M) moved nothing else. vs S15: bill -1.502%, MSE -0.056%, adjusted -1.557%; vs S14 -2.69%. Wall about +3.8 s central from the shadow (S16 expected worst ~73 s); S16 projection runs when the engineer's e_S16bb6 vs e_S15ref pair lands.
    - 2026-09-24 15:46 [engineer] S16 (--bb6, candS16bb6_mem.py a4a1e0f7) GATES: porter-tool pair vs S15 (6 nets) flops -1.511%, user +7.84% (204.8 vs 189.9 s; ~+3.7 s expected worst at 0.47 s/%); predictions max rel 2.0/2.1e-6 vs S15; client/server byte-identical to in-process (r35/r100); einsum PASS, validate PASS; engineer2 verified bit-exact vs their independent wrapper. residual_probe (client/server, sequential, load 16): only _bb_product moves, +6.1 ms cold / -0.1 ms warm (one-time binding of the fixed-shape depth-6 batched trees); S15 was +0.024 s cold vs S13, so the porter's paired gate decides. Ready for the porter's VALID40 vs S15 rows.
    2026-09-24 16:14 [engineer2] S16 WALL (engineer's pair e_S16bb6 vs e_S15ref, 6 nets, done 15:41): user +7.84% (se 0.35%; +1.97..+2.82 s/net, shadow said +2.61 s), bill -1.51%, op calls +420/net, no paging. project_wall_v3 (base S13; S14+ab+bb6 additive): bill -3.64%, expected worst 74.7 / 72.9 / 71.1 s, room 25.3 / 27.1 / 28.9 s. Residual +420 requests ~ +2.4 ms/net (porter's paired gate vs S15 in the S16 chain measures it). Engineer's request-cut fragment over my ab functions: I agreed (overriding fragment, not an edit; my verification = 4-net bit-exact parity vs par4_engD6).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/altbasis/depth6_ab_shadow_v2.py](code/fable_engineer2_v1/altbasis/depth6_ab_shadow_v2.py)
- [code/fable_engineer2_v1/altbasis/depth6_price.py](code/fable_engineer2_v1/altbasis/depth6_price.py)
- [code/fable_engineer2_v1/altbasis/depth6_batched_parity.py](code/fable_engineer2_v1/altbasis/depth6_batched_parity.py)
- [code/fable_engineer_v1/bb6_fragment.py](code/fable_engineer_v1/bb6_fragment.py)
- [code/fable_porter_v1/stack/stack_S14_ab_mem.py](code/fable_porter_v1/stack/stack_S14_ab_mem.py)
- [code/fable_engineer_v1/candS16bb6_mem.py](code/fable_engineer_v1/candS16bb6_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L421](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L421)
- [TEAM_LOG_2026-09-22_to_09-26.md#L434](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L434)
- [TEAM_LOG_2026-09-22_to_09-26.md#L435](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L435)
- [TEAM_LOG_2026-09-22_to_09-26.md#L445](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L445)
- [TEAM_LOG_2026-09-22_to_09-26.md#L452](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L452)
- [TEAM_LOG_2026-09-22_to_09-26.md#L470](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L470)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332210: batched six-level covariance and residual-plane products](../../../submissions/332210/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-six-level-batched-covariance-residual-plane-products-improve-four) · [Copy provenance](../../../PROVENANCE.json)


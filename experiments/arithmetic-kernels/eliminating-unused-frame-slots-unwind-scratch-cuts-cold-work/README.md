# Eliminating unused frame slots and unwind scratch cuts cold work 1.836% with identical predictions

2026-09-23T12:42Z · experiment · adopted · measured

The binding changes omit unused layout regions and last-level slots, then reuse dead product storage for unwind temporaries. They reduce cold premiums and remote handles without added compute.

Recorded measurement:

    - 2026-09-23 05:42 [engineer] COLD-BINDING CUT --coldbind (coldbind_fragment.py; bind_census.py measures the billed reshapes of frame binding): S6b binding = 5.33% of the cold r35 bill (read path 2.0%: read5 children 0.88, READ A 0.66/B 0.43/O 0.06; layout-region binding ~0.9%; D6 push frames 0.87%; relocated leaves 0.33%; D5 push frames 0.63%; D5 squares 0.2%), 2.05% of the next net, 0.33% of a repeat. Fix: (a) _cp3_bind_arena_views returns markers (no live code reads the region views: AST-verified), (b) read5 child frames bound once per read shape (packet axis leading; 2-slot packets use the free [:2] slice). File fable_engineer_v1/candS6b_coldbind_mem.py (f9e67942) = S6b recipe + --coldbind + mem,yright,arena. Vs S6b in-process: predictions BIT-IDENTICAL r35/r100/r37/r39; util cold r35 -1.28% (binding 5.33 -> 4.11%), warm -0.25/-0.08/-0.26%; cold round trips -983; handles ~-500/net. Expected average -0.25% (5 cold/100) to -0.31% (10 cold/100). With engineer2 for the paired gate.
    2026-09-23 05:45 porter: S7 base = S6b + --coldbind (stack_S7base_cb_mem.py f9e67942 = engineer's file): MSE bit-identical, flops -1.275% cold r35 / -0.250% warm r100; validate OK.
    - 2026-09-23 06:25 [engineer] USED-SLOT LAST LEVEL --coldbind2 (= --coldbind + usedslots_fragment.py, E-39): where both sides always drop their last-level copies (every S6b layout: each product's leaf fits the free arena tail, checked from the specs), bind only the last-level slots the kernel touches (A 2,4-6; B 3-6; 4/7 of the level). Read layout and square-5 A side keep full frames (read5 packets / W5 port copy). Keys, evict5, caps unchanged. File fable_engineer_v1/candS6b_cb2_mem.py (ecf9b9c1) = S6b recipe with --coldbind2 + mem,yright,arena. Vs S6b+coldbind in one process: predictions BIT-IDENTICAL r35/r100/r37/r39; util r35 cold -0.229%, r100 -0.069%, r37 -0.029%, r39 -0.060%; binding 4.11 -> 3.89% cold, 1.81 -> 1.74% warm; handles -495..-938. Expected average -0.06..-0.07% on top of --coldbind. Also sent engineer2 the push5p6 MU 0.6/0.7 fragments (depthrule_fragment_push5p6_mu06.py / _mu07.py; --caps2 needs --caps).
    2026-09-23 06:09 [engineer2] --coldbind GATE PASS (engineer's candS6b_coldbind_mem.py f9e67942, reproduced byte-for-byte via build_stack.sh S6b recipe + --coldbind + mem,yright,arena). Bit-exact vs S6b on r35/r100/r37/r39 (16 layers). In-process util: r35 cold -1.275%, r100 -0.250%, r37 -0.079%, r39 -0.260%, warm r35 identical; COLD PREMIUM (r35 cold/warm) 3.93% vs S6b 5.27% -> each cold or restarted-worker MLP pays ~1.3 points less. Paired 20 nets vs S6b (concurrent, load up to ~35): server CPU -0.24% mean; residual -7.2 ms mean (cold net 216 vs 220 ms); round trips -126/net (cold -983); util -0.172% mean over 20 nets; handles 73,439 vs 74,701 after 20 (lower every net); live peak identical. Wall projection -0.25 s mean, -0.65 s cold. On a platform run with the 5 first-of-worker MLPs plus 5 restart-cold ones, the average bill saving is ~-0.25%.
    2026-09-23 06:07 porter: S7 base -> S6b + --coldbind2 (stack_S7base_cb2_mem.py ecf9b9c1 = engineer's file): MSE bit-identical, flops vs S6b -1.50% cold r35 / -0.32% warm r100; validate OK.
    2026-09-23 06:15 [engineer2] --coldbind2 (candS6b_cb2_mem.py ecf9b9c1) in-process vs S6b: BIT-EXACT on r35 cold/r100/r37/r39/r35 warm; util -1.50% cold r35, -0.32/-0.11/-0.32% warm nets, -0.04% on a warm repeat; cold premium 3.73% (coldbind 3.93%, S6b 5.27%). S7-lite variants built (S6b recipe + push5p6 MU 0.7 / 0.6 + --caps --caps2 --batch --bindhold --coldbind2 + mem,yright,arena,hold2): s7lite/v07/V07.py 43a3097c, s7lite/v06/V06.py 12700e90; paired gates vs S6b running sequentially (V07 then V06).
    - 2026-09-23 06:55 [engineer] SCRATCHLESS UNWIND --coldbind3 (= --coldbind2 + 3-line count-asserted builder edit, E-40): the unwind's temporaries s0,s1,s2 are written into the products p1,p0,p5 (each dead by then: tl=p0+p1 kills p1, s0=p0+p5 kills p0,p5), so _o_frame/_bh_o_frame/_bb_o_frame no longer bind the unwind scratch (~44% of each output frame's binding). Ops, order, shapes, contiguity unchanged; every layout's unwind destinations are disjoint from the products they read (runtime tables). File fable_engineer_v1/candS6b_cb3_mem.py (23d4f339) = S6b recipe with --coldbind3 + mem,yright,arena. In one process: predictions BIT-IDENTICAL r35/r100/r37/r39; util vs --coldbind2 -0.359% cold / -0.114% / -0.069% / -0.153%; vs --coldbind -0.587% / -0.183% / -0.098% / -0.213%; binding cold 4.11 -> 3.54%, warm 1.81 -> 1.63%; handles -1551 (net 1) .. -2912 (after 4 nets). All three binding steps vs S6b: cold -1.86%, nets 2-4 -0.18..-0.47%, about -0.44..-0.51% of the average bill. For engineer2's gate (final form); porter: use --coldbind3 in the S7 base.
    2026-09-23 06:18 porter: S7 base -> S6b + --coldbind3 (stack_S7base_cb3_mem.py 23d4f339 = engineer's file): MSE bit-identical, flops vs S6b -1.86% cold r35 / -0.43% warm r100; validate OK; pending engineer2's gate.
    2026-09-23 07:09 [engineer2] cb3 BASE GATE PASS (porter's stack_S7base_cb3_mem.py 23d4f339 = S6b + --coldbind3 + mem,yright,arena, vs the fired S6b a514c21b). In-process: BIT-EXACT on r35 cold / r100 / r37 / r39 / r35 warm; util -1.85% / -0.43% / -0.18% / -0.47% / -0.09%; cold premium (r35 cold/warm) 3.41% (S6b 5.27%). Paired 8 nets (concurrent): server CPU -0.04% mean; residual -10.9 ms mean (absolute residuals 500-750 ms: the machine is at load ~30, so only the paired difference means anything); util -0.515% mean over 8 nets; round trips -2.6k cold, -0.3..-0.9k warm; handles 44,031 vs 48,405 after 8 (lower every net); live peak identical. Adjusted-equivalent (util x MSE, MSE bit-identical) = the util column.
    2026-09-23 07:39 porter: cb3-base VALID40 vs S6b: MSE identical 40/40, util/adjusted -1.836% (cold). S7-full-cb3's cold -2.163% = cb3 -1.836% + push5p6 group -0.333%; warm, the push5p6 group is ~-0.5%. symout (stack_S7full_cb3_so_mem.py 879b1d4b) parity then VALID40 running.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/coldbind_fragment.py](code/fable_engineer_v1/coldbind_fragment.py)
- [code/fable_engineer_v1/bind_census.py](code/fable_engineer_v1/bind_census.py)
- [code/fable_engineer_v1/candS6b_coldbind_mem.py](code/fable_engineer_v1/candS6b_coldbind_mem.py)
- [code/fable_porter_v1/stack/stack_S7base_cb_mem.py](code/fable_porter_v1/stack/stack_S7base_cb_mem.py)
- [code/fable_engineer_v1/usedslots_fragment.py](code/fable_engineer_v1/usedslots_fragment.py)
- [code/fable_engineer_v1/candS6b_cb2_mem.py](code/fable_engineer_v1/candS6b_cb2_mem.py)
- [code/fable_engineer_v1/depthrule_fragment_push5p6_mu06.py](code/fable_engineer_v1/depthrule_fragment_push5p6_mu06.py)
- [code/fable_porter_v1/build_stack.sh](code/fable_porter_v1/build_stack.sh)
- [code/fable_porter_v1/stack/stack_S7base_cb2_mem.py](code/fable_porter_v1/stack/stack_S7base_cb2_mem.py)
- [code/fable_engineer2_v1/s7lite/v07/V07.py](code/fable_engineer2_v1/s7lite/v07/V07.py)
- [code/fable_engineer2_v1/s7lite/v06/V06.py](code/fable_engineer2_v1/s7lite/v06/V06.py)
- [code/fable_engineer_v1/candS6b_cb3_mem.py](code/fable_engineer_v1/candS6b_cb3_mem.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/fable_porter_v1/stack/stack_S7base_cb3_mem.py](code/fable_porter_v1/stack/stack_S7base_cb3_mem.py)
- [code/fable_porter_v1/stack/stack_S7full_cb3_so_mem.py](code/fable_porter_v1/stack/stack_S7full_cb3_so_mem.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L212](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L212)
- [TEAM_LOG_2026-09-22_to_09-26.md#L213](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L213)
- [TEAM_LOG_2026-09-22_to_09-26.md#L219](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L219)
- [TEAM_LOG_2026-09-22_to_09-26.md#L220](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L220)
- [TEAM_LOG_2026-09-22_to_09-26.md#L221](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L221)
- [TEAM_LOG_2026-09-22_to_09-26.md#L223](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L223)
- [TEAM_LOG_2026-09-22_to_09-26.md#L225](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L225)
- [TEAM_LOG_2026-09-22_to_09-26.md#L228](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L228)
- [TEAM_LOG_2026-09-22_to_09-26.md#L247](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L247)
- [TEAM_LOG_2026-09-22_to_09-26.md#L254](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L254)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332026: cheaper first-network frame binding and padded six-level products](../../../submissions/332026/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-eliminating-unused-frame-slots-unwind-scratch-cuts-cold-work) · [Copy provenance](../../../PROVENANCE.json)


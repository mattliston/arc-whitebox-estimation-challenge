# Six-level read estimates omitted 64-padding: all-class work saving becomes a 0.39% increase

2026-09-24T18:17Z · experiment · rejected · measured

Arithmetic estimates that ignored divisibility requirements overstated deeper-read savings. Ordinary six-level squares and seven-level pushes were also uncompetitive at the measured shapes.

Recorded measurement:

    2026-09-24 11:17 [engineer2] DEPTH OPTIONS E-46 NEVER PRICED (for the R320 wall room; per net; platform kernel = local user CPU x 1.2-1.5; MSE rounding-level, so adjusted ~ bill):
      (1) SIX-LEVEL READS, PAPER (no kernel exists: the read layout has 4 levels + read5 packets; level 6 at k=8192 needs ~5e8 floats > the 3.9e8-float arena, so it needs packetized levels 5-6). Built on the engineer's in-situ E-46 read data (S7, 6 warm nets, depth-5 bill and user per call) + the kernel's Winograd bill formula + his in-situ user-CPU model extrapolated to L6 (two coefficient variants): best quarter of read classes -0.85% bill for +3.1..+3.8 s user (+3.7..+5.7 s platform); best half -1.06% for +4.3..+5.3 s (+5.1..+8.0 s); all -1.11% for +7.9..+9.7 s (+9.5..+14.5 s; small reads get dearer at six). Rate 0.23-0.28% bill per user-second for the best quarter, like E-46's best push slices (0.16-0.24). (fable_engineer2_v1/depth6/price_reads_l6.py)
      (2) SIX-LEVEL SQUARES, IN SITU on S11 (depth6/depth_shadow_e2.py = the engineer's v6 tool + squares; r35 cold / r100 warm, 1 thread): squares via _mm are 7.3% of the bill; all eligible squares to six: +0.004% / -0.081% bill for +5.2 / +4.8 s user; bill-saving classes only: -0.12% / -0.17% for +3.4 / +3.7 s. DEAD (~0.04% bill per user-second; the square rule already sits right). Batched covariance products (_mm_batch, fixed depth 5) not covered.
      (3) SEVEN-LEVEL PUSHES, PAPER (depth6/price_push_l7.py): all six-level pushes to seven = bill +0.38% (worse) for +19.5 s user; 823k 8x8 leaf matrices per product; level arrays 2.8x the arena. DEAD.
      Fit with the R320 room (central 6.0 s, pessimistic 3.8, optimistic 8.0): only the best quarter of six-level reads fits (R320 central 94.0 -> ~98.7 s; pessimistic -> ~101.9 s, over). It needs a new packetized six-level read kernel (hours; rounding change, so a VALID40) and one in-situ read measurement before trusting the L6 extrapolation (+-20% between variants).
    2026-09-24 11:45 [engineer2] SIX-LEVEL READS STOPPED (lead's rule: miss > 30% vs the paper price). My paper price (11:4x, best quarter -0.85% for +3.7..+5.7 s) was WRONG: it priced depth 6 at the true dims, but a six-level split needs m, k, n divisible by 64, and 70 of 117 read classes (60%) are odd multiples of 32 in m or n (e.g. 928 x 8128 x 992). Re-priced as push5p6 prices its padded pushes (depth-6 bill at dims padded to 64 + zero-pad copies) plus the only design that fits the arena (level 6 inside read5's packets with one level-4 slot per packet: +227 requests per read at 0.41 ms, level-5 slot copies 7x3x343x(mk+kn)/1024): ALL reads -1.11% -> +0.39% (worse); only 28 classes still save; best 28: -0.25% bill for +3.5..+4.2 s platform (0.06%/s); best 3 (dims already multiples of 64: 896x8192x960, 1024x4096x1024, 832x8192x960): -0.11% for ~+0.7 s. Not worth a multi-hour kernel build (fable_engineer2_v1/read6/price_read6_padded.py, price_read6_design.py). No code written into any stack. Note for depth work that pads rows (e.g. padding deadcut35's pushes back to 1024 rows): the same padding arithmetic applies.
    - 2026-09-24 12:00 lead: --colpack (engineer E-49, candS11cp_mem.py c6ca3ebf) bit-identical, bill -0.29% cold / -0.35% warm, requests -500..-700/net; a post-diet stage, now with the porter. Six-level reads CLOSED (engineer2: 70/117 read classes need 64-padding, so all reads +0.39% and the best 28 classes only -0.25% for +3.5-4.2 s). Wall-room candidates: copy-free Winograd levels (engineer) and less frame binding / column keying E-44 (engineer2), both re-priced against S11 + deadcut35 (+R320) room (engineer2: S12 pessimistic room ~8.8 s, additive, pending the measured pair).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/depth6/price_reads_l6.py](code/fable_engineer2_v1/depth6/price_reads_l6.py)
- [code/fable_engineer2_v1/depth6/depth_shadow_e2.py](code/fable_engineer2_v1/depth6/depth_shadow_e2.py)
- [code/fable_engineer2_v1/depth6/price_push_l7.py](code/fable_engineer2_v1/depth6/price_push_l7.py)
- [code/fable_engineer2_v1/read6/price_read6_padded.py](code/fable_engineer2_v1/read6/price_read6_padded.py)
- [code/fable_engineer2_v1/read6/price_read6_design.py](code/fable_engineer2_v1/read6/price_read6_design.py)
- [code/fable_engineer_v1/candS11cp_mem.py](code/fable_engineer_v1/candS11cp_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L344](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L344)
- [TEAM_LOG_2026-09-22_to_09-26.md#L345](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L345)
- [TEAM_LOG_2026-09-22_to_09-26.md#L346](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L346)
- [TEAM_LOG_2026-09-22_to_09-26.md#L347](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L347)
- [TEAM_LOG_2026-09-22_to_09-26.md#L348](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L348)
- [TEAM_LOG_2026-09-22_to_09-26.md#L363](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L363)
- [TEAM_LOG_2026-09-22_to_09-26.md#L364](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L364)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-six-level-read-estimates-omitted-64-padding-all-class) · [Copy provenance](../../../PROVENANCE.json)


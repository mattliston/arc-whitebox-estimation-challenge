# Retained frame-binding and body aliases reduce hidden deallocation trips without changing work

2026-09-23T11:34Z · experiment · adopted · measured

Holding binding intermediates and temporary remote aliases reduced free trips. These trips mainly cost platform wall time rather than participant residual.

Recorded measurement:

    - 2026-09-23 04:34 [engineer] COLD-MLP ROUND TRIPS (engineer2: first MLP per worker = +12k requests +2.9k free trips of frame binding, sets both maxima): --bindhold (bindhold_fragment.py) holds binding intermediates (arena slices + discarded level arrays from _exact_view/_read5_view/_cp3_bind_arena_views) in _BH_HOLD until the kernel call ends. Census r35 cold (round trips = requests + free trips): S5e 57,948; +bindhold 55,251 (-2,697, util identical); +batch 55,402 (-2,546); +batch+bindhold 52,571 (-5,377, -9.3%). Residual unchanged in mirrored pairs (0.2156/0.2146 vs 0.2140/0.2144 s). Combined file fable_engineer_v1/candS5e_batch_bindhold_mem.py (844ae0eb) sent to engineer2's gate; S6 kernel flags would add --batch --bindhold.
    2026-09-23 05:11 [engineer2] OPTIONAL LAST STAGE `hold2` (build_e2mem.py <final.py> <out> hold2, run alone on a finished file): the e2 v8 AST hold pass applied last, over 33 body functions incl. the porter's _gn_dcov / _k31_row_scale (the layer-0 _nonlinear_first_gaussian excluded: holding its r-loop float64 planes cost +204 MB). On S7 (s7/S7_mem_hold2.py 1d000f3c): bit-exact vs S7 r35/r100 all 16 layers, bill identical; free trips 5,710 -> 2,431 (r35) and 5,578 -> 2,315 (r100), round trips -6..-8% per net; handles identical; live peak 2,405 MB (+22 MB, now at _gn_dcov). Paired 6 nets vs S7: server CPU -0.6%, residual UNCHANGED (-0.4 ms). Refinement of the residual law: the ~4 us per round trip is per REQUEST (client encode/RemoteArray/decode); free trips are nearly free client-side (also seen in engineer's --bindhold). So hold2 buys ~1 s of platform wall per MLP (3.3k free trips at ~0.3 ms), not residual.
    2026-09-23 05:23 [engineer2] ROBUSTNESS PACKAGE built and checked: fable_engineer2_v1/robust/R_mem_hold2.py (626d46eb) = S6b recipe with --bindhold appended to the kernel flags (stack 96dea419 = porter's stack_S6b.py + the 38-line bindhold fragment) + build_e2mem.py mem,yright,arena,hold2. In-process vs the fired S6b's saved predictions (r35 cold, r100, r37, r39): BIT-EXACT on all 16 layers, bill delta 0 on all four. Paired 6 nets vs S6b (concurrent): requests identical per net; free trips cold 8,535 -> 2,425 and warm 5,931-6,842 -> 2,246-2,299; round trips -11% cold, -8..-10% warm (~-1.8 s platform wall on the cold MLP, ~-1.2 s warm); server CPU -0.3%; residual -3.7 ms (noise; requests unchanged); handles identical; util identical; live peak 2,405-2,414 MB (+31..+37 MB, hold2). RSS caveat: macOS RSS read 6.74 GB max in that run but 5.94 GB for the same file in a repeat (bindhold-only twin 5.88 GB, paired): RSS noise ~0.8 GB, live-array peak is the reliable figure. hold2 vs bindhold-only (paired 6 nets): round trips -3.3k/net, CPU +0.2%, residual +2.5 ms (noise).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/bindhold_fragment.py](code/fable_engineer_v1/bindhold_fragment.py)
- [code/fable_engineer_v1/candS5e_batch_bindhold_mem.py](code/fable_engineer_v1/candS5e_batch_bindhold_mem.py)
- [code/fable_engineer2_v1/build_e2mem.py](code/fable_engineer2_v1/build_e2mem.py)
- [code/fable_engineer2_v1/s7/S7_mem_hold2.py](code/fable_engineer2_v1/s7/S7_mem_hold2.py)
- [code/fable_engineer2_v1/robust/R_mem_hold2.py](code/fable_engineer2_v1/robust/R_mem_hold2.py)
- [code/fable_porter_v1/stack/stack_S6b.py](code/fable_porter_v1/stack/stack_S6b.py)
- [code/fable_engineer2_v1/hold2.py](code/fable_engineer2_v1/hold2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L194](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L194)
- [TEAM_LOG_2026-09-22_to_09-26.md#L200](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L200)
- [TEAM_LOG_2026-09-22_to_09-26.md#L202](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L202)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332026: cheaper first-network frame binding and padded six-level products](../../../submissions/332026/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-retained-frame-binding-body-aliases-reduce-hidden-deallocation-trips) · [Copy provenance](../../../PROVENANCE.json)


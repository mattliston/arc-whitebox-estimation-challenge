# Natural-order tree views remove 1,455-1,565 calls with identical outputs, work and peak live memory

2026-09-25T00:33Z · experiment · adopted · measured

Changing view construction and merging cache entries preserved every checked product output. Memory and handle tests confirmed that the optimization did not defer large live arrays.

Recorded measurement:

    2026-09-24 17:33 [engineer2] S16r3b (engineer's candS16r3b_mem.py ca974768 = S16 a4a1e0f7 + E-61 --resid3 view trim, no GC) IDENTITY PASS: code review of the edit to my alt-basis views OK (natural-order level-1 views map every element as before; same ops, same order; 4-value unpackers only on paths unreachable while _AB_ON). prodhash r35/r100: 201/201 and 203/203 tree-product outputs bit-identical, bill +0 per product, op calls -1,455 / -1,565 per net (single -6.1..-6.5, batched -14, colpack -9.5..-10.6, read5 -4.8..-5.4, W5 -5..-6.4 per call). parity.py 35/100/37/39 vs S16: bit-exact all layers, bill delta 0. 10-net run_mem vs memS16x10 queued on the lock (behind the engineer's run_mem_setup).
    2026-09-24 19:29 [engineer2] (late record: finished 18:12, notification arrived 19:29) S16r3b run_mem 10-net vs S16 (memS16r3bx10 vs memS16x10): handles at net end -1.2k..-2.6k per net, session max 27,503 vs 30,074; live at net end and peak live IDENTICAL on all 10 nets; server RSS peak 4,569 vs 6,579 MB; client RSS 319 vs 327 MB; requests -1.46k..-1.68k per net. MEMORY GATE PASS. Local residual +0.08..+0.15 s is unpaired and load-confounded (same-compute server CPU +25% in that window); the platform (S17 #332224) showed residual max -46 ms.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candS16r3b_mem.py](code/candS16r3b_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L517](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L517)
- [TEAM_LOG_2026-09-22_to_09-26.md#L585](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L585)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332224: half-strength unit birth feed, refit, lean views, and setup prebinding](../../../submissions/332224/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-natural-order-tree-views-remove-1-455-1-565) · [Copy provenance](../../../PROVENANCE.json)


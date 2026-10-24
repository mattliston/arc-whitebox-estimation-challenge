# In-place tree Winograd removes level copies: 35.24% less user CPU and 1.750% less work

2026-09-24T19:40Z · experiment · adopted · measured

A deepest-level-first tree layout puts plain child slots in place and reuses product storage during unwind. It preserves predictions at the client/server boundary while sharply reducing transforms, requests, memory and handles.

Recorded measurement:

    - 2026-09-24 12:55 [engineer] E-53 IN-PLACE TREE WINOGRAD KERNEL (IPT; fable_engineer_v1/ipt_bench_v2.py, S11 kernel vs IPT on the same inputs, OpenBLAS env, 1 thread, 6 interleaved pairs): each operand is ONE tree array (deepest level outermost), every block's 4 quadrants sit in a room of 7 chunks so the next level's plain slots are already in place: no level copies below level 1, one leaf matmul, in-place unwind, long contiguous runs. Bit-identical on 1024x896x4096 / 1024x384x4096 / 1024x896x1920; bill -1.55 / -1.33 / -1.90%; ops 119 -> 103; user CPU 374 -> 181 / 257 -> 134 / 233 -> 114 ms (about -50%). (v1 with the root level outermost was +100-150 ms slower: deepest-level ops touched 1 block in 7.) S12 per-level profile (op_cpu_level_S12.json, r35): transforms 21.5 s + copies 3.4 s + leaf matmuls 9.9 s of 46.6 s thread CPU per net. Copy-free level L-1 (B side, copyfree_bench_v2.py): bit-identical after a layout fix, -0.65..-0.69% per packet, +20 ops, CPU neutral; superseded by IPT. Next: --ipt as one bit-exact flag on S12 (push paths first).
    - 2026-09-24 12:40 lead: ENGINEER E-53 IN-PLACE TREE WINOGRAD KERNEL (--ipt): each operand is one tree array with the DEEPEST level outermost, so the next level's plain slots are already in place. No level copies below level 1, one leaf matmul over all 7^L pairs, in-place unwind. Bit-identical to S11's kernel. Six-level push packets: user CPU 374->181 ms, bill -1.33..-1.90% per product. Card scale (if the halving holds): about -10..-12 s local user CPU/net and about -1.2% bill; the freed wall then funds --m64 and more six-level shapes. Copy-free levels are superseded. GO: stage 1 = push paths as S13 (gates: bit-identity, einsum, validate, paired CPU, server live-memory peak); stage 2 = squares, read5, batched, W5 as S14. Column keying is re-priced after IPT stage 1.
    - 2026-09-24 13:33 [engineer] S13 READY = S12 (c20d67c7) + --ipt v3 (fable_engineer_v1/candS12ipt3_mem.py 2a5dd324; build_ipt_v3.py post-stage, applied last): ALL Winograd products except engineer2's read6 (its trees need 512M floats > the 392M arena) run on the in-place tree kernel. Predictions byte-identical to S12 in process (r35/r100) and at the client/server boundary (cs_probe.py). Porter's paired user-CPU census (6 nets): flops -1.750% (cold -2.49, warm -1.30..-1.78), user -35.24%, backend -35.70%. engineer2's run_mem.sh (same window): live-array peak 2555 -> 2424 MB, handles 25.7k -> 16.2k, server CPU 45.3/44.5 -> 32.2/31.0 s, wall 50.4/49.2 -> 36.8/34.9 s, requests 48.5k/43.1k -> 39.3k/37.1k, residual lower. Einsum gate PASS, validate PASS (gate_S12ipt3_checks.txt). Projection: S12 ~3.016e-9 -> S13 ~2.963e-9 (same MSE, util x0.9825). Recommend firing S13 in place of S12. v2 -> v3: tree caps 48 -> 160 (a net has ~70 shapes per side) and chunk views by one unstack per level (v2 had RAISED requests). Old-kernel m64 pair (porter's tool): flops -0.71%, user +7.97%; to be re-measured on S13.
    2026-09-24 13:37 porter: S13 (2a5dd324 = S12 + ipt v3) reproduced byte-identically; in-process predictions identical to S12 (all 16 layers, r35/r100), flops -2.49/-1.75%. fire_chain_S13.sh running: smoke -> VALID40 vs S11 -> identity vs S12 rows -> bar -> stage fable_fire_v1/fire_candS13/.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/ipt_bench_v2.py](code/fable_engineer_v1/ipt_bench_v2.py)
- [code/fable_engineer_v1/copyfree_bench_v2.py](code/fable_engineer_v1/copyfree_bench_v2.py)
- [code/fable_engineer_v1/candS12ipt3_mem.py](code/fable_engineer_v1/candS12ipt3_mem.py)
- [code/fable_engineer_v1/build_ipt_v3.py](code/fable_engineer_v1/build_ipt_v3.py)
- [code/fable_engineer_v1/cs_probe.py](code/fable_engineer_v1/cs_probe.py)
- [code/fable_engineer2_v1/run_mem.sh](code/fable_engineer2_v1/run_mem.sh)
- [code/fable_porter_v1/fire_chain_S13.sh](code/fable_porter_v1/fire_chain_S13.sh)
- [code/fable_engineer_v1/ipt_fragment_v3.py](code/fable_engineer_v1/ipt_fragment_v3.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L390](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L390)
- [TEAM_LOG_2026-09-22_to_09-26.md#L391](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L391)
- [TEAM_LOG_2026-09-22_to_09-26.md#L392](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L392)
- [TEAM_LOG_2026-09-22_to_09-26.md#L394](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L394)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [op_cpu_level_S12.json](op_cpu_level_S12.json) · result
- [gate_S12ipt3_checks.txt](gate_S12ipt3_checks.txt) · record

</details>

## Related work

- shipped in: [Submission #332195: column packing, inactive-row selection, six-level reads, and in-place tree products](../../../submissions/332195/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-place-tree-winograd-removes-level-copies-35-24-less) · [Copy provenance](../../../PROVENANCE.json)


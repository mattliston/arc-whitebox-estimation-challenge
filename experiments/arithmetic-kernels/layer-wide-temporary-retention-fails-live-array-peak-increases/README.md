# Layer-wide temporary retention fails: live-array peak increases 1,469 MB

2026-09-23T16:28Z · experiment · rejected · measured

Keeping large temporary scopes alive reduced deallocation trips but substantially increased peak memory, CPU time and residual.

Recorded measurement:

    - 2026-09-23 09:28 [engineer] --lean v2 FAILED the memory/CPU constraints; v3 built. v2 (layer-scoped release: kernel holds and hold2 scopes released once per layer + the live loop's last-layer arrays kept to the next boundary + split quadrants + small holds), paired ABBA with engineer2's run_mem.sh (load 15-20): on S8, round trips -1,783 cold / -1,753 warm (requests -759, free trips -1,024/-994) but live-array peak +1,469 MB (deterministic), server CPU +8.8/+13.0 s, residual +164/+389 ms. On S8' (no hold2, so no scope registration) live-array peak +701 MB, server CPU +5.3/+4.7 s, residual +4/+19 ms. Lesson: extending lifetimes to batch frees costs more (bigger live set, bulk frees) than the trips save; hold2's function-scoped holds already did the cheap part (S8' without hold2 pays 7.9k free trips vs S8's 2.6k). v3 = split quadrants/corners (-759 requests) + function-scoped holds only (select, the old-tier finite guard, _sym_upper, read5 packet slices per kernel call); no lifetime extension. v3: bit-identical predictions and util on r35/r100/r37/r39 for S8 (candS8so3L3_mem.py 11f7510a) and S8' (candS8pso3L3_mem.py bde6ec4f, cb3 recipe + --symout + --lean); einsum/outer audit identical to S7 / cb3. Paired server ABBA for v3 running.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/run_mem.sh](code/fable_engineer2_v1/run_mem.sh)
- [code/fable_engineer_v1/candS8so3L3_mem.py](code/fable_engineer_v1/candS8so3L3_mem.py)
- [code/fable_engineer_v1/candS8pso3L3_mem.py](code/fable_engineer_v1/candS8pso3L3_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L289](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L289)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-layer-wide-temporary-retention-fails-live-array-peak-increases) · [Copy provenance](../../../PROVENANCE.json)


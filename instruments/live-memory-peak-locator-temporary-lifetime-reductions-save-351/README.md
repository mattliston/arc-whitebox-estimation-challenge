# Live-memory peak locator and temporary lifetime reductions save 351 MB without changing predictions

2026-09-23T08:35Z · tool · adopted · measured

A server wrapper identifies the operation and live buffers at peak usage. Removing dead operands and redundant float64 temporaries lowered the peak while preserving all layer outputs and work.

Recorded measurement:

    2026-09-23 01:35 [engineer2] LIVE-PEAK LOCATOR + BIT-EXACT MEMORY DIET. New server wrapper fable_engineer2_v1/server_mem_v2.py (+ census_e2_fine.py, run_mem_fine.sh) names the op, client call stack and the 25 largest live buffers at each prediction's live-array peak. S4 (#331961 file): peak 3,015 MB r35 / 3,010 MB r100 sits at the live-loop diagonal line (_predict_binary:3244): kernel arena 1,494 MB + FOUR (1024 x 22.8k) float64 arrays (pp64, pp2 and two transients, 178 MB each) + the dead previous factors p (84 MB) + the dead read operands left/right (114 MB). Builder fable_engineer2_v1/build_e2mem.py <in> <out> (3 count-asserted edits in the live loop: *lam in place, del left/right after the read, del p after the push; requires T31 locals() plumbing off) -> mem_diet/candS4_mem.py: bit-exact vs S4 on r35/r100 all 16 layers, bill identical, round trips -55; live peak 2,664 / 2,661 MB (-351 MB), server peak RSS 4.15 -> 3.67 GB, handles identical. New peak = pair map line 912 (the (5,3,3,n,n) float32 product, 180 MB) with pp64/pp2 still held; diagonal site now ~2,640 MB. Paired server CPU vs S4 running (pair_cpu_S4vMEM.log).
    2026-09-23 01:42 [engineer2] E2-MEM diet PASSES every gate: paired-concurrent server CPU vs S4 (pair_cpu_S4vMEM.log, load 15-17, 2 rounds) r35 77.8 vs 78.5 s / 72.1 vs 73.2 s, r100 69.7 vs 70.8 s / 70.5 vs 70.8 s (-0.4 to -1.6%, both rounds lower); bit-exact r35/r100 all layers, bill identical, live peak -351 MB (3,015 -> 2,664), handles identical. Stack it last: python fable_engineer2_v1/build_e2mem.py <in> <out>. Next: optional stage `yright` (gather only the young columns of right for the old-tier read; bit-exact, bill ~-0.1%) in test.
    2026-09-23 02:05 [engineer2] build_e2mem.py now also takes engineer's float32 third-order form (build_f32.py third): on candS3_HE_f32third.py the diet is bit-exact (r35/r100 all layers, bill identical). Live peaks: f32third alone moves S3-HE's peak from 2,676 to ~2,075 MB (float64 plane gone; new peak = terminal-layer k3 cube, _predict_binary:3199); dr2a 3,007 MB (same diagonal site as S4) -> 2,656 MB with the diet. So S5 (S4 + push5 + L15 cut + diet) should sit near 2,660 MB and ~2,450 MB if f32third joins. One-command gate for the S5 file when it exists: fable_engineer2_v1/gate_fire.sh <cand> <ref=S4> <tag> (6-prediction memory/handles/peak site + 2 paired CPU rounds, waits out quiet windows).

## Programs

- [code/fable_engineer2_v1/server_mem_v2.py](code/fable_engineer2_v1/server_mem_v2.py)
- [code/fable_engineer2_v1/census_e2_fine.py](code/fable_engineer2_v1/census_e2_fine.py)
- [code/fable_engineer2_v1/run_mem_fine.sh](code/fable_engineer2_v1/run_mem_fine.sh)
- [code/fable_engineer2_v1/build_e2mem.py](code/fable_engineer2_v1/build_e2mem.py)
- [code/fable_engineer2_v1/mem_diet/candS4_mem.py](code/fable_engineer2_v1/mem_diet/candS4_mem.py)
- [code/fable_engineer_v1/build_f32.py](code/fable_engineer_v1/build_f32.py)
- [code/fable_engineer_v1/candS3_HE_f32third.py](code/fable_engineer_v1/candS3_HE_f32third.py)
- [code/fable_engineer2_v1/gate_fire.sh](code/fable_engineer2_v1/gate_fire.sh)
- [code/fable_engineer2_v1/hold2.py](code/fable_engineer2_v1/hold2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L139](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L139)
- [TEAM_LOG_2026-09-22_to_09-26.md#L141](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L141)
- [TEAM_LOG_2026-09-22_to_09-26.md#L146](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L146)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [pair_cpu_S4vMEM.log](pair_cpu_S4vMEM.log) · log

</details>

## Related work

- shipped in: [Submission #331992: final third-moment row selection, five-level products, and memory reductions](../../submissions/331992/README.md)

[Research graph](../../graph/index.html#node=2026-09-23-live-memory-peak-locator-temporary-lifetime-reductions-save-351) · [Copy provenance](../../PROVENANCE.json)


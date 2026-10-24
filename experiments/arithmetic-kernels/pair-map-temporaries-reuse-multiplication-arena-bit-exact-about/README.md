# Pair-map temporaries reuse the multiplication arena: bit-exact and about 1.2-1.6% less server CPU

2026-09-23T10:25Z · experiment · adopted · measured

The pair map writes temporary series products into the otherwise idle kernel arena. Gathering only young read columns adds a small work saving.

Recorded measurement:

    2026-09-23 03:25 [engineer2] S6 COMPONENT READY: build_e2mem.py stage `arena` (use `build_e2mem.py <in> <out> mem,arena`): the pair map's three series products (72/180/96 MB float32 per layer) and its two tri sums are written with out= into views of the kernel arena (pure scratch between kernel calls, engineer 02:03; the pair map makes no kernel call). Bit-exact vs stack_S5f_mem.py r35/r100 all 16 layers; bill +0.03% on the first prediction of a process only (five arena views bound once), identical afterwards. Paired server CPU vs S5e (stack_S5e_mem.py vs mem_diet/stack_S5e_mem_arena.py 3331ab66, 2 rounds, load 8): r35 -1.5% / -1.6%, r100 -1.5% / -1.2%; round trips -41 / -56; residual within noise. Log pair_cpu_S5evARENA.log.
    2026-09-23 03:49 [engineer2] S6 + ARENA (fable_engineer2_v1/mem_diet/candS6_p6caps_mem_arena.py dfd19c44 = engineer's S6 stack + build_e2mem.py mem,arena): bit-exact vs candS6_p6caps_mem.py on r35/r100 all 16 layers; bill +119,537,664 flops (+0.03%) on the first prediction of a process only, identical after. Paired server CPU vs S6 (2 rounds, load 6.5): r35 -0.9% / -1.6%, r100 -1.4% / -0.9%; round trips -41 / -56; residual within +-5 ms. Net vs S5e: ~+3.1% server CPU for -0.43% bill. Log pair_cpu_S6vARENA.log.
    2026-09-23 03:54 [engineer2] S6 BODY STACK OPTION `build_e2mem.py <S6 stack> <out> mem,yright,arena` -> fable_engineer2_v1/mem_diet/candS6_p6caps_mem_yr_arena.py (7b609890): bit-exact vs candS6_p6caps_mem.py on r35/r100 all 16 layers; bill r35 -0.043% (first prediction: yright -0.073% + one-time arena binding +0.030%), r100 -0.076% (warm, yright alone); arena's paired CPU -1.2..-1.6% (measured on S5e and S6). yright = gather only the young columns of `right` for the old-tier read (the old columns are never read).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/build_e2mem.py](code/fable_engineer2_v1/build_e2mem.py)
- [code/fable_porter_v1/stack/stack_S5f_mem.py](code/fable_porter_v1/stack/stack_S5f_mem.py)
- [code/fable_porter_v1/stack/stack_S5e_mem.py](code/fable_porter_v1/stack/stack_S5e_mem.py)
- [code/fable_engineer2_v1/mem_diet/stack_S5e_mem_arena.py](code/fable_engineer2_v1/mem_diet/stack_S5e_mem_arena.py)
- [code/fable_engineer2_v1/mem_diet/candS6_p6caps_mem_arena.py](code/fable_engineer2_v1/mem_diet/candS6_p6caps_mem_arena.py)
- [code/fable_engineer_v1/candS6_p6caps_mem.py](code/fable_engineer_v1/candS6_p6caps_mem.py)
- [code/fable_engineer2_v1/mem_diet/candS6_p6caps_mem_yr_arena.py](code/fable_engineer2_v1/mem_diet/candS6_p6caps_mem_yr_arena.py)
- [code/fable_engineer2_v1/hold2.py](code/fable_engineer2_v1/hold2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L175](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L175)
- [TEAM_LOG_2026-09-22_to_09-26.md#L178](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L178)
- [TEAM_LOG_2026-09-22_to_09-26.md#L180](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L180)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [pair_cpu_S5evARENA.log](pair_cpu_S5evARENA.log) · log
- [pair_cpu_S6vARENA.log](pair_cpu_S6vARENA.log) · log

</details>

## Related work

- shipped in: [Submission #332005: joint closure refit, batched products, and memory-layout changes](../../../submissions/332005/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-pair-map-temporaries-reuse-multiplication-arena-bit-exact-about) · [Copy provenance](../../../PROVENANCE.json)


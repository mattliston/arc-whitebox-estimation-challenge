# Padded depth-six pushes save 0.426% work for 4.40% more server CPU

2026-09-23T09:13Z · experiment · inconclusive · measured

Padding selected pushes enabled another multiplication level, but the combined candidate traded a small score gain for increased compute and residual time.

Recorded measurement:

    - 2026-09-23 02:13 [engineer] S6 CANDIDATE (queued by lead): fable_engineer_v1/candS3_HEB_push5p6.py (sha 3e166670) = push5 + six levels on zero-padded 1024-row pushes whose k or n is not /64 (depthrule_fragment_push5p6.py via make_depthrule_fragment_v3.py; 23 shape classes, ~7.8 products/net). Vs push5: util -0.20% r35 cold, -0.41/-0.80/-0.55% r100/r37/r39 (mean -0.49%); MSE +0.33/-0.39/+0.12/+0.06% (noise); requests about equal. Compute open (model +3 s loaded, ~+1-2 s quiet expected); quiet window 5 was contaminated by two other flopscope lanes -> handed to engineer2's concurrent-pair gate. Stack swap for S6: --depthrule depthrule_fragment_push5p6.py.
    - 2026-09-23 03:21 [engineer] S6 CANDIDATE fable_engineer_v1/candS6_p6caps_mem.py (sha 0fd6043c) = fired S5e recipe (build_stack.sh --gn=gn_knobs/v3.json --l15cut --join=v2 -- --holdelim --bigarena --depthrule ... --f32 third --evict5, then build_e2mem.py; this recipe with push5 rebuilds S5e byte-for-byte, c34bdd79) with --depthrule depthrule_fragment_push5p6.py and --caps. Vs S5e in-process (r35 cold, r100, r37, r39): util -0.21/-0.38/-0.83/-0.52% (mean -0.49%), MSE +0.24/+0.08/-0.04/-0.02% (rounding), adj mean -0.42%, handles slightly below S5e. Handed to engineer2 for one combined gate (20-net handles, residual per net, CPU vs S5e). --lru retired (engineer2: +20..+60 ms residual per net from its walk).
    2026-09-23 03:43 [engineer2] S6 COMBINED GATE (candS6_p6caps_mem.py 0fd6043c vs fired S5e c34bdd79, both run concurrently over the same 20 distinct nets, so every column is paired; paired20.py): util mean -0.426% (-0.83..+0.05% per net); SERVER CPU mean +4.40% (median +4.39%, range -0.2..+7.3%, ~+1.9 s per net at load 8); residual +10.1 ms mean (+4..+21 ms; S6 max 272 ms); round trips about equal; handles bounded by --caps: 62,684 after 20 nets (max 62,828; S5e 74,132); live peak identical 2,386-2,404 MB. Verdict: memory, handles and residual PASS; compute is the price of push5p6 (~+3.7 s platform at the 2.4x factor, inside S4's ~19 s headroom): -0.43% bill for +4.4% server CPU. My `arena` stage (-1.4% paired CPU, bit-exact) would pay back about a third of it: S6+arena built as fable_engineer2_v1/mem_diet/candS6_p6caps_mem_arena.py (dfd19c44).
    - 2026-09-23 03:42 [engineer] S6 GATE READ (engineer2 paired 20 nets, candS6_p6caps_mem vs fired S5e): --caps PASS (handles flat ~63k from net 16: 62,684 vs 74,132 after 20; peak memory identical; residual +10 ms mean). push5p6: util -0.43% for server CPU +4.40% (~+1.9 s/net at load 8, ~+3.5-4.5 s platform). Dial if headroom is tight (model, per net): padded-D6 MU 0.6 -0.40% bill / +2.1 model-s (~+3.1% CPU), MU 0.7 -0.28% / +1.35 (~+2.0%), MU 0.8 -0.16% / +0.7; one-line rebuild each. Decide with S5e's graded worst-MLP wall.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/candS3_HEB_push5p6.py](code/fable_engineer_v1/candS3_HEB_push5p6.py)
- [code/fable_engineer_v1/depthrule_fragment_push5p6.py](code/fable_engineer_v1/depthrule_fragment_push5p6.py)
- [code/fable_engineer_v1/make_depthrule_fragment_v3.py](code/fable_engineer_v1/make_depthrule_fragment_v3.py)
- [code/fable_engineer_v1/candS6_p6caps_mem.py](code/fable_engineer_v1/candS6_p6caps_mem.py)
- [code/fable_porter_v1/build_stack.sh](code/fable_porter_v1/build_stack.sh)
- [code/fable_engineer2_v1/build_e2mem.py](code/fable_engineer2_v1/build_e2mem.py)
- [code/fable_engineer2_v1/paired20.py](code/fable_engineer2_v1/paired20.py)
- [code/fable_engineer2_v1/mem_diet/candS6_p6caps_mem_arena.py](code/fable_engineer2_v1/mem_diet/candS6_p6caps_mem_arena.py)
- [code/fable_engineer2_v1/hold2.py](code/fable_engineer2_v1/hold2.py)
- [code/fable_engineer_v1/depthrule_fragment_v0q.py](code/fable_engineer_v1/depthrule_fragment_v0q.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L152](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L152)
- [TEAM_LOG_2026-09-22_to_09-26.md#L174](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L174)
- [TEAM_LOG_2026-09-22_to_09-26.md#L176](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L176)
- [TEAM_LOG_2026-09-22_to_09-26.md#L177](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L177)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [v3.json](v3.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-padded-depth-six-pushes-save-0-426-work-4) · [Copy provenance](../../../PROVENANCE.json)


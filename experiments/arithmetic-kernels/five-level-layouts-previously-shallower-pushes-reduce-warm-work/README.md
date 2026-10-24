# Five-level layouts for previously shallower pushes reduce warm work about 1.9%

2026-09-23T08:30Z · experiment · adopted · measured

New layouts enabled five multiplication levels on shapes that could not use six. They passed prediction, compute and memory checks before being combined with final-layer row selection.

Recorded measurement:

    - 2026-09-23 01:30 [engineer] S5 CANDIDATE (depth reallocation at a compute budget): fable_engineer_v1/candS3_HEB_push5.py (sha 36573984) = S3-HEB + --depthrule depthrule_fragment_push5.py. Pushes that cannot take six levels (k or n not /64, 384-row pushes) now take FIVE via a new layout (make_push_layout.py, reproduces the W-F D6 layout exactly), 4096 packets like D6. Util vs S3-HEB: r35 cold -0.49%, r100 -1.95%, r37 -1.87%, r39 -1.94%; MSE -0.10/-0.05/+0.06/+0.35% (rounding); adjusted -0.6/-2.0/-1.8/-1.6%. Paired quiet census r35 cold, 6 pairs in 3 windows: server CPU push5 - HEB median +0.5 s (-2.0, -1.7, +0.5, +0.6, +0.7, +7.6 window-start outlier); requests +3%; peak RSS 6.0-6.07 GB (HEB 5.7-6.0). Pulling D6 off the k=384 pushes (dr2b) saves ~1 s but gives back 0.5% bill; D6 1024^3 squares (dr2c) -1.1% more bill for +4.4..+5.3 s = over budget. Frame-bill lesson: unpacketized D5 binds ~0.5 Gflop per new shape (dr1 lost 0.6-0.8% cold). Details: fable_engineer_v1/RESULT.md top.
    - 2026-09-23 02:00 [engineer] S5-LIKE STACK SMOKE: porter's build_stack.sh --l15cut -- --holdelim --bigarena --depthrule depthrule_fragment_push5.py --f32 third (no gn/refit) -> fable_engineer_v1/candStack_l15_push5_f32_check.py (sha 9aad7022) runs clean; util r35 .18011 / r100 .17522 = -1.80% / -3.37% vs S3-HEB (push5 + f32 + row cut add up); MSE -0.03% / +0.43% vs S3-HEB (the row cut's own +0.01/+0.41%). Flags sent to porter; --f32 third is the lead's call.
    2026-09-23 02:20 [engineer2] S5 DRAFT GATE (lead's recipe built in fable_engineer2_v1/s5draft/: stack 6aff8dd4 = S4 + L15 cut + push5; + build_e2mem.py = S5draft_mem.py 338c02db): diet bit-exact vs stack; stack vs S4 util -1.30%/-2.85%, MSE +0.18/+0.15% (r35/r100). Memory PASS: live peak 2,662-2,679 MB over 6 predictions (plateau, repeats exact; S4 3,015), RSS 4.53 GB. Paired server CPU vs S4 PASS: r35 -0.4%, r100 +1.7% (2 rounds). HANDLES: 36,074 after 4 distinct nets (S4 33,082, bank 35,786) -> push5 caches ~750 more views per new net; 20-distinct-net test S5 vs S4 running (~02:45) before any fire.
    2026-09-23 02:35 porter: S5 (stack_S5_mem.py, sha 276455ff...): validate OK; diet bit-exact; parity vs S4+l15cut r35 cold adj -1.14% / r100 warm -2.41% (MSE +0.08/+0.14%); VALID40 adj 0.90083 vs bank 40/40, vs S4 0.9776 (40/40), raw vs S4+l15cut 0.99979; panel8 0.8998 / virgin6 0.9075 vs bank.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/candS3_HEB_push5.py](code/fable_engineer_v1/candS3_HEB_push5.py)
- [code/fable_engineer_v1/depthrule_fragment_push5.py](code/fable_engineer_v1/depthrule_fragment_push5.py)
- [code/fable_engineer_v1/make_push_layout.py](code/fable_engineer_v1/make_push_layout.py)
- [code/fable_porter_v1/build_stack.sh](code/fable_porter_v1/build_stack.sh)
- [code/fable_engineer_v1/candStack_l15_push5_f32_check.py](code/fable_engineer_v1/candStack_l15_push5_f32_check.py)
- [code/fable_engineer2_v1/build_e2mem.py](code/fable_engineer2_v1/build_e2mem.py)
- [code/fable_engineer2_v1/s5draft/S5draft_mem.py](code/fable_engineer2_v1/s5draft/S5draft_mem.py)
- [code/fable_porter_v1/stack/stack_S5_mem.py](code/fable_porter_v1/stack/stack_S5_mem.py)
- [code/fable_engineer2_v1/hold2.py](code/fable_engineer2_v1/hold2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L138](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L138)
- [TEAM_LOG_2026-09-22_to_09-26.md#L145](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L145)
- [TEAM_LOG_2026-09-22_to_09-26.md#L156](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L156)
- [TEAM_LOG_2026-09-22_to_09-26.md#L158](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L158)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

## Related work

- shipped in: [Submission #331992: final third-moment row selection, five-level products, and memory reductions](../../../submissions/331992/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-five-level-layouts-previously-shallower-pushes-reduce-warm-work) · [Copy provenance](../../../PROVENANCE.json)


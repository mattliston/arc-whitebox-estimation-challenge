# Static low-residual stack projects 1.4747e-9 on a boundary pair but only ties the baseline on eight networks

2026-10-16T16:20:51Z · experiment · rejected · measured

Gains were stacked on the LR2 base through the stack gate: the layer-15 row cut (1.4870e-9 projected), plus the 0.002 floor (1.4856e-9), S1 (+ byte-identical diets), S3 (+ energy-ordered join rows JORD, five-full dropped) projected 1.4747e-9 on the boundary bill, the first clean build under #336412. The 8-network worker-like session put its public-mix bill at +0.447% vs the estimator submitted as #336412 (S2tr) (the rank-100 boundary net is the cheapest warm net), a projected 1.4768e-9, a tie; its own 24-seed accuracy read was -0.277%. S3 was held as the clean safe candidate and never fired.

Recorded measurement:

    stkl15rc30 9f38be95: slow 42,403/40,761, worst 92.0 s, median .3161, bill -0.513% vs LR2, prereg 1.4870e-9 (+0.755% vs 336412). stkl15rc30dcv2: slow 42,403/40,779, median .3160, prereg 1.4856e-9. S1 c38c4fd6: slow 42,234/40,618. S3 6921feaa: normal 41,929/40,634, slow 42,390/40,772, worst 91.6 s, setup 4.442/4.585, median .3154; bill -1.144% vs LR2 = +0.305% vs S2tr on boundary nets, dmse -0.384% (STK4) -> prereg 1.4747e-9; 8-net bill vs S2tr cold +0.810%, warm mean +0.407% (se .08) -> public mix +0.447% -> 1.4768e-9; S3 own 24-seed xMSE -0.277% (SE .304); projected platform residual cold .331, warm median .316, max .323.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_passrel_v1/stack_v1/cand_s3.py](code/lead/runs/opus_passrel_v1/stack_v1/cand_s3.py)
- [code/lead/runs/opus_ladder2h_v1/apply_jord_v1.py](code/lead/runs/opus_ladder2h_v1/apply_jord_v1.py)
- [code/lead/runs/opus_passrel_v1/s3_v1/cand_s3base_v1.py](code/lead/runs/opus_passrel_v1/s3_v1/cand_s3base_v1.py)
- [code/lead/runs/opus_passrel_v1/stack_v1/cand_l15rc30.py](code/lead/runs/opus_passrel_v1/stack_v1/cand_l15rc30.py)
- [code/lead/runs/opus_lowres_v1/cand_lowres_LR2_v3.py](code/lead/runs/opus_lowres_v1/cand_lowres_LR2_v3.py)
- [code/lead/runs/opus_passrel_v1/stack_v1/cand_l15rc30dcv2.py](code/lead/runs/opus_passrel_v1/stack_v1/cand_l15rc30dcv2.py)
- [code/lead/runs/opus_passrel_v1/stack_v1/cand_s1.py](code/lead/runs/opus_passrel_v1/stack_v1/cand_s1.py)
- [code/lead/runs/opus_passrel_v1/s1_v1/cand_s1base_v1.py](code/lead/runs/opus_passrel_v1/s1_v1/cand_s1base_v1.py)
- [code/fire/fire_once.py](code/fire/fire_once.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4286](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4286)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4288](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4288)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4292](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4292)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4311](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4311)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4324](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4324)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4329](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4329)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4331](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4331)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4342](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4342)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4355](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4355)

<details>
<summary>All result files and records (8)</summary>

- [STACK_s3.json](../../../instruments/stack-gate-v1-v3-base-plus-appended-fragments-through/STACK_s3.json) · record
- [STACK_l15rc30.json](STACK_l15rc30.json) · record
- [STACK_l15rc30dcv2.json](STACK_l15rc30dcv2.json) · record
- [STACK_s1.json](STACK_s1.json) · record
- [RELEASE_CONFIRMED_v1.json](RELEASE_CONFIRMED_v1.json) · record
- [LADDER2_GRADE24_latest.json](LADDER2_GRADE24_latest.json) · result
- [ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json](../../../records/research/2026-10-15-per-network-wall-limit-corrected-120-seconds-inherited-100/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Five byte-identical request reductions save 390 cold and 386 warm calls, funding a 0.47% bill reduction](../../flop-accounting/five-byte-identical-request-reductions-save-390-cold-386/README.md)
- ← enables: [Static low-residual base with two-strip covariance and pruned reads: +1.466% bill, not submitted](../../../records/research/2026-10-16-static-low-residual-base-two-strip-covariance-pruned-reads/README.md)
- enables: [Candidate-specific prebind refit saves 603 cold and 495 warm requests under slow setup, with identical bytes](../../setup-budget/candidate-specific-prebind-refit-saves-603-cold-495-warm/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-static-low-residual-stack-projects-1-4747e-9-boundary) · [Copy provenance](../../../PROVENANCE.json)


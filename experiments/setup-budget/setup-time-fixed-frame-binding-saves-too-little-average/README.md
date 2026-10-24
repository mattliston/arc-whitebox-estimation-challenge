# Setup-time fixed-frame binding saves too little average work for its setup-time risk

2026-09-23T14:03Z · experiment · rejected · measured

Only a small part of cold frame binding has network-independent shapes. Setup binding preserved predictions, but its average saving was too small to justify consuming the setup wall budget.

Recorded measurement:

    - 2026-09-23 07:03 [engineer] SETUP-TIME FRAME BINDING --setupbind (setupbind_fragment_v2.py, E-42). Census first (frame_census.py, cold_premium_census.py on r35/r100/r37, each cold in a fresh process): the cold premium is 3.5-3.7% of a cold net's bill and is all binding (frame reshapes 3.00%, read5 children 0.54%, pair-map arena 0.03%); a cold net binds 263-278 frames, but only 25 have shapes fixed by the width and kernel tiles (0.50%) plus the 1024-unit pair-map arena (0.03%); everything else is shaped by the MLP's active counts. Estimator.setup now binds those 26 items through the kernels' own builders, largest first, inside a 2.0 s wall; with no flopscope session (local client harness: NoBudgetContextError) it opens its own BudgetContext; any failure restores every cache and global. Checks: (1) server path: setup arrays survive into predict's BudgetContext (the cold predict uses them); the platform opens a setup session (eval pages: rivals' setup_executions flops). (2) cold r35 bill -0.53% (in-process and server path), predictions bit-identical (server path 0/16384 entries differ). (3) local server path: setup 0.47 s + 0.12 s client start through import, setup server CPU 0.22 s. (4) warm r100/r37 bills and handles identical. Also bit-exact under the S7-lite flags (caps2, bindhold, hold2): cold -0.535%, warm unchanged. Files: fable_engineer_v1/candS6b_cb3sb2_mem.py (87ece267) = S6b recipe + --coldbind3 --setupbind + mem,yright,arena; candS7lite_pad07_cb3sb_mem.py (f48a0f9f). Worth -0.03% (5 cold per 100) to -0.05% (10 cold) of the average bill.
    2026-09-23 07:18 [engineer2] S7-full cb3 (porter's stack_S7full_cb3_mem.py 9011916a) in-process vs S6b: util -2.07 / -0.85 / -0.95 / -1.03%, MSE +0.04 / -0.09 / -0.02 / +0.01%, ADJUSTED-EQUIVALENT -2.04 / -0.93 / -0.96 / -1.01% (r35 cold / r100 / r37 / r39; mean -1.24%, warm nets -0.97%); cold premium 3.52%. Paired 20-net run vs S6b in progress (~07:50). SETUPBIND: fable_engineer2_v1/build_e2setup.py <in> <out> [budget_s=2.0] appends Estimator.setup: (1) arena + layout markers (the live loop's own init), (2) fixed-shape products on zero operands to bind their frames (batched 1024^3 D5 P=2/P=3, single 1024^3 square, layer-0 gram512), (3) body parameter uploads and buffers; opens its own BudgetContext if none is active; any exception stops it silently (predict binds lazily). In-process probe on the cb3 base: all steps done in 1.51 s (shim) / 1.33 s (bare, own context), import 0.09 s, setup flops 8.8e9. Bit-exactness / cold-bill test and paired client/server check queued after the S7-full gate (swap 1.66/2 GB).
    2026-09-23 08:08 [engineer2] SETUPBIND (fixed shapes) RESULT: bit-exact (cb3 + setup vs S6b on r35 cold/r100/r37/r39/r35 warm), but SMALL: cold r35 util -0.195% in-process (cold premium 3.41% -> 3.21%), warm nets identical. Client/server paired 6 nets vs the cb3 base: setup (shim) 2.08 s + 0.07 s import under load 16-19, 1,300 requests + 120 free trips, 8.7e9 setup flops; cold net -0.164% bill, -667 round trips, residual unchanged; warm nets, handles, peak identical. Verdict: not worth a SETUP_TIMEOUT risk (on the platform 1,300 requests ~0.65 s + 2.4x server time ~3.6 s could approach the 5 s limit) for -0.16% on 5-10 cold MLPs. Only ~0.2 of the 3.4% cold binding bill is network-independent; the rest is push/read/W5 frames keyed by live/column counts that RECUR across nets. Running a 24-net frame-shape census (fable_engineer2_v1/setupbind/frame_census.py) to price a census-driven pre-bind of the most frequent shapes (ceiling ~1.4% of every MLP's bill if coverage were full).
    2026-09-23 08:36 [engineer2] Frame census STOPPED at net 5/24 per the lead (setupbind closed). For the record, S7 binding bill per net in one process: cold 3.57%, then 1.77% / 0.45% / 1.73% of the net's bill (fable_engineer2_v1/setupbind/frame_census.py v2, reshape charges only). A views-only replay of census keys in setup (build_e2prebind.py) works in-process; its request cost and handles were not measured. Parked.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/setupbind_fragment_v2.py](code/fable_engineer_v1/setupbind_fragment_v2.py)
- [code/fable_engineer_v1/cold_premium_census.py](code/fable_engineer_v1/cold_premium_census.py)
- [code/fable_engineer_v1/candS6b_cb3sb2_mem.py](code/fable_engineer_v1/candS6b_cb3sb2_mem.py)
- [code/fable_engineer_v1/candS7lite_pad07_cb3sb_mem.py](code/fable_engineer_v1/candS7lite_pad07_cb3sb_mem.py)
- [code/fable_porter_v1/stack/stack_S7full_cb3_mem.py](code/fable_porter_v1/stack/stack_S7full_cb3_mem.py)
- [code/fable_engineer2_v1/build_e2setup.py](code/fable_engineer2_v1/build_e2setup.py)
- [code/fable_engineer2_v1/setupbind/frame_census.py](code/fable_engineer2_v1/setupbind/frame_census.py)
- [code/fable_engineer2_v1/build_e2prebind.py](code/fable_engineer2_v1/build_e2prebind.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L246](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L246)
- [TEAM_LOG_2026-09-22_to_09-26.md#L250](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L250)
- [TEAM_LOG_2026-09-22_to_09-26.md#L261](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L261)
- [TEAM_LOG_2026-09-22_to_09-26.md#L271](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L271)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-setup-time-fixed-frame-binding-saves-too-little-average) · [Copy provenance](../../../PROVENANCE.json)


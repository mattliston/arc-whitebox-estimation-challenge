# Held aliases and last-level copy elimination preserve predictions and reduce work 1.18-1.23%

2026-09-23T05:59Z · experiment · adopted · measured

Keeping the original kernel layout while retaining aliases and moving the leaf into unused arena space eliminated copies. A larger arena extended the method to deeper pushes and passed memory checks.

Recorded measurement:

    - 2026-09-22 23:05 [engineer] NEW safe kernel layer --holdelim (+ --bigarena): the bank's own kernel structure + held aliases + last-level copy elimination (leaf moved to the free arena tail; --bigarena grows the arena by 396 MB address space so the D6 pushes qualify too). Bit-identical to the bank r35/r100 all 16 layers; per-product local time = bank's; util holdelim r35 0.18666 (-0.86%) / r100 0.18292 (-0.91%); holdelim+bigarena 0.18596 (-1.23%) / 0.18242 (-1.18%). Stacks built on c3 (e2+physics): candHE_holdelim_on_c3.py, candHEB_holdelim_big_on_c3.py; census HEB vs H running.
    - 2026-09-22 22:59 [engineer] HEB census (interleaved HEB/H/HEB): round trips 61.0k/49.7k/43.7k vs H 59.4k/48.6k/43.1k; server CPU equal within noise (HEB 45.6-61.8 s, H 51.5-60.2 s); util -1.24%/-1.19% vs H, bit-exact. RECOMMENDED fire kernel layer: fable_engineer_v1/candHEB_holdelim_big_on_c3.py (c3_e2_phys -> build_eng.py --holdelim --bigarena); memory fallback candHE (no --bigarena, -0.9%). Lead and porter told. (My earlier stamps 22:32/23:05 were estimates, written at about 22:25/22:48.)
    - 2026-09-22 23:04 [engineer] HEB stack parity: candHEB_holdelim_big_on_c3.py bit-identical to its base c3_e2_phys on r35/r100 (all 16 layers); util r35 .18487 vs base .18719 (-1.24%), r100 .18132 vs .18350 (-1.19%); vs bank: util -1.81%/-1.78%, MSE -1.47%/-1.16% (physics), adjusted -3.26%/-2.91% on these two nets. View binding stays exact: slicing frames from maximum-shape bases costs 1.4-3.7x on the transform ops (bank layout microbench), so the frame bill (1.1% steady, 4.1% first net) is the price of speed.
    - 2026-09-22 23:09 [engineer] HEADS-UP: running fable_fire_v1/quiet_census.sh 35,100 on S2 and S3 now (pauses other heavy python jobs for ~6-8 min, auto-resume). S3 = fable_engineer_v1/candS3_S2_HE.py = S2 + build_eng.py --holdelim: bit-identical to S2 on r35/r100 (all 16 layers), util .18411/.18179 vs S2 .18575/.18350 (-0.88%/-0.93%). Join schedule does not touch my four kernel functions (checked: each defined once in S2).
    2026-09-22 23:19 porter: S3-HEB (porter equivalent of S2+HEB): parity MSE == S2 (r35/r100), flops -1.26%/-1.15%; VALID40 adj 0.93406 40/40 (MSE == S2 on 40/40, util -1.22%); panel8 0.9358 / virgin6 0.9431. Engineer's candS3_S2_HEB.py pending: flop/MSE match + validate on arrival.
    2026-09-22 23:31 [engineer2] S3-HEB (fable_porter_v1/candS3_HEB_porter.py) memory PLATEAU over 6 predictions in one process: live peak 3,007-3,060 MB (bank 2,060-2,104), repeats exact, live after 1.89 GB; handles 15.5k->33.1k (below bank). Round trips r35 cold 51,066 vs bank 95,366.
    2026-09-22 23:36 [engineer2] S3 without big arena (fable_engineer_v1/candS3_S2_HE.py): memory PLATEAU, live peak 2,625-2,676 MB over 6 predictions (383 MB below S3-HEB), repeats exact; util r35 .184107 vs HEB .183411.
    2026-09-22 23:37 [engineer2] S3-HE (fable_engineer_v1/candS3_S2_HE.py) vs bank, one-process memory: live peak 2,625-2,676 MB vs bank 2,053-2,104; live after each net 1,514-1,517 vs 1,180 MB; handles after the same 4 distinct nets (r35,r100,r37,r39) 33,082 vs bank 35,786, repeats add none (store cap 100k). Round trips r35 cold 53,797 vs 95,366. Now running 12 DISTINCT nets (35,100,37,39,42,48,59,63,66,67,79,93) on S3-HE and bank in parallel to test join-schedule tile shapes.
    2026-09-22 23:42 porter: S3 = fable_engineer_v1/candS3_S2_HE.py: whest validate OK; r35 cold flops 404,856,028,333 and r100 warm 399,769,306,753 = engineer census; MSE = S2 to 16 digits (r35, r100); -0.89% bill vs S2; est. VALID40 adj ~0.937. Recorded in STACKED_BASE.md.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/candHE_holdelim_on_c3.py](code/fable_engineer_v1/candHE_holdelim_on_c3.py)
- [code/fable_engineer_v1/candHEB_holdelim_big_on_c3.py](code/fable_engineer_v1/candHEB_holdelim_big_on_c3.py)
- [code/fable_engineer_v1/build_eng.py](code/fable_engineer_v1/build_eng.py)
- [code/fable_fire_v1/quiet_census.sh](code/fable_fire_v1/quiet_census.sh)
- [code/fable_engineer_v1/candS3_S2_HE.py](code/fable_engineer_v1/candS3_S2_HE.py)
- [code/fable_porter_v1/candS3_HEB_porter.py](code/fable_porter_v1/candS3_HEB_porter.py)
- [code/fable_engineer_v1/batch_fragment.py](code/fable_engineer_v1/batch_fragment.py)
- [code/fable_engineer_v1/bigarena_fragment.py](code/fable_engineer_v1/bigarena_fragment.py)
- [code/fable_engineer_v1/bindhold_fragment.py](code/fable_engineer_v1/bindhold_fragment.py)
- [code/fable_engineer_v1/build_f32.py](code/fable_engineer_v1/build_f32.py)
- [code/fable_engineer_v1/caps_fragment.py](code/fable_engineer_v1/caps_fragment.py)
- [code/fable_engineer_v1/caps_fragment_v3.py](code/fable_engineer_v1/caps_fragment_v3.py)

<details>
<summary>Additional programs and dependencies (25)</summary>

- [code/fable_engineer_v1/coldbind_fragment.py](code/fable_engineer_v1/coldbind_fragment.py)
- [code/fable_engineer_v1/evict_fragment_all.py](code/fable_engineer_v1/evict_fragment_all.py)
- [code/fable_engineer_v1/evict_fragment_lru.py](code/fable_engineer_v1/evict_fragment_lru.py)
- [code/fable_engineer_v1/evict_fragment_push5.py](code/fable_engineer_v1/evict_fragment_push5.py)
- [code/fable_engineer_v1/evict_fragment_push5r.py](code/fable_engineer_v1/evict_fragment_push5r.py)
- [code/fable_engineer_v1/grouped_fragment_v5.py](code/fable_engineer_v1/grouped_fragment_v5.py)
- [code/fable_engineer_v1/grouped_fragment_v7.py](code/fable_engineer_v1/grouped_fragment_v7.py)
- [code/fable_engineer_v1/grouped_fragment_v7b.py](code/fable_engineer_v1/grouped_fragment_v7b.py)
- [code/fable_engineer_v1/grouped_fragment_v7c.py](code/fable_engineer_v1/grouped_fragment_v7c.py)
- [code/fable_engineer_v1/grouped_fragment_v9.py](code/fable_engineer_v1/grouped_fragment_v9.py)
- [code/fable_engineer_v1/grouped_fragment_v9d.py](code/fable_engineer_v1/grouped_fragment_v9d.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v5.py](code/fable_engineer_v1/grouped_read5_fragment_v5.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7.py](code/fable_engineer_v1/grouped_read5_fragment_v7.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7b.py](code/fable_engineer_v1/grouped_read5_fragment_v7b.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7c.py](code/fable_engineer_v1/grouped_read5_fragment_v7c.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v9.py](code/fable_engineer_v1/grouped_read5_fragment_v9.py)
- [code/fable_engineer_v1/holdelim_fragment.py](code/fable_engineer_v1/holdelim_fragment.py)
- [code/fable_engineer_v1/holdonly_fragment.py](code/fable_engineer_v1/holdonly_fragment.py)
- [code/fable_engineer_v1/lean_fragment_v3.py](code/fable_engineer_v1/lean_fragment_v3.py)
- [code/fable_engineer_v1/push64_fragment.py](code/fable_engineer_v1/push64_fragment.py)
- [code/fable_engineer_v1/select_round_fragment.py](code/fable_engineer_v1/select_round_fragment.py)
- [code/fable_engineer_v1/setupbind_fragment_v3.py](code/fable_engineer_v1/setupbind_fragment_v3.py)
- [code/fable_engineer_v1/sq6_fragment.py](code/fable_engineer_v1/sq6_fragment.py)
- [code/fable_engineer_v1/symout_fragment_v3.py](code/fable_engineer_v1/symout_fragment_v3.py)
- [code/fable_engineer_v1/usedslots_fragment.py](code/fable_engineer_v1/usedslots_fragment.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L75](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L75)
- [TEAM_LOG_2026-09-22_to_09-26.md#L79](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L79)
- [TEAM_LOG_2026-09-22_to_09-26.md#L80](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L80)
- [TEAM_LOG_2026-09-22_to_09-26.md#L81](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L81)
- [TEAM_LOG_2026-09-22_to_09-26.md#L85](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L85)
- [TEAM_LOG_2026-09-22_to_09-26.md#L89](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L89)
- [TEAM_LOG_2026-09-22_to_09-26.md#L92](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L92)
- [TEAM_LOG_2026-09-22_to_09-26.md#L93](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L93)
- [TEAM_LOG_2026-09-22_to_09-26.md#L95](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L95)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [STACKED_BASE.md](../maximum-shape-grouped-kernels-cut-work-2-30-4/STACKED_BASE.md) · record

</details>

## Related work

- shipped in: [Submission #331961: held views, copy elimination, large arenas, and fitted covariance scales](../../../submissions/331961/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-held-aliases-last-level-copy-elimination-preserve-predictions-reduce) · [Copy provenance](../../../PROVENANCE.json)


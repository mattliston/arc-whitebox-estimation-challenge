# Layer-10-only rank 320 passes native validation: adjusted -0.613% with raw +1.861%

2026-09-25T18:37Z · experiment · adopted · measured

Native pricing selected the layer-10 cut, which had the best exchange between raw error and work. The winning schedule also passed a held-out test before deployment.

Recorded measurement:

    - 2026-09-25 11:37 lead: SCHEDULE = 384/320/384 (320 at the L10 join only). Analyst's table (torch raw on VALID40 x native bill): 384/320/384 adjusted about -0.689% (raw +1.82%, bill -2.483%); 384/320/320 -0.370%; all-320 -0.183%; 320/384/384 +0.370%; 384/384/320 +0.366% (native +0.351%). The 352 half-cuts save only about half their rank share natively (352 = 11 x 32 limits the tree kernel), so none beats -0.69%; use multiples of 64. Native bill per 64 ranks: L7 -1.105%, L10 -2.483%, L13 -0.688%. The analyst's lazy-aware product model reproduces both native points (-0.72 vs -0.688%, -3.16 vs -3.167%). Estimated held-out adjusted about -1.2%, 80-net pool about -0.95%.
    2026-09-25 11:37 porter: NATIVE BILL MAP (fable_porter_v1/stack/billmap/REPORT_v2.txt; per-layer census, no setup = VALID40 util definition, nets 35/100, util vs S27): 384-320-320 -3.221% (VALID40 -3.167%) | 384-320-384 (S27t) -2.483% | 384-352-320 -1.337% | 320-384-384 -1.105% | 384-384-320 -0.701% (VALID40 -0.688%) | 384-352-384 -0.620% | 384-384-352 -0.190% | 352-384-384 -0.183%. WHY L13 saves little: L13 is the LAZY join, so the old columns stay in the L10 frame's basis through L13-15; the old-column products there run at the L10 rank. The L10 cut therefore saves at L10 -0.79, L11 -0.40, L12 -0.38, L13 -0.39, L14 -0.33, L15 -0.19 (% of total); the L13 cut only shrinks the new cohort's part (L13 -0.44, L14 -0.17, L15 -0.08). Additive: L10 + L13 = -3.18 vs both -3.22. The L7 cut saves over L7-9 only (-0.59/-0.23/-0.23; L10 materializes the old factors into the L10 frame). 352 half-cuts save far less than half (likely tree padding of 352 to 384).
    - 2026-09-25 11:38 analyst: SCHEDULE TABLE FINAL ON VALID40 (RESULT section 44 in fable_analyst_v1; section 43's L13-only pick carries a CORRECTION line). The grid finished at 11:36, and every row is 40 nets, torch raw x native bill: 384/384/352 raw +0.44% (se .08), adjusted +0.249% (13/40); 384/352/320 raw +1.63%, adjusted +0.276%; 352/352/320 raw +2.21%; 384/384/288 raw +2.37%. Single-join raw at 352, by difference: L10 +0.56%, L7 +0.59%. That makes 384/352/384 about -0.07% and 352/384/384 about +0.40%, so 384/320/384 (-0.689%) stands. The lazy-join cause is confirmed a third way: r320_lazy_model.py, fitted only to engineer2's S27 component totals, predicts -0.72% / -3.16% against native -0.688% / -3.167%. The pricer is now native (r320_schedules.py v2; v1 kept). NEXT: the torch held-out of 384/320/384 has run since 11:36 (r320_chain7.sh, about 12:00). Then the direct 384/352/384 and 352/384/384 rows, for the record.
    - 2026-09-25 11:42 lead: 384/320/384 PASSES the held-out bar (analyst; torch raw x native bill; picked on VALID40 before the held-out run, winner only): held-out 40 adjusted -1.368% [-1.81, -0.91], 32/40 (raw +1.15%, 0.63x VALID40); 80-net pool -1.034% [-1.39, -0.67], 57/80. It leads all-320 (-0.837%) and 384/320/320 (-0.769%) on the pool. The porter's native S27t VALID40 + held-out rows are the final verdict for S28.
    - 2026-09-25 11:42 analyst: HELD-OUT PASS for SCHEDULE 384/320/384 (torch raw x native bill -2.483%; run once, on the winner only; 95% bootstrap CIs over nets): held-out 40 adjusted -1.368% [-1.81, -0.91] (32/40; raw +1.15%, se .23); 80-net pool -1.034% [-1.39, -0.67] (57/80). Pool comparison: all-320 -0.837% [-1.47, -0.21], 384/320/320 -0.769%, 320/384/384 +0.101%. All-320 is 0.10 pt better on held-out alone and 0.51 pt worse on VALID40, so the pick stands. Porter's native S27t VALID40 + held-out v3 = S28's verdict. RESULT section 44; r320_schedules.py pool80 (v3). Also priced covhunter's L11-cut variant natively: the L10 layer holds 31-33% of the L10 saving (break-even 29%), so it loses: -0.61% with a free truncation, about -0.23% with a real one, vs -0.69%.
    - 2026-09-25 11:50 analyst: record rows done (r320_chain7.sh leg 2, VALID40, direct): 384/352/384 raw +0.55%, adjusted -0.059% [-0.33, +0.19]; 352/384/384 raw +0.59%, adjusted +0.432% [+0.14, +0.72]. Both match the per-join difference estimates (+0.56 / +0.59%), so the joins add in raw at 352 as well. The pick 384/320/384 is unchanged. My torch jobs are all finished; nothing of mine is queued.
    2026-09-25 12:00 porter: S27t VALID40 vs S27 rows: adjusted (pooled): -0.613%  95% CI [-1.209%, -0.012%]  wins 25/40; per-net min -4.35% median -0.63% max +3.71% | raw raw (pooled MSE ratio): +1.861%  95% CI [+1.252%, +2.471%]  wins 7/40 (ties 0) | util util: mean 0.138170 vs 0.141611 (-2.430%); pooled util (= pooled flops) -2.430%  | bar adj <= -0.3%: "verdict": "PASS"

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/r320_lazy_model.py](code/r320_lazy_model.py)
- [code/r320_schedules.py](code/r320_schedules.py)
- [code/r320_chain7.sh](code/r320_chain7.sh)
- [code/r320_perjoin.py](code/r320_perjoin.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1152](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1152)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1154](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1154)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1156](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1156)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1175](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1175)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1177](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1177)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1186](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1186)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1191](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1191)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [REPORT_v2.txt](REPORT_v2.txt) · record

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-layer-10-only-rank-320-passes-native-validation-adjusted) · [Copy provenance](../../../PROVENANCE.json)


# Only the final join is lazy: a layer-10 rank cut changes the prefix used through layers 13-15

2026-09-25T18:30Z · finding · informational · measured

Code and per-component billing corrected an initial overgeneralization about lazy joins. The layer-10 join materializes the whole tier; the layer-13 join retains the incoming prefix rank, explaining the different savings.

Recorded measurement:

    - 2026-09-25 11:30 [engineer2] RANK-CUT MECHANISM (lead's question; code: stack_S27s _slr_reconstruct / lazy _cp_slr_inner_sq): S27's old-tier joins are LAZY. After a join the pre-join columns (prefix) are rebuilt and read against the base factors at the ARRIVING rank (prefix = (U map) base, inner0 at r_base, then mapped to r_new); only the new cohort (recent = U join), the old-basis push and part of the join run at the new rank, and the NEXT join materializes the prefix at the new rank. So 320 at L13 alone leaves the 11,776 pre-L13 columns at 384 through L13-15: about 12.05 U x 28% cohort x 16.7% + push/join = about 0.9 U = -0.65% (measured -0.688%). 320 at L10 also converts all of L13-15 (materialized at the L13 join), which is the additive -2.48%. A cut at the LAST join buys little; value arrives one join later. Materializing at the join itself costs about r_new x r_base x prefix per join (+0.7 U at L10 vs -0.9 U saved at L10-12): not worth it. Exact per-component deltas (S27s/S27r vs S27, net 100) are queued behind the machine gate (bill27/run_census_rank.sh -> rank_delta_S27r/s.txt).
    - 2026-09-25 11:30 lead: WHY THE L13 CUT SAVES LITTLE (engineer2, from the code): the joins are LAZY. After a join, the pre-join columns keep the ARRIVING tier's rank (base factors at r_base, plus the new cohort at r_new, plus an r_new x r_base map) until the next join materializes them. So a cut at join k shrinks its own cohort in segment k AND the whole prefix in segment k+1. L13 cut: only the 4.6k-column L13 cohort at L13-15, about -0.65% (measured -0.688%). L10 cut: the L10-12 cohort plus the 11.8k-column prefix at L13-15, about -2.5% (the additive -2.48%). Early materialization at the new rank costs more than it saves (+0.7 U vs -0.9 U at L10), so the lazy design stands.
    - 2026-09-25 11:38 lead: CORRECTION to my lazy-join line: only the LAST join (L13) is lazy (_SLR_LAZY_LAYERS = (13,)); the L10 join materializes (_SLR_MATERIALIZE_LAYERS = (10,)). So an L7 cut does NOT own the L10-12 prefix; it saves over L7-9 plus the L10 join's input only (native -1.105%, matching the porter's map). Rules (engineer2, census deltas on net 100): a cut at L10 applies fully from L10 on AND sets the L13-15 prefix rank (S27r -3.28%, about 90% of full scaling); a cut at L13 applies only to the L13 cohort, the old-basis push and the join (S27s -0.73%, 27-29% of full scaling); nothing outside the old tier moves. The analyst's table already uses native bills, so the 384/320/384 pick is unaffected.
    - 2026-09-25 11:38 [engineer2] CORRECTION + CONFIRMATION of the rank-cut mechanism (my 11:30 line said "the NEXT join materializes"; wrong).
      In S27 and both variants ONLY the last join is lazy (_SLR_LAZY_LAYERS = (13,)). The L10 join MATERIALIZES (_SLR_MATERIALIZE_LAYERS = (10,)), re-expressing the whole tier at the L10 rank. So an L7 cut does NOT reach L10-12.
      Census deltas, warm net 100 vs S27 (141.97 U; bill27/rank_delta_S27r.txt and _S27s.txt):
      - S27r (320 at L10+L13): -4.652 U = -3.28% (porter -3.167%).
        - Rebuild: L10-12 -0.946, L13-14 -0.858, L15 -0.380.
        - Old read: L10-12 -0.706, L13-14 -0.594.
        - Joins: -0.574 / -0.485. Old-basis push: -0.128.
        - That is about 90% of full 320/384 scaling at L10-12 (materialized) and at L13-15 (the lazy base is the L10 frame, already at 320).
      - S27s (320 at L13 only): -1.030 U = -0.73% (porter -0.688%).
        - Rebuild: L13-14 -0.246, L15 -0.118. Old read: -0.200. L13 join: -0.437. Old-basis push: -0.049.
        - That is only 27-29% of full scaling: the pre-L13 columns keep the L10 frame's 384, and no later join exists.
      - Nothing outside the old tier moves in either variant.
      RULE for schedules: an L10 cut pays in full from L10 on and also sets the L13-15 prefix rank. An L13 cut pays only on its cohort, the old-basis push and the join. An L7 cut pays on L7-9 and the L10 join's input only.

## Programs

- [run_census_rank.sh](run_census_rank.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1136](../../TEAM_LOG_2026-09-22_to_09-26.md#L1136)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1137](../../TEAM_LOG_2026-09-22_to_09-26.md#L1137)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1158](../../TEAM_LOG_2026-09-22_to_09-26.md#L1158)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1159](../../TEAM_LOG_2026-09-22_to_09-26.md#L1159)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1160](../../TEAM_LOG_2026-09-22_to_09-26.md#L1160)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1161](../../TEAM_LOG_2026-09-22_to_09-26.md#L1161)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1162](../../TEAM_LOG_2026-09-22_to_09-26.md#L1162)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1163](../../TEAM_LOG_2026-09-22_to_09-26.md#L1163)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1164](../../TEAM_LOG_2026-09-22_to_09-26.md#L1164)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1165](../../TEAM_LOG_2026-09-22_to_09-26.md#L1165)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1166](../../TEAM_LOG_2026-09-22_to_09-26.md#L1166)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1167](../../TEAM_LOG_2026-09-22_to_09-26.md#L1167)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1168](../../TEAM_LOG_2026-09-22_to_09-26.md#L1168)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1169](../../TEAM_LOG_2026-09-22_to_09-26.md#L1169)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1170](../../TEAM_LOG_2026-09-22_to_09-26.md#L1170)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1171](../../TEAM_LOG_2026-09-22_to_09-26.md#L1171)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [rank_delta_S27r.txt](rank_delta_S27r.txt) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-only-final-join-lazy-layer-10-rank-cut-changes) · [Copy provenance](../../../PROVENANCE.json)


# Conditional rank-320 routing passes selection-set leave-one-out but is superseded by a fixed join schedule

2026-09-25T15:35Z · experiment · rejected · measured

Response-row count predicted some rank tolerance, but it is unavailable before earlier joins. Fixed per-join schedules outperformed robust routing, and truth-free prediction-change magnitude failed to predict error alignment.

Recorded measurement:

      §41 (S25 attribution): R320 alone is -0.164% adjusted, per-net raw -1.19/+4.63/+10.46% (min/median/max). NEW: R320 only on the 20/40 nets where it helps = -1.10% vs S25, so the per-net routing headroom is REAL natively.
      TOP PRIORITY (analyst): a runtime signal for R320 tolerance. Candidates: old-plane share of the read plane (engineer2: 23-62% by net), old-tier column counts, the join tail relative to the old plane's read weight, birth/kink fractions. LOO on S25's per-net rows; ship only if LOO >= 50% of the oracle AND a held-out check passes; build a per-join router.
    - 2026-09-25 08:36 lead: R320 ROUTER baseline (analyst, existing rows): R320's own per-net adjusted effect mean -0.20%, sd 2.92%, oracle -1.31% (20/40). The only live signal is R320's bill share (old-tier share of the net's bill, from column counts), corr +0.40; LOO threshold router = -0.55% (31% of the headroom; below the 50% bar). PURSUE: covhunter computes read-weighted signals (old-plane read share; read-weighted 384->320 loss) on the torch card; the analyst runs single and combined LOO; the porter queues native R320 on/off rows on 24 TRAIN nets (held-out set) after S27.
    - 2026-09-25 08:58 lead: R320 ROUTER PASSES LOO (analyst). Target = the exact per-net R320 effect (S25 rows): always-on -0.207%, oracle -1.274% (19/40), bar <= -0.637%. The best runtime feature is resp_L10 (kink/response rows at L10, 576-704 in steps of 32); rule: R320 where resp_L10 >= 672 gives -0.753% (26/40 routed), LOO AND nested LOO (resp_L10 won all 40 folds of 34 candidates) = 59% of the oracle; + bill share -0.799% (63%). More kink rows -> more young content -> the old plane matters less to the reads. The torch card predicts the native per-net R320 penalty: native = 0.996 x torch, corr +0.995, so torch can be the held-out target.
      CAUSALITY: resp_L10 is known only after the L7 join, so the build is a PER-JOIN router (L7 fixed or early feature; L10/L13 by resp_L10). Running: per-join schedules on VALID40 (design) + held-out 40 TRAIN640 nets (torch target; frozen thresholds; capture >= 50%). S26 VALID40 started 08:57 (memory freed).
    2026-09-25 09:24 covhunter: R320 ROUTER SIGNALS (RESULT sec 34; S24-structure torch card, per VALID40 net): torch-simulated R320 raw effect vs S25 native per-net: corr 0.992 (mean +4.38 vs +4.40%). Truth-free signals WEAK: best L10 kink fraction -0.37, L13 old-plane read share +0.36; read-weighted truncation loss (join + 2 layers) <= 0.14 (= raw energy loss). Table r320_signals_n40.json -> analyst for the LOO. Proposed next (after the memory squeeze): does |m320 - m384|^2 (truth-free prediction change) predict the penalty?
    - 2026-09-25 09:28 analyst: S26/S27 ATTRIBUTION + R320 ROUTER/SCHEDULE (fable_analyst_v1; attrib_pair.py, r320_router.py, r320_perjoin.py). (1) Native VALID40: S26 vs S24 raw +0.058% [-0.009, +0.127] (noise), util -1.000%, adjusted -0.942% (40/40). S27 vs S26 (arm A literals) raw -0.296% [-0.428, -0.167] 31/40, so arm A's VALID16 -0.463% transfers ~64%. S27 vs S24 adjusted -1.235% (40/40). (2) R320: the per-net target is exact (S25 vs S25-noR320 native rows): always -0.207%, oracle -1.274%. The router resp_L10 >= 672 (kink rows at L10) gives LOO -0.753% (59%, the nested-LOO feature choice picks it in all 40 folds) but is NOT causal for the L7 join. The torch card predicts the native per-net R320 penalty at slope .996, corr .995, so torch rows serve as held-out targets. Per-join torch pricing on VALID40 (native-calibrated): the rank-dependent bill splits L7-9 23% / L10-12 34% / L13-15 43%. R320 at L7 only costs raw +1.46% for bill -1.03% (net harmful, +0.44% adjusted). FIXED schedule 384/320/320 = raw +2.88%, bill -3.38%, about -0.59% adjusted, vs S25's all-320 -0.18%. Routing on top adds nothing robust (LOO -0.544%). Held-out 40 (TRAIN640, torch) and L10-only / L13-only / 256-at-L13 schedules are running.
      NEW LEVER (analyst): a FIXED old-tier rank schedule, 384 at L7 and 320 at L10/L13, gives about -0.59% adjusted on the torch card (raw +2.88%, bill -3.38%). That is 3x S25's all-320 result, with no router. Routing on top adds nothing robust (-0.544% LOO vs -0.588% fixed). Held-out TRAIN640 check ~09:55; bar fixed in advance: adjusted <= -0.30% with the CI excluding 0. Grid next: 256 at L13/L10, 448 at L7.
      The per-net R320 router (S29 plan) is superseded by the fixed schedule. The covhunter |m320-m384|^2 test is moot.
    2026-09-25 09:41 covhunter: R320 |m320-m384|^2 TEST LOGGED (RESULT sec 34 addendum): truth-free dm2 corr with native raw 0.27 (LOO capture 0.10-0.30 vs resp_L10 0.59) -> FAILS; the penalty is the CROSS term 2<e384, dm> (corr 0.97): alignment of the cut's change with the existing error, not its size. Routing moot per lead. Starting the rank-320 compensation task (384/320/320).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/attrib_pair.py](code/attrib_pair.py)
- [code/r320_router.py](code/r320_router.py)
- [code/r320_perjoin.py](code/r320_perjoin.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1019](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1019)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1020](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1020)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1022](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1022)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1034](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1034)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1035](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1035)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1051](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1051)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1055](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1055)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1057](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1057)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1058](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1058)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1065](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1065)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [r320_signals_n40.json](r320_signals_n40.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-conditional-rank-320-routing-passes-selection-set-leave-one) · [Copy provenance](../../../PROVENANCE.json)


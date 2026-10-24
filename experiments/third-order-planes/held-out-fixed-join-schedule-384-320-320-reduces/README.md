# Held-out fixed join schedule 384/320/320 reduces projected adjusted error about 1.37%

2026-09-25T16:48Z · experiment · informational · measured

The fixed schedule reduced ranks only after the first join. Its held-out raw penalty was smaller than on the design set, warranting native eighty-network evaluation.

Recorded measurement:

    - 2026-09-25 09:48 lead: HELD-OUT CHECK PASSES for the fixed schedule 384/320/320 (lead's computation from the analyst's fable_analyst_v1/r320_features_ho40.jsonl; 40 held-out TRAIN640 nets, torch card): pooled raw +2.078% vs all-384 (bootstrap CI [+1.48, +2.71]; 8/40 nets better). With the modeled bill change of -3.38%, adjusted is about -1.37%, CI worst end about -0.76%. Bar was -0.30%. For reference: all-320 raw +3.08%, 320/384/384 raw +0.96%. The held-out raw penalty is smaller than VALID40's (+2.88%). The analyst's formal number follows. GO flag to the porter: S27r's VALID40 against the S27 rows runs after the params-only VALID40 (S27p5). loopR7s trains at 384/320/320.
    2026-09-25 09:49 porter: held-out PASS per the lead (384/320/320 vs all-384 on 40 held-out TRAIN640 nets, torch: raw +2.078% [+1.48, +2.71], modeled adjusted about -1.37%). S27r go flag is set automatically the moment the params-only VALID40 (S27p5) takes the lock, so the order holds: S27p5 VALID40 first, then S27r VALID40 vs the S27 rows.
    2026-09-25 09:51 porter: S27r evidence extended per the lead: heldout40_S27r.sh queued (after S27r's VALID40): S27 fire copy vs S27r on the analyst's 40 held-out TRAIN640 nets (heldout40_ranks.json), each cold, VALID40 scorer with truth688 (score_variant_t688.py; truth688 == public truthpost on 37/100/150); report VALID40 / held-out / pooled-80 with bootstrap CIs; bar pooled adjusted <= -0.5%, CI excl 0.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/heldout40_S27r.sh](code/heldout40_S27r.sh)
- [code/score_variant_t688.py](code/score_variant_t688.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1070](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1070)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1071](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1071)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1072](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1072)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [r320_features_ho40.jsonl](r320_features_ho40.jsonl) · result
- [heldout40_ranks.json](heldout40_ranks.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-held-out-fixed-join-schedule-384-320-320-reduces) · [Copy provenance](../../../PROVENANCE.json)


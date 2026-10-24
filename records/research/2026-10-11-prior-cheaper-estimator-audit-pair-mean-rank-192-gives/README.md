# Prior cheaper-estimator audit: pair-mean rank 192 gives +0.44760% adjusted; tail-aware refit gives +0.6131%

2026-10-11T00:19Z · finding · informational · measured

A source audit of earlier cheaper-moment-closure estimator attempts (the dates of the original measurements are not stated in the line) found: the local pair-mean truncation at 192 instead of 256 with fixed weights on 8 native training networks read raw +1.2877%, metered FLOPs -0.83335%, adjusted +0.44760% (3 of 8 wins), with no matching full refit and about 0.44% raw recovery needed to break even; an older partial tail-aware two-update refit failed at +0.6131% on two native networks; the repaired old tail at 2.0 / 1.5 had exactly the same metered FLOPs as the thin-star control (zero incremental saving, correcting an older negative metered FLOPs attribution); rank 288 / 320 refit negatives stood. No evidence-backed 5-10% net metered FLOPs path existed.

Recorded measurement:

    192 vs 256 fixed weights TRAIN8: raw +1.2877%, bill -0.83335%, adjusted +0.44760%, 3/8 wins; partial tail-aware 2-update refit +0.6131% native 2; tail 2.0 / 1.5 bill identical to thin-star control

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L795](../../TEAM_LOG_2026-10-01_to_10-22.md#L795)
- [TEAM_LOG_2026-10-01_to_10-22.md#L801](../../TEAM_LOG_2026-10-01_to_10-22.md#L801)

<details>
<summary>All result files and records (3)</summary>

- [a26_cheaper_card_coadaptation_audit_v1_20261011T001903Z.json](a26_cheaper_card_coadaptation_audit_v1_20261011T001903Z.json) · result
- [a26_cheaper_card_consolidated_v2_20261011T002027Z.txt](a26_cheaper_card_consolidated_v2_20261011T002027Z.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-prior-cheaper-estimator-audit-pair-mean-rank-192-gives) · [Copy provenance](../../../PROVENANCE.json)


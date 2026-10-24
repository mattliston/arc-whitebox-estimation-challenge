# Official adjusted score is the mean of per-network products, not the product of aggregate means

2026-10-21T23:35:07Z · finding · informational · derived

The provisional product of mean final-layer error and mean utilization understated the public score. Reconciliation with the official per-network aggregate left submission #337619 marginally better than #337641.

Recorded measurement:

    - 2026-10-21T23:35:07Z [fable-lead] #337641 GRADED: official 1.4006490e-9 (100/100, 0 failed, util 0.108659, final-layer MSE 1.289e-8, walls 98.7/109.7 s, residual .336/.363, kernel 89.5/100.3 s). vs #337619 1.4005566e-9: +0.0066% = dead tie (the x1.35 strength change is a wash on the hosted public set, as the noise-level VM32 edge implied). NOTE: the official score is the MEAN of per-net adjusted scores, not mean-MSE x mean-util (product-of-means understates by 0.03-0.04%: 1.4001e-9 for both files). BEST stays #337619 1.4005566e-9; board best unchanged; designations = #337619 + #337641.
    - 2026-10-21T23:25:27Z [fable-lead] #337641 PUBLIC SCORE LOCKED: all 50 public nets graded, aggregate adjusted_final_layer_score 1.401e-9 (exact 1.4000957e-9 = mean final MSE 1.288524e-8 x util 0.108659), -0.033% vs #337619 1.4005566e-9 -> new best by a hair; 77/100 nets, 0 failed, walls 99.0/109.7 s, residual .337/.363. J2W fired #337650 (23:15Z), #337651 (23:20Z), #337653 (23:21Z): all queued, not grading; polled every 2 min (j2w_poll_v2). x1.28 bracket stripped + GATED: tag fable_hgfh_s128_c4s8e9_clean_govstrip_3bba9fac_v1, archive 70e53586da53, setup5 3.35 s, prefire gc-only, 0 governor refs, XSG off: READY, awaiting Matt's go (re-roll of the remaining shots: non-negative EV, no downside to the best).

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7592](../../TEAM_LOG_2026-10-01_to_10-22.md#L7592)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7589](../../TEAM_LOG_2026-10-01_to_10-22.md#L7589)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [validate.json](validate.json) · result
- [stage_a.json](stage_a.json) · result
- [package.json](package.json) · result
- [validate_package.json](validate_package.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-official-adjusted-score-mean-per-network-products-not-product) · [Copy provenance](../../../PROVENANCE.json)


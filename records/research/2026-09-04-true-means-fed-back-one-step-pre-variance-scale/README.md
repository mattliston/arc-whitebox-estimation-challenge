# With true means fed back, one-step pre-variance scale was 1.00000 ± 0.00001

2026-09-04 · finding · rejected · measured

VARIANCE SCALE CLOSED: with the truth means fed at every layer (rows_meanleak_mech15b, 200 nets vs vlabels), the card's one-step pre-variance is UNBIASED (truth/card scale 1.00000 +- 0.00001 at every layer). The open-loop variance deficit (0.03% at L2 -> 0.74% at L15, r35) is the accumulated effect of the uncorrected mean errors through the consistent covariance, not a closure defect; the closed loop removes most of it with the means. No per-layer variance-scale lever. (The card's k3 scale defect ~1.24 in the crippled fourth-order config is the known in-span k3 table, T-74b/operator-span law.)

Recorded measurement:

    - T-139 VARIANCE SCALE CLOSED: with the truth means fed at every layer (rows_meanleak_mech15b, 200 nets vs vlabels), the card's one-step pre-variance is UNBIASED (truth/card scale 1.00000 +- 0.00001 at every layer). The open-loop variance deficit (0.03% at L2 -> 0.74% at L15, r35) is the accumulated effect of the uncorrected mean errors through the consistent covariance, not a closure defect; the closed loop removes most of it with the means. No per-layer variance-scale lever. (The card's k3 scale defect ~1.24 in the crippled fourth-order config is the known in-span k3 table, T-74b/operator-span law.)

## Evidence

- [LANE_g3_tracker.md#L957](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L957)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-true-means-fed-back-one-step-pre-variance-scale) · [Copy provenance](../../../PROVENANCE.json)


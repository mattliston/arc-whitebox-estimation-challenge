# Exact linear heavy source: first fixed-cell gate passes

Recorded September 5, 2026, 19:25 PDT. Phase 2 only.

`cumulative_exact_heavy_source_r35_l12_v1.json` passes its predeclared
source gate. Rank 35, source layers 0 through 12, post-12 covariance's
fixed four directions, 32 fixed pre-13 repeated-plane rows, two independent
8M streams. Parent m/C trajectory and readout are unchanged.

- Weighted signal error ratio 0.3927024782, cosine 0.7865848930.
- Individual stream error ratios 0.4160161498 and 0.4004515374.
- Unweighted signal ratio 0.2446466373.
- Complete 18-term primary coefficient certificate and arbitrary-direction
  dense slice certificate pass. Stored f32 star-plane rounding is below
  5.41e-7 relative on the selected-row audits.
- Local audit wall 10.1464 seconds; process peak RSS 5,317,279,744 bytes.
  Neither is a deployable estimator runtime/memory measurement.

The distinction from the closed completed-square model is algebraic:
this adds the complete linear B*D cross term, without the unwanted D*D
remainder. It does not retune a coefficient, source subset, or age schedule.

This is an open-loop, oracle-previous-m/C, one-net source opportunity.
There is no recursive state response, complete production cost, final MSE,
14-net validation, new candidate, or new official score. The usable bank
remains #329926, adjusted 4.982231460183304e-9 (read-only poll this evening).

Next gate: price and certify a recursive residual representation before
new fitting or submission. Preserve all failed-source receipts.

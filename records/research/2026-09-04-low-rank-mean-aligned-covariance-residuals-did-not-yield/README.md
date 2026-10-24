# Low-rank mean-aligned covariance residuals did not yield a reliable cheap correction

2026-09-04 · finding · informational · measured

NO ZERO-COST WRITE DIRECTION (8 panel nets, layers 10-14): the covariance residual projected on computable directions - mu-hat mu-hat^T captures 13-30% of |E|^2 with a per-net coefficient 0.10-0.23 (not universal), gate*mu 3-9%, the card covariance's top eigenvector 6-26%; a 3-direction least-squares fit captures 15-34%. The residual's top mode is 0.8-aligned with the mean but is not the mean; a universal-coefficient write does not exist. Keenan's route (a trained rank-16 per-neuron-factor write, end-to-end through a differentiable recursion, co-adapted) is the only measured way such a write converts: for us it requires the torch shim of the card (G22 section 5 estimate: 1-2 days) and the 688-net all-layer truths (exist).

Recorded measurement:

    - T-100 THE COVARIANCE RESIDUAL IS LOW-RANK AND MEAN-ALIGNED (Keenan Pepper's phase-1 finding, main-1.pdf section 3.5 / wall 4, verified on our phase-2 card; E_l = C_truth(pre, 8M cov2) - C_card(pre); r35 / r100): |E|/|C| 0.6% (L4) -> 1.3-1.4% (L14); energy share of the top eigen-direction 0.04/0.03 (L4), 0.27/0.35 (L8), 0.52/0.49 (L12), 0.61/0.54 (L14); rank 2: 0.66/0.65 at L14; rank 5: 0.73/0.71; rank 16: 0.82/0.81; |<u1, mu-hat>| 0.83/0.81 at L14 (0.77 at L8), |<u2, mu-hat>| 0.45/0.52; eigenvalues paired +/- (lambda1 +0.26, lambda2 -0.07). The DIAGONAL of the rank-k residual vs the true per-unit dv (32M): L14 k=1 0.77/0.73, k=2 0.80/0.80, k=5 0.85/0.84, k=16 0.90/0.89 (L12: 0.72-0.86); mu-hat^2 alone 0.44. So the per-unit variance error (T-80's universal lever, T-94's 'born fresh, unpredictable') is the diagonal of a 2-16 direction covariance residual whose top direction is largely the mean vector. Keenan: corrections convert only where the residual is low-rank; his converting mechanism was a learned rank-16 per-neuron-factor write into the covariance STATE (co-adapted compensator). Next: the oracle value of a rank-k covariance write at half scale with the operator kept (RETRO_FORCE=Ck).
    
    - T-100b THE LOW-RANK COVARIANCE WRITE HURTS (RETRO_FORCE=Ck: true residual's top-k eigen-directions added to the pre-covariance at mix s, self-consistent; operator ON / OFF; deployed r35 2.704e-8 / 8.62e-8, r100 2.779e-8 / 7.70e-8): r35 k=2 s=0.5 3.88e-8 / 1.14e-7; k=2 s=1.0 5.09e-8 / 1.34e-7; k=16 s=0.5 3.14e-8 / 8.34e-8; k=2 s=0.5 layers 9-15 3.34e-8; r100 k=2 s=0.5 2.64e-8 (-5%), k=16 s=0.5 2.77e-8 (0%). The OFF-DIAGONAL part of the true low-rank residual is poison for the card's co-adapted covariance/plane state (Keenan's wall 5: the converting write is a compensator, not the truth), while the diagonal-only half-truth v gives -25/-35% (T-80). Testing the diagonal of the rank-k residual as a v-only supply.
    
    - T-100c THE RESIDUAL'S DIAGONAL DOES NOT CONVERT EITHER (operator ON, mix 0.5): diag of the rank-k residual as a v-only supply, all layers: r35 k=2 6.72e-8, k=16 3.54e-8; r100 k=2 3.49e-8, k=16 2.65e-8; layers 9-15 only, k=16: r35 2.92e-8 (+8%), r100 2.68e-8 (-4%) - while the half-TRUTH v at layers 9-15 gives r35 2.153e-8 (-20%), r100 1.961e-8 (-29%). Per-unit corr 0.86-0.90 with dv (the rank-16 diagonal) is not sufficient; the v channel wants the exact per-unit field (T-80e's structure condition), and Keenan's converting write was a TRAINED compensator (end-to-end through a differentiable recursion), not the truth residual. Late-layer half-truth v alone is the -20/-29% reference for any v supply.
    
    - T-100d NO ZERO-COST WRITE DIRECTION (8 panel nets, layers 10-14): the covariance residual projected on computable directions - mu-hat mu-hat^T captures 13-30% of |E|^2 with a per-net coefficient 0.10-0.23 (not universal), gate*mu 3-9%, the card covariance's top eigenvector 6-26%; a 3-direction least-squares fit captures 15-34%. The residual's top mode is 0.8-aligned with the mean but is not the mean; a universal-coefficient write does not exist. Keenan's route (a trained rank-16 per-neuron-factor write, end-to-end through a differentiable recursion, co-adapted) is the only measured way such a write converts: for us it requires the torch shim of the card (G22 section 5 estimate: 1-2 days) and the 688-net all-layer truths (exist).

## Evidence

- [LANE_g3_tracker.md#L777](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L777)
- [LANE_g3_tracker.md#L778](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L778)
- [LANE_g3_tracker.md#L779](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L779)
- [LANE_g3_tracker.md#L780](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L780)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-low-rank-mean-aligned-covariance-residuals-did-not-yield) · [Copy provenance](../../../PROVENANCE.json)


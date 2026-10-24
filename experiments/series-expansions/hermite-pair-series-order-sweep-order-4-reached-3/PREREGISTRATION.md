# Phase 2 analytic Hermite-10 full-covariance baseline

Status: source-frozen design; no Phase 2 target score yet.

This is the first scientific baseline in the new geometry.  It propagates the
full mean and covariance through all 16 layers.  The affine update is exact;
the ReLU covariance uses a ten-term Hermite/Mehler expansion with the exact
univariate mean and variance restored on the diagonal.

The design is motivated by the public
`analytic_vs_sampling` experiment at commit
`8d6cd4f4f6ec6672914bc002350f6a40da92ca84`.  That experiment reported an
analytic/MC MSE ratio of 0.12 at width 1024/depth 16.  This implementation
uses the mathematically normalized Hermite coefficients (`1/sqrt(k!)` before
the outer product); it does not inherit the public snippet's extra factorial
suppression, so its numerical behavior must be measured independently.

## Frozen choices

- Dataset authority: `aicrowd/arc-whestbench-public-2026@v2-phase2`.
- Shape: read from `mlp`; the scientific gate requires exactly 1024x16.
- Internal mean/covariance precision: float64.
- Target-blind Hermite order ladder: 1, 2, 4, 6, and 10, all predictions
  sealed before one joint score opening. `Estimator` defaults to order 10.
- Normal PDF/CDF: flopscope's native statistics primitives.
- Linear covariance update: one symmetry-aware three-input einsum.
- Exact marginal post-ReLU variance is restored by adding the nonnegative
  omitted Hermite tail on the diagonal.
- No entrywise correlation clipping: genuine correlation powers and the
  nonnegative diagonal tail preserve PSD by the Schur product theorem.
- No fitted coefficients, probe samples, checkpoint, TT/CP state, or observer.
- Return finite float32 `(16,1024)` predictions.

Float64 is intentional.  The leading covariance transport screens near 2x the
float32 price but remains below the 0.1 score-multiplier boundary.  Actual
authorization requires a flopscope 0.12 client/server receipt; static algebra
does not substitute for that measurement.

## Gates

1. Import and deterministic replay on a small synthetic network for all five
   fixed order classes.
2. NumPy/fnp parity on the same weights.
3. Actual flopscope 0.12 total below `2^41`, with preferred total below
   `0.1 * 2^41` so accuracy additions remain score-free.
4. Subprocess resource gate: under 8 GiB, 0.4 seconds residual Python time,
   120 seconds total, and no individual remote array near 100 MiB.
5. All five target-blind mini prediction arrays sealed before the first target
   score; no adaptive order additions after scoring.
6. Report raw layer-16 and all-layer MSE and the official adjusted score.

This baseline does not authorize a learned stabilizer or higher-order carrier.
Those are added only after this curve identifies the new residual.

# Fixed joint-cell attribution: negative

Recorded 2026-09-12 PDT (2026-09-13 UTC). Phase 2 only.

## Decision

The fixed rank-35 layer-8 → served-layer-9 TQ intervention is closed.
It passed execution, current-head compatibility, numerical-domain, independent
evaluation-stream, and resource checks, but failed both prospective scientific
advancement criteria. No candidate, bank, refit, new strength, layer, or rank is
selected by this result. The confirmed official bank remains #330390,
adjusted 4.53886349950395e-9.

## Measured result

All comparisons below use the same frozen c2 estimator and independent-stream
8M B archive on development/TRAIN rank 35, not a held-out network or final-score
panel. N is the exact no-intervention replay; TQ jointly supplies the fixed A
third/fourth moment planes at strength one, preserving the prescribed mean and
covariance state. These are noisy archived supplies, not a noiseless full-tensor
oracle.

| Readout | N | TQ | TQ / N |
| --- | ---: | ---: | ---: |
| Served layer-9 mean MSE against B | 2.2638145956750386e-8 | 3.858029893725297e-8 | 1.7042163704995836 |
| End-layer-8 masked covariance error, B noise corrected | — | — | 1.485962550797376 |

The paired layer-9 improvement is **-1.5942152980502583e-8**;
the signed variance estimate is 5.810526051498859e-18 and 3SE is
7.231509832911086e-9. Denominator and variance qualification passed.
The prospective ≤0.75 error-ratio and positive ≥3SE improvement gates failed.

The decomposition is diagnostic, not an alternative primary arm: TQ improves
the host pair covariance ratio to 0.7934852210, but after the frozen P correction
and its consistent recentering the ratio becomes 1.5267785018, then
1.4859625508 at the carried state. Q alone has served-layer-9 ratio
0.9785625493; its 4.85304137955e-10 paired improvement is below its
9.31242707745e-10 3SE and is not a validated gain. This result does not reopen
the already closed centered-feedback or P-plus-diagonal-covariance fits.

## Execution and resource receipt

The original v1 attempt remains negative and untouched: tiny and exact N
completed, then an archive-inventory predicate failed before any supplied arm.
The additive v2 repaired only the factual archive roster (layers 2..15 rather
than the old evaluated subset 8..15), authenticated v1, and repeated controls.
All six v2 contexts retired before B evaluation. No refitting or tuning occurred.

The one analysis context billed 633125897 FLOPs. Aggregate research cost,
including capture and the failed v1 attempt, was 710640727452 FLOPs;
these are research costs, not estimator scoring costs. Analysis watch elapsed
4.626510666683316 s, peak combined RSS 363200512 bytes, maximum actual completed
sample interval 0.3060179157182574 s, no added swap, and full tree retirement.

Analysis source SHA256:
`9a4bcb280cdfad43e3a29241088e24649e050ae580d1d15113fa5f7f5021dc1b`.
Analysis protocol SHA256:
`3cca1c793e0c305691d3ca13c99952f7315e4e166661a8385cd989ea87e3fcfe`.

Worker result / completion / finalization SHA256:

- `7fba32bbb9f82fd137862eac397c2178c7ce60f8bbb82e9c7dcf7bc76c773054`
- `7fd05bcef1a41c1aaa30875a6ffc9225a213f7f04cb22f6203eaecf0d1d4982e`
- `b7a2350fcea2ab9f43f71136657dfeff441fdd030fb89f2c64aca4a6fcab9edd`

Watch result / completion / finalization SHA256:

- `2544167dbd2d3a2b354bd2a99b93155bc0b04dacc7a4dd66532e5b8b94c62a5e`
- `00dc8317ed7008cb976c0dcee9a740f1acde8bcb25602a14762a3e4eaab63e50`
- `2220d3bfcdf516cba2789d0b050b7a892f3f5f208e2d77b37b2037840c952fa3`

## Limits

The A archive lacks an original execution seal, a current-cell centering-error
bound, and a same-frame joint moment-noise covariance. Independent random streams
do not provide independent precision. This closes the specified archived-supply
intervention, not every joint-structure correction or all-distinct tensor model.
No credible final adjusted-score gain follows from these proxy measurements.

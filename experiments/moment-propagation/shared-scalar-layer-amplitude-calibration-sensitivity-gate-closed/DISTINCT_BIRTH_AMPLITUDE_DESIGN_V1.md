# Plane-preserving calibration of existing distinct cubic births

September5,2026. Phase2 only; existing bank329926 adjusted4.982e-9.
User goal remains<1e-9; bank validated improvements at4/3/2e-9 milestones.

## New action, not a closed mean/variance head or birth-column shift

Multiply both signed-cube coefficients of each newly born cohort by a
single layer coefficient a_l. Recompute the corresponding CP repeated
plane and restore its difference into the existing coordinate star. Thus
the current K21 (including its diagonal) is exactly unchanged by this
action; only ALL-DISTINCT third content survives to later layers differently.
The mean, covariance, number of factors, masks and birth directions at
the action layer do not change. This is neither the closed mean-prefix
gain nor a direct per-unit skew/variance correction, and changes no tensor
factor geometry. Potential inference cost is negligible, not yet metered.

The Gaussian leading-order birth coefficient is analytically correct;
this tests a finite-width effective correction to omitted distinct content,
not a claim that its leading coefficient was wrong. Global shared layer
coefficients are allowed estimator parameters; no per-evaluation-MLP fit.

## Bounded initial sensitivity gate

First certify that the signed cubes plus star have exactly the same
repeated entries for a_l=.9/1/1.1 on synthetic arrays. Then ranks35/100,
fixed current matched head, layers1..14 as14 scalar source parameters.
Generate central responses at coefficients1+-.02, all other coefficients1;
baseline must be byte-identical to saved current means. The two-net
linear-response bound fits ONE SHARED vector of coefficient offsets,
bounded to+-.10, using stored1e9 final means (research targets only).
This is an information/sensitivity screen, not a validated candidate.

Require >=20% predicted final-MSE improvement on BOTH nets using the shared
bounded coefficients. Then execute the actual nonlinear carrier once on
both with that fixed vector; require <=10% response-prediction error and
>=15% real raw improvement on both. No amplitude/window/basis sweep.

Only passing both would justify a separately preregistered24-net
20train/4held-out global-coefficient fit, followed by14-panel and exact
runtime/packaging gates. Failing closes this scalar-layer-amplitude family
before a larger fit. No learned row-wise amplitudes are implied or authorized
by a failed scalar fit; such a different action needs its own reasoning.

At most2 nice10 CPU workers with3BLAS threads; no MPS work required.
Preserve all baselines, Jacobians, predictions and receipts.

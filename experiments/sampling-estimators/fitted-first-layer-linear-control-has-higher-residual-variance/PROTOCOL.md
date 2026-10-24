# Strategic-reset diagnostic: antithetic first-layer control

Frozen prospectively on 2026-09-08 UTC, before this diagnostic is run.
Phase 2 only: width 1024, depth 16. Final-layer MSE is the scored raw
objective; returning all sixteen layer means does not make their average
the scoring objective. No existing work or negative receipt is superseded.

## Bank and two routes

Official bank #329926 remains adjusted 4.982231460183304e-9 (preserved
graded receipt), raw 2.2448352261505987e-8. No reset result is a new bank.
The user has authorized plausible experiments, with resource safeguards,
and milestone submissions below 4e-9, 3e-9 and 2e-9 toward strict <1e-9.
There is no authorization to post to the discussion forum.

1. SELECTED: an exactly centered first-layer control of an antithetic true
   network evaluation. Pairing eliminates all odd input components, so the
   previous unpaired shallow-control odd-chaos obstruction is not decisive.
   The first-layer absolute-Gaussian covariance is analytically available.
   Neither fact establishes sufficient even-component capture. The missing
   information is its actual held-out residual variance; a small independent
   path experiment can measure it without any mean-truth files. There is
   no defensible numerical gain forecast yet. The required magnitude is
   stated below rather than presented as a prediction.

   Complete-work cost inventory for a possible paid implementation: analytic
   feature covariance, random training paths, feature whitening/least-squares
   construction, independent correction paths, RNG, activations, control
   applications, reductions, all-layer return values, casts and setup. None
   may be charged to a free oracle. For planning only, reserve 300B FLOPs
   for construction and 70,429,900.8 FLOPs per independent antithetic pair
   (the exact classical dense leading subtotal 67,076,096 plus 5% allowance).
   A 1T total allowance then permits 9,938 correction pairs. The 300B and
   5% allowances are unverified estimates, NOT complete metered bills; a
   passing capacity test must next establish an actual lawful constructor,
   full flopscope bill and production runtime. A cheaper weights-only
   coefficient supplier would be a separate demonstrated improvement.

2. ALTERNATIVE: discard the present CP/head recurrence and use a Gaussian
   forecast plus a backward, frozen-gate terminal third-cumulant response.
   A single final endpoint has a genuine linear-depth suffix identity: 14
   suffix products and 15 common-base applications at full width. About 30
   more square products construct a basic Gaussian covariance forecast.
   This is about 126.6B leading classical FLOPs, before all source, pair-
   kernel, scalar-response, memory and assembly work; a 150--220B planning
   envelope is not yet a complete bill. At the 0.1 score floor it would
   need raw <3e-8 for the first-place target and <1e-8 for the final goal.
   The unresolved assumption is severe: omitting intermediate skew-induced
   mean/covariance feedback may dominate the final error. The actual D-only
   current-head test was bad (development adjusted 1.92375e-7), but is not
   this model. There is no demonstrated accuracy forecast for the new model.
   Cheapest decisive test would be one fully specified two-net forward
   comparison against Gaussian-only, with actual own complete cost, after
   its terminal source and omitted feedback are unambiguously defined.

Select route 1's short missing-information diagnostic now, not a long fit
or a declaration that it is a competitive candidate. Keep route 2 at the
model-definition stage. Preserve the already-trained birth-direction arm
as unjudged; do not silently discard it or resume its old release pipeline.

## Fixed statistical experiment

Canonical development ranks 35 and 100, not pristine holdouts. For each:
4096 fresh independent Gaussian TRAIN inputs and 4096 independent TEST
inputs. Fixed seeds recorded by the checker; no seed, ridge or rank search.
MPS batch 128, one job. No existing target means or fitted heads are read.

For row input x, define Y=(F(x)+F(-x))/2 and

    u_i = |(x W0)_i|/2 - ||W0[:,i]||/sqrt(2*pi).

Then E[u]=0 exactly in the ideal Gaussian law. Its analytic covariance is

    Sigma_ij = sigma_i sigma_j/(2*pi)
        * [sqrt(1-rho_ij^2) + rho_ij asin(rho_ij) - 1].

Fit ONE unregularized affine linear readout on TRAIN using analytic
whitening and QR. Freeze it before inspecting TEST. Independent estimation
of E[Y-A^T u] remains unbiased in exact arithmetic for every fixed A.
Reuse the first preactivation for both signs; fifteen subsequent matrices
are applied to each branch. Only one final control application is needed.
Intermediate means can be returned using the already computed samples;
they do not require additional fitted controls for the final-only score.

Report plain, antithetic and fixed-readout residual variance per output
averaged over the 1024 final units. Report 16-block empirical jackknife
uncertainty. This is an IID variance/cost proxy, NOT measured final MSE.
The fitted training intercept does not provide a control mean or a free
estimate of the true network mean.

Also report one optimistic population-class floor: split TEST into two
independent halves. For the frozen readout residual R, compute unbiased
within-half variances v_s and cross-covariances B_s=Cov(R,u). The statistic

    (v_1+v_2)/2 - trace(B_1 Sigma^{-1} B_2^T)/1024

is unbiased, conditional on TRAIN, for the population-optimal linear-u
residual variance. Its population target is invariant to the fitted A.
Use the fixed block scheme for explicitly empirical uncertainty; this is
not a certified finite-sample confidence bound. Negative, ill-conditioned
or numerically unresolved estimates do not count as a passing capacity
result. A poor finite-sample readout does not by itself close the ideal
feature class. No numerical rank tuning or validation refit is allowed.

## Prospective decision rule and limitations

For the stated 1T/300B/9938-pair planning scenario, use

    expected_adjusted_proxy = v_R/9938 * (1e12 / 2^41).

An empirical upper uncertainty endpoint below 6.556167934102733e-5 on
BOTH nets supports investigating an actual paid implementation for <3e-9;
below 2.1853893113675777e-5 supports the <1e-9 route. It does not pass
either milestone. A low optimistic class floor with a high fitted residual
only supports a coefficient-acquisition investigation, not a score claim.

As a more optimistic reference, zero construction and extra-operation cost
with classical dense independent pairs still requires residual variance
below approximately 9.8352e-5 for <3e-9 and 3.2784e-5 for <1e-9. If the
class-floor lower uncertainty endpoint exceeds the exact <3e-9 yardstick
3e-9*2^41/67,076,096 on both nets, this fixed first-layer linear-feature IID
route does not advance. This does not close richer controls, fast batched
circuits, RQMC, or different features. Inconclusive means inconclusive,
not permission to tune this diagnostic until it passes.

Any later candidate needs faithful finite-precision centering, actual raw
error and complete compute in the same configuration, the independent14
panel, leakage/provenance checks, flopscope-only packaging, and production
120s/0.4s/8GB checks. Native raw-NumPy input bills are not automatically
official FNP-input complete bills. No uploads follow from this proxy alone.

## Resources and records

One nice-10 numerical process, one MPS job, CPU BLAS1, no AP+threadpoolctl.
Target <=300s; abort at the frozen checker limit and retain partial output
if slower. Combined process RSS plus current MPS driver allocation <4GB
under the checker's conservative double-counting convention. Record both
separately as well. Check current swap before and after; pre-run swap is
776.75MiB on the 48GiB machine. Same-path CPU64 versus MPS32 checks on fixed
eight-path TRAIN and TEST subsets per net qualify numerical scale; this
does not certify a submitted implementation. Save source/weight hashes,
seeds, dimensions, conditioning, phase timings, memory and all diagnostics.
Use new output directories, preserve partial failures, no overwrites.

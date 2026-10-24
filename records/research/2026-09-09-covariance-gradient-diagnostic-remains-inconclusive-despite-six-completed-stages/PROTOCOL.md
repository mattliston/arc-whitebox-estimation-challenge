# Current-bank covariance-gradient gap: one diagnostic, not another fit

September 9, 2026 UTC / September 8 Pacific. Phase 2 only, 1024 by 16.
Official bank 330390 is confirmed at adjusted 4.53886349950395e-9,
raw 2.120838555441651e-8. Goal remains strict official adjusted <1e-9,
with validated milestone banks below 4e-9, 3e-9 and 2e-9. No forum posts,
commits, promotions, root-estimator changes or frozen-file overwrites.

## The two strongest remaining routes, before computation

1. **Independent covariance action with complete carrier gradients.**
   The old rank-eight congruence fit was genuinely evaluated dynamically,
   but its training froze K21/k4 and masks. Its four raw ratios were
   1.01223, .98541, .99907, .99337: no material improvement. The old
   shadow/full response certificate was already within about 6%, so it
   is NOT established that omitted carrier derivatives caused failure.
   Current d50 full-carrier training never independently parameterized C.
   Earlier covariance geometry puts 81--83% of observed late covariance
   error in blocks with at least one top-eight direction. That supports
   action capacity, NOT coefficient predictability or a current raw gain.

   Expected endpoint magnitude is UNKNOWN; no 12% or 34% forecast is
   supported. With roughly .9B extra local arithmetic for five late rank8
   writes, and unchanged other work, those are approximately the raw
   improvements needed for the official 4e-9 and 3e-9 milestones. The
   action plus parameter setup and changed selectors/CP work must all be
   metered in a future candidate; .9B is not a complete-card receipt.
   The current independent14 mean bill is 467682934475.3571 F. Thus a
   first planning estimate is about 468.6B, conditional on the same
   geometry; no product of aggregate raw and aggregate F is a score.

   Missing premise: do the omitted higher-state derivatives materially
   change an endpoint descent direction? The cheapest decisive test is
   the two-network gradient comparison below, not a new training run.

2. **Per-pair best-cube recompression.** Exact small algebra exists, but
   the later archived metadata bound narrowed the old 31% all-eligible
   estimate. With the old .1 per-pair tensor-error criterion, even free
   acquisition/restoration permits only about 11.18%/10.79% gross savings;
   an ideal rank-2N restoration reduces this to about 9.97%/9.62%.
   These are old P0/Q shapes, not a d50 bound. Actual first-jet eligibility
   was 4/10784 and 7/10752, and padding erased its saving. No actual
   best-cube endpoint score was measured. A different whole-pair dictionary
   compression did run and was 4.91 times worse adjusted. Current endpoint
   error and complete FNP-safe acquisition/cast-repair/packing cost are
   missing. No supported milestone-size expected gain remains. Do not
   force a current recapture merely from the superseded 31% estimate.

Select route 1's missing-gradient diagnostic ONLY. A diagnostic pass is
not approval for a fit, a raw improvement, a complete score or a bank.

## Exact current action and two derivative models

Parent is the d50 terminal birth-direction card, frozen P and frozen
deployment birth coefficients. All 16 P hooks remain live; freezing their
parameters does not detach endogenous hidden-state inputs. Use layers
10 through 14, after existing mean recentering and active masking but
before Bpost/trace_post/star construction. Retain original radial timing.

At each write, center the current returned P hidden state H by its current
column means, divide by sqrt(mean(centered_H^2)+1e-6), clip to [-8,8],
and append a constant-one column, giving H65. Fixed V0 is a 65 by 8
PCG64DXSM(seed9309095) standard-normal matrix divided by sqrt(65), stored
as FP32. Five A matrices, each 65 by 8, start at exactly zero. There is no
optimizer or parameter update in this test.

    U = .02*sigma*tanh(H65 @ A_layer)*alive/sqrt(n)
    V = (H65 @ V0)*alive/(sigma*sqrt(n))
    R = C @ V; S = V.T @ C
    C_new = C + U @ S + R @ U.T + (U @ (V.T @ R)) @ U.T

The literal stored-C formula does not assume exact symmetry. It is a
congruence in real arithmetic; PSD is preserved only if incoming C is
PSD. No unconditional symmetrization or data-dependent null bypass.
Vectors are formed in FP32; action products use the card's moment dtype.

FULL retains all physical derivatives. READ_CUT detaches k21 and incoming
radial immediately before nonlinear(), with those same detached locals
used by current P features and subsequent birth/trace calculations.
The current means, Cpre and P hidden state remain differentiable. This
is a NEW precisely defined frozen-preK3/radial surrogate, not a claim to
reproduce the old frozen-k4 shadow: k4 still depends on current variance.
Both modes must have identical forward values and physical selections.

## Qualification, data boundary and decision

Before training labels are opened, qualify one fixed n7 action fixture,
including a deliberately asymmetric C and both zero/nonzero A. Compare
the factored expression to explicit (I+UV.T)C(I+VU.T), NumPy64 and Torch64;
relative error <=1e-11. Check Torch64 directional derivatives against
central differences at 1e-5 and half-step, relative <=1e-6. Use fixed
RNG fixtures, no step or seed search. Qualify FP32 MPS action outputs
and non-negligible gradients against this CPU reference to <=1e-4.

Only reused TRAIN ranks 35 and 100 may be decoded; no VALID truth reads,
no independent14 targets, no new sampling. Authenticate canonical weight
manifests and existing 1e9 training-mean target hashes before decoding.
No targets enter a model input, action, feature, selection or normalizer.
Seal target-free zero-action forward results and baseline parent parity
before opening the two final-mean labels. All16 same-backend served FP32
outputs must match the current parent byte-for-byte at zero A; FULL and
READ_CUT must also agree. CPU64 versus MPS32 maximum output difference
must be <=1e-5. Check positive finite prevariances and checkpoint-selection
replay. Negative-zero differences are reported as failures, not waived.

For each network, compute the A gradient of served final MSE / 2e-8 at
zero A in both modes. Record raw error and gradient norms, relative
difference, cosine, and the FULL directional derivative along READ_CUT's
negative-gradient direction. This is final-error information on training
networks, NOT a fitted endpoint result or adjusted score. There is no
cost derivative through the discrete selections and no claim of one.

Before interpreting a gap, independently qualify CPU64/MPS32 nonzero
gradient vectors: <=5% relative discrepancy in each mode, unless the
reference norm is below 1e-10, in which case the test is inconclusive.
Tiny action finite differences are numerical qualification; optional
additional full-card finite differences require a separately frozen
protocol, not an adaptive response to unfavorable gradient data.

Information gate, fixed now: on BOTH networks,
norm(g_full-g_cut)/norm(g_full) >= .30 and cosine(g_full,g_cut) <= .90.
If this fails, the missing-gradient argument does not justify a renewed
congruence fit. A pass earns a separate prospective final-score fit design,
not automatic training. A missing or failed numerical qualification earns
no scientific pass. Preserve results and all adverse receipts unchanged.

## Resources and scope

One nice10 numerical process at a time, requested CPU threads <=3, BLAS1;
one MPS job, fallback disabled. Never import threadpoolctl into AP. Max
process RSS 12GB and MPS driver memory 16GB, monitored at <=.5s intervals;
900s per wide backend process, 120s tiny qualification. Release each
network/graph before the next. Check swap before and after each stage.
No native participant/runtime success is implied by offline limits.
No fitted payload, candidate build, quota use or external write follows.

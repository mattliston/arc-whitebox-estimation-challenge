# Q24 certified connected-state shutdown probe v1

Status before execution: `PREREGISTERED_TARGET_FREE_CONSTRUCTION_PROBE_ONLY`.

This probe is additive and binds the frozen manual-rho Q24 parent estimator SHA-256
`7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00`.
It opens no dataset rows, targets, truth values, public-row weights, scores, or
submission state and performs no width-1024 run.

## Fixed intervention

At each recursive ReLU reset, compute the Stieltjes margins

- `A = r1*r3 - r2*r2`, and
- `B = (r4-r2*r2)*(r2-r1*r1) - (r3-r1*r2)^2`

on the stored fp64 marginal tuple.  Certify the evaluation sign with the frozen
outward-rounded error DAG: local roundoff is bounded by `2^-52*abs(result)` and
every nonnegative error addition or multiplication is rounded upward with
`nextafter(...,+inf)`.  A unit is invalid iff `A < -E_A` or `B < -E_B`.

Do not alter that layer's reported mean, variance, covariance, or raw marginal
tuple.  Instead shut down the complete outgoing high-order connected state for
the invalid coordinate:

- set `K4[i]=0`;
- set every row or column incident to `i` in `K21`, `K31`, and `K22` to zero;
- multiply the CP3 and CP4 post-ReLU gate by the valid-unit mask; and
- multiply the CP3 birth coefficient by the same mask.

This makes the invalid coordinate Gaussian at connected orders three and four
for future propagation while retaining its lower-order state.  There is no
epsilon, magnitude threshold, layer threshold, target feedback, retry, or
fixture-specific branch.

## Frozen fixture roster

Use the exact ten target-free fixtures frozen by
`q24_two_point_moment_repair_probe_v1/run_probe.py` SHA-256
`2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d`:
the four canonical width-8 depth-16 fixtures plus official-He widths 32, 64,
and 128 for seeds 20260825 and 20260826.

## Frozen construction gates

The mechanism passes this bounded probe only if all conditions hold:

1. at least one certified-invalid unit activates the shutdown;
2. no fixture finite under the parent becomes nonfinite or raises under shutdown;
3. every returned invalid-incident connected target is represented exactly zero;
4. for every pair where both outputs are finite, `RMS(child-parent) <= RMS(parent)`
   and `maxabs(child-parent) <= maxabs(parent)`; if the corresponding parent scale
   is zero, the disagreement must be exactly zero.

Parent failure recovery is supportive only.  A pass authorizes source construction
and independent audit, not a full-shape run, accuracy claim, package, submission,
or graph promotion.  A failure closes this exact intervention; its predicates may
not be tuned against the result.

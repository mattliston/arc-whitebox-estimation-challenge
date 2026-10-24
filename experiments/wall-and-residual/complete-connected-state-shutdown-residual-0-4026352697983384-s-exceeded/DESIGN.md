# Tail-stability successor decision

## Decision

Select the carrier-complete, certificate-triggered connected-state shutdown
defined in `PREREGISTER.md` as the next smallest scientifically coherent arm.
It preserves the current mean and lower-order state, removes the invalid
coordinate's outgoing order-three/four state, and globally shuts the one
pairing channel that the representation cannot mask coordinatewise.

This is not a claim that the raw marginal tuple was repaired.  It is a
parameter-free stability intervention on future connected state.

## Why the other mechanisms are not selected

### First-two-moment-preserving Stieltjes completion

For a nonnegative law with `m=r1>0`, `q=r2`, and `v=q-m^2>0`, feasibility
requires

```
r3 >= q^2/m
r4 >= q^2 + (r3-m*q)^2/v >= q^3/m^2.
```

The reciprocal growth is intrinsic as `m` approaches zero with floor-dominated
`q`; an epsilon does not remove it without changing the physical moments.
The direct two-point boundary completion is the coordinatewise-minimal form,
yet its frozen target-free probe activated 4,741 times, introduced four
parent-finite-to-nonfinite regressions, left 191 negative represented `A`
values and 337 negative represented `B` values, and recovered no failures.

The proposed lexicographic upward completion has the same division problem.
It is additionally ill-conditioned through `1/v` and can inject a large
fourth moment into a near-dead coordinate.  A gamma completion moves into the
strict interior but asymptotically doubles the minimal third moment and
multiplies the minimal fourth moment by six.  None is the smallest bounded
tail-stability arm.

`_VAR_FLOOR=1e-30` remains only the parent's downstream conditioning device.
It is not a lawful denominator threshold or physical Stieltjes completion:
`r1=0,r2>0` cannot be the first two moments of a nonnegative random variable.

### Marginal Gaussian fallback

The preactivation Gaussian positive-part tuple is a coherent univariate law
and is scale-aware without division.  Replacing only `raw1..raw4`, however,
leaves pair Edgeworth moments and compact connected carriers derived from a
different state.  The measured negative-moment-triggered spelling confirmed
that this inconsistency is not benign: 1,609 activations, three new
parent-finite regressions, one negative post-fallback fourth moment, and a
`1.7171762496955415e15` final disagreement MSE on one otherwise-finite HE32
fixture.  Its two recoveries do not pass its fatal gates.

A preactivation incident-cumulant quarantine could recompute marginal and pair
moments from a single masked input state and mask pushed factors before reset.
That spelling is more invasive: it changes the current mean/lower moments,
duplicates the univariate calculation, touches dense pair planes and rank
carriers, and has no construction-pass evidence.  A conservative all-active
projection was `+1,721,355,972` FLOPs, or `96.9977073674%` of budget on the
manual-rho parent.  It remains a fallback design, not the highest-value next
arm.

### Output-side connected shutdown

The frozen construction probe is the first arm with positive recursive
evidence: 2,647 activations, zero parent-finite regressions, three recoveries,
exact-zero masked target arrays, and all parent-scale guards passed.  It also
preserves the current reported row, so a detector activation cannot directly
create the catastrophic current-layer mean replacement seen in the marginal
fallback.

The first source spelling is nevertheless incomplete.  Its target arrays are
zeroed before `_reset_stars`, but `_reset_stars` then fits one global
`pairing_alpha`.  In the current coordinate basis, the coordinate-star carrier
has no off-diagonal `K22` capacity.  Therefore an invalid-incident pair still
receives

```
pairing_alpha * (variance_i*variance_j + 2*covariance_ij^2)
```

unless the scalar alpha is zero.  The quartic star also carries nonzero pieces
that cancel the pairing term in `K4/K31` at the reset; dense propagation can
remix the underlying tensor even when a checked target entry was zero.  The
source hold receipt correctly rejects that spelling.

The representation has no coordinate-masked pairing carrier.  Forcing
`pairing_alpha=0` whenever any coordinate is certified invalid is consequently
the smallest exact carrier-complete change.  It is broader than the coordinate
mask because it also removes valid-valid residual `K22` information in that
one scalar channel, so it must earn a fresh probe rather than inherit the
first arm's pass.

## Exact source seams

All line references below are to manual-rho parent SHA-256
`7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00`.

1. `_univariate_edgeworth` at lines 395-446 remains numerically unchanged.
   Its returned stored tuple feeds the exact outward certificate.
2. `_full_relu_reset` at lines 1266-1339 computes one mask after the
   univariate tuple, keeps raw/lower-order outputs, masks connected targets,
   and returns the mask explicitly.
3. `_predict` at lines 1490-1519 receives the mask.  Lines 1522-1539 mask the
   Gaussian gate and birth before `_cp_post_read` and `_cp4_reset`; lines
   1553-1564 pass the masked gate/birth into `_cp_grow`.
4. `_reset_stars` at parent lines 1093 onward gains an optional
   `valid_high_order` argument.  The all-valid test selects the nominal alpha;
   any invalid coordinate selects represented zero before quartic residual
   construction.  The H1 call passes `None`; recursive calls pass the mask.
5. The manual left-associated `_q24_context` recurrence at lines 449-482 is
   unchanged.
6. The final output call at lines 1441-1451 is unchanged.  There is no outgoing
   carrier after the final reported row.

The existing masked gate is sufficient for CP factors.  In CP3, it zeros
inherited `p/q` rows and invalid-coordinate birth amplitudes; in CP4, it zeros
inherited `u` rows and makes the invalid diagonal birth coefficient zero
because both target and inherited diagonal are zero.  Raw identity-factor
columns with zero coefficient need not themselves be zero.  Carrier authority
must reconstruct the weighted state.

## Cost projection

The rejected first source's small-width exact fit is
`350*w^2 + 4620*w` FLOPs and 2,268 calls over the manual-rho parent.  The
separately frozen CS2 source implements the all-valid predicate with fourteen
boolean `all` reductions and fourteen scalar fp64 `where` operations, giving
the measured small-width law

```
350*w^2 + 4634*w + 98 FLOPs
2,296 calls
```

at depth 16.  At width 1024:

| Quantity | Projection |
|---|---:|
| Increment | 371,746,914 FLOPs |
| Total | 2,131,652,533,303 FLOPs |
| Budget used | 96.9363342530% |
| Headroom | 67,370,722,249 FLOPs |
| Calls | 9,504 |

The original count-based preregistration estimate was 14 FLOPs higher; the
post-source reconciliation preserves that preregistration and binds the exact
CS2 measurements.  The independent instrumentation-inclusive upper envelope is larger but also
admissible: total `2,132,108,300,245`, 96.9570601340%, with
66,914,955,307 FLOPs headroom.  Neither number is a literal successor meter.
Runtime, memory, and residual remain open; the parent's local residual margin
was only 22.9 ms.

## Evidence boundary

The 9/13 certified negatives prove only that those stored marginal tuples are
infeasible under the determinant-evaluation certificate.  They do not prove
that infeasibility causes the board tail.  Likewise, the public five-row
concentration motivates a tail-stability experiment but identifies no causal
row, layer, unit, or state channel.  No public identity or score participates
in the intervention.

The next result can establish construction and carrier coherence only.  A
separate target-free full-shape resource gate and a later authorized accuracy
gate are still required before any promotion.

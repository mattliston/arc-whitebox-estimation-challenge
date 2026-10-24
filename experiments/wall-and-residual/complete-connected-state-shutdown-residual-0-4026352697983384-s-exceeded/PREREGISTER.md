# Q24 carrier-complete certified connected shutdown v1

Status: `PREREGISTERED_DESIGN_ONLY__NOT_EXECUTED`.

This namespace freezes the smallest source successor to the rejected first
connected-shutdown spelling.  It authorizes no dataset or target access, no
public-row use, no width-1024 execution, no package, no submission, and no
canonical graph edit.

## Question

When a represented post-ReLU raw tuple has an outward-certified negative
Stieltjes determinant, can the closure preserve its current mean and
lower-order state while removing its complete outgoing order-three/four
carrier, without a new recursive instability?

The uncertainty is narrower than the already-passed construction probe.  The
first source spelling zeros connected target arrays and masks CP gates, but a
global nonzero `pairing_alpha` can reconstruct invalid-incident fourth-order
state after that checked seam.  This preregistration changes that one carrier
decision and strengthens the invariant check from target arrays to the stored
carrier itself.

## Frozen parents and negative controls

- Manual-rho Q24 parent estimator: SHA-256
  `7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00`.
- Certified stored-tuple diagnostic receipt: SHA-256
  `4a7929243f8813fc56be4b2fb9cf97970bd1766678d0d3dad86a856dc6578dcb`.
  Its evidence is limited to stored fp64 tuples and contains 9 certified
  negative `A` observations and 13 certified negative `B` observations,
  all late and near-dead.
- Connected-shutdown construction result and receipt: SHA-256
  `5fe96611212de6beb5d785198d3f4635b46ea42169efa7257ebcf69e0e305bc7`
  and
  `1c3c18670d9f06554b1488499c45e4b85a2f8d3e0c0efd263c0b08b2c84d9b6b`.
  The fixed ten-fixture probe recorded 2,647 activations, zero
  parent-finite regressions, three recoveries, and exact zero returned target
  entries.
- Rejected first connected-shutdown source: SHA-256
  `2532e58b76fd839d699b089c11e729d958f33edc1499816e9c75ac590b6aef2c`;
  hold receipt SHA-256
  `4ee1d35434ec0dabb85f1846856f04c4dfc0cc46292aad82ceb4cc72c72dac96`.
  It is a mandatory negative control and must not be frozen or metered as a
  candidate.
- Stieltjes red-team receipt: SHA-256
  `79c4087923e4d18b41420203c3a1bd93e95af3ab5634058ea97357b4ef19aee0`.
- Two-point and marginal-only Gaussian fallback rejection receipts:
  SHA-256
  `5b9af5b607b9102d2140bc92a643e9ab626c35e0ff7bead530d6e8abb247a519`
  and
  `855819aeda1021c536136290402c45fd154c6d46c754505a34ef4e41565d6d0b`.

The public board result is post-hoc motivation only.  Its rounded public
scores concentrate 86.7525471% of displayed raw error in five of fifty rows,
but neither those identities nor any public score, threshold, weight, or
rank may enter this source, fixture roster, or gate.

## Exact certificate

Use the exact `_TWO_U`, `_up`, `_add_sub_error`, `_multiply_error`, and
`_certified_invalid_marginals` arithmetic already present in the rejected
first source.  On stored `(r1,r2,r3,r4)`, define

```
A = r1*r3 - r2*r2
B = (r4-r2*r2)*(r2-r1*r1) - (r3-r1*r2)^2
```

and mark a coordinate invalid exactly when `A < -E_A` or `B < -E_B` under
that outward error DAG.  `ZERO_COMPATIBLE` is not invalid.  There is no
epsilon or mean, variance, scale, coefficient-of-variation, layer, row,
identity, score, or public-tail threshold.

The certificate treats the stored raw inputs as exact and bounds only the
displayed determinant evaluation.  It does not certify upstream moment
generation, a marginal probability law, or a joint law.

## Frozen intervention

Retain the first source's output-side behavior exactly:

1. Do not change the current reset's `raw1..raw4`, mean, variance, repaired
   covariance, or reported row.
2. Set invalid `K4` entries and every invalid-incident entry of `K21`, `K31`,
   and `K22` to represented zero.
3. Mask the post-ReLU Gaussian gate and CP3 birth coefficient by the same
   valid-coordinate mask.  This suppresses inherited CP3/CP4 factors and
   their invalid-coordinate births through the existing `_cp_post_read`,
   `_cp4_reset`, and `_cp_grow` seams.
4. Rebuild cubic and quartic residual state from the masked targets and
   masked CP providers.

Add one and only one rule inside `_reset_stars`: for recursive resets with a
valid-coordinate mask, set the scalar covariance-pairing carrier to zero when
any coordinate is invalid, before `three_alpha`, `quartic_diagonal`, and the
quartic residual are formed.

Source-ready spelling:

```python
nominal_pairing_alpha = numerator / denominator
if valid_high_order is None:
    pairing_alpha = nominal_pairing_alpha
else:
    all_valid = (
        fnp.count_nonzero(valid_high_order)
        == int(valid_high_order.shape[0])
    )
    pairing_alpha = fnp.where(
        all_valid,
        nominal_pairing_alpha,
        0.0,
    )
```

`_initialize_h1` continues to call `_reset_stars` with
`valid_high_order=None`; recursive calls pass their explicit mask.  No final
layer action is added because there is no outgoing connected state after the
reported final mean.

This is deliberately global for the single `pairing_alpha` channel.  The
current representation has no coordinate-masked pairing carrier, so keeping a
nonzero scalar alpha would retain invalid-incident `K22`, while attempting to
cancel it with the coordinate star would retain a nonzero fourth-order tensor
that dense propagation can remix.  The tradeoff is explicit: any activation
also discards valid-valid residual `K22` information formerly represented by
that scalar pairing channel for that reset.

## Frozen fixture roster

Use exactly the ten target-free depth-16 fixtures from
`q24_two_point_moment_repair_probe_v1/run_probe.py`, SHA-256
`2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d`:
four canonical width-8 fixtures and the six official-He fixtures at widths
32, 64, and 128 with seeds 20260825 and 20260826.  Do not add, remove, or
select fixtures after viewing the result.

Run three source arms: unchanged manual-rho parent, rejected first source as a
negative control, and the pairing-complete successor.  All runs are
target-free and use a generous local probe budget; width 1024 remains
forbidden.

## Carrier reconstruction authority

At every activated reset, record and verify all of the following after the
compact state has been rebuilt:

1. invalid-incident returned `K21/K31/K22` and invalid `K4` targets are exact
   represented zeros;
2. the scalar `pairing_alpha` is exactly zero;
3. the effective CP3 carrier plus cubic residual reconstructs exact-zero
   invalid-incident `K21` in the current coordinate basis;
4. the effective CP4 carrier plus quartic star plus pairing component
   reconstructs exact-zero invalid `K4` and invalid-incident `K31/K22` in the
   current coordinate basis;
5. an identity push and a fixed signed-permutation push preserve the mapped
   exact-zero incident subspace; and
6. all reconstructed fields and pushed fields are finite.

Zero-coefficient identity birth columns may contain nonzero factor entries;
the invariant is on the effective weighted carrier, not on an unweighted
factor row.  The first source must be evaluated by this same reconstruction
authority; checking only its masked target arrays is insufficient.

## Construction gates

The successor passes only if every condition below holds:

1. at least one certified-invalid unit activates;
2. no parent-finite fixture becomes nonfinite or raises;
3. every carrier reconstruction and one-push invariant above passes exactly;
4. every no-activation fixture is byte-identical to the parent;
5. for every finite parent/successor pair,
   `RMS(successor-parent) <= RMS(parent)` and
   `maxabs(successor-parent) <= maxabs(parent)`; a zero parent scale requires
   exact-zero disagreement;
6. outputs are deterministic and byte-identical over two repeats; and
7. a tiny-width FlopScope delta law is exact at widths 4, 8, 16, and 32 and
   projects strictly below `2^41` at width 1024.

Recovery of a parent failure is supporting evidence, not required.  The
raw tuple may remain Stieltjes-infeasible; that is not a gate failure because
this arm is a connected-state shutdown, not a moment repair.

## Cost and runtime gates

The source-ready cost hypothesis relative to the measured manual-rho parent
is

```
incomplete-source delta: 350*w^2 + 4620*w, +2268 calls
pairing completion:       14*w + 112,       +28 calls
combined hypothesis:      350*w^2 + 4634*w + 112, +2296 calls
```

At `w=1024`, this is a projected increment of `371,746,928` FLOPs,
projected total `2,131,652,533,317`, utilization `96.9363342536%`, and
headroom `67,370,722,235`.  This is a projection, not a literal meter result.
The instrumentation-inclusive red-team envelope, `827,513,856` added FLOPs,
also projects below budget at `96.9570601340%` with `66,914,955,307`
headroom.

No runtime pass is claimed.  A later separately authorized full-shape gate
must literal-meter FLOPs and record exact operation calls, wall time,
FlopScope residual, backend time, and descendant RSS.  Its terminal limits
remain `FLOPs <= 2^41`, prediction wall `<=120 s`, residual `<=0.4 s`,
descendant RSS `<8 GiB`, finite output, and zero failures.  The measured
manual-rho residual had only 22.9 ms of local headroom, so the runtime gate is
independently fatal even though the FLOP projection has ample margin.

## Outcome split

- If any construction or carrier gate fails, close this pairing-complete
  spelling.  Do not tune the trigger or add a threshold.
- If the first source unexpectedly passes the stronger reconstructed-carrier
  gate and the successor is byte-identical, classify the added rule as
  redundant and do not promote it on theory alone.
- If all bounded gates pass, freeze a new source and request separate
  independent static and full-shape resource authority.  Do not open targets
  or scores.
- Accuracy evaluation, packaging, and submission remain separate held gates
  even after resource success.

## Access discipline

Forbidden in this namespace: dataset rows, targets, teachers, truth, public
row identities or weights, score arithmetic as a decision input, sampling,
fitting, width-1024 execution, package construction, submission, or canonical
graph mutation.

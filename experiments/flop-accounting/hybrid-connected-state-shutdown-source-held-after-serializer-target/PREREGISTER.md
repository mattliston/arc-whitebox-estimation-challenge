# CS3 localized masked pairing-covariance carrier: design and source-probe preregistration

Status at seal: `PREREGISTERED_TARGET_FREE_DESIGN_ONLY__NO_CANDIDATE_SOURCE_OR_EXECUTION`.

## Question

Can CS2's globally disabled covariance-pairing channel be replaced, only on a
certified-invalid reset, by a masked covariance Wick carrier that preserves
valid-valid pairing information, keeps every invalid-incident current-layer
quartic component exactly zero, closes under one linear push, preserves the
no-activation path byte-for-byte, and remains within the Phase-2 budget even
if all 14 recursive resets activate?

## Parent evidence

The exact parent is CS2 estimator SHA-256
`4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779`,
source HOLD SHA-256
`c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c`,
and independent strict audit SHA-256
`cba8bfbb437f32aa377ca72da74b9504894e6380429e2acf0e5a059d7ef8afcb`.
Its fixed target-free width-1024 comparator receipt SHA-256
`151de92da658f15de016b256c74cb00255bd1d4aea8c4d8ef3934bab10665645`
passed source stability with 51 invalid unit-reset activations, zero
parent-finite regressions, and zero connected-carrier violations. The
separately preregistered board change-size gate failed on all three seeds;
rejection receipt SHA-256
`e1c9003a836684b7b064fa15a662b46b6ef0dff6e96a82d24749c60bc901b323`.

The open uncertainty is whether CS2's global `pairing_alpha=0` discards too
much valid-valid quartic information. No target, board row, score, or public
tail value may choose or tune this arm.

## Frozen mathematical carrier

Let `C` be the repaired post-ReLU covariance at a recursive reset, let
`m_i = 1` exactly when unit `i` passes the existing outward-rounded A/B
certificate, and let `M=diag(m)`. On an affected reset define the stored
pairing covariance as the represented fp32 matrix

```text
S32 = fp32(M C M)
S   = fp64(S32)
```

where masking is performed by exact Boolean selection, so every invalid row,
column, and diagonal is represented zero. The fp32 state precision is frozen:
it is the only spelling whose all-14-reset worst-case projected bill fits the
current headroom, and it matches the existing cubic/quartic compact-carrier
precision. The fitted fourth-order carrier is

```text
T_ijkl = alpha * (S_ij*S_kl + S_ik*S_jl + S_il*S_jk).
```

Its observable planes are

```text
K4_i    = 3*alpha*S_ii^2
K31_ij  = 3*alpha*S_ii*S_ij
K22_ij  = alpha*(S_ii*S_jj + 2*S_ij^2).
```

Thus any component with an invalid current-layer endpoint is exactly zero.
After a linear map `W`, carry this state exactly one layer as
`S32_next = (W32.T @ S32) @ W32`; promote that stored result to fp64 and use
`Wick(S_next)` in the existing quartic push. Algebraically,
`W^(tensor 4) T = alpha*Wick(W.T S W)`, so `(alpha,S)` is a closed one-layer
carrier. Dense `W` may spread the valid source subspace; after the push the old
mask is consumed, and the next ReLU reset applies its own new certificate.

Fit `alpha` by the parent's same off-diagonal least-squares projection after
the CP4 provider is subtracted, but replace the global covariance pairing plane
by

```text
P = offdiag(outer(diag(S),diag(S)) + 2*S*S)
alpha = <sym(K22),P> / <P,P>.
```

The connected targets are already zero on invalid incidents, so the full-plane
inner product is exactly the valid-valid fit. If `<P,P>` is represented exact
zero, set `alpha` to represented `+0.0`: the basis tensor itself is zero, so
this is the unique minimum-norm coefficient on an exact null face, not an
epsilon or fitted threshold. No nonzero floor is allowed.

Subtract the matching `3*alpha*diag(S)^2` and
`3*alpha*diag(S)[:,None]*S` from K4/K31 before storing the quartic star. CP4,
cubic, and target masks remain exactly CS2.

## Frozen no-activation path

No-activation behavior is a fatal gate. If every unit is valid, do not cast,
mask, store, or separately push `S`. Carry a Python `None` sentinel meaning
"use the lower covariance carrier". At the next push select the already
computed fp64 `pre_covariance` and `pre_variance` objects directly before any
pairing arithmetic. Do not recompute `W.T @ C @ W`, do not use an eagerly
evaluated array `where`, and do not round-trip through fp32. Remove CS2's
scalar `where(True, alpha, 0)`; the unchanged positive-denominator division
must feed the parent pairing operations directly. Identity and near-identity
outputs must be byte-identical to the manual-rho parent and CS2.

## Exact source seams for a future candidate

No source is authorized by this file. A later source arm may change only:

1. `_shutdown_connected_state`: return its already-computed `valid_plane` in
   addition to the four masked targets; do not recompute the plane.
2. `_reset_stars`: accept `valid_plane`/all-valid state, select either the
   original covariance or the frozen `S32/S` construction, fit the pairing
   coefficient on that covariance, handle only the exact-zero basis boundary,
   and return the `S32` carrier or `None`.
3. `_predict`: carry `S32` for one layer; only when it is not `None`, compute
   `(weight32.T @ S32) @ weight32`, promote once to fp64, and take its diagonal.
   Otherwise alias the existing `pre_covariance`/`pre_variance` exactly.
4. `_push_quartic` and `_push_quartic_diagonal`: consume the selected pairing
   covariance and variance for the Wick term; star arithmetic is unchanged.
5. State returns/calls and target-free observer hooks needed to reconstruct
   the current carrier and one-push identity.

H1 is unchanged and uses the original lower covariance pairing. Q24, moment
generation, covariance repair, CP3, CP4, stars, production guard, and final
reported mean remain otherwise unchanged.

## Frozen cost hypothesis

Let `w` be width and `a` the number of affected recursive resets whose `S32`
is pushed once (`0 <= a <= 14`). Relative to frozen CS2, the exact proposed
operation spelling has the preregistered weighted-FLOP delta

```text
Delta_F(a,w) = a*(4*w^3 + 8*w^2) - 112.
```

Per affected reset this comprises three fp64/fp32 matrix conversions, one
fp32 matrix `where`, and two fp32 dense matrix multiplies. The `-112` removes
CS2's one fp64 scalar `where` at each of 14 recursive resets. Diagonal views
and Python sentinel selection bill zero FLOPs. The tentative call delta is
`8*a - 14`, plus one zero-FLOP scalar constructor only on an exact null-basis
face. A future literal source meter is authoritative; this hypothesis may not
authorize full-width execution or promotion.

At `w=1024`, the all-14 worst-case delta is `60,246,982,544` FLOPs. Added to
CS2's current small-width projection `2,131,652,533,303`, it gives
`2,191,899,515,847`, or `99.6760498241%` of `2^41`, with projected headroom
`7,123,739,705`. The fp64-state alternative costs
`a*(8*w^3+4*w^2)-112`; at all 14 resets it exceeds the current headroom and is
not selected.

## Smallest future target-free probe

After a separate source preregistration and immutable source hash, run exactly
the existing ten CS2 source-probe fixtures—no fixture additions, removals,
seed search, or threshold changes. This is the smallest existing roster that
contains two no-activation byte-parity controls, activating algebraic fixtures,
and finite/nonfinite regression controls through width 128.

The source probe passes only if all of the following hold:

1. source diff, legal imports, production API, guard, and frozen unrelated AST
   regions pass;
2. at least one localized carrier activates and no parent-finite fixture
   becomes nonfinite or raises;
3. every no-activation fixture is byte-identical to both parent and CS2;
4. immediately after each affected reset, `S32`, Wick(S), pairing contribution,
   target, CP3, CP4, cubic-star, quartic-star, and aggregate connected state
   have exact zero on every invalid incident required by their representation;
5. the exact-zero pairing-basis boundary, if exercised by a constructed tiny
   algebra fixture, returns finite represented `+0.0` without epsilon;
6. on every affected reset the valid-valid K22 least-squares residual norm is
   no larger than the alpha-zero CS2 residual, with an independent direct
   projection replay;
7. identity and fixed signed-permutation one-push checks preserve mapped
   invalid zeros exactly; a width-at-most-6 dense one-push check agrees with an
   independently materialized four-index Wick tensor to a preregistered
   floating error bound derived before execution, not an observed tolerance;
8. all finite-pair output disagreements remain within parent RMS and maxabs
   scale, with exact zero required when parent scale is zero;
9. two repetitions are byte-deterministic; and
10. exact small-width source deltas fit `Delta_F(a,w)` for multiple controlled
    activation counts, while the all-14 width-1024 projection remains below
    `2^41`.

Failure closes this exact fp32 localized-carrier spelling. It does not permit
switching to fp64, changing the certificate, relaxing invariants, selecting a
different fixture, or opening full-width, meter, target, package, or board
authority.

## Access and authorization boundary

- dataset rows, targets, teachers, public score rows, and private state: zero;
- candidate source edits: zero;
- full-width runs and literal meter runs: zero;
- packages, submissions, network, and credential access: zero;
- canonical graph edits: zero;
- currently authorized work: read-only mathematics/source audit and immutable
  design packaging only.

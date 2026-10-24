# CS3B resource-law hybrid localized pairing carrier: design and source-build preregistration

Status at seal:
`PREREGISTERED_TARGET_FREE_DESIGN__SOURCE_BUILD_BLOCKED_PENDING_COHERENT_RECEIPT_AND_INDEPENDENT_AUDIT`.

Sealed at `2026-08-24T13:58:15Z`, before any CS3B candidate source, source
probe, estimator run, full-width run, resource meter, target or board access,
package, submission, or graph edit.

## Question and parent boundary

Can the standalone dense-fp32 CS3 pairing-covariance push be replaced by an
exact fp64 low-rank invalid-coordinate correction whenever its operation bill
is no larger than the frozen dense block, while retaining that dense block as
a worst-case fallback, preserving the nominal no-activation path, and keeping
the all-14-reset resource envelope below `2^41`?

The exact source parent is frozen CS2 estimator SHA-256
`4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779`,
source HOLD SHA-256
`c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c`,
and independent strict audit SHA-256
`cba8bfbb437f32aa377ca72da74b9504894e6380429e2acf0e5a059d7ef8afcb`.
Its literal width-1024 resource audit SHA-256
`a136f4031c14b354a00d983e249a406b5c03642a202042032e9f426034c836cc`
confirmed exactly `2,131,652,533,303` FLOPs and `9,504` calls, but terminally
failed the residual-time gate by `0.002635269798338391` seconds.

The standalone dense CS3 artifacts are main preregistration SHA-256
`3ddf814fa95737303fa0184d963f9ff734e6ff250d76ef5ab9dbd1db8197275e`,
corrective addendum SHA-256
`4fbe0d3c227332504433cd648cb28978fdac27edeea18971d06d9902e9540306`,
design receipt SHA-256
`af9c919e900abb5cae1965c8c2e27cb564ade9d19886514e30b6262d091f8be3`,
author HOLD SHA-256
`9ab43dd3e385f0bf043e23668ef0eb5563f32f4c5e913fcdd4e6a5e64cb5ca70`,
and independent HOLD SHA-256
`060cfbab98c85f7f9d46d49edd29726a985ae50fbc2c85b47660a35704a178db`.
No standalone dense-CS3 source is authorized.  Its fp32 path survives only as
the fallback frozen below.

The comparator's observed invalid counts may motivate examining a low-rank
representation, but they do not set the branch boundary.  The selector below
is derived only from the pinned operation-cost laws and must be evaluated from
the current width and certificate-mask cardinality.  No fixture outcome,
target, score, row, seed search, wall observation, or fitted threshold may
alter it.

## Frozen three-state carrier

At each recursive reset compute the existing outward-rounded certificate and
retain its existing `valid_high_order` vector.  Reuse exactly
`fnp.all(valid_high_order)` and convert that already billed zero-dimensional
result to a native Boolean.  Never reduce the `w` by `w` `valid_plane`.

The outgoing pairing state has exactly three native modes:

1. `NOMINAL`: every unit is valid.  Carry no matrix payload and no indices.
2. `LOWRANK64`: at least one unit is invalid and the exact low-rank bill below
   is no larger than the frozen dense block.  Carry the invalid integer index
   vector.
3. `DENSE32`: at least one unit is invalid and the low-rank bill is larger.
   Carry the represented dense `S32=fp32(M C M)` matrix.

The mode is an explicit native enum/integer used only for residual control
flow.  The payload invariant is exact: no payload in `NOMINAL`, indices only
in `LOWRANK64`, and `S32` only in `DENSE32`.

On an affected reset, form the invalid mask exactly as

```text
invalid = fnp.invert(valid_high_order)
idx     = fnp.flatnonzero(invalid)
r       = idx.shape[0]
```

The invert and scan are both billed.  `r` is read only from result shape; no
array value is extracted through native NumPy, `.item()`, a raw view, a
buffer, or a list.  The all-valid branch occurs before these operations, so
the nominal path performs neither.

## Frozen exact low-rank algebra

Let `C` be the represented fp64 repaired covariance at the affected reset,
`W` the next represented fp64 weight, `E` the column selector for `idx`, and
`M=I-E E^T`.  No structural or bitwise symmetry of `C` is assumed.

Preserve the parent's existing left-associated lower-covariance calculation
and merely bind its already-existing intermediate:

```text
left = W.T @ C
Q    = left @ W
```

Define

```text
U     = W[idx, :]          = E.T @ W
Crows = C[idx, :]          = E.T @ C
G     = Crows[:, idx]      = E.T @ C @ E
A     = left[:, idx]       = W.T @ C @ E
B     = Crows @ W          = E.T @ C @ W
Lc    = U.T @ B
Rc    = A @ U
H     = G @ U
J     = U.T @ H
Qmask = ((Q - Lc) - Rc) + J
```

The combine order and all five matmul operand orders are frozen.  Algebraically,

```text
Qmask = W.T @ (M C M) @ W
      = Q - U.T@B - A@U + U.T@(G@U).
```

This identity is valid for nonsymmetric represented `C`; replacing `A` by
`B.T`, replacing `Rc` by `Lc.T`, post-symmetrizing, reassociating a matmul, or
using a Cholesky/eigendecomposition is forbidden.  The identity is exact over
real arithmetic, not a claim of bit equivalence to a dense fp64 or fp32 GEMM
spelling.  A future source probe must measure the represented disagreement
against a separately materialized fp64 `W.T @ (M C M) @ W` reference under a
bound frozen before that result is viewed.

At the affected reset itself, `LOWRANK64` forms one fp64
`S=fnp.where(valid_plane,C,0.0)` for the current K22 fit and K4/K31
decomposition.  It carries `idx`, not `S`.  At the next push it uses `Qmask`
and `diag(Qmask)` only for the Wick channel.  The lower covariance remains
the existing `Q`.  After the next ReLU, the old selector is consumed and the
new reset overwrites the state from its new repaired covariance and
certificate.  The same selected Wick covariance/variance must be used in the
ordinary and final-diagonal quartic branches.

## Frozen dense fallback and nominal path

`DENSE32` is exactly the corrected standalone CS3 operation spelling:

```text
S32       = fnp.where(valid_plane, fnp.asarray(C, fnp.float32), 0.0)
S_current = fnp.asarray(S32, fnp.float64)
S32_next  = (W32.T @ S32) @ W32
S_next    = fnp.asarray(S32_next, fnp.float64)
```

Use `S_current/diag(S_current)` for the current fit and
`S_next/diag(S_next)` for the one-layer Wick push.  No new rounding, repair,
threshold, or symmetry operation is permitted.

`NOMINAL` must be byte- and operation-ledger equivalent to the CS2 numerical
path except for the separately authorized Python helper inlining below and
removal of CS2's scalar `where(True,alpha,0)`.  Before any pairing arithmetic,
alias the already-computed fp64 `pre_covariance/pre_variance` exactly.  Do not
cast, copy, mask, recompute `W.T@C@W`, construct an eager array `where`, scan
indices, or allocate a localized covariance payload.

## Frozen resource-only selector and cost law

Under pinned flopscope `0.12.0`, the complete low-rank affected-reset bill is

```text
Lr(r,w) = 16*r*w^2 + 4*r^2*w + 8*w^2 + 8*r*w + 4*r^2 + 2*w.
```

It comprises one fp64 current-state matrix `where`, two zero-FLOP diagonal
views, one Boolean invert, one `flatnonzero`, four advanced gathers, five fp64
matmuls, one zero-FLOP transpose, and three fp64 matrix combines: 18 calls.
The exact arithmetic derivation is

```text
five matmuls plus three combines = 16*r*w^2 + 4*r^2*w - 4*r*w
four advanced gathers            = 12*r*w + 4*r^2
one fp64 matrix where             = 8*w^2
invert plus flatnonzero scans     = 2*w.
```

The frozen dense block is

```text
D0(w) = 4*w^3 + 8*w^2
```

and has eight localized calls.  The resource selector is exactly

```text
LOWRANK64 iff Lr(r,w) <= D0(w); otherwise DENSE32.
```

It is a conservative operation-bill comparison, not an accuracy decision.
The affected dense branch has the additional common selector work and
therefore actually bills `Dh(w)=D0(w)+2*w` and ten calls.  No hard-coded rank
threshold is permitted.  At width 1024 the formula happens to select
`LOWRANK64` for `r<=241`: `Lr(241,1024)=4,291,806,084 <= 4,303,355,904`, while
`Lr(242,1024)=4,310,571,792` fails the inequality.

Relative to literal CS2, for `a_l` low-rank resets and `a_d` dense resets,

```text
Delta_F = sum_k Lr(r_k,w) + a_d*Dh(w) - 112
Delta_calls = 18*a_l + 10*a_d - 14 + n_null,
```

where `a_l+a_d<=14`, `-112/-14 calls` removes CS2's 14 scalar fp64 `where`
operations, and `n_null` counts the one zero-FLOP constructor call required by
the exact-null fitting branch.  Because selected low-rank bills never exceed
`D0` and the dense branch is `D0+2w`, the all-14 worst-case width-1024
projection is

```text
2,131,652,533,303 + 14*(4,303,355,904 + 2,048) - 112
= 2,191,899,544,519 FLOPs
= 99.67605112792626% of 2^41
```

with `7,123,711,033` FLOPs projected headroom.  A future literal candidate
source meter is authoritative.  Neither observed invalid rank nor backend
timing may replace this worst-case gate.

## Frozen pairing fit and corrected null face

For either affected mode, fit the parent's same off-diagonal K22 projection
after subtracting CP4, using the selected current represented `S`:

```text
P = offdiag(outer(diag(S),diag(S)) + 2*S*S)
alpha = <sym(K22),P> / <P,P>.
```

If the represented denominator is exact zero, `P` is a null off-diagonal K22
fitting plane; it does not imply the full Wick tensor is zero.  Freeze
represented `+0.0` as the unique minimum-norm coefficient among the equally
good K22 fits, leave unmatched representable K4/K31 in the quartic star, and
report the K22 residual.  Preserve the regular fp64 division bill on that
branch with an operation-equivalent `+0.0/1.0` tracked division and one
zero-FLOP scalar constructor.  No epsilon, gamma threshold, variance floor,
or fitted null threshold is allowed.

## Authorized residual-only certificate inlining

The CS2 literal meter's only terminal failure was residual time.  A future
CS3B source may mechanically inline `_up`, `_add_sub_error`, and
`_multiply_error` call bodies inside `_certified_invalid_marginals` to remove
Python helper-call frames.  This authorization is strictly conditional on
byte- and operation-equivalence:

- retain every `fnp` operation, operand, dtype, temporary dependency, and
  evaluation order exactly;
- replace every `_up(x)` only by `fnp.nextafter(x,fnp.inf)` at that exact point;
- do not reassociate, fuse, vectorize across expressions, use native NumPy,
  change `_TWO_U`, replace outward rounding by a gamma bound, or alter either
  certified sign comparison;
- do not change `valid_high_order`, its outward semantics, or any downstream
  mask on any input.

The reference certificate has six `_multiply_error` expansions, five
`_add_sub_error` expansions, and this exact 154-call operation ledger per
reset:

```text
nextafter 57, abs 23, multiply 35, add 28, subtract 5,
negative 2, less 2, bitwise_or 1, zeros_like 1.
```

Inlining removes 68 Python helper frames per certificate evaluation (57
`_up`, six `_multiply_error`, five `_add_sub_error`) and 952 frames over 14
recursive resets, but it must remove zero tracked calls and zero FLOPs.  This
is a residual hypothesis only; no residual-time improvement is claimed before
a separately authorized literal meter.

## Frozen smallest source probe

No source is authorized until an additive coherent design receipt binds an
independent audit of this exact preregistration.  A later immutable candidate
may run only the following target-free source probe before any full-width or
meter authority:

1. Static diff restricts changes to the named carrier/state seams, binding the
   existing `left=W.T@C; Q=left@W` operation order and the certificate macro
   expansion.
2. A deterministic certificate-equivalence probe compares parent-helper and
   inlined forms on exact determinant-boundary tuples, their one-ULP
   perturbations in both directions, signed zeros, subnormal-scale finite
   tuples, large finite tuples, and a fixed-seed finite fuzz roster.  It
   requires byte-identical invalid masks, byte-identical named intermediates,
   and the exact 154-call ledger above.  No observed difference is tolerated.
3. Primitive nonsymmetric matrices at widths 4, 5, 6, and 8 exercise every
   rank and both sides of the exact resource selector.  The low-rank result is
   compared with separately materialized fp64 `W.T@(M*C*M)@W` under a
   pre-result gamma bound; no symmetry shortcut is allowed.
4. Exact small-width source deltas must match `Lr`, `Dh`, and the mixed total
   law for multiple low/dense activation patterns and exact crossover cases.
5. Run exactly the existing ten CS2 construction fixtures.  Require at least
   one `LOWRANK64` activation, at least one forced algebraic `DENSE32` selector
   fixture, zero parent-finite regressions, no-activation byte parity, two-run
   byte determinism, and parent-scale finite disagreements.
6. Reconstruct every current and one-push pairing, star, CP3, CP4, and
   aggregate connected carrier.  Require exact invalid-incident zeros at the
   reset, exact mapped zeros only for identity/signed permutations, the dense
   floating bound for dense weights, correct null-face K4/K31 decomposition,
   and valid-valid K22 residual no worse than alpha-zero CS2.

Failure of certificate equivalence, algebra, cost, nominal parity, finiteness,
or any exact carrier invariant closes this precise CS3B spelling.  It does not
authorize changing the selector, assuming covariance symmetry, tuning a
threshold, switching all paths to fp64, changing the fixture roster, or
opening full-width, meter, accuracy, package, or board authority.

## Access and authorization boundary

- CS3B candidate estimator source files: zero;
- estimator executions and width-1024 runs: zero;
- dataset rows, targets, teachers, public rows, scores, and private state:
  zero;
- packages, submissions, network, and credentials: zero;
- canonical graph edits: zero;
- current authority: immutable design, independent audit, and source-builder
  handoff preparation only.


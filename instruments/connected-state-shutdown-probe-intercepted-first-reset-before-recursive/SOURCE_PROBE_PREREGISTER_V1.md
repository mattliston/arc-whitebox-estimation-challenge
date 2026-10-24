# Q24 CS3B frozen target-free source probe v1

Status at seal:
`PREREGISTERED_AND_SOURCE_FROZEN__EXECUTION_BLOCKED_PENDING_INDEPENDENT_STATIC_SOURCE_PASS_AND_ONE_USE_RELEASE`.

Sealed at `2026-08-24T14:31:58Z`, after construction and immediate HOLD of
candidate source SHA-256
`49060ee223137d98fc2b43d94e4d5e5d99290c7289bccd3043e3e078775b0aaa`,
but before importing or executing that candidate.  Candidate import count,
candidate execution count, full-width runs, literal-meter runs, dataset or
target reads, score reads, packages, submissions, network accesses, and graph
edits were all zero at seal.

## Immutable chain and execution lock

The exact source parent is CS2 estimator SHA-256
`4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779`.
The candidate source HOLD is SHA-256
`ac29611a20280de259c7ab2f970d865b09ca1c417d1ac09d64aeab5a89d23bb5`.
The corrected CS3B author design receipt and one-source-build release are
SHA-256 `087a4d9cecd66908e42446ccd294a53da9dd697b382ef96e03729fe8de07f894`
and `ba13a061c505d2ea61a748daf3586c6c2bace296b343a062a37c665c84864bc4`.
The corrected independent design audit is SHA-256
`f458c9dbae04e3c7745380a459a074f380f894321eb3ce612a4caac9f5b39a39`.

The independent static-source verdict is intentionally unknown at this seal.
The runner may be created and parsed, but must not import either estimator or
execute any numerical probe until a separate one-use release binds an
independent populated static-source PASS for the exact candidate SHA and this
preregistration/runner pair.  Absence, HOLD, or mismatch of that verdict is a
terminal execution blocker.

## Exact certificate-equivalence roster

Compare CS2's helper-spelled certificate, CS3B's inlined certificate, and two
observer-local trace spellings with the same operation order.  One vectorized
roster contains:

- the finite joint boundary `(r1,r2,r3,r4)=(1,2,4,8)`, on which represented
  determinant margins are at the A/B boundary;
- all single-field one-ULP neighbors of that tuple in both directions;
- explicit `+0.0` and `-0.0` combinations;
- the smallest positive subnormal and signed/subnormal-scale finite tuples;
- the finite scale families `(m,2m^2,4m^3,8m^4)` for
  `m in {2^-200,2^-40,2^40,1e30}`; and
- exactly 256 fixed-fuzz finite tuples from NumPy `default_rng(20260824)`,
  formed with signed standard-normal coefficients times powers of two drawn
  uniformly from integer exponents `[-120,120]` and rejected only if an input
  field is nonfinite.

Required gates are byte-identical invalid masks, byte-identical named stored
intermediates and error bounds between helper and inline trace spellings, and
for each candidate/helper certificate invocation the exact tracked-call
ledger:

```text
nextafter 57, abs 23, multiply 35, add 28, subtract 5,
negative 2, less 2, bitwise_or 1, zeros_like 1; total 154.
```

No numeric tolerance, NaN canonicalization, signed-zero normalization, or
mask-only waiver is permitted.

## Exact resource-selector and nonsymmetric algebra roster

For every width `w in {4,5,6,8}` and every invalid count `r in {1,...,w}`,
construct a deterministic nonsymmetric fp64 `C`, fp64 `W`, and a non-prefix
invalid index set from `SeedSequence([20260824,w,r])`.  Do not symmetrize `C`.
Exercise exactly the source spelling selected by

```text
Lr*(r,w) = 16*r*w^2 + 4*r^2*w + 8*w^2 + 20*r*w + 8*r^2 + 2*w
D0(w)    = 4*w^3 + 8*w^2
LOWRANK64 iff Lr* <= D0; otherwise DENSE32.
```

For LOWRANK64, require exactly `Lr*` tracked FLOPs and 18 calls for the
localized current/reset-plus-next-push block.  For DENSE32, require exactly
`Dh=D0+2*w` tracked FLOPs and 10 calls.  Also replay, by integer arithmetic
without allocating those matrix widths, the witnesses `(w,r,mode)`
`(5,1,DENSE32)`, `(6,1,LOWRANK64)`, `(128,29,LOWRANK64)`,
`(128,30,DENSE32)`, `(1024,241,LOWRANK64)`, and
`(1024,242,DENSE32)`.

For each low-rank matrix row compare

```text
Qmask=((Q-U.T@B)-A@U)+U.T@(G@U)
```

against separately materialized fp64
`Qref=(W.T@where(valid_plane,C,0))@W`.  Freeze the elementwise bound before
seeing results as

```text
u = 2^-53
gamma = k*u/(1-k*u),  k = 32*w + 16*r + 16
bound = gamma * (((abs(W).T@abs(C))@abs(W))
                 + ((abs(W).T@abs(S))@abs(W)) + 1)
```

and require `abs(Qmask-Qref) <= bound` everywhere.  This is a deliberately
conservative represented forward-error gate, not a bit-equivalence claim.
The four gathers, five matmuls, and three combine operand/order spellings must
match the frozen source AST; no `C` symmetry assumption is permitted.

## Frozen estimator fixtures and carrier observers

Run exactly the ten construction fixtures emitted by helper SHA-256
`2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d`:
identity width 8, near-identity width 8, QR width 8, pure-normal width 8, and
the two fixed HE seeds at each width 32, 64, and 128.  For each fixture run
the frozen CS2 parent once and CS3B twice.  Observer wrappers may copy
target-free intermediate arrays after the original operation returns, but
must not change any returned value or branch.

The ten-fixture gates are:

1. at least one natural LOWRANK64 activation;
2. no CS2-finite fixture becomes nonfinite or raises under CS3B;
3. two CS3B runs have byte-identical outputs and byte-identical mode/mask
   records;
4. every fixture with zero certificate activations is output-byte-identical
   to CS2;
5. for every finite pair, both RMS and maximum absolute disagreement are no
   larger than the corresponding CS2 output scale (zero scale requires exact
   zero disagreement);
6. current connected targets, CP3/CP4 providers, cubic/quartic stars, and their
   aggregate reconstructions have zero exact invalid-incident entries;
7. selected current covariance has exact invalid rows/columns zero on every
   affected reset;
8. identity or signed-permutation pushes preserve mapped invalid zeros
   exactly; dense pushes need only pass the frozen arithmetic bound below;
9. current K4/K31/K22 reconstruction residuals pass the fp32 bound
   `gamma(64*w,u32)*(1+maxabs(target components))`, and the valid-valid K22
   residual norm is no worse than the same CP-adjusted residual with
   `alpha=+0.0`, plus that bound; and
10. the final-diagonal and ordinary quartic calls consume the same mode's
    selected pairing covariance/variance law.

For a DENSE32 one-push, compare the represented fp32 two-GEMM result against
the same stored fp32 operands evaluated in fp64.  With
`gamma_w=w*2^-24/(1-w*2^-24)`, freeze

```text
E1 = gamma_w*(abs(W32.T as fp64) @ abs(S32 as fp64))
E2 = E1@abs(W32 as fp64)
     + gamma_w*(abs(first32 as fp64)@abs(W32 as fp64))
     + 64*2^-52*(((abs(W32.T)@abs(S32))@abs(W32))+1).
```

Require elementwise error no larger than `E2`.

## Forced branch and exact-null controls

In addition to, and clearly separated from, the ten natural fixtures:

- run one synthetic width-5 depth-16 identity-weight candidate with an
  observer override that returns exactly one certified-invalid coordinate at
  every reset.  The frozen law requires DENSE32 (`Lr*=738>D0=700`); require
  14 affected resets, no LOWRANK64 inference, exact current invalid zeros,
  and a finite output.  This override is only a branch/control fixture and
  cannot count as a natural certificate activation or accuracy result;
- run one width-6 signed-permutation primitive for each invalid count selected
  LOWRANK64 and require exact mapped zero rows/columns; and
- call the frozen `_reset_stars` on an exact null off-diagonal plane
  `S=diag(1,0,...)` with nonzero representable K4/K31 content.  Require
  positive-zero alpha bytes, the same one tracked division and call ledger as
  an ordinary nonnull control, correct K4/K31 star reconstruction, and an
  explicitly reported K22 residual.  The full Wick tensor is not asserted
  zero.

## Decision and authority

The result is immutable at `runs/source_probe_v1/results.json`.  Any missing
record, hash mismatch, exception in a required control, certificate byte
difference, resource mismatch, algebra-bound failure, parent-finite
regression, nondeterminism, carrier violation, or failed exact-null check
returns `REJECT_SOURCE_PROBE` and closes these exact source bytes.  No retry,
threshold change, fixture removal, selector change, or source mutation is
authorized.

A PASS authorizes only an immutable source-probe result and independent replay
audit.  Width-1024 estimator comparison, literal metering, residual timing,
targets, scores, packaging, submission, board promotion, and canonical graph
mutation remain separately held.

# Full-budget coherent CP4 residual-carrier audit

Status: **target-free design and resource audit only**.  This file opens no
dataset, weight stack, target, teacher, score, submission, or protected role.
It edits neither the estimator nor the canonical graph and grants no execution
authority.

## Verdict

Promote the adaptive cap-2048 CP4 carrier to algebra qualification and a
literal Phase-2 FlopScope meter.  An active source scaffold was added by a
separate lane while this read-only audit was in progress; it passes its local
target-free algebra/static checks but is not source-frozen.  It is the highest-value
safe addition found for the current FB-EC4-32 host because it retains genuine
all-distinct quartic history while the existing H4 residual still replays the
same corrected `d/A` planes exactly and fits the remaining `B` plane.  Promotion
here is an engineering/scientific-qualification decision, not an accuracy
claim.

The decisions by neighboring rung are:

- coordinate-reset/newest-only cap 1024: **HOLD**.  It is algebraically only a
  reparameterization of the H4 diagonal birth and retains no history beyond
  the transition already represented by H4.  An adaptive cap-1024 selector can
  retain historical rays, but cannot preserve as much current and inherited
  content as cap 2048 and has no compensating resource need;
- adaptive cap 2048: **PROMOTE to source + literal-meter qualification**;
- adaptive cap 2560: **PROMOTE only as the stretch rung after cap 2048 passes
  its literal meter**.  It carries 25% more CP rank and still has a projected
  17.669B arithmetic margin under the current stored-marginal reread law;
- adaptive cap 2688: **HOLD** pending an unexpectedly favorable literal meter;
  its projected margin is only 8.259B under the same reread law;
- cap 3072 with adaptive individual-ray truncation: **HOLD cost**.  Its
  push/read and twelve rank-1024 dropped-plane updates exceed the available
  289.821B before lower-order work;
- cap 4096: **HOLD cost** even without adaptive-drop accounting; and
- exact dense first-transition K3 emission/action: **HOLD cost and memory**.

Keep the bivariate pair sum at `Q=32`.  The remaining budget has higher
structural value as CP4 carrier rank or literal-meter reserve than as
`Q32 -> Q38` resolution.

## State, affine action, and initialization

The provider is one signed symmetric CP4 tensor

```text
G4 = sum_(r=1)^R lambda_r u_r ^ tensor4,       R <= cap.
```

Signed `lambda` is allowed; this is a cumulant representation, not a positive
probability mixture.  For an affine map with weight `W`, set

```text
V = W^T U.
```

The exact visible planes of this represented tensor are

```text
d_i   = sum_r lambda_r V_ir^4,
A_ij  = sum_r lambda_r V_ir^3 V_jr,
B_ij  = sum_r lambda_r V_ir^2 V_jr^2,

A = (V^3 diag(lambda)) V^T,
B = (V^2 diag(lambda)) (V^2)^T,
d = diag(B).
```

Thus `A` is the directed K31 plane and `B` is the symmetric K22 plane of the
same tensor.  Neither plane can be omitted: `A` is required to subtract the
provider from the H4 K31 residual, and `B` is required both by Q32 and by the
H4 pairing gauge.  The H4 matrix products cannot be reused for these reads,
because their coordinate-star factors differ from `U`; the already-computed
CP4 planes can, however, be reused across the nonlinear reset as below.

At `h1`, initialize

```text
U = I,
lambda_i = d_i^(h1)
         = (3/4 - 1/pi - 3/(2 pi^2)) * variance(z1_i)^2.
```

This marginal is exact for a rectified zero-mean Gaussian.  Subtract these
identity-provider planes in the initial H4 residual reset.  At this point the
CP diagonal tensor is exactly the same tensor as the removed diagonal
coordinate-star term, so the total state and its W2 action are unchanged.
Novel information first appears after the inherited dense rays survive the h2
reset and act at z3.

## Exact nonlinear reset and truncation law

It is important to distinguish two provider tuples:

- `P_pre=(d_pre,A_pre,B_pre)`, added once to the pushed H4 planes before the
  Q32 consumer; and
- `P_post=(d_post,A_post,B_post)`, formed after gating, birth, and truncation,
  then subtracted once from the post-ReLU target before resetting H4.

They are planes of the same causal carrier on opposite sides of the nonlinearity
but are not numerically identical.  Passing `P_pre` into the post-ReLU residual
seam would be a source bug.

Let

```text
g_i = Phi(mu_i / sigma_i).
```

Gate every inherited factor using the same leading tangent law as the CP3
carrier:

```text
U_g = diag(g) V.
```

No second full CP4 read is needed.  The inherited post-gate planes follow
elementwise from the preconsumer planes:

```text
d_g = g^4 * d_pre,
A_g = diag(g^3) A_pre diag(g),
B_g = diag(g^2) B_pre diag(g^2).
```

Append one coordinate block

```text
U_birth = I,
lambda_birth = d_target - d_g.
```

Before truncation the candidate CP tensor therefore has marginal plane exactly
`d_target`; its identity birth has zero strict-off-diagonal A and B action.

When the candidate rank exceeds the cap, rank every atom by

```text
e_r = lambda_r^2 (sum_i U_ir^2)^4
    = ||lambda_r u_r^tensor4||_F^2.
```

This is the exact squared Frobenius norm of the individual rank-one quartic
atom and is invariant under `u_r -> c u_r, lambda_r -> lambda_r/c^4`.
Use metered stable `fnp.argsort(-e, stable=True)` so raw index is the
deterministic tie break.  Keep the first `cap` rays and call the dropped set
`D`.  At cap 2048, every
steady-state candidate has rank 3072 and exactly 1024 rays are dropped.

Read the dropped tensor coherently:

```text
d_D   = sum_(r in D) lambda_r U_ir^4,
A_D   = (U_D^3 diag(lambda_D)) U_D^T,
B_D   = (U_D^2 diag(lambda_D)) (U_D^2)^T.
```

The exact retained-provider planes are then

```text
d_post = d_target - d_D,
A_post = A_g - A_D,          # strict off diagonal
B_post = B_g - B_D.          # strict off diagonal
```

This complement spelling is why only two rank-1024 dropped-plane GEMMs are
required per capped reset.  It is cheaper than rereading the retained
rank-2048 tensor.  If a dropped ray came from the coordinate birth, its
off-diagonal `A_D/B_D` contribution is exactly zero; the equations remain
valid without a special case.

Now form residual target planes

```text
d_R = d_target - d_post,
A_R = A_target - A_post,
B_R = B_target - B_post,
S_ij = C_ii C_jj + 2 C_ij^2,
alpha = <B_R,S> / <S,S>             (strict upper triangle),
H_ii = d_R_i - 3 alpha C_ii^2,
H_ij = A_R_ij - 3 alpha C_ii C_ij.
```

The post-reset total is

```text
F4 = G4_retained + Star4(H) + alpha P_C.
```

By construction its marginal K4 and directed K31 planes are exactly
`d_target` and `A_target`; its K22 plane is the orthogonal one-dimensional
projection of the remaining residual onto `S`, exactly the incumbent H4
gauge but after subtracting the coherent retained CP contribution.  The
subsequent PSD repair is still applied to the total Q32 covariance.

Top-energy selection is preferable to a fixed age rule.  Repeated gates
naturally reduce stale rays, while dense affine transport can preserve an old
ray if it remains strong.  A newest-only FIFO rule guarantees renewal but
throws away potentially valuable early history; an oldest-only rule can starve
the carrier of new births.  Individual energy is not a best-rank CP theorem:
nonorthogonal atoms can cancel.  Qualification must therefore report age
histograms, signed cancellation ratios, selected/dropped plane norms, and
finite/PSD behavior before any scientific promotion.

## Why this is new information, not duplicated state

The current host already contains:

- CP16384, which is an order-three carrier and supplies K3/K21;
- H3, an order-three coordinate-star residual;
- `Star4(H)+alpha P_C`, which exactly replays the visible quartic d/A planes
  and fits one B direction but discards most all-distinct quartic history;
- a total-covariance PSD repair; and
- a Q32 `1+L3+L4+1/2 L3^2` consumer of d/A/B.

At a reset, an identity CP4 block is algebraically identical to the diagonal
part of `Star4(H)`.  After a dense affine map and another nonlinear reset,
retained non-axis CP rays contain all-distinct `1111`, `211`, and other
quartic action that a fresh coordinate star does not remember.  The residual
split prevents duplicate d/A/B accounting, while Q32 and the PSD repair make
the old diagonal-CP4 failure modes addressable.  This is therefore a genuine
new carrier beginning at z3, not another K31 formula stacked on the consumer.

## Current-source audit boundary

The concurrently added cap-2048 source spells the correct causal order,
initializes the h1 identity block, separates pre- and post-provider tuples,
uses a stable energy sort, reads the dropped A/B planes coherently, and
specializes W16 to the marginal.  Its local `verify_algebra.py` and
`verify_static.py` both pass.

Two claims must nevertheless remain qualification gates rather than frozen
facts:

1. The source materializes
   `inherited_u=float32(gate32[:,None]*pushed_u)` but derives inherited planes
   A/B by float64 elementwise scaling of the pre-gate planes.  It now rereads
   `d` from the rounded stored factors, so the coordinate birth closes the
   actual stored marginal.  In finite float32,
   `round(g*u)^p` is not identically `g^p u^p`.  Dropped planes are then read
   from the rounded stored factors and subtracted from the analytically scaled
   A/B candidate planes.  Consequently the returned A/B complement is exact
   in the ideal real-arithmetic CP law above, but is not bit-exactly the plane
   of the stored float32 retained factors.  H3/H4 themselves are also stored in
   float32, so an implementation may deliberately accept this as the declared
   carrier-rounding model; it must not describe the finite stored state as an
   exact d/A replay until a tiny-tensor rounding residual is measured or the
   representation is changed.
2. The checked-in resource ledger reports a `234,451,171,328` CP4 subtotal:
   the 233.997B dominant subtotal, `397,488,128` enumerated lower-order work,
   and `56,608,768` for stored-gated inherited-d rereads.  The last number
   counts only each matvec dot.  The literal `_cp4_read_diagonal` also forms
   `u2=u*u` and `u2*u2`, costing another `2nr` per reread, or `56,623,104`
   across one rank-1024 and thirteen rank-2048 resets.  Those post-gate
   `inherited_u` powers cannot reuse the pre-gate plane-read powers already in
   the `397,488,128` subtotal.  The explicit vectorized spelling below thus
   enumerates `510,720,000` lower-order operations and a
   `234,507,794,432` CP4 subtotal.  This does not threaten the budget, but the
   literal meter--not the smaller checked-in ledger headline--is
   authoritative.  Gather, cast, and dispatch deltas remain outside even the
   enumerated scalar count.

Neither issue changes the PROMOTE-to-meter verdict.  Both block a claim of
bit-exact finite-state reconstruction or an exact resource PASS.

## Width-1024 arithmetic ledger

The exact FB-EC4-32 specialized envelope is

```text
base                                    1,909,202,324,994
budget = 2^41                           2,199,023,255,552
available                                  289,820,930,558
```

For cap 2048, ranks at W2 through W16 are

```text
r = [1024, 2048, ..., 2048],
sum(W2..W16 r) = 29696,
sum(W2..W15 r) = 27648.
```

With a float32 dense dot of length `k` billed as `2k-1`, the dominant carrier
arithmetic is

| item | exact arithmetic |
|---|---:|
| one factor push, W2..W16 | 62,246,617,088 |
| K31 A reads, W2..W15 | 57,967,378,432 |
| K22 B reads, W2..W15 | 57,967,378,432 |
| W16 marginal d read plus n-vector host add | 8,388,608 |
| 13 x two rank-1024 dropped A/B GEMMs | 55,807,311,872 |
| **dominant subtotal** | **233,997,074,432** |

The W16 row combines an exact `n(4r-1)` factor-power/dot count with the final
`n` additions.  A direct full-rank spelling of the two planes is exact even
when implemented blockwise: the block GEMM dot counts plus block-plane adds
collapse to `n^2(2r-1)`.

A concrete vectorized reset spelling adds the following scalar arithmetic:

| lower-order item | arithmetic |
|---|---:|
| W2..W15 factor powers and lambda scales | 113,246,208 |
| stored gated-factor d rereads, h2..h15 | 113,231,872 |
| provider adds into the preconsumer host | 29,374,464 |
| inherited factor gates | 28,311,552 |
| post-gate A/B elementwise actions | 58,720,256 |
| gate powers | 28,672 |
| coordinate-birth residuals | 14,336 |
| provider subtraction before H4 reset | 29,374,464 |
| 13 energy constructions | 54,659,072 |
| 13 weighted argsorts (`4 k ceil(log2 k)`) | 1,916,928 |
| energy negations | 39,936 |
| dropped-factor powers and lambda scales | 54,525,952 |
| retained-plane complement updates | 27,276,288 |
| **lower-order subtotal** | **510,720,000** |

Hence the enumerated cap-2048 add-on is

```text
CP4 cap2048 add-on                        234,507,794,432
combined static envelope                2,143,710,119,426
remaining below 2^41                       55,313,136,126
```

This is not a literal-meter PASS.  Gather/take billing, casts, dispatch shape,
transport overlap, and the complete host's actual meter remain to be measured.
The 55.313B remainder is deliberately not claimed as free compute.

### Stretch cap 2560

The same adaptive law at cap 2560 has ranks
`1024,2048,2560,...,2560`.  Its first truncation drops rank 512 and its next
twelve truncations drop rank 1024:

```text
push/A/B/final main                       217,913,491,456
drop A/B GEMMs                             53,659,828,224
lower-order spelling before d reread           439,750,656
stored gated-factor d rereads                   138,397,696
CP4 cap2560 add-on                        272,151,468,032
combined static envelope                2,181,353,793,026
remaining below 2^41                       17,669,462,526
```

Cap 2560 is the recommended full-budget stretch after the cap-2048 literal
meter is known.  Cap 2688 projects to `281,562,386,432` add-on arithmetic and
only `8,258,544,126` total headroom; cap 2800 leaves only `23,990,526` and is
not a safe design boundary.

## Memory

For cap 2048, the exact persistent CP4 payload is

```text
U       1024 x 2048 x 4                 8,388,608 bytes
lambda  2048 x 4                            8,192 bytes
total                                       8,396,800 bytes
```

The uncapped live candidate is `1024 x 3072` float32 plus 3072 weights,
`12,595,200` bytes.  One dropped rank-1024 factor block is `4,198,400` bytes
including weights.  Even retaining simultaneous pre, gated, and dropped A/B
planes plus factor-power work gives a conservative add-on working bound below
96 MiB.  This is small beside the current Q32 group-three bound
`830,472,192` bytes and the 8-GiB solution limit, but only a combined literal
peak-RSS run can pass the memory gate.

## Q32 headroom decision

Under the current static convention, increasing the pair cutoff by one term
costs exactly

```text
Delta_Q = (14*4*17 + 4) * 3 * 1024^2 * 2
        = 6,014,631,936.
```

The first term covers fourteen resets, four pair raw moments, and seventeen
derivative channels; `+4` is the Gaussian h1 pair work.  Therefore cap 2048
plus Q38 would cost another `36,087,791,616`, for combined total
`2,179,797,911,042` and only `19,225,344,510` projected headroom.  Q41 is the
arithmetic maximum, leaving just `1,181,448,702`, and is not meter-safe.

Frozen strict-F1 source diagnostics already place the maximum Q16-vs-Q32
action envelope at `3.340183908336817e-5`; individual rows range down to about
`1.28e-5`.  That is orders below the structural carrier errors measured in the
same research line.  It is not a Phase-2 trajectory theorem, but it is direct
evidence that pair-series resolution is not the leading missing state.  The
recommended cutoff is consequently **Q=32**, preserving 55.313B at cap 2048
or using part of it for the cap-2560 stretch rather than buying Q38.

Q38 would also increase the group-three rho working bound from
`830,472,192` to `981,467,136` bytes.  Memory still fits in isolation, but the
extra 144 MiB provides no evidenced structural gain.

## Exact early-layer alternative

The exact zero-threshold all-distinct Gaussian h1 emitter is mathematically
available, but its width-1024 direct action is not an add-on candidate.  With

```text
P = C(1024,2) = 523,776,
T = C(1024,3) = 178,433,024,
```

the frozen literal spelling has

```text
formula ordinary arithmetic               13,387,193,856
ordered K21 action arithmetic           1,646,763,617,280
complete ordinary arithmetic            1,660,150,811,136
sqrt calls                                  714,256,896
asin calls                                  535,822,848.
```

That one transition alone is 5.73 times the entire pre-CP4 headroom.  A dense
float64 width-1024 third-order emitter tensor occupies exactly 8 GiB before
the mean, covariance, CP3, H3/H4, Q32 workspace, or any temporary.  The
independently cleared complete first-order emitter supplies all `aaa/aab/abc`
classes, but it does not provide a deployable compression or change these
economics.  It remains HOLD relative to the coherent CP4 carrier.

## Evidence bindings

All evidence below was read-only in this audit.

| evidence | SHA-256 | interpretation |
|---|---|---|
| `fullbudget_k21_k4_closure_v1/DESIGN.md` | `0586e53cc8063c32e686822440f92ec7f776da9b088597aa8390d271b7ad8d00` | current active cap-2048 equations and qualification boundary |
| `fullbudget_k21_k4_closure_v1/estimator.py` | `663a6593e5b6574f08634f921d5a871303d1f9cb981d06010b1b712df8ad4265` | current unfrozen active CP4 source with stable raw-index tie and stored-factor marginal reread |
| `fullbudget_k21_k4_closure_v1/resource_ledger.py` | `646293a738aa7a1159dd81569173f1713297c5ec58f0ece0b4695eef9df3e983` | current source projection; inherited-d rereads omit two post-gate factor-power passes totaling 56,623,104 |
| `FULL_BUDGET_HIGH_ORDER_ARCHITECTURE_STATUS.md` | `8b8c75be4931ff4d2c6e613266010fd5baae10a8bda7be5c04f951e5e6475016` | current CP16384/H3/H4/PSD/Q32 composition |
| `../x3_r0c1_kappa3_ceiling/diagonal_cp4_rolling_action/RESULTS.json` | `138541d8357e23c88c676dd9ea413f55a60a1ab4a8b3eed12405eb4b7d4e28e7` | 77.892% oracle K4 mean-gain retention; mixed-plane and PSD failure |
| `../x3_r0c1_kappa3_ceiling/coherent_rank24_carrier_cp4_k31_provider/FINDING.md` | `7dd37b112f7655dba6df4e1fd801da02506acfa60facc8dacc582139b44df148` | local A improvement but 1.000800x adverse final ratio in the old host |
| `../x3_r0c1_kappa3_ceiling/coherent_next_strict_ritz24_pairadjoint_cp4_k31_provider/FINDING.md` | `85399bcf92037bcbc5643d7882a87411a28d9a5fb8b0a9eb640123a57c0c4535` | source descent passed, old-host final ratio 1.0000816x adverse |
| `../x3_r0c1_kappa3_ceiling/carried_tt3_standard_rank23/RESULTS.json` | `4d9ad5a8d3894e541e6fb96646b9dc0f862e56d61f234757f1d6accea8aec3f9` | maximum frozen Q16-vs-Q32 envelope 3.34018e-5 |
| `../x3_r0c1_kappa3_ceiling/FULL_KAPPA3_EMITTER_DERIVATION.md` | `bb0de111504bf31d5624d8353cf50a12ff12fb335a6d9e8d4868bba5a93890b` | complete first-order aaa/aab/abc equations |
| `../x3_r0c1_kappa3_ceiling/full_kappa3_emitter_v2_candidate/INDEPENDENT_SOURCE_AUDIT.json` | `e239087fe28ee8c08fbcb5bb96226f59bd8ee8f48b10d71b2c1151547fa4890c` | complete emitter source algebra cleared, no transport authority |
| `../x2_network_birth_all_distinct_singular_economics_audit/runs/run-002/economics_results.json` | `60c9e19014ca43aab30c21549bdd997667524c55f3d2baa32fc38b6f13a21b3f` | literal width-256 formula/action census used for exact scaling |

The two old coherent K31 providers are negative controls, not duplicates of
this proposal.  They supplied a small independently scheduled A plane in an
older incomplete host and did not jointly residualize a carried CP tensor's
`d/A/B` against coherent H4, run the full Q32 consumer, and repair the total
covariance spectrally.  Their failures argue against coefficient-one stacking
and make the single-add/single-subtract residual law mandatory.

## Qualification gates

Before any scientific or board use, a successor must independently verify:

1. tiny-tensor equality for CP push, gate, birth, complement/drop, and total
   `CP4 + H4 + alpha P_C` d/A/B planes;
2. exact initialization equivalence and first possible divergence at z3;
3. the distinction between `P_pre` and `P_post` in call order;
4. energy gauge invariance, deterministic ties, keep/drop complement, and
   noncontiguous gather behavior;
5. a literal full-shape FlopScope bill, residual wall below 0.4 seconds,
   total wall below 120 seconds, peak RSS below 8 GiB, finite `[16,1024]`
   output, and no computation outside FlopScope;
6. source-only age/cancellation/plane-norm diagnostics and PSD behavior; and
7. one frozen cap-2048 decision before the cap-2560 stretch is considered.

No Q extension, dense emitter, target open, fit, sampling arm, or submission
is licensed by this audit.

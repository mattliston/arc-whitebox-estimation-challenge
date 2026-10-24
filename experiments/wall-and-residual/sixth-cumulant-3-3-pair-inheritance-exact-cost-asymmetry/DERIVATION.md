# Raw K33 pair-inheritance local-reset carrier

Status: **STATIC DERIVATION ONLY — target-free discriminator not yet released**.

This leaf proposes one bounded, signed, order-six pair-inheritance mechanism:
the off-diagonal connected cumulant

\[
K_{33,ij}=\operatorname{cum}(X_i,X_i,X_i,X_j,X_j,X_j),\qquad i\ne j,
\]

born at a post-ReLU reset and consumed exactly one affine transition later as
the repeated-`3+3` contribution to the next marginal sixth cumulant.  It is
not a lambda-cubed sidecar, a low-rank plane, a full sixth-order tensor, or a
partition-complete `kappa6` closure.

## 1. Frozen host and coefficient convention

The only production parent considered here is the board-proven D4 source:

`p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py`

SHA-256:
`6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e`.

This binding matters.  D4 has `_Q24_TERMS=7`, so its bivariate Mehler sum is
exactly `t=0,...,6`.  Its 17 derivative addresses are

```text
(0,0),
(0,3),(1,2),(2,1),(3,0),
(0,4),(1,3),(2,2),(3,1),(4,0),
(0,6),(1,5),(2,4),(3,3),(4,2),(5,1),(6,0).
```

The coefficient table reaches `q=38`, but the new centered source uses only
the gathered indices `q=t+a` through 12.  No `t=0,...,24` ancestor and no
lambda3 channels 17:35 are permitted.

For preactivation coordinate `i`, let

\[
c_{p,q,i}=\mathbb E_G[X_{i,+}^p\,\mathrm{He}_q(Z_i)],\quad p=1,2,3,
\]

be the existing D4 coefficient rows, and let `mu_i` be the corrected Q24
post-ReLU mean.  Define the raw centered-cubic coefficient row

\[
d_{q,i}=c_{3,q,i}-3\mu_i c_{2,q,i}+3\mu_i^2c_{1,q,i}
          -\mu_i^3\mathbf 1[q=0].
\]

The last term is a constant over the full Gaussian measure.  It is not a
ReLU-support indicator: `c_{0,q}=1[q=0]`.

For derivative order `a` and Mehler order `t`, D4 gathers

\[
D_{a,t,i}=d_{t+a,i}\,\sigma_i^{-a}.
\]

It does **not** multiply by `sigma_i^{-(t+a)}`.  The correlation power
`rho_ij^t` supplies the Mehler order; only the derivative address contributes
the explicit inverse scale, exactly as in the frozen `_q24_context`.

Let `(a_r,b_r)` be the 17 addresses above and let `omega_r,ij` be the frozen
raw D4 weights for `1+L3+L4+0.5 L3^2`.  One new pair contraction gives the
centered mixed moment

\[
C_{33,ij}=\sum_{r=0}^{16}\omega_{r,ij}
 \sum_{t=0}^{6}\frac{\rho_{ij}^t}{t!}
 D_{a_r,t,i}D_{b_r,t,j}.
\]

This is one centered-feature contraction.  A production implementation must
also replay it on a tiny fixture against the expanded raw
`M32/M23/M33` polynomial; it must not silently replace it with one raw `M33`
call.

## 2. Exact connected assembly

Write `C_ab(i,j)=E[(X_i-mu_i)^a (X_j-mu_j)^b]`.  Exhaustive enumeration of
the 203 set partitions of three labelled `i` and three labelled `j` variables
gives

\[
\begin{aligned}
K_{33}={}&C_{33}-3C_{20}C_{13}-3C_{02}C_{31}-9C_{11}C_{22}
           -C_{30}C_{03}-9C_{21}C_{12}\\
         &+18C_{20}C_{02}C_{11}+12C_{11}^3.
\end{aligned}
\]

Using the host's connected planes, with `v_i=C20`, `v_j=C02`, covariance
`C=C11`, `K13=K31.T`, and `K12=K21.T`, the same identity is

\[
\boxed{
K_{33}=C_{33}-3v_iK_{13}-3v_jK_{31}-9CK_{22}
       -k_{3,i}k_{3,j}-9K_{21}K_{12}-9v_iv_jC-6C^3.}
\]

All products in this display are elementwise.  Existing `K31` and `K22` are
connected cumulants, not the central moments `C31` and `C22`; confusing them
would omit the final `-9 v_i v_j C - 6 C^3` content.

Before use, record the pre-symmetrization relative asymmetry, require it below
the preregistered tolerance, then set

\[
K_{33}\leftarrow\tfrac12(K_{33}+K_{33}^{\mathsf T}),\qquad
\operatorname{diag}(K_{33})\leftarrow0.
\]

The literal FlopScope 0.12 spelling for the second assignment is
`fnp.fill_diagonal(k33, 0.0)`.  The superficially equivalent
`fnp.copyto(fnp.diagonal(k33), 0.0)` is forbidden because FlopScope exposes
the diagonal as a read-only view and that call raises `ValueError`.

The diagonal is deliberately zero.  Marginal `kappa6_i` is a different
source and has no authority in this leaf.

## 3. Exact one-transition action and the factor 10

Let `W[i,j]` map post-ReLU coordinate `i` to the next preactivation coordinate
`j`, and let `U=W**3`, already resident in D4.  The exact contribution of the
off-diagonal repeated-`3+3` sector to the next raw marginal sixth cumulant is

\[
\begin{aligned}
\Delta\kappa^{(33)}_{6,j}
 &=20\sum_{i<k}W_{ij}^3K_{33,ik}W_{kj}^3\\
 &=10\,[\operatorname{diag}(U^{\mathsf T}K_{33}U)]_j\\
 &=10\sum_i U_{ij}(K_{33}U)_{ij}.
\end{aligned}
\]

The multinomial multiplicity is 20.  The matrix quadratic form counts both
orientations of every unordered off-diagonal pair, hence its coefficient is
10, not 20.  Tiny ordered-six-index enumeration must match this identity
before any width-1024 run.

This action is exact only for the `[3,3]` sector.  It omits the diagonal
`kappa6`, the two-index `[5,1]`/`K51` and `[4,2]`/`K42` sectors, and every
three-or-more-coordinate sixth tensor.  Those omissions do not contaminate
the measured `[3,3]` contribution because affine cumulant push is additive by
index multiplicity.  They do mean this leaf is not a full `kappa6` closure.

## 4. Raw scale frame and L6 reset endpoints

Production stores `K33` in raw post-activation units.  It is consumed once by
raw `W**3`; no recurrent K33 congruence exists.  The next preactivation scale
is applied only at the consumer:

\[
g_{6,j}=\Delta\kappa^{(33)}_{6,j}\,\sigma_{\mathrm{next},j}^{-6}.
\]

This native raw convention is coordinate-scale exact and avoids dividing by
`tau_i^3 tau_j^3` on nearly dead post-ReLU units only to multiply those scales
back.  A standardized `B3` identity is allowed only as a tiny synthetic parity
oracle, never as the production representation.

For the univariate ReLU reset, the direct `L6` column is

\[
\Delta m_{p,j}=c_{p,6,j}\,g_{6,j}/720,
\]

for the required raw moment rows.  In particular,

\[
\Delta\mu_j=\sigma_j\phi(\alpha_j)\mathrm{He}_4(-\alpha_j)g_{6,j}/720.
\]

For the bivariate reset, the existing derivative-6 features already contain
`sigma^-6`.  Therefore add the **raw** endpoints

```text
weights[10]  # (0,6) += raw_k6[None, :] / 720
weights[16]  # (6,0) += raw_k6[:, None] / 720
```

and do not divide by `sigma^6` again.  The tiny verifier binds the identity
`(raw_k6/720)*(c[p,6]/sigma^6) = (g6/720)*c[p,6]` at nonunit scales.

## 5. Exact versus projected recurrence

The following statements are exact:

1. the D4-Q7 centered `C33` source under the frozen 17-channel pair law;
2. the connected `K33` assembly;
3. the one-transition `[3,3] -> marginal-kappa6` action; and
4. the direct `L6/720` reset coefficient.

A loop which discards the rest of `W^(tensor 6) K33`, retains only the
marginal `kappa6` vector, and feeds that vector into later low-order Q24 state
is a **diagonal projection closure**, not exact sixth-tensor transport.  It is
not licensed by this derivation.  This distinction is material because
`P2R0F3B2A` found violent autonomous instability for a different partial
higher-order state.  The first numerical authority here is source/action only;
autonomous feedback, targets, training, and deployment all remain HOLD.

The source-only discriminator runs on the frozen D4 trajectory, where every
incoming K33-derived `kappa6` is identically zero.  If a later autonomous
projection is ever licensed, it must center `d_q` around the L6-augmented
corrected post-ReLU mean and its next-layer `C33` contraction must use the
same augmented Edgeworth weights as that reset, including the incoming raw
`kappa6/720` entries at `(0,6)` and `(6,0)`.  Reusing either a stale Q24-only
mean or stale Q24-only weights after changing the reset would break recurrence
association and is forbidden.

Consequently this source screen cannot validate either augmented association:
incoming `kappa6=0` makes the augmented mean and augmented `(0,6)/(6,0)`
weights equal their base values.  Those laws are derivational requirements,
not source-screen evidence.  A later autonomous-reset gate must test both
associations explicitly after the two target-blind source rows pass.

Zero-amplitude parity is mandatory at the full public D4 entrypoint,
`Estimator.predict -> _predict_corrected`: with the K33 consumer amplitude
fixed to zero, the internal host mean tape, `native_raw`, `native_sigma`, the
learned native-J correction, and final max-corrected public prediction must
each be byte-identical to a fresh-process frozen-parent call.  The incremental
bill is the full instrumented public-predict bill minus that full base bill.
Passing the source discriminator licenses only a separately reviewed
reset/stability design, not a target open.  Any future augmented recurrence
must recompute the native tape and J correction on the augmented state; it may
not reuse a frozen base-J correction.

## 6. Strict graph dedupe

- **Lambda3 host and failed full-plane S0:** lambda3 channels 17:35 are
  forbidden.  K33 uses one new centered-cubic pair statistic under D4's base
  channels 0:16 and an exact `W**3` marginal-sixth action.  It is not a
  weakened/full-plane retry or a low-rank continuation of the closed S0.
- **`P2R0G8` mixed-D6 `(1,2,3)`:** that is a three-coordinate Gaussian
  quotient shell.  K33 is the two-coordinate repeated `(3,3)` pair-inheritance
  sector and is not its full or quotient read.
- **`N5H6A1A2A4/A5/A5A/A5B`:** `(3,2,1)` star and `(2,2,2)` triangle exhaust
  fully joint degree-six Gaussian births only after marginal/pair inheritance
  is quotiented out.  This leaf instantiates the explicitly excluded pair
  inheritance; it does not claim a third all-distinct birth orbit.
- **`R0C1E1B12A112P16`:** P16 learns a carried 17x17 state on a fixed pair
  chart and failed its action gate.  K33 is an analytic connected moment with
  a fixed exact action, no chart recurrence, and no learned state transition.
- **`P2R0F2N`, G11A, Tucker/HT:** those are low-rank covariance/directional
  actions or tensor compressions.  K33 is a dense signed analytic pair plane
  consumed without rank projection.
- **`P2R0F3B2A`:** that experiment carried diagonal/K41/K32 fifth-order state;
  its E arm was explicitly `kappa6`-free.  K33 adds only the absent `[3,3]`
  contribution to marginal sixth order on stable D4 and does not retry K5.
- **F3 fifth-order branches (`[311]`, `[221]`, TT/CP collision carriers):**
  those concern complete fifth-order tensors/reset compilers.  This leaf is a
  two-index sixth-order statistic with one exact W3 quadratic action.

The repository contained no `K33`, `kappa33`, or `M33` implementation before
this design.  The graph's phrase "after marginal/pair inheritance" is treated
as conceptual coverage, not evidence that this missing production pair sector
was already measured.

## 7. Honest resource boundary

The independently frozen static pricing authority is

`p2r0c1d28_k33_local_reset_static_pricing_v1/K33_STATIC_PRICING_RECEIPT.json`

SHA-256:
`379d85b23513cf2d0a4ca0937b6bdf628b2b855ef29c533d5d4a7c682b30e954`.

Its immutable accounting correction is

`p2r0c1d28_k33_local_reset_static_pricing_v1/K33_STATIC_PRICING_ACCOUNTING_ERRATUM_V1.json`

SHA-256:
`351e12bd2e491d8e91cc9da6901e2763332461ab74cee17526fa6a20df05d3af`.
Both files remain byte-preserved historical authorities.  Their executable
diagonal spelling and dtype-blind numeric ledger are narrowly superseded by
`K33_PROTOCOL_ACCOUNTING_PRECEDENCE_V2.json`, SHA-256
`a950b1af3a8a211e9682def2a53596258022709f99e6bd6fbd6e4fbb33bf4778`.
That V2 authority is itself bound to the independent multi-width official
FlopScope audit `INDEPENDENT_DTYPE_ACCOUNTING_HOLD_V1.json`, SHA-256
`9ad342149f4a9628eadbcc2dd144ef84f7389d66705c0f0149969273e4fd05c6`,
and to the writable-diagonal source SHA-256
`09961351c544036c46cf8028dba9941f422d4e53043304efa402b024984bc777`.
It changes no algebra, rank protocol, memory bound, or firewall.

The original pricing receipt also contains a legacy rank-35 discriminator
sentence.  It is superseded only on that protocol point by
`K33_PROTOCOL_AUTHORITY_V1.json`, SHA-256
`9c84584c7cc54ec6efd42eec9b41e2eff1c49c272c93aed5d9703b1eba7fc823`.
That immutable authority binds singleton rank 201 first and conditional rank
643 second; it changes no science, accounting, or memory field.

The client-side target-blind output scope is extended, and only extended, by
`K33_COORDINATE_PRESERVATION_AUTHORITY_V1.json`, SHA-256
`7c95892746654c68db34b84fe18174a35e108016256ad92ecc75d6b194465295`.
For each otherwise authorized singleton rank, the immutable result may store
`raw_k6` as float64 `[15,1024]` in source-layer order h1:h15 and the exact
`L6` mean birth as float64 `[15,1024]` in consumer-layer order h2:h16.  Both
banks are fetched and persisted client-side with file and content hashes; no
`C33` or `K33` full plane is persisted.  These coordinates are preserved only
to avoid a scientific rerun before the predeclared K33-to-delta-q screen after
both source ranks pass.  They authorize no selection, target join, training,
autonomous feedback, full-estimator run, or submission.

For 15 post-ReLU sources/actions (one initial Gaussian source plus 14 full Q24
resets), its exact raw incremental ledger is:

| component | FLOPs |
|---|---:|
| centered coefficients/context | 39,768,064 |
| initial Gaussian C33 | 41,957,376 |
| fourteen Q7 K33 kernels | 10,187,964,416 |
| fourteen weighted channel reductions | 968,884,224 |
| connected assembly/symmetry/zero diagonal | 629,176,320 |
| fifteen `K33 @ W3` actions | 64,455,966,720 |
| L6 endpoint/univariate injection | 58,841,088 |
| **increment** | **76,382,558,208** |

These are literal dtype-aware FlopScope costs.  The centered row is formed on
all 39 coefficient rows before the Q7 gather; a float64 advanced gather bills
eight units per output element.  The one initial source bills
`40*n^2+14*n`; every later standalone K33 kernel bills `694*n^2`, and its
weighted reduction bills `66*n^2`.  Dividing the host's batched four-family
Q24 bill by four is not a valid price for this differently associated
standalone contraction.

For future production, the endpoint bill is
`15*6*n + 14*(4*n^2 + 2*n)=58,841,088`: each nonfinal reset materializes the
float64 `raw_k6/720` vector once and reuses it for two oriented fp64 plane
adds, while all 15 univariate births apply the raw-k6 scale exactly once.
D4 totals become
`[1,396,254,268,791; 1,432,977,846,792; 1,394,331,517,440]`, leaving at
least `766,045,408,760` FLOPs below `2^41` on the frozen max fixture.

The honest new scratch is `153,665,536` bytes (146.546875 MiB), dominated by
one 17-plane fp64 kernel bank of `142,606,336` bytes.  Projected D4 process RSS
is `6,887,735,296` bytes, below 8 GiB; the largest new array is below the
official 4 GiB single-array cap.  These are static bounds, not runtime
authority.  The stale 43.09B dispatch projection cannot bound the corrected
76.38B spelling, so an official-environment target-free residual/RSS and
remote-`fill_diagonal` gate is mandatory.

The target-blind source discriminator has a different literal arithmetic
spelling from this future production arm.  It omits the `58,841,088`-FLOP
endpoint injection and substitutes a fixed, diagnostic-only mean tangent.
For the 15 local `L6` births, set `u2=b2` and perform exactly 14 subsequent
transports

```text
u_r = b_r + Phi(alpha_r) * (W_r.T @ u_(r-1)),  r=3,...,16.
```

At width 1024, the first birth bills `6*n`; each of the 14 later
birth-plus-transport updates bills `4*n^2+8*n`, for `58,841,088` tangent
FLOPs.  Mixed fp32-weight/fp64-vector matvecs resolve and bill at the fp64
rate.  The exact remote
source-discriminator increment is therefore

```text
76,382,558,208 - 58,841,088 + 58,841,088
  = 76,382,558,208 FLOPs.
```

The equal totals are coincidental; the endpoint and tangent operation graphs
remain distinct and both must be audited literally.

This tangent is only a deterministic telemetry channel.  Its magnitude is
not a hard scientific gate because it omits covariance and connected-plane
feedback.  Likewise, fixed-layer `L6/L3^2` birth ratios are reported without
threshold or selection authority.  The remote arithmetic core returns
target-free source/action captures sequentially; summaries are computed on
the client after fetch.  The two permitted coordinate banks are likewise
client-side artifacts and add no remote arithmetic.  Any future runner that adds remote diagnostic
arithmetic must be re-priced and re-sealed before release.

Using the corrected raw max-fixture bill ratio
`1,432,977,846,792 / 1,356,595,288,584 = 1.0563046023016395`, screening
ceilings are raw `<2.7841468996246215e-8` merely to tie the D4 adjusted bank,
`<1.1978874072224143e-8` against adjusted `7.67408396767286e-9`, and
`<1.1504213699536031e-8` against the user's requested `7.37e-9`.  Final
promotion must use the literal per-network scoring law
`mean_i[MSE_i*max(0.1,F_i/2^41)]`; these fixture ratios are not submission
authority.  Raw `<=1.3e-8` is only a scientific capacity gate.

## 8. Current disposition

**GO** to freeze and independently review the target-blind source/action
discriminator in `PREREGISTRATION.json` and the tiny static verifier.

**HOLD** every width-1024 numerical run until an independent reviewer binds
the final hashes.  No K33 rank-35 call is permitted: the first possible K33
numerical call is exactly one singleton target-blind rank-201 source/action
discriminator, with no retry; rank 643 requires an immutable passing rank-201
receipt and a fresh singleton release.  Targets, training, an autonomous
feedback recurrence, a full estimator, and submission remain forbidden.

# P2R0F3B1A independent reverse-Hermite audit

Status: **terminal, do not promote to target access**.  The source-only run
ended `CLOSE_RANK_COST_DICHOTOMY`.  Reverse/forward parity passed at
`1.9245e-16`, while K12 failed accuracy and alignment on all four endpoints
(geometric-mean MSE improvement `1.31045e-4`, where `>=3` was required).  The
exact finite reverse-Hermite interpretation closes by identity plus rank/cost.
The cheap construction is a reduced, mean-only forward Duhamel correction
written in reverse notation; it is neither an exact backward functional nor
the tangent/adjoint of the full level-C closure state.

There is one important nuance for the finite-width interpretation.  The
deployed full-rank K22 residual supply contains the dominant, GEMM-factorable
quartic collision channels, but not every quartic diagram.  The exact missing
K22 crossed contraction is the already identified `2F` route.  It is therefore
algebraically distinct from the deployed supply, but it is not new to this
audit and has already measured target effect `+0.0763%` alone (slightly worse)
and `-0.5156%` when combined with the kappa211 planes.  Its generic exact cost
is quartic unless a future adjoint is proved compressible.  This is a real
missing diagram and an effect-capped one, not a credible 3x route as presently
specified.

## Audit boundary and inspected artifacts

No development target, observer store, identity, fit, or score was opened.
The competing experiment directory was read but never edited.  At first
inspection it contained only the preregistration; after its source-only run
terminated, the audit inspected the sealed runner and terminal artifacts:

- `p2r0f3b1a_reverse_hermite_v1/PREREGISTRATION.json`, SHA-256
  `dde466214752ab4959bb5af9e5ee5677a071c2c466b7cb7f1e3a4cb4b2b6b016`.
- `p2r0f3b1a_reverse_hermite_v1/PREREGISTRATION_SEAL.json`, SHA-256
  `6a62d26ff1fd3d0538c8d440d6d80aa42ac2ab8658a0724d592d50355cd20054`;
- `p2r0f3b1a_reverse_hermite_v1/run_discriminator.py`, SHA-256
  `ac38af712704b443c3a20c74e3dbe404c6600a5909e59de23c11ff7073086c9e`;
- `p2r0f3b1a_reverse_hermite_v1/RESULT.json`, SHA-256
  `785d789a9ce588989f6ab070d84c6dfdf8ec4faaacc3eabf1bbe791fb49b70e6`;
- `p2r0f3b1a_reverse_hermite_v1/TERMINAL_RECEIPT.json`, SHA-256
  `cc5c5b094c2594297ff311aac5f366362c9d10a25833d53f54baf4f51910aa68`.

The audit also read the source/receipts that define the current K22 supply and
the already measured missing routes:

- `public_supply_card_v1/build_estimator_supply.py`, SHA-256
  `ccc5ab0b709688f53eb1edd814219f92d0f5966956b071f63c22714abcce53ba`;
- `public_missing_routes_v1/missing_routes_core.py`, SHA-256
  `362b6d5fd466b8e40e2ea687072c9d2312a6840a39bccbe0687c04adc5c085a2`;
- `public_missing_routes_v1/VERIFY_MISSING_ROUTES.json`, SHA-256
  `d668db62657ea7977b4faf9df9d0547b27490142bb08feb67b70dda0b46a89ab`;
- `public_missing_routes_v1/MISSING_ROUTES_RESULT.json`, SHA-256
  `ca559f11483c8a54e4bc5a43739eb56eef05d263ea4328271f043306c94a8d33`;
- `public_k211_supply_v1/LEDGER_STEP1_RESULT.json`, SHA-256
  `22d6c18b7ee450bdc669f77fb9ba83ad6b60e48fa42365072623cf4e17647382`.

`verify_audit.py` is dependency-free and target-free.  It checks the core
identities with exact rational arithmetic.

## 1. Why an exact finite reverse-Hermite recursion does not exist

Let `He_k` be the probabilists' Hermite polynomial and `phi` the standard
normal density.  For `f(x)=x_+`, its Hermite coefficient is

```text
a_k = E[Z_+ He_k(Z)] / k!.
```

For every `k >= 2`, integration by parts twice gives

```text
E[Z_+ He_k(Z)] = phi(0) He_{k-2}(0).
```

Consequently

```text
a_2m = phi(0) (-1)^(m-1) (2m-3)!! / (2m)! != 0
```

for every `m >= 1`.  A scalar ReLU already has an infinite Hermite tail.
After dense mixing, a ridge `relu(w^T x)` spreads that tail over generic
multivariate multi-indices.  Thus no nonconstant finite-degree polynomial /
Hermite subspace containing the endpoint coordinate is invariant under the
Koopman pullback `U f = f o ReLU o W`.

Changing the basis does not perform the integration.  The exact backward
function is simply the remaining ReLU subnetwork, which has a compact network
description, but computing its Gaussian expectation is the original problem.
The model is deterministic after the Gaussian input, so
`X_l | X_(l-1)` is a point mass.  A claimed layerwise Gaussian "smoothing" is
an approximation (the forward closure in another direction), unless one
conditions on the full previous state or activation pattern.  An invertible
first dense layer in dimension `n` already has `2^n` nonempty sign cones;
activation-pattern conditional integration is exponential in that route.

## 2. Reverse projection is the dual of forward projection

For a layer map `T_l`, write `U_l f=f o T_l` and let `P_l` be its adjoint on
measures.  For an orthonormal finite Hermite basis `psi_0,...,psi_M`, define

```text
M_l[i,j] = <psi_i, U_l psi_j>.
```

The projected forward density coefficients and backward functional
coefficients obey

```text
p_l       = M_l^T p_(l-1),
v_(l-1)   = M_l v_l.
```

Therefore the final scalar is identically

```text
v_L^T M_L^T ... M_1^T p_0
  = p_0^T M_1 ... M_L v_L.
```

Changing Gaussian references merely inserts Gram matrices and does not change
the duality.  Likewise, integration by parts makes the cumulant differential
operator acting on the test function the adjoint of its Edgeworth density
bracket.  With the same retained terms and projections, forward
cumulant/Edgeworth and reverse Hermite contain exactly the same algebra.
Output-specific projection can change allocation and cost, but not remove the
discarded tail or supply a missing diagram.

The exact-rational check in `verify_audit.py` independently verifies the
finite Duhamel transpose identity for five layers and three outputs.

## 3. Audit of the frozen cheap P2R0F3B1A operator

The preregistration is commendably explicit that reverse direction is not a
new operator.  Its candidate sources are the next univariate Edgeworth terms:

```text
g5_hat C5 / 120,
g3 g4 C7 / 144,
g3^3 C9 / 1296,
(g3 g5_hat / 720 + g4^2 / 1152) C8,
g3^2 g4 C10 / 1728,
g3^4 C12 / 31104.
```

Those coefficients are correct formal exponential-series coefficients.  The
following limitations are decisive:

1. **It is a partial mean chain, not the full closure adjoint.**  A local mean
   insertion changes later pre-means, central moments, covariance, cumulant
   standardization, and reset state.  The full first-order Duhamel operator is
   the Jacobian of that complete state map.  The frozen operator propagates
   only `v <- (v @ W_k) .* Phi(mu_k/sigma_k)` while every other channel is
   held fixed.  Its forward/reverse parity gate proves only the transpose of
   this reduced operator.

2. **It is not the exact pathwise response.**  For the real network, the
   response contains the expectation of a product of downstream random gate
   matrices.  In general

   ```text
   E[D_(l+1) W ... D_D] != E[D_(l+1)] W ... E[D_D],
   ```

   because gates across layers are correlated through the same Gaussian
   input.  Replacing each gate by its marginal probability `Phi` is precisely
   a mean-field/closure step.

3. **The principal source ladder was already magnitude- and alignment-tested.**
   `LEDGER_STEP1_RESULT.json` asks exactly whether the `He5` closure-k5,
   `He7 k3*k4/144`, and `He9 k3^3/1296` columns own the mean drift.  It reports
   a net relative RMS `1.1e-6` to `7.5e-6`, versus measured drift
   `1.8e-4` to `2.3e-4`, and pushed per-unit `|corr| <= 0.114`.  The K12 rung
   remains a legitimate frozen source-only check, but reverse transport cannot
   amplify absent or orthogonal source information into a systematic 3x gain.

4. **Two widths do not identify an asymptotic exponent.**  The n=8/n=16 ratio
   is useful as the preregistered diagnostic it claims to be, not as proof of
   a quenched `1/n` law.  A fixed-weight benchmark also requires a quenched
   expansion; an annealed random-weight Onsager correction can average away
   the network-specific signal the estimator must predict.

The exact full-state check that would repair finding 1 is straightforward:
scale every local source by `epsilon`, rerun the complete frozen closure with
feedback at `+epsilon` and `-epsilon`, and compare the converged central
difference to (a) a full-state tangent/VJP and (b) the proposed mean-only
transport.  Only (a) is an adjoint of level C.  This is source-only.

## 4. The finite-width / Onsager interpretation and K22

Suppose a baseline state recursion has a leading local correction

```text
s_(l+1) = F_l(s_l) + epsilon b_l(s_l) + O(epsilon^2).
```

Writing `J_l=DF_l` gives

```text
delta s_(l+1) = J_l delta s_l + b_l,
lambda_l      = J_l^T lambda_(l+1),
delta g       = sum_l lambda_(l+1)^T b_l.
```

This is an identity.  If `b_l` is the K22 residual source, its reverse-adjoint
sum is exactly the derivative at zero source strength of inserting the same
K22 source forward.  The current full-rank supply already inserts, at every
layer,

```text
k4  += 3 diag((W^2)^T K W^2),
k31 += 3 (((W^2)^T K) .* W^T) W,
k22 += (W^2)^T K W^2.
```

It therefore contains that leading Duhamel term and, because it updates the
state at full strength, also its repeated/nonlinear feedback within the host.
Calling the first derivative "reverse Onsager" does not make it a new accuracy
term.

The deployed spelling is not the complete push of the K22 collision tensor.
For a symmetric zero-diagonal K, exact tensor contraction gives

```text
k22_exact[a,b]
  = ((W^2)^T K W^2)[a,b] + 2 F[a,b],
F[a,b] = sum_ij (W_ia W_ib) K_ij (W_ja W_jb).
```

`verify_audit.py` proves this with exact fractions; the independent existing
small-n receipt reports relative error below `1.2e-15`.  The same proof shows
the deployed k4 diagonal and k31 channels are complete for K.  Thus `2F` is a
credible algebraically distinct missing quartic diagram, but not an unmeasured
one: the two-row host result was `+0.0007628` relative alone and `-0.0051562`
with all kappa211 planes, ending `SUB_MATERIAL_LANE_CLOSES`.

Material beyond it must come from source content the current representation
does not carry: higher collision partitions / higher cumulants / provider
error.  The proposed univariate K5--K12 marginal ladder is not an enumeration
of those joint diagrams.

For cost, forming all `F[a,b]` is generically quartic.  A scalar adjoint
contraction can fall to `O(r n^3)` if its K22-plane sensitivity has certified
rank `r`; however, an entrywise bivariate ReLU Jacobian Hadamard-multiplies a
low-rank sensitivity by a generic dense coefficient matrix and can make it
full rank in one step.  Compression is possible, but exact low rank is not an
identity and needs a measured residual bound.  Producing all 1024 required
endpoint coordinates also removes much of a one-functional saving.

There is no classic dynamical Onsager feedback in this feed-forward,
untied-matrix graph.  "Loop" here means a connected Wick/cumulant contraction.
Its content must be named in `b_l`; the adjoint only contracts that content
with endpoint sensitivity.

## 5. Width-1024 rank and cost boundary

The number of multivariate Hermite coefficients of exact homogeneous degree
`K` is `C(n+K-1,K)`; through degree `K` it is `C(n+K,K)`.

| K | homogeneous degree K | through K | float32 bytes through K |
|---:|---:|---:|---:|
| 2 | 524,800 | 525,825 | 2,103,300 |
| 3 | 179,481,600 | 180,007,425 | 720,029,700 |
| 4 | 46,081,900,800 | 46,261,908,225 | 185,047,632,900 |
| 5 | 9,474,438,804,480 | 9,520,700,712,705 | 38,082,802,850,820 |

Degree 4 already exceeds the 8 GiB solution limit by more than 21x.  The
degree-5 homogeneous coefficient count alone is 4.31x the complete
`2^41 = 2,199,023,255,552` FLOP budget before a coefficient is transported or
all outputs are evaluated.  A generic dense tensor also cannot have fixed CP
rank: even the permissive parameter-count bound for nonsymmetric rank-one
terms is `rank >= C(n+K-1,K)/(K n)` (about 11.2 million components at K=4).

These are generic explicit-representation bounds, not a claim that every
structured functional requires materializing the basis.  They do prove the
dichotomy relevant here: the cheap vector ladder is cheap because it projects
away the multivariate state; a non-equivalent exact full Hermite arm must
exhibit and verify additional structure rather than call the vector ladder
exact.

### What the rank result does and does not close

The coefficient count closes an explicit generic Hermite tensor and any claim
that the frozen n-vector source ladder is secretly that tensor.  It does **not**
prove that every approximate, weight-adaptive CP/TT/ridge factorization is over
budget.  Nor does this run test a VJP of the complete retained closure state,
or an exact/approximated connected downstream-gate response.  Those are
structured, non-equivalent backward representations in principle.

None follows from this source family: every frozen source is an n-vector made
only from marginal `g3`, `g4`, and univariate `Cq`; every transport step remains
an n-vector multiplied by a marginal-gate Jacobian.  Re-factoring those same
operations cannot create a joint index, a connected gate pair, or a missing
cumulant plane.  A successor must introduce such an object explicitly and
receipt its rank/error.  Exact fixed-rank closure is not supported: one ReLU
has an infinite Hermite tail, and generic dense mixing plus entrywise
bivariate response can turn a low-rank matrix sensitivity full-rank in one
step.

## 6. Exact small-n verification plan

The competing 20-million-sample antithetic references are high precision, not
exact.  A clean exact ladder is:

1. **Width 8, depth 4, one intrinsic Gaussian dimension.**  Embed `z=t u`.
   Biasless positive homogeneity makes the answer exactly
   `phi(0) * (f(u) + f(-u))`.  This tests every layer while giving an analytic
   endpoint.
2. **Width 8, depth 4, two intrinsic dimensions.**  Enumerate the angular
   conic intervals of the homogeneous ReLU network.  On each interval the
   network is linear; integrate its sine/cosine form and the independent
   Gaussian radius analytically.  This checks correlated downstream gates
   and exposes the failure of products of marginal `Phi` gates.
3. **Hermite duality.**  On the same network, construct a finite projected
   transfer matrix and verify forward density and backward functional results
   agree to roundoff for every K.  Compare both with the exact conic answer;
   the common difference is truncation error.
4. **Full-state tangent.**  Perform the `+/- epsilon` complete-closure check
   described above with a decreasing epsilon ladder.  Require the full VJP to
   converge to the central difference before interpreting any reduced mean
   chain.
5. **Diagram completeness.**  At n=4, explicitly push collision partitions
   `[4]`, `[31]`, `[22]`, `[211]`, and `[1111]`; compare each with the proposed
   local source.  The `[22]` check must expose `2F`, not silently call the
   deployed channel complete.
6. **Scaling only after algebra.**  Use at least n=8,16,32,64 and multiple
   independent weight families.  Report quenched per-network slopes; do not
   infer `1/n` from two coupled widths.

## 7. Post-run artifact review

The seal binds the preregistration and runner and records that `RESULT.json`
did not exist at seal time.  The terminal firewall records zero development
targets, observer/selection artifacts, fits, and submissions.  The result
reproduces the baseline exactly, keeps K4 bitwise identical, and proves the
cheap reverse and forward calculations agree to `1.9244557334329387e-16`.
The source-identity check missed its unusually strict `1e-13` numerical bar by
a small amount (`2.37097e-13`); this is not load-bearing.

| endpoint | baseline MSE | K12 MSE | baseline/K12 | K12 alignment |
|---|---:|---:|---:|---:|
| family1 n8 d4 | 5.80743e5 | 1.94528e18 | 2.98539e-13 | -0.999999 |
| family1 n16 d5 | 2.12314e-3 | 3.10689e-2 | 0.0683365 | 0.095979 |
| family2 n8 d4 | 1.33253e-2 | 9.01991e-1 | 0.0147732 | -0.765430 |
| family2 n16 d5 | 2.62631e-3 | 2.68405e-3 | 0.978486 | 0.153281 |

Family1 n8 is an unmistakable closure-envelope explosion (`g3_rms` reaches
`2.0e19` and `g4_rms` `1.8e27`), so its spectacular failure should not be used
to quantify normal-width performance.  It is not needed for the verdict: K12
also loses on the other three final endpoints, none of their alignments reaches
0.5, K12 is not the best aggregate rung, and both registered width-scaling
diagnostics fail.  K5's isolated best endpoint is only `1.468x`, while its
four-endpoint geometric mean is `0.00439x`; there is no frozen target-free rule
that can select it into a passing family.

The terminal receipt's `CLOSE_RANK_COST_DICHOTOMY` is therefore supported by
multiple independent gates, not by the one unstable small-n row or the
marginal source-identity tolerance.

### Single missing diagram/state advice

The single algebraically proven omission from the deployed K22 residual push
is the crossed pair `2F` term above.  It is the correct answer to "is every
leading K22 loop already represented?": **no**.  It is also already closed as
an accuracy route by direct measurement (`+0.0763%` alone; `-0.5156%` with the
kappa211 planes).  It should not be run again under an adjoint name.

If backward structure is screened again, the first genuinely non-equivalent
**state** is the full retained-state adjoint, especially
`Lambda_l^22 = partial m_D / partial K22_l`, including connected downstream
gate response rather than products of marginal `Phi` gates.  The only safe
first experiment is source-only: compute its exact small-n VJP, singular-value
spectrum, and contraction with the already enumerated collision partitions.
This is a representation-feasibility screen, not evidence of an accuracy
source.  Since `2F` is effect-capped, any accuracy successor must first name a
new born joint-cumulant innovation; this K5--K12 marginal family supplies none.

## Recommendation

- **Close** the literal exact finite reverse-Hermite route as
  `CLOSE_RANK_COST_DICHOTOMY`.
- **Classify** reverse versus forward Duhamel for a fixed source as an identity,
  not a new method.
- **Accept the frozen source-only terminal** `CLOSE_RANK_COST_DICHOTOMY`; do not
  promote it to target access or select a lower rung post hoc.
- **Do not reopen K22 under an unnamed "Onsager" label.**  The genuinely
  missing `2F`/kappa211 plane class is already proven and measured submaterial.
  A successor must name a different missing joint diagram, prove its quenched
  width order, and show a compressible endpoint contraction before any target
  access.

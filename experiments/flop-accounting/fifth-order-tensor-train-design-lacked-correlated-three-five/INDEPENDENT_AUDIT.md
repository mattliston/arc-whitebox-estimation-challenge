# Independent red-team audit: PC-TT5 design receipt

Status: **HOLD — REPAIR THE SPECIFICATION BEFORE D0 OR ANY IMPLEMENTATION**  
Date: 2026-08-26  
Scope: additive, target-free, read-only audit of the design sources  
Audited design: `../p2r0f3b2b_partition_complete_tt5_design_v1/DESIGN_RECEIPT.md`

## Executive verdict

The scientific hypothesis is not disproved: a jointly reset, partition-complete
third-through-fifth cumulant state is genuinely different from B2A's projected
diag/K41/K32 carry and from B1A's Duhamel rewrite. The sentinel embedding also
gives a coherent way to apply an affine map to several tensor orders in one
order-five container.

The current receipt is nevertheless **not an executable mathematical or cost
specification**. It must not advance to its stated D0. The blockers are prior to
the rank-16 scientific question:

1. The multivariate reset-entry oracle is undefined for three through five
   distinct coordinates; univariate `c[p,q]` identities do not provide a
   finite exact evaluator for those entries.
2. D0 is set to `n=4` but claims to test `[11111]` and a nonzero all-distinct
   fifth-order fixture. Both require at least five coordinates.
3. Two proposed parity reductions are false as written: setting only
   `Lambda_5=0` does not recover the order-four reset, and zeroing only the
   three-or-more-distinct K5 sectors does not recover B2A.
4. The unweighted sentinel construction makes total Frobenius/action-energy
   rank gates nearly blind to K3 and K4 at width 1024.
5. Ordinary TT-cross/rounding does not preserve the required permutation
   symmetry, and no symmetry gate or reconstruction rule is specified.
6. The proposed no-eigh host does not certify a positive-semidefinite
   covariance, while the reset is explicitly defined relative to a
   multivariate Gaussian measure.
7. The reported line-item arithmetic adds correctly, but the 8,192-FLOP
   entry cap, the 20B allowance, the integrated host subtraction, and several
   TT-cross operations are not proved. Therefore the 68.634B margin is not a
   strict envelope.
8. The suffix-action rank metric omits future nonlinear resets and is not the
   action that determines endpoint mean error.

Verdict taxonomy:

```text
scientific representation idea       CONDITIONAL SURVIVOR
current reset/oracle specification    FAIL
current D0                            FAIL
current D1                            HOLD / UNDERSPECIFIED
current rank/action gates             FAIL
current production cost envelope      FAIL AS A STRICT BOUND
legal next action                     SPECIFICATION REPAIR ONLY
```

No development target, target value, fit, score, board submission, estimator,
canonical graph, or original design source was opened or modified by this
audit.

## 1. What is mathematically coherent

### 1.1 Sentinel affine embedding

Let an order-`r` cumulant occupy the slices of a five-mode tensor with exactly
`r` active coordinates and `5-r` sentinels. With

```text
Wbar = diag(1, W),
```

modewise application of `Wbar` leaves the sentinel positions fixed and applies
one copy of `W` to every active index. This is a valid direct-sum embedding of
orders three, four, and five. Subject to orientation conventions, D0 can verify
the affine push exactly.

Modewise multiplication preserves existing TT ranks. If `W` is invertible,
the exact unfolding ranks are preserved. This part of the receipt is sound.

### 1.2 Fifth-order collision classes

The listed integer partitions

```text
[5], [41], [32], [311], [221], [2111], [11111]
```

are the complete multiplicity classes of total order five. A complete reset
must also enumerate the lower-order classes explicitly:

```text
order 3: [3], [21], [111]
order 4: [4], [31], [22], [211], [1111].
```

### 1.3 Edgeworth tilt grading

The declared operator shell

```text
I + Lambda3 + Lambda4 + Lambda3^2/2
  + Lambda5 + Lambda3 Lambda4 + Lambda3^3/6
```

is the complete formal Edgeworth shell through **tilt order three** if and only
if `grade(Lambda_r)=r-2`. It is not an exact probability density and it is not
the same thing as merely retaining cumulants whose tensor order is at most
five. The repaired receipt must state the grading and the operator convention,
for example

```text
Lambda_r = (1/r!) sum_{i1,...,ir} kappa_r[i1,...,ir]
           (-partial_{i1}) ... (-partial_{ir}),
```

including raw-versus-standardized units, factorial/multiplicity convention,
sign, and how integration by parts is applied.

## 2. The reset-entry oracle is presently undefined

For an outgoing all-distinct fifth raw moment, even the Gaussian term requires

```text
E_G[ product_{a=1}^5 relu(H_{i_a}) ],
```

for a general correlated five-variate Gaussian. The Edgeworth terms require
the corresponding expectations after mixed derivatives through total order
nine. Univariate `c[p,q] = E[X_+^p He_q(Z)]` values do not determine this in a
finite product formula.

The exact Hermite-diagram identity uses one nonnegative bond count for every
pair of distinct variables. For `s` distinct coordinates there are
`e=s(s-1)/2` edges. A literal total-bond-order cutoff `Q=24` contains

```text
C(Q+e, e)
```

bond configurations:

| distinct coordinates | edges | configurations through Q=24 |
|---:|---:|---:|
| 2 | 1 | 25 |
| 3 | 3 | 2,925 |
| 4 | 6 | 593,775 |
| 5 | 10 | 131,128,140 |

The bivariate Q24 host therefore does not extend to five coordinates by simply
reusing its `c[p,q]` table. Alternatives are possible, but each must be bound:

- a frozen multivariate Hermite/diagram truncation, with a total-order rule and
  a numerical remainder gate;
- a multivariate truncated-Gaussian/orthant routine, with accuracy and all
  special-function FLOPs priced;
- a new factorized analytic identity, with an algebraic proof and an
  independent reference.

No such choice appears in the receipt. “Dense enumeration” of cumulant entries
is not an independent reference for the Gaussian integral; it only enumerates
state values and set partitions.

The 8,192-FLOP cap is consequently not proved. For a generic five-coordinate
query, the linear shells alone can depend on all

```text
C(7,3) + C(8,4) + C(9,5) = 35 + 70 + 126 = 231
```

restricted incoming cumulants. Evaluating those as independent fixed-index
rank-16 TT entries would cost roughly `231 * (3*2*R^2 + 2*R) = 362,208`
counted matmul-style FLOPs before any Gaussian kernel, products, Mobius
conversion, or special functions. A fused contraction could be cheaper, so
this is not an information-theoretic lower bound; it proves that the receipt
needs the promised explicit reduction rather than an ordinary TT-entry loop.

### Required oracle repair

Before D0, freeze:

1. the exact `Lambda_r` convention and tilt grading;
2. the Gaussian marginal evaluator for every `s=1,...,5`;
3. any Hermite/bond cutoff and a reference-error tolerance;
4. a literal counted-operation trace for the worst `[11111]` query;
5. a separately implemented high-precision reference, not the same formula
   with dense state storage.

Failure to meet 8,192 counted FLOPs after all TT contractions, special
functions, connected conversion, and symmetry handling closes the proposed
width-1024 cost envelope.

## 3. Exact D0 contradictions

### 3.1 `n=4` cannot expose `[11111]`

`[11111]` means five distinct coordinate labels. A nonzero all-distinct
order-five fixture also requires five distinct labels. D0 specifies `n=4`, so
both requested checks are impossible. D0 must use `n>=5`; `n=5` is minimal.

### 3.2 `Lambda5=0` does not recover E4

With only `Lambda5` disabled, the declared reset remains

```text
E4 + Lambda3 Lambda4 + Lambda3^3/6,
```

where

```text
E4 = I + Lambda3 + Lambda4 + Lambda3^2/2.
```

Therefore exact K4-level reproduction requires disabling the **entire
tilt-order-three shell**, not merely setting K5 or `Lambda5` to zero.

### 3.3 The stated B2A projection is insufficient

B2A's ledger level D carries projected lower state

```text
K3: diag + K21
K4: diag + K31 + K22
K5: diag + K41 + K32
```

and evaluates bivariate pair resets with its frozen Mehler cutoff. PC-TT5
instead declares complete K3 and K4 sectors and a multivariate reset. Zeroing
only K5 sectors `[311]`, `[221]`, `[2111]`, and `[11111]` leaves those lower
differences intact and cannot reproduce B2A.

Valid B2A parity requires all of the following simultaneously:

1. project K3 to `[3],[21]` and zero `[111]`;
2. project K4 to `[4],[31],[22]` and zero `[211],[1111]`;
3. project K5 to `[5],[41],[32]` and zero the remaining classes;
4. use the same univariate/bivariate reset shell, correlation clipping, and
   Mehler cutoff as B2A;
5. match push, connected conversion, diagonal conventions, and covariance
   handling.

Only then is B2A a valid null reduction.

## 4. Sentinel-sector conditioning invalidates the current rank gate

The augmented tensor contains these ordered slots:

```text
K5 sector:  1 * n^5
K4 sector:  5 * n^4
K3 sector: 10 * n^3.
```

At `n=1024`, if standardized per-entry magnitudes are comparable,

```text
K4/K5 slot mass = 5/n   = 0.0048828125
K3/K5 slot mass = 10/n^2 = 0.0000095367431640625.
```

Thus an unweighted approximation can discard all K3 and K4 entries while
retaining more than 99% of total slot energy. Those lower orders are exactly
the sectors already known to control the mean trajectory. A total-jet `r_99`
or 10% residual is therefore not an action-completeness gate.

The design needs either separate TT blocks or an explicitly preconditioned
sentinel tensor. At minimum it must:

- standardize cumulants by products of marginal scales;
- weight each order independently of its combinatorial slot count;
- report residual and retained action separately for K3, K4, and K5;
- gate every multiplicity class, including all-distinct lower sectors;
- verify that removing the preconditioner reconstructs the physical tensor.

## 5. Permutation symmetry is not preserved by the stated algorithm

The exact cumulant tensors are symmetric and the sentinel permutations must
agree. A generic fixed-order TT representation and ordinary TT-cross/round do
not enforce invariance under the 120 permutations of five modes. Canonical TT
gauge and positive pivots remove gauge ambiguity; they do not impose physical
index-permutation symmetry.

If permutations disagree, the affine push propagates a non-cumulant tensor and
the value assigned to a collision class depends on slot order. A repair must
specify one of:

- a symmetry-preserving tensor-network parameterization;
- an exactly symmetrized evaluator/round with its rank and cost paid;
- a canonical reconstruction rule whose output is proved symmetric.

D2 and D3 must gate maximum permutation discrepancy in both physical and
future-action norms. D0 must check symmetry before compression; the compressed
symmetry check cannot wait until endpoint MSE.

## 6. PSD conflict with the no-eigh host

The bound no-eigh build receipt states that the unconditional eigendecomposition
repair was removed from the main path, while one floor-arm `eigh` remains. The
main path symmetrizes and rescales covariance but does not prove positive
semidefiniteness.

Pairwise clipping of correlations into `[-1,1]` does not make a correlation
matrix, or all of its five-coordinate principal submatrices, positive
semidefinite. PC-TT5 nevertheless defines its reset relative to a multivariate
Gaussian reference. An indefinite covariance means that reference measure does
not exist.

This is also a cost conflict. Restoring the trim-card 14-call spectral repair
would add `270,582,939,648` billed FLOPs and break the proposed envelope. The
remaining data-dependent floor arm is not a fixed zero-cost guarantee, and the
new reset can change whether it triggers.

Before the no-eigh subtraction can be used, the integrated design must provide
a target-free PSD invariant or a separately priced repair. Checking only
diagonal positivity or pairwise correlations is insufficient. Treating an
indefinite covariance as a merely formal diagram parameter would be a different
closure and must not be called a Gaussian reference.

## 7. Rank and action gates need repair

### 7.1 Affine suffix is not future endpoint action

The endpoint mean depends on every later nonlinear ReLU reset. Applying only
the remaining affine matrices measures a Kronecker suffix action on the tensor,
not the Frechet action of the autonomous closure trajectory. Components small
under an affine suffix can be injected into lower cumulants and amplified by a
later reset.

The action metric must include the Jacobian of the complete future reset/push
map, or use direct paired dense-versus-compressed endpoint effects. It must be
reported per order and collision sector.

### 7.2 TT-cross residual is not automatic

TT-cross is an interpolation algorithm; it does not minimize Frobenius or the
declared suffix-action norm. A posterior residual is meaningful only if it is
computed against an independently available dense/matrix-free reference with
a frozen witness design. Sampled entry checks do not certify a global SVD or
action norm.

At width 32 a dense order-five tensor has `32^5 = 33,554,432` entries, so an
exact or tightly bounded reference is possible in principle. The receipt must
freeze how matrix-free products are computed and certified, how singular-value
ties are handled, and how the TT-cross residual is evaluated independently.

### 7.3 Width transfer remains conjectural

Widths 8, 16, and 32 with three families can falsify obvious rank growth, but
the slope gate cannot certify rank 16 at width 1024. Passing it may license only
a target-free production-shape rank witness, not a Phase-2 estimator. This is
consistent with the receipt's conditional-science framing once the action norm
is repaired.

## 8. Cost audit

### 8.1 Arithmetic identities that pass

The printed scalar-slot count is correct:

```text
(1025) * (3*16^2 + 2*16) = 820,000.
```

The printed cross-query count follows its declared conservative formula:

```text
2 * 15 * 5 * 1025 * 16^2 = 39,360,000.
```

At the hypothetical cap:

```text
39,360,000 * 8192 = 322,437,120,000.
```

The four increment rows add correctly:

```text
25,202,700,000
+ 322,437,120,000
+  10,076,160,000
+  20,000,000,000
= 377,715,980,000.
```

The historical subtraction and final addition also add correctly:

```text
2,023,256,192,375 - 270,582,939,648 = 1,752,673,252,727
1,752,673,252,727 + 377,715,980,000 = 2,130,389,232,727
2^41 - 2,130,389,232,727 = 68,634,022,825.
```

### 8.2 Why the envelope does not pass

- The dominant 322.437B row is conditional on the undefined/unproved entry
  evaluator.
- The 20B “collision reinstall, connected conversion, checking, output
  allowance” is a reserve, not an operation ledger.
- The exact 25.203B affine row is not derived in the receipt from primitive
  shapes and counted operations.
- TT-cross pivot search/max-volume work, validation witnesses, symmetry
  enforcement, preprocessing/caching, TT round/SVD, multivariate special
  functions, and PSD enforcement are not fully bound.
- The host subtraction comes from a historical trim-card meter plus a build
  diff, not a literal meter of the integrated PC-TT5 state machine. The
  no-eigh build retains a data-dependent eigensolve arm.
- Replacing projected K3/K4 state with a complete TT jet changes the host's
  reset and covariance path. A removed operation from the old path is not by
  itself an upper bound on the new integrated path.

The line arithmetic is therefore a useful aspiration, not a strict Phase-2
admission bound.

## 9. D0/D1 are not yet minimal valid fail-fast stages

### Repaired D0 sequence

1. **D0a — scalar and bivariate operator parity.** Freeze the operator
   normalization. Prove E4 parity by disabling the entire tilt-order-three
   shell. Prove exact B2A projection parity using all lower-order projections
   and the same Mehler cutoff.
2. **D0b — multivariate oracle and cost.** On analytic 3D, 4D, and 5D fixtures,
   compare the entry oracle to an independent high-precision reference. Bind
   truncation error and a literal worst-entry operation count at or below
   8,192.
3. **D0c — dense partition/symmetry check at `n=5`.** Exercise every order-3,
   order-4, and order-5 collision class and all sentinel permutations. Check
   Mobius conversion, affine orientation, raw/standardized units, and the
   PSD invariant.
4. **D0d — cheap sector-conditioned rank witness.** At width 8, report exact
   per-order/per-class unfolding tails before spending on multi-seed autonomous
   references. This is a fail-fast on the stated rank-16 premise.

### Repaired D1 sequence

1. Run one reset and then two resets on the first width-8 seed; require finite
   covariance, per-order state accuracy, and improvement before a full-depth
   recursion.
2. Run the complete dense width-8 case.
3. Only then run width 16, one frozen seed at a time.
4. Freeze antithetic sample count, batch order, reference precision, MC standard
   errors, and confidence-aware ratio gates. “Exact/synthetic reference” and
   “antithetic reference” are not interchangeable.
5. Define `level-C MSE / dense-jet MSE >= 4` and the intermediate-layer `2x`
   rule algebraically for each layer, including denominator/noise floors.
6. Report K3, K4, and K5 state errors separately. Endpoint improvement alone
   cannot establish that the complete state or cancellation mechanism is
   correct.

D1 dense efficacy should precede expensive width-transfer claims, but a cheap
per-sector rank witness belongs before the full three-family D1/D2 ladder.

## 10. Promotion rule after repair

Only all of the following would promote the design back to
`TARGET_FREE_FAIL_FAST_READY`:

```text
R0 operator convention and tilt grading frozen
R1 independent 1D..5D reset-oracle parity passed
R2 worst [11111] literal entry cost <= 8192 including all work
R3 n=5 all-partition and symmetry gate passed
R4 exact E4 and exact B2A projection nulls passed
R5 covariance PSD invariant proved and priced
R6 sector-balanced, per-order TT rank/action gates passed
R7 future action includes nonlinear resets or direct endpoint action
R8 integrated primitive FLOP ledger and literal meter fit below 2^41
R9 sequential D1 reference/noise gates frozen
```

Until then, implementing the proposed TT-cross state or running D0 would test
an underdefined oracle and could not yield an interpretable pass/fail result.


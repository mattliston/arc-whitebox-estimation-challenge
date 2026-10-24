# B2B2 independent-Gaussian linked-cluster grade-3 constraint receipt

Status: **CLOSE IG-LC3 AS AN EXACT OR PRODUCTION B2B2 EVALUATOR**  
Date: 2026-08-26  
Scope: target-free derivation and static arithmetic audit only; no experiment,
target, estimator, graph, submission, or network action

## Decision

Expanding around a diagonal independent-Gaussian reference does remove the
undefined multivariate Gaussian integral. The resulting total-grade-three
operator is finite, algebraically explicit, and permutation-equivariant.
However, it is only a Taylor projection of the correlated-base `E5` reset. It
is not collision-complete for fifth-order carry, is not B2A-equivalent, is not
accurate on the supplied reached-correlation anchor `|rho| approximately 0.54`,
and has no credible `<=8,192`-FLOP circuit when its rank-16 carrier reads are
counted.

Therefore this exact spelling, called `IG-LC3`, should close before D0. The
broader multivariate reset-oracle question may remain open only through a
covariance-resummed or independently tail-certified representation with a
fused carrier contraction.

## 1. Exact formal derivation

Let `G0` have the requested pre-activation mean and diagonal variance but
independent coordinates. Let `DeltaSigma` contain the off-diagonal covariance.
On expectations, define commuting constant-coefficient differential operators

```text
Lambda2 = sum_{i<j} DeltaSigma_ij partial_i partial_j
Lambda_r = (1/r!) sum_{i_1,...,i_r}
             kappa_r[i_1,...,i_r] partial_i1 ... partial_ir,  r=3,4,5.
```

The full correlated-Gaussian `E5` spelling is formally

```text
H0 = I
H1 = Lambda3
H2 = Lambda4 + (1/2)Lambda3^2
H3 = Lambda5 + Lambda3 Lambda4 + (1/6)Lambda3^3

R_E5 = exp(Lambda2) (H0 + H1 + H2 + H3).
```

Give `Lambda2` and `Lambda3` grade one, `Lambda4` grade two, and
`Lambda5` grade three. The complete total-grade-three projection is

```text
P3 = sum_{t=0}^3 [ sum_{m=0}^{3-t} Lambda2^m/m! ] H_t.                 (1)
```

Equivalently, with

```text
A = Lambda2 + Lambda3,  B = Lambda4,  C = Lambda5,
```

equation (1) is

```text
P3 = I + A + (B + A^2/2) + (C + A B + A^3/6).                         (2)
```

This proves that the proposed finite formula is internally complete at its
declared grade. It also gives its exact limitation:

```text
R_E5 - P3
 = sum_{t=0}^3 [ sum_{m=4-t}^infinity Lambda2^m/m! ] H_t.             (3)
```

The leading missing shells are `Lambda2^4/24`,
`Lambda2^3 Lambda3/6`, `Lambda2^2 H2/2`, and `Lambda2 H3`.

Because every input tensor is symmetric, the operators commute, and the
independent base factorizes, (1) is permutation-equivariant in exact
arithmetic. A canonical sorted-index evaluator can make the entry map
deterministically symmetric. This proof applies to the evaluator, not to an
unconstrained rank-truncated TT carrier.

## 2. Collision-sector audit

For a pure correlated-Gaussian reset (`kappa3=kappa4=kappa5=0`), a connected
output cumulant on `k` distinct coordinate vertices first needs a spanning
tree of `k-1` off-diagonal `Lambda2` bonds. Therefore:

| carried order | collision class | distinct coordinates | first pure-Gaussian bond grade | present in `P3`? |
|---|---|---:|---:|---|
| K3 | `[3]` | 1 | 0 | yes |
| K3 | `[21]` | 2 | 1 | yes |
| K3 | `[111]` | 3 | 2 | yes |
| K4 | `[4]` | 1 | 0 | yes |
| K4 | `[31]`, `[22]` | 2 | 1 | yes |
| K4 | `[211]` | 3 | 2 | yes |
| K4 | `[1111]` | 4 | 3 | yes |
| K5 | `[5]` | 1 | 0 | yes |
| K5 | `[41]`, `[32]` | 2 | 1 | yes |
| K5 | `[311]`, `[221]` | 3 | 2 | yes |
| K5 | `[2111]` | 4 | 3 | yes |
| K5 | `[11111]` | 5 | **4** | **no** |

`P3` can return a nonzero `[11111]` entry when incoming all-distinct
`kappa5`, overlapping `kappa3` hyperedges, or mixed higher-cumulant hyperedges
are nonzero. That is not collision completeness: on the exact Gaussian null
fixture it suppresses the leading generated `[11111]` term identically.

Thus the formula covers the *address space* of every K3/K4/K5 class but not the
required reset physics of every class. This distinction is load-bearing after
B2A's partial-state instability.

## 3. Two analytic counterexamples

### 3.1 Pair/B2A counterexample

Let `X_i = relu(Z_i)` for standard jointly Gaussian `(Z_1,Z_2)` with positive
correlation `rho`, and set all higher input cumulants to zero. At zero marginal
mean, the Hermite coefficients are

```text
c1[1] = 1/2,  c1[2] = phi(0),  c1[3] = 0,  c1[4] = -phi(0).
```

The exact covariance is

```text
Cov(X1,X2)
 = [sqrt(1-rho^2) + (pi-acos(rho))rho - 1] / (2 pi)
 = sum_{q>=1} c1[q]^2 rho^q/q!.
```

`P3` stops after `q=3`. Its first omitted term is strictly positive:

```text
Delta4 = rho^4/(48 pi).                                                 (4)
```

At the supplied reached-domain anchor `rho=0.54`,

```text
Delta4                         = 5.638764140779998e-4
exact tail after grade 3       = 6.208722781950449e-4
tail after covariance order 12 = 4.463166911050642e-8
tail after covariance order 14 = 9.266690109699383e-9.
```

These are evaluations of the displayed closed form and Hermite series, not
network metrics. They show that grade three is not a meaningful covariance
truncation on the reached domain; even this easiest pair control needs about
fourteen covariance orders for a `1e-8` raw-moment tolerance.

### 3.2 Missing all-distinct fifth tree

For five zero-mean, unit-scale ReLU marginals with common small positive
correlation `rho` and no higher input cumulants, the omitted grade-four
`[11111]` spanning-tree shell is

```text
T4 = rho^4 [5 A4 A1^4 + 60 A2^3 A1^2 + 60 A3 A2 A1^3],
A1=1/2, A2=phi(0), A3=0, A4=-phi(0),

T4 = 0.827735076388167 * rho^4.                                        (5)
```

The multiplicities are the 5 labeled stars, 60 labeled paths, and 60 labeled
degree-`(3,2,1,1,1)` trees. `P3` returns zero on this fixture because at most
three covariance bonds cannot connect five vertices. At `rho=0.54`, the
single missing shell in (5) is `0.07038277707692864`. Higher shells may partly
cancel it, so this number is a shell coefficient rather than a claim about the
full cumulant; near zero it proves relative error tending to one.

A necessary shell gate for absolute `[11111]` entry tolerance `epsilon` is

```text
|rho| <= (epsilon / 0.827735076388167)^(1/4).
```

It is `0.0104840` for `epsilon=1e-8` and `0.00331533` for `epsilon=1e-10`, far
below the supplied reached-domain anchor.

## 4. Exact relation to B2A and the whole-shell gate

`ledger_rig.py` level D uses the bivariate correlated Gaussian/Mehler base and
the same `H0+H1+H2+H3` higher-cumulant shell, with covariance powers retained
through its frozen pair cutoff. Consequently `P3` is exactly the
**total-grade-three Taylor projection** of the formal B2A operator, not B2A.
Their difference starts with equation (3), and the positive pair control (4)
proves non-parity even when every higher cumulant is zero.

Using exact B2A only for tuples with at most two distinct coordinates while
using `P3` for three to five coordinates does not repair this. Raw moments of
overlapping marginals would then come from different formal measures, and the
set-partition connected conversion would mix inconsistent truncations. It
would also violate the intended single common reset law.

The correct K4-level ablation remains removal of all of `H3`, not merely
`Lambda5`. This correction is compatible with (1) but does not restore the
omitted covariance dressing.

## 5. Term counts and circuit constraint

For five selected coordinates, the symmetric operator has

```text
Lambda2 off-diagonal types = C(5,2) = 10
Lambda3 types              = C(7,4) = 35
A types                                   45
B=Lambda4 types             = C(8,4) = 70
C=Lambda5 types             = C(9,4) = 126.
```

Before connected cancellations, equation (2) contains

```text
1 + 45 + 70 + C(46,2) + 126 + 45*70 + C(47,3)
= 20,642 generic operator monomials.
```

For the all-distinct five-slot cumulant, deleting monomials whose hyperedge
support is disconnected leaves exactly:

| shell | connected monomials |
|---|---:|
| `A^2/2` | 15 |
| `C` | 1 |
| `A B` | 180 |
| `A^3/6` | 2,365 |
| **total** | **2,561** |

The count is obtained by enumerating the 45/70/126 symmetric multisets above,
taking unordered pairs/triples for powers of `A`, and retaining only support
hypergraphs connected on all five labeled vertices. Repeated-slot collision
classes additionally need local Mobius/Leonov-Shiryaev decorations; 2,561 is
therefore not a full worst-case FLOP count.

Before connected cancellation, the reachable five-coordinate channel set
through (2) has 1,987 entries:

```text
total derivative degree 0,2,3,4,5,6,7,8,9
reachable channel count  1,10,35,70,126,210,330,490,715.
```

After connected-support cancellation and accumulated-incidence aggregation,
the all-distinct sector has 5 grade-two and 126 grade-three response bins. The
independent static derivation in `../p2r0f3b2b2_ig_lc3_reset_derivation_v1/`
gives an optimistic algebra-only subtotal of 6,808 operations **assuming all
primitive cumulant values are already in registers**. That subtotal excludes
the collision-specific formal-log compiler, index work, symmetry, and carrier
extraction, so it is a useful plausibility ceiling rather than a proved
executable circuit. Direct cubic convolution and ungrouped connected-sum
circuits are larger. This is a constraint on the known circuits, not a general
arithmetic-circuit lower-bound theorem.

### Carrier reads already break the cap

A five-coordinate query needs the restricted symmetric state values

```text
kappa3: 35,  kappa4: 70,  kappa5: 126;  total TT values: 231.
```

For ordinary order-five TT cores with rank 16, one selected entry costs 1,519
literal scalar multiplies/adds, or 1,568 under the billed `2R^2` matmul
convention:

```text
3 * [R^2 multiplies + R(R-1) adds] + [R multiplies + (R-1) adds] = 1,519
3 * (2R^2) + 2R = 1,568 billed FLOPs.
```

Reading the 231 values separately therefore costs `350,889` literal operations
or `362,208` billed FLOPs before one `P3` term. Even shared-prefix contraction
is `107,744` billed FLOPs, and the optimistic varying-fiber lower spelling is
`61,440`, as independently enumerated in the companion derivation. A fused
contraction would have to combine three nonlinear copies of the cumulant
carrier for `A^3`; the obvious tensor-network contraction grows in powers of
rank and no sub-8,192 circuit is derived. Merely saying "fuse the reads" is not
a cost proof.

This directly invalidates the old cost table's assumption that the analytic
entry map, including its incoming state, fits under 8,192 FLOPs.

## 6. Source-domain truncation rule

An honest independent-base truncation must specify all of the following before
an accuracy run:

1. a source-only certified domain for every selected correlation submatrix,
   marginal standardized mean/scale, and carried cumulant norm;
2. an entrywise tail tolerance derived from autonomous state stability and the
   final raw-MSE objective;
3. a rigorous majorant for every remainder in (3), propagated through connected
   conversion; and
4. an explicit out-of-domain law that does not silently clip covariance or mix
   incompatible per-sector operators.

The analytic controls above are necessary domain gates. In particular, a
domain containing `rho=0.54` fails grade three by orders of magnitude. A
geometric assertion such as "the next shell should be smaller" is insufficient:
the first omitted pair shell is known exactly and is already `5.64e-4`.

The pair control indicates covariance order roughly 14 just to reach a
`1e-8` raw pair-moment tail at `rho=0.54`; the five-coordinate connected series
needs its own bound and cannot inherit the bivariate number. Such a bound would
restore the high-dimensional summation/cost problem that `IG-LC3` was intended
to avoid.

## 7. Verdict and reopen conditions

Proof-level positives:

- (1)--(2) are the unique complete total-grade-three independent-base operator;
- the finite operator is symmetric and target-free;
- every K3/K4 class and every K5 class through four distinct coordinates has
  its leading pure-Gaussian connectivity tree by grade three.

Terminal constraints:

- the pure-Gaussian `[11111]` fifth sector starts at grade four and is absent;
- `P3` is not exact B2A or correlated-base `E5`;
- reached correlations around `0.54` invalidate the grade-three accuracy
  premise analytically; and
- known evaluator and carrier-read circuits exceed the 8,192-FLOP cap.

Close `IG-LC3` as an exact/production B2B2 spelling. Reopen the broader B2B2
oracle only with a materially different receipt that supplies both:

1. covariance resummation or a certified multivariate covariance tail on the
   reached source domain, including at least the `[11111]` tree shell; and
2. a fused state representation/evaluator whose literal worst-entry count,
   including nonlinear carrier contractions and symmetry, is `<=8,192` (or a
   fully recomputed production design that remains below `2^41`).

No D0 command is authorized by this receipt.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| `public_exact_ledger_v1/ledger_rig.py` | `c74c92862a3afad7e44a077c80f87fdd09c66be4f02718db24a21ea7b5f4a581` |
| `public_trivariate_derivation_v1/DERIVATION.md` | `5fec258709489fc5c36361bb6bb5fb3e0b8f780502d4ef9b48132bf040bb9444` |
| `public_trivariate_derivation_v1/L5_DERIVATION.md` | `6c0dfdc022799953f8bd0f0372a7dbfdf8dbf9eeaf8649cb03f03600c49fee92` |
| `estimator_design/SCALING_AND_GATES.json` | `a552c864510c6fa90a29c4b95a1b53929627eda09fd3f0e44b8d9dc472b6ed31` |
| `p2r0f3b2b2_ig_lc3_reset_derivation_v1/DERIVATION.md` | `7c922b58d0baae6a90abd2edce1f19f03b5f5c0f219561630307ee07697332c1` |
| `p2r0f3b2b2_ig_lc3_reset_derivation_v1/enumerate_hypergraphs.py` | `25cb168fc2678dca4497fb1e326e06b02ca08c3b9fbb47006b2d4d592eeaf540` |

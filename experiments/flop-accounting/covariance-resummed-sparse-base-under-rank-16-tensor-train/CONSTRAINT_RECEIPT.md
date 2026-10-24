# B2B3 covariance-resummed sparse-base constraint receipt

Status: **TERMINAL UNDER THE CURRENT TT16 CARRIER; EXTERNAL PROOF/CIRCUIT
CONSTRAINT ONLY**  
Date: 2026-08-26  
Scope: target-free derivation, existing-source census, and static arithmetic
audit only; no estimator, target, graph, submission, or network action

## Decision

A mathematically valid strong-bond reset can be defined by a Gaussian Markov
forest.  Selected correlations are not inserted into an otherwise diagonal
matrix; instead, each selected tree edge is used as a Gaussian conditional,
which makes every within-tree covariance the product of its path correlations.
The remaining covariance is then a weak operator around that non-diagonal
base.  This genuinely resums covariance and is not the closed independent-base
`IG-LC3` spelling.

The representation does not survive the production gate.  At Q24, the
ordinary base contraction fits the approximately `9,935`-FLOP/query ceiling
only for a degree-two path-like base.  More decisively, covariance resummation
does not remove the incoming K3/K4/K5 state: the already-derived optimistic
TT16 carrier spelling costs at least `61,440` FLOPs/query before any base
moment or linked response work.  A path base alone costs about `3,900` more.

No existing source artifact measures either the path/tree residual or a
rank-one/two spectral-base residual, and an entrywise residual maximum alone
is not a multivariate tail theorem.  Therefore no D0 or source-capture run is
authorized.  The exact spelling closes under the current carrier.  Reopening
requires the single external certificate `CRX0` in section 9: a common-base
remainder theorem and a fused base-plus-carrier circuit below the integrated
ceiling.

## 1. Existing source domain and deduplication

`SOURCE_DOMAIN_SUMMARY.json` is a deterministic read-only reduction of the
already sealed aug7 target-free channel 23.  Its reproducer verifies all 688
row file hashes, parity, and firewalls before reading the arrays.  Across
`688*16*1024` source cells, the maximum absolute off-diagonal correlation per
row has:

| statistic | value |
|---|---:|
| median | `0.2705558240` |
| q90 | `0.3666554034` |
| q99 | `0.4397597989` |
| q999 | `0.4984088838` |
| maximum | `0.6479182839` |
| fraction above `0.2` | `0.7503767679` |
| fraction above `0.3` | `0.3648946895` |
| fraction above `0.4` | `0.0384571164` |

Layerwise q99 rises from `0.1375218034` at L0 to `0.4973327038` at L15.
Thus the non-small correlation domain is population-wide, not explained by a
few near-coincident pairs.

The older coincident-bond census reports zero pairs above `0.9` on ten rows.
That is consistent but not useful here: its lowest threshold is `0.9`, while
the live domain is concentrated roughly between `0.2` and `0.5`.  Channel 24
of the aug7 tape is likewise identically zero because it uses the same `0.9`
threshold.  Neither artifact records degree at a useful threshold, cycles,
treewidth, spanning-tree distortion, or a spectral low-rank residual.

A repository text audit found no prior Chow-Liu, maximum-spanning-tree,
Gaussian Markov forest, or weak-remainder covariance reset.  The nearest
low-rank artifacts are non-equivalent:

- `q24_cs3b_hybrid_exact_lowrank_pairing_covariance_design_v1` is an
  invalid-coordinate Wick transport correction, not a Gaussian base;
- `public_full688_rank16_cross_unit_covariance_cv_v1` is a learned observer
  state, not an autonomous reset law; and
- `public_quadrature_derivation_v1` studies input quadrature/Hermite retention,
  not conditional latent-Gaussian K3--K5 moments.

## 2. The only explicit PSD strong-bond construction

Let `R` be the standardized source correlation matrix.  Select a deterministic
forest `T` with edge labels `r_e=R_ij`, `|r_e|<1`.  For a root in each component,
define

```text
Y_root ~ N(0,1),
Y_v = r_(parent(v),v) Y_parent(v) + sqrt(1-r_e^2) epsilon_v,
epsilon_v iid N(0,1).
```

The resulting correlation base `B_T` is

```text
(B_T)_ii = 1,
(B_T)_ij = product of r_e along the unique T path i--j
             if i,j are in one component,
           = 0 otherwise.                                      (1)
```

Equation (1) proves that `B_T` is PSD, unit-diagonal, and exact on every
selected edge.  Define the weak remainder

```text
W = R - B_T,   diag(W)=0.                                      (2)
```

Simply setting selected covariance entries in `I+S` and all other entries to
zero is not an equivalent construction and need not be PSD.  For example, a
three-leaf star with edge correlation `a` has minimum eigenvalue
`1-sqrt(3)|a|`, negative for `|a|>1/sqrt(3)`.  Threshold, degree, and acyclicity
alone therefore do not license the naive matrix spelling.

A general low-treewidth covariance graph is also underspecified.  Sparse
covariance is not sparse precision; a valid chordal/junction-tree base would
need explicit consistent clique covariances.  The Markov forest above is the
only frozen construction audited here.

One deterministic degree-two version is a greedy maximum-weight spanning
path: sort all edges by `(-|R_ij|, i, j)`, accept an edge iff it joins two
different components and both endpoint degrees remain below two, and continue
until `n-1` edges are accepted.  Every intermediate component is a path, and
the complete graph always contains an endpoint-to-endpoint edge, so the rule
terminates in a spanning path.  It is deterministic and polynomial-time; it
is not claimed to solve the NP-hard maximum-weight Hamiltonian-path problem.

## 3. Exact operator and relation to IG-LC3/E5

Let expectation under the Gaussian base `(mu,B_T)` be `E_B`.  Define

```text
LambdaW = sum_(i<j) W_ij partial_i partial_j,
Lambda_r = (1/r!) sum_(i_1,...,i_r)
             kappa_r[i_1,...,i_r] partial_i1 ... partial_ir.

H0 = I
H1 = Lambda3
H2 = Lambda4 + Lambda3^2/2
H3 = Lambda5 + Lambda3 Lambda4 + Lambda3^3/6.
```

The formal correlated E5 reset is

```text
R_E5 = exp(LambdaW) (H0+H1+H2+H3) E_B.                         (3)
```

With `LambdaW,Lambda3` grade one, `Lambda4` grade two, and `Lambda5`
grade three, the strong-base grade-three projection is

```text
P3_B = sum_(t=0)^3 [sum_(m=0)^(3-t) LambdaW^m/m!] H_t E_B.      (4)
```

All powers of the selected tree covariance are inside `E_B`; only `W` is
truncated.  If `B=I`, (4) is exactly the closed `IG-LC3` operator.  If `W=0`,
it is the correlated-base E5 shell.  In general the exact omitted operator is

```text
R_E5 - P3_B
 = sum_(t=0)^3 [sum_(m=4-t)^infinity LambdaW^m/m!] H_t E_B.     (5)
```

For a selected two-coordinate tree edge, `W_ij=0`; the bivariate covariance
dressing is therefore the existing correlated-base/Mehler law, up to its
frozen Q24 numerical truncation.  Nonselected pairs are not B2A-equivalent.

## 4. Collision completeness, especially `[11111]`

For a queried set `S` of distinct coordinates, let `c_T(S)` be the number of
base-forest components intersected by `S`.  A pure-Gaussian connected output
on `S` needs at least `c_T(S)-1` weak `LambdaW` bonds, because strong-base
dependence cannot connect different components.  Therefore a grade-three weak
expansion covers the leading pure-Gaussian connectivity shell iff

```text
c_T(S) <= 4.                                                    (6)
```

For K3 and K4, at most three or four distinct vertices occur, so grade three
can connect an entirely factorized base.  For K5 `[11111]`, a five-component
tuple requires four weak bonds and is omitted exactly as in IG-LC3.

Consequences:

- a connected spanning tree/path gives `c_T(S)=1` and formally supplies every
  K3/K4/K5 collision sector, including pure-Gaussian `[11111]`;
- a forest with at most four global components also meets the necessary
  connectivity condition; and
- a threshold forest with five or more components is not collision-complete
  for arbitrary K5 queries, even if every selected edge is evaluated exactly.

Direct `Lambda5`, mixed, or incoming nonzero K5 terms do not repair a missing
pure-Gaussian null sector.  Equation (6) is a structural proof, not an
accuracy guarantee.

## 5. Accuracy conditions: what is proved and what is missing

Define

```text
eta_inf = max_(i!=j) |W_ij|,
s = || B^(-1/2) W B^(-1/2) ||_2.                              (7)
```

The Loewner condition `-s B <= W <= s B`, with `s<1`, passes to every
principal submatrix, so it is a useful source-only uniform domain condition
for all K3--K5 tuples.  An honest tail theorem for (5) also needs:

1. a lower bound on every relevant `B_S` eigenvalue;
2. reached standardized mean and scale bounds;
3. norms for incoming K3, K4, and K5;
4. a bound for every `H_t` response under the base; and
5. propagation of the entry remainder through connected conversion and the
   autonomous depth-16 stability map.

`eta_inf` alone is not sufficient: ten weak pair coordinates exist on a
five-tuple, their diagrams combine, and a nearly singular base amplifies
covariance derivatives.

Two exact controls quantify the necessary scale without pretending to be a
full theorem.  For zero-mean unit ReLU pairs,

```text
K(rho) = [sqrt(1-rho^2) + (pi-acos(rho))rho] / (2 pi),
K''''(rho) = (1+2rho^2) / [2 pi (1-rho^2)^(5/2)].
```

If `|B_ij|+|W_ij|<=0.7`, the third-order Taylor remainder is at most

```text
0.07068847947 * |W_ij|^4.
```

This bivariate control is below `1e-8` when `|W_ij|<=0.0193938`.
Independently, the first omitted all-distinct five-ReLU tree shell around an
independent base is `0.8277350764*eta^4`, whose `1e-8` crossing is
`eta=0.0104840`.  These are proof-level scalar fixtures.  They are not a
common-base K5 tail theorem, and no production claim may promote them as one.

The current repository has neither `eta_inf` nor `s` for a Markov path/tree.
It also has no useful strong-bond degree census below `0.9`.  That missing
source structure is an external constraint, not permission to infer that a
sparse base is accurate.

## 6. Q24 treewidth/degree cost

Let `Q=25` denote Hermite orders `0..24`.  A Gaussian Markov tree density is a
product of bivariate Mehler edge kernels.  After local univariate response
tables are precomputed, an induced five-coordinate path can be contracted as
three `Q x Q` matrix-vector products, four diagonal edge scalings, and one end
dot product:

```text
3*(2 Q^2) + 4 Q + 2 Q = 3,900 billed scalar FLOPs.              (8)
```

Equation (8) is for one Gaussian base raw moment only.  It excludes connected
conversion, weak/higher-cumulant response algebra, carrier extraction,
symmetry, and table construction.

At one degree-three tree node, the local triple Hermite integral is nonzero
for exactly `4,147` triples `(p,q,r)` in `0..24` satisfying even total degree
and the three triangle inequalities.  The ordinary direct contraction needs
two multiplications plus an accumulation per term, about `12,441` operations,
already above `9,935`.  Degree four is larger.  This is an explicit circuit
witness, not a universal arithmetic lower bound; it makes maximum selected
degree two the only credible ordinary Q24 tree spelling.

Likewise a width-two junction-tree clique carries `Q^3=15,625` order states
before response work.  A claim that low treewidth alone is cheap is therefore
false at Q24; the actual clique circuit must be priced.

## 7. Integrated carrier cost is terminal

The covariance base does not eliminate the restricted incoming state for one
five-coordinate query:

```text
K3 values C(7,4) = 35
K4 values C(8,4) = 70
K5 values C(9,4) = 126
total                    231.
```

The prior independently enumerated TT16 costs are:

| ordinary carrier spelling | billed FLOPs/query |
|---|---:|
| separate scalar TT entries | `362,208` |
| shared-prefix trie | `107,744` |
| optimistic varying-fiber spelling | `61,440` |

The full design has only about `1,743` FLOPs/query beyond its original
`8,192` claim, so the absolute ceiling is approximately `9,935` even if all
other unpriced work is free.  The most favorable known carrier is therefore
`6.18x` over the absolute ceiling before (8) or one linked response term.

Strong-bond resummation changes response kernels, not this state dependence.
It cannot inherit the independent-base algebra-only `6,808` subtotal while
silently dropping carrier reads.  The current TT16 representation is a
terminal cost blocker for (4).

## 8. Diagonal-plus-low-rank Gaussian alternative

A globally consistent dense alternative was audited before closure.  Let the
repaired correlation eigensystem provide the top `r` factors and define

```text
B_r = D + A A^T,
D_i = 1 - ||A_i||^2 >= 0,
W_r = R - B_r.                                                  (9)
```

Because the host already computes a full eigensystem during covariance
repair, `A` can in principle reuse those eigenpairs; after the host's diagonal
renormalization, factor rows receive the same renormalizer.  No second
eigendecomposition is mathematically necessary.

Conditioned on `g~N(0,I_r)`, coordinates in (9) are independent:

```text
Z_i = mu_i + A_i g + sqrt(D_i) epsilon_i.
```

Thus a five-coordinate Gaussian raw moment is an `r`-dimensional integral of
five univariate conditional moments.  With a Q24 tensor-product rule, the
node count is `25^r`, and even with all univariate tables precomputed the
five-factor product/sum costs at least `5*25^r-1` operations:

| rank | nodes | raw five-factor subtotal |
|---:|---:|---:|
| 1 | 25 | 124 |
| 2 | 625 | 3,124 |
| 4 | 390,625 | 1,953,124 |

Rank one or two could fit the Gaussian-base subtotal.  Rank four cannot.  A
rank-four product rule small enough for `9,935` would be at most six nodes per
axis (`6^4=1,296`), but no source-domain error theorem for that rule exists,
especially when some `D_i` approach zero and conditional ReLU responses lose
smoothness.  Sparse cubature would need a new certified rule, not an analogy
to the closed quadrature family.

More importantly, (9) still needs the same 231 K3--K5 carrier values and the
same nonlinear E5 contractions.  It therefore fails the integrated cost gate
even at rank one.  No existing artifact records `max|W_r|`, the generalized
residual norm, `min D_i`, or rank-one/two spectral capture.  The low-rank base
does not rescue the current successor and is not opened as a second arm.

## 9. Sole reopen constraint `CRX0`

There is no licensed metric experiment.  The exact branch may reopen only
with one proof-and-circuit artifact, `CRX0`, satisfying every item below for
one base formula frozen in advance (no tree/low-rank sweep):

1. **Common base:** prove PSD, unit diagonal, query-slot symmetry, and one
   globally consistent marginal law for all K3/K4/K5 collision sectors.
2. **Source domain:** from a parity-preserving target-free capture, bind
   `eta_inf`, a Loewner/generalized residual bound `s<1`, base conditioning,
   reached mean/scale, and incoming K3--K5 norms.  Existing row maxima are not
   a substitute.
3. **Remainder theorem:** derive a numerical majorant for all terms in (5),
   including `[11111]`, connected conversion, Q24/base quadrature tail, and
   depth-16 propagation.  Its tolerance must be derived from the raw
   `<1e-8` objective rather than chosen post hoc.
4. **Integrated circuit:** give a literal cold-cache worst `[11111]` circuit,
   including fused incoming carrier, base contraction, every H3 class,
   symmetry, normalization, and output assembly, at `<=9,935` counted FLOPs.
   Amortization is allowed only with the complete layer cost divided by the
   exact query count and all cache construction priced.
5. **Dense parity:** prove all K3/K4/K5 collision sectors against dense `n=5`
   fixtures to relative error `<=1e-12` before any target or development
   metric.

The known TT16 implementations fail item 4 analytically.  `CRX0` therefore
requires a materially new fused carrier identity or state representation, not
an implementation optimization around the present evaluator.

## 10. Graph-ready recommendation

Suggested status only; this receipt does not edit the canonical graph:

```text
P2R0F3B2B3  covariance-resummed sparse/low-treewidth E5 reset
status: closed_design_under_current_TT16
reason: valid Markov-base algebra, but no source remainder theorem and
        known integrated carrier floor 61,440 > 9,935

P2R0F3B2B3X CRX0 common-base remainder + fused-carrier certificate
status: blocked_external_constraint
dependents: base accuracy, dense stability, full cost, development metric
```

Do not create separate path, treewidth, and low-rank metric children.  They
share the same unresolved remainder theorem and the same terminal carrier
floor.

## Evidence hashes

The complete read-only verifier is:

```bash
python pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b3_covariance_resummed_sparse_base_v1/verify_static.py
```

It reproduces the sealed source census, verifies bound arithmetic and carrier
counts, checks the bound evidence hashes, and returns
`PASS_STATIC_CONSTRAINT_VERIFICATION` without loading a target or estimator.

| artifact | SHA-256 |
|---|---|
| `p2r0f3b2b3_covariance_resummed_sparse_base_v1/SOURCE_DOMAIN_SUMMARY.json` | `e6543e29d688669a5ad7c0f13d52a60d6cb991567475744be183d62c8f837666` |
| `p2r0f3b2b3_covariance_resummed_sparse_base_v1/summarize_existing_source_domain.py` | `d23bb6156efc1ffa60390d0b1973bb113dee036d14ed409b3f9e9125e1842ff6` |
| `p2r0f3b2b3_covariance_resummed_sparse_base_v1/STATIC_VERIFICATION.json` | `fb4dd200abca08475296878d3cd3907f4cacbd3e750c67a8700233a6f27e4047` |
| `p2r0f3b2b3_covariance_resummed_sparse_base_v1/verify_static.py` | `2063e9538fbb16dfa66139c94aa9352eadbdff84cf3f202909e4ba809cd49cef` |
| `public_aug7_source_observable_tape_v1/GENERATION_RECEIPT.json` | `d2e36181b09c0048791f130ba75897b2962c44eeab87b37086c8f4a6fcb5bfbd` |
| `public_aug7_source_observable_tape_v1/generate_aug7.py` | `7391d75ce5f3bc237a272749060b79c91eb9ac776316e730dd02a987a7a8d38e` |
| `public_coincident_bond_census_v1/CENSUS_COUNTS.json` | `a550aac38157cafd98c8693d165d5d9c2224e38075ade4cc87a4b0cde9c6a8d7` |
| `public_coincident_bond_census_v1/capture_pairs.py` | `544b4105b8620a6e3330bb659c61d2e8fc8524caf85e3d27e5872fd272a812af` |
| `public_native_r10_source_uncertainty_pilot_v1/GENERATION_RECEIPT.json` | `d6729e20d6851c09d834f9bbaa1cd52a709ce01e64aa87b029ecb15c4555faf8` |
| `p2r0f3b2b2_independent_gaussian_linked_cluster_grade3_v1/CONSTRAINT_RECEIPT.md` | `d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba` |
| `p2r0f3b2b2_ig_lc3_reset_derivation_v1/DERIVATION.md` | `7c922b58d0baae6a90abd2edce1f19f03b5f5c0f219561630307ee07697332c1` |
| `public_q24_native_transport_energy_final_workspace_v4/estimator.py` | `2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76` |
| `q24_cs3b_hybrid_exact_lowrank_pairing_covariance_design_v1/DESIGN.md` | `a44ee5ce1c3f5e6cf0a29d6885d4e42e76e1349b8790ab37a0196b1cd13c50d0` |

## Actions taken

- target values read: `0`
- observer-selection targets opened: `0`
- public-name joins: `0`
- estimator/source experiments launched: `0`
- canonical graph edits: `0`
- submissions/network actions: `0`

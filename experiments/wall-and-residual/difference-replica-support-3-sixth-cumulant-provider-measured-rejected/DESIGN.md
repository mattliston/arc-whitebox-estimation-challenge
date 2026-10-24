# Difference-replica block carrier for the complete support-3+ K6 action

Status: **STATIC BUILD ONLY — NO ROW, TARGET, ESTIMATOR, RELEASE, OR RUN**  
Date: 2026-08-27

## Decision

This package specifies a weights-only, recursively closed provider for the
**true preactivation connected sixth-cumulant action with coordinate support
at least three**.  It is not called `K6_NG`: it does not subtract the complete
matched-Gaussian post-ReLU cumulant from the hidden state.

The provider propagates independent input paths through the actual affine/ReLU
network, but it does not use their mean as a prediction.  Instead it uses
independent differences and a three-block replica identity to estimate the
connected K6 tensor without materializing it.  Every support sector is read by
an exact symmetric-polynomial quotient.  The construction is target-blind,
unbiased for the stated action, and exact under complete finite-law
enumeration.

No production source is licensed yet.  A fixed split action-energy gate must
pass in every layer and every support-3+ sector before an estimator is built.

## State and reset law

Let `X_l` be the true post-ReLU activation at layer `l`.  Draw three mutually
independent pools of independent path pairs from the standard-normal input and
push every path through the same network:

```text
A: 512 differences,  D^A_r = X_l(x^A_r) - X_l(x'^A_r)
B: 192 differences,  D^B_r = X_l(x^B_r) - X_l(x'^B_r)
C: 128 differences,  D^C_r = X_l(x^C_r) - X_l(x'^C_r).
```

The production path count is therefore exactly

```text
2*(512 + 192 + 128) = 1664.
```

All 1664 Gaussian inputs are independent and seeded only from `mlp.seed` and a
frozen domain separator.  Pools may be reused across layers because a
deterministic network preserves independence of distinct input paths.  Within
each replica product, factors always come from distinct pools.  Cross-type
reuse changes covariance between estimators but never their expectations.

For an independent copy `X'`, `D=X-X'` is exactly centrally symmetric and
cumulant additivity gives

```text
cum6(D) = cum6(X) + cum6(-X') = 2*cum6(X).
```

All odd moments of `D` vanish.  Hence the exact tensor identity is

```text
cum6(X) = 1/2 M6(D)
          - 15/2 Sym(M4(D) tensor M2(D))
          + 15 Sym(M2(D) tensor M2(D) tensor M2(D)).       (1)
```

The fixed replica estimator instantiates (1) as

```text
T_hat = (1/(2*512)) sum_r (D^A_r)^6
        - (15/(2*192)) sum_r Sym((D^A_r)^4 (D^B_r)^2)
        + (15/128) sum_r Sym((D^A_r)^2 (D^B_r)^2 (D^C_r)^2).  (2)
```

The products in the second and third terms are unbiased because their factors
come from independent pools.  Equation (2) is a replica-product estimator,
not a plug-in cumulant and not a same-sample product.  It needs no estimated
mean, covariance inverse, fit, target, or truth tape.

The reset is just exact path propagation:

```text
Z_(l+1) = H_l W_(l+1)
H_(l+1) = ReLU(Z_(l+1)).
```

The difference pools are rebuilt from their paired rows after each ReLU.  This
is closed for any number of resets.  The static verifier checks the identity
after two exact rational affine/ReLU resets of both parity-witness laws.

## Exact support-3+ action

One block term contains factor vectors `v_1,...,v_k` with multiplicities
`lambda=(lambda_1,...,lambda_k)`, summing to six.  The three types in (2) are
`[6]`, `[4,2]`, and `[2,2,2]`.  For an action direction `w`, define

```text
z_(b,i) = v_(b,i) w_i
P_alpha = sum_i product_b z_(b,i)^alpha_b,
P_0 = n,
F_lambda = product_b P_(e_b)^lambda_b.
```

An exact inclusion/exclusion over one- and two-coordinate monomials gives

```text
Q_3+(lambda; w)
 = F_lambda
   - 1/2 sum_(0<=alpha<=lambda) binom(lambda,alpha)
       * (P_alpha P_(lambda-alpha) - P_lambda)
   + (n-2) P_lambda.                                      (3)
```

For `[6]`, (3) reduces to

```text
p1^6 - 6 p1 p5 - 15 p2 p4 - 10 p3^2 + 30 p6.
```

Every `P_alpha` is one matrix map from a mixed factor row to `W^q`, where
`q=sum(alpha)`.  Pack all rows of the same total degree and perform exactly
six float64 GEMMs per transition.  The fixed row counts are

```text
q:             1     2     3     4     5    6
packed rows: 1280  1856  1984  1856  1280  832
total map-row units: 9088.
```

No K33, K42, or K51 plane is constructed or imported.  Formula (3) removes
all support-at-most-two monomials inside each replica block.

For the source gate only, the same `P_alpha` values are decomposed into the
seven sectors `[411]`, `[321]`, `[222]`, `[3111]`, `[2211]`, `[21111]`, and
`[111111]` using a frozen partition-lattice Moebius table.  This adds no GEMM
to the gate.  The compressed nonzero polynomial term counts for carrier types
`[6]`, `[4,2]`, and `[2,2,2]` are respectively 48, 126, and 282.  The static
verifier reconstructs these tables and checks every sector against direct
ordered-index enumeration.

## What is new and what it supersedes

This is a complete true-source action, not a local matched-Gaussian diagram.
It includes any physical degree-six star, triangle, and longer-path content
present in the actual activation law.  Therefore:

- the failed Gaussian mixed-D6 and higher-path shells are diagnostic controls,
  never additive components of this candidate;
- if this provider is ever activated, it replaces every experimental G8/G16
  degree-six shell spelling rather than stacking with it;
- the live D4 estimator contains no deployed G8/G16 shell, so the current
  production association has no such double count;
- D4's `1/2 L3^2` term is the disconnected quadratic Edgeworth contribution,
  while (2) is the connected `L6` tensor.  They are distinct cumulant terms.

The provider is also distinct from Kerdock.  Kerdock used a degree-three
structured cubature and a separately learned/selected complete directional
interface.  This state uses independent Gaussian paths only as a causal source
for a fixed, unbiased connected-cumulant replica identity; its action decoder
is exact algebra and has no endpoint basis, q3:q5 query bank, fit, or selected
ray family.

## Frozen deployment arithmetic

Constants are `n=1024`, 15 source/action transitions, ranks
`(512,192,128)`, 1664 float32 paths, and 9088 float64 map-row units.  A literal
official-FlopScope-0.12 upper-bound census is:

```text
IID normal generation                         27,262,976
15 float32 path GEMMs                     52,319,354,880
15 path ReLUs                                 25,559,040
difference subtraction                         12,779,520
float64 weight casts/powers                    188,743,680
float64 difference casts                        25,559,040
factor powers and mixed products               271,319,040
six packed float64 degree maps             571,488,337,920
total-Q3+ polynomial assembly                1,411,645,440
type reductions/scales                          25,620,480
D4 association upper bound                     189,050,880
-----------------------------------------------------------
enumerated incremental core                 625,985,232,896
fixed conversion/copy/call padding            2,786,811,904
frozen incremental admission cap            628,772,044,800
```

The float64 GEMM line charges twice the ordinary dot-product count:

```text
15 * 2 * n * (2*n-1) * 9088.
```

The fixed padding is an admission cushion, not permission to omit operations
from the eventual meter receipt.  The full implementation must measure at or
below the cap; a static formula is not official runtime authority.

Against banked D4's maximum `1,356,595,288,584`, the frozen cap gives

```text
max combined FLOPs       1,985,367,333,384
utilization              0.9028405354
margin below 2^41          213,655,922,168
```

The packed-D4 projected peak is `5,411,373,056` bytes.  The explicit
incremental simultaneous arrays sum to `220,053,504` bytes; rounding that to a
256 MiB arena cap gives projected total `5,679,808,512` bytes and margin
`2,910,126,080` below 8 GiB.

The deployment path has 15 float32 path GEMMs and exactly 90 added float64
action GEMMs.  Sector diagnostics are not executed in the estimator.  No
scalar coercion, `.item()`, or device/host truth branch is allowed inside a
layer.  Because local evaluator 0.16 / FlopScope 0.12 sync behavior is not
comparable to the current evaluator 0.15 / FlopScope 0.11 report, residual
authority requires a lean matched-parent A/B/A run reporting absolute and
differential residual.  Harness diagnostic time must not be charged as, or
confused with, estimator residual.  The candidate must remain strictly below
0.4 seconds; the banked D4 residual leaves only about 48.49 ms of nominal
headroom.

## Exact source and variance gates

The first gate has two conjunctive parts.

1. **Exact finite-law oracle.**  Reconstruct (1)--(3) in rational arithmetic;
   verify the `P_+ / P_-` lower-marginal witness, two exact affine/ReLU resets,
   and all seven sector actions of a rank-one Rademacher law.  Exact complete
   enumeration must have zero discrepancy.
2. **Production split action energy.**  Divide every pool into two disjoint
   equal halves and emit sector actions `y1,y2`.  For each of 15 frozen source
   layers, each of three already registered target-blind role ranks, and each
   of seven sectors, compute

   ```text
   S = <y1,y2>,
   N = ||y1-y2||^2 / 2.
   ```

   Require `S>0` and `N/S <= 1/9` in every cell.  With independent unbiased
   halves, `S` estimates signal energy and `N` estimates per-half noise, so
   `S/(S+N) >= 0.9`.  There is no pooling rescue, layer omission, sector
   omission, alternate rank, seed retry, or shrink fit.  A single failure
   closes this spelling before a D4 build or target access.

Only after both gates pass may a zero-amplitude five-component D4 parity and
official resource A/B/A be built.  Only after that may one coefficient-one,
all-site target gate be separately authorized.

At the frozen cap, the nonadaptive target outcomes are:

```text
raw MSE <= 1.9755664581e-8   to beat banked D4 adjusted score
raw MSE <  8.4999328972e-9   to beat the live leader
```

The first requires a 32.8246% raw reduction from D4.  This high-cost provider
is not useful for a small correction.

## Firewall

- no row, MLP weight, target, score, observer selection, or deployment truth
  is opened by this package;
- no estimator or fixture is called;
- no source/action gate is executed;
- no graph, release, candidate, or submission is created;
- all ranks, coefficients, pool custody, sectors, layers, failure rules, and
  score thresholds are frozen before any scientific access.


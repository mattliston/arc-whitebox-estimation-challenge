# Non-Gaussian support-3+ connected K6 provider constraint

Status: **EXTERNAL CONSTRAINT — STATIC ALGEBRA ONLY; NO S0, ROW, TARGET,
ESTIMATOR, RELEASE, OR SUBMISSION**  
Date: 2026-08-27

## Decision

The one genuinely distinct sixth-order state left after the measured marginal
and pair programs is

```text
K6_NG,3+(h_l) = Q_3+ { cum6(ReLU Z_l) - cum6(ReLU G_l) },
G_l ~ N(mu_l, Sigma_l).
```

`Q_3+` removes every entry supported on fewer than three coordinates.  The
Gaussian subtraction removes the already-tested Gaussian degree-6 star and
triangle births.  Therefore this object excludes the marginal `[6]` sector,
the pair `[51]`, `[42]`, and `[33]` sectors, and the Gaussian `(3,2,1)` /
`(2,2,2)` mechanisms.

This is new information, but no autonomous provider/reset exists.  Rebuilding
it from the current D4 Q7/CP3/CP4 state would be a representation repair, not
new information.  The node is consequently a concrete external algorithm
constraint, not a runnable experiment.

## Exact quotient and size

At width `n`, the symmetric order-six state has `C(n+5,6)` entries.  The
support-at-most-two portion contains

```text
n + 5*C(n,2)
```

entries: one marginal pattern and the two orientations of `[51]`, the two
orientations of `[42]`, and one `[33]` pattern per unordered pair.  At
`n=1024`:

```text
all symmetric K6 entries        1,624,866,254,968,320
support <= 2 entries                    2,619,904
support >= 3 quotient           1,624,866,252,348,416
fp64 dense bytes               12,998,930,018,787,328
```

The remaining multiplicity sectors are `[411]`, `[321]`, `[222]`, `[3111]`,
`[2211]`, `[21111]`, and `[111111]`, with affine diagonal-action
multiplicities `30, 60, 90, 120, 180, 360, 720`.

A literal unique-coefficient dot bank over all 1024 output directions costs at
least

```text
n*(2*q_3plus - 1) = 3,327,726,084,809,554,944
```

ordinary scalar operations per transition.  Fifteen transitions cost
`49,915,891,272,143,324,160`, before source/reset work and before the official
float64 multiplier.  Dense storage and action are therefore impossible.

## Minimal matrix-free CP/Waring action

For one symmetric atom `u^6`, one output column `w`, and `z_i=u_i*w_i`, define
`p_a=sum_i z_i^a`.  Exact inclusion/exclusion gives

```text
<Q_3+(u^6), w^6>
  = p1^6 - 6*p1*p5 - 15*p2*p4 - 10*p3^2 + 30*p6.       (1)
```

Equation (1) is verified exactly against ordered six-index enumeration by
`verify_static.py`.  For a signed Waring carrier

```text
K6_NG,3+ = sum_(r=1)^R lambda_r Q_3+(a_r^6),
```

all output actions can be formed from six powered factor maps.  Under official
FlopScope 0.12 float64 weighting, those maps alone cost at least

```text
24*R*n^2 per transition
360*R*n^2 = 377,487,360*R over 15 transitions.          (2)
```

The banked D4 maximum bill is `1,356,595,288,584`, leaving
`842,427,966,968` below `2^41`.  Equation (2) permits at most `R=2231` even if
powers, reductions, quotient arithmetic, source, reset, recompression,
diagnostics, and receipts are free.  `R=2232` already costs
`842,551,787,520`.  This is a necessary action bound, not a candidate bill.

Tucker, TT, and TensorSketch do not create a source.  Optimistic action-only
proxies cap a dense symmetric Tucker mode rank near 43 and a uniform TT rank
near 57 before reset/recompression.  Degree-six TensorSketch/CtR can reduce
action cost, but its variance dependence is already exponential in degree
before the 41-term connected-cumulant cancellation.  It remains admissible
only with a deterministic uniform action-error certificate against an
independent exact source oracle; split-sketch agreement is insufficient.

Primary references for that narrow sketch statement are Wacker, Ohana, and
Filippone (AISTATS 2023), Han, Avron, and Shin (ICML 2020), and Sharma et al.
(UAI 2026).  None supplies the missing non-Gaussian recursive K6 provider.

## Exact lower-state non-identifiability

For `x in {-1,+1}^6`, define

```text
P_theta(x) = 2^-6 * (1 + theta*product_i x_i),
theta in {-1,+1}.
```

The two laws have identical every-five-coordinate marginal.  Hence they have
identical means, covariance, every cumulant through order five, and every
sixth-order entry supported on at most five labels—including all marginal and
pair K6 sectors.  Their all-distinct connected K6 equals `theta`.

Let `A_i=1+x_i` and `w=(1,1,1,-1,-1,-1)`.  Exact enumeration gives

```text
E_+ ReLU(w^T A) = 3/4,
E_- ReLU(w^T A) = 9/8.
```

Thus D4 plus every marginal/pair sixth-order carrier cannot identify the next
ReLU mean.  This proves that a new provider is necessary; it does not prove
that K6 alone is sufficient for the full hidden law.

## Exact D4 association, conditional on a future provider

If a future provider emits the raw affine action `s`, use coefficient one at
all 15 actions.  Define `x=s*pre_sigma^-6/720`.  Before D4's validity/floor
operations, add `x*c[p,6]` to univariate raw moments `p=1..4`.  In the pair
weights add raw `s[None,:]/720` to `(0,6)` and `s[:,None]/720` to `(6,0)`;
the pair derivative features already carry `sigma^-6`, so a second scale is
wrong.  Then recompute every covariance repair, connected K21/K4/K31/K22,
CP3/CP4 reset, native tape, native-J, and final clamp descendant.  Zero
amplitude must branch around all mutations and remain byte-identical to D4.

This association is a specification only.  There is no licensed provider or
autonomous estimator.

## Single external node, four conjunctive certificates

A future child may open only when one immutable package supplies all four:

1. **NOVELTY / RESET.** A target-free autonomous source and two-reset law,
   exact on small-width rational truth, including the `P_+ / P_-` witness.
   It must not reconstruct K6 from the existing D4 summaries.
2. **CARRIER / ACTION.** A fixed CP/Tucker/TT/sketch/circuit representation
   with exact small-width parity and a one-sided certificate retaining at
   least 90% action energy at every tested layer and separately in all seven
   multiplicity sectors.  No pooled rescue or representation sweep.
3. **D4 / RESOURCE.** Literal zero-amplitude five-component public parity,
   full official FlopScope 0.12 source/reset/recompression accounting below
   `2^41`, peak RSS below 8 GiB, and residual strictly below 0.4 seconds.
4. **Q-CAPACITY.** Only after 1--3 pass, one separately authorized frozen-
   roster coefficient-one/all-site test must close more than 60% of the
   transported D4 late-q gap and meet the exact cost-adjusted raw ceiling.

The Gaussian D6 star and triangle each closed only about 9--11% of squared
gap; D8/D10 path correlations are about 0.02.  The exact physical q10:q15
oracle proves that capacity exists, but gives no attribution to K6.  Therefore
there is no lawful S0, target opening, integration, or submission today.

## Governance

- no rank or row is selected;
- no source or estimator is implemented or called;
- no target, score, weight payload, or public name is opened;
- no release, run, package, or submission exists;
- K33 and K42/K51 terminal no-retry receipts remain controlling for those
  spellings and are not reused as K6 authority.


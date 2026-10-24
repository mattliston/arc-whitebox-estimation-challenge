# P2R0F3C late-layer Hermite control-variate audit

Date: 2026-08-26  
Scope: target-free theory and frozen public/source receipts only  
Decision: **close the currently implementable route; retain one external moment-oracle constraint**

## Question and terminal answer

The only pathwise-control sliver not already covered by the H1/degree-2,
quadrature, or QMC closures is a scalar high-degree Hermite control at the last
preactivation.  Freeze one output lane, write its layer-15 preactivation as

\[
H=\mu+\sigma Z=\sigma(Z+a),\qquad a=\mu/\sigma,
\]

and the requested final activation as \(Y=H_+\).  Under a Gaussian law, degree
8 is strong enough on variance.  On the frozen 200-row target-free Q24-native
\((a,\sigma)\) tape, degree 8 captures 0.9988974254 of Gaussian activation
variance, leaving mean residual variance 8.24288215e-5.  The feature arithmetic
is cheap enough to coexist with a 64,000-path sampler.

That is not yet a deployable estimator.  An unbiased control also needs the
true means of the eight Hermite controls under the actual layer-15 law.  No
audited provider supplies them with an error certificate.  In particular,
submission #328439 carries approximate pre-mean/pre-covariance and approximate
third/fourth cumulants only.  Its `q38` table is a table of *Gaussian ReLU
response coefficients*, and its final univariate correction is frozen at
\(1+L_3+L_4+\tfrac12L_3^2\); neither object is an authority for
\(E[He_d(U)]\), \(d=1,\ldots,8\), under the true preactivation law.

Estimating those means with the same full-path samples cancels the control;
estimating them with an independent full-cost split is provably no better than
plain Monte Carlo.  Therefore this audit adds no runnable open leaf.  It reduces
the sliver to the external constraint `CERTIFIED_L15_HERMITE8_MOMENT_ORACLE`
defined below.

## Exact one-step Gaussian projection

Use probabilists' orthonormal Hermites
\(\psi_d(z)=He_d(z)/\sqrt{d!}\), standard-normal density \(\phi\), and CDF
\(\Phi\).  The coefficients \(b_d=E[Y\psi_d(Z)]\) are

\[
\begin{aligned}
b_0/\sigma &= \phi(a)+a\Phi(a),\\
b_1/\sigma &= \Phi(a),\\
b_d/\sigma &= \phi(a)\,He_{d-2}(-a)/\sqrt{d!},\qquad d\ge2.
\end{aligned}
\]

Also

\[
V(a)=\operatorname{Var}[(Z+a)_+]
=(1+a^2)\Phi(a)+a\phi(a)-[\phi(a)+a\Phi(a)]^2.
\]

Thus the exact degree-\(D\) Gaussian capture and residual are

\[
C_D(a)=\frac{\sum_{d=1}^D(b_d/\sigma)^2}{V(a)},\qquad
R_D(a,\sigma)=\sigma^2V(a)[1-C_D(a)].
\]

The verifier uses the numerically stable identity, for \(t=|a|\),
\(q=\Phi(-t)\), \(p=\phi(t)\), and \(r=p-tq\),

\[
R_1(t)=(1+t^2)q-tp-r^2-q^2,
\]

then subtracts the explicitly generated squared coefficients for degrees
2 through \(D\).  This avoids saturated-tail cancellation.

At the worst central gate \(a=0\):

| degree through D | exact capture C_D(0) |
|---:|---:|
| 1 | 0.733471103462130 |
| 2 | 0.966942206924260 |
| 4 | 0.986398132212771 |
| 6 | 0.992234909799324 |
| 8 | 0.994840614079035 |
| 10 | 0.996259275297989 |
| 12 | 0.997129817409620 |
| 16 | 0.998116132852901 |
| 24 | 0.998964060057158 |
| 32 | 0.999323781802025 |

The public real-network variance range 0.0643--0.0862 and the idealized
65,536-path ceiling require capture 0.9898077760--0.9923972158 for raw MSE
below 1e-8.  Degree 6 misses the high-variance edge (0.99223491 versus
0.99239722); degree 8 passes.  This is why the surviving discriminator is
frozen at D=8, not D=6 and not an adaptive degree search.

## Target-free native-tape census

The source is exactly observer-fit ranks 35:235 from the frozen Q24-native
tape.  `raw20[...,3]` is alpha and `sigma.npy` is the preactivation sigma; only
layer index 15 is used.  The receipt binds every result/raw20/sigma file through
a canonical 200-row SHA-256 roster digest.  No target, truth, score, name join,
or observer-selection field is involved.

Across 204,800 lanes:

- alpha range: -9.63902092 to 9.61539268; median -0.00404315; RMS 3.16526100;
- sigma range: 0.25343156 to 0.78555053; mean 0.40009768;
- Gaussian mean activation variance: 0.07476031415.

Energy-weighted Gaussian results are:

| D | aggregate capture | mean residual variance | residual/N at N=64,000 |
|---:|---:|---:|---:|
| 1 | 0.9575481104 | 3.17371660e-3 | 4.95893219e-8 |
| 2 | 0.9895521487 | 7.81084645e-4 | 1.22044476e-8 |
| 4 | 0.9967146639 | 2.45612756e-4 | 3.83769932e-9 |
| 6 | 0.9982735363 | 1.29070971e-4 | 2.01673391e-9 |
| 8 | 0.9988974254 | 8.24288215e-5 | 1.28795034e-9 |
| 12 | 0.9994096874 | 4.41319518e-5 | 6.89561747e-10 |
| 24 | 0.9997946490 | 1.53521081e-5 | 2.39876689e-10 |
| 32 | 0.9998671481 | 9.93205220e-6 | 1.55188316e-10 |

For D=8 the 200 per-row residual-variance range is
5.46857542e-5--1.13388287e-4; at 64,000 paths every row's Gaussian sampling
term is below 1.772e-9.  These are source-only Gaussian feasibility numbers,
not accuracy claims about the true non-Gaussian law.

## Bias and required moment authority

Freeze deterministic normalization \(U=(H-\hat\mu)/\hat\sigma\), coefficients
\(\beta_d\), and supplied moment means \(\hat m_d\).  The CV estimator is

\[
\hat\theta=\bar Y-\sum_{d=1}^8\beta_d\bar\psi_d(U)
             +\sum_{d=1}^8\beta_d\hat m_d.
\]

Its bias is exactly

\[
B=\sum_{d=1}^8\beta_d[\hat m_d-E\psi_d(U)].
\]

Approximate \(\hat\mu,\hat\sigma\) do not invalidate the identity, but then
even the degree-1 and degree-2 moment means need authority.  If normalization
were by the *true* mean and variance, the higher Hermite means would be

\[
\begin{array}{c|l}
d & E[He_d(U)]\\ \hline
3&\kappa_3\\
4&\kappa_4\\
5&\kappa_5\\
6&\kappa_6+10\kappa_3^2\\
7&\kappa_7+35\kappa_4\kappa_3\\
8&\kappa_8+56\kappa_5\kappa_3+35\kappa_4^2.
\end{array}
\]

So third/fourth cumulants alone cannot center D=8.  The bias certificate must
bound the *combined activation-unit bias* above, not merely relative error in
individual cumulants.  Using the target-free native D=8 residual point model,
the maximum fleet RMS bias left by a 1e-8 budget is:

| integrated path count | sampling term | remaining RMS bias ceiling |
|---:|---:|---:|
| 65,454 | 1.25933971e-9 | 9.34914985e-5 |
| 64,000 | 1.28795034e-9 | 9.33383612e-5 |
| 13,285 | 6.20465348e-9 | 6.16063837e-5 |
| 9,585 (1.877e12-host illustration) | 8.59977272e-9 | 3.74196109e-5 |

These are algebraic ceilings before uncertainty or safety margin.  A reopen
certificate must use an upper bound on both terms, not these point estimates.

## Why sample-estimated moments do not rescue the route

Let \(S\) be the degree-8 projection, \(R=Y-S\), and suppose the ideal
projection makes \(R\) and \(S\) orthogonal.  If the moment of \(S\) is
estimated from an independent full-path split, with \(n_R+n_S=N\), then

\[
\operatorname{Var}(\bar R_{n_R}+\bar S_{n_S})
=V_R/n_R+V_S/n_S.
\]

The optimal allocation is proportional to \(\sqrt{V_R}:\sqrt{V_S}\), giving

\[
\min\operatorname{Var}
=\frac{(\sqrt{V_R}+\sqrt{V_S})^2}{N}
>\frac{V_R+V_S}{N}=\frac{V_Y}{N}.
\]

If the same samples provide the moment, the two sample-control means cancel
and the estimator is exactly \(\bar Y\).  Equal-cost cross-fitting merely
repackages one of these cases.  A gain therefore requires a genuinely cheaper
unbiased prefix/moment oracle; full-path Monte Carlo, Sobol, or QMC moment
estimation is not that oracle.

## Width-1024 cost audit

The public dense path lower-bound convention is 2^25 = 33,554,432 FLOPs.
For D=8, a conservative streaming feature bound per path and output vector is

\[
[2+3(D-1)+2D+2]\,1024=(5D+1)1024=41,984\ \text{FLOPs}.
\]

This includes standardization; the normalized Hermite recurrence; eight
multiply-accumulates; residual formation and reduction.  A separate 128 FLOPs
per lane (131,072 total) bounds coefficient and supplied-moment setup.  The
unknown cost of producing certified moments is `C_oracle` and is not hidden in
that setup allowance.

Exact integer cases:

- if prefix/moments are free, adding the D=8 arithmetic changes the theoretical
  ceiling from 65,536 to 65,454 paths;
- the frozen measured paired 64,000-path sampler costs 2,187,012,965,375 FLOPs;
  adding 2,686,976,000 feature FLOPs and 131,072 setup FLOPs still leaves
  9,323,183,105 below 2^41;
- the existing Hermite-10 full-covariance provider costs 103,923,320,832 FLOPs,
  but is a Gaussian propagation provider, not a certified true-moment oracle.
  Adding it to the frozen 64k sampler exceeds the board by 94,600,137,727
  FLOPs.  A conservative affine repricing of that sampler leaves 61,182 paths;
- the audited historical no-eigh static host bound is 1,752,673,252,727, leaving
  446,350,002,825.  Dense paths plus D=8 features fit only 13,285 paths;
- submission #328439's score ratio implies a board-weighted compute equivalent
  of about 1,708,040,753,256 FLOPs.  That is neither a per-network static bound
  nor additive headroom, so it is recorded but never used as a deployment
  budget;
- an explicitly non-authoritative 1.877e12 host illustration leaves 9,585 paths.
  It requires capture 0.99850933--0.99888805 over the public variance range;
  native aggregate D=8 clears the variance-only edge by only about 9.4e-6 and
  leaves merely 3.74e-5 RMS for moment bias.

## Dedupe

- The exact-centered H1/arccos and full-H1 cross-fit families stop at linear or
  quadratic controls.  D=8 at the final scalar preactivation is theoretically
  distinct, but their sample-estimated-mean loophole is closed by the split
  variance calculation above.
- The public quadrature derivation closes whole-network degree-2 CV and
  quadrature economics.  This audit does not reopen that family.
- Sobol/QMC is already represented by the full-budget sampler.  Using it to
  estimate the missing Hermite means at full-path cost gives no independent
  cheap expectation authority.
- The analytic Hermite-10 full-covariance estimator assumes its propagated law;
  order 10 in its name does not certify true layer-15 moments through degree 8.

## External reopen constraint

`CERTIFIED_L15_HERMITE8_MOMENT_ORACLE` must simultaneously provide:

1. For every network and output lane, a source-only value for
   \(E[\psi_d((H_{15}-\hat\mu)/\hat\sigma)]\), d=1..8, or directly for
   \(E[S_8]\), relative to the exact frozen normalization and coefficients.
2. A target-free certified upper bound on fleet mean squared combined bias
   \(E_j[B_j^2]\), including normalization and truncation error; individual
   cumulant heuristics are insufficient.
3. An integrated source meter with
   `host + C_oracle + setup + N*(2^25+41984) <= 2^41` (or a stricter exact
   meter), and a target-free upper residual-variance bound satisfying
   `U_residual/N + U_bias_squared < 1e-8`.
4. Exact small-width parity for moments and CV centering, with no targets,
   training, score, or submission licensed by this constraint itself.

Until an external construction satisfies all four clauses, the route is
`CLOSED_CURRENT_FORM__EXTERNAL_CERTIFIED_MOMENT_ORACLE_ONLY`.

## Access declaration

Targets opened: 0.  Target values decoded: 0.  Fits: 0.  Scores: 0.
Submissions: 0.  Estimators executed: 0.  Production runs: 0.  Network calls:
0.  Canonical graph edits: 0.  Only frozen source artifacts and target-free
native feature tapes were read.

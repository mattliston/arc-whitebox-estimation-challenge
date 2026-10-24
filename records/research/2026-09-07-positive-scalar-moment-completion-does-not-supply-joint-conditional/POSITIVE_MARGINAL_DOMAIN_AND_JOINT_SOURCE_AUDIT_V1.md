# Positive marginal completion does not supply the missing joint source

Phase 2 only, September 7, 2026. Use filesystem mtime for completion time.
This is an additive analytical/source audit. No network arrays, fit, new
capture, numerical estimator, Monte Carlo acquisition or score measurement.
Preserve the two-memory v2 note and its implementation qualifications,
the response-row exception closeout, and every earlier frozen receipt.

## Prospective two-route comparison

Confirmed official bank #329926 remains adjusted 4.982231460183304e-9.
Best qualified independent-14 LOCAL candidate 0b84 remains 4.788089831892398e-9,
noise-corrected raw 2.245315959186444e-8 and mean complete FLOPs
467915916017.2143. Its SHA was rechecked unchanged in this audit:
0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96.
This is not a new board poll or a new local score.

1. CP-free nonlinear two-memory compression: the preceding v2 construction's
   conditional complete source census is 278925711248F, multiplier
   .12684072828415083. Multiplying unchanged current mean raw by this
   proposed fixed cost gives 2.8479751149123525e-9, NOT an accuracy forecast.
   A complete implementation at that cost needs raw below
   2.3651709041589128e-8 for adjusted <3e-9, or below 7.883903013863043e-9
   for <1e-9. Exact coordinate-star transport and a cost spelling exist;
   successful compression of the omitted all-distinct information does not.
   Its source-variance domain was the cheapest immediate analytic test.
2. New causal joint-cumulant supply: rank35's privileged local oracle remains
   the evidence for opportunity (K3-offdiagonal next-mean ratio .3275057;
   joint K3/K4 .0209139). These are neither independent final raw nor
   attainable forecasts. No compact new supplier or complete deployment
   cost has been constructed. The cheapest next test is source/ownership
   algebra, not a fit. The signed-path result .8483354 at roughly 1.81x
   projected full cost does not justify that old implementation's revival.

Neither has a credible present end-to-end accuracy case for substantial
compute. The selected work in this turn is the first route's domain check,
alongside an independent exact-source audit for the required raw route.
All actual advancement must use same-configuration raw times complete
multiplier; a raw increase is acceptable if its complete cost reduction pays.

## 1. A squared-Hermite scalar law

Let H3(z)=z^3-3z and q_c(z)=phi(z)(1+c H3(z))^2/(1+6c^2).
For Z with that density, put X=Z/sqrt(V), where
V=(1+42c^2)/(1+6c^2). Direct Gaussian polynomial integration gives

    E[X]=0, E[X^2]=1,
    gamma(c)=12c sqrt(1+6c^2)/(1+42c^2)^(3/2),
    k(c)=Cum4(X)=216c^2(1-12c^2)/(1+42c^2)^2.

The scalar density is nonnegative, normalized and has full support except
for isolated zeros. Its largest achievable absolute standardized skew is
.7390554304290361, attained at c^2=(sqrt(43)-6)/42. Exact intended-skew
matching needs an explicit inverse branch and range policy. The simpler
c=t/12 gives gamma=t-(5/12)t^3+O(t^5), not exactly t, and forces
k=(3/2)t^2+O(t^4). Thus a single squared-H3 law is not a literally
zero-fourth marginal repair with an unchanged third diagonal.

The general squared-polynomial-times-Gaussian construction is established
prior art, not a newly invented density family: Gallant and Nychka's author
abstract, https://www.aronaldg.com/pubs/emet87.html. The formulas and choices
in this note were independently derived here; no density fitting was done.

## 2. Constructive zero-fourth scalar completion

The single-component fourth mismatch is not a general obstruction. Let
q_* be the equally weighted mixture of the STANDARDIZED q_{+1/sqrt(6)}
and q_{-1/sqrt(6)} laws. Then q_* has mean0, variance1, skew0 and
excess fourth -9/16. For c^2<=1/12 set

    w(c)=k(c)/(k(c)+9/16),
    p_c=(1-w(c))*standardized_q_c + w(c)*q_*.

Here 0<=k(c)<=1 and 0<=w<=16/25. Equal component means/variances imply
that p_c has mean0, variance1, EXACTLY zero fourth cumulant and skew
Gamma(c)=(1-w(c))*gamma(c). In particular, this is not an attempt to
cancel fourth cumulants of components with different variances.

One fully specified mathematical parameterization for every finite intended
t is c=tanh(t/sqrt(12))/sqrt(12). It yields

    c=t/12-t^3/432+O(t^5),
    w=(8/3)t^2+O(t^4),
    Gamma=t-(28/9)t^3+O(t^5).

Actual skew is bounded by 4/9 and is NOT globally monotone in t. This
parameterization explicitly approximates the intended third moment; it
is not a universal exact moment solver or a chosen estimator variant.
No sweep, target-derived parameter or refit was used to choose it.

Root and the accuracy agent independently checked the scalar formulas. Root also
checked k's maximum at c^2=1/66. For the Gamma bound, with x=c^2,
Gamma^2=144x(1+6x)(1+42x)/(1+468x-2844x^2)^2. The nonnegative difference
for the claimed bound factors as

    (1+468x-2844x^2)^2 -729x(1+6x)(1+42x)
    =(1-12x)*(1+219x+180972x^2-674028x^3).

The last cubic is positive on [0,1/12].

## 3. Exact positive ReLU variance, not floating-point qualification

For any finite mu, sigma>0 and finite parameter, this density assigns
positive mass on both sides of the ReLU threshold and a nondegenerate
continuous positive part. Hence Var(ReLU(mu+sigma X)) is strictly positive
in exact arithmetic, including the prior A8,t=-.02 counterexample domain.

For standardized q_c, let b=-(mu/sigma)*sqrt(V), sigma'=sigma/sqrt(V).
Raw post moments through degree3 are

    sigma'^m/(1+6c^2) * integral_b^infinity
      (z-b)^m [1+2c H3(z)+c^2 H3(z)^2] phi(z) dz.

These use Gaussian truncated moments only through degree9; the q_* branch
cancels its odd term. The completed mixture needs two fixed polynomial-
Gaussian evaluations, O(n) scalar work per layer. For fixed alpha=mu/sigma,
its mean and raw second agree with first-cubic Edgeworth through first order:

    mean = mG - sigma*t*alpha*phi(alpha)/6 + O(t^2),
    raw2 = r2G + sigma^2*t*phi(alpha)/3 + O(t^2).

This expansion is not uniform in extreme alpha. Gaussian tail recurrences,
variance subtraction, mixture rounding and normalization can still fail in
floating point. No such implementation was tested. The old 278925711248F
census does NOT certify this changed scalar/joint reader's complete cost.
No old runtime, workspace, Cholesky or PSD qualification is inherited.

Most importantly: changing this marginal requires explicit third-diagonal
ownership and consistent recentering of raw pair moments. Zero MARGINAL
fourth does not define mixed K31/K22, and arbitrary D offdiagonals need not
be realizable with these marginals. A declared approximate state splice is
possible, but no fully specified new deployed model is admitted here.

## 4. Positive marginals cannot determine the required joint information

For three coordinates the continuous positive densities

    p_+(z),p_-(z) = product_i phi(z_i)
                   *[1 +/- epsilon product_i tanh(z_i)],  0<epsilon<=1,

have exactly the same Gaussian univariate and bivariate marginals. They
have C=I and zero repeated-third planes, but

    Cum(Z1,Z2,Z3) = +/- epsilon*(E_G[Z*tanh(Z)])^3 != 0.

Integrating out any coordinate annihilates the perturbation. This exact
information witness is a general continuous law, NOT a claim of Phase2
MLP realizability or measured error. A causal latent state could retain
extra information; the witness does not close every such state. It shows
why marginal validity alone cannot establish the requisite supplier.

## 5. Exact missing source and subtraction ownership

With truly centered incoming Y and fixed gates g, define
r_i=ReLU(mu_i+Y_i)-E[ReLU(mu_i+Y_i)]-g_i Y_i. Then r is centered and

    T_true^+ = G^tensor3 T_true^-
               + cyclic gg E[YYr] + cyclic g E[Yrr] + E[rrr].

For a source coordinate k, v_k=C_kk, c_i=C_ik/v_k and
epsilon_i=Y_i-c_iY_k, the first mixed moment is exactly

    E[Y_iY_j r_k] = c_i c_j E[Y_k^2 r_k] + E[r_k Xi_ij^(k)(Y_k)],
    Xi_ij^(k)(y) = c_i y E[epsilon_j|Y_k=y]
                   + c_j y E[epsilon_i|Y_k=y]
                   + E[epsilon_i epsilon_j|Y_k=y]
                   - (C_ij-C_ik C_jk/v_k).

The conditional-matrix remainder vanishes under Gaussian conditional
regression. A positive scalar law supplies neither it nor the coupled
E[Yrr]/E[rrr] terms. This is the precise missing statistic, not a new
cheap way to compute it.

Using a bank-centered residual r_hat instead introduces eta=E[r_hat]:
Cum(Y_i,Y_j,r_hat_k)=E[Y_iY_j r_hat_k]-C_ij eta_k. Omitting that unknown
centering term would not be exact. Define S_bank=T_bank^+-G^tensor3 T_bank^-
using the ACTUAL bank star/trace/masks/head-conditioned reset. Then

    DeltaT^+ = G^tensor3 DeltaT^- + S_true-S_bank.

Subtracting only an ideal Gaussian newborn is insufficient. Root inspected
the active 0b84 _predict_binary source at lines1431--1540: its born/trace/
star reset, active masks and hook-updated covariance must be respected.
This inspection is not byte-level certification of a new source bridge.

For next W, put X=W.T GY and R=W.T r. The required repeated-plane source
query is the following FIVE mixed matrices, minus the matching bank query:

    E[X_i^2 R_j] +2E[X_i R_i X_j] +2E[X_i R_i R_j]
      +E[R_i^2 X_j] +E[R_i^2 R_j].

The independently derived dense-circuit costs illustrate why naming those
matrices is not an implementation: one n^3 FP32 source array at n1024
uses4294967296B; ordinary three-mode dense transport costs6593848541184
scalar arithmetic operations. Even the stated repeated-plane-only dense
realization costs2202242383872 operations, before acquiring the source or
carrying feedback. Root verified the formulas 3*(2n^4-n^3) and
2n^4+3n^3-2n^2. These are explicit ordinary dense circuits, NOT native
whole-card bills or universal lower bounds against structured algorithms.

## Decision and next admission requirement

The scalar variance/zero-fourth obstacle has a constructive analytical
completion; it no longer justifies saying every positive completion must
introduce a marginal fourth cumulant. The frozen v2 model itself remains
unchanged and still has its witnessed negative-variance domain.

There is still NO new raw-accuracy evidence, no compression-capacity result
and no complete executable cost for this changed law. Neither a full fit
nor an eight-network recapture is admitted. The next substantive experiment
needs an explicit causal representation of the missing joint remainder,
or credible current-state compression evidence, plus a complete cost and
prospective actual adjusted-score test. Novelty/positivity alone are not
such evidence. No closed family or completed response-row exception reopens.

Read-only public source check: the 2026 SNP dynamics paper uses sample-based
coefficient estimation, not a ready-made deterministic Phase2 closure:
https://arxiv.org/html/2604.09375v1. Its unfiltered degree2/3 multivariate
basis would have180006400 coefficients at dimension1024 (direct count),
not an affordable two-matrix implementation. No code was cloned or run,
and its applications were not treated as Phase2 score evidence.

Resources checked: swap784.75MiB of2048MiB, no numerical worker in ps;
no MPS work, uploads, forum posts, commits, promotions or frozen edits.
Immediate target strictly <3e-9; final strictly <1e-9 remains active, with
validated milestone banks <4e-9, <3e-9 and <2e-9. No milestone is earned here.

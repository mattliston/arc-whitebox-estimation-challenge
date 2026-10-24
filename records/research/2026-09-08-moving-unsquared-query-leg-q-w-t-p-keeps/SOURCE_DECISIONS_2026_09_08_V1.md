# Phase 2 source decisions: no new score or numerical admission

Clock checked September 8, 2026 at 07:19:14 UTC / 00:19:14 PDT.
This is an additive mathematical/source record. No candidate, old receipt,
target array, fit, package or official submission was changed. The root and
the two existing agents performed source review and scalar cost arithmetic.
The completed response-row exception remains closed to tuning/refitting.

## Confirmed reference and advancement objective

Official #329926 remains adjusted 4.982231460183304e-9, from its saved graded
receipt, not a new poll. Fully qualified local resident_gram512_v1/estimator.py,
SHA 0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc,
remains independent14 adjusted 4.7768418485455745e-9, NCraw
2.2458484596872894e-8, mean complete 466705080466.9286F. All its existing
native, remote and release receipts remain intact; no numerical extension
of those qualifications occurred here. No milestone upload is earned.

The objective is strictly below 1e-9 OFFICIALLY, with qualified banks below
4/3/2e-9. Fixed-raw uniform complete-cost reductions would have to exceed
16.2627% / 37.1970% to cross 4/3e-9. At the .1 multiplier floor the present
mean raw still gives 2.24585e-9. Thus a raw route is necessary, not optional.
These are planning conditions, not forecasts or products of independently
averaged score columns.

The two strongest specific source questions examined were:

1. An answer-weighted compression metric for the current signed third bank.
   The metric is available below; an affordable effective constructor is
   not. Expected raw gain is UNKNOWN. A historical shape already owes
   36.265B additional products before selection/solve/assembly, reducing an
   old 88.711B modeled saving to 52.446B before extras. This is not a
   complete current bill. Cheapest decisive test: derive the metric,
   check whether it differs from the closed kernel objective, and charge
   its acquisition. Completed; no new compression experiment admitted.
2. Acquisition of actual joint angular-energy or conditional-variance
   information. Exact input-derivative and replica suppliers exist below,
   but their same-reference signed consumer variance and value are unknown.
   Literal shared-energy acquisition costs 134.152M ordinary product FLOPs
   per depth16 row, plus the stated other costs. Complete estimator cost
   is unavailable. Cheapest decisive work: establish the exact source and
   distinguish it from the closed probe/conditional arms. Completed; no
   score magnitude is supported and no numerical run is forced.

## 1. A tilted cubic metric is not yet a compression algorithm

For a symmetric third-tensor error T and fresh w~N(0,tau^2 I),

    E[T(w,w,w)^2] = tau^6 * (6*||T||_F^2 + 9*||trace(T)||^2).

Exact physical repeated-plane restoration makes the current-coordinate
error all-distinct, hence trace-free. In that case this is precisely the
old all-distinct kernel objective, with kernel s1^3-3*s1*s2+2*s3.
WHOLE_PAIR_KERNEL_FIXED_DEV_V1_CLOSED.md already records 19.025% complete
cost saving and 4.910x worse adjusted error for its fixed construction.
That arm is not reopened.

A different LOCAL mean-response metric freezes next variance at
v0=tau^2*trace(C)>0. The deployed physical derivative is
-alpha*phi(alpha)*delta_k3/(6*v), so

    R(T) = E[(m.w)^2 exp(-(m.w)^2/v0) T(w,w,w)^2] / (72*pi*v0^3).

For m!=0, define u=m/||m||, beta=2*tau^2*||m||^2/(v0+2*tau^2*||m||^2),
eta=1-beta and Aj=||T(u^j, .^(3-j))||_F^2. For trace-free T,

    R = tau^8*||m||^2*sqrt(eta)/(72*pi*v0^3) *
        [eta*(6*A0-18*beta*A1+27*beta^2*A2-15*beta^3*A3)
         +eta^2*(36*A1-108*beta*A2+90*beta^2*A3)].

Root independently checked the tilt to covariance I-beta*u*u.T and the
two directional integration-by-parts terms. The integral representation
is nonnegative despite signed coefficients. At m=0 this particular
first-order mean metric vanishes, not the heads' or covariance sensitivity.
On tensors with T(u,.,.)=0 it remains a positive multiple of ||T||_F^2;
the weight tilt provides no general low-rank guarantee.

For T=Pi_distinct sum_r lambda_r*p_r^3, acquiring T(u,.,.) needs
P*diag(lambda*(P.T*u))*P.T and the FULL physical CP plane for the repeated
subtractions. The current response-masked plane is not that full plane.
Given a K-atom dictionary, the extra metric construction uses one
a-by-M-by-a weighted Gram, one a-by-a-by-K action, seven K-by-a-by-K
feature Grams, and additional contractions/selection/solve/assembly.
On the OLD post7 a896,M13376,K1024 geometry, the additional ordinary FP32
product subtotal is exactly

    a*a*(2*M-1) + a*K*(2*a-1) + 7*K*K*(2*a-1)
      = 36,265,377,792F.

This does not price a current candidate. A core also owes basis, CP-to-core,
three-leg transport, lifted reads and star repair. Physical tensor equality
alone does not preserve differently masked CP/star readers. Future actual
weights, covariance/mean feedback and P-head responses remain outside the
local metric. No cheap dictionary/core with demonstrated retained value
was found.

## 2. Exact energy acquisition without a Jacobian trace probe

Let X,Z be independent standard Gaussians in input dimension d, H the
actual homogeneous prefix output, J=dH/dX, N=||H||^2, eta=E[N]/d and
epsilon=N-eta*||X||^2. Define

    X_t = t*X + sqrt(1-t*t)*Z,
    v = J(X).T*H(X) - eta*X,
    a_t = J(X_t)*v.

Gaussian covariance interpolation gives the exact matrix

    K = E[epsilon*H*H.T]
      = 2*integral_0^1 E[a_t*H(X_t).T + H(X_t)*a_t.T] dt.

One energy VJP and one JVP supply all entries; no full Jacobian or
Hutchinson trace probe is needed. A literal row has two forwards, one
reverse and one tangent: 4*L*n*(2*n-1) ordinary product FLOPs for uniform
width n. At L16,n1024 this is 134,152,192F, BEFORE RNG, gates, accumulation,
eta acquisition, current reference subtraction and consumers. Batched/fast
products may change this subtotal; it is not an algorithmic lower bound
or a metered complete bill.

Root independently checked the factor two and the tempting local identity:

    K = E[(J*v)*H.T + H*(J*v).T] + .5*integral H*H.T*phi_d d(Delta epsilon),
    Delta epsilon = 2*(||J||_F^2-d*eta) + 2*sum_k H_k*Delta H_k.

The first local term has 3*L products (100,614,144F at L16,n1024), but the
remainder includes inherited facet measures. Ordinary second AD does not
measure those kinks. H_k vanishes on its OWN final ReLU facet, not every
upstream facet: ReLU(x1)+ReLU(x2), realizable as a later positive ReLU,
is positive on x1=0,x2>0. Dropping the inherited measures is not exact.

This K is the unnormalized raw energy pair. The angular K_E in the prior
norm-remainder note is K/[d*(d+2)], not K itself. An approximate eta0 does
not keep epsilon centered: covariance interpolation would then supply a
covariance, and the missing E[epsilon0]*E[H*H.T] term must be included.
Subtraction against a bank reference requires matching eta, mean/covariance
and readout ownership. No useful residual variance was measured.

The old PAIRED_MEHLER_PROBE_REJECTION_V1_CLOSED.md rejects a different
fixed per-output trace-probe circuit. It does not reject this shared-gradient
identity; nor does avoiding that probe establish a competitive variance.

## 3. Conditional replicas: exact new statistic, no value witness

For a fixed input projection A=U.T*X, write M(A)=E[H|A] and
Sigma(A)=Cov(H|A). The additional mixed third source is the sum of the
three placements of E[(M-m) tensor Sigma]. Three paths sharing A, with
independent complementary inputs, give the exact representation

    S = .5*(H1-H2)*(H1-H2).T,
    E[(H0-m) tensor S] = E[(M-m) tensor Sigma].

This costs three complete prefix forwards per sample plus basis/mean
acquisition, actions/accumulation and downstream use. Conditional covariance
alone also omits E[third_cumulant(H|A)]. Four paths are needed for a basic
covariance-of-conditional-covariances supplier. Neither construction has
controlled current-consumer variance after subtracting owned contributions.
The full-covariance GMM4 and fixed conditional-feature closeouts remain
preserved; they do not turn this identity into an effective estimator.

## 4. Correction to the old subspace-integration interpretation

Root read measure_realnet.py lines267 onward and the scalar receipts.
The old P3 fixes the TOP subspace A and samples its complement B. If
F=mu+f_A+f_B+f_AB is the orthogonal two-block ANOVA decomposition, it
estimates E_A Var_B(F|A)=V_B+V_AB. Its reported capture is V_A/V.

Perfectly integrating A while sampling B instead leaves V_B. It removes
V_A+V_AB, not just V_A. Thus the old top8 6.526% result is a first-order
ANOVA share, NOT the variance removed by perfect integration of top8.
No original file is edited; this correction narrows the prose bound.

The Gaussian conditional Poincare theorem gives the population bound

    (V_A+V_AB)/V <= E||U.T*grad(F)||_F^2 / V.

Plugging in the OLD sampled gradient Gram and variance gives .214248775
for top2 and .754954615 for top8. These are empirical plug-in quantities,
NOT certified population bounds: 160 sampled Jacobians and the selected
estimated eigenspace have finite-sample uncertainty. They provide no
positive high-capture witness. The separately EXECUTED two-direction GH
grid still had MSE .0010106713 at4096 forwards, about53.4x matched MC;
that failed fixed numerical rule is unaffected and is not reopened.

Preserved originals, all August25,2026 PDT by filesystem mtime:

- realnet_measurements.json, 19:35:23, SHA
  27584cc121de0e0194f591876841475487f2382b23591bf35432ead136a9751a.
- measure_realnet.py, 19:35:01, SHA
  8d784db44d409844e5194542adf2ecc4796cdcb6ca252bcbd76600108ab1239c.
- realnet_rules.json, 19:49:55, SHA
  acdaa4046bd218b55f6800d9f07fd71f827c6ea861730f567912bbd9a0537809.

## 5. Remaining exact rearrangement and primary-source screen

For Q=W.T*P, moving the unsquared query leg gives
K_IJ=(Q_I^2*Lambda*P.T)*W_J. It retains two products instead of the
existing one, while full Q remains necessary for marginal cubes/state.
The ordinary-product saving is
r*[(c-a)*(2*R-1)-c*(2*a-1)]. It loses when support shrinks; even the OLD
a832->c864,R17728 expansion gives -302272*r FLOPs. This is not a current
D4 bill or a universal arithmetic bound. No prototype is warranted.

New public-source check: [Kuang and Lin v2](https://arxiv.org/html/2601.22307v2)
is the May7 revision of their Gaussian moment-matching work; exact Gaussian
maps do not retain a new non-Gaussian joint state. The [Mato et al. paper](https://arxiv.org/html/2510.21742v2)
studies equilibrium recurrent networks with regular odd activations and
weight-averaged correlators (equations2.2/2.8). Its July31 journal publication
does not supply a fixed-weight ReLU MLP estimator. This is a source-scope
decision, not an empirical rejection of a translated algorithm.

No numerical run is admitted by these results. Missing evidence remains
specific: a cheap compression constructor with retained complete-head value,
or a non-owned raw source with consumer-weighted accuracy/variance and an
affordable complete implementation. The goal remains active and unmet.

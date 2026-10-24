# Exact-centered first-layer quadratic control: source-only preflight

September 7, 2026 PDT. Phase 2 only. New algebra and source audit; no
estimator run, target-array access, sampling, coefficient fit, native meter,
submission or forum action. Existing candidates and closeouts are unchanged.

## Decision and score status

This construction provides a legitimate nonlinear control with an analytic
mean. It does **not** provide evidence of the variance reduction required by
the score. No numerical experiment is recommended on the strength of exact
centering alone. The user's instruction not to force an unsupported run
continues to apply.

Official bank: #329926, adjusted 4.982231460183304e-9. Retained response-row
component: independent14 adjusted 4.83424098592912e-9, versus the matched
bank's 4.996669003386079e-9. These are different evaluation populations; the
retained component is not a new official bank. No milestone was crossed.

At retained NC raw 2.2454337457623213e-8, the scoring floor alone gives
2.2454337457623213e-9. Raw must fall by more than 55.46% to reach below1e-9
even at that floor. Compute reductions alone cannot achieve the goal.

## 1. An exact mean without a moment-closure center

Write the Gaussian input X=R U, where R has the chi_n law and U is uniform
on the unit sphere, independently. The bias-free network and its first
layer satisfy

    F(X)=R f(U),  H0(X)=ReLU(W0^T X)=R B(U).

For any deterministic weights-only matrix A, set s=A^T H0. Its mean ms and
second moment M2 are known from the exact first-layer Gaussian ReLU pair
kernel. For Cpre=W0^T W0, sigma_i=sqrt(Cpre_ii) and
rho_ij=Cpre_ij/(sigma_i sigma_j),

    m0_i = sigma_i / sqrt(2*pi),
    M0_ij = sigma_i*sigma_j/(2*pi) *
            [sqrt(1-rho_ij^2) + (pi-arccos(rho_ij))*rho_ij],
    ms = A^T m0,  M2_j = (A^T M0 A)_jj.

Equivalently use C0=M0-m0*m0^T, vs_j=(A^T C0 A)_jj and M2=vs+ms^2.
Zero-variance cases and rho endpoints require a numerically qualified
implementation; none is supplied or silently assumed here.

For each output define, with value zero at X=0,

    N_j(X) = s_j(X)^2/R - (M2_j/n)*R.

Since q_j(U)=s_j(X)^2/R^2 and E[q_j]=M2_j/n, N_j=R(q_j-E[q_j]) has exact
mean zero. So do fixed combinations of N, s-ms and R-E[R]. An IID average
of F minus such a control is unbiased in exact arithmetic. An approximate
moment-closure mean of a deep activation is not used for centering.

Products of first-layer ReLUs are not restricted to a linear odd part:
the odd part of ReLU(x)ReLU(y) is (x*abs(y)+abs(x)*y)/4. Division by R
preserves nonlinear angular information. Thus the shallow *linear*
readout obstruction in COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md,
section4, does not close this product construction. This is not evidence
that its angular information is accurate enough.

## 2. A fully specified coefficient source is possible without fitting

One example, not a recommendation or an implemented candidate, is to obtain
hidden-layer response slopes G1,...,G14 from the deterministic diagonal
Gaussian recursion and form

    A = W1 G1 W2 G2 ... W14 G14 W15.

The diagonal recursion propagates pre-means W^T m and pre-variances
(W*W)^T v, then uses the univariate Gaussian ReLU mean, variance and
response slope Phi(pre_mean/pre_sd). It selects coefficients only; it
does not certify deep control means.

Let muDG15 be its final pre-mean. For vs>0, choose the fixed Gaussian
second-order projection coefficients

    g = Phi(muDG15/sqrt(vs)),
    beta = phi(muDG15/sqrt(vs))/(2*sqrt(vs)).

This is the quadratic response of an affine surrogate
muDG15-ms+s. Its centered polynomial control is

    g*(s-ms) + beta*((s-ms)^2-vs)
      = ell*(s-ms) + beta*(s^2-M2),
    ell = g-2*beta*ms.

The coefficients may be poor, but the displayed centering is exact for
the actual first-layer features. For a deterministic zero-variance s,
the centered controls vanish; divisions above would have to be bypassed.
No calibration labels, sample-dependent coefficients or refit are implied.

## 3. Radial normalization has a limited, exact dominance theorem

For one output let q=s^2/R^2, q0=E[q]=M2/n, mu=E[R],
delta=n-mu^2 and kappa=mu/delta. The chi law gives

    E[R^3]=(n+1)*mu = e3,  E[R^4]=n*(n+2) = e4.

Define the radius-orthogonal quadratic

    D = R^2-n-kappa*(R-mu),
    Qperp = (s^2-M2)-q0*kappa*(R-mu)
          = R^2*(q-q0)+q0*D.

Both Qperp and N=R*(q-q0) are centered and orthogonal to R-mu.
For every degree-one homogeneous Y=R*y(U),

    Cov(Y,Qperp)=e3*Cov(y,q),
    Cov(Y,N)=n*Cov(y,q).

Replacing beta*Qperp by beta*e3/n*N therefore preserves its covariance
with the target and with every other centered degree-one homogeneous
control. Its variance decreases by

    beta^2 * [(e4-e3^2/n)*Var(q) + q0^2*Var(D)] >= 0,
    Var(D)=2*n-mu^2/delta.

Nonnegativity follows from Cauchy--Schwarz and the variance of D. With
the other controls fixed, the estimator variance cannot increase.
The coefficient mapping needs no target-dependent optimization.

IMPORTANT: the unorthogonalized radius coefficient is not unchanged.

    beta*(s^2-M2) + gamma*(R-mu)

maps to

    beta*e3/n*N + (gamma+beta*q0*kappa)*(R-mu).

For the polynomial in section2, retain ell*(s-ms) and use gamma=0 in this
mapping. Omitting the beta*q0*kappa adjustment would not prove the stated
dominance. The result extends to fixed linear combinations of quadratics
by combining their angular q functions.

This fixes radial mismatch, not missing angular variation. The chi-moment
factor e3^2/(n*e4)=1-1/(2*n)+O(n^-2) differs from one by order1/n, but the
full gain also removes q0^2*Var(D). Without a bound on q0^2/Var(q), there
is no universal order1/n bound on the relative estimator improvement.
With only an optimal radius control and this one angular control, and
Var(q)>0, the population residual variance would be

    n * [Var(f) - Cov(f,q)^2/Var(q)].

When Var(q)=0, N is identically zero and is omitted. This population-optimal
expression is a diagnostic identity, not an available fitted coefficient
or the measured variance of section2's fixed control. The actual angular
correlation remains unknown.

## 4. Cost and the missing advancement evidence

Leading classical FP32 setup arithmetic at n=1024:

- One W0 Gram product, fourteen products for A, and one C0*A product:
  sixteen square GEMMs, 34.359738368B FLOPs.
- In addition: the exact n-by-n pair kernel; the entire diagonal-Gaussian
  trajectory; diagonal scalings, coefficient functions, reductions,
  input copies, casts and assembly. These are not yet metered.
- Each sample: all sixteen actual network matrix-vector applications,
  reusing H0, plus A^T H0. Leading subtotal 35,651,584 FLOPs per sample,
  before RNG, input norm, control arithmetic, activations and all-layer
  mean accumulation. Batched fast multiplication needs its own bill.
- Extra setup state includes C0 and A plus buffers; the streaming control
  retains A and vectors. Exact maximum simultaneous buffers, wall time,
  dtype error and memory ceilings have not been qualified.

These are a cost inventory and classical leading subtotal, **not a
complete flopscope cost estimate**, runtime certificate or scored result.
No claim of complete-cost qualification is made.

For IID paths, actual mean residual variance vR, sample count Nsample,
complete setup Csetup and complete per-path c give expected adjusted MSE

    (vR/Nsample) * max(0.1, (Csetup+Nsample*c)/2^41).

Ignoring all setup and extra work already requires vR<6.16809e-5 for
adjusted<1e-9 under the seventeen-matvec convention. Against the archived
rank35 output variance .06679448554 this is over99.907% capture. This is
a requirement, not an observed or predicted magnitude, and not a universal
lower bound on fast batched implementations or randomized QMC.

No retained receipt supplies that capture for this exact fixed control.
The old affine tests are not this construction. The ideal degree12
final-preactivation Gaussian residual 4.41320e-5 belongs to a different
control with missing true moments and cannot be transferred here.

The cheapest numerical falsifier would be a predeclared, fixed-coefficient
paired residual-variance measurement on two development nets, without mean
truths or fitting, together with the same configuration's full cost. It
would establish only a stochastic variance/cost proxy, not independent14
final-score evidence. No positive capture forecast currently justifies
starting it. Required information is the actual angular residual variance
and a complete implementable cost, not another centering identity.

## 5. Remaining routes and engineering cross-check

The two mathematically specific raw routes are (A) this exact-centered
first-layer quadratic control, and (B) the archived high-degree final
preactivation Hermite control with a genuine true-law moment provider.
A has a mean provider but no adequate residual-variance witness; B has
an encouraging ideal-Gaussian residual but no true-law center, bias bound,
actual residual or complete acquisition cost. Neither supports a credible
numerical adjusted-score forecast or a substantial run now. No previously
closed family is reopened by this note.

The separate covariance-support source audit is negative: both current
candidates already crop the active covariance block in _cov_compact
(lines329--339) and retain the cold diagonal contribution. The effective
predictor restores the diagonal after off-diagonal masking, so cold rows
are not guaranteed zero. Removing them is not an exact optimization.
The archived-shape cold-product opportunity is below0.5% of whole cost
even if those products were all free; this is not a current-response meter.
No additional zero-support invariant or meaningful savings was found.

Next advancement requires evidence for an actual raw-times-complete-cost
gain. The already-qualified small response-row component remains available
for combination, without refit or another exception run. Milestone banks
remain below4e-9, below3e-9 and below2e-9, with the final goal below1e-9.

# Weight-space angular readout: explicit coefficient, unpaid source

Phase 2, September 7, 2026 PDT, after23:00. Source/algebra only, no arrays,
model run, sampling, fit, numerical benchmark or submission. Root and the
existing accuracy agent independently checked the formulas. Existing
closeouts and the completed response-row exception are not reopened.

## Result and decision

There is a direct mean readout for a concrete norm-weighted matrix. This is
different from the closed input-chaos controls or covariance-error trace/
mean-ray substitution. However, its acquisition is not supplied by current
moments, and the state is not closed even under a general linear layer.
It therefore does NOT establish an affordable raw-error route yet.

Given the matrix, the terminal1024 quadratic reads take one dense product
and reductions, about2.147B ordinary FP32 dot-product operations before
extras. The source, reference subtraction and remaining angular-error
terms are unpriced. No expected raw improvement, complete candidate bill or
current-head compatibility is claimed. Those are prerequisites to a run,
not details to fill in after reporting a proxy gain.

Official #329926 remains4.982231460183304e-9; qualified local0c21 remains
4.7768418485455745e-9 on independent14, with NCraw2.2458484596872894e-8
and mean complete466705080466.9286F. No new score acquisition here.

## 1. Conditional-on-prefix weight Hermite coefficients

Fix the entire actual prefix H(X), including all of its realized weights.
For ideal fresh Gaussian column w=sigma*xi, sigma=sqrt(2/n), define

    m=E H, tau=E||H||, A=E[H H.T/||H||], tr(A)=tau.

The matrix integrand is defined as zero at H=0. Let
mu(w)=E_X ReLU(w.T H). The degree<=2 Gaussian-weight projection is

    mu_2(w) = .5*w.T*m + sigma*phi0*tau
              + sigma*phi0/2*(xi.T*A*xi-tau).

Derivation: for fixed h, the zeroth coefficient is sigma*phi0*||h||;
E_xi grad_xi ReLU(sigma*xi.T h)=sigma*h/2; and the expected distributional
Hessian is sigma*phi0*h*h.T/||h||. Gaussian Hermite reconstruction gives
the displayed trace-subtracted quadratic. Averaging over H is permitted
by its finite second moment. Also

    mu(w)-mu(-w)=w.T*m,

so higher odd WEIGHT chaoses vanish. This does not say the nonlinear
network has no higher odd INPUT chaos. Nor does it say the truncation has
small error on any specified observed column.

The primary Hermite/dual-kernel background is in sections6 and8 of
[Daniely, Frostig and Singer](https://arxiv.org/html/1602.05897).
The particular conditional-prefix coefficient and its deployment analysis
above are our derivation, not a Phase2 accuracy result from that paper.

The Gaussian ensemble expected squared truncation error is not divided by
the number of output columns. Averaging1024 output SQUARED errors estimates
that expectation; it does not average signed errors before squaring.

## 2. Root's radial-exact spherical variant

For n>=2 write w=r_w*omega, ||omega||=1. Put

    a_n=Gamma(n/2)/(2*sqrt(pi)*Gamma((n+1)/2)).

The spherical degree<=2 projection at each fixed r_w is

    .5*w.T*m + r_w*a_n*
      [tau + (n+2)/(2*(n+1))*(n*omega.T*A*omega-tau)].

Define zero at w=0. Unlike the Gaussian polynomial, this expression is
exactly degree-one homogeneous in w. It removes a radial approximation,
not the higher even ANGULAR terms or the missing A acquisition.

Root derived the coefficient from a uniform sphere coordinate t:

    E[t_+]=a_n,
    E[t^3 1(t>0)]=2*a_n/(n+1),
    Var(t^2)=2*(n-1)/(n^2*(n+2)).

The coefficient multiplying t^2-1/n is consequently
a_n*n*(n+2)/(2*(n+1)). The accuracy agent independently checked it.
No numerical experiment or degree/rank sweep followed this identity.

The spherical statement uses uniform directions, the Gaussian statement
uses continuous ideal Gaussian weights. The installed generator rounds
Gaussian weights to float32. One can evaluate either expression at those
observed weights, but an exact orthogonality/error statement for the ideal
law does not automatically transfer to the rounded generator.

## 3. Conditioning on the mean direction preserves the gate, but adds a field

When m!=0, let u=m/||m|| and w=a*u+sigma*xi_perp. For
t=u.T H, V=(I-u*u.T)H and r=||V||, the degree<=2 expansion in the transverse
Gaussian weight coordinates is

    K0(a) + sigma*xi_perp.T*b(a)
      + sigma/2*(xi_perp.T*A(a)*xi_perp-tr(A(a))),

where

    K0(a)=E GaussianReLUMean(a*t, sigma^2*r^2),
    b(a)=E[Phi(a*t/(sigma*r))*V],
    A(a)=E[phi(a*t/(sigma*r))*V*V.T/r].

At r=0 use ReLU(a*t) in K0, zero in b/A. This follows from the same
first/second Gaussian-weight derivative identities. It keeps the observed
parallel weight a fixed, instead of approximating its gate with a low-degree
polynomial. But it needs an a-indexed field, not one fixed extra matrix.
Higher odd transverse chaoses generally return because a is held fixed.

No current moment or fixed angular-energy label specifies these functions
of the ratio t/r. Simply declaring the conditional residual Gaussian or
reusing ordinary C would require a separate approximation/value argument.

## 4. The missing recurrence appears already at a linear layer

Define the finite angular measure

    nu(dtheta)=E[||H|| 1{H/||H|| in dtheta}],

with no mass assigned by H=0. Its first three angular moments are
tau=integral nu, m=integral theta dnu and A=integral theta*theta.T dnu.
For a general linear push Y=W.T H,

    tau_pre=integral ||W.T theta|| dnu,
    A_pre=integral W.T*theta*theta.T*W/||W.T theta|| dnu.

Again define the matrix integrand as zero when W.T theta=0. In particular,
A_pre is NOT W.T A W. The changing denominator depends on angular
information beyond those three moments. A ReLU adds further dependence.
Consequently the terminal product price does not establish a cheap
layerwise carrier or authorize pricing this as one covariance trajectory.

Actual-prefix sampling could acquire contractions rather than the full
angular tensor, but all prefix forwards and its sampling variance remain
paid. No such qualified variance/source receipt was found or generated.

## 5. Current-bank ownership and old scope

Root read both CONDITIONAL_WEIGHT_VALUE_ATTRIBUTION files. Their rank35
coherent8M cells8..14 tested the trace/mean-ray conditional quadratic action
of a fixed covariance error, with an approximate weight-dependent response
factor. Residual ratio.9988424362589163 and mean-ray value energy
.0007666375523621658 fail their gates. That is not this norm-weighted A,
but it is not positive evidence for it either.

ANGULAR_LABEL_TANGENT_AND_INNOVATION_SCOPE_V1.md supplies a fixed centered
energy-label c,K initializer and Gaussian weighted-moment tangent. Neither
is A, A(a), or their nonlinear recurrence. A ratio-weighted angular moment
cannot be inferred just from a low-order polynomial energy label.

A new correction to the all-P bank must subtract the SAME-reference owned
coefficient before adding a new one. Subtracting only a Gaussian mean/C
coefficient while retaining the bank's cumulant responses and learned head
does not establish this ownership. The head itself is not a supplied
probability law whose norm-weighted A can be read for free.

The decisive missing evidence is a fixed affordable supplier of the signed
true-minus-owned coefficient, its remaining angular truncation error on
actual output columns, and a same-configuration complete bill. This audit
identifies a direct readout and an explicit nonclosure; it does not launch
an oracle, training job, network run or parameter rescue.

# Reflected centered-ReLU marginal inversion: target-free algebra

Phase 2 research, September 6, 2026. This is a scalar reference-law
construction, not a measured reconstruction of the network's joint law.
It changes no bank, source protocol, target seal or submission.

Let Y=(Z+alpha)+, Z standard normal, and let m_j=E[Y^j], v=m_2-m_1^2,
k=m_3-3*m_1*m_2+2*m_1^3, gamma=k/v^(3/2). Write g=Phi(alpha), Q=1-g.
Differentiation under the integral gives

    m_1'=g, m_2'=2*m_1, m_3'=3*m_2,
    v'=2*m_1*Q,
    k'=3*Q*(v-m_1^2),
    gamma'=3*Q*(m_2^2-m_1*m_3)/v^(5/2) < 0.

The strict sign follows from Cauchy–Schwarz applied to Y^(1/2) and
Y^(3/2); conditional on Y>0 the distribution is not a point mass.
As alpha tends to plus infinity gamma tends to zero from above. As alpha
tends to minus infinity, the positive tail moments are asymptotic to
j!*phi(|alpha|)/|alpha|^(j+1), so gamma tends to infinity. Hence every
strictly positive skew has a unique finite alpha. A finite numerical
bracket is an implementation domain, not a restriction of this theorem.

For requested nonzero signed skew gamma_target, eta=sign(gamma_target),
choose alpha from gamma(alpha)=abs(gamma_target) and tau=sigma/sqrt(v).
Then

    f(z)=eta*tau*((eta*z+alpha)+-m_1(alpha))

is centered, has variance sigma^2 and the requested signed skew. It is
monotone increasing for either eta. Zero skew uses the Gaussian limit
f(z)=sigma*z. Adding a supplied mean centers the auxiliary law as requested,
but may give support on negative values: this is NOT asserted to be the
literal nonnegative post-ReLU distribution.

The first two unnormalized Hermite responses are

    E[f(Z)*Z]=tau*Phi(alpha),
    E[f(Z)*(Z^2-1)]=eta*tau*phi(alpha).

Those signs and the later tree/trace/star affine action have an independent
synthetic certificate in `reflected_copula_tree_action_certificate_v1.json`.
It does not certify covariance inversion, latent positive semidefiniteness,
the omitted triangle/cycle terms or source accuracy.

## Stable scalar spelling

Set t=abs(alpha), p=phi(t), Q=Phi(-t), r=p-t*Q. For alpha>=0:

    mean=alpha+r,
    variance=1-Q-t*r-r*r,
    kappa3=(t*t-1)*r+2*t*Q+3*r*Q+3*t*r*r+2*r*r*r.

For alpha<0:

    mean=r,
    variance=Q-t*r-r*r,
    kappa3=(t*t+2)*r-t*Q-3*r*Q+3*t*r*r+2*r*r*r.

These avoid subtracting O(alpha^3) raw moments when positive alpha is large.
Negative-tail cancellation still needs numerical qualification and cannot
be called harmless at arbitrary alpha or precision. No silent clipping,
PSD repair, enlarged inversion bracket or target-selected tolerance is
authorized by this note.

## Synthetic numerical certificate to be run once

`certify_reflected_relu_marginal_v1.py` compares these moments with
independently integrated positive tail moments at ten fixed alpha values
from -8 to 12, checks the exact derivative identity, and recovers seven
fixed synthetic alpha values by 64 bisection steps in [-8,12]. Relative
moment tolerance1e-8; alpha recovery tolerance1e-10. The script is inert
without --execute and refuses to overwrite its receipt. No real network,
target array, fit or production timing is included.

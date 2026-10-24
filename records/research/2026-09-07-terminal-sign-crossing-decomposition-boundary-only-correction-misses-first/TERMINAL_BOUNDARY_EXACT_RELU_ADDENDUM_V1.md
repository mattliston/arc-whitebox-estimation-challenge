# Exact bias-free ReLU counterexample to boundary-only inference

Addendum to `TERMINAL_BOUNDARY_RESIDUAL_SOURCE_AUDIT_V1.md`, whose SHA256 is
`ec578f8afbc4a8322b85c096b2f9a314e83c9cc4df204f46de4625c8d6fd74ee`.
Root proposed this piecewise-linear variant; the accuracy agent checked
the identities below independently by algebra. No numerical execution,
network/target arrays, or claim about actual Phase2 weights.

Let X have independent standard-normal coordinates, n>=2, and define

    A = X_1,
    q = sign(X_1) * min(|X_1|, |X_2|),
    e = epsilon*(q-A/2),       a=A+e,
    0 < epsilon < 2.

The exact clamp representation is

    q = ReLU(X_1+|X_2|) - ReLU(X_1-|X_2|) - |X_2|,
    |X_2| = ReLU(X_2)+ReLU(-X_2).

This is positively homogeneous and piecewise linear and has a finite
bias-free ReLU implementation, with signed linear output weights and
ordinary positive/negative channels to carry X_1 between layers. Thus,
unlike the preceding smooth toy, it can literally be a ReLU-network
preactivation. It is not a claim about the random challenge networks.

Oddness in X_1 gives E[q]=0. Gaussian integration by parts for this
Lipschitz piecewise-linear function gives

    E[X_1 q] = E[partial_1 q] = P(|X_1|<|X_2|) = 1/2.

The tie has probability zero. Oddness/parity and independence give
E[X_j q]=0 for j!=1. Since

    q^2 = min(X_1^2,X_2^2),
    X_1^2-X_2^2 = (X_1+X_2)(X_1-X_2),

and the latter factors are independent N(0,2),

    E[q^2] = 1 - (1/2) E|X_1^2-X_2^2| = 1-2/pi.

Therefore

    E[e]=0,     E[X_j e]=0 for every j,
    Var(e)=epsilon^2*(3/4-2/pi),
    Var(a)=1+epsilon^2*(3/4-2/pi).

But a=(1-epsilon/2)A+epsilon*q has exactly A's sign, since both
coefficients are positive and q has A's sign. Hence the exact terminal
crossing remainder B=|a| 1[A*a<0] vanishes identically.

Write phi0=1/sqrt(2*pi). A two-dimensional polar integral yields

    E[min(|X_1|,|X_2|)] = phi0*(4-2*sqrt(2)),
    E[q*1(A>0)] = phi0*(2-sqrt(2)).

Consequently the complete ReLU mean is

    E[e*1(A>0)] = epsilon*phi0*(3/2-sqrt(2)) != 0,
    E[ReLU(a)] = phi0*[1+epsilon*(3/2-sqrt(2))].

The affine projection is unchanged, its unexplained variance is quadratic
in epsilon, and there is no terminal gate switch at any epsilon in the
stated interval; nevertheless the ReLU mean changes linearly in epsilon.
The moment-matched Gaussian mean changes only quadratically. A method that
adds only curvature/sign-crossing information therefore misses the leading
term, even for an exact bias-free ReLU preactivation.

This is not a raw-performance result, nor a proof against a method that
actually supplies the signed conditional expectation E[e|A]. The particular
toy is analytically integrable because q was specified explicitly; no
comparably cheap exact conditional supplier for the actual deep network is
provided by it. The prior note's no-launch/source-and-cost conclusion is
unchanged.

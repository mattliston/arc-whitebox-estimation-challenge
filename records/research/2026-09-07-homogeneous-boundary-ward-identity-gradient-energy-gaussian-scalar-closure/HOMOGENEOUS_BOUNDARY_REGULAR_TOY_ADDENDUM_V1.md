# Ward audit addendum: a regular nonlinear homogeneous boundary

Phase2 source/algebra audit, September6,2026 PDT. No network/target arrays,
numeric execution, random samples, fit or estimator. The frozen parent
`HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md` is unchanged (SHA256
`2039e57d3bf939ad83ea2ede2b8c55b4acea0f3c778cf3b1f1729657e4577893`).

## Independent parent review and the example's precise qualification

The parent's smooth Ward identities, conditional conservation equation and
Gaussian scalar-closure cancellation are correct with their stated
integrability assumptions. Its caution about signed distributional ReLU
facet measures and coincident zero facets is necessary and retained.

The parent example `(x1^2-x2^2)/||x||` correctly checks the unconditional
identity and its exact positive-part mean. It does NOT instantiate a regular
finite-density boundary for n>=3. In orthogonal coordinates A,B, its numerator
is2AB. Fix the remaining coordinates in an annulus away from zero. Near the
intersection A=B=0, the denominator is bounded and nonzero, and the zero-level
coarea density contains an integral proportional to `dA/|A|`. Consequently
`p_a(0)` diverges logarithmically. This is a density singularity, not an atom.

The gradient-energy-weighted boundary integral can still be finite: the
vanishing gradient cancels the coarea singularity. But it must not be written
as an unqualified product of a finite `p(0)` and an ordinary conditional
energy. The parent's section4 remains an unconditional check only.

## A regular replacement for the section2 boundary check

Fix n>=3 and `|epsilon|<1`, and let

```
a(x) = x1 + epsilon*r,       r=||x||,
X=R U,                     k=E[chi_n].
```

This is degree-one homogeneous, globally Lipschitz and smooth away from the
origin. It is nonlinear when epsilon is nonzero, but is not proposed as a
literal Phase2 network or a numerical surrogate. Direct differentiation gives

```
grad a = e1 + epsilon*U,
b = Delta a = epsilon*(n-1)/R,
t = ||grad a||^2 = 1 + 2*epsilon*U1 + epsilon^2.
```

The origin contributes no extra delta term: boundary flux of the bounded
gradient through a radius-delta sphere is `O(delta^(n-1))`. The Laplacian's
`1/R` singularity is Gaussian-integrable, as are the required products.
Gaussian tails control infinity. On the nonzero cone `a=0`, `U1=-epsilon`
and `||grad a||^2=1-epsilon^2>0`; the level boundary is regular.

Using `E[R^2]=n`, `E U1=0` and `E U1^2=1/n`,

```
mu       = epsilon*k,
E[a^2]   = 1 + epsilon^2*n,
v        = 1 + epsilon^2*(n-k^2),
tbar     = 1 + epsilon^2,
E[a*b]   = epsilon^2*(n-1),
c        = Cov(a,b) = epsilon^2*(n-1-k^2).
```

The chi recurrence `E[1/R]=k/(n-1)` also gives `E b=mu`. Strict
Cauchy--Schwarz gives `k*E[1/R]>1`, hence `k^2>n-1`. Therefore c is strictly
negative for nonzero epsilon, and `c+tbar=v` exactly. No covariance sign
assumption is being smuggled into the Ward closure.

## Finite density and the exact positive-part mean

Write

```
f_U(u) = c_n*(1-u^2)^((n-3)/2),   -1<u<1,
c_n = Gamma(n/2)/(sqrt(pi)*Gamma((n-1)/2)),
F_n(epsilon) = P(U1 > -epsilon)
             = 1/2 + integral_0^epsilon f_U(u) du.
```

Conditioning on R shows

```
p_a(0) = E[1/R] f_U(-epsilon)
       = phi(0)*(1-epsilon^2)^((n-3)/2),
tau(0) = E[t | a=0] = 1-epsilon^2.
```

Both are finite. The two terms of the parent's boundary formula are

```
E[1[a>0] b] = epsilon*k*F_n(epsilon),
p_a(0)*tau(0) = phi(0)*(1-epsilon^2)^((n-1)/2).
```

Independently integrate the target itself:

```
E a_+ = k E[(U1+epsilon)_+]
      = epsilon*k*F_n(epsilon)
        + [k*c_n/(n-1)]*(1-epsilon^2)^((n-1)/2).
```

The gamma recurrence gives `k*c_n/(n-1)=phi(0)`. Thus this direct integral
equals the two boundary terms exactly, providing a regular nonlinear check
of the boundary coefficient and sign without samples or numerical quadrature.

The conditional fields also satisfy the exact relation

```
beta(s) = epsilon*(n-1)*E[1/R | a=s],
tau(s)  = 1-epsilon^2 + (2s/(n-1))*beta(s).
```

In particular `tau(0)=1-epsilon^2` differs from `tbar=1+epsilon^2` by
`2*epsilon^2`. This is an explicit example of the joint conditional information
that an unconditional scalar energy cannot supply. It does not authorize
substituting the boundary energy alone into an otherwise inconsistent
Gaussian closure; beta, tau and p must still obey the same conservation law.

For a compact exact specialization, n=3 gives `k=4*phi(0)`,
`F_3(epsilon)=(1+epsilon)/2`, `p_a(0)=phi(0)` and
`E a_+=phi(0)*(1+epsilon)^2`. No epsilon or dimension sweep was run.

## Endpoint

This addendum strengthens the parent's regularity example, not its estimator
claim. The scalar Gaussian closure still collapses exactly to the Gaussian
ReLU mean. Neither this toy nor the singular harmonic example supplies a
cheap boundary-conditioned derivative field for the real network.

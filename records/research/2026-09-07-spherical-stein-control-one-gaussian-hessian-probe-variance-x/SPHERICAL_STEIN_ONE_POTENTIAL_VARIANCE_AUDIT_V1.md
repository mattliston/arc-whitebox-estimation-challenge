# Spherical Stein control: exact recurrence and a trace-noise obstruction

Phase2, source/algebra only, 2026-09-06 PDT. No network arrays, targets,
sampling, fits, numerical certificate, candidate or meter. This is a new
mathematical audit, not an experiment authorization.

## Verdict

The proposed conditional-on-radius centering is correct and has a cheap
all-output directional implementation. It avoids the large Gaussian-Stein
drift-probe noise. However, **one Gaussian Hessian-trace probe leaves at
least approximately half of every nonlinear spherical-harmonic component,
even with oracle potential coefficients and an optimally chosen radial
extension**. More independent probes do not beat plain MC on those components
at the declared classical dense circuit cost. This is stronger than arguing
that one smoothing temperature cannot match a broad spectrum.

Rademacher probes have a fixed-coordinate exception: diagonal Hessians have
no trace noise. The universal Gaussian bound cannot simply be copied to
them. An exact Haar-orientation average retains almost the same bound, but
does not prove a bound for an individually chosen fixed network/basis.
No arbitrary smoothing-temperature experiment is recommended.

## 1. Exact centering, including the projector's dependence on x

Condition on `r=||x||>0`, put `m=n-1` and `P=I-xx^T/r^2`. Let `u(x)` be a
smooth scalar potential; apply the same construction coordinatewise for all
network outputs. For its restriction `G` to the radius-r sphere,

```
Delta_S G = tr(P Hess(u) P) - (m/r^2) x^T grad(u).
```

The sphere is closed, so its uniform integral is zero. For an independent
probe with `E[zz^T]=I`, set `v=Pz` and

```
S(x,z) = (r^2/m) v^T Hess(u(x)) v - x^T grad(u(x)).
```

Then `E_z[S | x]=(r^2/m) Delta_S G` and `E[S | r]=0`. Thus `f+S` has exactly
the same expectation as f under the ideal Gaussian input law. Any additional
deterministic control coefficient may be absorbed into u.

The tangent vector v is FROZEN when taking its second directional derivative.
Differentiating `P(x)z` again inside the JVP would change the operator. The
displayed curvature drift already accounts for the projector's dependence
on x. Likewise the radial JVP uses the frozen seed x to evaluate
`x^T grad(u)`. No derivative of a random probe distribution is needed.

The potential must be fixed independently of the correction sample/probe,
or fitted with a separately justified cross-fitting scheme and full cost.
The centering theorem alone does not authorize sample-dependent fitting.

## 2. Complete directional recurrence for a literal smooth near-network

For fixed positive, weights-only temperatures `t_l` (scalar or per-unit),
replace each ReLU by

```
phi_t(a) = a Phi(a/t) + t phi(a/t),
g_t(a) = Phi(a/t),       h_t(a) = phi(a/t)/t.
```

Initialize four streams `(h, h_r, h_v, h_vv)=(x,x,v,0)`. At each layer,
using the same affine orientation as the true network,

```
a     = W^T h
a_r   = W^T h_r
a_v   = W^T h_v
a_vv  = W^T h_vv

h     = phi_t(a)
h_r   = g_t(a) * a_r
h_v   = g_t(a) * a_v
h_vv  = g_t(a) * a_vv + h_t(a) * a_v^2.
```

Final `h_r` and `h_vv` supply the two terms of S for every output. The actual
ReLU path is a separate fifth stream. No full input Jacobian or per-output
Hessian is stored. If temperatures depend on x, these recurrences are no
longer complete unless all their derivatives are added.

With q independent probes averaged in the trace, the smooth forward and
radial streams are shared; each probe adds a first- and second-directional
stream. The nominal count is `3+2q` path equivalents including the actual
path. At the first layer, actual/smooth/radial affine products are identical
and each zero-seed second-directional product can be skipped. Therefore the
literal depth16 dense matrix-vector subtotal is

```
16(3+2q) - (2+q) = 46+31q matrix-vector products.
```

For q=1,n=1024 this is77 products, or161,401,856 scalar multiplication/addition
FLOPs under the `n(2n-1)` convention. This is an OPTIMISTIC subtotal, not a
metered full bill. Extra work includes input/probe RNG; norms, projections
and reductions;16 true ReLUs;16 smooth PDF/CDF evaluations and value/radial
maps; and the directional products/squares for each probe. Source temperatures
and any fit must also be acquired. No complete native runtime claim is made.

Memory is linear in batch size times width times the stream count, besides
the16 weight matrices. Current/next-layer buffers suffice; no depth-long
autograd tape or n-by-n Jacobian is necessary. All unknown elementwise/RNG
costs are nonnegative and hence cannot weaken the lower-cost comparison below.
Fast matrix multiplication or alternative structured trace rules would need
their own complete accounting; the dense circuit count is not a universal
arithmetic lower bound on every possible implementation.

## 3. A Gaussian-probe obstruction independent of the smoothing choice

All expectations in this section are over the radius-r uniform sphere unless
the probe is explicitly included. Write

```
A = P Hess(u) P
H = Hess_S G
A = H + a(x) P,       a(x)=x^T grad(u)/r^2.
```

Changing the radial extension of G changes a but not the intrinsic Hessian.
Its trace-free part `H0=H-(Delta_S G/m)P` therefore obeys

```
||A||_F^2 >= ||H0||_F^2.
```

For Gaussian z, `Var_z(z^T A z | x)=2||A||_F^2`. Averaging q independent
probes divides this variance by q. This lower bound already grants an
optimally chosen radial extension, even if the literal network cannot
realize it.

Use spherical harmonics `G=sum_l G_l`, with
`L_l=l(l+n-2)` and `Delta_S G_l=-(L_l/r^2)G_l`. The integrated Bochner identity
gives

```
E ||Hess_S G||_F^2
 = E (Delta_S G)^2 - ((m-1)/r^2) E ||grad_S G||^2.
```

This follows by integrating Bochner's formula and using the sphere's Ricci
curvature `(m-1)/r^2`; see [Dai and Wei, section1.1, theorem1.1.1](https://web.math.ucsb.edu/~dai/Ricci-book.pdf).
Subtracting the squared trace and using orthogonality yields, for each l,

```
E ||Hess_S^0 G_l||_F^2
 = (m-1) L_l (L_l-m) / (m r^4) * E G_l^2.
```

Consequently the probe-noise energy relative to the squared mean-control
energy `(L_l^2/m^2) E G_l^2` is at least

```
rho_l/q,     rho_l = 2(m-1)/m * (1-m/L_l),       l>=1.
```

This is zero for l=1: a linear potential needs no Hessian trace at all. For
l>=2 it is substantial. At n=1024,
`rho_2=.9999980907869`; it increases toward `2(m-1)/m`.

Now let the centered target restricted to the sphere be `F=sum_(l>=1) F_l`.
Probe noise is zero-mean conditional on x, so it is orthogonal to every
target or conditional-mean-control term. Even optimizing G separately in
every harmonic, completing the square gives

```
Var_{x,z}(F+S | r)
 >= sum_(l>=1) [rho_l/(q+rho_l)] * ||F_l||_2^2.
```

The harmonic sum is valid because both the deterministic residual and the
integrated trace-free Hessian quadratic form diagonalize in these
eigenspaces. It is not an independence assumption between layers or neurons.
For q=1, the l=2 retained fraction is at least .4999995226963, and higher
degrees retain more. This remains true with an exact spherical Poisson
potential; an ideal inverse eigenvalue filter removes conditional-mean
mismatch, not the sampled Hessian noise.

## 4. More independent probes cannot rescue this dense circuit

Relative to the16-product plain forward, the optimistic per-harmonic
variance-times-cost multiplier is at least

```
M_l(q) = ((46+31q)/16) * rho_l/(q+rho_l).
```

For n=1024,l>=2,q>=1, its infimum is at least
`(31/16)rho_2 = 1.9374963009`. Additional derivative/RNG work only increases
it. The simpler unshared `(3+2q)` count gives the approximately2 bound, but
the77-product first-layer reuse must not be omitted in a complete census.

This comparison uses independent sample/probe averages and the same dense
matrix-product convention for both estimators. It does not prove a lower
bound for an arbitrary RQMC rule, correlated exact trace design, basis-aware
Hessian integration or hypothetical faster surrogate circuit.

Conditioning on radius also leaves radial variation of the target's sphere
mean. For a bias-free degree-one target `f(RU)=R f(U)`, the exact Gaussian
variance decomposition is

```
Var f(X) = E[R^2] sum_(l>=1) ||f_l(U)||^2
           + Var(R) * (E_U f(U))^2.
```

The l=1 term is the Gaussian-input linear projection; the last term is the
known input-radius contribution. Thus this theorem concerns the remaining
nonlinear angular energy, not every part of the Gaussian Hermite expansion
interchangeably. No new numerical spherical-spectrum claim is made here.

## 5. Rademacher qualification and the precise surviving exception

For independent Rademacher coordinates,

```
Var_z(z^T A z) = 2 (||A||_F^2 - sum_i A_ii^2).
```

A fixed coordinate basis with nearly diagonal A can evade the Gaussian
Frobenius lower bound. This is a genuine scope limitation, not a demonstrated
property of the current random-network Hessians.

For a fixed symmetric A subjected to a Haar-uniform orthogonal change of
coordinates Q, elementary fourth sphere moments give

```
E_Q Var_z(z^T Q^T A Q z)
 = 2/(n+2) * [n ||A||_F^2 - (tr A)^2].
```

Since `A=H0+cP` with `tr H0=0` and `rank P=m=n-1`, this is at least
`2n/(n+2) ||H0||_F^2`. Hence the orientation-averaged version of the bound
retains `rho_l` multiplied by `n/(n+2)=.9980506823`.

Applying that average to a random-network family requires the target and
potential to rotate together while the probe coordinate basis remains
fixed; a literal smoothing with rotation-invariant, weights-only
temperatures has that covariance. It does not permit replacing an
individual network's diagonal energy by its Haar expectation, or optimizing
a new basis/potential after each rotation and reusing the same proof.
No basis acquisition, temperature sweep or trace-variance run is authorized.

## Novelty and endpoint

The older Gaussian-input Stein/OU source audit correctly lacked an affordable
high-capture control. This spherical formula is a genuinely cheaper exact
centering circuit, not a repeat of that missing-Jacobian claim. The new
Bochner/probe result explains why its cheapness does not supply the required
raw accuracy. It closes the stated independent-Gaussian-probe, one-potential
sampling route at its declared dense cost, rather than claiming all Stein
fields or all stochastic estimators are impossible.

An action-specific, cheaply diagonalizable tangent Hessian could fall outside
this closure, but no such state is identified. A single arbitrarily chosen
smooth near-network potential is therefore not promoted to an experiment.

Root independently checked the quadratic-harmonic instance and arithmetic
constants using exact rational arithmetic in
`certify_spherical_stein_constants_v1.py` (SHA256
`53828c296325679ed4151234aa159551e1c20e69651d89abc750b12f5945f224`).
The retained `spherical_stein_constants_certificate_v1.json` passes on
n=3,4,8,16,64,1024 and gives the exact infimum `523775/270336` at n=1024.
This agent read both files without executing them. The certificate does NOT
prove the general Bochner identity, measure any actual network variance,
or claim the fixed/adapted-basis Rademacher extension.

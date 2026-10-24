# Homogeneous boundary-mean identity: the scalar energy closure is redundant

Phase 2, September 6, 2026 PDT. New source/algebra audit, not a network
experiment, score, candidate, or permission to vary a closed arm. Root
derived the identities below independently while the two agents audited
conditional-factor and terminal-dependency alternatives. No target arrays
were read and no numerical certificate is claimed by this document.

## Question and answer

Can the exact relation between a bias-free network's mean and its input
Laplacian give a new, cheap mean estimator by carrying a per-neuron input
gradient energy in addition to its mean and variance?

Not with the simple Gaussian-regression closure. A required cross moment
cancels the extra gradient energy exactly, leaving the ordinary Gaussian
ReLU mean. Omitting that cross moment creates an inconsistent formula,
not new accuracy information. More general boundary-conditioned derivative
fields are genuinely different, but have no closed inexpensive supplier
in this proposal.

This is a scoped negative result about that scalar closure. It does not
claim all integration-by-parts or boundary-integral estimators fail.

## 1. Exact identities and regularity

Let X be standard normal in R^n. First take a real degree-one positively
homogeneous function a that is C^2 away from the origin, with sufficient
Gaussian integrability and origin-boundary decay for integration by parts.
Write

    b(X) = Delta a(X),       t(X) = ||grad a(X)||^2,
    mu = E a,               v = Var(a).

Euler's identity gives X dot grad a = a. Gaussian integration by parts,
first with grad a and then with a grad a, therefore gives

    E b = mu,
    E[a b] + E t = E[a^2].

Consequently, with c = E[(a-mu)b],

    c + E t = v.                                      (1)

In particular c need not be zero or positive. Input gradient energy is
not interchangeable with activation variance for a nonlinear homogeneous
prefix. Under the same assumptions, any admissible differentiable psi
satisfies the more informative identity

    E[a psi(a)] = E[b psi(a)] + E[t psi'(a)].           (2)

For polynomial psi(a)=a^k this reads

    E[a^(k+1)] = E[a^k b] + k E[a^(k-1) t].

The right side is new conditional derivative information unless supplied
by the same moment/density closure; it is not determined by E t alone.

Literal ReLU prefixes require additional care: their ordinary Hessian is
zero almost everywhere but their DISTRIBUTIONAL Hessian is a signed measure
on activation facets. Replacing b by an autodiff Hessian that ignores those
facets invalidates the identities. For a piecewise-linear prefix, the first
two identities can instead be interpreted as Gaussian-weighted pairings
with the distributional Laplacian. Multiplication by the continuous a in
E[a b] is well-defined. An additional composition with a new ReLU requires
the appropriate measure-valued chain rule at coincident zero facets; no
unverified convention such as 1[a>0] times a zero-boundary atom is assumed.

## 2. Boundary formulation and the missing conditional field

In the regular case, assume the law of a has a density p near zero and that
the boundary is regular enough for the following coarea/chain formula:

    E relu(a) = E[1[a>0] b] + p(0) E[t | a=0].         (3)

One derivation is to use the distributional chain rule

    Delta relu(a) = 1[a>0] Delta a + delta(a)||grad a||^2

and apply Euler plus Gaussian integration by parts to relu(a). Formula (3)
has the qualifications just stated; degeneracies and boundary atoms need
their own treatment in a literal ReLU implementation.

Put beta(s)=E[b|a=s] and tau(s)=E[t|a=s], when these functions exist. The
weak form (2) is exactly the one-dimensional conservation relation

    s p(s) = beta(s) p(s) - d/ds [tau(s)p(s)].         (4)

If the tail boundary term vanishes, integrating (4) over s>0 gives (3).
Thus obtaining beta from an assumed p and tau, then using (3), merely
reproduces the positive-part integral of that same assumed p. It does not
repair the assumed marginal law by itself.

For piecewise-linear a, beta(s)p(s) should generally be treated as a
pushforward signed measure of Delta a rather than casually assuming b is
an ordinary sampled random variable. The smooth calculation identifies
what information is required; it is not a cheap ReLU facet sampler.

## 3. Exact collapse of the proposed scalar Gaussian closure

Approximate a by N(mu,v), let sigma=sqrt(v), alpha=mu/sigma and

    p0 = phi(alpha)/sigma,       g = Phi(alpha).

Use the affine regression closure and constant conditional energy

    beta(s) = mu + (c/v)(s-mu),
    tau(s) = tbar,               tbar = E t.

The two terms in (3) become

    integral_(0,infinity) beta(s)p_G(s) ds = mu*g + c*p0,
    p_G(0) tau(0) = tbar*p0.

Enforce the exact moment relation (1). Their sum is

    mu*g + (c+tbar)*p0 = mu*Phi(alpha) + sigma*phi(alpha).

This is the ordinary Gaussian ReLU mean, for EVERY value of the supplied
gradient energy. It also satisfies the full Gaussian form of (4), since
beta(s)=s-(tbar/v)(s-mu). Paying to propagate tbar cannot improve this mean.

Setting c=0 while substituting a non-Gaussian tbar for v breaks (1).
Calling the discrepancy a Ward correction would hide this inconsistency.
Conversely an independently approximated boundary tau(0), non-affine
beta, or a non-Gaussian p can change the mean, but their joint centered
law and propagation must be supplied and priced explicitly.

## 4. Independent analytic nonlinear example

For n>=3 consider the homogeneous, smooth-away-from-origin function

    a(x) = (x_1^2-x_2^2)/||x||.

It is not a literal ReLU network example or a claim about Phase 2 variance.
Write X=R U, R chi_n and U uniform on the unit sphere, and
Y(U)=U_1^2-U_2^2. Y is a degree-two spherical harmonic with eigenvalue 2n.
Elementary sphere moments give E Y=0 and E Y^2=4/[n(n+2)]. Hence

    mu = 0,
    v = E R^2 E Y^2 = 4/(n+2),
    E t = E[Y^2 + ||grad_S Y||^2] = 4(2n+1)/[n(n+2)],
    E[a b] = (n-1-2n) E Y^2 = -4(n+1)/[n(n+2)].

The latter two quantities sum exactly to v, checking both the coefficient
and sign in (1). At large n, gradient energy is about twice the variance,
with a compensating negative cross moment.

For an independent check of why the Gaussian approximation remains an
approximation, write U_1^2-U_2^2=(U_1^2+U_2^2)cos(2 theta). Rotational
symmetry makes theta uniform and independent of U_1^2+U_2^2. Therefore

    E|Y| = (2/n)(2/pi) = 4/(pi n),
    E relu(a) = 2 E[chi_n]/(pi n).

The scalar closure above instead returns

    sqrt(v)/sqrt(2*pi) = 2/[sqrt(n+2)*sqrt(2*pi)].

The difference is the non-Gaussian angular law. Merely adding the exact
unconditional gradient energy cannot remove it under the stated closure.

## 5. Relation to inspected prior work and next action

The prior coupled-control and spherical-Stein notes studied zero-mean
stochastic corrections and their derivative/probe costs. This note instead
examines a direct deterministic mean recursion. Its no-op conclusion is
algebraic and does not rely on the Gaussian-probe variance bound.

The organizer [paper](https://arxiv.org/html/2605.05179v1), sections 1 and
4, was read in the current turn. It distinguishes fixed-weight input
averages from weight-ensemble averages and propagates cumulants through
nonlinearities. It does not supply the conditional gradient/facet fields
needed to turn (3) into a cheaper fixed-weight estimator. This note's
specific boundary-closure derivation is independent, not attributed to
that paper.

No network run, scalar energy fit, Hessian sampling, Ward-residual feature
fit, or candidate rewrite follows from this identity. A materially distinct
boundary method would need a correct nonconstant conditional-field source,
all facet terms, a bounded accuracy test, and a full inference-cost model.

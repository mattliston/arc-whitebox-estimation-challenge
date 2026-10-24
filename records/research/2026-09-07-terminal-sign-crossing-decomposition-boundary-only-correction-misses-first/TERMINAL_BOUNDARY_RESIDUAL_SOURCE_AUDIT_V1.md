# Terminal boundary residual: the missing term is not just curvature

Phase 2 only. Recorded September 6, 2026 PDT; clock checked September 7,
01:46:47 UTC. Source/algebra audit only: no numerical execution, network or
target arrays, fit, sampling, candidate, meter, or submission. Existing
sources and receipts are unchanged.

## Conclusion

**No new complete, affordable raw-error mechanism is substantiated.** A
terminal sign-crossing correction is a precise possible decomposition, but
it does not supply its own essential signed first-order expectation. The
current cheap state supplies neither that conditional expectation nor a
certified low-variance residual. Splitting an existing shifted-ridge control
into two pieces is not a new estimator unless one piece acquires a genuinely
cheaper, correctly centered source.

This is not a universal impossibility theorem for conditional integration.
It is a no-launch conclusion for the specified terminal-curvature idea with
the inspected sources. No temperature, direction, rank, suffix, or sample
sweep follows.

The newly completed fixed Dirichlet probe is separately closed: its saved
`dirichlet_gram_probe_pilot_v1/result.json` has qualification PASS but source/
parent ratio 1.00093491602506, source/control 0.99684985081411 and 2/4 parent
wins on held-out ranks37/39/42/48. All three scientific gates fail. This
audit does not reinterpret that result as an inconclusive fit.

## 1. Exact terminal decomposition and its ownership

Let X be standard Gaussian, A=m+p^T X a fixed, nondegenerate affine
reference, a(X) the actual final preactivation, e=a-A, and G=1[A>0]. Then

    ReLU(a) = ReLU(A) + G e + B(A,e),
    B(A,e) = |A+e| 1[A(A+e)<0],
    0 <= B(A,e) <= |e| 1[|A|<=|e|].

The identities hold almost surely because P(A=0)=0. Thus

    E ReLU(a) = E ReLU(A) + E[G e] + E[B].             (1)

The first expectation is exactly Gaussian-integrable. The last term is
localized to actual/reference sign disagreements. But the middle term is
signed and generally first order in e. Neither E[e]=0 nor E[Ae]=0 makes
it vanish. It requires the actual conditional prefix residual E[e|A].

For e=epsilon*d and a sufficiently regular joint density/conditional second
moment near A=0, the familiar leading boundary approximation is

    E[B] = epsilon^2/2 * p_A(0) E[d^2|A=0]
           + o(epsilon^2).

This statement requires its regularity assumptions; a literal ReLU prefix
can have degenerate pieces. More importantly, it approximates only B, not
the missing E[G e]. Dropping that term invalidates the estimator even if the
boundary approximation is exact. Matching Gaussian moments through a low
order does not provide an actual conditional-prefix law.

## 2. Almost linear, no sign crossings, nonzero first-order mean correction

There is an explicit degree-one homogeneous counterexample, with no sampled
quantities. For n>=3, R=||X|| and 0<epsilon<(n+2)/3, take

    A=X_1,
    e=epsilon*X_1*(X_1^2/R^2 - 3/(n+2)).

Define its value at R=0 by continuity. Sphere moments give

    E[e]=0,                  E[X_j e]=0 for every j,
    E[e^2] = epsilon^2 * 6(n-1)/[(n+2)^2(n+4)].       (2)

Nevertheless a=A+e=X_1*[1+epsilon*(U_1^2-3/(n+2))]
has exactly A's sign everywhere away from zero. Therefore B=0 identically.
Writing phi0=1/sqrt(2*pi), the exact gated residual is

    E[G e] = -epsilon*phi0*(n-1)/[(n+1)(n+2)],
    E ReLU(a) = phi0*[1-epsilon*(n-1)/((n+1)(n+2))]. (3)

For completeness, E[U_1^3 1(U_1>0)] /
E[U_1 1(U_1>0)] = 2/(n+1), and E[R] E[U_1 1(U_1>0)]=phi0;
these give (3). The sixth/fourth sphere moments give (2).

Thus the perturbation has O(epsilon^2/n^2) variance, the affine projection
is unchanged, and the terminal kink never switches, yet the mean changes
at O(epsilon/n). This is a smooth-away-from-origin homogeneous example,
not an exact finite ReLU-network construction or an observed Phase2 error.
It disproves the claimed implication from near-linearity/no crossings to
a boundary-only correction. It does not show that the current cumulant
card misses this particular example; its fourth moment can already encode
the leading effect.

## 3. The closest actual implementations already expose the missing source

Paths in this section are relative to the Phase2 experiment directory.

- `public_quadrature_derivation_v1/realnet_rules.py`, `build_affine`, carries
  the Gaussian response chain through all16 layers and evaluates a paired
  affine control on the actual Gaussian inputs. Its recorded capture was
  about0.205 on rank35. The late-layer *retention* approached one, but final
  whole-input linear mass remained about0.22. Calling the actual network
  globally almost affine on that evidence is incorrect. These historical
  numbers are not bounds on every nonlinear surrogate or input direction.
- `rqmc_fullbudget_rank1024_cv/estimator.py`, `_ResponseEstimator.predict`,
  already builds the full-input response R_l=R_(l-1) W_l diag(Phi(alpha_l)),
  evaluates only the final shifted-ReLU proxy, and applies fixed or
  leave-one-shift-out control coefficients. Its actual correction is
  ReLU(a)-ReLU(A)=G e+B, exactly the sum in (1). That source's retained
  immutable-array meter failure was before accuracy: it is an unqualified
  implementation, not a measured scientific rejection and not a new lead.
- `p2r0f3c_late_layer_hermite_cv_audit_v1/AUDIT.md` found a cheap terminal
  high-degree polynomial residual, but its true-law control means remain
  unprovided. Dividing (1) into a conditional linear expectation and a
  curvature term is subject to the same centering ownership requirement.
- `astra_phase2_reexamination_v1/HOMOGENEOUS_BOUNDARY_MEAN_WARD_AUDIT_V1.md`
  separately proved that an unconditional gradient-energy scalar plus
  Gaussian regression returns the ordinary Gaussian mean. A local actual
  Jacobian does not fix that: its gates depend on X, so integrating X while
  holding that Jacobian fixed changes the conditional law.

The old quadrature document's universal "no tractable surrogate" language
is not used here. Its Gaussian-ensemble centering interpretation was
corrected by `NNGP_CENTERING_AUDIT_V1_CORRECTION.md`; the concrete executed
affine-control and GH2 sources retain their narrower empirical meaning.

## 4. Complete-price prerequisite, not a boundary-only variance claim

With a fixed affine reference for all1024 outputs, constructing the full
response from known scalar gates takes15 dense matrix products, leading
30*n^3=32.212B arithmetic. Acquiring those gates/locations from a card,
mean/covariance propagation, or calibration is ADDITIONAL work. Per coupled
sample, the actual16-layer path plus the dense affine query requires at
least the DECLARED classical leading subtotal

    c = 34*n^2 = 35,651,584 FLOPs,                    (4)

plus RNG, activations, G/e/B arithmetic, reductions and ingress. These are
not native meter figures and not lower bounds against all fast matrix
multiplication algorithms. Storing the response costs n^2 numbers and a
streamed batch avoids storing all paths; memory is not the decisive gap.

If both G e and B are estimated on the same paths, their sum is the existing
shifted-ridge residual. For an independently known E[G e], a hypothetical
boundary-only estimator would instead have variance Var(B)/N, but that
independent source and its bias/cost are absent. Estimating E[G e] on those
same paths does not remove its variance; estimating it independently at
full path cost adds another acquisition term, not a free centering oracle.

For any fully specified centered residual R with mean-source bias b and
setup/source cost C0, the IID information prerequisite is

    (mean_j b_j^2 + mean_j Var(R_j)/N)
        * max(0.1, (C0+N*c)/2^41) < 1e-9.           (5)

Even ignoring C0, all other work and bias, (4) requires mean residual
variance below6.168094117647e-5. If someone supplies a cheap centering law
for G e, the falsifiable prerequisite is a joint certificate for its
combined bias AND the correctly centered B residual in (5), not merely a
small sign-disagreement rate or a good approximation to p_A(0).

The inspected sources provide no such centered supplier. Accordingly this
audit proposes no sampling, fit, numerical screen or new candidate. Its
useful result is the exact ownership obstruction and counterexample above.

# Trace-dependent normal variance–mean consumer: bounded algebra screen

Phase 2 only, September 6, 2026. Independent accuracy research by the
fresh_accuracy agent. No production change, fit, Monte Carlo, or score.

## Mechanism and distinction

The current third tensor contains `Sym3(b,B)`, in addition to the CP and
coordinate-star pieces. The closed Gamma radial consumer integrated a
variance scale with an unchanged conditional mean, and a formally scaled
third-cumulant response. It did not integrate a mean that is correlated
with that scale. This proposed consumer uses that already-carried `b`.

Let `q = radial / 3 > 0`, `a = b/q`, and independent

```
S ~ Gamma(shape=1/q, scale=q)
Z ~ N(0, V),       V = B - b b^T/q
Z0 ~ N(0, D),      D = C - B
Y = mu + a (S-1) + sqrt(S) Z + Z0.
```

This is an actual probability model only when V and D are positive
semidefinite. Its mean is mu and covariance is exactly C. With
`x=b.t` and `y=t^T B t`, its directional cumulants are

```
K3[t^3] = 3 x y - x^3/q
K4[t^4] = 3 q y^2 + 6 x^2 y - 3 x^4/q
K5[t^5] = 30 q x y^2 - 6 x^5/q
K6[t^6] = 30 q^2 y^3 + 180 q x^2 y^2 - 90 x^4 y.
```

The exact cumulant generating function proves these identities:

```
K(t) = mu.t + (t^T D t)/2 - x/q
       - log(1 - x - q*y/2 + x^2/2)/q.
```

Thus its third tensor is NOT exactly the bank trace, and its fourth
tensor is NOT exactly the bank radial tensor. These differences must
be subtracted explicitly; substituting the mixture for the bank would
silently change existing lower-order physics and is not proposed.

## Baseline and subtraction ownership

For b and zero b, at the same C, B, q, calculate mixture post-ReLU first
and RAW second moments. For each, also calculate the FIRST-ORDER
Gaussian/Edgeworth raw moments from that mixture's own exact K3/K4,
including its actual K31 plane. The proposed increment is

```
Delta R = (R_mix(b) - R_mix(0)) - (R_EW(b) - R_EW(0)).
```

Add Delta mean and Delta raw second moment to the ORIGINAL parent
read, then center once using the new mean. The b=0 Gamma consumer is
not added; all CP/star/radial state stays the original bank. Do not use
the parent's covariance-pairing K31 approximation for this subtraction.
Do not reconstruct linear Edgeworth RAW seconds as cov+outer(full_mean,
full_mean), which would add an unintended quadratic third term.

This is a formal residual correction once added to an unrelated bank
law. It contains b-dependent fifth/higher cumulants and higher products
of lower cumulants. The previously closed full L3^2 density experiment
does cover one of those latter subterms; it does not establish value
of this different complete increment. A positive result must be
attributed to the full specified increment, not to L3^2 as a new idea.

## Fixed qualification, before any value test

1. Tiny exact Gamma-polynomial/Gaussian-Wick moments through order six
   must match these tensor/directional formulas to 1e-11 relative (with
   an absolute 1e-13 floor). Exact first two moments must also match.
2. Only inspect already-archived, target-free bank tapes on ranks35/100,
   fixed layers8..14. Reconstruct b and the previous active post-C from
   those F32 tapes, and report approximation status prominently. In
   source coordinates the necessary condition is
   `b_active^T C_active^{-1} b_active <= q`; b must have no inactive
   component. Require at least 10% margin for every selected cell.
   Report any negative cold diagonal and active covariance eigenvalue;
   do not clip them or declare an exact probability certificate from
   rounded tapes. If these data are too weak, acquire exact states in
   a separately authorized replay before any value claim.
3. If feasible, use exactly eight Gamma nodes and 32 only as numerical
   reference, as in the prior radial experiment. No node/q/range sweep.
   Require <=1e-8 relative output disagreement and <=1e-3 disagreement
   in the induced covariance increment. A failed gate stops this version.

Only after root approval and exact-state qualification: use the same
seven established coherent rank35 cells and frozen next-weight response
as the old Gamma experiment. Fixed information gates are covariance
error <=.95, next-mean-response error <=.90, at least5/7 value wins.
This is not a scored endpoint or permission for coadaptation/rollout.

## Cost and resources

The conditional covariance is D + S V and conditional mean mu+a(S-1).
Its correlation changes with S, so 16 separate Gaussian pair maps are
required (eight b, eight zero), plus two exact-K31 Edgeworth difference
maps, node preparation, centering, and the feasibility check. These are
O(16 * Gaussian-order * n^2) arithmetic, NOT a shared correlation map.
No new dense weight push or full covariance factorization is needed to
evaluate the conditional pair maps. A production PSD proof/guard is
not yet solved and must be priced if it needs matrix work. Full K21 and
radial-state consumers are not implemented or priced; this is an
additive first/raw-second response proposal only.

One nice10 CPU worker, <=2GB, no MPS, no new targets/large acquisitions.
Preserve all artifacts and closed receipts. No forum posts or uploads.

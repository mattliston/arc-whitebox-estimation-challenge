# Fused directional innovation/action interface

Status: **source query and arithmetic frozen; endpoint-action rank pending**  
Date: 2026-08-26  
Scope: target-free exact algebra and static pricing only

## Decision

The smallest complete source coordinate is not a dense cumulant tensor and is
not a coordinate marginal.  It is one **directional connected cumulant**

```text
kappa_q(u) = cum(u^T(H-m), ..., u^T(H-m)),       q=2,3,4,5.
```

One scalar `kappa_q(u)` contracts every collision sector of the symmetric
order-`q` tensor with `u` in every mode.  At order five this includes
`[5],[41],[32],[311],[221],[2111],[11111]` with the exact permutation
multiplicities.  Computing the scalar cumulant by the ordinary set-partition
Mobius formula also keeps every connected cancellation.  No 3--5 dimensional
ReLU oracle and no generic tensor-train extraction is hidden in the source
query.

For a rank-`R` symmetric action spelling, one sample projection per ray supplies
the five base coordinates

```text
mean(u), covariance(u), kappa3(u), kappa4(u), kappa5(u).
```

The covariance is part of the resummed base.  The grade-three composite shells

```text
kappa3^2,  kappa3*kappa4,  kappa3^3
```

are derived from those same coordinates; they are not separately sampled and
must not be dropped or linearized independently.  With `R=13`, this is exactly
`65` sampled scalars per layer, `390` over `h2:h7`, or `520` over `h2:h9`.

This freezes an executable source/action interface.  It does **not** certify
that the actual Phase-2 endpoint action has rank at most 13.  That measured,
source-only action-rank gate is the remaining external dependency.  If it
fails, the interface is exact but cannot carry enough endpoint action inside
the budget.

## 1. Marginals versus joint coordinates

A coordinate marginal uses `u=e_i` and sees only `[q]`.  Existing exact-ledger
evidence already makes this a sidecar rather than the main repair: the coherent
oracle-input marginal high-order mean chain is only
`2.2367330586834924e-10` MSE (`1.4955711479844388e-5` RMS), while the carried
pair/covariance channel owns the larger residual growth.

A dense ray `u` is materially different.  Expanding `kappa_q(u)` gives

```text
sum_(i1,...,iq) K_q[i1,...,iq] u[i1]...u[iq],
```

so joint planes and all-distinct sectors are present.  The exact `n=5`
verifier supplies a nonzero witness for every collision class through order
five and proves dense contraction parity with rational arithmetic.

The incumbent reference along the same ray is explicit:

```text
C3(u) = sum_i k3d_i u_i^3
      + 3 sum_(i!=j) K21_ij u_i^2 u_j

C4(u) = sum_i k4d_i u_i^4
      + 4 sum_(i!=j) K31_ij u_i^3 u_j
      + 3 sum_(i!=j) K22_ij u_i^2 u_j^2

C5(u) = sum_i k5d_i u_i^5
      + 5 sum_(i!=j) K41_ij u_i^4 u_j
      + 10 sum_(i!=j) K32_ij u_i^3 u_j^2.
```

Every absent collision class is zero in this restricted reference, never
silently folded into a represented plane.  A single-anchor state innovation
is `kappa_q^sample(u)-Cq(u)`.  Accumulated innovations from different anchors
must not be naively summed: use one anchor, or supply a teacher-forced local
reset defect and its full-state Duhamel action.  This ownership rule prevents
double-counting.

## 2. Endpoint action contract

For anchor layer `l`, ray `j`, and endpoint coordinate `o`, the action lane must
return source-only coefficients for all eight blocks

```text
a1, a2, a3, a4, a5, a33, a34, a333.
```

The correction is

```text
delta_y[o] = sum_(l,j) [
    a1   delta(mean)
  + a2   delta(covariance)
  + a3   delta(kappa3)
  + a4   delta(kappa4)
  + a5   delta(kappa5)
  + a33  delta(kappa3^2)
  + a34  delta(kappa3*kappa4)
  + a333 delta(kappa3^3)
].
```

The coefficients must be built around the full mean/covariance base.  A pooled
rank score is insufficient: retained action must be reported separately for
covariance, orders three through five, and the mixed-shell block.  Ordinary
first-order VJP/JVP supplies only the linear blocks; it does not define the
three composite blocks or the missing joint reset births.

The immediate scalar ReLU mean block is an exact special case.  For
`y_a=w_a^T h`, variance `v_a`, directional cumulants `t3,t4,t5`, and
`c_a[k]=E[(N(mu_a,v_a))_+ He_k((Y-mu_a)/sqrt(v_a))]`, its complete formal
grade-three correction is

```text
c[3] t3/(6 v^(3/2))
+ c[4] t4/(24 v^2)
+ c[5] t5/(120 v^(5/2))
+ c[6] t3^2/(72 v^3)
+ c[7] t3*t4/(144 v^(7/2))
+ c[9] t3^3/(1296 v^(9/2)).
```

This exact one-step formula is a necessary action-rank discriminator, not a
replacement for the nonlinear full-state suffix action.

## 3. Frozen arithmetic law

Let `N` be the number of sampled paths, `n=1024`, `R` the number of shared
symmetric action rays, and `L` the number of exposed anchor layers.  A literal
dense dot-product circuit costs

```text
C_projection = L*N*R*(2*n-1).
```

For each projected ray, a frozen circuit performs the mean reduction, centers
the values, forms powers two through five, accumulates four moments, normalizes
them, and performs the connected conversions

```text
kappa4 = M4 - 3 M2^2,
kappa5 = M5 - 10 M3 M2.
```

Its count is

```text
C_statistics = L*R*(10*N+10).
```

At `N=65,536`, `R=13`:

| anchors | sampled coordinates | projection + statistics |
|---|---:|---:|
| `h2:h7` (`L=6`) | `390` | `10,514,989,836` FLOPs |
| `h2:h9` (`L=8`) | `520` | `14,019,986,448` FLOPs |

Projecting the complete incumbent diag/plane reference by the explicit formulas
above costs at most

```text
C_reference = L*R*(17*n^2 - 2*n + 5),
```

or `1,390,252,422` FLOPs for six layers and `1,853,669,896` for eight.
Endpoint assembly and the eight coefficient tables are lower order.  Prefix
path generation is a separate Kerdock-lane bill and must be added literally.

The pricing assumes one shared symmetric ray per rank component.  A
non-symmetric multilinear action may use direct tuples
`(u1,...,uq)` and the same Mobius cumulant, but every distinct projection must
then be priced.  It cannot inherit the shared-ray count.

## 4. Gates and terminal boundary

The interface is executable only when all of the following hold:

1. Kerdock prefix generation plus the frozen query/reference counts and the D1
   host remain below the exact per-row `2^41` law.
2. Split-block action-weighted query error is at most `2e-5` endpoint RMS.
3. Rank `R<=13` retains enough actual-weight action separately in covariance,
   K3, K4, K5, and mixed shells.  Report both the `99%` completeness diagnostic
   and the `70%` objective-relevance diagnostic.
4. Small-width dense parity includes every collision sector and every mixed
   shell, as the bundled verifier does for the source query.
5. One anchor owns an accumulated state correction, or multi-layer terms are
   genuine teacher-forced local defects.  No repeated accumulated correction
   may be summed through Duhamel.

Marginal-only sampling is closed as a principal repair by the existing ledger
scale.  The complete directional source query remains open only until the
actual action-rank and prefix-budget dependencies return.

## 5. Pathwise control-variate boundary

The closure has one known nonconstant pathwise lift with exactly matching
analytic mean: the Gaussian-equivalent affine Stein chain

```text
g(z) = A_eff z + c_eff,
E[g] = c_eff = the closure Gaussian mean chain.
```

Its frozen precomposition costs `32,279,363,584` FLOPs and evaluation costs
`2,099,200` FLOPs per path.  It captures only `0.205049` of measured variance
(cross-row range `0.190319--0.208409`).  More importantly, an antipodal
Kerdock rule integrates the affine lift exactly, so

```text
E[g] + Q_Kerdock[f-g] = Q_Kerdock[f]
```

identically.  It provides no incremental variance reduction.

The resummed directional interface above is distributional.  It does not
currently define a richer pointwise `g(z)` whose true Gaussian mean is exactly
the closure output.  Adding zero-mean input Hermite terms is formally legal,
but terms in Kerdock's exactness space cancel in the same way and the existing
degree-capture ceilings remain far below the required control-variate capture.
The missing stacking interface is therefore a cheap nonlinear pathwise lift
outside Kerdock's exactness space, with a proved analytic mean identity and
roughly `>=97%` variance capture.  No such lift exists in the current source.

## 6. Verification

Run:

```bash
python p2r0f3b2b6_fused_directional_innovation_action_v1/verify_interface.py
```

It opens no target, fixture, estimator, or network service.  All scientific
parity checks use exact rational arithmetic.

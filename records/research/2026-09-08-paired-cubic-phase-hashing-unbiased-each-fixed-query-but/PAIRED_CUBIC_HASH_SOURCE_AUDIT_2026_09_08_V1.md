# Paired cubic phase hashing: exact fixed-query noise, no numerical admission

September 8, 2026 PDT. Phase 2 only. Source and algebra audit, independently
checked by root and the existing accuracy agent. No network arrays, truth
arrays, estimator execution, random experiment, fitting or candidate change.
This does not reopen the closed rank/compression, thinning or pair-kernel
families. No novelty claim about the general sketching technique is made.

## Question and decision

Could a small phase-hashed bank preserve paired cubic cancellations while
removing a substantial part of the growing CP push/read cost?

The fixed-query expectation is correct, but pairing does not by itself make
the relative collision noise small. An explicit Gaussian-column reference
below gives noise-to-signal ratio approaching 11/3 already at the generic
inner-width break-even point. That is not measured bank geometry and is not
a universal sketch lower bound. It supplies no positive admission evidence.
The actual nonlinear feedback also destroys the straightforward fixed-query
unbiasedness argument. No seed, bucket-count, rank or head-training run is
admitted from this source audit.

## Exact scalar law

Let real x_r,y_r, r=1..R, be fixed. In a CP application these can be the
fixed query's projections on a signed pair, with the real cube root of the
pair weight absorbed into both vectors. Choose independently uniform bucket
labels h_r in 1..B and independent phases eta_r among the three cube roots
of unity. Define the REAL estimator

    Z = Re sum_b [(sum_{r:h_r=b} eta_r*x_r)^3
                  - (sum_{r:h_r=b} eta_r*y_r)^3].

Then E Z = T = sum_r (x_r^3-y_r^3). Set

    A_rs = x_r^2*x_s-y_r^2*y_s,
    D_rst = x_r*x_s*x_t-y_r*y_s*y_t.

The exact variance, over both phases and buckets, is

    Var Z = 9/(2B) sum_{r<s} (A_rs+A_sr)^2
            + 18/B^2 sum_{r<s<t} D_rst^2.                 (1)

The repeated-pair terms are conjugate phase characters, so their REAL
parts combine before squaring. Using A_rs^2+A_sr^2 instead would be wrong.
The distinct pair and triple characters are orthogonal conditional on the
hash. Correlations between collision events therefore introduce no missing
cross-covariance. Pair and triple collisions have probabilities 1/B and
1/B^2 under the specified independent hash.

For the deliberately special equal-column example x_r=x,y_r=y, with
x^3 != y^3, (1) gives

    Var Z / T^2 = 9*(R-1)/(B*R)
                  +3*(R-1)*(R-2)/(B^2*R).

That favorable coherent-query example must not be assumed for current CP
projections. Queries with cancellations can have a much smaller T.

## O(R) evaluation of the fixed-query variance

Write S_ab=sum_r x_r^a*y_r^b and Sx_k=sum_r x_r^k, with Sy_k analogous.
The pair sum in (1) is P_x+P_y-2Q, where

    P_x = Sx_4*Sx_2 + Sx_3^2 - 2*Sx_6,
    Q   = S_22*S_11 + S_21*S_12 - 2*S_33.

For E3(z)=((sum z)^3-3*(sum z)*(sum z^2)+2*sum z^3)/6,
the triple sum is E3(x^2)+E3(y^2)-2*E3(x*y).
These are exact real-arithmetic identities. Their cancellation-sensitive
finite-precision implementation has not been qualified. No numerical
evaluation on archived/current factors was performed.

## Analytic reference model: absolute cancellation is not relative capture

For this section ONLY, suppose the pairs (x_r,y_r) are iid centered jointly
Gaussian across r, with equal variance sigma^2 and within-pair correlation
rho. This is a declared reference model, NOT a measured property of a net.
Wick moments give

    E_data T^2 = R*sigma^6*(30-18*rho-12*rho^3),
    E_data (A_rs+A_sr)^2 = sigma^6*(12-4*rho-8*rho^3),
    E_data D_rst^2 = 2*sigma^6*(1-rho^3).

Consequently, as rho approaches one from below,

    E_data Var_hash(Z|x,y) / E_data T^2
      -> (7/6)*(R-1)/B + (1/3)*(R-1)*(R-2)/B^2.        (2)

This is a RATIO OF EXPECTATIONS, not E[Var(Z|x,y)/T^2]. Sharing the paired
phase reduces the absolute noise, but the true paired signal shrinks too.
At B approximately R/2, (2) approaches 11/3 as R grows; at B approximately
R/4 it approaches 10. These are reference-model arithmetic, not Phase 2
raw-error ratios, complete-score forecasts or empirical rejection receipts.

## Complete-cost and feedback ownership

For each bucket retain complex U_b and V_b. Real implementation uses four
B-wide factor arrays. A generic real affine push therefore has inner width
4B instead of the parent's 2R, before support/padding and youngest sharing.
Its directed repeated-plane expression is

    (Ur^2-Ui^2) @ Ur.T - (2*Ur*Ui) @ Ui.T
    -(Vr^2-Vi^2) @ Vr.T + (2*Vr*Vi) @ Vi.T,

with the appropriate response/active row selections. Thus this generic
reader also has four B-wide products. B<R/2 is only a dominant-inner-width
prerequisite for a saving, NOT a complete arithmetic break-even point.
No price discount for complex arithmetic is assumed.

Initialization, random phases and hashes, real weight roots, bucket
accumulation, all birth sources, gates/masks, scalar diagonals for all rows,
omitted-radial restoration, CP/star rebasing, covariance, all16 current P
heads, ingress/copies and output handling remain paid. Early layers need
their actual smaller ranks priced; replacing every early bank by a fixed
large B is not a free saving. No complete FNP program or full-card bill is
provided here.

Fixed real affine maps and row gates commute with the bucket sums. In the
actual card, however, gates, births, radial/mean feedback and future queries
depend on earlier sketch errors. They are not independent of the phases.
Equation (1) cannot be applied as an unbiased complete-estimator guarantee,
nor can per-layer variances be summed while ignoring shared-seed feedback.
An approximate biased method is not categorically forbidden, but needs
actual compatible raw-times-complete-cost evidence; this audit supplies
none and does not authorize reopening a closed compression arm.

The inspected older CountSketch sections of LANE_fused_pair_source.md and
LANE_h4_square_lift.md concern bilinear action sketches and their reset
ownership. They do not supply a measured cubic-pair collision spectrum or
a cheap unbiased nonlinear recursion. The bounded phrase/source search
located no execution receipt for the exact construction above; absence
from that search is not a positive performance witness.

## Endpoint

The qualified local reference remains unused_materializations_v1/estimator.py
SHA409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f,
adjusted4.770235434670105e-9 on the existing independent14 panel, with
unchanged NCraw2.2458484596872894e-8. No raw or score progress follows from
this algebra. The missing ingredients are favorable ACTUAL consumed-action
noise, a head-compatible recursive law, and a complete lower-cost program.
The official strictly-below1e-9 goal and milestone-only policy are unchanged.

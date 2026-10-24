# Shared signed-pair quartic: source and cost audit

2026-09-10 UTC, Phase2. Additive independent audit of
SHARED_PAIR_QUARTIC_SOURCE_DRAFT_20260910.md; that draft and every old failure
remain unchanged. No model/array/target read, numerical experiment, fit,
estimator edit or new execution is selected by this note.

## Exact convention and what can be shared

Write a symmetric tensor as its homogeneous polynomial, without factorial
normalization. `Sym_sum(U,P,P)` denotes the THREE-term sum, not its average.
For a hub h, B_hh=0, coordinate e_h, and opposite quartic weights gamma:

    gamma[(B+t e)^4-(B+s e)^4]
      =4 u1 e B^3+6 u2 e^2 B^2+4 u3 e^3 B+u4 e^4,
    uk=gamma*(t^k-s^k).

The Gaussian one-cubic/three-linear connected-tree tensor has hub
coefficient epsilon_h=E[f'''(X_h)]=-alpha_h*phi(alpha_h)/sigma_h^2 and
off-hub B_ih=g_i*C_ih. Its polynomial is
`4 sum_h epsilon_h z_h (B_h dot z)^3`. Thus u1=epsilon matches precisely
that all-distinct source IF those are the actual B columns. It is not the
whole Gaussian fourth cumulant: two-quadratic/two-linear trees and higher
Hermite graphs remain separate. At zero mean epsilon=0, whereas one ReLU
Gaussian variable has nonzero fourth cumulant
`sigma^4*(3/4-1/pi-3/(2*pi^2))`; the missing source is already evident there.

Actual COMP factors have BD/NODE-modified bases, selected births, active
masks and stored float32 t,s. Their basis is not g*C. Choosing gamma from
the gap therefore matches a modified-basis tensor, not automatically the
physical Gaussian source. Gamma itself must have a declared stored dtype;
gamma32*(t32-s32) need not equal epsilon. Zero/small gaps, finite weights
and the existing safe-birth/discriminant/clipping law need qualification,
not a new unannounced denominator or branch. The two factors can share
their existing affine and diagonal-gate transport for this specified tensor.

## Complete repeated-index content is not optional

For a collection of these hub columns, set uk=0 on unselected births. For
i!=j the actual stored tensor, evaluated after widening stored values, has

    Q31_ij = u3_i*B_ji + u1_j*B_ij^3,
    Q22_ij = u2_i*B_ji^2 + u2_j*B_ij^2,
    Q4_i   = u4_i,
    sum_ij Q22_ij = sum_i u4_i + 2 sum_h u2_h ||B_h||^2.

These birth faces cost O(n^2) to form, but are constrained by the OLD cubic
t/s choice: u2/u1=t+s, u3/u1=t^2+t*s+s^2, etc. They are not independently
chosen Gaussian fourth faces. Removing them at birth and keeping only the
all-distinct part requires explicit fourth counterterms and their later
transport/reset law. The current cubic star is not that fourth carrier.
The earlier exact-heavy mixed-source audit documents why immediate plane
equality alone does not establish future CP/star gauge equivalence.

At any later read with carried columns P and quartic weights gamma:

    k4_i = sum_r gamma_r*P_ir^4,
    K31  = (P^3*gamma) @ P.T,
    K22  = (P^2*gamma) @ (P^2).T.

Current COMP `_nonlinear_pair_batch` instead forms standardized
`T31=t4[:,None]*rho`. Adding only marginal k4 or only K22 does not insert
this general tensor. All existing pair/mean/variance/trace/head consumers
and actual repeated diagonals must follow the declared new input law.

## Leading fourth-to-third descendants can aggregate, but are not free

For Q4=sum_r gamma_r P_r^4 the leading quadratic-vertex emission is
`(gamma/2)*Sym_sum(d*P^2,g*P,g*P)`. One extra vector U per quartic factor
lineage can aggregate its historical leading emissions:

    pp = W.T @ P                  # existing factor push
    uu = W.T @ U                  # additional factor push
    P_next = g*pp
    U_next = g*uu + d*pp^2,
    T_extra = (gamma/2)*Sym_sum(U,P,P).

Use the same declared masks; append zero U for a newly born quartic factor.
This is an exact real-arithmetic identity under the specified linear/gate
maps and leading emission, not exact nonlinear cumulant evolution. It avoids
three new independently carried CP3 atoms at every layer, but does require
the new U transport. Its plane and diagonal are

    K21_extra = (U*P*gamma) @ P.T + .5*(P^2*gamma) @ U.T,
    k3_extra  = 1.5*sum_r gamma_r*U_r*P_r^2.

The first plane product can ALGEBRAICALLY fuse into the incumbent
`(P^2*lambda3)@P.T`; one additional mixed product remains. Preserving the
old floating-point grouping instead costs both extra products. Null bytes,
rounding, birth/current-plane subtraction and reset partitions require a
new certificate. This identity was not found as an executed archive arm.

## Cost is not a complete inexpensive native candidate yet

The marginal-only read has a simple conservative full-width operation
subtotal: widen P, square twice, and a weighted reduction, about10*n*R
FP64-rate-weighted F without power reuse. With n1024 and the maximal
R_l=2*n*l over l1..15 this is2.5165824B F. It excludes radial residual
reset, all other consumers and coefficient/packing/guard work. It is a
marginal experiment, not the full source, and is NOT an actual native bill.

With the leading U recurrence and algebraic fusion, the full-width product
subtotal is one U push plus K31, K22 and one extra K21 product. If M denotes
an actual chosen matrix kernel tariff, the extra subtotal per transition is

    M(n,previous_active,R) + 3*M(n,R,n),

before shape-specific gathering, powers, scalar reads, masks, diagonal/star
repair, quartic birth, radial reference and ordinary COMP operations. A
full-width classical float32 upper-envelope is8*n^2*sum_l R_l, or
1920*n^3=2,061,584,302,080 product F under that maximal roster. It is an
unpacked structural upper-envelope, not a lower bound or measured packed
cost. At R24000 the same four-product full-width subtotal is201.326592B
per transition. Partial rows and the actual packed kernels can reduce it,
but no complete qualified schedule below500B has been supplied. Avoid
presenting the absence of an extra P push as an absent full-source cost.

Exact subtraction of each historical radial projection additionally needs
its birth metric/countertensor transported. The four-cycle audit already
prices the fifteen full countermetric congruences at about483.2B CLASSICAL
products before their descendants. A cheaper per-layer radial reset could
subtract the trace of ALL retained gated and newborn quartic content,
`delta_r=-3*sum_r gamma_r*||P_r||^4 / ((tr C)^2+2||C||F^2)` for the chosen
symmetric metric. That explicitly reprojects historical content and is an
approximation, not exact causal subtraction. Current asymmetric-metric
conventions also need their literal owned trace treatment. No one such
counterterm/reset choice has been selected here, so a complete candidate
upper price is not yet defined.

## Prior scope, value and decision

- `candidate_recovered_cp3_8k_v1.py` has a separate pushed CP4 bank, unit-vector
  diagonal births, energy cap256 and quartic-star/radial residual reset;
  it does not assign quartic weights to the current balanced pair factors.
- `GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md` tested a fixed two-quadratic/
  two-linear source mode: isolated same-source ratio.99355293, failing.5.
  That is a different supplier, not positive evidence for this one.
- `FOURTH_PLANE_PAIR_SPAN_V1_CLOSED.md` fixed analytic field span left
  isolated next-mean ratio.88304182 and covariance.78991702, failing.20.
  It was not this historical shared-factor tensor.
- `FOURTH_EMISSION_ORACLE_V2_CLOSED.md` true single-cell fourth emission
  against accumulated third error gave.992040038, failing its gate.
  `FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md` gave about.611 isolated
  next-mean/covariance ratios, not endpoint raw gains.
- `GAUSSIAN_FOUR_CYCLE_SOURCE_PREFLIGHT_V1.md` and
  `EXACT_HEAVY_MIXED_BIRTH_REPRESENTATION_SOURCE_AUDIT_V1.md` explain the
  radial countermetric, general K31 and partition-sensitive gauge debts.
  None turns the cheap scalar read into a full causal supplier.

No archived executed endpoint was found for this precise dual-order pair
assignment/aggregated-U law. That coverage fact supplies no magnitude.
The missing premise is natural-strength, current-COMP signed error value
of an explicitly defined residual tensor and its full consumers, together
with a complete compatible counterterm and affordable schedule. A tensor
identity or a large standalone norm is insufficient. No fit, source/step/
layer sweep or numerical arm is recommended from the present record.

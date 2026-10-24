# Gaussian four-quadratic cycle: source scope and paid-carry distinction

September 7, 2026 PDT. Phase 2 only. Root and the accuracy agent checked
these identities independently. No new network/state/target array, sample,
fit, numerical source test or candidate was executed for this route.

## A genuinely distinct source, not a fourth-order format breakthrough

For Z~N(0,C), symmetric PSD C, and q_i=d_i*(Z_i^2-C_ii)/2, the pure
four-quadratic contribution is the sum of the three undirected four-cycles,
weighted by d_i*d_j*d_k*d_l. Its scalar output contraction is

    Q_cycle[w,w,w,w] = 3*trace((C*diag(d*w))^4).

Its repeated planes, including diagonal limits, are

    Q_iijj = d_i^2*d_j^2*(2*C_ii*C_jj*C_ij^2 + C_ij^4),
    Q_iiij = 3*d_i^3*d_j*C_ii^2*C_ij^2,
    Q_iiii = 3*d_i^4*C_ii^4.

In the inspected prior work:

- public_g29_bond_kernel_v1/block_feed_1024.py adds a doubled path edge;
  that is not a four-edge cycle. Its path-feed result does not close this.
- public_g23_k3_supply_v1/cert_star4.py includes cycles in two synthetic4D
  examples, contributing4.7%/8.7% of total fourth cumulant. These are not
  width1024 source-value measurements.
- GAUSSIAN_FOURTH_R1_SOURCE_TEST_V1_CLOSED.md tests a two-quadratic /
  two-linear source with a middle-edge approximation, not this all-four-
  quadratic source. Its closeout is unchanged.
- The inspected SSC basis consists of nine fixed covariance/moment
  matrices, not the causal cycle factor d*u^2. Old CP4 unit-vector births
  and the three-edge Gaussian star source likewise do not execute this
  birth. CP4 and matrix-mode representations themselves are old machinery.

This is a distinct physical-source question. It is not permission to
reopen any prior closed source, carrier, fit or mode/window sweep.

## Exact stochastic representation and why it is not free

For an independent z~N(0,C), set

    V = diag(d) * [C elementwise (z*z.T-C)] * diag(d),
    S(V)_ijkl = V_ij*V_kl + V_ik*V_jl + V_il*V_jk.

Wick's identity gives

    E[V_ij V_kl] = d_i*d_j*d_k*d_l*C_ij*C_kl
                   *(C_ik*C_jl+C_il*C_jk),
    Q_cycle = (1/2) E[S(V)].

This holds with repeated indices and singular PSD C. An indefinite or
asymmetric bank matrix is not automatically the required Gaussian law.

On a FROZEN deterministic linear/gate trajectory, N modes can accumulate
all independent zero-mean births using

    M_s <- J.T*M_s*J + V_s,new,
    Q_hat = (1/(2*N))*sum_s S(M_s),  J=W*diag(g).

Fresh independent birth innovations make cross-birth bilinears vanish in
expectation. Correlated reused draws or sample-dependent feedback do not
inherit that frozen-trajectory statement. Aggregation adds variance: for
a fixed ray, transported birth quadratic forms y_b of variance v_b give

    Var(Q_hat[ray^4]) = 9/(4*N) * [sum_b kappa4(y_b) + 2*(sum_b v_b)^2].

Thus this unmodified Gaussian construction has relative variance at least
2/N for a nonzero ray. Deterministic radial scalar subtraction does not
lower absolute probe variance. A paid trace-known MATRIX control is needed
for controlled trace variance, including cross-output covariances.

Every full matrix mode congruence costs two dense products,4*n^3 classical
operations per transition. Its fourth-to-third feedback has both vector-
matrix and all-distinct mixed-cube descendants. Dropping those descendants
or charging only fourth-plane reads is not complete cost. The cheap
two-product trace quadratic-form probe also does not price a Hutch++
matrix-vector action: that action needs five batched products plus ranges.

## Deterministic rank-one cycle supplier

For a selected rank-one Gaussian covariance C1=u*u.T, define f=d*u^2.
Its cycle source is exactly Q1=3*f^4. For the actual covariance this is
a source approximation, NOT evidence that the real C is rank one.
For Q=kappa*f^4, leading fourth-to-third feedback has polynomial

    (kappa/4)*[(q+p)^3+(q-p)^3-2*q^3],
    q=d*f^2,   p=g*f.

Three CP3 atoms suffice in exact arithmetic. Stored-factor rounding,
balanced scaling, diagonal/star repair, masking and all later pushes/
reads must still be paid and qualified. One birth per layer gives15
outgoing CP4 factors and315 extra cubic atoms before the last layer in
the straightforward recurrence, not zero-cost scalar additions.

The same-source radial coefficient at a symmetric current metric M is

    rho = 9*||f||^4 / [(trace M)^2 + 2*||M||_F^2].

An EXACT historical residual Q1-rho*S(M)/3 requires its birth metric to
be retained and transported. With a birth each layer, metric congruences
and terminal diagonal reads alone cost approximately483.2B classical
operations at1024x16, before their full-matrix fourth-to-third descendants.
That is not the inexpensive rank-one route.

A CHEAP APPROXIMATE alternative uses the existing bank's per-layer
radial projection: after the full nonlinear fourth trace is formed,
subtract the trace of ALL retained gated and newborn CP4 factors when
resetting the residual radial coefficient. Subtracting only the newest
birth recounts old retained CP4 content. For general weights kappa_r, the
coefficient adjustment is -3*sum_r kappa_r*||f_r||^4 / denominator.
This explicitly reprojects historical complements; it is not exact
countermetric carry or a parity theorem. For an asymmetric bank metric,
the literal K22-sum convention and a symmetric-tensor trace differ.

The positive CP4 piece plus its cubic descendants has a small classical
product subtotal relative to the~472B retained bill. It is NOT yet a
complete native candidate price: source-mode acquisition, full readers,
extra elementwise work, selection, rounding and packing remain to be fixed.
In particular current _nonlinear_pair_batch uses T31=t4[:,None]*rho.
Supplying general CP4 to k4 and K22 alone is not a faithful fourth input.
The actual K31 must be read as residual_radial*v_i*C_ij plus
sum_r kappa_r*f_ri^3*f_rj, with compatible normalization. All pair-map
outputs, trace-vector updates, ordinary cubic births, star reset, omitted-
row radial source, masks and frozen heads must follow the chosen full law.
Do not use any post-current-correction feature to leak a fitting target.

## Evidence, missing fact and no-run decision

Existing current_covariance_error_r35_v1.json and r100 counterpart have
pre12 principal-mode covariance trace shares0.0662053477893 and
0.0722222936631, verified by a read-only metadata extraction. These are
not rank-one covariance dominance. Nor do they bound the derivative-
weighted cycle concentration or its useful residual: four-cycle weighting
is a different quantity. No saved current receipt provides that value.

Existing terminal true-k4-error attribution captures4.6%/2.7% linearly;
fourth-only local supplies improve covariance/next-mean error about10--16%.
These are qualified oracle/local directions, neither universal ceilings
nor evidence that this supplier achieves any given final raw reduction.

The cheapest missing prerequisite is the natural-strength, explicitly
radial-subtracted correction on a sealed causal state, including its
covariance/next-mean consumers. A norm can exclude insufficient impact;
advancement needs alignment with a qualified error reference and enough
value to overcome the COMPLETE chosen implementation bill. No such
receipt or credible final-score magnitude is presently available.

Decision: preserve this distinct causal source and cost distinction, but
do not force a network run from the rank-one identity or covariance
spectrum alone. Any later approximate source test needs a fixed full
recurrence and prospective raw-times-complete-cost criteria; no fitting,
source/rank/layer/strength sweep is silently admitted here. Raw accuracy
remains necessary for below1e-9, and this source is not yet its solution.

# Shared-Waring escape from the Price-Hessian barrier

Date: 2026-08-30.  Scope: exact symbolic audit only.  No numerical experiment
is used.

## Executive verdict

A common symmetric Waring bank is a genuine exception to the generic
four-index Price-Hessian barrier.  If the **future quartic costate itself** is

\[
 L=\sum_{\alpha=1}^R\lambda_\alpha v_\alpha^{\otimes4},
\]

then every two-index Hessian contraction is two shared-factor products,
`O(R n^2)`.  Dense congruence and a separable coordinate gate preserve the
same number of factors.

That theorem does not make the present CP4/D10 implementation a resident
future query.  The literal CP4 bank is a target-blind **primal** quartic
carrier; D10 carries a two-index reverse covariance word, not a Waring
decomposition of the reverse quartic costate.  Reusing the primal factors
computes an incumbent CP4-owned action.  The complete-old residual need not
share those factors and, under collision removal, can have quadratic Waring
rank even when the unprojected tensor is one ray.

There is one exact but short-lived operand reuse.  A full identity cohort in
the CP4 bank transports an initially identity Price metric through the same
dense-plus-diagonal maps.  Cap `2n` retains the first two complete cohorts,
but the third identity birth presents `3n` factors to an energy selector, so
no complete older cohort survives by theorem.  This supplies a conditional
metric (G), not the missing future costate (L).

## 1. Types which must not be identified

There are three different objects at a synchronized layer cut:

1. (S_l^{CP}\in\operatorname{Sym}^4\mathbb R^n), the forward, target-blind
   CP4 carrier stored as factors and signed weights;
2. (L_l^{fut}\in(\operatorname{Sym}^4\mathbb R^n)^*\), the
   endpoint-conditioned strict-depth quartic costate; and
3. (G_l\in\operatorname{Sym}_n), a two-index Price metric or covariance
   reverse mask.

Euclidean self-duality lets all three be written with arrays, but it does not
make their causal roles interchangeable.  The shared-Waring theorem below is
a cheap reverse theorem only when its factor bank represents
(L_l^{fut}).  Applying it to the stored bank represents
(S_l^{CP}), which is already part of the incumbent forward state.

## 2. Exact shared-Waring Hessian contraction

For a fully symmetric quartic tensor define

\[
 [\mathsf C_G(L)]_{cd}
 =\sum_{a,b}G_{ab}L_{abcd}.
\tag{2.1}
\]

Let (V\in\mathbb R^{R\times n}) have row
(V_{\alpha,:}=v_\alpha^T).  Substitution of one Waring ray gives

\[
 \begin{aligned}
 [\mathsf C_G(v_\alpha^{\otimes4})]_{cd}
 &=v_{\alpha c}v_{\alpha d}
   \sum_{a,b}v_{\alpha a}G_{ab}v_{\alpha b}\\
 &=(v_\alpha^TGv_\alpha)
   (v_\alpha v_\alpha^T)_{cd}.
 \end{aligned}
\]

Therefore

\[
 \boxed{
 \mathsf C_G(L)
 =V^T\operatorname{Diag}\!\left(
 \lambda\odot\operatorname{diag}(VGV^T)
 \right)V.}
\tag{2.2}
\]

This formula includes signed (lambda).  It requires no positivity or
orthogonality of the factors.

If (G\odot R) denotes the normalized symmetric pair product, full symmetry
also gives

\[
 \langle L,G\odot R\rangle
 =\langle\mathsf C_G(L),R\rangle_F
 =\sum_\alpha\lambda_\alpha
   (v_\alpha^TGv_\alpha)(v_\alpha^TRv_\alpha).
\tag{2.3}
\]

Thus a scalar pair-product query can omit the final Gram in (2.2) when both
rowwise quadratic-form vectors are already available.

## 3. Exact dense and coordinate-reset transport laws

Let

\[
 \mathscr T_A L=(A^T)^{\otimes4}L.
\]

The factors transform as (v_\alpha\mapsto A^Tv_\alpha), or (V\mapsto VA).
Direct substitution in (2.1) gives the complete metric law

\[
 \boxed{
 \mathsf C_G(\mathscr T_A L)
 =A^T\mathsf C_{AGA^T}(L)A.}
\tag{3.1}
\]

For a coordinate gate (D=\operatorname{Diag}g), this specializes to

\[
 \boxed{
 \mathsf C_G(D^{\otimes4}L)
 =D\,\mathsf C_{DGD}(L)\,D.}
\tag{3.2}
\]

Both operations preserve the exact Waring factor count.

The declared CP4-style coordinate reset has the idealized form

\[
 \mathcal R_{D,d}(L)
 =D^{\otimes4}L+\Delta(d),
 \qquad
 \Delta(d)=\sum_{i=1}^nd_i e_i^{\otimes4}.
\tag{3.3}
\]

Its exact Hessian action is

\[
 \boxed{
 \mathsf C_G(\mathcal R_{D,d}L)
 =D\mathsf C_{DGD}(L)D
  +\operatorname{Diag}(d\odot\operatorname{diag}G).}
\tag{3.4}
\]

The inherited bank keeps its rank and the birth appends at most (n) axis
rays.  Selecting a subset of stored rays is exact only for the retained
subtensor.  It supplies no identity for the discarded complement after those
factors are forgotten.

## 4. Exact shared-factor arithmetic

With (V) resident, (2.2) is evaluated by

1. (X=VG);
2. (q_\alpha=X_{\alpha,:}V_{\alpha,:}^T); and
3. one weighted Gram
   (V^T\operatorname{Diag}(\lambda\odot q)V).

The leading work is (2Rn^2), or (2R/n) width-(n) dense-product
equivalents.  A fresh dense factor push costs another (Rn^2), so a
transport-plus-contraction step is (3R/n) equivalents.  At (R=n) these
counts are two and three; at the literal CP4 cap (R=2n) they are four and
six.  A resident primal CP4 push saves the transport only if the required
costate factors are literally the same buffer, not merely the same shape.

A single Price trace (G=I) is cheaper because
(operatorname{diag}(VIV^T)) is just the row-norm vector.  This is why a
fixed-seam source trace can be cheap while a transported anisotropic future
query is not.

## 5. Coordinate births generically make the exact rank grow by (n)

Iterating (3.3) through invertible dense/separable maps gives an explicit
cohort expansion

\[
 L_T
 =\sum_{s=0}^T\sum_{i=1}^n
 d_{s,i}\,v_{s,i}^{\otimes4},
 \qquad
 v_{s,i}=A_{T-1}\cdots A_s e_i,
\tag{5.1}
\]

with the obvious transpose convention absorbed into (A_s).  This upper
bound is generically exact.

Let

\[
 z(v)=\operatorname{svec}(vv^T)
 \in\mathbb R^{N_2},
 \qquad N_2={n(n+1)\over2}.
\]

The `(2,2)` catalecticant of (5.1) is

\[
 \operatorname{Cat}_{22}(L_T)
 =Z\operatorname{Diag}(d)Z^T,
 \qquad
 Z=[z(v_{s,i})]_{s,i}.
\tag{5.2}
\]

For generic dense maps, the columns of (Z) are independent while their
number (R=(T+1)n) is at most (N_2).  Nonzero cohort weights then give

\[
 \operatorname{rank}\operatorname{Cat}_{22}(L_T)=R.
\tag{5.3}
\]

Every symmetric Waring decomposition with (r) rays has catalecticant rank
at most (r).  Hence

\[
 \boxed{\operatorname{wrank}(L_T)=R=(T+1)n
 \quad\text{generically for }R\le N_2.}
\tag{5.4}
\]

At (n=1024), ten full birth cohorts contain only 10,240 rays, far below
(N_2=524,800), so the generic exact rank grows linearly throughout the
whole suffix.  A cap (2n) cannot be an exact complete-history theorem after
the third cohort is presented.

## 6. Collision projection can turn one Waring ray into quadratic rank

Low rank before complete-old subtraction says nothing about rank after it.
There is a finite exact counterexample using only the all-distinct projector.
Let (v_i\ne0) for every coordinate and let
(mathcal P_{1111}) retain only entries with four pairwise distinct indices.
Flatten (mathcal P_{1111}(v^{\otimes4})) on unordered off-diagonal pair
indices (P=\{\{i,j\}:i<j\}).  Its pair-pair matrix is

\[
 M=D_v A_{KG(n,2)}D_v,
 \qquad
 (D_v)_{\{i,j\},\{i,j\}}=v_iv_j,
\tag{6.1}
\]

where (A_{KG(n,2)}) is the adjacency matrix of the Kneser graph: two pairs
are adjacent exactly when they are disjoint.  Its three eigenvalues are

\[
 {n-2\choose2},\qquad -(n-3),\qquad 1,
\tag{6.2}
\]

with their standard multiplicities.  They are all nonzero for (n\ge4).
Therefore

\[
 \operatorname{rank}M={n\choose2}.
\tag{6.3}
\]

The same flattening of one fourth power has rank one, and a sum of (r)
fourth powers has rank at most (r).  Consequently

\[
 \boxed{
 \operatorname{wrank}\bigl(
 \mathcal P_{1111}(v^{\otimes4})
 \bigr)\ge {n\choose2}.}
\tag{6.4}
\]

Thus removing the `(4),(31),(22),(211)` collision faces can raise one dense
ray from rank one to at least 523,776 at width 1,024.  The same lower bound
holds for the more relevant residual which keeps both `211` and `1111` but
removes `(22)`.  On off-diagonal pair indices, put
(z_{\{i,j\}}=v_iv_j).  Every two *distinct* pair indices then represent
either a `211` or a `1111` entry, while the equal-pair diagonal is the removed
`22` entry.  The flattening is exactly

\[
 \operatorname{Diag}(z)
 (\mathbf1\mathbf1^T-I)
 \operatorname{Diag}(z),
\tag{6.4a}
\]

which has full rank \({n\choose2}\).  Thus either the pure-all-distinct
quotient or the combined `211+1111` quotient has quadratic Waring rank for a
single generic dense ray.

This does not forbid action-level Schur subtraction.  If the full ray can be
contracted locally against the exact future query, its old scalar response
can be subtracted afterward without factorizing the residual tensor.  It does
forbid transporting the tensor-level complete-collision residual as the same
small Waring bank.

More generally, if

\[
 L_\perp=L-\sum_j\beta_jO_j,
\tag{6.5}
\]

and the old tensors do not use the identical factor set, concatenation is the
only automatic CP formula.  Generic squared-factor independence makes the
catalecticant rank additive up to (N_2).  A shared bank survives an old
projector only under the explicit invariant-subspace condition

\[
 \Pi_{old}\operatorname{span}\{v_\alpha^{\otimes4}\}
 \subseteq
 \operatorname{span}\{v_\alpha^{\otimes4}\}.
\tag{6.6}
\]

The complete endpoint/corpus Schur projector is not known to satisfy (6.6).

## 7. What the literal cap-2n CP4 tape really makes resident

The documented source operations are

\[
 U\leftarrow W^TU,
 \qquad
 U\leftarrow D_gU,
 \qquad
 U\leftarrow[U,I_n],
\tag{7.1}
\]

followed by top-energy selection to `cap=2n`.  Consider one full identity
cohort (U_c=I_n) at its birth and transport it only by the first two maps.
Its Gram obeys

\[
 G_c=U_cU_c^T,
 \qquad
 G_c^+=D_gW^TG_cWD_g.
\tag{7.2}
\]

Therefore a Price metric born as (I_n) and transported by **the same
separable maps** is represented exactly by that cohort.  The CP coefficients
(lambda) are irrelevant to (7.2).

The horizon is exact:

- after the first birth there is one full cohort;
- after the second birth there are exactly (2n) columns, so both cohorts
  are resident before any cap loss;
- the third birth presents (3n) columns to the cap-(2n) energy selector.

At that third birth, the selector may retain arbitrary individual columns
from all three cohorts.  Its ordering is controlled by the CP weights and
factor energies, not by preservation of the metric (7.2).  Hence

\[
 \boxed{
 \text{no complete cohort, and therefore no exact old }G_c,
 \text{ is resident by theorem after the third birth}.}
\tag{7.3}
\]

A selected subcohort represents (U_cSU_c^T), not (U_cU_c^T).  Rebuilding
the missing columns from the weights is recomputation, not residency.

There is a second mismatch.  The D10 covariance reverse uses a general
pairwise Schur response kernel, whereas (7.2) uses the rank-one separable
kernel (gg^T).  Equality requires a literal theorem that the relevant D10
kernel is this same separable gate.  Marginal gate equality alone is
insufficient for correlated ReLU pairs.

Thus the cohort theorem can supply a short-lived (G)-operand for a special
Price route.  It does not supply the Waring factors of (L_l^{fut}).

## 8. Literal CP4/D10 future-query residency audit

The D10 word has the typed recurrence

\[
 H_l=W_{l+1}B_{l+1}W_{l+1}^T,
 \qquad
 B_l=D_l\odot H_l,
\tag{8.1}
\]

plus scalar accumulators.  These are matrices in
(operatorname{Sym}_n).  The required reverse object for a generic Price
q211 step is the operator

\[
 \mathcal H_{L_l}:G\longmapsto\mathsf C_G(L_l),
\tag{8.2}
\]

or a shared-Waring decomposition of (L_l).  One value of (8.2), one partial
trace, or one D10 matrix does not determine the operator; the harmonic
trace-zero counterexample from the Price-Hessian barrier changes
(mathsf C_G) for anisotropic (G) while leaving (mathsf C_I) unchanged.

The joint literal state

\[
 (U_l,\lambda_l)_{CP4}\quad\text{and}\quad(B_l,H_l)_{D10}
\tag{8.3}
\]

therefore does not contain a factorization of (L_l^{fut}): the first tuple
is primal and target blind; the second is reverse but only second order.  The
common use of (W_l) proves algebraic adjointability, not buffer identity.

If the retained primal CP4 tensor is inserted for (L_l), (2.2) is exact but
its response is incumbent-owned.  If instead (L_l) denotes a true-minus-old
or complete-old-projected future costate, neither tuple in (8.3) determines
its factors.  Projection can also invoke the rank explosion (6.4).

The only exact deployment escape is one of the following:

1. construct a target-conditioned Waring bank for the future costate and show
   that all suffix maps are dense/separable with a bounded number of births;
2. prove that the complete-old projector preserves its common factor span as
   in (6.6);
3. contract each new/dropped ray at birth against an already resident exact
   future query, subtract old response at the scalar level, and retire it; or
4. prove a low-rank quotient of (8.2) directly.

None is a consequence of the present CP4/D10 tape definition.

## 9. Ownership split and final verdict

The actions separate as follows.

- **Retained CP4 action:** exact shared-factor evaluation, but old/duplicate.
- **Coordinate birth before it enters CP4:** diagonal and cheap, but its
  same-site visible action is old unless a strict-depth response survives the
  complete arena.
- **Dropped CP4 tail:** factors are briefly available at the selection seam;
  its all-distinct response may be new, but future-query residency and
  action-level old subtraction are open.  Once discarded, it is not a
  resident carrier.
- **True-minus-AQ or true-minus-D11 residual:** potentially novel, but no CP4
  factorization is supplied by the incumbent bank.
- **Tensor-level complete-old residual:** shared-Waring rank is not stable;
  (6.4) gives a quadratic lower bound from one ray.

Hence shared Waring is a real `O(n^3)` algebraic escape, but not a resident
novel-value escape in the current architecture.  It promotes only after a
literal future-costate factor bank or local contract-and-retire receipt is
identified at the synchronized complete-old cut.

## Graph-ready nodes

### PRICE_SHARED_WARING_CONTRACTION_THEOREM

- status: solved_exact;
- formula:
  \(\mathsf C_G(L)=V^T\operatorname{Diag}(\lambda\odot
  \operatorname{diag}(VGV^T))V\);
- transport: (V\mapsto VA), with
  \(\mathsf C_G((A^T)^{\otimes4}L)=A^T\mathsf C_{AGA^T}(L)A\);
- cost: (2R/n) dense-product equivalents for contraction and (3R/n)
  including a fresh factor push.

### PRICE_SHARED_WARING_RESET_RANK_GROWTH

- status: solved_generic_exact_rank;
- reset: dense/separable transport preserves rank, each full coordinate birth
  appends (n) rays;
- certificate: `(2,2)` catalecticant rank is ((T+1)n) for generic cohorts
  while ((T+1)n\le n(n+1)/2);
- consequence: cap (2n) cannot exactly retain three or more generic full
  birth cohorts.

### PRICE_COLLISION_PROJECTION_WARING_EXPLOSION

- status: closed_shared_bank_after_tensor_projection;
- witness: the all-distinct projection of one dense ray has off-pair
  flattening equal to a diagonally scaled Kneser `KG(n,2)` adjacency;
- lower bound: Waring rank at least \({n\choose2}=523,776\) at width 1,024;
- escape: scalar action-level old subtraction after exact local contraction,
  not tensor-level residual transport.

### PRICE_CP4_COHORT_METRIC_HORIZON

- status: solved_exact_two_cohort_horizon;
- identity: a full identity cohort transported by `_cp4_push` and the
  separable gate satisfies (G=UU^T) for the matching Price metric;
- horizon: two complete cohorts fit cap (2n); the third birth invokes
  arbitrary energy selection, so no complete old cohort survives by theorem;
- limitation: supplies a special (G)-operand, not the future Waring costate
  (L), and does not equal a generic correlated D10 Schur kernel.

### PRICE_CP4_D10_FUTURE_QUERY_RESIDENCY_NOGO

- status: closed_for_literal_state;
- CP4 object: primal target-blind ((U,\lambda));
- D10 object: reverse two-index ((B,H)) plus scalars;
- missing object: shared-Waring factors or full Hessian operator of the
  complete-old-projected strict-depth quartic costate;
- implication: same weights and resident primal pushes do not constitute
  reverse-query buffer reuse.

### PRICE_SHARED_WARING_NOVEL_RESIDUAL_GATE

- status: open_conditional_contract_and_retire;
- incumbent retained CP4 action is old;
- candidate: dropped/new/true-minus-old ray contracted at its birth against
  an exact resident future query, then Schur-subtracted in response space;
- promotion: bounded factor count before contraction, exact future-query
  residency, synchronized ownership subtraction, and endpoint value receipt.

## 10. Action-level contract-and-retire against the literal D10 word

The only remaining possibility is to avoid transporting a projected quartic
tensor.  This section derives exactly what the existing D10 reverse matrices
can contract while a new or dropped CP4 factor is still available.

Let (O_l^{31}) be the literal directed-K31 read at a synchronized cut,

\[
 [O_l^{31}T]_{ij}=T(e_i,e_i,e_i,e_j),
\tag{10.1}
\]

with the deployed diagonal/strict-off-diagonal convention represented by a
fixed mask (M_{31}).  The D10 recurrence forms a matrix (Q_l^{31}) which
already includes the `1/6`, gate, transpose, and covariance-injection
normalization, so its exact source action is

\[
 \rho_l^{31}(T)
 =\langle Q_l^{31},M_{31}\odot O_l^{31}T\rangle_F.
\tag{10.2}
\]

For one signed Waring ray (T=\lambda f^{\otimes4}), put
(ar Q=M_{31}\odot Q_l^{31}).  Then

\[
 \boxed{
 \rho_l^{31}(\lambda f^{\otimes4})
 =\lambda(f^{\odot3})^T\bar Q f.}
\tag{10.3}
\]

For a batch (F=[f_1,\ldots,f_R]\in\mathbb R^{n\times R}), define the
per-ray response vector

\[
 a(F,\bar Q)
 =\operatorname{colsum}\left(
 F^{\odot3}\odot(\bar QF)
 \right)\in\mathbb R^R.
\tag{10.4}
\]

The exact batch scalar and factor JVP are

\[
 \boxed{
 \rho_l^{31}(F,\lambda)=\lambda^Ta(F,\bar Q),}
\tag{10.5}
\]

\[
 \boxed{
 \nabla_{f_\alpha}\rho_l^{31}
 =\lambda_\alpha\left[
 3f_\alpha^{\odot2}\odot(\bar Qf_\alpha)
 +\bar Q^T f_\alpha^{\odot3}
 \right].}
\tag{10.6}
\]

Thus, once (F) and (ar Q) are resident, all (R) ray actions need one
dense (n\times n) by (n\times R) product plus lower-order reductions.
No quartic costate is needed for this *typed K31 path*.

The analogous visible collision contractions are

\[
 \rho^{22}(F,\lambda)
 =\sum_\alpha\lambda_\alpha
 (f_\alpha^{\odot2})^TQ^{22}f_\alpha^{\odot2},
\tag{10.7}
\]

and an elementwise K4 diagonal reduction.  They are available only when the
corresponding literal reverse masks exist.  A D10 K31 matrix is not a q211 or
q1111 mask merely because all arrays derive from one quartic source.

## 11. Same-site annihilation theorem and the strict-depth escape

Let (O_l) denote the complete literal equality interface used by the old
CP4/star/Q24 consumer at the cut: at minimum `(4),(31),(22)` with its exact
typed conventions.  Every same-site reverse query constructible solely from
those matrices lies in

\[
 \operatorname{ran}O_l^*.
\tag{11.1}
\]

If the local tensor residual is the equality-dark component

\[
 T_l^\circ=P_{\ker O_l}T_l,
\tag{11.2}
\]

then for every literal old/D10 multiplier (q_l),

\[
 \boxed{
 \langle O_l^*q_l,T_l^\circ\rangle
 =\langle q_l,O_lT_l^\circ\rangle=0.}
\tag{11.3}
\]

This is the exact typing no-go for a same-site contract-and-retire proposal.
Computing (10.3) for the full ray and subtracting the same visible old action
leaves zero local novelty.  It does not acquire the collision-dark tensor.

Let (J_l^{(4)}) be at least one future quartic transition and let
(O_{l+1}) be the next visible read.  The first strict-depth query is

\[
 \boxed{
 (J_l^{(4)})^*O_{l+1}^*q_{l+1}.}
\tag{11.4}
\]

Unlike (11.1), (11.4) need not lie in (operatorname{ran}O_l^*).  Dense
mixing is precisely what makes

\[
 O_{l+1}J_l^{(4)}P_{\ker O_l}\ne0
\tag{11.5}
\]

generically.  For a separably transported ray
(f=A_l v), (11.4) is evaluated by (10.3) using the *next-cut* D10 matrix.
This is an exact one-transition K31 contract-and-retire feature.

Its scope is also exact.  If the declared added path is

\[
 T_l\xrightarrow{J_l^{(4)}}K31_{l+1}
 \xrightarrow{\text{D10 covariance tangent}}\text{endpoint},
\tag{11.6}
\]

then the next-cut (Q_{l+1}^{31}) supplies its complete future action.  It
does **not** supply the full true quartic response through q211, q1111, or any
other branch outside (11.6).  A tensor in
(ker O_{l+1}^{31}) with nonzero later response is an immediate type
counterexample.  Thus (11.6) is a lawful new scalar feature, not a proof that
the quartic future costate has become resident.

## 12. Complete-old Schur subtraction must be done in response space

The local equality projection (11.2) and the complete endpoint old projection
are different operations.  Let (z\in\mathbb R^N) be the corpus response
obtained by summing the strict-depth actions (10.5) over every chosen birth,
drop, and layer.  Let (E\in\mathbb R^{N\times p}) be the complete old design
at the exact package lineage and let (V\succ0) be the value metric.  The
lawful new response is

\[
 \boxed{
 z_\perp=(I-\Pi_E^V)z,
 \qquad
 \Pi_E^V=E(E^TVE)^\dagger E^TV.}
\tag{12.1}
\]

Projecting each ray or each layer separately is invalid because the old
coefficients and cross-layer covariances are joint.  Equation (12.1) avoids
the tensor Waring explosion in Section 6: evaluate the easy full-ray scalar
first, then subtract old responses.

For target residual (e=(I-\Pi_E^V)y), the exact one-coefficient conditional
gain is

\[
 \boxed{
 \Delta\operatorname{MSE}
 ={\langle e,z_\perp\rangle_V^2
   \over\langle z_\perp,z_\perp\rangle_V},}
\tag{12.2}
\]

with zero gain when (z_\perp=0).  Equivalently, one may form an old-adjusted
reverse query as a linear combination of endpoint adjoints, but only if the
reverse words for every required old column are actually supplied.  One
literal D10 word does not make the whole projector in (12.1) resident.

This gives the sharp ownership split:

- same-site retained CP4 action is in (E) and vanishes in (12.1);
- a delayed dropped-ray response may survive because (11.5) changes its
  visible geometry;
- if the incumbent depth arena already contains that delayed response,
  (12.1) kills it even though it was locally new.

## 13. Coordinate-birth fusion theorem

For a full coordinate birth

\[
 T_d=\sum_{a=1}^nd_a e_a^{\otimes4},
\tag{13.1}
\]

let (F=A_l=[f_1,\ldots,f_n]) be the matrix whose columns are the transported
coordinate rays at the chosen future K31 cut.  Equations (10.4)--(10.5) give

\[
 \boxed{
 \rho_{birth}
 =\sum_{a=1}^n d_a
 (f_a^{\odot3})^T\bar Q f_a
 =d^T\operatorname{colsum}
 \left(F^{\odot3}\odot(\bar QF)\right).}
\tag{13.2}
\]

Hence all (n) coordinate rays fuse into one dense product per layer once
(F) is resident.  If the transition is a single affine map from the identity
birth, (F) is just that weight/gate matrix and does not require a separate
factor push.  If the rays have already reached a drop seam and must cross one
additional dense map, obtaining (F) costs one factor push and (13.2) costs a
second dense product.

There is no cross-layer one-product fusion for generic weights.  The ten
products have different (F_l,Q_l); stacking them forms a block-diagonal
operation whose row volume and arithmetic are the sum of the ten products.
A common eigensystem/intertwiner could be an exception, but must be literal.

Coordinate birth has an ownership trap.  At its birth,

\[
 \Delta(d)=\operatorname{Star4}(\operatorname{Diag}d),
\tag{13.3}
\]

so the CP4 coordinate block and the incumbent diagonal Star tensor are
identical.  Pure affine transport preserves that equality.  Therefore its
same-site and first affine visible actions are old and (12.1) returns zero.
Novelty can begin only after a nonlinear reset treats retained CP factors and
the old reconstructed Star/Q24 state differently.  At that first divergence,
the next strict K31 action is still given by (13.2), but it is not a free
coordinate-birth action: the appropriate post-reset factors and next-cut D10
query must both be present.

## 14. Sharp arithmetic and adjusted-score gates

Let (R_l) be the number of rays contracted at layer (l).  In dense-product
equivalents, the exact leading incremental count is

\[
 \boxed{
 C_{new}
 =\sum_l(c_{push,l}+c_{act,l}){R_l\over n},}
\tag{14.1}
\]

where each coefficient is zero or one according to literal buffer reuse.
The action coefficient is zero only if the exact delayed K31 plane for that
batch is already materialized at the same cut; a current-cut dropped plane is
not such a buffer because its future transform depends on the omitted
`211/1111` entries.

Consequently:

- ten full coordinate batches with resident future factors cost at least
  (C_{new}=10);
- ten full dropped batches needing one further push cost at least
  (C_{new}=20);
- the additive one-equivalent slot requires
  
  \[
  \sum_l(c_{push,l}+c_{act,l})R_l\le n,
  \tag{14.2}
  \]
  
  so it cannot contain ten rank-(n) layer batches;
- removing D10 does not fund this route, because (Q_l^{31}) is the object
  which makes (10.3) possible.  A replacement would need a different reverse
  construction and loses the claimed residency.

If the held f32 pack has utilization (u_0=0.5629) and one dense equivalent
adds (kappa=0.00161), a package with raw reduction fraction (g) improves
adjusted score only if

\[
 \boxed{
 g>1-{u_0\over u_0+\kappa C_{new}}.}
\tag{14.3}
\]

For `C_new=10` this break-even reduction is about (2.78\%); for
`C_new=20` it is about (5.41\%).  To demand an adjusted improvement fraction
(eta), replace (14.3) by

\[
 \boxed{
 g\ge1-(1-\eta){u_0\over u_0+\kappa C_{new}}.}
\tag{14.4}
\]

At (eta=20\%\), the corresponding raw gates are about (22.2\%) and
(24.3\%).  These adjusted gates do not override the harder current
49-equivalent package cap: an additive `10` or `20` is infeasible before
value.

## 15. Contract-and-retire verdict

There is an exact local GO and a package-level NO-GO.

**Exact local GO.**  A known ray batch can be pushed to the next K31 cut and
contracted against the literal next-cut D10 matrix by (10.5).  A full
coordinate cohort fuses into one `O(n^3)` product by (13.2).  Response-level
Schur subtraction (12.1) avoids tensor residual factorization.

**No complete future-query GO.**  The contraction sees only the declared
K31 path (11.6).  Same-site old subtraction is identically zero by (11.3),
and q211/q1111 future effects outside that path still require a quartic
costate or another typed consumer.

**Current package NO-GO.**  The first potentially novel coordinate/dropped
actions occur only after a nonlinear divergence.  Across ten full batches
they cost at least ten equivalents with every factor push free, or twenty
when dropped factors need one more push, versus one additive unit.  The route
can reopen only if the complete value concentrates into ray batches obeying
(14.2), or if a paid buffer is identified whose literal contents are the
next-cut delayed plane/action—not merely the current dropped plane.

## Additional graph-ready nodes

### PRICE_D10_RAY_ACTION_FUSION

- status: solved_exact_typed_K31_action;
- scalar: \(\lambda(f^3)^TQ^{31}f\);
- batch vector: column sums of
  \(F^3\odot(Q^{31}F)\);
- cost: one (n\times n) by (n\times R) product with factors resident;
- scope: complete for the declared K31-to-D10 path only.

### PRICE_SAMESITE_OLD_ACTION_ANNIHILATION

- status: closed_proof;
- theorem: (T^\circ\in\ker O_l) is annihilated by every literal same-site
  query in \(\operatorname{ran}O_l^*\);
- implication: full-ray contraction minus matching old visible action has zero
  same-site novelty;
- strict-depth escape: ((J_l^{(4)})^*O_{l+1}^*q\) may detect the residual and
  requires the actual future transition.

### PRICE_COORDINATE_BIRTH_BATCH_FUSION

- status: solved_local_formula_old_at_first_transition;
- formula: (13.2), one dense action product for all (n) axis births;
- ownership: coordinate CP4 birth equals diagonal Star4 at birth and remains
  equal under pure affine transport;
- novelty: only after a nonlinear CP-versus-old reconstruction divergence.

### PRICE_CONTRACT_RETIRE_COMPLETE_OLD_SCHUR_GATE

- status: exact_response_projection_solved / value_open;
- residual: (z_\perp=(I-\Pi_E^V)z\) after the joint cross-layer response is
  assembled;
- close: (z_\perp=0) when the depth arena owns the delayed action;
- gain: equation (12.2), not local tensor novelty or unprojected ray energy.

### PRICE_CONTRACT_RETIRE_ADDITIVE_COST_NOGO

- status: closed_for_ten_full_batches;
- minimum with resident factors: ten equivalents;
- minimum for dropped factors needing one extra transition: twenty;
- hard additive gate: weighted total ray count (14.2) must not exceed (n);
- adjusted break-even: 2.78%/5.41% raw for ten/twenty equivalents, but both
  already fail the one-equivalent package slot.

## 16. Causal liveness of a dropped-ray/future-query contraction

Equation (10.5) assumes simultaneous access to the dropped factor bank at
layer (l) and the strict future matrix (Q_{l+1}^{31}).  A standard causal
forward/reverse schedule does not provide that automatically.

The CP4 energy selector sees and discards its dropped bank during the forward
sweep.  The D10 matrix (Q_{l+1}^{31}) depends on the endpoint condition and
all later reset masks, so it is available only after the old forward suffix
has been completed and its reverse sweep has reached (l+1).  Therefore an
exact implementation must choose at least one of:

1. retain every dropped (F_l,\lambda_l) until the reverse sweep, requiring
   
   \[
   M_{drop}=\sum_l nR_l+\sum_lR_l
   \tag{16.1}
   \]
   
   stored scalars before workspaces;
2. complete the old forward and D10 reverse sweeps, retain the (Q_l^{31})
   matrices, and deterministically replay the CP4 forward selectors to
   regenerate each dropped bank; or
3. prove a literal fused schedule in which an already-computed future query
   is cached before the candidate forward replay.

There is no one-pass causal option which both forgets the ray at its forward
drop and later contracts that same ray against an as-yet-unconstructed future
query.  Recomputability from weights is not buffer residency.  The arithmetic
gate (14.1) must therefore be augmented by the storage in (16.1) or by the
literal replay cost; the current CP4 push and D10 reverse being separately
paid does not prove their operands overlap in lifetime.

### PRICE_CONTRACT_RETIRE_CAUSAL_LIVENESS_GATE

- status: exact scheduling trilemma;
- conflict: dropped CP4 factors occur on the forward sweep, strict D10 queries
  occur on the later reverse sweep;
- exact options: store \(\sum_l nR_l\) factor entries, replay CP4 after caching
  D10 queries, or exhibit a literal two-pass fused buffer schedule;
- no-go: one-pass "contract before discard" without a precomputed future
  query;
- promotion: add the chosen storage/replay receipt to (14.1), not just the
  local batch-product count.

## 17. Minimum sufficient statistic for an unknown future K31 query

Fix the strict future cut after every required factor push/reset has already
been applied.  For a discarded cohort

\[
 F=[f_1,\ldots,f_R]\in\mathbb R^{n\times R},
 \qquad \lambda\in\mathbb R^R,
\]

define its directed K31 plane

\[
 \boxed{
 A(F,\lambda)
 =\sum_{\alpha=1}^R\lambda_\alpha
   f_\alpha^{\odot3}f_\alpha^T
 =F^{\odot3}\operatorname{Diag}(\lambda)F^T.}
\tag{17.1}
\]

Every later matrix query is exactly

\[
 \rho_Q(F,\lambda)=\langle Q,A(F,\lambda)\rangle_F.
\tag{17.2}
\]

Thus (A) is the Riesz/minimum **linear** sufficient statistic for all future
(Q).  Testing (17.2) on the matrix units (Q=E_{ij}) recovers every entry of
(A), so any other exact statistic for arbitrary (Q) must determine (A).
For a strict-off-diagonal convention, replace (A) by
(M_{31}\odot A); its diagonal is the separately typed K4 collision.

The factor bank is a nonlinear encoding of (17.1).  It can be smaller than a
dense plane when (R<n), but it is already essentially information-minimal.

## 18. Sharp stable-sketch dimension lower bound

Let

\[
 \Phi_R:(F,\lambda)\longmapsto A(F,\lambda).
\]

Each ray has the scale gauge
((f_\alpha,\lambda_\alpha)\sim
(c f_\alpha,c^{-4}\lambda_\alpha)), so the parameter quotient has dimension
(Rn).  This upper bound is attained until the ambient matrix dimension
saturates.

Take (R\le n), (lambda_\alpha=1), and
(f_\alpha=e_\alpha).  For a perturbation (h_\alpha),

\[
 D(f^{\odot3}f^T)_{e_\alpha}[h_\alpha]
 =3(e_\alpha^{\odot2}\odot h_\alpha)e_\alpha^T
  +e_\alpha^{\odot3}h_\alpha^T.
\tag{18.1}
\]

As (h_\alpha) varies, (18.1) spans the complete (alpha)-th matrix row;
different (alpha)'s give disjoint rows.  Hence

\[
 \operatorname{rank}D\Phi_R=Rn
 \qquad(R\le n).
\tag{18.2}
\]

At (R=n), (Phi_n) is therefore a submersion and its image contains an
open subset of (mathbb R^{n\times n}).  For (R>n), the first (n) rays
already give that open set.  Consequently the cohort image has exact local
dimension

\[
 \boxed{d_R=n\min\{R,n\}.}
\tag{18.3}
\]

Now let a one-pass sketch
(sigma(F,\lambda)\in\mathbb R^m) be continuous and locally differentiable,
and suppose a decoder returns (17.2) for every future (Q).  Matrix-unit
queries give a decoder from (sigma) to (Phi_R).  The chain rule and
(18.2), or invariance of local dimension, imply

\[
 \boxed{m\ge n\min\{R,n\}.}
\tag{18.4}
\]

At a full discarded cohort (R=n=1024), at least (n^2=1,048,576) real
coordinates are necessary.  If only strict off-diagonal queries are allowed,
the same point gives the sharp lower bound (n(n-1)=1,047,552).  Arbitrary
discontinuous encodings of infinitely precise reals are excluded; they are
neither algebraic floating-point algorithms nor stable deployment sketches.

Equation (18.4) assumes no replay of the generating prefix.  If all weights,
gates, and deterministic selection decisions are retained, the cohort can be
regenerated with less memory, but then the replay arithmetic is the payment.

## 19. Exact generic storage/arithmetic frontier

For an arbitrary future query, the exact one-pass choices are:

1. **Retain the rays.**  Store (nR+R) scalars.  Once (Q) arrives, (10.4)
   costs (R/n) dense-product equivalents.  A not-yet-applied strict future
   push costs another (R/n).
2. **Store the future-cut plane.**  Form (17.1) for (R/n) equivalents, store
   (n^2) scalars, and later take an (O(n^2)) inner product with (Q).
   Strict-off-diagonal storage can omit (n) diagonal entries.
3. **Replay.**  Store no cohort statistic beyond deterministic checkpoint
   metadata, cache the future query, and regenerate (F).  This pays every
   replayed factor push, gate, selection, and plane/action contraction.

The optimal stable storage scale is therefore

\[
 \boxed{M_{generic}=\Theta(n\min\{R,n\})}
\tag{19.1}
\]

unless replay is allowed.  For (R<n), the original factor bank attains this
scale; forming a dense plane wastes memory.  For (R\ge n), the plane attains
the (n^2) lower bound and can compress a cap-(2n) bank by about a factor of
two.  A generic low-rank factorization of (A) cannot beat (18.4): at
(R=n), (A) is generically full rank.

The future-cut qualifier is binding.  The current dropped K31 plane cannot be
updated to the strict future plane from two-index data; Section 11's
`211/1111` witness proves the missing information.  The factors must survive
through the strict transition before (17.1) is formed.

## 20. General query-subspace theorem

Let (mathcal Q\subseteq\mathbb R^{n\times n}) be a **known linear subspace**
containing every possible future query.  Only the class of (A) modulo
(mathcal Q^\perp) matters.  If (B_1,\ldots,B_d) is a basis of
(mathcal Q), an exact sufficient statistic is

\[
 s_j=\langle B_j,A\rangle,
 \qquad j=1,\ldots,d.
\tag{20.1}
\]

For a full cohort whose plane ranges over an open set, no stable exact sketch
uses fewer than

\[
 \boxed{\dim\mathcal Q}
\tag{20.2}
\]

coordinates.  For (R<n), the generic lower bound is
(min\{Rn,\dim\mathcal Q\}) when the projection of the cohort tangent is in
general position.

This theorem makes the distinction between **rank** and **known query
subspace** exact.  A nonlinear family can have low pointwise rank while its
linear span is the entire matrix space.

## 21. Arbitrary rank-k queries do not compress the sketch

Suppose the future query is revealed later only with the promise
(operatorname{rank}Q\le k), for any (k\ge1).  Rank-one matrices already
contain every matrix unit,

\[
 E_{ij}=e_ie_j^T.
\tag{21.1}
\]

Therefore the linear span of the allowed query family is all
(mathbb R^{n\times n}), and (18.4) is unchanged.  In particular,

\[
 \boxed{
 \text{an arbitrary future rank-one query still requires the full generic
 statistic.}}
\tag{21.2}
\]

There are two exact exceptions.

1. If both factors of (Q=XY^T) are known before the cohort is discarded,
   evaluate (10.5) immediately and retain one scalar.
2. If all query columns/rows lie in fixed known subspaces, use the compressed
   statistic in the next section.

Approximate randomized bilinear sketches are outside this exact theorem and
need a separate endpoint-error probability receipt.

## 22. Fixed left/right query spaces

Let

\[
 Q=UKV^T,
 \qquad U\in\mathbb R^{n\times p},
 \quad V\in\mathbb R^{n\times q},
\tag{22.1}
\]

where (U,V) are known at discard time and (K) is revealed later.  The
minimum linear statistic is

\[
 \boxed{
 S_{UV}=U^TAV
 =(U^TF^{\odot3})\operatorname{Diag}(\lambda)(F^TV)
 \in\mathbb R^{p\times q}.}
\tag{22.2}
\]

It uses (pq) stored scalars, and
(ho_Q=\langle K,S_{UV}\rangle_F).  Its direct leading arithmetic is

\[
 \boxed{
 C_{UV}simeq
 {R(p+q)\over n^2}+{Rpq\over n^3}}
\tag{22.3}
\]

dense-product equivalents, before a nonresident future factor push.

Promising that (K) itself has rank at most (k\ge1) does not reduce the
(pq) storage: rank-one (p\times q) matrices span the whole coefficient
space.  If only the right space (V) is fixed and the left factor is
arbitrary, store (AVin\mathbb R^{n\times q}); it costs about
(2Rq/n^2) equivalents to form and (nq) scalars to store.

## 23. Shared-eigenspace query class

Suppose every future query is symmetric and diagonal in one **known fixed**
orthogonal basis (S=[s_1,\ldots,s_n]):

\[
 Q=S\operatorname{Diag}(\theta)S^T.
\tag{23.1}
\]

Then the exact statistic is only

\[
 \boxed{
 h=\operatorname{diag}(S^TAS),
 \qquad
 h_i=\sum_\alpha\lambda_\alpha
 (s_i^Tf_\alpha^{\odot3})(s_i^Tf_\alpha).}
\tag{23.2}
\]

It stores (n) scalars.  Forming (S^TF^{\odot3}) and (S^TF) costs
approximately (2R/n) equivalents, followed by rowwise reductions.  A fixed
pair of left/right bases with diagonal coefficients has the identical
(n)-scalar result.

If the eigenbasis is not known until the reverse sweep and may be arbitrary,
the family of all symmetric eigensystems spans (operatorname{Sym}_n); the
minimum returns to (n(n+1)/2).  "The queries commute after they are built"
is insufficient.  The common basis must be literal and available before the
cohort is discarded.

## 24. D10 Schur/Kraus query classes

The D10 K31 query has an entrywise response-kernel form.  Abstract one branch
as

\[
 Q=H\odot G,
 \qquad
 G=\sum_{t=1}^s a_tb_t^T.
\tag{24.1}
\]

If the reverse matrix belongs to fixed spaces

\[
 H=UKV^T,
\tag{24.2}
\]

then

\[
 H\odot G
 =\sum_{t=1}^sD_{a_t}U K V^TD_{b_t}.
\tag{24.3}
\]

All future (K)'s are answered by the (p\times q) statistic

\[
 \boxed{
 S_{G,U,V}(A)
 =\sum_{t=1}^s
 U^TD_{a_t}A D_{b_t}V.}
\tag{24.4}
\]

Using the rays directly, a transparent upper census is

\[
 \boxed{
 C_{G,U,V}lesssim
 s\left[{R(p+q)\over n^2}+{Rpq\over n^3}\right],}
\tag{24.5}
\]

with (pq) stored scalars.  Common diagonal factors or stacked projections
can lower constants, but not the statistic dimension.

Kraus rank by itself gives no memory theorem.  If (H) is arbitrary and
(G) has full entrywise support, Hadamard multiplication by (G) is
injective.  Even (s=1), (G=ab^T) with all coordinates nonzero, preserves
the full dimension of the (H)-space.  Therefore:

- arbitrary nonsymmetric (H): (n^2) query dimensions;
- arbitrary symmetric (H): (n(n+1)/2) query dimensions, with the exact
  statistic the appropriate symmetric part of (G\odot A);
- support-sparse (G): the dimension can fall to the number of live typed
  entries;
- (H=UKV^T) with fixed small (U,V): the (pq) escape (24.4).

The literal D10 formula has two oriented gate terms and a transpose.  Apply
(24.4) to each and add them before storage; the same conclusion holds.

## 25. Endpoint rank-one is not a generic memory escape

Before a Schur reset, a scalar endpoint may produce a symmetric rank-one
matrix (H=hh^T).  If (h) is not known at discard time and can vary over
(mathbb R^n), the family ({hh^T}) spans all of
(operatorname{Sym}_n).  Hence its exact statistic is

\[
 \operatorname{sym}A,
 \qquad \text{dimension }{n(n+1)\over2}.
\tag{25.1}
\]

After a full-support D10 Schur gate,

\[
 (hh^T)\odot G=D_hGD_h.
\tag{25.2}
\]

The span is still a full (n(n+1)/2)-dimensional image of
(operatorname{Sym}_n), even when (G) has matrix/Kraus rank one.  Moreover
the correlated ReLU Schur gate generally raises the pointwise matrix rank of
(25.2), so terminal rank one is not preserved as a low-rank buffer.

If (h=Uc) belongs to a known (k)-dimensional subspace, the exact exception
is

\[
 \operatorname{sym}(U^TAU)
 \quad\text{or its Kraus-adjusted version (24.4)},
\tag{25.3}
\]

requiring (k(k+1)/2) scalars.  If (h) itself is known before discard, the
one scalar action can be computed immediately.

For a nonsymmetric rank-one family (xy^T) with independent arbitrary
(x,y), (21.1) applies and the full (n^2) statistic is necessary.

## 26. Sharp memory/fusion verdict

For a full discarded CP4 cohort and a generic future D10/K31 query, there is
no exact subquadratic one-pass sketch.  The sharp choices are:

- retain the (n\times n) factor cohort until the reverse query;
- push it to the strict future cut, form and retain its (n\times n) K31
  plane, then discard the factors;
- or replay the cohort after the query is known.

Low pointwise query rank, low Kraus rank, and endpoint scalarity do not change
this theorem because their adaptive query families span full matrix or full
symmetric space.  Exact subquadratic storage exists only from a query space
known in advance: fixed left/right spaces, a fixed common eigensystem, sparse
typed support, or a finite query dictionary.

For the current ten-layer full-cohort route, storing future planes costs ten
(n^2) arrays and forming them costs one (R/n=1) equivalent per layer after
the factors reach the strict cut.  This changes lifetime, not the Section 14
hard arithmetic closure.  A smaller exact online object requires a literal
fixed query subspace in the deployed D10 word; no such subspace follows from
its current generic Schur/Kraus description.

## Further graph-ready nodes

### PRICE_DISCARDED_COHORT_MIN_SUFFICIENT_STATISTIC

- status: solved_exact;
- statistic: directed future-cut plane
  (A=F^3\operatorname{Diag}(\lambda)F^T);
- decoder: every future action is \(\langle Q,A\rangle\);
- generic stable dimension: (n\min(R,n)), equal to (n^2) for a full
  cohort;
- strict convention: (n(n-1)) off-diagonal coordinates plus separately
  owned K4 diagonal.

### PRICE_GENERIC_FUTURE_QUERY_SKETCH_LOWER_BOUND

- status: closed_subquadratic_generic_sketch;
- witness: at (R=n,F=I), the cohort-to-K31-plane Jacobian has rank (n^2);
- theorem: every continuous/differentiable one-pass exact sketch for all
  future matrices has at least (n^2) real coordinates;
- replay caveat: less memory is possible only by regenerating the discarded
  cohort and paying its arithmetic.

### PRICE_LOW_RANK_QUERY_ADAPTIVE_NOGO

- status: closed_proof;
- rank-one matrix queries contain every (E_{ij}), hence span all matrices;
- consequence: an unknown future rank-(k) promise does not reduce storage;
- escape: query factors known before discard or confined to fixed known
  left/right spaces.

### PRICE_FIXED_SUBSPACE_QUERY_SKETCH

- status: conditional_exact_GO;
- query: (Q=UKV^T) with fixed (U,V);
- statistic: (U^TAV), storage (pq);
- work: equation (22.3);
- gate: the factor spaces must be known at the forward discard cut, not
  inferred later from the realized reverse query.

### PRICE_SHARED_EIGENSPACE_QUERY_SKETCH

- status: conditional_exact_GO;
- fixed basis statistic: \(\operatorname{diag}(S^TAS)\), (n) scalars;
- work: two (n\times n) by (n\times R) projections, about (2R/n)
  equivalents;
- close: basis discovered only after the reverse sweep or varying freely by
  layer/network.

### PRICE_D10_SCHUR_KRAUS_SKETCH_GATE

- status: exact structured formula / generic compression closed;
- formula: (24.4) for fixed reverse factor spaces;
- positive case: (pq) storage, Kraus-weighted projection cost (24.5);
- negative case: full-support Hadamard gates preserve the dimension of an
  arbitrary (H)-space even at Kraus rank one.

### PRICE_ENDPOINT_RANKONE_SKETCH_NOGO

- status: closed_as_generic_O_n_memory_claim;
- arbitrary symmetric rank-one queries span \(\operatorname{Sym}_n\), requiring
  (n(n+1)/2) coordinates;
- nonsymmetric rank-one queries require (n^2);
- exact escape: endpoint direction in a fixed known (k)-space, requiring
  (k(k+1)/2) coordinates.

### PRICE_FUTURE_PLANE_STORE_REPLAY_FRONTIER

- status: solved_exact_tradeoff;
- retain rays: (nR+R) storage, (R/n) later action cost;
- store future plane: (n^2) storage, (R/n) formation cost, cheap later
  query;
- replay: minimal cohort storage but full deterministic regeneration cost;
- implication: memory fusion does not rescue ten full batches from the
  additive arithmetic no-go.

## 27. Exact half-plane sketch for the literal oriented D10 formula

The abstract D10 discussion can be specialized without assuming a low-rank
reverse mask.  At the future cut the literal query is

\[
 Q^{31}(H)
 ={1\over6}\left[
 H\odot G^{31}+(H\odot G^{13})^T
 \right],
 \qquad H=H^T.
\tag{27.1}
\]

The gate matrices (G^{31},G^{13}) are forward-cut objects and can be known
when the cohort reaches that cut; only (H) arrives on the later reverse
sweep.  For its directed plane (A), Hadamard self-adjointness gives

\[
\begin{aligned}
 \langle Q^{31}(H),A\rangle
 &={1\over6}\left[
 \langle H,G^{31}\odot A\rangle
 +\langle H,G^{13}\odot A^T\rangle
 \right]\\
 &=\left\langle H,
 {1\over6}\operatorname{sym}\left(
 G^{31}\odot A+G^{13}\odot A^T
 \right)\right\rangle .
\end{aligned}
\tag{27.2}
\]

Therefore the exact online sufficient statistic for this literal D10 family
is

\[
 \boxed{
 S_{D10}(A)
 ={1\over6}\operatorname{sym}\left(
 G^{31}\odot A+G^{13}\odot A^T
 \right)\in\operatorname{Sym}_n.}
\tag{27.3}
\]

It can be formed immediately after the one-GEMM construction of (A), then
the directed plane and factors may be discarded.  Storage falls from (n^2)
to (n(n+1)/2), or to the strict upper triangle plus the separately typed
diagonal when the K31 collision convention excludes it.  The later reverse
action is the lower-order Frobenius product (langle H,S_{D10}\rangle).

This factor-two saving is generically sharp.  If for every unordered pair
({i,j}) the coefficient map in (27.3) is nonzero, varying the symmetric
matrix (H) over its coordinate basis gives

\[
 \dim\{Q^{31}(H):H=H^T\}={n(n+1)\over2}.
\tag{27.4}
\]

Equivalently, the adjoint map (A\mapsto S_{D10}(A)) is generically onto
(operatorname{Sym}_n) because the full-cohort (A) ranges over an open
matrix set.  Low endpoint rank and low Kraus rank do not reduce (27.4) unless
they confine (H) to a fixed subspace known at the forward cut.

The memory and work for ten full cohorts are consequently

\[
 M=10{n(n+1)\over2}
 \quad\text{stored scalars},
 \qquad
 C_{form}=10
 \quad\text{dense equivalents},
\tag{27.5}
\]

after every required strict factor push.  This solves the causal overlap with
half the generic plane memory, but it does not change the ten-equivalent
arithmetic no-go.

### PRICE_LITERAL_D10_HALFPLANE_SKETCH

- status: solved_exact_and_generically_sharp;
- statistic: equation (27.3), formed from forward-known gate matrices and the
  discarded cohort's future-cut directed plane;
- storage: (n(n+1)/2) per layer, or strict triangular support under the
  collision convention;
- lower bound: generic full-support oriented gates make the D10 query space
  full-dimensional in \(\operatorname{Sym}_n\);
- consequence: exact factor-two memory fusion is available, but no
  subquadratic generic sketch and no arithmetic relief.

## 28. Reverse-free all-layer fusion by a forward tangent accumulator

There is an exact time/memory dual of storing (27.3).  Let the declared D10
linear covariance tangent be

\[
 E_{l+1}=\mathcal T_l(E_l)+I_l,
\tag{28.1}
\]

with the affine conjugation and reset placement absorbed into
(mathcal T_l), and let the ray batch create the already strict-depth typed
injection (I_l).  Starting from (E_0=0), one forward sweep gives

\[
 E_L=\sum_l\mathcal T_{l+1:L}I_l.
\tag{28.2}
\]

For terminal mask (B_L), the reverse matrices satisfy the adjoint recurrence,
so

\[
 \boxed{
 \langle B_L,E_L\rangle
 =\sum_l\langle H_l,I_l\rangle.}
\tag{28.3}
\]

Thus all layer actions can be fused online into one (n\times n) state.  No
dropped cohort, future plane, or D10 matrix must be stored across the
forward/reverse boundary.

The arithmetic trade is exact.  A separate dense covariance word costs two
dense products per crossed layer, approximately (2L) equivalents, in
addition to source-plane formation

\[
 C_{source}=\sum_lR_l/n.
\tag{28.4}
\]

For ten full cohorts this is `20+10=30` equivalents before reset hooks.  If a
literal D14 state already obeys the identical recurrence and accepts the same
injection at the same cut, adding (I_l) to that **same aggregate feature** is
lower-order and the 20 products can be reused.  But then the ray correction is
not a separately refittable response: it has been fused with D14's existing
feature coefficient.

Keeping separate old and new columns for complete-old Schur residualization
requires two distinguishable tangent states, so the second conjugation stream
returns.  Equivalently, a fixed physics coefficient may use the shared state;
an independently scored/refitted feature may not claim that reuse without an
extra linear-output channel which preserves both responses.

Therefore the forward accumulator is a genuine memory escape, but not an
additive arithmetic escape:

- separate candidate: one (n^2) state, about 30 equivalents for ten full
  cohorts;
- fused fixed correction in literal D14: one shared state and about ten new
  source-plane equivalents;
- neither fits the one-equivalent additive slot;
- a lower cost requires the novel strict plane itself to be an already-paid
  D14 injection, which is a buffer/ownership identity not implied by common
  weights.

### PRICE_FORWARD_TANGENT_MEMORY_FUSION

- status: solved_exact_memory_time_duality;
- identity: terminal forward accumulator (28.3) equals the sum of D10 reverse
  actions;
- memory: one (n^2) tangent state instead of ten stored half-planes;
- separate-feature cost: (2L+\sum_lR_l/n\), about 30 equivalents for ten
  full cohorts;
- D14 reuse: conditional only for one fused fixed feature; separate Schur/refit
  control requires a distinguishable state and restores the propagation cost;
- verdict: exact memory escape, current additive arithmetic no-go unchanged.

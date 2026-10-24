# Traffic probability closure

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## Conventions and the traffic refinement

Let `A_i` be real symmetric latent matrices and use the unhalved quadratic
chaos convention

\[
Q_A=z^TAz-\operatorname{tr}A=I_2(A),\qquad z\sim N(0,I).
\]

Every fourth-order formula below is divided by 16 if the implementation uses
`Q_A=I_2(A)/2`.  Write

\[
Y_{ij}=A_i\circ A_j:=\tfrac12(A_iA_j+A_jA_i),\qquad
X_i=\operatorname{svec}(A_i),\qquad K=XX^T,
\]

and let

\[
P=X^TK^+X,\qquad Z_{ij}=(I-P)Y_{ij}.
\]

For an edge-labelled test graph `T`, the ordinary traffic state and its
injective version are

\[
\tau[T]=\sum_{\phi:V(T)\to[r]}\prod_{e=(u,v)}
(A_{\gamma(e)})_{\phi(u),\phi(v)},\qquad
\tau^\circ[T]=\sum_{\phi\ {m injective}}\prod_e
(A_{\gamma(e)})_{\phi(u),\phi(v)},
\]

with the harmless problem-specific normalization suppressed.  Ordinary
noncommutative moments see only directed cycles.  Traffic states also retain
trees, rooted decorations, repeated-edge motifs, and the equality pattern of
the graph vertices.  The exact conversion is

\[
\tau^\circ[T]=\sum_{\pi\in\Pi(V(T))}
\mu(\hat0,\pi)\tau[T/\pi],\qquad
\tau[T]=\sum_{\pi\in\Pi(V(T))}\tau^\circ[T/\pi],
\]

where

\[
\mu(\hat0,\pi)=\prod_{B\in\pi}(-1)^{|B|-1}(|B|-1)!.
\]

Trees are rooted traffic messages; a unicyclic observable is a cyclic trace
whose vertices carry rooted-tree messages.  An all-distinct cumulant is the
injective traffic state obtained by the first formula.  Thus tree and
unicyclic closure is constructive whenever the recurrence is series-parallel:
contract rooted trees toward the unique cycle, then take the remaining cycle
trace.  It is not a generic bounded-state theorem.

### Exact-basis and bounded-motif no-go

The minimal exact depth-`L` traffic state is the quotient-and-gluing closure of
all rooted test graphs born by the first `L` layer maps.  It must be closed
under edge composition, root gluing, Wick contraction, and every graph
quotient used by Mobius inversion.  In the worst case the number of quotient
types already contains a Bell-number family, and independent cycles generated
at successive layers make the motif family grow with depth.  A bound on motif
degree alone therefore does not bound the separator state.

More strongly, no fixed bounded motif family can guarantee `>0.925` of the
value for an arbitrary fixed network.  Injective traffic coordinates of
distinct test graphs can be varied independently on finite matrix models; an
endpoint functional can consequently be supported on the first omitted graph.
For a particular network a retained motif subspace `S` passes exactly when

\[
\frac{\|\Pi_S v\|^2}{\|v\|^2}>0.925,
\]

which requires a network-specific tail or observability theorem.  Degree or
treewidth by itself supplies no such lower bound.  Depth ten is manageable
only under an additional series-parallel/common-separator law or a certified
target-conditioned low-rank pullback.

## Exact C4 separator and NCQC factorization

For four quadratic-chaos slots the connected fourth cumulant is

\[
\kappa_4(A,B,C,D)=16\{\langle Y_{AB},Y_{CD}\rangle+
\langle Y_{AC},Y_{BD}\rangle+\langle Y_{AD},Y_{BC}\rangle\}.
\]

Hence a cut of rooted `C4` exposes precisely the pair-plane map

\[
\mathcal Y:\operatorname{Sym}^2\mathbb R^n\to\operatorname{Sym}_r,
\qquad e_i\odot e_j\mapsto Y_{ij}.
\]

The exact two-terminal separator rank is

\[
r_{C4}=\operatorname{rank}\mathcal Y
=\operatorname{rank}(P\mathcal Y)+\operatorname{rank}((I-P)\mathcal Y).
\]

It is generically ambient-rank rather than bounded motif rank.  A single
target-conditioned endpoint adjoint contracts this separator to one scalar
functional, but ten different causal pullbacks require the rank of their
backward block-Krylov span.  Without a common invariant wedge that rank can
grow to `r_C4`; scalar output alone does not prove a one-word carrier.

For the external `211` specialization,

\[
q_4^{211}(i;j,k)=16[\langle Y_{ii},Y_{jk}\rangle
+2\langle Y_{ij},Y_{ik}\rangle].
\]

With `t_{a;ij}=8<A_a,Y_ij>`, orthogonal projection gives the exact split

\[
q_\parallel^{211}=\tfrac14[t_{ii}^TK^+t_{jk}
+2t_{ij}^TK^+t_{ik}],
\]

\[
q_\perp^{211}=16[\langle Z_{ii},Z_{jk}\rangle
+2\langle Z_{ij},Z_{ik}\rangle].
\]

The first is the column Gram tied to the same singular spectrum as the H3/fish
row Gram and receives no new source credit.  The second is the only independent
NCQC quartic candidate.  Its exact zero gate is

\[
(I-P)\mathcal Y=0,
\]

equivalently, the reachable trajectory span is Jordan closed.  A generic full
`q_perp` provider needs the residual pair-plane map and has `Theta(n^3)`
factor size when `r^2=Theta(n)`; no uniformly smaller exact representation is
available.  A target-conditioned K31 contraction can be streamed without that
materialization because

\[
\sum_b w_b Z_{ab}=(I-P)(A_a\circ A_w),\qquad A_w=\sum_b w_bA_b.
\]

This is the lawful `O(n^3)`-or-better contraction route; it does not create a
generic four-leg carrier.

The repeated endpoint specializations are

\[
K_{31,\perp}(a,b)=48\langle Z_{aa},Z_{ab}\rangle,
\]

\[
K_{22,\perp}(a,b)=16[\langle Z_{aa},Z_{bb}\rangle+2\|Z_{ab}\|^2].
\]

Although the pair-label Gram is PSD, `K31_perp` is signed: it is cubic in the
`a` leg and linear in the `b` leg.  It is therefore not confined to even
variance.  `K22` is the norm-like even specialization.

## Normal order and the three equality partitions

For `Q_A=I_2(A)`, put `U_AB=I_4(A sym_tensor B)`.  The product formula is

\[
Q_AQ_B=U_{AB}+4Q_{A\circ B}+2\langle A,B\rangle.
\]

Consequently

\[
\begin{split}
\kappa_4(A,A,B,C)
={}&\langle U_{AA},U_{BC}\rangle
+32\langle A^2,B\circ C\rangle\\
&-8\langle A,B\rangle\langle A,C\rangle\\
={}&16[2\operatorname{tr}(A^2BC)+\operatorname{tr}(ABAC)]\\
={}&48\operatorname{tr}(A^2BC)+8\operatorname{tr}([A,B][A,C]).
\end{split}
\]

The last commutator contrast can affect K22 but vanishes in K31.  The first
line is the required H4 plus H2-return minus disconnected-pair ledger; taking
only the normal-ordered H4 square is not the connected cumulant.

With the half-scaled convention `Q_M=I_2(M)/2`,

\[
U_M=:Q_M^2:=Q_M^2-\tfrac12\operatorname{tr}M^2-2Q_{M^2}
=\tfrac14I_4(M\widetilde\otimes M).
\]

In a fixed latent coordinate basis its Hermite equality sectors have
coefficients

\[
\begin{array}{c|l}
(4)&\tfrac14M_{ii}^2H_4(z_i)\\
(31)&M_{ii}M_{ij}H_3(z_i)z_j\\
(22)&(\tfrac12M_{ii}M_{jj}+M_{ij}^2)H_2(z_i)H_2(z_j)\\
(211)&(M_{ii}M_{jk}+2M_{ij}M_{ik})H_2(z_i)z_jz_k\\
(1111)&2(M_{ij}M_{kl}+M_{ik}M_{jl}+M_{il}M_{jk})z_iz_jz_kz_l.
\end{array}
\]

These are latent-coordinate labels `lambda_z`.  They are not external-neuron
equality `lambda_ext`, and neither is a traffic-vertex collision
`lambda_graph`.  In particular, injective C4(1111) is only one latent sector
of the square, while external `q4(211)` receives contributions from every
latent sector after dense transport.  Full `U_M`, external `q211`, and
injective C4 are not interchangeable providers.

For a four-vertex C4, internal graph Mobius inversion has the exact signature

\[
[1;\ 6\times(-1);\ 3\times(+1);\ 4\times(+2);\ 1\times(-6)]
\]

on graph partitions `(1111),(211),(22),(31),(4)`.  Equivalently,

\[
\tau_{C4}^\circ=\tau_{C4}-\sum_6\tau_{211}
+\sum_3\tau_{22}+2\sum_4\tau_{31}-6\tau_4.
\]

For a rooted traffic, use the restricted partition lattice that never merges
a protected terminal.  External K31 injectivity is a separate two-block
Mobius operation

\[
(\mathfrak I_{31}F)(a,b)=(1-\delta_{ab})F(a,b).
\]

The removed diagonal is the K4 quotient.  For the perpendicular NCQC piece it
is exactly

\[
K_{4,\perp}(a)=48\|Z_{aa}\|^2,
\]

and must be assigned to the local/K4 owner exactly once.

## Canonical quotient/projection theorem

The unambiguous provider order is

\[
\boxed{
\text{Wick normal order}\to
\lambda_z\text{ split}\to
\text{root-protected graph Mobius}\to
\text{causal transport}\to
\lambda_{ext}\text{ Mobius}\to
q_\parallel/q_\perp\to
\text{old-span Schur}.}
\]

The following swaps are legal, with their hypotheses explicit.

1. Normal order and latent Hermite projection commute because the latter
   refines homogeneous chaos: `L_lambda N4=N4 L_lambda=L_lambda`.
2. Normal order and graph quotient commute only for the Wick-compatible
   quotient `N Q_pi N`.  Raw coordinate substitution before normal order is
   not the same operation.
3. Root-protected graph Mobius and external-label Mobius commute.  They act on
   disjoint partition posets, hence their incidence-algebra operators tensor.
4. A deterministic color-linear transport commutes with normal order and with
   internal rooted graph quotients.  It also commutes with the fixed matrix
   projection `P` on pair planes:
   `Zbar_ab=sum_ij W_ia W_jb Z_ij`.  This fails if transport changes the latent
   basis without carrying `P`, or if `P` is recomputed after transport.
5. Every fixed linear operator above commutes with a full-stream difference.
   The nonlinear diagonal map `A -> q_perp(A,A,A,A)` does not; it becomes
   linear only after lifting to the four-slot tensor.

The following swaps are forbidden.

1. `lambda_z` projection and graph quotient do not commute.  Identifying two
   vertices of a latent `(1111)` C4 creates a `(211)` term.  Project-then-
   quotient retains that counterterm; quotient-then-project-to-1111 kills it.
2. External/root equality projection and dense transport do not commute.  A
   two-source tensor with only off-diagonal entries, transported by a map that
   sends both sources to one endpoint, becomes diagonal.  Source masking keeps
   it while endpoint masking kills it.
3. Recomputing `P_D` and `P_A` in two streams is forbidden.  In ambient
   `R^2`, take `span(D)=span(e1)`, `span(A)=R^2`, and pair-plane `e2`; it is
   wholly perpendicular in the first split and zero in the second.
4. Old-span Schur residualization is last.  For old span `span(e1+e2)`, with
   `q_parallel=e1` and `q_perp=e2`, projecting the total and then attempting to
   split cannot reproduce the separately attributed components.
5. Dense transport and any equality-sector truncation are forbidden to swap
   unless the transport is proved block-monomial on that equality basis.

## Checkpoint 4: canonical late endpoint K31 provider

Let `D` be the baseline stream, `A=D+R` the full shared NCQC stream, and use
the same frozen `P`, transport, quotient ledger, and old-provider span in both
streams.  Let

\[
\bar A_a=\sum_iW_{ia}A_i,\quad
\bar Y_{ab}=\bar A_a\circ\bar A_b,\quad
\bar Z_{ab}=(I-P)\bar Y_{ab}.
\]

Before graph collision refinement, the complete normal-ordered K31 identity is

\[
\begin{split}
K_{31}^{NO}(a,b)
={}&\langle U_{\bar A_a\bar A_a},U_{\bar A_a\bar A_b}\rangle
+32\langle \bar A_a^2,\bar A_a\circ\bar A_b\rangle\\
&-8\|\bar A_a\|^2\langle\bar A_a,\bar A_b\rangle\\
={}&48\operatorname{tr}(\bar A_a^3\bar A_b)\\
={}&\tfrac34t_{aa}^TK^+t_{ab}+48\langle\bar Z_{aa},\bar Z_{ab}\rangle.
\end{split}
\]

This identity is the main duplicate checksum: the first term is H4, the next
two are its return and pair counterterm, the `tK+t` term is the already-owned
H3/fish column Gram, and only the final Gram is the new NCQC source.

Let `G_rp` denote root-protected internal graph Mobius, `L_lambda` the complete
latent-sector ledger, `T_W` causal transport, `Q_perp` the fixed-P residual
pairing, and `S_old` the final Schur projector.  The full collision-subtracted
late candidate is

\[
\boxed{
\Phi_{31,\perp}^{late}
=S_{old}\,Q_\perp\,\mathfrak I_{31}\,T_W\,G_{rp}
\left(\sum_{\lambda_z}L_{\lambda_z}\right)N_{conn}
[A^{\otimes4}-D^{\otimes4}].}
\]

Here `N_conn` means the full H4/H2/H0 connected ledger above, not H4 alone.
Diagrammatically, for every allowed internal graph partition `pi`, this is

\[
(1-\delta_{ab})16\sum_{\pi}\mu(\pi)
\sum_{\sigma\in\mathcal P_2(4)}
\langle Z^{\pi}_{\sigma_1},Z^{\pi}_{\sigma_2}\rangle_{\pi},
\]

with slots `(a,a,a,b)`, followed by full-minus-baseline and `S_old`.  In the
unquotiented sector the three pairings coincide and give
`(1-delta_ab)48<Zbar_aa,Zbar_ab>`.  Quotient terms are retained with the C4
Mobius signature above until an old owner is identified; they are not silently
dropped.

### One-stream causality

At layer `l`, all H3, fish, `q_parallel`, and `q_perp` reads must use the same
pre-injection state `A_l`.  They are staged simultaneously and injected only
after the read.  “Late q_perp” means late in provider ownership, not a read
from a state already modified by the old NCQC consumer.  Reading
`q_perp(A_l+J old(A_l))` creates unledgered nested diagrams and breaks the
fourth-order checksum.  Computing `Z` before or after color transport is legal
under frozen `P`; applying the external K31 mask before transport is not.

### Exact two-stream polarization

Set

\[
Z^A_{ab}=Z^D_{ab}+\delta Z_{ab},\qquad
\delta Z_{ab}=(I-P)(D_a\circ R_b+R_a\circ D_b+R_a\circ R_b).
\]

Then the only correct incremental feature is full-new minus full-baseline:

\[
\frac{K_{31,\perp}(A)-K_{31,\perp}(D)}{48}
=\langle Z^D_{aa},\delta Z_{ab}\rangle
+\langle\delta Z_{aa},Z^D_{ab}\rangle
+\langle\delta Z_{aa},\delta Z_{ab}\rangle.
\]

An `R`-only square retains only the last residual-quartic corner and loses all
mixed terms.  If

\[
L_{ab}=(I-P)(D_a\circ R_b+R_a\circ D_b),\qquad
H_{ab}=(I-P)(R_a\circ R_b),
\]

the formal-parameter audit

\[
Z_{ab}(s)=Z^D_{ab}+sL_{ab}+s^2H_{ab}
\]

requires the four nonconstant coefficients

\[
\begin{array}{c|l}
s^1&\langle L_{aa},Z^D_{ab}\rangle+\langle Z^D_{aa},L_{ab}\rangle\\
s^2&\langle H_{aa},Z^D_{ab}\rangle+\langle L_{aa},L_{ab}\rangle+\langle Z^D_{aa},H_{ab}\rangle\\
s^3&\langle H_{aa},L_{ab}\rangle+\langle L_{aa},H_{ab}\rangle\\
s^4&\langle H_{aa},H_{ab}\rangle.
\end{array}
\]

Every fixed linear stage in the boxed pipeline can be applied to these
coefficients termwise.  No nonlinear feature formation and no stream-specific
projection may be moved through the subtraction.

## Universal direct-scalar K31 audit

For arbitrary scalar variables, first center
`u=U-EU`, `v=V-EV`.  The universal K31 action is

\[
\boxed{\kappa_4(U,U,U,V)=E[u^3v]-3E[u^2]E[uv].}
\]

The factor three is the complete disconnected-pair counterterm: there are
three choices of the `U` paired with `V`.  For the NCQC variables above,

\[
E[Q_A^3Q_B]-3E[Q_A^2]E[Q_AQ_B]
=48\operatorname{tr}(A^3B)
=\tfrac34t_{aa}^TK^+t_{ab}+48\langle Z_{aa},Z_{ab}\rangle.
\]

Therefore direct scalar evaluation on the full pre-injection endpoint owns
every source external-`211`, external-`1111`, latent Hermite sector, and C4
graph-collision contribution exactly once.  Expanding

\[
U_a=\sum_iw_{ia}Q_i,
\]

shows that all ordered source tuples occur with their natural multiplicity.
Ordinary traffic already sums every vertex-equality class once; Mobius is an
alternative decomposition into injective classes, not an additional source.
Thus one must choose exactly one of these two ledgers:

1. evaluate the total scalar cumulant directly; or
2. sum the disjoint injective quotient pieces over all graph partitions.

Adding injective C4, external q211, or latent q1111 on top of the direct scalar
cumulant double counts.  Conversely, applying C4 Mobius to the direct scalar
and retaining only its injective term loses the quotient pieces and is no
longer universal K31.

Direct scalar evaluation is a valid `q_perp` provider only after the exact
subtractions

\[
\mathfrak I_{31}\left[
\kappa_4^{scalar}-\tfrac34t_{aa}^TK^+t_{ab}
-K_{31}^{\text{other old owners}}
\right],
\]

followed by `S_old`.  The diagonal removed by `I31` belongs to K4 exactly
once.  If `U,V` contain sources beyond the shared NCQC core, four-slot
provider polarization is required before this subtraction; a scalar total
cannot by itself identify which core supplied the term.

## Minimal exact symbolic audit and gates

No numerical run is needed.  The following identities are jointly sufficient
to catch a lost or duplicated term.

1. **Wick checksum:** the three-term normal-order expression equals
   `48 tr(A_a^3 A_b)` symbolically.
2. **Latent completeness:** `sum_lambda L_lambda=N4` and
   `L_lambda L_mu=delta_lambda_mu L_lambda`.
3. **Graph checksum:** every internal quotient has exactly its Mobius
   coefficient `[1,-1,+1,+2,-6]` with multiplicities `[1,6,3,4,1]`.
   When auditing the direct ordinary scalar traffic instead, the 15 kernel
   partitions each have coefficient `+1`; mixing these two coefficient vectors
   is itself a duplicate/loss error.
4. **External checksum:** `diag(I31 K)=0`; the removed diagonal equals the
   separately owned K4 quotient once.
5. **NCQC split:** `P^2=P=P*`, `<PY,(I-P)Y'>=0`, and
   `K31_NO=(3/4)t_aa^T K+ t_ab+K31_perp`.
6. **Transport naturality:** with frozen `P`,
   `Zbar_ab=sum_ij W_ia W_jb Z_ij`; endpoint equality is tested only after
   this identity.
7. **Four-slot stream checksum:** replace the three `a` slots and one `b`
   slot by `D+s_1R,D+s_2R,D+s_3R,D+tR`.  The perpendicular fourth cumulant is
   multiaffine and must contain each of the 15 nonempty square-free monomials
   in `{s_1,s_2,s_3,t}` exactly once.  This is generically minimal: the 15
   polarized cross terms are independent components of a four-linear map.
8. **Provider checksum:** the symbolic tag
   `(chaos grade,lambda_z,lambda_graph,lambda_ext,stream subset)` has exactly
   one owner.  H3/fish owns `q_parallel`; the late word owns only `q_perp`.
9. **Final duplicate gate:** with old feature matrix `E`, the candidate is
   novel iff
   `rank([E,Phi])>rank(E)`, equivalently its exact Schur complement
   `g_qq-g_qE G_EE^+g_Eq` is nonzero (with the usual range condition).
10. **Final zero gate:** if every reachable residual pair plane lies in
    `span{A_i}`, then `q_perp` is identically zero and the lane closes before
    deployment analysis.

## Current next question

Derive the actual endpoint adjoint defining the target-conditioned K31 slice
and prove whether its ten late pullbacks have a bounded common Krylov/Jordan
module.  If so, the boxed provider can be contracted as one late residual word
without materializing the generic pair-plane map.  If not, the exact separator
rank theorem is a deployability no-go even though the algebraic q-perp source
is genuine.  In parallel, determine from the existing provider definitions
whether the direct scalar K31 action is already present; if it is, the only
lawful change is its frozen-P/old-span residual, not a new q211/C4 addend.

## Required graph-node split

These results must not be represented by one omnibus graph node.  The clean
independent nodes and edges are:

1. `TRAFFIC_K31_NORMAL_ORDER_ID` — **proved**.  H4 plus H2 return minus the
   three disconnected pairings equals `48 tr(A_a^3 A_b)`.
2. `TRAFFIC_K31_COLLISION_OWNERSHIP` — **theorem proved; repository ownership
   audit open**.  Ordinary scalar traffic owns all 15 graph-kernel partitions
   with `+1`; injective C4 instead uses the Mobius vector and cannot be added
   to the scalar total.
3. `TRAFFIC_K31_QSPLIT_FIXED_P` — **proved**.  `q_parallel=(3/4)tK+t` is tied
   to H3/fish; only fixed-P `q_perp=48<Zaa,Zab>` can be new.  It is zero under
   Jordan closure.
4. `TRAFFIC_K31_SCALAR_CONTRACTION` — **algebra proved; presence/deployability
   open**.  Centered direct scalar evaluation uses exactly
   `E[u^3v]-3E[u^2]E[uv]`, then endpoint root masking.
5. `TRAFFIC_K31_DUPLICATE_COST_GATE` — **duplicate gate proved; carrier cost
   open**.  Apply full two-stream polarization and final old-span Schur; pass
   only if the Schur complement is nonzero and the target-conditioned Krylov
   module fits the deployment budget.

Use the dependency chain `1 -> 2 -> 3 -> 4 -> 5`, plus `2 -> 4` for the
total-versus-injective exclusivity and `3 -> 5` for the exact zero/Schur gate.

## Checkpoint 5: actual CP3/Q24/DTOT2 duplicate ledger

### What the existing state owns

The deployed Q24 lineage does not store an abstract `q_parallel` and a separate
`q_perp`.  Its exact carried fourth-order object is the connected off-diagonal
K31 plane.  At a reset, `_connected_planes` forms the centered raw moment and
subtracts the disconnected pairing exactly once; schematically,

\[
K_{31}^{target}(i,j)
=M_{31}(i,j)-\text{mean counterterms}
-3\,v_iC_{ij},\qquad i\ne j.
\]

It then zeros the diagonal and stores that diagonal in the separate K4 branch.
The target plane is decomposed, without remainder, into

\[
\boxed{
K_{31}^{old}
=K_{31}^{CP4}+K_{31}^{star}
+3\alpha\,\operatorname{offdiag}(v[:,None]C).} \tag{T5.1}
\]

`_quartic_residual` subtracts the CP4 plane once; `_reset_stars` subtracts the
connected covariance-pairing ansatz once; the residual is the stored quartic
coordinate star.  `_assemble_quartic_provider` reverses these two subtractions
at the next pre-state.  Therefore (T5.1) is an exact ownership identity for the
closure's own K31 plane.

The `3 alpha v C` term in (T5.1) is **not** the disconnected cumulant
counterterm.  The latter has already been removed by `_connected_planes`.
`3 alpha v C` is a fitted connected old provider and must be included in the
old-action subtraction.  Treating it as another Gaussian disconnection would
subtract it twice.

CP3 feeds the reset moments through the carried K21/H3 state, and its effect is
already folded into `K31_target`.  It must not receive a second subtraction
after the full old K31 plane is contracted.  Likewise, CP4, star, and pairing
are mutually exclusive storage pieces of (T5.1), not four separately creditable
features.

DTOT2 is a consumer, not another K31 source.  In the tensor instrument the
closure-carried K31 plane enters the K21 emission as

\[
K_{31}\odot E_{31}+K_{31}^T\odot E_{13},
\]

where `E31,E13` are the local Mehler/Edgeworth response matrices, including the
connected M21-minus-`2 r1 M11` combination.  If `B_l` is the reverse adjoint on
the emitted K21 plane, the exact local K31 costate is

\[
\boxed{
\Lambda_l=B_l\odot E_{31,l}+(B_l\odot E_{13,l})^T
+\Lambda_l^{other\ K31\ consumers}.} \tag{T5.2}
\]

Changing `DTOT` changes the distinct-cubic bond transport, not ownership of
the supplied K31 plane.  Thus DTOT2 contributes to `Lambda_l`; it is never
added as a second quartic provider.

### Local action-level subtraction theorem

Let the full candidate's centered local scalar contexts be `(U_l,V_l)` and let
`Lambda_l` be the same-site, same-time adjoint in (T5.2).  Define

\[
\chi_l^{full}=\sum_a\{
E[U_{la}^3V_{la}]-3E[U_{la}^2]E[U_{la}V_{la}]\},
\]

\[
\chi_l^{old}=\langle\Lambda_l,K_{31,l}^{old}\rangle_F.
\]

Then

\[
\boxed{\Delta\chi_l=\chi_l^{full}-\chi_l^{old}} \tag{T5.3}
\]

is the complete action-level novelty subtraction.  It is local because both
terms live at the K31 birth/consumption site; no fourth tensor is transported.
The old contraction can use either the already assembled plane or the three
pieces in (T5.1), and after `Lambda_l` is known costs only Frobenius/elementwise
contractions.

No trajectory projection `P` appears in (T5.3).  The `q_parallel/q_perp` split
is a useful attribution of the total K31 action but is not part of the actual
CP3/Q24 storage basis.  Therefore action-level subtraction is invariant under
the choice of NCQC span and cannot suffer the stream-specific-`P` failure.  A
later q-split may be reported diagnostically, but it cannot change (T5.3) or
create another source credit.

The theorem has one exact synchronization premise:

\[
\boxed{
\chi_l^{base}:=\kappa_4(U_l^D,U_l^D,U_l^D,V_l^D)
=\langle\Lambda_l,K_{31,l}^{old}\rangle_F.} \tag{T5.4}
\]

The variables must be centered with the same moments, evaluated before the
same reset/injection, use the same K31 normalization, and have the same
off-diagonal convention.  If (T5.4) has not been proved, the exact ledger is

\[
\Delta\chi_l=
[\chi_l(D+R)-\chi_l(D)]
+[\chi_l(D)-\chi_l^{old}]. \tag{T5.5}
\]

The first bracket is the genuine mixed-stream innovation; the second is a
zeroth-degree closure/action mismatch.  It may be valuable, but it must be
tagged separately rather than mislabeled `q_perp`.

### Proof that no mixed monomial is lost

Polarize the four K31 slots independently:

\[
\mathcal K(s_1,s_2,s_3,t)=
\kappa_4(D_a+s_1R_a,D_a+s_2R_a,D_a+s_3R_a,D_b+tR_b).
\]

Because the cumulant is four-linear,

\[
\mathcal K-\mathcal K(0,0,0,0)
=\sum_{\varnothing\ne S\subseteq\{1,2,3,4\}}
\left(\prod_{j\in S}\xi_j\right)
\kappa_4(X_1^S,X_2^S,X_3^S,X_4^S), \tag{T5.6}
\]

where `xi=(s1,s2,s3,t)` and slot `j` uses `R` iff `j in S`.  The 15 nonempty
subsets occur exactly once.  Direct full-minus-baseline scalar evaluation is
precisely (T5.6), so it preserves every mixed term without storing or naming
them.  Subtracting the already-connected old plane changes none of these 15
coefficients.  If (T5.4) fails, it adds only the separately visible constant
defect in (T5.5).

This also proves why the following alternatives are wrong:

- evaluating only `kappa(R,R,R,R)` keeps one of the 15 terms;
- adding a separately formed q211, q1111, C4, CP3 fish, or Q24 pairing term to
  (T5.3) duplicates a subset already in the scalar action;
- subtracting `3 E[U^2]E[UV]` after contracting connected `K31_old` removes the
  disconnection twice;
- subtracting CP4, star, and pairing after already subtracting their assembled
  plane removes the old action twice;
- treating DTOT2 as a source duplicates a consumer path already represented in
  `Lambda_l`.

### Complete five-node status for the actual lineage

1. `TRAFFIC_K31_NORMAL_ORDER_ID` — **proved and matched to Q24 form**.
   `_connected_planes` implements centering plus the single `-3 v C`
   disconnection, then zeros the K31 diagonal.
2. `TRAFFIC_K31_COLLISION_OWNERSHIP` — **proved for external roots; internal
   naming is ledger-only**.  K31 diagonal is owned by K4; the ordinary scalar
   action owns all internal graph kernels, so no separate C4 addend is lawful.
3. `TRAFFIC_K31_QSPLIT_FIXED_P` — **not required for subtraction**.  The split
   theorem remains true, but the actual old state owns the assembled connected
   plane.  Stream-specific `P` is eliminated by (T5.3).
4. `TRAFFIC_K31_SCALAR_CONTRACTION` — **local theorem proved, synchronization
   receipt (T5.4) open**.  The action is acquired from three scalar moments or
   the exact affine-quadratic Jordan formula and is contracted at the local
   birth site.
5. `TRAFFIC_K31_DUPLICATE_COST_GATE` — **old subtraction cost closed; new
   action acquisition cost open**.  The old plane contraction is elementwise.
   Only acquisition of the full new scalar action and its exact costate remains
   a deployment question.

### Minimal open cost node

The unique minimal cost node is
`K31_LOCAL_ACTION_ACQUISITION_COST`, not a q4 carrier node.  Its terminal
question is whether the same-site engine exposes

\[
\sum_a E[U_a^3V_a],\qquad
\sum_a E[U_a^2],\qquad
\sum_a E[U_aV_a]
\]

and `Lambda_l` from the existing Site-A/Site-B work.

Under an exact affine-quadratic shared Gaussian context,

\[
\chi=48\operatorname{tr}(N^3R)
+24\{h^TNRh+h^TN^2r\},
\]

or equivalently it needs only `N^2`, `N circ R`, and rank-one H1 pairings.
If Q24's banked square is exactly the same causal `N^2`, the sole new dense
operation is one mixed product `NR` per Bellman context per active layer:
`10m` products on a ten-transition suffix.  If that square is not the same
object, one additional `N^2` product is needed, giving `20m`.  If neither the
direct moments nor the affine-quadratic closure is exact, generic acquisition
can recover the four-leg cost obstruction and the lane has no proved cheap
implementation.

Therefore the next proof is only: establish the same-site receipt (T5.4), show
that the deployed Q24 square equals the primal `N^2`, and determine the number
`m` and reuse status of the actual Bellman contexts.  No `P`, q211 bank, C4
carrier, or fourth tensor belongs in this cost node.

## Checkpoint 6: acquisition-cost receipts against the actual storage

### Receipt A — post-reset same-site synchronization is exact

Let `X_i,X_j` denote the formal Q24 pair law at one reset, with raw moments
`raw2_i,raw3_i,m11_ij,m21_ij,m31_ij`, means `mu_i,mu_j`, variance `v_i`, and
covariance `C_ij`.  Direct expansion gives

\[
\begin{split}
E[(X_i-\mu_i)^3(X_j-\mu_j)]
={}&m31_{ij}-\mu_j raw3_i-3\mu_i m21_{ij}
+3\mu_i\mu_j raw2_i\\
&+3\mu_i^2m11_{ij}-3\mu_i^3\mu_j.
\end{split} \tag{T6.1}
\]

Therefore

\[
\kappa_4(X_i,X_i,X_i,X_j)
=\text{right side of (T6.1)}-3v_iC_{ij}. \tag{T6.2}
\]

This is term-for-term the deployed `_connected_planes` K31 expression.  The
routine then zeros `i=j`, while `_connected_planes` stores the corresponding
four-equal connected cumulant in K4.  Thus the post-reset same-site receipt is
closed exactly, including centering, normalization, the single disconnected
counterterm, and the external collision convention.

For any local adjoint plane `Lambda`, set

\[
U_i=X_i-\mu_i,qquad V_i=\sum_j\Lambda_{ij}(X_j-\mu_j).
\]

Cumulant multilinearity and (T6.2) prove

\[
\boxed{
\sum_i\kappa_4(U_i,U_i,U_i,V_i)
=\langle\Lambda,K_{31}^{Q24,target}\rangle_F.} \tag{T6.3}
\]

No latent realization, q-split, or positivity assumption is needed.

### Receipt B — pre-consumption synchronization has two meanings

The pre-layer plane reconstructed by (T5.1) is, by construction, the formal
K31 cumulant supplied to Q24/DTOT2.  Hence contraction with the exact local
costate is algebraically synchronized:

\[
\chi_l^{formal}=\langle\Lambda_l,K_{31,l}^{assembled}\rangle_F. \tag{T6.4}
\]

What is **not** proved is that this assembled closure plane equals the K31
cumulant of the exact network activation law.  Calling an independently
observed exact moment `K31_true`, the only possible novelty is

\[
\rho_l^{31}=\langle\Lambda_l,
K31_l^{true}-K31_l^{assembled}\rangle_F. \tag{T6.5}
\]

Thus formal same-site synchronization is closed; truth-to-closure
synchronization is precisely the remaining acquisition problem, not a hidden
normalization uncertainty.

### Receipt C — the deployed Q24 state does not contain the AQ `N^2`

The objects have incompatible types:

- the quartic star is one external `n by n` residual plane containing K31 off
  diagonal and K4 on its diagonal;
- the covariance pairing is another external plane `3 alpha v[:,None] C`;
- CP4 stores a signed symmetric decomposition
  `sum_r lambda_r u_r tensor_power 4` and reads external K31/K22 planes from
  powers of the coordinate factors;
- CP3 stores a signed `sym(p,p,q)` decomposition of a cubic tensor.

By contrast, an affine-quadratic realization requires an oriented latent
matrix `N_i in Sym_r` for every coordinate and the bank
`{N_i^2}_{i=1}^n`.  None of the deployed arrays has that type.  In particular,
`u_r^2` in the CP4 read is a coordinatewise power of an external CP factor,
not matrix multiplication `N_i^2`.

There are two independent symbolic obstructions.

1. A real Gaussian quadratic chaos always has
   `kappa4(Q_N)=48 tr(N^4)>=0`.  CP4 birth weights are the signed residual
   `k4_target-inherited_d` and are not constrained nonnegative.  A negative
   rank-one CP4 atom therefore has no representation as the fourth cumulant of
   one real AQ matrix.
2. Even invariant moment equality cannot recover the oriented square.  In
   dimension two let

   \[
   N=\begin{bmatrix}1&0\\0&0\end{bmatrix},\qquad
   N'=ONO^T,\qquad
   R=\begin{bmatrix}1&0\\0&-1\end{bmatrix},
   \]

   with `O` a 45-degree rotation.  All traces `tr(N^k)=tr((N')^k)` agree, so
   every one-coordinate covariance/CP3/Q24 invariant agrees, but

   \[
   \operatorname{tr}(N^3R)=1,\qquad
   \operatorname{tr}((N')^3R)=0. \tag{T6.6}
   \]

   The missing datum is orientation relative to the dual matrix.

Consequently the premise “reuse Q24's banked square as the same causal
`N^2`” is false for the actual lineage.  `Q24_N2_REUSE` is closed as a no-go.
The CP4 factor powers may still evaluate their own already-owned action, but
that is a duplicate of (T5.1), not an AQ realization of a novel action.

### Minimum context count

For an arbitrary scalar-endpoint K31 adjoint `Lambda`, equation (T6.3) uses the
single batched context family

\[
\{(U_i,V_i):V_i=\sum_j\Lambda_{ij}U_j\}_{i=1}^n. \tag{T6.7}
\]

All K31 consumers can be summed into `Lambda` before (T6.7).  Thus the minimum
number of Bellman context **families** is

\[
\boxed{m_{min}=1} \tag{T6.8}
\]

for a nonzero scalar K31 action; `m=0` only when the adjoint or desired
innovation is identically zero.  The `n` row contexts are batched coordinates,
not `n` independent matrix words, and `rank(Lambda)` does not increase this
count.  Multiple independently scored endpoints would each require a family
unless their adjoints are combined before scoring.

### Exact reuse ledger

The actual state reuses the following items lawfully:

- `m20` and `m11`, hence the disconnected scalar subtraction, from the carried
  variance/covariance;
- the formal Q24 `m31/K31` action, already assembled and therefore entirely
  duplicate;
- the off-diagonal root mask and separate K4 diagonal receipt;
- CP4/star/pairing contributions through their single assembled old plane;
- DTOT2's K31 response matrices when forming the local adjoint (T5.2), if its
  reverse K21 costate is available.

It does **not** reuse an oriented AQ `N`, `N^2`, `N circ R`, or an exact
beyond-closure `m31`.  Constructing the AQ formula from the current summaries
is therefore not a one-product marginal extension; it first requires a new
oriented state whose existence/observability is unproved.

Because `m20,m11` and the entire formal connected K31 action are already
owned, the information-minimal new scalar is only

\[
\boxed{
\rho_l^{31}
=\sum_i\left(E_{true}[U_i^3V_i]
-E_{Q24}[U_i^3V_i]\right)
\quad\text{with matched lower moments.}} \tag{T6.9}
\]

or the full matched version (T6.5) if lower moments also differ.  Any new
calculation that uses only the existing Q24 Edgeworth moments returns zero
after the duplicate subtraction.

### Split node statuses and the remaining cost gate

1. `K31_POST_RESET_SYNC_RECEIPT` — **proved/closed** by (T6.1)--(T6.3).
2. `K31_PRESTATE_FORMAL_SYNC_RECEIPT` — **proved/closed** by assembled-plane
   ownership (T6.4).
3. `K31_TRUTH_CLOSURE_SYNC_GATE` — **open and exactly equal to (T6.5)**; this
   is the possible source of value.
4. `Q24_N2_TYPE_IDENTITY` — **refuted/closed NO** by the signed-CP4 and
   orientation counterexamples.
5. `K31_CONTEXT_COUNT` — **proved `m_min=1`** for one scalar endpoint.
6. `K31_OLD_ACTION_LOCAL_REUSE` — **proved/closed**; it is an elementwise
   contraction after `Lambda` is available.
7. `K31_DTOT2_ADJOINT_REUSE` — **formula proved, billing reuse open**.  Decide
   whether the reverse K21 costate needed in (T5.2) is already present at the
   exact consumption site; otherwise price one reverse-plane chain.
8. `K31_TRUE_M31_ACTION_ACQUISITION` — **open and decisive**.  Acquire (T6.9)
   directly in the single context family, or prove a target-specific
   factorization.  A generic exact joint law can restore the four-leg cost
   barrier, so no cheap bound follows from scalar output alone.

The minimal open cost node is therefore
`K31_TRUE_M31_ACTION_ACQUISITION`, with `K31_DTOT2_ADJOINT_REUSE` as its sole
upstream cost premise.  The earlier `10m` versus `20m` AQ estimate is not
applicable to the deployed state because `Q24_N2_TYPE_IDENTITY` failed.  The
next derivation must target the adjoint-weighted true-minus-Q24 raw fourth
moment directly; rebuilding a latent q4 tensor or a fictitious Q24 `N^2` is
both unnecessary and unjustified.

## Checkpoint 7: the residual true-M31 action node

### Site A/B information audit

At Site A, immediately after `_full_relu_reset`, the card has the Q24-produced
mean/covariance and transient connected `k4_target,K31_target,K22_target`, plus
the CP3/CP4/star state used to re-encode those planes.  Its `m31` is the output
of the Q24 pair-law/Edgeworth object.  It is not an independently observed
moment of the exact network activation law.

At Site B, after covariance conjugation and before the reset, the card has
`pre_mean,pre_covariance`, the assembled closure planes
`pre_K21,pre_k4,pre_K31,pre_K22`, and their factor storage.  Again there is no
joint-law object on which arbitrary signal mixtures can be evaluated.

The DTOT2 instrument truth-feeds mean/variance/covariance, but normally consumes
the closure-carried K31/K22 planes.  Its optional oracle-fourth branch reads
offline banked truth planes; that is an experimental oracle, not a deployed
Site A/B observable.  Therefore every ordinary Site A/B calculation is
measurable with respect to the old-state sigma algebra `S_old` and can at most
reproduce the formal Q24 action already removed in (T6.5).

### What a lawful diagonal-q4 JVP would provide

For the **actual-law functional**

\[
\Phi_{true}(U)=\sum_i\kappa_4(U_i,U_i,U_i,U_i),
\]

cumulant multilinearity gives

\[
\boxed{
D\Phi_{true}(U)[V]
=4\sum_i\kappa_4(U_i,U_i,U_i,V_i).} \tag{T7.1}
\]

With `V=Lambda U`, one quarter of (T7.1) is the complete scalar K31
contraction.  Thus a native JVP of the actual-law diagonal fourth potential is
sufficient and information-minimal for a fixed adjoint.

Three superficially similar objects are not this JVP:

1. the stored vector `k4_i=Phi_i(U_i)` contains values only, not cross-unit
   directional information;
2. automatic differentiation of the Q24 code computes the derivative of the
   closure map `Phi_Q24(state)`, and hence only the old action;
3. differentiating `k4` with respect to covariance/state parameters is not the
   signal-space mixing derivative `U -> U+epsilon Lambda U` unless a complete
   fourth-law intertwining theorem is supplied.

The CP4 factor potential does have an evaluable JVP, but it returns only the
already-owned CP4 summand in (T5.1).  The star/pairing pieces behave the same
way.  Summing their JVPs reconstructs Q24 K31 and is exactly zero after the old
action subtraction.

If an actual-law potential oracle could be evaluated on arbitrary mixtures,
an exact no-gradient polarization is also possible.  Set

\[
d_1=\Phi(U+V)-\Phi(U-V),\qquad
d_2=\Phi(U+2V)-\Phi(U-2V).
\]

Then

\[
\boxed{
\sum_i\kappa_4(U_i,U_i,U_i,V_i)=\frac{8d_1-d_2}{48}.} \tag{T7.2}
\]

This needs four callable actual-law potential evaluations.  Existing Site A/B
stores only the value at the coordinate basis and cannot supply any of the
four arms.

### Explicit information-theoretic counterexample

The failure is stronger than a missing implementation.  Let `U` be symmetric
uniform on `{-2,-1,1,2}` and put

\[
c=\frac{E[U^4]}{E[U^2]}=\frac{17}{5},\qquad
h(U)=U^3-cU.
\]

Then

\[
E[Uh(U)]=0,qquad E[U^2h(U)]=0,qquad
E[U^3h(U)]=\frac{18}{5}\ne0. \tag{T7.3}
\]

Let `epsilon` be an independent symmetric variable and define two joint laws

\[
V_+=\epsilon+\theta h(U),\qquad
V_-=\epsilon-\theta h(U). \tag{T7.4}
\]

The laws have identical marginals for both coordinates, identical means,
variances and covariance, identical marginal k3/k4, identical cubic cross
moments, and identical K22/even cross data.  In particular,

\[
E[UV_\pm]=E[U^2V_\pm]=E[UV_\pm^2]=0,
\]

while every even expression in the sign of `theta` agrees.  Hence the CP3,
diagonal-Q24, covariance, and even pair-law inputs coincide.  Nevertheless,

\[
\boxed{E[U^3V_\pm]=\pm\frac{18}{5}\theta.} \tag{T7.5}
\]

Thus their true K31 actions differ although every existing diagonal/lower
Site A/B law object is identical.  The Q24 closure produces the same formal
action for both, so the residual (T6.5) also differs.  No deterministic
function, refit, code JVP, or local contraction of the existing state can
recover the true residual universally.

More abstractly, the old state consists of finitely many moment functionals.
The target function `sum_i U_i^3(Lambda U)_i` is outside their span whenever
true K31 is omitted.  A small signed-measure perturbation orthogonal to all old
functionals but not to this target gives the same conclusion for any proposed
finite rearrangement of the existing summaries.

### Minimal new scalar interface

For a fixed network and reverse adjoint, the necessary and sufficient new
interface at layer `l` is one scalar call

\[
\boxed{
\operatorname{K31ActionResidual}(l,\Lambda_l)
:=\frac14D\Phi_{true,l}(U_l)[\Lambda_lU_l]
-\langle\Lambda_l,K31_l^{Q24}\rangle_F.} \tag{T7.6}
\]

Equivalently, with matched carried lower moments it may return only

\[
\sum_i\left(
E_{true}[U_{li}^3(\Lambda_lU_l)_i]
-E_{Q24}[U_{li}^3(\Lambda_lU_l)_i]
\right). \tag{T7.7}
\]

If lower moments are not matched, the interface must use the complete centered
cumulant form, including the single `-3 m20 m11` difference.  It must also
declare the pre/post-reset site and off-diagonal/K4 convention.  It returns one
scalar per active layer, or may accumulate their Bellman-weighted sum into one
suffix scalar.  It does not expose a full K31 plane, q211 tensor, traffic basis,
or latent core.

This interface is minimal: one real number is sufficient for the fixed scalar
action, and counterexample (T7.3)--(T7.5) proves that zero new numbers cannot be
sufficient.  A different independently scored endpoint requires its own
adjoint call unless the interface is promoted to a reusable linear functional
of `Lambda`.

### Information/cost conclusion

Existing Site A/B objects cannot produce (T7.6).  Once an actual-law JVP is
provided natively, the incremental state is one scalar accumulator and the
old subtraction is elementwise.  Without such an interface, exact evaluation
for a generic nonlinear network law can require resolving the same unrestricted
connected four-leg diagrams that the scalar contraction was meant to avoid;
scalar output dimension alone gives no `O(n^3)` arithmetic theorem.

Four-potential polarization (T7.2) is exact but costs four actual-law oracle
calls.  A Q24-only potential or JVP is free but exactly duplicate.  Monte Carlo
or offline oracle planes are new data-acquisition mechanisms, not lawful local
reuse.  Therefore this lane has an information no-go under the current card
and an open cost only under the new scalar interface.

### Split graph nodes

1. `K31_SITE_A_LAW_OBJECT_AUDIT` — **closed NO**: only the Q24 moment law and
   its connected target planes are present.
2. `K31_SITE_B_LAW_OBJECT_AUDIT` — **closed NO**: only carried formal planes,
   covariance, and factor state are present.
3. `K31_DIAG_Q4_STORED_VALUE_NOGO` — **proved** by (T7.3)--(T7.5); diagonal
   values do not identify a cross direction.
4. `K31_TRUE_Q4_JVP_SUFFICIENCY` — **identity proved, interface absent** by
   (T7.1).
5. `K31_TRUE_Q4_POLARIZATION` — **identity proved, four-call deployment open**
   by (T7.2).
6. `K31_TRUE_M31_INFORMATION_NOGO` — **proved for the current state** by the
   explicit same-state/different-action pair.
7. `K31_ACTION_RESIDUAL_SCALAR_INTERFACE` — **specified, not implemented** by
   (T7.6)--(T7.7).
8. `K31_TRUE_M31_ACTION_ACQUISITION` — **blocked on node 7 or an equivalent
   target-specific factorization**.  No AQ/Q24 reuse path remains.

Edges are `SITE_A_AUDIT,SITE_B_AUDIT -> STORED_VALUE_NOGO ->
INFORMATION_NOGO`; `TRUE_Q4_JVP_SUFFICIENCY` and `TRUE_Q4_POLARIZATION` are the
two positive parents of `ACTION_RESIDUAL_SCALAR_INTERFACE`, which alone can
reopen `TRUE_M31_ACTION_ACQUISITION`.

## Checkpoint 8: leading true-minus-assembled D3f star

### Exact normal-order identity

For a square-integrable function of a standard Gaussian vector, write its
Hermite expansion as

\[
f=\cdots+I_3(C_f)+\cdots,
\qquad C_f=\frac1{3!}E[D^3f]. \tag{T8.1}
\]

Let the H1 coefficients of the repeated K31 direction and its Bellman dual be
`u` and `v`, and let `C_U,C_V` be the projected H3 coefficients.  Wiener-chaos
orthogonality gives

\[
E[I_3(C)I_1(a)I_1(b)I_1(c)]
=3!\langle C,\operatorname{sym}(a\otimes b\otimes c)\rangle. \tag{T8.2}
\]

There is no disconnected cumulant subtraction in (T8.2): `I3` is orthogonal
to every H1 factor, so all pair partitions containing it vanish.  Choosing the
H3 term in each of the four K31 slots yields

\[
\boxed{
\Delta\kappa_{H3\times H1^3}
=18\langle C_U,u\otimes u\otimes v\rangle
+6\langle C_V,u^{\otimes3}\rangle.} \tag{T8.3}
\]

With derivative tensors `T_U=6C_U,T_V=6C_V`, this is

\[
\boxed{
\Delta\kappa_{D3}
=3\langle T_U,u\otimes u\otimes v\rangle
+\langle T_V,u^{\otimes3}\rangle.} \tag{T8.4}
\]

Equation (T8.4) is already Wick normal ordered.  If raw Gaussian cubics are
used instead, the necessary identities are

\[
:L_u^2L_v:=L_u^2L_v-\|u\|^2L_v-2\langle u,v\rangle L_u,
\]

\[
:L_u^3:=L_u^3-3\|u\|^2L_u. \tag{T8.5}
\]

Those H1 return terms belong to the mean/covariance/CP3 lower interface and
must not be added to (T8.4).

For a ReLU composite `f=(g)_+`, the distributional chain rule is

\[
\boxed{
D^3f=\delta'(g)(Dg)^{\otimes3}
+\delta(g)\sum_{3\ cyclic}D^2g\otimes Dg
+1_{\{g>0\}}D^3g.} \tag{T8.6}
\]

For affine-quadratic `g`, the last term vanishes and (T8.6) reduces to the
previously derived surface formula involving `u+Az` and `A`.  Formula (T8.6),
not a raw third moment, is the exact local source object.

### Equality/collision ownership

Normal order does not impose distinct coordinate labels.  For a symmetric
trilinear tensor `T`, its injective/all-distinct contraction is

\[
\begin{split}
T^\circ(a,b,c)
={}&T(a,b,c)
-\sum_{i,k}T_{iik}a_ib_ic_k
-\sum_{i,j}T_{iji}a_ib_jc_i\\
&-\sum_{i,j}T_{ijj}a_ib_jc_j
+2\sum_iT_{iii}a_ib_ic_i. \tag{T8.7}
\end{split}
\]

This is Mobius inversion on `Pi_3`, with signature

\[
[+1;\ 3\times(-1);\ +2]
\]

for graph-label partitions `(111),(21),(3)`.  In the repeated directions,

\[
\begin{split}
T^\circ(u,u,v)
={}&T(u,u,v)-\sum_{ik}T_{iik}u_i^2v_k\\
&-2\sum_{ij}T_{iij}u_iu_jv_i
+2\sum_iT_{iii}u_i^2v_i, \tag{T8.8}
\end{split}
\]

\[
T^\circ(u,u,u)=T(u,u,u)
-3\sum_{ik}T_{iik}u_i^2u_k
+2\sum_iT_{iii}u_i^3. \tag{T8.9}
\]

The `(3)` quotient is the univariate k3 diagonal.  The three `(21)` quotients
are the oriented K21/repeated-hook planes.  D11's CP3 read and cubic coordinate
star own these collision terms.  The external K31 collision `a=b` is separate:
the local adjoint must be off diagonal because D11 zeros K31's diagonal and
owns it in K4.  Applying an unmasked trace action would duplicate K4.

### Does H31/CP3 already own the star?

Yes, for the assembled model.  The production CP3 cap is large enough to keep
all model births over depth 16; `full-birth` refers to model births, not truth.
The carried CP factors supply the modeled all-distinct cubic content, while
the coordinate star is reset to `K21_target-CP3_post_K21` and owns the repeated
hook residual.  After push, their sum gives the complete modeled H3 interface
used by Q24.

The Q24 H31 response consumes the same modeled k3/K21 data when forming
`m31` and connected K31.  Therefore, for the assembled tensor

\[
T^{old}=T^{CP3}+T^{cubic\ star},
\]

the direct action (T8.4) satisfies the duplicate receipt

\[
\boxed{
\mathcal A_{D3}(T^{old})
=\langle\Lambda_{off},K31^{H31}[T^{old}]\rangle_F.} \tag{T8.10}
\]

Re-evaluating (T8.10) through a derivative or all-distinct schedule is not a
new source.  Subtract at the matching site: post-weight/pre-ReLU cores subtract
assembled `pre_k31`; post-reset cores subtract `k31_target`.  Mixing the two
sites creates a spurious defect.

The only possible same-site source is

\[
T^{res}=T^{true}-T^{old}. \tag{T8.11}
\]

Its graph-injective action is obtained by replacing both tensors in (T8.4)
with the collision-subtracted residuals from (T8.7)--(T8.9).  However, K31 is
not the complete old fourth-order interface.  The final novelty operator is

\[
\boxed{
\mathcal A_{D3}^{novel}
=(I-P_{\mathcal O_4})\,
\mathcal A_{D3}^{\circ}(T^{true}-T^{old}),} \tag{T8.12}
\]

where `O4` is the complete old CP4 plus K4/K31/K22 star/pair response span,
including positive-depth arena words.  Using only `ker(K31)` can falsely
credit a term already visible through K4, K22, or an old depth word.

### First nonduplicate topology and its query

After the `(3)` and `(21)` quotients and the complete old modeled component are
removed, the first surviving connected topology is the injective trivalent
reset star

\[
\mathsf{Star}_3^\circ:
\quad\text{one exact }D^3f\text{ vertex joined to three protected H1 legs}.
\]

It is a tree, not a fish or C4 loop.  Consequently its action is locally
queryable without materializing a cubic tensor if the reset exposes

\[
\boxed{
\operatorname{D3Action}_l(f;a,b,c)
=E[D^3f_l(a,b,c)]} \tag{T8.13}
\]

together with the existing k3/K21 collision receipts needed in (T8.7).  Two
calls, for `(U;u,u,v)` and `(V;u,u,u)`, give (T8.4); their assembled CP3/H31
counterparts are then subtracted locally.

No current Site A/B object implements (T8.13).  Thus
`Star3_exact-minus-assembled` is the first topology that is simultaneously
nonduplicate and scalar-local **conditional on a new directional-D3 reset
interface**.  Under the present card there is no topology satisfying both
properties: every available local D3/H31 value is old, and the exact residual
query is absent.

### Separate graph nodes and statuses

1. `D3_H3H1_NORMAL_ORDER_ID` — **proved/terminal**, equations
   (T8.1)--(T8.6).
2. `D3_GRAPH_COLLISION_MOBIUS` — **proved/terminal**, equations
   (T8.7)--(T8.9).
3. `D3_K3_K21_COLLISION_OWNERSHIP` — **proved old**: `(3)` is k3 and `(21)`
   is CP3/cubic-star K21.
4. `D3_CP3_H31_ASSEMBLED_DUPLICATE` — **proved old**, receipt (T8.10), with
   binding off-diagonal and same-site convention.
5. `D3_TRUE_MINUS_ASSEMBLED_SOURCE` — **mathematically defined/open** by
   (T8.11); current state does not observe it.
6. `D3_COMPLETE_H4_SCHUR_GATE` — **binding/open**: apply (T8.12) against CP4,
   K4/K31/K22, and every old arena word.
7. `D3_DIRECTIONAL_RESET_INTERFACE` — **specified/absent**, equation (T8.13).
8. `D3_INJECTIVE_STAR3_CANDIDATE` — **first conditional GO topology**; it
   opens only if nodes 5--7 pass and closes if its Schur residual is zero.

Edges are `NORMAL_ORDER -> COLLISION_MOBIUS -> COLLISION_OWNERSHIP ->
ASSEMBLED_DUPLICATE`; `TRUE_MINUS_ASSEMBLED_SOURCE` and
`DIRECTIONAL_RESET_INTERFACE` feed `INJECTIVE_STAR3_CANDIDATE`, which must pass
`COMPLETE_H4_SCHUR_GATE` before receiving novelty credit.

## Checkpoint 9: typed Z22 future-completion witness

The directional-D3 star remains terminal under the current card: its
exact-minus-assembled interface is absent.  This checkpoint does not reopen it.
It instead uses a fourth-order plane that Q24 already computes transiently and
then discards.

### Connected and collision-subtracted definition

At post-reset layer `l`, define the symmetric off-diagonal residual

\[
\boxed{
Z_l=\operatorname{offdiag}\operatorname{sym}
\{K22_l^{target}-K22_l^{CP4}-\alpha_lP_{22}(C_l)\},} \tag{T9.1}
\]

where

\[
P_{22}(C)_{ij}=v_iv_j+2C_{ij}^2. \tag{T9.2}
\]

`K22_target` is already the connected cumulant

\[
\kappa_4(X_i,X_i,X_j,X_j)
=E[\widetilde X_i^2\widetilde X_j^2]-v_iv_j-2C_{ij}^2. \tag{T9.3}
\]

Thus (T9.1) is connected and no Gaussian Wick disconnection is subtracted
again.  The term `alpha P22(C)` is a fitted **connected old provider shape**,
not the disconnection in (T9.3).  The fit gives the exact current-plane receipt

\[
\langle Z_l,P_{22}(C_l)\rangle_F=0. \tag{T9.4}
\]

up to the explicit symmetrization convention.

The graph/external equality type of `Z` is `(22)`: two occurrences of one root
and two of another.  Root injectivity is only

\[
(\mathfrak I_{22}Z)_{ij}=(1-\delta_{ij})Z_{ij}. \tag{T9.5}
\]

Its sole coarser root quotient is `(4)`, already owned by K4.  This `[+1,-1]`
two-block Mobius operation is distinct from latent-Hermite and internal-traffic
collisions.  A general fourth tensor also has `(31),(211),(1111)` sectors;
`Z22` does not determine them.

### Minimal typed completion

The smallest coordinate-basis witness is the pure-22 symmetric lift

\[
\mathcal E_{22}(Z)_{ijkl}
=\begin{cases}
Z_{pq},&\{i,j,k,l\}\text{ is the multiset }\{p,p,q,q\},\ p\ne q,\\
0,&\text{otherwise.}
\end{cases} \tag{T9.6}
\]

All six permutations of `p,p,q,q` receive the same value.  Consequently

\[
R_{22}\mathcal E_{22}(Z)=Z,\qquad
R_{31}\mathcal E_{22}(Z)=R_{40}\mathcal E_{22}(Z)=0. \tag{T9.7}
\]

This is a typed minimal representative, not a theorem that the physical
missing tensor has zero 211/1111 completion.  Its effect is causally future
only: it is born after reset `l`, first mixes through `W_{l+1}`, and may first
be consumed at reset `l+1`.

### Exact first dense K31, K4, and K22 actions

Write `W` for the source-to-destination dense map and `w_a=W[:,a]`.  Direct
enumeration of the six permutations in (T9.6) gives

\[
\boxed{
\delta K31_{ab}
=3(w_a^{\odot2})^TZ(w_a\odot w_b).} \tag{T9.8}
\]

Equivalently, with `W2=W odot W`,

\[
\boxed{
\delta K31=3\{W\odot(ZW2)\}^TW.} \tag{T9.9}
\]

The diagonal quotient is

\[
\boxed{
\delta k4_a=3(w_a^{\odot2})^TZw_a^{\odot2}.} \tag{T9.10}
\]

It must be routed to K4, not left on the zero-diagonal K31 plane.  For two
distinct downstream roots,

\[
\boxed{
\begin{split}
\delta K22_{ab}
={}&(w_a^{\odot2})^TZw_b^{\odot2}\\
&+2(w_a\odot w_b)^TZ(w_a\odot w_b).
\end{split}} \tag{T9.11}
\]

The first term in (T9.11) is `W2^T Z W2`.  The second is a crossed
Khatri-Rao quadratic form.  Q24's next reset consumes these planes through its
literal `L4` branches: K31/transpose with the `/6` `(31)/(13)` response and
K22 with the `/4` `(22)` response.  Any physical completion must respect all
three generated collision types; a K31-only calculation is a Bellman witness,
not a complete forward recurrence.

### Exact overlap with the deployed old state

At the birth site, overlap is partitioned as follows.

1. CP4's actual K22 plane is arithmetically subtracted once in (T9.1).
2. The fitted covariance-pairing direction is subtracted once and is
   orthogonal by (T9.4).
3. The quartic coordinate star has current `(31)/(4)` support and therefore
   does not store `Z22` at birth; after dense mixing, however, its old word can
   overlap the same downstream K31/K22 action.
4. CP4 is a full symmetric tensor, so subtracting its realized current K22
   contribution prevents source duplication but does not make the chosen
   pure-22 lift orthogonal to the whole CP4 future-response span.
5. The depth arena already contains positive-depth CP4, pairing, and quartic-
   star words.  Endpoint collinearity with those words is not excluded by
   (T9.4).

Therefore the lawful action is

\[
\boxed{
\chi_Z^{novel}
=(I-P_{\mathcal O_4})
\left\langle\Lambda_{future},
R_*W^{\otimes4}\mathcal E_{22}(Z)\right\rangle,} \tag{T9.12}
\]

where `O4` is the complete CP4/pair/star/depth-arena response span and `R_*`
contains the separately typed K4/K31/K22 consumers.  Arithmetic subtraction
at birth and endpoint Schur residualization are both required.

### Smallest nonduplicate contraction and cost

The cheapest first witness is the off-diagonal K31 Bellman action.  Let
`Lambda31` be the future K31 costate with zero diagonal and set

\[
R=ZW2,qquad B=W\Lambda_{31}^T.
\]

Then (T9.9) implies the scalar identity

\[
\boxed{
\chi_{Z\to31}=3\langle R,W\odot B\rangle_F.} \tag{T9.13}
\]

`Z` is available transiently while fitting `alpha`, and `W2` is already a D11
weight object.  Forming `R=ZW2` costs one dense matrix product.  If the reverse
consumer already exposes `B=W Lambda31^T`, the rest is elementwise and the
incremental cost is one GEMM-equivalent for this transition.  If `B` is not
reusable, it costs one more, for two.  The K4 diagonal (T9.10) is available
from the diagonal of `W2^TR` with cheap reduction or a reused contraction.

The full K22 plane is different.  Its first term needs one additional dense
product once `R` is available.  The crossed term in (T9.11) is generically
`O(n^4)`: computing it for all `(a,b)` applies a dense quadratic form to every
Khatri-Rao column `w_a odot w_b`.  If the K22 Bellman matrix has rank `k`, its
scalar action can be reduced to `O(k n^3)` by factoring the costate; no
constant-cost result follows for generic rank `n`.

Hence the smallest locally queryable nonduplicate object is

\[
S_{old}\chi_{Z\to31}, \tag{T9.14}
\]

costing one to two dense products after the exact arena Schur gate.  It is a
one-step value witness.  Promoting it to a self-consistent full fourth-order
provider additionally requires either a low-rank/factorized K22 action or a
certified bound showing that the omitted crossed-K22 branch is Bellman-dark.

### Split nodes and statuses

1. `D3_DIRECTIONAL_RESET_INTERFACE` — **terminal NO under current state**;
   retained only as a closed sibling.
2. `Z22_CONNECTED_ROOT_OWNERSHIP` — **proved/terminal**, equations
   (T9.1)--(T9.5).
3. `Z22_PURE22_MINIMAL_LIFT` — **proved as a typed representative, physical
   completion open**, equations (T9.6)--(T9.7).
4. `Z22_FIRST_DENSE_K31_K4_ACTION` — **proved/local**, equations
   (T9.8)--(T9.10).
5. `Z22_FIRST_DENSE_K22_ACTION` — **formula proved; generic cost NO**, equation
   (T9.11).
6. `Z22_CP4_PAIR_BIRTH_SUBTRACTION` — **proved/closed** by the literal reset
   partition and (T9.4).
7. `Z22_COMPLETE_ARENA_SCHUR_GATE` — **open/binding**, equation (T9.12).
8. `Z22_K31_SCALAR_WITNESS_COST` — **conditional GO at one to two GEMMs**,
   equations (T9.13)--(T9.14); requires reverse-product reuse audit.
9. `Z22_FULL_PROVIDER_K22_COST` — **open only under low-rank costate or dark-
   branch certificate; generic full-plane route closed**.

Edges are `CONNECTED_ROOT_OWNERSHIP -> PURE22_MINIMAL_LIFT ->
FIRST_DENSE_K31_K4_ACTION,FIRST_DENSE_K22_ACTION`; birth subtraction and all
three dense actions feed `COMPLETE_ARENA_SCHUR_GATE`; only after that gate may
`K31_SCALAR_WITNESS_COST` be promoted.  `FULL_PROVIDER_K22_COST` is a separate
child and must not be hidden inside the cheap scalar witness.

## Checkpoint 10: exact Bellman rank of the crossed-22 branch

This checkpoint separates three statements which must not be conflated:

1. the `Z22 -> K31` contraction is an exact, cheap partial Bellman witness;
2. a physical pure-22 completion also generates a crossed K22 response;
3. a nonzero Schur residual of the first statement says nothing quantitative
   about the omitted second statement.

### Typed crossed map and its exact adjoint

Put

\[
U=W\odot W,\qquad u_a=U[:,a]=w_a^{\odot2},\qquad
q_{ab}=w_a\odot w_b. \tag{T10.1}
\]

The expensive part of (T9.11) is the symmetric off-diagonal plane

\[
(\mathcal X_WZ)_{ab}=2q_{ab}^{T}Zq_{ab}. \tag{T10.2}
\]

Only the symmetric off-diagonal part of the future K22 covector is typed as
K22, so write

\[
\bar\Lambda_{22}=\operatorname{offdiag}\operatorname{sym}\Lambda_{22}.
\tag{T10.3}
\]

Including the `/4` factor in the literal Q24 action (C74), the crossed scalar
and its source-side adjoint are exactly

\[
\begin{split}
\mathcal A_{22}^{\times}(Z)
&=\frac12\sum_{a,b}\bar\Lambda_{22,ab}
q_{ab}^{T}Zq_{ab}\\
&=\langle Z,\Omega_{\times}\rangle_F,\\
\boxed{\Omega_{\times}}
&=\frac12\sum_{a,b}\bar\Lambda_{22,ab}q_{ab}q_{ab}^{T}.
\end{split} \tag{T10.4}
\]

Elementwise, with

\[
r_{ij}=W[i,:]\odot W[j,:],
\]

this is

\[
(\Omega_{\times})_{ij}
=\frac12r_{ij}^{T}\bar\Lambda_{22}r_{ij}. \tag{T10.5}
\]

Thus this is a quadratic-measurement adjoint, not the already available
ordinary reverse product `W Lambda31^T`.

### Bellman separation rank

For a symmetric spectral factorization

\[
\bar\Lambda_{22}=\sum_{s=1}^{r}\sigma_sx_sx_s^{T},
\qquad r=\operatorname{rank}\bar\Lambda_{22}, \tag{T10.6}
\]

define

\[
A_s=W\operatorname{Diag}(x_s)W^{T}. \tag{T10.7}
\]

Then (T10.4) becomes the exact factorization

\[
\boxed{
\Omega_{\times}=\frac12\sum_{s=1}^{r}
\sigma_s(A_s\odot A_s).} \tag{T10.8}
\]

Each `A_s` is one dense congruence (one dense product after diagonal scaling),
followed by an elementwise square and a Frobenius contraction with `Z`.
Therefore rank `r` gives an exact `O(r n^3)` target-conditioned action without
forming the full forward K22 plane.  The nonsymmetric version uses a rank
factorization `Lambda=sum x_s y_s^T` and
`A(x_s) odot A(y_s)`.

The rank is generically not an artifact of this derivation.  Restrict both
source and destination to off-diagonal symmetric pairs.  The coefficient from
source pair `{i,j}` to destination pair `{a,b}` in (T10.2) is, up to a fixed
nonzero factor,

\[
W_{ia}W_{ib}W_{ja}W_{jb}. \tag{T10.9}
\]

Take `W=I+epsilon A`, with `A_ii=0` and every `A_ab A_ba` nonzero.  For the
matching pair `{i,j}={a,b}`, (T10.9) has leading order
`epsilon^2 A_ab A_ba`; a pair sharing one root starts at order `epsilon^3`,
and a disjoint pair starts at order `epsilon^4`.  The determinant of the
off-diagonal pair map therefore has a nonzero leading diagonal monomial.  It
is not the zero polynomial.  Consequently

\[
\boxed{\mathcal X_W:\operatorname{Sym}_0(n)\to
\operatorname{Sym}_0(n)\text{ is invertible for generic dense }W,} \tag{T10.10}
\]

where `Sym_0` denotes symmetric zero-diagonal planes.  Its adjoint is also
injective.  Hence a generic nonzero K22 covector is not made future-dark by
dense mixing, and no constant-dimensional universal compression of the
off-diagonal action exists.

### The K4 diagonal quotient and the true minimum rank

Using `rank(offdiag Lambda22)` directly is slightly too pessimistic because
the missing diagonal is owned by K4.  For the pure-22 lift,

\[
(\delta K22)_{aa}=3u_a^TZu_a=(\delta k4)_a. \tag{T10.11}
\]

Thus, for any diagonal vector `d`, the complete C74 action is unchanged under

\[
\boxed{
(\Lambda_{40},\bar\Lambda_{22})
\sim(\Lambda_{40}-6d,
\bar\Lambda_{22}+\operatorname{Diag}d).} \tag{T10.12}
\]

Indeed the added K22 term is `(1/4)d^T deltaK22=(3/4)d^T g`, where
`g_a=u_a^T Z u_a`, while the K4 change is
`(-6/24)d^T delta k4=-(3/4)d^T g`.  If the crossed and uncrossed K22 terms are
accounted separately, their compensating K4 shifts are respectively `-4d`
and `-2d`; together they give `-6d`.

The exact Bellman rank relevant to a full K22+K4 scalar evaluator is therefore

\[
\boxed{
r_{22}^{\rm Bell}
=\min_{d\in\mathbb R^n}
\operatorname{rank}
(\bar\Lambda_{22}+\operatorname{Diag}d).} \tag{T10.13}
\]

This diagonal-completion quotient is the sharp duplicate-aware rank.  A
constant-cost route exists precisely when (T10.13) is `O(1)` (or when the
dual source `Z` itself has `O(1)` rank).  In the latter case, writing
`Z=sum_t tau_t z_t z_t^T` gives

\[
\mathcal X_WZ
=2\sum_t\tau_t
\{W^T\operatorname{Diag}(z_t)W\}^{\odot2}, \tag{T10.14}
\]

again one congruence per source-rank term.

The actual D11 covector has the natural diagonal completion

\[
M_{22}=B\odot G22,
\qquad \bar\Lambda_{22}=\operatorname{offdiag}M_{22}. \tag{T10.15}
\]

Therefore

\[
r_{22}^{\rm Bell}\leq\operatorname{rank}(B\odot G22)
\leq\operatorname{rank}(B)\operatorname{rank}(G22). \tag{T10.16}
\]

This is a genuine low-rank escape if either the Bellman plane or the literal
Q24 derivative admits a small factorization.  There is no such structural
identity in the current recurrence: `B=W H W^T` is generically full rank and
the finite-rho Q24 derivative `G22` is a dense entrywise-kernel plane, also
generically full rank.

There is also a sharp generic lower bound even after the legal diagonal
completion.  Symmetric matrices with nullity at least `k` have codimension
`k(k+1)/2`.  Only `n` diagonal entries may vary in (T10.13).  Hence, for a
generic prescribed off-diagonal plane, a completion of nullity `k` is
impossible whenever

\[
\frac{k(k+1)}2>n. \tag{T10.17}
\]

At width `n=1024`, `44*45/2=990` while `45*46/2=1035`, so a generic completion
has

\[
\boxed{r_{22}^{\rm Bell}\geq 1024-44=980.} \tag{T10.18}
\]

This closes the constant-context crossed-K22 route for a generic Bellman
plane.  It can reopen only with an explicit small-rank factorization, not with
the statement that a diagonal was omitted.

### Exact future-dark gate

Let `S_Z` be the admissible Z22 source subspace after birth subtraction and
the complete old-arena source/endpoint Schur quotient.  The crossed branch is
universally future-dark on that source precisely when

\[
\boxed{
P_{S_Z}\operatorname{offdiag}\Omega_{\times}=0.} \tag{T10.19}
\]

For one realized source this weakens only to
`<Z,Omega_times>=0`; that scalar cancellation is not a universal darkness
certificate.  Sufficient special cases are `barLambda22=0`, a transport whose
active `q_ab` vanish on `S_Z`, or membership of the pulled-back action in the
complete old arena.  For generic dense `W`, (T10.10) says the first unquotiented
branch is dark only on a proper algebraic set.  Cancellation against K31 or K4
also does not make K22 dark; it makes only the total action zero for that one
covector.

### Why the `R=ZU` basis is sufficient but not free

If `U=W odot W` is invertible, the already formed K31 intermediate `R=ZU`
retains all information in `Z`.  Every scalar linear action can formally be
written

\[
\langle Z,L\rangle_F
=\langle R,H\rangle_F,
\qquad H=LU^{-T}. \tag{T10.20}
\]

For the crossed branch, however,

\[
H_{\times}=\Omega_{\times}U^{-T}. \tag{T10.21}
\]

Equations (T10.8) and (T10.13) are still required to compile
`Omega_times`, followed by a dense solve.  Thus (T10.20) removes an
information obstruction, not the Bellman-rank cost.  Nor is invertibility of
`U` automatic: an invertible normalized Hadamard matrix `W` has
`W odot W=(1/n)11^T`, which has rank one.  Any `R`-basis proposal therefore
needs separate invertibility and conditioning receipts.  The existing K31
reverse product cannot supply (T10.21): setting `Lambda31=0` while taking a
nonzero `Lambda22` leaves that product zero but gives a generically nonzero
crossed adjoint by (T10.10).

### What the one-product K31 witness may lawfully claim

Let `s31` be the endpoint correction generated by (T9.13), let `O` be the
complete old CP3/Q24/CP4/pair/star/depth-arena design, and put

\[
s_{31}^{\perp}=(I-P_O)s_{31},
\qquad e_O=(I-P_O)e. \tag{T10.22}
\]

If `s31_perp` is nonzero, the K31-only feature is a mathematically lawful new
feature.  Its exact one-feature incremental projection value is

\[
\boxed{
\Delta_{31\mid O}
=\frac{\langle e_O,s_{31}^{\perp}\rangle^2}
{\|s_{31}^{\perp}\|^2}.} \tag{T10.23}
\]

Thus full old-arena Schur residualization permits **credit to the restricted
K31 witness itself**.  It does not certify the value, sign, or completeness of
the physical Z22 lift.  A minimal counterexample has `O=0`: choose
`s31=e,s22=-e`; then the K31 witness has perfect individual value while the
complete source has zero action.  Conversely `s31=0,s22=e` makes the cheap
witness dark while the omitted branch owns all value.  These examples remain
valid after replacing every vector by its component orthogonal to a nontrivial
old arena.

Therefore K31-only may be promoted as a target-free approximate closure
feature after (T10.22), but may not be described as a complete Z22 provider or
used as a lower/upper bound on its full gain.  Promotion to a physical
self-consistent provider requires either the low-rank gate (T10.13), the exact
darkness gate (T10.19), or explicit inclusion of the omitted K22/K4 actions.
Pure mathematics proves admissibility and generic nonzeroness, not the
magnitude of (T10.23).

### Split node ledger

1. `Z22_K31_WITNESS_EXACT` -- **proved/conditional GO**: (T9.13) is an exact
   one-to-two-product partial Bellman action; it receives novelty credit only
   after the complete old-arena Schur gate (T10.22).
2. `Z22_K31_WITNESS_VALUE` -- **open quantitative**: its lawful value is
   (T10.23); no analytic receipt currently proves a goal-scale fraction.
3. `Z22_CROSSED_K22_ADJOINT_IDENTITY` -- **proved/terminal**, equations
   (T10.4)--(T10.8).
4. `Z22_CROSSED_K22_GENERIC_VISIBILITY` -- **proved/terminal**: the dense
   off-diagonal pair map is generically injective by (T10.9)--(T10.10).
5. `Z22_K22_BELLMAN_COMPLETION_RANK` -- **generic constant-cost NO; conditional
   reopen**: exact quotient rank (T10.13), generic lower bound (T10.18), reopen
   only on an explicit low-rank receipt such as (T10.16).
6. `Z22_K22_FUTURE_DARK_GATE` -- **specified/open for the instance**:
   (T10.19) is necessary and sufficient; generic dense transport does not
   satisfy it automatically.
7. `Z22_R_BASIS_COMPLETENESS` -- **information YES, cost NO without a new
   factorization**: (T10.20)--(T10.21), plus a separate invertibility and
   conditioning gate on `W odot W`.
8. `Z22_FULL_PROVIDER_VALUE` -- **not creditable from K31 alone**: only the
   complete typed C74 action after full Schur has physical-provider value;
   the two-vector counterexamples following (T10.23) are the exact duplicate/
   omission gate.

The graph edges must keep `K31_WITNESS_EXACT -> K31_WITNESS_VALUE` separate
from `CROSSED_K22_ADJOINT_IDENTITY -> K22_BELLMAN_COMPLETION_RANK` and
`K22_FUTURE_DARK_GATE`.  All three typed actions and the arena Schur gate feed
`Z22_FULL_PROVIDER_VALUE`; the cheap K31 node must not bypass K22 completeness.

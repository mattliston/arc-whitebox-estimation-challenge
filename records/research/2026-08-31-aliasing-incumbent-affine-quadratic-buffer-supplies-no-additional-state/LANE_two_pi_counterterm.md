# Two-PI Counterterm / NCQC Shared-Core Lane

Status: mathematical derivations complete through the joint-cone theorem; deployability/observability gate under active derivation.  No numerical result is claimed here.

## 1. D11-reference 2PI counterterm

Normal-order every cubic vertex about the same D11 covariance.  Let `T_D` be the union of the cubic content already represented by G11/Q24 and CP3/star, counted once, and let `R=T-T_D` be the omitted normal-ordered cubic vertex.  For

\[
\Phi_3[T;G]=\frac1{12}T_{abc}T_{def}G_{ad}G_{be}G_{cf},
\]

the D11-reference functional is

\[
\Delta\Phi_3[G]=\Phi_3[T_D+R;G]-\Phi_3[T_D;G].
\]

Therefore

\[
\Delta\Sigma_{ij}=\frac12\left[(T_D)_{iab}G_{ac}G_{bd}R_{jcd}
+R_{iab}G_{ac}G_{bd}(T_D)_{jcd}
+R_{iab}G_{ac}G_{bd}R_{jcd}\right].
\]

The pure `T_D x T_D` fish has counterterm coefficient zero: it is the same normal-ordered `m=3`/Q24 diagram already inside D11.  Equivalently, after pair whitening with `F_V` and `F_D`,

\[
\boxed{\Delta\Sigma=\tfrac12(F_VF_V^T-F_DF_D^T),}
\]

not `1/2 (F_V-F_D)(F_V-F_D)^T`.

## 2. Generic cost and the NCQC exception

For an opaque residual cubic tensor, the task contraction has the triangular-prism factor graph and treewidth 3, hence the standard dense contraction costs `Theta(n^4)` arithmetic and `Theta(n^3)` state.  A generic symmetric cubic also needs CP rank `Omega(n^2)` by dimension counting.

NCQC evades this obstruction with

\[
V_{ijk}=8\,\mathrm{tr}(A_iA_jA_k),\qquad A_i\in\mathrm{Sym}_r,\qquad r^2=\Theta(n).
\]

For `G=SS^T`, put `B_alpha=sum_i S_{i alpha} A_i` and `Y_{alpha beta}=B_alpha circ B_beta`, where `A circ B=(AB+BA)/2`.  Then

\[
F^V_{a,\alpha\beta}=8\langle A_a,Y_{\alpha\beta}\rangle_F,
\]

and

\[
\Sigma^{VV}=32XL_JX^T,\qquad
L_J=\sum_{\alpha\beta}y_{\alpha\beta}y_{\alpha\beta}^T.
\]

In explicit matrix indices,

\[
H_{qr,vw}=\sum_\alpha B_\alpha[qr]B_\alpha[vw]=(X^TGX)_{qr,vw},
\]

\[
L_{pq,uv}=\sum_{r,w}H_{qr,vw}H_{rp,wu}.
\]

Forming `H`, the crossed contraction `L` (`r^6=n^3`), and `XLX^T` is `O(n^3)` with `O(n^2)` state.  Finite D11 star/pair cross motifs are also `O(n^3)`; for example a star gives `M_a=sum_i C_ai B_i` and `tr(A_b M_a^2)`.

## 3. Information retained by the fish

The full fish is PSD, but the D11 counterterm difference is indefinite:

\[
h^T\Delta\Sigma h=\frac12\big(\|F_V^Th\|^2-\|F_D^Th\|^2\big).
\]

It retains signed pairwise inner products between external cubic slices and the sign of their energy difference.  It loses global cubic sign and every right-pair rotation `F -> F O`; it cannot replace the direct signed `k3` tensor.

## 4. Exact H3/H4 quadratic-chaos hierarchy

For `Q_A=z^T A z-tr(A)`,

\[
\log E\exp(\sum_i t_iQ_{A_i})
=\sum_{m\ge2}\frac{2^{m-1}}m\mathrm{tr}(\sum_i t_iA_i)^m.
\]

Consequently

\[
K21_{ij}=8\,\mathrm{tr}(A_i^2A_j),
\]

\[
K31_{ij}=48\,\mathrm{tr}(A_i^3A_j),
\]

\[
K22_{ij}=16\langle A_i^2,A_j^2\rangle+32\|A_i\circ A_j\|_F^2.
\]

The `211` fourth cumulant is

\[
q_4^{211}(A;B,C)
=16\langle A^2,B\circ C\rangle+32\langle A\circ B,A\circ C\rangle
=48\mathrm{tr}(A^2BC)+8\mathrm{tr}([A,B][A,C]).
\]

For `B=C`, the commutator term is `-8 ||[A,B]||_F^2`, an exact negative Lie-sector correction.

With `U_{A,B}=I_4(A \widetilde\otimes B)` and `Q_AQ_B=U_{A,B}+4Q_{A\circ B}+2<A,B>`, normal ordering gives

\[
\kappa_4(A,A,B,C)
=\langle U_{A,A},U_{B,C}\rangle
+32\langle A^2,B\circ C\rangle
-8\langle A,B\rangle\langle A,C\rangle.
\]

## 5. Row/column-Gram identity and exact zero-double-credit split

Let `X` have rows `svec(A_i)`, `K=XX^T`, `P=X^T K^+ X`, `Y_ij=A_i circ A_j`, and `t_ij=8 X y_ij`.  Then

\[
\langle P Y_{ij},P Y_{kl}\rangle
=\frac1{64}t_{ij}^T K^+t_{kl}.
\]

The trajectory-identifiable C4 component is

\[
q_{4,\parallel}(i,j,k,l)
=\frac14\sum_{\text{three pairings}}t_{\mathrm{pair1}}^T K^+t_{\mathrm{pair2}}.
\]

The exact complement uses `Y_perp=(I-P)Y`:

\[
q_{4,\perp}(i,j,k,l)
=16\sum_{\text{three pairings}}\langle Y^\perp_{\mathrm{pair1}},Y^\perp_{\mathrm{pair2}}\rangle.
\]

For `211`,

\[
q^{211}_{4,\parallel}
=\frac14[t_{ii}^TK^+t_{jk}+2t_{ij}^TK^+t_{ik}],
\]

\[
q^{211}_{4,\perp}
=16\langle Y^\perp_{ii},Y^\perp_{jk}\rangle
+32\langle Y^\perp_{ij},Y^\perp_{ik}\rangle.
\]

If `T` is the H3 pair-slice matrix and `W=G tensor G`, set `That=K^{+1/2} T W^{1/2}`.  The normalized fish and parallel-q4 Grams are `That That^T` and `That^T That`; they share exactly every nonzero singular value.  They are two task consumers of one source orbit, never two source hypotheses.

The complement is nonzero exactly when the trajectory span `A=span{A_i}` is not Jordan closed:

\[
q_{4,\perp}\ne0\iff A\circ A\not\subseteq A.
\]

Indeed `q4_perp(u,u,u,u)=48 ||(I-P)A(u)^2||_F^2`.

## 6. Shared `O(n^3)` fish and contracted-q4 core

For a fixed trajectory pair metric `W`, form

\[
H_{qr,vw}=\sum_{jk}W_{jk}A_j[qr]A_k[vw]=X^TWX.
\]

Let

\[
M_{qs}=\sum_rH_{qr,rs},\qquad
R(H)_{(pq),(rs)}=H_{qr,sp}.
\]

Then all rooted contracted `211` actions are

\[
\sum_{jk}W_{jk}q^{211}_{i;jk}
=32\mathrm{tr}(A_i^2M)+16\mathrm{vec}(A_i)^TR(H)\mathrm{vec}(A_i).
\]

The fish uses the same `H` and its crossed square.  A fixed number of deployed contractions is therefore `O(n^3)/O(n^2)`; this does not authorize materializing the full three-index q4 tensor, whose standard route is `O(n^4)`.

## 7. Four-consumer Schur ledger

The consumers are signed H3 `h`, counterterm fish `f`, parallel q4 `p`, and perpendicular q4 `r`.  With `Z=[h,f,p]`, Gram `A=<Z,Z>`, `g=<Z,r>`, `gamma=<r,r>`, target crosses `c=<e,Z>` and `c_r=<e,r>`, the exact joint gain is

\[
\Delta=c^TA^+c+\frac{(c_r-g^TA^+c)^2}{\gamma-g^TA^+g}.
\]

The last term is zero if its denominator is zero.  It supplies the missing `2.86e-10` iff

\[
|c_r-g^TA^+c|\ge
\sqrt{2.86\times10^{-10}\,(\gamma-g^TA^+g)}.
\]

This is 27.73% of the `1.0313e-9` oracle quartic capacity, or squared correlation 2.78% with the post-perfect-v+k3 residual.

## 8. Joint C3 suppression / positive-q4 cone

For the multi-pole line metric `R_L=sum_r h_L(delta_r)P_r`, define

\[
D=T_D R_L^{1/2},\qquad U=(V-T_D)R_L^{1/2}.
\]

Then

\[
F_D=\tfrac12DD^*,\quad F_V=\tfrac12(D+U)(D+U)^*,\quad
\Delta F=\tfrac12(DU^*+UD^*+UU^*).
\]

On value subspace projector `P_v`, the `C3<0.8466` gate is

\[
P_v\Delta F P_v\preceq-P_v(\mathrm{Herm}A+\tau I)P_v,
\qquad \tau=\nu/0.8466-1
\]

(`tau=0.1812` for `nu=1`).  The sharp relaxed PSD feasibility condition is

\[
\frac12P_vDD^*P_v\succeq P_v(\mathrm{Herm}A+\tau I)P_v.
\]

Negative feedback is compatible with PSD because only `F_V=F_D+Delta F` must be PSD.

A same-core existence construction starts from any C3-feasible `A_i^-` and extends

\[
A_i=A_i^-\oplus\epsilon B_i,\qquad
B_i=\begin{pmatrix}0&C_i\\C_i^T&0\end{pmatrix}.
\]

All triple traces of the `B` block vanish, while `B_i circ B_j` is block diagonal and orthogonal to the carried off-diagonal span.  Hence this leaves H3/fish/C3 unchanged and creates pure q4-perpendicular capacity.  All Grams remain PSD.  The extension is dimension/cost admissible if `r_-+2s<=r_max` with `r_max^2=O(n)`.

It adds H2 Gram `epsilon^2 K_B`; fixed-H2 deployment needs

\[
\epsilon^2K_B\preceq K_{\rm slack}.
\]

If its residualized q4 task direction is `r_0`, `a=<e_Z,r_0>`, `b=||r_0||^2`, and `alpha=epsilon^4`, the fixed-coefficient gain is `2 alpha a-alpha^2 b`.  Its unconstrained maximum is `a^2/b`; H2 slack caps `alpha`.

## 9. Deployability gate under active derivation

The direct-sum construction proves existence, not lawful inferability.  A usable estimator must additionally prove:

1. observation-fiber identifiability of every consumed moment (or a deterministic white-box construction), modulo conjugation/left-right gauges;
2. a rank-complete cumulative trajectory frame for the carried subspaces;
3. a two-stream layer update that transports the even/Jordan core and odd/bipartite core without truth-fed state;
4. `O(n^2)` state and `O(n^3)` time for a fixed number of task contractions;
5. an explicit PSD H2 slack receipt;
6. held-out q4 identities fixed by the same core, without per-channel fitting.

Current next question: characterize necessary and sufficient observation maps, gauge/rank conditions, two-stream closure, and H2 slack under which the existential `A^- plus bipartite B` construction becomes a lawful deployed estimator.  If only H2 and H3 are observed, the bipartite block is generally not identifiable: H3 is exactly blind to it and an H2 Gram fixes `vec(C_i)` only up to a generic `O(s^2)` rotation, whereas q4 depends on the smaller left-right matrix gauge.

## 10. Deployability theorem: observation, gauge, rank, transport, and slack

### 10.1 Prediction observability is the exact requirement

Let `theta=(A^-,C)` denote the graded core, let `O(theta)` be all lawful white-box observations available to the deployed estimator, and let `C(theta)` be the collection of consumers actually used downstream (signed H3, counterterm fish, and contracted q4-perpendicular actions).  A lawful estimator exists if and only if the consumer map is constant on every observation fiber:

\[
O(\theta)=O(\theta')\Longrightarrow C(\theta)=C(\theta').
\]

Equivalently, there must exist a map `Cbar` with `C=Cbar o O`.  Recovering literal matrices is stronger than necessary.  Literal parameter identifiability holds only when every observation fiber is one orbit of the physical gauge group.  Locally this is

\[
\ker D O_\theta=T_\theta(G\cdot\theta).
\]

For the even core the gauge is simultaneous orthogonal conjugation.  For the bipartite block `C_i`, the physical gauge is

\[
(O(s)_L\times O(s)_R)\rtimes Z_2,
\]

acting by `C_i -> U C_i V^T` plus optional bipartition swap/transposition.

### 10.2 Why H2 plus H3 is insufficient for the bipartite block

For

\[
B_i=\begin{pmatrix}0&C_i\\C_i^T&0\end{pmatrix},
\]

all cubic traces vanish.  Its H2 contribution is

\[
(K_B)_{ij}=2\mathrm{tr}(B_iB_j)=4\mathrm{tr}(C_iC_j^T).
\]

Thus H2 determines the rows `vec(C_i)` only through their Gram and leaves a generic `O(s^2)` right rotation.  The q4 consumer depends on matrix multiplication (`C_i C_j^T`, `C_i^T C_j`) and is invariant only under the much smaller physical left-right gauge.  Therefore H2 plus H3 cannot determine q4-perpendicular predictions.

To make the block observable, one needs either:

1. lawful cross-q4/cross-layer observations whose stabilizer reduces the `O(s^2)` ambiguity to the physical left-right gauge; or
2. a fixed, preregistered, permutation-equivariant white-box map from weights/carried state to `C_i`, synchronized across layers.

Within-layer q22 diagonals supply only norms of commutator/pair features and do not synchronize their cross orientation.  q211-type cross information or a lawful white-box anchor is required.

### 10.3 Concrete rank conditions

Let `X_E^traj` stack the vectorized even matrices over all trajectory contexts and let `X_C^traj` stack `vec(C_i)`.  Full-core identification requires

\[
\mathrm{rank}(X_E^{traj})=d_E,
\qquad
\mathrm{rank}(X_C^{traj})=d_C=s^2,
\]

plus the stabilizer/gauge condition above.  For task prediction on a smaller reachable subspace, replace full rank by injectivity of the observation Jacobian on that reachable quotient.

Independent per-layer Gram factorizations are not enough: their `O(d)` gauges can jump by layer.  A usable trajectory core needs one globally synchronized gauge satisfying the known weight/reset transport.  Equivalently, the joint cross-layer observation fiber after transport constraints must be a single global gauge orbit, not a product of per-layer gauge orbits.

### 10.4 Necessary and sufficient H2 slack

Let

\[
S=K_{lawful}-K_-
\]

be the covariance slack computed from lawful carried quantities.  A prescribed normalized block Gram `K_B` can be inserted at scale `t=epsilon^2` exactly when

\[
S-tK_B\succeq0.
\]

A matrix realization by `s x s` matrices exists exactly when

\[
K_B\succeq0,\qquad \mathrm{rank}(K_B)\le s^2.
\]

For positive scale, one additionally needs `range(K_B) subset range(S)`.  The largest lawful scale is

\[
t_{max}=\frac1{\lambda_{max}(S^{+/2}K_BS^{+/2})}.
\]

Across a trajectory, one fixed block scale must obey `t <= min_l t_max,l`.  Since q4 amplitude scales as `epsilon^4=t^2`, its maximum physical amplitude is `alpha_max=(min_l t_max,l)^2`.  The fixed-coefficient quartic gain is the capped quadratic recorded in section 8; it must exceed `2.86e-10`.

### 10.5 Two-stream transport theorem

Use the `Z_2` grading with block-diagonal matrices as the even stream and off-diagonal matrices as the odd stream.  Jordan multiplication obeys

\[
E\circ E,\ O\circ O\in E,\qquad E\circ O\in O.
\]

Hence two algebraic streams are sufficient.  They can be packed as one graded forward matrix-core stream; a second D10-style reverse H4/value-adjoint stream consumes q4 births without materializing a forward H4 tensor.

At a linear layer,

\[
E'_a=\sum_iW_{ai}E_i,\qquad C'_a=\sum_iW_{ai}C_i.
\]

If `d_E+d_C=O(n)`, these are at most two `n x n`-scale matrix multiplies, `O(n^3)` time and `O(n^2)` state.  A nonlinear/reset layer remains in the two-stream class if it uses a constant number of local `r x r` products and a constant number of global `d x d` Gram aggregates, while D10 contracts q4 through fixed pair metrics.  A full `Y_ij` bank or `n` root-specific pair metrics would require `O(n^3)` state or `O(n^4)` work and fails the gate.

The existential no-H3 perturbation remains exactly decoupled across depth only for a triangular/tangent update in which `O circ O` q4 births are consumed by the reverse adjoint rather than fed back into the cubic forward core.  A fully self-consistent even/odd update is still two-stream algebraically, but its generated even part can alter later H3/C3 and therefore must re-satisfy the joint cone at every layer.

### 10.6 Existence versus a usable estimator

The hierarchy is:

1. **Existence:** a direct-sum latent block with the desired signs exists.
2. **Generated:** a deterministic white-box map produces it, but it may still be an arbitrary unidentifiable orientation.
3. **Observable:** every lawful observation fiber gives the same consumed predictions modulo physical gauge.
4. **Transportable:** one synchronized gauge obeys the layer recurrences with at most the graded forward core plus reverse H4 adjoint.
5. **Deployable:** all trajectory slack LMIs, rank limits, `O(n^3)/O(n^2)` costs, and the raw Schur threshold hold.

The current mathematical conclusion is conditional: the bipartite extension passes existence, PSD, and asymptotic cost gates, but it is not yet a usable estimator until a lawful orientation anchor/cross-q4 observation and an explicit two-stream reset recurrence are supplied.  H2/H3 alone do not supply them.

Current next question: identify the actual white-box anchor available in the D11 lineage and prove that its observation stabilizer is precisely the physical left-right gauge; otherwise close the bipartite extension as unobservable despite its latent existence.

## 11. Architecture-specific D14 + D10 word and billing gates

Cross-lane reconciliation sharpened the generic two-stream statement.

D14 already consumes one dense symmetric response word

\[
E_{l+1}=W_{l+1}^T(D_l\odot E_l+I_l^{FH})W_{l+1}
\]

for 14 transitions, billed as 28 dense GEMMs.  It carries the physically normalized sum of frame and signed-H3 injections; it does not carry an independent NCQC core or q4/Kraus family.  Therefore `A^-` and `C` may not be separately propagated in addition to `E`.  They fit the D14 word only if they are regenerated locally from lawful carried state and contribute a local, fixed-coefficient `I_l^{FH}`, or if all separately tunable tails satisfy a rank-1 shared-stream condition.

D10 supplies the second late H4/reverse word, approximately 20 GEMMs.  Its exact closure gate is

\[
T_l^*\mathcal U_{l+1}\subseteq\mathcal U_l,
\qquad
\mathcal U_l=\mathrm{Sym}^2\mathrm{span}\{a_l,b_l\},
\]

with both boundary pullbacks collinear with the single earlier word.  A Jordan-projected q211 source is sufficient only if reset and endpoint annihilate `ker J`; otherwise the full H4 square is needed.  Equivalently, every component outside `U_l` must lie in the kernel of the remaining value adjoint.

The generic q4-perpendicular source currently has effective Kraus rank 11.  An endpoint scalar does not reduce its forward/reverse word rank.  One D10 word is sufficient only if the task-relevant Kraus image collapses into the invariant `U_l` bundle (or rank-1 shared tail); otherwise the bipartite q4 construction fails the word-count gate.

D14 plus D10 already bill 48 of the 49 dense-GEMM equivalents.  The `O(n^3)` asymptotic fish/q211 algorithms are deployable only if all extra dense work is at most one GEMM-equivalent total, is fused into an already billed conjugation, or is reduced below-GEMM by diagonal/Hadamard/low-rank structure.  Computing a fresh `H=X^TWX` and crossed contraction at every layer violates the actual bill despite its asymptotic class.

### Current deployability verdict

The direct-sum NCQC construction is a valid latent existence proof, but no usable two-word estimator has yet been proved.  Three independent obstructions remain:

1. no lawful orientation anchor has reduced the bipartite `O(s^2)` H2 factor gauge to the physical left-right gauge;
2. separately propagating `A^-` or `C` would be a forbidden third dense word, while local regeneration has not been derived;
3. generic q4-perpendicular Kraus rank 11 and fresh `O(n^3)` contractions exceed D10's one-word / one-remaining-GEMM capacity.

Reopen/deploy only on receipts proving all of:

- a target-blind local source map from deployed `m,v,C`, weights, or a typed carried cumulant state;
- fixed physical frame:H3:fish coefficients for D14 (or a rank-1 shared tail);
- the D10 invariant-subspace and boundary-collinearity identities, including annihilation of every discarded `ker J`/Kraus direction;
- reuse/fusion keeping added billing at or below one GEMM-equivalent;
- the per-layer H2 slack LMIs and the `2.86e-10` Schur gain condition.

Current next question: determine whether the actual D11 carried covariance conjugation can compute the local NCQC `H`/q211 injection in-place and whether the value adjoint kills ten of the eleven q4-perpendicular Kraus modes.  Without both facts, close the deployable bipartite extension while retaining the mathematical source theorem.

## 12. Lawful-orientation-anchor classification and closure

Let `d=s^2`.  H2 identifies a factor `Z` with `ZZ^T=K_B` only up to the right action `Z -> ZQ`, `Q in O(d)`.  A physical matrix interpretation of its columns is a reduction of this structure group to

\[
H=(O(s)_L\times O(s)_R)\rtimes Z_2,
\]

equivalently a choice of point in the homogeneous space `O(d)/H`.

Deployed weights and H2 tensors carry unit indices; they are invariant under the independent latent right-gauge `O(d)`.  There is no `O(d)`-equivariant map from such gauge-invariant data to `O(d)/H`: equivariance at a fixed datum would require a point of `O(d)/H` fixed by every element of `O(d)`, which is impossible because `H` is a proper subgroup.  In invariant-tensor language, H2 supplies only the Euclidean contraction tensor; it cannot generate the matrix/Jordan multiplication tensor needed to distinguish a tensor-product reshape.  Therefore no permutation/orthogonal-natural map from weights plus H2 can *infer* the physical left-right orientation without an additional cross tensor.

The possible constructions split into three classes:

1. **Gram/isotropic maps.**  They depend only on H2 contractions.  Their q4 output is either undefined up to `O(d)` or collapses to exchangeable/collective invariants.
2. **Spectral or Krylov frame choices.**  Simple-spectrum weight/covariance operators, possibly with a covariant seed, can choose an ordered frame and then apply an arbitrary reshape.  This *generates* a convention but does not infer physical matrix multiplication from the observation fiber.  Degeneracy/sign/holonomy must be resolved, and carrying or rebuilding a full frame costs a dense extra word or fresh per-layer `O(n^3)` work, violating the D14+D10 budget.
3. **Fixed-template equivariant maps.**  `C_i=sum_{a<=k} phi_a(i) M_a`, with permutation-equivariant scalar/vector features `phi_a` and fixed matrices `M_a`.  These are lawful and cheap, but `rank(K_B)<=k`.  D10's shared-tail constraint reduces the currently compatible case to approximately `k=1`; that is the collective rank-1 family, not the generic q4-perpendicular Kraus-rank-11 source.

A canonical symmetric square root followed by a fixed reshape is not a counterexample: a generic neuron permutation/orthogonal change acts on its columns by a generic `O(d)` element, not by a left-right Kronecker gauge, so q4 predictions change.  Weight eigenframes only hide this arbitrary reduction and fail the observation-fiber criterion unless a lawful higher-order/cross-layer anchor fixes it.

### Closure decision

The broad bipartite direct-sum source is **nondeployable in this form**.  A trivial rank-1 template construction can fit the two-word architecture, but it has only collective q4 capacity and is already outside the required broad rank-11 mechanism.  Reopen the broad source only if one of the following concrete anchors is found in the deployed lineage:

- a target-blind typed fourth/cross cumulant whose stabilizer is exactly the physical left-right gauge;
- an already carried cross-layer tensor-product frame requiring no additional dense word;
- a proof that the value adjoint annihilates all but one template/Kraus direction and that this rank-1 direction alone passes the `2.86e-10` Schur gate.

Current next question: redirect the NCQC identities to local, gauge-invariant consumer formulas of the already carried D11 planes, or close the bipartite carrier permanently and retain NCQC only as an oracle/source-structure theorem.

## 13. K31 affine--quadratic pivot: exact gauge and residual audit

Let the lawful affine--quadratic observable be

\[
X_i=h_i^Tz+(z^TA_i z-\operatorname{tr}A_i),\qquad A_i=A_i^T,quad z\sim N(0,I).
\]

For one contracted task coordinate `a`, set

\[
N_a=\sum_iW_{ia}A_i,\quad R_a=\sum_iP_{ia}A_i,
\quad h_a=\sum_iW_{ia}h_i,\quad r_a=\sum_iP_{ia}h_i,
\quad P=WQ_{31}^T.
\]

The cumulant generating function

\[
\log E e^{t^TX}=
-\operatorname{tr}A(t)-\tfrac12\log\det(I-2A(t))
+\tfrac12h(t)^T(I-2A(t))^{-1}h(t)
\]

gives the exact repeated fourth cumulant

\[
\kappa_4(U_a,U_a,U_a,V_a)
=48\operatorname{tr}(N_a^3R_a)
+24\{h_a^TN_aR_ah_a+h_a^TN_a^2r_a\}.
\]

This scalar is invariant under the **physical** Gaussian-coordinate gauge

\[
A_i\mapsto O^TA_iO,\qquad h_i\mapsto O^Th_i,qquad O\in O(r),
\]

by cyclicity of trace and orthogonality.  Thus an explicitly carried, target-blind affine--quadratic coefficient state supplies its own lawful orientation and needs no bipartite `C_i` factor or arbitrary reshape.

It is **not** invariant under the larger H2-factor gauge.  To make the obstruction exact, put

\[
g((A,h),(B,k))=2\operatorname{tr}(AB)+h^Tk
\]

and define the symmetric affine--quadratic product

\[
(A,h)\star(B,k)=
\left(4(A\circ B)+\operatorname{sym}(hk^T),\;2(Ak+Bh)\right).
\]

Then `kappa_3(x,y,z)=g(x star y,z)`, while direct expansion gives

\[
\kappa_4(x,x,x,y)
=\tfrac32g(x\star x,x\star y)-3\|h_x\|^2h_x^Th_y.
\]

Let `S=span{x_i}` and `Pi` be its `g`-orthogonal projector.  H2 and H3 determine only `Pi(x_i star x_j)`.  Consequently

\[
\begin{aligned}
\kappa_{31,ij}
=&\;\tfrac32g(\Pi(x_i\star x_i),\Pi(x_i\star x_j))
-3\|h_i\|^2h_i^Th_j\\
&+\underbrace{\tfrac32g((I-\Pi)(x_i\star x_i),(I-\Pi)(x_i\star x_j))}_{R^{AQ}_{ij}}.
\end{aligned}
\]

The last term is the exact remaining gauge-sensitive fourth-order datum.  In the pure-quadratic normalization used above it reduces to

\[
R^{AQ}_{ij}=48\left\langle(I-\Pi)A_i^2,
(I-\Pi)(A_i\circ A_j)\right\rangle,
\]

and the H2/H3-visible parallel term is `(3/4) t_ii^T K^+ t_ij`.  The repeated-index specialization kills the explicit commutator term because `[A_i,A_i]=0`, but it does **not** kill this Jordan-orthogonal term.  Hence a universal closed form does not by itself make K31 identifiable from moment H1/H2/H3; it is identifiable only when the oriented `(A_i,h_i)` coefficients are themselves lawful white-box state, or when `S` is proved closed under `star` so that `R^{AQ}=0`.

### Exact marginal cost and two-stream gate

Writing `X_A` for the matrix whose rows are `vec(A_i)`, all `N_a` are one dense product `W^T X_A`.  Since `P=WQ31^T`, all `R_a` are `Q31 N`, a second dense product.  The linear blocks cost `O(n^2r)` and all per-`a` traces cost `O(nr^3)=O(n^{5/2})` for `r^2=Theta(n)`, both below a dense `n^3` GEMM.  Memory is `nr^2=Theta(n^2)`.

Therefore:

- from raw affine--quadratic coefficients, exact K31 costs two GEMM-equivalents and one additional oriented dense word, so it does not fit the present D14+D10 `48/49` ledger;
- if `N` is already exactly the existing late word and `R=Q31 N` can use the one remaining GEMM (or be fused into a billed contraction), the scalar trace evaluation fits;
- moment H2/H3 state without that oriented word cannot replace it, because doing so drops `R^{AQ}` and silently reintroduces the orientation no-go.

### Updated decision

Retire the existential bipartite factor as the route to this K31 action.  Promote the direct affine--quadratic K31 action only conditionally: **GO as a mathematical/lawful consumer if the deployed lineage already exposes `(A_i,h_i)` in a common Gaussian coordinate; NO-GO on current two-stream deployment unless `N` aliases the D10 word or the second product is fused.**  H2/H3 moments alone remain insufficient unless affine--quadratic `star`-closure is proved.

Current next question: audit whether the direct lane's purported affine--quadratic state is an explicit oriented coefficient state already present in D11, rather than only its H2/H3 moment image, and whether `N` aliases a billed D10 stream exactly.

## 14. D14/D10 alias audit, cost closure, and graph-ready split

### 14.1 Typed-state audit

The currently defined triangular schedules do **not** contain the affine--quadratic factor `N`:

- D14 carries a unit-by-unit symmetric primal tangent `E_l` for the combined frame/signed-H3 injection.  Its actual nonlinear reset coefficients are still unspecified.
- D10 carries a unit-by-unit symmetric reverse mask `B_l`, its conjugate `H_l=W_{l+1}B_{l+1}W_{l+1}^T`, and scalar accumulators.
- The AQ factor is `N_l=W_l^T X_{A,l}`, a unit-by-latent-pair rectangular array, where row `i` of `X_A` is `svec(A_{l,i})` and `s=r(r+1)/2`.

The index types differ: `(unit,unit)` for `E,B,H` versus `(unit,latent-pair)` for `N`.  Dimension compatibility is not an alias theorem.  An alias would require a specified equivariant intertwiner from the D14 unit-pair word to the oriented latent-pair coefficient word, and that intertwiner is exactly the missing orientation/graded-reset structure.  It cannot be booked from H2 moments.

There is also a causal mismatch.  `N_l` is a target-blind forward AQ coefficient, while `B_l,H_l,Q_l^{31}` depend on the endpoint adjoint.  They cannot be one state for a family of endpoint masks unless a common invariant rank-one quotient is proved.  No such quotient is in the current D10 definition.

### 14.2 Exact `R=Q31 N` identity and irreducible marginal

Let rows of `X_A` be the packed `A_i`.  Since

\[
N=W^TX_A,qquad P=W(Q^{31})^T,
\]

the second AQ factor satisfies the exact useful identity

\[
\boxed{R=P^TX_A=Q^{31}W^TX_A=Q^{31}N.}
\]

Thus `P` need not be materialized.  But this does not make `R` free.  For generic dense `Q31`, forming `Q31 N` costs one `n x n` by `n x s` contraction, namely `s/n` full-GEMM equivalents per active layer.  The scalar alternative forms the all-pairs matrix

\[
F_{ab}=48\operatorname{tr}(N_a^3N_b)+\text{the two affine vector terms}
\]

and contracts `sum_ab Q31_ab F_ab`; forming `F` has the same `n x s` by `s x n` cost.  It removes explicit `R`, not the contraction.

This contraction cannot share arithmetic with the already billed

\[
H=W B W^T,qquad Q^{31}=\tfrac16\{H\odot G^{31}+(H\odot G^{13})^T\},
\]

because it has a new latent-pair right-hand side and occurs after a generic Hadamard mask.  Kernel batching can save materialization/launch overhead but not the `n^2s` multiply-add count.  Exact sharing would require a proved low-rank/separable `Q31`, a paid identical pair Gram, or an invariant one-dimensional AQ image; none is in the current schedule.

If `N` is absent, forming it costs another `s/n` GEMM-equivalents per layer.  If a future D14 reset proves `N` is already its graded core, the remaining action still costs `s/n` per layer plus `O(nr^3)` local cube/vector words.  Over the ten-layer D10 suffix the unavoidable aggregate alone costs

\[
10s/n
\]

GEMM-equivalents: about `5.16` for `r=32`, `10.11` for `r=45`, and larger for the broader proposed cores.  This exceeds the single full-GEMM equivalent left by D14+D10 (`48/49`) even under the most favorable `N`-reuse premise.  Therefore the current alias/fusion loophole is closed by type and cost.

### 14.3 Exact old-span and counterterm ledger

Three distinct subtractions must not be conflated:

1. **Normal ordering / fourth cumulant:**
   \[
   \kappa_4(U,U,U,V)=E[U^3V]-3E[U^2]E[UV].
   \]
   The AQ formula already represents this cumulant; the reducible pair-pair term must not be added back.
2. **D11-reference 2PI counterterm:** the cubic self-energy lane uses
   \[
   \Delta\Sigma=\tfrac12\{V(G\otimes G)V^T-T_D(G\otimes G)T_D^T\}.
   \]
   This removes the already counted D11 cubic fish.  It is a separate source/consumer from the direct K31 cumulant and is not subtracted a second time inside the AQ trace.
3. **Equality/collision and old-span quotient:** apply the exact D11 external-collision mask to the complete AQ K31 action, then residualize its scalar/vector against the full old design span, including D11/Q24/CP3 and the counterterm-correct H3/fish consumers.

In symbols, if `C_D11` is the frozen equality/collision operator and `O` is the complete old feature matrix, the only creditable K31 feature is

\[
f_{31}^{new}=(I-O O^+)\,C_{D11} f_{31}^{AQ}.
\]

For the pure quadratic part,

\[
K31_{ij}^{AQ}
=\tfrac34t_{ii}^TK^+t_{ij}
+48\langle Z_{ii},Z_{ij}\rangle.
\]

The first term is the H3/fish-predictable `q_parallel` consumer and should receive zero new credit after the joint old-span projection.  The second is the q4-perpendicular slice.  The affine vector terms must be residualized jointly against H1/H2 old features rather than presumed independent.  Old-span residualization is cheap after the scalar action is formed, but it cannot remove the need to compute the raw `Q31`--AQ contraction.

### 14.4 Graph-ready nodes

#### `P2R0G19B_MATH_2PI_ORIENTATION_NOGO`

- `label`: H2 factor orientation no-go
- `kind`: scientific
- `status`: closed_proof
- `priority`: high
- `summary`: Weights plus H2 do not identify a reduction of the latent factor gauge `O(s^2)` to the physical left-right Kronecker gauge; spectral reshapes generate an unverified convention, and fixed templates collapse to low rank.
- `terminal_criterion`: Reopen only with a typed cross/fourth observable, an already carried tensor-product intertwiner, or a proved value-visible rank-one quotient.
- `evidence`: Sections 12 and 14.1.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_AQ_ANCHOR`

- `label`: Explicit affine--quadratic physical-gauge anchor
- `kind`: scientific
- `status`: solved
- `priority`: high
- `summary`: Explicit coefficients `(A_i,h_i)` in one Gaussian coordinate make the complete K31 action invariant under simultaneous latent orthogonal conjugation and remove the arbitrary bipartite reshape.
- `terminal_criterion`: Exact cumulant formula and physical-gauge invariance proved; deployment is delegated to the separate alias/cost node.
- `evidence`: Section 13.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_AQ_STAR_OBSTRUCTION`

- `label`: H2/H3 star-closure obstruction
- `kind`: scientific
- `status`: closed_proof
- `priority`: high
- `summary`: Moment H2/H3 determine only the projected AQ product.  The complete K31 action retains the Jordan-orthogonal fourth datum `R_AQ`; it vanishes only under proved AQ star-closure.
- `terminal_criterion`: Reopen a moment-only decoder only with `S star S subset S` or a direct fourth action observable.
- `evidence`: Section 13.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_AQ_ALIAS_COST`

- `label`: AQ N/R alias and fusion audit
- `kind`: deployment
- `status`: closed_cost
- `priority`: high
- `summary`: Current D14/D10 states are unit-pair words, not the unit-by-latent-pair `N`.  Even granted free `N`, exact `R=Q31N` costs `s/n` GEMM-equivalents per layer, exceeding the one-total-GEMM slack across D10, and it cannot share the existing conjugation for generic Hadamard-masked `Q31`.
- `terminal_criterion`: Reopen only with an explicit typed D14 identity `N=Phi(E)`, plus a paid identical Gram or exact fusion reducing the full ten-layer aggregate to at most one GEMM-equivalent.
- `evidence`: Sections 14.1--14.3 and the direct-diagram D10 architecture definition.
- `decomposition_children`: `[]`

Current next question: redirect effort from the closed two-word AQ carrier to the universal scalar K31 action only if the deployed engine already exposes its projected fourth action moment at Site A/B; that route avoids `N/R` entirely but is a new observable, not an H2/H3 reconstruction.

## 15. Direct Site-A/B fourth-action interface

### 15.1 Universal JVP and polarization identities

Let `X` be the actual centered pre-reset unit law, let the columns of `W` be `w_a`, and define

\[
U_a=w_a^TX,qquad V_a=p_a^TX,qquad P=W(Q^{31})^T,
\]

so that `V=Q31 U`.  For the actual-law diagonal fourth-cumulant potential

\[
\Phi_{true}(W)=\sum_a\kappa_4(w_a^TX,w_a^TX,w_a^TX,w_a^TX),
\]

cumulant multilinearity gives

\[
\boxed{
D\Phi_{true}(W)[P]
=4\sum_a\kappa_4(U_a,U_a,U_a,V_a)
=4\langle Q^{31},D^{31}_{true}\rangle_F.}
\]

Thus the desired action is exactly one quarter of a **local signal-mixing JVP**, with the source law held fixed.  A full-network derivative that also differentiates earlier resets is a different tangent and double-counts transport/frame effects.

If an actual-law potential can be called on arbitrary mixtures, no automatic differentiation is required.  Put

\[
d_1=\Phi(U+V)-\Phi(U-V),\qquad
d_2=\Phi(U+2V)-\Phi(U-2V).
\]

The odd part of the quartic polynomial yields

\[
\boxed{
\sum_a\kappa_4(U_a,U_a,U_a,V_a)=\frac{8d_1-d_2}{48}.}
\]

This needs four evaluations of an actual-law potential; stored diagonal values at `U` do not supply those calls.

The scalar is lawful and orientation-free.  It is invariant under an orthogonal change of the source coordinate when `X,W,P` transform covariantly, and under simultaneous neuron relabeling

\[
U\mapsto\Pi U,qquad V\mapsto\Pi V,qquad
Q\mapsto\Pi Q\Pi^T.
\]

The K31 equality sector is only permutation-natural under mixing of the output labels; a generic orthogonal output mixing rotates it into the other quartic partitions.  The deployed architecture requires the former symmetry, while the full tensor/JVP statement supplies source-basis orthogonal invariance.

### 15.2 What D11 actually exposes

The on-disk D11 lineage does materialize a **modeled** K31 plane.  In
`p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py`, `_connected_planes`
constructs `arena[9] = k31` from the Q24/Edgeworth pair moment `m31`, subtracts
the mean and `3 variance x covariance` terms, and zeros the diagonal.
`_reset_stars` then combines its off-diagonal entries with diagonal `k4` and
the CP4/star residual into the carried quartic state.

Consequently

\[
\chi_{asm}=\langle Q^{31},D^{31}_{Q24+CP4+star}\rangle_F
\]

is an existing, cheap contraction if the transient plane and reverse mask are co-located.  It is also exactly an old-span duplicate.  Differentiating the Q24 diagonal potential, the CP4 factor potential, or the star/pairing potential merely reconstructs their already assembled K31 summands.  A stored vector of diagonal `k4` values has no derivative at all unless its evaluator/computation graph is retained.

What is absent is the actual-law defect

\[
\boxed{
\rho_{31}
=\left\langle Q^{31},
C_{D11}(D^{31}_{true}-D^{31}_{asm})\right\rangle_F,}
\]

followed by the common old-span residualizer.  Site A has closure-produced
`m31/k31_target` and Site B has the assembled pre-reset planes; neither site
has samples from the exact joint law, an actual-law potential evaluator, or a
JVP callback.  The offline truth-plane branch is an experimental oracle, not
a deployed observable.

An AQ JVP supplies only the decomposition

\[
\rho_{31}
=\langle Q,C(D^{31}_{AQ}-D^{31}_{asm})\rangle
+\langle Q,C(D^{31}_{true}-D^{31}_{AQ})\rangle.
\]

The second term is the beyond-AQ defect and still needs an action oracle or a
valid endpoint error bound.  This is a fourth-cumulant transition defect, not
the cubic 2PI counterterm; subtracting the D11 fish again here would be a
duplicate subtraction.

### 15.3 Explicit same-state/different-action theorem

The missing information is not recoverable by rearranging the old summaries.
Let `U` be uniform on `{-2,-1,1,2}`, put

\[
c=\frac{E U^4}{E U^2}=\frac{17}{5},qquad h(U)=U^3-cU,
\]

and let `epsilon` be independent and symmetric.  Direct calculation gives

\[
E[Uh(U)]=E[U^2h(U)]=0,qquad E[U^3h(U)]=\frac{18}{5}.
\]

For

\[
V_\pm=\epsilon\pm\theta h(U),
\]

the two laws have the same marginal distributions, means, variances,
covariance, marginal K3/K4, cubic cross moments, and even K22 data.  Hence all
diagonal/lower CP3-Q24 inputs agree, and the deterministic Q24 closure returns
the same assembled K31 action.  Nevertheless

\[
E[U^3V_\pm]=\pm\frac{18}{5}\theta.
\]

Therefore any estimator measurable with respect to the present Site-A/B state
has worst-case absolute action error at least `18|theta|/5` on this pair, and
worst-case squared error at least `(18 theta/5)^2`.  In general, if `O_old` is
the closed span of retained moment functions, exact action identification is
equivalent to

\[
\sum_aU_a^3(Q U)_a\in O_{old}
\]

modulo almost-sure equality and the known covariance subtraction.  Otherwise
a signed-measure perturbation in the annihilator of `O_old` preserves every
carried expectation while changing the action.

For diagonal-q4 values alone the tensor statement is even simpler: a residual
`S in Sym^4(R^n)` may have `S_iiii=0` for all `i` but
`S_aaab != 0`.  Hence the diagonal values determine the K31 action only when
the requested `Q` has no off-diagonal support or an additional structural law
fixes the polarization.

### 15.4 Minimal interface dimension and exact cost ledger

For one fixed reverse adjoint `Q`, one new real scalar per active layer is
necessary and sufficient:

\[
\operatorname{K31ActionResidual}(l,Q_l)
=\frac14D\Phi_{true,l}(U_l)[Q_lU_l]
-\langle Q_l,D^{31}_{asm,l}\rangle_F,
\]

with the same centering, equality/collision mask, and pre/post-reset convention
as D11.  It can instead accumulate the Bellman-weighted suffix sum into one
scalar.  For a `k`-dimensional family of independent endpoint adjoints, the
minimal reusable interface dimension is the rank of the residual action map on
that family; generically it is `k`.  Supporting arbitrary off-diagonal `Q`
requires the full residual K31 covector/plane, `Theta(n^2)` state.

Costs split sharply:

- existing modeled action, once `Q` and `D31_asm` are co-located: `O(n^2)`
  elementwise work and one scalar accumulator, but zero new information/value;
- native actual-law scalar callback: one scalar state and negligible
  consumption cost, but acquisition cost is an **unpriced new interface**;
- four-call polarization: four actual-law potential calls and therefore not a
  reuse of the stored diagonal;
- oriented AQ JVP with base `N,h` and reverse `Q/H` already present:
  \[
  n^2s+O(n^2r+nr^3)
  \]
  arithmetic per layer, or `s/n+O(r/n+r^3/n^2)` full-GEMM equivalents.  At
  `n=1024,r=45` the audited census is about `1.23` per layer; absent `N` it is
  about `2.24`, and absent the reverse conjugation about `4.24`.  Every case
  exceeds the one-total-GEMM slack over D10.

For a generic nonlinear law, scalar output dimension supplies no arithmetic
upper bound: exact acquisition can require the unresolved four-leg law.  The
interface is deployable only if the engine natively exposes it at total added
cost at most one GEMM-equivalent over the whole suffix.  Merely naming the
scalar does not prove that cost.

### 15.5 Separate graph-ready node

#### `P2R0G19B_MATH_2PI_SITE_AB_TRUE_K31_ACTION`

- `label`: Native Site-A/B true K31 API absence
- `kind`: scientific
- `status`: closed_proof
- `priority`: highest
- `summary`: The actual-law K31 residual would be one lawful scalar JVP per
  fixed adjoint, but the deployed estimator contract and D11 lineage expose no
  such callback or callable actual-law potential.  The only present K31 action
  is the modeled Q24/CP4/star old-span duplicate.
- `terminal_criterion`: Closed by the static contract/provenance audit in
  Section 16.  Reopen only after a contest-API change or a concrete estimator-
  computable factorization from weights and participant-generated probes.
- `evidence`: Section 15; on-disk D11 `_connected_planes` and `_reset_stars`;
  the explicit `V_+ / V_-` information witness.
- `decomposition_children`: `[]`

This node is independent of the closed orientation, AQ-star, and AQ-alias
branches.  Those closures neither prove nor refute the existence of a native
actual-law scalar callback.

Current next question: inspect the white-box problem contract itself, not the
Q24 estimator, for any exact activation-law primitive that can evaluate this
single JVP at Site A/B.  If no such primitive exists, close the node as an
information interface absent from the deployed problem rather than pricing an
imaginary scalar oracle.

## 16. Static problem-contract and provenance audit

### 16.1 Participant-facing API: exhaustive field census

The installed challenge SDK and the pristine starter-kit contract agree on
the complete deployed interface:

\[
\operatorname{predict}(self,mlp,budget)\to \text{array}(depth,width).
\]

The frozen `MLP` dataclass contains only

\[
(width,depth,weights,seed,name),
\]

where `weights` is the ordered list of dense matrices, `seed` is participant-
facing reproducibility entropy, and `name` is a pure function of the seed.  The
subprocess runner serializes only `width,depth,weights,seed`.  `SetupContext`
contains width/depth/budget/API version, a scratch directory, the participant's
own submission directory, and a run seed.  It contains no per-MLP law object.

In particular, neither `MLP`, `SetupContext`, nor the runner payload contains:

- activation samples or ground-truth moments;
- a moment/cumulant callback;
- a diagonal-fourth-potential evaluator;
- a JVP/VJP/autodiff service;
- Site-A/B hooks or reverse-adjoint calls;
- a hidden dataset handle.

Ground truth is generated or loaded by the scoring layer outside `predict` and
is never passed to the estimator.  Participant numerical work occurs inside
the counted budget context.  Static shipped files may be loaded from the
submission directory, but they cannot contain private per-MLP truth and add no
new observation beyond a pretrained function of the supplied weights.

A static scan of the D11 estimator lineage likewise finds no
`ORACLE_FOURTH`, `mps_planes`, `mps_cov2`, true-M31 callback, or potential-JVP
entry point.  Its transient `m31/k31` objects are the analytic Q24 closure
objects already classified as old span.

### 16.2 `ORACLE_FOURTH` provenance: offline only

`ORACLE_FOURTH` occurs in the attribution instrument
`public_g19_attribution_v1/run_tensor_recursion_v4.py`, not in a deployable
estimator.  Before entering its layer loop that script reads local experiment
artifacts

\[
\texttt{mps_cov2_rank*.npz},\quad
\texttt{mps_planes_rank*.npz},\quad
\texttt{states_rank*.npz},
\]

plus the sealed observer-fit weight store.  When the environment switch is
set it assigns

\[
k4_l\leftarrow\texttt{mc_pre_k4}[l],\qquad
K22_l\leftarrow\texttt{Pl[K22_l]},\qquad
K31_l\leftarrow\texttt{Pl[K31_l]}.
\]

Those are banked Monte Carlo/truth-plane diagnostics.  The files are not
fields of `MLP`, are not serialized by the runner, and are absent from the
D11 package.  `ORACLE_FOURTH` is therefore a scientific intervention proving
capacity, not a lawful deployment source or latent callback.

### 16.3 Separate API-presence and mathematical-sufficiency nodes

#### `P2R0G19B_MATH_2PI_SITE_AB_TRUE_K31_API`

- `label`: Native true-K31 action API presence
- `kind`: deployment
- `status`: closed_proof
- `priority`: highest
- `summary`: The official estimator contract passes only weights, shape,
  budget, and reproducibility seeds.  No actual-law fourth potential/JVP/action
  callback exists; `ORACLE_FOURTH` reads offline truth artifacts.
- `terminal_criterion`: Satisfied by the SDK dataclass, runner-payload, D11
  package, and offline-oracle provenance census.  Reopen only on an official API
  change.
- `evidence`: Section 16.1--16.2; starter-kit estimator contract; installed
  `whestbench/domain.py`, `sdk.py`, and `runner.py`.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_SITE_AB_TRUE_K31_JVP_THEOREM`

- `label`: True diagonal-q4 JVP sufficiency theorem
- `kind`: scientific
- `status`: solved
- `priority`: high
- `summary`: If an actual-law potential evaluator existed, one quarter of its
  local signal-mixing JVP is exactly the fixed-adjoint K31 action; four-point
  polarization is an equivalent evaluator-only construction.  One scalar is
  necessary and sufficient for one fixed adjoint.
- `terminal_criterion`: The multilinear JVP/polarization identities and the
  same-state/different-action lower bound are proved.  This theorem makes no API
  existence claim.
- `evidence`: Section 15.1, 15.3, and 15.4.
- `decomposition_children`: `[]`

The earlier open action-interface status is reconciled to the first node and
closed.  Mathematical sufficiency remains solved separately; an absent API
does not invalidate the identity.

### 16.4 Next lawful source supplied by the contract

After excluding an evaluator callback, the contract supplies exactly one
route to new per-network law information beyond deterministic functions of the
weights: use the grader-provided `mlp.seed` to generate participant-side
standard-normal inputs and push them through the supplied weights and exact
ReLU.  These on-budget trajectories are lawful because the input law and
forward map are the problem definition.  They are not ground-truth access.

For `S>=4` exact trajectories, let `U_sa` be the centered signal at an active
layer, `V_s=Q U_s`, and let `m_pq,a` be sample central comoments with divisor
`S`.  The multivariate fourth k-statistic

\[
\widehat\kappa_{31,a}
=\frac{S^2}{(S-1)(S-2)(S-3)}
\left[(S+1)m_{31,a}-3(S-1)m_{20,a}m_{11,a}\right]
\]

is unbiased for the true repeated fourth cumulant.  Summing over `a` and
subtracting the deterministic assembled Q24 action gives a lawful estimator of
the missing scalar.  A samplewise analytic control variate may reduce its
eighth-moment variance, but the old action as a scalar expectation alone is
not such a control variate.

The exact primitive cost exposes the obstacle.  Generating an `S x n` signal
through one dense layer costs `S n^2`; applying one dense reverse `Q_l` to the
same samples costs another `S n^2`.  Exact trajectories from input through 14
layers plus actions on the ten-layer suffix therefore cost at least

\[
(14+10)S n^2=24(S/n)
\]

full-`n^3` GEMM equivalents, excluding lower-order random generation and
moment accumulation.  With only one full-GEMM equivalent left, `S<=floor(n/24)`
(`42` at `n=1024`).  The estimator variance is controlled by eighth moments
and scales only as `1/S`; no theorem currently places forty-two probes inside
the `3.80e-6` endpoint error allowance.  The campaign's prior naive-MC blend
receipt is consistent with this obstruction.

Thus the next lawful source is **contract-native seeded trajectory probing,
only with a very strong action-specific analytic control variate or a cheaper
target-specific factorization**.  Naive sampling is lawful but not promoted.
Any successor must prove both

\[
\text{total probe/action cost}\le 1\ \text{GEMM-equivalent}
\]

and an adjoint-weighted variance/error bound sufficient for the conditional
`2.86e-10` quartic gain.  Offline truth planes and participant-inaccessible
ground truth cannot appear in that proof.

Current next question: derive a zero-mean, weights-only Stein/control-variate
representation of the K31 action residual whose per-probe variance is small
enough at `S<=42`, or close trajectory acquisition by the eighth-moment noise
bound and redirect to a deterministic in-card source.

## 17. Contract-native trajectory escape: exact k-statistic, eighth-moment variance, and closure

This section resolves the question at the end of Section 16 without using any
numerical experiment.  The conclusion is deliberately scoped: seeded
trajectory probing is lawful, and an unbiased estimator exists, but the
present D11/AQ/control family supplies no variance certificate compatible with
the one-GEMM residual budget.  This closes the **current trajectory form**; it
is not a theorem that no future, explicitly constructed same-row Stein control
can work.

### 17.1 Projected defect and its unbiased fourth k-statistic

Fold the layer and unit labels into `j=(l,a)`.  For one exact Gaussian input
trajectory let `U_j` denote the true-centered signal and set

\[
V_{l}=Q_lU_l,
\qquad
x(Z)=\sum_j U_j^3V_j,
\qquad
A_j(Z)=U_j^2,
\qquad
B_j(Z)=U_jV_j.
\]

Writing `alpha=E A`, `beta=E B`, and `mu_x=E x`, the true projected repeated
fourth cumulant and the missing D11 defect are

\[
\chi_{31}=\mu_x-3\alpha^T\beta,
\qquad
\rho_{31}=\chi_{31}-\chi_{31}^{\rm asm}.
\]

Here `chi_31^asm` must use exactly the deployed Q24/collision/old-span
convention.  It is deterministic, so its subtraction changes the target but
not sampling variance.

For `S>=4` independent trajectories, define sample central comoments with
divisor `S`.  The standard repeated multivariate fourth k-statistic is

\[
\widehat\kappa_{31,j}=
\frac{S^2}{(S-1)(S-2)(S-3)}
\left[(S+1)m_{31,j}-3(S-1)m_{20,j}m_{11,j}\right].
\]

Consequently

\[
\widehat\rho_{31}=\sum_j\widehat\kappa_{31,j}
-\chi_{31}^{\rm asm}
\]

is unbiased, including when the true means are unknown.  This must be formed
from independent trajectories.  Antithetic or randomized-quasi-Monte-Carlo
rows are dependent; applying the ordinary row k-statistic to them is not
unbiased.  They require independent blocks and therefore cannot be credited as
`S` independent rows without a new block-kernel proof.

### 17.2 Exact finite-sample variance uses moments only through order eight

There is a concise exact variance formula that automatically retains all
cross-unit and cross-depth dependence.  For one `j`, put
`Y_1=Y_2=Y_3=U_j` and `Y_4=V_j`.  For a partition `pi` of `{1,2,3,4}`, let

\[
\mu(\pi)=(|\pi|-1)!(-1)^{|\pi|-1}.
\]

Define the symmetric four-row cumulant kernel by assigning the blocks of every
partition injectively to four rows and averaging over those assignments:

\[
H_j(z_1,z_2,z_3,z_4)=
\sum_{\pi\in\Pi_4}\mu(\pi)
\frac{1}{(4)_{|\pi|}}
\sum_{\iota:\pi\hookrightarrow[4]}
\prod_{C\in\pi}\prod_{r\in C}Y_{r,j}(z_{\iota(C)}),
\]

where `(4)_k=4!/(4-k)!`.  Let `H=sum_j H_j`.  Then `E H=chi_31`, and the
k-statistic is the order-four U-statistic with kernel `H`.  For `c=1,...,4`,
define the overlap covariance

\[
\zeta_c=\operatorname{Cov}\!\left(
H(Z_1,\ldots,Z_4),
H(Z_1,\ldots,Z_c,Z'_{c+1},\ldots,Z'_4)
\right).
\]

The exact finite-`S` variance is

\[
\boxed{
\operatorname{Var}(\widehat\rho_{31})=
\sum_{c=1}^4
\frac{\binom4c\binom{S-4}{4-c}}{\binom S4}\,\zeta_c .}
\]

Every product in a `zeta_c` has total degree at most eight in the joint
trajectory signals.  Thus this is an exact eighth-moment formula, not an
independence or Gaussian-output approximation.  It also shows why per-layer
variance estimates cannot be added: the same input drives every layer, so the
cross-depth eighth moments are part of `zeta_c`.

For a more explicit best-case diagnostic, suppose the true centering were
known.  The symmetric order-two kernel

\[
h(z,z')=\frac{x(z)+x(z')}{2}
-\frac32\left[A(z)^TB(z')+A(z')^TB(z)\right]
\]

also has expectation `chi_31`.  With `delta A=A-alpha` and
`delta B=B-beta`, its first Hoeffding influence and degenerate remainder are

\[
\psi=x-\mu_x-3\left[(A-\alpha)^T\beta
+\alpha^T(B-\beta)\right],
\]

\[
h_2(z,z')=-\frac32\left[
\delta A(z)^T\delta B(z')+\delta A(z')^T\delta B(z)
\right].
\]

Therefore, exactly,

\[
\operatorname{Var}(U_h)=
\frac{E\psi^2}{S}+\frac{2E h_2^2}{S(S-1)}.
\]

If `C_A=Cov(A)`, `C_B=Cov(B)`, and
`C_AB=E[(A-alpha)(B-beta)^T]`, independence of the two rows gives

\[
E h_2^2=\frac92\left[
\operatorname{tr}(C_AC_B)+\operatorname{tr}(C_{AB}^2)
\right],
\]

and hence

\[
\boxed{
\operatorname{Var}(U_h)=\frac{E\psi^2}{S}
+\frac9{S(S-1)}
\left[\operatorname{tr}(C_AC_B)+\operatorname{tr}(C_{AB}^2)\right].}
\]

Although `tr(C_AB^2)` need not be positive separately when `C_AB` is not
symmetric, the bracket is nonnegative because it is `(2/9)E h_2^2`.  If
`alpha,beta` were known **exactly** from a lawful source, one could subtract
`3 alpha^T beta` deterministically and sample only `x`.  D11 supplies modeled,
not exact, lower moments, so that simplification is not presently available.

### 17.3 Exact optimal-control theorem and why D11/AQ do not instantiate it

Let `G` be any vector of symmetric, mean-zero, four-row control kernels with
lawfully known expectation.  Define overlap matrices and vectors

\[
\Gamma^{(c)}=\operatorname{Cov}(G,G_c),
\qquad
\gamma^{(c)}=\operatorname{Cov}(G,H_c),
\]

using the same `c` shared rows as above, and let

\[
w_{S,c}=\frac{\binom4c\binom{S-4}{4-c}}{\binom S4},
\quad
\Gamma_S=\sum_cw_{S,c}\Gamma^{(c)},
\quad
\gamma_S=\sum_cw_{S,c}\gamma^{(c)},
\quad
v_S=\sum_cw_{S,c}\zeta_c.
\]

For `widehat chi-beta_C^T widehat G`, finite-sample quadratic minimization
gives

\[
\boxed{
\beta_C^*=\Gamma_S^+\gamma_S,
\qquad
v_{\min}=v_S-\gamma_S^T\Gamma_S^+\gamma_S.}
\]

At first Hoeffding order this reduces to projection of `psi` onto the
samplewise control span.  If `Gamma=Cov(C)`, `c=Cov(C,psi)`, then

\[
\sigma_{\rm eff}^2
=\operatorname{Var}(\psi)-c^T\Gamma^+c
=\operatorname{Var}(\psi)(1-R^2).
\]

This theorem separates three frequently conflated objects:

1. `chi_31^asm` is a deterministic target subtraction.  Its random control
   kernel is zero, so it cannot reduce variance.
2. D11/Q24 exposes no same-Gaussian-row surrogate trajectory.  An independently
   generated surrogate has zero covariance with the true-row kernel and hence
   optimal coefficient zero.
3. AQ would be a legitimate same-row control only if its oriented Gaussian
   core, exact expectation, and row coupling were explicitly constructed.
   Sections 13--14 show that this state is neither present nor free; treating
   its scalar expectation as a control repeats the error in item 1.

For a standard Gaussian input `Z`, a genuine weights-only Stein family can be
built from

\[
\mathcal T f=\nabla\!\cdot f-Z^Tf,
\qquad E\mathcal T f=0.
\]

There is a formal zero-variance solution: expand the centered influence in
Hermite chaoses, `psi=sum_{k>=1} psi_k`, solve the Ornstein--Uhlenbeck Poisson
equation with coefficients `-psi_k/k`, and use its Stein image.  For a deep
ReLU trajectory this requires the full activation-region/higher-chaos
solution; it is algebraically equivalent to acquiring the missing true-law
action rather than a cheap D11 observable.  A finite Stein span succeeds
exactly only when `psi` lies in that span.  Otherwise the orthogonal Hermite
remainder is precisely `sigma_eff^2`.  No current candidate supplies either
that span identity or an eighth-moment covariance certificate.

### 17.4 Arithmetic and accuracy lower bounds

For generic dense weights, producing one exact trajectory through fourteen
layers requires at least `14 n^2` arithmetic-scale work.  The fixed projected
action requires ten generic dense `Q_l U_l` matrix-vector products on layers
5--14, another `10 n^2`.  Therefore `S` independent rows cost at least

\[
C_{\rm probe}\ge24S n^2,
\qquad
\frac{C_{\rm probe}}{n^3}\ge\frac{24S}{n},
\]

before evaluating any nontrivial control.  A scalar quadratic contraction does
not evade the generic dense matrix-vector lower bound.  There is no certified
low-rank `Q_l` factor in this branch.  With at most one full-GEMM equivalent
left and `n=1024`,

\[
\boxed{S\le\left\lfloor\frac{1024}{24}\right\rfloor=42.}
\]

Any sample-dependent Stein/AQ control adds work and can only lower this cap.
Because every U-statistic Hoeffding component contributes nonnegative
variance, the first-order residual alone yields the necessary condition

\[
\operatorname{Var}(\widehat\rho_{31})
\ge\frac{\sigma_{\rm eff}^2}{S}.
\]

For the strict endpoint action allowance `epsilon=3.80e-6`, the most optimistic
`S=42` gate is

\[
\boxed{
\sigma_{\rm eff}^2\le42\epsilon^2=6.0648\times10^{-10},
\qquad
\sigma_{\rm eff}\le2.4627\times10^{-5}.}
\]

Even the weaker requirement that sampling noise not exceed the conditional
quartic raw value `delta=2.86e-10` requires

\[
\boxed{
\sigma_{\rm eff}^2\le42\delta=1.2012\times10^{-8},
\qquad
\sigma_{\rm eff}\le1.0960\times10^{-4}.}
\]

Both are necessary, not sufficient: the degenerate U-statistic terms and the
cost of the control tighten them.  More generally, a residual influence
variance `sigma_eff^2` implies

\[
S\ge\frac{\sigma_{\rm eff}^2}{\epsilon^2},
\qquad
\boxed{
\frac{C_{\rm probe}}{n^3}\ge
\frac{24}{n}\frac{\sigma_{\rm eff}^2}{\epsilon^2}.}
\]

Equivalently an optimal control must prove

\[
1-R^2\le
\frac{42\epsilon^2}{\operatorname{Var}(\psi)}
\]

and separately bound the degenerate remainder.  Neither a deterministic D11
subtraction, an uninstantiated AQ expectation, nor the formal OU inverse is
such a receipt.

### 17.5 Graph-ready reconciliation

#### `P2R0G19B_MATH_2PI_SITE_AB_TRUE_K31_API`

- `label`: Contract/API audit for true projected K31
- `kind`: scientific
- `status`: closed_proof
- `priority`: high
- `summary`: No deployed callback, true diagonal-fourth potential evaluator,
  JVP, or polarization API exists.  Offline `ORACLE_FOURTH` truth planes are
  evaluation artifacts and are participant-inaccessible.
- `terminal_criterion`: Reopen only on an exact contract path and provenance
  receipt for an actual-law evaluator.
- `evidence`: Sections 16.1--16.3.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_RHO31_KSTAT_ESTIMATOR`

- `label`: Unbiased seeded-trajectory rho31 k-statistic
- `kind`: scientific
- `status`: solved
- `priority`: medium
- `summary`: Independent contract-native Gaussian trajectories admit the exact
  repeated fourth k-statistic, summed against the fixed adjoint and reduced by
  the deterministic assembled action.
- `terminal_criterion`: The estimator is proved unbiased for unknown means;
  dependent-row variants require a separate block-kernel proof.
- `evidence`: Section 17.1.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_RHO31_EIGHTH_VARIANCE`

- `label`: Exact eighth-moment variance and optimal-CV theorem
- `kind`: scientific
- `status`: solved
- `priority`: high
- `summary`: The full finite-sample variance is the four overlap-covariance
  sum; the true-centered reduction gives an explicit influence plus degenerate
  formula.  The exact finite-S optimal control is a pseudoinverse projection.
- `terminal_criterion`: All cross-depth dependence and moments through order
  eight are retained; no output-independence approximation is used.
- `evidence`: Sections 17.2--17.3.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_RHO31_CONTROL_GATE`

- `label`: D11/AQ/Stein control availability gate
- `kind`: scientific
- `status`: closed_this_form
- `priority`: high
- `summary`: D11 is a deterministic subtraction, AQ has no lawful same-row
  oriented realization, and the formal zero-variance Stein solution requires
  the missing full ReLU chaos law.  No current control proves the required
  residual influence or degenerate-term bound.
- `terminal_criterion`: Reopen only with an explicit same-row weights-only
  zero-mean control, its known expectation, exact eighth-moment covariance
  receipt, and total cost within the remaining budget.
- `evidence`: Sections 17.3--17.4.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_2PI_RHO31_TRAJECTORY_COST`

- `label`: Seeded-trajectory K31 acquisition cost bound
- `kind`: scientific
- `status`: closed_cost
- `priority`: high
- `summary`: Fourteen forward matrix-vector layers plus ten dense Q actions
  cost at least `24 S n^2`, hence one-GEMM slack permits at most 42 independent
  rows.  The necessary residual standard deviations are `2.4627e-5` for the
  strict endpoint gate and `1.0960e-4` for the weaker raw-value gate.
- `terminal_criterion`: Reopen only with a certified target factorization that
  lowers the generic dense action cost or a control satisfying both variance
  gates with `S<=42` including its own billed work.
- `evidence`: Section 17.4.
- `decomposition_children`: `[]`

**Checkpoint-9 verdict:** the lawful estimator exists, but naive seeded
trajectory acquisition and the presently available D11/AQ/control family are
`NO-GO` under one-GEMM slack.  The next deterministic source must expose new
true-law information from weights while entering an already paid recurrent
word; otherwise this K31 defect branch is closed.

## 18. Checkpoint 10: seeded contraction of the transient Z22 Bellman scalar

### 18.0 Scope: this randomizes arithmetic, not the activation law

The transient plane

\[
Z_l=\operatorname{offdiag}\operatorname{sym}
\{K22_l^{\rm target}-K22_l^{\rm CP4}-\alpha_lP22(C_l)\}
\]

is already a deterministic D11 closure object.  So are
`W_l`, `X_l=W_l odot W_l`, and every compiled Bellman covector.  A seed can
randomize a trace or finite sum involving these objects, but it does **not**
sample the true activation law and cannot turn modeled `Z_l` into
`Z_l^true`.  This is categorically different from Section 17's trajectory
k-statistic:

- Section 17 randomizes Gaussian inputs to acquire an unknown true-law
  cumulant defect.
- This section randomizes contraction indices to approximate an already
  determined modeled Bellman scalar.

Accordingly a seeded Z22 contraction is not, by itself, a control variate for
`rho31`.  Independent index probes have zero covariance with the trajectory
influence, and reusing the same integer seed does not create a lawful
probabilistic coupling.  A samplewise true-trajectory functional with known
expectation would still be required for that purpose.

### 18.1 Exact K31 trace and the complete one-step term population

For the cheap K31 witness, absorb all literal `/6`, transpose, layer, and
endpoint coefficients into `gamma_l`, and put

\[
Y_l=W_l\odot P_l,
\qquad P_l=W_l\Lambda_{31,l}^{T}.
\]

The already derived action is

\[
\boxed{
\chi_{31,l}=\gamma_l\operatorname{tr}(X_l^TZ_lY_l)
=\sum_{f=1}^n d_{lf},
\qquad
d_{lf}=\gamma_l x_{lf}^{T}Z_ly_{lf}.}
\tag{18.1}
\]

Before the literal response normalization, `gamma_l=3`.  Equation (18.1) is
just the `n`-context Bellman sum; one selected column is one exact context,
not the complete scalar.

For completeness, a physical pure-22 lift also emits K4 and both K22 pieces.
Let

\[
q_a=w_a^{\odot2},\qquad c_{ab}=w_a\odot w_b,
\]

and let the displayed covectors be those before the literal Q24 denominators.
The complete one-step scalar can be represented as a finite population with
terms

\[
\begin{aligned}
a^{40}_a&=\frac{3}{24}\Lambda_{40,a}\,q_a^TZq_a,\\
a^{31}_{ab}&=\frac{3}{6}\bar\Lambda_{31,ab}\,q_a^TZc_{ab},\\
a^{22,u}_{ab}&=\frac{1}{4}\bar\Lambda_{22,ab}\,q_a^TZq_b,\\
a^{22,x}_{ab}&=\frac{2}{4}\bar\Lambda_{22,ab}\,c_{ab}^TZc_{ab}.
\end{aligned}
\tag{18.2}
\]

The K22 indices are off diagonal; its diagonal quotient is routed through
K4.  Both K31 orientations are absorbed in `bar Lambda31`.  Thus

\[
\chi_l^{\rm complete}=\sum_a a^{40}_a
+\sum_{a,b}\left(a^{31}_{ab}+a^{22,u}_{ab}+a^{22,x}_{ab}\right).
\tag{18.3}
\]

Every individual term in (18.2) can be evaluated with one dense `Z`-vector
product plus `O(n)` formation/reduction once the covector entry is available.
Therefore random term sampling bypasses the **exact** rank-980 crossed-K22
construction, but only by accepting sampling variance.  It does not make the
crossed branch dark or remove it from the physical action.

### 18.2 Unbiased column and complete-term estimators

For the collapsed K31 trace, sample `I_s` uniformly from `{1,...,n}`.  Then

\[
\boxed{
\widehat\chi_{31,l}=\frac{n}{S}\sum_{s=1}^S d_{lI_s},
\qquad E\widehat\chi_{31,l}=\chi_{31,l}.}
\tag{18.4}
\]

Sampling columns without replacement preserves unbiasedness.  More generally,
for positive probabilities `p_f`,

\[
\widehat\chi_{31,l}=\frac1S\sum_s\frac{d_{lI_s}}{p_{I_s}}
\tag{18.5}
\]

is the Horvitz--Thompson estimator with replacement.

The same construction applies to the complete action.  Fold all active
layers, collision families, and their indices into a finite set `T`, with
term values `a_t` from (18.2), and sample `t_s` with probability `p_t`:

\[
\boxed{
\widehat\chi^{\rm complete}
=\frac1S\sum_{s=1}^S\frac{a_{t_s}}{p_{t_s}},
\qquad
E\widehat\chi^{\rm complete}=\sum_{t\in\mathcal T}a_t.}
\tag{18.6}
\]

This is a contract-native estimator of the **modeled complete one-step
scalar**, conditional on having the complete covectors at the transient reset.
It is not an estimator of unmodeled 211/1111 content or of the true-minus-D11
law defect.

### 18.3 Exact finite-population and sparse-quadratic variance theorems

Let

\[
\bar d_l=\frac1n\sum_fd_{lf},
\qquad
D_l^2=\sum_f(d_{lf}-\bar d_l)^2.
\]

Uniform sampling with replacement in (18.4) has exact variance

\[
\operatorname{Var}(\widehat\chi_{31,l})=\frac{nD_l^2}{S}.
\tag{18.7}
\]

For a uniform subset of `S` distinct columns, the exact finite-population
correction is

\[
\boxed{
\operatorname{Var}(\widehat\chi_{31,l})
=\frac{n(n-S)}{S(n-1)}D_l^2.}
\tag{18.8}
\]

Thus `S=n` is zero variance and is simply the known exact one-product
contraction.  Randomization creates no improvement for one layer if that one
product already fits.

There is an exact interpolation between column sampling and Hutchinson.  Set

\[
A_l=\gamma_lX_l^TZ_lY_l,
\qquad H_l=\operatorname{sym}A_l,
\qquad \chi_{31,l}=\operatorname{tr}H_l.
\]

Choose a uniformly random size-`k` coordinate subset and independent signs,
and set each selected coordinate of `xi` to `sqrt(n/k)`.  Then

\[
\widehat\chi_{k}=\xi^TH_l\xi
=\gamma_l(X_l\xi)^TZ_l(Y_l\xi)
\tag{18.9}
\]

is unbiased.  Write

\[
D_H^2=\|\operatorname{diag}H_l-	frac{\operatorname{tr}H_l}{n}{\bf1}\|_2^2,
\qquad
O_H^2=\|\operatorname{offdiag}H_l\|_F^2.
\]

Conditioning first on the chosen subset and then on the signs gives

\[
\boxed{
v_k:=\operatorname{Var}(\widehat\chi_k)
=\frac{n(n-k)}{k(n-1)}D_H^2
+\frac{2n(k-1)}{k(n-1)}O_H^2.}
\tag{18.10}
\]

The endpoints are exact:

- `k=1`: uniform column sampling, variance `n D_H^2`;
- `k=n`: Rademacher Hutchinson, variance `2 O_H^2`.

With `X,Y` materialized, one probe needs one dense `Z` matvec and two
`k`-column accumulations, i.e.

\[
c(k)n^2=n^2+2kn+O(n),
\qquad c(k)=1+\frac{2k}{n}+O(n^{-1}).
\tag{18.11}
\]

Ignoring the harmless integer rounding, minimizing cost times variance within
this family has a closed form.  Put

\[
a_0=nD_H^2-2O_H^2,
\qquad b_0=2O_H^2-D_H^2.
\]

The cost-adjusted minimizer is `k=1` if `a_0<=0`, `k=n` if `b_0<=0`, and
otherwise

\[
k_* = \operatorname{clip}_{[1,n]}
\sqrt{\frac{na_0}{2b_0}}.
\tag{18.12}
\]

This proves there is no distribution-free preference between column sampling
and Hutchinson: a matrix with constant diagonal favors columns, while a
diagonal matrix favors dense signs.  A variance claim needs the two displayed
action-matrix energies.

For the complete term estimator (18.6), the exact with-replacement variance is

\[
\boxed{
\operatorname{Var}(\widehat\chi^{\rm complete})
=\frac1S\left(\sum_{t\in\mathcal T}\frac{a_t^2}{p_t}
-\left(\sum_ta_t\right)^2\right).}
\tag{18.13}
\]

Cauchy--Schwarz proves the oracle lower bound

\[
\sum_t\frac{a_t^2}{p_t}\ge\left(\sum_t|a_t|\right)^2,
\tag{18.14}
\]

with equality only for `p_t proportional to |a_t|`.  Hence even an oracle
importance sampler in this one-term family has

\[
\operatorname{Var}\ge
\frac{(\sum_t|a_t|)^2-(\sum_ta_t)^2}{S}.
\tag{18.15}
\]

The oracle probabilities require the very term census being avoided, so
(18.15) is an optimistic lower bound, not an implementable proposal.  It also
exposes the cancellation obstruction: a small signed Bellman scalar can have
large irreducible term-sampling variance.

Across layers, marginal variances may not be root-sum-squared when the same
probe is reused.  Shared sparse signs give the exact quadratic-form covariance
through `H_sum=sum_l H_l`; its variance is (18.10) with the diagonal/off-
diagonal Gram energies of `H_sum`, including every cross-layer inner product.
With independent, without-replacement column samples and allocations `S_l`,

\[
V_{\rm strat}=\sum_l
\frac{n(n-S_l)}{S_l(n-1)}D_l^2.
\tag{18.16}
\]

For a total `B=sum_l S_l` and an interior continuous optimum,
`S_l proportional to D_l`, and

\[
\boxed{
V_{\rm strat}^{*}
=\frac{n^2}{n-1}\frac{(\sum_lD_l)^2}{B}
-\frac{n}{n-1}\sum_lD_l^2.}
\tag{18.17}
\]

If an allocation hits `S_l=n`, the usual capped water-filling continuation is
exact.  Equation (18.17), rather than a sum of marginal RMS claims, is the
minimum statistic needed to certify a multi-layer seeded contraction.

### 18.4 Exact control-variate theorem and present availability

For any per-column control `g_f` whose total `G=sum_f g_f` is known exactly,
sampling the same columns gives

\[
\widehat\chi_{\beta}
=\frac{n}{S}\sum_{f\in\mathcal S}(d_f-\beta g_f)+\beta G,
\qquad E\widehat\chi_{\beta}=\chi
\]

under uniform sampling without replacement.  With replacement one instead
subtracts the zero-mean statistic `g_I/p_I-G`.  The exact optimal scalar
coefficient is

\[
\beta^*=\frac{\operatorname{Cov}_p(d_I/p_I,g_I/p_I-G)}
{\operatorname{Var}_p(g_I/p_I-G)},
\tag{18.18}
\]

with the Gram-pseudoinverse generalization for several controls.  Under
uniform sampling without replacement, (18.8) holds with `d_f-dbar` replaced
by its least-squares residual on the centered control vectors.

For sparse quadratic probes, a control matrix `C` with known trace has
zero-mean row statistic `xi^T C xi-tr C`.  The covariance metric is exactly

\[
\begin{split}
\langle H,C\rangle_k={}&
\frac{n(n-k)}{k(n-1)}
\langle\operatorname{diag}_0H,\operatorname{diag}_0C\rangle\\
&+\frac{2n(k-1)}{k(n-1)}
\langle\operatorname{offdiag}H,\operatorname{offdiag}C\rangle_F.
\end{split}
\tag{18.19}
\]

Projection in this metric gives the exact optimal coefficients and residual
variance.

The current state supplies no such additional control:

1. An old D11 action total without evaluable per-column terms or a quadratic
   matrix is only a constant and has zero control variance.
2. CP4 and pairing are already subtracted in `Z`.  Their source Frobenius
   orthogonality does not imply projection in either the population metric
   (18.8) or the Bellman quadratic metric (18.19).
3. Collapsing the K31 pair sum into `Y_l` is a genuine analytic
   Rao--Blackwellization, but it requires `P_l/Y_l` to exist.  There is no
   analogous constant-rank collapse for crossed K22; its exact diagonal-
   completion rank remains generically at least 980.
4. No same-index weights-only proxy with both a cheap known total and a proved
   residual correlation is exposed by the current API.

Thus the optimal-control theorem is solved, but the control-availability gate
is closed in the present form.

### 18.5 API/co-location and arithmetic gates

The contract permits deterministic probes derived from `mlp.seed`; no external
oracle is needed.  The deployed co-location premise is nevertheless missing:

- `Z_l` exists only transiently inside `_reset_stars` and is not returned;
- the future `Lambda31/Lambda22` context is not proved available at that point;
- `P_l=W_l Lambda31_l^T` and `Y_l=W_l odot P_l` are definitions, not free
  deployed buffers.

Deployment therefore needs a compiled reverse context, a proved D10 reuse, or
a reverse pass followed by reset recomputation.  The last option must bill the
recomputation and generally destroys the one-unit advantage.

Conditional on `Y_l` already existing, one K31 column costs one `Z_l y_lf`
matrix-vector product, so `S` columns cost

\[
C_l(S)=\frac{S}{n}quad\text{GEMM equivalents}+O(S/n^2).
\tag{18.20}
\]

All `n` columns batch to one exact `Z_lY_l` product: a single selected
transition is a conditional exact GO with zero sampling error.  If `P/Y` is
absent, each selected column additionally needs `W_l q_f`; the leading cost is
`2S/n`, and one-unit slack permits at most `S=512`.  Dense-sign probes are even
less attractive in that case because every selected `P` column must be
formed.

For several layers/families, each directly sampled term of (18.2) costs one
`Z` matvec once the raw covector entry is present.  One-GEMM slack therefore
permits at most

\[
\boxed{
\sum_l S_l\le n=1024}
\tag{18.21}
\]

such contexts, or `512` when one additional dense context action is required
per term.  This is materially more draws than Section 17's `S<=42`, but the
population now contains `Theta(Ln^2)` complete-action terms.  Sample count
alone is not a variance certificate.

Most importantly, a K31-only seed estimate remains a partial witness.  It
cannot receive complete-provider credit unless K4, uncrossed K22, and crossed
K22 are included by (18.6), or the already stated future-dark/low-rank gates
are proved.  Random sampling does not weaken the ownership requirement.

### 18.6 Endpoint and raw-gain gates

Let `sigma_p^2` be the exact one-draw variance in (18.13), after every lawful
control and old-arena subtraction.  Since `S<=n`, the strict endpoint error
allowance `epsilon=3.80e-6` requires

\[
\boxed{
\sigma_p\le\epsilon\sqrt n
=1.2160\times10^{-4}.}
\tag{18.22}
\]

Merely keeping expected algorithmic MSE below the conditional quartic value
`delta=2.86e-10` requires

\[
\boxed{
\sigma_p\le\sqrt{n\delta}
=5.4117\times10^{-4}.}
\tag{18.23}
\]

If context formation halves the cap to `S<=512`, these tighten respectively
to `8.5984e-5` and `3.8266e-4`.  Degenerate ownership error and incomplete
branches are biases and cannot be inserted into these variance allowances.

For one layer with `Y` absent and optimal `S=512` distinct-column sampling,
(18.8) gives the especially transparent necessary gates

\[
D_l\le\epsilon\sqrt{1023/1024},
\qquad
D_l\le\sqrt{\delta}\sqrt{1023/1024},
\tag{18.24}
\]

for the strict and weaker criteria.  No current receipt bounds `D_l`, the
sparse energies in (18.10), or the complete population norm in (18.13).

After residualizing across networks against the complete CP3/Q24/CP4/pair/
star/depth arena, let `s_Z` be the exact residual signal amplitude and let
`eta=sqrt(V_probe)` bound the seeded contraction error.  A robust positive raw
gate is

\[
\boxed{
s_Z\ge \eta+\sqrt{2.86\times10^{-10}}
=\eta+1.6912\times10^{-5}.}
\tag{18.25}
\]

Equivalently the realized feature must pass its own final Schur test

\[
\Delta_{\widehat Z\mid O}
=\frac{\langle e_O,\widehat z_Z^{\perp}\rangle^2}
{\|\widehat z_Z^{\perp}\|^2}
\ge2.86\times10^{-10}.
\tag{18.26}
\]

Unbiasedness over probe seeds does not imply (18.26) for the one fixed seed
used by the estimator.

### 18.7 Graph-ready node split

#### `P2R0G19B_MATH_Z22_SEEDED_API_COLOCATION`

- `label`: Contract-native Z22 seed and co-location audit
- `kind`: scientific
- `status`: conditional_open
- `priority`: high
- `summary`: The contract seed lawfully randomizes deterministic contraction
  indices, but transient Z22 and the future Bellman covectors/P/Y are not
  proved co-located in deployed D11.
- `terminal_criterion`: Promote only with a concrete buffer/provenance receipt
  or a fully billed reverse-and-recompute schedule.
- `evidence`: Sections 18.0 and 18.5.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_Z22_SEEDED_UNBIASED_ESTIMATOR`

- `label`: Unbiased seeded Z22 Bellman contraction
- `kind`: scientific
- `status`: solved
- `priority`: medium
- `summary`: Uniform/importance column sampling exactly estimates the collapsed
  K31 trace; Horvitz--Thompson term sampling exactly estimates the complete
  K4/K31/uncrossed-K22/crossed-K22 one-step scalar.
- `terminal_criterion`: The target and every sampled term must use the literal
  typed response coefficients; the theorem does not estimate true-law or
  211/1111 defects.
- `evidence`: Sections 18.1--18.2.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_Z22_SEEDED_VARIANCE_THEOREM`

- `label`: Exact finite-population and sparse-probe variance
- `kind`: scientific
- `status`: solved
- `priority`: high
- `summary`: Exact with/without-replacement, sparse-sign, cross-layer
  stratification, and oracle importance lower bounds are derived.  Column
  spread and action-matrix diagonal/off-diagonal energies are the sufficient
  statistics.
- `terminal_criterion`: A numeric promotion must provide these deterministic
  norms after the complete old-arena projection; marginal layer RMS is not
  enough.
- `evidence`: Section 18.3.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_Z22_SEEDED_CONTROL_AVAILABILITY`

- `label`: Z22 seeded control-variate availability
- `kind`: scientific
- `status`: closed_this_form
- `priority`: high
- `summary`: Optimal population and quadratic control projections are exact,
  but no current D11 object supplies a same-index evaluable control with a
  cheap known total.  Source CP4/pair orthogonality is not probe-metric
  orthogonality.
- `terminal_criterion`: Reopen with explicit control terms, exact totals,
  shared-probe cost, and a residual variance receipt satisfying Section 18.6.
- `evidence`: Section 18.4.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_Z22_SINGLE_TRANSITION_COST`

- `label`: Exact one-transition Z22 contraction cost
- `kind`: scientific
- `status`: conditional_go
- `priority`: medium
- `summary`: If Y/P is already exposed, all n K31 contexts batch into one GEMM
  and give the exact scalar with zero probe error.  Without that reuse, exact
  acquisition costs two and only an approximate half-census fits.
- `terminal_criterion`: Requires a co-location receipt and full old-arena
  residualization; it licenses only the K31 witness unless all typed branches
  are included.
- `evidence`: Section 18.5.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_Z22_SUFFIX_SAMPLING_COST`

- `label`: Multi-layer complete Z22 sampling cost
- `kind`: scientific
- `status`: closed_cost_current
- `priority`: high
- `summary`: One-GEMM slack buys at most 1024 direct term probes (512 with one
  missing context action) over a population of order `L n^2`; no present norm
  or control receipt certifies the resulting variance.
- `terminal_criterion`: Reopen only if equations (18.22)--(18.24) are certified
  with every control and covector acquisition included in the same unit.
- `evidence`: Sections 18.5--18.6.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_Z22_SEEDED_ENDPOINT_GAIN`

- `label`: Seeded Z22 endpoint/Schur promotion gate
- `kind`: scientific
- `status`: closed_this_form
- `priority`: high
- `summary`: Neither unbiasedness nor the 24x draw-count advantage proves raw
  value.  The realized old-arena-residualized feature must satisfy both the
  probe-error gate and a `2.86e-10` Schur gain; K31-only sampling cannot claim
  complete-provider value while crossed K22 remains visible.
- `terminal_criterion`: Reopen with `s_Z>=eta+1.6912e-5`, the exact realized
  Schur receipt, and either full typed term sampling or a crossed-K22 darkness/
  low-rank proof.
- `evidence`: Section 18.6.
- `decomposition_children`: `[]`

**Checkpoint-10 verdict:** seeded context/term sampling is a mathematically
valid and materially cheaper arithmetic approximation than true-trajectory
`rho31` acquisition.  It is an exact one-GEMM GO for one K31 transition only
when the reverse context is already exposed.  It is a scoped `NO-GO` for the
multi-layer or complete-provider promotion currently at issue: it acquires no
new true-law information, has no certified column/term variance or analytic
control, lacks a deployed co-location receipt, and does not erase the generic
crossed-K22 branch.

## 19. Deterministic AQ-H4 Fourier/surface acquisition audit

This section audits the exact fused score from the H4 reverse lane against the
literal deployed D11 state.  It also supplies a fully spectral truncation and
discretization certificate.  No numerical quadrature is assumed or run.

### 19.1 Fused score and diagonal spectral form

For

\[
g(z)=c+b^Tz+\frac12(z^TMz-\operatorname{tr}M),
\qquad z\sim N(0,I_r),
\]

the fused Hermite-H4 action is

\[
\mathcal A_H=E[g_+(z)\langle\operatorname{He}_4(z),H\rangle].
\]

The exact surface identity is

\[
\mathcal A_H=E\left[
\delta''(g)H[p,p,p,p]+6\delta'(g)H[M,p,p]+3\delta(g)H[M,M]
\right],
\qquad p=b+Mz.
\tag{19.1}
\]

Let

\[
S_t=(I-itM)^{-1},\quad m_t=S_tb,\quad C_t=MS_tM,
\]

and

\[
\chi(t)=e^{it(c-\operatorname{tr}M/2)}
\det(I-itM)^{-1/2}
\exp\left(-\frac{t^2}{2}b^TS_tb\right).
\tag{19.2}
\]

Then

\[
\boxed{
\begin{aligned}
\mathcal A_H={1\over2\pi}\int_{\mathbb R}\chi(t)\{&(it)^2[
H[m_t^4]+6H[C_t,m_t,m_t]+3H[C_t,C_t]]\\
&+6it[H[M,m_t,m_t]+H[M,C_t]]+3H[M,M]\}\,dt .
\end{aligned}}
\tag{19.3}
\]

Diagonalize `M=O diag(lambda) O^T` and write `beta=O^T b`.  In this basis,

\[
(m_t)_j={\beta_j\over1-it\lambda_j},
\qquad
(C_t)_{jj}={\lambda_j^2\over1-it\lambda_j},
\tag{19.4}
\]

and

\[
\boxed{
|\chi(t)|=
\prod_j(1+t^2\lambda_j^2)^{-1/4}
\exp\left[-{t^2\over2}
\sum_j{\beta_j^2\over1+t^2\lambda_j^2}\right].}
\tag{19.5}
\]

For `H=Sym(U^3 tensor V)`, only the rotated coordinates of `b,U,V`, the
eigenvalues, and four scalar resolvent sums are needed at a node.  Thus a
factored fused node is `O(r)` per mixed direction; a square/Jordan rank-`k`
costate is `O(kr)` after its one-time basis rotations.  No four-index H4 state
is needed.

### 19.2 Static deployed-state audit: D11 has no generalized-chi-square cache

The literal deployed D11 state is documented and implemented as physical
mean/covariance, CP3 factors, a cubic coordinate star, CP4 factors, a quartic
coordinate star, and one pairing coefficient.  Its frozen Hermite roots are
the univariate Q38/Edgeworth machinery; they are not nodes for (19.3).  The
only `eigh` in the estimator is the exceptional near-dead physical covariance
repair.  It diagonalizes an `n by n` correlation plane, discards the
eigenvectors, and is bypassed in the ordinary branch.

There is no deployed field or returned buffer for any of

\[
(c_i,b_i,M_i),\quad O_i,\quad \lambda_i,\quad O_i^Tb_i,
\quad \det(I-itM_i),\quad (I-itM_i)^{-1},
\tag{19.6}
\]

nor for AQ directions/costates rotated into `O_i`.  Therefore the answer to
the concrete reuse question is **no**: the current D11 cannot evaluate the
fused score from resident generalized-chi-square characteristic data, because
that characteristic data is not resident.

This is not merely a naming mismatch.  Expanding the logarithm of (19.2)
contains the infinite sequence

\[
\{\operatorname{tr}M^q, b^TM^qb:q\ge0\}.
\tag{19.7}
\]

Any fixed finite collection of low moment/collision summaries leaves spectra
with the same retained power sums but different next power sum when the latent
rank exceeds the number of constraints.  Their characteristic functions, and
generically their fused H4 actions, differ.  Orientation relative to `U,V`
adds a second missing Gram.  Hence a finite-moment D11 receipt cannot be
silently relabeled as the AQ spectral oracle; an explicit AQ factor-state
construction is a new provider.

### 19.3 Certified spectral truncation bounds

Let `P_H(t)` denote the braces in (19.3).  Tensor Cauchy--Schwarz gives the
pointwise envelope

\[
\begin{aligned}
|P_H(t)|\le\|H\|_{HS}\{&t^2[
\|m_t\|^4+6\|C_t\|_F\|m_t\|^2+3\|C_t\|_F^2]\\
&+6|t|\|M\|_F[\|m_t\|^2+\|C_t\|_F]
+3\|M\|_F^2\}.
\end{aligned}
\tag{19.8}
\]

Consequently the conjugate-symmetric cutoff error is always bounded, whenever
the right side is integrable, by

\[
\boxed{
\epsilon_{tail}(T)\le{1\over\pi}
\int_T^\infty |\chi(t)|\,B_H(t)\,dt,}
\tag{19.9}
\]

where `B_H` is the right side of (19.8).  All quantities in (19.9) are
deterministic functions of the spectrum and rotated vector.

There are two useful closed forms.

#### Nonzero quadratic support, with no linear null component

Suppose `M` has rank `s`, every nonzero eigenvalue obeys
`|lambda_j|>=lambda_*`, and the null-space projection of `b` is zero.  Put

\[
a={\|b\|\over\lambda_*},\qquad q={s\over2},
\qquad m_F=\|M\|_F.
\]

For `t>0`,

\[
|\chi(t)|\le(t\lambda_*)^{-s/2},\quad
\|m_t\|\le a/t,\quad \|C_t\|_F\le m_F/t.
\tag{19.10}
\]

Inserting these into (19.8) yields, for `s>2`,

\[
\boxed{
\begin{aligned}
\epsilon_{tail}(T)\le
{\|H\|_{HS}\lambda_*^{-q}\over\pi}
\bigg[&{12m_F^2\over q-1}T^{1-q}
+{12m_Fa^2\over q}T^{-q}\\
&+{a^4\over q+1}T^{-q-1}\bigg].
\end{aligned}}
\tag{19.11}
\]

This is a completely explicit absolute tail certificate.  If `s<=2` and
`b` has no null component, the determinant envelope alone does not make the
generic constant term in (19.8) absolutely integrable.  The Fourier identity
still exists distributionally or conditionally, but no absolute finite-`T`
certificate follows without a candidate-specific cancellation or a charged
mollifier.

#### A nonzero linear component in the quadratic null space

Let `b_0=P_{ker M}b`, `q_0=||b_0||^2/2>0`,
`B=||b||`, `C_0=||M^2||_F`, and `m_F=||M||_F`.  Then

\[
|\chi(t)|\le e^{-q_0t^2},\qquad
\|m_t\|\le B,qquad \|C_t\|_F\le C_0.
\]

Define

\[
A_2=B^4+6C_0B^2+3C_0^2,quad
A_1=6m_F(B^2+C_0),quad A_0=3m_F^2.
\]

Then

\[
\boxed{
\epsilon_{tail}(T)\le{\|H\|_{HS}\over2\pi}
\sum_{p=0}^2 A_p q_0^{-(p+1)/2}
\Gamma\left({p+1\over2},q_0T^2\right),}
\tag{19.12}
\]

where `A_p` is paired with `t^p`.  This Gaussian certificate includes the
ordinary affine case `M=0,b!=0`.

### 19.4 Certified discretization on the retained interval

The absence of real resolvent poles makes a direct deterministic bound easy.
Let `L=||M||_2`, `B=||b||`, `C_0=||M^2||_F`, and
`c_0=c-tr(M)/2`.  On `|t|<=T`,

\[
\|m_t^{(j)}\|\le j!L^jB,qquad
\|C_t^{(j)}\|_F\le j!L^jC_0.
\tag{19.13}
\]

For `ell=log chi`, valid on the branch continuous from zero, define

\[
\begin{aligned}
L_1(T)&=|c_0|+\tfrac12\sum_j|\lambda_j|
+TB^2+\tfrac12T^2LB^2,\\
L_2(T)&=\tfrac12\sum_j\lambda_j^2
+B^2(1+2TL+T^2L^2).
\end{aligned}
\tag{19.14}
\]

Then `|chi'|<=L_1` and `|chi''|<=L_1^2+L_2`.  Let
`P_j(T)` be the explicit product-rule envelope for
`sup_{|t|<=T}|P_H^(j)(t)|`, obtained from (19.8) and the derivative bounds
(19.13), for `j=0,1,2`.  No unknown probability quantity enters `P_j`.
For

\[
F(t)={\chi(t)P_H(t)\over2\pi},
\]

one has

\[
\sup_{[-T,T]}|F''|le{1\over2\pi}
\{(L_1^2+L_2)P_0+2L_1P_1+P_2\}.
\tag{19.15}
\]

Therefore a composite trapezoid rule with spacing `h=2T/K` obeys the fully
certified discretization bound

\[
\boxed{
\epsilon_{disc}(T,K)\le{Th^2\over12\pi}
\{(L_1^2+L_2)P_0+2L_1P_1+P_2\}.}
\tag{19.16}
\]

This bound is conservative but executable from the spectrum alone.  A sharper
exponential alternative follows from the analytic strip.  Every singularity
of the resolvent is at imaginary distance at least `1/L`.  For any
`0<rho<1/L` for which

\[
N_\rho=\int_{\mathbb R}
(|F(t+i\rho)|+|F(t-i\rho)|)dt<\infty,
\]

contour shifting gives the infinite-trapezoid alias bound

\[
\boxed{
\epsilon_{alias}(h)\le
{N_\rho\over e^{2\pi\rho/h}-1}.}
\tag{19.17}
\]

The finite-node rule adds the spectral tail (19.9) for the omitted nodes.
Equations (19.11)/(19.12) plus either (19.16) or (19.17) are the required
truncation-and-discretization receipt; quoting a node count without them is
not a certificate.

### 19.5 No universal finite-node exact rule

There is no fixed finite real-node Fourier rule that evaluates (19.3) exactly
for every AQ parameter tuple.  It already fails on the affine subfamily
`M=0,b!=0`.  Varying `c`, the exact nonzero surface score is a Gaussian density
times a polynomial in `c`.  A `K`-node Fourier rule produces instead a finite
exponential polynomial

\[
\sum_{j=1}^K a_j e^{it_jc}.
\]

A nonzero finite real-frequency exponential polynomial cannot decay at both
`c -> +infinity` and `c -> -infinity`, whereas the Gaussian-polynomial score
does.  Hence equality for every `c` is impossible.  Special cases with their
own closed form are not excluded, but a generic implementation must carry a
nonzero certified quadrature error.  In the low-rank non-absolutely-integrable
case it must additionally carry regularization bias; finite-node exactness
cannot be asserted by distributional Fourier notation.

### 19.6 State and operation census

For `n` physical units, latent rank `r`, `k` structured mixed directions, `K`
certified nodes, and `D` active layers, the minimal local data are

\[
\begin{array}{c|c}
\text{object}&\text{state}\\ \hline
(c_i,b_i,M_i)&O(nr^2)\\
(O_i,\lambda_i,O_i^Tb_i)&O(nr^2)\text{ if cached},\\
\text{rotated mixed directions/costates}&O(nkr)\text{ or streamed }O(kr),\\
\text{node scratch/output}&O(K+kr)\text{ streamed}.
\end{array}
\tag{19.18}
\]

The leading arithmetic is

\[
\boxed{
C_{AQ}={D\over n^2}
\{c_{eig}r^3+c_{rot}kr^2+c_{node}Kkr\}
+C_{AQstate}+C_{module},}
\tag{19.19}
\]

in full-`n^3` GEMM equivalents.  `C_AQstate` is the presently unpriced cost of
constructing the AQ state; `C_module` is any new physical reverse stream.  The
node term is local arithmetic and needs no physical `n by n` product once the
state, eigenbasis, and directions truly exist.

Thus “no new dense products” is valid **if and only if** all of the following
reuse premises hold:

1. exact AQ `(c,b,M)` is already resident at the same reset;
2. its eigensystem is resident, or the `r^3` term is billed inside the cap;
3. the complete-old-residualized `H/U/V` costate is a literal view of the
   resident reverse module and is available in the same AQ frame;
4. the fused result is consumed locally, with no additional physical
   propagation word.

Sharing only `W`, covariance, a CP rank, or univariate Hermite roots satisfies
none of these premises.  For the current D11, `C_AQstate` and the eigensystem
are new, so zero-dense-product reuse is closed.  Even ignoring state
construction, unreused eigensystems contribute about `0.87` equivalent for
`r=45,D=10` and `1.59` for `r=55,D=10`; the latter already exceeds the one-
unit slack.  With eigensystems and AQ state genuinely reused, the sharp
leading rank/node gate is

\[
\boxed{
Dk(r^2+Kr)\le n^2}
\tag{19.20}
\]

before constants and module work.

### 19.7 Comparison with the seeded arithmetic trace estimator

The two deterministic routes solve different acquisition problems.

| Property | AQ Fourier/surface score | Z22 seeded trace |
|---|---|---|
| Target | New AQ-model Hermite-H4 birth | Already modeled deterministic Z22 Bellman action |
| New law information | Yes, conditional on exact AQ state | No |
| Error | AQ model + quadrature + module leakage | Sampling variance, or zero for one exact product |
| Natural dimension | latent `r` | physical width `n` |
| Best conditional cost | `Dk(r^2+Kr)/n^2` with full reuse | one GEMM for one exact K31 transition |
| Current resident state | absent | Z22 exists transiently; reverse context unproved |
| Completeness issue | future cyclic-module closure | crossed-K22/K4 ownership |

The Fourier oracle can beat the one-product trace economically when `r<<n`,
the AQ eigensystem and reverse directions are truly reused, and a small `K`
satisfies the spectral error bound.  The trace route is preferable when its
one exact physical product already fits: it has neither AQ model error nor
quadrature error.  Neither is automatically a control variate for the other,
because their targets and randomizations differ.

### 19.8 Exact GO condition and graph split

An AQ Fourier provider is a deployment GO exactly when

\[
\boxed{
\begin{gathered}
C_{AQ}\le1,\\
\sum_{l=5}^{14}|\beta_l|
\left[\sqrt{24}\|e_l\|_2\|H_l\|_{HS}
+\epsilon_{tail,l}+\epsilon_{disc,l}
+\epsilon_{module,l}\right]
\le3.80\times10^{-6},\\
L_l^*\mathcal S_{l+1}\subseteq\mathcal S_l
\text{ at every active cut},\\
(I-P_O)\mathcal A_H\ne0
\text{ and its realized Schur gain meets the raw gate}.
\end{gathered}}
\tag{19.21}
\]

#### `P2R0G19B_MATH_AQ_H4_D11_CHARACTERISTIC_REUSE`

- `label`: D11 generalized-chi-square characteristic reuse audit
- `kind`: scientific
- `status`: closed_proof_current
- `priority`: high
- `summary`: Deployed D11 contains no AQ `(c,b,M)`, AQ eigensystem, resolvent,
  determinant, or rotated fused-score costate.  Its univariate Hermite roots
  and exceptional physical covariance repair are not reusable characteristic
  data.
- `terminal_criterion`: Reopen only with literal buffer/provenance receipts for
  all objects in (19.6), not a moment-matching inference.
- `evidence`: Section 19.2.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_AQ_H4_SPECTRAL_TAIL_CERTIFICATE`

- `label`: AQ-H4 spectral Fourier tail theorem
- `kind`: scientific
- `status`: solved
- `priority`: high
- `summary`: Equations (19.9)--(19.12) give absolute algebraic or Gaussian
  cutoff bounds from the nonzero spectrum and the linear null component.
- `terminal_criterion`: Low-rank cases lacking absolute decay require a
  candidate-specific cancellation or charged regularization bias.
- `evidence`: Section 19.3.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_AQ_H4_DISCRETIZATION_CERTIFICATE`

- `label`: AQ-H4 finite-node discretization theorem
- `kind`: scientific
- `status`: solved
- `priority`: high
- `summary`: Spectral derivative envelopes yield the certified composite-
  trapezoid bound (19.16); analytic-strip distance yields the optional
  exponential alias bound (19.17).
- `terminal_criterion`: A deployed node count must instantiate one bound and
  combine it with the tail; node count alone is not evidence.
- `evidence`: Section 19.4.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_AQ_H4_FINITE_NODE_EXACTNESS_NOGO`

- `label`: No universal exact finite Fourier rule
- `kind`: scientific
- `status`: closed_proof
- `priority`: medium
- `summary`: The affine AQ subfamily already separates Gaussian-polynomial
  surface scores from every finite real-frequency exponential polynomial.
- `terminal_criterion`: Special closed-form strata remain allowed; generic AQ
  quadrature must carry certified nonzero error.
- `evidence`: Section 19.5.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_AQ_H4_STATE_OPERATION_CENSUS`

- `label`: AQ-H4 state and marginal operation census
- `kind`: scientific
- `status`: solved_conditional
- `priority`: high
- `summary`: With genuine state/eigen/costate reuse, local nodes require no
  physical GEMM and obey `D k(r^2+Kr)<=n^2`; current D11 must additionally
  acquire the missing AQ state and eigensystems.
- `terminal_criterion`: All four reuse premises in Section 19.6 and the full
  billed constants must be documented.
- `evidence`: Sections 19.6--19.7.
- `decomposition_children`: `[]`

#### `P2R0G19B_MATH_AQ_H4_DEPLOYMENT_GATE`

- `label`: AQ-H4 Fourier provider deployment gate
- `kind`: scientific
- `status`: closed_current_state
- `priority`: high
- `summary`: The mathematical oracle and error formulas are exact, but current
  D11 lacks its state/reuse interface.  Model, quadrature, module, cost, old-
  span novelty, and Schur gates must all pass (19.21).
- `terminal_criterion`: Reopen only with an augmented lawful AQ-state provider
  whose instantiated census and certified total error satisfy (19.21).
- `evidence`: Sections 19.2 and 19.8.
- `decomposition_children`: `[]`

**AQ-H4 verdict:** the surface oracle is a mathematical conditional GO and a
potential latent-dimension economic win, but it is a deployment `NO-GO` as a
zero-cost extension of the present D11.  The claimed characteristic cache is
absent, no universal finite node rule is exact, and the missing AQ state plus
spectral/cyclic receipts must be supplied before its local `O(Kr)` advantage
is billable.

# All-distinct / deploy-frame feedback

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## Checkpoints 1--3: squarefree cubic provenance, exact DTOT2 duplicate, and co-adapted tangent

### Squarefree cubic source and deployable arithmetic circuit

For the leading all-distinct ReLU source, put

\[
g_i=\Phi(m_i/\sigma_i),\qquad
\beta_i=\varphi(m_i/\sigma_i)/\sigma_i,\qquad
c_{ij}=g_jC_{ij},\quad c_{ii}=0.
\]

Then the symmetric squarefree cubic tensor is

\[
S_{ijk}=\beta_i c_{ij}c_{ik}+\beta_jc_{ji}c_{jk}+\beta_kc_{ki}c_{kj},
\]

and its polynomial has the exact implicit representation

\[
p_S(x)=3\sum_i\beta_i x_i\left[(c_i^Tx)^2-\sum_jc_{ij}^2x_j^2\right].
\]

This is an arithmetic circuit of quadratic state size and cubic-or-better dense work; it is not a proof of new physics. A generic oblivious tensor sketch needs roughly 146--147 plane equivalents before its worst-case correlation can exceed 0.925, so a small generic sketch is not a viable carrier.

### Exact duplicate theorem for DTOT2

Let \(R(T)_{ij}=T_{iij}\), let \(L\) be the canonical repeated-plane lift

\[
p_{L(M)}(x)=\sum_iM_{ii}x_i^3+3\sum_{i\ne j}M_{ij}x_i^2x_j,
\]

and let \(Q=I-LR\). For the exact third-chaos recurrence

\[
T_{\ell+1}=\mathcal A_\ell T_\ell+S_{2,\ell},
\]

the split \(M=RT\), \(H=QT\) obeys

\[
\binom{M^+}{H^+}=
\begin{bmatrix}
R\mathcal AL&R\mathcal AQ\\
Q\mathcal AL&Q\mathcal AQ
\end{bmatrix}
\binom{M}{H}+\binom RQ S_2.
\]

Therefore the smallest term missing from a plane-only recurrence over two steps is exactly

\[
E_2(\Delta)=R\mathcal A_{s+1}Q\mathcal A_sL\Delta.
\]

The graph duplicate gate is formula-level, not score-level:

\[
R\mathcal A_{s+1}\mathcal A_sL
\stackrel{?}{=}
R\mathcal A_{s+1}LR\mathcal A_sL.
\]

The provenance circuit is a representation of the same source/transport physics. If the full DTOT2/tensor-memory engine retains \(Q\mathcal A L\), it already contains this mechanism; only a cheaper exact realization could be novel.

### Coupled deploy-frame tangent and compact-mask reconciliation

With \(B=W^T\), \(G=\operatorname{diag}(g)\), and \(A=BG\), write schematically

\[
m^+=B[\mu+\mathcal E_\mu R(T)],\qquad
C^+=B[K_0+\mathcal E_C R(T)]B^T,
\]

\[
T^+=A^{\otimes3}T+B^{\otimes3}S_2.
\]

The exact tangent contains

\[
\delta T^+=A^{\otimes3}\delta T+
\sum_{r=1}^3A^{\otimes(r-1)}\otimes\delta A\otimes A^{\otimes(3-r)}T
+B^{\otimes3}DS_2[\delta\theta],
\]

in addition to the direct consumer terms in \(\delta m^+\) and \(\delta C^+\). Gate derivatives are

\[
\delta g_r=g_{r+1}\delta m+\tfrac12g_{r+2}\delta v,
\]

and, for a two-site Gaussian gate moment,

\[
\delta G_{pq}=G_{p+1,q}\delta m_i+G_{p,q+1}\delta m_j
+\tfrac12G_{p+2,q}\delta v_i+\tfrac12G_{p,q+2}\delta v_j
+G_{p+1,q+1}\delta C_{ij}.
\]

For the squarefree source,

\[
\delta c_{ij}=C_{ij}\delta g_j+g_j\delta C_{ij},
\]

\[
\delta S_{ijk}=\delta\beta_i c_{ij}c_{ik}
+\beta_i\delta c_{ij}c_{ik}+\beta_ic_{ij}\delta c_{ik}+\mathrm{cyclic}.
\]

Thus the block Jacobian is

\[
J=\begin{bmatrix}J_{\theta\theta}&J_{\theta T}\\J_{T\theta}&J_{TT}\end{bmatrix}.
\]

The compact deploy-frame mask formula

\[
\epsilon_2^+=W^T\bigl[G_{11}\odot\epsilon_2
+\delta G_{11}\odot\epsilon_1+I(G,k_2)+\delta I(\delta G,k_1)\bigr]W
\]

is exactly the consumer half of this tangent. The \(\delta A\)-on-\(T\) and \(DS_2\) terms generate \(k_2\); they are absent only when the higher-chaos carrier is truth-fixed rather than deploy-frame generated. The minimal genuinely co-adapted loop is \(J_{\theta T,\ell+1}J_{T\theta,\ell}\).

## Checkpoint 4: signed Schur suppression and independent quartic cone

For

\[
J=\begin{bmatrix}A&B\\C&D\end{bmatrix},
\qquad
J_{\mathrm{eff}}(z)=A+B(zI-D)^{-1}C,
\]

and cubic source norms \(c_\theta,c_T\), a sufficient resolvent bound at \(z=1\) is

\[
C_3\le
\frac{c_\theta+bc_T/(1-\delta)}
{1-a-bc/(1-\delta)}.
\]

More generally, \(C_3\le\nu/\operatorname{dist}(1,W(J_{\rm eff}))\). At pre-Taylor source alignment \(q_s=.78\), \(d^3=(1-q_s^2)^{3/2}=.24506\); with exact other channels and zero remainder the raw target requires \(C_3\le .8466\). For unit normalized numerator it is sufficient that

\[
\lambda_{\max}(\operatorname{Herm}J_{\rm eff})<-.1812.
\]

A positive monotone self-energy \(C=B^*\), \(D\succeq0\) cannot do this; anti-aligned feedback is necessary. With finite-depth poles \(D=\sum_r\delta_rP_r\),

\[
R_{D,L}=\sum_{j=0}^{L-1}D^j=\sum_rh_L(\delta_r)P_r,
\qquad h_L(\delta)=\frac{1-\delta^L}{{1-\delta}},
\]

and the signed Loewner pass interval is

\[
(\nu/.8466-1)I\prec K_L-\operatorname{Herm}A\prec I.
\]

For the NCQC/D11 subtraction, every pole must use

\[
\Delta\Sigma_L=\tfrac12\sum_rh_r
\left[VP_rV^*-T_DP_rT_D^*\right].
\]

Writing \(V=T_D+U\), both cross terms \(UP_rT_D^*+T_DP_rU^*\) are mandatory. Omitting them or separately adding the same H3/q4-parallel Gram double-counts the cubic fish.

Whiten \(X=T_DR_L^{1/2}\), \(U=(V-T_D)R_L^{1/2}\). Then

\[
F_D=\tfrac12XX^*,\qquad
\Delta F=\tfrac12(XU^*+UX^*+UU^*),\qquad
F_D+\Delta F\succeq0.
\]

On the task subspace \(S\), C3 suppression asks

\[
P_S\Delta FP_S\preceq-P_S(\operatorname{Herm}A+\tau I)P_S,
\qquad \tau=\nu/.8466-1.
\]

The relaxed PSD headroom condition

\[
P_SF_DP_S\succeq P_S(\operatorname{Herm}A+\tau I)P_S
\]

is necessary and sufficient before off-subspace stability constraints; equality can be realized by \(U=-X\) on \(S\).

For the orthogonal quartic sector let \(P\) project in the trace metric onto \(\mathcal V=\operatorname{span}\{A_i\}\), put \(Y_{ij}=A_i\circ A_j\), \(R_{ij}=(I-P)Y_{ij}\), and

\[
Q^\perp_{ijkl}=16\left[
\langle R_{ij},R_{kl}\rangle+
\langle R_{ik},R_{jl}\rangle+
\langle R_{il},R_{jk}\rangle\right].
\]

It is a PSD quartic form because \(Q^\perp(x,x,x,x)=48\|R(x,x)\|^2\). Its useful gain is conditional on the deployed feature space \(E\):

\[
G_{\perp|E}=G_{\perp\perp}-G_{\perp E}G_E^+G_{E\perp},\quad
c_{\perp|E}=c_\perp-G_{\perp E}G_E^+c_E,
\]

\[
\Delta_4=c_{\perp|E}^TG_{\perp|E}^+c_{\perp|E}.
\]

The joint cone is nonempty algebraically: extend any signed base by

\[
A_i=A_i^{(-)}\oplus\epsilon B_i,qquad
B_i=\begin{bmatrix}0&C_i\\C_i^T&0\end{bmatrix}.
\]

Every triple product of the bipartite \(B_i\)'s is off diagonal, so its trace is zero. Consequently the cubic vertex, fish, subtraction, and signed C3 gate are unchanged, while \(B_i\circ B_j\) is block diagonal and supplies a pure structural quartic-perpendicular block. If \(t=\epsilon^4\), unit endpoint feature \(q_0\), residual \(r\), \(a=\langle r,q_0\rangle\), and \(b=\|q_0\|^2\), its fixed-correction gain is

\[
G_4(t)=2at-bt^2.
\]

The unconstrained quartic gate is \(a>0\) and \(a^2/b\ge2.86\times10^{-10}\).

## Checkpoint 5: deployment realism of the direct-sum quartic block

### A canonical label-free specialization

Let \(K\succeq0\) be a resident white-box pair kernel and choose any factorization \(K=CC^T\), with row vectors \(c_i^T\). The factor never has to be formed. Define the rank-one bipartite dilation

\[
B_i(K)=\begin{bmatrix}0&c_i^T\\c_i&0\end{bmatrix}.
\]

This construction is label-free, neuron-permutation equivariant, and invariant to the factor gauge \(C\mapsto CO\). It has

\[
\langle B_i,B_j\rangle_F=2K_{ij},\qquad
\operatorname{tr}(B_iB_jB_k)=0.
\]

With \(Y_{ij}=B_i\circ B_j\), direct multiplication gives the exact kernel identity

\[
\langle Y_{ij},Y_{kl}\rangle
=K_{ij}K_{kl}+\tfrac12(K_{ik}K_{jl}+K_{il}K_{jk}).
\]

Since every \(Y_{ij}\) is block diagonal while both the old block and the \(B\)-span are orthogonal/off diagonal, this whole square lift is structurally perpendicular. In particular,

\[
Q^\perp_{iijk}=32K_{ii}K_{jk}+64K_{ij}K_{ik}.
\]

For a known label-free contraction mask \(\alpha\), all repeated-index planes collapse to

\[
Z_\alpha(K)_{jk}=\sum_i\alpha_iQ^\perp_{iijk}
=32\left(\sum_i\alpha_iK_{ii}\right)K_{jk}
+64[K\operatorname{diag}(\alpha)K]_{jk}.
\]

Thus this specialization needs no latent factor, no fourth-order tensor, and one dense matrix product if \(K\operatorname{diag}(\alpha)K\) is not already available. It is an implicit polynomial kernel.

### Exact H2 and signed-counterterm gates

For the scaled block \(\epsilon B\), put \(t=\epsilon^4\). Its unavoidable second-chaos increment is

\[
\Delta H_2=2\epsilon^2K=2\sqrt t\,K.
\]

Let \(S_2\succeq0\) be independently available H2 headroom. Exact feasibility while leaving the signed base untouched is

\[
2\sqrt t\,K\preceq S_2.
\]

If \(\operatorname{ran}K\subseteq\operatorname{ran}S_2\), define

\[
\Lambda_2=\lambda_{\max}(S_2^{\dagger/2}KS_2^{\dagger/2}).
\]

Then

\[
t\le t_{H2}=\frac1{4\Lambda_2^2}.
\]

If the range condition fails, no positive amplitude is feasible. In scalar norm slack \(h_2\), the corresponding exact bound is \(t\le h_2^2/(4\|K\|^2)\).

The anti-aligned parallel counterterm is retained exactly when the old and new propagators remain block-separable,

\[
P_-\widetilde R_LP_-=R_L,\qquad P_-\widetilde R_LP_B=0.
\]

Then the new block has zero cubic vertex and changes neither \(V\), \(T_D\), nor \(\Delta\Sigma\). If H2 normalization couples the blocks, let

\[
M_3=-P_S(\operatorname{Herm}A+\tau I+\Delta F_-)P_S\succeq0
\]

be the signed margin and let \(\Psi_2\) be the induced counterterm change from an H2 perturbation. The exact additional gate is

\[
P_S\operatorname{Herm}\Psi_2(2\sqrt t\,K)P_S\preceq M_3.
\]

For linear \(\Psi_2\), this gives another generalized-eigenvalue cap \(t\le1/(4\Lambda_{32,+}^2)\). A strict H2 equality with a faithful positive H2 observation has no nonzero solution: \(2\sqrt tK=0\) forces \(t=0\). Therefore a genuine positive-chaos direct sum is not free; it needs H2 headroom or a joint retune that re-proves the signed C3 gate.

Combining H2 and C3, set

\[
t_{\max}=\min(t_{H2},t_{C3}).
\]

The exact constrained gain gate is

\[
\widehat t=\min(a/b,t_{\max}),\qquad
a>0,\qquad 2a\widehat t-b\widehat t^2\ge\Delta_{\rm req}.
\]

Equivalently \(a^2/b\ge\Delta_{\rm req}\) and \(t_{\max}\ge t_-\), where

\[
t_-=(a-\sqrt{a^2-b\Delta_{\rm req}})/b.
\]

### State, arithmetic, and adjusted-score gates

Measure one stream as one resident \(n\times n\) array and one matmul-equivalent as one dense \(n^3\) product. For a generic \(C_i\in\mathbb R^{p\times q}\), storing all factors costs \(pq/n\) streams and forming their pair Gram costs \(pq/n\) matmul-equivalents. The generic Jordan contraction has feature dimension

\[
d_J=p(p+1)/2+q(q+1)/2
\]

and, absent an algebraic collapse, direct evaluation costs \(d_J\) matmul-equivalents per source layer. Hence the generic existential block does not meet a one/two-matmul gate.

The rank-one Gram lift above is the collapse: with resident \(K\), it costs one product per source layer, and its result may be written directly into the existing covariance-source destination. Its incremental persistent state is zero and its maximum live requirement is the resident \(K\) plus that destination. If \(K\) must instead be formed from an \(n\times q\) factor, the exact accounting is

\[
N_{\rm mm}=L_Q(1+\chi_Kq/n)+2L_Ts_{\rm new},
\]

where \(L_Q\) is the number of source sites, \(L_T\) the number of separately transported transitions, and \(s_{\rm new}\) the number of new covariance-like streams. The stream gate is

\[
(q/n)\chi_{C,\rm new}+\chi_{K,\rm new}+\chi_{Z,\rm unaliased}+s_{\rm new}
\le S_{\max},\qquad S_{\max}\in\{1,2\}.
\]

At f32 billing \(\kappa=0.00161\) utilization per matmul-equivalent, a raw reduction \(\Delta\) is adjusted-score positive exactly when

\[
(R_0-\Delta)(u_0+\kappa N_{\rm mm})<R_0u_0,
\]

or

\[
\Delta>\frac{R_0\kappa N_{\rm mm}}{u_0+\kappa N_{\rm mm}}.
\]

Using \(R_0=2.9409\times10^{-8}\), \(u_0=.5709\), and only the minimum quartic gain \(2.86\times10^{-10}\), the exact ceiling is \(N_{\rm mm}<3.48\): at most three total new f32 matmul-equivalents. A separately transported stream over 14 layers costs 28 equivalents and needs about \(2.15\times10^{-9}\) raw gain; two streams need about \(4.01\times10^{-9}\). Therefore the minimum-size quartic opportunity can fire only if it rides an existing transport and is fused/pre-existing or evaluated at no more than three total sites.

### Observability and novelty separation theorem

The generic direct-sum existence proof is not an observability proof. Let the old state record all pair and cubic traces. For one index, compare

\[
C=\operatorname{diag}(1,0),\qquad C'=I_2/\sqrt2.
\]

Their bipartite dilations have the same \(\operatorname{tr}B^2=2\) and \(\operatorname{tr}B^3=0\), but

\[
\operatorname{tr}B^4=2,\qquad \operatorname{tr}{B'}^4=1.
\]

Hence identical H2/H3 observables admit different quartic-perpendicular sectors. No function of an existing degree-at-most-three moment state can recover the generic \(q4_\perp\). Raw white-box weights help only after an explicit equivariant map \(\Gamma(\text{weights, gates})=\{C_i\}\) is specified.

The rank-one Gram lift supplies such a map implicitly and cheaply, but it is a synthetic Gram closure, not proof that the physical noncommutative \(q4_\perp\) equals that closure. Moreover structural perpendicularity to \(\operatorname{span}\{A_i\}\) is not the same as statistical innovation relative to the deployed estimator. Since \(Z_\alpha(K)\) is a quadratic polynomial of an already resident pair kernel, it is a duplicate whenever

\[
(I-P_E)Z_\alpha(K)=0.
\]

Thus there are three distinct outcomes:

1. Generic free \(C_i\): algebraically feasible but non-identifiable and non-deployable.
2. Rank-one Gram lift: observable, equivariant, and budget-compatible, but potentially an old covariance-polynomial duplicate.
3. Physical noncommutative square lift from explicit white-box matrices: genuinely identifiable, but it needs a separate cost proof and generally exceeds the one/two-product gate without more structure.

Even for the cheap lift, label-free generation cannot guarantee target alignment: replacing a compatible endpoint residual \(r\) by \(-r\) leaves all white-box state fixed and flips \(a=\langle r,q_0\rangle\). Offline fitting may choose the sign/coefficient, but the useful conditional gain remains an empirical premise.

### Graph-ready split nodes for this checkpoint

- **Transport identity (proved):** the rank-one dilation yields
  \(Z_\alpha=32(\sum_i\alpha_iK_{ii})K+64K\operatorname{diag}(\alpha)K\).
  Terminal criterion: symbolic equality under arbitrary PSD \(K\). Evidence: the displayed Jordan inner-product identity. Edge: supplies the quartic source to the existing covariance transport.
- **Producer/consumer chain rule (proved):** a deploy-frame carrier must include \(J_{T\theta}=\delta A\text{-on-}T+DS_2\), while compact \(\delta G\) masks are the consumer half. Terminal criterion: show both halves in the same recurrence. Edge: producer precedes source injection, consumer follows it.
- **Signed stability (conditional pass):** retain \(P_S\Delta FP_S\preceq-P_S(\operatorname{Herm}A+\tau I)P_S\), with the indirect H2 gate above. Terminal criterion: C3 \(<.8466\) with positive Schur margin. Edge: depends on block-separable propagation or bounded \(\Psi_2\).
- **PSD/double-count gate (proved form, open premise):** use the full functional subtraction \(V R V^*-T_DRT_D^*\); add only the conditional \((I-P_E)q4_\perp\); require \(2\sqrt tK\preceq S_2\). Terminal criterion: joint PSD, no pure-\(T_D\) skeleton repeated, nonzero conditional Schur gain.
- **Acquisition/novelty premise (open):** determine whether the resident D11/F2J algebra already contains \(K\operatorname{diag}(\alpha)K\), and whether the physical white-box square lift has a cheaper identity. Kill if the residualized source is identically zero.
- **Cost premise (open):** count exact existing versus new products and aliases. Pass only if the gain satisfies the adjusted-cost inequality; at the minimum quartic gain no more than three new f32 matmul-equivalents are affordable.

### Current next question

Can the newly identified tensor-free local K31 Bellman action be written as the same conditional quartic source \(Z_\alpha(K)\), or does its scalar-at-birth contraction retain noncommutative information that escapes the cheap Gram lift? The decisive derivation is an exact composition of that local action with the deploy-frame producer/consumer tangent, followed by a symbolic projection against D11, NCQC/H3-parallel, and the rank-one Gram closure.

## Checkpoint 7: universal K31 Bellman action composed with the deploy-frame secant

### Exact local adjoint identity

Let \(X=(X_i)\) be centered, let \(W\) be the fixed layer weight map, and let \(Q\equiv Q_{31}\) be the reverse mask paired with the repeated-three/one fourth-cumulant plane. Define

\[
U_a=\sum_iW_{ia}X_i,\qquad
P=WQ^T,\qquad
V_a=\sum_iP_{ia}X_i.
\]

Because

\[
V_a=\sum_bQ_{ab}U_b,
\]

multilinearity of the joint cumulant gives

\[
\langle Q,K_{31}(U)\rangle_F
=\sum_{a,b}Q_{ab}\kappa_4(U_a,U_a,U_a,U_b)
=\sum_a\kappa_4(U_a,U_a,U_a,V_a).
\]

For centered variables this is exactly

\[
\mathcal B_{31}(X,Q)
=\sum_a\left[\mathbb E(U_a^3V_a)
-3\,\mathbb E(U_a^2)\mathbb E(U_aV_a)\right].
\]

No K31 tensor need be materialized. This is an adjoint evaluation of the same forward source, not an additional correction.

If the forward plane recurrence is

\[
K_{\ell+1}=\mathcal A_\ell K_\ell+S_\ell
\]

and the reverse recurrence is \(Q_\ell=\mathcal A_\ell^*Q_{\ell+1}\), the discrete Green identity is

\[
\langle Q_L,K_L\rangle-
\langle Q_0,K_0\rangle
=\sum_{\ell=0}^{L-1}\langle Q_{\ell+1},S_\ell\rangle.
\]

Each summand is the scalar-at-birth formula above. Therefore adding both a forward K31 carry and this Bellman sum doubles the same mechanism exactly.

### Exact deploy-frame secant and producer/consumer split

For two deploy frames, denote endpoint quantities by unprimed and primed symbols and put \(\Delta S=S'-S\), \(\Delta Q=Q'-Q\). At every layer,

\[
\langle Q',S'\rangle-\langle Q,S\rangle
=\langle Q,\Delta S\rangle+\langle\Delta Q,S'\rangle.
\]

Equivalently,

\[
=\langle Q,\Delta S\rangle+
\langle\Delta Q,S\rangle+
\langle\Delta Q,\Delta S\rangle.
\]

The first term is the producer/source secant, the second the consumer-mask secant, and the third is the finite co-adaptation cross term. It is missed by a frozen-mask evaluation. Summing the endpoint Bellman representations gives the exact total secant

\[
\Delta\mathcal J
=\sum_\ell\left[
\langle Q_{\ell+1},\Delta S_\ell\rangle
+\langle\Delta Q_{\ell+1},S'_\ell\rangle
\right]
\]

up to the explicit initial boundary term.

The reverse tangent itself obeys

\[
\delta Q_\ell=\mathcal A_\ell^*\delta Q_{\ell+1}
+(D\mathcal A_\ell[\delta\theta_\ell])^*Q_{\ell+1}.
\]

The second summand is the reverse/consumer analogue of the earlier \(\delta A\)-on-tensor producer term. Both are required in a deploy-frame co-adapted calculation.

At the scalar level, with \(\mathsf K(A,B,C,D)=\kappa_4(A,B,C,D)\), the infinitesimal derivative is

\[
\delta\mathcal B_{31}
=\sum_a\left[
3\mathsf K(\delta U_a,U_a,U_a,V_a)
+\mathsf K(U_a,U_a,U_a,\delta V_a)
\right].
\]

In centered raw moments this is

\[
\sum_a\Bigl(
3\mathbb E[U_a^2\delta U_aV_a]+\mathbb E[U_a^3\delta V_a]
-3\{2\mathbb E[U_a\delta U_a]\mathbb E[U_aV_a]
+\mathbb E[U_a^2](\mathbb E[\delta U_aV_a]+\mathbb E[U_a\delta V_a])\}
\Bigr).
\]

For a coupled finite secant \(U'=U+d\), \(V'=V+e\), exact polarization yields seven scalar cumulants:

\[
\begin{aligned}
\Delta\mathsf K={}&3\mathsf K(d,U,U,V)+3\mathsf K(d,d,U,V)+\mathsf K(d,d,d,V)\\
&+\mathsf K(U,U,U,e)+3\mathsf K(d,U,U,e)
+3\mathsf K(d,d,U,e)+\mathsf K(d,d,d,e).
\end{aligned}
\]

The endpoint telescope above is preferable when no canonical coupling of the two approximate laws is available.

### Acquisition theorem: tensor-free is not moment-free

The covariance state determines \(\mathbb E U_a^2\) and \(\mathbb E U_aV_a\), but it does not determine \(\mathbb E U_a^3V_a\). In one dimension with \(W=P=1\), a standard Gaussian and a Rademacher variable both have mean zero, variance one, and third moment zero, while

\[
\kappa_4(X,X,X,X)=0\quad\hbox{versus}\quad -2.
\]

Consequently no state containing only moments through degree three can evaluate the universal K31 action. The identity removes tensor materialization only under an additional label-free fourth-moment oracle

\[
\mathfrak M_4(X;w,w,w,p)
=\mathbb E[(w^TX)^3(p^TX)]
\]

for the actual white-box law. An analytic ReLU/Gaussian integral, exact finite latent representation, or an already resident fourth-chaos state could supply it; a pair-kernel closure merely makes a model assumption. Under coordinate independence it collapses to \(\sum_iw_i^3p_i\kappa_4(X_i)\), but that assumption deletes the all-distinct sector and therefore cannot establish the desired mechanism.

This also separates the Bellman action from the cheap rank-one Gram lift. The latter is determined by \(K\) and produces only

\[
32(\sum_i\alpha_iK_{ii})K+64K\operatorname{diag}(\alpha)K.
\]

The K31 action is equal to this form only if the chosen fourth-moment oracle satisfies the corresponding Wick/Gram closure identity. A nonzero difference is precisely the higher-law information the cheap lift cannot observe.

### Exact residual and double-count gate

The subtraction \(\mathbb E U^3V-3\mathbb E U^2\mathbb E UV\) already removes the Gaussian Wick pairing. It does not automatically remove connected fourth-order components already represented by D11, the NCQC/H3-parallel Gram, or another quartic feature.

Let \(\mathscr E_\ell\) be the source subspace generated by all already deployed fourth-order mechanisms and let \(\Pi_\ell\) be its projector in the source covariance metric. The source-level new action is exactly

\[
b_{\ell,\perp}
=\langle Q_{\ell+1},(I-\Pi_\ell)S_\ell\rangle
=\langle (I-\Pi_\ell^{*})Q_{\ell+1},S_\ell\rangle.
\]

If this projection cannot itself be implemented tensor-free, the exact linear-estimator alternative is conditional residualization across the corpus. For scalar Bellman feature \(z\) and old feature vector \(E\),

\[
g_{z|E}=g_{zz}-g_{zE}G_E^+g_{Ez},\qquad
c_{z|E}=c_z-g_{zE}G_E^+c_E,
\]

and

\[
\Delta_{\rm raw}=c_{z|E}^2/g_{z|E}.
\]

The mechanism is a duplicate if \(g_{z|E}=0\), and it is task-dead if \(c_{z|E}=0\). It is forbidden to add the Bellman scalar on top of its forward K31 endpoint, or to add the unsubtracted fourth moment on top of a covariance/Wick term.

### State and adjusted-cost gate

The scalar-at-birth method needs no forward K31 stream. Its favorable implementation taps

\[
P=WQ^T
\]

while \(P\) is already the first intermediate of the reverse plane conjugation, evaluates the scalar action, and then completes that conjugation. In that case \(P\) costs no new dense product and no persistent stream. If the reverse \(Q_{31}\) stream itself is new, however, it still costs one plane and two dense products per transported layer; the local identity does not erase that cost.

If an exact law representation uses \(M_\ell\) scalar atoms/basis functions, put \(\rho_\ell=M_\ell/n\). Forming a new \(V\)-law costs \(\rho_\ell\) matmul-equivalents and \(\rho_\ell\) streams. A general accounting is

\[
N_{\rm mm}=\sum_{\ell\in\mathcal L}
\left(\chi_{P,\ell}+\rho_\ell\chi_{V,\ell}+c_{4,\ell}\right)
+2L_Ts_{Q,\rm new},
\]

where \(c_{4,\ell}\) is the fourth-moment oracle cost normalized by one \(n^3\) product. The live-state gate is the resident reverse mask plus its tapped intermediate and the law representation; any claimed one/two-stream implementation must exhibit legal buffer aliasing explicitly.

With raw gain \(\Delta\), exact adjusted feasibility remains

\[
N_{\rm mm}<\frac{u_0\Delta}{\kappa(R_0-\Delta)}.
\]

At \(\Delta=2.86\times10^{-10}\), this is \(N_{\rm mm}<3.48\). Hence a new depth-14 reverse plane (28 products) cannot be justified by the minimum quartic opportunity; the action must reuse an existing reverse plane/intermediate or produce a much larger conditional raw gain.

### Graph-ready split nodes for K31

- **K31 adjoint transport identity -- proved.** Criterion: verify the discrete Green identity and the scalar formula for arbitrary \(Q,W,X\). Evidence: cumulant multilinearity. Edge: alternative implementation of, never additive to, forward K31 transport.
- **K31 producer/source secant -- proved.** Criterion: retain \(\langle Q,\Delta S\rangle\) at each birth. Edge: consumes the deploy-frame source change.
- **K31 consumer/reverse secant -- proved.** Criterion: retain \(\langle\Delta Q,S'\rangle\), including \((D\mathcal A[\delta\theta])^*Q\). Edge: closes the co-adaptation loop with the producer node.
- **K31 fourth-moment acquisition -- open.** Criterion: exhibit a label-free exact \(\mathfrak M_4\) from resident white-box state. Kill if only degree-at-most-three moments are available; the Gaussian/Rademacher counterexample is terminal evidence for that restricted state.
- **K31 residual/double-count -- open.** Criterion: residualize against D11, NCQC/H3-parallel, the Gram lift, and any forward K31 action; require \(g_{z|E}>0\) and \(c_{z|E}\ne0\). Kill if the symbolic or conditional residual is zero.
- **K31 signed/PSD role -- conditional.** As a terminal scalar feature it changes no covariance and cannot spend the signed Schur margin. If injected into a forward covariance, reapply the C3 numerical-range and covariance-PSD gates; scalar equivalence alone provides no PSD certificate.
- **K31 cost/acquisition premise -- open.** Criterion: tap an already billed \(P=WQ^T\), show exact oracle/buffer cost, and satisfy the adjusted inequality. A newly transported reverse plane fails at minimum gain.

### Current next question

Does the actual D11/ReLU white-box representation contain an exact scalar fourth-moment oracle for \(\mathbb E(U^3V)\), and is \(P=WQ^T\) already a billed intermediate of an existing reverse plane? Without both facts the K31 identity is a clean adjoint theorem but not yet a deployable raw-reduction mechanism.

## Checkpoint 8: exact affine--quadratic oracle and the sharp beyond-AQ duplicate gate

The fourth-moment acquisition question has an exact answer on the affine--quadratic Gaussian-chaos subspace. Let \(g\sim N(0,I_r)\) and write two centered variables as

\[
U=u^Tg+(g^TAg-\operatorname{tr}A),\qquad
V=v^Tg+(g^TBg-\operatorname{tr}B),
\]

with symmetric \(A,B\). Their joint cumulant generating function is obtained from

\[
K(\ell,M)=-\tfrac12\log\det(I-2M)-\operatorname{tr}M
+\tfrac12\ell^T(I-2M)^{-1}\ell.
\]

The total-degree-four terms are the fourth log-determinant term and \(2\ell^TM^2\ell\). Extracting the coefficient with three copies of \((u,A)\) and one copy of \((v,B)\) gives the exact identity

\[
\boxed{
\kappa_4(U,U,U,V)
=48\operatorname{tr}(A^3B)
+24\left[u^TABu+u^TA^2v\right].
}
\]

As a normalization check, setting \(V=U\) gives

\[
\kappa_4(U)=48\operatorname{tr}(A^4)+48u^TA^2u,
\]

the standard fourth cumulant of a linear-plus-quadratic Gaussian form.

For the K31 Bellman action, let \((u_a,A_a)\) be the affine--quadratic coefficients of \(U_a\) and define the reverse mixtures

\[
v_a=\sum_bQ_{ab}u_b,qquad
B_a=\sum_bQ_{ab}A_b.
\]

Then the entire AQ action is

\[
\mathcal B_{31}^{AQ}
=\sum_a\left\{
48\operatorname{tr}(A_a^3B_a)
+24[u_a^TA_aB_au_a+u_a^TA_a^2v_a]
\right\}.
\]

This is label-free and tensor-free when the resident state already contains \(u_a,A_a\). If \(r^2\asymp n\), flattening the \(A_a\)'s gives one \(n\times n\)-scale stream. Forming all \(B_a=\sum_bQ_{ab}A_b\) is one dense matmul-equivalent, the \(B\)-stream is a second stream, and the local trace work is \(O(nr^3)=O(n^{5/2})\), subleading to \(n^3\). The linear mixtures are cheaper.

This closes only the AQ acquisition premise. It simultaneously proves a duplicate theorem: \(\mathcal B_{31}^{AQ}\) is the scalar adjoint of the exact DTOT2/quadratic-chaos square-lift action. It must not be counted as new on top of that action.

For a full white-box law decompose

\[
X=X^{\le2}+R_{>2}
\]

in Gaussian chaos. The sharp beyond-AQ feature is

\[
\mathcal B_{31}^{>AQ}
=\mathcal B_{31}(X,Q)-\mathcal B_{31}^{AQ}(X^{\le2},Q).
\]

By fourth-cumulant multilinearity this residual is exactly the sum of polarized terms containing at least one \(R_{>2}\) leg. It is not determined by \((u_i,A_i)\): the earlier same-low-moments/different-fourth-cumulant example remains a terminal non-identifiability proof. Therefore the universal Bellman identity yields a deployable exact oracle for the already-owned AQ/DTOT2 sector, while the potentially novel beyond-AQ sector still needs a label-free higher-chaos oracle.

### Additional graph-ready split nodes

- **AQ scalar oracle -- proved.** Criterion: match the boxed CGF identity for arbitrary symmetric \(A,B\) and vectors \(u,v\). Evidence: exact degree-four coefficient extraction.
- **AQ/DTOT2 duplicate -- proved.** Criterion: pair the forward quadratic-chaos K31 source with \(Q\) and recover \(\mathcal B_{31}^{AQ}\). Edge: mutually exclusive implementations, not stackable gains.
- **Beyond-AQ acquisition -- open.** Criterion: compute the residual terms containing at least one \(R_{>2}\) leg from resident white-box state. Kill for an AQ-only state by non-identifiability.
- **AQ cost -- conditional pass.** Criterion: existing \(A\)-stream, one reverse mixture into a second stream, and adjusted-cost headroom. Open premise: whether that mixture is already billed/fusable.

### Revised next question

Is there a similarly closed scalar formula for the first omitted ReLU chaos sector (one H3/H4 leg coupled to three AQ legs) that uses the already resident NCQC/H3 state without reconstructing a fourth-order tensor, and after subtracting the NCQC row/column Gram is any conditional component left?

## Checkpoint 9: exact one-H3-leg Bellman oracle

Let

\[
L_u=u^Tg,\qquad Q_A=g^TAg-\operatorname{tr}A,qquad
H_T=T:g^{\otimes3}:,
\]

where \(:\cdot:\) denotes Wick ordering and \(T\) is fully symmetric. Joint cumulants of Wick polynomials are sums of connected pairings between their vertices.

For one cubic and three linear vertices, the only connected pairing joins each cubic leg to one linear leg, so

\[
\kappa_4(H_T,L_u,L_v,L_w)=6T(u,v,w).
\]

For degree pattern \((3,1,2,2)\), there are exactly three connected multigraph topologies. Counting labelled half-edge pairings gives

\[
\boxed{
\begin{aligned}
F(T,u,A,B)
&:=\kappa_4(H_T,L_u,Q_A,Q_B)\\
&=24T(u,\operatorname{sym}(AB))
+12T(Au,B)+12T(Bu,A),
\end{aligned}}
\]

where \(T(x,M)=\sum_{abc}T_{abc}x_aM_{bc}\). The coefficients have a direct topology interpretation:

- 24 pairings for edges \(T-L,T-A,T-B,A-B\);
- 12 pairings for edges \(L-A,T-A,2(T-B)\);
- 12 pairings for the interchange \(A\leftrightarrow B\).

In one dimension the formula becomes \(48tuab\), agreeing with the direct moment

\[
\mathbb E[(g^3-3g)g(g^2-1)^2]=48.
\]

Now let

\[
U=L_u+Q_A+H_T+\cdots,\qquad
V=L_v+Q_B+H_S+\cdots.
\]

Parity and cumulant multilinearity show that the complete K31 action linear in the H3 coefficients is

\[
\boxed{
\begin{aligned}
\mathcal K_{31}^{(1\times H3)}={}&
18T(u,u,v)+6S(u,u,u)\\
&+3F(T,v,A,A)+6F(T,u,A,B)+3F(S,u,A,A).
\end{aligned}}
\]

The first line is the \((3,1,1,1)\) degree pattern. The second is the \((3,1,2,2)\) pattern. All other choices with exactly one H3 leg have odd total Gaussian degree and vanish.

This is an exact scalar-at-birth formula, but its acquisition requirement is sharper than merely possessing repeated H3 planes. It needs the trilinear oracle

\[
T(x,y,z)
\]

and the vector--matrix oracle

\[
T(x,M)
\]

for matrices in the span of \(A\), \(B\), \(\operatorname{sym}(AB)\), and their gate/transport mixtures. A repeated-plane map \(R(T)_{ij}=T_{iij}\) does not determine these contractions for generic dense arguments. Two tensors differing by a nonzero \(Q=I-LR\) component have identical plane state but can give different boxed actions.

Therefore this checkpoint creates an exact symbolic gate:

- if the NCQC/H3 implementation exposes all displayed contractions, the one-H3 Bellman action is a formula-level duplicate and must only replace its tensor carry;
- if it exposes only repeated planes or row/column Grams, the scalar formula identifies the missing all-distinct oracle but does not generate it;
- any proposed compression must prove that its contraction error is small on the specific matrix family \(\{\operatorname{sym}(AB),A,B\}\), not merely in global tensor norm.

### Graph-ready split nodes

- **H3 connected-diagram identity -- proved.** Criterion: reproduce the two boxed formulas from connected Wick pairings. Evidence: topology counts and the one-dimensional normalization check.
- **H3 contraction acquisition -- open.** Criterion: answer \(T(x,y,z)\) and \(T(x,M)\) for every Bellman-generated argument using resident state. Kill for plane-only state by a nonzero \(Q\)-sector counterexample.
- **H3/NCQC duplicate -- conditional.** Criterion: symbolically match every term to the NCQC/H3 row/column action. If matched, use only one adjoint implementation.
- **H3 all-distinct residual -- open.** Criterion: apply \(Q=I-LR\) to the H3 provider and show a nonzero conditional task contraction after old-feature residualization.
- **H3 cost -- open.** Criterion: exhibit the contraction oracle within the two-stream and adjusted-product gates; storing the generic full \(T\) is disallowed.

### Revised next question

Can the needed map \(M\mapsto T(\cdot,M)\), restricted to Bellman matrices generated by \(A,B,\operatorname{sym}(AB)\), be represented by a low-degree noncommutative Krylov module whose dimension is bounded independently of width? Such a restriction could evade the generic all-distinct sketch lower bound; failure would convert the acquisition node into a structural no-go.

## Checkpoint 10: restricted Bellman-module lower bound and the only exact low-state escape

For a symmetric H3 coefficient \(T\in\operatorname{Sym}^3(\mathbb R^r)\), define the linear contraction map

\[
\Lambda_T:\operatorname{Sym}(r)\to\mathbb R^r,
\qquad [\Lambda_T(M)]_a=\sum_{bc}T_{abc}M_{bc}.
\]

Then every difficult term in the one-H3 Bellman action is

\[
T(x,M)=x^T\Lambda_T(M).
\]

Assume the resident vector arguments span \(\mathbb R^r\), the quadratic coefficients \(\{A_i\}\) span \(\operatorname{Sym}(r)\), and the reverse mask \(Q\) is allowed to be dense. Since \(B_a=\sum_bQ_{ab}A_b\), the Bellman matrices \(B_a\) also range over all of \(\operatorname{Sym}(r)\). Therefore the query set \(T(x,B)\) spans the full dual of \(\operatorname{Sym}^3(\mathbb R^r)\). The extra \(\operatorname{sym}(AB)\) arguments can only enlarge, not reduce, this set.

This proves two lower bounds.

1. **Worst case.** Any non-injective linear sketch of the H3 provider has a nonzero null tensor. Because the Bellman queries span the full dual, some legal query separates that tensor from zero. Thus no compressed sketch can guarantee any positive correlation uniformly over H3 providers.
2. **Isotropic average case.** For \(n\) independent H3 coefficient tensors, the ambient coefficient dimension is

   \[
   D_{\rm tot}=n\binom{r+2}{3}.
   \]

   A \(k\)-stream state has at most \(kn^2\) scalar degrees of freedom. For an isotropic provider, the maximum expected retained squared correlation of any rank-\(kn^2\) linear sketch is

   \[
   \rho^2\le\frac{kn^2}{D_{\rm tot}}
   =\frac{kn}{\binom{r+2}{3}}.
   \]

   In the natural full-symmetric parameterization \(n=\dim\operatorname{Sym}(r)=r(r+1)/2\), this becomes

   \[
   \boxed{\rho^2\le\frac{3k}{r+2}.}
   \]

   Requiring \(\rho>.925\) forces

   \[
   k>.925^2\frac{r+2}{3}.
   \]

   At \(n\simeq1024\), \(r\simeq44\)--45, so the isotropic threshold is more than 13 full \(n^2\) streams. One or two streams are far below it.

This is a structural statement, not a claim that the actual task provider is isotropic. A task-specific low-dimensional alignment can evade the average-case bound, but then that alignment is an explicit premise requiring proof or measurement; it cannot follow from the Bellman restriction alone.

There is one natural exact escape. Suppose all Bellman matrices lie in a fixed Jordan algebra \(\mathcal A\subseteq\operatorname{Sym}(r)\) of dimension \(k_A\), all vector arguments lie in a fixed module \(\mathcal U\subseteq\mathbb R^r\) of dimension \(k_U\), and these spaces are invariant under every layer transport and gate derivative. Then only the restriction

\[
\Lambda_T|_{\mathcal A}:\mathcal A\to\mathcal U
\]

is needed, with \(k_Ak_U\) degrees of freedom. A concrete example is a simultaneously diagonalizable/commutative family contained in

\[
\mathcal A=\operatorname{span}\{I,H,\ldots,H^{d-1}\}
\]

where the minimal polynomial of \(H\) has degree \(d=O(1)\); products reduce back into \(\mathcal A\), so \(\operatorname{sym}(AB)=AB\) never grows the module. This is the same algebraic phenomenon that makes a one-pole Pad\'e/Schur closure exact.

For a generic dense noncommuting matrix family, the generated associative/Jordan algebra is full, so \(k_A\) grows like \(r^2\) and the escape disappears.

### Graph-ready split nodes

- **Bellman query-span theorem -- proved under full-span premises.** Criterion: \(\operatorname{span}\{u_i\}=\mathbb R^r\), \(\operatorname{span}\{A_i\}=\operatorname{Sym}(r)\), and dense reverse-mask reachability. Consequence: restricted queries determine the full H3 provider.
- **One/two-stream H3 lower bound -- proved for worst-case/isotropic models.** Criterion: linear sketch and the displayed ambient dimensions. Kill universal .925 claims for \(k\le2\); task-specific claims remain open.
- **Low-algebra escape -- conditional.** Criterion: exhibit a transport-invariant Jordan algebra and vector module of bounded dimension. Pass yields an exact low-state oracle; kill if algebra closure reaches full \(\operatorname{Sym}(r)\).
- **Task-alignment premise -- open.** Criterion: prove that the actual endpoint adjoint lives predominantly in the retained module. This cannot be inferred from provider energy alone.

### Revised next question

Can the endpoint reverse masks themselves be shown to occupy a low-dimensional invariant algebra even when the forward \(A_i\) family is full? If the adjoint query span is low, only a task-weighted quotient of \(\Lambda_T\) is needed; if successive gate derivatives make that mask algebra full, the one/two-stream route is closed symbolically.

## Checkpoint 11: exact arbitrary-query contraction of the structured all-distinct ReLU source

The generic H3 lower bound does not apply to a local source that retains its special provenance factorization. For the all-distinct ReLU birth source, let

\[
S_{ijk}=\beta_i c_{ij}c_{ik}+eta_jc_{ji}c_{jk}+\beta_kc_{ki}c_{kj},
\qquad c_{ii}=0,
\]

on triples with \(i,j,k\) pairwise distinct. For an arbitrary vector \(x\) and symmetric matrix \(M\), define

\[
S(x,M)=\sum_{ijk}S_{ijk}x_iM_{jk}.
\]

The contraction has the exact identity

\[
\boxed{
\begin{aligned}
S(x,M)={}&\sum_i\beta_ix_i
\left[c_i^TMc_i-\sum_jc_{ij}^2M_{jj}\right]\\
&+2\sum_i\beta_i\left[
(c_i^Tx)\left(\sum_kc_{ik}M_{ik}\right)
-\sum_jc_{ij}^2x_jM_{ij}
\right].
\end{aligned}}
\]

Proof: the cyclic term centered at \(i\) factors as \(c_i^TMc_i\); subtracting the forbidden collision \(j=k\) gives the first line. The term centered at \(j\) factors into \((c_j^Tx)(\sum_kc_{jk}M_{jk})\), again with the forbidden \(i=k\) removed. The term centered at \(k\) is identical after interchanging \(j,k\), because \(M\) is symmetric.

The formula passes a polynomial consistency check. Taking \(M=xx^T\), the second line equals twice the first, so

\[
S(x,xx^T)=3\sum_i\beta_ix_i
\left[(c_i^Tx)^2-\sum_jc_{ij}^2x_j^2\right],
\]

which is the earlier squarefree cubic circuit exactly.

### Deployment consequence

Let \(C\) be the matrix with rows \(c_i^T\). All terms except \(c_i^TMc_i\) cost \(O(n^2)\). The latter is the diagonal of \(CMC^T\), which can be obtained by forming \(CM\) once and taking rowwise inner products with \(C\). Therefore an arbitrary dense Bellman query costs

\[
\text{one }n^3\text{ matmul-equivalent}+O(n^2),
\]

with the resident \(C\)-stream and one scratch/destination stream. There is no tensor state, no rank restriction on \(M\), and no generic sketch correlation loss.

Combined with the discrete Green identity, this evaluates every structured all-distinct source at birth against its exact reverse query. It can therefore replace a forward broad-spectrum H3 carry for a scalar endpoint functional. Its novelty is computational: the represented physics is exactly the full forward transport.

### Scope and secant gates

1. A single reverse mask evaluates one scalar endpoint functional. If the deployed estimator needs \(d\) independent endpoint functionals with unrelated masks, the cost multiplies by the dimension of their mask span.
2. The displayed source is label-free only when \(\beta\) and \(C\) are generated from the deployed white-box frame. It does not by itself supply truth-frame source differences.
3. For two frames, source and query must be varied together. The exact local secant is

   \[
   S'(x',M')-S(x,M)
   =[S'-S](x',M')+S(x'-x,M')+S(x,M'-M),
   \]

   where the first bracket means changing \((\beta,C)\) at fixed new query. Expanding it produces the finite cross terms automatically. A frozen reverse mask omits the last query change and its interaction with the source change.
4. Collision sectors are excluded explicitly by the two subtraction terms. Any existing all-distinct approximation must still be conditionally projected out; support separation alone only removes collision duplication.

### Cost implication at full-chain scale

The earlier \(2.86\times10^{-10}\) minimum quartic opportunity can afford only three new products, but the certified full correction chain was approximately an 18.7% raw effect. If an adjoint implementation retained a raw reduction \(\Delta\) of that scale, its exact adjusted product allowance would be

\[
N_{\max}=\frac{u_0\Delta}{\kappa(R_0-\Delta)},
\]

which is on the order of eighty f32 matmul-equivalents for \(R_0=2.9409\times10^{-8}\), \(u_0=.5709\), and \(\Delta\approx.187R_0\). Thus a depth-wide reverse stream plus one source product per layer can be economically viable if, and only if, it recovers a substantial fraction of the certified chain. The correct gate is end-to-end retained raw gain, not the small isolated q4 threshold.

### Graph-ready split nodes

- **All-distinct arbitrary-query identity -- proved.** Criterion: symbolic equality of the boxed formula for arbitrary symmetric \(M\). Evidence: cyclic factorization and the rank-one polynomial check.
- **Adjoint transport replacement -- conditional pass.** Criterion: one scalar endpoint mask and exact Green identity; never add it to the same forward H3 carry.
- **Deploy-frame source acquisition -- open.** Criterion: generate \(\beta,C\) label-free at every birth and identify whether the desired correction is the deployed source or an unavailable truth-source difference.
- **Secant co-adaptation -- proved form, open inputs.** Criterion: vary source and reverse query jointly using the displayed endpoint secant; require deployable primed/new-frame ingredients.
- **Collision/double-count -- proved collision subtraction, open old-feature overlap.** Criterion: exclude repeated indices algebraically and residualize any old all-distinct action statistically/symbolically.
- **Endpoint mask-span cost -- open.** Criterion: determine the number of independent scalar masks and exact reuse of reverse conjugation intermediates; apply the full-chain adjusted allowance.

### Revised next question

Can the certified 18.7% variance-chain endpoint correction be written as one (or a small fixed number of) scalar endpoint functionals whose reverse masks are generated entirely from the deployed D11 frame? If yes, the boxed birth contraction is an exact broad-spectrum carrier replacement; if the correction requires the full per-neuron truth-plane field to update downstream gates, the adjoint only recovers a frozen linear functional and the co-adapted source/query secant remains the missing piece.

## Checkpoint 12: the certified frozen chain is necessarily one block adjoint

The campaign handoff states that the certified 18.7% replay is exactly linear in its per-layer supply deltas, and the final MSE estimate is scalar. Abstract the fixed deploy-frame tangent as

\[
\delta\theta_{\ell+1}=J_\ell\delta\theta_\ell+B_\ell S_\ell,
\qquad
\delta y=r^T\delta\theta_L.
\]

Define one reverse block mask by

\[
\lambda_L=r,
\qquad
\lambda_\ell=J_\ell^T\lambda_{\ell+1}.
\]

Then repeated substitution gives the exact discrete-adjoint identity

\[
\boxed{
\delta y
=\lambda_0^T\delta\theta_0
+\sum_{\ell=0}^{L-1}
\langle B_\ell^T\lambda_{\ell+1},S_\ell\rangle.
}
\]

Consequently the fixed-tangent certified chain never requires the full forward supply field merely to evaluate its scalar endpoint value. Its broad-spectrum content is retained exactly by the reverse masks; every structured all-distinct H3 birth pairing can use the one-product formula from Checkpoint 11.

This is a theorem about transport, not yet about source availability. The decisive provenance split is:

1. If each certified supply \(S_\ell\) is the omitted all-distinct birth term computed from deployed \((m,v,C)\), the frozen 18.7% mechanism has an exact label-free adjoint realization.
2. If \(S_\ell\) is a truth-minus-deploy plane error, or was evaluated in a truth-fed frame, the adjoint transports it cheaply but does not make that source observable.
3. If downstream gates must respond to the correction, the fixed \(J_\ell\) action is only the first term of the deploy-frame secant. The source/query co-adaptation terms derived earlier remain necessary.

### Graph-ready split nodes

- **Certified scalar adjoint -- proved from stated linearity.** Criterion: scalar endpoint and fixed linear tangent. Consequence: one block reverse mask evaluates the full frozen chain.
- **Certified source provenance -- open and now decisive.** Criterion: express every banked dK21/dK22/dK31 supply as a label-free function of deployed white-box state. Kill exact deployment if a truth-fed quantity remains.
- **Frozen-versus-co-adapted boundary -- proved distinction.** Criterion: fixed Jacobian uses the boxed adjoint; joint secant additionally needs \(\Delta J\), \(\Delta Q\), and source/query cross terms.
- **Economic capacity -- conditional pass.** If a substantial fraction of 18.7% survives, depth-wide adjoint/source products fit the exact adjusted allowance; if only the isolated q4 minimum survives, they do not.

### Revised next question

Derive the certified supply decomposition into (a) a locally generated omitted source, (b) transport error, and (c) truth-frame gate error. Only part (a) is closed by the all-distinct birth circuit. Parts (b)--(c) require either the exact secant recursion or an additional observable counterterm.

## Checkpoint 13: two-stream second-order co-adaptation without an H3 carrier

Introduce a homotopy that turns on the omitted label-free source:

\[
\theta_{\ell+1}(\varepsilon)
=F_\ell(\theta_\ell(\varepsilon))
+\varepsilon s_\ell(\theta_\ell(\varepsilon)),
\]

where \(F_\ell\) is the deployed map and \(s_\ell\) is the structured all-distinct birth correction. At \(\varepsilon=0\), define

\[
e_\ell=\partial_\varepsilon\theta_\ell|_0,
\qquad
h_\ell=\partial_\varepsilon^2\theta_\ell|_0,
\qquad
J_\ell=DF_\ell(\theta_\ell).
\]

Direct differentiation gives the exact recurrences

\[
\boxed{e_{\ell+1}=J_\ell e_\ell+s_\ell,}
\]

\[
\boxed{h_{\ell+1}=J_\ell h_\ell
+D^2F_\ell[e_\ell,e_\ell]
+2Ds_\ell[e_\ell].}
\]

Let the scalar endpoint be \(y(\theta_L)\) and define

\[
\lambda_L=Dy(\theta_L),\qquad
\lambda_\ell=J_\ell^T\lambda_{\ell+1}.
\]

Eliminating the second tangent \(h\) by the adjoint identity yields

\[
\boxed{
y'(0)=\sum_\ell\lambda_{\ell+1}^Ts_\ell,
}
\]

and

\[
\boxed{
y''(0)=D^2y[e_L,e_L]
+\sum_\ell\lambda_{\ell+1}^T
\left(D^2F_\ell[e_\ell,e_\ell]+2Ds_\ell[e_\ell]\right).
}
\]

Therefore Taylor orders zero through two of the jointly co-adapted scalar endpoint need:

1. one forward tangent state \(e_\ell\), whose size is the existing deploy state rather than an H3 tensor;
2. one reverse adjoint state \(\lambda_\ell\);
3. local Hessian-vector contractions for \(F_\ell\); and
4. local directional derivatives of the structured source.

The derivative \(Ds_\ell[e_\ell]\) is obtained by differentiating the one-product all-distinct contraction with respect to \(\beta\), \(C\), the reverse vector, and the reverse matrix. It never requires materializing \(Ds\) as a tensor.

The exact endpoint expansion is

\[
y(1)=y(0)+y'(0)+\tfrac12y''(0)+R_3,
\qquad
|R_3|\le\frac16\sup_{0\le t\le1}|y'''(t)|.
\]

This supplies the constructive premise behind the portfolio convention that a joint map absorbs Taylor orders zero--two and leaves a normalized \(C_3d^3\) remainder. The signed Schur/Pad\'e analysis controls this remaining third-order response; it must not be applied to a source alignment that is already an endpoint correlation.

### Cost boundary

Before fusion, a dense covariance-like forward tangent costs roughly two products per layer, a reverse adjoint another two, and the all-distinct source contraction one. Over 14 layers this is about 70 matmul-equivalents, before local Hessian reuse. That fits the approximately 80-product adjusted allowance only if the implementation retains a large fraction of the certified 18.7% raw effect. It is not economical for only the isolated \(2.86\times10^{-10}\) quartic gain. DTOT2 depth truncation, tapping conjugation intermediates, and in-place source injection are therefore mathematical cost premises, not optional optimizations.

### Graph-ready split nodes

- **First tangent recurrence -- proved.** Criterion: homotopy derivative gives \(e^+=Je+s\). Edge: produces the forward co-adapted state.
- **Second tangent recurrence -- proved.** Criterion: second derivative gives \(h^+=Jh+D^2F[e,e]+2Ds[e]\). Edge: consumes the first tangent.
- **Second-order adjoint elimination -- proved.** Criterion: the two boxed scalar derivative identities. Consequence: no H3 carrier through Taylor order two.
- **Local source directional oracle -- conditional.** Criterion: differentiate the arbitrary-query source circuit with resident \(\delta\beta,\delta C,\delta x,\delta M\); kill if any direction is truth-only.
- **Third-order signed remainder -- conditional.** Criterion: apply the prior \(C_3\) numerical-range/PSD gate only after orders zero--two are included.
- **Two-stream economic gate -- open.** Criterion: exact billed product count plus retained raw gain; pass near full-chain scale, kill near isolated-q4 scale.

### Revised next question

Can the Hessian-vector term \(D^2F[e,e]\) and source derivative \(Ds[e]\) be expressed using the same gate-derivative masks and existing covariance conjugations, so the nominal five products per layer collapse to the two already billed forward/reverse passes plus at most one source product?

## Checkpoint 14: portfolio ceiling, replacement semantics, and earliest kill

The scalar-at-birth adjoint and the forward D14/all-distinct stream are two evaluations of the same Green identity. Therefore:

\[
\text{raw capacity of adjoint implementation}
\le \text{raw capacity of the certified forward chain}.
\]

Using the stated certified ceiling, this is at most 18.7% of the current raw,

\[
\Delta_{\max}\approx .187(2.9409\times10^{-8})
\approx5.50\times10^{-9}.
\]

It is forbidden to count D14 plus its adjoint as two raw gains. The adjoint replaces D14 for the selected scalar endpoint. It can stack with a D10 mechanism only after conditional residualization proves D10 represents a different source sector.

Without fusion, a 14-layer reverse plane costs 28 products and 14 arbitrary-query birth contractions cost 14 more, so \(N=42\). If births are only at L5--14, \(N=38\). The exact adjusted break-even formula gives approximately

\[
\Delta_{\rm break}(42)\simeq3.11\times10^{-9},
\qquad
\Delta_{\rm break}(38)\simeq2.85\times10^{-9}.
\]

Thus the full certified ceiling can pay for either implementation; a small isolated quartic gain cannot. At \(\Delta=5.50\times10^{-9}\), the total product allowance is about 81.

The weakest required provenance condition is not full tensor equality. It is the task-weighted condition

\[
\boxed{
\langle Q_\ell,
dK^{\rm cert}_\ell-S_{AD}(\beta_\ell^D,C_\ell^D)
\rangle=0
}
\]

for every layer and deployment instance, or a norm bound small enough to preserve the required raw gain. This is the earliest symbolic kill gate: a nonzero truth-dependent contracted residual proves that the deployed all-distinct birth circuit cannot reproduce the certified chain, regardless of its transport efficiency.

### Graph-ready split nodes

- **D14 replacement semantics -- proved.** Criterion: same source and endpoint pairing under the Green identity. Edge: mutually exclusive implementations.
- **18.7% raw ceiling -- conditional on certified attribution.** Criterion: adjoint source equals certified source in the task-weighted quotient. Never extrapolate beyond the forward oracle value.
- **Depth-wide adjusted economics -- conditional pass.** Criterion: retained raw gain exceeds the exact 38/42-product break-even.
- **Task-weighted provenance -- open, earliest kill.** Criterion: the boxed contraction equality. Kill immediately on any indispensable truth-frame remainder.
- **D10 stacking -- open.** Criterion: nonzero conditional innovation after projecting the adjoint feature against D10; source-label claims alone are insufficient.

### Revised next question

Can the certified banked supply be algebraically separated into \(S_{AD}(\beta^D,C^D)\) plus a truth-frame residual whose pairing with the reverse mask vanishes? This task-weighted quotient, rather than full plane reconstruction, is the narrowest possible proof of deployability.

## Checkpoint 15: exact reuse of gate Hessians and the source directional derivative

Consider a covariance-like deployed map

\[
F_C(\theta)=W^TK(\theta)W.
\]

For a reverse covariance mask \(\Lambda\), cyclicity gives

\[
\langle\Lambda,D^2F_C[e,e]\rangle
=\langle W\Lambda W^T,D^2K[e,e]\rangle.
\]

The matrix \(W\Lambda W^T\) is exactly the reverse conjugation already needed to propagate \(\Lambda\). Hence the scalar Hessian-vector action needs no forward Hessian conjugation and no Hessian state.

For a bivariate Gaussian gate moment \(G_{pq}\), define the directional shift operator

\[
\mathcal D_e=
\delta m_i\,E_p+\delta m_j\,E_q
+\tfrac12\delta v_i\,E_p^2+\tfrac12\delta v_j\,E_q^2
+\delta C_{ij}\,E_pE_q,
\]

where \(E_pG_{pq}=G_{p+1,q}\) and \(E_qG_{pq}=G_{p,q+1}\). Then

\[
DG_{pq}[e]=\mathcal D_eG_{pq},
\qquad
D^2G_{pq}[e,e]=\mathcal D_e^2G_{pq}.
\]

All entries are elementwise combinations of higher gate masks and cost \(O(n^2)\). For a one-site gate moment,

\[
D^2g_r[e,e]=g_{r+2}(\delta m)^2
+g_{r+3}\delta m\delta v
+\tfrac14g_{r+4}(\delta v)^2.
\]

The structured source derivative also reuses its primal product. Write the arbitrary-query contraction as

\[
S=\sum_i\beta_i\left[x_id_i+2(a_ib_i-h_i)\right],
\]

where

\[
d_i=c_i^TMc_i-\sum_jc_{ij}^2M_{jj},\quad
a_i=c_i^Tx,\quad
b_i=\sum_kc_{ik}M_{ik},\quad
h_i=\sum_jc_{ij}^2x_jM_{ij}.
\]

At fixed reverse query \((x,M)\), a forward direction \((\dot\beta,\dot C)\) gives

\[
\dot d_i=2\dot c_i^TMc_i-2\sum_jc_{ij}\dot c_{ij}M_{jj},
\]

\[
\dot a_i=\dot c_i^Tx,\qquad
\dot b_i=\sum_k\dot c_{ik}M_{ik},\qquad
\dot h_i=2\sum_jc_{ij}\dot c_{ij}x_jM_{ij},
\]

and

\[
\boxed{
DS[e]=\sum_i\dot\beta_i[x_id_i+2(a_ib_i-h_i)]
+\sum_i\beta_i[x_i\dot d_i+2(\dot a_ib_i+a_i\dot b_i-\dot h_i)].
}
\]

The primal source evaluation has already formed \(CM\), hence every \(Mc_i\) needed by \(\dot d_i\) is resident. All remaining derivative terms are rowwise or elementwise \(O(n^2)\). In particular, no separate \(\dot C M\) product is necessary.

### Cost conclusion

Through Taylor order two, the dense operations are therefore only:

1. the forward tangent covariance conjugation;
2. the reverse adjoint conjugation; and
3. at most one \(CM\) source product per active layer.

The gate Hessian and \(Ds[e]\) terms piggyback on those intermediates. This realizes the nominal two-stream co-adaptation theorem without an additional Hessian stream.

### Graph-ready split nodes

- **Gate-Hessian reverse reuse -- proved.** Criterion: cyclic pairing of the covariance conjugation and the \(\mathcal D_e^2\) mask identity. Cost: elementwise after reverse transport.
- **Source directional reuse -- proved.** Criterion: the boxed derivative and resident \(CM\). Cost: \(O(n^2)\), no derivative matmul.
- **Order-two dense-cost count -- conditional pass.** Criterion: legal reuse/aliasing of forward tangent, reverse mask, and source scratch; count at most five dense products per full layer before any already-billed overlap.
- **Higher-order remainder -- open.** These identities do not bound \(y'''\); retain the signed \(C_3\) gate.

### Revised next question

The only remaining formula-level obstacle to an exact order-two deployable correction is source provenance: identify whether the banked certified \(dK\) supplies equal the displayed local source and derivative in the task-weighted quotient, or isolate the first truth-only term.

## Checkpoint 16: correction from the future-observability/cap audit

The portfolio audit establishes that the synchronized affine--quadratic K31 action is already an old D11/Q24 action, so its novel gain is exactly zero. This agrees with Checkpoint 8: \(\mathcal B_{31}^{AQ}\) is a duplicate, not a new feature.

It also identifies future-dark directions: a source can vanish under the same-site K31 observation yet become visible after dense downstream transport. A low-rank same-site action does not acquire those modes. The exact all-distinct arbitrary-query formula from Checkpoint 11 is different: because it accepts the complete dense future costate \(M\), it is a direct complete-future-costate oracle for this structured H3 source. It therefore solves the information-acquisition problem for that source even when \(M\) is full rank.

The correction is economic. Under the currently reported base-48/cap-49 stack, only one new matmul-equivalent is available. Evaluating the oracle at L5--14 needs ten independent products \(C_\ell M_\ell\); evaluating all 14 layers needs fourteen. The operands differ by layer, so batching them does not change billed arithmetic. Therefore:

\[
\boxed{
\text{all-distinct future-costate oracle is information-pass but cap-fail}
}
\]

unless one of the following is proved:

1. every required \(C_\ell M_\ell\) is an already billed intermediate of the D11/F2J forward or reverse recursion;
2. the task-weighted source support is confined to one layer;
3. a cross-layer algebraic identity evaluates the exact sum with at most one product; or
4. the cap is replaced by adjusted-score economics and the retained raw gain is near the certified full-chain scale.

The last option is economically plausible but is not equivalent to satisfying the current cap discipline.

### Graph-ready split nodes

- **AQ novelty -- closed duplicate.** Evidence: exact AQ scalar equals D11/Q24 action. Novel gain zero.
- **Structured H3 future-dark acquisition -- proved.** Evidence: arbitrary dense \(M\) is accepted by the boxed all-distinct contraction; no same-site rank assumption.
- **Cap-49 cost -- currently closed.** Evidence: one independent \(CM\) product per active layer versus one available equivalent.
- **CM fusion premise -- open.** Criterion: identify the exact operands as an already billed intermediate, not merely a similar conjugation. This is now the earliest cost reopen.
- **Adjusted-economics alternative -- conditional.** Full-chain retained gain can pay for depth-wide work, but this requires an explicit policy decision and provenance pass.

### Revised next question

Does the reverse deploy-frame recurrence already form \(C_\ell M_\ell\), perhaps as the first half of a covariance/Hessian conjugation, with exactly the same gated covariance \(C_\ell\) and complete future mask \(M_\ell\)? If not, the current cap closes the otherwise exact carrier replacement.

## Checkpoint 17: CM fusion audit -- negative

The direct K31 adjoint circuit forms dense intermediates of the type

\[
H=WBW^T
\]

and then a Hadamard reverse mask such as

\[
Q=(H\odot G_{31})/3.
\]

Its paid first product is \(WB\). The all-distinct source oracle instead requires

\[
CM,
\qquad C=\text{gated covariance/source factor},
\qquad M=\text{complete future H3 query}.
\]

There is no algebraic identity equating \(C\) with \(W\) or another paid left factor, nor \(M\) with its matching paid right-hand side. The Hadamard mask does not contain the missing matrix product. Appending \(M\) as additional right-hand-side columns still increases billed arithmetic.

Therefore one \(CM\) per active layer is genuinely incremental. Under base-48/cap-49, the depth-wide exact all-distinct adjoint route is closed.

### Final graph update for this branch

- **CM fusion premise -- closed/no-go for the proven circuit.** Evidence: operand mismatch \((W,B)\ne(C,M)\); Hadamard postprocessing cannot recover a generic matrix product.
- **Depth-wide exact H3 adjoint under cap-49 -- closed.** Cost: ten to fourteen new equivalents versus one available.
- **One-layer support reopen -- open but unsupported.** Pass only if the entire task-weighted residual is algebraically localized to a single birth.
- **Cross-layer aggregate reopen -- open.** Pass only with an exact identity evaluating \(\sum_\ell S_\ell(x_\ell,M_\ell)\) using at most one paid product for unrelated layer operands.
- **Relaxed adjusted-economics replacement -- policy-conditional.** It can be favorable near full 18.7% retained gain, but it does not satisfy the current cap.

### Current next question

The exact all-distinct adjoint route is now mathematically characterized and deployment-closed under the standing cap. The remaining high-value question is whether a different source family has a scalar-at-birth contraction that uses the already paid \(WB\) operands, rather than requiring the independent \(CM\) product.

## Checkpoint 18: cap-compatible scalar trace sketch for the all-distinct future-costate action

Although the exact \(CM\) product is cap-closed, only one scalar contraction of its diagonal is required. In the first line of the source formula, put

\[
w_i=\beta_ix_i,qquad D_w=\operatorname{diag}(w),
\]

and isolate

\[
t=\sum_iw_i c_i^TMc_i.
\]

For symmetric \(M\), define

\[
H=\operatorname{sym}\left[(CM)^TD_wC\right].
\]

Then \(t=\operatorname{tr}H\). For independent Rademacher probes \(z_s\),

\[
\boxed{
\widehat t_k=\frac1k\sum_{s=1}^k
(CMz_s)^TD_w(Cz_s)
}
\]

is unbiased because \(\mathbb E[z_sz_s^T]=I\). It needs only the three matrix-vector actions

\[
Mz_s,\qquad C(Mz_s),\qquad Cz_s,
\]

and no dense state.

For symmetric \(H\), the exact conditional Rademacher variance is

\[
\boxed{
\operatorname{Var}_z(\widehat t_k\mid H)
=\frac2k\|\operatorname{offdiag}H\|_F^2.
}
\]

All other terms in the arbitrary-query source formula are exact \(O(n^2)\) operations, so this is the only sketch error.

### Exact correlation and cost gate

Let \(t\) denote the desired centered scalar contribution across the deployment distribution and assume independent probe noise. Then

\[
q^2=\frac{\operatorname{Var}(t)}
{\operatorname{Var}(t)+\frac2k\mathbb E\|\operatorname{offdiag}H\|_F^2}.
\]

The requirement \(q>.925\) is exactly

\[
\frac2k\mathbb E\|\operatorname{offdiag}H\|_F^2
<\left(\frac1{.925^2}-1\right)\operatorname{Var}(t)
=.16873\,\operatorname{Var}(t).
\]

Therefore

\[
k>k_{\min}:=
\frac{2\mathbb E\|\operatorname{offdiag}H\|_F^2}
{.16873\,\operatorname{Var}(t)}.
\]

Using a conservative count of three \(n^2\) matrix-vector operations per probe, \(L\) active layers cost at most

\[
N_{\rm mm}\simeq\frac{3kL}{n}
\]

dense-equivalents. Under one available equivalent,

\[
k\le k_{\max}:=\frac{n}{3L}.
\]

At \(n=1024\) and \(L=10\), \(k_{\max}\simeq34\), so a 32-probe scalar sketch fits the cap. The joint symbolic pass gate is

\[
\boxed{k_{\min}\le k_{\max}.}
\]

There is no worst-case guarantee: \(\operatorname{tr}H\) can be zero while its off-diagonal Frobenius norm is arbitrarily large. Conversely, diagonal dominance or a strong task-level trace signal can make a small probe count sufficient. This is an algebraic trace sketch, not naive Monte Carlo sampling of network outputs; the previously measured MC-output no-go does not by itself determine this variance ratio.

Full \(n\)-probe orthogonal probing recovers the trace exactly but exceeds the cap. Low-rank exactness is another special case: if the future mask has certified rank \(r_M\), the action can be evaluated in \(O(r_Mn^2)\), but the current reverse-mask audit says generic masks are full rank.

### Graph-ready split nodes

- **Hutchinson scalar identity -- proved.** Criterion: the boxed unbiased estimator and exact off-diagonal variance formula.
- **One-equivalent cost -- conditional pass.** Criterion: \(3kL/n\le1\); 32 probes fit ten layers at width 1024 under the conservative convention.
- **.925 correlation -- open scalar premise.** Criterion: the boxed \(k_{\min}\le k_{\max}\) ratio after residualizing old features.
- **Worst-case guarantee -- closed/no-go.** Evidence: zero-trace, large-off-diagonal symmetric matrices.
- **MC-output duplicate -- not implied.** The probe randomness estimates an algebraic trace; compare variances explicitly before importing the old sampling no-go.

### Revised next question

Can \(\|\operatorname{offdiag}H\|_F^2\) be bounded analytically by the already known task-weighted source energy, or reduced deterministically with an old-feature control variate, strongly enough to certify \(k_{\min}\le32\) without a numerical campaign?

## Checkpoint 19: local AQ-costate factorization versus future-dark support

There is a cheaper exact representation when the H3 source is consumed locally by the affine--quadratic state. Let

\[
O_\ell:\mathcal H_3\to\mathcal H_{\le2}
\]

be the local H3-to-AQ consumer and let \(\lambda_{\ell+1}\in\mathcal H_{\le2}\) be the downstream AQ costate. The local source query is then, by definition,

\[
q_\ell=O_\ell^*\lambda_{\ell+1}.
\]

It can be paired at the ReLU surface as a cubic jet of the schematic form

\[
\frac16\mathbb E\left[
((I-P_{\rm old})\lambda_i)\,\delta'(g_i)h_i^3
\right],
\]

using the linear/quadratic coefficients of \(\lambda_i\) and local latent contractions. In this factorized case no physical \(CM\) product is required; the full AQ costate is carried backward by the existing truncated gate transfer and weight-bank mix.

The exact obstruction is a future-dark query component. Define

\[
\Pi_{O^*}=O^*(OO^*)^+O,
\qquad
q_\perp=(I-\Pi_{O^*})q.
\]

Then:

\[
q_\perp=0
\]

is necessary and sufficient for exact factorization through the local AQ costate. If

\[
\langle q_\perp,S_{AD}\rangle\ne0,
\]

the local AQ quotient misses a source component that becomes visible only after future H3 transport; the arbitrary-query \(CM\) oracle is then required.

The certified 18.7% replay favors the factorized interpretation. Its banked truth-minus-estimated dK21/dK22/dK31 planes are injected directly into the covariance tangent at every L5--14, and attribution is linear in those local injections. It is not a receipt for an H3 birth transported autonomously until a later observation. Hence the frozen certified query should lie in \(\operatorname{range}O^*\), subject to the displayed symbolic projection audit.

This does not solve source provenance: the banked injected planes are truth-minus-estimated quantities. A deployable cubic surface formula must still reproduce their task-weighted action from deployed white-box state.

### Graph-ready split nodes

- **Local AQ-costate cubic-jet factorization -- conditional pass.** Criterion: \(q_\perp=0\) after old-ownership subtraction. Consequence: no incremental \(CM\) product.
- **Certified replay query locality -- supported by provenance, not fully proved.** Evidence: direct per-layer covariance injections. Symbolic quotient projection remains binding.
- **Future-dark component -- open for a new self-consistent provider.** Criterion: nonzero \(\langle q_\perp,S_{AD}\rangle\). If present, return to the cap-closed arbitrary-query route or the scalar sketch.
- **Truth-minus-estimated source acquisition -- open.** Even a cheap query cannot evaluate an unavailable banked source.
- **Hutchinson route -- fallback only.** Needed only if \(q_\perp\ne0\) and its variance/cost gate passes.

### Revised next question

Under the explicit ReLU gate maps, prove the identity between the local cubic-jet surface pairing and the arbitrary-query formula \(S(x,M)\) on \(\operatorname{range}O^*\). This would close the quotient audit symbolically and isolate truth-minus-estimated source acquisition as the sole remaining blocker for the frozen 18.7% target.

## Checkpoint 20: local cubic-jet equality proved; computation is still generically dense

For arbitrary reverse arguments \(x,M\), define the cubic test polynomial

\[
f_{x,M}(h)=(x^Th)(h^TMh).
\]

Its third derivative is constant:

\[
\partial_{abc}f_{x,M}
=2(x_aM_{bc}+x_bM_{ac}+x_cM_{ab}).
\]

For a symmetric cubic source \(S\),

\[
\boxed{
\frac16\langle S,\nabla^3f_{x,M}\rangle
=S(x,M).
}
\]

Thus the arbitrary-query contraction is exactly the local cubic Edgeworth/surface action of an AQ test polynomial. Substituting the factored all-distinct ReLU source recovers the boxed formula from Checkpoint 11. This proves local AQ ownership and \(q_\perp=0\) on the structured same-site source manifold; collision ownership is removed by all-distinct support or \((I-P_{\rm old})\).

This identity does **not** by itself remove the arithmetic needed to evaluate the Gaussian surface contraction. For generic dense \(C,M\), the term \(\operatorname{diag}(CMC^T)\) remains.

An exact sub-dense escape exists under structural rank. If

\[
C=ZL^T,
\qquad Z,L\in\mathbb R^{n\times k},
\]

then

\[
\operatorname{diag}(CMC^T)
=\operatorname{diag}\left[Z(L^TML)Z^T\right].
\]

It costs

\[
O(n^2k+nk^2)
\]

rather than \(O(n^3)\), with normalized dense-equivalent cost approximately

\[
k/n+(k/n)^2
\]

per layer.

For a gated covariance represented as a Schur product,

\[
C=C_{\rm cov}\odot C_{\rm gate},
\]

the general rank bound is

\[
\operatorname{rank}C
\le \operatorname{rank}C_{\rm cov}\,
\operatorname{rank}C_{\rm gate}.
\]

Two factors of rank \(\sqrt n\) therefore permit generic full rank \(n\). A useful exact low-rank pass needs a proved common/rank-one gate factor or another bounded Schur-rank identity on the lawful source manifold. Ambient AQ factorization alone is insufficient.

### Graph-ready split nodes

- **Cubic surface/arbitrary-query equality -- proved.** Criterion: the boxed third-derivative identity. Consequence: same-site AQ ownership is settled.
- **Generic physical CM cost -- remains closed under cap.** Ownership factorization does not imply arithmetic fusion.
- **Structural-rank exact escape -- conditional.** Criterion: exhibit \(C=ZL^T\) with total depth cost \(\sum[k_\ell/n+(k_\ell/n)^2]\le1\).
- **Schur-rank premise -- open and likely stringent.** Product-rank upper bounds do not prove low realized rank; two moderate-rank factors can be full.
- **Hutchinson trace sketch -- active fallback.** It requires no rank premise but must pass the exact off-diagonal noise gate.

### Revised next question

Can the lawful all-distinct source factor \(C\) be shown analytically to have bounded Schur rank after old-span subtraction, or does the broad-spectrum/identity-specific nature of the certified variance channel force generic full rank? If full rank, only the scalar trace-sketch variance ratio can reopen this path under cap-49.

## Checkpoint 21: corrected branch-specific cost accounting and optimal trace controls

The earlier blanket cap closure was too conservative. Under the synchronized r45/10 accounting supplied by the reverse-adjoint lane, there are two distinct branches:

1. If the reverse intermediate \(H=WBW^T\) is reused from the paid circuit, its existing cost is 22.83 equivalents. Ten exact \(CM\) source products bring the total to

   \[
   22.83+10=32.83,
   \]

   which passes the r45 envelope.
2. If \(H\) must be rebuilt, the additional 20 equivalents give

   \[
   22.83+20+10=52.83,
   \]

   which fails.

Thus the exact all-distinct adjoint route is **alive conditional on an explicit H-reuse receipt**. The no-direct-fusion result remains true: every \(CM\) is a genuine product, but those products fit the correct reused-H branch budget.

For the structural-rank alternative \(C=ZL^T\), the corrected normalized per-layer cost is

\[
x+2x^2,
\qquad x=k/n.
\]

Under worst-branch slack .617, the factor route passes when

\[
x+2x^2\le.617,
\]

or approximately \(x\le.359\), i.e. \(k\lesssim367\) at width 1024. A \(k=O(\sqrt n)\) factor is comfortably inside this bound.

### Optimal old-feature control for the scalar sketch

Let \(H_1,\ldots,H_p\) be cheap old-feature matrices with exactly known traces. The controlled estimator is

\[
\widehat t_{\alpha}
=\sum_j\alpha_j\operatorname{tr}H_j
+\frac1k\sum_s z_s^T
\left(H-\sum_j\alpha_jH_j\right)z_s.
\]

It is unbiased and has exact variance

\[
\frac2k\left\|operatorname{offdiag}
\left(H-\sum_j\alpha_jH_j\right)\right\|_F^2.
\]

The minimum-variance coefficient solves

\[
G\alpha=b,
\qquad
G_{ij}=\langle\operatorname{off}H_i,\operatorname{off}H_j\rangle,
\quad
b_i=\langle\operatorname{off}H,\operatorname{off}H_i\rangle.
\]

Hence the sharp noise numerator is the off-diagonal Schur residual after projecting onto the old matrix-action span. This couples double-count subtraction and sketch variance correctly.

At source correlation .925, the maximum retained explained fraction is \(.925^2=.8556\). Applied to the certified 18.7% ceiling, this is about a 16.0% relative raw effect, or \(4.71\times10^{-9}\) at the current raw. It is useful but cannot alone reach raw \(<10^{-8}\).

### Corrected graph-ready nodes

- **Exact CM path with H reuse -- conditional pass.** Cost 32.83. Binding evidence: exact package/circuit receipt that \(H\) is already paid and available.
- **Exact CM path with H rebuild -- closed.** Cost 52.83 exceeds r45.
- **Structural-rank path -- conditional.** Pass inequality \(x+2x^2\le.617\).
- **Controlled Hutchinson path -- open.** Use the off-diagonal Schur residual, not the unconditioned \(H\), in the .925 gate.
- **Standalone raw target -- closed.** Even perfect recovery of the 18.7% chain does not reduce \(2.94\times10^{-8}\) below \(10^{-8}\); this lane must stack with larger co-adapted/raw mechanisms.

### Revised next question

Obtain the explicit H-reuse and source-provenance receipts. With both, the exact ten-layer all-distinct adjoint is deployment-feasible; without H reuse, fall back to a certified low-rank factor or controlled scalar sketch.

## Checkpoint 22: source-specific rank clarification

For the all-distinct source used throughout this lane,

\[
c_{ij}=g_jC_{ij},
\]

so the source factor matrix is

\[
C_{\rm source}=C_{\rm cov}\operatorname{diag}(g).
\]

This is a column scaling, not a generic high-rank Schur gate. Therefore

\[
\operatorname{rank}C_{\rm source}
\le\operatorname{rank}C_{\rm cov},
\]

with equality when all gate factors are nonzero on the covariance range.

The structural-factor pass \(k/n\le.359\) is thus exactly a deployed covariance-rank receipt:

\[
\operatorname{rank}C_{\rm cov}\le367
\]

at width 1024 is sufficient for the corrected exact low-rank cost branch. If the deployed covariance is generically full rank, the factor escape remains closed; if an exact input/feature factor of rank at most 367 is already resident, it passes without invoking a gate-Schur-rank assumption.

### Graph-ready correction

- **Gate-rank multiplier concern -- inapplicable to this specific source.** The gate is diagonal column scaling.
- **Covariance-rank receipt -- open.** Criterion: exact lawful factor rank at most 367, not numerical/effective rank unless approximation error is separately bounded.
- **Full-rank covariance branch -- unchanged.** Use exact CM with H reuse or the controlled trace sketch.

## Checkpoint 23: frozen 18.7% ceiling versus goal-scale self-consistent capacity

The 18.7% number is a sharp ceiling only for the certified **frozen-tangent replay**. It is not an upper bound on a self-consistent source whose deploy frame and gates co-adapt through Taylor order two.

Under the fixed portfolio convention, let

\[
d=(1-q_s^2)^{1/2},\qquad
r_v=C_3d^3+\eta,
\qquad
a_v=\phi\,q_v^2,
\qquad
q_v^2\ge1-r_v^2.
\]

The exact raw-target sufficient condition is

\[
1.7316\left[1-\phi+\phi(C_3d^3+\eta)^2\right]
+R_{\rm other}\le.07453,
\]

where

\[
R_{\rm other}=.4370(1-a_3)+.10313(1-a_4).
\]

Solving for the allowed third-order constant gives

\[
\boxed{
C_3\le
\frac{
\sqrt{\left[(.07453-R_{\rm other})/1.7316-(1-\phi)\right]/\phi}
-\eta}{d^3}.
}
\]

The expression under the square root must be positive. At \(q_s=.78\), \(\phi=1\), \(R_{\rm other}=0\), and \(\eta=0\), this reduces to

\[
C_3\le.8466.
\]

It implies capture of at least about 95.7% of the variance channel, which is goal-scale and can exceed the frozen 18.7% replay value. Checkpoints 13 and 15 give a tensor-free construction for Taylor orders one and two; the signed Schur/Pad\'e gate controls the remaining \(C_3d^3\) term.

Therefore the complete mathematical route is:

1. label-free source provenance in the task-weighted quotient;
2. exact/reused-H all-distinct adjoint and forward tangent through order two;
3. coefficient fidelity \(\phi\) and other-channel capture sufficient to leave positive target slack; and
4. signed stable feedback satisfying the boxed \(C_3\) inequality without PSD violation or D11 double count.

### Corrected graph-ready nodes

- **Frozen 18.7% ceiling -- applies only to fixed Jacobian replay.** Keep as the empirical first-order receipt.
- **Self-consistent raw capacity -- conditionally goal-scale.** Criterion: boxed fidelity/C3 inequality.
- **Order-two construction -- proved algebraically, source inputs open.** Forward tangent plus reverse adjoint, with reused local Hessians/source derivatives.
- **C3 signed stability -- open conditional gate.** Use the earlier multi-pole numerical-range and PSD/counterterm conditions.
- **Raw<1e-8 path -- nonempty mathematically.** It is not certified until provenance, H reuse, fidelity, and signed remainder all pass.

## Checkpoint 24: coefficient-fidelity fail-fast threshold

Even with a perfect post-resummation variance endpoint \(q_v=1\), zero remainder, and exact other channels, the target inequality requires

\[
1.7316(1-\phi)\le.07453.
\]

Therefore

\[
\boxed{\phi\ge1-.07453/1.7316\simeq.95696.}
\]

Any coefficient fidelity below about 95.7% kills the raw-target path regardless of source correlation or signed feedback quality.

At \(q_s=.78\), \(\eta=0\), and exact other channels, the coupled \((\phi,C_3)\) frontier is severe:

\[
\begin{array}{c|c}
\phi & C_{3,\max}\\ \hline
1.00 & .8466\\
.99 & \approx .746\\
.98 & \approx .626\\
.97 & \approx .473\\
.96 & \approx .230
\end{array}
\]

Thus .8466 is valid only at essentially perfect coefficient fidelity. Every residual in k3/q4 or nonzero \(\eta\) tightens the frontier further.

### Graph-ready split node

- **Coefficient-fidelity gate -- open, fail-fast.** Kill the goal-scale variance path immediately if \(\phi<.95696\) even under optimistic exact-other-channel assumptions; otherwise use the full boxed frontier rather than the \(\phi=1\) constant.

## Checkpoint 25: exact typed-source provenance receipt

For each typed local source \(\tau\in\{21,22,31\}\), let

\[
S_\ell^\tau(\theta),\qquad
\theta_T=\theta_D+e.
\]

Define the residual against the deployed source and its first co-adapted motion by

\[
R_\ell^\tau
=dK_{\ell,\rm cert}^\tau
-S_\ell^\tau(\theta_D)
-DS_\ell^\tau(\theta_D)[e].
\]

Taylor's theorem gives the exact decomposition

\[
\boxed{
R_\ell^\tau
=\int_0^1(1-t)
D^2S_\ell^\tau(\theta_D+te)[e,e]dt
+R_{\ell,\rm hidden}^\tau,
}
\]

where \(R_{\rm hidden}\) is, by definition, any truth-only moment/source component not parameterized by the declared deploy state \(\theta\).

With the pulled causal Bellman masks \(Q_\ell^\tau\), the only provenance error relevant to the scalar task is

\[
\boxed{
E_Q=\sum_{\ell,\tau}\langle Q_\ell^\tau,R_\ell^\tau\rangle.
}
\]

It obeys

\[
|E_Q|\le
\frac12\sum_{\ell,\tau}
\|Q_\ell^\tau\|
\sup_{0\le t\le1}\|D^2S_\ell^\tau(\theta_D+te)\|
\|e_\ell\|^2
+\left|\sum_{\ell,\tau}
\langle Q_\ell^\tau,R_{\ell,\rm hidden}^\tau\rangle\right|.
\]

Adjoint reuse is exact:

\[
\langle Q,DS[e]\rangle=\langle DS^*Q,e\rangle.
\]

Thus the source directional contractions derived in Checkpoint 15 are exactly the required first co-adapted provenance term.

### Mandatory producer/consumer distinction

The compact \(\delta G\) masks represent the consumer/frame derivative. A self-generated supply also moves through the producer:

\[
C_\theta+C_k k_\theta,
\]

where \(C_kk_\theta\) contains the earlier \(\delta A\)-on-\(T\) and \(DS_2\) terms. If the certified \(dK\) is defined after transport, these producer terms must be folded into \(DS\) before \(R\) is formed. Otherwise reachable co-adaptation would be mislabeled as a truth-only residual.

Normalize \(|E_Q|\) by the ideal variance-channel endpoint correction to obtain the additive \(\eta\) used in the fidelity/C3 gate. This turns source provenance into a quantitative remainder rather than a binary assertion.

### Graph-ready split nodes

- **Typed source Taylor partition -- proved.** Criterion: the boxed integral remainder for each 21/22/31 source.
- **Task-weighted provenance receipt -- proved form, open value.** Criterion: bound \(E_Q\), not full plane error.
- **Producer motion ownership -- mandatory.** Include \(\delta A\)-on-\(T\) and \(DS_2\) before assigning hidden-source status.
- **Hidden truth-only source -- open.** Criterion: nonzero residual after the complete declared state/source tangent; feeds \(\eta\).
- **Adjoint DS reuse -- proved.** No new tensor or dense derivative product is required.

## Checkpoint 26: closed norm bound for the all-distinct source remainder

Before squarefree projection and cyclic symmetrization, write

\[
B=\sum_i\beta_i e_i\otimes c_i\otimes c_i.
\]

The all-distinct source is \(S=P_{\rm sf}(B+\mathrm{cyclic}_2+\mathrm{cyclic}_3)\). Both the squarefree projection and index permutations are norm nonexpansive, hence

\[
\|S''\|_F\le3\|B''\|_F.
\]

Along any twice differentiable frame path,

\[
\begin{aligned}
B''=\sum_i e_i\otimes\bigl[
&\beta_i''c_i\otimes c_i
+2\beta_i'(c_i'\otimes c_i+c_i\otimes c_i')\\
&+\beta_i(c_i''\otimes c_i+2c_i'\otimes c_i'+c_i\otimes c_i'')
\bigr].
\end{aligned}
\]

Since the first-mode vectors \(e_i\) are orthogonal,

\[
\boxed{
\|S''\|_F\le3\left(\sum_i a_i^2\right)^{1/2},
}
\]

where

\[
a_i=
|\beta_i''|\|c_i\|^2
+4|\beta_i'|\|c_i\|\|c_i'\|
+2|\beta_i|\left(\|c_i'\|^2+\|c_i\|\|c_i''\|\right).
\]

Therefore every task mask obeys

\[
|\langle Q,S''\rangle|
\le3\|Q\|_F\left(\sum_i a_i^2\right)^{1/2}.
\]

For the lawful source factor \(c_i=g\odot C_i\),

\[
c_i'=g'\odot C_i+g\odot C_i',
\]

\[
c_i''=g''\odot C_i+2g'\odot C_i'+g\odot C_i''.
\]

The derivatives of \(g\) and \(\beta\) are the existing shifted Gaussian gate masks. Along an affine secant in \((\beta,c)\), \(\beta''=c''=0\) and the bound simplifies to the two middle quadratic terms.

Inserting half the path supremum of this bound into Checkpoint 25 gives an explicit sufficient contribution to \(\eta\). The bound can be loose because it discards task cancellations, but it is label-free and tensor-free.

### Graph-ready split node

- **All-distinct source-curvature bound -- proved.** Criterion: evaluate/bound the displayed gate and covariance tangent norms. Edge: supplies the observable quadratic part of \(\eta\); any remaining hidden-source action is separate.

## Checkpoint 27: exact hidden/provenance remainder slack

Solving the full target inequality for the additive normalized provenance remainder gives

\[
\boxed{
\eta_{\max}=
\sqrt{\frac{(.07453-R_{\rm other})/1.7316-(1-\phi)}{\phi}}
-C_3d^3.
}
\]

The path is feasible only if \(\eta_{\max}\ge0\). At \(q_s=.78\), \(\phi=1\), and exact other channels, the total allowed endpoint residual radius is approximately .20746. Hence

\[
\begin{array}{c|c}
C_3&\eta_{\max}\\ \hline
.8466&\approx0\\
.80&\approx.0114\\
.75&\approx.0237
\end{array}
\]

Thus a signed constant merely below .8466 is not enough in practice: unless \(C_3\) is substantially smaller, the observable source-curvature bound plus the truth-only hidden receipt must be around one percent of the ideal variance correction.

### Graph-ready split node

- **Provenance/hidden eta gate -- open, fail-fast.** Normalize the Checkpoint-25 task receipt and require it below the boxed \(\eta_{\max}\). Any other-channel residual or fidelity loss tightens the bound.

## Checkpoint 28: analytic coefficient fidelity inside the declared closure

The all-distinct source coefficient need not be learned. Three independent algebraic normalizations fix it:

1. the cyclic source definition
   \(S_{ijk}=\beta_ic_{ij}c_{ik}+\mathrm{cyclic}\);
2. the polynomial check
   \(S(x,xx^T)=3\sum_i\beta_ix_i[(c_i^Tx)^2-\sum_jc_{ij}^2x_j^2]\); and
3. the Edgeworth surface identity
   \((1/6)\langle S,\nabla^3f_{x,M}\rangle=S(x,M)\).

If the deployed reverse adjoint is the exact derivative of the declared closure, these identities imply

\[
\phi=1
\]

for the analytic source action. There is no cross-net fitted scale whose transfer fidelity must be estimated.

Disagreement with the true network should be assigned to the typed source curvature/hidden receipt \(\eta\), not silently absorbed by refitting the analytic coefficient. An empirical rescaling would reintroduce the severe \(\phi\ge.95696\) gate and can confound a structural source error with coefficient error.

### Graph-ready split node

- **Analytic coefficient fidelity -- proved within closure.** Criterion: exact combinatorial/Hermite normalization and exact deployed adjoint. Set \(\phi=1\) internally; route closure mismatch to \(\eta\). External truth validity remains open.

## Checkpoint 29: floating-point fidelity gate for the signed contraction

For a standard f32 dense product,

\[
|\operatorname{fl}(CM)-CM|
\le\gamma_n|C||M|,
\qquad
\gamma_n=\frac{nu}{1-nu}.
\]

At \(n=1024\) and f32 unit roundoff \(u\simeq5.96\times10^{-8}\),

\[
\gamma_n\simeq6.1\times10^{-5}.
\]

Define the source-action cancellation condition number \(\kappa_S\) as the absolute-value forward-error bound for every term in the arbitrary-query formula divided by the magnitude of the exact signed scalar action. Then

\[
\frac{|\widehat S-S|}{|S|}
\lesssim\gamma_n\kappa_S+O(u).
\]

To keep numerical coefficient fidelity above the absolute fail-fast floor \(.95696\), it is sufficient that

\[
\gamma_n\kappa_S<.04304,
\]

or approximately

\[
\boxed{\kappa_S\lesssim705.}
\]

The source is signed, so this is a real premise: large cancellation can amplify a small matrix-product backward error. If the condition bound exceeds this threshold, compensated accumulation or a higher-precision source contraction is required even though the analytic coefficient is exact.

### Graph-ready split node

- **Numerical coefficient fidelity -- open norm receipt.** Pass f32 if \(\kappa_S\lesssim705\); otherwise cost a compensated/f64 variant and reapply utilization economics.

## Checkpoint 30: exact Hermite-diagram ownership of the tree source

Let \((Z_i,Z_j,Z_k)\) be a standardized Gaussian triple and let each centered scalar nonlinearity have Hermite expansion

\[
f_i(Z_i)=\sum_{m\ge1}\frac{c_{i,m}}{m!}H_m(Z_i),
\qquad
c_{i,m}=\mathbb E[f_i(Z_i)H_m(Z_i)].
\]

Write \(a,b,c\) for the numbers of Wick edges between pairs \((i,j),(i,k),(j,k)\). The exact connected-diagram expansion is

\[
\boxed{
\kappa_3(f_i,f_j,f_k)=
\sum_{\substack{a,b,c\ge0\\
a+b>0,\ a+c>0,\ b+c>0}}
\frac{c_{i,a+b}c_{j,a+c}c_{k,b+c}}{a!b!c!}
\rho_{ij}^a\rho_{ik}^b\rho_{jk}^c.
}
\]

The three total-edge-two terms are

\[
(a,b,c)=(1,1,0),(1,0,1),(0,1,1).
\]

They are exactly the three spanning trees. After restoring means and variances, these are

\[
\beta_i c_{ij}c_{ik}+\beta_jc_{ji}c_{jk}+\beta_kc_{ki}c_{kj}.
\]

Therefore the all-distinct source used in this lane is not an ad hoc fit: it is precisely the complete DTOT2/tree sector of the Gaussian connected expansion. Every omitted Gaussian diagram has total edge count at least three, including the triangle and all multiedge corrections.

Define the absolute connected majorant

\[
\mathcal M_{ijk}(r_0)=
\sum_{\rm connected}
\frac{|c_{i,a+b}c_{j,a+c}c_{k,b+c}|}{a!b!c!}
r_0^{a+b+c}
\]

for any \(r_0<1\) where it is finite. If all pair correlations satisfy \(|\rho|\le r<r_0\), then the omitted Gaussian source obeys

\[
\boxed{
|R_{\ge3}|
\le\left(\frac r{r_0}\right)^3\mathcal M_{ijk}(r_0).
}
\]

This is a rigorous label-free contribution to the hidden-source \(\eta\) bound. It explains algebraically why DTOT2 retains the full tree sector and why triangle motifs are higher order. It does not bound non-Gaussianity of the preactivation law; that remainder must be recorded separately.

### Graph-ready split nodes

- **Gaussian tree-source ownership -- proved.** Criterion: the boxed Hermite diagram expansion; DTOT2 source equals all edge-two connected diagrams.
- **Gaussian higher-diagram remainder -- bounded conditionally.** Criterion: correlation envelope \(r<r_0\) and finite ReLU Hermite majorant.
- **Non-Gaussian preactivation remainder -- open.** It is not part of the Gaussian diagram tail and feeds the hidden truth-only receipt.

## Checkpoint 31: certified-supply provenance audit -- no equality receipt

The banked certified dK21/dK22/dK31 planes are full truth/tangent supply deltas. The local all-distinct tree \(S_{AD}(\beta_D,C_D)\) is one modeled reset source. No current algebraic receipt proves

\[
dK_{\ell,\rm cert}=O_\ell S_{AD,\ell}.
\]

The comparison must be made at the same synchronized pre-reset or post-reset cut and in the complete old/interface quotient. Define

\[
R_\ell=dK_{\ell,\rm cert}-O_\ell S_{AD,\ell}
\]

only after aligning those cuts and subtracting all already owned actions. The weakest valid endpoint credit gate is

\[
\boxed{
\left|\sum_\ell\langle Q_\ell,R_\ell\rangle\right|
\le\epsilon
\left|\sum_\ell\langle Q_\ell,dK_{\ell,\rm cert}\rangle\right|.
}
\]

Exact certified credit has \(\epsilon=0\). The current portfolio value gate requires at least 86.2% retention, hence

\[
\epsilon\le.138.
\]

The complete reverse interface must include K4 and crossed-K22 costates in addition to K31. K31-only orthogonality can miss or duplicate collision faces. Ambient all-distinct support does not by itself prove novelty because CP3/depth transport can carry its signed orientation into an old interface.

The Hermite tree theorem remains valid: it proves what \(S_{AD}\) represents. It does not prove how much of the certified truth-minus-estimated supply that source owns.

### Corrected graph-ready nodes

- **Certified-source equality -- open/no receipt.** Do not credit the frozen 18.7% or goal-scale variance channel to \(S_{AD}\) without the synchronized endpoint gate.
- **Complete-interface residualization -- mandatory.** Include K31, K4, uncrossed and crossed K22, plus all old CP3/depth actions.
- **86.2% value-retention gate -- open, earliest kill.** Require \(\epsilon\le.138\) before cost, C3, or raw-capacity claims.
- **Hermite tree ownership -- still proved locally.** It supplies a candidate source, not certified global value.

## Checkpoint 32: economic retention is weaker than goal-scale provenance

The \(\epsilon\le.138\) condition preserves at least 86.2% of the certified source value and is an economic/portfolio screen. It is not sufficient for raw \(<10^{-8}\).

If the synchronized endpoint residual acts as coefficient/value loss, the absolute fidelity floor requires

\[
\epsilon\le1-.95696=.04304
\]

even under perfect endpoint correlation and exact other channels. If it instead enters as an additive normalized remainder, it must satisfy the much tighter \(\eta_{\max}\) gate; at \(C_3=.8\), \(\phi=1\), and exact other channels this is approximately .0114.

The source compared to the certified dK must be the complete order-two assembled provider, schematically

\[
\widehat{dK}^{(2)}
=O S_D+O\,DS_D[e]+(DO[e])S_D
+\text{producer terms }(\delta A\text{-on-}T,DS_2),
\]

with K31, K4, and both K22 interfaces synchronized. Applying the retention gate to bare \(S_D\) would charge reachable co-adaptation to the hidden residual.

### Graph-ready split nodes

- **86.2% economic screen -- open.** Require \(\epsilon\le.138\) to keep the branch.
- **Goal-scale provenance -- stricter open gate.** Require coefficient-like loss \(\le.04304\) or additive residual \(\le\eta_{\max}\).
- **Complete order-two assembled provider -- mandatory definition.** Bare-source comparisons are invalid for self-consistent capacity.

## Checkpoint 33: explicit shifted-ReLU Hermite coefficients

For

\[
f(z)=(m+\sigma z)_+,
\qquad a=m/\sigma,
\]

Stein's identity gives

\[
c_1=\mathbb E[f(Z)H_1(Z)]=\sigma\Phi(a),
\]

and distributional differentiation gives, for \(r\ge2\),

\[
\boxed{
c_r=\mathbb E[f(Z)H_r(Z)]
=\sigma(-1)^{r-2}H_{r-2}(a)\varphi(a).
}
\]

In particular,

\[
c_2=\sigma\varphi(a).
\]

The edge-two tree coefficient is therefore

\[
c_{i,2}c_{j,1}c_{k,1}\rho_{ij}\rho_{ik}
=\frac{\varphi(a_i)}{\sigma_i}
\Phi(a_j)\Phi(a_k)C_{ij}C_{ik},
\]

which is exactly

\[
\beta_i g_jg_kC_{ij}C_{ik}.
\]

This fixes the tree normalization directly from the ReLU law. It also makes every higher Gaussian diagram and the majorant in Checkpoint 30 an explicit label-free function of \((m,v,C)\). Only non-Gaussian preactivation corrections and truth/deploy frame mismatch remain outside that Gaussian source hierarchy.

### Graph-ready split node

- **Shifted-ReLU Hermite source coefficients -- proved.** Criterion: Stein/distributional derivative identity. Consequence: tree coefficient and Gaussian tail are analytically normalized with no fit.

## Checkpoint 34: full self-consistent cost/interface gate

The 32.83 cost receipt covers the ten exact \(CM\) source actions when the reverse \(H\) bank is reused. A complete order-two self-consistent closure also needs the forward tangent \(e\). Its branch-specific dense cost is

\[
\boxed{
C_{\rm dense}=22.83+10
+20\,\mathbf1_{\{H\ \mathrm{not\ reused}\}}
+20\,\mathbf1_{\{e/JVP\ \mathrm{not\ reused}\}}.
}
\]

The local gate-Hessian and \(DS[e]\) actions are already piggybacked and do not add dense products. Under an r45 envelope:

\[
\begin{array}{c|c}
\text{reuse branch}&C_{\rm dense}\\ \hline
H\text{ and }e\text{ both reused}&32.83\ \text{pass}\\
\text{exactly one reused}&52.83\ \text{fail}\\
\text{neither reused}&72.83\ \text{fail}
\end{array}
\]

The structural-factor source path replaces the fixed ten-product term by

\[
\sum_{\ell=5}^{14}(x_\ell+2x_\ell^2),
\qquad x_\ell=k_\ell/n.
\]

The mandatory interface is the synchronized complete set of K31, K4, uncrossed K22, and crossed K22 costates. A K31-only cost or ownership receipt is insufficient.

### Graph-ready split nodes

- **Frozen/source-action cost -- conditional pass at 32.83.** Requires reverse-H reuse.
- **Self-consistent order-two cost -- stricter conditional pass.** Requires both reverse-H and forward-e/JVP reuse under r45.
- **Complete interface -- mandatory.** K31+K4+both K22 faces, synchronized pre/post reset.
- **Variance arm necessity -- proved from capacity ladder.** Even exact k3/q4 require \(a_v\ge.95696\); the variance/all-distinct arm cannot be replaced by V3/V4 capacity.

## Checkpoint 35: non-Gaussian hidden-source non-identifiability

Let \(A,B\) be independent Rademacher variables and set

\[
C=AB.
\]

Then \((A,B,C)\) has zero means, unit variances, zero pairwise covariances, and the same one-dimensional marginals as three independent Rademacher variables. However,

\[
\mathbb E[ABC]=1
\]

for the dependent triple and zero for the independent triple.

For a Rademacher variable,

\[
X_+=(1+X)/2,
\]

so its centered ReLU output is \(X/2\). Hence the all-distinct centered ReLU third cumulant is

\[
1/8
\]

for \((A,B,AB)\) and zero for the independent triple.

Both laws have identical \((m,v,C)\), identical marginal traffic probabilities, and identical pairwise gate data. The Gaussian DTOT2/tree source sees zero pair correlations and predicts zero in both cases.

Therefore no estimator using only

\[
(m,v,C,g,\beta)
\]

can recover or uniformly bound the signed non-Gaussian all-distinct H3 source. At least one signed H3/interface action is information-theoretically mandatory. This is a terminal proof for the restricted-state hidden-source node; it does not say the actual network remainder is large, only that low-order state cannot certify it small.

### Graph-ready split nodes

- **Low-order non-Gaussian H3 recovery -- closed/no-go.** Evidence: pairwise-independent Rademacher triple with nonzero triple cumulant.
- **Signed H3/interface statistic -- mandatory for certification.** It may be a complete future-costate action rather than a stored tensor, but it cannot be inferred from pairwise state.
- **Actual hidden-source magnitude -- open.** Requires a signed residual receipt; Gaussian correlation bounds do not cover it.

## Checkpoint 36: exact H4-score lowering identity and tree double-count gate

Let

\[
\mathcal K_4[g;A^3,B]
=\mathbb E\left[g_+:A^3B:\right],
\]

where \(A,B\) are centered jointly Gaussian latent linear forms, \(v_A=\mathbb E A^2\), and \(c_{AB}=\mathbb E AB\). The tied Wick polynomial is

\[
:A^3B:
=A^3B-3v_AAB-3c_{AB}A^2+3v_Ac_{AB}.
\]

Holding the raw latent variables and gate law fixed while differentiating the score covariance gives

\[
\boxed{
\partial_{v_A}:A^3B:=-3:AB:,
\qquad
\partial_{c_{AB}}:A^3B:=-3:A^2:.
}
\]

Therefore

\[
\boxed{
\mathbb E[g_+:AB:]
=-\frac13\partial_{v_A}\mathcal K_4[g;A^3,B].
}
\]

The same descendant is a mean-translation Hessian. Expanding the translated score,

\[
\partial_t^2:(A+t)^3B:|_{t=0}=6:AB:,
\]

so

\[
\boxed{
\mathbb E[g_+:AB:]
=\frac16\partial_t^2
\mathcal K_4[g;(A+t)^3,B]|_{t=0}.
}
\]

The two representations obey the normal-ordering heat relation

\[
\partial_{v_A}\mathcal K_4
=-\tfrac12\partial_t^2\mathcal K_4.
\]

They are one source descendant and must not be counted twice. In a fully polarized score \(:A_1A_2A_3B:\), differentiation with respect to \(\operatorname{Cov}(A_1,A_2)\) gives \(-:A_3B:\) with no tied combinatorial factor.

Equivalently, translating two independent slots gives

\[
\partial_t\partial_s
\mathcal K_4[A_1+t,A_2+s,A_3,B]|_{0}
=\mathbb E[g_+:A_3B:],
\]

while

\[
\partial_{C_{12}}\mathcal K_4
=-\mathbb E[g_+:A_3B:].
\]

Thus \(\partial_{C_{12}}=-\partial_t\partial_s\) in polarized coordinates. Tying the three \(A\)-slots sums three covariance pairs but six ordered translation derivatives, producing the factors \(-3\) and \(6\) above.

For the lawful ReLU tree choose

\[
A=\sigma_jg_jZ_j,
\qquad
B=\sigma_kg_kZ_k.
\]

Then the H2 gate score is

\[
\mathbb E[(m_i+\sigma_iZ_i)_+:AB:]
=\beta_ic_{ij}c_{ik}.
\]

Summing the three central-vertex choices gives the exact all-distinct tree \(S_{ijk}\). Hence \(S\) is not independent H4 physics: it is the covariance-JVP/translation-Hessian lowering of the contract-native H4 score.

### Complete-interface ownership and residual

Let \(\mathcal L_{4\to2}\) denote this lowering map, with the tied constants above. The self-consistent order-two tree already owns \(\mathcal L_{4\to2}\mathcal K_4\). A new H4 action must be residualized against

\[
\mathcal E_{\rm old}
+\operatorname{range}(\mathcal L_{4\to2}^*),
\]

imposing the heat relation once rather than treating covariance JVP and translation Hessian as two columns.

At a synchronized cut, the complete supply residual is schematically

\[
R_\ell^{\rm complete}
=dK_{\ell,\rm cert}
-dK_{\ell,\rm old}
-O_\ell\widehat S_\ell^{(2)}
-O_{4,\ell}K_{4,\ell}^{\perp},
\]

paired with the complete interface

\[
Q_\ell=(Q_{31},Q_4,Q_{22},Q_{22\times})_\ell.
\]

The H4 conditional innovation uses the usual Schur complement after including the tree-lowering column in the old block.

### Exact reuse condition

Let \(P_{\rm law}\) project onto the lawful source-provider range. The H4 derivative can replace the separately evaluated tree action exactly iff

\[
\boxed{
P_{\rm law}\left(Q_4-\mathcal L_{4\to2}^*Q_S\right)=0,
}
\]

with the same layer cut, gate, latent scaling, normal-order convention, and old/interface projection. Here \(\mathcal L_{4\to2}^*\) lifts a tree/H2 costate into the H4 dual, because \(\langle Q_S,\mathcal Lp\rangle=\langle\mathcal L^*Q_S,p\rangle\). Under this condition the H4 oracle's covariance JVP supplies the tree scalar and the separate tree action must be removed. If the projected difference is nonzero, H4 has an independent conditional action and the tree computation/cost remains.

### Graph-ready split nodes

- **H4-to-tree covariance JVP -- proved.** Criterion: the tied Wick derivative identities.
- **H4 translation-Hessian equivalence -- proved.** Criterion: \(\partial_v=-\tfrac12\partial_\mu^2\); one descendant, not two features.
- **ReLU all-distinct tree identification -- proved.** Lawful leaf scaling gives \(\beta_ic_{ij}c_{ik}\), cyclically.
- **H4/tree double-count -- mandatory quotient.** Add the lowering range to the complete old span before any H4 gain claim.
- **Complete-interface residual -- open value.** Use K31, K4, uncrossed K22, and crossed K22 at the synchronized cut.
- **Derivative reuse -- conditional.** Pass iff the provider-projected costate equality holds; otherwise retain separate cost and Schur residual.

## Checkpoint 37: score-coordinate derivative versus full physical JVP

The H4-to-tree identity differentiates the normal-order covariance parameters while holding the raw latent variables and gate law fixed. A physical network parameter \(\theta\) generally moves all three. Its derivative decomposes as

\[
\begin{aligned}
D_\theta\mathcal K_4={}&
\mathbb E[D_\theta g_+:A^3B:]\\
&+\mathbb E[g_+D_{A,B}:A^3B:]\\
&+\mathbb E[g_+D_{\rm score\ covariance}:A^3B:].
\end{aligned}
\]

Only the third line is the exact H2/tree lowering derived in Checkpoint 36. The first is gate/frame motion; the second is producer/latent-factor motion. They belong to the complete co-adapted provider and must not be removed merely because the tree descendant is old.

The translation-Hessian identity has the same scope: the Wick slots are translated while \(g_+\) is held fixed. If the same physical parameter also translates the gate, product-rule cross terms remain.

### Graph-ready correction

- **Score-coordinate tree descendant -- exact duplicate.** Subtract once.
- **Gate JVP and latent-factor JVP -- separate open actions.** Retain unless independently matched to old/self-consistent terms.
- **Whole physical JVP reuse -- conditional.** Valid only if the extra derivative terms vanish or have explicit ownership receipts.

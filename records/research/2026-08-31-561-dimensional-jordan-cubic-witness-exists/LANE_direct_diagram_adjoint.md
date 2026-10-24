# Direct diagram adjoint

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## 2026-08-30 — direct diagram contractions and topology boundary

### Convention

For a standard Gaussian input and probabilists' Hermites, write

\[
c_m(f)=\mathbb E[f(Z)H_m(Z)].
\]

The connected contribution of a labeled multigraph with vertex degrees \(d_v\) and edge multiplicities \(m_e\) is

\[
\frac{\prod_v c_{d_v}}{\prod_e m_e!}\prod_e R_e^{m_e}.
\]

For the shifted ReLU \(f(z)=\sigma(z+a)_+\),

\[
c_1=\sigma\Phi(a),\qquad
c_m=\sigma\phi(a)H_{m-2}(-a),\quad m\ge2.
\]

These normalizations are the source of every factorial below; changing the Hermite convention requires changing them coherently.

### Cubic Hermite path: exact \(O(n^3)\) contraction

Let the three vertices of a two-edge path carry diagonal coefficient vectors \(x,y,z\), and let its two edge kernels be \(A,C\). For a K31 endpoint, put

\[
P=WQ_{31}^{\mathsf T},\quad
X=D_xW,\quad Y=D_yW,\quad Z=D_zP.
\]

The unrestricted scalar contraction is

\[
\Phi_{\rm path}
=\sum_{j,a}Y_{ja}(A^{\mathsf T}X)_{ja}(CZ)_{ja}
=\langle Y,(A^{\mathsf T}X)\odot(CZ)\rangle .
\]

The leaf collision \(i=k\) is

\[
\Phi_{i=k}
=\langle X\odot Z,(A\odot C^{\mathsf T})Y\rangle,
\]

so the all-distinct path is \(\Phi_{\rm path}^{\circ}=\Phi_{\rm path}-\Phi_{i=k}\). Both terms need only dense matrix multiplication and Hadamard products, hence \(O(n^3)\). A triangle closes the third edge and creates the K4/treewidth-3 contraction after the K31 projection; generically that is \(O(n^4)\), not a disguised path contraction.

### Two-root all-distinct four-cycle

For a zero-diagonal directed edge matrix \(B\), set

\[
C=B\odot B^{\mathsf T},\qquad d=\operatorname{diag}(B^2),
\]

\[
S_{\rm opp}=\tfrac12\{B^2\odot(B^2)^{\mathsf T}-C^2\},
\]

\[
P_s=B^3-D_dB-BD_d+B^{\odot2}\odot B^{\mathsf T},qquad
S_{\rm adj}=B\odot P_s^{\mathsf T}.
\]

The appropriate endpoint plane \(Q\) contracts these as \(\langle Q,S\rangle\). For the H2 square use \(B=D_{c_2}R_{\rm off}\); for the literal H4 square use \(B=D_{c_4}(R_{\rm off}^{\odot2}/2)\). This is an \(O(n^3)\) identity only when the endpoint genuinely has two roots/self-energy form. Projecting four free legs through dense \(W\) into a generic K31/K22 endpoint creates a wheel/K4 minor and restores the generic \(n^4\) barrier.

### Fish and series-parallel boundary

The order-\(r\) two-vertex fish plane is

\[
\Sigma^{(r)}_{ij}=
\frac{c_{r+1,i}c_{r+1,j}}{r!}R_{ij}^{r}.
\]

Thus \(r=2\) is the cubic fish and \(r=3\) the quartic sunset. More generally, fixed two-terminal series-parallel diagrams are evaluable in \(O(n^3)\): series composition is matrix multiplication with the join weight on the diagonal; parallel composition is Hadamard multiplication. A K4 minor, equivalently treewidth at least three in the projected physical-index graph, is the clean generic \(O(n^4)\) boundary. Special algebraic rank or a canonical matrix-unit basis can lower that cost, but arbitrary dense physical cores cannot be assumed to have it.

## Equality sectors \(q_3(2,1)\) and \(q_4(2,1,1)\)

Let \(X_i\) be centered and define the repeated supervertex

\[
g_i=X_i^2-\mathbb E X_i^2.
\]

If \(X_i=\sum_{p\ge1}c_{p,i}H_p/p!\), the Hermite coefficient of \(g_i\) is

\[
d_{m,i}=m!\!\sum_{p+q-2r=m}
\frac{c_{p,i}c_{q,i}}
{r!(p-r)!(q-r)!}.
\]

### The \(q_3(2,1)\) plane

Define

\[
H_{rs}=\kappa(X_r,X_r,X_s)=\operatorname{Cov}(g_r,X_s)
=\sum_m\frac{d_{m,r}c_{m,s}}{m!}R_{rs}^{m}.
\]

With \(H_0=H-\operatorname{Diag}(\operatorname{diag}H)\) and \(P=WQ_{21}^{\mathsf T}\), its exact K21 endpoint is

\[
\Phi_{321}
=\langle W^{\odot2},H_0P\rangle
+2\langle W\odot P,H_0W\rangle.
\]

The removed diagonal is exactly the three copies of the \(q_3(3)\)/CP3 collision. Therefore this direction is not new if the existing CP3/K21 state already stores the mixed \((2,1)\) plane with the same normalization; otherwise it is a missing equality sector, not an all-distinct cubic tensor.

### The \(q_4(2,1,1)\) plane

The exact repeated-root reduction is

\[
\kappa_4(X_r,X_r,X_s,X_t)
=\kappa_3(g_r,X_s,X_t)-2C_{rs}C_{rt}.
\]

For a center-root path with Hermite edge degrees \(\alpha,\beta\), the coefficient is

\[
\frac{[d_{\alpha+\beta,r}-2c_{\alpha,r}c_{\beta,r}]
c_{\alpha,s}c_{\beta,t}}
{\alpha!\beta!}
R_{rs}^{\alpha}R_{rt}^{\beta}.
\]

Consequently every star/path component has a direct \(O(n^3)\) K31 contraction. For a separable component

\[
H_{rst}=u_rv_sw_tA_{rs}C_{rt},
\]

define

\[
\mathcal S(X,Y,Z)=\sum_{r,a}X_{ra}(AY)_{ra}(CZ)_{ra},
\]

\[
\mathcal S^\circ(X,Y,Z)=\mathcal S(X,Y,Z)
-\sum_{s,a}Y_{sa}Z_{sa}[(A\odot C)^{\mathsf T}X]_{sa}.
\]

Then

\[
\Phi_{31,211}=3\{\mathcal S^\circ(D_uW^{\odot2},D_vW,D_wP)
+\mathcal S^\circ(D_u(W\odot P),D_vW,D_wW)\}.
\]

The corresponding path formula is the cubic path identity above. Collision accounting for a designated equal pair is

\[
q_4^{\rm unrestricted,pair}
=q_4(211)+3q_4(31)+2q_4(22)+6q_4(4).
\]

Thus Q24 supplies the collision boundary but not the off-diagonal three-label sector. Generic \(q_4(211)\) has order \(n^3/2\) degrees of freedom and cannot be reconstructed from an \(n^2\) Q24 plane without extra structure. In a generic K22 contraction, cross-leaf collisions form the same K4 obstruction; the correct cheap operation is to reuse the Q24 collision receipt rather than recompute a four-index object.

## NCQC/Jordan realization of the external quartic

For centered quadratic Gaussian chaos

\[
q_i=z^{\mathsf T}A_i z-\operatorname{tr}A_i,qquad A_i=A_i^{\mathsf T},
\]

write \(A\circ B=(AB+BA)/2\). The complete \((2,1,1)\) value is

\[
q^{\rm ext}_{4,iijk}
=16\{\langle A_i^2,A_j\circ A_k\rangle
+2\langle A_i\circ A_j,A_i\circ A_k\rangle\}
\]

\[
=48\operatorname{tr}(A_i^2A_jA_k)
+8\operatorname{tr}([A_i,A_j][A_i,A_k]).
\]

The first term is the commuting square/fish direction. The second is the signed commutator Gram and is the clean trajectory-dark direction. Its Q24 diagonal is

\[
q^{\rm ext}_{4,iijj}=48\operatorname{tr}(A_i^2A_j^2)
-8\|[A_i,A_j]\|_F^2.
\]

Hence Q24 can reveal \(\|[A_i,A_j]\|^2\), but it cannot reveal the cross angle
\(\langle[A_i,A_j],[A_i,A_k]\rangle\). That is precisely why Q24 does not determine \(q_4(211)\).

For a K31 endpoint define

\[
N_a=\sum_jW_{ja}A_j,\quad R_a=\sum_kP_{ka}A_k,
\quad S_a=\sum_iW_{ia}^2A_i^2,
\quad T_a=\sum_iW_{ia}P_{ia}A_i^2.
\]

The square contribution is

\[
144\sum_a\{\operatorname{tr}(S_aN_aR_a)
+\operatorname{tr}(T_aN_a^2)\},
\]

and the commutator contribution is

\[
24\sum_{a,i}\{W_{ia}^2\operatorname{tr}([A_i,N_a][A_i,R_a])
+W_{ia}P_{ia}\operatorname{tr}([A_i,N_a]^2)\}.
\]

The square part has standard trace/matrix \(O(n^3)\) evaluation. For arbitrary dense physical cores the commutator sum is the diagonal of a weighted Jordan/sandwich superoperator and is not generically an \(O(n^3)\) calculation; the usual cost is above cubic when the latent matrix dimension squared scales as \(n\). A canonical matrix-unit Jordan frame does reduce it to cubic, but an arbitrary dense change of physical coordinates followed by coordinatewise ReLU does not preserve that frame. This is the exact no-go boundary: NCQC algebra exposes the missing direction, but does not by itself grant a cheap physical-basis contraction.

For reference, in a canonical symmetric matrix basis, if \(A_i=E_{pp}\),

\[
\langle A_i^2,N\circ R\rangle=(NR)_{pp},\qquad
\operatorname{tr}(A_iNA_iR)=N_{pp}R_{pp};
\]

if \(A_i=(E_{pq}+E_{qp})/\sqrt2\),

\[
\langle A_i^2,N\circ R\rangle
=\tfrac12[(NR)_{pp}+(NR)_{qq}],
\]

\[
\operatorname{tr}(A_iNA_iR)
=\tfrac12[N_{qq}R_{pp}+N_{pp}R_{qq}+2N_{pq}R_{pq}].
\]

These identities explain the special-frame speedup and also why it does not transfer automatically.

### Minimal signed provider

Choose a skew probe \(S\) and let

\[
z_{ij}=\langle S,[A_i,A_j]\rangle.
\]

The rank-one signed approximation to the dark tensor is

\[
H^-_{ijk}=-8z_{ij}z_{ik}.
\]

A Q24-compatible completion can take

\[
z_{ij}=\operatorname{sign}\langle S,[A_i,A_j]\rangle
\sqrt{d_{ij}},\qquad d_{ij}=\|[A_i,A_j]\|_F^2.
\]

With \(Z=[z_{ij}], U=ZW, V=ZP, D=Z^{\odot2}\), the exact all-distinct K31 scalar is

\[
-24\{\langle W^{\odot2},U\odot V\rangle
+\langle W\odot P,U^{\odot2}\rangle
-\langle W^{\odot2},D(W\odot P)\rangle
-\langle W\odot P,DW^{\odot2}\rangle\}.
\]

For K22 the unrestricted scalar is

\[
-8\{\langle Q,(W^{\odot2})^{\mathsf T}U^{\odot2}\rangle
+\langle Q,(U^{\odot2})^{\mathsf T}W^{\odot2}\rangle
+4\langle Q,(W\odot U)^{\mathsf T}(W\odot U)\rangle\},
\]

followed by the two Q24 q22 collision receipts for the exact distinct sector. The natural target-adjoint compression gate is the signed skew-space spectral mass: retain a provider only if the top retained absolute eigenmass captures the stipulated fraction (the working screen was 0.93). Before any such construction, the Q24 Cauchy/support ceiling should be used as the fail-fast gate.

## Target-free support ceiling and master units

For each repeated index \(i\), let

\[
G_i(j,k)=\langle[A_i,A_j],[A_i,A_k]\rangle\succeq0,
\qquad \operatorname{diag}G_i=d_i.
\]

The K31 and K22 endpoint coefficient matrices for the commutator Gram can be written

\[
B^{31}_i=24\operatorname{OffDiag}\operatorname{Sym}
\sum_a[W_{ia}^2w_ap_a^{\mathsf T}+W_{ia}P_{ia}w_aw_a^{\mathsf T}],
\]

\[
B^{22}_i=8\operatorname{OffDiag}\operatorname{Sym}
\sum_{ab}Q_{ab}[W_{ia}^2w_bw_b^{\mathsf T}
+W_{ib}^2w_aw_a^{\mathsf T}+4W_{ia}W_{ib}w_aw_b^{\mathsf T}].
\]

The signed dark endpoint is \(-\sum_i\operatorname{tr}(B_iG_i)\). With only Q24 norms known, the sharp label-free ceiling is the SDP

\[
\max_{G_i\succeq0,\,\operatorname{diag}G_i=d_i}
|\operatorname{tr}(B_iG_i)|.
\]

Two closed upper bounds are

\[
\min\{s_i^{\mathsf T}|B_i|s_i,
\|B_i\|_{\rm op}\mathbf1^{\mathsf T}d_i\},
\qquad s_i=\sqrt{d_i}.
\]

Using absolute endpoint weights gives a safe global supremum by summing these bounds over \(i\). This is an amplitude/value ceiling, not a target-correlation certificate: after arbitrary rescaling, every nonzero scalar candidate can have correlation one with some compatible target. Without an assumption linking the ideal target to the external source, the target-free lower bound on squared alignment is exactly zero.

The quartic master units are

\[
V_4=1.0313\times10^{-9},\qquad
a_{4,\min}=\operatorname{clip}_{[0,1]}
\left[1-\frac{0.7453-17.316r_v^2-4.370(1-a_3)}{1.0313}\right].
\]

With exact variance and cubic channels, \(a_{4,\min}=0.27732\). The corresponding recovered value and RMS are

\[
0.27732V_4=2.86000116\times10^{-10},
\qquad \sqrt{0.27732V_4}=1.691\times10^{-5}.
\]

The overall target is infeasible, even with an exact quartic, unless

\[
17.316r_v^2+4.370(1-a_3)\le0.7453.
\]

## Complete external quartic and subspace-angle certificate

Let \(X\) have rows \(\operatorname{svec}(A_i)\), \(K=XX^{\mathsf T}\),

\[
\Pi=X^{\mathsf T}K^+X,quad
Y_{ij}=A_i\circ A_j,quad
Z_{ij}=(I-\Pi)\operatorname{svec}(Y_{ij}),
\quad t_{ij}=8X\operatorname{svec}(Y_{ij}).
\]

Then the exact external \((2,1,1)\) tensor splits as

\[
q^{\rm ext}_{4,iijk}=q_{\parallel,iijk}+q_{\perp,iijk},
\]

\[
q_{\parallel,iijk}
=\tfrac14[t_{ii}^{\mathsf T}K^+t_{jk}
+2t_{ij}^{\mathsf T}K^+t_{ik}],
\]

\[
q_{\perp,iijk}
=16[\langle Z_{ii},Z_{jk}\rangle
+2\langle Z_{ij},Z_{ik}\rangle].
\]

Collision subtraction (31/22/4) and old-span residualization are linear operations applied after this split. The latent-Hermite quartic \(q_4^z\) is a separate partition and must not be silently identified with \(q_\perp\).

For a target quartic correction \(y_4\) in the MSE Hilbert space, \(\|y_4\|^2=V_4\), and a candidate subspace \(\mathcal S\),

\[
a_4=\frac{\|P_{\mathcal S}y_4\|^2}{V_4}.
\]

For any witness \(f\in\mathcal S\),

\[
a_4\ge\frac{\langle y_4,f\rangle^2}{V_4\|f\|^2},
\qquad
a_4\ge1-\frac{\|y_4-f\|^2}{V_4}.
\]

Thus the exact threshold is either cosine \(>\sqrt{0.27732}=0.52661\), or relative coefficient-one error \(<\sqrt{0.72268}=0.850106\). After the known 4/31/22 equality sectors are residualized away, the same-source identity is

\[
R q^{\rm ext}_{211}+R q^{\rm ext,res}_{1111}
=R q^{\rm ext,full}.
\]

So the candidate should use the complete external K31 witness; uncertainty in the artificial 211-versus-1111 split then cancels. For four all-distinct labels,

\[
q^{\rm ext}_{4,ijkl}=16\sum_{\rm three\ pairings}
\langle A_i\circ A_j,A_k\circ A_l\rangle,
\]

and its complete K31 contraction is simply

\[
48\sum_a\operatorname{tr}(N_a^3R_a),
\]

which is \(O(n^3)\). By contrast the complete K22 contraction contains

\[
16\sum_{ab}Q_{ab}
[2\operatorname{tr}(N_a^2N_b^2)+\operatorname{tr}(N_aN_bN_aN_b)],
\]

whose crossed term is the generic K4/weighted-superoperator obstruction. The economical route is therefore complete K31 plus known collision subtraction; generic K22 should be closed by cost unless special structure is proved.

## Exact-rest support SDP

Index unordered pairs by \(\alpha=(i,j)\) and put

\[
G_Z[\alpha,\beta]=\langle z_\alpha,z_\beta\rangle,
\qquad z_{ij}=(I-\Pi)\operatorname{svec}(Y_{ij}).
\]

Given its known diagonal and collision linear functionals, the feasible set is

\[
\mathcal C(S)=\{G\succeq0:\operatorname{diag}G=d,\ \mathcal A(G)=g_{\rm coll}\}.
\]

For any endpoint pullback \(B\),

\[
h_{\mathcal C}(B)=\max_{G\in\mathcal C}\langle B,G\rangle
=\min_{\nu,\mu}\{\nu^{\mathsf T}d+\mu^{\mathsf T}g:
\operatorname{Diag}\nu+\mathcal A^*\mu-B\succeq0\}.
\]

The admissible scalar interval is \([-h(-B),h(B)]\), its width is
\(w=h(B)+h(-B)\), and the cheap bound is

\[
w\le\operatorname{tr}(G)[\lambda_{\max}(B)-\lambda_{\min}(B)].
\]

The midpoint provider has worst-case scalar error \(r=w/2\). This is an uncertainty certificate, not a target-alignment certificate. A useful alignment statement requires at least the realizability assumption that the ideal external dark Gram belongs to \(\mathcal C(S)\), plus a bound on everything outside that external family. Under that assumption,

\[
a_4\ge1-rac{(\|r\|_{L^2}+\|e_{\rm out}\|_{L^2})^2}{V_4}.
\]

The allowable total squared error at the working threshold is
\(0.72268V_4=7.45299884\times10^{-10}\). Equivalently, for a single feature the minimal geometric assumption is

\[
y_4=\alpha f+\epsilon,quad \epsilon\perp f,quad
\|\epsilon\|^2<0.72268V_4.
\]

No label-free positive lower bound on alignment follows from support amplitudes alone: two targets of equal norm can be chosen parallel and orthogonal to the same feasible external witness.

## Checkpoint 4 — reverse Bellman identity and its closure obstruction

### Exact endpoint adjoint for a fixed tangent chain

Consider the deployed covariance tangent recursion

\[
\epsilon_{l+1}=W_{l+1}^{\mathsf T}
[D_l\odot\epsilon_l+I_l]W_{l+1}.
\]

Starting from endpoint mask \(B_L\) (for a diagonal readout, \(B_L=\operatorname{Diag}h\)), define

\[
H_l=W_{l+1}B_{l+1}W_{l+1}^{\mathsf T},
\qquad B_l=D_l\odot H_l.
\]

Then

\[
\langle B_{l+1},\epsilon_{l+1}\rangle
=\langle B_l,\epsilon_l\rangle+\langle H_l,I_l\rangle,
\]

and hence

\[
\langle B_L,\epsilon_L\rangle
=\langle B_0,\epsilon_0\rangle
+\sum_l\langle H_l,I_l\rangle.
\]

This is an exact Bellman/telescoping identity for the stated linear tangent chain; it does not claim that the chain is an exact closure of the network.

If the K31 injection is

\[
I_l^{31}=\tfrac16[G_l^{31}\odot D_l^{31}
+G_l^{13}\odot(D_l^{31})^{\mathsf T}],
\]

the exact source-plane adjoint is

\[
Q_l^{31}=\tfrac16[H_l\odot G_l^{31}
+(H_l\odot G_l^{13})^{\mathsf T}],
\]

because \(\langle H_l,I_l^{31}\rangle=\langle Q_l^{31},D_l^{31}\rangle\).

Let

\[
D_{ab}^{31}=\sum_{ijkl}W_{ia}W_{ja}W_{ka}W_{lb}T_{ijkl},
\qquad P=W(Q_l^{31})^{\mathsf T}.
\]

The coordinate endpoint tensor and adjoint functional are

\[
\Xi_{ijkl}=\sum_aW_{ia}W_{ja}W_{ka}P_{la},
\qquad
\mathcal A_{B,l}[T]=\langle\operatorname{Sym}\Xi,T\rangle.
\]

For the complete external quadratic-core cumulant,

\[
\boxed{\mathcal A_{B,l}[T_l^{\rm ext}]
=48\sum_a\operatorname{tr}(N_{l,a}^3R_{l,a})},
\]

where \(N_{l,a}=\sum_iW_{ia}A_{l,i}\) and
\(R_{l,a}=\sum_iP_{ia}A_{l,i}\). This includes the complete external K31 tensor before the explicit old-span/collision residualizer. In particular, its all-distinct AA sector is not a separate missing term. Global latent-Hermite siblings such as \(\operatorname{Sym}(h\otimes M_i)\) or \(\operatorname{Sym}(S\otimes M_i)\) must be treated as separate source partitions unless an exact cumulant identity shows that they reproduce part of \(T_l^{\rm ext}\).

### Exact residual identity, but no closed value identity from the current state

Let \(e_l=s_l-\widehat s_l\) be the exact-minus-tracked state and let \(L_l\) be the deployed tangent map. Define the residual, rather than approximating it, by

\[
\rho_l=e_{l+1}-L_le_l-I_l^{\rm ext}.
\]

Reverse propagation by \(L_l^*\) gives the exact identity

\[
\langle B_L,e_L\rangle
=\langle B_0,e_0\rangle
+\sum_l\mathcal A_{B,l}[T_l^{\rm ext}]
+\sum_l\langle H_l,\rho_l\rangle.
\]

This is the strongest valid white-box Bellman statement. It is exact because \(\rho_l\) is defined as the transition defect. It is not a computable closure from the current \(\le4\) state, because that state does not determine \(\rho_l\).

The defect must be typed as

\[
\rho_l=\rho_l^{\rm supp}+\rho_l^z+\rho_l^{22}
+\rho_l^{\ge5}+\rho_l^{\rm frame}+\rho_l^{\rm state}
+\rho_l^{\rm coll}+\rho_l^{\rm num},
\]

with a terminal term \(\rho_{\rm term}\):

1. \(\rho^{\rm supp}\): the difference between the chosen midpoint/rank provider and the true feasible \(q_\perp\) Gram;
2. \(\rho^z\): latent-Hermite quartic partitions not generated by the external quadratic core;
3. \(\rho^{22}\): ideal endpoint geometry carried by the generic K22 crossed-AA route and therefore outside the selected K31 provider span;
4. \(\rho^{\ge5}\): higher-chaos/moment feed-down under the next nonlinear layer;
5. \(\rho^{\rm frame}\): co-adaptation of means, variances, covariance, derivative masks, and the tangent frame, including \((J_l-L_l)e_l\) for the path-integrated exact Jacobian \(J_l\);
6. \(\rho^{\rm state}\): error already present in the transported \(m,C,\mathrm{CP3},\mathrm{Q24}\), and core states;
7. \(\rho^{\rm coll}\): equality-sector/collision or lineage-normalization mismatch;
8. \(\rho^{\rm num}\): provider rank/truncation and arithmetic error;
9. \(\rho_{\rm term}\): terminal nonlinear/readout error not represented by \(B_L\).

For each family define the scalar endpoint defect

\[
\delta_B^r=\sum_l\langle H_l,\rho_l^r\rangle,
\qquad
\delta_B^{\rm term}=\rho_{\rm term}.
\]

A label-free bound, if the corresponding source norms are observable, is

\[
\boxed{\eta_B
\le \eta_{\rm init}+\eta_{\rm supp}+\eta_z+\eta_{22}
+\eta_{\ge5}+\eta_{\rm frame}+\eta_{\rm state}
+\eta_{\rm coll}+\eta_{\rm num}+\eta_{\rm term}},
\]

where \(\eta_r=\|\delta_B^r\|_{L^2}\) and
\(\eta_{\rm init}=\|\langle B_0,e_0\rangle\|_{L^2}\). For the support term,

\[
\eta_{\rm supp}
\le\left\|\sum_l\frac{h_{\mathcal C_l}(B_l^Z)
+h_{\mathcal C_l}(-B_l^Z)}2\right\|_{L^2},
\]

with the cheaper pointwise radius

\[
r_l\le\tfrac12\operatorname{tr}(G_l)
[\lambda_{\max}(B_l^Z)-\lambda_{\min}(B_l^Z)].
\]

For any hidden tensor family a generic dual-norm receipt is

\[
\eta_r\le
\left\|\sum_l\|\mathcal A_{B,l}^r\|_*\,
\|T_l^r\|\right\|_{L^2}.
\]

For the nonlinear frame defect, a second-order form is

\[
\eta_{\rm frame}\le
\left\|\sum_l\tfrac12\|B_{l+1}\|_*
\sup_{\xi\in[s_l,\widehat s_l]}\|D^2F_l(\xi)\|
\|e_l\|^2\right\|_{L^2}.
\]

These are genuine no-label bounds, but the current state supplies neither the latent/higher source norms nor a sufficiently small frame/state defect. Therefore it cannot evaluate a threshold-small \(\eta_B\).

### Impossibility theorem

Let \(S_{\le4}\) denote all currently retained observables. If two white-box hierarchy completions \(P_1,P_2\) agree on \(S_{\le4}\) but have different endpoint-projected transition defects, then every estimator measurable with respect to \(S_{\le4}\) has minimax error at least

\[
\tfrac12\|y_B(P_1)-y_B(P_2)\|_{L^2}.
\]

Such completions exist for two independent reasons. First, even within the external quartic, the feasible Gram set \(\mathcal C(S)\) has nonzero support width unless all endpoint pullbacks are fixed by its equality constraints. Second, exact moments through degree four do not close a subsequent ReLU transition: distinct distributions can share those moments while differing in one-sided tail allocation and higher moments, hence in \(\rho^{\ge5}\) and the next derivative frame. The compressed present state is weaker still. Consequently there is no exact Bellman/value identity determined by the current \(\le4\) state, and no positive target-alignment certificate without an additional source-to-target/closure assumption. The support midpoint is minimax for its typed fiber, but it does not solve the other fibers.

### Terminal projection gate and minimal missing observable

Let \(\varphi\) be the available provider stack, \(G=\mathbb E[\varphi\varphi^{\mathsf T}]\), and let \(c_B=\mathbb E[\varphi f_B]\) for the complete external K31 Bellman witness

\[
f_B=R_{\rm old}\sum_l\mathcal A_{B,l}[T_l^{\rm ext}].
\]

Its projection is \(g_B=c_B^{\mathsf T}G^+\varphi\), with

\[
F_B=\|g_B\|=\sqrt{c_B^{\mathsf T}G^+c_B}.
\]

If the ideal residual quartic obeys \(\|y_4-f_B\|\le\eta_B\), then, for
\(u_B=g_B/F_B\),

\[
\langle y_4,u_B\rangle
\ge \langle f_B,u_B\rangle-\eta_B
=F_B-\eta_B.
\]

Therefore the exact sufficient fire gate is

\[
\boxed{\sqrt{c_B^{\mathsf T}G^+c_B}
\ge\eta_B+1.691\times10^{-5}}.
\]

The current \(\le4\) state cannot certify this gate because it cannot certify \(\eta_B\). The information-minimal additional observable is the single endpoint-adjoint Bellman innovation per net,

\[
\boxed{\rho_B
=\langle B_0,e_0\rangle
+\sum_l\langle H_l,\rho_l\rangle+\rho_{\rm term}
=y_4-f_B},
\]

together with a white-box, label-free certificate \(\|\rho_B\|_{L^2}\le\eta_B\). This scalar is sufficient because every omitted tensor matters only through this adjoint projection; it is minimal for the same reason. If an aggregate scalar oracle is unavailable, the minimal typed substitute is the set of per-layer adjoint-projected innovations for (i) external support fiber, (ii) latent \(q_4^z\), (iii) out-of-span K22, (iv) higher-chaos feed-down, and (v) frame/state error. Observing full missing tensors is unnecessary.

An assumption-only substitute is an actual closure theorem: the ideal dark external Gram lies in \(\mathcal C(S)\), all non-external terms have a stated combined \(L^2\) radius, and the deployed tangent-frame remainder obeys the Hessian bound above. Without that theorem or the projected innovation observable, the 0.850106 approximation tolerance is not a certificate.

### Current successor question

Can the exact white-box engine expose \(\rho_B\), or at least its five typed adjoint projections, without materializing any four-index tensor? The promising route is to accumulate each omitted transition innovation directly against \(H_l\) at the same Site-A/Site-B covariance conjugation where the K31 injection is consumed. If it can, the next purely symbolic task is to derive each scalar from the existing NCQC/Jordan traces and prove that q1111 is already included while the generic K22 scalar is either bounded below the remaining \(\eta_B\) budget or closed by cost. If it cannot, the terminal gate is mathematically unidentifiable from the current state and this lane should close rather than launch another feature fit.

## Checkpoint 4 continuation — universal projected-cumulant identity

There is a stronger tensor-free identity than the NCQC trace formula. Let \(X_i\) be the **actual centered** layer variables, with no Gaussian-chaos assumption. For each output column define

\[
U_a=\sum_iW_{ia}X_i,qquad
V_a=\sum_iP_{ia}X_i,qquad P=W(Q^{31})^{\mathsf T}.
\]

Multilinearity of the joint cumulant gives

\[
\boxed{
\langle Q^{31},D^{31}\rangle
=\sum_a\kappa_4(U_a,U_a,U_a,V_a)
}
\]

for the complete K31 action, regardless of the latent/external decomposition. Since the variables are centered,

\[
\boxed{
\kappa_4(U,U,U,V)
=\mathbb E[U^3V]-3\mathbb E[U^2]\mathbb E[UV].
}
\]

Proof: \(D_{ab}^{31}=\kappa_4(Y_a,Y_a,Y_a,Y_b)\) for
\(Y_a=\sum_iW_{ia}X_i\), while
\(V_a=\sum_bQ_{ab}^{31}Y_b\). Summing \(Q_{ab}^{31}D_{ab}^{31}\) and using cumulant multilinearity yields the formula.

This resolves the successor question at the level of information structure: the complete K31 quartic innovation can be observed without a four-index tensor. The exact per-layer q4 defect relative to the external NCQC provider is

\[
\boxed{
\rho_{B,l}^{q4}
=\sum_a\left[
\mathbb E(U_{l,a}^3V_{l,a})
-3\mathbb E(U_{l,a}^2)\mathbb E(U_{l,a}V_{l,a})
-48\operatorname{tr}(N_{l,a}^3R_{l,a})
\right].
}
\]

It already combines external \(q_{211}\), external all-distinct \(q_{1111}\), latent-Hermite quartic pieces, and every mixed chaos partition. The equality-sector/old-span receipts must still be subtracted with the same lineage convention, but there is no reason to reconstruct the internal partition split.

Equivalently, write \(U=E+R\) and \(V=F+S\), where \((E,F)\) are the quadratic-core projections and \((R,S)\) contain all other chaos. The exact latent/external mismatch is

\[
\begin{aligned}
\Delta\kappa={}&3\kappa(R,E,E,F)+3\kappa(R,R,E,F)
+\kappa(R,R,R,F)+\kappa(E,E,E,S)\\
&+3\kappa(R,E,E,S)+3\kappa(R,R,E,S)
+\kappa(R,R,R,S).
\end{aligned}
\]

This is the complete typed expansion; a separately named \(q_4^z\) is one subset of these seven mixed/rest terms, not an eighth additive correction.

The information-minimal layer observable is the one scalar \(\rho_{B,l}^{q4}\). A slightly less fused but still tensor-free observable is the collection

\[
m_{31,a}=\mathbb E[U_a^3V_a],\qquad
m_{20,a}=\mathbb E[U_a^2],\qquad
m_{11,a}=\mathbb E[U_aV_a].
\]

No physical four-index state is needed. This is an algebraic information result, not yet a cost theorem: evaluating the projected fourth moments from an arbitrary exact network law may itself be expensive. It becomes deployably useful only if the white-box engine already exposes these action moments or a closed scalar recurrence for them.

A purely norm-based no-label bound is also immediate. For centered random variables,

\[
|\kappa_4(A,B,C,D)|\le4\|A\|_4\|B\|_4\|C\|_4\|D\|_4.
\]

Let \(e=\|E\|_4,r=\|R\|_4,f=\|F\|_4,s=\|S\|_4\). Then

\[
|\Delta\kappa|\le4\{3e^2rf+3er^2f+r^3f
+e^3s+3e^2rs+3er^2s+r^3s\}.
\]

After summing over \((l,a)\) and taking the population \(L^2\) norm, this supplies a lawful \(\eta_{q4}\) whenever the projected rest \(L^4\) norms are available. The present state does not expose those norms, so this is another precise statement of the missing observable rather than a current certificate.

For comparison, the universal K22 identity is

\[
\kappa_4(U_a,U_a,U_b,U_b)
=\mathbb E[U_a^2U_b^2]
-\mathbb E[U_a^2]\mathbb E[U_b^2]
-2\mathbb E[U_aU_b]^2.
\]

Thus a K22 endpoint can also be reduced to an endpoint-weighted scalar quadratic form in the square vector \((U_a^2)_a\), but a generic dense \(Q^{22}\) requires the full pair-action covariance and retains the K4/\(O(n^4)\) symbolic barrier. Low-rank endpoint \(Q^{22}\), a closed Gaussian/Jordan law, or an engine-provided scalar action moment is required to escape it. This cleanly distinguishes the deployable complete-K31 route from the generic K22 no-go.

### Architecture-name audit

The cross-lane labels `D14` and `D10` are schedules, not yet two fully defined coupled recurrences. The exact `D10` object is the reverse adjoint above, active on L5--L14 and initialized at the terminal readout; it transports \(B_l\), forms \(H_l\), and consumes the scalar K31 q4 births without feeding them into the primal state. At the L5 boundary it collapses to its accumulated value projection and is zero below the suffix. For \(k\) directions it carries \(k\) reverse masks and \(k\) scalar accumulators.

`D14` presently denotes one triangular primal nonlinear-frame/signed-H3 tangent bundle through all fourteen transitions, abstractly

\[
e_{l+1}=L_le_l+I_l^{FH},
\]

with reset masks frozen to the baseline trajectory. A proposed graded matrix realization carries even and odd cores, linearly mixes each by \(W\), and uses the parity closure
\(E\circ E,O\circ O\in E\), \(E\circ O\in O\). However, its actual signed-H3/frame nonlinear reset coefficients have not been derived. Feeding the D10/q4 birth back into the even primal core would create a different self-consistent architecture and invalidate the triangular Bellman accounting. Therefore any argument that treats D14 as an already-defined state/update/reset system has a real proof hole.

### Revised next question

Does the exact white-box layer law provide the projected action moments
\(\sum_a\mathbb E[U_a^3V_a]\) and their covariance subtractions at Site A/B in \(O(n^3)\), or can the actual activation chaos be arranged so those moments reduce to the NCQC trace plus a constant number of additional Jordan traces? If yes, \(\rho_B^{q4}\) becomes an exact tensor-free observable and removes the latent/external support ambiguity for K31. If no, the algebra proves that the necessary new state is precisely the projected fourth action moment, and no refit of the existing \(\le4\) summaries can manufacture it.

## Exact affine-quadratic Gaussian closure of the projected action

The first nontrivial positive answer is available when the actual retained layer law is affine quadratic in one Gaussian core:

\[
X_i=h_i^{\mathsf T}z+z^{\mathsf T}A_i z-\operatorname{tr}A_i,
\qquad z\sim N(0,I).
\]

For source parameters \(\theta\), put \(A=\sum_i\theta_iA_i\) and
\(h=\sum_i\theta_i h_i\). The exact joint cumulant-generating function is

\[
K(\theta)
=-\operatorname{tr}A-\tfrac12\log\det(I-2A)
+\tfrac12h^{\mathsf T}(I-2A)^{-1}h.
\]

Its homogeneous degree-four part is

\[
K_4=2\operatorname{tr}(A^4)+2h^{\mathsf T}A^2h.
\]

For the K31 projections define, in addition to \(N_a,R_a\),

\[
h_a=\sum_iW_{ia}h_i,qquad r_a=\sum_iP_{ia}h_i.
\]

Extracting the coefficient of \(x^3y\) in
\(K_4(xN_a+yR_a,xh_a+yr_a)\) and multiplying by \(3!\) gives

\[
\boxed{
\kappa_4(U_a,U_a,U_a,V_a)
=48\operatorname{tr}(N_a^3R_a)
+24\{h_a^{\mathsf T}N_aR_ah_a
+h_a^{\mathsf T}N_a^2r_a\}.
}
\]

The first term is the pure external quadratic-core witness. The second is the complete H1-H1-H2-H2 latent/external cross contribution. It is the exact decoder, in this affine-quadratic model, for the structured cubic sibling often written \(\operatorname{Sym}(h\otimes A_i)\). It needs only matrix-vector/Jordan contractions and therefore does not create a four-index state or a generic K4 cost.

No pure H1 fourth cumulant remains because a Gaussian linear form has zero fourth cumulant. The formula is complete for the affine-quadratic law; any additional H3/H4 chaos belongs in the residual expansion of the previous section. If all units share one global vector \(h\), then \(h_a=(\sum_iW_{ia})h\) and similarly for \(r_a\), a further reduction.

For a coupled reverse implementation, let

\[
\mathcal F(N,R,h,r)
=48\operatorname{tr}(N^3R)
+24[h^{\mathsf T}NRh+h^{\mathsf T}N^2r].
\]

On symmetric matrix variables its exact gradients are

\[
\nabla_R\mathcal F
=48N^3+24\operatorname{Sym}(Nhh^{\mathsf T}),
\qquad
\nabla_r\mathcal F=24N^2h,
\]

\[
\nabla_h\mathcal F
=24[(NR+RN)h+N^2r],
\]

\[
\begin{aligned}
\nabla_N\mathcal F={}&48(N^2R+NRN+RN^2)\\
&+24\{\operatorname{Sym}(Rhh^{\mathsf T})
+\operatorname{Sym}[h(Nr)^{\mathsf T}]
+\operatorname{Sym}[(Nh)r^{\mathsf T}]\}.
\end{aligned}
\]

These adjoints provide a direct synchronized H1/H2 provider and a normalization check for any proposed \(J_h,K_S,L_S\) construction.

This sharpens the no-label radius:

\[
\rho_{B,l}^{q4}
=\rho_{B,l}^{>2}
\]

after the displayed H1/H2 formula is included, where \(\rho^{>2}\) now means only source chaos beyond affine quadratic plus state/frame error. Calling the whole latent mismatch unobservable would therefore be too pessimistic if the D14 state already supplies the linear core \(h_i\) in a synchronized gauge.

The affine-quadratic K22 companion is also exact. For two projected coordinates \((N,h)\) and \((M,r)\),

\[
\boxed{
\begin{aligned}
\kappa_4(X_N,X_N,X_M,X_M)
={}&32\operatorname{tr}(N^2M^2)
+16\operatorname{tr}(NMNM)\\
&+8\{h^{\mathsf T}M^2h+r^{\mathsf T}N^2r
+2h^{\mathsf T}(NM+MN)r\}.
\end{aligned}}
\]

The vector/matrix mixed terms factor through ordinary matrix-vector products and dense endpoint mixing in \(O(n^3)\) or better. The sole generic obstruction is the pure crossed trace \(\operatorname{tr}(N_aN_bN_aN_b)\) under a dense K22 endpoint weight. This locates the K22 cost no-go precisely: latent H1-H2 coupling is not the expensive part; the noncommutative AA crossed word is.

### Convention reconciliation with the half-quadratic H4 lane

The H4 lane uses

\[
Q_A(z)=\tfrac12(z^{\mathsf T}Az-\operatorname{tr}A),
\qquad X=u^{\mathsf T}z+Q_A(z).
\]

This is the present lane's convention after substituting \(N=A/2\), \(R=B/2\), \(h=u\), and \(r=v\). Therefore

\[
\boxed{
\kappa_4(X_A,X_A,X_A,X_B)
=3\operatorname{tr}(A^3B)
+3u^{\mathsf T}(AB+BA)u
+6u^{\mathsf T}A^2v.
}
\]

There is no discrepancy: \(48/16=3\) for the pure term, while
\(24u^{\mathsf T}(A/2)(B/2)u=3u^{\mathsf T}(AB+BA)u\) because the two scalar orderings are equal.

For

\[
\mathcal F_{1/2}(A,B,u,v)
=3\operatorname{tr}(A^3B)
+3u^{\mathsf T}(AB+BA)u
+6u^{\mathsf T}A^2v,
\]

the exact reverse gradients are

\[
\nabla_B\mathcal F_{1/2}
=3A^3+6\operatorname{Sym}(Auu^{\mathsf T}),
\qquad
\nabla_v\mathcal F_{1/2}=6A^2u,
\]

\[
\nabla_u\mathcal F_{1/2}
=6[(AB+BA)u+A^2v],
\]

\[
\begin{aligned}
\nabla_A\mathcal F_{1/2}
={}&3(A^2B+ABA+BA^2)
+6\operatorname{Sym}(Buu^{\mathsf T})\\
&+6\operatorname{Sym}[u(Av)^{\mathsf T}]
+6\operatorname{Sym}[(Au)v^{\mathsf T}].
\end{aligned}
\]

These are exactly the scaled versions of the \(48/24\) gradients above.

### Direct-moment acquisition versus AQ inference

The universal scalar identity and the AQ formula answer different questions.

- **Direct action moment:** observing
  \(\mathbb E[U_a^3V_a]\), \(\mathbb E[U_a^2]\), and
  \(\mathbb E[U_aV_a]\) gives the exact complete K31 scalar for any centered
  law. It is assumption-free but not automatically cheap. For a generic
  correlated nonlinear vector, deriving those action moments from its
  coordinate law contains the same four-leg K4 diagrams and is generically
  \(O(n^4)\).
- **AQ inference:** the trace/vector formula is \(O(n^3)\) or better, but is
  exact only when the actual retained variable lies in
  \(H_1\oplus H_2\) of one shared Gaussian core with synchronized
  \(u_i,A_i\). For an approximate AQ state, the beyond-AQ scalar is an
  uncontrolled residual until its projected \(L^4\) norm or direct action
  moment is observed.
- **Population covariance:** AQ removes within-network tensor acquisition.
  It does not automatically remove an eighth-order population statistic.
  If the retained state conditions on the full oriented \(u_i,A_i\), the AQ
  action is deterministic and its conditional covariance is zero. If only
  invariant summaries are retained, the quartic action remains random in
  the hidden core orientation and its covariance is eighth order.

At latent rank \(r=45\), \(s=r(r+1)/2=1035\) and \(s/n=1.0107\) for
\(n=1024\). The often-quoted \(\sim1.23\) GEMM-equivalents per layer is
valid only for a factorable marginal schedule:

1. the projected core bank \(N_a=\sum_iW_{ia}A_i\) is already carried by the
   shared stream;
2. the endpoint \(Q^{31}\) and covariance conjugation are already paid;
3. no explicit \(P\) or \(R\) bank is formed;
4. \(\sum_{ab}Q_{ab}\operatorname{tr}(N_a^3N_b)\) is evaluated by local
   cubes plus one \(n\times s\) by \(s\times n\) aggregate
   (\(s/n=1.0107\)), with roughly \(0.2\) equivalent local/vector overhead.

If \(N\) is not reusable, add one rectangular \(W\)-to-core mix,
\(s/n\simeq1.01\), for about \(2.24\) per layer. If the reverse covariance
conjugation \(H=W B W^{\mathsf T}\) is not reusable, add two more, for about
\(4.24\) per layer. Explicitly forming \(P\) and two independent \(N/R\)
mixes is about \(1+2s/n=3.02\) before local work and is dominated by the
factorable aggregate. The earliest cost gate is therefore algebraic reuse,
not a timing claim.

Refined packed-state constants use \(d=s+r\), including the H1 vector. At
\(r=45\), \(d=1080\): a post-\(W\) carried bank needs one \(d/n=1.0547\)
aggregate plus two local matrix products
\(2r^3/n^2=0.1738\), totaling \(1.2285\) per layer
(\(12.285\) over ten). Starting from a pre-\(W\) bank needs two rectangular
mixes plus the local products, about \(2.283\) per layer. At \(r=32\),
\(d=560\): the corresponding pre-\(W\) cost is
\(2d/n+2r^3/n^2=1.15625\) per layer, \(16.1875\) over fourteen; the post-\(W\)
cost is \(0.609375\) per layer. Rebuilding the reverse \(H\) conjugation adds
two per active layer. These are algebraic-equivalent censuses, not measured
runtime, and after the novelty correction below they apply only to a defect
proxy or future completion.

### Diagonal-potential/JVP factorization

For any centered layer law and coefficient vector \(w\), define

\[
\Phi(w)=\kappa_4(w^{\mathsf T}X,w^{\mathsf T}X,
w^{\mathsf T}X,w^{\mathsf T}X).
\]

By polarization,

\[
\boxed{
\kappa_4(U,U,U,V)=\tfrac14D\Phi(w)[p].
}
\]

Equivalently,

\[
\kappa_4(A,A,B,C)
=\tfrac12D_Y\{\kappa_4(A,A,Y,Y)\}_{Y=B}[C].
\]

Thus q4(211) is a directional derivative of the Q24 quadratic form, and K31 is a directional derivative of the diagonal q4 potential. Stored Q24 values on a fixed basis do **not** determine this derivative; an evaluable potential/JVP or the corresponding cross-angle state is required.

In the half-quadratic AQ convention,

\[
\Phi(A,u)=3\operatorname{tr}(A^4)+12u^{\mathsf T}A^2u,
\]

\[
g_A=\nabla_A\Phi
=12A^3+24\operatorname{Sym}(Auu^{\mathsf T}),
\qquad
g_u=\nabla_u\Phi=24A^2u.
\]

Therefore, with direction banks \(B=QN\) and \(v=Qu\),

\[
\sum_a\kappa_{31,a}
=\tfrac14\sum_a[\langle g_{A,a},B_a\rangle
+g_{u,a}^{\mathsf T}v_a].
\]

Associativity moves \(Q\) onto the Gram of the local gradient bank with the already-carried \((N,u)\) bank. This proves the factorable one-aggregate schedule behind the conditional \(1.23\)-GEMM census. It also gives a general acquisition principle: if an exact \(O(n^3)\) straight-line program evaluates the scalar diagonal quartic potential, its Jacobian-vector product evaluates K31 at constant-factor asymptotic cost. If the potential itself hides K4 diagrams, differentiation does not remove that barrier.

### Static scalar versus recursive moment closure

The arbitrary-law action is tensor-free at a fixed layer, but this does not make its nonlinear recursive pullback closed. For

\[
\mathcal K(U,V)=\kappa_4(U,U,U,V)
=\mathbb E[U^3V]-3C_{UU}C_{UV},
\]

the pathwise first variation, for centered perturbations, is

\[
\delta\mathcal K
=\mathbb E[\psi_U\,\delta U+\psi_V\,\delta V],
\]

\[
\boxed{
\psi_U=3(U^2V-2C_{UV}U-C_{UU}V),\qquad
\psi_V=U^3-3C_{UU}U.
}
\]

Thus the reverse influence is cubic. Pulling it through a correlated ReLU reset generates the H3/ker-J hierarchy unless a post-reset action-moment oracle or an exact finite cyclic module is supplied. This proves the important separation:

- the universal scalar identity solves the **static information/output** problem;
- it does not solve the **recursive acquisition/state** problem.

Any graph or cost claim that promotes the first result directly to a deployed \(O(n^3)\) recurrence is invalid.

## Exact secant-adjoint alternative to the frozen tangent

The frame defect is not intrinsic to reverse Bellman propagation. For any differentiable layer map \(F_l\), let \(e_l=s_l-\widehat s_l\) and define the path-integrated Jacobian

\[
\overline J_l=\int_0^1DF_l(\widehat s_l+t e_l)\,dt.
\]

Then the fundamental theorem of calculus gives

\[
F_l(s_l)-F_l(\widehat s_l)=\overline J_l e_l
\]

exactly. If the reverse mask is propagated by \(\overline J_l^*\), the nonlinear frame term vanishes identically. The obstruction is observability: \(\overline J_l\) depends on the exact state path. It is sufficient to expose only its action on \(e_l\), or more minimally the scalar
\(\langle B_{l+1},(\overline J_l-L_l)e_l\rangle\).

For the normalized ReLU correlation kernel

\[
\Psi(c)=\frac{\sqrt{1-c^2}+(\pi-\arccos c)c}{\pi},
\]

\[
\Psi'(c)=1-\frac{\arccos c}{\pi},
\qquad
\Psi''(c)=\frac1{\pi\sqrt{1-c^2}}.
\]

Thus the exact scalar secant mask is

\[
\overline D(c,\widehat c)
=\frac{\Psi(c)-\Psi(\widehat c)}{c-\widehat c},
\]

with the derivative value used when the denominator is zero. If both correlations stay in
\(|c|,|\widehat c|\le1-\gamma\), Taylor's theorem yields the entrywise frozen-tangent defect

\[
|\Psi(c)-\Psi(\widehat c)-\Psi'(\widehat c)(c-\widehat c)|
\le
\frac{|c-\widehat c|^2}
{2\pi\sqrt{2\gamma-\gamma^2}}.
\]

After pairing with the reverse mask and conjugating by the known weights, this gives an analytic \(\eta_{\rm frame}\) from a correlation margin and a state-error norm. Near \(|c|=1\) the Hessian diverges, so a uniform margin or an exact secant action is necessary; a global smoothness constant cannot be asserted.

## Graph-ready split dispositions

These results must not be represented by one omnibus node.

### K31_UNIVERSAL_SCALAR_IDENTITY

- kind: theorem;
- status: solved_exact;
- summary: for any centered finite-fourth-moment layer law, the complete K31 tensor action equals \(\sum_a[\mathbb E U_a^3V_a-3\mathbb E U_a^2\mathbb E U_aV_a]\);
- terminal criterion: orientation \(P=WQ_{31}^{\mathsf T}\) and centering convention checked;
- scope warning: solved only as a static endpoint identity; its cubic reverse influence is not a closed ReLU recurrence;
- dependency: reverse covariance adjoint supplies \(Q_{31}\);
- child/dependent: AQ specialization and moment-acquisition interface.

### K31_AQ_CLOSED_FORM

- kind: conditional theorem;
- status: theory_pass_conditional;
- summary: under one synchronized Gaussian \(H_1\oplus H_2\) law, the universal action is the \(48/24\) formula, equivalently the half-quadratic \(3/3/6\) formula;
- terminal criterion: prove the actual carried layer variable has the AQ form in the same gauge at every active layer;
- dependency: K31_UNIVERSAL_SCALAR_IDENTITY;
- failure disposition: beyond-AQ residual remains open, not silently zero.

### K31_AQ_REVERSE_GRADIENTS

- kind: derivation;
- status: solved_exact;
- summary: explicit gradients with respect to both quadratic cores and both linear cores are given in both normalization conventions;
- terminal criterion: symbolic differentiation and convention conversion agree;
- dependency: K31_AQ_CLOSED_FORM;
- dependent: fused provider/reset architecture.

### K31_MOMENT_ACQUISITION_INTERFACE

- kind: observability/cost gate;
- status: open_blocked_high_priority;
- summary: acquire the exact projected \((m_{31},m_{20},m_{11})\) action or prove the AQ inferred route and its factorable aggregate;
- terminal criterion: an exact \(O(n^3)\), label-free source with synchronized provenance and a marginal census respecting reuse;
- dependencies: K31_UNIVERSAL_SCALAR_IDENTITY, optionally K31_AQ_CLOSED_FORM;
- close condition: generic moment acquisition retains a K4/\(n^4\) contraction and no engine-provided scalar exists.

### K31_SECANT_FRAME_CORRECTION

- kind: conditional theorem;
- status: theory_pass_conditional;
- summary: a path-integrated Jacobian makes the Bellman frame telescope exact; the frozen ReLU tangent has the stated correlation-margin Hessian bound;
- terminal criterion: expose the exact secant action or prove a uniform \(\gamma\) margin plus state-error norm that fits \(\eta_{\rm frame}\);
- dependency: reverse Bellman adjoint;
- dependent: terminal amplitude gate.

### K31_BEYOND_AQ_RESIDUAL

- kind: identifiability/gate;
- status: open_unidentified;
- summary: all chaos beyond synchronized affine quadratic, plus out-of-span K22 and state defects, is one endpoint-adjoint residual with an \(L^4\) upper bound but no current observable radius;
- terminal criterion: direct projected action makes it exact, or a white-box \(L^2\) bound makes \(\eta_B\) small enough for
  \(\sqrt{c_B^{\mathsf T}G^+c_B}\ge\eta_B+1.691\times10^{-5}\);
- dependencies: K31_AQ_CLOSED_FORM, K31_MOMENT_ACQUISITION_INTERFACE, and K31_SECANT_FRAME_CORRECTION;
- close condition: two compatible retained-state completions straddle the terminal threshold.

### Required edges

\[
\text{reverse adjoint}\to\text{universal identity}
\to\text{AQ closed form}\to\text{reverse gradients},
\]

\[
\text{universal identity}\to\text{moment acquisition}
\to\text{beyond-AQ residual},
\]

\[
\text{AQ closed form}\to\text{moment acquisition},\qquad
\text{secant frame}\to\text{beyond-AQ residual},
\]

\[
\{\text{reverse gradients},\text{moment acquisition},
\text{beyond-AQ residual}\}\to\text{terminal K31 gate}.
\]

## Candidate exact projective D14 reset

The missing D14 reset can be defined mathematically, although it is a new architecture rather than a deployed-state identity. In the half-quadratic convention let

\[
g(z)=m+u^{\mathsf T}z+Q_A(z),
\qquad
Q_A(z)=\tfrac12(z^{\mathsf T}Az-\operatorname{tr}A),
\qquad f(z)=g(z)_+.
\]

The exact \(H_0\oplus H_1\oplus H_2\) Hermite projection of \(f\) is

\[
\boxed{m^+=\mathbb E f,}
\]

\[
\boxed{
u^+=\mathbb E[zf]
=\mathbb E[1_{\{g>0\}}(u+Az)],
}
\]

\[
\boxed{
A^+=\mathbb E[\nabla_z^2f]
=\Pr(g>0)A
+\mathbb E[\delta(g)(u+Az)(u+Az)^{\mathsf T}].
}
\]

The identities follow from multivariate Gaussian Stein integration and the distributional Hessian

\[
\nabla^2 g_+
=1_{\{g>0\}}A+\delta(g)(u+Az)(u+Az)^{\mathsf T}.
\]

At a linear layer, \(m,u,A\) mix by the physical weight matrix. At a nonlinear layer, the three boxed equations define a synchronized deterministic reset. The carried state is \(n\) scalars, \(n\times r\) linear coefficients, and \(n\times r(r+1)/2\) quadratic coefficients: \(O(n^2)\) state when \(r^2=O(n)\). Physical mixing is \(O(n^3)\).

The exact tangent of this reset has an especially simple operator form. If

\[
\mathcal R(g)=P_{\le2}(g_+),
\]

then for every AQ perturbation \(\delta g\),

\[
\boxed{
D\mathcal R_g[\delta g]
=P_{\le2}[1_{\{g>0\}}\delta g].
}
\]

On the Gaussian \(L^2\) AQ space,

\[
T_g=P_{\le2}M_{1_{\{g>0\}}}P_{\le2}
\]

is self-adjoint positive semidefinite and

\[
\boxed{0\preceq T_g\preceq I,\qquad \|T_g\|_{2\to2}\le1.}
\]

Thus the nonlinear AQ reset is tangent-stable. A single carried word can apply \(T_g\) by the same local surface expectations without materializing the full AQ-to-AQ operator. However, the family of neuron-specific \(T_{g_i}\) can still generate a large cyclic module under dense physical mixing; contraction does not defeat the Burnside/rank obstruction.

In coordinates, for
\[
\delta g=\delta m+\delta u^{\mathsf T}z+Q_{\delta A},\quad
v=u+Az,\quad w=\delta u+\delta A z,
\]

\[
\delta m^+=\mathbb E[1_{\{g>0\}}\delta g],
\]

\[
\delta u^+
=\mathbb E[1_{\{g>0\}}w+\delta(g)\delta g\,v],
\]

\[
\delta A^+
=\mathbb E\!\left[
1_{\{g>0\}}\delta A
+\delta(g)(vw^{\mathsf T}+wv^{\mathsf T}+\delta g\,A)
+\delta'(g)\delta g\,vv^{\mathsf T}
\right].
\]

These formulas define the one-word D14 update/reset precisely.

The required local expectations are white-box affine-quadratic Gaussian surface integrals. With

\[
\chi(t)=\mathbb E e^{it g}
=e^{it(m-\operatorname{tr}A/2)}
\det(I-itA)^{-1/2}
\exp\!\left[-\tfrac12t^2u^{\mathsf T}(I-itA)^{-1}u\right],
\]

put \(C_t=(I-itA)^{-1}\) and \(\mu_t=itC_tu\). Then, with the real value of the Fourier integral understood,

\[
\mathbb E[\delta(g)(u+Az)]
=\frac1{2\pi}\int_{\mathbb R}
\chi(t)(u+A\mu_t)\,dt,
\]

\[
\mathbb E[\delta(g)(u+Az)(u+Az)^{\mathsf T}]
=\frac1{2\pi}\int_{\mathbb R}\chi(t)
\{(u+A\mu_t)(u+A\mu_t)^{\mathsf T}
+AC_tA\}\,dt.
\]

Also

\[
\Pr(g>0)=\tfrac12+\frac1\pi\int_0^\infty
\frac{\operatorname{Im}\chi(t)}t\,dt.
\]

Thus the reset has a one-dimensional integral representation after local \(r\times r\) spectral work; no physical four-index tensor is required. This is a symbolic existence/cost result, not authorization for a quadrature run.

This projected reset co-carries the signed cubic and diagonal quartic potentials:

\[
\kappa_3(g-m)=\operatorname{tr}(A^3)+3u^{\mathsf T}Au,
\]

\[
\kappa_4(g-m)=3\operatorname{tr}(A^4)+12u^{\mathsf T}A^2u.
\]

Consequently the D14 AQ state can supply both the signed-k3 local law and the K31 quartic JVP without materializing H3 or H4 tensors.

### Exact projection loss and a no-label \(L^4\) ceiling

The reset is a projection architecture, not an exact finite ReLU closure. Its discarded residual

\[
r=f-(m^++(u^+)^{\mathsf T}z+Q_{A^+}(z))
\]

has exact squared norm

\[
\boxed{
\|r\|_2^2
=\mathbb E[f^2]-(m^+)^2-\|u^+\|^2
-\tfrac12\operatorname{tr}[(A^+)^2].
}
\]

It also has a fully white-box fourth-norm bound. Since \(|f|\le|g|\) and Gaussian degree-two hypercontractivity gives
\(\|P_{\le2}f\|_4\le3\|P_{\le2}f\|_2\),

\[
\boxed{
\|r\|_4
\le(\mathbb E g^4)^{1/4}
+3\left[(m^+)^2+\|u^+\|^2
+\tfrac12\operatorname{tr}(A^+)^2\right]^{1/2}.
}
\]

For \(x=g-m\),

\[
\kappa_2=\|u\|^2+\tfrac12\operatorname{tr}A^2,\quad
\kappa_3=\operatorname{tr}A^3+3u^{\mathsf T}Au,\quad
\kappa_4=3\operatorname{tr}A^4+12u^{\mathsf T}A^2u,
\]

\[
\mathbb E g^4
=m^4+6m^2\kappa_2+4m\kappa_3+\kappa_4+3\kappa_2^2.
\]

For a physical projection, Minkowski yields

\[
\|R_a\|_4\le\sum_i|W_{ia}|\|r_i\|_4,\qquad
\|S_a\|_4\le\sum_i|P_{ia}|\|r_i\|_4.
\]

Substituting these in the seven-term cumulant-mismatch inequality above produces a target-free \(\eta_{q4}\) ceiling using only AQ reset objects. It may be loose, but it is a legitimate fail-fast certificate: if the ceiling already exceeds the Bellman slack, the projective AQ route cannot be certified without a sharper residual observable.

Because the AQ reset is an **orthogonal Hermite projection**, its residuals \(R,S\in H_{\ge3}\) have zero covariance with all AQ variables \(E,F\in H_1\oplus H_2\). The generic seven-term bound therefore sharpens to

\[
\boxed{
\begin{aligned}
|\Delta\kappa|\le{}&
3re^2f+6r^2ef+r^3f+e^3s\\
&+6re^2s+3r^2es+4r^3s,
\end{aligned}}
\]

where \(e=\|E\|_4,r=\|R\|_4,f=\|F\|_4,s=\|S\|_4\). For example,

\[
\kappa(R,E,E,F)=\mathbb E[RE^2F],
\]

\[
\kappa(R,R,E,F)=\mathbb E[R^2EF]
-\mathbb E[R^2]\mathbb E[EF].
\]

This is the relevant bound on \(\chi_{\rm true}-\chi_{\rm AQ}\) after the novelty correction below.

The leading exact beyond-AQ diagram is the H3--H1-cubed star. If

\[
f_i=\cdots+I_3(C_i)+\cdots,
\qquad C_i=\tfrac1{3!}\mathbb E[D^3f_i],
\]

and \(C_U,C_V\) are its \(W,P\) projections, then

\[
\boxed{
\Delta\kappa_{H3,H1^3}
=18\langle C_U,u\otimes u\otimes v\rangle
+6\langle C_V,u^{\otimes3}\rangle.
}
\]

Equivalently, for derivative tensors \(T=6C\), the coefficients are \(3\) and \(1\). For \(f=g_+\) with AQ \(g\), distributional differentiation gives

\[
\mathbb E[D^3f]
=\mathbb E\!\left[
\delta'(g)(u+Az)^{\otimes3}
+\delta(g)\sum_{\text{three pairs}}
A\otimes(u+Az)
\right].
\]

This identifies the leading true-minus-AQ source, but the modeled same-site H31 component is old D11. Only its exact-minus-assembled defect or positive-depth transport can be novel.

The complete first-order H3 correction around AQ also contains the connected degree pattern \((3,1,2,2)\). For standard Wiener kernels define

\[
\mathcal K(Z;a,b,c)
=6\langle Z,\operatorname{Sym}(a\otimes b\otimes c)\rangle,
\]

\[
\begin{aligned}
\mathcal L(Z;a,M,R)=12\langle Z,\,
&\operatorname{Sym}(Ma\otimes R)
+\operatorname{Sym}(Ra\otimes M)\\
&+2\operatorname{Sym}(a\otimes(M\circ R))\rangle.
\end{aligned}
\]

For

\[
U_0=I_1(u)+I_2(N),\qquad
V_0=I_1(r)+I_2(R),
\]

and perturbations \(I_3(Z)\), \(I_3(Y)\),

\[
\boxed{
\begin{aligned}
\delta\kappa_{31}={}&
3\{\mathcal K(Z;u,u,r)
+\mathcal L(Z;r,N,N)
+2\mathcal L(Z;u,N,R)\}\\
&+\mathcal K(Y;u,u,u)
+3\mathcal L(Y;u,N,N).
\end{aligned}}
\]

The combinatorial factors follow by connected pairing enumeration:

- \((3,1,1,1)\): one star, \(3!=6\);
- \((3,1,2,2)\), H1 leaf on an H2 vertex: one double edge, \(3!1!2!2!/2!=12\);
- H1 leaf on H3: all remaining edges single, \(24\), giving the factor \(2\) in \(\mathcal L\).

For the half-quadratic convention use \(N=A/2\). If \(T=\mathbb E[D^3f]\), then the Wiener kernel is \(Z=T/6\).

Crucially, these scalar actions do not require storing \(T\). With
\(v(z)=u+Az\),

\[
\boxed{
\begin{aligned}
\langle T,\operatorname{Sym}(a\otimes b\otimes c)\rangle
=\mathbb E[&
\delta'(g)(a^{\mathsf T}v)(b^{\mathsf T}v)(c^{\mathsf T}v)\\
&+\delta(g)\{(a^{\mathsf T}Ab)(c^{\mathsf T}v)
+(a^{\mathsf T}Ac)(b^{\mathsf T}v)
+(b^{\mathsf T}Ac)(a^{\mathsf T}v)\}],
\end{aligned}}
\]

\[
\boxed{
\begin{aligned}
\langle T,\operatorname{Sym}(a\otimes M)\rangle
=\mathbb E[&
\delta'(g)(a^{\mathsf T}v)(v^{\mathsf T}Mv)\\
&+\delta(g)\{2(Aa)^{\mathsf T}Mv
+(a^{\mathsf T}v)\operatorname{tr}(AM)\}].
\end{aligned}}
\]

Both are one-dimensional tilted-Gaussian surface integrals after local matrix contractions. Therefore the exact contracted H3-minus-assembled-H31 defect has an \(O(n^3)\) route without an \(r^3\) state. Its novelty still depends on subtracting the modeled H31 scalar and the full old endpoint span.

In the special one-eigen quadratic cavity

\[
g=S_i+\lambda_i(\chi_1^2-1),
\]

the matrix-polynomial subtraction theorem reports that the old square/AQ action cancels the \(\delta(g)\) part and the remaining affordable cubic reset coefficient is

\[
-10 f_{S_i}'(0).
\]

This is the scalar-cavity specialization of the \(\delta'(g)\) surface terms above. It gives an immediate structural fail-fast gate: a threshold-symmetric cavity has \(f_{S_i}'(0)=0\), so this cubic local provider vanishes. Extension of that cancellation from the one-eigen cavity to the full noncommutative core remains to be proved; it must not be assumed from symmetry of an unrelated marginal.

At first Edgeworth order, for a nearly Gaussian cavity of variance \(\sigma^2\) and third cumulant \(\kappa_3\),

\[
f_S'(0)\simeq-\frac{\kappa_3}{2\sigma^5\sqrt{2\pi}},
\qquad
-10f_S'(0)\simeq
\frac{5\kappa_3}{\sigma^5\sqrt{2\pi}}.
\]

So the surviving scalar defect is threshold-local skewness to leading order. It must be residualized against the incumbent signed-k3 response; otherwise the apparent cubic provider may simply rename an old direction.

For a centered Lindeberg quadratic cavity with spectral sums \(s_2,s_3\),
\(\sigma^2=2s_2\), \(\kappa_3=8s_3\), so equivalently

\[
-10f'(0)
=\frac{40\phi(0)s_3}{(2s_2)^{5/2}}.
\]

The matrix-polynomial lane bounds the cubic-to-linear eigenmode ratio by

\[
2.82\,\mu^3,\qquad
\mu=\lambda_{\max}/\sqrt{s_2}.
\]

For a flat \(r=55\) spectrum this is about \(0.0069\) per reset. This is a strong source-level fail-fast ceiling, but it becomes a raw-value no-go only after multiplication by the strict-depth adjoint weights in the same Bellman norm. The escape regimes are threshold mean shift, a spectral spike, or a non-Lindeberg cavity; each also threatens common-coefficient transfer and must be separately certified.

The cancellation is a general reset-jet identity. For

\[
\mathcal R(g)=P_{\le2}\operatorname{ReLU}(g)
\]

and a perturbation \(h\),

\[
D^k\mathcal R_g[h^{\otimes k}]
=P_{\le2}[\operatorname{ReLU}^{(k)}(g)h^k].
\]

Therefore

\[
D\mathcal R_g[h]=P_{\le2}[1_{\{g>0\}}h],
\]

\[
\tfrac12D^2\mathcal R_g[h,h]
=\tfrac12P_{\le2}[\delta(g)h^2],
\]

\[
\boxed{
b_3
=(I-P_{\rm old})\,
\tfrac16P_{\le2}[\delta'(g)h^3].
}
\]

The first jet is the old linear tangent and the second is the old square/AQ-Q24 action. Hence \(b_3\) is the first potentially novel local reset object after binding old-span subtraction. It must be valued only through the future-completion formula above.

For \(g=S+\lambda x\) with the stated one-eigen cavity,

\[
\frac{\lambda^3}{6}\mathbb E[x^4]\mathbb E[\delta'(S)]
=-\frac{60}{6}f_S'(0)\lambda^3
=-10f_S'(0)\lambda^3.
\]

This explains both the coefficient and the symmetry no-go.

#### AQ_CUBIC_JET_SURVIVOR

- status: theory_pass_exact_local_open_value;
- claim: \(b_3=(I-P_{\rm old})P_{\le2}[\delta'(g)h^3]/6\) is the first local reset jet not already assigned to D11 tangent/Q24;
- dependencies: deployed old-jet audit, candidate AQ reset, and K31_AQ_FUTURE_COMPLETION;
- terminal criterion: nonzero threshold-density-slope action outside the complete old depth span, one-word deployable transport, and sufficient strict-depth Bellman gain;
- close conditions: threshold-symmetric cavity, zero old-span residual, cyclic-module rank/cost failure, or master-amplitude ceiling failure.

Do not conflate this node with K31_H3_DEFECT_CONTRACTION. The latter is a same-site latent Wiener-\(I_3\) true-minus-assembled action. The cubic jet is a third Fréchet derivative with respect to a state perturbation, lands back in the AQ \(H_{\le2}\) state, and earns value only through positive-depth transport. Their common \(\delta'\) factor does not make their ownership or state type identical.

### Direct complete-future-costate contraction

The cubic jet need not be materialized. Carry one AQ future costate
\(\lambda_{l,i}\in H_{\le2}\) backward. Because

\[
T_g^*=T_g=P_{\le2}M_{1_{\{g>0\}}}P_{\le2},
\]

the nonlinear pullback maps one AQ polynomial to one AQ polynomial. The linear physical pullback is one dense mix of the \(n\times d\) coefficient bank, \(d=1+r+r(r+1)/2=O(n)\).

After removing every old/local costate component, the exact cubic-jet value at a birth is

\[
\boxed{
\langle\lambda,b_3\rangle
=\frac16\mathbb E\!\left[
((I-P_{\rm old})\lambda)(z)\,
\delta'(g(z))\,h(z)^3
\right].
}
\]

Projection \(P_{\le2}\) disappears from the scalar because
\(\lambda\in H_{\le2}\). The expectation is another local tilted-Gaussian surface integral. Thus a forward AQ state plus a backward AQ costate uses two \(n\times d\) physical mixes per active layer and no \(r^3\) state. This statement concerns the latent AQ quotient. Evaluating a generic physical all-distinct surface query may still require \(\operatorname{diag}(CMC^{\mathsf T})\), i.e. one additional physical GEMM per layer, unless \(C\) has a supplied low-rank factor or the product is already billed.

This is the precise complete-future-costate residual oracle. Generic neuron-specific gates do not enlarge its state beyond the full AQ bank; Burnside is a no-go only for compressing that bank to a smaller common word module. The remaining binding gates are:

1. the costate must exclude zero-length and every old depth-arena word;
2. the desired future-dark source must factor through \(H_{\le2}\);
3. local surface evaluation and the two packed mixes must fit billing;
4. the adjoint-weighted cubic amplitude must clear the master threshold.

For a source \(S_{\rm AD}\), same-site observation map \(O_l\), and full future query \(q_l\), the exact quotient test is

\[
q_l^\perp
=\left[I-O_l^*(O_lO_l^*)^+O_l\right]q_l.
\]

The AQ-costate fusion is complete for the queried source iff

\[
\boxed{\langle q_l^\perp,S_{\rm AD}\rangle=0.}
\]

If this scalar is nonzero, the source has a genuinely future-dark component outside the local AQ consumer and an additional query contraction is necessary. A full-rank ambient query does not prove failure because the lawful source manifold is only \(O(n^2)\)-dimensional.

The certified 18.7% tangent replay injects banked \(dK_{21},dK_{22},dK_{31}\) planes directly into the covariance tangent at each L5--L14. Therefore its frozen-chain provenance favors the local-factor case; it is not evidence that one H3 birth is transported before later observation. The \(q^\perp\) test becomes binding only for the new self-consistent/coadapted completion.

For the structured local all-distinct H3 source, the factor test can be proved exactly. Define the cubic AQ test polynomial

\[
f_{x,M}(h)=(x^{\mathsf T}h)(h^{\mathsf T}Mh).
\]

Then

\[
\partial_{abc}f_{x,M}
=2(x_aM_{bc}+x_bM_{ac}+x_cM_{ab}).
\]

For a symmetric all-distinct source \(S\),

\[
\boxed{
\frac16\langle S,\nabla^3f_{x,M}\rangle
=S(x,M).
}
\]

Thus every query of the form used by the local covariance tangent is exactly a cubic AQ surface pullback. After collision ownership and \((I-P_{\rm old})\) subtraction, \(q^\perp=0\) on this structured **local** source manifold. This proves ownership/factorization of the scalar; it does **not** prove sub-GEMM arithmetic. The coordinate evaluation can still contain \(\operatorname{diag}(CMC^{\mathsf T})\). The identity also does not cover a transported future-dark functional that no longer has the same-site cubic test form.

The corrected \(r=45\), L5--L14 census is:

- packed forward/reverse AQ bank and local products: \(22.83\);
- generic exact \(CM\): \(10\);
- reverse \(H\) conjugations, if not reusable: \(20\).

Thus the H-reuse branch is \(32.83<49\), while the H-rebuild branch is
\(52.83>49\). The earlier \(42.83\) figure omitted \(CM\) and is not a generic total.
At \(r=32\) over fourteen layers the analogous generic totals are
\(16.1875+14=30.1875\) with H reuse and
\(16.1875+14+28=58.1875\) with H rebuild.

An exact structural escape is a supplied factor

\[
C=ZL^{\mathsf T},\qquad \operatorname{rank}C=k.
\]

Then

\[
\operatorname{diag}(CMC^{\mathsf T})
=\operatorname{diag}[Z(L^{\mathsf T}ML)Z^{\mathsf T}]
\]

costs, in dense-GEMM equivalents,

\[
\frac{k}{n}+2\left(\frac{k}{n}\right)^2
\]

per layer after the factor is available. In the H-rebuild branch the remaining
per-layer allowance is

\[
\frac{49-42.83}{10}=0.617.
\]

Writing \(x=k/n\), the exact gate

\[
x+2x^2\le0.617
\]

gives \(x\le0.359\), or \(k\lesssim367\) at \(n=1024\). A clean asymptotic pass is \(k=O(\sqrt n)\). For a Schur product
\(C=C_{\rm cov}\odot C_{\rm gate}\), the available Khatri--Rao factor has rank at most
\(k_{\rm cov}k_{\rm gate}\); that product, not either marginal rank, must pass the gate. A generic rank-\(n\) \(C\) costs one GEMM per layer.

The campaign billing audit reports that Site-A/B covariance injections can ride an existing deployed conjugation at negligible marginal bill. That supports, but does not by itself prove, the H-reuse branch here: the exact reverse-costate lineage must identify the same stored operands and operation. A forward in-place injection receipt is not automatically a reverse-\(H\) buffer identity.

For a one-word design, the lawful white-box provider-selection problem is the symmetric cubic form

\[
\mathcal J(h)
=\frac16\mathbb E[\lambda(z)\delta'(g(z))h(z)^3],
\qquad \|h\|_{L^2}=1,
\]

where \(\lambda\) is the old-span-residualized future costate. Its largest absolute Z-eigenvalue is the best one-word source value. A stationary provider obeys

\[
\boxed{
P_{\le2}[\lambda\,\delta'(g)h^2]
=6\mathcal J(h)\,h.
}
\]

This is a target-adjoint but label-free optimization. It does not justify adapting the word after seeing task labels, and a common scheduled word across layers must additionally satisfy the layer-dependent Krylov/wedge condition.

## Binding novelty correction — total same-site K31 is old

The deployed D11 lineage already assembles the modeled same-site K31 plane: its connected-plane step forms K31 from analytic \(m_{31}\) with the mean and \(3\,\mathrm{variance}\times\mathrm{covariance}\) disconnections removed, and its star reset combines off-diagonal K31 with diagonal K4. In the campaign decomposition this is the CP4+star+H31 old action.

Therefore the scalar

\[
\chi_l^{\rm assembled}
=\langle Q_l^{31},D_{l,\rm assembled}^{31}\rangle
\]

is old-span. The universal scalar identity and AQ closed form are valuable acquisition/verification identities, but evaluating the same modeled total by a new trace schedule creates no raw-MSE novelty.

The only lawful same-site target is the exact action defect

\[
\boxed{
\rho_{31,l}
=\langle Q_l^{31},
D_{l,\rm true}^{31}-D_{l,\rm assembled}^{31}\rangle
=\chi_l^{\rm true}-\chi_l^{\rm assembled}.
}
\]

Direct action moments give

\[
\chi_l^{\rm true}
=\sum_a[\mathbb E U_{l,a}^3V_{l,a}
-3\mathbb E U_{l,a}^2\mathbb E U_{l,a}V_{l,a}]
\]

and hence exact \(\rho_{31,l}\). The AQ proxy is

\[
\boxed{
\widehat\rho_{31,l}^{\rm AQ}
=\chi_l^{\rm AQ}-\chi_l^{\rm assembled}.
}
\]

Its approximation error is exactly

\[
\rho_{31,l}-\widehat\rho_{31,l}^{\rm AQ}
=\chi_l^{\rm true}-\chi_l^{\rm AQ},
\]

which is the quantity bounded by the seven-term \(L^4\) mismatch inequality and the projective-reset residual ceiling above. Thus those bounds remain useful, but they certify a defect proxy rather than total K31 value.

If \(d=\rho_{31}^{\rm true}\), \(f=\widehat\rho_{31}^{\rm AQ}\), and
\(\|d-f\|\le\eta_{>AQ}\), then for every affordable provider span
\(\mathcal S\),

\[
\boxed{
(\|P_{\mathcal S}f\|-\eta_{>AQ})_+^2
\le\|P_{\mathcal S}d\|^2
\le(\|P_{\mathcal S}f\|+\eta_{>AQ})^2.
}
\]

This is both a fire and a fail-fast gate. If the upper amplitude is below
\(1.691\times10^{-5}\), the defect route cannot supply the nominal quartic
shortfall; if the lower amplitude clears it, the route is certified under the
Bellman identification of \(d\) with the ideal missing value.

If the D11 assembled polynomial is algebraically identical to AQ in the same carried state, then
\(\widehat\rho_{31,l}^{\rm AQ}=0\). In that case synchronized AQ has zero same-site novel direction; only beyond-AQ action or changed future dynamics can help.

Every defect candidate must also be Schur-residualized against the old endpoint feature span. Subtracting CP3/Q24 collision receipts does not change the physical reverse mask \(Q^{31}\), which remains generically full rank:

\[
Q^{31}
=\tfrac16[H\odot G^{31}+(H\odot G^{13})^{\mathsf T}]
=\tfrac13H\odot G^{31}
\]

when \(H=H^{\mathsf T}\) and \(G^{13}=(G^{31})^{\mathsf T}\).

### Separate future-visible AQ completion

A new projective AQ reset can still be novel even when its local K31 scalar is duplicate. Let

\[
\delta s_l
=F_l^{\rm AQ}(s_l)-F_l^{\rm D11}(s_l)
\]

be the reset/state-transition difference after removing every component already injected by the D11 K31/Q24 assembly. Its lawful value is downstream only:

\[
\boxed{
\Delta_{\rm future}
=\sum_l\langle B_{l+1}^{\rm downstream},\delta s_l\rangle
}
\]

for the linear tangent, or the corresponding exact secant-adjoint pairing. No same-site \(\chi_l\) may be added again. This is the co-adaptation route: AQ changes the state seen by later nonlinear resets, and only that future response can be credited.

A sharp no-double-counting expansion is available. Let the new AQ word have local births \(b_j\), transport

\[
z_{l+1}=T_lz_l+b_l,
\]

and local value consumer \(C_lz_l\). The assembled D11 action already contains every zero-length term
\(\langle B_j,C_jb_j\rangle\). The genuinely new response is exactly

\[
\boxed{
\Delta_{\rm future}
=\sum_{j<k}
\left\langle B_k,\,
C_kT_{k-1}\cdots T_{j+1}b_j
\right\rangle.
}
\]

Equivalently, propagate the reverse word

\[
\lambda_j=T_j^*\lambda_{j+1}+C_j^*B_j
\]

and subtract all zero-length birth-consumer pairings. This formula turns “co-adaptation” into a falsifiable object: only strict positive-depth paths count. A common/small cyclic module for the \(T_l^*\) words yields the proposed D14 schedule; generic correlated resets generate the full module by Burnside and close the route on state/cost.

Because the deployed D11 lineage already contains a depth arena, zero-length removal is necessary but not sufficient. Let \(\mathcal O\) be the full old endpoint-response/Krylov span. The lawful candidate is

\[
\boxed{
\Delta_{\rm future}^{\rm novel}
=(I-P_{\mathcal O})\Delta_{\rm future}.
}
\]

Equivalently, every positive-depth word already represented by the old arena must be deleted before value credit. The strict-depth formula is the pre-residual source; Schur residualization against the complete old module is binding.

### Superseding graph nodes

#### K31_SAME_SITE_TOTAL_DUPLICATE

- status: closed_duplicate;
- claim: \(\chi^{\rm assembled}\), including its AQ trace re-expression, is already D11 CP4+star+H31;
- dependencies: K31_UNIVERSAL_SCALAR_IDENTITY and deployed assembled-plane provenance;
- terminal criterion: exact component/normalization audit;
- disposition: no source fit or raw-gain credit.

#### K31_TRUE_DEFECT_INTERFACE

- status: open_high_priority;
- claim: acquire \(\rho_{31}=\chi_{\rm true}-\chi_{\rm assembled}\), not total \(\chi\);
- dependency: K31_UNIVERSAL_SCALAR_IDENTITY and K31_SAME_SITE_TOTAL_DUPLICATE;
- terminal criterion: exact direct projected moments or another white-box action oracle, followed by old-span residualization;
- no-go: generic exact-moment acquisition remains K4/\(n^4\).

#### K31_AQ_DEFECT_PROXY

- status: theory_pass_conditional;
- claim: \(\widehat\rho_{31}^{AQ}=\chi_{AQ}-\chi_{\rm assembled}\) with certified error \(\eta_{>AQ}\);
- dependencies: K31_AQ_CLOSED_FORM, K31_TRUE_DEFECT_INTERFACE, and K31_BEYOND_AQ_RESIDUAL;
- terminal criterion:
  \[
  \sqrt{c_{\widehat\rho}^{\mathsf T}G^+c_{\widehat\rho}}
  \ge\eta_{>AQ}+1.691\times10^{-5};
  \]
- immediate close condition: \(\chi_{AQ}\equiv\chi_{\rm assembled}\).

#### K31_AQ_FUTURE_COMPLETION

- status: open_architecture;
- claim: future-visible value of the exact projective AQ reset relative to D11;
- dependencies: candidate D14 reset and K31_SECANT_FRAME_CORRECTION;
- terminal criterion: a no-double-counting transition defect \(\delta s_l\), exact downstream adjoint, deployable recurrence, and endpoint gain above the master deficit;
- scope: local same-site K31 value is excluded.

#### K31_H3_DEFECT_CONTRACTION

- status: theory_pass_conditional_high_priority;
- claim: the complete first-order true H3-minus-AQ K31 defect is the displayed \(\mathcal K/\mathcal L\) scalar and each required tensor contraction is a local surface integral;
- dependencies: candidate AQ reset, K31_TRUE_DEFECT_INTERFACE, and deployed H31 provenance;
- terminal criterion: subtract the assembled H31 kernel/action and full old endpoint span, prove the surface-integral acquisition/census, and apply the robust defect gate;
- state/cost: no \(r^3\) state; \(O(n^3)\) directional contractions under constant-complexity local integral evaluation;
- close condition: exact-minus-assembled contracted H3 action is zero/old-span or its upper amplitude is below \(1.691\times10^{-5}\).

The earlier K31_MOMENT_ACQUISITION_INTERFACE node must now point to K31_TRUE_DEFECT_INTERFACE. The earlier AQ total-action node is a mathematical specialization/provider implementation, not a novelty node.

## Endpoint leverage theorem for source acquisition

Let \(P\) be an acquired source rowspace, \(x\) the true source, and \(a\) the endpoint adjoint. Even

\[
\tau=\frac{\|Px\|^2}{\|x\|^2}\to1
\]

does not imply scalar capture. In \(\mathbb R^2\), take

\[
x=\sqrt\tau\,e_1+\sqrt{1-\tau}\,e_2,\quad
P=e_1e_1^{\mathsf T},\quad a=e_2.
\]

Then \(\langle a,Px\rangle=0\) while
\(\langle a,x\rangle=\sqrt{1-\tau}\). Hence every source-only lower bound on K31 scalar novelty is zero.

The sharp elementary leverage receipt is

\[
|\ell-\widehat\ell|
=|\langle a,(I-P)x\rangle|
\le\|(I-P)a\|\,\|x\|\sqrt{1-\tau}.
\]

With

\[
\lambda=\frac{|\langle a,x\rangle|}{\|a\|\|x\|},
\qquad
\mu=\frac{\|(I-P)a\|}{\|a\|},
\]

\[
\frac{|\ell-\widehat\ell|}{|\ell|}
\le\frac{\mu\sqrt{1-\tau}}{\lambda}.
\]

A sign/amplitude certificate therefore needs
\(\mu\sqrt{1-\tau}<\lambda\). The strongest structural sufficient condition is \(Pa=a\), which makes the scalar exact regardless of source tail. Population versions replace \(\mu\) by the restricted endpoint operator norm \(\|\mathcal A(I-P)\|/\|\mathcal A\|\). Source observability without this adjoint-angle receipt is support-only.

## Exact weighted-surface cancellation criterion for the cubic reset jet

The scalar marginal condition \(f_g'(0)=0\) is not the correct cancellation test once the cubic reset birth is paired with a Bellman costate. Let \(\phi\in\mathcal H_{\le2}\) be any target-quotient test and let \(h\in\mathcal H_{\le2}\) be the perturbation direction. Define the distributional weighted surface density

\[
w_{\phi,h}(t)
:=\mathbb E\!\left[\phi(z)h(z)^3\,\delta(g(z)-t)\right].
\]

For the first post-square reset birth

\[
b_3(h)=\frac16(I-P_{\rm old})P_{\le2}[\delta'(g)h^3],
\]

every quotient pairing is exactly

\[
\boxed{
\langle\phi,b_3(h)\rangle
=-\frac16w_{\phi,h}'(0),
\qquad \phi\in (I-P_{\rm old})\mathcal H_{\le2}.
}
\]

Indeed, \(\delta'(g)=\partial_g\delta(g)\), while
\(\partial_t\delta(g-t)=-\partial_g\delta(g-t)\), so differentiation of \(w\) gives the displayed sign. Therefore

\[
b_3(h)=0\quad\hbox{in the complete target quotient}
\iff
w_{\phi,h}'(0)=0
\quad\hbox{for every quotient test }\phi.
\]

This is strictly stronger than the marginal-density condition \(f_g'(0)=0\): that condition covers only \(\phi h^3\equiv1\), which is generally not among the required weighted tests.

### Exact symmetry certificate

A sufficient cancellation certificate is a Gaussian-measure-preserving involution \(J\) such that

\[
g(Jz)=-g(z),
\qquad
\phi(Jz)h(Jz)^3=\phi(z)h(z)^3
\]

for every target-quotient \(\phi\). Changing variables by \(J\) makes \(w_{\phi,h}(t)=w_{\phi,h}(-t)\), hence \(w_{\phi,h}'(0)=0\). For a centered pure quadratic cavity \(g(z)=Q_A(z)\), this requires, up to the trace convention,

\[
J^{\mathsf T}AJ=-A,
\]

so the nonzero spectrum of \(A\) must occur in matched \(+/-\) pairs with compatible eigenspaces. Generic centering does not imply this anticommuting symmetry. Thus exact cubic-jet cancellation is exceptional; approximate Lindeberg smallness is only an amplitude prior and cannot close the node symbolically.

### Graph node: AQ_CUBIC_JET_WEIGHTED_SYMMETRY

- status: solved_exact_gate;
- claim: the complete quotient cubic reset birth vanishes iff all weighted surface slopes \(w_{\phi,h}'(0)\) vanish;
- dependency: AQ_CUBIC_JET_SURVIVOR and the full old-span projector;
- sufficient close certificate: a measure-preserving sign-flip involution satisfying the two displayed identities for every quotient test;
- no-go boundary: \(f_g'(0)=0\) alone is insufficient;
- next question: prove a nonzero weighted slope for one deployable quotient costate, or upper-bound the entire weighted-slope operator in the Bellman norm.

## Coarea/Stein reduction of the cubic \(\delta'\) action

The weighted slope need not be evaluated by differentiating a density numerically. Assume \(g\) is \(C^2\) and that \(\nabla g\ne0\) on the zero surface. Put

\[
q(z)=\nabla g(z),\qquad
d(z)=q(z)^{\mathsf T}q(z),\qquad
v(z)=q(z)/d(z),
\]

so \(v\cdot\nabla g=1\). For Gaussian divergence

\[
\operatorname{div}_{\gamma}a
=\operatorname{div}a-z^{\mathsf T}a,
\]

integration by parts applied to \(Fv\,\delta(g)\) gives

\[
0=\mathbb E\!\left[
\operatorname{div}_{\gamma}(Fv)\delta(g)
+Fv\cdot\nabla\delta(g)\right].
\]

Since \(v\cdot\nabla\delta(g)=\delta'(g)\), the exact identity is

\[
\boxed{
\mathbb E[F\,\delta'(g)]
=-\mathbb E\!\left[\delta(g)\,
\operatorname{div}_{\gamma}(Fv)\right].
}
\]

Thus the cubic Bellman pairing, with \(F=\phi h^3\), is an ordinary zero-surface expectation:

\[
\boxed{
\langle\phi,b_3(h)\rangle
=-\frac16\mathbb E\!\left[
\delta(g)\operatorname{div}_{\gamma}
\left(\phi h^3\frac{\nabla g}{\|\nabla g\|^2}\right)
\right].
}
\]

For the half-quadratic affine--quadratic gate

\[
g(z)=m+u^{\mathsf T}z+\frac12(z^{\mathsf T}Az-\operatorname{tr}A),
\qquad q=u+Az,
\]

one has

\[
\operatorname{div}v
=\frac{\operatorname{tr}A}{d}
-\frac{2q^{\mathsf T}Aq}{d^2},
\qquad
z^{\mathsf T}v=\frac{z^{\mathsf T}q}{d}.
\]

Therefore, for arbitrary smooth \(F\),

\[
\boxed{
\mathbb E[F\delta'(g)]
=-\mathbb E\!\left[
\delta(g)\left\{
\frac{\nabla F\cdot q}{d}
+F\left(
\frac{\operatorname{tr}A-z^{\mathsf T}q}{d}
-\frac{2q^{\mathsf T}Aq}{d^2}
\right)\right\}\right].
}
\]

For \(F=\phi h^3\),

\[
\nabla F=h^3\nabla\phi+3\phi h^2\nabla h.
\]

This is a tensor-free, label-free acquisition formula on the AQ state. It uses the same zero-level surface as the exact AQ reset, although it asks for additional rationally weighted surface moments; those moments are not implied by the currently carried reset summaries.

### Linear-gate specialization

If \(A=0\), then \(q=u\) and \(d=\|u\|^2\) are constant. On the surface \(g=0\), \(z^{\mathsf T}u=-m\), hence

\[
\boxed{
\mathbb E[F\delta'(m+u^{\mathsf T}z)]
=-\frac1{\|u\|^2}
\mathbb E\!\left[\delta(m+u^{\mathsf T}z)
\{u^{\mathsf T}\nabla F+mF\}\right].
}
\]

All remaining expectations are conditional Gaussian polynomial moments on a hyperplane and therefore have finite Wick formulas for AQ \(\phi,h\).

### Boundary and fail-fast interpretation

The only analytic obstruction in this reduction is a critical zero,
\(\nabla g=0=g\), where the \(d^{-1}\) factors are singular. A uniform certificate follows if

\[
\|\nabla g\|\ge\sigma_*>0
\quad\hbox{on }g^{-1}(0),
\]

because the displayed formula then bounds the cubic action by ordinary surface norms with factors \(\sigma_*^{-2}\) and \(\sigma_*^{-4}\). Without such transversality, no uniform amplitude bound can follow from the marginal density alone: quadratic gates can have arbitrarily sharp critical zero surfaces.

### Generic noncancellation

The weighted-slope functional is not identically zero even for a symmetric marginal gate. In one dimension take \(g(z)=z\), \(h(z)=1\), and \(\phi(z)=z\). Then

\[
w_{\phi,h}(t)=t\,\varphi(t),
\qquad w_{\phi,h}'(0)=\varphi(0)\ne0,
\]

although \(f_g'(0)=0\). On any regular finite-dimensional AQ parameter stratum these surface integrals are real analytic. Since one explicit parameter point is nonzero, exact cancellation lies in a proper analytic zero set unless forced by the involution certificate above. Therefore “centered/symmetric gate” is not a lawful generic close argument.

### Graph nodes

#### AQ_CUBIC_JET_STEIN_SURFACE

- status: solved_exact_interface;
- claim: every \(\delta'\) cubic Bellman action equals the displayed ordinary \(\delta\)-surface expectation;
- dependency: AQ_CUBIC_JET_WEIGHTED_SYMMETRY and a regular zero surface;
- acquisition: AQ coefficients plus rationally weighted boundary moments;
- novelty: still requires full old-depth residualization.

#### AQ_CUBIC_JET_CRITICAL_SURFACE_GATE

- status: open_binding;
- pass condition: a white-box lower bound \(\|\nabla g\|\ge\sigma_*>0\) on the active zero surface, or an exact stratified treatment of its critical set;
- no-go: marginal density and low-order cumulants alone do not control the \(d^{-2}\) curvature term;
- next question: determine whether the deployed AQ gates have a structural transversality margin or whether their zero surfaces contain quadratic critical points.

## Closed-form AQ transversality margin

The preceding critical-surface gate is symbolically decidable. Let \(A^+\) be the Moore--Penrose inverse and decompose

\[
u=u_R+u_0,\qquad
u_R=AA^+u,\qquad
u_0=(I-AA^+)u\in\ker A.
\]

After the range shift \(z_R=y-A^+u_R\), the half-AQ gate and its gradient become

\[
g(z)=\delta+\frac12y^{\mathsf T}Ay+u_0^{\mathsf T}z_0,
\qquad
\nabla g=Ay+u_0,
\]

where \(z_0\in\ker A\) and

\[
\boxed{
\delta=m-\frac12\operatorname{tr}A
-\frac12u_R^{\mathsf T}A^+u_R.
}
\]

Because \(\operatorname{range}A\perp\ker A\),

\[
\|\nabla g\|^2=\|Ay\|^2+\|u_0\|^2.
\]

### Case 1: \(u_0\ne0\)

The kernel coordinate \(z_0\) can satisfy \(g=0\) for any \(y\), and

\[
\boxed{
\inf_{g=0}\|\nabla g\|^2=\|u_0\|^2.
}
\]

Thus any nonzero affine component in \(\ker A\) supplies an exact global transversality margin.

### Case 2: \(u_0=0\)

The boundary equation is

\[
y^{\mathsf T}Ay=-2\delta.
\]

If \(\delta=0\), the shifted critical point \(y=0\) lies on the boundary and the transversality margin is zero. If \(\delta\ne0\), let

\[
\lambda_*(\delta)
=\min\{|\lambda|:\lambda\in\operatorname{spec}(A),\ 
\lambda\delta<0\}.
\]

If this set is empty, the zero surface is empty and every \(\delta(g)\) or \(\delta'(g)\) action vanishes. Otherwise,

\[
\boxed{
\inf_{g=0}\|\nabla g\|^2
=2|\delta|\,\lambda_*(\delta).
}
\]

Proof: in an eigenbasis, write \(t_j=y_j^2\ge0\). The constraint is
\(\sum_j\lambda_jt_j=-2\delta\) and the objective is
\(\sum_j\lambda_j^2t_j\). This linear program is minimized by placing all mass on the eigenvalue of required sign and smallest magnitude; adding an eigenvalue of the opposite sign strictly increases the objective.

The degenerate constant case \(A=0=u\) is separate: the boundary is empty for \(m\ne0\) and is all of space for \(m=0\).

Consequently the exact white-box surface margin is

\[
\boxed{
\sigma_*^2=
\begin{cases}
\|u_0\|^2,&u_0\ne0,\\
2|\delta|\lambda_*(\delta),&u_0=0,\ \delta\ne0,\ g^{-1}(0)\ne\varnothing,\\
0,&u_0=0,\ \delta=0,
\end{cases}}
\]

with the empty-boundary branch closed at zero action. This replaces the open transversality search by an eigenspectrum/pseudoinverse receipt already intrinsic to the AQ state.

### Graph update: AQ_CUBIC_JET_CRITICAL_SURFACE_GATE

- status: solved_exact_geometry;
- pass receipt: the displayed \(\sigma_*^2>0\);
- zero-action receipt: empty zero surface;
- singular branch: \(\delta=0=u_0\), requiring a stratified critical-surface treatment or direct closure;
- remaining open quantity: the weighted surface norm of \(\phi h^3\) and its derivatives after complete old-span projection.

## Minimal coefficient/norm certificate for the cubic AQ birth

Let

\[
F(z)=\frac16\delta'(g(z))h(z)^3.
\]

The complete projected birth \(b_3=P_{\le2}F\) is determined by only

\[
c=\mathbb E[F],\qquad
p=\mathbb E[zF],\qquad
S=\mathbb E[(zz^{\mathsf T}-I)F].
\]

In the half-quadratic convention,

\[
\boxed{
b_3(z)=c+p^{\mathsf T}z
+\frac12(z^{\mathsf T}Sz-\operatorname{tr}S).
}
\]

Hermite orthogonality gives its exact Gaussian energy:

\[
\boxed{
\|b_3\|_2^2
=c^2+\|p\|^2+\frac12\|S\|_F^2.
}
\]

Each coefficient is an ordinary boundary expectation by the Stein-surface identity above, taking \(F\)'s prefactor respectively against \(1,z\), and \(zz^{\mathsf T}-I\). No H3 tensor is involved.

Let \(\Pi_{\rm old}\) denote the old-span projector in this coefficient space, using metric

\[
G_{\le2}=\operatorname{diag}(1,I,\tfrac12 I_{\rm sym}).
\]

Then the exact local target-free novelty radius is

\[
\boxed{
\eta_{3,l}
=\|(I-\Pi_{{\rm old},l})b_{3,l}\|_{G_{\le2}}.
}
\]

For a pulled-back Bellman costate \(\lambda_l\),

\[
|\langle\lambda_l,(I-\Pi_{\rm old})b_{3,l}\rangle|
\le
\|(I-\Pi_{\rm old})\lambda_l\|_{G_{\le2}}
\eta_{3,l}.
\]

A safe complete-path radius is therefore

\[
\boxed{
\eta_B
\le\sum_l
\|(I-\Pi_{{\rm old},l})\lambda_l\|_{G_{\le2}}\eta_{3,l},
}
\]

or the corresponding quadratic Gram contraction when cross-layer covariances are retained. If this upper radius is below \(1.691\times10^{-5}\), the cubic-AQ route cannot meet the quartic master requirement even at perfect target angle.

This is also a minimal additional observable statement. If the current state does not expose \(c,p,S\) or an equivalent exact reset JVP, no amplitude-only Bellman certificate is derivable from lower AQ moments: signed \(\delta'\)-surface orientation can change while those moments remain fixed.

### Graph node: AQ_CUBIC_JET_COEFFICIENT_RADIUS

- status: solved_exact_certificate;
- state: one scalar, one latent vector, and one symmetric latent matrix per local birth;
- inputs: rationally weighted AQ surface moments or the third directional JVP of the exact AQ reset;
- output: exact local \(G_{\le2}\) norm and a safe complete-path \(\eta_B\);
- terminal close: \(\eta_B<1.691\times10^{-5}\);
- limitation: it is an upper/amplitude gate, not a lower target-angle certificate.

The numerical threshold applies only after the layerwise scalar bound is expressed in the deployed response normalization and its corpus \(L^2\) norm is taken. A latent-state norm by itself must not be compared directly with \(1.691\times10^{-5}\).

## Independent normalization audit of the \(M^2\) cubic boundary birth

For the unscaled quadratic convention

\[
g=b+a^{\mathsf T}z+z^{\mathsf T}Mz-\operatorname{tr}M,
\qquad v=a+2Mz,
\]

the reset cubic tensor is

\[
T=\mathbb E[D^3\operatorname{ReLU}(g)]
=\mathbb E[\delta'(g)v^{\otimes3}+6\delta(g)J_v(M)],
\]

where \(J_v(M)=\operatorname{Sym}(v\otimes M)\) uses averaged three-slot symmetrization. The factor six is exact because the Hessian of \(g\) is \(2M\).

Let \(\sigma=\|a\|\), \(e=a/\sigma\), \(\alpha=b/\sigma\), and perturb \(M\mapsto\varepsilon M\). At order zero,

\[
T_0=-\sigma\alpha\varphi(\alpha)e^{\otimes3}.
\]

At order one, writing \(m=e^{\mathsf T}Me\), direct expansion gives

\[
T_1=\varphi(\alpha)\left[
mH_4(\alpha)e^{\otimes3}
+6H_2(\alpha)\operatorname{Sym}(e^{\otimes2}\otimes Me)
+6J_e(M)\right].
\]

The constants follow from

\[
\mathbb E[(x^2-1)\delta''(b+\sigma x)]
=\frac{H_4(\alpha)\varphi(\alpha)}{\sigma^3},
\qquad
\mathbb E[x\delta'(b+\sigma x)]
=\frac{H_2(\alpha)\varphi(\alpha)}{\sigma^2}.
\]

Two independent reductions audit the reported full second-order expression.

### One-dimensional reduction

For \(r=1\), put \(M=m\) and \(q=x^2-1=H_2(x)\). The coefficient of \(\varepsilon^2m^2\) in
\(\delta'(g)v^3+6\varepsilon m\delta(g)v\) is

\[
\frac{\sigma^3}{2}q^2\delta'''
+6\sigma^2xq\delta''
+12\sigma x^2\delta'
+12x\delta
+6\sigma q\delta'.
\]

Using

\[
q^2=H_4+4H_2+2,\quad
xq=H_3+2H_1,\quad
x^2=H_2+1,
\]

its expectation is

\[
\boxed{
-\frac{m^2\varphi(\alpha)}{\sigma}
\left[\frac12H_7+8H_5+31H_3+24H_1\right]e^{\otimes3}.
}
\]

This equals the specialization of the complete \(M^2\) formula term by term.

### Purely transverse reduction

If \(Me=0\), then the quadratic perturbation is independent of the gate coordinate at zeroth order. Wick contraction leaves

\[
\boxed{
T_2=-\frac{\varphi(\alpha)}{\sigma}
\left[
\operatorname{tr}(M^2)H_3\,e^{\otimes3}
+12\alpha J_e(M^2)\right],
}
\]

again exactly the specialization of the complete expression.

The Fourier representation also has the correct orientation. With
\(K_t=I-2itM\), the tilted mean of \(v\) is

\[
a+2itM K_t^{-1}a=K_t^{-1}a,
\]

and its tilted covariance is \(4MK_t^{-1}M\); the \(+it\) multiplying the gradient-cube term comes from \(-\mu_3'(0)\).

Therefore the \(M^2\) tagged-birth formula is normalization- and sign-correct. The notation \(T^{(2)}\) must denote the coefficient of \(\varepsilon^2\), rather than the unhalved second derivative. Novelty still begins only after subtracting the deployed order-zero/order-one H31 span and the complete depth arena.

### Exact carrier caveat for the \(M^2\) tags

A short nonlinear description per unit is not a short linear carrier after dense mixing. The family

\[
J_e(S)=\operatorname{Sym}(e\otimes S)
\]

already spans all of \(\operatorname{Sym}^3(\mathbb R^r)\). Indeed every coordinate basis tensor has the form

\[
\operatorname{Sym}(e_i\otimes e_j\otimes e_k)
=J_{e_i}(\operatorname{Sym}(e_j\otimes e_k)).
\]

Restricting \(S\) to square/PSD forms does not save the linear span: rank-one matrices \(vv^{\mathsf T}=M^2\) span \(\operatorname{Sym}(r)\) under signed linear combinations. Therefore unit-specific terms such as \(J_{e_i}(M_i^2)\), \(J_{e_i}(M_iPM_i)\), and \(J_{\mu_i}(M_i)\) can generate the full cubic space when unit eigenframes vary.

Consequently

\[
\sum_iW_{ia}T_i^{(2)}
\]

is generically a full-span H3 object even though every summand has a handful of tags. Exact deployment needs one of:

1. contraction at birth against an already pulled-back complete costate;
2. a common transport-invariant eigenbasis/Krylov dictionary;
3. an explicit low-rank projection with a certified endpoint-weighted tail;
4. the full H3 carrier.

Mixing the tag parameters separately is invalid because \(J_e(S)\) is bilinear. This is the precise boundary between a cheap reset-local scalar and an unaffordable future H3 state.

## Exact finite-secant completion of the cubic reset jet

Let

\[
\mathcal R(g)=P_{\le2}\operatorname{ReLU}(g),
\qquad g,h\in\mathcal H_{\le2}.
\]

The first three directional derivatives are

\[
D\mathcal R_g[h]=P_{\le2}[H(g)h],
\]

\[
D^2\mathcal R_g[h,h]=P_{\le2}[\delta(g)h^2],
\]

\[
D^3\mathcal R_g[h,h,h]=P_{\le2}[\delta'(g)h^3].
\]

Hence the entire post-square nonlinear reset correction for a finite AQ direction \(h\) is

\[
\boxed{
\mathcal N_{\ge3}(g;h)
=\mathcal R(g+h)-\mathcal R(g)
-D\mathcal R_g[h]
-\frac12D^2\mathcal R_g[h,h].
}
\]

Taylor's integral identity gives the equivalent cubic-surface representation

\[
\boxed{
\mathcal N_{\ge3}(g;h)
=\frac12\int_0^1(1-t)^2
P_{\le2}[\delta'(g+th)h^3]\,dt.
}
\]

This is not a cubic truncation: it sums every reset word of order at least three along the exact AQ secant. For a Bellman costate \(\lambda\in\mathcal H_{\le2}\),

\[
\begin{aligned}
\langle\lambda,\mathcal N_{\ge3}(g;h)\rangle
={}&\langle\lambda,\mathcal R(g+h)-\mathcal R(g)\rangle\\
&-\langle\lambda,P_{\le2}[H(g)h]\rangle
-\frac12\langle\lambda,P_{\le2}[\delta(g)h^2]\rangle.
\end{aligned}
\]

Thus an exact AQ reset oracle can acquire the complete finite-secant value using endpoint reset outputs plus the already typed linear/square responses; no explicit \(\delta'\) moment, cubic tensor, or numerical density derivative is required.

The reset is stable in the native Gaussian \(L^2\) norm:

\[
\boxed{
\|\mathcal R(g_1)-\mathcal R(g_2)\|_2
\le\|\operatorname{ReLU}(g_1)-\operatorname{ReLU}(g_2)\|_2
\le\|g_1-g_2\|_2.
}
\]

Consequently \(D\mathcal R_g=P_{\le2}M_{H(g)}P_{\le2}\) is a self-adjoint positive contraction wherever the derivative exists. A synchronized AQ trajectory using the exact reset therefore does not amplify a local state perturbation at the reset itself; amplification can only enter through dense transport/frame maps.

In fact the projected reset is firmly nonexpansive on
\(\mathcal H_{\le2}\):

\[
\boxed{
\|\mathcal R(g_1)-\mathcal R(g_2)\|_2^2
\le
\langle \mathcal R(g_1)-\mathcal R(g_2),\,g_1-g_2\rangle.
}
\]

This follows from firm nonexpansiveness of the pointwise positive-part projection and the fact that \(g_1-g_2\in\mathcal H_{\le2}\), so the outer orthogonal projector may be moved across the inner product. It gives a global secant cocoercivity receipt, not merely a local Jacobian bound.

### Ownership and interpretation

The same-site total K31/K22 response remains old D11/Q24. The lawful new object is not \(\mathcal R(g+h)\) as a static feature, but the strict-depth trajectory effect generated when

\[
(I-P_{\rm old})\mathcal N_{\ge3}(g_l;h_l)
\]

is inserted into the next AQ state and transported to later consumers. The direction \(h_l\) must itself be label-free and generated by the deployed synchronized state; a truth-minus-estimated direction is not a provider.

### Graph node: AQ_FINITE_SECANT_POSTSQUARE

- status: solved_exact_local_provider;
- claim: the displayed endpoint-difference formula acquires all reset orders \(\ge3\) in \(\mathcal H_{\le2}\) without an H3/H4 tensor;
- dependencies: exact projective AQ reset, a label-free AQ secant direction, and explicit ownership of the first two reset derivatives;
- cost premise: constant-factor differentiation/evaluation of the AQ reset, plus the existing AQ state transport; no physical \(CM\) product for the local quotient action;
- novelty gate: strict positive depth and complete D11/Q24/depth-arena Schur residualization;
- value gate: the response of this finite-secant state must meet the master quartic/variance deficit after its beyond-AQ residual bound.

## Positive-depth diagonal polarization completes every external quartic equality sector

The generic fixed-site K22 contraction is expensive because its endpoint covector is arbitrary. The physical next-layer endpoint has much more structure. Carry the half-quadratic AQ coefficient itself:

\[
X_i=u_i^{\mathsf T}z+Q_{A_i}(z),
\qquad
Q_A(z)=\frac12(z^{\mathsf T}Az-\operatorname{tr}A).
\]

For one downstream column \(w\), dense mixing gives

\[
u(w)=\sum_iw_i u_i,\qquad
A(w)=\sum_iw_i A_i.
\]

The exact diagonal affine--quadratic fourth cumulant is

\[
\boxed{
\Phi_4(u,A)=3\operatorname{tr}A^4+12u^{\mathsf T}A^2u.
}
\]

Evaluating \(\Phi_4(u(w),A(w))\) implicitly polarizes the complete external quartic tensor along the physical dense column \(w\).

### Pure-AA polarization

The symmetric four-linear pure-quadratic kernel is

\[
\boxed{
\begin{aligned}
\kappa_Q(A_1,A_2,A_3,A_4)
={}&\operatorname{tr}(A_1A_2A_3A_4)\\
&+\operatorname{tr}(A_1A_2A_4A_3)\\
&+\operatorname{tr}(A_1A_3A_2A_4).
\end{aligned}}
\]

Its equality sectors are

\[
\kappa_4(i)=3\operatorname{tr}A_i^4,
\]

\[
\kappa_{31}(i,j)=3\operatorname{tr}(A_i^3A_j),
\]

\[
\kappa_{22}(i,j)
=2\operatorname{tr}(A_i^2A_j^2)
+\operatorname{tr}(A_iA_jA_iA_j),
\]

\[
\boxed{
\kappa_{211}(i;j,k)
=2\operatorname{tr}(A_i^2A_jA_k)
+\operatorname{tr}(A_iA_jA_iA_k),
}
\]

and, for four distinct labels,

\[
\boxed{
\begin{aligned}
\kappa_{1111}(i,j,k,l)
={}&\operatorname{tr}(A_iA_jA_kA_l)\\
&+\operatorname{tr}(A_iA_jA_lA_k)\\
&+\operatorname{tr}(A_iA_kA_jA_l).
\end{aligned}}
\]

Consequently

\[
3\operatorname{tr}\left(\sum_iw_iA_i\right)^4
\]

has the exact multinomial expansion

\[
\begin{aligned}
={}&\sum_iw_i^4\kappa_4(i)
+4\sum_{i\ne j}w_i^3w_j\kappa_{31}(i,j)\\
&+6\sum_{i<j}w_i^2w_j^2\kappa_{22}(i,j)\\
&+12\sum_i\sum_{\substack{j<k\\j,k\ne i}}
w_i^2w_jw_k\kappa_{211}(i;j,k)\\
&+24\sum_{i<j<k<l}
w_iw_jw_kw_l\kappa_{1111}(i,j,k,l).
\end{aligned}
\]

Thus the missing \(211\) and \(1111\) sectors are not separately carried: they are generated with the correct collision coefficients when the coadapted \(A_i\) state is mixed before the next diagonal read.

### Mixed H1--H2 polarization

The complete affine--quadratic four-linear kernel is

\[
\boxed{
\begin{aligned}
\kappa_4(X_1,X_2,X_3,X_4)
={}&\kappa_Q(A_1,A_2,A_3,A_4)\\
&+\sum_{\{p,q\}\subset\{1,2,3,4\}}
u_p^{\mathsf T}
(A_rA_s+A_sA_r)u_q,
\end{aligned}}
\]

where \(\{r,s\}\) is the complement of \(\{p,q\}\). Setting all four arguments equal gives \(3\operatorname{tr}A^4+12u^{\mathsf T}A^2u\). This is the exact fixed pairing between the two-linear/two-quadratic diagrams and the pure-AA diagrams; it is not a fitted coefficient.

### Complexity and topology boundary

Let

\[
d=r+\frac{r(r+1)}2
\]

be the packed \((u,A)\) width. One dense mix of all AQ coefficients costs \(d/n\) width-GEMM equivalents. At \(r=32,n=1024\),

\[
d=560,\qquad d/n=0.546875.
\]

Across fourteen layers this is \(7.65625\) equivalents. The local diagonal potential needs one \(r\times r\) square per unit:

\[
\operatorname{tr}A^4=\|A^2\|_F^2,\qquad
u^{\mathsf T}A^2u=\|Au\|^2,
\]

costing \(nr^3/n^3=r^3/n^2=0.03125\) equivalents per layer at \(r=32\), plus lower-order reductions. Reverse gradients may require a second local \(r^3\) product, still only \(0.0625\) per layer.

This is the topology-specific escape from the generic external \(n^4\) barrier:

- arbitrary fixed-site K22 costate: still generic crossed-word no-go;
- physical dense mix followed by a diagonal AQ potential: exact \(O(n^2r^2+nr^3)\);
- no external \(211\) or \(1111\) state is formed.

### Ownership boundary

At zero depth, the resulting total diagonal quartic is only a re-expression of the incumbent same-site Q24 content. Novelty requires that the \(A_i\) coefficients be updated by a new synchronized/projective AQ reset and then mixed into a later layer. The lawful response is the strict-depth difference between this coadapted polarization and the old D11/Q24 trajectory, followed by complete arena residualization.

### Graph node: AQ_DIAGONAL_POLARIZATION_COMPLETION

- status: solved_exact_mechanism;
- claim: a transported \((u_i,A_i)\) state plus the diagonal potential \(\Phi_4\) spans every external affine--quadratic quartic equality sector with exact coefficients;
- state/cost: packed width \(d=r+r(r+1)/2\), one dense coefficient mix per layer, one or two local latent matrix products;
- dependencies: synchronized AQ reset and strict-depth ownership;
- no-go boundary: does not solve beyond-AQ H3/H4 residuals or a generic fixed-site K22 Bellman covector;
- campaign relevance: this is the label-free carrier mechanism by which a D14 trajectory can generate the trajectory-dark \(211/1111\) fiber without an external tensor.

## Complete AQ cumulant-potential stack through order four

For a half-AQ state \(X=(u,A)\), define the homogeneous diagonal potentials

\[
\boxed{
\Phi_2(u,A)=\|u\|^2+\frac12\operatorname{tr}A^2,
}
\]

\[
\boxed{
\Phi_3(u,A)=\operatorname{tr}A^3+3u^{\mathsf T}Au,
}
\]

\[
\boxed{
\Phi_4(u,A)=3\operatorname{tr}A^4+12u^{\mathsf T}A^2u.
}
\]

These are respectively the second, third, and fourth cumulants of

\[
u^{\mathsf T}z+\frac12(z^{\mathsf T}Az-\operatorname{tr}A).
\]

Because cumulants are symmetric multilinear, every joint AQ cumulant through order four is obtained by polarization. If \(S,V,W,Z,T\) denote AQ coefficient pairs, then

\[
D\Phi_4(S)[V]=4\kappa_4(S,S,S,V),
\]

\[
\boxed{
D^2\Phi_4(S)[V,W]
=12\kappa_4(S,S,V,W),
}
\]

\[
D^3\Phi_4(S)[V,W,Z]
=24\kappa_4(S,V,W,Z),
\]

\[
D^4\Phi_4(0)[V,W,Z,T]
=24\kappa_4(V,W,Z,T).
\]

Analogously,

\[
D\Phi_3(S)[V]=3\kappa_3(S,S,V),
\qquad
D^2\Phi_3(S)[V,W]=6\kappa_3(S,V,W),
\]

and the Hessian of \(\Phi_2\) is twice the covariance bilinear form.

The explicit cubic three-linear kernel is

\[
\boxed{
\begin{aligned}
\kappa_3(X_1,X_2,X_3)
={}&\frac12\operatorname{tr}(A_1A_2A_3+A_1A_3A_2)\\
&+u_1^{\mathsf T}A_3u_2
+u_1^{\mathsf T}A_2u_3
+u_2^{\mathsf T}A_1u_3.
\end{aligned}}
\]

It gives \(\operatorname{tr}A^3+3u^{\mathsf T}Au\) on the diagonal and automatically completes the external \(21\) and \(111\) equality sectors after dense mixing, just as \(\Phi_4\) completes \(31,22,211,1111\).

Therefore the packed \((u,A)\) trajectory plus

\[
\{\Phi_2,\Phi_3,\Phi_4\}
\]

is a complete label-free scalar generator for every affine--quadratic covariance, cubic, and quartic action. K31 is a first JVP of \(\Phi_4\); K22/q211 is a second JVP; all-distinct q1111 is the fourth polarization. The stack never materializes a third- or fourth-order external tensor.

This sharply isolates the remaining model error:

\[
\text{true exact-}q
=\text{AQ potential stack}
+\text{beyond-AQ chaos/state residual}.
\]

Within the AQ manifold there is no missing equality sector and no free diagram coefficient to fit. Any failure of the synchronized carrier must be attributed to reset projection error, frame/secant error, ownership subtraction, or deployment arithmetic—not to absent \(q\le4\) AQ diagrams.

### Graph node: AQ_CUMULANT_POTENTIAL_STACK

- status: solved_exact_within_AQ;
- provider: transported packed \((u,A)\) coefficients;
- reads: \(\Phi_2,\Phi_3,\Phi_4\) and their scalar JVP/HVP actions;
- coverage: all AQ joint cumulants and equality sectors through order four;
- strict limitation: no claim about latent H3/H4 or non-AQ residual law;
- decisive next gate: bound the response-normalized beyond-AQ residual tightly enough to decide the required variance-channel fidelity.

## Exact Duhamel identity for the beyond-AQ trajectory residual

Let \(\mathcal H=L^2(\gamma)\), let \(P=P_{\le2}\), and compare the true unit-function trajectory

\[
f_{l+1}=\operatorname{ReLU}(W_l^{\mathsf T}f_l)
\]

with the synchronized AQ Galerkin trajectory

\[
a_{l+1}=P\operatorname{ReLU}(W_l^{\mathsf T}a_l).
\]

At the approximate preactivation

\[
g_l=W_l^{\mathsf T}a_l,
\]

define the exactly computable local projection residual

\[
\boxed{
r_{l+1}
=(I-P)\operatorname{ReLU}(g_l).
}
\]

For \(e_l=f_l-a_l\), define the coordinatewise secant multiplier

\[
d_{l,i}(z)=
\begin{cases}
\dfrac{\operatorname{ReLU}(g_{l,i}+\Delta g_{l,i})
-\operatorname{ReLU}(g_{l,i})}{\Delta g_{l,i}},
&\Delta g_{l,i}\ne0,\\[2mm]
H(g_{l,i}),&\Delta g_{l,i}=0,
\end{cases}
\]

where \(\Delta g_l=W_l^{\mathsf T}e_l\). Then \(0\le d_{l,i}\le1\) pointwise and

\[
\boxed{
e_{l+1}=D_l^{\rm sec}W_l^{\mathsf T}e_l+r_{l+1}.
}
\]

This is exact, not a linearization. Iteration yields

\[
\boxed{
e_L=\sum_{j<L}
\left(D_{L-1}^{\rm sec}W_{L-1}^{\mathsf T}\right)
\cdots
\left(D_{j+1}^{\rm sec}W_{j+1}^{\mathsf T}\right)
r_{j+1},
}
\]

with the appropriate initialization term if \(e_0\ne0\).

For a scalar endpoint covector \(\lambda_L\), reverse the exact secant maps:

\[
\lambda_l
=W_lD_l^{\rm sec}\lambda_{l+1}.
\]

The complete beyond-AQ response error is then

\[
\boxed{
\langle\lambda_L,e_L\rangle
=\sum_{j<L}\langle\lambda_{j+1},r_{j+1}\rangle.
}
\]

This identity types every omitted AQ term: there is no additional equality-sector defect inside the potential stack; all remaining error is a local \(H_{\ge3}\) reset residual transported by exact secant costates.

### Label-free residual energy

For one AQ gate \(g\), if

\[
\mathcal R(g)=m_++u_+^{\mathsf T}z
+\frac12(z^{\mathsf T}A_+z-\operatorname{tr}A_+),
\]

then

\[
\boxed{
\|(I-P)\operatorname{ReLU}(g)\|_2^2
=\mathbb E[\operatorname{ReLU}(g)^2]
-m_+^2-\|u_+\|^2-\frac12\|A_+\|_F^2.
}
\]

Thus every source norm in the Duhamel sum is available from the same exact AQ reset oracle.

Without observing the true secant gates, a safe target-free bound uses

\[
\|D_l^{\rm sec}\|\le1
\]

and gives

\[
\boxed{
\|e_L\|_2
\le
\sum_{j<L}
\left(\prod_{k=j+1}^{L-1}\|W_k\|_{\rm op}\right)
\|r_{j+1}\|_2.
}
\]

A sharper Bellman bound replaces the spectral products by any certified norms of the pulled-back secant costates:

\[
|\langle\lambda_L,e_L\rangle|
\le\sum_{j<L}\|\lambda_{j+1}\|_2\|r_{j+1}\|_2.
\]

### Raw-target fidelity gate

In the optimistic exact-other-channel and unit coefficient-fidelity branch, the master inequality permits normalized variance-response residual at most

\[
\boxed{
r_v\le
\sqrt{\frac{0.07453}{1.7316}}
\simeq0.20746.
}
\]

More generally, with coefficient fidelity \(\phi\) and other-channel remainder \(R_{\rm other}\),

\[
\boxed{
r_v^2
\le
\frac{(0.07453-R_{\rm other})/1.7316-(1-\phi)}{\phi}.
}
\]

The right-hand side must be positive. Therefore a response-normalized Duhamel upper bound below this frontier is a symbolic sufficient certificate that AQ truncation is accurate enough for the raw-target lane. A loose upper bound above it is inconclusive, not a no-go; a valid lower bound above it closes the lane.

### Graph node: AQ_BEYOND_RESIDUAL_DUHAMEL

- status: solved_exact_identity/open_numeric_radius;
- sources: exact local AQ projection residuals \(r_l\);
- transport: exact ReLU secant costates;
- label-free safe bound: weight-operator products times reset residual energies;
- goal gate: normalized variance residual below the displayed \(0.20746\) frontier in the optimistic branch, tightened by \(\phi<1\) and other-channel loss;
- minimal missing observable for a sharp certificate: task-weighted secant costate norms or their direct residual pairings.

## Two-crossing theorem and an exact spectral leakage certificate

Put \(Q=I-P_{\le2}\). At a frozen AQ ReLU gate let

\[
M_g=M_{H(g)}
\]

be multiplication by the binary gate. Relative to
\(\mathcal H=P\mathcal H\oplus Q\mathcal H\),

\[
M_g=
\begin{pmatrix}
T_g & B_g\\
C_g & D_g
\end{pmatrix},
\qquad
T_g=PM_gP,\quad
C_g=QM_gP,\quad B_g=C_g^*.
\]

The synchronized AQ tangent keeps \(T_g\). Any omitted path that starts in an AQ state and is later read by an AQ endpoint must leave through \(C_g\) and return through \(B_k\). Hence every AQ-to-AQ correction outside the retained trajectory contains at least two cross-subspace factors. A single discarded residual has zero immediate AQ value:

\[
\langle\lambda_P,r_Q\rangle=0.
\]

This is the exact reason strict positive depth is required.

Because \(M_g^2=M_g\),

\[
\begin{aligned}
C_g^*C_g
&=PM_gQM_gP\\
&=PM_g^2P-PM_gPM_gP\\
&=T_g-T_g^2.
\end{aligned}
\]

Therefore the exact one-gate leakage norm is

\[
\boxed{
c_g^2:=\|C_g\|_{\rm op}^2
=\|T_g-T_g^2\|_{\rm op}
=\max_{\lambda\in\operatorname{spec}T_g}
\lambda(1-\lambda)
\le\frac14.
}
\]

No H3 basis is required: \(T_g\) is precisely the finite AQ tangent matrix already generated by the reset scalar potential. The AQ subspace is invariant iff every eigenvalue of \(T_g\) is \(0\) or \(1\).

For a specific AQ direction \(h\), the exact discarded tangent energy is sharper than the operator maximum:

\[
\boxed{
\|QH(g)h\|_2^2
=\langle h,(T_g-T_g^2)h\rangle.
}
\]

Indeed \(\|H(g)h\|^2=\langle h,T_gh\rangle\) and
\(\|PH(g)h\|^2=\|T_gh\|^2\). The Hilbert--Schmidt aggregate is similarly

\[
\|C_g\|_{\rm HS}^2
=\operatorname{tr}(T_g-T_g^2)
=\sum_j\lambda_j(1-\lambda_j).
\]

Thus a compiled Bellman direction can use the quadratic form rather than the worst eigenvalue, and a portfolio of directions can use the trace or its restricted Gram.

Positive homogeneity makes the local reset residual itself a special case. Since

\[
\operatorname{ReLU}(g)=H(g)g,
\qquad g\in P\mathcal H,
\]

\[
\boxed{
\mathcal R(g)=T_g g,\qquad
r(g)=QH(g)g=C_g g.
}
\]

Therefore

\[
\boxed{
\|r(g)\|_2^2
=\langle g,(T_g-T_g^2)g\rangle
\le\frac14\|g\|_2^2.
}
\]

Moreover the residual map

\[
\mathfrak r:g\mapsto Q\operatorname{ReLU}(g)
\]

has derivative \(D\mathfrak r_g=QH(g)P=C_g\) almost everywhere. Since
\(\|C_g\|\le1/2\) at every regular gate,

\[
\boxed{
\|\mathfrak r(g_1)-\mathfrak r(g_2)\|_2
\le\frac12\|g_1-g_2\|_2.
}
\]

Thus the \(H_{\ge3}\) reset residual is universally one-half-Lipschitz with respect to AQ state error. The universal \(1/2\) is only a coarse ceiling; the directional quadratic form in \(T_g-T_g^2\) is the fail-fast quantity.

### Multi-layer path bound

Dense unit mixing commutes with \(P/Q\), because it acts only on unit indices. Let

\[
\kappa_l=\|W_l\|_{\rm op},\qquad c_l=\|Q M_{g_l}P\|_{\rm op}.
\]

Majorizing the diagonal \(P\to P\) and \(Q\to Q\) gate blocks by one, every omitted \(P\to P\) path is bounded by an even product of crossing norms. Consequently

\[
\boxed{
\left\|P\mathcal T_{L:0}P
-\mathcal T_{L:0}^{AQ}\right\|_{\rm op}
\le
\left(\prod_{l=0}^{L-1}\kappa_l\right)
\left[
\frac12\left\{\prod_l(1+c_l)+\prod_l(1-c_l)\right\}-1
\right].
}
\]

The bracket is the sum of all even elementary products of the \(c_l\) of degree at least two. For small leakage,

\[
\frac12\{\prod(1+c_l)+\prod(1-c_l)\}-1
=\sum_{j<k}c_jc_k+O(c^4).
\]

This is a worst-case target-free bound; exact Bellman geometry can be much smaller.

For a nonbinary exact secant multiplier \(0\le d\le1\), the same construction gives

\[
C^*C=PD^2P-(PDP)^2
\preceq PDP-(PDP)^2,
\]

so the binary-gate spectral expression remains a safe envelope when the retained secant operator is used.

### Frozen-tangent fail-fast scale

If \(\prod\kappa_l\) is normalized to one and the ten active layers have a common bound \(c_l\le c\), the optimistic variance-fidelity frontier requires roughly

\[
\binom{10}{2}c^2\lesssim0.20746,
\qquad c\lesssim0.068
\]

at leading order. For fourteen layers the analogous scale is about \(c\lesssim0.048\). These are sufficient worst-case scales, not necessary conditions; failure of the bound is inconclusive.

### Graph node: AQ_TANGENT_TWO_CROSSING_CERTIFICATE

- status: solved_exact_frozen_tangent;
- local observable: spectrum of \(T_g=P H(g)P\);
- leakage: \(c_g^2=\|T_g-T_g^2\|\);
- global sufficient bound: the even-product formula above;
- interpretation: all AQ-invisible value requires at least an exit and re-entry, so first-order source energy alone overstates endpoint error;
- limitations: weight-product majorization may be loose, and nonlinear finite-secant/frame error remains a separate term.

## Scalar-potential realization of the exact projective AQ reset

For

\[
g(z)=m+u^{\mathsf T}z
+\frac12(z^{\mathsf T}Az-\operatorname{tr}A),
\]

define the single scalar potential

\[
\boxed{
F(m,u,A)=\mathbb E[\operatorname{ReLU}(g(z))].
}
\]

Write \(F_m=\partial_mF\), \(F_u=\nabla_uF\), \(F_{mm}=\partial_m^2F\), \(F_{mu}=\partial_m\nabla_uF\), and \(F_{uu}=\nabla_u^2F\). The complete \(H_{\le2}\) reset coefficients follow exactly from Gaussian integration by parts:

\[
\boxed{
m_+=F,
}
\]

\[
\boxed{
u_+=uF_m+AF_u,
}
\]

and

\[
\boxed{
\begin{aligned}
A_+={}&AF_m
+uu^{\mathsf T}F_{mm}\\
&+uF_{mu}^{\mathsf T}A
+AF_{mu}u^{\mathsf T}
+AF_{uu}A.
\end{aligned}}
\]

Proof: with \(q=\nabla_zg=u+Az\),

\[
\mathbb E[z\operatorname{ReLU}(g)]
=\mathbb E[H(g)q]
=uF_m+AF_u,
\]

while

\[
\mathbb E[(zz^{\mathsf T}-I)\operatorname{ReLU}(g)]
=\mathbb E[H(g)A+\delta(g)qq^{\mathsf T}],
\]

and

\[
\begin{aligned}
\mathbb E[\delta(g)qq^{\mathsf T}]
={}&uu^{\mathsf T}F_{mm}
+uF_{mu}^{\mathsf T}A\\
&+AF_{mu}u^{\mathsf T}
+AF_{uu}A.
\end{aligned}
\]

Thus one scalar law and its derivatives generate the whole projective reset. No multivariate surface tensor is needed.

### One-dimensional characteristic integral

The characteristic function of \(g\) is

\[
\chi_g(t)
=e^{it(m-\operatorname{tr}A/2)}
\det(I-itA)^{-1/2}
\exp\!\left[
-\frac{t^2}{2}u^{\mathsf T}(I-itA)^{-1}u
\right].
\]

Since \(\mathbb E g=m\) and

\[
\mathbb E|g|
=\frac2\pi\int_0^\infty
\frac{1-\operatorname{Re}\chi_g(t)}{t^2}\,dt,
\]

the reset potential has the exact one-dimensional representation

\[
\boxed{
F(m,u,A)
=\frac m2+\frac1\pi\int_0^\infty
\frac{1-\operatorname{Re}\chi_g(t)}{t^2}\,dt.
}
\]

Parameter differentiation under this integral, or equivalent resolvent differentiation, supplies every derivative in the reset formulas. The same differentiated scalar oracle supplies the tangent, Hessian, cubic JVP, and finite-secant completion described above.

For \(A=0\), this reduces to the ordinary Gaussian ReLU law and gives

\[
u_+=u\,\Phi(m/\|u\|),
\qquad
A_+=\frac{\varphi(m/\|u\|)}{\|u\|}uu^{\mathsf T},
\]

which checks the normalization.

### Exact residual energy from the same reset coefficients

No additional moment integral is required for
\(\mathbb E[\operatorname{ReLU}(g)^2]\). Pointwise,

\[
g\,\operatorname{ReLU}(g)=\operatorname{ReLU}(g)^2.
\]

Because \(g\in\mathcal H_{\le2}\), it is orthogonal to the discarded part of \(\operatorname{ReLU}(g)\). Therefore

\[
\boxed{
\mathbb E[\operatorname{ReLU}(g)^2]
=\langle g,\mathcal R(g)\rangle
=m\,m_+ +u^{\mathsf T}u_+
+\frac12\operatorname{tr}(AA_+).
}
\]

Subtracting the retained energy gives the exact local beyond-AQ norm

\[
\boxed{
\begin{aligned}
\rho_{\ge3}^2
={}&m\,m_+ +u^{\mathsf T}u_+
+\frac12\operatorname{tr}(AA_+)\\
&-m_+^2-\|u_+\|^2-\frac12\|A_+\|_F^2.
\end{aligned}}
\]

This expression is nonnegative by orthogonal projection and is an immediate normalization/implementation check. It supplies every source norm required by AQ_BEYOND_RESIDUAL_DUHAMEL from \(F\) and its derivatives alone.

### Convex master potential for the reset

Define

\[
\boxed{
J(m,u,A)
=\frac12\mathbb E[\operatorname{ReLU}(g)^2].
}
\]

Equip AQ coefficient space with its Gaussian Hermite metric

\[
\langle(m,u,A),(\tilde m,\tilde u,\tilde A)\rangle_{AQ}
=m\tilde m+u^{\mathsf T}\tilde u
+\frac12\operatorname{tr}(A\tilde A).
\]

For an arbitrary AQ perturbation \(h\),

\[
DJ_g[h]
=\mathbb E[\operatorname{ReLU}(g)h]
=\langle\mathcal R(g),h\rangle_{AQ}.
\]

Therefore

\[
\boxed{
\mathcal R(g)=\nabla_{AQ}J(g).
}
\]

In ordinary coordinates this says

\[
\partial_mJ=m_+,\qquad
\nabla_uJ=u_+,\qquad
\nabla_AJ=\frac12A_+.
\]

Moreover,

\[
\boxed{
\nabla_{AQ}^2J(g)
=P_{\le2}M_{H(g)}P_{\le2}=T_g,
\qquad 0\preceq T_g\preceq I.
}
\]

Thus \(J\) is convex and one-smooth in the AQ metric, and firm nonexpansiveness of the reset is the standard gradient-cocoercivity theorem. Positive homogeneity gives Euler's identity

\[
2J(g)=\langle g,\mathcal R(g)\rangle_{AQ},
\]

which is exactly the reset-energy formula above.

Finally,

\[
\partial_mJ=F,\qquad
J(m,u,A)=\int_{-\infty}^{m}F(s,u,A)\,ds.
\]

Hence the one-dimensional characteristic oracle for \(F\), followed by a mean antiderivative or a direct stop-loss-squared formula, yields a single scalar master potential whose AQ gradient is the full reset. Higher AQ derivatives of \(J\) generate the tangent, square response, and cubic jet with automatically consistent normalization.

### Graph node: AQ_RESET_CONVEX_MASTER_POTENTIAL

- status: solved_exact_structure;
- scalar: \(J=\frac12E(g_+)^2\);
- reset: Hermite-metric gradient of \(J\);
- tangent: Hessian \(T_g\), PSD contraction;
- diagnostics: Euler energy identity and nonnegative discarded norm;
- consequence: forward reset and reverse adjoint can share one scalar differentiated oracle, eliminating independent normalization choices.

## Exact affine-Gaussian fidelity benchmark

The local retained-energy fraction can be evaluated in closed form when \(A=0\). Write

\[
g=m+u^{\mathsf T}z,\qquad
\sigma=\|u\|,\qquad
\alpha=m/\sigma,
\]

and let \(\varphi,\Phi\) denote the standard Gaussian density and CDF. The reset coefficients are

\[
m_+=\sigma[\varphi(\alpha)+\alpha\Phi(\alpha)],
\]

\[
u_+=u\,\Phi(\alpha),
\]

\[
A_+=\frac{\varphi(\alpha)}{\sigma}uu^{\mathsf T}.
\]

The total and retained output energies are

\[
\boxed{
E_{\rm tot}(\alpha)
=\sigma^2[(1+\alpha^2)\Phi(\alpha)+\alpha\varphi(\alpha)],
}
\]

\[
\boxed{
E_{\rm AQ}(\alpha)
=\sigma^2\left[
(\varphi+\alpha\Phi)^2+\Phi^2+\frac12\varphi^2
\right].
}
\]

Hence the exact local AQ energy fidelity is

\[
\boxed{
q_{\rm AQ}^2(\alpha)
=
\frac{
(\varphi(\alpha)+\alpha\Phi(\alpha))^2
+\Phi(\alpha)^2+\frac12\varphi(\alpha)^2
}{
(1+\alpha^2)\Phi(\alpha)+\alpha\varphi(\alpha)
}.
}
\]

At the centered gate,

\[
\boxed{
q_{\rm AQ}^2(0)
=\frac{\pi+3}{2\pi}
\simeq0.977465,
}
\]

and

\[
\frac{\rho_{\ge3}^2}{E_{\rm tot}}
=\frac{\pi-3}{2\pi}
\simeq0.022535.
\]

Thus a centered affine Gaussian ReLU locally clears the optimistic campaign fidelity bar \(0.95696\), but by only about \(2.05\) percentage points.

Solving

\[
q_{\rm AQ}^2(\alpha)=0.95696
\]

on the negative-mean branch gives approximately

\[
\boxed{\alpha_{\rm crit}\simeq-0.33.}
\]

For example, at \(\alpha=-1/2\) the retained fraction is only about \(0.9365\), already below the idealized bar. Positive means rapidly improve the fraction, while rare negative-tail gates are poorly represented by only two Hermite degrees.

This comparison is a local energy diagnostic, not automatically the end-to-end coefficient fidelity \(\phi\): Bellman weights, dense transport, two-crossing cancellation, and other channels matter. It is nevertheless a sharp analytic triage:

- if the task-weighted gate population is concentrated near \(\alpha\ge-0.33\) with modest quadratic deformation, D14 has enough local Gaussian fidelity to remain plausible;
- if value-carrying layers are dominated by substantially negative standardized means, an \(H_{\le2}\)-only trajectory is unlikely to clear the goal without an explicit H3/rest companion.

For general AQ \(A\), the exact analogous diagnostic is already available from the convex master potential:

\[
\boxed{
q_{\rm AQ}^2(m,u,A)
=\frac{\|\nabla_{AQ}J\|_{AQ}^2}{2J}
=1-\frac{\rho_{\ge3}^2}{2J}.
}
\]

### Graph node: AQ_LOCAL_FIDELITY_TRIAGE

- status: solved_exact_local;
- Gaussian centered benchmark: \(0.977465\);
- idealized threshold crossing: \(\alpha\simeq-0.33\);
- general AQ read: \(\|\nabla J\|^2/(2J)\);
- limitation: local source energy is not an end-to-end target-angle certificate;
- fail-fast use: identify whether task-weighted gates require an H3/rest companion before building the full trajectory.

## First-variation robustness of the centered Gaussian fidelity

Let the base gate be

\[
g_0=\sigma x,\qquad x=e^{\mathsf T}z,
\]

and perturb it by a centered half-quadratic direction

\[
h=Q_M(z)=\frac12(z^{\mathsf T}Mz-\operatorname{tr}M).
\]

Write

\[
R_\varepsilon=\mathcal R(g_0+\varepsilon h),\qquad
N(\varepsilon)=\|R_\varepsilon\|_2^2,\qquad
D(\varepsilon)=\langle g_0+\varepsilon h,R_\varepsilon\rangle.
\]

Since \(D\mathcal R_{g_0}=T_0=P H(x)P\) is self-adjoint and
\(R_0=T_0g_0\),

\[
N'(0)=2\langle T_0R_0,h\rangle,
\qquad
D'(0)=2\langle R_0,h\rangle.
\]

Both \(R_0\) and \(T_0R_0\) are functions only of \(x\), with Hermite degree at most two after projection. Conditional expectation gives

\[
\mathbb E[h\mid x]
=\frac12(e^{\mathsf T}Me)(x^2-1).
\]

Therefore

\[
\boxed{
e^{\mathsf T}Me=0
\quad\Longrightarrow\quad
N'(0)=D'(0)=0.
}
\]

In particular,

\[
\boxed{
\left.\frac d{d\varepsilon}
\frac{\|R_\varepsilon\|^2}
{\langle g_0+\varepsilon h,R_\varepsilon\rangle}
\right|_{\varepsilon=0}=0
\qquad\text{if }e^{\mathsf T}Me=0.
}
\]

Thus every noncollective quadratic perturbation orthogonal to the gate-axis scalar changes local AQ energy fidelity only at second order. The only first-order deformation is the collective scalar
\(m=e^{\mathsf T}Me\), precisely the direction represented by the order-one \(e^{\otimes3}\), \(\operatorname{Sym}(e^2\otimes Me)\), and Jordan sibling terms.

This independently explains why the first genuinely identity-specific boundary correction appears at \(M^2\). It also means the centered \(0.977465\) fidelity margin is not immediately destroyed by a small transverse/noncommuting AQ deformation, although a quantitative second-order bound is still required.

### Graph node: AQ_LOCAL_FIDELITY_FIRST_VARIATION

- status: solved_exact;
- base: centered affine Gaussian gate;
- result: all \(e^{\mathsf T}Me=0\) quadratic directions have zero first fidelity variation;
- ownership: the surviving first-order collective direction is expected in old H31/Q24 and must be audited;
- next gate: bound the second-order identity-specific degradation using the audited \(M^2\) boundary basis or the exact scalar potential.

## Exact second-order fidelity loss for a noncollective quadratic perturbation

The preceding second-order gate is also analytic at the centered Gaussian base. Decompose, with \(P_e=I-ee^{\mathsf T}\),

\[
e^{\mathsf T}Me=0,\qquad
w=P_eMe,\qquad
B=P_eMP_e.
\]

Writing \(z=xe+y\), \(y\perp e\), the half-quadratic perturbation is

\[
h=Q_M(z)
=x\,w^{\mathsf T}y
+\frac12(y^{\mathsf T}By-\operatorname{tr}B).
\]

Put

\[
b^2=\frac12\operatorname{tr}B^2,\qquad
c=\varphi(0)^2=\frac1{2\pi}.
\]

At \(g_0=\sigma x\), the first reset variation is

\[
D\mathcal R_{g_0}[h]
=(\varphi(0)+x/2)\,w^{\mathsf T}y
+\frac14(y^{\mathsf T}By-\operatorname{tr}B).
\]

The cross and transverse-quadratic pieces are orthogonal. Using

\[
D^2\mathcal R_{g_0}[h,h]
=P_{\le2}[\delta(\sigma x)h^2]
\]

and \(g_0\delta(g_0)=0\), the total and retained energies have expansions

\[
\boxed{
\frac{D(\varepsilon)}{\sigma^2}
=\frac12
+\frac{\varepsilon^2}{2\sigma^2}
(\|w\|^2+b^2)
+O(\varepsilon^3),
}
\]

\[
\boxed{
\begin{aligned}
\frac{N(\varepsilon)}{\sigma^2}
={}&\left(\frac14+\frac{3}{4\pi}\right)\\
&+\frac{\varepsilon^2}{\sigma^2}
\left[
\left(\frac14+c\right)\|w\|^2
+\left(\frac14+\frac c2\right)b^2
\right]
+O(\varepsilon^3).
\end{aligned}}
\]

Dividing gives the exact second-order local fidelity law

\[
\boxed{
q_{\rm AQ}^2(\varepsilon)
=\frac{\pi+3}{2\pi}
-\frac{\varepsilon^2}{2\pi\sigma^2}\|w\|^2
-\frac{\varepsilon^2}{\pi\sigma^2}b^2
+O(\varepsilon^3).
}
\]

Since \(b^2=\operatorname{tr}B^2/2\),

\[
\boxed{
q_{\rm AQ}^2(\varepsilon)
=\frac{\pi+3}{2\pi}
-\frac{\varepsilon^2}
{2\pi\sigma^2}
\left(\|w\|^2+\operatorname{tr}B^2\right)
+O(\varepsilon^3).
}
\]

Ignoring the explicitly higher-order remainder, clearing the idealized \(0.95696\) bar requires

\[
\boxed{
\frac{\varepsilon^2}
{2\sigma^2}
\left(\|w\|^2+\operatorname{tr}B^2\right)
\lesssim
\pi(0.977465-0.95696)
\simeq0.0644.
}
\]

Equivalently,

\[
\frac{|\varepsilon|}
{\sigma}\sqrt{\|w\|^2+\operatorname{tr}B^2}
\lesssim0.359
\]

at second order. This is a concrete magnitude gate for the identity-specific \(M^2\) regime. It is not a finite-\(M\) theorem until the \(O(\varepsilon^3)\) remainder is bounded by the exact scalar potential or characteristic resolvent.

### Graph node: AQ_LOCAL_FIDELITY_SECOND_ORDER

- status: solved_exact_coefficient;
- degradation: \(\varepsilon^2(\|w\|^2+\operatorname{tr}B^2)/(2\pi\sigma^2)\);
- optimistic scale: weighted transverse size \(\lesssim0.359\);
- dependency: centered affine base and \(e^{\mathsf T}Me=0\);
- remaining gate: rigorous finite-deformation remainder and task-weighted aggregation.

## Rigorous finite-deformation fidelity radius from global Lipschitz geometry

A conservative finite-\(\varepsilon\) certificate needs no Taylor remainder. At the centered affine base \(g_0=\sigma x\),

\[
\|r(g_0)\|
=\sigma a_0,
\qquad
a_0=\sqrt{\frac{\pi-3}{4\pi}},
\]

and

\[
\|\operatorname{ReLU}(g_0)\|=\frac{\sigma}{\sqrt2}.
\]

For an arbitrary AQ perturbation \(\Delta g\), put

\[
t=\|\Delta g\|/\sigma.
\]

The one-half-Lipschitz residual theorem and unit-Lipschitz ReLU theorem give

\[
\|r(g_0+\Delta g)\|
\le\sigma(a_0+t/2),
\]

\[
\|\operatorname{ReLU}(g_0+\Delta g)\|
\ge\sigma(1/\sqrt2-t).
\]

Therefore, for \(t<1/\sqrt2\),

\[
\boxed{
q_{\rm AQ}^2(g_0+\Delta g)
\ge
1-
\left(
\frac{a_0+t/2}{1/\sqrt2-t}
\right)^2.
}
\]

To guarantee \(q_{\rm AQ}^2\ge q_*^2\), it is sufficient that

\[
\boxed{
t\le
\frac{\sqrt{1-q_*^2}/\sqrt2-a_0}
{1/2+\sqrt{1-q_*^2}}.
}
\]

At \(q_*^2=0.95696\),

\[
\boxed{t\lesssim0.0573.}
\]

Thus every AQ perturbation within about \(5.7\%\) of the centered affine input norm is rigorously inside the optimistic local fidelity bar, regardless of its matrix orientation. This safe ball is deliberately conservative. For a noncollective quadratic perturbation, the exact second-order calculation predicts the much wider weighted scale \(0.359\); the annulus between the two must be decided by the exact scalar potential/characteristic oracle rather than by extrapolating the series.

### Graph node: AQ_LOCAL_FIDELITY_SAFE_BALL

- status: solved_rigorous_sufficient;
- radius: \(\|\Delta g\|/\|g_0\|\lesssim0.0573\) for the \(0.95696\) local bar;
- assumptions: centered affine Gaussian base and AQ perturbation;
- strength: orientation-free, finite deformation, no quadrature;
- limitation: conservative and still local rather than endpoint-task weighted.

## Exact centered-Gaussian rest spectrum and incumbent ownership

For the centered affine gate

\[
f(z)=\sigma(e^{\mathsf T}z)_+,
\]

write \(x=e^{\mathsf T}z\). Since

\[
x_+=\frac12x+\frac12|x|,
\]

the only odd Hermite coefficient is \(H_1\). In particular,

\[
\boxed{\langle f,H_3(x)\rangle=0.}
\]

For \(n=2k\ge2\), Gaussian integration by parts gives

\[
\mathbb E[x_+H_{2k}(x)]
=\mathbb E[D^{2k}x_+]
=(-1)^{k-1}(2k-3)!!\,\varphi(0).
\]

The normalized degree-\(2k\) energy is therefore

\[
\boxed{
E_{2k}
=\sigma^2
\frac{[(2k-3)!!]^2\varphi(0)^2}{(2k)!}.
}
\]

The first omitted mode after AQ is the collective rank-one H4 coefficient:

\[
\boxed{
E_4=\frac{\sigma^2}{48\pi}.
}
\]

Relative to total output energy \(\sigma^2/2\), H4 alone contributes

\[
\frac{E_4}{E_{\rm tot}}=\frac1{24\pi}\simeq0.013263.
\]

Thus

\[
\boxed{
q_{AQ+H4}^2
=\frac{\pi+3}{2\pi}+\frac1{24\pi}
\simeq0.990728.
}
\]

The next H6 energy is

\[
E_6=\frac{\sigma^2}{160\pi},
\qquad
\frac{E_6}{E_{\rm tot}}=\frac1{80\pi}\simeq0.003979.
\]

AQ plus H4 plus H6 therefore retains about \(0.994707\) of centered local output energy.

This changes the ownership interpretation. At the affine centered base, the omitted H3 source is exactly absent and the leading rest is a single collective H4 atom \(e^{\otimes4}\), the natural object of incumbent CP4/diagonal-H4 machinery. If that incumbent atom and its positive-depth word are already carried, the proposed synchronized AQ trajectory should not be charged for the base \(1/(24\pi)\) contribution again. The genuinely new rest begins when noncollective quadratic deformation creates signed H3/H4 components at order \(M^2\), subject to the separate old-arena audit.

### Graph node: AQ_GAUSSIAN_REST_OWNERSHIP

- status: solved_exact_base_spectrum;
- H3 at centered affine base: zero;
- first rest: rank-one H4, \(1/(24\pi)\) of output energy;
- AQ+H4 local fidelity: \(0.990728\);
- ownership dependency: verify incumbent CP4/H4 carries the same normalized atom and depth word;
- novelty boundary: identity-specific \(M^2\) H3/H4 deformation after complete old-span subtraction.

### Literal D11 CP4 ownership receipt

The deployed reset audit confirms the centered affine H4 atom is old:

- \(\texttt{\_cp4\_reset}\) gates inherited rank-one factors and appends the coordinate birth
  \[
  (\lambda_i,e_i),\qquad
  \lambda_i=k4_i^{target}-d_i^{inherited};
  \]
- \(\texttt{\_cp4\_read\_planes}\) reads \(\lambda_i e_i^{\otimes4}\) into K4, K31, and K22;
- \(\texttt{\_quartic\_residual}\) subtracts the retained CP4 planes before star/pair fitting.

In output-energy-normalized units, our raw affine coefficient

\[
-\frac1{\sqrt{2\pi}}
\]

becomes

\[
-\frac1{\sqrt\pi},
\]

because the centered ReLU output energy is \(1/2\). Likewise the raw covariance contribution

\[
\frac{\rho^4}{48\pi}
\]

becomes the deployed normalized kernel

\[
\frac{\rho^4}{24\pi}.
\]

Thus the normalization matches exactly. A direct centered-affine H4 oracle is duplicate. Lawful H4 novelty is restricted to:

1. CP4-dropped tail after complete star/pair/depth residualization; or
2. identity-specific quadratic/noncommutative H4 outside the retained external rank-one atoms, such as the \(q_{4,\perp}(211)\) commutator Gram.

The centered H6 contribution \(1/(80\pi)\) is not CP4-H4, although any later old-word return still requires a separate arena audit.

### Graph node: CP4_AFFINE_H4_OWNED

- status: solved_literal_duplicate;
- normalization: raw \(1/(48\pi)\), output-normalized \(1/(24\pi)\);
- deployed provider: \(\texttt{\_cp4\_reset}\) plus \(\texttt{\_cp4\_read\_planes}\);
- subtraction: \(\texttt{\_quartic\_residual}\) before stars;
- direct-H4 disposition: closed unless CP4 tail or identity-specific complete residual.

## Complete affine-Gaussian Hermite spectrum and H3/H4 credit

For the noncentered affine gate

\[
f(x)=(m+\sigma x)_+,\qquad \alpha=m/\sigma,
\]

Gaussian integration by parts gives, for every \(n\ge2\),

\[
\begin{aligned}
\mathbb E[fH_n(x)]
&=\mathbb E[D^nf(x)]\\
&=\sigma(-1)^nH_{n-2}(\alpha)\varphi(\alpha).
\end{aligned}
\]

Therefore the exact normalized degree-\(n\) energy is

\[
\boxed{
E_n(\alpha)
=\sigma^2
\frac{\varphi(\alpha)^2H_{n-2}(\alpha)^2}{n!},
\qquad n\ge2.
}
\]

In particular,

\[
\boxed{
E_3=\frac{\sigma^2\alpha^2\varphi(\alpha)^2}{6},
\qquad
E_4=\frac{\sigma^2(\alpha^2-1)^2\varphi(\alpha)^2}{24}.
}
\]

If incumbent H31/CP3 and CP4 carry these exact affine atoms with the same normalization and positive-depth ownership, the correct local retained fraction is

\[
\boxed{
q_{AQ+H3+H4}^2(\alpha)
=q_{AQ}^2(\alpha)
+\frac{
\varphi(\alpha)^2
\left[
\alpha^2/6+(\alpha^2-1)^2/24
\right]
}{
(1+\alpha^2)\Phi(\alpha)+\alpha\varphi(\alpha)
}.
}
\]

Illustratively, at \(\alpha=-1/2\), AQ alone retains about \(0.936\), while exact H3 plus H4 raises the local fraction to about \(0.975\), again above the idealized \(0.95696\) bar. At \(\alpha=-1\), AQ plus H3/H4 is only about \(0.944\); the H5 term,

\[
E_5
=\sigma^2\frac{\varphi(\alpha)^2H_3(\alpha)^2}{120},
\]

is then material and lifts the affine local fraction to roughly \(0.970\). Thus substantially negative gates can require a beyond-H4 rest companion even when H3/H4 ownership is perfect.

These are exact affine source energies, not endpoint gains. Their proper use is to credit already-carried chaos before forming the residual radius, never to add a duplicate feature.

### Graph node: AQ_AFFINE_REST_SPECTRUM

- status: solved_exact;
- formula: \(E_n=\sigma^2\varphi(\alpha)^2H_{n-2}(\alpha)^2/n!\);
- use: exact local ownership credit for H3/H4 and analytic tail triage;
- warning: negative gates near \(\alpha=-1\) make H5 and above material;
- next gate: match deployed CP3/H31/CP4 normalization and determine task-weighted standardized-mean support.

## Coupled D10+D14 closure: exact AQ low chaos plus Gaussian Mehler rest

The synchronized AQ trajectory should not discard an affine Gaussian tail that the incumbent covariance recursion already knows how to carry. For two standardized jointly Gaussian coordinates \(X,Y\) with correlation \(\rho\), Mehler's identity gives

\[
\operatorname{Cov}(f_i(X),f_j(Y))
=\sum_{n\ge1}
\frac{c_{i,n}c_{j,n}}{n!}\rho^n,
\qquad
c_{i,n}=\mathbb E[f_i(X)H_n(X)].
\]

For affine ReLUs,

\[
c_{i,n}
=\sigma_i(-1)^n
H_{n-2}(\alpha_i)\varphi(\alpha_i),
\qquad n\ge2.
\]

Therefore the exact Gaussian rest beyond AQ is

\[
\boxed{
K_{ij}^{G,>2}
=\sigma_i\sigma_j
\varphi(\alpha_i)\varphi(\alpha_j)
\sum_{n\ge3}
\frac{
H_{n-2}(\alpha_i)H_{n-2}(\alpha_j)
}{n!}\rho_{ij}^n.
}
\]

The signs cancel because both degree-\(n\) coefficients contain \((-1)^n\).

Let \(R_i^{AQ}\in\mathcal H_{\le2}\) be the exact projective AQ reset output. The natural coupled covariance closure is

\[
\boxed{
C_{ij}^{D10+D14}
=\langle R_i^{AQ},R_j^{AQ}\rangle
+K_{ij}^{G,>2},
}
\]

with the Gaussian rest evaluated in the matched affine/covariance frame used by D10. At \(A_i=A_j=0\), this is exactly the full Gaussian ReLU covariance, not an approximation. Hence centered H4, negative-mean H3/H5, and the entire affine Mehler tail are not D14 losses if the incumbent D10 branch already carries the same rest.

For \(A\ne0\), D14 supplies the exact coadapted \(H_{\le2}\) projection and all AQ \(q\le4\) potentials, while D10 supplies an affine control variate for the uncarried rest. The genuinely missing object is

\[
\boxed{
\Delta K_{ij}^{\rm rest}
=
\langle r_i^{\rm true},r_j^{\rm true}\rangle
-K_{ij}^{G,>2},
}
\]

after subtracting every incumbent H31/CP4/Q24 correction already applied to that rest. This is much smaller in principle than charging the entire AQ projection residual.

### Exact Mehler truncation bound

If the D10 implementation truncates at degree \(p\), Cauchy--Schwarz gives

\[
\boxed{
\left|
\sum_{n>p}\frac{c_{i,n}c_{j,n}}{n!}\rho^n
\right|
\le
|\rho|^{p+1}
\sqrt{E_{i,>p}E_{j,>p}},
}
\]

where

\[
E_{i,>p}
=\sum_{n>p}\frac{c_{i,n}^2}{n!}
\]

is the exact one-unit omitted Hermite energy, available from the total stop-loss energy minus the retained finite sum. This provides a symbolic truncation-depth receipt. Correlations near \(|\rho|=1\) are the only slow regime; away from one the bound decays geometrically.

For a desired pairwise absolute tolerance \(\epsilon_{ij}\) and
\(0<|\rho_{ij}|<1\), a sufficient symbolic truncation depth is

\[
\boxed{
p\ge
\left\lceil
\frac{
\log\!\left(\epsilon_{ij}/\sqrt{\bar E_i\bar E_j}\right)
}{
\log|\rho_{ij}|
}
\right\rceil-1,
}
\]

where \(\bar E_i\ge E_{i,>p}\) and \(\bar E_j\ge E_{j,>p}\) are any conservative \(p\)-independent rest-energy bounds. At \(|\rho|=1\), degree truncation alone has no geometric certificate and the exact same-direction scalar law or a residual-energy bound must be used.

### Graph node: D10_D14_GAUSSIAN_REST_COUPLING

- status: solved_exact_at_affine_base/conditional_beyond_affine;
- D14 role: exact coadapted \(H_{\le2}\) state and AQ cumulant potentials;
- D10 role: full affine Gaussian \(H_{\ge3}\) covariance rest via Mehler;
- exact base: \(A=0\);
- new residual: true non-Gaussian rest minus Gaussian rest and all old H31/CP4/Q24 corrections;
- truncation gate: \(|\rho|^{p+1}\sqrt{E_{i,>p}E_{j,>p}}\);
- decisive audit: confirm the deployed D10 kernel/frame and subtraction normalization match this decomposition.

## Exact bivariate Gaussian ReLU kernel as a no-truncation fallback

The affine Gaussian rest can also be evaluated without a Mehler series. Let

\[
G_i=m_i+\sigma_iX,\qquad
G_j=m_j+\sigma_jY,
\]

where \((X,Y)\) are standard Gaussian with correlation \(\rho\), and set

\[
\alpha=m_i/\sigma_i,\qquad
\beta=m_j/\sigma_j,\qquad
s=\sqrt{1-\rho^2}.
\]

Define

\[
P=\Phi_2(\alpha,\beta;\rho),
\]

\[
A=\Phi\!\left(\frac{\beta-\rho\alpha}{s}\right),
\qquad
B=\Phi\!\left(\frac{\alpha-\rho\beta}{s}\right),
\]

and let \(\varphi_2(\alpha,\beta;\rho)\) be the bivariate standard normal density. The needed truncated moments are

\[
\boxed{
I_x
=\mathbb E[X\,1_{\{G_i>0,G_j>0\}}]
=\varphi(\alpha)A+\rho\varphi(\beta)B,
}
\]

\[
\boxed{
I_y
=\varphi(\beta)B+\rho\varphi(\alpha)A,
}
\]

\[
\boxed{
\begin{aligned}
I_{xy}
={}&\mathbb E[XY\,1_{\{G_i>0,G_j>0\}}]\\
={}&\rho P
-\rho\alpha\varphi(\alpha)A
-\rho\beta\varphi(\beta)B\\
&+(1-\rho^2)\varphi_2(\alpha,\beta;\rho).
\end{aligned}}
\]

Therefore

\[
\boxed{
\begin{aligned}
\mathbb E[(G_i)_+(G_j)_+]
={}&m_im_jP
+m_i\sigma_jI_y
+m_j\sigma_iI_x\\
&+\sigma_i\sigma_jI_{xy}.
\end{aligned}}
\]

Subtracting the one-unit means gives the exact covariance. At
\(\alpha=\beta=0\), the formula reduces to the standard arc-cosine ReLU kernel

\[
\frac{\sigma_i\sigma_j}{2\pi}
\left[
\sqrt{1-\rho^2}
+(\pi-\arccos\rho)\rho
\right].
\]

The limits \(\rho\to\pm1\) are evaluated by the corresponding one-dimensional degenerate Gaussian law, avoiding cancellation in the displayed \(s^{-1}\) arguments.

This formula costs \(O(1)\) scalar special-function work per pair and no dense matrix multiplication. Whether it is cheaper than a fixed Mehler polynomial under the board's literal billing is an implementation question, but mathematically it removes truncation error completely and is the natural audit reference.

### Graph node: D10_GAUSSIAN_REST_BIVARIATE_EXACT

- status: solved_exact;
- inputs: two means, two scales, one correlation;
- output: exact affine ReLU pair covariance;
- use: reference oracle and \(|\rho|\simeq1\) fallback for D10_D14_GAUSSIAN_REST_COUPLING;
- cost caveat: scalar \(\Phi_2\) evaluation/billing must be audited against Mehler truncation.

## The coupled closure sequence: Gaussian rest, coadapted AQ cumulants, and D11 response

The constructive mechanism can now be stated without a tensor carrier.

1. **D10 Gaussian base.** Propagate means/covariances and evaluate the exact affine Gaussian ReLU pair kernel \(K_G\), by Mehler or the bivariate formula above.
2. **D14 coadapted state.** Propagate packed \((m,u,A)\) coefficients and apply the exact convex projective reset \(\nabla_{AQ}J\).
3. **Exact connected sources.** Evaluate \(\Phi_3,\Phi_4\) and their polarizations on the mixed D14 state. This supplies every connected AQ K21/K22/K31/q4 equality sector with no fitted diagram coefficient.
4. **Incumbent response geometry.** Apply the existing D11/Q24 Gaussian response derivatives to those connected cumulants at the same Site-A/Site-B covariance injection where the certified variance correction has value.
5. **Joint recurrence.** Feed the corrected covariance/frame back into the next D10+D14 reset, rather than freezing the first-order tangent.

Formally, for a centered law with Gaussian match \(G\), the cumulant response is the connected differential operator

\[
\boxed{
\mathbb E[F(X)]
=
\left.
\exp\!\left[
\sum_{q\ge3}\frac1{q!}\kappa_q(X):\partial^q
\right]
\mathbb E[F(G+x)]
\right|_{x=0},
}
\]

interpreted distributionally for ReLU and analytically after Gaussian convolution. Through first order in connected non-Gaussianity,

\[
\boxed{
\Delta\mathbb E[F]
=\frac1{3!}\kappa_3:\partial^3K_G
+\frac1{4!}\kappa_4:\partial^4K_G
+R_{\ge5,\,{\rm nonlinear}}.
}
\]

The next formal terms include

\[
\frac1{2(3!)^2}\kappa_3^{\otimes2}:\partial^6K_G,
\qquad
\frac1{3!\,4!}\kappa_3\otimes\kappa_4:\partial^7K_G,
\]

plus \(\kappa_5,\kappa_6,\ldots\). Existing CP/star/counterterm branches may own some of these disconnected/nonlinear responses and must be subtracted literally.

This gives the mathematical interpretation of the certified 18.7% frozen chain: it is a first-order value receipt for local connected-cumulant supply into the covariance response. It is not the capacity ceiling of the self-consistent sequence above. The new architecture changes the sources and the frame jointly at every layer, while D10 keeps the full Gaussian rest.

### Exact error partition

After old-component subtraction, the remaining response error is partitioned into

\[
\boxed{
\rho_{\rm total}
=\rho_{\rm reset}^{>AQ}
+\rho_{\rm cumulant}^{q\ge5/\rm nonlinear}
+\rho_{\rm frame/secant}
+\rho_{\rm quadrature}
+\rho_{\rm ownership}.
}
\]

The lane now has analytic gates for the AQ reset residual, tangent leakage, finite secant, Gaussian-rest truncation, and characteristic quadrature. The genuinely open physics term is the connected beyond-q4/nonlinear response after incumbent CP3/CP4/star ownership.

### Graph node: D10_D14_CUMULANT_RESPONSE_SEQUENCE

- status: solved_mechanism/open_residual_radius;
- base: exact D10 Gaussian covariance kernel;
- state: exact projective D14 \((m,u,A)\);
- sources: exact AQ \(\Phi_3,\Phi_4\) polarizations;
- response: existing D11/Q24 connected derivative sites;
- novelty: joint positive-depth recurrence, not static same-site moments;
- terminal proof: response-normalized sum of the five typed residuals must satisfy the master \(r_v,\phi,R_{\rm other}\) frontier.

## Current lane decision record

### Constructive candidate

The strongest derivation-backed carrier is the coupled self-consistent sequence

\[
\boxed{
\text{D10 Gaussian rest}
+\text{D14 exact projective AQ state}
+\text{D11 connected q3/q4 response}.
}
\]

Its mathematical ingredients are now closed:

1. packed target-blind state \((m,u,A)\);
2. exact reset \(\mathcal R=\nabla_{AQ}J\);
3. exact \(\Phi_2,\Phi_3,\Phi_4\) joint-cumulant generators;
4. physical dense-mix polarization completing \(21,111,31,22,211,1111\);
5. exact affine Gaussian rest by Mehler or a bivariate normal formula;
6. strict-depth secant/Duhamel and tangent-leakage identities.

The rank-dependent packed width and leading costs are

\[
d(r)=r+\frac{r(r+1)}2,
\]

\[
C_{\rm one\ mix/layer}
=\frac{d(r)}n+\frac{r^3}{n^2},
\]

\[
C_{\rm state+direction/layer}
=2\frac{d(r)}n+2\frac{r^3}{n^2},
\]

before reset quadrature, hooks, and any already-resident-state credit.

At \(r=32\), \(d=560\). One state mix plus one local product is about \(0.5781\) per layer, or \(8.09\) over fourteen; the synchronized state+direction estimate is \(1.15625\) per layer, or \(16.19\) over fourteen, reduced to about \(8.53\) only if the post-weight state is already resident.

At \(r=45\), \(d=1080\). The corresponding state+direction estimate is about \(2.283\) per layer, or \(22.83\) over ten; with a resident post-weight state it is about \(1.2285\) per layer, or \(12.285\) over ten.

At \(r=55\), \(d=1595\). One state mix alone costs \(1.5576\) per layer and one local \(r^3\) product costs \(0.1587\). State+direction with two local products is about \(3.4325\) per layer, or \(34.33\) over ten; even with the post-weight state resident it is about \(1.8749\) per layer, or \(18.75\) over ten. Thus the favorable r32 census must not be transferred to r55. Literal rank choice, billing, and reuse are binding architecture gates.

### Latent-rank fidelity gate

Choosing \(r=32\) is itself an approximation if the synchronized AQ state naturally occupies \(r=45\) or \(55\). Let \(\Pi_k\) be the common latent projection and

\[
\widetilde u_i=\Pi_ku_i,\qquad
\widetilde A_i=\Pi_kA_i\Pi_k.
\]

There is no universal preservation theorem for the noncommutative q4 sector under \(k<r\). For one diagonal pure-AA potential, a deterministic error bound follows from the noncommutative telescoping identity:

\[
A^4-B^4
=A^3(A-B)+A^2(A-B)B+A(A-B)B^2+(A-B)B^3.
\]

Hence

\[
\boxed{
|\operatorname{tr}(A^4-B^4)|
\le
\|A-B\|_*
\sum_{j=0}^3
\|A\|_{\rm op}^{3-j}\|B\|_{\rm op}^{j}.
}
\]

Analogous bounds for \(u^{\mathsf T}A^2u\) involve both
\(\|u-\widetilde u\|\) and \(\|A-\widetilde A\|\). These are upper certificates only; generic families \(\{A_i\}\) can span all of \(\operatorname{Sym}(r)\), and their commutator Gram can be full.

Therefore a cheap rank-32 D14 requires an endpoint-weighted receipt for

\[
\Delta\Phi_q
=\Phi_q(u,A)-\Phi_q(\widetilde u,\widetilde A)
\]

after physical mixing and old-span residualization. Source Frobenius energy or a single-net SVD is insufficient. The prior identity-specific carrier/compression failures make this a high-risk gate, not a bookkeeping detail.

### Graph node: AQ_LATENT_RANK_COST_FIDELITY

- status: open_binding;
- cheap branch: \(r=32\), conditional on response-weighted \(\Phi_3/\Phi_4\) projection tail;
- exact/high-rank branch: \(r=45/55\), arithmetic substantially larger;
- no-go: no target-blind universal q4 preservation under a common \(k<r\) projection;
- terminal receipt: projected-vs-full D14 response error inside the master residual budget.

### Retained-product suffix exception for \(q_{4,\perp}(211)\)

If the D10/D14 implementation already retains the nonsymmetric latent product

\[
P_{ij}=A_iA_j,
\]

then its skew half is

\[
\operatorname{skew}P_{ij}
=\frac12(A_iA_j-A_jA_i)
=\frac12[A_i,A_j].
\]

Therefore the commutator cross angle in the q211 formula,

\[
\langle[A_i,A_j],[A_i,A_k]\rangle_F,
\]

is a Frobenius pairing of already resident skew products and needs no new latent GEMM. This is a real arithmetic exception for a fixed suffix action.

It is not a complete low-rank provider. The audited generic anchor-commutator Gram has external rank about \(990\) at \(r=45\) and \(1023\) at \(r=55\). A rank-one or few-rank suffix is therefore only a scalar target candidate; it requires an endpoint-weighted concentration/tail receipt and complete CP4/star/depth residualization.

### Graph node: Q4PERP_RETAINED_PRODUCT_SUFFIX

- status: conditional_zero_GEMM_scalar;
- premise: exact \(A_iA_j\) product resident in the same gauge/cut/lifetime;
- read: skew-product Frobenius pairing;
- novelty: identity-specific q211 commutator sector after old-span subtraction;
- completeness: generic low-rank suffix closed by anchor-Gram rank \(990/1023\);
- promotion: task-weighted scalar/tail certificate, not source rank-one intuition.

### Proven duplicate/closed branches

- total same-site modeled K31, including its AQ trace form: old D11/Q24;
- centered affine H4 atom: exact old CP4 object, raw \(1/(48\pi)\), normalized \(1/(24\pi)\);
- generic fixed-site crossed-K22 costate: cost no-go absent low rank;
- per-unit short \(M^2\) tags as an exact future carrier: no, because their linear span is full \(\operatorname{Sym}^3\);
- source energy/correlation without endpoint adjoint angle: no positive value lower bound.

### Genuine candidate novelty

- coadapted positive-depth \(A_i\) trajectory;
- noncommutative \(q_{4,\perp}(211)\) and all-distinct q1111 generated by diagonal polarization after a later dense mix;
- finite-secant reset terms of order at least three after literal old tangent/square subtraction;
- CP4-dropped or identity-specific H3/H4 rest after complete arena residualization;
- true-minus-Gaussian non-AQ rest in the coupled D10+D14 closure.

### Exact fail-fast gates

1. **Master response:** optimistic normalized variance residual \(r_v\le0.20746\), tightened by coefficient fidelity and other channels.
2. **Local AQ fidelity:** \(q_{AQ}^2=\|\nabla J\|^2/(2J)\); affine threshold crosses \(0.95696\) near \(\alpha=-0.33\).
3. **Centered deformation:** rigorous safe ball \(\|\Delta g\|/\|g_0\|\le0.0573\); second-order noncollective scale
   \[
   \sqrt{\|w\|^2+\operatorname{tr}B^2}/\sigma\lesssim0.359.
   \]
4. **Frozen tangent leakage:**
   \[
   c_l^2=\|T_l-T_l^2\|,
   \]
   with exact directional form \(\langle h,(T-T^2)h\rangle\) and the even-crossing path bound.
5. **AQ characteristic oracle:** analytic scalar and differentiated tail bounds; singular critical strata separated.
6. **Gaussian rest:** Mehler tail
   \[
   |\rho|^{p+1}\sqrt{E_{i,>p}E_{j,>p}},
   \]
   or exact bivariate fallback.
7. **Ownership:** full CP3/CP4/star/pair/depth-arena Schur subtraction before any gain credit.

### Current next question

Does the literal deployed D10/D11 lineage already implement the affine Gaussian-rest subtraction and enough of the positive-depth \(A_i\) trajectory that the proposed coupled recurrence is duplicate, or is the synchronized exact projective reset genuinely absent? If absent, the next mathematical certificate is the response-normalized non-Gaussian-rest radius

\[
\sum_l\left\langle
\lambda_{l,Q},
r_l^{true}-r_l^{Gaussian/old}
\right\rangle
\]

against the \(0.20746\) frontier. That single typed quantity decides whether the complete exact-q mechanism can plausibly deliver goal-scale raw reduction.

### Graph node: AQ_RESET_SCALAR_POTENTIAL

- status: solved_exact_interface;
- state input/output: one mean, one latent vector, one symmetric latent matrix;
- primitive: one scalar one-dimensional characteristic integral per unit plus derivatives; reset residual energy is then algebraic;
- downstream: AQ_FINITE_SECANT_POSTSQUARE, AQ_CUMULANT_POTENTIAL_STACK, and AQ_BEYOND_RESIDUAL_DUHAMEL;
- arithmetic caveat: exact analytic identity does not by itself certify a finite quadrature/truncation error or deployed billing;
- next gate: a symbolic quadrature/resolvent error bound tight enough that reset-oracle error fits inside the \(r_v\) fidelity frontier.

## Analytic tail receipt for the AQ characteristic integral

Let \(\lambda_j\) be the eigenvalues of \(A\), and write \(u_j\) in the same eigenbasis. The characteristic modulus is exactly

\[
\boxed{
|\chi_g(t)|
=\prod_j(1+t^2\lambda_j^2)^{-1/4}
\exp\!\left[
-\frac{t^2}{2}\sum_j
\frac{u_j^2}{1+t^2\lambda_j^2}
\right].
}
\]

The apparent \(1/t^2\) tail in \(F\) comes from the constant term, so split it analytically:

\[
\boxed{
\begin{aligned}
F={}&\frac m2+\frac1\pi
\int_0^T\frac{1-\operatorname{Re}\chi_g(t)}{t^2}\,dt\\
&+\frac1{\pi T}
-\frac1\pi\int_T^\infty
\frac{\operatorname{Re}\chi_g(t)}{t^2}\,dt.
\end{aligned}}
\]

Let \(k=\operatorname{rank}A\), let

\[
\lambda_{\min}^+
=\min_{\lambda_j\ne0}|\lambda_j|,
\qquad
C_A=\prod_{\lambda_j\ne0}|\lambda_j|^{-1/2}.
\]

For \(T\lambda_{\min}^+\ge1\),

\[
|\chi_g(t)|\le C_A t^{-k/2},
\]

and therefore the omitted residual tail obeys

\[
\boxed{
\left|
\frac1\pi\int_T^\infty
\frac{\operatorname{Re}\chi_g(t)}{t^2}\,dt
\right|
\le
\frac{C_A}
{\pi(1+k/2)T^{1+k/2}}.
}
\]

If \(u_0\) is the component of \(u\) in \(\ker A\), the stronger exact factor

\[
\exp(-t^2\|u_0\|^2/2)
\]

multiplies this bound. Thus \(A=0,u\ne0\) has a Gaussian tail; the deterministic case \(A=0=u\) needs no quadrature.

This certifies the scalar \(F\) value without a long numerical convergence campaign. Parameter derivatives of \(\chi\) introduce explicit resolvent/polynomial factors and require their own differentiated versions of the same tail bound; the scalar receipt alone must not be reused silently for reset derivatives.

### Differentiated tail order

The needed derivatives are explicit:

\[
\partial_m\chi=it\chi,\qquad
\partial_m^2\chi=-t^2\chi,
\]

\[
\nabla_u\chi=-t^2K_t^{-1}u\,\chi,
\]

\[
\nabla_u^2\chi
=\left[
t^4(K_t^{-1}u)(K_t^{-1}u)^{\mathsf T}
-t^2K_t^{-1}
\right]\chi,
\qquad K_t=I-itA.
\]

Mixed derivatives follow by products of these factors. If \(A\) has full rank \(k=r\), then for \(t\lambda_{\min}^+\ge1\),

\[
\|K_t^{-1}\|_{\rm op}
\le\frac1{t\lambda_{\min}^+}.
\]

Under this full-rank resolvent bound, every fixed parameter derivative \(D^\alpha\chi\) of total order \(d=|\alpha|\) admits a recursively explicit constant \(K_\alpha(A,u)\) such that

\[
|D^\alpha\chi(t)|
\le K_\alpha(1+t)^d|\chi(t)|.
\]

After division by the \(t^2\) in the \(F\) integral, the residual derivative tail therefore satisfies

\[
\boxed{
|R_{\alpha}(T)|
\le
\frac{K_\alpha C_A}
{\pi(k/2+1-d)}
T^{-(k/2+1-d)},
\qquad d<1+\frac k2,
}
\]

up to replacing \(1+t\) by a harmless factor \(2t\) for \(t\ge1\). The constants are obtained directly from the displayed resolvent derivatives, not fitted.

The exact reset formulas require \(d\le2\). For full-rank \(r=32\), the worst differentiated polynomial tail is therefore order

\[
T^{-(1+16-2)}=T^{-15}.
\]

If \(A\) is rank deficient, derivatives in kernel-coupling directions can grow faster than \(t^d\); the displayed polynomial exponent must not be reused. A nonzero \(u_0\in\ker A\) restores Gaussian damping and dominates any fixed polynomial growth. If both full rank and nullspace damping fail, higher reset derivatives may genuinely be nonsmooth and must use a direction-specific resolvent bound or a stratified/distributional treatment rather than the full-rank formula.

### Graph node: AQ_RESET_CHARACTERISTIC_TAIL

- status: solved_scalar_tail;
- leading tail: analytic \(1/(\pi T)\);
- residual: polynomial order \(T^{-1-k/2}\), with optional Gaussian nullspace damping;
- dependency: eigenvalue/rank receipt for each AQ \(A\);
- derivative extension: solved for full-rank \(A\) under \(d<1+r/2\), and separately under Gaussian nullspace damping; singular rank-deficient strata remain direction-specific.

## Literal D10 ownership correction: only the finite Gaussian block \(n=3,\ldots,6\)

The deployed-lineage audit changes the coupling identity materially.  At the first
centered reset the Gaussian bivariate ReLU covariance is exact, but at later D11/Q24
resets the weight-zero baseline has `_Q24_TERMS=7`, i.e. Hermite orders
\(n=0,\ldots,6\).  After mean-outer subtraction, the resident covariance owns
\(n=1,\ldots,6\), not the entire Mehler tail.

For

\[
f_i(x)=\sigma_i(x+\alpha_i)_+,
\qquad \alpha_i=m_i/\sigma_i,
\]

with probabilists' Hermites, the raw coefficients are

\[
c_{i,1}=\sigma_i\Phi(\alpha_i),
\qquad
c_{i,n}=\sigma_i\phi(\alpha_i)H_{n-2}(-\alpha_i),\quad n\ge2.
\]

Hence the literal centered D10 baseline is

\[
C_{ij}^{G,\le6}
=\sum_{n=1}^{6}\frac{c_{i,n}c_{j,n}}{n!}\rho_{ij}^{,n}.
\]

Let the exact AQ projection after reset be

\[
R_i^+=m_i^+ + u_i^{+\mathsf T}z
+\frac12(z^{\mathsf T}A_i^+z-\operatorname{tr}A_i^+).
\]

The once-only coupled covariance is therefore

\[
\boxed{
C_{ij}^{\rm coupled}
=C_{ij}^{G,\le6}
-c_{i,1}c_{j,1}\rho_{ij}
-\frac12c_{i,2}c_{j,2}\rho_{ij}^2
+u_i^{+\mathsf T}u_j^+
+\frac12\operatorname{tr}(A_i^+A_j^+).
}
\]

Thus D14 replaces exactly the Gaussian \(n=1,2\) block while D10 continues to own
only \(n=3,\ldots,6\).  The missing affine-Gaussian tail is

\[
T_{ij}^{G,\ge7}
=\sum_{n\ge7}\frac{c_{i,n}c_{j,n}}{n!}\rho_{ij}^{,n},
\]

with the label-free receipt

\[
\boxed{
|T_{ij}^{G,\ge7}|
\le |\rho_{ij}|^7
\sqrt{E_{i,>6}^{G}E_{j,>6}^{G}},
\qquad
E_{i,>6}^{G}=\sum_{n\ge7}\frac{c_{i,n}^2}{n!}.
}
\]

The sixteen EC4 response channels remain separate and once-only.  CP4 ownership of
the affine \(H_4\) atom does not imply ownership of \(n\ge7\).  Consequently the
correct beyond-AQ residual is

\[
\Delta C_{ij}
=C_{ij}^{\rm true}
-C_{ij}^{\rm coupled}
-C_{ij}^{\rm EC4/CP3/CP4/star,\,once},
\]

and not “true covariance minus a full D10 Gaussian tail.”  Any proposal claiming an
exact full Gaussian rest must either use the exact bivariate kernel, extend the
series, or charge the displayed tail.

### Graph node: D10_LITERAL_MEHLER_3TO6

- status: literal_ownership_solved;
- owns: centered Gaussian Hermite orders \(3,4,5,6\) after D14 replaces orders \(1,2\);
- does_not_own: Gaussian \(n\ge7\), non-Gaussian beyond-AQ rest, or EC4 response channels;
- exact coupling: boxed subtraction/addition identity above;
- fail-fast receipt: endpoint-weight the \(|\rho|^7\sqrt{E_{>6}E_{>6}}\) tail before claiming completeness.

## Affine \(H_3\) synchronization remains an unresolved literal-overlap audit

For the affine slice \(g=m+\sigma x\), the raw third Hermite coefficient is

\[
c_3=\sigma\phi(\alpha)H_1(-\alpha)
=-\sigma\alpha\phi(\alpha),
\qquad \alpha=m/\sigma.
\]

In multivariate notation this is the collective rank-one tensor

\[
T_0=-\sigma\alpha\phi(\alpha)e^{\otimes3}
\]

(the Wiener kernel is \(T_0/3!\)).  Existing CP3 reset moments and H31/H13
consumers make duplication plausible, but they do **not** establish equality at the
same pre/post-reset seam, normalization, diagonal convention, and positive-depth
recurrence.  Unlike the now-literal CP4 affine-H4 receipt, no code-level equality
receipt has yet been produced for this affine-H3 source.  Therefore \(T_0\) remains
inside the synchronization residual until an algebraic diff proves otherwise.

### Graph node: CP3_AFFINE_H3_OWNERSHIP_AUDIT

- status: open_literal_overlap;
- proved: affine source is the collective rank-one tensor above;
- not proved: exact equality to the stored CP3/H31 source along the deployed seam;
- discipline: do not subtract it as old span merely by analogy with CP4.

## Cheap exact extension of the missing Gaussian Mehler tail

The fact that D10 stops at order six is not a generic-GEMM obstruction.  On the
affine Gaussian baseline every extra Hermite order is an elementwise \(n\times n\)
update.

Let

\[
d_{i,n}=\frac{c_{i,n}}{\sqrt{n!}},
\qquad R=(\rho_{ij}).
\]

Then the order-\(n\) covariance block is

\[
K^{(n)}=\operatorname{Diag}(d_n)R^{\circ n}
\operatorname{Diag}(d_n),
\]

and

\[
K^{G,\le p}=\sum_{n=1}^{p}K^{(n)}.
\]

Both coefficient and correlation powers obey scalar/elementwise recurrences:

\[
H_{k+1}(x)=xH_k(x)-kH_{k-1}(x),
\qquad
R^{\circ(n+1)}=R^{\circ n}\circ R.
\]

Therefore extending D10 from \(p=6\) to any fixed \(p\) costs
\(O((p-6)n^2)\) multiply-adds and no dense matrix product.  This closes the affine
Gaussian tail to a certified tolerance before spending any state rank on the
non-Gaussian problem.

For one unit the exact retained-energy fraction through order \(p\) is

\[
\boxed{
q_{\le p}^2(\alpha)=
\frac{
[\phi(\alpha)+\alpha\Phi(\alpha)]^2
+\Phi(\alpha)^2
+\phi(\alpha)^2\displaystyle\sum_{n=2}^{p}
\frac{H_{n-2}(-\alpha)^2}{n!}
}
{(1+\alpha^2)\Phi(\alpha)+\alpha\phi(\alpha)}.
}
\]

This is a symbolic, label-free per-unit gate.  Choose the smallest \(p_i\) for which
the omitted energy is below the allotted endpoint tolerance; no empirical retuning
is needed.

At the centered seam only even orders survive.  The exact order-\(2k\) energy is

\[
E_{2k}
=\frac{\sigma^2}{2\pi}
\frac{[(2k-3)!!]^2}{(2k)!}.
\]

Since AQ owns \(n\le2\), D10 through six raises the centered retained output-energy
fraction from

\[
\frac{\pi+3}{2\pi}=0.977465\ldots
\]

to

\[
\boxed{
\frac{\pi+3}{2\pi}
+\frac1{24\pi}
+\frac1{80\pi}
=0.994707\ldots .
}
\]

Thus the literal missing \(n\ge7\) Gaussian tail at a centered affine reset is only
\(0.5293\%\) of output energy.  This does not prove the non-Gaussian D14 mechanism,
but it removes a false attribution of that small remainder to an already-resident
D10 state.

For an endpoint adjoint \(B\), the exact tail matrix \(T_{>p}\) is positive
semidefinite by the Schur product theorem.  Two label-free bounds are

\[
\boxed{
|\langle B,T_{>p}\rangle|
\le \|B\|_{\rm op}\sum_i E_{i,>p}^{G}
}
\]

and the correlation-sensitive bound

\[
\boxed{
|\langle B,T_{>p}\rangle|
\le\sum_{ij}|B_{ij}|\,|\rho_{ij}|^{p+1}
\sqrt{E_{i,>p}^{G}E_{j,>p}^{G}}.
}
\]

The first is safe for arbitrary signs in \(B\); the second is usually sharper away
from nearly-collinear pairs.  Either supplies the exact stop rule for choosing
\(p\).

### Graph node: D10_MEHLER_ELEMENTWISE_EXTENSION

- status: exact_cheap_closure;
- provider: affine Gaussian Hermite orders \(n\ge7\);
- cost: \(O((p-6)n^2)\), no GEMM-equivalent dense product;
- gates: exact \(q_{\le p}^2(\alpha)\) and endpoint-adjoint tail bounds above;
- dependency: `D10_LITERAL_MEHLER_3TO6`;
- limitation: closes only the affine Gaussian rest, not the non-Gaussian AQ residual.

### Stronger affine specialization: exact one-\(\Phi_2\) pair kernel

The finite Mehler extension is a cheap fallback, not the affine mathematical
ceiling.  For jointly Gaussian standardized \((Z_i,Z_j)\) with correlation \(\rho\),
put

\[
X=m_i+\sigma_iZ_i,
\quad Y=m_j+\sigma_jZ_j,
\quad \alpha=m_i/\sigma_i,
\quad \beta=m_j/\sigma_j,
\quad s=\sqrt{1-\rho^2}.
\]

Define

\[
P=\Phi_2(\alpha,\beta;\rho),
\quad
A=\Phi\left(\frac{\beta-\rho\alpha}{s}\right),
\quad
B=\Phi\left(\frac{\alpha-\rho\beta}{s}\right),
\]

\[
I_x=\phi(\alpha)A+\rho\phi(\beta)B,
\qquad
I_y=\phi(\beta)B+\rho\phi(\alpha)A,
\]

\[
I_{xy}
=\rho P-\rho\alpha\phi(\alpha)A
-\rho\beta\phi(\beta)B
+(1-\rho^2)\phi_2(\alpha,\beta;\rho).
\]

Then

\[
\boxed{
E[X_+Y_+]
=m_im_jP+m_i\sigma_jI_y+m_j\sigma_iI_x
+\sigma_i\sigma_jI_{xy}.
}
\]

Thus every affine Mehler order, including \(n\ge7\), is closed exactly with one
bivariate Gaussian cdf evaluation per covariance entry, no persistent chaos state,
and no dense matrix transport.  Its value/JVP/Hessian follow by differentiating the
same scalar kernel.  The remaining engineering choice is:

- exact \(\Phi_2\) kernel on all pairs;
- cheap elementwise Q7/Q8 extension plus a certified \(|\rho|^9\) tail;
- or exact kernel only on the tail-bound exception set.

This affine result does not contradict `AQ_BIVARIATE_CHARACTERISTIC_ORACLE`: that
node concerns two **quadratic** Gaussian laws with noncommuting matrix pencils.  The
affine specialization collapses to a scalar closed form.

### Graph node: D10_EXACT_BIVARIATE_RELU_KERNEL

- status: exact_affine_tail_closure;
- provider: all affine Gaussian Mehler orders and their scalar derivatives;
- state/transport: none beyond resident mean, variance, covariance;
- arithmetic: one \(\Phi_2\) evaluation per pair, requiring a literal runtime/billing audit;
- fallback: Q7/Q8 elementwise recurrence with endpoint-weighted \(|\rho|^9\) receipt;
- limitation: does not close primitive non-Gaussian D14 rest.

## Exact bivariate AQ-ReLU oracle and its generic cost boundary

There is an exact tensor-free oracle for the complete pair covariance of two AQ
preactivations.  It is useful as a reference and as a sparse-exception fallback,
but it does not by itself solve the dense deployed cost problem.

Let

\[
X=m_i+u_i^{\mathsf T}z
+\frac12(z^{\mathsf T}A_i z-\operatorname{tr}A_i),
\qquad
Y=m_j+u_j^{\mathsf T}z
+\frac12(z^{\mathsf T}A_j z-\operatorname{tr}A_j).
\]

For real \(t,s\), define

\[
M(t,s)=tA_i+sA_j,
\quad h(t,s)=tu_i+su_j,
\]

\[
c(t,s)=t\left(m_i-\frac12\operatorname{tr}A_i\right)
+s\left(m_j-\frac12\operatorname{tr}A_j\right).
\]

Their exact joint characteristic function is

\[
\boxed{
\chi_{ij}(t,s)
=e^{ic(t,s)}
\det(I-iM(t,s))^{-1/2}
\exp\!\left[-\frac12h(t,s)^{\mathsf T}
(I-iM(t,s))^{-1}h(t,s)\right].
}
\]

No fourth-order tensor appears.  The positive-part cross moment follows from
\(x_+y_+=(xy+x|y|+y|x|+|x||y|)/4\).  In particular,

\[
E[XY]=-\partial_t\partial_s\chi_{ij}(0,0),
\]

\[
E[X|Y|]
=\frac2\pi\int_0^\infty
\frac{E[X]-\operatorname{Im}\partial_t\chi_{ij}(0,s)}{s^2}\,ds,
\]

and symmetrically for \(E[Y|X|]\).  Finally,

\[
\boxed{
E[|X||Y|]
=\frac4{\pi^2}\int_0^\infty\!\int_0^\infty
\frac{\mathcal I(t,s)}{t^2s^2}\,dt\,ds,
}
\]

where

\[
\mathcal I
=1-\operatorname{Re}\chi(t,0)-\operatorname{Re}\chi(0,s)
+\frac12\operatorname{Re}\{\chi(t,s)+\chi(t,-s)\}.
\]

Subtracting the two scalar reset means gives the exact AQ pair covariance.  This
oracle includes all equality sectors and every cumulant order of the common latent
quadratic law.

### Why this is not the dense carrier

For a generic collection \(\{A_i\}_{i=1}^n\), each pair uses a different symmetric
matrix pencil \(tA_i+sA_j\).  A dense panel therefore needs
\(n(n+1)/2\) determinant/resolvent families.  Simultaneous diagonalization reduces
each pencil evaluation to \(O(r)\), but generic noncommuting \(A_i\) do not admit it.
With no shared structure, direct evaluation is at least pairwise and carries
\(O(n^2r^3)\) algebra before quadrature.

Contracting with a generic endpoint mask \(B\) does not remove the obstruction:

\[
\langle B,C\rangle
=E[f(z)^{\mathsf T}Bf(z)]-E[f]^{\mathsf T}BE[f],
\qquad f_i=X_{i,+}.
\]

Even when \(B=bb^{\mathsf T}\), the integrand
\((\sum_i b_iX_{i,+})^2\) is not the positive part of one AQ scalar, so the one-scalar
determinant formula does not apply.  An exact reduction needs an additional shared
commuting dictionary, a bounded number of distinct AQ laws, or a certified sparse
set of exceptional pairs.

The practical symbolic use is therefore hybrid: extend the affine Mehler series
elementwise everywhere, use its endpoint-weighted tail receipt to identify the few
pairs (if any) that remain material, and invoke the bivariate oracle only on that
certified exception set.

### Graph node: AQ_BIVARIATE_CHARACTERISTIC_ORACLE

- status: exact_reference_sparse_fallback;
- result: boxed joint characteristic and positive-part moment identities;
- generic dense cost: no-go without shared commuting/low-dictionary structure;
- viable use: certified sparse exception set after `D10_MEHLER_ELEMENTWISE_EXTENSION`;
- tail caveat: determinant decay must be certified for the full pencil direction,
  including singular angular strata, not inferred from ranks of \(A_i,A_j\) separately.

## Exact rank-one counterexample: AQ fidelity has no universal positive floor

The local AQ projection is excellent near an affine centered seam, but no
distribution-free theorem can promote that observation to every quadratic
preactivation.  A one-dimensional pure quadratic already defeats any universal
lower bound.

Let

\[
g_t(X)=X^2-t^2,
\qquad f_t=(g_t)_+,
\qquad X\sim N(0,1),\quad t\ge0.
\]

Write \(Q_t=\bar\Phi(t)\), \(\phi_t=\phi(t)\).  Since \(f_t\) is even, its
\(\mathcal H_{\le2}\) projection has only the constant and \(H_2=X^2-1\)
coefficients.  Direct Gaussian tail integration gives

\[
c_0=E[f_t]
=2\{t\phi_t+(1-t^2)Q_t\},
\]

\[
c_2=E[f_tH_2]
=4(t\phi_t+Q_t),
\]

and

\[
D_t=E[f_t^2]
=2\{(-t^3+3t)\phi_t
+(t^4-2t^2+3)Q_t\}.
\]

Therefore the exact retained AQ fraction is

\[
\boxed{
q_{\rm AQ}^2(t)
=\frac{c_0^2+c_2^2/2}{D_t}.
}
\]

At \(t=1\), this is

\[
q_{\rm AQ}^2(1)
=\frac{[2\phi(1)]^2}{4[\phi(1)+Q_1]}
+\frac{4[\phi(1)+Q_1]^2}{2[\phi(1)+Q_1]}
=0.9474\ldots,
\]

already below the required \(0.95696\ldots\) local-energy proxy.  More sharply,
Mills expansion yields

\[
c_0\sim\frac{4\phi_t}{t},
\qquad c_2\sim4t\phi_t,
\qquad D_t\sim\frac{16\phi_t}{t},
\]

so

\[
\boxed{
q_{\rm AQ}^2(t)\sim\frac12t^3\phi(t)\longrightarrow0.
}
\]

Thus even rank-one \(A\) can put essentially all positive-part energy above chaos
two when the threshold selects a rare quadratic tail.  Spectral rank alone is not
a fidelity certificate, and adding only \(\Phi_3,\Phi_4\) cannot give an all-law
guarantee.

The minimal analytic assumption needed for a positive guarantee is a **threshold
regularity/mass condition**, e.g. a lower bound on \(E[g_+^2]\) together with a
non-negligible affine direction or a direct lower bound on
\(\|P_{\le2}g_+\|^2/E[g_+^2]\).  In the actual architecture this quantity is already
available from the exact convex reset potential:

\[
q^2=\frac{\|\nabla_{AQ}J(g)\|_{AQ}^2}{2J(g)}.
\]

It must be evaluated/certified per seam; it cannot be replaced by a universal
constant.

### Graph node: AQ_FIDELITY_RANK1_NOGO

- status: exact_counterexample;
- conclusion: no universal positive lower bound for AQ retained energy, even at rank one;
- obstruction: rare quadratic threshold, not high matrix rank;
- required gate: exact potential ratio \(\|\nabla J\|^2/(2J)\) or an explicit threshold-mass assumption;
- consequence: D14 construction remains conditional until seamwise fidelity or endpoint-weighted residual is certified.

## Finite conditional-direction fidelity theorem for an arbitrary AQ law

The rank-one no-go does not prevent a useful finite certificate.  An arbitrary AQ
preactivation can be split into one exactly integrated scalar quadratic plus a
conditionally centered transverse perturbation.  Convexity then gives a rigorous
lower bound on retained chaos-two energy for perturbations of any size.

Let

\[
g(z)=m+u^{\mathsf T}z
+\frac12(z^{\mathsf T}Az-\operatorname{tr}A).
\]

Choose a unit vector \(e\) (normally \(e=u/\|u\|\)), write
\(z=xe+y\), and decompose

\[
s=e^{\mathsf T}u,
\quad v=(I-ee^{\mathsf T})u,
\quad a=e^{\mathsf T}Ae,
\]

\[
w=(I-ee^{\mathsf T})Ae,
\quad B=(I-ee^{\mathsf T})A(I-ee^{\mathsf T}).
\]

Then

\[
g=g_{\parallel}(x)+h(x,y),
\]

\[
g_{\parallel}(x)
=m+sx+\frac a2(x^2-1),
\]

\[
h=v^{\mathsf T}y+xw^{\mathsf T}y
+\frac12(y^{\mathsf T}By-\operatorname{tr}B),
\]

with the exact identities

\[
E[h\mid x]=0,
\qquad
\tau^2:=E[h^2]
=\|v\|^2+\|w\|^2+\frac12\|B\|_F^2.
\]

Let \(f=g_+\), \(f_{\parallel}=(g_{\parallel})_+\), and use the nonnegative
quadratic test

\[
p_{\lambda}(x)=(x+\lambda)^2
\in\mathcal H_{\le2},
\qquad
\|p_\lambda\|_2^2=\lambda^4+6\lambda^2+3.
\]

Pointwise convexity gives

\[
(g_{\parallel}+h)_+
\ge (g_{\parallel})_+
+\mathbf 1_{g_{\parallel}>0}h.
\]

Multiplication by \(p_\lambda\ge0\), conditioning on \(x\), and Cauchy in the
Hermite space yield

\[
\|P_{\le2}f\|_2^2
\ge
\frac{\langle f_\parallel,p_\lambda\rangle^2}
{\lambda^4+6\lambda^2+3}.
\]

Also, \(q(r)=r_+^2/2\) has one-Lipschitz derivative, so the descent lemma gives

\[
(g_\parallel+h)_+^2
\le f_\parallel^2+2f_\parallel h+h^2.
\]

The cross term vanishes after conditioning.  Therefore

\[
\boxed{
\frac{\|P_{\le2}f\|_2^2}{\|f\|_2^2}
\ge
\frac{K_\parallel}{D_\parallel+\tau^2},
\qquad
K_\parallel
=\sup_{\lambda\in\mathbb R}
\frac{M_\lambda^2}{\lambda^4+6\lambda^2+3},
}
\]

where

\[
M_\lambda=E[f_\parallel(x)(x+\lambda)^2],
\qquad D_\parallel=E[f_\parallel^2].
\]

This theorem is finite; there is no small-\(A\) remainder.

### Closed one-dimensional evaluation

Put \(c=m-a/2\), so \(g_\parallel=(a/2)x^2+sx+c\), and let
\(S=\{x:g_\parallel(x)>0\}\).  Define

\[
I_k(S)=E[x^k\mathbf1_S],\qquad 0\le k\le4.
\]

Then

\[
\boxed{
\begin{aligned}
M_\lambda={}&
\left(\frac a2I_4+sI_3+cI_2\right)\\
&+2\lambda\left(\frac a2I_3+sI_2+cI_1\right)\\
&+\lambda^2\left(\frac a2I_2+sI_1+cI_0\right),
\end{aligned}}
\]

and

\[
\boxed{
D_\parallel
=\frac{a^2}{4}I_4+asI_3+(s^2+ac)I_2
+2scI_1+c^2I_0.
}
\]

If \(a\ne0\), the boundary roots are those of
\((a/2)x^2+sx+c=0\).  According to the sign of \(a\), \(S\) is the interval
between them or its two-tail complement; if the discriminant is negative, it is
all or empty.  Every \(I_k\) is explicit from

\[
J_0(t)=\bar\Phi(t),\quad J_1(t)=\phi(t),
\quad J_k(t)=t^{k-1}\phi(t)+(k-1)J_{k-2}(t).
\]

Thus the certificate requires only Gaussian cdf/pdf evaluations and a one-variable
quartic stationary equation for \(\lambda\), not a numerical model run.

### Affine-transverse specialization

When \(e=u/\|u\|\) and \(e^{\mathsf T}Ae=0\), put
\(\alpha=m/\|u\|\) and
\(t_A^2=\|A\|_F^2/(2\|u\|^2)\).  The bound simplifies to

\[
\frac{\|P_{\le2}f\|^2}{\|f\|^2}
\ge\frac{K_+(\alpha)}{d_\alpha+t_A^2},
\]

\[
d_\alpha=(1+\alpha^2)\Phi(\alpha)+\alpha\phi(\alpha),
\]

\[
K_+(\alpha)=\sup_\lambda
\frac{[a_0\lambda^2+b_0\lambda+c_0]^2}
{\lambda^4+6\lambda^2+3},
\]

\[
a_0=\phi(\alpha)+\alpha\Phi(\alpha),
\quad b_0=2\Phi(\alpha),
\quad c_0=2\phi(\alpha)+\alpha\Phi(\alpha).
\]

All finite stationary candidates solve

\[
\boxed{
\Phi(\alpha)\lambda^4
-[\phi(\alpha)+2\alpha\Phi(\alpha)]\lambda^3
+3\phi(\alpha)\lambda-3\Phi(\alpha)=0,
}
\]

with the \(|\lambda|\to\infty\) constant-test limit also checked.

### Endpoint-adjoint residual certificate

For a family of units, let

\[
\epsilon_i
=D_{\parallel,i}+\tau_i^2-K_{\parallel,i}.
\]

Since all \(R_i=P_{\le2}f_i\) lie in the same Gaussian chaos subspace, the exact
beyond-AQ covariance

\[
S_{ij}=\langle (I-P)f_i,(I-P)f_j\rangle
\]

is positive semidefinite and satisfies \(S_{ii}\le\epsilon_i\).  Hence for any
endpoint adjoint mask \(B\), entirely label-free,

\[
\boxed{
|\langle B,S\rangle|
\le \|B\|_{\rm op}\sum_i\epsilon_i
}
\]

and

\[
\boxed{
|\langle B,S\rangle|
\le\sum_{ij}|B_{ij}|\sqrt{\epsilon_i\epsilon_j}.
}
\]

This is the requested no-label \(\eta_B\) template.  The Bellman/K31 gate is
certified whenever the retained witness exceeds this residual radius plus the
required normalized margin \(1.691\times10^{-5}\).  If it fails, the minimal extra
observable is not a target label: it is a tighter diagonal residual energy
\(S_{ii}=2J_i-\|\nabla J_i\|^2\), which the exact AQ scalar potential already
provides.  Using that exact diagonal in the same PSD bounds strictly sharpens
\(\epsilon_i\).

### Graph node: AQ_FIDELITY_CONDITIONAL_1D_CERTIFICATE

- status: exact_finite_label_free_bound;
- scope: arbitrary affine-quadratic preactivation and arbitrary perturbation size;
- observable: one-dimensional truncated moments plus transverse energy \(\tau^2\);
- endpoint output: PSD residual radius \(\eta_B\) for any reverse mask;
- dependency: `AQ_RESET_CONVEX_MASTER_POTENTIAL` for the optional exact-diagonal sharpening;
- limitation: a loose endpoint \(\|B\|_{op}\) can still make the certificate inconclusive, but not invalid.

## Sharp target-free support from exact AQ residual diagonals

The scalar convex reset potential supplies more than a bound.  For each unit,

\[
d_i:=\|(I-P_{\le2})f_i\|_2^2
=2J_i-\|\nabla_{AQ}J_i\|_{AQ}^2
\]

is exact.  If no cross-residual observable is carried, the complete target-free
uncertainty set is at least bounded by the elliptope slice

\[
\mathcal E(d)=\{S\succeq0:\operatorname{diag}S=d\}.
\]

For an endpoint mask \(B\), its exact support relaxation is

\[
\boxed{
h_d(B)=\max_{S\in\mathcal E(d)}\langle B,S\rangle
=\min_{y\in\mathbb R^n}
\left\{y^{\mathsf T}d:\operatorname{Diag}(y)-B\succeq0\right\}.
}
\]

The unsigned no-label uncertainty radius is

\[
\boxed{
\eta_B(d)=\max\{h_d(B),h_d(-B)\}.
}
\]

The previously displayed operator and entrywise bounds are cheap feasible relaxations
of this SDP.  Additional known collision/old-span moments enter as linear constraints
on \(S\) and can only shrink \(\eta_B\).

This statement also gives the sharp impossibility boundary: diagonals alone cannot
produce a positive alignment lower bound with an arbitrary signed target mask.
Residual vectors with the same norms can be rotated so that their Gram matrix
approaches different exposed points of \(\mathcal E(d)\).  What the label-free state
can prove is an **uncertainty radius**, not a favorable sign.  A white-box witness
\(W_B\) is certified only through

\[
\boxed{
W_B-\eta_B(d)\ge 1.691\times10^{-5}
}
\]

in the common normalized units.  If this fails, the minimal extra observable is one
linear cross-residual functional \(\langle C,S\rangle\) chosen to cut the exposed SDP
direction for \(B\), not a generic amplitude statistic.

### Graph node: AQ_RESIDUAL_DIAGONAL_SDP

- status: exact_target_free_support_relaxation;
- inputs: exact diagonal rest energies from `AQ_RESET_CONVEX_MASTER_POTENTIAL`;
- output: \(\eta_B(d)\) by the displayed primal/dual SDP;
- no-go: diagonal energies alone imply no positive target alignment;
- next observable if inconclusive: one exposed-direction cross-residual measurement.

## Exact first-order true-minus-AQ K31 defect and a typed remainder

The residual covariance certificate must not be confused with the K31 cumulant
defect.  The latter has its own exact multilinear expansion.  This is the correct
novelty object after the same-site D11/Q24 scalar has been subtracted.

Let centered true variables be

\[
U=U_0+r,
\qquad V=V_0+s,
\]

where \(U_0,V_0\in\mathcal H_{\le2}\) are the AQ components and
\(r,s\perp\mathcal H_{\le2}\).  For

\[
\kappa_{31}(U,V)=\kappa_4(U,U,U,V),
\]

the exact first variation at \((U_0,V_0)\) is

\[
\boxed{
\delta\kappa_{31}^{(1)}
=\langle r,\psi_U\rangle+\langle s,\psi_V\rangle,
}
\]

\[
\psi_U
=3\{U_0^2V_0-2C_{UV}U_0-C_{UU}V_0\},
\qquad
\psi_V=U_0^3-3C_{UU}U_0.
\]

Here \(C_{UU}=E[U_0^2]\), \(C_{UV}=E[U_0V_0]\).  Because \(r,s\) are orthogonal
to chaos at most two, the linear counterterms vanish inside the pairings, and

\[
\delta\kappa_{31}^{(1)}
=3\langle r,P_{\ge3}(U_0^2V_0)\rangle
+\langle s,P_{\ge3}(U_0^3)\rangle.
\]

Thus the exact no-label first-order radius is

\[
\boxed{
|\delta\kappa_{31}^{(1)}|
\le
3\sqrt{d_U}\,\|P_{\ge3}(U_0^2V_0)\|_2
+\sqrt{d_V}\,\|P_{\ge3}(U_0^3)\|_2,
}
\]

where \(d_U=\|r\|_2^2,d_V=\|s\|_2^2\) come from the exact convex-potential
diagonals.  The two polynomial norms are finite Wick contractions of AQ data; no
unknown tensor is required to bound them.

The complete exact expansion is

\[
\begin{aligned}
\Delta\kappa_{31}={}&
3\kappa(r,U_0,U_0,V_0)+\kappa(U_0,U_0,U_0,s)\\
&+3\kappa(r,r,U_0,V_0)+3\kappa(r,U_0,U_0,s)\\
&+\kappa(r,r,r,V_0)+3\kappa(r,r,U_0,s)
+\kappa(r,r,r,s).
\end{aligned}
\]

This types every mismatch term and shows exactly where a mere residual-energy
receipt stops.  If

\[
a=\|r\|_4,
\quad b=\|s\|_4,
\quad U_4=\|U_0\|_4,
\quad V_4=\|V_0\|_4,
\]

then \(|\kappa_4(X_1,X_2,X_3,X_4)|\le4\prod_i\|X_i\|_4\) gives the safe nonlinear
remainder

\[
\boxed{
\begin{aligned}
|\Delta\kappa_{31}-\delta\kappa_{31}^{(1)}|
\le{}&12a^2U_4V_4+12aU_4^2b+4a^3V_4\\
&+12a^2U_4b+4a^3b.
\end{aligned}
}
\]

A conservative label-free \(L^4\) bound is

\[
\|r\|_4
\le\|g_+\|_4+\|P_{\le2}g_+\|_4
\le\|g\|_4+3\|P_{\le2}g_+\|_2,
\]

using degree-two Gaussian hypercontractivity.  If this is too loose, the minimal
additional observable is the **univariate fourth residual moment** \(E[r^4]\), not
the full cross-unit K31 tensor.  It can be obtained from the scalar AQ law or bounded
by a certified tail integral.

For an endpoint-weighted sum \(\sum_a\omega_a\kappa_{31,a}\), summing these radii
with \(|\omega_a|\) is safe; a residual-Gram SDP across the two influence families
is sharper.  Acquisition of the signed first-order defect itself needs only the two
surface functionals

\[
\langle r,P_{\ge3}(U_0^2V_0)\rangle,
\qquad
\langle s,P_{\ge3}(U_0^3)\rangle,
\]

which identifies the minimal novel moment interface outside old D11/Q24.

### Graph node: K31_TRUE_MINUS_AQ_DEFECT

- status: exact_first_variation_and_typed_remainder;
- old object excluded: same-site total AQ/D11 K31 scalar;
- novel source: two high-chaos surface pairings displayed above;
- label-free radius: exact \(L^2\) first-order bound plus \(L^4\) nonlinear remainder;
- minimal extra observable: signed surface pairings, and residual fourth moment only if the nonlinear radius is material.

## Exact AQ influence norms in matrix algebra only

The first-order K31 radius above needs two norms.  Both are computable exactly in
\(O(r^3)\) latent matrix algebra, without constructing an H3/H4 tensor.

Use centered half-quadratic variables

\[
U=u^{\mathsf T}z+\frac12(z^{\mathsf T}Az-\operatorname{tr}A),
\quad
V=v^{\mathsf T}z+\frac12(z^{\mathsf T}Bz-\operatorname{tr}B),
\]

and put

\[
K_2=E[U^2]=\|u\|^2+\frac12\operatorname{tr}A^2,
\quad
C=E[UV]=u^{\mathsf T}v+\frac12\operatorname{tr}AB.
\]

For \(F_V=U^3\), Gaussian integration by parts gives its chaos-zero/one/two
coefficients

\[
c_V=E[U^3]=\operatorname{tr}A^3+3u^{\mathsf T}Au,
\]

\[
\boxed{p_V=E[zU^3]=3K_2u+6A^2u,}
\]

\[
\boxed{
S_V=E[(zz^{\mathsf T}-I)U^3]
=6(A^3+Au u^{\mathsf T}+u u^{\mathsf T}A)+3K_2A.
}
\]

Consequently

\[
\boxed{
\|P_{\ge3}(U^3)\|_2^2
=E[U^6]-c_V^2-\|p_V\|^2-\frac12\|S_V\|_F^2.
}
\]

For \(F_U=U^2V\), the low coefficients are

\[
c_U=E[U^2V]
=\operatorname{tr}(A^2B)+2u^{\mathsf T}Av+u^{\mathsf T}Bu,
\]

\[
\boxed{
p_U
=2Cu+2A^2v+2ABu+K_2v+2BAu,
}
\]

and

\[
\boxed{
\begin{aligned}
S_U={}&2(uv^{\mathsf T}A+Avu^{\mathsf T}+ABA)
+2CA+K_2B\\
&+2(uu^{\mathsf T}B+Buu^{\mathsf T}
+Auv^{\mathsf T}+vu^{\mathsf T}A
+A^2B+BA^2).
\end{aligned}
}
\]

Thus

\[
\boxed{
\|P_{\ge3}(U^2V)\|_2^2
=E[U^4V^2]-c_U^2-\|p_U\|^2-\frac12\|S_U\|_F^2.
}
\]

The remaining sixth moments also need no tensor.  For any centered AQ scalar
\(W=(q,C_W)\),

\[
\kappa_k(W)
=\frac{(k-1)!}{2}\operatorname{tr}C_W^k
+\frac{k!}{2}q^{\mathsf T}C_W^{k-2}q,
\qquad k\ge2,
\]

and

\[
E[W^6]
=\kappa_6+15\kappa_4\kappa_2
+10\kappa_3^2+15\kappa_2^3.
\]

In particular,

\[
\boxed{
E[U^4V^2]
=\frac1{15}[x^2]\,E[(U+xV)^6],
}
\]

where the right side is obtained by substituting
\((u+xv,A+xB)\) into the two scalar cumulant formulas and retaining the \(x^2\)
coefficient.  Matrix polynomial products through degree six are \(O(r^3)\); the
ambient unit dimension and generic \(n^4\) diagram never appear.

These identities distinguish the two costs cleanly:

- **norm/certificate:** exact in latent \(O(r^3)\) algebra;
- **signed novel action:** still needs the two residual surface pairings, because
  their sign is not determined by AQ moments.

### Graph node: K31_AQ_INFLUENCE_NORMS

- status: exact_tensor_free_certificate;
- output: both high-chaos influence norms used by `K31_TRUE_MINUS_AQ_DEFECT`;
- complexity: \(O(r^3)\) per distinct AQ pair, no ambient H3/H4 tensor;
- no-go boundary: norms do not determine the signed residual alignment.

## Finite-chaos completeness of the linearized K31 defect

Although the ReLU residual \(r=(I-P_{\le2})g_+\) has infinitely many chaos orders,
its **first-order** K31 action does not.  The AQ influence polynomials

\[
U_0^2V_0,qquad U_0^3
\]

have total Gaussian degree at most six.  Orthogonality to chaos at most two gives

\[
\boxed{
\delta\kappa_{31}^{(1)}
=3\sum_{k=3}^{6}
\langle r_k,P_k(U_0^2V_0)\rangle
+\sum_{k=3}^{6}
\langle s_k,P_k(U_0^3)\rangle.
}
\]

Thus a complete linearized moment interface needs exactly four chaos bands,
\(H_3,H_4,H_5,H_6\), after literal old-span subtraction.  No \(H_{\ge7}\) source can
contribute at first order.  Conversely, an H3-only provider is not a complete K31
defect provider unless the H4--H6 pairings are proved old or dark.

The ownership audit is consequently:

- \(H_3\): affine collective source is plausible CP3 overlap but still awaits a
  literal seam diff; identity-specific/noncollective H3 can be novel;
- \(H_4\): centered affine atom is literally CP4-owned; noncommutative
  q211/q4-perp and CP4-dropped tail remain possible;
- \(H_5,H_6\): the affine Gaussian D10 baseline owns the corresponding finite
  Mehler covariance blocks, but primitive non-Gaussian source/action pairings are
  not implied by that covariance receipt;
- \(H_{\ge7}\): irrelevant to first variation, but can re-enter the nonlinear
  residual terms.

For any polynomial \(F\) in the AQ base, each band can be obtained locally by the
Stein coefficient

\[
T_k(F)=E[\nabla^kF(z)],\qquad 3\le k\le6,
\]

and contracted at birth against the corresponding source coefficient.  This is an
acquisition interface, not a demand to store the ambient tensors: topology-specific
source factorizations may evaluate the scalar contractions directly.

### Graph node: K31_LINEAR_DEFECT_H3_TO_H6

- status: exact_completeness_partition;
- theorem: only chaos orders 3--6 enter the true-minus-AQ K31 first variation;
- no-go: H3 alone is incomplete without literal H4--H6 ownership/darkness receipts;
- dependency: `K31_TRUE_MINUS_AQ_DEFECT`, CP3/CP4/D10 literal overlap nodes;
- next: audit primitive non-Gaussian H5/H6 action or prove its endpoint radius subcritical.

## Closed H5/H6 AQ query tensors for K31

The two highest bands in the preceding partition have especially simple query
geometry.  Let

\[
T_k(F)=E[H_k(z)F(z)]=E[\nabla^kF(z)]
\]

be the raw probabilists'-Hermite coefficient tensor, and let `Sym` mean averaged
full slot symmetrization.  Degree counting and the constant fifth/sixth derivatives
give

\[
\boxed{
T_6(U^3)=90\,\operatorname{Sym}(A\otimes A\otimes A),
}
\]

\[
\boxed{
T_5(U^3)=90\,\operatorname{Sym}(u\otimes A\otimes A),
}
\]

and

\[
\boxed{
T_6(U^2V)=90\,\operatorname{Sym}(A\otimes A\otimes B),
}
\]

\[
\boxed{
T_5(U^2V)
=30\,\operatorname{Sym}(v\otimes A\otimes A)
+60\,\operatorname{Sym}(u\otimes A\otimes B).
}
\]

The normalization is fixed by the scalar check
\(U=uH_1+(a/2)H_2\), \(V=vH_1+(b/2)H_2\): the \(H_6\) coefficient of
\(U^2V\) is \(90a^2b\), while its \(H_5\) coefficient is
\(30a^2v+60uab\).

Therefore the primitive non-Gaussian H5/H6 audit does not need to discover an
arbitrary target tensor.  It needs only source contractions against the four
polarized products above.  A source representation that can contract symmetric
matrix products directly can evaluate these scalar actions without materializing
\(r^5\) or \(r^6\) arrays.  Conversely, a covariance-only D10 receipt does not
determine these signed contractions.

### Graph node: K31_H5_H6_AQ_QUERIES

- status: exact_query_geometry;
- result: four boxed polarized query tensors with scalar-normalized constants;
- opportunity: topology-specific direct contraction against primitive H5/H6 source;
- no-go: covariance/energy ownership alone does not determine signed action.

## Exact H3 K31 influence action through the all-distinct scalar oracle

The H3 band admits a completely explicit topology-specific contraction.  For a
symmetric H3 source tensor \(T\), write

\[
\mathcal S_T(x,M)=\sum_{ijk}T_{ijk}x_iM_{jk}.
\]

With the half-quadratic AQ convention used throughout this lane, Gaussian product
calculus gives

\[
\boxed{
\begin{aligned}
\langle T,T_3(U^2V)\rangle={}&
6\mathcal S_T(v,uu^{\mathsf T})
+6\mathcal S_T(Bu,A)
+6\mathcal S_T(Au,B)\\
&+12\mathcal S_T(u,A\circ B)
+6\mathcal S_T(Av,A)
+6\mathcal S_T(v,A^2),
\end{aligned}}
\]

and

\[
\boxed{
\langle T,T_3(U^3)\rangle
=6\mathcal S_T(u,uu^{\mathsf T})
+18\mathcal S_T(Au,A)
+18\mathcal S_T(u,A^2).
}
\]

Here \(A\circ B=(AB+BA)/2\).  The constants are fixed by the one-dimensional
check: the raw \(H_3\) coefficient of \(U^2V\) is

\[
6u^2v+24uab+12a^2v,
\]

and that of \(U^3\) is \(6u^3+36ua^2\).

For the structured all-distinct ReLU birth source with rows \(c_i^{\mathsf T}\),
the independent source lane proved

\[
\begin{aligned}
\mathcal S_T(x,M)={}&\sum_i\beta_ix_i
\left[c_i^{\mathsf T}Mc_i-\sum_jc_{ij}^2M_{jj}\right]\\
&+2\sum_i\beta_i\left[
(c_i^{\mathsf T}x)\left(\sum_kc_{ik}M_{ik}\right)
-\sum_jc_{ij}^2x_jM_{ij}\right].
\end{aligned}
\]

Thus the complete H3 contribution to the linearized K31 defect is a fixed number of
scalar oracle calls and never requires an H3 tensor.  Calls sharing the same matrix
fuse by linearity; for \(U^2V\), the two \(M=A\) terms become
\(6\mathcal S_T(Bu+Av,A)\).  The \(M=uu^{\mathsf T}\) call is rank one and cheap.
The remaining generic query matrices are

\[
A,\quad B,\quad A\circ B,\quad A^2.
\]

The arithmetic boundary is exact: a generic source oracle needs
\(\operatorname{diag}(CMC^{\mathsf T})\), hence one \(CM\) product for each distinct
unreused query matrix.  It drops to \(O(n^2)\) only when that product is already
resident or \(M\) has a certified bounded-rank factorization.  Local AQ ownership
proves information completeness, not sub-GEMM arithmetic.

### Graph node: K31_H3_ALLDISTINCT_DIRECT_ACTION

- status: exact_tensor_free_scalar_action;
- output: boxed half-convention H3 contractions for both K31 influence slots;
- heavy query set: \(A,B,A\circ B,A^2\), modulo literal resident-product reuse;
- collision handling: inherited from the structured all-distinct oracle;
- no-go boundary: without resident products/bounded rank, each generic query costs one matrix product.

## Gaussian-deformation master potential: exact tensor-free H3--H6 actions

There is a stronger unification.  Every symmetric vector/matrix-product query of a
ReLU Hermite tensor is a directional derivative of one scalar Gaussian-deformed
reset mean.

Let a source preactivation be

\[
g(z)=m+h^{\mathsf T}z
+\frac12(z^{\mathsf T}Cz-\operatorname{tr}C),
\qquad f(z)=g(z)_+,
\]

and define

\[
\mathcal F(\mu,\Sigma)
=E_{Z\sim N(\mu,\Sigma)}[f(Z)].
\]

For a vector \(x\), write \(D_x=x^{\mathsf T}\nabla\); for a symmetric matrix
\(M\), write \(L_M=M:\nabla^2\).  Gaussian translation and the heat equation give
the exact identity

\[
\boxed{
D_\mu^pD_\Sigma^q\mathcal F
[x_1,\ldots,x_p;M_1,\ldots,M_q]
=2^{-q}E[D_{x_1}\cdots D_{x_p}
L_{M_1}\cdots L_{M_q}f(Z)].
}
\]

Since \(T_k(f)=E[\nabla^kf]\), the right side is precisely the contraction of
\(T_{p+2q}(f)\) against

\[
\operatorname{Sym}(x_1\otimes\cdots\otimes x_p
\otimes M_1\otimes\cdots\otimes M_q).
\]

No Hermite tensor is constructed.

### The scalar potential is already an AQ characteristic integral

If \(Z=\mu+L\xi\), \(LL^{\mathsf T}=\Sigma\), then the same source law becomes

\[
g(Z)=m' + h'^{\mathsf T}\xi
+\frac12(\xi^{\mathsf T}C'\xi-\operatorname{tr}C'),
\]

with

\[
\boxed{
\begin{aligned}
m'={}&m+h^{\mathsf T}\mu+\frac12\mu^{\mathsf T}C\mu
+\frac12\operatorname{tr}[C(\Sigma-I)],\\
h'={}&L^{\mathsf T}(h+C\mu),\\
C'={}&L^{\mathsf T}CL.
\end{aligned}}
\]

Therefore \(\mathcal F(\mu,\Sigma)\) is exactly the scalar AQ reset mean already
given by the determinant/resolvent characteristic formula.  Its directional
derivatives inherit the differentiated tail certificates in
`AQ_RESET_CHARACTERISTIC_TAIL`.

### Direct primitive H5/H6 K31 actions

Combining the deformation identity with `K31_H5_H6_AQ_QUERIES` gives the complete
signed source actions

\[
\boxed{
\langle T_6(f),T_6(U^3)/90\rangle
=8D_\Sigma^3\mathcal F[A,A,A],
}
\]

\[
\boxed{
\langle T_6(f),T_6(U^2V)/90\rangle
=8D_\Sigma^3\mathcal F[A,A,B],
}
\]

and

\[
\boxed{
\langle T_5(f),T_5(U^3)/90\rangle
=4D_\mu D_\Sigma^2\mathcal F[u;A,A],
}
\]

\[
\boxed{
\begin{aligned}
\langle T_5(f),T_5(U^2V)\rangle={}&
120D_\mu D_\Sigma^2\mathcal F[v;A,A]\\
&+240D_\mu D_\Sigma^2\mathcal F[u;A,B].
\end{aligned}}
\]

Equivalently, restoring the coefficients in the first three displays gives
\(720D_\Sigma^3\mathcal F[A,A,A]\),
\(720D_\Sigma^3\mathcal F[A,A,B]\), and
\(360D_\mu D_\Sigma^2\mathcal F[u;A,A]\), respectively.

The same identity recovers the H3 surface oracle as

\[
\boxed{
\mathcal S_{T_3(f)}(x,M)
=2D_\mu D_\Sigma\mathcal F[x;M],
}
\]

and supplies every H4 vector-vector/matrix-matrix polarization by the corresponding
\((p,q)=(2,1)\) or \((0,2)\) derivative.

### Cost and novelty boundary

For latent rank \(r\), a source unit needs the same determinant/resolvent family as
the exact AQ reset.  Once that factorization is resident, a fixed number of
directional derivatives uses matrix traces/solves in \(O(r^3)\) or less; it never
scales as \(r^5,r^6\) or ambient \(n^4\).  Directions sharing the same
\(A,B,u,v\) reuse the resolvent.

This is an exact **signed** primitive high-chaos oracle for the AQ-induced ReLU
source.  It still obeys three binding subtractions:

1. remove literal CP3/CP4/D10/Q24 same-site action once;
2. use the positive-depth Bellman query, not the duplicate same-site total K31 mask;
3. for a co-adapted finite correction, vary both source parameters and the reverse
   query; a frozen derivative supplies only the fixed-tangent action.

It does not generate a truth-minus-deployed source unavailable from white-box AQ
parameters, and it does not by itself bound the nonlinear-in-residual K31 remainder.

### Graph node: AQ_GAUSSIAN_DEFORMATION_H3_H6_ORACLE

- status: exact_tensor_free_signed_action;
- master object: scalar \(\mathcal F(\mu,\Sigma)=E_{N(\mu,\Sigma)}g_+\);
- covers: all polarized H3, H4, H5, H6 source/query contractions at first order;
- complexity: fixed directional derivatives of the resident AQ resolvent, \(O(r^3)\) per source family;
- dependencies: positive-depth costate, literal old-span subtraction, differentiated-tail receipt;
- no-go boundary: unavailable truth-frame source and nonlinear residual diagrams remain external.

## Exact univariate residual fourth moment from a tilted AQ potential

The nonlinear K31 radius in `K31_TRUE_MINUS_AQ_DEFECT` need not use the very loose
hypercontractive bound.  The exact fourth moment of one AQ projection residual is
also a scalar-potential derivative.

Let

\[
f(z)=g(z)_+,
\qquad
R(z)=P_{\le2}f
=c+a^{\mathsf T}z+\frac12(z^{\mathsf T}Bz-\operatorname{tr}B),
\qquad r=f-R.
\]

For \(p=0,1,2,3,4\), define

\[
G_p(\eta)=E[f(z)^p e^{\eta R(z)}],
\qquad f^0:=1.
\]

Then

\[
G_p^{(j)}(0)=E[f^pR^j]
\]

and binomial expansion gives the exact identity

\[
\boxed{
E[r^4]
=\sum_{j=0}^{4}(-1)^j{4\choose j}
G_{4-j}^{(j)}(0).
}
\]

Each \(G_p\) is still a one-scalar AQ truncated-moment potential.  Indeed, for
\(I-\eta B\succ0\), put

\[
\Sigma_\eta=(I-\eta B)^{-1},
\qquad
\mu_\eta=\eta\Sigma_\eta a,
\]

\[
\boxed{
Z_R(\eta)
=e^{\eta(c-\operatorname{tr}B/2)}
\det(I-\eta B)^{-1/2}
\exp\!\left[
\frac{\eta^2}{2}a^{\mathsf T}\Sigma_\eta a
\right].
}
\]

Completing the Gaussian square yields

\[
\boxed{
G_p(\eta)
=Z_R(\eta)
E_{Z\sim N(\mu_\eta,\Sigma_\eta)}[g(Z)_+^p].
}
\]

Under this shifted covariance, \(g\) remains an affine-quadratic scalar with the
transformed parameters already displayed in
`AQ_GAUSSIAN_DEFORMATION_H3_H6_ORACLE`.  Its positive truncated moments for
\(p\le4\) follow from the same one-dimensional characteristic law; equivalently,

\[
E[X_+^p]=p\int_0^\infty t^{p-1}P(X>t)\,dt
\]

with the cdf obtained by Fourier inversion and the determinant tail certified.

Therefore \(\|r\|_4=E[r^4]^{1/4}\) is an exact label-free scalar receipt.  Applying
it separately to the \(U\)- and \(V\)-slot residuals makes every term in the
nonlinear K31 remainder bound explicit from white-box AQ parameters.  No cross-unit
fourth tensor is required merely to certify the remainder magnitude.

This does not determine its sign.  If the certified magnitude is too large, the
minimal next observable remains a signed residual-product action, not another
univariate moment.

### Graph node: AQ_RESIDUAL_L4_TILTED_POTENTIAL

- status: exact_univariate_magnitude_receipt;
- output: exact \(E[(g_+-P_{\le2}g_+)^4]\);
- method: four derivatives of tilted scalar AQ truncated-moment potentials;
- use: closes the label-free nonlinear radius in `K31_TRUE_MINUS_AQ_DEFECT`;
- limitation: magnitude only; signed multi-residual alignment remains external.

## Constructive sequence: AQ state + reverse path-integral high-chaos correction

The preceding identities assemble into one coherent tensor-free architecture rather
than a disconnected feature list.

Consider a one-parameter family of layer maps

\[
x_{\ell+1}(t)=F_\ell(x_\ell(t),t),
\qquad 0\le t\le1,
\]

interpolating from the incumbent AQ/D10/D11 closure at \(t=0\) to the proposed
high-chaos-corrected closure at \(t=1\).  Let \(Y(t)\) be the scalar endpoint and
define

\[
J_\ell(t)=D_xF_\ell(x_\ell(t),t),
\qquad
\lambda_L(t)=\nabla Y(x_L(t)),
\qquad
\lambda_\ell(t)=J_\ell(t)^{\mathsf T}\lambda_{\ell+1}(t).
\]

Differentiating the recurrence and applying the discrete Green identity gives

\[
\boxed{
\frac{dY}{dt}
=\sum_{\ell}
\left\langle\lambda_{\ell+1}(t),
\partial_tF_\ell(x_\ell(t),t)\right\rangle.
}
\]

Therefore the co-adapted finite correction is exactly

\[
\boxed{
Y(1)-Y(0)
=\int_0^1\sum_\ell
\langle\lambda_{\ell+1}(t),s_\ell(t)\rangle\,dt,
\qquad
s_\ell=\partial_tF_\ell.
}
\]

At fixed \(t\), **if the complete sewn Bellman pullback at that reset is represented
in the AQ q\(\le4\) product-query family**, its source bands H3--H6 are exactly the
Gaussian-deformation derivatives of
`AQ_GAUSSIAN_DEFORMATION_H3_H6_ORACLE`.  Under that representation neither a
forward H3/H4/H5/H6 tensor nor a frozen-query approximation is required: the reverse
path integral varies source and costate together.  The representability premise is
binding.  A pullback containing an uncompiled future gate/indicator or a trajectory
component outside the sewn AQ costate is not supplied by the scalar jet.

### Minimal coupled state and once-only ownership

The sequence is:

1. carry the D14 AQ coordinates \((m,u,A)\);
2. replace the affine Gaussian covariance block by the exact bivariate D10 kernel
   (or certified Q7/Q8 tail extension);
3. obtain \(\Phi_3,\Phi_4\) and their K31/K22/q211 polarizations from the AQ scalar
   cumulant potential;
4. run one reverse endpoint costate through the coupled state;
5. at each reset, consume only the complete-old-projected H3--H6 source action via
   Gaussian-deformation derivatives;
6. integrate the source/costate pairing along the closure path if the finite secant,
   rather than the frozen tangent, is needed.

The complete-old projection must subtract D10 finite Gaussian response, CP3, CP4,
star/H31, Q24, and every same-site total K31 action before the positive-depth scalar
is credited.

### What this proves and what it does not

It proves a tensor-free **local mechanism** for the complete linearized q\(\le4\)
source action and an exact path identity for co-adaptation inside the chosen closure
family, conditional on complete sewn-costate representability at every source seam.
It does not prove that the deployed AQ path equals the truth-frame path.  The gap is
now isolated to three scalar certificates:

\[
\text{(i) old-residualized endpoint witness},
\quad
\text{(ii) beyond-AQ uncertainty }\eta_B,
\quad
\text{(iii) path/source provenance error}.
\]

The fire condition is the common-unit inequality

\[
\boxed{
W_{\rm path}-\eta_B-\eta_{\rm provenance}
\ge 1.691\times10^{-5}
}
\]

plus the master raw-value and adjusted-cost gate.  The exact residual diagonal SDP,
tilted L4 receipt, and differentiated characteristic tails make \(\eta_B\) fully
label-free.  If the inequality fails, no amount of provider amplitude rhetoric can
rescue the lane without one new signed cross observable.

### Graph node: AQ_REVERSE_PATH_INTEGRAL_CLOSURE

- status: exact_mechanism_conditional_value;
- provider: D14 AQ state + exact scalar Gaussian-deformation source actions;
- transport: one reverse Bellman costate, no forward high-chaos tensor;
- co-adaptation: exact source/costate path integral;
- open gates: deploy-vs-truth provenance, old-residualized witness size, adjusted arithmetic;
- additional open gate: complete positive-depth Bellman pullback lies in the sewn AQ
  product-query family (otherwise a bivariate/future-gate oracle is needed);
- terminal fail-fast: boxed witness-minus-radii inequality and master raw frontier.

## Arithmetic scale of the deformation oracle

The oracle's latent arithmetic is small enough to audit seriously.  With \(n\) source
units and latent rank \(r\), one dense \(r\times r\) factorization/contraction per
unit costs

\[
\boxed{
\frac{nr^3}{n^3}=\frac{r^3}{n^2}
}
\]

ambient GEMM-equivalents per layer.  At \(n=1024\), the exact ratios are

\[
\frac{32^3}{1024^2}=0.03125,
\quad
\frac{45^3}{1024^2}=0.08690\ldots,
\quad
\frac{55^3}{1024^2}=0.15867\ldots .
\]

Across fourteen r32 layers this is \(0.4375\) per independent derivative family;
across ten layers it is \(0.8690\) at r45 and \(1.5867\) at r55.  The complete
K31 H5/H6 query uses only the two primitive matrix directions \(A,B\):
\(A^2\) and \(A\circ B\) are formed after transforming \(A,B\) into the resident
source eigenbasis.  Vector directions cost only \(O(r^2)\).

Hence even four independent cubic directional contractions have latent leading
scales about \(1.75\), \(3.48\), and \(6.35\) GEMM-equivalents over the respective
r32/r45/r55 depth schedules, before constant factors.  The eigenfactorization used
by the scalar AQ reset and all derivative directions must be shared.  A separate
factorization or quadrature-matrix multiply per direction would multiply this cost
and is not authorized by the identity.

The binding implementation receipt is therefore a constant-factor census:

\[
C_{\rm oracle}
=c_{\rm eig}\,Lr^3/n^2
+c_{\rm dir}\,Lr^3/n^2
+O(Lr^2/n^2),
\]

with literal reuse of the reset eigensystem/resolvent.  This is a much narrower
question than “can an H6 tensor fit?”

### Graph node: AQ_DEFORMATION_ORACLE_COST

- status: symbolic_cost_scale_solved;
- leading unit: \(Lr^3/n^2\) per factorization/directional family;
- favorable branch: shared AQ eigensystem and only primitive directions \(A,B\);
- kill: any implementation that refactorizes per derivative direction or per Hermite band;
- open: literal constant-factor/billing census against the chosen r32/r45/r55 carrier.

## Closed H4 AQ query geometry and deformation action

For completeness, the H4 band also has a two-matrix form.  With
\(q_U=u+Az\), the fourth derivative of \(U^3\) is

\[
\nabla^4(U^3)
=36\operatorname{Sym}(q_U\otimes q_U\otimes A)
+18U\operatorname{Sym}(A\otimes A).
\]

Taking expectation and using \(E[U]=0\), \(E[q_Uq_U^{\mathsf T}]=uu^{\mathsf T}+A^2\),
gives

\[
\boxed{
T_4(U^3)
=36\operatorname{Sym}[(uu^{\mathsf T}+A^2)\otimes A].
}
\]

Polarizing \(U\mapsto U+tV\) and taking one third of the linear coefficient yields

\[
\boxed{
\begin{aligned}
T_4(U^2V)=12\operatorname{Sym}\{&
[uv^{\mathsf T}+vu^{\mathsf T}+AB+BA]\otimes A\\
&+[uu^{\mathsf T}+A^2]\otimes B\}.
\end{aligned}
}
\]

The scalar checks are

\[
T_4(U^3)=36a(u^2+a^2),
\]

\[
T_4(U^2V)=24uva+12u^2b+36a^2b.
\]

Against a source \(f=g_+\), the exact signed actions are therefore

\[
\boxed{
\langle T_4(f),T_4(U^3)\rangle
=144D_\Sigma^2\mathcal F[uu^{\mathsf T}+A^2,A],
}
\]

and

\[
\boxed{
\begin{aligned}
\langle T_4(f),T_4(U^2V)\rangle
=48D_\Sigma^2\mathcal F[&uv^{\mathsf T}+vu^{\mathsf T}+AB+BA,A]\\
+48D_\Sigma^2\mathcal F[&uu^{\mathsf T}+A^2,B].
\end{aligned}
}
\]

This includes affine and noncommutative H4 geometry.  Literal CP4/star/Q24 action
must be subtracted; the remainder supplies precisely the identity-specific H4
candidate rather than a duplicate affine atom.

### Graph node: K31_H4_AQ_QUERY_DEFORMATION

- status: exact_tensor_free_signed_action;
- query rank: two symmetric matrix directions per term;
- source oracle: second covariance derivative of \(\mathcal F\);
- novelty: only after CP4/star/Q24 and collision/Schur subtraction;
- link: explicit H4 specialization of `AQ_GAUSSIAN_DEFORMATION_H3_H6_ORACLE`.

## Complete H3--H6 K31 action from one tilted residual generator

The bandwise formulas admit an even simpler exact representation.  Let a source
ReLU output and its AQ projection be

\[
f=g_s(z)_+,
\qquad R=P_{\le2}f,
\qquad r=f-R.
\]

Let the centered AQ consumer variables be

\[
U=u^{\mathsf T}z+\frac12(z^{\mathsf T}Az-\operatorname{tr}A),
\quad
V=v^{\mathsf T}z+\frac12(z^{\mathsf T}Bz-\operatorname{tr}B).
\]

Define the two-parameter residual generator

\[
\boxed{
\mathcal H_r(\alpha,\beta)
=E[(f-R)e^{\alpha U+\beta V}].
}
\]

Projection orthogonality implies that the value and first derivatives vanish at the
origin:

\[
\mathcal H_r(0,0)=\partial_\alpha\mathcal H_r(0,0)
=\partial_\beta\mathcal H_r(0,0)=0.
\]

Second parameter derivatives need **not** vanish, because \(U^2,UV,V^2\) contain
chaos three and four; they are precisely lower equality-sector response and must be
old-span audited.  The complete signed one-residual K31 action is

\[
\boxed{
3\langle r,U^2V\rangle
=3\,\partial_\alpha^2\partial_\beta
\mathcal H_r(0,0).
}
\]

For the residual \(s=f_V-R_V\) in the fourth slot, define the analogous generator
\(\mathcal H_s\).  Then the entire first-order true-minus-AQ defect is

\[
\boxed{
\delta\kappa_{31}^{(1)}
=3\partial_\alpha^2\partial_\beta\mathcal H_r(0,0)
+\partial_\alpha^3\mathcal H_s(0,0).
}
\]

The cumulant counterterms need not be added: they pair only with chaos at most two
and vanish against \(r,s\).

### Exact scalar evaluation

Put

\[
D_{\alpha\beta}=\alpha A+\beta B,
\qquad
q_{\alpha\beta}=\alpha u+\beta v,
\]

\[
\Sigma_{\alpha\beta}=(I-D_{\alpha\beta})^{-1},
\qquad
\mu_{\alpha\beta}=\Sigma_{\alpha\beta}q_{\alpha\beta}.
\]

Completing the Gaussian square gives

\[
\boxed{
Z_{UV}(\alpha,\beta)
=e^{-\operatorname{tr}D_{\alpha\beta}/2}
\det(I-D_{\alpha\beta})^{-1/2}
\exp\!\left[
\frac12q_{\alpha\beta}^{\mathsf T}
\Sigma_{\alpha\beta}q_{\alpha\beta}
\right].
}
\]

Hence

\[
E[f e^{\alpha U+\beta V}]
=Z_{UV}(\alpha,\beta)
E_{N(\mu_{\alpha\beta},\Sigma_{\alpha\beta})}[g_s(Z)_+],
\]

which is exactly the scalar Gaussian-deformation AQ mean.  The projection term is
also closed:

\[
E[R e^{\alpha U+\beta V}]
=Z_{UV}\left[
c_R+a_R^{\mathsf T}\mu
+\frac12\{\mu^{\mathsf T}B_R\mu
+\operatorname{tr}[B_R(\Sigma-I)]\}
\right].
\]

Subtracting these two expressions and differentiating only to total order three
returns the complete H3+H4+H5+H6 action automatically, with every collision and
Hermite normalization consistent.  The explicit band formulas above are now
independent audits and ownership splitters, not separate deploy computations.

### Why this is the minimal acquisition interface

The generator uses only:

- source AQ parameters \((m_s,h_s,C_s)\);
- source projection parameters \((c_R,a_R,B_R)\);
- the two consumer directions \((u,A),(v,B)\);
- one scalar AQ reset potential and its third directional jet.

It is label-free and signed.  It computes the exact AQ-induced high-chaos residual
action that a repeated-plane carrier was trying to approximate.  After complete
D11/Q24/CP3/CP4/star subtraction, any surviving positive-depth value is genuinely
new.

It remains a first-order-in-residual identity.  Multi-residual terms require the
bivariate residual-product generator or the exact L4 radius already derived.

### Graph node: K31_TILTED_RESIDUAL_GENERATOR

- status: exact_complete_first_order_action;
- formula: two third derivatives of one/two scalar tilted AQ reset means;
- replaces: separate H3/H4/H5/H6 tensor acquisition at deployment;
- old-span discipline: subtract assembled same-site action after exact generator evaluation;
- open: positive-depth witness size and nonlinear residual-product remainder.

## Universal one-residual connected generator for every q3/q4 equality sector

The tilted identity is not specific to K31.  Let

\[
X_j=u_j^{\mathsf T}z
+\frac12(z^{\mathsf T}A_jz-\operatorname{tr}A_j),
\qquad j=1,\ldots,k,
\]

and let \(r=f-P_{\le2}f\).  Define

\[
\mathcal H_r(t_1,\ldots,t_k)
=E\left[r\exp\left(\sum_jt_jX_j\right)\right].
\]

Because \(r\) is centered and \(E[rX_j]=0\), every Wick/cumulant partition that
pairs \(r\) with one AQ argument vanishes.  Therefore, for \(k=2,3\),

\[
\boxed{
\kappa_{k+1}(r,X_1,\ldots,X_k)
=E[rX_1\cdots X_k]
=\partial_{t_1}\cdots\partial_{t_k}\mathcal H_r(0).
}
\]

The same Gaussian completion evaluates \(\mathcal H_r\) for any number of query
directions using

\[
D(t)=\sum_jt_jA_j,qquad q(t)=\sum_jt_ju_j,qquad
\Sigma=(I-D)^{-1},\quad\mu=\Sigma q.
\]

Thus every **one-residual** q3/q4 equality-sector action is a second or third scalar
jet:

### q3(2,1)

For \(U=U_0+r_U\), \(V=V_0+r_V\),

\[
\boxed{
\delta\kappa_{21}^{(1)}
=2\,\partial_U\partial_V\mathcal H_{r_U}(0)
+\partial_U^2\mathcal H_{r_V}(0).
}
\]

### q4(3,1)

\[
\boxed{
\delta\kappa_{31}^{(1)}
=3\,\partial_U^2\partial_V\mathcal H_{r_U}(0)
+\partial_U^3\mathcal H_{r_V}(0).
}
\]

### q4(2,2)

\[
\boxed{
\delta\kappa_{22}^{(1)}
=2\,\partial_U\partial_V^2\mathcal H_{r_U}(0)
+2\,\partial_U^2\partial_V\mathcal H_{r_V}(0).
}
\]

### q4(2,1,1)

For repeated slot \(i\) and singleton slots \(j,k\),

\[
\boxed{
\begin{aligned}
\delta\kappa_{211}^{(1)}={}&
2\,\partial_i\partial_j\partial_k\mathcal H_{r_i}(0)\\
&+\partial_i^2\partial_k\mathcal H_{r_j}(0)
+\partial_i^2\partial_j\mathcal H_{r_k}(0).
\end{aligned}
}
\]

### q4(1,1,1,1)

\[
\boxed{
\delta\kappa_{1111}^{(1)}
=\sum_{a=1}^{4}
\prod_{b\ne a}\partial_b\,\mathcal H_{r_a}(0).
}
\]

Repeated derivatives automatically enforce the 31/22/211 multiplicities.  No
explicit collision enumeration, q4 tensor, or generic K22 four-index contraction is
needed at acquisition time.  Literal old-span subtraction is still sector-specific:
CP3/Q24 for q3, CP4/star/H31/Q24 for q4, and the complete q211/q1111 Schur/collision
projection before novelty credit.

This theorem is exactly the requested label-free **local/decomposable-costate**
source feature stack spanning the complete one-residual linearized q\(\le4\)
correction.  Endpoint aggregation is cheap only when the complete-old-projected
costate has bounded Waring/polarization rank.  A generic mask may require one scalar
jet family per supported source/query pair.  Independently, terms with two or more
residual slots are not generated by \(\mathcal H_r\); they require mixed
residual-product potentials or the certified nonlinear radius.

### Graph node: AQ_ONE_RESIDUAL_Q3_Q4_GENERATOR

- status: exact_complete_local_linearized_q_stack;
- covers: q3(21), q4(31), q4(22), q4(211), q4(1111);
- interface: second/third jets of scalar tilted residual means, multiplied by the
  endpoint costate decomposition rank;
- multiplicities/collisions: repeated derivatives exact; old-span collision subtraction remains binding;
- no-go boundary: two-or-more-residual diagrams.

## Residual-product hierarchy and the generic K22/nonlinear cost boundary

The remaining diagrams also have an exact scalar hierarchy, but its arity explains
the cost no-go.

For residual sources \(r_a\), define

\[
\mathcal H_{a_1\ldots a_p}(t)
=E\left[r_{a_1}\cdots r_{a_p}
\exp\left(\sum_jt_jX_j\right)\right].
\]

A q4 term with \(p\) residual slots and \(4-p\) AQ slots is a
\((4-p)\)-th derivative of this generator, followed by the ordinary connected
pairing subtraction.  Thus the hierarchy is exact.

### One distinct residual law: still cheap

If all residual slots are powers of the same \(r=f-R\), expand \(r^p\) binomially.
Every term is

\[
E[f^aR^b e^{\sum t_jX_j}],
\]

which is a derivative of the univariate tilted truncated-moment potential in
`AQ_RESIDUAL_L4_TILTED_POTENTIAL`.  Therefore same-source terms such as

\[
\kappa(r,r,U,V),qquad \kappa(r,r,r,V)
\]

remain tensor-free and univariate.

### Two distinct residual laws: bivariate AQ oracle

Terms such as

\[
\kappa(r,U,U,s),
\quad \kappa(r,r,U,s),
\quad \kappa(r,r,r,s)
\]

need joint truncated moments of two AQ ReLU sources after the common Gaussian query
tilt.  They are exact through the two-quadratic characteristic pencil in
`AQ_BIVARIATE_CHARACTERISTIC_ORACLE`, but generic evaluation is pair-specific.

For an endpoint with a full-rank K22/K31 pair mask, this reintroduces the
\(n^2\)-family barrier.  It becomes viable only if:

- the residual pair is the same source or a deterministic/JVP derivative of it;
- the endpoint pair mask has certified bounded rank/support;
- the two AQ matrix laws share a commuting dictionary;
- or the L4 remainder radius is already below the value tolerance.

### Three/four distinct residual laws

These require trivariate/four-variate positive-part AQ moments.  No one-scalar
Gaussian completion exists, and generic all-distinct q1111 acquisition returns to
the forbidden high-arity family.  At deployed width the only lawful route is a
factorized source identity, a very low-dimensional endpoint mask, or a magnitude
closure by Holder/SDP bounds.

This proves the sharp boundary:

\[
\boxed{
\text{one-residual q3/q4 is locally scalar and cheap for a bounded-rank costate;}
\quad
\text{generic endpoint masks or multi-residual q4 are not.}
}
\]

The result also clarifies the generic K22 no-go.  The **linearized** K22 action is
cheap via `AQ_ONE_RESIDUAL_Q3_Q4_GENERATOR`; the obstruction begins at the cross
two-residual K22 sector, not at the one-residual equality pattern itself.

### Graph node: AQ_RESIDUAL_PRODUCT_ARITY_BOUNDARY

- status: exact_hierarchy_and_cost_nogo;
- cheap: any number of powers of one residual law plus a bounded number of
  decomposable AQ query directions;
- conditional: two distinct residual laws via bivariate AQ characteristic oracle;
- generic no-go: three/four distinct laws or full-rank pair masks without structure;
- consequence: linearized K22 is locally open to scalar jets, but a generic
  full-rank endpoint mask can still create an \(\Omega(n)\) query-family barrier;
  nonlinear cross-residual K22 adds a separate arity barrier.

## q211 cross-lane audit: local-source GO, transport/additive NO

The H4 reverse audit confirms the exact scope of the Gaussian-deformation result.
Any local one-residual H211 query is a linear combination of

\[
\operatorname{Sym}(R\otimes S),
\]

so its source action is exactly

\[
4D_\Sigma^2\mathcal F[R,S]
\]

in the half-quadratic convention.  After sewing and subtracting Q22/Q31/q4
collision traces, the remaining Hessian action is precisely the q4-perp commutator
sector.  Thus moment acquisition for the local q211 source is solved.

Three limitations are binding:

1. the complete sewn positive-depth q211 Bellman costate must already be available;
   the local scalar jet does not propagate or invent the D10 future query;
2. literal zero-cost reuse requires covariance directions such as
   \(A^2,AB+BA\) to be views/reductions of the resident full product
   \(P=M_iN_a\) together with the needed original-index collision summary
   \(D_{ab}\); \(\operatorname{skew}P\) alone is not a covariance direction;
3. current D11 lacks the AQ state/eigensystem, so this is a replacement-state
   mechanism, not an additive patch to D11.

Accordingly the correct disposition is:

\[
\boxed{
\text{local q211 moment acquisition: GO;}
\quad
\text{future cyclic closure/costate provenance: OPEN;}
\quad
\text{additive D11 patch: NO.}
}
\]

### Graph node: AQ_Q211_LOCAL_GO_TRANSPORT_OPEN

- status: solved_local_action_open_transport;
- source: \(4D_\Sigma^2\mathcal F[R,S]\) after complete collision/old subtraction;
- dependency: sewn q211 Bellman costate and resident full-product/collision receipts;
- architecture: D14/AQ replacement state, not additive D11 feature.

## Tilt-costate rank gate: why local scalar closure need not survive depth

The scalar generator is constant-cost only for a bounded number of polarized query
terms.  This condition has a sharp q4 rank certificate.

For a symmetric H4 query tensor \(Q\), define its pair flattening as the self-adjoint
operator on \(\operatorname{Sym}_r\)

\[
\langle M,\mathscr F_Q(N)\rangle_F
=\langle Q,\operatorname{Sym}(M\otimes N)\rangle.
\]

One covariance-Hessian jet

\[
D_\Sigma^2\mathcal F[M,N]
\]

corresponds to a symmetrized rank-one bilinear form on this pair space, hence has
pair-flattening rank at most two.  If a sewn Bellman q4 costate decomposes into
\(R\) such scalar jets, necessarily

\[
\boxed{
R\ge\frac12\operatorname{rank}\mathscr F_Q.
}
\]

Since

\[
d=\dim\operatorname{Sym}_r=\frac{r(r+1)}2,
\]

a generic q4 costate has rank \(d\), forcing \(R\ge d/2\).  Thus a generic
positive-depth costate cannot be consumed by a bounded number of scalar jets.  The
local K31 and q211 queries pass because their explicit pair flattenings have fixed
polarization rank; a dense future pullback may not.

The gate has three exact escape routes:

1. the actual task-weighted costate has low pair rank after complete old-span
   residualization;
2. a resident full-product/Hessian operator applies the complete costate in one
   shared contraction, so arithmetic does not scale with polarization rank;
3. the discarded pair-rank tail has an endpoint support receipt below the master
   tolerance.

Without one of these, `AQ_GAUSSIAN_DEFORMATION_H3_H6_ORACLE` is a local identity,
not a depth carrier.

### Graph node: AQ_TILT_COSTATE_PAIR_RANK_GATE

- status: exact_necessary_rank_bound;
- certificate: \(R\ge\operatorname{rank}(\mathscr F_Q)/2\);
- local q4: bounded-rank pass;
- generic sewn q4: \(R=\Omega(r^2)\) no-go;
- escapes: task-low-rank, resident full-operator reuse, or certified tail.

## Observation-rank theorem: depth itself does not increase local query rank

The pair-rank warning has one important exact refinement.  Let a local H4/H3 source
\(T_\ell\) enter the carried closure state only through a linear observation

\[
s_\ell=O_\ell T_\ell\in\mathbb R^k.
\]

All later nonlinear state evolution may be arbitrarily deep.  For a scalar endpoint,
reverse differentiation produces a state costate \(\lambda_\ell\in\mathbb R^k\),
and the exact local source query is

\[
\boxed{
Q_\ell=O_\ell^*\lambda_\ell.
}
\]

Therefore

\[
Q_\ell\in\operatorname{range}O_\ell^*,
\qquad
\dim\operatorname{span}\{Q_\ell\}\le k,
\]

independently of future depth.  If the \(c\)-th observation channel has polarization
rank \(r_c\), then

\[
\operatorname{rank}_{\rm tilt}(Q_\ell)
\le\sum_{c=1}^k r_c.
\]

Thus a finite carried AQ/q4 channel interface gives a bounded scalar-jet pullback
even after many layers.  Reverse AD through that interface is the “resident full
operator” escape in `AQ_TILT_COSTATE_PAIR_RANK_GATE`.

The exact obstruction is not depth but **future-dark source content**.  If the true
dynamics has a later observation \(F_{>\ell}\) such that

\[
\boxed{
P_{\ker O_\ell}F_{>\ell}^*\lambda_L\ne0,
}
\]

then a component killed by the local carried observation becomes endpoint-visible
later.  No reverse computation through the reduced closure can recover it.  One must
enlarge \(O_\ell\), add a nonlocal carrier, or prove the displayed component has
subcritical endpoint norm.

This theorem reconciles the two conclusions:

- local D14/q4 observation with a bounded number of complete channels preserves
  bounded tilt rank through arbitrary closure depth;
- a generic true future costate outside that observation range still has
  \(\Omega(r^2)\) pair rank and remains a no-go.

### Graph node: AQ_OBSERVATION_RANK_DEPTH_INVARIANCE

- status: exact_if_and_only_if_boundary;
- pass: complete source enters through bounded carried observation \(O_\ell\);
- result: reverse query rank bounded by channel rank, independent of depth;
- fail: nonzero future-dark component \(P_{\ker O}F^*\lambda\);
- terminal audit: prove observation completeness on the desired source manifold or bound the dark component.

## Pure-quadratic H4 polarization recovers the master 2:1 topology

As a normalization/topology check, take three centered half-quadratics

\[
X_A=\frac12(z^{\mathsf T}Az-\operatorname{tr}A),
\quad X_B,\quad X_C.
\]

In the fourth derivative of \(X_AX_BX_C\), the only derivative allocation is
\((2,1,1)\).  There are \(4!/(2!1!1!)=12\) allocations for each choice of the
twice-differentiated factor.  Gaussian averaging gives

\[
\boxed{
\begin{aligned}
T_4(X_AX_BX_C)=12\operatorname{Sym}\{&
A\otimes(B\circ C)
+B\otimes(A\circ C)\\
&+C\otimes(A\circ B)\}.
\end{aligned}
}
\]

Since \(B\circ C=(BC+CB)/2\), each twice/once/once topology carries its two ordered
path contractions with equal weight, while the sewing/collision term appears once.
This is exactly the master

\[
2\,T\otimes_1A+A\otimes_1A
\]

ratio after the equality-sector projection.  On the diagonal \(A=B=C\), the formula
reduces to

\[
36\operatorname{Sym}(A^2\otimes A),
\]

matching `K31_H4_AQ_QUERY_DEFORMATION`.  At order six, the sole allocation
\((2,2,2)\) has coefficient \(90\), matching
\(T_6(X_AX_BX_C)=90\operatorname{Sym}(A\otimes B\otimes C)\).

Thus the tilted/deformation generator does not alter the portfolio's 2:1 physics;
it is a scalar acquisition of the same complete equality-sector law.

### Graph node: AQ_Q4_MASTER_RATIO_AUDIT

- status: exact_normalization_topology_check;
- result: boxed distinct-slot H4 polarization and diagonal/H6 checks;
- conclusion: scalar generator preserves the master 2:1 path/sewing ratio.

## Exact quartic fire gate for the tilted-generator stack

With variance and k3 exact, the master requires

\[
a_4\ge0.27732,
\qquad V_4=1.0313\times10^{-9}.
\]

The corresponding required recovered component amplitude is

\[
\boxed{
\sqrt{a_4V_4}
=\sqrt{2.860001\times10^{-10}}
=1.69115\times10^{-5}.
}
\]

Let \(w\) be the complete-old-projected tilted-generator response in the quartic
task Hilbert space.  Let

\[
\eta_{\rm tot}
=\eta_{\rm multi-residual}
+\eta_{\rm future-dark}
+\eta_{\rm provenance}
+\eta_{\rm arithmetic/tail}
\]

be a safe common-unit radius for everything not represented by the one-residual
generator.  The sharp triangle sufficient condition is

\[
\boxed{
\|w\|-\eta_{\rm tot}
\ge1.69115\times10^{-5}.
}
\]

Equivalently, the guaranteed explained quartic energy is at least

\[
(\|w\|-\eta_{\rm tot})_+^2.
\]

The complementary approximation tolerance is

\[
\boxed{
\sqrt{(1-a_4)V_4}
=\sqrt{0.7453\times10^{-9}}
\simeq2.730\times10^{-5},
}
\]

or relative norm \(\sqrt{1-a_4}=0.8501\ldots\).  Thus the architecture need not
reconstruct the ideal quartic correction exactly; it must either clear the first
witness-minus-radius gate or prove total ideal-approximation error below the second
number.

All pieces of \(\eta_{\rm tot}\) now have symbolic interfaces:

- multi-residual: exact tilted L4 moments plus cumulant Holder bound;
- future-dark: observation-kernel/costate support bound;
- provenance: deploy-vs-truth source secant;
- arithmetic/tail: differentiated characteristic and eigensystem receipts.

No provider should advance to a numerical campaign unless this inequality is at
least algebraically capable of passing under its declared rank/channel assumptions.

### Graph node: AQ_TILTED_GENERATOR_QUARTIC_GATE

- status: exact_master_threshold;
- pass: \(\|w\|-\eta_{tot}\ge1.69115\times10^{-5}\);
- equivalent error tolerance: \(2.730\times10^{-5}\) or relative \(0.8501\);
- dependency: old-span projection and all four typed uncertainty receipts.

## Singleton-slot Waring barrier for a generic endpoint K31 mask

The local tilted generator has a severe orientation asymmetry after an endpoint
matrix mixes units.  Let \(U=(U_1,\ldots,U_n)\), let

\[
V_a=\sum_jQ_{aj}U_j,
\qquad s_a=\sum_jQ_{aj}r_j,
\]

and consider the first-order K31 correction

\[
\sum_a\left[3\langle r_a,U_a^2V_a\rangle
+\langle s_a,U_a^3\rangle\right].
\]

The repeated-slot branch is local:

\[
3\langle r_a,U_a^2V_a\rangle
\]

is one mixed third jet per source \(a\), because \(V_a\) is itself one AQ scalar.
The singleton branch rearranges as

\[
\boxed{
\sum_a\langle s_a,U_a^3\rangle
=\sum_j\left\langle r_j,
P_j(U)\right\rangle,
\qquad
P_j(U)=\sum_aQ_{aj}U_a^3.
}
\]

For formal independent coordinates \(x_a\), the symmetric coefficient tensor of
\(P_j(x)=\sum_aQ_{aj}x_a^3\) is diagonal.  Its mode-one flattening has rank

\[
\operatorname{rank}P_j^{(1)}
=|\operatorname{supp}Q_{:j}|.
\]

Every Waring representation

\[
P_j(x)=\sum_{q=1}^{R_j}c_q\ell_q(x)^3
\]

has flattening rank at most \(R_j\).  Therefore

\[
\boxed{
R_j\ge|\operatorname{supp}Q_{:j}|.
}
\]

For a generic dense endpoint mask, \(R_j\ge n\) for every source column.  Exact
singleton acquisition then needs \(\Theta(n^2)\) source/query jet families, even
though each individual jet is scalar.  A low matrix rank of \(Q\) does not by itself
reduce the Waring rank of the diagonal cubic.

The algebraic escape conditions are:

1. the singleton branch is literally old, zero, or bounded below the quartic radius;
2. each relevant column of the complete-old-projected \(Q\) has bounded support;
3. the AQ laws \(U_a\) satisfy a proved low-dimensional cubic relation/dictionary;
4. a resident aggregate operator contracts the diagonal cubic against all source
   potentials without enumerating source/query pairs.

The first condition corresponds to a frozen-query/repeated-slot correction.  It is
not a fully co-adapted K31 correction unless the omitted singleton response is in
\(\eta_{\rm tot}\).

### Graph node: K31_SINGLETON_WARING_NOGO

- status: exact_generic_cost_lower_bound;
- repeated slot: \(O(n)\) scalar jets;
- singleton slot: \(\Theta(n^2)\) for generic dense \(Q\);
- proof: diagonal-cubic mode-one flattening rank;
- required escape: old/small singleton, sparse costate, low cubic dictionary, or resident aggregate contraction.

## Cheap label-free radius for dropping the singleton branch

The Waring no-go does not force enumeration merely to decide whether the singleton
branch matters.  Let

\[
k_a=\|P_{\ge3}(U_a^3)\|_2^2,
\qquad d_j=\|r_j\|_2^2,
\]

where both are available from `K31_AQ_INFLUENCE_NORMS` and the exact AQ residual
potential.  Define the positive semidefinite influence Gram

\[
K_{ab}=\langle P_{\ge3}(U_a^3),P_{\ge3}(U_b^3)\rangle.
\]

For column \(q_j=Q_{:j}\),

\[
\|P_{\ge3}P_j\|_2^2=q_j^{\mathsf T}Kq_j.
\]

Direct-sum Cauchy gives

\[
\boxed{
|\text{singleton}|
\le
\sqrt{\left(\sum_jd_j\right)
\operatorname{tr}(Q^{\mathsf T}KQ)}.
}
\]

If the full cross-unit \(K\) is unavailable, PSD and the operator norm of \(Q\)
give the entirely diagonal fallback

\[
\boxed{
|\text{singleton}|
\le
\|Q\|_{\rm op}
\sqrt{\left(\sum_jd_j\right)
\left(\sum_ak_a\right)}.
}
\]

This bound is label-free, requires only \(O(n)\) exact per-unit influence norms, and
is the earliest fail-fast decision for the cheap repeated-slot architecture:

- if the radius fits inside \(\eta_{\rm tot}\), omit the singleton branch and retain
  the \(O(n)\) repeated-slot jets;
- if it does not, the generic \(\Theta(n^2)\) acquisition remains blocked unless a
  structural Waring escape is proved.

### Graph node: K31_SINGLETON_RADIUS_GATE

- status: exact_safe_upper_bound;
- cheap inputs: residual diagonals, per-unit cubic influence norms, \(\|Q\|_{op}\);
- pass: radius below remaining quartic uncertainty allowance;
- fail: must prove a Waring/aggregate escape or abandon full co-adapted K31 acquisition.

## Minimal deployable provider: repeated-slot tilted K31 action

After the singleton split, the exact bounded-cost provider is concrete.  Mix the AQ
state once through the endpoint/reverse K31 matrix \(Q\):

\[
v_a=\sum_jQ_{aj}u_j,
\qquad
B_a=\sum_jQ_{aj}A_j,
\qquad
V_a=(v_a,B_a).
\]

Then evaluate one local residual third jet per source unit:

\[
\boxed{
W_{\rm rep}
=3\sum_a
D^3\mathcal H_{r_a}(0)[U_a,U_a,V_a].
}
\]

This is the exact repeated-slot part of the complete-old-projected first-order K31
defect.  It preserves identity-specific AQ matrices and all H3--H6 source bands.

Let

\[
d_r=r+\frac{r(r+1)}2
\]

be the packed AQ state width.  One unit-index state mix costs \(d_r/n\) ambient
GEMM-equivalents per layer; the local scalar jets cost
\(c_{\rm jet}r^3/n^2\).  Across \(L\) layers,

\[
\boxed{
C_{\rm rep}
=L\frac{d_r}{n}
+c_{\rm jet}L\frac{r^3}{n^2},
}
\]

before literal reuse.  At \(n=1024\), the state-mix terms are

\[
7.65625\quad(r32,L14),
\qquad
10.546875\quad(r45,L10),
\qquad
15.57617\quad(r55,L10),
\]

and one local r3 family adds respectively

\[
0.4375,qquad0.8690,qquad1.5867.
\]

If \((v,B)=Q(u,A)\) or the post-mix packed state is already resident for the D14/q4
costate, the first term is reuse rather than additive billing.  Otherwise it must be
charged literally.

The provider is promotable only if all four conditions hold:

1. the repeated-slot response is positive-depth and old-arena residualized;
2. the singleton radius from `K31_SINGLETON_RADIUS_GATE` fits \(\eta_{tot}\);
3. the sewn costate \(Q\) and packed mix are white-box/resident;
4. \(W_{rep}-\eta_{tot}\) clears the quartic master gate after adjusted cost.

### Graph node: K31_REPEATED_SLOT_TILTED_PROVIDER

- status: exact_bounded_cost_candidate;
- action: one packed \(Q\)-mix plus one third scalar jet per unit/layer;
- coverage: repeated-slot one-residual H3--H6;
- excluded: singleton and multi-residual branches, carried in certified radius;
- cost: boxed r32/r45/r55 symbolic census with reuse gate.

## Independent tilted-generator audit: algebra PASS, generic deployment FAIL

An independent derivation audited every convention and multiplicity.  The following
are exact:

\[
D=\alpha A+\beta B
\]

with no extra factor two in the half-quadratic convention; projection subtraction;
all q21/q31/q22/q211/q1111 slot multiplicities; and ordinary repeated derivatives
with no Taylor factorial.  In Wiener normalization,

\[
\boxed{
E[rU^2V]
=\sum_{k=3}^{6}\frac1{k!}
\langle T_k(r),T_k(U^2V)\rangle,
}
\]

and similarly for \(E[rU^3]\).  This proves again that one local third jet contains
exactly H3--H6 and no higher band.

The audit also confirms the critical deployment correction: the stack is
**entrywise exact**, not generically endpoint-cheap.  The singleton Waring barrier
is binding for a dense strict-depth K31 costate.

### Full transport/billing receipt

The incremental mix formula in `K31_REPEATED_SLOT_TILTED_PROVIDER` is not the full
architecture bill.  With no resident packed state, the independently reconciled
transport floor/base costs are

\[
\begin{array}{c|ccc}
&r32& r45&r55\\ \hline
\text{transport floor}&15.31&21.09&31.15\\
\text{full base}&16.19&22.83&34.33
\end{array}
\]

GEMM-equivalents.  Each additional unfused jet family adds

\[
0.4375,qquad0.8690,qquad1.5867
\]

respectively.  A generic \(q=n\) singleton decomposition would add approximately

\[
448,qquad889.9,qquad1624.8,
\]

and is closed by cost.  Even the base needs raw gains of approximately

\[
4.43\%,qquad6.13\%,qquad8.94\%
\]

merely to break adjusted score at \(u=0.5629\).

### Global Bellman/refit gate

Local old-feature subtraction is not sufficient.  Let

\[
d_\perp
=(I-\Pi_{\rm old})\sum_\ell T^+_{\ell\to L}s_\ell
\]

be the complete strict-depth endpoint response after global old-span projection,
and put

\[
c=\langle e,d_\perp\rangle,
\qquad G=\|d_\perp\|^2.
\]

For a free scalar refit, the exact raw gain is

\[
\boxed{\Delta R_{\rm free}=c^2/G.}
\]

With an estimated \(\hat c\), alignment error \(\eta\), and safe denominator upper
\(G_+\), the robust gain is

\[
\boxed{
\Delta R_{\rm robust}
\ge\frac{(|\hat c|-\eta)_+^2}{G_+}.
}
\]

To move raw \(2.9409\times10^{-8}\) below \(10^{-8}\), this must reach

\[
\boxed{1.9409\times10^{-8}.}
\]

If \(G=0\), the lane is an exact **global duplicate**, even if every local CP3/CP4/
H31/Q24 subtraction looked novel.  The earlier
\(1.69115\times10^{-5}\) quartic gate is only the conditional quartic-ingredient
threshold under exact-other-channel assumptions; it is not the full raw-goal gate.

### Graph node: AQ_TILTED_GENERATOR_GLOBAL_BELLMAN_GATE

- status: exact_global_gain_identity;
- pass: robust gain above the desired raw/adjusted threshold;
- global duplicate: \(G=0\);
- deployment prerequisite: bounded strict-depth costate rank after global old projection;
- generic dense singleton: cost-closed.

## Value ceiling: a local quartic jet cannot be goal-scale by itself

The scalar acquisition identities are exact, but their value must not be
misrepresented.  The entire ideal quartic channel is worth only

\[
V_4=1.0313\times10^{-9}.
\]

Even recovering it perfectly would move raw MSE only from

\[
2.9409\times10^{-8}
\quad\text{to}\quad
2.83777\times10^{-8},
\]

far above the \(10^{-8}\) campaign goal.  The required goal-scale reduction is

\[
1.9409\times10^{-8},
\]

about \(18.8\) times the full local quartic opportunity.

Therefore the tilted generator matters at campaign scale only if one of two things
is proved:

1. its strict-positive-depth Bellman response supplies the large variance/k3 value
   channels rather than merely the local quartic projection; or
2. it is a low-cost ingredient stacked with an independently certified variance
   carrier.

If global old projection leaves only the local q4 response, the lane is a bounded
ingredient, not the raw-goal solution, regardless of its exactness.

### Graph node: AQ_LOCAL_Q4_GOAL_NOGO

- status: exact_value_ceiling;
- local maximum: \(1.0313\times10^{-9}\) raw gain;
- campaign need: \(1.9409\times10^{-8}\);
- consequence: require positive-depth variance response or stack with a separate carrier.

## Fusion/value tension: commuting dictionaries erase q4-perp novelty

One apparent escape from the singleton Waring cost is a common eigenbasis.  If every
source/query AQ matrix is simultaneously diagonalizable, all tilted derivatives
reduce to scalar eigenvalue sums and can be batched without pairwise matrix
factorizations.

But simultaneous diagonalization implies

\[
[A_i,A_j]=0\qquad\forall i,j.
\]

The transverse q211/q4-perp contribution is built from the commutator Gram

\[
\langle[A_i,A_j],[A_i,A_k]\rangle_F,
\]

and therefore vanishes identically in the common-eigensystem branch.  What remains
is precisely the parallel/Jordan square physics that NCQC/Q24/CP4 already observes
most strongly.

Likewise, an identical-source/permutation dictionary makes aggregate cubic jets
cheap, but the prior symmetric-permutation intervention destroyed about ninety
percent of the identity-specific value.  Hence the two easiest fusion assumptions
are anti-correlated with the desired novel sector.

This does not rule out a task-specific low dictionary, but it sets its burden:

\[
\boxed{
\text{the dictionary must be low-rank for arithmetic while retaining a certified
noncommutative/identity-specific residual action.}
}
\]

### Graph node: K31_SINGLETON_FUSION_VALUE_TENSION

- status: exact_commuting_novelty_nogo;
- cheap common eigenbasis: q4-perp commutator value exactly zero;
- cheap permutation dictionary: empirically/value-level incompatible with identity-specific correction;
- only escape: low arithmetic dictionary with explicit noncommutative endpoint tail receipt.

## Exact source-jet dictionary criterion for singleton fusion

Let \(\mathcal T_j\) denote the complete-old-projected third derivative functional
of source residual \(r_j\) on the AQ query space:

\[
\mathcal T_j(X,Y,Z)
=D^3\mathcal H_{r_j}(0)[X,Y,Z].
\]

The singleton action is

\[
S_{\rm sing}=\sum_{ij}Q_{ij}\mathcal T_j(U_i,U_i,U_i).
\]

If the source functionals have dictionary rank \(k\),

\[
\mathcal T_j=\sum_{h=1}^{k}C_{jh}\mathcal T^{(h)},
\]

then

\[
\boxed{
S_{\rm sing}
=\sum_{h=1}^{k}\sum_i
\left(\sum_jQ_{ij}C_{jh}\right)
\mathcal T^{(h)}(U_i,U_i,U_i).
}
\]

This costs \(kn\) dictionary/query actions rather than \(n^2\).  Crucially, low
ordinary rank of \(Q\) is neither necessary nor sufficient; the compression lives
in the **source third-jet row space**.

Conversely, if the \(\mathcal T_j\) are linearly independent on the span of the
diagonal query cubes \(U_i^{\otimes3}\), any source-separable exact formula requires
\(k\ge n\).  The minimal symbolic deployment gate is therefore

\[
\boxed{
k_T:=\operatorname{rank}
\left\{(I-\Pi_{\rm old})D^3\mathcal H_{r_j}
\big|_{\operatorname{span}\{U_i^{\otimes3}\}}ight\}
=O(1),
}
\]

or a task-weighted tail bound after truncating this functional dictionary.  This is
the exact algebraic object that a resident-product fusion must compress.

### Graph node: K31_SOURCE_JET_DICTIONARY_GATE

- status: exact_iff_factorization;
- pass: old-projected source third-jet row rank \(k_T=O(1)\);
- cost: \(O(k_Tn)\) local actions;
- fail: generic independent source jets force \(k_T\ge n\) and \(n^2\) work;
- note: matrix rank of endpoint \(Q\) is not the relevant certificate.

## Certified 18.7% chain is a mechanism target, not the raw-goal value

Even an exact carrier for the already-certified 18.7% chain would recover only

\[
0.187\,(2.9409\times10^{-8})
=5.49948\times10^{-9}.
\]

It would leave

\[
2.39095\times10^{-8}
\]

raw MSE, still requiring another

\[
1.39095\times10^{-8}
\]

to reach \(10^{-8}\).  Therefore reproducing the chain is a strong validation of
source acquisition and transfer, but not campaign completion.  A goal-scale D14/
tilted architecture must expose additional ideal variance/k3 value approaching the
known near-exact oracle ceiling, or stack with a separate carrier.

### Graph node: CERTIFIED_CHAIN_RAW_VALUE_CEILING

- status: exact_arithmetic_ceiling;
- 18.7% gain: \(5.49948\times10^{-9}\);
- residual raw: \(2.39095\times10^{-8}\);
- conclusion: chain reproduction validates mechanism but cannot alone hit raw \(10^{-8}\).

## Generic full source-jet rank witness

The source-jet dictionary gate is not generically small, even in the simplest
affine family.  Let

\[
g_j=\alpha+v_j^{\mathsf T}z,
\qquad \|v_j\|=1,
\qquad \alpha\ne0,
\]

and let \(r_j=(g_j)_+-P_{\le2}(g_j)_+\).  Its raw H3 coefficient is

\[
c_3=-\alpha\phi(\alpha),v_j^{\otimes3}.
\]

Choose linear AQ queries \(U_i=v_i^{\mathsf T}z\).  Since \(U_i^3\) has H3 component
in direction \(v_i^{\otimes3}\),

\[
\boxed{
D^3\mathcal H_{r_j}(0)[U_i,U_i,U_i]
=-\alpha\phi(\alpha)(v_j^{\mathsf T}v_i)^3.
}
\]

The polynomial-kernel Gram

\[
K_{ji}=(v_j^{\mathsf T}v_i)^3
\]

has generic rank

\[
\min\left(n,{r+2\choose3}\right).
\]

At \(r=55\), \({57\choose3}=29260>1024\), so generic source/query directions give
full rank \(n\).  Hence no dimension theorem can make the source third-jet
dictionary \(O(1)\); low rank must be a special deployed/old-projected property.

The affine H3 atom may ultimately be CP3-owned, so this example is an algebraic rank
witness rather than a novelty receipt.  After complete old projection, the same
question must be asked on the surviving harmonic/noncommutative source manifold.
Independent H3 reset analysis already finds a surviving old-complement dimension
larger than \(n\), so old projection does not provide a dimension-only escape.

### Graph node: K31_SOURCE_JET_GENERIC_FULL_RANK

- status: exact_generic_rank_counterexample;
- witness: biased affine ReLU gives cubic polynomial-kernel rank \(n\) at r55/n1024;
- caveat: affine atom may be old; repeat rank gate after complete old projection;
- consequence: bounded source-jet rank requires special structure, not latent dimension alone.

## Isotropic cubic-kernel effective-rank stress test

The generic-rank witness is also quantitatively broad in an isotropic model.  For
independent uniform unit vectors in \(\mathbb R^r\),

\[
E[(v_i^{\mathsf T}v_j)^6]
=\frac{15}{r(r+2)(r+4)},
\qquad i\ne j.
\]

For the cubic Gram \(K_{ij}=(v_i^{\mathsf T}v_j)^3\),

\[
E[\operatorname{tr}K^2]
=n+n(n-1)\frac{15}{r(r+2)(r+4)}.
\]

The associated mean participation-rank scale is

\[
\boxed{
r_{\rm eff}
\simeq
\frac{n^2}{E\operatorname{tr}K^2}
=\frac{n}{1+(n-1)15/[r(r+2)(r+4)]}.
}
\]

At \(r=55,n=1024\), the sixth-moment denominator is

\[
55\cdot57\cdot59=184965,
\]

giving \(r_{\rm eff}\simeq945\).  For any realized PSD Gram, Cauchy implies that a
rank-\(k\) subspace capturing a fraction \(\gamma\) of its trace must satisfy

\[
\boxed{k\ge\gamma^2r_{\rm eff}.}
\]

At \(\gamma=0.925\), this isotropic stress requires roughly eight hundred
directions, not one or two.  This is not a task-weighted impossibility theorem—the
endpoint can select a special direction—but it is the correct prior against an
unproved bounded source-jet dictionary.

### Graph node: K31_SOURCE_JET_ISOTROPIC_RANK_STRESS

- status: exact_isotropic_moment_bound;
- r55/n1024 participation scale: about 945;
- 92.5% trace capture stress: about 800 directions;
- limitation: task-weighted low-rank alignment remains possible but must be proved.

## Full-AQ r49 plus Gaussian rest: local tilted acquisition is not the missing rest plane

The preferred r49 construction has two exactly typed forward pieces.  At a reset,
write

\[
Y_i=(g_i)_+,
\qquad
X_i=P_{\le2}Y_i,
\qquad
r_i=Y_i-X_i .
\]

The full-AQ state carries the coefficients of every \(X_i\), while the D10 branch
supplies the matched affine Gaussian covariance rest.  Thus the genuinely omitted
same-cut covariance source is

\[
\boxed{
\Delta K^{\rm rest}_{ij}
=\langle r_i,r_j\rangle-K^{G,>2}_{ij},
}
\]

followed by complete-old feature residualization.  At the centered affine base this
vanishes, and its first variation in a centered quadratic deformation vanishes by
the even/odd parity theorem.  Neither fact changes its arity: it is a **two-residual
Gram**.

The centered qualifier is binding.  For standardized bias \(\alpha\), define

\[
d_\alpha(q)=P_{>2}[1_{\{X>-\alpha\}}q],
\qquad
r_\alpha=P_{>2}(X+\alpha)_+,
\]

\[
b_\alpha(q)=\|q1_{\{|X|<|\alpha|\}}\|_2,
\qquad
h_\alpha=\|r_\alpha-r_0\|_2.
\]

Then the exact safe off-bias split is

\[
\boxed{
|\langle d_\alpha(q),r_\beta\rangle|
\le b_\alpha(q)\|r_\beta\|_2+\|q\|_2h_\beta.
}
\]

The first term is gate-sign switching; the second is drift of the affine baseline
rest itself.  A threshold-band bound alone omits the latter.  Moreover

\[
d(\alpha)=\Phi(\alpha)(1-\Phi(\alpha))-\phi(\alpha)^2
\le d(0)=\frac14-\frac1{2\pi},
\]

so

\[
\boxed{
h_\alpha\le
\sqrt{\frac14-\frac1{2\pi}}\,|\alpha|
\simeq0.3014|\alpha|.
}
\]

The pairwise first-variation bound adds the displayed term in both orientations.
Thus centered parity is an exact zero theorem, while biased use requires both
\(b\) and \(h\) receipts.

By contrast, the exact tilted residual generator

\[
\mathcal H_{r_i}(t)=E[r_i e^{U_t}]
\]

returns pairings of one residual with products of AQ variables.  Its third jet
acquires the one-residual K31/q211 terms exactly, but no collection of its univariate
jets determines \(\langle r_i,r_j\rangle\) for generic distinct sources.  The
missing plane requires a bivariate residual oracle, a resident feature Gram, a
rigorous envelope, or a separately acquired scalar Bellman action.  Therefore

\[
\boxed{
\text{local one-residual acquisition}
\;\not\Rightarrow\;
\text{r49 Gaussian-rest source acquisition}.
}
\]

The finite centered envelope remains useful once a reverse query is supplied.  If

\[
B=aa^T+cb^T+bc^T+ab^T+ba^T+bb^T
\]

is the six-outer-product entrywise envelope for \(|\Delta K^{\rm rest}|\), then for
any specified reverse plane \(\Lambda\)

\[
\boxed{
|\langle\Lambda,\Delta K^{\rm rest}\rangle|
\le \langle |\Lambda|_{\rm entry},B\rangle .
}
\]

This is six reductions.  It is a bound on one action; it neither forms the forward
plane nor constructs \(\Lambda\).

### Graph node: R49_GREST_LOCAL_GENERATOR_BOUNDARY

- status: solved exact interface / global source acquisition open;
- owned: full-AQ low chaos and matched affine Gaussian covariance rest;
- missing source: complete-old \(\Delta K^{\rm rest}\), a two-residual Gram;
- tilted generator: exact only for one-residual AQ-product actions;
- allowed shortcut: six-reduction envelope after an actual reverse plane is supplied.

## No resident reverse operator in the declared r49/D10 state

Let \(P_l\) be the frozen linear propagation from a covariance innovation at layer
\(l\) to the endpoint and let \(H\succeq0\) be the endpoint metric.  Its reverse
Bellman metric is

\[
H_l=P_l^*HP_l .
\]

For the pure variance tangent, Price's theorem gives the concrete reverse reset

\[
\boxed{
\Lambda_l
=\Gamma_l\odot
\bigl(W_{l+1}\Lambda_{l+1}W_{l+1}^T\bigr),
\qquad
(\Gamma_l)_{ij}=P(g_{li}>0,g_{lj}>0).
}
\]

The proposed r49 state contains only the forward packed \((m,u,A)\) bank and the
forward D10 covariance/rest planes.  Reusing the same weights makes the forward
conjugation algebraically adjointable, but it does not make the reverse matrices,
reset Jacobians, or accumulated error resident.  Indeed, even from a rank-one
terminal plane \(vv^T\), one reset gives

\[
\Gamma_l\odot(vv^T)=\operatorname{Diag}(v)\Gamma_l\operatorname{Diag}(v),
\]

which is generically full rank.  Scalarity of the endpoint therefore supplies no
bounded-rank reverse receipt.

A literal dense reverse chain costs two width-square products per crossed layer
before reset-adjoint work.  It can be used off card for certification, but it is
not hidden in the approximately forty-equivalent forward r49 census.  Conversely,
inserting a known source into an already-paid forward covariance conjugation is a
different operation and does not acquire the reverse query or the missing source.

### Graph node: R49_GREST_RESIDENT_REVERSE_NOGO

- status: closed for the declared state / open only under a literal buffer-reuse receipt;
- forward state: \((m,u,A)\) plus D10 covariance/rest objects;
- missing object: full-rank reverse covariance Bellman chain and accumulated-error cross state;
- rank warning: a scalar head becomes generically full rank after one correlated ReLU reset;
- cost warning: algebraic adjointability is not residency.

## Minimum causal Bellman cross observable for r49

Let the concrete rank-capacity residual be the initial endpoint error \(c\), and let
\(\varepsilon_l\) denote the omitted non-Gaussian-rest, quadrature, or seam
innovation born at layer \(l\).  Put

\[
z_l=P_l\varepsilon_l,
\qquad
Z_{l-1}=c+\sum_{k<l}z_k,
\]

and normalize by \(\|y\|_H^2\).  The local transported energy and the unique
aggregate causal cross scalar are

\[
d_l={E\|z_l\|_H^2\over\|y\|_H^2},
\qquad
\boxed{
\chi_l={E\langle Z_{l-1},z_l\rangle_H\over\|y\|_H^2}
={E\langle x_l^-,\varepsilon_l\rangle_{H_l}\over\|y\|_H^2}.
}
\]

They give the exact prefix recursion

\[
\boxed{S_l=S_{l-1}+d_l+2\chi_l.}
\]

One \(\chi_l\) per injection is sufficient for one fixed scalar endpoint
certificate and necessary among causal scalar summaries: with \(S_{l-1},d_l\)
fixed, Cauchy permits every cross value in

\[
[-\sqrt{S_{l-1}d_l},\sqrt{S_{l-1}d_l}],
\]

all with identical retained local energies.  A single noncausal total cross scalar
is enough for the final endpoint only.  Neither option closes the forward
coadapted covariance state; that requires

\[
S_l^{\rm mat}=E[x_l^-\varepsilon_l^T]
\]

or a proof that every unobserved component is Bellman-dark.

There is an exact reverse-free *diagnostic* polarization:

\[
\boxed{
2\chi_l=
{\|Z_{l-1}+z_l\|_H^2-\|Z_{l-1}\|_H^2-\|z_l\|_H^2
\over\|y\|_H^2}.
}
\]

Truthful hybrid endpoint paths can therefore measure the minimal scalar off card
without materializing \(H_l\).  This does not construct \(\Delta K^{\rm rest}\)
and is not a deployed target-free provider.

The local tilted generator computes this \(\chi_l\) only under the additional
functional-observability condition

\[
\boxed{
\lambda_l\in\operatorname{ran}O_l^*
\quad\Longleftrightarrow\quad
\ker O_l\subseteq\ker\lambda_l,
}
\]

where \(O_l\) is the complete local one-residual observation map and \(\lambda_l\)
is the pulled-back accumulated-error query.  Same-site AQ orthogonality does not
imply this condition because \(x_l^-\) contains prior higher-chaos errors.

### Graph node: R49_GREST_MINIMAL_CAUSAL_CHI

- status: solved information theorem / acquisition open;
- scalar certificate: one \(d_l,\chi_l\) pair per active omission;
- exact update: \(S_l=S_{l-1}+d_l+2\chi_l\);
- reverse-free diagnostic: endpoint polarization of three hybrid norms;
- limitation: scalar certification is not forward coadapted state closure.

## Sharp r49 coherent and block-elliptope gates

The r49 dimension diagnostic gives

\[
e_{cap}=0.171590,
\qquad
e_*=0.204019
\]

for the ideal \(6\times10^{-9}\) target.  For an aggregate rest error of energy
\(e_{rest}\) and normalized cross \(\chi\), the identity is

\[
\boxed{
e_{tot}=0.171590+e_{rest}+2\chi.
}
\]

Hence the exact signed gate is

\[
\boxed{
\chi\le{0.032429-e_{rest}\over2}.
}
\]

Orthogonality allows the full \(e_{rest}\le0.032429\) budget.  With orientation
unknown, Cauchy's inequality is sharp and robust passage requires

\[
\boxed{
e_{rest}
\le(\sqrt{0.204019}-\sqrt{0.171590})^2
\simeq1.40\times10^{-3}.
}
\]

For layerwise scalar energies set

\[
a_0=\sqrt{0.171590},
\qquad
a_l=\sqrt{d_l},
\qquad
A=\sum_la_l .
\]

The complete scalar-energy elliptope has the exact range

\[
\boxed{
\left(\max\{2\max_la_l-A,0\}\right)^2
\le e_{tot}\le A^2.
}
\]

Thus the upper endpoint below \(e_*\) is a robust pass and the lower endpoint
above \(e_*\) is a robust close.  Between them cross acquisition is logically
necessary.

With full local Gram blocks \(E_l\), define \(K=P^*HP\).  The sharp geometry is

\[
\mathcal U_*
=\max_{G\succeq0,\,G_{ll}=E_l}\langle K,G\rangle,
\qquad
\mathcal L_*
=\min_{G\succeq0,\,G_{ll}=E_l}\langle K,G\rangle.
\]

Their duals are

\[
\mathcal U_*
=\min_{Y_l}\left\{\sum_l\operatorname{tr}(Y_lE_l):
\operatorname{blkdiag}Y_l\succeq K\right\},
\]

\[
\mathcal L_*
=\max_{Y_l}\left\{\sum_l\operatorname{tr}(Y_lE_l):
K\succeq\operatorname{blkdiag}Y_l\right\}.
\]

Then \(\mathcal U_*\le e_*\|y\|_H^2\) robustly passes and
\(\mathcal L_*>e_*\|y\|_H^2\) robustly closes.  These are the sharp label-free
second-order gates when the endpoint metric is known.

Finally, \(e_{cap}=0.171590\) is currently a flat dimension diagnostic, not a
realized vector.  The cross \(\chi\) is not even typed until a frozen response
projection or joint-fit convention defines the concrete \(r_{cap}\).  Defining
capacity from an AQ-only projection and then setting its cross with a newly added
rest column to zero is invalid; orthogonality must be recomputed in the joint
space.

### Graph node: R49_GREST_COHERENCE_GATE

- status: solved sharp identity and robust pass/close geometry;
- exact signed condition: \(\chi\le(0.032429-e_{rest})/2\);
- no-cross allowance: \(0.032429\);
- unknown-coherence allowance: approximately \(1.40\times10^{-3}\);
- block certificate: \(\mathcal U_*\) passes, \(\mathcal L_*\) closes;
- prerequisite: a concrete frozen capacity residual, not dimension counting alone.

## Synchronization of the full-AQ reset object and its q211 ownership

The literal AQ engine uses three different objects which must not share one name:

\[
g_i^{AQ}=m_i+u_i^Tz+Q_{A_i}(z)
\]

is the pre-reset AQ random variable,

\[
Y_i^{AQ}=(g_i^{AQ})_+
\]

is its unprojected ReLU law, and

\[
\boxed{X_i^{AQ}=P_{\le2}Y_i^{AQ}}
\]

is the post-reset object actually represented by the carried \((m,u,A)\) bank.
The state therefore owns the complete AQ cumulant

\[
\kappa_4(X_a^{AQ},X_b^{AQ},X_c^{AQ},X_d^{AQ}),
\]

not the complete fourth cumulant of \(Y^{AQ}\).  The latter contains the omitted
high-chaos reset residual

\[
r_i^{AQ}=Y_i^{AQ}-X_i^{AQ},
\qquad
r_i^{AQ}\perp H_{\le2}.
\]

There are consequently two architecture variants:

1. **Stored-AQ state only.**  The owned fourth cumulant is
   \(\kappa_4(X^{AQ})\), and the reset-surface difference
   \(\kappa_4(Y^{AQ})-\kappa_4(X^{AQ})\) remains a candidate source after old
   subtraction.
2. **AQ plus an exact fused H211/reset-surface oracle.**  If that oracle really
   evaluates and inserts the complete \(\kappa_4(Y^{AQ})\) response, including all
   collision sewing, then the whole reset-surface bracket is additionally owned.
   The novel q211 source reduces to the preactivation-law mismatch
   \(\kappa_4(Y^{true})-\kappa_4(Y^{AQ})\).

The second branch is an augmented architecture with a separate source, transport,
and billing obligation.  It cannot be inferred from storage of \((m,u,A)\) or from
the Gaussian covariance-rest plane.

For the true preactivation law there is a second, logically independent mismatch

\[
\delta_i^{pre}=Y_i^{true}-Y_i^{AQ}.
\]

Consequently the exact true-minus-carried decomposition is

\[
\boxed{
\kappa_4(Y^{true})-\kappa_4(X^{AQ})
=\bigl[\kappa_4(Y^{true})-\kappa_4(Y^{AQ})\bigr]
 +\bigl[\kappa_4(Y^{AQ})-\kappa_4(X^{AQ})\bigr].
}
\]

The first bracket is the preactivation/capacity/secant defect.  The tilted reset
residual generator discussed here closes only the second bracket.  Conflating the
two silently assumes exact AQ preactivation representation.

At an exactly synchronized seam, put \(X=X^{AQ}\), \(r=r^{AQ}\), and
\(Y=X+r\).  Multilinearity of cumulants gives the exact reset defect

\[
\boxed{
\Delta\kappa_4
=\sum_{\varnothing\ne S\subseteq\{1,2,3,4\}}
\kappa_4(r_S,X_{S^c}).
}
\]

Because every \(r_i\) is centered and orthogonal to every AQ variable \(X_j\),
the one-residual terms have no disconnection subtraction:

\[
\kappa_4(r_a,X_b,X_c,X_d)=E[r_aX_bX_cX_d]
=D^3\mathcal H_{r_a}(0)[X_b,X_c,X_d].
\]

For the q211 entry \((a,a,b,c)\), the complete one-residual contribution is

\[
\boxed{
\begin{aligned}
\Delta^{(1)}_{aabc}
={}&2D^3\mathcal H_{r_a}[X_a,X_b,X_c]\\
&+D^3\mathcal H_{r_b}[X_a,X_a,X_c]
 +D^3\mathcal H_{r_c}[X_a,X_a,X_b].
\end{aligned}}
\]

The higher-residual pieces are, with cumulant notation retaining the necessary
same-residual covariance subtraction,

\[
\begin{aligned}
\Delta^{(2)}_{aabc}={}&
\kappa_4(r_a,r_a,X_b,X_c)
+2\kappa_4(r_a,X_a,r_b,X_c)\\
&+2\kappa_4(r_a,X_a,X_b,r_c)
+\kappa_4(X_a,X_a,r_b,r_c),\\
\Delta^{(3)}_{aabc}={}&
\kappa_4(r_a,r_a,r_b,X_c)
+\kappa_4(r_a,r_a,X_b,r_c)\\
&+2\kappa_4(r_a,X_a,r_b,r_c),\\
\Delta^{(4)}_{aabc}={}&
\kappa_4(r_a,r_a,r_b,r_c).
\end{aligned}
\]

Thus the univariate tilted generator acquires \(\Delta^{(1)}\) exactly.  It does
not acquire \(\Delta^{(2)}+\Delta^{(3)}+\Delta^{(4)}\), which require bivariate,
trivariate, and four-source residual information or certified radii.

The lawful additive source is finally

\[
\boxed{
T_{211}^{nov}
=\mathcal R_{old}\mathcal P_{211}
\left[\kappa_4(Y^{true})-\kappa_4(X^{AQ})\right],
}
\]

where \(\mathcal R_{old}\) includes the literal CP4/Q24/star/depth response span.
The total AQ q211 is not added again.

### Affine-base counterexample

Gaussian covariance rest must not be mistaken for connected-q4 ownership.  Let
\(Z\sim N(0,1)\), set \(a=(2\pi)^{-1/2}\), and take

\[
Y=Z_+,
\qquad
X=P_{\le2}Y=a+\frac12Z+\frac a2(Z^2-1).
\]

Direct moments give

\[
\kappa_4(Y)=\frac34-2a^2-6a^4,
\qquad
\kappa_4(X)=3a^2+3a^4,
\]

and hence

\[
\boxed{
\kappa_4(Y)-\kappa_4(X)
=\frac34-5a^2-9a^4\ne0.
}
\]

Give three distinct external units the same preactivation \(Z\); their
\((a,a,b,c)\) entry equals this value.  A small nonsingular perturbation of the
correlation preserves nonzero value by continuity.  D10 Gaussian rest owns
\(\langle r_i,r_j\rangle\) at this base, not this connected fourth cumulant.
CP4/Q24 may remove typed pieces, so the final statement is not that every affine
entry survives, but that no affine-zero theorem exists without a literal proof
that \(\mathcal R_{old}\) annihilates the complete expression.

In the augmented exact-surface-oracle branch, by contrast, identical affine laws
give \(Y^{true}=Y^{AQ}\), so the *remaining* true-minus-oracle q211 defect is zero.
This conditional zero belongs to the oracle branch, not to the stored-AQ state.

### Graph node: FULL_AQ_Q211_TRUE_MINUS_STATE_OWNERSHIP

- status: exact decomposition / higher-residual acquisition open;
- state-owned object: \(\kappa_4(P_{\le2}(g^{AQ})_+)\);
- novel object: complete-old true-minus-state cumulant only;
- tilted acquisition: exact one-residual piece \(\Delta^{(1)}\);
- arity boundary: two through four residual slots need separate oracles/radii;
- affine warning: Gaussian covariance rest does not make connected q211 vanish.
- architecture fork: an exact fused H211 surface oracle can own the AQ reset
  bracket, but must be costed and sewn separately.

## Audit of the native three-generator q211 carrier

The proposed native carrier stores

\[
T=a\operatorname{Sym}(A,A,B,C).
\]

For a deterministic linear/diagonal factor map \(L=D_gW^T\), tensor naturality is
exact:

\[
\boxed{
L^{\otimes4}T
=a\operatorname{Sym}(LA,LA,LB,LC).
}
\]

Its K4/K31/K22 collision reads are exact polarizations, so the three-vector state
is a valid representation of **one already-specified tensor** under the deployed
separable CP factor law.  This proves the advertised small carriage cost; it does
not supply its source amplitude or prove true nonlinear reset transport.

### No natural few-atom theorem for the true-minus-AQ source

The one-residual source can be viewed as the source-indexed family of cubic forms

\[
\mathcal T_i(x,y,z)=D^3\mathcal H_{r_i}(0)[x,y,z].
\]

After mapping AQ directions into physical unit directions, its fourth tensor has
mode-one flattening equal to the source-third-jet matrix.  One
\(\operatorname{Sym}(A,A,B,C)\) atom has mode-one range contained in
\(\operatorname{span}\{A,B,C\}\), hence flattening rank at most three.  Therefore

\[
\boxed{
R_{native}\ge{1\over3}
\operatorname{rank}\{\mathcal R_{old}\mathcal T_i\}_i.
}
\]

The biased-affine polynomial-kernel witness makes the unprojected source-jet rank
generically \(n\); complete-old projection is the only possible rank collapse and
must be demonstrated.  Thus even the structured one-residual piece has no generic
\(O(1)\)-atom decomposition.

For an unrestricted q211 coordinate sector the obstruction is stronger.  Its
dimension is

\[
n{n-1\choose2}=\Theta(n^3),
\]

whereas one native atom has at most \(3n+1\) parameters.  A generic tensor in that
sector therefore requires \(\Omega(n^2)\) atoms.  The r49-generated source lies on
a smaller analytic manifold, so this dimension count is a universal-state warning,
not a claim that every r49 instance attains the generic \(\Omega(n^2)\) rank.  The
binding source gate is the actual complete-old native secant rank or a certified
task-weighted tail.

### Exact future-costate condition

Let \(J_l\) be the full frozen Fréchet reset/transition operator on the sewn chaos
state, and let \(L_l^{\otimes4}\) be the deployed factor map.  Native factor
transport reproduces the full future action if and only if, on every live carrier
tensor,

\[
\boxed{
J_l^{4\leftarrow4}T=L_l^{\otimes4}T
}
\]

and every cross-sector output \(J_l^{s\leftarrow4}T\) is identical to the explicit
K4/K31/K22 collision read inserted exactly once into sector \(s\).  Equivalently,
if \(E_l^{reset}\) denotes the difference between the true sewn map and this
deployed map, then exactness is \(E_l^{reset}T=0\) at every crossed reset.

For a terminal costate \(Q_L\), the missed action obeys the honest bound

\[
\boxed{
|\Delta V_{reset}|
\le\sum_l\|(J_{l+1:L})^*Q_L\|_*\,
\|E_l^{reset}T_l\|.
}
\]

Closure of one linear leg, or closure between reset seams, does not imply this
identity.  A correlated ReLU fourth-order tangent is generically a sewn operator
with collision/lower-sector coupling; the deployed diagonal gate is a model of it.
Thus the native carrier solves storage and affine transport, while the complete
future-costate/reset-fidelity gate remains open.

### Graph node: Q211_NATIVE_THREE_GENERATOR_PATH_AUDIT

- status: exact deployed factor transport / source and true-reset closure open;
- ownership: carry only \(\mathcal R_{old}[\kappa_4(Y^{true})-\kappa_4(X^{AQ})]\);
- local oracle: one-residual tilted piece only;
- atom gate: complete-old source-third-jet/native secant rank, generically not constant;
- path gate: sewn reset naturality \(J_lT=L_l^{\otimes4}T\) plus exact once-only collision reads;
- value gate: endpoint dual norm of every reset defect, not one-leg fidelity.

## D14 prefix through L4 and a full-AQ r55 suffix

Consider the hybrid which retains the incumbent two-sided D14 evolution through
the cut after L4, initializes one rank-55 affine--quadratic state there, then runs
the one-sided AQ recurrence plus D10 Gaussian rest through L5--14.  This removes
the specific raw-Gaussian cubic-null obstruction of r49: the section begins after
nonlinear evolution and is allowed to have signed K3.  It does not make the seam
or the late residual exact.

### Exact weak seam equations

Use the half-quadratic convention

\[
X_i=m_i+u_i^Tz+\frac12(z^TA_i z-\operatorname{tr}A_i),
\qquad
s_i=(u_i,A_i).
\]

The covariance and cubic polarizations are

\[
B_2(s_i,s_j)=u_i^Tu_j+\frac12\operatorname{tr}(A_iA_j),
\]

\[
\begin{aligned}
B_3(s_1,s_2,s_3)={}&
\frac12\operatorname{tr}(A_1A_2A_3+A_1A_3A_2)\\
&+u_1^TA_3u_2+u_1^TA_2u_3+u_2^TA_1u_3.
\end{aligned}
\]

In particular,

\[
\boxed{
K^{21}_{ij}
=\operatorname{tr}(A_i^2A_j)
+u_i^TA_ju_i+2u_i^TA_iu_j.
}
\]

Let \((m^D,C^D,K^{21,D})\) be the D14 cut interface.  A weak exact section is
precisely a target-free choice of \(s_i\) satisfying

\[
\boxed{
m_i=m_i^D,
\qquad B_2(s_i,s_j)=C^D_{ij},
\qquad B_3(s_i,s_i,s_j)=K^{21,D}_{ij}
}
\]

for every \(i,j\).  If

\[
\Psi_r:V_r^n/O(r)\to\operatorname{Sym}_n\oplus\mathbb R^{n\times n},
\qquad \Psi_r(s)=(C,K^{21}),
\]

then existence is exactly

\[
(C^D,K^{21,D})\in\operatorname{im}\Psi_{55}.
\]

At \(n=1024,r=55\),

\[
\dim V_{55}=1595,
\qquad
p_{55}^{AQ}=1024(1595)-\frac{55\cdot54}{2}=1,631,795,
\]

while

\[
q=\frac{1024\cdot1025}{2}+1024^2=1,573,376.
\]

Thus a regular full-AQ fiber has dimension

\[
\boxed{p_{55}^{AQ}-q=58,419.}
\]

The previously quoted 2,099-dimensional fiber belongs to the *pure quadratic*
r55 model, not full AQ.  Parameter surplus removes the positive flat dimension
tail only if \(D\Psi_{55}\) is actually onto.  A stable local section additionally
requires

\[
\boxed{
\sigma_{seam}
=\sigma_{\min}(D\Psi_{55}|_{\mathrm{gauge}^{\perp}})>0
}
\]

and a single permutation/gauge-equivariant target-free branch over the corpus.

Several exact source-only necessary conditions can fail before any inverse solve:

\[
\boxed{C^D\succeq0,\qquad \ker C^D\subseteq\ker K^{21,D},}
\]

and, row by row,

\[
\boxed{
K^{21,D}_{i,:}(C^D)^+K^{21,D,T}_{i,:}
\le8(C^D_{ii})^2.
}
\]

The last inequality follows by representing the K21 row as the \(B_2\)-pairing
with

\[
T_i=(2A_iu_i,,2(A_i^2+u_iu_i^T))
\]

and bounding \(\|T_i\|_{B_2}^2\le8C_{ii}^2\).

For a scalar AQ direction, if

\[
v=\|u\|^2+\frac12\operatorname{tr}A^2,
\qquad
t={\frac12\operatorname{tr}A^2\over v},
\]

then

\[
\boxed{
{|\kappa_3(X)|\over v^{3/2}}
\le\sqrt{2t}(3-t)\le2\sqrt2.
}
\]

A strong seam which also matches a scalar fourth cumulant must obey

\[
\boxed{
0\le\kappa_4(X)
=3\operatorname{tr}A^4+12u^TA^2u
\le12v^2.
}
\]

Violating either cone is an exact no-go for every AQ rank.

Polarizing the scalar skewness cone gives a stronger rank-independent test using
only every two-unit D14 subspace.  For all anchors \(i,j\) and all real
\(\alpha,\beta\), every AQ seam must satisfy

\[
\boxed{
\begin{aligned}
&|\alpha^3K_{ii}^{21}
+3\alpha^2\beta K_{ij}^{21}
+3\alpha\beta^2K_{ji}^{21}
+\beta^3K_{jj}^{21}|\\
&\qquad\le2\sqrt2
(\alpha^2C_{ii}+2\alpha\beta C_{ij}+\beta^2C_{jj})^{3/2}.
\end{aligned}}
\]

One violation proves that no AQ section exists at any latent rank.

### Weak versus strong future seam

Matching \((m,C,K21)\) does not match the live D14 CP3/CP4/star/depth state or its
all-distinct cubic and quartic content.  Let \(T_3^D,T_4^D\) be those cut objects
in the complete old quotient and \(T_3^{AQ}(s),T_4^{AQ}(s)\) the AQ tensors.  Let
\(\mathcal B_3^+,\mathcal B_4^+\) be the full L5--14 backward-query spaces.  The
strong seam is exactly

\[
\boxed{
\langle\Lambda_3,T_3^D-T_3^{AQ}(s)\rangle=0
\quad\forall\Lambda_3\in\mathcal B_3^+,
}
\]

\[
\boxed{
\langle\Lambda_4,T_4^D-T_4^{AQ}(s)\rangle=0
\quad\forall\Lambda_4\in\mathcal B_4^+.
}
\]

Full tensor equality is sufficient; future-dark annihilation is necessary and
sufficient for the frozen endpoint.  The large weak-section fiber can be used for
future selection only if the required functional map \(\Theta\) satisfies

\[
\boxed{
D\Theta|_{\ker D\Psi_{55}}
\text{ is onto the required quotient with a positive restricted singular value}.
}
\]

Otherwise the seam mismatch is a new Bellman innovation.  A weak section is not
an exact initialization of the complete future state.

The covariance split must also be synchronized.  If D10 adds matched Gaussian
rest, fitting the AQ Gram to the *total* D14 covariance and then adding rest again
double-counts.  The exact seam identity is of the form

\[
\boxed{
C^D=C_X(s)+K^{G,>2}+\Delta K^{rest}_{seam},
}
\]

with every incumbent contribution included once.  Declaring the seam exact with
\(\Delta K^{rest}_{seam}=0\) requires proof that the displayed difference is in
the AQ Gram image and that the same Gaussian frame is used on both sides.

### What the prefix actually fixes

The prefix avoids the r49 Gaussian cubic-null section and makes covariance rank
nonbinding.  It does not prove AQ moment-image membership, a stable section, the
strong future identities, or truth-exactness.  A perfect seam copies any inherited
D14 error rather than removing it.

If the loss averages sixteen reported layer rows and L0--L4 are five frozen rows,
the hybrid has the conditional floor

\[
\operatorname{raw}_{hybrid}\ge{5\over16}M_{pre}.
\]

Thus raw below \(6\times10^{-9}\) requires

\[
M_{pre}<1.92\times10^{-8},
\]

and raw below \(10^{-8}\) requires \(M_{pre}<3.2\times10^{-8}\).  If only four
prefix rows are frozen, the corresponding thresholds are \(2.4\times10^{-8}\)
and \(4.0\times10^{-8}\).  Cut/report indexing must be fixed before applying this
floor.

### Exact suffix Bellman gate

Let \(e_0\) collect propagated D14 truth error, strong-seam mismatch, and any
frozen-prefix score error in one common endpoint normalization.  Let \(e_{suf}\)
be the aggregate L5--14 Gaussian-rest/reset/quadrature error and \(\chi\) its
cross with the inherited block.  For the \(6\times10^{-9}\) target,

\[
\boxed{
e_0+e_{suf}+2\chi\le0.204019.
}
\]

Genuine Bellman orthogonality permits

\[
e_{suf}\le0.204019-e_0,
\]

whereas unknown coherent orientation permits only

\[
\boxed{
e_{suf}\le(\sqrt{0.204019}-\sqrt{e_0})^2.
}
\]

For layer energies \(d_l\), the sharp scalar robust pass is

\[
\boxed{
\left(\sqrt{e_0}+\sum_{l=5}^{14}\sqrt{d_l}\right)^2\le0.204019.
}
\]

The exact causal alternative is ten \(\chi_l\) values starting from \(E_4=e_0\),
or the same block-elliptope upper/lower programs with the seam/prefix as block zero.
The r55 cut does not change the resident-reverse no-go: it shortens a generic dense
reverse covariance chain to ten layers, about twenty width-square products before
reset adjoints, none of which is in the forward subtotal.

### Exact conditional cost census

With four prefix transitions, ten suffix transitions, and \(K\) reset nodes,

\[
\boxed{
\begin{aligned}
C_{D14\to AQ55+G}(K)
={}&2(4)+2(10)+10{1595\over1024}\\
&+10{55^3\over1024^2}
+10K{55^2\over1024^2}.
\end{aligned}}
\]

At \(K=32\),

\[
\boxed{C\simeq46.086,\qquad\text{nominal slack}\simeq2.914.}
\]

This is not a complete receipt.  It omits or conditions on:

1. the exact boundary convention (one extra two-sided push raises the subtotal by
   two);
2. literal CP3/CP4/Q24 prefix operations not represented by the abstract D14 word;
3. construction and stable fiber selection of the nonlinear r55 seam;
4. formation and ownership of \(\Delta K^{rest}\), cumulant reads, hooks, and
   quadrature error;
5. dtype, stabilization, and precision costs;
6. every reverse/cross observable if required on card.

Most importantly, the charged coefficient push \(W^TS\) and the all-pairs
low-chaos Gram \(SS^T\) are distinct generic rectangular products.  If D10 needs
the matched correlation plane and no literal resident plane/common-subexpression
identity supplies the Gram, the missing read adds

\[
10{1595\over1024}=15.576
\]

equivalents and closes the cost path.  A forward source/plane identity, not merely
the fact that both objects use \(S\), is the binding fusion receipt.

### Graph node: D14_L4_TO_FULL_AQ55_WEAK_SECTION

- status: open exact existence and stability;
- identity: match \((m,C,K21)\) through \(\Psi_{55}\);
- regular fiber: 58,419 dimensions, not 2,099;
- fail-fast: PSD, kernel, leverage, skewness, and quartic-cone tests;
- promotion: image membership, positive gauge-restricted singular value, target-free branch.

### Graph node: D14_L4_TO_FULL_AQ55_STRONG_SEAM

- status: open future-equivalence gate;
- requirement: every D14-minus-AQ cubic/quartic seam discrepancy is future-dark;
- fiber gate: \(D\Theta|_{\ker D\Psi}\) onto/stable;
- covariance gate: AQ low Gram plus Gaussian rest equals the D14 plane exactly once.

### Graph node: D14_PREFIX_AQ55_SUFFIX_COST_RECEIPT

- status: conditional subtotal only;
- K32 arithmetic: 46.086 with 2.914 nominal slack;
- terminal fusion: avoid a distinct 15.576-equivalent low-chaos Gram read;
- missing costs: seam section, literal prefix state, rest source, hooks, precision, reset error.

### Graph node: AQ55_SUFFIX_GREST_BELLMAN_GATE

- status: exact endpoint identity / observables open;
- target: \(e_0+e_{suf}+2\chi\le0.204019\);
- causal certificate: ten late \(\chi_l\) values or full block elliptope;
- initialization verdict: Gaussian cubic-null solved, resident reverse and two-residual rest not solved.

## Quadratic true-minus-AQ q211: exact cross-plus-square action and r49 cap

Suppose an explicitly typed portion of the true-minus-AQ reset residual is itself
quadratic,

\[
E_i=Q_{\Delta M_i},
\]

and exclude every higher-chaos residual slot.  Let a common q211 costate frame have
core mixtures \(N_a\), and define the ordered products

\[
P_{ia}^{A}=M_i^{A}N_a^{A},
\qquad
P_{ia}^{T}=M_i^{T}N_a^{T},
\qquad
\Delta P_{ia}=P_{ia}^{T}-P_{ia}^{A}.
\]

For

\[
\mathcal G(P,Q)=32\langle P,Q\rangle_F
+16\langle P^T,Q\rangle_F,
\]

the exact quadratic q211 difference is

\[
\boxed{
S_{211}^{T}-S_{211}^{A}
=\sum_{i,a,b}(\Theta_i)_{ab}
\left[2\mathcal G(P_{ia}^{A},\Delta P_{ib})
+\mathcal G(\Delta P_{ia},\Delta P_{ib})\right].
}
\]

The first term is the one-residual AQ--defect cross; the second is the
two-quadratic-residual square.  Computing only the square misses the leading
physics.  Adding the \(P^A,P^A\) term duplicates the AQ q211 whenever the complete
AQ surface is explicitly owned.

The covariance-Hessian/tilted potential evaluates the same symmetric bilinear
without forming a fourth tensor.  Reusing per-unit AQ eigensystems and quadrature
nodes makes its scalar contractions cheap **after the required covariance
directions have been exposed**.  It does not imply that the cross-unit ordered bank

\[
P_{ia}^{A}=M_i^A\left(\sum_jU_{ja}M_j^A\right)
\]

is resident.  A stored eigensystem, symmetric gradient, or final scalar surface
value loses the ordered orientation needed by the cross term.

Direct expansion gives

\[
\boxed{
\Delta P_{ia}
=M_i^A\Delta N_a+\Delta M_iN_a^A+\Delta M_i\Delta N_a.
}
\]

Absent literal product reuse or low-rank structure, these are three product
families.  At \(r=49,n=1024,L=10\), one fresh product bank per common factor costs

\[
\boxed{c_{build}=10{49^3\over1024^2}=1.12199.}
\]

The subsequent native reductions cost

\[
c_{dense}=0.02290,k(k+1),
\qquad
c_{diag}=0.04580,k
\]

equivalents for a dense or common-diagonal \(k\)-frame.  Under nine equivalents,
before hooks and memory traffic, the exact arithmetic caps are:

\[
\begin{array}{c|c|c}
\text{fresh product banks}&\text{dense frame}&\text{common diagonal}\\\hline
0&k\le19&k\le196\\
1&k\le6&k\le7\\
2&k\le3&k\le3\\
3&k\le2&k\le2.
\end{array}
\]

The zero-bank row is conditional on a literal synchronized tape identity.  It does
not follow from an already-computed AQ scalar.  These caps apply only to the
quadratic residual block; genuine H3+ slots return to the residual-generator arity
and source-rank gates.

Finally, if \(\Delta M\) lies wholly in the freely refitted r49 AQ parameter
tangent, its finite AQ cross-plus-square response is a reparameterization rather
than a new additive feature.  Novelty requires the complete-old/complete-state
normal component, or an explicitly frozen section whose missed direction is then
graded end to end.

### Graph node: AQ_R49_Q211_QUADRATIC_RESIDUAL_ACTION

- status: exact cross-plus-square identity / product residency open;
- ownership: omit AQ--AQ base when the complete AQ surface is owned;
- acquisition: covariance-Hessian scalar is tensor-free but needs ordered-direction provenance;
- nine-equivalent caps: \(k=19/6/3/2\) for zero/one/two/three fresh dense-bank cases;
- scope: quadratic residual slots only; higher chaos and future costate remain separate.

## Unified full-AQ affine-seam rank theorem: r49, r54, and r55

The dimension threshold has an exact differential explanation.  For general latent
rank \(r\), use the unscaled convention

\[
X_i=u_i^Tz+z^TM_i z-\operatorname{tr}M_i
\]

and observe \((C,K21)\).  At an affine seam \(M_i=0\), let the \(n\times r\)
matrix \(U\) of linear coefficients have full column rank.  Then

\[
DC=hU^T+Uh^T,
\]

so

\[
\boxed{
\operatorname{rank}DC
=nr-\frac{r(r-1)}2.
}
\]

Its missing covariance normal is

\[
\mathcal N_U=\{Z=Z^T:ZU=0\},
\qquad
\boxed{
\dim\mathcal N_U={(n-r)(n-r+1)\over2}.}
\]

At the same point, linear variations \(h\) do not affect K21 and quadratic
variations \(H\) do not affect covariance.  Even if the affine K21 derivative
\(F_U\) is onto all \(n^2\) directed entries, the joint observation differential
therefore has cokernel exactly \(\mathcal N_U\).  No finite parameter surplus can
be borrowed at \(M=0\).

For a nonzero near-affine prefix \(M_i=tN_i\), eliminate the onto K21 block and
project the leading new covariance variation onto \(\mathcal N_U\).  The unique
Schur map is

\[
\boxed{
\mathscr S_{U,N}
=\Pi_U\mathcal L_N|_{\ker F_U}:
\ker F_U\to\mathcal N_U,
}
\]

where

\[
(\mathcal L_NH)_{ij}
=2\operatorname{tr}(H_iN_j+N_iH_j).
\]

Assuming \(F_U\) onto,

\[
\dim\ker F_U=n{r(r+1)\over2}-n^2.
\]

The exact dimension surplus of the Schur map is

\[
\boxed{
\dim\ker F_U-\dim\mathcal N_U
=p_r^{AQ}-q,
}
\]

where \(p_r^{AQ}\) is the full-AQ parameter count modulo \(O(r)\) and
\(q=\dim(C,K21)\).  Thus the global parameter threshold and the local covariance-
normal opening are the same theorem, not separate heuristics.

At \(n=1024\):

\[
\begin{array}{c|c|c|c}
r&\dim\ker F_U&\dim\mathcal N_U&\text{surplus/deficit}\\\hline
49&205,824&475,800&-269,976\\
54&472,064&470,935&+1,129\\
55&528,384&469,965&+58,419.
\end{array}
\]

Consequently:

- r49 cannot open a complete \((C,K21)\) local section even after a generic
  nonzero perturbation; its 269,976 shortfall is exactly the flat capacity deficit;
- r54 is the knife-edge possible section, requiring an onto
  \(470,935\)-row Schur map with only 1,129-dimensional residual fiber;
- r55 has much more fiber freedom, but is still rigorously not a section at the
  affine point and needs the same actual-prefix surjectivity proof.

There is a further universal hook constraint which **closes the first-order r54
escape**.  Let \(L=\operatorname{span}\{u_i\}\) and
\(E=\operatorname{Sym}^2L\).  On the shared index-linear tangent sector

\[
H_i=H(u_i),
\]

write

\[
T(x,y,z)=x^TH(y)z,
\qquad T(x,y,z)=T(z,y,x).
\]

The affine K21 derivative is

\[
(F_UH)_{ij}=2T(u_i,u_j,u_i)+4T(u_i,u_i,u_j)
=6(\operatorname{Sym}T)(u_i,u_i,u_j).
\]

Hence

\[
\boxed{
F_U|_{L\otimes\operatorname{Sym}^2L}
\text{ factors through }\operatorname{Sym}^3L.
}
\]

At r54, the shared tangent has dimension

\[
54\cdot1485=80,190,
\]

but \(\dim\operatorname{Sym}^3\mathbb R^{54}=\binom{56}{3}=27,720\), giving a
universal 52,470-dimensional hook kernel.  After quotienting the affine K21
image, the first-order covariance-normal capacity is at most

\[
\boxed{419,594<470,935.}
\]

Therefore for every prefix bank \(N\),

\[
\boxed{
\operatorname{rank}\mathscr S_{U,N}^{(54)}\le419,594,
\qquad
\operatorname{cork}\mathscr S_{U,N}^{(54)}\ge51,341.
}
\]

The previous r54 Schur-surjectivity condition remains a correct conditional
algebraic statement, but its premise is universally impossible at first order.
At least 51,341 normal singular values can open only at \(O(t^2)\) or later, so a
near-affine r54 section, if it eventually exists, is at best
\(\Omega(t^{-2})\)-conditioned in those modes.

The same hook count does **not** dimensionally close r55.  There

\[
(n-r)d-n^2+\binom{r+2}{3}
=969\cdot1540-1,048,576+29,260
=472,944,
\]

against 469,965 required covariance normals.  Thus the effective universal
first-order margin is only

\[
\boxed{472,944-469,965=2,979,}
\]

far smaller than the raw 58,419 parameter fiber but still nonnegative.  r55 is the
first of these ranks not killed by the hook-symmetry count; an actual full-rank
Schur minor remains unproved.

If a square Jacobian minor uses the Schur block, its leading determinant and new
singular values obey

\[
\det J(t)=c,t^{\dim\mathcal N_U}+O(t^{\dim\mathcal N_U+1}),
\]

\[
\boxed{
\sigma_{new}(D\Phi_r)
=|t|\sigma(\mathscr S_{U,N})+O(t^2).
}
\]

Therefore a section arbitrarily close to affine is necessarily ill-conditioned.
The D14-L4 proposal is valuable because its cut can have finite quadratic content,
but neither r54 nor r55 has a proved regular actual-cut tuple until this Schur map
is onto with an acceptable restricted singular value.

### Graph node: FULL_AQ_AFFINE_SEAM_SCHUR_THRESHOLD

- status: exact rank theorem / actual nonzero-cut minors open;
- affine result: all ranks miss \((n-r)(n-r+1)/2\) covariance normals;
- r49: dimensionally impossible, deficit 269,976;
- r54/r55: possible only through Schur surjectivity, with fibers 1,129/58,419;
- stability: condition number diverges at least as \(1/(|t|\sigma_{min}\mathscr S)\).

### Graph node: R54_SCHUR_HOOK_NOGO

- status: proved universal first-order no-go;
- hook factorization: \(L\otimes\operatorname{Sym}^2L\to\operatorname{Sym}^3L\);
- r54 deficit: at least 51,341 covariance normals for every prefix;
- consequence: those modes are only second-order or worse and near-affine conditioning is \(\Omega(t^{-2})\);
- r55 comparison: first-order dimension-feasible, but with only 2,979 effective hook margin.

## Why the AQ coefficient push does not automatically form its Gram

Let \(S\in\mathbb R^{n\times d}\) be the packed AQ coefficient bank.  A dense
transition computes

\[
Y=W^TS,
\]

while the low-chaos covariance needed by the Gaussian-rest coupling is

\[
G=YY^T.
\]

These are distinct operations.  In the declared bilinear-GEMM plus unitwise-reset
circuit, the first product is bilinear in \((W,S)\); linear reductions and
elementwise unitwise operations cannot create the cross-unit products in \(YY^T\).
Associativity only changes the charge:

\[
YY^T=W^T(SS^T)W,
\]

which needs an already-resident \(SS^T\) plus a two-sided conjugation, or a fresh
Gram plus that conjugation.  Thus a common input operand is not a one-product CSE
receipt.

There is also an information-level obstruction to replacing the Gram by an
elementwise D10 kernel.  Take

\[
A=\operatorname{diag}(a_1,a_2,a_3),
\qquad
B=Q\operatorname{diag}(b_1,b_2,b_3)Q^T.
\]

The marginal spectra and pre-reset covariance determine only

\[
\operatorname{tr}(AB)
=\sum_{pq}a_pb_qQ_{pq}^2.
\]

For a nonlinear spectral AQ reset \(f\), the post-reset inner product is

\[
\operatorname{tr}(f(A)f(B))
=\sum_{pq}f(a_p)f(b_q)Q_{pq}^2.
\]

In dimension at least three there are continuous orthostochastic variations
\(Q_{pq}^2\) preserving the first weighted sum and changing the second for generic
nonlinear \(f\).  Marginal spectra, means, and one correlation therefore do not
determine the AQ low Gram.  Exact escapes are:

1. a literally resident Gram or identical paid destination;
2. an affine reset map \(f\), which is not the exact ReLU AQ projection;
3. a shared eigenbasis/low matrix algebra, which removes generic q4-perp value;
4. an explicitly carried relative-eigenbasis/pair state, with its own cost;
5. a target-conditioned scalar action rather than a coadapted forward plane.

This is a no-CSE theorem within the declared circuit/state interface, not a
general lower bound against an unknown fast matrix-multiplication identity.

### Graph node: AQ_LOW_GRAM_PUSH_CSE_NOGO

- status: exact typed-circuit and information no-go;
- distinction: \(W^TS\) coefficient transport does not materialize \((W^TS)(W^TS)^T\);
- D10 insufficiency: marginal spectra plus correlation lose relative eigenbasis;
- consequence: D14-prefix/AQ55 cost closes if the extra 15.576-equivalent Gram has no literal reuse;
- escapes: resident Gram, affine/common-algebra structure, paid pair state, or scalar-only endpoint action.

## Exact-Gram rank/depth frontier under the 49-equivalent cap

The Gram no-CSE result creates a sharp schedule frontier rather than merely adding
an unspecified cost.  Let \(p\) incumbent D14 transitions precede an AQ suffix of
\(L\) transitions, with

\[
p+L=14.
\]

Let

\[
d_r=r+\frac{r(r+1)}2.
\]

Charge two products per prefix D14 transition, two per suffix D10 Gaussian-rest
transition, one AQ coefficient push per suffix transition, and one low-chaos Gram
at every suffix reset except the terminal state.  If the terminal head reads its
scalar variance directly from the factor bank, no terminal all-pairs Gram is
needed.  With \(K\) reset nodes the exact symbolic subtotal is

\[
\boxed{
C_{exactGram}(r,L,K)
=28+(2L-1){d_r\over n}
+L{r^3\over n^2}
+LK{r^2\over n^2}
+C_{seam}.
}
\]

The constant 28 is \(2p+2L\).  If a full terminal covariance plane is reported or
consumed, replace \((2L-1)d_r/n\) by \(2Ld_r/n\).  The one-Gram saving is therefore
conditional on a literal scalar/factored terminal read.

At \(n=1024,K=32,C_{seam}=0\), the maximum integer suffix lengths satisfying the
strict arithmetic cap are

\[
\boxed{
\begin{array}{c|c|c}
r&L_{max}&C(r,L_{max},32)\ \text{approximately}\\\hline
44&10&48.59\\
45&9&47.27\\
46&9&48.13\\
47&8&46.54\\
48&8&47.34\\
49&8&48.15\\
50&8&48.97\\
51&7&46.92\\
52&7&47.67\\
53&7&48.43\\
54&6&45.97\\
55&6&46.64.
\end{array}}
\]

Every number excludes the seam solve, hooks, stabilization, old ownership, and
meter margin.  It is a mathematical frontier, not a deploy receipt.

### Full L5--14 window

Keeping all ten value-bearing late transitions at \(K=32\) forces \(r\le44\) once
the Gram is honestly charged.  Full-AQ r44 has

\[
p_{44}^{AQ}=1,057,870,
\qquad
q-p_{44}^{AQ}=515,506,
\]

so its flat interface tail is

\[
\boxed{
e_{cap,44}={515,506\over1,573,376}\simeq0.32764.
}
\]

The raw-\(10^{-8}\) normalized allowance is

\[
\beta_{10}={1\over2.9409}\simeq0.34003.
\]

Thus r44 is also the first rank which can cross raw \(10^{-8}\) in the ideal flat
capacity calculation, but it leaves only

\[
\boxed{
\beta_{10}-e_{cap,44}\simeq0.01239
}
\]

of orthogonal reset/rest energy.  With completely unknown positive coherence, the
sharp Cauchy allowance shrinks to

\[
\boxed{
(\sqrt{\beta_{10}}-\sqrt{e_{cap,44}})^2
\simeq1.15\times10^{-4}.
}
\]

This is only about \(3.6\times10^{-10}\) raw margin under orthogonality and much
less without it.  r44 cannot reach the ideal \(6\times10^{-9}\) target even in the
flat zero-reset-error limit because \(e_{cap,44}>0.204019\).

Rank 45 has a more useful capacity margin, but at \(L=10\) its non-quadrature
subtotal is already about

\[
48.908.
\]

Since its ten-layer quadrature coefficient is approximately \(0.01931K\), the
strict 49-equivalent arithmetic permits only

\[
\boxed{K\le4}
\]

before seam/hooks.  No four-node reset-accuracy theorem currently exists.

Hence the exact trilemma is:

1. preserve the full L5--14 mechanism window and use r44, accepting a near-zero
   reset/coherence budget;
2. use r45 with at most four reset nodes, requiring a new analytic accuracy proof;
3. use larger rank and start the AQ suffix later, forfeiting early value and
   inheriting a frozen-prefix floor;
4. break the frontier only with a literal Gram fusion/resident-state receipt or a
   larger billed budget.

The flat dimension tail is not a corpus response projection, so every capacity
number above remains a screening diagnostic.  A later cut must additionally pass
the actual D14-prefix error/value floor and the strong seam theorem.

### Graph node: AQ_EXACT_GRAM_DEPTH_RANK_FRONTIER

- status: exact arithmetic/capacity screen;
- K32 full-window maximum: r44;
- r44 gate: only 0.01239 orthogonal or about 1.15e-4 coherent rest allowance for raw \(10^{-8}\);
- r45 full-window gate: at most four quadrature nodes before all extras;
- higher-rank escape: later cut with explicit frozen-prefix/value loss;
- dependency: terminal scalar Gram elision and zero-cost seam are optimistic premises.

### Concrete r55 late-cut branch

If every suffix reset, including the terminal one, must form the separated AQ Gram,
the r55 subtotal is

\[
C_{paid}(L,K)
=28+2L{1595\over1024}
+L{55^3\over1024^2}
+LK{55^2\over1024^2}.
\]

At \(K=32\), \(L=6\) costs about 48.197 and \(L=7\) about 51.563.  Thus the
largest fully paid branch is D14 through L8 and AQ55/D10 over L9--14.  Under the
literal terminal-total-covariance switch described above, its cost falls to 46.640.

The frozen chain-value map makes this branch worth retaining as a priority: the
recorded L9--L15 cells contain 15.64 of 17.46 variance-channel attribution
(approximately 89.6%) and 1.57 of 2.17 k3 attribution (approximately 72.4%).
These are mechanism-support numbers, not a nonlinear raw-MSE prediction.

If that cut freezes nine of sixteen reported rows with mean raw error \(M_{pre}\),
the unavoidable floor is

\[
\operatorname{raw}_{hybrid}\ge{9\over16}M_{pre}.
\]

It requires \(M_{pre}<1.777\overline7\times10^{-8}\) for raw below \(10^{-8}\)
and \(M_{pre}<1.066\overline6\times10^{-8}\) for raw below
\(6\times10^{-9}\).  This reported-row premise must be checked against the actual
benchmark indexing; if the output is terminal-only, the row-floor statement does
not apply.

### Graph node: D14_L8_AQ55_SUFFIX_GRAM_PAID

- status: exact conditional cost reopen / seam, surface, and value open;
- strict cost: 48.197 at K32 with six suffix resets;
- terminal-total variant: 46.640 under a literal no-future-separated-rest consumer;
- support: 89.6% variance and 72.4% k3 linear attribution in the retained window;
- hard gates: actual r55 Schur minor, prefix floor, H4 surface, six Bellman crosses, complete meter.

## Checkpoint 62: correction -- there is no reported-row prefix floor

All preceding \((p/16)M_{pre}\) claims (including the L4-cut \(5/16\) and
L8-cut \(9/16\) versions) are **superseded and invalid for the
actual board objective**.  The pinned Phase-2 source audit states

\[
\operatorname{score}
=\operatorname{MSE}(\widehat\mu_{16},\mu_{16})
\max\{0.1,F/B\},
\]

and explicitly marks all-layer MSE as diagnostic.  The prediction has shape
\(16\times1024\), but only the final layer is scored.  Therefore keeping the D14
prefix through L8 does **not** add the first nine layer MSEs to board raw, and no
factor \(9/16\) is lawful.

Let \(s_c\) be the exact cut state, \(\widehat s_c\) the inherited D14 cut state,
and \(\mathcal F\) the complete true suffix input--output map.  For a suffix
closure \(\widehat{\mathcal F}\), the exact terminal error is

\[
e_L
=\widehat{\mathcal F}(\widehat s_c)-\mathcal F(s_c)
=\underbrace{\mathcal F(\widehat s_c)-\mathcal F(s_c)}_{
  \text{transported cut error}}
 +\underbrace{\widehat{\mathcal F}(\widehat s_c)
             -\mathcal F(\widehat s_c)}_{
  \text{suffix model error}}.
\]

The two terms may contract, lie in endpoint-null directions, or cancel.  Thus
\(\|\widehat s_c-s_c\|\), a cut-layer MSE, or a collection of earlier layer MSEs
alone implies **no positive terminal lower bound**.  This is sharp: a constant
suffix makes the transported term zero; an allowed suffix correction can cancel
it exactly.

In the tangent regime, with \(J_c=D\mathcal F(s_c)\), cut error \(\delta_c\),
and suffix innovation \(z\),

\[
\|e_L\|^2
=\|J_c\delta_c\|^2+\|z\|^2
  +2\langle J_c\delta_c,z\rangle+O(\|\delta_c\|^3).
\]

A genuine prefix floor requires all of the following, on the actual cut-error
subspace \(E_c\):

1. a restricted singular-value receipt
   \(\|J_c x\|\ge\sigma_c\|x\|\) for \(x\in E_c\);
2. a cut-error norm or covariance receipt in the same normalization as final raw;
3. a cancellation/angle bound for the suffix innovation.

For example, if \(\|z\|\le b\) and
\(\|J_c\delta_c\|\ge a\), reverse triangle gives only

\[
\boxed{\|e_L\|^2\ge(a-b)_+^2.}
\]

Equivalently, if the endpoint energies \(d_c,d_s\) and normalized cross
\(\chi_c\) are available, the exact gate is

\[
\boxed{e_{final}=d_c+d_s+2\chi_c.}
\]

This is the same Bellman-coherence problem as the reset innovations, not an
independent row-average floor.

The existing P2R0F2M/P2R0F2K `autonomous_layer_mse` arrays cannot fill this gap:
they are diagnostic layerwise errors of different learned estimators, not the
D14 cut-state error transported through the proposed r55 suffix.

### Graph node: D14_PREFIX_REPORTED_ROW_FLOOR_CORRECTION

- status: solved correction / prior floor superseded;
- theorem: prefix layer MSE gives no nonzero final-layer floor without restricted endpoint gain and angle information;
- exact replacement: transported-cut plus suffix Bellman energy/cross identity;
- evidence: `OFFICIAL_SOURCE_AUDIT.json` final-layer-only score contract;
- dependency: `D14_L8_AQ55_SUFFIX_GRAM_PAID` must drop the \(9/16\) gate and use an endpoint cut-error certificate instead.

## Checkpoint 63: fixed-endpoint section versus causal typed seam

Final-layer-only scoring also sharpens what an exact seam has to preserve.  It
does **not**, however, make the seam free.

Let \(\mathcal M_r\) be the gauge-quotiented AQ coefficient manifold and let

\[
\Psi:\mathcal M_r\to\mathcal W,
\qquad \Psi(s)=(C(s),K21(s))
\]

be the weak cut-interface map.  For a fixed known suffix weight stack \(W^+\),
let

\[
\Phi_{W^+}:\mathcal M_r\to\mathbb R^n
\]

be the complete old-owned terminal mean produced from that seam, with all
Gaussian-rest/CP/Q24 ownership synchronized.  At a regular weak seam \(s_0\), a
local section that matches both the weak interface and a prescribed terminal
vector exists exactly when

\[
\boxed{
D\Phi_{W^+}(s_0)|_{\ker D\Psi(s_0)}:
\ker D\Psi(s_0)\twoheadrightarrow\mathbb R^n .
}
\]

Equivalently, in adjoint form,

\[
\boxed{
D\Phi_{W^+}(s_0)^*\lambda\in\operatorname{ran}D\Psi(s_0)^*
\quad\Longrightarrow\quad \lambda=0 .
}
\]

The restricted smallest singular value is the exact stability constant.  This
is just the implicit-function theorem applied to \((\Psi,\Phi_{W^+})\), but it
has an important count.  At r55 the weak fiber has dimension 58,419, so a
regular fixed-terminal section still has

\[
58,419-1,024=57,395
\]

free dimensions.  At r54 the nominal numbers are 1,129 and 105, respectively,
although the separate hook obstruction prevents citing the near-affine r54 weak
section as regular.  A single scalar endpoint would consume only one dimension,
but preserving only the unknown board MSE scalar would require target labels;
the lawful target-free object is the complete 1,024-vector.

### Why this does not solve deployment

There are two distinct section notions.

1. A **causal/local section** depends only on the cut state.  To be correct for
   every admissible dense suffix, it must annihilate the union of all future
   query spaces.  Dense mixing and neuron-specific reset multipliers generally
   regenerate the complete typed interface, so this returns to the strong seam
   unless an invariant future-dark subspace is proved.
2. A **fixed-suffix endpoint section** may depend on all future weights.  It only
   needs the boxed 1,024 endpoint constraints, but it must acquire
   \(D\Phi_{W^+}^*\lambda\), or an algebraically equivalent future action, and
   solve for a fiber point on every new network.

The latter is target-free--future weights and the incumbent predicted endpoint
are lawful--but it is not resident in the declared forward D14/D10 state.  There
are only three exact implementation possibilities:

\[
\boxed{
\begin{array}{ll}
\text{cold cut inversion:}&
  \text{factor/solve the nonlinear }(C,K21,\Phi_{W^+})\text{ section};\\
\text{warm AQ carry:}&
  \text{propagate the r55 AQ state from the input through the prefix};\\
\text{literal CSE/view:}&
  \text{prove the needed AQ coefficients/future action are already resident.}
\end{array}}
\]

Cold inversion has no current bounded-product formula, warm carry destroys the
late-cut cost saving, and no literal D14-to-AQ view has been exhibited.  Matching
the incumbent terminal action by continuing the omitted D14 suffix likewise
reintroduces the very transport cost the cut removed.  An abstract full-rank
minor therefore proves local representability, not a deployable seam.

This also types the Bellman objection precisely: reducing the strong tensor seam
to 1,024 terminal constraints and acquiring those constraints are separate
questions.  The first is a dimension/angle theorem; the second is a causal
computation and billing theorem.

### Graph node: AQ55_FIXED_ENDPOINT_SECTION_GATE

- status: exact IFT reduction / acquisition open;
- weak-fiber count: 58,419; target-free terminal-vector residual fiber: 57,395;
- pass: restricted endpoint Jacobian is onto with positive smallest singular value;
- adjoint certificate: no nonzero terminal covector pulls back into the weak-interface adjoint range;
- deployment gate: cold nonlinear seam solve, warm prefix carry, or literal resident view must be separately billed/proved.

### Graph node: AQ55_CAUSAL_SEAM_ACQUISITION_TRILEMMA

- status: exact architecture boundary;
- local suffix-independent route: complete future-query/strong seam unless an invariant dark space exists;
- suffix-dependent route: only 1,024 constraints, but requires a reverse/future action or equivalent continuation;
- current declared r55 late-cut subtotal includes none of the three acquisition mechanisms;
- implication: `D14_L8_AQ55_SUFFIX_GRAM_PAID` remains a conditional arithmetic shell, not yet an executable closure.

## Checkpoint 64: acquisition costs close the generic late-cut implementations

The terminal-total r55/K32 subtotal is approximately 46.640, so its nominal
headroom is only 2.360 equivalents.  Each generic acquisition branch above is
already larger before any H4 surface, stabilization, or seam conditioning cost.

### Warm carry

An r55 AQ coefficient bank has packed width

\[
d_{55}=55+\frac{55\cdot56}{2}=1595.
\]

Even if every reset coefficient and every AQ Gram were supplied for free,
carrying that bank across one dense width-1024 transition requires the
rectangular product \(W^TS\), charged as

\[
\frac{d_{55}}{1024}=1.5576171875
\]

equivalents.  Warm-carrying through the eight omitted prefix transitions thus
adds at least

\[
\boxed{8\frac{1595}{1024}=12.4609375,}
\]

giving at least 59.101 total.  Honest AQ Grams, latent reset work, and quadrature
only increase it.  Therefore warm carry cannot fit unless those rectangular
products are literal common subexpressions of the resident D14 computation.

### Fixed-future action

The two known exact generic ways to acquire the terminal action over a six-layer
suffix are:

1. continue the omitted two-sided D14 word forward, costing \(2\cdot6=12\)
   equivalents;
2. propagate the dense covariance/Bellman plane backward, also costing two dense
   products per transition before reset adjoints, hence 12.

Either raises 46.640 to at least 58.640.  This is a cost receipt for the known
generic algorithms, not an algebraic lower bound against every possible fused
circuit.  A cheaper claim must exhibit a bounded-rank invariant costate or a
literal resident product; endpoint scalarity alone does not provide either,
because a correlated ReLU gate Hadamard multiplier generically makes even a
rank-one terminal plane full rank after one pullback.

### Cold cut

Covariance-first inversion already requires a factorization

\[
C=FF^T,
\]

followed by a Stiefel orientation \(F=C^{1/2}Q\) solving the \(n^2\) signed
\(K21\) equations (and, for endpoint synchronization, 1,024 more equations).
The full-rank/minor theorem proves only that such a solution may locally exist;
it supplies neither \(C^{1/2}\), \(Q\), nor a bounded-step solve.  No cold-cut
operation appears in the declared meter.  Treating the IFT inverse as zero cost
would confuse representability with an estimator.

Thus the only surviving arithmetic escape is an exact resident-view/fusion
identity at the D14 cut.  It must name the stored factor, its gauge, the signed
orientation, and the suffix consumer; a dense covariance plane alone cannot do
so because simultaneous AQ sign reversal fixes \(C\) and flips \(K21\).

### Graph node: AQ55_LATE_CUT_ACQUISITION_COST_CLOSE

- status: generic algorithms cost-closed / literal CSE escape open;
- warm lower subtotal: +12.4609 from coefficient pushes alone;
- known fixed-future acquisitions: +12 by forward continuation or dense reverse before reset adjoints;
- cold inversion: representability theorem only, no bounded billed decoder;
- sole reopen: prove the AQ seam bank and signed orientation are literal resident views or a cheaper exact invariant circuit.

## Checkpoint 65: r55 first-order seam is universally obstructed

The positive 58,419 global parameter surplus does not make the r55 cut locally
regular near an affine seam.  There is a stronger Veronese-null obstruction.

Let \(P=L\oplus K\), \(\dim L=r\), \(\dim K=m=n-r\), and
\(E=\operatorname{Sym}^2L\), \(d=r(r+1)/2\).  Define

\[
\mathcal Q_U:E\to\mathbb R^n,
\qquad (\mathcal Q_UG)_i=u_i^TGu_i.
\]

For the affine repeated-cubic differential \(F_U\), right projection onto
\(K\) kills the term whose right factor is \(U\), so

\[
\operatorname{rank}F_U\le nr+operatorname{rank}(\mathcal Q_U)(n-r).
\]

Thus affine \(K21\) surjectivity forces \(\mathcal Q_U\) onto.  At r55,

\[
d=1540,qquad E_0=\ker\mathcal Q_U,qquad\dim E_0=516.
\]

Let \(\mathcal B=\operatorname{ran}F_U^*\cap(K\otimes E)\).  The exact
annihilator identity for the Schur quotient implies

\[
\mathcal B^\perp
=\{H\in K\otimes E:F_UH\in F_U(L\otimes E)\}
\subseteq K\otimes E_0.
\]

For any prefix bank \(N\in K\otimes E\), only its projection
\(N_0\in K\otimes E_0\) can therefore be seen by the dual Schur action.  Since
\(N_0\) is an \(969\times516\) matrix,

\[
R=\ker N_0^T,qquad \dim R\ge453.
\]

Every \(Z\in\operatorname{Sym}R\) is killed by the Schur adjoint.  Hence, for
**every** r55 prefix,

\[
\boxed{
\dim\ker\mathscr S_{U,N}^*\ge\frac{453\cdot454}{2}=102,831,
}
\]

\[
\boxed{
\operatorname{rank}\mathscr S_{U,N}\le367,134<469,965.
}
\]

If the affine \(F_U\) is not onto the seam already fails; if it is onto, the
displayed obstruction applies.  Exact block elimination has only odd normal
orders

\[
J_{normal}(t)=t\mathscr S_1+t^3\mathscr S_3+O(t^5),
\]

so a finite nonaffine cut is not universally impossible, but at least 102,831
near-affine modes can open no earlier than cubic order and have inverse
conditioning \(\Omega(t^{-3})\).

The general invisible dimension is at least

\[
2n-r-\frac{r(r+1)}2.
\]

Therefore first-order surjectivity is universally impossible until

\[
\boxed{\frac{r(r+3)}2\ge2n.}
\]

At \(n=1024\), r63 is the first rank not killed; r62 still fails.

### Cost consequence at the new threshold

For r63, \(d_{63}=2079\).  With K32 and terminal Gram elision, the exact
conditional census is

\[
C_{63}(L)=28+(2L-1)\frac{2079}{1024}
 +L\frac{63^3+32\,63^2}{1024^2}.
\]

It gives approximately

\[
\boxed{C_{63}(5)=48.0704,\qquad C_{63}(6)=52.4905.}
\]

If the terminal full AQ Gram is needed, even \(L=5\) costs about 50.1007.
Thus the first rank not universally first-order obstructed can cover only five
late transitions, only under terminal-factor read, with less than one equivalent
left for seam acquisition and all surfaces.  It does not repair the acquisition
trilemma.

### Graph node: R55_VERONESE_NULL_SCHUR_NOGO

- status: proved universal first-order no-go;
- r55 nullity: at least 102,831 normal modes for every prefix;
- finite escape: cubic or higher nonaffine section, with an explicit induced-map and conditioning receipt;
- threshold: r63 first rank not killed by the Veronese-null theorem;
- cost interaction: r63 supports at most five late transitions under the optimistic terminal-Gram switch.

## Checkpoint 66: complete-Q24 strong-seam rank obstruction

Weak matching is also generically insufficient for a prefix whose complete Q24
pair sector must be absorbed by the AQ fiber.  At an independent Gaussian cut,
for each unordered pair \(i<j\) define

\[
h_{ij}=H_2(G_i)H_2(G_j).
\]

These pair scores are orthogonal to \(m,C,K21\).  With the first future weight
equal to the identity, their off-diagonal ReLU-covariance actions satisfy

\[
\langle h_{ij},(G_i)_+(G_j)_+\rangle
=\mathbb E[Z_+H_2(Z)]^2=\frac1{2\pi},
\]

while the mismatched pair actions vanish.  Therefore the map from the Q24
\((2,2)\) pair sector to next covariance contains a diagonal nonzero minor of
size

\[
\binom{1024}{2}=523,776.
\]

Nonvanishing of that minor persists on an open dense set of first future
weights.  An r55 weak fiber has only 58,419 dimensions, so it cannot generically
select away an independently variable complete-pair strong target; the remaining
codimension is at least 465,357.

This theorem is conditional on “complete Q24” meaning those pair directions are
independent cut tangents.  If the deployed Q24 state lies on a smaller manifold,
its actual tangent rank must replace 523,776.  The clean architectural escape is
to continue the Q24/D10 pair state as a separately owned suffix word rather than
claim AQ fiber absorption; that ownership and cost must be explicit.

The independence premise has a finite CP4 certificate.  For

\[
T_4=\sum_\alpha\lambda_\alpha v_\alpha^{\otimes4},
\qquad Y_{i\alpha}=v_{\alpha i}^2,
\]

the pair read is \(R_{22}(T_4)=Y\operatorname{Diag}(\lambda)Y^T\).  A symmetric
zero-diagonal dual mask \(Q\) annihilates every CP-factor tangent only if

\[
\boxed{
\operatorname{Diag}(v_\alpha)Q(v_\alpha\odot v_\alpha)=0
\quad\text{for every active }\alpha.
}
\]

If the active factors have no zero coordinates and \(\operatorname{rank}Y=n\),
then \(QY=0\) and therefore \(Q=0\).  Hence \(DR_{22}\) is onto all
\(\binom n2\) off-diagonal pair coordinates.  A CP4 cap of 2048 is large enough
for this rank-n condition generically.  With zeros or constrained factors, the
displayed map itself is the exact finite injectivity test; dense-storage dimension
must not be substituted for it.

### Graph nodes

- `D14_L8_STRONG_SEAM_MINIMAL_OBSERVABILITY_QUOTIENT`: exact future-response quotient, tangent rowspace supplied by the tied six-step adjoint;
- `D14_L8_WEAK_MOMENT_GENERIC_NOGO`: weak \((m,C,K21)\) equality alone is generically future-insufficient;
- `D14_L8_COMPLETE_Q24_PAIR_RANK_NOGO`: complete pair absorption requires rank 523,776 generically, far above the r55 fiber;
- `D14_L8_AQ55_FIBER_STRONG_SELECTION_GATE`: only the actually retained, separately owned state may be removed before applying the restricted future-Jacobian rank test.

## Checkpoint 67: the free `W1-as-factor` AQ seed is ancillary noise

There is a tempting escape from cold covariance factorization.  Since the first
preactivation covariance is

\[
C_1=W_1W_1^T,
\]

one could choose a fixed isometry

\[
J:\mathbb R^{1024}\hookrightarrow
\mathbb R^r\oplus\operatorname{Sym}_r
\]

and map every row of \(W_1\) through \(J\).  This exactly matches covariance
without an eigendecomposition because the AQ feature dimension is at least 1024
for r45 and above.  It does **not** give a value-bearing signed orientation.

For square Gaussian \(W_1\), polar decomposition gives, conditional on
\(C_1\),

\[
W_1=C_1^{1/2}O,
\qquad O\sim\operatorname{Haar}(O(n)),
\]

with \(O\) independent of \(C_1\).  Because the input is isotropic Gaussian,
the exact network mean satisfies

\[
\mu_{16}=f(C_1,W_2,\ldots,W_{16});
\]

it is conditionally independent of the polar factor \(O\).  For any estimator
\(\widehat\mu(C_1,O,W^+)\), conditional Pythagoras gives

\[
\boxed{
\mathbb E_O\|\widehat\mu-\mu\|^2
=\|\mathbb E_O\widehat\mu-\mu\|^2
 +\mathbb E_O\|\widehat\mu-\mathbb E_O\widehat\mu\|^2.
}
\]

Thus every orientation-dependent component of the fixed-\(J\) AQ seed is pure
conditional variance and cannot improve population MSE.  The Bayes-improving
part is its Haar average, a function only of \(C_1,W^+\).

This conclusion is stronger than an aesthetic invariance objection.  It says
that an arbitrary factor orientation has zero conditional task information.  In
particular, odd AQ core tensors average to zero under \(O\mapsto-O\); even
averages reduce to orthogonal/Wick pairings.  After exact visible-collision and
old Gaussian/pairing subtraction, the residual signed orientation from a fixed
reshape has zero conditional mean.  End-to-end training across isotropic first
layers can only learn to suppress it or replace it by an invariant function.

The exact escapes are:

1. construct a seed measurable from \(C_1\) itself (a canonical square root,
   spectral frame, or another invariant section);
2. average/integrate the polar gauge exactly;
3. prove the consumed AQ functional is independent of the factor orientation,
   in which case it was already a covariance functional rather than a signed
   AQ orientation.

The raw factor \(W_1\) is computationally free but statistically ancillary.  A
canonical \(C_1\)-measurable factor removes the nuisance but restores the cold
factorization/section cost.  Spectral choices additionally have a conditioning
constant controlled by inverse eigenvalue gaps; no such factor is a literal
view of the current D14 state.

### Graph node: AQ_RAW_FACTOR_ORIENTATION_ANCILLARY_NOGO

- status: exact conditional-risk no-go;
- free seed: fixed isometric reshape of \(W_1\) matches covariance but carries Haar polar noise;
- risk identity: orientation fluctuation adds a nonnegative conditional-variance term and no bias correction;
- old projection: odd signed orientation averages to zero; even invariant pairings require duplicate audit;
- reopen: a canonical covariance-measurable section or a proved orientation-independent consumer, with separate cost/conditioning.

## Checkpoint 68: r63 is also the first dimensional complete-pair seam

The Veronese threshold has an independent interpretation.  The AQ quotient
parameter dimension is

\[
p_r=nD_r-\frac{r(r-1)}2,
\qquad D_r=\frac{r(r+3)}2.
\]

Weak \((C,K21)\) data have dimension

\[
q_{weak}=\frac{n(n+1)}2+n^2.
\]

Adding only the independent off-diagonal Q24 \((2,2)\) pair sector contributes
\(n(n-1)/2\), so the combined target has the especially simple dimension

\[
\boxed{q_{weak}+q_{22,off}=2n^2.}
\]

At \(n=1024\),

\[
\begin{array}{c|c|c}
r&p_r&p_r-2n^2\\\hline
62&2,061,469&-35,683\\
63&2,126,943&+29,791.
\end{array}
\]

Thus r63 is the first rank with enough dimensions even to attempt simultaneous
weak-interface and complete-K22-pair matching.  Adding the target-free terminal
mean vector leaves

\[
\boxed{29,791-1,024=28,767}
\]

nominal fiber dimensions.  This is only a count--the combined Jacobian minor,
membership, and conditioning remain mandatory--but it explains why the two
independent r63 thresholds coincide.

It also makes the cost conflict sharp: the first dimensionally adequate
complete-pair rank fits only a five-transition suffix under terminal Gram
elision, with about 0.93 equivalents before cold section acquisition.  Carrying
Q24 as a separate D10 word avoids the combined section count but must retain its
literal cut state and ownership; absorbing it into r55 is generically impossible.

### Graph node: R63_COMPLETE_PAIR_SEAM_DIMENSION_THRESHOLD

- status: exact necessary dimension threshold / sufficiency open;
- identity: weak interface plus off-diagonal K22 has exactly \(2n^2\) coordinates;
- r62 deficit: 35,683; r63 surplus: 29,791, or 28,767 after terminal vector preservation;
- dependency: CP4/Q24 tangent injectivity, full combined minor, seam algorithm, and five-layer cost receipt.

## Checkpoint 69: absorbing every visible q4 collision plane first becomes possible at r77

If the cut treats all visible fourth-order collision planes as independently
variable, their coordinate count is

\[
q_{4,vis}
=\underbrace{n}_{(4)}
 +\underbrace{n(n-1)}_{(31)}
 +\underbrace{\frac{n(n-1)}2}_{(22)}
=\frac{3n^2-n}{2}.
\]

Since \(q_{weak}=(3n^2+n)/2\), the complete weak-plus-visible-q4 seam has

\[
\boxed{q_{weak}+q_{4,vis}=3n^2}
\]

coordinates.  At width1024,

\[
\begin{array}{c|c|c}
r&p_r&p_r-3n^2\\\hline
76&3,071,198&-74,530\\
77&3,150,994&+5,266.
\end{array}
\]

Thus r77 is the first AQ rank with enough dimensions even before a combined
Jacobian certificate; preserving the terminal vector leaves only 4,242 nominal
fiber dimensions.

At r77, \(D_r=3080\) and

\[
\frac{77^3+32\,77^2}{1024^2}\simeq0.6163
\]

per suffix transition.  The optimistic terminal-elision census gives

\[
\boxed{C_{77}(3)\simeq44.888,qquad C_{77}(4)\simeq51.520.}
\]

Even a fully paid terminal Gram fits at three transitions (about 47.896), but
four do not.  Therefore absorbing the entire q4 collision interface trades the
separate D10 word for a suffix too short to cover the broad L5--14 mechanism.
This count is conditional on independent q4 tangents; a constrained CP4/Q24
manifold uses its actual tangent rank.  Conversely, any claim of complete
arbitrary-plane absorption must respect the r77 threshold.

### Graph node: R77_COMPLETE_VISIBLE_Q4_SEAM_THRESHOLD

- status: exact necessary dimension/cost theorem;
- identity: weak plus `(4),(31),(22)` collision planes has exactly \(3n^2\) coordinates;
- first adequate rank: r77, with only 5,266 pre-endpoint surplus;
- schedule: at most three suffix transitions under the declared cost formula;
- implication: broad-depth designs must keep Q24/D10 separately owned rather than absorb the full collision state into AQ.

## Checkpoint 70: CP4 cap 2048 generically spans the complete visible q4 tangent

The independence premise in Checkpoint 69 can be proved for the literal CP4
parameterization.  Pair a visible fourth-order dual mask with the quartic
polynomial

\[
p(x)=\sum_i a_i x_i^4
 +\sum_{i\ne j}b_{ij}x_i^3x_j
 +\sum_{i<j}c_{ij}x_i^2x_j^2.
\]

For

\[
T_4=\sum_{\alpha=1}^R\lambda_\alpha v_\alpha^{\otimes4},
\]

with nonzero \(\lambda_\alpha\), a dual mask annihilates every factor tangent
\(\delta v_\alpha\) exactly when

\[
\nabla p(v_\alpha)=0\qquad(\alpha=1,\ldots,R).
\]

Construct \(R=2n\) factors in two blocks.

1. Take \(v_i=e_i\), \(i=1,\ldots,n\).  Then

   \[
   \partial_i p(e_i)=4a_i,
   \qquad
   \partial_jp(e_i)=b_{ij}\quad(j\ne i).
   \]

   Hence annihilation on these \(n\) tangent spaces forces every `(4)` and
   oriented `(31)` coefficient to vanish.

2. The residual polynomial is

   \[
   p_{22}(x)=\sum_{i<j}c_{ij}x_i^2x_j^2.
   \]

   Choose another \(n\) dense factors with no zero coordinates such that the
   squared vectors \(y_\alpha=v_\alpha\odot v_\alpha\) form a basis of
   \(\mathbb R^n\).  Writing \(C\) for the symmetric zero-diagonal matrix of
   \(c_{ij}\),

   \[
   \partial_i p_{22}(v_\alpha)
   =2v_{\alpha i}(Cy_\alpha)_i.
   \]

   Thus all gradients vanish only if \(CY=0\), and invertibility of \(Y\)
   gives \(C=0\).

Therefore the dual annihilator is zero and

\[
\boxed{
D R_{4,31,22}:
T_{CP4}\longrightarrow\mathbb R^{(3n^2-n)/2}
\text{ is onto at an explicit }R=2n\text{ point}.}
\]

Surjectivity is a nonzero-minor condition, so it holds on a Zariski-open dense
set of CP4 factor banks.  At \(n=1024\), the deployed cap 2048 is exactly the
constructive \(2n\) threshold.  Dense transported factors need not resemble the
coordinate construction; the construction proves the generic minor is not the
zero polynomial.

Consequences:

- the complete visible q4 tangent really has dimension \((3n^2-n)/2\) for a
  generic cap-2048 CP4 state;
- r77 is the first AQ rank with enough parameter dimension to absorb it together
  with weak \((C,K21)\), not merely a pessimistic arbitrary-plane count;
- r55/r63 broad suffixes must keep CP4/Q24 as a separately owned word or accept a
  large strong-seam mismatch;
- an actual bank with truncated rank, zero coordinates, tied factors, or zero
  lambdas uses the finite gradient-evaluation rank as its literal certificate.

### Graph node: CP4_CAP2048_COMPLETE_VISIBLE_TANGENT

- status: proved explicit surjective point / generic full-rank theorem;
- construction: n coordinate rays recover K4/K31, n dense squared-basis rays recover K22;
- target rank: \((3n^2-n)/2=1,572,352\) at width1024;
- edge: certifies the independent-tangent premise of `R77_COMPLETE_VISIBLE_Q4_SEAM_THRESHOLD`;
- deployed exception gate: evaluate the symbolic finite gradient map for the actual retained factor manifold.

## Checkpoint 71: n CP3 rays generically span the full K21 plane

The analogous cubic statement is even sharper.  Pair a repeated-cubic dual mask
with

\[
p_3(x)=\sum_{i,j}b_{ij}x_i^2x_j,
\]

whose \(n^2\) coefficients are the `(3)` diagonal and oriented `(21)` plane.
For a CP3 bank \(T_3=\sum_\alpha\lambda_\alpha v_\alpha^{\otimes3}\), tangent
annihilation is \(\nabla p_3(v_\alpha)=0\).  At the \(n\) coordinate rays,

\[
\partial_i p_3(e_i)=3b_{ii},
\qquad
\partial_jp_3(e_i)=b_{ij}\quad(j\ne i).
\]

Therefore \(n\) nonzero coordinate CP3 factors give an onto Jacobian to all
\(n^2\) K21 coordinates; by the same nonzero-minor argument a generic cap at
least \(n\) does too.

This removes a possible loophole in the seam counts: a generic full-width CP3
state does make K21 an independently variable \(n^2\)-dimensional target.  Any
smaller-rank deployed exception must provide its actual gradient-evaluation
rank, not infer a reduction from the CP representation alone.

### Graph node: CP3_CAPN_FULL_K21_TANGENT

- status: proved explicit surjective point / generic theorem;
- construction: n coordinate rays;
- target rank: \(n^2=1,048,576\);
- use: validates the full weak-interface K21 dimension in the r55/r63/r77 seam thresholds.

## Checkpoint 72: exact AQ automorphism and cold-anchor theorem

Checkpoint 67's conditional-risk argument can be strengthened to a group
obstruction.  On

\[
\mathcal V_r=\mathcal H_1\oplus\mathcal H_2
\simeq\mathbb R^r\oplus\operatorname{Sym}_r,
\]

the complete AQ law automorphism group is exactly the latent action

\[
(u,M)\mapsto(Ru,RMR^T),\qquad R\in O(r),
\]

not the full orthogonal group of the covariance feature space.  Indeed,

\[
\kappa_4(X_{u,M})
=48\{\operatorname{tr}M^4+u^TM^2u\}
\]

vanishes exactly on \(M=0\), so the law intrinsically identifies \(\mathcal H_1\).
An automorphism restricts there to some \(R\in O(r)\); products of H1 variables
span H2 and force \(M\mapsto RMR^T\).

Consequently a fixed covariance isometry

\[
J:\mathbb R^n\hookrightarrow\mathcal V_r
\]

could descend through the input-rotation gauge only if

\[
JO=\rho(R(O))J\qquad\forall O\in O(n).
\]

For a generic tuple this induces a homomorphism
\(\mathfrak{so}(n)\to\mathfrak{so}(r)\).  Simplicity of \(\mathfrak{so}(n)\)
for \(n\ge5\) makes every nonzero homomorphism injective, impossible for
\(n>r\).  The zero homomorphism would require \(JO=J\) for all connected
rotations, contradicting isometry.  Hence no fixed `W1` reshape is a lawful AQ
law seed at width1024 and r55 or below.

Haar projection gives the exact alignment split.  The pulled-back cubic has
zero invariant part.  The quartic invariant part is one-dimensional:

\[
\overline b_4
=c\,\operatorname{Sym}(g\otimes g),
\]

which becomes

\[
c(C_{ij}C_{kl}+C_{ik}C_{jl}+C_{il}C_{jk}).
\]

Thus the arbitrary cubic orientation has exactly zero average alignment; the
quartic has only a complete Wick pairing average, and its residual has zero
alignment only after subtracting that **full polarized tensor**, not merely its
`31/22/4` faces.  Collision sewing leaves all-distinct interior directions and
does not repair the seed.

A generic lawful anchor requires a covariant simple-spectrum operator plus a
covariant sign vector/flag.  If \(A=V\Lambda V^T\), eigenvalue ordering supplies
lines and a vector \(b\) orients them via \(v_k=P_kb/\|P_kb\|\).  Its condition
number is at least of order

\[
\boxed{
(\gamma_{eig}\gamma_{sign})^{-1},
\quad
\gamma_{eig}=\min_{k\ne l}|\lambda_k-\lambda_l|,
\quad
\gamma_{sign}=\min_k\|P_kb\|.
}
\]

Eigenvalue collisions restore an internal orthogonal gauge; sign zeros create
branch flips.  A PSD square root followed by fixed \(J\) repeats the intertwiner
obstruction, while Cholesky is not neuron-permutation natural.

Two refinements make the anchor requirement algebraic.  Under permutations,
\(\mathbb R^{1024}=\mathbf1\oplus\mathrm{Std}\) with
\(\dim\mathrm{Std}=1023\).  Any homomorphism
\(S_{1024}\to O(r)\), \(r\le55\), is trivial on the alternating subgroup up to
the parity quotient, so the induced AQ standard-plus-symmetric-square action
cannot contain this 1023-dimensional standard module.  Fixed \(J\) therefore
fails even permutation naturality, without using continuous rotations.

Likewise, ordered eigenlines without signs retain a \((\mathbb Z_2)^{1024}\)
gauge.  Realizing those independent sign flips inside latent \(O(r)\) would
require 2-rank at least 1024, while the 2-rank of \(O(r)\) is \(r\le55\).
Hence the covariant sign flag is algebraically necessary, not merely a numerical
tie-breaking convenience.

### Graph nodes

- `AQ_COLD_SEED_FEATURE_AUTOMORPHISM`: exact law automorphisms are latent \(O(r)\);
- `AQ_COLD_SEED_FIXED_J_EQUIVARIANCE_NOGO`: no \(O(1024)\)-equivariant fixed feature isometry for r<=55;
- `AQ_COLD_SEED_HAAR_ALIGNMENT`: cubic average zero, quartic average exactly the Wick projector;
- `AQ_COLD_SEED_COLLISION_SEW_NOGO`: equality-face replacement does not identify all-distinct state;
- `AQ_COLD_SEED_CANONICAL_ANCHOR_GATE`: covariant frame/flag and eigengap/sign conditioning are mandatory.

## Checkpoint 73: the suffix-uniform future-dark tensor kernel is zero

The causal-seam branch can be closed without a generic Burnside assumption.
Let \(\Delta T_3\in\operatorname{Sym}^3\mathbb R^n\).  After one dense affine
map \(W\), its repeated-cubic read contains

\[
R_{21}(W^{\otimes3}\Delta T_3)_{ij}
=\Delta T_3(w_i,w_i,w_j),
\]

where \(w_i\) is row \(i\) of \(W\).  If this vanishes for every dense \(W\)
in any open set, polynomial continuation lets \(w_i=x,w_j=y\) range freely, so

\[
\Delta T_3(x,x,y)=0\qquad\forall x,y.
\]

Polarization of the symmetric trilinear form gives \(\Delta T_3=0\).

Likewise, for \(\Delta T_4\in\operatorname{Sym}^4\mathbb R^n\), the next
diagonal quartic read contains

\[
R_4(W^{\otimes4}\Delta T_4)_i
=\Delta T_4(w_i,w_i,w_i,w_i).
\]

Vanishing for an open set of dense \(W\) makes the associated homogeneous
quartic polynomial zero everywhere, and polarization gives \(\Delta T_4=0\).

Therefore

\[
\boxed{
\bigcap_{W\ \mathrm{dense}}\ker R_{21}W^{\otimes3}=\{0\},
\qquad
\bigcap_{W\ \mathrm{dense}}\ker R_4W^{\otimes4}=\{0\}.
}
\]

The same conclusion holds if the reads are followed by a reset with a nonzero
Hermite response coefficient.  Hence a seam initializer that is local to the
prefix and must be correct for every possible future weight stack has no
nontrivial all-distinct cubic or quartic future-dark quotient.  Weak collision
matching cannot be promoted to universal causal correctness.

For one **fixed** supplied suffix, large kernels can exist; selecting within
them is precisely the future-dependent endpoint-section problem of Checkpoint
63.  This theorem cleanly separates the two claims: local universality requires
full tensors, while low-dimensional endpoint equivalence requires acquiring the
actual future weights/costate.

### Graph node: AQ_CAUSAL_SEAM_UNIFORM_DARK_KERNEL_ZERO

- status: proved by one-step polarization;
- cubic witness: next K21 repeated read;
- quartic witness: next K4 diagonal read;
- conclusion: no nonzero suffix-uniform all-distinct seam mismatch is future-dark;
- escape: fixed-suffix selection with explicit future-action acquisition and billing.

## Checkpoint 74: exact rank--window frontier and the r62 cubic compromise

The Veronese-null seam obstruction and the exact AQ suffix bill admit one
closed symbolic frontier.  Put

\[
 e_r={r(r+1)\over2},\qquad d_r=r+e_r={r(r+3)\over2},
\]

and let `L` be the number of AQ/Gaussian-rest suffix resets.  At width
`n=1024` and quadrature depth `K=32`, the declared incremental cost per reset
is

\[
 \boxed{\Delta_r={2d_r\over1024}
 +{r^3+32r^2\over1024^2}.}
\]

Thus

\[
 C_{\rm full}(r,L)=28+L\Delta_r
\]

when the terminal reset materializes the separated all-pairs AQ Gram, while

\[
 \boxed{C_{\rm terminal\;factored}(r,L)
 =28+L\Delta_r-{d_r\over1024}}
\]

is valid only if the final consumer literally reads the scalar/factored total
and no later operation consumes the separated AQ/rest plane.

The first rank not universally killed at first order is `r=63`, because

\[
 {r(r+3)\over2}\ge2n
\]

first holds there.  Since both costs increase strictly with rank, the complete
first-order-eligible windows are

\[
 \boxed{
 \begin{array}{c|c|c}
 \text{terminal convention}&\text{longest suffix}&\text{rank window}\\ \hline
 \text{full separated Gram}&L=4&63\le r\le68\\
 \text{terminal factored read}&L=5&63\le r\le64.
 \end{array}}
\]

The corresponding cheapest subtotals are

\[
 C_{\rm full}(63,4)=45.68054,
 \qquad
 C_{\rm terminal\;factored}(63,5)=48.07040.
\]

The next reset is arithmetically closed for every first-order-eligible rank:
`C_full(63,5)=50.10067` and
`C_terminal_factored(63,6)=52.49053`.  The surviving r63 five-reset shell has
only `0.92960` headroom, before any seam solve, nonlinear H4 surface, hooks, or
stabilization.

There is a sharper higher-order compromise.  Let

\[
 g_r=1024d_r-{r(r-1)\over2}-1,573,376,
 \qquad
 \rho_r=(2048-d_r)_+,
 \qquad
 k_r={\rho_r(\rho_r+1)\over2}.
\]

Here `g_r` is the weak `(C,K21)` parameter surplus after latent orthogonal
gauge and `k_r` is the universal lower bound on first-order normal cokernel.
The exact frontier around the only plausible ranks is

\[
\begin{array}{c|r|r|r|r|c|c}
r&d_r&g_r&\rho_r&k_r&L_{\max}^{\rm full}&
L_{\max}^{\rm factored}\\ \hline
54&1539&1,129&509&129,795&6&6\\
55&1595&58,419&453&102,831&6&6\\
56&1652&116,732&396&78,606&6&6\\
57&1710&176,068&338&57,291&5&6\\
58&1769&236,427&279&39,060&5&6\\
59&1829&297,809&219&24,090&5&5\\
60&1890&360,214&158&12,561&5&5\\
61&1952&423,642&96&4,656&5&5\\
62&2015&488,093&33&561&4&5\\
63&2079&553,567&0&0&4&5\\
64&2144&620,064&0&0&4&5.
\end{array}
\]

Ranks at most53 fail the global weak-interface dimension count at every
order.  Ranks54--62 are not globally dimension-closed, but their first-order
Schur map has a forced cokernel.  If

\[
 \mathscr J_r(t)=t\mathscr S_1+t^3\mathscr S_3+O(t^5),
\]

and `c=dim coker(S1)`, the exact next gate is

\[
 \boxed{
 \pi_{\operatorname{coker}\mathscr S_1}\mathscr S_3
 \big|_{\ker\mathscr S_1}:
 \mathbb R^{g_r+c}\twoheadrightarrow\mathbb R^c.}
\]

At r62 and maximal first-order rank this is a map
`R^488654 -> R^561`.  The five-reset terminal-factored subtotal is
`47.43295`, leaving `1.56705`; only 561 modes are universally forced to cubic
order.  This makes r62 the sharp conditional arithmetic compromise, but not a
proved section: failure of the displayed cubic gate sends the surviving modes
to fifth order, and even success gives `Omega(t^-3)` inverse conditioning near
the affine seam.

Finally, neither r62 nor r63 is resident in the literal D14 lineage.  D14
stores typed moment/response planes plus CP3/CP4/star/Q24 carriers, not an
oriented `(u_i,A_i)` bank with a common `Sym^2 R^r` labeling.  A CP4 width-2048
factor is a signed fourth-order carrier, not an AQ feature factor.  Warmly
carrying r63 through nine omitted prefix transitions costs at least
`9 d63/1024=18.27246`, and a five-step dense future/reverse action already
costs ten products before reset adjoints.  Therefore these are arithmetic
shells pending a literal cold section or resident-view theorem.

### Graph node: AQ_RANK_WINDOW_VERONESE_COST_FRONTIER

- status: exact symbolic cost/obstruction frontier; seam acquisition open;
- first-order threshold: r63;
- longest first-order-eligible suffix: four resets with full terminal Gram or five with a literal factored terminal read;
- conditional compromise: r62, five resets, subtotal 47.43295, 561 cubic-order modes;
- all-order dimension threshold: r54; r53 and below closed;
- deployment blocker: no resident oriented AQ bank, and known warm/reverse paths exceed the remaining margin.

## Checkpoint 75: Price q211 trace collapse is source-cheap, not query-free

For a strict-triple Price source define

\[
 Z(R)_{i;jk}=R_{jk},\qquad i\ne j,k,
\]

in the same ordered/symmetric convention as a future q211 costate `L`.  Direct
summation gives the exact collapse

\[
 \boxed{
 \langle L,Z(R)\rangle
 =\langle\Gamma_L,R\rangle_F,
 \qquad
 (\Gamma_L)_{jk}=\sum_{i\ne j,k}L_{i;jk}.}
\]

Hence the source-side action is only an `O(n^2)` reduction after
`Gamma_L` exists; it must not be billed as a generic rank decomposition of a
three-index Price tensor.

The identity does **not** make `Gamma_L` resident.  D10's forward Price gate is
a local covariance derivative, not this future trace.  The declared full-AQ
state stores forward `(m,u,A)` and local potentials, while D10/D11 contains no
reverse q211 costate.  Nor is `range Z` generically invariant under dense
mixing followed by neuron-specific reset sewing.  An exact two-index reverse
closure would require an intertwiner

\[
 J_l Z=Z\widetilde J_l
\]

through every suffix step.  Without that theorem one must build the full
future action and trace it, or bill a new fused reverse recurrence; even dense
plane conjugation alone is two matrix products per transition before reset
adjoints.

### Graph node: PRICE_Q211_TRACE_COLLAPSE_RESIDENCY_GATE

- status: exact contraction collapse / future-query acquisition open;
- solved: `<L,Z(R)>=<Gamma_L,R>` removes the generic CP source-rank charge;
- absent: `Gamma_L` is neither the D10 local gate nor a full-AQ forward state;
- promotion gate: prove the dense/reset intertwiner or bill a reverse action;
- implication: cheap local moment acquisition does not solve Bellman closure.

## Checkpoint 76: synchronized full-AQ ownership has two distinct branches

The phrase `AQ output` must not identify three different objects.  At one
reset define

\[
 g^A_i=m_i+I_1(u_i)+Q_{A_i},\qquad
 Y^A_i=(g^A_i)_+,
\]

and let the actually carried full-AQ state be

\[
 \boxed{X^A_i=P_{\le2}Y^A_i.}
\]

Thus `ReLU(g_AQ)` is the exact nonlinear AQ-law output, while the stored AQ
word is its degree-at-most-two projection.  Put

\[
 r^A_i=Y^A_i-X^A_i=P_{>2}Y^A_i,
 \qquad
 \delta_i=Y^{\rm true}_i-Y^A_i.
\]

For the external connected q211 functional

\[
 T_{211}(i,j,k;V)=\kappa_4(V_i,V_i,V_j,V_k),
\]

there is an exact once-only split

\[
\boxed{
T_{211}(Y^{\rm true})-T_{211}(X^A)
=\underbrace{T_{211}(Y^A)-T_{211}(X^A)}_{\text{intrinsic AQ reset surface}}
+\underbrace{T_{211}(Y^{\rm true})-T_{211}(Y^A)}_{\text{true-minus-AQ deformation}}.}
\]

The first bracket contains at least one `r^A` slot; the second contains at
least one `delta` slot.  More explicitly, with `A=Y^A`, the deformation term
is

\[
\begin{aligned}
&2\kappa(\delta_i,A_i,A_j,A_k)
 +\kappa(\delta_i,\delta_i,A_j,A_k)
 +\kappa(A_i,A_i,\delta_j,A_k)
 +\kappa(A_i,A_i,A_j,\delta_k)\\
&+2\kappa(\delta_i,A_i,\delta_j,A_k)
 +2\kappa(\delta_i,A_i,A_j,\delta_k)
 +\kappa(A_i,A_i,\delta_j,\delta_k)\\
&+\kappa(\delta_i,\delta_i,\delta_j,A_k)
 +\kappa(\delta_i,\delta_i,A_j,\delta_k)
 +2\kappa(\delta_i,A_i,\delta_j,\delta_k)
 +\kappa(\delta_i,\delta_i,\delta_j,\delta_k).
\end{aligned}
\]

The intrinsic surface is the identical formula with base `X^A` and increment
`r^A`.  This makes the architecture conditional and unambiguous:

1. **Stored-AQ-only branch.**  Carrying `(m,u,A)` owns
   `T211(X^A)` only.  Its additive candidate before incumbent subtraction is
   `T211(Ytrue)-T211(X^A)`, including the nonzero intrinsic AQ surface.
2. **AQ plus exact reset-surface oracle.**  If the fused H211/tilted oracle
   literally computes `T211(Y^A)` in the same convention, it owns the first
   bracket.  The only new candidate is then the second bracket, which has at
   least one true-minus-AQ residual slot and vanishes when the true law equals
   the AQ law.

Both branches must finally apply the complete CP4/star/Q24/depth old-span
projector.  CP4 ownership of the affine H4 atom and finite Q24 ownership of
some collision diagrams do not justify replacing the first bracket by zero
unless an exact surface receipt covers the whole bracket.

The intrinsic term is not a technicality.  Let `G` be standard normal,
`Y=G_+`, and `s=(2pi)^(-1/2)`.  Then

\[
 P_{\le2}Y=s+\frac12H_1(G)+\frac{s}{2}H_2(G).
\]

Shift invariance of fourth cumulants and direct Gaussian moments give

\[
 \kappa_4(Y)
 ={3\over4}-{1\over\pi}-{3\over2\pi^2},
\qquad
 \kappa_4(P_{\le2}Y)
 ={3\over2\pi}+{3\over4\pi^2},
\]

so

\[
 \boxed{
 \kappa_4(Y)-\kappa_4(P_{\le2}Y)
 ={3\over4}-{5\over2\pi}-{9\over4\pi^2}\ne0.}
\]

Taking three perfectly correlated affine units turns this into a q211
counterexample; analyticity makes it nonzero on a nearby nonsingular set.
Therefore an affine Gaussian preactivation does **not** make the
stored-AQ-only q211 residual vanish.  It makes only the augmented-oracle
true-minus-AQ bracket vanish.  The on-disk finite Q24/CP4 arena may remove a
subspace of the counterexample after old projection, but stored AQ state alone
does not own it.

### Graph nodes

- `FULL_AQ_STORED_Q211_OWNERSHIP`: owns `T211(P<=2 ReLU(gAQ))`; intrinsic reset surface remains;
- `FULL_AQ_H211_SURFACE_ORACLE`: conditional augmentation owning `T211(ReLU(gAQ))-T211(XAQ)` with its separate bill;
- `FULL_AQ_TRUE_MINUS_AQ_Q211`: only the at-least-one-delta expansion; zero at identical true/AQ law;
- `FULL_AQ_Q211_OLD_SPAN_SUBTRACTION`: CP4/star/Q24/depth projection applies after the exact two-bracket split.

## Checkpoint 77: exact Price-trace transport theorem and the generic n4 barrier

Checkpoint 75's partial trace is cheap at a fixed source seam, but it is not a
closed reverse state through a generic dense map.  For a symmetric quartic
costate `L` define

\[
 \mathsf C_G(L)_{cd}=\sum_{a,b}G_{ab}L_{abcd},
 \qquad G\in\operatorname{Sym}_n.
\]

If

\[
 G\odot R={1\over6}\big(
 G_{ab}R_{cd}+G_{ac}R_{bd}+G_{ad}R_{bc}
 +G_{bc}R_{ad}+G_{bd}R_{ac}+G_{cd}R_{ab}\big),
\]

then full symmetry gives the exact scalar identity

\[
 \boxed{\langle L,G\odot R\rangle
 =\langle\mathsf C_G(L),R\rangle_F.}
\]

The Price trace of Checkpoint 75 is the special metric `G=I`, followed by the
known `31/22/4` collision exclusions.

Let one dense linear step be `W`, so the pulled-back costate is

\[
 L^-=(W^T)^{\otimes4}L^+.
\]

Direct index contraction gives

\[
 \boxed{
 \mathsf C_I(L^-)
 =W^T\mathsf C_{WW^T}(L^+)W.}
\]

More generally,

\[
 \mathsf C_H((W^T)^{\otimes4}L)
 =W^T\mathsf C_{WHW^T}(L)W.
\]

Therefore the ordinary partial trace `C_I(L+)` is insufficient unless `W` is
conformal, `L` belongs to a special invariant family, or the action is already
factorized.  The obstruction has a two-dimensional exact witness.  Let `L`
be the symmetric tensor of the harmonic polynomial

\[
 p(x_1,x_2)=x_1^4-6x_1^2x_2^2+x_2^4,
\]

so `L1111=L2222=1` and every permutation of `L1122` equals `-1`.  Then

\[
 \mathsf C_I(L)=0,
\]

but for `G=diag(g1,g2)`,

\[
 \mathsf C_G(L)
 =(g_1-g_2)\operatorname{diag}(1,-1),
\]

which is nonzero whenever the dense-map metric is anisotropic.  Two future
costates can thus have the same resident Price trace and different predecessor
actions.  This is an information-theoretic failure, not merely an inefficient
implementation.

The obstruction survives **complete collision subtraction** with a pure
all-distinct witness.  In four coordinates take the symmetric tensor of

\[
 p(x)=x_1x_2x_3x_4,
\]

so each permutation of `L1234` is `1/24` and every equality entry is zero.
Again `C_I(L)=0`.  But for the positive-definite metric

\[
 G=I+\varepsilon(e_1e_2^T+e_2e_1^T),\qquad |\varepsilon|<1,
\]

one has

\[
 [\mathsf C_G(L)]_{34}=\varepsilon/12\ne0.
\]

Thus even perfect K4/K31/K22 collision receipts do not make the ordinary
Price trace transportable; the missing information can live entirely in the
q1111 interior.

To answer every possible dense-map metric exactly one must in general retain
the linear operator

\[
 \mathcal H_L:\operatorname{Sym}_n\to\operatorname{Sym}_n,
 \qquad G\mapsto\mathsf C_G(L),
\]

which is simply the quartic tensor reshaped as a symmetric Hessian operator.
Its generic information size is order `n^4`; no universal two-index reverse
closure exists.

There is, however, a topology-specific `O(k n^3)` escape.  If the complete
old-projected future costate has pair-product rank `k`,

\[
 L=\sum_{\alpha=1}^k B_\alpha\odot C_\alpha,
\]

then

\[
\boxed{
\mathsf C_G(B\odot C)
={1\over6}\left[
 \langle G,B\rangle C+\langle G,C\rangle B
 +2(BGC+CGB)
\right].}
\]

Each atom needs only a constant number of dense matrix products, hence
`O(n^3)` arithmetic per atom and `O(n^2)` storage.  Commuting/diagonal atoms
are cheaper still.  This is the exact boundary:

- source partial trace at a fixed seam: `O(n^2)`;
- transport of a certified bounded pair-product costate: `O(k n^3)`;
- generic future costate with no factor receipt: an order-`n^4` operator, so
  the advertised scalar/plane trace recursion is impossible.

### Graph node: PRICE_Q211_DENSE_TRANSPORT_HESSIAN_BARRIER

- status: exact transport identity and counterexample;
- no-go: `C_I(L)` does not determine `C_{WWT}(L)` for anisotropic dense `W`;
- generic state: quartic Hessian operator `G -> C_G(L)`, order n4;
- structured GO: `k` pair-product atoms give the displayed O(k n3) recurrence;
- promotion gate: complete-old, strict-depth future costate must have certified bounded pair-product rank, not merely a cheap local Price source.

## Checkpoint 78: a lawful cold AQ anchor needs 1,013 explicit directions at r55

The cold-seed obstruction can be made finite and sharp for Krylov/rational
constructions.  Suppose `k` neuron-space directions are canonically anchored
and a `q=n-k` dimensional completion remains arbitrary.  Independence of the
chosen completion is equivalent to the AQ feature representation containing

\[
 \mathbf1^k\oplus\operatorname{Std}_q
\]

under some faithful homomorphism `tau:O(q)->O(r)`.  For `r=55`, low-dimensional
orthogonal representation theory forces the nontrivial block to be a standard
`q`-block plus `p=55-q` trivial latent coordinates.  The fixed dimension in

\[
 \mathcal H_1\oplus\mathcal H_2
 =\mathbb R^{55}\oplus\operatorname{Sym}^2\mathbb R^{55}
\]

is at most

\[
 f(q)=p+1+{p(p+1)\over2}.
\]

The necessary and sufficient count `1024-q<=f(q)` becomes

\[
 p^2+p-1936\ge0.
\]

Hence

\[
 \boxed{q\le11,\qquad k\ge1013.}
\]

This is sharp: `tau=Std_11 plus 1^44` gives 1,035 fixed AQ coordinates and a
latent standard 11-block, so any 1,013 anchored directions plus an arbitrary
11-completion define the same AQ law up to latent `O(11)`.

Therefore a block Krylov flag

\[
 K_d=[B,CB,\ldots,C^dB],\qquad B\in\mathbb R^{1024\times s},
\]

must satisfy the exact column lower bound

\[
 \boxed{s(d+1)\ge1013.}
\]

A rational family `{r_l(C)b_a}` likewise needs `s m>=1013`; poles can change
conditioning but not the number of directions.  One seed needs degree at least
1,012, 32 seeds still need degree31, and even55 seeds need degree18.  A
bounded-degree one/few-vector anchor is therefore closed.

There is a conditional exact construction.  If an ordered,
input-gauge-invariant, neuron-covariant seed bank produces a square nonsingular
block Krylov matrix—e.g. `s=32,d=31`—take its positive-diagonal QR `K=QR`.
For any covariance factor `WW^T=C`, QR-factor

\[
 W^TQ=AT
\]

and set `F=WA=QT^T`.  Then `FF^T=C`, `F` is input-gauge invariant, and it is
neuron-covariant.  The 1,013-column partial variant uses the exact `O(11)`
completion theorem above.  The literal receipts are

\[
 \sigma_{1013}(K)>0
 \quad\text{or}\quad \sigma_{1024}(K)>0,
\]

plus regularity of the compressed covariance QR.  In the one-seed square case

\[
 \det K=\det(V)\prod_j\beta_j\prod_{i<j}(\lambda_j-\lambda_i),
\]

so the route inherits exactly the anchor-projection and eigengap conditioning
it was meant to avoid.  Generic exact polynomial square-root interpolation
needs degree at least `n-1`; a rational type `(a,b)` needs at least
`a+b+1>=n` parameters.  The construction avoids the word eigendecomposition,
not global factorization or its conditioning.

### Graph nodes

- `AQ_COLD_ANCHOR_KRYLOV_FLAG_LOWER_BOUND`: sharp q<=11 / k>=1013 representation theorem at r55;
- `AQ_COLD_ANCHOR_RATIONAL_COLUMN_NOGO`: every explicit rational flag needs sm>=1013;
- `AQ_COLD_ANCHOR_BLOCK_KRYLOV_CONSTRUCTION`: conditional exact covariant QR construction;
- `AQ_COLD_ANCHOR_KRYLOV_CONDITIONING_GATE`: minimum Krylov singular value and compressed-covariance regularity;
- `AQ_COLD_ANCHOR_EXACT_MATRIX_FUNCTION_DEGREE_NOGO`: generic bounded-degree polynomial/rational shortcuts closed.

## Checkpoint 79: shared-Waring Price contraction is O(n3) even at rank n

Pair-product rank alone overstates the cost for a common Waring bank.  Suppose
the complete future quartic costate is literally

\[
 L=\sum_{\alpha=1}^k\lambda_\alpha v_\alpha^{\otimes4},
 \qquad
 V_{\alpha,:}=v_\alpha^T.
\]

Then the Hessian contraction from Checkpoint 77 collapses exactly to

\[
 \boxed{
 \mathsf C_G(L)
 =V^T\operatorname{Diag}\!\left(
 \lambda\odot\operatorname{diag}(VGV^T)
 \right)V.}
\]

This needs only the rowwise quadratic forms of `G`.  With `V` resident, form
`VG`, take row dot products with `V`, and form one weighted Gram.  The leading
cost is `2 k n^2`, or `2k/n` dense-`n^3` equivalents, not `k` separate
matrix-matrix contractions.  In particular a generic diagonal endpoint pulled
through one dense map has catalecticant/pair rank `n` but still costs only a
constant number of matrix products because all atoms share one factor bank.

Dense transport preserves the representation:

\[
 (W^T)^{\otimes4}L
 =\sum_\alpha\lambda_\alpha(W^Tv_\alpha)^{\otimes4},
 \qquad V^- = V^+W.
\]

Updating the bank costs another `k n^2`, so an unfused transport-plus-source
action is `3k/n` matrix-product equivalents per layer.  At `k=n` this is
three; at the literal cap `k=2n` it is six.  If `V^+W` is already a resident
CP4 push, only the `2k/n` contraction increment remains.

For a source injected at every one of `L` suffix resets, those charges are
causal and add: even the best `k=n` exact bank costs `3L` unfused or `2L` with
literal transport reuse.  Thus a five-reset r62/r63 shell would need 15 or 10
additional equivalents, far above its 1.56705/0.92960 headroom.  The identity
is arithmetically viable there only for one isolated seam, a substantially
smaller residual factor count, or a stronger same-destination fusion that
also removes the weighted-Gram contraction.  A forward CP4 factor push is not
by itself such a fusion for a backward endpoint costate.

The representation also survives a purely separable coordinate Jacobian:
`v_alpha -> D v_alpha`.  It is not automatically closed through a complete
nonlinear reset.  Coordinate quartic births append up to `n` new factors per
reset, so an exact untruncated bank can grow linearly with depth.  Pair/star
sewing and an H211 surface can also leave the pure-Waring variety.  A cap-2048
CP4 tape is therefore an exact carrier only for its retained factors, not a
proof that the complete future costate has `k<=2048`.

There is also a binding ownership condition.  The incumbent D11 CP4 action is
old span.  Reusing its factor push to contract the same costate computes an
already-owned quartic direction.  An additive route must first form

\[
 L_\perp=(I-\Pi_{\rm CP4/star/Q24/depth})L
\]

and prove that **this residual** still has a resident shared-Waring
factorization of bounded `k`.  CP4-dropped tail factors are a possible source,
but their factor count, target coupling, and truncation error need separate
receipts.  A low rank for the pre-projection bank says nothing about the rank
after Schur subtraction.

### Graph node: PRICE_Q211_SHARED_WARING_ESCAPE

- status: exact O(k n2) contraction identity / residual factor receipt open;
- formula: `C_G(L)=V^T Diag(lambda*diag(VGV^T)) V`;
- dense transport: one shared factor push `V<-VW`, preserving Waring form;
- arithmetic: contraction `2k/n` and fresh transport plus contraction `3k/n` GEMM-equivalents per layer;
- ownership: incumbent retained CP4 component is duplicate; only complete-old-projected tail can count;
- closure gate: bounded residual factor count and reset/depth stability, not catalecticant rank alone.

## Checkpoint 80: reverse-free pair propagation and the multi-injection rank law

For a fixed supplied suffix there is an exact alternative to constructing the
future quartic costate: propagate each Price source **forward**.  Naturality of
the normalized pair product gives, for every linear map `P`,

\[
 \boxed{P^{\otimes4}(G\odot R)
 =(PGP^T)\odot(PRP^T).}
\]

Thus a source `I odot R_l` born at layer `l` reaches the endpoint as

\[
 G_l\odot S_l,
 \qquad
 G_l=P_{l\to L}P_{l\to L}^T,
 \quad
 S_l=P_{l\to L}R_lP_{l\to L}^T.
\]

With sources at several layers, the exact endpoint perturbation is

\[
 \boxed{T_L=\sum_l G_l\odot S_l.}
\]

This proves a fixed-suffix `O(n^3)` route for one isolated injection without
any n4 tensor.  It also exposes why the broad L5--14 carrier is different.  If
the transported metrics span

\[
 h=\dim\operatorname{span}\{G_l\},
\]

choose a basis `Ghat_a` and fold the corresponding coefficients into matrices
`Shat_a`; then `T_L` has `h` pair atoms.  For generic dense suffix products the
`G_l` are independent until `h=min(number of injections,n(n+1)/2)`, and the
appropriate Sym2-by-Sym2 catalecticant has the same generic lower rank.  Hence
there is no exact one-plane aggregation of many layer births.  The finite
compression receipt is the rank of that catalecticant, not the rank of any
single local `R_l`.

Causally updating one pair atom through a dense transition requires the two
conjugations `G<-WGW^T` and `R<-WRW^T`; with `h` live atoms the cost is
`O(h n^3)` per transition.  Since a new independent metric is born at each
source layer, naive exact forward propagation over `L` injections has
triangular `O(L^2 n^3)` work.  It collapses to `O(L n^3)` only under an exact
common-metric/low-catalecticant theorem or a resident conjugation that updates
the whole bank jointly.  Coordinatewise separable Jacobians preserve pair
form; a nonlinear reset can only increase the required response rank unless a
closed pair-product action is proved.

This yields the precise fixed-versus-causal boundary:

- one fixed seam and known suffix: forward pair propagation is an exact
  tensor-free GO;
- several injections with proportional/constrained transported metrics: rank-
  `h` GO after a finite catalecticant receipt;
- generic L5--14 births: independent metric factors accumulate, so the
  reverse-free construction is cost-closed absent fusion.

### Graph node: PRICE_Q211_FORWARD_PAIR_METRIC_RANK

- status: exact forward identity / multi-layer compression conditional;
- one source: two matrix conjugations per dense step, no q4 tensor;
- many sources: exact state `sum_l G_l odot S_l`, generic pair rank equals transported-metric/catalecticant rank;
- generic causal cost: triangular in injection depth;
- reopen: proportional metrics, low catalecticant, or literal joint conjugation reuse.

## Checkpoint 81: correction—CP4 primal factors are not a resident costate

Checkpoint 79's algebra is exact, but its ownership language needs a typed
correction.  The literal D11 CP4 bank is a **forward primal/source**
factorization.  The matrix `V` in

\[
 L=\sum_\alpha\lambda_\alpha v_\alpha^{\otimes4}
\]

is instead a factorization of the **future Bellman costate**.  Equality of the
two tensor formats does not identify their factors, directions, or time
orientation.  A resident forward CP4 push can be reused for Checkpoint 79 only
if a literal reverse-mode/self-duality theorem shows that the required costate
uses that same bank.

Likewise, old-span subtraction is global in response space.  For a source
innovation `s`, the typed object is

\[
 d_\perp=(I-\Pi_{old})F s,
\]

or equivalently a pulled costate `F^*(I-Pi_old)` in the scalar action.  In
general

\[
 F^*(I-\Pi_{old})\ne(I-\Pi_{\rm local\;CP4})F^*,
\]

so one may not residualize the local costate by simply deleting factors that
look CP4-like.  A novel strict-depth Price source can couple to a Waring-shaped
costate even though the incumbent owns a different CP4 source.

The correct promotion gates for the shared-Waring escape are therefore:

1. construct the complete-old-projected **endpoint response** or its Bellman
   costate;
2. prove that this costate, not the forward source bank, has a bounded shared
   Waring factorization;
3. prove literal reverse/forward factor reuse before subtracting its transport
   cost;
4. exclude only a same-site source already carried by CP4, rather than every
   costate of the same algebraic shape.

This supersedes the informal phrase in Checkpoint 79 that `L_perp` is obtained
by a local CP4 projector.  The arithmetic identity remains unchanged; the
residency/ownership claim is strictly weaker.

### Graph correction: PRICE_Q211_SHARED_WARING_ESCAPE

- exact formula unchanged;
- resident factor status: open—forward CP4 bank is not the future costate;
- old-span operation: endpoint/depth Schur projection, not local factor deletion;
- duplicate case: same-site CP4 source only, unless a stronger response-space identity is proved.

## Checkpoint 82: exact pair-atom Gram and a tensor-free coherence receipt

The pair-product representation also supplies its own exact cross-layer Gram.
For symmetric matrices `A,B,C,D`, contraction of the definition in Checkpoint
77 gives

\[
\boxed{
\langle A\odot B,C\odot D\rangle
={1\over6}\left[
 \langle A,C\rangle\langle B,D\rangle
 +\langle B,C\rangle\langle A,D\rangle
 +4\operatorname{tr}(D A C B)
\right].}
\]

The four-word trace is invariant under exchanging the two source pairs by
cyclicity plus reversal for symmetric matrices.  In particular,

\[
 \boxed{
 \|G\odot R\|_F^2
 ={1\over6}\left[
 \|G\|_F^2\|R\|_F^2+\langle G,R\rangle^2
 +4\operatorname{tr}(G^2R^2)
 \right].}
\]

Thus for the forward-propagated multi-injection state

\[
 T=\sum_lG_l\odot S_l,
\]

every Gram entry between layers is evaluable with ordinary matrix inner
products and one four-word trace—`O(n^3)`, not `O(n^4)`.  The exact total
Frobenius energy is

\[
 \|T\|^2=\sum_{l,m}
 \langle G_l\odot S_l,G_m\odot S_m\rangle.
\]

This is a white-box coherence receipt for the pair-source family.  It can
replace a worst-case triangle/elliptope bound when the Bellman endpoint metric
is literally the quartic Frobenius metric or when its known linear read has
already been applied to each pair atom.  It does **not** by itself identify
alignment with the ideal MSE target, and it does not reduce the causal
transport count in Checkpoint 80; it only removes uncertainty about mutual
source orientation once the endpoint responses exist.

### Graph node: PAIR_PRODUCT_GRAM_TRACE_IDENTITY

- status: exact tensor-free source Gram;
- cost: O(n3) per pair of propagated atoms;
- use: exact cross-layer coherence/RSS for a proved pair-product endpoint family;
- limit: no target alignment and no saving in constructing the transported atoms.

## Checkpoint 83: literal CP4 cohort reuse has an exact two-birth horizon

The deployed CP4 factor tape supplies one type-correct piece of Checkpoint 80's
**forward** pair propagation.  A coordinate birth appends an identity cohort
`U_s=I`.  The literal updates are

\[
 U\leftarrow W^TU,
 \qquad
 U\leftarrow D_{gate}U
\]

across the linear push and reset.  Hence for a cohort born at `s`,

\[
 \boxed{G_{s\to t}=U_{s\to t}U_{s\to t}^T}
\]

is exactly the transported metric of the Price pair source, in the same fp32
carrier convention.  This is a primal forward identity, so it does not make
the future Bellman costate resident and does not conflict with Checkpoint 81.

The retention horizon is sharp.  Each reset appends `n` coordinate factors and
the literal cap is `2n=2048`.  The first two complete cohorts fit without
truncation.  At the third birth the candidate has `3n` factors and the tape
keeps the top `2n` by

\[
 \lambda_\alpha^2\|u_\alpha\|^8.
\]

That ordering can retain an arbitrary mixture of cohorts; it gives no theorem
that all `n` columns of any older metric survive.  Since the unweighted metric
`UU^T` depends on every cohort column even when its quartic lambda is small,
the CP4 energy receipt does not bound the Price-metric error from a dropped
column.  Thus exact resident reuse covers at most two untruncated birth cohorts,
not the ten L5--14 injections.

Two further charges remain even within that horizon:

1. `_cp4_read_planes` forms lambda-weighted cubic/quartic powers of `U`; it does
   not materialize the unweighted Gram `UU^T`.  Building `G` is one additional
   width-by-cohort Gram unless the endpoint action consumes `U` directly.
2. the companion matrix `S=P R P^T` must still ride a proved covariance
   conjugation, and the endpoint pair action/global old projection must still
   be evaluated.  Resident metric factors solve neither operation.

Extending CP4 to retain ten complete cohorts would require at least `10n`
columns before any other CP4 atoms, so it is not a hidden use of the current
cap.  Compressing cohorts is exactly the transported-metric/catalecticant gate
of Checkpoint 80.

The already-frozen attribution receipt confirms that the two-cohort horizon is
not the broad carrier.  Its variance cells sum to `17.46`; the final two cells
are `L14_v=2.90` and `L15_v=2.42`, only `5.32/17.46=30.5%` of that linear
attribution.  Adding L13 raises the share to `8.12/17.46=46.5%`, but the third
cohort is exactly where the cap loses its retention guarantee.  These are
chain-attribution percentages, not a raw-MSE prediction, yet they rule out
describing exact two-cohort reuse as coverage of the measured L5--14 value.

### Graph node: CP4_PRICE_METRIC_COHORT_REUSE

- status: exact primal metric identity / horizon two cohorts;
- literal update: identity birth, dense factor push, row gate;
- exact metric: `G=UU^T` in carrier precision;
- cap no-go: third birth energy truncation gives no full-cohort guarantee;
- remaining work: unweighted Gram/read, companion covariance action, endpoint Schur projection;
- scope: forward source propagation only, not reverse-costate residency.

## Checkpoint 84: lambda-free tail certificate for dropped Price metrics

Although the CP4 energy order does not control an unweighted Price metric, the
factor bank present at the drop seam gives an exact independent receipt.  If
`U_D` contains the discarded columns, define

\[
 \Delta G=U_DU_D^T\succeq0.
\]

Then

\[
 \|\Delta G\|_F^2
 =\operatorname{tr}\big((U_D^TU_D)^2\big),
 \qquad
 \|\Delta G\|_F\le\operatorname{tr}\Delta G
 =\sum_{\alpha\in D}\|u_\alpha\|^2.
\]

For any symmetric companion source `S`, Checkpoint 82 gives its exact dropped
pair energy,

\[
\boxed{
\|\Delta G\odot S\|_F^2
={1\over6}\left[
 \|\Delta G\|_F^2\|S\|_F^2
 +\langle\Delta G,S\rangle^2
 +4\operatorname{tr}(\Delta G^2S^2)
\right].}
\]

The completely reduction-only bound is

\[
 \boxed{
 \|\Delta G\odot S\|_F
 \le \|\Delta G\|_F\|S\|_F
 \le\left(\sum_{\alpha\in D}\|u_\alpha\|^2\right)\|S\|_F.}
\]

There is also a conditional lower receipt because every term in the exact norm
is nonnegative:

\[
 \boxed{
 \|\Delta G\odot S\|_F
 \ge {1\over\sqrt6}\|\Delta G\|_F\|S\|_F.}
\]

At the first `3n -> 2n` truncation exactly `n` columns are dropped.  If their
minimum norm is `mu_D`, then

\[
 \|\Delta G\|_F
 \ge {\operatorname{tr}\Delta G\over\sqrt n}
 \ge\sqrt n\,\mu_D^2,
\]

and hence the local lost pair norm is at least
`sqrt(n/6) mu_D^2 ||S||F`.  An invertible separable suffix multiplies this by
the fourth power of its minimum singular value.  Converting that local lower
bound to final raw-MSE loss still requires a restricted minimum singular value
of the endpoint read and an anti-cancellation/Schur angle; without those, the
endpoint can annihilate a large tensor tail.

If a remaining separable suffix has product `P`, both pair factors are
conjugated, so

\[
 \|P\Delta GP^T\odot PSP^T\|_F
 \le\|P\|_{op}^4\|\Delta G\|_F\|S\|_F.
\]

Diagonal ReLU gates only multiply the same bound by their operator norms.  A
known endpoint read contributes its operator norm once more.  Summing these
radii across drop seams gives a rigorous target-free tail ceiling; storing the
dropped matrices instead permits the exact cross terms from Checkpoint 82.

This does not recover a discarded cohort.  It supplies the correct fail-fast
gate: if the propagated tail radius is below the remaining quartic RMS budget,
the cap is harmless for the Price route; if the lower/actual receipt is large,
the two-cohort shortcut cannot certify the broad carrier.  Lambda-weighted CP4
energy must not be substituted for this metric receipt.

### Graph node: CP4_DROPPED_PRICE_METRIC_RADIUS

- status: exact local norm and safe suffix radius;
- data: discarded factor columns, independent of CP4 lambdas;
- cheap bound: `(sum ||u_drop||²)||S||F` times suffix/read operator norms;
- use: upper harmlessness certificate and conditional lower close gate for the CP4 two-cohort approximation;
- limit: a small upper bound certifies harmlessness; otherwise no recovery or target alignment follows.

## Checkpoint 85: metric-trajectory compression is a distinct label-free gate

The failed low-rank compression of physical supply-error planes does not
mathematically settle the pair-metric trajectory in Checkpoint 80.  Let the
endpoint Price state be

\[
 T=\sum_{l=1}^L G_l\odot S_l.
\]

Choose an `h`-dimensional metric subspace and write

\[
 G_l=\sum_{a=1}^h c_{la}\widehat G_a+E_l.
\]

Then the retained tensor is exactly

\[
 T_h=\sum_{a=1}^h\widehat G_a\odot
 \left(\sum_lc_{la}S_l\right),
\]

and the omitted tensor is `sum_l E_l odot S_l`.  It has the safe label-free
radius

\[
 \boxed{
 \|T-T_h\|_F
 \le\sum_l\|E_l\|_F\|S_l\|_F.}
\]

The exact squared radius is also tensor-free:

\[
\boxed{
\begin{aligned}
\|T-T_h\|_F^2={1\over6}\sum_{l,m}\{&
\langle E_l,E_m\rangle\langle S_l,S_m\rangle\\
&+\langle S_l,E_m\rangle\langle E_l,S_m\rangle
+4\operatorname{tr}(S_mE_lE_mS_l)\}.
\end{aligned}}
\]

For the cheapest conformal approximation take

\[
 \alpha_l={\operatorname{tr}G_l\over n},
 \qquad E_l=G_l-\alpha_lI.
\]

Then

\[
 T_1=I\odot\sum_l\alpha_lS_l,
\]

so all births aggregate into one covariance-like plane.  The harmonic
counterexamples of Checkpoint 77 live exactly in the anisotropic residual
`E_l`.  A rigorous one-plane promotion therefore requires the propagated
anisotropy radius above—near-isotropy cannot be inferred from width or random-
matrix folklore.

Indeed the square iid-Gaussian model predicts a large residual.  If
`W_ij~N(0,1/n)` and `G=WW^T`, then exactly

\[
 \mathbb E\|G\|_F^2=2n+1,
 \qquad
 \mathbb E\min_\alpha\|G-\alpha I\|_F^2
 =n+1-{2\over n}.
\]

The ratio of these expectations tends to `1/2`, so the relative Frobenius
anisotropy tends to `1/sqrt(2)`, not zero.  Width alone therefore gives no
concentration toward the conformal branch; an actual structured-weight receipt
is mandatory.

More generally the unweighted best metric subspace comes from the tiny
`L by L` Gram matrix `K_lm=<G_l,G_m>`; the pair-weighted optimum uses the exact
quadratic form displayed above.  This is a new label-free fail-fast object,
not a target fit.  Passing the radius still requires a causal way to generate
the coefficients/bases within the bill; failing it closes common-metric
compression without constructing any q4 tensor.

### Graph node: PRICE_METRIC_TRAJECTORY_COMPRESSION_GATE

- status: exact h-metric decomposition and tail radius;
- h=1: conformal/isotropic aggregate `I odot sum alpha_l S_l`;
- certificate: exact pair-error Gram or safe sum-of-products bound;
- distinction: prior SVD NOGOs concern supply planes, not transported Jacobian metrics;
- deployment gate: low h plus a billed causal basis/coefficients; otherwise Checkpoint 80's rank growth remains.

## Checkpoint 86: cyclic-kernel H3 ownership depends on typed slots

Let an unsymmetrized mixed source `J_{ij,k}` be symmetric only in `(i,j)`, and
let `Cyc J=J_{ij,k}+J_{jk,i}+J_{ki,j}`.  Under the ordered-slot Euclidean
inner product and the convention

\[
 \kappa_3={2\over3}\operatorname{Cyc}J,
\]

the adjoint calculation gives `C*=2 Embed`, `CC*=4I`, and

\[
 P_{cyc}J={\kappa_3\over2}.
\]

Therefore

\[
 J_\perp=J-P_{cyc}J\in\ker(\operatorname{Cyc})
\]

pairs to zero with every fully symmetric H3 query.  A CP3 or threshold JVP
whose source is an actual symmetric cubic tensor can own only this cyclic
projection; the kernel is not a hidden symmetric H3 feature.

The exception is typed causal transport.  If the residual slot and the two AQ
slots pass through different operators before final symmetrization, `J` is a
distinguished-slot mixed moment, not yet a cubic cumulant, and a nonsymmetric
future query can see `J_perp`.  That requires a literal distinguished-slot
state/interface.  If the source is symmetrized first and all three slots then
share one linear transport, `J_perp` is annihilated at birth and cannot be
resurrected later.

Strict-all-distinct or equality-weighted coordinate measures change the
adjoint constants; using the induced weighted projector preserves the kernel
conclusion.

### Graph node: H3_CYCLIC_KERNEL_TYPED_SLOT_GATE

- status: exact ownership decomposition;
- symmetric H3: cyclic projection only, kernel identically invisible;
- possible novelty: distinguished residual slot with unequal causal transport;
- gate: literal unsymmetrized J state and matching nonsymmetric future query.

## Checkpoint 87: explicit cubic core of the r62 seam gate

For the half-AQ convention the weak cubic plane is

\[
 K_{ij}=\operatorname{tr}(A_i^2A_j)
 +u_i^TA_ju_i+2u_i^TA_iu_j.
\]

At a near-affine bank `A=tN`, the quadratic-matrix part of its Jacobian is

\[
 D_AK=F_U+t^2\mathcal M_N,
\]

where

\[
\boxed{
(\mathcal M_NH)_{ij}
=\operatorname{tr}\left[
(H_iN_i+N_iH_i)N_j+N_i^2H_j
\right].}
\]

For an oriented K21 dual plane `Lambda`, direct collection of every `H_k`
term gives

\[
\boxed{
(\mathcal M_N^*\Lambda)_k
=\sum_j\Lambda_{kj}(N_kN_j+N_jN_k)
 +\sum_i\Lambda_{ik}N_i^2.}
\]

This is the only genuinely cubic matrix word in the first fallback after the
affine K21 block; the other terms are the explicit Schur corrections from the
`u`-variation and tangent-covariance eliminations.

Those corrections have a closed form.  Write the full differential in block
notation

\[
 J(t)=
 \begin{pmatrix}
 D&tL_N\\
 tG_N&F_U+t^2\mathcal M_N
 \end{pmatrix},
\]

where factors inside `L_N,G_N,M_N` follow the selected half-AQ convention.
Let `P_T` and `Pi_U` project covariance rows onto the affine tangent and normal
spaces, let `P` be a right inverse of `D` on the tangent space, and let `R` be a
right inverse of `F_U`.  For `H0 in ker F_U`, take

\[
 h_1=-P P_TL_NH_0,
\]

and

\[
 H_2=-R\left(\mathcal M_NH_0-G_NP P_TL_NH_0\right).
\]

Then the exact reduced series is

\[
\boxed{
J_{normal}(t)H_0
=t\mathscr S_1H_0+t^3\mathscr S_3H_0+O(t^5),}
\]

with

\[
\boxed{
\mathscr S_1=\Pi_UL_N|_{\ker F_U},
\qquad
\mathscr S_3=-\Pi_UL_NR
\left(\mathcal M_N-G_NP P_TL_N\right)|_{\ker F_U}.}
\]

This formula replaces the placeholder `u/tangent Schur terms` below; it also
shows directly that gauge columns supply no quadratic escape.

Let `S1` be the first normal covariance Schur map, let `Z` lie in its cokernel,
and choose `Lambda_Z` so that `S1*Z=F_U*Lambda_Z` in the quotient convention.
For a leading kernel direction

\[
 H_0\in\ker F_U\cap\ker\mathscr S_1,
\]

the direct cubic pairing is

\[
 -\langle\Lambda_Z,\mathcal M_NH_0\rangle
 =-\langle\mathcal M_N^*\Lambda_Z,H_0\rangle,
\]

followed by the stated tangent Schur projection.  Thus a finite symbolic cubic
certificate is obtained by choosing bases `{Z_b}` and `{H_a}` and testing the
row rank of the corrected matrix

\[
 \boxed{
 \mathcal C_{ba}=\langle Z_b,\mathscr S_3H_a\rangle,}
\]

where the displayed `M_N*Lambda` term is the direct cubic core of this exact
Schur expression.

At r62 it needs row rank561.  The exact parity

\[
 \Psi(U,-A)=(C(U,A),-K(U,A))
\]

explains why the normal Schur series contains odd powers `t,t^3,t^5,...` and
why no quadratic rescue exists.  It does **not** annihilate the displayed
Jordan-square word, so parity alone supplies no fifth-order no-go.  A universal
cubic obstruction, if present, must be a genuine syzygy placing every
projected `M_N*Lambda_Z` in the affine/first-Schur row span.

### Graph node: R62_CUBIC_SCHUR_EXPLICIT_CORE

- status: exact M_N/M_N* formula; full 561-row rank certificate open;
- parity: even orders forbidden, cubic is first possible lift;
- finite gate: corrected projected matrix C_ba on ker F intersect ker S1;
- close condition: a nonzero cokernel Z whose corrected cubic adjoint vanishes on every kernel direction.

## Checkpoint 88: the r62 Jordan core has no universal 561-mode syzygy

The forced first-order invisible space at r62 has dimension `rho=33`; its
target is `Sym^2 R`, of dimension561.  The cubic adjoint in Checkpoint 87
contains the Jordan catalecticant of the matrices carried by those invisible
physical combinations.  Abstract it as

\[
 \mathcal J_B:\operatorname{Sym}^2R\to\operatorname{Sym}_r,
 \qquad
 \mathcal J_B(Z)=\sum_{a,b}Z_{ab}(B_aB_b+B_bB_a).
\]

There is no universal rank defect in this core when `rho<=r`.  Choose linearly
independent vectors `v_1,...,v_rho in R^r` with every inner product
`v_a^Tv_b` nonzero and put

\[
 B_a=v_av_a^T.
\]

Then

\[
 B_aB_b+B_bB_a
 =(v_a^Tv_b)(v_av_b^T+v_bv_a^T).
\]

The matrices `{v_av_b^T+v_bv_a^T:a<=b}` form a basis of the symmetric
operators on `span{v_a}`, and the displayed nonzero scalars preserve linear
independence.  Therefore

\[
 \boxed{\operatorname{rank}\mathcal J_B={\rho(\rho+1)\over2}.}
\]

At r62, rho33, this is exactly561.  Moreover the first-order condition places
the invisible combinations `B_a` in the Veronese-visible complement `E1` of
`ker Q_U`, whose dimension is `n=1024`; an arbitrary prefix bank can assign 33
such combinations while using its `E0` block to attain the maximal rank929.
Thus neither latent dimension nor Jordan multiplication forces any of the 561
modes to fifth order.

This is not yet a proof that the **full** cubic Schur map is onto.  The
constructed Jordan basis must survive the affine K21 right inverse, the
`G_N P P_T L_N` correction, and the normal projection in Checkpoint 87.
The remaining finite certificate is precisely the 561-row minor of that full
map.  The theorem does, however, close a broad class of proposed no-gos: a
failure cannot be blamed on parity, `rho>r`, commutativity, or insufficient
Jordan-pair dimension.  It must be a compatibility syzygy in the surrounding
Schur blocks or a failure of the actual D14 prefix tuple.

### Graph node: R62_JORDAN_CUBIC_CORE_FULL_RANK

- status: explicit rank-561 Jordan-core construction / full Schur minor open;
- witness: 33 nonorthogonal independent rank-one matrices in Sym62;
- consequence: no universal fifth-order obstruction from the cubic matrix algebra;
- remaining gate: survival through F-right-inverse, tangent correction, and normal projection at the actual cut.

## Checkpoint 89: a 20-product dense reverse is not economically closed by itself

The arithmetic `cap49` screens a narrow shell; it must not be promoted to a
universal cost no-go when the candidate raw gain is large.  The billing audit
gives approximately

\[
 \delta u=0.00161
\]

per f32 width-1024 matrix-product equivalent.  A ten-transition dense reverse
with two products per transition therefore adds

\[
 \Delta u\simeq0.0322.
\]

Relative to the dtype-packed estimator, its raw break-even is only

\[
 \boxed{g_{break}=1-{u_{pack}\over u_{pack}+0.0322}.}
\]

For `u_pack=0.5629` this is `5.41%`; for the panel mean `u_pack=0.5709` it is
`5.34%`.  Thus any source with a credible raw gain above roughly5.5% pays for
the dense reverse before considering other utility reductions.

Against bank `u_bank=0.6065`, a raw gain `g` co-fired with the dtype pack and
20-product reverse has adjusted-score ratio

\[
 \boxed{R_{adj}=(1-g){u_{pack}+0.0322\over0.6065}.}
\]

For the certified `g=18.7%` chain, this is about `0.7977` at the r35 utility
and `0.8084` at the panel-mean utility: a `20.2%` or `19.2%` adjusted gain,
respectively.  The 20% fire line requires `g>=18.47%` in the first convention
or `g>=19.55%` in the second.  Small additional utility packing or raw value
decides the borderline.

This is an economic reopen, not a source theorem.  A reverse pass still needs
the correct future costate, reset adjoints, and a deployable local innovation;
the certified 18.7% truth-fed chain does not prove those.  But once a
source/value certificate reaches this scale, rejecting the route because it
exceeds a one-equivalent headroom is mathematically wrong.  For the campaign
goal `raw<1e-8` (roughly a two-thirds raw reduction), substantially larger
white-box adjoint bills can be rational if they acquire the missing physics.

There is a binding type caveat.  The 20-product count is for **one fixed scalar
endpoint functional** with one dense matrix costate.  The deployed estimator
must output an `n`-vector and has no label-derived terminal residual with which
to seed that scalar adjoint.  A target-free deployable correction therefore
needs either a proved one/few-head endpoint factorization, a multi-output
reverse with its rank bill, or the original forward tangent/carrier.  Without
such a factorization the cheap reverse is an off-card grading/certification
device, not a prediction feature.  The economic theorem says cost is not the
obstruction **after** typing; it does not remove this endpoint-rank barrier.

### Graph node: DENSE_REVERSE_ADJUSTED_ECONOMIC_REOPEN

- status: exact billing inequality / source and implementation open;
- ten-layer reverse increment: about0.0322 utility;
- packed-estimator raw break-even: about5.4%;
- certified-chain scale: about19.2–20.2% adjusted improvement versus bank;
- implication: cost is conditional on raw value, not a universal cap49 close.
- type gate: one/few target-free endpoint heads; a label-seeded scalar adjoint is not deployable.

## Checkpoint 90: ReLU Schur-multiplier Kraus rank blocks a one-matrix suffix

For a covariance perturbation, the exact local Gaussian/ReLU Jacobian has a
Schur-multiplier block

\[
 \mathcal S_\Gamma(X)=\Gamma\odot X,
\]

followed or preceded by a dense covariance conjugation.  If `Gamma` is PSD and

\[
 \Gamma=\sum_{a=1}^qv_av_a^T,
\]

then

\[
 \boxed{
 \Gamma\odot X=\sum_{a=1}^qD_{v_a}XD_{v_a}.}
\]

The minimum possible number of diagonal congruence terms is exactly
`q=rank Gamma`: any such representation is a Gram factorization of `Gamma`.
After a dense map each term has its own path matrix.  Across several resets,
the generic Kraus/path count multiplies until it reaches the ambient
superoperator rank.  A single suffix matrix `P` satisfying

\[
 X\mapsto PXP^T
\]

therefore exists only in the rank-one/conformal gate branch or on a specially
annihilated source subspace.

Low latent AQ rank does not imply low physical Kraus rank.  If

\[
 K=UU^T,\qquad
 \Gamma_M=\sum_{m=0}^Mc_mK^{\odot m},
\]

then `K^{odot m}` has the symmetric-power factorization

\[
 K^{\odot m}=U^{\odot m}(U^{\odot m})^T,
\]

with generic feature rank

\[
 \min\left(n,{r+m-1\choose m}\right).
\]

For r49, the constant, linear, and quadratic feature counts already total
`1+49+1225>1024`; with generic nonzero coefficients a biased gate is full
rank by degree2.  At exactly centered threshold the indicator kernel has only
the constant plus odd arcsine powers, so the linear truncation has rank at most
50, but the cubic feature count `{51 choose3}=20,825` makes the exact physical
kernel generically full rank.  Finite width therefore does not create a
bounded exact Kraus family.

This cleanly separates three mechanisms:

1. **Forward aggregate covariance carrier.**  Because all source planes are
   added before applying the same linear Schur/conjugation operator, one dense
   `n by n` plane propagates every birth in two matrix products per layer.  This
   is the correct exact causal architecture if local source planes are known.
2. **One/few suffix matrices.**  Valid only after a certified low-Kraus gate
   approximation, with multiplicative depth error/rank control.
3. **Scalar reverse.**  Cheap for a fixed endpoint query, but subject to the
   endpoint-rank/type gate of Checkpoint89.

Thus the broad variance-channel problem remains source acquisition, not
covariance transport: Site-A/B insertion into the already-paid forward plane
is algebraically optimal.  Replacing it by a direct one-sided suffix formula
does not evade the gate kernel.

### Graph node: RELU_COVARIANCE_SCHUR_KRAUS_RANK_BARRIER

- status: exact minimal Kraus-rank theorem;
- rank: `rank Gamma`, generically n by degree2 (biased) or degree3 (centered) at r49;
- no-go: one suffix matrix through multiple exact resets;
- exact GO: aggregate all source planes in the resident covariance carrier;
- approximate gate: low-Kraus truncation plus depthwise tail certificate.

## Checkpoint 91: collision subtraction explodes one Waring ray to rank C(n,2)

The shared-Waring escape has a decisive ordering constraint: contract first,
then subtract old response.  Tensor-level collision projection can destroy the
shared bank even for one ray.

Let `v_i` be nonzero and let `P_1111` retain only entries of `v^tensor4` with
four pairwise-distinct indices.  Flatten on unordered off-diagonal pair indices

\[
 \mathcal P=\{\{i,j\}:i<j\},\qquad |\mathcal P|={n\choose2}.
\]

The flattening is

\[
 \boxed{M=D_vA_{KG(n,2)}D_v,}
\]

where `(D_v)_{\{i,j\}}=v_iv_j` and two pair vertices are adjacent in the
Kneser graph exactly when they are disjoint.  The three Kneser eigenvalues are

\[
 {n-2\choose2},\qquad -(n-3),\qquad1,
\]

all nonzero for `n>=4`.  Hence

\[
 \operatorname{rank}M={n\choose2}.
\]

Every sum of `r` fourth powers has `(2,2)` flattening rank at most `r`, so

\[
\boxed{
\operatorname{wrank}(P_{1111}v^{\otimes4})
\ge {n\choose2}.}
\]

At width1024 this is523,776.  The more relevant quotient that keeps both
`211` and `1111` but removes `22` has the same lower bound: its off-pair
flattening is

\[
 \operatorname{Diag}(z)(\mathbf1\mathbf1^T-I)\operatorname{Diag}(z),
 \qquad z_{\{i,j\}}=v_iv_j,
\]

which is full rank on all `{n choose2}` pair coordinates.

Thus neither a low-rank full tensor nor a low-rank incumbent CP4 bank implies a
low-rank complete-collision residual.  A local projector preserves a shared
bank only under the explicit invariant-subspace condition

\[
 \Pi_{old}\operatorname{span}\{v_\alpha^{\otimes4}\}
 \subseteq\operatorname{span}\{v_\alpha^{\otimes4}\},
\]

which is not true for the equality-sector projector above and is not known for
the endpoint/corpus Schur projector.

The surviving order of operations is exact **action-level** subtraction:
contract the unprojected new/dropped ray locally against the complete future
query, compute its old scalar/response contribution in the same convention,
subtract there, and retire the ray.  Attempting to transport
`P_old_perp(v^tensor4)` as a small Waring state is generically impossible.

For completeness, coordinate births themselves also grow exact rank.  After
`T+1` generic full birth cohorts,

\[
 L_T=\sum_{s=0}^T\sum_i d_{s,i}v_{s,i}^{\otimes4}
\]

has `(2,2)` catalecticant rank and symmetric Waring rank exactly `(T+1)n`
while this is at most `n(n+1)/2`.  Ten cohorts at width1024 therefore have
generic exact rank10,240, far below the ambient ceiling but far above cap2048.

The coordinate-reset Hessian action remains tensor-free **before** projection:

\[
 \boxed{
 \mathsf C_G(D^{\otimes4}L+\sum_id_ie_i^{\otimes4})
 =D\mathsf C_{DGD}(L)D
 +\operatorname{Diag}(d\odot\operatorname{diag}G).}
\]

This identity is the algebraic basis of the contract-and-retire attempt; it
does not make the needed future query resident.

### Graph nodes

- `PRICE_COLLISION_PROJECTION_WARING_EXPLOSION`: one dense ray becomes Waring rank at least C(n,2) after 1111 or combined 211+1111 collision projection;
- `PRICE_SHARED_WARING_RESET_RANK_GROWTH`: generic full coordinate births add exactly n catalecticant/Waring rank per reset;
- `PRICE_SHARED_WARING_NOVEL_RESIDUAL_GATE`: only pre-projection local contract-and-retire can preserve the cheap algebra; future-query residency and response-space subtraction remain open.

## Checkpoint 92: r62 cubic gate reduces to one quotient projection

The full Schur cubic minor can be tested without carrying either right inverse.
Use the exact block notation of Checkpoint 87 and put

\[
 \mathcal A_N=\mathcal M_N-G_NP P_TL_N.
\]

For `Z in coker S1`, choose `Lambda_Z` from the adjoint relation

\[
 L_N^*Z=F_U^*\Lambda_Z
\]

in the normal/quotient convention.  Since `F_U R=I`, direct adjoint
cancellation gives, for every

\[
 H\in\ker F_U\cap\ker\mathscr S_1,
\]

the exact identity

\[
\boxed{
 \langle Z,\mathscr S_3H\rangle
 =-\langle\Lambda_Z,\mathcal A_NH\rangle.}
\]

Therefore a nonzero first-order cokernel mode is forced to fifth order iff

\[
\boxed{
 \mathcal A_N^*\Lambda_Z
 \in\operatorname{range}F_U^*+L_N^*\mathcal N_U,}
\]

where `N_U` is the covariance-normal dual space.  This is the complete cubic
syzygy; the arbitrary right inverses used to write a section do not enter the
minor.

There is an even cleaner sufficient certificate.  Restrict source directions
to

\[
 \mathcal K_0=\ker F_U\cap\ker L_N.
\]

Then the tangent correction vanishes and `A_N H=M_N H`.  At r62 the ambient
quadratic-parameter dimension is

\[
 1024\cdot{62\cdot63\over2}=1,999,872,
\]

so purely by codimension

\[
\boxed{
 \dim\mathcal K_0
 \ge1,999,872-1,048,576-524,800=426,496.}
\]

Choose bases `{Z_b}_{b=1}^{561}` of the forced first-order cokernel and any
`H_a in K0`.  If

\[
 \boxed{
 C_{ba}=\langle\mathcal M_N^*\Lambda_{Z_b},H_a\rangle}
\]

has row rank561, the **full** cubic Schur map is onto.  Equivalently the 561
vectors `M_N*Lambda_Zb` must be independent after projection modulo
`range F_U*+range L_N*`.  This is exactly the compatibility condition left by
Checkpoint88's full-rank Jordan core; no `P`, `R`, or tangent bookkeeping
remains.

### Graph node: R62_CUBIC_QUOTIENT_CERTIFICATE

- status: exact right-inverse-free equivalence / explicit 561-minor open;
- syzygy: `A_N*Lambda_Z in range F* + L* N_U`;
- strong sufficient domain: `K0=kerF intersect kerL`, dimension at least426,496;
- certificate: rank561 of `<M_N*Lambda_Zb,H_a>`;
- consequence: only a genuine quotient dependence can force fifth order.

## Checkpoint 93: an incumbent D10 query cannot contract-and-retire novelty

Action-level subtraction avoids the Waring-rank explosion, but the literal
future query must come from the enhanced dynamics.  Let

\[
 \mathcal F_{old}:\mathcal T_4\to\mathcal Y
\]

be the complete D10/CP4/Q24 future response from a local quartic source to the
endpoint, including every incumbent equality read and depth transition.  A
literal D10 reverse exposes functionals only in

\[
 \operatorname{range}\mathcal F_{old}^*.
\]

For any endpoint covector `lambda` and ray `T`, it computes

\[
 \langle\mathcal F_{old}^*\lambda,T\rangle
 =\langle\lambda,\mathcal F_{old}T\rangle.
\]

If the same old response is then subtracted at action level, the remainder is
identically zero.  In the even sharper local convention, if `O_old` is the
same-site K4/K31/K22 read and `T_perp in ker O_old`, then every literal local
D10 query obeys

\[
 \boxed{
 \langle O_{old}^*\lambda,T_\perp\rangle=0.}
\]

Hence a retained/dropped CP4 ray cannot acquire a novel all-distinct value by
contracting it only against the incumbent D10 reverse and then Schur-
subtracting the incumbent action.  That procedure either reproduces an old
response or returns zero.

Strict depth does not contradict the theorem.  A locally old-dark ray can
become visible after a **new** all-distinct/q211 transition, but then the
required future operator is

\[
 \mathcal F_{new}\ne\mathcal F_{old},
\]

and its adjoint is precisely the missing quartic query.  The lawful
contract-and-retire scalar is

\[
 \langle\lambda,
 (I-\Pi_{old})\mathcal F_{new}T\rangle,
\]

which cannot be reconstructed from `F_old*lambda` without an intertwiner or a
new response state.

The minimal reopen can be only one step, not a full quartic reverse.  If there
is a typed local bridge

\[
 J_l^{new}:\mathcal T_4\to\mathcal X_{old,l+1}
\]

into a state that D10 already carries, then

\[
 \mathcal F_{new}
 =\mathcal F_{old,l+1\to L}J_l^{new},
 \qquad
 \mathcal F_{new}^*\lambda
 =(J_l^{new})^*\lambda_{l+1}^{old}.
\]

In that case the resident old reverse **after the bridge** plus one exact local
new adjoint is sufficient for contract-and-retire.  This is precisely the
typed H211/q211 source-to-covariance bridge that must be derived and billed.
If the ray remains future-dark for more than one novel transition, no such
factorization exists and the larger new query state returns.

### Graph node: D10_CONTRACT_RETIRE_NOVELTY_ZERO

- status: exact row-space/adjoint no-go for literal incumbent query;
- incumbent contraction: reproduces only `F_old T`;
- after matching old subtraction: zero additive novelty;
- strict-depth escape: a proved one-step new bridge into the old D10 state, or otherwise the enhanced future adjoint;
- implication: action-level ordering avoids rank explosion but does not supply the missing query.

## Checkpoint 94: the r62 cubic gate has a large intrinsic silent test space

Checkpoint92 removed every arbitrary right inverse from the cubic Schur gate.
The remaining 561-dimensional question can be made still more literal.  Let

\[
 E_0=\ker Q_U\subset\operatorname{Sym}^2\mathbb R^{62},
 \qquad
 Q_U(H)=(u_j^T H u_j)_{j=1}^{1024}.
\]

For a regular affine bank, `Q_U` is onto, hence

\[
 \dim E_0={62\cdot63\over2}-1024=929.
\]

For each unit define the local silent subspace

\[
 D_i=\{H\in E_0:H u_i=0\}.
\]

The vector equation `H u_i=0` imposes at most62 additional linear
conditions, so

\[
 \boxed{\dim D_i\ge929-62=867.}
\]

If a variation has `H_i in D_i` for every unit, then its affine weak-interface
variation vanishes without cancellation across units: `H_i in E0` kills every
Veronese evaluation `u_j^T H_i u_j`, while `H_i u_i=0` kills the distinguished
local bilinear terms.  Thus

\[
 \bigoplus_iD_i\subset\ker\mathcal F_U.
\]

Intersecting also with the linear-in-`A` block gives

\[
 \mathcal K_{sil}=\left(\bigoplus_{i=1}^{1024}D_i\right)
                  \cap\ker\mathcal L_N
 \subset \ker\mathcal F_U\cap\ker\mathcal L_N=\mathcal K_0,
\]

and therefore

\[
 \boxed{
 \dim\mathcal K_{sil}
 \ge1024\cdot867-524,800=363,008.}
\]

On this space both the affine constraint and the tangent correction vanish,
so the complete cubic action is exactly the intrinsic matrix-Jordan block
`M_N`.  For a basis `{Z_b}` of `coker S1`, choose `Lambda_Zb` from
`L_N* Z_b=F_U* Lambda_Zb`, and form

\[
 G_{bc}=\left\langle
 \Pi_{sil}\mathcal M_N^*\Lambda_{Z_b},
 \Pi_{sil}\mathcal M_N^*\Lambda_{Z_c}
 \right\rangle .
\]

Then

\[
 \boxed{G\succ0\quad\Longrightarrow\quad
        \operatorname{rank}S_3=561.}
\]

Equivalently, any 561 silent variations whose pairing matrix with the
`M_N*Lambda_Zb` has nonzero determinant are an exact finite receipt for the
full cubic seam.  This is stronger than a raw dimension count: it tests the
actual constrained quotient and cannot be invalidated by a choice of right
inverse.  What remains open is an explicit lawful prefix bank `(U,N)` making
this Gram positive definite.

### Graph node: R62_CUBIC_SCHUR_SILENT_GRAM_CERTIFICATE

- status: exact finite certificate / explicit positive minor open;
- intrinsic domain: `K_sil=(direct-sum D_i) intersect ker L`, dimension at least363,008;
- certificate: the 561 by561 projected Gram of `M*Lambda_Z` is positive definite;
- consequence: no ambient-dimension or tangent-compatibility obstruction remains at cubic order.

## Checkpoint 95: exact one-step CP4-ray action fusion and its limit

Let `O_l:T4->(K4,K31,K22)` be the literal same-site equality read.  As in
Checkpoint93, a complete collision-old residual `Tcircle in ker O_l` has zero
pairing with every same-site D10 query.  After one genuine strict-depth
quartic bridge, however, a transported rank-one ray can be contracted without
building a fourth-order tensor.

For a transported factor `f`, a retained scalar weight `lambda`, and the
literal next-cut K31 reverse matrix `Q` (with the read convention absorbed
into `Q`), the scalar is

\[
 \boxed{
 \rho=\lambda\,(f^{\odot3})^TQf.}
\]

For a resident factor bank `F=[f_1,\ldots,f_R]`, all rays batch as

\[
 a=\operatorname{colsum}\bigl(F^{\odot3}\odot(QF)\bigr),
 \qquad \rho=\lambda^Ta.
\]

Thus one dense `QF` product contracts every resident ray.  Coordinate births
are the special case `F=I`; all `n` birth actions batch in one product.
This is complete only for the declared **next K31 read into the resident D10
covariance-tangent path**.  It is not the complete q211/q1111 future adjoint.

The old subtraction must be done on the summed endpoint response,

\[
 z_\perp=(I-\Pi_{old}^{V})z,
\]

not by deleting collision faces from each ray.  This preserves the low-rank
factorization and avoids Checkpoint91's Waring explosion.  It also exposes an
important ownership identity: a coordinate birth
`Delta(d)=sum_i d_i e_i^{\otimes4}` is exactly its diagonal Star4
representation at birth and remains equal under affine transport.  Therefore
its first-affine action is old-owned; novelty can begin only after nonlinear
CP-versus-incumbent reconstruction paths diverge.

If `R_l` factors are resident at each active layer, the marginal contraction
cost scales as

\[
 C_{new}=\sum_l(c_{push}+c_{act}){R_l\over n}.
\]

Ten full resident batches already cost at least10 GEMM-equivalents; if dropped
factors require a fresh push, at least20.  With only about one unit of
headroom in the r62 shell, the additive multi-layer route is therefore closed
unless the action shares a literal paid product or the total weighted retained
rank across layers is at most `n`.  The algebraic fusion is real, but it is a
one-leg building block rather than a free complete carrier.

### Graph nodes: PRICE_D10_RAY_ACTION_FUSION and PRICE_CONTRACT_RETIRE_ADDITIVE_COST_NOGO

- exact GO: one transported factor bank plus a literal next-cut K31 query, one `QF` product;
- ownership: same-site collision residual is zero; coordinate birth is Star4-old under affine transport;
- complete subtraction: endpoint response Schur projection, never tensor-face deletion;
- no-go: ten full additive batches cost at least10, above r62 headroom absent exact product reuse;
- open escape: a paid resident `QF`, bounded total rank, or a proved multi-leg bridge into the old D10 state.

## Checkpoint 96: causal liveness and the exact universal sketch lower bound

The factor-action identity in Checkpoint95 is algebraically local but not
automatically schedulable.  A dropped CP4 factor is known during the forward
selector; its strict-future D10 query is known only after the later masks have
been completed and the reverse sweep reaches the cut.  Exact evaluation must
therefore do at least one of:

1. retain the discarded factor banks until reverse time;
2. retain/reconstruct every selector decision and replay the forward CP4 path;
3. exhibit a literal two-pass buffer schedule in which the same factors and
   future query coexist.

One-pass discard followed by a future contraction is impossible by causality;
recomputability is not residency.

There is also a sharp information statement.  For a fixed post-drop factor
bank `{(lambda_a,f_a)}` the next-cut K31 action against a symmetric query `Q`
is

\[
 \rho(Q)=\sum_a\lambda_a(f_a^{\odot3})^TQf_a
        =\langle Q,S_{31}\rangle,
\qquad
 S_{31}=\operatorname{Sym}\sum_a\lambda_a f_a^{\odot3}f_a^T.
\]

Thus `S31` is the minimal linear sufficient statistic for **all** next-cut
symmetric `Q`: a sketch may be smaller only after proving that future queries
lie in a known proper subspace.  Before an arbitrary future dense transport
`P`, the family becomes

\[
 \rho(P,Q)=\left\langle
             \operatorname{Sym}\sum_a\lambda_af_a^{\otimes4},
             P^{\otimes4,*}L_Q
           \right\rangle .
\]

Taking a one-row query makes the right factor `x^{otimes4}` for arbitrary
`x`; those rays span all of `Sym4(R^n)`.  Consequently a universal pre-
transport **linear** sketch must determine the entire quartic Waring sum.
On the rank-`R` secant manifold its local information dimension is at least
`Rn` (until ambient saturation).  An explicit witness is the coordinate bank
`f_a=e_a`: its differentials

\[
 \operatorname{Sym}(e_a^{\otimes3}\otimes h_a)
\]

have disjoint `aaab` supports and total rank `Rn`.  Hence

\[
 \boxed{
 \dim(\text{universal exact causal sketch})
 \ge\min\{Rn,{n+3\choose4}\}.}
\]

The ordinary `nR` factor bank is therefore locally information-optimal for a
generic unknown future dense query/transport.  Low-rank `Q` alone does not
help before its directions are known.  Exact compression is possible only
for a predeclared query space `Qcal`, a shared factor/eigen dictionary, sparse
transport, or a replay schedule.  If `Qcal` has basis `{Q_b}_{b=1}^d` at the
actual cut, the `d` actions `<Q_b,S31>` are sufficient **after transport**;
this does not yield a closed pre-transport statistic unless that query space
is invariant under the intervening fourth-order pullback.

### Graph node: PRICE_CONTRACT_RETIRE_CAUSAL_LIVENESS_AND_SKETCH_NOGO

- status: exact scheduling and information lower bound;
- causal requirement: store, replay, or two-pass fuse every dropped factor needed by a later reverse query;
- generic lower bound: at least `Rn` real degrees before arbitrary dense future transport;
- only escapes: a proved invariant low-dimensional query space, structured transport, shared dictionary, or literal replay;
- implication: the one-product batch formula is not an online discard sketch.

## Checkpoint 97: even a successful r62 cubic seam is intrinsically `t^-3` conditioned

The 561-minor in Checkpoints92/94 is an existence certificate, not a stability
certificate.  After eliminating the regular affine blocks and the range of
the first Schur map, the forced-dark normal operator has the parity expansion

\[
 \widehat S(t)=t^3\widehat S_3+t^5\widehat S_5+\cdots,
 \qquad
 \widehat S_3:ker S_1\longrightarrow\operatorname{coker}S_1,
\]

where `dim coker S1=561`.  If the cubic certificate passes and

\[
 \sigma_3=\sigma_{\min}(\widehat S_3|_{K_c})>0
\]

on some 561-dimensional complement `K_c`, then for a fifth-order remainder
bound `||R_5(t)||<=M_5`,

\[
 |t|^3(\sigma_3-t^2M_5)
 \le \sigma_{\min}(\widehat S(t)|_{K_c})
 \le |t|^3(\|\widehat S_3\|+t^2M_5).
\]

In particular, whenever `t²M5<=sigma3/2`, the dark inverse has the exact
scaling

\[
 \boxed{
 \|\widehat S(t)^+\|
 \asymp {|t|^{-3}\over\sigma_3},
 \qquad
 \|\widehat S(t)^+\|
 \ge {1\over |t|^3(\|\widehat S_3\|+t^2M_5)}.}
\]

Thus a dark seam displacement `delta z` requires parameter motion at least

\[
 \|\delta\theta\|
 \ge {\|\delta z\|
       \over |t|^3(\|\widehat S_3\|+t^2M_5)}.
\]

Equivalently, a trusted parameter ball of radius `Rtheta` can cover at most a
dark target ball of radius

\[
 R_z\le R_\theta |t|^3(\|\widehat S_3\|+t^2M_5).
\]

This is a universal consequence of the affine rank obstruction; no choice of
coordinates or right inverse removes it.  At r62 the normal target dimension
is

\[
 {962\cdot963\over2}=463,203,
\]

with first-order rank462,642 and cubic cokernel561.  Any full normal square
minor therefore has leading order at least

\[
 t^{462,642+3\cdot561}=t^{464,325},
\]

an excess order `2*561=1,122` beyond a regular first-order seam.  The useful
finite gate is not merely `det G !=0`; it is a lower bound on `sigma3` together
with the actual nonaffine scale `|t|` and a fifth-order radius.  A tiny
quadratic prefix can be algebraically onto yet unusably unstable.

### Graph node: R62_CUBIC_SECTION_CONDITIONING_GATE

- status: exact asymptotic conditioning theorem;
- unavoidable scale: dark inverse norm grows as `|t|^-3/sigma3`;
- determinant order: at least464,325, including1,122 excess powers from the 561 cubic modes;
- promotion gate: explicit `sigma3`, actual cut nonaffine scale, fifth-order bound, and target displacement radius;
- implication: cubic surjectivity alone does not certify a stable D14-to-AQ section.

## Checkpoint 98: r63 buys a two-power conditioning improvement for 0.638 products

The terminal-factored five-reset frontier gives

\[
 C_{62}=47.43295,
 \qquad
 C_{63}=48.07040,
 \qquad
 \Delta C=0.63745.
\]

Fix the perturbation gauge by `||N||=1`, so `|t|` is the physical quadratic
size of the cut.  At r62, every lawful local section must recover the final
561 modes through the cubic block and hence has dark inverse scale

\[
 \kappa_{62,dark}\asymp {1\over |t|^3\sigma_{3,62}}.
\]

At r63 the Veronese-null obstruction disappears.  Conditional on an onto
first Schur minor with smallest singular value `sigma1,63`, the corresponding
scale is only

\[
 \kappa_{63,dark}\asymp {1\over |t|\sigma_{1,63}}.
\]

Therefore

\[
 \boxed{
 {\kappa_{62,dark}\over\kappa_{63,dark}}
 \asymp {\sigma_{1,63}\over\sigma_{3,62}}|t|^{-2}.}
\]

The extra arithmetic is economically tiny.  With the billing audit's
`0.00161` absolute utilization per f32 product-equivalent,

\[
 \Delta u\simeq0.00161\cdot0.63745=0.001026.
\]

At the dtype-pack scale `u=0.5629`, equal-raw adjusted cost differs by only

\[
 {0.001026\over0.5629}=0.182\%.
\]

So if a new oriented AQ cut view must be built at all, r62 is not the default
economy point: r63 dominates whenever its first-order section is regular and
the resulting raw/fidelity improvement exceeds about0.18%.  This is a
conditional architecture comparison, not a claim that the r63 seam exists;
both still require an actual lawful cut tuple and resident/cold-anchor receipt.
But the burden reverses: r62 should be chosen only if its measured cubic
conditioning and extra 0.93-equivalent headroom are worth the `t^-2`
stability loss.

### Graph node: R62_VS_R63_SECTION_FRONTIER

- status: exact conditional stability/cost comparison;
- r62: cubic dark scale `t^-3`, cost47.43295;
- r63: possible first-order dark scale `t^-1`, cost48.07040;
- marginal billing:0.63745 product-equivalents, about0.182% at u0.5629;
- decision gate: actual r63 first-order singular value versus r62 cubic singular value and physical cut amplitude.

## Checkpoint 99: the forward Q24 polynomial has a cheap Price derivative, not a resident reverse

At fixed marginal coefficients, the literal seven-term Gaussian pair kernel
has the Hermite form

\[
 K_{ij}^{\le6}(\rho)
 =\sum_{m=1}^{6}{c_{i,m}c_{j,m}\over m!}\rho^m,
\]

after removing the mean-outer `m=0` term.  Its exact correlation derivative is

\[
 \boxed{
 \Gamma_{ij}^{\le5}(\rho)
 ={\partial K_{ij}^{\le6}\over\partial\rho}
 =\sum_{m=0}^{5}{c_{i,m+1}c_{j,m+1}\over m!}\rho^m.}
\]

Therefore, if a reverse covariance query `Lambda+` is already supplied at the
same unclipped fixed-marginal seam, the local Gaussian reset pullback is the
elementwise operation

\[
 \bar\rho=\Gamma^{\le5}\odot\Lambda^+.
\]

The forward `_q24_context` really does form powers `rho^0,...,rho^6`, so this
derivative can be recomputed with no new dense matrix product.  It is a valid
replacement-tape optimization.

It is **not** a residency theorem for the current D10/D11 package.  The literal
estimator is an autonomous forward loop: the Q24 powers are workspace reused
at each layer, `_connected_planes` forms forward equality planes, and
`_reset_stars` consumes them.  It stores no endpoint costate, no reverse
`Lambda`, and no per-layer Price factors.  Exact reverse use requires storing
or replaying the forward seam and paying the dense covariance conjugation
adjoint.

Moreover the displayed Schur multiplier is only the fixed-marginal smooth
block.  The deployed path also contains:

- `rho=clip(C/(sigma_i sigma_j),-1,1)`, whose active-set Jacobian must be used;
- coefficient and inverse-`sigma` dependence, producing diagonal/marginal
  adjoints in addition to `Gamma odot Lambda`;
- covariance repair/floor operations, whose Jacobian is not the Price kernel;
- Edgeworth K3/K4/K31/K22 branches, with their own source adjoints.

Thus “Q24 already owns the reverse” is false.  The exact conditional statement
is: a newly scheduled reverse AD through the same seven-term polynomial gets
the degree0--5 Price multiplier elementwise for free **after** its query and
forward state coexist; only the omitted high-Hermite tail and the non-Q24
branches remain to bound.  This can reduce the marginal cost of a proposed
dense reverse, but cannot create one from the forward word.

### Graph node: Q24_PRICE_REVERSE_RECOMPUTE_NOT_RESIDENCY

- solved identity: derivative of the n<=6 pair kernel is the n<=5 Price multiplier;
- cost: elementwise once `rho`, coefficients, and reverse `Lambda` coexist;
- literal status: current D10/D11 has no resident reverse or retained per-layer query;
- exactness caveats: clip, marginal dependence, repair/floors, and Edgeworth branches;
- architecture use: conditional CSE inside a newly billed/replayed reverse, never zero-cost ownership.

## Checkpoint 100: a label-free L4 radius for the true-minus-AQ q211 source

The eleven-term deformation in Checkpoint76 admits a sharp, cheap amplitude
envelope before any tensor is formed.  For centered scalar variables,

\[
 |\kappa_4(X_1,X_2,X_3,X_4)|
 \le4\prod_{s=1}^4\|X_s\|_4,
\]

because the raw fourth product and each of the three covariance-pair products
is bounded by the same Holder product.

Define per-unit positive radii

\[
 a_i=\|Y_i^A-EY_i^A\|_4,
 \qquad
 d_i=\|\delta_i-E\delta_i\|_4,
\]

and the nonnegative q211 envelope

\[
 P_{ijk}=(a_i+d_i)^2(a_j+d_j)(a_k+d_k)-a_i^2a_ja_k.
\]

Multilinearity plus the exact binomial multiplicities of Checkpoint76 gives

\[
 \boxed{
 |T_{211}(Y^{true})-T_{211}(Y^A)|_{ijk}
 \le4P_{ijk}.}
\]

This is already target-free and label-free.  It is also rank-two as an
ordinary three-way envelope.  Put

\[
 x'=(a+d)^{\odot2},\quad x=a^{\odot2},
 \qquad y'=a+d,\quad y=a.
\]

Then `P=x' tensor y' tensor y' - x tensor y tensor y`, and its Frobenius norm
is available from six scalar reductions:

\[
 \boxed{
 \|P\|_F^2
 =\|x'\|^2\|y'\|^4+
   \|x\|^2\|y\|^4-
   2\langle x',x\rangle\langle y',y\rangle^2.}
\]

For any synchronized q211 endpoint mask `Lambda`,

\[
 |\langle\Lambda,\Delta T_{211}\rangle|
 \le4\langle|\Lambda|,P\rangle
 \le4\|\Lambda\|_F\|P\|_F.
\]

If the complete old-projected suffix operator from the local q211 plane to
the endpoint has Frobenius operator norm at most `tau_l`, then the entire
multi-layer response obeys

\[
 \left\|\sum_lF_l\Delta T_{211,l}\right\|
 \le4\sum_l\tau_l\|P_l\|_F.
\]

This is a rigorous coherent upper bound.  It can close the true-minus-AQ
route without labels whenever the right side is below the required endpoint
amplitude (for the current master gate, `1.691e-5` after matching units and
normalization).  Passing the radius is not an alignment certificate.  The
same formula, replacing base `Y^A` by `X^A` and increment `delta` by `r^A`,
bounds the intrinsic AQ surface in the stored-AQ-only branch.

### Graph node: FULL_AQ_Q211_L4_RADIUS_GATE

- status: exact label-free source/action upper bound;
- input: per-unit centered L4 radii of the AQ output and true-minus-AQ residual;
- cost: vector reductions only for the source Frobenius radius;
- robust close: `4 sum tau_l ||P_l||F` below the required endpoint amplitude;
- limitation: an upper radius cannot certify target alignment or future-query residency.

## Checkpoint 101: a live CP4 factor frame gives a lawful full-AQ covariance section

The fixed-embedding no-go does not apply if the AQ latent frame is derived from
a live neuron-covariant state.  A CP ray has the scaling gauge
`lambda_a f_a^4=(lambda_a/c^4)(cf_a)^4`.  Therefore first define the
gauge-invariant direction

\[
 g_a=|\lambda_a|^{1/4}f_a,
\]

which changes only by a sign under a ray rescaling.  Let `F in R^{n by r}` be
`r` selected nonzero columns `g_a` at the cut, using an invariant score with a
strict selection gap.  Write `f_i` for its `i`th row and form the quadratic Veronese frame

\[
 P(F)_{i,:}=\operatorname{svec}(f_if_i^T)
 \in\mathbb R^{d_2},
 \qquad d_2={r(r+1)\over2}.
\]

Its row Gram is available without forming `P`:

\[
 \boxed{P(F)P(F)^T=(FF^T)^{\odot2}.}
\]

Assume the exact frame gate

\[
 \operatorname{rank}P(F)=n,
\]

which is generically possible once `d2>=n` and the selected factors are dense.
Set

\[
 K=P P^T,
 \qquad O_F=K^{-1/2}P.
\]

Then `O_F O_F^T=I_n`.  Let `C_X` be the once-only target low-chaos covariance
after subtracting the synchronized Gaussian-rest and incumbent contributions,
choose

\[
 U=\epsilon F,
 \qquad R=C_X-UU^T\succ0,
\]

and define the quadratic AQ coefficient rows by

\[
 \boxed{
 X_A=\sqrt2,R^{1/2}O_F
    =\sqrt2,R^{1/2}K^{-1/2}P(F).}
\]

Interpreting row `i` of `X_A` as `svec(A_i)` gives the exact half-AQ
covariance identity

\[
 \boxed{
 UU^T+{1\over2}X_AX_A^T=C_X.}
\]

A sufficient positive-residual choice is
`epsilon²<lambda_min(C_X)/||F||op²`; semidefinite cuts use the corresponding
support/pseudoinverse version.

This construction is lawful.  Under a neuron permutation `Pi` and a signed
permutation `S` of the selected live factors,

\[
 F' =\Pi F S,
 \qquad
 P(F')=\Pi P(F)\rho_2(S),
\]

where `rho2` is the orthogonal action on `Sym2`.  Functional calculus gives
`K'^-1/2=Pi K^-1/2 Pi^T`, hence

\[
 O_{F'}=\Pi O_F\rho_2(S),
 \qquad X_A'=\Pi X_A\rho_2(S).
\]

The resulting AQ laws differ only by the legitimate latent `O(r)` gauge.  A
fixed `J` has no such live transformation law; this is exactly why the CP4
frame evades, rather than contradicts, the cold-anchor theorem.  It requires
a selector whose ties and column genealogy are themselves equivariant.

### The remaining K21 section is an exact Stiefel problem

Every full-rank quadratic coefficient bank with the same residual covariance
has the form

\[
 X_A=\sqrt2 R^{1/2}O,
 \qquad OO^T=I_n.
\]

Thus the covariance-preserving freedom is the row Stiefel manifold
`St(n,d2)`.  With `A_i(O)=svec^-1(X_A[i,:])`, define

\[
 \mathcal K_{U,R}(O)_{ij}
 =\operatorname{tr}(A_i^2A_j)
  +u_i^TA_ju_i+2u_i^TA_iu_j.
\]

An exact weak seam exists in this anchored branch iff the D14 target K21 lies
in the image of this map.  A stable local branch at `O*` requires

\[
 \boxed{
 D\mathcal K_{U,R}(O_*)
 :\{\dot O:\dot OO_*^T+O_*\dot O^T=0\}
 \twoheadrightarrow\mathbb R^{n\times n}}
\]

with a positive restricted smallest singular value.  At `n=1024,r=63`,
`d2=2016` and

\[
 \dim\operatorname{St}(1024,2016)
 =1024\cdot2016-{1024\cdot1025\over2}
 =1,539,584,
\]

leaving a 491,008-dimensional regular fiber after matching K21.  At r62 the
corresponding numbers are `d2=1953`, Stiefel dimension1,475,072 and surplus
426,496; near an affine cut its last561 directions retain Checkpoint97's cubic
conditioning.

### Arithmetic boundary

The identity is not a zero-cost converter.  `FF^T` costs only `r/n` of a
width product and `K=(FF^T)^2` is elementwise, but a generic exact converter
still needs the two matrix functions `R^1/2,K^-1/2`, a dense left factor
`B=sqrt2 R^1/2K^-1/2`, and the K21 Stiefel solve.  Fusing initialization with
the first coefficient push gives

\[
 W^TX_A=(W^TB)P,
\]

so, relative to the already charged generic `W^TX_A`, the unavoidable generic
increment is at least one additional width-square product for `W^TB`, before
the matrix functions and solve.  This alone slightly exceeds the r63/L5
terminal shell's `0.9296` headroom; r62/L5 leaves only about`0.567` after it.
The r63/L4 shell has more room but a shorter value-bearing suffix.

Exact arithmetic escapes require `B` to be diagonal/low-rank/in a paid matrix
algebra, `R` proportional to `K`, or literal reuse of the needed left product
and matrix functions.  Otherwise the construction solves lawfulness and
covariance image membership but not the deploy bill.  Strong CP3/CP4 future
matching, complete-old subtraction, and Gaussian-rest coherence remain
separate gates.

### Graph nodes

- `CP4_SEEDED_AQ_COVARIANCE_SECTION`: exact live-frame formula and covariance identity;
- `CP4_SEEDED_AQ_EQUIVARIANCE`: lawful under neuron permutations and latent factor gauge; fixed-J no-go avoided;
- `CP4_SEEDED_AQ_K21_STIEFEL_GATE`: exact image/surjectivity/conditioning test, open at the actual cut;
- `CP4_SEEDED_AQ_CONVERSION_COST_GATE`: at least one extra dense left product plus matrix functions/solve, current L5 shells conditionally closed;
- `CP4_SEEDED_AQ_STRONG_SEAM_GATE`: cubic/quartic future equivalence and old/rest ownership still required.

## Checkpoint 102: exact dual/Stiefel gate and an equivariant K21 refit flow

The K21 seam gate in Checkpoint101 need not be represented by a gigantic
explicit Jacobian.  For fixed `U` and current matrices `A_i`, a tangent bank
`H_i` has

\[
\boxed{
\begin{aligned}
(D_A\mathcal K[H])_{ij}={}&
\operatorname{tr}\!\left[(H_iA_i+A_iH_i)A_j+A_i^2H_j\right]\\
&+u_i^TH_ju_i+2u_i^TH_iu_j.
\end{aligned}}
\]

For an arbitrary oriented K21 dual plane `Lambda`, its exact matrix adjoint is

\[
\boxed{
\begin{aligned}
(D_A\mathcal K^*\Lambda)_k={}&
\sum_j\Lambda_{kj}(A_kA_j+A_jA_k)
+\sum_i\Lambda_{ik}A_i^2\\
&+\sum_i\Lambda_{ik}u_iu_i^T
+2\sum_j\Lambda_{kj}\operatorname{sym}(u_ku_j^T).
\end{aligned}}
\]

Let `G_X(Lambda)` be the matrix whose `k`th row is the `svec` of this
adjoint.  Since

\[
 X_A=\sqrt2 R^{1/2}O,
 \qquad \dot X_A=\sqrt2R^{1/2}\dot O,
\]

the ambient adjoint in Stiefel coordinates is

\[
 G_O(\Lambda)=\sqrt2R^{1/2}G_X(\Lambda).
\]

For the row-Stiefel constraint `OO^T=I`, the orthogonal tangent projector is

\[
 \Pi_{T_O}(G)=G-\operatorname{sym}(GO^T)O.
\]

Hence the exact rank certificate is

\[
 \boxed{
 D\mathcal K(O)\text{ is onto}
 \iff
 \Pi_{T_O}\!\left(\sqrt2R^{1/2}G_X(\Lambda)\right)=0
 \Longrightarrow\Lambda=0.}
\]

Equivalently the endpoint plane Gram

\[
 \Lambda\mapsto
 \left\|\Pi_{T_O}(\sqrt2R^{1/2}G_X(\Lambda))\right\|_F^2
\]

must be positive definite on all `n^2` oriented K21 masks.  This types the
remaining weak-section receipt entirely in matrix/Jordan operations and the
row-Stiefel projector; it also shows that a symmetric-only mask audit is
insufficient because K21 is oriented.

There is a canonical target-free branch construction.  Define

\[
 \mathcal E(O)={1\over2}
 \|\mathcal K_{U,R}(O)-K^{21,D}\|_F^2
\]

and start at the live-frame anchor `O_F`.  Its Riemannian gradient is the
displayed tangent adjoint with
`Lambda=mathcal K(O)-Ktarget`.  Every operation is equivariant under neuron
permutation and the inherited factor gauge.  Therefore deterministic gradient
flow (or a correspondingly equivariant retraction scheme) remains lawful; if
it converges to a zero-residual regular point, it supplies the missing
permutation-equivariant section without a lexicographic branch choice.

This resolves the **definition** of the refit, not its deployment economics or
global convergence.  Each gradient evaluation contains dense all-unit Jordan
aggregates and a Stiefel retraction; an unbounded iterative solve cannot be
hidden in the suffix bill.  The fail-fast order is: live-frame rank and PSD
gate, evaluate the dual singular certificate, then only if it is regular seek
a bounded-iteration contraction/billing proof.

### Graph node: CP4_SEEDED_AQ_K21_DUAL_GATE

- solved: exact K21 differential, adjoint, and row-Stiefel tangent projection;
- onto iff the projected adjoint has zero kernel on oriented K21 masks;
- lawful branch: equivariant gradient flow from the CP4 live-frame anchor;
- open: global zero-residual convergence, singular-value floor, iteration count, and bill.

## Checkpoint 103: pair-support symmetry and dimension forbid the tempting 1024-column shortcut

The covariance construction can use a restricted H2 feature support of size
`p`, but the K21 implicit-function gate imposes a sharp dimension floor.  With
`U` fixed, the covariance-preserving row-Stiefel manifold has dimension

\[
 np-{n(n+1)\over2}.
\]

Surjectivity onto the oriented `n^2` K21 plane is impossible unless

\[
 np-{n(n+1)\over2}\ge n^2,
\]

or equivalently

\[
 \boxed{p\ge\left\lceil{3n+1\over2}\right\rceil.}
\]

At `n=1024`, this is

\[
 \boxed{p\ge1537.}
\]

Thus selecting only a square set of 1024 pair columns can match covariance but
can never provide generic exact K21 control: its Stiefel dimension is only
523,776, exactly half the required oriented plane dimension.

There is also a representation-theoretic lawfulness constraint.  Under
permutation of the `r=63` selected CP rays, unordered pair coordinates split
into only two orbits:

\[
 \{(a,a)\}:63,
 \qquad
 \{(a,b):a<b\}:\binom{63}{2}=1953.
\]

An invariant support is a union of these orbits, so its possible sizes are
`0,63,1953,2016`; there is no lawful fixed 1024-pair subset.  A data-dependent
subset would need its own equivariant, gapped selection theorem.  The clean
minimal invariant support meeting the dimension floor is therefore the full
off-diagonal orbit `p=1953`; the fully `O(63)`-covariant choice is all
`p=2016` Sym2 coordinates.

Including the 63 H1 coordinates, the respective first coefficient widths are

\[
 {1953+63\over1024}=1.96875,
 \qquad
 {2016+63\over1024}=2.0302734375
\]

product-equivalents.  Even the off-diagonal orbit saves only`0.06152` relative
to full Sym2.  It does not pay the generic extra left product or matrix
functions in Checkpoint101.  More aggressive support compression and generic
exact K21 control are mathematically incompatible.

### Graph node: CP4_SEEDED_AQ_PAIR_SUPPORT_NOGO

- exact K21 dimension floor: at least1537 active H2 coordinates;
- lawful S63 orbit sizes:63 diagonal and1953 off-diagonal;
- fixed 1024-pair shortcut: closed both by dimension and factor-permutation symmetry;
- minimal invariant candidate:1953 off-diagonal pairs, only0.06152 product saved versus full Sym2;
- implication: conversion economics require matrix-product/root reuse, not pair-support trimming.

## Checkpoint 104: release `U` after anchoring -- r63 is the first rank with enough complete-pair seam fiber

The live CP4 bank should initialize the AQ gauge; it must not necessarily
freeze all `nr` H1 coefficients.  For a fixed once-only low covariance
`C_X\succ0`, the complete covariance-preserving full-AQ manifold has the exact
parameterization

\[
\boxed{
 U\in\mathbb R^{n\times r},\quad UU^T\prec C_X,
 \qquad
 X_A(U,O)=\sqrt2\,[C_X-UU^T]^{1/2}O,
 \qquad OO^T=I_n.}
\]

Conversely every full-row-rank `(U,X_A)` satisfying

\[
 UU^T+{1\over2}X_AX_A^T=C_X
\]

has this form.  Quotienting the latent `O(r)` gauge, its dimension is

\[
 \boxed{
 nr+nd_2-{n(n+1)\over2}-{r(r-1)\over2}.}
\]

After matching the oriented `n^2` K21 plane, the regular weak-seam fiber is

\[
 f_r=nr+nd_2-{n(n+1)\over2}-{r(r-1)\over2}-n^2.
\]

At width1024,

\[
\begin{array}{c|c|c|c}
r&d_2&\text{fixed-}U\text{ post-K21 fiber}&f_r\text{ with }U\text{ released}\\\hline
62&1953&426,496&488,093\\
63&2016&491,008&553,567
\end{array}
\]

This interacts decisively with the complete-Q24 strong seam.  The independent
`(2,2)` pair sector has the exact generic tangent rank

\[
 \binom{1024}{2}=523,776
\]

at the first future ReLU covariance read.  Therefore, if this pair response is
not retained in a separate incumbent CP4/Q24 state and must be selected by the
AQ weak fiber, dimension alone gives

\[
\boxed{
\begin{aligned}
r62:&\quad488,093<523,776 &&\text{closed by }35,683,\\
r63\text{ with frozen }U:&\quad491,008<523,776
   &&\text{closed by }32,768,\\
r63\text{ joint }(U,O):&\quad553,567>523,776
   &&\text{possible margin }29,791.
\end{aligned}}
\]

Thus r63 is not merely the first rank escaping the affine Veronese-null
obstruction.  It is also the first of r62/r63 with enough **full weak-section
fiber** to absorb a generic complete pair seam, and only if the H1 bank is
allowed to coadapt after using CP4 to choose the lawful initial gauge.

The correct refit is consequently joint constrained optimization in `(U,X_A)`
or `(U,O)`, initialized at the equivariant CP4 point.  A deterministic
equivariant flow preserves lawfulness while releasing the artificial
`U=epsilon F` restriction.  The result is still conditional: the combined
K21-plus-strong-pair differential must actually be onto with a useful singular
floor, and a separate retained incumbent CP4/Q24 pair state removes the stated
523,776 burden instead of allowing it to be counted twice.

### Graph node: CP4_SEEDED_AQ63_JOINT_STRONG_SEAM_THRESHOLD

- exact manifold: covariance-preserving joint `(U,O)` modulo latent gauge;
- r62 full weak fiber488,093, below complete-pair rank by35,683;
- r63 frozen-U fiber491,008, below by32,768;
- r63 joint fiber553,567, first dimensionally open case with margin29,791;
- binding branch: either retain incumbent pair state separately or prove joint strong-seam surjectivity, never both;
- open: actual combined minor, conditioning, lawful convergence, and bill.

## Checkpoint 105: full Sym2 width is symmetry-mandatory for the r63 complete-pair seam

Allow only `p<=2016` active H2 coordinates but keep all r63 H1 variables.  If
`g_E` is the dimension of the latent `O(63)` gauge that preserves the chosen
hard support, the exact post-K21 joint fiber dimension is

\[
 f_E(p)=1024(63+p)-524,800-1,048,576-g_E.
\]

For a generic/non-invariant hard support `g_E=0`.  Weak K21 matching alone
then requires

\[
 \boxed{p\ge1474.}
\]

To retain enough remaining freedom for the independent complete `(2,2)` pair
response, require

\[
 f_E(p)\ge523,776,
\]

which gives the sharp integer floor

\[
 \boxed{p\ge1985.}
\]

The only fixed supports invariant under permutation of the 63 live CP rays
have sizes `0,63,1953,2016`.  The off-diagonal orbit `p=1953` is enough for a
weak seam, has `g_E=0`, and leaves fiber491,008, so it misses the complete-pair
rank by32,768 dimensions.  Full `Sym2` restores the complete `O(63)` gauge
`g_E=1953` but has `p=2016` and leaves fiber553,567.  Therefore

\[
 \boxed{
 \text{a lawful fixed-support r63 seam carrying the generic complete pair
 response must use all }2016\text{ Sym2 coordinates}.}
\]

Its margin is only29,791 dimensions.  Dropping the 63 diagonal coordinates
saves `63/1024=0.06152` of a product but destroys generic strong-seam capacity.
This makes the width/cost boundary exact: full Sym2 is not a conservative
implementation choice in the replacement-state branch; it is forced by the
pair-sector tangent rank.  If incumbent CP4/Q24 continues to carry that pair
sector separately, the strong burden is removed and the off-diagonal support
can remain a weak-seam candidate, subject to once-only ownership.

### Graph node: AQ63_COMPLETE_PAIR_FULL_SYM2_NECESSITY

- generic hard-support weak threshold: p>=1474 (`g_E=0`);
- generic hard-support weak plus complete-pair threshold: p>=1985;
- lawful fixed S63 supports:0,63,1953,2016;
- conclusion: replacement-state strong seam forces full p=2016, margin29,791;
- branch caveat: separately retained incumbent pair state removes this requirement and its gain credit.

## Checkpoint 106: the late r63 shells need large nonlinear amplification to meet the raw goal

The exact cost frontier must be paired with the already banked frozen-chain
value map.  Summing its signed cell attributions gives

\[
\begin{array}{c|c|c|c|c}
\text{cells}&v&k3&k4&\text{total points}\\\hline
L12\text{--}L15&10.28&1.15&-0.05&11.38\\
L11\text{--}L15&12.58&1.15&-0.04&13.69\\
L10\text{--}L15&14.32&0.89&-0.06&15.15\\
L9\text{--}L15&15.64&1.57&-0.19&17.02.
\end{array}
\]

These are not upper bounds on a coadapted architecture, but they are the only
certified value scale for the corresponding late injection windows.  Relative
to them, a20% raw fire requires amplification factors

\[
 \boxed{
 1.758\ (L12\text{--}15),\quad
 1.461\ (L11\text{--}15),\quad
 1.320\ (L10\text{--}15),\quad
 1.175\ (L9\text{--}15).}
\]

From the current raw `2.9409e-8`, reaching `1e-8` requires about a66.0% raw
reduction.  The corresponding factors are

\[
 \boxed{
 5.80,\quad4.82,\quad4.36,\quad3.88.}
\]

Therefore the r63 four/five-reset shells cannot be promoted merely because
their seam is dimensionally possible and their arithmetic nearly fits.  If
the schedule maps respectively to the `L12--15` or `L11--15` cells, it must
produce a roughly1.76x/1.46x nonlinear-coadaptation gain even to clear the fire
bar, and5.8x/4.82x to reach the campaign raw goal.  The six/seven-cell suffix
has a much smaller fire amplification burden but its state-conversion bill is
tighter.  A different cut indexing must substitute its exact cells before
using the table.

### Graph node: AQ63_LATE_SUFFIX_VALUE_AMPLIFICATION_GATE

- status: exact arithmetic on the certified signed value map / not a ceiling;
- late four-cell certified scale:11.38 points, needs1.758x to fire and5.80x for raw1e-8;
- late five-cell scale:13.69 points, needs1.461x and4.82x;
- L9--15 scale:17.02 points, needs1.175x and3.88x;
- implication: late-shell feasibility is not goal feasibility; demand an explicit coadaptation amplification mechanism/ceiling.

## Checkpoint 107: finite-depth Neumann gate for the required coadaptation amplification

There is an exact mechanism by which the self-consistent AQ suffix can exceed
the frozen injection map: corrected earlier states change later sources.  Stack
the value-normalized layer corrections and write the first-order causal system

\[
 x=b+\mathcal N x,
\]

where `b` is the frozen-source response and `N` is strictly lower block
triangular across an `L`-step suffix.  Since `N^L=0`, no asymptotic assumption
is needed:

\[
 \boxed{
 x=(I-\mathcal N)^{-1}b
  =\sum_{k=0}^{L-1}\mathcal N^k b.}
\]

If `||N||<=gamma` in the same endpoint/Bellman norm, the amplification obeys

\[
 \boxed{
 {\|x\|\over\|b\|}
 \le S_L(\gamma)
 :=\sum_{k=0}^{L-1}\gamma^k.}
\]

Consequently `S_L(gamma)>=A_req` is a necessary condition for explaining any
claimed factor `A_req` by first-order coadaptation.  Combining with
Checkpoint106 gives the approximate minimum norm receipts

\[
\begin{array}{c|c|c|c|c}
\text{cell window}&L&A_{fire}&\gamma_{fire,min}&
A_{raw<1e-8},\ \gamma_{goal,min}\\\hline
L12\text{--}15&4&1.758&0.456&5.80,\ 1.254\\
L11\text{--}15&5&1.461&0.318&4.82,\ 0.982\\
L10\text{--}15&6&1.320&0.243&4.36,\ 0.870\\
L9\text{--}15&7&1.175&0.149&3.88,\ 0.793.
\end{array}
\]

Thus a four-step late suffix cannot reach the raw goal under a nonexpansive
coadaptation operator at all: `S4(1)=4<5.80`.  A five-step suffix needs an
almost norm-one positive feedback path (`gamma about0.982`).  The seven-step
window still needs a strong but materially safer `gamma about0.793`.

The exact projective AQ reset itself cannot supply an expansive factor:
`D R_g=P<=2 M_{H(g)}P<=2` is a positive contraction, and the omitted
`H>=3` residual map is globally one-half-Lipschitz.  Therefore any required
`gamma>1` (and most of a near-unit signed feedback mode) must be localized in
the dense unit transport/frame normalization or in the connected-cumulant
response feedback, not attributed vaguely to the ReLU reset.  This provides a
typed mechanism audit: bound those two blocks separately and multiply them
through the causal path.

This is also an error-sensitivity theorem.  For an operator defect `E`, the
resolvent identity is

\[
 (I-N-E)^{-1}-(I-N)^{-1}
 =(I-N-E)^{-1}E(I-N)^{-1}.
\]

The same high-gain path that supplies the missing raw improvement amplifies
reset, seam, quadrature, and ownership defects.  Therefore a near-critical
L5 mechanism is credible only with a correspondingly smaller complete
residual radius; gain and error cannot be budgeted independently.  Nonlinear
finite corrections may exceed the linear Neumann prediction, but then their
second/higher derivative remainder must be the explicit source of the excess,
with its own sign and radius certificate.

### Graph node: AQ_SUFFIX_COADAPTATION_NEUMANN_GATE

- mechanism: exact nilpotent causal resolvent `sum_{k=0}^{L-1}N^k`;
- necessary norm receipts: table above;
- L4 raw-goal close under `||N||<=1`; L5 requires near-critical0.982; L7 requires0.793;
- robustness: the same resolvent multiplies every seam/reset/source defect;
- promotion: a value-normalized feedback-norm or signed eigenpath certificate plus nonlinear remainder bound.

## Checkpoint 108: explicit rank-561 candidate for the corrected r62 tilted-wedge core

The raw Jordan witness of Checkpoint88 is not by itself the full silent cubic
minor.  After imposing `N^T R=0`, square-column terms cancel and the pure
`q_i=u_iu_i^T` Jordan term is killed by the local anchor constraints.  The
surviving intrinsic covector has the one-residual wedge form

\[
 \Phi_N(Z)
 =\operatorname{skew}\sum_i n_i\otimes
 P_0\{n_i,Q_U^*(Ze_i)\},
 \qquad Z\in\operatorname{Sym}R,quad\dim R=33,
\]

modulo the anchor adjoint.  Here `P0` projects matrices to `E0=ker Q_U`.

There is an explicit algebraic construction making the **pre-anchor-quotient**
wedge map injective.  Split the latent space as

\[
 \mathbb R^{62}=V_{33}\oplus C_{29}.
\]

Choose:

- independent zero-diagonal matrices `E_a in Sym(C)`, `a=1,...,33`;
- a cross-block matrix `D0 in V tensor C`;
- `d_a=D0+E_a`;
- independent `v_b in V` for which
  `w_raw,b={D0,v_bv_b^T}=sym(v_b tensor D0v_b)` are independent.

Construct the Veronese frame from points of two types:

\[
 u\in V,
 \qquad
 u=v+e_k,quad v\perp D_0e_k,quad k=1,\ldots,29,
\]

where `e_k` are coordinate axes of `C`.  Every such point is isotropic for
all `E_a` and for `D0`: the `E_a` have zero diagonal on a one-sparse C part,
and the cross form vanishes by the displayed orthogonality.  Yet their
rank-one span has dimension at least

\[
 \dim\operatorname{Sym}V+29\cdot32+29
 =561+928+29=1518.
\]

Hence one can select 1024 independent Veronese rows so that `Q_U` is onto,
while keeping `D0,E_a in E0`.  Choose their 1024-dimensional span `E1` to
contain

\[
 M_b=v_bv_b^T
\]

but avoid the 33-dimensional cross span of the `w_raw,b`.  Then

\[
 w_b=P_0w_{raw,b}
\]

remain independent, lie in a block disjoint from the `E_a`, and

\[
 P_0\{d_a,M_b\}=w_b
\]

is independent of `a`.  Therefore

\[
 \Phi(Z)
 =\sum_{a,b}z_{ab}(D_0+E_a)\wedge w_b
\]

contains the coefficient block

\[
 \boxed{
 \sum_{a,b}z_{ab}E_a\wedge w_b,}
\]

which is injective even on all 33-by-33 matrices, hence certainly on the 561
symmetric modes.

The required physical `R` vectors can be realized by disjoint exact second-
difference triples:

\[
 r_b=(1/2,1/2,-1),qquad
 u_{b,+}=a_b+v_b,quad u_{b,-}=a_b-v_b,quad u_{b,0}=a_b,
\]

for which `r_b^TU=0` and `Q_U^*r_b=v_bv_b^T`.  Choosing corresponding
`N` rows `c_b+d_b,c_b-d_b,c_b` gives `r_b^TN=0`, as required by the maximal
first-order kernel construction.

This proves that the corrected one-residual wedge algebra has enough rank and
supplies a concrete finite witness schema.  One gate remains before claiming
the full seam: verify that the 561 `E_a wedge w_b` modes survive the exact
local anchor-adjoint quotient `H_i u_i=0` and the remaining pseudoforest/
first-Schur constraints.  If that quotient kills a combination, the witness
must be modified; the pre-quotient rank alone is not promotion.

### Graph node: R62_CUBIC_TILTED_WEDGE_EXPLICIT_CANDIDATE

- status: **superseded/closed as a seam witness by Checkpoint110**; pre-anchor rank561 only;
- latent split:33+29, with C-block `E_a` and cross-block `w_b`;
- Veronese frame:1024 independent isotropic rays available from a1518-dimensional span;
- physical realization:33 disjoint second-difference triples;
- terminal gate: injectivity after the exact anchor-adjoint and first-Schur quotient.

## Checkpoint 109: exact r63 first-order defect-angle theorem

Removing the universal Veronese-null block at r63 is necessary but not
sufficient.  Put `m=n-r=961`, `E=Sym2(R63)`, and, for a Veronese-onto anchor
`U`, `E0=ker Q_U` with `dim E0=992`.  Assume separately that the affine K21
differential `F_U` is onto.  Let

\[
 \mathcal A=\operatorname{range}F_U^*,\quad
 \mathcal B=\mathcal A\cap(K\otimes E),\quad
 \mathcal C=\mathcal B^\perp\subset K\otimes E_0,
\]

and `D_U=C^perp` inside `K tensor E0`.  For the prefix block `N0`, the exact
right-inverse-free Schur adjoint norm is

\[
 \boxed{
 \|S_{1,U,N}^*Z\|_{quot}
 =\|P_{\mathcal C}(ZN_0)\|_F,
 \qquad Z\in\operatorname{Sym}K.}
\]

Therefore

\[
 \boxed{
 S_{1,U,N}\text{ is onto}
 \iff
 \{Z\in\operatorname{Sym}K:ZN_0\in\mathcal D_U\}=\{0\}.}
\]

If cubic-gradient evaluation on the 1024 anchor points is injective, its rank
is

\[
 \dim\operatorname{Sym}^3\mathbb R^{63}=43,680,
\]

and the compatibility defect has the exact dimension

\[
 \boxed{
 \dim\mathcal D_U=63\cdot1024-43,680=20,832.}
\]

This is the structured defect missed by the naive assertion that full-row
`N0` is sufficient.

There is a sharp conditional existence theorem.  If every nonzero complex
matrix in `D_U` has row rank at least176, then a Zariski-open nonempty set of
`961 by992` prefix blocks passes the seam.  The incidence margin at the first
possible rank is

\[
 176\cdot992-
 \left({176(2\cdot961-176+1)\over2}-1\right)
 =20,857>20,832.
\]

Rank175 gives only20,651, so176 is sharp for this coarse dimension proof.  A
stronger exact compression-profile criterion uses

\[
 \mu_s(U)=\max_{R\in Gr(s,K)}
 \dim(\mathcal D_U\cap(R\otimes E_0))
\]

and requires

\[
 \boxed{
 \mu_s(U)<{s(s+61)\over2}+1
 \quad\text{for every }s=1,\ldots,961.}
\]

The actual-cut certificate remains the positive Gram
`Z -> ||P_C(ZN0)||²`; the rank-176 theorem only proves that lawful good
prefixes can exist if the anchor defect is sufficiently high-rank.

The coarse rank-176 route has an important symmetry failure mode.  Let
`M3(U)` denote the cubic Veronese moment/evaluation tensor of the anchor and
let `Omega` be a rank-`s` skew latent generator.  The infinitesimal pattern

\[
 Y_{\Omega,i}=\Omega u_i
\]

obeys the exact cubic-gradient adjoint identity

\[
 \Gamma^*Y_\Omega=\Omega\cdot M_3(U).
\]

If `M3(U)=0`, or if a nonzero `Omega` stabilizes it, affine surjectivity maps
this pattern into a nonzero compatibility defect whose row rank is at most

\[
 sr-{s(s-1)\over2}.
\]

For the smallest nonzero skew rank `s=2` at r63 this is

\[
 2\cdot63-1=125<176.
\]

Hence a cubic-centered/isotropic anchor cannot satisfy the simple
`delta(U)>=176` sufficient theorem.  This does not prove its actual seam
singular: the sharper compression profile or the concrete Gram can still
pass for a special `N0`.  It does mean that deliberate CP4 anchor selection
should not enforce cubic centering and then cite the rank-176 genericity
shortcut.

Conditioning factorizes cleanly.  With

\[
 \alpha(U,N)=\inf_{Z\ne0}
 {\operatorname{dist}(ZN_0,\mathcal D_U)\over\|ZN_0\|_F},
\]

\[
 \boxed{
 \sigma_{1,63}\ge\sigma_{\min}(N_0)\alpha(U,N).}
\]

For `J63(t)=tS1+t^3S3+...`, this yields a stable inverse at scale `t^-1`
when `t²||S3||` is below the displayed first-order margin.  The value-weighted
variant needs positivity only after composing with the complete-old L5 suffix
rowspace; local dark modes irrelevant to that suffix need not be opened.

### Graph nodes

- `R63_FIRST_SCHUR_DEFECT_ANGLE_GATE`: exact iff `P_C(ZN0)` has trivial symmetric kernel;
- `R63_DEFECT_RANK176_GENERIC_EXISTENCE`: conditional Zariski-open existence theorem, not actual-cut proof;
- `R63_FIRST_SCHUR_CONDITIONING`: product of prefix row singular value and defect angle;
- `R63_VALUE_WEIGHTED_SEAM_GATE`: compose the Gram only with the complete-old suffix-observable rowspace.
- `R63_CUBIC_CENTERED_DEFECT_CAVEAT`: cubic-centered anchors have a rank-at-most125 defect and fail the coarse rank176 theorem.

## Checkpoint 110: the explicit r62 triple witness fails -- 33 diagonal modes are exact anchor adjoints

Checkpoint108's pre-anchor rank does **not** survive the local constraint.  For
one second-difference triple write

\[
 u_+=a+v,\quad u_-=a-v,\quad u_0=a,
\]

\[
 n_+=c+d,\quad n_-=c-d,\quad n_0=c.
\]

For a skew feature operator `A`, the anchor map is

\[
 \mathfrak T(A)_i=(n_iA)u_i.
\]

Its exact adjoint is

\[
 \mathfrak T^*(y)
 =\operatorname{skew}\sum_i n_i\otimes L_{u_i}(y_i),
 \qquad
 L_u(y)=P_0\operatorname{sym}(uy^T).
\]

Let `M=vv^T=Q_U^*r` for the triple coefficient
`r=(1/2,1/2,-1)`, and put `t=dv`.  Choose local anchor costates

\[
 y_+=t,\qquad y_-=t,qquad y_0=-2t.
\]

Linearity of `L` gives

\[
\begin{aligned}
\mathfrak T^*(y)
={}&(c+d)\wedge L_{a+v}(t)
 +(c-d)\wedge L_{a-v}(t)
 -2c\wedge L_a(t)\\
={}&2d\wedge L_v(dv)\\
={}&d\wedge P_0\{d,vv^T\}.
\end{aligned}
\]

But the last expression is exactly the diagonal tilted-wedge column
`Phi(rr^T)` for this construction.  Hence

\[
 \boxed{
 \Phi(rr^T)\in\operatorname{range}\mathfrak T^*,
 \qquad
 \langle A,\Phi(rr^T)\rangle=0
 \quad\forall A\in\ker\mathfrak T.}
\]

With 33 disjoint triples, the 33 diagonal matrices `r_a r_a^T` are
independent.  Therefore the fully anchored cubic rank of this explicit family
is at most

\[
 \boxed{561-33=528.}
\]

No background correction can preserve their pairing: if `A_corr` enforces the
anchors while leaving a nonzero diagonal action, it would contradict the
adjoint identity above.  This pinpoints the skew-consistency failure in the
proposed `B`-supported primal correction.

The no-go is exact for the disjoint second-difference witness family,
independent of `a,c,D0,E_a` and the background frame.  It is not yet a
universal theorem for every possible r62 prefix: a non-triple realization of
the 33-dimensional physical nullspace may have a different diagonal quotient.
The r62 route returns to the general projected Gram of Checkpoint94; the
specific Checkpoint108 witness is closed and cannot certify cubic
surjectivity.

### Graph node: R62_SECOND_DIFFERENCE_CUBIC_ANCHOR_NOGO

- status: solved exact no-go for the explicit triple witness;
- identity: every diagonal tilted-wedge column is a local anchor adjoint;
- forced rank loss: at least33, so rank at most528<561;
- supersedes: `R62_CUBIC_TILTED_WEDGE_EXPLICIT_CANDIDATE` as a promotion route;
- remaining scope: other non-triple prefixes or fifth-order opening.

## Checkpoint 111: deployed Q24 owns only the linear radial-H4 jet, not the order-r^-2 scale-mixture tail

The literal deployed D11 reset cleanly separates two notions of order that
must not be conflated.  Its Edgeworth density is exactly

\[
  1+L_3+L_4+\tfrac12L_3^2.
\]

The nine first-order weights are the four cubic faces
`(k3,k21^T,k21,k3^T)` and the five quartic faces
`(k4,k31^T,k22,k31,k4^T)`.  The only nonlinear weights are the ten terms in
the self-convolution of the four `L3` faces.  There is no `L4^2`, `L3 L4`,
`L6`, `k6`, or `q6` path in `_edgeworth_weights` or
`_four_pair_edgeworth`.

Meanwhile `_Q24_TERMS=7` means only that the Gaussian Mehler kernels use
correlation powers

\[
  \rho^t,\qquad t=0,\ldots,6.
\]

It does **not** raise the cumulant/Edgeworth truncation order.  Thus, for a
symmetric common Gaussian scale mixture with scale `S`, `L3=0` and the
deployed reset owns the Gaussian base plus the term linear in
`Var(S)`--the H4-square/Q24 Hessian jet.  At the next radial order,
`kappa_3(S)` supplies an `L6` term and the exponential supplies
`1/2 L4^2`; both are absent.  Therefore an exact joint-semicircle/radial
mixture has a nontrivial old-subtracted reset tail beginning at order
`r^-2`, subject afterward to the ordinary CP4/star/same-action Schur
subtraction.

### Graph node: Q24_RADIAL_EDGEWORTH_ORDER_AUDIT

- status: solved literal semantic audit;
- owned: Gaussian Mehler powers through rho^6, linear L3/L4, and 1/2 L3^2;
- omitted: L4^2, L3L4, L6 and all higher radial cumulant orders;
- implication: a symmetric scale-mixture tail first becomes locally novel at
  order r^-2, not at the linear H4-square order.

## Checkpoint 112: the r63 first-order seam reduces to a 20,832-dimensional principal-angle determinant

Checkpoint109's defect Gram admits a smaller exact dual test.  Write

\[
 A:Z\mapsto ZN_0,
 \qquad L=A^*A,
 \qquad T=P_{\mathcal D_U}A L^{-1/2},
\]

on `Sym(K)` after removing any kernel of `A`.  Since
`P_C=I-P_D`, the Schur Gram is

\[
 \boxed{
 G_{63}=L^{1/2}(I-T^*T)L^{1/2}.}
\]

Consequently the exact seam gate is

\[
 \boxed{
 \|T\|_{op}<1
 \iff
 \det_{\mathcal D_U}(I-TT^*)>0.}
\]

The latter determinant lives only on the 20,832-dimensional defect space,
not on the full 462,241-dimensional `Sym(K)` domain.  If `N0 N0^T=I`, the
nontrivial Schur singular values are exactly the sines of the principal
angles between `range(A)` and `D_U`.  In general they are weighted by `L`.
Moreover at most `dim D_U=20,832` modes can lose the unprojected floor, so at
least

\[
 462,241-20,832=441,409
\]

domain modes retain the squared lower floor from `lambda_min(L)` (with the
precise count adjusted if `A` has a kernel).  This does not replace the
actual-cut certificate, but it makes both determinant evaluation and
conditioning bookkeeping defect-sized.

### Graph node: R63_DEFECT_PRINCIPAL_ANGLE_DETERMINANT

- status: solved exact reduction;
- gate: `det_D(I-TT*)>0` on dimension20,832;
- geometry: seam singulars are weighted principal-angle sines;
- consequence: only the defect-sized block can degrade the unprojected row floor.

## Checkpoint 113: exact common-scale ReLU plane, its old-subtracted identity, and the centered rank-one collapse

The joint-semicircle closure has an exact scalar-mixture representation.  Its
positive common scale `S` has Laplace transform

\[
 \mathbb E e^{-uS}
 =\exp\left[-{r\over2}J\!\left({4u\over r}\right)\right],
\]

with cumulants

\[
 \kappa_m(S)
 ={4^{m-1}(m-1)!\,\mathrm{Cat}_m\over r^{m-1}},
\]

so in particular

\[
 \kappa_2={8\over r},\qquad
 \kappa_3={160\over r^2},\qquad
 \kappa_4={5376\over r^3}.
\]

Conditioned on `S=s`, the complete neuron vector is Gaussian with

\[
 h\mid S=s\sim N(m,L+sQ).
\]

Let `a(s)` be the vector of conditional ReLU means and `B(s)` the raw
conditional ReLU second-moment plane.  Every entry of both is an ordinary
univariate or bivariate truncated-Gaussian formula using only

\[
 v_i(s)=L_{ii}+sQ_{ii},\qquad
 c_{ij}(s)=L_{ij}+sQ_{ij}.
\]

Thus the exact mixed covariance is

\[
 \boxed{
 K_{mix}=\mathbb E B(S)-\mathbb E a(S)\,\mathbb E a(S)^T.}
\]

This is a label-free `O(q n^2)` elementwise acquisition for a certified
one-dimensional quadrature with `q` scale nodes; it never constructs a
third- or fourth-order tensor.  Exact finite deployment still needs a
quadrature/analytic-integration receipt and synchronization with the literal
Gaussian-pair implementation.

### Exact once-only subtraction of the Q24 linear H4 jet

Write `a_k=d^k a(s)/ds^k|_{s=1}` and similarly `B_k`.  Put

\[
 p={\kappa_2\over2}a_2,
\]

and define the exact scalar-mixture Taylor remainders

\[
 r_a=\mathbb E a(S)-a_0-p,
 \qquad
 r_B=\mathbb E B(S)-B_0-{\kappa_2\over2}B_2.
\]

The Gaussian reset is `K0=B0-a0 a0^T`; the Q24-owned linear radial H4 plane is

\[
 K_{H4}^{lin}
 ={\kappa_2\over2}
 \left(B_2-a_2a_0^T-a_0a_2^T\right).
\]

Direct expansion, including the unconditional-mean product, gives the exact
novel plane after these two theoretical owners:

\[
\boxed{
 R_{rad}
 =r_B-pp^T-(a_0+p)r_a^T-r_a(a_0+p)^T-r_ar_a^T.}
\]

This identity is safer than subtracting density terms informally: it owns the
mean correction exactly once.  Against the literal deployed engine the
additive source must instead subtract its actual Q24/CP4/star response, or
replace the entire reset; Q24's separate rho-series truncation must not be
silently called the exact Gaussian base.

The first omitted asymptotic plane is

\[
\begin{aligned}
 R_{rad}^{(2)}={}&{\kappa_3\over6}
  \left(B_3-a_3a_0^T-a_0a_3^T\right)\\
 &+{\kappa_2^2\over8}
  \left(B_4-a_4a_0^T-a_0a_4^T\right)
  -{\kappa_2^2\over4}a_2a_2^T,
\end{aligned}
\]

which types the absent `L6` and `L4^2/2` pieces and their centralization
counterterm.

### Rigorous target-free radius

For `delta=S-1`, Taylor's integral remainder gives, for either vector- or
matrix-valued `f`,

\[
 r_f=\mathbb E\left[
 {\delta^3\over2}\int_0^1(1-t)^2
 f^{(3)}(1+t\delta)\,dt\right].
\]

Define the corresponding norm envelopes

\[
 \epsilon_a=\mathbb E\left[{|\delta|^3\over2}
 \int_0^1(1-t)^2\|a^{(3)}(1+t\delta)\|_2dt\right],
\]

and `epsilon_B` with the Frobenius norm of `B^(3)`.  Then

\[
\boxed{
 \|R_{rad}\|_F
 \le \epsilon_B+\|p\|_2^2
 +2\|a_0+p\|_2\epsilon_a+\epsilon_a^2.}
\]

Consequently any supplied local adjoint plane `Lambda` obeys

\[
 |\langle\Lambda,R_{rad}\rangle|
 \le\|\Lambda\|_F\left(
 \epsilon_B+\|p\|^2+2\|a_0+p\|\epsilon_a+\epsilon_a^2\right).
\]

This is a true target-free close gate once composed with the endpoint
transport norms; it is an amplitude ceiling, not an alignment certificate.

### Centered pure-quadratic collapse

If `m=0` and `L=0`, ReLU homogeneity makes

\[
 a(s)=\sqrt{s}\,d,qquad B(s)=sB_0,qquad
 d_i={\sqrt{Q_{ii}}\over\sqrt{2\pi}}.
\]

The entire exact scale-mixture correction is therefore rank one:

\[
 K_{mix}-K_0=left[1-(\mathbb E\sqrt S)^2\right]dd^T.
\]

After subtracting the linear H4 owner,

\[
\boxed{
 R_{rad}=eta_r dd^T,qquad
 \beta_r=1-(\mathbb E\sqrt S)^2-{\kappa_2\over4}.}
\]

The Mellin--Laplace identity

\[
 \mathbb E\sqrt S={1\over2\sqrt\pi}
 \int_0^\infty[1-\mathbb Ee^{-uS}]u^{-3/2}du
\]

makes `beta_r` a single exact scalar.  Its leading omitted order is

\[
 \beta_r=-{\kappa_3\over8}+{7\kappa_2^2\over32}+O(r^{-3})
 =-{6\over r^2}+O(r^{-3}).
\]

Hence every endpoint action in this centered branch reduces to

\[
 \boxed{
 \langle\Lambda,R_{rad}\rangle
 =\beta_r,d^T\Lambda d,}
\]

and the source radius is `|beta_r| ||d||^2`.  It is precisely one collective
rank-one direction, so it cannot repair the measured broad-spectrum variance
carrier unless nonzero `m`, nonzero affine covariance `L`, or a later
nonlinear transition breaks this collapse.  Existing evidence that the
collective rank-one plane is nearly value-null is therefore directly relevant
to this special branch.

### Graph nodes

- `JOINT_SEMICIRCLE_COMMON_SCALE_RELU_PLANE`: solved exact local O(q n^2) provider; quadrature and architecture-validity gates open;
- `JOINT_SEMICIRCLE_RADIAL_OLD_SUBTRACTION`: solved exact Gaussian-plus-linear-H4 subtraction including mean counterterms;
- `JOINT_SEMICIRCLE_RADIAL_RADIUS_GATE`: solved derivative-envelope amplitude ceiling;
- `JOINT_SEMICIRCLE_CENTERED_COLLECTIVE_COLLAPSE`: solved rank-one no-go for the centered pure-quadratic branch;
- `JOINT_SEMICIRCLE_NONCOLLECTIVE_REOPEN`: requires affine `L`, bias `m`, or strict-depth nonlinear mixing plus complete-old response subtraction.

## Checkpoint 114: the r62 rank-one anchor cancellation is not universal

The 33-mode loss in Checkpoint110 is a property of aligned affine
second-difference triples, not a theorem about every regular r62 seam.  An
exact rational four-row motif has

\[
 r=(-3,1,1,2),qquad r^TU=r^TN=0,
\]

and a skew coefficient operator `A` satisfying every local anchor equation,
yet

\[
 \boxed{
 A\in\ker\mathcal T_{anc},qquad
 \langle\Phi_N(rr^T),A\rangle={16\over3}\ne0.}
\]

The construction embeds without changing this pairing into the literal
`n=1024,r=62,dim(E0)=929,dim(R)=33` dimensions while keeping `[U N]` full
column rank.  In the core cross-block model, put

\[
 B=\sum_i r_i v_i v_i^T,qquad
 K_x=\sum_i r_i x_i x_i^T.
\]

The complete rank-one tilted-wedge action collapses to the commutator scalar

\[
 \boxed{
 \langle\Phi_N(rr^T),A\rangle
 ={1\over2}\operatorname{tr}(A[B,K_x]).}
\]

The residual constraints kill the first moments `sum r_i v_i` and
`sum r_i x_i`; they do not force the two weighted second moments to commute.
The affine triple does force the relevant alignment and is therefore a
special zero.  The full construction and dimension lift are persisted in
`LANE_rankone_anchor_universality_counterexample.md`.

### Graph node: R62_RANKONE_ANCHOR_UNIVERSALITY_COUNTEREXAMPLE

- status: solved exact counterexample to a universal cubic no-go;
- detector: a coefficient-plane skew rotation with pairing16/3;
- implication: non-affine/overlapping motifs can retain diagonal `Z=rrT` modes;
- next gate: combine enough such commutator motifs to make the complete 561-mode anchored Gram positive.

## Checkpoint 115: an exact overlapping r62 construction survives all 561 local-anchor modes

The next gate in Checkpoint114 is now solved at the local-anchor level.  Split

\[
 W=V_{34}\oplus C_1\oplus Z_{27},
 \qquad X=V_{34}\odot C_1\subset E_0,
\]

and let `S(x)` denote the normalized cross matrix.  Use 34 overlapping
three-row groups.  Their residual labels in `Y=R^33` are

\[
 p_0=\sum_{k=1}^{33}e_k,qquad p_a=e_a,quad1\le a\le33.
\]

For a basis `f_0,...,f_33` of `V`, set

\[
 \mathbf1=\sum_{a=0}^{33}f_a,qquad v_a=f_a+\mathbf1.
\]

Within group `a`, take

\[
 \alpha=(1,-2,1),qquad
 \beta_a=(a,-2(a+1),a+2),
\]

with `u_at=alpha_t v_a+z_at` and cross coefficient
`(n_at)_X=S(beta_at v_a)`.  Because every coefficient direction is parallel
to the `V` projection of its anchor, the full detector algebra satisfies

\[
 \boxed{\mathfrak{so}(X)\subset\ker\mathcal T_{anc}.}
\]

For `Theta in Sym(Y)`, the exact wedge projection is

\[
\boxed{
 P_{\wedge^2X}\Phi_N(P\Theta P^T)
 =6\sum_{0\le a<b\le33}
 (D_a-D_b)(v_a^Tv_b)(p_a^T\Theta p_b)
 S(v_a)\wedge S(v_b),}
\]

where

\[
 D_a=6a^2+12a+8.
\]

All multipliers are nonzero.  The 561 functionals
`p_a^T Theta p_b` form a basis of `Sym(33)^*`: the pairs among1--33 read the
528 off-diagonals, after which the 33 pairs `(0,a)` read the diagonals.  The
561 bivectors `S(v_a) wedge S(v_b)` form a basis of `wedge^2 X`.  Hence the
displayed map is an isomorphism.  Every nonzero image pairs with an element
of `ker T_anc`, proving

\[
 \boxed{\operatorname{rank}\overline\Phi_N=561.}
\]

In the displayed ordered pair bases the determinant is explicit:

\[
 \boxed{
 \det=\pm35^{33}
 \prod_{0\le a<b\le33}1296(a-b)(a+b+2)\ne0.}
\]

The construction admits an exact regular completion to
`n=1024,r=62,dim(E0)=929` with `rank[U N]=991`, so its physical residual
space is exactly33-dimensional.  Full details, including independent
Veronese extension and the safe completion columns, are persisted in
`LANE_r62_full561_overlapping_anchor_design.md`.

This is an existence theorem for the intrinsic cubic/local-anchor quotient.
It does **not** yet prove simultaneous membership in the actual D14 prefix
chart, maximal first-Schur rank, a useful cubic singular value, or a deployed
acquisition schedule.

A final typing audit makes the chart caveat concrete.  The normal-Schur
derivation used elsewhere fixes the coefficient gauge

\[
 \operatorname{range}N\subset K=\ker U^T,
 \qquad U^TN=0.
\]

The displayed overlapping completion proves `rank[U N]=991` and
`R=ker U^T intersect ker N^T`, but it does not impose `U^TN=0`.  Indeed its
active cross bank has, group by group,

\[
 \sum_t\alpha_t\beta_{a,t}=6(a+1),
 \qquad
 \sum_t\beta_{a,t}z_{a,t}=-(3a+4)w_a,
\]

so `U^T N_X` is generically nonzero.  Projecting `N` to `P_KN` preserves the
physical residual space and rank under the expected regularity, but has not
been proved to preserve the automatic anchor-silent `so(34)` detector or the
explicit minor.  Therefore the theorem is exact for the intrinsic displayed
pair, while promotion to the fixed normal-Schur gauge requires either a
gauge-invariance proof or a `K`-orthogonal reconstruction of the same minor.

### Graph node: R62_TILTED_WEDGE_FULL561_OVERLAPPING_CONSTRUCTION

- status: solved exact full-rank theorem for the intrinsic local-anchor quotient;
- mechanism: 34 overlapping groups convert diagonal residual information into cross-group commutators;
- result: all561 `Sym(R33)` modes survive;
- remaining gates: `U^TN=0` normal-chart compatibility (or gauge invariance), actual-prefix realization, simultaneous affine/first-Schur minors, conditioning, and cost.

## Checkpoint 116: exact zero-bias L3L4/L3-cubed reset completion and its target-action screen

The source audit in Checkpoint111 exposes a cheaper completion than a new
carrier.  At zero reset bias, the omitted `L3 L4+L3^3/6` contribution to the
standardized off-diagonal raw ReLU pair moment collapses entirely to arrays
already present at the D11 reset seam.

Use the literal standardized conventions

\[
 c_{3i}={\kappa_{iii}\over\sigma_i^3},qquad
 c_{4i}={\kappa_{iiii}\over\sigma_i^4},
\]

\[
 P_{ij}={\kappa_{ijj}\over\sigma_i\sigma_j^2}
 ={(k21)_{ji}\over\sigma_i\sigma_j^2},qquad
 R_{ij}={\kappa_{ijjj}\over\sigma_i\sigma_j^3}
 ={(k31)_{ji}\over\sigma_i\sigma_j^3}.
\]

Let `rho` be the Gaussian correlation plane, `D_x=Diag(x)`, and

\[
 s_i=-{5\over96}c_{3i}c_{4i}+{35\over864}c_{3i}^3.
\]

With `phi0=1/sqrt(2 pi)`, the complete standardized increment is

\[
\boxed{
 \Delta_{std}=\phi_0\,\operatorname{OffDiag}\left[
 {R D_{c3}+D_{c3}R^T\over24}
 +{P D_{c4}+D_{c4}P^T\over32}
 -{5(PD_{c3^2}+D_{c3^2}P^T)\over96}
 +(D_s\rho+\rho D_s)
 \right].}
\]

The physical raw-pair/covariance increment is

\[
 \boxed{\Delta_C=D_\sigma\Delta_{std}D_\sigma}
\]

off diagonal.  At zero bias the corresponding omitted univariate mean term
vanishes, so the initial centered-covariance increment equals this raw
increment.  The diagonal must be supplied by a consistent univariate raw-2
completion; the displayed formula must never be copied onto the diagonal.
Its natural standardized order is `r^-3/2` when `c3=O(r^-1/2)` and
`c4,P,R=O(r^-1)`.

### Direct endpoint-adjoint scalar

For a symmetric local covariance adjoint `Lambda`, define

\[
 \widetilde\Lambda
 =\operatorname{OffDiag}(D_\sigma\Lambda D_\sigma).
\]

Symmetry combines each oriented pair, giving the exact tensor-free scalar

\[
\boxed{
\begin{aligned}
 \chi={}&\langle\Lambda,\Delta_C\rangle\\
 ={}&\phi_0\left[
 {1\over12}\langle\widetilde\Lambda,RD_{c3}\rangle
 +{1\over16}\langle\widetilde\Lambda,PD_{c4}\rangle
 -{5\over48}\langle\widetilde\Lambda,PD_{c3^2}\rangle
 +2\langle\widetilde\Lambda,\rho D_s\rangle
 \right].
\end{aligned}}
\]

This is four `O(n^2)` reductions.  Forward acquisition similarly needs only
elementwise scale/symmetrize operations on live `k3,k21,k4,k31,rho`; no new
moment carrier or matrix product is algebraically required.  A literal flop
census and buffer-liveness receipt remain necessary.

The matching label-free local radius is

\[
\boxed{
\begin{aligned}
 \|\Delta_C\|_F\le\phi_0\bigg[&
 {1\over12}\|D_\sigma R D_{c3}D_\sigma\|_F
 +{1\over16}\|D_\sigma P D_{c4}D_\sigma\|_F\\
 &+{5\over48}\|D_\sigma P D_{c3^2}D_\sigma\|_F
 +2\|D_\sigma\rho D_sD_\sigma\|_F\bigg].
\end{aligned}}
\]

After propagation, sum these bounds with the appropriate endpoint operator
norms.  If that coherent upper bound cannot reach the required residual
action, the route closes without a fit.

### Ownership and value gates

The identity itself is not an endpoint-value theorem.

1. The literal D11 density reset omits these cross orders, so they are not
   owned by `_four_pair_edgeworth`.
2. The underlying `k3/k21/k4/k31` arrays are incumbent sources.  A
   deterministic reset replacement may use them once; a separately fitted
   feature must still be projected through the complete CP3/CP4/star/Q24
   response span.
3. The formula is exact only at zero standardized bias.  Away from zero it
   needs the full derivative-7/9 kernels or a certified bias remainder; using
   it unchanged is an approximation, not a closure identity.
4. It corrects the covariance-reset/variance lane.  The banked exact-chain
   attribution assigns 17.46 of the 18.7 fixed-chain percentage points to
   that entire lane.  Thus even the optimistic hypothesis that this one
   completion reproduces **all** of the banked variance correction gives
   only17.46%, below the 20% fire bar and far below the approximately66%
   reduction needed for raw `1e-8`.  This is a comparator screen, not a
   universal ceiling on a jointly retrained nonlinear architecture.

For late suffixes, the banked variance comparator gives

\[
\begin{array}{c|c|c|c}
\text{cells}&\text{variance points}&\text{amplification to20\%}&
\text{amplification to raw }10^{-8}\\\hline
L12\text{--}L15&10.28&1.95&6.42\\
L11\text{--}L15&12.58&1.59&5.25\\
L10\text{--}L15&14.32&1.40&4.61\\
L9\text{--}L15&15.64&1.28&4.22
\end{array}
\]

These are minimum multipliers relative to the complete banked variance
direction, before old-span loss.  The actual promotion gate for a propagated
new endpoint column `z_perp` remains

\[
 {\bigl(|\widehat c|-\eta\bigr)_+^2\over G_+}
 \ge5.8818\times10^{-9}
\]

for a 20% raw move, or `1.9409e-8` for raw `1e-8`, with
`c=<e,z_perp>` and `G=||z_perp||^2` in the campaign normalization.

### Graph nodes

- `D11_L3L4_L3CUBE_ZERO_BIAS_IDENTITY`: solved exact local off-diagonal formula;
- `D11_L3L4_DIRECT_ADJOINT_ACTION`: solved four-reduction target action and norm radius;
- `D11_L3L4_BIAS_REMAINDER_GATE`: open; zero-bias reuse requires a full biased kernel or certified remainder;
- `D11_L3L4_COMPLETE_OLD_RESPONSE_GATE`: open for additive gain, once-only for deterministic replacement;
- `D11_L3L4_BANKED_VALUE_SCREEN`: fixed-chain comparator below the20% fire bar; coadaptation required for promotion.

## Checkpoint 117: disjoint common-scale and literal shared-z AQ have different r^-2 tails

Checkpoint113's conditional Gaussian law `N(m,L+sQ)` is exact for a
**disjoint** affine Gaussian block plus the common radial quadratic scale.  A
literal AQ neuron generally uses the same latent Gaussian `z` in its linear
and quadratic pieces.  Under the operator-isotropy closure, that shared-z
law instead has characteristic exponent

\[
 \boxed{
 \log\varphi(t)
 =-{r\over2}J\!\left({2q\over r}\right)
 -{\ell\over2}R\!\left({2q\over r}\right),}
\]

where

\[
 q=t^TQt,qquad \ell=t^TLt,qquad
 R(x)={2\over1+\sqrt{1+4x}}.
\]

Its linear H4 term is

\[
 {q^2+\ell q\over r},
\]

while the first omitted characteristic H6 term is

\[
 \boxed{
 -{1\over r^2}\left({10\over3}q^3+4\ell q^2\right).}
\]

Thus the independent-radial `q^3` completion is incomplete for the literal
shared-z AQ state: a broad mixed `ell q^2` source appears at the same order.
It can still be consumed tensor-free by the corresponding sixth-derivative
Gaussian/Q24 kernel because its source is a sum of two factored covariance
polynomials, but it is not the centered collective rank-one plane of
Checkpoint113.  The centered rank-one collapse therefore applies only to the
pure/disjoint scale-mixture branch.

The operator-isotropy premise itself needs a finite receipt.  For the actual
shared quadratic operator `B` and linear vector `v`, the obstruction is the
family

\[
 \boxed{
 \delta_k=v^TB^kv-\ell\,\tau(B^k),}
\]

with the normalized trace `tau`.  Nonzero `delta_k` produces identity-specific
anisotropic corrections outside the two scalar polynomials above.  A valid
provider must either carry those defects or bound their complete endpoint
action; setting them to zero by isotropy is an approximation.

### Graph nodes

- `AQ_SHARED_Z_RADIAL_MIXED_H6`: solved characteristic-level q3 plus ell-q2 tail at order r^-2;
- `AQ_SHARED_Z_VS_DISJOINT_SCALE_SPLIT`: ownership/model distinction; centered rank-one no-go is not universal to shared-z AQ;
- `AQ_OPERATOR_ISOTROPY_DEFECT_GATE`: open finite receipt based on `v^T B^k v-ell tau(B^k)`.

## Checkpoint 118: r63 full-seam determinant versus the much cheaper target-action Gram

The 20,832-dimensional determinant in Checkpoint112 is the correct gate for
matching **every** first-order seam-normal mode.  Endpoint value needs a
strictly smaller certificate.  Let

\[
 \mathscr S_1:\mathcal H_{AQ}\longrightarrow\operatorname{Sym}K
\]

be the quotient first-Schur map, with adjoint identity

\[
 \mathscr S_1^*Z=P_{\mathcal C}(ZN_0).
\]

For one complete-old endpoint/seam adjoint `lambda in Sym(K)`, its exact
first-order lever is

\[
\boxed{
 \gamma_\lambda
 =\|\mathscr S_1^*\lambda\|^2
 =\|P_{\mathcal C}(\lambda N_0)\|_F^2
 =\langle\lambda,G_{63}\lambda\rangle.}
\]

Thus the target is first-order dark iff

\[
 \lambda N_0\in\mathcal D_U,
\]

even if the full determinant fails elsewhere.  Conversely, with allowed
quotient parameter norm `R`, the maximum attainable signed scalar action is

\[
 \boxed{R\sqrt{\gamma_\lambda}.}
\]

Cancelling a required scalar seam action `a` needs parameter norm at least

\[
 \boxed{|a|/\sqrt{\gamma_\lambda}.}
\]

At a near-affine prefix scale `t`, replace `gamma_lambda` by
`t^2 gamma_lambda+O(t^4)`; the familiar `t^-1` conditioning appears directly.

For `q` endpoint adjoints collected as a column map `B`, the complete
target-relevant certificate is only

\[
 \boxed{G_B=B^*G_{63}B.}
\]

Positive definiteness of this `q by q` matrix is necessary and sufficient to
open their span at first order.  It must be formed after the complete-old
suffix projection; using raw local K21/K22 coordinates can credit duplicate
directions.

There is also an exact mismatch/value formula that does not require full
surjectivity.  If `d` is the actual cut mismatch and `H_B` is the final
Bellman value metric pulled back to the seam, then the irreducible
first-order endpoint error after optimally choosing the AQ section is

\[
\boxed{
 \min_h\|d-\mathscr S_1h\|_{H_B}^2
 =d^T\left[
 H_B-H_B\mathscr S_1
 (\mathscr S_1^*H_B\mathscr S_1)^+
 \mathscr S_1^*H_B
 \right]d.}
\]

This is the sharp target-action/value screen.  The full seam determinant is
needed only if the architecture claims an exact state section for arbitrary
future consumers.  A board-oriented fixed suffix can promote on `G_B`
instead, provided the section norm, cubic remainder, and complete-old
projection are all certified.

### Graph nodes

- `R63_TARGET_ACTION_DEFECT_GRAM`: solved scalar lever `||P_C(lambda N0)||^2`;
- `R63_TARGET_SUBSPACE_COMPRESSED_GRAM`: solved q-dimensional value gate after old projection;
- `R63_BELLMAN_IRREDUCIBLE_SEAM_ERROR`: solved exact H-weighted Schur residual;
- dependency: full `R63_DEFECT_PRINCIPAL_ANGLE_DETERMINANT` only for universal-state claims.

## Checkpoint 119: target-compressed r63 is an exact small solve, but the current package supplies none of its operands

This checkpoint audits the target-specific alternative to the universal
`20,832`-dimensional determinant against the literal repository and the
banked perfect-state endpoint target.  The conclusion is a **current-lineage
deployment close**, not a theorem that every future r63 architecture is
impossible:

> if a lawful r63 cut tuple, its prefix block `N0`, and a complete-old endpoint
> costate were resident, the target section would be a small exact solve;
> none of those three objects is resident or serialized, and acquiring even
> one of them exhausts the five-reset bill.

### 119.1 Literal target that the repository does contain

`public_g19_attribution_v1/PERFECT_STATE_CEILING.json` freezes the ten-network
oracle ladder

\[
  3.19722539477067\,10^{-8}
  \longrightarrow
  9.254687418209847\,10^{-9}
\]

from the three endpoint response families `(v,k3,k4)`.  The exact response
recurrence used by `run_perfect_ceiling.py` is, for each channel `a`,

\[
 r_{a,0}=0,
 \qquad
 r_{a,l}=\phi_l\,(r_{a,l-1}W_l)+\delta s_{a,l},
\]

where, writing `f(m,v,k3,k4)` for `edg2_relu_mean`,

\[
\begin{aligned}
 \delta s_{v,l}&=f(h_m,v_l^{true},h_3,h_4)-f(h_m,h_v,h_3,h_4),\\
 \delta s_{3,l}&=f(h_m,h_v,k_{3,l}^{true},h_4)-f(h_m,h_v,h_3,h_4),\\
 \delta s_{4,l}&=f(h_m,h_v,h_3,k_{4,l}^{true})-f(h_m,h_v,h_3,h_4).
\end{aligned}
\]

The ten truth/state/deployed/weight tuples needed to reconstruct these
vectors are all on disk for ranks

`35,100,150,200,250,300,400,500,600,650`.

Only `rvec_oracle_rank0035.npz` serializes the resulting three `1024`-vectors;
the other nine are recoverable by the frozen recurrence but are not stored.
The near-universal LOO coefficients are approximately

\[
 (\beta_4,\beta_3,\beta_v)=(0.16,0.33,0.82).
\]

These files give a concrete **endpoint target**.  They do not give an r63
seam adjoint: `r_a` is a forward response vector in output space, not a
matrix `lambda_a in Sym(K)`.

The exact panel residual is likewise reconstructible as

\[
 e=mc\_post[15]-deployed[15].
\]

It is observer truth and therefore diagnostic-only.  It cannot be read by a
deployed estimator.

### 119.2 Exact code-ready target section once operands are supplied

Let `q` complete-old seam costates be `lambda_a in Sym(K)` and let

\[
 T_a=\lambda_aN_0\in K\otimes E_0,
 \qquad
 R_a=P_{\mathcal C}T_a.
\]

Collect the `R_a` as the column operator `R:R^q -> H_AQ`.  Then

\[
 G=R^*R,
 \qquad
 G_{ab}=\langle R_a,R_b\rangle_F.
\]

For desired signed seam actions `a in R^q`, the exact minimum-norm linear
section is

\[
 \boxed{
 h_{min}=R G^+a,
 \qquad
 \|h_{min}\|^2=a^TG^+a,}
\]

provided `(I-GG^+)a=0`.  At physical near-affine scale `t`, replace this by

\[
 \boxed{
 h_{min}(t)=t^{-1}RG^+a,
 \qquad
 \|h_{min}(t)\|^2=t^{-2}a^TG^+a.}
\]

No universal Gram or determinant is needed.  A matrix-free implementation of
the defect projection is

```text
for each endpoint costate lambda[a]:
    T[a] = lambda[a] @ N0
    y[a] = solve_least_squares(D, T[a])
         = (D* D)^+ D* T[a]
    R[a] = T[a] - D y[a]                 # P_C T[a]
G[a,b] = frobenius_inner(R[a], R[b])
theta  = pinv(G) @ desired_actions
h      = sum_a theta[a] * R[a]
```

Here `D` is any matrix-free realization with `range(D)=D_U`.  Materializing
`D` is forbidden by memory: a float32 basis would occupy

\[
 20,832\cdot961\cdot992\cdot4
 =79,437,582,336\ \mathrm{bytes}
 \simeq73.98\ \mathrm{GiB}.
\]

By contrast, one target residual `R_a` is only about `3.64 MiB`; three use
about `10.91 MiB`.  Target compression solves the determinant/storage
problem, but only after the operands exist.

### 119.3 Nonlinear conditioning and remainder gate

Let the target-compressed finite-cut Jacobian be

\[
 B^*J_{63}(t)=tR^*+t^3K_3+t^5K_5(t),
\]

and suppose `||K_3||<=M3`, `||K_5(t)||<=M5`.  If `g` is the smallest positive
eigenvalue of `G`, then

\[
 \boxed{
 \sigma_{target}(t)
 \ge |t|\sqrt g-|t|^3M_3-|t|^5M_5.}
\]

A sufficient stable-section gate is

\[
 t^2M_3+t^4M_5\le {1\over2}\sqrt g,
\]

which gives inverse norm at most `2/(|t|sqrt(g))`.  For the linear
minimum-norm section above, the target-action remainder is bounded by

\[
 \boxed{
 \|\delta a_{nl}\|
 \le(t^2M_3+t^4M_5)\sqrt{a^TG^+a}.}
\]

This must be added to cut-model error, complete-old projection error, and
source error; none may be hidden by rescaling the fitted endpoint column.

### 119.4 Endpoint value is a second Gram, not the seam-action Gram

Positive `G` certifies controllability of the requested seam actions.  It
does **not** certify raw MSE.  Let `z` be the final complete-old-projected
output response produced by `h`, and let `e_perp` be the correspondingly
residualized deployed error.  In campaign normalization,

\[
 c=\langle e_\perp,z\rangle,
 \qquad
 V=\langle z,z\rangle.
\]

Adding `alpha z` changes MSE by

\[
 \Delta MSE(\alpha)=2\alpha c-\alpha^2V.
\]

The optimal gain is `c^2/V`.  For several endpoint columns `Z`,

\[
 \boxed{
 \Delta MSE_{best}=c^TG_Z^+c,
 \quad
 c=Z^*e_\perp,
 \quad
 G_Z=Z^*Z.}
\]

Thus a scalar VJP can select an action while a forward JVP/hybrid is still
needed to measure the response norm and cross-Gram.  A label-seeded scalar
VJP is an offline diagnostic, not a deployable source.

At the board raw baseline `2.9409071835573285e-8`, the exact gates are

\[
\begin{aligned}
 \Delta MSE_{20\%}&\ge5.881814367114657\,10^{-9},\\
 \Delta MSE_{raw<10^{-8}}&>1.9409071835573285\,10^{-8},\\
 R^2_{partial}&>0.659969,\\
 |\rho_{partial}|&>0.812385.
\end{aligned}
\]

The ten-net perfect-state target itself has explained fraction `0.710540`
and correlation `0.842935`.  Because its ideal floor is only
`7.45312581790153e-10` below `1e-8`, an implementation tied to this oracle
must retain at least

\[
 \boxed{
 1-{7.45312581790153\,10^{-10}\over
       2.271756652949685\,10^{-8}}
 =0.967192}
\]

of the oracle explained energy, corresponding to ingredient correlation at
least

\[
 \boxed{q_{section}\ge0.983459.}
\]

The robust endpoint-amplitude budget is only

\[
 \boxed{
 {\|z-z_*\|\over\sqrt N}
 <\sqrt{10^{-8}}-\sqrt{9.254687418209847\,10^{-9}}
 =3.79871\,10^{-6}.}
\]

The `0.812385` partial-correlation gate is the absolute board gate; the
`0.983459` ingredient gate is the stronger conditional fidelity needed when
claiming to approximate the banked perfect-state route.

### 119.5 Literal operand audit

The repository contains no executable r63 candidate.  The only `r63/aq63`
files are mathematical lane notes.  The relevant literal objects are:

| Required object | What exists | Why it is not the object |
|---|---|---|
| r63 `U,A` cut tuple | no NPZ/code artifact | D11 exposes only forward moment/CP/star state |
| `N0 in R^(961 x 992)` | absent | no common `Sym2(R63)` chart exists in the deployed state |
| complete-old `lambda` or `B` | absent | D11 is a forward loop and stores no reverse query |
| endpoint oracle target | ten truth/state tuples; one serialized `rvec_oracle` | forward target vectors, not seam costates; truth is illegal at deployment |
| CP4 factors | ephemeral `cp4_lambda,cp4_u`, cap2048 | represent `sum lambda f^4`; selecting63 rays does not construct 1024 matrices `A_i` or match `C,K21` |
| full-width backward state | P2R0F2M learned four channels | mean-tangent surrogate, not covariance/AQ costate; OOF failed |

The P2R0F2M read-only diagnostic does contain a label-seeded endpoint
residual pullback through the **mean** tangent
`diag(phi)W`.  Its trained target-free four-channel successor completed at
`3.233766009119748e-8`, slightly worse than its J parent
`3.2334958229470984e-8`, and failed its maximum-network gate.  This does not
prove r63 impossible, but it closes the claim that a ready learned resident
costate can be borrowed from that experiment.

The CP4 state cannot repair the typing gap.  `_cp4_reset` creates and prunes
signed fourth-order rays, and `_cp4_read_planes` consumes only their equality
reads.  A lawful r63 state instead needs

\[
 C=UU^T+\tfrac12XX^T
\]

and the AQ `K21` polynomial in `(U,A_i)`.  No exact map from the signed CP4
factorization to this joint tuple is implemented or implied by the equality
planes.

### 119.6 Exact information obstruction from the D4 state

The missing target contraction is not recoverable from the carried equality
state by algebra alone.  The parity pair of positive laws already constructed
in the strict-depth lane agrees on

\[
 C,K3,K4,K31,K22
\]

and therefore on every deterministic D11 equality/CP/star construction from
those objects, while differing in an all-distinct quartic coordinate.  After
a dense next weight, the difference enters an equality face as

\[
 \Delta\kappa_{aaab}
 =12\sum_u K_{bu}\prod_{r\ne u}K_{ar},
\]

which is generically nonzero.

Hence two admissible hidden laws can present the same legal cut state and
future weights but require different perfect-state endpoint corrections.  A
deterministic D4-to-r63 initializer emits the same state for both.  Without an
additional dark-fiber observable or an actual-law bound, its worst-case
target alignment has no positive source-only lower bound.

The information-minimal repair for one frozen scalar endpoint is the signed
dark contraction

\[
 \chi_\lambda
 =\langle\lambda,\Delta_{dark}\rangle
\]

together with the response-energy/Gram receipt.  Emitting all1024 target-free
means generically needs a high-rank output-costate family; replacing it by the
training residual is label leakage.

### 119.7 Arithmetic close and the depth/value trilemma

At `n=1024`, forming one supplied `lambda N0` costs

\[
 {961^2\cdot992\over1024^3}=0.853215
\]

product-equivalents.  Three cost `2.55965`.  Therefore:

| suffix | base r63 shell | remaining to49 | plus one `lambda N0` | plus three |
|---|---:|---:|---:|---:|
| four-reset full terminal Gram |45.68054|3.31946|46.53376|48.24019|
| five-reset terminal-factor read |48.07040|0.92960|48.92362|50.63005|

The table charges neither defect projection nor costate acquisition nor cut
construction.  A generic dense covariance costate needs at least one dense
conjugation per crossed transition and ordinarily two; the prior literal
audit prices five transitions at about ten equivalents before reset
adjoints.  Even an endpoint rank-one plane becomes generically full rank
after one Price/ReLU pullback.  The five-reset row is therefore closed even
for `q=1`; `q=3` is closed by `lambda N0` alone.  The four-reset row has no
room for its reverse acquisition after the three contractions.

Explicit defect storage is above the8-GiB cap, so matrix-free iteration is
mandatory and adds an unbilled conditioning-dependent cost.  Building the
absent `U,A,N0` from CP4 also requires a nonlinear seam solve and dense
factor/root work not present in either base row.

A three-reset full row costs about `41.26040` and could in principle buy more
offline algebra.  Its banked endpoint scale is only the `L13--L15` cells:

\[
 2.80+0.90-0.12+2.90+2.42=8.90\%
\]

of the frozen-chain baseline.  It needs `2.247x` amplification merely to
fire20% and `7.42x` to reach raw `1e-8`.  For a three-step causal feedback
with norm `gamma`, `S_3(gamma)=1+gamma+gamma^2`; fire needs approximately
`gamma>=0.724`, while the raw goal needs `gamma>=2.08`.  The latter cannot be
attributed to the contractive AQ ReLU reset.

The established four/five-reset comparison is therefore decisive:

- four resets need `5.80x` goal amplification and `gamma>=1.254`, while
  target acquisition does not fit its remaining bill;
- five resets need `4.82x` and near-unit `gamma>=0.982`, but have only0.9296
  equivalent left and cannot even form three `lambda N0` products;
- three resets have arithmetic room only by accepting a target-value window
  that needs an untyped expansive mechanism above2.

### 119.8 Verdict and exact reopen conditions

`R63_TARGET_COMPRESSED_DEPLOYMENT_CURRENT_LINEAGE` is **closed**.  No estimator
edit is justified because the code would have to invent `U,A,N0` or read
observer truth.  The exact formulas above are integration-ready only after a
future branch supplies all of:

1. a lawful, finite, conditioned D4-to-r63 cut tuple and `N0`;
2. a target-free complete-old endpoint query or a literal forward section
   rule whose output column can be graded without labels at deployment;
3. matrix-free `P_C` with a measured iteration/error bill;
4. target Gram `G` with stable positive eigenvalues and the nonlinear bound
   in119.3;
5. endpoint gain above the exact raw gate and, for the perfect-state claim,
   section correlation at least0.983459;
6. a total meter below49 equivalents and a literal8-GiB/120-second receipt.

The useful positive result is the direct formula `h=RG^+a`: a future resident
costate/N0 design should evaluate this target-sized solve and never return to
the universal20,832 determinant.  In the present repository, however, target
compression changes the size of the final solve but not the missing
information or reverse-time acquisition cost.

## Checkpoint 120: deployability audit for shared-z mixed H6 versus exact pair inheritance

This checkpoint answers a narrower implementation question than the preceding
AQ seam work: can the omitted shared-z sixth-order characteristic term

\[
 -{1\over r^2}\left({10\over3}q^3+4\ell q^2\right),
 \qquad q=t^TQt,\quad \ell=t^TLt,
 \tag{120.1}
\]

be inserted into the literal held f32 D11/Q24 reset without a new sixth-order
tensor or pair kernel?  The answer has two parts.

1. **Conditional AQ PASS.**  If the reset cut literally supplies the split
   covariance planes `Q` and `L`, every marginal and two-coordinate H6 face is
   an elementwise polynomial.  D11 already evaluates all seven total-degree-six
   derivative channels, so the reset addition is `O(n^2)` and needs no new
   Gaussian/Mehler contraction.
2. **Current D11 FAIL.**  The deployed state supplies only total covariance
   `C`, K3/K21, K4/K31/K22, CP3/star, and CP4/star.  It has no oriented AQ
   `(U,A_i)` bank and no `Q,L` split.  The split is not exactly identifiable
   from those totals.  Consequently (120.1) is not a lawful additive D11
   provider in the repository as it stands.

The already executed K33 and K42/K51 packages are the primary *observable*
D11 H6 source.  They are exact for their declared pair-inheritance action but
produce only a marginal sixth-cumulant vector at the next cut.  They do not
contain the polarized pair-H6 faces supplied by (120.1).

### 120.1 Literal D11 receiver audit

The relevant source is

`public_d11_util_halving_v1/estimator_cap.py`.

- `_DERIVATIVE_CHANNELS` already contains exactly

  ```text
  (0,6),(1,5),(2,4),(3,3),(4,2),(5,1),(6,0).
  ```

- `_edgeworth_weights` writes `weights[10:17]` as the seven faces of
  `L3^2/2`.
- `_four_pair_edgeworth` applies all17 weights at once to all four required
  raw pair moments `(M11,M21,M31,M22)`.
- `_univariate_edgeworth` already has an order-six column
  `(c1[6],c2[6],c3[6],c4_6)` with
  `c4_6=-24 sigma^4 alpha phi(alpha)`.
- `_full_relu_reset` then performs the authoritative diagonal overwrite,
  mean-outer subtraction, covariance repair, and connected-plane conversion.

Therefore a new connected H6 source must only change the degree-six
coefficient weights.  It must not invoke `_g_four_all` a second time, alter
`_Q24_TERMS`, or add a new derivative family.

### 120.2 Characteristic sign and complete face table

For two coordinates put

\[
 q=Ax^2+2Bxy+Dy^2,
 \qquad
 \ell=ax^2+2bxy+dy^2,
 \tag{120.2}
\]

where

\[
 A=Q_{ii},\ B=Q_{ij},\ D=Q_{jj},
 \qquad
 a=L_{ii},\ b=L_{ij},\ d=L_{jj}.
\]

Because `i^6=-1`, the minus sign in the characteristic exponent is a
**positive** connected sixth cumulant:

\[
 \boxed{
 \kappa_6(t)={1\over r^2}
 \left(2400q^3+2880\ell q^2\right).}
 \tag{120.3}
\]

Writing

\[
 \kappa_6(x,y)=\sum_{p=0}^6 {6\choose p}K_{p,6-p}x^py^{6-p},
\]

gives the four nonredundant faces

\[
\begin{aligned}
K_{60}={1\over r^2}[&2400A^3+2880aA^2],\\
K_{51}={1\over r^2}[&2400A^2B+1920aAB+960bA^2],\\
K_{42}={1\over r^2}[&480A^2D+1920AB^2+384aAD+768aB^2\\
                    &+1536bAB+192dA^2],\\
K_{33}={1\over r^2}[&1440ABD+960B^3+576aBD+576bAD\\
                    &+1152bB^2+576dAB].
\end{aligned}
\tag{120.4}
\]

The remaining faces are `K24=K42.T`, `K15=K51.T`, and the broadcast marginal
`K06=K60[j]`.  It is cheaper and less error-prone to add the already
factorial-normalized weights directly:

\[
\begin{aligned}
w_{60}={1\over r^2}[&{10\over3}A^3+4aA^2],\\
w_{51}={1\over r^2}[&20A^2B+16aAB+8bA^2],\\
w_{42}={1\over r^2}[&10A^2D+40AB^2+8aAD+16aB^2\\
                    &+32bAB+4dA^2],\\
w_{33}={1\over r^2}[&40ABD+{80\over3}B^3+16aBD+16bAD\\
                    &+32bB^2+16dAB].
\end{aligned}
\tag{120.5}
\]

Here `w_pq=K_pq/(p!q!)`.  In literal D11 order the once-only update is

```python
weights[10] += w60[None, :]   # (0,6)
weights[11] += w51.T          # (1,5)
weights[12] += w42.T          # (2,4)
weights[13] += w33            # (3,3)
weights[14] += w42            # (4,2)
weights[15] += w51            # (5,1)
weights[16] += w60[:, None]   # (6,0)
```

and the univariate degree-six coefficient becomes

```python
degree6 = (k3*k3/72.0 + w60) * inv_sigma6
```

before multiplying the existing `c[p,6]` rows.  No extra scale belongs in the
pair weights: `_q24_context` already attaches `sigma_i^-p sigma_j^-q` through
the derivative features.

### 120.3 Exact insertion and liveness schedule

For an AQ-augmented D11 host the insertion point is after
`pre_k4,pre_k31,pre_k22` have been assembled and before `_full_relu_reset`.
At that point the affine-pushed AQ coefficient bank must supply

\[
 U_{pre}=W^TU,
 \qquad
 A_{pre,j}=\sum_iW_{ij}A_i.
\]

Then

\[
 L=U_{pre}U_{pre}^T,
 \qquad
 Q=C-L
 \tag{120.6}
\]

is exact only if the same AQ tuple owns the total `pre_covariance C`; a
separate D11 PSD repair cannot be silently folded into `Q`.

The zero-allocation schedule is:

1. compute `diag(L)=rowsum(U_pre**2)` and `diag(Q)=diag(C)-diag(L)` for the
   univariate `w60`;
2. call the unchanged `_edgeworth_weights` so its ten product scratch planes
   are released;
3. form `L=U_pre@U_pre.T` into `fusion_context[15][0]`, set `Q=C-L`, and stream
   (120.5) through the remaining scratch planes into `weights[10:17]`;
4. discard `L,Q` before `_four_pair_edgeworth` and run the literal downstream
   reset unchanged.

At the final preactivation only the row-norm diagonal and the univariate
`w60` are needed; no full Gram is formed.  If the AQ covariance builder
already forms `L` or `Q` separately, retaining that view removes the only new
matrix product.  If it forms only a concatenated total Gram, the cheapest
exact recovery is the skinny `UU^T` Gram.  Forming
`Q=(1/2)XX^T` directly at `r=63`, where `X` has2016 columns, is about1.97
dense-product equivalents per reset and is strictly dominated.

### 120.4 Arithmetic and RSS receipt for the conditional AQ path

For the explicit in-place schedule behind (120.5), shared monomials need49
real arithmetic operations per pair entry and the seven accumulator updates
need seven more.  Forming `Q=C-L` adds one, giving

\[
 \boxed{57n^2+O(n)}
 \tag{120.7}
\]

real operations per full reset.  The exact skinny-Gram arithmetic is

\[
 \boxed{n^2(2r-1)}.
 \tag{120.8}
\]

With `n=1024,r=63`, fourteen pair resets and one final univariate reset, the
conservative FlopScope-style bill (f32 Gram, binary64 elementwise arithmetic
charged twice) is

\[
 14\{1024^2(125)+2(57)1024^2\}+O(15nr)
 <3.51\times10^9.
 \tag{120.9}
\]

This is below0.00160 of the `2^41` budget.  If `Q` or `L` is already retained,
the bill falls below1.68 billion.  These are exact algebraic upper formulas
for the stated circuit, not a substitute for an official operation-call
census.

The ten existing `fusion_context[15]` planes are dead after the L3-square
convolution, so they hold the split and monomials.  Thus a resident split has
zero incremental plane RSS.  A nonresident f32 `UU^T` temporary adds at most
4 MiB (8 MiB if conservatively materialized in fp64) and dies before the pair
kernel.  The source has no cross-layer lifetime.  A literal A/B/A run must
still verify the8-GiB and120-second contract.

### 120.5 Why current D11 cannot manufacture the split

Under the same isotropic model the linear H4 law is

\[
 \kappa_4(t)={24\over r}(t^TQt)(t^TCt).
 \tag{120.10}
\]

It suggests an elementwise diagnostic inversion

\[
\begin{aligned}
 Q_{ii}&={rK_{4,i}\over24C_{ii}},\\
 Q_{ij}&={rK_{31,ij}/12-Q_{ii}C_{ij}\over C_{ii}},\\
 K_{22,ij}&={4\over r}
 (Q_{ii}C_{jj}+Q_{jj}C_{ii}+4Q_{ij}C_{ij}).
\end{aligned}
\tag{120.11}
\]

The reverse K31 orientation, K22 equation, and `Q>=0,C-Q>=0` are necessary
consistency checks.  They do not make this an exact D11 acquisition:

- D11's K4/K31/K22 planes include CP4/star and identity-specific finite-law
  content, not only (120.10);
- a projection or PSD repair changes the law and hence the H6 source;
- even exact agreement through fourth order does not determine sixth order.
  For pure quadratic scalar examples, squared spectra `(1,1,0)` and
  `(4/3,1/3,1/3)` have equal second and fourth power sums but sixth power sums
  `2` and `22/9`.

More intrinsically, exact half-AQ has

\[
 \kappa_6(U)=60\operatorname{tr}A^6+360u^TA^4u.
 \tag{120.12}
\]

Equation (120.3) replaces these two terms by the semicircle/isotropy values

\[
 \operatorname{tr}A^6={40q^3\over r^2},
 \qquad
 u^TA^4u={8\ell q^2\over r^2}.
\]

Neither defect is observable from `C` and the proposed `Q` plane alone.  Thus
(120.11) is a target-free *model projection*, useful for a diagnostic only;
it is not a lawful exact provider.  A deployable exact claim needs the
oriented `(U,A_i)` state or a certified bound on the complete sixth-word
defect.

Finally, H6 alone is not the full `r^-2` density completion.  Exponentiating
the already-owned H4 operator also creates `L4^2/2`, a degree-eight channel.
The closed owned-Edgeworth sentinel tested that old product family and does
not own or refute the new connected H6 source, but an "exact through r^-2"
claim must include both.

### 120.6 Exact status of K33+K42+K51 pair inheritance

Let `K_s,6-s(i,j)` denote the connected post-reset cumulant with exactly two
distinct source coordinates.  The complete at-most-two-coordinate
contribution to the next marginal cumulant is

\[
\begin{aligned}
\kappa_{6,a}^{(\le2)}={}&
 \sum_iW_{ia}^6K_{6,i}
 +6\sum_{ij}W_{ia}^5K_{51,ij}W_{ja}\\
&+15\sum_{ij}W_{ia}^4K_{42,ij}W_{ja}^2
 +10\sum_{ij}W_{ia}^3K_{33,ij}W_{ja}^3.
\end{aligned}
\tag{120.13}
\]

The existing public K33 and K42/K51 sources verify the last three terms.  The
K33 factor is10 because its symmetric full matrix counts both orientations;
the oriented K42 and K51 planes use15 and6.  Their connected assemblies
subtract every lower partition, so the result is distinct from D11's
disconnected `K3^2/72` density term.

The public packages deliberately omit the first, marginal `K6_i`, term.
Adding the separately audited projected marginal source costs about139.88
million operations, but prior marginal-K5/K6 evidence was flat and it still
does not supply any sector with three or more distinct source identities:

```text
411, 321, 3111, 222, 2211, 21111, 111111.
```

Most importantly, (120.13) is a **marginal projection**, not a complete
bivariate H6 transport.  A single mixed allocation in the K42 contribution
to an output `(4,2)` face is

\[
 \boxed{
 8\,(w_a^3\odot w_b)^T K_{42}(w_a\odot w_b).}
 \tag{120.14}
\]

There are `n^2` pair-specific Khatri-Rao queries `w_a odot w_b`.  For generic
dense `K42` and `W`, evaluating (120.14) for all `(a,b)` is `O(n^4)` (or
requires an `O(n^3)` pair-feature bank).  K33 has the same obstruction.
Therefore the verified source/action can lawfully supply only the next
**marginal** K6 vector under the card budget.  The AQ polynomial is genuinely
additional: it closes all seven consumer pair faces in `O(n^2)` by imposing a
strong low-state law.

### 120.7 Correct causal D11 injection and old subtraction

For the observable pair-inheritance path, call the output of (120.13)
`raw_k6`.  Its only legal D11 receiver is

```python
degree6 = (k3*k3/72.0 + raw_k6/720.0) * inv_sigma6
weights[10] += raw_k6[None, :] / 720.0
weights[16] += raw_k6[:, None] / 720.0
```

followed by the unchanged `_four_pair_edgeworth`, diagonal overwrite,
centering, repair, and connected assembly.  The cross slots `(1,5)` through
`(5,1)` remain zero because they were not transported.

For an autonomous recurrence, the corrected mean and these augmented weights
must then be used to build the next K33/K42/K51 source; a stale parent mean or
stale parent weights breaks association.  The final native tape and native-J
correction must likewise be recomputed.  The existing coordinate-only runs
had incoming K6 identically zero and therefore did **not** validate this
nonzero two-reset association.

There is no additional hand subtraction inside the reset.  The source planes
are connected cumulants, and D11 has no connected L6 owner.  For an additive
value screen define the feature as the complete candidate output minus the
byte-identical source-off parent output; complete-old response residualization
happens only after that end-to-end difference is formed.  If the AQ source and
the pair-inheritance source refer to the same preactivation law, they are
alternative estimates of the same connected K6 and must not be summed: use
`K6_pair-K6_AQ` as a model-residual diagnostic or choose one owner.

### 120.8 Resource and value gates for the observable pair path

The measured coordinate-only bills are

\[
\begin{array}{c|r}
\text{source/action}&\text{incremental operations}\\\hline
K33&76,180,449,280\\
K42+K51&151,825,424,384\\
\text{sum}&228,005,873,664.
\end{array}
\]

One shared production endpoint was statically priced at29,420,544 operations;
the optional marginal source at139,880,448.  Hence a conservative unfused
complete-at-most-two-coordinate upper is

\[
 \boxed{228,175,174,656\text{ operations}.}
 \tag{120.15}
\]

This double-counts some coefficient/context work and is an upper bound, not a
new official census.  Added to the largest held f32-panel bill
`1,293,908,824,784`, it gives `1,522,083,999,440`, or about0.6922 of `2^41`.
Thus FLOPs do not close an oracle-scale raw gain, although the utilization
penalty is material.

The K42/K51 static process projection is `7,068,327,936` bytes.  Even naively
adding the entire K33 scratch ceiling `153,665,536` gives
`7,221,993,472 < 8 GiB`; a production stream can reuse the same planes and is
smaller.  RSS is statically feasible.  Runtime is not certified: the literal
K42/K51 singleton measured `0.5109498 s` against its frozen `<0.4 s` residual
gate and is terminally closed; K33 also terminally closed on its exact-bill
and pre-symmetry gates.  Those receipts do not invalidate the algebra, but
they mean no current package has deployment authority and the frozen spellings
must not be silently retried.

The endpoint value gate remains

\[
 {(|\widehat c|-\eta)_+^2\over G_+}
 \ge5.8818143671\,10^{-9}
\]

for a20% raw move and `1.9409071836e-8` for raw below `1e-8`.  Magnitude of
`raw_k6` alone supplies no alignment certificate.

### 120.9 Minimal already-banked feature export

The two terminal source runs used the same rank201 row (`mlp_id=287`) and the
same weight hash.  Their target-free arrays are

```text
K33_RANK0201_BUNDLE/K33_RANK0201_RAW_K6.npy
K33_RANK0201_BUNDLE/K33_RANK0201_L6_MEAN_BIRTH.npy
K42_K51_SOURCE_ACTION_RANK0201_BUNDLE/combined_raw_k6.npy
```

all in their respective public experiment directories.  Every K33 raw entry
is nonzero, so the exact local response factor on that frozen parent row is

```python
h6_response = k33_mean_birth / k33_raw_k6
pair_raw_k6 = k33_raw_k6 + k42_k51_raw_k6
pair_mean_birth = h6_response * pair_raw_k6
```

This is the minimal no-new-row export for an independent endpoint value
screen.  It is only a local-mean/JVP feature: it omits covariance/plane
feedback, marginal K6, all >=3-identity sectors, and the nonzero autonomous
association gate.  It must not be represented as a deployable estimator or
as evidence for the AQ mixed source.

### 120.10 Terminal decision

- `D11_PAIR_INHERITED_MARGINAL_H6`: **exact new connected source/action, but
  partial and current literal spellings terminally closed**.
- `D11_PAIR_H6_FULL_BIVARIATE_TRANSPORT`: **generic cost no-go** by
  (120.14).
- `AQ_SHARED_Z_H6_FACE_COMPILER`: **code-ready O(n^2) conditional PASS** with
  exact faces (120.4)--(120.5).
- `D11_TO_AQ_SPLIT_FROM_EQUALITY_PLANES`: **exact no-go**; (120.11) is only a
  model projection.
- `AQ_SHARED_Z_H6_CURRENT_ESTIMATOR`: **closed for the present repository**
  because no literal AQ split or sixth-word defect receipt exists.
- `PAIR_H6_RANK201_BANKED_VALUE_FEATURE`: **ready for independent read-only
  endpoint screening**, with no autonomous/deployment authority.

## 121. Full-688 q lineage and the first unexecuted physical cubic-costate arm

This section is a source and design audit only.  It launched no estimator,
training job, numerical screen, or target decode.  Its purpose is to remove a
stale status, enumerate the actual ideal-response assets, and specify the
shortest target-free experiment that is not already a relabeling of a failed
q-head, signed-pair, CP3, or learned-adjoint arm.

### 121.1 Exact ideal-response inventory

There are three different objects that earlier notes sometimes called a
"q oracle."  They must remain separate.

1. The alpha-0.75 J refit constructed, in RAM,

   \[
   q_l^*=e_l-\Phi_l\odot(e_{l-1}W_l),\qquad l=10,\ldots,15,
   \tag{121.1}
   \]

   where `e=truth-host-alpha075J`.  Its exact suffix-oracle endpoint MSE was
   `6.249252614225924e-9`.  The arrays `base_state`, `base_q`, `delta_q`, and
   `residual_final` were not persisted by
   `public_q24_qhead_full688_refit_v1/run_refit.py`; only the promoted model and
   result were retained.

2. A later cached audit reconstructed the same identity for the exact OOF-J
   trajectory.  The frozen tensors now exist at

   ```text
   public_full688_oracle_correction_fixed_head_cv_v1/OOFJ_SUFFIX_ORACLE_Q.npy
   public_full688_oracle_correction_fixed_head_cv_v1/OOFJ_SUFFIX_ORACLE_STATE.npy
   ```

   with shape `[688,6,1024]`, source hash
   `ff16a91fd2ccb229325ebd757d05712c6f7658538699257b887f37a1280b97e2`
   for q, and exact OOF-J suffix-oracle MSE
   `6.238892067519586e-9`.  The underlying full trajectories are also frozen:
   `patched_error.npy` has shape `[688,16,1024]` and
   `OOF_PREDICTIONS.npy` has shape `[5,688,16,1024]`.  Therefore arbitrary
   corrected-layer total-q teachers are reconstructible without a new MPS row.

3. The ingredient-specific true-k3 response exists only on the ten G19 panel
   nets.  `public_g19_attribution_v1/PERFECT_STATE_CEILING.json` gives

   \[
   M_v=1.4655837809914389\,10^{-8},\quad
   M_{v+k3}=1.0286083864840677\,10^{-8},\quad
   M_{v+k3+k4}=9.254687418209847\,10^{-9}.
   \tag{121.2}
   \]

   Thus the conditional cubic value is
   `V3=4.369753945073712e-9`.  There is no full-688 true-k3 tensor,
   delta-k3 vector, or pure-k3 q teacher in the repository.  A full-688 fit can
   lawfully use total delta-q, but calling that response "q3" would be false.
   Cubic ownership can only be checked after freezing, on the ten-net ideal
   column.

The fixed-head capacity audit is also binding.  A train-only uncentered spatial
PCA basis needs `991/1024` directions to retain the required held-network q
energy (`0.9677588528` at 991, versus `0.9667712557` at 990).  A few global
spatial heads are therefore closed.  This does not close a network-specific
full-width field constructed from that network's own weights.

### 121.2 What was already executed, and why it failed

The bidirectional smoother is **not** an unlaunched design.  The complete run is
in `public_full688_backward_adjoint_state_cv_v1/`.  Its exact result was

\[
 M_M=3.233766009119748\,10^{-8},\qquad
 M_J=3.2334958229470984\,10^{-8},
 \tag{121.3}
\]

a regression of `8.355853461528007e-5` fraction, with 330 wins and 358 losses.
The gross paired movement was `2.8390468007599816e-7`, but 98.699% cancelled;
only layer 2 improved, layers 3--15 regressed, and the last layer cancelled
95.2699% of the peak regression.  Its transpose transport was exact, but its
four backward sources were learned local sources, not a frozen endpoint
residual or a physical moment action.  It learned a second free feature path,
not a Bellman smoother.

The other relevant receipts are:

| arm | held result | terminal lesson |
|---|---:|---|
| alpha-0.75 full-688 q head | `3.066199675740282e-8` | only `1.636%` gain over its incumbent q head |
| native22 independent H256 capacity | `2.988906304823771e-8` | closed only `2.464820359873312%` of its base-to-oracle gap |
| signed-pair V4R2 | `3.024343612511056e-8` | `2.8371%` worse than the board raw; pair-on incremental gain only `0.0229673%` |
| learned backward M | `3.233766009119748e-8` | no gain over OOF J, large cancellation |
| CP3 diagonal action scalar | local q `5.397630556845628e-9` versus `5.397469698650961e-9` | negative aggregate reduction; correlations essentially zero |
| collective refit sentinel | `3.154306195654431e-8` to `3.110044895712371e-8` | `1.4032%`, below the 8% continue gate |

The evidence-based prior for another feature made only from the same local
raw20 coordinates is therefore low.  The next arm must add a genuinely new
network-specific operator action, not width or another local scalar.

### 121.3 Exact new target-free field

The raw tape already exposes, at every layer,

```text
raw20[...,3]  = alpha
raw20[...,5]  = Phi
raw20[...,6]  = pdf
raw20[...,10] = standardized total K3 = k3/sigma^3
```

The cubic part of the frozen D11 univariate mean is therefore exactly

\[
 c_l=-{\sigma_l\alpha_l\varphi(\alpha_l)\over6}\,g_{3,l},
 \qquad g_{3,l}=\frac{k_{3,l}}{\sigma_l^3}.
 \tag{121.4}
\]

For a row-vector forward tangent

\[
 T_{l+1}x=\Phi_{l+1}\odot(xW_{l+1}),
 \tag{121.5}
\]

the Euclidean transpose is

\[
 T_{l+1}^*y=W_{l+1}\bigl(\Phi_{l+1}\odot y\bigr).
 \tag{121.6}
\]

Hence the complete future cubic-action scan on layers 5--15 is

\[
 b_{15}=c_{15},\qquad
 b_l=c_l+W_{l+1}\bigl(\Phi_{l+1}\odot b_{l+1}\bigr),
 \quad l=14,\ldots,5.
 \tag{121.7}
\]

Equation (121.7) is an exact transpose accumulation of a **known physical
cubic mean-action field**.  It is not an endpoint-residual VJP and supplies no
label information.  It is also not itself a correction: D11 already owns
`c_l`, so adding either `c` or `b` to the estimator output would double count
old action.  Their legal use is as predictor coordinates for a new missing-q
head.

This representation is distinct from every executed arm above.  The old q
heads see local raw20 but not this weight-transposed future field; the CP3
screen sees one local provider split but no future transpose; signed-pair
propagates pair banks forward; and M learns arbitrary sources rather than
binding the source to (121.4).  The fixed-PCA no-go does not apply because
`b_l` is a different full-width vector for each network.

There is nevertheless no new information-theoretic observable here.  M saw
raw20 and the same transpose operator, so one of its four channels had (121.7)
inside its hypothesis class.  What remains unexecuted is the **hard binding**
of the source to the exact physical product (121.4), together with direct
delta-q supervision instead of the free multistate endpoint objective.  This
is a materially different inductive bias, but M's failure must lower its prior;
it cannot be advertised as new raw data.

### 121.4 Code-ready inference hook

The minimal frozen arm extends the existing 22-input q-head law by three
coordinates:

\[
 \operatorname{clip}(c_l/R_l),\qquad
 \operatorname{clip}(b_l/R_l),\qquad
 \operatorname{clip}(b_l/(\sigma_lR_l)),
 \tag{121.8}
\]

where every clip is to `[-8,8]` and `R_l` is fit-fold J's layer RMS.  The head
is one shared `25 -> 64 SiLU -> 64 SiLU` trunk with eleven zero-initialized
scalar heads, layers 5--15.  It has exactly

\[
25(64)+64+64(64)+64+11(64)+11=6539
\tag{121.9}
\]

parameters.  Literal float32 inference is:

```python
c = -(sigma * raw20[..., 3] * raw20[..., 6] * raw20[..., 10]) / 6.0
b[15] = c[15]
for layer in range(14, 4, -1):
    b[layer] = c[layer] + weights[layer + 1] @ (
        raw20[layer + 1, :, 5] * b[layer + 1]
    )

d = zeros(1024)
for layer in range(5, 16):
    qhat = sigma[layer] * layer_rms[layer] * head(features25(layer))
    d = raw20[layer, :, 5] * (d @ weights[layer]) + qhat
answer = native_host[15] + foldJ[15] + d
```

The same arithmetic is available in a packaged all-data model; `foldJ` above
is replaced by the frozen promoted J base.  No truth, ideal moment, MPS sample,
model identity, or fixed spatial basis appears at inference.

### 121.5 One lawful accuracy transaction

Reuse the already frozen balanced four-fold assignment, exactly 516 fit and
172 held networks per fold, with uint8 assignment hash
`0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c`.
For fold `f`, fit J and all normalizers only on its 516 rows.  Define

\[
 e_l^{(f)}=y_l-h_l-A_l^{(f)},\qquad
 q_l^*=e_l^{(f)}-\Phi_l\odot(e_{l-1}^{(f)}W_l),\qquad l=5,\ldots,15.
 \tag{121.10}
\]

As in the prior suffix oracle, the correction state starts at zero.  Exact
q therefore leaves only the transported layer-4 residual; it does not force
endpoint truth.  Before fitting, reconstruct this exact start-5 oracle and
close the route immediately if its pooled endpoint MSE is not below `1e-8`.

Train only the mean eleven-layer full-node loss

\[
 {1\over11}\sum_{l=5}^{15}
 \frac{\|\widehat q_l-q_l^*\|^2}{R_{q,l}^2},
 \tag{121.11}
\]

where each `R_q` is fit-fold-only.  Endpoint MSE is a score, never a training
loss or checkpoint selector.  There is one fixed epoch, seed, row order, and
architecture.  Run a paired `source_on` and `source_zero` transaction with
identical 25-input bytes, initialization, batches, and targets; the latter
sets the three fields (121.8) exactly to zero.  The already executed
independent-H256 arm is a frozen secondary comparator, not another tuned arm.

This paired construction separates new information from ordinary head
capacity.  `source_on` must beat `source_zero` in local-q and endpoint MSE in
every completed fold.  No all-688 model is fit unless the complete OOF gate
passes.

### 121.6 Exact incremental resource shell

Using the same conservative head census as the frozen q-head lineage, the
eleven-layer input-25 head costs

\[
11(1024)\{11391+8256\}=221,303,808
\tag{121.12}
\]

operations, including its feature/nonlinearity reserve.  The ten reverse
matvecs in (121.7) cost `10*2*1024^2=20,971,520`; the eleven physical q
transports cost `23,068,672`; and a 16-operation-per-node reserve for forming
and scaling (121.4), (121.8) is `180,224`.  Thus

\[
 \boxed{C_{\rm cubic\ costate}=265,524,224}
 \tag{121.13}
\]

operations, or `0.000120746437460184` of `2^41`.  Storing both `b` and `c` for
all eleven layers in float32 uses at most `90,112` bytes; `c` may instead be
recomputed, halving that.  The weights, raw20, sigma, and J state are already
live.  This is an arithmetic/RSS receipt, not a runtime receipt; wall time is
metered only after an accuracy pass.

### 121.7 Fail-fast value gates and honest prior

With f32-panel mean utilization `0.5709125`, adding (121.13) gives provisional
`u=0.571033246437460`.  Relative to bank adjusted score `1.78362e-8`, a 20%
package move requires

\[
 M_{20}=\frac{0.8(1.78362\,10^{-8})}{u}
       =2.49879671438058\,10^{-8};
 \tag{121.14}
\]

30% requires `2.18644712508300e-8`, and adjusted break-even is
`3.12349589297572e-8`.  These are planning numbers; the exact deployed cut
must be re-billed before packaging.

Use fold 0 as a predeclared sentinel.  Its prior base and oracle are
`3.0923227278552953e-8` and `6.1455295488600225e-9`.  The sentinel must beat
both zero-source and independent H256, have at least 60% network wins and
paired `z>=3`, and reach the halfway-to-package threshold

\[
 M_{\rm sentinel}\le
 {3.0923227278552953\,10^{-8}+M_{20}\over2}
 =2.79555972111794\,10^{-8}.
 \tag{121.15}
\]

That already demands a `9.59677%` raw reduction and `11.9770%` of the exact-q
gap.  Failure closes this spelling without the remaining folds.  If it passes,
run folds 1--3 untouched.  A standalone package candidate requires pooled OOF
raw no greater than (121.14), every fold better than `source_zero`, at least
60% pooled wins, and `z>=3`.

The honest prior is only about 2--7% exact-q-gap closure: native22 H256 achieved
2.465%, the signed-pair incremental field 0.023%, the collective sentinel
1.403%, and the learned adjoint zero.  Fold 0 needs 23.9540% total gap closure
to reach (121.14), about ten times the strongest prior.  Raw below `1e-8`
needs 84.4438% closure.  Therefore this is a cheap, scientifically clean
fail-fast discriminator, not a likely redemption claim.

There is one additional value warning.  Removing the entire measured cubic
value from board raw gives

\[
2.9409214512554626\,10^{-8}-V_3
=2.503946056748091\,10^{-8},
\tag{121.16}
\]

which at provisional `u` is a `19.8351%` adjusted improvement.  A pure cubic
arm would need `101.1784%` of measured `V3` to clear the standalone 20% bar.
Thus even a nearly perfect q3 owner should be held for a tiny stack unless it
also captures another residual component.

After the all-four-fold gate, and never as a tuning input, evaluate cubic
ownership on the ten G19 nets.  Concatenate the final candidate-minus-zero
endpoint vector `z` and the complete-old-residualized ideal k3 endpoint column
`g3`; report

\[
 a_3={\langle z,g3\rangle^2\over\|z\|^2\|g3\|^2}
 \tag{121.17}
\]

and the fitted sign.  This diagnostic distinguishes a genuinely cubic
provider from a total-q feature that wins through another channel; it cannot
retroactively select or retune the model.

### 121.8 Terminal statuses

- `FULL688_TOTAL_Q_IDEAL_ASSETS`: **available and reconstructible**; OOF-J
  suffix q/state are persisted, and arbitrary start layers can be rebuilt from
  frozen trajectories, gates, and weights.
- `FULL688_TRUE_K3_TEACHER`: **not available**; true-k3 is ten-panel-only.
- `FULL688_FIXED_SPATIAL_Q_HEAD`: **closed**; held q energy needs 991/1024
  directions.
- `FULL688_LEARNED_BIDIRECTIONAL_ADJOINT`: **executed and closed**, not
  unlaunched.
- `FULL688_D11_CUBIC_ACTION_REVERSE_FIELD`: **exact target-free representation,
  apparently unexecuted**; source audit found no prior raw20-field10 recurrence
  of (121.7).
- `FULL688_CUBIC_COSTATE_DELTA_Q_HEAD`: **one sentinel fully specified but not
  authorized**; close at (121.15), otherwise request separate authority for the
  untouched remaining folds.

### 121.9 Design-only freeze

The sole queued successor is frozen at

```text
public_full688_d11_cubic_action_transpose_q_sentinel_v1/
```

with exact roots:

```text
README.md             8aa71c41c9be80284ee28ec2232b4d12ea63abe2782cf19931927640217c7dcc
PREREGISTRATION.json  861e97a719cbedf6bf198400ae7063370f6d3313042deaecb82e399f862b8235
FREEZE_RECEIPT.json   eb8bb96f85b1314c4a6665271f06b85113113ad2ce4d9f190263f56938d089ed
```

All 688 native `raw20/sigma/host` rows and matching weights already exist as
target-blind frozen primitives.  The derived `c` and `b` fields do **not**
exist and this freeze created no cache.  It also opened no target bundle and
created no runner, claim, model, or result.  Launch is interlocked behind the
currently active root-scheduled 82-arm and requires a separate exact-byte
execution release.

## 122. Terminal fold-0 cubic-action transpose result

The 82-arm closed, the exact-byte runner received independent GO, and the sole
authorized fold-0 sentinel executed once as PID 32501.  The durable pre-target
claim is `e64d73da...`.  Before fit target values were materialized or used,
the held start-5 oracle identity gave

\[
 M_{J,0}=3.0923227278552953\,10^{-8},\qquad
 M_{q^*,0}=4.698527912579728\,10^{-10}<10^{-8}.
\tag{122.1}
\]

Thus the action/transport lane has far more than the required capacity.  The
provider failed.  The exact paired metrics were

\[
\begin{array}{c|cc}
 & \text{source-on} & \text{source-zero}\\ \hline
 M_{15} & 3.033603794829636\,10^{-8} & 3.033159078239202\,10^{-8}\\
 M_q & 4.652187726788149\,10^{-9} & 4.648566510428816\,10^{-9}.
\end{array}
\tag{122.2}
\]

The appended field worsened endpoint MSE by `4.44716590434e-12` (0.01466%)
and local-q MSE by `3.62121635933e-12` (0.07790%).  It also lost to frozen
independent256 (`3.0220317427127064e-8`) and missed the material fold gate
(`2.79555972111794e-8`).  Although 74.42% of networks improved over bare fold
J with `z=7.576`, source-zero improved slightly more; those passes belong to
the ordinary 22-feature q head, not to `c,b`.

Cached-result-only analysis confirms no hidden subgroup: source-on beats
source-zero on 48.84% of rows and its paired z against source-zero is -0.214.
The source-on appended input weights changed materially from initialization,
so the null result is not caused by the model ignoring them.  Its sampled fit
objective was also worse than source-zero.  The conclusion is structural:
`c` is old-owned and locally reconstructible, while `b` is merely a
target-blind transpose readout of that old source, not an observable of the
missing ideal q.  This is a provider/observability failure, not an action
capacity failure.

Terminal bindings:

```text
runner  c47c36caf2877f1b31b76446d29817e844d1318f206d7b9040192d8dc0d18b37
release 99122771701ee879b264d842559ad5b69927a579d700cfdbfa449c13c0eb54ce
claim   e64d73daa5cae27b311822fdaf60bd28af1197c31fd4cb6886c6538ea56c5f47
result  e1ed27d0882dddb09698da78d167f7e34096ce44742b210b64792a3d20babd7f
models  868969ca197f19d1e2e70b7bbf51aa83ea7724b4571122eeab3826eb70437f20
```

Status `FULL688_CUBIC_COSTATE_DELTA_Q_HEAD` is now **closed measured**.  The
binding decision is `CLOSE_THIS_SPELLING__DO_NOT_RUN_FOLDS_1_TO_3`; retry,
retuning, feature caching, all-data fit, integration, meter, and board action
are all forbidden.  The complete receipt is in
`public_full688_d11_cubic_action_transpose_q_sentinel_v1/TERMINAL_RECEIPT.md`.

## 123. Post-q physical V/K3/K4 fallback audit

The owned L3/L3-squared four-field successor also closed on its sole fold:
parent-to-exact-q gap recovery was `0.0009354760270741128`, versus the frozen
`0.08` requirement, although its source-on arm beat source-zero locally and
at the endpoint.  Its result is
`d84c5831d3392961d04fcf23323794ca75858190924672aa21c4e496731b98f5`.
Together with (122), this says the exact suffix has ample q-action capacity
but neither tested target-blind q source reveals a material share of it.

The dormant physical-state design has a stale activation interlock: it is
reachable only when the exact start-5 q oracle fails `1e-8`, whereas (122.1)
passed by a large margin.  The frozen v1 bytes are therefore unreachable and
must not execute.  A lawful post-q v2 successor would instead bind the oracle
pass plus terminal failure of the finite frozen q-provider portfolio.

For the physical perfect-state mechanism, one provider-decomposition scalar
per predicted degree is information-minimal.  At fixed total variance, k3,
and k4, the three independent provider redistribution fibers form a
three-dimensional direct sum; fewer than three added scalar reads leave a
nonzero fiber direction invisible.  The literal minimum reads are

\[
\begin{aligned}
 e2_j&=\operatorname{raw2}_{j}-\operatorname{diag}M11^{pair}_j,\\
 uV_l&=(W_l^{\circ2})^T e2_{l-1}/\sigma_l^2,\\
 u3_l&=\operatorname{diag}(K21^{CP3}_l)/\sigma_l^3,\\
 u4_l&=K4^{CP4}_{l,iiii}/\sigma_l^4.
\end{aligned}
\tag{123.1}
\]

Native20 already contains total k3 but not total k4.  Thus a public-tape head
adds four width fields—`k4/sigma^4,uV,u3,u4`—even though only the last three
are provider splits.  In a fused D11 implementation total k4 is a zero-product
exposure and `u3,u4` are reductions of live operands.  Only `uV` adds a skinny
action.

The first exact-Gaussian reset has `e2_0=0`, so the dormant design's fifteen
variance pushes overcount one identically zero action.  The exact minimum is
fourteen squared-W pushes plus fourteen fused response transports:

\[
 28\,1024(2047)=58,691,584\ \text{FLOPs}
 =0.02734375\ \text{dense equivalents}.
\tag{123.2}
\]

Including the 24->32->3 head and its existing scalar reserve gives a total
upper arithmetic bill `90,148,864`, or `0.04199934355153884` dense
equivalents, with a 256 KiB streaming reserve and no new dense plane.

The single falsifier is rank35 held against the other nine panel networks,
training only standardized `(delta v,delta k3,delta k4)` state labels.  With
the frozen rank35 LOO betas, require gates-off source-zero parent identity,
finite state, combined response cosine at least `0.983459`, explicit endpoint
response RMS error at most `3.79871e-6`, noise-corrected MSE below `1e-8`, and
nonnegative component-wise response dots.  The exact held physical oracle is
`7.153517343378912e-9`.  Failure closes the spelling; pass requests unchanged
nine-net LOO authority.  There is no source-only alignment theorem because
these equality-provider splits do not identify arbitrary all-distinct dark
fibers.

The complete static receipt is
`public_v34_physical_state_response_sentinel_v1/INDEPENDENT_POST_Q_FALLBACK_CROSS_AUDIT.md`.

## 124. V34 v2 static audit and rank-35 pre-exposure correction

The append-only v2 static algebra at
`public_v34_physical_state_response_sentinel_v2/` passes its typed checks:
native20 plus `[total4,uV,u3,u4]` is a 24-field interface with three minimal
provider-split coordinates; the first exact-reset variance action is zero;
the 24->32->3 head has 899 parameters; and the exact L1-start compiler is

\[
 r_l=\phi_l(r_{l-1}W_l)+\beta_Vs^V_l+\beta_3s^3_l+\beta_4s^4_l.
\tag{124.1}
\]

Linearity proves by induction that (124.1) equals the three separately
transported columns, and exact predicted physical gaps reproduce the frozen
perfect-state column.  The resource shell is also correct: fourteen `W^2`
skinny actions plus fourteen signed-W actions, `58,691,584` dense-action
FLOPs, and `90,148,864` arithmetic operations excluding `491,520`
comparisons (`0.04199934355153884` dense equivalents).

However, an independent experimental-validity audit found a binding
pre-exposure problem.  Rank35's exact physical oracle floor, its LOO betas,
and its choice as the favorable held network were visible before the v2
freeze.  Therefore the written rank35 9/1 split and cosine/RMS/raw gates are
valid descriptive parity checks but not a fresh confirmatory falsifier.  V2
must not receive an execution release even after two algebra audits.

The lawful next artifact is append-only V2R1: select a fresh held network by
a hash rule frozen before new state-label generation/open, seal fit and held
MC state manifests separately, and audit the runner/release chronology before
any fit or held access.  No V2 runner exists, which is correct at this point.
The independent receipt is
`public_v34_physical_state_response_sentinel_v2/INDEPENDENT_DIRECT_DIAGRAM_STATIC_AUDIT.json`.
## Checkpoint 125 — V34 V2R1 fresh-panel and block-statistics audit

The append-only V2R1 path is now statically coherent but remains completely
unauthorized for data or execution.  The immutable stage-1 bytes are release
repair `c7e4aafa...`, row-0 selection receipt `5f6f4c43...`, and freeze
`f816c400...`.  The append-only panel binding `16c4eb19...` additionally binds
the winner-free selector `589e3d8c...`, the complete receipt `22be0d94...`, and
the H4 confirmatory protocol `f978719a...`.  The target-blind panel order is
global ranks `[757,746,780,759]`; row 757 is fail-fast only and all four rows,
not row 757 alone, are required for package evidence.  Rank 35 is historically
exposed and is excluded from both fit and held roles.

I added a standalone data-free reference scorer in
`public_v34_physical_state_response_sentinel_v2r1/block_statistics.py`.  Its
synthetic verifier covers the literal two-stream `16 x 500,000` law, exact
off-diagonal block MSE, paired on-minus-zero gain, two-sample delete-one
jackknife, raw and corrected adverse four-SE gates, cross-response algebra,
replica-ball inequalities, oracle residual construction, independent panel
aggregation, byte parity, sequential stop law, official-target-last
chronology, and fail-closed boundaries.  The implementation uses MC blocks as
the only replication units; the 1024 output coordinates remain one vector
functional.  Synthetic verification passed 15/15 fixtures under Python
3.11.5 / NumPy 1.26.4 without opening repository data or targets.

The sharp statistical distinction is binding.  For a prediction frozen before
MC,

```text
M_U(y)=sum_(a!=b)<T_a-y,T_b-y>/(32*31*1024)
```

is exactly unbiased for squared error to the MC-law endpoint mean, and the
paired statistic cancels the common block's quadratic sampling noise.  The
physical EDG2 response is nonlinear in estimated moments, so its finite-P
cross products are *not* called unbiased.  Both the raw and two-sample
jackknife-corrected scalar must pass the same adverse four-SE gate, and the
replica ball is only a deterministic guard against either observed replicate,
not a proof that the unknown population response lies in that ball.  Four-SE
is a frozen nominal decision rule rather than distribution-free coverage.
Consequently `P=8M x 2` is not analytically certified to resolve the threshold:
a wide or boundary-crossing interval is terminal
`MC_RESOLUTION_INSUFFICIENT`, with no adaptive samples or replacement row.

The authoritative raw branch is `OFFICIAL_TARGET_LAST_RAW`.  Every model,
normalization, provider, MC, compiled-response, and response-gate hash freezes
before the current row's official endpoint can become program-visible.  That
last reader is used only for exact raw `<1e-8` and source-on-minus-zero `<0`;
all MC state/capacity/cosine/RMS/component gates remain conjunctive.  The
future reader must disclose page-level decode if physical row pruning is not
provable.

The provider path remains exactly `native20+[total4,uV,u3,u4]`, a 24→32→3
pointwise head with 899 parameters and no cross-Gram, endpoint, q, network-ID,
or residual input.  The exact L1 recurrence costs fourteen squared-weight and
fourteen signed-weight actions, with conditional arithmetic upper
`90,148,864 = 0.04199934355` dense-GEMM equivalents, 3,596 parameter bytes,
and 256 KiB streaming workspace.  Execution is still HOLD pending the literal
D11 capture adapter/parity receipt, fit and label manifests, isolated MC
generator, trainer/model freeze, scorer integration, releases/claims,
official-target reader, and full meter/RSS receipt.

## Checkpoint 126 — raw-block generator sufficient-statistic reference

Pending the production implementation, I pinned the generator/scorer seam in
`public_v34_physical_state_response_sentinel_v2r1/mc_block_reference.py`.
For each stream the only accepted raw payload is float64 pre-ReLU sums of
orders 1--4 with shape `[16,16,1024,4]`, float64 post-ReLU sums of orders 1--2
with shape `[16,16,1024,2]`, and sixteen int64 counts equal to 500,000.  These
statistics reconstruct pooled and every delete-one `m,v,k3,k4,post_mean`
without reopening samples.  Endpoint block means are the order-one post sums
at layer 15 divided by their own counts.

The exact batch schedule must stop at every block boundary:

```text
7 * 65,536 + 41,248 = 500,000.
```

A generic 65,536 loop that lets a batch straddle two 500k blocks is therefore
invalid even if its final stream count is eight million.  The reference also
reproduces all eight frozen PCG64DXSM seed digests/integers for the ordered
four-row panel.  Six synthetic fixtures passed: batching, seeds, raw cumulant
reconstruction, all delete-one states, endpoint extraction, and fail-closed
shape/dtype/count handling.  This pins accumulator semantics only; the future
generator must still prove float32 normal generation/network propagation,
isolated inputs, exact completion, durable seals, and zero model/target access.

## Checkpoint 127 — held-scorer oracle law and production HOLD

I added the append-only data-free audit reference
`public_v34_physical_state_response_sentinel_v2r1/held_scorer_contract_reference.py`
and eight adversarial fixtures in `verify_held_scorer_contract_reference.py`.
The reference and verifier hashes are respectively `efacffb6...` and
`6ea810f8...`; all eight fixtures passed under the frozen Python 3.11.5 /
NumPy 1.26.4 stack.  Receipt `HELD_SCORER_CONTRACT_STATIC_VERIFICATION.json`
has SHA-256 `0fd101dd...`.  No repository data, targets, MC payloads, weight
arrays, model arrays, estimator, or training worker was opened.

For one stream with sixteen equal blocks and a prediction `y` frozen before
MC, the conventional noise-corrected endpoint MSE is exactly

```text
||Tbar-y||^2/n
  - sum_b ||T_b-Tbar||^2/[16*15*n]
= sum_(a!=b)<T_a-y,T_b-y>/(16*15*n).
```

Hence the existing per-stream off-diagonal U-statistic is the exact required
source-on score.  The same noise correction cancels in on-minus-zero, so both
streams must separately satisfy `M_s(on)-M_s(zero)<0`; the current moving
production scorer omitted those two comparisons.

The physical-oracle residual has a different type.  Let

```text
e_s = Tbar_s - deployed - F(moments_s)
e_s,-b = Tbar_s,-b - deployed - F(moments_s,-b).
```

The conventional per-stream correction is

```text
M_oracle,s = ||e_s||^2/n
  - (15/16n) sum_b ||e_s,-b - mean_c e_s,-c||^2.
```

For a linear block mean this reduces exactly to the off-diagonal U-statistic.
For the nonlinear EDG2 map it is only the frozen first-order jackknife noise
correction, not an exact finite-P unbiased identity.  Likewise, compiling a
separate physical response from each 500k block and applying an off-diagonal
U-statistic to the joint residuals estimates the *block-law* residual, not the
population physical-oracle residual.  A scalar counterexample with block
moments `x_b in {-1,+1}` and nonlinear response `F(x)=x^2` gives block score
one while the pooled residual is zero.  The independent-stream cross residual
plus both families of delete-one recomputations, raw/corrected adverse four-SE
upper gates, and the two conventional stream guards are therefore all
necessary and must be labeled honestly.

The moving scorer remained HOLD at this checkpoint: it lacked the joint
block-oracle diagnostic, per-stream oracle corrections, per-stream on<zero,
first-Gaussian analytic sanity, direct RMS jackknife, confidence-domain
resolution, complete stream/capture authentication, and scorer release/claim
binding.  Its result serializer also sent NumPy pseudovalue arrays directly to
JSON.  The official-target reader still accepted caller-supplied endpoint
arrays and hash strings, violating the H4 `ffac2c13...` hard firewall; it must
authenticate the durable claim, release, response freeze/payload, arrays, and
target manifest before opening the exact row.  No production bytes are called
stable or executable yet.

## Checkpoint 128 — synthetic production-plumbing audit

I added `verify_production_plumbing_synthetic.py` (SHA-256 `e285cd94...`) as
an independent zero-data audit of the moving production modules.  Five
fixtures pass:

1. An in-memory fake D11 module exercises the literal capture hooks for all
   fifteen targets.  It proves the initialization q38 read is excluded, the
   first `uV` is exact byte zero, `e2_previous[0]=0`, reset discrepancy `t-1`
   feeds target `t`, all seam counts are exactly 15/14/15/15/15/15/1, the
   native 20 columns are byte-preserved, and hooks restore after capture.
2. A tiny response recurrence proves L1 starts from zero and uses neither W0,
   W1, nor gate0.  Target `t>0` uses `gate[t]*(state@W[t+1])`, matching a
   separate provider-step calculation component by component.
3. A tiny PCG64DXSM stream proves float32 normal/matmul/ReLU propagation,
   float64 raw-power accumulation, and exact no-straddle block totals against
   a separate loop.
4. Temporary append-only artifacts reject both a second O_EXCL write and a
   symlinked parent escape.
5. The frozen 899-parameter NumPy deployment spelling exactly matches an
   independently expanded float32 affine/ReLU/affine calculation.

These passes certify algebraic plumbing only and will be rerun against the
eventual stable hashes.  They do not clear the scorer, official-target
firewall, resource, chronology, release, claim, or real-data holds.

### Binding oracle-name reconciliation

Root resolved the phrase `cross/block/jackknife upper`: it denotes the
independent full-stream cross residual `e_A^T e_B/1024` together with the
sixteen A-side and sixteen B-side delete-one-block recomputations and their
two-sample jackknife upper gate.  It does **not** denote an off-diagonal
U-statistic on 32 separately compiled nonlinear single-block responses.  The
latter remains evidence-only because it targets the 500k block-law residual,
as the `F(x)=x^2` counterexample proves.  Production is correct to persist it
as a labeled diagnostic but must not promote it into the oracle-capacity gate.

### Component-evidence versus deploy-cost split

The moving `response_runtime.compile_response` keeps three separate endpoint
states so the held scorer can gate V/K3/K4 signs.  It therefore performs
`3*14=42` signed weight matvecs, not fourteen.  The frozen 28-action resource
claim (`14` squared-weight uV actions plus `14` signed response actions) is
valid only for a separate combined deploy recurrence

```text
s_t = gate_t * (s_{t-1} @ W_t) + sum_c beta_c source_{t,c},
```

with the L1 transport elided from exact zero.  Component-separated compilation
is lawful off-card evidence but cannot be passed to the meter as the billed
challenge spelling.  A stable implementation must land and hash the
combined-only path, prove endpoint equivalence to the sum of the three held
columns, and meter the complete estimator through that path; otherwise the
literal census is 42 signed plus 14 squared actions.  Either fits the numeric
budget, but only the fused path supports the declared exact receipt.

## Checkpoint 130 — seed-domain HOLD supersedes Checkpoint 126 seed receipt

The raw-block reconstruction and block-boundary algebra in Checkpoint 126
remain correct, but its eight seed receipts are **revoked**.  Binding
`RELEASE_REPAIR.md` line 113 requires each stream seed to be a deterministic
function of `(mlp_seed, raw-weights SHA, stream tag A/B)`.  Both the moving
generator and the prior `mc_block_reference.py` instead hashed
`split/global-rank/mlp-id/mlp-seed/replicate` and omitted the raw-weights SHA.
Consequently the prior `MC_BLOCK_REFERENCE_STATIC_VERIFICATION.json` cannot
authorize generation even though its internal replay passed.

No MC stream has run.  Before any release, both generator and reference must
append the exact selected row's `raw_weights_sha256` and explicit A/B tag to
the seed preimage, recompute all eight digest/u64 pairs, rerun the synthetic
reference, and supersede every downstream source/hash/freeze binding.  This is
a hard pre-data HOLD rather than a statistical caveat.

### Official-target preclaim hash-open no-go

The moving release firewall recomputed `official_target_last` source bindings
by opening and hashing the target manifest and target bundle inside
`claim_stage`, before it durably wrote the claim.  This contradicts its own
`semantic_opens_before_claim=0` assertion and the H4 ordered firewall.  A
preclaim release may compare its target digest strings only to already-frozen
compile-time commitments; it may not re-open the target bytes to establish
them.  The actual manifest and bundle rehash must occur inside the isolated
official scorer after the unique claim exists and after every non-target
predecessor has authenticated.  This is a second hard pre-data HOLD.

## Checkpoint 131 — corrected-public-parent seam and residual-domain gate

The V34 source-zero baseline is the public corrected D11 prediction, not the
internal Q24 host returned by `predict_with_native_tape`.  This is forced by
three exact lineage facts:

1. the frozen response contract says `prediction_15 = deployed_15 + r_15`;
2. `PERFECT_STATE_CEILING.json` loads `deployed_rank*.npy`, and
   `run_deployed_only.py` created those bytes by calling
   `Estimator().predict`; and
3. in pinned `public_d11_util_halving_v1/estimator_cap.py`, public
   `Estimator.predict` returns `_predict_corrected`, namely

```text
host,native,sigma = _predict(..., return_native_tape=True)
j = transport_correction(native,sigma,weights,_TRANSPORT_PAYLOAD)
deployed = float32(max(host+j,0)).
```

Therefore a capture that stores the first return of
`predict_with_native_tape` as `parent_output` silently replaces the frozen
host+P2R0F2J baseline by raw Q24 and invalidates both source-zero parity and
the ceiling comparison.  The exact one-pass lawful capture is to hook the
single `_predict` call made inside `_predict_corrected`, retain that call's
native tape for the target-blind head, and invoke the public
`Estimator.predict(mlp,2^41)` as the parent entrypoint.  Its output must then
be byte-compared with a fresh uninstrumented public call.  Physical
`(m,v,k3,k4)` fields remain the raw reset seams; only the additive endpoint
baseline is corrected.  Matrix's moving capture adapter now follows this
spelling, but it is not frozen or independently receipt-bound yet.

For the binding independent-A/B oracle residual, define

```text
e_A = truth_A - deployed - z_A,
e_B = truth_B - deployed - z_B,
theta = <e_A,e_B>/1024.
```

The two-sample delete-one jackknife supplies both raw and bias-corrected
four-SE intervals.  Because this scalar is an MSE cross estimator, the same
`MC_RESOLUTION_INSUFFICIENT` rule used for response cross squared error must
require both intervals to remain in `[0,infinity)`.  Merely testing their
upper endpoints against `1e-8` would allow a negative or wholly
below-domain cross residual to pass as apparent capacity.  This is separate
from the nonlinear single-block oracle diagnostic, which remains nonbinding.

At the 12:02 PDT checkpoint the stable verdict remains HOLD.  Outstanding
static blockers are: the old seed receipt still binds revoked reference
hashes; the moving held scorer contains a duplicate `weights` formal and is
not syntactically valid; its oracle-cross interval-domain check is absent;
the corrected-parent capture has no stable synthetic parity receipt; and no
single exact implementation freeze yet binds the repaired seed, fused deploy,
release firewall, scorer, capture, and resource bytes.  No target, weight,
cache, MC payload, model, estimator, training, or network execution occurred
in this audit.

## Checkpoint 132 — fixed-panel versus population uncertainty

The frozen ordered rows `[757,746,780,759]` are contractual test cases, not a
declared iid network sample.  The stop law requires each row to pass every
gate individually, and only four passes request package review.  Conditional
on those four fixed rows and independent MC streams by row, if `x_i` estimates
a fixed row scalar with MC variance `v_i`, then

```text
xbar_fixed = (1/4) sum_i x_i,
SE_MC(xbar_fixed) = sqrt(sum_i v_i)/4.
```

Thus `independent_panel_aggregate` has correct arithmetic only when explicitly
labeled **fixed-panel conditional MC uncertainty**.  Between-row differences
are deterministic heterogeneity of the chosen test cases, not MC error in
that conditional estimand.  The exact official-target raw values have no MC
variance at all: the binding evidence is their all-four rowwise conjunction;
their arithmetic mean is descriptive and, given four strict individual
passes, redundant for the same threshold/sign gates.

The same formula is not a population or unseen-network standard error.  If a
future result treats the four rows as population replicates, omitting
between-row variation understates uncertainty.  Such a claim needs a
separately preregistered network-level or random-effects interval (at minimum
the observed network spread `s(x_i)/sqrt(4)`, with the weakness of only four
units disclosed).  V2R1 should therefore freeze fixed-panel semantics and
forbid population-generalization language in `panel_aggregate.py`; otherwise
the aggregate remains HOLD.  No data were opened for this ruling.

## Checkpoint 133 — exact frozen V2R1 implementation audit

The stable implementation set bound by manifest
`efb093d59000f76be614c9268754fc589c09e243ac891b1902c43a633f332a22`
and freeze
`5a0d937711794ad7d53e1449faf54f67b49d21c110a57b2533fece39ae82f6eb`
is **HOLD**.  Its algebraic core is nevertheless clean: `py_compile` and all
39 frozen synthetic fixtures pass under the exact Python 3.11.5 / NumPy
1.26.4 / Torch 2.7.0 runtime, with no data, target, weights, model, estimator,
cache, training, or MC access.

The binding independent receipt is
`public_v34_physical_state_response_sentinel_v2r1/INDEPENDENT_DIRECT_IMPLEMENTATION_AUDIT.json`,
SHA-256
`a3f2769081e6f431d2215ea642861184cd1414ac8dfdce9a7d2e604601a359b0`.
It records fourteen release-level blockers: the official resource receipt is
missing from the FSM keyset and returned transaction; held semantics invoke a
meter forbidden by their own permission; train/deploy head dtypes disagree;
audits are hashed but not semantically required to PASS; unknown permissions
are accepted; the fit claim admits a release-hash splice; revoked seed
authorities remain bound; the bound static receipt was not produced under the
pinned NumPy; the model freeze lacks a full modified-estimator meter; global
successor absence is incomplete; claims omit runtime versions; response
evidence is discarded; downstream provenance rehash is incomplete; and the
literal public-output/streaming resource path is not certified.  A synthetic
dtype witness measured maximum absolute train/deploy head disagreement
`4.4338989226844205e-05`.

The next admissible state is an append-only R1 set that repairs every blocker,
keeps the frozen bytes immutable, and receives fresh independent PASS
receipts.  This HOLD receipt itself cannot authorize R1 execution.

## Checkpoint 134 — Direct R1 PASS schema and adversarial verifier

The append-only Direct acceptance harness is frozen in four files:

```text
DIRECT_IMPLEMENTATION_PASS_AUDIT_SCHEMA_R1.json       394742aeeb0190c9...
direct_pass_audit_contract.py                         f7a17162ac7d8f67...
verify_direct_pass_audit_contract.py                  72721b6caa1980ef...
DIRECT_IMPLEMENTATION_PASS_CONTRACT_STATIC_VERIFICATION_R1.json
                                                       2ecba0765e856761...
```

The pure receipt validator accepts one canonical PASS document and rejects
all 75 isolated adverse mutations.  The numeric reference fixture enforces
the Lane5 public-output ruling
`3591a1e736c116bcb5b71d2711f9f4a0cd4fdbaad2ecfee497f13e49b5c1f424`:
the public output is `(16,1024)` float32, the zero path returns the direct
parent, rows 0--14 are parent-identical, and row 15 is
`float32(float64(parent[15]) + r15_float64)`.  It also binds the exact 4,004
byte mixed learned payload, the exact packaged float32 endpoint used by the
scientific gates, and a meter around the literal full-return callable.

A temporary zero-data response payload retains all 28 required state and
V/K3/K4 response families and replays their hashes and component compiler.
The future production adapter must repeat this at the literal full and 16
delete-one shapes.  The global causality fixture enumerates 17 nodes and 85
canonical artifact paths and rejects six representative early successor,
continuation, and aggregate artifacts.

Runtime authority is stage-specific.  Fit, held generation, held semantics,
and panel aggregation use Anaconda Python 3.11.5 / NumPy 1.26.4 / Torch
2.7.0.  Official-target-last uses the data-gate venv with Python 3.11.5 /
NumPy 2.4.6 / PyArrow 25.0.1 and does not import Torch.  The executable paths,
hashes, and versions are claim fields; cross-using the two environments is a
hard negative test.

The reference verifier is deliberately not an implementation PASS.  A future
stable R1 manifest must bind an adapter at
`production_audit_adapter.run_direct_r1_zero_data_suite`, make it drive the
actual candidate transaction functions, and reject the same 75 mutations.
No moving repair, repository data, target, weight, estimator, capture,
training, MC, scoring, release, claim, or network action occurred.

To keep independent evidence acyclic, the final Direct PASS file is fixed as
`INDEPENDENT_DIRECT_IMPLEMENTATION_AUDIT_R1.json` with schema
`v34-v2r1-independent-direct-implementation-pass-audit-r1-v1`.  It binds the
Cubic and H4 acceptance schemas/static receipts, not their future final PASS
files.  The fit firewall separately rehashes and parses all three final PASS
documents against the common R1 candidate bytes.

## Checkpoint 135 — explicit cross-runtime stage split and Direct-harness reconciliation

The prior Checkpoint 134 harness is now historical acceptance evidence, not a
complete production FSM. Read-only inspection of the two installed
environments proves the required split. The canonical numeric invocation is
`/Users/matt/anaconda3/bin/python3.11`, with Python 3.11.5, NumPy 1.26.4,
Torch 2.7.0, and neither FlopScope nor WhestBench installed. The data-gate
invocation is `data_gate/.venv/bin/python`, with Python 3.11.5, NumPy 2.4.6,
FlopScope 0.12.0, WhestBench 0.16.0, threadpoolctl 3.6.0, PyArrow 25.0.1,
and no Torch module. Both paths resolve to binary SHA-256
`6116d7bcb852d61f4384daa1c43488771e6794636d959f2a7065b54c92a070a1`,
so a binary hash alone cannot distinguish them: claims must also bind the
literal invocation, `sys.executable`, `sys.prefix`, environment/lock hashes,
and package/version manifest. `/Users/matt/anaconda3/bin/python` is resolved
alias metadata only, not the canonical numeric command.

The exact causal program has 28 nodes:

```text
data_gate FIT_CAPTURE(9)
 -> Anaconda FIT_TRAIN + nine uninstrumented reference outputs
 -> data_gate FIT_RESOURCE_METER(9)
 -> Anaconda MODEL_FREEZE_HASH_ONLY
 -> for each held row:
      data_gate HELD_CAPTURE_METER
      -> Anaconda HELD_GENERATION_A(16x500k)
      -> Anaconda HELD_GENERATION_B(16x500k)
      -> Anaconda HELD_SEMANTIC_SCORE
      -> data_gate OFFICIAL_TARGET_LAST
      -> continuation before the next row
 -> Anaconda PANEL_AGGREGATE.
```

Every fit-resource row requires three bitwise-identical complete outputs and
intermediate tapes: Anaconda uninstrumented, data-gate uninstrumented, and
data-gate FlopScope-instrumented. All are C-contiguous `(16,1024)` float32;
normalized float32 heads, hidden preactivations, hidden values, standardized
gaps, physical float64 gaps, responses, and public outputs must have maximum
absolute and ULP difference zero. The meter wraps the literal generated
full-return estimator, never an endpoint helper or diagnostic dictionary.
The hash-only freeze runs no inference. On held rows capture/meter precedes
both MC streams, semantic scoring consumes its sealed outputs without
reinvoking the estimator, and official scoring opens only the selected target
after semantic and response freezes authenticate.

The frozen addendum, verifier, and static receipt are respectively:

```text
DIRECT_STAGE_SPLIT_RUNTIME_ADDENDUM_R1.json
  8c70f1b7bcc26f419f0bcfbc42903b6e107db7c4132668cff7a8735d1bf55522
verify_direct_stage_split_runtime_addendum_r1.py
  9599ca9c99191ac735e9334ccbcc9d80ca1d860f5ce73e1424c969907d722ad4
DIRECT_STAGE_SPLIT_RUNTIME_ADDENDUM_STATIC_VERIFICATION_R1.json
  9f61a28707c6ab698629896a86d8d231e8ac7194f119bd55cb4d1948159fb5df
```

The verifier accepts the 28-node control, rejects all 30 isolated
runtime/order/import/output/hash mutations, and obtains exact synthetic
three-way parity under both installed NumPy stacks. This is reference-only:
the synthetic meter leg does not invoke FlopScope, and production must repeat
literal FlopScope on all nine sealed captures.

An append-only reconciliation
`DIRECT_PASS_CONTRACT_STAGE_SPLIT_RECONCILIATION_R1.json`, SHA-256
`81f642b304c6e7a30bd04ae82effb7f8e4724e2e0e8dea76e9f7dea0b6d8e5c8`,
preserves the four already bound Checkpoint 134 bytes but makes this addendum
binding wherever they conflict. It also corrects a historical synthetic
schema typo: the literal head is `24 -> 32 -> 3`, with 899 float32 parameters;
51 is the count of float64 normalizer values (`24+24+3`), not the input width.
The total learned numeric payload remains exactly `899*4 + 51*8 = 4,004`
bytes. Production cannot claim Direct PASS from the old validator alone; it
must bind the reconciliation and all three addendum hashes.

No repository data, weights, capture, MC, target, estimator, FlopScope meter,
training, scoring, or network action occurred. Every authority bit remains
false.

## Checkpoint 136 — moving 28-node firewall audit and complete result typing

Matrix's moving firewall at SHA-256
`7b33d14588823743bfe035e24e5e963bb95ae695e88b576df23b7606874d77a4`
correctly spells the 28-node order, stage runtimes, capture-only
`FIT_CAPTURE`, meter-only `FIT_RESOURCE_METER`, statistics-only held semantic
permissions, target-digest deferral, and the principal predecessor graph.
Its synthetic verifier reports 265 zero-data checks. These are moving bytes,
not an execution candidate, and the implementation remains **HOLD**.

The exact blockers found by read-only inspection are:

1. `orchestrator.py` is still the old five-stage monolith. It performs
   capture, training, metering, and freeze in `run_fit`, emits both MC streams
   in one held-generation call, and invokes estimator/capture/meter inside
   held semantics. Its stage names are absent from the new FSM.
2. `official_target_last.py` and `panel_aggregate.py` likewise call obsolete
   `stage_paths("fit")`, `stage_paths("held_generation")`, and
   `stage_paths("held_semantic")` identities. Updating the firewall alone
   cannot produce a runnable split transaction.
3. Result finalization calls `load_owned_claim`, whose postclaim absence check
   rejects the stage's already written seal. The required lawful order
   payload -> seal -> RESULT therefore deadlocks; RESULT-first instead allows
   a PASS before owned evidence exists.
4. The result writer accepts caller-supplied 64-hex strings and an arbitrary
   `access` mapping. It neither resolves/re-hashes canonical owned files nor
   validates their schemas, access counters, or final successor absence.
5. Global absence enumerates only release/claim/result/closure/failure plus
   one generic seal. It omits canonical capture arrays, models, normalizers,
   resource/parity receipts, MC blocks, response evidence, target-scoring
   payloads, manifests, and aggregate files. A stale future payload can thus
   preexist without colliding with a claim.
6. Official target authentication has no row or owned-claim argument and does
   not internally validate the target-last claim; direct invocation can hash
   target files preclaim.
7. Predecessor validation checks only a generic keyset and PASS flags. It does
   not verify stage/row/stream identity, schema, release/claim cross-links,
   canonical payload/seal hashes, access, or scientific gates. Continuation
   can therefore mistake a syntactic PASS for an official all-gates PASS.
8. Installed-distribution validation is not an import ledger. The official
   data-gate process could import FlopScope/WhestBench, the meter could import
   PyArrow, and an Anaconda statistics stage could import Torch/PyArrow
   without the claim/result detecting it.
9. Candidate Direct validation still dispatches only the historical Direct
   contract; it does not semantically enforce the split-runtime
   reconciliation or a future candidate-specific 28-node Direct PASS audit.
10. `_validate_complete_fit_provenance` reads `seal_sha256` from the model
    result, but the lawful model result owns `model_hash_freeze_sha256`; its
    current comparison is identically against `None`.
11. The adversarial verifier monkeypatches hashing, implementation freeze,
    and candidate audits and does not exercise claim/finalization, seal
    schemas, stage workers, or official claim guarding. It is useful shape
    evidence but not production evidence.

H4 has now frozen the complete result typing in
`H4_COMPLETE_STAGE_RESULT_KEYSET_SUPERSESSION_R1.json`, SHA-256
`fd6fe0021e1956bbfc738e7fe3b88f192d6588ea4694676dacf2811773bca30a`
(verifier `c03ef7588455205641058b8e8972da374ab7f88dfa3a65499e092698db3cc994`,
static receipt
`3983104291e653202b04bcb6773f75623d830efd86b7bbc56bdc927e65244948`).
It supplies ten exact result classes, 28 exact instance assignments, owned
fields, forbidden fields, and no null placeholders. The moving generic
`v34-v2r1-r1-stage-result-v1` writer cannot satisfy it.

The Direct runtime addendum did not supersede the generic release/claim
identity keysets. A parallel append-only override is still required:
fit/model/aggregate release and claim records omit both `row_index` and
`stream_tag`; held capture/semantic/official and continuation carry only the
exact row; A/B generation carries row plus exact stream. Null identity
placeholders are forbidden. H4 is preparing this final typing layer.

Current next question: after Matrix binds both complete stage schemas, does
the new worker suite enumerate and O_EXCL-write every stage-owned payload,
validate the owned seal before terminal RESULT, and independently replay all
28 predecessor/scientific-gate contracts without importing a forbidden
module? Until stable bytes answer all three, no PASS receipt can be issued.

No repository arrays, weights, labels, estimator, capture, meter, MC,
official target, training, scoring, or network operation occurred in this
checkpoint.

## Checkpoint 137 — minimal rank-35 descriptive value-first sentinel contract

Root has suspended the full 28-node production audit in favor of one minimal
value-first diagnostic.  The candidate is append-only under
`public_v34_rank35_descriptive_sentinel_v1/` and has only three external
runtime stages:

```text
data_gate CAPTURE: nine fit rows + exposed rank35, target-blind
Anaconda TRAIN: fixed fit on the nine state-label rows only
data_gate SCORE: exact public D11 + frozen head/response + exposed rank35 raw target
```

This is deliberately descriptive, not confirmatory: rank35's identity,
perfect-state response, LOO betas, and oracle value were exposed before this
runner. A PASS can answer whether the frozen spelling has value on that
known diagnostic, but cannot authorize packaging or a population/fresh-held
claim.

The acceptance contract is narrow and exact:

* Training rows are exactly observer-fit role ranks
  `[100,150,200,250,300,400,500,600,650]`, hence exactly
  `9*15*1024 = 138240` points. Rank35 may be captured target-blind, but its
  features, states, targets, hashes, distributional summaries, and scores are
  absent from the training dataset, input mean/std, label RMS, optimizer,
  checkpoint choice, and every model-selection branch. The TRAIN release
  should not grant a rank35-read permission at all.
* The head is literally `Linear(24,32)-ReLU-Linear(32,3)`, 899 float32
  parameters. Input population mean/std (`ddof=0`) and three label RMS values
  use only the nine fit rows; only exact zero scales are replaced by one.
  Training is fixed PyTorch CPU AdamW, seed `0x56334`, learning rate `0.001`,
  weight decay `0.0001`, batch 4096, 64 epochs, 34 steps per epoch, exactly
  2176 optimizer/scheduler steps, cosine `T_max=2176`, fixed final epoch, no
  validation, early stop, q/endpoint loss, network identity, or retry.
* Response coefficients are immutable constants
  `(0.8273280548817595, 0.32346157239952833,
  0.15489222117700333)` for V/K3/K4. They are never recomputed or selected by
  the runner. Response starts at zero-based layer 1, elides the first zero
  transport, then uses the row-vector recurrence
  `gate_l * (state @ W_l)` and component-pure EDG2 sources in float64.
* The parent is the public D11 source with SHA-256
  `ace56bf60b267666e4f2a6d8ae772a089f234ebc725f34a4798b2f9e8284cbc2`.
  Source-zero returns the parent object/bytes directly. Source-on returns one
  C-contiguous `(16,1024)` float32 array: rows 0--14 are byte-identical to the
  parent, and row 15 is exactly
  `float32(float64(parent[15]) + r15_float64)`, with no clipping or float32
  pre-add. The exact packaged float32 endpoint is what SCORE compares to the
  raw target.
* CAPTURE and SCORE use the data-gate Python/NumPy/FlopScope/WhestBench stack
  with no Torch import; TRAIN uses Anaconda Python/NumPy/Torch with no
  FlopScope/WhestBench/target access. Model and normalizer bytes are frozen
  before SCORE can claim or open the rank35 target. A score-time D11 rerun
  must replay the target-blind rank35 capture/parent bytes exactly.
* Every stage has an exclusive release and durable `O_EXCL` claim. Any
  postclaim exception is terminal; no stage can resume or retry. SCORE target
  access begins only after its claim and only after rehashing the frozen
  model/normalizer/source/capture chain. Neither PASS nor FAIL grants another
  head, beta, layer window, target, or run.

Immediate static falsifiers are: perturb rank35 capture/state/target and show
the fit dataset and normalizers do not change; mutate any architecture,
training, beta, orientation, cast, parent row, runtime, or claim field and
require rejection; verify source-zero object identity and source-on rowwise
bytes on a tiny synthetic fixture. The next question is whether Matrix's
landed runner enforces these properties at the code path, rather than only
recording them in a manifest.

No production data, target, weights, estimator, capture, meter, training,
score, cache, or network operation occurred while deriving this contract.

### Checkpoint 137 chronology correction

The initial three-stage sketch above is superseded by the stricter binding
H4 checklist
`H4_RANK35_DESCRIPTIVE_SENTINEL_MINIMAL_RELEASE_CHECKLIST_R1.json`, SHA-256
`d7b0524c383fbf978cd605a92b15dfcf61dcfad28ab299b75ab30092a82dea58`.
The exact transaction order is four stages:

```text
FIT_CAPTURE_9
 -> FIT_TRAIN_MODEL_FREEZE
 -> RANK35_CAPTURE_OUTPUT_FREEZE
 -> RANK35_TARGET_SCORE.
```

`FIT_CAPTURE_9` may not touch or materialize the rank35 manifest, weights,
capture, state labels, or target. The immutable model/normalizer freeze is
therefore causally prior to **all** rank35 access, not merely prior to rank35
target access. Only the third stage opens rank35 manifest/weights and creates
the complete target-blind source-zero/source-on output freeze. A fourth,
separate durable one-shot score claim is then required before opening the
already exposed rank35 row-15 target. This stronger chronology prevents even
unsupervised/transductive rank35 feature influence on normalization or model
selection and is the only chronology eligible for Direct PASS.

## Checkpoint 138 — stable minimal rank-35 implementation PASS

The append-only candidate
`public_v34_rank35_descriptive_sentinel_v1/` is now exact-byte frozen at:

* implementation manifest
  `4efc59b30c30ddfa46dbac2a21dfc6857247bfa40f358721d60d3ed2d93f5ce3`;
* implementation freeze
  `d77a1348edbe1d67025489a7221802c3ddb56c03a5d138b232196be539254037`;
* static verification
  `e10f69045fcdc5893fcdd4fd0d51e4993b54090276268c60cabfd6353e9674e7`;
* H4 scope amendment
  `5674c5f21ddd523c4def4d94ef86de6807538081bea006f49e04bbeaae542f62`.

The Direct exact-byte audit is **PASS**. Both the Anaconda and data-gate
runtimes replayed all seven frozen static/synthetic tests. An independent
nonzero synthetic fixture also compared the deployed fused recurrence to a
separately evaluated component-pure provider recurrence with distinct
per-layer transport actions; it matched through the final float64 response
and exact float32 endpoint cast. Adversarial score fixtures confirmed the
strict `<1e-8` boundary, the independent `source_on < source_zero` conjunct,
consistent PASS/FAIL booleans, and rejection of NaN or extra/ill-typed result
evidence.

The final source closure enforces:

1. exactly nine target-blind fit captures before any rank35 access;
2. one fixed 138240-node, 899-parameter state-only fit and model freeze;
3. first rank35 manifest/weight access only in the post-model target-blind
   output stage;
4. a complete metered C-contiguous float32 `(16,1024)` return whose rows
   0--14 and source-zero path are checked by bytes, with only row15 replaced;
5. first target access only after the immutable output freeze and a separate
   same-PID durable score claim;
6. exact typed results and strict descriptive-only raw-MSE gate; and
7. sequential root authorization, release, claim, predecessor rehash, and
   terminal no-retry semantics.

The independent receipt is
`INDEPENDENT_DIRECT_IMPLEMENTATION_AUDIT.json`, SHA-256
`5ebe12067868940a115e5c0341e5a26c262946a972c1426cf4e360e7d7af77d8`.
It grants no release, execution, target, retry, package, board, submission, or
network authority. Rank35 remains previously exposed; a future result is
descriptive development evidence only. The H4 scope amendment permits the
full exposed `(16,1024)` target array to be materialized after the score
claim while requiring that only row15/1024 values enter either MSE; it also
defers the 256-KiB streaming proof to packaging and grants no package
authority.

No production target, label, weight array, estimator, capture, meter,
training, scoring, MC, or network operation occurred in this audit.

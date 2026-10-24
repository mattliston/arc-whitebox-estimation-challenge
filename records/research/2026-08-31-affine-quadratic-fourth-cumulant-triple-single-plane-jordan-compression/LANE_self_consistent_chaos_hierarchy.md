# Self-consistent chaos hierarchy

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## 2026-08-31: finite-chaos closure and exact algebra

Use Wiener integrals normalized so that

\[
I_1(a)I_2(A)=I_3(a\widetilde\otimes A)+2I_1(Aa),
\]

\[
I_2(A)I_2(B)=I_4(A\widetilde\otimes B)+4I_2(A\circ B)+2\operatorname{tr}(AB),
\]

\[
I_1(a)I_3(B)=I_4(a\widetilde\otimes B)+3I_2(B\mathbin\contr a).
\]

Thus, for
\(X_i=I_1(u_i)+\epsilon I_2(A_i)+\epsilon^2I_3(B_i)+O(\epsilon^3)\),

\[
\delta v_{ij}=2\epsilon^2\operatorname{tr}(A_iA_j),
\qquad
K_{21,ij}=2\epsilon\{u_i^TA_ju_i+2u_i^TA_iu_j\}.
\]

The order-four output is not an independent list: it contains both irreducible
\(H_3\) information and Jordan squares of the same \(H_2\) generators. A literal
\(H_{\le4}\) random-variable space is not invariant under ReLU; a lawful finite
closure must use exact Gaussian response (equivalently Price/Edgeworth kernels)
at every reset.

The exact-state H2/H3/H4 benchmark raw error is \(9.2547\times10^{-9}\), leaving
only \(0.07453\times10^{-8}\) slack to the goal. With channel values, in units of
\(10^{-8}\),

\[
\Delta_v=1.7316,\qquad \Delta_3=0.4370,\qquad \Delta_4=0.10313,
\]

any candidate with captured-energy fractions \(a_v,a_3,a_4\) has the sufficient
gate

\[
1.7316(1-a_v)+0.4370(1-a_3)+0.10313(1-a_4)<0.07453. \tag{C1}
\]

If all but the variance channel are exact, \(a_v\ge0.95696\), hence the
optimally scaled predictor must have correlation at least \(0.97824\). Here
\(a\) is energy captured; an unqualified \(q=0.78\) must be disambiguated:
if it is correlation, \(a=q^2=0.6084\); if it is already energy, \(a=0.78\).

## Ordered Lie--Jordan mechanism over the ten birth layers

For the ten late births, ordered composition has the exact second-order form

\[
U_{10}\cdots U_1
=I+\epsilon\sum_tA_t+
\epsilon^2\left(\sum_tB_t+\frac12\sum_tA_t^2+
\sum_{s<t}A_tA_s\right)+O(\epsilon^3).
\]

There are exactly \(\binom{10}{2}=45\) causal cross-layer products. Their
unified split is

\[
A_tA_s=\tfrac12\{A_t,A_s\}+\tfrac12[A_t,A_s],
\]

so the Jordan part drives the variance/H4 square channel and the Lie part drives
the odd/H3 sibling. With a common associative algebra the prefix recurrence is

\[
Q^+=\bar Q+A,
\]

\[
J^+=\bar J+B+\tfrac12(A\bar Q-\bar QA),
\]

\[
S^+=\bar S+\tfrac12(A\bar Q+\bar QA)+\tfrac12A^2.
\]

Nine ordered prefix products collect all 45 pairs; explicitly materializing both
symmetric and antisymmetric parts uses eighteen products. The ten local squares
are the existing Q24 terms. Novelty is conditional: after CP3 plus local Q24 all
45 ordered cross terms are new; if Q24 already squares the accumulated prefix,
the symmetric cross terms are old and only commutators are new; if the deployed
lineage already transports ordered products, neither is new. Compact exactness
requires an algebra homomorphism; a generic tensor projection is not closed.

At third order, the exact power-prefix recurrence is

\[
R_2=B+\tfrac12A^2,\qquad
R_3=C+\tfrac12\{A,B\}+\tfrac16A^3,
\]

\[
U_3^+=\bar U_3+R_3+A\bar U_2+R_2\bar U_1.
\]

The corresponding BCH Lie coordinate is

\[
K^+=\bar K+C+\tfrac12([A,\bar J]+[B,\bar Q])
+\tfrac1{12}([A,[A,\bar Q]]+[\bar Q,[\bar Q,A]]).
\]

This prefix representation is exact and avoids enumerating the 120 ordered
triples.

## Co-adapted producer and consumer tangent

Let \(\theta=(m,v,C)\), let \(T\) be full cubic provenance, let \(k=R(T)\), and
write

\[
T^+=A(\theta)^{\otimes3}T+B^{\otimes3}S_2(\theta).
\]

Along \(\theta=\theta_0+t\theta_1+t^2\theta_2+\cdots\) and
\(T=tT_1+t^2T_2+t^3T_3+\cdots\),

\[
T_1^+=A_0^{\otimes3}T_1+B^{\otimes3}S_2(\theta_0),
\]

\[
T_2^+=A_0^{\otimes3}T_2+\mathcal A_1T_1
+B^{\otimes3}DS_2[\theta_1],
\]

\[
T_3^+=A_0^{\otimes3}T_3+\mathcal A_1T_2+\mathcal A_2T_1
+B^{\otimes3}\{DS_2[\theta_2]+\tfrac12D^2S_2[\theta_1,\theta_1]\}.
\]

Here \(\mathcal A_1\) is the sum of the three one-leg insertions of
\(\delta A_1\), and \(\mathcal A_2\) contains one-leg \(\delta A_2\) plus all
two-leg \(\delta A_1\delta A_1\) insertions. The consumer through order three is

\[
\epsilon_3^+=B\big[G_{11,0}\odot\epsilon_3
+G_{11,1}\odot\epsilon_2+G_{11,2}\odot\epsilon_1
+I(G_0,k_3)+I(G_1,k_2)+I(G_2,k_1)+U_3\big]B^T.
\]

The deployed-mask terms are only the partial derivative of the consumer at
fixed \(k\). The \(\delta A\)-on-\(T\) and \(DS_2\) terms are the missing chain
rule factor \(C_k k_\theta\). They require the producer tangent whenever \(k\)
is self-generated; they vanish only for an exogenous/truth-fixed \(k\). A closed
projected recurrence for \(k\) exists only if

\[
\ker R\subseteq\ker(RA^{\otimes3}),
\]

which is generically false.

The Taylor source-fidelity ladder, with normalized residual allowance
\(\tau=0.18113\), is

\[
2d+d^2\le\tau\Rightarrow d\le0.0868
\]

for a frozen plug-in,

\[
d^2\le\tau\Rightarrow d\le0.4256
\]

after the first tangent, and

\[
C_3d^3\le\tau\Rightarrow d\le(0.18113/C_3)^{1/3}
\]

after the exact order-two tangent. These are Taylor remainder gates, not the
final endpoint energy gate (C1).

## Signed feedback and counterterm discipline

For the coupled block

\[
\begin{pmatrix}A&B\\C&D\end{pmatrix},
\qquad
J_{\mathrm{eff}}(z)=A+B(zI-D)^{-1}C.
\]

In one mode the feedback factor is

\[
C_{\mathrm{fb}}=\frac1{1-a-bc/(1-\delta)}.
\]

Monotone \(bc\ge0\) amplifies. Suppression requires \(bc=-k<0\), and the stable
window needed for a factor below 0.8466 is

\[
(1-\delta)(a+0.1812)<k<1-a\delta.
\]

The lawful 2PI counterterm is

\[
\Delta\Sigma=\tfrac12\{V(G\otimes G)V^T-T_D(G\otimes G)T_D^T\},
\]

never \(\tfrac12(V-T_D)(G\otimes G)(V-T_D)^T\). The latter invents cross
evidence. The exact quartic 211 identity

\[
q_4(A;B,C)=48\operatorname{tr}(A^2BC)
+8\operatorname{tr}([A,B][A,C])
\]

contains the negative Lie correction
\(-8\|[A,B]\|_F^2\) on \(B=C\). An explicit coupled state and its eliminated
Dyson self-energy are alternative coordinates, never additive score.

## External q4 Schur provider

Let \(V=\operatorname{span}\{A_i\}\), let \(P\) be its Frobenius projector, and
define

\[
Y_{ij}=A_i\circ A_j,\qquad Z_{ij}=(I-P)Y_{ij},
\qquad G^\perp_{ij,kl}=\langle Z_{ij},Z_{kl}\rangle_F.
\]

Then

\[
Q^\perp_{ijkl}=16\{G^\perp_{ij,kl}+G^\perp_{ik,jl}+G^\perp_{il,jk}\},
\]

and in the 211 collision

\[
q^\perp_{iijk}=16\{\langle Z_{ii},Z_{jk}\rangle
+2\langle Z_{ij},Z_{ik}\rangle\}.
\]

In particular \(Q^\perp(x,x,x,x)=48\|Z(x,x)\|_F^2\ge0\), and this provider is
nonzero exactly when \(V\) is not Jordan closed. Under an invertible dense
transport \(W\),

\[
Z'_{ab}=\sum_{ij}W_{ia}W_{jb}Z_{ij},
\]

so the full object transports as \(\operatorname{Sym}^4\). The parallel part is
already represented by Q24/old-span terms; only \(Z=(I-P)Y\) is new. This is an
external collision term, not the latent \(q_{4,z211}\) source.

The commutator Gram is not an independent provider. After the same old-span
projection,

\[
\langle[A_a,A_b],[A_c,A_d]\rangle_\perp
=4\{G^\perp_{ac,bd}-G^\perp_{ad,bc}\}. \tag{C2}
\]

Writing the reshuffle in (C2) as \(\mathcal L G^\perp\), every NC/commutator
readout and every direct q4 readout are linear functionals of one Schur Gram.
Their evidence must therefore be joint-residualized and counted once.

## Checkpoint 4 correction: predictable external-q4 Bellman identity

The earlier three-forward-state recurrence
\((\theta,N,R)\) was too strong as a deployability claim. Merely transporting a
supported \(R\) gives no value unless the future endpoint is observable from it;
generic rank-64 exterior dynamics cannot be closed by one to three word
contexts. For a fixed scalar endpoint, however, the exact object is a backward
Bellman contraction, not a universal forward closure.

Let a base trajectory obey \(x_{l+1}=F_l(x_l)\). Let an external Schur provider
enter as

\[
x_{l+1}=F_l(x_l)+\mathcal B_l(x_l)G_l,
\qquad
G_l=\epsilon^2G_{l,2}+\epsilon^3G_{l,3}+O(\epsilon^4),
\]

and let the scalar endpoint be \(y=\psi(x_L)\). Define

\[
A_{l,0}=DF_l[x_{l,0}],\qquad
A_{l,1}=D^2F_l[x_{l,0}][x_{l,1},\,\cdot],
\]

\[
B_{l,0}=\mathcal B_l(x_{l,0}),\qquad
B_{l,1}=D\mathcal B_l[x_{l,0}][x_{l,1}],
\]

and backward costates

\[
\lambda_{L,0}=D\psi[x_{L,0}],
\qquad
\lambda_{l,0}=A_{l,0}^*\lambda_{l+1,0},
\]

\[
\lambda_{L,1}=D^2\psi[x_{L,0}][x_{L,1},\,\cdot],
\qquad
\lambda_{l,1}=A_{l,0}^*\lambda_{l+1,1}
+A_{l,1}^*\lambda_{l+1,0}.
\]

Discrete adjoint summation gives the exact provider contribution through order
three:

\[
y_2^\perp=\sum_l\langle B_{l,0}^*\lambda_{l+1,0},G_{l,2}\rangle, \tag{C3}
\]

\[
y_3^\perp=\sum_l\left[
\langle B_{l,0}^*\lambda_{l+1,0},G_{l,3}\rangle
+\langle B_{l,1}^*\lambda_{l+1,0}
+B_{l,0}^*\lambda_{l+1,1},G_{l,2}\rangle\right]. \tag{C4}
\]

For a nonsingular correlated Gaussian and ReLU interpreted distributionally,
the fourth-cumulant response is

\[
\mathcal B_lG=\frac1{4!}\mathcal H_l(48\operatorname{Sym}_4G)
=2\mathcal H_lG,
\]

because the Price tensor \(\mathcal H_l=\mathbb E[D^4f_l]\) is fully
symmetric. With

\[
\Gamma_{l,0}=\operatorname{PairSym}(\mathcal H_{l,0}^*\lambda_{l+1,0}),
\]

\[
\Gamma_{l,1}=\operatorname{PairSym}\{\mathcal H_{l,0}^*\lambda_{l+1,1}
+(D\mathcal H_l[x_{l,1}])^*\lambda_{l+1,0}\},
\]

(C3)--(C4) become

\[
y_2^\perp=2\sum_l\langle\Gamma_{l,0},G_{l,2}^\perp\rangle, \tag{C5}
\]

\[
y_3^\perp=2\sum_l\{\langle\Gamma_{l,0},G_{l,3}^\perp\rangle
+\langle\Gamma_{l,1},G_{l,2}^\perp\rangle\}. \tag{C6}
\]

Equations (C5)--(C6) are the predictable external-q4 Bellman identity. They
prove that provider support alone implies no gain: if the Bellman kernels vanish
on the attainable Schur-Gram span, the contribution is exactly zero through
order three no matter how large \(\|G^\perp\|\) is.

If the NC sibling has Bellman kernel \(\Xi_l\), (C2) says the one lawful joint
kernel is

\[
\Omega_l=2\Gamma_l+\mathcal L^*\Xi_l, \tag{C7}
\]

and the local value is \(\langle\Omega_l,G_l^\perp\rangle\). Direct q4 and
commutator-square gains must not be added after (C7). A genuine odd H3 amplitude
may still need its own state; only its quadratic/q4 sibling is identified with
the Schur provider.

For the allowed ten-transition suffix, replace the old forward \(R\) state by
the scalar accumulator

\[
z_{l+1}=z_l+
\epsilon^2\langle\Omega_{l,0},G_{l,2}^\perp\rangle
+\epsilon^3\{\langle\Omega_{l,0},G_{l,3}^\perp\rangle
+\langle\Omega_{l,1},G_{l,2}^\perp\rangle\}, \tag{C8}
\]

with the sum restricted to the last ten transitions. This is mathematically
equivalent to full forward transport for the fixed scalar endpoint and has no
correlated-ReLU quotient leakage if the costates and local contractions are
exact. It does not give a universal closure for other endpoints.

## Stress tests and exact remaining gates

1. **Universal exterior closure is impossible at one to three contexts.** For
   rank 64, \(\dim\mathfrak{so}_{64}=2016\). If \(\mathcal O\) is the span of
   all backward-transported endpoint readouts, any exact \(k\)-word quotient
   must satisfy

   \[
   k\ge\dim\mathcal O. \tag{C9}
   \]

   A generic dense correlated transport has full or rapidly growing
   \(\mathcal O\); one to three words cannot be a universal state. Bellman
   compression evades (C9) only for the fixed endpoint and only if its local
   kernel is realizable by the permitted word class.

2. **Word-realizability gate.** Let \(\mathcal W_l\) be the pair-kernel family
   computable by the allowed suffix word(s), and let \(\mathcal G_l\) be the
   linear span of attainable Schur providers. Exactness requires

   \[
   \Omega_l\in\mathcal W_l+\mathcal G_l^\perp. \tag{C10}
   \]

   If one word exposes only a rank-one pair kernel, (C10) requires the attainable
   part of \(\Omega_l\) to have rank at most one. Generic dense Bellman kernels
   do not. The approximate gate is the joint, cross-layer provider-covariance
   projection error, not Frobenius rank in isolation.

3. **Suffix-completeness gate.** Let \(S_{\rm pre}\) and \(S_{\rm suf}\) be the
   Bellman contractions before and inside the last ten transitions. The
   irreducible omitted energy is

   \[
   h_{\rm suffix}=\frac{\operatorname{Var}
   (S_{\rm pre}-\mathbb E[S_{\rm pre}\mid\text{old},S_{\rm suf}])}
   {\operatorname{Var}(\text{channel oracle})}. \tag{C11}
   \]

   It must fit (C1). Raw support in the suffix is not a substitute for (C11).

4. **Approximate quotient/costate leakage gate.** For a quotient defect
   \(E_s=\pi_{s+1}T_s-\bar T_s\pi_s\), the exact telescoping identity is

   \[
   \pi_L\Phi_{L:l}-\bar\Phi_{L:l}\pi_l
   =\sum_{s=l}^{L-1}\bar\Phi_{L:s+1}E_s\Phi_{s:l}. \tag{C12}
   \]

   Hence the weighted sum of per-layer defects, not their unweighted mean, must
   be below the square root of the allocated endpoint-energy deficit. With ten
   comparably weighted layers and only one percent energy allocated, (C12)
   indeed forces roughly sub-one-percent amplitude defect per layer. Exact
   Bellman costates make this term zero; compressed costates need a certified
   bound.

5. **Hidden and tail gate.** If \(h_c,n_c,p_c,t_c\) are respectively hidden
   conditional/all-distinct, NC/word-representation, suffix/q4, and order-four-
   plus tail residual fractions, then a safe nonorthogonal bound is

   \[
   d_c\le(\sqrt{h_c}+\sqrt{n_c}+\sqrt{p_c}+\sqrt{t_c})^2. \tag{C13}
   \]

   Substitution into (C1) is a sufficient raw-below-\(10^{-8}\) certificate.
   Hidden energy alone must obey

   \[
   1.7316h_v+0.4370h_3+0.10313h_4<0.07453. \tag{C14}
   \]

   No resummation can beat (C14), because it cannot invent conditionally hidden
   all-distinct information.

The ReLU even tail after H4 is exactly

\[
E_{>4}(r)=\pi^{-1}\{\sqrt{1-r^2}+r\arcsin r-1-r^2/2-r^4/24\},
\]

with

\[
\frac{r^6}{80\pi}\le E_{>4}(r)
\le\frac{r^6}{80\pi}\left[1+\frac{25r^2}{56(1-r^2)}\right].
\]

This local bound still needs to be composed with the exact Bellman/adjoint norms
to certify \(t_c\) near critical correlations.

## Double-count boundaries (binding)

1. Explicit NC state versus eliminated Dyson self-energy: choose one coordinate.
2. Fish and its q4 sibling are the same \(AA^T/A^TA\) block.
3. Use the difference of Grams in the 2PI counterterm, never the Gram of a
   residual vertex.
4. By (C2), the NC commutator-square and external q4 Schur readout share one
   provider and one joint Bellman kernel.
5. The parallel Jordan square is Q24/old span; only \((I-P)Y\) is new.
6. External \(q_{4,\mathrm{ext211}}\) is not latent \(q_{4,z211}\).
7. Residualize against CP3/star, Q24, the existing tangent, commuting q4, and the
   NC shared block before attributing new value.
8. Two word amplitudes produce one \(\operatorname{Sym}^2\) family
   \(\{c_1^2,c_1c_2,c_2^2\}\); count its cross term once.
9. Hidden all-distinct information is neither q4-perp nor a Padé tail.
10. The moving projector \(DP\) is mandatory in \(G_{l,3}\).
11. Forward compactness requires quotient invariance; endpoint Bellman
    compression is not a universal replacement.
12. The second q4 mechanism is restricted to the last ten transitions, so its
    proof must include (C11).

## Minimal missing lemmas for a full theorem

The Bellman relevance identity (C3)--(C8) is now algebraically closed. A complete
raw-below-\(10^{-8}\) proof still needs exactly these independent ingredients:

1. **Deployable Bellman-kernel realizability:** prove (or refute) that the joint
   kernel \(\Omega=2\Gamma+\mathcal L^*\Xi\) lies in the one-extra-word suffix
   family with representation residual small enough for (C1).
2. **Conditional completeness of the ten-layer source:** certify (C11) together
   with the all-distinct hidden bound (C14). This is an information theorem, not
   a fit-quality statement.
3. **Correlated-ReLU stability:** bound the exact adjoint norms and any compressed
   quotient defects in (C12), including the near-critical correlation margin.
4. **Higher-chaos remainder:** compose the displayed exact ReLU tail with those
   adjoint norms through all late transitions to obtain \(t_c\).
5. **Energy conversion:** prove that the residualized Bellman statistic is the
   conditional oracle component, so optimally scaled/refit capture fractions
   enter (C1), with no empirical double count.

The highest-leverage next question is lemma 1: determine the attainable-pair
rank and hardware-word factorization of \(\Omega_l\) for a scalar raw-MSE
endpoint. If it is rank one modulo the old span, (C8) is the desired compact
mechanism; if its covariance-weighted residual violates (C1), the proposed
one-word q4 suffix is mathematically incapable of reaching the goal regardless
of support size.

## Checkpoint 5 pivot: direct tensor-free K31 Bellman action

The word-realizability question above can be bypassed if the Bellman contraction
itself is evaluated, rather than first constructing or transporting
\(Q^\perp\). For centered scalar contexts \(U_a,V_a\), the identity

\[
\boxed{\quad
\sum_a\kappa_4(U_a,U_a,U_a,V_a)
=\sum_a\left\{\mathbb E[U_a^3V_a]
-3\mathbb E[U_a^2]\mathbb E[U_aV_a]\right\}
\quad} \tag{C15}
\]

is universal. The context decomposition is the 3--1 decomposition of the
endpoint Price/Bellman tensor. Thus (C15) is a scalar action of K31, not an
approximation to a stored fourth-order tensor.

For \(g\sim N(0,I)\), take centered affine-quadratic contexts

\[
U=h^Tg+g^TNg-\operatorname{tr}N,
\qquad
V=r^Tg+g^TRg-\operatorname{tr}R,
\]

with symmetric \(N,R\). Their joint cumulant generating function is

\[
K(t,s)=-\tfrac12\log\det(I-2(tN+sR))
-\operatorname{tr}(tN+sR)
+\tfrac12(th+sr)^T(I-2(tN+sR))^{-1}(th+sr).
\]

The \(t^3s\) coefficient is

\[
8\operatorname{tr}(N^3R)
+4\{h^TNRh+h^TN^2r\}.
\]

Applying \(\partial_t^3\partial_s\), which multiplies this coefficient by six,
proves the exact tensor-free formula

\[
\boxed{\quad
\kappa_4(U,U,U,V)=48\operatorname{tr}(N^3R)
+24\{h^TNRh+h^TN^2r\}.
\quad} \tag{C16}
\]

The trace term is the all-quadratic ordered/Jordan--Lie contribution. The two
mixed terms are the H1/H2 siblings. They are one self-consistent cumulant action
whose downstream channel (variance, k3, or k4) is selected by the Bellman dual
\(V\); they are not three independently scored features.

### Minimal recurrence

Let \(q_l=\{(h_{la},N_{la})\}_a\) be the primal H1/H2 coefficients under the
exact Gaussian projection/reset map \(\mathcal P_l\), and let
\(p_l=\{(r_{la},R_{la})\}_a\) be its Bellman dual. The endpoint-specific
self-consistent recurrence is

\[
q_{l+1}=\mathcal P_l(q_l;\theta_l), \tag{C17}
\]

\[
p_l=D_q\mathcal P_l(q_l;\theta_l)^*p_{l+1}, \tag{C18}
\]

\[
z_{l+1}=z_l+\mathbf 1_{\{l\in\text{last 10}\}}
\left[\chi_l(q_l,p_{l+1})-\chi_l^{\rm old}\right], \tag{C19}
\]

where

\[
\chi_l=48\sum_a\operatorname{tr}(N_{la}^3R_{la})
+24\sum_a\{h_{la}^TN_{la}R_{la}h_{la}
+h_{la}^TN_{la}^2r_{la}\}. \tag{C20}
\]

Dense linear transport acts linearly on \((h,N)\); the reset in (C17) is the
exact H1/H2 Hermite/Price projection. Equation (C18) folds every downstream
correlated-ReLU transport into \(V\). Consequently (C19) needs no fourth tensor,
no exterior forward quotient, and no q4 word. It is the scalar-adjoint version
of (C3)--(C8). A backward pass is mathematically essential unless the same
costate has already been compiled into a forward word.

The old term in (C19) is

\[
\chi_l^{\rm old}=\langle\Lambda_l,K_{31,l}^{\rm CP3/Q24/DTOT2}\rangle.
\]

Subtraction must occur at the cumulant tensor/action level. The
\(-3\mathbb E[U^2]\mathbb E[UV]\) in (C15) is the defining disconnected
counterterm and must not be subtracted a second time. Once (C15) is used, the
q4-perp, commutator-square, fish, and K31 action are alternative decompositions
of this same contracted evidence; none may be added independently. Genuine H3
information outside this fourth-cumulant action remains separate.

### Exact acquisition and beyond-quadratic gates

For an exact context implementation define

\[
m_{31,a}=\mathbb E[U_a^3V_a],\qquad
m_{20,a}=\mathbb E[U_a^2],\qquad
m_{11,a}=\mathbb E[U_aV_a].
\]

If their acquired values have errors
\(\delta_{31,a},\delta_{20,a},\delta_{11,a}\), then

\[
|\widehat\chi-\chi|
\le\sum_a\left[|\delta_{31,a}|
+3\{|m_{11,a}||\delta_{20,a}|
+|m_{20,a}||\delta_{11,a}|
+|\delta_{20,a}\delta_{11,a}|\}\right]. \tag{C21}
\]

This converts moment-acquisition accuracy directly into endpoint-action error.

If \(U=A+e\) and \(V=B+f\), where \(A,B\) are the affine-quadratic projections
and all variables are centered, multilinearity of the fourth cumulant and
Hölder give, with
\(M_U=\max(\|U\|_4,\|A\|_4)\) and
\(M_V=\max(\|V\|_4,\|B\|_4)\),

\[
|\kappa_4(U,U,U,V)-\kappa_4(A,A,A,B)|
\le12\|e\|_4M_U^2M_V+4\|f\|_4M_U^3. \tag{C22}
\]

Indeed \(|\kappa_4(X_1,X_2,X_3,X_4)|\le4\prod_i\|X_i\|_4\), and a four-slot
telescoping replacement proves (C22). This is the rigorous beyond-quadratic
residual gate. Layer errors add in endpoint amplitude by Minkowski; their
squared, channel-normalized total enters (C13) and then (C1).

### Revised decisive lemma

The exterior rank and correlated-ReLU quotient obstructions no longer apply if
(C15) can be evaluated with the exact Bellman contexts. They reappear only as a
context-decomposition or moment-acquisition obstruction. The decisive missing
lemma is now:

> For the deployed scalar raw-MSE endpoint, the available primal/dual state must
> acquire the three moments in (C15), over the complete Bellman 3--1 context
> decomposition, with the combined error from (C21), the affine-quadratic tail
> (C22), the omitted pre-suffix action, and hidden all-distinct information small
> enough to satisfy (C1).

Failure of that lemma is a genuine information failure. Success would provide a
tensor-free, co-adapted, endpoint-specific mechanism joining variance, k3, and
k4 through one cumulant action.

Current next question: prove whether the existing H1/H2 forward state and its
compiled Bellman dual determine \(m_{31},m_{20},m_{11}\) exactly. If not, derive
the conditional variance of the missing moment action and compare it directly
with the \(0.07453\times10^{-8}\) total deficit budget; no support or rank proxy
can replace that gate.

### Exact Jordan acquisition form

For the affine-quadratic model, the scalar moments require no stochastic
acquisition once the full primal and dual H1/H2 coefficients are available:

\[
m_{20}=\|h\|^2+2\operatorname{tr}(N^2),
\qquad
m_{11}=h^Tr+2\operatorname{tr}(NR),
\]

\[
m_{31}=3m_{20}m_{11}+\chi.
\]

Moreover, symmetry of \(N,R\) rewrites (C16) as

\[
\boxed{\quad
\chi=48\langle N^2,N\circ R\rangle_F
+24\langle N\circ R,hh^T\rangle_F
+24\langle N^2,\operatorname{sym}(hr^T)\rangle_F.
\quad} \tag{C23}
\]

Thus the exact affine-quadratic action needs only the two Jordan objects
\(N^2\) and \(N\circ R\), plus rank-one H1 pairings. Q24 can supply the first
only if its square is the same causal/primal \(N^2\); the genuinely dual object
is the mixed Bellman Jordan product \(N\circ R\). This gives an exact
acquisition criterion:

> A compressed deployed state is sufficient for affine-quadratic K31 if and
> only if it preserves, modulo the old feature span, the three Frobenius
> pairings in (C23) for every endpoint context in the 3--1 decomposition.

If two admissible coefficient quadruples \((h,N,r,R)\) agree on every carried
summary but disagree on any of the pairings in (C23), no refit on that state can
recover the missing K31 action. This is the inverse-oracle/impossibility gate.
The remaining approximation is then exclusively the beyond-H2 context tail in
(C22), not an omitted q4 tensor.

For each context, if the primal square \(N^2\) is already the lawful Q24 state,
(C23) requires one new dense product \(NR\) per active transition; its transpose
supplies \(RN\), and every remaining contraction is \(O(r^2)\). Thus \(m\)
Bellman contexts over the ten-layer suffix cost exactly \(10m\) new matrix
products in this algebraic model, with no \(O(r^4)\) state. If \(N^2\) is not
already available, the count doubles. This count is valid only after verifying
that the Q24 square and the causal \(N\) in (C23) are identical objects.

The exact inverse-oracle gate is conditional variance. For the stacked new
action \(X=\sum_l(\chi_l-\chi_l^{\rm old})\) and deployed information
\(\mathcal S\),

\[
\inf_f\mathbb E[(X-f(\mathcal S))^2]
=\mathbb E\operatorname{Var}(X\mid\mathcal S). \tag{C24}
\]

Equivalently, for an old-estimator residual \(R\), the largest possible gain
from the state is

\[
\operatorname{Var}(\mathbb E[R\mid\mathcal S]). \tag{C25}
\]

No retraining can cross this bound. Channelwise conditional residual fractions
from (C24)--(C25), substituted in (C1), are the exact feasibility test; two
same-state/different-(C23) witnesses prove positive conditional variance and
therefore an irreducible deficit.

## Graph-node split (do not collapse)

The direct-action lane decomposes into five logically independent nodes:

1. **K31 scalar Bellman recurrence -- DERIVED.** Evidence: (C15), (C17)--(C20).
   Terminal criterion: exact 3--1 context decomposition and discrete-adjoint
   derivation accepted; no explicit q4 state remains.
2. **Affine-quadratic Jordan compression -- DERIVED, premise-conditional.**
   Evidence: log-mgf proof (C16) and Jordan form (C23). Terminal criterion:
   verify that deployed \((h,N,r,R)\) and the banked Q24 square are the same
   causal objects assumed by (C23); otherwise reopen as an acquisition node.
3. **Old-span/counterterm subtraction -- OPEN.** The rule is exact (subtract
   \(\langle\Lambda,K_{31}^{\rm CP3/Q24/DTOT2}\rangle\), retain the cumulant
   disconnection subtraction exactly once), but the full deployed old-span map
   has not been exhibited. Terminal criterion: an algebraic partition whose sum
   reproduces the full action and whose old component reproduces the deployed
   feature action identically.
4. **Beyond-affine-quadratic residual -- BOUND DERIVED, MAGNITUDE OPEN.**
   Evidence: (C22) plus the exact ReLU H>4 tail. Terminal criterion: compose
   certified per-layer L4 tail and Bellman norms so its channel deficits satisfy
   (C1), or close the lane if they cannot.
5. **Moment/action acquisition observability -- OPEN, DECISIVE.** Evidence:
   acquisition error (C21), sufficiency pairings (C23), and inverse-oracle gate
   (C24)--(C25). Terminal criterion: prove the deployed state preserves all
   required pairings/contexts and the suffix conditional residual fits (C1), or
   construct a same-state/different-action witness and quantify the resulting
   conditional variance.

The five statuses must not be merged: an exact scalar identity does not certify
AQ adequacy, old-span novelty, deployable observability, or endpoint energy.

For a concrete rank-one context test, define the symmetric matrix

\[
(D_i)_{jk}=\langle Z_{ii},Z_{jk}\rangle
+2\langle Z_{ij},Z_{ik}\rangle.
\]

Then a context \(u\) observes exactly

\[
Q^\perp(e_i,e_i,u,u)=16u^TD_iu. \tag{C26}
\]

After centering \(\widetilde D_i=D_i-\bar D\), contexts \(u_1,\ldots,u_m\)
have exact secant energy

\[
256\sum_{i,f}\langle\widetilde D_i,u_fu_f^T\rangle_F^2
=256\sum_i\langle\widetilde D_i,\mathcal F_U\widetilde D_i\rangle_F,
\]

where

\[
\mathcal F_U=\sum_f|u_fu_f^T\rangle\langle u_fu_f^T|.
\]

The restricted context map is injective on a target subspace \(S\) if and only
if \(\mathcal F_U|_S\) is positive definite. Since its rank is at most \(m\),
\(m<\dim S\) proves impossibility. Direct K31 evaluation avoids storing
\(Q^\perp\), but it does not evade this completeness condition unless its
Bellman 3--1 contexts already span the required endpoint subspace.

## Checkpoint 8: exact deployed old-span and counterterm partition

This section is tied to the actual recurrence in
`p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py` and to the DTOT
recurrence in `public_g19_attribution_v1/run_tensor_recursion_v4.py`. It is a
symbolic audit only; no scientific run was used.

### Collision maps and the implicit deployed tensors

Let \(R_{21}\) read the \(iij\) collision plane of a symmetric third tensor,
and let \(R_{40},R_{31},R_{22}\) read the fourth-order collision classes. Define
\(E_{21}(H)\) to be the symmetric third tensor supported on repeated-coordinate
patterns with \(R_{21}E_{21}(H)=H\). Define \(E_{31}(J)\) analogously by

\[
(E_{31}J)_{iiij}=J_{ij},
\]

with all permutations and the diagonal understood, and zero off-diagonal 22,
211, and 1111 support in that coordinate basis. Finally define the covariance
pairing tensor

\[
\mathcal P_\alpha(C)_{ijkl}
=\alpha(C_{ij}C_{kl}+C_{ik}C_{jl}+C_{il}C_{jk}). \tag{C27}
\]

Its collision reads are

\[
R_{40}\mathcal P_\alpha=3\alpha v_i^2,
\quad
(R_{31}\mathcal P_\alpha)_{ij}=3\alpha v_iC_{ij},
\]

\[
(R_{22}\mathcal P_\alpha)_{ij}
=\alpha(v_iv_j+2C_{ij}^2). \tag{C28}
\]

The deployed CP states are full tensors even though only collisions are read:

\[
T_l^{\rm CP3}=\sum_r\lambda_r
\operatorname{Sym}_{\rm sum}(p_r\otimes p_r\otimes q_r),
\qquad
K_l^{\rm CP4}=\sum_r\gamma_r u_r^{\otimes4}. \tag{C29}
\]

The deployed post-reset state is therefore the implicit pair

\[
T_l^D=T_l^{\rm CP3}+E_{21}(H_l^3), \tag{C30}
\]

\[
K_l^D=K_l^{\rm CP4}+\mathcal P_{\alpha_l}(C_l)
+E_{31}(H_l^4). \tag{C31}
\]

`_push_cubic` and `_push_quartic` are exactly the collision reads of
\(W^{\otimes3}E_{21}(H^3)\) and
\(W^{\otimes4}\{\mathcal P_\alpha+E_{31}(H^4)\}\); CP3/CP4 push their coherent
factors. Thus (C30)--(C31), not a hand list of planes, are the actual deployed
old tensor span.

### Exact CP3 read-before-inject partition

At transition \(l\), the code order is:

1. push the previous CP3 factors and cubic star;
2. read \(K_{21,l}^{\rm CP,pre}\) and add the pushed star plane;
3. let Q24 consume the total pre-plane;
4. form

   \[
   K_{21,l}^{\rm CP,post}
   =G_l^{\otimes3}K_{21,l}^{\rm CP,pre}+B_l^{21}, \tag{C32}
   \]

   where `_cp_post_read` includes the analytic coordinate birth \(B_l^{21}\);
5. set

   \[
   H_l^3=K_{21,l}^{\rm target}-K_{21,l}^{\rm CP,post}; \tag{C33}
   \]

6. only after (C33), `_cp_grow` materializes the same birth in the factor bank
   for future transitions.

Consequently the birth in (C32) occurs exactly once. Subtracting the pre-read
rather than the post-read from the target would duplicate the coordinate birth.
The identity

\[
R_{21}T_l^D=K_{21,l}^{\rm target} \tag{C34}
\]

is exact for the recurrence's own target.

### Exact CP4/Q24 partition

The pre-reset fourth collisions are assembled once as

\[
R_*K_l^{\rm pre}
=R_*W_l^{\otimes4}K_{l-1}^{\rm CP4}
+R_*W_l^{\otimes4}
\{\mathcal P_{\alpha_{l-1}}+E_{31}(H_{l-1}^4)\}. \tag{C35}
\]

This is `_assemble_quartic_provider`. Q24 then consumes the total
\((k_4,K_{31},K_{22})\) exactly once. Its literal pair weights are

\[
1+L_3(K_3,K_{21})+L_4(k_4,K_{31},K_{22})
+\tfrac12L_3(K_3,K_{21})^2, \tag{C36}
\]

including all CP3/star cross products because `_edgeworth_weights` receives the
already-summed \(K_{21}^{\rm pre}\).

After the reset, `_cp4_reset` first forms the retained CP4 provider, including
the current diagonal birth. `_quartic_residual` subtracts those provider planes
once. On the residual target it fits

\[
\alpha_l=
\frac{\langle\operatorname{sym}(K_{22}^{\rm target}
-K_{22}^{\rm CP4}),P_{22}(C_l)\rangle}
{\|P_{22}(C_l)\|^2}, \tag{C37}
\]

where \(P_{22}(C)=v\otimes v+2C^{\odot2}\), and sets

\[
H_{31,l}^4=K_{31,l}^{\rm target}-K_{31,l}^{\rm CP4}
-3\alpha_lv_l[:,\!None]\odot C_l, \tag{C38}
\]

\[
H_{40,l}^4=k_{4,l}^{\rm target}-k_{4,l}^{\rm CP4}
-3\alpha_lv_l^{\odot2}. \tag{C39}
\]

Therefore

\[
R_{31}K_l^D=K_{31,l}^{\rm target},
\qquad
R_{40}K_l^D=k_{4,l}^{\rm target} \tag{C40}
\]

exactly. The omitted same-basis fourth-order information begins with

\[
Z_{22,l}=K_{22,l}^{\rm target}-K_{22,l}^{\rm CP4}
-\alpha_lP_{22}(C_l), \tag{C41}
\]

which obeys

\[
\langle\operatorname{sym}Z_{22,l},P_{22}(C_l)\rangle=0 \tag{C42}
\]

by construction, plus unrepresented 211 and 1111 sectors. Q24 computes the
target \(K_{22}\), but (C41) is not retained in (C31).

### Static K31 duplicate theorem

For any centered law, with covariance \(C\), and any Bellman matrix \(\Lambda\),

\[
S_\Lambda(X)=\mathbb E[(X^{\odot3})^T\Lambda X]
-3\sum_{ab}\Lambda_{ab}C_{aa}C_{ab}
=\langle\Lambda,K_{31}\rangle. \tag{C43}
\]

In (C43), \(K_{31}\) denotes the augmented collision matrix whose off-diagonal
is the stored D11 `k31` plane and whose diagonal is the separately stored `k4`
vector. The deployed `k31` array itself is zeroed on the diagonal. Every term
in (C44)--(C45) uses this same augmentation, so (C40) supplies both pieces.

Combining (C38)--(C40) gives

\[
\langle\Lambda,K_{31}^{\rm target}\rangle
=\langle\Lambda,K_{31}^{\rm CP4}\rangle
+3\alpha_l\langle\Lambda,v[:,\!None]\odot C\rangle
+\langle\Lambda,H_{31}^4\rangle. \tag{C44}
\]

Hence a direct scalar K31 action on the same post-reset coordinate collision is
**exactly duplicated by D11**. It may only enter as the residual

\[
S_{\Lambda,l}^{\rm new}
=S_{\Lambda,l}^{\rm candidate}
-\langle\Lambda,K_{31,l}^{\rm target,D11}\rangle, \tag{C45}
\]

or as the future action of a tensor completion not represented by (C31). Adding
the full scalar action is invalid.

The subtraction can itself be evaluated tensor-free. For arbitrary current
coordinate contexts \(u,v\), write

\[
a_r=u^Tu_r,\qquad b_r=v^Tu_r,
\qquad C_{uu}=u^TCu,\qquad C_{uv}=u^TCv.
\]

The exact action of the full D11 implicit tensor (C31) is

\[
\begin{aligned}
\chi_D(u,v)={}&\sum_r\gamma_ra_r^3b_r
+3\alpha C_{uu}C_{uv}\\
&+\sum_iH_{ii}^4u_i^3v_i
+\sum_{i\ne j}H_{ij}^4
\{u_i^3v_j+3u_i^2u_jv_i\}.
\end{aligned} \tag{C45a}
\]

If the affine-quadratic candidate has combined context coefficients
\(h_u,N_u,h_v,N_v\), its admissible residual action is exactly

\[
\begin{aligned}
\chi_{\rm res}(u,v)={}&48\operatorname{tr}(N_u^3N_v)
+24\{h_u^TN_uN_vh_u+h_u^TN_u^2h_v\}\\
&-\chi_D(u,v).
\end{aligned} \tag{C45b}
\]

Summing (C45b) over the Bellman 3--1 context decomposition is the literal
CP4/pair/star counterterm. The CP3/Q24-generated K31 birth is already inside
the target split (C38)--(C40), so it must not be subtracted or added again under
a second diagram name.

The genuinely structural fourth-order residual is

\[
\Delta K_l=K_l^{\rm candidate}-K_l^D. \tag{C46}
\]

If the candidate is calibrated to the same current k4/K31 collisions, then

\[
R_{40}\Delta K_l=R_{31}\Delta K_l=0, \tag{C47}
\]

and its first possible value is future mixing, for example

\[
\delta S_{l\to l+1}
=\left\langle\Lambda_{l+1},
R_{31}(W_{l+1}^{\otimes4}\Delta K_l)\right\rangle
=\left\langle(W_{l+1}^{\otimes4})^*R_{31}^*\Lambda_{l+1},
\Delta K_l\right\rangle. \tag{C48}
\]

Equation (C48) is the exact old-span counterterm for the tensor-free Bellman
shortcut. The source is principally (C41) and its lawful 211/1111 completion,
not same-layer K31 support.

### Causal timing gate

The actual layer order is

\[
\text{pre-read}\ \longrightarrow\ \text{Q24 consume/report}
\ \longrightarrow\ \text{post target}\ \longrightarrow\
\text{CP/star split and store}. \tag{C49}
\]

Thus

\[
\frac{\partial y_l}{\partial K_l^{\rm post}}=0. \tag{C50}
\]

A completion born at reset \(l\) may be contracted immediately with a compiled
future costate, but its physical effect starts at transition \(l+1\). Feeding it
to Q24 at layer \(l\) and also adding its future Bellman action is a one-layer
causal duplicate. Conversely, a missing **pre** cumulant may be consumed at the
current reset, but must not also be labeled a post birth.

### DTOT2 versus CP3/Q24

DTOT2 is an alternative full-third-tensor completion, not an additive CP3
feature. Let

\[
\Delta T_l=T_l^{\rm DTOT2}-T_l^D. \tag{C51}
\]

If both recurrences are forced to the same post K21 target, then

\[
R_{21}\Delta T_l=0, \tag{C52}
\]

and only future mixing

\[
R_{21}(W_{l+1}^{\otimes3}\Delta T_l) \tag{C53}
\]

is structurally new. The actual files do not make their plane-reset formulas
symbolically identical: D11 uses the seven-rho-term Q24 kernel and always
includes the full \(\tfrac12L_3^2\) map, whereas
`run_tensor_recursion_v4.py::emission_K21` uses a 20-term pair series and its
second-order pattern map only when `SECOND_ORDER=1`. Therefore (C51) must be
split into its plane-estimator difference and zero-plane distinct completion;
neither whole recurrence may be added as independent evidence.

Once a DTOT2 plane difference changes \(K_{21}^{\rm pre}\), Q24 already creates
all of its linear and quadratic descendants. A scalar K31 action evaluated from
that changed state is not an additional feature; the lawful object is the total
endpoint finite difference of the co-adapted recurrence.

### Symbolic duplicate test for the actual recurrence

Use formal source tags

\[
c=\mathrm{CP3},\ s=\mathrm{cubic\ star},\
p=\mathrm{CP4},\ a=\mathrm{pairing},\ h=\mathrm{quartic\ star},
\]

and candidate residual tags \(d=\Delta T\), \(q=\Delta K\). At a pre-reset
Q24 call, baseline third-order tags are \(c+s\), and baseline fourth-order tags
are \(p+a+h\). The deployed old jet contains

\[
c,s,p,a,h,c^2,cs,s^2. \tag{C54}
\]

Adding \(d\) lawfully introduces

\[
d,cd,sd,d^2, \tag{C55}
\]

through the same Q24 call. Adding \(q\) introduces its single \(L_4\) response;
any \(d\)-\(q\) coupling generated by connectedization or later layers belongs
to the union once.

Let \(\mathcal R_l(k_3,k_4)\) denote the **literal** branch-fixed deployed reset,
including `_edgeworth_weights`, univariate correction, connectedization,
covariance repair, CP resets, and the future continuation. The exact union
increment is

\[
\Delta_{d,q}\mathcal R_l
=\mathcal R_l(k_3+d,k_4+q)-\mathcal R_l(k_3,k_4). \tag{C56}
\]

Independent attribution, if desired, is the Moebius partition

\[
\Delta_d=\mathcal R(k_3+d,k_4)-\mathcal R(k_3,k_4),
\]

\[
\Delta_q=\mathcal R(k_3,k_4+q)-\mathcal R(k_3,k_4),
\]

\[
\Delta_{dq}=\Delta_{d,q}-\Delta_d-\Delta_q. \tag{C57}
\]

Equations (C56)--(C57), applied with the time shift (C49), are the symbolic
duplicate test. A proposed scalar shortcut is a duplicate through order three
iff its tagged jet equals a coefficient already in (C54)--(C57) and both are
retained. It is novel only on the Bellman-observable residual tensors (C46) or
(C51). This test also handles the higher products introduced by literal
connectedization, so no assumption that the code is a perfectly truncated
polynomial is needed.

The affine-quadratic formula remains a static acquisition identity, not a proof
of reset closure. For general centered \(U,V\), the exact influences are

\[
\psi_U=3(U^2V-2C_{UV}U-C_{UU}V),
\qquad
\psi_V=U^3-3C_{UU}U. \tag{C58}
\]

Correlated-ReLU gate multiplication regenerates components outside H1/H2, so a
finite Jordan recurrence is exact only for a genuinely affine-quadratic Markov
law or a larger sufficient state.

### Node-3 subnodes (keep separate)

1. **3A CP3 read-before-inject partition -- DERIVED/CLOSED.** Equations
   (C30), (C32)--(C34). Reopen only if the deployed lineage changes call order.
2. **3B CP4/pair/star tensor partition -- DERIVED/CLOSED.** Equations
   (C31), (C35)--(C42). The discarded K22/211/1111 completion remains a separate
   acquisition problem.
3. **3C same-layer scalar K31 duplicate -- DERIVED/NOGO AS ADDITIVE.** Equations
   (C43)--(C45). Only a residual difference is admissible.
4. **3D future completion counterterm -- DERIVED, SOURCE OPEN.** Equation (C48)
   is exact; determining \(\Delta K\) belongs to acquisition observability.
5. **3E DTOT2 overlap -- PARTITION DERIVED, ALIGNMENT OPEN.** Equations
   (C51)--(C53). The actual plane kernels differ, so a common-source symbolic
   alignment is still required before any value attribution.
6. **3F symbolic duplicate protocol -- DERIVED/CLOSED.** Equations
   (C54)--(C57), with causal shift (C49), are the binding test.

The highest-leverage next question is no longer whether scalar K31 exists. It is
whether the deployable primal/dual moments determine the Bellman action of the
zero-current-K31 completion \(\Delta K\), especially the discarded
\(Z_{22}\rightarrow K_{31}\) future-mixing path, with conditional error small
enough for (C1).

## Checkpoint 9: future H4 completion in the full D11-interface kernel

The CP3 and CP4 read-before-inject results above remain terminal. This
checkpoint promotes only node 3D, the H4 completion discarded by the post-reset
interface.

### Full old interface and its typed kernel

For the retained CP4 factors define the fourth-tensor and 22-collision spans

\[
\mathcal U_l^4=\operatorname{span}\{u_r^{\otimes4}:r\le r_4\},
\qquad
\mathcal U_l^{22}=\operatorname{span}
\{b_rb_r^T:b_r=u_r^{\odot2}\}. \tag{C59}
\]

The complete linearized D11 H4 interface at a post reset consists of

1. the retained CP4 coordinates in \(\mathcal U_l^4\);
2. every 40 and 31 collision, represented by the quartic star after CP4
   subtraction;
3. the one residual-22 coordinate along
   \(P_{22}(C)=v\otimes v+2C^{\odot2}\), represented by \(\alpha\).

Let \(\mathcal I_l^D\) denote this interface. Its Frobenius-Schur residual is

\[
\Delta K_l=\Pi_{\ker\mathcal I_l^D}K_l. \tag{C60}
\]

Because the 40 and 31 classes are fully removed, (C60) has the unique
coordinate-multiplicity decomposition

\[
\Delta K_l=E_{22}(Z_l)+E_{211}(L_l)+E_{1111}(M_l). \tag{C61}
\]

The explicit 22 source is

\[
Z_l=\Pi_{(\mathcal U_l^{22}+\operatorname{span}\{P_{22}(C_l)\})^\perp}
\operatorname{sym}K_{22,l}^{\rm target}. \tag{C62}
\]

Equivalently, subtract the complete CP4 22 action and then the pairing
coordinate before taking the orthogonal remainder. The star contributes no
off-diagonal 22 component. Therefore (C62) is outside the full CP4/pair/star
old span, not merely outside the current K31 plane.

For width \(n\) and CP4 rank \(r_4\),

\[
\dim\mathcal Z_{22}\ge {n\choose2}-r_4-1. \tag{C63}
\]

At \(n=1024,r_4=2048\), this lower bound is

\[
523776-2049=521727. \tag{C64}
\]

The unprojected coordinate dimensions of the other typed sources are

\[
\dim\mathcal H_{211}=n{n-1\choose2},
\qquad
\dim\mathcal H_{1111}={n\choose4}. \tag{C65}
\]

At width 1024 the 211 dimension alone is 535,299,072. Subtracting the at-most
2049 CP4/pair constraints cannot make either generic source compact.

### First nonzero downstream collision action

Let \(w_a\) be column \(a\) of the next dense weight matrix \(W\). Current-layer
action is zero by (C50). At the next pre-interface, the 22 source gives exactly

\[
\delta k_{4,a}=3(w_a^{\odot2})^TZ(w_a^{\odot2}), \tag{C66}
\]

\[
\delta K_{31,ab}
=3(w_a^{\odot2})^TZ(w_a\odot w_b), \tag{C67}
\]

\[
\delta K_{22,ab}
=(w_a^{\odot2})^TZ(w_b^{\odot2})
+2(w_a\odot w_b)^TZ(w_a\odot w_b). \tag{C68}
\]

These constants use the convention that \((E_{22}Z)_{iijj}=Z_{ij}\) for each
of the six permutations. In particular, (C66) equals
\(6\sum_{i<j}Z_{ij}W_{ia}^2W_{ja}^2\).

For a 211 source, indexed by a repeated coordinate \(i\) and two distinct
singletons \(j<k\), the next 31 action on arbitrary contexts \(u,v\) is

\[
\begin{aligned}
E_{211}(L)[u,u,u,v]
=\sum_{i;j<k}L_{i;jk}\{&6u_iu_ju_kv_i
+3u_i^2u_kv_j\\
&+3u_i^2u_jv_k\}.
\end{aligned} \tag{C69}
\]

Its 40 action is \(12\sum L_{i;jk}u_i^2u_ju_k\), and its 22 action is

\[
\sum_{i;j<k}L_{i;jk}\{2u_i^2v_jv_k+2u_ju_kv_i^2
+4u_iu_jv_iv_k+4u_iu_kv_iv_j\}. \tag{C70}
\]

For a 1111 source on unordered distinct quadruples \(S=\{i,j,k,m\}\),

\[
E_{1111}(M)[u,u,u,u]
=24\sum_SM_S\prod_{s\in S}u_s, \tag{C71}
\]

\[
E_{1111}(M)[u,u,u,v]
=6\sum_SM_S\sum_{r\in S}v_r\prod_{s\in S\setminus\{r\}}u_s, \tag{C72}
\]

\[
E_{1111}(M)[u,u,v,v]
=4\sum_SM_S\sum_{A\subset S,\ |A|=2}
\prod_{i\in A}u_i\prod_{j\in S\setminus A}v_j. \tag{C73}
\]

Equations (C66)--(C73) are the typed sources that dense transport converts into
the next Q24 interface.

Let \(\Lambda_{40},\Lambda_{31},\Lambda_{22}\) be the exact adjoints obtained by
pulling the endpoint Bellman covector through the literal next Q24 reset and
connectedization. Before absorbing Edgeworth denominators into those adjoints,
the first downstream scalar action is

\[
\mathcal A_{l\to l+1}^{(1)}
=\frac1{24}\langle\Lambda_{40},\delta k_4\rangle
+\frac1{6}\langle\Lambda_{31},\delta K_{31}\rangle
+\frac1{4}\langle\Lambda_{22},\delta K_{22}\rangle, \tag{C74}
\]

with both 31 orientations included. In coordinate-free form,

\[
\mathcal A_{l\to l+1}^{(1)}
=\left\langle
\Pi_{\ker\mathcal I_l^D}(W^{\otimes4})^*
\left(\tfrac1{24}R_{40}^*\Lambda_{40}
+\tfrac1{6}R_{31}^*\Lambda_{31}
+\tfrac1{4}R_{22}^*\Lambda_{22}\right),
\Delta K_l\right\rangle. \tag{C75}
\]

Thus one-step value is nonzero exactly when the pulled-back Bellman tensor has a
nonzero projection onto the full old-interface kernel. This is the strict
novelty gate.

### Nonzero Schur-residual proof

The 22 residual is not merely dimensionally possible; it is a nonzero analytic
source of the actual Q24 map. Consider standardized jointly Gaussian inputs and
their centered ReLU outputs. Put

\[
\mu=\mathbb E[Z_+]=\frac1{\sqrt{2\pi}},
\qquad
v=\operatorname{Var}(Z_+)=\frac12-\frac1{2\pi}.
\]

For a pair with correlation \(\rho\), let \(c(\rho)\) be the output covariance
and let

\[
K_{22}(\rho)=\operatorname{cum}(X,X,Y,Y).
\]

Price's identity at independence gives

\[
c'(0)=(\mathbb E[\mathbf1_{Z>0}])^2=\frac14. \tag{C76}
\]

For \(A=(Z_+-\mu)^2\),

\[
\mathbb E[A']=2\mathbb E[(Z_+-\mu)\mathbf1_{Z>0}]=\mu,
\]

so

\[
\left.\frac d{d\rho}\operatorname{Cov}(A_1,A_2)\right|_{\rho=0}
=\mu^2. \tag{C77}
\]

Since

\[
K_{22}(\rho)=\operatorname{Cov}(A_1,A_2)-2c(\rho)^2,
\]

we obtain

\[
K_{22}'(0)=\mu^2\ne0. \tag{C78}
\]

In contrast, the D11 pairing plane is

\[
P_{22}(\rho)=v^2+2c(\rho)^2,
\qquad P_{22}'(0)=0. \tag{C79}
\]

Therefore \(K_{22}(\rho)\) is not a scalar multiple of the pairing plane on any
neighborhood of zero. At the first reset the coordinate CP4 births have no
off-diagonal 22 span. Any covariance with at least two nonidentical small pair
correlations consequently produces \(Z\ne0\). The seven-term Q24 rho series
contains the exact linear coefficient used in (C76)--(C78), so this is a witness
for the actual deployed symbolic recurrence, not only the infinite Mehler map.

For this nonzero \(Z\), a rotation mixing a supported coordinate pair makes
(C66) or (C67) nonzero. Hence the one-step collision map is not identically zero
on \(\ker\mathcal I^D\). For dense \(W\) and a non-annihilating Bellman
covector, zero action is confined to the zero set of a nontrivial polynomial.
The residual and its future visibility are therefore generically nonzero.

### Minimal state and work

For the full next K31 plane, let \(W_2=W\odot W\) and \(R=ZW_2\). Then

\[
\delta K_{31}=3(R\odot W)^TW, \tag{C80}
\]

so the complete 31 exposure uses two dense matrix products: \(ZW_2\) and
\((R\odot W)^TW\). The first part of (C68) is \(W_2^TR\), one further product;
the second is the pair-conjugation

\[
\mathcal Q_Z(W)_{ab}=(w_a\odot w_b)^TZ(w_a\odot w_b), \tag{C81}
\]

which is the existing large pair-conjugation form. The 40 vector is the
diagonal of (C80). Thus a universal one-step interface costs three ordinary
products plus (C81), while retaining the \(n^2\) residual plane.

For a fixed endpoint with a Bellman 3--1 decomposition
\((u_f,v_f)_{f=1}^m\), no plane need persist. Each 22 action is simply

\[
3(u_f^{\odot2})^TZ(u_f\odot v_f), \tag{C82}
\]

so the transient \(Z\) already present during `_reset_stars` can be contracted
immediately, accumulated into one scalar, and discarded. The work is
\(O(mn^2)\), or one \(n\times n\) by \(n\times m\) product, rather than an H4
carrier. Exact generic 211 and 1111 sources, however, require respectively
\(O(n^3)\) and \(O(n^4)\) acquisition state unless a direct moment oracle or a
proved target-conditioned factorization supplies their contractions.

### Promoted-node statuses

1. **FC-A full D11-interface kernel -- DERIVED/CLOSED.** Equations
   (C59)--(C65).
2. **FC-B nonzero 22 Schur source -- GENERICALLY PROVED; INSTANCE MAGNITUDE
   OPEN.** Equations (C76)--(C79) prove the actual symbolic source is nonzero.
3. **FC-C first downstream Bellman action -- DERIVED/CLOSED.** Equations
   (C66)--(C75).
4. **FC-D minimal 22 carrier/action -- DERIVED/CLOSED CONDITIONALLY ON COSTATE.**
   Equations (C80)--(C82); the remaining premise is an available compiled
   Bellman context.
5. **FC-E 211 acquisition -- OPEN.** Its action is known by (C69)--(C70), but no
   compact sufficient provider is proved.
6. **FC-F 1111 acquisition -- OPEN.** Its action is known by (C71)--(C73), but no
   compact sufficient provider is proved.

CP3 timing node 3A, CP4 timing/partition node 3B, and the same-layer K31 NOGO
node 3C remain terminal and are not reopened by this promotion.

The next decisive question is quantitative: bound the channel energy of the
immediately available statistic (C82), after residualizing it against the old
native features, and separately bound the conditional 211/1111 remainder. The
raw-goal claim is admissible only if their combined deficits satisfy (C1).

## 2026-08-31 — checkpoint 10: exact transient-Z22 value and D10 replacement gate

This checkpoint separates six logically different claims.  The local
statistic and its arithmetic are exact.  Nonzero source support is proved.
Its corpus energy, its conditional target covariance, and the size of the
independent 211/1111 remainder are not known from algebra alone.

### The exact plane that is transiently available in reset_stars

Write \(K_{\ell}^{22,\mathrm{cp4}}\) for the active CP4 provider plane and set

\[
\widehat K_\ell^{22}
=\frac12\left[
K_\ell^{22}-K_\ell^{22,\mathrm{cp4}}
\left(K_\ell^{22}-K_\ell^{22,\mathrm{cp4}}\right)^T
\right].
\]

With \(v_\ell=\operatorname{diag}C_\ell\), define the zero-diagonal pairing
plane

\[
\mathsf P_\ell
=\operatorname{offdiag}\left(
v_\ell v_\ell^T+2C_\ell^{\odot2}
\right),
\qquad
\alpha_\ell
=\frac{\langle\widehat K_\ell^{22},\mathsf P_\ell\rangle_F}
{\|\mathsf P_\ell\|_F^2}.
\]

The information discarded by the current reset is therefore the explicit
transient plane

\[
\boxed{
Z_\ell
=\operatorname{offdiag}\left(
\widehat K_\ell^{22}-\alpha_\ell\mathsf P_\ell
\right).
}
\tag{C83}
\]

It obeys

\[
Z_\ell=Z_\ell^T,\qquad
\operatorname{diag}Z_\ell=0,\qquad
\langle Z_\ell,\mathsf P_\ell\rangle_F=0.
\tag{C84}
\]

The CP4 subtraction in (C83) is a causal counterterm: it removes the realized
provider contribution once.  The last identity in (C84) is a literal
Frobenius orthogonality statement.  The CP4 subtraction need not make
\(Z_\ell\) statistically orthogonal to every fitted old feature across a
corpus; that stronger claim requires the Schur test below.

### Exact next-layer Bellman statistic

The plane in (C83) is born after the source-layer pre-read.  It cannot alter
the already-consumed same-layer K31 plane.  Let \(W_\ell\) be the following
dense reset, with columns \(w_{\ell a}\), and let
\(Q_{\ell+1}^{31}\) be the complete frozen D10 reverse mask at the first
consumer.  Define

\[
X_\ell=W_\ell\odot W_\ell,\qquad
R_\ell=W_\ell(Q_{\ell+1}^{31})^T,\qquad
Y_\ell=W_\ell\odot R_\ell.
\tag{C85}
\]

The 22-to-31 collision identity (C67) gives

\[
\begin{aligned}
s_{Z,\ell}
&=3\sum_{a,b}(Q_{\ell+1}^{31})_{ab}
(w_{\ell a}^{\odot2})^TZ_\ell
(w_{\ell a}\odot w_{\ell b})\\
&=3\sum_a
(w_{\ell a}^{\odot2})^TZ_\ell
(w_{\ell a}\odot r_{\ell a})\\
&=\boxed{\,3\langle Z_\ell X_\ell,Y_\ell\rangle_F\,}.
\end{aligned}
\tag{C86}
\]

Equivalently, the source-plane Bellman mask is

\[
\mathsf B_\ell^Z
=3\operatorname{offdiag}\operatorname{sym}
\left(Y_\ell X_\ell^T\right),
\qquad
s_{Z,\ell}=\langle\mathsf B_\ell^Z,Z_\ell\rangle_F.
\tag{C87}
\]

No additional \(1/6\) belongs in (C86): that Edgeworth normalization is
already inside \(Q^{31}\).  Equations (C86)--(C87) use the full symmetric
zero-diagonal representation of \(Z\).  An implementation storing only
\(i<j\) must use twice the strict-upper contraction.  These conventions are
the only factor-of-two ambiguity.

For a ten-transition suffix the scalar replacement feature is

\[
S_Z=\sum_{\ell\in{\cal D}_{10}}s_{Z,\ell}.
\tag{C88}
\]

Thus no H4 plane has to survive a reset: contract the live plane in (C83)
against its compiled future context, add one scalar, and discard it.  The
actual generic D10 decomposition has \(m=n\) contexts, one per width column.
A fixed-small-\(m\) spelling is exact only after a separately proved
low-rank factorization of the D10 mask.

### Source novelty versus full-old-arena Schur novelty

Equations (C83)--(C84) prove that \(Z\) is outside the retained pairing
direction, and (C76)--(C79) prove that the symbolic ReLU reset generates a
nonzero such direction.  This is source support.  The exact target-blind
pointwise action bound is

\[
|s_{Z,\ell}|
\le
\|Z_\ell\|_F\,
\left\|
\Pi_{\mathsf P_\ell^\perp}
\mathsf B_\ell^Z
\right\|_F,
\tag{C89}
\]

where the projector is valid because of (C84).  This is tighter than counting
the pairing component of the Bellman mask as new evidence.

Full-arena novelty is a different, corpus-level statement.  Let \(F\) contain
the intercept, D14, and every incumbent feature common to the old and proposed
packages.  Let \(D\) be the old D10 feature block and let \(z\) be the corpus
vector whose entries are (C88).  With

\[
\widetilde z=(I-P_F)z,\qquad
\widetilde D=(I-P_F)D,
\]

put

\[
G_D=\widetilde D^T\widetilde D,\qquad
g_{Dz}=\widetilde D^T\widetilde z.
\]

The exact full-old-arena residual and its target-blind energy are

\[
z_{\perp\mathrm{old}}
=\widetilde z-\widetilde D\,G_D^\dagger g_{Dz},
\qquad
V_{Z\mid\mathrm{old}}
=\|z_{\perp\mathrm{old}}\|_2^2
=\|\widetilde z\|_2^2-g_{Dz}^TG_D^\dagger g_{Dz}.
\tag{C90}
\]

Therefore

\[
\boxed{V_{Z\mid\mathrm{old}}>0}
\tag{C91}
\]

is necessary and sufficient for statistical novelty after the complete old
linear arena.  Source support does not by itself prove (C91), because
different source statistics can be collinear across a finite corpus.

For an old-arena residual target \(r_{\mathrm{old}}\), define

\[
c_{Z\mid\mathrm{old}}
=\langle r_{\mathrm{old}},z_{\perp\mathrm{old}}\rangle.
\]

The exact gain of stacking this scalar, if it were affordable, would be

\[
\Delta R_{Z\mid\mathrm{old}}
=
\frac{c_{Z\mid\mathrm{old}}^2}{V_{Z\mid\mathrm{old}}}.
\tag{C92}
\]

Thus (C91) is not a value certificate; value additionally requires
\(c_{Z\mid\mathrm{old}}\ne0\).

### Target-blind norm and the coefficient-one Bellman certificate

The quantities in (C89)--(C91) use network moments, old features, and compiled
costates, but no target residual.  They are therefore the correct
target-blind support census.  Let

\[
V_Z=\|(I-P_F)z\|_2^2.
\tag{C93}
\]

By Minkowski and (C89),

\[
\sqrt{V_Z}
\le
\sum_{\ell\in{\cal D}_{10}}
\left\|
\|Z_\ell\|_F
\left\|\Pi_{\mathsf P_\ell^\perp}\mathsf B_\ell^Z\right\|_F
\right\|_{L^2(\mathrm{corpus})}.
\tag{C94}
\]

A feature norm is not a scale-invariant regression-gain ceiling by itself.
It becomes a physical value certificate only under the coefficient-one
Bellman statement

\[
(I-P_F)r=(I-P_F)z+e_Z,\qquad \|e_Z\|_2\le\eta_Z.
\tag{C95}
\]

Then Cauchy--Schwarz gives the rigorous lower bound

\[
\boxed{
\Delta R_Z
\ge
\left(\sqrt{V_Z}-\eta_Z\right)_+^2.
}
\tag{C96}
\]

Indeed
\(\langle (I-P_F)r,\widetilde z\rangle
\ge V_Z-\eta_Z\sqrt{V_Z}\), followed by division by \(V_Z\).
Without (C95), target-blind support can neither prove the sign of the target
covariance nor a positive MSE gain.

### Exact arithmetic ledger: replacement fits conditionally, stacking does not

For generic full-width D10 contexts, \(X_\ell\) is elementwise-free and
\(Z_\ell X_\ell\) is one square GEMM per transition.  There are three cases:

1. If \(R_\ell=W_\ell(Q_{\ell+1}^{31})^T\) is already supplied by the primary
   stream, (C86) costs exactly ten GEMM equivalents over D10.
2. If \(Q_{\ell+1}^{31}\) is supplied but \(R_\ell\) is not, forming
   \(R_\ell\) and \(Z_\ell X_\ell\) costs exactly twenty.
3. If the future costate/mask itself is absent, its reverse propagation must
   also be charged.  Calling the final result a scalar does not waive those
   charged cuts.

The incumbent D14 plus D10 schedule costs

\[
2(14)+2(10)=48
\tag{C97}
\]

under the 49-unit cap.  Consequently a generic ten-layer \(Z\) route cannot
be stacked: it would cost at least \(48+10=58\).  Replacing D10 is arithmetically
admissible in cases 1 and 2:

\[
C_{\mathrm{D14}+Z}=38\quad\hbox{or}\quad48,
\tag{C98}
\]

respectively.  It is not admissible if rebuilding the missing future costate
pushes the total above 49.  A purported \(m\ll n\) route costs
\(\sum_\ell m_\ell/n\) GEMM equivalents only if the exact contexts are already
available; the actual unfactorized D10 mask has generic \(m=n\).

### Can replacement clear \(2.86\times10^{-10}\)?

Replacement and stacking have different Schur tests.  Continue to condition
on the common block \(F\), and let

\[
\widetilde r=(I-P_F)r,\quad
c_Z=\langle\widetilde r,\widetilde z\rangle,\quad
c_D=\widetilde D^T\widetilde r.
\]

The two competing conditional gains are

\[
\Delta R_Z=\frac{c_Z^2}{V_Z},\qquad
\Delta R_D=c_D^TG_D^\dagger c_D.
\tag{C99}
\]

Replacing D10 clears the remaining quartic raw requirement
\(g_0=2.86\times10^{-10}\) if and only if

\[
\boxed{
\Delta R_Z-\Delta R_D\ge g_0.
}
\tag{C100}
\]

Combining (C96) with (C100) gives the sufficient certificate

\[
\boxed{
\left(\sqrt{V_Z}-\eta_Z\right)_+^2
-\Delta R_D
\ge2.86\times10^{-10}.
}
\tag{C101}
\]

Equivalently, in the nontrivial branch,

\[
\sqrt{V_Z}
\ge
\eta_Z+\sqrt{\Delta R_D+2.86\times10^{-10}}.
\tag{C102}
\]

Even with no lost D10 value and \(\eta_Z=0\), this requires amplitude at least

\[
\sqrt{2.86\times10^{-10}}=1.691\times10^{-5}.
\tag{C103}
\]

Using \(1.0313\times10^{-9}\) as the verified ceiling of the complete quartic
ladder, a necessary capture fraction for a quartic-only replacement is

\[
\boxed{
p_Z
\ge
\frac{2.86\times10^{-10}+\Delta R_D}{1.0313\times10^{-9}}
=0.27732+
\frac{\Delta R_D}{1.0313\times10^{-9}}.
}
\tag{C104}
\]

In particular, if

\[
\Delta R_D>1.0313\times10^{-9}-2.86\times10^{-10}
=7.453\times10^{-10},
\tag{C105}
\]

then no feature confined to that quartic ceiling can win the replacement
comparison.  Conversely, the algebra above does not establish that
\(\Delta R_D\), \(V_Z\), or \(\eta_Z\) passes (C101).  The honest conclusion is
that replacement is cost-feasible but value-unresolved.

### Independent 211/1111 remainder bound

Let the hidden source at layer \(\ell\) be

\[
H_\ell
=E_{211}(L_\ell)+E_{1111}(M_\ell),
\]

and let \(\mathcal B_\ell\) be the complete H4 Bellman covector pulled back to
that source layer.  Its omitted scalar action is

\[
S_H=\sum_\ell
\left\langle
\Pi_{211+1111}\mathcal B_\ell,H_\ell
\right\rangle_F.
\tag{C106}
\]

Sector orthogonality, Cauchy--Schwarz, and Minkowski give

\[
\begin{aligned}
\|S_H\|_{L^2}
\le\sum_\ell\Big\|
&\|\Pi_{211}\mathcal B_\ell\|_F
\sqrt{12\sum_{i,j,k}^{\rm distinct}|L_{\ell;i;jk}|^2}\\
&+\|\Pi_{1111}\mathcal B_\ell\|_F
\sqrt{24\sum_{i<j<k<m}|M_{\ell;ijkm}|^2}
\Big\|_{L^2}.
\end{aligned}
\tag{C107}
\]

The factors 12 and 24 are the permutation multiplicities:

\[
\|E_{211}(L_\ell)\|_F^2
=12\sum|L_{\ell;i;jk}|^2,\qquad
\|E_{1111}(M_\ell)\|_F^2
=24\sum|M_{\ell;ijkm}|^2.
\tag{C108}
\]

If layer actions are independently proved orthogonal, the squared
layer bounds may be summed; absent that proof, the amplitude sum in (C107) is
the lawful bound.

There is no finite bound of the form

\[
\|S_H\|_{L^2}\le c\|S_Z\|_{L^2}
\tag{C109}
\]

from the old interface or reset_stars alone.  The 22, 211, and 1111
coefficient spaces are a direct sum: one may set \(Z=0\) while varying
\(L\) or \(M\).  Hence \(\eta_Z\) in (C95)--(C102) must explicitly include
the conditional 211/1111 Bellman remainder (plus mask/frame motion), unless a
separate moment law bounds it.  Treating (C107) as zero would silently assume
the decisive missing lemma.

### Checkpoint-10 split-node statuses

1. **Z10-A live transient statistic — DERIVED/CLOSED.**  Equations
   (C83)--(C88) give the exact read-before-inject action and all normalization
   conventions.
2. **Z10-B full-arena Schur novelty — CRITERION CLOSED; INSTANCE OPEN.**
   Equations (C90)--(C92).  The ReLU source is generically nonzero, but
   \(V_{Z\mid\mathrm{old}}>0\) and target covariance have not been measured or
   analytically lower-bounded on the actual corpus.
3. **Z10-C target-blind norm/Bellman certificate — DERIVED; MAGNITUDE OPEN.**
   Equations (C89), (C93)--(C96).  A norm is evidence only after the
   coefficient-one residual premise and its error \(\eta_Z\) are certified.
4. **Z10-D portfolio cost — STANDARD STACK CLOSED_COST; REPLACEMENT
   CONDITIONAL_OPEN.**  Equations (C97)--(C98).  Replacement fits at 38 or 48
   only with the stated future-context provenance.
5. **Z10-E \(2.86\times10^{-10}\) value gate — OPEN.**  Equations
   (C99)--(C105) are exact.  The missing quantities are \(V_Z\), \(\eta_Z\),
   and conditional lost D10 gain \(\Delta R_D\).
6. **Z10-F 211/1111 remainder — BOUND DERIVED; MAGNITUDE OPEN.**  Equations
   (C106)--(C109).  No Z-only bound exists.

The CP3 timing, CP4 timing/partition, and same-site K31 duplicate siblings
remain terminal.  This checkpoint opens no same-site route: its only new
quantity is the delayed 22-to-31 Bellman completion in (C86).

### Addendum: generic quotient-rank lower bound rules out a tiny context bank

There is also a sharp algebraic obstruction to replacing the \(m=n\) contexts
in (C86) by one or two exact contexts.  Let

\[
\mathcal S_{\rm old}^{22}
=\mathcal U_{\rm CP4}^{22}
+\operatorname{span}\{\mathsf P_{22}\}
+\operatorname{Diag}_n ,
\qquad
\dim\mathcal S_{\rm old}^{22}\le r_4+1+n.
\tag{C110}
\]

The diagonal is included because \(Z\) annihilates it.  For a Bellman mask

\[
M=\operatorname{sym}\left[
(W\odot R)(W\odot W)^T
\right],
\]

define its effective quotient rank

\[
q_{\rm eff}(M)
=\min_{S\in\mathcal S_{\rm old}^{22}}\operatorname{rank}(M-S).
\tag{C111}
\]

The variety of symmetric \(n\times n\) matrices of rank at most \(q\) has
codimension

\[
\operatorname{codim}{\cal V}_{q}
=\frac{(n-q)(n-q+1)}2.
\tag{C112}
\]

Consequently a generic affine translate by a subspace of dimension \(d\)
cannot meet \({\cal V}_q\) when the right side of (C112) exceeds \(d\).
For \(n=1024\), \(r_4=2048\), hence \(d\le3073\); because

\[
\frac{78\cdot79}{2}=3081>3073,
\]

we obtain the generic lower bound

\[
\boxed{q_{\rm eff}\ge1024-77=947.}
\tag{C113}
\]

The mask family is dominant, not an artificially small subset: fix a dense
\(W\) for which \(W\odot W\) is invertible.  Varying the generic context
\(R\) varies \(W\odot R\) freely, and the map
\(Y\mapsto\operatorname{sym}(Y(W\odot W)^T)\) is onto
\(\operatorname{Sym}_n\).  Thus (C113) applies to a generic compiled D10 mask.

An exact rank-\(q\) context contraction costs \(q/n\) square-GEMM equivalents
per transition.  Even after the complete old quotient, a generic ten-layer
suffix therefore costs at least

\[
10\frac{947}{1024}=9.248046875
\tag{C114}
\]

units.  This nearly matches the direct ten-unit realization and closes the
generic one-/two-context shortcut.  Only a proved nongeneric invariant of the
actual frozen masks, or exact reuse of already billed contexts, can evade
(C113)--(C114).

### Addendum: analytic small-correlation size of the ReLU Z22 birth

The generic-nonzero witness can be sharpened to a target-blind leading norm.
Let

\[
f(z)=z_+-\mu,\qquad
\mu=\frac1{\sqrt{2\pi}},\qquad
v_0=\mathbb E f(Z)^2=\frac12-\frac1{2\pi},
\]

and \(g(z)=f(z)^2-v_0\).  In the orthonormal probabilists-Hermite basis,

\[
\langle f,\psi_1\rangle=\frac12,\qquad
\langle f,\psi_2\rangle=\frac{\mu}{\sqrt2},
\]

\[
\langle g,\psi_1\rangle=\mu,\qquad
\langle g,\psi_2\rangle=\frac{1-1/\pi}{\sqrt2}.
\tag{C115}
\]

For a standard bivariate Gaussian pair of correlation \(\rho\), let
\(c(\rho)=\mathbb E[f(Z_1)f(Z_2)]\).  Mehler's identity and (C115) give

\[
c(\rho)=\frac14\rho+\frac1{4\pi}\rho^2+O(\rho^3).
\tag{C116}
\]

The connected 22 cumulant is

\[
\begin{aligned}
k_{22}(\rho)
&=\operatorname{Cov}\!\left(f(Z_1)^2,f(Z_2)^2\right)-2c(\rho)^2\\
&=\frac1{2\pi}\rho+
\left[
\frac12\left(1-\frac1\pi\right)^2-\frac18
\right]\rho^2+O(\rho^3),
\end{aligned}
\tag{C117}
\]

whereas the retained pairing plane is

\[
p_{22}(\rho)=v_0^2+2c(\rho)^2
=v_0^2+\frac18\rho^2+O(\rho^3).
\tag{C118}
\]

Thus the new source is linear in correlation while its old pairing competitor
has no linear response.

For unequal preactivation scales, index unordered pairs by \(e=(i,j)\), put
\(a_e=\sigma_i^2\sigma_j^2\), and let

\[
\bar\rho_w
=\frac{\sum_ea_e^2\rho_e}{\sum_ea_e^2}.
\]

Projecting the vector
\(a_ek_{22}(\rho_e)\) on the pairing vector
\(a_ep_{22}(\rho_e)\) gives, uniformly for
\(\epsilon=\max_e|\rho_e|\to0\),

\[
(Z_{22})_e
=\frac{a_e}{2\pi}\left(\rho_e-\bar\rho_w\right)
+O(a_e\epsilon^2),
\tag{C119}
\]

and, in the full symmetric zero-diagonal convention,

\[
\boxed{
\|Z_{22}\|_F^2
=\frac{2}{(2\pi)^2}
\sum_ea_e^2(\rho_e-\bar\rho_w)^2
+O\!\left(\epsilon^3\sum_ea_e^2\right).
}
\tag{C120}
\]

Equation (C120) is a quantitative, label-free source witness: heterogeneous
pair correlations force a first-order residual, while an equicorrelated
plane can be absorbed at this order.  At a layer with an additional CP4
provider, apply the expansion to the CP4-subtracted plane in (C83); (C120)
is exact as written for the local Gaussian ReLU birth and remains a lower
bound only after separately bounding the provider's linear response.

Finally define the target-blind local Bellman angle

\[
\gamma_\ell
=\frac{
\left\langle Z_\ell,
\Pi_{\mathsf P_\ell^\perp}\mathsf B_\ell^Z\right\rangle_F
}{
\|Z_\ell\|_F
\left\|\Pi_{\mathsf P_\ell^\perp}\mathsf B_\ell^Z\right\|_F
}.
\tag{C121}
\]

Then (C86) is exactly the product of the two norms in (C121) and
\(\gamma_\ell\).  Genericity proves only \(\gamma_\ell\ne0\) off an algebraic
zero set; it supplies no uniform lower bound.  Consequently (C120) quantifies
the source but cannot by itself establish the endpoint threshold (C101).

## 2026-08-31 — checkpoint 11: self-consistent all-distinct order-two replacement

This checkpoint combines the exact all-distinct H3 birth circuit with the
transient-Z22 completion.  It keeps six questions separate: source identity,
deploy ownership, Bellman value, arithmetic/reuse, coefficient fidelity, and
the higher-order remainder.

### Source identity

At layer \(\ell\), define the deployed squarefree cubic source by

\[
\boxed{
(S_{\ell}^{AD})_{ijk}
=\beta_{\ell i}c_{\ell,ij}c_{\ell,ik}
+\beta_{\ell j}c_{\ell,ji}c_{\ell,jk}
+\beta_{\ell k}c_{\ell,ki}c_{\ell,kj}
}
\tag{C122}
\]

when \(i,j,k\) are pairwise distinct, and zero on every collision.  Its lawful
factor is

\[
c_{\ell,ij}=g_{\ell j}C_{\ell,ij}.
\]

Thus \(S^{AD}\) is linear in \(\beta\), quadratic in the gated covariance
factor, and contains no truth label by definition.  Any typed plane source is
obtained by a declared local producer

\[
s_\ell^\tau
=\mathcal L_\ell^\tau S_\ell^{AD},
\qquad \tau\in\{21,22,31\}.
\tag{C123}
\]

The producer \(\mathcal L^\tau\) is part of the source identity.  Omitting its
frame derivative would omit the \(\delta A\)-on-\(T\) and \(DS_2\) terms and
would misclassify reachable co-adaptation as hidden information.

### Exact state jets through order two

Turn on the source with a homotopy

\[
\theta_{\ell+1}(\varepsilon)
=F_\ell(\theta_\ell(\varepsilon))
+\varepsilon s_\ell(\theta_\ell(\varepsilon)).
\tag{C124}
\]

Use coefficient jets, not derivative jets:

\[
\theta_\ell(\varepsilon)
=\theta_{\ell0}+\varepsilon e_\ell+\varepsilon^2z_\ell
+\varepsilon^3w_\ell+O(\varepsilon^4).
\]

With \(J_\ell=DF_\ell(\theta_{\ell0})\) and zero initial perturbation, exact
coefficient matching gives

\[
\boxed{
e_{\ell+1}=J_\ell e_\ell+s_\ell,
}
\tag{C125}
\]

\[
\boxed{
z_{\ell+1}
=J_\ell z_\ell
+\frac12D^2F_\ell[e_\ell,e_\ell]
+Ds_\ell[e_\ell].
}
\tag{C126}
\]

Let the scalar endpoint have expansion

\[
y(\varepsilon)=y_0+\varepsilon A_1+\varepsilon^2A_2+O(\varepsilon^3),
\]

and define the deployed reverse costate

\[
\lambda_L=Dy(\theta_{L0}),\qquad
\lambda_\ell=J_\ell^T\lambda_{\ell+1}.
\]

Discrete Green duality then gives the exact first two Bellman coefficients

\[
\boxed{
A_1=\sum_\ell\langle\lambda_{\ell+1},s_\ell\rangle,
}
\tag{C127}
\]

\[
\boxed{
A_2
=\frac12D^2y[e_L,e_L]
+\sum_\ell\left\langle\lambda_{\ell+1},
\frac12D^2F_\ell[e_\ell,e_\ell]+Ds_\ell[e_\ell]
\right\rangle.
}
\tag{C128}
\]

These formulas require one forward first tangent and one reverse costate, but
no forward second-tangent state.

The source/query form makes co-adaptation ownership even more explicit.
Absorb each local injection map into a source query \(Q_\ell(\varepsilon)\).
If \(\dot Q_\ell=\partial_\varepsilon Q_\ell|_0\), then

\[
\boxed{
2A_2
=\sum_\ell\left[
\langle\dot Q_\ell,S_\ell^{AD}\rangle
+\langle Q_{\ell0},DS_\ell^{AD}[e_\ell]\rangle
\right].
}
\tag{C129}
\]

The first term is query motion and the second is source motion.  Neither may
be dropped.  For a direct-state source the query derivative is generated by

\[
\dot\lambda_L=D^2y[e_L,\cdot],
\]

\[
\dot\lambda_\ell
=J_\ell^T\dot\lambda_{\ell+1}
+\left(D^2F_\ell[e_\ell,\cdot]+Ds_\ell\right)^T
\lambda_{\ell+1}.
\tag{C130}
\]

Equations (C128) and (C129) are identical coordinates for the same scalar;
they are never additive evidence.

### Where the transient Z22 closure enters

Define the local second-order forcing

\[
r_\ell^{[2]}
=\frac12D^2F_\ell[e_\ell,e_\ell]+Ds_\ell[e_\ell].
\tag{C131}
\]

After subtracting the complete old CP4 and pairing contributions, its H4
partition is

\[
r_\ell^{[2]}
=r_{\ell,\mathrm{old}}^{[2]}
+E_{22}(Z_\ell^{[2]})
+E_{211}(L_\ell^{[2]})
+E_{1111}(M_\ell^{[2]}).
\tag{C132}
\]

Here \(E_{22}(Z)\) is the fully symmetric tensor with
\((E_{22}Z)_{iijj}=Z_{ij}\); \(E_{211}\) has entries \(iijk\) with three
distinct indices; and \(E_{1111}\) has four distinct indices.  The transient
plane \(Z_\ell^{[2]}\) is exactly (C83) applied to the coefficient forcing
(C131).

For the next dense reset, put

\[
X_\ell=W_\ell\odot W_\ell,\quad
R_\ell=W_\ell(Q_{\ell+1}^{31})^T,\quad
Y_\ell=W_\ell\odot R_\ell.
\]

The complete contribution of this second-order 22 coordinate is

\[
\boxed{
A_{2,Z}
=\sum_{\ell\in{\cal D}_{10}}
3\langle Z_\ell^{[2]}X_\ell,Y_\ell\rangle_F.
}
\tag{C133}
\]

There is no extra factor \(1/2\): it is already present in the definition
(C131) of the coefficient source.  Equation (C133) is a scalar-adjoint
replacement for the \(E_{22}(Z)\) component of the forward second tangent.
Adding both would double count it.

The remaining \(211/1111\) terms in (C132) are order-two omissions, not part
of the order-three Taylor tail.  Unless their pulled Bellman action is bounded
by (C107) or acquired separately, the phrase “exact through order two” is
false.  Likewise, \(A_{2,Z}\) is new relative to the old arena only through
its Schur residual (C90); source-sector novelty alone is insufficient.

### Exact frozen-replay provenance receipt

The certified 18.7% replay is linear in its frozen per-layer typed supplies.
Let

\[
A_{\rm cert}
=\sum_{\ell,\tau}
\langle Q_{\ell0}^{\tau,D},dK_{\ell,\rm cert}^{\tau}\rangle.
\tag{C134}
\]

The all-distinct first coefficient reproduces that replay if and only if its
task-weighted residual vanishes:

\[
\boxed{
E_{\rm frozen}
=\sum_{\ell,\tau}
\left\langle Q_{\ell0}^{\tau,D},
dK_{\ell,\rm cert}^{\tau}
-\mathcal L_\ell^\tau S_\ell^{AD}(\theta_\ell^D)
\right\rangle
=0.
}
\tag{C135}
\]

An allowed approximate replacement needs a corpus-norm bound on
\(E_{\rm frozen}\), not merely a favorable mean.  If the certified costate is
also different, the exact first-coefficient discrepancy is

\[
\begin{aligned}
A_1^*-A_1^D
=\sum_\ell\{&
\langle Q_{\ell0}^D,\Delta S_{\ell0}\rangle
+\langle\Delta Q_{\ell0},S_{\ell0}^D\rangle\\
&+\langle\Delta Q_{\ell0},\Delta S_{\ell0}\rangle\}.
\end{aligned}
\tag{C136}
\]

The quadratic cross term in (C136) cannot be assigned to either source or
query alone.

For a desired self-consistent reference path, define
\(S_1=DS[e]\).  Its exact second-coefficient provenance receipt is

\[
\boxed{
\begin{aligned}
2E_{\rm prov,2}
=\sum_\ell\{&
\langle Q_{\ell1}^*,S_{\ell0}^*\rangle
+\langle Q_{\ell0}^*,S_{\ell1}^*\rangle\\
&-\langle Q_{\ell1}^D,S_{\ell0}^D\rangle
-\langle Q_{\ell0}^D,S_{\ell1}^D\rangle
\}.
\end{aligned}
}
\tag{C137}
\]

Equations (C135)--(C137) are the narrowest provenance gates.  Full tensor
equality is unnecessary, but every endpoint-visible truth-fed component must
appear in these receipts.  In particular, banked truth-minus-deploy planes
are not made deployable merely by transporting their adjoint cheaply.

### Exact coefficient-fidelity gate

Work in the corpus Hilbert space after projecting out the common old arena.
Let \(V\) be the ideal variance-channel correction, let
\(\mathcal U_{\le2}\) be the deployed order-one/two feature span, and write

\[
U=\Pi_{\mathcal U_{\le2}}V,\qquad
q_2^2=\frac{\|U\|^2}{\|V\|^2}.
\tag{C138}
\]

If the physically frozen implementation is \(\widehat U\), define its
coefficient fidelity exactly by

\[
\boxed{
\phi
=1-\frac{\|\widehat U-U\|^2}{\|U\|^2}.
}
\tag{C139}
\]

Then the recovered variance energy is exactly

\[
a_v=\phi q_2^2.
\tag{C140}
\]

For one scalar coefficient ratio
\(\gamma=\widehat\gamma/\gamma_*\),
\(\phi=1-(\gamma-1)^2\).

Let

\[
a_{\rm req}
=1-\frac{.07453-R_{\rm other}}{1.7316},
\qquad
R_{\rm other}=.4370(1-a_3)+.10313(1-a_4).
\tag{C141}
\]

The coefficient gate at a certified endpoint alignment \(q_2\) is therefore

\[
\boxed{
\phi\ge\frac{a_{\rm req}}{q_2^2}.
}
\tag{C142}
\]

It is impossible if \(q_2^2<a_{\rm req}\).  With exact other channels and
\(q_2=1\), (C142) gives

\[
\phi\ge.95696,\qquad
|\gamma-1|\le\sqrt{1-.95696}=.20746.
\tag{C143}
\]

The following finite-perturbation bound turns local coefficient provenance
into a lawful lower bound on (C139).  If
\(\widehat\beta_i=\beta_i+\delta\beta_i\) and
\(\widehat c_i=c_i+\delta c_i\), then

\[
\|\widehat S^{AD}-S^{AD}\|_F
\le3\left(\sum_i b_i^2\right)^{1/2},
\tag{C144}
\]

where

\[
b_i
=|\delta\beta_i|(\|c_i\|+\|\delta c_i\|)^2
+|\beta_i|\left(
2\|c_i\|\|\delta c_i\|+\|\delta c_i\|^2
\right).
\tag{C145}
\]

This follows by expanding
\((\beta+\delta\beta)(c+\delta c)^{\otimes2}
-\beta c^{\otimes2}\), using the orthogonality of the first-mode \(e_i\), and
then the three nonexpansive cyclic projections.

The corresponding exact finite perturbation of the Z22 action is

\[
\delta s_Z
=\langle\mathsf B_Z,\delta Z\rangle
+\langle\delta\mathsf B_Z,Z\rangle
+\langle\delta\mathsf B_Z,\delta Z\rangle,
\tag{C146}
\]

so

\[
|\delta s_Z|
\le
\|\mathsf B_Z\|_F\|\delta Z\|_F
+\|\delta\mathsf B_Z\|_F\|Z\|_F
+\|\delta\mathsf B_Z\|_F\|\delta Z\|_F.
\tag{C147}
\]

Summing the pulled versions of (C144) and (C147), together with derivative
coefficient errors, gives an endpoint bound \(E_{\rm coef}\).  Therefore

\[
\phi\ge1-\left(\frac{E_{\rm coef}}{\|U\|}\right)^2.
\tag{C148}
\]

Truth-only source error belongs to the provenance radius below, not to
\(E_{\rm coef}\); keeping these two budgets separate prevents a coefficient
refit from laundering unavailable information.

### Exact third jet and a closed third-order remainder bound

The coefficient third tangent is

\[
\boxed{
\begin{aligned}
w_{\ell+1}
=J_\ell w_\ell
&+D^2F_\ell[e_\ell,z_\ell]
+\frac16D^3F_\ell[e_\ell,e_\ell,e_\ell]\\
&+Ds_\ell[z_\ell]
+\frac12D^2s_\ell[e_\ell,e_\ell].
\end{aligned}
}
\tag{C149}
\]

For a rigorous remainder on the whole interval, put

\[
G_\ell(\theta,t)=F_\ell(\theta)+t\,s_\ell(\theta),
\]

and let \(e_\ell(t)=\partial_t\theta_\ell(t)\),
\(h_\ell(t)=\partial_t^2\theta_\ell(t)\).  The exact derivative-level third
recurrence is

\[
\begin{aligned}
\partial_t^3\theta_{\ell+1}
=D_\theta G_\ell\,\partial_t^3\theta_\ell
&+3(D^2F_\ell+tD^2s_\ell)[e_\ell,h_\ell]\\
&+(D^3F_\ell+tD^3s_\ell)[e_\ell,e_\ell,e_\ell]\\
&+3Ds_\ell[h_\ell]+3D^2s_\ell[e_\ell,e_\ell].
\end{aligned}
\tag{C150}
\]

Let \(E_\ell=\sup_t\|e_\ell(t)\|\),
\(H_\ell=\sup_t\|h_\ell(t)\|\),
\(\Lambda_\ell=\sup_t\|\lambda_\ell(t)\|\).  Let
\(F_{r,\ell}\) and \(S_{r,\ell}\) bound the operator norms of the \(r\)-th
derivatives of \(F_\ell\) and \(s_\ell\) on the homotopy, and let \(Y_2,Y_3\)
bound the endpoint derivatives.  Adjoint elimination of the last state in
(C150) yields

\[
\begin{aligned}
M_3:={}&Y_3E_L^3+3Y_2E_LH_L\\
&+\sum_\ell\Lambda_{\ell+1}\Big[
3(F_{2,\ell}+S_{2,\ell})E_\ell H_\ell
+(F_{3,\ell}+S_{3,\ell})E_\ell^3\\
&\hspace{40mm}
+3S_{1,\ell}H_\ell+3S_{2,\ell}E_\ell^2
\Big],
\end{aligned}
\tag{C151}
\]

and hence the closed bound

\[
\boxed{
|R_3|
=\left|y(1)-y_0-A_1-A_2\right|
\le\frac16M_3.
}
\tag{C152}
\]

For the all-distinct source itself, a general one-dimensional frame path has

\[
\|D^3S^{AD}\|_F
\le3\left(\sum_i(a_i^{(3)})^2\right)^{1/2},
\tag{C153}
\]

with

\[
\begin{aligned}
a_i^{(3)}={}&
|\beta_i'''|\|c_i\|^2
+6|\beta_i''|\|c_i\|\|c_i'\|\\
&+6|\beta_i'|
\left(\|c_i\|\|c_i''\|+\|c_i'\|^2\right)\\
&+|\beta_i|
\left(2\|c_i\|\|c_i'''\|
+6\|c_i'\|\|c_i''\|\right).
\end{aligned}
\tag{C154}
\]

Along an affine path in \((\beta,c)\), this reduces to

\[
\|D^3S^{AD}\|_F
\le18\left(\sum_i|\beta_i'|^2\|c_i'\|^4\right)^{1/2}.
\tag{C155}
\]

Thus the source part of \(S_{3,\ell}\) in (C151) is explicit and tensor-free.
The remaining quantities are gate-derivative and tangent/costate norm
receipts; (C152) is not numerically certified until they are bounded.

### Unified fidelity, provenance, and remainder gate

Normalize all unrepresented endpoint actions by \(\|V\|\).  Let

\[
\eta_{\rm prov}
=\frac{\|E_{\rm frozen}+E_{\rm prov,2}\|}{\|V\|},
\quad
\eta_{2111}
=\frac{\|A_{2,211}+A_{2,1111}\|}{\|V\|},
\quad
\eta_3=\frac{M_3}{6\|V\|}.
\tag{C156}
\]

Using the lawful triangle bound

\[
\rho=\eta_{\rm prov}+\eta_{2111}+\eta_3,
\]

the endpoint span satisfies \(q_2^2\ge1-\rho^2\) whenever the represented
orders zero through two are the coefficient-one Bellman projection.  The
single sufficient raw-target gate is therefore

\[
\boxed{
\phi\left[
1-(\eta_{\rm prov}+\eta_{2111}+\eta_3)^2
\right]
\ge a_{\rm req}.
}
\tag{C157}
\]

Equivalently,

\[
\eta_{\rm prov}+\eta_{2111}+\eta_3
\le
\sqrt{1-\frac{a_{\rm req}}{\phi}}.
\tag{C158}
\]

At exact coefficient fidelity and exact other channels, the total normalized
radius allowed by (C158) is \(0.20746\).  If a pre-Taylor source alignment
\(q_s=.78\) is used and \(\eta_3=C_3(1-q_s^2)^{3/2}=.24506C_3\), then

\[
.24506C_3+\eta_{\rm prov}+\eta_{2111}\le.20746.
\tag{C159}
\]

This recovers \(C_3\le.8466\) only when both lower-order remainder terms and
coefficient error vanish.  It is a fail-fast inequality, not a forecast.

A Padé/Dyson spelling does not create a second value term here.  It may
replace the Taylor tail in \(\eta_3\) only if it matches (C127)--(C128),
obeys the signed counterterm and pole-stability gates, and supplies a smaller
certified norm for the same remainder.  A positive-semidefinite fish
resummation cannot be credited as suppression, and a resummed prediction may
not be added to \(A_1+A_2\).

### Replacement, duplication, and conditional new value

The ownership decision is now exact.

1. If (C135) passes, \(A_1=A_{\rm cert}\).  The all-distinct adjoint is an
   implementation replacement for the certified frozen 18.7% replay.  The
   two values must never be added.
2. If (C135) fails, the common task-weighted projection is still duplicate,
   while the residual is unavailable truth/provenance error unless another
   deployed source supplies it.  Cheap transport does not repair that failure.
3. \(A_2\) is absent from a frozen-Jacobian replay by definition, but it is
   not automatically new relative to D11/F2J.  Its admissible value is only

   \[
   A_{2,\rm new}
   =(I-P_{\mathrm{old},A_1})A_2.
   \tag{C160}
   \]

   The Z22 part uses the stricter full-old Schur test (C90).
4. Equation (C133) and a forward \(E_{22}(Z)\) second tangent are two
   coordinates for the same value.  The same is true of the coupled-state and
   eliminated-adjoint spellings.  Each pair is mutually exclusive.

Therefore the certified \(5.50\times10^{-9}\)-scale first-order effect is a
ceiling for the replacement component, not for the Schur-new second-order
component.  Conversely, no second-order gain may be asserted from support or
Taylor algebra without the target covariance of (C160).

### Cost and reuse ledger

Keep arithmetic conditional on operand identity:

- the synchronized reused-H reverse circuit costs \(22.83\) equivalents;
- ten exact all-distinct \(C_\ell M_\ell\) birth products cost \(10\);
- a nonreused first forward tangent costs \(C_e=28\);
- the ten-layer Z22 action costs \(C_Z=10\) if
  \(R_\ell=W_\ell(Q_{\ell+1}^{31})^T\) is resident, \(20\) if only \(Q\) is
  resident, and more if the future mask must be rebuilt;
- gate Hessians and \(DS[e]\) add no dense product after the proved
  reverse/source reuse;
- exact 211/1111 acquisition is not included.

Thus the strongest standalone order-two ledger with the existing H receipt is

\[
\boxed{
C_{\le2}=22.83+10+28+C_Z
=70.83\ \hbox{or}\ 80.83.
}
\tag{C161}
\]

Without H reuse, add \(20\).  Under the literal base-48/cap-49 discipline this
path is closed.  Under replacement economics, the reported full frozen
18.7% effect has an allowance of about \(81\) products, so the \(70.83\)
branch is conditionally feasible and the \(80.83\) branch has essentially no
slack.  Any hidden-sector acquisition or mask rebuild closes the latter at
that retained value.  A larger genuinely Schur-new order-two gain changes the
adjusted allowance, but it must be proved before being used as cost credit.

The cost ledger is a replacement package: \(A_1\) replaces its D14/frozen
carrier, while (C133) occupies the D10 portfolio slot.  The latter is not a
claim that old D10 already carries \(Z_{22}\); removing D10 loses its old
conditional value.  Hence packaging additionally requires

\[
\boxed{
\Delta R(A_1+A_{2,\rm new})
-\Delta R_{D10,\rm lost}
\ge\Delta R_{\rm package\ requirement}.
}
\tag{C162}
\]

Stacking the same forward carrier at base 48 would double bill its work and
value, while ignoring the subtraction in (C162) would overstate replacement
economics.

The frozen replay also cannot reach the raw goal by itself.  At
\(R_0=2.9409\times10^{-8}\),

\[
\Delta R_{\rm cert}
=.187R_0=5.499483\times10^{-9},
\qquad
R_0-\Delta R_{\rm cert}=2.3909517\times10^{-8}.
\]

Therefore the Schur-new order-two mechanism and/or independent channels must
supply at least

\[
\boxed{
2.3909517\times10^{-8}-10^{-8}
=1.3909517\times10^{-8}
}
\tag{C163}
\]

beyond the replacement value.  This is why a successful provenance receipt
for (C135) is necessary but nowhere near sufficient for raw \(<10^{-8}\).

Finally, the derivative bound (C151) presumes a regular homotopy.  A mandatory
analytic gate is

\[
\inf_{t,\ell,i}v_{\ell i}(t)>0,\qquad
\sup_{t,\ell,i\ne j}|\rho_{\ell,ij}(t)|<1,
\tag{C164}
\]

with no variance-floor branch switch.  ReLU Gaussian moment derivatives
become singular at degenerate covariance boundaries; if (C164) fails, the
finite constants in (C151) have not been established and the order-two
Taylor certificate is invalid.

### Checkpoint-11 graph-node split

1. **AD11-SOURCE-IDENTITY — CLOSED/DERIVED.**  Equations (C122)--(C123).
   Source degree, collision convention, and producer ownership are explicit.
2. **AD11-DEPLOY-OWNERSHIP — OPEN/HIGHEST-PRIORITY.**  Equations
   (C134)--(C137).  Pass only with task-weighted deployed provenance; a
   truth-fed residual is terminal for the claimed replacement value.
3. **AD11-ORDER12-VALUE — FORMULA CLOSED; INSTANCE OPEN.**  Equations
   (C125)--(C133), followed by the old-arena residual (C160).
4. **AD11-COST-REUSE — CONDITIONAL.**  Equation (C161).  The resident-H,
   forward-tangent, and resident-\(R\) receipts are separate premises.
5. **AD11-COEFFICIENT-FIDELITY — OPEN/FAIL-FAST.**  Equations
   (C138)--(C148).  Even a perfect endpoint span fails below
   \(\phi=.95696\) when the other channels are exact.
6. **AD11-THIRD-REMAINDER — BOUND CLOSED; MAGNITUDE OPEN.**  Equations
   (C149)--(C159).  The 211/1111 order-two omission remains a separate term.
7. **AD11-FROZEN-REPLAY-OWNERSHIP — CLOSED THEOREM.**  A passing first
   coefficient replaces, never stacks with, the 18.7% replay.  Only the
   Schur-residual second coefficient can be new.

The next decisive receipt is (C135) on the actual certified supplies.  If it
passes, bound the three radii in (C156) and grade (C157).  If it fails, stop
calling the all-distinct path a carrier for the certified 18.7% mechanism and
retain it only as a separately valued, deploy-source feature.

### Actual certified-supply provenance audit

A read-only audit of the certified replay fixes the present status more
strongly.  The replay implementation in
public_g19_attribution_v1/run_chain_value_map.py loads
mps_truth_rank, tensor_planes_rank, and dvhat_rank artifacts.  The explicit
plane audit in run_t0_chain_check.py forms

\[
dK21_\ell
=K21_{\ell}^{\rm tensor\ plane}
-K21_{\ell}^{\rm host},
\]

\[
dK22_\ell
=K22_{\ell}^{\rm tensor\ plane}
-K22_{\ell}^{\rm host},\qquad
dK31_\ell
=K31_{\ell}^{\rm tensor\ plane}
-K31_{\ell}^{\rm host}.
\tag{C165}
\]

Those are offline truth-minus-deploy supplies.  Therefore:

1. an adjoint fed the same banked planes exactly replaces/reparameterizes the
   frozen replay, but it is not a deployable estimator;
2. the label-free source (C122) is deployable in principle, but the existing
   18.7% receipt is not its value receipt; and
3. until (C135) is proved for \(S^{AD}(\beta^D,C^D)\), assigning the 18.7%
   value to the deploy-source path is invalid.

Accordingly, node **AD11-DEPLOY-OWNERSHIP** has two branches:

- **truth-fed replay branch — CLOSED NONDEPLOYABLE / DUPLICATE;**
- **deploy-source branch — OPEN VALUE-UNVERIFIED.**

This provenance finding does not close the independent self-consistent
feature.  It closes only the attempted inheritance of the certified replay's
value.

The replay value also uses leave-one-network-out channel coefficients fitted
on the other panel networks.  That is a legitimate cross-fit estimate, but it
is not an exact receipt for one packaged frozen coefficient vector.  The
shift from the LOO costates \(Q_{\ell0}^{D,\mathrm{LOO}}\) to the packaged
costate belongs in \(\Delta Q\) in (C136) and in the fidelity loss (C139).

## 2026-08-31 — checkpoint 11B: AQ-law residual versus endpoint summaries

The direct H4 score lane uses a true preactivation represented as
\(X_\ell=g_\ell+e_\ell\), where \(g_\ell\) is the declared affine--quadratic
law.  Put

\[
U_\ell
=K_4((g_\ell+e_\ell)_+)-K_4((g_\ell)_+).
\tag{C166}
\]

ReLU Lipschitzness gives the lawful local bound

\[
|\langle U_\ell,H_\ell\rangle|
\le\sqrt{24}\,\|e_\ell\|_{L^2}\|H_\ell\|_{\rm HS}.
\tag{C167}
\]

The question is whether the endpoint radii in (C156) or the banked
truth-minus-host planes identify the right side of (C167).  They do not,
absent an additional observability theorem.

### Exact collision partition and reference split

Every symmetric fourth-cumulant error has the orthogonal partition

\[
U_\ell
=E_{40}(u_{\ell,40})
+E_{31}(u_{\ell,31})
+E_{22}(u_{\ell,22})
+E_{211}(u_{\ell,211})
+E_{1111}(u_{\ell,1111}).
\tag{C168}
\]

For a symmetric H4 covector, the full contraction is exactly

\[
\begin{aligned}
\langle U_\ell,H_\ell\rangle
={}&\sum_i u_{40,i}H_{iiii}
+4\sum_{i\ne j}u_{31,ij}H_{iiij}\\
&+6\sum_{i<j}u_{22,ij}H_{iijj}\\
&+12\sum_{\substack{i;\,j<k\\i,j,k\ {\rm distinct}}}
u_{211,i;jk}H_{iijk}\\
&+24\sum_{i<j<k<m}u_{1111,ijkm}H_{ijkm}.
\end{aligned}
\tag{C169}
\]

The factors are permutation multiplicities.  A \(dK21\) plane is an H3
quantity and does not enter the local H4 contraction (C169); it can affect a
later score only through the dynamical provenance map.

There is also a mandatory reference distinction.  With \(T,H,AQ\) denoting
truth, host, and the affine--quadratic law,

\[
\boxed{
K_4^T-K_4^{AQ}
=(K_4^T-K_4^H)+(K_4^H-K_4^{AQ}).
}
\tag{C170}
\]

The banked truth-minus-host \(dK22,dK31,dK4\) planes observe collision
projections of the first bracket only.  They are projections of \(U_\ell\)
only after proving \(K_4^H=K_4^{AQ}\) on the relevant sectors.  The second
bracket is an AQ-reference defect and is not supplied by the certified plane
files.

### What the three eta summaries actually constrain

The quantities in (C156) are endpoint images:

\[
\eta_{\rm prov}\|V\|
=\|T_{\rm prov}(\text{typed source/query defects})\|,
\]

\[
\eta_{2111}\|V\|
=\left\|
\sum_\ell\langle
\mathcal H_{\ell,2111}^{\rm candidate},
E_{211}(L_\ell^{[2]})+E_{1111}(M_\ell^{[2]})
\rangle
\right\|,
\]

\[
\eta_3\|V\|\ge\|R_3\|.
\tag{C171}
\]

They are not layerwise random-law residual norms.  In particular:

1. \(\eta_{\rm prov}\) uses the declared typed producer and its candidate
   endpoint query, not the complete direct-H4 AQ-reference error in (C170).
2. \(\eta_{2111}\) is a coherent endpoint action of the order-two hidden
   sectors.  It controls the direct D10 hidden action only if its layer
   weights and H4 masks are exactly the same.
3. \(\eta_3\) is a coherent higher-order endpoint remainder, not an upper
   bound on any individual \(\|e_\ell\|_{L^2}\).

Even when the masks match, a coherent sum does not control the absolute
layer sum used in (C167).  Two nonzero layer actions can have equal magnitude
and opposite sign, making all three endpoint summaries zero while
\(\sum_\ell|\beta_\ell||\langle U_\ell,H_\ell\rangle|\) remains positive.
The scale of such a cancelling pair is unrestricted by the summaries.

There is an even more basic coupling obstruction: two random variables can
have the same law and hence identical cumulants and endpoint summaries while
being coupled differently to the AQ representative.  For a centered
Gaussian, \(X=g\) and \(X=-g\) have the same law, but the chosen residuals are
\(e=0\) and \(e=-2g\).  Thus \(\|e\|_{L^2}\) is not a moment observable until
the AQ approximation specifies a canonical coupling or orthogonal
projection.

### Exact identifiability theorem

At the linear moment-state level, collect all layer H4 errors in
\(\mathcal U=\bigoplus_{\ell=5}^{14}\mathcal H_{4,\ell}\).  Let
\(\mathcal A:\mathcal U\to\mathcal Y\) contain every actually available
signed observation: the typed plane projections, provenance contractions,
and any eta actions whose masks and signs are retained.  Let

\[
\mathcal L_H(U)
=\sum_{\ell=5}^{14}\beta_\ell
\langle U_\ell,H_\ell\rangle
\tag{C172}
\]

be the coherent D10 score.  Then the available summaries identify
\(\mathcal L_H(U)\) for every admissible \(U\) if and only if

\[
\boxed{
\mathcal L_H\in\operatorname{ran}\mathcal A^*.
}
\tag{C173}
\]

Equivalently, \(\ker\mathcal A\subseteq\ker\mathcal L_H\).  If this holds,
there is a coefficient \(c\) with
\(\mathcal L_H=c^*\mathcal A\), giving

\[
|\mathcal L_H(U)|\le\|c\|\,\|\mathcal A U\|.
\tag{C174}
\]

If it fails, choose
\(U_0\in\ker\mathcal A\) with
\(\mathcal L_H(U_0)\ne0\).  Scaling \(U_0\) proves that no finite bound on the
D10 score can be a function only of the available summaries.  The large
211/1111 fibers and cross-layer cancellation make this the generic case;
no lower-frame/coercivity receipt for \(\mathcal A\) is currently present.

The same conclusion follows for the random residual.  Bounds of the form
\(\|\mathcal A e\|\le C\|e\|\) go in the wrong direction.  An upper bound on
\(\|e\|\) from observations requires a lower frame inequality

\[
\|\mathcal A e\|\ge\sigma\|e\|,
\qquad\sigma>0,
\tag{C175}
\]

on the admissible residual class.  The truth-minus-host plane map has a
nontrivial higher-chaos/all-distinct kernel, so (C175) is false without an
additional model restriction.

### Lawful endpoint-weighted D10 bounds and the minimal new observable

If canonical layer residual norms are supplied, (C167) gives

\[
\boxed{
|E_{\rm AQ}^{D10}|
\le\sqrt{24}
\sum_{\ell=5}^{14}
|\beta_\ell|\,\|H_\ell\|_{\rm HS}\,\|e_\ell\|_{L^2}.
}
\tag{C176}
\]

Including quadrature and module errors, the strict endpoint-amplitude gate is

\[
\sum_{\ell=5}^{14}|\beta_\ell|
\left[
\sqrt{24}\|H_\ell\|_{\rm HS}\|e_\ell\|_{L^2}
+\epsilon_{{\rm quad},\ell}
+\epsilon_{{\rm module},\ell}
\right]
\le3.80\times10^{-6}.
\tag{C177}
\]

For the coherent scalar score, the information-theoretically minimal extra
observable is one residual action.  Let

\[
\mathcal L_\perp
=(I-P_{\operatorname{ran}\mathcal A^*})\mathcal L_H.
\]

Then

\[
\boxed{
z_{\perp,H}=\mathcal L_\perp(U)
}
\tag{C178}
\]

together with \(\mathcal A U\) determines the complete coherent D10 score,
and no smaller scalar interface can do so when
\(\mathcal L_\perp\ne0\).

For the robust absolute-layer certificate (C177), a signed coherent scalar is
insufficient because cancellations are forbidden.  The minimal direct
nonnegative receipt is

\[
\boxed{
\mathcal R_{{\rm AQ},H}
=\sum_{\ell=5}^{14}
|\beta_\ell|\,\|H_\ell\|_{\rm HS}\,\|e_\ell\|_{L^2}.
}
\tag{C179}
\]

Equivalently, one may expose the ten layerwise signed residual actions and
take their absolute values.  The scalar (C179) is enough for the Lipschitz
certificate but requires a canonical definition of \(e_\ell\).  If direct
fourth-score queries are available, exposing
\(\langle U_\ell,H_\ell\rangle\) is sharper and avoids the coupling ambiguity
of \(\|e_\ell\|\).

One conditional endpoint relation can be stated exactly.  Let
\(P_H^{T-H}\) be (C169) contracted with every available truth-minus-host
40/31/22 plane, let \(R_{2111}^{T-H}\) be the missing 211/1111 contraction,
and put

\[
R_{H-AQ}
=\sum_{\ell=5}^{14}\beta_\ell
\langle K_{4,\ell}^{H}-K_{4,\ell}^{AQ},H_\ell\rangle.
\]

Then

\[
\boxed{
E_{\rm AQ}^{D10}
=P_H^{T-H}+R_{2111}^{T-H}+R_{H-AQ}.
}
\tag{C180}
\]

If, and only if, the hidden tensor source, masks, and layer weights defining
\(\eta_{2111}\) equal those in (C180), it gives

\[
\|R_{2111}^{T-H}\|\le\eta_{2111}\|V\|.
\tag{C181}
\]

There is no corresponding implication from
\(\eta_{\rm prov}\) or \(\eta_3\) to \(R_{H-AQ}\) without an ownership
identity placing the AQ-reference defect inside those exact endpoint maps.
Thus the strongest lawful bound from the current matching information is

\[
\|E_{\rm AQ}^{D10}\|
\le\|P_H^{T-H}\|+\eta_{2111}\|V\|+\|R_{H-AQ}\|,
\tag{C182}
\]

and the last term is precisely the unobserved scalar (C178) after removing
any row-space component.

### AQ-residual graph split

1. **AQ11B-REFERENCE-IDENTITY — OPEN.**  Prove the host AQ law equals the
   scored AQ reference, or retain the second bracket in (C170).
2. **AQ11B-ETA-TO-LAYER-NORM — CLOSED NONIDENTIFIABLE.**  Endpoint radii and
   plane projections do not upper-bound \(\|e_\ell\|\); (C175) has no current
   lower-frame receipt.
3. **AQ11B-D10-COHERENT-IDENTIFIABILITY — CRITERION CLOSED, INSTANCE OPEN.**
   Equation (C173); test the actual direct-H4 costate against the observation
   row space.
4. **AQ11B-MINIMAL-COHERENT-OBSERVABLE — DERIVED.**  Equation (C178).
5. **AQ11B-ROBUST-LAYER-RECEIPT — OPEN.**  Acquire (C179), or the ten direct
   signed H4 residual actions, and apply (C177).

The existing \(\eta_{\rm prov},\eta_{2111},\eta_3\) remain valid endpoint
budgets for their declared path.  They cannot be silently reused as the AQ
model-error receipt for a different H4 mask family.

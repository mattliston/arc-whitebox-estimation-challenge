# Matrix-polynomial closure

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## 2026-08-31 — matrix-polynomial/Jordan reset

### Exact quadratic-chaos reset

Let

\[
X_M=z^\top Mz-\operatorname{tr}M=I_2(M),\qquad M\in\operatorname{Sym}_r.
\]

The best second-Wiener-chaos projection of \(f(X_M)=\operatorname{ReLU}(X_M)\) is \(I_2(A_f)\), where

\[
A_f=\frac12\mathbb E[f(X_M)(zz^\top-I)]
=\Pr(X_M>0)M+2M\,\mathbb E[zz^\top\delta(X_M)]\,M.
\]

Hence \(A_f\) commutes with \(M\).  For a fixed \(r\)-dimensional core it is exactly an interpolating polynomial in \(M\) of degree at most \(r-1\).  For a degree-\(k\) polynomial space

\[
\mathcal P_k(M)=\operatorname{span}\{I,M,\ldots,M^k\},
\]

the exact optimal reset is obtained by Frobenius-projecting \(A_f\) onto \(\mathcal P_k(M)\).  If this projection is \(A_k\), then

\[
q_k^2=\frac{2\|A_k\|_F^2}{\operatorname{Var}(f(X_M))},\qquad
q_{\max}^2=\frac{2\operatorname{tr}(A_f^2)}{\operatorname{Var}(f(X_M))}.
\]

No choice of powers, Chebyshev polynomials, or resolvents can exceed \(q_{\max}\); rational functions still remain in the same latent second-chaos space.

### Gaussian Hermite degree audit

For centered \(Z_+\), \(Z\sim N(0,1)\),

\[
\operatorname{Var}(Z_+)=\frac12-\frac1{2\pi}=0.340845\ldots
\]

and the nonzero centered Hermite energies begin

\[
e_1=\frac14,\quad e_2=\frac1{4\pi},\quad
e_4=\frac1{48\pi},\quad e_6=\frac1{160\pi}.
\]

The corresponding full-output correlations are approximately

- \(H_1\): \(0.856\);
- through \(H_2\): \(0.98333\);
- through \(H_4\): \(0.99317\);
- through \(H_6\): \(0.99611\).

For only the nonlinear residual beyond \(H_1\), the correlations are \(0.936\), \(0.974\), and \(0.985\) through \(H_2,H_4,H_6\).  Thus an optimistic one-reset Gaussian surrogate needs scalar degree two for \(0.925\), and degree six for \(0.983\) on the nonlinear residual.  Under a conservative multiplicative ten-reset model, end-to-end \(0.925\) needs per-layer \(0.99223\) and scalar degree four, while end-to-end \(0.983\) needs per-layer \(0.99829\) and scalar degree twelve.  These are scalar-Hermite degrees, not guarantees for matrix-power shadows.

### The H4 square-lift obstruction

The exact Wiener product identity is

\[
X_M^2-\mathbb E X_M^2
=I_4(\operatorname{sym}M\otimes M)+4I_2(M^2).
\]

Consequently centered \(M^2\) is only the second-chaos contraction shadow of the genuine fourth-chaos square lift.  Put

\[
\tau=\frac{\operatorname{tr}M^4}{(\operatorname{tr}M^2)^2}=\frac1{r_{\rm eff}}.
\]

The exact fraction of the normalized square lift lying in \(I_2(M^2)\) is

\[
\frac{4\tau}{1+6\tau}.
\]

For \(r_{\rm eff}\simeq45\) this is about eight percent; for a Wigner-like spectrum it is about fourteen percent.  Most leading nonlinear ReLU energy is therefore in the irreducible \(I_4\) tensor.  Any finite family \(I_2(P(M))\) remains in second chaos and cannot recover it.  A pure matrix-polynomial reset is therefore closed as a standalone raw-\(<10^{-8}\) route unless the exact value-weighted \(q_{\max}\) is proved above the task threshold and the discarded H4 adjoint value is bounded below the remaining error budget.

### First noncommuting invariant

For two cores,

\[
\log\mathbb E e^{itX_M+isX_N}
=\frac12\sum_{m\ge2}\frac{2^m}{m}
\operatorname{tr}(itM+isN)^m.
\]

Power-Gram features capture the mixed cyclic words through order three.  At order four a new independent word appears:

\[
\operatorname{tr}(MNMN)
=\operatorname{tr}(M^2N^2)-\frac12\|[M,N]\|_F^2.
\]

The commutator defect is exactly the irreducible H4/noncommuting obstruction.  Resolvents and Chebyshev functions of individual matrices do not restore it.

### Cost and state growth

With \(r(r+1)/2\asymp n\), \(k\) matrix shadows use \(O(kn^2)\) storage and \(O(kn^3)\) dense transport; local powers/eigendecompositions are \(O(kn^{5/2})\).  A genuine H4 state occupies \(O(n^3)\) generic storage, and H6 occupies \(O(n^4)\).  Local rank-one H4 kernels become secant mixtures after dense mixing, and the next nonlinear reset generically splits their common dictionary into the full \(\operatorname{Sym}^2(\operatorname{Sym}_r)\) space.

## 2026-08-31 — commutator operator-basis theorem

Let \(L:\mathbb R^n\to\operatorname{Sym}_r\) synthesize the cores, \(Le_i=M_i\), and define

\[
\beta_L:\Lambda^2\mathbb R^n\to\mathfrak{so}_r,
\qquad \beta_L(u\wedge v)=[Lu,Lv].
\]

Because every elementary skew matrix is a commutator of two symmetric matrices,

\[
[\operatorname{Sym}_r,\operatorname{Sym}_r]=\mathfrak{so}_r.
\]

Generic nearly spanning cores therefore give

\[
\operatorname{rank}\beta_L=q=\dim\mathfrak{so}_r=\frac{r(r-1)}2\asymp n.
\]

Under invertible dense mixing \(L'=LW\),

\[
\beta_{L'}=\beta_L\circ\Lambda^2W,
\]

so this rank is exactly invariant across linear layers.

For an orthonormal basis of \(\operatorname{Sym}_r\),

\[
\beta_L\beta_L^*=\frac{r+2}{2}I_{\mathfrak{so}_r}.
\]

The proof is the Casimir identity

\[
\|\operatorname{ad}_A:\operatorname{Sym}_r\to\operatorname{Sym}_r\|_{\rm HS}^2
=(r+2)\|A\|_F^2.
\]

Thus all \(q\) operator singular values are flat in the isotropic model.  At \(r=45\), \(q=990\); correlation \(0.925\) requires at least 848 coherent modes, and \(0.983\) requires at least 957.

In an elementary skew basis, each coefficient plane

\[
C^{ab}_{ij}=\langle A_{ab},[M_i,M_j]\rangle
\]

has rank at most \(2r-2\) and is implicitly factorized by the original \(O(n^2)\) core entries.  This does not give a low-dimensional operator basis: after the H4 operation, \(C^{ab}\odot C^{ab}\) generically has full plane rank \(n\).  The implicit factorization is useful only for instantaneous commutators, not inherited H4 amplitudes.

Polynomial reset obeys

\[
[M^a,N^b]
=\sum_{u=0}^{a-1}\sum_{v=0}^{b-1}
M^uN^v[M,N]N^{b-1-v}M^{a-1-u}.
\]

Hence the reset-invariant operator module is the bimodule generated by the cores.  If their associative algebra is \(\operatorname{Mat}_r\), Burnside forces every nonzero commutator seed to generate all of \(\mathfrak{so}_r\).  An exact small operator basis can exist only when

\[
d_{\rm NC}=\dim([\mathcal A,\mathcal A]\cap\mathfrak{so}_r)
\]

is small for the common core algebra \(\mathcal A\), or when the endpoint induces a low-dimensional value quotient.

The target-conditioned sharp criterion uses the whitened value map \(H_{\rm val}\) from operator injections to stacked endpoint residuals.  A \(k\)-mode orthogonal carrier can attain correlation \(\rho\) iff its frozen common subspace captures

\[
\frac{\sum_{j\le k}\sigma_j(H_{\rm val})^2}
{\sum_j\sigma_j(H_{\rm val})^2}\ge\rho^2.
\]

Generic isotropy reduces this exactly to \(k/q\).

The full C4 even-ring object is operator-valued.  A scalar plane resolvent is exact only under coherent low rank or an isotropic factorization \(\mathscr K=I_{\mathfrak{so}_r}\otimes B\).  If \(\mathscr K=I\otimes B+\Delta\) and both operator norms are at most \(\kappa<1\), the even/odd resolvent error is bounded by

\[
\|R(\mathscr K)-R(I\otimes B)\|
\le\frac{\|\Delta\|}{(1-\kappa)^2}.
\]

Criticality therefore amplifies even small quenched anisotropy.

## 2026-08-31 — exceptional common Jordan algebras and cost bounds

Closure under linear combinations and spectral squares makes a common core space a Euclidean Jordan algebra \(J\subseteq\operatorname{Sym}_r\), with H4 carrier \(\mathfrak g=[J,J]\subseteq\mathfrak{so}_r\).  The embeddable simple factors have dimensions

- \(H_s(\mathbb R)\): \((\dim J,\dim\mathfrak g)=(s(s+1)/2,s(s-1)/2)\);
- \(H_s(\mathbb C)\): \((s^2,s^2-1)\);
- \(H_s(\mathbb H)\): \((s(2s-1),s(2s+1))\);
- spin factor: \((m+1,m(m-1)/2)\), but its traceless cubic vanishes and all H3 terms have a scalar/star leg.

The Albert algebra is not a special matrix Jordan algebra and its derivation algebra already has dimension 52.  Universally for a special representation in \(r\) real dimensions,

\[
\dim J\le r+\dim\mathfrak g.
\]

The exact new direct-q4 sector is

\[
Q_X(Y,Z)=8\operatorname{tr}([X,Y][X,Z]).
\]

Within a real block with generic diagonal anchor \(X=\operatorname{diag}(\lambda)\),

\[
Q_X(Y,Z)=-16\sum_{a<b}(\lambda_a-\lambda_b)^2Y_{ab}Z_{ab}.
\]

Its anchored commutator-angle Gram therefore has exact rank \(\binom{s}{2}\).  For blocks \(s_b\),

\[
d_4=\sum_b\binom{s_b}{2},\qquad
d_3=\sum_b\binom{s_b+1}{2}=r+d_4.
\]

With at most 49 explicit operator channels and \(r=45\), \(d_4\le49\), \(d_3\le94\).  The isotropic correlation ceilings are only

\[
\rho_4\le\sqrt{49/990}=0.223,qquad
\rho_3\lesssim\sqrt{94/1024}=0.303.
\]

The smallest factor with genuinely non-star signed H3 is \(H_3(\mathbb R)\), with \((d_3,d_4)=(6,3)\).  \(2\times2\) blocks are spin factors and revert to the dead star family.  A low-dimensional irreducible Lie generator set does not evade the result: unless its anticommutators obey a Clifford/spin identity, spectral reset generates the full matrix algebra by Burnside.

Transporting one dense H4 plane through \(L\) layers by \(W^\top C W\) costs about \(2L\) f32-GEMM equivalents.  At \(L=10\), a total cap of 49 permits only two such planes.  Even charging only one contraction per channel per layer permits four.  Reaching the isotropic task thresholds would require 848 or 957 channels, hence at least 8,480 or 9,570 equivalents even under the optimistic one-contraction accounting.

Even a hypothetical fused block implementation cannot rescue capacity.  Exact block-local alternating products cost at least \(Ln^2\sum_bs_b^3\).  With budget 49, put \(B=49n/L\).  Cauchy gives

\[
(\sum_bs_b^2)^2\le r\sum_bs_b^3\le rB,
\]

and therefore

\[
d_4\le\frac{\sqrt{rB}-r}{2}.
\]

At \(r=45,n=1024\), this yields \(d_4\le215\) for \(L=10\) and \(d_4\le179\) for \(L=14\), with correlation ceilings about \(0.466\) and \(0.425\).  Thus the generic small-algebra route is closed.

## 2026-08-31 — value-targeted NCQC functional quotient

Let \(\mathcal K=\mathfrak{so}_r\).  A \(k\)-word carrier uses \(U=\operatorname{span}\{A_1,\ldots,A_k\}\) and coefficient planes

\[
c^a_{ij}=\langle A_a,[M_i,M_j]\rangle.
\]

Its projected q4-perp source is

\[
\widehat Q_i(j,t)=-8\sum_{a=1}^kc^a_{ij}c^a_{it},
\]

or, after joint refitting, \(-8c_{ij}^\top\Theta c_{it}\).  One word exposes one lifted quartic direction; two words expose at most the three directions in \(\operatorname{Sym}^2U\).

An exact target quotient exists iff there is \(\pi:\mathcal K\to\mathbb R^k\), \(N=\ker\pi\), such that

1. \(T_{\nu\ell}N\subseteq N\) for every relevant reset transition, so \(\pi T_{\nu\ell}=R_{\nu\ell}\pi\);
2. every future q4 adjoint annihilates \(N\odot\mathcal K\), so the endpoint functional factors through \(\operatorname{Sym}^2\pi\).

For an orthogonal word space this is the adjoint-invariance condition

\[
T_{\nu\ell}^*U\subseteq U.
\]

One word must be a common left-eigenmatrix word; two words must form one common adjoint-invariant plane.  This functional quotient, not a small forward algebra, is the only non-star exceptional route.

Let \(\mathcal H_4\) be the whitened lifted value-Hankel map and \(R_U=\Pi_{\operatorname{Sym}^2U}\).  Define

\[
\tau_U=\frac{\|\mathcal H_4(I-R_U)\|_{\rm HS}^2}{\|\mathcal H_4\|_{\rm HS}^2},
\qquad
\ell_U^2=\frac{\|\mathcal H_4R_U-\widehat{\mathcal H}_{4,U}\|_{\rm HS}^2}{\|\mathcal H_4\|_{\rm HS}^2}.
\]

The two errors have orthogonal right-input sectors, so an optimally refit reduced carrier reaches correlation \(\rho\) iff

\[
\tau_U+\ell_U^2\le1-\rho^2.
\]

The allowed totals are 0.144375 for \(\rho=0.925\) and 0.033711 for \(\rho=0.983\).

If the lifted value Gram has eigenvalues \(\mu_1\ge\mu_2\ge\cdots\), one word necessarily needs \(\mu_1/\sum\mu_j\) above the threshold, and its leading eigenvector must be a decomposable square \(A\odot A\).  Two words necessarily need \((\mu_1+\mu_2+\mu_3)/\sum\mu_j\) above the threshold, and the retained lifted sector must have a common two-dimensional base support, exactly \(\operatorname{Sym}^2U\).  A low unconstrained spectral tail without this factorability is insufficient.

For approximate adjoint invariance, put

\[
\eta=\max_{\nu,\ell}\|(I-P_U)T_{\nu\ell}^*P_U\|,
\qquad \|T_{\nu\ell}\|\le\chi.
\]

The lifted one-step leakage is at most \(2\chi\eta+\eta^2\), and over \(L\) layers

\[
\ell_U\le(2\chi\eta+\eta^2)\frac{1-\chi^{2L}}{1-\chi^2}.
\]

At \(\chi=1,L=10\) with zero spectral tail, \(\eta\) must be at most approximately 0.0188 for correlation 0.925 and 0.00914 for 0.983.

## 2026-08-31 — backward block-Krylov characterization

A scalar q4 endpoint is a quadratic form \(x^\top H_Lx\); scalar output does not imply rank-one \(H_L\).  With direct seeds \(S_\ell\),

\[
H_\ell=S_\ell+T_\ell^*H_{\ell+1}T_\ell.
\]

For a fixed finite schedule, the exact smallest base-word support is

\[
U_{\rm FH}=\sum_{\ell=0}^L\operatorname{ran}H_\ell.
\]

Factor the endpoint/direct seed supports into columns \(B_s\).  The exact finite-horizon dimension is the rank of the admissible backward suffix matrix

\[
\mathcal O_{\rm sched}=
[B_L,T_{L-1}^*B_L,T_{L-2}^*T_{L-1}^*B_L,\ldots,B_\ell,T_{\ell-1}^*B_\ell,\ldots].
\]

Equivalently, construct \(U_L=\operatorname{ran}B_L\) and

\[
U_\ell=\operatorname{ran}B_\ell+T_\ell^*U_{\ell+1},
\]

then sum the layer spaces.  For a truly common invariant word basis, start with \(U^{(0)}=U_{\rm FH}\) and iterate

\[
U^{(m+1)}=U^{(m)}+\sum_{\nu,\ell}T_{\nu\ell}^*U^{(m)}
\]

to stabilization.  The result is

\[
U_*=\operatorname{alg}\{T_{\nu\ell}^*\}\,U_{\rm FH}.
\]

It is the unique smallest common adjoint-invariant word space.  Its exact dimension is the rank of the associated block-Krylov matrix.  A common one/two-word space exists exactly when this rank is at most one/two; algebraically, all relevant two-way or three-way exterior products vanish.

A scalar linear endpoint gives a one-dimensional *layer-dependent bundle*, not a common one-dimensional space.  For one generic transition with a cyclic endpoint vector, the suffix Krylov matrix has rank \(\min(L+1,q)\).  Under an irreducible transition algebra, Burnside gives \(U_*=\mathcal K\), dimension \(q\).  A scalar quadratic endpoint may start full-rank immediately.  Thus \(k\le2\) is a measure-zero structural exception, not a consequence of scalar output.

For an approximate word space, the optimal weighted orbit tail is the singular-value tail of \(\mathcal O_{\rm sched}\).  With \(P=P_U\), transition defect \(\eta=\|(I-P)T^*P\|\), and \(\|T\|\le\chi\), telescoping a word of length \(m\) gives

\[
\|T_w^*P-PT_w^*P\|\le m\chi^{m-1}\eta.
\]

The induced quartic error is at most \(2\epsilon+\epsilon^2\).

## Current checkpoint question — layer-dependent D14 + D10 two-bundle schedule

The only surviving architecture under analysis is one layer-dependent word active through all 14 layers plus one q4-perp suffix word active only over L5--14.  This is not assumed to be a common invariant basis.  The current task is to derive the exact schedule-specific rank/wedge conditions, boundary-collapse condition at L5, leakage budget, generic Burnside no-go, and any noncommuting/non-star symmetry class that can support this two-bundle schedule.

## 2026-08-31 — D14 shared word plus D10 q4-perp suffix word

Let \(a_\ell\) denote the word active through the full depth and \(b_\ell\) the suffix word active only on L5--14.  Define the layer-dependent bundles

\[
U_\ell=\operatorname{span}\{a_\ell\}\quad(\ell<5),
\qquad
U_\ell=\operatorname{span}\{a_\ell,b_\ell\}\quad(\ell\ge5),
\]

with orthoprojectors \(P_\ell\).  This schedule does not require one common invariant basis.

For direct endpoint/q4 seed-support columns \(B_t\), form the exact backward suffix-orbit matrix at layer \(\ell\),

\[
\mathcal O_\ell=
[B_\ell,T_\ell^*B_{\ell+1},T_\ell^*T_{\ell+1}^*B_{\ell+2},\ldots].
\]

The two-bundle schedule is exact iff

\[
\operatorname{rank}\mathcal O_\ell\le1\quad(\ell<5),
\qquad
\operatorname{rank}\mathcal O_\ell\le2\quad(\ell\ge5),
\]

and every quadratic q4 adjoint at layer \(\ell\) is supported in \(\operatorname{Sym}^2U_\ell\).  Equivalently, all pair wedges of suffix columns vanish before L5, and all triple wedges vanish on L5--14.  This rank statement is both necessary and sufficient for the fixed schedule.

The L5 boundary has the extra collapse condition

\[
\operatorname{rank}(T_4^*|_{U_5})\le1.
\]

For orthonormal \(a_5,b_5\), the exact residual of the best one-word boundary is the second singular value of \(T_4^*|_{U_5}\); it must vanish.  A hard suffix birth gives the particularly clean identity \(T_4^*b_5=0\).  More generally both late words may map to the same early direction.  A generic full-rank transition on the late plane fails this gate.

For approximate bundles define

\[
\eta_\ell=\|(I-P_\ell)T_\ell^*P_{\ell+1}\|,
\qquad
\xi_\ell=\|(I-P_\ell)B_\ell\|.
\]

These are schedule-specific transition and direct-seed leakages.  If \(\|T_\ell\|\le\chi_\ell\), backward telescoping bounds the base-word error by the weighted sum of the \(\eta_\ell\) and \(\xi_\ell\).  On the q4 lift the one-step transition leakage is at most

\[
2\chi_\ell\eta_\ell+\eta_\ell^2.
\]

Writing \(R_\ell=\Pi_{\operatorname{Sym}^2U_\ell}\) and adding the direct lifted seed tail \(\|(I-R_\ell)S_\ell\|\), the accumulated implementation leakage must combine with the ideal schedule spectral tail under the same exact endpoint rule

\[
\tau_{\rm sched}+\ell_{\rm sched}^2\le1-\rho^2.
\]

The allowed totals remain 0.144375 and 0.033711 for correlations 0.925 and 0.983.

For generic independent transition/source data, the suffix-orbit rank is

\[
\operatorname{rank}\mathcal O_\ell
=\min\left(q,\sum_{t\ge\ell}\operatorname{rank}B_t\right)
\]

on a Zariski-open set: a maximal minor is a nonzero polynomial because one can explicitly choose transitions sending the seed columns to distinct basis vectors.  Thus three independent future seed directions kill the late two-bundle gate, and two kill the early one-bundle gate.  A generic scalar q4 endpoint can already have full-rank quadratic adjoint \(H_L\), so scalarity gives no protection.

If the reset-adjoint family generates \(\operatorname{End}(\mathcal K)\), Burnside implies that the ensemble orbit of any nonzero seed is all of \(\mathcal K\).  Therefore a formula required to work over the full admissible transition family cannot have a one/two-dimensional bundle.  Important qualification: one fixed deterministic chain with exactly one rank-one terminal linear seed and no direct layer seeds always has a moving one-dimensional backward bundle.  Burnside is a no-go only once the architecture must cover multiple source branches, a quadratic seed of rank greater than one, or an ensemble of transition words.  The ten late q4 injections are generically such a multi-branch source.

The deploy cost exactly explains this schedule: one two-sided dense word over D14 costs about 28 f32-GEMM equivalents and one suffix word over D10 costs about 20, totaling 48 under the 49-equivalent cap.  There is essentially no remaining dense-matmul allowance for source formation; it must be fused or ride an existing contraction.

### Exact noncommuting symmetry exception

A finite sign grading supplies a genuine non-star example.  In \(J=\operatorname{Sym}_3(\mathbb R)\), conjugation by the Klein-four diagonal sign group splits

\[
\mathfrak{so}_3
=\operatorname{span}\{A_{12}\}\oplus
\operatorname{span}\{A_{13}\}\oplus
\operatorname{span}\{A_{23}\}
\]

into three distinct one-dimensional character sectors.  If every realized reset adjoint and endpoint seed is grading-equivariant, each sector is an exact word bundle.  One may choose \(A_{12}\) for D14 and \(A_{13}\) for D10, with the latter hard-gated at L5.  The forward Jordan algebra is noncommutative, the q4-perp commutator angles are nonzero, and

\[
\operatorname{tr}(XYZ)
\supset X_{12}Y_{23}Z_{31}+\text{permutations},
\]

so the algebra does not reduce to a commuting CP/star sector.  This survives only with an exact sign-equivariance certificate for every transition and seed.  Generic dense realized cores break the grading.

### Current terminal question

Determine whether the actual q4-perp endpoint/direct seeds admit the early rank-one/late rank-two suffix wedges and the L5 rank-one boundary identity symbolically.  Absent such an exact grading, parity, or intertwining identity, generic multi-branch/Burnside growth closes the D14+D10 architecture despite its exact billing fit.

## 2026-08-31 — universal scalar K31 Bellman action

The K31 consumer is linear-affine once the base trajectory and gates are fixed.  Write its tangent-plane recurrence as

\[
E_{\ell+1}=W_{\ell+1}^\top(D_\ell\odot E_\ell+J_\ell)W_{\ell+1},
\qquad Y=\langle H_L,E_L\rangle.
\]

The exact backward Bellman masks are

\[
\Lambda_L=H_L,
\qquad
B_\ell=W_{\ell+1}\Lambda_{\ell+1}W_{\ell+1}^\top,
\qquad
\Lambda_\ell=D_\ell\odot B_\ell.
\]

The contribution of a source born at layer \(\ell\) is consumed immediately as the scalar

\[
a_\ell=\langle B_\ell,J_\ell\rangle.
\]

For the K31 channel

\[
J_\ell^{31}=\frac16\left(G31_\ell\odot dK31_\ell
+G13_\ell\odot dK31_\ell^\top\right),
\]

so the exact universal action is

\[
a_\ell^{31}=\langle R_\ell^{31},dK31_\ell\rangle,
\quad
R_\ell^{31}=\frac16\left(B_\ell\odot G31_\ell
+(B_\ell\odot G13_\ell)^\top\right).
\]

After compiling the adjoint schedule, the complete K31 effect is implemented by

\[
z_0=0,
\qquad z_{\ell+1}=z_\ell+a_\ell^{31},
\qquad Y_{31}=z_L.
\]

This streaming accumulator has minimal causal state dimension one whenever at least one layer action is nonzero.  Its temporal Hankel matrix is the rank-one Hankel matrix of an integrator.  A zero-state streaming realization cannot remember earlier nonzero actions.  This rank-one claim is target- and horizon-compiled: without the future-dependent masks \(R_\ell^{31}\), a uniform causal realization retains the original plane-system Hankel rank, generically \(O(n^2)\).

The Bellman identity eliminates the need to transport a q4/K31 carrier after its birth.  Consequently the following previous obstructions disappear as *cross-layer causal-state* obstructions:

- the common one/two-word adjoint-invariance requirement;
- the D14+D10 suffix rank and wedge tests;
- the L5 rank-collapse gate;
- Burnside expansion of the K31 consumer orbit;
- the 20-GEMM D10 q4 word transport.

They can reappear as local source-evaluation arithmetic, but not as K31 consumer-state lower bounds.

The reverse dense adjoint costs two conjugation GEMMs per traversed layer: about 28 f32 equivalents over D14 or 20 over the late D10 suffix.  The scalar accumulator itself is negligible.  Thus replacing D14+D10 transport (48) by one D14 Bellman plane (28) leaves at most 21 equivalents for exact local provider actions; if only late K31 actions need masks, the 20-equivalent suffix adjoint leaves at most 29.  Marginal cost is lower only when the exact Bellman mask is already materialized or when the source can be injected into an already-billed carried covariance before its conjugation.

### Exact local q4 action factorization

The noncommuting direct-q4 term is

\[
Q_{rst}=8\operatorname{tr}([M_r,M_s][M_r,M_t]).
\]

Let \(\Gamma_{r,st}\) be the scalar Bellman coefficients after all K31 contractions and let \(C_re_s=[M_r,M_s]\).  Only the symmetric part of \(\Gamma_r\) matters, and

\[
A_{q4}=-8\sum_r\operatorname{tr}(C_r\Gamma_rC_r^*).
\]

If

\[
\Gamma_r=\sum_{a=1}^{k_r}\lambda_{ra}u_{ra}u_{ra}^\top,
\]

then the action can be evaluated without materializing the q4 tensor:

\[
A_{q4}=-8\sum_{r,a}\lambda_{ra}
\left\|\left[M_r,\sum_su_{ra,s}M_s\right]\right\|_F^2.
\]

Thus the temporal Hankel rank is one, but the exact local provider-action rank is the rank, simultaneous factor rank, or tensor rank of the \(\Gamma_r\) slices.  A generic full-rank \(\Gamma\) can still exceed the remaining arithmetic budget.  Scalar output alone does not imply low local action rank.

More generally, if the contracted local provider is affine-quadratic in an already carried feature \(x_\ell\),

\[
a_\ell=\alpha_\ell+b_\ell^\top x_\ell+x_\ell^\top Q_\ell x_\ell,
\]

it is evaluable in place with no new dynamic state.  For a random feature with mean \(\mu\) and covariance \(\Sigma\),

\[
\mathbb E a_\ell
=\alpha_\ell+b_\ell^\top\mu
+\mu^\top Q_\ell\mu+\operatorname{tr}(Q_\ell\Sigma).
\]

This is exact only if the required first/second moments and the contraction with \(Q_\ell\) are already available or cheaply factorized.  A quartic dependence on a hidden random state still requires higher moments.

### Obstructions that remain

The Bellman action solves consumption, not supply.  It does not establish that \(dK31_\ell\), or merely its scalar action, is determined by the retained state.  The exact functional-observability condition at each layer is

\[
\ker O_{\rm retained,\ell}
\subseteq
\ker\left(R_\ell^{31}\circ dK31_\ell\right).
\]

If two hidden cumulant states have identical retained moments but different Bellman actions, no scalar accumulator can repair the provider.

For a hidden provider recursion \(h_{\ell+1}=F_\ell h_\ell+\cdots\), the consumer masks merely induce local rows \(g_\ell\); the provider adjoint obeys

\[
p_\ell=F_\ell^*p_{\ell+1}+g_\ell.
\]

Its own block-Hankel/backward orbit can still have full rank by Burnside.  The information lower bound's fixed-scalar exception applies only when this exact provider adjoint can be contracted before the hidden information is discarded.

Further remaining gates are:

- exact non-Gaussian/ReLU reset closure and inherited H4 supply;
- local \(\Gamma\)-slice factor rank and contraction cost;
- multiple endpoint heads or rows requiring multiple Bellman schedules;
- dependence of \(D,G31,G13\) on the correction itself;
- nonlinear co-adaptation.  The affine Bellman recursion is exact for arbitrary additive sources with fixed transitions, but only first-order if the proposed correction changes gates or the base trajectory.

### Current terminal question

Derive the actual K31 Bellman coefficient slices \(\Gamma_r\) symbolically and decide whether their symmetric action has a shared factorization that fits the 21--29 remaining f32-GEMM equivalents.  In parallel, prove whether the q4/H3 provider action is observable from the current retained cores before reset.  These replace the now-obsolete D14+D10 cross-layer word-rank gate.

## 2026-08-31 -- local provider factor and Jordan-circuit cost

### Exact simultaneous-factor criterion for an abstract q4 action

For the abstract commutator-angle action above, the least width of a shared
two-sided factor is exactly

\[
k_*=\dim\operatorname{span}_r\operatorname{range}(\Gamma_r)
=\operatorname{rank}[\Gamma_1\ \Gamma_2\ \cdots\ \Gamma_n].
\]

Indeed, a common \(U\in\mathbb R^{n\times k}\) and symmetric slices
\(\Theta_r\) satisfying

\[
\Gamma_r=U\Theta_rU^\top\qquad(1\le r\le n)
\]

exist if and only if every range \(\operatorname{range}(\Gamma_r)\) lies in
\(\operatorname{range}(U)\); hence the displayed concatenated unfolding rank
is both necessary and sufficient.  With mixed cores

\[
N_a=\sum_sU_{sa}M_s,
\]

the action becomes

\[
A_{q4}=-8\sum_{r,a,b}\Theta_{r,ab}
\langle[M_r,N_a],[M_r,N_b]\rangle_F.
\]

Because \(M_r,N_a\) are symmetric, \([M_r,N_a]=M_rN_a-(M_rN_a)^\top\),
so one \(r\times r\) product per pair \((r,a)\) suffices.  Separate
low-rank factorizations for each \(\Gamma_r\) are not enough: without a
shared \(U\), merely forming the anchor-dependent mixed cores costs about
one full dense GEMM per factor per layer.  Thus the relevant exact gate is
the common unfolding rank, not the maximum slice rank.

For approximation, ordinary Frobenius tail of \(\Gamma\) is not the right
certificate.  If \(P_U\) is the shared projection and \(A_\nu\) denotes the
realized commutator action on net \(\nu\), the required value-weighted tail is

\[
\epsilon_U^2=
\frac{\mathbb E_\nu|A_\nu(\Gamma)-
A_\nu(P_U\Gamma P_U)|^2}
{\mathbb E_\nu|A_\nu(\Gamma)|^2}.
\]

After including all other leakage terms, this must fit within \(1-0.9253^2
=0.14381991\) or \(1-0.98346^2=0.03280643\).  Equivalently, one must take a
weighted SVD in the seminorm induced by the realized commutator maps
\(C_r:u\mapsto[M_r,\sum_su_sM_s]\), not an unweighted SVD of the coefficient
tensor.

### The actual K31 action has a stronger arithmetic-circuit factorization

The contracted fourth-cumulant identity is

\[
\kappa_4(X(B),X(B),X(B),X(D))=48\operatorname{tr}(B^3D).
\]

Flatten the retained symmetric cores as
\(A^\flat\in\mathbb R^{n\times d}\), \(d=r(r+1)/2\), and define

\[
B_a=\sum_iW_{ia}A_i,
\qquad
D_a=\sum_i(WQ31^\top)_{ia}A_i.
\]

Associativity eliminates the nominal dense intermediate
\(P=WQ31^\top\):

\[
D_a=\sum_jQ31_{aj}B_j.
\]

Therefore the exact tensor-free scalar provider action is

\[
\boxed{\mathcal B_{31}=48\sum_{a=1}^n\operatorname{tr}(B_a^3D_a)},
\qquad B=W^\top A,\quad D=Q31\,B.
\]

This is more informative than the abstract \(\Gamma\)-rank test.  Its
coefficient-tensor unfoldings can be full rank, yet its arithmetic circuit
uses only two shared rectangular transforms and pointwise small-matrix
Jordan operations.  Consequently, full \(\Gamma\) rank is not a no-go for
this particular K31 action.

The minimal in-place Jordan object is explicit.  Since \(B,D\) are
symmetric,

\[
\operatorname{tr}(B^3D)
=\langle B^2,B\circ D\rangle_F,
\qquad B\circ D=\tfrac12(BD+DB).
\]

Thus, if the reset already materializes \(B^2\), exactly one additional
\(r\times r\) product per core forms \(B\circ D\).  If \(B^2\) is absent,
two products are sufficient and, for generic noncommuting \(B,D\), one
cannot suffice in a bilinear straight-line circuit because the desired
scalar has bidegree \((3,1)\).

For \(n=1024,r=55\),

\[
d=1540,\qquad d/n=1.50390625,
\qquad s_r=r^3/n^2\simeq0.15867.
\]

A ten-layer retained-core chain with one rectangular core mix and one
small-matrix product bank per layer costs

\[
10(d/n+s_r)\simeq16.626.
\]

The fused endpoint \(D=Q31B\) and one Jordan product bank cost
\(d/n+s_r\simeq1.663\).  Hence the exact optimistic chain plus scalar K31
action costs

\[
\boxed{11(d/n+s_r)\simeq18.288<21},
\]

leaving only about \(2.712\) full-GEMM equivalents.  More generally, if a
reset uses \(c_R\) small-matrix product banks per layer, the endpoint uses
\(c_P\), and \(g\) extra rectangular acquisitions are required,

\[
\mathcal C=10(d/n+c_Rs_r)+d/n+c_Ps_r+g(d/n)\le21.
\]

At \(g=0,c_P=1\), this requires \(c_R\lesssim2.71\).  Three extra context
acquisitions already cost \(3d/n\simeq4.512\) and therefore do not fit;
multiple contexts are viable only if they are free natural-layer
observations with already-carried fourth legs.

### Exact deployment gate and graph split

The local circuit is deployable only after the correct NCQC representative
is determined from retained information.  If \(F_A\) is the hidden inverse
fiber, \(M:F_A\to\mathbb R^s\) is the acquired-context map, and
\(\ell=D\mathcal B_{31}|_{F_A}\), then the sharp scalar observability gate is

\[
\ker M\subseteq\ker\ell
\quad\Longleftrightarrow\quad
\ell\in\operatorname{row}M.
\]

With

\[
\tau_{31}=\|(M^\dagger)^*\ell\|^{-1},
\]

acquisition error \(\eta\) gives
\(|\delta\mathcal B_{31}|\le\eta/\tau_{31}\).  This conditioning and the
non-Gaussian/ReLU reset closure remain open; the tensor-free action identity
does not solve them.

The problem graph should therefore contain four separate claims:

1. **Temporal K31 Bellman Hankel rank one:** proved after compiling the
   target/horizon adjoint.
2. **Local provider factorization:** proved algebraically by
   \(B=W^\top A,D=Q31B\); the generic shared-\(\Gamma\) theorem is a fallback,
   not a necessary restriction on this structured circuit.
3. **Affine-quadratic/Jordan-object cost:** conditional GO at
   \(18.288/21\), provided one-square reset and zero paid context
   acquisitions.
4. **Reset and scalar observability:** open; requires the kernel inclusion,
   a quantitative \(\tau_{31}\), and a proof that contexts/fourth legs ride
   the existing trajectory.

The next mathematical question is no longer whether a q4 word survives a
backward Krylov chain.  It is whether the reset supplies a well-defined and
well-conditioned \(A^\flat\) representative on which the exact local circuit
can act, without spending the residual \(2.712\) equivalents on new
contexts.

### Full affine--quadratic action and exact reset-adjoint reuse

The pure-matrix formula is only the H2/H2 sector.  For centered Gaussian
affine--quadratic contexts

\[
U=h^Tg+g^TNg-\operatorname{tr}N,
\qquad
V=r^Tg+g^TRg-\operatorname{tr}R,
\]

the complete 3--1 cumulant is

\[
\kappa_4(U,U,U,V)
=48\operatorname{tr}(N^3R)
+24\{h^TNRh+h^TN^2r\}.
\]

Define the two Jordan objects and two reused vectors

\[
S=N^2,qquad J=N\circ R=\tfrac12(NR+RN),
\qquad u=Nh,qquad v=Nr.
\]

Then the action has the in-place form

\[
\boxed{\kappa_4(U,U,U,V)
=48\langle S,J\rangle_F
+24\{h^TJh+u^Tv\}.}
\]

This identity gives the sharp reuse criterion.  For a quadratic reset

\[
\mathcal R(N)=\alpha N+\beta N^2,
\]

the Frobenius adjoint of its derivative is

\[
D\mathcal R_N^*[R]
=\alpha R+\beta(NR+RN)
=\alpha R+2\beta J.
\]

Hence a correct backward pass through the square reset already forms the
same \(J\) required by the K31 action.  The forward pass already forms
\(S=N^2\).  Under these two exact reuse conditions, the fourth-cumulant
action adds no matrix-product bank at all: only Frobenius inner products and
three \(r\)-dimensional matrix--vector contractions remain.  If the backward
reset does not expose \(J\), one extra product bank suffices; if the forward
reset does not expose \(S\), a second one is required.

The linear-chaos objects do not threaten the dense budget.  Carrying \(h\)
and \(r\) alongside an \(r\times r\) symmetric core widens each rectangular
mix from \(d\) to \(d+r\).  At \(r=55,n=1024\), this adds only

\[
r/n=0.0537109375
\]

GEMM-equivalents per traversed layer.  Eleven such mixed steps change the
optimistic total from about \(18.288\) to about \(18.879\).  Three local
matrix--vector banks cost

\[
3r^2/n^2\simeq0.00866
\]

per action layer; even at all ten late layers this is below \(0.087\).
Thus the full affine--quadratic action remains below about \(18.97/21\),
provided the primal context is already retained and only one matching dual
context chain is new.  If both primal and dual chains are new, their dense
transport costs double and the route fails the 21-equivalent cap.

### One-bundle tensor certificate

The hardware-relevant provider rank is the number of separately transported
context bundles, not the scalar-output rank and not the number \(n\) of
batched contexts.  A fourth-order 3--1 Bellman tensor is exactly one-bundle
K31-factorable when it has the partially symmetric diagonal form

\[
T=\sum_{a=1}^n u_a^{\otimes3}\otimes v_a,
\]

or, in matrix notation,

\[
T=(W^{\otimes3}\otimes P)\Delta_4,
\]

where \(u_a\) and \(v_a\) are corresponding columns of \(W\) and \(P\), and
\(\Delta_4=\sum_ae_a^{\otimes4}\).  For the supplied K31 action
\(P=WQ31^T\), so this certificate holds and fuses to
\(B=W^TA,D=Q31B\).  A sum of \(t\) such diagonal forms requires \(t\)
separate context bundles unless they share the same \(W\)-mixed primal
cores.  A generic 3--1 tensor does not have \(t=1\), even though its final
contraction is scalar.

This identifies the precise remaining factor test: prove that the *complete*
q4-to-endpoint derivative, after subtracting the already represented old
K31/CP contribution, lies in this one-bundle diagonal 3--1 subspace.  The
equivalent invariant statement is the endpoint-sufficiency kernel condition

\[
\ker\mathcal M_{31}\subseteq\ker\mathcal V.
\]

If it fails, the orthogonal derivative component is genuine all-distinct or
non-diagonal fourth-leg information; the scalar Bellman accumulator has not
removed it, only exposed exactly where another local moment family or
transported state is required.

## 2026-08-31 -- novelty correction: total AQ action versus residual provider

The preceding \(18.288/21\) (or full-AQ \(<18.97/21\)) calculation prices the
**total** affine--quadratic K31 action.  It must not be interpreted as the cost
of a new raw-MSE feature.  At the same birth/consumption site the deployed
D11/Q24 lineage already owns the assembled connected K31 plane

\[
K31^{old}=K31^{CP4}+K31^{star}
+3\alpha\,\operatorname{offdiag}(v[:,None]C),
\]

and DTOT2 is one of its consumers, hence part of the same-site adjoint
\(\Lambda_l\), not another provider.  The exact novelty ledger is

\[
\boxed{\rho_l^{31}
=\langle\Lambda_l,K31_l^{true}-K31_l^{assembled}\rangle_F.}
\]

When the lower moments are synchronized, the connected counterterms cancel
and this reduces to

\[
\boxed{\rho_l^{31}
=\sum_i\left(
\mathbb E_{true}[U_i^3V_i]
-\mathbb E_{Q24}[U_i^3V_i]
\right),
\qquad V_i=\sum_j\Lambda_{ij}U_j.}
\]

The assembled-old contraction is only an elementwise/Frobenius contraction
once \(\Lambda_l\) is available.  Therefore, if an independently lawful
true-law oracle exists, one may compute its total action and subtract the old
scalar essentially for free; baseline subtraction does not double the true
oracle's arithmetic cost.

However, the affine--quadratic action itself is exactly the scalar adjoint of
the quadratic-chaos square-lift/DTOT2 sector.  Consequently:

- if \((h,N;r,R)\) is reconstructed only from the current formal
  D11/Q24/DTOT2 state, its complete AQ action is already owned and cancels
  after the displayed subtraction;
- the \(18.288\) circuit can then be only a mutually exclusive replacement
  implementation (possibly a utility play), never additive raw value;
- it becomes a raw feature only if its oriented cores represent information
  about the true activation law that is absent from the assembled closure.
  Existence and observability of those cores are precisely unproved.

Write the true activation context as

\[
X=X^{\le2}+R_{>2}.
\]

Fourth-cumulant multilinearity shows that the potentially novel residual

\[
\mathcal B_{31}^{>AQ}
=\mathcal B_{31}(X,\Lambda)
-\mathcal B_{31}^{AQ}(X^{\le2},\Lambda)
\]

is exactly the sum of polarized terms containing at least one
\(R_{>2}\) leg.  These terms are not determined by \((h,N)\): two laws can
share every H1/H2 coefficient and still have different fourth cumulants.
Thus exact subtraction does **not** preserve the AQ circuit or its provider
rank in general.  A translate by an already-known old statistic can leave the
generic residual moment tensor full-rank.

There are only three exact rank outcomes:

1. **Formal AQ equals assembled old:** residual provider rank is zero and the
   whole AQ total cancels.
2. **A true-law total action has an independently proved one-bundle oracle:**
   residual acquisition has the same marginal circuit as that true oracle,
   because the old scalar is free; this does not follow from Q24 state.
3. **Only low-chaos/AQ state is available:** beyond-AQ residual is
   nonidentifiable and may recover the original all-distinct/four-leg provider
   rank.  The scalar consumer theorem supplies no cost reduction.

Equivalently, with \(\mathcal O_l\) denoting the actually retained old state,
the decisive local rank is the conditional quotient rank

\[
r_{res,l}
=\operatorname{rank}_{provider}
\left(\Lambda_l\circ(M_{31}^{true}-M_{31}^{assembled})
\mid\mathcal O_l\right),
\]

not the rank of the total AQ formula.  It is zero on the formal closure,
one-bundle only after a new true-law factor theorem, and generically
uncontrolled otherwise.

### Corrected graph split

1. **Temporal scalar residual accumulator:** the rank-one Hankel theorem
   remains proved conditional on acquiring each local \(\rho_l^{31}\).
2. **Total AQ K31 circuit:** algebraically proved but marked
   `duplicate/closed for raw novelty`; its cost is relevant only as a
   replacement implementation.
3. **Old same-site subtraction:** proved and negligible after the local
   adjoint is available.
4. **True-minus-assembled residual acquisition:** open and decisive; the
   \(18.288\) AQ cost must not be attached to it without a beyond-AQ or
   independently oriented true-core representation theorem.
5. **Reset/observability:** open; current Q24 arrays do not contain oriented
   latent \(N,N^2,N\circ R\), so the earlier Jordan-product reuse premise does
   not hold for the deployed lineage.

The next actionable question is to derive a tensor-free oracle for the first
omitted higher-chaos terms (one H3/H4 leg coupled to three AQ legs), then
subtract its NCQC/CP3/Q24 row-space component.  Recomputing the total AQ action
cannot produce nonzero residual value.

## 2026-08-31 -- first polynomial degree that can survive subtraction

The square lift \(N^2\) is the AQ/Q24/DTOT2-owned sector.  A matrix-polynomial
reset can therefore create raw novelty only through a coefficient of degree at
least three, or through a better oriented true-law state on which that
coefficient is evaluated.

There is an exact scalar cavity expansion for the H2 reset coefficient.  In an
eigenbasis of a centered quadratic form, write

\[
X=S_i+\lambda_i x_i,
\qquad x_i=g_i^2-1,
\qquad S_i=\sum_{j\ne i}\lambda_j(g_j^2-1),
\]

with \(S_i\) independent of \(x_i\).  Let

\[
\Psi_i(q)=\mathbb E(S_i+q)_+,
\qquad
r_i(\lambda_i)=\mathbb E[x_i\Psi_i(\lambda_i x_i)].
\]

If the cavity density \(f_i\) is smooth at the threshold, then

\[
\Psi_i'(0)=\Pr(S_i>0),\quad
\Psi_i''(0)=f_i(0),\quad
\Psi_i'''(0)=-f_i'(0),\quad
\Psi_i^{(4)}(0)=f_i''(0).
\]

Using

\[
\mathbb E x_i^2=2,quad
\mathbb E x_i^3=8,quad
\mathbb E x_i^4=60,quad
\mathbb E x_i^5=544,
\]

gives the exact local Taylor jet

\[
\boxed{
r_i(\lambda)
=2p_i\lambda+4f_i(0)\lambda^2
-10f_i'(0)\lambda^3
+\frac{68}{3}f_i''(0)\lambda^4+O(\lambda^5),
}
\qquad p_i=\Pr(S_i>0).
\]

Thus the only degree-three novelty is controlled by the slope of the
leave-one-out density at the ReLU threshold.  For a symmetric cavity law,
\(f_i'(0)=0\) exactly: the cubic matrix word vanishes and the next possible
novel word is degree four.  A centered Gaussian bath is in this closed class.
Nonzero degree three requires threshold asymmetry, for example a stable
generalized-chi-square skew or nonzero standardized mean.

An ordinary common-coefficient polynomial

\[
a_1N+a_2N^2+a_3N^3
\]

matches this jet only when \(p_i,f_i(0),f_i'(0)\) are common across the
eigen-directions, or when their variation is below the target-weighted error
gate.  Exact coefficients are leave-one-out spectral statistics; a dominant
eigenvalue makes them coordinate-dependent and defeats a universal matrix
polynomial.  The common-coefficient approximation is justified only by a
no-dominant-mode/cavity certificate.

### Sharp degree budget

One extra matrix power over all \(n\) latent cores costs

\[
10s_r=10r^3/n^2\simeq1.5867
\]

GEMM-equivalents on the ten active layers.  Starting from the optimistic
oriented total-circuit cost \(18.288\), a cubic closure costs about
\(19.875<21\), while a quartic closure costs about
\(21.461>21\).  Including the H1 vectors moves the cubic total to about
\(20.56\), still feasible, but leaves no room for another power.  Hence:

- **degree three is the only new polynomial word that can fit** the residual
  provider allowance without an additional fusion/reuse theorem;
- **if the cavity is threshold-symmetric, this entire polynomial lane closes
  under the present cap**, because its only affordable novel coefficient is
  identically zero;
- **reopen** only with a proof that \(-10f_i'(0)\) is nonzero, sufficiently
  common across directions/nets, and that the resulting cubic Bellman action
  is not in the old CP3/Q24/NCQC row space.

For a rigorous approximation gate, if \(\sup_{|q|\le q_*}|f_i''(q)|\le L_i\)
on the relevant cavity interval, the cubic truncation remainder satisfies

\[
|R_i^{(3)}|
\le \frac{L_i}{24}\,\mathbb E|x_i|^5\,|\lambda_i|^4
\]

inside that interval, with the tail event bounded separately.  The
Bellman-weighted squared sum of these remainders, plus cavity-jet variation,
must be at most \(1-0.9253^2\) or \(1-0.98346^2\) of the residual action
energy.  Polynomial degree by itself is not a fidelity certificate.

### Translation-invariance of conditional provider rank

There is a sharp answer to whether exact baseline subtraction changes the
local circuit rank.  Let \(\mathcal O_l\) include the already assembled old
scalar \(\chi_l^{old}\), and let \(C(\cdot\mid\mathcal O_l)\) be any exact
provider complexity for which adding a retained scalar is free.  Then

\[
\boxed{
C(\chi_l^{true}-\chi_l^{old}\mid\mathcal O_l)
=C(\chi_l^{true}\mid\mathcal O_l).}
\]

One direction computes the true action and subtracts the retained old scalar;
the reverse direction computes the residual and adds that scalar.  The same
identity holds for conditional separated rank and arithmetic-circuit rank.
Thus subtraction leaves exactly the same marginal circuit/rank **if and only
if the proposed circuit really computes the true total action**.  When the AQ
circuit computes only the formal old action, both sides are already retained
and the novel residual is identically zero.

### Surface-distribution identity for the first surviving word

The cavity coefficient has a direct reset-tangent interpretation.  In the
sense of distributions under a smooth cavity law,

\[
(S+\lambda x)_+
=S_++\lambda H(S)x
+\frac{\lambda^2}{2}\delta(S)x^2
+\frac{\lambda^3}{6}\delta'(S)x^3+O(\lambda^4).
\]

The H2 projection contracts once more with \(x\).  Therefore its first three
terms are

\[
\lambda\,\mathbb E[x^2H(S)]
+\frac{\lambda^2}{2}\mathbb E[x^3\delta(S)]
+\frac{\lambda^3}{6}\mathbb E[x^4\delta'(S)].
\]

The \(H(S)\) term is the ordinary AQ tangent and the \(\delta(S)\) term is the
square/Q24 lift.  After same-site subtraction of those owned pieces, the first
possible residual is exactly the \(\delta'\)-surface functional.  Independence
of \(S,x\), \(\mathbb E x^4=60\), and
\(\mathbb E\delta'(S)=-f_S'(0)\) give

\[
\frac{\lambda^3}{6}\mathbb E[x^4]\mathbb E[\delta'(S)]
=-10f_S'(0)\lambda^3.
\]

In the full AQ reset notation the corresponding local novelty is the old-span
quotient of

\[
\frac16P_{H\le2}[\delta'(g)(\delta g)^3].
\]

This is the exact earliest-gate object.  A symmetry certificate that kills
this quotient closes every polynomial degree affordable under the current
cap; a nonzero certificate identifies the precise cubic provider whose
observability and Bellman value must be proved.

### Lindeberg/Edgeworth size of the cubic survivor

The same identity gives a fail-fast asymptotic size bound in the
many-small-eigenvalue regime.  Put

\[
s_2=\sum_{j\ne i}\lambda_j^2,qquad
s_3=\sum_{j\ne i}\lambda_j^3,qquad
\sigma^2=2s_2,qquad
\kappa_3(S_i)=8s_3.
\]

For a centered Lindeberg cavity, the first density Edgeworth term is

\[
f_i(s)=\frac1\sigma\phi(s/\sigma)
\left[1+\frac{\gamma_1}{6}H_3(s/\sigma)\right]
+\text{remainder},
\qquad
\gamma_1=\frac{\kappa_3(S_i)}{\sigma^3}.
\]

Since \(H_3'(0)=-3\),

\[
f_i'(0)
=-\frac{\phi(0)\kappa_3(S_i)}{2\sigma^5}
+\text{remainder},
\]

and hence

\[
-10f_i'(0)
=\frac{40\phi(0)s_3}{(2s_2)^{5/2}}
+\text{remainder}.
\]

Define spectral concentration

\[
\mu=\frac{\max_j|\lambda_j|}{\sqrt{s_2}}.
\]

Using \(|s_3|\le\max|\lambda_j|s_2\), the cubic-to-linear coefficient at an
eigendirection obeys, to first Edgeworth order,

\[
\frac{|a_3\lambda_i^3|}{|\lambda_i|}
\le \frac{40\phi(0)}{2^{5/2}}\mu^3
\simeq2.82\mu^3.
\]

For a flat \(r=55\) spectrum, \(\mu\simeq r^{-1/2}\), giving only about
\(0.0069\) per active reset at this order.  Even ten perfectly coherent
layers would provide only about \(0.069\) error-amplitude correction, or
about \(1-(1-0.069)^2\simeq13.3\%\) raw-MSE reduction.  This is below the
20--30% fire scale and far below the roughly \(0.42\) amplitude correction
needed to move \(2.9\times10^{-8}\) toward \(10^{-8}\).

This is an asymptotic theorem with an Edgeworth remainder, not a uniform
finite-spectrum bound.  It exposes the only exceptions:

- a nonzero threshold/mean shift, for which even a Gaussian cavity has
  \(f'(-m)\ne0\);
- a spectrally concentrated/spiked core with \(\mu\) large;
- a non-Lindeberg cavity whose density slope is not controlled by the first
  three cumulants.

Those exceptions are also the regimes in which common cavity coefficients
become coordinate- and net-dependent.  Therefore the cubic lane has a sharp
transfer tradeoff: the diffuse regime is stable but too small, while the
large-signal regime requires a spike/mean-shift certificate and a separate
proof that its coefficient transfers across nets.

For the important mean-shift exception the Gaussian formula is exact.  If
\(S\sim N(m,\sigma^2)\), \(a=m/\sigma\), then

\[
f_S'(0)=\frac{a\phi(a)}{\sigma^2},
\qquad
a_3=-\frac{10a\phi(a)}{\sigma^2},
\qquad
a_1=2\Phi(a).
\]

Hence the cubic-to-linear ratio in eigendirection \(i\) is

\[
R_i=rac{5|a|\phi(a)}{\Phi(a)}
\frac{\lambda_i^2}{\sigma^2}.
\]

For \(a\ge0\), the first scalar factor is at most about \(1.47\).  In a flat
\(r=55\) bath, \(\lambda_i^2/\sigma^2\simeq1/(2r)\), so even the optimal
positive mean gives only about \(0.0134\) per layer, or \(0.134\) across ten
perfectly coherent layers.  That can reach a roughly 20% raw move in an ideal
alignment, but it remains far below the roughly \(0.42\) amplitude needed for
the campaign goal.  Goal-scale mean-driven cubic value therefore requires an
effective spectral rank on the order of eighteen or smaller, or a separate
Bellman amplification certificate.  Negative \(a\) can make the ratio to the
tiny linear response large, but its absolute surface density is exponentially
small and must be judged by the absolute \(1.691\times10^{-5}\) gate.

These mean-shift ratios are for the total cubic word before old-span
projection.  An exactly flat-magnitude spectrum has \(N^3=c^2N\), so its
novel cubic is zero regardless of the mean shift.  Any fire estimate must also
include the minimal-polynomial residual factor derived below; the quoted
flat-r55 number is an upper scale for the unresidualized word, not a raw-gain
prediction.

### Ownership correction: cubic reset jet is future-visible, not same-site K31

The cubic object

\[
b_{3,l}=\frac16(I-P_{old})P_{H\le2}
[\delta'(g_l)(\delta g_l)^3]
\]

is a third Frechet derivative of the AQ reset and lands back in the
\(H\le2\) state.  It is not the same-site latent-H3
true-minus-assembled K31 defect.  Because the local D11/Q24 action is already
owned, \(b_{3,l}\) earns novelty only through strictly positive-depth future
paths.

For a transition defect

\[
z_{l+1}=T_lz_l+b_l,
\]

with local consumers \(C_kz_k\), the exact no-double-counting value is

\[
\boxed{
\Delta_{future}
=\sum_{j<k}\left\langle B_k,
C_kT_{k-1}\cdots T_{j+1}b_j\right\rangle.}
\]

All zero-length birth/consumer pairings are deleted.  A compiled future
costate folds this strict-depth sum to one local scalar per birth, so the
temporal accumulator remains rank one; nevertheless the reverse AQ reset
chain and the source acquisition of \(b_{3,l}\) must still be paid.  Finally,
the result must be projected off the complete old D11 endpoint/Krylov arena,
not merely off the local Q24 scalar.

This creates two independent novelty nodes:

1. **AQ cubic-jet future completion:** the displayed strict-depth action of
   \(b_3\), governed in the scalar cavity by \(-10f'(0)\).
2. **Same-site H3 K31 defect:** the exact true-H3-minus-assembled surface
   contraction.  Its common \(\delta'\) distribution does not give it the
   same ownership or state type as the cubic reset jet.

The \(2.82\mu^3\) and flat-r55 \(0.0069\)-per-reset estimates are
source-relative.  Promotion requires composing them with the strict-depth
Bellman norms and showing the resulting old-span-residual amplitude exceeds
the campaign threshold \(1.691\times10^{-5}\), including the Edgeworth and
reset-model remainders.

Writing \(L_l^{>0}\) for the strict-positive-depth map from a birth at layer
\(l\) to the endpoint and \(P_{\mathcal O}\) for the complete old arena, the
required bridge is

\[
A_3=left\|(I-P_{\mathcal O})
\sum_lL_l^{>0}b_{3,l}\right\|_V
\le\sum_l
\|(I-P_{\mathcal O})L_l^{>0}\|_{op}\,\|b_{3,l}\|.
\]

If the cavity estimate supplies
\(\|b_{3,l}\|\le(2.82\mu_l^3+\varepsilon_l)\|b_{1,l}\|\), then the exact
fail-fast bound is

\[
A_3\le\sum_l(2.82\mu_l^3+\varepsilon_l)
G_l\|b_{1,l}\|,
\qquad
G_l=\|(I-P_{\mathcal O})L_l^{>0}\|_{op}.
\]

The cubic lane closes if this upper bound is below
\(\eta_3+1.691\times10^{-5}\).  Source-relative \(0.0069\) alone is not an
endpoint-value proof; a claimed exception must exhibit the corresponding
Bellman gain \(G_l\) rather than assuming it.

### Exact algebra gate for a one-word cubic reset

When the AQ context has no latent linear leg, orthogonal equivariance forces
the reset H2 coefficient to commute with \(N\).  For a fixed realized simple
spectrum it is therefore a spectral function of \(N\), and by
Cayley--Hamilton it is some polynomial of degree at most \(r-1\).  The cheap
cubic specialization \(a_3N^3\) is exact at third order only when the cavity
slopes \(f_i'(0)\) agree across eigen-directions.

With a nonzero latent linear leg \(h\), equivariance permits the full algebra
generated by

\[
N\quad\text{and}\quad P=hh^T.
\]

For generic simple-spectrum \(N\) and a cyclic vector \(h\) (all eigenbasis
coordinates nonzero), this algebra is all of \(M_r\).  Indeed the spectral
projectors \(E_{ii}\) are polynomials in \(N\), and

\[
E_{ii}PE_{jj}=h_ih_jE_{ij}
\]

supplies every matrix unit.  Hence no exact one/two-word matrix-polynomial
closure of width independent of \(r\) exists in the generic affine case.

The surviving exact exceptions are:

- \(h=0\), leaving scalar spectral calculus in \(N\);
- \(h\) supported on one or a few eigenspaces of \(N\);
- \(N\) with a bounded number of eigenvalue blocks and \(h\) respecting that
  block decomposition;
- a target-conditioned quotient that annihilates all but one/two generated
  words.

This sharpens the cubic transfer tradeoff.  A scalar threshold mean may make
\(f'(0)\) nonzero while preserving the spectral one-word algebra.  A generic
latent linear mean can also make the cubic large, but simultaneously expands
the exact reset algebra to full Burnside dimension.  The latter cannot be
called a one-word cubic mechanism without an endpoint-quotient certificate.

### Exact degree/correlation criterion after the old quotient

Let \(y\) be the strict-depth Bellman-valued reset defect in its endpoint
Hilbert norm \(\|\cdot\|_V\), and let \(\mathcal P_k\) be the realizable
matrix-polynomial word space of degree at most \(k\), after all complete-old
arena directions are included.  The best degree-\(k\) optimally refit feature
has squared correlation

\[
\rho_k^2
=\frac{\|\Pi_{\mathcal P_k}y\|_V^2}{\|y\|_V^2}
=1-\frac{\|(I-\Pi_{\mathcal P_k})y\|_V^2}{\|y\|_V^2}.
\]

Because degree at most two is old, the correct novel degree-three capture is
the quotient ratio

\[
\boxed{
\rho_{3\mid2}^2
=1-
\frac{\|(I-\Pi_{\mathcal P_3})y\|_V^2}
{\|(I-\Pi_{\mathcal P_2})y\|_V^2}.}
\]

Consequently, the affordable cubic closure reaches correlations \(0.9253\)
or \(0.98346\) if and only if its Bellman-weighted tail fractions are at most

\[
1-0.9253^2=0.14381991,
\qquad
1-0.98346^2=0.03280643.
\]

Equivalently, after orthonormalizing polynomial words in the actual
Bellman/value spectral measure with coefficients \(c_j\), degree three needs

\[
\frac{\sum_{j\ge4}|c_j|^2}{\sum_{j\ge3}|c_j|^2}
\le0.14381991\quad\text{or}\quad0.03280643.
\]

This is the exact answer; ordinary Frobenius or unweighted Chebyshev error is
not.  In a threshold-symmetric cavity \(c_3=0\), so
\(\rho_{3\mid2}=0\): adding the only affordable degree yields no novel
correlation at all.  If symmetry is broken, the cubic is viable only when it
contains at least \(85.618\%\) or \(96.719\%\) of the residual value energy.

For an analytic scalar spectral reset with Chebyshev ellipse parameter
\(R>1\) and \(|f(z)|\le M_R\), the coefficient bound

\[
|c_j|\le2M_RR^{-j}
\]

gives the sufficient tail certificate

\[
\sum_{j>k}|c_j|^2
\le\frac{4M_R^2R^{-2(k+1)}}{1-R^{-2}}.
\]

This may certify degree three when the standardized spectrum stays far from a
threshold singularity.  Near a small-bath/spiked threshold, \(R\downarrow1\),
the required degree rises precisely in the regime where the deployment cap
forbids degree four.

### Forward cubic cost versus noncommuting pullback cost

The statement that degree three adds one product bank applies to a forward
birth or forward polynomial reset.  A generic reverse pullback is more
expensive.  For

\[
p(N)=a_1N+a_2N^2+a_3N^3
\]

and a symmetric output costate \(R\),

\[
\boxed{
Dp_N^*[R]
=a_1R+a_2(NR+RN)
+a_3(N^2R+NRN+RN^2).}
\]

With \(N^2\) stored, the square pullback needs one product \(NR\), since its
transpose gives \(RN\).  The cubic addition generically needs two more:
\(N(NR)=N^2R\) and \((NR)N=NRN\); transposition gives \(RN^2\).  Therefore

\[
\text{forward cubic increment}=10s_r\simeq1.5867,
\]

but

\[
\text{generic cubic-pullback increment}=20s_r\simeq3.1734.
\]

The latter does not fit a residual allowance of \(2.712\).  It collapses to
one extra product only in the commuting exception \([N,R]=0\), where

\[
Dp_N^*[R]=a_1R+2a_2NR+3a_3N^2R,
\]

or when the reverse products are already billed/fused.  Thus the
\(19.875<21\) cubic figure is valid for a forward local birth evaluated by an
already compiled baseline Bellman chain.  It is not a cost certificate for a
new self-consistent noncommuting cubic forward-and-reverse reset.

For the actual batched contexts, the commuting exception is itself sharply
testable.  With

\[
D_a=\sum_bQ_{ab}B_b,
\]

define

\[
\mathcal C_a:q\mapsto\sum_bq_b[B_a,B_b]
\in\mathfrak{so}_r.
\]

Then \([B_a,D_a]=0\) if and only if the \(a\)-th row of \(Q\) lies in
\(\ker\mathcal C_a\).  Since \(e_a\in\ker\mathcal C_a\), and for generic
cores with

\[
\dim\mathfrak{so}_{55}=1485>n-1=1023
\]

the map has rank \(n-1\), one has generically

\[
\ker\mathcal C_a=\operatorname{span}\{e_a\}.
\]

Thus the low-cost commuting pullback requires \(Q\) to be diagonal row by row
(or to satisfy an exceptional common-centralizer identity).  A generic dense
Bellman mixing cannot use it; the diagonal case collapses toward the already
owned star/commuting sector.

### Exact full-core symmetry certificate for the cubic surface

Marginal threshold symmetry alone is not enough in the noncommutative AQ
problem because the Bellman test is correlated with the reset surface.  For a
test \(\phi\in H_{\le2}\), define the weighted surface density

\[
w_\phi(t)=\mathbb E\!\left[
\phi(z)h(z)^3\delta(g(z)-t)\right].
\]

Then, distributionally,

\[
\boxed{
\left\langle\phi,b_3\right\rangle
=\frac16\mathbb E[\phi h^3\delta'(g)]
=-\frac16w_\phi'(0).}
\]

Therefore the cubic birth vanishes in the target quotient if and only if
\(w_\phi'(0)=0\) for every quotient Bellman test \(\phi\).  The scalar
condition \(f_g'(0)=0\) tests only the unweighted special case.

A sufficient exact certificate is a Gaussian-measure-preserving involution
\(J\) satisfying

\[
g(Jz)=-g(z),
\qquad
\phi(Jz)h(Jz)^3=\phi(z)h(z)^3
\]

for every target test.  It makes every \(w_\phi\) even.  For a pure quadratic
context, an orthogonal certificate requires

\[
J^TNJ=-N,
\]

so the nonzero spectrum of \(N\) must occur in \(+/-\) pairs with compatible
test parity.  This is an exceptional algebraic symmetry, not a consequence of
ordinary centering.  Diffuse generic cores have only the approximate
Lindeberg/Edgeworth suppression above.

The weighted derivative can nevertheless be reduced to an ordinary surface
oracle.  Gaussian coarea/Stein integration gives, whenever the zero surface is
transverse,

\[
\boxed{
\mathbb E[F\delta'(g)]
=-\mathbb E\left[\delta(g)\,
\operatorname{div}_\gamma
\left(F\frac{\nabla g}{\|\nabla g\|^2}\right)\right],}
\]

where \(\operatorname{div}_\gamma v=\operatorname{div}v-z^Tv\).  For the
half-AQ convention \(q=\nabla g=u+Az\), \(d=q^Tq\), this is

\[
\mathbb E[F\delta'(g)]
=-\mathbb E\!\left[\delta(g)\left\{
\frac{\nabla F\cdot q}{d}
+F\left(\frac{\operatorname{tr}A-z^Tq}{d}
-\frac{2q^TAq}{d^2}\right)
\right\}\right].
\]

Thus an existing \(\delta(g)\) boundary-integral engine can acquire the cubic
surface without numerically differentiating a density, but it must support
rational \(d^{-1},d^{-2}\) moments.  The sharp new premise is transversality:

\[
\inf_{g(z)=0}\|u+Az\|^2>0.
\]

If the quadratic surface contains its critical point, the rational reduction
is singular and no constant-cost claim follows.  Also, marginal symmetry is
provably insufficient: \(g(z)=z,F(z)=z\) has a symmetric standard-normal
density but \(\mathbb E[z\delta'(z)]=-\mathbb E\delta(z)\ne0\).

For

\[
g(z)=m+u^Tz+\frac12(z^TAz-\operatorname{tr}A),
\]

this transversality margin is explicit.  Let

\[
u_R=AA^+u,
\qquad u_0=(I-AA^+)u,
\qquad
\delta=m-\frac12\operatorname{tr}A
-\frac12u_R^TA^+u_R.
\]

After shifting the range coordinates,

\[
g=\delta+\frac12y^TAy+u_0^Tz_0,
\qquad
\nabla g=Ay+u_0.
\]

If \(u_0\ne0\), then

\[
\inf_{g=0}\|\nabla g\|^2=\|u_0\|^2.
\]

If \(u_0=0\), then \(\delta=0\) puts the critical point on the boundary and
the margin is zero.  For \(\delta\ne0\), the boundary is empty when no
eigenvalue \(\lambda\) has \(\lambda\delta<0\); otherwise

\[
\boxed{
\inf_{g=0}\|\nabla g\|^2
=2|\delta|\min_{\lambda\delta<0}|\lambda|.}
\]

This gives a purely algebraic close/reopen test for the rational surface
oracle before any quadrature or provider construction.

### Cubic minimal-polynomial residual after the old degree-two span

Even a nonzero coefficient \(a_3\) does not make \(N^3\) novel.  The local old
polynomial span already contains \(I,N,N^2\).  Put

\[
s_j=\operatorname{tr}(N^j),
\qquad
G_2=(s_{a+b})_{0\le a,b\le2},
\qquad
v=(s_3,s_4,s_5)^T.
\]

The exact Frobenius old-span residual is

\[
R_{3\mid2}=N^3-\Pi_{\operatorname{span}\{I,N,N^2\}}N^3,
\]

with squared norm

\[
\boxed{
\|R_{3\mid2}\|_F^2=s_6-v^TG_2^\dagger v.}
\]

This is the Schur complement of the degree-three moment Hankel matrix.  It
vanishes exactly when the realized spectral measure satisfies a quadratic
annihilating polynomial, in particular whenever \(N\) has at most three
distinct eigenvalues.  With only the old linear word projected out,

\[
\left\|N^3-\frac{s_4}{s_2}N\right\|_F^2
=s_6-\frac{s_4^2}{s_2},
\]

the \(s_2\)-weighted variance of \(\lambda^2\).  Thus a flat-magnitude
spectrum \(\lambda_i\in\{+c,-c\}\) has \(N^3=c^2N\) and zero cubic novelty,
even if its threshold-density slope is nonzero.

The affordable birth is therefore not \(a_3N^3\), but

\[
a_3R_{3\mid2},
\]

followed by the complete old endpoint/Krylov quotient.  A large cubic signal
requires both a nonzero weighted surface slope and a nontrivial spectral
minimal-polynomial residual.  Spectral flatness stabilizes the cavity but
drives this second novelty factor to zero; spectral spread makes the word new
but worsens cavity universality and transfer.

For a symmetric Wigner-semicircle spectral shape normalized by
\(s_2/r=1\), the moments are \(s_4/r=2\), \(s_6/r=5\), and odd moments
vanish.  Hence

\[
R_{3\mid2}=N^3-2N,
\qquad
\frac{\|R_{3\mid2}\|_F^2}{\|N^3\|_F^2}=\frac15.
\]

An exactly symmetric semicircle also has \(s_3=0\), so its first Edgeworth
cubic coefficient vanishes before this projection is applied.  For a
near-semicircle with a small nonzero threshold slope, the old-span projection
suppresses the remaining cubic source amplitude by an additional factor
\(1/\sqrt5\).  Thus the canonical diffuse symmetric random-matrix regime is a
fail-fast NO; a quantitative non-Wigner skew/mean exception is required even
before the Bellman-value gate.

### Sharp orientation interval for the cubic action

Power traces determine the spectrum and the norm of \(R_{3\mid2}\), but not
its orientation relative to the Bellman costate.  Let \(q_i\) be the
eigenvalues of \(R_{3\mid2}\) and \(r_i\) those of a symmetric costate \(R\).
Over every relative orthogonal orientation consistent with those spectra,
von Neumann's trace inequality is sharp:

\[
\sum_i r_i^\downarrow q_i^\uparrow
\le \operatorname{tr}(R O\operatorname{Diag}(q)O^T)
\le \sum_i r_i^\downarrow q_i^\downarrow.
\]

The attainable values fill this interval continuously.  Hence the strongest
orientation-free absolute lower bound is zero whenever the interval straddles
zero, while the sharp upper bound is the larger absolute endpoint.  Spectral
moments or Frobenius energy alone therefore cannot promote the cubic source.
The information-minimal extra object for a frozen endpoint is its signed,
old-span-residualized Bellman action scalar; coadaptation again requires the
span of the rotated costates.

## 2026-08-31 -- scope correction: K31 does not by itself complete q4

The scalar K31 Bellman accumulator solves only the K31 consumer channel.  A
transient order-four residual \(Z_{22}\) has four distinct next-interface
emissions.  With \(q_a=u_a^{\odot2}\) and
\(c_{ab}=u_a\odot u_b\), they are

\[
K_{4,Z}(a)=3q_a^TZq_a,
\qquad
K_{31,Z}(a,b)=3q_a^TZc_{ab},
\]

and

\[
\boxed{
K_{22,Z}(a,b)=q_a^TZq_b+2c_{ab}^TZc_{ab}.}
\]

The last two terms are the uncrossed and crossed K22 emissions.  A K31-only
action does not determine the crossed term.  For a generic dense K22 consumer
\(\Lambda^{22}\), its missing scalar is

\[
z_\times
=2\sum_{ab}\Lambda^{22}_{ab}c_{ab}^TZc_{ab}.
\]

Dimensionally and algebraically there are generic \(Z\) directions in the
kernel of every acquired K31 action but not in the kernel of this crossed-K22
action.  They provide an explicit witness that

\[
\ker\mathcal M_{31}\not\subseteq\ker\mathcal V_{full\ q4}
\]

unless the crossed receipt is supplied independently.

Therefore the full q4 carrier may be discarded after one reset if and only if
the next deployed interface receives, with matching provenance, all four of

1. K4;
2. K31;
3. uncrossed K22;
4. crossed K22.

If all four are already assembled by D11/Q24, this is another duplicate and no
new state is needed.  If crossed K22 is absent, the information-minimal frozen
addition is one complete crossed/future Bellman scalar per birth; genuine
coadaptation needs the span of its varying action covectors or the plane
itself.  The problem graph must keep this K22 completion audit separate from
the proved temporal K31 rank-one node.

### Exact collision-sector separation: K31 is blind to q4-perp

The NCQC quartic with collision type (211) splits exactly into the commuting
word and the commutator-angle word

\[
q_4(r,r,s,t)
=48\operatorname{tr}(M_r^2M_sM_t)
+8\operatorname{tr}([M_r,M_s][M_r,M_t]).
\]

For fixed repeated index (r), define the positive semidefinite Gram

\[
G^{(r)}_{st}
=\langle[M_r,M_s],[M_r,M_t]\rangle_F
=-\operatorname{tr}([M_r,M_s][M_r,M_t]).
\]

Then the irreducible q4-perp sector is exactly (-8G^{(r)}).  Its collision
restrictions prove a stronger separation than a generic dimension count:

\[
\begin{array}{c|c}
\text{collision channel}&\text{q4-perp readout}\\ \hline
31: (r,r,r,t)&-8G^{(r)}_{rt}=0,\\
22: (r,r,s,s)&-8G^{(r)}_{ss}=-8\|[M_r,M_s]\|_F^2,\\
211: (r,r,s,t)&-8G^{(r)}_{st}.
\end{array}
\]

Thus every K31 action, scalar or tensorial, is **identically blind** to the
commutator-angle q4-perp source; this is not merely a failure of a particular
carrier.  K22 sees only the Gram diagonal unless a crossed consumer supplies
off-diagonal angles.  The full (211) source requires the value-visible part
of (G^{(r)}_{st}) for (s\ne t).

Consequently the universal scalar K31 Bellman fold eliminates temporal state
only for the commuting/K31 channel.  It cannot eliminate the q4-perp
backward-Krylov carrier.  That carrier closes only if the crossed-K22/211
endpoint annihilates the off-diagonal Gram, if the off-diagonal Gram is already
owned by D11/Q24/NCQC, or if its endpoint-weighted action admits its own fused
scalar acquisition.  No refinement of the K31 local circuit can change this
orthogonality.

The crossed branch also has a sharp target-conditioned rank.  If
\(\bar\Lambda_{22}\) is the symmetric off-diagonal K22 costate, then

\[
\Omega_\times
=\frac12\sum_{a,b}\bar\Lambda_{22,ab}
(w_a\odot w_b)(w_a\odot w_b)^T.
\]

For \(\bar\Lambda_{22}=\sum_{s=1}^r\sigma_sx_sx_s^T\),

\[
\Omega_\times
=\frac12\sum_{s=1}^r\sigma_s
\{W\operatorname{Diag}(x_s)W^T\}^{\odot2},
\]

so one dense congruence is required per Bellman factor.  Accounting for the
K4-owned diagonal, the exact minimum factor count is

\[
r_{22}^{Bell}
=\min_d\operatorname{rank}
(\bar\Lambda_{22}+\operatorname{Diag}d).
\]

For a generic off-diagonal plane at \(n=1024\), the codimension of nullity
\(k\) implies \(k(k+1)/2\le n\), hence \(k\le44\) and

\[
\boxed{r_{22}^{Bell}\ge980.}
\]

Therefore a constant-cost complete K22 provider is generically impossible.
It can reopen only through an explicit low-rank diagonal completion, a
low-rank source \(Z\), or the exact future-dark quotient
\(P_{S_Z}\operatorname{offdiag}\Omega_\times=0\).  The cheap one/two-product
K31 scalar remains a lawful *partial feature* after full old-arena Schur
residualization, but it supplies neither a bound nor a completeness receipt
for the omitted K22 value.

The exact cheap partial feature is explicit.  With

\[
U=W\odot W,
\qquad R_Z=ZU,
\qquad B=W\Lambda_{31}^T,
\]

the first dense K31 action is

\[
\boxed{
\chi_{Z\to31}=3\langle R_Z,W\odot B\rangle_F.}
\]

Because \(Z\) is transiently available at its Q24 birth and \(U\) is a D11
weight object, \(R_Z=ZU\) costs one dense product.  If the reverse consumer
already exposes \(B\), everything else is elementwise; otherwise this is two
products.  Across ten active births the feature therefore costs 10--20
equivalents and fits the 21-equivalent partial-provider allowance.  It is the
only currently proved constant-cost new Z22 statistic, but its gain lower
bound is zero until the complete old-arena residual action is signed and
compared with the \(1.691\times10^{-5}\) amplitude gate.

## 2026-08-31 -- stable even Padé reset as a degree-cost bypass

The cubic lane is not the only way to exceed the old square lift.  In a
centered Gaussian cavity, the exact eigenvalue reset series derived above is

\[
r(\lambda)=\lambda+a_2\lambda^2+a_4\lambda^4+a_6\lambda^6+O(\lambda^8),
\]

where, with \(f_0=(\sigma\sqrt{2\pi})^{-1}\),

\[
a_2=4f_0,
\qquad
a_4=-\frac{68}{3}\frac{f_0}{\sigma^2},
\qquad
a_6=\frac{1646}{5}\frac{f_0}{\sigma^4}.
\]

Here \(r(\lambda)=\mathbb E[(g_i^2-1)\operatorname{ReLU}(X)]\) is the
unnormalized H2 numerator.  Under the conventional Wiener matrix coefficient
\(A_f=\frac12\mathbb E[\operatorname{ReLU}(X)(zz^T-I)]\), multiply
\(r,a_2,a_4,a_6,A,B\) by one half.  The Padé denominator \(c\), relative
tail, and cost are unchanged.

The last coefficient uses
\(\mathbb E(\chi_1^2-1)^7=79008\) and
\(f^{(4)}(0)=3f_0/\sigma^4\).

Put \(x=\lambda^2\).  The \([1/1]\) Padé form

\[
x\frac{a_2+b x}{1+c x}
\]

matches all three even coefficients when

\[
c=-\frac{a_6}{a_4}
=\frac{2469}{170\sigma^2}>0,
\qquad
b=a_4+a_2c
=\frac{9034}{255}\frac{f_0}{\sigma^2}>0.
\]

Writing

\[
A=\frac bc=\frac{18068}{7407}f_0,
\qquad
B=a_2-A=\frac{11560}{7407}f_0,
\]

gives the implementation-friendly reset

\[
\boxed{
r_{Pad\acute e}(N)
=N+A N^2+B N^2(I+cN^2)^{-1}.}
\]

The denominator is strictly positive definite for every symmetric \(N\), so
there is no spectral pole or step-size gate.  Since \(N^2\) is already formed,
the entire beyond-square resummation needs one Cholesky/multiple-right-hand-
side solve, not explicit \(N^4\) or \(N^6\).  Equivalently,

\[
N^2(I+cN^2)^{-1}
=c^{-1}\{I-(I+cN^2)^{-1}\}.
\]

If that local SPD solve bills below about \(1.71\) ordinary product-bank
equivalents, ten layers fit the residual \(2.712\) allowance.  Standard flop
counts suggest this is plausible but do not constitute the required deployed
billing receipt.

In ordinary flop units, Cholesky costs \(r^3/3\) and the two triangular
solves for \(r\) right-hand sides cost about \(2r^3\) together, versus
\(2r^3\) for one dense GEMM.  This predicts about \(7/6\) product-bank
equivalents, or \(1.851\) across ten r55 layers.  On that arithmetic model the
full H1/H2 total is about \(18.97+1.85=20.82/21\).  This is deliberately a
flop certificate, not a board-billing receipt; kernel efficiency and counted
operation classes remain an independent gate.

This rational reset remains an H2 spectral shadow.  It cannot recover the
irreducible H4/crossed-K22 sector, and its local same-site K31 re-expression is
still duplicate.  Its only lawful novelty is strict positive-depth change of
the reset trajectory after complete old-arena subtraction.  Also, matching
\(\lambda\)-Taylor degrees two/four/six is **not** the same as capturing scalar
Hermite H2/H4/H6 energy; no 0.925/0.983 correlation follows from the Padé order.
Promotion requires the exact Bellman-weighted rational tail, oriented-core
observability, solve cost, and future-value gates.

After subtracting the matched old Gaussian jet \(N+a_2N^2\), the novel local
birth can be formed directly as

\[
\boxed{
\Delta r_{Pad\acute e}(N)
=B\{N^2(I+cN^2)^{-1}-N^2\}
=a_4N^4(I+cN^2)^{-1}.}
\]

Thus no total-action duplication is needed: the old square cancels
algebraically before strict-depth propagation.  This residual must still be
projected off \(\operatorname{span}\{I,N,N^2\}\) and the full old endpoint
arena.  If \(N\) has at most three distinct eigenvalues, every rational
function of \(N\) lies in that local span and the Padé novelty is zero.  The
solve bypass therefore needs the same nontrivial spectral-minimal-polynomial
certificate as the cubic lane.

Finally, the exact Gaussian leave-one-out variance is

\[
\sigma_i^2=2(s_2-\lambda_i^2),
\]

so \(A,B,c\) are direction-dependent.  Replacing them by one global
\(\sigma^2\simeq2s_2\) needs the same no-dominant-mode certificate as the
polynomial cavity.  Exact coefficients may be written as commuting matrix
functions of \(2(s_2I-N^2)\), but then inverse-square-root/resolvent work must
also be billed and becomes singular in the spiked limit.  The simple one-solve
formula is therefore a diffuse-cavity approximation, not a universal reset
identity.

### First omitted coefficient: the one-pole Padé has a material order-eight miss

For a centered Gaussian cavity, write $x=\chi_1^2-1$, let
 $\mu_j=\mathbb E x^j$, and let $f_0=(\sigma\sqrt{2\pi})^{-1}$.
The exact formal coefficients of the even reset correction are

\[
a_{2m}=\frac{(-1)^{m-1}(2m-3)!!\,f_0\,\mu_{2m+1}}
                 {(2m)!\,\sigma^{2m-2}}.
\]

In particular, $\mu_9=20{,}314{,}880$ and

\[
a_8=-\frac{158710}{21}\frac{f_0}{\sigma^6}
    \simeq-7557.62\frac{f_0}{\sigma^6}.
\]

The one-pole approximant predicts instead

\[
a_8^{[1/1]}=-a_6c
=-\frac{1646}{5}\frac{2469}{170}\frac{f_0}{\sigma^6}
\simeq-4781.15\frac{f_0}{\sigma^6}.
\]

Thus it reproduces only about (63.3\%\) of the first omitted coefficient;
the leading formal error is

\[
r(\lambda)-r_{[1/1]}(\lambda)
=-2776.47\,\frac{f_0}{\sigma^6}\lambda^8+O(\lambda^{10}).
\]

This is a fail-fast warning, not by itself a no-go: in the diffuse regime the
factor $(\lambda^2/\sigma^2)^3$ can still suppress the miss.  But no promotion
is lawful from matching through degree six alone.  The required gate is the
Bellman-weighted *full* remainder, including leave-one-out variance variation;
a scalar Taylor coefficient or an unweighted local norm cannot establish the
0.9253/0.98346 target correlations.

There is a second analytic warning.  Since
 $\mu_{2m+1}\sim(4m+1)!!$, successive coefficient magnitudes satisfy

\[
\left|\frac{a_{2m+2}}{a_{2m}}\right| \sim \frac{8m}{\sigma^2}.
\]

The local moment series is therefore Gevrey/asymptotic with zero complex
radius of convergence after averaging over the unbounded chi-square direction.
Finite-order Padé may be useful under an explicit no-spike/small-
 $\lambda^2/\sigma^2$ certificate, but it is not a convergent universal
matrix-polynomial closure.

## 2026-08-31 — exact finite-secant closure dominates a reset polynomial, conditionally

Let $\mathcal H_{\le2}$ be the affine-quadratic Gaussian Hilbert space,
 $P=P_{\le2}$, and

\[
R(g)=P[(g)_+].
\]

For $g,h\in\mathcal H_{\le2}$, the exact remainder beyond the reset's old
zeroth, tangent, and square jets is

\[
\boxed{
\mathcal N_{\ge3}(g;h)
=R(g+h)-R(g)-DR_g h-\frac12D^2R_g[h,h]
=\frac12\int_0^1(1-t)^2
 P[\delta'(g+th)h^3]\,dt .}
\]

This is just Taylor's integral identity along the one-dimensional secant; it
is exact distributionally.  Consequently, if (R(g+h)), (DR_gh), and
 (D^2R_g[h,h]) are lawful AQ endpoint objects, one extra exact reset
evaluation captures *all* cubic-and-higher reset words along (h).  No finite
matrix polynomial, Padé order, or (\delta') oracle is then needed.  Also,
 (R) is nonexpansive in (L^2):

\[
\|R(g+h)-R(g)\|_2\le \|h\|_2,
\]

because both ReLU and the orthogonal projection are contractions.

There is an exact causal version.  For a base AQ recurrence

\[
g_{l+1}=A_lR(g_l)+s_l
\]

and a proposed fixed source sequence (b_l), define one difference state
 (\Delta_l=\widetilde g_l-g_l).  Then

\[
\boxed{
\Delta_{l+1}
=A_l\{R(g_l+\Delta_l)-R(g_l)\}+b_l .}
\]

Thus one parallel AQ trajectory resums every reset order and every earlier
birth into a single Markov state.  Polynomial degree and formal word-rank do
not grow.  Multiple layer births may share this state only after their
coefficients are fixed; independently tunable birth coefficients require
independent tangent/secant states or a joint refit that fixes their aggregate.

This theorem has four binding limitations.

1. **Provider:** (b_l) must already be a lawful AQ birth.  A scalar H4/K31
   Bellman score is not an AQ vector, and an irreducible crossed-K22 source is
   not automatically representable as a perturbation $h\in\mathcal H_{\le2}$.
2. **Old-span subtraction:** novelty is
   (P_{\rm old}^{\perp}\Delta_L), at the correctly aligned endpoint.  The
   zeroth/tangent/square pieces owned by D11/Q24/NCQC cannot be billed or valued
   again.
3. **Cost:** an extra trajectory costs one full AQ transport/reset bundle from
   its first birth layer.  It fits the remaining cap only if the exact reset is
   in-place or the suffix is short enough; the identity is not a billing
   theorem.
4. **Observability:** exact resummation removes truncation leakage but not
   source or endpoint darkness.  Promotion still requires

   \[
   \frac{\|P_{\rm value}P_{\rm old}^{\perp}\Delta_L\|^2}
        {\|P_{\rm old}^{\perp}\Delta_L\|^2}
   \ge \rho^2
   \]

   in the actual value seminorm, with $1-\rho^2=0.14381991$ or
   $0.03280643$.

The generic compression obstruction also remains.  Linearizing the exact
difference recurrence at (\Delta=0) gives

\[
\dot\Delta_{l+1}=A_lT_{g_l}\dot\Delta_l+\dot b_l,
\qquad T_g=P M_{1_{g>0}}P .
\]

Any smooth exact (k)-coordinate secant carrier has, at the origin, a
 (k)-coordinate carrier for this finite-horizon block-Krylov system.
Therefore (k) is at least the dimension of its reachable space modulo the
common kernel of all future endpoint costates.  For a generic neuron-specific
family (T_{g_l}), Burnside growth makes that quotient full on any endpoint
family that separates the AQ state.  One or two scalar/matrix words are exact
only under a genuine invariant quotient, a fixed target-conditioned dark
kernel, or a special OU/second-quantized intertwiner.  The finite secant
removes the *degree* obstruction; it does not prove the required low-dimensional
carrier.

Hence the exact finite secant is the preferred reset completion whenever a
lawful low-cost AQ birth and one extra suffix trajectory are available.  The
one-pole Padé is only a fallback when the exact endpoint reset cannot be
carried, and the crossed-K22/H4 provider remains a separate open node in both
cases.

## 2026-08-31 — rigorous diffuse-cavity gate for the one-pole reset

Although the averaged moment series has zero radius, a *finite-order real
remainder* bound is available.  Let

\[
\Psi(q)=\mathbb E(S+q)_+,qquad S\sim N(0,\sigma^2),
\]

and (F(q)=\Psi(q)-q/2), which is even.  Since
 (F^{(8)}(q)=f_\sigma^{(6)}(q)), Fourier inversion gives the sharp global
bound

\[
\sup_q|f_\sigma^{(6)}(q)|
\le\frac1{2\pi}\int_{\mathbb R}|t|^6e^{-\sigma^2t^2/2}\,dt
=15\frac{f_0}{\sigma^6}.
\]

For $x=\chi_1^2-1$, $x\ge-1$, so

\[
\mathbb E|x|^9\le \mathbb Ex^9+2
=20{,}314{,}882.
\]

Taylor's integral remainder through degree six therefore proves, for every
real $\lambda$,

\[
\left|r(\lambda)-\left(\lambda+a_2\lambda^2
+a_4\lambda^4+a_6\lambda^6\right)\right|
\le A_8\frac{f_0|\lambda|^8}{\sigma^6},
\]

where

\[
A_4=\frac{68}{3},\qquad A_6=\frac{1646}{5},\qquad
A_8=\frac{15(20{,}314{,}882)}{8!}\simeq7557.63,
\]

and $a_4=-A_4f_0/\sigma^2$, $a_6=A_6f_0/\sigma^4$.

After the old linear/square terms are removed, put

\[
n(\lambda)=r(\lambda)-\lambda-a_2\lambda^2,qquad
\widehat n(\lambda)=\frac{a_4\lambda^4}{1+c\lambda^2},
\qquad c=\frac{C}{\sigma^2},quad C=\frac{2469}{170}.
\]

The rational-minus-degree-six difference is exact:

\[
\left|\widehat n-(a_4\lambda^4+a_6\lambda^6)\right|
=\frac{A_6C}{1+C\theta^2}
\frac{f_0|\lambda|^8}{\sigma^6},qquad
\theta=\frac{|\lambda|}{\sigma}.
\]

Consequently

\[
|n-\widehat n|
\le (A_8+A_6C)\frac{f_0|\lambda|^8}{\sigma^6},
\]

while

\[
|n|\ge\frac{f_0|\lambda|^4}{\sigma^2}
\{A_4-A_6\theta^2-A_8\theta^4\}.
\]

Whenever the denominator is positive, a rigorous pointwise relative-error
certificate is therefore

\[
\boxed{
\epsilon_P(\theta)
=\frac{(A_8+A_6C)\theta^4}
{A_4-A_6\theta^2-A_8\theta^4}.}
\]

For a desired correlation $\rho$, let
 $\delta_\rho=\sqrt{1-\rho^2}$.  The unprojected spectral reset is
guaranteed to have correlation at least $\rho$ if

\[
\theta_{\max}^2\le
\frac{-\delta_\rho A_6+
\sqrt{\delta_\rho^2A_6^2+4\delta_\rho A_4
\{A_8+A_6C+\delta_\rho A_8\}}}
{2\{A_8+A_6C+\delta_\rho A_8\}}.
\]

This is approximately $\theta_{\max}\lesssim0.14$ for $\rho=0.9253$
and $\theta_{\max}\lesssim0.12$ for $\rho=0.98346$.  With the exact
leave-one-out variance,

\[
\theta_i^2=\frac{\lambda_i^2}{2(s_2-\lambda_i^2)},
\]

so these correspond roughly to maximum spectral fractions
 $|\lambda_i|/\sqrt{s_2}\lesssim0.20$ and $0.17$, respectively.

The final old-arena projection is decisive.  Define its retained novel-energy
fraction

\[
\gamma_{old}
=\frac{\|P_{old}^\perp n(N)\|_V}{\|n(N)\|_V}.
\]

Projection contracts the approximation error but may shrink the signal much
more, so the actual sufficient condition is

\[
\epsilon_P(\theta_{\max})
\le\gamma_{old}\sqrt{1-\rho^2}
\]

in the deployed value seminorm.  Equal-magnitude/tiny-minimal-polynomial
spectra have $\gamma_{old}=0$ and remain exact NOs despite excellent
diffuseness.  This theorem therefore supplies a sharp analytic reopen gate:
the Padé lane is viable only when both the no-spike inequality and a nonzero
old-residual spectral fraction hold, followed by signed Bellman
noncancellation and the solve-cost receipt.

### Proof-level polynomial-degree thresholds

The same real remainder theorem gives rigorous sufficient degrees, rather
than relying on scalar-Hermite analogy.  Approximate the novel reset by

\[
n_4(\lambda)=a_4\lambda^4,qquad
n_6(\lambda)=a_4\lambda^4+a_6\lambda^6.
\]

With

\[
D(\theta)=A_4-A_6\theta^2-A_8\theta^4>0,
\]

their pointwise relative errors obey

\[
\epsilon_4(\theta)
\le\frac{A_6\theta^2+A_8\theta^4}{D(\theta)},
\qquad
\epsilon_6(\theta)
\le\frac{A_8\theta^4}{D(\theta)}.
\]

For $\delta_\rho=\sqrt{1-\rho^2}$, degree four is certified when

\[
(1+\delta_\rho)(A_6y+A_8y^2)
\le\delta_\rho A_4,qquad y=\theta_{\max}^2,
\]

and degree six is certified when

\[
(1+\delta_\rho)A_8y^2+\delta_\rho A_6y
\le\delta_\rho A_4.
\]

The resulting sufficient diffuse-cavity thresholds are approximately

\[
\begin{array}{c|cc}
&\rho=0.9253&\rho=0.98346\\ \hline
N^4&\theta_{\max}\lesssim0.119&\theta_{\max}\lesssim0.094\\
N^6&\theta_{\max}\lesssim0.153&\theta_{\max}\lesssim0.136\\
[1/1]\text{ rational}&\theta_{\max}\lesssim0.14&
\theta_{\max}\lesssim0.12.
\end{array}
\]

Thus, in the canonical equal-magnitude r55 scale
 $\theta=1/\sqrt{2(55-1)}\simeq0.096$, the rigorous local conclusion is:

- $N^4$ clears the 0.9253 gate but narrowly misses the conservative 0.98346
  certificate;
- $N^6$ clears both;
- the one-pole solve lies between them in guaranteed accuracy while resumming
  higher even powers.

This aligns the algebra with the cost cliff.  If $N^2$ is resident, $N^4$
needs one additional small-core product bank, whose prior arithmetic estimate
is about 1.587 equivalents across ten r55 layers and fits the 2.712 margin.
 $N^6$ needs a second bank, about 3.173 total, and misses the margin.  The
one-pole SPD solve was estimated at about 1.851 and conditionally fits.  These
are arithmetic estimates only; exact lineage billing remains a separate gate.

All three degree statements are pre-old-projection sufficient conditions.
After residualization, replace $\delta_\rho$ in the displayed inequalities
by $\gamma_{old}\delta_\rho$.  In particular the equal-magnitude spectrum
used for the favorable diffuseness illustration has
 $\gamma_{old}=0$, because its minimal polynomial already places every
even spectral function in the old span.  A real GO needs diffuse *and*
spectrally diverse cores, not a flat-spectrum surrogate.

### Exact spectral-diversity certificate for surviving the old polynomial span

The nonzero $\gamma_{old}$ premise need not be left as a genericity claim.
First consider the local old span $\operatorname{span}\{I,N,N^2\}$ with
Frobenius weights.  For any four distinct eigenvalues
 $\lambda_1,\ldots,\lambda_4$ carrying positive normalized weights
 $p_i$, put

\[
w_i=\frac1{\prod_{j\ne i}(\lambda_i-\lambda_j)}.
\]

Then $\sum_iw_i\lambda_i^k=0$ for $k=0,1,2$.  Therefore, for any scalar
spectral reset function (f), its weighted squared distance from every
quadratic polynomial on these four spectral atoms is exactly

\[
\boxed{
d_4(f)^2=
\frac{f[\lambda_1,\lambda_2,\lambda_3,\lambda_4]^2}
{\sum_{i=1}^4w_i^2/p_i},}
\]

where $f[\lambda_1,\ldots,\lambda_4]=\sum_iw_if(\lambda_i)$ is the third
divided difference.  The error on the full spectrum is at least this
four-atom error.  Thus a concrete nonzero divided difference is a robust
witness that the candidate has not collapsed into $I,N,N^2$; the
denominator also supplies an explicit conditioning margin rather than a
Zariski-only nonzero minor.

For the one-pole novel function

\[
f_P(\lambda)=\frac{\lambda^4}{1+c\lambda^2}
=\frac{\lambda^2}{c}-\frac1{c^2}
+\frac1{c^2(1+c\lambda^2)},
\]

the first two terms are already old.  Hence its certificate is simply

\[
f_P[\lambda_1,\ldots,\lambda_4]
=\frac1{c^2}\sum_i\frac{w_i}{1+c\lambda_i^2}.
\]

This expression is the exact local close/reopen test: zero closes that chosen
spectral witness; a magnitude bounded away from zero, together with separated
atoms and nonnegligible $p_i$, proves a stable local residual.

When the spectrum is sign-symmetric, the odd old word $N$ is orthogonal to
the even reset and the problem reduces to affine approximation in
 $x=\lambda^2$.  Choose three distinct squared eigenvalues
 $x_1<x_2<x_3$, with weights $p_i>0$, and put

\[
v_i=\frac1{\prod_{j\ne i}(x_i-x_j)}.
\]

For $f(x)=x^2/(1+cx)$, its second divided difference is

\[
f[x_1,x_2,x_3]
=\frac1{(1+cx_1)(1+cx_2)(1+cx_3)}.
\]

Consequently the exact three-atom residual is

\[
\boxed{
d_3(f)^2=
\frac1{\{(1+cx_1)(1+cx_2)(1+cx_3)\}^2}
\left(\sum_{i=1}^3\frac{v_i^2}{p_i}\right)^{-1}.}
\]

This shows precisely why a flat $\pm a$ spectrum is dead: it has only one
squared eigenvalue, so no three-atom witness exists.  Conversely, three
well-separated squared eigenvalues of nonnegligible mass give a quantitative
positive local residual without fitting anything.  The certificate still
must be repeated in the deployed value seminorm and after the *full* D11/Q24/
NCQC arena, which may contain more than local quadratic polynomials, but it is
the correct analytic first gate for $\gamma_{old}>0$.

## 2026-08-31 — target-conditioned Krylov evaluation of the rational action

The full $r\times r$ Cholesky solve is unnecessary when the candidate is
needed only through a frozen local Bellman action.  Let $X=N^2\succeq0$, let
 $R$ be the symmetric local costate after removing the old $I,X$ action,
and consider

\[
\chi=\operatorname{tr}\left(R\frac{X^2}{I+cX}\right).
\]

Polynomial division and the old orthogonality
 (\operatorname{tr}R=\operatorname{tr}(RX)=0) give the exact identity

\[
\boxed{
\chi=\frac1{c^2}\operatorname{tr}\{R(I+cX)^{-1}\}.}
\]

If

\[
R=\sum_{a=1}^{k_R}s_a v_a v_a^\top
\]

is its signed spectral/factor decomposition, then

\[
\chi=\frac1{c^2}\sum_{a=1}^{k_R}s_a
v_a^\top(I+cN^2)^{-1}v_a.
\]

Each term is a scalar Stieltjes quadratic form.  It can be evaluated by
 $m$-step Lanczos/CG using only applications of $N$ to vectors; no
matrix-matrix product, inverse, or additional transported word is required.
The exact common factor gate is the old-residual costate rank $k_R$, or its
value-weighted numerical rank for an approximation.

Put

\[
\kappa\le1+c\|N\|_{op}^2=1+C\theta_{\max}^2,qquad
q=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
\]

For $s_a\ge0$, zero-start CG/Gauss quadrature gives the quadratic-form
relative error

\[
0\le
\frac{v_a^\top A^{-1}v_a-v_a^\top y_{a,m}}
{v_a^\top A^{-1}v_a}
\le4q^{2m},qquad A=I+cN^2.
\]

For an indefinite costate the same bound is multiplied by the unavoidable
signed-cancellation condition number

\[
\kappa_{sign}
=\frac{\sum_a|s_a|v_a^\top A^{-1}v_a}
{\left|\sum_a s_a\,v_a^\top A^{-1}v_a\right|}.
\]

If this denominator is near zero, no orientation-free value guarantee exists.
Under the Padé diffuse gate $C\theta_{\max}^2\lesssim0.3$, $q$ is only about
$0.06$, so even a
very short deterministic Krylov rule is spectrally accurate.  The remaining
question is rank, not polynomial degree.

In arithmetic units, one local small-core matrix product bank costs
 $nr^3$, while one application of $N^2$ to $k_R$ vectors costs about
 $2nk_Rr^2$, a fraction $2k_R/r$ of a bank.  Across the ten-layer r55
suffix, using the prior 1.587-equivalent product-bank estimate, $m$ CG steps
cost approximately

\[
1.587\frac{2mk_R}{55}
\]

equivalents.  The 2.712 margin therefore permits roughly $k_R\le23$ for
 $m=2$, or $k_R\le15$ for $m=3$, before non-matmul billing and source
formation.  A full-rank costate should instead use the one-Cholesky circuit;
the Krylov route is specifically a low-rank endpoint exception.

The same observation makes polynomial actions cheap: $v^\top N^4v$ uses
two $N$-matvecs and $v^\top N^6v$ four.  Thus a rank-$k_R$ frozen Bellman
screen can test the exact degree-four/six actions without ever constructing
the corresponding matrix powers.

Scope is binding.  This theorem evaluates a same-site additive source against
an already compiled frozen costate.  It does not update the future AQ
trajectory, handle nonlinear coadaptation, or see q4-perp (which K31 is
identically blind to).  Those cases still require the exact secant trajectory
or a separate crossed-H4 provider/quotient.

## 2026-08-31 — q4-perp factor gradients and reverse-separability audit

For the exact commutator-angle action

\[
S=-8\sum_{i,j,k}\Lambda_{i,jk}
\langle[M_i,M_j],[M_i,M_k]\rangle_F,
\]

suppose first that each slice has a signed factorization

\[
\Lambda_i=\sum_{a=1}^k\lambda_{ia}u_{ia}u_{ia}^\top,
\qquad N_{ia}=\sum_j u_{ia,j}M_j.
\]

Then

\[
S=-8\sum_{i,a}\lambda_{ia}\|[M_i,N_{ia}]\|_F^2.
\]

Holding the factor coefficients fixed, put $C_{ia}=[M_i,N_{ia}]$.  The
exact symmetric-core gradient is

\[
\boxed{
\nabla_{M_p}S=-16\left\{
\sum_a\lambda_{pa}[[M_p,N_{pa}],N_{pa}]
+\sum_{i,a}\lambda_{ia}u_{ia,p}[M_i,[M_i,N_{ia}]]
\right\}.}
\]

This follows from

\[
\nabla_A\|[A,B]\|_F^2=2[[A,B],B],
\qquad
\nabla_B\|[A,B]\|_F^2=2[A,[A,B]].
\]

The deployability distinction is exact.  If the factors depend on the anchor
 $i$, forming all $N_{ia}$ needs $nk$ separate mixtures of $n$ core
matrices, costing $O(kn^3)$ when the flattened core dimension is $O(n)$.
If instead one global factor bank exists,

\[
\Lambda_i=U\Theta_iU^\top,
\qquad N_a=\sum_jU_{ja}M_j,
\]

the $N_a$ are formed once in $O(kn^2)$, followed by
 $O(nkr^3)$ local commutator products and small $k\times k$ contractions.
The exact minimal global width is still

\[
k_*=\operatorname{rank}[\Lambda_1\ \cdots\ \Lambda_n].
\]

For diagonal $\Theta_i=\operatorname{Diag}(\lambda_{ia})$, the gradient also
fuses globally.  Define

\[
H_a=\sum_i\lambda_{ia}[M_i,[M_i,N_a]].
\]

Then

\[
\nabla_{M_p}S=-16\left\{
\sum_a\lambda_{pa}[[M_p,N_a],N_a]
+\sum_aU_{pa}H_a\right\},
\]

so neither the action nor its joint-refit gradient requires anchor-specific
mixed cores.

There is a sharp reverse-structure audit.  A CP quartic costate

\[
H=\sum_a h_a u_a^{\otimes4}
\]

retains global factors under a common Gaussian leg map $K$, because

\[
(K^*)^{\otimes4}H
=\sum_a h_a(K^*u_a)^{\otimes4}.
\]

Thus a terminal diagonal costate transported only by common linear/OU legs
has a global $U$, although its rank may still be $n$.  A generic realized
ReLU adjoint
 $T_{g_i}=P_{\le2}M_{1_{g_i>0}}P_{\le2}$ is neuron-specific and is not a
fourfold common-leg intertwiner.  It generically makes the slice factor spaces
depend on $i$, restoring the $O(kn^3)$ formation cost and Burnside growth.

Therefore a resident D10 word proves cheap q4-perp acquisition only if its
actual symbolic recurrence exhibits a common $U$/CP/OU factorization at
every reset.  The fact that it is billed as one word is not such a proof.  The
reopen certificate is the explicit identity

\[
\operatorname{range}\Lambda_i\subseteq\operatorname{range}U
\quad\text{for every anchor and layer},
\]

plus $k_*$ within the remaining cost allowance; failure at one ReLU reset
closes the global-preaggregation shortcut.

### Exact zero-GEMM commutator readout from a paid Jordan product

There is one concrete way around the otherwise fatal local commutator-product
charge.  For symmetric $M,N_a$, define

\[
P_a=MN_a,
\qquad J_a=M\circ N_a=\frac{P_a+P_a^\top}{2},
\qquad A_a=\frac{P_a-P_a^\top}{2}=\frac12[M,N_a].
\]

The symmetric and skew parts are Frobenius-orthogonal, so

\[
\boxed{
\langle[M,N_a],[M,N_b]\rangle_F
=4\{\langle P_a,P_b\rangle_F-\langle J_a,J_b\rangle_F\}
=4\langle A_a,A_b\rangle_F.}
\]

Therefore, if the D14 base word and D10 suffix word already pay for the full
product $P=MN$ in order to form the Jordan reset $J=M\circ N$, the
q4-perp commutator-angle scalar costs no additional dense product.  One simply
retains the antisymmetric half before symmetrization and performs Frobenius
reductions.  In the rank-one case,

\[
-8\lambda\|[M,N]\|_F^2=-32\lambda\|A\|_F^2.
\]

For a $k$-factor common bank, all pair actions use the small Gram
 $(\langle A_a,A_b\rangle)_{ab}$.  This removes the extra local-GEMM
objection **only for factor directions already represented by the paid
bundles**.  It does not create the global factor $U$, lower $k_*$, or fix
anchor-dependent ReLU reverse slices.  The exact implementation receipt is:

1. the counted Jordan path literally forms $P_a=M N_a$ at the same seam;
2. $P_a$ remains available before destructive symmetrization;
3. the residual q4-perp coefficient $\Theta_i$ contracts the resulting skew
   Gram after complete old subtraction.

If these hold, local q4-perp consumption can fit inside the one-equivalent
slack even though independent commutator products cannot.  If any fails, the
previous $O(nkr^3)$ local-product charge remains binding.

This fusion is complete only for the commutator-angle q4-perp scalar.  It does
not by itself close the full signed-H3 seam.  The complete product inventory
also contains anchor squares and factor--factor products: K4 needs (M^2),
K31 needs (M^2,MN), K22 needs (MN,N^2), and repeated-anchor q211 H3 can
need (M_iN_a,M_iN_b,N_aN_b).  Retaining the full (P=MN) jointly serves the
H3 Jordan half and q211 skew half, but (N_aN_b) and (M_i^2) require their
own literal-ownership or source-specific factorization receipts.  Therefore
zero-GEMM q4-perp does not imply zero-GEMM complete seam closure.

## 2026-08-31 — exact generalized-chi-square reset as a resolvent integral

The matrix-polynomial approximation is not mathematically necessary for the
H2 reset itself.  For

\[
X_M=z^\top Mz-\operatorname{tr}M,
\]

the exact characteristic function is

\[
\varphi_M(t)
=e^{-it\operatorname{tr}M}\det(I-2itM)^{-1/2}.
\]

The tilted Gaussian covariance identity gives

\[
\mathbb E[zz^\top e^{itX_M}]
=\varphi_M(t)(I-2itM)^{-1}.
\]

Consequently, with
 $B_M=\mathbb E[zz^\top\delta(X_M)]$ and
 $p_M=\Pr(X_M>0)$, Fourier inversion and Gil--Pelaez give

\[
\boxed{
B_M=\frac1\pi\operatorname{Re}\int_0^\infty
\varphi_M(t)(I-2itM)^{-1}\,dt,}
\]

\[
\boxed{
p_M=\frac12+\frac1\pi\int_0^\infty
\frac{\operatorname{Im}\varphi_M(t)}{t}\,dt.}
\]

Substitution into the exact second-chaos projection yields

\[
\boxed{
A_{\operatorname{ReLU}}(M)=p_MM+2MB_MM.}
\]

If $M=U\operatorname{Diag}(\lambda_j)U^\top$, then $B_M$ and $A(M)$
share the same eigenvectors and

\[
b_j=\frac1\pi\operatorname{Re}\int_0^\infty
\frac{\varphi_M(t)}{1-2it\lambda_j}\,dt,
\qquad
a_j=p_M\lambda_j+2\lambda_j^2b_j.
\]

Thus one eigendecomposition supplies the exact reset eigenvalues through a
shared scalar quadrature.  No explicit $M^4,M^6,\ldots$ states are needed,
and all resolvent nodes are diagonal evaluations after the same eigensolve.
The state remains one $r\times r$ matrix per core.

The integral has a useful analytic tail certificate:

\[
|\varphi_M(t)|
=\prod_j(1+4t^2\lambda_j^2)^{-1/4}.
\]

If at least $m>2$ eigenvalues obey $|\lambda_j|\ge a>0$, then for
 $T\ge(2a)^{-1}$,

\[
\left\|\frac1\pi\int_T^\infty
\varphi_M(t)(I-2itM)^{-1}\,dt\right\|_{op}
\le
\frac{(2a)^{-m/2}T^{1-m/2}}
{\pi(m/2-1)},
\]

and the corresponding probability-integral tail is at most

\[
\frac{(2a)^{-m/2}T^{-m/2}}{\pi(m/2)}.
\]

Diffuseness therefore helps twice: many active eigenvalues force rapid
large-$t$ decay, while small $\|M\|_{op}$ pushes the nearest complex
resolvent singularity $t=(2i\lambda_j)^{-1}$ away from the real axis and
accelerates quadrature on the finite interval.

With $r^2\asymp n$, local eigendecomposition of all $n$ cores costs
 $O(nr^3)=O(n^{5/2})$, asymptotically below one $O(n^3)$ dense transport;
the post-eigenvalue quadrature is only $O(kn r)$ for $k$ nodes.  This is a
mathematical complexity result, not a deployed billing receipt.  It becomes
an actionable exact-H2 reset route if the eigensolver/quadrature census fits
the remaining allowance or is already resident/fused.

The scope limit is exact: $A_{\operatorname{ReLU}}(M)$ is the optimal H2
projection and cannot recover the irreducible H4 commutator-angle sector.
Therefore this resolvent route can eliminate matrix-polynomial truncation as
an H2 ingredient error, but it cannot by itself solve q4-perp or exceed the
previous $q_{\max}$ ceiling.  Its promotion gates are: nonduplicate relative
to the actual Q24 reset, enough Bellman value in the H2 reset residual, a
quadrature/eigensolve billing receipt, and independent treatment or certified
darkness of H4.

### Exact noncentral affine-quadratic ReLU reset

The resolvent construction extends without approximation to the full AQ
preactivation

\[
g(z)=m+u^\top z+z^\top Mz-\operatorname{tr}M.
\]

Define

\[
D(t)=I-2itM,qquad C(t)=D(t)^{-1},qquad
\mu(t)=it\,C(t)u.
\]

Completing the Gaussian square gives

\[
\boxed{
\varphi_g(t)=
e^{it(m-\operatorname{tr}M)}det D(t)^{-1/2}
\exp\left\{-\frac{t^2}{2}u^\top C(t)u\right\},}
\]

\[
\mathbb E[ze^{itg}]=\varphi_g(t)\mu(t),
\qquad
\mathbb E[zz^\top e^{itg}]
=\varphi_g(t)\{C(t)+\mu(t)\mu(t)^\top\}.
\]

Let the boundary moments be

\[
S_0=\mathbb E\delta(g),\qquad
S_1=\mathbb E[z\delta(g)],\qquad
S_2=\mathbb E[zz^\top\delta(g)].
\]

They are obtained from one shared quadrature:

\[
S_0=\frac1\pi\operatorname{Re}\int_0^\infty\varphi_g(t)\,dt,
\]

\[
S_1=\frac1\pi\operatorname{Re}\int_0^\infty
\varphi_g(t)\mu(t)\,dt,
\]

\[
S_2=\frac1\pi\operatorname{Re}\int_0^\infty
\varphi_g(t)\{C(t)+\mu(t)\mu(t)^\top\}\,dt.
\]

Also

\[
p=\Pr(g>0)
=\frac12+\frac1\pi\int_0^\infty
\frac{\operatorname{Im}\varphi_g(t)}{t}\,dt,
\]

and, since $\mathbb Eg=m$,

\[
m_+=\mathbb E(g)_+
=\frac m2+\frac1\pi\int_0^\infty
\frac{1-\operatorname{Re}\varphi_g(t)}{t^2}\,dt.
\]

Write the exact AQ projection as

\[
P_{\le2}(g)_+
=m_++u_+^\top z+z^\top M_+z-\operatorname{tr}M_+.
\]

Gaussian integration by parts gives a closed formula from the same boundary
moments.  Put

\[
h=S_0u+2MS_1.
\]

Then

\[
\boxed{u_+=pu+2Mh,}
\]

\[
\boxed{
M_+=pM+\frac12\left{
S_0uu^\top+2uS_1^\top M+2MS_1u^\top+4MS_2M
\right}.}
\]

The proof is the first/second Stein identities

\[
\mathbb E[z(g)_+]=\mathbb E[1_{g>0}(u+2Mz)],
\]

\[
\frac12\mathbb E[(g)_+(zz^\top-I)]
=pM+\frac12\mathbb E[\delta(g)(u+2Mz)(u+2Mz)^\top].
\]

After diagonalizing $M$, every $C(t)$ is diagonal in the same basis and
 $\mu(t)\mu(t)^\top$ is rank one.  A $k$-node evaluation therefore costs
 $O(r^3+kr^2)$ per core, or $O(nr^3+nkr^2)$ across the layer, while
returning the exact H0/H1/H2 reset up to certified one-dimensional quadrature
error.  This is the strongest spectral-reset candidate in this lane: it
subsumes every finite matrix polynomial and the one-pole Padé for the AQ
projection.  It remains conditional on being nonduplicate with the deployed
Q24 reset and on the local eigensolve/quadrature fitting billing; it still does
not carry irreducible H4.

## 2026-08-31 — synthesized candidate: self-consistent NCQC Galerkin engine

The preceding pieces close into one end-to-end architecture that does not
inject a frozen correction into D11.  Use a common latent Gaussian
 $z\in\mathbb R^r$ and represent every unit at a cut by

\[
h_i=m_i+I_2(M_i)
=m_i+z^\top M_i z-\operatorname{tr}M_i,
\qquad M_i\in\operatorname{Sym}_r.
\]

For a dense layer $W$, transport is exactly one-sided in the unit index:

\[
\widetilde m_a=b_a+\sum_iW_{ia}m_i,
\qquad
\widetilde M_a=\sum_iW_{ia}M_i.
\]

At the reset, replace each $(\widetilde m_a,\widetilde M_a)$ by the exact
H0/H2 projection of $(\widetilde h_a)_+$ from the resolvent formulas above:

\[
(m_a^+,M_a^+)=\mathcal R_{AQ}(\widetilde m_a,\widetilde M_a).
\]

This defines a closed nonlinear recurrence on the core bank.  It automatically
retunes every future covariance and cumulant to the changed reset; there is no
hand injection and no frozen-beta transfer.  Matrix powers are not separately
propagated: the exact spectral reset compresses their complete H2 effect back
into one new core $M_a^+$ at every layer.

All required second moments are native Gram contractions:

\[
\mathbb Eh_i=m_i,
\qquad
\operatorname{Cov}(h_i,h_j)=2\operatorname{tr}(M_iM_j).
\]

This does not impose a covariance-rank loss at the campaign dimensions.  Let
 $d=r(r+1)/2$.  For every positive semidefinite
 $C\in\mathbb R^{n\times n}$ with $d\ge\operatorname{rank}C$, choose
 $L\in\mathbb R^{n\times d}$ such that $2LL^\top=C$, and map the $i$th
row of $L$ isometrically through `svec` to $M_i\in\operatorname{Sym}_r$.
Then

\[
2\operatorname{tr}(M_iM_j)=C_{ij}
\]

exactly.  At $r=55$, $d=1540>1024=n$, so every deployed covariance has a
pure-core representation.  Moreover, dense transport preserves it exactly:

\[
2(W^\top L)(W^\top L)^\top=W^\top C W.
\]

The one-sided core mix is therefore an exact factorized replacement for the
two-sided covariance conjugation, not a low-rank approximation.  The modeling
question begins at signed $K_3$, reset spectra, and H4—not at covariance.

For a scalar endpoint $y=c+\sum_i v_i h_i$, put

\[
M_y=\sum_i v_iM_i.
\]

Then the predicted endpoint moments are obtained without forming an
 $n\times n$ covariance:

\[
\boxed{
\mathbb Ey=c+v^\top m,
\qquad
\operatorname{Var}y=2\|M_y\|_F^2.}
\]

The same cores generate their signed non-Gaussian structure coherently:

\[
\kappa_3(h_i,h_j,h_k)=8\operatorname{tr}(M_iM_jM_k),
\]

and the fourth cumulant contains both the commuting trace word and the exact
commutator-angle q4-perp word.  Thus the engine retains identity-specific
broad-spectrum NCQC geometry; unlike the failed collective tracker, it does
not replace the plane by rank one.  It also never transports a separate q4
tensor: the q4 implied by the current cores is regenerated at consumption.

### Arithmetic capacity

Flatten the symmetric cores into
 $A\in\mathbb R^{n\times d}$, $d=r(r+1)/2$.  One layer transport is the
rectangular product $W^\top A$, costing $d/n$ square-GEMM equivalents.
For $n=1024,r=55,d=1540$, fourteen transports cost

\[
14\frac d n\simeq21.05.
\]

Using the already-derived AQ spectral census, fourteen local eigensystem
passes cost approximately

\[
14\frac{r^3}{n^2}\simeq2.22
\]

before constants, and $K$ shared quadrature nodes add approximately

\[
14K\frac{r^2}{n^2}\simeq0.0404K.
\]

Even one optional terminal core Gram costs only $d/n\simeq1.50$, although
the scalar endpoint formula avoids it.  Thus the pure-core recurrence has a
sub-49 arithmetic path with substantial room for certified quadrature.  This
is not yet a deployed billing receipt: eigensolver constants, f64 charging,
memory layout, and any retained D11 side state must be counted.  The proposal
is a *replacement engine*, so old covariance/K31/q4 transports cannot also be
charged unless independently required.

### Exact approximation loss and fail-fast gate

The only intrinsic local loss is the irreducible reset remainder

\[
e_i=(I-P_{\le2})(h_i)_+.
\]

Its energy is computable from the same scalar law:

\[
\|e_i\|_2^2
=\mathbb E(h_i)_+^2-(m_i^+)^2-2\|M_i^+\|_F^2.
\]

For vector $L^2$ norm and exact weight transport, nonexpansiveness gives the
rigorous forward bound

\[
E_{\ell+1}
\le\|W_{\ell+1}\|_{op}E_\ell+\|e_{\ell+1}\|_{L^2(\ell_2)},
\]

and hence the usual product-weighted sum of discarded reset energies at the
endpoint.  The sharper binding test is the complete old-residualized Bellman
action of these $e_\ell$.  The engine can reach correlation $\rho$ only if
the value-visible discarded fraction is at most $1-\rho^2$, namely
 $0.14381991$ or $0.03280643$ for the two campaign gates.

This is exactly where H4 enters: the engine regenerates q4 from its current H2
cores but discards inherited irreducible H4 at every reset.  If the H4
Bellman action exceeds the remaining error budget, no matrix-only Galerkin
closure can win and the paid D10 q4-perp suffix is necessary.  If it is dark or
fits the D10 fused antisymmetric readout, the core engine is closed.

The discarded object has an exact plane interpretation.  Write

\[
(h_i)_+=m_i^++I_2(M_i^+)+e_i,
\qquad e_i\perp H_{\le2}.
\]

Then the true post-reset covariance splits orthogonally as

\[
\boxed{
C^+_{ij}=2\operatorname{tr}(M_i^+M_j^+)+E^{H4+}_{ij},
\qquad E^{H4+}_{ij}=\langle e_i,e_j\rangle.}
\]

The omitted plane $E^{H4+}$ is positive semidefinite, and its diagonal is
exactly the locally computable residual energy above.  This identifies the
late variance-channel carrier: it is the future-visible part of a residual
Gram, not an arbitrary fourth-order tensor.  Its off-diagonal entries contain
the commutator-angle/q4-perp information that individual spectra cannot
recover.

This yields a natural hybrid under the cap.  The D14 base is the one-sided
NCQC core bank, costing about $21.05$, rather than a two-sided $28$-cost
plane.  A full D10 symmetric residual-plane suffix costs $20$, so the two
states total about $41.05$ before the roughly $2.22$ local spectral factor
and quadrature.  On arithmetic counts this leaves materially more slack than
the old $28+20=48$ schedule.  The D10 source must be the complete-old-
residualized $E^{H4+}$ action (or an exact fused equivalent), and its reset
transition still needs the two-bundle invariance/leakage certificate; PSD at
birth does not prove closure through a correlated ReLU gate.  But it gives the
correct state, sign, and cost decomposition for the variance-dominant lane.

### Identifiability and training law

Static $C,K_3$ matching leaves a large inverse fiber of core representatives.
Representatives on that fiber have different spectra and therefore different
exact reset maps.  This explains why frozen per-net injections and frozen
betas fail: they choose a representative without future reset value.  The
lawful fit is joint across depth/end value:

1. initialize cores from the NCQC $C,K_3$ fiber;
2. propagate them through the exact AQ reset recurrence;
3. optimize the core representative and any shared calibration only through
   the end-to-end training loss, with the F2J machinery;
4. hold the architecture and all coefficients fixed for the sentinel and
   transfer folds.

Initialization should occur **after** the first exactly Gaussian ReLU reset,
not by forcing the Gaussian input field itself into pure quadratic cores.  A
quadratic-core factor can match any covariance when (d\ge n), but it
generically introduces nonzero signed (K_3).  An exact zero-(K_3)
construction uses a bipartite off-diagonal symmetric subspace

\[
M_X=\begin{pmatrix}0&X\\X^\top&0\end{pmatrix},
\qquad \operatorname{tr}(M_XM_YM_Z)=0,
\]

whose maximum dimension at (r=55) is only
 (lfloor r^2/4\rfloor=756<1024).  Thus covariance capacity alone does
not certify a faithful Gaussian input representation at r55.  The lawful
route is: compute the first Gaussian reset by the exact existing kernel, then
fit the first NCQC core bank to the resulting non-Gaussian (C,K_3) state and
start the Galerkin recurrence there.  This also avoids spending core capacity
on a sector the Gaussian machinery already solves exactly.

The proof-level close conditions are: no stable NCQC initialization;
value-visible AQ remainder above budget; full-old projection annihilates the
core correction; or the exact replacement census exceeds 49.  The reopen/GO
conditions are a stable jointly fitted core trajectory, nonzero held-out
Bellman residual, certified AQ reset quadrature, and endpoint raw gain under
the fixed sentinel gate.  Of the constructions in this lane, this is the only
one that simultaneously provides broad-spectrum identity-specific supply,
automatic coadaptation, exact generalized-chi-square H2 reset, and an
arithmetic path below the cap.

### PSD Bellman theorem for the omitted variance Gram

The variance channel has a structural sign certificate.  Let

\[
G_{ij}=\mathbb E[1_{\{g_i>0\}}1_{\{g_j>0\}}].
\]

This is a Gram matrix and hence $G\succeq0$.  Price's theorem for an
additive Gaussian covariance perturbation gives the ReLU covariance tangent

\[
H\longmapsto G\odot H.
\]

Therefore, if the next-cut covariance costate is $\Lambda_{l+1}$, the
reverse covariance recursion is

\[
\boxed{
\Lambda_l
=G_l\odot(W_{l+1}\Lambda_{l+1}W_{l+1}^\top).}
\]

Starting from a scalar variance endpoint
 $\Lambda_L=vv^\top\succeq0$, congruence and the Schur product theorem
prove inductively that

\[
\Lambda_l\succeq0
\qquad\text{at every layer}.
\]

Since the omitted reset source is also a Gram,
 $E_l^{H4+}\succeq0$, its exact frozen Bellman action satisfies

\[
\boxed{
\langle\Lambda_l,E_l^{H4+}\rangle\ge0.}
\]

More explicitly, if
 $\Lambda_l=\sum_a\sigma_av_av_a^\top$ with $\sigma_a\ge0$, then

\[
\langle\Lambda_l,E_l^{H4+}\rangle
=\sum_a\sigma_a\left\|\sum_i v_{a,i}e_i\right\|_2^2.
\]

Thus the lawful H4 residual cannot exhibit a true negative transfer
coefficient in the frozen variance tangent.  Sign flips seen in compressed
carriers diagnose supply error, old-span contamination, site mismatch, or a
non-variance channel—not the exact residual-Gram correction.  This is stronger
than a generic low-rank prior and provides a natural one-sided sentinel test.

The qualification is precise: the theorem is exact for the covariance
Bellman tangent with fixed gates/base trajectory.  Nonlinear coadaptation and
coupled K31/K22 objectives may add indefinite costate components, and full-old
residualization can remove PSD pieces.  The graph should therefore record
positivity on the pure variance child, not on the entire adjusted-score
functional.

### Exact raw-MSE target gate

Let (r=y-\widehat y_{old}) be the old prediction residual after projecting
out every already-owned calibration column, and let
 (z=\widehat y_{core}-\widehat y_{old}) be the similarly residualized new
engine correction.  An optimal single frozen scale gives

\[
\min_\alpha\|r-\alpha z\|_2^2
=\|r\|_2^2\left(1-\rho_{partial}^2\right),
\qquad
\rho_{partial}=\frac{\langle r,z\rangle}{\|r\|\|z\|}.
\]

Starting from raw (2.9409\times10^{-8}), reaching raw (10^{-8}) therefore
requires

\[
\boxed{
|\rho_{partial}|
>\sqrt{1-\frac{1.0}{2.9409}}
\simeq0.8124.}
\]

Reaching (6\times10^{-9}) requires approximately
 (|\rho_{partial}|>0.8922).  These are correction-to-old-residual
correlations, not local reset correlations.  The analytic 0.9253/0.98346
closure gates are deliberately stronger ingredient conditions, but neither
PSD sign nor local spectral accuracy alone proves the necessary (0.8124)
end-to-end partial correlation.

This is the correct fail-fast sentinel for the Galerkin engine: fit all
allowed coefficients on the fixed training side, freeze them, and compute the
single correction column on the sentinel.  Below $0.8124$ it cannot reach
the campaign raw goal by scalar cofire even at zero utility cost; above it,
the raw-$10^{-8}$ path is mathematically open subject to amplitude stability
on the remaining folds.

### Core-only ceiling and why the H4 residual plane is essential

The exact AQ reset removes polynomial error but does **not** make the pure-core
engine a complete ReLU closure.  In the diffuse limit the normalized quadratic
form tends to a scalar Gaussian $Z$.  The centered ReLU energies begin

\[
e_1=\frac14,
\qquad e_2=\frac1{4\pi},
\qquad V=\operatorname{Var}(Z_+)=\frac12-\frac1{2\pi}.
\]

The scalar H1 component is an original H2/core direction, so the pure-core
correlation tends to only

\[
\boxed{
\rho_{core}\longrightarrow\sqrt{e_1/V}\simeq0.856.}
\]

The scalar H2 component is the square lift.  If
 $\tau=\operatorname{tr}M^4/(\operatorname{tr}M^2)^2$, only the fraction

\[
s_2(\tau)=\frac{4\tau}{1+6\tau}
\]

of that component contracts back into $I_2(M^2)$; the remainder is genuine
H4.  At diffuse r55 scale this shadow is only about (6.6\%\).  Keeping every
matrix polynomial improves the finite-rank number but does not change the
diffuse $0.856$ limit.

By contrast, adding the complete-old-residualized H4 square-lift Gram restores
the rest of $e_2$ without double counting.  The combined one-reset Gaussian
surrogate reaches

\[
\boxed{
\rho_{core+H4}
=\sqrt{\frac{e_1+e_2}{V}}
\simeq0.98333.}
\]

This pins down the architecture decision.  The NCQC Galerkin core is the
correct cheap, coadapted base carrier, but core-only is generically closed for
the 0.925/0.983 ingredient gates.  The late PSD H4 residual plane is not an
optional embellishment; it is the object that upgrades the base from the H1
ceiling to the H1+H2 ReLU ceiling.  Higher scalar H4/H6 sectors and depth
coherence still determine whether the end-to-end target is met.

For the leading square-lift residual, the plane entry is explicitly

\[
\langle U_{M_i},U_{M_j}\rangle
=\operatorname{tr}(M_iM_jM_iM_j)
+\frac12\operatorname{tr}(M_iM_j)^2
\]

\[
=\operatorname{tr}(M_i^2M_j^2)
+\frac12\operatorname{tr}(M_iM_j)^2
-\frac12\|[M_i,M_j]\|_F^2.
\]

The first two terms are the commuting/star Gram; the last is exactly the
noncommutative q4-perp correction.  Hence a commuting or CP/star core makes
the new D10 source vanish, while the paid-Jordan antisymmetric readout computes
the missing term without a new local product whenever the relevant mixed
directions are already present.  This identity connects the H4 energy ceiling,
the commutator carrier, and the zero-GEMM fusion in a single plane formula.

## 2026-08-31 — sharp r45 capacity threshold and variable-width schedule

At any cut there are only (n) unitwise H2 coefficient tensors.  Let their
whitened Gram be flat of rank (n), the worst case for compression.  A
 (operatorname{Sym}_r) core has feature dimension

\[
d_r=\frac{r(r+1)}2.
\]

No linear/isometric core embedding can retain more than the fraction
 (d_r/n) of a flat H2 target.  Therefore correlation (\rho) requires

\[
\boxed{d_r\ge\lceil\rho^2n\rceil.}
\]

At (n=1024):

- (\rho=0.9253) needs at least (877) directions, so (r=42)
  ((d_{42}=903)) is the minimum while (r=41) ((861)) fails;
- (\rho=0.98346) needs at least (991) directions, so (r=45)
  ((d_{45}=1035)) is the minimum;
- (r=44) has only (d_{44}=990) and the sharp flat-target ceiling

  \[
  \sqrt{990/1024}\simeq0.98326<0.98346.
  \]

Thus r45 is the exact minimal high-fidelity matrix-core size under a worst-case
flat H2 target.  r55 adds inverse-fiber/provider flexibility but is not the
deployment-optimal capacity point.

This yields a variable-width cost schedule tailored to the observed L5--14
value window.  Carry the H1/Gaussian factor for all fourteen transitions, but
activate the new H2/NCQC core and H4 residual plane only for the ten late
transitions.  In square-GEMM arithmetic:

\[
C_{H1}=14,
\qquad
C_{H2}=10\frac{1035}{1024}\simeq10.11,
\qquad
C_{H4plane}=20.
\]

The transport total is therefore about

\[
\boxed{44.11,}
\]

leaving about (4.89) under the 49 cap.  Ten r45 local spectral
factorizations cost about
 (10r^3/n^2\simeq0.87), and (K) quadrature nodes add approximately
 (10Kr^2/n^2\simeq0.0193K).  This leaves room for a modest certified
quadrature in arithmetic units.  Equivalently, the H1 and late H2 coefficient
banks can be viewed as one base rectangular stream whose width grows from
 (n) to (n+d_{45}) at L5, plus one D10 symmetric residual plane.

Two premises remain open.  First, the early nonlinear H2 state must be
complete-old-owned or Bellman-dark before L5; the value-map attribution makes
this plausible but does not prove it.  Second, an exact reset of an H1 factor
plus compressed H2 core generates new H2 tensors outside the current embedded
subspace.  Because there are only (n) unit outputs and (d_{45}\ge n), their
Gram can be re-embedded losslessly at each cut, but the embedding's Jordan/
signed-K3 orientation is not determined by the Gram.  The F2J joint refit must
choose that moving fiber representative.  This is the precise coadaptation
problem, not a covariance-rank problem.

## 2026-08-31 — local fidelity does not compose for free: exact Duhamel gate

Let the complete-old-projected AQ state error satisfy the normalized causal
recurrence

\[
\delta_{l+1}=T_lW_l\delta_l+s_l,
\qquad \|T_l\|_{\mathrm{task}}\le 1,
\]

where \(s_l\) is the local omitted reset innovation and
\(\kappa_l=\|W_l\|_{\mathrm{task}}\).  If the retained local subspace has
task-weighted fidelity \(q_l\) on the relevant source \(f_l\), then

\[
\|s_l\|=\sqrt{1-q_l^2}\,\|f_l\|.
\]

Unrolling the recurrence gives the deterministic finite-horizon bound

\[
\boxed{
\|\delta_L\|
\le
\sum_l\left(\prod_{j>l}\kappa_j\right)
\sqrt{1-q_l^2}\,\|f_l\| .}
\]

This is the correct bridge from a local H2/H4 correlation calculation to an
endpoint raw-MSE claim.  If \(S\) is the norm of the endpoint target and the
right-hand side divided by \(S\) is \(\eta\), the worst-case endpoint
correlation is at least \(\sqrt{1-\eta^2}\).  Since the current raw-MSE goal
requires partial correlation \(>0.8124\), the certified gate is

\[
\boxed{\eta<\sqrt{1-0.8124^2}\simeq0.583.}
\]

If the propagated innovations are mutually orthogonal in the endpoint Gram
(or form a martingale-difference sequence in that metric), the triangle sum
can be replaced by the root-sum-square of the same weighted terms.  That
improvement requires an explicit endpoint-Gram orthogonality certificate; it
cannot be inferred from layerwise Hermite orthogonality because different
reset/transport maps mix the sectors.

For the scalar surrogate fidelity \(q\simeq0.98333\) of core plus complete
H4 square lift, the local omitted amplitude is
\(\sqrt{1-q^2}\simeq0.1291\).  Across ten equally weighted coherent
exposures the deterministic gate therefore requires the sum of downstream
gain/source ratios to be below about \(0.583/0.1291=4.51\).  Under certified
orthogonality, the corresponding root-sum-square must be below 4.51.  Thus
the attractive local 0.983 number is not by itself an end-to-end GO: the
remaining graph node must certify the propagated source weights and their
coherence.  This also gives a fail-fast close condition for the r45/H4
schedule before any numerical campaign.

## 2026-08-31 — exact D11-interface audit of the analytic star fiber

The lawful two-core fiber from the cubic-provider lane fixes every trace word
of degree at most three while varying the star edge squares by

\[
(\dot X_2,\dot X_3,\dot X_4)=(2,-3,1),
\qquad A=\operatorname{diag}(5,6,7,9),
\qquad B=dI+S.
\]

It proves lower-moment independence of q211, but it is not dark to the full
quartic interface.  Directly,

\[
\begin{aligned}
D\operatorname{tr}(A^2B^2)
 &= (25+36)2+(25+49)(-3)+(25+81)1=6,\\
D\operatorname{tr}(ABAB)
 &=2(5)\{6(2)+7(-3)+9(1)\}=0.
\end{aligned}
\]

Consequently

\[
D\|[A,B]\|_F^2
=2D\{\operatorname{tr}(A^2B^2)-\operatorname{tr}(ABAB)\}=12.
\]

For centered Gaussian quadratic forms, the repeated-pair fourth cumulant is

\[
q_4(A,A,B,B)
=32\operatorname{tr}(A^2B^2)+16\operatorname{tr}(ABAB)
=48\operatorname{tr}(A^2B^2)-8\|[A,B]\|_F^2.
\]

All covariance pairings are fixed along the fiber, so the true K22 moment has
the same derivative

\[
\boxed{D K22(A,B)=192.}
\]

The canonical commuting/star part has derivative \(48(6)=288\), while the
noncommutative q211-perp correction has derivative \(-8(12)=-96\).  Their
sum is 192.  The marginal K4 and both K31 collision faces remain fixed: the
star graph is bipartite, its odd closed walks vanish, and the constraints
already fix the needed lower trace weights.

Therefore this tangent cannot certify complete-D11-interface darkness.  A
CP4 constructed only from the fixed covariance/cubic data annihilates it; an
exact star factor containing \(\operatorname{tr}(A^2B^2)\) sees 288; and the
true K22 plane sees 192.  Exact star subtraction still leaves the novel
q211 residual \(-96\).  Calling that residual old-owned requires a literal
receipt that D11 consumes the full K22/Z22 residual rather than only its CP4
and pairing projections.

## 2026-08-31 — correction: r45 closes H2, but r55 is the sharp generic C+K21 rank

The r45 theorem above is a covariance/H2 capacity theorem, not a complete
NCQC state theorem.  A core tuple has, modulo common orthogonal conjugacy,

\[
p_r=1024\frac{r(r+1)}2-\frac{r(r-1)}2
\]

local degrees of freedom, whereas the complete covariance plus repeated
signed-cubic interface has

\[
q=\frac{1024\cdot1025}{2}+1024^2=1,573,376
\]

coordinates.  The relevant values are

\[
\begin{array}{c|c|c|c}
r & d_r & p_r & p_r-q\\ \hline
45&1035&1,058,850&-514,526\\
54&1485&1,519,209&-54,167\\
55&1540&1,575,475&+2,099.
\end{array}
\]

Thus r55, not r45, is the sharp dimension threshold for a locally generic
joint \((C,K21)\) provider.  On a flat joint target, any rank-\(p_r\)
tangent model has the worst-case energy/correlation ceiling

\[
\rho_r^2\le\frac{p_r}{q}.
\]

In particular r54 has

\[
\rho_{54}\le\sqrt{1-54167/1573376}\simeq0.98264<0.98346,
\]

while r45 has only \(\rho_{45}\simeq0.820\).  Therefore the earlier r45
schedule is lawful only if the missing signed-cubic normal bundle is already
owned, Bellman-dark, or strongly target-underweighted.  It must not be called
a generic high-fidelity NCQC closure.

There is, however, a sharp target-conditioned r54 exception.  If
\(P_{54}\) projects onto the actual differential range of the r54
\((C,K21)\) map in the task metric, then r54 reaches a desired correlation
\(\rho_*\) if and only if

\[
\boxed{
\frac{\|(I-P_{54})g\|^2}{\|g\|^2}\le1-\rho_*^2,}
\]

where \(g\) is the complete-old-residual endpoint covector pulled back to
this interface.  For \(\rho_*=0.98346\), the allowed tail is
\(0.0328064\); the flat r54 tail is about \(0.034427\).  Hence r54 misses
the high gate by only about \(0.00162\) of total target energy.  A modest,
explicit target underweighting of its 54,167-dimensional normal bundle would
reopen it; raw dimension rhetoric does not.

### 49-equivalent schedules after the correction

The generic r55 core costs \(10(1540/1024)=15.0391\) for ten late
transports.  A ten-layer eigendecomposition costs about 1.5867, and each
shared quadrature node over all ten layers costs about 0.02885.  Keeping the
H1/D14 base and only nine of ten D10 H4 residual-plane injections gives

\[
C_{55,9H4}(K)
=14+15.0391+18+1.5867+0.02885K.
\]

With \(K=12\), this is approximately

\[
\boxed{48.972,}
\]

leaving only 0.028 equivalent: mathematically feasible, operationally
knife-edge, and with no room for an unfused acquisition.  Because the H4
Bellman covariance contributions are PSD, dropping the least valuable of ten
injections loses at most one tenth of their total value; the omitted layer
must be selected by the frozen value weights, not tuned on the sentinel.

The conditional r54 alternative has substantially better economics:

\[
C_{54,9H4}(K)
=14+10(1485/1024)+18
+10(54^3/1024^2)+10K(54^2/1024^2).
\]

The fixed part is about 48.004, so roughly 35 quadrature nodes fit under 49.
Its exact reopen certificate is the target-tail inequality above.  The
decision is therefore crisp: r55 is generic but knife-edge; r54 is deployable
with useful reset accuracy only if its complete-old-residual normal-tail
energy is at most 0.0328064.

The word “modest” in the preceding energy comparison must not be confused
with “generic.”  Under an isotropic endpoint-covector null, the r54 normal
tail has the exact beta law

\[
\|(I-P_{54})g\|^2/\|g\|^2
\sim\operatorname{Beta}(54167/2,1519209/2).
\]

Its mean is about 0.034427 and its standard deviation is about

\[
\sqrt{\frac{2(0.034427)(0.965573)}{1573378}}
\simeq2.06\times10^{-4}.
\]

The 0.0328064 gate is therefore roughly 7.9 standard deviations below the
isotropic mean.  Random orientation will not rescue r54; reopening requires
a structural Bellman alignment with the r54 tangent range.  This converts
the r54 branch from a vague “near miss” into a precise symmetry/ownership
question.

## 2026-08-31 — channel-conditioned rank selection: r50 is the first plausible full-D10 point

The missing dimensions for every \(r\le54\) lie in the conditional signed
cubic channel, not in covariance.  Indeed \(d_r\ge1024\) for all
\(r\ge45\), so at a full-row-rank covariance factor the covariance map is a
submersion.  After fixing covariance, the available quotient-fiber dimension
is \(p_r-524800\); hence the generic conditional K21 normal codimension is

\[
m_r=q-p_r.
\]

Let \(g=(g_C,g_K)\) be the complete-old-residual endpoint covector after
Schur-orthogonalizing K21 against covariance, and define

\[
\theta_K=\frac{\|g_K\|^2}{\|g\|^2},
\qquad
\lambda_r=\frac{\|P_{N_r}g_K\|^2}{\|g_K\|^2},
\]

where \(N_r\) is the normal space of
\(D K21|_{\ker DC}\).  The exact target-tail gate factors as

\[
\boxed{\theta_K\lambda_r\le1-\rho_*^2.}
\]

This separates two physically different questions: how much endpoint energy
is actually in the signed-cubic channel, and how much of that channel misses
the rank-r NCQC tangent.

Under an isotropic-within-K21 diagnostic null,
\(E\lambda_r=m_r/1048576\).  The resulting capacity table is

\[
\begin{array}{c|c|c|c}
r&m_r&E\lambda_r&
\text{largest }\theta_K\text{ allowed at }\rho_*=0.98346\\ \hline
45&514526&0.49069&0.0669\\
46&467467&0.44581&0.0736\\
47&419385&0.39996&0.0820\\
48&370280&0.35313&0.0929\\
49&320152&0.30532&0.1075\\
50&269001&0.25654&0.1279\\
51&216827&0.20678&0.1587\\
52&163630&0.15605&0.2102\\
53&109410&0.10434&0.3144\\
54& 54167&0.05166&0.6350.
\end{array}
\]

The observed carrier attribution assigns only 2.2 of 18.7 improvement points
to k3 and 17.5 to the variance channel.  That linear contribution ratio is
not the squared metric energy \(\theta_K\), so it is not a proof.  It does,
however, identify r50 as the first rank worth a structural target-tail audit:
if \(\theta_K\) is near 0.118 and the conditional orientation is not
adversarial, r50 is just on the passing side while r49 is just on the failing
side.  This is a preregistered analytic gate, not permission to tune r on a
sentinel.

The r50 economics materially improve the surviving schedule.  Keeping all
ten late H4-plane injections gives

\[
C_{50,10H4}(K)
=14+10(1275/1024)+20
+10(50^3/1024^2)+10K(50^2/1024^2).
\]

The fixed part is about 47.6433.  With \(K=32\) quadrature nodes the total is
about 48.406, leaving about 0.594 equivalent for fused acquisition; with
\(K=48\) it is about 48.788, leaving about 0.212.  Therefore r50 is the first
conditional branch that can keep the complete D10 H4 suffix and a credible
exact-AQ reset under 49.  Its terminal condition is not a generic dimension
claim: it is the measured/proved inequality
\(\theta_K\lambda_{50}\le0.0328064\), together with the reset quadrature
error and acquisition ownership receipts.

## 2026-08-31 — exact endpoint composition is a block-elliptope problem

The Duhamel triangle bound can be sharpened without labels if the exact local
omitted-residual Gram blocks \(E_l\), rather than only their traces, are
available.  Put \(A_l=T_lW_l\), let \(P_l\) transport an injection at l to
the endpoint, and stack \(P=[P_0\ \cdots\ P_L]\).  For endpoint metric
\(H\succeq0\), define \(K=P^THP\).  The sharp worst-case endpoint residual
energy over every joint law consistent with the local blocks is

\[
\boxed{
\mathcal U_*
=\max_{G\succeq0}\langle K,G\rangle
\quad\text{subject to}\quad G_{ll}=E_l.}
\]

This is exact at second order, since every feasible block Gram is realized by
a jointly Gaussian residual vector.  Its dual is

\[
\boxed{
\mathcal U_*
=\min_{Y_l}\sum_l\operatorname{tr}(Y_lE_l)
\quad\text{subject to}\quad
\operatorname{blkdiag}(Y_l)\succeq K.}
\]

For two propagated sources with
\(C_l=H^{1/2}P_lE_lP_l^TH^{1/2}\), the closed value is

\[
\operatorname{tr}C_0+\operatorname{tr}C_1
+2\operatorname{tr}(C_0^{1/2}C_1C_0^{1/2})^{1/2}.
\]

Thus matrix support geometry can beat the triangle law even without assuming
independence.  Conversely, if each \(E_l\) is collapsed to the scalar energy
\(d_l\), the elliptope support is exactly

\[
\left(\sum_l\sqrt{d_l}\right)^2.
\]

No scalar-only argument can certify RSS.  The minimal extra scalar observable
per injection is

\[
\boxed{
\chi_l=E\langle x_l^-,\varepsilon_l\rangle_{H_l},
\qquad H_l=P_l^THP_l,}
\]

the new residual contracted against the complete accumulated past in the
future Bellman metric.  The exact recursion is

\[
S_l=S_{l-1}+d_l+2\chi_l.
\]

Hence \(\chi_l=0\) for all l is necessary and sufficient for the scalar RSS
recursion; adjacent-layer covariance and local-Euclidean orthogonality are not
enough.  For the NCQC engine this identifies a concrete low-bandwidth
acquisition target: acquire/bound one Bellman cross scalar per active layer,
or retain sufficient matrix-valued residual geometry for the dual.  The raw
gate is \(\mathcal U_*<(1-0.8124^2)\|y\|_H^2\).

## 2026-08-31 — literal D11 CP4/star observability fails on the Vandermonde q211 fiber

The cubic-provider lane constructs a seven-dimensional analytic core curve
whose lower moments, all marginal K4, and every pairwise K31/K22 face are
fixed, but whose genuine off-diagonal 211 cumulant obeys

\[
Dq_{211}(A,A,B,C)=32.
\]

Auditing the literal deployed D11 state resolves the remaining CP4 ownership
question.  Its CP4 tensor is represented as

\[
\widehat T=\sum_s\lambda_su_s^{\otimes4}.
\]

Dense push only sends \(u_s\mapsto W^Tu_s\).  At reset, inherited factors
are gated and the only physical births are coordinate atoms

\[
\left(K4_i-\widehat T^{\rm inherited}_{iiii}\right)e_i^{\otimes4}.
\]

The top-energy cap then selects among these factors.  There is no acquisition
port from a true all-distinct amplitude, K31, or K22 into the CP4 factor
state.  The routine that reads CP4 planes merely evaluates
\((iiii),(iiij),(iijj)\) degenerations from factors that were already
carried.

The separate D11 residual-star state also cannot see the witness.  It stores
the K31 residual after subtracting \(3\alpha vC\), with K4 on the diagonal;
K22 enters only through the single pairing projection

\[
\alpha=rac{\langle\operatorname{sym}K22,P22\rangle}
{\langle P22,P22\rangle}.
\]

The Vandermonde tangent fixes every one of these quantities, as well as all
gate inputs.  Holding the inherited internal CP4 carrier fixed therefore
gives

\[
\boxed{
D(\lambda,u,\text{quartic-star},\alpha)=0,
\qquad Dq_{211,\rm offdiag}=32.}
\]

This remains valid away from cap-selection tie surfaces; in fact the complete
candidate list itself is unchanged along the tangent.  A rank-one CP4 tensor
can transport all-distinct information if that information was previously
acquired, but the deployed reset has no such birth map.  Therefore the
literal D11 CP4/star arena fails local q211 observability by an explicit
lawful counterexample.  The only reopen path is a new all-distinct/q211
acquisition or a proof that this physical tangent is endpoint Bellman-dark;
relabeling the existing CP4 carrier is not a solution.

## 2026-08-31 — why the undercomplete r50 branch may transfer better than r55

There is a second benefit to the target-conditioned r50 branch.  Let

\[
\Phi_r:\{M_i\}/O(r)\longrightarrow(C,K21)
\]

be the NCQC observation map and let \(\Psi_r\) be its q211/H4 functional.
At r55, a regular \(D\Phi_{55}\) has a 2,099-dimensional kernel, and the
explicit fibers above show that \(D\Psi_{55}\) does not vanish on that
kernel.  Hence no locally stable function of \((C,K21)\) alone can determine
q211 on the r55 model.

For any undercomplete rank with \(p_r<q\), in particular r50, the generic
alternative is that \(D\Phi_r\) has full column rank modulo conjugacy.  On
the model manifold there is then no latent observation fiber, and

\[
\boxed{
D\Psi_r=L_rD\Phi_r,
\qquad
L_r=D\Psi_r(D\Phi_r)^+.}
\]

Thus q211 is locally a deterministic function of the overdetermined
\((C,K21)\) observations, with stability constant
\(\|L_r\|\le\|D\Psi_r\|/\sigma_{\min}(D\Phi_r)\).  This turns the abstract
“coadaptation” requirement into two exact structural gates:

1. the conditional target normal tail \(\theta_K\lambda_r\) must be small;
2. the restricted smallest singular value of \(D\Phi_r\) must stay bounded
   away from zero across nets.

Under those gates, lowering rank can remove the unstable r55 fiber and make
the induced q211 law more transferable, rather than merely compressing it.
This does not contradict the Vandermonde no-go.  A true state outside the r50
model can still move in a D11-dark q211 direction while fixing all pairwise
observations.  The r50 claim therefore additionally assumes that the physical
state's off-manifold residual is captured by the explicit H4 plane or is
Bellman-small.  Full-column-rank identifiability is an on-manifold theorem,
not a universal reconstruction theorem.

## 2026-08-31 — reset choice at r50: degree four is closed, one-pole Padé is conditional

The rigorous diffuse-cavity reset thresholds can be converted into an exact
effective-rank gate.  With leave-one-eigenvalue-out variance,

\[
\theta_{\max}^2
=\max_i\frac{\lambda_i^2}{2(s_2-\lambda_i^2)}.
\]

If \(R_{\rm eff}=s_2/\lambda_{\max}^2\), then

\[
\theta_{\max}\le t
\quad\Longleftrightarrow\quad
\boxed{R_{\rm eff}\ge1+\frac1{2t^2}.}
\]

For the high local gate \(\rho=0.98346\), the previously proved sufficient
thresholds therefore require

\[
\begin{array}{c|c|c}
\text{reset}&t&\text{required }R_{\rm eff}\\ \hline
N^4&0.094&\gtrsim57.6\\
N^6&0.136&\gtrsim28.0\\
[1/1]\text{ Pad\'{e}}&0.120&\gtrsim35.7.
\end{array}
\]

Hence degree four is rigorously closed at every r50 core, even for a perfectly
flat spectrum.  Degree six and the one-pole solve are feasible, conditional
on the displayed spectral-rank receipts.  A flat r50 core has
\(\theta=1/\sqrt{2(50-1)}\simeq0.101\), so it passes the one-pole gate with
some analytic margin.

The prior one-pole arithmetic was about 1.851 equivalents for ten r55
resets.  Scaling by \((50/55)^3\) gives about 1.39 at r50, provided \(N^2\)
is resident and the solve receives the same favorable billing model.  The
full-D10 r50 transport plus H4 plane costs about 46.451 before reset, so the
one-pole total is about

\[
\boxed{47.84,}
\]

leaving roughly 1.16 equivalents for acquisition and safety margin.  This is
more robust than the K-node Fourier census, but it carries three additional
premises: every live core satisfies \(R_{\rm eff}\ge35.7\); the
complete-old-residual spectral fraction \(\gamma_{old}\) is large enough;
and the Pad\'e solve cost is genuinely realized.  The formal series is
Gevrey and has zero complex radius, so matching coefficients alone is not a
GO.

At the flat r50 point the conservative real-remainder bounds are more
informative than the threshold table alone.  Here
\(y=\theta^2=1/98\) and

\[
D=A_4-A_6y-A_8y^2\simeq18.521.
\]

The resulting relative local reset-error bounds are approximately

\[
\epsilon_4=0.224,
\qquad
\epsilon_6=0.0425,
\qquad
\epsilon_{[1/1]}=0.0694.
\]

Their corresponding one-step geometric correlation floors are about 0.9746,
0.9991, and 0.9976.  The deterministic endpoint amplitude budget 0.583 then
permits coherent normalized downstream source-weight sums of only

\[
2.60\quad(N^4),
\qquad13.7\quad(N^6),
\qquad8.40\quad([1/1]).
\]

Thus degree six is the only one of these three that can certify ten equal,
unit-gain exposures by the triangle theorem without any cross-layer
orthogonality.  Its cost, assuming a resident \(N^2\), is two r50 product
banks, about 2.384 equivalents across ten layers.  Added to the 46.451
transport/H4 base, this is about 48.835, leaving only 0.165.  The rational
reset is the balanced branch: about one extra equivalent of deployment
margin, but it must pass either the block-elliptope composition bound or the
Bellman-cross acquisition because ten fully coherent equal exposures would
slightly exceed its deterministic 8.40 allowance.

## 2026-08-31 — the rank-50 deficit numerically matches the 6e-9 raw target

There is a useful capacity calibration hidden in the same dimension table.
Under the explicit diagnostic assumptions that the response-weighted joint
\((C,K21)\) target is flat, \(D\Phi_r\) has full allowable rank, and the
model is fit by orthogonal projection, the expected unexplained energy is

\[
e_r=1-\frac{p_r}{q}=\frac{q-p_r}{q}.
\]

Multiplying by the current raw \(2.9409\times10^{-8}\) gives the idealized
capacity-only raw floor.  Three ranks are especially revealing:

\[
\begin{array}{c|c|c}
r&e_r&2.9409\times10^{-8}e_r\\ \hline
45&0.32702&9.62\times10^{-9}\\
49&0.20348&5.98\times10^{-9}\\
50&0.17097&5.03\times10^{-9}.
\end{array}
\]

Thus r45 is the knife-edge dimension for the campaign goal raw below 1e-8,
and r49 is the knife-edge dimension for the ideal 6e-9 target.  Neither has
meaningful allowance for reset, conditioning, off-manifold q211, or endpoint
misalignment.  Rank 50 is the first one with a useful reserve.

Indeed

\[
e_{6e-9}=\frac{6\times10^{-9}}{2.9409\times10^{-8}}
\simeq0.20402,
\]

so after the flat r50 capacity tail the remaining squared-error allowance is

\[
e_{6e-9}-e_{50}\simeq0.03305.
\]

The corresponding correlation is

\[
\sqrt{1-0.03305}\simeq0.98334,
\]

almost exactly the campaign's 0.98346 high ingredient gate.  This identifies
the mathematical role of r50: r49 can hit 6e-9 only in the unattainable
zero-overhead ideal, while r50 reserves approximately one high-gate residual
budget for reset/provider error.

This arithmetic is a prioritization theorem, not a performance promise.
Parameter dimension becomes response energy only after whitening by the
actual corpus/Bellman map; the exact replacement for the flat calculation is

\[
\rho_r^2
=\frac{z^TP_{\operatorname{range}(J D\Phi_r)}z}{z^Tz},
\]

where \(J\) maps an interface perturbation to the complete-old-residual
corpus response.  Moreover, capacity and reset errors add as energies only
when their endpoint cross term vanishes.  Joint training on the deployed
recurrence supplies the first-order projection orthogonality; otherwise the
block-elliptope/Bellman-cross bound must price their coherence.  The sharp
fail-fast experiment is therefore the frozen r50 response-range projection
and its smallest singular value, not a standalone local moment score.

### Cost-semantics correction

Every r45/r50 census in this section assumes a **new one-sided H1 factor
bundle** costing one product per transition, hence 14 over D14.  It replaces
the incumbent two-sided D14 word whose documented cost is 28; it is not an
additive neighbor of the incumbent D14+D10=48 schedule.  The replacement is
lawful only if the one-sided factor recurrence reproduces the old D14
variance/signed-H3 endpoint response and reset semantics.  If that receipt
fails, all displayed sub-49 r50 totals are invalid.  This replacement premise
must be a parent node of every r50 cost claim.

## 2026-08-31 — Clifford exact-reset exception: zero-product q211 but full-state capacity no-go

Suppose the H4 Clifford exception holds with a common involution J:

\[
M_i=\alpha_iJ+O_i,
\quad\{O_i,J\}=0,
\quad O_i^2=\omega_i^2I.
\]

Its q211 scalar does not actually require the resident full product \(M_iJ\).
Since

\[
\alpha_i=\frac{\operatorname{tr}(M_iJ)}r,
\qquad
\omega_i^2=\frac{\|M_i\|_F^2}r-\alpha_i^2,
\]

one has

\[
\boxed{-8\|[M_i,J]\|_F^2=-32r\omega_i^2,}
\]

which is available from reductions if \(M_i,J\) are literally resident.
Thus the scalar q211 action can be zero-new-GEMM under the exact Clifford
certificate.

The exception cannot, however, be promoted to a broad full-state closure.
For dense mixing \(O(w)=\sum_iw_iO_i\) to satisfy
\(O(w)^2\in\mathbb RI\) for every live dense direction, polarization forces

\[
\boxed{\{O_i,O_j\}=2g_{ij}I}
\]

on the complete active span.  This is a spin-factor/Clifford Jordan algebra.
Its covariance is

\[
C_{ij}=2r(\alpha_i\alpha_j+g_{ij}),
\]

so \(\operatorname{rank}C\le1+\dim\operatorname{span}\{O_i\}\).  A finite
r matrix representation admits only a small number of independent
anticommuting Clifford generators (growing at most logarithmically with r for
faithful full anticommutation), nowhere near the 877/991 flat feature ranks.
Hence exact Clifford preservation as the full D14 state degenerates to the
low-rank CP/star capacity no-go.

An elementary Clifford-algebra count makes this quantitative.  After
orthonormalizing the positive Gram \(g\), k independent generators satisfy
\(\Gamma_a\Gamma_b+\Gamma_b\Gamma_a=2\delta_{ab}I\).  Their even products
span at least \(2^{k-1}\) independent matrices, while the ambient matrix
algebra has dimension \(r^2\).  Therefore

\[
2^{k-1}\le r^2,
\qquad
k\le1+2\log_2r.
\]

At r50--55 this gives \(k\le12\), hence
\(\operatorname{rank}C\le13\).  Its flat width-1024 correlation ceiling is
at most

\[
\sqrt{13/1024}\simeq0.113,
\]

which is nowhere near either campaign gate.  The full-state Clifford branch
is therefore closed by capacity, not merely “unlikely.”

A sharper irreducible-representation count is also available.  The k
orthonormalized O-generators together with J give k+1 complex Clifford
generators.  Every unital representation has dimension at least
\(2^{\lfloor(k+1)/2\rfloor}\); the possible odd-Clifford projection onto one
simple summand does not reduce this minimum.  Since r<=55, k=11 would require
dimension 64, so actually k<=10.  Hence rank(C)<=11 and the flat correlation
ceiling is at most \(\sqrt{11/1024}\simeq0.104\).  Degenerate generator Gram
only lowers the effective k and creates no loophole.

At the same cut, literal D11 pairing and K4/K31 coordinate-star reads can
represent the trace siblings \(U_{II},U_{IJ},U_{IO_i}\) after their exact
typed identification.  Their future ownership is not proved: the proposed
D14 nonlinear reset is still an abstract/candidate recurrence, and literal
CP4 lacks an all-distinct birth port.  The Clifford theorem therefore remains
viable only as a target-conditioned auxiliary/common suffix word whose broad
base covariance is owned elsewhere, not as the r50 replacement engine.

### Source-acquisition correction for every r50+D10 census

The D10 source charged above is only the complete-old-residualized leading H4
square-lift Gram.  It is not the full AQ omitted-residual Gram

\[
E^{\rm rest}_{ij}=\langle e_i,e_j\rangle.
\]

The full object would make same-cut covariance exact, but its off-diagonal
entries require bivariate joint AQ reset integrals for every unit pair; no
such acquisition was included in the cost.  Even the leading H4 square Gram
is generically expensive to assemble from every pair of cores.  The
commutator zero-product identity removes a product only for already-resident
target factors; it does not materialize the complete plane for free.

Therefore the cost statuses are:

- full \(E^{\rm rest}\): closed under the displayed census absent a new
  bivariate fusion theorem;
- leading H4 square plane: open only under literal incumbent-plane ownership,
  resident-product fusion, or a target-conditioned low-rank Bellman action;
- the local 0.98333 surrogate leaves H6+ and cannot be described as an exact
  covariance reset.

This acquisition premise and the one-sided-D14 replacement premise are
independent parents of all sub-49 r50 totals.

## 2026-08-31 — full AQ correction: r49, not pure-quadratic r50, is the first 6e-9-with-margin rank

The preceding capacity counts treated only the quadratic matrices.  If the
proposed one-sided D14 replacement literally carries the full AQ coefficients

\[
g_i=m_i+u_i^Tz+Q_{M_i}(z),
\]

then the linear bank contributes nr additional physical parameters and must
be included consistently in both cost and capacity.  Modulo common latent
rotation,

\[
\boxed{
p_r^{AQ}=1024\left\{\frac{r(r+1)}2+r\right\}
-\frac{r(r-1)}2.}
\]

The sharp generic dimension threshold for joint \((C,K21)\) moves from r55
to r54:

\[
p_{53}^{AQ}=1,518,238=q-55,138,
\qquad
p_{54}^{AQ}=1,574,505=q+1,129.
\]

More importantly for the raw target,

\[
\begin{array}{c|c|c}
r&q-p_r^{AQ}&(q-p_r^{AQ})/q\\ \hline
48&321128&0.20410\\
49&269976&0.17159\\
50&217801&0.13843.
\end{array}
\]

Under the same explicitly flat response-space diagnostic, r48 bottoms out at
about \(6.002\times10^{-9}\), just missing the ideal 6e-9 target before any
reset/provider error.  Rank 49 bottoms out near
\(5.046\times10^{-9}\) and leaves normalized squared-error allowance

\[
0.204019-0.171590\simeq0.03243,
\]

equivalent to correlation about 0.98365.  Therefore full-AQ r49 is the exact
analogue of the earlier pure-quadratic r50 insight: it is the first rank with
approximately one high-gate error budget beyond the 6e-9 capacity target.

Its one-sided transport bank has width

\[
r+d_r=49+1225=1274,
\]

so carrying it through all fourteen transitions costs

\[
14(1274/1024)\simeq17.418
\]

equivalents, replacing the incumbent D14 cost 28.  Adding a D10 H4 plane
costs 20, for a transport total about 37.418.  Fourteen local r49 eigensolves
cost about 1.571, and each quadrature node across all fourteen layers costs
about 0.0321.  Even 32 nodes give a core-reset/transport subtotal near

\[
\boxed{40.02,}
\]

leaving almost nine equivalents for source acquisition and safety.  This is
the strongest arithmetic branch derived in this lane.

It also has the strongest premises.  A lawful initialization/section must map
the raw Gaussian or first exact Gaussian-ReLU state into r49 AQ cores without
creating an uncontrolled signed K3; the one-sided recurrence must replace the
actual D14 endpoint response; the response-weighted range and conditioning
gates must pass; and the H4 source-acquisition hole remains.  Pure
parameter dimension proves none of these.  If the u bank is not independently
resident and transported, revert to the conservative pure-quadratic r50
branch rather than borrowing its degrees of freedom for free.

### Generic raw-Gaussian initialization obstruction

The initialization premise is substantial.  Let

\[
V_r=\mathbb R^r\oplus\operatorname{Sym}_r
\]

be the centered AQ coefficient space.  Its cubic cumulant polarization is

\[
\mathcal T((u,A),(v,B),(w,C))
=8\,\operatorname{symtr}(ABC)
+2\{u^TBv+u^TCw+v^TAw\},
\]

with normalization chosen so that
\(\mathcal T((u,A)^3)=8\operatorname{tr}A^3+6u^TAu\).  To represent an
arbitrary full-rank Gaussian width-1024 state exactly, the 1024 AQ feature
vectors must be linearly independent and their span S must satisfy

\[
\mathcal T|_{S^3}=0.
\]

At r49 the ambient dimension is 1274.  For any s define the Hessian bilinear
form \(B_s(x,y)=\mathcal T(s,x,y)\).  At a point \(s=(0,A)\), this form is
block diagonal: its linear block is A and its matrix block is Jordan
multiplication \(B\mapsto A\circ B\).  It is nondegenerate whenever A is
invertible and no two eigenvalues of A sum to zero.  Hence Hessian
nondegeneracy is a nonempty Zariski-open condition in \(V_r\).

If a cubic-null S contains any such regular s, then S is totally isotropic
for the nondegenerate symmetric form \(B_s\), forcing

\[
\dim S\le\frac{\dim V_r}{2}=637.
\]

Therefore a 1024-dimensional exact Gaussian seed would have to lie entirely
inside the exceptional Hessian-discriminant variety.  The standard lawful
bipartite construction has only
\(p(q+1)\le625\) dimensions at r49, illustrating the gap.  This is a generic
no-go, not a universal proof that no exceptional 1024-dimensional null space
exists; such an exception would require an explicit algebraic construction.

Consequently the full-AQ r49 engine should not be claimed to initialize an
arbitrary raw Gaussian layer exactly.  Its lawful routes are: construct and
prove an exceptional high-dimensional cubic-null section; start after the
first exact Gaussian-ReLU reset with an explicit deployable AQ section; or
bound the early spurious-K3 response as Bellman-dark.  Merely matching the
initial covariance Gram is insufficient.

## 2026-08-31 — exact three-squared-level spectral quotient theorem

There is one sharp exception to the conservative Padé \(\gamma_{old}\) bound.
Suppose a core has exactly three distinct squared eigenvalues
\(x_1,x_2,x_3\).  Every even spectral reset is then a vector in
\(\mathbb R^3\), while the complete old even span \(\{I,M^2\}\) has
dimension two.  Therefore

\[
\dim\left(\operatorname{span}\{f(M^2)\}/
\operatorname{span}\{I,M^2\}\right)=1.
\]

Consequently the exact reset novel residual and any polynomial/rational
approximant with a nonzero quotient component are collinear after old-span
subtraction.  Their absolute residual correlation is exactly one; a fitted
scalar removes amplitude mismatch.  For

\[
h(x)=\frac{x^2}{1+cx},
\]

the second divided difference is explicitly

\[
\boxed{
h[x_1,x_2,x_3]
=\frac1{(1+cx_1)(1+cx_2)(1+cx_3)}>0,}
\]

so the one-pole Padé has nonzero novelty whenever the three levels are
distinct.  This is stronger than a pointwise approximation theorem.

It is not a deployable multi-layer closure by itself.  Exact repeated
eigenvalues reduce the core-orbit parameter dimension, and a generic dense
linear combination of differently oriented three-level matrices develops a
full spectrum at the next layer.  Requiring the three-level property to be
closed under all dense combinations returns to the Clifford/spin-factor
algebra, which has already failed capacity.  A nearly three-cluster spectrum
preserves full parameter dimension and gives an approximate one-dimensional
quotient, but needs an explicit cluster-gap/within-cluster perturbation bound
at every layer.  Thus this theorem is a local reset exception or a diagnostic
regularizer, not an escape from the dense-transport obstruction.

## 2026-08-31 — preferred source repair: r49 AQ low chaos plus incumbent Gaussian rest

The leading-H4 acquisition hole has a more lawful alternative.  Let
\(R_i^{AQ}=P_{\le2}(g_i)_+\) be the exact coadapted AQ projection.  The
incumbent covariance recursion already has an affine Gaussian frame, for
which the full higher-chaos rest is available by the Mehler kernel or the
closed bivariate Gaussian ReLU formula.  Define

\[
\boxed{
C_{ij}^{AQ+Grest}
=\langle R_i^{AQ},R_j^{AQ}\rangle+K_{ij}^{G,>2}.}
\]

At \(M_i=M_j=0\) this is exactly the full ReLU covariance, not a truncation.
The Gaussian-rest source is an elementwise function of matched means,
variances, and correlations and is already the semantic object of D10; it
does not require pairwise r-by-r AQ determinant calculations.

For nonzero quadratic cores the genuinely unowned source is only

\[
\boxed{
\Delta K_{ij}^{rest}
=\langle r_i^{true},r_j^{true}\rangle-K_{ij}^{G,>2},}
\]

after subtracting every incumbent H31/CP4/Q24 contribution.  This difference
vanishes at the affine base.  It is therefore the correct object for the
cross-block/pure-B perturbation bounds and the endpoint block-elliptope gate;
charging the entire AQ omitted Gram double-counts the affine rest.

This yields the preferred conditional architecture:

1. full-AQ r49 one-sided recurrence replaces D14 and supplies coadapted low
   chaos plus signed K3/q211 potentials;
2. the incumbent D10 Gaussian/Mehler rest supplies all affine H>=3 covariance;
3. only \(\Delta K^{rest}\) must pass the deformation/Bellman error gate.

Its arithmetic is the roughly 40-equivalent r49 subtotal above and has a
literal source at the affine base.  This does not prove the non-Gaussian rest
small: gates with a large pure-B deformation, negative standardized means,
or near-unit pair correlations can make it material.  But it replaces the
generic unbilled H4-plane formation problem by a sharply defined residual
that is zero at a controlled base point.  The H4-square branch remains a
secondary option only if its separate fusion receipt is obtained.

### Exact centered-parity theorem: Gaussian rest is first-order correct in every quadratic direction

The residual difference above has a favorable exact cancellation at a
zero-bias affine base.  Let \(g_i=u_i^Tz\), let \(q_i=Q_{M_i}(z)\) be any
centered quadratic perturbation, and put

\[
r_i(q_i)=(I-P_{\le2})(g_i+q_i)_+.
\]

At q=0,

\[
(g_i)_+=\frac12g_i+\frac12|g_i|,
\]

so after removing H0,H1,H2 the Gaussian rest \(r_i(0)\) contains only even
global chaoses H4,H6,... .  Its first variation is

\[
\dot r_i=(I-P_{\le2})[1_{\{g_i>0\}}q_i].
\]

Using

\[
1_{\{g_i>0\}}=\frac12+\frac12\operatorname{sgn}(g_i),
\]

the first term \(q_i/2\) lies entirely in H2 and is removed.  The second is
odd under global sign reversal \(z\mapsto-z\); after its possible H1 part is
removed, \(\dot r_i\) contains only odd chaoses H3,H5,... .  Therefore, for
every pair i,j and arbitrary linear directions \(u_i,u_j\),

\[
\boxed{
\langle\dot r_i,r_j(0)\rangle
=\langle r_i(0),\dot r_j\rangle=0.}
\]

Equivalently,

\[
\boxed{
D\,\Delta K^{rest}\big|_{M=0}=0.}
\]

Thus AQ-low-chaos plus Gaussian rest is first-order exact in every centered
quadratic direction, including collective, cross-block, and pure-B
deformations.  The missing rest begins at the first nonlinear remainder in
the quadratic deformation, not at O(||M||).  This is the parity mechanism
that makes the r49+Gaussian-rest architecture materially more plausible than
an uncorrected AQ projection.

Bias is a binding exception.  For standardized threshold alpha, the even
part of the gate differs from 1/2 only on \(|X|<|\alpha|\):

\[
\frac{1_{\{X>-\alpha\}}+1_{\{-X>-\alpha\}}}{2}
=\frac12+\frac{\operatorname{sgn}(\alpha)}2
1_{\{|X|<|\alpha|\}}.
\]

Hence the only first-order even-rest leakage is a small-threshold-band
functional.  A simple bound is

\[
\|q,1_{\{|X|<|\alpha|\}}\|_2
\le\|q\|_4
\Pr(|X|<|\alpha|)^{1/4},
\]

with \(\Pr(|X|<a)\le\sqrt{2/\pi}\,a\).  Exact tilted Gaussian moments can
sharpen it.  The graph gate should therefore track standardized means and
quadratic deformation jointly; zero-bias parity must not be silently applied
to negative-mean units.

### Finite centered bound using only one-unit sign-flip integrals

The parity result has a nonasymptotic completion that avoids bivariate AQ
acquisition.  Write

\[
(g_i+q_i)_+=(g_i)_++1_{\{g_i>0\}}q_i+R_i.
\]

Pointwise, the switching remainder satisfies

\[
|R_i|\le |q_i|1_{\{|g_i|\le|q_i|\}},
\]

and projection is contractive.  Define the one-unit quantities

\[
a_i=\|q_i\|_2,
\quad
b_i=\left(E[q_i^2 1_{\{|g_i|\le|q_i|\}}]\right)^{1/2},
\quad
c_i=\|r_i(0)\|_2.
\]

The first variation has norm at most \(a_i\), the projected nonlinear
remainder has norm at most \(b_i\), and centered parity kills the two
Gaussian-rest/first-variation cross terms exactly.  Expanding the residual
Gram therefore gives the entrywise rigorous bound

\[
\boxed{
|\Delta K_{ij}^{rest}|
\le
a_ia_j+c_ib_j+b_ic_j+a_ib_j+b_ia_j+b_ib_j.}
\]

Every input on the right is a univariate AQ or affine quantity.  The matrix
envelope is a sum of six rank-one outer products, so a frozen Bellman costate
can contract it by reductions without ever forming bivariate determinant
integrals.  This is an error certificate, not a correction provider, but it
is enough to close or retain the r49+Gaussian-rest architecture before an
expensive source implementation.  The sign-flip term b is the decisive
quantity; the transverse-slope and tilted-density bounds from the fidelity
lane can sharpen it when the crude indicator estimate is loose.

### Exact 6e-9 coherence budget for the r49 branch

Let \(e_{cap}=0.171590\) denote the flat diagnostic r49 capacity tail, let
\(e_{rest}\) be the normalized endpoint energy of all reset/Gaussian-rest
approximation error, and let

\[
\chi=\frac{\langle r_{cap},\delta_{rest}\rangle}{\|y\|^2}.
\]

The total error is exactly

\[
e_{tot}=e_{cap}+e_{rest}+2\chi.
\]

The 6e-9 target allows \(e_{tot}\le e_*\simeq0.204019\).  Therefore:

- with projection/Bellman orthogonality, \(\chi=0\), the rest may use the
  full \(e_{rest}\le0.032429\) high-gate budget;
- with unknown worst-case coherence,

  \[
  e_{rest}
  \le(\sqrt{e_*}-\sqrt{e_{cap}})^2
  \simeq1.40\times10^{-3}.
  \]

Thus lack of a cross receipt shrinks the allowable reset/rest energy by more
than a factor of twenty.  More generally the exact gate is

\[
\boxed{
\chi\le\frac{e_*-e_{cap}-e_{rest}}2.}
\]

This is why joint refitting on the deployed recurrence or acquiring the one
aggregate Bellman cross scalar is not optional bookkeeping: it is the
difference between a plausible 3.24% residual-energy allowance and a
near-exact 0.14% requirement.

## 2026-08-31 — complete full-AQ cumulants and the once-only q211 boundary

Fix the unhalved quadratic convention

\[
G_a=u_a^Tz+z^TA_az-\operatorname{tr}A_a,
\qquad z\sim N(0,I_r).
\]

(For the convention \(\frac12(z^TMz-\operatorname{tr}M)\), substitute
\(A=M/2\).)  The joint cumulant generating function is

\[
\log E e^{\sum_at_aG_a}
=-\frac12\log\det(I-2A(t))-\operatorname{tr}A(t)
+\frac12u(t)^T(I-2A(t))^{-1}u(t),
\]

where \(A(t)=\sum_at_aA_a\) and \(u(t)=\sum_at_au_a\).  Expansion gives the
complete covariance and repeated cubic

\[
C_{ij}=u_i^Tu_j+2\operatorname{tr}(A_iA_j),
\]

\[
\boxed{
\kappa(G_i,G_i,G_j)
=8\operatorname{tr}(A_i^2A_j)
+2u_i^TA_ju_i+4u_i^TA_iu_j.}
\]

For two arbitrary AQ directions \(X=(u,A)\), \(Y=(v,B)\), the complete
quartic repeated-pair cumulant is

\[
\boxed{
\begin{aligned}
\kappa(X,X,Y,Y)={}&32\operatorname{tr}(A^2B^2)
+16\operatorname{tr}(ABAB)\\
&+8\{u^TB^2u+v^TA^2v+2u^T(AB+BA)v\}.
\end{aligned}}
\]

Polarizing the last formula gives, for \(Y=(v,B),Z=(w,C)\),

\[
\begin{aligned}
\kappa(X,X,Y,Z)={}&32\operatorname{tr}(A^2BC)
+16\operatorname{tr}(ABAC)\\
&+4u^T(BC+CB)u+8v^TA^2w\\
&+8u^T\{(AB+BA)w+(AC+CA)v\}.
\end{aligned}
\]

The trace sector splits exactly as

\[
32\operatorname{tr}(A^2BC)+16\operatorname{tr}(ABAC)
=48\operatorname{tr}(A^2BC)
+8\operatorname{tr}([A,B][A,C]).
\]

Thus a full-AQ bank owns the entire connected quartic of its **AQ random
variables**, not just the pure-matrix commutator.  For a rank-one target
slice \(\Lambda_i=vv^T\), put

\[
B_v=\sum_jv_jA_j,\qquad v_u=\sum_jv_ju_j.
\]

The contracted q211 action is the boxed repeated-pair formula with
\((u,A)=(u_i,A_i)\) and \((v,B)=(v_u,B_v)\).  A rank-\(c\) target slice is a
sum of \(c\) such reductions.  Therefore no CP4 tensor or native six-ray
repair is additive to a full-AQ q211 implementation; those are alternative
implementations of the same AQ action.

There is one important implementation qualification.  At reset let

\[
Y_i^{AQ}=(g_i^{AQ})_+,
\qquad X_i=P_{\le2}Y_i^{AQ}.
\]

The stored post-reset AQ coefficients determine \(\kappa_4(X^{AQ})\), but
not \(\kappa_4(Y^{AQ})\).  Ownership of the latter requires the fused H211
reset-surface/Fourier oracle evaluated from the **pre-reset** AQ law.  If that
oracle is included, the once-only source is

\[
T_{211}^{AQ}=\Pi_{211,off}\kappa_4(Y^{AQ}).
\]

The only lawful additional source is then

\[
\boxed{
T_{211}^{rem}
=R_{old}\Pi_{211,off}
\{\kappa_4(Y^{true})-\kappa_4(Y^{AQ})\},}
\]

where \(R_{old}\) subtracts CP3/CP4/star/Q24/H31 and affine connected terms
exactly once.  The Gaussian rest is a two-point covariance control variate;
it is not a fourth-cumulant credit.  If the pre-reset law is exactly affine
Gaussian and the AQ representative is the same affine variable, then
\(Y^{true}=Y^{AQ}\) and this residual is zero.  Away from that base, every
term in its multilinear expansion contains at least one true-minus-AQ slot.

## 2026-08-31 — bias correction to the centered-parity theorem

The zero-bias parity cancellation above is exact, but the statement that
only a threshold-band term is needed away from zero bias is incomplete: the
affine Gaussian rest itself develops odd higher chaos when the standardized
mean is nonzero.

For one standardized coordinate define

\[
f_\alpha(X)=(X+\alpha)_+,
\quad r_\alpha=P_{>2}f_\alpha,
\quad d_\alpha(q)=P_{>2}[1_{\{X>-\alpha\}}q],
\]

where \(q\) is a centered quadratic chaos.  At \(\alpha=0\), \(r_0\) is
even and \(d_0(q)\) is odd, so their pairing vanishes even for correlated
unit directions.  Put

\[
a(q)=\|q\|_2,
\quad
\beta_\alpha(q)
=\|q(1_{\{X>-\alpha\}}-1_{\{X>0\}})\|_2,
\]

\[
h_\alpha=\|r_\alpha-r_0\|_2,
\qquad c_\alpha=\|r_\alpha\|_2.
\]

The gate difference is supported on the one-sided interval between
\(-\alpha\) and zero, hence

\[
\beta_\alpha(q)
\le\left(E[q^2 1_{\{|X|<|\alpha|\}}]\right)^{1/2}.
\]

The baseline parity drift is an exact univariate Hermite quantity:

\[
\boxed{
h_\alpha^2
=\sum_{n\ge3}\frac{
\{\phi(\alpha)H_{n-2}(-\alpha)
-\phi(0)H_{n-2}(0)\}^2}{n!}.}
\]

It also has a sharper global path bound.  Since

\[
\partial_\alpha r_\alpha=P_{>2}1_{\{X>-\alpha\}},
\]

its squared norm is

\[
\Phi(\alpha)-\Phi(\alpha)^2-\phi(\alpha)^2
-\frac{\alpha^2\phi(\alpha)^2}{2}
\le d(0),
\]

where \(d(0)=1/4-1/(2\pi)\) is proved below to be the global maximum.
Consequently

\[
\boxed{h_\alpha\le\sqrt{d(0)}\,|\alpha|
\simeq0.3014|\alpha|.}
\]

For affine scale \(\sigma\), multiply \(h,c\) by \(\sigma\).  Adding and subtracting
the zero-bias functions gives the rigorous first-variation bound

\[
\boxed{
|D\Delta K^{rest}_{ij}|
\le
\beta_i c_j+a_i h_j+c_i\beta_j+h_i a_j.}
\]

Thus the correct finite-bias gate has two independent inputs: sign-band
mass \(\beta\), and affine-rest parity drift \(h\).  Neither can be deleted.

There is also a nonasymptotic biased version of the prior six-outer-product
bound.  Let

\[
s_i=\left(E[q_i^2,1_{\{|g_i|\le|q_i|\}}]\right)^{1/2}
\]

be the nonlinear sign-switch remainder.  Then

\[
\begin{aligned}
|\Delta K_{ij}^{rest}|\le{}&
a_ia_j+c_is_j+s_ic_j+a_is_j+s_ia_j+s_is_j\\
&+\beta_ic_j+a_ih_j+c_i\beta_j+h_ia_j.
\end{aligned}
\]

At zero bias \(\beta=h=0\), recovering the earlier centered bound.  The
right side is a ten-outer-product envelope and still contracts with one
specified Bellman plane using reductions only.  It certifies an action or an
error; it does not construct the missing two-residual forward plane.

The sign-switch norm has a universal superlinear bound requiring no
independence between the affine coordinate and q.  For
\(g\sim N(0,\sigma^2)\), any \(\tau>0\) gives

\[
\begin{aligned}
s^2
&=E[q^2 1_{\{|g|\le|q|\}}]\\
&\le
\|q\|_4^2\Pr(|g|\le\tau)^{1/2}
+\frac{\|q\|_4^4}{\tau^2}\\
&\le
\|q\|_4^2
\left(\sqrt{\frac2\pi}\frac{\tau}{\sigma}\right)^{1/2}
+\frac{\|q\|_4^4}{\tau^2}.
\end{aligned}
\]

Optimizing tau yields, with
\(c=\sqrt{2/\pi}\),

\[
\boxed{
\frac{s}{\sigma}
\le C_{sw}
\left(\frac{\|q\|_4}{\sigma}\right)^{6/5},
\qquad
C_{sw}
=\left[(4^{1/5}+4^{-4/5})c^{2/5}\right]^{1/2}
\simeq1.23.}
\]

Thus the centered true-minus-Gaussian covariance rest is rigorously
superlinear in standardized AQ deformation even without a transverse-slope
or conditional-independence premise.  The exponent 6/5 is conservative; a
surface-margin receipt can improve it.

For a pure centered quadratic \(q=z^TAz-\operatorname{tr}A\), the fourth
moment is exact:

\[
\|q\|_2^2=2s_2,
\qquad
\|q\|_4^4=12s_2^2+48s_4,
\quad s_k=\operatorname{tr}A^k.
\]

With spectral fourth stable rank

\[
r_4=\frac{s_2^2}{s_4},
\]

one obtains

\[
\left(\frac{\|q\|_4}{\|q\|_2}\right)^4
=3+\frac{12}{r_4},
\]

and hence the fully native gate

\[
\boxed{
\frac{s}{\sigma}
\le C_{sw}
\left(3+\frac{12}{r_4}\right)^{3/10}
\left(\frac{\|q\|_2}{\sigma}\right)^{6/5}.}
\]

The multiplier after \(C_{sw}\) approaches \(3^{3/10}\) for a diffuse
core and is at most \(15^{3/10}\) under arbitrary concentration.  Thus
variance and one spectral stable-rank scalar suffice to evaluate the bound.

## 2026-08-31 — r54 versus r49 and the typed post-reset section theorem

Let \(\Theta_r\) be the bank of \(1024\) AQ coefficient pairs modulo the
common \(O(r)\) gauge, and define the static interface map

\[
\Phi_r:\Theta_r/O(r)\longrightarrow (C,K21)
\]

by the exact formulas above.  A lawful post-first-reset section is not an
arbitrary covariance factorization.  It is a deterministic map \(S_r\) such
that

\[
\boxed{\Phi_r(S_r(C,K21))=(C,K21)}
\]

in a fixed gauge and with deployable conditioning.  By the submersion
theorem, a local smooth section exists near a particular interface if and
only if a gauge-fixed Jacobian of \(\Phi_r\) is full row rank there.  Its
first-order sensitivity is at least \(1/\sigma_{min}(D\Phi_r)\).  Dimension
surplus is necessary but not sufficient, and it gives neither actual-cut
membership nor a runtime construction.

For \(q=1,573,376\),

\[
p_{54}^{AQ}=1,574,505=q+1,129,
\qquad
p_{55}^{AQ}=1,631,795=q+58,419.
\]

Hence r54 can eliminate the generic static \((C,K21)\) tail at a certified
regular point, whereas r49 generically cannot and must use a target-weighted
projection.  Its all-14-layer cost with 32 reset nodes is

\[
14\frac{1539}{1024}+20
+14\frac{54^3}{1024^2}
+32\,14\frac{54^2}{1024^2}
\simeq44.39.
\]

Conditional on a regular section, r54 therefore dominates r49's flat
capacity/error trade while remaining under 49.  It does **not** solve the
raw Gaussian seed: zero K3 there still requires a 1024-dimensional
exceptional cubic-null section.  Starting after a genuinely typed
non-Gaussian reset can avoid that special seed, but only through the boxed
section receipt.

## 2026-08-31 — D14-through-L4 plus r55 suffix: exact census and first fatal premise

The proposed hybrid keeps literal D14 for four early transitions and starts
a full-AQ r55 recurrence for the ten transitions L5--14.  Its optimistic
dense-equivalent census is arithmetically consistent:

\[
\begin{aligned}
N={}&8
+10\frac{55+55\cdot56/2}{1024}
+20
+10\frac{55^3}{1024^2}
+32\,10\frac{55^2}{1024^2}\\
\simeq{}&46.09.
\end{aligned}
\]

Thus it has about 2.91 equivalents of margin.  It avoids both the raw
Gaussian cubic-null seed and the r49/r54 static target-tail issue **if** the
L4 seam supplies an actual synchronized r55 AQ bank.

Literal D14 does not do that.  Its state at the seam is a dense response or
correction plane \(E_4\).  Factoring the corrected covariance into a Gram
bank matches \(C_4\), but it does not determine the signed \(K21_4\), q211,
reset spectra, or a common latent gauge.  Therefore

\[
\boxed{
E_4\in\operatorname{Sym}_{1024}
\quad\not\Rightarrow\quad
\{(m_i,u_i,A_i)\}_{i=1}^{1024}.}
\]

The first fatal premise is precisely this type mismatch.  The branch reopens
if some other resident state supplies the typed bank, or if a deterministic,
conditioned r55 section of the complete synchronized \((C,K21)\) interface
is proved.  Parameter surplus alone is not that proof.

Once a typed bank exists, the AQ low-chaos covariance transport itself is
exact.  With row feature

\[
F_i=(u_i,\sqrt2\,\operatorname{svec}A_i),
\qquad C^{AQ}=FF^T,
\]

dense mixing gives \(F^+=W^TF\), hence

\[
F^+(F^+)^T=W^TC^{AQ}W.
\]

This proves replacement of a two-sided covariance conjugation, but not of
D14's complete signed/frame injection.  For a linearized D14 endpoint, let
\(\lambda_l\) be its exact reverse Bellman covectors, \(e_4\) the seam
interface error, and \(d_l\) every suffix source/transition defect after old
subtraction.  The exact first-order replacement identity is

\[
\boxed{
\delta y
=\langle\lambda_4,e_4\rangle
+\sum_{l=5}^{14}\langle\lambda_{l+1},d_l\rangle.}
\]

Vanishing of the total scalar is necessary and sufficient for that fixed
endpoint; vanishing term by term is a stronger sufficient condition.  For a
finite nonlinear replacement, interpolate the true and AQ recurrences.
Discrete adjoint differentiation gives the exact homotopy identity

\[
y^{true}-y^{AQ}
=\int_0^1\left[
\langle\lambda_4(t),\dot x_4(t)\rangle
+\sum_{l=5}^{14}
\langle\lambda_{l+1}(t),\partial_tF_l^t(x_l(t))\rangle
\right]dt.
\]

This is the proof-level meaning of “replaces D14 value.”  A covariance match
at one seam is not enough.

Finally, the literal D10 lineage owns only affine Gaussian Hermite orders
3--6 after the AQ block replaces orders 1--2.  It does not currently own the
full Gaussian rest.  The same 20-product carrier can plausibly take the exact
bivariate Gaussian kernel as its local source without another dense
conjugation, but that is a source replacement requiring a literal billing and
subtraction receipt.  Otherwise the affine \(n\ge7\) tail and the complete
non-Gaussian true-minus-AQ residual remain in \(d_l\).

### Graph-ready disposition

- `AQ_R54_REGULAR_SECTION`: arithmetic pass; open on a full-row-rank witness,
  actual-cut membership, gauge, and \(\sigma_{min}\).
- `D14_L4_AQ_R55_SUFFIX_COST`: solved conditional, 46.09 equivalents.
- `D14_L4_AQ_TYPED_SEAM`: blocked by state type; covariance factorization is
  insufficient.
- `AQ_SUFFIX_D14_VALUE_GREEN_IDENTITY`: solved exact; local source pairings
  and the seam pairing are the binding receipt.
- `D10_EXACT_GAUSSIAN_REST_SOURCE`: literal orders 3--6 only; exact-kernel
  replacement open, non-Gaussian rest still open.

### Exact covariance-first reduction of the r54/r55 seam problem

The section problem can be reduced to a single orientation equation.  Put

\[
D_r=r+\frac{r(r+1)}2
\]

and identify each AQ pair with its whitened feature vector

\[
f_i=(u_i,\sqrt2\operatorname{svec}A_i)\in\mathbb R^{D_r}.
\]

If the desired seam covariance \(C\succ0\), fix its positive square root
\(L=C^{1/2}\).  Every feature bank satisfying \(FF^T=C\) is exactly

\[
\boxed{F=LQ,\qquad QQ^T=I_{1024},}
\]

with \(Q\) on the row-Stiefel manifold
\(\mathrm{St}(1024,D_r)\).  Conversely every such \(Q\) matches covariance
exactly.  Therefore all signed seam difficulty is the equation

\[
\boxed{\Psi_C(Q):=K21(LQ)=K21_* .}
\]

The common latent \(O(r)\) action is a subgroup of the right orthogonal
action on the feature coordinates and must be quotiented once.  The exact
available orientation dimensions are

\[
\dim\mathrm{St}(1024,D_r)-\frac{r(r-1)}2.
\]

For r54 this is

\[
1024(1539)-\frac{1024\cdot1025}{2}-1431
=1,049,705
=1024^2+1,129,
\]

and for r55 it is

\[
1024(1595)-\frac{1024\cdot1025}{2}-1485
=1,106,995
=1024^2+58,419.
\]

This recovers the prior surplus counts but gives them an operational meaning:
covariance is free after the square root; r54 has only 1,129 orientation
degrees beyond the entire K21 plane, whereas r55 has 58,419.

For \(Q\in\mathrm{St}(1024,D_r)\), its tangent consists of

\[
\delta Q,Q^T+Q\,\delta Q^T=0.
\]

A complete regularity certificate is

\[
\boxed{
P_{T_Q\mathrm{St}}\nabla_Q
\langle T,\Psi_C(Q)\rangle=0
\quad\Longrightarrow\quad T=0,}
\]

after removing the common latent-gauge tangent.  Equivalently the restricted
Jacobian \(D_Q\Psi_C\) has row rank \(1024^2\).  At such a solution the
implicit-function theorem supplies a locally lawful seam section, and its
condition number is \(1/\sigma_{min}(D_Q\Psi_C)\).  This is strictly stronger
than parameter counting and strictly weaker than an explicit cheap deploy
algorithm.

It also exposes why literal D14 cannot initialize the suffix: \(E_4\) can
fix \(L\), but it supplies neither the target \(K21_*\) as a synchronized
law nor an orientation \(Q\) solving the boxed equation.  The minimal typed
seam receipt is therefore \((C_4,K21_4,Q_4)\), with the last component
certified by the Stiefel equation and singular-value gate.

## 2026-08-31 — signed residual hierarchy beyond the AQ reset surface

The covariance-rest envelope is not a signed-H3 or q211 certificate.  At a
reset write centered variables

\[
Y_a^{true}=Y_a^{AQ}+e_a,
\]

where the AQ surface oracle owns all moments of \(Y^{AQ}\) that it explicitly
queries.  Multilinearity gives the exact repeated-cubic defect

\[
\begin{aligned}
\Delta\kappa_{iij}={}&
2E[e_iY_i^{AQ}Y_j^{AQ}]
+E[(Y_i^{AQ})^2e_j]\\
&+E[e_i^2Y_j^{AQ}]
+2E[e_iY_i^{AQ}e_j]+E[e_i^2e_j].
\end{aligned}
\]

Let \(p_a=\|Y_a^{AQ}\|_3\) and \(d_a=\|e_a\|_3\).  Holder gives the
entrywise fail-fast envelope

\[
\boxed{
|\Delta\kappa_{iij}|
\le(p_i+d_i)^2(p_j+d_j)-p_i^2p_j.}
\]

Likewise every centered fourth cumulant obeys

\[
|\kappa_4(Z_1,Z_2,Z_3,Z_4)|
\le4\prod_{a=1}^4\|Z_a\|_4.
\]

Therefore, with \(p_a=\|Y_a^{AQ}\|_4\) and
\(d_a=\|e_a\|_4\),

\[
\boxed{
|\Delta\kappa_{iijk}|
\le4\{(p_i+d_i)^2(p_j+d_j)(p_k+d_k)
-p_i^2p_jp_k\}.}
\]

These bounds preserve no sign, so they are conservative Bellman error gates,
not providers.  They do prove a clean hierarchy:

1. the fused AQ reset surface owns \(K21/q211\) of \(Y^{AQ}\);
2. the Gaussian rest owns a two-point affine covariance reference only;
3. every remaining signed source contains at least one residual \(e\);
4. pairwise residual Grams cannot identify its sign, so either a signed
   surface action, a forward residual carrier, or the displayed worst-case
   Bellman envelope is mandatory.

This is the signed analogue of the covariance-rest boundary and prevents a
small two-point rest norm from being silently treated as exact H3 value.

### Information-theoretic seam no-go from covariance alone

There is a sharp reason that a covariance or D14 variance plane cannot be
silently decoded into the missing AQ orientation.  Under simultaneous feature
negation

\[
(u_i,A_i)\mapsto(-u_i,-A_i)\quad\text{for every }i,
\]

the complete covariance is unchanged, while every cubic cumulant changes
sign:

\[
C\mapsto C,
\qquad K21\mapsto-K21.
\]

Consequently no deterministic map that factors only through \(C\) can be an
exact AQ seam initializer on a class containing both orientations.  Against
the two indistinguishable states, any predicted cubic plane \(\widehat K\)
has

\[
\max\{\|\widehat K-K21\|,
        \|\widehat K+K21\|\}\ge\|K21\|.
\]

If the literal early D14 word \(E_4\) is claimed to evade this witness because
it contains a signed-H3 response, the required receipt is an explicit seam
decoder and its invariant kernel.  A generic symmetric \(E_4\) has only
\(1024\cdot1025/2\) coordinates, whereas the synchronized \((C,K21)\)
interface has \(1,573,376\); naming the carried word “signed” does not make
it a complete interface.

For one frozen scalar endpoint, the seam can in principle be weakened to the
future-observable quotient.  Let \(\mathcal O_4\) be the complete linearized
map from seam perturbations to every suffix consumer entering that endpoint.
Two sections are equivalent exactly when their difference lies in
\(\ker\mathcal O_4\).  Thus the minimal condition is

\[
\boxed{
\mathcal O_4\{\Phi_r(Q_4)-(C_4,K21_4)\}=0.}
\]

One same-site Bellman scalar is sufficient only if the remaining nonlinear
suffix never makes an unobserved direction visible.  With generic dense
mixing and neuron-specific AQ reset Jacobians, the backward orbit is Burnside
full, so this quotient generically returns to the complete typed interface.
Reopening with a smaller quotient requires an exact invariant/future-dark
certificate, not merely a scalar endpoint at the first seam.

## 2026-08-31 — universal raw-Gaussian AQ no-go from positive fourth cumulant

The earlier cubic-null discussion is necessary only if one asks to match
covariance and K3 while tolerating a non-Gaussian law.  Exact Gaussian
initialization has a much simpler universal obstruction.  Specializing the
full-AQ quartic formula to one variable gives

\[
\boxed{
\kappa_4(u^Tz+z^TAz-\operatorname{tr}A)
=48\{\operatorname{tr}(A^4)+u^TA^2u\}\ge0.}
\]

For real symmetric \(A\), equality holds if and only if \(A=0\).  Therefore
an AQ polynomial with an exactly Gaussian marginal has no quadratic part.
If a width-1024 AQ bank is jointly Gaussian, this applies to every coordinate,
so all \(A_i=0\) and

\[
\operatorname{rank}C\le r.
\]

Hence no r49, r54, or r55 AQ bank can exactly represent a full-rank
width-1024 Gaussian seed.  There is no exceptional nonfaithful or cubic-null
loophole: a cubic cancellation can still leave the strictly positive fourth
cumulant above.

The obstruction is quantitative.  Whiten any positive-definite target
covariance to \(C=I_n\).  The combined AQ feature map is then an isometry
from \(\mathbb R^n\) into \(\mathbb R^r\oplus\operatorname{Sym}_r\).  The
linear bank has rank at most r, so there is a subspace of dimension at least
\(n-r\) on which the combined linear coefficient vanishes.  For every unit
vector \(w\) in that subspace, the combined core \(A_w=\sum_iw_iA_i\)
satisfies

\[
2\operatorname{tr}(A_w^2)=1.
\]

Power-mean/Cauchy on its r eigenvalues gives

\[
\operatorname{tr}(A_w^4)
\ge\frac{\operatorname{tr}(A_w^2)^2}{r}
=\frac1{4r},
\]

and consequently

\[
\boxed{\kappa_4(G_w)\ge\frac{12}{r}.}
\]

At r55, at least 969 independent whitened directions have excess fourth
cumulant at least \(12/55\).  This is a law-level mismatch, not merely a
parameter-count warning.  It validates the L4 non-Gaussian suffix pivot as
the only current way to avoid raw initialization, while making the typed
L4 section receipt even more decisive.

The positivity is a global bank constraint, not just a marginal statement.
For every unit-index direction w,

\[
G_w=\sum_iw_iG_i
\]

is another AQ variable, and therefore

\[
\boxed{
\kappa_4(G_w,G_w,G_w,G_w)\ge0\qquad\text{for every }w.}
\]

Hence the AQ fourth-cumulant tensor is a nonnegative quartic form.  If a true
seam has a direction \(w\) with \(\kappa_4^{true}(w^4)<0\), then every AQ-law
approximation has directional error at least

\[
-\kappa_4^{true}(w^4).
\]

This obstruction holds at every r.  A two-point Gaussian-rest covariance
does not repair it; the negative-kurtosis sector must be owned by an
incumbent CP4/Q24 connected term or remain in the signed true-minus-AQ
residual.  The graph should therefore distinguish an AQ law image from a
connected-cumulant source model that is allowed a separately signed q4
correction.

The same identity is the exact H4 square lift.  For two unit-index
directions i,j define

\[
\Phi_{ij}
=\left(
\operatorname{sym}(A_iA_j),
\frac{A_iu_j+A_ju_i}{2}
\right)
\in\operatorname{Sym}_r\oplus\mathbb R^r.
\]

Its feature dimension is

\[
D_r=\frac{r(r+1)}2+r=\frac{r(r+3)}2.
\]

Polarizing the diagonal fourth-cumulant formula gives

\[
\boxed{
\begin{aligned}
\kappa_4(G_i,G_j,G_k,G_l)=16\{&
\langle\Phi_{ij},\Phi_{kl}\rangle
+\langle\Phi_{ik},\Phi_{jl}\rangle\\
&+\langle\Phi_{il},\Phi_{jk}\rangle\}.
\end{aligned}}
\]

On the diagonal this is
\(48(\|A(w)^2\|_F^2+\|A(w)u(w)\|^2)\), as required.  Therefore the complete
pre-reset AQ q4 tensor is a three-pairing Gram symmetrization of pair-feature
rank at most \(D_r\).  This is the precise relation between the AQ quartic,
the H4 square lift, and the degree-two matrix-polynomial product feature.

It is an ownership/rank theorem, not a free acquisition theorem.  Forming all
\(\Phi_{ij}\) generically requires the pairwise matrix products that the
budget audit already exposed.  A low-rank Bellman slice can contract them
without a full tensor, but a reusable forward q4 plane still needs a literal
resident-product or fusion receipt.

There is also a global variance-share statement.  In the whitened frame,

\[
I_n=C_{lin}+C_{quad},
\qquad C_{lin}=UU^T,
\qquad \operatorname{rank}C_{lin}\le r.
\]

Since \(0\preceq C_{lin}\preceq I\), one has

\[
\operatorname{tr}C_{lin}\le r,
\qquad
\boxed{\operatorname{tr}C_{quad}\ge n-r.}
\]

Thus r55 must encode at least \(969/1024\simeq94.63\%\) of total raw-seed
variance in quadratic chaos; r49 must encode at least 95.21%.  The
zero-bias parity theorem is a local expansion around an affine core and
cannot certify this seed: it is necessarily a large non-Gaussian
deformation, even before endpoint weighting.

## 2026-08-31 — hidden AQ-Gram cost: frozen versus self-consistent D10 frame

The 46.09 r55 suffix census has a previously implicit frame fork.  Let

\[
F_l\in\mathbb R^{1024\times1595}
\]

be the carried AQ feature bank.  The paid one-sided transition forms
\(F_{l+1}=W_l^TF_l\), but it does **not** form the pairwise AQ covariance

\[
G_{l+1}=F_{l+1}F_{l+1}^T.
\]

If the D10 Gaussian rest is evaluated in the self-consistently matched total
AQ covariance frame, its pairwise correlations require this Gram before
every reset that has a later nonlinear consumer.  A generic formation costs

\[
\frac{1595}{1024}=1.5576
\]

dense-GEMM equivalents per layer, exactly the same rectangular scale as the
one-sided transport.  The terminal L14 scalar uses
\(v^TFF^Tv=\|F^Tv\|^2\), so only nine suffix Grams are necessary.  The
missing charge is 14.019, so

\[
\boxed{46.09+14.019\simeq60.10>49.}
\]

With a scalar terminal, the analogous fourteen-transition charges use
thirteen Grams: 19.535 at r54 and 16.174 at r49.  A
right-hand side shaped like an existing D10 plane is not a fusion receipt:
the paid D10 conjugation and \(FF^T\) are different products.

There are therefore two distinct architectures:

1. **Frozen/incumbent Gaussian control-variate frame.**  D10 uses the
   already resident D11 covariance/correlation frame.  No AQ Gram is formed,
   so the 46.09 census survives.  But the rest is not self-consistently
   matched to the changed AQ trajectory; frame mismatch and non-Gaussian
   rest both enter the Bellman defect.
2. **Self-consistent matched frame.**  D10 recomputes its Gaussian rest from
   the AQ-updated total covariance.  This has the desired closure semantics,
   but exceeds the cap unless a literal paid product equals \(FF^T\), a
   lower-dimensional exact Gram theorem is proved, or D10 is replaced by a
   target-conditioned route with a complete future-dark certificate.

A final endpoint reduction \(v^TFF^Tv=\|F^Tv\|^2\) avoids the Gram for one
terminal scalar, but not for a rest source that passes through later
nonlinear resets.  Thus the frame choice is a parent premise of every
sub-49 AQ+D10 cost claim.

### Graph-ready split

- `AQ_D10_FROZEN_FRAME_COST`: 46.09 for the L4/r55 suffix, conditional on
  accepting frame mismatch as an error.
- `AQ_D10_SELFCONSISTENT_GRAM_COST`: 60.10 for the r55 suffix generically;
  closed under cap
  absent literal product fusion.
- `AQ_D10_FRAME_MISMATCH_BELLMAN_GATE`: open; centered parity is only a local
  aid and does not erase the mismatch.

### Exact attenuation of a frozen Gaussian-rest frame error

The frozen-frame branch has a useful exact Lipschitz constant.  For

\[
f_\alpha(X)=(X+\alpha)_+,
\qquad r_\alpha=P_{\ge3}f_\alpha,
\]

let \(R_{\alpha\beta}(\rho)\) be the covariance of the two rest functions
under a standard bivariate Gaussian pair of correlation \(\rho\).  The
Hermite coefficients satisfy

\[
c_1=\Phi(\alpha),
\qquad c_2=\phi(\alpha).
\]

The Gaussian Sobolev/Malliavin derivative energy of the omitted rest is
therefore

\[
\boxed{
d(\alpha):=\|D r_\alpha\|_2^2
=\Phi(\alpha)-\Phi(\alpha)^2-\phi(\alpha)^2.}
\]

Indeed the total derivative energy is
\(E1_{\{X>-\alpha\}}=\Phi(\alpha)\); the H1 and H2 contributions are
respectively \(c_1^2\) and \(c_2^2\).  Differentiating Mehler's series and
using the OU contraction gives

\[
\boxed{
|R_{\alpha\beta}(\rho_1)-R_{\alpha\beta}(\rho_0)|
\le
\sqrt{d(\alpha)d(\beta)},|\rho_1-\rho_0|.}
\]

In fact the centered value is the **global maximum**.  The function is even,
and for \(a\ge0\)

\[
d'(a)=-\phi(a)g(a),
\qquad
g(a)=2\Phi(a)-1-2a\phi(a),
\]

while \(g(0)=0\) and \(g'(a)=2a^2\phi(a)\ge0\).  Hence

\[
\boxed{
0\le d(\alpha)\le d(0)
=\frac12-\frac14-\frac1{2\pi}
\simeq0.090845.
}
\]

For scaled variables \(\sigma_i(X+\alpha_i)_+\), and with marginal scales
held fixed, the \(\sigma_i\sigma_j\) factors cancel when correlation change
is written as raw covariance change.  Hence

\[
|\Delta K_{ij}^{G,>2}|
\le\sqrt{d(\alpha_i)d(\alpha_j)},|\Delta C_{ij}|.
\]

For a frozen Bellman plane \(\Lambda\), this becomes the directly usable
receipt

\[
\boxed{
|\langle\Lambda,\Delta K^{G,>2}\rangle|
\le
\langle |\Lambda|,
L\odot|\Delta C|\rangle,
\quad L_{ij}=\sqrt{d(\alpha_i)d(\alpha_j)}.}
\]

Thus a frozen Gaussian-rest frame transmits at most about 9.1% of its
correlation-frame discrepancy locally for **every standardized mean**, not
only at the centered gate.  This does not make the frame
exact, does not cover changed marginal scales without their derivative
terms, and must still be accumulated through the Bellman/Duhamel gate.  It
does, however, give the cheap 46.09 branch a substantially sharper analytic
error certificate than the generic 1-Lipschitz ReLU-kernel bound.

## 2026-08-31 — exact affine-rank obstruction and near-affine Schur section gate

The Stiefel condition above now has an exact differential refinement.  At an
affine tuple \(M_i=0\), let \(U\in\mathbb R^{n\times r}\) contain the linear
rows and assume full column rank.  The covariance differential is

\[
DC=hU^T+Uh^T,
\]

so its range has dimension

\[
nr-\frac{r(r-1)}2
\]

and its normal space is

\[
\mathcal N_U=\{Z=Z^T:ZU=0\},
\qquad
\dim\mathcal N_U=\frac{(n-r)(n-r+1)}2.
\]

At the same affine point, the K21 differential in quadratic directions is

\[
\boxed{
(F_UH)_{ij}=2u_i^TH_ju_i+4u_i^TH_iu_j.}
\]

Under the explicit Veronese minor conditions

\[
\operatorname{rank}\{u_iu_i^T\}_{i=1}^n=n,
\]

and, for every p, rank n-1 of the projected outer products in
\(\operatorname{Sym}(u_p^\perp)\), \(F_U\) is onto all \(n^2\) oriented
K21 entries.  Even then no affine local section exists, because the entire
\(\mathcal N_U\) covariance normal is missed.

For r54 the exact numbers are

\[
\operatorname{rank}DC=53,865,
\qquad
\dim\mathcal N_U=470,935,
\]

so the complete affine Jacobian has rank 1,102,441 and cokernel 470,935.
This is distinct from the fourth-cumulant Gaussian no-go: it proves failure
even for the weaker local \((C,K21)\) section map at \(M=0\), while the
positive-\(\kappa_4\) theorem proves failure of exact law matching.

For a near-affine prefix \(M_i=tN_i\), eliminate the surjective K21 block and
define

\[
\mathcal L_N(H)_{ij}
=2\operatorname{tr}(H_iN_j+N_iH_j),
\]

\[
\boxed{
\mathscr S_{U,N}
=\Pi_U\mathcal L_N|_{\ker F_U}:
\ker F_U\longrightarrow\mathcal N_U,}
\]

where \(\Pi_U(Y)=P_UYP_U\) and
\(P_U=I-U(U^TU)^{-1}U^T\).  The near-affine joint map is a submersion for
all sufficiently small nonzero t, away from isolated minor roots, if and
only if

\[
\boxed{\operatorname{rank}\mathscr S_{U,N}=\dim\mathcal N_U.}
\]

For r54 its domain and codomain dimensions are

\[
472,064\longrightarrow470,935,
\]

leaving exactly 1,129 dimensions.  For r55 the corresponding dimensions are

\[
528,384\longrightarrow469,965,
\]

leaving 58,419.  Thus r55 has materially more algebraic slack, but still
needs a nonzero Schur minor at the actual prefix.

The new singular values obey

\[
\boxed{
\sigma_{new}(D\Phi_r)
=|t|\,\sigma(\mathscr S_{U,N})+O(t^2).}
\]

Consequently a section arbitrarily close to affine is necessarily
ill-conditioned like \(1/(|t|\sigma_{min})\).  A small quadratic prefix is
not automatically a gentle bridge out of the affine obstruction.

### Atomic graph claims

- `R54_AFFINE_SEAM_RANK_OBSTRUCTION`: proved, 470,935-dimensional cokernel.
- `R54_NEAR_AFFINE_SCHUR_SECTION_GATE`: exact iff gate
  \(\operatorname{rank}\mathscr S=470,935\), current nonzero minor open.
- `R55_NEAR_AFFINE_SCHUR_SECTION_GATE`: exact iff gate
  \(\operatorname{rank}\mathscr S=469,965\), with 58,419 domain slack;
  actual-prefix minor open.
- `AQ_NEAR_AFFINE_CONDITIONING`: proved \(|t|\sigma(\mathscr S)\) scaling.

### r48 target-conditioned Schur section: the deficit is a selectable covariance normal

The same decomposition is especially informative at r48.  The affine K21
Veronese block can still be onto because

\[
1024\le\dim\operatorname{Sym}_{48}=1176,
\qquad
1023\le\dim\operatorname{Sym}_{47}=1128.
\]

After solving all \(n^2\) oriented K21 tangents, its quadratic kernel has

\[
\dim\ker F_U=1024\cdot1176-1024^2=155,648.
\]

The affine covariance normal has

\[
\dim\mathcal N_U
=\frac{(1024-48)(1025-48)}2
=476,776.
\]

Thus the near-affine r48 Schur map is

\[
\boxed{
\mathscr S_{U,N}:\mathbb R^{155,648}
\longrightarrow\mathbb R^{476,776},}
\]

and a maximal-rank image leaves

\[
476,776-155,648=321,128
\]

unmatched covariance-normal directions.  This is exactly the prior flat
r48 parameter deficit.  It is not an undifferentiated loss of C and K21:
under the Veronese minors, K21 can be matched first and the remaining choice
is which 155,648-dimensional covariance-normal image to retain.

Let \(\mathcal B\subseteq\mathcal N_U\) be the span of the complete backward
Bellman covariance queries that the seam must serve.  The exact tangent
quotient gate is

\[
\boxed{
\operatorname{rank}(P_{\mathcal B}\mathscr S_{U,N})
=\dim\mathcal B.}
\]

It is possible only if \(\dim\mathcal B\le155,648\).  For one fixed covector
\(\lambda\in\mathcal N_U\), the nonzero-value condition reduces to

\[
\mathscr S_{U,N}^*\lambda\ne0,
\]

and the near-affine conditioning scales as
\(|t|\|\mathscr S^*\lambda\|\).  Therefore a target-aligned r48 prefix can
in principle outperform the flat 0.20410 tail when the corpus/future
Bellman span is much smaller than 155,648.  This is particularly relevant
because the certified value is variance-channel dominated.

This is an exact local tangent theorem, not a deployable section by itself.
It still requires a target-blind/generalizing construction of \(N\), a fixed
gauge, actual-cut membership, and a nonzero restricted minor.  But it gives
the correct objective for the leading r48 branch: match the complete signed
K21 tangent, then orient the accessible covariance-normal Schur image toward
the complete future Bellman span.

### Graph node `R48_TARGET_SCHUR_SECTION`

- status: exact dimension/iff theorem, construction open;
- K21 block: generically onto under explicit Veronese minors;
- covariance Schur: 155,648 to 476,776, flat deficit 321,128;
- task pass: \(P_{\mathcal B}\mathscr S\) onto;
- conditioning: restricted smallest singular value times \(|t|\);
- dependency: complete Bellman span, not a same-site scalar proxy.

### Joint target quotient: do not spend a million directions on unused K21 entries

The preceding construction deliberately matched the full K21 plane first.
That is conservative for a fixed board objective.  Let

\[
\mathcal B_C\subseteq\operatorname{Sym}_n,
\qquad
\mathcal B_K\subseteq\mathbb R^{n\times n}
\]

be the **complete future** Bellman query spans for covariance and K21,
including every suffix consumer and corpus endpoint after old subtraction.
At a near-affine prefix the relevant quadratic-direction map is

\[
\boxed{
\mathcal T_{U,N}(H)
=\left(P_{\mathcal B_C}\mathcal L_NH,
       P_{\mathcal B_K}F_UH\right).}
\]

Together with the h-directions spanning the rank-r covariance tangent, the
AQ seam is exact on the selected task quotient if and only if this restricted
map is onto the remaining query coordinates.  In adjoint form, the exact
gate is

\[
\boxed{
\mathcal L_N^*Z+F_U^*\Lambda=0,
\quad (Z,\Lambda)\in\mathcal B_C\oplus\mathcal B_K
\quad\Longrightarrow\quad (Z,\Lambda)=0,}
\]

after the horizontal gauge and after removing covariance-tangent queries
already served by h.

For r48 the H-domain has dimension

\[
1024\cdot1176=1,204,224.
\]

Thus the elementary target-capacity condition is only

\[
\dim\mathcal B_C^{normal}+\dim\mathcal B_K
\le1,204,224,
\]

not full-interface surjectivity onto 1,573,376 coordinates.  When the
complete endpoint/corpus backward span is much smaller, the flat 321,128
tail is a conservative generic-interface penalty.  The correct r48 design
allocates H jointly toward the variance queries carrying 93% of certified
value and only those signed-K21 directions that actually return future
value.

This theorem does not permit a same-site K21 scalar proxy.  The spaces must
be the closed backward orbits through all future reset consumers; generic
Burnside expansion can still make them large.  It does, however, identify a
proof-level redemption route: certify the restricted adjoint injectivity
rather than solve an unnecessary million-coordinate K21 section.

### Graph node `R48_JOINT_BELLMAN_SECTION`

- status: exact iff theorem, actual restricted minor open;
- map: \((P_{B_C}\mathcal L_N,P_{B_K}F_U)\);
- benefit: trades unused K21 interface directions directly for high-value
  covariance-normal directions;
- close: restricted adjoint has a nonzero kernel or complete backward span
  exceeds/rank-defeats the H domain.

## 2026-08-31 — goal-scale retention gate for any AQ replacement

Let \(R_0=2.9409\times10^{-8}\), let the stated near-exact architecture
oracle be \(R_{or}\in[7.09,9.89]\times10^{-9}\), and suppose an approximate
replacement retains a fraction \(\eta\) of the oracle improvement.  Then

\[
R(\eta)=R_0-\eta(R_0-R_{or}).
\]

The raw target \(R(\eta)\le10^{-8}\) requires

\[
\boxed{
\eta\ge\frac{R_0-10^{-8}}{R_0-R_{or}}.}
\]

At the two oracle endpoints this is respectively

\[
\eta\ge0.8696
\qquad\text{and}\qquad
\eta\ge0.9944.
\]

Thus the entire seam/reset/frame/provider stack may lose only about 13.0% of
the best-oracle improvement, and only about 0.56% if the relevant oracle is
the upper end of the quoted range.  This is why neither an 18.7%-scale local
receipt nor the 9.0845% frozen-rest attenuation alone proves the campaign
goal.  The r55 suffix must be nearly complete in the endpoint Bellman metric.

## 2026-08-31 — self-consistent Gram rank frontier for the raw 1e-8 target

If the matched D10 frame is mandatory, both the one-sided AQ transport and
the AQ Gram must be charged.  Reducing r creates a distinct Pareto frontier.
For four early D14 transitions, ten AQ/D10 suffix transitions, and 32 scalar
reset nodes, there are ten one-sided AQ transports but only nine AQ Gram
formations: the L14 scalar endpoint uses
\(v^TFF^Tv=\|F^Tv\|^2\) and needs no terminal plane.  The optimistic census is

\[
\boxed{
N_r=8+20
+19\frac{D_r}{1024}
+10\frac{r^3}{1024^2}
+320\frac{r^2}{1024^2},
\qquad D_r=\frac{r(r+3)}2.}
\]

The flat static interface tail remains

\[
e_r=\frac{q-p_r^{AQ}}q,
\qquad
p_r^{AQ}=1024D_r-\frac{r(r-1)}2,
\quad q=1,573,376.
\]

The relevant points are

\[
\begin{array}{c|c|c|c|c}
r&D_r&N_r&e_r&R_0e_r\\ \hline
43&989&47.67&0.35690&1.0496\times10^{-8}\\
44&1034&48.59&0.32764&9.636\times10^{-9}\\
45&1080&49.53&0.29773&8.756\times10^{-9}\\
46&1127&50.49&0.26717&7.857\times10^{-9}
\end{array}
\]

Since \(10^{-8}/R_0\simeq0.340032\), r44 is the first rank whose flat
static capacity lies below the campaign raw target and whose fully
self-consistent nine interior frames fit under 49; it leaves only

\[
0.340032-0.327643\simeq0.01239
\]

normalized error energy.  Rank45 leaves about 0.04230 and costs 49.53 with
all nine frames matched.  Freezing only the least valuable nonterminal frame
saves \(1080/1024=1.0547\), giving 48.47 while incurring one attenuated frame
defect.

This table is not an end-to-end certificate: the flat target orientation is
heuristic, reset errors recur at ten layers, and r43--45 cannot use the
r54/r55 exact-generic section count.  It does establish the correct design
choice under exact-frame semantics.  The alternatives are not “r55 or
nothing”:

- r55/frozen D10 frame: 46.09, exact-generic static capacity conditional,
  frame mismatch open;
- r44/self-consistent frame: 48.59, flat raw just below 1e-8, almost no error
  budget;
- r45/eight-matched-one-frozen frame: 48.47, meaningful error headroom and
  one frozen-frame Bellman defect.

Held dtype/util savings may change adjusted economics, but they must be
applied to the literal packaged operations rather than multiplied into this
census by assumption.

### Hybrid exact/frozen-frame Pareto frontier

At fixed r, omitting one nonterminal AQ Gram saves \(D_r/1024\) equivalents
and turns that layer from matched-frame to frozen-frame semantics.  Applying
the exact 49 line to the nine possible interior Grams gives

\[
\begin{array}{c|c|c|c|c}
r&\text{matched frames}&\text{frozen frames}&N_r&e_r\\ \hline
44&9&0&48.59&0.32764\\
45&8&1&48.47&0.29773\\
46&7&2&48.28&0.26717\\
48&6&3&48.88&0.20410\\
49&5&4&48.52&0.17159\\
51&4&5&48.89&0.10462\\
52&3&6&48.32&0.07015\\
54&2&7&48.43&0\ \text{nominally}\\
55&1&8&47.64&0\ \text{nominally}
\end{array}
\]

Rows r47, r50, and r53 are Pareto-dominated by the next displayed rank:
they require the same number of frozen frames while retaining less static
capacity.  The zero tails at r54/r55 remain conditional on the section
minor; they are not certified exact seams.

For any fixed r, let \(\epsilon_l^{fr}\) be the local frozen-frame error,
let \(P_l\) be its future linearized map, and define the normalized endpoint
amplitude

\[
a_l=\frac{\|P_l\epsilon_l^{fr}\|_H}{\|y\|_H}.
\]

The exact rest Lipschitz theorem supplies the local bound

\[
\|\epsilon_l^{fr}\|_F
\le0.090845\|\Delta C_l\|_F,
\]

and hence

\[
a_l\le
0.090845\frac{\|P_l\|_{F\to H}\|\Delta C_l\|_F}{\|y\|_H}.
\]

A resident scalar Bellman plane additionally supplies
\(|\langle\Lambda_l,\epsilon_l^{fr}\rangle|/\|y\|_H^2\) as a signed
first-order value receipt, but that scalar must not be called an endpoint
norm without a norm bridge.

All Gram costs are equal across layers at that r, so the exact minimax
allocation is to match the layers with the largest certified \(a_l\) and
freeze those with the smallest.  With no cross information, the coherent
raw gate is

\[
\boxed{
\sqrt{e_r}+\sum_{l\in\mathcal F}a_l
+a_{other}\le\sqrt{10^{-8}/R_0}.}
\]

The block-elliptope/RSS receipt can replace the sum when cross geometry is
available.

The independent near-exact architecture oracle lies at normalized raw
\(R_{or}/R_0\in[0.241,0.336]\).  Static tails much below about 0.24 therefore
buy capacity that the stated oracle says other errors cannot use.  This
selects r48 as the natural mathematical sweet spot:

\[
\boxed{
\text{r48, six matched interior frames, three lowest-Bellman frames frozen,}
\quad N\simeq48.88.}
\]

Its flat static floor is about \(6.00\times10^{-9}\), already below the best
quoted oracle, while it retains twice as many exact rest frames as r54 and
six times as many as r55.  Promotion still requires a lawful target-weighted
L4 r48 projection (it cannot exactly section generic \((C,K21)\)), its
smallest response singular value, and the three frozen-frame Bellman
receipts.  This is a candidate architecture, not a raw certificate.

The 48.88 census contains no generic seam solve.  It leaves only about 0.12
equivalent, whereas literal D14 supplies a plane, not a covariance square
root or a Stiefel orientation.  Consequently

\[
\boxed{
\text{AQ L4 section acquisition/factor residency is a blocking parent of
the entire Pareto table}.}
\]

A resident factor, a fused square root, or a fixed target-blind orientation
map from an existing D11 factor could clear this parent.  A per-net generic
nonlinear \(K21\) orientation solve cannot be declared free merely because a
local section exists.  If it has material cost, every displayed row must add
that cost or remove another matched frame.

## 2026-08-31 — do not mix Galerkin-law and connected-cumulant semantics

Two distinct AQ+D10 architectures have appeared in the derivations.

### A. AQ law/Galerkin replacement

Here the carried object is literally

\[
X_i=P_{\le2}(h_i)_+,
\]

its covariance contribution is \(FF^T\), and D10 adds a higher-chaos
Gaussian control variate.  Claims about the AQ feature dimension, the
\((C,K21)\) section, the positive fourth-cumulant seed obstruction, and the
r44--r48 Gram frontier belong to this branch.  Exact self-consistent plane
semantics require the nine interior Grams unless a literal fusion is proved.

### B. Gaussian-base connected-cumulant correction

Here D10 owns and propagates the total Gaussian covariance plane, while AQ
cores are only a structured source model for connected \(\kappa_3,\kappa_4\)
corrections injected through the existing D11 response derivatives.  No
literal \(FF^T\) plane need be formed, so the cheap 46.09-style census can
remain possible.  But the AQ cores are then not a replacement law whose Gram
is the covariance.  The static \((C,K21)\) capacity floor, exact-Gaussian
seed arguments, and full-law q211 ownership cannot be credited to this
branch without a separate identification theorem.  Its binding error is the
complete connected \(q\ge5\), nonlinear/secant, and true-minus-AQ source
remainder.

The branches may share local Fourier/reset formulas and a typed core bank,
but their value and cost receipts are mutually noninterchangeable:

\[
\boxed{
\text{A's full-law capacity/value cannot be combined with B's no-Gram cost.}}
\]

### Atomic graph split

- `AQ_GALERKIN_LAW_BRANCH`: full-law semantics, Gram/section costs binding.
- `AQ_CONNECTED_CUMULANT_BRANCH`: no-Gram source semantics, higher connected
  remainder and response completeness binding.
- `AQ_BRANCH_SEMANTIC_NO_DOUBLE_COUNT`: proved separation; shared formulas do
  not authorize mixed accounting.

## 2026-08-31 — exact effective-rank bound for the connected cumulant tail

Branch B has a clean scalar source-radius theorem.  For

\[
G=u^Tz+z^TAz-\operatorname{tr}A,
\qquad V=\operatorname{Var}G=\|u\|^2+2\operatorname{tr}A^2,
\]

the cumulant generating function gives, for every \(q\ge3\),

\[
\boxed{
\kappa_q(G)
=2^{q-1}(q-1)!\operatorname{tr}A^q
+2^{q-3}q!u^TA^{q-2}u.}
\]

Define the spectral concentration parameter

\[
\delta=\frac{2\|A\|_{op}}{\sqrt V}
=\sqrt{\frac2{r_{eff}}},
\qquad
r_{eff}=\frac{V}{2\|A\|_{op}^2}.
\]

Using
\(|\operatorname{tr}A^q|le\|A\|_{op}^{q-2}\operatorname{tr}A^2\)
and \(|u^TA^{q-2}u|\le\|A\|_{op}^{q-2}\|u\|^2\) yields

\[
\boxed{
\frac{|\kappa_q(G)|}{q!V^{q/2}}
\le\left(\frac12+\frac1q\right)\delta^{q-2}.}
\]

Thus the complete connected generator tail beyond q4 satisfies, for
\(\delta<1\),

\[
\boxed{
\sum_{q\ge5}rac{|\kappa_q(G)|}{q!V^{q/2}}
\le\frac{0.7\,\delta^3}{1-\delta}.}
\]

At \(r_{eff}=48\), \(\delta=\sqrt{2/48}\simeq0.2041\) and the bound is
about 0.0075; at effective rank55 it is about 0.006.  This is a rigorous
connected-source coefficient bound, not a response bound.

If only the first-order q3/q4 response is retained, put

\[
S\le\frac56\delta+\frac34\delta^2
+\frac{0.7\delta^3}{1-\delta}.
\]

In any analytic response norm for which the standardized cumulant
differential operators have norm at most their scalar coefficients, the
missing nonlinear exponential terms obey

\[
\|e^{\mathcal A}-I-\mathcal A_3-\mathcal A_4\|
\le e^S-1-S.
\]

At effective rank48 this scalar majorant is about 0.023.  A joint recurrence
that resums all powers of the owned q3/q4 operators removes this nonlinear
piece and leaves only the 0.0075 connected-high-q tail.

The qualification is binding for ReLU: its derivatives are distributional,
and multivariate target contractions can amplify the scalar coefficients.
Promotion requires the tilted-Gaussian/surface Bellman norm bridge.  Still,
the theorem gives an exact fail-fast spectral gate:

- diffuse \(r_{eff}\gtrsim48\): q>=5 connected source is sub-percent at the
  generator level;
- concentrated \(\delta\to1\): the cheap q3/q4 closure loses its analytic
  tail certificate regardless of nominal latent rank.

## 2026-08-31 — correction: the connected-cumulant series has zero ReLU-response radius

The preceding generator-tail estimate is valid as a statement about the
log-characteristic function inside \(|\delta t|<1\), but it cannot be
promoted by an abstract exponential-operator norm for ReLU.  There is an
exact one-dimensional counterexample.

Let

\[
X_r=\frac{\chi_r^2-r}{\sqrt{2r}},\qquad
\delta=\sqrt{\frac2r}.
\]

Its standardized cumulants are

\[
\kappa_q(X_r)=(q-1)!\,\delta^{q-2},\qquad q\ge3.
\]

For \(f(x)=x_+\) and \(Z\sim N(0,1)\), the formal first-order cumulant
response of order \(q\) is

\[
a_q=\frac{\kappa_q(X_r)}{q!}\,\mathbb E f^{(q)}(Z).
\]

Since \(f^{(q)}=\delta_0^{(q-2)}\) for \(q\ge2\), odd terms vanish at a
centered threshold and, for \(q=2m\),

\[
|a_{2m}|=
\frac{\phi(0)}{2m}(2m-3)!!\,\delta^{2m-2}.
\]

Therefore

\[
\frac{|a_{2m+2}|}{|a_{2m}|}
=\delta^2\frac{(2m-1)(2m)}{2m+2}
\longrightarrow\infty.
\]

Thus the ReLU cumulant-response series has **zero radius of convergence for
every nonzero quadratic spectral radius**, even though the connected
generator coefficients have the geometric bound derived above.  A nonzero
bias is worse in the sense that the odd Hermite derivatives no longer
vanish.  Pairwise ReLU covariance inherits the same obstruction by holding
one coordinate affine/nonzero.

Consequences:

1. The values 0.0075 (connected \(q\ge5\) generator tail) and 0.023
   (formal nonlinear majorant) are spectral diagnostics only.  They are not
   raw-MSE, reset, or Bellman error certificates.
2. A q3/q4 connected-cumulant branch needs a **finite-remainder theorem**:
   a characteristic-function contour bound, a saddlepoint/Edgeworth
   remainder with an explicit threshold-density constant, or an exact
   finite-secant reset.  Infinite operator exponentiation on ReLU is not a
   lawful proof route.
3. The formal series is asymptotic.  In the pure chi-square witness the
   smallest terms occur around \(m\asymp(2\delta^2)^{-1}=r/4\); low orders
   can still be accurate at r48--55, but that fact requires a finite-order
   remainder rather than summing the divergent tail.

This is an atomic close/reopen gate:

- `AQ_CONNECTED_CUMULANT_ANALYTIC_EXPONENT`: **CLOSED** by the exact
  chi-square/ReLU witness above.
- `AQ_CONNECTED_CUMULANT_FINITE_REMAINDER`: **OPEN**; reopen only with a
  finite contour/saddlepoint/secant bound in the actual biased bivariate
  reset and its Bellman contraction.

## 2026-08-31 — Veronese-null refinement of the target-conditioned AQ seam

The r55 resident-adjoint audit proved a universal first-order obstruction
which also sharpens the r48 target-quotient proposal.  The useful consequence
is not merely “full section fails”: it gives the exact large covariance
subspace which a target-conditioned seam cannot see.

Let \(U\in\mathbb R^{n\times r}\) have full column rank,
\(L=\operatorname{col}U\), \(K=L^\perp\), \(m=n-r\), and
\(E=\operatorname{Sym}^2L\), \(d=r(r+1)/2\).  At the affine seam, full K21
surjectivity forces the Veronese evaluation map

\[
\mathcal Q_U:E\to\mathbb R^n,
\qquad (\mathcal Q_UG)_i=u_i^TGu_i
\]

to be onto.  Hence \(E_0=\ker\mathcal Q_U\) has dimension \(p=d-n\).
After eliminating all K21 tangents, only the component

\[
N_0\in K\otimes E_0\simeq\mathbb R^{m\times p}
\]

of a quadratic prefix can move the remaining covariance normal.  If
\(R=\ker N_0^T\subset K\), then

\[
\dim R\ge(m-p)_+,
\qquad
\operatorname{Sym}R\subseteq\ker\mathscr S_{U,N}^*.
\]

Consequently the first-order covariance-normal rank is bounded sharply by

\[
\boxed{
\operatorname{rank}\mathscr S_{U,N}
\le \frac{m(m+1)}2-
\frac{(m-p)_+(m-p+1)_+}{2}.}
\]

For a complete future covariance-query space \(B_C\subseteq\operatorname{Sym}K\),
an exact target-conditioned seam after full K21 matching therefore requires

\[
\boxed{B_C\cap\operatorname{Sym}R=\{0\},}
\]

and in particular \(\dim B_C\) cannot exceed the displayed visible rank.
For a single Bellman covector \(Z\), the gate is the concrete scalar/matrix
condition \(P_R ZP_R\ne Z\) in the weak sense that \(Z\notin\operatorname{Sym}R\);
uniform conditioning additionally requires a positive smallest singular
value of \(Z\mapsto[ZN_0]\) on the actual query span.

At \(n=1024\):

\[
\begin{array}{c|c|c|c|c|c}
r&m&p&\dim R\text{ at least}&\dim\operatorname{Sym}R\text{ at least}
&\text{visible rank at most}\\ \hline
48&976&152&824&339900&136876\\
49&975&201&774&299925&175875\\
54&970&461&509&129795&341140\\
55&969&516&453&102831&367134
\end{array}
\]

This supersedes the crude r48 source count \(155648\): after insisting on
full K21 matching, at most \(136876\) independent covariance-normal queries
are first-order reachable, and an explicit \(339900\)-dimensional normal
subspace is invisible.  It does **not** kill a scalar or genuinely thin
future Bellman quotient; it makes its orientation relative to the actual
\(R\) the decisive certificate.

The same formula shows why nominal dimension surplus at r55 was misleading.
Full first-order covariance normal surjectivity is universally impossible
until

\[
\frac{r(r+3)}2\ge2n;
\]

for \(n=1024\), r63 is the first rank not killed by this obstruction.  Any
r48--r55 near-affine full-section claim is therefore closed.  A finite
nonaffine seam can only reopen through higher odd Schur orders and must carry
their \(t^{-3}\) (or worse) conditioning receipt.

Atomic status:

- `AQ_R48_FULL_K21_TARGET_COVARIANCE_QUOTIENT`: **OPEN only conditionally**
  on \(B_C\cap\operatorname{Sym}R=0\), a restricted singular value, and seam
  acquisition cost.
- `AQ_R48_R55_NEAR_AFFINE_FULL_SECTION`: **CLOSED at first order** by the
  Veronese-null invisible normal.
- `AQ_GENERIC_FIRST_ORDER_SECTION_THRESHOLD`: r63, already incompatible
  with the current under-49 full-law census unless a major fusion removes
  other costs.

## 2026-08-31 — exact ownership boundary for the H4 square-lift vector leg

In the half-scaled AQ convention, the quartic square-lift uses

\[
\Psi_{ij}=(\operatorname{sym}(A_iA_j),\ b_{ij}),
\qquad b_{ij}=A_i u_j+A_j u_i.
\]

The vector leg is not automatically owned by a scalar H3 tensor or by the
repeated K21 plane.  The exact factorization makes the missing premise
explicit.  Let \(U\in\mathbb R^{n\times r}\) have rows \(u_i^T\), put

\[
C_L=UU^T,
\qquad B_i=UA_iU^T,
\qquad (B_i)_{jk}=u_j^TA_i u_k.
\]

If \(U\) has full latent column rank, then

\[
U^TC_L^+U=I_r
\]

and hence

\[
\boxed{UA_iA_kU^T=B_iC_L^+B_k.}
\]

Therefore every vector-leg Gram is a sum of four entries of such products:

\[
\begin{aligned}
\langle b_{ij},b_{kl}\rangle
={}&u_j^TA_iA_k u_l+u_j^TA_iA_lu_k\\
&+u_i^TA_jA_k u_l+u_i^TA_jA_lu_k.
\end{aligned}
\]

This gives an exact reuse route **only if** the resident state exposes the
unsymmetrized mixed-cubic faces \(B_i\) and the \(C_L^+\) action (or literal
vectors \(A_i u_j\)).  The symmetric H3 tensor contains only combinations

\[
B_i(j,k)+B_j(i,k)+B_k(i,j)

\]

together with the pure-core trace term; K21 contains fewer repeated
combinations.  Neither object identifies the individual \(B_i\) in general.
The pure-core NCQC q211 formula omits this affine vector leg entirely.

If \(U\) is rank deficient, the obstruction is stronger: components of
\(A_i u_j\) orthogonal to \(\operatorname{span}\{u_k\}\) are invisible to
every mixed H3 face but still contribute to \(\langle b,b\rangle\).

Atomic gate:

- `AQ_H4_VECTOR_LEG_H3_REUSE`: **conditional**, exact iff a complete
  unsymmetrized \(B_i\) provider plus \(C_L^+\) is literally resident and its
  products fit the census; repeated K21/symmetric H3 alone does not clear it.

## 2026-08-31 — lawful finite q3/q4 remainder via cavity Lindeberg replacement

There is a finite replacement theorem which avoids the divergent ReLU
cumulant series and states exactly what the cheap connected-source branch
would have to acquire.

Let \(Y_1,\ldots,Y_s\) be independent centered scalar or vector summands,
let \(G_j\) be independent Gaussians with
\(\operatorname{Cov}G_j=\operatorname{Cov}Y_j\), and suppose the reset input
also contains an independent nondegenerate Gaussian cavity
\(Z_0\sim N(0,\Sigma_0)\).  Put

\[
\bar f(x)=\mathbb E[f(x+Z_0)].
\]

For the hybrid background

\[
H_j=Y_1+\cdots+Y_{j-1}+G_{j+1}+\cdots+G_s,
\]

Taylor's theorem applied inside the exact Lindeberg telescoping sum gives

\[
\begin{aligned}
&\mathbb E f\left(Z_0+\sum_jY_j\right)
-\mathbb E f\left(Z_0+\sum_jG_j\right)\\
={}&\sum_j\frac1{3!}\,
  \mathbb E\,D^3\bar f(H_j)[Y_j^{\otimes3}]
-\sum_j\frac1{3!}\,
  \mathbb E\,D^3\bar f(H_j)[G_j^{\otimes3}]\\
&+\sum_j\frac1{4!}\,
  \mathbb E\,D^4\bar f(H_j)[Y_j^{\otimes4}-G_j^{\otimes4}]
+\mathcal R_5.
\end{aligned}
\]

The displayed Gaussian cubic term is zero.  In one dimension the fourth
difference is exactly the fourth cumulant, so these are literal q3/q4
sources.  More generally they are the third and fourth cumulant tensors.
The finite remainder obeys

\[
\boxed{
|\mathcal R_5|
\le\frac{M_5(\bar f)}{5!}
\sum_j\left(\mathbb E\|Y_j\|^5+\mathbb E\|G_j\|^5\right),
\quad
M_5(\bar f)=\sup_x\|D^5\bar f(x)\|_{op}.}
\]

This is an ordinary finite Taylor remainder, not an Edgeworth asymptotic
series.  For a scalar hinge \(f_a(x)=(x+a)_+\) and cavity variance \(\tau^2\),

\[
M_5(\bar f)=\tau^{-4}
\sup_x|H_3(x)\phi(x)|
=c_3\tau^{-4},
\]

where

\[
c_3=\frac{\sqrt{6(3-\sqrt6)}}{\sqrt{2\pi}}
e^{-(3-\sqrt6)/2}.
\]

For the bivariate pair reset
\(f_{ab}(x,y)=(x+a)_+(y+b)_+\), the same theorem holds with the actual
two-dimensional cavity covariance and its finite fifth Fréchet-derivative
norm.  Under a common rescaling \(\Sigma_0\mapsto\tau^2\Sigma_0\), the
homogeneous pair-reset part scales as \(M_5=O(\tau^{-3})\); degeneracy of the
cavity makes the constant diverge and is therefore a real gate.

For an AQ eigen-coordinate,

\[
Y_j=u_jZ_j+\lambda_j(Z_j^2-1),
\]

the fifth moments in the certificate are one-dimensional, exact, and
additive across coordinates.  A coarse label-free fallback follows from
Gaussian hypercontractivity,

\[
\|Y_j\|_5\le4\|Y_j\|_2,
\]

so the remainder is controlled by the \(5/2\)-leverage sum
\(\sum_j\operatorname{Var}(Y_j)^{5/2}\), not by latent rank alone.

The theorem exposes two previously hidden premises:

1. q3/q4 corrections are evaluated at the **hybrid backgrounds \(H_j\)**.
   Replacing all of them by one Gaussian-background derivative omits the
   cross/nonlinear q3/q4 interactions; those are not part of \(\mathcal R_5\).
2. A useful bound needs a certified independent Gaussian cavity (or an
   equivalent density-derivative bound) in every biased bivariate reset seen
   by the Bellman action.  A covariance plane by itself is not proof that
   such an independent summand exists.

Thus a lawful reopen path for the connected branch is either:

- implement/resum the hybrid q3/q4 sources and certify the fifth-moment
  remainder above; or
- evaluate an exact finite AQ reset/secant which subsumes the hybrids.

A single Gaussian-background q3/q4 injection with the geometric cumulant
tail is still **HOLD**.

### Pairwise qualification: common reducing blocks, not scalar effective rank

The independent-summand theorem is automatically available for one scalar AQ
form, because its single symmetric matrix can be diagonalized.  It is **not**
automatically available for the bivariate reset which controls covariance.

For a family of symmetric cores \(A_1,\ldots,A_p\), an orthogonal
decomposition

\[
\mathbb R^r=V_1\oplus\cdots\oplus V_s
\]

makes the vector of quadratic forms a sum of independent Gaussian-block
summands if and only if every \(V_\alpha\) is a common reducing subspace of
every \(A_i\).  Equivalently, the block projections are projections in the
commutant of the real star-algebra

\[
\mathfrak A=\operatorname{alg}^*(A_1,\ldots,A_p).
\]

The finest lawful independent decomposition is therefore the central/reducing
block decomposition of \(\mathfrak A\), not the separate eigendecomposition
of each core.  A generic pair of symmetric matrices generates the full
matrix algebra and is irreducible by Burnside; then \(s=1\).  Scalar spectral
diffuseness \(r_{eff}\simeq48\) gives no small bivariate Lindeberg summands in
that case.

If \(v_\alpha\) denotes the covariance scale of common block \(V_\alpha\),
the finite fifth-order certificate contains

\[
\sum_\alpha v_\alpha^{5/2}
\le (\max_\alpha v_\alpha)^{3/2}\sum_\alpha v_\alpha.
\]

Writing

\[
\eta_{block}=\frac{\max_\alpha v_\alpha}{\sum_\alpha v_\alpha},
\]

the normalized finite remainder scales at best as
\(\eta_{block}^{3/2}\) times the cavity derivative constant.  Generic
irreducibility gives \(\eta_{block}=1\), so it supplies no CLT gain; many
small common blocks give the desired decay.

This links the finite-remainder gate directly to the q4-perpendicular
commutator sector.  The same noncommuting algebra which creates the missing
signed q211 information also prevents separate scalar eigensystems from
justifying a bivariate q3/q4 truncation.  A proof based only on per-unit
effective rank is therefore type-invalid for the variance channel.

Atomic gate:

- `AQ_CONNECTED_PAIR_COMMON_BLOCK_REMAINDER`: **HOLD** unless the actual
  pair/future-query core algebra has certified small common blocks (small
  \(\eta_{block}\)), or a noncommutative finite-remainder theorem replaces
  the independent-block argument.

## 2026-08-31 — noncommutative escape: H4 square lift is the exact AQ Stein-discrepancy Gram

There is a finite noncommutative formulation which does not require common
eigenblocks and explains why the H4 square lift is the right object.

Use the half-scaled AQ convention

\[
G_i=u_i^Tz+\frac12(z^TA_i z-\operatorname{tr}A_i).
\]

For the Gaussian Ornstein--Uhlenbeck generator \(L\), define the Malliavin
Stein matrix

\[
\Gamma_{ij}(z)=
\langle DG_j,-DL^{-1}G_i\rangle.
\]

Because the linear and quadratic chaoses have OU degrees one and two,

\[
DG_j=u_j+A_jz,
\qquad
-DL^{-1}G_i=u_i+\frac12A_i z.
\]

For every smooth test \(F\), Gaussian integration by parts gives the exact
identity

\[
\mathbb E[G_iF(G)]
=\sum_j\mathbb E[\Gamma_{ij}(z)\,\partial_jF(G)].
\]

Only \(\Gamma^s=(\Gamma+\Gamma^T)/2\) contracts against the symmetric Hessian
in the multivariate Gaussian Stein equation.  Writing

\[
C_{ij}=\mathbb E G_iG_j
=u_i^Tu_j+\frac12\operatorname{tr}(A_iA_j),
\]

one obtains exactly

\[
\boxed{
\Gamma^s_{ij}-C_{ij}
=\frac34(A_i u_j+A_j u_i)^Tz
+\frac12 I_2(\operatorname{sym}(A_iA_j)).}
\]

Consequently

\[
\boxed{
\mathbb E[(\Gamma^s_{ij}-C_{ij})(\Gamma^s_{kl}-C_{kl})]
=\frac9{16}\langle b_{ij},b_{kl}\rangle
+\frac12\langle S_{ij},S_{kl}\rangle,}
\]

where

\[
b_{ij}=A_i u_j+A_j u_i,
\qquad S_{ij}=\operatorname{sym}(A_iA_j).
\]

Thus the exact Stein-discrepancy feature is

\[
\Theta_{ij}=(2^{-1/2}S_{ij},\,3b_{ij}/4),
\]

which is the same two-component feature as the H4 square lift
\(\Psi_{ij}=(S_{ij},b_{ij})\), with known channel rescalings.  No commuting or
simultaneous-diagonalization premise is used.

Let \(Z_C\) be Gaussian with covariance \(C\), and let \(U_f\) solve the
multivariate Gaussian Stein equation for a reset/Bellman test \(f\).  Then

\[
\boxed{
\mathbb Ef(G)-\mathbb Ef(Z_C)
=\mathbb E\langle C-\Gamma^s,
\nabla^2U_f(G)\rangle.}
\]

For any carried approximation \(\widehat\Gamma\), subtracting its action
leaves the exact finite residual

\[
\mathbb E\langle\widehat\Gamma-\Gamma^s,
\nabla^2U_f(G)\rangle,
\]

which is bounded by the product of the task-weighted Stein-kernel residual
norm and the Bellman-Hessian norm.  This is a finite identity, not a
cumulant series; a Gaussian cavity or coarea bound is needed only to certify
the Hessian norm for ReLU.

Interpretation:

- H3 is the linear projection/action of the centered Stein matrix;
- the two-channel H4 square lift is its exact \(L^2\) Gram;
- higher matrix-polynomial words arise by iterating conditional Stein/Gamma
  operators, not by summing a divergent ReLU cumulant exponential.

This supplies a mathematically lawful noncommutative replacement for the
common-block Lindeberg route.  Its deployment gate is exactly the one already
identified by the H4 lane: can the future Bellman Hessian contract the
resident \(S_{ij}\) and \(b_{ij}\) providers without materializing the full
pair feature bank?  If yes, the q3/q4 correction should be formulated as a
Stein-kernel action; if no, the identity proves where the missing cost lives.

Atomic node:

- `AQ_H4_SQUARE_STEIN_KERNEL_IDENTITY`: **SOLVED mathematically**; provider
  ownership, Bellman-Hessian contraction, ReLU Hessian norm, and census are
  open implementation/value parents.

### Exact H3/H4 no-double-count Schur complement

The Stein representation also gives the exact boundary between an H3
linear action and genuinely new H4-square energy.  Define

\[
J_{ij,k}=\mathbb E[(\Gamma^s_{ij}-C_{ij})G_k].
\]

Directly from the two chaos channels,

\[
\boxed{
J_{ij,k}
=\frac34\langle A_i u_j+A_j u_i,u_k\rangle
+\frac12\operatorname{tr}(A_iA_jA_k).}
\]

If \(C=\operatorname{Cov}G\), the best linear/H3 prediction of the centered
Stein matrix is

\[
\Pi_{H3}(\Gamma^s_{ij}-C_{ij})
=J_{ij,:}C^+G.
\]

The complete symmetric cubic cumulant is recovered by cyclic symmetrization:

\[
\boxed{
\kappa_3(G_i,G_j,G_k)
=\frac23\left(J_{ij,k}+J_{ik,j}+J_{jk,i}\right).}
\]

The genuinely new square-lift covariance after H3 subtraction is therefore
the PSD Schur complement

\[
\boxed{
K^{\perp H3}_{ij,kl}
=K^\Gamma_{ij,kl}
-J_{ij,:}C^+J_{kl,:}^T,}
\]

where

\[
K^\Gamma_{ij,kl}
=\frac9{16}\langle b_{ij},b_{kl}\rangle
+\frac12\langle S_{ij},S_{kl}\rangle.
\]

This is the exact no-double-count formula.  Merely subtracting the symmetric
\(\kappa_3\) tensor need not equal the projection above: cyclic
symmetrization loses the individual provider coordinates \(J_{ij,k}\).
A resident H3 implementation owns the full linear Stein projection only if
it exposes \(J\) (or an algebraically equivalent unsymmetrized provider), not
just K21/repeated or fully symmetrized cubic scalars.

For any Bellman pair mask, its novel H4-square energy is the corresponding
contraction of \(K^{\perp H3}\).  This supplies an exact earliest value gate
before materializing any square-lift provider.

### Exact low-rank Bellman fusion of the Stein/H4 provider

For a deterministic symmetric pair mask \(\Lambda\), the entire centered
Stein discrepancy contracts to one AQ scalar:

\[
T_\Lambda(z)=
\sum_{ij}\Lambda_{ij}(\Gamma^s_{ij}(z)-C_{ij}).
\]

Let

\[
\Lambda=\sum_{a=1}^k\sigma_a v_av_a^T,
\qquad
M_a=\sum_i(v_a)_iA_i,
\qquad
\bar u_a=\sum_i(v_a)_iu_i.
\]

Symmetry of \(\Lambda\) gives the exact identities

\[
\sum_{ij}\Lambda_{ij}\operatorname{sym}(A_iA_j)
=\sum_a\sigma_a M_a^2,
\]

and

\[
\sum_{ij}\Lambda_{ij}(A_i u_j+A_j u_i)
=2\sum_a\sigma_a M_a\bar u_a.
\]

Therefore

\[
\boxed{
T_\Lambda(z)
=\sum_{a=1}^k\sigma_a
\left[\frac32(M_a\bar u_a)^Tz
+\frac12 I_2(M_a^2)\right].}
\]

This is the sought target-conditioned factorization.  A rank-k Bellman mask
does not require the \(n^2\)-pair square-lift bank: it requires k mixed cores,
k \(r\times r\) squares, and k matrix-vector products.  The acquisition of
the mixed cores is a multiplication of the k mask factors by the flattened
core bank.  Its leading arithmetic scales as

\[
O(knr^2+kr^3)
\]

per layer, versus materializing all pair providers.

The qualification is exact and important: the Hessian in the Stein identity
is generally a **state-dependent ReLU surface mask**, not automatically a
fixed deterministic \(\Lambda\).  The fusion applies literally if the
resident reverse/reset machinery reduces that surface action to k
deterministic pair factors (or a sum/integral of k-factor terms).  Otherwise
its functional Schmidt rank, not the rank of a same-site covariance costate,
sets k.

Atomic node:

- `AQ_STEIN_H4_LOW_RANK_BELLMAN_FUSION`: exact formula above; promote under a
  certified complete-future surface-mask rank k and a census for forming the
  \(M_a\).  Same-site low rank alone is insufficient.

Under the optimistic 46.09 r55 suffix census, the leading ten-layer cost of
this contraction is

\[
10k\left(\frac{r^2}{n^2}+\frac{r^3}{n^3}\right)
\simeq0.0304k
\quad(n=1024,r=55).
\]

The nominal 2.91-unit margin therefore permits at most about

\[
\boxed{k\le95}
\]

complete-future deterministic surface factors before ancillary
factorization/storage costs.  A generic full-rank \(k=1024\) mask costs about
31 units and is closed; a genuinely low functional rank in the tens is
compatible with the nominal margin.  This is a sharp fail-fast rank target
for the reverse audit.

### One-reset rank explosion and the only structured escape

At a centered Gaussian reset, the exact gate-pair kernel is

\[
H_{ij}=\mathbb P(Z_i>0,Z_j>0)
=\frac14+\frac1{2\pi}\arcsin\rho_{ij}.
\]

If the incoming endpoint mask is rank one, \(B=vv^T\), the reverse pair mask
after the reset contains

\[
B\circ H=\operatorname{diag}(v)H\operatorname{diag}(v).
\]

For nonzero v this has the same rank as H.  The positive kernel expansion is

\[
H=\frac14\mathbf1\mathbf1^T+
\frac1{2\pi}\left(ho+\frac16\rho^{\circ3}
+\frac3{40}\rho^{\circ5}+\cdots\right).
\]

If \(\rho=XX^T\) with feature rank r, the degree-d Hadamard power is the Gram
of symmetric tensor features \(x_i^{\otimes d}\), of generic rank

\[
\min\left(n,{r+d-1\choose d}\right).
\]

Already d=3 has feature dimension

\[
{57\choose3}=29260>1024
\quad(r=55,n=1024).
\]

Therefore a single exact centered ReLU reset generically turns a rank-one
Bellman mask into rank n.  The deterministic low-rank Stein fusion is not
generically preserved by depth; its exact k<=95 gate fails after one reset.

The useful structured split is

\[
H=\underbrace{\frac14\mathbf1\mathbf1^T+
\frac1{2\pi}\rho}_{\operatorname{rank}\le r+1}
+\underbrace{R_{\ge3}}_{\text{generic full rank}}.
\]

In addition, \(R_{\ge3}\) contains the non-negligible diagonal correction

\[
(R_{\ge3})_{ii}=\frac12-\frac14-\frac1{2\pi}
=\frac14-\frac1{2\pi}=0.090845\ldots,
\]

the same high-chaos derivative energy found in the Gaussian-rest theorem.
So a low-rank approximation must split off the diagonal explicitly; no
spectral argument based only on small off-diagonal correlations may discard
it.

This yields a three-part deploy test:

1. diagonal surface action, preferably fused with the already-paid per-unit
   eigensystem/reset;
2. rank-at-most-r+1 constant/linear correlation action;
3. Bellman-weighted norm of the off-diagonal \(R_{\ge3}\) tail.

Only if part 3 is below the endpoint error gate does the effective functional
rank stay in the tens.  Otherwise the exact low-rank fusion is cost-closed by
the generic rank-n result.

For the centered kernel, part 3 has an exact coherence bound.  If
\(|\rho_{ij}|\le\mu<1\) off diagonal and

\[
d_\mu=\sqrt{1-\mu^2}\left(1+\sqrt{1-\mu^2}\right),
\]

then

\[
|\arcsin x-x|
\le\frac{|x|^3}{3d_\mu}
\qquad(|x|\le\mu),
\]

because

\[
\frac1{\sqrt{1-t^2}}-1
=\frac{t^2}{\sqrt{1-t^2}(1+\sqrt{1-t^2})}.
\]

Hence, after removing the diagonal,

\[
\boxed{
\|R_{\ge3}^{off}\|_F
\le\frac1{6\pi d_\mu}
\left(\sum_{i\ne j}|\rho_{ij}|^6\right)^{1/2}
\le\frac{\mu^2}{6\pi d_\mu}\|\rho^{off}\|_F.}
\]

For a reverse mask B,

\[
\|B\circ R_{\ge3}^{off}\|_F
\le\frac{\mu^3}{6\pi d_\mu}\|B^{off}\|_F.
\]

This converts the qualitative low-rank escape into a measurable
coherence/Bellman-tail certificate.  It is exact for centered Gaussian gates;
biased gates require their corresponding bivariate threshold-kernel bound.

At r55 the rank-(r+1)=56 low part costs nominally about
\(0.0304\cdot56\simeq1.70\) units over ten layers.  A separately formed
per-unit diagonal square stream costs about
\(10r^3/n^2\simeq1.59\) units, so the two total about 3.29 and exceed the
2.91 margin by roughly 0.38.  The structured escape therefore fits the
46.09 census only if the diagonal action reuses the already-paid per-unit
eigensystems/reset (or another operation is removed); that fusion is a
literal cost premise, not a rounding detail.

### Bias makes the reverse rank explosion occur at degree two

For standardized Gaussian preactivations with thresholds \(a_i\), the exact
gate kernel has the integrated Mehler expansion

\[
\boxed{
H_{ij}
=\Phi(a_i)\Phi(a_j)
+\phi(a_i)\phi(a_j)
\sum_{m\ge1}\frac{\rho_{ij}^m}{m!}
H_{m-1}(a_i)H_{m-1}(a_j),}
\]

with probabilists' Hermite polynomials.  The first terms are

\[
H=Phi(a)\Phi(a)^T
+\operatorname{diag}(\phi(a))\rho\operatorname{diag}(\phi(a))
+\frac12\operatorname{diag}(a\phi(a))
  (\rho^{\circ2})
  \operatorname{diag}(a\phi(a))
+\cdots.
\]

At zero bias the degree-two term vanishes and degree three is the first
generic rank-n term, reproducing the arcsine result.  At nonzero generic
bias, degree two already has feature dimension

\[
{r+1\choose2}=1540>1024\qquad(r=55),
\]

so a single biased reset generically makes the reverse mask full rank one
Hermite degree earlier.

If \(q_\infty=\max_i|a_i\phi(a_i)|\), its off-diagonal size obeys

\[
\left\|H^{(2),off}\right\|_F
\le\frac{q_\infty^2}{2}
\left(\sum_{i\ne j}|\rho_{ij}|^4\right)^{1/2},
\]

and for a mask B,

\[
\|B\circ H^{(2),off}\|_F
\le\frac{q_\infty^2\mu^2}{2}\|B^{off}\|_F.
\]

Thus the rank-(r+1) structured escape additionally requires either centered
thresholds or a Bellman-weighted certificate that this bias-induced
degree-two block is negligible.  “Small off-diagonal correlations” alone is
not enough when biases are material.

### A resident dense D10 mask is not a free Stein-provider contraction

The D10 reverse may already own the dense pair mask \(\Lambda\), but the new
core contraction

\[
Q_\Lambda=\sum_{ij}\Lambda_{ij}\operatorname{sym}(A_iA_j)
\]

is not one of the ordinary \(n\times n\) covariance conjugations.  Without a
low-rank or polynomial factorization of \(\Lambda\), write the core bank by
latent middle index b.  Then \(Q_\Lambda\) is a sum of r products

\[
Q_\Lambda=\operatorname{sym}\sum_{b=1}^r L_b\Lambda R_b,
\]

where \(L_b\in\mathbb R^{r\times n}\) and
\(R_b\in\mathbb R^{n\times r}\).  The leading dense cost is

\[
O(n^2r^2)
\]

per layer, or

\[
\frac{r^2}{n}\simeq2.95
\]

full n-by-n GEMM equivalents per layer at r55.  Across ten layers this is
about 29.5 units, consistent with the k=n factor calculation and far outside
the remaining budget.

Therefore “D10 owns \(\Lambda\)” clears source ownership but not provider
action cost.  A literal fusion must expose low functional rank, a special
Hadamard/tensor factorization, or a paid contraction already having the
\(L_b\Lambda R_b\) type.  Ordinary D10 pair-mask propagation is not that
receipt.

## 2026-08-31 — exact matrix-power Stein hierarchy for a scalar AQ core

The Stein formulation gives a clean answer to the original matrix-polynomial
closure question.  For

\[
G=u^Tz+\frac12 I_2(A)
\]

define the centered iterated Gamma operators recursively by

\[
\Gamma_1^c
=\langle DG,-DL^{-1}G\rangle-\mathbb E\langle DG,-DL^{-1}G\rangle,
\]

and

\[
\Gamma_{k+1}^c
=\langle DG,-DL^{-1}\Gamma_k^c\rangle
-\mathbb E\langle DG,-DL^{-1}\Gamma_k^c\rangle.
\]

Then an induction on chaos degree gives the exact closed form

\[
\boxed{
\Gamma_k^c
=\frac{k+2}{2}(A^ku)^Tz
+\frac12 I_2(A^{k+1}),
\qquad k\ge1.}
\]

Indeed, if the linear and quadratic coefficients are
\((\alpha_k,\beta_k)\), one Gamma step sends

\[
(\alpha_k,\beta_k)\mapsto
(\alpha_k+\beta_k,\beta_k),
\]

with \((\alpha_1,\beta_1)=(3/2,1/2)\).  Its exact state norm is

\[
\boxed{
\|\Gamma_k^c\|_2^2
=\frac{(k+2)^2}{4}u^TA^{2k}u
+\frac12\operatorname{tr}A^{2k+2}.}
\]

Thus:

- the H4/Stein square lift is precisely the degree-two object \(A^2,Au\);
- each further finite Stein correction increases matrix degree by one, not
  by an uncontrolled tensor order;
- for a pure quadratic core the entire hierarchy is one matrix-power stream;
  no new rank appears before mixing different cores.

If \(a=\|A\|_{op}<1\) in the normalized units relevant to the Stein
recursion, then

\[
\|\Gamma_k^c\|_2
\le
a^k\left[
\frac{k+2}{2}\|u\|+
\frac1{\sqrt2}\|A\|_F
\right].
\]

This is a finite residual-state bound.  Multiplying it by the corresponding
finite derivative norm of the Stein/Bellman solution certifies truncation;
it does not sum the divergent ReLU cumulant series.

For a vector family \(G_i\), mixed Gamma iterations replace powers by
symmetrized ordered words

\[
A_{i_0}A_{i_1}\cdots A_{i_k}
\]

and corresponding word actions on the u-vectors.  The degree still grows
linearly, but the word-space dimension is generically the full matrix algebra
by Burnside.  The scalar one-stream theorem therefore survives across units
only under one of the previously derived conditions: a common small algebra,
a target-conditioned low functional rank, or a certified polynomial
relation/minimal polynomial.

For an orthogonal polynomial projection whose normalized omitted energy is
\(e_d\), the correlation gates are exactly

\[
e_d\le1-0.9253^2\simeq0.1438,
\qquad
e_d\le1-0.98346^2\simeq0.0328.
\]

Equivalently the residual amplitude must be below about 0.3792 or 0.1811.
The norm formula above supplies the scalar-core numerator; the Bellman
derivative and multiword amplification supply the remaining denominator.

### Exact resolvent generating function and polynomial truncation gate

The scalar Gamma hierarchy has a closed matrix-resolvent generating
function.  For \(|t|\|A\|_{op}<1\),

\[
\boxed{
\sum_{k\ge1}t^{k-1}\Gamma_k^c
=\frac12\left[
A(3I-2tA)(I-tA)^{-2}u
\right]^Tz
+\frac12I_2\left(A^2(I-tA)^{-1}\right).}
\]

Thus a rational/Padé closure is not heuristic in the one-core sector: it is
the exact generating object for all matrix-power Stein states.  Truncating
the quadratic resolvent after k=d has exact tail matrix

\[
t^d A^{d+2}(I-tA)^{-1},
\]

and hence

\[
\|R_d^{quad}\|_F
\le\frac{\|A\|_F\|A\|_{op}}{1-q}q^d,
\qquad q=|t|\|A\|_{op}<1.
\]

The linear tail has the same geometric factor times the explicit polynomial
prefactor inherited from \((I-tA)^{-2}\).  Consequently any proposed
resolvent weight t has an immediate degree gate:

\[
q^d\frac{C_{lin/quad}(d,q)}{1-q}
\le0.3792
\quad\text{or}\quad
\le0.1811
\]

for the two correlation targets, before Bellman amplification.  If q is
near one, low degree cannot pass; if q is bounded well below one, degree two
or three can be sufficient in the scalar sector.

Finally, Cayley--Hamilton makes the scalar hierarchy exactly finite at the
minimal-polynomial degree of A, at most r.  This algebraic exactness does not
solve the vector-family problem: different cores generate a noncommutative
word algebra whose dimension is generically \(r^2\), and a separate cost or
target-quotient certificate remains necessary.

## 2026-08-31 — exact diagonal Stein fusion with a paid spectral reset

The diagonal part of the full-rank gate mask has a literal no-extra-GEMM
fusion if the AQ reset already reconstructs one spectral matrix function per
unit.  Let

\[
A_i=Q_i\operatorname{diag}(\lambda_i)Q_i^T
\]

and suppose the paid reset forms

\[
F_i=Q_i\operatorname{diag}(f_i(\lambda_i))Q_i^T.
\]

For a diagonal reverse coefficient \(d_i\), the quadratic Stein provider is
\(d_iA_i^2/2\).  Instead of reconstructing it separately, form

\[
\boxed{
F_i+\frac{d_i}{2}A_i^2
=Q_i\operatorname{diag}
\left(f_i(\lambda_i)+\frac{d_i}{2}\lambda_i^2\right)Q_i^T.}
\]

This uses the same eigenvectors and the same matrix reconstruction as the
paid reset; only the diagonal eigenvalue expression changes.  The linear
provider \((3d_i/2)A_i u_i\) uses the already-needed rotated vector
\(Q_i^Tu_i\) and costs only \(O(r^2)\) per unit.

Therefore the earlier 1.59-unit separate diagonal-square charge can be zero
at leading GEMM order under three literal premises:

1. the reset already reconstructs \(F_i\) from the same eigensystem;
2. the Stein diagonal term may be accumulated into that same matrix output
   rather than requiring a separately addressable stream;
3. reverse coefficients \(d_i\) are available before reconstruction.

If these type/order premises hold, the rank-(r+1) low-correlation action
costs about 1.70 units and fits the nominal 2.91 margin, leaving about 1.21
for the certified off-diagonal higher-Hermite tail.  If the diagonal provider
must remain separately observable, the fusion does not apply and the prior
3.29-unit cost remains.

Atomic node:

- `AQ_STEIN_DIAGONAL_SPECTRAL_FUSION`: exact algebraic fusion; deployment
  status depends only on same-output semantics and reverse scheduling, not on
  approximation quality.

**Current-lineage type verdict.**  The full-AQ/D10 spelling does not satisfy
those premises.  Its paid \(F_i\) reconstruction is a forward local state,
whereas \(d_i\) is a future Bellman costate/read weight and the declared state
has no reverse accumulator at that point.  Hence the expression above is an
exact algebraic common-subexpression only for a future fused forward--reverse
schedule or a proved resident-costate theorem.  It is **type-invalid as a
free saving in the current schedule**, so the 1.59 charge remains in the
present census.

### Ownership resolution: diagonal is old-owned in the augmented literal contract

There is a different, lawful reason not to bill the diagonal as new.  In the
declared **augmented mathematical FULL-AQ plus Gaussian-rest** suffix, require
the per-unit AQ eig/quadrature to write exact

\[
\mathbb E Y_i,qquad \mathbb E Y_i^2,
\]

and require the rest/cross source to complete that same total diagonal
covariance.  Then post-reset variance is already a state-owned observable.
Complete-old projection annihilates every purely diagonal Bellman/Stein
provider, so the \(0.090845I\) gate term and the separate 1.59-unit reverse
family are duplicate, not novel.

This is not the invalid forward/reverse CSE above; it is an ownership
subtraction.  Under this exact diagonal contract, the centered rank-(r+1)
low-correlation off-diagonal action costs about 1.70 and fits the 2.91 margin,
leaving about 1.21 for hooks and the certified off-diagonal higher-Hermite
tail.

The qualification is binding:

- stored AQ coefficients alone do not own total diagonal variance;
- the on-disk Q24/D10 truncation through finite Hermite degree does not by
  itself prove exact rest completion;
- at a D14-to-AQ seam, any missing AQ/rest cross or univariate tail
  initialization reopens the diagonal as a synchronization defect.

Thus the promoted architecture must make exact univariate second moment and
rest-cross completion a literal interface field/gate.  With that addition,
the diagonal rank explosion is removed by duplicate subtraction rather than
by an impossible same-time reverse fusion.

Atomic node:

- `AQ_STEIN_DIAGONAL_OLD_OWNERSHIP`: **SOLVED conditionally** for the
  augmented exact-univariate contract; **OPEN/false** for the current
  finite-Q24 baseline or an unverified D14 seam.

## 2026-08-31 — exact reverse-tail bound after P2 plus D10 Hermite ownership

The integrated Mehler formula also shows what remains of the ReLU reverse
surface after complete-old subtraction.  Write the standardized gate
indicator coefficients as

\[
c_m(a)=\frac{\phi(a)H_{m-1}(a)}{\sqrt{m!}},
\qquad m\ge1.
\]

Then

\[
H_{ij}-\Phi(a_i)\Phi(a_j)
=\sum_{m\ge1}c_m(a_i)c_m(a_j)\rho_{ij}^m.
\]

The low AQ \(P_{\le2}\) covariance owns reset Hermite degrees n<=2, whose
Price derivatives are gate degrees m<=1.  If the literal D10/Q24 reverse
owns affine-Gaussian ReLU covariance degrees n=3,...,6, it additionally owns
gate degrees m=2,...,5.  Complete-old reverse residualization therefore
starts at

\[
\boxed{m=6.}
\]

The indicator Hermite energy identity is

\[
\sum_{m\ge1}c_m(a)^2
=\Phi(a)(1-\Phi(a))\le\frac14.
\]

Hence, if \(|\rho_{ij}|\le\mu<1\) off diagonal, Cauchy--Schwarz gives the
bias-uniform exact bound

\[
\boxed{
|H^{old\perp}_{ij}|
\le\frac{\mu^6}{4}
\qquad(i\ne j).}
\]

For a reverse mask B,

\[
\boxed{
\|B\circ H^{old\perp,off}\|_F
\le\frac{\mu^6}{4}\|B^{off}\|_F.}
\]

At centered thresholds, indicator parity kills even m.  Since m=1,3,5 are
old-owned, the first residual is m=7.  The arcsine expansion gives the
sharper entrywise bound

\[
|H^{old\perp}_{ij}|
\le\frac1{2\pi}\frac{\mu^7}{1-\mu^2}.
\]

This is an important separation of roles:

- the **resident dense D10 reverse** may consume a new AQ Stein/H4 source
  through its already-owned m<=5 action without a new reverse chain;
- only the strict higher-Hermite reverse tail needs a new functional-rank or
  error certificate;
- the theorem bounds that tail by sixth order in off-diagonal coherence,
  uniformly in bias, provided the D10 n=3,...,6 ownership is literal and
  complete-old residualization is performed.

It does not make acquisition of the new forward H4/Stein source free.  It
does, however, remove the generic full-rank gate kernel as an argument for
paying a second dense reverse if the existing D10 reverse can be reused at
the exact injection seam.

Atomic node:

- `AQ_STEIN_D10_REVERSE_REUSE_TAIL`: **SOLVED mathematically**; promotion
  requires literal n=3,...,6 D10 reverse ownership, same-seam source
  injection, measured/certified coherence \(\mu\), and Bellman normalization.

As a nonbinding prior, an existing w1024 receipt elsewhere in this campaign
reported maximum off-diagonal \(|\rho|\simeq0.515\).  Substitution gives
approximately \(\mu^6/4\simeq0.0047\), and the centered bound is about
0.0021.  These are encouraging scale checks, not a certificate for the new
AQ suffix: its own coherence and Bellman amplification must be registered.

**Current-lineage ownership correction.**  The deployed D11/Q24 estimator is
a forward loop, not a resident reverse.  `_q24_context` forms rho powers and
derivative feature tables; the local pair routines immediately sum them into
forward M11/M21/M31/M22 and then connected K planes.  No endpoint costate,
reverse recurrence, or retained m=2,...,5 Price-factor bank exists at the
source seam.  Therefore the theorem above does **not** make a second reverse
free in the current lineage.  It reduces the tail of either:

1. a new literal same-reset fused contraction for which the future query is
   already supplied, or
2. a newly billed reverse tape/recurrence.

Forward Q24 temporaries alone are not ownership of their adjoint action.
Accordingly `AQ_STEIN_D10_REVERSE_REUSE_TAIL` remains solved only as a tail
bound and is **OPEN/COST-UNOWNED** as a deployment claim.

## 2026-08-31 — low-Gram typing audit and the lawful forward in-place route

The on-disk D11 lineage has only one covariance state.  It computes

\[
C^{pre}=W^TCW
\]

and all Q24/reset logic consumes that total plane.  It does not retain a
separately typed low-chaos Gram \(G\).  Therefore no current-lineage claim may
borrow a free

\[
G=W_{feat}\operatorname{diag}(scale)W_{feat}^T
\]

for AQ/rest cross completion.

The augmented **paid-Gram** AQ suffix is different.  If its coefficient bank
has the chaos metric square root embedded, \(S_{AQ}=W_{feat}\sqrt{scale}\),
then the explicitly charged read

\[
S_{AQ}S_{AQ}^T
=W_{feat}\operatorname{diag}(scale)W_{feat}^T
\]

is exactly the required G.  It can be reused only if it remains separately
live until the cross planes are formed; adding it immediately to the total
D10 covariance and discarding it destroys the typing needed for reuse.

For independent centered low and rest components with covariance planes G
and S and diagonals \(g=\operatorname{diag}G\),
\(s=\operatorname{diag}S\), their exact raw fourth cross completions are

\[
\boxed{
K31^{cross}
=3\left(\operatorname{Diag}(g)S+
        \operatorname{Diag}(s)G\right),}
\]

and

\[
\boxed{
K22^{cross}
=g s^T+s g^T+4G\circ S.}
\]

Once both G and S are resident, these are elementwise/outer-product
operations, \(O(n^2)\), with no new dense GEMM.  The formulas are raw-moment
cross completion; they must be converted/subtracted consistently before
claiming connected-cumulant novelty.

This exposes the lawful architecture which avoids the absent reverse:

1. form and retain the paid AQ low Gram G;
2. form the exact local AQ/rest source and cross planes at the reset;
3. inject them into the **existing forward total covariance/moment state**;
4. let the already-paid D10 forward conjugations carry the corrected plane,
   with all gates recomputed/coadapted.

No endpoint reverse is needed for this forward in-place spelling.  It is
precisely the coadapted route demanded by the earlier carrier failures.  Its
remaining costs are source formation and the separately paid Gram, not a
second 20-unit covariance chain.

Cost/status boundary:

- the cheap 46.086 ten-reset suffix did not charge the low Gram and cannot
  use this reuse theorem;
- the strict six-reset late suffix at about 48.197 does charge it and may
  reuse G if liveness is literal, leaving about 0.803;
- the terminal-total optimization omits the last separated Gram, so the
  reuse is unavailable at that final reset unless no cross consumer remains.

Atomic node:

- `AQ_PAID_LOW_GRAM_FORWARD_CROSS_FUSION`: exact algebra/source/cost type;
  deployment is conditional on retained G liveness, exact rest independence
  semantics, same-reset residualization, and the late-suffix census.

## 2026-08-31 — exact chi-square check: q3-squared cancels two thirds of the q4 mean response

The q4-only local scale can be badly misleading when the quadratic spectrum
has nonzero cubic trace.  There is an exact solvable witness.  Let

\[
X_r=\frac{\chi_r^2-r}{\sqrt{2r}},
\qquad k=r/2.
\]

Gamma recurrence gives

\[
\boxed{
\mathbb E(X_r)_+
=\frac{k^{k-1/2}e^{-k}}{\Gamma(k)}.}
\]

The same recurrence gives the exact second positive-part moment

\[
\boxed{
\mathbb E(X_r)_+^2=Q(k+1,k),}
\]

where Q is the regularized upper incomplete gamma function.  Hence

\[
\operatorname{Var}(X_r)_+
=Q(k+1,k)-
\left(\frac{k^{k-1/2}e^{-k}}{\Gamma(k)}\right)^2,
\]

providing a closed finite-reset benchmark for both mean and variance.

Stirling's expansion therefore yields

\[
\frac{\mathbb E(X_r)_+}{\phi(0)}
=1-\frac1{6r}+\frac1{72r^2}+O(r^{-3}).
\]

By contrast, the isolated q4 Edgeworth term is

\[
-\frac{\gamma_4}{24}=-\frac1{2r},
\qquad \gamma_4=\frac{12}{r}.
\]

The missing cancellation is exactly the q3-squared H6 term.  At zero bias,

\[
\mathbb E f^{(4)}(Z)=-\phi(0),
\qquad
\mathbb E f^{(6)}(Z)=3\phi(0),
\quad f(x)=x_+.
\]

Hence the complete first order in inverse effective rank is

\[
\boxed{
\frac{\Delta\mu}{\phi(0)}
=\frac{\gamma_3^2-\gamma_4}{24}.}
\]

For equal positive eigenvalues,
\(\gamma_3^2=8/r\) and \(\gamma_4=12/r\), giving \(-1/(6r)\), exactly the
Stirling coefficient.  For a sign-symmetric/semicircle-like spectrum,
\(\gamma_3=0\), so the q4-only coefficient is correct at this order.

The corresponding zero-bias variance contribution at this order, after the
lower q3 response is treated separately, is

\[
\boxed{
\frac{\Delta V_{q4+q3^2}}{V_G}
=\frac{\gamma_4-\gamma_3^2}{12(\pi-1)}.}
\]

Thus the pure-q4 variance estimate is also reduced by a factor of three in
the equal-positive-spectrum witness, while it is unchanged for cubic-null
spectra.

This is a no-double-count gate, not a reason to delete q4.  The deployed Q24
table explicitly contains a k3-squared/72 channel.  Any AQ/Stein q4 value
claim must state whether that old channel is held fixed, jointly replaced,
or subtracted.  The analytic 1.7--1.8% per-layer semicircle scale is plausible
only when the actual signed spectra have small \(\gamma_3^2/\gamma_4\);
nominal rank alone is insufficient.

Atomic node:

- `AQ_Q4_Q3SQUARED_CANCELLATION`: solved exact witness and general local
  coefficient; actual-spectrum ratio and complete-old ownership determine
  value.

### Exact sign-balanced witness validates the semicircle/cubic-null q4 scale

The complementary equal-magnitude sign-balanced spectrum is also exactly
solvable.  Let the quadratic core have m positive and m negative equal
eigenvalues, total latent rank \(r=2m\), and standardize it as

\[
X_{\pm,r}
=\frac{\chi_m^2-\widetilde\chi_m^2}{2\sqrt m}.
\]

Writing \(p=m/2=r/4\), the beta--gamma factorization of two independent
Gamma(p,1) variables gives

\[
\boxed{
\mathbb E(X_{\pm,r})_+
=\frac{\Gamma(p+1/2)}{\sqrt\pi\,\Gamma(p)\sqrt m}.}
\]

The law is exactly symmetric, so

\[
\mathbb E(X_{\pm,r})_+^2=\frac12,
\qquad
\operatorname{Var}(X_{\pm,r})_+
=\frac12-\left[\mathbb E(X_{\pm,r})_+\right]^2.
\]

The gamma-ratio expansion is

\[
\boxed{
\frac{\mathbb E(X_{\pm,r})_+}{\phi(0)}
=1-\frac1{2r}+\frac1{8r^2}
+\frac5{16r^3}+O(r^{-4}).}
\]

Here \(\gamma_3=0\), \(\gamma_4=12/r\), so the leading coefficient is
exactly the q4 prediction \(-\gamma_4/24=-1/(2r)\).  Likewise

\[
\frac{\Delta V}{V_G}
=\frac1{(\pi-1)r}+O(r^{-2}),
\]

exactly validating the q4-only zero-bias variance coefficient for a
cubic-null diffuse spectrum.

Together the two solvable witnesses establish the real gate:

- equal-positive spectrum: q3-squared cancels two thirds of q4;
- equal sign-balanced spectrum: q3 vanishes and q4 scale is asymptotically
  exact with an explicit \(O(r^{-2})\) remainder.

The mechanism hypothesis therefore depends on spectral sign balance, not
merely effective rank.  The per-unit ratio
\(\gamma_3^2/\gamma_4\) is the correct first analytic screen.

### Universal pure-core cancellation interval

For an arbitrary pure quadratic spectrum, put

\[
s_j=\operatorname{tr}A^j.
\]

Then

\[
\gamma_3^2=8\frac{s_3^2}{s_2^3},
\qquad
\gamma_4=12\frac{s_4}{s_2^2},
\]

so

\[
\boxed{
\frac{\gamma_3^2}{\gamma_4}
=\frac23\frac{s_3^2}{s_2s_4}
\le\frac23,}
\]

by Cauchy--Schwarz.  Equality requires the nonzero eigenvalues to be equal
with one sign (up to the equality conditions); sign balance reduces the
ratio and exact cubic-null balance makes it zero.

Therefore q3-squared can cancel at most two thirds of the zero-bias q4
mean/variance response of any pure Gaussian quadratic core:

\[
\boxed{
\frac{\gamma_4}{72}
\le\frac{\gamma_4-\gamma_3^2}{24}
\le\frac{\gamma_4}{24}.}
\]

If \(r_4=s_2^2/s_4\), the magnitude of the mean correction lies between

\[
\frac1{6r_4}
\quad\text{and}\quad
\frac1{2r_4},
\]

and the relative variance correction lies between

\[
\boxed{
\frac1{3(\pi-1)r_4}
\quad\text{and}\quad
\frac1{(\pi-1)r_4}.}
\]

Thus the pure-core q4 mechanism cannot be sign-cancelled to zero by its own
q3-squared completion.  Its local scale has a rigorous factor-three interval;
complete-old ownership and endpoint coherence, rather than local sign, are
the remaining ways it can lose value.

### Affine-quadratic extension: at most three quarters cancellation

For the full scalar AQ variable

\[
G=u^Tz+z^TAz-\operatorname{tr}A,
\]

write

\[
V=\|u\|^2+2s_2,
\quad
\kappa_3=8s_3+6u^TAu,
\quad
\kappa_4=48\left(s_4+u^TA^2u\right).
\]

Cauchy gives

\[
|s_3|\le\sqrt{s_2s_4},
\qquad
|u^TAu|\le\|u\|\sqrt{u^TA^2u}.
\]

Putting \(X=\sqrt{2s_2}\), \(Y=\|u\|\),
\(P=\sqrt{s_4}\), and \(Q=\sqrt{u^TA^2u}\),

\[
|\kappa_3|
\le4\sqrt2\,XP+6YQ.
\]

The diagonal bilinear map has operator norm 6, so

\[
\boxed{
\kappa_3^2
\le36(X^2+Y^2)(P^2+Q^2)
=\frac34 V\kappa_4.}
\]

Equivalently, for standardized cumulants,

\[
\boxed{\gamma_3^2/\gamma_4\le3/4.}
\]

Thus even with an affine component, the q3-squared completion cancels at
most three quarters of the local zero-bias q4 coefficient.  The combined
mean term retains magnitude at least \(\gamma_4/96\), and the q4-plus-q3²
variance term retains at least one quarter of the q4-only value.  As before,
this concerns the local matched-Gaussian expansion; old ownership and
end-to-end Bellman coherence remain separate gates.

### Arbitrary-bias q4 plus q3-squared local response

The complete same-order formula is also explicit away from zero bias.  Let

\[
m_1(a)=\phi(a)+a\Phi(a),
\]

and use probabilists' Hermite polynomials.  Since

\[
\frac{d^4}{da^4}\mathbb E(Z+a)_+
=H_2(a)\phi(a),
\qquad
\frac{d^6}{da^6}\mathbb E(Z+a)_+
=H_4(a)\phi(a),
\]

the combined first local mean response is

\[
\boxed{
\frac{\Delta\mu_{4,3^2}}{s\phi(a)}
=\frac{\gamma_4}{24}H_2(a)
+\frac{\gamma_3^2}{72}H_4(a).}
\]

For the raw second moment,

\[
\frac{d^4}{da^4}\mathbb E(Z+a)_+^2=-2a\phi(a),
\qquad
\frac{d^6}{da^6}\mathbb E(Z+a)_+^2=-2H_3(a)\phi(a).
\]

Therefore the variance response is

\[
\boxed{
\Delta V_{4,3^2}
=-\frac{s^2\phi(a)}{12}
\left\{
\gamma_4\left[a+m_1(a)H_2(a)\right]
+\frac{\gamma_3^2}{3}
\left[H_3(a)+m_1(a)H_4(a)\right]
\right\}.}
\]

At a=0 these reduce to the cancellation formulas above.  Away from zero,
the H2 and H4 (or H3/H4 variance) factors can have different signs, so the
factor-three/factor-four magnitude interval is not a layerwise sign theorem.
The exact bias and ratio \(\gamma_3^2/\gamma_4\) determine coherence and must
be used in the Bellman telescope.

## 2026-08-31 — generic K31 source-read cost and the minimal-polynomial escape

Even when the paid low Gram G is resident, a generic connected AQ K31 source
is a different rectangular product.  In the pure half-scaled core,

\[
\boxed{
K31_{ij}^{core}
=\kappa_4(G_i,G_i,G_i,G_j)
=3\operatorname{tr}(A_i^3A_j).}
\]

Let \(P_1\) and \(P_3\) be the n-by-d matrices with rows
\(\operatorname{svec}(A_i)\) and \(\operatorname{svec}(A_i^3)\), where

\[
d=\frac{r(r+1)}2.
\]

Then

\[
G_{quad}=\frac12P_1P_1^T,
\qquad
K31^{core}=3P_3P_1^T.
\]

Thus the paid Gram product does not arithmetically form K31.  Once P3 is
available from the per-unit eigensystems, the generic exact read still costs
one n-by-d by d-by-n product, namely

\[
\frac{d}{n}
\]

full GEMM equivalents per layer.  At r55 this is

\[
\frac{1540}{1024}\simeq1.504
\]

per layer, or about 9.02 over the six-reset late suffix, far beyond its 0.803
strict slack.  Affine vector-leg terms add providers; they do not remove this
pure-core lower requirement.

There is an exact exceptional escape.  If every core obeys a low common
minimal-polynomial relation which expresses

\[
A_i^3=\alpha_iA_i+\beta_iI+\gamma_iA_i^2
\]

using already resident feature rows, then K31 reduces to row-scaled existing
Grams/cross-Grams and no fresh generic P3P1T product is needed.  Equal
magnitude sign-balanced cores satisfy \(A_i^3=\lambda_i^2A_i\) exactly.  A
generic diffuse core does not.

For a semicircle spectral law, the best scalar degree-one approximation to
\(\lambda^3\) has normalized correlation

\[
\frac{m_4}{\sqrt{m_2m_6}}
=\frac2{\sqrt5}\simeq0.8944,
\]

using \(m_2=1,m_4=2,m_6=5\).  Even before cross-unit/Bellman effects, this
misses both 0.9253 and 0.98346 provider-correlation gates.  Adding even powers
does not improve an odd target under a sign-symmetric law.

Therefore the strict six-reset paid-Gram schedule cannot claim a generic
full K31 source inside its remaining slack.  It reopens only if:

1. complete-old subtraction removes this K31 sector;
2. the actual cores pass a low-minimal-polynomial/Bellman-weighted residual
   gate; or
3. a same-node surface formula obtains the needed **residual action** without
   materializing P3P1T.

Atomic node:

- `AQ_K31_GENERIC_SOURCE_READ_COST`: generic exact source is cost-closed for
  late-six; exceptional minimal-polynomial or action-only fusion remains
  open.

## 2026-08-31 — new carrier proposal: co-encode K31 in the one paid polynomial Gram

The generic two-product cost suggests changing the state, not trying to read
both products.  Let

\[
(P_m)_{i,:}=\operatorname{svec}(A_i^m),
\qquad
P_\theta=P_1+\theta P_3,
\]

and use the single paid PSD carrier Gram

\[
G_\theta=\frac12P_\theta P_\theta^T
\]

in place of the literal quadratic covariance Gram.  Then

\[
\boxed{
G_\theta
=G_0+rac\theta2(P_3P_1^T+P_1P_3^T)
+\frac{\theta^2}{2}P_3P_3^T.}
\]

Since \(K31^{core}=3P_3P_1^T\),

\[
\boxed{
\left.\frac{dG_\theta}{d\theta}\right|_{\theta=0}
=\frac16(K31^{core}+K31^{core,T}).}
\]

Thus the symmetric K31 direction is exactly tangent to a one-Gram PSD
matrix-polynomial carrier.  No second rectangular product is required: the
finite \(\theta^2P_3P_3^T\) term is part of the new jointly refitted state,
not silently dropped.

More generally, for

\[
p(A)=\sum_{m=1}^q c_mA^m,
\]

one paid Gram contains the rank-one degree-pair combination

\[
G_p=\frac12\sum_{m,l}c_mc_lP_mP_l^T.
\]

It can therefore carry one target-selected combination of covariance,
commuting K31, and commuting K22 words at the original Gram cost.  It cannot
represent an arbitrary coefficient matrix over degrees: exact capacity is
rank one in the small polynomial-degree index.  Two paid polynomial bundles
give rank two, and so on.

This is a precise target-quotient gate.  Let H be the Bellman-weighted PSD
Gram of the required degree-pair coefficients after complete-old
subtraction.  The best one-bundle retained fraction is

\[
\boxed{
q_1^2=\lambda_{max}(H)/\operatorname{tr}H.}
\]

One bundle passes the two correlation screens iff

\[
q_1^2\ge0.8562
\quad\text{or}\quad
q_1^2\ge0.9672,
\]

respectively.  This is a small degree-space rank test, not an n-by-n SVD.

For a semicircle spectrum, the scalar functions lambda and lambda^3 have
moment Gram

\[
\begin{pmatrix}1&2\\2&5\end{pmatrix}.
\]

Although lambda alone correlates with lambda^3 by only \(2/\sqrt5\), a
target which needs one fixed combination \(a\lambda+b\lambda^3\) is captured
exactly by choosing \(p(\lambda)=a\lambda+b\lambda^3\).  The failure occurs
only when the future target needs two independent degree combinations.

### Lawfulness conditions

This is a new coadapted closure, not a free identity for the old AQ law.

1. Define the residual plane by total covariance minus the carrier Gram,
   \(S_\theta=C-G_\theta\), and require \(S_\theta\succeq0\).  A sufficient
   local gate is
   \(\|G_\theta-G_0\|_{op}<\lambda_{min}(S_0)\).
2. The literal AQ law core in the new state must be
   \(B_i=p(A_i)\), so its covariance is genuinely \(G_\theta\).  Reset and
   transport then use B, not a hidden A whose different Gram was never
   formed.  Spectral invertibility of p may help initialize B from a seam,
   but it does **not** make the laws \(I_2(A)\) and \(I_2(p(A))\) identical;
   the change is absorbed only by joint refit and the residual plane.
3. The alternating word \(\operatorname{tr}(A_iA_jA_iA_j)\), equivalently
   the q4-perpendicular commutator sector, is not a Gram of separate
   one-matrix polynomials.  It remains an explicit residual/source gate.
4. The directed antisymmetric part of K31 is not stored in a symmetric Gram.
   Promotion requires that complete future value uses the symmetric action
   or separately bounds the directed residual.
5. Joint refit is mandatory: hand-inserting theta while leaving the rest and
   gates frozen repeats the measured carrier-transfer failure.

This proposal is the exact mathematical way to exploit the co-adaptation law:
spend the already-required Gram on the target-relevant polynomial mixture,
and let the Gaussian-rest/total-covariance state absorb the orthogonal
covariance bookkeeping.  Its earliest close gate is the small degree-space
eigenvalue fraction q1 squared plus PSD rest feasibility.

If one instead keeps A as the physical AQ law while treating \(G_\theta\) as
an auxiliary source carrier, this ceases to be the full-law/Galerkin branch:
it is the connected-source branch and may not borrow full-AQ capacity or
independent-rest claims without separately forming the true A Gram.  This
semantic fork prevents the proposal from regaining two products by relabeling
one.

PSD feasibility has an exact scale gate.  For an unscaled candidate Gram
\(\widetilde G_p\) and total \(C\succ0\), the largest lawful carrier scale in
\(G_p=\alpha^2\widetilde G_p\) is

\[
\boxed{
\alpha_{max}^2
=\lambda_{max}
\left(C^{-1/2}\widetilde G_p C^{-1/2}\right)^{-1}.}
\]

Then \(S=C-G_p\succeq0\) iff \(0\le\alpha^2\le\alpha_{max}^2\).  Thus a PSD
decomposition always exists after sufficient shrinking, but quartic source
amplitude scales as \(\alpha^4\); an arbitrarily small feasible carrier is
not a value certificate.  The exact promotion gate is the endpoint gain at
or below \(\alpha_{max}\).

Atomic node:

- `AQ_POLYNOMIAL_ONE_GRAM_COADAPTED_CARRIER`: new open architecture; exact
  K31 tangent and rank-one degree-capacity theorem solved, target coefficient
  rank, PSD rest, directed/commutator tail, and refit value open.

## 2026-08-31 — free-semicircular closure: the generic K31 read collapses to one quadratic Gram

There is a stronger broad-spectrum exceptional law than a low minimal
polynomial.  Suppose the centered matrix cores form a free semicircular
family with normalized trace \(\tau=r^{-1}\operatorname{tr}\) and covariance

\[
c_{ij}=\tau(A_iA_j).
\]

The free Wick rule is

\[
\tau(A_iA_jA_k)=0,
\]

and

\[
\boxed{
\tau(A_iA_jA_kA_l)
=c_{ij}c_{kl}+c_{il}c_{jk}.}
\]

Unlike a commuting/star approximation, this retains a nonzero alternating
word and its commutator angle; it closes that word from the same covariance
Gram by the noncrossing Wick law.

Let the half-scaled quadratic covariance plane be

\[
Q_{ij}=\frac12\operatorname{tr}(A_iA_j)=\frac r2c_{ij},
\qquad q_i=Q_{ii}.
\]

Then the complete pure-core repeated quartic planes collapse exactly to

\[
\boxed{
K4_i=\frac{24}{r}q_i^2,}
\]

\[
\boxed{
K31_{ij}=\frac{24}{r}q_iQ_{ij},}
\]

and

\[
\boxed{
K22_{ij}=\frac8r
\left(q_iq_j+2Q_{ij}^2\right).}
\]

Thus all three are exactly the one-scalar pairing family with

\[
\boxed{\alpha=8/r}
\]

used by the deployed quartic-star compression.  The nominal P3P1T product
has disappeared: free Wick gives
\(\operatorname{tr}(A_i^3A_j)=2r c_{ii}c_{ij}\) directly from Q.

The q4-perpendicular sector is not set to zero.  For two cores,

\[
\tau([A_i,A_j]^2)
=2\left(c_{ij}^2-c_{ii}c_{jj}\right),
\]

so the signed commutator contribution is fixed by the same Gram.  This is a
noncommutative closure, not the previously dead commuting/collective one.

### Affine-isotropic extension

Let \(L_{ij}=u_i^Tu_j\), let \(C=L+Q\) be the total low AQ covariance, and
assume the vectors are isotropic relative to the free cores in the precise
second-word sense

\[
u_a^TA_iA_j u_b=L_{ab}c_{ij}.
\]

Then the full affine-quadratic repeated planes reduce to

\[
\boxed{
K4_i=\frac{24}{r}q_iC_{ii},}
\]

\[
\boxed{
K31_{ij}=\frac{12}{r}
\left(q_iC_{ij}+C_{ii}Q_{ij}\right),}
\]

and

\[
\boxed{
K22_{ij}=\frac4r
\left(q_iC_{jj}+C_{ii}q_j+4Q_{ij}C_{ij}\right).}
\]

These require only the separately retained quadratic Gram Q, the already
carried total covariance C, diagonals, outer products, and Hadamard products.
They cost \(O(n^2)\) per reset and include the vector-leg H4 terms under the
isotropy premise.

### Exact defect and fail-fast gate

For arbitrary cores define the free-Wick defect

\[
E_{ijkl}
=\tau(A_iA_jA_kA_l)
-c_{ij}c_{kl}-c_{il}c_{jk}.
\]

Then the pure-core source errors are exactly

\[
\Delta K31_{ij}=3rE_{iiij},
\]

and

\[
\Delta K22_{ij}=r\left(2E_{iijj}+E_{ijij}\right).
\]

The affine extension adds the explicit isotropy defects

\[
D_{ab;ij}=u_a^TA_iA_j u_b-L_{ab}c_{ij}.
\]

Therefore promotion does not require a vague “random matrix” assumption.  It
requires the complete-old/Bellman-weighted norm of these E and D combinations
to clear the raw endpoint gate.  The closure is exact on a free semicircular
family and has a fully typed residual off that family.

### Cost implication

The paid suffix Gram should be the quadratic Q Gram alone, not the
concatenated linear-plus-quadratic feature Gram; total C is already the D10
plane.  Its inner dimension is \(d=r(r+1)/2=1540\) at r55 rather than1595.
Over six resets this saves

\[
6\frac{1595-1540}{1024}\simeq0.322
\]

equivalents, moving the strict late-six subtotal from about48.197 to about
47.875 and leaving roughly1.125 for hooks/defect handling.  No K31/K22 dense
source product is added.

This is a concrete candidate for the broad variance mechanism: Wigner-like
signed spectra give the previously derived cubic-null q4 scale, while free
noncrossing Wick makes its cross-unit planes a cheap function of Q and C.
It also explains why a scalar pairing-alpha can carry real noncommutative
value when the generic matrix-word read is unaffordable.

Atomic node:

- `AQ_FREE_SEMICIRCULAR_QGRAM_CLOSURE`: exact law and O(n²) affine formulas
  solved; actual-core E/D defect, seam initialization, complete-old novelty,
  and end-to-end value remain open.

## 2026-08-31 — correction: split-Q retention is free CSE, not a smaller Gram bill

The preceding `47.875` subtotal is not justified by the literal strict-suffix
contract.  Its paid 1595-wide AQ feature bank is the concatenation of the 55
linear coefficients and the 1540 half-quadratic coefficients.  Splitting the
single product into

\[
L=UU^T,\qquad Q=PP^T/2,\qquad C=L+Q
\]

has the same leading inner dimension (55+1540=1595).  It lets the
implementation retain (Q) before the merge at only (O(n^2)) extra work,
but it does not remove the (UU^T) work needed to construct the total AQ
covariance.  Therefore the strict late-six census remains approximately

\[
\boxed{48.197,}
\]

with about .803 rather than1.125 headroom, unless a separate literal owner of
(L) is identified.  The free-Q formulas remain a cost-neutral source
replacement; only the claimed .322 saving is withdrawn.

Atomic correction:

- `AQ_SPLIT_Q_GRAM_COST_CORRECTION`: solved; retain-Q CSE is lawful, .322
  saving closed absent a separately paid linear Gram.

## 2026-08-31 — complete external q4 law and its exact finite-r residual

The collision formulas are faces of one master tensor.  Under the ordered
isotropy premise, for every four external unit indices,

\[
\boxed{
\kappa_4(i,j,k,l)
=\frac4r\sum_{(ab\mid cd)\in
\{ij\mid kl,ik\mid jl,il\mid jk\}}
\left(C_{ab}Q_{cd}+Q_{ab}C_{cd}\right).}
\]

For the pure matrix leg, set
(S_{ij}=\operatorname{sym}(A_iA_j)).  Directly,

\[
\frac1r\langle S_{ij},S_{kl}\rangle_F
=\frac12\left[
\tau(A_iA_jA_kA_l)+\tau(A_iA_jA_lA_k)
\right].
\]

Hence the exact pure-core residual around the free-Q law is

\[
\boxed{
R^A_{ijkl}
=\frac r2\sum_{(ab\mid cd)}
\left(E_{abcd}+E_{abdc}\right).}
\]

For the affine vector leg, define

\[
D_{ab;ij}=u_a^TA_iA_ju_b-L_{ab}c_{ij}.
\]

One pair-lift residual is exactly

\[
\begin{aligned}
R^u_{ij;kl}
={}&D_{jl;ik}+D_{jk;il}\\
&+D_{il;jk}+D_{ik;jl},
\end{aligned}
\]

and the complete vector residual is

\[
\boxed{R^u_{ijkl}=R^u_{ij;kl}+R^u_{ik;jl}+R^u_{il;jk}.}
\]

Thus the candidate has a literal Bellman test: contract (R^A+R^u), after
complete-old subtraction, against the endpoint q4 costate.  No generic
random-matrix assertion is needed in the promotion criterion.

## 2026-08-31 — the defect is a free cumulant, and dense mixing makes it CP3/CP4 Star

There is an exact extension beyond semicircular sources.  For any centered
noncommutative variables, the fourth free-cumulant identity is

\[
\tau(A_iA_jA_kA_l)
=c_{ij}c_{kl}+c_{il}c_{jk}
+\kappa_4^{\mathrm{free}}(A_i,A_j,A_k,A_l).
\]

Therefore the tensor called (E) above is not merely an error notation:

\[
\boxed{E_{ijkl}=\kappa_4^{\mathrm{free}}(A_i,A_j,A_k,A_l).}
\]

Let arbitrary centered post-reset cores (B_a) be freely independent, with

\[
v_a=\tau(B_a^2),\qquad
s_a=\kappa_3^{\mathrm{free}}(B_a),\qquad
e_a=\kappa_4^{\mathrm{free}}(B_a).
\]

For the next dense linear mix

\[
A_i=\sum_a W_{ia}B_a,
\]

multilinearity and vanishing mixed free cumulants give exactly

\[
c_{ij}=\sum_aW_{ia}W_{ja}v_a,
\]

\[
\boxed{
\tau(A_iA_jA_k)
=\sum_aW_{ia}W_{ja}W_{ka}s_a,}
\]

and

\[
\boxed{
E_{ijkl}
=\sum_aW_{ia}W_{ja}W_{ka}W_{la}e_a.}
\]

So the complete degree-three/four matrix law is:

1. the noncrossing two-Gram (Q,C) pairing tensor;
2. a coordinate CP3 Star source carrying the third free cumulant;
3. a coordinate CP4 Star source carrying the fourth free cumulant.

This is exact for freely independent reset cores even when each individual
core is far from semicircular.  Semicircularity is only the special case
(s_a=e_a=0).  It also states the ownership boundary sharply: if literal
CP3/CP4 sources are already in the complete-old interface, the free-Q law is
an exact cheap reconstruction and cannot be credited as new raw capacity.
If the strict replacement census dropped those sources, their omission is a
real residual rather than a relabeling of the pairing term.

Without exact freeness the general transport identity remains

\[
E^{\rm next}_{ijkl}
=\sum_{a,b,c,d}W_{ia}W_{jb}W_{kc}W_{ld}E^{\rm prev}_{abcd}.
\]

The diagonal-source reduction is therefore equivalent to saying that the
previous mixed free-cumulant tensor is diagonal in unit algebra.  This is a
much stronger and directly falsifiable certificate than spectral
semicircularity alone.

For row-normalized dense mixing with

\[
\max_{ia}|W_{ia}|\le\mu,
\]

the diagonal-source contributions obey deterministic incoherence bounds

\[
\sum_a|W_{ia}W_{ja}W_{ka}|\le\mu,
\qquad
\sum_a|W_{ia}W_{ja}W_{ka}W_{la}|\le\mu^2.
\]

Thus (mu\asymp n^{-1/2}) gives the conservative free-CLT scales
(O(n^{-1/2})) for CP3 and (O(n^{-1})) for CP4, before any random-sign
cancellation.  These bounds do not apply if reset creates broad mixed
cumulants across unit algebras; that is the decisive structural gate.

Atomic node:

- `AQ_FREE_CUMULANT_STAR_DECOMPOSITION`: exact theorem solved; literal
  CP3/CP4 ownership in the strict suffix and ordered-vector-module freeness
  remain open.

## 2026-08-31 — finite real-GOE bias and the useful normalization cancellation

The real-symmetric finite-r gauge has a small but nonzero crossing bias.  Let
jointly Gaussian GOE matrices be normalized so that

\[
\mathbb E\,\tau(A_iA_j)=c_{ij}.
\]

Entry Wick contraction around the trace cycle gives the exact expectation

\[
\boxed{
\mathbb E\,\tau(A_iA_jA_kA_l)
=c_{ij}c_{kl}+c_{il}c_{jk}
+\epsilon_r c_{ik}c_{jl},}
\]

where

\[
\boxed{
\epsilon_r=\frac{r+3}{(r+1)^2}.}
\]

At r55, (epsilon_r\simeq .01849).  On each repeated collision face K4,
K31, and K22 this adds the same relative multiplier

\[
1+\epsilon_r/2\simeq1.00925.
\]

The mean crossing bias can therefore be absorbed by replacing the nominal
pairing coefficient (8/r) by

\[
\boxed{\alpha_{GOE}=\frac8r(1+\epsilon_r/2).}
\]

The fluctuation, not the mean, is the harder finite-r gate.  Standard GOE
trace CLT coefficients give

\[
\operatorname{Var}(\operatorname{Tr}A^2)\to4,
\quad
\operatorname{Var}(\operatorname{Tr}A^4)\to72,
\quad
\operatorname{Cov}(\operatorname{Tr}A^2,\operatorname{Tr}A^4)\to16.
\]

Consequently a raw unnormalized core has asymptotic relative standard
deviation

\[
\frac{\sqrt{72}}{2r}=\frac{\sqrt{18}}r
\]

in its fourth moment, about7.7% at r55.  But after fixing its observed
quadratic norm, the scale-free kurtosis

\[
\frac{\tau(A^4)}{\tau(A^2)^2}
\]

has first-order fluctuation proportional to
(\delta_4-4\delta_2), whose variance is

\[
72+16\cdot4-8\cdot16=8.
\]

Its relative standard deviation around the semicircle value2 is therefore

\[
\boxed{\frac{\sqrt8}{2r}=\frac{\sqrt2}{r},}
\]

about2.57% at r55.  This explains why a per-core Q-normalized free closure
can be materially sharper than an unnormalized GOE heuristic, while still
falling short of a uniform .983 certificate without a Bellman-weighted
defect bound or fixed-spectrum design.

Atomic node:

- `AQ_FREE_Q_FINITE_R_GOЕ_GATE`: expectation correction solved; uniform
  finite-r E/D concentration and actual-cut defect remain open.

## 2026-08-31 — final-only target-value gate for the polynomial one-Gram
and free-semicircular Q-Gram candidates

The two preceding proposals remove the generic second K31 product, but that
is only a compute result.  This section places both candidates inside the
exact five-reset r62 Bellman/Pythagorean gate and separates three questions:

1. whether a PSD carrier scale is physically lawful;
2. whether the resulting fixed endpoint columns have enough target value;
3. whether their source and defect acquisition fits the remaining Cap49
   arithmetic.

### Exact r62 Q-Gram subtotal

At r62 put

\[
 e={62\cdot63\over2}=1953,
 \qquad d=62+e=2015.
\]

The terminal-factored five-reset subtotal (C_{62,5}=47.43295) contains
five width-(d) feature transports and four paid width-(d) Grams.  Both
new candidates retain the affine/feature transport but replace each paid
concatenated Gram by the quadratic or polynomial Gram of width (e).
Therefore their common optimistic subtotal is exactly

\[
\boxed{
 C_{Q,62,5}
 =28+{5d+4e\over1024}
  +5{62^3+32\,62^2\over1024^2}
 =C_{62,5}-{4\cdot62\over1024}
 \simeq47.19076.}
\tag{MP.1}
\]

The pre-hook headroom is about

\[
\boxed{s_{Q,62,5}\simeq1.80924.}\tag{MP.2}
\]

One additional generic width-(e) Gram costs

\[
 {e\over1024}={1953\over1024}\simeq1.90723>s_{Q,62,5}.
 \tag{MP.3}
\]

Hence either candidate is arithmetically viable only if the polynomial/Q
Gram replaces a paid Gram, the terminal Gram is genuinely omitted, and all
K31/K22 reads and defect handling use reductions, elementwise operations, a
resident product, or off-card certification.  A terminal separated Q Gram,
a second polynomial Gram, or one generic defect Gram closes the five-reset
route before hooks.  Local polynomial evaluation, writes, stabilization,
the r62 seam, and initialization still have to fit (MP.2).

### PSD scaling and the exact five-reset final cross

At reset (j), let (C_j\succ0) be the carried total covariance and let
(\widetilde Q_j\succeq0) be the unscaled polynomial or quadratic Gram.
Write

\[
 Q_j(t_j)=t_j\widetilde Q_j,
 \qquad S_j(t_j)=C_j-Q_j(t_j),
 \qquad t_j=\alpha_j^2.
\]

The exact elliptope interval is

\[
\boxed{
 0\le t_j\le t_j^{\max}:=
 \lambda_{\max}
 (C_j^{-1/2}\widetilde Q_jC_j^{-1/2})^{-1}.}
\tag{MP.4}
\]

Sufficiently small positive scale is therefore always feasible.  It does
not follow that the carrier has target value.

Let (r_0) be the complete-old residual at the final endpoint and let
(x_j(t_{1:j})) be the exact prediction change obtained from the (j)-th
hybrid replacement in the nonlinear five-reset telescope.  Then

\[
 r(t)=r_0-\sum_{j=1}^5x_j(t_{1:j}) \tag{MP.5}
\]

is an identity.  Put (R_0=\|r_0\|_H^2).  The one final-only Bellman cross
is

\[
\boxed{
 R_0\Chi(t)=
 -\sum_j\langle r_0,x_j\rangle_H
 +\sum_{i<j}\langle x_i,x_j\rangle_H.}
\tag{MP.6}
\]

Consequently

\[
\boxed{
 \|r(t)\|_H^2
 =R_0+\sum_j\|x_j\|_H^2+2R_0\Chi(t).}
\tag{MP.7}
\]

Equations (MP.6)--(MP.7), or the final norm itself, are necessary and
sufficient for one fixed final output.  Local PSD of (Q_j,S_j), and even
the positive covariance-Bellman action
(\langle\Lambda_j,E_j^{H4+}\rangle\ge0), determines neither the corpus
crosses in (MP.6) nor their signs.  PSD scaling is not an RSS or coherence
theorem.  With only the six diagonal endpoint energies, the full sharp
polygon/elliptope interval of (N.217)--(N.218) remains attainable.  Thus the
unconditional target-value lower bound from the local Gram laws alone is
zero.

There is a useful fixed-map specialization.  In the affine-isotropic
free-Wick formulas the mixed vector/core source is homogeneous in (t_j),
while the pure-core fourth source is homogeneous in (t_j^2).  If states,
future maps, gates, and ownership are frozen, collect their endpoint columns
in (X) and put

\[
 \varphi(t)=(t_1,t_1^2,\ldots,t_5,t_5^2)^T,
 \quad Z=(I-P_{old})X,
 \quad b=Z^*Wr_0,
 \quad H=Z^*WZ\succeq0. \tag{MP.8}
\]

Then the exact physical-scale objective is

\[
\boxed{
 \|r(t)\|_H^2
 =R_0-2b^T\varphi(t)+\varphi(t)^TH\varphi(t),
 \quad0\le t_j\le t_j^{\max}.}
\tag{MP.9}
\]

Hence the sharp physically tied gain is the finite box problem

\[
\boxed{
 \Gamma_{phys}=
 \max_{0\le t\le t^{\max}}
 \{2b^T\varphi(t)-\varphi(t)^TH\varphi(t)\}.}
\tag{MP.10}
\]

For a pure-core source, set (s_j=t_j^2); (MP.10) becomes a concave PSD
quadratic program on (0\le s_j\le(t_j^{\max})^2).  Formula (MP.9) is not
exact for a self-consistent nonlinear coadapted suffix unless the displayed
endpoint columns truly remain fixed; in that case use the hybrid identity
(MP.5)--(MP.7) instead.

### Joint final refit and the sharp raw target

Once an architecture, every scale, and every endpoint column are frozen, the
cheapest exact final-only test is ordinary complete-old Schur geometry.  With
(Z,b,H) now denoting all retained candidate columns,

\[
\boxed{
 m_{new}=R_0-b^TH^\dagger b.}
\tag{MP.11}
\]

The raw-(10^{-8}) target is equivalent to

\[
\boxed{
 b^TH^\dagger b>R_0-10^{-8}=1.9409\times10^{-8},}
\tag{MP.12}
\]

or a normalized gain above (0.659968).  The raw-(6\times10^{-9}) target
requires gain (2.3409\times10^{-8}), normalized about (0.795981).
For one column, (MP.12) is exactly the previously derived partial-correlation
gate (|\rho_{partial}|>0.8124).

Positive rescaling of any nonzero fixed column does not change its column
space, so it does not change (MP.11) under an unconstrained joint final
refit.  Thus an arbitrarily small but positive PSD-feasible scale does not
reduce ideal least-squares value; it worsens coefficient conditioning and
the relative importance of implementation/source error.  If physical reset
coefficients are tied as in (MP.10), or a coefficient bound is imposed, this
scale invariance no longer applies.

The no-cross Pythagorean route is narrower.  Let (V_1) be old columns plus
one frozen polynomial/Q-Gram dictionary and let (V_\infty) be a literal
fixed full-provider endpoint space.  Only if

\[
 V_{old}\subseteq V_1\subseteq V_\infty \tag{MP.13}
\]

and both designs are least-squares refit in the same corpus metric does

\[
\boxed{
 m_1=m_\infty+|(P_\infty-P_1)y\|_H^2.}
\tag{MP.14}
\]

Changing (p(A)), (t_j), the residual plane, reset maps, or the r62 state
changes the endpoint space and invalidates (MP.13).  A coadapted one-Gram
architecture is therefore not automatically a Pythagorean truncation of the
old AQ engine.  Freeze it first and use (MP.11), or exhibit literal nesting
and use (MP.14).

### Sharp target theorem for one polynomial bundle

Let (\mathscr L) be the frozen complete-old endpoint map from the symmetric
small degree-pair coefficient matrix to the corpus endpoint.  One paid
polynomial bundle has coefficient matrix (cc^T\), so its exact best
one-column value is

\[
\boxed{
 \Gamma_{1G}=
 \sup_{c:\ t^{\max}(c)>0}
 {\left|\langle r_0,(I-P_{old})\mathscr L(cc^T)\rangle_H\right|^2
  \over
  \|(I-P_{old})\mathscr L(cc^T)\|_H^2}.}
\tag{MP.15}
\]

It reaches raw (10^{-8}) by scalar cofire iff
(\Gamma_{1G}>1.9409\times10^{-8}).  The scale in (MP.4) cancels from
(MP.15) when positive and when a free terminal scalar is allowed.

The degree-space fraction
(q_1^2=\lambda_{max}(H_{degree})/\operatorname{tr}H_{degree}) remains an
exact rank-one *ingredient-fidelity* test.  It is not (MP.15): it contains no
target residual covector.  High (q_1^2) can be target-orthogonal, and a
lower-energy degree direction can carry all target value.  Therefore the
polynomial one-Gram candidate remains a cost-compatible conditional branch,
but its present PSD and degree-rank receipts alone give no raw improvement.

### Finite-dimensional obstruction to an exact broad free-semicircular law

The free-Wick proposal has an additional exact rank obstruction.  Let
(m=\operatorname{rank}(c)=\operatorname{rank}(Q)), whiten the covariance
on its support, and call the resulting symmetric (r\times r) cores
(B_1,\ldots,B_m).  Then

\[
 \tau(B_iB_j)=\delta_{ij},
 \qquad
 \tau(B_iB_jB_kB_l)
 =\delta_{ij}\delta_{kl}+\delta_{il}\delta_{jk}. \tag{MP.16}
\]

Consider all ordered products (X_{ij}=B_iB_j\in M_r).  Their Frobenius
Gram is

\[
\begin{aligned}
 \langle X_{ij},X_{kl}\rangle_{HS}
 &=\tau(B_jB_iB_kB_l)\\
 &=\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}.
\end{aligned}\tag{MP.17}
\]

In pair coordinates this is

\[
 \boxed{\mathcal G_X=I_{m^2}+\operatorname{vec}(I_m)
 \operatorname{vec}(I_m)^T\succ0.} \tag{MP.18}
\]

Thus all (m^2) products are linearly independent in the (r^2)-dimensional
space (M_r), proving

\[
 \boxed{m\le r.} \tag{MP.19}
\]

At r62, an exact degree-four free-semicircular family therefore satisfies

\[
\boxed{\operatorname{rank}Q\le62.}\tag{MP.20}
\]

It cannot be an exact full-rank 1024-unit Q-Gram law.  The surviving exact
case is a low-rank structured Q branch, not the generic (e=1953)-direction
AQ section advertised by the broad-spectrum heuristic.

The approximate obstruction is quantitative.  In whitened coordinates let
(\mathcal E_{(ij),(kl)}=E_{jikl}) be the free-Wick defect in the product
Gram.  Since every actual product Gram has rank at most (r^2), when (m>r)

\[
\boxed{
 \|\mathcal E\|_{op}\ge1,
 \qquad
 \|\mathcal E\|_F\ge\sqrt{m^2-r^2}.}
\tag{MP.21}
\]

The first bound follows by testing a unit vector in the nullspace of the
actual product Gram against (MP.18); the second is the Eckart--Young tail of
(MP.18), whose (m^2-1) small eigenvalues equal one.  Therefore a
high-rank Q bank cannot have uniformly small free-Wick defect.  It can survive
only if the complete-old future Bellman map annihilates the large whitened
defect directions, which is a target-conditioned theorem, not a random-matrix
prior.

For a proposed free-Wick endpoint column (\widehat z), let the actual
complete-old column be (z) and certify
(\|z-\widehat z\|_H\le\epsilon_{ED}) from the E/D defects.  Writing
(\widehat b=\langle r_0,\widehat z\rangle_H) and
(\widehat h=\|\widehat z\|_H^2), its true optimal one-column gain obeys

\[
\boxed{
 {(|\widehat b|-\sqrt{R_0}\epsilon_{ED})_+^2
  \over(\sqrt{\widehat h}+\epsilon_{ED})^2}
 \le \Gamma_{true}
 \le
 {(|\widehat b|+\sqrt{R_0}\epsilon_{ED})^2
  \over(\sqrt{\widehat h}-\epsilon_{ED})_+^2}.}
\tag{MP.22}
\]

The upper bound is informative only when its denominator is positive.  The
lower bound must exceed (MP.12) to promote; if the upper bound is below
(MP.12), close.  PSD source sign does not replace (\widehat b).

### Disposition

1. **Polynomial one-Gram:** conditional cost GO under (MP.1)--(MP.3), but
   value remains open.  Use (MP.11)/(MP.15), not (q_1^2), as the raw gate.
2. **Free-semicircular Q-Gram, generic high-rank form:** closed as an exact
   finite-r law by (MP.19)--(MP.21).
3. **Free-semicircular low-rank or projected form:** conditional survivor
   only with rank (Q\le62), or with a certified Bellman-dark E/D defect;
   it must pass (MP.22).
4. **Both candidates:** local PSD feasibility supplies no endpoint
   improvement and no sign for (\Chi).  A fixed jointly refit design uses
   the Schur/Pythagorean gate; a self-consistent coadapted design needs the
   exact hybrid final norm or aggregate cross.
5. **Cap:** any extra generic Gram, terminal separated Q Gram, or new dense
   five-step reverse closes the optimistic r62 shell.  Only fused/reduction
   source reads and off-card final validation fit the stated route.

Atomic graph-ready nodes:

- `AQ_R62_QGRAM_EXACT_COST` -- **solved conditional cost**, (MP.1)--(MP.3);
- `AQ_PSD_SCALE_FINAL_CHI` -- **solved exact theorem**, (MP.4)--(MP.10),
  application requires the actual hybrid endpoint blocks;
- `AQ_PSD_SCALE_SCHUR_INVARIANCE` -- **solved for fixed nonzero columns**,
  (MP.11)--(MP.12);
- `AQ_ONE_GRAM_TARGET_VALUE_GATE` -- **open application**, exact gate
  (MP.15); degree-space (q_1^2) is not target value;
- `AQ_ONE_GRAM_PYTHAGOREAN_SEMANTICS` -- **conditional**, literal fixed
  nesting (MP.13) is mandatory;
- `AQ_FREE_WICK_FINITE_RANK_NOGO` -- **solved no-go** for
  (\operatorname{rank}Q>r), (MP.16)--(MP.21);
- `AQ_FREE_SEMICIRCULAR_LOW_RANK_SURVIVOR` -- **conditional only** at
  rank Q at most 62 or under a target-dark-defect proof;
- `AQ_FREE_SEMICIRCULAR_DEFECT_VALUE_GATE` -- **open application**, robust
  promote/close interval (MP.22);
- `AQ_QGRAM_EXTRA_PRODUCT_CAP` -- **closed**: one extra width-1953 Gram or
  a terminal separated Q Gram exceeds the r62 five-reset headroom.

## 2026-08-31 — collision-only K31 obstruction: the missing word is exactly the cubic Chebyshev residual

The full-product rank no-go is stronger than some deployed interfaces need,
so it is useful to isolate the repeated K31 face itself.  Let

\[
V=\operatorname{span}\{B_1,\ldots,B_m\}\subset\operatorname{Sym}_0(r)
\]

and whiten it in normalized trace:

\[
\tau(B_iB_j)=\delta_{ij}.
\]

For one core define

\[
\mu_{k,i}=\tau(B_i^k)
\]

and remove its scalar and linear cubic pieces,

\[
R_i=B_i^3-\mu_{4,i}B_i-\mu_{3,i}I.
\]

Then (R_i\perp\operatorname{span}\{I,B_i\}), and the exact vector of K31
errors across the other external units is

\[
\boxed{
\left(\tau(B_i^3B_j)-\mu_{4,i}\delta_{ij}\right)_{j=1}^m
=\left(\tau(R_iB_j)\right)_{j=1}^m.}
\tag{MP.23}
\]

Consequently

\[
\boxed{
\sum_j\left|\tau(R_iB_j)\right|^2
=\|P_VR_i\|_\tau^2,}
\tag{MP.24}
\]

where

\[
\boxed{
\|R_i\|_\tau^2
=\mu_{6,i}-\mu_{4,i}^2-\mu_{3,i}^2.}
\tag{MP.25}
\]

For a symmetric semicircle core, ((\mu_2,\mu_3,\mu_4,\mu_6)=(1,0,2,5)),
so

\[
R_i=B_i^3-2B_i,
\qquad \|R_i\|_\tau^2=1.
\]

This is the third Chebyshev/Jordan word.  Free-Wick predicts its expectation
against an independent core is zero, but it does not make its projection
onto a finite common unit span vanish.

If a generic m-dimensional subspace containing (B_i) is used inside the
(d_0=\dim\operatorname{Sym}_0(r)=r(r+1)/2-1) dimensional traceless matrix
space, rotational transversality gives

\[
\mathbb E\|P_VR_i\|_\tau^2
=f\|R_i\|_\tau^2,
\qquad
f=\frac{m-1}{d_0-1}.
\tag{MP.26}
\]

When the diagonal coefficient is matched, the expected ingredient fidelity
of the scalar-Q K31 prediction is therefore

\[
\boxed{
q_{31}^2
=\frac{\mu_4^2}
{\mu_4^2+f(\mu_6-\mu_4^2-\mu_3^2)}.}
\tag{MP.27}
\]

At r55, m=1024, (d_0=1539), and a semicircle gives

\[
f=1023/1538\simeq.665,
\qquad
q_{31}^2\simeq.8575,
\qquad
q_{31}\simeq.9260.
\]

Thus the free-Q closure is mathematically on the knife edge of the .925
ingredient screen and fails the .983 screen decisively even before affine
isotropy and reset leakage.  At r49 it gives about.909; at r62 about.940.
Increasing r helps because a fixed 1024-unit span occupies a smaller fraction
of the matrix coefficient space, not because the cubic word disappears.

The sharp sixth-moment gates follow directly from (MP.27).  For a centered
core with (mu_4=2), strict .983 fidelity at r55 requires approximately

\[
\boxed{\mu_6\le4.204,}
\tag{MP.28}
\]

whereas loose .925 fidelity allows approximately

\[
\boxed{\mu_6\le5.01.}
\tag{MP.29}
\]

The semicircle value5 only barely clears the loose gate.  Any claimed strict
free-Q K31 closure must therefore exhibit a near-minimal sixth moment or a
Bellman-dark projection of (R_i); pairwise fourth moments alone cannot
certify it.

Atomic node:

- `AQ_FREE_Q_K31_CHEBYSHEV_RESIDUAL`: solved exact projection theorem;
  generic r55 estimate passes .925 only marginally and closes .983 absent a
  low-sixth-moment or target-dark certificate.

## 2026-08-31 — deterministic collision averages in the complete symmetric frame

The preceding projection picture has exact completeness receipts.  For a
normalized-trace orthonormal basis
(\{B_a\}_{a=1}^{d_0}) of
(\operatorname{Sym}_0(r)), the symmetric-matrix completeness identities are

\[
\boxed{
\sum_aB_aXB_a
=\frac r2(X^T+\operatorname{tr}X\,I)-X,}
\tag{MP.30}
\]

and

\[
\boxed{\sum_aB_a^2=d_0I.}
\tag{MP.31}
\]

For a traceless symmetric unit (B_i), these imply

\[
\sum_a\tau(B_i^2B_a^2)=d_0,
\tag{MP.32}
\]

and

\[
\sum_a\tau(B_iB_aB_iB_a)=\frac r2-1.
\tag{MP.33}
\]

If in addition (	au(B_i^4)=2), the mean off-diagonal square-word defect
from its free value1 is exactly

\[
-\frac1{d_0-1},
\]

while the mean alternating word is

\[
\boxed{
\frac{r/2-3}{d_0-1}.}
\tag{MP.34}
\]

At r55 the latter is about.0159.  The broad full-product defect is therefore
forced, but it can be spread as (O(1/r)) over collision entries.  This is
why the full-tensor rank no-go does not by itself close a collision-projected
Bellman interface; (MP.23)--(MP.29) are the sharper K31 gate for that case.

## 2026-08-31 — the unique three-atom escape and its Jacobi/H3 generalization

Equation (MP.25) shows exactly how to remove the cubic word.  Equality

\[
\mu_6-\mu_4^2-\mu_3^2=0
\]

holds iff the cubic polynomial is already in
(\operatorname{span}\{I,B,B^2\}) on the spectrum.  In the centered symmetric
case with (mu_2=1,mu_3=0,mu_4=2), the unique nondegenerate law is

\[
\boxed{
\nu_3=\frac14\delta_{-\sqrt2}
+\frac12\delta_0
+\frac14\delta_{+\sqrt2},}
\tag{MP.35}
\]

and the core obeys

\[
\boxed{B^3=2B.}
\tag{MP.36}
\]

This law matches the variance and fourth moment of the semicircle but has
(mu_6=4), so it passes both K31 fidelity gates exactly.  Randomly conjugated
copies remain noncommuting and retain q4-perpendicular commutator energy.
For two independent Haar conjugates of any fixed traceless spectrum with
(	au(B^2)=1), orthogonal invariance gives

\[
\boxed{
\mathbb E\tau(A^2B^2)=1,\qquad
\mathbb E\tau(ABAB)=
\eta_r:=\frac{r-2}{(r-1)(r+2)}.}
\tag{MP.37}
\]

Thus the remaining mean K22 correction is only the finite-r multiplier
(1+\eta_r/2), about1.00861 at r55.

More generally, the three-node Gaussian/Jacobi quadrature of a scalar core
spectral measure matches moments zero through five and has a cubic
annihilator

\[
p_3(x)=x^3-a x^2-bx-c.
\]

For the quantized matrix,

\[
\boxed{B^3=aB^2+bB+cI.}
\tag{MP.38}
\]

Therefore the directed repeated quartic source reduces exactly to

\[
\boxed{
\operatorname{tr}(B_i^3B_j)
=a_i\operatorname{tr}(B_i^2B_j)
+b_i\operatorname{tr}(B_iB_j)
+c_i\operatorname{tr}(B_j).}
\tag{MP.39}
\]

At a centered seam the last term vanishes.  Hence one common three-atom
Jacobi closure supplies K31 from the already-required signed-H3 plane plus
the quadratic Gram, with no P3P1 product.  It preserves each core's moments
through degree five; its error first appears in degree six and in cross-core
ordered products.

There are two exact deployment caveats.

1. Rank-r matrices realize quadrature weights only in multiples of (1/r).
   The symmetric weights in (MP.35) are exact only when r is divisible by4;
   r56 is the nearest exact rank, whereas r55 incurs an (O(1/r)) weight or
   fourth-moment error.
2. Dense linear mixing does not preserve a three-point minimal polynomial.
   The closure must be a literal spectral retraction before the reset whose
   law it approximates, or it must carry the cubic residual word.  Applying
   it only after a reset and then mixing does not prove (MP.38) at the next
   preactivation.  A spectral retraction changes the AQ law and therefore
   requires joint refit plus a Bellman-weighted retraction-error certificate.

The local operation itself is cheap once the AQ eigendecomposition is paid:
mapping eigenvalues to three nodes and reconstructing uses the existing
spectral path, and the carried coefficient width is unchanged.  The
potential value is structural, not an extra column: it trades uncontrolled
cubic-word leakage for a controlled degree-six reset error.

Atomic node:

- `AQ_THREE_ATOM_JACOBI_CLOSURE`: exact K31/H3/Q reduction solved; r56
  multiplicity, preactivation spectral retraction, cross K22 defect, and
  end-to-end value are open.

### Precision correction and r56 census

Equality in (MP.25) means

\[
B^3\in\operatorname{span}\{I,B\},
\]

not merely (\operatorname{span}\{I,B,B^2\}).  The generic three-node
relation (MP.38) transfers its (aB^2) term to the signed-H3 plane; it does
not make the Q-only residual vanish.  Moreover, generic per-core Gaussian
quadratures have different coefficients ((a_i,b_i,c_i)).  A common cheap
law requires common nodes/recurrence, or separately typed per-row scalar
coefficients in the H3/Q action.  The physical K31 plane also restores the
overall factor3 and, when normalized trace is used, the factor r.

The symmetric law (MP.35) is special:

\[
a=c=0,\qquad b=2,
\]

so its pure matrix K31 is Q-owned without any H3 plane.  It deliberately
sets the pure cubic trace to zero; affine H3 and finite-algebra cross defects
remain separate.  The Jacobi idea does not solve K22, all-distinct q4, or the
affine vector (b_{ij}) leg.

The exact rank56 paid-Gram arithmetic is nevertheless notable.  With

\[
d_{56}=56+56\cdot57/2=1652,
\]

the strict six-reset, K32 local-quadrature census is

\[
\boxed{
28+12\frac{1652}{1024}
+6\frac{56^3}{1024^2}
+192\frac{56^2}{1024^2}
=48.93848,}
\tag{MP.40}
\]

leaving only.06152.  The terminal-total version is

\[
\boxed{
28+11\frac{1652}{1024}
+6\frac{56^3}{1024^2}
+192\frac{56^2}{1024^2}
=47.32520,}
\tag{MP.41}
\]

leaving1.67480.  Here K32 is the number of local AQ quadrature nodes; it is
not a resident K21/H3 plane.  Thus rank56 is lawful only if the three-point
spectral retraction is absorbed inside the paid eig/quadrature path and no
new cross-unit H3 or CP array is silently appended.

Atomic node:

- `AQ_R56_THREE_ATOM_COST`: strict arithmetic solved at48.93848 and
  terminal-total at47.32520; typed spectral retraction and value remain open.

## 2026-08-31 — exact reset invariant module for an s-atom spectral core

The low-minimal-polynomial proposal has an additional exact reset theorem.
Let

\[
A=\sum_{\alpha=1}^s\lambda_\alpha P_\alpha,
\qquad
u_\alpha=P_\alpha u,
\]

and let the AQ ReLU projection map be

\[
(m,u,A)\longmapsto(m^+,u^+,A^+).
\]

Orthogonal equivariance under every rotation which fixes (A) and (u)
forces

\[
\boxed{
u^+\in\operatorname{span}\{u_1,\ldots,u_s\},}
\tag{MP.42}
\]

and

\[
\boxed{
A^+\in
\operatorname{span}
\left(
\{P_\alpha\}_{\alpha=1}^s
\cup
\{\operatorname{sym}(u_\alpha u_\beta^T):
1\le\alpha\le\beta\le s\}
\right).}
\tag{MP.43}
\]

Proof: the stabilizer contains

\[
\prod_\alpha O(P_\alpha\mathbb R^r\cap u_\alpha^\perp).
\]

Its fixed vector space is exactly the span in (MP.42).  Its fixed symmetric
matrix space consists of a scalar on each orthogonal complement plus the
outer products between the distinguished directions (u_\alpha), which is
(MP.43).  The AQ projection is equivariant because both the Gaussian law and
ReLU are orthogonally invariant.

The reset matrix module dimension is at most

\[
\boxed{s+s(s+1)/2.}
\tag{MP.44}

For the three-atom closure this is only9.  In the pure-core case (u=0),
the reset is a spectral function of A and remains in
(\operatorname{span}\{P_-,P_0,P_+\}); hence it still has at most three
eigenvalues and another cubic minimal polynomial.  The affine rank-one
terms enlarge the fixed module but do not create a generic r-by-r reset
operator at that same site.

This theorem concerns the unitwise nonlinear reset.  A dense mix of matrices
with different projector frames still leaves the module and is why a
spectral retraction is required at every new preactivation.

## 2026-08-31 — exact three-group characteristic function and compressed quadrature

At rank r divisible by4, put (p=r/4) and take the symmetric three-atom core

\[
A=\sqrt2(P_+-P_-),
\qquad
\operatorname{rank}P_+=\operatorname{rank}P_-=p,
\qquad
\operatorname{rank}P_0=2p.
\]

Decompose (u=u_++u_-+u_0) by these projectors.  The centered preactivation

\[
h=m+u^Tz+\frac12(z^TAz-\operatorname{tr}A)
\]

is a Gaussian plus a difference of two noncentral chi-squares.  Its exact
characteristic function is

\[
\boxed{
\begin{aligned}
\varphi_h(t)={}&e^{itm}
e^{-t^2\|u_0\|^2/2}\\
&\times
(1-i\sqrt2t)^{-p/2}
\exp\left[-\frac{t^2\|u_+\|^2}
{2(1-i\sqrt2t)}\right]\\
&\times
(1+i\sqrt2t)^{-p/2}
\exp\left[-\frac{t^2\|u_-\|^2}
{2(1+i\sqrt2t)}\right].
\end{aligned}}
\tag{MP.45}
\]

The centering phases cancel because the positive and negative multiplicities
are equal.  Thus every scalar surface integral needed by the exact ReLU AQ
projection depends only on

\[
(m,\|u_+\|^2,\|u_-\|^2,\|u_0\|^2,p),
\]

not on 56 unrelated eigenvalues.  Differentiating the same scalar transform
supplies all coefficients of the invariant basis (MP.42)--(MP.43).

In the centered pure-core normalization

\[
X=\frac{\chi_p^2-\widetilde\chi_p^2}{2\sqrt p},
\]

the characteristic function reduces to

\[
\boxed{
\varphi_X(t)=left(1+\frac{t^2}{p}\right)^{-p/2}
=\left(1+\frac{4t^2}{r}\right)^{-r/8}.}
\tag{MP.46}
\]

It is symmetric with unit variance, so

\[
\mathbb EX_+^2=\frac12,
\]

and its exact ReLU mean is

\[
\boxed{
\mathbb EX_+
=\frac{\Gamma(p/2+1/2)}
{\sqrt\pi\,\Gamma(p/2)\sqrt p}.}
\tag{MP.47}

Its standardized cumulants begin

\[
\gamma_4=\frac{24}{r},
\qquad
\gamma_6=\frac{1920}{r^2}.
\tag{MP.48}

A semicircle-spectrum core has the same (gamma_4) but
(gamma_6=2400/r^2).  Therefore the spectral retraction preserves the full
q4 reset response and first differs in the sixth-cumulant term.  At zero
bias, the leading mean difference is

\[
\frac{\Delta\mathbb EX_+}{\phi(0)}
\simeq\frac{\Delta\gamma_6}{240}
=\frac{2}{r^2},
\tag{MP.49}

only about (6.38\times10^{-4}) at r56.  Equation (MP.49) is an asymptotic
jet, not a uniform tail certificate; the exact characteristic function
(MP.45) gives the lawful one-dimensional Fourier/secant remainder gate.

Here the displayed magnitude is (2/r^2): with
(\Delta=\hbox{three-atom}-\hbox{semicircle}), the sign is negative because
(1920-2400=-480), so

\[
\frac{\Delta\mathbb EX_+}{\phi(0)}\simeq-\frac2{r^2}.
\]

### Billing implication

The generic local quadrature term in the paid-Gram census is

\[
sK\frac{r^2}{n^2}.
\]

For an s-atom invariant module, K quadrature samples determine only
(O(s^2)) scalar coefficients.  The matrix reconstruction then costs one
spectral/projector reconstruction plus (s(s+1)/2) rank-one outer-product
writes, rather than K generic r-by-r surface matrices.  If the existing
(sr^3/n^2) eig/reconstruction term literally owns the projector sum, the
remaining r-squared coefficient work replaces K by at most

\[
K_{eff}=s(s+1)/2.

For the three-atom r56 closure, (K_{eff}\le6) instead of32.  Under this
typed CSE, (MP.40) drops by

\[
6(32-6)\frac{56^2}{1024^2}\simeq.4666
\]

to approximately

\[
\boxed{48.472,}
\tag{MP.50}

with about.528 headroom; the terminal-total spelling drops to about46.859.
This saving is conditional on the ledger's r-cubed term already including
the projector reconstruction.  If it charges only eigensolution and not the
output reconstruction, add that write explicitly.

Atomic nodes:

- `AQ_SATOM_RESET_INVARIANT_MODULE`: exact theorem solved;
- `AQ_R56_THREE_GROUP_RESET_TRANSFORM`: exact characteristic/reset scalar
  reduction solved;
- `AQ_R56_K32_MODULE_CSE`: conditional cost reduction to about48.472;
  requires literal eig/reconstruction ownership and unchanged output state.

### Scope correction: the invariant module is a lower-reset theorem

The (K_{eff}\le6) reduction applies only to the single-unit AQ reset
surface/reconstruction.  It does not form mixed q211/H4 directions from
other cores (A_j,A_k).  Such directions generally do not share the gate
core's projectors (P_\alpha); rotating an aggregate direction into every
unit's three blocks is a new action/product unless a target factor or common
algebra is proved.  Therefore (MP.50) may replace the local K32 ledger term
only.  It is not a free full-H4 Hessian or affine-vector-leg implementation.

Likewise, the scalar

\[
\eta_r=\frac{r-2}{(r-1)(r+2)}
\]

in (MP.37) is an ensemble-mean amplitude.  Precisely,

\[
\mathbb E\tau(A^2B^2)=1,
\qquad
\mathbb E\tau(ABAB)=\eta_r,
\]

so the mean normalized pure K22 collision is (2+\eta_r), versus free value
2, and

\[
\mathbb E\tau([A,B]^2)=2(\eta_r-1).
\]

It controls the pair-collision mean, including its commutator face.  It is
not a squared-energy error, an all-distinct q4 theorem, or a uniform bound
for one realized 1024-core family.  Haar fluctuation is still (O(1/r));
promotion needs its actual Bellman contraction or a sharp deterministic
orbit/design certificate.

## 2026-08-31 — exact Frobenius price of retracting a semicircle to the three-atom orbit

The spectral retraction has a sharp local obstruction.  Let X have the
unit-variance semicircle density

\[
\rho(x)=\frac1{2\pi}\sqrt{4-x^2},\qquad |x|\le2.
\]

Among all monotone assignments with target law (MP.35), the Frobenius-optimal
coupling is the quantile map

\[
T(X)=\sqrt2\,\operatorname{sign}(X)\mathbf1_{\{|X|>a\}},
\]

where the upper-tail threshold has mass one quarter.  Write
(a=2\cos\theta).  The threshold is determined exactly by

\[
\boxed{
\theta-\frac12\sin2\theta=\frac\pi4.}
\tag{MP.51}

The maximal normalized Frobenius correlation with any exact symmetric
three-atom retraction is then

\[
\boxed{
\rho_{3\leftarrow sc}
=\mathbb E[XT(X)]
=\frac{\sqrt2}{3\pi}(4-a^2)^{3/2}
=\frac{8\sqrt2}{3\pi}\sin^3\theta
\simeq0.9189.}
\tag{MP.52}

The proof is the rearrangement inequality: for a fixed target atomic law,
the increasing quantile coupling maximizes (\mathbb E[XT]).  The integral
is

\[
2\sqrt2\int_a^2x\rho(x)\,dx
=\frac{\sqrt2}{3\pi}(4-a^2)^{3/2}.
\]

Thus an exact preactivation retraction of a generic semicircle core has

\[
\rho^2\simeq.8444,
\qquad
\sqrt{1-\rho^2}\simeq.3945.
\]

It misses even the .925 ingredient gate.  The excellent local reset
moment match in (MP.48)--(MP.49) does not erase this coefficient-space loss.
The exact three-atom branch therefore cannot be promoted as a generic
full-core replacement.  It survives only if

1. the complete-old Bellman map is dark on the retraction residual;
2. the learned preactivation law is already much closer to the three-atom
   orbit than a semicircle; or
3. a soft/more-atom closure trades some cubic residual for higher retraction
   fidelity.

This is the fail-fast gate which must be checked before exploiting the r56
census.

Atomic node:

- `AQ_THREE_ATOM_SEMICIRCLE_RETRACTION_NOGO`: solved generic no-go at
  correlation about.9189; target-dark or already-quantized exceptions remain.

## 2026-08-31 — exact semicircular log-determinant generator and the zero-bias homogeneity no-go

The free-Q law admits an exact probabilistic interpretation which determines
its raw covariance value.  For a unit-variance semicircle spectral measure,
define

\[
J(v)=
\log\frac{1+\sqrt{1+4v}}2
+\frac1{1+\sqrt{1+4v}}-\frac12.
\tag{MP.53}
\]

It follows from the semicircle Cauchy transform that

\[
\int\log(1-x\lambda)\,d\mu_{sc,c}(\lambda)
=\log\frac{1+\sqrt{1-4cx^2}}2
+\frac1{1+\sqrt{1-4cx^2}}-\frac12.
\tag{MP.54}
\]

Indeed, differentiating the left side gives

\[
-\frac1x\left[
\frac{2}{1+\sqrt{1-4cx^2}}-1
\right],
\]

and the value at x=0 fixes the constant.

Let (X_i) be standardized half-quadratic forms whose matrix cores form an
abstract joint semicircular family with covariance (C).  Every linear
combination has a semicircle spectrum, so its exact characteristic function
is

\[
\boxed{
\mathbb E e^{it^TX}
=F_r(t^TCt),
\qquad
F_r(s)=\exp\left[-\frac r2J\left(\frac{2s}{r}\right)\right].}
\tag{MP.55}

The symmetric spectral pairing representation makes (F_r) completely
monotone as a function of s.  Therefore there is a nonnegative scalar S with

\[
\mathbb ES=1,
\qquad
F_r(s)=\mathbb E e^{-Ss/2},
\]

and

\[
\boxed{X\ \overset{d}{=}\ \sqrt{S}\,G,\qquad G\sim N(0,C).}
\tag{MP.56}

This immediately proves an exact no-go.  ReLU is positively homogeneous, so
at zero bias

\[
\boxed{
\mathbb E[(X_i)_+(X_j)_+]
=\mathbb ES\,\mathbb E[(G_i)_+(G_j)_+]
=\mathbb E[(G_i)_+(G_j)_+].}
\tag{MP.57}

Thus the complete uncentered ReLU second-moment action of the pure
semicircular/free-pairing core is identical to the Gaussian arc-cosine
kernel.  It cannot supply a broad raw covariance correction, despite its
nonzero q4 and commutator words.

The mean is

\[
\mathbb E(X_i)_+
=\mathbb E\sqrt S\,\frac{\sqrt{C_{ii}}}{\sqrt{2\pi}},
\]

so centering changes the covariance only by

\[
\boxed{
\frac{1-(\mathbb E\sqrt S)^2}{2\pi}
\sqrt{\operatorname{diag}C}\,
\sqrt{\operatorname{diag}C}^{,T}.}
\tag{MP.58}

This is rank one in the common-radial model.  Since
(\operatorname{Var}S=8/r),

\[
\mathbb E\sqrt S=1-1/r+O(r^{-2}),
\]

and the counterterm scale is only (2/r+O(r^{-2})).  This gives an exact
mechanistic explanation for the measured near-zero value of a collective
rank-one q4 carrier: the noncommutative free pairing sector is real, but
ReLU homogeneity makes its uncentered covariance Gaussian-equivalent.

Atomic node:

- `AQ_FREE_RADIAL_RELU_ZERO_BIAS_NOGO`: exact; pure free-Q pairing gives no
  uncentered covariance novelty and only a rank-one centered counterterm.

## 2026-08-31 — the surviving cheap broad term: affine plus radial quadratic mixture

There is one nontrivial extension.  If the affine Gaussian component is
independent of the quadratic radial component, write

\[
h=g+\sqrt{S}\,q,
\qquad
g\sim N(0,L),\quad q\sim N(0,Q),
\]

with (g,q,S) independent.  Conditional on S=s, h is Gaussian with

\[
C(s)=L+sQ.
\]

Therefore the exact uncentered pair reset is the one-dimensional mixture

\[
\boxed{
M^+_{ij}=\mathbb E_S\,mathcal K_{m ReLU}
\left(L_{ij}+SQ_{ij},
L_{ii}+SQ_{ii},
L_{jj}+SQ_{jj}ight),}
\tag{MP.59}


where

\[
\mathcal K_{m ReLU}(c,v,w)
=\frac{\sqrt{vw}}{2\pi}
\left[
\sqrt{1-\rho^2}+(\pi-\arccos\rho)\rho
\right],
\quad \rho=c/\sqrt{vw}.
\tag{MP.60}

The means are

\[
\boxed{
\mu_i^+=\frac1{\sqrt{2\pi}}
\mathbb E_S\sqrt{L_{ii}+SQ_{ii}},}
\tag{MP.61}


and the centered plane is (M^+-\mu^+(\mu^+)^T).  Equations
(MP.59)--(MP.61) cost (O(Kn^2)) for K scalar mixture nodes, with no dense
matrix product.  Although S is scalar, the correction is generally full
rank because every entry has a different affine/quadratic ratio.

For small radial variance, its leading all-orders replacement correction is

\[
\boxed{
M^+-\mathcal K_{m ReLU}(L+Q)
=\frac{\operatorname{Var}S}{2}
D_Q^2\mathcal K_{m ReLU}(L+Q)+O(\kappa_3(S)).}
\tag{MP.62}

This is the only broad component left by the pure-free no-go: it is the
interaction between affine covariance and radial quadratic covariance, not
the pure q4 pairing tensor.

The independence premise is strong.  In the ordinary AQ representation the
linear vector and matrix core use the same Gaussian z, and the exact
characteristic exponent contains

\[
u(t)^T(I-iA(t))^{-1}u(t),

not a product of a Gaussian factor and (MP.55).  The mixture theorem becomes
literal if the architecture uses disjoint latent blocks for the affine and
quadratic coefficients, or if an operator-valued isotropy theorem makes the
resolvent coupling negligible.  Using disjoint blocks does not increase the
coefficient-bank width—one may keep r linear features plus
(r(r+1)/2) quadratic features—but it changes the law and must be jointly
refit.

Finally, (MP.59) is the all-orders resummation of the same affine/H4 square
sector already represented approximately by formal AQ/Q24.  It is a
replacement candidate, not additive value.  Promotion requires showing that
the old truncation error's endpoint partial correlation clears the raw gate;
the pure-free part is closed by (MP.57).

Atomic node:

- `AQ_AFFINE_RADIAL_MIXTURE_KERNEL`: exact under disjoint-block
  independence; O(Kn²) broad reset replacement, old-span novelty and raw
  value open.

## 2026-08-31 — closed Hessian formula for the affine-radial broad feature

The leading term in (MP.62) has a compact elementwise form.  At S=1 write

\[
v_i=L_{ii}+Q_{ii},\quad
v_j=L_{jj}+Q_{jj},\quad
c=L_{ij}+Q_{ij},\quad
\rho=\frac c{\sqrt{v_iv_j}},
\]

and define the diagonal quadratic fractions

\[
a_i=\frac{Q_{ii}}{v_i},qquad
a_j=\frac{Q_{jj}}{v_j}.
\]

The derivative of the conditional correlation is

\[
\boxed{
\dot\rho_Q
=\frac{Q_{ij}}{\sqrt{v_iv_j}}
-\frac\rho2(a_i+a_j).}
\tag{MP.63}

For

\[
k(\rho)=\frac1{2\pi}
\left[\sqrt{1-\rho^2}+(\pi-\arccos\rho)\rho\right],
\]

one has

\[
k'(\rho)=\frac{\pi-\arccos\rho}{2\pi},qquad
k''(\rho)=\frac1{2\pi\sqrt{1-\rho^2}}.
\]

Direct differentiation of
(\mathcal K(s)=\sqrt{v_i(s)v_j(s)}k(\rho(s))) gives the exact cancellation

\[
\boxed{
\frac{\mathcal K''(1)}{\sqrt{v_iv_j}}
=-\frac{(a_i-a_j)^2}{8\pi}\sqrt{1-\rho^2}
+\frac{\dot\rho_Q^{\,2}}{2\pi\sqrt{1-\rho^2}}.}
\tag{MP.64}

The two terms have a clear interpretation.

1. Unequal diagonal quadratic fractions lower the mixed second moment.
2. A mismatch between the Q-correlation direction and the total-correlation
   direction raises it, with enhanced sensitivity near a correlation edge.

If (Q=\alpha(L+Q)) on the pair, then (a_i=a_j=\alpha) and
(\dot\rho_Q=0); the radial correction vanishes exactly, recovering the
pure-homogeneity theorem.  Therefore the broad feature is driven entirely by
heterogeneity of the affine/quadratic split.

The radial mixing variable in (MP.56) has exact first cumulants

\[
\boxed{
\mathbb ES=1,\qquad
\operatorname{Var}S=\frac8r,qquad
\kappa_3(S)=\frac{160}{r^2}.}
\tag{MP.65}

The first follows from covariance normalization.  The latter two follow by
matching

\[
\gamma_4(X)=3\operatorname{Var}S=24/r
\]

and

\[
\mathbb EX^6=15\mathbb ES^3
=15+360/r+2400/r^2.
\]

Thus the explicit leading broad source is

\[
\boxed{
\Delta M_{ij}^{(4)}
=\frac4r\sqrt{v_iv_j}
\left[
-\frac{(a_i-a_j)^2}{8\pi}\sqrt{1-\rho^2}
+\frac{\dot\rho_Q^{\,2}}{2\pi\sqrt{1-\rho^2}}
\right].}
\tag{MP.66}

The next radial correction is

\[
\frac{80}{3r^2}D_Q^3\mathcal K,

before fourth and higher radial cumulants.  Equation (MP.66) is exactly the
free/H4-square affine interaction written without a matrix word.  It is an
(O(n^2)) fail-fast feature.  Since formal AQ/Q24 may already own its leading
jet, the potential raw novelty of (MP.59) is specifically the all-orders
tail after subtracting (MP.66) and every other retained derivative channel.

Atomic node:

- `AQ_AFFINE_RADIAL_HESSIAN_FEATURE`: exact cheap formula solved; complete-old
  subtraction and all-orders tail value open.

### Exact fourth radial moment and a nonasymptotic tail gate

The next even quadratic-form cumulant is

\[
\kappa_8(X)=\frac{564480}{r^3}.
\]

Using

\[
\mathbb EX^8
=105+210\kappa_4+28\kappa_6+35\kappa_4^2+\kappa_8
=105\mathbb ES^4
\]

gives

\[
\boxed{
\mathbb ES^4
=1+\frac{48}{r}+\frac{832}{r^2}+\frac{5376}{r^3},}
\tag{MP.67}

and hence

\[
\boxed{
\mathbb E(S-1)^4
=\frac{192}{r^2}+\frac{5376}{r^3}.}
\tag{MP.68}

Therefore

\[
\boxed{
\mathbb E|S-1|^3
\le
\sqrt{\frac8r
\left(\frac{192}{r^2}+\frac{5376}{r^3}\right)}.}
\tag{MP.69}

For any pair whose covariance ray
(C(s)=L+sQ) stays inside a region where the third directional derivative is
bounded by (M_{3,ij}), Taylor's theorem yields the rigorous pointwise tail
bound

\[
\boxed{
\left|
\mathbb E\mathcal K(C(S))
-\mathcal K(C(1))
-\frac4rD_Q^2\mathcal K(C(1))
\right|
\le
\frac{M_{3,ij}}6
\sqrt{\frac8r
\left(\frac{192}{r^2}+\frac{5376}{r^3}\right)}.}
\tag{MP.70}

At r56 the scalar absolute-third-moment bound in (MP.69) is about.1146, so
the multiplier before (M_3) is about.0191.  A complete-old Q24 subtraction
can replace the left side by the first omitted order and sharpen this
further.  Correlations approaching +/-1 make the arc-cosine derivatives
singular; those pairs need a direct secant evaluation rather than (MP.70).

Atomic node:

- `AQ_AFFINE_RADIAL_TAIL_BOUND`: exact radial moments and Taylor gate solved;
  deployed pairwise derivative margin/old-order subtraction open.

## 2026-08-31 — bias reopens the pure radial sector

The no-go (MP.57) is exactly a zero-threshold statement.  Let

\[
\mathcal G_{ij}(m_i,m_j;C)
=\mathbb E[(m_i+G_i)_+(m_j+G_j)_+],
\qquad G\sim N(0,C).
\]

For a pure common-radial quadratic bank,

\[
\boxed{
M^{rad}_{ij}
=\mathbb E_S\left[
S\,\mathcal G_{ij}
\left(\frac{m_i}{\sqrt S},\frac{m_j}{\sqrt S};C\right)
\right].}
\tag{MP.71}
\]

At zero mean, Gaussian ReLU homogeneity makes the bracket independent of S
after the leading factor and recovers (MP.57).  Nonzero, unit-dependent
standardized means break that cancellation and make (MP.71) generally full
rank.

Let

\[
\mathcal D_m=m_i\partial_{m_i}+m_j\partial_{m_j}
\]

be the two-mean Euler operator.  For

\[
F(s)=s\,\mathcal G_{ij}(m_i/\sqrt s,m_j/\sqrt s;C),
\]

direct differentiation gives

\[
\boxed{
F''(1)=\frac14(\mathcal D_m^2-2\mathcal D_m)
\mathcal G_{ij}(m_i,m_j;C).}
\tag{MP.72}
\]

Since \(\operatorname{Var}S/2=4/r\), the leading bias-driven radial source is

\[
\boxed{
\Delta M^{bias,(4)}_{ij}
=\frac1r(\mathcal D_m^2-2\mathcal D_m)
\mathcal G_{ij}(m_i,m_j;C).}
\tag{MP.73}
\]

This is another \(O(n^2)\) H4 feature, now driven by heterogeneous normalized
means rather than an affine/quadratic covariance split.

The one-unit checks are explicit.  With
\(Y=m+\sqrt{sv}Z\), \(a=m/\sqrt v\),

\[
\frac{d^2}{ds^2}\mathbb E(Y)_+\bigg|_{s=1}
=\frac{\sqrt v\,\phi(a)}4(a^2-1),
\tag{MP.74}
\]

and

\[
\frac{d^2}{ds^2}\mathbb E(Y)_+^2\bigg|_{s=1}
=-\frac{v\,a\phi(a)}2.
\tag{MP.75}
\]

Multiplication by \(4/r\) reproduces the standard q4/Edgeworth bias
responses.  Consequently:

1. pure free-Q is closed for broad uncentered covariance only at zero bias;
2. nonzero means reopen a cheap broad radial feature;
3. its leading term is still formal H4/Q24-owned, so additive novelty begins
   only after complete-old subtraction or in an exact all-orders replacement.

Atomic node:

- AQ_BIASED_RADIAL_RELU_KERNEL: exact mixture and leading Euler feature
  solved; actual standardized-mean heterogeneity and Q24-tail value open.

## 2026-08-31 — exact quadratic resolvent on the three-node Jacobi closure

Let the common cubic annihilator be

\[
p(x)=x^3-a x^2-bx-c,
\qquad p(A)=0.
\]

The divided-difference identity

\[
p(z)I-p(A)
=(zI-A)\left[
(z^2-az-b)I+(z-a)A+A^2
\right]
\]

gives the exact local resolvent

\[
\boxed{
(zI-A)^{-1}
=\frac{(z^2-az-b)I+(z-a)A+A^2}
{z^3-az^2-bz-c}.}
\tag{MP.76}
\]

For the symmetric three-atom law \(A^3=2A\),

\[
\boxed{
(zI-A)^{-1}
=\frac1zI+\frac1{z^2-2}A
+\frac1{z(z^2-2)}A^2.}
\tag{MP.77}
\]

Thus every local Fourier/resolvent quantity used by the generalized
chi-square ReLU reset closes exactly on the three channels
\(\{I,A,A^2\}\).  No Padé truncation or degree tail exists at a unitwise
reset.  More generally, an s-node minimal polynomial makes the resolvent an
exact degree-\(s-1\) polynomial by the same divided-difference formula.

This exactness does not reduce alternating products of different cores:
relations such as \(A_i^3=2A_i\) simplify repeated K31 words, but
\(A_iA_jA_iA_j\) still carries relative projector geometry.  Hence local
reset rank closes at three while cross-core q4perp remains governed by the
Haar/free defect gates.

Atomic node:

- AQ_THREE_NODE_EXACT_RESOLVENT: solved; local reset degree2 exact, cross-core
  alternating geometry remains open.

## 2026-08-31 — literal Q24 ownership: the radial r-to-the-minus-two tail is genuinely omitted

The deployed density reset has the exact symbolic order

\[
1+L_3+L_4+\frac12L_3^2.
\tag{MP.78}
\]

Its derivative channels contain:

1. linear k3/k21 terms through \(L_3\);
2. linear k4/k31/k22 terms through \(L_4\);
3. only the self-convolution \(L_3^2/2\).

It contains no \(L_4^2\), no \(L_3L_4\), no sixth-cumulant \(L_6\), and no
q6 path.  The seven Q24 terms are Gaussian Mehler powers
\(\rho^0,\ldots,\rho^6\) inside derivative kernels; they are not a cumulant
order-six closure.

For the symmetric common-radial law, \(L_3=0\).  Complete-old Q24 therefore
owns the Gaussian base and the linear \(\operatorname{Var}S\)/H4 Hessian
(MP.66), but not the order-\(r^{-2}\) terms

\[
\boxed{
\frac{\kappa_3(S)}6D_Q^3\mathcal K
=\frac{80}{3r^2}D_Q^3\mathcal K,}
\tag{MP.79}
\]

nor the \(L_4^2/2\) contribution contained in the fourth radial moment.
Consequently the exact mixture (MP.59)/(MP.71) has a nonzero,
analytically typed old-subtracted tail beginning at order \(r^{-2}\).
Equations (MP.67)--(MP.70) bound it without invoking an unowned reverse
tape.

Downstream CP4/Star actions still have to be subtracted at their literal
consumer.  They do not change the reset-order fact above.

Atomic node:

- AQ_RADIAL_R2_TAIL_NOT_Q24_OWNED: solved source-order audit; endpoint value
  and downstream Star overlap open.

## 2026-08-31 — soft three-level Pareto closure: r55 can pass the loose retraction gate

Exact semicircle fourth-moment matching forced the poor p=1/2 retraction.
There is a useful one-parameter minimal-polynomial family

\[
\boxed{
\nu_p=\frac p2\delta_{-1/\sqrt p}
+(1-p)\delta_0
+\frac p2\delta_{+1/\sqrt p},
\qquad0<p\le1.}
\tag{MP.80}
\]

It has

\[
\mu_2=1,\qquad
\mu_{2k}=p^{1-k},\qquad
\boxed{A^3=p^{-1}A.}
\tag{MP.81}
\]

Thus the cubic word and local resolvent close exactly for every p.  The
quadratic fourth moment is \(1/p\), trading H4 amplitude against spectral
retraction fidelity.

Let \(a=2\cos\theta\) be the semicircle quantile whose two tails have total
mass p.  Then

\[
\boxed{
p=\frac2\pi(\theta-\sin\theta\cos\theta),}
\tag{MP.82}
\]

and the Frobenius-optimal monotone retraction to \(\nu_p\) has exact
correlation

\[
\boxed{
R(p)=\frac{8\sin^3\theta}{3\pi\sqrt p}.}
\tag{MP.83}
\]

The unconstrained maximum satisfies

\[
\boxed{
3\theta\cos\theta-3\sin\theta+2\sin^3\theta=0,}
\tag{MP.84}
\]

giving \(p\) about.63 and \(R\) about.933.  No three-level law approaches
the .983 screen, but a soft law can clear .925.

At r55, equal positive/negative multiplicities make p discrete:

\[
p=\frac{2k}{55}.
\]

The smallest convenient loose-gate choice is

\[
\boxed{
k=15,\qquad p=\frac{30}{55}=\frac6{11},\qquad
A^3=\frac{11}{6}A.}
\tag{MP.85}
\]

Its semicircle retraction correlation from (MP.83) is about.927, while

\[
\mu_4=\frac{11}{6}
\]

is only8.33% below the semicircle value2.  The zero-bias H4 mean response
differs even less: its standardized q4 is

\[
\gamma_4=\frac{12}{pr},
\]

so the leading ReLU-mean discrepancy from a semicircle core is

\[
\boxed{
\frac{\Delta\mathbb EX_+}{\phi(0)}
\simeq\frac{1-1/(2p)}r
=\frac1{12r},}
\tag{MP.86}
\]

about \(1.52\times10^{-3}\) at r55.  The local law therefore sacrifices a
small reset response to move coefficient fidelity across the loose gate.

Because it is still a three-atom invariant module, the conditional local
K32-to-6 CSE saves

\[
6(32-6)\frac{55^2}{1024^2}\simeq.4500
\]

from the strict r55 subtotal:

\[
\boxed{48.197\longrightarrow47.747,}
\tag{MP.87}
\]

leaving about1.253 for typed hooks.  The terminal-total spelling becomes
about46.190.  These costs retain all earlier scope caveats: the saving is
only lower reset, not mixed H4 action.

This soft closure is a more credible engineering point than exact p=1/2:
it keeps a one-word cubic/resolvent law, fits rank55, and passes the loose
coefficient gate.  It does not certify end-to-end value, repeated
retractions across six layers, affine-vector geometry, or the strict .983
ingredient screen.  A scalar H4 calibration may correct its8.33% local
amplitude mismatch, but that is a replacement coefficient and cannot be
double-counted as additive capacity.

Atomic node:

- AQ_R55_SOFT_THREE_LEVEL_PARETO: conditional candidate; exact local algebra
  and cost solved, six-layer Bellman retraction/value and mixed H4 remain
  open.

## 2026-08-31 — all radial cumulants and the exact formal resummation operator

The common scale in (MP.56) is normalized by

\[
\boxed{
\mathbb E e^{-uS}
=\exp\left[-\frac r2J\left(\frac{4u}{r}\right)\right].}
\tag{MP.88}
\]

Equating the scale-mixture characteristic expansion with the semicircle
quadratic-form cumulants gives, for every \(m\ge1\),

\[
\boxed{
\kappa_m(S)
=\frac{4^{m-1}(m-1)!\,\mathrm{Cat}_m}{r^{m-1}},}
\tag{MP.89}
\]

where

\[
\mathrm{Cat}_m=\frac1{m+1}\binom{2m}{m}.
\]

Indeed,

\[
\kappa_{2m}(X)
=\frac{(2m-1)!\,2^{m-1}\mathrm{Cat}_m}{r^{m-1}},
\]

and matching
\(\kappa_m(S)/(2^m m!)=\kappa_{2m}(X)/(2m)!\) proves (MP.89).
The first values are

\[
\kappa_1(S)=1,\quad
\kappa_2(S)=\frac8r,\quad
\kappa_3(S)=\frac{160}{r^2},\quad
\kappa_4(S)=\frac{5376}{r^3}.
\]

For any analytic covariance-ray observable f, the exact formal cumulant
operator is

\[
\boxed{
\mathbb E f(S)
=\left.
\exp\left[
\sum_{m\ge2}\frac{\kappa_m(S)}{m!}\partial_s^m
\right]f(s)\right|_{s=1}.}
\tag{MP.90}
\]

After Gaussian plus linear H4 subtraction, the complete order-\(r^{-2}\)
tail is therefore

\[
\boxed{
\frac{80}{3r^2}f^{(3)}(1)
+\frac8{r^2}f^{(4)}(1).}
\tag{MP.91}
\]

The second term is exactly the omitted \(L_4^2/2\) channel.  At order
\(r^{-3}\), the next displayed terms are

\[
\boxed{
\frac{224}{r^3}f^{(4)}(1)
+\frac{320}{3r^3}f^{(5)}(1)
+\frac{32}{3r^3}f^{(6)}(1),}
\tag{MP.92}
\]

coming respectively from \(\kappa_4/24\),
\(\kappa_2\kappa_3/12\), and \(\kappa_2^3/48\).
Equations (MP.88)--(MP.92) give either an exact one-dimensional mixture
evaluation or a systematically residualized analytic series.  The latter
still needs a convergence/secant bound near correlation edges; it is not a
license to use the divergent connected-cumulant response series globally.

Atomic node:

- AQ_RADIAL_CUMULANT_RESUMMATION: exact scale cumulants and omitted
  \(r^{-2}\) operator solved; Bellman action/correlation edge control open.

## 2026-08-31 — lawful forward radial-replacement recurrence

The radial theorem can be installed as a complete coadapted forward law,
without an unowned reverse.  At an active reset retain:

\[
m_l,\qquad C_l,\qquad Q_l,
\]

where \(C_l\) is the carried total covariance and \(Q_l\) is the separately
retained paid quadratic AQ Gram.  Define

\[
\boxed{L_l=C_l-Q_l.}
\tag{MP.93}
\]

The first gate is \(L_l\succeq0\).  Conditional on the common scale S, use

\[
h_l\mid S=s\sim N(m_l,L_l+sQ_l)
\tag{MP.94}
\]

and replace the current finite Edgeworth pair reset by the exact mixture

\[
\boxed{
M_{l,+}
=\mathbb E_S\,
\mathbb E_{N(m_l,L_l+SQ_l)}
[h_+h_+^T],}
\tag{MP.95}
\]

\[
\boxed{
\mu_{l,+}
=\mathbb E_S\,
\mathbb E_{N(m_l,L_l+SQ_l)}[h_+],}
\tag{MP.96}
\]

\[
\boxed{
C_{l,+}=M_{l,+}-\mu_{l,+}\mu_{l,+}^T.}
\tag{MP.97}
\]

These matrices are automatically lawful:

\[
M_{l,+}\succeq0,\qquad C_{l,+}\succeq0,
\]

because they are the second moment and covariance of the literal mixture
law (MP.94).  No entrywise repair is needed.

In parallel, the paid local AQ reset produces the next quadratic coefficient
bank and its Gram \(Q_{l,+}\).  The synchronization gate is

\[
\boxed{C_{l,+}-Q_{l,+}\succeq0.}
\tag{MP.98}
\]

If it fails, the largest lawful quadratic scale is again

\[
\alpha_{\max}^2
=\lambda_{\max}
\left(C_{l,+}^{-1/2}\widetilde Q_{l,+}
C_{l,+}^{-1/2}\right)^{-1}.
\tag{MP.99}
\]

Then dense mixing transports \(C\) and the AQ coefficient bank in the
already-paid forward paths, and the construction repeats.  This is a
self-consistent Galerkin replacement: the independent Gaussian rest absorbs
everything outside the retained quadratic bank at every reset.

### Cost and novelty

For K scalar S-nodes, (MP.95) is K elementwise bivariate Gaussian ReLU kernel
evaluations:

\[
\boxed{O(Kn^2)\quad\text{or roughly}\quad K/n}
\]

dense-GEMM equivalents per reset.  At n1024,K32 and six resets this is about
.1875, fitting the strict r55 paid-Gram headroom .803.  A direct analytic
order-\(r^{-2}\) implementation of (MP.91) is \(O(n^2)\) with no K factor.

This must replace, not add to, the Q24 pair plane.  The exact old-subtracted
novelty is

\[
\boxed{
R^{rad}_{l}
=M^{mixture}_{l,+}
-M^{Q24}_{l,+},}
\tag{MP.100}
\]

with downstream CP/Star overlap removed at its consumer.  The source audit
proves (MP.100) begins nontrivially at order \(r^{-2}\).  Injecting the full
replacement plane into the existing forward covariance recurrence satisfies
the coadaptation requirement which the frozen hand-injection carriers
violated.

Atomic node:

- AQ_FORWARD_RADIAL_REPLACEMENT: exact model/PSD recurrence and cap-compatible
  cost solved; seam \(Q\preceq C\), next-reset synchronization, finite-matrix
  model error, and raw endpoint value remain open.

## 2026-08-31 — exact Gamma radial law for the soft three-level closure

The soft three-level spectrum has a simpler exact scalar mixture than the
semicircle generator.  Let k be each of the equal positive and negative
multiplicities.  The standardized pure quadratic form is

\[
X_k=\frac{\chi_k^2-\widetilde\chi_k^2}{2\sqrt k},
\]

with characteristic function

\[
\boxed{
\mathbb E e^{itX_k}
=\left(1+\frac{t^2}{k}\right)^{-k/2}.}
\tag{MP.101}
\]

This is exactly a normal variance mixture

\[
\boxed{
X_k\overset d=\sqrt S\,G,\qquad
S\sim\operatorname{Gamma}\left(\frac k2,\frac2k\right),
\quad G\sim N(0,1),}
\tag{MP.102}
\]

where the Gamma parameters are shape and scale.  Thus
\(\mathbb ES=1\), and

\[
\boxed{
\kappa_m(S)
=\frac{2^{m-1}(m-1)!}{k^{m-1}}.}
\tag{MP.103}
\]

For the r55 Pareto point (MP.85), k=15.  Its first radial cumulants are

\[
\operatorname{Var}S=\frac2{15},\qquad
\kappa_3(S)=\frac8{225},\qquad
\kappa_4(S)=\frac{48}{3375}.
\tag{MP.104}
\]

Replacing the abstract semicircle scale in the forward recurrence by this
Gamma scale makes every retained one-unit marginal exactly equal to the
three-level quadratic law.  The pair model is

\[
\boxed{
h\mid S=s\sim N(m,L+sQ),
\qquad S\sim\Gamma(15/2,2/15).}
\tag{MP.105}
\]

It remains a common-scale closure assumption across different unit cores,
but no longer has a marginal-law mismatch.

After Q24's linear H4 subtraction, its exact order-\(k^{-2}\) tail is

\[
\boxed{
\frac{4}{3k^2}D_Q^3\mathcal K
+\frac1{2k^2}D_Q^4\mathcal K.}
\tag{MP.106}
\]

At k15 the coefficients are about.00593 and.00222.  The full expectation can
be evaluated particularly naturally by Gauss-Laguerre quadrature in S.
At zero bias and with L=0, positive homogeneity again makes the uncentered
second moment exactly Gaussian; the novel terms require means or
affine/quadratic split heterogeneity.

Combining the conditional local-reset CSE (MP.87) with K32 pair-mixture
evaluation gives the optimistic strict subtotal

\[
\boxed{
47.747+6\frac{32}{1024}\simeq47.935,}
\tag{MP.107}
\]

leaving about1.065 before additional hooks.  A lower-node Gamma quadrature
or analytic (MP.106) uses less.

Atomic node:

- AQ_R55_SOFT_GAMMA_RADIAL_CLOSURE: exact one-unit law and cap-compatible
  pair-reset model solved; common-scale finite-matrix defect, six-layer
  retraction, synchronization, and raw value remain open.

## 2026-08-31 — exact sixth-cumulant defect gate for any radial closure

The leading omitted tail is only useful if the actual cross-unit sixth
cumulants are close to the radial prediction.  For half-quadratic forms
\(\xi_i=\tfrac12I_2(A_i)\), the exact joint cumulant is

\[
\boxed{
\kappa_p(\xi_{i_1},\ldots,\xi_{i_p})
=\frac1{2p}\sum_{\pi\in S_p}
\operatorname{tr}
(A_{i_{\pi(1)}}\cdots A_{i_{\pi(p)}}).}
\tag{MP.108}
\]

If \(Q_{ij}=\mathbb E[\xi_i\xi_j]
=\tfrac12\operatorname{tr}(A_iA_j)\), a common radial scale predicts

\[
\boxed{
\kappa_{2m}^{rad}(i_1,\ldots,i_{2m})
=\kappa_m(S)
\sum_{\mathcal P_2}
\prod_{(a,b)\in\mathcal P_2}Q_{i_ai_b},}
\tag{MP.109}
\]

where the sum is over all pairings of the 2m legs.  In particular the exact
sixth-order model defect is

\[
\boxed{
\Delta^{(6)}_{i_1\ldots i_6}
=\frac1{12}\sum_{\pi\in S_6}
\operatorname{tr}
(A_{i_{\pi(1)}}\cdots A_{i_{\pi(6)}})
-\kappa_3(S)\sum_{\mathcal P_2}
\prod_{(a,b)\in\mathcal P_2}Q_{i_ai_b}.}
\tag{MP.110}
\]

The fourth-order counterpart is the earlier E/D free-Wick defect.  Since
Q24 already owns the linear fourth-order action, (MP.110) is the first new
structural certificate for the radial tail.  Promotion needs the
complete-old Bellman contraction of (MP.110), plus the square of the actual
fourth-cumulant defect, to lie below the endpoint error gate.  Matching a
single-unit variance-gamma marginal or a q4 collision face is insufficient.

For a pair (i,j), all required sixth collision faces are encoded compactly
by comparing

\[
\frac1{12}\operatorname{tr}(tA_i+sA_j)^6
\]

with the coefficient polynomial generated by

\[
\frac{\kappa_3(S)}{48}
\left(t^2Q_{ii}+2tsQ_{ij}+s^2Q_{jj}\right)^3.
\tag{MP.111}
\]

This is the exact fail-fast object for a collision-only pair interface; a
full six-index tensor need not be formed.

Atomic node:

- AQ_RADIAL_SIXTH_CUMULANT_DEFECT: exact structural gate solved; actual-cut
  target contraction remains open.

### Coherence correction for the soft-Gamma combination

A common scalar scale necessarily satisfies, for standardized uncorrelated
coordinates,

\[
\kappa_4(i,i,i,i)=3\kappa_4(i,i,j,j).
\tag{MP.112}
\]

The soft p=6/11 marginal has

\[
\kappa_4(i,i,i,i)=6/k=.400,
\]

so its common-Gamma prediction is
\(\kappa_4(i,i,j,j)=2/k=.1333\).  But independent Haar-oriented quadratic
cores have

\[
\mathbb E\kappa_4(i,i,j,j)
=\frac{4(2+\eta_r)}r
\simeq.1467
\]

at r55.  The roughly10% cross defect is real.  Exact p=1/2 is the unique
three-level fourth-moment point which matches the leading common-radial
ratio, up to the finite-r eta correction.

Therefore the soft-Gamma node is an internally PSD replacement model with
exact one-unit marginals, not an exact joint law for generic rotated cores.
The clean common-radial candidate is the generic/full-AQ semicircle closure
at strict cost

\[
\boxed{48.197+.1875=48.3845}
\tag{MP.113}
\]

for six K32 pair-mixture resets.  It avoids spectral retraction but retains
the typed-seam and finite-rank E/D gates.  The soft three-level branch remains
a separate polynomial closure and must not borrow exact common-scale
semantics.

## 2026-08-31 — exact centered old-subtracted radial remainder

Let \(a(s)\) be the vector of conditional Gaussian ReLU means and \(B(s)\)
the raw conditional pair second moment for
\(N(m,L+sQ)\).  Write \(a_j=\partial_s^ja(1)\),
\(B_j=\partial_s^jB(1)\), and

\[
p=\frac{\kappa_2(S)}2a_2,
\]

\[
r_a=\mathbb Ea(S)-a_0-p,\qquad
r_B=\mathbb EB(S)-B_0-\frac{\kappa_2(S)}2B_2.
\]

After exact Gaussian plus linear-H4 ownership, the centered old-subtracted
radial plane is

\[
\boxed{
\begin{aligned}
R_{rad}={}&r_B-pp^T\\
&-(a_0+p)r_a^T-r_a(a_0+p)^T-r_ar_a^T.
\end{aligned}}
\tag{MP.114}
\]

Its leading omitted terms are

\[
\boxed{
\begin{aligned}
R_{rad}^{(2)}={}&
\frac{\kappa_3}{6}
\left(B_3-a_3a_0^T-a_0a_3^T\right)\\
&+\frac{\kappa_2^2}{8}
\left(B_4-a_4a_0^T-a_0a_4^T\right)
-\frac{\kappa_2^2}{4}a_2a_2^T.
\end{aligned}}
\tag{MP.115}
\]

If scalar integral/secant errors obey
\(\|r_B\|_F\le\epsilon_B\) and
\(\|r_a\|_2\le\epsilon_a\), then

\[
\boxed{
\|R_{rad}\|_F
\le\epsilon_B+\|p\|_2^2
+2\|a_0+p\|_2\epsilon_a+\epsilon_a^2.}
\tag{MP.116}
\]

In the centered pure case \(m=0,L=0\), homogeneity gives

\[
a(s)=\sqrt s\,d,\qquad B(s)=sB_0,
\]

and the full remainder collapses exactly to

\[
\boxed{
R_{rad}=\beta\,dd^T,\qquad
\beta=1-(\mathbb E\sqrt S)^2-\frac{\kappa_2(S)}4.}
\tag{MP.117}
\]

For the semicircle scale,

\[
\boxed{\beta=-\frac6{r^2}+O(r^{-3}).}
\tag{MP.118}
\]

Hence even the genuinely Q24-omitted q4-squared/q6 tail is only a collective
rank-one direction in the pure centered sector.  Its endpoint action is
\(\beta d^T\Lambda d\).  The measured collective-carrier failure therefore
closes this sector as the source of broad variance.  Noncollective radial
value requires nonzero standardized means, a nontrivial affine/rest plane L,
or later nonlinear mixing.

Atomic node:

- AQ_CENTERED_RADIAL_TAIL_RANK1_NOGO: exact; pure centered omitted tail is
  collective rank1, leaving only bias/L heterogeneity.

## 2026-08-31 — operator-isotropic shared-z resolvent: the missing mixed L6 term

The independent-affine mixture is not the only scalar closure.  Return to
the literal shared Gaussian AQ variable.  For a test vector t over units,
write

\[
B(t)=\sum_i t_iA_i,\qquad
v(t)=\sum_i t_iu_i,
\]

\[
q=t^TQt=\frac12\operatorname{tr}B(t)^2,\qquad
\ell=t^TLt=v(t)^Tv(t).
\]

The exact finite-AQ characteristic function is

\[
\det(I-iB)^{-1/2}
\exp\left[-\frac12v^T(I-iB)^{-1}v\right]
\]

after centering the quadratic trace.  Impose the operator-isotropy law

\[
\boxed{
v(t)^Tf(B(t))v(t)
=\ell\,\tau[f(B(t))]}
\tag{MP.119}
\]

for the resolvent functions in question, and the joint-semicircle spectral
law for B(t).  Then the complete characteristic exponent collapses to

\[
\boxed{
\log\varphi(t)
=-\frac r2J\left(\frac{2q}{r}\right)
-\frac{\ell}{2}
R\left(\frac{2q}{r}\right),}
\tag{MP.120}
\]

where

\[
\boxed{
R(x)=\frac{2}{1+\sqrt{1+4x}}.}
\tag{MP.121}
\]

Thus the exact same-z affine/quadratic law is determined by the two resident
Grams L and Q, without an independence approximation.  The formal expansion
is

\[
\boxed{
\begin{aligned}
\log\varphi(t)
={}&-\frac12(\ell+q)
+\frac1r(q^2+\ell q)\\
&-\frac1{r^2}
\left(\frac{10}{3}q^3+4\ell q^2\right)\\
&+\frac1{r^3}\left(14q^4+20\ell q^3\right)
+O(r^{-4}).
\end{aligned}}
\tag{MP.122}
\]

The \(r^{-1}\) polynomial is exactly the linear affine H4-square/Price
sector.  The deployed Q24 reset owns it.  At the first omitted order,
however, there are two sixth-degree sources:

\[
\boxed{
K_6^{new}(t)
=-\frac1{r^2}
\left[\frac{10}{3}(t^TQt)^3
+4(t^TLt)(t^TQt)^2\right].}
\tag{MP.123}
\]

The first is the pure radial L6 term.  The second is a genuinely mixed
shared-z affine/quadratic L6 term absent from the independent-mixture model.
Both are universal polynomials of L and Q.  Exponentiating (MP.122) also
produces the omitted \(L_4^2/2\) density channel.

This gives a particularly cheap implementation route.  Q24 already
evaluates Gaussian Mehler/derivative kernels through degree6; it lacks only
the q6 source coefficients.  Contracting (MP.123) with those resident
degree-six derivative tables is elementwise \(O(n^2)\), not a new matrix
product or reverse.  The \(L_4^2\) term needs derivative degree8 and is a
separate small extension.

The exact gates are:

1. the resolvent-isotropy defect
   \(v^TB^kv-\ell\tau(B^k)\) for k through4 at the omitted order;
2. the finite-matrix sixth word defect (MP.110);
3. complete-old CP/Star subtraction;
4. the endpoint Bellman contraction of the resulting L6 plane.

Unlike the pure centered radial tail, the mixed \(\ell q^2\) polynomial can
be broad whenever L and Q are not proportional.  It is the most economical
same-gauge noncollective provider identified in this lane.

Atomic node:

- AQ_SHARED_Z_ISOTROPIC_L6_PROVIDER: exact under (MP.119); universal
  \(q^3+\ell q^2\) omitted source, O(n²) degree-six action, isotropy/endpoint
  value open.

## 2026-08-31 — earlier omitted reset order: complete the owned L3/L4 exponential

The literal Q24 density truncation

\[
1+L_3+L_4+\frac12L_3^2
\]

is complete only through the usual diffuse order \(r^{-1}\), with

\[
L_3=O(r^{-1/2}),\qquad L_4=O(r^{-1}).
\]

Because the fixed-site cumulant differential operators have constant
coefficients, they commute.  Expanding the exact formal operator

\[
\exp(L_3+L_4+L_5+L_6+\cdots)
\]

shows that the complete next order \(r^{-3/2}\) is

\[
\boxed{
L_5+L_3L_4+\frac16L_3^3.}
\tag{MP.124}
\]

Even if no fifth-cumulant provider is acquired, the two terms

\[
\boxed{L_3L_4+\frac16L_3^3}
\tag{MP.125}
\]

use only already-owned signed-H3 and H4 coefficients.  They are absent from
the deployed reset and are generically larger in asymptotic order than the
new L6 source (MP.123).

At order \(r^{-2}\), the already-owned-source completion contains

\[
\boxed{
\frac12L_4^2+\frac12L_3^2L_4+\frac1{24}L_3^4,}
\tag{MP.126}
\]

in addition to new-provider terms \(L_6+L_3L_5\).  Q24 currently owns none
of (MP.125)--(MP.126).

The Gaussian derivative degrees are:

\[
\deg(L_3L_4)=7,\quad
\deg(L_3^3)=9,\quad
\deg(L_4^2)=8,\quad
\deg(L_3^2L_4)=10,\quad
\deg(L_3^4)=12.
\]

The existing Mehler recurrence through degree6 can be extended to these
degrees elementwise; this is \(O(n^2)\) scalar work and does not require a
new matrix word, plane transport, or reverse tape.  The exact novelty is a
reset replacement action, with downstream CP/Star duplicates subtracted.

There are two safety gates.

1. A finite Edgeworth polynomial is not a positive density.  Use the
   completed terms as moment/action features or inside a secant/resummed
   law; do not assert distributional positivity.
2. The actual standardized H3/H4 sizes and Gaussian derivative norms must
   satisfy a remainder radius.  Diffuse order counting alone is not a value
   certificate.

Since the certified chain attributes a small but nonzero value to k3, these
owned-source interaction terms should be tested before paying for any new
L5/L6 matrix carrier.  They are the strictest fail-fast consequence of the
source audit.

Atomic node:

- AQ_OWNED_L3_L4_EXPONENTIAL_COMPLETION: exact omitted operator algebra
  solved; degree7/9/8 action, positivity radius, downstream overlap, and raw
  value open.

## MP-RESET-OWNED-EXPONENTIAL — exact order bookkeeping, orthogonality, and pairwise convolution

Let the standardized local law have constant-coefficient cumulant operators

\[
\mathcal L_d={(-1)^d\over d!}\,\kappa_d:\nabla^d,
\qquad \mathcal L_d=O(\epsilon^{d-2}),\quad \epsilon=r^{-1/2}.
\]

Because all \(\mathcal L_d\) commute, the formal density/reset operator is

\[
\exp\!\left(\sum_{d\ge3}\mathcal L_d\right).
\]

Writing \(A=\mathcal L_3/\epsilon\), \(B=\mathcal L_4/\epsilon^2\),
\(C=\mathcal L_5/\epsilon^3\), and \(D=\mathcal L_6/\epsilon^4\), its exact expansion is

\[
\begin{aligned}
I
{}&\epsilon A\\
{}&+\epsilon^2\left(B+{A^2\over2}\right)\\
{}&+\epsilon^3\left(C+AB+{A^3\over6}\right)\\
{}&+\epsilon^4\left(D+AC+{B^2\over2}+{A^2B\over2}+{A^4\over24}\right)
+O(\epsilon^5).
\end{aligned}
\]

Thus the deployed \(I+\mathcal L_3+\mathcal L_4+\mathcal L_3^2/2\) stops after
order \(r^{-1}\).  Before acquiring any new fifth/sixth cumulant, the paid H3/H4
sources determine the exact omitted pieces

\[
\boxed{\mathcal L_3\mathcal L_4+\mathcal L_3^3/6}\qquad (r^{-3/2})
\]

and

\[
\boxed{\mathcal L_4^2/2+\mathcal L_3^2\mathcal L_4/2+\mathcal L_3^4/24}
\qquad (r^{-2}).
\]

The genuinely new partners are \(\mathcal L_5\) at \(r^{-3/2}\), and
\(\mathcal L_6+\mathcal L_3\mathcal L_5\) at \(r^{-2}\).

### Why omitting L5 does not make the owned completion algebraically redundant

After whitening, \(\mathcal L_d\phi\) lies in Gaussian chaos degree \(d\).
Consequently, at order \(r^{-3/2}\),

\[
\mathcal L_5\phi\in\mathcal H_5,\qquad
\mathcal L_3\mathcal L_4\phi\in\mathcal H_7,\qquad
\mathcal L_3^3\phi\in\mathcal H_9,
\]

which are mutually orthogonal in \(L^2(\phi^{-1})\).  Therefore adding the
owned degree-7/9 terms strictly reduces the leading density error by their
squared norm even when the unknown degree-5 term remains.  A downstream scalar
observable can still give oppositely signed actions, so the deployment gate is
the task/Bellman contraction, but there is no density-level cancellation excuse
for leaving these terms out.

### Literal bivariate construction

For a neuron pair, write

\[
\mathcal L_3=\sum_{a+b=3}u_{ab}\,\partial_x^a\partial_y^b,
\qquad
\mathcal L_4=\sum_{c+d=4}v_{cd}\,\partial_x^c\partial_y^d,
\]

where factorials and signs are absorbed into \(u,v\).  Then every new owned
weight is only a finite convolution of already formed pairwise arrays:

\[
w^{34}_{pq}=\sum_{a+c=p\atop b+d=q}u_{ab}v_{cd},\qquad p+q=7,
\]

\[
w^{333}_{pq}={1\over6}\sum_{a_1+a_2+a_3=p\atop b_1+b_2+b_3=q}
u_{a_1b_1}u_{a_2b_2}u_{a_3b_3},\qquad p+q=9.
\]

The order-\(r^{-2}\) weights are analogous convolutions at total derivative
degrees 8, 10, and 12.  Hence no matrix word, plane, reverse tape, or dense
matrix product is required once the literal pairwise H3/H4 weight arrays are
available.  The only new operations are a constant number of Hadamard products
and Gaussian derivative/Mehler channels.

At zero correlation, if \(h(x)=x_+\),

\[
E[h(Z)H_0(Z)]=\phi(0),\quad E[h(Z)H_1(Z)]=1/2,
\]

\[
E[h(Z)H_{2m}(Z)]=(-1)^{m-1}(2m-3)!!\,\phi(0),\quad
E[h(Z)H_{2m+1}(Z)]=0\ (m\ge1).
\]

Therefore the degree-7, degree-9, and degree-5 actions survive at an independent
zero-bias pair only through collision splits \((1,6)/(6,1)\),
\((1,8)/(8,1)\), and \((1,4)/(4,1)\), respectively.  This is a sharp fail-fast
screen: if the retained H3/H4 tensors have negligible contractions in those
one-leg collision sectors (and biases/correlations do not reopen them), the
owned exponential completion is value-dark despite being mathematically exact.

### Close/reopen rule

Close the owned-completion node only if the complete cut-to-final Bellman action
annihilates both convolution sectors, or their task-weighted action is below the
required raw-gain threshold after old-span subtraction.  Reopen immediately if
either degree-7 or degree-9 pair collision has nonzero target action.  Do not
charge it as a new H5/H6 provider and do not credit it twice through a generic
"higher Edgeworth" node.

### Exact independent, zero-bias action: diagonal scaling of K21/K31

For standardized pair \((X_i,X_j)\) with zero correlation, define the oriented
collision cumulants

\[
c_{3j}=\kappa_{jjj},\quad c_{4j}=\kappa_{jjjj},\quad
P_{ij}=\kappa_{ijj},\quad R_{ij}=\kappa_{ijjj}.
\]

Use the conventional cumulant operator

\[
\mathcal L_d=\sum_{a+b=d}{\kappa_{i^a j^b}\over a!b!}
\partial_i^a\partial_j^b
\]

after Gaussian integration by parts (so derivative signs have already canceled).
Only the derivative splits \((1,6),(6,1)\) from \(\mathcal L_3\mathcal L_4\)
and \((1,8),(8,1)\) from \(\mathcal L_3^3/6\) survive.  Their oriented
coefficients are

\[
[\partial_i\partial_j^6](\mathcal L_3\mathcal L_4)
={c_{3j}R_{ij}\over3!3!}+{P_{ij}c_{4j}\over2!4!},
\]

\[
[\partial_i\partial_j^8](\mathcal L_3^3/6)
={P_{ij}c_{3j}^2\over 2\,2!\,(3!)^2}.
\]

Since for \(h(x)=x_+\),

\[
E h'(Z)=1/2,\qquad E h^{(6)}(Z)=3\phi(0),\qquad
E h^{(8)}(Z)=-15\phi(0),
\]

the exact owned \(r^{-3/2}\) pair correction at \(\rho=0,m=0\) is

\[
\boxed{
\begin{aligned}
\Delta^{\rm owned}_{ij}=\phi(0)\bigg[&
{c_{3j}R_{ij}+c_{3i}R_{ji}\over24}
+{c_{4j}P_{ij}+c_{4i}P_{ji}\over32}\\
&-{5\over96}\left(c_{3j}^2P_{ij}+c_{3i}^2P_{ji}\right)
\bigg].
\end{aligned}}
\]

Thus this candidate is not a new tensor carrier.  In matrix notation it is a
symmetrized column scaling of the paid oriented K31 plane \(R\) and K21 plane
\(P\) by the paid diagonal stars \(c_3,c_4\).  Formation is Hadamard/diagonal
scaling only.  It can nevertheless be broad whenever K21/K31 is broad; it is
not inherently a collective rank-one correction.

This formula is also a sharp normalization receipt for any implementation:
at an independent zero-bias sentinel, a proposed degree-7/9 kernel must reduce
exactly to the boxed expression (with the same standardized-cumulant convention).
Bias and nonzero correlation reopen the other derivative splits but do not alter
the source-ownership conclusion.

### Arbitrary-correlation zero-bias theorem

The preceding collision formula in fact remains unchanged for every
\(|\rho|<1\).  Let

\[
D_{pq}(\rho)=\left.\partial_{m_i}^p\partial_{m_j}^q
E[(Z_i+m_i)_+(Z_j+m_j)_+]\right|_{m_i=m_j=0},
\]

where \((Z_i,Z_j)\) are standard Gaussian with correlation \(\rho\).  For
\(p,q\ge2\), simultaneous sign symmetry of the bivariate density gives

\[
D_{pq}(\rho)=0\qquad\text{whenever }p+q\text{ is odd}.
\]

For the one-leg sectors, conditioning on the even-order coordinate gives

\[
D_{1,2m}(\rho)={1\over2}\phi^{(2m-2)}(0),
\]

which is independent of \(\rho\).  In particular

\[
D_{16}=3\phi(0)/2,qquad D_{18}=-15\phi(0)/2.
\]

The only correlation-reopened sectors are the pure-side derivatives.  A direct
conditional-Gaussian calculation gives

\[
D_{0,2m+1}(\rho)=(-1)^m{(2m-1)!!\over2}\rho\phi(0),
\]

so

\[
D_{07}=-15\rho\phi(0)/2,qquad D_{09}=105\rho\phi(0)/2.
\]

Consequently the exact zero-bias, arbitrary-correlation owned correction is the
previous boxed broad collision term plus

\[
\boxed{
\Delta^{\rm pure}_{ij}=\rho_{ij}\phi(0)
\left[-{5\over96}(c_{3i}c_{4i}+c_{3j}c_{4j})
+{35\over864}(c_{3i}^3+c_{3j}^3)\right].}
\]

This extra term is merely the existing correlation matrix with row/column
diagonal-star scaling.  Thus correlation does not generate a hidden new
degree-7/9 carrier at zero bias: all broad information remains exactly in the
paid oriented K21/K31 planes.  Nonzero means are the only way to reopen the
interior derivative splits at this order.

### Exact biased derivative recurrence (no Mehler-depth premise)

All derivative kernels needed by the owned completion can be evaluated exactly
at constant work per neuron pair.  Let \(\Sigma=\begin{psmallmatrix}1&\rho\\
\rho&1\end{psmallmatrix}\), \(P=\Sigma^{-1}\), and define multivariate Gaussian
Hermites by

\[
H_{ab}^{\Sigma}(x,y)=(-1)^{a+b}\phi_\Sigma(x,y)^{-1}
\partial_x^a\partial_y^b\phi_\Sigma(x,y).
\]

For \(p,q\ge2\), the biased ReLU-pair derivative is exactly

\[
\boxed{D_{pq}(m,n;\rho)=
H_{p-2,q-2}^{\Sigma}(-m,-n)\phi_\Sigma(-m,-n).}
\]

The Hermites obey the two-coordinate recurrence

\[
H_{\alpha+e_i}=(Px)_iH_\alpha-
\sum_jP_{ij}\alpha_jH_{\alpha-e_j},
\]

so all total degrees through 12 require only a fixed triangular table.

Boundary cases are one-dimensional derivatives.  With
\(s=\sqrt{1-\rho^2}\),

\[
D_{1q}=(-1)^{q-2}\left.\partial_y^{q-2}
\left[\phi(y)\Phi\left({m+\rho y\over s}\right)\right]
\right|_{y=-n},
\]

and

\[
D_{0q}=(-1)^{q-2}\left.\partial_y^{q-2}\left\{
\phi(y)\left[s\phi\left({m+\rho y\over s}\right)
+(m+\rho y)\Phi\left({m+\rho y\over s}\right)\right]\right\}
\right|_{y=-n}.
\]

The \(p0,p1\) cases follow by symmetry.  Hence the degree-7/9/8/10/12
completion needs neither a longer Mehler series nor quadrature in principle.
It is an exact \(O(n^2)\) local kernel table.  The only analytic conditioning
gate is \(|\rho|\to1\), where \(P\) diverges; use the same declared correlation
clip/active-set as the baseline or a stable conditional-limit formula, and do
not compare a differently clipped candidate to the old reset.

## AQ-OWNED-COMPLETION-IDENTITY (atomic graph node)

**Status: PROVED.**  The exact old-source density increment missing from the
deployed order-\(r^{-1}\) reset is

\[
\Delta\mathcal E_{3/2}=\mathcal L_3\mathcal L_4+\mathcal L_3^3/6.
\]

It is determined entirely by the already paid signed-H3 and H4 cumulants, but
it is a new reset functional of those sources.  "Old source" is therefore not
the same as "old output".  The identity is independent of any matrix-carrier,
free-probability, or AQ approximation.

## AQ-OWNED-COMPLETION-LITERAL-D11-CENSUS (atomic graph node)

**Status: algebra typed; exact bill still OPEN.**  The literal D11 implementation
currently has 17 pair channels: degrees 0, 3, 4, and 6, with four H3 weights,
five H4 weights, and seven \(\mathcal L_3^2/2\) convolution weights.  It forms
the oriented arrays

\[
u=(k_{3j}/3!,\ k_{21,ji}/2!,\ k_{21,ij}/2!,\ k_{3i}/3!),
\]

\[
v=(k_{4j}/4!,\ k_{31,ji}/3!,\ k_{22,ij}/(2!2!),\
k_{31,ij}/3!,\ k_{4i}/4!).
\]

The new degree-7 weights are exactly the length-8 polynomial convolution
\(u*v\).  The degree-9 weights are the length-10 convolution \(u^3/6\), or,
reusing the seven stored \(u^2/2\) channels, \((u^2/2)*u/3\).  This needs 20
and 28 pair-plane Hadamard products, respectively (plus within-channel sums),
and zero dense matrix products.

The univariate correction must be changed at the same time.  Its two new
standardized weights are

\[
w_7={k_3k_4\over144\,\sigma^7},\qquad
w_9={k_3^3\over1296\,\sigma^9}.
\]

The existing coefficient bank already contains \(c_{p,7},c_{p,9}\) for
\(p=1,2,3\).  For the separately formed fourth raw moment, the exact missing
coefficients are

\[
c_{4,7}=24\sigma^4(\alpha^2-1)\phi(\alpha),
\]

\[
c_{4,9}=24\sigma^4(\alpha^4-6\alpha^2+3)\phi(\alpha).
\]

The current pair context has derivative base rows only 0 through 6 and inverse
sigma powers only through 6.  A literal matched-Q24 extension to order
\(r^{-3/2}\) therefore needs base rows 0 through 9, inverse powers through 9,
and 18 additional derivative channels, but **no new correlation powers**:
the same declared Mehler truncation \(t=0,\ldots,6\) applies.  Calling this
mathematically "exact" means exact relative to that same \(t\le6\) Gaussian
kernel approximation; the finite Hermite recurrence above removes the Mehler
premise if desired.

A naive monolithic extension changes the four-moment weighted-kernel temporary
from \(4\times17\times n^2\) to \(4\times35\times n^2\) f64 entries: at
\(n=1024\), an extra 576 MiB, plus 144 MiB for 18 persistent weight planes.
It must instead stream small derivative groups and reuse the ten-plane product
workspace.  Arithmetic in the existing einsum path is linear in channel count,
so its incremental pair-kernel bill is exactly \((18/17)\) times the current
17-channel kernel block, before small Hadamard/source overhead.  Therefore
"no GEMM" is not a zero-cost certificate.  Promotion under the strict 0.803
slack requires either the metered streamed bill or a proved recurrence/CSE bill;
the full biased four-moment implementation is HOLD until then.  The boxed
zero-bias M11 specialization is much cheaper but is not by itself a lawful
four-moment closure update.

## AQ-OWNED-COMPLETION-OLD-SPAN (atomic graph node)

**Status: exact subtraction rule PROVED; quotient value OPEN.**  Let \(z\)
collect the four marginal raw moments and the four pair raw moments
\((M_{11},M_{21},M_{31},M_{22})\), and let \(F\) be the literal connected-plane,
covariance-repair, clipping, and state-write map.  The only lawful local novelty
is

\[
\nu=F(z+\Delta z)-F(z),
\]

with \(\Delta z\) formed from the degree-7/9 density terms in *both* the
univariate and bivariate paths.  One must not hand-inject the raw \(M_{11}\)
formula into covariance or the connected planes.

This matters because the existing nonlinear conversion already generates some
order-\(r^{-3/2}\) products from its retained order-\(r^{-1/2}\) and
order-\(r^{-1}\) moments.  Re-expanding connected cumulants and then adding all
formal cross terms would count those pieces twice.  Baseline-versus-candidate
evaluation through the identical map \(F\) is the exact subtraction.

On a fixed repair/clip active set,

\[
\nu=DF_z[\Delta z]+O(\|\Delta z\|^2).
\]

The complete-old quotient is \((I-P_{\mathcal O})\nu\), where \(\mathcal O\)
is the complete same-cut old response space, not merely the source span of
H3/H4.  A repair active-set change must be reported as a separate nonlinear
repair contribution; it is not evidence that the Edgeworth term itself has
large value.

## AQ-OWNED-COMPLETION-POSITIVITY-REMAINDER (atomic graph node)

**Status: global density positivity IMPOSSIBLE for a nonzero finite odd-order
completion; moment admissibility/remainder OPEN.**  A finite Edgeworth density
whose highest nonzero term contains H7 or H9 is sign-indefinite: its odd-degree
polynomial tends to opposite signs in opposite tails.  Thus no global density
positivity proof exists for this truncation except when the degree-7/9 sector
vanishes.  This does not invalidate its asymptotic moment action, but it forbids
describing it as an exact probability law.

At minimum, the updated nonnegative-variable moment Hankel matrix

\[
\begin{pmatrix}
1&r_1&r_2\\r_1&r_2&r_3\\r_2&r_3&r_4
\end{pmatrix}\succeq0
\]

and the updated covariance must be PSD before repair.  Matching the baseline
repair active set is the clean sentinel.

The unknown \(\mathcal L_5\phi\in\mathcal H_5\) is orthogonal in Gaussian
density norm to the owned H7/H9 pieces, but its endpoint action can have the
opposite sign.  If \(\Gamma_\ell\) is the local future Bellman test, a certified
sign requires a bound of the form

\[
|\langle\Gamma_\ell,\mathcal L_5\phi\rangle|
+|R_{\ge2,\ell}|<
|\langle\Gamma_\ell,(\mathcal L_3\mathcal L_4+\mathcal L_3^3/6)\phi\rangle|.
\]

Without an H5 bound, this candidate is an exact old-source completion and a
high-priority screen, not a proof that the true reset error decreases for the
endpoint task.

## AQ-OWNED-COMPLETION-ENDPOINT-VALUE (atomic graph node)

**Status: action formula PROVED; required contractions OPEN.**  Let \(T_{\ell:L}\)
be the fixed-baseline tangent from the local written state to the final output,
and let \(g_\ell=T_{\ell:L}^*e\) be the residual-weighted endpoint adjoint.  The
first-order signed gain contribution is

\[
a_\ell=\langle DF_z^*g_\ell,\Delta z_\ell\rangle.
\]

This scalar is necessary but not sufficient for MSE: the endpoint correction
vector \(d\), its residual inner product, and its norm are all required.  For
baseline residual \(e\), the exact fixed-coefficient raw gain is

\[
G=-2\langle e,d\rangle-\|d\|^2.
\]

To move raw MSE from \(2.9409\times10^{-8}\) to \(10^{-8}\), the normalized
gain must be at least \(1.9409\times10^{-8}\).  If a single new endpoint
direction may be optimally rescaled after complete-old projection,

\[
G_{\max}={\langle e,d_\perp\rangle^2\over\|d_\perp\|^2},\qquad
d_\perp=(I-P_{\mathcal O})d,
\]

so goal-scale success requires

\[
|\operatorname{corr}(e,d_\perp)|\ge
\sqrt{1-10^{-8}/(2.9409\times10^{-8})}=0.8124.
\]

The analogous optimal-direction gates are 0.4472 for a 20% raw move and 0.5477
for a 30% move.  A large sum of local Bellman scalars without the endpoint norm
does not pass this gate.  The next lawful calculation is the old-projected
\((\langle e,d\rangle,\|d\|^2)\) pair, not another source-amplitude proxy.

## THREE-ATOM-Q211-COST-EXCLUSION (atomic graph node)

**Status: CLOSED as a free reuse claim.**  The r56 strict 48.9385 and terminal
47.3252 three-atom censuses cover only the AQ coefficient push/Gram and the local
single-unit lower reset module.  They contain no q211/K22 reverse rectangular
pass, no CP3/CP4/Star reverse tape, and no anchor-specific \(N\) bank.  Therefore
the generic rank-one K22-costate pullback requiring an \(r^2/n\) rectangular
pass per seam is fresh cost; it cannot be hidden inside the prior ~0.47 local
K32 CSE.  Reopen only on a complete-old/Bellman annihilation proof.

## AQ-OWNED-COMPLETION-ZEROBIAS-M11-ACTION (final in-flight receipt)

Let \(D_3=\operatorname{diag}(c_3)\), \(D_4=\operatorname{diag}(c_4)\),
\(D_a=\operatorname{diag}(a)\), where

\[
a_i=-{5\over96}c_{3i}c_{4i}+{35\over864}c_{3i}^3,
\]

and let \(S=\operatorname{diag}(\sigma)\).  With \(P_{ij}\) and \(R_{ij}\)
the standardized oriented K21/K31 collision arrays defined above, the physical
off-diagonal raw-pair increment is

\[
\Delta M_{11}=\phi(0)S\left[
{RD_3+(RD_3)^T\over24}
+{PD_4+(PD_4)^T\over32}
-{5\{PD_3^2+(PD_3^2)^T\}\over96}
+D_a\rho+\rho D_a\right]S.
\]

For a symmetric off-diagonal raw-M11 Bellman costate \(\Lambda\), put
\(\widetilde\Lambda=S\Lambda S\).  Its exact action is

\[
\boxed{
\begin{aligned}
\langle\Lambda,\Delta M_{11}\rangle=\phi(0)\bigg[&
{\langle\widetilde\Lambda,RD_3\rangle\over12}
+{\langle\widetilde\Lambda,PD_4\rangle\over16}\\
&-{5\langle\widetilde\Lambda,PD_3^2\rangle\over48}
+2\langle\widetilde\Lambda,D_a\rho\rangle\bigg].
\end{aligned}}
\]

Thus the M11-only sentinel requires four \(O(n^2)\) inner products and no new
feature tensor.  It is not the full value receipt until marginal and
M21/M31/M22 raw-costate actions are included.  Before \(S\) scaling,

\[
\|\Delta M_{11}\|_F\le\phi(0)\left[
{\|RD_3\|_F\over12}+{\|PD_4\|_F\over16}
+{5\|PD_3^2\|_F\over48}+2\|D_a\rho\|_F\right].
\]

This agrees with the independently integrated Checkpoint116 action/type audit;
no normalization discrepancy was found.

## 2026-08-31 — streamed owned-L3/L4 completion: literal screen and closure

The exact formal reset completion from (MP.124) was implemented on the held
packed D11 lineage in
`public_d11_owned_edgeworth_stream_v1/estimator_owned.py`.  The default arm
adds only

\[
L_3L_4+\frac16L_3^3,
\]

and the separately identified owned-order-\(r^{-2}\) arm adds

\[
\frac12L_4^2+\frac12L_3^2L_4+\frac1{24}L_3^4.
\]

Every term is written into the raw univariate and bivariate moments before
the unchanged diagonal replacement, validity gates, covariance repair, and
connected-state assembly.  Thus the measured feature is the literal
\(F(z+\Delta z)-F(z)\), not a hand-injected connected plane.  The pair
completion streams one convolution-weight plane, one product plane, and one
four-moment derivative kernel; it retains no new persistent full plane.

`TINY_VERIFICATION.json` establishes source-off byte identity to the parent,
explicit-convolution parity to \(3.59\times10^{-15}\) for the leading arm and
\(3.25\times10^{-15}\) for the owned-\(r^{-2}\) arm, and a positive synthetic
covariance minimum eigenvalue.  The rank-35 physical sentinel then closed both
arms:

| arm | fixed raw change | residual alignment | optimal same-rank gain | utility |
|---|---:|---:|---:|---:|
| \(L_3L_4+L_3^3/6\) | +0.6032% | -0.02619 | 0.0901% | 0.66511 |
| plus owned \(r^{-2}\) | +0.3554% | -0.001522 | 0.000304% | 0.64774 |

The packed parent raw MSE was \(2.67649149\times10^{-8}\) at utility
0.56297.  Both fixed physical arms worsen raw MSE, their optimally rescaled
diagnostics miss the 8% continuation gate by orders of magnitude, and their
incremental bills are respectively 224,605,460,716 and 186,402,451,672.
Therefore no ten-net gate, package, tuning, or stack is lawful.

Atomic disposition:

- `AQ_OWNED_L3L4_L3CUBE_STREAM_SEMANTICS`: **proved/verified**.
- `AQ_OWNED_L3L4_L3CUBE_VALUE`: **closed at rank-35 fail-fast sentinel**.
- `AQ_OWNED_R2_EXPONENTIAL_VALUE`: **closed at the same sentinel**.

## 2026-08-31 — prior L6 containment audit and causal all-pair source candidate

The repository already contained exact target-blind source/action packages
for K33 and the oriented K42/K51 complement.  They compute

\[
r^{(6)}=10\,\mathrm{diag}((W^3)^TK_{33}W^3)
+15\,\mathrm{diag}((W^4)^TK_{42}W^2)
+6\,\mathrm{diag}((W^5)^TK_{51}W),
\]

with the correct transpose orientations encoded by the full source planes.
Those packages stop at a `combined_raw_k6` coordinate and explicitly perform
zero feedback.  Hence they contain the source algebra but not the causal reset
candidate requested here.

The smallest lawful causal reuse is now implemented in
`public_d11_pair_l6_causal_v1/estimator_pair_l6.py`, SHA-256
`b807dbdf68b50f45d7060c54c8c7938ffd446d339f6021545dc34b2554d22dd0`.
It composes the two verified source wrappers on the pinned D4 source
`6aedce54...`, sums the three raw marginal actions exactly once, and inserts

\[
x_i={r_i^{(6)}\over720\sigma_i^6}
\]

through the existing univariate derivative-six coefficient row before the
validity gates.  In the bivariate reset it simultaneously adds
\(r_j^{(6)}/720\) to derivative slot \((0,6)\) and
\(r_i^{(6)}/720\) to slot \((6,0)\).  The ordinary repair, connected assembly,
source rebirth, native tape, and native-J correction then run on the changed
state, so the construction is an autonomous \(F(z+\Delta z)-F(z)\) recurrence.

The typing boundary is essential.  This is the exact marginal-H6 reset
response induced by the verified complete two-coordinate *source-to-marginal*
action.  It is not the full polarized pair-H6 transport: target slots
\((1,5),(2,4),(3,3),(4,2),(5,1)\) would require additional polarized affine
actions and are explicitly omitted.  The earlier coordinate-only measured
increments, 76.18 billion FLOPs for K33 and 151.83 billion for K42/K51, imply
an incremental candidate estimate near 228 billion; the degree-six endpoint
itself is elementwise.  The literal sentinel bill remains authoritative.

Compilation, import, and all three source-hash bindings pass in
`STATIC_RECEIPT.json`.  A deterministic geometry guard delegates the validator's
small `(depth=2,width=4)` fixture to the pinned parent while preserving the
causal branch only at `(16,1024)`; `whest validate` passes.  Exactly one
coefficient-one rank-35 value run is
assigned to the cubic lane; this lane performs no duplicate target run.

Atomic disposition:

- `PAIR_L6_PRIOR_COORDINATE_CONTAINMENT`: **proved** — source/action exists,
  causal reset does not.
- `PAIR_L6_CAUSAL_MARGINAL_RESET_SEMANTICS`: **implemented/static pass**.
- `PAIR_L6_FULL_POLARIZED_TARGET_COLLISIONS`: **open and not claimed**.
- `PAIR_L6_CAUSAL_MARGINAL_VALUE`: **rank-35 sentinel delegated/pending**.

## 2026-08-31 — causal marginal-L6 rank-35 terminal

The sole fixed-coefficient rank-35 value run completed on the validated
candidate hash.  The authoritative receipt is
`public_g19_attribution_v1/PAIR_L6_RANK0035_VALUE_GATE.json`, SHA-256
`d28cf0ab00bb2182ff0ba0adddd7d517450fe0b74af48357a8c52a74834b6b8f`.
The exact pinned-parent and candidate endpoint MSEs including MC noise were
\(3.5173128774\times10^{-8}\) and \(3.5166092648\times10^{-8}\).  After the
same MC-noise subtraction,

\[
2.6757656009\times10^{-8}
\longrightarrow
2.6750619883\times10^{-8}.
\]

Thus the physical gain is only
\(7.0361260\times10^{-12}=0.02629575\%\), versus the preregistered 8% gate.
The new direction has correct sign but weak alignment (Pearson 0.1447442,
cosine 0.1481190) and RMS \(1.26935\times10^{-7}\).  Its bill is also
material: 1,548,003,718,519 versus 1,319,997,539,703 for the parent, an exact
increment of 228,006,178,816 or 0.1036852 of the phase budget.  The runtime
association receipt confirms 15 three-sector actions, 15 univariate L6
endpoints, and 14 bivariate marginal endpoints.

The 0.0263% gain misses the continuation bar by more than a factor 300, so no
ten-network expansion, fit, package, or stack is permitted.  The exact pinned
parent is also not byte-identical to the stored deployed-rank35 output, which
would independently require a lineage bridge before packaging.

Independent algebra audit in `LANE_h4_reverse_adjoint.md` section 94 passes
the factors 10/15/6, raw `/720` normalization, endpoint orientation, single
\(\sigma^{-6}\), diagonal replacement, and causal association.  It also
proves the advertised boundary is strict: a target mixed source with
\(K33_{12}=1\) can have zero marginal sixth cumulant (so this candidate writes
zero) while its centered pair response changes by

\[
\Delta M_{22}={\phi(0)^2\over9\sigma_1\sigma_2}\ne0.
\]

Therefore the narrow implementation is mathematically sound, but the tested
marginal-H6 mechanism is terminally value-dead.  Full polarized target H6,
one-coordinate K6, support-at-least-three sources, and higher exponential
grade-four products remain logically distinct; none may be credited to or
rescued by this result.

Final atomic disposition:

- `PAIR_L6_CAUSAL_MARGINAL_VALUE`: **closed, 0.0263% < 8%**.
- `PAIR_L6_CAUSAL_MARGINAL_COST`: **closed, +0.103685 budget**.
- `PAIR_L6_CAUSAL_MARGINAL_SEMANTICS`: **independently audited pass**.
- `PAIR_L6_TEN_NET_AND_PACKAGE`: **not authorized**.

## 2026-08-31 — full-688 oracle-correction fixed-head compression audit

The cached-only audit is frozen at
`public_full688_oracle_correction_fixed_head_cv_v1/`.  Its authoritative
`RESULT.json` has SHA-256
`7e2c607ed9ac94be9e7099f8e9b7ab56c415c8ffbf7222aa8e7b6c3cb9716fa1`;
the spectra artifact `FIXED_HEAD_SPECTRA.npz` has SHA-256
`b1ae8981f628f4b2b26497eef8b395ddd94a9491eedd9162edc11deb61792cba`.
It used only the frozen full-688 truth-minus-host cache and frozen OOF-J
trajectories.  The exact alpha=.75 q-refit work arrays were RAM-only and are
not claimed.  Instead the measured suffix oracle is the exact OOF-J identity

\[
q_l=r_l-\phi_l(r_{l-1}W_l),\qquad
s_l=\phi_l(s_{l-1}W_l)+q_l,\quad l=10,\ldots,15,
\]

with `s_9=0`.  It reduces OOF-J endpoint MSE from
`3.2334958229470984e-8` to `6.238892067519586e-9`, within 0.166% of the
separately measured exact-alpha oracle `6.249252614225924e-9`.

For each of the four frozen network folds, an uncentered spatial PCA basis
was fit on the other three folds only.  The reported held energy is

\[
{\sum_f\|X_{\mathrm{hold},f}V_{\mathrm{train},f,1:k}\|_F^2
 \over \sum_f\|X_{\mathrm{hold},f}\|_F^2}.
\]

Both the six-layer q correction and the rolled correction state require
`k=991/1024` to retain the campaign threshold `96.7192%`: q rises from
`0.9667712557` at k=990 to `0.9677588528` at k=991, and state rises from
`0.9670231920` to `0.9679757818`.  Endpoint-only full-data SVDs look
misleadingly smaller (526--527 dimensions), because a held training fold has
rank only 490--541.  At the common learnable rank cap 490, held endpoint
energy is only 48.0404% for the full residual and 47.8685% for the suffix
correction.  No fold reaches raw MSE below `1e-8` within its train rank.  The
apparent k=707 full-residual and k=876 suffix thresholds require arbitrary
held-null-space completion and are explicitly nonlearnable diagnostics.

This closes a fixed global output basis, not a field-conditioned full-width
head.  In particular, four backward *vector* channels are not rank four;
the frozen bidirectional-adjoint design remains algebraically possible but
needs an architecture-specific induced-response receipt.  Its cached
fixed-mode proxy is already dead: even eight Jacobian modes retain only
7.089% ordinary and 2.231% hard energy.  The executed learned backward
adjoint (`3.2337660091e-8`) and signed-pair V4R2 (`3.0243436125e-8`) remain
closed prior art.

Atomic disposition:

- `ORACLE_FIXED_SPATIAL_HEAD_96P7192`: **no-go, 991/1024 directions**.
- `ORACLE_ENDPOINT_RAW_LT_1E8_WITH_HELD_TRAIN_BASIS`: **no-go**.
- `ORACLE_LOW_DIMENSIONAL_DEPLOYABLE_BASIS`: **none emitted**.
- `FIELD_CONDITIONED_FULL_WIDTH_HEAD`: **not tested by this no-go**; the next
  lawful object is the frozen 82-field local/signed/absolute/squared message
  head, analyzed as a shared pointwise conditional map rather than a spatial
  PCA basis.

## 2026-08-31 — cached 82-field geometry and oracle-information audit

The cached-only audit is frozen at
`public_q24_82field_capacity_audit_v1/`.  `RESULT.json` has SHA-256
`d94bab7c2ecbbfe81056ff5b00f05057ba82e69cd83f9bae9049d0f418bf1644`;
the program has SHA-256
`4bac6567570b04c195aac5f8d00191c85b37a98ff2b99fc7dee9adf7e1273271`.
It invokes no estimator and uses the complete target-blind signed, absolute,
and square full-688 field caches.  The numeric geometry covers 80/82 candidate
inputs: the alpha=.75 base-state and base-q local coordinates were RAM-only,
so the local block here is correctly called raw20 rather than local22.

In the featurewise-standardized target-free gauge, the 80 fields have
numerical rank 69, participation ratio 7.724, and entropy effective rank
13.849.  Absolute and squared-weight messages are almost the same field:
17/20 canonical correlations exceed 0.9.  Their conditional residual trace
fractions after the other blocks are 8.45% and 8.70%.  Signed messages are
distinct (maximum canonical correlation only about 0.616 against either
even block) and retain 32.0% conditional trace; local raw retains 42.2%.

The separate response diagnostic uses the already cached OOF-J suffix oracle
q and the fixed q-head folds.  Its zero-source MSE is
`5.796595918345804e-9`.  Four-fold held linear fractions of q energy explained
are 0.31819% by local raw, 0.12396% by signed alone, 0.40736% by local+signed,
and only 0.46348% by all 80 fields.  Thus absolute+square add merely 0.05612
percentage points beyond local+signed in this readout.  Symmetric Shapley
shares are 0.30198% local, 0.10707% signed, 0.01856% absolute, and 0.03587%
square.  Even the deliberately optimistic response-leaky full-corpus linear
projection explains only 0.52221%.

This is a strong low-linear-information warning, not an upper bound on the
frozen 256-wide nonlinear head.  The sole preregistered OOF arm remains the
only promotion authority.  Its post-run decomposition is frozen as one joint
model: target-free block Schur geometry plus activation/Jacobian Grams, then
straight-line Aumann-Shapley fields whose four contributions sum exactly to
the joint prediction minus its zero-baseline value.  No factorial retraining
or post-hoc arm selection is licensed.

Atomic disposition:

- `Q82_FIXED_SPATIAL_PCA_OBSTRUCTION`: **does not apply** — fields emit
  network-specific full-width outputs.
- `Q82_ABS_SQUARE_TARGET_FREE_DISTINCTNESS`: **weak; highly redundant**.
- `Q82_80FIELD_HELD_LINEAR_ORACLE_INFORMATION`: **0.46348% of q energy**.
- `Q82_NONLINEAR_VALUE`: **owned only by the sole running OOF arm**.

## 2026-08-31 — frozen full-width successor field ranking

`public_q24_82field_capacity_audit_v1/SUCCESSOR_FIELD_DESIGNS.md`, SHA-256
`12378be78329f25172ab76218c4a4a3821255a1e9af489b625dcaa9c57de45a8`,
freezes four target-blind designs without reading a target or authorizing a
run.  The key information theorem is elementary but binding: transported
first moments do not determine nonlinear-before-transport moments.  Equal
edge weights on `(a,-a)` and `(0,0)` give the same first moment and different
second moments; the two-coordinate version proves `P(xy)` is not determined
by `Px,Py`.

The ranking is:

1. Push the exact owned L3 and L3² ReLU mean/variance actions *after forming
   them at the source*, through W and W².  Four skinny fields across six seams
   cost `0.0234375` GEMM-equivalent.  Same-site actions are deterministic from
   local22 and add no information; multiplication before transport is the
   nonduplicate object.
2. Add the 20-channel W² conditional dispersion
   `E_{W²}[x²]-E_{W²}[x]²`, the exact heterogeneity missing from the current
   first-moment bank.  Cost `0.1171875` GEMM-equivalent and direct relevance
   to the broad variance channel.
3. Seed a backward scan with intrinsic L3 mean/variance actions rather than a
   free learned reverse state.  Two suffix reverse columns cost about
   `0.009765625`; an endpoint-aware loss is required to falsify it cleanly.
4. Recur the even absolute/square message banks across paths.  This is exact
   remote information but has the weakest prior given present redundancy and
   old W-power results; both full recurrences cost `0.5078125`.

Every design has a target-free conditional-rank gate and a single fixed-fold
sentinel.  Continue only for correct-sign q movement and at least 8% recovery
of the 82-parent-to-exact-q-oracle endpoint gap; passing authorizes the other
three folds with no retuning.  The local cubic scalar alone, post-transport
pointwise polynomials, and further fixed spatial modes are explicitly closed
as information successors.

## 2026-08-31 — direct cubic-transpose overlap audit

The static audit is persisted in
`public_q24_82field_capacity_audit_v1/DIRECT_TRANSPOSE_OVERLAP_AUDIT.md`,
SHA-256
`911c875b0ee33c4c80b7224abc435e7e9ca4e26eaa781d7db8903cdd18c41f77`.
No target, derived cache, or training run was opened.

Writing the deployed mean tangent as

\[
T_lx=D_{\phi_l}W_l^Tx,\qquad T_l^*y=W_lD_{\phi_l}y,
\]

the frozen direct sentinel is exactly

\[
b_l=c_l+T_{l+1}^*b_{l+1},\quad b_{15}=c_{15},qquad
c_l=-{\sigma_l\alpha_l\varphi_l\gamma_{3,l}\over6}.
\]

Thus it is a literal identity with the intrinsic L3-seeded **mean** backward
completion on layers 5--15, not a neighboring idea.  The local c coordinate
adds no physical-output information because
`c/(sigma*rms)=-alpha*pdf*gamma3/(6*rms)` is already a local22 function; the
new object is the future adjoint word b (shown in two scale gauges).  It does
not test a variance backward branch, whose reset tangent remains untyped.

The owned four-field forward bundle has zero shared exposed transported
coordinates with the direct arm.  Its mean fields are the causal words
`T_l c_(l-1)` and `T_l c33_(l-1)`; its variance fields are incoming W² pushes
of `dv3` and `dv33`.  Direct instead exposes future adjoint words and contains
neither L3² nor variance.  Sharing c as a primitive does not collapse these
opposite causal word modules generically.

The direct loss is local delta-q with zero endpoint weight.  Since exact q at
layer l is prefix-measurable, future weights in b have no causal local-q
information conditional on the complete prefix; they can help only through
missing-prefix proxies or network-level correlations.  A failure therefore
forbids another mean-L3 backward spelling under the same supervision.

Successor decision:

- if the direct arm's generic exact start-5 q-oracle gate is not below
  `1e-8`, close the entire same-rollout q-head lane;
- if that oracle gate passes but the direct source-on sentinel fails, the one
  next arm is the four-field owned L3/L3² action-before-transport bundle,
  paired against identical zeroed features.  It is causal, includes variance,
  repairs a proved product-moment information kernel, and costs only
  `0.0234375` GEMM-equivalent across six seams.

Atomic disposition:

- `DIRECT_EQ_INTRINSIC_L3_MEAN_BACKWARD`: **proved exact**.
- `DIRECT_TESTS_VARIANCE_BACKWARD`: **false**.
- `DIRECT_FORWARD_FOUR_FIELD_OVERLAP`: **source primitive only; zero exposed
  transported-coordinate overlap**.
- `POST_DIRECT_FAIL_SINGLE_SUCCESSOR`: **owned L3/L3² forward four-field
  bundle, only if the generic q-oracle gate passes**.

## 2026-08-31 — four-field product-before-transport generator

The sole conditional successor is now implemented without opening a target,
native row, derived cache, estimator, or training path.  The in-memory source
is
`public_q24_owned_l3_l3square_forward_fields_v1/feature_generator.py`,
SHA-256
`5f4ed4a777f5ab80141bc2e3d1ced73b973f7a60b9f694dd330ca461b545c086`.
Its static receipt is `STATIC_VERIFICATION.json`, SHA-256
`905ccaf22a1444aede1df641155a3a36fadd8690958218a36566f64d095f0d94`.

On each target layer 10--15, the generator forms the exact old-owned source
actions `(dmu3,dv3,dmu33,dv33)` at the preceding layer and only then applies
`D_gate W^T` to the mean pair and `(W^2)^T` to the incoming diagonal-variance
pair.  The output order is frozen as
`(mean_l3,prevariance_l3,mean_l3_squared,prevariance_l3_squared)`, with mean
divided by target sigma and variance by target sigma squared.  It assigns no
unproved variance gate and uses no corpus normalization; any later feature
standardization must remain fit-fold-only.

Tiny independent checks pass:

- explicit float64 formula replay differs from the float32 kernel by at most
  `2.1004418537806657e-8`;
- resident-W² and one-square-scratch modes are byte-identical;
- independent source/target layer permutations commute with the generator to
  `7.450580596923828e-9`; and
- an exact two-node witness keeps `W^T alpha` and `W^T gamma3` identical while
  changing the transported mean-L3 product field by `-1/3`.

The exact width-1024, six-seam dense-product bill with resident W² is
`50,307,072` dot FLOPs, exactly `24/1024=0.0234375` dense-GEMM equivalents.
All local arithmetic raises this to `50,558,976` operations.  The explicit
module-owned peak is `163,840` bytes.  Without resident W², the lawful reused
square scratch raises the totals to `56,850,432` operations and `4,358,144`
bytes; the emitted bytes remain identical.

`IMPLEMENTATION_FREEZE.json`, SHA-256
`ac8be2bb247e505f37919823e988c8622058bf45385bd0faa5cdc223610f9666`,
binds the arm as a single four-field source-on versus identical-zero control.
It has no execution authority and is reachable only if the direct arm's exact
start-5 q oracle is strictly below `1e-8` **and** the direct source-on sentinel
fails its frozen continuation gate.  Oracle failure closes the same-rollout
q-head lane; direct success continues only the direct preregistration.

Atomic disposition:

- `OWNED_L3_L3SQUARE_PRODUCT_TRANSPORT_GENERATOR`: **static PASS**.
- `PRODUCT_TRANSPORT_PERMUTATION_EQUIVARIANCE`: **static PASS**.
- `RESIDENT_W2_INCREMENTAL_COST`: **0.0234375 GEMM-equivalent; 163,840-byte
  explicit peak**.
- `FOUR_FIELD_EXECUTION_AUTHORITY`: **false; doubly interlocked conditional
  successor only**.

## 2026-08-31 — frozen fold-0 owned-L3/L3² product-before-transport sentinel

The sole successor interlock is now satisfied: the direct start-5 q oracle
passed, while the direct source-on arm failed both paired endpoint and local-q
sign gates.  Two independent terminal audits bind the direct result before any
new authority is usable.

The lawful successor is **not** the abandoned 26-input draft.  It appends the
four jointly frozen fields to the complete sealed 82-input parent:

- inputs `22 local + 20 signed + 20 |W| + 20 W² + 4 new = 86`;
- hidden trunk `256 x 3`, six layer-specific heads on L10--15;
- `155,398` trainable parameters;
- the zero control keeps the full parent82 input and zeros only new4 after
  fit-516 normalization;
- both arms share initialization, row order, 256-node samples, q RMS,
  AdamW, and the parent's exact `13*129` cosine schedule.

The frozen value denominator is the saved fold-0 parent, not the direct
start-5 oracle and not the paired zero arm:

```text
parent82 = 3.0023717906766396e-8
oracle   = 6.1455295488600225e-9
gate     = parent82 - .08*(parent82-oracle)
         = 2.8113462838133887e-8.
```

Passing additionally requires source-on to beat source-zero strictly at both
endpoint and local q.  The isolated zero-to-on recovery is reported only as a
diagnostic, preserving the preregistered SHA12378 denominator.

Before any target claim, the released runner rehashes every signed/abs/squared
cache row, forms parent60 and new4 in memory, and measures the Schur rank of
new4 conditional on the **full** parent82 input (constant + exact target-blind
local22 + parent60), on the frozen 32-rank/256-node sample.  Every layer must
have nonzero conditional rank.  Rank-zero or nonfinite fields write an
exclusive fsynced `PRETARGET_TERMINAL_CLOSURE.json`; they cannot be retried.
No derived feature cache is written.

Stable bundle:

- directory: `public_full688_d11_owned_l3_l3square_q_sentinel_v1/`
- `run_fold0.py`: `a6c891cd19b62e1d48280308d4607fefd2a9f8d6d2eaf4a0277dc230c5af6cc5`
- `feature_generator.py`: `5c85616fdd102986cae4b97309d1ddedee5564d3c490b0ebf198fe7ac56fbda7`
- `STATIC_VERIFICATION.json`: `c0a3c18d81872e6c216bb5eaa4cc36deff716ddb5b1fdcf5c288d292059e779a`
- `PREREGISTRATION.json`: `b3d7197b0410f93096fa32d8c018f0e4037bc710e1d5de5da43f84e220b9c624`
- `FOLD0_RUN_CLAIM_CONTRACT.json`: `09cb93a6381b3ecc80196529f5fca055be0bf148c8ce6448226da1946f2de11a`
- `FOLD0_EXECUTION_RELEASE.json`: `16715386a00d64347a20d1d4e85ecf5410d2c4d86abe79179c6dab78d3ec622b`

Target/cache-free static verification passed.  Preflight is deliberately HOLD
until both hash-bound H4 and cubic runner audits land.  At freeze: target
values decoded `0`, cache rows opened `0`, training/scoring runs `0`, derived
caches `0`, estimator calls `0`.

## 2026-08-31 — terminal fold-0 verdict for the owned-L3/L3² four-field bundle

Both required independent audits landed PASS before execution:

- H4 runner audit `INDEPENDENT_H4_RUNNER_AUDIT.json`, SHA-256
  `a5010c9595885ff8a944464c0acb16ea82b2c8f2af8aebfc0c6ad34cc138391a`;
- cubic runner audit `INDEPENDENT_CUBIC_RUNNER_AUDIT.json`, SHA-256
  `341785c61a6746d17760d3fb5e1969552420fe3c4db0f2d2beec313643f7e739`.

The frozen runner was launched exactly once.  Its durable claim has SHA-256
`09d6d1b0bc0b4e0271eb06b682d904998a6aab6fcfc2757b930f10b384902901`.
The pretarget full-parent82 Schur diagnostic had minimum rank `4`, all `1,376`
cache rows verified, and the independent claim audit passed with SHA-256
`624e9293ed60a3d45223bbad4052949fd8454e98de7587b027c5d9c5971f6730`.

The run completed with terminal status
`FAIL_FOLD0_FOUR_FIELD_SENTINEL__TERMINAL_NO_RETRY` and decision
`CLOSE_FOUR_FIELD_BUNDLE__DO_NOT_RUN_FOLDS_1_TO_3`.  The authoritative result
is `FOLD0_RESULT.json`, SHA-256
`d84c5831d3392961d04fcf23323794ca75858190924672aa21c4e496731b98f5`;
the paired model artifact has SHA-256
`1d464e4fc445a65df20b889e0ce652c1eafce26288dd10d802dbc51a304862dc`.

The fields have a real, correctly signed effect:

- source-on endpoint MSE `3.0001380433987615e-8` versus source-zero
  `3.003137412896003e-8`;
- source-on local-q MSE `5.273407806729197e-9` versus source-zero
  `5.2765354435969155e-9`;
- source-on beats source-zero on `0.6046511627906976` of networks, with paired
  endpoint `z=3.6170942617330617`.

But the effect is decisively too small.  Relative to the frozen parent82 and
exact-q oracle, source-on recovers only
`0.0009354760270741128 = 0.0935476%` of the available gap, versus the binding
`8%` gate.  Its endpoint improvement over parent82 is only
`2.2337472778781588e-11`; the paired endpoint improvement over source-zero is
`2.9993694972416666e-11`.  Thus the four fields are identifiable and
directionally useful, but explain essentially none of the demonstrated q
oracle gap.

Atomic disposition:

- `FOUR_FIELD_FULL82_SCHUR_PRETARGET_GATE`: **PASS; minimum rank 4**.
- `FOUR_FIELD_PAIRED_SIGN_CONTROL`: **PASS at endpoint and local q**.
- `FOUR_FIELD_FIXED_PARENT82_GAP_RECOVERY`: **FAIL; 0.0935476% versus 8%**.
- `OWNED_L3_L3SQUARE_PRODUCT_TRANSPORT_VALUE`: **terminal NOGO**.
- `FOUR_FIELD_RETRY_OR_FOLDS_1_TO_3`: **forbidden by the one-shot close law**.

## 2026-08-31 — corrected post-q V34 physical-state sentinel v2 freeze

The exact start-5 q oracle passed, but its direct transpose provider and the
owned L3/L3² four-field provider both closed.  This makes the old V34 v1
activation premise (q-oracle failure) false.  A new v2 freeze is therefore the
only lawful spelling of the physical-state successor; v1 remains unreachable.

The v2 provider interface adds four fields beyond native20:

```text
pre_k4/sigma^4,
((W^2).T @ (raw2_univariate-diag(M11_pair_before_overwrite)))/sigma_target^2,
diag(cp_pre_k21)/sigma^3,
cp4_pre_provider_diagonal/sigma^4.
```

Only the last three provider-split directions are new; total k4 is a
zero-product exposure absent from native20.  The CP3 and CP4 identities are
literal reads of the pinned D11 source.  The pair/univariate variance
coordinate is captured before the diagonal overwrite.  It is a consistency
diagnostic, not a diagonal-covariance sufficiency theorem.  At the exact first
reset its discrepancy is identically zero, so the layer-1 variance push is
elided.

The frozen shared head is `24 -> 32 ReLU -> 3`, exactly 899 parameters, and
predicts standardized physical preactivation gaps `(dV,d3,d4)` on layers
1--15.  The nine fit networks are
`100,150,200,250,300,400,500,600,650`; rank35 is the sole held network.  Its
training loss contains only the three physical state gaps and has no q,
endpoint, residual, identity, observer-selection, or board target.  The full
fit schedule, normalization, seed, final epoch, and rank35 LOO coefficients
are frozen in the design before any row is opened.

The exact component-pure EDG2 sources fuse into

```text
r_0 = 0
r_l = phi_l*(r_(l-1)@W_l) + betaV*sV_l + beta3*s3_l + beta4*s4_l,
l=1,...,15.
```

With exact state gaps this is exactly the frozen rank35 V+K3+K4 perfect-state
column.  The first signed-W transport is zero and is also elided.  The exact
incremental shell is therefore fourteen squared-W actions plus fourteen
signed-W actions: 58,691,584 dense-action FLOPs.  Including the head and a
320-operation-per-node physical reserve gives 90,148,864 arithmetic
operations, `0.04199934355153884` dense-GEMM equivalents, 256 KiB streaming
workspace, and 3,596 parameter bytes.  Recomputing CP providers, forming a
new W², or storing an all-layer tape voids this bill.

The sole held sentinel requires source-zero byte parity, finite values,
combined response cosine at least `0.983459`, combined response RMS error at
most `3.79871e-6`, rank35 noise-corrected MSE strictly below `1e-8`,
nonnegative component dots, and source-on strictly better than source-zero.
Any failure closes the exact split/head/schedule/window spelling without a
retry.  A pass requests but does not grant the other nine LOO evaluations.

Stable static bundle:

- directory `public_v34_physical_state_response_sentinel_v2/`;
- `DESIGN.md` SHA-256
  `6db96ef9c3fd59b87dddab46f2466102fa16a425dd27afa6b8d819aa4c95379a`;
- `provider_fields.py` SHA-256
  `17545cc2741838674ba87a870905ccf55902841cd121c47e96fb1651d3dbd3c6`;
- `STATIC_VERIFICATION.json` SHA-256
  `e3c16e43d71aef6a62bc23d063a99c10c8f6d0d221792c0c38065947e3154f77`;
- `DESIGN_FREEZE.json` SHA-256
  `5632879f9a741ac72d701af182c98aa607060d8ef58dd2fa3c80f7e089d033d5`.

The synthetic algebra verification passed with zero target/cache/estimator
access.  The current state is deliberately **static only**: two independent
design audits must land before a separate nine-fit asset manifest/release or
any runner is allowed; rank35 remains sealed until the fit model and
normalization hashes are frozen.

Atomic disposition:

- `V34_V1_INTERLOCK`: **unreachable; do not execute**.
- `V34_V2_PROVIDER_ALGEBRA`: **static PASS**.
- `V34_V2_VALUE`: **unknown; one held rank35 falsifier**.
- `V34_V2_EXECUTION_AUTHORITY`: **false pending independent audits and later
  releases**.

## 2026-08-31 — V34 V2R1 fresh-held data-firewall repair

Independent v2 audits passed the provider algebra/resource shell but closed
rank35 as confirmatory evidence: its identity, exact oracle, response column,
and LOO coefficients were public before the freeze.  Stable v2 remains an
immutable descriptive mechanism check.  V2R1 changes only the data firewall.

All 688 observer-fit rows are development-exposed.  A standalone target-blind
minimum-hash over the 88 observer-selection rows selected:

```text
global hash rank       757
split/store role rank  57 / 45
mlp_id                 887
selector digest        005767ad9eb0d22cd3946a07d592664cbd15b16ed02b6939d500eade64abc9ab
row manifest           2e41a5ded1d111e158d54d2a5bb1c1f7ccb47a64bcab633187ff2dda76acc9bb
weights file           57b7dcf3624ec94342ab5ad6de8d750df93714ab10112cd6cf051026a583d5a6
raw weights            a6605c447a51d50907be1bb6a2b2358a2d781cd832ddec43fc71ec1702860d74.
```

The official `all_layer_means` target remains unopened.  Selection decoded no
target, feature, state, or label array.  One post-selection opaque weights-file
integrity hash read is disclosed; no weight array was deserialized or used
semantically.  There is no replacement held row.

The nine fit ranks remain exactly
`100,150,200,250,300,400,500,600,650`; rank35 is excluded.  The fitted model
and normalization must freeze before held-label generation.  Two isolated MC
streams then use exactly P8M each, split into 16 consecutive 500k blocks per
stream (32 independent blocks total).  They retain every block's raw moment
summary and seal physical preactivation moments and post means separately
from the target-blind D11 provider capture.

Endpoint raw uncertainty uses the exact 32-block U-statistic, delete-block
pseudovalues, and a frozen one-sided four-SE upper bound.  Paired source gain
uses its per-block linear statistic and the same four-SE law.  The nonlinear
EDG2 response compiler uses a two-sample delete-one-block jackknife plus
replica-ball lower cosine/upper RMS/component-sign guards.  Negative or
boundary-crossing cross estimates and unresolved intervals terminate as
`MC_RESOLUTION_INSUFFICIENT`; no extra samples or new held row are allowed.

V2R1 freezes `OFFICIAL_TARGET_LAST_RAW`.  Only after model, normalization,
provider capture, both MC packages, compiled responses, and every hash freeze
may an exclusive one-use reader open rank757's official endpoint target.  It
is used only for exact source-on raw MSE below `1e-8` and source-on versus zero;
MC remains the physical-oracle, cosine, RMS, component, and resolution guard.
A rank757 pass requests but does not grant a fresh panel, package, board, or
submission action.

Current candidate bindings:

- `public_v34_physical_state_response_sentinel_v2r1/RELEASE_REPAIR.md`
  SHA-256 `c7e4aafa64cdbdb59cb129a453e8ab5c5e06e4539b5e5e1a0e98ce5f3de2c811`;
- `HELD_SELECTION_RECEIPT.json` SHA-256
  `5f6f4c43bea394bdd441312515ac38efc999bdb073e3e0caa0c36a68faf5b7d5`;
- `V2R1_FREEZE.json` SHA-256
  `f816c40020c778cbf4c4b1255f806a493beec30e9d74c2346190a91404066657`;
- independent protocol derivation SHA-256
  `18682a525ec444828518a4a5209f2c0a56aebe3f5399601997deb1f550288114`.

No MC generator, provider capture, fit, held claim, official target reader, or
meter is authorized.  Before any data action, the exact two seed commitments,
generator/capture/manifest builders, result U-stat/jackknife code, fit-only
release, held claims, official-last reader, and total modified-estimator meter
must freeze and pass independent audits.

Atomic disposition:

- `V34_RANK35_CONFIRMATORY_STATUS`: **closed; descriptive development only**.
- `V34_V2R1_HELD_SELECTION`: **target-blind PASS, observer-selection rank757**.
- `V34_V2R1_MC_AND_OFFICIAL_TARGET_PROTOCOL`: **static solved requirements**.
- `V34_V2R1_DATA_AND_EXECUTION_AUTHORITY`: **false pending implementation and
  independent audits**.

### Append-only V2R1 package-evidence panel binding

The single rank757 transaction remains a fail-fast stage, not package
evidence.  A winner-free selector spec and complete 88-row receipt freeze the
first four observer-selection hashes as the only lawful ordered panel:

```text
panel 0: global757 / role45 / mlp887
panel 1: global746 / role34 / mlp288
panel 2: global780 / role68 / mlp780
panel 3: global759 / role47 / mlp449.
```

Failure or MC inconclusiveness on any row is terminal.  A row pass requests
only the next predeclared row; it does not grant it automatically.  Skipping,
replacement, reordering, adaptive sampling, or selecting another network is
forbidden.  Only an all-four pass with individual and aggregate physical
oracle/response/official-target-last gates can serve as package evidence, and
even that requests package review rather than granting integration, board, or
submission authority.

Append-only bindings:

- confirmatory data protocol SHA-256
  `f978719aacfde1eeb94b94513cf020a806dd5f5afb1c3aed5df43569769b37bd`;
- winner-free selector spec SHA-256
  `589e3d8c4606cc4b57081458b9197370e77753f2a1fb9d1b247d1eac2de6c0ca`;
- four-row selection receipt SHA-256
  `22be0d94054cb026375709ea78d9f2b9586325405876ebed5126b681a7e2770f`;
- `V2R1_PANEL_BINDING.json` SHA-256
  `16c4eb19b095e52ea852319a2fb78e72f8431c5addd3b3025a29f342ddbbc193`.

The immutable stage-1 hashes remain `c7e4aafa...`, `5f6f4c43...`, and
`f816c400...`; the panel binding does not rewrite them.  No row's official
target or physical MC label has been opened or generated.

## 2026-08-31 — V34 V2R1 production implementation freeze

The static implementation suite is now complete and byte-frozen.  Its sole
production entrypoint is `orchestrator.py`; helper modules cannot create
authority.  Every stage first authenticates an externally issued one-use
release, rehashes the complete implementation freeze and predecessor chain,
then durably claims before any semantic payload read.  The ordered transaction
is fit -> held generation -> held semantic -> official-target-last; only four
immutable row PASS receipts permit the panel aggregate claim.

The critical implementation properties are now literal:

- the captured source-zero is the corrected public
  `Estimator.predict -> _predict_corrected` output, with a one-pass native-tape
  latch and a fresh public byte-parity replay;
- the deploy path transports one fused V/K3/K4 response state, exactly 14
  signed-W actions, while three split recurrences are off-card component-sign
  evidence only;
- the literal metered fused endpoint, not its NumPy surrogate, drives all MC
  gates and the immutable response payload;
- the MC seed preimage is exactly
  `domain|mlp_seed|raw_weights_sha256|A/B`, with eight frozen seed receipts;
- each stream retains 16 consecutive 500k block summaries; the binding oracle
  gate is the independent A/B cross residual under the two-sample delete-one
  jackknife, conjunctive with both conventional stream corrections; nonlinear
  block-local EDG2 U-statistics are diagnostic only;
- the official target bundle and manifest are not opened or hashed preclaim;
  the claimed same-PID scorer rehashes the response/model/MC/resource chain
  before its one target read;
- a durable resource receipt binds the complete modified estimator, exact
  fused recurrence, FLOPs below `2^41`, RSS below 8 GiB, and wall below 120 s;
- panel response gates have an exact nonempty keyset.  The four rows are fixed
  contractual test cases: all-four individual PASS is binding.  Within-row
  jackknife aggregation is only conditional MC uncertainty for the fixed
  panel mean, never population or unseen-network inference.

Frozen artifacts:

- `IMPLEMENTATION_MANIFEST_V2R1.json` SHA-256
  `efb093d59000f76be614c9268754fc589c09e243ac891b1902c43a633f332a22`;
- `IMPLEMENTATION_FREEZE_V2R1.json` SHA-256
  `5a0d937711794ad7d53e1449faf54f67b49d21c110a57b2533fece39ae82f6eb`;
- `IMPLEMENTATION_STATIC_VERIFICATION_V2R1.json` SHA-256
  `c395adc76b149defc04b6569ea494b0f2ddf9b83757e5a986d9e1fdba3d0c870`;
- corrected seed-law receipt `MC_BLOCK_REFERENCE_STATIC_VERIFICATION_R1.json`
  SHA-256
  `8db856df076d897f2373234ca137627aedc920c8cd8dbb0dcaa105bf08348038`;
- fixed-panel interpretation
  `PANEL_FIXED_CONTRACTUAL_INTERPRETATION_V2R1.json` SHA-256
  `b9c0e0f49ae63e98652d48a7146943c70b2e9ed1f2a5f02be79e28d51067526d`.

Static verification is green: Python compilation passed; 15 block-statistic,
8 held-scorer contract, 6 MC reconstruction/seed, and 10 whole-plumbing
fixtures passed (39 total), including public-parent correction, fused-call
graph, JSON persistence, target-last anti-replay ordering, exact panel gate
keyset, symlink/O_EXCL corruption, and deep manifest/freeze source rehash.

No production estimator, weight array, target, fit label, cache, MC generator,
trainer, or scorer was opened or run while constructing or testing this
suite.  Execution remains **HOLD** until independent Cubic, Direct, and H4
audits bind these exact bytes and a separate fit-only release is issued.

Atomic disposition:

- `V34_V2R1_IMPLEMENTATION`: **static PASS / byte-frozen**.
- `V34_V2R1_CORRECTED_PARENT_AND_FUSED_DEPLOY`: **static PASS**.
- `V34_V2R1_SCORER_FIREWALL_AND_PANEL_AGGREGATE`: **static PASS**.
- `V34_V2R1_DATA_EXECUTION_OR_PACKAGE_AUTHORITY`: **false; independent audits
  and explicit staged releases still required**.

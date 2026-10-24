# Cubic vertex / NCQC observability

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## 1. NCQC cubic provider and exact-state obstruction

Represent the independent quadratic-chaos core by symmetric matrices
\(A_i\in\operatorname{Sym}_r\) and

\[
Q_i=\zeta^\top A_i\zeta-\operatorname{tr}A_i.
\]

Then

\[
C_{ij}=2\operatorname{tr}(A_iA_j),\qquad
H^{(3)}_{ijk}=8\operatorname{tr}(A_iA_jA_k).
\]

For a dense transition, put \(B_a=\sum_iW_{ia}A_i\).  The repeated cubic
slice is

\[
K21_{ab}=8\operatorname{tr}(B_a^2B_b),
\]

which has an exact \(O(n^2)\)-state, \(O(n^3)\)-action implementation.  A
lawful nonlinear update has the exact degree-two projection

\[
A^+=E[\Phi((\mu+Q_B)/\sigma)]B
+2B\,E[\zeta\zeta^\top\phi((\mu+Q_B)/\sigma)/\sigma]B.
\]

This is a genuinely noncommutative cubic vertex, unlike CP3/star.  However,
an arbitrary exact cubic tensor cannot have \(O(n^2)\) state: access to
\(T(x,x,z)\) for arbitrary future dense \(x,z\), followed by polarization,
recovers all of \(\operatorname{Sym}^3(\mathbb R^n)\), which is
\(\Theta(n^3)\)-dimensional.  Any claimed exact \(O(n^2)\) generic provider
therefore needs additional algebraic structure.

The inverse value budget also gives a hard fidelity gate: even with exact
quartic information, the cubic partial correlation must be at least 0.9107;
cubic alone cannot cross the campaign target, and at least 27.7% of the
available quartic value is mandatory.

## 2. Static identifiability of the NCQC core

For \(\Phi(A)=(C,K21)\), the differential is

\[
DC[H]_{ij}=2\operatorname{tr}(H_iA_j+A_iH_j),
\]

\[
DK[H]_{ij}=8\operatorname{tr}((A_iH_i+H_iA_i)A_j+A_i^2H_j).
\]

The gauge directions are \(H_i=[\Omega,A_i]\).  At \(n=1024\), the
quotient parameter dimension and observation dimension are

\[
p_r=1024\,\frac{r(r+1)}2-\frac{r(r-1)}2,
\qquad q=\frac{1024\cdot1025}{2}+1024^2=1,573,376.
\]

Thus \(r=54\) has \(p_{54}=1,519,209\), while \(r=55\) has
\(p_{55}=1,575,475\).  At every \(r=55\) point the differential kernel has
dimension at least

\[
f=p_{55}-q=2,099.
\]

At a point where \(D(C,K21)\) has full row rank \(q\) modulo conjugacy, the
constant-rank theorem makes the local fiber dimension exactly \(2,099\).
This lane has **not** constructed an analytic nonzero \(q\)-minor or a pure-
core tuple proving that such a point exists.  Therefore “generic exact
\(2,099\)” is conditional on the full-row-rank witness; the unconditional
statement is the \(\ge2,099\) kernel bound.

So \(C,K21\) alone cannot identify an \(r=55\) core, even modulo
conjugacy.  An explicit one-core obstruction is that fixed
\(\operatorname{tr}A^2,\operatorname{tr}A^3\) do not fix
\(\operatorname{tr}A^4\).

Useful held-out identities are

\[
K31_{ij}=48\operatorname{tr}(A_i^3A_j),
\]

\[
K22_{ij}=32\operatorname{tr}(A_i^2A_j^2)
+16\operatorname{tr}(A_iA_jA_iA_j)
=48P_{ij}-8\lVert[A_i,A_j]\rVert_F^2,
\]

where \(P_{ij}=\operatorname{tr}(A_i^2A_j^2)\).  Necessary consistency
conditions include

\[
P-\frac1{32}KC^{-1}K^\top\succeq0,
\qquad
K31_{ii}=K22_{ii}\ge\frac32(KC^{-1}K^\top)_{ii},
\]

\[
\frac{12C_{ii}^2}{r}\le K31_{ii}\le12C_{ii}^2,
\qquad 0\le K22_{ij}\le\sqrt{K31_{ii}K31_{jj}}.
\]

## 3. Trajectory and functional observability

Full latent identification is unnecessary.  The correct local condition for
the value map \(\Xi=\{H3,q4(211),K31,K22,2PI\}\) is

\[
\ker D\mathcal O_L\subseteq\ker D\Xi
\quad\text{modulo conjugacy}.
\]

If this fails, observations can differ only at \(O(t^2)\) while a value
target differs at \(O(t)\), so no stable estimator exists.

Dense mixing and a spectral per-core reset cannot enlarge the generated
\(*\)-algebra \(\operatorname{alg}^*\{A_i\}\).  If mixing is invertible and
the spectral response is injective on the relevant spectrum, that algebra is
preserved exactly.  A commuting/star seed therefore remains commuting.  A
necessary genericity gate is scalar commutant (equivalently full matrix
algebra by Burnside).  Reset conditioning is controlled by divided
differences

\[
\frac{\rho(\lambda_p)-\rho(\lambda_q)}{\lambda_p-\lambda_q}
\]

off diagonal and by the eigenvalue Jacobian on diagonal modes; zeros are
trajectory-dark directions.

The strongest held-out fiber probe is the mixed quartic

\[
q4(211)_{iijk}=32\operatorname{tr}(A_i^2A_jA_k)
+16\operatorname{tr}(A_iA_jA_iA_k),
\]

or, with \(X\circ Y=(XY+YX)/2\),

\[
q4(211)_{iijk}=16\langle A_i^2,A_j\circ A_k\rangle
+32\langle A_i\circ A_j,A_i\circ A_k\rangle.
\]

K31 and K22 are only degenerations of this tensor.  The 2PI fish is not an
independent source: after subtracting the already-carried D11/CP3 term, its
total is \(VG^2V^\top-DG^2D^\top\).  It is a derived signed-H3 variance arm,
not separate value credit.

## 4. Universal rotation tomography and its limit

Because \(1024=4^5\), there are 513 real mutually unbiased orthonormal
bases.  Their centered rank-one projectors give an orthogonal decomposition

\[
513(1023)=524,799=\dim\operatorname{Sym}^2_0(\mathbb R^{1024}).
\]

Consequently 513 bases reconstruct the signed-H3 map
\(\operatorname{Sym}^2_0\to V\) with traceless singular value exactly one;
baseline plus 512 rotations is sharp among orthonormal-basis-projector
tomography schemes.  Full mixed q4, regarded as
\(\operatorname{Sym}^2\to\operatorname{Sym}^2\), has the same universal
frame; K31/K22 alone do not.

For an arbitrary task direction and \(m\) MUB blocks, the worst-case energy
bound is \(\rho^2\le m/513\).  Uniform correlation 0.9253 therefore needs
at least 440 blocks and 0.98346 at least 497 blocks.  This is a universal
bound, not the much smaller value-targeted bound below.

## 5. Value-targeted rank accounting

Let

\[
F_A=\ker D(C,K21)_A/\text{gauge},\qquad
f_v=\operatorname{rank}(D\Xi_A|_{F_A})\le2,099,
\]

and let the endpoint adjoint subspace have dimension \(k\).  One lawful
rotated context exposes at most

\[
b=k(n-1)=1023k
\]

centered scalar probes.  Therefore the exact necessary count is

\[
m\ge\left\lceil\frac{f_v}{1023k}\right\rceil.
\]

At the worst-case \(f_v=2,099\), this is 3 contexts for \(k=1\), 2 for
\(k=2\), and 1 for \(k\ge3\).  If \(\lambda_i\) are descending eigenvalues
of the task Gramian, define

\[
r_\rho=\min\left\{r:\sum_{i\le r}\lambda_i\ge
\rho^2\sum_i\lambda_i\right\}.
\]

Then \(m\ge\lceil r_\rho/(1023k)\rceil\).  For a flat 2,099-dimensional
task, \(r_{0.9253}=1,798\) and \(r_{0.98346}=2,031\); \(k=1\) needs two
contexts for either average-energy target and \(k\ge2\) needs one.

With a 49-GEMM total over ten layers, a context costing \(c\) GEMMs per
layer permits at most

\[
m_{\max}=\left\lfloor\frac{49}{10c}\right\rfloor,
\]

namely four at \(c=1\) or two at \(c=2\).

## 6. Exterior correction / no-go

An exact graded matrix-unit seed cannot live at \(r=55\).  For a split
\(r=p+q\), its odd block has dimension \(pq\le\lfloor r^2/4\rfloor\); at
\(r=55\) this is only 756, below \(n=1024\).  The full matrix-unit seed
requires \(p=q=32\), hence \(r=64\).

At \(r=64\),

\[
A(E)=\begin{pmatrix}0&E\\E^\top&0\end{pmatrix},\qquad E\in M_{32},
\]

has \(C\propto I\), \(K21=0\), and every input rotation preserves those
moments.  Its diagonal quartic is

\[
\operatorname{tr}A(E)^4=2\operatorname{tr}(EE^\top)^2.
\]

The continuous stabilizer has dimension
\(2\dim O(32)=992\), so the q4-changing orbit has dimension

\[
\frac{1024\cdot1023}{2}-992=522,784.
\]

Thus exact uniform observation needs
\(m\ge\lceil522,784/(1023k)\rceil\): 512, 256, 171, 128 contexts for
\(k=1,2,3,4\).  In particular any one-to-three-context construction has
uniform restricted singular value zero on this family.  This rules out the
proposed exterior witness; it does not rule out a pointwise generic r55
witness.

## 7. Checkpoint 4: sharp conclusion for a generic r55 core

For a fixed regular r55 core, the rank counts in section 5 are the sharpest
purely analytic lower bounds currently justified.  They make an O(1)-context
solution dimensionally possible, but do not prove one.  Define the
task-normalized restricted condition

\[
\tau_\Xi(A,W_{1:m})=
\inf_{h:\,\lVert D\Xi_Ah\rVert=1}
\lVert D\mathcal O_{A,W_{1:m}}h\rVert.
\]

A GO requires an explicit core and contexts with

\[
\det\left[(M|_{F_v})^*(M|_{F_v})\right]>0,
\qquad \tau_\Xi>0.
\]

Dimension counting, \(C,K21\), and the word "generic" do not imply that
minor is nonzero.  If one explicit nonzero minor is found, analyticity would
only prove pointwise Zariski-generic rank.  It still gives no uniform
conditioning: the regular irreducible locus is open rather than compact and
accumulates on the reducible/commuting/spectral discriminant where the
restricted minors vanish.  Hence the infimum of the unqualified condition
bound over generic cores is zero.  There is no uniform O(1)-context
correlation guarantee from the supplied premises.

This is a HOLD, not an exact no-go: a particular well-conditioned r55 core
may admit the dimension-minimal 3/2/1-context construction.  To claim 0.9253
or 0.98346 correlation, the observation/model mismatch divided by
\(\tau_\Xi\) and the target signal must be below, respectively,

\[
\sqrt{1-0.9253^2}=0.3792,\qquad
\sqrt{1-0.98346^2}=0.1811.
\]

### Current next question

Construct, symbolically, one normalized regular irreducible r55 tuple and
one set of at most four lawful contexts for which a 2,099-dimensional
restricted minor is provably nonzero.  Failing that, reduce the target to a
certified low-rank adjoint subspace and prove a positive lower bound on
\(\tau_\Xi\) on an explicitly compact, discriminant-separated class.  No
board or fidelity claim is justified before one of these two certificates
exists.

## 8. Pointwise transversality refinement

There is an abstract nonzero-minor theorem for a **fixed** scalar-adjoint
target tangent.  Let \(T\subset\operatorname{Sym}^2_0(\mathbb R^n)\) have
dimension \(f\), and define the centered diagonal observation

\[
\mathcal D_{W_{1:m}}(H)=
\big(\operatorname{diag}_0(W_s^\top H W_s)\big)_{s=1}^m.
\]

For every nonzero traceless symmetric \(H\), the zero-diagonal basis locus

\[
Z_H=\{W\in O(n):\operatorname{diag}_0(W^\top H W)=0\}
\]

has codimension \(n-1\).  Existence follows from Schur--Horn; at a
zero-diagonal representative with connected nonzero off-diagonal graph, the
differential is the weighted graph-divergence map and has rank \(n-1\).
The incidence variety

\[
I=\{([H],W_1,\ldots,W_m):H\in T\setminus0,\ W_s\in Z_H\}
\]

therefore obeys

\[
\dim I\le(f-1)+m\big(\dim O(n)-(n-1)\big).
\]

If \(m(n-1)\ge f\), its projection cannot fill \(O(n)^m\).  Hence a
Zariski-open set of basis tuples makes \(\mathcal D|_T\) injective.  For a
fixed contracted r55 value tangent of dimension 2,099, three generic lawful
contexts therefore have a nonzero restricted minor; two suffice for target
ranks 1,798 or 2,031.

This refines the prior HOLD: pointwise rank existence is proved abstractly,
but no useful condition number or context-selection rule follows from
\(C,K21\), because the target tangent itself varies along their unidentified
fiber.  A board-relevant claim still needs a fiber-specific lower bound on
\(\tau_\Xi\).

For endpoint dimension \(k>1\), the nominal \(1023k\) rank per context is
not automatically a transversality bound.  A target direction whose endpoint
tensor has Schmidt rank one imposes only \(1023\) independent zero-diagonal
conditions.  The factor \(k\) is valid only after proving an endpoint-Schmidt
or simultaneous-diagonal codimension gate for every nonzero value-relevant
fiber direction.

## 9. Four-context joint H3/q4 theorem

The nonlinear reset supplies a useful two-coefficient pencil.  If

\[
R(X)=\alpha X+\beta X^2+O(X^3),
\]

then

\[
\operatorname{tr}(R(X_a)^2R(X_b))
=\alpha^3\operatorname{tr}(X_a^2X_b)
+\alpha^2\beta\left[
\operatorname{tr}(X_a^2X_b^2)+2\operatorname{tr}(X_a^3X_b)
\right]+O(5).
\]

After scalar-adjoint contraction and removal (or separate recording) of the
identity component, write the cubic and quartic tangent matrices as
\((U_h,V_h)\in E\oplus E\), with
\(E=\operatorname{Sym}^2_0(\mathbb R^{1024})\).  Context \(s\) observes

\[
\operatorname{diag}_0\left[
W_s^\top(U_h+t_sV_h)W_s
\right],\qquad t_s=\frac{\beta_s}{\alpha_s}.
\]

Take four distinct ratios \(t_1,\ldots,t_4\).  For every nonzero pair
\((U,V)\), at most one combination \(U+t_sV\) can vanish.  Thus at least
three contexts impose a zero-diagonal condition of codimension 1,023.
For any fixed joint target tangent \(S\subset E\oplus E\) of dimension at
most 2,099, the bad incidence has codimension at least

\[
3(1023)-(2099-1)=971>0.
\]

Therefore a Zariski-open set of four lawful basis contexts is injective on
**any fixed** such \(S\).  If \(S\) contains no coefficient-rank-one pair
\((U,tU)\), three contexts already suffice; a generic 2,099-plane has this
property because the rank-one coefficient cone has codimension
\(\dim E-1\), vastly larger than 2,099.

The four-context construction costs 40 GEMMs over ten layers when the
incremental cost is one GEMM per context per layer, fitting the 49-GEMM
envelope.  It does not fit if the true incremental cost is two GEMMs per
layer.  The coefficient-separation determinant for two operating points is

\[
\alpha_1^2\alpha_2^2
(\alpha_1\beta_2-\alpha_2\beta_1),
\]

where the leading multiplication denotes the product
\(\alpha_1^2\alpha_2^2(\alpha_1\beta_2-\alpha_2\beta_1)\); it must be bounded
away from zero, not merely nonzero.  The final fidelity gate remains the
actual restricted \(\tau_\Xi\) of the realized four transitions.

### Updated next question

Determine the actual reset ratios \(t_s=\beta_s/\alpha_s\) symbolically from
the ten-layer operating points and prove that four admissible layers have
separated ratios and one-GEMM incremental cost.  Then lower-bound the
restricted spatial singular value for their realized dense weights.  If
either coefficient separation or the one-GEMM accounting fails, this route
cannot meet the 49-GEMM envelope.

## 10. Symbolic reset ratios for a Gaussian-smoothed ReLU

Let

\[
f(q)=E_g[(\mu+q+\sigma g)_+]
=\sigma\phi(a+q/\sigma)+(\mu+q)\Phi(a+q/\sigma),
\qquad a=\mu/\sigma.
\]

Then \(f'(0)=\Phi(a)\) and \(f''(0)=\phi(a)/\sigma\).  The second-chaos
product formula is

\[
P_2\left[(\zeta^\top B\zeta-\operatorname{tr}B)^2\right]
=4\left(\zeta^\top B^2\zeta-\operatorname{tr}B^2\right).
\]

Consequently

\[
R(B)=\alpha B+\beta B^2+O(B^3),\qquad
\alpha=\Phi(a),\quad \beta=\frac{2\phi(a)}{\sigma},
\]

and the context ratio is

\[
t=\frac{\beta}{\alpha}
=\frac{2\phi(a)}{\sigma\Phi(a)}.
\]

At zero mean this reduces to \(t=4/(\sigma\sqrt{2\pi})\).  Thus four
distinct ratios require genuine layerwise variation of \(\sigma\) or of the
standardized mean.  If normalization fixes both, all reset ratios coincide
and the four-context quartic-separation premise fails.

## 11. Tensor-free local K31 Bellman action

For centered Gaussian quadratic forms \(X(A)\),

\[
\kappa_4(X(A),X(B),X(C),X(D))
=16\left[\operatorname{tr}(ABCD)+\operatorname{tr}(ABDC)
+\operatorname{tr}(ACBD)\right].
\]

In particular, \(\kappa_4(X(B),X(B),X(B),X(D))=48\operatorname{tr}(B^3D)\).
For the supplied Bellman action, flatten the carried cores into
\(A^\flat\in\mathbb R^{n\times d}\), \(d=r(r+1)/2\), and put

\[
P=WQ31^\top,\qquad
B_a=\sum_iW_{ia}A_i,\qquad
D_a=\sum_iP_{ia}A_i.
\]

The exact scalar is therefore

\[
\boxed{\mathcal B_{31}=48\sum_{a=1}^n\operatorname{tr}(B_a^3D_a)}.
\]

No fourth-order tensor is formed.  This is a local task factorization, but it
assumes the correct NCQC representative has already been selected.  The
three/four-context theorem by itself only identifies a hidden fiber; it does
not make this scalar well-defined from \(C,K21\).

### Restricted scalar gate

Let \(M:F_A\to\mathbb R^s\) be the acquired context map and let
\(\ell=D\mathcal B_{31}|_{F_A}\).  Exact local task identification is the
strictly weaker condition

\[
\ker M\subseteq\ker\ell,
\quad\text{equivalently}\quad
\ell\in\operatorname{row}M.
\]

Define the scalar-specific stability

\[
\tau_{31}=\left\|(M^\dagger)^*\ell\right\|^{-1}.
\]

An acquisition perturbation of norm \(\eta\) then gives

\[
|\delta\mathcal B_{31}|\le\eta/\tau_{31}
\le \|\ell\|\eta/\sigma_{\min}(M|_{F_A/\ker\Xi}).
\]

The corresponding normalized error must be below 0.3792 or 0.1811 for the
0.9253 or 0.98346 correlation gates.  Injectivity without this scalar
condition number is not a fidelity result.

### Acquisition and action cost gate

For \(n=1024,r=55\),

\[
d=1540,\qquad \frac dn=1.50390625,\qquad
s_r=\frac{r^3}{n^2}\approx0.15867
\]

in dense \(n^3\)-GEMM equivalents.  A ten-layer core chain using one
rectangular core mix and one \(r\times r\) product per layer costs

\[
10(d/n+s_r)\approx16.626.
\]

At the endpoint, \(P=WQ31^\top\), the mix \(D=P^\top A^\flat\), and one
additional per-core product cost

\[
1+d/n+s_r\approx2.663.
\]

Thus the optimistic exact chain plus local K31 action is about

\[
16.626+2.663=19.289<21.
\]

The remaining margin is only about 1.71 GEMM-equivalents.  It can absorb at
most one additional rectangular core mix, or about ten additional
\(r\times r\) product banks, but not three or four independent ten-layer
context chains.  Therefore the 3/4 contexts fit the residual budget only if
they are harvested from already-computed natural layers and their Bellman
fourth legs are already carried.  If every context needs a new
\(n\times n\) by \(n\times1540\) acquisition, the budget fails.

More generally, with reset cost \(c_Rs_r\) per layer, endpoint product cost
\(c_Ps_r\), and \(g\) extra rectangular acquisitions,

\[
\mathcal C=
10(d/n+c_Rs_r)+1+d/n+c_Ps_r+g(d/n).
\]

The 21-GEMM gate requires \(\mathcal C\le21\).  At \(g=0,c_P=1\), this
requires \(c_R\lesssim2.08\).  An eigensolver or spectral reset with a larger
constant fails even though a single matrix-square reset passes.

## 12. Required graph-node split

These claims must remain distinct in the problem graph:

1. **NCQC fiber dimension** -- closed theorem: regular r55 fiber is 2,099D.
2. **Abstract context injectivity** -- closed conditional theorem: three
   scalar contexts, or four distinct-ratio joint H3/q4 contexts, have a
   nonzero minor for a fixed target tangent.
3. **Quantitative conditioning** -- open: certify realized
   \(\tau_\Xi\) and \(\tau_{31}\).
4. **Context acquisition cost** -- hold: prove contexts and fourth legs ride
   the existing chain with \(g=0\), and prove \(c_R\lesssim2.08\).
5. **K31 task factorization** -- exact theorem for
   \(48\sum_a\operatorname{tr}(B_a^3D_a)\), conditional on a correctly
   identified carried core.

Combining any of nodes 2--5 would incorrectly turn an identifiability theorem
into a deployable fidelity claim.

## 13. Cost correction: fuse away \(P=WQ31^\top\)

The preceding 19.289 estimate is conservative by one full GEMM.  The
intermediate \(P\) need not be materialized.  Indeed,

\[
D_a=\sum_iP_{ia}A_i
=\sum_{i,j}W_{ij}Q31_{aj}A_i
=\sum_jQ31_{aj}B_j.
\]

Thus \(D=Q31\,B^\flat\) is one rectangular mix of the already-computed
forward cores.  The endpoint marginal is

\[
d/n+s_r\approx1.663,
\]

and the corrected ten-layer chain plus local K31 action is

\[
10(d/n+s_r)+(d/n+s_r)\approx18.288<21.
\]

The true residual margin is about 2.712 GEMM-equivalents.  The corrected
general budget is

\[
\mathcal C=
10(d/n+c_Rs_r)+d/n+c_Ps_r+g(d/n)\le21.
\]

At \(g=0,c_P=1\), the reset constant may be as large as approximately
\(c_R=2.71\).  This still cannot pay for three extra rectangular context
acquisitions (cost about 4.512), so the qualitative acquisition gate is
unchanged: the contexts must ride the existing trajectory.

## 14. Low-rank Bellman-adjoint acquisition

If \(Q31=LR^\top\) has rank \(k\), the fused fourth leg can be formed as

\[
E=R^\top B^\flat,\qquad D=LE.
\]

The two thin products cost

\[
\frac{2knd}{n^3}=\frac{2kd}{n^2}
\approx0.002937\,k
\]

GEMM-equivalents, instead of \(d/n\approx1.504\).  Four Bellman-matched
fourth-leg acquisitions on already-computed natural layers therefore cost
only about \(0.01175k\).  This makes the four-context route economically
plausible when the endpoint adjoint really is low rank; the algebraic rank
must be certified rather than inferred from the symbol \(k\).

For an approximate rank-\(k\) adjoint \(\widetilde Q\), define

\[
G(B)_{aj}=\operatorname{tr}(B_a^3B_j).
\]

Then the deterministic action error obeys

\[
|\Delta\mathcal B_{31}|
\le48\lVert Q31-\widetilde Q\rVert_F\lVert G(B)\rVert_F.
\]

This error, divided by the K31 task signal, must independently pass the
0.3792/0.1811 normalized fidelity gate before low-rank acquisition receives
credit.

## 15. Quantitative coefficient/spatial conditioning split

For scalar reset coefficients, let row \(s\) of the two-source coefficient
matrix be

\[
R_s=\alpha_s^3(1,t_s).
\]

Then

\[
\det(R^\top R)=
\sum_{s<t}\alpha_s^6\alpha_t^6(t_s-t_t)^2,
\]

and therefore

\[
\sigma_{\min}(R)^2
\ge
\frac{\sum_{s<t}\alpha_s^6\alpha_t^6(t_s-t_t)^2}
{\sum_s\alpha_s^6(1+t_s^2)}.
\]

This converts the qualitative "distinct ratios" condition into a numerical
separation certificate.  Let \(\gamma_J\) be the lower singular value of the
fiber-to-\((U,V)\) Jacobian after quotienting its target kernel, and define the
realized spatial-angle constant by

\[
\sum_s\left\lVert\operatorname{diag}_0
(W_s^\top H_sW_s)\right\rVert^2
\ge\gamma_{\rm sp}^2\sum_s\lVert H_s\rVert_F^2
\]

on the coefficient-pencil image.  Then the useful factorized gate is

\[
\tau_\Xi\ge\gamma_J\gamma_{\rm sp}\sigma_{\min}(R).
\]

For ReLU, \(m(a)=\phi(a)/\Phi(a)\) obeys
\(m'(a)=-m(a)(a+m(a))<0\), so fixed \(\sigma\) and separated standardized
means give separated ratios.  Fixed \(a,\sigma\) gives coefficient rank one
exactly.

This scalar-pencil formula is valid only when \(\alpha,\beta\) are common
within a context.  If they vary neuron by neuron, the reset expansion has
separate weights \(\alpha_a^2\beta_b\) and
\(2\alpha_a\beta_a\alpha_b\); that richer weighted design must be certified
directly and cannot be replaced by the 2-by-2 determinant above.

## 16. K31 novelty / baseline-subtraction correction

The factorization

\[
48\sum_a\operatorname{tr}(B_a^3D_a)
\]

is the **total** NCQC quartic action.  If D11/CP3 already carries a baseline
quartic action, the NCQC result must replace that action or be differenced
against an exactly matched baseline; adding it wholesale double counts.

If \(A=A_D+A_R\), with corresponding \(B=B_D+B_R\) and \(D=D_D+D_R\),
the lawful residual is

\[
\Delta\mathcal B_{31}=48\sum_a\left\{
\operatorname{tr}\left[(B_{D,a}+B_{R,a})^3(D_{D,a}+D_{R,a})\right]
-\operatorname{tr}(B_{D,a}^3D_{D,a})\right\}.
\]

All noncommutative D--R cross words in the cube are required.  The R-only
term \(48\sum_a\operatorname{tr}(B_{R,a}^3D_{R,a})\) is not the residual and
will generally miss the leading cross contribution.  The novelty gate is an
exact baseline replay and a correction that vanishes identically at
\(A_R=0\).

## 17. Cheap-reset fidelity bound (cross-lane h6 result)

Let \(b_F=\lVert B\rVert_F\), \(b_{op}=\lVert B\rVert_{op}\),
\(\rho=b_{op}/b_F\), and \(x=b_F/\sigma\).  The exact reset expansion is

\[
R(B)=\alpha B+\beta B^2
+f'''(0)\{\operatorname{tr}(B^2)B+4B^3\}+\operatorname{Rem}_4,
\]

with \(f'''(0)=-a\phi(a)/\sigma^2\).  The h6 remainder derivation gives

\[
\begin{aligned}
\lVert R-\alpha B-\beta B^2\rVert_F
\le{}& |a|\phi(a)\frac{b_F^3}{\sigma^2}(1+4\rho^2)\\
&+\phi(0)\frac{b_F^4}{\sigma^3}
\left[2+8\rho^2+\frac{\sqrt6}{3}\sqrt{1+4\rho^2}\right].
\end{aligned}
\]

At \(a=0\), the cubic term vanishes exactly and

\[
\frac{\lVert R-\alpha B-\beta B^2\rVert_F}{\alpha b_F}
\le\sqrt{\frac2\pi}\,x^3
\left[2+8\rho^2+\frac{\sqrt6}{3}\sqrt{1+4\rho^2}\right].
\]

This is the analytic fidelity gate that can justify the one-product reset
used in the 18.288-GEMM estimate.  It must be transported through all layers;
the local inequality alone is not an end-to-end correlation certificate.

## 18. Reverse-action and acquisition caveat (cross-lane h4 result)

With \(D=Q31\,B\), the exact gradient with respect to core \(B_p\) is

\[
48\left[
B_p^2D_p+B_pD_pB_p+D_pB_p^2
+\sum_aQ31_{ap}B_a^3
\right].
\]

The final term is the derivative through the fourth leg and must not be
omitted.  The 18.288-GEMM estimate assumes the required Q31 bundle is already
available.  A separate ten-layer target-conditioned matrix bundle is about
20 GEMM-equivalents in the h4 accounting, and an unfused local Jordan-reset
JVP can add another ten.  Either cost destroys the 21-GEMM claim.  Therefore
the acquisition-cost node stays HOLD until the existing card is shown to
carry/fuse these objects; the tensor-free local contraction alone does not
prove deploy economics.

For the special diagonal quartic replacement \(B=R+\Delta\), the exact
noncommutative trace correction is

\[
4\operatorname{tr}(R^3\Delta)
+4\operatorname{tr}(R^2\Delta^2)
+2\operatorname{tr}(R\Delta R\Delta)
+4\operatorname{tr}(R\Delta^3)
+\operatorname{tr}(\Delta^4),
\]

again showing why an R-only residual is invalid.

## 19. Source observability does not imply K31 value capture

The direct-adjoint lane supplied a sharp counterexample.  Let the acquired
source rowspace be \(P=\operatorname{span}(e_1)\), let

\[
x=\sqrt\tau\,e_1+\sqrt{1-\tau}\,e_2,
\]

and take the endpoint adjoint \(a=e_2\).  Source energy capture is exactly
\(\lVert Px\rVert^2/\lVert x\rVert^2=\tau\), arbitrarily close to one, but
the captured scalar is zero while
\(\langle a,x\rangle=\sqrt{1-\tau}\).  Hence no positive lower bound on
K31 scalar fidelity follows from source-only \(\tau_\Xi\).

Define endpoint leverage and adjoint leakage by

\[
\lambda=\frac{|\langle a,x\rangle|}{\lVert a\rVert\lVert x\rVert},
\qquad
\mu=\frac{\lVert P^\perp a\rVert}{\lVert a\rVert}.
\]

Then

\[
\frac{|\langle a,(I-P)x\rangle|}{|\langle a,x\rangle|}
\le\frac{\mu\sqrt{1-\tau}}{\lambda}.
\]

A sign/amplitude certificate requires
\(\mu\sqrt{1-\tau}<\lambda\).  The strong sufficient condition \(Pa=a\)
gives exact scalar capture despite an unobserved source tail.  In the
population/operator setting, replace \(\mu\) by
\(\lVert A(I-P)\rVert/\lVert A\rVert\) and certify in the endpoint-weighted
norm.  The graph conditioning node must therefore track \(\tau_{31}\) or
this adjoint-angle/leverage receipt, not generic H3/q4 source correlation.

## 20. Checkpoint 8: exact D11 same-site duplicate audit

The reconstructed affine--quadratic K31 action has **zero same-site source
novelty** in the deployed D11 lineage.  The ownership chain is exact:

1. `pre_k31` enters `_full_relu_reset` and `_edgeworth_weights` through the
   H13/H31 weights `k31.T/6` and `k31/6`.
2. `_connected_planes` forms the centered off-diagonal `k31_target`, including
   the single disconnected subtraction \(-3v_iC_{ij}\).
3. `_quartic_residual` removes the active CP4 plane exactly once;
   `_reset_stars` separates the fitted pairing and residual star.
4. On the next transition, `_push_quartic` reconstructs star plus pairing and
   `_assemble_quartic_provider` adds CP4 exactly once.

Consequently

\[
K31_{\rm old}=K31_{\rm CP4}+K31_{\rm star}+K31_{\rm pair}
\]

is already the complete D11/Q24 K31 interface at that site.  H31 is its reset
consumer, not another independent source, and CP3 effects are already folded
through the reset moments.  The NCQC/AQ scalar

\[
\chi_{AQ}=48\sum_a\operatorname{tr}(B_a^3D_a)
\]

(plus the affine H1--H2 terms when present) may replace this old action, but
may not be added to it.  Its same-site old-action novelty score is exactly
zero.

Two seam conditions are binding:

- D11 stores `k31_target` off diagonal.  The diagonal fourth cumulant belongs
  to the separate K4 vector.  Therefore use \(\operatorname{offdiag}Q31\) in
  the K31 contraction, or split \(\operatorname{diag}Q31\) and contract it
  with K4 using the K4 normalization.
- Post-weight/pre-ReLU AQ cores must be compared with assembled `pre_k31`.
  Post-reset cores must be compared with `k31_target` and its post costate.
  A pre/post subtraction is not novelty.

This closes only the **old-action novelty** node.  The abstract observability
and tensor-free replacement identities remain true but receive no raw-gain
credit.

## 21. The only same-site residuals

Let \(C_{D11}\) denote the frozen diagonal/equality/collision convention and
let \(O\) be the complete old feature span.  The only creditable exact scalar
is

\[
\boxed{
\rho_{31}=(I-OO^+)\,
\left\langle Q31,
C_{D11}(K31_{\rm true}-K31_{\rm assembled})
\right\rangle .}
\]

It is a normal-ordered fourth-cumulant transition defect, not the 2PI cubic
fish counterterm.  Subtracting the D11 fish again would be a second
subtraction.

The AQ decomposition is

\[
\rho_{31}^{\rm prequotient}
=\underbrace{\langle Q,C(K31_{AQ}-K31_{\rm assembled})\rangle}
_{\text{synchronization-defect proxy}}
+\underbrace{\langle Q,C(K31_{\rm true}-K31_{AQ})\rangle}
_{\text{beyond-AQ residual}}.
\]

The first term is computable from a lawful oriented AQ core, but it is only a
proxy until the second term has a certified endpoint-weighted radius.  If
\(\lVert\chi_{\rm true}-\chi_{AQ}\rVert_2\le\eta_{AQ}\), and the old-span
residualized proxy has norm \(s\), the guaranteed residual signal is only
\((s-\eta_{AQ})_+\).  A nonzero proxy by itself is not novelty.

Writing \(X=X_{AQ}+R_{>2}\), multilinearity gives the exact beyond-AQ
expansion

\[
\begin{aligned}
\Delta\kappa_4={}&
3\kappa_4(R_U,U_0,U_0,V_0)
+3\kappa_4(R_U,R_U,U_0,V_0)
+\kappa_4(R_U,R_U,R_U,V_0)\\
&+\kappa_4(U_0,U_0,U_0,R_V)
+3\kappa_4(R_U,U_0,U_0,R_V)
+3\kappa_4(R_U,R_U,U_0,R_V)
+\kappa_4(R_U,R_U,R_U,R_V).
\end{aligned}
\]

Every term contains at least one higher-chaos leg.  None is determined by the
r55 AQ core or its 2,099-dimensional moment fiber.

## 22. Information-minimal residual observable and context count

For one fixed endpoint adjoint, one projected scalar is necessary and
sufficient:

\[
z_{31}=\sum_{l,a}\left{
E[U_{l,a}^3V_{l,a}]-3E[U_{l,a}^2]E[U_{l,a}V_{l,a}]
\right\}-\chi_{\rm assembled}.
\]

Equivalently, acquire the true action and subtract the known assembled action.
This collapses the endpoint fiber width to zero.  For a family of endpoints,
the minimal number of scalar observables is

\[
r_{\Lambda}=\operatorname{rank}
\left\{\ell_e|_{\ker S_{\rm current}}\right\}_e,
\]

not one scalar per nominal panel member if their restricted adjoints are
dependent.  Per-layer scalars are required only when layers are queried or
injected independently; a frozen aggregate endpoint requires one aggregate
scalar.

The earlier three/four-context theorem concerns identification of a fixed AQ
fiber and is now an **old-action observability theorem**, not a novelty
mechanism.  The beyond-AQ fiber is not 2,099-dimensional and no O(1) generic
context bound follows.  If true-moment diagonal contexts expose a finite
value-relevant residual rank \(f_{>AQ}\), the capacity lower bound is

\[
m\ge\left\lceil\frac{f_{>AQ}}{1023k}\right\rceil
\]

with the same endpoint-Schmidt and condition-number gates.  Without a proved
finite \(f_{>AQ}\), the only sharp minimum is the directly matched scalar:
one per independent endpoint adjoint.

### Cost

If an engine natively exposes the true projected action, residual formation is
one scalar subtraction.  If it exposes a true K31 plane, contraction costs
\(O(n^2)\) per layer, or \(L/n\) GEMM-equivalents (about 0.0098 for ten
layers), and only one scalar per layer need be retained.  These arithmetic
costs do **not** price creation of the true moment oracle.

No computation on the existing D11/AQ summaries can create that oracle:
same-summary/different-fourth-action counterexamples make the acquisition
information-theoretically necessary.  Computing the AQ proxy costs the
previously derived carried-core budget, but does not remove
\(\eta_{AQ}\).  A generic explicit H3/H4 completion is superquadratic state
unless a low-rank theorem is supplied, so it currently has no <=21-GEMM
certificate.

## 23. Future-visible completion is a separate residual

Let \(O_l\) be the **complete** old D11 fourth-order interface at layer \(l\):
the CP4 carried state together with the K4/K31/K22 star/pair readouts.  Define

\[
N_l=\ker O_l.
\]

Using only \(\ker K31_l\) is invalid because a direction may already be
visible through K22, K4, or CP4.  A presently invisible completion \(r_l\in
N_l\) has future endpoint value at layer \(m>l\) exactly when

\[
\boxed{
P_{N_l}\,\mathcal R_{m:l}^*\lambda_m\ne0,}
\]

where \(\mathcal R_{m:l}\) is the full intervening reset/transport tangent.
Its value is the scalar contraction

\[
\left\langle P_{N_l}\mathcal R_{m:l}^*\lambda_m,r_l\right\rangle.
\]

For a fixed endpoint this contracted scalar is again information-minimal.  A
state completion supporting several endpoints must carry at least

\[
r_{\rm fv}=\dim\operatorname{span}
\{P_{N_l}\mathcal R_{m:l}^*\lambda_{m,e}\}_{l,m,e}
\]

independent coordinates.  The direction is a NO-GO when the projected
pullback is zero.  It is a HOLD, not a deployable source, when nonzero but no
lawful residual oracle or positive conditioning bound exists.

The cheap case is a native scalar innovation tapped at the transition and
contracted with an already-carried reverse adjoint.  A new reverse bundle,
new higher-chaos state, or independent context chain must be charged
separately; the tensor-free old K31 contraction cannot pay that cost.

## 24. Required node separation after the duplicate theorem

1. **Old K31 action observability** -- solved mathematics, zero novelty.  It
   covers AQ fiber/context injectivity and the tensor-free replacement action.
2. **Same-site D11 duplicate** -- closed: CP4 plus star plus pairing/H31 owns
   the assembled action exactly once, with time/gauge/diagonal seams.
3. **Beyond-AQ/synchronization residual novelty** -- open: acquire
   \(\rho_{31}\) or prove endpoint-weighted \(\eta_{AQ}\), then residualize
   against the complete old span.
4. **Residual acquisition cost** -- open: a native scalar is cheap, but its
   source is absent; AQ proxy cost is not exact-oracle cost.
5. **Future-visible completion** -- open and distinct: certify
   \(P_{N_l}\mathcal R^*\lambda\ne0\), its rank, source, and cost.

No edge from node 1 may be labeled as new raw gain.

## 25. Exact consumer/source dual identity and cost reconciliation

The same-site duplicate can be written without implementation names.  With
the deployed derivative kernels \(G31,G13\), local mask \(H\), and assembled
source plane \(D31\), define

\[
I31=\frac{G31\odot D31+G13\odot D31^\top}{6},
\]

\[
Q31=\frac{H\odot G31+(H\odot G13)^\top}{6}.
\]

Frobenius duality gives the exact identity

\[
\boxed{\langle H,I31\rangle=\langle Q31,D31\rangle.}
\]

Taking \(D31=D31_{\rm assembled}\) proves old-span ownership directly.
Replacing it by \(D31_{\rm true}-D31_{\rm assembled}\) gives the unique
same-site defect.  The existing interface exposes the assembled plane, not a
true-law `m31` or its JVP.

Cost statements must distinguish three objects:

- an optimized AQ **old-action replacement** can reuse the paired factorization
  at roughly 1.23 GEMM-equivalents per active layer, but has zero novelty;
- a generic exact unprojected fourth-action acquisition is \(O(n^4)\) and is
  outside budget;
- a native endpoint-projected defect is only one scalar per layer and has
  negligible contraction/storage cost, but the required oracle is not in the
  current state.

The earlier 18.288 figure prices an explicit r55 NCQC carried-core chain; it
is neither the cheapest old-action decoder nor the price of an exact
beyond-AQ oracle.

## 26. Lawful redirection of NCQC: future-visible AQ completion

The duplicate theorem does not kill an oriented AQ core if it carries
information in the kernel of the complete old fourth-order interface and that
information becomes visible only after a later dense reset.  Let

\[
R_l^{AQ}=(I-E_lO_l)T_l^{AQ}\in N_l=\ker O_l,
\]

where \(T_l^{AQ}\) is the full implicit AQ quartic tensor and \(E_lO_l\) is
the exactly matched old CP4/K4/K31/K22 reconstruction.  Its first lawful
future value is

\[
z_{l\to m}^{fv}
=\left\langle Q_m,
O_m\mathcal R_{m:l}R_l^{AQ}\right\rangle
=\left\langle
P_{N_l}\mathcal R_{m:l}^*O_m^*Q_m,
R_l^{AQ}\right\rangle.
\]

This is novel only when the projected pullback is nonzero.  At one pure-AQ
transition, its tensor-free value can be evaluated as

\[
z^{fv}=
48\sum_a\operatorname{tr}(B_a^3D_a)
-\left\langle Q,\operatorname{Push}_{31}(E_lO_lT_l^{AQ})\right\rangle,
\]

with the affine cross terms added when H1 is present.  The subtraction must
push the **complete** old interface; subtracting only old K31 leaves K4/K22/
CP4 leakage and is not a novelty proof.

For one fixed endpoint, the scalar \(z^{fv}\) is information-minimal.  If an
oriented core must instead be inferred from target-blind trajectory contexts,
let \(f_{fv}\) be the rank of the future-completion map on the surviving core
fiber.  Then

\[
m\ge\left\lceil\frac{f_{fv}}{1023k}\right\rceil,
\qquad f_{fv}\le2099
\]

is the capacity lower bound, with the earlier transversality, endpoint-angle,
and condition-number gates.  The old three/four-context theorem may be reused
only here, after replacing the full AQ target tangent by this completion
quotient.

The orientation must be lawful.  A core propagated from a synchronized
white-box initialization may qualify.  A per-layer r55 core arbitrarily refit
from \(C,K21\) does not: its 2,099-dimensional choice changes \(z^{fv}\), so
the completion is not a source observable.

For an already carried core, a single additional transition costs about
\(d/n+c_Rs_r\), approximately 1.66 GEMM-equivalents for a one-product reset.
A ten-transition suffix plus the fused endpoint action is the prior 18.288
conditional estimate.  It remains contingent on a reused Q31/reverse bundle,
the reset remainder gate, exact old-interface subtraction, and positive
endpoint-weighted residual after old-span projection.

## 27. Generic higher-chaos state is outside the residual budget

The first omitted Gaussian-chaos tensors already give a sharp state/action
no-go.  At latent rank \(r=55\), one symmetric H3 coefficient has

\[
d_3=\binom{r+2}{3}=\binom{57}{3}=29,260
\]

entries per output unit.  Transporting the flattened \(n\times d_3\) state
through one dense width-1024 transition costs

\[
d_3/n\approx28.57
\]

GEMM-equivalents, exceeding the entire 21-GEMM residual allowance in one
layer.  H4 is much worse:

\[
d_4=\binom{r+3}{4}=\binom{58}{4}=424,270,
\qquad d_4/n\approx414.3.
\]

Thus a generic exact H3/H4 completion is not a deployable source.  A shared
dictionary compression has an O(n^2)-state count only for approximately

\[
s_3\le\left\lfloor\frac{n^2}{d_3+n}\right\rfloor=34,
\qquad
s_4\le\left\lfloor\frac{n^2}{d_4+n}\right\rfloor=2,
\]

when both the shared atoms and their per-unit coefficients are counted, and
before pricing transitions and reset rank growth.  Such a
dictionary receives no credit without an endpoint-weighted tail bound and a
proof that dense transport/reset preserves its rank.  This leaves the native
projected residual scalar as the only currently justified exact
budget-compatible observable.

## 28. Exact residual API and consumption cost

For the actual centered layer law \(\nu_l\), columns \(w_a\) of \(W_l\),
and columns \(p_a\) of \(P_l=W_lQ_l^\top\), define

\[
\mathsf A_l(\nu_l;W_l,Q_l)=
\sum_a\left[
\int(w_a^\top x)^3(p_a^\top x)\,d\nu_l(x)
-3(w_a^\top C_lw_a)(w_a^\top C_lp_a)
\right].
\]

Then the exact same-site defect is

\[
\rho_{31,l}=\mathsf A_l-\langle Q_l,D^{31}_{l,\rm assembled}\rangle
\]

with the frozen collision and old-span quotient applied afterward.  An
equivalent label-free white-box API is a diagonal quartic potential

\[
\Phi_l(w)=\kappa_{4,\nu_l}(w^\top X_l,w^\top X_l,w^\top X_l,w^\top X_l)
\]

and its directional derivative, since

\[
\mathsf A_l=\frac14\sum_aD\Phi_l(w_a)[p_a].
\]

Values of \(\Phi\) on one fixed basis are not enough; the JVP in the physical
fourth-leg directions is the information-bearing interface.

Once a lawful scalar exists, consumption is cheap.  The preferred fixed-
endpoint realization is a scalar Bellman accumulator.  If a forward symmetric
plane is compulsory, let \(\mathcal E_l\) be the admissible injection space
and \(g_l=P_{\mathcal E_l}B_l\).  The minimum-Frobenius endpoint-equivalent
plane is

\[
\Delta_l=\frac{\rho_{31,l}}{\lVert g_l\rVert_F^2}g_l,
\qquad g_l\ne0.
\]

This matches only the fixed tangent and is not a physical multi-endpoint
closure.  A dense plane add is about \(2.1\times10^6\) flops per layer and can
ride existing conjugations; acquisition remains the blocker.

The synchronized-AQ proxy costs approximately 1.23 GEMM-equivalents per layer
only with reusable projected cores and reverse objects, 2.24 without the core
bank reuse, and 4.24 without reverse-conjugation reuse.  Even the optimistic
ten-layer 12.3 total exceeds a one-equivalent schedule headroom unless it
replaces/fuses an old operation.  These costs are relevant only for a proved
unowned defect or delayed completion; recomputing total AQ K31 earns zero.

## 29. Checkpoint-8 decision and next question

- **Close** additive same-site AQ/NCQC K31: exact D11 duplicate, zero Schur
  gain irrespective of reconstruction accuracy.
- **Keep open** one endpoint-projected true-minus-assembled scalar, conditional
  on a native fourth-potential JVP or a certified beyond-AQ radius.
- **Keep open separately** the future-visible completion quotient, conditional
  on a lawful oriented initialization, nonzero complete-interface pullback,
  endpoint conditioning, and the suffix cost gate.
- **Close** generic explicit H3/H4 carriage under the 21-GEMM budget absent a
  rank/tail theorem.

The next decisive audit is binary: does the white-box moment engine expose
\(D\Phi_l(w_a)[p_a]\), or an exactly equivalent projected true `m31` action,
at Site A/B?  If not, same-site novelty is information-theoretically absent
from current state.  In parallel, test symbolically whether the lawfully
initialized AQ core has a nonzero component in
\(P_{\ker O_l}\mathcal R_{m:l}^*\lambda_m\); only that delayed completion can
redeem the existing NCQC context/conditioning work.

## 30. Ambient analytic witness for future visibility

The future-completion quotient is nonvacuous at the full H4 level.  In the
current coordinate basis, star and pairing faces are generated by equality
monomials \(x_i^4,x_i^3x_j,x_i^2x_j^2\).  The all-distinct quartic subspace
has dimension \(\binom{1024}{4}\).  A rank-2048 CP4 carrier imposes at most
2,048 independent Waring-atom directions there, so a nonzero all-distinct
quartic \(R\) remains in the complete old-interface kernel.

For the explicit monomial \(R(x)=x_1x_2x_3x_4\), its polarized 3+1 face is

\[
T_R(u,u,u,v)=\frac14\sum_{s=1}^4v_s\prod_{t\ne s}u_t.
\]

Taking \(u=w_a,v=w_b\) from a generic dense transition makes this nonzero.
Hence a current-interface-invisible all-distinct direction becomes K31-visible
after one generic rotation.  This proves an ambient future-visible kernel,
not yet an NCQC provider: promotion still requires that the lawfully oriented
r55 completion intersects this kernel and that its pulled-back endpoint
adjoint has nonzero, well-conditioned value after CP4/old-span subtraction.

## 31. The exact r55 complete-interface test, and why the ambient witness is insufficient

Let \(\mathcal F_{55}\) be the regular inverse-fiber tangent after quotienting
orthogonal conjugacy.  The earlier count gives

\[
\dim \mathcal F_{55}=2099.
\]

Let \(\Psi_l\) map a lawful core tangent at a cut to the full closure state,
and let \(O_l^{\rm complete}\) denote the *complete* deployed D11 interface:
CP3/CP4 state, K4/K31/K22 star and pairing readouts, and every old depth-arena
feature.  The only relevant present-time observability matrix is

\[
J_l=D(O_l^{\rm complete}\circ\Psi_l)|_{\mathcal F_{55}},
\qquad N_l=\ker J_l.
\]

Thus a lawful r55 dark direction exists if and only if

\[
\operatorname{rank}J_l<2099. \tag{31.1}
\]

This is only a source statement.  For endpoint \(e\), let \(g_{l,e}\) be the
pulled-back covector of the strict-future Bellman value on
\(\mathcal F_{55}\).  A present-dark direction has future value if and only if

\[
P_{N_l}g_{l,e}\ne0. \tag{31.2}
\]

For a single scalar endpoint the value-relevant rank is therefore zero or one.
For a panel, it is exactly the rank of the Gram matrix

\[
\Gamma_{ef}=\langle P_{N_l}g_{l,e},P_{N_l}g_{l,f}\rangle.
\]

In an isotropic tangent metric, the endpoint-retention fraction is

\[
\tau_{55,e}=\frac{\|P_{N_l}g_{l,e}\|^2}{\|g_{l,e}\|^2}.
\]

If lawful tangents have covariance \(\Sigma_l\), the correct weighted version
is

\[
\tau_{55,e}^{(\Sigma)}=
\frac{g_{l,e}^{\mathsf T}P_{N_l}\Sigma_lP_{N_l}g_{l,e}}
     {g_{l,e}^{\mathsf T}\Sigma_lg_{l,e}}. \tag{31.3}
\]

This \(\tau\) is endpoint retention, not source-energy retention and not an
inverse-conditioning constant.  If contexts provide a measurement map
\(M_l\) with noise metric \(V_l\), the separate acquisition condition is

\[
\gamma_{l,e}=
\inf_{\substack{h\in N_l\\ |g_{l,e}^{\mathsf T}h|=1}}
\|V_l^{-1/2}M_lh\|>0. \tag{31.4}
\]

Equations (31.1)--(31.4) are necessary and sufficient at the tangent level.
They also prove that Section 30 cannot certify an r55 provider.  An enormous
ambient kernel does not force intersection with a fixed 2,099-dimensional
lawful image: whenever the complete interface has rank at least 2,099, one
can choose a 2,099-plane transverse to its kernel, or instead choose one that
intersects it, without changing any ambient dimension count.  The
all-distinct monomial proves only the ambient statement.  Until a restricted
minor of \(J_l\), or an explicit lawful tangent in \(N_l\), is exhibited, its
use as evidence for r55 future completion is closed.

## 32. Z22 is a concrete transient, but not an r55-intersection proof

The deployed connected post-reset residual has the precise convention

\[
Z=\operatorname{offdiag}\operatorname{sym}\!\left(
K22_{\rm target}-K22_{\rm CP4}
-\alpha\{vv^{\mathsf T}+2C\odot C\}\right),
\qquad \langle Z,P22\rangle=0.
\]

Its minimal pure-22 lift has only \(iijj\) entries.  Writing \(w_a\) for row
\(a\) of a dense transition \(W\), direct polarization gives

\[
\begin{aligned}
\delta K31_{ab}&=3(w_a^{\odot2})^{\mathsf T}Z(w_a\odot w_b),\\
\delta k4_a&=3(w_a^{\odot2})^{\mathsf T}Zw_a^{\odot2},\\
\delta K22_{ab}&=(w_a^{\odot2})^{\mathsf T}Zw_b^{\odot2}
 +2(w_a\odot w_b)^{\mathsf T}Z(w_a\odot w_b).
\end{aligned} \tag{32.1}
\]

These identities establish a lawful one-transition future action.  They do
not establish that \(Z\) lies in the r55 NCQC inverse-fiber image; \(Z22\) is
a separate D11 transient and must remain a separate graph node.

For a target K31 costate \(\Lambda_{31}\), define

\[
W2=W\odot W,\qquad B=W\Lambda_{31}^{\mathsf T},
\qquad M=W\odot B.
\]

Then the entire target-conditioned K31 action factors as

\[
\chi_Z=3\langle ZW2,M\rangle
      =3\langle Z,MW2^{\mathsf T}\rangle. \tag{32.2}
\]

Hence the exact source covector is

\[
G_Z^{\rm raw}=3P_{\mathcal Z}\operatorname{sym}(MW2^{\mathsf T}), \tag{32.3}
\]

where \(P_{\mathcal Z}\) enforces symmetry, zero diagonal, collision
convention, and the already-subtracted pairing/CP4 birth space.  Let
\(N_Z=\ker O^{\rm complete}|_{\mathcal Z}\), and put
\(\bar G_{Z,e}=P_{N_Z}G_{Z,e}^{\rm raw}\).  The exact fixed-endpoint rank is
one iff \(\bar G_{Z,e}\ne0\); the panel rank is the Gram rank of the
\(\bar G_{Z,e}\).  With a lawful source covariance \(\Sigma_Z\), its
endpoint-conditioned retention is

\[
\tau_{Z,e}=
\frac{\bar G_{Z,e}^{\mathsf T}\Sigma_Z\bar G_{Z,e}}
     {(G_{Z,e}^{\rm raw})^{\mathsf T}\Sigma_ZG_{Z,e}^{\rm raw}}. \tag{32.4}
\]

The local birth complement against a 2,048-direction CP4 carrier and one
pairing direction has dimension at least

\[
\binom{1024}{2}-2048-1=521727.
\]

Price differentiation at zero correlation also proves that the source is
generically nonzero: the true K22 derivative is \(\mu^2\), whereas the
pairing derivative is zero.  Neither fact proves \(\bar G_Z\ne0\), because
the complete star/CP4/depth arena is larger than the birth quotient.

The local acquisition price of (32.2) is one additional dense product if
\(W\Lambda_{31}^{\mathsf T}\) is already available, and two otherwise.  This
is a **per-active-layer** price, not a network-total price.  More exactly, if
\(q_l\) is the minimum separation rank of the endpoint mask after quotienting
CP4, pairing, and diagonal-invisible masks, then

\[
C_Z=\sum_l\frac{q_l}{1024}+C_{\rm context}+C_{\rm hook}. \tag{32.5}
\]

The generic full D10 context has \(q_l\) of order 1,024 (the existing
transversality bound is \(q_l\ge947\)); ten layers therefore cost at least
\(9470/1024\approx9.25\) even when the reverse context is owned.  Under the
base-48/cap-49 one-equivalent slack, the exact reopen is
\(\sum_lq_l\le1024\), after all context/hook costs, or reuse of the required
dense products from an already billed path.  A single chosen transition can
fit in one product, but is not the complete depth feature and needs its own
value receipt.

The crossed K22 action in (32.1) is worse: its generic diagonal-completion
rank is at least 980 and it has no certified one-product factorization.
The full Z22 provider is therefore closed under the present cap.  The
strict-future K31 scalar remains **open only under complete-arena Schur
novelty plus the network-total rank/reuse gate (32.5)**.

## 33. Cubic reset jet: lawful AQ image, unresolved complete-interface quotient

For

\[
g(z)=b+a^{\mathsf T}z+z^{\mathsf T}Mz-\operatorname{tr}M
\]

and a lawful perturbation \(h(z)\), the raw cubic reset birth into the AQ
\(H_{\le2}\) state and its complete-old quotient are

\[
\widetilde b_{3,l}(h)=\frac16P_{\le2}[\delta'(g_l)h_l^3],
\qquad
b_{3,l}^{\rm new}=(I-P_{\rm old})\widetilde b_{3,l}(h). \tag{33.1}
\]

This is a strict-future state birth, not the signed H3 defect below and not a
same-site K31 action.  Its zero-length contribution must be removed.  If
\(T_s\) is the AQ transition and \(\lambda_{j+1,e}^{\rm future}\) the compiled
strict-future costate, its endpoint action is exactly

\[
\Xi_{3,e}(h)=\sum_j
\langle\lambda_{j+1,e}^{\rm future},b_{3,j}^{\rm new}(h)\rangle, \tag{33.2}
\]

equivalently the sum of the propagated terms
\(\langle B_k,C_kT_{k-1}\cdots T_{j+1}b_{3,j}^{\rm new}\rangle\) over \(j<k\).

Let \(\widetilde B_{3,l}=D\widetilde b_{3,l}|_{\mathcal F_{55}}\),

\[
J_{3,l}=O_l^{\rm complete}\widetilde B_{3,l},\qquad
N_{3,l}=\ker J_{3,l},\qquad
g_{3,l,e}=\widetilde B_{3,l}^{*}\lambda_{l+1,e}^{\rm future}.
\]

The cubic jet gives a nonzero lawful completion precisely when

\[
P_{N_{3,l}}g_{3,l,e}\ne0. \tag{33.3}
\]

Its endpoint rank and \(\tau\) are therefore the Gram rank and covariance-
weighted projection ratios of Section 31 with \((N,g)=(N_3,g_3)\).  The
formal factor \((I-P_{\rm old})\) in (33.1) does not prove nonzero residue:
the old complete AQ/depth interface may span the entire lawful image of
\(\widetilde B_3\).  A restricted minor or one explicit residual tangent is still
required.

There are two sharp local suppressions.  In a scalar cavity eigenmode the
cubic coefficient is

\[
-10 f'_{S_i}(0)\lambda_i^3.
\]

It vanishes exactly at a symmetric threshold whenever symmetry forces
\(f'_{S_i}(0)=0\).  More generally, with
\(\mu=\lambda_{\max}/(\sum_i\lambda_i^2)^{1/2}\), the flat centered
Edgeworth source bound is

\[
\frac{\|b_3\|}{\|b_1\|}\le 2.82\mu^3. \tag{33.4}
\]

For a flat r55 spectrum this is about \(0.0069\) per reset.  This is a source
amplitude ceiling, not endpoint \(\tau\); even coherent addition cannot
replace the complete-arena projection test.

The generic acquisition cost is decisive.  The r55 AQ state width is

\[
d=1+55+\frac{55\cdot56}{2}=1596.
\]

One dense packed AQ mix costs \(d/1024=1.55859375\) GEMM-equivalents.  A
forward and reverse stream over ten active layers costs \(31.171875\), and
over fourteen layers \(43.640625\); even one ten-layer stream costs
\(15.5859375\).  Therefore a standalone oriented cubic jet is closed under
one total residual GEMM.  A common-slope collapse to one already transported
word is a lawful exception only if an algebraic identity proves both its
orientation and endpoint costate remain in that one-dimensional module;
without that identity it cannot be credited as a cheap provider.

## 34. Signed H3 defect: outside AQ, exact action, generic carriage closed

The genuine cubic object is the residual signed Wiener tensor

\[
Z_3=I_3(T_{\rm true}-T_{\rm CP3/assembled})
    \in\operatorname{Sym}^3(\mathbb R^{55}),
\qquad \dim\operatorname{Sym}^3(\mathbb R^{55})=\binom{57}{3}=29260.
\]

It is not an \(H_{\le2}\) AQ state and must not be merged with the cubic jet.
For the quadratic preactivation
\(g=b+a^{\mathsf T}z+z^{\mathsf T}Mz-\operatorname{tr}M\), its exact ReLU
reset source is the distributional expectation

\[
T=\mathbb E\!\left[
 \delta'(g)(a+2Mz)^{\otimes3}
 +6\delta(g)J_{a+2Mz}(M)
\right], \tag{34.1}
\]

where \(J_x(M)\) is the fully symmetrized insertion of one vector and one
matrix.  The nonlinear dependence of the level-set weights on \(M\) prevents
(34.1) from closing in a fixed finite polynomial word algebra.

The first K31 reverse action is nevertheless tensorially exact.  In the local
H1/H2 tangent notation \((u,N;r,R)\), with transformed cubic component \(Y\),

\[
\delta\kappa_{31}=\langle Z_3,G_{Z_3}\rangle+\langle Y,G_Y\rangle, \tag{34.2}
\]

\[
\begin{aligned}
G_{Z_3}={}&18\operatorname{Sym}(u^{\otimes2}\otimes r)\\
&+72\{J_{Ru}(N)+J_{Nu}(R)+2J_u(N\circ R)
        +J_{Nr}(N)+J_r(N^2)\},\\
G_Y={}&6u^{\otimes3}+72\{J_{Nu}(N)+J_u(N^2)\}.
\end{aligned} \tag{34.3}
\]

This supplies the exact endpoint covector, not a cheap source provider.  Let
\(\widetilde B_{H3}\) be the raw lawful reset-source map before subtraction
of the complete CP3/old interface, and let

\[
J_{H3}=O^{\rm complete}\widetilde B_{H3},\qquad
N_{H3}=\ker J_{H3},
\]

and let \(g_{H3,e}\) be the pullback of (34.2).  A novel endpoint direction
exists iff \(P_{N_{H3}}g_{H3,e}\ne0\); endpoint rank and \(\tau\) again follow
from the projected-covector Gram formulas of Section 31.  Neither the ambient
29,260 dimension nor a source-norm estimate establishes this condition.

In particular, the external H2 shadow sees only Gram quantities
\(6\langle Z_i,Z_j\rangle\), invariant under \(Z\mapsto-Z\), while the signed
action (34.2) flips.  This proves nonidentifiability from H2 alone.  It does
*not* prove a kernel of the complete CP3/depth interface, whose deployed cap
is 16,384 and may carry oriented information on the actual reset image.
Accordingly the old sign witness is closed as evidence of full-interface
novelty until an explicit CP3-Schur residual or restricted-rank theorem is
given.

Generic transport is far outside budget: one dense mix of the r55 cubic state
already costs

\[
\binom{57}{3}/1024=28.57421875
\]

GEMM-equivalents.  A shared symmetric-CP dictionary of rank \(k\) reduces the
first-tangent price to about \(4k/1024\) per layer; one total GEMM over ten
layers permits \(k\le25\), and over fourteen permits \(k\le18\).  Such an arm
is admissible only if its **endpoint-weighted** best-rank tail loses at most
13.8% of the required value, after full old-span residualization.

An oblivious CountSketch gives the independent sharp alternative

\[
\operatorname{Var}\widehat{\langle Z_3,G\rangle}
\le\frac2k\|Z_3\|^2\|G\|^2.
\]

Writing

\[
\mathcal C=
\frac{\mathbb E\|Z_3\|^2\|G\|^2}
     {\mathbb E\langle Z_3,G\rangle^2},
\]

correlation at least \(0.92845\) requires \(k\ge12.49\mathcal C\).  At cost
\(2k/1024\) per layer, the one-GEMM ten-layer envelope gives \(k\le51\), hence
requires \(\mathcal C\le4.08\).  These are novelty/fidelity gates, not
promises: without the complete-interface Schur receipt and one of these tail
certificates, generic H3 carriage is **closed**.

## 35. Checkpoint-9 decisions and graph split

The three mechanisms have different mathematical status and must not share a
single “future completion” node.

1. **R55 complete-interface intersection — OPEN, unproved.**  The exact gate
   is \(\operatorname{rank}J_l<2099\), followed by
   \(P_{\ker J_l}g_e\ne0\).  Section 30's ambient all-distinct witness is
   closed as evidence for this claim.  Quantitative \(\tau\), context
   conditioning \(\gamma\), and context acquisition cost remain separate
   children.

2. **Z22 birth and task factorization — PROVED.**  Equations (32.1)--(32.3)
   give a nonzero lawful transient and its exact one-step K31 scalar.
   **Z22 endpoint novelty — OPEN:** the complete-arena projected covector and
   \(\tau_Z\) are not known.  **Z22 acquisition — GENERIC NO-GO / CONDITIONAL
   REOPEN:** one product is required per active layer even when the reverse
   product is reused; ten-layer generic cost is at least \(9.25\) equivalents.
   It fits one total equivalent only if \(\sum_lq_l\le1024\), the dense action
   is already billed/reused, or a separately certified single-cut feature is
   used.  **Full Z22 recurrence — CLOSED** because crossed K22 has generic
   diagonal-completion rank at least 980.

3. **AQ cubic-jet membership — PROVED; nonzero quotient — OPEN.**  It is a
   lawful \(H_{\le2}\) strict-future birth, but it is exactly zero under the
   symmetric-slope condition and its complete-interface Schur residual is
   unproved.  **Generic acquisition — CLOSED** at 31.171875 GEMM-equivalents
   for ten-layer forward/reverse carriage.  A one-word exception requires a
   common-slope cyclic-module identity, not merely a source-amplitude bound.

4. **H3 defect action — PROVED; source novelty — OPEN.**  Equations
   (34.1)--(34.3) give the true signed source and endpoint action.  The H2
   sign ambiguity is closed as a proof against the full CP3 interface.
   **Generic carriage — CLOSED.**  A compressed arm reopens only with a
   complete-interface Schur receipt plus the rank-tail or CountSketch gate in
   Section 34.

The next purely analytic receipts are now sharply identified: (i) a nonzero
restricted minor of \(O^{\rm complete}\Psi|_{\mathcal F_{55}}\), or a lawful
explicit vector in its kernel; (ii) the complete-arena adjoint projection of
the Z22 covector (32.3); (iii) a proof or refutation of the one-word
common-slope module for (33.1); and (iv) the rank of the actual reset-source
image modulo the deployed 16,384-direction CP3 interface.  No ambient
dimension or source-energy argument substitutes for any of these four
receipts.

## 36. Exact constant-slope cubic-jet projection and one-word obstruction

The proposed common-slope loophole can be audited without a numerical rank
test.  Let

\[
q_H(z)=z^{\mathsf T}Hz-\operatorname{tr}H.
\]

For any sufficiently regular scalar \(F\), Gaussian integration by parts
gives the exact second-chaos projection

\[
P_2F=q_{A_F},\qquad A_F=\frac12\mathbb E[\nabla^2F(z)]. \tag{36.1}
\]

Applying (36.1) to \(F=q_H^3\), using
\(\mathbb E[q_Hzz^{\mathsf T}]=2H\),
\(\mathbb E q_H^2=2\operatorname{tr}H^2\), and
\(\mathbb E q_H^3=8\operatorname{tr}H^3\), yields

\[
P_{\le2}[q_H^3]
=8\operatorname{tr}H^3
{}+q_{\,24H^3+6\operatorname{tr}(H^2)H}. \tag{36.2}
\]

Therefore, if the reset slope is genuinely constant \(c\) on the latent
state, the raw cubic jet is

\[
\widetilde b_3
=\frac{4c}{3}\operatorname{tr}H^3
{}+q_{\,4cH^3+c\operatorname{tr}(H^2)H}. \tag{36.3}
\]

This is a sharp algebraic correction to the informal “one common slope means
one word” claim.  Even in the most favorable constant-weight case the matrix
birth lies in \(\operatorname{span}\{H,H^3\}\).  It collapses to the original
orientation alone exactly when

\[
H^3\in\operatorname{span}\{H\}, \tag{36.4}
\]

equivalently, for symmetric \(H\), all nonzero eigenvalues of \(H\) have the
same squared magnitude.  In that exceptional case the jet creates no new
orientation beyond the original hidden tangent; its novelty is exactly the
complete-interface test (31.2), not a new cubic-provider credit.  Otherwise
the raw matrix module has dimension at least two before old-span projection.

For a nonconstant weight \(w(z)=F(g(z))\), (36.1) gives the exact audit
formula

\[
\begin{aligned}
2A={}&\mathbb E\big[
q_H^3\{F''(g)\nabla g\nabla g^{\mathsf T}
       +F'(g)\nabla^2g\}\\
&+3F'(g)q_H^2\{\nabla g\nabla q_H^{\mathsf T}
                 +\nabla q_H\nabla g^{\mathsf T}\}\\
&+6F(g)q_H\nabla q_H\nabla q_H^{\mathsf T}
 +3F(g)q_H^2\nabla^2q_H\big], \tag{36.5}
\end{aligned}
\]

with \(P_2[wq_H^3]=q_A\).  Here
\(\nabla g=a+2Nz\), \(\nabla^2g=2N\),
\(\nabla q_H=2Hz\), and \(\nabla^2q_H=2H\).  Thus a genuine ReLU level-set
weight introduces noncommuting words in \(N\) and \(H\).  For a generic
simple-spectrum \(N\) and an \(H\) whose off-diagonal graph is connected in
the eigenbasis of \(N\), Burnside's theorem gives

\[
\operatorname{alg}^*(N,H)=M_{55}.
\]

Consequently there is no universal exact one-dimensional word module for
(36.5).  A cheap one-word jet now requires one of three explicit receipts:
(a) the constant-weight spectral degeneracy (36.4), (b) a commuting/reducible
algebra plus endpoint annihilation of every omitted word, or (c) an already
billed complete costate contraction at birth.  A shared scalar slope by
itself is insufficient.

## 37. Tensor-free implicit H3 action by a one-dimensional resolvent integral

There is a stronger way to consume the signed H3 defect than transporting a
29,260-coordinate tensor.  Keep the quadratic preactivation notation

\[
g(z)=b+a^{\mathsf T}z+z^{\mathsf T}Mz-\operatorname{tr}M,
\qquad x(z)=\nabla g=a+2Mz,
\]

and let \(G\in\operatorname{Sym}^3(\mathbb R^r)\) be the **strict-future,
complete-old-residualized** endpoint costate from (34.2)--(34.3).  The desired
true source action is

\[
s_G=\left\langle
\mathbb E[\delta'(g)x^{\otimes3}+6\delta(g)J_x(M)],G
\right\rangle. \tag{37.1}
\]

Define, for real \(t\),

\[
D_t=I-2itM,
\]

\[
\varphi(t)=e^{it(b-\operatorname{tr}M)}
\det(D_t)^{-1/2}
\exp\!\left[-\frac{t^2}{2}a^{\mathsf T}D_t^{-1}a\right], \tag{37.2}
\]

\[
\mu_t=D_t^{-1}a,
\qquad \Sigma_t=4MD_t^{-1}M. \tag{37.3}
\]

Completing the complex Gaussian square proves

\[
\mathbb E[e^{itg}x^{\otimes3}]
=\varphi(t)\{\mu_t^{\otimes3}
 +3\operatorname{Sym}(\mu_t\otimes\Sigma_t)\},
\]

\[
\mathbb E[e^{itg}J_x(M)]=\varphi(t)J_{\mu_t}(M).
\]

Using
\(\delta(y)=(2\pi)^{-1}\int e^{ity}\,dt\) and
\(\delta'(y)=(2\pi)^{-1}\int it e^{ity}\,dt\), (37.1) becomes the exact
one-dimensional formula

\[
\boxed{
s_G=\frac1{2\pi}\int_{-\infty}^{\infty}\!\varphi(t)
\left[
it\{G[\mu_t,\mu_t,\mu_t]+3G[\mu_t,\Sigma_t]\}
+6\langle G,J_{\mu_t}(M)\rangle
\right]dt .} \tag{37.4}
\]

Here \(G[\mu,\Sigma]=\sum_{ijk}G_{ijk}\mu_i\Sigma_{jk}\).  Conjugate
symmetry reduces (37.4) to twice the real part over \(t>0\).  The structured
representation (34.3) is crucial: contractions of
\(J_v(S)\) with \(\mu^{\otimes3}\) and
\(\operatorname{Sym}(\mu\otimes\Sigma)\) use only vector--matrix products and
traces.  No H3 tensor is formed.

If \(M=U\operatorname{diag}(\lambda_j)U^{\mathsf T}\), every integrand term
uses the scalar resolvents \((1-2it\lambda_j)^{-1}\).  The exact modulus

\[
|\varphi(t)|=
\prod_j(1+4t^2\lambda_j^2)^{-1/4}
\exp\!\left[-\frac{t^2}{2}
\sum_j\frac{(U^{\mathsf T}a)_j^2}{1+4t^2\lambda_j^2}\right] \tag{37.5}
\]

gives a deterministic truncation certificate.  For any cutoff \(T\), a
lawful absolute tail bound is

\[
\epsilon_T\le\frac1\pi\int_T^\infty |\varphi(t)|\,\|G\|_F
\left[t\{\|\mu_t\|^3+3\|\mu_t\|\|\Sigma_t\|_F\}
      +6c_J\|\mu_t\|\|M\|_F\right]dt, \tag{37.6}
\]

where \(c_J\) is fixed by the symmetrization convention.  If \(M\) has
\(s\) eigenvalues with \(|\lambda_j|\ge\lambda_*\), the determinant factor
alone is at most \((1+4t^2\lambda_*^2)^{-s/4}\); on the invertible support,
\(\|\mu_t\|=O(t^{-1})\) and \(\|\Sigma_t\|_F=O(t^{-1})\), so the tail is
integrable with order \(T^{-s/2}\).  A null-space component of \(a\) instead
supplies Gaussian decay in (37.5).

The operation-count gate is also explicit.  With \(q\) quadrature nodes,
one eigensystem per physical unit, latent rank \(r\), width \(n\), and \(L\)
active layers, the leading count relative to an \(n^3\) GEMM is

\[
C_{\rm imp}
\asymp L\left(
c_{\rm eig}\frac{r^3}{n^2}
+c_{\rm node}\frac{q r^2}{n^2}
\right)+C_{\rm costate}. \tag{37.7}
\]

At \(r=55,n=1024,L=10\), eigensystems without reuse already contribute
\(10r^3/n^2\approx1.587\) before constants.  Thus base-48/cap-49 requires
reuse of the reset eigensystems.  With that reuse and unit leading node
constant, the raw one-equivalent bound is \(q\le34\), before charging the
strict-future costate.  Formula (37.6), not an assumed quadrature depth,
decides whether that node count is faithful.

Finally, this is a provider only for the **unowned residual**

\[
s_G^{\rm new}=s_G-\langle T_{\rm CP3/assembled},G\rangle
\]

after complete response-space Schur projection.  It receives no same-site
K31 credit and still obeys the r55 functional-observability condition
\(\ker D\mathcal O\subseteq\ker Ds_G^{\rm new}\).  The sharp GO gate is:
(i) nonzero projected endpoint scalar, (ii) truncation plus core/costate error
bounded by (37.6) tightly enough for the endpoint-value threshold, and
(iii) the full marginal count (37.7) at most one.  Subject to those three
receipts, (37.4) is an exact label-free implicit-derivative cubic provider;
without them it remains a mathematically valid but unpriced action oracle.

### 37.1 Convention, structure, and ownership audit

The Fourier and symmetrization constants have been independently checked.
With averaged three-slot symmetrization \(J\),

\[
\langle J_v(A),J_\mu(B)\rangle
=\frac13\left[(v^{\mathsf T}\mu)\operatorname{tr}(AB)
 +v^{\mathsf T}(AB+BA)\mu\right]. \tag{37.8}
\]

Moreover
\(\operatorname{Sym}(u\otimes u\otimes r)=J_r(uu^{\mathsf T})\) and
\(u^{\otimes3}=J_u(uu^{\mathsf T})\).  Thus the full \(G_Z/G_Y\) query is a
six-node plus three-node Jordan list, and (37.8) evaluates every Fourier node
in \(O(r^2)\) arithmetic after basis preparation.  Since averaged
symmetrization is an orthogonal projection,
\(\|J_\mu(M)\|_F\le\|\mu\|\|M\|_F\), so (37.6) may take \(c_J=1\).

This node cost assumes more than an eigendecomposition: all structured
costate matrices must be rotated into the eigenbasis of \(M\) once, after
which \(\Sigma_t\) is kept diagonal there rather than materialized by an
\(O(r^3)\) conjugation at every node.  Those setup rotations belong in
\(C_{\rm costate}\).

The binding deployed ownership test is stricter still.  The resident
48-equivalent schedule already pays for a D14 base stream and a 20-equivalent
D10 suffix reverse word.  The costate \(G_l\) in (37.4) is free only if, at
the identical reset site and covariance frame, it is a literal mask/view or
scalar combination of that resident D10 costate and obeys the same backward
recurrence and boundary collapse.  Sharing \(W,Q\), latent rank, or a
post-hoc algebraic conversion is not reuse.  A generic cubic future gradient
is a new cyclic-module row and reopens the 20-equivalent bundle, immediately
closing cap-49.  Therefore the present disposition of the implicit provider
is **mathematical GO, deployment HOLD**: exact action and error certificate
are solved, but `G_ownership` is an independent zero-credit gate before the
quadrature-node allowance in (37.7) is relevant.

## 38. Complete one-step H3-to-H4 adjoint and the exact D10 reuse seam

The K31-only query in Section 34 extends to every old fourth-order emission.
Write \(X(a,M)=I_1(a)+I_2(M)\), and define

\[
\mathcal J(a;N,P)
=J_{Pa}(N)+J_{Na}(P)+2J_a(N\circ P),
\]

\[
\begin{aligned}
\mathcal G((a,M),(b,N),(c,P))={}&
6\operatorname{Sym}(a\otimes b\otimes c)\\
&+12\{\mathcal J(a;N,P)+\mathcal J(b;M,P)
                     +\mathcal J(c;M,N)\}. \tag{38.1}
\end{aligned}
\]

The Wiener product rule gives the universal slot identity

\[
\mathbb E[I_3(Z)X(a,M)X(b,N)X(c,P)]
=\langle Z,\mathcal G((a,M),(b,N),(c,P))\rangle. \tag{38.2}
\]

Indeed the only surviving parity classes are H1/H1/H1, with coefficient
\(3!=6\), and H1/H2/H2, with

\[
\mathbb E[I_3(Z)I_1(a)I_2(N)I_2(P)]
=12\langle Z,\mathcal J(a;N,P)\rangle.
\]

Equation (38.1) therefore generates all collision types without a new tensor:

- K4 uses four identical slot pullbacks;
- K31 uses three copies of
  \(\mathcal G((u,N),(u,N),(r,R))\) for the U-source and one copy of
  \(\mathcal G((u,N),(u,N),(u,N))\) for the V-source, reproducing
  (34.3);
- K22 uses two copies of
  \(\mathcal G((u,N),(r,R),(r,R))\) for the U-source and two copies of
  \(\mathcal G((r,R),(u,N),(u,N))\) for the V-source.

Let \(L_l^{3\to4}\) be the one-transition tangent from a reset-local H3 birth
to the complete old D11 fourth-order product state, and let its exact resident
Bellman costate at the next seam be

\[
Q_l^{(4)}=(Q4_l,Q31_l,Q22_l,\ldots).
\]

Then the chain rule proves

\[
\langle Q_l^{(4)},L_l^{3\to4}Z_l\rangle
=\langle (L_l^{3\to4})^*Q_l^{(4)},Z_l\rangle, \tag{38.3}
\]

with the complete local query

\[
G_l=(L4_l)^*Q4_l+(L31_l)^*Q31_l+(L22_l)^*Q22_l+\cdots. \tag{38.4}
\]

Every term in (38.4) is a Jordan list by (38.1), so it can be inserted into
the implicit resolvent action (37.4).  This proves the **immediate full-seam
reuse theorem**: when the deployed reverse tape literally owns the complete
product-state costate at the synchronized post-W/pre-reset seam, \(G_l\) is a
local adjoint application, not an independent backward cyclic row.

The implementation premise is not yet certified.  D11's forward seam does
carry `pre_k4,pre_k31,pre_k22` through `_full_relu_reset`, followed by
`k4_target,k31_target,k22_target` through `_reset_stars`, `_cp4_reset`, and
the next `_push_quartic`.  An exact reverse of that full tape would expose the
coordinates in (38.4).  The current D10 object \(b_l\), however, is specified
only abstractly as a `q4_perp` word; no receipt yet proves it is a literal
view of all \((Q4,Q31,Q22)\) coordinates at this seam.  In particular,
\((L31)^*Q31\) alone is incomplete and cannot receive full value credit.

This also draws the exact boundary of the theorem.  It covers the first
H3-to-old-H4 emission and all of its K4/K31/K22 faces.  Any component that
persists through an unrepresented H3 state and emits only at a later cut is
outside \(L^{3\to4}\); it needs a new future row or an endpoint tail bound.
The graph must therefore separate:

- `H3_IMMEDIATE_FULL_SEAM_REUSE` — **solved theorem** by (38.1)--(38.4);
- `D10_PRODUCT_COSTATE_LITERAL_OWNERSHIP` — **open implementation receipt**;
- `H3_IMPLICIT_SOURCE_QUADRATURE_COST` — **conditional** on Sections 37.6--37.7;
- `H3_HIDDEN_PERSISTENCE_TAIL` — **open**, generic carriage closed absent a
  compression/tail certificate.

## 39. Exact hidden-H3 persistence series and sharp tail gate

The omission left by the one-step theorem has an exact operator form.  On
normalized third-chaos kernels define the reset operator by

\[
6\langle\mathcal R_jZ,G\rangle
=\mathbb E[1_{\{g_j>0\}}I_3(Z)I_3(G)]. \tag{39.1}
\]

Let \(P_j^{\perp}\) project away the complete old CP3/depth interface at cut
\(j\), and define the unowned residual transition

\[
R_j=P_{j+1}^{\perp}\mathcal R_jP_j^{\perp},
\qquad \rho_j=\|R_j\|. \tag{39.2}
\]

Let \(g_j=(L_j^{3\to4})^*Q_{j+1}^{(4)}\) be the complete immediate query from
(38.4).  For one residual birth \(z_l\), its exact strict-future value is the
finite Volterra series

\[
V_l(z_l)=\sum_{m=l}^{L-1}
\left\langle g_m,
R_{m-1}R_{m-2}\cdots R_lz_l\right\rangle, \tag{39.3}
\]

where the product is the identity at \(m=l\).  Section 38 computes the first
term.  The omitted hidden-persistence tail obeys the sharp operator bound

\[
|E_l^{\rm persist}|
\le\|z_l\|
\sum_{m=l+1}^{L-1}\|g_m\|
\prod_{j=l}^{m-1}\rho_j. \tag{39.4}
\]

For an endpoint-relative certificate define

\[
\lambda_l=\frac{|\langle g_l,z_l\rangle|}
                  {\|g_l\|\|z_l\|},
\qquad
\kappa_{m|l}=\frac{\|g_m\|}{\|g_l\|}.
\]

Whenever \(\lambda_l>0\),

\[
\frac{|E_l^{\rm persist}|}{|\langle g_l,z_l\rangle|}
\le\frac1{\lambda_l}
\sum_{m=l+1}^{L-1}\kappa_{m|l}
\prod_{j=l}^{m-1}\rho_j. \tag{39.5}
\]

This leverage factor is unavoidable: a small source-transition norm alone
does not control relative endpoint error when the immediate scalar is nearly
orthogonal to its costate.  In the uniform special case
\(\rho_j\le\rho<1\), \(\kappa_{m|l}\le\kappa\), (39.5) reduces to

\[
\frac{|E_l^{\rm persist}|}{|\langle g_l,z_l\rangle|}
\le\frac{\kappa}{\lambda_l}
\frac{\rho(1-\rho^{L-l-1})}{1-\rho}. \tag{39.6}
\]

There is a useful exact ideal limit.  Equation (39.1) is a positive
contraction, so \(0\preceq\mathcal R_j\preceq I\).  If the ReLU gate is
conditionally independent of third chaos, then
\(\mathcal R_j=p_jI\).  Since the collective scalar transport \(p_jI\) is
old CP3-owned, (39.2) gives \(R_j=0\): all value is in the one-step term.
For a correlated gate, \(\rho_j\) is precisely the noncollective reset
leakage after the complete CP3 quotient.

Without a cavity/leakage receipt the universal bound is only
\(\rho_j\le1\), and \(\lambda_l\) can be arbitrarily small.  Hence no
nonzero fidelity guarantee for the one-step provider follows from ReLU
contractivity alone.  `H3_HIDDEN_PERSISTENCE_TAIL` is promoted only when the
right side of (39.4), aggregated over births in the endpoint covariance
metric, is below the assigned unexplained-value budget; (39.5) is the sharp
per-birth relative gate.  This turns the previously qualitative hidden-row
caveat into a measurable leakage-and-leverage theorem.

## 40. Exact centered parity: after the owned Jordan term, novelty starts at cubic order

The reset source has a useful nonperturbative symmetry.  Write

\[
g_{b,M}(z)=b+a^{\mathsf T}z+z^{\mathsf T}Mz-\operatorname{tr}M,
\qquad
F_{b,M}(z)=(g_{b,M}(z))_+,
\]

\[
T(b,M)=\mathbb E[D_z^3F_{b,M}(z)].
\]

Since

\[
g_{b,M}(-z)=-g_{-b,-M}(z)
\]

and \((-x)_+=x_+-x\), while \(D^3g_{-b,-M}=0\), the third-order chain rule
under \(z\mapsto-z\) gives the exact identity

\[
\boxed{T(b,M)=-T(-b,-M).} \tag{40.1}
\]

In particular,

\[
T(0,M)=-T(0,-M), \tag{40.2}
\]

so **every even power of \(M\) vanishes exactly at zero effective bias**.
This strengthens the isolated observation \(T^{(2)}=0\): it is an all-orders
parity theorem.

Let \(T^{[k]}\) denote the coefficient of \(\epsilon^k\) in
\(T(b,\epsilon M)\).  At \(b=0\),

\[
T(0,M)=T^{[1]}(M)+T^{[3]}(M)+T^{[5]}(M)+\cdots. \tag{40.3}
\]

The order-one term is precisely the collective/Jordan span generated by
\(e^{\otimes3}\), \(\operatorname{Sym}(e^{\otimes2}\otimes Me)\), and
\(J_e(M)\).  Therefore, conditional on the existing H31/CP3 ownership audit
proving that span is already present, the lawful true-minus-assembled signed
H3 source begins at \(O(M^3)\), not \(O(M^2)\).  At nonzero standardized bias
\(\alpha=b/\|a\|\), joint parity permits the stronger
\(O(|\alpha|M^2)\) branch; its exact second-order coefficients are all odd in
\(\alpha\).

Taylor's theorem makes the source gate explicit.  At \(b=0\),

\[
\left\|T(0,M)-T^{[1]}(M)\right\|
\le \frac16
\sup_{0\le s\le1}
\left\|D_M^3T(0,sM)[M,M,M]\right\|. \tag{40.4}
\]

The derivative in (40.4) is obtained by differentiating the resolvent
integrand (37.4); the same characteristic modulus (37.5) supplies its
integrable envelope.  There is no uniform numerical constant without a
nondegenerate density/spectral gap, because the ReLU boundary density and its
derivatives can diverge at a singular quadratic form.

For a discriminant-separated diffuse family with
\(\varepsilon=\|M\|_{\rm op}/\|a\|=O(r^{-1/2})\) and a bounded standardized
third derivative in (40.4), the first unowned centered source is
\(O(\varepsilon^3)=O(r^{-3/2})\); at \(r=55\) this is the same
few-per-thousand scale as the earlier flat-spectrum cubic-jet ceiling.  This
is only a source-scale statement, but it supplies a fail-fast rule:

- centered, diffuse cores require a certified coherent endpoint leverage large
  enough to overcome the \(M^3\) suppression;
- a material H3 provider needs a nonzero bias branch, a spectral spike, a
  boundary-density amplification, or a proved ownership mismatch already at
  order one;
- if none is present, Fourier exactness cannot rescue value—the exact oracle
  is computing a parametrically small residual.

Accordingly H3_SOURCE_MAGNITUDE_PARITY is a solved theorem, while the
instance-specific standardized derivative bound and order-one ownership are
separate quantitative gates.

### 40.1 Portfolio calibration and no-double-counting

The current portfolio calibration turns the qualitative tail clauses above
into two numerical terminal conditions.  After whitening by the endpoint H3
value metric, the squared norm of hidden persistence, Fourier truncation,
source approximation, and old-ownership mismatch together must be at most

\[
0.05210
\]

of the true signed-H3 defect energy.  After joint residualization with all
retained AQ/H4 features, the incremental raw-MSE gain must be at least
\(2.86\times10^{-10}\).  The source space to be credited is exactly

\[
(I-P_{S_{\rm old}})\operatorname{span}\{J_4,F_3\},
\]

where \(J_4\) is the AQ H4 score/JVP family and \(F_3\) the absolute H3 source
family.  The AQ JVP contains a bias derivative of H3, not the absolute signed
source; overlap is removed by this joint projector rather than by declaring
the two lanes identical.

## 41. Z22 has the right broad-spectrum source law, and exactly the hard consumption rank

The transient Z22 source admits a sharp small-correlation expansion that
clarifies its qualitative advantage over the centered H3 residual.  For
standard centered ReLU

\[
f(z)=z_+-\frac1{\sqrt{2\pi}},
\qquad
v_0=\frac12-\frac1{2\pi},
\]

Mehler expansion gives, for a bivariate standard Gaussian of correlation
\(\rho\),

\[
\operatorname{Cov}(f(Z_1)^2,f(Z_2)^2)-2\operatorname{Cov}(f(Z_1),f(Z_2))^2
=\frac{\rho}{2\pi}+O(\rho^2), \tag{41.1}
\]

whereas the retained pairing plane has no linear response.  With unequal
preactivation scales \(\sigma_i\), put

\[
a_{ij}=\sigma_i^2\sigma_j^2,\qquad
\bar\rho_w=
\frac{\sum_{i<j}a_{ij}^2\rho_{ij}}
     {\sum_{i<j}a_{ij}^2}.
\]

After the fitted pairing coordinate is removed,

\[
(Z22)_{ij}
=\frac{a_{ij}}{2\pi}(\rho_{ij}-\bar\rho_w)
+O(a_{ij}\epsilon^2),
\qquad \epsilon=\max_{i<j}|\rho_{ij}|. \tag{41.2}
\]

In the full symmetric zero-diagonal convention,

\[
\|Z22\|_F^2
=\frac{2}{(2\pi)^2}
\sum_{i<j}a_{ij}^2(\rho_{ij}-\bar\rho_w)^2
+O\!\left(\epsilon^3\sum_{i<j}a_{ij}^2\right). \tag{41.3}
\]

Equivalently, if \(C^{\rm pre}_{ij}=\sigma_i\sigma_j\rho_{ij}\),
\(D_\sigma=\operatorname{diag}(\sigma_i)\), and
\(v=(\sigma_i^2)_i\), then

\[
Z22^{(1)}
=\frac1{2\pi}\operatorname{offdiag}\!
\left(D_\sigma C^{\rm pre}D_\sigma-\bar\rho_wvv^{\mathsf T}\right).
\tag{41.4}
\]

Thus heterogeneous correlations create a **first-order, dense,
identity-specific covariance congruence**.  This is exactly the kind of
broad-spectrum source that survived the carrier attribution, and it explains
why rank-one, block, and low-CP carriers fail: except for the subtracted
rank-one mean, (41.4) inherits the full correlation geometry.  An
equicorrelated plane is the exact first-order null case.

The same fact blocks additive deployment.  Although (41.4) is formed from
resident covariance and diagonal scales in \(O(n^2)\), its complete K31
consumer uses

\[
3\langle Z22(W\odot W),\,W\odot(W\Lambda_{31}^{\mathsf T})\rangle.
\]

For generic dense \(W,\Lambda_{31}\), quotienting CP4, pairing, and diagonal
masks still leaves separation rank at least 947.  So the source is cheap and
physically well matched, but its endpoint action costs about one dense
product per active layer; this is not removed by recognizing the covariance
congruence.  The crossed-K22 branch retains its separate rank-at-least-980
obstruction.

The exact promotion gate is therefore a replacement/fusion gate, not a
source-existence gate:

1. complete-arena residual angle
   \(\gamma_l=\langle Z_l,\bar G_l\rangle/(\|Z_l\|\|\bar G_l\|)\) must supply
   the required Schur value;
2. either the required \(Z22(W\odot W)\) products are already billed, or the
   network-total quotient ranks satisfy \(\sum_lq_l\le1024\);
3. every omitted K4/K22 emission is included or bounded.

This makes Z22 the strongest current **source-law** candidate, while leaving
its arithmetic as the binding obstruction.  It should be attacked as a swap
into an existing broad-spectrum covariance/q4 contraction, not as another
low-rank additive carrier.

## 42. Exact Z22 fusion/replacement theorem

The only algebraic route around the rank-947 cost is operand replacement in an
already billed application of the same operator.  Put

\[
X_l=W_l\odot W_l,\qquad
Y_l=W_l\odot(W_l\Lambda_{31,l}^{\mathsf T}).
\]

Suppose the deployed path literally evaluates, at the synchronized Z22 seam,

\[
R_l=A_lX_l \tag{42.1}
\]

as one dense product, with \(A_l\) the compressed old K22 pairing/provider
plane.  Linearity gives

\[
(A_l+Z_l)X_l=R_l+Z_lX_l, \tag{42.2}
\]

and therefore the desired scalar difference is

\[
3\langle(A_l+Z_l)X_l,Y_l\rangle
-3\langle A_lX_l,Y_l\rangle
=3\langle Z_lX_l,Y_l\rangle. \tag{42.3}
\]

Thus there are exactly two zero-extra-product implementations:

1. **replacement:** the downstream model uses the full plane \(A_l+Z_l\)
   instead of \(A_l\), and is jointly refit;
2. **scalar differencing:** the old contraction
   \(3\langle A_lX_l,Y_l\rangle\) is already available as a scalar, so (42.3)
   is recovered with only \(O(n^2)\) reductions.

Merely having some product involving \(A_l\), \(W_l\), or covariance is not
reuse.  The right factor must be the identical \(X_l=W_l\odot W_l\), the
left operand slot must accept the transient \(Z_l\) at the same cut and
collision convention, and \(Y_l\) must already be owned.  If the old pairing
path uses a rank-one/Hadamard analytic shortcut rather than the literal dense
product (42.1), inserting a dense Z22 reopens the full product.  Likewise, if
the old matrix \(A_lX_l\) is still independently required downstream, one
summed product cannot return both matrices; (42.2) is then a true replacement,
not a free additive feature.

For the actual reset decomposition, \(A_l\) is the fitted pairing component
and \(A_l+Z_l\) is the complete CP4-subtracted connected K22 plane.  Hence
replacement has a clear physical interpretation: stop compressing that plane
to one pairing coordinate before its existing consumer.  This is also why a
joint refit is mandatory—the operation changes an owned ingredient rather
than adding an orthogonal feature.

The exact fusion gate is consequently:

\[
\boxed{\text{literal }A_l(W_l\odot W_l)\text{ kernel at the Z22 seam}
\quad+\quad
\text{owned }Y_l
\quad+\quad
\text{replacement-safe downstream use}.} \tag{42.4}
\]

If any clause fails, the generic network cost (32.5) applies.  This theorem
turns the vague “absorb Z22 into covariance” idea into a source-code-level
identity: ordinary covariance conjugation \(WCW^{\mathsf T}\) is the wrong
operator and supplies no fusion credit.

## 43. Stronger Z22 exception: contract the transient against a literal Q22 costate

There is one route stronger than the operand fusion in Section 42.  At the
reset seam, regard the complete connected K22 plane as a product-state
coordinate

\[
K22_l=A_l+Z_l,
\]

where \(A_l\) is the CP4/pairing-owned component and \(Z_l\) the discarded
residual.  If the strict-future Bellman functional has the **literal same-site
full-plane costate**

\[
Q22_l=\frac{\partial V}{\partial K22_l},
\]

then the complete first-order value of the transient is simply

\[
\boxed{\delta V_l=\langle Q22_l,Z_l\rangle_F.} \tag{43.1}
\]

This is the adjoint identity behind every expanded future formula.  For
example, the K31-induced contribution

\[
3\langle Z_l(W_l\odot W_l),
          W_l\odot(W_l\Lambda_{31,l}^{\mathsf T})\rangle
\]

is just (43.1) with

\[
X_l=W_l\odot W_l,\qquad
Y_l=W_l\odot(W_l\Lambda_{31,l}^{\mathsf T}),\qquad
Q22_l^{(31)}=3P_{\mathcal Z}\operatorname{sym}(Y_lX_l^{\mathsf T}).
\tag{43.2}
\]

Thus the rank-947 product is the cost of **reconstructing** this costate from
an upstream K31 factorization.  It disappears if \(Q22_l\) itself is already
materialized at the reset-stars seam.  Because \(Z_l\) is also materialized
there, (43.1) costs only \(O(n^2)\) reductions per layer, about \(L/n\)
GEMM-equivalents over \(L\) full-width layers.

The ownership condition is exceptionally strict:

1. the paid D10 reverse object must be a literal full-plane Q22 coordinate at
   the identical pre/post-reset cut, gauge, diagonal convention, and CP4
   subtraction;
2. it must be the costate **before** the old projection
   \(K22\mapsto(\alpha,\mathrm{star},\mathrm{CP4})\).  The exact reverse of a
   model that already discarded \(Z\) has zero sensitivity on \(Z\) and
   cannot certify (43.1);
3. Q22 must include all downstream K4/K31/K22/depth paths, not only the
   K31-induced mask (43.2);
4. the resulting corpus scalar must still be Schur-residualized against the
   complete old response span.

If the resident D10 word is merely an abstract q4-perp factor, an upstream
Q31 object, or a post-compression pairing coordinate, forming the missing
full-plane Q22 reopens the generic rank/cyclic-module cost.  Conversely, if
the proposed Base48 D10 suffix is explicitly defined as this pre-compression
Q22 reverse plane, Z22 becomes a reset-local direct source with essentially
zero marginal arithmetic and no hidden future branch: the full Bellman
costate has already summed those branches.

This is now the single highest-value implementation receipt:

\[
\boxed{\text{D10 b_l is literal pre-compression Q22 at reset-stars}.}
\tag{43.3}
\]

Z22_DIRECT_Q22_CONTRACTION is mathematical GO conditional on (43.3);
without that literal ownership it falls back to Sections 32 and 42.

### 43.1 Concrete ownership verdict and the complete Q22 mask

The seam theorem has now been audited against the deployed decomposition.
Current ownership does **not** satisfy (43.3).  The old reset maps the
transient plane only to

\[
\alpha_l=\frac{\langle K22_l,P22_l\rangle}{\|P22_l\|^2},
\]

so its reverse Q22 row is proportional to \(P22_l\).  By construction
\(\langle Z_l,P22_l\rangle=0\).  Hence the exact old D11 reverse tape has
zero sensitivity to Z22; no old-row novelty can be claimed.

The proposed D10 \(b_l\) is still specified only as an abstract q4-perp word,
not as a concrete reverse tape of the full pre-compression product state.
Literal ownership is therefore **open**, not resident.

For completeness, let \(q_a=w_a^{\odot2}\), \(c_{ab}=w_a\odot w_b\), and
let \(Q=[q_a]\).  A symmetric future K22 costate \(\Lambda_{22}\) pulls back
to the exact source-plane mask

\[
\boxed{
\Omega_{22}
=Q\Lambda_{22}Q^{\mathsf T}
+2\sum_{a,b}(\Lambda_{22})_{ab}c_{ab}c_{ab}^{\mathsf T}.}
\tag{43.4}
\]

Thus the free scalar is \(\langle\Omega_{22},Z\rangle\), not
\(\langle\Lambda_{22},Z\rangle\) across mismatched frames.  The first term is
the uncrossed branch; the second is the crossed branch with generic
diagonal-completion rank at least 980.  Likewise the K31-induced source mask
is \(\Omega_{31}=3P_{\mathcal Z}\operatorname{sym}(Y X^{\mathsf T})\).
Neither upstream \(\Lambda_{31}\), a K31 Bellman plane, nor factors for it
determine the complete \(\Omega_{22}\).

The final graph status is therefore:

- Z22_TRANSIENT_SEAM — solved theorem;
- OLD_PAIRING_Q22_ZERO — solved exact orthogonality;
- D10_Q22_LITERAL_OWNERSHIP — open implementation/design receipt;
- Z22_DIRECT_Q22_CONTRACTION — conditional GO only if the resident array is
  the already-pulled-back \(\Omega_{22}\) at the same frame and lifetime.

## 44. Deterministic 34-node resolvent certificate, and its uniform no-go

The optimistic reused-eigenbasis budget in Section 37 permits at most 34
unit-cost Fourier nodes.  This does not justify an assumed quadrature depth,
but it admits an exact instance-conditioned theorem.

Let \(M=U\operatorname{diag}(\lambda_j)U^{\mathsf T}\),
\(\alpha=U^{\mathsf T}a\), and define

\[
\begin{aligned}
m(t)^2&=\sum_j\frac{\alpha_j^2}{1+4t^2\lambda_j^2},\\
v(t)^2&=16\sum_j\frac{\lambda_j^4}{1+4t^2\lambda_j^2},\\
p(t)&=\prod_j(1+4t^2\lambda_j^2)^{-1/4}
\exp\!\left[-\frac{t^2}{2}
\sum_j\frac{\alpha_j^2}{1+4t^2\lambda_j^2}\right].
\end{aligned}
\tag{44.1}
\]

Then the exact absolute truncation certificate for (37.4) is

\[
E_{\rm tail}(T)=\frac{\|G\|_F}{\pi}
\int_T^\infty p(t)
\left[t\{m(t)^3+3m(t)v(t)\}+6m(t)\|M\|_F\right]dt. \tag{44.2}
\]

It has a closed spectral envelope.  Let
\(S=\{j:\lambda_j\ne0\}\), \(s=|S|\),

\[
A_0=\|P_{\ker M}a\|,\qquad
A_1^2=\sum_{j\in S}\frac{\alpha_j^2}{4\lambda_j^2},\qquad
C_\lambda=\prod_{j\in S}(2|\lambda_j|)^{-1/2}.
\]

If \(A_0=0\),

\[
E_{\rm tail}(T)\le\frac{\|G\|_FC_\lambda}{\pi}
\left[
\frac{A_1^3}{s/2+1}T^{-(s/2+1)}
+\frac{12A_1\|M\|_F}{s/2}T^{-s/2}
\right]. \tag{44.3}
\]

For full-rank r55 the powers are \(T^{-57/2}\) and \(T^{-55/2}\).
If \(A_0>0\), put \(c=A_0^2/2\),
\(m_T=(A_0^2+A_1^2/T^2)^{1/2}\), and

\[
\Psi_\nu(c,T)=\frac12c^{-(\nu+1)/2}
\Gamma\!\left(\frac{\nu+1}{2},cT^2\right).
\]

Then

\[
E_{\rm tail}(T)\le\frac{\|G\|_FC_\lambda}{\pi}
\left[
m_T^3\Psi_{1-s/2}(c,T)
+12m_T\|M\|_F\Psi_{-s/2}(c,T)
\right]. \tag{44.4}
\]

For the core interval \([0,T]\), write the integrand as
\(e^{i\beta t}A(t)\), \(\beta=b-\operatorname{tr}M\).  Its resolvent branch
points are \(t_j=-i/(2\lambda_j)\).  Under
\(t=T(1+z)/2\), define

\[
z_j=-1-\frac{i}{\lambda_jT},\qquad
\rho_*=\min_j\left|z_j+\sqrt{z_j^2-1}\right|>1,
\tag{44.5}
\]

choosing the square-root branch with modulus greater than one.  For any
\(1<\rho<\rho_*\), let \(B_\rho=\sup_{z\in E_\rho}|A(T(1+z)/2)|\).
A 34-point shifted Gauss--Legendre rule has the analytic bound

\[
E_{\rm GL}\le
\frac{4TB_\rho e^{|\beta|h_\rho}}{\pi(\rho-1)}\rho^{-67}, \tag{44.6}
\]

where \(h_\rho\) is the maximum imaginary height of the mapped ellipse.
For large \(|\beta|\), interpolate only \(A\) at 34 shifted Chebyshev nodes
and integrate \(e^{i\beta t}\) times the degree-33 interpolant exactly
(Filon weights).  Then

\[
E_{\rm Filon}\le
\frac{4TB_\rho}{\pi(\rho-1)}\rho^{-33}, \tag{44.7}
\]

independent of the oscillation frequency.  Therefore 34 nodes are certified
for one instance precisely when some \(T,\rho\) obey

\[
E_{\rm tail}(T)+\min(E_{\rm GL},E_{\rm Filon})
\le\epsilon_{\rm allowed}. \tag{44.8}
\]

There is no unconditional 34-node theorem:

1. For any fixed node/weight rule independent of bias, the lawful family
   \(M=0,a=e_1,G=e_1^{\otimes3}\) has exact action
   \[
   s(b)=-\frac{b}{\sqrt{2\pi}}e^{-b^2/2}.
   \]
   A finite-node approximation is almost periodic in \(b\), while \(s(b)\)
   decays.  Its uniform error is at least
   \(e^{-1/2}/(2\sqrt{2\pi})\), for every finite node count.
2. Phase-adapted Filon removes that bias obstruction, but no
   spectrum-uniform bound survives unbounded condition number.  As
   \(\kappa=\|M\|_{\rm op}/\lambda_{\min}\to\infty\), the closest Bernstein
   branch has \(\log\rho_*=O(\kappa^{-1/2})\); a fixed exponent 33 cannot
   contract uniformly.
3. No relative certificate is possible without an endpoint signal floor,
   because the linear functional \(G\mapsto s_G\) has a nontrivial kernel.

Finally, even the optimistic scalar-node arithmetic nearly saturates the
entire slack:

\[
\frac{10\cdot34\cdot55^2}{1024^2}
=\frac{1{,}028{,}500}{1{,}048{,}576}<1. \tag{44.9}
\]

Only 20,076 normalized multiply-add units remain; any non-view costate
rotation, extra Jordan-node constant, or source setup closes the arm.  Thus
34 nodes are an instance-specific fidelity certificate after literal
costate/eigenbasis reuse, never a generic deployment promise.

## 45. Full-seam H3 factor rank and corrected arithmetic no-go

The nine Jordan nodes in Section 37 are the query for **one directional
K31 interaction**.  They are not the factor rank of a complete dense
\((Q4,Q31,Q22)\) costate.  This distinction has an analytic lower-bound
witness.

Restrict the full K31 pullback to its pure-H1 singleton leg.  For source
coordinate \(j\), it contains

\[
\mathcal T_j=6\sum_i(Q31)_{ij}u_i^{\otimes3}. \tag{45.1}
\]

Choose \(u_1,\ldots,u_{55}\) as the standard basis and all displayed
coefficients nonzero.  Then (45.1) is a Fermat cubic.  Its mode-1 flattening
has rank 55, whereas a sum of \(k\) directional cubes has flattening rank at
most \(k\).  Hence

\[
k_{\rm signed}\ge55. \tag{45.2}
\]

The witnessing 55-by-55 minor is nonzero, so (45.2) holds on a Zariski-open
set of complete costates.  Literal Q-coordinate ownership removes copying;
it does not turn this rank-55 query into one nine-node interaction.

The corrected optimistic factor-native census is as follows.  Put

\[
d=55+\frac{55\cdot56}{2}=1595,
\]

let \(k\) be the physical Q-factor rank, \(\rho\) the total signed latent
factor rank of nonresident primitive matrices, \(q\) the certified Fourier
node count, and \(a=1\) for a factor-native physical contraction
(\(a=2\) when row expansion is also required).  Let \(c_{\rm eig}\) count
new per-unit eigensystems and \(c_{\rm rot}=1\) for factor-vector rotation,
\(2\) when dense hatted matrices are materialized.  Across ten active layers,

\[
\boxed{
C_{\rm add}\gtrsim
\frac{10}{1024^2}
\left[
c_{\rm eig}55^3+a\,k\,1595
+c_{\rm rot}\rho55^2+9q\rho55
\right]
+C_{\rm factor}+C_{\rm hook}.}
\tag{45.3}
\]

Consequently a necessary one-equivalent gate is

\[
166375c_{\rm eig}+1595a\,k+3025c_{\rm rot}\rho+495q\rho
\le104857.6. \tag{45.4}
\]

This gate is attained, up to the displayed constants, by a literal-factor
circuit: rotate each factor once and evaluate all Jordan/resolvent
contractions from those factors.  Discovering the factors by a new
eigendecomposition/SVD is separately charged.

The implications are immediate:

- dense physical Q has \(k=1024\), costing \(15.5762\) equivalents before
  any latent query;
- one new eigensystem per unit has
  \(10\cdot55^3/1024^2=1.5867\), so eigensystems must be literal residents;
- even with physical pulls free and \(q=0\), (45.4) requires
  \(\rho\le34\) for \(c_{\rm rot}=1\), or \(\rho\le17\) for
  \(c_{\rm rot}=2\), while the generic theorem gives \(\rho\ge55\);
- at \(q=1\), \(\rho\le29\); at \(q=9\), \(\rho\le14\);
- the Fourier node count \(q\) is independent of the nine structural Jordan
  terms and must still pass Section 44.

Thus the immediate full-seam identity (38.3) is an information/reverse-row
theorem, not a zero-cost theorem.  Generic implicit H3 deployment is closed
even with literal Q views.  It reopens only for pre-rotated resident factors
whose joint \(k,\rho,q\) satisfy (45.4), or for the distinct Z22 exception in
Section 43 where a literal full-plane Q22 makes the source action a direct
matrix dot.

## 46. NCQC commutator face: zero-product antisymmetric fusion

The most deployable part of the noncommutative core is not a transported H3
tensor but the commutator component of mixed q4.  For symmetric cores,

\[
K22_{ij}=48\operatorname{tr}(M_i^2M_j^2)
-8\|[M_i,M_j]\|_F^2,
\]

and a mixed-211 Bellman costate \(\Lambda_i\) consumes

\[
S_{211}
=-8\sum_{i,j,k}(\Lambda_i)_{jk}
\langle[M_i,M_j],[M_i,M_k]\rangle_F. \tag{46.1}
\]

If all costate slices have one common factor space,

\[
\Lambda_i=U\Theta_iU^{\mathsf T},\qquad
N_a=\sum_jU_{ja}M_j,
\]

then

\[
S_{211}
=-8\sum_i\sum_{a,b}(\Theta_i)_{ab}
\langle[M_i,N_a],[M_i,N_b]\rangle_F. \tag{46.2}
\]

Now suppose the resident AQ/D10 update literally forms the full products

\[
P_{ia}=M_iN_a,\qquad
J_{ia}=\frac{P_{ia}+P_{ia}^{\mathsf T}}2,\qquad
A_{ia}=\frac{P_{ia}-P_{ia}^{\mathsf T}}2.
\]

Symmetric/skew orthogonality gives the exact zero-new-product identity

\[
\boxed{
\langle[M_i,N_a],[M_i,N_b]\rangle_F
=4\{\langle P_{ia},P_{ib}\rangle_F
     -\langle J_{ia},J_{ib}\rangle_F\}.}
\tag{46.3}
\]

Thus retaining the antisymmetric half before the already-billed
symmetrization exposes the genuinely noncommutative q4-perp scalar with only
small Gram reductions.  It is neither CP3/star nor a duplicate of the Jordan
half: commuting cores make it vanish identically.

The loophole is exact only when:

1. the D14 core is the same \(M_i\), and the D10 suffix supplies the same
   common-factor \(N_a\) at the synchronized reset cut;
2. the full \(P_{ia}\), not only a separately recomputed \(J_{ia}\), is a
   resident intermediate;
3. every \(\Lambda_i\) shares the factor space \(U\) within the billed rank;
4. the commutator scalar is jointly Schur-residualized against CP4, Z22,
   H3, and the complete old arena.

Without common factors, forming anchor-dependent \(N_{ia}\) is generically
\(O(kn^3)\) per layer and closed.  Under the four receipts, (46.3) is the
cleanest current NCQC mechanism: a broad-spectrum, label-free,
noncommutative provider obtained by preserving information an owned matrix
product presently discards.

## 47. A diffuse ReLU reset destroys terminal q211 rank one in one step

The common-factor receipt in Section 46 is not implied by a rank-one
terminal costate.  The failure already occurs for uniformly diffuse,
bounded-condition reset legs.

Let the live repeated-anchor block \(I\) have size \(m\), and let a disjoint
pair block \(J\) have size \(m+1\).  Choose a normalized Hadamard matrix
\(H_J\), with constant first column \(h_0\), and nonconstant sign columns
\(h_i\), \(1\le i\le m\).  Put

\[
 u=h_0,\qquad s_i=\sqrt{m+1}\,h_i,
 \qquad c={1\over1+\epsilon},\qquad
 D_i=c\,\operatorname{Diag}(1+\epsilon s_i),
 \quad0<\epsilon<1,
 \]

and define the anchor-dependent pair pullback

\[
 B_i=D_iH_J.
\]

Every entry of \(B_i\) is \(O(m^{-1/2})\), and

\[
 \sigma(B_i)\subset
 \left[{1-\epsilon\over1+\epsilon},1\right],
 \qquad
 \kappa(B_i)={1+\epsilon\over1-\epsilon}. \tag{47.1}
\]

Moreover, all diagonal entries of \(D_i\) lie in \((0,1]\), so they are
literal Gaussian-ReLU slope probabilities after choosing the corresponding
standardized biases.  Nevertheless

\[
 v_i:=B_i^{\mathsf T}u=c(e_0+\epsilon e_i). \tag{47.2}
\]

Start with the same terminal rank-one slice
\(\Lambda_i=uu^{\mathsf T}\) for every \(i\in I\).  Under the exact reverse
recurrence

\[
 \Lambda'_p=\sum_{i\in I}A_{pi}B_i^{\mathsf T}
             \Lambda_iB_i,
\]

take \(A\) to be any invertible matrix on \(I\), for example a normalized
Hadamard matrix.  It is then itself diffuse and condition one.  Since
inverting the anchor mixing recovers every \(v_iv_i^{\mathsf T}\), the
smallest common pair support at the preceding cut is

\[
 \dim\operatorname{span}\{v_i:i\in I\}=m. \tag{47.3}
\]

Thus one reset expands a terminal common rank one to \(\Omega(m)\).  Taking
latent dimension at least \(2m+2\) makes this \(\Omega(r)\); when latent and
width dimensions scale together it is also \(\Omega(n)\).  The frame matrix
\(V=[v_1\ \cdots\ v_m]\) has squared singular values

\[
 c^2(m+\epsilon^2),qquad
 c^2\epsilon^2\quad\hbox{with multiplicity }m-1. \tag{47.4}
\]

Hence no rotation hides the expansion: after the leading direction, the
common-support spectrum has a flat \((m-1)\)-dimensional tail.

### 47.1 A matching well-conditioned NCQC commutator supply

The expanded directions can all have genuine q211 value.  In latent
dimension \(d\ge |I|+|J|+1\), let

\[
 S_a=E_{0a}+E_{a0},\qquad
 M_a=\mu I+\eta Q S_aQ^{\mathsf T},
 \qquad0<\eta<\mu,
\]

where \(Q\) is a flat orthogonal matrix.  These cores are positive definite
with condition number \((\mu+\eta)/(\mu-\eta)\); their noncommuting parts are
dense with entries at most \(2\eta/d\).  For \(i\in I\) and \(j,k\in J\),

\[
 [M_i,M_j]=\eta^2Q(E_{ij}-E_{ji})Q^{\mathsf T},
\]

and therefore the exact commutator-Gram supply is

\[
 (G_i)_{jk}
 =\langle[M_i,M_j],[M_i,M_k]\rangle_F
 =2\eta^4\delta_{jk}. \tag{47.5}
\]

For the canonical common-support projection \(P_S\), \(\dim S=k<m\),
work in the inverse-mixed anchor coordinates (equivalently, take \(A=I\)
for this scalar; an orthogonal Hadamard \(A\) preserves the panel norm below).
Equations (46.1) and (47.5) then give the exact lost q211 magnitude

\[
 \begin{aligned}
 |\Delta S_{211}|
 &=16\eta^4\sum_i\|P_{S^\perp}v_i\|^2\\
 &\ge16\eta^4c^2\epsilon^2(m-k). \tag{47.6}
 \end{aligned}
\]

Relative to the full positive supply magnitude, the lost fraction is at
least

\[
 {\epsilon^2(m-k)\over m(1+\epsilon^2)}. \tag{47.7}
\]

For \(\epsilon=1/2\), (47.6) is
\(16\eta^4(m-k)/9\), and (47.7) is \((m-k)/(5m)\).

### 47.2 The panel-complete theorem and the scalar-refit loophole

Equation (47.6) is exact for a state/Galerkin truncation.  A stronger
statement for *any* common-support approximation needs a task-complete
supply metric, because one scalar can always be spoofed: if only one
\(\langle\Lambda_i,G_i\rangle\) is observed, a rank-one surrogate can rescale
its coefficient to match that scalar while representing the wrong costate.
Endpoint scalar rank is therefore not provider rank.

The needed panel can be made explicitly NCQC-realizable.  Let
\(\{H_\alpha\}\) be an orthonormal basis of symmetric matrices on \(J\),
scaled so \(\|H_\alpha\|_{op}\le1\), and choose \(0<\delta<1\).  Use paired
commutator supplies

\[
 G_{i,\alpha}^{\pm}
 =2\eta^4(I\pm\delta H_\alpha). \tag{47.8}
\]

They are realized by replacing the star pair cores in (47.5) with the
bounded-condition mixtures \((I\pm\delta H_\alpha)^{1/2}\).  Flat conjugacy
again makes the physical cores dense, while increasing \(\mu\) keeps every
core uniformly positive definite.  The difference of the two q211 actions
is

\[
 A_{i,\alpha}^{+}(X)-A_{i,\alpha}^{-}(X)
 =-32\eta^4\delta\langle X_i,H_\alpha\rangle_F. \tag{47.9}
\]

Consequently this panel seminorm is exactly Frobenius tomography:

\[
 \sum_{i,\alpha}|A_{i,\alpha}^{+}(X)-A_{i,\alpha}^{-}(X)|^2
 =(32\eta^4\delta)^2\sum_i\|P_JX_iP_J\|_F^2. \tag{47.10}
\]

After undoing the invertible anchor mixing, for any common factor space
\(S\) of dimension \(k\), the optimal supported slice is
\(P_S(v_iv_i^{\mathsf T})P_S\).  Writing
\(t=\|v_i\|^2=c^2(1+\epsilon^2)\) and
\(b_i=\|P_{S^\perp}v_i\|^2\),

\[
 \|v_iv_i^{\mathsf T}-(P_Sv_i)(P_Sv_i)^{\mathsf T}\|_F^2
 =2tb_i-b_i^2\ge tb_i.
\]

Together with (47.4), every rank-\(k\) common-support approximation therefore
loses panel action norm at least

\[
 32\eta^4\delta\,c^2\epsilon
 \sqrt{(1+\epsilon^2)(m-k)}, \tag{47.11}
\]

or relative squared panel energy at least the fraction in (47.7).
For \(\epsilon=1/2\), fitting a squared-loss allowance of \(0.14381991\)
therefore requires \(k/m\ge0.28090045\), while the stricter
\(0.03280643\) allowance requires \(k/m\ge0.83596785\).  In particular, a
fixed rank-one reverse word fails both gates as \(m\) grows.

The conclusion is sharp in scope.  Diffuse ReLU gates, well-conditioned
pair legs, a rank-one terminal costate, and a single billed reverse stream do
not imply a low-rank q211 provider.  Rank one survives only under the literal
collinearity receipt \(B_i^{\mathsf T}u\parallel B_j^{\mathsf T}u\), or under
a quotient/panel theorem showing that the transverse frame in (47.4) is
unobservable.  A target-conditioned scalar fit cannot serve as that theorem.

## 48. Exact H3 reset counterexample: maximal diffuseness does not imply scalar leakage

The hidden-persistence norm in Section 39 has a sharp analytic counterexample.
For a centered purely linear gate \(g=a^{\mathsf T}z\), global sign symmetry
gives

\[
\mathcal R_g=\frac12I
\]

on the entire third Wiener chaos: the product
\(I_3(Z)I_3(G)\) is even, and the two half-spaces partition Gaussian space.
Thus scalar CP3 transport is exact in the linear cavity.

Now take the maximally diffuse radial quadratic gate

\[
g(z)=\lambda(\|z\|^2-r),\qquad \lambda>0.
\]

The O(\(r\))-equivariant H3 representation splits into traceless harmonic
cubics and trace cubics.  Schur's lemma and radial conditioning give the exact
two-sector reset

\[
\boxed{
\mathcal R_g
=\alpha_rP_{\rm harmonic}+\beta_rP_{\rm trace},
\qquad
\alpha_r=\Pr(\chi^2_{r+6}>r),\quad
\beta_r=\Pr(\chi^2_{r+2}>r).}
\tag{48.1}
\]

For the harmonic sector, the squared cubic contributes a radial \(R^6\)
weight, shifting the chi-square degrees by six.  For the trace witness
\(I_3(J_v(I))=(v^{\mathsf T}z)(\|z\|^2-r-2)\), direct integration gives

\[
\begin{aligned}
\beta_r
={}&\frac12\{(r+4)Q_{r+6}(r)
-2(r+2)Q_{r+4}(r)+(r+2)Q_{r+2}(r)\}\\
={}&Q_{r+2}(r),
\end{aligned}
\]

where the last equality is the incomplete-gamma recurrence.

At \(r=55\), these eigenvalues are approximately
\(\alpha_{55}=0.6919\) and \(\beta_{55}=0.5505\).  Hence the scalar-half
approximation has operator error at least \(0.1919\), or about 38.4% relative
error on the harmonic sector.  This is a maximally stable-rank quadratic
form, so **stable rank/diffuseness alone cannot certify small H3 reset
leakage**.

The radial family itself has an exact cheap exception: carrying the two
sector coefficients closes H3 reset without a tensor.  But it is collective
and supplies no generic identity-specific orientation.  For unit-varying
anisotropic quadratic cores, the common O(\(r\)) projectors are lost and a
two-sector carrier is invalid without an exact intertwiner.

Accordingly (39.5) cannot use a diffuse-spectrum proxy for \(\rho_j\).  It
needs either a linear-dominant perturbation bound, an exact radial two-sector
receipt, or the actual complete-CP3 quotient norm.  This closes any generic
claim that r55 diffuseness by itself makes hidden H3 persistence negligible.

## 49. Fused H211 resolvent certificate and common-factor batching

The H31 score is not the correct query for the off-diagonal noncommutative
`211` face.  This section gives the corresponding finite-node theorem for
the fused score and keeps local mathematical acquisition separate from
current D11 ownership.

Use the H4/AQ convention

\[
g(z)=c+b^{\mathsf T}z+\frac12(z^{\mathsf T}Mz-\operatorname{tr}M),
\qquad p(z)=b+Mz,
\]

and let \(R,S,T\) be latent directions.  If
\(A=R^{\mathsf T}z,B=S^{\mathsf T}z,C=T^{\mathsf T}z\), the correct Wick
score is

\[
\begin{aligned}
:A^2BC:={}&A^2BC-\|R\|^2BC-2\langle R,S\rangle AC
-2\langle R,T\rangle AB-\langle S,T\rangle A^2\\
&+\|R\|^2\langle S,T\rangle
+2\langle R,S\rangle\langle R,T\rangle .
\end{aligned} \tag{49.1}
\]

Thus

\[
E[g_+(Z):A^2BC:]=K_4(g_+)[R,R,S,T].
\]

### 49.1 Exact scalar Fourier integrand

Put

\[
D_t=I-itM,\quad S_t=D_t^{-1},\quad
m_t=S_tb,\quad C_t=MS_tM,
\]

\[
\chi(t)=e^{it(c-\operatorname{tr}M/2)}
\det(D_t)^{-1/2}
\exp\!\left[-\frac{t^2}{2}b^{\mathsf T}S_tb\right].
\]

For \(X,Y\in\{R,S,T\}\), write

\[
x=R^{\mathsf T}m_t,\quad y=S^{\mathsf T}m_t,
\quad z=T^{\mathsf T}m_t,
\quad c_{XY}=X^{\mathsf T}C_tY,
\quad d_{XY}=X^{\mathsf T}MY.
\]

Define

\[
\begin{aligned}
Q_2={}&x^2yz+c_{RR}yz+2c_{RS}xz+2c_{RT}xy+c_{ST}x^2\\
&+c_{RR}c_{ST}+2c_{RS}c_{RT},\\
Q_1={}&d_{RR}yz+2d_{RS}xz+2d_{RT}xy+d_{ST}x^2\\
&+d_{RR}c_{ST}+d_{ST}c_{RR}
+2d_{RS}c_{RT}+2d_{RT}c_{RS},\\
Q_0={}&d_{RR}d_{ST}+2d_{RS}d_{RT}.
\end{aligned} \tag{49.2}
\]

The complete fused action is the one-dimensional integral

\[
\boxed{
K_4(g_+)[R,R,S,T]
=\frac1{2\pi}\int_{\mathbb R}\chi(t)
\{(it)^2Q_2(t)+(it)Q_1(t)+Q_0\}\,dt .}
\tag{49.3}
\]

This follows either by inserting
\(H=\operatorname{Sym}(R^{\otimes2}\otimes S\otimes T)\) in the general
H4 surface formula, or directly by taking the tilted-Gaussian expectation of
\(A^2BC\).  In particular, (49.2) contains every `211`, `31`, `22`, and `4`
collision with the correct Wick coefficient; the pure external `211` read
must perform the old collision subtractions afterward.

### 49.2 Deterministic 34-node certificate

Let \(M=U\operatorname{diag}(\lambda_j)U^{\mathsf T}\) and
\(\alpha=U^{\mathsf T}b\).  On the real axis,

\[
|\chi(t)|=
\prod_j(1+t^2\lambda_j^2)^{-1/4}
\exp\!\left[-\frac{t^2}{2}
\sum_j\frac{\alpha_j^2}{1+t^2\lambda_j^2}\right].
\tag{49.4}
\]

Set

\[
m(t)^2=\sum_j\frac{\alpha_j^2}{1+t^2\lambda_j^2},
\qquad
v(t)^2=\sum_j\frac{\lambda_j^4}{1+t^2\lambda_j^2}.
\]

For an arbitrary symmetric H4 query \(H\), including the aggregate query
below, a deterministic tail certificate is

\[
\boxed{
E_{\rm tail}^{(4)}(T)\le
\frac{\|H\|_{\rm HS}}{\pi}\int_T^\infty |\chi(t)|
\left[
t^2\{m^4+6vm^2+3v^2\}
+6t\{\|M\|_Fm^2+\|M\|_Fv\}
+3\|M\|_F^2
\right]dt .}
\tag{49.5}
\]

This uses the exact characteristic decay, not a nominal asymptotic order.
For a full-rank core, let

\[
B_1^2=\sum_j\frac{\alpha_j^2}{\lambda_j^2},
\qquad C_\lambda=\prod_j|\lambda_j|^{-1/2}.
\]

Since \(m(t)\le B_1/t\), \(v(t)\le\|M\|_F/t\), (49.5) gives at
\(r=55\)

\[
\boxed{
E_{\rm tail}^{(4)}(T)\le
\frac{\|H\|_{\rm HS}C_\lambda}{\pi}
\left[
\frac{2B_1^4}{57}T^{-57/2}
+\frac{24B_1^2\|M\|_F}{55}T^{-55/2}
+\frac{24\|M\|_F^2}{53}T^{-53/2}
\right].}
\tag{49.6}
\]

The last term is why an H31 or H3 tail certificate cannot simply be reused:
the H4 surface action has a slower \(T^{-53/2}\) worst-case tail.

For the core interval write the integrand in (49.3) as
\(e^{i\beta t}A_{211}(t)\), where
\(\beta=c-\operatorname{tr}M/2\).  Its branch points are
\(t_j=-i/\lambda_j\).  Under \(t=T(1+z)/2\), put

\[
z_j=-1-\frac{2i}{\lambda_jT},\qquad
\rho_*=\min_j\left|z_j+\sqrt{z_j^2-1}\right|,
\tag{49.7}
\]

with the square-root branch chosen to give modulus greater than one.  For
\(1<\rho<\rho_*\), let
\(B_{211,\rho}=\sup_{z\in E_\rho}|A_{211}(T(1+z)/2)|\), and let
\(h_\rho\) be the mapped ellipse's maximum imaginary height.  Exactly as in
Section 44, 34 shifted Gauss--Legendre nodes give

\[
E_{\rm GL}^{211}\le
\frac{4TB_{211,\rho}e^{|\beta|h_\rho}}
{\pi(\rho-1)}\rho^{-67},
\tag{49.8}
\]

while phase-extracted 34-node Filon--Chebyshev gives

\[
E_{\rm Filon}^{211}\le
\frac{4TB_{211,\rho}}{\pi(\rho-1)}\rho^{-33}.
\tag{49.9}
\]

Thus one reset is certified only when some \(T,\rho\) obey

\[
E_{\rm tail}^{(4)}(T)
+\min(E_{\rm GL}^{211},E_{\rm Filon}^{211})
\le\epsilon_{211}.
\tag{49.10}
\]

Errors add over reset sites.  Section 44's fixed-bias, unbounded-condition,
and zero-signal impossibility theorems apply unchanged, so 34 nodes remain
an instance certificate rather than a uniform fidelity promise.

### 49.3 Exact common-factor contraction

Suppose the external pair-leg costate at anchor \(i\) has a common support

\[
\Lambda_i=V\Theta_iV^{\mathsf T},\qquad
V=[V_1,\ldots,V_k],\quad \Theta_i=\Theta_i^{\mathsf T}.
\]

Multilinearity turns the whole pair slice into the one aggregate tensor

\[
H_i=\sum_{a,b}(\Theta_i)_{ab}
\operatorname{Sym}(R_i^{\otimes2}\otimes V_a\otimes V_b),
\qquad
\|H_i\|_{\rm HS}\le
\|R_i\|^2\|V\Theta_iV^{\mathsf T}\|_F.
\tag{49.11}
\]

At one Fourier node define

\[
\begin{gathered}
x=R_i^{\mathsf T}m_t,\quad a_C=R_i^{\mathsf T}C_tR_i,
\quad a_M=R_i^{\mathsf T}MR_i,\\
y=V^{\mathsf T}m_t,\quad v_C=V^{\mathsf T}C_tR_i,
\quad v_M=V^{\mathsf T}MR_i,\\
C_V=V^{\mathsf T}C_tV,\qquad M_V=V^{\mathsf T}MV,\qquad
\Theta=\Theta_i.
\end{gathered}
\]

Summing (49.2) over \(a,b\) gives, without forming an H4 tensor,

\[
\boxed{
\begin{aligned}
\bar Q_2={}&(x^2+a_C)
\{y^{\mathsf T}\Theta y+\langle\Theta,C_V\rangle_F\}
+4xv_C^{\mathsf T}\Theta y+2v_C^{\mathsf T}\Theta v_C,\\
\bar Q_1={}&a_My^{\mathsf T}\Theta y
+4xv_M^{\mathsf T}\Theta y+x^2\langle\Theta,M_V\rangle_F\\
&+a_M\langle\Theta,C_V\rangle_F
+a_C\langle\Theta,M_V\rangle_F
+4v_M^{\mathsf T}\Theta v_C,\\
\bar Q_0={}&a_M\langle\Theta,M_V\rangle_F
+2v_M^{\mathsf T}\Theta v_M.
\end{aligned}}
\tag{49.12}
\]

All `H211` directions at the same physical reset therefore reuse the same
\(t_q,\chi(t_q),m_{t_q},C_{t_q}\).  The rowwise values needed for
\(\langle\Theta,C_V\rangle\) can be precomputed, so a dense \(\Theta\)
costs \(O(rk+k^2)\) per node after \(O(r^2k+rk^2)\) setup; diagonal
\(\Theta\) costs \(O(rk)\) per node after \(O(r^2k)\) setup.

The pure off-diagonal external `211` face is **not** obtained by silently
zeroing the diagonal of \(V\Theta V^{\mathsf T}\), which generically destroys
the low rank.  Instead evaluate the common-factor all-pair action and subtract
the `s=t`, `s=i`, and `t=i` collision faces with the synchronized old
K22/K31/K4 receipts.  For the commutator q4-perpendicular source the K31
faces vanish algebraically, but the diagonal K22 face still must be removed.

### 49.4 Joint node/rank cost and lawful reuse

With reused eigensystems, ten active layers, \(r=55,n=1024,q=34\), the
optimistic unit-constant cost for a dense common-factor slice is

\[
\boxed{
C_{211}^{\rm dense}\gtrsim
\frac{10}{1024^2}
\{55^2k+55k^2+34(55k+k^2+55)\}
=\frac{10}{1048576}(89k^2+4895k+1870).}
\tag{49.13}
\]

Thus even before source formation, recurrence, and non-unit constants, the
one-equivalent gate requires \(k\le16\).  For diagonal \(\Theta_i\), charging
all displayed setup and backbone terms gives

\[
C_{211}^{\rm diag}\gtrsim
\frac{10}{1048576}(4984k+1870),
\tag{49.14}
\]

so \(k\le20\).  If a replacement schedule literally shares the H31 spectral
backbone, the final 1870 term is paid once and the optimistic diagonal gate
relaxes to \(k\le21\).  These are necessary arithmetic gates, not billing
receipts.

The reuse rules are exact:

1. H31 and H211 may share Fourier abscissae, characteristic values,
   resolvents, \(m_t\), and \(C_t\) at the **same** reset.  H211 must still
   pass its own slower-tail bound (49.6).
2. Different pair directions and common-factor slices at that reset share
   nodes and use (49.12); only their contractions differ.
3. Different physical units may share a chosen abscissa grid, but not
   characteristic/resolvent values unless their AQ spectra and frames are
   literally identical.  Filon weights also depend on each unit's
   \(\beta\).
4. Reusing an eigensystem does not make the rotations of new \(V_a\) free.
5. The algebraic antisymmetric-product reuse in Section 46 supplies the
   commutator Gram, not the Fourier surface samples.  Neither reuse implies
   the other.

### 49.5 Information and deployment disposition

**Current-D11 information: NO.**  The q4-perpendicular `211` commutator
Gram has zero K31 restriction, old Q22 sees only its diagonal collision, and
the resident CP4/star/pair summaries do not determine the off-diagonal
three-label block.  A stored H31 scalar is not a stored set of Fourier nodes
and cannot reconstruct (49.12).  Counting those old collision reads as the
new off-diagonal action would be duplicate credit.

**Replacement-state local acquisition: mathematical GO, deployment
conditional.**  A replacement row may jointly acquire H31 and the
complete-old-residual H211 action from the same AQ spectral backbone, using
(49.3), (49.10), and (49.12), provided that:

1. the common pair support survives every reverse cut at the billed rank;
2. the exact external collision subtractions and complete old-interface
   Schur projection are applied once;
3. AQ model error plus summed quadrature error is below the value threshold;
4. direction aggregation, rotations, costate construction, and recurrence
   fit together with (49.13) or (49.14); and
5. the row displaces resident billed work or otherwise fits the single
   remaining equivalent.

Consequently the fused H211 oracle closes the local information gap but does
not prove that current D11 already contains the answer.  Its strongest lawful
use is a jointly designed replacement state, not an additive H31 patch.

## 50. Exact H4 reset gate for a sewn rank-one Q22+Q211 suffix

This section specializes the fourth-chaos multiplication theorem to the
collision-complete pair slice used by the q211 route.  Let

\[
F={\bf1}_{\{S>0\}},\qquad p=EF,
\]

and define the symmetric Hermite moment tensors by

\[
\langle C_q,V\rangle=E[(F-p)I_q(V)].
\]

On a symmetric fourth-chaos tensor `T`, the reset operator has the exact
direct-action form

\[
\boxed{
\mathcal K_FT=pT
+4\operatorname{Sym}(C_2\otimes_1T)
+3\operatorname{Sym}(C_4\otimes_2T)
+{2\over3}\operatorname{Sym}(C_6\otimes_3T)
+{1\over24}(C_8\otimes_4T).}
\tag{50.1}
\]

The coefficients follow by projecting
`I_{2s}(C_{2s}/(2s)!) I_4(T)` back to fourth chaos.  Equation (50.1), not a
single mean slope, is the self-consistent H4 reset law.

### 50.1 Exact common-factor block condition

Let the collision-sewing map embed a common suffix as

\[
J_u\theta=\{\theta_i uu^{\mathsf T}\}_i.
\]

It includes both the Q22 diagonal and the off-diagonal Q211 face with the
binding collision normalization.  Let `E` denote exact extraction of the
sewn pair slices after the H4 action, and let

\[
R_O=I-P_{O_{\rm complete}}
\]

be the certified orthogonal complete-old Schur projector at that seam.  Write
`L_0=pI` and let `L_2,L_4,L_6,L_8` be the four terms in (50.1).  The fixed
suffix `u` is preserved level by level for every anchor coefficient `theta`
if and only if

\[
\boxed{
(I-\Pi_u)\,E R_O L_qJ_u=0,
\qquad q=0,2,4,6,8,}
\tag{50.2}
\]

where

\[
(\Pi_u\Lambda)_i
={\langle\Lambda_i,uu^{\mathsf T}\rangle_F\over\|u\|^4}
uu^{\mathsf T}.
\]

This is both necessary and sufficient; it tests every off-support block, not
only the diagonal Q22 or one K31 scalar.  If a new common suffix `v` is
allowed, replace `Pi_u` by `Pi_v` and require one and the same `v` for all
five levels.  In particular, when the nonzero scalar term survives the old
projection with pair range `u`, every other live level must also return range
`u`.  A different range for even one of `C2,C4,C6,C8` forces common width at
least two.

For one fixed realized gate, cross-level cancellation can be admitted.  Its
weaker exact condition is

\[
(I-\Pi_v)ER_O( L_0+L_2+L_4+L_6+L_8)J_u=0.
\tag{50.3}
\]

Such a cancellation is not a levelwise/common-recurrence certificate and is
destroyed by changing the standardized bias, the gate direction, or the old
quotient.

### 50.2 Minimal post-old block and singular certificate

Flatten the pair-row mode of all output slices and all input anchor basis
vectors.  Define

\[
Y_q=\operatorname{Flat}_{\rm pair}(ER_OL_qJ_u),
\qquad
Y_{\rm lev}=[Y_0\ Y_2\ Y_4\ Y_6\ Y_8].
\tag{50.4}
\]

Because every output block is symmetric, the minimum exact common pair
support for a levelwise reset is

\[
\boxed{k_*^{\perp}=\operatorname{rank}Y_{\rm lev}.}
\tag{50.5}
\]

Thus rank one is equivalent to `sigma_2(Y_lev)=0`; a prescribed suffix `u`
is equivalent to

\[
\boxed{(I-P_u)Y_{\rm lev}=0.}
\tag{50.6}
\]

For a merely target-conditioned total action use
`Y_tot=Flat_pair(ER_O K_F J_u)` instead.  This distinction prevents an
accidental cancellation between Hermite levels from being promoted to a
reset-invariant state.

Let

\[
\tau_k(Y)=\left(\sum_{j>k}\sigma_j(Y)^2\right)^{1/2}.
\]

For symmetric slices, the optimal two-sided common-support error

\[
\epsilon_{k,\rm sym}
=\min_{\dim U=k}
\left(\sum_i\|\Lambda_i-P_U\Lambda_iP_U\|_F^2\right)^{1/2}
\]

obeys the sharp elementary sandwich

\[
\boxed{
\tau_k(Y)\le\epsilon_{k,\rm sym}\le\sqrt2\,\tau_k(Y).}
\tag{50.7}
\]

Indeed, in `U plus U_perp` blocks the left residual contains `(B^T,C)`
whereas the symmetric two-sided residual contains `(B,B^T,C)`.  Singular
tails are therefore a valid necessary certificate and a factor-`sqrt(2)`
safe Frobenius certificate.  They are not an endpoint-value receipt: the
binding scalar error remains the supply-weighted contraction

\[
\boxed{
E_{22+211}
=\sum_{l,i}w_l
\langle\Lambda_{i,l}-\widehat\Lambda_{i,l},G_{i,l}\rangle_F,}
\tag{50.8}
\]

with the q211 normalization factor restored where applicable.

### 50.3 Exact linear-gate rank-two obstruction

The obstruction can be solved in closed form.  Let `G,V,X` be orthonormal,
let the linear ReLU gate be `F=1_{G>t}`, and let the sewn rank-one state be

\[
T=H_2(X)H_2(U),\qquad U=\alpha G+\beta V,
\qquad \alpha^2+\beta^2=1.
\]

This is one repeated anchor `X` with the complete rank-one Q22+Q211 suffix
`UU^T`.  Fourth chaos decomposes by the number of `G` legs.  Put

\[
\lambda_k(t)={1\over k!}\int_t^\infty H_k(x)^2\phi(x)\,dx.
\]

Only `k=0,1,2` occur in this slice, and the reset is exactly

\[
\mathcal K_FT=H_2(X)\{
\alpha^2\lambda_2H_2(G)
+2\alpha\beta\lambda_1GV
+\beta^2\lambda_0H_2(V)\}.
\tag{50.9}
\]

Hence its new pair matrix in the `(G,V)` plane is

\[
\boxed{
R_t(U;G)=
\begin{pmatrix}
\alpha^2\lambda_2&\alpha\beta\lambda_1\\
\alpha\beta\lambda_1&\beta^2\lambda_0
\end{pmatrix},
\qquad
\det R_t=\alpha^2\beta^2
(\lambda_0\lambda_2-\lambda_1^2).}
\tag{50.10}
\]

Therefore an oblique suffix remains rank one if and only if

\[
\boxed{
\alpha\beta=0
\quad\hbox{or}\quad
\lambda_0(t)\lambda_2(t)=\lambda_1(t)^2.}
\tag{50.11}
\]

Preservation of the same suffix `U`, rather than some new rank-one suffix,
requires the stronger identity `lambda_0=lambda_1=lambda_2` on the live
components.  Explicitly,

\[
\lambda_0=p,
\quad\lambda_1=p+t\phi(t),
\quad\lambda_2=p+{t^3+t\over2}\phi(t).
\tag{50.12}
\]

At the centered odd half-space `t=0`, all even gate tensors vanish and the
scalar identity is exact.  At `t=1`, however,

\[
\lambda_1=\lambda_2=p+\phi(1),
\qquad
\lambda_0\lambda_2-\lambda_1^2
=-(p+\phi(1))\phi(1)\ne0.
\tag{50.13}
\]

For the maximally oblique suffix `alpha^2=beta^2=1/2`, the discarded singular
value of the best rank-one pair approximation is

\[
\boxed{
s_2={
\sqrt{T_0^2+(p+\phi(1))\phi(1)}-T_0\over2}
\simeq0.0694,
\qquad T_0={2p+\phi(1)\over2}.}
\tag{50.14}
\]

Thus the full `3.80e-6` endpoint allowance requires either exact endpoint
annihilation of this second pair direction or endpoint-weighted incoming
amplitude at most approximately

\[
{3.80\times10^{-6}\over0.0694}\simeq5.48\times10^{-5},
\tag{50.15}
\]

before any other error allocation.

The individual Hermite mechanism is also explicit.  On the pair degrees
`k=0,1,2`, the `C2` contribution has eigenvalue sequence

\[
(0,m_2,2m_2),\qquad m_2=E[FH_2(G)]=t\phi(t),
\]

and the `C4` contribution is `(0,0,m_4/2)`, where
`m_4=phi(t)H_3(t)`.  The `C6` and `C8` contractions vanish for this
transverse-anchor slice because it contains at most two gate legs.  Thus
already `C2` creates a nonzero off-support determinant whenever
`alpha beta t` is nonzero; `C4` can cancel it only at the exceptional total
identity (50.11).

More generally, let `(r_{q0},r_{q1},r_{q2})` be the three gate-leg
eigenvalues contributed by one even level `q`.  For an oblique suffix that
level produces a rank-one updated factor if and only if

\[
r_{q0}r_{q2}=r_{q1}^2.
\]

All live levels produce the **same** updated common factor if and only if,
in addition, their geometric ratios `r_q1/r_q0=r_q2/r_q1` agree across
`q` (with the zero cases handled by support deletion).  They preserve the
original `U` precisely when `r_q0=r_q1=r_q2` on its live components.  This
is the two-dimensional scalar specialization of the block condition (50.2).

### 50.4 r55 rank and annihilation count

For a full linear gate at `r=55`, the five gate-leg eigenspaces have
dimensions

\[
\boxed{395010,\quad27720,\quad1485,\quad54,\quad1.}
\tag{50.16}
\]

At `t=1`, `K_F-pI` has singular value `phi(1)` on `29259` dimensions and
`7phi(1)/6` on one further dimension.  Therefore

\[
\boxed{
\operatorname{rank}(K_F-pI)=29260,
\qquad
\operatorname{srank}(K_F-pI)simeq21497.4.}
\tag{50.17}
\]

Any scalar-reset proof must annihilate this entire endpoint-visible leakage
space; if the complete-old/endpoint null space fails to contain the
`29259`-dimensional `phi(1)` eigenspace, a direction with leakage
`phi(1)` survives.

For the sewn pair route the sharper common-support statement is local.  Let
`G_i` be the neuron-specific linear gate direction.  Whenever

\[
0<|\langle U,G_i\rangle|<1,
\qquad
\lambda_0(t_i)\lambda_2(t_i)\ne\lambda_1(t_i)^2,
\]

and the transverse output is not killed by the old quotient or endpoint,
(50.10) has range `span(U,G_i)`.  Consequently

\[
\boxed{
k_*^{\perp}
=\dim\operatorname{span}
\{U,G_i:\ i\text{ is live and nonannihilated}\}}
\tag{50.18}
\]

in the linear-gate model.  Generic r55 gate directions make this dimension
`55`, not one.  Relative to the optimistic acquisition caps in Section 49,
the complete-old/endpoint quotient must therefore eliminate at least

- `39` independent pair directions for the dense cap `k<=16`;
- `35` for the diagonal cap `k<=20`; or
- `34` even under the shared-backbone cap `k<=21`.

The exact endpoint-annihilation alternative is

\[
\boxed{
\sum_{l,i}w_l
\langle G_{i,l}^{\rm supply},
R_{t_i}(U;G_i)-\widehat R_{i,l}\rangle_F=0,}
\tag{50.19}
\]

or the corresponding absolute value below `3.80e-6`.  Stable rank,
coordinate diffuseness, and a rank-one terminal suffix imply none of
(50.2), (50.6), or (50.19).

### Graph split

- `P2R0G19B_MATH_Q22Q211_H4_LEVEL_BLOCKS` — **solved theorem**: exact sewn reset and levelwise invariant-block condition, (50.1)--(50.6).
- `P2R0G19B_MATH_Q22Q211_POSTOLD_SINGULAR_GATE` — **open frozen-net receipt**: compute `Y_lev`, its common-support spectrum, and the supply-weighted error (50.8).
- `P2R0G19B_MATH_Q22Q211_LINEAR_GATE_RANK2` — **solved no-go**: an oblique biased linear gate makes one sewn rank-one suffix rank two, (50.9)--(50.15).
- `P2R0G19B_MATH_Q22Q211_R55_COMMON_WIDTH` — **conditional structural no-go**: generic live gates require width 55; only the explicit old/endpoint annihilation counts in (50.18)--(50.19) can reduce it to the billed cap.

## 51. Complete-old H3 reset leakage: exact gate and stable-rank no-go

Let (F=1_{\{g>0\}}), (p=\mathbb E F), and let (P_\perp) be the
orthogonal complement of the **complete** old H3 interface.  Write

\[
B=\mathcal R_g-pI,
\qquad
R_\perp=P_\perp\mathcal R_gP_\perp
          =pP_\perp+P_\perp BP_\perp .
\tag{50.1}
\]

This section separates three questions which a diffuseness proxy had
conflated: scalar reset probability (p), non-scalar H3 leakage (B), and
whether that leakage survives the complete-old quotient.

### 50.1 Exact even-chaos theorem

For (q=2,4,6), define the symmetric Hermite-moment tensor (C_q) by

\[
\langle C_q,V\rangle
=\mathbb E[(F-p)I_q(V)].
\]

The Wiener product formula gives the exact identity

\[
\boxed{
\begin{aligned}
6\langle BZ,G\rangle={}&
18\langle C_2,Z\widetilde\otimes_2G\rangle
+9\langle C_4,Z\widetilde\otimes_1G\rangle
+\langle C_6,Z\widetilde\otimes G\rangle .
\end{aligned}}
\tag{50.2}
\]

Thus H3 reset leakage sees **exactly** the even gate chaoses (2,4,6);
stable rank is not the relevant invariant.  In particular,

\[
\boxed{
\|P_\perp BP_\perp\|_{\rm op}
\le \min\!\left\{
\max(p,1-p),
3\|C_2\|_F+\frac32\|C_4\|_F+\frac16\|C_6\|_F
\right\}.}
\tag{50.3}
\]

If (W_q=\|P_q(F-p)\|_2^2=\|C_q\|_F^2/q!\), a convenient scalar
corollary is

\[
\|P_\perp BP_\perp\|_{\rm op}
\le 3\sqrt2\sqrt{W_2}+3\sqrt6\sqrt{W_4}+2\sqrt5\sqrt{W_6}
\le \sqrt{92(W_2+W_4+W_6)}.
\tag{50.4}
\]

These are genuine analytic upper bounds, but small maximum coordinate
influence does not make their right-hand side small: it controls no more
than the distribution of a fixed total even-chaos mass across coordinates.

There is also an exact formula displaying the missing dependence on the
AQ parameters.  Put (u(z)=a+2Mz) and (Q=2M).  Gaussian integration by
parts, interpreted distributionally (or by mollifying the threshold), gives

\[
\boxed{
C_q=\sum_{j=0}^{\lfloor q/2\rfloor}
\frac{q!}{2^j j!(q-2j)!}\,
\operatorname{Sym}\mathbb E\!\left[
\delta^{(q-j-1)}(g)
Q^{\otimes j}\otimes u^{\otimes(q-2j)}
\right].}
\tag{50.5}
\]

Consequently a bound phrased only through
(\|M\|_{\rm op}/\|M\|_F), stable rank, or coordinate sizes omits the
boundary-density tensors which actually determine the reset.

### 50.2 Two sharp no-go families

First take a biased linear cavity (g=u^{\mathsf T}z-\tau),
(\|u\|=1).  Relative to (S=u^{\mathsf T}z), H3 decomposes as

\[
\mathcal H_3=\bigoplus_{j=0}^3
\frac{\operatorname{He}_j(S)}{\sqrt{j!}}
\otimes\mathcal H_{3-j}(u^\perp),
\]

and (B) is scalar on every block.  Its four eigenvalue shifts are

\[
\boxed{
\begin{aligned}
\delta_0&=0,\\
\delta_1&=\tau\phi(\tau),\\
\delta_2&=\frac{\tau(\tau^2+1)}2\phi(\tau),\\
\delta_3&=\frac{\tau^5-\tau^3+6\tau}{6}\phi(\tau).
\end{aligned}}
\tag{50.6}
\]

At (\tau=1), all three nonzero shifts equal (\phi(1)), on a subspace
of dimension

\[
\binom r2+(r-1)+1=\frac{r(r+1)}2.
\tag{50.7}
\]

Rotating (u) makes every coordinate (O(r^{-1/2})).  Adding an
arbitrarily small, flat, high-stable-rank quadratic form preserves (50.6)
by continuity, so neither coordinate diffuseness nor the scale-invariant
stable rank repairs the claim.

Nor does requiring the quadratic term to be full-rank, perfectly
conditioned, and variance-balanced.  Let (r=2s),

\[
M_s=(2\sqrt s)^{-1}I_{2s},\qquad a=cu,
\]

with (u) coordinate-flat, and standardize the threshold by
(\tau\sqrt{1+c^2}).  Then
(\operatorname{sr}(M_s)=r), (\kappa(M_s)=1), and
(2\|M_s\|_F^2=1).  In the diffuse Gaussian limit, for
(H=\operatorname{He}_3(u^{\mathsf T}z)/\sqrt6) and
(\rho^2=c^2/(1+c^2)),

\[
\boxed{
\lim \langle H,BH\rangle
=\phi(\tau)\left{
\frac{\rho^6}{6}\operatorname{He}_5(\tau)
+\frac32\rho^4\operatorname{He}_3(\tau)
+3\rho^2\tau
\right\}.}
\tag{50.8}
\]

For (c=\tau=1), this is (7\phi(1)/8\), an order-one obstruction.
There is simultaneously a growing ((s-1))-dimensional spectator module
whose every centered reset eigenvalue tends to
(\tau\phi(\tau)/[2(1+c^2)]).  Hence maximal stable rank, condition one,
balanced linear/quadratic variance, and vanishing coordinate influence
still do not produce operator-norm scalarization.  The pointwise diffuse
cavity limit and the supremum over H3 directions do not commute.

### 50.3 Quotient-safe implication and the only valid gate

Let (E) be an eigenspace of (B) with eigenvalue (\delta).  If the
remaining terms have the same sign (as in the (\tau=1) linear example),
then exactly

\[
\|P_\perp BP_\perp\|_{\rm op}
\ge |\delta|\,\|P_E P_\perp\|_{\rm op}^2.
\tag{50.9}
\]

In particular, if (\dim E>\operatorname{rank}P_{\rm old}), then
(E\cap\operatorname{ran}P_\perp\ne0) and the full eigenvalue survives.
If the old span can contain (E), no dimension-free claim is lawful: the
principal angle in (50.9), or the actual Schur residual, is required.

Section 48 gives a stronger quotient-safe r55 witness.  For the flat radial
quadratic reset, the harmonic H3 eigenspace has dimension

\[
\binom{r+2}{3}-r.
\]

At (r=55) this is (29205).  Therefore, **conditional on the documented
16384 rank being the complete-old H3 rank cap**, at least (12821)
harmonic directions remain in the quotient, and on all of them

\[
R_\perp=\alpha_{55}I,qquad
\alpha_{55}=\Pr(\chi^2_{61}>55)\simeq0.6919.
\tag{50.10}
\]

Thus finite width 55 supplies no generic small hidden-persistence tail;
even maximal spectral diffuseness can leave a one-reset norm at least
(0.6919).  The centered leakage is exactly
(\alpha_{55}-p_{55}), (p_{55}=\Pr(\chi^2_{55}>55)), and is strictly
larger than the already recorded half-gate discrepancy (0.1919).
This is a one-reset guarantee.  A multi-reset product lower bound additionally
requires a common surviving harmonic subspace (or explicit principal-angle
control) across cuts.

The necessary-and-sufficient certification object is therefore

\[
\boxed{
\epsilon_{3,\perp}
=\left\|P_\perp\,
\mathbb E[(F-p)\Phi_3\Phi_3^{\mathsf T}]
P_\perp\right\|_{\rm op},}
\tag{50.11}
\]

where (\Phi_3) is any orthonormal H3 feature vector.  Equivalently one may
certify the quotient projections of the three tensors in (50.2).  Exact
odd parity (the centered linear gate), a proved radial intertwiner, or
endpoint orthogonality to those projected tensors are valid shortcuts.
Stable rank is not.

**Status.**  Generic scalar H3 reset from diffuseness is closed NO-GO;
generic r55 hidden-tail smallness is closed NO-GO.  The deployed instance
remains open only at the explicit complete-old projector in (50.11).
The next mathematical question is whether the actual old H3 interface has
rank (16384) after all collision/depth summaries are included and, if so,
whether its principal angles to the AQ-generated even-chaos modules admit an
analytic lower or upper bound without constructing the full 29260-state
operator.

## 52. Corpus-complete q211 rank certificates after the old quotient

The scalar-refit loophole in Section 47 has three distinct versions.  They
must not be conflated when interpreting an eight-net panel.

### 50.1 Fixed direction: two targets are necessary and sufficient

Suppose the proposed one-factor recurrence fixes a residual rank-one template
\(R=uu^{\mathsf T}\), and only one scalar amplitude \(\theta\) may be fitted
across a family of targets.  Let \(Q_O\) be the orthogonal projection away
from the complete old interface, let

\[
 y_\nu=\langle Q_O\Lambda,Q_OG_\nu\rangle,
 \qquad
 g_\nu=\langle Q_OR,Q_OG_\nu\rangle,
\]

and let \(W\succ0\) be the target/corpus metric.  The exact best scalar-refit
error is

\[
 \boxed{
 d_{\rm scale}^2
 =\min_\theta\|y-\theta g\|_W^2
 =y^{\mathsf T}Wy-
 {\left(g^{\mathsf T}Wy\right)^2\over g^{\mathsf T}Wg}.}
 \tag{50.1}
\]

One nonzero target is always spoofable.  Two independent targets are the
minimum that can reject rescaling: in unweighted coordinates the certificate
is simply

\[
 y_1g_2-y_2g_1\ne0. \tag{50.2}
\]

This theorem applies only when the same \(\theta\) is binding across the two
targets.  If each network receives its own free amplitude \(\theta_\nu\),
then for every finite corpus

\[
 \theta_\nu={y_\nu\over g_\nu}
\]

matches every scalar separately whenever \(g_\nu\ne0\).  Thus **no finite
number of one-scalar nets certifies provider rank under per-net rescaling**.
Pooling nets is lawful only after a frozen cross-net alignment/amplitude law
has been proved, not fitted on the same panel.

### 50.2 Unknown rank-one direction: the minimal universal certificate is three

If the common factor direction may also be chosen, a rank-one symmetric
matrix on a \(q\)-dimensional pair chart has \(q\) continuous degrees of
freedom (scale plus projective direction).  On the smallest nontrivial chart
\(q=2\), write its three symmetric coordinates as

\[
 K=\begin{pmatrix}K_{11}&K_{12}\\K_{12}&K_{22}\end{pmatrix}.
\]

Every rank-one candidate obeys

\[
 \det K=K_{11}K_{22}-K_{12}^2=0. \tag{50.3}
\]

Three independent supplies determining \((K_{11},K_{22},K_{12})\) are
therefore sufficient, and generically necessary.  With only two scalar
coordinates, the two-dimensional rank-one cone has an image with interior,
so a generic response can still be matched by changing direction and scale.
The familiar statement that two targets defeat *rescaling* and the statement
that three targets defeat an *arbitrary rank-one provider* are both correct;
they address different model classes.

### 50.3 Exact complete-old rank-\(k\) lower bound

Fix one anchor combination

\[
 X_\omega=\sum_i\omega_i\Lambda_i.
\]

A common support of width \(k\) implies
\(\operatorname{rank}X_\omega\le k\) for every fixed \(\omega\).  Choose an
orthonormal pair chart \(E\in\mathbb R^{n\times q}\), \(q=k+1\), and define
the orthonormal symmetric atoms

\[
 F_{aa}=E e_ae_a^{\mathsf T}E^{\mathsf T},\qquad
 F_{ab}=E{(e_ae_b^{\mathsf T}+e_be_a^{\mathsf T})\over\sqrt2}E^{\mathsf T}.
\]

The simple minor test is lawful after old subtraction when

\[
 \operatorname{Sym}(E)\subset\mathcal O^{\perp}. \tag{50.4}
\]

More generally, (50.4) may be replaced by an explicitly verified
rank-faithful quotient chart
\(L_E:\operatorname{Sym}_q\hookrightarrow\mathcal O^{\perp}\).  Without
this receipt, projecting a rank-\(k\) matrix through an arbitrary old quotient
can create full-rank chart coordinates, so an ordinary determinant is not a
valid provider-rank test.

Schur-project every residual supply against old state and chart-orthogonal
nuisance.  For

\[
 D_q={q(q+1)\over2},
\]

let the resulting measurement matrix be

\[
 T_{\nu,(ab)}
 =\langle Q_OG_\nu,L_E(S_{ab})\rangle_F, \tag{50.5}
\]

and whiten it by the corpus metric, \(\widetilde T=W^{1/2}T\).  If
\(\operatorname{rank}T=D_q\), the chart matrix \(K\) is identifiable.  For
every width-\(k\) common-support surrogate,

\[
 \boxed{
 \|W^{1/2}(y-\widehat y)\|_2
 \ge \sigma_{\min}(\widetilde T)
 \left(\sum_{j>k}\sigma_j(K)^2\right)^{1/2}.}
 \tag{50.6}
\]

This is Eckart--Young after the complete-old and nuisance Schur projections.
For \(q=k+1\), it reduces to

\[
 \|W^{1/2}(y-\widehat y)\|_2
 \ge \sigma_{\min}(\widetilde T)\sigma_{k+1}(K). \tag{50.7}
\]

If response/model uncertainty has whitened norm at most \(\varepsilon\), a
lawful exclusion requires the right side of (50.7) to exceed
\(\varepsilon\).  Equivalently, reconstruction by \(T^+\) requires
\(\sigma_{k+1}(K)>\|T^+\|\varepsilon\).

The number \(D_q\) is the exact minimal universal target count.  If fewer
than \(D_q\) independent scalars are observed, the measurement map has a
nonzero symmetric kernel element \(H\).  For any positive-definite truth
\(K\), choose a nonzero eigenvalue \(\lambda\) of
\(K^{-1/2}HK^{-1/2}\).  Then

\[
 K-\lambda^{-1}H
\]

has the identical measured response and is singular.  Thus no noninjective
linear measurement family can universally distinguish rank \(q\) from rank
at most \(q-1\).  Conversely, \(D_q\) independent supplies reconstruct
\(K\), so the count is also sufficient.

### 50.4 Smallest lawful use of the existing eight-net panel

The target counts implied by (50.5) are

\[
 \begin{array}{c|c|c}
 \text{provider to reject}&q&\text{independent supplies required}\\ \hline
 \text{fixed rank-one scale only}&-&2\\
 \text{arbitrary rank one}&2&3\\
 \text{arbitrary rank two}&3&6\\
 \text{arbitrary rank three}&4&10.
 \end{array} \tag{50.8}
\]

Therefore an eight-target panel can universally reject rank one on a
two-dimensional old-dark chart, and can reject rank two on a
three-dimensional chart.  It cannot universally reject rank three without
additional structure.  The strongest economical allocation is six
independent residual supplies for the \(q=3\) chart and two untouched
holdouts.  The smallest rank-one certificate uses three independent supplies
and leaves five holdouts.

This allocation is compatible with the existing eight nets only under one of
two binding receipts:

1. each selected net exposes the required independent q211 consumer channels
   on the **same within-net state**; or
2. preregistered alignment maps and amplitudes identify one frozen chart
   matrix \(K\) across nets before their target values are examined.

If the bank contains only one q211 scalar per net and each net has its own
costate/amplitude, its eight rows do not constitute tomography of one matrix.
In that case all eight can be spoofed separately and the honest graph status
remains `panel rank unproved`.  The operational certificate is therefore:
verify (50.4) or the generalized rank-faithful chart, residualize every
supply against the complete old arena, require
\(\operatorname{rank}T=3\) (or six) with a reported
\(\sigma_{\min}(W^{1/2}T)\), apply (50.7) on those preregistered discovery
supplies, freeze the resulting provider decision, and only then test its
predicted response ratios on the untouched targets.

## 53. Exact product-fusion theorem and the corrected q211 carrier boundary

This section reconciles the zero-product commutator identity in Section 46
with the full signed-H3 carrier audit.  The important distinction is between
one already-specified q211 scalar and a reusable, source-independent cubic
adjoint.  The former can be free; the latter is not implied by the same
identity.

### 53.1 A typed necessary-and-sufficient theorem for literal product reuse

Let

\[
 C,Z\in\mathbb R^{n\times r},\qquad
 M,A^T\in\mathbb R^{r\times n},
\]

and suppose all four factors have their generic rank (r).  Then

\[
 \boxed{CM=ZA^T}
 \tag{53.1}
\]

if and only if there is one (S\in GL(r)) such that

\[
 \boxed{Z=CS,\qquad A^T=S^{-1}M.}
 \tag{53.2}
\]

Indeed, rank (r) in (53.1) makes the two column spaces equal, so (Z=CS)
for an invertible (S); left cancellation by the full-column-rank (C)
then gives (M=SA^T).  The converse is immediate.  The rank-deficient
version is the same statement on the common active quotient, with the two
null supports matched.  Thus a shared weight matrix, common storage, or a
common latent dimension is not a fusion receipt: the synchronized gauge in
(53.2) is the receipt.

There is a slightly weaker theorem when only the *sum* of two actions is
needed.  In the circuit model that may reuse the already-paid left factor
(C), apply linear preprocessing to its right operand, but may not factor a
new dense output, one width-(r) product exists exactly when

\[
 \operatorname{col}Z\subseteq\operatorname{col}C.
\]

For generic ranks this is (Z=CS), and

\[
 CM+ZA^T=C(M+SA^T). 
\tag{53.3}
\]

The right-factor version is

\[
 \operatorname{row}A^T\subseteq\operatorname{row}M,qquad
 A^T=TM,qquad
 CM+ZA^T=(C+ZT)M. 
\tag{53.4}
\]

Without one of these typed inclusions, the lawful direct circuit is the
width-(2r) block product

\[
 [\,C\;Z\,]
 \begin{bmatrix}M\\A^T\end{bmatrix},
\]

which is two products, not a fusion.  Equations (53.3)--(53.4) are necessary
only in this zero-new-factorization circuit; an accidental low-rank
factorization of the realized sum is not free unless its factors are already
provided.

Complete-old residualization gives a useful algebraic obstruction.  If
(W\succ0), (C^TWC\succ0), and the new factor obeys the Schur condition

\[
 C^TWZ=0,
\tag{53.5}
\]

then a common-left representation (Z=CS) forces
((C^TWC)S=0), hence (S=0) and (Z=0).  Therefore a nonzero feature that
has genuinely been residualized against the paid (C)-span cannot also
claim common-left fusion with that span.

### 53.2 The exact NCQC exception: one product has a symmetric and an
antisymmetric consumer

For symmetric AQ cores (M_i),

\[
 \kappa_{211}(i;j,k)
 =48\operatorname{tr}(M_i^2M_jM_k)
 -8\langle[M_i,M_j],[M_i,M_k]\rangle_F.
\tag{53.6}
\]

Write a common-support consumer as

\[
 \Lambda_{i,jk}=\sum_{a,b=1}^k
 U_{ja}\Theta_{i,ab}U_{kb},
 \qquad N_a=\sum_jU_{ja}M_j,
\tag{53.7}
\]

where each (Theta_i) is symmetric.  After the parallel term, collisions,
and complete-old interface have been removed, the entire AQ q211-perpendicular
scalar is

\[
 \boxed{
 S_{211,\perp}^{AQ}
 =-8\sum_{i,a,b}\Theta_{i,ab}
 \langle[M_i,N_a],[M_i,N_b]\rangle_F.}
\tag{53.8}
\]

The norm-sum simplification is exact only when the (Theta_i) are literally
diagonal.  If the full products

\[
 P_{ia}=M_iN_a,qquad J_{ia}=\operatorname{sym}P_{ia}
\tag{53.9}
\]

are resident at the same AQ seam, then

\[
 \langle[M_i,N_a],[M_i,N_b]\rangle_F
 =4\{\langle P_{ia},P_{ib}\rangle_F
       -\langle J_{ia},J_{ib}\rangle_F\}. 
\tag{53.10}
\]

This is the genuine NCQC-times-singleton fusion: the symmetric view of one
paid (P_{ia}) feeds the compatible Jordan/H3 slot and its antisymmetric view
feeds (53.8).  No second multiplication is needed for this scalar.  A
resident symmetrized product, norm, or Gram value is weaker than a resident
full (P_{ia}) and is not enough to claim every downstream slot.

Equation (53.8) is complete only for the AQ q211-perpendicular scalar.  It is
not the full q211 cumulant unless the (48\operatorname{tr}(M_i^2M_jM_k))
parallel term has been supplied exactly once, and it is not the full true-law
carrier.  With an independent signed cubic defect (Z_{3,p}), the source has
the schematic exact expansion

\[
 S_{211}^{true}
 =S_{211,\parallel}^{AQ}+S_{211,\perp}^{AQ}
  +\sum_p\langle Z_{3,p},G_{211,3,p}\rangle
  +R_{\ge2}(Z_3).
\tag{53.11}
\]

The last two terms are neither determined nor transported by (53.8).

### 53.3 Full reusable H3 state versus one native scalar query

Materializing a source-independent complete cubic adjoint generically exposes
anchor-square words (M_i^2) and factor-pair words (N_aN_b), in addition
to (P_{ia}).  A direct construction across depth (D) has the census

\[
 C_{\rm anchor}=\frac{D r^3}{n^2},
 \qquad
 C_{\rm pair}=\frac{D h r^3}{n^3},
 \quad
 h=\frac{k(k+1)}2
\tag{53.12}
\]

for a full live (Theta) slice, with (h=k) on a literally diagonal slice,
before reset-source evaluation.  At (D=10,r=55,n=1024), the anchor-square
term alone is (1.5867) width-GEMM equivalents unless those words are
resident.  This closes the generic full-adjoint interpretation under the
one-equivalent slack; it does not close a specially factorized scalar.

In particular, for a CP query atom (c^{\otimes3}),

\[
 \langle c^{\otimes3},J_x(M\circ N)\rangle
 =(c\cdot x)(Mc\cdot Nc),
\tag{53.13}
\]

so matvec-native evaluation can bypass explicit matrix products.  Therefore
(53.12) is an exact census for the reusable generic adjoint/direct word
construction, not a universal scalar lower bound.  A proposed scalar bypass
must separately bind its CP rank, tail, reset quadrature, endpoint leverage,
and persistence-row ownership.

### 53.4 Sharp factor fidelity and novelty gates

Define

\[
 \mathcal C_i u=[M_i,\sum_j u_jM_j],
 \qquad G_i=\mathcal C_i^*\mathcal C_i\succeq0.
\tag{53.14}
\]

If a common-factor approximation replaces (Lambda_i) by
(widetilde\Lambda_i), (E_i=\Lambda_i-\widetilde\Lambda_i), its exact
commutator-action error obeys both sharp dual-norm bounds

\[
 |\Delta S|
 \le 8\sum_i\|E_i\|_{op}\operatorname{tr}G_i,
\tag{53.15}
\]

and

\[
 |\Delta S|
 \le8
 \left(\sum_i\|E_i\|_F^2\right)^{1/2}
 \left(\sum_i\|G_i\|_F^2\right)^{1/2}.
\tag{53.16}
\]

Neither yields relative fidelity without a nonzero action floor, because
signed (Theta_i) can cancel.  Quadrature, beyond-AQ, collision, and factor
errors must be bounded separately and then combined in the endpoint metric.

Finally let (f) be the corpus vector produced by (53.8), (P_{\cal O}) the
weighted projector onto the complete old D11/Q24 arena, and
(f_\perp=(I-P_{\cal O})f).  For old-residual target (y_\perp), the exact
one-feature least-squares gain is

\[
 \boxed{
 \Delta_{\rm MSE}
 =\frac{\langle y_\perp,f_\perp\rangle_W^2}
        {\langle f_\perp,f_\perp\rangle_W}.}
\tag{53.17}
\]

Thus a lawful promotion needs: (f_\perp\ne0), the preregistered gain floor,
an endpoint-metric error certificate for (53.15)--(53.16) plus AQ/quadrature,
and literal same-seam product provenance.  The graph split is:

1. `AQ_Q211_PERP_SCALAR_FUSION`: exact theorem, conditional GO on full
   resident (P_{ia}) and complete residualization;
2. `Q211_FULL_H3_CARRIER`: current NO for a generic reusable adjoint under
   the one-equivalent slack;
3. `Q211_FACTORIZED_SCALAR_BYPASS`: OPEN, requiring CP/tail/leverage and
   strict-depth ownership receipts;
4. `AD_K31_PRODUCT_FUSION`: generic NO unless (53.2), (53.3), or (53.4) is
   proved at the synchronized residual seam.

The next decisive question is implementation-independent but concrete: does
the paid recurrence retain the *full* (P_{ia}) at the q211 reset seam, and
does its corpus vector survive the complete-old Schur quotient?  Without
both receipts, the zero-product identity remains a mathematical provider,
not a deployable raw-MSE claim.

## 54. An explicit lawful lower-moment fiber with changing noncommutative q211

The dimension count behind the r55 NCQC fiber can be strengthened to an
analytic witness.  It proves that covariance and every cubic joint moment do
not determine the commutator source even locally.  It does **not** by itself
prove darkness to the complete D11 H4 interface, so it closes only the
lower-moment identifiability question.

Let

\[
 A=\operatorname{diag}(5,6,7,9)
\]

and, for (|t|<1/3), let (S(t)) be the zero-diagonal symmetric star with

\[
 S_{12}=\sqrt{1+2t},\qquad
 S_{13}=\sqrt{1-3t},\qquad
 S_{14}=\sqrt{1+t},
\tag{54.1}
\]

and no other edges.  Put (B(t)=dI+S(t)), where a fixed generic nonzero
(d) makes (B(t)) nonsingular.  Set

\[
 (X_2,X_3,X_4)=(1+2t,1-3t,1+t).
\]

Then

\[
 \sum_jX_j=3,
 \qquad
 11X_2+12X_3+14X_4=37
\tag{54.2}
\]

for every (t).  Since a star has no triangles,
(operatorname{tr}S(t)^3=0).  Consequently every joint trace word in
(A,B(t)) of degree at most three is constant:

\[
\begin{aligned}
 &\operatorname{tr}B=4d,
 &&\operatorname{tr}(AB)=d\operatorname{tr}A,\\
 &\operatorname{tr}B^2=4d^2+6,
 &&\operatorname{tr}(A^2B)=d\operatorname{tr}A^2,\\
 &\operatorname{tr}(AB^2)=d^2\operatorname{tr}A+37,
 &&\operatorname{tr}B^3=4d^3+18d.
\end{aligned}
\tag{54.3}
\]

The pure-(A) traces are of course fixed.  For centered Gaussian quadratic
forms these are exactly the data entering all means, covariances, and joint
third cumulants of the two-variable AQ law.

The noncommutative fourth-order residual is not constant.  Directly,

\[
 \begin{aligned}
 \|[A,B(t)]\|_F^2
 &=2\{1X_2+4X_3+16X_4\}\\
 &=42+12t.
 \end{aligned}
\tag{54.4}
\]

Thus the tangent ((\dot X_2,\dot X_3,\dot X_4)=(2,-3,1)) lies in the exact
kernel of the complete degree-(le3) moment map but has nonzero q211
derivative.  Because (A) has simple spectrum, an orthogonal conjugacy that
fixes (A) can only change coordinate signs; it preserves every (X_j).
The family is therefore not a disguised conjugacy orbit.

Block-summing the same fixed spectator pair embeds (54.1) in dimension 55
without changing any equality in (54.3) or the derivative in (54.4).  A
common scalar shift makes both blocks full rank while leaving commutators
unchanged.  Hence the witness is lawful in an r55 AQ core, not an artifact of
a singular low-dimensional matrix.

The exact conclusion is

\[
 \ker D(C,K_3)\not\subseteq\ker D\,q_{211,\perp}.
\tag{54.5}
\]

In the campaign notation, this proves that the noncommutative commutator is
an independent provider relative to (C,K21)/CP3 data.  To claim future
completion one still needs the stronger restricted map

\[
 h\in\ker D O_{\rm complete},
 \qquad Dq_{211,\perp}h\ne0,
\tag{54.6}
\]

or, for endpoint value, (P_{\ker DO_{\rm complete}}g\ne0).  The star witness
may be visible to K22/CP4 and must not be counted as a complete-interface
dark direction without that extra rank/Schur receipt.  Graph status:
`NCQC_LOWER_MOMENT_Q211_INDEPENDENCE = SOLVED`; `NCQC_COMPLETE_INTERFACE_Q211_DARKNESS = OPEN`.

## 55. Complete-H4 audit of the star fiber: K22 sees it, while the
noncommutative residual survives a C/K3-only CP4 subtraction

The Section 54 witness is deliberately a lower-moment witness.  Its exact H4
audit prevents it from being misreported as an off-diagonal or
complete-interface dark direction.  Write

\[
 X=\operatorname{tr}(A^2B^2),
 \qquad Y=\operatorname{tr}(ABAB).
\]

For (dot X_{\rm edge}=(2,-3,1)),

\[
 \dot X=(25+36)2+(25+49)(-3)+(25+81)1=6,
\tag{55.1}
\]

whereas

\[
 \dot Y=2\cdot5\{6(2)+7(-3)+9(1)\}=0.
\tag{55.2}
\]

The centered Gaussian collision cumulant is

\[
 \kappa_4(q_A,q_A,q_B,q_B)=32X+16Y
 =48X-8\|[A,B]\|_F^2.
\tag{55.3}
\]

Therefore

\[
 \dot\kappa_4=192,
 \qquad
 \frac{d}{dt}\{48X\}=288,
 \qquad
 \frac{d}{dt}\{-8\|[A,B]\|_F^2\}=-96.
\tag{55.4}
\]

In the normalized K22 convention these are the same nonzero direction up to
the fixed slot factor.  A complete observer containing the full K22 plane
thus kills this one-dimensional lower-moment fiber.  Moreover, because the
witness sets (j=k=B), it is a collision/diagonal-slice witness; it is not an
analytic witness for a collision-free off-diagonal q211 block.

There is nevertheless a useful ownership distinction.  A CP4 provider that
is a function only of the fixed (C,K3) data has zero tangent on this family.
A provider that explicitly owns the canonical (48X) direction sees (288).
After subtracting that canonical direction, the genuinely noncommutative
residual still has derivative (-96).  Hence `old owned` follows only if the
actual deployed interface consumes the full K22/Z22 residual at this seam.
The current D11 seam audit instead exposes only its pairing-plane projection;
under that implementation fact the remaining direction is a concrete Z22
candidate, not a proved old feature.

The corrected graph statuses are:

1. `NCQC_CK21_Q211_INDEPENDENCE`: SOLVED by Section 54;
2. `STAR_FIBER_FULL_K22_DARKNESS`: CLOSED -- K22 derivative is nonzero;
3. `STAR_FIBER_OFFDIAG_Q211_WITNESS`: CLOSED -- the construction is a
   (j=k) collision;
4. `D11_COMPRESSED_K22_RESIDUAL_NOVELTY`: OPEN/conditional on the literal
   deployed pairing-only ownership receipt and complete-arena Schur test.

## 56. Vandermonde star theorem: all pairwise H4 faces fixed while the
off-diagonal q211 face moves

The collision defect in Section 55 can be removed analytically.  This gives
an explicit local family whose covariance, all cubic moments, and every
one- or two-coordinate fourth-order face are fixed, while a genuinely
off-diagonal (211) cumulant changes.

Take a seven-dimensional block with one center of eigenvalue (a_0) and six
leaves of distinct eigenvalues (x_1,\ldots,x_6):

\[
 A=\operatorname{diag}(a_0,x_1,\ldots,x_6).
\]

For a leaf vector (v\in\mathbb R^6), let

\[
 S(v)=\begin{bmatrix}0&v^T\\v&0_{6\times6}\end{bmatrix}.
\tag{56.1}
\]

Set (B=S(b)), (C=S(c)), with the base point

\[
 b_j=1,qquad c_j=x_j^3.
\tag{56.2}
\]

Define the five constraints

\[
 F(b)=\left(
 \sum b_j^2,
 \sum x_jb_j^2,
 \sum x_j^2b_j^2,
 \sum x_j^3b_j,
 \sum x_j^4b_j
 \right).
\tag{56.3}
\]

At (b=(1,\ldots,1)), (DF) has rows proportional to the Vandermonde rows
(1,x,x^2,x^3,x^4), hence rank five.  Its one-dimensional kernel is
generated by the barycentric vector

\[
 w_j=\frac1{\prod_{\ell\ne j}(x_j-x_\ell)},
\tag{56.4}
\]

which obeys

\[
 \sum_jw_jx_j^p=0\quad(0\le p\le4),
 \qquad
 \sum_jw_jx_j^5=1.
\tag{56.5}
\]

For example, at (x_j=j),

\[
 w=\frac1{120}(-1,5,-10,10,-5,1).
\]

The implicit-function theorem supplies a local analytic curve (b(t))
through the base point with (F(b(t))=F(b(0))) and (dot b(0)=w).

### 56.1 Why the entire old pairwise moment interface is fixed

The star algebra gives

\[
 \operatorname{tr}S(u)S(v)=2u\cdot v,
\tag{56.6}
\]

every trace of an odd number of star factors equal to zero, and

\[
 \operatorname{tr}S(u)S(v)S(s)S(t)
 =(u\cdot v)(s\cdot t)+(v\cdot s)(t\cdot u).
\tag{56.7}
\]

The first three entries of (F) fix the norm of (b) and the (x)- and
(x^2)-weighted norm needed by all (A,B) cubic, K22, K31, and marginal K4
faces.  The last two entries fix

\[
 b\cdot c=\sum x_j^3b_j,
 \qquad
 \sum x_jb_jc_j=\sum x_j^4b_j,
\tag{56.8}
\]

which are respectively the (B,C) covariance and the mixed cubic
(operatorname{tr}(ABC)).  Equation (56.7) then fixes every K4/K31/K22
face involving only (B,C), because those faces depend only on
(|b|,|c|), and (b\cdot c).  The (A,C) faces are fixed because (C)
is fixed.  Thus the complete collection of marginal and pairwise H4 data,
as well as every joint moment of degree at most three, is constant along the
curve.

### 56.2 The all-distinct 211 action is transverse

For the four slots (A,A,B,C), put

\[
 X=\operatorname{tr}(A^2BC),
 \qquad Y=\operatorname{tr}(ABAC).
\]

The star formulas give

\[
 X=a_0^2\sum x_j^3b_j+\sum x_j^5b_j,
 \qquad
 Y=2a_0\sum x_j^4b_j.
\tag{56.9}
\]

Hence (56.5) implies

\[
 \dot X(0)=1,qquad \dot Y(0)=0,
\tag{56.10}
\]

and therefore

\[
 \boxed{
 \frac d{dt}\kappa_4(q_A,q_A,q_B,q_C)\bigg|_{t=0}
 =\frac d{dt}(32X+16Y)=32.}
\tag{56.11}
\]

Equivalently,

\[
 \frac d{dt}\langle[A,B],[A,C]\rangle_F
 =2\sum_j(a_0-x_j)^2c_jw_j=2,
\tag{56.12}
\]

so the noncommutative residual contributes (-16) while the commuting
parallel word contributes (+48).  Here (B) and (C) are distinct slots:
this is not a K22 collision.

Fixed scalar identities may be added to (B,C), and identical spectator
blocks may be appended, to obtain nonsingular r55 cores.  Expanding the
shifted trace words only introduces the lower-degree quantities already
fixed by (56.3), so the transversality in (56.11) survives.

The resulting exact theorem is

\[
 \ker D O_{\rm pairwise\ H4}
 \not\subseteq
 \ker D q_{211,\rm offdiag}.
\tag{56.13}
\]

This is stronger than the C/K21 count: no collection of marginal K4 and
pairwise K31/K22 faces can reconstruct the off-diagonal 211 provider.  It is
also the sharp analytic explanation for why diagonal K22 collision data
cannot identify a shared off-diagonal (Theta_i) block.

The remaining ownership question is narrower.  A deployed CP4 state might
already carry a genuinely all-distinct quartic coordinate not reducible to
the pairwise faces.  Therefore Section 56 proves
`PAIRWISE_H4_TO_OFFDIAG_Q211 = NO-GO`, while
`COMPLETE_CP4_TO_OFFDIAG_Q211` still requires the literal CP4 interface map
and a Schur test on this explicit tangent.

## 57. Complete linear signed-H3 pullback of q211: the sole generic missing
word is the anchor-square block

Section 53 separated the AQ commutator scalar from its signed-H3 tangent.
The full pullback can now be written explicitly, and the earlier product
census can be sharpened.

Write one AQ variable as

\[
 X_i=I_1(a_i)+I_2(M_i),
 \qquad
 \eta_a=(b_a,N_a)=\sum_jU_{ja}(a_j,M_j).
\]

For three linear--quadratic pairs (	heta_s=(a_s,M_s)), define

\[
 \mathcal C_x(A,B)
 =J_{Bx}(A)+J_{Ax}(B)+2J_x(A\circ B),
\tag{57.1}
\]

and

\[
 \mathcal B(\theta_1,\theta_2,\theta_3)
 =6\operatorname{Sym}(a_1\otimes a_2\otimes a_3)
 +12\sum_{\rm cyc}\mathcal C_{a_1}(M_2,M_3).
\tag{57.2}
\]

The Wiener-chaos contraction is exact:

\[
 E[I_3(Z)X_{\theta_1}X_{\theta_2}X_{\theta_3}]
 =\langle Z,\mathcal B(\theta_1,\theta_2,\theta_3)\rangle.
\tag{57.3}
\]

For

\[
 S_{211}=\sum_{i,j,k}\Lambda_{i;jk}
 \kappa_4(X_i,X_i,X_j,X_k),
 \qquad \Lambda_i=U\Theta_iU^T,
\]

the exact linear signed-H3 pullback at source (p) is

\[
 \boxed{
 G_{3,p}^{211}
 =2\sum_{a,b}\Theta_{p,ab}\mathcal B(\theta_p,\eta_a,\eta_b)
 +2\sum_{i,a,b}U_{pa}\Theta_{i,ab}
   \mathcal B(\theta_i,\theta_i,\eta_b).}
\tag{57.4}
\]

The repeated-anchor part expands as

\[
\begin{aligned}
G_{3,p}^{\rm rep}
={}&12\sum_{ab}\Theta_{p,ab}\operatorname{Sym}(a_p,b_a,b_b)\\
&+24\sum_{ab}\Theta_{p,ab}
 \{\mathcal C_{a_p}(N_a,N_b)
  +\mathcal C_{b_a}(M_p,N_b)
  +\mathcal C_{b_b}(M_p,N_a)\},
\end{aligned}
\tag{57.5}
\]

and the singleton part is

\[
\begin{aligned}
G_{3,p}^{\rm sing}
={}&12\sum_{iab}U_{pa}\Theta_{i,ab}
  \operatorname{Sym}(a_i,a_i,b_b)\\
&+48\sum_{iab}U_{pa}\Theta_{i,ab}
 \{\mathcal C_{a_i}(M_i,N_b)
   +J_{M_ib_b}(M_i)+J_{b_b}(M_i^2)\}.
\end{aligned}
\tag{57.6}
\]

Assume the complete ordered bank

\[
 P_{ia}=M_iN_a
\tag{57.7}
\]

is resident.  Its skew half supplies the AQ commutator action and its
symmetric half (J_{ia}=M_i\circ N_a) supplies every anchor--factor Jordan
term.  More strongly,

\[
 \boxed{
 N_aN_b=\sum_iU_{ia}P_{ib},
 \qquad
 N_a\circ N_b=\sum_iU_{ia}J_{ib}.}
\tag{57.8}
\]

Thus the factor--factor words in (57.5) are exact linear views of the full
(P) bank and require no new latent matrix multiplication.  The only
generic missing dense block is

\[
 \boxed{
 G_{3,p}^{\rm sq}=48\sum_bJ_{b_b}(Q_{pb}),
 \qquad
 Q_{pb}=\sum_i(U\Theta_i)_{pb}M_i^2.}
\tag{57.9}
\]

The neighboring (J_{M_ib_b}(M_i)) term needs a matvec, not a materialized
square.

### 57.1 Exact common-support criterion for recovering the square block

Let (S=\operatorname{col}U), (Q_U=I-P_S), and

\[
 \Phi_M(C)=\sum_{i,j}C_{ij}M_iM_j.
\]

Uniform recovery of an ordered word (Phi_M(C)) from (P) alone is
equivalent to

\[
 CQ_U=0.
\tag{57.10}
\]

When transpose views are allowed, as they are for the Jordan/H3 half, the
necessary-and-sufficient condition is

\[
 \boxed{Q_UCQ_U=0.}
\tag{57.11}
\]

For a realized finite-r core these conditions may be weakened modulo
(ker\Phi_M), but finding and conditioning such accidental coefficients is
not a structural zero-cost receipt.

For (57.9), define

\[
 D_{pb}=\operatorname{Diag}_i((U\Theta_i)_{pb}).
\]

Then universal recovery of every square aggregate is exactly

\[
 \boxed{Q_UD_{pb}Q_U=0\quad\text{for every live }(p,b),}
\tag{57.12}
\]

after replacing (D_{pb}) by its effective old/collision-residual
coefficient.  An individual (M_i^2) requires (e_i\in S).  A nonnegative
full-support diagonal (D) therefore forces (S=\mathbb R^n); only signed
isotropic exceptions can pass at low k.

### 57.2 Collision masks do not commute with the shared rotation

If (Lambda_i=U\Theta_iU^T) is already the exact collision-subtracted slice,
the preceding formulas apply directly.  If an unmasked factorization is
formed first and (j=i,k=i,j=k) are removed afterward, the exact
all-distinct product is

\[
\begin{aligned}
\sum_{\substack{j,k\ne i\\j\ne k}}
 U_{ja}U_{kb}M_jM_k
={}&N_aN_b-U_{ia}M_iN_b-U_{ib}N_aM_i\\
&+2U_{ia}U_{ib}M_i^2-D_{ab},
\end{aligned}
\tag{57.13}
\]

where

\[
 D_{ab}=\sum_jU_{ja}U_{jb}M_j^2.
\tag{57.14}
\]

Thus post-factor collision subtraction creates weighted square corrections;
it does not create an independent (N_aN_b) multiplication.  The correction
must be a literal resident K4/K31/K22 read, pass (57.12), or be charged.

The AQ identity on (Z_3=0) does not determine this normal derivative.  One
may add (langle Z_3,H\rangle) to the chosen parallel extension and subtract
it from the perpendicular extension without changing the AQ identity.
Therefore the lawful cubic residual is

\[
 G_{3,\perp}
 =(I-P_{\rm old})\Pi_{\rm coll}
 \bigl(G_3^{211}-G_{3,\parallel}^{\rm old}\bigr),
\tag{57.15}
\]

using the literal old reverse tape.  Although ([M_i,M_i]=0) removes AQ
commutator collisions, (mathcal B(\theta_i,\theta_i,\theta_k)) contains a
nonzero (J_{a_k}(M_i^2)) term.  Hence AQ collision cancellation does not
remove the signed-H3 square block.  The centered pure-quadratic case
(a_i=0) is the genuine parity exception: its entire linear H3 pullback
vanishes.

A source-specific CP query can still bypass materialization:

\[
 \langle c^{\otimes3},J_b(M_i^2)\rangle
 =(c^Tb)\|M_ic\|^2.
\tag{57.16}
\]

This certifies one scalar action, not a reusable cubic adjoint.

### 57.3 Exact D10/r55 arithmetic and Schur gates

If (m_\times) new (55\times55) products and (m_2) r-squared-scale
matvec/AXPY passes are required per layer, then at depth ten

\[
 \boxed{
 C_{10}=
 \frac{1{,}663{,}750m_\times+30{,}250m_2}
      {1{,}073{,}741{,}824}
 +C_{\rm source}+C_{\rm recurrence}\le1.}
\tag{57.17}
\]

At most 645 new latent products per layer fit when every other term is zero.
All 1,024 anchor squares cost

\[
 \frac{10\cdot55^3}{1024^2}=1.5867,
\]

so the generic reusable pullback is closed.  A CP/native scalar remains
conditional on its own rank, source, and recurrence costs.

For a multifeature candidate (F), complete-old arena (O), corpus weight
(W), and target residual (y), set

\[
 R_O=I-O(O^TWO)^+O^TW,
 \quad S=F^TWR_OF,
 \quad d=F^TWR_Oy.
\]

The exact incremental least-squares gain is

\[
 \boxed{\Delta_{\rm MSE}=d^TS^+d.}
\tag{57.18}
\]

Structural novelty is (S\ne0); endpoint value is (d\ne0).  The
commutator scalar, its signed-H3 tangent, CP4, K4/K31/K22, Z22, and the
existing H3 arena must enter this Schur test jointly.  Local symmetric/skew
orthogonality does not imply response-space novelty.

## 58. A zero-product algebraic view can still consume the entire billing
slack

Equation (57.8) is an identity in any associative algebra.  It also gives
the reverse order from the full bank,

\[
 N_bN_a=\sum_iU_{ib}P_{ia}.
\tag{58.1}
\]

For symmetric cores (P_{ia}^T=N_aM_i); without symmetry, transpose is not
the reverse word.  If only (J_{ia}=\operatorname{sym}P_{ia}) is retained,
only (operatorname{sym}(N_aN_b)) survives.  For example, the product of
(operatorname{diag}(1,-1)) with the two-coordinate swap has zero symmetric
part and nonzero skew part.

Suppose (q) requested result matrices have factor weights
(T_\ell\in\mathbb R^{k\times k}):

\[
 S_\ell=\sum_{a,b}(T_\ell)_{ab}N_aN_b.
\]

Put (W_\ell=UT_\ell).  A contraction-native accumulation is

\[
 \boxed{S_\ell=\sum_{i,b}(W_\ell)_{ib}P_{ib}.}
\tag{58.2}
\]

It avoids (k^2) matrix products, but its classical FMA count per layer is

\[
 qnk^2+qnkr^2.
\tag{58.3}
\]

The product-first streaming alternative forms each linear view and consumes
it immediately, with count

\[
 k^2(n+q)r^2.
\tag{58.4}
\]

The exact classical gate is the minimum of (58.3)--(58.4), plus hook and
source costs.  At (D=10,n=1024,r=55), the first route has normalized cost

\[
 C_A=\frac{10q(k^2+3025k)}{1048576}.
\tag{58.5}
\]

For one requested result this is (0.9919) at (k=34) and (1.6155) at
(k=55).  Hence even a multiplication-free factor view can exhaust the
one-equivalent slack.  Explicitly materializing every factor pair is much
worse: (33.35) equivalents at (k=34) and (87.27) at (k=55), before
the downstream consumer.  The full (P) bank itself contains
(nkr^2) scalars per layer (105.3 million at k34), so a deployable route must
stream/fuse it rather than retain it naively.

The square obstruction in (57.12) is sharp.  With (U=(1,1)^T), a signed
combination can pass: a symmetric combination of (P_1-P_2) recovers
(M_1^2-M_2^2).  A positive combination need not.  Taking

\[
 M_1=\begin{bmatrix}1&0\\0&0\end{bmatrix},
 \qquad
 M_2=\begin{bmatrix}0&1\\1&0\end{bmatrix}
\]

gives no linear combination of (P_1,P_2,P_1^T,P_2^T) equal to
(M_1^2+M_2^2).  This embeds directly in r55.  Accidental span saturation
for one realized bank is not a structural certificate because discovering
its coefficients already requires access to the missing square target.

Finally, fixed projections on the *result matrices* commute with the closure
identity (57.8).  Projections on original component labels do not: the
all-distinct mask produces exactly the square corrections in (57.13).  This
is the precise boundary between lawful common-factor batching and a hidden
collision cost.

## 59. Literal D11 ownership audit: the Vandermonde off-diagonal source is
in the complete old reset kernel

Section 56 left open whether the deployed CP4 state owned an all-distinct
coordinate beyond its pairwise faces.  The literal D11 state map closes that
question.

The carried CP4 state has the rank-one form

\[
 \widehat T=\sum_s\lambda_su_s^{\otimes4}.
\tag{59.1}
\]

Its push step transports only the existing factors (u_s\mapsto W^Tu_s).
At reset, the implementation gates inherited factors and appends only
coordinate births

\[
 (k4_i-\widehat k4_i)e_i^{\otimes4}.
\tag{59.2}
\]

There is no port that acquires a new true all-distinct (iijk) amplitude.
The K31 and K22 planes are reads of degenerations of factors that are already
present; they are not fitted back into a new all-distinct factor.  The
separate star update stores the K31 residual (with its pairing subtraction)
and the diagonal term, while K22 is compressed to one pairing-plane scalar
(alpha).

Along the Section 56 analytic curve:

1. the gate inputs and every marginal (k4_i) are fixed;
2. every K31 and K22 face is fixed;
3. the pairing plane and its scalar (alpha) are fixed;
4. the inherited CP4 factors are fixed.

Consequently, away from irrelevant cap tie surfaces,

\[
 D(\lambda_s,u_s,\mathrm{star}_h,\alpha)\,\dot b=0,
\tag{59.3}
\]

whereas Section 56 gives

\[
 D\kappa_{211,\rm offdiag}\,\dot b=32.
\tag{59.4}
\]

The candidate factor energies are unchanged too, so the conclusion is not
created by a cap-selection discontinuity.  A previously acquired rank-one
CP4 tensor can of course *transport* all-distinct content.  The theorem is
that this D11 reset has no lawful mechanism to *birth* the new direction in
(59.4).

Thus the exact local observability statement is

\[
 \boxed{
 \ker D O_{\rm D11,complete}\not\subseteq
 \ker D q_{211,\rm offdiag}.}
\tag{59.5}
\]

This upgrades the graph status to
`LITERAL_D11_CP4_STAR_TO_OFFDIAG_Q211 = NO-GO / CLOSED PROOF` and
`OFFDIAG_Q211_INDEPENDENT_SOURCE = SOLVED`.  What remains open is no longer
source novelty; it is acquisition and carriage: construct the missing
off-diagonal scalar/state through (53.8), (57.4), or a source-specific native
query while satisfying the one-equivalent cost and endpoint Schur gates.

## 60. Sharp low-cost exceptions for the resident-product closure

The arithmetic in Section 58 admits three important exact refinements.  They
do not alter the generic square obstruction, but they identify the only
credible low-cost deployment geometries.

### 60.1 Symmetric and scalar consumers

If every requested (T_\ell) is symmetric, the final result matrix is
symmetric.  One may accumulate only

\[
 r_s=\frac{r(r+1)}2
\]

entries and mirror once, without separately symmetrizing each (P_{ia}).
Equation (58.5) becomes

\[
 C_{A,\rm sym}
 =\frac{10q(k^2+1540k)}{1048576}.
\tag{60.1}
\]

For (q=1,k=55), this is (0.8367) equivalents, so every rank through 55
fits *for this contraction alone*.  Antisymmetric output similarly uses
(r_a=r(r-1)/2=1485).

If the matrix dies immediately into one fixed linear functional (Q), let

\[
 h_{ib}=\langle Q,P_{ib}\rangle.
\]

After these scalars are available, each common-factor slice costs only the
(UT) formation plus scalar dots.  For a trace query the contraction length
is (r), giving

\[
 C_{\rm trace}
 =\frac{10(k^2+55k)}{1048576},
\tag{60.2}
\]

or (0.02886) at k34 and (0.05770) at k55.  Distinct anchor-dependent
queries restore their target rank unless those queries themselves share a
proved factorization.

The square block has an especially useful cyclic identity:

\[
 \boxed{
 \operatorname{tr}\!\left[
   \left(\sum_i d_iM_i^2\right)N_aN_b\right]
 =\sum_i d_i\langle P_{ia},P_{ib}\rangle_F.}
\tag{60.3}
\]

Thus a square aggregate that dies immediately into factor-pair traces does
not require a matrix-valued square bank.  This does not help a square matrix
that must persist or act on a query outside the factor-pair span.

For the q211 commutator itself, diagonal (Theta_i) needs only skew norms.
The depth-ten cost of reading the upper skew triangles is

\[
 C_{\rm skew,diag}=\frac{10k\,1485}{1048576},
\tag{60.4}
\]

which is (0.4815) at k34 and (0.7787) at k55, before source/recurrence
cost.  A dense symmetric (Theta) needs the entire skew Gram and costs

\[
 C_{\rm skew,dense}
 =\frac{10\,1485\,k(k+1)}{2\cdot1048576};
\tag{60.5}
\]

low factor rank or a resident Gram is therefore required.

### 60.2 Exact quotient dimension and approximation residual

Let

\[
 \mathcal K_U
 =\{w:Q_U\operatorname{Diag}(w)Q_U=0\}.
\]

For a generic (k)-plane in (mathbb R^n),

\[
 \dim\mathcal K_U
 =n-\min\left\{n,\frac{(n-k)(n-k+1)}2\right\}.
\tag{60.6}
\]

At width 1,024 and k34 or k55 this is generically zero: transpose views
recover no nonzero diagonal square direction.  With an old square-summary
space (W_0), the exact recoverable coefficient space is

\[
 E=W_0+\mathcal K_U.
\]

If collision weights span (H=\operatorname{span}\{u_a\odot u_b\}), the
minimal number of genuinely new square summaries is

\[
 \boxed{\dim((H+E)/E).}
\tag{60.7}
\]

The coefficient-norm projection is also exact.  With (P) only, the
irreducible ordered residual is (CQ_U); with transpose/reverse views it is

\[
 \boxed{Q_UCQ_U.}
\tag{60.8}
\]

For a flat collision (D\propto I), its relative Frobenius residual is

\[
 \sqrt{\frac{n-k}{n}},
\tag{60.9}
\]

namely (0.9833) at k34 and (0.9728) at k55.  Common factors alone leave
nearly all of this direction; the flat diagonal loophole works only when the
total-square summary is separately owned.

For an anchored all-distinct mask with dense weight (w), the square-weight
vectors are

\[
 g_i=2w_i e_i-w.
\]

The matrix with these columns is

\[
 2\operatorname{Diag}(w)-w\mathbf1^T,
\]

whose determinant is

\[
 2^n\left(\prod_iw_i\right)(1-n/2).
\tag{60.10}
\]

For (n\ne2) and full-support (w), the anchored masks span every square
direction.  Aggregating anchors first can collapse this to the target
costate dimension, which is why target-specific scalars remain possible
even though a universal carrier is closed.

### 60.3 Schur-algebra closure: the exact structured escape

Let (S=\operatorname{col}U) be a unital algebra under entrywise product,
let every requested pair weight (w=u_a\odot u_b) lie in (S), and let the
downstream anchor costates (c) also lie in (S).  After aggregating anchors,
the all-distinct square residue has weight

\[
 T_{w,c}=2(c\odot w)-(\mathbf1^Tc)w,
\tag{60.11}
\]

which remains in (S).  Therefore an old square-summary bank (W_0\supseteq
S) closes every collision residue with only (dim S=k) summaries.

Unital Schur subalgebras are precisely block-constant coordinate spaces.
Walsh/character columns from a product-closed subgroup are the canonical
orthogonal realization:

\[
 u_a\odot u_b=n^{-1/2}u_{a\oplus b}.
\tag{60.12}
\]

For diagonal (Theta), (u_a^{\odot2}=n^{-1}\mathbf1), so one resident
total-square matrix closes every diagonal collision.  Cross modes or
per-anchor masks reopen the obstruction unless the full Schur algebra and
its square summaries are owned.

This yields the sharp structured GO condition:

1. factor/costate columns form one synchronized unital Schur algebra;
2. anchors are aggregated before square materialization;
3. the old state contains one square summary per algebra block;
4. the scalar or symmetric contraction costs (60.1)--(60.5), reset source,
   and recurrence jointly fit the cap.

Common (U) without these four receipts is not enough.

## 61. Normal form of the new off-diagonal CP4 tangent and its minimal read
interface

The Section 56 tangent has an especially simple quartic normal form.  On the
three-coordinate span ((A,B,C)), every support-at-most-two coordinate is
fixed, and star parity also gives

\[
 \dot T(A,B,B,C)=\dot T(A,B,C,C)=0.
\]

The sole nonzero coordinate is

\[
 \dot T(A,A,B,C)=32.
\tag{61.1}
\]

Hence the diagonal quartic polynomial is

\[
 \boxed{
 \dot p(\alpha,\beta,\gamma)
 =\dot T[(\alpha A+\beta B+\gamma C)^4]
 =384\alpha^2\beta\gamma.}
\tag{61.2}
\]

In the two cyclic trace necklaces

\[
 X=\operatorname{tr}(A^2BC),
 \qquad Y=\operatorname{tr}(ABAC),
\]

the tangent is ((\dot X,\dot Y)=(1,0)).  The Gaussian cumulant row
((32,16)) acts by 32, the commutator row (2(X-Y)) acts by 2, and the
parallel/residual split acts by (+48,-16).

One unrestricted mixed-slot scalar (T_{AABC}) is necessary and sufficient
to see this one-dimensional fiber.  The complete 211 quotient on three
labels has coordinates (AABC,ABBC,ABCC); over n labels it has

\[
 3\binom n3=\frac{n(n-1)(n-2)}2
\tag{61.3}
\]

coordinates, in addition to (\binom n4) all-distinct 1111 coordinates.
Thus a universal exact interface is cubic-state-sized.

### 61.1 Six diagonal words are necessary and sufficient

If only diagonal CP4 words (p(v)=T[v^4]) may be read, exact polarization is

\[
\boxed{
\begin{aligned}
T_{AABC}=\frac1{48}\{&p(A+B+C)-p(A+B-C)\\
&-p(A-B+C)+p(A-B-C)\\
&-2p(B+C)+2p(B-C)\}.
\end{aligned}}
\tag{61.4}
\]

Six is minimal: the real symmetric Waring rank of the monomial
(\alpha^2\beta\gamma) is six (the complex monomial-rank lower bound is six,
and (61.4) attains it over the reals).  The standalone tangent therefore has
an exact signed-CP decomposition on these six rays.  A single generic ray can
*detect this known one-dimensional tangent* but cannot reconstruct an
arbitrary 211 coordinate.

For a diagonal-word state evaluated at rays (u_s),

\[
 Dp(u_s)[h]=384u_{A,s}^2u_{B,s}u_{C,s}.
\tag{61.5}
\]

Coordinate rays and all support-at-most-two rays annihilate the tangent; a
generic ray with all three coordinates nonzero sees it.

### 61.2 Encoder information theorem and resident-factor exception

Let

\[
 L=(C,K3,d,K31,K22)
\]

contain covariance, cubic data, marginal fourth moments, and every pairwise
K31/K22 face.  Section 56 proves (DL[h]=0).  Therefore every deterministic
or differentiable CP4 encoder, covariance-pairing/star provider, or Q24 read
whose literal inputs factor only through (L) has zero derivative on h,
regardless of its nominal rank or how dense its decoded tensor appears.
This is an encoder-information obstruction, not a decoder-capacity bound.

A true resident signed CP state is different:

\[
 T_{CP}=\sum_s\lambda_su_s^{\otimes4},
 \qquad
 T_{CP,AABC}=\sum_s\lambda_su_{A,s}^2u_{B,s}u_{C,s}.
\tag{61.6}
\]

This mixed read is an (O(r_{CP})) Hadamard dot and uses no dense product.
For moving factors, its exact derivative includes the lambda derivative and
all four factor derivatives; if the state represents the true tensor, their
sum must equal 32.  Nominal CP rank alone proves nothing about whether the
encoder ever acquired this tangent.  A Q24 implementation consuming genuine
resident full CP factors is outside the factor-through-L annihilation theorem
and must be audited through its Jacobian.

On a lawful tangent space (H), let (J=DO_{old}) and
(g=D(T_{AABC})).  Exact old ownership is equivalent to each of

\[
 \ker J\subseteq\ker g,
 \qquad g\in\operatorname{row}J,
 \qquad g(I-J^+J)=0.
\tag{61.7}
\]

With tangent covariance (Sigma), the exact Schur residual is

\[
 s=g\Sigma g^T-g\Sigma J^T(J\Sigma J^T)^+J\Sigma g^T.
\tag{61.8}
\]

For (Sigma) positive definite on (H), (s=0) means owned and (s>0)
means novel.  The Section 56 h has (Jh=0,gh=32), giving the literal D11
novelty certificate in Section 59.

### 61.3 One dense rotation makes the hidden source visible later

For arbitrary coordinate rows (u,v,z), the first transported faces are

\[
 \dot K4(u)=384u_A^2u_Bu_C,
\tag{61.9}
\]

\[
 \dot K31(u,v)=96\{u_A^2(u_Bv_C+u_Cv_B)
 +2u_Av_Au_Bu_C\},
\tag{61.10}
\]

and

\[
\begin{aligned}
\dot K22(u,v)=64\{&u_A^2v_Bv_C+v_A^2u_Bu_C\\
&+2u_Av_A(u_Bv_C+u_Cv_B)\}.
\end{aligned}
\tag{61.11}
\]

The corresponding q211 read follows by the four-slot symmetrization of
(A,A,B,C).  These identities explain the mechanism: same-site coordinate
and pairwise readers are blind, while one generic dense transition makes a
state that actually carried h visible in ordinary K4/K31/K22 faces.

All constants in this section use raw cumulants.  A deployed tensor stored
with the (1/4!\) convention scales every action uniformly by (1/24) and
does not change any rank, ownership, or novelty conclusion.

### 61.4 After the already-paid pairwise quotient, the exact CP rank is two,
not six

Equation (61.4) reconstructs the full tensor with zero leakage into pairwise
faces.  The campaign, however, works after those faces have already been
owned and Schur-subtracted.  Modulo support-at-most-two monomials,

\[
 \boxed{
 (\alpha+\beta+\gamma)^4+(\alpha-\beta-\gamma)^4
 \equiv24\alpha^2\beta\gamma.}
\tag{61.12}
\]

Every omitted term in (61.12) uses at most two labels and is canceled by the
resident d/K31/K22/star/pair interface.  Therefore the raw tangent
(384\alpha^2\beta\gamma) has an exact two-ray quotient representation with
rays

\[
 u_+=(1,1,1),\qquad u_-=(1,-1,-1)
\]

and weights 16,16.

Rank one is impossible in the full three-coordinate q211 quotient.  A ray
(u=(\alpha,\beta,\gamma)) contributes, up to the common slot factor, the
three-vector

\[
 \alpha\beta\gamma(\alpha,\beta,\gamma)
\tag{61.13}
\]

to ((AABC,ABBC,ABCC)).  A nonzero AABC entry from one ray necessarily
brings at least one of the other two entries.  The tailored pair above
cancels them and leaves only AABC.  Three generic rays span the entire
three-dimensional quotient; two special rays suffice for this coordinate.

The exact hierarchy is therefore:

1. one direct mixed scalar observes the known one-dimensional tangent;
2. one generic diagonal word detects it but does not isolate it;
3. two tailored diagonal words/CP rays represent AABC exactly modulo the
   already-owned pairwise interface;
4. six reads/rays represent the full tensor with no pairwise leakage.

This quotient-rank-two identity is the smallest lawful CP4 birth mechanism
for the Section 56 missing source.  Its deployment still requires a way to
obtain the signed amplitude and the two moving rays from lawful state; the
algebraic carrier itself is exact.

There is one binding seam caveat.  After scaling to match
(384\alpha^2\beta\gamma), the two rays leak

\[
\begin{aligned}
&32\alpha^4+192\alpha^2\beta^2+192\alpha^2\gamma^2
 +32\beta^4+128\beta^3\gamma\\
&\qquad+192\beta^2\gamma^2+128\beta\gamma^3+32\gamma^4.
\end{aligned}
\tag{61.14}
\]

All of this is pairwise, but cancellation requires a *complete* pairwise H4
residual interface.  Deployed D11 compresses K22 largely to the pairing
direction, so membership of (61.14) in the literal cancelable star/pair span
must be proved.  Without that receipt the unconditional no-leak construction
remains the six-ray block in (61.4).  Thus quotient rank two is conditional
on full pairwise cancellation; full-tensor rank six is unconditional.

### Graph-ready split

- `Q211_FIXED_FACTOR_TWO_TARGET` -- **solved theorem** by (50.1)--(50.2).
- `Q211_PER_NET_SCALAR_CORPUS` -- **closed no-go**: no finite corpus defeats
  independently refitted per-net amplitudes.
- `Q211_COMPLETE_OLD_RANK_K_BOUND` -- **solved conditional theorem** by
  (50.4)--(50.7); the rank-faithful quotient chart is binding.
- `Q211_EIGHT_NET_PANEL_CERTIFICATE` -- **open structural receipt**: prove
  same-state consumer multiplicity or frozen cross-net alignment, then report
  residual design rank and smallest singular value.  Eight unrelated scalar
  endpoints do not satisfy the receipt.

### 61.5 Rank two is exact only on the restricted CP4 + Star4 + pairing gauge

For the Vandermonde star, (C_{AB}=C_{AC}=0).  Write

\[
 s_1=C_{AA}C_{BB},\quad
 s_2=C_{AA}C_{CC},\quad
 s_3=C_{BB}C_{CC}+2C_{BC}^2,
\]

and

\[
 \rho_{BC}=\frac{C_{BC}}{\sqrt{C_{BB}C_{CC}}}.
\tag{61.15}
\]

The covariance-pairing plane has K22 direction
(S=(s_1,s_2,s_3)) and q211 direction
((C_{AA}C_{BC},0,0)).  For any desired pre-pair amplitude (q'), choose an
arbitrary (p_0>0) and set

\[
 \beta=\sqrt{p_0}(s_1/s_2)^{1/4},\qquad
 \gamma=\sqrt{p_0}(s_2/s_1)^{1/4},
\]

\[
 \alpha=1,\qquad
 \alpha'=-\frac{p_0\sqrt{s_1s_2}}{s_3},
 \qquad \delta=\alpha-\alpha',
\]

\[
 t=q'/\delta,\qquad
 \lambda=t/p_0,\qquad
 \mu=-t/(\alpha'p_0).
\tag{61.16}
\]

The two signed CP rays

\[
 u=(\alpha,\beta,\gamma),
 \qquad v=(\alpha',\beta,\gamma)
\]

with weights (lambda,mu) have q211 quotient vector exactly
((q',0,0)) and K22 leakage

\[
 \eta S,\qquad \eta=q'/\sqrt{s_1s_2}.
\]

Subtracting (eta) times the resident covariance-pairing plane cancels this
leakage and leaves

\[
 q'(1-\rho_{BC})e_A.
\tag{61.17}
\]

Taking

\[
 q'=32/(1-\rho_{BC})
\]

therefore produces the exact Section 56 tangent.  The remaining marginal
and K31 leakage lies in the resident Star4 plane and has no triplet-support
q211 component, so it can be removed without changing (61.17).

Strict nondegeneracy is analytic.  Here

\[
 \rho_{BC}=\frac{b\cdot c}{\|b\|\|c\|},
 \qquad b=(1,\ldots,1),\quad c=(x_j^3),
\]

and distinct (x_j) make (b,c) nonproportional.  Strict Cauchy--Schwarz
gives (|\rho_{BC}|<1).  Thus the two-ray construction is exact on the
three-label restriction of the CP4 + Star4 + covariance-pairing gauge.
Rank one remains impossible because its unwanted ABBC and ABCC components
cannot both be canceled by that restricted pairing direction.

Section 65.7 gives the binding width-1024 audit: one deployed global pairing
coefficient cancels the two-ray leakage only if the complete-plane identity
(65.29) holds.  Therefore
`VSTAR_CP4_STAR_PAIR_RANK2 = SOLVED LOCAL / GLOBAL CONDITIONAL`, and the
four-ray coordinate-Star4 or six-ray no-leak constructions are the exact
global fallbacks.

## 65. Deployable moving birth for the rank-two q211 quotient

Section 61 proves decoder capacity.  A deployable provider additionally needs
an autonomous source amplitude and oriented frame, an exact insertion into the
existing CP4 reset, protected transport through the cap, and positive endpoint
value after the complete-old quotient.  These are logically separate claims.

### 65.1 Algebraic birth map and a balanced two-ray gauge

At reset (l), let the old post-reset fourth-order state be

\[
 T_l^0=G_l^{CP}+\operatorname{Star4}(H_l)+\alpha_lP_{C_l}.
\tag{65.1}
\]

Suppose a lawful source oracle supplies an oriented frame ((A_l,B_l,C_l))
and the residual mixed coordinate

\[
 a_l=
 (T_l^{\rm target}-T_l^0)(A_l,A_l,B_l,C_l),
\tag{65.2}
\]

with the other two q211 coordinates removed.  The Section 56 frame has
(C_{AB}=C_{AC}=0).  Put

\[
 s_1=C_{AA}C_{BB},\qquad
 s_2=C_{AA}C_{CC},\qquad
 s_3=C_{BB}C_{CC}+2C_{BC}^2,
\tag{65.3}
\]

and (\rho=C_{BC}/\sqrt{C_{BB}C_{CC}}).  The free (p_0) gauge in
(61.16) has a useful balanced choice

\[
 p_0=\frac{s_3}{\sqrt{s_1s_2}}.
\tag{65.4}
\]

It makes (\alpha=1), (\alpha'=-1), and gives the equal-norm rays

\[
 u_+=A+\sqrt{\frac{s_3}{s_2}}B
          +\sqrt{\frac{s_3}{s_1}}C,
\qquad
 u_-=-A+\sqrt{\frac{s_3}{s_2}}B
          +\sqrt{\frac{s_3}{s_1}}C.
\tag{65.5}
\]

For the moment keep the frame orientation fixed and define

\[
 q_l'=\frac{a_l}{1-\rho_l},\qquad
 \lambda_{l,+}=\lambda_{l,-}
 =\frac{q_l'\sqrt{s_1s_2}}{2s_3},
\qquad
 \eta_l=\frac{q_l'}{\sqrt{s_1s_2}}.
\tag{65.6}
\]

Then

\[
 \Delta G_l^{CP}
 =\lambda_{l,+}u_+^{\otimes4}
  +\lambda_{l,-}u_-^{\otimes4}
\tag{65.7}
\]

has q211 vector ((q_l',0,0)) and K22 leakage (\eta_lS).  Insert it
into the literal reset by the simultaneous updates

\[
 \Delta\alpha_l=-\eta_l,
\tag{65.8}
\]

\[
 \Delta H_{ii}=-(\Delta d^{CP})_i+3\eta_lC_{ii}^2,
\qquad
 \Delta H_{ij}=-(\Delta K31^{CP})_{ij}
                  +3\eta_lC_{ii}C_{ij}.
\tag{65.9}
\]

Equations (65.8)--(65.9) leave every total marginal, K31, and K22
face unchanged.  Star4 has no triplet-support q211 coordinate, while the
pairing adjustment contributes (-\eta_lC_{AA}C_{BC}).  Therefore the
only surviving change is

\[
 q_l'(1-\rho_l)=a_l
\tag{65.10}
\]

in the required AABC slot.  This is an exact amendment of the existing
`CP4 + Star4 + alpha P_C` reset, not a second addition to its visible planes.

For storage, let

\[
 r_l^2=1+s_3/s_2+s_3/s_1.
\]

Replacing (u_\pm) by (u_\pm/r_l) and each lambda by
(\lambda r_l^4) preserves (65.7) exactly and gives unit-norm factor
columns.  This removes the arbitrary CP scaling from finite-state range.

### 65.2 The source-acquisition claim is separate and currently blocking

The old tuple

\[
 (C,K3,d,K31,K22,\operatorname{Star4},P_C)
\]

has zero Jacobian on the Vandermonde tangent.  It can construct every
coefficient in (65.3)--(65.9) *after* (a_l) and the oriented frame are
provided, but it cannot construct (a_l) itself.  Consequently a lawful
moving birth requires one of:

1. a forward mixed fourth-coordinate oracle at the reset;
2. an exact oriented AQ/NCQC word evaluation
   (32\operatorname{tr}(A^2BC)+16\operatorname{tr}(ABAC)); or
3. another label-free source map whose Jacobian contains the same residual
   covector after old-state subtraction.

An endpoint-contracted reverse scalar is sufficient to score one frozen
endpoint, but is not an autonomous forward CP state unless its frame and
amplitude are label-free and available at inference.  Likewise, Section 56
proves existence of a source direction, not that the true width-1024 residual
has rank one.  If the source quotient has (R) independent factorized modes,
the exact carrier requires (2R) moving rays (or (6R) for the no-pairing
fallback), and the source must supply all (R) amplitudes and frames.

For a known forward AQ triple there is an exact source formula.  With

\[
 P_{AB}=M_AM_B,\qquad P_{AC}=M_AM_C,
\]

the full Gaussian q211 word is

\[
 q_{ABC}^{AQ}
 =32\langle P_{AB},P_{AC}\rangle_F
  +16\langle P_{AB}^T,P_{AC}\rangle_F.
\tag{65.11a}
\]

Subtracting the resident CP4/Star4/pairing mixed read gives the lawful
(a_l) in (65.2).  If the full products (P_{AB},P_{AC}) are resident at the
same reset, acquisition is only (O(r^2)) contraction work.  If either must
be formed, the missing matrix products are acquisition cost and cannot be
charged to the two-ray decoder.  For a general factorized source dictionary,
this cost and the number of amplitudes scale with its mode rank (R).

A lawful target-specific bridge is possible without confusing a reverse
scalar with forward state.  A Bellman/q211 costate computed only from known
network weights may preregister a small deterministic dictionary of frames
((A_p,B_p,C_p)).  The *forward* AQ core must then evaluate (65.11a)
separately for every chosen frame and supply its amplitude.  For one frame
this needs at most two (r\times r) products; the prohibitive (nr^3) census
returns only when modes across all anchors are requested.  The dictionary
must be frozen independently of residual targets and must transfer across the
panel.  Complete-old Schur value, not reverse leverage alone, decides whether
the selected modes are useful.

Thus

\[
 \boxed{\text{rank-two decoding is solved; source acquisition is not.}}
\tag{65.11}
\]

The actual reset insertion is

\[
 U_{\rm cand}=[\,U_g\;I\;u_+\;u_-\,],
\qquad
 \lambda_{\rm cand}
 =[\,\lambda_g\;\lambda_{\rm coord}\;
     \lambda_+\;\lambda_-\,],
\tag{65.12}
\]

followed by (65.8)--(65.9).  Both custom rays must be retained or removed
as one group.  Dropping one destroys the q211 cancellations and invalidates
the visible-plane closure.

### 65.3 Exact signed-orientation stabilization of kappa

In the fixed frame, the apparent condition number is

\[
 \kappa_l=\frac1{|1-\rho_l|}.
\tag{65.13}
\]

It need not blow up.  The represented physical tensor is invariant under

\[
 C_l^\circ=\epsilon_l C_l,qquad
 a_l^\circ=\epsilon_l a_l,qquad
 \epsilon_l\in\{-1,1\},
\tag{65.14}
\]

because (a_l^\circ A_l^2B_lC_l^\circ=a_lA_l^2B_lC_l).  Signed CP rays
allow this gauge at no arithmetic or rank cost.  Choose

\[
 \epsilon_l=-\operatorname{sign}(\rho_l)
\]

(either sign at zero).  Then

\[
 \rho_l^\circ=-|\rho_l|,qquad
 q_l'=\frac{\epsilon_la_l}{1+|\rho_l|},
\qquad
 \boxed{\kappa_l^\circ=\frac1{1+|\rho_l|}\le1.}
\tag{65.15}
\]

Hence the dangerous (1-\rho) cancellation is a removable orientation
artifact, not an intrinsic instability of the signed rank-two carrier.  If
an external convention forbids the singleton sign flip, use the exact
six-ray construction (61.4), which has no (1-\rho) division.

If inconsistent covariance values (\rho,\widehat\rho) are used by the CP
and pairing branches, the delivered amplitude is

\[
\widehat a_{\rm del}
 =\widehat a\frac{1-\rho}{1-\widehat\rho}.
\tag{65.16}
\]

More exactly, with (d=1-\rho),
(\widehat a=a+\delta a), and
(\widehat\rho=\rho+\delta\rho),

\[
 \widehat q'-q'
 =\frac{d\,\delta a+a\,\delta\rho}
        {d(d-\delta\rho)},
\qquad
 |\widehat q'-q'|
 \le\frac{|\delta a|+|q'|\,|\delta\rho|}
            {|d|-|\delta\rho|}.
\tag{65.16a}
\]

The signed orientation replaces (d) by (1+|\rho|\ge1), making the bound
uniform whenever both branches use the same sign convention.

Thus the two branches must consume the same stored covariance bytes and
orientation.  In the signed gauge the first-order amplification of amplitude,
covariance, and roundoff errors is at most order one; in the fixed gauge it
is multiplied by (\kappa_l) (and the covariance error in (q_l') by
(|a_l|\kappa_l^2)).  At (|\rho|=1), the two singleton directions are
linearly dependent and AABC is a pairwise K22 direction.  Its complete-old
Schur residual is zero; the correct action is to close it as owned, not to
form a large carrier.

### 65.4 Transport through depth and reset-model error

Let the deployed CP factor map between two reset seams be

\[
 \mathcal T_m u=D_{g_m}W_m^Tu.
\tag{65.17}
\]

A birth at layer (l) therefore moves as

\[
 u_{l,\pm}^{(m)}
 =\mathcal T_{m-1}\cdots\mathcal T_lu_{l,\pm},
\qquad
 \lambda_{l,\pm}^{(m)}=\lambda_{l,\pm}.
\tag{65.18}
\]

Affine transport preserves the two-ray rank exactly.  At later resets the
ordinary CP4 split rereads the transported factors and moves their visible
plane contribution out of Star4/pairing exactly once.  No new q211 tensor
state is needed.

This is exact for the deployed diagonal-gate CP law.  It is not a theorem
that the true ReLU fourth-cumulant map equals that law.  If (E_m^{reset}) is
the custom-source reset defect and (Q_{e,m}) the pulled-back endpoint
costate, the exact endpoint error bound is

\[
 |e_{e}^{reset}|
 \le\sum_m
 \|Q_{e,m}\|_*\,\|E_m^{reset}\|,
\tag{65.19}
\]

or the corresponding transported operator-norm bound when the defect is
measured at birth.  This reset fidelity is a separate promotion gate.

### 65.5 Fixed-cap displacement versus reserved-rank expansion

The incumbent truncator ranks individual CP atoms by their Frobenius energy.
That rule is unsafe for (65.7): the two atoms and pairing correction can be
large while their lawful quotient is small.  The pair must have one group ID
and be selected atomically.

There are two exact deployment choices.

**Fixed cap.**  Reserve two slots and remove two incumbent rays before the
ordinary candidate sort.  This adds no dense factor rank, but the scientific
candidate is

\[
 \Delta T_l^{deploy}=\Delta T_l^{birth}-D_l^{inc},
\tag{65.20}
\]

not the birth alone.  Here (D_l^{inc}) is the complete all-distinct content
of the displaced rays after their visible planes are reclosed into Star4 and
pairing.  Endpoint grading and Schur residualization must include the minus
term.  Individual energy is not a value bound for it.

**Reserved-rank expansion.**  Increase the cap by two for every live birth.
Then the usual rank-1024 coordinate drop is unchanged and no incumbent
all-distinct state is sacrificed.  If (b_m) custom rays are live at edge
(m), the exact dominant increment for a push plus one K31 and one K22 read is

\[
 \Delta F_{dense}
 =(6n^2-n)\sum_m b_m.
\tag{65.21}
\]

Relative to one width-(n) GEMM, its exact ratio is

\[
 \frac{(6n^2-n)\sum_mb_m}{n^2(2n-1)}.
\tag{65.22}
\]

For ten consecutive rank-two births retained through a conservative ten
reads, (\sum_mb_m=2(1+\cdots+10)=110).  At (n=1024), (65.21) is
(691,947,520) arithmetic, or about (0.3224) width-GEMM equivalents, before
source acquisition, hooks, scalar work, and literal metering.  The exact
six-ray fallback triples this dominant increment to about (0.9671).  Maximum
persistent factor memory is only

\[
 20(1024+1)4=82,000\ \text{bytes}
\]

for the two-ray lane, or (246,000) bytes for six rays.  These figures prove
that rank carriage is cheap; they do not price the missing source oracle or
grant a total-budget pass.

### 65.6 Complete-old Schur score and endpoint value

Let (B_l) denote the *net* deployed birth, including (65.20) when the cap is
fixed, and let (\mathcal R_{e:l}^{(4)}) be its deployed transport to endpoint
(e).  Its exact endpoint feature is

\[
 f_e=\sum_l
 \left\langle Q_e,
 \mathcal R_{e:l}^{(4)}B_l\right\rangle.
\tag{65.23}
\]

With resident factors, each direct mixed CP read is the Hadamard dot in
(61.6); a dense endpoint contraction is zero-new-product only when the
corresponding (Q_eU) products are already resident.

Across the frozen corpus, let (F) contain the proposed birth features and
(O) the complete old D11/Q24/CP3/depth design, including the incumbent CP4
state and all refittable old Jacobian columns.  In metric (W), define

\[
 F_\perp=(I-P_O^W)F,
 \qquad y_\perp=(I-P_O^W)y.
\tag{65.24}
\]

The exact jointly refitted MSE gain is

\[
 \boxed{
 \Delta_{MSE}
 =y_\perp^TWF_\perp
  (F_\perp^TWF_\perp)^+
  F_\perp^TWy_\perp.}
\tag{65.25}
\]

For one feature this reduces to

\[
 \Delta_{MSE}
 =\frac{\langle y_\perp,f_\perp\rangle_W^2}
        {\langle f_\perp,f_\perp\rangle_W}.
\tag{65.26}
\]

The provider is old-owned exactly when (F_\perp=0).  A physics-fixed unit
coefficient improves MSE only when

\[
 2\langle y_\perp,f_\perp\rangle_W
 >\|f_\perp\|_W^2.
\tag{65.27}
\]

Neither exact rank-two algebra nor cheap transport implies either inequality.
To reach raw MSE (10^{-8}) from raw (2.9409\times10^{-8}), the joint Schur
gain from this lane and any stack partners must be at least
(1.9409\times10^{-8}); that is the endpoint-value gate, not the local
(a_l\ne0) gate.

### Graph-ready split

1. `VSTAR_RANK2_BALANCED_RESET_INSERTION` -- **solved exact algebra** by
   (65.3)--(65.10).
2. `VSTAR_RANK2_SOURCE_ORACLE` -- **open/blocking**: supply label-free
   amplitudes and oriented frames; low moments and pairwise Q24 cannot.
3. `VSTAR_RANK2_SIGN_GAUGE_STABILITY` -- **solved**: signed orientation gives
   (\kappa\le1); same-byte covariance is binding.
4. `VSTAR_RANK2_CP_TRANSPORT` -- **solved for the deployed CP gate law**;
   true-reset fidelity remains open via (65.19).
5. `VSTAR_RANK2_CAP_POLICY` -- **conditional**: either grade the displaced
   incumbent term or reserve rank; group-atomic retention is mandatory.
6. `VSTAR_RANK2_COMPLETE_OLD_SCHUR` -- **open endpoint receipt**: require
   (F_\perp\ne0) and the gain in (65.25) at the campaign threshold.

### 65.7 Local-to-global correction: rank two does not fit one global pairing
coefficient generically; rank four is the exact deployable birth

Equations (65.3)--(65.10) are exact on the three-label restricted plane.
The deployed pairing gauge, however, has one *global* scalar (alpha) on

\[
 S^{global}_{ij}=C_{ii}C_{jj}+2C_{ij}^2.
\tag{65.28}
\]

Let (B_{22}^{(2)}) be the complete width-(n) K22 plane leaked by the two
new CP rays.  A scalar pairing update is lawful if and only if

\[
 \boxed{B_{22}^{(2)}=\eta S^{global}}
\tag{65.29}
\]

on the literal strict-off-diagonal mask.  Matching the three entries
((AB,AC,BC)) proves only the restriction of (65.29).  Outside that block,
the two-ray plane is zero or rank at most two, whereas a generic
(vv^T+2C\odot C) plane is dense and has rank greater than two.  Changing the
single deployed (alpha) therefore perturbs K22 entries which Star4 cannot
repair.  Equivalently, with

\[
 \eta_*=\frac{\langle B_{22}^{(2)},S^{global}\rangle}
               {\langle S^{global},S^{global}\rangle},
 \qquad
 Z_{22}^{birth}=B_{22}^{(2)}-\eta_*S^{global},
\tag{65.30}
\]

rank two is globally exact if and only if (Z_{22}^{birth}=0), or if a
separate localized Z22 provider carries (Z_{22}^{birth}).  The current
single-alpha D11 gauge supplies neither fact generically.  Thus the earlier
`SOLVED EXACT` rank-two statement is a local quotient theorem, not an
unconditional width-1024 reset insertion.

There is an exact global repair requiring only four CP rays and the already
complete K31 Star4 plane.  For an arbitrary desired raw coordinate (a),

\[
\boxed{
\begin{aligned}
 \Delta G_4^{(4)}=\frac a4\{&
 (A+B+C)^{\otimes4}-(A+B-C)^{\otimes4}\
 &-(A-B+C)^{\otimes4}+(A-B-C)^{\otimes4}\}.
\end{aligned}}
\tag{65.31}
\]

Its polynomial is

\[
 12a\,A^2BC+4a\,(B^3C+BC^3).
\tag{65.32}
\]

Therefore it has exactly the target q211 coordinate (a), zero K22 leakage,
zero marginal leakage, and only the two pairwise K31 coordinates

\[
 (\Delta K31)_{BC}=(\Delta K31)_{CB}=a.
\tag{65.33}
\]

The reset amendment

\[
 \Delta H_{BC}=\Delta H_{CB}=-a
\tag{65.34}
\]

cancels those two faces.  No pairing update, rho division, localized K22
state, or six-ray completion is required.  This identity is global because
every ray is supported in the supplied three-frame and (65.32) enumerates
all of its nonzero monomials.

The word `global` here refers to a literal coordinate-label q211 source:
(A,B,C) are three feature axes, so (B^3C) and (BC^3) are exactly two rows of
the deployed coordinate-Star4 image.  For a dense oriented frame, the
binding condition is instead

\[
 B^{\odot3}\odot C+C^{\odot3}\odot B
 \in\operatorname{im}(\operatorname{Star4}).
\tag{65.35}
\]

It fails generically for a coordinate-star provider.  A dense-frame source
must either prove (65.35), decompose into literal coordinate triples, or use
the six-ray no-leak tensor.  Thus rank four is unconditional for each
coordinate q211 entry, not for an arbitrary rotated mode.

The four rays must be retained as one group.  Ten consecutive four-ray
births have

\[
 \sum_m b_m=4(1+\cdots+10)=220,
\]

so (65.21) gives exactly (1,383,895,040) dominant arithmetic at
(n=1024), about (0.6447) width-GEMM equivalents, before acquisition,
hooks, scalar work, and the literal meter.  Maximum persistent factor memory
is (164,000) bytes.  Fixed-cap deployment displaces four incumbent rays per
live birth and must use the net feature in (65.20).

The corrected deployment hierarchy is now:

1. rank two -- exact only on the restricted pair quotient, or globally after
   proving (65.29) / supplying localized Z22;
2. rank four -- exact global `CP4 + Star4` moving birth for literal
   coordinate triples (or after proving (65.35)), with no kappa;
3. rank six -- exact full tensor with no Star4 compensation.

Graph corrections:

- `VSTAR_RANK2_BALANCED_RESET_INSERTION` -- **conditional/restricted**, not
  a global D11 GO; terminal receipt is (65.29).
- `VSTAR_RANK2_GLOBAL_PAIRING_OBSTRUCTION` -- **solved no-go generically** by
  the rank/support mismatch in (65.28)--(65.30).
- `VSTAR_RANK4_STAR_GLOBAL_INSERTION` -- **solved exact algebra for coordinate
  q211 entries**; a dense frame additionally requires (65.35).  It remains
  conditional on the source oracle, grouped retention, reset fidelity, cost,
  and complete-old Schur value.

### 65.8 Persistence correction: only the no-leak six-ray block is an
unconditional multi-reset CP carrier

The rank-two and rank-four identities sew a CP block to a compensating
pairing or Star4 block at the birth seam.  Same-seam equality is not enough
for a moving carrier.  Let (R_{CP,l}), (R_{S,l}), and (R_l^{(4)}) denote the
actual next-reset maps applied to the CP factors, the compensator, and the
represented total fourth tensor.  Gauge sewing persists if and only if

\[
 \boxed{
 R_{CP,l}G_l+R_{S,l}S_l
 =R_l^{(4)}(G_l+S_l)}
\tag{65.36}
\]

at every crossed reset, with the compensating state itself retained.

For the rank-four coordinate birth, (S_l) contains
(-(B^3C+BC^3)).  One affine transition sends it to

\[
 -(WB)^{\otimes3}\odot(WC)
 -(WB)\odot(WC)^{\otimes3}.
\tag{65.37}
\]

For generic dense (W), (WB,WC) are dense and (65.37) is outside the image
of the next coordinate-Star4 provider.  The deployed CP state retains its
four factor rays, while the Star4/pair state is reconstructed from visible
planes rather than retaining this complete oriented tensor.  Therefore no
current algebraic receipt proves (65.36).  The analogous issue applies to
the rank-two pairing sew, in addition to its global K22 obstruction.

Consequently:

1. rank two is a local restricted-quotient decoder;
2. rank four is an exact same-site or one-transition coordinate source, and
   becomes a multi-reset carrier only after proving (65.36); but
3. the six-ray identity (61.4) is a pure CP tensor with zero compensating
   leakage, so ordinary factor transport carries it through every deployed
   CP reset without a gauge-sewing premise.

For ten consecutive six-ray births, the dominant rank arithmetic in
(65.21) is about (0.9671) width-GEMM equivalents.  Since this excludes the
source oracle, hooks, grouping, reset-error evaluation, and literal-meter
margin, the full ten-birth persistent route has no current one-equivalent
cost certificate.  It can reopen only with fewer active births, an already
paid factor-basis push/read reuse, or a proved invariant compensator satisfying
(65.36).  A last-transition rank-four birth remains arithmetically cheaper
because it is consumed before another reset.

Graph corrections:

- `VSTAR_RANK4_STAR_GLOBAL_INSERTION` -- **solved one-transition algebra / HOLD
  for multi-reset persistence** pending (65.36).
- `VSTAR_RANK6_PERSISTENT_CP_CARRIER` -- **solved exact transport algebra / cost
  HOLD**: about (0.9671) dominant equivalents for ten births before all extras.
- `VSTAR_COMPENSATOR_RESET_NATURALITY` -- **open blocking identity** (65.36).

### 65.9 Exact three-generator q211 carrier: the six CP atoms need not be
materialized

The six-ray Waring rank is the minimum number of *fourth-power atoms* needed
for a no-leak CP representation.  It is not the state rank of the structured
q211 tensor.  Store that tensor in its native form

\[
 \boxed{S_l=a_l\operatorname{Sym}(A_l,A_l,B_l,C_l),}
\tag{65.38}
\]

where the convention is (S_l(A_l,A_l,B_l,C_l)=a_l) in an orthonormal
coordinate frame, equivalently its diagonal polynomial is

\[
 p_l(x)=12a_l(A_l^Tx)^2(B_l^Tx)(C_l^Tx).
\tag{65.39}
\]

Only the three generator vectors ((A_l,B_l,C_l)) and one scalar amplitude
are persistent.  For component vectors (A,B,C) at any transported seam,
the exact visible planes are

\[
 \boxed{K4=12a\,A^{\odot2}\odot B\odot C,}
\tag{65.40}
\]

\[
\boxed{
 K31=3a\{(A^{\odot2}\odot B)C^T
          +(A^{\odot2}\odot C)B^T
          +2(A\odot B\odot C)A^T\},}
\tag{65.41}
\]

and

\[
\boxed{
\begin{aligned}
 K22=2a\{&A^{\odot2}(B\odot C)^T
 +(B\odot C)(A^{\odot2})^T\\
 &+2(A\odot B)(A\odot C)^T
 +2(A\odot C)(A\odot B)^T\}.
\end{aligned}}
\tag{65.42}
\]

Equations (65.40)--(65.42) are the polarizations of (65.39), so they
include every collision coefficient exactly.  They are also identical to
reading the virtual six-ray block in (61.4), but avoid materializing its six
factor columns.

The carrier is closed under every deployed affine/diagonal-gate factor map:

\[
 (A,B,C,a)
 \longmapsto
 (D_gW^TA,D_gW^TB,D_gW^TC,a).
\tag{65.43}
\]

Indeed, fourth-tensor naturality sends
(\operatorname{Sym}(A,A,B,C)) to the same symmetric word in the three
transported generators.  There is no Star4 or pairing compensator, so the
sewing obstruction (65.36), the global K22 obstruction (65.29), and
(\kappa=1/|1-\rho|) all disappear.  True-versus-deployed nonlinear reset
error remains governed by (65.19), exactly as for the ordinary CP4 gate law.

For one live group, the dense-width work is conservatively:

1. three transported columns in (65.43);
2. three outer-product terms in (65.41); and
3. four outer-product terms in (65.42).

Thus ten consecutive births, with
(1+\cdots+10=55) live group-layers, cost at most

\[
 \boxed{\frac{(3+3+4)55}{1024}
 =\frac{550}{1024}\simeq0.5371}
\tag{65.44}
\]

width-GEMM equivalents in this conservative rank-one-update census, before
source acquisition, hooks, scalar powers, and literal metering.  The maximum
persistent state is only

\[
 10\cdot3\cdot1024\cdot4=122,880\ \text{bytes}
\]

plus ten amplitudes and group metadata.  Shared generator spans can reduce
the three-column push further; (65.44) does not assume that reuse.

At a coordinate birth ((A,B,C)=(e_i,e_j,e_k)), all same-site K4/K31/K22
faces in (65.40)--(65.42) vanish, so the source enters without any visible
plane compensation.  After dense transport, the formulas expose precisely
the future K4/K31/K22 action from Section 61.3.  The provider should therefore
be inserted as its own structured fourth-order state and included once in the
ordinary Q24 plane totals before the residual Star4/pair reset.

For a dense frame, (65.38) generally has nonzero birth collisions.  It is an
exact carrier for the *full* structured tensor, but not automatically for its
strict off-diagonal q211 projection.  Applying the coordinate collision mask
destroys the three-generator form generically, and subtracting collisions only
at birth creates another persistence-sewing obligation.  Therefore a dense
costate-selected frame is a lawful forward birth only if its full structured
source is desired or if the Section 63 collision-sewing/naturality receipt is
proved.  Otherwise it remains an endpoint-scalar consumer.  Literal distinct
coordinate triples are collision-free and need no such premise.

This dedicated generator bank leaves the incumbent CP4 cap and its energy
ordering untouched; (65.44) is its complete incremental dense-rank charge.
Forcing it through the incumbent atom truncator instead materializes six
virtual CP rays per mode.  A fixed-cap implementation would then displace six
incumbent rays per live group and must grade the minus-incumbent term in
(65.20).  A generator group is indivisible under either policy.

This changes the deployment conclusion:

- `VSTAR_Q211_THREE_GENERATOR_CARRIER` -- **solved exact transport and plane
  algebra**, with conservative ten-birth dominant cost (0.5371) equivalents;
- `VSTAR_Q211_THREE_GENERATOR_SOURCE` -- **open/blocking**: the label-free
  amplitudes and coordinate triples/mode dictionary are still absent;
- `VSTAR_Q211_THREE_GENERATOR_ENDPOINT` -- **open**: complete-old Schur value,
  reset fidelity, source cost, hook cost, and literal meter must clear the
  remaining margin.

### 65.10 No universal one-mode source theorem

The seven-dimensional Vandermonde construction can be block-direct-summed
inside the r55 AQ core.  Seven disjoint copies fit because
(7\cdot7=49\le55).  Their tangents preserve all old pairwise faces
independently and change seven disjoint coordinate q211 entries.  Arbitrary
linear combinations therefore give a seven-dimensional lawful source space,
and endpoint costates may weight its seven coordinates independently.

Consequently neither Section 56 nor the three-generator carrier proves that
one coordinate triple per layer captures the physical residual.  For a frozen
(R)-mode dictionary, let (F_R) be its complete-old-residualized endpoint
design.  Its promotion criterion is the actual task-weighted captured energy
or Schur gain, not local tensor rank.  The direct-sum witness proves that no
architecture-independent theorem can set (R=1), even at latent rank 55.

- `VSTAR_Q211_MODE_SPARSITY` -- **open/blocking endpoint receipt**: certify a
  small frozen mode dictionary with the required complete-old captured value.
- `VSTAR_Q211_R1_UNIVERSALITY` -- **closed no-go** by the seven-block direct
  sum.

## 62. Sharp universal-state lower bound and the target-specific streamed
provider

The solved source theorem does not imply that a generic reusable q211 tensor
fits the card.  The exact common-support rank gives a sharp separation.

Let the target q211 costate have symmetric slices
(Lambda_1,\ldots,\Lambda_n\in\operatorname{Sym}_n), and concatenate them:

\[
 \mathsf L=[\,\Lambda_1\;\Lambda_2\;\cdots\;\Lambda_n\,]
 \in\mathbb R^{n\times n^2}.
\]

A shared-support representation

\[
 \Lambda_i=U\Theta_iU^T,qquad U\in\mathbb R^{n\times k},
\tag{62.1}
\]

exists exactly when one k-plane contains the range of every slice.  Therefore

\[
 \boxed{k_{\min}=\operatorname{rank}\mathsf L.}
\tag{62.2}
\]

For generic slices (k_{\min}=n).  Dense slice coordinates then require

\[
 n\,k(k+1)/2=\Theta(n^3)
\]

state.  Conversely, an (O(n^2)) dense common-support state forces
(k=O(\sqrt n)), so it cannot be universally exact.  A diagonal
(Theta_i) avoids cubic storage only when all symmetric slices are
simultaneously diagonalizable, equivalently when they commute pairwise;
generic costates do not.

The coefficient approximation lower bound is equally sharp.  For an
orthogonal rank-k projector (P), put

\[
 E(P)^2=\sum_i\|\Lambda_i-P\Lambda_iP\|_F^2.
\]

Then

\[
 \boxed{
 E(P)^2\ge\sum_{j>k}\sigma_j(\mathsf L)^2.}
\tag{62.3}
\]

Choosing P as the leading left-singular subspace gives the matching
factor-two upper bound

\[
 E(P)^2\le2\sum_{j>k}\sigma_j(\mathsf L)^2.
\tag{62.4}
\]

Thus a small k is lawful only with a measured joint-range tail, not from the
rank of any single slice.

### 62.1 Exact target-specific action from resident products

For one fixed endpoint/portfolio, suppose the *costate*, rather than the
universal source, has the frozen factorization (62.1).  Define

\[
 N_a=\sum_jU_{ja}M_j,qquad P_{ia}=M_iN_a.
\]

The exact AQ 211 action before collision subtraction is

\[
 \boxed{
 S_{211}^{AQ}
 =\sum_{i,a,b}\Theta_{i,ab}
 \{32\langle P_{ia},P_{ib}\rangle_F
   +16\langle P_{ia}^T,P_{ib}\rangle_F\}.}
\tag{62.5}
\]

Its noncommutative residual alone is

\[
 S_{\rm comm}
 =-32\sum_{i,a,b}\Theta_{i,ab}
 \langle\operatorname{skew}P_{ia},
        \operatorname{skew}P_{ib}\rangle_F.
\tag{62.6}
\]

If

\[
 \Theta_i=B_i\operatorname{diag}(\lambda_i)B_i^T
\]

has signed rank (s_i), (62.6) becomes

\[
 -32\sum_{i,t}\lambda_{it}
 \left\|\sum_a(B_i)_{at}\operatorname{skew}P_{ia}\right\|_F^2.
\tag{62.7}
\]

This streams one anchor at a time.  Its additional state is
(O(nks+nk+kr^2)), and its action is (O(nksr^2)); for
(r^2=O(n)) and (ks=O(n)), these are respectively (O(n^2)) and
(O(n^3)).  It is label-covariant under simultaneous permutation of
anchors/factors and invariant under latent orthogonal conjugacy because only
Frobenius products of conjugated matrices appear.

The decisive provenance condition is literal residency of the full
(P_{ia}) at the synchronized seam.  If it must be formed anew, its D10 cost
is

\[
 \frac{10k\,55^3}{1024^2}=1.5867k,
\tag{62.8}
\]

so even k1 violates the one-equivalent slack.  The streamed provider is a
conditional GO only as a view/consumer of an already-paid product bank.

### 62.2 Joint reset-and-action cap

Including the 34-node H211 reset evaluator from Section 49 and one diagonal
(Theta) norm pass gives the optimistic necessary count

\[
 C_{\rm joint,diag}
 \ge\frac{10\{(4984+3025)k+1870\}}{1048576}.
\tag{62.9}
\]

Thus k12 fits narrowly; k13 fits only if the 1,870-operation backbone is
already paid, before collision, hook, or recurrence cost.  With a dense
symmetric slice and triangular Gram evaluation, the corresponding bound is

\[
 C_{\rm joint,dense}
 \ge\frac{10\{89k^2+4895k+1870
       +3025k(k+1)/2\}}{1048576}.
\tag{62.10}
\]

It admits k6 only narrowly and rejects k7 before extras.  Forming a full
k-by-k Gram rather than its symmetric triangle gives the conservative k5
cap.  These are necessary screens, not deployment receipts.

A Walsh/Schur-closed support can close collisions as in Section 60, but a
proper support remains exact through depth only if every gate profile is
block-constant and every dense transition maps one partition algebra to the
next.  Generic gate profiles separate all n coordinates and force k=n.
Under (62.9), even the structured power-of-two choice is practically limited
to k8.

### 62.3 The raw-(10^{-8}) fidelity requirement

Let (m_0) be current raw MSE, (m_\star) the oracle MSE for an exact
provider, and let (	au) be the fraction of oracle squared endpoint action
captured after the complete-old Schur quotient.  The ideal linear projection
gives

\[
 m_{\rm new}=m_0-\tau(m_0-m_\star).
\]

Reaching a goal (m_g) therefore requires

\[
 \boxed{
 \tau\ge\tau_{\rm req}
 =\frac{m_0-m_g}{m_0-m_\star}.}
\tag{62.11}
\]

With (m_0=2.9409\times10^{-8}), (m_g=10^{-8}), and the current oracle
range (m_\star=7.09\)--(9.89\times10^{-9}), this is

\[
 \tau_{\rm req}=0.8696\text{--}0.9944,
 \qquad
 \sqrt{\tau_{\rm req}}=0.9325\text{--}0.9972.
\tag{62.12}
\]

The wide interval is important: if the applicable oracle is near its weaker
end, essentially perfect task-weighted fidelity is required.  Coefficient
energy, source norm, or same-site correlation cannot substitute for this
endpoint-Schur metric.

The recommended mechanism is therefore not a generic reusable tensor.  It
is a preregistered, fixed target-specific low-rank scalar (62.5)--(62.7),
used terminally, or converted into a carrier only with the applicable seam
proof.  The two-ray birth needs the full-plane pairing identity; the four-ray
coordinate + Star4 birth is same-site/one-transition unless its compensator
intertwines through every reset; the six-ray block is the exact persistent
fallback.  GO requires:

1. literal resident P provenance and exact collision reads;
2. common-support k/rank satisfying (62.9) or (62.10);
3. the Section 49 AQ/quadrature certificate;
4. positive complete-old Schur novelty, already witnessed structurally by
   Section 59 but still required on the target portfolio;
5. captured endpoint energy satisfying (62.11).

Absent any one receipt, the provider is HOLD/NO-GO rather than a raw-MSE
claim.

## 63. Final exact streamed-provider theorem: sew collisions at consumption,
not in the factor state

The shared-support factor must describe the unmasked pair costate

\[
 \bar\Lambda_i=U\Theta_iU^T.
\]

For an anchor i, define the all-distinct projector

\[
 \mathcal A_i(X)
 =(I-e_ie_i^T)
 \bigl(X-\operatorname{Diag}\operatorname{diag}X\bigr)
 (I-e_ie_i^T).
\tag{63.1}
\]

One must not materialize (mathcal A_i(\bar\Lambda_i)): the label mask
generically destroys the low-rank/common-support structure.  Instead, sew
the collisions into the scalar consumer.  For a symmetric H4 slice
(H_i(j,k)),

\[
\boxed{
\begin{aligned}
\langle\mathcal A_i(\bar\Lambda_i),H_i\rangle
={}&\langle\bar\Lambda_i,H_i\rangle
-2\sum_j\bar\Lambda_{i,ij}K31_{ij}\\
&-\sum_j\bar\Lambda_{i,jj}K22_{ij}
+2\bar\Lambda_{i,ii}K4_i.
\end{aligned}}
\tag{63.2}
\]

For the commutator component the K31 and all-equal terms vanish, but the
diagonal K22 subtraction does not.  Synchronized K22/K31/K4 reads, or a
charged exact recomputation, are mandatory.

### 63.1 Reset action and recurrence

At each AQ reset, reuse the already-paid eigensystem and the same 34
frequencies, characteristic values, resolvents, means, and covariances for
all pair atoms.  For a signed-rank-s slice

\[
 \Theta_i=B_i\operatorname{diag}(\lambda_{it})B_i^T,
\]

form the factor directions (widetilde V_{it}=VB_i[:,t]) and accumulate the
fused H211 reset functional from Section 49.  Every reset has its own
certificate

\[
 E_{\rm tail}^{(4)}(T)
 +\min(E_{\rm GL}^{211},E_{\rm Filon}^{211})
 \le\epsilon_{211,l},
\tag{63.3}
\]

and the endpoint quadrature error is

\[
 \epsilon_{\rm quad}\le\sum_l|w_l|\epsilon_{211,l}.
\tag{63.4}
\]

Node reuse does not make a 34-node rule a uniform theorem; (63.3) is an
instance certificate.

An exact common-support reverse recurrence exists if the pair maps obey

\[
 B_{\ell i}^TU_{\ell+1}=U_\ell R_{\ell i}.
\tag{63.5}
\]

Then

\[
 \boxed{
 \Theta_{\ell,p}
 =\sum_iA_{\ell,pi}
 R_{\ell i}\Theta_{\ell+1,i}R_{\ell i}^T.}
\tag{63.6}
\]

Dense symmetric slices are closed by (63.6).  Diagonal slices remain
diagonal for every coefficient family exactly when the (R_{\ell i}) are
monomial in one common basis.  Signed rank s is not generically closed under
the anchor sum: the concatenated transformed factors must retain rank at most
s at every cut.

The persistent-state classes are

\[
\begin{array}{c|c|c}
\Theta\text{ class}&\text{state}&O(n^2)\text{ condition}\\ \hline
\text{diagonal}&O(nk)&k\le n\\
\text{dense symmetric}&O(nk^2)&k^2=O(n)\\
\text{signed rank }s&O(nks)&ks=O(n).
\end{array}
\tag{63.7}
\]

Streaming each (P_{ia}) through both its symmetric and skew consumers avoids
the otherwise enormous (nkr^2) persistent bank.

### 63.2 Why Walsh closure is exact but rarely lawful

At (n=1024=2^{10}), normalized Walsh characters obey

\[
 u_a\odot u_b=n^{-1/2}u_{a\oplus b}.
\tag{63.8}
\]

A product-closed subgroup is therefore a unital Schur algebra, and in its
primitive coset-indicator basis multiplication by any gate profile in the
algebra is diagonal.  This makes diagonal (Theta) and collision weights
exactly closed.

Conversely, every unital Schur subalgebra of (mathbb R^n) is a
block-constant coordinate space.  If the joint gate profiles distinguish all
coordinates, the algebra they generate contains every coordinate indicator,
so any invariant unital support has

\[
 \boxed{k=n.}
\tag{63.9}
\]

This is sharp: m indistinguishable coordinate blocks give an m-dimensional
algebra.  Across depth, every dense W must also map one partition algebra
exactly to the next.  Walsh coordinates alone do not supply this receipt.

### 63.3 Robust endpoint-gain certificate

Let (widehat f) be the computed candidate, (f) the exact feature, (R_O)
the complete-old weighted Schur projector, and suppose

\[
 \|R_O(f-\widehat f)\|_W\le\epsilon.
\]

Write (widehat f_\perp=R_O\widehat f) and (y_\perp=R_Oy).  The exact
one-feature gain has the certified lower bound

\[
 \boxed{
 \Delta_{\rm MSE}(f)\ge
 \frac{
 \left(
 |\langle y_\perp,\widehat f_\perp\rangle_W|
 -\|y_\perp\|_W\epsilon
 \right)_+^2}
 {(\|\widehat f_\perp\|_W+\epsilon)^2}.}
\tag{63.10}
\]

Also (|\widehat f_\perp\|_W>\epsilon) certifies structural novelty.  The
error budget must include support truncation, recurrence drift, AQ model
error, collision error, signed-H3 contraction, and quadrature.  The old span
must jointly contain CP4, K4/K31/K22, Z22, existing H3, and every D11 feature.

The final disposition is:

- **conditional GO:** one preregistered target/portfolio; exact per-cut
  support recurrence; literal resident full P; scalar collision sewing;
  certified reset/H3 errors; total incremental cost at most one equivalent;
  and (63.10) clears the raw-gain gate;
- **HOLD:** actual small-k ranks, P ownership, collision reads, or endpoint
  value are not proved;
- **NO-GO:** a generic reusable q211/H3 carrier, newly forming P, masking and
  assuming rank survives, using only resident sym(P), or claiming Walsh
  closure without block-constant gates and compatible dense transitions.

## 64. Restricted rank-two amplitude: forward mode versus reverse scalar,
and the corrected global hierarchy

For one fixed forward triple ((A,B,C)), covariance Gram--Schmidt supplies
the normalized rays and

\[
 \rho=\frac{C_{BC}}{\sqrt{C_{BB}C_{CC}}},
\]

while existing K3/Star4 and the covariance-pairing plane supply every
pairwise leakage cancellation in Section 61.5.  The AQ commutator coordinate
is

\[
 r_{ABC}=-8\langle[A,B],[A,C]\rangle_F
\tag{64.1}
\]

in the commutator convention.  It is structurally new relative to literal
D11, but it is already part of the full-AQ Section 49 reset.  Against an r49
AQ replacement the lawful birth amplitude is instead

\[
 a_{ABC}^{new}
 =R_{old}\{\kappa_{ABC}^{true}-\kappa_{ABC}^{AQ}\},
\]

after every incumbent CP3/CP4/star/Q24 term is removed.  It equals the
displayed commutator scalar only when a separate same-seam ownership theorem
proves that the parallel word is old and the commutator is the entire
residual.  For a generic amplitude a, the restricted two-ray pre-pair map is

\[
 q'=\frac{a_{ABC}^{new}}{1-\rho}.
\tag{64.2}
\]

This looks ill-conditioned as (\rho\to1), but signed CP factors have an
orientation gauge.  Replace

\[
 C\mapsto C^\circ=\varepsilon C,qquad
 a^{new}\mapsto (a^{new})^\circ=\varepsilon a^{new},qquad
 \varepsilon=-\operatorname{sign}\rho.
\]

The physical product (a^{new}A^2BC) is unchanged, whereas

\[
 \rho^\circ=-|\rho|,
 \qquad
 \boxed{q'^\circ=\frac{\varepsilon a^{new}}{1+|\rho|}.}
\tag{64.3}
\]

Thus the restricted pair conversion has condition factor at most one.  If an
external orientation convention forbids the sign flip, (64.2) binds.  The
four-ray + Star4 construction in Section 65.7 avoids the division for a
same-site/one-transition coordinate birth, and the unconditional six-ray
construction avoids it for a persistent carrier.  At (|\rho|=1), B and C
are collinear and the triplet source collapses into the pairwise quotient;
the complete-old Schur projection should remove it rather than amplify it.

With amplitude error (delta a^{new}) and correlation error
(delta|\rho|), (64.3) gives the explicit first-order bound

\[
 |\delta q'|
 \le\frac{|\delta a^{new}|}{1+|\rho|}
 +\frac{|a^{new}|\,|\delta|\rho||}{(1+|\rho|)^2}
 +O(\delta^2).
\tag{64.4}
\]

This must enter the endpoint epsilon in (63.10).

There is a sharp information distinction.  The reverse weighted scalar

\[
 S_\Lambda
 =-8\sum_{i,a,b}\Theta_{i,ab}
 \langle[M_i,N_a],[M_i,N_b]\rangle
\tag{64.5}
\]

identifies one costate-dependent linear functional.  It does not reconstruct
the label-free forward amplitudes (r_{ABC}), nor the moving rays needed to
birth a generic CP4 state.  For one preregistered known mode with endpoint
leverage L, it is enough to set

\[
 q'=S_\Lambda/[L(1-\rho)]
\]

(or use the oriented denominator (1+|\rho|)) only when the full-plane pairing
receipt holds.  For R independent forward quotient modes, a lawful carrier
needs R independent amplitudes.  It needs 2R CP columns in the restricted
pairing quotient, 4R for global coordinate modes with Star4, or 6R for
rotated/no-Star modes, absent an additional sharing identity.

Accordingly:

- `RANK2_KNOWN_FORWARD_MODE_BIRTH`: local/full-plane-pairing conditional;
- `RANK4_COORDINATE_STAR_BIRTH`: exact same-site/one-transition; persistent
  only with a sewn CP+Star compensator recurrence;
- `RANK6_ROTATED_NO_STAR_BIRTH`: exact unconditional persistent fallback;
- `FIXED_REVERSE_SCALAR_TERMINAL_ACTION`: exact conditional GO;
- `REVERSE_SCALAR_TO_GENERIC_FORWARD_STATE`: information-theoretic NO-GO;
- `R_MODE_PORTFOLIO`: requires a proved mode dictionary, R amplitudes, and the
  applicable 2R/4R/6R column, displacement, cost, and Schur receipt.

## 67. No-double-count split: carried AQ projection, optional fused surface,
and covariance-only Gaussian rest

The native/CP discussion above compares to literal D11.  The proposed AQ
replacement has two distinct ownership levels which must not be conflated.
At reset \(l\), define

\[
 g_i^{AQ}=m_i+u_i^Tz+Q_{M_i}(z),\qquad
 Y_i^{AQ}=(g_i^{AQ})_+,\qquad
 X_i^{AQ}=P_{\le2}Y_i^{AQ},
\]

and let \(Y_i^{true}\) be the true activation at the synchronized seam.  The
literal carried state is \(X^{AQ}\), so it automatically owns only

\[
 T_{211}^{X}=\Pi_{211,off}\kappa_4(X^{AQ}). \tag{67.1a}
\]

An augmented branch may additionally own

\[
 T_{211}^{Y,AQ}=\Pi_{211,off}\kappa_4(Y^{AQ}) \tag{67.1b}
\]

by explicitly including and billing the complete Section 49 fused reset
surface at the same seam, frame, collision convention, and consumer.  This
ownership does not follow from storing \((m,u,M)\) or \(X^{AQ}\) alone.
Under the augmented premise, Section 49 contains the complete AQ parallel
and noncommutative pieces; at the pre-reset quadratic level they are

\[
 48\operatorname{tr}(M_i^2M_jM_k)
 -8\langle[M_i,M_j],[M_i,M_k]\rangle_F. \tag{67.2}
\]

The resident-\(P\) contraction and the fused surface are therefore
implementations of the augmented source, not additive corrections to it.
Adding a native/four-ray/six-ray carrier with that same AQ amplitude counts
q211 twice.  Without the fused surface, only (67.1a) is owned and
\(\kappa_4(Y^{AQ})-\kappa_4(X^{AQ})\) remains in the candidate defect.

The affine Gaussian rest owns a different object.  With

\[
 Y_i^G=(m_i+u_i^Tz)_+,qquad X_i^G=P_{\le2}Y_i^G,qquad
 r_i^G=Y_i^G-X_i^G,
\]

it supplies the two-point covariance
\(K_{ij}^{G,>2}=\langle r_i^G,r_j^G\rangle\).  It does **not** own the
connected tensor \(\kappa_4(Y^G)-\kappa_4(X^G)\).  Incumbent
CP3/CP4/star/Q24 affine connected pieces enter the complete-old projector
once; the covariance rest is neither added nor subtracted as a fourth
cumulant.

Define the declared owned comparator

\[
 C=\begin{cases}
 Y^{AQ},&\text{if the complete Section 49 q211 surface is included},\\
 X^{AQ},&\text{for the carried-state-only AQ branch}.
 \end{cases} \tag{67.2a}
\]

Then the sole lawful additive off-diagonal source is

\[
 \boxed{
 T_{211}^{rem}(C)=R_{old}\Pi_{211,off}
 \{\kappa_4(Y^{true})-\kappa_4(C)\}.} \tag{67.3}
\]

Here \(R_{old}\) uses the same raw-versus-`/24` convention as the consumer.
Writing \(Y_a^{true}=C_a+e_a\) in slots \(a=(i,i,j,k)\), multilinearity gives

\[
 \kappa_4(Y^{true})-\kappa_4(C)
 =\sum_{\varnothing\ne S\subseteq\{1,2,3,4\}}
 \kappa_4(Z_1^S,Z_2^S,Z_3^S,Z_4^S), \tag{67.4}
\]

where \(Z_s^S=e_s\) for \(s\in S\) and \(Z_s^S=C_s\) otherwise.  Every
term contains at least one true-minus-comparator slot.

For the augmented comparator \(C=Y^{AQ}\), exact law matching
\(Y^{true}=Y^{AQ}\) gives

\[
 \boxed{T_{211}^{rem}(Y^{AQ})=0.} \tag{67.5}
\]

There is no corresponding affine-vanishing theorem for the carried
projection \(C=X^{AQ}\).  Let \(a=(2\pi)^{-1/2}\), \(Y=Z_+\), and

\[
 X=P_{\le2}Y=a+\tfrac12Z+\tfrac a2(Z^2-1).
\]

Then

\[
 \kappa_4(Y)=\tfrac34-2a^2-6a^4,\qquad
 \kappa_4(X)=3a^2+3a^4,
\]

so \(\kappa_4(Y)-\kappa_4(X)=\tfrac34-5a^2-9a^4\ne0\).  By continuity,
nearby nondegenerate correlated copies also have nonzero off-diagonal `211`
response.  Covariance-only Gaussian rest cannot erase this connected defect.

### 67.1 When the defect becomes a resident-tape scalar

For a fixed endpoint/portfolio with literal reverse costate
\(\Lambda_{211,l}\), the only additional Bellman datum is

\[
 \boxed{s_l^{rem}(C)=
 \langle\Lambda_{211,l},T_{211,l}^{rem}(C)\rangle.} \tag{67.6}
\]

A lawful direct oracle for (67.6) is sufficient for a first-order frozen
endpoint and needs no forward CP state.  The unprojected AQ subtraction is
already a resident scalar only in the augmented branch that includes
Section 49; in the state-only branch the resident subtraction is
\(\kappa_4(X^{AQ})\).  The bank \(P_{ia}=M_iN_a\) depends only on AQ cores:
it cannot manufacture the true-minus-comparator signed action.  Calling its
AQ word a residual is duplicate credit in the augmented branch and
incomplete ownership in the state-only branch.

One scalar is insufficient for nonlinear persistence, adaptive endpoints,
or an independent portfolio; those uses require a forward mode dictionary
and native/no-leak carrier.  Graph split:

1. `AQ_CARRIED_PROJECTION_Q211`: owned as \(\kappa_4(X^{AQ})\);
2. `AQ_FUSED_SURFACE_Q211`: conditionally owned as \(\kappa_4(Y^{AQ})\)
   only when Section 49 is explicitly included;
3. `AFFINE_GAUSSIAN_REST`: covariance reference only, no connected q211;
4. `TRUE_MINUS_COMPARATOR_Q211_ACTION`: open signed scalar oracle for the
   comparator in (67.2a);
5. `TRUE_MINUS_COMPARATOR_Q211_FORWARD_CARRIER`: conditional on source,
   reset persistence, cap, and complete-old Schur value.

Before valuation, apply the robust Schur bound (63.10) to \(s^{rem}(C)\),
never to the total AQ q211 action.

## 66. Native three-generator carrier: exact source rank, closure, shared
compression, plane cost, and endpoint-value gates

Section 65.9 proves that one raw typed word

\[
 \Phi(A,B,C;a)=a\,\operatorname{Sym}(A,A,B,C)
\tag{66.1}
\]

can replace the six fourth-power atoms used by a Waring implementation.
That is a representation theorem for one supplied word.  It does not prove
that the actual complete-old residual source is one word, that the nonlinear
reset preserves the word, or that its endpoint value is material.  This
section separates those receipts.

### 66.1 The lawful amplitude is true minus the declared comparator, not the
resident-P term

At reset (l), define the only lawful new source by Section 67:

\[
 S_l^{rem}(C)=R_{old,l}\Pi_{211,off}
 \{\kappa_4(Y_l^{true})-\kappa_4(C_l)\},
 \qquad C_l\text{ as in (67.2a)}.
\tag{66.2}
\]

For a declared frame ((A,B,C)), the native amplitude is the corresponding
coordinate of (66.2), with the same raw-versus-/24 normalization as the
consumer:

\[
 a_{new}(A,B,C)
 =\langle e_{AABC}^{\#},S_l^{rem}(C)\rangle.
\tag{66.3}
\]

It is not the pre-reset noncommutative word

\[
 -8\langle[A,B],[A,C]\rangle_F.
\tag{66.4}
\]

Equation (66.4), and its common-factor contraction from resident full
products (P), are part of the augmented Section 49 replacement.  Reusing it
as (a_{new}) would count q211 twice there; in the carried-state-only branch it
still does not equal the missing \(Y^{true}-X^{AQ}\) action.  A nonzero
carrier therefore needs a true-minus-comparator fourth-coordinate oracle, a
higher-chaos/rest state, or an independently proved residual identity.  The
banked AQ products alone cannot supply (66.3).

### 66.2 Exact native atom rank for a fixed doubled generator

Fix a nonzero anchor (A).  Any source with that doubled anchor can be written

\[
 S_A(H)=\operatorname{Sym}(A,A,H),\qquad H=H^T,
\tag{66.5}
\]

where (S_A(B\odot C)=\operatorname{Sym}(A,A,B,C)).  Let
((n_+(H),n_-(H))) be the positive and negative inertia of (H), ignoring zero
eigenvalues.  Then the exact minimum number of real native atoms is

\[
 \boxed{r_{nat,A}(H)=\max\{n_+(H),n_-(H)\}.}
\tag{66.6}
\]

Indeed, (B\odot C) has at most one positive and at most one negative
eigenvalue.  Inertia subadditivity therefore gives the lower bound in
(66.6).  Conversely, pair a positive eigenpair ((p,u)) with a negative
eigenpair ((-q,v)) by

\[
 B=\sqrt p\,u+\sqrt q\,v,qquad
 C=\sqrt p\,u-\sqrt q\,v,
\tag{66.7}
\]

so that (B\odot C=p uu^T-q vv^T).  Unpaired eigenvalues use (B=C)
or (B=-C).  This attains the bound.

Thus a rank-(r) singleton coefficient matrix does not automatically require
(r) atoms: opposite signs can share an atom, but two same-sign eigenvalues
cannot.  For a fixed-anchor source, the exact best rank-(R) Frobenius tail is

\[
 E_{A,R}^2
 =\sum_{j>R}p_j^2+\sum_{j>R}q_j^2,
\tag{66.8}
\]

where ((p_j)) and ((q_j)) are the decreasing positive and negative
eigenvalue magnitudes.

For a literal strict coordinate-q211 source, this theorem gives an exact
block census.  For every doubled label (i), form the symmetric singleton
matrix on the complementary labels

\[
 (H_i)_{jk}=(S_l^{rem})_{iijk},\qquad j,k\ne i,\quad j\ne k,
\tag{66.8a}
\]

with the diagonal q22 entries removed by the complete-old projector.  Every
q211 coordinate belongs to exactly one doubled-label block.  If native atoms
are required to use literal coordinate anchors (A=e_i), the exact minimum is

\[
 \boxed{R_{coord}(S_l^{rem})=
 \sum_i\max\{n_+(H_i),n_-(H_i)\}.}
\tag{66.8b}
\]

This is the exact raw rank for a coordinate-anchored representation.  The
eigenvectors used in (66.7) are generally dense singleton directions, so an
individual atom can have q22 collisions even though the sum over the complete
(H_i) decomposition has zero diagonal and reproduces the collision-free
source.  The complete spectral block must therefore be grouped atomically.
If the old q22 space permits an independently sewn diagonal (D_i) in this
block, the exact coordinate-anchored quotient rank is instead

\[
 R_{coord}^{old}([S])=
 \sum_i\min_{D_i:\ S_{e_i}(D_i)\in\mathcal O_l}
 \max\{n_+(H_i+D_i),n_-(H_i+D_i)\}.
\tag{66.8bq}
\]

A dense anchor can mix several (i)-blocks and sometimes lower the algebraic
count further, but it simultaneously creates dense collision and
other-partition faces; that saving is deployable only with complete
collision-sewing and reset-naturality receipts.  Equation (66.8b) is the
exact safe raw-group rank before those additional quotient optimizations.

Even an arbitrary diagonal q22 completion does not make a generic block
low-rank.  Let (m) be the singleton-block size and let (d_i\le m) be the
dimension of the lawful old diagonal-completion space.  The variety of
symmetric matrices with corank at least (k) has codimension
(k(k+1)/2).  Therefore, for generic fixed off-diagonal data, a lawful
completion cannot have corank (k) when

\[
 {k(k+1)\over2}>d_i.
\tag{66.8bg}
\]

Writing

\[
 k_i^*=\max\{k:k(k+1)/2\le d_i\},
\]

the generic sewn block still obeys

\[
 \operatorname{rank}(H_i+D_i)\ge m-k_i^*,
 \qquad
 R_{coord,i}^{old}\ge
 \left\lceil{m-k_i^*\over2}\right\rceil.
\tag{66.8bh}
\]

This bound already grants the most favorable sign balance and all (d_i)
completion freedoms.  Hence a cheap native lane cannot rely on q22 sewing to
make a generic source sparse; it needs special task-visible spectral/atomic
structure.

With a total coordinate-native budget (R), an exact achievable raw
Frobenius source tail is the discrete allocation

\[
 E_{coord,R}^2=
 \min_{\substack{R_i\ge0\\\sum_iR_i\le R}}
 \sum_i\left\{
 \sum_{j>R_i}p_{ij}^2+
 \sum_{j>R_i}q_{ij}^2\right\},
\tag{66.8c}
\]

where ((p_{ij})) and ((q_{ij})) are the signed spectral magnitudes of
(H_i).  The complete-old quotient tail additionally minimizes this expression
over the lawful diagonals (D_i), as in (66.8bq).  Thus (66.8c) is a safe raw
upper bound and the diagonal-minimized version is the exact sewn receipt;
local amplitude size alone is neither.

### 66.3 Complete-old sewing and generic source-rank lower bounds

Let (\mathcal O_l) be the complete CP3/CP4/Star4/pair/Q22/affine old space at
the literal seam.  The relevant algebraic rank is the quotient rank

\[
 r_{nat}^{old}([S])
 =\min_{O\in\mathcal O_l}
 \min\left\{R:S+O=\sum_{r=1}^R
 \Phi(A_r,B_r,C_r;a_r)\right\}.
\tag{66.9}
\]

This is source rank *after* exact old sewing.  Applying a strict collision
mask directly to one atom can have high native rank; (66.9) permits using a
raw representative only when its difference from the desired source is
literally old-owned.  Multi-reset deployment additionally requires that the
chosen old representative remain sewn under the reset maps.  Quotient rank
alone does not prove that persistence property.

There are three useful exact lower bounds.  For the (1|3) and (2|2)
flattenings define

\[
 r_{13}^{old}([S])=\min_{O\in\mathcal O_l}
 \operatorname{rank}\operatorname{Flat}_{1|3}(S+O),
\quad
 r_{22}^{old}([S])=\min_{O\in\mathcal O_l}
 \operatorname{rank}\operatorname{Flat}_{2|2}(S+O).
\tag{66.10}
\]

One native atom has (1|3) rank at most (3) and (2|2) rank at most (4), since
the exposed pair vectors lie in

\[
 \operatorname{span}\{A\odot A,A\odot B,A\odot C,B\odot C\}.
\]

Hence

\[
 \boxed{r_{nat}^{old}([S])\ge
 \max\left\{
 \left\lceil r_{13}^{old}/3\right\rceil,
 \left\lceil r_{22}^{old}/4\right\rceil
 \right\}.}
\tag{66.11}
\]

The cone of single native atoms has dimension at most (3n-2): the amplitude
and three vectors have three continuous scaling gauges.  If the complete-old
q211 quotient has dimension (D_Q), a generic quotient element therefore
needs

\[
 \boxed{R_{generic}\ge\left\lceil D_Q/(3n-2)\right\rceil,}
\tag{66.12}
\]

apart from possible secant defects which can only increase the generic rank.
Before old projection, the strict coordinate q211 space already has
(n\binom{n-1}{2}) coordinates.  The seven-block witness in Section 65.10 is
a concrete low-dimensional warning; (66.12) is the generic warning.  A small
carrier is lawful only after a task-visible low-rank or endpoint-tail
certificate for (S_l^{rem}).

For a fixed anchor, (66.6)--(66.8) can be placed inside the minimization over
(O).  For several anchors, cross-anchor sharing is possible, so the sum of
the fixed-anchor ranks is an upper bound, not a universal equality; the
flattening bounds remain valid.

#### 66.3a Source rank versus costate-visible rank

For a pulled-back endpoint costate (G), define its fixed-anchor singleton
matrix by

\[
 (\Gamma_A)_{jk}
 =\left\langle G,
 \operatorname{Sym}(A,A,e_j,e_k)\right\rangle.
\tag{66.12a}
\]

Then the source/costate action is exactly

\[
 \langle G,S_A(H)\rangle=\langle\Gamma_A,H\rangle_F.
\tag{66.12b}
\]

If the costate restriction itself must be represented by native atoms, its
minimum fixed-anchor atom count is

\[
 \max\{n_+(\Gamma_A),n_-(\Gamma_A)\},
\tag{66.12c}
\]

by the same inertia theorem.  For a literal coordinate costate, form
((\Gamma_i)_{jk}=G_{iijk}) on the strict singleton block.  Its exact raw
coordinate-anchor atom count is

\[
 R_{coord}(G)=\sum_i\max\{n_+(\Gamma_i),n_-(\Gamma_i)\},
\tag{66.12cq}
\]

with the same complete-old diagonal minimization as (66.8bq).  But
reproducing endpoint actions is a weaker task than representing either
tensor.  For a portfolio
((\Gamma_{A,1},\ldots,\Gamma_{A,P})), the exact minimum atom count that
reproduces all portfolio actions of (H) is

\[
\boxed{
 r_{vis}(H;\Gamma_{1:P})=
 \min_{\Delta:\ \langle\Gamma_p,\Delta\rangle=0\ \forall p}
 \max\{n_+(H+\Delta),n_-(H+\Delta)\}.}
\tag{66.12d}
\]

For one nonzero costate and one nonzero desired scalar, (r_{vis}\le1): choose
an eigenvector (v) of (\Gamma_A) with nonzero eigenvalue and scale
((vv^T)) to match the scalar.  This is exactly why a known tangent can be
served by one resident scalar or one tuned atom even when the true source has
large native rank.  The construction is target-specific; it does not recover
(H), serve a new independent costate, or certify panel transfer.  A reusable
forward claim must use (r_{nat}^{old}), whereas a frozen endpoint/portfolio
claim may lawfully use (r_{vis}) if the costates were preregistered.

### 66.4 Exact affine closure and the unresolved nonlinear-reset premise

For every linear factor map (L), tensor naturality gives

\[
 L^{\otimes4}\Phi(A,B,C;a)=\Phi(LA,LB,LC;a).
\tag{66.13}
\]

Thus affine propagation is exact.  A common coordinatewise gate is also
exact *for the deployed factor law* by taking (L=D_gW^T); an additive
deterministic bias does not change a fourth cumulant.  Three transported
generators are sufficient, independent of the six-ray Waring rank.

This does not establish closure under the full nonlinear moment reset.  Let
(\mathfrak R_l) be the literal Frechet reset map on fourth perturbations and
(Q_l) the complete-old quotient.  The one-atom closure defect is

\[
 E_l(A,B,C)=Q_{l+1}\mathfrak R_lQ_l\Phi(A,B,C;1)
 -Q_{l+1}\Phi(L_lA,L_lB,L_lC;1).
\tag{66.14}
\]

Exact persistent closure is equivalent to (E_l=0) on every live generator
triple, or more generally to (E_l\in\mathcal O_{l+1}) before quotient.  The
plane-read identities below do not by themselves prove (66.14).  If a reset
defect has native rank (r_{E,l}), exact state rank grows by that many atoms;
if it is truncated, its endpoint action belongs in the error budget.  A
proof that

\[
 \mathfrak R_l\Phi=D_l^{\otimes4}\Phi+O_l(\Phi),
 \qquad O_l(\Phi)\in\mathcal O_{l+1},
\tag{66.15}
\]

would close the premise.  Until the literal reset supplies (66.15), the
native carrier is exact for affine/common-gate factor push, not certified for
the true nonlinear reset.

#### 66.4a Joint-gate kernel criterion and generic reset obstruction

The obstruction can be stated more sharply.  On a strict q211 seam, suppose
the fourth-order reset tangent, after old collision terms are removed, is a
Schur multiplier by a symmetric joint-gate kernel (\mathcal K_l):

\[
 Q_{l+1}\mathfrak R_l(T)
 =Q_{l+1}(\mathcal K_l\odot T).
\tag{66.15a}
\]

If the relevant restriction of the kernel has a symmetric separable
decomposition

\[
 \mathcal K_l=\sum_{q=1}^{s_l}d_{lq}^{\otimes4},
\tag{66.15b}
\]

then one native atom maps to at most (s_l) native atoms:

\[
 \mathcal K_l\odot\Phi(A,B,C;a)
 =\sum_{q=1}^{s_l}\Phi(d_{lq}\odot A,
 d_{lq}\odot B,d_{lq}\odot C;a).
\tag{66.15c}
\]

The common diagonal-gate rule is exactly the special case (s_l=1).  A
general slot-dependent CP decomposition of (\mathcal K_l) does not preserve
the repeated (A,A) type and can require still more atoms.  Thus the relevant
reset complexity is the symmetric separable rank, or a costate-visible tail,
of the *joint* gate kernel restricted to the live source; it is not the three
generator count at birth.

If (R_l^{birth}) new atoms enter at each reset and no exact cancellations are
proved, the resulting atom-count upper bound is

\[
 R_m^{live}\le
 \sum_{l\le m}R_l^{birth}\prod_{j=l}^{m-1}s_j.
\tag{66.15d}
\]

For (s_j=1) this is the triangular live-incidence census used in (66.25).
Any persistent (s_j>1) changes the arithmetic qualitatively and must replace
the advertised (H=55) count; silently truncating back to one atom moves the
discarded branches into (E_{reset}).

For correlated Gaussian preactivations, joint ReLU gates are not generically
separable.  Already

\[
 \mathbb E[1_{\{G_i>0\}}1_{\{G_j>0\}}]
 ={1\over4}+{\arcsin\rho_{ij}\over2\pi}
\tag{66.15e}
\]

differs from the product of marginal gate probabilities whenever
(\rho_{ij}\ne0).  The higher derivative expectations entering the connected
four-point reset inherit analogous joint dependence.  Therefore no
architecture-independent theorem identifies the true reset with one common
(D_g^{\otimes4}) factor.  The deployed factor law may still be a useful
approximation, but exact promotion requires either the literal (s_l=1)
receipt, a small-(s_l) kernel factorization, or charging its endpoint-visible
tail through (66.29).

### 66.5 Shared-generator compression

Suppose (R) live atoms have all generators in a common (d)-dimensional span:

\[
 A_r=G\alpha_r,qquad B_r=G\beta_r,qquad C_r=G\gamma_r,
 \qquad G\in\mathbb R^{n\times d}.
\tag{66.16}
\]

Transport (G) once and retain the core coefficients.  The exact persistent
state is

\[
 nd+R(3d+1)
\tag{66.17}
\]

scalars, versus (3Rn+R) without sharing, and the dense push costs (d/n)
width-GEMM equivalents per seam.  The union-span rank (d), not the atom count,
is the exact push rank.  Common reverse factors reduce this cost only when
they are the same forward generators at the same seam; a target-specific
reverse factorization is not automatically reusable forward state.

For the fixed-anchor matrix source (S_A(H)), the minimum generator span is
itself exact:

\[
 \boxed{d_A=\dim\bigl(\operatorname{span}\{A\}
 +\operatorname{range}H\bigr).}
\tag{66.17a}
\]

Every decomposition must span (\operatorname{range}H), because
(H=\sum_r a_rB_r\odot C_r); the inertia construction (66.7) uses only that
range and attains the bound.  For a coordinate anchor and singleton block,
(d_A=1+\operatorname{rank}H).  Thus opposite-sign pairing can halve the atom
count without halving the push span.  With several blocks, the exact shared
push dimension is the rank of the union of all anchors and completed-block
ranges.

Linear generator sharing does not automatically compress collision reads,
because Hadamard products increase feature rank.  Let (V_p(G)) contain all
degree-(p) columnwise Hadamard monomials of (G), with

\[
 h_2\le\binom{d+1}{2},\qquad h_3\le\binom{d+2}{3}.
\tag{66.18}
\]

There are exact coefficient matrices satisfying

\[
 K31=V_3(G)M_{31}G^T,qquad
 K22=V_2(G)M_{22}V_2(G)^T,
\tag{66.19}
\]

with

\[
 \operatorname{rank}M_{31}\le3R,qquad
 \operatorname{rank}M_{22}\le4R.
\tag{66.20}
\]

Factoring these coefficient matrices reduces the outer-product read counts
to their actual ranks.  It is a real saving only when the Hadamard bases or
their required combinations are resident; forming and storing a large
(V_2,V_3) is part of the cost.  Without that receipt, the honest generic
read cost remains (3R) plus (4R), even when (d\ll3R).

#### 66.5a Exact anchor-matrix carrier

For a fixed doubled generator, an atom list is not the minimal moving state.
Store the whole singleton matrix in signed low-rank form

\[
 H=FJF^T,\qquad r=\operatorname{rank}H,
\tag{66.20a}
\]

where (J) is a nonsingular signed (r\times r) core.  The typed state is
(S_A(H)=\operatorname{Sym}(A,A,H)).  Under a linear/common-gate factor map,

\[
 (A,F,J)\longmapsto(LA,LF,J),
\tag{66.20b}
\]

so it transports only (r+1) columns and preserves all cancellations among
the inertia atoms exactly.

Let (h=\operatorname{diag}H) and (D_A=\operatorname{diag}A).  Summing the
atom formulas gives the exact direct reads

\[
 K4=12A^{\odot2}\odot h,
\tag{66.20c}
\]

\[
 K31=6D_A^2H+6(A\odot h)A^T,
\tag{66.20d}
\]

and

\[
 K22=2\{A^{\odot2}h^T+h(A^{\odot2})^T\}
 +8D_AHD_A.
\tag{66.20e}
\]

With (H=FJF^T), these require respectively (0), at most (r+1), and at most
(r+2) outer-product equivalents.  Including transport, the conservative
per-seam count is

\[
 \boxed{C_{anchor\text{-}matrix}=3r+4}
\tag{66.20f}
\]

rank-one equivalents, versus (10R) for an unshared list of the minimum
(R=\max(n_+,n_-)) native atoms.  Since (R\le r\le2R), the matrix carrier is
never asymptotically worse and can nearly halve the atom-list read charge.
Its state is (n(r+1)+r^2) scalars for a dense core, or (n(r+1)+r) after a
signed spectral gauge, versus (3Rn+R) for the atom list.

For coordinate blocks, use one ((e_i,F_i,J_i)) group per retained (H_i).
The exact multi-block push rank is the union rank of all anchors and factor
columns, while the reads sum by block unless their physical plane factors
are also shared.  Truncating (F_i,J_i) changes (H_i), so q22 diagonal sewing
and the endpoint tail must be recomputed; individual eigenmodes may not be
dropped outside the grouped old-projection rule.

More explicitly, for blocks ((H_1,\ldots,H_p)) the minimum common singleton
generator dimension is

\[
 d_H=\operatorname{rank}[\,H_1\ H_2\ \cdots\ H_p\,].
\tag{66.20g}
\]

There is a basis (F\in\mathbb R^{n\times d_H}) and symmetric cores (K_i)
with (H_i=FK_iF^T), and no smaller (F) can contain every block range.  If the
anchors have union rank (d_A), the exact factor-push rank is

\[
 d_{push}=\dim\left(\operatorname{span}\{A_1,\ldots,A_p\}
 +\operatorname{range}F\right)\le d_A+d_H.
\tag{66.20h}
\]

This Tucker-style block carrier is the strongest lawful shared-generator
compression.  It is useful only when (d_{push}) and the core/read ranks are
small; a large atom count does not by itself imply a large push, and a small
push does not by itself make the K31/K22 reads cheap.

The aggregate block reads have exact rank gates as well.  Put

\[
 Z_{31}=\sum_iD_{A_i}^2FK_i,qquad
 U=[A_i\odot h_i]_i,qquad A_*=[A_i]_i.
\]

Then the first K31 family is (6Z_{31}F^T), of rank at most (d_H), and the
second is (6UA_*^T).  Hence

\[
 r_{31}^{block}\le d_H+\operatorname{rank}(UA_*^T).
\tag{66.20i}
\]

For K22 define

\[
 Z_{22}=[D_{A_1}F\ \cdots\ D_{A_p}F],
 \qquad K_*=\operatorname{blockdiag}(K_1,\ldots,K_p).
\]

The conjugation family is exactly (8Z_{22}K_*Z_{22}^T).  With
((X=[A_i^{\odot2}]_i)) and ((Y=[h_i]_i)), the other family is
(2(XY^T+YX^T)).  Therefore

\[
 r_{22}^{block}\le
 \operatorname{rank}Z_{22}+2\operatorname{rank}(XY^T).
\tag{66.20j}
\]

These are literal physical-plane ranks, not atom counts.  In particular the
common singleton basis always compresses the first K31 family, whereas the
Hadamard-conjugated span (Z_{22}) can still grow to (p d_H); K22 is the
binding shared-read gate unless that Khatri--Rao span is small or resident.

### 66.6 Exact plane reads and comparison with six-ray CP and resident P

Under the Section 65 normalization, one native atom has

\[
 K4=12a\,A^{\odot2}\odot B\odot C,
\tag{66.21}
\]

\[
 K31=3a\{(A^{\odot2}\odot B)C^T
 +(A^{\odot2}\odot C)B^T
 +2(A\odot B\odot C)A^T\},
\tag{66.22}
\]

and

\[
\begin{aligned}
 K22=2a\{&A^{\odot2}(B\odot C)^T
 +(B\odot C)(A^{\odot2})^T\\
 &+2(A\odot B)(A\odot C)^T
 +2(A\odot C)(A\odot B)^T\}.
\end{aligned}
\tag{66.23}
\]

Thus the full K4/K31/K22 read requires respectively (0), (3), and (4)
outer-product equivalents, plus elementwise powers.  Transpose/view fusion
may lower literal writes, but (0/3/4) is the conservative no-fusion bill.

For linearly independent ((A,B,C)), the symmetric Waring rank of
(A^2BC) is exactly six.  An explicit identity is

\[
\begin{aligned}
48x^2yz={}&(x+y+z)^4-(x+y-z)^4-(x-y+z)^4+(x-y-z)^4\\
&-2\{(y+z)^4-(y-z)^4\}.
\end{aligned}
\tag{66.24}
\]

Consequently a materialized six-ray CP provider pays six pushes, six K31
reads, and six K22 reads per live atom-seam, while the native provider pays
(3+3+4).  If (H=\sum_mR_m) is the number of live atom-seams, the generic
dominant censuses are

\[
 C_{native}=10H/n,qquad C_{6CP}=18H/n.
\tag{66.25}
\]

For ten consecutive single-atom births, (H=1+\cdots+10=55), recovering the
conservative native count (550/1024\simeq0.5371) versus approximately
(0.967) for materialized six-ray CP.  Peak factor state is (3R_{max}n)
versus (6R_{max}n) scalars.  A virtual six-ray implementation can transport
the three base generators and reconstruct its six combinations, but a
generic CP reader still pays six K31 and six K22 terms; without native
algebra its count is ((3+6+6)H/n).

With shared generators and proved read ranks, the exact replacement for the
first formula in (66.25) is

\[
 C_{native}^{shared}
 ={1\over n}\sum_m\{d_m+r_{31,m}+r_{22,m}\}
 +C_{basis}+C_{hooks}.
\tag{66.26}
\]

For the anchor-matrix carrier with one equal-rank (r) block born at each of
ten consecutive resets, the corresponding conservative count is

\[
 C_{anchor\text{-}matrix,10}
 ={55(3r+4)\over1024}.
\tag{66.26a}
\]

Thus the pre-hook, pre-source one-equivalent gate is (r\le4).  A rank-two
block reproduces the (0.5371) one-atom count; ranks three and four still fit
arithmetically, while rank five does not.  This is a real reopening relative
to an unshared atom list, but source acquisition, literal hooks, reset error,
and billing margin make (r=4) a tight upper edge rather than a fire claim.
Unequal ranks use (\sum_{m,l\le m}(3r_l+4)/n), and the shared block-rank
bounds (66.20i)--(66.20j) replace it when proved.

The resident-P scalar path is cheaper still for one frozen endpoint: when
the required products are already present, it adds contractions but no new
matrix product or moving generators.  It has two decisive limitations.
First, the current resident-P formula computes the already-owned AQ action
(66.4), not the new residual (66.3).  Second, even a lawful resident oracle
for (\langle\Lambda,S^{rem}\rangle) supplies one endpoint action, not a
forward state that survives a different costate/reset.  If that oracle is
evaluated label-free on every corpus network and accumulated through the
fixed endpoint, its outputs *are* a deployable scalar endpoint feature and
their variance can be graded directly; this is the preferred board path when
it exists.  One symbolic contraction or one-network action does not by itself
provide that corpus variance/value receipt, and the scalar remains no
substitute for a reusable multi-endpoint carrier.

### 66.7 Endpoint-visible source tails and robust raw-MSE value

Let (\widehat S_l) be the retained native-atom approximation and let
(g_l=\mathcal P_{e:l}^*Q_e) be the literal endpoint costate pulled back to
the birth seam.  If

\[
 \epsilon_{l,R}=\|S_l^{rem}-\widehat S_l\|,
\]

then the endpoint action lost to source truncation obeys

\[
 |E_{source}|
 \le\sum_l\|g_l\|_*\epsilon_{l,R}.
\tag{66.27}
\]

For a fixed-anchor eigen-expansion, (66.8) gives the exact unweighted source
tail.  A tighter task-visible receipt contracts every omitted atom with
(g_l) directly.  If (\rho_j) bounds the quotient reset/transport operators,
an omitted depth tail from a birth (z_l) obeys

\[
 |E_{tail,l}|
 \le\|z_l\|\sum_{m>l}\|g_m\|
 \prod_{j=l}^{m-1}\rho_j.
\tag{66.28}
\]

Reset-model defects from (66.14) add

\[
 |E_{reset}|\le\sum_m\|g_{m+1}\|_*\|E_m\|.
\tag{66.29}
\]

These are action bounds, not MSE gains.  Across the frozen corpus, let
(\widehat f_\perp) be the complete-old residualized feature built by the
retained carrier, let (f_\perp) be the exact feature, and certify

\[
 \|f_\perp-\widehat f_\perp\|_W\le\epsilon_f.
\]

Writing

\[
 \widehat b=\langle y_\perp,\widehat f_\perp\rangle_W,
 \qquad \widehat v=\|\widehat f_\perp\|_W^2,
 \qquad Y=\|y_\perp\|_W,
\]

the exact one-feature optimal Schur gain has the robust lower bound

\[
 \boxed{
 \Delta_{MSE}^{true}\ge
 {\bigl(|\widehat b|-Y\epsilon_f\bigr)_+^2
  \over(\sqrt{\widehat v}+\epsilon_f)^2}.}
\tag{66.30}
\]

For a frozen coefficient (t), an equivalent direct gate is

\[
\begin{aligned}
 \Delta_{MSE}^{true}(t)\ge{}&
 2t\widehat b-t^2\widehat v\\
 &-2|t|\,\|y_\perp-t\widehat f_\perp\|_W\epsilon_f
 -t^2\epsilon_f^2.
\end{aligned}
\tag{66.31}
\]

The error (\epsilon_f) must include source atomic tail, reset defects, cap
displacement, plane-read approximation, and amplitude acquisition error.
Promotion requires the lower bound, not the central estimate, to clear the
desired raw-MSE gain and the adjusted-cost fire bar.  In the current campaign,
reaching raw (10^{-8}) from (2.9409\times10^{-8}) requires at least
(1.9409\times10^{-8}) complete-old joint gain.  A local nonzero amplitude,
large reverse scalar, or low arithmetic bill is not a substitute for
(66.30).

### 66.8 Decision split

1. `Q211_NATIVE_ATOM_ALGEBRA` -- **solved**: one supplied full word uses
   three generators and the exact (0/3/4) plane reads.
2. `Q211_NATIVE_SOURCE_RANK` -- **open/blocking**: certify
   (r_{nat}^{old}(S^{rem})) or an endpoint-visible atomic tail; no universal
   one-mode theorem exists.
3. `Q211_NATIVE_NO_DOUBLE_COUNT` -- **solved**: the amplitude is
   true-minus-full-AQ after complete-old projection; resident-P AQ credit is
   excluded.
4. `Q211_NATIVE_AFFINE_PUSH` -- **solved exact** by (66.13).
5. `Q211_NATIVE_NONLINEAR_RESET` -- **open/blocking**: prove (66.15) or charge
   (66.29); plane reads alone do not prove closure.
6. `Q211_NATIVE_SHARED_GENERATORS` -- **conditional GO** when the forward
   union span and Hadamard read ranks are literal same-seam state.
7. `Q211_NATIVE_READ_COST` -- **solved conservative algebra** at
   (10H/n), but source, basis, hooks, and literal billing remain outside it.
8. `Q211_NATIVE_ENDPOINT_VALUE` -- **open/blocking**: the robust complete-old
   gain (66.30) must clear the raw target and fire bar.
9. `Q211_NATIVE_COMMON_GATE_UNIVERSALITY` -- **closed no-go**: correlated
   joint-gate kernels are not universally rank one; reopen persistence only
   with the literal restricted kernel rank/tail receipt in (66.15b).

## 68. Exact signed-surface oracle for the true-minus-comparator q211 defect,
and a sharp pairwise-rest no-go

Section 67 identifies the only lawful new source but leaves open what must be
measured to acquire it.  There is an exact scalar answer for a fixed Bellman
endpoint, and an equally exact obstruction to obtaining that answer from the
Gaussian rest alone.

Fix the comparator \(C\) declared in (67.2a), and let

\[
 A_a=C_a-E C_a,\qquad
 e_a=(Y_a^{true}-C_a)-E(Y_a^{true}-C_a),
 \qquad A_a(t)=A_a+t e_a .
\]

Translation invariance of cumulants makes this centering harmless.  Direct
differentiation of the four-linear cumulant gives

\[
\begin{aligned}
 \Delta_{i;jk}
 :={}&\kappa_4(A_i+e_i,A_i+e_i,A_j+e_j,A_k+e_k)
       -\kappa_4(A_i,A_i,A_j,A_k)\\
 ={}&\int_0^1 \big\{
 2\kappa_4(e_i,A_i(t),A_j(t),A_k(t))
 +\kappa_4(A_i(t),A_i(t),e_j,A_k(t))\\
 &\hspace{38mm}
 +\kappa_4(A_i(t),A_i(t),A_j(t),e_k)
 \big\}\,dt .
\end{aligned} \tag{68.1}
\]

Thus for a fixed same-seam costate \(\Lambda_{i;jk}\), the exact additional
datum is the single signed surface action

\[
 \boxed{
 s_{\Lambda}^{rem}
 =\int_0^1\sum_{i,j,k}\Lambda_{i;jk}\,
 D\kappa_{iijk}[A(t)]\,e\,dt,} \tag{68.2}
\]

followed by the collision convention and complete-old projection from
(67.3).  Equation (68.2) is the precise sense in which a true-minus-comparator
source can become a resident-tape scalar: an implementation must expose this
*signed mixed surface functional*.  A norm, covariance, or AQ-core word is
not the same datum.

### 68.1 Even every pairwise law leaves the sign unidentified

The failure is stronger than a missing inequality.  Let \(P,Q,R\) be
independent Rademacher variables (equivalently signs of three independent
Gaussian coordinates), fix \(0<\epsilon<1\), and for
\(\sigma\in\{-1,+1\}\) put

\[
 A_\sigma=P(1+\sigma\epsilon QR),\qquad B=Q,\qquad C=R . \tag{68.3}
\]

Adding a common constant larger than \(1+\epsilon\) makes all three
variables positive, so the construction is also the exact output of a ReLU
acting in its linear region; cumulants are unchanged by the shift.  The two
worlds \(\sigma=+1\) and \(\sigma=-1\) have identical one-dimensional and
pairwise joint laws.  For example, the sign change is absorbed by the
unobserved Rademacher in each two-coordinate marginal.  In particular they
have the same covariance, the same marginal moments, and the same
true-minus-comparator residual Gram if the comparator is \((P,Q,R)\):

\[
 e_A=\sigma\epsilon PQR,\qquad e_B=e_C=0,qquad
 E e_A^2=\epsilon^2,qquad E(e_A P)=E(e_AQ)=E(e_AR)=0. \tag{68.4}
\]

Nevertheless all covariance pairings in the `211` cumulant vanish and

\[
 \boxed{
 \kappa_4(A_\sigma,A_\sigma,B,C)=2\sigma\epsilon .} \tag{68.5}
\]

Indeed \(A_\sigma^2=(1+\epsilon^2)+2\sigma\epsilon QR\).
Consequently no statistic that factors through *even the complete family of
pairwise laws* can determine the sign of the q211 defect.  Against the two
indistinguishable worlds, every scalar estimate \(\widehat s\) obeys the
sharp minimax lower bound

\[
 \max_{\sigma=\pm1}
 |\widehat s-2\sigma\epsilon|^2\ge 4\epsilon^2 . \tag{68.6}
\]

The affine Gaussian-rest covariance is therefore useful as a size/control
variate receipt but cannot be an amplitude oracle for (67.6).

### 68.2 The obstruction is already a signed cubic-chaos tangent

There is a smooth Gaussian version at first order.  For independent standard
Gaussians \(X,Y,Z\), set

\[
 A_\sigma=X(1+\sigma\epsilon YZ),\qquad B=Y,\qquad C=Z . \tag{68.7}
\]

Again the residual is the signed third-chaos direction
\(e_A=\sigma\epsilon XYZ\), its covariance data are independent of
\(\sigma\), and the q211 defect is exactly \(2\sigma\epsilon\).  Hence the
missing datum is not an exotic fourth-order tail: it appears in the linear
response of q211 to a hidden signed H3 orientation.

More generally use the Wiener convention

\[
 E[I_3(Z)I_3(G)]=3!\langle Z,G\rangle_{\mathrm{Sym}^3}.
\]

At a linear Gaussian base \(A_a=I_1(u_a)\) with cubic residuals
\(e_a=I_3(Z_a)\), the exact first variation is

\[
\begin{aligned}
 D\kappa_{iijk}[e]
 ={}&12\langle Z_i,\operatorname{Sym}(u_i,u_j,u_k)\rangle\\
 &+6\langle Z_j,\operatorname{Sym}(u_i,u_i,u_k)\rangle
 +6\langle Z_k,\operatorname{Sym}(u_i,u_i,u_j)\rangle .
\end{aligned} \tag{68.8}
\]

After contraction with a fixed \(\Lambda\), (68.8) has the compiled form

\[
 s_{\Lambda}^{(1)}=\sum_a\langle Z_a,G_a^{(3)}(\Lambda,u)\rangle . \tag{68.9}
\]

The Gaussian rest supplies only Gram entries
\(6\langle Z_a,Z_b\rangle\), which are invariant under the sign ambiguity
in (68.7) and, more generally, under a common orthogonal rotation of the
hidden H3 coordinates.  A literal resident reverse tensor
\(G^{(3)}\) therefore reduces acquisition to one signed dot product, but it
does not manufacture the missing signed \(Z\).  This is exactly the boundary
between a terminal scalar and a forward native-q211/H3 carrier.

### 68.3 Remainder and fidelity gate

For centered variables, Holder's inequality gives

\[
 |\kappa_4(X_1,X_2,X_3,X_4)|
 \le 4\prod_{s=1}^4\|X_s\|_4 . \tag{68.10}
\]

If all comparator slots have \(L^4\) norm at most \(A\) and all residual slots at
most \(E\), the terms beyond the signed linear surface obey

\[
 \boxed{
 |R_{\ge2}|\le
 4\{6A^2E^2+4AE^3+E^4\}.} \tag{68.11}
\]

Thus a deployable scalar route has a clean two-part gate:

1. acquire the signed linear action (68.9), or the full interpolated action
   (68.2), from a lawful true-minus-comparator state/oracle;
2. either evaluate the nonlinear residual slots directly or certify them by
   (68.11) (with any available finite-chaos hypercontractive sharpening).

For a proposed signed H3 sketch \(\widehat Z\), the endpoint error is

\[
 |s_{\Lambda}^{(1)}-\widehat s_{\Lambda}^{(1)}|
 \le \|Z-\widehat Z\|\,\|G^{(3)}_\Lambda\| . \tag{68.12}
\]

No positive correlation or Schur-gain lower bound follows from
\(\|Z\|\) or its Gram alone, by (68.3)--(68.6).  The required novelty receipt
is the signed endpoint cosine/action after the complete-old projection.

Graph split:

- `TRUE_MINUS_COMPARATOR_Q211_SURFACE_IDENTITY`: solved by (68.1)--(68.2);
- `PAIRWISE_GAUSSIAN_REST_TO_Q211_SIGN`: NO-GO by (68.3)--(68.6);
- `SIGNED_H3_TO_Q211_LINEAR_ACTION`: solved by (68.8)--(68.9);
- `SIGNED_H3_SOURCE_ACQUISITION`: open; Gram/rest data are insufficient;
- `Q211_NONLINEAR_RESIDUAL_BOUND`: solved conditionally by (68.11), but an
  instance norm receipt is still required;
- `TRUE_MINUS_COMPARATOR_FORWARD_NATIVE_CARRIER`: conditional on signed source,
  reset persistence, cap, and complete-old Schur value.

Current next question: does the proposed replacement tape already retain a
signed H3 coordinate \(Z\) at the exact reset seam, with the same collision
and frame convention as \(G^{(3)}_\Lambda\)?  If not, the declared AQ branch
plus Gaussian rest cannot acquire (67.6); a signed H3 sketch/direct
mixed-moment oracle is a genuinely new state, not a reinterpretation of the
resident AQ action.  The comparator must be \(X^{AQ}\) unless the separate
Section 49 surface is actually included.

## 69. Exact conditional/Fourier acquisition of one projected q211 action

Section 68 proves that the missing signed action is not identifiable from
pairwise rest data.  This section gives the complementary positive theorem:
under a *literal independent Gaussian-rest decomposition*, one frozen
complete-old-projected endpoint can be acquired as one scalar expectation,
without ever materializing a forward CP4/q211 tensor.  It also states exactly
what information that scalar oracle must contain.

Let \(\Pi_{off}\) be the consumer's collision mask and let
\(\mathcal R_{old}\) be the complete-old residualizer in the same raw-versus-
`/24` convention.  Move both maps to the reverse side once:

\[
 \widetilde\Lambda
 :=\Pi_{off}^{*}\mathcal R_{old}^{*}\Lambda .
 \tag{69.1}
\]

Then, without any orthogonality assumption,

\[
 \langle\Lambda,\mathcal R_{old}\Pi_{off}
       (\kappa_4(Y^{true})-\kappa_4(C))\rangle
 =\langle\widetilde\Lambda,
       \kappa_4(Y^{true})-\kappa_4(C)\rangle .
 \tag{69.2}
\]

Thus complete-old ownership and collision sewing do not force a forward
tensor: for a fixed endpoint they only change the compiled costate.  Put
\(L_i[j,k]=\operatorname{Sym}_{jk}\widetilde\Lambda_{i;jk}\).  For a
post-reset vector \(Y\), mean \(\mu\), covariance \(V\), centered vector
\(z=Y-\mu\), and \(v_i=V_{i,:}^{T}\), define

\[
 F_{\widetilde\Lambda,\mu}(Y)
   :=\sum_i z_i^2 z^T L_i z,
 \qquad
 Q_{\widetilde\Lambda}(V)
   :=\sum_i\{V_{ii}\langle L_i,V\rangle_F
                    +2v_i^TL_iv_i\}.
 \tag{69.3}
\]

The fourth-cumulant pairing identity gives the exact scalar surface formula

\[
 \boxed{
 S_{\widetilde\Lambda}(Y)
 =E F_{\widetilde\Lambda,\mu}(Y)
  -Q_{\widetilde\Lambda}(V).}
 \tag{69.4}
\]

In particular, once the true and comparator first two moments are resident, the only
new fourth-order datum for this endpoint is one signed expectation
\(E F_{\widetilde\Lambda,\mu^{true}}(Y^{true})\).  For
\(C=X^{AQ}\), the comparator action is algebraic from the carried
projection.  For \(C=Y^{AQ}\), it is owned only by the explicitly included
Section 49 surface.  Equation (69.4), unlike a norm or Gram receipt,
includes the sign and every off-diagonal interference term.

### 69.1 A finite conditional Gaussian-rest identity

Assume at the reset, as a statement about random variables rather than only
covariances, that the true output has

\[
 U^{true}=G+R^{true},\qquad G\sim N(0,\Sigma_G),\qquad
 G\perp R^{true},\qquad Y^{true}=(U^{true})_+ .
 \tag{69.5}
\]

For the augmented comparator \(C=Y^{AQ}\), additionally assume
\(U^{AQ}=G+R^{AQ}\), \(G\perp R^{AQ}\), and
\(Y^{AQ}=(U^{AQ})_+\).  The two worlds may have different residual laws.
For the state-only comparator \(C=X^{AQ}\), no second ReLU-law premise is
needed because its action is subtracted directly from the carried state.  A
covariance decomposition
\(\operatorname{Cov}U=\Sigma_G+\Sigma_R\) is not enough for (69.5); literal
independence is the premise that makes the following factorization lawful.

For a nonempty coordinate set \(S\), a positive multi-index \(\beta\) on
\(S\), and \(r\in\mathbb R^{S}\), write

\[
 \Psi_S(s;r)
 :=e^{s^Tr+\frac12s^T\Sigma_{G,S}s}
   \Phi_S^+(r+\Sigma_{G,S}s,\Sigma_{G,S}),
 \tag{69.6}
\]

where \(\Phi_S^+(m,\Sigma)=P\{N(m,\Sigma)>0\}\).  Exponential tilting of a
Gaussian proves

\[
 \boxed{
 \mathcal M_{\beta,S}(r)
 :=E_G\prod_{a\in S}(G_a+r_a)_+^{\beta_a}
 =\left.\partial_s^\beta\Psi_S(s;r)\right|_{s=0}.}
 \tag{69.7}
\]

For an all-distinct `211` word \((i,i,j,k)\), its conditional centered
surface kernel is the finite twelve-term binomial combination

\[
\begin{aligned}
 H^{\mu}_{i;jk}(r_i,r_j,r_k)
 ={}&\sum_{b_i=0}^{2}\sum_{b_j=0}^{1}\sum_{b_k=0}^{1}
 {2\choose b_i}(-\mu_i)^{2-b_i}
 (-\mu_j)^{1-b_j}(-\mu_k)^{1-b_k}\\
 &\hspace{14mm}\cdot
 \mathcal M_{b,S_b}(r_{S_b}),
 \qquad S_b=\{a:b_a>0\},
\end{aligned}
\tag{69.8}
\]

with \(\mathcal M_{0,\varnothing}=1\).  The smaller-dimensional orthant
kernel must be used when an exponent is zero; replacing every term by one
three-way orthant event would incorrectly center inactive ReLU coordinates.

Compile the endpoint kernel

\[
 \mathcal H_{\widetilde\Lambda,\mu}(r)
 :=\sum_{i,j,k}\widetilde\Lambda_{i;jk}
       H^{\mu}_{i;jk}(r_i,r_j,r_k).
 \tag{69.9}
\]

For the augmented comparator \(C=Y^{AQ}\), equations (69.4)--(69.9) give
the exact acquisition identity

\[
\boxed{\begin{aligned}
 s^{rem}_\Lambda
 ={}&E_{R^{true}}\mathcal H_{\widetilde\Lambda,\mu^{true}}(R^{true})
    -E_{R^{AQ}}\mathcal H_{\widetilde\Lambda,\mu^{AQ}}(R^{AQ})\\
   &-Q_{\widetilde\Lambda}(V^{true})
    +Q_{\widetilde\Lambda}(V^{AQ}).
\end{aligned}}
\tag{69.10}
\]

For the carried-state-only comparator \(C=X^{AQ}\), replace the second
conditional term and its covariance pairing by the direct resident action:

\[
 \boxed{
 s_\Lambda^{rem}(X^{AQ})
 =E_{R^{true}}\mathcal H_{\widetilde\Lambda,\mu^{true}}(R^{true})
  -Q_{\widetilde\Lambda}(V^{true})
  -S_{\widetilde\Lambda}(X^{AQ}).} \tag{69.10a}
\]

This is a finite conditional identity: only derivatives through total degree
four of at most trivariate Gaussian orthant probabilities occur.  More
importantly, the whole right side can be accumulated as one signed scalar
for one frozen endpoint.  It never requires a CP decomposition, a q211 plane,
or the \(O(n^3)\) forward tensor.  If the first two moments are not already
resident, their degree-one and degree-two versions of (69.7) are additional
required inputs; they are not implied merely by naming a Gaussian rest.

### 69.2 Fourier form: only trivariate residual laws, but not finitely many generic samples

There is an equivalent identity that makes the information requirement
transparent.  For \(S=\{i,j,k\}\), define

\[
 f^{\mu}_{i;jk}(x_S)
 =((x_i)_+-\mu_i)^2((x_j)_+-\mu_j)((x_k)_+-\mu_k),
 \quad f_{\epsilon}=f e^{-\epsilon\|x_S\|^2}.
 \tag{69.11}
\]

Use \(\widehat f(t)=\int e^{-it^Tx}f(x)dx\) and
\(\phi_{R,S}(t)=E e^{it^TR_S}\).  If \(\Sigma_{G,S}>0\), Fourier inversion
and Gaussian convolution give

\[
 E f^{\mu}_{i;jk}(G_S+R_S)
 =\lim_{\epsilon\downarrow0}(2\pi)^{-3}
 \int_{\mathbb R^3}\widehat f^{\mu}_{i;jk,\epsilon}(t)
 e^{-\frac12t^T\Sigma_{G,S}t}\phi_{R,S}(t)\,dt .
 \tag{69.12}
\]

The limit is justified by a finite fourth moment; singular Gaussian rest
follows by an arbitrarily small isotropic regularization and a second limit.
Summing (69.12) with \(\widetilde\Lambda_{i;jk}\) and applying (69.4) recovers
(69.10).  Therefore the exact endpoint needs no full joint residual law:
the costate-weighted collection of *trivariate* residual characteristic
functions is sufficient.  In the augmented \(C=Y^{AQ}\) branch, when the
true and AQ means agree, their surface
difference is literally the same Fourier kernel applied to
\(\phi_{R,S}^{true}-\phi_{R,S}^{AQ}\); otherwise the two known centering
kernels in (69.12) are used separately.

No universal finite-frequency shortcut follows.  Fix frequencies
\(t_1,\ldots,t_m\).  Point evaluation of a characteristic function supplies
only the expectations of the finite family
\(1,\cos(t_l^Tr),\sin(t_l^Tr)\).  The conditional surface kernel in (69.9)
is not in that span: along a positive ray it has quartic growth, whereas a
finite trigonometric span is bounded.  Choose more support points than real
constraints.  Linear algebra then gives a nonzero signed measure \(\nu\)
with zero mass and \(\widehat\nu(t_l)=0\) for every queried frequency, but
\(\int\mathcal H\,d\nu\ne0\).  Adding sufficiently small \(\pm\eta\nu\)
to a strictly positive finitely supported law produces two probability laws
with identical queried Fourier data and different q211 actions.

Hence (69.12) is an exact integral oracle, not a claim that a fixed generic
quadrature is exact.  A finite rule becomes exact only after a separately
proved finite-dimensional model, for example

\[
 \phi_R(t)-\phi_R^{AQ}(t)=\sum_{q=1}^{d}\theta_q\psi_q(t).
 \tag{69.13}
\]

Then the endpoint is \(k^T\theta\), with \(k_q\) obtained by inserting
\(\psi_q\) in (69.12), and samples \(A\theta\) suffice if and only if

\[
 \boxed{k\in\operatorname{rowspan}A.}
 \tag{69.14}
\]

This is the exact finite-Fourier acquisition gate; approximation error or a
conditioning claim is a separate receipt.

### 69.3 Minimal information and the terminal-scalar/forward-state boundary

Let \(\mathcal M_{owned}\) be the linear span of all functions of the reset
residual whose expectations are actually resident (constants, any proved
AQ-core actions, and any literal rest measurements).  On an unrestricted
admissible support, a target expectation \(E\mathcal H\) is determined by
those measurements for every residual law if and only if

\[
 \boxed{[\mathcal H]=0
 quad\text{in}\quad \mathcal F/\mathcal M_{owned}.}
 \tag{69.15}
\]

Necessity follows by the same signed-measure nullspace construction; the
reverse direction is immediate.  This statement allows arbitrary nonlinear
decoders: because both the measurements and target are affine in the law,
an exact decoder on all mixtures must be affine on the attainable moment
set.  For endpoint kernels \(\mathcal H_1,\ldots,\mathcal H_p\), the minimum
number of new exact scalar channels is

\[
 \boxed{
 \dim\operatorname{span}\{[\mathcal H_1],\ldots,[\mathcal H_p]\}
 \subseteq\mathcal F/\mathcal M_{owned}.}
 \tag{69.16}
\]

For one frozen endpoint that dimension is either zero (already owned) or
one; the direct signed surface expectation in (69.10) is therefore minimal.
For an adaptive portfolio, a future nonlinear reset, or a family of
independent reverse costates, it is generically larger than one.  That is the
precise point at which a signed H3/native-q211 dictionary or another forward
state becomes necessary.

Section 68.1 supplies a stronger obstruction than the finite-frequency
argument: two worlds can have identical *complete pairwise laws*, AQ cores,
and residual Grams while their q211 actions are \(\pm2\epsilon\).  For a
costate coefficient \(\lambda\) on that word, every rule using only those
owned data has worst-case squared error at least
\(4\lambda^2\epsilon^2\).  Adding the same independent Gaussian noise to
both worlds preserves equality of every pairwise law; after coordinatewise
ReLU the q211 separation persists for sufficiently small positive noise by
continuity.  Thus a nondegenerate Gaussian-rest label does not evade the
information-theoretic no-go.

The lawful source receipt is consequently one of exactly three things:

1. the direct signed scalar in (69.10) for a fixed compiled endpoint;
2. the requisite trivariate residual laws/characteristic action in (69.12);
3. a proved finite-dimensional signed residual coordinate satisfying
   (69.13)--(69.14).

AQ cores plus a covariance-only Gaussian rest provide none of these.  They
determine the carried comparator and pair terms, not the
true-minus-comparator signed surface.  The unprojected comparator action is
available only when the separate Section 49 surface is included.

Graph split:

- `Q211_COMPLETE_OLD_REVERSE_COMPILATION`: solved exactly by (69.1)--(69.2);
- `Q211_CONDITIONAL_GAUSSIAN_SURFACE_ORACLE`: solved conditionally on the
  literal independence premise (69.5), by (69.6)--(69.10);
- `Q211_TRIVARIATE_FOURIER_ACTION`: solved by (69.11)--(69.12); no full
  forward tensor or full residual joint law is required;
- `Q211_GENERIC_FINITE_FOURIER_QUADRATURE`: NO-GO; finite samples do not
  identify the action for arbitrary residual laws;
- `Q211_FINITE_FAMILY_ROWSPAN_GATE`: exact necessary-and-sufficient gate
  (69.13)--(69.14);
- `Q211_OWNED_INFORMATION_QUOTIENT`: solved minimal-channel theorem
  (69.15)--(69.16);
- `GAUSSIAN_REST_LITERAL_INDEPENDENCE_RECEIPT`: open provenance gate;
- `Q211_DIRECT_SIGNED_SURFACE_SOURCE`: open unless the deployed tape exposes
  one of the three lawful receipts above;
- `Q211_FORWARD_NATIVE_STATE`: unnecessary for one frozen scalar endpoint,
  but still required at the quotient rank for adaptive/multi-endpoint or
  nonlinear persistence.

## 70. Dependent projection residual: exact endpoint jet, ownership audit,
and the one-scalar deployability boundary

The independent decomposition in Section 69.1 is sufficient but not
necessary.  The actual AQ projection residual is dependent on its base, and
the exact fixed-endpoint theorem is simpler when stated through their *joint*
cumulant generator.

There are two objects that must not be conflated.  For any actual post-reset
law \(Z\), let

\[
 A=P_{AQ}Z,qquad r=Z-A,qquad
 a=A-EA .
 \tag{70.1}
\]

Here \(P_{AQ}\) is the orthogonal projection onto the resident constants plus
joint AQ \(H_{\le2}\) space.  Thus \(Er=0\) and

\[
 E[r_i a_j]=0\quad\hbox{for every }i,j,
 \tag{70.2}
\]

but \(r\) is generally neither independent of \(a\) nor Gaussian.  In the
parametric r49 construction specifically,

\[
 Z=Y^{AQ}=(g^{AQ})_+,qquad A=X=P_{\le2}Y^{AQ}.
 \tag{70.3}
\]

The packed \((m,u,A)\) state stores the projection in (70.3).  It does not by
itself store the connected fourth cumulant of \(Y^{AQ}\).  Section 49's fused
surface oracle is a separate, optional computation which *does* acquire the
full AQ q211 action when its query, quadrature, and cost receipts are paid.
The D10 covariance rest must never be credited with that connected q4 action.

### 70.1 The independence-free one-scalar identity

Define the centered joint cumulant generating function

\[
 \mathcal K_{a,r}(u,v)
 :=\log E\exp(u^Ta+v^Tr)
 \tag{70.4}
\]

in a neighborhood of zero, or interpret its fourth jet formally when only
fourth moments exist.  For the complete-old/collision-projected reverse
costate \(\widetilde\Lambda\) from (69.1), put

\[
 \mathscr D_{\widetilde\Lambda}
 :=\sum_{i,j,k}\widetilde\Lambda_{i;jk}
 \partial_{t_i}^2\partial_{t_j}\partial_{t_k}.
 \tag{70.5}
\]

Because \(Z-EZ=a+r\), the exact projection-defect action is

\[
 \boxed{
 s^{proj}_{\widetilde\Lambda}(Z)
 =\left.\mathscr D_{\widetilde\Lambda}
 \{\mathcal K_{a,r}(t,t)-\mathcal K_{a,r}(t,0)\}
 \right|_{t=0}.}
 \tag{70.6}
\]

No factorization of the joint law appears.  Equation (70.6) is the weakest
exact observable for one frozen endpoint: one contracted fourth jet of the
*joint* AQ-base/residual law.  Equivalently it is the signed surface scalar
in (69.4), with the covariance pairings included automatically by the log.

For the campaign's true-minus-full-AQ source, use distinct laws
\(Z^T=Y^{true}\) and \(Z^A=Y^{AQ}\).  Their projections and residuals give
the bookkeeping identity

\[
\begin{aligned}
 \langle\widetilde\Lambda,
   \kappa_4(Z^T)-\kappa_4(Z^A)\rangle
 ={}&s^{proj}_{\widetilde\Lambda}(Z^T)
    -s^{proj}_{\widetilde\Lambda}(Z^A)\\
 &+\langle\widetilde\Lambda,
   \kappa_4(P_{AQ}Z^T)-\kappa_4(P_{AQ}Z^A)\rangle .
\end{aligned}
\tag{70.7}
\]

If the two worlds are synchronized to the same projected AQ state, the last
line vanishes.  If Section 49's \(Y^{AQ}\) surface has actually been
evaluated, the entire AQ side of (70.7) is already owned and must be
subtracted once.  Re-expanding its dependent residual with tilted potentials
would be an alternative implementation of the same AQ action, not a second
source.  Conversely, merely carrying \(P_{\le2}Y^{AQ}\) does not make the
Section 49 surface resident.

### 70.2 Exact residual-arity split

For one word \((i,i,j,k)\), multilinearity gives the complete grouped
expansion

\[
\begin{aligned}
 \Delta\kappa_{iijk}^{(1)}={}&
 2\kappa(r_i,a_i,a_j,a_k)
 +\kappa(a_i,a_i,r_j,a_k)
 +\kappa(a_i,a_i,a_j,r_k),\\
 \Delta\kappa_{iijk}^{(2)}={}&
 \kappa(r_i,r_i,a_j,a_k)
 +2\kappa(r_i,a_i,r_j,a_k)
 +2\kappa(r_i,a_i,a_j,r_k)
 +\kappa(a_i,a_i,r_j,r_k),\\
 \Delta\kappa_{iijk}^{(3)}={}&
 \kappa(r_i,r_i,r_j,a_k)
 +\kappa(r_i,r_i,a_j,r_k)
 +2\kappa(r_i,a_i,r_j,r_k),\\
 \Delta\kappa_{iijk}^{(4)}={}&
 \kappa(r_i,r_i,r_j,r_k),\\
 \kappa(Z_i,Z_i,Z_j,Z_k)-\kappa(a_i,a_i,a_j,a_k)
 ={}&\sum_{p=1}^{4}\Delta\kappa_{iijk}^{(p)}.
\end{aligned}
\tag{70.8}
\]

Orthogonality (70.2) makes every one-residual cumulant its raw centered
moment.  A two-residual term retains exactly the residual-residual times
AQ-AQ pairing; every residual-AQ pairing vanishes.  Three-residual terms are
again raw centered moments, while the four-residual term retains its ordinary
residual pairings.  Hence (70.8) is both an arity census and a collision-safe
way to audit a proposed scalar generator.

Although there are four residual *slots*, q211 uses only the three distinct
source laws \(r_i,r_j,r_k\).  The existing one-residual tilted generator

\[
 \mathcal H_{r_i}(t)=E[r_i e^{t^Ta}]
 \tag{70.9}
\]

spans exactly \(\Delta\kappa^{(1)}\) when the required AQ query directions
and source law are the ones actually supplied.  A univariate tilted
truncated-moment potential also spans terms containing powers of one
residual law, such as \(r_i^2\).  Pair-specific bivariate potentials can add
the two-source terms.  Neither family generically spans

\[
 \kappa(r_i,a_i,r_j,r_k)quad\hbox{or}\quad
 \kappa(r_i,r_i,r_j,r_k),
 \tag{70.10}
\]

which depend on the trivariate residual copula.  Thus the full dependent
q211 defect is not a third jet of the existing one-residual AQ potential.
It is the *diagonal joint jet* (70.6).

There are two important ownership qualifications:

1. if \(Z=Y^{AQ}\) is literally generated by the resident affine-quadratic
   scalar law, Section 49 can evaluate its full surface directly and thereby
   bypass the arity split; this is paid AQ ownership, not D10 ownership;
2. the same AQ potential does not evaluate \(Y^{true}\) unless a literal
   equality-in-law/sufficient-statistic theorem places the truth residual in
   that parametric family.

### 70.3 A dependent high-chaos witness invisible to every univariate and
pairwise residual oracle

The missing trivariate orientation is not a technicality.  Let nine
independent standard Gaussians be divided into three triples and define

\[
 \xi=\prod_{a=1}^{3}\operatorname{sign}G_a,qquad
 \eta=\prod_{a=4}^{6}\operatorname{sign}G_a,qquad
 \zeta=\prod_{a=7}^{9}\operatorname{sign}G_a.
 \tag{70.11}
\]

These are independent Rademacher variables.  Each is orthogonal to every
Gaussian polynomial of total degree at most two.  For
\(0<\epsilon<1\) and \(\sigma\in\{-1,+1\}\), set

\[
 r_i^{\sigma}=\xi(1+\sigma\epsilon\eta\zeta),qquad
 r_j=\eta,qquad r_k=\zeta .
 \tag{70.12}
\]

Choose \(c>1+\epsilon\) and let
\(Z^sigma=(c+r_i^sigma,c+r_j,c+r_k)\).  Every coordinate is positive, so
this is literally a ReLU output in its linear region, and

\[
 P_{AQ}Z^\sigma=(c,c,c),qquad
 (I-P_{AQ})Z^\sigma=(r_i^\sigma,r_j,r_k).
 \tag{70.13}
\]

The two signs have identical univariate and pairwise joint laws.  Therefore
they agree on all residual marginals, every bivariate tilted potential, all
means/covariances, all K31/K22 faces, and every statistic factoring through
pairwise laws.  Their output third cumulants also agree.  More specifically,
the sign-dependent component \(\xi\eta\zeta\) starts at Gaussian chaos order
nine.  Thus the H3 component, every literal D10 Mehler band through order
six, and every H4/coordinate-CP4 birth are identical in the two worlds.
Nevertheless

\[
 \boxed{
 \kappa_4(r_i^\sigma,r_i^\sigma,r_j,r_k)
 =E[(r_i^\sigma)^2\eta\zeta]
 =2\sigma\epsilon .}
 \tag{70.14}
\]

This witness simultaneously establishes:

- AQ projection coefficients do not determine the dependent residual q211;
- all univariate *and all bivariate* residual potentials do not determine it;
- D10 degrees 3--6, CP3, coordinate CP4, K31/K22 star and pairing summaries
  can remain fixed while its sign flips;
- a higher-chaos norm or residual Gram cannot replace the signed action.

The literal D11 conclusion is independently strengthened by Section 59:
the complete deployed CP4/star reset map has a tangent in its kernel on
which off-diagonal q211 changes.  Thus (70.14) is not relying only on a
dimension count or on omitting an old interface.

### 70.4 Exact row-span criterion

Let \(\mathcal M_{res}\) denote the linear span of all reset test functions
whose expectations are literally available: the AQ parametric coordinates,
the D10 degree-3--6 Gaussian block, deployed CP3/CP4/star/pair reads, and any
tilted residual potentials actually evaluated.  Means and covariance
pairings are included.  Let \(h_{\widetilde\Lambda}\) be the centered
quartic surface polynomial representing (70.6).

For unrestricted finitely supported perturbations of the dependent joint
law, existing observables identify the endpoint action if and only if

\[
 \boxed{
 h_{\widetilde\Lambda}\in
 \mathcal M_{res}+\mathcal N,}
 \tag{70.15}
\]

where \(\mathcal N\) is the space of pointwise/projection identities imposed
on every admissible law.  Equivalently, every signed measure annihilating
the resident observables must annihilate the surface.  If (70.15) fails,
choose a finite-support signed null measure \(\nu\) with
\(\int h\,d\nu\ne0\) and perturb a positive baseline by
\(\pm\delta\nu\).  The two laws have identical resident state and different
q211 action.  This proves nonidentifiability even for an arbitrary nonlinear
decoder of the resident numbers.

Locally, (70.15) is the differential criterion

\[
 \boxed{
 \ker D O_{resident}\subseteq
 \ker D s_{\widetilde\Lambda}.}
 \tag{70.16}
\]

Section 59 gives an explicit violation for current D11, while (70.11)--
(70.14) gives a global violation even after all univariate and bivariate
residual-law observables are added.  Complete-old reverse compilation makes
the ownership implication sharp: if old CP3/CP4/star/D10 truly spans the
query, the projected \(\widetilde\Lambda\) or its surface class vanishes; if
it does not vanish, those same old objects cannot be reused to infer the
missing signed coefficient.

### 70.5 Literal source, bandwidth, and arithmetic census

The phrase “one scalar” concerns information bandwidth, not automatically
arithmetic or provenance.

| interface | exact information supplied | incremental output/state | binding cost or gap |
|---|---|---:|---|
| packed \(P_{AQ}Z\) | constants, H1, H2 coefficients | already resident | no dependent q4 action |
| Section 49 \(Y^{AQ}\) surface | complete AQ q211 action for its supplied query | one scalar per evaluated query | optional paid quadrature/resolvent and query-rank cost; not implied by the packed state |
| D10 | literal affine-Gaussian Mehler bands 3--6 and pair covariance rest | already resident | no non-Gaussian connected q211 sign |
| CP3 | deployed cubic factors/actions | already resident | sign-changing witness begins at H9; complete-old contribution is subtracted |
| CP4 + star/pair | transported capped CP4, coordinate births, K31 star, K22 pairing read | already resident | Section 59 reset-birth kernel; no generic offdiag q211 source port |
| univariate AQ tilted jets | one residual law against AQ products; powers of that same residual | one scalar jet per source/query tuple | misses generic two-/three-source residual copulas |
| bivariate residual potential | one selected residual pair law | one scalar family per pair/query tuple | q211 still has the three-source terms (70.10) |
| direct signed surface | complete (70.6) or true-minus-AQ (70.7) for one frozen \(\widetilde\Lambda\) | exactly one scalar per active seam and endpoint | needs the actual joint law and the compiled reverse query; neither is resident in r49/D10 |

If the direct surface query is supplied in native rank \(R\),

\[
 h(z)=\sum_{q=1}^{R}c_q
 (A_q^Tz)^2(B_q^Tz)(C_q^Tz),
 \tag{70.17}
\]

then a streaming read uses at most \(3R\) projections, stores
\(R(3n+1)\) scalars before sharing, and costs \(O(nR)\) arithmetic per
sample plus the expectation/integration mechanism.  This is genuinely low
bandwidth when \(R\) is certified small.  With no factor receipt, the
all-distinct q211 query has

\[
 N_{211}=n{n-1\choose2}
 \tag{70.18}
\]

independent anchor/pair coefficients: at \(n=1024\), 535,299,072 numbers,
about 2.0 GiB in f32 or 4.0 GiB in f64 per seam.  A one-scalar *output* does
not make that dense query free.  Moreover the declared r49/D10 forward state
does not contain the positive-depth reverse Bellman matrix; forming its dense
chain is a separate reverse-conjugation/reset cost.

For \(L\) active seams, a genuine fused oracle returns only \(L\) scalars
(80 bytes in f64 when \(L=10\)).  But it is deployable only under three
literal receipts:

1. the oracle integrates the actual dependent truth law, not the AQ
   comparator or a covariance-matched Gaussian surrogate;
2. the complete-old projected reverse query is resident or has a certified
   small decomposition such as (70.17);
3. its scalar is target-frozen.  A changed endpoint, adaptive refit, or
   reusable forward repair requires as many channels as the rank of the
   endpoint surface classes in the quotient (69.16).

Thus one additional signed surface measurement is mathematically sufficient
and information-minimal for one fixed endpoint.  It is not presently a
zero-cost resident measurement.  The remaining deployability problem is no
longer “how to represent q211”; it is to exhibit the actual-law scalar oracle
and a low-rank positive-depth query at the same seam.

Atomic graph split:

- `Q211_DEPENDENT_JOINT_CGF_IDENTITY`: **solved exact**, no independence, by
  (70.4)--(70.7);
- `Q211_DEPENDENT_RESIDUAL_ARITY_SPLIT`: **solved exact** by (70.8); q211
  needs up to three distinct residual laws;
- `AQ_TILTED_ONE_RESIDUAL_Q211_SPAN`: **solved conditional** for the
  one-residual sector and same-source powers only;
- `AQ_BIVARIATE_TO_FULL_Q211`: **NO-GO generically**, by (70.10)--(70.14);
- `D10_CP3_CP4_STAR_TO_DEPENDENT_Q211`: **NO-GO for the current interface**,
  by the H9 sign witness plus Section 59's literal reset kernel;
- `AQ_Y_SURFACE_ORACLE_OWNERSHIP`: **conditional**; owned only when Section
  49 is actually evaluated/charged, and never by covariance rest alone;
- `Q211_DEPENDENT_ROWSPAN_GATE`: **solved necessary and sufficient** by
  (70.15)--(70.16);
- `Q211_FIXED_ENDPOINT_SIGNED_SURFACE_CHANNEL`: **information GO / deployment
  conditional**; exactly one scalar once the actual-law oracle and query are
  supplied;
- `Q211_SURFACE_QUERY_FACTOR_RANK`: **open blocking cost receipt**; certify
  small \(R\) in (70.17), otherwise (70.18) binds;
- `Q211_ACTUAL_DEPENDENT_SOURCE_ORACLE`: **open blocking provenance receipt**;
- `R49_Q211_REVERSE_QUERY_RESIDENCY`: **open/absent in declared forward
  state**; do not hide the reverse chain inside D10;
- `Q211_MULTIENDPOINT_CHANNEL_RANK`: **solved by quotient rank (69.16)**;
  one target-specific scalar does not become a reusable forward carrier.

## 73. Exact factorization and cost theorem for the D10 Bellman q211 query

Sections 69--70 reduce a frozen endpoint to the signed surface

\[
 F_L(z)=\sum_{i=1}^n z_i^2 z^TL_i z,
 \qquad L_i=L_i^T,
 \tag{73.1}
\]

with collision and complete-old projection already compiled into the slices
\(L_i\).  The scalar output is cheap only if the *query slices* have an exact
factorization that is born and propagated without first materializing the
dense Bellman object.  This section gives the necessary-and-sufficient ranks,
the exact contraction, and the generic obstruction.

### 73.1 The two intrinsic ranks

For the realized family \(L=(L_1,\ldots,L_n)\), define

\[
 \mathcal S_L:=\sum_i\operatorname{im}L_i,
 \qquad
 k_*:=\dim\mathcal S_L,
 \qquad
 p_*:=\dim\operatorname{span}\{L_1,\ldots,L_n\}.
 \tag{73.2}
\]

These are exactly the pair-support rank and anchor rank.

**Common-support theorem.**  There are \(U\in\mathbb R^{n\times k}\) with
orthonormal columns and symmetric \(\Theta_i\in\operatorname{Sym}(k)\)
such that

\[
 L_i=U\Theta_iU^T\quad\hbox{for every }i
 \tag{73.3}
\]

if and only if \(k\ge k_*\).  Indeed, symmetry makes
\(\operatorname{im}L_i\subseteq\operatorname{col}U\) equivalent to
\(L_i=UU^TL_iUU^T\).  Thus a minimal \(U\) is any orthonormal basis of
\(\mathcal S_L\), and

\[
 k_* =\operatorname{rank}[L_1\ L_2\ \cdots\ L_n].
 \tag{73.4}
\]

Inside a minimal support, form the anchor-pair flattening

\[
 \mathfrak L_{i,:}=\operatorname{vech}(\Theta_i)^T
 \in\mathbb R^{d_k},qquad d_k={k(k+1)\over2}.
 \tag{73.5}
\]

Then \(\operatorname{rank}\mathfrak L=p_*\).  Consequently there are
\(C\in\mathbb R^{n\times p_*}\) and a basis
\(B_1,\ldots,B_{p_*}\in\operatorname{Sym}(k_*)\) such that

\[
 \boxed{
 L_i=U\left(\sum_{\alpha=1}^{p_*}C_{i\alpha}B_\alpha\right)U^T.}
 \tag{73.6}
\]

Conversely every representation (73.6) has pair-support rank at most \(k\)
and anchor rank at most \(p\).  Thus \((k_*,p_*)\), not the rank of one
terminal slice, is the exact Tucker-style compression certificate.

### 73.2 Exact tensor-free contraction

Let

\[
 q=U^Tz,qquad h=C^T(z\odot z),qquad
 g_\alpha=q^TB_\alpha q .
 \tag{73.7}
\]

Substitution in (73.1) gives

\[
 \boxed{F_L(z)=h^Tg.}
 \tag{73.8}
\]

No \(n\times n\times n\) q211 object occurs.  With dense \(B_\alpha\), a
literal coefficient and per-integrand census is

\[
 \boxed{
 N_{Tucker}=nk+np+p{ k(k+1)\over2},
 \qquad
 C_{Tucker}=O\!\left(nk+np+pk^2\right).}
 \tag{73.9}
\]

Common pair support alone corresponds to \(p=p_*=\dim\operatorname{span}
\{L_i\}\), which may still be \(n\); it is not by itself a cheap surface.

There is a sharper partially symmetric CP form.  Define \(R_*\) to be the
least \(R\) for which

\[
 \Theta_i=\sum_{q=1}^{R}D_{iq}
 \operatorname{Sym}(b_qc_q^T),
 \qquad b_q,c_q\in\mathbb R^k .
 \tag{73.10}
\]

Then

\[
 \boxed{
 F_L(z)=\sum_{q=1}^{R}
 \{D_{:q}^T(z\odot z)\}
 (b_q^TU^Tz)(c_q^TU^Tz).}
 \tag{73.11}
\]

The exact storage and contraction census becomes

\[
 \boxed{
 N_{CP}=nk+nR+2kR,
 \qquad
 C_{CP}=O(nk+nR+kR).}
 \tag{73.12}
\]

Every dense \(B_\alpha\) can be spectrally split, so
\(R_*\le p_*k_*\).  Conversely \(R_*\ge p_*\), because one term in
(73.10) is rank one in the anchor-versus-pair flattening.  Formula (73.11)
includes the native doubled-ray query of Section 70.17 as a more restricted
special case.  Shared rays or diagonal \(B_\alpha\) reduce the displayed
counts further, but only when the sharing is an exact identity.

The covariance pairing subtraction in (69.4) uses the same factors.  For
example

\[
 \langle L_i,V\rangle
 =\sum_\alpha C_{i\alpha}
 \langle B_\alpha,U^TVU\rangle,
 \qquad
 v_i^TL_iv_i
 =\sum_\alpha C_{i\alpha}
 (U^Tv_i)^TB_\alpha(U^Tv_i),
 \tag{73.13}
\]

so an exact factorized raw surface does not hide a dense cumulant
counterterm.

### 73.3 The exact D10 reverse-closure criterion

Write the one-reset Bellman pullback in the typed form already audited in
Sections 47 and 63,

\[
 L_{\ell,p}
 =\sum_i A_{\ell,pi}
 B_{\ell i}^T L_{\ell+1,i}B_{\ell i},
 \tag{73.14}
\]

where \(A_\ell\) mixes repeated-anchor labels and \(B_{\ell i}\) is the
anchor-dependent pair leg through the D10/ReLU reset.  For the *realized*
future slices, the smallest exact preceding pair space is

\[
 \boxed{
 \mathcal S_\ell^*
 =\operatorname{span}_i
 B_{\ell i}^T\operatorname{im}L_{\ell+1,i},}
 \tag{73.15}
\]

provided the visible columns of \(A_\ell\) can be recovered; in particular
this is exact when the relevant anchor mixing is full column rank.  Without
that premise, the minimal realized support is obtained directly from the
output slices in (73.14), while (73.15) remains the architecture-complete
support with no cancellation assumption.

Suppose \(L_{\ell+1,i}=U_{\ell+1}\Theta_{\ell+1,i}U_{\ell+1}^T\).  A
width-\(k_\ell\) common support propagates exactly if and only if there are
matrices \(R_{\ell i}\) satisfying

\[
 \boxed{
 B_{\ell i}^TU_{\ell+1}=U_\ell R_{\ell i}
 \quad\hbox{for every active }i.}
 \tag{73.16}
\]

Under (73.16), the entire physical recurrence reduces to

\[
 \boxed{
 \Theta_{\ell,p}
 =\sum_i A_{\ell,pi}
 R_{\ell i}\Theta_{\ell+1,i}R_{\ell i}^T.}
 \tag{73.17}
\]

This is necessary, not merely sufficient, when the future \(\Theta_i\) may
range over full-support forms.  It is the literal seam receipt that a claimed
common factor \(U\) must satisfy.

For anchor-rank closure, let
\(\mathcal B_{\ell+1}=\operatorname{span}\{B_\alpha\}\subseteq
\operatorname{Sym}(k_{\ell+1})\).  The minimal architecture-complete
preceding matrix space is

\[
 \boxed{
 \mathcal B_\ell^*
 =\sum_i R_{\ell i}\mathcal B_{\ell+1}R_{\ell i}^T.}
 \tag{73.18}
\]

For a realized terminal coefficient family, replace the right side by the
span of the actual matrices in (73.17).  A fixed \(p\)-dimensional basis is
closed only if every congruence in (73.18) returns to that basis.  If the
semigroup generated by nonsingular \(R_{\ell i}\) is Zariski dense in
\(GL(k)\), the \(\operatorname{Sym}^2\) representation is irreducible;
hence the only nonzero universally invariant matrix space is

\[
 \mathcal B=\operatorname{Sym}(k),qquad p=d_k.
 \tag{73.19}
\]

Low \(p\) therefore requires a literal simultaneous structure: common
monomial/diagonal action, block algebra, shared invariant rays, or another
proved proper congruence-invariant subspace.  A numerical low rank at the
terminal seam does not provide it.

The CP form has the same boundary.  Substituting (73.10) into (73.14) creates
one transformed atom

\[
 \operatorname{Sym}{(B_{\ell i}^TUb_q)
                     (B_{\ell i}^TUc_q)^T\}
 \tag{73.20}
\]

for every active \((i,q)\).  Absent exact coalescence, its rank multiplies by
the number of anchors before saturating the ambient partially symmetric
rank.  Recompressing after materialization is not a free recurrence.

### 73.4 Generic lower bounds, including the actual D10 reset geometry

For a generic all-distinct q211 family, each slice is a generic symmetric
zero-diagonal matrix on the \(n-1\) coordinates other than its anchor.  Such
a slice has rank \(n-1\), and two different anchor slices generically span
all of \(\mathbb R^n\).  The \(n\) slices are also generically linearly
independent.  Therefore

\[
 \boxed{k_*=n,qquad p_*=n\quad\hbox{generically}.}
 \tag{73.21}
\]

The Tucker census (73.9) then returns to order \(n^3/2\).

The CP obstruction is stronger than the flattening bound \(R\ge n\).  The
all-distinct q211 coefficient space has dimension

\[
 N_{211}=n{n-1\choose2}.
 \tag{73.22}
\]

One anchor-pair atom in (73.10) has at most \(3n-2\) continuous parameters
before collision projection (two scaling gauges are redundant).  Hence a
secant-variety dimension count gives the generic exact lower bound

\[
 \boxed{
 R_*\ge
 \left\lceil{n{n-1\choose2}\over3n-2}\right\rceil.}
 \tag{73.23}
\]

At \(n=1024\), (73.23) is \(R_*\ge174{,}365\).  Substitution in (73.12)
again requires approximately the 535 million coefficients of the dense
query.  Native doubled-ray atoms form a subset of the anchor-pair atoms, so
the same lower bound applies to them.

This generic conclusion does not rely on arbitrary unstructured reset legs.
Section 47 constructs bounded-condition, diffuse ReLU legs of the literal
form \(B_i=D_iH\) for which a common terminal rank-one slice
\(uu^T\) pulls back to \(v_iv_i^T\) with

\[
 \dim\operatorname{span}\{v_i\}=m.
 \tag{73.24}
\]

With invertible anchor mixing, every slice is recoverable, so any common
support has \(k\ge m\).  Taking the active anchor block proportional to
width makes one D10-style reset expand rank one to \(\Omega(n)\) despite
diffuse gates and bounded condition number.  This is the sharp counterexample
to inferring cheap positive-depth structure from a scalar or rank-one head.

### 73.5 Storage, acquisition, and recurrence are three separate bills

1. **Offline coefficient storage.**  A frozen factor table is useful only if
   it is valid for the network being evaluated.  D10 gates, pair legs, and
   Bellman slices are network-specific.  A factorization obtained after
   constructing a dense per-network \(L\) has already paid the forbidden
   storage/acquisition bill and does not establish deployability.

2. **Per-network surface acquisition.**  Given lawful factors, (73.8) or
   (73.11) evaluates the integrand at the stated cost and returns one scalar
   expectation.  The actual dependent-law integration from Section 70 is a
   separate oracle.  Neither factorizing the query nor possessing an AQ
   surface oracle manufactures the other.

3. **Positive-depth reverse recurrence.**  The factors must be generated
   directly through (73.16)--(73.18), including every gate, collision, and
   old-span convention.  The declared r49/D10 object is a forward state; it
   does not make this reverse Bellman recurrence resident.  New physical
   pulls \(B_i^TU\), congruences \(R_iB_\alpha R_i^T\), anchor mixing, and
   any rank growth are separately charged.

Approximate compression needs an endpoint error receipt.  If
\(\delta=\max_i\|L_i-\widehat L_i\|_{op}\), then

\[
 |E(F_L-F_{\widehat L})|
 \le \delta E\|z\|^4,
 \tag{73.25}
\]

and, for covariance \(V\), the cumulant pairing error obeys

\[
 |Q_L(V)-Q_{\widehat L}(V)|
 \le\delta\{(\operatorname{tr}V)^2+2\|V\|_F^2\}.
 \tag{73.26}
\]

These errors must enter the robust complete-old bound (63.10).  Approximate
small rank without (73.25)--(73.26) is not an exact cost escape.

The deployable conclusion is therefore sharp:

\[
 \boxed{
 \begin{gathered}
 \text{one frozen q211 surface is cheap}\iff
 \text{its realized }(k_*,p_*)\text{ or }R_*\text{ is small,}\\
 \text{and the same factor class closes under every D10 reverse reset.}
 \end{gathered}}
 \tag{73.27}
\]

Atomic graph split:

- `D10_Q211_COMMON_PAIR_SUPPORT_RANK`: **solved exact** by
  (73.2)--(73.4);
- `D10_Q211_ANCHOR_FLATTEN_RANK`: **solved exact** by (73.5)--(73.6);
- `D10_Q211_TUCKER_SURFACE_CONTRACTION`: **solved exact/costed** by
  (73.7)--(73.9) and (73.13);
- `D10_Q211_PARTIAL_CP_SURFACE_CONTRACTION`: **solved exact/costed** by
  (73.10)--(73.12);
- `D10_Q211_REVERSE_SUPPORT_INTERTWINER`: **open literal receipt**;
  necessary-and-sufficient condition (73.16);
- `D10_Q211_REVERSE_ANCHOR_SPACE_CLOSURE`: **open literal receipt**;
  condition (73.18), generically full by (73.19);
- `D10_Q211_GENERIC_FACTOR_RANK_NOGO`: **closed generic theorem** by
  (73.21)--(73.24), including the bounded-condition D10 geometry;
- `D10_Q211_FACTOR_STORAGE_PROVENANCE`: **open**; factors must be generated
  per network without first constructing dense \(L\);
- `D10_Q211_DEPENDENT_SURFACE_ACQUISITION`: **open and separate** from query
  compression;
- `D10_Q211_APPROXIMATE_QUERY_ERROR`: **solved bound**, instance receipt
  still required through (73.25)--(73.26);
- `D10_Q211_REVERSE_RECURRENCE_BILL`: **open**; no forward-state reuse credit
  without literal buffers and operations.

## 74. A concrete state-only reopen: the exact first affine-ReLU rest band

Section 67 shows that the carried projection \(X=P_{\le2}Y\) does not own
the connected affine-Gaussian rest \(\kappa_4(Y)-\kappa_4(X)\).  The full
Section 49 surface owns it when that optional oracle is included.  In the
state-only branch, however, its first omitted Hermite band is an exact,
label-free covariance polynomial and admits a target-factorized
\(O(n^2)\)-action evaluator.

Let \((Z_1,\ldots,Z_n)\) be standardized jointly Gaussian with correlation
matrix \(R=(\rho_{ij})\), and put

\[
 F(z)=z_+-\phi_0,\qquad \phi_0=(2\pi)^{-1/2}.
\]

For orthonormal probabilists' Hermites
\(h_m=\operatorname{He}_m/\sqrt{m!}\), integration by parts gives

\[
 F(z)=\frac12h_1(z)+\sum_{s\ge1}
 \frac{\phi_0\operatorname{He}_{2s-2}(0)}{\sqrt{(2s)!}}
 h_{2s}(z). \tag{74.1}
\]

In particular,

\[
 c_1=\frac12,\qquad c_2=\frac{\phi_0}{\sqrt2},\qquad
 c_4=-\frac{\phi_0}{\sqrt{24}}. \tag{74.2}
\]

The carried affine projection is \(X=c_1h_1+c_2h_2\).  By the connected
Wick-diagram theorem, the lowest-degree term in
\(\kappa_4(F_1,F_2,F_3,F_4)-\kappa_4(X_1,X_2,X_3,X_4)\) has chaos degrees
\((4,2,1,1)\), in all ordered assignments.  For an assignment with H4
vertex \(a\), H2 vertex \(b\), and remaining vertices \(c,d\), the unique
connected graph has two \(ab\) edges and one each of \(ac,ad\).  Its 24
pairings give

\[
 E[h_4(Z_a)h_2(Z_b)h_1(Z_c)h_1(Z_d)]
 =2\sqrt3\,\rho_{ab}^2\rho_{ac}\rho_{ad}.
\]

Multiplying (74.2) yields the especially simple exact coefficient

\[
 c_4c_2c_1^2(2\sqrt3)=-\frac{\phi_0^2}{8}
 =-\frac1{16\pi}. \tag{74.3}
\]

### 74.1 Closed q211 polynomial

For the repeated word \((i,i,j,k)\), write

\[
 x=\rho_{ij},\qquad y=\rho_{ik},\qquad z=\rho_{jk}.
\]

Summing the twelve ordered H4/H2 assignments gives the complete first-rest
band

\[
 \boxed{
 \Delta_{i;jk}^{[4]}
 =-\frac1{16\pi}B_4(x,y,z),} \tag{74.4}
\]

where

\[
 \boxed{
B_4(x,y,z)=
2xy(1+x+y)+2z(x^3+y^3)+z^2(x^2+y^2).} \tag{74.5}
\]

There is also an exact global factor view.  Let \(r_a=R_{a,:}\),
\(s_a=r_a^{\odot2}\), and let \(\operatorname{Sym}_{\Sigma}\) denote the
unnormalized sum over the twelve distinct placements of four slots with the
last factor repeated.  Then the full first-rest tensor before collision
projection is

\[
 \boxed{
 \Delta^{[4]}=-\frac1{16\pi}
 \sum_{a=1}^n
 \operatorname{Sym}_{\Sigma}(e_a,s_a,r_a,r_a).} \tag{74.5a}
\]

Equation (74.5a) explains both sides of the deployability boundary.  It is
an exact row-factorized source built only from resident covariance, but it
has \(n\) anchor terms before any task-specific compression.  It is not a
small forward native carrier merely because each summand is structured.
Conversely a frozen low-rank reverse query can contract the sum without
materializing it, as shown below.  Literal CP4/Q24 ownership is exactly the
row-span test of (74.5a) after the old projection; naming either interface
does not prove that ownership.

This is a connected fourth-cumulant contribution, not the D10 pairwise rest
covariance.  It is exactly determined by the lawful covariance bytes and is
label-free.  It must still be collision-masked and complete-old residualized.
It is additive only when the declared comparator is \(C=X^{AQ}\); when the
complete Section 49 \(Y^{AQ}\) surface is included, (74.4) is already part
of that owned action and receives zero separate credit.

Equations (74.1)--(74.5) use zero bias and unit marginal variance.  For
zero-bias variables \(G_a=\sigma_a Z_a\), positive homogeneity multiplies
the q211 word by \(\sigma_i^2\sigma_j\sigma_k\).  A nonzero standardized
bias changes the Hermite coefficients separately at every unit; the same
connected-diagram derivation remains valid, but the universal coefficient
\(-1/(16\pi)\) and the compact symmetric polynomial (74.5) must be replaced
by the corresponding slot-dependent \(c_4(a)c_2(b)c_1(c)c_1(d)\) sum.
Therefore (74.4) is not a bias-agnostic deployment formula.

The diagonal check is informative but not a value claim.  At
\(x=y=z=1\),

\[
 \Delta_{iiii}^{[4]}=-\frac{3}{4\pi}. \tag{74.6}
\]

Meanwhile Section 67's exact scalar affine defect is

\[
 \Delta_{iiii}^{all}
 =\frac34-\frac5{2\pi}-\frac9{4\pi^2}.
\]

Both are negative, and the H4 band accounts for the absolute fraction

\[
 \frac{3/(4\pi)}{5/(2\pi)+9/(4\pi^2)-3/4}\approx0.872
 \tag{74.7}
\]

on the coincident scalar word.  This 87.2% number is only a normalization
receipt; off-diagonal endpoint fidelity must be certified in its own metric.

### 74.2 Exact fixed-costate action without an n-cubed tensor

Let \(L_i[j,k]\) be the symmetric, complete-old/collision-compiled q211
costate.  For row \(r_i=R_{i,:}\), the matrix with entries
\(B_4(R_{ij},R_{ik},R_{jk})\) is

\[
\begin{aligned}
 \mathcal B_i={}&2r_ir_i^T
 +2\{r_i^{\odot2}r_i^T+r_i(r_i^{\odot2})^T\}\\
 &+2\{D_{r_i^{\odot3}}R+RD_{r_i^{\odot3}}\}
 +D_{r_i^{\odot2}}R^{\odot2}
 +(R^{\odot2})D_{r_i^{\odot2}} .
\end{aligned} \tag{74.8}
\]

Thus the exact band action is

\[
 s^{[4]}=-\frac1{16\pi}\sum_i\langle L_i,\mathcal B_i\rangle_F. \tag{74.9}
\]

A dense \(L\) still costs \(\Theta(n^3)\); “one scalar output” does not make
the query free.  But suppose one costate atom is

\[
 L_i[j,k]=\alpha_i\,\operatorname{Sym}_{jk}(\beta_j\gamma_k).
\]

Write \(R_p=R^{\odot p}\),
\(p_\beta=R\beta,p_\gamma=R\gamma\), and
\(q_\beta=R_2\beta,q_\gamma=R_2\gamma\).  Direct contraction of (74.8)
gives

\[
\begin{aligned}
 \langle L,\mathcal B\rangle={}&
 2\langle\alpha,p_\beta\odot p_\gamma\rangle\\
 &+2\langle\alpha,q_\beta\odot p_\gamma
                         +p_\beta\odot q_\gamma\rangle\\
 &+2\alpha^TR_3\{\beta\odot p_\gamma
                         +\gamma\odot p_\beta\}\\
 &+\alpha^TR_2\{\beta\odot q_\gamma
                         +\gamma\odot q_\beta\}.
\end{aligned} \tag{74.10}
\]

This uses eight matrix-vector passes per generic atom and no dense
matrix-matrix product.  If \(\beta=\gamma\), symmetry reduces it to four
passes:

\[
 2\langle\alpha,p^{\odot2}\rangle
 +4\langle\alpha,q\odot p\rangle
 +4\alpha^TR_3(\beta\odot p)
 +2\alpha^TR_2(\beta\odot q). \tag{74.11}
\]

Across ten active seams, the optimistic arithmetic census is therefore
\(80R_c/n\) width-GEMM equivalents for generic CP costate rank \(R_c\), or
\(40R_c/n\) for symmetric atoms, before factor construction, reverse
recurrence, hooks, and writes.  At \(n=1024\), the one-equivalent necessary
screens are respectively

\[
 R_c\le12\quad\hbox{or}\quad R_c\le25. \tag{74.12}
\]

Shared singleton dictionaries can reuse the first four products and improve
this count; unrelated atoms cannot.

### 74.3 Honest promotion and tail gate

There is a sharp correlation-only source ceiling.  If the three off-diagonal
correlations obey \(|x|,|y|,|z|\le\rho\le1\), then

\[
 \boxed{
 |B_4(x,y,z)|\le2\rho^2+4\rho^3+6\rho^4,\qquad
 |\Delta_{i;jk}^{[4]}|
 \le\frac{2\rho^2+4\rho^3+6\rho^4}{16\pi}.} \tag{74.13}
\]

For \(0\le\rho\le1\), equality in the polynomial bound is attained by
\(x=y=z=\rho\) (when the equicorrelation triple is lawful), so this is not a
loose triangle estimate in the positive-correlation regime.  It supplies a
fail-fast amplitude ceiling once a lawful correlation envelope is known.
It still does not imply endpoint value.

For covariance-quantization/error propagation, on the same cube

\[
 |\partial_xB_4|,|\partial_yB_4|
 \le2\rho+6\rho^2+8\rho^3,
 \qquad
 |\partial_zB_4|\le8\rho^3. \tag{74.14}
\]

Thus coordinate errors \((\delta x,\delta y,\delta z)\) have the immediate
deterministic first-order/Lipschitz certificate obtained by multiplying these
bounds and \(1/(16\pi)\).

There is also an exact weak-cross-correlation audit which is more relevant
than the coincident-word check.  Set \(z=0\) and expand at \(x=y=0\).
Gaussian Price differentiation gives, for centered \(F(Z)=Z_+-\phi_0\),

\[
 [xy]\,\kappa_4(F_i,F_i,F_j,F_k)
 =E[(F^2)''](EF')^2-2(EF')^4
 =\frac18-\frac1{4\pi}. \tag{74.15}
\]

For \(X=P_{\le2}F=\tfrac12Z+	frac{\phi_0}{2}(Z^2-1)\), the same
calculation gives

\[
 [xy],\kappa_4(X_i,X_i,X_j,X_k)=\frac1{4\pi}.
\]

Therefore the complete affine projection defect begins as

\[
 \boxed{
 \Delta_{i;jk}^{all}
 =\frac{\pi-4}{8\pi},xy+O(x^2y+xy^2+|z|\rho^2),} \tag{74.16}
\]

whereas (74.4) contributes \(-xy/(8\pi)\).  The first H4 band has the
correct sign but overshoots the exact leading magnitude by the factor

\[
 \frac1{4-\pi}\approx1.165; \tag{74.17}
\]

the summed higher-band correction at this order is
\((\pi-3)xy/(8\pi)\), opposite in sign and about 16.5% of the full defect
magnitude.  Unlike (74.7), this is a genuine off-diagonal local fidelity
receipt.  It remains an asymptotic statement, not a panel-wide correlation
certificate when the three correlations are not small.

Because a fitted scalar coefficient absorbs global normalization, use the
calibrated feature

\[
 \widehat\Delta^{[4]}=(4-\pi)\Delta^{[4]}. \tag{74.18}
\]

Equations (74.4) and (74.16) imply

\[
 \Delta^{all}-\widehat\Delta^{[4]}=O(\rho^3)
 \quad\text{when}\quad |x|,|y|,|z|\le\rho\downarrow0, \tag{74.19}
\]

where the zero-bias parity eliminates an order-two \(xz\) or \(yz\) defect.
Thus, under a corpus signal-floor receipt
\(\|xy\|_W\ge c_0\rho^2\), the calibrated band has relative endpoint error
\(O(\rho/c_0)\) and correlation \(1-O(\rho^2/c_0^2)\).  This is the precise
small-correlation reopen: the *direction* becomes asymptotically exact even
though the uncalibrated band overshoots by 16.5%.  Without the signal floor,
source smallness again gives no relative/correlation guarantee.

Equation (74.4) is exact for the first omitted affine band, not for the full
affine rest and not for true-minus-AQ model error.  Let

\[
 s^{tail}=\langle\widetilde\Lambda,
 [\kappa_4(Y^G)-\kappa_4(X^G)]-\Delta^{[4]}\rangle .
\]

The branch is fidelity-safe only with either a direct bound on
\(|s^{tail}|\), a signed higher-band evaluation, or a corpus endpoint error
receipt folded into (66.30).  The diagonal fraction (74.7), Hermite energy,
or pair-rest norm does not lower-bound off-diagonal task correlation.

Graph split:

- `AFFINE_RELU_Q211_H4_BAND`: solved exactly by (74.1)--(74.5);
- `AFFINE_H4_BAND_STATE_ONLY_NOVELTY`: conditional GO only for
  \(C=X^{AQ}\) after complete-old Schur residualization;
- `AFFINE_H4_BAND_WITH_SECTION49`: duplicate/zero credit;
- `AFFINE_H4_BAND_CP_COSTATE_ACTION`: solved exactly by (74.8)--(74.11);
- `AFFINE_H4_BAND_QUERY_RANK_COST`: conditional on a literal rank
  \(R_c\le12\) generic or \(R_c\le25\) symmetric decomposition;
- `AFFINE_H4_BAND_TAIL_FIDELITY`: open blocking receipt;
- `AFFINE_H4_BAND_RAW_VALUE`: open; apply the complete-old robust Schur gate,
  not the diagonal 87.2% normalization.

Current next question: is the D10 positive-depth q211 costate a literal
small-\(R_c\) decomposition at the same reset seam, and is the augmented
Section 49 surface actually in the billed replacement?  The answer decides
whether (74.4) is a cheap new state-only source, an expensive query, or exact
duplicate credit.

## 77. Price-jet provider: the complete small-correlation affine q211 defect
through second order

Section 74 isolates one Hermite band.  For weak off-diagonal correlations it
is better to resum *all* chaos degrees contributing through quadratic order.
Gaussian Price differentiation gives such a provider using only univariate
ReLU data and three covariance matvecs per costate atom.

For unit \(a\), standardize the affine preactivation as

\[
 G_a=\sigma_a(Z_a-\tau_a),\qquad Y_a=(G_a)_+,qquad
 F_a=Y_a-EY_a,
\]

where \(P_a=\overline\Phi(\tau_a)\), \(\varphi_a=\phi(\tau_a)\), and

\[
 \mu_a=EY_a=\sigma_a(\varphi_a-\tau_aP_a). \tag{77.1}
\]

The first two raw Hermite coefficients are

\[
 b_{1a}=E[F_a\operatorname{He}_1(Z_a)]=\sigma_aP_a,
 \qquad
 b_{2a}=E[F_a\operatorname{He}_2(Z_a)]=\sigma_a\varphi_a. \tag{77.2}
\]

Thus the carried projection is

\[
 X_a=P_{\le2}F_a=b_{1a}Z_a+\frac{b_{2a}}2(Z_a^2-1). \tag{77.3}
\]

Define the two *square-response defects*

\[
 d_{1a}=E[(F_a^2)']-E[(X_a^2)'],
 \qquad
 d_{2a}=E[(F_a^2)'']-E[(X_a^2)'']. \tag{77.4}
\]

Distributional differentiation at the ReLU threshold gives the closed
formulas

\[
 \boxed{
 d_{1a}=2\sigma_a\mu_a\Phi(\tau_a)-2b_{1a}b_{2a},} \tag{77.5}
\]

\[
 \boxed{
 d_{2a}=2\sigma_a^2P_a-2\sigma_a\mu_a\varphi_a
        -2(b_{1a}^2+b_{2a}^2).} \tag{77.6}
\]

Indeed \(F_a'=\sigma_a1_{Z_a>\tau_a}\),
\(F_a''=\sigma_a\delta_{\tau_a}\), while
\(X_a'=b_{1a}+b_{2a}Z_a\) and \(X_a''=b_{2a}\).

### 77.1 Exact quadratic Taylor jet of the connected defect

For three distinct labels write

\[
 x=\rho_{ij},\qquad y=\rho_{ik},\qquad z=\rho_{jk}.
\]

At \((x,y,z)=(0,0,0)\), Price's identity applied to
\(\kappa_4(F_i,F_i,F_j,F_k)\) gives the only three possible connected
quadratic monomials:

\[
\begin{aligned}
 [xy]\,\kappa_4(F_i,F_i,F_j,F_k)
 &=[E(F_i^2)''-2(EF_i')^2](EF_j')(EF_k'),\\
 [xz]\,\kappa_4(F_i,F_i,F_j,F_k)
 &=E(F_i^2)'\,(EF_j'')\,(EF_k'),\\
 [yz]\,\kappa_4(F_i,F_i,F_j,F_k)
 &=E(F_i^2)'\,(EF_j')\,(EF_k'').
\end{aligned} \tag{77.7}
\]

The projection \(X\) has the same \(EF'=b_1\) and \(EF''=b_2\), so all
pairing terms cancel in the difference.  Therefore

\[
 \boxed{
\begin{aligned}
 \Delta_{i;jk}^{Price}
 :={}&\kappa_4(F_i,F_i,F_j,F_k)
       -\kappa_4(X_i,X_i,X_j,X_k)\\
 ={}&d_{2i}b_{1j}b_{1k}\,xy
 +d_{1i}b_{2j}b_{1k}\,xz
 +d_{1i}b_{1j}b_{2k}\,yz
 +O(\rho^3),
\end{aligned}} \tag{77.8}
\]

uniformly on a compact threshold/variance set, where
\(\rho=\max(|x|,|y|,|z|)\).  This is the *complete* affine projection defect
through second order, not a single Hermite truncation.

At zero bias and unit variance,

\[
 b_1=\frac12,quad b_2=\phi_0,quad d_1=0,quad
 d_2=\frac12-\frac2\pi=\frac{\pi-4}{2\pi}. \tag{77.9}
\]

Hence (77.8) reduces exactly to Section 74.16:
\((\pi-4)xy/(8\pi)+O(\rho^3)\).  In this sense the factor
\((4-\pi)\) calibration of the H4 band is not empirical; it is the exact
all-chaos Price resummation at quadratic order.

### 77.2 Three-matvec target action

Let one partially symmetric CP costate atom be

\[
 L_i[j,k]=\alpha_i\operatorname{Sym}_{jk}(\beta_j\gamma_k).
\]

Use elementwise products and define

\[
 p_\beta=R(\beta\odot b_1),\qquad
 p_\gamma=R(\gamma\odot b_1),\qquad
 q_\alpha=R(\alpha\odot d_1). \tag{77.10}
\]

Directly contracting (77.8) gives

\[
 \boxed{
\begin{aligned}
 \langle L,\Delta^{Price}\rangle={}&
 \langle\alpha\odot d_2,p_\beta\odot p_\gamma\rangle\\
 &+\langle\beta\odot b_2,q_\alpha\odot p_\gamma\rangle
 +\langle\gamma\odot b_2,q_\alpha\odot p_\beta\rangle .
\end{aligned}} \tag{77.11}
\]

Thus a generic atom requires three \(n\)-dimensional covariance matvecs; a
symmetric atom \(\beta=\gamma\) requires two.  Across ten active seams, the
optimistic cost is

\[
 C_{Price}^{generic}=30R_c/n,qquad
 C_{Price}^{sym}=20R_c/n. \tag{77.12}
\]

At \(n=1024\), the one-equivalent necessary screens are

\[
 \boxed{R_c\le34\quad\text{generic},\qquad R_c\le51\quad\text{symmetric},}
 \tag{77.13}
\]

before reverse-factor construction, hooks, writes, and complete-old
residualization.  Shared \(\beta,\gamma\) dictionaries can reuse matvecs.

### 77.3 Novelty, fidelity, and fail-fast gates

The Price jet is a deterministic function of resident affine means,
variances, thresholds, and covariance.  It is therefore a label-free
deployable source *for the carried-state-only comparator* \(C=X^{AQ}\),
subject to the literal low-rank reverse query in Section 73.  It is already
part of \(\kappa_4(Y^{AQ})\) when the complete Section 49 surface is included
and receives zero additive credit there.  Its biased H3 and centered H4
subbands must not be stacked separately with (77.8).

Let \(J_3\) bound the third derivatives of the complete trivariate defect on
the lawful correlation cube and compact threshold/variance set.  Taylor's
theorem gives the explicit form of the remaining fidelity obligation,

\[
 |\Delta^{all}-\Delta^{Price}|
 \le \frac{J_3}{6}(|x|+|y|+|z|)^3. \tag{77.14}
\]

For translating this into a correlation gate, if the corpus endpoint feature
\(f\) and its Price approximation \(\widehat f\) satisfy

\[
 \|f-\widehat f\|_W\le\epsilon\|f\|_W,
\]

then the sharp worst-case angle bound is

\[
 \operatorname{corr}_W(f,\widehat f)\ge\sqrt{1-\epsilon^2}. \tag{77.14a}
\]

Equality is attained by an error with parallel component \(-\epsilon^2 f\)
and orthogonal component of norm \(\epsilon\sqrt{1-\epsilon^2}\|f\|\).
Thus the campaign correlation landmarks 0.9325, 0.98346, and 0.9972 require
respectively

\[
 \epsilon\le0.3612,qquad0.1811,qquad0.0748. \tag{77.14b}
\]

These are endpoint-relative error gates; (77.14) supplies them only after a
nonzero norm/action floor converts its absolute remainder into \(\epsilon\).

The first generic arithmetic obstruction inside this cubic remainder is
already explicit.  Section 72.10's centered \((4,2,2,2)\) triangle is

\[
 b_{i4}b_{i2}b_{j2}b_{k2}\,xyz,
\]

and its CP-query action is the three-factor trace (72.20), requiring one
query-specific dense product per seam absent literal fusion.  Across ten
seams that alone exceeds the one-equivalent slack.  Thus (77.8) is not only
the complete quadratic jet; it is the last universally matvec-native
correlation order before a generic dense triangle appears.  The rooted
\(f=0\) resummation in Section 78 is the only route to extend it without
paying that triangle, and must treat the discarded \(f\ge1\) action as error.

At zero bias/unit scale the triangle coefficient is
\(-\phi_0^4xyz=-xyz/(4\pi^2)\).  Relative to the leading Price coefficient
\((\pi-4)xy/(8\pi)\), its pointwise magnitude ratio is exactly

\[
 \boxed{
 {2|z|\over\pi(4-\pi)}\approx0.7415|z|.} \tag{77.14c}
\]

This quantifies the first broad-rank omission: it is automatically below
7.42% of the leading word when \(|\rho_{jk}|\le0.1\).  Other cubic and
higher diagrams still belong in the total remainder, so (77.14c) is not by
itself a full-tail bound.

The provider is a fail-fast GO only when:

1. the augmented Section 49 surface is absent from the declared bill;
2. the complete-old-projected D10 costate passes the \(R_c\) and invariant-
   reverse gates of Sections 73 and 75;
3. a correlation envelope plus nonzero feature/action floor turns (77.14)
   into the required endpoint-correlation certificate;
4. the robust Schur lower bound (66.30), with (77.14) included in its error,
   clears the raw-MSE and adjusted-cost fire thresholds.

Graph split:

- `AFFINE_Q211_PRICE_JET`: solved exactly through quadratic correlation
  order by (77.5)--(77.8);
- `AFFINE_Q211_PRICE_JET_CP_ACTION`: solved by (77.10)--(77.13);
- `AFFINE_Q211_PRICE_JET_STATE_ONLY_NOVELTY`: conditional GO for
  \(C=X^{AQ}\);
- `AFFINE_Q211_PRICE_JET_WITH_SECTION49`: duplicate/zero credit;
- `AFFINE_Q211_PRICE_JET_REMAINDER`: conditional on the compact-set third-
  derivative bound and correlation envelope;
- `AFFINE_Q211_PRICE_JET_OLD_SCHUR_VALUE`: open blocking receipt.

This is currently the sharpest analytic low-cost affine q211 provider in the
cubic lane: it resums the leading signed defect rather than guessing a
single chaos band, while preserving the exact ownership split.

### 77.4 Exact zero-bias pairing/trace decomposition

The complete-old audit simplifies further at zero bias.  Let \(C\) be the
unnormalized covariance, \(v_i=C_{ii}\), and

\[
 c_0={\pi-4\over8\pi}.
\]

Positive homogeneity turns (77.8) into

\[
 \Delta_{i;jk}^{Price}=c_0 C_{ij}C_{ik}. \tag{77.15}
\]

Let \(P_C=\operatorname{Sym}_{pair}(C\otimes C)\) be the ordinary covariance
pairing tensor, and let

\[
 Z_{v,C}=\operatorname{Sym}_{pair}(\operatorname{Diag}(v)\otimes C).
\]

On every strict q211 coordinate,

\[
 (P_C)_{iijk}=v_iC_{jk}+2C_{ij}C_{ik},
 \qquad
 (Z_{v,C})_{iijk}=v_iC_{jk}.
\]

Therefore

\[
 \boxed{
 \Pi_{211,off}\Delta^{Price}
 ={c_0\over2}\Pi_{211,off}(P_C-Z_{v,C}).} \tag{77.16}
\]

This is an exact ownership test, not an approximation.  If the literal old
interface owns both the full covariance-pairing tensor and the synchronized
variance-trace tensor \(Z_{v,C}\), the zero-bias Price jet has zero Schur
novelty.  Owning one global K22 pairing scalar or the visible K31/K22 faces
does not by itself prove ownership of \(Z_{v,C}\) on the off-diagonal q211
consumer.  If only \(P_C\) is old-owned, the remaining trace complement is
fixed exactly by (77.16); Section 80 audits whether D11/Q24/Star4 actually
contains that row.

## 75. Maximal structured reopen: block-monomial Bellman algebras and the
two-operator obstruction

The generic theorem in Section 73 does not exclude exact symmetry classes.
This section classifies the strongest such escape.  Diagonal, Walsh,
block-Schur, simultaneous-spectral, and finite-Krylov proposals are all
instances of one layerwise *block-monomial congruence algebra*.  The same
classification also shows why a separating reset and one genuinely mixing
transition force the state back to full width.

### 75.1 Necessary and sufficient invariant-algebra conditions

At each reverse seam choose a vector support
\(\mathcal S_\ell=\operatorname{col}U_\ell\) and a symmetric matrix space
\(\mathcal B_\ell\subseteq\operatorname{Sym}(\mathcal S_\ell)\).  The
factorized q211 class is closed under the actual D10 pullback (73.14) if and
only if, for every active anchor leg,

\[
 \boxed{
 B_{\ell i}^T\mathcal S_{\ell+1}\subseteq\mathcal S_\ell,
 \qquad
 R_{\ell i}\mathcal B_{\ell+1}R_{\ell i}^T
 \subseteq\mathcal B_\ell,}
 \tag{75.1}
\]

where

\[
 B_{\ell i}^TU_{\ell+1}=U_\ell R_{\ell i}.
 \tag{75.2}
\]

These are exactly the support and anchor-space conditions (73.16)--(73.18),
now imposed as a layerwise algebra rather than checked after a dense object
has been formed.  Anchor mixing changes only the coefficient vectors when
(75.1) holds.

There are two further maps at the literal q211 seam.  Exact cheap closure
also requires

\[
 \boxed{
 \Pi_{off,i}^*(U_\ell\mathcal B_\ell U_\ell^T)
 \subseteq U_\ell\mathcal B_\ell U_\ell^T+\mathcal O_{coll,\ell},
 \qquad
 \mathcal R_{old,\ell}^*(U_\ell\mathcal B_\ell U_\ell^T)
 \subseteq U_\ell\mathcal B_\ell U_\ell^T,}
 \tag{75.3}
\]

where \(\mathcal O_{coll}\) is the *literally resident* K4/K31/K22 collision
space.  Thus a raw factorized representative may be consumed cheaply when
its collision faces are sewn by old scalar reads.  Applying every
anchor-specific strict mask first usually destroys the algebra: a unital
Schur space invariant under all coordinate deletions contains every
coordinate indicator and is the full coordinate algebra.  Collision sewing
is therefore a structural condition, not a cosmetic implementation detail.

### 75.2 Exact block-monomial classification

Let a \(k\)-dimensional support be decomposed into internal blocks

\[
 E_1\oplus\cdots\oplus E_s,qquad \dim E_a=b_a,qquad
 \sum_ab_a=k,
 \tag{75.4}
\]

and define

\[
 \mathcal B_{\mathbf b}
 :=\operatorname{Sym}(E_1)\oplus\cdots\oplus
   \operatorname{Sym}(E_s),qquad
 p_{\mathbf b}=\sum_a{b_a(b_a+1)\over2}.
 \tag{75.5}
\]

For domain blocks \(E_a^+\) and codomain blocks \(E_c^-\), the congruence
condition

\[
 R\mathcal B_{\mathbf b^+}R^T
 \subseteq\mathcal B_{\mathbf b^-}
 \tag{75.6}
\]

holds if and only if every input block is mapped wholly into one output
block:

\[
 \boxed{R E_a^+\subseteq E_{\pi(a)}^-\quad\hbox{for every }a.}
 \tag{75.7}
\]

To prove necessity, apply (75.6) to every rank-one \(xx^T\) with
\(x\in E_a^+\).  The block-diagonal rank-one matrix
\((Rx)(Rx)^T\) can occupy only one output block; linearity then puts all of
\(RE_a^+\) in the same block.  Sufficiency is immediate.  For a nonsingular
two-sided recurrence with matched block dimensions, (75.7) is precisely a
block permutation with arbitrary invertible within-block maps.

The diagonal case \(b_a=1\) is especially sharp:

\[
 R\operatorname{Diag}(k)R^T\subseteq\operatorname{Diag}(k)
 \quad\Longleftrightarrow\quad
 R\text{ has at most one nonzero entry per column}. 
 \tag{75.8}
\]

For nonsingular \(R\), this says exactly that \(R\) is monomial.  Allowing
layer gauges \(S_\ell\), the necessary-and-sufficient simultaneous-spectral
condition is therefore

\[
 \boxed{
 S_\ell^{-1}R_{\ell i}S_{\ell+1}
 \text{ is monomial for every active }i.}
 \tag{75.9}
\]

Commuting, simultaneously diagonalizable restricted legs are one sufficient
special case.  Monomial legs need not commute, so (75.9) is strictly the
larger exact class.  Replacing scalar blocks by \(b_a>1\) gives the maximal
simultaneous block-spectral class (75.7).

For a coefficient family spanning all of (75.5), the Section 73 ranks are

\[
 \boxed{
 k_*=k,qquad p_*=p_{\mathbf b},qquad R_*=p_{\mathbf b}.}
 \tag{75.10}
\]

The last equality uses the canonical partially symmetric basis: \(k\)
diagonal atoms \(e_ae_a^T\) and \(p_{\mathbf b}-k\) within-block atoms
\(\operatorname{Sym}(e_ae_b^T)\).  If the realized endpoint occupies only
a proper subspace of this basis, replace \(p_{\mathbf b}\) by its exact
anchor-flattening rank and \(R_*\) by its exact partially symmetric CP rank.

### 75.3 Coordinate-Schur and Walsh are the same escape in different gauges

Every unital subalgebra of \(\mathbb R^n\) under entrywise multiplication is
the space of vectors constant on the cells of a partition
\(\mathcal P=\{C_1,\ldots,C_k\}\).  Let \(U_{\mathcal P}\) be its normalized
block-indicator basis.  A diagonal ReLU/reset profile preserves this vector
space exactly if and only if the profile is block-constant.  A dense
transition preserves it exactly when

\[
 W_\ell^TU_{\mathcal P_{\ell+1}}
 =U_{\mathcal P_\ell}T_\ell.
 \tag{75.11}
\]

For the *diagonal pair algebra* to remain cheap, mere subspace preservation
is not enough: after the layer gauges in (75.9), every restricted
\(T_\ell\) and reset multiplier must be monomial.  If \(T_\ell\) is a
generic dense \(k\times k\) matrix, the vector support stays at \(k\) but
the pair algebra expands to \(\operatorname{Sym}(k)\), with
\(p=k(k+1)/2\).

At \(n=2^d\), the span of a subgroup of \(k\) Walsh characters is exactly a
block-constant algebra on the cosets of its annihilator.  Multiplication by
a gate in that algebra is convolution in the character basis and diagonal
in the primitive block-indicator basis.  Hence a Walsh proposal has

\[
 k_*=p_*=R_*=k
 \tag{75.12}
\]

only if every transition normalizes the primitive rays as in (75.9).
“Walsh coordinates” without subgroup closure, block-constant gates, and a
monomial restricted transition do not supply an invariant q211 algebra.

There is also an exact actual-law symmetry class.  If the input law is
invariant under within-cell permutations, every weight/bias map intertwines
those permutation actions, ReLU is applied coordinatewise, and the endpoint
metric plus old projector obey the same symmetry, then the full non-Gaussian
activation law remains block-exchangeable and its reverse costate remains in
the corresponding orbit algebra at every depth.  This is a lawful
non-generic true-law class, not merely a Gaussian control variate.  Generic
dense weights break the intertwining identity; exchangeability of the input
alone is insufficient.

### 75.4 Finite Krylov/projective-orbit exception for one endpoint

A universal algebra is stronger than needed for one frozen low-rank head.
Let \(\mathcal V_L\) contain the terminal pair rays and recursively define

\[
 \mathcal V_\ell
 =\{B_{\ell i}^Tv:v\in\mathcal V_{\ell+1}, i\text{ active}\}.
 \tag{75.13}
\]

For a symmetric-ray endpoint, the exact realized ranks are

\[
 \boxed{
 k_\ell=\dim\operatorname{span}\mathcal V_\ell,qquad
 p_\ell=\dim\operatorname{span}\{vv^T:v\in\mathcal V_\ell\},}
 \tag{75.14}
\]

after applying the actual anchor mixing and deleting zero coefficient
directions.  For general pair atoms, replace \(vv^T\) by
\(\operatorname{Sym}(vw^T)\) and propagate both rays.  Equation (75.14) is
the exact finite-depth block-Krylov certificate; no ambient SVD claim is
needed.

The strongest finite-state exception is a projective dictionary
\(v_1,\ldots,v_R\) satisfying

\[
 B_{\ell i}^Tv_q=s_{\ell iq}v_{\pi_{\ell i}(q)}.
 \tag{75.15}
\]

Then the ray count never grows, \(k\le R\), \(p\le R\), and a generic
coefficient family on the dictionary has \(p=R\).  This is precisely the
monomial algebra restricted to the endpoint orbit.  If even one leg sends a
ray outside the finite projective dictionary, the CP state branches; a
post-hoc fit back to \(R\) is approximate and must use (73.25)--(73.26).

### 75.5 Exact Section 74 H4-band cost inside the invariant classes

For the full block algebra (75.5), use \(k\) symmetric diagonal atoms and
\(p-k\) generic off-diagonal atoms in (74.10)--(74.11).  Across ten active
seams, the exact optimistic matvec-equivalent census is

\[
 \boxed{
 C_{74}(\mathbf b)
 ={40k+80(p_{\mathbf b}-k)\over n}
 ={80p_{\mathbf b}-40k\over n}.}
 \tag{75.16}
\]

Thus the one-equivalent source-action screen is

\[
 80p_{\mathbf b}-40k\le n.
 \tag{75.17}
\]

For diagonal/simultaneous-spectral/Walsh closure, \(p=R=k\) and

\[
 \boxed{C_{74}=40k/n,\qquad k\le25\text{ at }n=1024.}
 \tag{75.18}
\]

This is the Section 74 action only.  Monomial congruences make coefficient
reset local, but a dense anchor mixer still applies to each of the \(p\)
coefficient channels.  Its ten-seam optimistic bill is

\[
 C_{anchor}\ge10p/n. 
 \tag{75.19}
\]

If the layer support basis must itself be transported rather than supplied
by a literal partition/spectral gauge, add at least \(10k/n\).  Therefore a
diagonal class costs optimistically \(50k/n\) including anchor recurrence,
or \(60k/n\) including new support transports, giving one-equivalent screens
\(k\le20\) or \(k\le17\), respectively.  For block-monomial maps with dense
within-block transforms, the congruences add

\[
 O\!\left({10\sum_ab_a^3\over n^2}\right)
 \tag{75.20}
\]

beyond (75.19).  Hooks, actual-law acquisition, collision sewing, and writes
remain separate.

Strict collision masking can erase this advantage.  The exact cost formulas
apply either to the already-compiled \(L_i\) when it remains in the algebra,
or to a raw factorized representative satisfying the sewing condition
(75.3).  A coarse block/Walsh state followed by all per-anchor coordinate
deletions refines to the singleton partition and returns to \(k=n\).

### 75.6 Deterministic two-operator obstruction

Noncommutation alone is not enough: two block-monomial matrices can fail to
commute while preserving a cheap algebra.  The exact obstruction is
*separation plus mixing*.

Let

\[
 D=\operatorname{Diag}(d_1,\ldots,d_k)
 \tag{75.21}
\]

have distinct eigenvalues, and let the directed graph of \(T\) (edge
\(b\to a\) when \(T_{ab}\ne0\)) be strongly connected.  Every
\(D\)-invariant subspace is a coordinate span, because the coordinate
spectral projectors are polynomials in \(D\).  Strong connectivity leaves
no nonempty proper coordinate span invariant under \(T\).  Hence

\[
 \boxed{
 \mathcal S\ne0, D\mathcal S\subseteq\mathcal S,
 T\mathcal S\subseteq\mathcal S
 \quad\Longrightarrow\quad \mathcal S=\mathbb R^k.}
 \tag{75.22}
\]

Equivalently, if all \(T_{ab}\ne0\), then

\[
 E_{aa}TE_{bb}=T_{ab}E_{ab}
 \tag{75.23}
\]

and the algebra generated by \(D,T\) contains every matrix unit.  This is a
constructive Burnside theorem: \(\operatorname{alg}(D,T)=M_k\).

There is a matching pair-space result.  Assume additionally that the
unordered products \(d_ad_b\), \(a\le b\), are all distinct and one column
of \(T\) has no zero entry.  Congruence by \(D\) has the symmetric matrix
units as distinct eigenvectors.  Starting with the corresponding diagonal
unit \(E_{qq}\), congruence by \(T\) gives

\[
 TE_{qq}T^T=T_{:q}T_{:q}^T,
 \tag{75.24}
\]

which has a nonzero component in every symmetric matrix-unit direction.
Polynomial spectral projectors of \(X\mapsto DXD^T\) isolate all those
directions.  Therefore the smallest nonzero congruence space containing the
diagonal endpoint class and closed under \(D,T\) is

\[
 \boxed{\operatorname{Sym}(k),qquad p=k(k+1)/2.}
 \tag{75.25}
\]

This applies directly to the typed D10 legs.  If
\(B_i=D_iH\) with nonsingular diagonal gates, two common-support
intertwiners imply

\[
 (B_j^T)^{-1}B_i^TU
 =D_j^{-1}D_iU
 =U(R_j^{-1}R_i).
 \tag{75.26}
\]

Thus the next-seam support must be invariant under every relative reset
diagonal \(D_j^{-1}D_i\).  A ratio with simple spectrum forces a coordinate
support.  At the neighboring seam, a genuinely dense \(H^T\) sends every
nonzero coordinate ray to a full-support vector, while the next separating
relative reset again requires a coordinate support.  No nonzero proper
support survives; \(k=n\).  If the relative products are multiplicatively
simple, the pair algebra simultaneously expands to (75.25).

The only exact escape from this deterministic obstruction is visible in its
premises: repeated reset signatures create blocks, and every transition must
map/permute those blocks or a finite endpoint dictionary exactly.  Those are
precisely the block-Schur/Walsh/simultaneous-spectral/Krylov classes above.
Generic dense transitions and unit-distinguishing ReLU profiles satisfy the
opposite conditions and therefore admit no small exact actual-law or Bellman
algebra.

Atomic graph split:

- `D10_Q211_INVARIANT_ALGEBRA_CRITERION`: **solved necessary and sufficient**
  by (75.1)--(75.3);
- `D10_Q211_BLOCK_MONOMIAL_CLASSIFICATION`: **solved exact** by
  (75.4)--(75.10);
- `D10_Q211_SCHUR_WALSH_ESCAPE`: **conditional exact GO** only with
  block-constant gates, primitive-ray-normalizing transitions, and collision
  sewing;
- `D10_Q211_ACTUAL_LAW_BLOCK_SYMMETRY`: **conditional exact GO** for a
  block-equivariant network and invariant input law;
- `D10_Q211_FINITE_KRYLOV_ENDPOINT`: **conditional exact GO**, with ranks
  certified by (75.13)--(75.15);
- `D10_Q211_H4_INVARIANT_ACTION_COST`: **solved exact screen** by
  (75.16)--(75.20);
- `D10_Q211_STRICT_MASK_BLOCK_BREAK`: **closed obstruction** unless (75.3)
  supplies literal collision sewing;
- `D10_Q211_TWO_OPERATOR_SUPPORT_NOGO`: **closed deterministic theorem** by
  (75.21)--(75.23) and the D10 specialization (75.26);
- `D10_Q211_TWO_OPERATOR_PAIRSPACE_NOGO`: **closed deterministic theorem**
  by (75.24)--(75.25);
- `D10_Q211_ACTUAL_INVARIANT_RECEIPT`: **open for deployed networks**; must
  exhibit repeated reset blocks plus exact transition normalization at every
  active seam, not approximate spectral clustering.

## 72. Affine-Gaussian ReLU rest q211: exact connected diagrams, rank
obstruction, finite-band ownership, and target-scalar gates

This section treats a narrower premise than Sections 68--70: the preactivation
is *literally multivariate Gaussian*.  Under that premise the connected ReLU
rest is determined label-free by carried means, variances, and correlations.
The missing issue is not statistical identifiability; it is whether the
degree-collapsed D10 interface or a small carrier/query can evaluate the
result.

### 72.1 Exact Hermite and connected-diagram formula

Write

\[
 G_i=m_i+\sigma_iX_i,
 \qquad E X_iX_j=\rho_{ij},
 \qquad \tau_i=-m_i/\sigma_i,
\]

and (Y_i=(G_i)_+).  With probabilists' Hermites,

\[
 Y_i=\sum_{q\ge0}a_{iq}\operatorname{He}_q(X_i),
 \qquad b_{iq}:=q!a_{iq}.
\tag{72.1}
\]

Stein integration by parts gives

\[
 b_{i1}=\sigma_i\overline\Phi(\tau_i),
 \qquad
 b_{iq}=\sigma_i\phi(\tau_i)
 \operatorname{He}_{q-2}(\tau_i),\quad q\ge2.
\tag{72.2}
\]

The constant coefficient is irrelevant to cumulants.  At zero bias,
(b_{i1}=\sigma_i/2), all odd (b_{iq}) for (q\ge3) vanish, and

\[
 b_{i,2s}=\sigma_i\phi(0)(-1)^{s-1}(2s-3)!!.
\tag{72.3}
\]

For four labeled slots (v=1,\ldots,4), let
(\mathcal C(q_1,\ldots,q_4)) be the connected loop-free multigraphs with
edge multiplicities (m_{uv}\ge0) and vertex degrees
(\sum_{u\ne v}m_{uv}=q_v).  The connected Wick-diagram theorem is

\[
 \kappa_4\{\operatorname{He}_{q_1}(X_{i_1}),\ldots,
 \operatorname{He}_{q_4}(X_{i_4})\}
 =\left(\prod_vq_v!\right)
 \sum_{M\in\mathcal C(q)}
 \prod_{u<v}{\rho_{i_ui_v}^{m_{uv}}\over m_{uv}!}.
\tag{72.4}
\]

Let (Y^{(2)}=P_{\le2}Y).  Multilinearity and (72.4) give the exact missing
connected source

\[
\boxed{
 \Delta T_{i_1i_2i_3i_4}
 =\sum_{\substack{q_v\ge1\\\max_vq_v\ge3}}
 \left(\prod_vb_{i_vq_v}\right)
 \sum_{M\in\mathcal C(q)}
 \prod_{u<v}{\rho_{i_ui_v}^{m_{uv}}\over m_{uv}!}.}
\tag{72.5}
\]

For q211 slots ((i,i,j,k)), name the six edge counts

\[
 m_{12}=a,\ m_{13}=b,\ m_{14}=c,\
 m_{23}=d,\ m_{24}=e,\ m_{34}=f.
\]

Since (\rho_{ii}=1), each term in (72.5) is

\[
 {\rho_{ij}^{b+d}\rho_{ik}^{c+e}\rho_{jk}^{f}
  \over a!b!c!d!e!f!},
\tag{72.6}
\]

subject to connectedness and the four degree equations.  Equations
(72.2), (72.5), and (72.6) are a complete state-only acquisition formula
under the literal Gaussian premise.

### 72.2 First biased and centered bands

For nonzero bias, the first omitted degree pattern is ((3,1,1,1)).  Its
connected graph is the unique star centered at the H3 vertex.  With
(x=\rho_{ij}), (y=\rho_{ik}), and (z=\rho_{jk}), the complete q211 band is

\[
\boxed{\begin{aligned}
 \Delta^{[3]}_{i;jk}={}&
 2b_{i3}b_{i1}b_{j1}b_{k1}xy\\
 &+b_{j3}b_{i1}^2b_{k1}x^2z
  +b_{k3}b_{i1}^2b_{j1}y^2z.
\end{aligned}}
\tag{72.7}
\]

Define (v_{a,p}=b_{p1}\rho_{ap}).  Globally this is the q211 read of the
finite star family

\[
 \sum_a b_{a3}\operatorname{Sym}(e_a,v_a,v_a,v_a),
\tag{72.8}
\]

with the repeated-slot multiplicities in (72.7).  It is label-free and uses
only the signed D10 H3 coefficient plus (R).

At zero bias, (b_3=0).  The first omitted pattern is ((4,2,1,1)).  For an
ordered H4 vertex (h), H2 vertex (t), and leaves (u,v), the unique connected
diagram has a double ((h,t)) edge and one edge from (h) to each leaf:

\[
 {1\over2}b_{h4}b_{t2}b_{u1}b_{v1}
 \rho_{ht}^2\rho_{hu}\rho_{hv}.
\tag{72.9}
\]

Summing the twelve assignments gives exactly the polynomial in
(74.4)--(74.5).  Thus Section 74 is the complete lowest-total-degree centered
band, not an approximation within that degree pattern.

It is not the complete one-H4 response.  The connected
((4,2,2,2)) sector already contains the diagram

\[
 a=2,\quad b=c=f=1,\quad d=e=0,
\]

and the two choices of which repeated (i)-slot owns H4 give the exact
triangle subband

\[
 \boxed{
 \Delta^{tri}_{i;jk}
 =b_{i4}b_{i2}b_{j2}b_{k2}
 \rho_{ij}\rho_{ik}\rho_{jk}.}
\tag{72.10}
\]

This term is inside D10's nominal degree-4 availability but outside the
first band (74.4).

### 72.3 Rooted low-rank diagrams versus the sharp broad-rank obstruction

For fixed (i), every diagram term has the singleton-matrix form

\[
 D_{x_i}\,R^{\odot f}\,D_{y_i}
\tag{72.11}
\]

or its transpose, where (f=m_{34}) counts edges directly joining the
singleton labels ((j,k)).  Terms with (f=0) are outer products and are
native-low-rank after grouping equal degree patterns.  These are the rooted
or no-singleton-edge sector.

Terms with (f\ge1) are generically broad.  If the singleton correlation
submatrix is positive definite, every integer Schur power
(R^{\odot f}) is positive definite; nonzero diagonal scalings preserve its
rank.  Symmetrized sums are generically nonsingular.  Bias creates this
obstruction already in (72.7): its last two terms are

\[
 b_{i1}^2\{D_{x_i}RD_{b_1}+D_{b_1}RD_{x_i}\},
 \qquad (x_i)_j=b_{j3}\rho_{ij}^2.
\tag{72.12}
\]

In the centered first band, (74.8) contains

\[
 D_{r_i^{\odot2}}R^{\odot2}
 +(R^{\odot2})D_{r_i^{\odot2}}.
\tag{72.13}
\]

Choose a valid covariance with small equal anchor correlations and a
positive-definite singleton block.  The leading matrix is a positive
multiple of ((R^{\odot2}+\mathbf1\mathbf1^T)), hence nonsingular.  Its
determinant is therefore not the zero polynomial, proving generic full rank.
The stricter triangle band (72.10) is the congruence

\[
 b_{i4}b_{i2}
 D_{b_2\odot r_i}R D_{b_2\odot r_i},
\tag{72.14}
\]

which has exactly the rank of (R) when its diagonal factors are nonzero.

Collision masking removes a diagonal but does not supply a low-rank theorem.
Even granting arbitrary q22 diagonal completion, the generic rank remains
(m-O(\sqrt m)) by (66.8bg)--(66.8bh).  Consequently:

- the rooted (f=0) truncation may have a small native carrier;
- the complete affine-Gaussian rest has no generic small native or
  anchor-matrix carrier;
- a small forward claim requires a task-visible complete-old spectral tail,
  not pair-rest energy.

### 72.4 What D10 covariance rest and finite CP4/Q24 bands do and do not own

The D10 two-point higher-chaos covariance is

\[
 K_{uv}^{G,>2}
 =\sum_{q\ge3}{b_{uq}b_{vq}\over q!}\rho_{uv}^q.
\tag{72.15}
\]

It contains only two-vertex, equal-degree diagrams and collapses the degree
index.  The connected source (72.5) contains unequal degrees and four-vertex
topologies.  For example, neither ((3,1,1,1)) nor ((4,2,1,1)) is a datum in
(72.15), and the triangle (72.10) is not a pair covariance.  Hence the
collapsed rest matrix alone is not an exact source port.

Under the literal Gaussian premise, however, no new stochastic information
is needed: ((m,\sigma,R)) and the known coefficients (72.2) determine every
diagram.  Degree-resolved D10 bands 3--6 plus the carried H1/H2 pieces are
sufficient to *form the corresponding finite diagram truncation* if their
signed coefficients and the required connected contractions are exposed.
Naming the bands, or carrying their pair covariances, does not perform those
contractions.  Because ReLU has infinitely many nonzero Hermite coefficients,
no finite band is exact for generic (R).

The full Section 49 affine/AQ ReLU surface already includes (72.5) for its
Gaussian comparator and query.  In that augmented branch, adding (72.7),
(74.4), or any higher affine diagram is duplicate credit.  They are new only
in the state-only comparator branch (C=P_{\le2}Y), after the literal
collision and complete-old residualizers.

## 71. Exact strict-q211 ReLU reset, native-rank gate, and the global
squarefree quotient certificate

This section closes two gaps left by Sections 47, 61, and 66.  First it gives
the exact first-order coordinatewise-ReLU map on a strict q211 perturbation,
including the three possible repeated-anchor orientations.  Second it records
the global quotient-rank theorem needed to distinguish a genuinely small CP4
carrier from a low-rank commutator Gram that does not have lawful CP factors.

### 71.1 Exact one-coordinate pushed scores

Let G be standard normal, let

\[
 Y_t=(G-t)_+,
 \qquad Q_t=\Pr(G>t),\qquad P_t=1-Q_t,
 \qquad \varphi_t={e^{-t^2/2}\over\sqrt{2\pi}},
\]

and put

\[
 \mu_t=EY_t=\varphi_t-tQ_t,
 \qquad h_t=Y_t-\mu_t.
\tag{71.1}
\]

The score of the pushed law is the conditional expectation of the Gaussian
Hermite score.  At the ReLU atom and on its active branch it is exactly

\[
 E[H_1(G)\mid Y_t=y]
 =\begin{cases}
 y+t,&y>0,\\
 -\varphi_t/P_t,&y=0,
 \end{cases}
\tag{71.2}
\]

\[
 E[H_2(G)\mid Y_t=y]
 =\begin{cases}
 (y+t)^2-1,&y>0,\\
 -t\varphi_t/P_t,&y=0.
 \end{cases}
\tag{71.3}
\]

Consequently, at an independent Gaussian base, the strict input score
(H_2(G_i)H_1(G_j)H_1(G_k)) remains the product of one pushed H2 score and
two pushed H1 scores.  This is exact score-level nativity; it must not be
confused with rank one after re-expansion in output cumulant coordinates.

For the latter re-expansion define the four scalar response coefficients

\[
 B_{1}(t)=E[h_tH_1(G)]=Q_t,
 \qquad
 B_{2}(t)={1\over2}E[h_tH_2(G)]={\varphi_t\over2},
\tag{71.4}
\]

\[
 A_{1}(t)=E[h_t^2H_1(G)]=2\mu_tP_t,
 \qquad
 A_{2}(t)={1\over2}E[h_t^2H_2(G)]
 =Q_t-\mu_t\varphi_t.
\tag{71.5}
\]

For a scaled ReLU ((\sigma (G-t)_+)), the B coefficients acquire one factor
of (\sigma) and the A coefficients acquire (\sigma^2).  Centering and any
subsequent nonzero output rescaling do not alter the gate ratios below.
The identities in (71.4)--(71.5) follow from

\[
 E[(G-t)_+H_2(G)]=\varphi_t,
 \qquad E[(G-t)_+^2H_2(G)]=2Q_t,
\]

and are not a mean-slope approximation.

### 71.2 Complete independent reset on the three q211 orientations

For one unordered triple (i,j,k), write

\[
 q=(q_{i;jk},q_{j;ik},q_{k;ij})^T.
\]

With the conventional Gram--Charlier normalization, its density perturbation
is one half of

\[
 q_{i;jk}H_{2,i}H_{1,j}H_{1,k}
 +q_{j;ik}H_{1,i}H_{2,j}H_{1,k}
 +q_{k;ij}H_{1,i}H_{1,j}H_{2,k}.
\tag{71.6}
\]

The exact output q211 vector is (q'=R^{ijk}q), where

\[
\boxed{
 R^{ijk}_{r r}=A_{2,r}\!\prod_{s\in\{i,j,k\}\setminus\{r\}}B_{1,s},
 \qquad
 R^{ijk}_{r a}=A_{1,r}B_{2,a}B_{1,b}\quad(r\ne a),}
\tag{71.7}
\]

and b is the third label.  Thus the reset is block diagonal over unordered
triples but is not diagonal over the three choices of repeated anchor.

There is a useful exact diagonal-plus-rank-one form.  Put

\[
 a_r={A_{2,r}\over B_{1,r}},\qquad
 b_r={\mu_rP_r\over Q_r},\qquad
 c_r={\varphi_r\over Q_r},\qquad
 \delta_r=a_r-b_rc_r
 =1-{\mu_r\varphi_r\over Q_r^2}.
\tag{71.8}
\]

Then

\[
\boxed{
 R^{ijk}=Q_iQ_jQ_k
 \{\operatorname{Diag}(\delta_i,\delta_j,\delta_k)
 +(b_i,b_j,b_k)^T(c_i,c_j,c_k)\}.}
\tag{71.9}
\]

At the unbiased ReLU,
((Q=P=1/2)) and
((\mu=\varphi=1/\sqrt{2\pi})), and

\[
 R^{ijk}={1\over8}
 \left\{\left(1-{2\over\pi}\right)I_3
 +{1\over\pi}{\bf1}{\bf1}^T\right\}.
\tag{71.10}
\]

Its two contrast eigenvalues are ((1-2/pi)/8) and its symmetric eigenvalue
is ((1+1/pi)/8).  Hence the literal unbiased reset has orientation rank
three.  A universal strict-q211 provider cannot replace it by one mean gate,
although one specially chosen input vector can of course land in one output
direction.

The fourth-cumulant subtraction introduces no missing terms in (71.7).
Every lower-order moment observable omits at least one of the three distinct
coordinates, whereas each strict input derivative contains all three.
Its fourth derivative is therefore zero.  Means are harmless because
cumulants of order at least two are translation invariant.

### 71.3 The exact same-anchor gate ratio and when a native word survives

If only the same-anchor branch of (71.7) is live after the complete-old
projection, its Schur kernel is

\[
 K_{i;jk}=A_{2,i}B_{1,j}B_{1,k}.
\tag{71.11}
\]

Let (D=Diag(B_{1,i})) and define the scale-invariant anchor ratio

\[
 \boxed{
 r_i={A_{2,i}\over B_{1,i}^2}
 ={Q_i-\mu_i\varphi_i\over Q_i^2}.}
\tag{71.12}
\]

For a zero-bias ReLU this is the common constant

\[
 r_0={2(\pi-1)\over\pi}.
\tag{71.13}
\]

Unequal positive preactivation scales do not change (r_i), by ReLU
homogeneity.  Standardized thresholds do.  If (r_i=r) on the live support,
then for every native three-generator word

\[
 K\odot P_{211}^{ext}\operatorname{Sym}(A,A,B,C)
 =rP_{211}^{ext}\operatorname{Sym}(DA,DA,DB,DC).
\tag{71.14}
\]

Thus constant (r) is the exact one-common-leg-map closure condition for the
same-anchor channel.  Conversely, it is necessary for closure of every
full-support native word.  It suffices to test the degenerate native word
with (A=B=C=v).  Its q211 coefficient is
(v_i^2v_jv_k).  After reset it is

\[
 r_iw_i^2w_jw_k,\qquad w=Dv.
\tag{71.15}
\]

If this equals one CP quotient atom (u_i^2u_ju_k) on at least four nonzero
coordinates, ratios on overlapping triples first force (u_i) proportional
to (r_iw_i), and then force (r_ir_jr_k) to be constant for every triple.
Comparing two triples that differ in one label gives (r_i=r_j) for all live
labels.  Hence a nonconstant anchor ratio gives an analytic rank-growth
counterexample: one coupled native ray requires at least two quotient rays.
The same conclusion holds for a Zariski-generic full-support
(Sym(A,A,B,C)) word; special support-three words are the expected exception,
because one triple has no overlapping-triple compatibility condition.

Equation (71.14) concerns the same-anchor branch.  The complete reset also
contains the two cross-anchor columns in (71.7).  On the literal abstract
three-coordinate alphabet ((A,B,C)), coordinate-independent responses stay
in the span of the three Chow orientations

\[
 A^2BC,\qquad B^2AC,\qquad C^2AB,
\tag{71.16}
\]

but (71.10) proves that all three orientations are needed for a universal
unbiased-ReLU interface.  After dense expansion of (A,B,C) into external
labels, even this three-word statement is not automatic: q211 projection is
not equivariant under a generic dense change of coordinates.  It requires
the complete expanded collision tensors to be old-owned.  Coordinate-dependent
response ratios destroy the common three-by-three orientation map unless an
additional separable-kernel certificate is supplied.

### 71.4 Exact correlated-Gaussian kernel

The same result has a closed correlated form.  Let (G) be any centered
unit-variance Gaussian vector, put ((I_i={\bf1}_{\{G_i>t_i\}})), and let
((\delta_i=\delta(G_i-t_i))) denote the boundary distribution.  Generalized
Gaussian integration by parts gives, for the row with repeated output i,

\[
\boxed{
 R^{ijk}_{i i}
 =E[(I_i-\mu_i\delta_i)I_jI_k],
 \qquad
 R^{ijk}_{i j}
 =E[h_iI_i\delta_jI_k],}
\tag{71.17}
\]

with the third entry obtained by swapping j and k.  Coordinate output scales
multiply a row by the corresponding
((\sigma_i^2\sigma_j\sigma_k)).
Equivalently,

\[
 R^{ijk}_{i i}
 =\Pr(I_iI_jI_k=1)
 -\mu_i\varphi_i
 \Pr(I_jI_k=1\mid G_i=t_i),
\tag{71.18}
\]

\[
 R^{ijk}_{i j}
 =\varphi_jE[h_iI_iI_k\mid G_j=t_j].
\tag{71.19}
\]

These formulas include the connected fourth-cumulant response, for the
same missing-coordinate derivative reason used below (71.10).  They reduce
exactly to (71.7) under independence.  Correlation therefore replaces the
diagonal-plus-rank-one product probabilities by conditional bivariate gate
kernels; marginal slope probabilities alone are not an exact reset.

The sharp reopen quantity is the complete-old, live-support separable rank
of (71.17), not the number of generators in the incoming word.  For the
same-anchor part it is the least (s) for which

\[
 K_{i;jk}=\sum_{q=1}^s\lambda_qd_{q,i}^2d_{q,j}d_{q,k}
\tag{71.20}
\]

on every live distinct triple, together with cancellation or old ownership
of the induced 1111 block.  One native CP atom then produces at most (s)
native atoms, and (s=1) is equivalent on a full support to constant (r_i)
in the independent case.  For the complete orientation-mixing operator,
stack the three columns in (71.17); its matrix/separable rank after the old
projection is a necessary lower bound and a matching factorization is
sufficient.  A rank-(s) reopen is lawful only when

\[
 sR_{live}\le R_{cap}
\quad\hbox{and}\quad
 \|Q_{old}^{\perp}(\mathfrak R-\mathfrak R_s)T\|_{value}
 \le\epsilon_{reset}.
\tag{71.21}
\]

This is the promised small-separable-kernel gate.  It is sharp: local
singular rank is necessary, a common global factorization is sufficient,
and any truncated singular tail is charged explicitly rather than silently
folded back into one native word.

### 71.5 Exact global q211 quotient rank: squarefree catalecticant theorem

Sections 61.4--61.5 are local three-label statements.  Dense factors require
a global certificate because their fourth powers leak into other q211
triples and into 1111.  Let

\[
 E=\binom{[n]}2,\qquad z(v)_{ij}=v_iv_j,
\]

and form the edge-by-edge matrix

\[
 L(q)_{ij,ik}=q_{i;jk},\qquad
 L(q)_{ij,kl}=0\quad\hbox{for four distinct labels}.
\tag{71.22}
\]

Its diagonal is unspecified because it is the literal 22 pair gauge.  Let
((\mathcal B_{\rm old})) be the edge-block image of any additional literal
frozen
CP4/Star4 gauge.  Then the exact signed-ray rank after that gauge is

\[
\boxed{
 r_{q211}(q)=\min\left\{k:\
 L(q)+D+B_0=\sum_{s=1}^k\lambda_sz(v_s)z(v_s)^T,
 \ D\ {\rm diagonal},\ B_0\in\mathcal B_{\rm old}\right\}.}
\tag{71.23}
\]

Proof is immediate from the squarefree block of the middle catalecticant:
((\operatorname{Cat}_2(v^4)|_E=z(v)z(v)^T)).  Adjacent edges encode exactly
211, disjoint
edges encode exactly 1111, and identical edges encode the free 22 diagonal.
Conversely, equality of this block leaves only 4/31/22 collision patterns,
so (71.23) is sufficient as well as necessary.

Thus

\[
 \rho_L(q):=\min_{D,B_0}\operatorname{rank}(L(q)+D+B_0)
 \le r_{q211}(q),
\tag{71.24}
\]

but ordinary rank can be very loose: every factor must lie on the
squarefree Veronese (z(v)), whose nonzero support is a clique.  In equivalent
anchor-slice notation, exact sharing by (k) columns requires

\[
 q_{i;jk}=\sum_{s=1}^k\lambda_sv_{s,i}^2v_{s,j}v_{s,k},
 \qquad
 0=\sum_{s=1}^k\lambda_sv_{s,i}v_{s,j}v_{s,k}v_{s,l}
\tag{71.25}
\]

for every distinct triple and quadruple.  The same factor's anchor square,
pair direction, and global 1111 cancellation are tied; unrelated low-rank
Gram slices do not satisfy this automatically.

One unordered triple has exact quotient rank one iff all three of its q211
coordinates are nonzero; choose (v=(a,b,c)) and weight (1/(abc)).  Every
nonzero triple vector with a zero has exact rank two.  In particular AABC
has rank two, while three compatible anchored coordinates can share one ray.

This gives the precise native-AABC reconciliation.  In the abstract
generator alphabet, with ((A=a^Tx,B=b^Tx,C=c^Tx)),

\[
 \boxed{
 A^2BC\equiv{(A+B+C)^4+(A-B-C)^4\over24}}
 \pmod{4/31/22}.
\tag{71.25a}
\]

Thus its quotient signed-ray rank is two.  Its full tensor Waring rank is
six, by the monomial-rank theorem for (x^2yz), while its transported
three-generator Chow/group rank is one.  These are three different ranks.
The two rays in (71.25a) leave the full collision tensor

\[
 2A^4+12A^2B^2+12A^2C^2+2B^4+8B^3C
 +12B^2C^2+8BC^3+2C^4.
\tag{71.25b}
\]

After dense expansion of (a,b,c), (71.25b) itself has external q211 and
1111 coordinates.  Consequently, with q1111 forced to zero, quotient rank
two is lawful only if the literal old 4/31/22 interface owns the *complete
dense generator tensors* in (71.25b), not merely their coordinate-collision
names.  The advertised three K31 plus four K22 terminal reads are sufficient
exactly when their terminal-adjoint span contains every non-CP4 contraction
of (71.25b), including the covariance-pairing completion, at the same seam,
and the remaining external 1111 block is zero or literally old-CP4-owned.
Otherwise the two-ray group leaks and (71.23), not the local rank-two count,
is binding.

For one anchored Gram mode (x_A^2(u^Tx)^2), with (u_A=0), the exact quotient
rank is two when (u) has exactly two nonzero labels and three when it has at
least three.  The upper bound is

\[
 (A+u)^4+(A-u)^4-2u^4=12A^2u^2+2A^4.
\tag{71.26}
\]

For the lower bound, any two-ray cancellation on a triple wholly inside
(supp u) forces the two rays proportional there.  Cancelling the two
unwanted anchor orientations on ((A,j,k)) then forces the same proportionality
at A, which also cancels the desired A-anchor coordinate.

For Gram completions

\[
 G_i=\sum_a\lambda_{ia}u_{ia}u_{ia}^T,\qquad
 R=\sum_i\operatorname{rank}G_i,
\]

the paired construction costs

\[
 2R+r_{q211}\!\left(\Pi_{211+1111}
 \sum_{i,a}\lambda_{ia}u_{ia}^{\otimes4}\right)
 \le3R.
\tag{71.27}
\]

It costs (2R) within this construction exactly when the aggregate dense
leak has zero 211 and 1111 projection or is literally old-owned.  A shared
correction costs less than R precisely when that aggregate leak has smaller
quotient Waring rank.  This is the correct global replacement for applying
the two-ray Section 61.5 identity independently to dense eigenvectors.

### 71.6 NCQC has no intrinsic Cap49 theorem

The commutator-Gram form

\[
 q_{i;jk}=\langle[M_i,M_j],[M_i,M_k]\rangle_F
\tag{71.28}
\]

does not imply low quotient rank.  Indeed, choose an arbitrary off-diagonal
q211 tensor.  For each anchor i, add a sufficiently large diagonal to its
symmetric slice to obtain a PSD completion (G_i), and factor it.  In an
independent anchor-i matrix block choose one symmetric matrix (H_i) and let
all nonanchor matrices be commuting diagonal matrices (D(d_{ij})).  Since

\[
 \langle[H_i,D(d)],[H_i,D(e)]\rangle_F
 =2\sum_{a<b}(H_i)_{ab}^2(d_a-d_b)(e_a-e_b),
\tag{71.29}
\]

the weighted graph Laplacian on the diagonal subspace realizes any PSD
(G_i).  Direct summing the anchor blocks makes nonanchor commutators vanish
inside each block and gives exactly (71.28), with no cross-anchor
contamination.  Ambient dimension at most

\[
 \sum_i(\operatorname{rank}G_i+1)\le n^2
\tag{71.30}
\]

suffices.  Therefore arbitrary q211 tensors on 32 active labels already fit
inside matrix width 1024, while their generic quotient rank is far above 49.

The q211 coefficient space has dimension

\[
 3\binom n3=n\binom{n-1}2.
\]

A sum of k signed rays has at most (kn) continuous parameters.  Hence a
Zariski-generic q211 tensor has quotient rank at least

\[
 \boxed{\binom{n-1}2.}
\tag{71.31}
\]

This is 1431 at (n=55); the target space has 78,705 coordinates while a
Cap49 ray family has at most 2,695 continuous dimensions before fixed old
gauges.  At 32 active labels the corresponding lower bound is 465.  NCQC
or Gram terminology alone therefore cannot reopen Cap49.  The actual
matrices must exhibit the coupled-Veronese identities (71.25).

Finally, exact tensor rank does not by itself lower-bound one scalar endpoint
error: one signed ray can match one nonzero linear functional by scaling.
For broad-spectrum fidelity, the panel/layer costates must see the discarded
directions.  An ordinary-rank relaxation gives the rigorous coefficient
error lower bound

\[
 \min_{B_0,D}\sum_{s>49}\sigma_s(L(q)-B_0+D)^2,
\tag{71.32}
\]

but converting (71.32) into raw-MSE loss requires a lower bound for the
panel value map on that residual.  This separates the exact Cap49 no-go from
an unjustified claim about one tuned scalar endpoint.

### 72.5 Exact target scalar and the arithmetic boundary

Compile collision sewing and complete-old ownership into
(\widetilde\Lambda) as in (69.1), and put
(L_i[j,k]=\operatorname{Sym}_{jk}\widetilde\Lambda_{i;jk}).  The desired
state-only action is exactly

\[
\boxed{
 s_\Delta
 =\sum_{i,j,k}L_i[j,k]\Delta T_{iijk}
 =S_{\widetilde\Lambda}(Y)
  -S_{\widetilde\Lambda}(Y^{(2)}).}
\tag{72.16}
\]

Here each scalar surface on the right is the first-two-moment-corrected
expectation (69.3)--(69.4).  Thus a frozen endpoint can be evaluated either
by the diagram series (72.5) or by a trivariate Gaussian ReLU moment for
each ((i,j,k)); no four-index tensor is required.  This is an exact
target-scalar theorem, not yet a cheap one: a generic dense query still has
(\Theta(n^3)) entries.  The exact query ranks in Section 73 remain the
necessary compression receipt.

The leading biased band does have a cheap CP action.  For one query atom

\[
 L_i[j,k]=\alpha_i\operatorname{Sym}_{jk}(\beta_j\gamma_k),
\]

write

\[
 p_\beta=R(\beta\odot b_1),\qquad
 p_\gamma=R(\gamma\odot b_1),\qquad
 q=R^{\odot2}(\alpha\odot b_1^{\odot2}).
\]

Direct contraction of (72.7) gives

\[
\boxed{\begin{aligned}
 s_\Delta^{[3]}={}&
 2\langle\alpha\odot b_3\odot b_1,
              p_\beta\odot p_\gamma\rangle\\
 &+\langle\beta\odot b_3,q\odot p_\gamma\rangle
  +\langle\gamma\odot b_3,q\odot p_\beta\rangle .
\end{aligned}}
\tag{72.17}
\]

It uses three matrix-vector passes per generic atom, or two when
(\beta=\gamma).  Across ten active seams, before reverse construction,
hooks, and writes, the optimistic width-GEMM bills are (30R_c/n) and
(20R_c/n).  At (n=1024), the one-equivalent necessary screens are therefore

\[
 R_c\le34\quad\hbox{or}\quad R_c\le51.
\tag{72.18}
\]

There is a sharp reason this matvec count does not extend to every diagram.
For a monomial with singleton-edge exponents ((u,v,f)), anchor weight
(w_i), and singleton weights ((d_j,e_k)), one CP orientation contracts as

\[
 \sum_{i,j,k}w_i(d_j\beta_j)(e_k\gamma_k)
 (R_{ij})^u(R_{ik})^v(R_{jk})^f
 =\operatorname{tr}\!\left{
 D_wR^{\odot u}D_{d\odot\beta}R^{\odot f}
 D_{e\odot\gamma}(R^{\odot v})^T\right\}.
\tag{72.19}
\]

The symmetrized query adds the exchanged ((\beta,j)\leftrightarrow
(\gamma,k)) orientation when it is distinct.  If an exponent is zero, the
corresponding Schur power is the all-ones rank-one matrix, recovering a
matvec/outer-product contraction.  When (u,v,f>0), however, (72.19)
contains a generic dense matrix product.

For example, the triangle obstruction (72.10) has the exact query action

\[
\boxed{
 s_\Delta^{tri}
 =\operatorname{tr}\!\left{
 D_{\alpha\odot b_4\odot b_2}R
 D_{\beta\odot b_2}R
 D_{\gamma\odot b_2}R\right}.}
\tag{72.20}
\]

One independent diagonal scaling requires one query-specific dense product,
followed by an (O(n^2)) Frobenius contraction.  Paying that product at each
of ten seams already exceeds the total one-equivalent budget.  It becomes
lawful only if the product is an existing D10 contraction, is fused into an
already billed product, or the actual query/Schur factors have a certified
low-rank tail.  Consequently, "D10 carries bands 3--6" is an information
claim; it is not a proof that the complete connected action is cheap.  The
rooted subbands are plausible matvec carriers, while triangle-bearing
subbands face the dense-product gate.

### 72.6 Hermite-tail, endpoint-value, and promotion gates

Let

\[
 Y_i^{(Q)}=P_{\le Q}Y_i,\qquad
 e_{iQ}=\|Y_i-Y_i^{(Q)}\|_4,
\]

and define

\[
 M_{iQ}=\max\{\|Y_i-EY_i\|_4,
                \|Y_i^{(Q)}-EY_i\|_4\}.
\]

For centered random variables,

\[
 |\kappa_4(A,B,C,D)|
 \le4\|A\|_4\|B\|_4\|C\|_4\|D\|_4.
\]

Telescoping the four arguments therefore gives the rigorous q211
truncation bound

\[
\boxed{
 |\Delta T_{iijk}-\Delta T^{(Q)}_{iijk}|
 \le4\{2e_{iQ}M_{iQ}M_{jQ}M_{kQ}
       +e_{jQ}M_{iQ}^2M_{kQ}
       +e_{kQ}M_{iQ}^2M_{jQ}\}.}
\tag{72.21}
\]

Hence the compiled endpoint error is bounded by

\[
 |s_\Delta-s_\Delta^{(Q)}|
 \le\sum_{i,j,k}|L_i[j,k]|\,E^{(Q)}_{i;jk},
\tag{72.22}
\]

where (E^{(Q)}_{i;jk}) is the right-hand side of (72.21).  Every
(e_{iQ}) and (M_{iQ}) is a one-dimensional Gaussian integral determined by
(\tau_i,\sigma_i) and (72.2); no joint simulation premise is needed.
This is an absolute bound.  It supplies no relative accuracy when the signed
endpoint action is small because of cancellation.

The truncation receipt in (72.22), query-factor approximation, reset error,
and any forward-carrier tail must all enter (\epsilon_f) in the robust Schur
gate (66.30).  Promotion requires that lower bound to clear both the raw-MSE
target and the adjusted-cost fire bar.  Hermite energy, pair-rest energy, or
a large unprojected scalar is not a substitute for the compiled
complete-old endpoint value.

The resulting decision split is:

1. `AFFINE_RELU_REST_DIAGRAM_SOURCE` -- **solved exactly**, under the literal
   Gaussian preactivation premise, by (72.2) and (72.5).
2. `AFFINE_RELU_BIASED_H3_ACTION` -- **solved**, with the CP cost and rank
   screens (72.17)--(72.18); novelty and complete-old value remain open.
3. `AFFINE_RELU_CENTERED_FIRST_H4_BAND` -- **solved** by Section 74 and
   identified here as the complete ((4,2,1,1)) band.
4. `AFFINE_RELU_COMPLETE_NATIVE_LOW_RANK` -- **generic no-go**: singleton
   edges generate the full-rank matrices (72.12)--(72.14); only rooted
   subbands inherit a universal small carrier.
5. `D10_COLLAPSED_PAIR_REST_AS_SOURCE` -- **no-go** by (72.15): it loses
   degree and connected-topology information.
6. `D10_DEGREE_BANDS_3_TO_6` -- **information-sufficient for their finite
   diagram truncation, arithmetic-conditional**: triangle products must be
   resident, fused, or compressed with a task-visible receipt.
7. `FINITE_HERMITE_BAND_EXACTNESS` -- **generic no-go** for ReLU; use the
   absolute tail gate (72.21)--(72.22).
8. `AFFINE_RELU_TARGET_SCALAR` -- **solved algebraically** by (72.16), but
   deployability is conditional on the Section 73 query ranks or an existing
   fused scalar surface.
9. `AFFINE_RELU_COMPLETE_OLD_VALUE` -- **open/blocking**: the retained action
   and its robust Schur lower bound have not been certified on the frozen
   corpus.

## 78. All-orders rooted Price/Fock provider and the exact first broad triangle

Section 77 resums every chaos contribution through quadratic correlation
order.  There is a strictly larger solvable sector: sum every diagram with
no direct singleton--singleton edge ((m_{jk}=0)).  Collapsing the two
repeated (i)-slots into their square response exposes an exact Price series
and shows both why this rooted sector is cheap after a finite chaos cut and
why the first omitted triangle is not.

For either centered coordinate function (H=F) from (77.1) or its projection
(H=X) from (77.3), define the signed derivative/Hermite data

\[
 b^H_{a,q}:=E[H_a\operatorname{He}_q(Z_a)]=E[H_a^{(q)}],
 \qquad
 h^H_{a,q}:=E[H_a^2\operatorname{He}_q(Z_a)]
            =E[(H_a^2)^{(q)}].
\tag{78.1}
\]

Distributional derivatives are intended.  These data have closed generating
functions.  With (r=t-\tau_a), (P_a(t)=\overline\Phi(\tau_a-t)), and
(\phi_a(t)=\phi(\tau_a-t)),

\[
\begin{aligned}
 B_a^F(t)&:=E[F_ae^{tZ_a-t^2/2}]\\
 &=\sigma_a\{rP_a(t)+\phi_a(t)\}-\mu_a,
\end{aligned}
\tag{78.2}
\]

and

\[
\begin{aligned}
 H_a^F(t)&:=E[F_a^2e^{tZ_a-t^2/2}]\\
 &=\sigma_a^2\{(r^2+1)P_a(t)+r\phi_a(t)\}\\
 &\quad-2\mu_a\sigma_a\{rP_a(t)+\phi_a(t)\}+\mu_a^2.
\end{aligned}
\tag{78.3}
\]

Thus ((B_a^F)^{(q)}(0)=b^F_{a,q}) and
((H_a^F)^{(q)}(0)=h^F_{a,q}).  For the quadratic projection,

\[
 B_a^X(t)=b_{1a}t+{b_{2a}\over2}t^2,
\tag{78.4}
\]

\[
\begin{aligned}
 H_a^X(t)={}&b_{1a}^2+{b_{2a}^2\over2}
 +2b_{1a}b_{2a}t+(b_{1a}^2+b_{2a}^2)t^2\\
 &+b_{1a}b_{2a}t^3+{b_{2a}^2\over4}t^4.
\end{aligned}
\tag{78.5}
\]

In particular, (h^F_{a,1}-h^X_{a,1}=d_{1a}) and
(h^F_{a,2}-h^X_{a,2}=d_{2a}) from (77.5)--(77.6).

### 78.1 Exact collapsed-slot Price series and rooted resummation

For one centered family (H), put

\[
 K_H(x,y,z):=\kappa_4(H_i,H_i,H_j,H_k).
\]

Repeated Price differentiation, with (H_i^2) treated as the function at the
anchor coordinate, gives the exact connected expansion

\[
\boxed{\begin{aligned}
 K_H(x,y,z)={}&
 \sum_{a,b\ge1}
 {\{h^H_{i,a+b}-2b^H_{i,a}b^H_{i,b}\}
    b^H_{j,a}b^H_{k,b}\over a!b!}x^ay^b\\
 &+\sum_{\substack{c\ge1,\ a,b\ge0\\a+b\ge1}}
 {h^H_{i,a+b}b^H_{j,a+c}b^H_{k,b+c}
  \over a!b!c!}x^ay^bz^c .
\end{aligned}}
\tag{78.6}
\]

Indeed, (a) derivatives in (x), (b) in (y), and (c) in (z) leave the three
independent-coordinate factors

\[
 (H_i^2)^{(a+b)},\qquad H_j^{(a+c)},\qquad H_k^{(b+c)}.
\]

When (c=0), connectedness requires (a,b\ge1), and the two
(\operatorname{Cov}(H_i,H_j)\operatorname{Cov}(H_i,H_k)) pairings produce
the subtraction in the first line.  When (c\ge1), only (a=b=0) is
disconnected, and that term is canceled by
(E H_i^2\operatorname{Cov}(H_j,H_k)).  This proves (78.6) without enumerating
the internal edge multiplicity between the repeated slots: it has already
been resummed into (h^H).

The exact no-singleton-edge defect is therefore

\[
\boxed{
 \Delta^{root}_{i;jk}(x,y)
 :=K_F^{(c=0)}(x,y)-K_X^{(c=0)}(x,y),}
\tag{78.7}
\]

with the first line of (78.6) used for each family.  It contains every power
(x^ay^b), not merely one Hermite band.

There is an equivalent Fock-space identity.  Let

\[
 T_rH_a(t)=\sum_{q\ge1}{b^H_{a,q}r^q\over q!}
             \operatorname{He}_q(t)
\tag{78.8}
\]

be the centered Ornstein--Uhlenbeck/conditional score, and let the Wick
product satisfy
(\operatorname{He}_a\diamond\operatorname{He}_b
=\operatorname{He}_{a+b}).  Then

\[
 K_H^{(c=0)}(x,y)
 =E\{H_i^2(T_xH_j\diamond T_yH_k)(Z_i)\}
  -2\langle H_i,T_xH_j\rangle
    \langle H_i,T_yH_k\rangle.
\tag{78.9}
\]

Whenever (x^2+y^2<1), (78.7) also equals
(K_F(x,y,0)-K_X(x,y,0)) for the lawful zero-((j,k))-correlation Gaussian.
Equation (78.9) is the diagrammatic continuation.  Outside an
absolute-convergence domain it must not be treated as a probability
identity: a simple sufficient deployment gate is

\[
 \sup_{i,j,k}(\rho_{ij}^2+\rho_{ik}^2)\le1-\delta
 \quad\hbox{for some }\delta>0.
\tag{78.10}
\]

The ((a,b)=(1,1)) term of (78.7) is
(d_{2i}b_{1j}b_{1k}xy), exactly the first term of (77.8).  Hence the rooted
sum is an all-orders extension in the ((x,y))-plane.

### 78.2 Finite-band slice rank, exact CP action, and the one-equivalent gate

For a chaos cut (Q), define the (n\times Q) anchor-dependent singleton
dictionary

\[
 U^{(Q)}_{i;jq}:={b^F_{j,q}\over q!}(R_{ij})^q,
 \qquad 1\le q\le Q,
\tag{78.11}
\]

and the symmetric local coefficient matrix

\[
\begin{aligned}
 A^{(Q)}_{i;qr}:={}&h^F_{i,q+r}-2b^F_{i,q}b^F_{i,r}\\
 &-\mathbf1_{q,r\le2}
   \{h^X_{i,q+r}-2b^X_{i,q}b^X_{i,r}\}.
\end{aligned}
\tag{78.12}
\]

The truncated rooted slice factors exactly as

\[
 \boxed{
 \Delta^{root,Q}_{i;:,:}
 =U_i^{(Q)}A_i^{(Q)}(U_i^{(Q)})^T,
 \qquad
 \operatorname{rank}\Delta^{root,Q}_{i;:,:}\le Q.}
\tag{78.13}
\]

This is a slice-rank theorem, not a free persistent carrier.  A coordinate-
anchored native realization uses up to
(\sum_i\operatorname{rank}A_i^{(Q)}) generators, generically (nQ), because
the repeated anchors (e_i\odot e_i) are independent.  The exact uncut
kernel has no universal finite rank.  In fact, (H_a^F(t)) contains the
non-exponential-polynomial factor (\overline\Phi(\tau_a-t)); hence its
Hankel derivative matrix ((h^F_{a,q+r})) has infinite rank.  Subtracting one
outer product and the finite (X) block cannot make it finite.  At finite
width the uncut slice is therefore generically full rank.

The target action is much cheaper.  For the same normalized partially
symmetric CP convention as Section 77, define

\[
 p_{\beta,q}={1\over q!}R^{\odot q}
                 (\beta\odot b^F_q),\qquad
 p_{\gamma,q}={1\over q!}R^{\odot q}
                 (\gamma\odot b^F_q).
\tag{78.14}
\]

Writing (p_\beta(i,:)) for the (Q)-vector of their (i)-th entries,

\[
\boxed{
 s_{root,Q}
 =\sum_i\alpha_i\,
 p_\beta(i,:)^TA_i^{(Q)}p_\gamma(i,:).}
\tag{78.15}
\]

Thus the apparent (Q^2) chaos pairing needs only (2Q) Schur-power matvecs
per generic CP atom, or (Q) when (\beta=\gamma), plus
(O(nQ^2)) local arithmetic.

To retain all of Section 77, add its two nonroot quadratic arms but not its
already-included (xy) arm:

\[
 \Delta^{hyb,Q}_{i;jk}=\Delta^{root,Q}_{i;jk}
 +d_{1i}b_{2j}b_{1k}xz+d_{1i}b_{1j}b_{2k}yz.
\tag{78.16}
\]

They reuse the (q=1) singleton products and require only
(q_\alpha=R(\alpha\odot d_1)).  If (h_Q) is the number of nonzero signed
chaos bands through (Q), the ten-seam optimistic bills are

\[
 C_{hyb}^{gen}={10(2h_Q+\mathbf1_{d_1\ne0})R_c\over n},
 \qquad
 C_{hyb}^{sym}={10(h_Q+\mathbf1_{d_1\ne0})R_c\over n}.
\tag{78.17}
\]

For generic bias (h_Q=Q).  At (n=1024), the one-equivalent screens for
(Q=2,4,6) are respectively

\[
\begin{array}{c|ccc}
 Q&2&4&6\\ \hline
 R_c^{gen}&20&11&7\\
 R_c^{sym}&34&20&14.
\end{array}
\tag{78.18}
\]

#### 78.2a The centered Q=2 provider is already a strict closed extension

At zero bias, the first three square-response defects needed by (Q=2) are

\[
 h^F_{i,2}-h^X_{i,2}
 =\sigma_i^2\left({1\over2}-{2\over\pi}\right),
 \qquad
 h^F_{i,3}-h^X_{i,3}=-\sigma_i^2\phi(0),
\tag{78.18a}
\]

\[
 h^F_{i,4}-h^X_{i,4}=-{2\sigma_i^2\over\pi}.
\tag{78.18b}
\]

Substitution in (78.7) gives the complete signed (Q=2) rooted defect in one
closed polynomial:

\[
\boxed{
 \Delta^{root,2}_{i;jk}
 =\sigma_i^2\sigma_j\sigma_k\left{
 {\pi-4\over8\pi}xy
 -{x^2y+xy^2\over8\pi}
 -{x^2y^2\over4\pi^2}\right\}.}
\tag{78.18c}
\]

It resums every internal repeated-anchor chaos compatible with singleton
degrees one and two.  Relative to Section 77 it keeps the exact calibrated
(xy) term, adds both rooted cubic arms, and adds the rooted quartic
(x^2y^2) arm.  Relative to the single H4 band in Section 74, the (xy)
coefficient is multiplied by ((4-\pi)), the two cubic coefficients agree,
and the quartic term is new.  Because the next nonzero centered singleton
degree is four, (78.18c) is the complete rooted correlation jet through
total degree four.

No (q_\alpha) pass is needed because (d_1=0).  The action uses only the
(R) and (R^{\odot2}) products in (78.14): four matvecs per generic CP atom
or two per symmetric atom.  Across ten seams the one-equivalent screens are

\[
 \boxed{R_c\le25\quad\hbox{generic},\qquad
        R_c\le51\quad\hbox{symmetric}.}
\tag{78.18d}
\]

Thus the centered (Q=2) arm is the cheapest genuine extension of Section
77.  It requires no D10 band above degree two; only the signed local square
responses (78.18a)--(78.18b) and the literal Schur-power actions are new.

At zero bias, (d_1=0) and odd bands above one vanish, so (Q=6) uses only
(h_Q=4) bands and the screens improve to (R_c\le12/25).  The case (Q=1)
is exactly Section 77.  Any (Q\ge2) is a strict algebraic extension whenever
one of the added rooted coefficients survives complete-old projection.
Therefore the extension fits below one D10 equivalent only conditionally on
(78.17), the actual Section 73 CP query rank, and a certified root tail.

Degree-resolved D10 matrices can supply (78.14) only if the signed band or
the bare (R^{\odot q}) action is exposed.  Recovering it from
(D_{b_q}R^{\odot q}D_{b_q}/q!) requires diagonal unscaling and has
conditioning proportional to (1/\min|b_{a,q}|); zeros make that interface
non-identifiable.  The collapsed rest (72.15) cannot supply the separate
passes.

An exact absolute root-tail receipt is

\[
\begin{aligned}
 \epsilon^{root,Q}_{i;jk}
 :=\sum_{\substack{a,b\ge1\\\max(a,b)>Q}}
 { |x|^a|y|^b\over a!b!}\,
 \big|{}&\{h^F_{i,a+b}-2b^F_{i,a}b^F_{i,b}\}
            b^F_{j,a}b^F_{k,b}\\
 &-\{h^X_{i,a+b}-2b^X_{i,a}b^X_{i,b}\}
            b^X_{j,a}b^X_{k,b}\big|.
\end{aligned}
\tag{78.19}
\]

It is finite under (78.10) and is determined entirely by the one-dimensional
generating functions (78.2)--(78.5).

### 78.3 The exact first excluded broad triangle

The second line of (78.6) gives every excluded coefficient without diagram
enumeration.  After the exact root and the two quadratic arms in (78.16),
the complete cubic nonroot term is

\[
\boxed{\begin{aligned}
 C^{(3)}_{i;jk}={}&
 {h^F_{i,2}b^F_{j,3}b_{1k}\over2}x^2z
 +{h^F_{i,2}b_{1j}b^F_{k,3}\over2}y^2z\\
 &+d_{2i}b_{2j}b_{2k}xyz\\
 &+{h^F_{i,1}b^F_{j,3}b_{2k}\over2}xz^2
 +{h^F_{i,1}b_{2j}b^F_{k,3}\over2}yz^2.
\end{aligned}}
\tag{78.20}
\]

At zero bias (b_3=0), the four side terms vanish.  The sole cubic remainder
is the **all-chaos-resummed triangle**

\[
 \boxed{C^{tri}_{i;jk}=d_{2i}b_{2j}b_{2k},xyz.}
\tag{78.21}
\]

For centered affine ReLU this coefficient is explicitly

\[
 d_{2i}b_{2j}b_{2k}
 =\sigma_i^2\sigma_j\sigma_k{\pi-4\over4\pi^2}.
\tag{78.21a}
\]

The H4-only triangle coefficient is
(-\sigma_i^2\sigma_j\sigma_k/(4\pi^2)); hence the exact internal-chaos
resummation multiplies that subband by ((4-\pi)).  The higher internal
chaoses cancel the remaining fraction rather than adding with the H4 sign.

There is also a scale-free small-correlation receipt.  Relative to the
leading centered rooted star in (78.18c), the exact cubic triangle obeys

\[
 { |d_{2i}b_{2j}b_{2k}xyz|\over
   |\sigma_i^2\sigma_j\sigma_k(\pi-4)xy/(8\pi)|}
 ={2\over\pi}|z|.
\tag{78.21b}
\]

If (x,y\ge0), every additional term in the retained polynomial (78.18c)
has the same sign as the leading star, so (2|z|/\pi) is also an upper bound
on the triangle-to-retained-(Q=2)-root ratio before the fourth-order broad
remainder.  With arbitrary signs the rooted terms can cancel and no relative
bound follows; the absolute gate (78.24) remains binding.

Section 72.10 is only its first H4 subband.  The exact coefficient in
(78.21) includes every internal repeated-slot contraction.  Its CP action is

\[
 \boxed{
 s_{tri}=\operatorname{tr}\!\left{
 D_{\alpha\odot d_2}R
 D_{\beta\odot b_2}R
 D_{\gamma\odot b_2}R\right}.}
\tag{78.22}
\]

This needs one query-specific dense product for each independent diagonal
scaling.  Ten seams therefore breach the one-equivalent budget unless the
product is literally resident/fused or the query factors have a certified
low-rank reduction.  Equation (78.22) is the exact arithmetic boundary
between the all-orders rooted matvec provider and the broad sector.

Let (J_4) bound every fourth correlation derivative of the nonroot defect on
the line segment from the origin to ((x,y,z)) inside a compact lawful
correlation/threshold set.  Taylor's theorem gives

\[
 \left|\Delta_{i;jk}-\Delta^{root}_{i;jk}
 -d_{1i}b_{2j}b_{1k}xz-d_{1i}b_{1j}b_{2k}yz
 -C^{(3)}_{i;jk}\right|
 \le {J_4\over24}(|x|+|y|+|z|)^4.
\tag{78.23}
\]

In the centered case, if the triangle is deliberately not billed, the
sharp third-order error gate is consequently

\[
 \boxed{
 |\Delta_{i;jk}-\Delta^{root}_{i;jk}|
 \le |d_{2i}b_{2j}b_{2k}xyz|
      +{J_4\over24}(|x|+|y|+|z|)^4.}
\tag{78.24}
\]

For a (Q)-band implementation, add (\epsilon^{root,Q}_{i;jk}) from (78.19).
The compiled endpoint error is the absolute (L)-weighted sum, as in
(72.22), and must enter (\epsilon_f) in (66.30).

Decision split:

1. `AFFINE_Q211_ROOTED_ALL_ORDERS` -- **solved exactly** by
   (78.6)--(78.9), subject to the convergence gate (78.10).
2. `AFFINE_Q211_ROOTED_FINITE_RANK` -- **solved at cut Q** by (78.13);
   the exact uncut slice is generically full rank and a forward native
   realization generically costs (nQ) anchored modes.
3. `AFFINE_Q211_ROOTED_CP_ACTION` -- **solved** by (78.14)--(78.17).
4. `AFFINE_Q211_ROOTED_STRICT_PRICE_EXTENSION` -- **conditional GO** for
   (Q\ge2) when the CP-rank, signed-band interface, and tail gates pass;
   (Q=1) is exactly Section 77.
5. `AFFINE_Q211_ROOTED_D10_COLLAPSED_REST` -- **no-go**; degree resolution
   and stable signed scaling are necessary.
6. `AFFINE_Q211_FIRST_BROAD_TRIANGLE` -- **solved exactly** by
   (78.21)--(78.22), but over budget absent fusion.
7. `AFFINE_Q211_ROOTED_FIDELITY` -- conditional on (78.19), (78.23), and the
   complete-old endpoint-weighted error.
8. `AFFINE_Q211_ROOTED_OLD_SCHUR_VALUE` -- open blocking receipt.

## 79. Multi-reset spectral collapse of q211 orientation: exact tail gate
and the transition-rotation obstruction

Section 71 gives the exact three-by-three reset block on the three choices of
repeated anchor inside one unordered triple.  Its small contrast eigenvalue
suggests a deep rank-one approximation, but that conclusion is valid only
when every intervening transition and collision projection preserves the
symmetric/contrast splitting.  This section gives the exact contraction,
the biased stability bound, and a counterexample showing why the seam receipt
is necessary.

### 79.1 Unbiased reset: exact depth-d error and correlation

Let

\[
 u={1\over\sqrt3}(1,1,1)^T,\qquad
 P_\parallel=uu^T,\qquad P_\perp=I-P_\parallel .
\tag{79.1}
\]

Equation (71.10) is

\[
 R_0=\lambda_\parallel P_\parallel+\lambda_\perp P_\perp,
\qquad
 \lambda_\parallel={1+1/\pi\over8},\qquad
 \lambda_\perp={1-2/\pi\over8}.
\tag{79.2}
\]

The exact contrast ratio is

\[
 \boxed{\rho_0={\lambda_\perp\over\lambda_\parallel}
 ={\pi-2\over\pi+1}.}
\tag{79.3}
\]

With no intervening orientation mixing,

\[
 \boxed{R_0^d=\lambda_\parallel^d
 \{P_\parallel+\rho_0^dP_\perp\}.}
\tag{79.4}
\]

Therefore the best rank-one operator approximation is
(\lambda_\parallel^dP_\parallel).  Its exact relative Frobenius error and
flattened correlation with the full operator are

\[
 \boxed{
 e_{F,d}={\sqrt2\,\rho_0^d\over
 \sqrt{1+2\rho_0^{2d}}},
 \qquad
 c_{F,d}={1\over\sqrt{1+2\rho_0^{2d}}}.}
\tag{79.5}
\]

For a particular input (q=q_\parallel+q_\perp), put
(\chi=\|q_\perp\|/\|q_\parallel\|).  When (q_\parallel\ne0), the exact
rank-one-output error and correlation are

\[
 e_d(q)={\rho_0^d\chi\over\sqrt{1+\rho_0^{2d}\chi^2}},
 \qquad
 c_d(q)={1\over\sqrt{1+\rho_0^{2d}\chi^2}}.
\tag{79.6}
\]

There is an indispensable signal-floor clause: a pure contrast input has
(q_\parallel=0), so the proposed rank-one output is zero even though the
absolute contrast norm contracts.  For a requested operator error
(\epsilon<1), (79.5) is equivalent to

\[
 d\ge
 {\log\{\sqrt{2(1-\epsilon^2)}/\epsilon\}\over-\log\rho_0}.
\tag{79.7}
\]

No empirical fit or asymptotic notation enters these formulas.

### 79.2 A common standardized bias preserves the eigenspaces exactly

Use the Section 71 notation

\[
 Q=\Pr(G>t),\qquad F=1-Q,\qquad
 \mu=\varphi(t)-tQ.
\]

If all three coordinates have the same standardized threshold, (71.9)
still has the exact eigenspaces (P_\parallel,P_\perp).  The two eigenvalues,
their gap, and their ratio are

\[
 \boxed{
 \lambda_\perp(t)=Q\{Q^2-\mu\varphi\},}
\tag{79.8}
\]

\[
 \boxed{
 \lambda_\parallel(t)
 =Q\{Q^2+\mu\varphi(3F-1)\},\qquad
 g(t)=\lambda_\parallel-\lambda_\perp
 =3Q\mu\varphi F,}
\tag{79.9}
\]

\[
 \boxed{
 \rho(t)=
 {Q^2-\mu\varphi\over Q^2+\mu\varphi(3F-1)}.}
\tag{79.10}
\]

For a sequence of common biases and no mixing, replace (\rho_0^d) in
(79.4)--(79.6) by (\prod_{\ell=1}^d|\rho(t_\ell)|).  A very negative
threshold approaches a linear always-open gate, closes the spectral gap,
and removes the contraction.  Bias cannot therefore be certified by
reusing the zero-bias ratio.

### 79.3 Heterogeneous gates: exact symmetrization and Davis--Kahan gate

For finite coordinate-specific thresholds, Section 71.9 has the form

\[
 R=Q_iQ_jQ_k\{\operatorname{Diag}(\delta)+bc^T\},
\]

\[
 b_r={\mu_rF_r\over Q_r},\qquad
 c_r={\varphi_r\over Q_r},\qquad
 \delta_r=1-{\mu_r\varphi_r\over Q_r^2}.
\tag{79.11}
\]

Although this matrix is not generally symmetric, it is exactly diagonally
symmetrizable.  With

\[
 S=\operatorname{Diag}\sqrt{b_r/c_r},\qquad
 w_r=\sqrt{b_rc_r},
\]

\[
 \boxed{
 S^{-1}RS=Q_iQ_jQ_k
 \{\operatorname{Diag}(\delta)+ww^T\}.}
\tag{79.12}
\]

Thus all spectral perturbation statements are exact in the
(S^TS)-weighted norm.  Let (\bar R) be a common-bias reference block with
gap (\bar g), and put

\[
 \varepsilon=\|S^{-1}RS-\bar R\|_2.
\]

If (\varepsilon<\bar g), Weyl and Davis--Kahan give

\[
 \boxed{
 \sin\angle(\widetilde u,u)
 \le{\varepsilon\over\bar g-\varepsilon},\qquad
 \widetilde\lambda_\parallel\ge
 \bar\lambda_\parallel-\varepsilon,\qquad
 |\widetilde\lambda_\perp|
 \le|\bar\lambda_\perp|+\varepsilon.}
\tag{79.13}
\]

The certified one-reset tail ratio is therefore

\[
 \boxed{
 \eta={|\bar\lambda_\perp|+\varepsilon\over
 |\bar\lambda_\parallel|-\varepsilon},}
\tag{79.14}
\]

provided the denominator is positive.  Euclidean-norm claims acquire the
additional condition factor (\kappa(S)); a well-conditioned gate profile is
not optional.  Across layers, the eigenvector drifts from (79.13) enter the
transition-coupling terms below.  Multiplying the scalar ratios while
ignoring those angles is invalid.

### 79.4 Exact transition/collision invariance receipt

Normalize each reset so that its symmetric gain is one, and let (T_\ell)
denote the complete map between two reset blocks: dense transition,
coordinate change, collision sewing, and complete-old projection.  Relative
to (P_\parallel\oplus P_\perp), define

\[
 a_\ell=|u^TT_\ell u|,\qquad
 \beta_\ell=\|P_\parallel T_\ell P_\perp\|_2,
\]

\[
 \gamma_\ell=\|P_\perp T_\ell P_\parallel\|_2,\qquad
 h_\ell=\|P_\perp T_\ell P_\perp\|_2.
\tag{79.15}
\]

Exact multiplicative spectral collapse holds when

\[
 \boxed{
 P_\perp T_\ell P_\parallel=0,\qquad
 P_\parallel T_\ell P_\perp=0,\qquad
 a_\ell>0.}
\tag{79.16}
\]

Then the depth-d contrast/symmetric operator ratio obeys

\[
 \boxed{
 \tau_d\le
 \prod_{\ell=1}^d\eta_\ell
 \prod_{\ell=1}^{d-1}{h_\ell\over a_\ell}.}
\tag{79.17}
\]

For forward source transport alone, the first zero in (79.16) prevents
re-excitation of contrast.  A reusable Bellman/reverse certificate needs
both zeros.  An orthogonal collision projector satisfies both exactly when
(u) is one of its invariant directions; if it kills (u), then
(a_\ell=0) and the proposed symmetric carrier has no live signal.

A sufficient architectural condition is permutation equivariance across
the three anchor orientations.  By the two-isotypic decomposition of the
permutation representation, such a map is block diagonal on
(P_\parallel,P_\perp).  A generic dense coordinate transition does not
satisfy this merely because it is well conditioned: external q211
projection is not equivariant under a generic dense change of basis, and a
collision projector can break the symmetry again at the next seam.

### 79.5 Sharp injected-tail recurrence and its deep floor

The preceding exact-zero condition can be relaxed quantitatively.  For a
state (x=x_\parallel+x_\perp), set

\[
 z_\ell={\|x_{\perp,\ell}\|\over\|x_{\parallel,\ell}\|}.
\]

Applying (T_\ell) and then a normalized reset with tail ratio
(\eta_{\ell+1}) gives the sharp norm bound

\[
 \boxed{
 z_{\ell+1}\le
 \eta_{\ell+1}
 {\gamma_\ell+h_\ell z_\ell\over
 a_\ell-\beta_\ell z_\ell},}
\tag{79.18}
\]

whenever the denominator is positive.  Thus a nonzero
(\gamma_\ell) injects new contrast after every reset; the tail generally
approaches a floor rather than (\rho_0^d).

For uniform bounds ((a,\beta,\gamma,h,\eta)), an invariant cone exists when

\[
 a>\eta h,\qquad
 (a-\eta h)^2\ge4\beta\eta\gamma.
\tag{79.19}
\]

Its smallest fixed-point radius is

\[
 \boxed{
 z_*={a-\eta h-
 \sqrt{(a-\eta h)^2-4\beta\eta\gamma}\over2\beta}}
\tag{79.20}
\]

for (\beta>0), and

\[
 \boxed{z_*={\eta\gamma\over a-\eta h}}
\tag{79.21}
\]

for (\beta=0).  Starting inside the invariant cone gives
(z_\ell\le\max\{z_0,z_*\}).  The corresponding rank-one output correlation
is at least ((1+z_*^2)^{-1/2}).  Equations (79.18)--(79.21), applied also to
the adjoint, are the sharp cheap deep-tail gate: they require only the
three-by-three orientation blocks and seam invariance norms, not a dense H4
transport.

### 79.6 Exact rotation counterexample

Small (\rho_0) alone is insufficient even for an orthogonal, condition-one
transition.  Choose any unit contrast vector (v), and on
(\operatorname{span}\{u,v\}) let

\[
 T u=v,\qquad Tv=-u.
\tag{79.22}
\]

Let (D=P_\parallel+\rho_0P_\perp) be the normalized reset.  In the ordered
basis ((u,v)),

\[
 DT=\begin{pmatrix}0&-1\\ \rho_0&0\end{pmatrix},
\qquad
 \boxed{(DT)^2=-\rho_0I_2.}
\tag{79.23}
\]

Therefore after every even number (2m) of reset-transition pairs, the two
singular values on this plane are exactly equal to (\rho_0^m).  The best
rank-one relative Frobenius error is (1/\sqrt2), independent of depth.
After every odd pair a symmetric input is pure contrast, so projection onto
the advertised symmetric mode has zero correlation.  Identity collision
projection already realizes the counterexample; any deployed seam that can
induce this two-dimensional rotation has no spectral-collapse certificate.

### 79.7 Binding deep-tail reopen criterion

A rank-one deep-tail carrier is lawful only when all of the following hold:

1. the per-layer biased/heterogeneous reset ratios satisfy (79.10)--(79.14);
2. the combined transition and literal collision projector satisfy either
   the exact invariance receipt (79.16) or the injected-tail gate
   (79.18)--(79.21), in both forward and adjoint directions;
3. the live source and panel costates have a certified nonzero symmetric
   component--pure contrast is not rescued by small absolute norm;
4. the resulting (z_d) is below the preregistered value-weighted tolerance,
   and every discarded contrast action is charged to the reset error budget;
5. the retained symmetric scalar/channel is literally resident or supplied
   inside the Cap49/billing allowance.

For a requested correlation (c_*), the exact vector gate is

\[
 z_d\le\sqrt{c_*^{-2}-1}.
\tag{79.24}
\]

This is a genuine fail-fast screen.  If (79.16) holds, the analytic product
(79.17) decides the tail immediately.  If it fails but (79.19) holds,
(79.20) gives the irreducible floor.  If the discriminant fails, the
symmetric gain vanishes, or the Section 79.23 rotation is not excluded, the
deep rank-one route is closed without a long numerical campaign.

Graph-ready split:

- Q211_RESET_SYMMETRIC_SPECTRAL_TAIL -- **solved exact** by
  (79.1)--(79.10);
- Q211_RESET_BIASED_DK_GATE -- **solved analytic** by
  (79.11)--(79.14);
- Q211_RESET_TRANSITION_INVARIANCE -- **exact conditional GO** only under
  (79.16);
- Q211_RESET_INJECTED_TAIL_FLOOR -- **solved sharp bound** by
  (79.18)--(79.21);
- Q211_RESET_ROTATION_REEXCITATION -- **closed counterexample** by
  (79.22)--(79.23);
- Q211_RESET_DEEP_RANK1_REOPEN -- **open deployed receipt**, requiring the
  five items above at every live seam.

## 80. Complete-old audit of the Price jet: pairing minus trace, typed
row span, and the cheapest lawful residual

Section 77 proves an exact quadratic correlation formula, but its statement
that the formula is novel for the carried-state-only comparator is too broad.
There are two different questions:

1. **source ownership:** are the coefficients needed to evaluate the formula
   present in the old state? and
2. **typed readout ownership:** does the literal CP4/Star4/pair/Q24 interface
   already emit the same strict-q211 tensor or endpoint action?

The first answer is always yes: the Price jet is a deterministic polynomial
of resident correlation and univariate reset coefficients.  The second answer
requires the exact row-span audit below.  In particular, source availability
does not make a new nonlinear q211 readout free.

### 80.1 Exact rooted-star and open-path trace words

Let \(R\) be the standardized preactivation correlation, let
\(r_i=R_{i,:}^{T}\), and let \(D_x=\operatorname{Diag}(x)\).  On the
strict \(i;jk\) slice, the quadratic part of (77.8) is the symmetric matrix

\[
 \boxed{
 Q_i^{P}=
 d_{2i}(r_i\odot b_1)(r_i\odot b_1)^T
 +d_{1i}\{D_{r_i\odot b_2}RD_{b_1}
             +D_{b_1}RD_{r_i\odot b_2}\}.}
 \tag{80.1}
\]

The first term is the rooted two-edge star \(i-j,i-k\).  The second is the
Jordan symmetrization of the two open paths \(i-j-k\) and \(i-k-j\).  Thus
(80.1) is already the complete CP-query trace-word decomposition: rank-one
root conjugation plus a symmetrized covariance path.  It contains no hidden
dense three-tensor.

This decomposition can also be written on one unordered triple.  Put

\[
 C^{(1)}=D_{b_1}RD_{b_1},\qquad
 s_i=C^{(1)}_{ij}C^{(1)}_{ik},\quad
 s_j=C^{(1)}_{ij}C^{(1)}_{jk},\quad
 s_k=C^{(1)}_{ik}C^{(1)}_{jk},
 \tag{80.2}
\]

and, wherever the displayed denominators are nonzero,

\[
 \eta_i={d_{2i}\over b_{1i}^2},\qquad
 \theta_{ij}={d_{1i}b_{2j}\over b_{1i}b_{1j}^2}.
 \tag{80.3}
\]

For \(q=(Q^P_{i;jk},Q^P_{j;ik},Q^P_{k;ij})^T\) and
\(s=(s_i,s_j,s_k)^T\),

\[
 \boxed{
 q=M_{ijk}^{P}s,qquad
 M_{ijk}^{P}=
 \begin{pmatrix}
  \eta_i&\theta_{ij}&\theta_{ik}\\
  \theta_{ji}&\eta_j&\theta_{jk}\\
  \theta_{ki}&\theta_{kj}&\eta_k
 \end{pmatrix}.}
 \tag{80.4}
\]

Consequently a single covariance-pairing coefficient can own at most the
common diagonal part of (80.4).  Heterogeneous diagonal modulation and every
off-diagonal path coefficient require additional literal rows.

At zero bias and unit scale, put

\[
 c={\pi-4\over8\pi}.
 \tag{80.5}
\]

Define the two symmetric covariance words explicitly by

\[
 (\mathcal P_R)_{abcd}
 =R_{ab}R_{cd}+R_{ac}R_{bd}+R_{ad}R_{bc},
 \tag{80.6}
\]

and

\[
\begin{aligned}
 (\mathcal Z_R)_{abcd}={}&
 \delta_{ab}R_{cd}+\delta_{ac}R_{bd}+\delta_{ad}R_{bc}\\
 &+\delta_{bc}R_{ad}+\delta_{bd}R_{ac}+\delta_{cd}R_{ab}.
\end{aligned}
 \tag{80.7}
\]

For distinct \(i,j,k\),

\[
 (\mathcal P_R)_{iijk}=R_{jk}+2R_{ij}R_{ik},\qquad
 (\mathcal Z_R)_{iijk}=R_{jk}.
 \tag{80.8}
\]

The exact zero-bias Price jet therefore obeys the sharper identity

\[
 \boxed{
 \Pi_{211}\Delta^{P,(2)}
 ={c\over2}\Pi_{211}(\mathcal P_R-\mathcal Z_R).}
 \tag{80.9}
\]

This corrects the tempting but false statement that the zero-bias jet is
merely the old pairing plane.  The pairing-minus-trace subtraction is what
removes the spurious \(R_{jk}\) leg and leaves \(cR_{ij}R_{ik}\).

### 80.2 Literal CP4/Star4/Q24 annihilation census

The old objects act on (80.1)--(80.9) as follows.

1. A coordinate Star4 tensor \(E_{31}(H)\) is supported on at most two
   coordinate labels.  Hence

   \[
    \Pi_{211}E_{31}(H)=0. \tag{80.10}
   \]

   In particular Star4 does **not** own the trace lift
   \(\mathcal Z_R\), whose \(iijk\) entry is nonzero on three labels.

2. A coordinate CP4 birth \(e_a^{\otimes4}\) also has zero strict-q211
   slice.  A genuinely inherited CP4 state is the only exception: its exact
   slice is

   \[
    (K^{CP4})_{iijk}=\sum_r\gamma_r u_{ri}^2u_{rj}u_{rk}. \tag{80.11}
   \]

   Those literal columns must be included in the old projector.  Nominal
   CP rank is not ownership of either word in (80.9).

3. The retained covariance-pairing state owns \(\mathcal P_C\), hence the
   appropriately scaled \(\mathcal P_R\) quadratic response.  It does not
   separately retain \(\mathcal Z_R\).  Indeed \(\mathcal P_R\) and
   \(\mathcal Z_R\) are linearly independent on any open correlation set:
   take \(R_{jk}=0\) but \(R_{ij}R_{ik}\ne0\), and then take
   \(R_{ij}=R_{ik}=0\) but \(R_{jk}\ne0\).

4. The literal seven-power Q24 kernels are bivariate edge functions.  Holding
   their supplied K3/K21/K4/K31/K22 planes fixed, for two different adjacent
   correlation edges \(e\ne f\), every such local row satisfies

   \[
    \partial_{R_e}\partial_{R_f}Q24_{ab}=0. \tag{80.12}
   \]

   Its \(L_3,L_4,\tfrac12L_3^2\) weights remain indexed by the same endpoint
   pair, and its output types are the 40/31/22 collision arrays.  There is no
   literal map lifting a pair array \(A_{jk}\) to the strict-q211 tensor
   \(\operatorname{Sym}(I\otimes A)_{iijk}=A_{jk}\).  Thus Q24 contains the
   scalar data \(R_{jk}\) but does not emit \(\mathcal Z_R\) at this seam.

It follows that the literal first-reset D11 interface, where the CP4 births
are coordinate-supported, owns the \(\mathcal P_R\) summand of (80.9) but not
the \(\mathcal Z_R\) summand.  At later resets inherited CP4 columns can own
some or all of the latter only by the explicit row-span test below.  This is
a typed-interface statement, not an information-theoretic one.

There is also a sharp warning against declaring the trace word to be a cheap
CP4 tensor.  Its diagonal quartic polynomial is

\[
 \mathcal Z_R[x,x,x,x]=6(x^Tx)(x^TRx). \tag{80.12a}
\]

The middle \(2|2\) catalecticant of (80.12a) is generically full rank
\(N=n(n+1)/2\).  Indeed its determinant is polynomial in \(R\), and at
\(R=I\) the induced map on \(\operatorname{Sym}^2\) is, up to positive
normalization, \(A\mapsto2A+\operatorname{tr}(A)I\), which is invertible.
Each signed CP ray \(u^{\otimes4}\) contributes rank one to this
catalecticant.  Hence an exact **standalone full-tensor** CP4 representation
of \(\mathcal Z_R\) generically needs at least \(N\) rays.  This lower bound
does not replace the more favorable strict-q211 quotient by resident
pair/star collision planes; that quotient is exactly why (80.13), rather
than standalone Waring rank, is binding.

### 80.3 Necessary-and-sufficient complete-old theorem

Let \(\mathcal M_2\) be the finite space of all linear edge and adjacent-edge
quadratic jets on strict-q211 slots.  Let \(J_{\rm old}^{(2)}\) contain,
with their literal types and collision sewing,

- every inherited CP4 slice (80.11);
- the covariance-pairing response;
- every actual Q24/EC4 row, rather than its family name;
- all K4/K31/K22 and Star4 collision rows; and
- any already-billed endpoint covariance-polynomial read.

Let \(G_P^{(2)}\) be the map (80.1), and write
\(J=J_{\rm old}^{(2)}\).  Then complete-old ownership is
equivalent to each of

\[
 \boxed{
 \ker J_{\rm old}^{(2)}\subseteq\ker G_P^{(2)}
 \iff G_P^{(2)}\in\operatorname{row}J_{\rm old}^{(2)}
 \iff G_P^{(2)}(I-J^+J)=0.}
 \tag{80.13}
\]

For a positive endpoint metric \(\Sigma\), its exact Schur novelty is

\[
 S_P=G_P\Sigma G_P^T
 -G_P\Sigma J^T(J\Sigma J^T)^+J\Sigma G_P^T. \tag{80.14}
\]

This is the only lawful duplicate test.  In particular, at zero bias, since
\(\mathcal P_R\in\mathcal O_{\rm old}\),

\[
 \boxed{
 \Pi_{\mathcal O_{\rm old}^{\perp}}\Delta^{P,(2)}
 =-{c\over2}\Pi_{\mathcal O_{\rm old}^{\perp}}\mathcal Z_R.}
 \tag{80.15}
\]

Thus the zero-bias lane closes as a duplicate exactly when the complete old
space contains the identity--covariance trace lift.  It is an exact trace
residual otherwise.  At the coordinate-birth first reset (80.10)--(80.12)
make that residual nonzero.  At a later reset, (80.11) must be projected
before claiming credit.

For biased heterogeneous units, quotienting the common pairing direction
from (80.4) leaves, in the minimal scalar-pairing gauge,

\[
 M_{ijk}^{\perp}
 =\operatorname{Diag}(\eta_i-\bar\eta,
                       \eta_j-\bar\eta,
                       \eta_k-\bar\eta)
  +(\theta_{ab}1_{a\ne b})_{a,b\in\{i,j,k\}},
 \tag{80.16}
\]

where \(\bar\eta\) is the complete-old weighted projection coefficient, not
an unweighted average.  More old coefficient rows simply replace
\(\bar\eta I\) by their exact projector.  Equation (80.16) isolates the two
possible biased novelties: anchor-dependent root amplitude and the open-path
Jordan word.  Neither is certified novel until (80.13) includes inherited
CP4 and every EC4 row.

### 80.4 No covariance-fixing Price tangent exists

The Section 56 Vandermonde tangent cannot certify Price-jet novelty.  Write

\[
 G_P=\Phi(R,b_1,b_2,d_1,d_2).
 \tag{80.17}
\]

For every lawful tangent \(h\),

\[
 DR[h]=Db_1[h]=Db_2[h]=Dd_1[h]=Dd_2[h]=0
 \quad\Longrightarrow\quad DG_P[h]=0. \tag{80.18}
\]

Therefore there is no tangent which fixes **every old input**, including the
resident covariance and marginal reset data, while changing the Price jet.
Any claim to the contrary confuses missing source information with a missing
nonlinear readout.

There is nevertheless an exact readout-level annihilator whenever
(80.13) fails.  In the endpoint metric take any nonzero column of

\[
 h_P=\Sigma^{1/2}
 \bigl(I-(J\Sigma^{1/2})^+(J\Sigma^{1/2})\bigr)
 \Sigma^{1/2}G_P^T. \tag{80.19}
\]

Then \(Jh_P=0\) and \(G_Ph_P\ne0\).  It fixes the old *emitted rows*, not the
raw covariance generator.  Similarly, a two-parameter correlation surface
\(R(t,u)=R_0+tH+uK\), with \(H\) on edge \(ij\) and \(K\) on edge \(ik\),
has a nonzero mixed rooted-star derivative while every edgewise Q24 row has
zero mixed derivative by (80.12).  The pairing row must still be projected;
this surface is not an information-novelty witness.

### 80.5 Cheapest exact consumer of the residual trace lift

No forward q211 tensor is required for (80.15).  Let
\(\bar R=R-\operatorname{Diag}(\operatorname{diag}R)\), and let one symmetric
CP costate atom be

\[
 L_i[j,k]=\alpha_i\operatorname{Sym}_{jk}(\beta_j\gamma_k).
\]

In the ordered strict-q211 convention, direct exclusion of \(j=k\), \(i=j\),
and \(i=k\) gives

\[
 \boxed{
\begin{aligned}
 \langle L,\Pi_{211}\mathcal Z_R\rangle
 ={}&(\mathbf1^T\alpha)\,\beta^T\bar R\gamma\\
 &-(\alpha\odot\beta)^T\bar R\gamma
  -(\alpha\odot\gamma)^T\bar R\beta .
\end{aligned}}
 \tag{80.20}
\]

The formula uses two covariance matvecs for a generic atom and one when
\(\beta=\gamma\); collision-normalization factors follow the costate's chosen
symmetrization convention.  If those matvecs are already resident, its
incremental dense cost is zero.  Otherwise, across ten active seams,

\[
 C_Z^{generic}=20R_c/n,qquad C_Z^{sym}=10R_c/n, \tag{80.21}
\]

so a one-equivalent screen gives \(R_c\le51\) generically and
\(R_c\le102\) symmetrically at \(n=1024\).  This improves the generic biased
Price ceilings (77.13), but it is not a free forward carrier.  It uses zero
new source bytes because \(R\) is resident.

For the biased residual (80.16), the root part retains the two-matvec action
and a genuinely surviving open-path part needs the third matvec in (77.11).
Only the complete-old-projected weights may be scored.  When the Section 49
affine surface is in the declared comparator, all of (80.1) is already owned
and the residual is identically zero.

Graph-ready split:

- `AFFINE_Q211_PRICE_SOURCE_INFORMATION` -- **closed duplicate:** all source
  inputs are resident, by (80.17)--(80.18);
- `AFFINE_Q211_PRICE_LITERAL_ROWSPAN` -- **solved theorem:** use
  (80.13)--(80.14), including inherited CP4 rows rather than nominal rank;
- `AFFINE_Q211_PRICE_ZERO_BIAS_PAIR_MINUS_TRACE` -- **solved exact:**
  (80.9);
- `AFFINE_Q211_PRICE_TRACE_OLD_OWNER` -- **literal D11 no-owner at coordinate
  birth / later conditional:** Q24 and Star4 annihilate it, inherited CP4 must
  be projected;
- `AFFINE_Q211_PRICE_TRACE_CP_ACTION` -- **solved exact:** two matvecs per
  generic atom, one per symmetric atom, by (80.20)--(80.21);
- `AFFINE_Q211_PRICE_BIASED_PATH_RESIDUAL` -- **conditional open:** exact
  residual is (80.16) after the complete-old projector;
- `AFFINE_Q211_PRICE_WITH_SECTION49` -- **closed duplicate:** zero additive
  credit.

## 81. Independent Gaussian-rest heat response: exact q211 interpolation,
target rank, and ownership

The independent Gaussian rest admits an exact path which is different from
both the full-AQ ReLU surface and the non-Gaussian true residual.  Let

\[
 A=g^{AQ},\qquad \xi\sim N(0,\Sigma_G),\qquad \xi\perp A,
\]

\[
 U(t)=A+\sqrt t\,\xi,qquad Y_a(t)=U_a(t)_+,qquad 0\le t\le1.
\tag{81.1}
\]

Compile collision sewing and complete-old projection into the symmetric
q211 costate (L) as in (69.1), and define

\[
 S_L(t)=\sum_{i,j,k}L_i[j,k]
 \kappa_4(Y_i(t),Y_i(t),Y_j(t),Y_k(t)).
\tag{81.2}
\]

Then (S_L(0)) is the full-AQ surface, while
(S_L(1)-S_L(0)) is only the response to the declared independent Gaussian
rest.

### 81.1 Cumulant heat identity

For functions (f_1,\ldots,f_m) of a common vector, put

\[
 \mathcal L_{\Sigma}f={1\over2}\operatorname{tr}
 (\Sigma\nabla^2f),
 \qquad
 \Gamma_{\Sigma}(f,g)=\nabla f^T\Sigma\nabla g.
\tag{81.3}
\]

Differentiating the moment--partition formula for a cumulant and using the
product rule for (\mathcal L_\Sigma) gives

\[
\boxed{\begin{aligned}
 {d\over dt}\kappa_m(f_1(U(t)),\ldots,f_m(U(t)))
 ={}&\sum_{r=1}^m
 \kappa_m(f_1,\ldots,\mathcal L_{\Sigma_G}f_r,\ldots,f_m)\\
 &+\sum_{r<s}\kappa_{m-1}
 \left(\Gamma_{\Sigma_G}(f_r,f_s),
       (f_u)_{u\ne r,s}\right).
\end{aligned}}
\tag{81.4}
\]

The second term is an ordinary cumulant after merging slots (r,s).  This is
why replacing it by a cumulant of (f_r') and (f_s') separately would be
wrong; for (m=2), for example, it is the expectation of their gradient
product.

Equation (81.4) is exact for smooth functions.  For ReLU it holds by smooth
approximation/distributional differentiation for (t>0), and extends to
(t=0) when the required AQ boundary densities exist and the displayed
derivatives are integrable.

### 81.2 Exact repeated-slot q211 derivative

Write, under the law at time (t),

\[
 I_a=\mathbf1_{\{U_a(t)>0\}},\qquad D_a=\delta(U_a(t)),
\tag{81.5}
\]

and abbreviate (K_{i;jk}(t)=\kappa_4(Y_i,Y_i,Y_j,Y_k)).  Since

\[
 \mathcal L_{\Sigma_G}Y_a={1\over2}(\Sigma_G)_{aa}D_a,
 \qquad
 \Gamma_{\Sigma_G}(Y_a,Y_b)=(\Sigma_G)_{ab}I_aI_b,
\tag{81.6}
\]

(81.4) gives the exact specialized response

\[
\boxed{\begin{aligned}
 \dot K_{i;jk}={}&
 (\Sigma_G)_{ii}\{
   \kappa_4(D_i,Y_i,Y_j,Y_k)+\kappa_3(I_i,Y_j,Y_k)\}\\
 &+{(\Sigma_G)_{jj}\over2}\kappa_4(Y_i,Y_i,D_j,Y_k)
  +{(\Sigma_G)_{kk}\over2}\kappa_4(Y_i,Y_i,Y_j,D_k)\\
 &+2(\Sigma_G)_{ij}\kappa_3(I_iI_j,Y_i,Y_k)
  +2(\Sigma_G)_{ik}\kappa_3(I_iI_k,Y_i,Y_j)\\
 &+(\Sigma_G)_{jk}\kappa_3(I_jI_k,Y_i,Y_i).
\end{aligned}}
\tag{81.7}
\]

The first line contains two distinct effects of the repeated anchor: heat
acting on either ReLU factor produces the delta insertion, while heat
joining the two copies produces (I_i^2=I_i).  All coefficients and
multiplicities in (81.7) are therefore fixed; no fitted tangent weight is
present.

For each strict word, define the symmetric response matrix
(\mathcal G^{i;jk}(t)) by its only nonzero entries

\[
\begin{aligned}
 \mathcal G^{i;jk}_{ii}&=2\{
 \kappa_4(D_i,Y_i,Y_j,Y_k)+\kappa_3(I_i,Y_j,Y_k)\},\\
 \mathcal G^{i;jk}_{jj}&=\kappa_4(Y_i,Y_i,D_j,Y_k),\\
 \mathcal G^{i;jk}_{kk}&=\kappa_4(Y_i,Y_i,Y_j,D_k),\\
 \mathcal G^{i;jk}_{ij}&=2\kappa_3(I_iI_j,Y_i,Y_k),\\
 \mathcal G^{i;jk}_{ik}&=2\kappa_3(I_iI_k,Y_i,Y_j),\\
 \mathcal G^{i;jk}_{jk}&=\kappa_3(I_jI_k,Y_i,Y_i).
\end{aligned}
\tag{81.8}
\]

Let

\[
 \mathcal G_L(t)=\sum_{i,j,k}L_i[j,k]\mathcal G^{i;jk}(t).
\tag{81.9}
\]

Then the complete target response has the compact exact form

\[
\boxed{
 \dot S_L(t)={1\over2}\langle\Sigma_G,\mathcal G_L(t)\rangle_F,
 \qquad
 S_L(1)-S_L(0)={1\over2}\int_0^1
 \langle\Sigma_G,\mathcal G_L(t)\rangle_Fdt.}
\tag{81.10}
\]

This is the heat analogue of the one-scalar surface (69.4).  Equivalently,
one may put (t\Sigma_G) into (69.6)--(69.10) and average over the AQ
residual (A): each q211 word requires only its trivariate AQ marginal, not a
four-coordinate law.  That exact conditional representation is still a
signed response-surface oracle, not a consequence of covariance alone.

### 81.3 First tangent and rigorous remainder

When the AQ boundary jet in (81.8) exists, the exact first-order heat
tangent is

\[
 \boxed{T_G^{(1)}={1\over2}
 \langle\Sigma_G,\mathcal G_L(0)\rangle_F.}
\tag{81.11}
\]

Let (\mathscr S_L(C)) denote the same compiled cumulant surface after adding
an independent (N(0,C)) to (A), so that (S_L(t)=\mathscr S_L(t\Sigma_G)).
Taylor's theorem gives the exact remainder identity

\[
\boxed{
 S_L(1)-S_L(0)-T_G^{(1)}
 =\int_0^1(1-t)
 D^2\mathscr S_L(t\Sigma_G)[\Sigma_G,\Sigma_G]dt.}
\tag{81.12}
\]

For the nuclear-norm covariance geometry define

\[
 M_{2,L}:=\sup_{0\le t\le1}
 \sup_{\|H\|_*\le1,\ \|K\|_*\le1}
 |D^2\mathscr S_L(t\Sigma_G)[H,K]|.
\tag{81.13}
\]

Then

\[
 \boxed{
 |S_L(1)-S_L(0)-T_G^{(1)}|
 \le {M_{2,L}\over2}\|\Sigma_G\|_*^2.}
\tag{81.14}
\]

There is a sometimes sharper response-drift form.  From (81.10),

\[
 R_G={1\over2}\int_0^1
 \langle\Sigma_G,\mathcal G_L(t)-\mathcal G_L(0)\rangle_Fdt.
\tag{81.15}
\]

If

\[
 \|\dot{\mathcal G}_L(t)\|_{op}
 \le\Gamma_L\|\Sigma_G\|_*
\tag{81.16}
\]

on the path, then

\[
 \boxed{|R_G|\le{\Gamma_L\over4}\|\Sigma_G\|_*^2.}
\tag{81.17}
\]

The constants in (81.13) or (81.16) are target-conditioned boundary-density
receipts.  ReLU's next heat derivatives contain derivatives of delta
densities.  If (A) has a threshold atom, lacks the requisite joint density,
or makes these quantities diverge as (t\downarrow0), the first-order route
is closed; the exact integral (81.10) may still exist, but (81.11) cannot be
promoted with a fictitious smooth remainder.

### 81.4 Target-conditioned covariance rank and cost

For the tangent define (k_0=\operatorname{rank}\mathcal G_L(0)).  If its
factorization is born directly as

\[
 \mathcal G_L(0)=U\Theta U^T,qquad U\in\mathbb R^{n\times k_0},
\tag{81.18}
\]

then

\[
 T_G^{(1)}={1\over2}\operatorname{tr}
 \{\Theta U^T\Sigma_GU\}.
\tag{81.19}
\]

Thus a dense resident covariance requires only (k_0) covariance matvecs
once the response factors are supplied.  For the exact heat path, the
corresponding intrinsic common-support rank is

\[
 k_{heat}:=\dim\operatorname{span}_{0\le t\le1}
            \operatorname{im}\mathcal G_L(t).
\tag{81.20}
\]

If (\mathcal G_L(t)=U\Theta(t)U^T) with one (n\times k_{heat}) matrix (U),
then every explicit generator contraction in (81.10) uses the single
projected covariance (U^T\Sigma_GU).  Across ten seams the optimistic
covariance-action bill is

\[
 C_{heat,cov}={10k_{heat}\over n},
 \qquad n=1024:\quad k_{heat}\le102
\tag{81.21}
\]

for one width-GEMM equivalent, before constructing (U,\Theta(t)), AQ
surface evaluation, quadrature/closed integration, hooks, and writes.  A
generic dense q211 costate makes (\mathcal G_L(t)) full rank through the
((j,k))-entry in (81.8); there is no universal small-(k_{heat}) theorem.
The actual Section 73 response factors are therefore binding.

If (\Sigma_G=BMB^T) has rank (r_G), then

\[
 U^T\Sigma_GU=(B^TU)^TM(B^TU),
\tag{81.22}
\]

so two supplied low-rank factors meet in (O(nk_{heat}r_G)) arithmetic
without a dense product.  Low covariance rank by itself does not construct
the nonlinear response factors.

There is also an exact target-conditioned covariance truncation gate.  For
any positive semidefinite rank-(r) approximation (\widehat\Sigma), set
(E=\Sigma_G-\widehat\Sigma) and interpolate
(C_s=\widehat\Sigma+sE).  The same heat identity gives

\[
\boxed{
 \mathscr S_L(\Sigma_G)-\mathscr S_L(\widehat\Sigma)
 ={1\over2}\int_0^1
 \langle E,\mathcal G_L(C_s)\rangle_Fds,}
\tag{81.23}
\]

and therefore

\[
 \boxed{
 |\mathscr S_L(\Sigma_G)-\mathscr S_L(\widehat\Sigma)|
 \le{1\over2}\|E\|_*
 \sup_{0\le s\le1}\|\mathcal G_L(C_s)\|_{op}.}
\tag{81.24}
\]

The smallest (r) satisfying the endpoint tolerance in (81.24), not the
variance explained by an unweighted covariance SVD, is the lawful
target-conditioned covariance rank.  A rank-(r) factor expresses each heat
contraction as (r) directional quadratic forms.  With a dense response
application this alone costs (10r/n) equivalents across ten seams, giving
the same necessary screen (r\le102) at width 1024.  Equations (81.21) and
(81.24) are necessary cost/error gates, not claims that the AQ response
oracle is free.

### 81.5 Exact covariance ownership and the three-way residual split

Because (\xi) is declared Gaussian, (\Sigma_G) determines its entire law.
Consequently, **conditional on the full relevant AQ trivariate laws and
literal independence**, no stochastic information about the rest beyond
(\Sigma_G) is needed for (81.10).  This is the positive ownership statement.

Covariance alone does not determine the response matrix.  A direct witness
keeps even the AQ first two moments fixed.  Let (Z_m) have the smooth
symmetric mixture law

\[
 {1\over2}N(-m,1-m^2)+{1\over2}N(m,1-m^2),qquad0<m<1,
\tag{81.25}
\]

and put

\[
 A_a=cZ_m+\sqrt{1-c^2}\,\varepsilon_a,qquad
 0<c<1,qquad \varepsilon_a\stackrel{iid}{\sim}N(0,1).
\tag{81.26}
\]

For every (m), (A) has the same zero mean, unit marginal variance, and
off-diagonal covariance (c^2), but its joint density and threshold jets vary
with (m).  In particular the delta and indicator cumulants in (81.8) vary.
With one fixed (\Sigma_G), (T_G^{(1)}) and the full heat response therefore
vary although all AQ and rest covariances agree.  This proves that a
covariance-only D10 state does not own (\mathcal G_L(t)).  A finite list of
AQ moments or the single value (S_L(0)) does not repair the missing boundary
surface.

The binding ownership decomposition is

\[
\boxed{\begin{aligned}
 S_L(Y^{true})-S_L(Y^{AQ})={}&
 \underbrace{S_L((A+\xi)_+)-S_L(A_+)}_{
 \text{independent Gaussian-rest response, (81.10)}}\\
 &+\underbrace{S_L(Y^{true})-S_L((A+\xi)_+)}_{
 \text{non-Gaussian/dependent residual gap}}.
\end{aligned}}
\tag{81.27}
\]

The first term receives zero new credit if the billed Section 49 evaluator
already evaluates the full convolved surface at (C=\Sigma_G); a Section 49
value only at (C=0) owns the baseline but not the response.  The second term
is not owned by (\Sigma_G), even if the true residual has that same
covariance: it requires the signed mixed surface of Section 68/69, and the
pairwise-law witness (68.3)--(68.6) permits its sign to change while every
covariance receipt remains fixed.  If the residual is not literally
independent Gaussian, substituting (81.10) for the second term is unlawful.

All tangent truncation, covariance-rank error from (81.24), response-factor
error, and the non-Gaussian gap must enter (\epsilon_f) in (66.30).

Graph-ready split:

- `GAUSSIAN_REST_Q211_HEAT_IDENTITY` -- **solved exact** by
  (81.4)--(81.10), conditional on literal independence/Gaussianity;
- `GAUSSIAN_REST_Q211_FIRST_TANGENT` -- **solved algebraically** by
  (81.11), but requires the AQ boundary jet in (81.8);
- `GAUSSIAN_REST_Q211_TANGENT_REMAINDER` -- **conditional** on the finite
  receipts (81.13) or (81.16);
- `GAUSSIAN_REST_Q211_TARGET_RANK` -- **solved gate** by
  (81.18)--(81.24); deployed (k_{heat}) and target-weighted covariance tail
  remain open;
- `D10_GAUSSIAN_COVARIANCE_INFORMATION` -- **sufficient only together with
  the AQ trivariate response surface**;
- `D10_COVARIANCE_ONLY_Q211_RESPONSE` -- **no-go** by (81.25)--(81.26);
- `GAUSSIAN_REST_WITH_FULL_CONVOLVED_SECTION49` -- **closed duplicate**;
- `GAUSSIAN_REST_WITH_T0_SECTION49_ONLY` -- **new response conditional on
  acquisition, rank, and complete-old value**;
- `TRUE_MINUS_GAUSSIAN_REST_GAP` -- **open/blocking and covariance-invisible**;
- `GAUSSIAN_REST_Q211_OLD_SCHUR_VALUE` -- open blocking receipt.

## 82. Depth-allocated rooted Price provider through heterogeneous q211 resets

Sections 78 and 79 solve two different halves of the same approximation
problem.  Section 78 resolves the source by singleton chaos degree but stops
at one seam.  Section 79 resolves the three q211 anchor orientations through
depth but treats one undifferentiated source.  Combining them gives a
degree-by-birth-depth allocation theorem, and also exposes three independent
obstructions which must not be hidden in one empirical fidelity number:
source-band truncation, orientation loss under transport, and the broad
nonroot diagrams.  Complete-old novelty and billed cost are separate again.

### 82.1 Exact Price shells and the live quotient source

At a live seam (\ell), define the incremental rooted shell

\[
 \Xi_{\ell q}:=\Delta^{root,q}_\ell-
                   \Delta^{root,q-1}_\ell,
 \qquad q\ge1,
 \qquad \Delta^{root,0}_\ell:=0.
\tag{82.1}
\]

Thus (\Xi_{\ell q}) is exactly the sum of the terms in (78.7) with
(\max(a,b)=q), and

\[
 \boxed{\Delta^{root,Q}_\ell=\sum_{q\le Q}\Xi_{\ell q},\qquad
 \Delta^{root}_\ell-\Delta^{root,Q}_\ell
       =\sum_{q>Q}\Xi_{\ell q}.}
\tag{82.2}
\]

This shell convention matters computationally.  Once all bands below (q)
are resident, adding shell (q) needs only the new degree-(q) Schur action;
all cross terms between (q) and lower degrees are local entries of
(A_i^{(Q)}).  At a centered gate a formally present shell with zero signed
singleton band is simply inactive and has zero pass cost.

Stack the three choices of repeated anchor in one orientation fiber
(\mathcal O\simeq\mathbb R^3), and let (\xi_{\ell q}) denote the resulting
three-oriented version of (\Xi_{\ell q}).  If (\Pi^{old}_\ell) is the
literal already-owned source map, the only source allowed to receive new
credit is

\[
 \boxed{g_{\ell q}:=N_\ell\xi_{\ell q},\qquad
        N_\ell:=I-\Pi^{old}_\ell.}
\tag{82.3}
\]

The map (N_\ell) need not preserve the symmetric orientation.  It is part
of the seam, not an after-the-fact discount.  In particular, a source can
be algebraically nonzero before (N_\ell) and have zero new value, while an
anisotropic old projection can turn a symmetric source into a contrast
source.

### 82.2 Exact shellwise reset transport and a true one-state error bound

Use the moving dominant reset direction at every layer, in the symmetrized
metric of (79.12).  Write its orthogonal projectors as (P_m) and
(Q_m=I-P_m).  Normalize reset (m+1) by its dominant eigenvalue so that

\[
 D_{m+1}P_{m+1}=P_{m+1},\qquad
 \|D_{m+1}Q_{m+1}\|\le\eta_{m+1}.
\]

All eigenvector drift, dense coordinate transport, collision sewing, and
complete-old projection between the two resets are included in (T_m).  For
one degree shell born at (\ell), put (x_m=p_m+c_m), where
(p_m=P_mx_m) and (c_m=Q_mx_m).  The exact propagation is

\[
\begin{aligned}
 p_{m+1}&=\lambda_{\parallel,m+1}
 P_{m+1}T_m(p_m+c_m),\\
 c_{m+1}&=\lambda_{\parallel,m+1}
 D_{m+1}Q_{m+1}T_m(p_m+c_m).
\end{aligned}
\tag{82.4}
\]

With ((a_m,\beta_m,\gamma_m,h_m)) defined as in (79.15), (82.4)
implies the rigorous nonnegative envelope

\[
 \boxed{
 \binom{P_{m+1}^{\#}}{C_{m+1}^{\#}}
 \preceq |\lambda_{\parallel,m+1}|
 \begin{pmatrix}
  a_m&\beta_m\\
  \eta_{m+1}\gamma_m&\eta_{m+1}h_m
 \end{pmatrix}
 \binom{P_m^{\#}}{C_m^{\#}},}
\tag{82.5}
\]

starting from ((P_\ell^{\#},C_\ell^{\#})
=(\|P_\ell g_{\ell q}\|,\|Q_\ell g_{\ell q}\|)).  This transports every
root degree through an arbitrary heterogeneous suffix.  It is often sharper
than multiplying full operator norms, and it retains the birth-depth
dependence exactly.

There is an important distinction between projecting the *full* state at
the endpoint and actually carrying one scalar through the suffix.  The
latter uses

\[
 \widehat p_{m+1}=\lambda_{\parallel,m+1}
 P_{m+1}T_mP_m\widehat p_m,
 \qquad \widehat p_\ell=P_\ell g_{\ell q}.
\tag{82.6}
\]

Let (e_m=p_m-\widehat p_m).  A closed, rigorous error recursion for this
genuine rank-one carrier is

\[
 \boxed{
 \begin{pmatrix}\widehat P_{m+1}\\E_{m+1}\\C_{m+1}\end{pmatrix}
 \preceq |\lambda_{\parallel,m+1}|
 \begin{pmatrix}
 a_m&0&0\\
 0&a_m&\beta_m\\
 \eta_{m+1}\gamma_m&\eta_{m+1}\gamma_m&
                    \eta_{m+1}h_m
 \end{pmatrix}
 \begin{pmatrix}\widehat P_m\\E_m\\C_m\end{pmatrix},}
\tag{82.7}
\]

with initial vector
((\|P_\ell g_{\ell q}\|,0,\|Q_\ell g_{\ell q}\|)).  At the endpoint,

\[
 \boxed{
 \|M_{L\leftarrow\ell}g_{\ell q}
   -\widehat M^{(1)}_{L\leftarrow\ell}P_\ell g_{\ell q}\|
 \le\sqrt{E_L^2+C_L^2}.}
\tag{82.8}
\]

The square root is valid because the two errors occupy orthogonal
symmetric and contrast subspaces in the symmetrized metric.  Euclidean
deployment bounds must include the associated (S_m) condition factors in
the blocks of (T_m).

Under the exact invariance receipt (\beta_m=\gamma_m=0), (82.7) reduces to

\[
 E_L=0,\qquad
 C_L\le C_\ell
 \prod_{m=\ell}^{L-1}
 |\lambda_{\parallel,m+1}|\eta_{m+1}h_m,
\tag{82.9}
\]

\[
 \widehat P_L=\widehat P_\ell
 \prod_{m=\ell}^{L-1}
 |\lambda_{\parallel,m+1}|a_m.
\tag{82.10}
\]

Hence the contrast-to-symmetric ratio of shell ((\ell,q)) is bounded by

\[
 {C_L\over\widehat P_L}
 \le {\|Q_\ell g_{\ell q}\|\over\|P_\ell g_{\ell q}\|}
 \prod_{m=\ell}^{L-1}{\eta_{m+1}h_m\over a_m}.
\tag{82.11}
\]

This ratio statement requires (P_\ell g_{\ell q}\ne0) and (a_m>0) along
the suffix.  A pure-contrast live shell has no rank-one signal regardless
of its absolute contraction.

Equations (82.7)--(82.11) are the precise depth discount.  Early contrast
can be cheap because it crosses many certified filters; a shell born at the
last seam cannot borrow that discount.  If (\beta) or (\gamma) is nonzero,
the three-state recursion, rather than (82.11), is binding: contrast can
feed the retained amplitude and can be re-injected from it.

### 82.3 Four separate endpoint ledgers

Let (\Psi) collect the actual endpoint costates or panel reads and equip its
output with the desired weighted Euclidean norm.  Denote the full
three-orientation suffix by (M_{L\leftarrow\ell}) and set

\[
 G^{211}_\ell:=\|\Psi M_{L\leftarrow\ell}N_\ell\|,
 \qquad
 G^{all}_\ell:=\|\Psi M^{all}_{L\leftarrow\ell}N^{all}_\ell\|.
\tag{82.12}
\]

First, the omitted rooted source has the exact entrywise majorant
(\epsilon^{root,Q}_\ell) of (78.19).  In any compatible source norm,

\[
 \boxed{E_{root}\le
 \sum_{\ell=1}^{10}G^{211}_\ell
       \|\epsilon^{root,Q_\ell}_\ell\|.}
\tag{82.13}
\]

This is a source approximation error.  It is present even with an exact
full orientation carrier.

Second, for every retained shell sent through a rank-one carrier, let
(\mathfrak e_{\ell q}^{(1)}:=\sqrt{E_L^2+C_L^2}) be the result of
(82.7).  Its transition/orientation charge is

\[
 \boxed{E_{trans}\le
 \sum_{\ell=1}^{10}\sum_{q\le Q_\ell}
 \|\Psi\|_{\ell q}\,\mathfrak e_{\ell q}^{(1)}.}
\tag{82.14}
\]

Here (\|\Psi\|_{\ell q}) may be restricted to the endpoint subspace reached
by that shell.  Using (79.18) alone in (82.14) is not sufficient: that
ratio bounds the full state's endpoint contrast but misses the symmetric
amplitude error created by the (\beta_m C_m) feedback in (82.7).

Third, the broad sector is not a high-(q) rooted tail.  In the centered case
define the birth-seam majorants

\[
 B^{tri}_\ell=
 \big\|d_{2i}b_{2j}b_{2k}\rho_{ij}\rho_{ik}\rho_{jk}\big\|,
 \qquad
 B^{(4+)}_\ell=
 \left\|{J_{4,\ell}\over24}
 (|x|+|y|+|z|)^4\right\|.
\tag{82.15}
\]

If the triangle is not supplied, the broad charge is

\[
 \boxed{E_{broad}\le
 \sum_{\ell=1}^{10}
 \{G^{211}_\ell B^{tri}_\ell+G^{all}_\ell B^{(4+)}_\ell\}.}
\tag{82.16}
\]

The first factor can use the q211 filter; the fourth-order remainder needs
the norm of the full lawful downstream map.  At biased gates,
(B^{tri}_\ell) is replaced by the norm of every omitted term in (78.20).
Increasing (Q_\ell) never reduces (82.16).  Only an explicit/fused triangle
provider or a small broad-source certificate removes that floor.

Fourth, complete-old novelty is not an approximation error and must not be
inserted into (82.13)--(82.16) as an empirical multiplier.  It is the exact
quotient operation (82.3).  Billing is likewise not fidelity: it is a
constraint on which of these exact sources may be carried.  With exact
arithmetic, the combined analytic endpoint certificate is

\[
 \boxed{\|y_{full}-\widehat y\|
 \le E:=E_{root}+E_{trans}+E_{broad}.}
\tag{82.17}
\]

Any floating-point/provider approximation has its own additive charge; it
does not alter the three mathematical terms above.

The norm certificate gives a sharp correlation and raw-MSE gate.  If
(Y_{min}\le\|y_{full}\|), (E<Y_{min}), and panel vectors are centered in
the same metric, then

\[
 \boxed{\operatorname{corr}(y_{full},\widehat y)
 \ge\sqrt{1-(E/Y_{min})^2}.}
\tag{82.18}
\]

This is sharp: the worst point lies where the radius-(E) error ball is
tangent to a ray from the origin.  If (r_*=r_0-y_{full}) is the residual
under the full correction and (\widehat r=r_0-\widehat y), then

\[
 \boxed{
 (\|r_*\|-E)_+^2\le\|\widehat r\|^2
 \le(\|r_*\|+E)^2.}
\tag{82.19}
\]

Thus an MSE improvement is certified only when
(\|r_*\|+E<\|r_0\|).  In an RMS-normalized corpus, a desired raw MSE
(m_{goal}) requires (\|r_*\|+E<\sqrt{m_{goal}}).  Provider correlation
without this residual-margin inequality does not certify a raw gain.

### 82.4 The exact 49-GEMM cost constraint

For each shell let (R_{c,\ell q}) be its live CP query rank, and let
(\nu_{\ell q}^{(r)}) be the number of distinct Schur-power batches after
literal cross-orientation reuse when carrying orientation rank
(r\in\{0,1,2,3\}).  One selected orientation costs at most two batches for a
generic ((\beta,\gamma)) query and one for a self-paired query, by (78.14).
Two or three orientations cost at most two or three times that number, but
these upper bounds must be replaced by the exact reuse receipt; they are not
automatically tight.  Equation (82.7) certifies (r=1); a selected two-plane
uses the identical block/Duhamel construction with a rank-two (P) and a
rank-one (Q).  Define

\[
 c_{\ell q}^{(r)}:=
 {\nu_{\ell q}^{(r)}R_{c,\ell q}\over n},
 \qquad c_{\ell q}^{(0)}=0.
\tag{82.20}
\]

With the two (d_1) side arms and any explicitly billed triangle option
included, the binding pooled constraint is

\[
 \boxed{
 \sum_{\ell=1}^{10}\left\{
  \sum_{q\le Q_\ell}c_{\ell q}^{(r_{\ell q})}
  +{\mathbf1_{d_{1,\ell}\ne0}R_{c,\ell}\over n}
 \right\}+C_{tri}\le49.}
\tag{82.21}
\]

Admissibility includes the degree-prefix rule: ordinarily
(r_{\ell q}=0) exactly when (q>Q_\ell).  Skipping a lower live shell does
not save its Schur pass if a higher shell still uses that degree in a cross
term; such a choice needs an explicitly billed compute-only mode.

For a common cut and rank at all seams, (82.21) reduces to the ten-seam
formula (78.17).  In general there is no per-seam budget of (4.9): the
remaining depth and the target value differ, so the 49 units must be pooled.
Local (O(nQ^2)) work must be added if the billing auditor counts it.

### 82.5 Target-conditioned rank allocation is a residualized Gram problem

The error certificate chooses a faithful provider.  Raw MSE chooses a
feature subspace, and these are not the same optimization.  Let (X_0) be
the complete-old design on the fixed corpus, (P_0) its weighted orthogonal
projector, and

\[
 r_0=(I-P_0)y,
 \qquad
 Z_{\ell q}^{(r)}=(I-P_0)\Phi_{\ell q}^{(r)},
\tag{82.22}
\]

where (\Phi_{\ell q}^{(r)}) is the endpoint feature block produced by shell
((\ell,q)) with orientation rank (r).  Equation (82.22) is the precise
complete-old novelty test: the block is new iff (Z_{\ell q}^{(r)}\ne0).
It has target value only if it also overlaps the residual.

For any feasible collection (S) of shell/rank choices, concatenate its
residualized blocks into (Z_S), and put

\[
 G_S=Z_S^TWZ_S,\qquad b_S=Z_S^TWr_0.
\tag{82.23}
\]

After the required joint refit, the exact attainable corpus MSE is

\[
 \boxed{
 M(S)=\|r_0\|_W^2-b_S^TG_S^\dagger b_S.}
\tag{82.24}
\]

This statement is exact conditional on the analytic feature blocks
(\Phi_{\ell q}^{(r)}) being fixed while their linear coefficients are
refit.  If F2J also changes the source or transition maps, (82.24) is the
exact inner linear solve at each outer parameter setting, not a claim that
the nonconvex outer refit has been solved.

Consequently the exact target-conditioned allocation theorem is

\[
 \boxed{
 S_*\in\arg\max_{S:\ C(S)\le49}
 b_S^TG_S^\dagger b_S,}
\tag{82.25}
\]

with the mutually exclusive rank choices and prefix degree costs from
(82.21).  This is a grouped quadratic knapsack, not a ranking by source
energy, tensor correlation, or univariate fit.

The exact marginal value of one scalar candidate (j) after a current set
(S) is the Schur-complement quotient

\[
 \boxed{
 \Delta(j\mid S)=
 {\{b_j-G_{jS}G_S^\dagger b_S\}^2\over
  G_{jj}-G_{jS}G_S^\dagger G_{Sj}},}
\tag{82.26}
\]

when the denominator is positive; a block candidate uses the corresponding
Schur-complement pseudoinverse.  Formula (82.26) is the analytic statement
of co-adaptation.  A high-fidelity shell can have zero marginal raw value
because it is old-owned or conditionally collinear, while a smaller shell
can have high value because it hits a residual direction.

There are two useful exact simplifications.

1. If the residualized shell blocks are mutually (W)-orthogonal, their
   gains (b_j^TG_j^\dagger b_j) add.  Equal one-GEMM incremental costs then
   make the optimum the 49 largest positive gains.  If degree cuts must be
   prefixes and each seam's gains are nonincreasing with degree, those 49
   choices automatically respect every prefix.
2. More generally, the additive worst-case certificate can be optimized
   exactly even when the target Gram blocks are not orthogonal.  If
   (\mathcal A_\ell(Q,r)) is the sum of (82.13) and (82.14) charged to seam
   (\ell) for choice ((Q,r)), define

\[
 D_0(b)=0,
 \qquad
 \boxed{D_\ell(b)=
 \min_{Q,r:\ C_\ell(Q,r)\le b}
 \{D_{\ell-1}(b-C_\ell(Q,r))+\mathcal A_\ell(Q,r)\}.}
\tag{82.27}
\]

Then (D_{10}(49)+E_{broad}) is the smallest separable rigorous error bound
available from these choices.  This dynamic program is a theorem about the
certificate, not about the signed exact MSE (82.24).  For unequal integer
costs, sorting gain/cost ratios solves only the fractional relaxation.

A particularly cheap **separable-kernel reopen** occurs when, after
complete-old projection, (i) each shell has a one-dimensional symmetric
orientation, (ii) every seam obeys (\beta=\gamma=0), and (iii) its endpoint
feature Gram is block-orthogonal across the ten births.  Then (82.9)--
(82.11) give every depth weight by scalar products, (82.24) is additive,
and (82.27) gives the exact budgeted cut without any dense H4 state.  Failure
of any one of these three clauses reopens either the three-state transport
or the coupled Gram problem; it cannot be repaired by calling (Q) a rank.

### 82.6 Four no-go statements and the positive finite-cut gate

There is no nontrivial data-free finite-(Q), rank-one guarantee over the
unrestricted affine-ReLU family.

First, the derivative Hankel matrix in Section 78 has infinite rank, so no
finite (Q) is exact generically.  More concretely, for a standardized ReLU
threshold (\tau), its singleton coefficients satisfy

\[
 b_1=\sigma\overline\Phi(\tau),
 \qquad
 b_q=\sigma\phi(\tau)\operatorname{He}_{q-2}(\tau),
 \quad q\ge2.
\tag{82.28}
\]

For every fixed (Q), generic thresholds leave a nonzero shell above (Q),
and (b_{Q+1}/b_2) grows polynomially with (\tau).  Thus no uniform relative
low-band dominance follows without a compact threshold/source-norm gate.
If the correlation domain approaches the boundary excluded by (78.10), the
absolute Price tail (78.19) is not a uniform certificate either.

Second, even if the source tail is small, the rotation (79.22)--(79.23)
makes the two orientation singular values equal after every even pair.
Hence depth alone gives no rank-one discount unless the transition receipt
needed by (82.7) is supplied.

Third, (Q\to\infty) only closes (E_{root}); it never closes the independent
triangle/broad floor (E_{broad}).  A rooted provider cannot converge to the
full affine q211 defect by increasing (Q).

Fourth, for any finite retained source span, if an omitted live shell has an
endpoint component outside that span, there is a costate which annihilates
every retained shell and selects that omitted component.  Its retained
correlation and raw value are zero.  This finite-dimensional annihilator
argument proves that no target-free rank allocation can replace (82.23)--
(82.26).

There is a positive conditional theorem.  On a compact threshold/variance
set satisfying (78.10) with a common (\delta>0), with uniformly bounded
downstream gains, the analytic Price series converges uniformly.  Therefore
(E_{root}\to0) as every (Q_\ell\to\infty).  If in addition (82.7) gives a
uniform transport error below tolerance, (82.16) is below the remaining
error allowance, the old-residual Gram is nonzero, and (82.21) fits, then a
finite depth-dependent set of cuts exists and (82.17)--(82.19) certify it.
These are exactly the assumptions missing from a data-free finite-(Q)
claim.

Graph-ready split:

- `PRICE_RESET_SHELL_PROPAGATOR` -- **solved exact/rigorous** by
  (82.1)--(82.11);
- `PRICE_RESET_TRUE_RANK1_ERROR` -- **solved** by the three-state recursion
  (82.7), which includes contrast-to-symmetric feedback;
- `PRICE_RESET_FOUR_WAY_LEDGER` -- **solved analytic** by
  (82.13)--(82.17), keeping rooted tail, transition loss, broad diagrams,
  and old ownership distinct;
- `PRICE_RESET_CORR_MSE_CERTIFICATE` -- **solved sharp** by
  (82.18)--(82.19);
- `PRICE_RESET_CAP49_ALLOCATION` -- **solved as an exact target-conditioned
  grouped Gram problem** by (82.20)--(82.27);
- `PRICE_RESET_SMALL_SEPARABLE_REOPEN` -- **conditional GO** under the three
  clauses following (82.27);
- `PRICE_RESET_DATA_FREE_FINITE_Q` -- **closed no-go** by (82.28), the
  transition rotation, the broad floor, and the costate annihilator;
- `PRICE_RESET_DEPLOYED_RECEIPTS` -- **open**: per-seam (N_\ell), the moving
  three-by-three blocks, exact pass reuse, tail/broad bounds, and the
  residualized target Gram are all required before assigning raw value.

## 84. Gaussian-AQ specialization: exact orthant heat endpoints, fusion
classes, and the one-pass obstruction

Section 81 permits a non-Gaussian AQ baseline and therefore needs a full AQ
boundary surface.  If the baseline itself is affine Gaussian, the entire
independent-rest heat path is determined by means and covariances and can be
written with trivariate orthant derivatives.  This removes the information
obstruction, but not the dense-query or finite-word obstruction.

Assume

\[
 A\sim N(m,C_0),\qquad \xi\sim N(0,\Sigma_G),\qquad A\perp\xi,
\]

and put

\[
 C(t)=C_0+t\Sigma_G,qquad U(t)\sim N(m,C(t)),qquad Y(t)=U(t)_+.
\tag{84.1}
\]

### 84.1 Exact trivariate orthant endpoint formula

For a coordinate set (S), a nonnegative multi-index (p) on (S), and
(C=C_{SS}(t)), define

\[
\boxed{
 \mathcal T_{p,S}(t):=
 \left.\partial_s^p\left[
 e^{s^Tm_S+\frac12s^TCs}
 \Phi_{|S|}^{+}(m_S+Cs,C)
 \right]\right|_{s=0}.}
\tag{84.2}
\]

Coordinates with (p_a=0) contribute an indicator of positivity; coordinates
with (p_a>0) contribute ((U_a)_+^{p_a}).  Thus (84.2) simultaneously supplies
all raw moments and indicator moments needed below.  In particular, for
(S=\{i,j,k\}),

\[
 M_{abc}^{i;jk}(t)
 :=E\{Y_i(t)^aY_j(t)^bY_k(t)^c\}
 =\mathcal T_{(a,b,c),S}(t).
\tag{84.3}
\]

Let (\nu_i=M_{100}), and obtain centered moments by the finite binomial
transform

\[
 \overline M_{abc}
 =\sum_{r\le a,s\le b,q\le c}
 {a\choose r}{b\choose s}{c\choose q}
 (-\nu_i)^{a-r}(-\nu_j)^{b-s}(-\nu_k)^{c-q}M_{rsq}.
\tag{84.4}
\]

Writing (V_{ab}=E[(Y_a-\nu_a)(Y_b-\nu_b)]), the exact q211 cumulant is

\[
\boxed{
 K_{i;jk}(t)=\overline M_{211}^{i;jk}(t)
 -V_{ii}(t)V_{jk}(t)-2V_{ij}(t)V_{ik}(t).}
\tag{84.5}
\]

Consequently the complete compiled Gaussian-rest response is the endpoint
difference

\[
\boxed{
 \Delta S_G
 =\sum_{i,j,k}L_i[j,k]\{K_{i;jk}(1)-K_{i;jk}(0)\},}
\tag{84.6}
\]

where every term is a derivative of one at-most-trivariate Gaussian orthant
probability.  Equation (84.6) is the entire path action, not a first-order
approximation and not a Hermite-band truncation.  It is equivalent to
integrating (81.10), but usually avoids numerical path quadrature if the
orthant evaluator is resident.

For reference, at zero mean the probability and pair moment primitives are

\[
 P(U_i,U_j,U_k>0)
 ={1\over8}+{1\over4\pi}
 \{\arcsin\rho_{ij}+\arcsin\rho_{ik}+\arcsin\rho_{jk}\},
\tag{84.7}
\]

\[
 E[U_{a,+}U_{b,+}]
 ={\sigma_a\sigma_b\over2\pi}
 \left\{\sqrt{1-\rho_{ab}^2}
 +\rho_{ab}\left({\pi\over2}+\arcsin\rho_{ab}\right)\right\}.
\tag{84.8}
\]

The remaining powers in (84.5) are obtained by the finite derivatives in
(84.2); no unobserved distributional parameter remains.

### 84.2 Explicit Gaussian boundary jet along the whole heat path

The delta terms in (81.8) also reduce to lower-dimensional orthant moments.
For one coordinate (a), let

\[
 p_a^{(t)}(0)={1\over\sqrt{2\pi C_{aa}(t)}}
 \exp\left\{-{m_a^2\over2C_{aa}(t)}\right\},
\tag{84.9}
\]

and, for (B) disjoint from (a), define the conditional Gaussian parameters

\[
 m_{B|a=0}=m_B-C_{Ba}C_{aa}^{-1}m_a,
 \qquad
 C_{B|a}=C_{BB}-C_{Ba}C_{aa}^{-1}C_{aB}.
\tag{84.10}
\]

Then for every multi-index (p) on (B),

\[
\boxed{
 E\left[\delta(U_a(t))\prod_{b\in B}U_{b,+}^{p_b}\right]
 =p_a^{(t)}(0)\,
 \mathcal T_{p,B}(m_{B|a=0},C_{B|a}).}
\tag{84.11}
\]

Here the notation on the right means the same tilted-orthant derivative as
(84.2) with the displayed conditional mean and covariance.  Indicator
products use (84.2) directly with zero derivative order on the indicator
coordinates.  For example, at zero mean,

\[
 E[\delta(U_a)I_bI_c]
 ={1\over\sqrt{2\pi C_{aa}}}
 \left\{{1\over4}+{1\over2\pi}
 \arcsin\rho_{bc|a}\right\},
\tag{84.12}
\]

where

\[
 \rho_{bc|a}=
 {C_{bc}-C_{ba}C_{ac}/C_{aa}\over
 \sqrt{(C_{bb}-C_{ba}^2/C_{aa})
       (C_{cc}-C_{ca}^2/C_{aa})}}.
\tag{84.13}
\]

Substitution of (84.2) and (84.11) into (81.8) gives
(\mathcal G_L(t)) explicitly for every (t).  Therefore the Section 81
boundary-jet information obstruction disappears under the Gaussian-AQ
premise.  The costate contraction and its common-support rank do not.

### 84.3 Exact low-rank fusion classes

There is one strong common-ray exception.  Suppose for all coordinates in
the live support

\[
 m=\mu v,qquad C(t)=q(t)vv^T,qquad v_a\ge0.
\tag{84.14}
\]

Then, for one scalar (Z\sim N(0,1)),

\[
 Y_a(t)=v_a(\mu+\sqrt{q(t)}Z)_+,
\]

and hence

\[
\boxed{
 K_{i;jk}(t)=\kappa_4((\mu+\sqrt{q(t)}Z)_+)
 v_i^2v_jv_k.}
\tag{84.15}
\]

The complete q211 tensor stays on one CP4 ray for the entire heat path.  If
that ray and its compiled endpoint action are resident, (84.6) reduces to
one scalar amplitude update.  The same conclusion holds blockwise when the
coordinates are partitioned into independent, sign-coherent one-ray blocks
and the literal collision/old projector preserves those blocks.  This is
the Gaussian version of the block-monomial escape in Section 75.

At zero mean, the scalar constant is

\[
 \kappa_+:=\kappa_4(Z_+)
 ={3\over4}-{1\over\pi}-{3\over2\pi^2},
\tag{84.16}
\]

so (84.15) becomes (q(t)^2\kappa_+v_i^2v_jv_k).

More generally, an exact low-rank heat fusion for one compiled target exists
when the response matrices have the Section 81 common support

\[
 \mathcal G_L(t)=U\Theta(t)U^T
\tag{84.17}
\]

with small fixed (U), and the resident evaluator supplies both the
orthant coefficients (\Theta(t)) and (U^T\Sigma_GU).  Low rank of
(C(t)) alone is insufficient: ReLU of a sum of two or more latent Gaussian
modes generically produces continuously varying threshold directions and a
full coordinate kernel.  Only a common-ray/block-monomial factorization, or
an actually certified target support (84.17), removes that obstruction.

The obstruction already occurs at covariance rank two.  Take
(u_a=(\cos\theta_a,\sin\theta_a)) and
(U_a=u_a^TZ) for (Z\sim N(0,I_2)).  Then

\[
 E[U_{a,+}U_{b,+}]
 ={1\over2\pi}\{\sin\vartheta_{ab}
 +(\pi-\vartheta_{ab})\cos\vartheta_{ab}\},
 \qquad
 \vartheta_{ab}=\arccos(u_a^Tu_b).
\tag{84.17a}
\]

This circular arc-cosine kernel has infinitely many nonzero Fourier
harmonics, so its Gram matrix on generic distinct angles has rank growing
with (n), despite (\operatorname{rank}C=2).  The pair-correction factors in
(84.5), and generically the connected q211 kernel itself, inherit this
unbounded coordinate rank.  Thus replacing the common-support receipt by
the latent covariance rank is mathematically invalid.

### 84.4 Exact relation to the Price pairing-minus-trace word

For the centered standardized Gaussian projection defect, the complete
quadratic correlation jet at each endpoint is (80.9).  Hence its heat
endpoint difference through that order is

\[
\boxed{
 \Delta K^{(2)}
 ={c\over2}\left[
 (\mathcal P_{R(1)}-\mathcal Z_{R(1)})
 -(\mathcal P_{R(0)}-\mathcal Z_{R(0)})\right],
 \qquad c={\pi-4\over8\pi}.}
\tag{84.18}
\]

Thus the resident pairing word and the Section 80 trace read can fuse the
**quadratic jet** exactly.  They do not fuse the complete Gaussian heat
response on an open correlation set.  Formula (84.8) already contains
(\sqrt{1-\rho^2}) and (\arcsin\rho), while Sections 72 and 78 exhibit
nonzero rooted and triangle terms at every higher order.  A finite span of
the polynomial Price words cannot equal those analytic functions on an open
interval.

Let (\mathcal O_{res}) be the literal typed span emitted by the resident
CP4/Star4/pair/Q24/trace evaluator after collision sewing.  The exact
reusable fusion criterion is

\[
\boxed{
 \Pi_{\mathcal O_{res}^{\perp}}
 \{K(m,C_0+\Sigma_G)-K(m,C_0)\}=0.}
\tag{84.19}
\]

For one frozen endpoint only, (84.19) may be weakened to annihilation of
that residual by the compiled costate (L).  Nominal covariance rank, a
small Price tangent, or membership of only the quadratic jet does not imply
(84.19).  The common-ray case (84.14)--(84.15) is an exact positive example;
generic positive-definite triples are a counterexample.

### 84.5 Sharp one-pass condition and counterexample

Suppose the baseline value (S_L(0)) is resident.  One additional endpoint
pass is exact if and only if the resident evaluator accepts
(C_1=C_0+\Sigma_G) and emits the same compiled centered moment/pairing
surface (84.5); then (84.6) is a subtraction of two identically typed
outputs.  If the alleged one pass supplies only the value and covariance
adjoint at (C_0), it returns the tangent (81.11).  That tangent equals the
given finite response exactly if and only if

\[
 \boxed{
 \int_0^1(1-t)
 D_{\Sigma_G}^2\mathscr S_L(C_0+t\Sigma_G)dt=0.}
\tag{84.20}
\]

This condition is sharp by the integral remainder (81.12).  Cancellation in
(84.20) can occur accidentally for one endpoint.  A reusable tangent which
is exact for every prefix (C_0+s\Sigma_G), (0\le s\le1), exists if and only
if

\[
 D_{\Sigma_G}^2\mathscr S_L(C_0+t\Sigma_G)=0
 \quad\hbox{for every }0\le t\le1,
\tag{84.20a}
\]

i.e. the target surface is affine on the covariance segment.

Even the cheapest rank-one path violates it generically.  In the zero-mean
common-ray case, let (q(t)=q_0+t\delta).  For

\[
 A_L:=\sum_{i,j,k}L_i[j,k]v_i^2v_jv_k,
\]

\[
 S_L(t)=\kappa_+A_L(q_0+t\delta)^2.
\tag{84.21}
\]

The tangent misses exactly

\[
 \boxed{
 S_L(1)-S_L(0)-S_L'(0)=\kappa_+A_L\delta^2,}
\tag{84.22}
\]

which is nonzero whenever the rest direction and endpoint are live.  Adding
an arbitrarily small independent diagonal covariance makes the example
positive definite while preserving a nonzero second directional derivative
by continuity.  Therefore neither low covariance rank nor one resident
adjoint pass proves exact finite-rest response.  The common-ray lane is
one-pass only when its **endpoint amplitude** (84.15), rather than merely
its tangent, is evaluated.

The deployment split is now exact:

- a resident full Gaussian orthant endpoint evaluator gives (84.6) in one
  additional identically typed endpoint pass;
- a resident Price pairing/trace evaluator gives only (84.18), with a
  higher-order error gate;
- a resident covariance adjoint gives only (81.11), exact under (84.20);
- a common-ray/block-monomial evaluator gives the full response by an
  amplitude update;
- otherwise the trivariate surface is information-determined but the dense
  target contraction remains the Section 73/81 rank problem.

Graph-ready split:

- `GAUSSIAN_AQ_Q211_ORTHANT_ENDPOINT` -- **solved exact** by
  (84.2)--(84.6);
- `GAUSSIAN_AQ_Q211_BOUNDARY_JET` -- **solved exact** by
  (84.9)--(84.13);
- `GAUSSIAN_AQ_COMMON_RAY_FUSION` -- **exact conditional GO** by
  (84.14)--(84.17);
- `GAUSSIAN_AQ_PRICE_TRACE_FUSION` -- **exact only through the quadratic
  jet** by (84.18); generic full-response fusion is closed by the
  nonpolynomial orthant kernel;
- `GAUSSIAN_AQ_LITERAL_RESIDENT_ROWSPAN` -- decided exactly by (84.19);
- `GAUSSIAN_AQ_ONE_TANGENT_PASS` -- **fixed-endpoint exact iff** (84.20),
  reusable exact iff (84.20a), with the closed rank-one counterexample
  (84.21)--(84.22);
- `GAUSSIAN_AQ_FULL_ENDPOINT_PASS` -- **conditional GO** when the resident
  evaluator emits the compiled surface (84.5) at (C_1);
- `GAUSSIAN_AQ_Q211_TARGET_COST` -- open unless common support (84.17) or
  the Section 73 factorization is supplied.

## 83. Price trace as an adjoint pair collapse: exact scalar and the
resident-tape obstruction

The forward trace word in Section 80 has generically huge CP rank, but a
fixed endpoint does not pay that rank.  Let

\[
 {\cal Q}_{211}=\{T_{i;jk}:i,j,k\ {\rm distinct},\ T_{i;jk}=T_{i;kj}\}
\]

with

\[
 \langle L,T\rangle_{\cal Q}
 ={1\over2}\sum_i\sum_{\substack{j,k\ne i\\j\ne k}}L_{i;jk}T_{i;jk}.
\]

Define

\[
 ({\cal Z}R)_{i;jk}=R_{jk},\qquad
 ({\cal Z}^{*}L)_{jk}=\sum_{i\notin\{j,k\}}L_{i;jk}.
 \tag{83.1}
\]

Then

\[
 \boxed{\langle L,{\cal Z}R\rangle_{\cal Q}
 =\langle{\cal Z}^{*}L,R\rangle_{\rm off}.} \tag{83.2}
\]

If \(A_{\rm old}\) is the actual synchronized old-interface residualizer and
\(\widetilde L=A_{\rm old}^{*}L\), the exact zero-bias Price residual is

\[
 \boxed{
 s_P(L,R)=-{c\over2}\langle\Gamma_{\widetilde L},R\rangle_{\rm off},
 \quad
 \Gamma_{\widetilde L}:={\cal Z}^{*}\widetilde L,\quad
 c={\pi-4\over8\pi}.} \tag{83.3}
\]

Residualization must precede the collapse.  For endpoints
\(L^{(1)},\ldots,L^{(m)}\), the exact query dimension is

\[
 p_\Gamma=\dim\operatorname{span}
 \{\Gamma_{\widetilde L^{(1)}},\ldots,\Gamma_{\widetilde L^{(m)}}\}.
 \tag{83.4}
\]

Thus one frozen endpoint has query dimension at most one even though the
forward tensor has full middle-catalecticant rank.

### 83.1 Why the local collapse does not propagate

For the unmasked insertion

\[
 {\cal Z}_{G}(R)=\operatorname{Sym}(G\otimes R),
\]

a dense map \(W\) gives the exact identities

\[
 W^{\otimes4}{\cal Z}_{G}(R)
 ={\cal Z}_{WGW^{T}}(WRW^{T}), \tag{83.5}
\]

\[
 \boxed{
 {\cal Z}_{G}^{*}(W^{T})^{\otimes4}L
 =W^{T}{\cal Z}_{WGW^{T}}^{*}(L)W.} \tag{83.6}
\]

Hence \(I\mapsto WW^{T}\), and neuron-specific resets introduce further
weighted legs.  Strict masks also create K4/K31/K22 collision terms.  A
single matrix \({\cal Z}_{I}^{*}L\) therefore has no generic autonomous
depth recurrence.  Closure requires a resident weighted-leg family
\({\cal G}\) satisfying

\[
 WGW^{T}\in{\cal G}
\]

at every linear cut and closure of the same family under reset and collision
sewing.  Scaled-orthogonal maps, the block-monomial algebras of Section 75,
or a literal full Bellman tape are the exact escapes.

Equivalently, for complete future map \(F\), resident pair consumer \(O\),
and pair supply \(C\),

\[
 \boxed{
 F{\cal Z}=OC
 \quad\Longleftrightarrow\quad
 {\cal Z}^{*}F^{*}=C^{*}O^{*}.} \tag{83.7}
\]

If \(CR\) is already old-owned this proves duplication; if it is new, the
lawful correction is that new pair supply.  Equality only for one costate is
target-specific, not a reusable carrier theorem.

### 83.2 Literal resident condition and cost

Let \(b_l\) be a paid reverse state, \(E_lb_l\) its exposed pair
coordinates, and \(D_lb_l\) its complete-old-residualized q211 costate.
Then the collapsed query is a resident view if and only if

\[
 \boxed{
 \ker E_l\subseteq\ker({\cal Z}^{*}D_l)
 \quad\Longleftrightarrow\quad
 {\cal Z}^{*}D_l=A_lE_l\ {\rm for\ some}\ A_l.} \tag{83.8}
\]

A transient object named \(Q_{22}\) qualifies only if it is the complete
reverse coordinate of the literal synchronized node
\(R_l\mapsto{\cal Z}(R_l)\), with the identical strict mask.  A direct K22
collision costate, a Price gate matrix, or an abstract q4-perp word is not
such a receipt.

The current symbolic D10/D11 census does not prove (83.8): its forward
\((m,U,A)\) and covariance-only rest contain neither the future q211 costate
nor \(\Gamma\), and the declared late-cut objects are not a literal complete
H211 reverse tape.  The honest current status is

\[
 \boxed{\text{local collapse solved; resident acquisition unproved}.}
 \tag{83.9}
\]

If \(\Gamma_l\) is resident, the final operation is one \(n^2\) dot per
seam:

\[
 C_{\rm dot}=D/n=10/1024=0.009765625. \tag{83.10}
\]

If only a rank-\(R_c\) partially symmetric CP costate is available, (80.20)
gives

\[
 C_{\rm CP}=2DR_c/n\quad{\rm generic},\qquad
 C_{\rm CP}=DR_c/n\quad{\rm symmetric}. \tag{83.11}
\]

Without (83.8) or a bounded-rank factorization, (83.6) reopens a weighted-leg
reverse chain; the cheap final dot does not erase that acquisition bill.

### 83.3 Comparator and value gates

If the fused Section 49 surface is part of the comparator, the Price
response is already in \(\kappa_4(Y^{AQ})\) and (83.3) has zero additive
credit.  If only \(X^{AQ}=P_{\le2}Y^{AQ}\) is carried, (83.3) is a missing
covariance-determined nonlinear readout, not new source information.  The
Gaussian-rest heat response of Section 81 and the dependent/non-Gaussian
gap remain separate.

For panel feature \(s_P\), let \(s_\perp\) be its exact residual against all
old columns and \(e\) the old prediction residual.  Its one-column gain is

\[
 \boxed{
 \Delta{\rm MSE}
 ={ \langle e,s_\perp\rangle^2
    \over N\langle s_\perp,s_\perp\rangle}.} \tag{83.12}
\]

No source-energy or tensor-rank statement gives a positive lower bound on
(83.12).

Graph-ready split:

- PRICE_TRACE_ADJOINT_COLLAPSE -- **solved exact**, (83.1)--(83.4);
- PRICE_TRACE_DENSE_NATURALITY -- **generic NO / structured conditional
  GO**, (83.5)--(83.7);
- PRICE_TRACE_RESIDENT_TAPE -- **open, no literal current receipt**, (83.8);
- PRICE_TRACE_LOCAL_DOT_COST -- **solved conditional**, (83.10);
- PRICE_TRACE_FACTOR_FALLBACK -- **solved conditional**, (83.11);
- PRICE_TRACE_SCHUR_VALUE -- **open**, exactly (83.12);
- PRICE_TRACE_WITH_FULL_AQ_SURFACE -- **closed duplicate**.

## 85. Full biased Price adjoint: three pair collapses, the cubic
open-path obstruction, and exact factor fallbacks

Section 83 collapses the zero-bias pairing-minus-trace residual because that
residual is linear in \(R\).  The biased Price jet (80.1) is quadratic and
has two topologies.  At one fixed network both topologies still have an exact
two-index covariance adjoint.  What fails generically is acquisition of that
adjoint from a reusable \(O(n^2)\) reverse state.

All formulas below use the complete-old/collision-compiled costate

\[
 \widetilde L=A_{\rm old}^{*}L,\qquad
 \widetilde L_{i;jk}=\widetilde L_{i;kj},
 \tag{85.1}
\]

extended by zero whenever two of \(i,j,k\) collide.  Residualization in
(85.1) occurs before every collapse.  To shorten notation put

\[
 a=b_1,\qquad h=b_2,\qquad p=d_2,\qquad q=d_1.
 \tag{85.2}
\]

For a symmetric matrix use

\[
 \langle A,B\rangle_{\rm off}
 ={1\over2}\sum_{i\ne j}A_{ij}B_{ij},\qquad
 \operatorname{Sym}_{+}A=A+A^T.
 \tag{85.3}
\]

### 85.1 Rooted-star and open-path actions

Split the Price tensor as

\[
 {\cal P}^{(0)}(R)_{i;jk}
 =p_i a_j a_kR_{ij}R_{ik},
 \tag{85.4}
\]

\[
 {\cal P}^{(1)}(R)_{i;jk}
 =q_iR_{jk}\{h_ja_kR_{ij}+a_jh_kR_{ik}\}.
 \tag{85.5}
\]

The q211 inner product and \(j,k\) symmetry give

\[
 \boxed{
 S_0:=
 \langle\widetilde L,{\cal P}^{(0)}(R)\rangle_{\cal Q}
 ={1\over2}\sum_i p_i x_i^T\widetilde L_i x_i,
 \qquad x_i=a\odot R_{i,:}^{T},}
 \tag{85.6}
\]

and

\[
 \boxed{
 S_1:=
 \langle\widetilde L,{\cal P}^{(1)}(R)\rangle_{\cal Q}
 =\sum_{i,j,k}^{\rm strict}
 q_i h_j a_k\widetilde L_{i;jk}R_{ij}R_{jk}.}
 \tag{85.7}
\]

The two Jordan orientations in (85.5) become the single oriented path in
(85.7); neither is discarded.

### 85.2 Exact covariance adjoint and the three elementary pair queries

Define three two-index contractions:

\[
 U^{0}_{ij}
 :=\sum_k\widetilde L_{i;jk}a_kR_{ik},
 \tag{85.8}
\]

\[
 U^{1}_{ij}
 :=\sum_k\widetilde L_{i;jk}a_kR_{jk},
 \qquad
 V^{1}_{jk}
 :=\sum_iq_i\widetilde L_{i;jk}R_{ij}.
 \tag{85.9}
\]

Thus \(U^0_{i,:}=\widetilde L_i(a\odot R_{i,:})\).  In \(U^1\), by
contrast, the covariance row is selected by the open-path middle vertex
\(j\), not by the repeated anchor \(i\).  This distinction is the source of
the generic obstruction.

Set

\[
 A_0=D_pU^0D_a,\qquad
 A_1=D_qU^1D_h,\qquad
 B_1=D_hV^1D_a.
 \tag{85.10}
\]

For every symmetric off-diagonal perturbation \(H\), direct differentiation
of (85.6)--(85.7) gives

\[
 DS_0[R](H)
 =\langle\Gamma_0,H\rangle_{\rm off},
 \qquad
 \boxed{\Gamma_0=\operatorname{offdiag}
                    \operatorname{Sym}_{+}(A_0),}
 \tag{85.11}
\]

\[
 DS_1[R](H)
 =\langle\Gamma_1,H\rangle_{\rm off},
 \qquad
 \boxed{\Gamma_1=\operatorname{offdiag}
                    \operatorname{Sym}_{+}(A_1+B_1).}
 \tag{85.12}
\]

These are the exact adjoints of the rooted-star and open-path Jordan words.
Because both actions are homogeneous of degree two,

\[
 \boxed{
 S_0+S_1={1\over2}
 \langle\Gamma_0+\Gamma_1,R\rangle_{\rm off}.}
 \tag{85.13}
\]

Hence the **minimal fixed-network sufficient query** for the covariance
reverse is the one symmetric matrix
\(\Gamma=\Gamma_0+\Gamma_1\).  Equations (85.8)--(85.9) are its three
elementary mode contractions.  For a scalar-only endpoint, \(U^1\) alone
already evaluates \(S_1\); the second path contraction \(V^1\) is necessary
for the complete covariance adjoint.  For arbitrary q211 costates neither
path contraction determines the other: the exact criterion for such a
reduction is the same kernel inclusion stated in Section 85.4.

The same three matrices also give every univariate-coefficient adjoint:

\[
 { \partial S_0\over\partial p_i}
 ={1\over2}\sum_j a_jR_{ij}U^0_{ij},
 \qquad
 { \partial S_0\over\partial a_j}
 =\sum_i p_iR_{ij}U^0_{ij},
 \tag{85.14}
\]

\[
 { \partial S_1\over\partial q_i}
 =\sum_jh_jR_{ij}U^1_{ij},\qquad
 { \partial S_1\over\partial h_j}
 =\sum_iq_iR_{ij}U^1_{ij},
 \tag{85.15}
\]

\[
 { \partial S_1\over\partial a_k}
 =\sum_jh_jR_{jk}V^1_{jk}.
 \tag{85.16}
\]

Thus no additional three-index reverse object is hidden in the local
calculus.  The issue is whether (85.8)--(85.9) can be acquired lawfully.

### 85.3 The minimal reusable three-index query and sharp open-path rank

For one unordered triple \(\{i,j,k\}\), let

\[
 \ell_{ijk}
 =(\widetilde L_{i;jk},
   \widetilde L_{j;ik},
   \widetilde L_{k;ij})^T.
 \tag{85.17}
\]

With the wedge monomials \(s=(s_i,s_j,s_k)^T\) from (80.2), the exact
coefficient query reusable over varying correlations is

\[
 \boxed{
 \xi_{ijk}=(M_{ijk}^{P})^T\ell_{ijk},\qquad
 S_P=\sum_{\{i,j,k\}}\xi_{ijk}^{T}s_{ijk}.}
 \tag{85.18}
\]

This is a three-component coordinate on every unordered triple: it is still
a strict-q211-sized, three-index object.  It is minimal because the wedge
monomials are linearly independent as polynomials.  Indeed the positive
definite correlation cone contains an open neighborhood of \(I\) in all
off-diagonal edge coordinates, so a polynomial vanishing there has every
wedge coefficient zero.

The obstruction already holds for the open-path word alone.  Its orientation
block is

\[
 M_{ijk}^{J}=
 \begin{pmatrix}
 0&\theta_{ij}&\theta_{ik}\\
 \theta_{ji}&0&\theta_{jk}\\
 \theta_{ki}&\theta_{kj}&0
 \end{pmatrix}.
 \tag{85.19}
\]

Using (80.3), its determinant simplifies exactly:

\[
 \boxed{
 \det M_{ijk}^{J}
 =\theta_{ij}\theta_{jk}\theta_{ki}
  +\theta_{ik}\theta_{ji}\theta_{kj}
 =2\prod_{r\in\{i,j,k\}}
       {d_{1r}b_{2r}\over b_{1r}^{3}}.}
 \tag{85.20}
\]

For finite affine ReLU thresholds \(b_{1r},b_{2r}>0\).  Therefore every
triple on which all three \(d_{1r}\) are nonzero has a full-rank
three-dimensional open-path coefficient map.  If

\[
 m=\#\{r:d_{1r}\ne0\},
\]

then, on the full q211 costate space supported on those \(m\) labels, any
exact **linear reusable encoding** that evaluates the open-path action for
every correlation in an open set has dimension at least

\[
 \boxed{3{m\choose3}.} \tag{85.21}
\]

Proof: the global coefficient map is block diagonal over unordered triples
with blocks \((M_{ijk}^{J})^T\); (85.20) gives rank three on each live block,
and polynomial independence prevents cross-triple cancellation.

Thus for \(m=\Theta(n)\) the exact generic query is
\(\Theta(n^3)\), not \(O(n^2)\).  Zero bias escapes because \(d_1=0\) and
the entire open path vanishes.  Sparse biased support can escape the
dimension bound only when \(m=O(n^{2/3})\), or when the live costates are
restricted to a separate \(O(n^2)\)-dimensional structured subspace.  A
cheap final matrix dot at one network does not defeat (85.21): forming its
matrix from a generic reusable costate is the cubic operation.

### 85.4 Necessary and sufficient resident-collapse theorem

Let \(\mathfrak A_2(R_l)\) denote the class of pair-state linear maps whose
application uses \(O(n^2)\) arithmetic and no unbilled dense
\(n^2\)-by-\(n^2\) operator.  Let \(b_l\) be the paid reverse state, let
\(D_lb_l=\widetilde L_l\) be its complete-old-residualized q211 costate, and
let \(E_lb_l\) expose a fixed number of resident pair matrices.  For the
network covariance \(R_l\), let

\[
 {\cal G}_{R_l}(\widetilde L_l)
 :=\Gamma_0+\Gamma_1
 \tag{85.22}
\]

be the adjoint map (85.8)--(85.12).  Algebraic residency is equivalent to

\[
 \boxed{
 \ker E_l\subseteq\ker({\cal G}_{R_l}D_l)
 \quad\Longleftrightarrow\quad
 {\cal G}_{R_l}D_l=A_l(R_l)E_l
 \quad\hbox{for some }A_l(R_l).}
 \tag{85.23}
\]

Exact \(O(n^2)\) resident acquisition holds if and only if the factor in
(85.23) can be chosen with

\[
 \boxed{A_l(R_l)\in\mathfrak A_2(R_l).} \tag{85.23a}
\]

The kernel condition alone is information-theoretic and does not make an
arbitrary pair-to-pair operator cheap.  Equations (85.23)--(85.23a) permit
cancellations among the three elementary queries and are jointly necessary
and sufficient.  Requiring the stronger tuple \((U^0,U^1,V^1)\) to factor
through \(E_l\) by local diagonal/Hadamard maps is a convenient sufficient
receipt, not a necessary one.

For one resident design to work on a family \(\mathfrak R_l\), the uniform
information condition is

\[
 \boxed{
 \ker E_l\subseteq
 \bigcap_{R\in\mathfrak R_l}
 \ker({\cal G}_{R}D_l).}
 \tag{85.24}
\]

and the corresponding factors \(A_l(R)\) must belong to
\(\mathfrak A_2(R)\) uniformly.

If only one frozen endpoint scalar is required, replace \({\cal G}_R\) in
(85.23) by the scalar functional \(S_R\); that weaker factorization is
target-specific and supplies no reusable covariance reverse.  Conversely,
if evaluation is required for an open correlation family and the live
costate space contains generic q211 directions, (85.21) proves that no
fixed \(O(n^2)\) \(E_l\) can satisfy (85.24).

If the old residualizer itself changes with \(R\), (85.11)--(85.16) are the
adjoints with \(A_{\rm old}\) frozen at the synchronized seam.  A derivative
of the projector contributes the separate term
\(\langle (DA_{\rm old}^{*})[H]L,{\cal P}(R)\rangle\); it may not be hidden
inside the Price collapse.

### 85.5 Exact CP and symmetric-Tucker fallbacks

Suppose first that the collision-compiled costate has one partially
symmetric CP atom

\[
 \widetilde L_{i;jk}
 =\alpha_i\,{ \beta_j\gamma_k+\gamma_j\beta_k\over2}.
 \tag{85.25}
\]

The three covariance products

\[
 P_\beta=R(\beta\odot a),\qquad
 P_\gamma=R(\gamma\odot a),\qquad
 Q_\alpha=R(\alpha\odot q)
 \tag{85.26}
\]

give both the scalar action (77.11) and the complete adjoint.  Explicitly,

\[
 U^0={1\over2}
 \{(\alpha\odot P_\gamma)\beta^T
   +(\alpha\odot P_\beta)\gamma^T\},
 \tag{85.27}
\]

\[
 U^1=\alpha u^T,\qquad
 u={1\over2}(\beta\odot P_\gamma+\gamma\odot P_\beta),
 \tag{85.28}
\]

\[
 V^1={1\over2}
 \{(Q_\alpha\odot\beta)\gamma^T
   +(Q_\alpha\odot\gamma)\beta^T\}.
 \tag{85.29}
\]

After the diagonal scalings in (85.10), (85.27)--(85.29) retain the same
outer-product ranks.  Thus \(\Gamma\) has a signed matrix factorization of
rank at most ten per generic atom and at most six when
\(\beta=\gamma\).  Across \(D\) seams and \(R_c\) atoms, the dense-product
bill is exactly

\[
 \boxed{
 C_{\rm CP}^{\rm generic}={3DR_c\over n},\qquad
 C_{\rm CP}^{\rm sym}={2DR_c\over n}.}
 \tag{85.30}
\]

Strict collision corrections are local \(O(nR_c)\) terms if (85.25) was
fit before masking; they must be added, but require no new covariance
product.

For a symmetric Tucker costate write

\[
 \widetilde L_{i;jk}
 =\sum_{\mu=1}^{r_a}
   \sum_{\nu,\omega=1}^{r_s}
 A_{i\mu}G_{\mu\nu\omega}B_{j\nu}B_{k\omega},
 \qquad G_{\mu\nu\omega}=G_{\mu\omega\nu}.
 \tag{85.31}
\]

Only two block covariance products are needed:

\[
 P=R(D_aB)\in{\mathbb R}^{n\times r_s},\qquad
 Q=R(D_qA)\in{\mathbb R}^{n\times r_a}.
 \tag{85.32}
\]

The exact scalar actions are

\[
 S_0={1\over2}\sum_{i,\mu,\nu,\omega}
 p_iA_{i\mu}G_{\mu\nu\omega}P_{i\nu}P_{i\omega},
 \tag{85.33}
\]

\[
 S_1=\sum_{j,\mu,\nu,\omega}
 h_jB_{j\nu}G_{\mu\nu\omega}Q_{j\mu}P_{j\omega}.
 \tag{85.34}
\]

Moreover,

\[
 U^0=H^0B^T,\quad
 H^0_{i\nu}
 =\sum_{\mu,\omega}A_{i\mu}G_{\mu\nu\omega}P_{i\omega},
 \tag{85.35}
\]

\[
 U^1=AK^T,\quad
 K_{j\mu}
 =\sum_{\nu,\omega}B_{j\nu}G_{\mu\nu\omega}P_{j\omega},
 \tag{85.36}
\]

\[
 V^1=H^1B^T,\quad
 H^1_{j\omega}
 =\sum_{\mu,\nu}Q_{j\mu}B_{j\nu}G_{\mu\nu\omega}.
 \tag{85.37}
\]

The covariance-product bill is therefore

\[
 \boxed{
 C_{\rm Tucker}={D(r_a+r_s)\over n},}
 \tag{85.38}
\]

or \(Dr_s/n\) when \(d_1=0\).  The honest additional storage is
\(n(r_a+r_s)+r_ar_s^2\), and the direct local core arithmetic is
\(O(Dnr_ar_s^2)\).  Equation (85.38) is a useful cap only when those local
terms and collision sewing also fit; naming a small mode rank without the
core bill is not a deployment receipt.

### 85.6 Ownership and disposition

The complete-old projection and comparator choice are independent gates:

1. Apply (85.1), then compute the adjoint or factor fallback.  A large
   unprojected Price action is not additive value.
2. If the full Section 49 affine surface belongs to the declared comparator,
   both (85.4) and (85.5) are already present in
   \(\kappa_4(Y^{AQ})\); every formula above has zero additive credit.
3. Against the carried \(P_{\le2}Y^{AQ}\) comparator, the source coefficients
   are resident, but (85.23) or a certified CP/Tucker factorization is still
   required to call the reverse action deployable.

Graph-ready split:

- AFFINE_PRICE_BIASED_ROOT_ADJOINT -- **solved exact**, (85.8),
  (85.10)--(85.11);
- AFFINE_PRICE_BIASED_PATH_ADJOINT -- **solved exact**, (85.9)--(85.12);
- AFFINE_PRICE_BIASED_FIXED_R_PAIR_COLLAPSE -- **solved local**, one
  symmetric matrix by (85.13);
- AFFINE_PRICE_BIASED_RESIDENT_COLLAPSE -- **conditional open**, exactly
  (85.23)--(85.24), including the \(O(n^2)\) operator gate (85.23a);
- AFFINE_PRICE_OPEN_PATH_CUBIC_NOGO -- **generic NO-GO**, lower bound
  \(3{m\choose3}\) by (85.20)--(85.21);
- AFFINE_PRICE_BIASED_CP_FALLBACK -- **solved conditional**, (85.25)--(85.30);
- AFFINE_PRICE_BIASED_TUCKER_FALLBACK -- **solved conditional**,
  (85.31)--(85.38);
- AFFINE_PRICE_BIASED_WITH_SECTION49 -- **closed duplicate**.

## 86. Closed zero-mean Gaussian q211 ReLU kernel, exact gradient, and the
finite-word obstruction

Section 84 gives the answer as a derivative of a trivariate orthant
probability.  In the centered unit-variance case every derivative can be
eliminated.  This section gives an elementary closed formula for

\[
 K(x,y,z):=\kappa_4(X_+,X_+,Y_+,Z_+),
\tag{86.1}
\]

where

\[
 E[XY]=x,\qquad E[XZ]=y,\qquad E[YZ]=z,
\]

and

\[
 \Delta:=1+2xyz-x^2-y^2-z^2>0.
\tag{86.2}
\]

The positive-semidefinite boundary is obtained by continuous limits.

### 86.1 Elementary orthant and boundary primitives

Put

\[
 \mu={1\over\sqrt{2\pi}},
 \qquad
 p(r)={1\over4}+{\arcsin r\over2\pi},
\tag{86.3}
\]

\[
 q(r)={\sqrt{1-r^2}+r(\frac\pi2+\arcsin r)\over2\pi},
 \qquad
 t(r)={\mu\over2}(1+r)^2.
\tag{86.4}
\]

Thus (p(r)=P(G_1,G_2>0)),
(q(r)=E[G_{1,+}G_{2,+}]), and
(t(r)=E[G_{1,+}^2G_{2,+}]) for a standardized pair of correlation (r).
The trivariate orthant probability is

\[
 P={1\over8}+{\arcsin x+\arcsin y+\arcsin z\over4\pi}.
\tag{86.5}
\]

Write (r_{12}=x,r_{13}=y,r_{23}=z).  For one boundary index (s) and the
other two indices ((a,b)=\{1,2,3\}\setminus\{s\}), define

\[
 \sigma_{a|s}=\sqrt{1-r_{sa}^2},\qquad
 \sigma_{b|s}=\sqrt{1-r_{sb}^2},
\]

\[
 \eta_s={r_{ab}-r_{sa}r_{sb}\over
              \sigma_{a|s}\sigma_{b|s}}.
\tag{86.6}
\]

The determinant identity

\[
 1-\eta_s^2={\Delta\over
 (1-r_{sa}^2)(1-r_{sb}^2)}
\tag{86.7}
\]

will control the boundary behavior.  Define three unnormalized boundary
moments

\[
 O_s:=\mu p(\eta_s),
\tag{86.8}
\]

\[
 H_{s\to b}:={\sigma_{b|s}(1+\eta_s)\over4\pi},
\tag{86.9}
\]

\[
 B_s:=\mu\,\sigma_{a|s}\sigma_{b|s}q(\eta_s).
\tag{86.10}
\]

They have the literal meanings

\[
 O_s=E[\delta(U_s)I_aI_b],\quad
 H_{s\to b}=E[\delta(U_s)I_aU_{b,+}],\quad
 B_s=E[\delta(U_s)U_{a,+}U_{b,+}].
\tag{86.11}
\]

The following first moments and mixed second moments are then elementary:

\[
 L_a:=O_a+r_{ab}O_b+r_{ac}O_c
 =E[U_{a,+}I_bI_c],
\tag{86.12}
\]

and, for either ordering ((b,c)) of the other two labels,

\[
\boxed{
 J_a:=r_{bc}P+r_{ab}H_{a\to c}+H_{b\to c}
 =E[I_aU_{b,+}U_{c,+}].}
\tag{86.13}
\]

Swapping (b,c) in the first expression gives the same value; this is a
nontrivial but exact boundary-moment identity.

Gaussian integration by parts now gives the only two trivariate raw moments
needed for q211:

\[
\boxed{
 M_{111}:=E[X_+Y_+Z_+]=B_1+xL_3+yL_2,}
\tag{86.14}
\]

\[
\boxed{
 M_{211}:=E[X_+^2Y_+Z_+]=J_1+xJ_2+yJ_3.}
\tag{86.15}
\]

For example, (86.15) follows by applying Stein's identity once to
(X\,X_+Y_+Z_+); the three terms are respectively the anchor boundary and
the two cross derivatives.  All apparent trivariate integration has now
been reduced to square roots and arcsines.

The full cumulant, including every centering term, is

\[
\boxed{\begin{aligned}
 K(x,y,z)={}&M_{211}-{1\over2}q(z)-2q(x)q(y)\\
 &-\mu\{t(x)+t(y)+2M_{111}\}\\
 &+2\mu^2\left\{{1\over2}+2q(x)+2q(y)+q(z)\right\}
 -6\mu^4.
\end{aligned}}
\tag{86.16}
\]

The five lines in (86.16) are respectively the raw four-point moment, the
three pair--pair partitions, the four singleton--triple partitions, the six
pair--singleton--singleton partitions, and the all-singleton partition.
In particular, (K(0,0,0)=0) exactly.

### 86.2 Exact covariance gradient and boundary behavior

Price differentiation of a cumulant merges the two differentiated slots.
Define

\[
 N_1:=P+xH_{2\to1}+yH_{3\to1}
 =E[X_+^2I_2I_3].
\tag{86.17}
\]

Then direct expansion of the resulting third cumulants gives

\[
\boxed{\begin{aligned}
 \partial_xK=2\bigg[{}&J_2-p(x)q(y)-\mu L_3
 -{\mu^2\over2}(1+x)+2\mu^2p(x)\bigg],
\end{aligned}}
\tag{86.18}
\]

\[
\boxed{\begin{aligned}
 \partial_yK=2\bigg[{}&J_3-p(y)q(x)-\mu L_2
 -{\mu^2\over2}(1+y)+2\mu^2p(y)\bigg],
\end{aligned}}
\tag{86.19}
\]

\[
\boxed{
 \partial_zK=N_1-{1\over2}p(z)-2\mu L_1+2\mu^2p(z).}
\tag{86.20}
\]

Equivalently, before evaluating the moments,

\[
 \partial_xK=2\kappa_3(I_XI_Y,X_+,Z_+),
\quad
 \partial_yK=2\kappa_3(I_XI_Z,X_+,Y_+),
\]

\[
 \partial_zK=\kappa_3(I_YI_Z,X_+,X_+).
\tag{86.21}
\]

Equations (86.18)--(86.20) are useful both as gradients and as an audit of
every multiplicity in (86.16).

The kernel and its first derivatives have finite one-sided limits on every
lawful covariance face.  In (86.6), conditional correlations tend to
(\pm1), while the prefactors in (86.9)--(86.10) cancel spurious conditional
variance singularities.  Uniform second- and higher-derivative bounds do
not survive without a determinant margin: differentiating
(\arcsin\eta_s) and using (86.7) produces a generic
(\Delta^{-1/2}) Hessian scale.  Thus a Taylor remainder certificate must
impose

\[
 \Delta\ge\delta>0,qquad |x|,|y|,|z|\le1-\delta,
\tag{86.22}
\]

or use exact endpoint formula (86.16).  At a rank-one sign-coherent corner,
(K(1,1,1)=\kappa_+) by the continuous limit (84.16).

### 86.3 Complete small-correlation expansion through cubic order

Let (\rho=\max(|x|,|y|,|z|)).  Expansion of either (86.16) or the exact
collapsed-slot Price series (78.6) gives

\[
\boxed{\begin{aligned}
 K(x,y,z)={}&{\pi-2\over8\pi}xy
 +{1\over4\pi}(xz+yz)\\
 &+{1\over8\pi}(x^2y+xy^2)
 +{\pi-1\over2\pi^2}xyz
 +O(\rho^4).
\end{aligned}}
\tag{86.23}
\]

There are no constant, linear, or other cubic monomials.  The first line is
the complete quadratic connected jet of the **full** ReLU surface.  It must
not be confused with the state-only projection defect in Section 77, whose
zero-bias (xy) coefficient is ((\pi-4)/(8\pi)) and whose ((xz,yz))
coefficients vanish.  The difference is the old (P_{\le2}) contribution.

The exact derivatives at independence are therefore

\[
 \nabla K(0)=0,qquad
 \partial_{xy}K(0)={\pi-2\over8\pi},qquad
 \partial_{xz}K(0)=\partial_{yz}K(0)={1\over4\pi}.
\tag{86.24}
\]

The triangle coefficient in the full surface is
((\pi-1)/(2\pi^2)); after subtracting the quadratic projection, the
state-only triangle becomes ((\pi-4)/(4\pi^2)) as in (78.21a).

### 86.4 No finite trace/pair/Jordan word for the full kernel

For one partially symmetric CP endpoint atom, the exact action is

\[
 \sum_{i,j,k}\alpha_i\operatorname{Sym}_{jk}(\beta_j\gamma_k)
 K(R_{ij},R_{ik},R_{jk}).
\tag{86.25}
\]

A finite collection of the resident trace/pair/Jordan words in Sections
77, 80, and 83 cannot evaluate (86.25) on an open covariance set.  There
are two independent proofs.

First, already the pair primitive (q(r)) in (86.4) contains
(\sqrt{1-r^2}) and (\arcsin r).  Therefore no finite polynomial in the edge
matrices (R^{\odot d}) equals the full surface on an interval.

Second, the obstruction persists even if a finite dictionary of arbitrary
univariate edge functions is allowed.  Set (z=0), so
(x^2+y^2<1).  With (F=Z_+-\mu), let

\[
 b_a=E[F\operatorname{He}_a(Z)],qquad
 h_a=E[F^2\operatorname{He}_a(Z)].
\]

The exact rooted slice is

\[
\boxed{
 K(x,y,0)=\sum_{a,b\ge1}
 {\{h_{a+b}-2b_ab_b\}b_ab_b\over a!b!}x^ay^b.}
\tag{86.26}
\]

Its square-response exponential generating function is

\[
 \sum_{a\ge0}{h_at^a\over a!}
 =(t^2+1)\Phi(t)+t\phi(t)
 -2\mu\{t\Phi(t)+\phi(t)\}+\mu^2.
\tag{86.27}
\]

Because (\Phi(t)) is not an exponential polynomial, the Hankel matrix
((h_{a+b})) has infinite rank.  Removing the one outer product
((2b_ab_b)) and restricting to the infinitely many active ReLU chaos
degrees cannot make it finite.  Hence the bivariate coefficient matrix in
(86.26) has infinite separable rank.

Every finite edge-function trace/pair/Jordan construction has finite
separable rank on the (z=0) slice: it is a finite sum of products of an
(x)-feature and a (y)-feature after the singleton edge is removed.  Equation
(86.26) contradicts such a representation.  Consequently the exact CP
action needs either

1. the dense trivariate orthant endpoint surface (86.16);
2. an infinite/controlled Hermite feature expansion with the Section 78
   tail gate; or
3. a special common-ray/block-monomial support such as (84.14)--(84.17).

A cheap finite Price or trace consumer is therefore an approximation or a
typed quotient readout, never an exact universal implementation of the full
Gaussian q211 kernel.

Graph-ready split:

- `GAUSSIAN_Q211_ZERO_MEAN_CLOSED_KERNEL` -- **solved exact** by
  (86.3)--(86.16);
- `GAUSSIAN_Q211_ZERO_MEAN_GRADIENT` -- **solved exact** by
  (86.17)--(86.21);
- `GAUSSIAN_Q211_COVARIANCE_BOUNDARY` -- **solved**: continuous kernel and
  gradient, generic Hessian blow-up without (86.22);
- `GAUSSIAN_Q211_SMALL_CORRELATION_JET` -- **solved through cubic** by
  (86.23)--(86.24);
- `GAUSSIAN_Q211_FINITE_POLYNOMIAL_WORD` -- **closed no-go** by (86.4);
- `GAUSSIAN_Q211_FINITE_EDGE_DICTIONARY` -- **closed generic no-go** by the
  infinite-rank rooted slice (86.26)--(86.27);
- `GAUSSIAN_Q211_EXACT_CP_ACTION` -- conditional only on a dense orthant
  surface or the structured exceptions in Section 84;
- `GAUSSIAN_Q211_COMPLETE_OLD_VALUE` -- open after literal projection of
  every full-AQ/Price/trace owner.

## 87. One-transition trace consume-and-retire: identity-pair fusion and
the exact one-Gram boundary

The strict zero-bias trace source need not become a depth carrier if the next
consumer reads only K4/K31/K22 collisions.  Transporting its three birth
collision classes separately appears to require three new Gram-like products.
The exact tensor identity below cancels two of them before they are formed.

For symmetric matrices \(A,B\), define

\[
\begin{aligned}
 {\cal Z}_{A}(B)_{abcd}={}&
 A_{ab}B_{cd}+B_{ab}A_{cd}
 +A_{ac}B_{bd}+B_{ac}A_{bd}\\
 &+A_{ad}B_{bc}+B_{ad}A_{bc},
\end{aligned}
 \tag{87.1}
\]

and

\[
 {\cal P}_{A,abcd}
 =A_{ab}A_{cd}+A_{ac}A_{bd}+A_{ad}A_{bc}.
 \tag{87.2}
\]

Thus \({\cal Z}_A(A)=2{\cal P}_A\).  Let
\({\cal D}=\sum_r e_r^{\otimes4}\), let \(E_{31}(H)\) be the coordinate
Star4 embedding with \(H\) strictly off diagonal, and let \(E_{22}(K)\)
embed the off-diagonal 22 collision array.  Write

\[
 R^\circ=R-I,\qquad J^\circ=\mathbf1\mathbf1^T-I
 \tag{87.3}
\]

for a standardized correlation \(R\).

### 87.1 Exact birth quotient and the hidden identity-pair cancellation

The full trace tensor has birth collisions

\[
 R_{40}{\cal Z}_I(R)=6\mathbf1,\qquad
 R_{31}{\cal Z}_I(R)=3R^\circ,\qquad
 R_{22}{\cal Z}_I(R)=2J^\circ.
 \tag{87.4}
\]

Consequently

\[
\begin{aligned}
 T_{\rm tr}
 :=\Pi_{211}{\cal Z}_I(R)
 ={}&{\cal Z}_I(R)-6{\cal D}
     -E_{31}(3R^\circ)-E_{22}(2J^\circ).
\end{aligned}
 \tag{87.5}
\]

But the 40 and 22 terms are not independent:

\[
 \boxed{
 6{\cal D}+E_{22}(2J^\circ)=2{\cal P}_I.}
 \tag{87.6}
\]

Both sides have K4 equal to \(6\), K22 equal to \(2\), zero off-diagonal
K31, and no strict 211/1111 entry, so (87.6) is a full tensor identity.
Therefore the strongest birth normal form is

\[
 \boxed{
 T_{\rm tr}
 ={\cal Z}_I(R)-2{\cal P}_I-E_{31}(3R^\circ).}
 \tag{87.7}
\]

Literal D11 owns the final term through its arbitrary coordinate Star4
matrix.  It also owns a single pairing plane tied to the actual covariance,
but not the independent identity pairing \({\cal P}_I\) unless a weighted
pair-leg family is explicitly resident.  Treating the constant K22 array as
an ordinary Star4 plane is invalid: Star4 has no off-diagonal 22 support.

### 87.2 Dense transport collapses to one cross-pair word

For one dense transition \(W\), put

\[
 G=WW^T,\qquad C=WRW^T,\qquad
 \Delta=C-G=W(R-I)W^T,
 \tag{87.8}
\]

and let

\[
 {\cal S}_W(H)=W^{\otimes4}E_{31}(H)
 \tag{87.9}
\]

denote the literal transported Star4 tensor.  Tensor naturality and
\({\cal Z}_G(G)=2{\cal P}_G\) give

\[
\begin{aligned}
 W^{\otimes4}T_{\rm tr}
 &={\cal Z}_G(C)-2{\cal P}_G-{\cal S}_W(3R^\circ)\\
 &=\boxed{{\cal Z}_G(\Delta)-{\cal S}_W(3R^\circ).}
\end{aligned}
 \tag{87.10}
\]

This is the consume-and-retire normal form.  The Star4 term rides the
existing D11 star push after the birth update
\(H^4\leftarrow H^4-3R^\circ\).  The only new non-old collision word is the
cross pairing \({\cal Z}_G(\Delta)\).

Let

\[
 g=\operatorname{diag}G,\qquad
 \delta=\operatorname{diag}\Delta.
 \tag{87.11}
\]

Its complete output collision planes are

\[
 \boxed{
 k_4^{\rm tr}=6\,g\odot\delta,}
 \tag{87.12}
\]

\[
 \boxed{
 K_{31}^{\rm tr}
 =3\{g[:,{\rm None}]\odot\Delta
       +\delta[:,{\rm None}]\odot G\},}
 \tag{87.13}
\]

\[
 \boxed{
 K_{22}^{\rm tr}
 =g\delta^T+\delta g^T+4G\odot\Delta.}
 \tag{87.14}
\]

K31 and K22 are read strictly off diagonal.  Equations
(87.12)--(87.14) use only elementwise operations once \(G\) and the already
carried covariance \(C\) are available.

For complete-old-residualized seam costates
\((\lambda_4,\Lambda_{31},\Lambda_{22})\), the exact new collision action is

\[
\boxed{
\begin{aligned}
 S_{\rm cross}={}&
 6\langle\lambda_4,g\odot\delta\rangle\\
 &+3\langle\Lambda_{31},
      g[:,{\rm None}]\odot\Delta
      +\delta[:,{\rm None}]\odot G\rangle_{\rm off}\\
 &+\langle\Lambda_{22},
      g\delta^T+\delta g^T+4G\odot\Delta\rangle_{\rm off}.
\end{aligned}}
 \tag{87.15}
\]

If \({\cal S}_W^*\) denotes the already-paid Star4 adjoint into its birth
matrix, the total strict-source action is

\[
 \boxed{
 S_{\rm tr}=S_{\rm cross}
 -3\langle{\cal S}_W^*
       (\lambda_4,\Lambda_{31},\Lambda_{22}),R^\circ\rangle_{\rm off}.}
 \tag{87.16}
\]

The second line is not a new feature or a new product; it is the literal
old Star4 action with a changed source matrix.

### 87.3 Why separate K22 transport wastes two products

The apparent expensive term is \(E_{22}(2J^\circ)\).  Define

\[
 H_{31}=W^{\odot3}W^T,\qquad
 H_{22}=W^{\odot2}(W^{\odot2})^T,\qquad
 h_4=\operatorname{diag}H_{22}.
 \tag{87.17}
\]

Since
\(E_{22}(2J^\circ)=2{\cal P}_I-6{\cal D}\), its transported collision
planes are

\[
 k_4^{22}=6(g^{\odot2}-h_4),
 \tag{87.18}
\]

\[
 K_{31}^{22}
 =6\{g[:,{\rm None}]\odot G-H_{31}\},
 \tag{87.19}
\]

\[
 K_{22}^{22}
 =2gg^T+4G^{\odot2}-6H_{22}.
 \tag{87.20}
\]

Computing (87.18)--(87.20) literally asks for the three generic products

\[
 WW^T,\qquad W^{\odot3}W^T,\qquad
 W^{\odot2}(W^{\odot2})^T.
 \tag{87.21}
\]

However the transported K4 birth \(6{\cal D}\) contributes exactly

\[
 6h_4,\qquad6H_{31},\qquad6H_{22}
 \tag{87.22}
\]

to the same three collision planes.  Adding (87.22) to
(87.18)--(87.20) cancels both \(H_{31}\) and \(H_{22}\) and gives the
collisions of \(2{\cal P}_G\).  This is (87.6) after transport.

Therefore \(W^{\odot3}W^T\) and
\(W^{\odot2}(W^{\odot2})^T\) are **not** lawful incremental products for
the jointly residualized strict trace source.  They may be reused if already
formed by the diagonal Star4 push, but computing them afresh is an algebraic
failure to fuse the K4 and constant-K22 subtractions.

### 87.4 Exactness of retirement

Let \({\cal R}_{\rm coll}\) read the full K4/K31/K22 planes at the consuming
seam and let \(F\) be everything downstream of that seam.  Retiring the
source after (87.12)--(87.14) is exact if and only if

\[
 \boxed{
 \ker{\cal R}_{\rm coll}\subseteq\ker F
 \quad\Longleftrightarrow\quad
 F=O{\cal R}_{\rm coll}\quad\hbox{for some }O.}
 \tag{87.23}
\]

Equivalently the synchronized downstream Bellman costate must have no
q211/1111 component at this seam.  A Q24 call that consumes only the three
collision planes satisfies the typed condition locally.  A retained CP4
state or any later consumer of noncollision fourth-order entries does not.
Same-seam collision equality alone cannot justify retirement across such a
consumer.

Complete-old residualization of the seam costates precedes (87.15).
Likewise, if the full Section 49 surface is the comparator, the trace
response is already present and has zero additive credit even though
(87.10) is computationally exact.

### 87.5 Minimum product census and the one-equivalent gate

Assume \(C=WRW^T\) is already part of the lawful covariance recurrence.
Let

- \(\chi_{G,l}=0\) if \(G_l=W_lW_l^T\) is resident or obtained by replacing
  an already-billed product, and \(1\) if it requires a new generic Gram;
- \(c_{\star,l}=0\) when the literal D11 Star4 push accepts
  \(-3R_l^\circ\) with no new product, and otherwise its honest incremental
  product bill;
- \(\epsilon_l\) be the measured matmul-equivalent bill for the elementwise
  formations, collision dots, hooks, and writes in
  (87.11)--(87.16).

For a set \({\cal B}\) of consume-and-retire births, the exact incremental
gate is

\[
 \boxed{
 C_{\rm inc}
 =\sum_{l\in{\cal B}}
   \{\chi_{G,l}+c_{\star,l}+\epsilon_l\}\le1.}
 \tag{87.24}
\]

Equation (87.24), not merely “one Gram,” is the binding all-in gate.  Under
a dense-product-only screen \(\epsilon_l\) is temporarily omitted; the
candidate then costs at most one product only if at most one live seam needs
a new \(G_l\).  Ten nonresident births cost ten products, despite each
source retiring after one transition.

The disposition is exact:

1. **GO:** (87.23) holds; the seam costates are complete-old projected;
   Star4 is synchronized with \(c_\star=0\); (87.24) passes; and the
   residual endpoint action clears the value gate.
2. **HOLD:** \(G\), the Star4 insertion seam, the downstream collision-only
   type, or the lower-order \(\epsilon_l\) bill is not literally verified.
3. **NO-GO:** (87.24) fails; a generic q211/1111 consumer remains after the
   seam; or the implementation separately forms either product in
   (87.21) after the fusion (87.6) is available.

Exact resident-product exceptions are:

- \(G=WW^T\) is already a paid row-Gram/covariance-baseline matrix;
- \(W\) is scaled orthogonal, block-monomial, or otherwise has a known
  \(G\) constructible in \(O(n^2)\) without a dense Gram;
- the weighted identity-pair leg \({\cal P}_I\mapsto{\cal P}_G\) is already
  in a resident pair family;
- one new Gram replaces, rather than supplements, an existing covariance
  product and the old \(C\) remains recoverable; or
- for a single low-rank endpoint, a separately certified factor contraction
  computes (87.15) without materializing generic full seam costates.

The last exception is target-specific and does not establish a reusable
carrier.

Graph-ready split:

- TRACE_211_BIRTH_IDENTITY_PAIR_FUSION -- **solved exact**, (87.4)--(87.7);
- TRACE_211_ONE_STEP_CROSS_PAIR -- **solved exact**, (87.8)--(87.16);
- TRACE_211_CONSTANT_K22_THREE_PRODUCT_TRAP -- **closed by fusion**,
  (87.17)--(87.22);
- TRACE_211_CONSUME_RETIRE_TYPE_GATE -- **conditional exact**, (87.23);
- TRACE_211_ONE_GRAM_COST_GATE -- **conditional GO**, exactly (87.24);
- TRACE_211_MULTI_SEAM_NONRESIDENT -- **NO-GO above one new Gram total**;
- TRACE_211_RESIDENT_GRAM_EXCEPTION -- **open literal receipt**;
- TRACE_211_WITH_SECTION49 -- **closed duplicate**.

## 89. Exact consume-or-carry theorem for a six-ray/native AABC source

The six-ray polarization solves a representation problem in the full
symmetric fourth-tensor space.  It does not by itself solve either an
external equality-mask problem or a nonlinear-reset closure problem.  This
section gives the exact one-transition escape and the corresponding
target-rank recurrence.

### 89.1 Three objects that must not be identified

Let

\[
 \Phi(A,B,C;a)=a\,\operatorname{Sym}(A,A,B,C)
 \in\operatorname{Sym}^4(\mathbb R^n).
 \tag{89.1}
\]

There are three different typed objects:

1. the **full tensor** \(\Phi\);
2. the **external equality-mask**
   \(P_{211}^{\rm ext}\Phi\), retaining only entries with exactly three
   distinct external labels and one doubled label; and
3. a **latent Hermite-211 component**, obtained by a chaos projection in the
   latent Gaussian coordinates.

For a linear external map \(W\), the full tensor is natural:

\[
 W^{\otimes4}\Phi(A,B,C;a)
 =\Phi(WA,WB,WC;a).
 \tag{89.2}
\]

The external mask intertwines on a live source space \(\mathcal S\) if and
only if

\[
 [P_{211}^{\rm ext},W^{\otimes4}]\,\mathcal S=0.
 \tag{89.3}
\]

A generic dense \(W\) fails (89.3): a coordinate monomial
\(x_i^2x_jx_k\) is sent to a polynomial having 4, 31, 22, 211, and 1111
equality sectors.  Monomial/permutation-diagonal maps satisfy (89.3)
globally; structured maps need the restricted identity (89.3) on the
actual live span.  Likewise a latent Hermite projection is persistent only
when its own projection commutes with the literal reset.  Neither fact
follows from (89.2).

If \(A=e_i,B=e_j,C=e_k\) are three distinct coordinate axes, then

\[
 \Phi(A,B,C;a)=P_{211}^{\rm ext}\Phi(A,B,C;a),
 \tag{89.4}
\]

so the full tensor is already a strict coordinate-211 source.  For a dense
frame, \((I-P_{211}^{\rm ext})\Phi\) generically contains a 1111 component
as well as K4/K31/K22 collisions.  Sewing only the three collision-plane
families is therefore insufficient for a dense strict-211 birth.

### 89.2 Four-ray self-cancellation and the safe six-ray fallback

The four-ray coordinate construction has diagonal polynomial

\[
 p_g=12a_gA_g^2B_gC_g
 +4a_g(B_g^3C_g+B_gC_g^3).
 \tag{89.5}
\]

Write its Star4 leakage as

\[
 L_g=4a_g(B_g^3C_g+B_gC_g^3).
\]

A collection of four-ray groups is an autonomous full-tensor carrier if and
only if

\[
 \boxed{\sum_gL_g=0\quad\text{in }\operatorname{Sym}^4(\mathbb R^n).}
 \tag{89.6}
\]

Then the sign-averaged collection self-cancels before transport, and (89.2)
preserves the cancellation under every later linear map.  Cancellation only
of the displayed K31 rows, or only after a coordinate Star4 projection, is
not enough: a later dense map can expose the remaining tensor.  When (89.6)
is not proved, the six-ray identity is the safe full-tensor representation
because its leakage vanishes atom by atom rather than only in an old
quotient.

Equation (89.6) still does not prove nonlinear-reset closure.  It removes the
CP-plus-Star sewing problem, not the joint-gate/reset-rank problem.

### 89.3 The exact one-transition consume-and-retire criterion

Let \(\mathcal S_l\) be the lawful full-tensor source span at a birth seam,
let

\[
 T_l=W_l^{\otimes4}:\mathcal S_l\longrightarrow
 \operatorname{Sym}^4(\mathbb R^n)
\]

be the complete linear segment before the next nonlinear reset, let
\(C_{l+1}\) be the literal resident consumption interface at its destination,
and let \(F_{l+1}\) be the complete future scalar/portfolio query.  The
source may be consumed into \(C_{l+1}\) and retired before the reset if and
only if

\[
 \boxed{
 \ker(C_{l+1}T_l|_{\mathcal S_l})
 \subseteq
 \ker(F_{l+1}T_l|_{\mathcal S_l}).}
 \tag{89.7}
\]

Equivalently, there is a matrix \(A_l\) such that

\[
 F_{l+1}T_l|_{\mathcal S_l}
 =A_lC_{l+1}T_l|_{\mathcal S_l}.
 \tag{89.8}
\]

This is necessary and sufficient, including for nonlinear decoders: on the
linear source fiber, an exact decoder for all mixtures must factor through
the quotient by \(\ker C_{l+1}T_l\).  A direct compiled endpoint scalar is
the rank-one special case of (89.8), and needs no forward carrier after the
action is formed.

If \(C_{l+1}\) exposes only K4/K31/K22/strict-211 planes, the unobserved
1111 sector gives the immediate obstruction.  For a generic dense \(W_l\),

\[
 P_{1111}^{\rm ext}T_l\Phi(e_i,e_i,e_j,e_k;1)\ne0.
 \tag{89.9}
\]

Hence plane-only retirement is exact precisely when the complete future
1111 adjoint annihilates this transported source (or a literal 1111 read is
added).  Static three-K31/four-K22 formulas describe visible faces; they do
not prove (89.7).

### 89.4 Collision sewing is a quotient-naturality condition

Suppose the desired source is a quotient class \([S_l]\) modulo the complete
old space \(\mathcal O_l\), and a cheap full representative is
\(\widehat S_l=S_l+O_l\).  Using that representative for one transition is
lawful if and only if

\[
 \boxed{
 F_{l+1}T_lO_l=0,
 }
 \tag{89.10}
\]

or, with a resident old interface at the destination,

\[
 T_lO_l\in\mathcal O_{l+1}+\ker F_{l+1}.
 \tag{89.11}
\]

For a family of old representatives, (89.11) must hold on their whole span.
This is the exact collision/compensator sewing theorem.  K4/K31/K22 ownership
at birth is insufficient when their chosen full tensors leave a transported
1111 component.  Literal coordinate AABC births avoid the premise by (89.4);
dense strict-mask births do not.

### 89.5 Minimum target-conditioned continuation rank

Choose finite bases on \(\mathcal S_l\), stack all preregistered future
queries in \(F_{l+1}\), and put

\[
 M_C=C_{l+1}T_l|_{\mathcal S_l},\qquad
 M_F=F_{l+1}T_l|_{\mathcal S_l}.
\]

The exact minimum number of additional scalar channels needed beyond the
resident consumer is

\[
 \boxed{
 q_l^{\rm extra}
 =\operatorname{rank}\!\begin{bmatrix}M_C\\M_F\end{bmatrix}
  -\operatorname{rank}M_C.}
 \tag{89.12}
\]

Thus \(q_l^{\rm extra}=0\) is exactly the retire condition (89.7).  For a
multi-reset carrier, the future row family is pulled backward through the
literal reset adjoints before applying (89.12).  Equivalently, if
\(\mathcal R_l\) is the resident/old row module, the target recurrence is

\[
 \mathcal F_l
 =\frac{P_{\mathcal S_l}T_l^*\mathcal F_{l+1}+\mathcal R_l}
        {\mathcal R_l}.
 \tag{89.13}
\]

Its dimension is the minimum target-conditioned continuation rank.  A
six-ray Waring rank, a three-generator Chow rank, and the rank in (89.13)
answer three different questions.

If the state crosses a coordinate-dependent nonlinear reset
\(\mathfrak R_l\), exact persistence additionally requires a cyclic module
\(V_l\) satisfying

\[
 P_{{\rm old},l+1}^{\perp}\mathfrak R_lV_l\subseteq V_{l+1}.
 \tag{89.14}
\]

The constructive rank explosion in Section 67 shows that (89.14) is
generically broad.  Six rays eliminate representation leakage but do not
make the joint ReLU gate separable.

### 89.6 Exact lifetime cap and value gate

Let \(d_g\) be the number of linear seams for which group \(g\) remains live,
and put \(H=\sum_gd_g\).  In the conservative rank-one-update census,

\[
 C_{6\rm ray}^{\rm dom}={18H\over1024},\qquad
 C_{4\rm ray}^{\rm dom}={12H\over1024},\qquad
 C_{native}^{\rm dom}={10H\over1024}.
 \tag{89.15}
\]

The four-ray number is lawful only under the full-tensor cancellation
(89.6), and the native number still needs the reset closure (89.14).
Ten consecutive births carried to the end have \(H=55\), so the safe
six-ray dominant count is

\[
 {990\over1024}=0.966796875,
 \tag{89.16}
\]

before source acquisition, collision sewing, hooks, writes, and error
control.  By contrast, ten births each consumed after one linear transition
have \(H=10\) and cost

\[
 {180\over1024}=0.17578125.
 \tag{89.17}
\]

This is the precise economic value of (89.7): it replaces triangular
persistence by one-transition incidence.  The full cap is

\[
 C_{\rm dom}+C_{\rm source}+C_{\rm collision}
 +C_{\rm query}+C_{\rm hooks}+C_{\rm error}\le1.
 \tag{89.18}
\]

Finally the source amplitude must be the complete-old residual of
\(\kappa_4(Y^{\rm true})-\kappa_4(C)\), not the already-owned AQ
commutator.  For a computed corpus feature \(\widehat f_\perp\), with
\(\|f_\perp-\widehat f_\perp\|_W\le\epsilon_f\), the lawful robust value
gate remains

\[
 \boxed{
 \Delta_{\rm MSE}^{\rm true}\ge
 {\bigl(|\langle y_\perp,\widehat f_\perp\rangle_W|
       -\|y_\perp\|_W\epsilon_f\bigr)_+^2
  \over(\|\widehat f_\perp\|_W+\epsilon_f)^2}.}
 \tag{89.19}
\]

Reaching raw \(10^{-8}\) from \(2.9409\times10^{-8}\) requires the joint
complete-old gain to reach \(1.9409\times10^{-8}\); neither a nonzero local
AABC amplitude nor a cheap retirement proves that value.

Graph-ready split:

- `Q211_AABC_OBJECT_TYPING` -- **solved exact**: full Sym4, external-211,
  and latent-Hermite-211 are distinct; the external intertwiner is (89.3).
- `Q211_FOUR_RAY_GROUP_CANCELLATION` -- **solved iff** by the full-tensor
  equality (89.6); plane-only cancellation is insufficient.
- `Q211_SIX_RAY_ONE_TRANSITION_RETIRE` -- **solved iff** by
  (89.7)--(89.8).
- `Q211_DENSE_AABC_1111_OBSTRUCTION` -- **solved generic no-go** by (89.9):
  K4/K31/K22/211 faces alone do not own a dense transported word.
- `Q211_AABC_COLLISION_NATURALITY` -- **solved iff / deployed open** by
  (89.10)--(89.11).
- `Q211_TARGET_CONTINUATION_RANK` -- **solved exact** by
  (89.12)--(89.13).
- `Q211_SIX_RAY_NONLINEAR_RESET` -- **generic HOLD** pending the cyclic
  module (89.14) or its endpoint-weighted tail.
- `Q211_SIX_RAY_LIFETIME_CAP` -- **solved census**: persistent ten-birth
  route is cap-tight at (89.16); one-transition retirement gives (89.17).
- `Q211_AABC_SOURCE_AND_VALUE` -- **open blocking**: lawful
  true-minus-comparator amplitudes and robust gain (89.19).

## 88. Exact CP3 fusion of the zero-mean Gaussian q211 surface: one broad
sandwich is the only non-matvec complement

Section 86 proves that the full q211 kernel has infinite separable rank if
viewed in isolation.  That is not the correct typed quotient when the same
seam already owns the exact third cumulant.  The entire trivariate
nonpolynomial part can be isolated as one CP3 term.

### 88.1 A two-variable formula for the repeated-slot raw moment

Let \(X,Y,Z\) be standardized centered Gaussian variables with

\[
 E[XY]=x,\qquad E[XZ]=y,\qquad E[YZ]=z.
\]

In addition to \(p,q,t,\mu\) from (86.3)--(86.4), define

\[
 h(r):=E[|U||V|]
 ={2\over\pi}\{\sqrt{1-r^2}+r\arcsin r\}, \tag{88.1}
\]

\[
 s(r):=E[U|U|\operatorname{sign}V]
 ={2\over\pi}\{\arcsin r+r\sqrt{1-r^2}\}. \tag{88.2}
\]

Then the repeated-slot raw moment has the exact elementary form

\[
\boxed{\begin{aligned}
 M_{211}={}&E[X_+^2Y_+Z_+]\\
 ={}&{1\over2}q(z)
 +{x^2+y^2\over4\pi}\sqrt{1-z^2}
 +xy\,p(z)\\
 &+{1\over4}\{x h(y)+y h(x)\}
 +{z\over8}\{s(x)+s(y)\}.
\end{aligned}} \tag{88.3}
\]

Proof is short.  Use

\[
 X_+^2={1\over2}(X^2+X|X|). \tag{88.4}
\]

For \(g(Y,Z)=Y_+Z_+\), second-order Gaussian integration by parts gives

\[
\begin{aligned}
 E[X^2g]={}&q(z)
 +{x^2+y^2\over2\pi}\sqrt{1-z^2}
 +2xy\,p(z), \tag{88.5}
\end{aligned}
\]

because

\[
 E[\delta(Y)Z_+]=E[Y_+\delta(Z)]
 ={1\over2\pi}\sqrt{1-z^2},\qquad
 E[I_YI_Z]=p(z).
\]

Global sign parity and first-order Stein identities give

\[
\begin{aligned}
 E[X|X|Y_+Z_+]
 ={}&{1\over2}\{x h(y)+y h(x)\}\\
 &+{z\over4}\{s(x)+s(y)\}, \tag{88.6}
\end{aligned}
\]

where, for example,

\[
 E[X|X|Y|Z|]=2x h(y)+z s(y). \tag{88.7}
\]

Equations (88.4)--(88.7) prove (88.3).  They also explain why the apparent
trivariate boundary terms in (86.15) cancel from \(M_{211}\).

### 88.2 Exact CP3-plus-edge decomposition

Let

\[
 K_3(x,y,z):=\kappa_3(X_+,Y_+,Z_+)
 =M_{111}-\mu\{q(x)+q(y)+q(z)\}+2\mu^3. \tag{88.8}
\]

Define

\[
\begin{aligned}
 A(x,y,z):={}&
 {x^2+y^2\over4\pi}\sqrt{1-z^2}
 +xy\,p(z)\\
 &+{1\over4}\{x h(y)+y h(x)\}
 +{z\over8}\{s(x)+s(y)\}. \tag{88.9}
\end{aligned}
\]

Substitution of (88.3) and (88.8) into the complete cumulant partition
(86.16) yields

\[
\boxed{\begin{aligned}
 K(x,y,z)={}&-2\mu K_3(x,y,z)+F_{\rm edge}(x,y,z),\\
 F_{\rm edge}:={}&A(x,y,z)-2q(x)q(y)-\mu\{t(x)+t(y)\}\\
 &+\mu^2+2\mu^2\{q(x)+q(y)\}-2\mu^4.
\end{aligned}} \tag{88.10}
\]

This is an exact identity on the positive-definite correlation domain and
extends continuously to its boundary.  For non-unit zero-bias variables,
the CP3 term is

\[
 -2\mu\sigma_i\,
 \kappa_3(Y_i,Y_j,Y_k), \tag{88.11}
\]

and every term in \(F_{\rm edge}\) is multiplied by
\(\sigma_i^2\sigma_j\sigma_k\).

Therefore the infinite-rank obstruction in (86.26)--(86.27) is precisely
compatible with (88.10): the resident CP3 tensor carries the infinite-rank
trivariate component.  After quotienting by an exact synchronized CP3
owner, the complement is a finite list of univariate edge kernels and their
trace words.  If CP3 is only approximate or at a different cut, (88.10)
is a joint correction identity, not a duplication claim.

### 88.3 Exact CP endpoint action and the one broad product

Consider one ordered partially symmetric endpoint atom

\[
 L_{i;jk}=\alpha_i\beta_j\gamma_k
\]

with strict-collision exclusions applied afterward.  Write elementwise
matrices

\[
 P=p[R],\quad Q=q[R],\quad H=h[R],\quad S=s[R],\quad
 T=t[R],\quad
 A_0={1\over4\pi}\sqrt{1-R^{\odot2}}. \tag{88.12}
\]

Every term of (88.10) except \(xy\,p(z)\) is a bounded list of matrix
matvecs and Hadamard products.  For example,

\[
\sum_{ijk}\alpha_i\beta_j\gamma_k
 R_{ij}^2(A_0)_{jk}
=\alpha^T R^{\odot2}\{\beta\odot(A_0\gamma)\}, \tag{88.13}
\]

\[
\sum_{ijk}\alpha_i\beta_j\gamma_k
 R_{ij}h(R_{ik})
=\alpha^T\{(R\beta)\odot(H\gamma)\}, \tag{88.14}
\]

\[
\sum_{ijk}\alpha_i\beta_j\gamma_k
 R_{jk}s(R_{ij})
=\alpha^T S\{\beta\odot(R\gamma)\}. \tag{88.15}
\]

The only dense sandwich is

\[
\boxed{
 B_{\alpha\beta\gamma}
 :=\sum_{ijk}\alpha_i\beta_j\gamma_k
 R_{ij}R_{ik}p(R_{jk})
 =\operatorname{tr}
 (D_\alpha R D_\beta P D_\gamma R).} \tag{88.16}
\]

It needs one query-specific dense product if no identical product is
resident.  All collision exclusions in (88.13)--(88.16) are two-index
corrections and do not create another dense product.

Consequently the exact zero-mean Gaussian q211 action has the sharp
implementation hierarchy:

1. a literal same-seam CP3 action plus a resident instance of (88.16):
   only a fixed number of matvecs is new;
2. a literal CP3 action but no resident sandwich: one new GEMM per active
   seam, which exceeds a one-equivalent ten-seam budget;
3. no exact CP3 action: the dense trivariate surface or a controlled
   Section 78 expansion remains necessary.

The fusion condition is literal equality of the paid product, including
\(\alpha,\beta,\gamma\), \(P\), cut, and collision mask.  Sharing \(R\), a
CP3 family name, or storage is not enough.

### 88.4 Ownership and value gates

Let \(J_3\) be the literal synchronized CP3 row and let \(J_{\rm edge}\)
contain the actual Q24/pair/trace edge rows.  Complete old ownership of
(88.10) is exactly

\[
 K\in\operatorname{row}(J_3,J_{\rm edge}). \tag{88.17}
\]

If the fused Section 49 surface is present, (88.17) holds by construction
and there is zero additive credit.  In the carried-\(P_{\le2}\) branch,
the CP3 term may be old while some edge complement remains a novel typed
readout.  In the true-minus-AQ branch, apply (88.10) separately to the
declared Gaussian comparator; it does not determine the dependent residual
law.

The literal D11 audit closes the optimistic current-owner premise.  Its
stored CP tensor is the finite closure
\(\widehat T_3=\sum_s\lambda_s\operatorname{Sym}(p_s,p_s,q_s)\), with only
the declared K21/diagonal acquisition reads.  At the first zero-bias reset
its all-distinct birth contains the finite
\(\mu(xy+xz+yz)/4\) closure term, whereas exact Gaussian \(K_3\) also
contains the \(\mu^3xyz\) sector and the infinite higher Hermite tail.
Coordinate Star4 has no all-distinct CP3 acquisition port.  Therefore

\[
 \delta K_3:=K_3^{\rm exact}-\widehat T_3\ne0 \tag{88.18}
\]

generically at the same seam, and (88.10) leaves
\(-2\mu\delta K_3\) in the residual.  The biased threshold JVP is absent as
well.  Thus (88.10) is a replacement/joint-correction factorization for the
current lineage, not a free reuse theorem.

Graph-ready split:

- GAUSSIAN_Q211_M211_EDGE_FORMULA -- **solved exact**, (88.3);
- GAUSSIAN_Q211_CP3_FUSION -- **solved exact**, (88.10)--(88.11);
- GAUSSIAN_Q211_EDGE_MATVEC_COMPLEMENT -- **solved exact**,
  (88.12)--(88.15);
- GAUSSIAN_Q211_BROAD_SANDWICH -- **one-product theorem**, (88.16);
- GAUSSIAN_Q211_LITERAL_CP3_OWNER -- **literal current D11 NO** by
  (88.18); replacement exact owner remains conditional;
- GAUSSIAN_Q211_SANDWICH_FUSION -- **open deployed receipt**;
- GAUSSIAN_Q211_WITH_FULL_AQ_SURFACE -- **closed duplicate**;
- GAUSSIAN_Q211_TRUE_MINUS_AQ -- **not determined by this identity**.

## 90. Arbitrary-bias Gaussian q211 is a first threshold jet of CP3

The zero-bias CP3 fusion in Section 88 is the symmetric special case of an
exact Stein differential identity.  For arbitrary thresholds, Gaussian
q211 still does not require an algebraically independent four-point surface:
it requires the CP3 value, its first threshold jet, and pair-only terms.

### 90.1 Raw Stein identity

Let

\[
 \rho_a=(X_a-\tau_a)_+,\qquad
 E[X_iX_j]=x,\quad E[X_iX_k]=y,\quad E[X_jX_k]=z,
\]

and define

\[
 M_{111}=E[\rho_i\rho_j\rho_k],\qquad
 M_{211}=E[\rho_i^2\rho_j\rho_k]. \tag{90.1}
\]

Because \(\rho_i^2=(X_i-\tau_i)\rho_i\), Stein's identity gives

\[
\begin{aligned}
 M_{211}={}&-\tau_i M_{111}
 +E[I_i\rho_j\rho_k]
 +xE[\rho_i I_j\rho_k]
 +yE[\rho_i\rho_j I_k].
\end{aligned} \tag{90.2}
\]

Threshold differentiation of the same raw triple moment gives

\[
 \partial_{\tau_i}M_{111}=-E[I_i\rho_j\rho_k],
 \tag{90.3}
\]

and the analogous identities for \(j,k\).  Therefore

\[
\boxed{
 M_{211}
 =-\left(\tau_i+\partial_{\tau_i}
          +x\partial_{\tau_j}
          +y\partial_{\tau_k}\right)M_{111}.}
 \tag{90.4}
\]

This is exact for every positive-semidefinite correlation triple, with
boundary values defined by continuity.  It uses no chaos truncation or
small-correlation assumption.

### 90.2 Connected-cumulant quotient

Write

\[
 m_a=E[\rho_a],\qquad M_{ab}=E[\rho_a\rho_b],
\]

and

\[
 K_3=M_{111}
 -m_iM_{jk}-m_jM_{ik}-m_kM_{ij}
 +2m_im_jm_k. \tag{90.5}
\]

Define

\[
 B_3:=m_iM_{jk}+m_jM_{ik}+m_kM_{ij}-2m_im_jm_k,
 \tag{90.6}
\]

and let

\[
\begin{aligned}
 R_{\le2}:={}&
 -m_jM_{iik}-m_kM_{iij}
 -M_{ii}M_{jk}-2M_{ij}M_{ik}\\
 &+2\{M_{ii}m_jm_k+2M_{ij}m_im_k
       +2M_{ik}m_im_j+M_{jk}m_i^2\}\\
 &-6m_i^2m_jm_k.
\end{aligned} \tag{90.7}
\]

Every object in (90.6)--(90.7) has support on at most two distinct unit
labels, apart from products of such objects.  Substituting (90.4) into the
complete fourth-cumulant partition yields

\[
\boxed{\begin{aligned}
 \kappa_4(\rho_i,\rho_i,\rho_j,\rho_k)
 ={}&-\mathcal D_i K_3+F_{\le2},\\
 \mathcal D_i:={}&
 \tau_i+2m_i+\partial_{\tau_i}
 +x\partial_{\tau_j}+y\partial_{\tau_k},\\
 F_{\le2}:={}&R_{\le2}-\mathcal D_i B_3.
\end{aligned}} \tag{90.8}
\]

Thus \(F_{\le2}\) is an explicit finite composition of means, bivariate
truncated moments, and their threshold derivatives.  The only genuinely
three-unit object is \(K_3\) and its first threshold jet.

For \(Y_a=\sigma_a\rho_a\), (90.8) becomes

\[
\boxed{
 \kappa_4(Y_i,Y_i,Y_j,Y_k)
 =-\sigma_i\mathcal D_i
   \kappa_3(Y_i,Y_j,Y_k)
 +\sigma_i^2\sigma_j\sigma_k F_{\le2}.}
 \tag{90.9}
\]

At \(\tau_i=\tau_j=\tau_k=0\), parity eliminates the independent threshold
jet and (90.9) reduces to the value-only CP3 fusion (88.10).

### 90.3 Typed ownership and the exact new channel

Equation (90.9) distinguishes four separate rows:

1. the CP3 value \(K_3\);
2. the anchor threshold derivative \(\partial_{\tau_i}K_3\);
3. the two leaf threshold derivatives, weighted by \(x\) and \(y\); and
4. the pair composition \(F_{\le2}\).

A literal CP3 value factorization owns only row 1.  It owns rows 2--3 if and
only if its same-seam evaluator exposes the corresponding mixed
gate/ReLU cumulants, equivalently its exact threshold JVP.  Reusing CP3
directions, covariance, quadrature nodes, or a family name is not a JVP
receipt.

For a frozen endpoint \(L\), define the CP3 scalar

\[
 S_3(\tau)=\langle L,K_3(\tau)\rangle. \tag{90.10}
\]

Only when the weights multiplying each slot derivative factor through one
common threshold direction does the jet action reduce to one directional
derivative of (90.10).  The factors \(x=R_{ij}\) and \(y=R_{ik}\) generically
make the leaf directions pair-dependent.  Otherwise the threshold jet is a
new typed reverse query whose target rank must be certified; a scalar output
does not make its acquisition free.

The exact complete-old test is

\[
 \mathcal D_iK_3\in
 \operatorname{row}(J_{CP3},J_{CP3,\tau},J_{Q24},J_{\rm pair}), \tag{90.11}
\]

using literal rows at the synchronized cut.  If the fused Section 49
surface is part of the comparator, (90.8) is already owned wholesale.  If
only the CP3 value is old, the lawful candidate is the residual threshold
jet plus the pair complement, never the total q211 surface.

### 90.4 Consequence for the cubic-provider search

The inverse-budget requirement for an independent cubic source must now be
read carefully.  In the affine-Gaussian comparator, q211 is generated by the
CP3 value and its first reset-boundary response.  The genuinely new
information, if any, is not another free fourth-order tensor; it is either

\[
 \Pi_{\rm old}^{\perp}J_{\tau}K_3 \tag{90.12}
\]

or the true-minus-Gaussian/dependent surface of Sections 68--81.  This gives
a fail-fast hierarchy:

1. audit the literal CP3 threshold JVP;
2. if absent, measure/certify its target-conditioned rank and cost;
3. only after residualizing it credit a signed q211 feature;
4. treat the remaining true-minus-Gaussian surface as an independent
   source oracle.

Graph-ready split:

- BIASED_GAUSSIAN_Q211_STEIN_IDENTITY -- **solved exact**, (90.4);
- BIASED_GAUSSIAN_Q211_CP3_JET_CLOSURE -- **solved exact**, (90.8)--(90.9);
- CP3_VALUE_OWNER -- **separate old row**, not JVP ownership;
- CP3_THRESHOLD_JVP_OWNER -- **open literal receipt**, gate (90.11);
- CP3_THRESHOLD_JVP_TARGET_RANK -- **open** unless one common direction
  factors (90.10);
- BIASED_GAUSSIAN_Q211_PAIR_COMPLEMENT -- **finite edge composition**,
  explicit in (90.6)--(90.8);
- BIASED_GAUSSIAN_Q211_WITH_FULL_SURFACE -- **closed duplicate**;
- BIASED_TRUE_MINUS_GAUSSIAN_Q211 -- **separate independent source**.

## 94. Projected q1111 retirement after a q211 birth: trace injectivity,
CP4 pullback, and the exact CP3-jet/broad-sandwich seam

A strict external-q211 tensor generally acquires every external collision
type, including q1111, after a dense change of basis.  This does not by
itself force a forward q1111 carrier.  The right question is whether the
declared consumer rows determine the part seen by the actual future CP4
portfolio.  This section gives the exact projected test and then specializes
it to the Section 87 trace word and the Sections 88/90 Gaussian source.

### 94.1 Exact projected-retirement rank

Let (B:\mathbb R^d\to\mathcal S_{211}) parameterize the lawful birth-source
space, let (T=W^{\otimes4}), and let (K) collect every literal known row
at the consuming seam: complete-old rows, collision reads, synchronized CP3
value/JVP rows, and any resident pair or broad-sandwich rows.  For future
queries (Q_1,\ldots,Q_p\in\operatorname{Sym}^4), define

\[
 M_K=KTB,
 \qquad
 (M_{1111})_{s,:}=\langle P_{1111}Q_s,TB(\,\cdot\,)\rangle .
 \tag{94.1}
\]

Then the exact minimum number of additional scalar rows needed solely for
the q1111 endpoint actions is

\[
 \boxed{
 r_{1111}^{\rm add}
 =\operatorname{rank}\!\begin{bmatrix}M_K\\M_{1111}\end{bmatrix}
  -\operatorname{rank}M_K
 =\operatorname{rank}\{M_{1111}\Pi_{\ker M_K}\}.}
 \tag{94.2}
\]

In particular, projected retirement is exact iff

\[
 \boxed{\ker M_K\subseteq\ker M_{1111}.} \tag{94.3}
\]

The identical statement with (P_{211}+P_{1111}) replaces (94.1) when the
future interface may read both noncollision sectors.  For one preregistered
scalar endpoint, (94.2) is either zero or one; for an adaptive rank-(p)
CP4 portfolio it can be as large as (p).  Thus a direct terminal scalar
can retire a source that is not a reusable forward state.

This is a target rank, not a product bill.  A row may be information-theoretic
rank one while its literal evaluation still requires a dense product.

### 94.2 The trace cross-pair has zero additional q1111 information rank

After the Section 87 identity-pair fusion, the non-old trace word is

\[
 Z={\cal Z}_G(\Delta),\qquad
 G=WW^T,\quad \Delta=W(R-I)W^T. \tag{94.4}
\]

Put (g_i=G_{ii}), (\delta_i=\Delta_{ii}), and denote its raw K4 and
ordered K31 faces by

\[
 k_i=6g_i\delta_i,
 \qquad
 L_{ij}=3(g_i\Delta_{ij}+\delta_iG_{ij}),\quad i\ne j. \tag{94.5}
\]

On the active support (A=\{i:g_i>0\}), these faces recover the whole
matrix (\Delta):

\[
 \boxed{
 \delta_i={k_i\over6g_i},\qquad
 \Delta_{ij}={L_{ij}\over3g_i}-{G_{ij}k_i\over6g_i^2}.}
 \tag{94.6}
\]

If (g_i=0), the Gram identity implies that row (i) of (W) is zero;
because (\Delta) has the same two-sided (W) factor, row and column (i)
of both (G) and (\Delta) vanish.  Hence (94.6) reconstructs all
physically nonzero entries.  Therefore

\[
 \boxed{
 (P_4,P_{31}^{\rm ordered})\big|_{\{{\cal Z}_G(\Delta)\}}
 \text{ is injective},\qquad r_{1111}^{\rm add}=0.}
 \tag{94.7}
\]

K22 is useful to the declared consumer but is not needed for this
information theorem.  A symmetric or compressed K31 summary is insufficient
unless it is separately proved injective on the same physical family.

For a full CP4 ray (u^{\otimes4}), the complete cross-pair action collapses
to

\[
 \boxed{
 \langle u^{\otimes4},{\cal Z}_G(\Delta)\rangle
 =6(u^TGu)(u^T\Delta u).} \tag{94.8}
\]

Equivalently, (94.6) expresses (94.8) directly through (k,L,G).  The
q1111-only action is (94.8) minus its 4/31/22/211 strata; it is still
determined by (94.6).  Equation (94.8) is for the full tensor, not for the
q1111 projection alone.

There is one indispensable qualification.  The complete transported strict
trace source is

\[
 W^{\otimes4}\Pi_{211}{\cal Z}_I(R)
 ={\cal Z}_G(\Delta)-{\cal S}_W(3R^\circ). \tag{94.9}
\]

The second term must ride the literal old Star4 transport.  It is not
recoverable from the cross-pair collisions.  Indeed, for a ray (u), with
(v=W^Tu),

\[
 \langle u^{\otimes4},{\cal S}_W(3R^\circ)\rangle
 =12\sum_{a\ne b}R_{ab}v_a^3v_b. \tag{94.10}
\]

The failure is exact, not merely generic.  Take one output row
(W=(1,2,3)) and an off-diagonal symmetric (Q) with
(Q_{12}=3,Q_{13}=-2,Q_{23}=0).  Then (WQW^T=0), so the cross word and
all its collision faces vanish, while

\[
 \{W^{\otimes4}\Pi_{211}{\cal Z}_I(I+\epsilon Q)\}[1,1,1,1]
 =360\epsilon. \tag{94.11}
\]

Thus (94.7) closes the new cross word only after the old-Star receipt in
(94.9); it cannot be used to infer the entire strict birth from (G,\Delta).

### 94.3 A literal CP4 endpoint consumes all transported partitions at birth

Let (S\) be any raw symmetric strict-q211 tensor, with
(S_{iijk}=s_{i;jk}=s_{i;kj}).  For one output CP4 ray (u), put
(v=W^Tu).  Adjoint naturality gives

\[
 \boxed{
 \langle u^{\otimes4},W^{\otimes4}S\rangle
 =S[v,v,v,v]
 =6\!\sum_{i}\!\sum_{\substack{j,k\ne i\\j\ne k}}
 s_{i;jk}v_i^2v_jv_k.} \tag{94.12}
\]

The factor is six for the ordered ((j,k)) sum, equivalently twelve for
(j<k).  Equation (94.12) includes the output 4/31/22/211/1111 pieces in
one birth-frame action.  Consequently a literal immediate CP4 endpoint can
consume the q1111 leakage without materializing a q1111 tensor.  A rank-(p)
endpoint uses its (p) pulled-back rays, followed by the exact residual-rank
compression (94.2).

This conclusion does not apply to a CP4 *state that must survive another
nonlinear reset*: then every later costate belongs in (Q_{1:p}), and the
continuation rank can grow.  Nor does a collision-only implementation become
a full CP4 endpoint merely because CP4 factors exist elsewhere in the card.

### 94.4 CP3 value/jet and the broad sandwich consume the Gaussian action,
not a second q1111 copy

For the zero-bias Gaussian source of Section 88, substitute the pulled-back
ray weights

\[
 \alpha=v^{\odot2},\qquad \beta=\gamma=v \tag{94.13}
\]

into the ordered q211 action.  Its CP3 part is the synchronized trilinear
query

\[
 -2\mu\,K_3[\,\sigma\odot v^{\odot2},v,v\,], \tag{94.14}
\]

up to the explicit strict-collision corrections.  If
(p_3(x)=K_3[x,x,x]), exact polarization is

\[
 K_3[a,v,v]
 ={1\over3}Dp_3(v)[a]
 ={p_3(v+a)-p_3(v-a)-2p_3(a)\over6}. \tag{94.15}
\]

Thus a literal same-seam CP3 factor/action or threshold directional jet can
consume (94.14).  A CP3 family name or value at unrelated rays cannot.
For arbitrary bias, Section 90 replaces (94.14) by the colored first
threshold-JVP action

\[
 -\sum_{ijk}^{\rm strict}
 v_i^2v_jv_k\,\sigma_i{\cal D}_iK_{3,ijk}. \tag{94.16}
\]

The (R_{ij},R_{ik})-weighted leaf derivatives in ({\cal D}_i) generally
do not factor through one common threshold direction.  They are owned only
by the literal colored JVP, not by CP3 value alone.

Every zero-bias edge term is a matvec/Hadamard action except the one Section
88 broad sandwich.  Under (94.13) it is exactly

\[
 \boxed{
 {\cal B}(v)
 =\operatorname{tr}
  (D_{v^{\odot2}}R D_v P D_vR)
 =\mathbb E\|D_vRD_v\,{\mathbf1}_{\{X>0\}}\|_2^2,\qquad P=p[R].}
 \tag{94.17}
\]

This is the full transported CP4-ray contribution of the rooted broad edge,
including whatever portion lands in q1111.  Dense transport does **not**
create a second broad sandwich.  If the synchronized CP3-threshold provider
already exposes the weighted gate Gram in (94.17), or the identical Section
88 product is resident, it fuses exactly.  CP3 value alone does not own this
pair-threshold Gram.  Strict collision masking adds only the declared
two-label corrections.

Therefore the exact Gaussian consume interface for a CP4 ray is:

1. the synchronized CP3 value at zero bias, or the colored first threshold
   JVP at general bias;
2. the finite pair/edge complement and strict collision corrections;
3. one instance of (94.17), unless that identical sandwich is resident; and
4. for the original trace source, the separate old-Star term (94.10).

Once these rows are included in (M_K), (94.2) decides whether anything is
left.  If the literal full Section 49 AQ surface is the same-seam declared
comparator, the Gaussian/AQ source is already old wholesale and
(r_{1111}^{\rm add}=0) by duplication.  This does not kill a
true-minus-AQ source or a source transported from another cut unless the
actual pulled-back endpoint row is included.

### 94.5 Exact disposition

1. **TRACE GO:** the old Star4 push in (94.9) is literal and ordered K31/K4
   are retained.  Then the new cross-pair q1111 information rank is zero for
   every downstream query by (94.6)--(94.7).
2. **TERMINAL CP4 GO:** all pulled-back rays are preregistered and their full
   birth actions (94.12) are evaluated.  No forward q1111 tensor is needed;
   charge the residual scalar rank (94.2) and the actual products.
3. **GAUSSIAN FUSION GO:** the synchronized CP3 value/colored JVP, pair
   complement, strict corrections, and identical broad sandwich (94.17) are
   all present.  Output q1111 is part of that same action, not a new source.
4. **HOLD:** full-AQ or CP4 is named but the literal cut, query, threshold
   JVP, old-Star push, or broad-product identity is not verified.
5. **NO-GO FOR RETIREMENT:** (94.3) fails for the complete future portfolio,
   or the source must pass another nonlinear reset without a closed
   continuation basis.

Graph-ready split:

- `Q211_PROJECTED_Q1111_RETIREMENT_RANK` -- **solved exact**, (94.1)--(94.3);
- `TRACE_CROSS_PAIR_COLLISION_INJECTIVITY` -- **solved exact**, zero new
  q1111 information rank by (94.4)--(94.7);
- `TRACE_OLD_STAR_SEPARATE_OWNER` -- **binding**, with counterexample
  (94.9)--(94.11);
- `Q211_CP4_ADJOINT_PULLBACK` -- **solved exact**, (94.12);
- `Q211_CP3_DIAGONAL_JET_POLARIZATION` -- **solved exact**, (94.14)--(94.15);
- `Q211_COLORED_THRESHOLD_JVP` -- **conditional owner**, (94.16);
- `Q211_BROAD_SANDWICH_IS_Q1111_COMPLETE` -- **solved exact**, (94.17): no
  second q1111 sandwich;
- `Q211_FULL_AQ_SAME_SEAM` -- **closed duplicate only at the literal same
  source/cut/query**;
- `Q211_FUTURE_CP4_CONTINUATION` -- **open/conditional** on (94.2) over all
  future costates and nonlinear-reset closure.

## 93. Quadratic-AQ q211 is the second latent-mean jet of CP3

The threshold-jet identity in Section 90 extends exactly to the full
noncommutative quadratic-chaos law.  This is the strongest algebraic
compression found in this lane: the AQ q211 surface is a finite tilted jet
of the AQ CP3 surface, not an independent fourth-order law.

### 93.1 Cameron--Martin jet identity

Let \(z\sim N(0,I_r)\), and write the centered quadratic preactivation as

\[
 g_a(z)=b_a+u_a^Tz+z^TA_az-\operatorname{tr}A_a,\qquad
 Y_a=(g_a)_+. \tag{93.1}
\]

For a function \(F\), define its latent-mean tilt

\[
 {\cal E}_F(h):=E_{z\sim N(h,I_r)}F(z). \tag{93.2}
\]

Differentiating the shifted Gaussian density gives

\[
 \nabla_h{\cal E}_F(0)=E[zF(z)],\qquad
 \nabla_h^2{\cal E}_F(0)=E[(zz^T-I_r)F(z)]. \tag{93.3}
\]

Let

\[
 M_{ijk}(h):=E_{N(h,I_r)}[Y_iY_jY_k]. \tag{93.4}
\]

Since \(Y_i^2=g_iY_i\), equations (93.1)--(93.3) imply

\[
\boxed{\begin{aligned}
 E[Y_i^2Y_jY_k]
 ={}&\left[
 b_i+u_i^T\nabla_h+
 \langle A_i,\nabla_h^2\rangle_F
 \right]M_{ijk}(h)\big|_{h=0}.
\end{aligned}} \tag{93.5}
\]

Define the anchor jet operator

\[
 {\cal G}_i:=
 b_i+u_i^T\nabla_h+\langle A_i,\nabla_h^2\rangle_F.
 \tag{93.6}
\]

Equation (93.5) is exact for arbitrary symmetric, mutually
noncommuting \(A_a\).  No diagonalization, commutativity, low correlation,
or chaos truncation is used.

### 93.2 Connected CP3 closure

Let

\[
 K_{3,ijk}(h):=\kappa_3(Y_i,Y_j,Y_k)
\]

under the same tilted latent law.  Define \(B_{3,ijk}(h)\) by

\[
 M_{ijk}(h)=K_{3,ijk}(h)+B_{3,ijk}(h), \tag{93.7}
\]

where \(B_3\) is the usual finite sum of means times pair moments.  Let
\(R_{\le2}^{AQ}\) denote the terms in the fourth-cumulant partition other
than the repeated raw fourth moment and \(-2m_iM_{ijk}\); explicitly it is
the expression (90.7) with AQ means and raw pair/repeated-pair moments.
Then (93.5) gives

\[
\boxed{\begin{aligned}
 \kappa_4(Y_i,Y_i,Y_j,Y_k)
 ={}&({\cal G}_i-2m_i)K_{3,ijk}(h)\big|_{h=0}
 +F_{\le2}^{AQ},\\
 F_{\le2}^{AQ}:={}&
 R_{\le2}^{AQ}
 +({\cal G}_i-2m_i)B_{3,ijk}(h)\big|_{h=0}.
\end{aligned}} \tag{93.8}
\]

All genuinely three-unit AQ dependence is therefore contained in

\[
 K_3(0),\qquad \nabla_hK_3(0),\qquad \nabla_h^2K_3(0). \tag{93.9}
\]

The remainder \(F_{\le2}^{AQ}\) uses only tilted means and pair moments and
their first two latent-mean jets.

For the affine law \(A_i=0\), translating \(h\) shifts the three biases by
\(u_s^Th\).  Thus

\[
 u_i^T\nabla_h
 =-\partial_{\tau_i}
 -R_{ij}\partial_{\tau_j}
 -R_{ik}\partial_{\tau_k}, \tag{93.10}
\]

and (93.8) reduces exactly to (90.8).  At zero bias it further reduces to
the value-only CP3 fusion (88.10).

### 93.3 What is and is not resident

A stored CP3 tensor \(K_3(0)\) owns only the zeroth jet in (93.9).  Exact
q211 ownership requires the contracted first and second latent-mean jets as
well.  Let

\[
 J_3^{(2)}
 :=\left(K_3(0),\nabla_hK_3(0),\nabla_h^2K_3(0)\right). \tag{93.11}
\]

For a strict-q211 endpoint \(L\), its exact AQ three-unit action is

\[
\boxed{\begin{aligned}
 S_L^{AQ}={}&
 \sum_{ijk}L_{i;jk}\big[
 (b_i-2m_i)K_{3,ijk}\\
 &\quad+u_i^T\nabla_hK_{3,ijk}
 +\langle A_i,\nabla_h^2K_{3,ijk}\rangle_F
 \big]+S_{L,\le2}.
\end{aligned}} \tag{93.12}
\]

The literal ownership criterion is

\[
 \ker E_{\rm paid}
 \subseteq\ker\big(L\mapsto S_L^{AQ}\big), \tag{93.13}
\]

where \(E_{\rm paid}\) contains the actual CP3 surface evaluator and any
exposed tilt derivatives.  Shared eigensystems, Fourier nodes, CP factors,
or a CP3 family name do not imply (93.13).

For one partially symmetric CP endpoint atom, the naive jet channel count is

\[
 1+r+{r(r+1)\over2}. \tag{93.14}
\]

At \(r=55\), this is \(1596\).  It can collapse only when the weighted
anchor banks

\[
 \{\alpha_i u_i\}_i,\qquad
 \{\alpha_i A_i\}_i \tag{93.15}
\]

factor through a low-dimensional common algebra or when reverse automatic
differentiation of the literal scalar CP3 evaluator computes the whole
contraction (93.12) without materializing the jet.  The exact
target-conditioned rank is the rank of the coefficient family in
(93.12) after quotienting the complete old row span; the ambient count
(93.14) is not itself a bill, but neither may it be silently set to one.

### 93.4 Replacement and source ledger

An explicitly billed fused H211 AQ surface oracle and the jet closure
(93.8) evaluate the same AQ q211 law.  They are alternative implementations,
not additive features.  Consequently:

1. if the full AQ surface oracle is present, (93.8) receives zero source
   credit;
2. if only CP3 value is carried, the missing lawful AQ feature is the
   complete-old residual of the first/second tilt jet plus
   \(F_{\le2}^{AQ}\);
3. the independent Gaussian-rest heat response is a further covariance
   interpolation around this AQ law;
4. the dependent/non-Gaussian true-minus-AQ residual is not determined by
   any finite jet of (93.4).

The inverse-budget statement that a cubic source is necessary is therefore
compatible with (93.8): within the quadratic-AQ model, the needed source is
the CP3 response jet.  Only the true-minus-AQ component requires genuinely
new distributional information.

Graph-ready split:

- AQ_Q211_CAMERON_MARTIN_IDENTITY -- **solved exact**, (93.5);
- AQ_Q211_CP3_SECOND_JET_CLOSURE -- **solved exact**, (93.8)--(93.9);
- AQ_CP3_VALUE_OWNER -- **does not own tilt jets**;
- AQ_CP3_TILT_JET_OWNER -- **open literal receipt**, (93.13);
- AQ_CP3_TILT_JET_TARGET_RANK -- **open/structured conditional**, 
  (93.12)--(93.15);
- AQ_Q211_JET_VS_FUSED_SURFACE -- **replacement identity, zero double
  credit**;
- AQ_Q211_GAUSSIAN_REST_RESPONSE -- **separate Section 81 source**;
- AQ_Q211_TRUE_MINUS_AQ -- **separate independent source**.

## 91. Exact intrinsic Gaussian surface defect: quadratic owner subtraction,
signed local cone, and endpoint-safe approximation gates

Sections 78 and 86 must be compared only after subtracting the same literal
quadratic owner.  This section performs that subtraction in closed form.  It
also separates three statements which are easy to conflate:

1. the sign of one standardized Gaussian q211 kernel entry;
2. the sign of a signed complete-old endpoint action; and
3. the fidelity of the cheap rooted (Q=2) evaluator.

Only the first statement can be certified entrywise.  The second needs a
costate-cone or an endpoint error gate.

### 91.1 The exact (P_{\le2}) cumulant and intrinsic defect

Let (F(G)=G_+-\mu), where

\[
 \mu={1\over\sqrt{2\pi}},
\]

and let (X=P_{\le2}F).  In probabilists' Hermites,

\[
 \boxed{X(G)={1\over2}\operatorname{He}_1(G)
       +{\mu\over2}\operatorname{He}_2(G).} \tag{91.1}
\]

The singleton coefficients of (X) are (b_1=1/2,b_2=\mu), and zero
above degree two.  Its square-response coefficients, read from (78.5), are

\[
 h_1=\mu,\qquad h_2={1\over2}+{1\over\pi},\qquad
 h_3=3\mu,\qquad h_4={3\over\pi}. \tag{91.2}
\]

Substitution in the exact connected series (78.6), with no truncation left
to take, gives

\[
\boxed{\begin{aligned}
 K_{\le2}(x,y,z)
 ={}&{1\over4\pi}
 (xy+x^2y+xy^2+xz+yz)\\
 &+{x^2y^2\over4\pi^2}
 +{\pi+2\over4\pi^2}xyz .
\end{aligned}} \tag{91.3}
\]

Consequently the standardized intrinsic Gaussian surface defect is the
following elementary closed function on the PSD correlation domain:

\[
 \boxed{K_{\rm rem}(x,y,z)=K(x,y,z)-K_{\le2}(x,y,z),} \tag{91.4}
\]

where (K) is given either by the orthant formula (86.3)--(86.16) or by the
CP3-fused formula (88.10).  Thus (91.4) contains no fitted coefficient and
no implicit conditional expectation.  For non-unit zero-bias affine
variables its q211 entry is

\[
 \sigma_i^2\sigma_j\sigma_k
 K_{\rm rem}(R_{ij},R_{ik},R_{jk}). \tag{91.5}
\]

Equation (91.4) is a Gaussian-comparator identity.  It neither supplies the
true-minus-Gaussian source nor proves ownership of the CP3 term at a
different cut.

For later error gates define the positive constants

\[
\begin{gathered}
 c_2={4-\pi\over8\pi},\qquad c_3={1\over8\pi},\qquad
 c_t={4-\pi\over4\pi^2},\qquad c_{22}={1\over4\pi^2},\\
 c_{31}={1\over12\pi},\qquad
 c_{z2}={\pi-1\over8\pi^2}.
\end{gathered} \tag{91.6}
\]

The complete Taylor polynomial through degree four is

\[
\boxed{\begin{aligned}
 K_{\rm rem}={}&-c_2xy-c_3(x^2y+xy^2)-c_txyz-c_{22}x^2y^2\\
 &-c_{31}(x^3z+y^3z)-c_{z2}(x^2z^2+y^2z^2)+R_5.
\end{aligned}} \tag{91.7}
\]

The first line follows from (86.23) and (91.3).  For the second line, the
only degree-four nonroot triples ((a,b,c)) in (78.6) which survive are

\[
 (3,0,1),(0,3,1),(2,0,2),(0,2,2).
\]

Using (b_1=1/2,b_2=\mu,b_3=0,b_4=-\mu) and
(h_2=1-1/\pi,h_3=2\mu) for the full ReLU gives exactly the last two
terms in (91.7).  This is also an audit that the broad fourth-order error
is not part of the rooted (Q=2) row.

### 91.2 Determinant-margin sign and monotonicity theorem

For (0<\delta<1), let

\[
 {\cal D}_\delta=
 \{(x,y,z):x,y,z\ge0,\ \Delta\ge\delta,\
                \max(x,y,z)\le1-\delta\}, \tag{91.8}
\]

and let the line segment from the origin to the queried point be included.
For nonnegative correlations this last condition is automatic: the
correlation matrix is convex along the segment and

\[
 {d\over dt}\Delta(tx,ty,tz)
 =-2t(x^2+y^2+z^2)+6t^2xyz\le0. \tag{91.9}
\]

Define the exact elementary derivative envelope

\[
 M_5(\delta):=
 \max_{u\in{\cal D}_\delta,\ |\alpha|=5}
       |\partial^\alpha K_{\rm rem}(u)| . \tag{91.10}
\]

It is finite and contains only derivatives of the square-root/arcsine
formula (86.16); (86.7) is the determinant-margin certificate.  No unknown
statistical quantity is hidden in (M_5).  Put (s=x+y+z),

\[
 \varepsilon_5={M_5(\delta)s^5\over120}, \tag{91.11}
\]

and

\[
\begin{aligned}
 S_4={}&c_2xy+c_3(x^2y+xy^2)+c_txyz+c_{22}x^2y^2\\
 &+c_{31}(x^3z+y^3z)+c_{z2}(x^2z^2+y^2z^2).
\end{aligned} \tag{91.12}
\]

Taylor's theorem now gives the two-sided, nonasymptotic bound

\[
 \boxed{-S_4-\varepsilon_5\le K_{\rm rem}
             \le-S_4+\varepsilon_5.} \tag{91.13}
\]

In particular,

\[
 \boxed{\varepsilon_5<S_4\quad\Longrightarrow\quad K_{\rm rem}<0.}
 \tag{91.14}
\]

There is an equally explicit coordinate-monotonicity certificate.  Let
(D_x=\partial_xS_4,D_y=\partial_yS_4,D_z=\partial_zS_4).  For example,

\[
\begin{aligned}
 D_x={}&c_2y+c_3(2xy+y^2)+c_tyz+2c_{22}xy^2\\
 &+3c_{31}x^2z+2c_{z2}xz^2,\\
 D_z={}&c_txy+c_{31}(x^3+y^3)
              +2c_{z2}z(x^2+y^2),
\end{aligned} \tag{91.15}
\]

with (D_y) obtained by swapping (x,y).  Multivariate Taylor's theorem
applied to each first derivative gives

\[
 \boxed{{M_5(\delta)s^4\over24}<D_r
 \quad\Longrightarrow\quad \partial_rK_{\rm rem}<0,
 \qquad r=x,y,z.} \tag{91.16}
\]

Thus negativity and coordinatewise decrease are proved on a literal
determinant-margin cone, not inferred from a few diagrams.

No global **strict** monotonicity theorem is possible: for example

\[
 K_{\rm rem}(0,0,z)=0
\]

for every lawful (z), since the anchor component is independent of the
other pair.  More importantly, the Hermite series is not coefficientwise
negative beyond (91.7).  Its degree-five part already contains

\[
 {x^4y+xy^4\over96\pi}
 +{\pi-1\over12\pi^2}xyz^3
 -{x^3yz+xy^3z\over12\pi^2}. \tag{91.17}
\]

Hence diagram signs alone prove neither global weak monotonicity nor a
global sign outside (91.14)--(91.16).  The exact formula (91.4) is binding
there.  This is the sharp logical conclusion: a certified negative local
cone, a disproof of strict/termwise monotonicity, and no unsupported global
sign extrapolation.

One boundary derivative is useful as a direct audit.  On the independent
face (y=z=0),

\[
 \boxed{\partial_yK_{\rm rem}(x,0,0)
 ={1\over2}\{q(x)-\mu^2(1+x)^2\}<0,qquad 0<x\le1.} \tag{91.18}
\]

Indeed (g(x)=\mu^2(1+x)^2-q(x)) has (g(0)=0) and

\[
 g'(x)={1+x\over\pi}-{1\over4}-{\arcsin x\over2\pi}>0;
\]

after multiplication by (2\pi), the last inequality is
(2(1+x)-\pi/2-\arcsin x>0), whose only interior critical point is a
maximum and whose endpoint values are positive.

### 91.3 Sharp Price and rooted-(Q=2) error bounds

The centered Price row and the exact rooted (Q=2) row are respectively

\[
 A_P=-c_2xy, \tag{91.19}
\]

\[
 A_{Q2}=-c_2xy-c_3(x^2y+xy^2)-c_{22}x^2y^2. \tag{91.20}
\]

Define

\[
\begin{aligned}
 E_P={}&c_3(x^2y+xy^2)+c_txyz+c_{22}x^2y^2\\
 &+c_{31}(x^3z+y^3z)+c_{z2}(x^2z^2+y^2z^2),\\
 E_{Q2}={}&c_txyz+c_{31}(x^3z+y^3z)
                  +c_{z2}(x^2z^2+y^2z^2).
\end{aligned} \tag{91.21}
\]

Equations (91.7)--(91.11) imply the signed and absolute receipts

\[
\boxed{\begin{aligned}
 -E_P-\varepsilon_5&\le K_{\rm rem}-A_P
                         \le-E_P+\varepsilon_5,\\
 -E_{Q2}-\varepsilon_5&\le K_{\rm rem}-A_{Q2}
                         \le-E_{Q2}+\varepsilon_5,
\end{aligned}} \tag{91.22}
\]

and therefore

\[
 |K_{\rm rem}-A_P|\le E_P+\varepsilon_5,qquad
 |K_{\rm rem}-A_{Q2}|\le E_{Q2}+\varepsilon_5. \tag{91.23}
\]

Whenever (91.14) holds, the corresponding exact-relative-error bounds are

\[
\boxed{
 { |K_{\rm rem}-A_P|\over|K_{\rm rem}|}
 \le {E_P+\varepsilon_5\over S_4-\varepsilon_5},
 \qquad
 { |K_{\rm rem}-A_{Q2}|\over|K_{\rm rem}|}
 \le {E_{Q2}+\varepsilon_5\over S_4-\varepsilon_5}.} \tag{91.24}
\]

The first omitted (Q=2) term is still the exact broad triangle.  Its
ratio to the leading Price star is

\[
 {c_txyz\over c_2xy}={2z\over\pi}, \tag{91.25}
\]

while (91.21) displays the complete fourth-order broad correction.  Thus a
small-(z) receipt, not merely small (x,y), is necessary for a cheap
rooted evaluator.

### 91.4 Endpoint alignment, an exact signed-costate counterexample, and
the value gate

Let (w_{i;jk}) denote the fully collision-masked, complete-old residual
costate weight, including the scale in (91.5).  If every live entry obeys
(91.14) and all nonzero (w_{i;jk}) have one sign, then the exact defect,
Price row, and (Q=2) row have the same endpoint-action sign.  This is the
minimal alignment cone.

For arbitrary signed costates define

\[
 \mathcal E_{Q2}=
 \sum_{i,j,k}|w_{i;jk}|
       \{E_{Q2}(x_{ij},y_{ik},z_{jk})+\varepsilon_{5,i;jk}\}. \tag{91.26}
\]

Then

\[
 \boxed{|S_{\rm exact}-S_{Q2}|\le\mathcal E_{Q2},\qquad
 \mathcal E_{Q2}<|S_{Q2}|
 \Longrightarrow \operatorname{sign}S_{\rm exact}
                 =\operatorname{sign}S_{Q2}.} \tag{91.27}
\]

There is no weaker universal action-sign claim.  To see this, take two
independent (3\times3) Gaussian blocks with the same
(x=y=\epsilon>0) and (0\le z_1<z_2), with (\epsilon,z_2)
sufficiently small that both satisfy (91.14).  Since (A_{Q2})
does not depend on (z), write

\[
 r(z)={K_{\rm rem}(x,y,z)\over A_{Q2}(x,y)}.
\]

From (91.7),

\[
 r'(0)={c_t\epsilon^2+2c_{31}\epsilon^3
                 \over|A_{Q2}(\epsilon,\epsilon)|}
          +O(\epsilon^2)>0. \tag{91.28}
\]

Hence (r_2:=r(z_2)>r_1:=r(z_1)>0).  Give the two endpoint entries weights
(1,-c), where

\[
 {r_1\over r_2}<c<1.
\]

Then

\[
 S_{Q2}=A_{Q2}(1-c)<0,qquad
 S_{\rm exact}=A_{Q2}(r_1-cr_2)>0. \tag{91.29}
\]

This is a lawful PSD, arbitrarily small-correlation sign-flip
counterexample.  Complete-old Schur residualization generally creates
exactly such signed weights, so entrywise negativity is not an endpoint
certificate.

Finally, let \(\widehat f\) be the endpoint correction produced by the
cheap row, (f=\widehat f+d) the exact correction, and
(\|d\|_2\le\epsilon_f), with (91.26) included in the construction of
(\epsilon_f).  If (e) is the current residual, the exact raw-MSE gain
obeys

\[
\boxed{
 G(f)\ge G(\widehat f)
 -2\|e-\widehat f\|_2\epsilon_f-\epsilon_f^2,
 \qquad
 G(g):=2\langle e,g\rangle-\|g\|_2^2.} \tag{91.30}
\]

Thus a robust positive-value gate is

\[
 G(\widehat f)>
 2\|e-\widehat f\|_2\epsilon_f+\epsilon_f^2, \tag{91.31}
\]

followed separately by the billed adjusted-score hurdle.  A pointwise
kernel sign, even when exact, is not a substitute for (91.27) or (91.31).

Graph-ready split:

- `GAUSSIAN_Q211_P2_SURFACE` -- **solved exact**, (91.1)--(91.3);
- `GAUSSIAN_Q211_INTRINSIC_DEFECT` -- **solved exact**, (91.4)--(91.5);
- `GAUSSIAN_Q211_DEFECT_FOURTH_JET` -- **solved exact**, (91.6)--(91.7);
- `GAUSSIAN_Q211_DEFECT_LOCAL_SIGN` -- **certified conditional**, gate
  (91.14);
- `GAUSSIAN_Q211_DEFECT_LOCAL_MONOTONICITY` -- **certified conditional**,
  gate (91.16); global strict/termwise claims are closed no-go;
- `GAUSSIAN_Q211_PRICE_ERROR` -- **bounded**, (91.21)--(91.24);
- `GAUSSIAN_Q211_ROOT_Q2_ERROR` -- **bounded**, (91.21)--(91.25);
- `GAUSSIAN_Q211_SIGNED_ACTION_ALIGNMENT` -- **conditional only** on
  (91.27); unconditional claim closed by (91.29);
- `GAUSSIAN_Q211_ENDPOINT_VALUE` -- **conditional GO** only through
  (91.30)--(91.31) and the separate billing hurdle;
- `GAUSSIAN_Q211_TRUE_MINUS_GAUSSIAN_VALUE` -- **not supplied** by this
  section.

## 97. First dependent-chaos response beyond quadratic AQ: signed cubic
source, gate-surface adjoint, and a rigorous second-order remainder

Section 93 closes q211 *inside* the quadratic-AQ family.  The first source
outside that family is a perturbation of the preactivation function, not a
latent-mean tilt.  Write

\[
 g_a^\epsilon=g_a+\epsilon H_a,qquad
 Y_a^\epsilon=(g_a^\epsilon)_+, \tag{97.1}
\]

where (g_a=g_a^{AQ}(z)), (z\sim N(0,I_r)), and (H_a) is centered and
orthogonal to latent chaoses zero, one, and two.  If (H) contains extra
randomness, replace it below by its conditional mean

\[
 \bar H_a(z):=E[H_a\mid z]. \tag{97.2}
\]

Only (97.2) contributes at first order.  Independent centered rest has
(\bar H=0), so its leading response is second order and belongs to the
heat/covariance lane of Section 81, not this section.

### 97.1 Exact Fréchet derivative in all four q211 slots

Put (I_a=\mathbf1_{g_a>0}), (Y_a=(g_a)_+), (m_a=EY_a),
(Z_a=Y_a-m_a), (C_{ab}=E[Z_aZ_b]), and
(T_{abc}=E[Z_aZ_bZ_c]).  Define the third cumulant-Wick influence

\[
\boxed{\begin{aligned}
 {\cal W}_3(a,b,c):={}&Z_aZ_bZ_c-C_{ab}Z_c-C_{ac}Z_b-C_{bc}Z_a
 -T_{abc}.
\end{aligned}} \tag{97.3}
\]

For an arbitrary fourth argument (A), direct expansion of the cumulant
partition gives

\[
 \boxed{\kappa_4(A,Y_a,Y_b,Y_c)
       =E[A\,{\cal W}_3(a,b,c)].} \tag{97.4}
\]

The subtraction of (T_{abc}) makes (97.3) mean zero and accounts exactly
for the derivative of every singleton--triple partition; the covariance
terms account for the three pair--pair partitions.  Therefore no separate
mean/pair response may be dropped merely because (H\perp P_{\le2}(z)).

Since

\[
 {d\over d\epsilon}(g_a+\epsilon H_a)_+\big|_{0}=H_aI_a
\]

in (L^1) under the density condition in Section 97.5, the strict q211
surface has the exact first variation

\[
\boxed{\begin{aligned}
 \dot K_{i;jk}:={d\over d\epsilon}
 \kappa_4(Y_i^\epsilon,Y_i^\epsilon,Y_j^\epsilon,Y_k^\epsilon)
 \big|_0
 ={}&2E[\bar H_iI_i{\cal W}_3(i,j,k)]\\
 &+E[\bar H_jI_j{\cal W}_3(i,i,k)]\\
 &+E[\bar H_kI_k{\cal W}_3(i,i,j)].
\end{aligned}} \tag{97.5}
\]

These are the four slot placements: the two equal anchor placements fuse
to the first term with multiplicity two; the two singleton placements
remain separately colored.  Equation (97.5) is valid for the full
dependent residual, not only for a cubic truncation.

### 97.2 Signed (H_3) tensor and exact AQ gate/delta contraction

Let (z^{\diamond q}) be the symmetric Wick power and define the signed
latent-chaos tensors

\[
 C_a^{(q)}:=E[\bar H_a(z)z^{\diamond q}],qquad q\ge3. \tag{97.6}
\]

Because of the assumed orthogonality, (C_a^{(3)}) is also the raw signed
cubic cumulant (E[H_az_pz_qz_r]).  Gaussian integration by parts gives,
for every sufficiently regular AQ influence (\Psi),

\[
 E[\bar H_a\Psi]
 =\sum_{q\ge3}{1\over q!}
   \left\langle C_a^{(q)},E[\nabla_z^q\Psi]\right\rangle . \tag{97.7}
\]

In particular, if (H_a) is a pure third-chaos perturbation,

\[
 \bar H_a={1\over3!}\langle C_a^{(3)},z^{\diamond3}\rangle,
\]

then (97.5) becomes the exact finite surface contraction

\[
\boxed{\begin{aligned}
 \dot K_{i;jk}^{(3)}={1\over3!}\bigg\{&
 2\langle C_i^{(3)},{\cal T}_{i|ijk}\rangle
 +\langle C_j^{(3)},{\cal T}_{j|iik}\rangle\\
 &+\langle C_k^{(3)},{\cal T}_{k|iij}\rangle\bigg\},\\
 {\cal T}_{a|bcd}:={}&
 E[\nabla_z^3\{I_a{\cal W}_3(b,c,d)\}].
\end{aligned}} \tag{97.8}
\]

This is the requested signed-H3 acquisition map.  Every tensor in
({\cal T}) is an AQ gate/boundary surface, already at the unperturbed
cut.  To make that statement literal, for the quadratic AQ function

\[
 v_a:=\nabla g_a=u_a+2A_az,qquad B_a:=\nabla^2g_a=2A_a,
\]

the distributional derivatives are

\[
\begin{aligned}
 \nabla Y_a&=I_av_a,\\
 \nabla^2Y_a&=I_aB_a+\delta(g_a)v_a^{\otimes2},\\
 \nabla^3Y_a&=3\delta(g_a)\operatorname{Sym}(B_a\otimes v_a)
              +\delta'(g_a)v_a^{\otimes3}, \tag{97.9}
\end{aligned}
\]

and

\[
\begin{aligned}
 \nabla I_a&=\delta(g_a)v_a,\\
 \nabla^2I_a&=\delta'(g_a)v_a^{\otimes2}+\delta(g_a)B_a,\\
 \nabla^3I_a&=\delta''(g_a)v_a^{\otimes3}
 +3\delta'(g_a)\operatorname{Sym}(B_a\otimes v_a). \tag{97.10}
\end{aligned}
\]

Finally,

\[
\boxed{\begin{aligned}
 \nabla^3(I_aW)={}&I_a\nabla^3W
+3\operatorname{Sym}(\nabla I_a\otimes\nabla^2W)\\
&+3\operatorname{Sym}(\nabla^2I_a\otimes\nabla W)
+(\nabla^3I_a)W,
\end{aligned}} \tag{97.11}
\]

with (W={\cal W}_3), expands (97.8) into a finite list of gate, delta,
delta-prime, and delta-double-prime surface intersections.  Equations
(97.9)--(97.11), rather than a new dense fourth tensor, are the exact
forward objects.

If (H) is not pure cubic, (97.8) is only its (P_3H) contribution.  The
unseen first-order tail is exactly

\[
\boxed{\begin{aligned}
 |\dot K-\dot K^{(3)}|\le{}&
 2\|P_{>3}\bar H_i\|_2
   \|P_{>3}(I_i{\cal W}_3(i,j,k))\|_2\\
 &+\|P_{>3}\bar H_j\|_2
   \|P_{>3}(I_j{\cal W}_3(i,i,k))\|_2\\
 &+\|P_{>3}\bar H_k\|_2
   \|P_{>3}(I_k{\cal W}_3(i,i,j))\|_2.
\end{aligned}} \tag{97.12}
\]

Thus "orthogonal through degree two plus a known cubic cumulant" is not by
itself an exact-source specification; it needs either (P_{>3}H=0) or the
tail gate (97.12).

### 97.3 Exact CP endpoint factorization of the reverse influence

Consider one pre-collision partially symmetric costate atom

\[
 L_{i;jk}=\alpha_i{\beta_j\gamma_k+\gamma_j\beta_k\over2}. \tag{97.13}
\]

Put (B_\beta=\beta^TZ), (c_{\beta\gamma}=\beta^TC\gamma), and

\[
 t_{\beta\gamma,i}=\sum_{j,k}T_{ijk}\beta_j\gamma_k,qquad
 \theta_{\xi,i}=\sum_kT_{iik}\xi_k. \tag{97.14}
\]

The two finite AQ reverse polynomials needed by the perturbation are

\[
\boxed{\begin{aligned}
 {\cal A}_{\beta\gamma,i}:={}&
 Z_iB_\beta B_\gamma-(C\beta)_iB_\gamma-(C\gamma)_iB_\beta
 -c_{\beta\gamma}Z_i-t_{\beta\gamma,i},\\
 {\cal B}_{\alpha\xi}:={}&
 \langle\alpha,Z^{\odot2}-\operatorname{diag}C\rangle B_\xi
 -2\langle\alpha\odot Z,C\xi\rangle
 -\langle\alpha,\theta_\xi\rangle.
\end{aligned}} \tag{97.15}
\]

They satisfy the exact contraction identities

\[
 {\cal A}_{\beta\gamma,i}
 =\sum_{j,k}\beta_j\gamma_k{\cal W}_3(i,j,k),
\qquad
 {\cal B}_{\alpha\xi}
 =\sum_{i,k}\alpha_i\xi_k{\cal W}_3(i,i,k). \tag{97.16}
\]

After fusing both anchor slots and both symmetric leaf slots, the complete
node-colored reverse influence is therefore

\[
\boxed{
 \Xi_{L,a}=I_a\{2\alpha_a{\cal A}_{\beta\gamma,a}
 +\beta_a{\cal B}_{\alpha\gamma}
 +\gamma_a{\cal B}_{\alpha\beta}\}.} \tag{97.17}
\]

Strict-collision exclusions apply the same one-/two-index corrections to
(97.17); they do not change its object count.  For a pure cubic source the
whole target response is

\[
\boxed{
 \dot S_L={1\over3!}\sum_a
 \left\langle C_a^{(3)},E[\nabla_z^3\Xi_{L,a}]\right\rangle .} \tag{97.18}
\]

Thus the CP endpoint never needs a materialized q211 tensor.  It needs the
resident AQ (C,T) contractions in (97.14)--(97.17), followed by the
signed cubic/source contraction in (97.18).  This is suitable for reverse
node fusion, but it is not free CP3-jet ownership.

### 97.4 Minimal signed statistics and the one-scalar criterion

For one strict q211 entry, the exact minimal first-order observables are

\[
\begin{aligned}
 s_{i|ijk}&=E[\bar H_iI_i{\cal W}_3(i,j,k)],\\
 s_{j|iik}&=E[\bar H_jI_j{\cal W}_3(i,i,k)],\\
 s_{k|iij}&=E[\bar H_kI_k{\cal W}_3(i,i,j)].
\end{aligned} \tag{97.19}
\]

There are three, not four, because of the repeated anchor.  They are
generically independent: perturbing only (H_i,H_j), or (H_k) changes
only the corresponding row in (97.19).  Orthogonality makes their literal
representatives

\[
 s_{a|bcd}
 =\langle\bar H_a,P_{\ge3}(I_a{\cal W}_3(b,c,d))\rangle_{L^2}. \tag{97.20}
\]

Knowing only (C^{(3)}) is sufficient exactly when (\bar H=P_3\bar H)
or when every higher-chaos influence in (97.12) is in the paid kernel.  A
sharp counterexample is (\bar H_a=\operatorname{He}_4(v^Tz)): it is
centered and orthogonal through degree three, so (C_a^{(3)}=0), while a
generic ReLU gate influence in (97.20) has a nonzero fourth Hermite
coefficient and hence a nonzero first response.  Two perturbations with the
same signed H3 tensor can therefore have different q211 responses.

For one fixed endpoint (L), (97.17) fuses (97.19) nodewise and only the
single final scalar

\[
 \dot S_L=\sum_aE[\bar H_a\Xi_{L,a}] \tag{97.21}
\]

need leave the seam.  This is a lawful one-target scalar fusion if and only
if the literal evaluator has simultaneous access to (\bar H_a) and
(Xi_{L,a}) at that cut.  It is not a reusable one-state provider for a
portfolio.  For targets (L^{(1)},\ldots,L^{(R)}), the minimum cubic-source
rank is

\[
 \boxed{\operatorname{rank}
 \{,E[\nabla^3\Xi_{L^{(r)},a}],: r=1,\ldots,R,\ a=1,\ldots,n\}
 \quad\text{modulo the complete-old row span}.} \tag{97.22}
\]

A common scalar owns the portfolio exactly when this quotient rank is at
most one.  Shared AQ moments, a common (C^{(3)}) family, or one CP3 tensor
does not imply (97.22).

### 97.5 Rigorous (O(\epsilon^2)) density-margin remainder

Index the four ordered slots by (s=1,2,3,4), with labels
((a_s)=(i,i,j,k)), and set

\[
\begin{aligned}
 Y_s(t)&=(g_{a_s}+tH_{a_s})_+,\\
 V_s(t)&=H_{a_s}\mathbf1_{g_{a_s}+tH_{a_s}>0},\\
 W_s(t)&=H_{a_s}^2\delta(g_{a_s}+tH_{a_s}).
\end{aligned} \tag{97.23}
\]

Distributional differentiation and cumulant multilinearity give the exact
second derivative

\[
\boxed{\begin{aligned}
 K''(t)={}&\sum_s\kappa_4(W_s(t),Y_{-s}(t))\\
 &+2\sum_{s<u}\kappa_4(V_s(t),V_u(t),Y_{-(s,u)}(t)).
\end{aligned}} \tag{97.24}
\]

Here (Y_{-s}) denotes the other three arguments, with their slot order
retained.  Define (M_v(t)=\max(1,\|Y_v(t)\|_4)),
(h_s=\|H_{a_s}\|_4), and the single-surface density norm

\[
\boxed{\begin{aligned}
 D_s(\epsilon):=\sup_{|t|\le|\epsilon|}
 \max_{A\subseteq\{1,2,3,4\}\setminus s}
 {\left|E\!left[H_{a_s}^2\delta(g_{a_s}+tH_{a_s})
          \prod_{v\in A}Y_v(t)\right]\right|
  \over\prod_{v\in A}M_v(t)}.
\end{aligned}} \tag{97.25}
\]

The coarea formula makes (97.25) literal.  If (H) is a function of (z),
its numerator is

\[
 \int_{\{g_{a_s}+tH_{a_s}=0\}}
 {H_{a_s}^2\prod_{v\in A}Y_v(t)\,\phi_r(z)
  \over\|\nabla(g_{a_s}+tH_{a_s})\|}\,dS. \tag{97.26}
\]

Thus finiteness follows from a positive gradient/density margin and the
displayed weighted surface moments.  For the cubic formula (97.8), impose
the corresponding joint-surface margin

\[
 \det[(\nabla g_a)^T\nabla g_b]_{a,b\in A}\ge d_*^{,2|A|}
 \quad\text{on }\{g_a=0:a\in A\},\quad |A|\le3, \tag{97.27}
\]

together with two bounded normal derivatives of the joint density.  This
is precisely the determinant/density condition needed by the
(delta',\delta'') terms in (97.10), not an informal nonsingularity claim.

The partition formula gives the universal bound

\[
 |\kappa_4(X_1,X_2,X_3,X_4)|
 \le26\prod_s\|X_s\|_4, \tag{97.28}
\]

and the same partition count, using (97.25), bounds a surface argument.
Consequently

\[
\boxed{\begin{aligned}
 B_2(\epsilon):=26\sup_{|t|\le|\epsilon|}\bigg[&
 \sum_sD_s(\epsilon)\prod_{v\ne s}M_v(t)\\
 &+2\sum_{s<u}h_sh_u\prod_{v\notin\{s,u\}}M_v(t)
 \bigg]
\end{aligned}} \tag{97.29}
\]

satisfies (sup_{|t|\le|\epsilon|}|K''(t)|\le B_2(\epsilon)).  Taylor's
theorem therefore yields the rigorous, nonasymptotic receipt

\[
\boxed{
 |K(\epsilon)-K(0)-\epsilon\dot K(0)|
 \le{\epsilon^2\over2}B_2(\epsilon).} \tag{97.30}
\]

For an endpoint costate, sum (97.30) with absolute collision-masked
weights.  The resulting bound enters the same robust value gate (91.30),
not a pointwise sign heuristic.

### 97.6 Ownership split

The Section 93 AQ jet differentiates the *latent Gaussian measure* while
holding every quadratic (g_a) fixed.  Equation (97.5) differentiates the
*preactivation function* in a chaos-(ge3) direction.  Their tangent
spaces are distinct.  Literal ownership of the new source for target (L)
requires

\[
 \ker E_{\rm old}\subseteq
 \ker\left(H\mapsto\sum_aE[\bar H_a\Xi_{L,a}]\right). \tag{97.31}
\]

An AQ CP3 value, its first/second Cameron--Martin jets, and all pair rows
contain no information about an independently variable (C^{(3)}), so
(97.31) fails generically.  Conversely, a paid full true-surface oracle
already evaluates (97.5) and receives zero additive credit from this
decomposition.

Graph-ready split:

- `TRUE_MINUS_AQ_Q211_FIRST_VARIATION` -- **solved exact**, (97.3)--(97.5);
- `TRUE_MINUS_AQ_SIGNED_H3_MAP` -- **solved exact for pure cubic H**,
  (97.6)--(97.11);
- `TRUE_MINUS_AQ_HIGHER_CHAOS_TAIL` -- **not owned by H3**, bounded by
  (97.12);
- `TRUE_MINUS_AQ_CP_REVERSE_INFLUENCE` -- **solved exact**,
  (97.13)--(97.18);
- `TRUE_MINUS_AQ_MINIMAL_SIGNED_STATISTICS` -- **three per q211 entry,
  one target scalar only after literal node fusion**, (97.19)--(97.22);
- `TRUE_MINUS_AQ_SECOND_ORDER_REMAINDER` -- **rigorously bounded** under
  (97.25)--(97.27), by (97.29)--(97.30);
- `TRUE_MINUS_AQ_VS_CP3_JET` -- **independent tangent/source**, ownership
  test (97.31);
- `TRUE_MINUS_AQ_WITH_FULL_TRUE_SURFACE` -- **closed duplicate**.

## 92. Adjoint audit of the arbitrary-bias CP3 threshold jet: one colored
derivative tensor and the edge-reanchoring obstruction

Section 90 identifies the exact forward objects.  This section answers the
deployment question: what reverse queries are actually required under a
partially symmetric q211 Bellman costate, when are they literal CP3/Q24
views, and what is the cheapest exact factor fallback?

Work first in standardized units; positive output scales are diagonal
factors that can be absorbed into the costate columns.  Let

\[
 T_{ijk}=K_3(\tau_i,\tau_j,\tau_k;
             R_{ij},R_{ik},R_{jk})
 \tag{92.1}
\]

be the symmetric CP3 tensor and define the anchor-colored threshold jet

\[
 A_{i;jk}:=\partial_{\tau_i}T_{ijk},
 \qquad A_{i;jk}=A_{i;kj}.
 \tag{92.2}
\]

Put

\[
 w_i=\tau_i+2m_i.
 \tag{92.3}
\]

The pair-only complement \(F_{\le2}\) in (90.8) is kept separate throughout.

### 92.1 Exact adjoint compression of the three threshold slots

Let \(L_{i;jk}=L_{i;kj}\) be collision-compiled and use the q211 inner
product of Section 83.  Define the edge-reanchoring operator

\[
 \boxed{
 ({\cal C}_RL)_{j;ik}
 ={1\over2}\{
 R_{ij}L_{i;jk}+R_{kj}L_{k;ji}\}.}
 \tag{92.4}
\]

The second term is exactly the singleton symmetrization \(i\leftrightarrow
k\).  Relabeling the two leaf derivatives in (90.8) gives

\[
\begin{aligned}
 &\left\langle L,\,
 \partial_{\tau_i}T
 +R_{ij}\partial_{\tau_j}T
 +R_{ik}\partial_{\tau_k}T
 \right\rangle_{\cal Q}\\
 &\hspace{25mm}
 =\boxed{\langle L+2{\cal C}_RL,A\rangle_{\cal Q}.}
\end{aligned}
 \tag{92.5}
\]

Therefore the complete three-unit CP3-jet action is

\[
 \boxed{
 {\cal J}_L
 =\langle D_w^{\rm anchor}L,T\rangle_{\cal Q}
  +\langle Q_L,A\rangle_{\cal Q},
 \qquad
 Q_L:=L+2{\cal C}_RL.}
 \tag{92.6}
\]

The Gaussian q211 contribution is \(-{\cal J}_L\), plus the separately
contracted \(F_{\le2}\).  Equation (92.6) is the minimum exact object count:

1. one ordinary CP3 value tensor \(T\);
2. one anchor-colored threshold tensor \(A\); and
3. one edge-reanchored costate \(Q_L\).

There are not three independent derivative tensors.  Symmetry fuses the two
leaf derivatives into the same anchor-colored tensor, but it does not remove
the covariance-dependent reanchoring operator.

The identical formula applies to the raw identity (90.4) after replacing
\(T=K_3,w=\tau+2m\) by \(T=M_{111},w=\tau\).  Connected-cumulant ownership
still requires the means, pairs, and their derivative rows in \(F_{\le2}\).

### 92.2 A CP3 value is not its colored threshold jet

Let \(b\) parameterize the synchronized local surface germ, let
\(Vb=T\), \(Jb=A\), and let \(Eb\) contain every actually paid
CP3/Q24/threshold row.  For the query \(Q_L\), its threshold action is a
literal resident reverse view if and only if

\[
 \boxed{
 \ker E\subseteq\ker(Q_L^*J)
 \quad\Longleftrightarrow\quad
 Q_L^*J=B_LE\quad\hbox{for some }B_L.}
 \tag{92.7}
\]

Universal ownership of the whole colored jet is the stronger condition

\[
 \boxed{\ker E\subseteq\ker J.} \tag{92.8}
\]

Taking \(E=V\) does not satisfy (92.8) generically.  At any threshold point
\(\tau^0\), the analytic perturbation

\[
 \delta T_{ijk}(\tau)
 =\sum_{r\in\{i,j,k\}}
   (\tau_r-\tau_r^0)H_{r;\{i,j,k\}\setminus r}
 \tag{92.9}
\]

has \(\delta T(\tau^0)=0\) but an arbitrary symmetric colored first jet
\(H\).  Thus equality of CP3 values at one cut cannot identify the jet.
For the fixed Gaussian family \(R,\tau\) contain enough source information
to recompute it, but that is a new nonlinear row unless (92.7) holds.

Likewise, Q24 ownership is literal only when its exposed derivative channel
is the same strict three-label boundary tensor \(A\), at the same cut and
with the same collision mask.  Pairwise coefficient-table derivatives own
the \(F_{\le2}\) rows; they do not by their name alone own \(A\).

There is one exact ordinary-JVP fusion.  If

\[
 Q_{i;jk}=Q_{j;ik}=Q_{k;ij}
 \quad\hbox{on every unordered triple}, \tag{92.10}
\]

then the three colored contractions are equal, and

\[
 \boxed{
 \langle Q,A\rangle_{\cal Q}
 ={1\over3}{d\over dt}
 \left.\langle Q,T(\tau+t\mathbf1)\rangle_{\cal Q}
 \right|_{t=0}.}
 \tag{92.11}
\]

Hence a literal common-threshold CP3 JVP owns precisely the fully symmetric
orientation component.  If it is the only derivative row, (92.10) is also
necessary: the two-dimensional orientation-contrast component lies in its
kernel.  The reanchored query \(Q_L=L+2{\cal C}_RL\) is generically not fully
symmetric.

### 92.3 Exact CP factor action and matvec census

Suppose the colored threshold tensor has a partially symmetric CP
factorization

\[
 A_{i;jk}
 =\sum_{s=1}^{r_\tau}
 \eta_{s,i}\,
 { \xi_{s,j}\zeta_{s,k}
  +\zeta_{s,j}\xi_{s,k}\over2},
 \tag{92.12}
\]

and one costate atom is

\[
 L_{i;jk}
 =\alpha_i\,
 { \beta_j\gamma_k+\gamma_j\beta_k\over2}.
 \tag{92.13}
\]

Before strict-collision subtraction, the direct anchor action of one pair
of atoms is

\[
\boxed{
\begin{aligned}
 \langle L,A_s\rangle_{\cal Q}
 ={1\over4}(\alpha^T\eta_s)
 \{(\beta^T\xi_s)(\gamma^T\zeta_s)
  +(\beta^T\zeta_s)(\gamma^T\xi_s)\}.
\end{aligned}}
 \tag{92.14}
\]

The edge-reanchored action is

\[
\boxed{
\begin{aligned}
 \langle{\cal C}_RL,A_s\rangle_{\cal Q}
 ={1\over8}\big[&
 (\alpha\odot\xi_s)^TR(\beta\odot\eta_s)
       (\gamma^T\zeta_s)\\
 &+(\alpha\odot\zeta_s)^TR(\beta\odot\eta_s)
       (\gamma^T\xi_s)\\
 &+(\alpha\odot\xi_s)^TR(\gamma\odot\eta_s)
       (\beta^T\zeta_s)\\
 &+(\alpha\odot\zeta_s)^TR(\gamma\odot\eta_s)
       (\beta^T\xi_s)\big].
\end{aligned}}
 \tag{92.15}
\]

Thus each costate/jet atom pair needs exactly the two covariance products

\[
 R(\beta\odot\eta_s),\qquad
 R(\gamma\odot\eta_s),
 \tag{92.16}
\]

or one when \(\beta=\gamma\).  All strict \(i=j,i=k,j=k\) exclusions are
one- and two-index corrections and add no dense covariance product.

For \(R_c\) costate atoms, \(r_\tau\) jet atoms, and \(D\) active seams, the
unshared dense-product bill is

\[
 \boxed{
 C_{\tau}^{\rm generic}
 ={2DR_cr_\tau\over n},\qquad
 C_{\tau}^{\rm sym}
 ={DR_cr_\tau\over n}.}
 \tag{92.17}
\]

At \(n=1024,D=10\), a one-equivalent screen requires

\[
 R_cr_\tau\le51\quad\hbox{generically},\qquad
 R_cr_\tau\le102\quad\hbox{for }\beta=\gamma,
 \tag{92.18}
\]

before factor construction, pair-complement reads, hooks, and writes.
Shared dictionaries reduce (92.17) only by the number of literally
identical vectors in (92.16).

If the ordinary CP3 tensor is resident as

\[
 T_{ijk}=\sum_{t=1}^{r_3}\lambda_tu_{t,i}u_{t,j}u_{t,k},
 \tag{92.19}
\]

the value term in (92.6) is a bounded list of factor dots; for one ray it is

\[
 { \lambda_t\over2}
 \{(w\odot\alpha)^Tu_t\}
 (\beta^Tu_t)(\gamma^Tu_t),
 \tag{92.20}
\]

again followed by collision corrections.  Its cost is \(O(nR_cr_3)\) and
uses no covariance matvec.  The minimum extra factor state for (92.12) is
\(O(nr_\tau)\) columns plus weights; a generic unfactored \(A\) is
strict-q211-sized.

### 92.4 Exact tangent-factor fusion and its limitation

There is a positive CP3 fusion theorem.  Suppose the literal CP3 factors at
the reset are coordinate-local functions of threshold:

\[
 u_{t,i}=u_t(\tau_i),\qquad
 {\partial u_{t,j}\over\partial\tau_i}=0
 \quad(i\ne j),
 \tag{92.21}
\]

and their weights have no unrecorded global threshold dependence.  Then

\[
 \boxed{
 A_{i;jk}
 =\sum_{t=1}^{r_3}
 \lambda_t\dot u_{t,i}u_{t,j}u_{t,k},}
 \tag{92.22}
\]

so \(r_\tau\le r_3\), the singleton dictionary is reused exactly, and only
the tangent columns \(\dot u_t\) are additional \(O(nr_3)\) state.

Equation (92.22) is a literal fusion only if the deployed CP3 representation
itself is produced by the coordinate-local map.  A globally fitted or
truncated CP factorization has basis-motion and weight-derivative terms;
differentiating the represented tensor is not the same as differentiating
its frozen factors.  Those missing terms must either be retained as
additional tangent atoms or charged as approximation error.

Even under (92.22), the reanchored costate is generically broad.  For one
atom (92.13), \({\cal C}_RL\) contains the three-way term

\[
 \beta_j(R_{ji}\alpha_i)\gamma_k. \tag{92.23}
\]

Choose \(\gamma=e_{k_0}\) and restrict to
\(i,j\ne k_0\).  The \(j|i\) flattening of the \(k_0\) slice contains

\[
 D_\beta (R^\circ)_{-k_0,-k_0}D_\alpha,
 \qquad R^\circ=R-I, \tag{92.24}
\]

whose rank is \(n-1\) for a generic dense correlation and nonzero
\(\alpha,\beta\).  Therefore a constant-rank forward CP representation of
\({\cal C}_RL\) is generically impossible; one costate atom can require at
least \(n-1\) reanchored CP rays.  Formula (92.15) is the lawful escape:
consume the broad query by matvecs without materializing it.

The edge JVP becomes zero-incremental only when the exact products in
(92.16) are already resident in CP3/Q24 with identical
\((\beta,\gamma,\eta_s,R)\), cut, dtype, and mask, or when all involved
columns lie in a certified product-closed block/Walsh algebra.  Sharing a
CP3 rank, a covariance matrix, or a quadrature table is not enough.

### 92.5 Complete disposition

The exact deployment order is:

1. residualize the seam costate against every old CP3/Q24/pair/full-surface
   row;
2. form \(Q_L=L+2{\cal C}_RL\);
3. apply the literal ownership test (92.7);
4. if it fails, use the factor gate (92.17) or close the route; and
5. add the independently residualized \(F_{\le2}\) action.

If the full Section 49 surface is the comparator, both the CP3 jet and
\(F_{\le2}\) are already owned and receive zero additive credit.  Against a
CP3-value-only comparator, (92.6) proves that the colored jet can still be a
new typed readout.  None of these identities identifies the
true-minus-Gaussian dependent residual.

Graph-ready split:

- CP3_THRESHOLD_JET_ADJOINT_FUSION -- **solved exact**, (92.4)--(92.6);
- CP3_THRESHOLD_JET_LITERAL_OWNER -- **open literal receipt**, (92.7)--(92.8);
- CP3_COMMON_SHIFT_JVP -- **exact symmetric-mode fusion**, (92.10)--(92.11);
- CP3_THRESHOLD_JET_CP_ACTION -- **solved conditional**, (92.12)--(92.18);
- CP3_THRESHOLD_LOCAL_FACTOR_FUSION -- **conditional GO**, (92.21)--(92.22);
- CP3_THRESHOLD_REANCHOR_CP_RANK -- **generic NO-GO for forward CP**,
  (92.23)--(92.24);
- CP3_THRESHOLD_EDGE_PRODUCT_FUSION -- **open literal product receipt**;
- CP3_THRESHOLD_WITH_FULL_SURFACE -- **closed duplicate**;
- CP3_THRESHOLD_TRUE_MINUS_GAUSSIAN -- **not supplied**.

## 95. Common-node CP3--q211 fusion: exact reduction theorem and why the
current one-dimensional resolvent nodes do not qualify

Section 93 expresses AQ q211 as a tilted CP3 jet.  For a factorized endpoint
there is an even simpler implementation if the CP3 surface is evaluated by
a literal common joint cubature.  The distinction between common joint nodes
and per-unit Fourier nodes is binding.

### 95.1 Exact centered CP action

Let \(X=Y-EY\) be the centered reset-output bank, with covariance
\(\Sigma\), and let one ordered strict-q211 costate atom before collision
exclusion be

\[
 L_{i;jk}=\alpha_i\beta_j\gamma_k. \tag{95.1}
\]

Define the scalar aggregates

\[
 A_2=\sum_i\alpha_iX_i^2,\qquad
 B=\sum_j\beta_jX_j,\qquad
 C=\sum_k\gamma_kX_k. \tag{95.2}
\]

Then the complete q211 cumulant action is

\[
\boxed{\begin{aligned}
 S_{211}(\alpha,\beta,\gamma)
 ={}&E[A_2BC]
 -(\alpha^T\operatorname{diag}\Sigma)(\beta^T\Sigma\gamma)\\
 &-2\alpha^T\{(\Sigma\beta)\odot(\Sigma\gamma)\}.
\end{aligned}} \tag{95.3}
\]

This follows by summing

\[
 \kappa_4(X_i,X_i,X_j,X_k)
 =E[X_i^2X_jX_k]
 -\Sigma_{ii}\Sigma_{jk}
 -2\Sigma_{ij}\Sigma_{ik}.
\]

Strict \(i,j,k\) exclusions are exact K4/K31/K22 corrections to (95.3);
they do not alter the common-node theorem below.

### 95.2 Common joint cubature theorem

Suppose the declared CP3 evaluator uses a common joint rule

\[
 E[F(X)]=\sum_{\nu=1}^{q}w_\nu F(X^{(\nu)})+\varepsilon_F \tag{95.4}
\]

with the same vector \(X^{(\nu)}\in\mathbb R^n\) available to every unit at
node \(\nu\).  A CP3 atom uses

\[
 A_1^{(\nu)}=\alpha^TX^{(\nu)},\qquad
 B^{(\nu)}=\beta^TX^{(\nu)},\qquad
 C^{(\nu)}=\gamma^TX^{(\nu)} \tag{95.5}
\]

and accumulates

\[
 \sum_\nu w_\nu A_1^{(\nu)}B^{(\nu)}C^{(\nu)}.
\]

At the same nodes q211 needs only the replacement

\[
 \boxed{
 A_2^{(\nu)}
 =\alpha^T(X^{(\nu)}\odot X^{(\nu)}),\qquad
 \sum_\nu w_\nu A_2^{(\nu)}B^{(\nu)}C^{(\nu)}.}
 \tag{95.6}
\]

Therefore, conditional on literal node and endpoint-factor reuse, the extra
arithmetic for \(R_c\) atoms and \(o\) live anchor orientations over \(D\)
seams is

\[
 \boxed{
 C_{\rm node,extra}={Dq\,oR_c\over n^2}}
 \tag{95.7}
\]

width-GEMM equivalents, plus \(O(DR_cn^2)\) pair/collision reductions
already visible in (95.3).  At \(D=10,q=34,n=1024\), the reduction
arithmetic alone permits

\[
 R_c\le \left\lfloor{1048576\over340\,o}\right\rfloor
 ={3084\over o}\quad\hbox{up to the displayed floor}. \tag{95.8}
\]

The theorem is streaming: no q211 tensor, CP4 carrier, or latent Hessian is
materialized.

### 95.3 Fidelity is not inherited from CP3

Node reuse proves arithmetic reuse, not accuracy reuse.  The CP3 integrand
contains \(A_1BC\), whereas q211 contains the heavier-tail \(A_2BC\).
The exact error is

\[
 \varepsilon_{211}
 :=E[A_2BC]-\sum_\nu w_\nu
 A_2^{(\nu)}B^{(\nu)}C^{(\nu)}. \tag{95.9}
\]

No bound on the CP3 error

\[
 \varepsilon_3
 :=E[A_1BC]-\sum_\nu w_\nu
 A_1^{(\nu)}B^{(\nu)}C^{(\nu)}
\]

implies a bound on (95.9): choose a signed residual rule annihilating
\(A_1BC\) but not \(A_2BC\).  Thus the H211 tail/surface certificate remains
separate even when every node is shared.  In the Fourier formulation this
is the slower fourth-order tail already isolated in Section 49.

### 95.4 Literal current-state audit

The common-node premise is stronger than a shared quadrature count.  A
per-unit one-dimensional Fourier/resolvent rule has nodes

\[
 (t_\nu;\,i)
\]

whose tilted law depends on the unit \(i\).  It does not provide one joint
latent vector \(X^{(\nu)}\) on which all three factors in (95.6) coexist.
Equal node indices, equal frequency counts, reused eigensystems, or stored
CP3 tensor factors do not turn those marginal nodes into (95.4).

The strict AQ55 K32/Section 49 implementation currently documented in this
lane is such a per-unit one-dimensional resolvent construction.  No literal
common \(z_\nu\), common joint Fourier triple, or per-node output bank
\(X^{(\nu)}\) has been established.  Therefore the present verdict is

\[
 \boxed{
 \text{node-fusion algebra GO; current literal common-node ownership NO}.}
 \tag{95.10}
\]

An exact reopen requires one of:

1. a genuine common latent cubature with a q211-specific error certificate;
2. a common-ray/block-monomial law reducing the joint surface to one
   coordinate;
3. a joint Fourier evaluator whose CP3 and q211 operands and nodes are
   byte-for-byte the same; or
4. replacement of the separate CP3/H211 paths by one jointly billed surface
   pass.

### 95.5 Ownership ledger

If (95.4) is built as a replacement evaluator, CP3 and AQ q211 are two
outputs of one pass.  Their shared arithmetic does not make either output an
additive feature relative to a comparator already containing the full
surface.  Conversely, a stored CP3 factor tensor supplies neither the
per-node squares in (95.6) nor the true-minus-AQ residual.

Graph-ready split:

- CP3_Q211_COMMON_NODE_FACTOR -- **solved exact**, (95.3)--(95.6);
- CP3_Q211_COMMON_NODE_EXTRA_COST -- **solved conditional**, (95.7);
- CP3_Q211_SHARED_FIDELITY -- **NO-GO without separate H211 receipt**,
  (95.9);
- CP3_Q211_CURRENT_K32_NODE_IDENTITY -- **literal NO**, (95.10);
- CP3_Q211_JOINT_SURFACE_REPLACEMENT -- **conditional GO**;
- CP3_Q211_TRUE_MINUS_AQ -- **not supplied**.

## 96. The exact missing CP3 tail and its fusion into the one q211
sandwich

The literal audit after Section 88 shows that deployed D11 owns a finite
CP3 closure, not exact Gaussian \(K_3\).  The leading missing CP3 sector can
nevertheless be folded into the same broad sandwich already required by
q211; it does not require a second dense product.

### 96.1 Exact connected Hermite series

For centered unit-variance zero-bias ReLU, write

\[
 X_+-E[X_+]=\sum_{d\ge1}{b_d\over d!}\operatorname{He}_d(X),
\]

with

\[
 b_1={1\over2},\qquad
 b_d=\mu\operatorname{He}_{d-2}(0)\quad(d\ge2),\qquad
 \mu={1\over\sqrt{2\pi}}. \tag{96.1}
\]

For correlations \(x=R_{ij},y=R_{ik},z=R_{jk}\), the exact connected
third cumulant is

\[
\boxed{
 K_3(x,y,z)=
 \sum_{\substack{a,b,c\ge0\\
                 (a,b,c)\ {\rm connects}\ \{i,j,k\}}}
 {b_{a+b}b_{a+c}b_{b+c}\over a!\,b!\,c!}
 x^ay^bz^c.} \tag{96.2}
\]

Here connected means either all three edge multiplicities are positive, or
exactly one is zero and the other two are positive.  Formula (96.2) is the
three-vertex connected Wick multigraph expansion.

The literal first-reset D11 birth owns the tree band

\[
 K_3^{\rm tree}={\mu\over4}(xy+xz+yz). \tag{96.3}
\]

With

\[
 \delta K_3:=K_3-K_3^{\rm tree}, \tag{96.4}
\]

the exact first three missing orders are

\[
\boxed{\begin{aligned}
 \delta K_3
 =\mu^3\bigg[&
 xyz-{1\over4}(x^2y^2+x^2z^2+y^2z^2)\\
 &+{1\over6}(x^3yz+xy^3z+xyz^3)
 \bigg]+O_{\rm Herm}(\rho^6),
\end{aligned}} \tag{96.5}
\]

where \(\rho=\max(|x|,|y|,|z|)\).  The remainder is not an informal
big-O: its exact absolute series certificate is

\[
 E_{3,5}(\rho)=
 \sum_{\substack{a+b+c\ge6\\{\rm connected}}}
 \left|
 {b_{a+b}b_{a+c}b_{b+c}\over a!\,b!\,c!}
 \right|\rho^{a+b+c}. \tag{96.6}
\]

The Section 72 Hermite \(L^4\) tail gives the corresponding
distribution-weighted alternative when a uniform \(\rho\) bound is too
loose.

### 96.2 Leading-tail sandwich fusion

Section 88 writes exact q211 as

\[
 K_{211}=-2\mu K_3+F_{\rm edge}, \tag{96.7}
\]

whose only dense CP-atom complement is

\[
 xy\,p(z). \tag{96.8}
\]

Subtracting the deployed tree owner leaves

\[
 -2\mu\delta K_3.
\]

At the leading missing triangle,

\[
 xy\,p(z)-2\mu(\mu^3xyz)
 =xy\{p(z)-2\mu^4z\}. \tag{96.9}
\]

Therefore define the single modified gate matrix

\[
 P^\sharp:=p[R]-2\mu^4R. \tag{96.10}
\]

For one ordered CP endpoint atom,

\[
\boxed{
 \sum_{ijk}\alpha_i\beta_j\gamma_k
 R_{ij}R_{ik}P^\sharp_{jk}
 =\operatorname{tr}
 (D_\alpha R D_\beta P^\sharp D_\gamma R).}
 \tag{96.11}
\]

This is the same one-product circuit as (88.16).  Hence adding the exact
leading \(\delta K_3\) triangle costs no second GEMM and no new forward
state.

The quartic missing band in (96.5) is matvec-only for a CP atom.  For
example,

\[
 \sum_{ijk}\alpha_i\beta_j\gamma_kR_{ij}^2R_{ik}^2
 =\alpha^T\{(R^{\odot2}\beta)
             \odot(R^{\odot2}\gamma)\}, \tag{96.12}
\]

and the other two orientations use two successive
\(R^{\odot2}\) matvecs.  The fifth-order terms
\(x^3yz,xy^3z,xyz^3\) reopen query-specific dense sandwiches unless an
identical powered-edge product is resident.  Thus the sharp one-product
truncation is the tree owner plus triangle and quartic tail.

### 96.3 Honest owner and fidelity gates

Equation (96.3) is the first-reset owner only.  At a later cut, inherited
CP3 columns must be evaluated and subtracted literally; nominal tree
language is not the old projector.  The lawful residual is

\[
 \delta K_{3,\rm old}
 =K_3^{\rm exact}-\Pi_{\rm old}^{CP3}K_3^{\rm exact}. \tag{96.13}
\]

The cheap fused provider consists of:

1. the complete-old residualized finite edge terms in (88.10);
2. the modified one-product sandwich (96.11);
3. the matvec quartic band (96.12); and
4. a certified tail no larger than
   \(2\mu E_{3,5}\) plus the omitted q211-edge tail.

It is exact through total correlation degree four and includes the leading
degree-three CP3 error.  It is not the exact full Gaussian surface unless
the higher-tail certificate is below the endpoint tolerance.  A full AQ
surface oracle owns all of it and closes additive credit.

The Base48-to-49 one-equivalent rule is a fail-fast convenience gate, not
the final adjusted-score economics when the verified f32 utility pack is
co-fired.  If the sandwich is nonresident at all ten seams, its ten f32
matmul-equivalents add about \(1.61\%\) relative billing at the audited
\(0.161\%\) per equivalent.  Applied to the held \(u=0.5629\) pack this is
approximately

\[
 0.5629(1+0.0161)\simeq0.5720, \tag{96.14}
\]

still below the current \(u=0.6002\).  Therefore the correct decision split
is:

1. use the one-equivalent gate for a zero-displacement implementation;
2. if it fails, retain the ten-product replacement when its
   complete-old-residualized raw value clears the fire bar; and
3. price the literal f32 package end to end before submission.

This does not prove raw value.  It prevents a potentially goal-scale raw
provider from being discarded solely because the convenience cap was
mistaken for a hard adjusted-score constraint.

Graph-ready split:

- D11_CP3_TREE_OWNER -- **literal first-reset finite owner**, (96.3);
- GAUSSIAN_CP3_MISSING_CONNECTED_SERIES -- **solved exact**, (96.2);
- GAUSSIAN_CP3_MISSING_Q5 -- **solved exact**, (96.5);
- CP3_TRIANGLE_Q211_SANDWICH_FUSION -- **solved exact**, (96.9)--(96.11);
- CP3_QUARTIC_Q211_MATVEC -- **solved exact**, (96.12);
- CP3_HIGHER_TAIL_ONE_PRODUCT -- **generic NO**, fifth order reopens
  products;
- CP3_Q211_Q4_FIDELITY -- **conditional** on (96.6) and endpoint weighting;
- CP3_Q211_WITH_FULL_AQ_SURFACE -- **closed duplicate**.

## 98. Joint trivariate determinant evaluator for quadratic AQ: exact
CP3--q211 node reuse and the triplet-density obstruction

Section 95's common-latent-node fusion is not the only exact joint
evaluator.  A quadratic Gaussian AQ law permits the latent variable to be
integrated out analytically at each *trivariate transform node*.  This
removes the need for common latent samples, but it exposes a different and
usually binding cost: a generic transform node belongs to one ordered
physical triple and has no per-unit eigensystem.

Let \(z\sim N(0,I_r)\), and for a physical unit \(a\) write

\[
 g_a(z)=c_a+b_a^Tz+\frac12
 \{z^TM_az-\operatorname{tr}M_a\},\qquad Y_a=(g_a)_+ .
 \tag{98.1}
\]

No commutativity among the real symmetric matrices \(M_a\) is assumed.

### 98.1 Exact joint characteristic determinant

For one ordered strict triple \((i,j,k)\), put

\[
 \begin{aligned}
 M(\zeta)&=\zeta_1M_i+\zeta_2M_j+\zeta_3M_k,\\
 b(\zeta)&=\zeta_1b_i+\zeta_2b_j+\zeta_3b_k,\\
 \bar c(\zeta)&=\sum_{a=1}^3\zeta_a
 \left(c_{i_a}-\frac12\operatorname{tr}M_{i_a}\right),\\
 D(\zeta)&=I-M(\zeta),\qquad S(\zeta)=D(\zeta)^{-1},
 \end{aligned}
 \tag{98.2}
\]

where \((i_1,i_2,i_3)=(i,j,k)\).  Choose a positive damping vector
\(\eta\) with

\[
 I-\eta_1M_i-\eta_2M_j-\eta_3M_k\succ0 .
 \tag{98.3}
\]

Such a vector always exists after taking the three positive components
sufficiently small.  On the tube
\(\zeta=\eta+it\), Gaussian completion gives the exact transform

\[
 \boxed{
 \Phi_{ijk}(\zeta)
 :=E\exp\{\zeta_1g_i+\zeta_2g_j+\zeta_3g_k\}
 =e^{\bar c(\zeta)}\det D(\zeta)^{-1/2}
 \exp\!\left\{\frac12b(\zeta)^TS(\zeta)b(\zeta)\right\}.}
 \tag{98.4}
\]

The determinant branch is the analytic continuation of the positive branch
at the origin.  Positive real part in (98.3) makes \(D(\eta+it)\)
nonsingular for every real \(t\).

There is also an exact source derivative which reuses the same factorization.
With \(m=S b(\zeta)\),

\[
 \boxed{
 \partial_{\zeta_1}\Phi_{ijk}=\Phi_{ijk}\,\ell_i,\qquad
 \ell_i=c_i-\frac12\operatorname{tr}M_i
 +\frac12\operatorname{tr}(SM_i)
 +b_i^Tm+\frac12m^TM_im .}
 \tag{98.5}
\]

Thus even the differentiated integrand needs no second determinant or
eigensystem.

### 98.2 Repeated q211 is one source derivative, or one changed scalar
weight

For every integer \(p\ge0\), inverse Laplace transformation gives

\[
 x_+^p=\frac{p!}{2\pi}
 \int_{\mathbb R}\frac{e^{(\eta+it)x}}
 {(\eta+it)^{p+1}}\,dt .
 \tag{98.6}
\]

Because
\(|\Phi(\eta+it)|\le\Phi(\eta)\), the three integrals below are
absolutely convergent.  The raw CP3 moment is

\[
 \boxed{
 M_{111}^{ijk}
 ={1\over(2\pi)^3}\int_{\mathbb R^3}
 {\Phi_{ijk}(\zeta)\over
  \zeta_1^2\zeta_2^2\zeta_3^2}\,dt .}
 \tag{98.7}
\]

Since \(g_iY_i=Y_i^2\), the repeated-slot moment is first

\[
 M_{211}^{i;jk}
 ={1\over(2\pi)^3}\int_{\mathbb R^3}
 {\partial_{\zeta_1}\Phi_{ijk}(\zeta)\over
  \zeta_1^2\zeta_2^2\zeta_3^2}\,dt .
 \tag{98.8}
\]

Integration by parts on the vertical
\(\zeta_1\) contour has zero boundary term and yields the cheaper form

\[
 \boxed{
 M_{211}^{i;jk}
 ={2\over(2\pi)^3}\int_{\mathbb R^3}
 {\Phi_{ijk}(\zeta)\over
  \zeta_1^3\zeta_2^2\zeta_3^2}\,dt .}
 \tag{98.9}
\]

Equations (98.7) and (98.9) use exactly the same determinant values.  The
anchor bias derivative supplies an independent audit:

\[
 \boxed{\partial_{c_i}M_{211}^{i;jk}=2M_{111}^{ijk}.}
 \tag{98.10}
\]

Connected \(K_3\) is obtained from (98.7) by the usual mean/pair
subtractions.  Connected q211 is obtained from (98.9) through

\[
 \kappa_4(Y_i,Y_i,Y_j,Y_k)
 =E[X_i^2X_jX_k]-\Sigma_{ii}\Sigma_{jk}
                    -2\Sigma_{ij}\Sigma_{ik},
 \tag{98.11}
\]

where \(X=Y-EY\).  Hence means, pairs, and collision sewing remain separate
old rows; the joint determinant supplies only the raw three-label surfaces.

### 98.3 What a shared transform node really buys

Let a trivariate cubature have geometric nodes and weights
\((\zeta^\nu,\omega_\nu)_{\nu=1}^Q\).  At a node define

\[
 F^\nu_{ijk}:=\Phi_{ijk}(\zeta^\nu),
 \quad
 a_\nu={\omega_\nu\over
 (\zeta_1^\nu)^2(\zeta_2^\nu)^2(\zeta_3^\nu)^2},
 \quad
 b_\nu={2a_\nu\over\zeta_1^\nu}.
 \tag{98.12}
\]

For one ordered CP endpoint atom
\(L_{i;jk}=\alpha_i\beta_j\gamma_k\), put

\[
 B_\nu(\alpha,\beta,\gamma)
 :=\sum_{i,j,k}^{\rm strict}
 \alpha_i\beta_j\gamma_kF^\nu_{ijk} .
 \tag{98.13}
\]

Then the same batched scalar is accumulated as

\[
 \widehat S_3=\sum_\nu a_\nu B_\nu,
 \qquad
 \widehat S_{211}=\sum_\nu b_\nu B_\nu .
 \tag{98.14}
\]

The other two anchor orientations merely replace \(2/\zeta_1^\nu\) by
\(2/\zeta_2^\nu\) or \(2/\zeta_3^\nu\).  A partially symmetric
\((\beta,\gamma)\) atom uses the corresponding swapped-node average.

Thus node locations, geometric cubature weights, determinant factorizations,
and characteristic values are shared exactly.  The *effective* CP3 and
q211 weights are not identical: they differ by \(2/\zeta_a^\nu\).  Using
(98.5) instead of (98.9) is also exact, but adds the label-dependent
\(\ell_i\) contraction and is never a stronger sharing theorem.

The current 34-node receipt in Sections 44 and 49 is one-dimensional: it
integrates one scalar quadratic form and its Hermite-directional resolvent.
It is not a cubature of the three-dimensional tube in (98.7).  Equal node
indices or equal node counts do not turn it into one.  A tensor product of
34 one-dimensional rules has

\[
 Q=34^3=39{,}304,
 \tag{98.15}
\]

not 34 nodes.  A genuinely new 34-point sparse trivariate rule may share
(98.12)--(98.14), but it needs its own instance certificate.

Node reuse also requires the nodewise values, not merely the accumulated
CP3 scalar.  On a fixed queried atom, a stored row
\(\sum_\nu a_\nu B_\nu\) determines
\(\sum_\nu b_\nu B_\nu\) for every unrestricted node vector
\(B\in\mathbb C^Q\) if and
only if

\[
 \boxed{\ker a^T\subseteq\ker b^T
 \quad\Longleftrightarrow\quad b=\lambda a.}
 \tag{98.15a}
\]

Because \(b_\nu/a_\nu=2/\zeta_1^\nu\) is not constant on a nondegenerate
grid, an accumulated CP3 value does not own q211.  Retained
\(F^\nu_{ijk}\), or an exact factor representation exposing every needed
node contraction, is the literal reuse receipt.  A smaller restricted AQ
family can evade (98.15a) only through the corresponding literal restricted
row-span identity.

Nor does the quadratic determinant have a universal 34-term separation in
the transform coordinates.  In latent dimension one, with two identical
pure quadratic generators and the third transform coordinate set to zero,
nonzero one-coordinate factors reduce a bivariate slice to

\[
 \Phi(x,y,0)=(1-x-y)^{-1/2}.
 \tag{98.15b}
\]

Its Taylor coefficient matrix is

\[
 C_{mn}={(\tfrac12)_{m+n}\over m!\,n!}.
 \tag{98.15c}
\]

After invertible diagonal row/column scalings this is the moment Hankel
matrix of the positive measure
\(t^{-1/2}e^{-t}dt\).  Every finite leading minor is strictly positive,
because \(\int p(t)^2t^{-1/2}e^{-t}dt>0\) for every nonzero polynomial
\(p\).  Hence (98.15b) has infinite exact separation rank.  A 34-term
one-dimensional CP3/resolvent expansion cannot be promoted to the exact
trivariate determinant by relabeling its nodes.

### 98.4 A separate, exact q211 tail certificate

The damped contour gives a simple deterministic tail bound.  Define

\[
 J_m(\eta)=\int_{\mathbb R}(\eta^2+t^2)^{-m/2}dt,
 \tag{98.16}
\]

so that

\[
 J_2(\eta)={\pi\over\eta},\qquad
 J_3(\eta)={2\over\eta^2}.
 \tag{98.17}
\]

For a truncation box
\(\Omega_T=\prod_a[-T_a,T_a]\), put

\[
 \begin{aligned}
 J_{2,>{T}}(\eta)
 &={2\over\eta}\left\{{\pi\over2}
       -\arctan(T/\eta)\right\},\\
 J_{3,>{T}}(\eta)
 &={2\over\eta^2}\left\{1-{T\over\sqrt{T^2+\eta^2}}\right\}.
 \end{aligned}
 \tag{98.18}
\]

Writing \(p=(2,2,2)\) for CP3 and \(p=(3,2,2)\) for anchor-first
q211, the union bound gives

\[
 \boxed{
 E_{p,\rm tail}(T)
 \le {c_p\Phi_{ijk}(\eta)\over(2\pi)^3}
 \sum_{a=1}^3J_{p_a,>T_a}(\eta_a)
 \prod_{b\ne a}J_{p_b}(\eta_b),}
 \tag{98.19}
\]

with \(c_{222}=1\) and \(c_{322}=2\).  This is conservative but exact and
contains no asymptotic assumption.  It also shows the precise sharing
limit: an *absolute-envelope* CP3 tail can be rescaled to bound q211, but a
signed CP3 quadrature error cannot.  The q211 core integrand has a
third-order pole at
\(\zeta_1=0\), rather than the CP3 second-order pole, and needs its own
polyellipse/sparse-grid error bound.  Therefore the same 34 abscissae can be
retained only when the q211-specific sum of (98.19) and its core cubature
bound meets the q211 tolerance.  The old CP3 certificate alone is not that
receipt.

There is no uniform fixed-34 theorem over unrestricted biases and spectra.
At fixed nodes the bias dependence is a finite sum of damped exponentials,
whereas the exact ReLU moments have Gaussian-tail behavior in the negative
bias direction and polynomial behavior in the positive direction.  A
fixed finite sum cannot match both uniformly.  Bias-adapted weights remove
that elementary obstruction but not branch hypersurfaces
\(\det D(\zeta)=0\) approaching the integration tube.  As in Section 44,
the node count is necessarily an instance certificate.

### 98.5 Eigensystem and Cap49 census

Let \(P\) be the number of ordered strict physical triples actually touched
after sparsity and exact old/collision projection, \(R_c\) the number of CP
endpoint atoms, \(D\) the active seam count, and \(Q\) the *total*
trivariate node count.  A generic noncommuting pencil needs one complex
\(r\)-dimensional factorization per triple and node.  Even giving every
unit arithmetic constant the favorable value one, the acquisition and
batched-action cost is bounded below by

\[
 \boxed{
 C_{\rm tri}^{\rm generic}
 \gtrsim {DPQ(r^3+R_c)\over n^3}.}
 \tag{98.20}
\]

All \(R_c\) atoms and both output surfaces reuse each determinant, which is
why \(R_c\) is additive rather than multiplicative with \(r^3\).  Forming
the pencils, lower-moment rows, strict masks, complex constants, and writes
are omitted from (98.20), so it is a necessary gate only.

Two distinct generic linear combinations of \(M_i,M_j,M_k\) do not share an
eigenbasis.  For real symmetric generators, a node-independent orthogonal
basis exists exactly when the three matrices commute pairwise.  In that
exception, one eigensystem per triple gives

\[
 C_{\rm tri}^{\rm commute}
 \gtrsim {DP\{r^3+Q(r+R_c)\}\over n^3}.
 \tag{98.21}
\]

If every physical \(M_a\) belongs to one globally and literally resident
simultaneous eigenbasis, even the \(r^3\) term can be removed:

\[
 C_{\rm tri}^{\rm global}
 \gtrsim {DPQ(r+R_c)\over n^3}.
 \tag{98.22}
\]

The literal 49-equivalent gate is therefore

\[
 \boxed{
 {DPQ(r^3+R_c)\over n^3}
 +C_{\le2}+C_{\rm coll}+C_{\rm build}\le49,}
 \tag{98.23}
\]

with (98.21) or (98.22) substituted only on receipt of the corresponding
commuting resident basis.  At \(n=1024,r=55,D=10,Q=34\), (98.23), before
all omitted costs and even with \(R_c=0\), permits only about 930 ordered
triples.  A dense physical surface costs more than \(5.6\times10^7\)
equivalents.  The tensor-product \(34^3\) rule costs more than 49
equivalents for even one generic triplet.  Even a globally resident common
basis leaves \(10\cdot34\cdot55=18{,}700\) equivalents for a dense surface.
Hence eigensystem sharing alone cannot reopen the route.

### 98.6 The only exact CP batching escape, and its generic obstruction

Define the strict, complete-old-projected node stack

\[
 {\cal F}_{ijk\nu}
 :=\Pi_{\rm new}^{\rm strict}F^\nu_{ijk}.
 \tag{98.24}
\]

An exact physical CP escape exists if this *four-way* stack has a common
factorization

\[
 \boxed{
 {\cal F}_{ijk\nu}
 =\sum_{h=1}^{K}u_{h,i}v_{h,j}w_{h,k}d_{h,\nu}.}
 \tag{98.25}
\]

Then

\[
 B_\nu(\alpha,\beta,\gamma)
 =\sum_h d_{h,\nu}
 (\alpha^Tu_h)(\beta^Tv_h)(\gamma^Tw_h),
 \tag{98.26}
\]

and the CP3/q211 node weights can be collapsed into two scalar coefficient
vectors on the same physical rays.  Conditional on the factors already
being literal resident state, the optimistic Cap49 action gate is

\[
 \boxed{
 C_{\rm acquire}
 +{DK\{3R_cn+2Q\}\over n^3}
 +C_{\le2}+C_{\rm coll}\le49.}
 \tag{98.27}
\]

For only the two integrated outputs, (98.25) may be weakened to a common-ray
factorization of the two weighted tensors.  Its minimum coupled quotient
rank, after old and collision projection, is the exact rank to insert in
(98.27).  A CP rank name before that projection is not a receipt, and
discovering the factors by enumerating (98.4) pays (98.20).

There is no generic bounded-\(K\) theorem.  It already fails in the affine
rank-one subclass \(M_a=0,\ b_a=q_ae_1\).  At one positive damping node,

\[
 F_{ijk}=e^{\zeta_1c_i+\zeta_2c_j+\zeta_3c_k}
 \exp\!\left\{\frac12
 (\zeta_1q_i+\zeta_2q_j+\zeta_3q_k)^2\right\}.
 \tag{98.28}
\]

After nonzero row and column scalings, an anchor-versus-pair flattening has
entries

\[
 \exp\{\zeta_1q_i x_{jk}\},qquad
 x_{jk}=\zeta_2q_j+\zeta_3q_k .
 \tag{98.29}
\]

Choose distinct real \(q_i\) and distinct \(x_{jk}\) on disjoint strict
label blocks.  The exponential Vandermonde kernel is strictly totally
positive, so every such \(m\)-by-\(m\) minor is nonzero.  Therefore

\[
 \boxed{\operatorname{rank}_{CP}F^\nu
 \ge\operatorname{rank}F^\nu_{i\mid(jk)}\ge m,}
 \tag{98.30}
\]

and the nonzero minor persists on an open set of quadratic AQ parameters.
Thus neither latent rank one, a shared transform grid, nor an existing CP3
factor count implies a small physical node-stack rank.  Old projection can
remove this witness only through a literal row-span receipt; that is exactly
why (98.24), not the unprojected source rank, is binding.

### 98.7 Disposition

The determinant route is an exact mathematical replacement for common
latent cubature, and CP3 plus AQ q211 are genuinely two outputs of one
trivariate surface pass.  It is not an additive feature when the full
Section 49 surface is already billed.  It also supplies no
true-minus-quadratic-AQ residual.

For the current architecture the lawful verdict is:

\[
 \boxed{
 \text{local joint-integrand fusion GO; dense trivariate deployment NO
 absent (98.25) and (98.27).}}
 \tag{98.31}
\]

Graph-ready split:

- AQ_TRIVARIATE_CHARACTERISTIC_DETERMINANT -- **solved exact**,
  (98.2)--(98.5);
- AQ_CP3_Q211_BROMWICH_JET -- **solved exact**, (98.7)--(98.10);
- AQ_CP3_Q211_TRIVARIATE_NODE_SHARE -- **exact node/determinant reuse but
  different effective weights**, (98.12)--(98.14);
- AQ_Q211_TRIVARIATE_TAIL -- **solved conservative certificate**,
  (98.16)--(98.19);
- AQ_CURRENT_34_NODE_REUSE -- **literal NO**: one-dimensional nodes are not
  trivariate cubature;
- AQ_CP3_SCALAR_TO_Q211_NODE_ROW -- **generic NO-GO** by (98.15a);
- AQ_TRIVARIATE_34_TERM_SEPARATION -- **infinite-rank NO-GO** by
  (98.15b)--(98.15c);
- AQ_NEW_34_NODE_TRIVARIATE_RULE -- **instance-conditional only** with its
  own q211 certificate;
- AQ_TRIVARIATE_EIGENSYSTEM_REUSE -- **generic NO; commuting conditional**,
  (98.20)--(98.22);
- AQ_TRIVARIATE_CAP49_ENUMERATION -- **closed for a dense surface** by
  (98.23);
- AQ_TRIVARIATE_COMMON_CP_STACK -- **conditional GO only** under
  (98.25)--(98.27);
- AQ_TRIVARIATE_GENERIC_SMALL_RANK -- **NO-GO**, (98.28)--(98.30);
- AQ_TRIVARIATE_WITH_FULL_SURFACE -- **replacement/zero double credit**;
- AQ_TRIVARIATE_TRUE_MINUS_AQ -- **not supplied**.

## 101. Direct Bellman scalar versus a finite transition repair: exact
homotopy identity, co-adaptation term, and raw-MSE fidelity gate

Many constructions in this lane end with one target-conditioned scalar
action.  That fact removes the need to *materialize the source tensor* for
one frozen endpoint, but it does not by itself make the finite source
repair equal to its baseline adjoint action.  This section separates those
two statements exactly.

### 101.1 Exact homotopy and the baseline direct action

Let the baseline closure trajectory in finite-dimensional Hilbert spaces be

\[
 x_{\ell+1}=F_\ell(x_\ell),\qquad J(0)=\Phi(x_D),
 \tag{101.1}
\]

and insert the proposed typed transition defect through the homotopy

\[
 x_{\ell+1}^{\epsilon}
 =F_\ell(x_\ell^{\epsilon})
   +\epsilon G_\ell(x_\ell^{\epsilon}),
 \qquad J(\epsilon)=\Phi(x_D^{\epsilon}).
 \tag{101.2}
\]

All maps below are assumed differentiable on the homotopy tube.  Define the
exact homotopy costate

\[
 \lambda_D^\epsilon=D\Phi(x_D^\epsilon),\qquad
 \lambda_\ell^\epsilon
 =D(F_\ell+\epsilon G_\ell)(x_\ell^\epsilon)^*
   \lambda_{\ell+1}^\epsilon .
 \tag{101.3}
\]

Differentiating (101.2) and telescoping the tangent/costate pairing gives
the exact identity

\[
 \boxed{
 J(1)-J(0)=
 \int_0^1\sum_{\ell=0}^{D-1}
 \langle\lambda_{\ell+1}^\epsilon,
         G_\ell(x_\ell^\epsilon)\rangle\,d\epsilon .}
 \tag{101.4}
\]

Thus one scalar per layer is information-sufficient at a *fixed value of
\(\epsilon\)* once its exact costate exists.  A baseline q211/H3/K31 action
is only the left-endpoint integrand

\[
 S_1=J'(0)=\sum_\ell
 \langle\lambda_{\ell+1},d_\ell\rangle,
 \quad
 d_\ell=G_\ell(x_\ell),
 \quad
 \lambda_\ell=A_\ell^*\lambda_{\ell+1},
 \quad A_\ell=DF_\ell(x_\ell).
 \tag{101.5}
\]

Consequently, `one output scalar' and `one exact finite-repair scalar' are
equivalent only after proving that the integrand in (101.4) is constant,
or after evaluating the integral with its own certified rule.  A frozen
baseline reverse row proves neither statement.

### 101.2 The exact co-adaptation scalar

Put \(\xi_0=\zeta_0=0\).  The first two state variations are

\[
 \begin{aligned}
 \xi_{\ell+1}
 &=A_\ell\xi_\ell+d_\ell,\\
 \zeta_{\ell+1}
 &=A_\ell\zeta_\ell
   +D^2F_\ell(x_\ell)[\xi_\ell,\xi_\ell]
   +2DG_\ell(x_\ell)\xi_\ell .
 \end{aligned}
 \tag{101.6}
\]

The exact second derivative can be written without storing \(\zeta\):

\[
 \boxed{\begin{aligned}
 J''(0)=&\ D^2\Phi(x_D)[\xi_D,\xi_D]\\
 &+\sum_{\ell=0}^{D-1}
 \left\langle\lambda_{\ell+1},
 D^2F_\ell(x_\ell)[\xi_\ell,\xi_\ell]
 +2DG_\ell(x_\ell)\xi_\ell\right\rangle .
 \end{aligned}}
 \tag{101.7}
\]

The two summands have distinct meanings.  The Hessian term is interaction
of upstream births through the nonlinear old closure; the \(DG_\ell\)
term is source re-evaluation on the changed trajectory.  Neither is owned
by the same-site scalar (101.5).  A joint refit may fit some of this
interaction empirically, but that is a value receipt rather than an
algebraic identity.

The obstruction is already one-dimensional.  Let
\(x_1^\epsilon=\epsilon a\), \(x_2^\epsilon=(x_1^\epsilon)^2\), and
\(\Phi(x)=x\).  At the baseline, every direct action in (101.5) is zero
because the second transition has derivative zero, whereas

\[
 J(1)-J(0)=a^2={1\over2}J''(0). \tag{101.8}
\]

Hence no theorem based only on the number or exactness of local scalar
actions can remove the co-adaptation gate.

### 101.3 A deterministic third-order certificate

The finite-repair error can be certified without a tensor norm guess.  On
the full homotopy tube define uniform operator bounds

\[
 \begin{array}{lll}
 a_\ell=\|D(F_\ell+\epsilon G_\ell)\|,&
 b_\ell=\|D^2(F_\ell+\epsilon G_\ell)\|,&
 c_\ell=\|D^3(F_\ell+\epsilon G_\ell)\|,\\
 g_{0\ell}=\|G_\ell\|,&
 g_{1\ell}=\|DG_\ell\|,&
 g_{2\ell}=\|D^2G_\ell\|.
 \end{array}
 \tag{101.9}
\]

Starting with \(U_0=V_0=W_0=0\), form the scalar majorants

\[
 \begin{aligned}
 U_{\ell+1}&=a_\ell U_\ell+g_{0\ell},\\
 V_{\ell+1}&=a_\ell V_\ell+b_\ell U_\ell^2
                         +2g_{1\ell}U_\ell,\\
 W_{\ell+1}&=a_\ell W_\ell+c_\ell U_\ell^3
 +3b_\ell U_\ell V_\ell
 +3g_{2\ell}U_\ell^2+3g_{1\ell}V_\ell.
 \end{aligned}
 \tag{101.10}
\]

These follow from the exact third tangent recursion

\[
 x'''_{\ell+1}=D H_\ell x'''_\ell
 +D^3H_\ell[x'_\ell,x'_\ell,x'_\ell]
 +3D^2H_\ell[x'_\ell,x''_\ell]
 +3D^2G_\ell[x'_\ell,x'_\ell]+3DG_\ell x''_\ell,
 \tag{101.11}
\]

where \(H_\ell=F_\ell+\epsilon G_\ell\).  If
\(p_j=\sup\|D^j\Phi\|\) on the endpoint tube, then

\[
 B_3:=p_1W_D+3p_2U_DV_D+p_3U_D^3
 \quad\Longrightarrow\quad
 \boxed{
 \left|J(1)-J(0)-J'(0)-{1\over2}J''(0)\right|
 \le {B_3\over6}.}
 \tag{101.12}
\]

Likewise the baseline-only provider obeys

\[
 \left|J(1)-J(0)-S_1\right|
 \le {1\over2}|J''(0)|+{B_3\over6}.
 \tag{101.13}
\]

This is the sharp discipline split for the proposed q211 scalar: either
carry the second action (101.7), certify (101.13), or call the result a
first-order screen.  Source smallness alone is not enough; the amplified
quantities \(U,V,W\) contain the depth conditioning.

### 101.4 Old-span and raw-MSE gate

Apply the complete-old corpus projector \(P_\perp\) to the exact finite
correction vector \(t\) and an implemented direct-action vector \(f\).
Projection is contractive, so an absolute remainder certificate survives,
but its *relative* size must be recomputed after projection.  If

\[
 \|P_\perp(t-f)\|_2
 \le\varepsilon\|P_\perp t\|_2,
 \qquad 0\le\varepsilon<1,
 \tag{101.14}
\]

then the geometric fidelity bound is

\[
 \boxed{
 \operatorname{corr}(P_\perp t,P_\perp f)
 \ge\sqrt{1-\varepsilon^2}.}
 \tag{101.15}
\]

For value, let \(r\) be the old-residualized target, and suppose the exact
finite correction has correlation \(\rho_t\) with \(r\).  The robust
one-feature lower bound is

\[
 \boxed{
 \operatorname{corr}(r,P_\perp f)
 \ge {\rho_t-\varepsilon\over1+\varepsilon}.}
 \tag{101.16}
\]

No source-norm, tensor-rank, or pointwise-sign statement can replace
\(\rho_t\).  With one fitted feature, the raw-MSE fraction removed is the
square of this correlation.

For the current baseline \(2.9409\times10^{-8}\), raw
\(10^{-8}\) requires

\[
 \rho_{10^{-8}}=
 \sqrt{1-{10^{-8}\over2.9409\times10^{-8}}}
 \simeq0.8124. \tag{101.17}
\]

Even in the optimistic case \(\rho_t=1\), (101.16) therefore requires

\[
 \varepsilon\le{1-0.8124\over1+0.8124}\simeq0.1035. \tag{101.18}
\]

Raw \(6\times10^{-9}\) requires correlation about \(0.8922\), hence
\(\varepsilon\lesssim0.0570\) under the same optimistic premise.  These
are the correct co-adaptation fidelity gates for a redemption-scale claim;
the 20% and 30% fire bars alone require correlations \(0.4472\) and
\(0.5477\), respectively.

### 101.5 A refitted feature is a different claim

The co-adaptation bounds above are necessary for claiming that the scalar
with its physical coefficient is a faithful finite transition repair.
They are **not** a necessary condition for using (S_1) as an empirically
refitted predictor.  With

\[
 r=P_\perp(y-\widehat y_0),\qquad f=P_\perp S_1,
 \tag{101.19}
\]

the optimal one-feature coefficient and exact corpus gain are

\[
 \boxed{
 \widehat\alpha={\langle r,f\rangle\over\|f\|^2},\qquad
 \Delta_{\rm MSE}={1\over N}{\langle r,f\rangle^2\over\|f\|^2}.}
 \tag{101.20}
\]

Thus a refitted first-variation feature reaches raw (10^{-8}) exactly
when its old-residualized corpus correlation reaches (0.8124), regardless
of whether (widehat\alpha=1) or whether (101.13) is small.  In that
branch, higher homotopy terms are useful prior information about alignment,
not an algebraic deployment gate.  Conversely, passing the 10.35% fidelity
bound does not prove value unless the exact finite repair itself aligns
with the target residual.

The two claims must therefore remain separate:

1. **physics repair:** coefficient fixed by the derived transition law;
   use (101.12)--(101.18);
2. **statistical feature:** coefficient frozen only after preregistered
   training; use (101.20) and held-out old-residualized correlation.

This distinction is especially important for the proposed layerwise q211
panel.  Joint refitting can absorb scale and some signed cancellation, but
it cannot manufacture a held-out direction that is absent from the feature
span.

### 101.6 Exactness and disposition

The baseline scalar (101.5) is an exact finite correction only in special
cases, for example when the reachable defect module sees affine old maps,
state-independent defects, and an affine endpoint, or more generally when
every higher homotopy derivative in (101.4) is endpoint-dark.  The exact
necessary statement is constancy of the projected homotopy integrand; a
low-dimensional cyclic-module version is developed separately in Section
102.

Graph-ready split:

- DIRECT_BELLMAN_HOMOTOPY_ACTION -- **solved exact**, (101.4);
- DIRECT_BELLMAN_BASELINE_SCALAR -- **first variation only**, (101.5);
- DIRECT_BELLMAN_COADAPTATION -- **solved exact second action**,
  (101.6)--(101.7);
- DIRECT_SCALAR_FINITE_EXACTNESS -- **generic NO**, counterexample
  (101.8);
- DIRECT_SCALAR_THIRD_REMAINDER -- **solved deterministic certificate**,
  (101.9)--(101.13);
- DIRECT_SCALAR_OLD_SPAN_FIDELITY -- **solved conditional**,
  (101.14)--(101.16);
- DIRECT_SCALAR_RAW_1E8_PHYSICS_GATE -- **requires at least 0.8124
  correlation and at most 10.35% optimistic relative co-adaptation error**,
  (101.17)--(101.18);
- DIRECT_SCALAR_REFIT_RAW_GATE -- **exact corpus theorem**: correlation
  at least 0.8124, with no coefficient-one premise, (101.19)--(101.20);
- DIRECT_SCALAR_VALUE -- **open until complete-old corpus alignment is
  established**.

## 100. Deploying the signed-cubic q211 response: exact star/Jordan adjoint,
common-sketch rank, and Cap-49 gates

Section 97 identifies the genuinely new first-order source.  Specialize now
to a reset-local signed cubic Wiener defect

\[
 H_a(z)=I_3(Z_a)
 ={1\over3!}\langle Z_a,z^{\diamond3}\rangle,
 \qquad Z_a\in\mathscr H_3:=\operatorname{Sym}^3(\mathbb R^{55}).
 \tag{100.1}
\]

Use orthonormal symmetric coordinates: multiplicity-
((3),(21),(111)) entries have weights (1,\sqrt3,\sqrt6).  Then

\[
 p:=\dim\mathscr H_3=\binom{57}{3}=29260, \tag{100.2}
\]

and the response is the Euclidean tensor pairing

\[
 \boxed{\dot S_L=\sum_a\langle Z_a,G_{L,a}\rangle,qquad
 G_{L,a}:={1\over3!}E[\nabla_z^3\Xi_{L,a}].} \tag{100.3}
\]

Complete-old subtraction, collision masks, corpus weights, and response-
space residualization are applied before any rank, sketch, or value claim.

### 100.1 Exact reduction to the existing star/Jordan alphabet

For a scalar AQ factor (f_s(g_s)), put

\[
 v_s=\nabla g_s,\quad B_s=\nabla^2g_s,\quad
 d_s=f_s'(g_s),\quad e_s=f_s''(g_s),\quad t_s=f_s'''(g_s),
\]

and

\[
\begin{aligned}
 Q_s^{(1)}&=d_sv_s,\\
 Q_s^{(2)}&=e_sv_s^{\otimes2}+d_sB_s,\\
 Q_s^{(3)}&=t_sv_s^{\otimes3}+3e_sJ_{v_s}(B_s).
\end{aligned} \tag{100.4}
\]

Here (J_v(B)=\operatorname{Sym}(v\otimes B)) is the averaged Jordan
insertion of Sections 34, 37, and 38.  For
(M=\prod_{s=1}^m f_s(g_s)),

\[
\boxed{\begin{aligned}
 \nabla^3M={}&\sum_sQ_s^{(3)}\prod_{r\ne s}f_r\\
 &+3\sum_{s\ne t}\operatorname{Sym}
  (Q_s^{(2)}\otimes Q_t^{(1)})
       \prod_{r\notin\{s,t\}}f_r\\
 &+6\sum_{s<t<u}\operatorname{Sym}
  (Q_s^{(1)}\otimes Q_t^{(1)}\otimes Q_u^{(1)})
       \prod_{r\notin\{s,t,u\}}f_r.
\end{aligned}} \tag{100.5}
\]

Expanding the middle line shows that every term is one of only

\[
 \mathsf S(v,w,u):=\operatorname{Sym}(v\otimes w\otimes u),
 \qquad J_v(B), \tag{100.6}
\]

with AQ gate/surface weights (I,\delta,\delta',\delta'') and centered-
output products.  This is the exact gate-weighted extension of (38.1); no
fourth-order tensor alphabet appears.

The CP influence (97.17) has eleven scalar monomial families before strict
collision corrections: five in ({\cal A}_{\beta\gamma}), three in each of
({\cal B}_{\alpha\gamma},{\cal B}_{\alpha\beta}).  An (m)-factor
monomial produces at most

\[
 m+m(m-1)+\binom m3 \quad\hbox{star families},qquad
 m^2 \quad\hbox{Jordan families}. \tag{100.7}
\]

The factor counts in (97.15)--(97.17) therefore give at most

\[
 36+3(8)+2+2(19+8+2)=120 \tag{100.8}
\]

pre-cancellation structural families per CP atom.  This is an arithmetic
audit, not a rank theorem.  Different surface weights make identically named
words different tensors, and the stars (v^{\otimes3}) span all of
(\mathscr H_3) by polarization.  The exact query-dictionary rank is

\[
 \boxed{r_G=\operatorname{rank}
 \{\Pi_{\rm old}^{\perp}E[\text{weighted nodes in (100.5)}]\}\le p,}
 \tag{100.9}
\]

not (9), (120), or the CP atom count by name alone.

If (Z_a) is the implicit boundary source (34.1), linearity lets the
resolvent formula (37.4) consume the node sum without materializing it.  With
reused eigensystems, its conservative Cap-49 node gate is

\[
 \boxed{10q\,m_{\rm eff}{55^2\over1024^2}
       +C_{\rm rotate}+C_{\rm hook}\le1.} \tag{100.10}
\]

Even with zero setup overhead, (q\,m_{\rm eff}\le34).  Thus a generic
120-family list needs substantial algebraic coalescence; a Jordan name does
not make its rotations or contractions free.

### 100.2 Three exact contraction ranks

Let ({\cal Z}:{\cal X}\to\bigoplus_\omega\mathscr H_3) be the admissible
owner-subtracted source map and
({\cal G}:{\cal Y}\to\bigoplus_\omega\mathscr H_3) the complete q211
query map, where (omega) ranges over net, layer, corpus row, and unit.
Absorb the endpoint metric and complete-old projector into these maps.  On
whitened nonnull domains define

\[
 C_{\rm bil}:={\cal Z}^*{\cal G}. \tag{100.11}
\]

Then

\[
 \boxed{r_{\rm bil}=\operatorname{rank}C_{\rm bil}} \tag{100.12}
\]

is the minimum total number of arbitrary paired scalar channels, tight by
the SVD.  It is not automatically the width of one latent-coordinate sketch
repeated at every (omega).

For a common (S:\mathscr H_3\to\mathbb R^k), put (M=S^TS\succeq0).
The exact target-conditioned common-sketch rank is

\[
\boxed{\begin{aligned}
 k_*:=\min_{M\succeq0}\ &\operatorname{rank}M\\
 \text{subject to }&
 {\cal Z}^*(I_\omega\otimes M){\cal G}
 ={\cal Z}^*{\cal G}.
\end{aligned}} \tag{100.13}
\]

Equation (100.13), not the fact that a frozen endpoint emits one number, is
the exact portfolio closure rank.  It also distinguishes a finite list of
matched scalar actions from a uniform replacement theorem.

For the stronger uniform coefficient-space guarantee, let (X,Y\subseteq
\mathscr H_3) be the complete-old residual source and query spans, with
orthonormal bases (U_X,U_Y).  Exactness is

\[
 U_X^T(M-I)U_Y=0, \tag{100.14}
\]

so

\[
 \boxed{k\ge r_\times:=\operatorname{rank}(U_X^TU_Y).} \tag{100.15}
\]

This is tight for an arbitrary endpoint-tuned factorization but not
necessarily for one-sparse CountSketch.  Every rank-(k) Gram satisfies

\[
 \boxed{\|U_X^TMU_Y-U_X^TU_Y\|_{\rm op}
 \ge\sigma_{k+1}(U_X^TU_Y).} \tag{100.16}
\]

### 100.3 Exact CountSketch law for the whole portfolio

Fix one bucket map and four-wise independent signs before looking at value.
For fixed (x,y\in\mathbb R^p),

\[
 E_S\langle Sx,Sy\rangle=\langle x,y\rangle, \tag{100.17}
\]

\[
\boxed{\begin{aligned}
 \operatorname{Var}_S\langle Sx,Sy\rangle
 ={1\over k}\bigg(&\|x\|^2\|y\|^2+\langle x,y\rangle^2
 -2\sum_cx_c^2y_c^2\bigg)
 \le{2\over k}\|x\|^2\|y\|^2.
\end{aligned}} \tag{100.18}
\]

For corpus response row (e), all layers and units occur inside

\[
 A_e:=\sum_{\ell,a}z_{e\ell a}g_{e\ell a}^T,qquad
 f_e=\operatorname{tr}A_e. \tag{100.19}
\]

Then

\[
 \widehat f_e-f_e=\operatorname{tr}[(S^TS-I)A_e] \tag{100.20}
\]

and the exact common-seed covariance is

\[
\boxed{
 \operatorname{Cov}_S(\widehat f_e-f_e,\widehat f_d-f_d)
 ={2\over k}\langle\operatorname{offsym}A_e,
                    \operatorname{offsym}A_d\rangle_F.} \tag{100.21}
\]

In particular,

\[
 \operatorname{Var}_S(\widehat f_e-f_e)
 ={1\over k}\sum_{c<d}(A_{e,cd}+A_{e,dc})^2. \tag{100.22}
\]

For the eight-net panel, stack response rows with declared metric (V).  If
(\Gamma_{ed}=\langle\operatorname{offsym}A_e,
\operatorname{offsym}A_d\rangle_F),

\[
 \boxed{E_S\|\widehat f-f\|_V^2={2\over k}\operatorname{tr}(V\Gamma).}
 \tag{100.23}
\]

The panel does not multiply deployed per-net arithmetic, but it enlarges the
frozen-seed certificate.  A shared seed makes layer and net errors coherent;
dividing the variance by ten or eight is invalid.  The preregistered random-
seed tail is only

\[
 \Pr_S\{\|\widehat f-f\|_V>\tau\}
 \le {2\operatorname{tr}(V\Gamma)\over k\tau^2}. \tag{100.24}
\]

For a frozen seed define

\[
 \epsilon_\times(S):=\|U_X^T(S^TS-I)U_Y\|_{\rm op}. \tag{100.25}
\]

If ({\cal A}_V) is the endpoint-weighted source/query product envelope,

\[
 \boxed{\|\widehat f-f\|_V\le\epsilon_\times(S){\cal A}_V,qquad
 \epsilon_\times(S)\ge\sigma_{k+1}(U_X^TU_Y).} \tag{100.26}
\]

For the documented (q_3>0.9736) requirement, even the best tuned rank-
(k) response must separately satisfy

\[
 {\sum_{j>k}\sigma_j^2\over\sum_j\sigma_j^2}<0.052103. \tag{100.27}
\]

The complete deterministic endpoint receipt is

\[
\boxed{
 \epsilon_\times{\cal A}_V+\eta_{\rm surface}+\eta_{\rm source}
 +\eta_{>3}+{\epsilon^2\over2}B_2(\epsilon)
 \le\tau_{\rm endpoint}.} \tag{100.28}
\]

For a pure (Z) source (eta_{>3}=0).  If the quoted raw certificate is
used, (\tau_{\rm endpoint}=3.80\times10^{-6}); otherwise the newly declared
tolerance replaces it.  Passing (100.28) still needs (91.30)--(91.31).

### 100.4 Lowest Cap-49 arithmetic and state

One explicit H3 coordinate/hash pass costs

\[
 {p\over n^2}={29260\over1048576}=0.0279045 \tag{100.29}
\]

width-GEMM equivalents per layer.  Let (a) be the number of fresh
rectangular width mixes per sketch channel and (b) the number of full
source/query coordinate passes.  Before local surface and hook work,

\[
 \boxed{10\left({ak\over1024}+b{29260\over1024^2}+g_{\rm node}\right)
       +C_{\rm hook}\le1.} \tag{100.30}
\]

At (g_{\rm node}=C_{\rm hook}=0), the integer ceilings are

\[
\begin{array}{c|ccc}
 &b=0&b=1&b=2\\ \hline
 a=2&51&36&22\\
 a=1&102&73&45.
\end{array} \tag{100.31}
\]

The conservative Section 97 path has two complete pullback mixes and both a
new source and query projection, hence (k\le22) before the 120-family
surface construction.  Literal resident equality can move it to another
cell; common weights, a Jordan alphabet, or scratch aliasing cannot.  With
(a=0), coordinate work alone costs (0.2790b) over ten layers and (k) is
state-limited, but no zero-mix identity is proved.

An arbitrary dense tuned projection costs about (0.2986k) equivalents over
ten layers with one dense projection plus two mixes, or (0.5776k) with
both source and query projections.  Cap-49 therefore permits only (k\le3)
or (k\le1), respectively, before hooks.

One f32 (n\times k) bank costs

\[
 4nk=4096k\ {\rm bytes}, \tag{100.32}
\]

and two banks cost (8192k).  Thus (k=22) needs (90{,}112) bytes for
one bank or (180{,}224) for two; (k=51) needs (208{,}896) or
(417{,}792).  Reset-local contraction lets all eight validation nets reuse
the buffers.  One explicit f32 (n\times p) H3 bank instead costs

\[
 4np=119{,}848{,}960\ {\rm bytes} \tag{100.33}
\]

per live seam.  State is cheap after sketching; formation and fidelity are
the generic gates.

### 100.5 Can one sketch serve eight nets and ten layers?

Only if the combined old-residualized condition (100.13), or the stronger
uniform condition (100.14), passes for the same seed.  Individual per-net
passes do not imply the union pass.

There is a sharp counterexample inside the star alphabet.  Let
(V\subseteq\Pi_{\rm old}^{\perp}\mathscr H_3) have dimension (d=k+1),
and let both admissible source and query spans contain (V).  Symmetric cubes
realize such a (V) because they span (\mathscr H_3).  Then

\[
 U_X^TU_Y=I_d,qquad \sigma_{k+1}=1. \tag{100.34}
\]

Every rank-(k) sketch has (0\ne v\in V\cap\ker S).  Its exact matched
action is (|v|^2), while its sketched action is zero.  The example may be
placed in one net/layer or distributed over the eight-by-ten portfolio;
old residualization helps only if it reduces the actual quotient dimension
to at most (k).

Therefore no symbolic theorem makes one common Cap-49 sketch sufficient.
Under the conservative cell of (100.31), reopening needs

\[
 \boxed{r_\times\le22,qquad
 \epsilon_\times(S){\cal A}_V+\text{tails}\le\tau_{\rm endpoint}.}
 \tag{100.35}
\]

Even if (r_\times\le k), compressed one-sparse CountSketch is not
generically exact: (S^TS-I) has forced collision off-diagonals, and
(100.14) is a proper algebraic condition on the endpoint subspaces.
Unbiasedness is only (100.17).

Graph-ready split:

- `Q211_H3_STAR_JORDAN_ADJOINT` -- **solved exact**, (100.3)--(100.8);
- `Q211_H3_STRUCTURAL_NAME_RANK` -- **closed no-go**; exact rank is (100.9);
- `Q211_H3_BILINEAR_RANK` -- **solved exact**, (100.11)--(100.12);
- `Q211_H3_COMMON_SKETCH_RANK` -- **solved characterization**, (100.13);
- `Q211_H3_UNIFORM_CROSS_RANK` -- **solved lower bound/tail**,
  (100.14)--(100.16);
- `Q211_H3_COUNTSKETCH_VARIANCE` -- **solved exact**, (100.17)--(100.23);
- `Q211_H3_FROZEN_SEED_GATE` -- **conditional**, (100.25)--(100.28);
- `Q211_H3_CAP49` -- **generic conservative ceiling (k\le22)** before
  node/hook work, (100.30)--(100.31);
- `Q211_H3_STATE` -- **state-feasible reset-locally**, (100.32)--(100.33);
- `Q211_H3_8NET_COMMON_SKETCH` -- **no universal guarantee**, counterexample
  (100.34); reopen only by (100.35);
- `Q211_H3_ENDPOINT_VALUE` -- **open** after deterministic tails and
  complete-old Schur projection.

## 103. Latent-mean CP3 adjoint for quadratic-AQ q211: exact seeded-VJP
factorization and the one-VJP obstruction

Section 93 proves that quadratic-AQ q211 is the second latent-mean jet of
CP3.  Reverse automatic differentiation can consume that jet without
materializing a q211 tensor, but it does not make the neuron-dependent
anchor coefficients disappear.  The exact number of differently seeded
CP3 jet actions is a rank of the anchor coefficient banks.

Use the Section 93 convention

\[
 g_i(z)=d_i+u_i^Tz+z^TA_iz-\operatorname{tr}A_i,\qquad
 Y_i=(g_i)_+,\qquad z\sim N(0,I_r),
 \tag{103.1}
\]

and let \(K_{ijk}(h)\) be the connected CP3 tensor under
\(z\sim N(h,I_r)\).  For physical coefficient vectors \(a,b,c\), define
the strict scalar CP3 action

\[
 {\cal K}(a,b,c;h)
 :=\sum_{i,j,k}^{\rm strict}a_i b_j c_kK_{ijk}(h).
 \tag{103.2}
\]

The \(j,k\) symmetrization of a partially symmetric endpoint is understood;
it changes none of the anchor-rank statements below.

### 103.1 Exact seeded-VJP factorization for one endpoint atom

Let one ordered q211 endpoint atom be

\[
 L_{i;jk}=\alpha_i\beta_j\gamma_k.
 \tag{103.3}
\]

The genuinely three-unit part of its quadratic-AQ action is, by (93.8),

\[
\begin{aligned}
 S_L^{(3)}={}&
 \sum_{i,j,k}^{\rm strict}\alpha_i\beta_j\gamma_k
 \big[
 (d_i-2m_i)K_{ijk}
 +u_i^T\nabla_hK_{ijk}
 +\langle A_i,\nabla_h^2K_{ijk}\rangle_F
 \big]_{h=0}.
\end{aligned}
 \tag{103.4}
\]

The pair/mean jet complement \(S_{L,\le2}\) remains separate.

Put \(U_{i,:}=u_i^T\), and factor the first anchor bank at its exact matrix
rank:

\[
 D_\alpha U=P_1V_1^T,\qquad
 p_1:=\operatorname{rank}(D_\alpha U),
 \tag{103.5}
\]

where the columns of \(P_1\) are \(p_s\) and those of \(V_1\) are
\(v_s\).  Equivalently,

\[
 \alpha_i u_i=\sum_{s=1}^{p_1}(p_s)_iv_s.
 \tag{103.6}
\]

Then the complete first-jet action is exactly

\[
 \boxed{
 S_L^{(1)}
 =\sum_{s=1}^{p_1}
 v_s^T\nabla_h{\cal K}(p_s,\beta,\gamma;0).}
 \tag{103.7}
\]

For the second jet, let

\[
 {\cal A}_{i,:}=\operatorname{vec}_{\rm sym}(A_i)^T,
 \qquad
 D_\alpha{\cal A}=P_2{\cal B}^T,
 \qquad
 p_2:=\operatorname{rank}(D_\alpha{\cal A}),
 \tag{103.8}
\]

and reshape column \(s\) of \({\cal B}\) into a symmetric matrix \(B_s\).
Thus

\[
 \alpha_iA_i=\sum_{s=1}^{p_2}(P_2)_{is}B_s.
 \tag{103.9}
\]

The exact Hessian action is

\[
 \boxed{
 S_L^{(2)}
 =\sum_{s=1}^{p_2}
 \left\langle B_s,
 \nabla_h^2{\cal K}(P_{2,:,s},\beta,\gamma;0)
 \right\rangle_F.}
 \tag{103.10}
\]

Finally, with

\[
 a_0=\alpha\odot(d-2m),
 \tag{103.11}
\]

the entire three-unit action is the tensor-free identity

\[
 \boxed{
 S_L^{(3)}
 ={\cal K}(a_0,\beta,\gamma;0)
 +S_L^{(1)}+S_L^{(2)}.}
 \tag{103.12}
\]

Equations (103.7), (103.10), and (103.12) are the exact reverse/JVP
implementation theorem.  Every term is a scalar CP3 action or a
directional derivative of one; no CP4 or q211 forward tensor is formed.

### 103.2 The ranks are necessary, not merely one construction

Consider implementations whose primitive is a scalar CP3 action
\({\cal K}(a,\beta,\gamma;h)\), with a common latent direction attached to
each first derivative.  A sum of \(q\) such first-direction queries has an
anchor coefficient bank of matrix rank at most \(q\).  Therefore

\[
 \boxed{
 q_{\rm first}\ge\operatorname{rank}(D_\alpha U)=p_1,}
 \tag{103.13}
\]

and (103.7) attains equality.  This is necessary and sufficient within the
scalar common-tilt interface.

Likewise, if one Hessian primitive accepts an arbitrary symmetric matrix
seed \(B\), a sum of \(q\) differently anchor-seeded traces has
anchor-versus-\(\operatorname{Sym}^2(\mathbb R^r)\) flattening rank at most
\(q\).  Hence

\[
 \boxed{
 q_{\rm Hessian}\ge
 \operatorname{rank}(D_\alpha{\cal A})=p_2,}
 \tag{103.14}
\]

and (103.10) attains equality.

If the circuit exposes only ordinary Hessian-vector products, define the
partially symmetric seeded rank

\[
 \rho_2:=
 \min\left\{q:
 \alpha_i(A_i)_{\mu\nu}
 =\sum_{s=1}^q
 q_{s,i}\lambda_s(v_s)_\mu(v_s)_\nu
 \right\}.
 \tag{103.15}
\]

Then exactly \(\rho_2\) signed directional second derivatives suffice, and
fewer are impossible in that primitive.  Allowing mixed derivatives
\(\partial_{v_s}\partial_{w_s}\) replaces (103.15) by the corresponding
partially symmetric Jordan rank.  A dense matrix seed is not one
Hessian-vector product merely because reverse AD is used.

For nonzero generic \(\alpha\), generic quadratic-AQ parameters at
\(n=1024,r=55\) give

\[
 \boxed{
 p_1=55,\qquad
 p_2=\min\left\{1024,{55\cdot56\over2}\right\}=1024.}
 \tag{103.16}
\]

Thus a generic endpoint needs at least one value seed, 55 first-jet seeds,
and 1024 arbitrary-matrix Hessian seeds.  Reverse AD makes each seed
tensor-free; it does not collapse 1080 independent seeded outputs to one.

The same obstruction can be stated without an AD model.  If a claimed
one-pass interface exposes rows \(E\) of the CP3 evaluator and \(J_L\) is
the desired combination in (103.12), literal ownership is exactly

\[
 \boxed{
 \ker E\subseteq\ker J_L
 \quad\Longleftrightarrow\quad
 J_L=B_LE\quad\text{for some }B_L.}
 \tag{103.17}
\]

A stored CP3 value row supplies the zeroth jet only and fails (103.17)
generically.

### 103.3 Exact common-dictionary reopen

There is a strong positive structured class.  Suppose, at the synchronized
reset,

\[
 u_i=\sum_{a=1}^{k_1}P_{ia}v_a,\qquad
 A_i=\sum_{a=1}^{k_2}Q_{ia}B_a,
 \tag{103.18}
\]

with the direction vectors \(v_a\), matrix seeds \(B_a\), and their CP3
JVP/Hessian-trace rows literally resident.  Then for every endpoint atom

\[
 p_1\le k_1,\qquad p_2\le k_2.
 \tag{103.19}
\]

The physical anchor seeds are simply
\(D_\alpha P_{:,a}\) and \(D_\alpha Q_{:,a}\); no fitted tensor
decomposition is needed.  If each \(B_a\) has signed spectral rank \(s_a\),
the HVP-only count is at most

\[
 \rho_2\le\sum_{a=1}^{k_2}s_a.
 \tag{103.20}
\]

The extreme rank-one law
\(u_i=p_i v,\ A_i=q_iB\) needs one first-direction and one matrix-Hessian
seed per endpoint atom.  This is the exact class in which the slogan
"one CP3 VJP gives q211" is nearly true.

The dictionary must be shared by the *actual* CP3 evaluator.  A common
latent dimension, common eigenvalue count, or independently fitted CP
factorization does not establish (103.18).  Basis-motion and quadrature
derivatives must be included if the evaluator's factors depend on the
tilt.

### 103.4 Multi-atom and Cap49 gate

For \(R_c\) endpoint atoms, separate application of (103.12) is always
exact.  A smaller batched count is lawful only after jointly factoring the
seeded coefficient families

\[
 C^{(1)}_{\rho i\mu}
 =\alpha_i^{(\rho)}(u_i)_\mu,\qquad
 C^{(2)}_{\rho i\mu\nu}
 =\alpha_i^{(\rho)}(A_i)_{\mu\nu},
 \tag{103.21}
\]

while keeping the corresponding
\((\beta^{(\rho)},\gamma^{(\rho)})\) leaf seeds tied to \(\rho\).
Factoring the anchor bank after dropping that endpoint label is an invalid
reuse.

Let \(c_{0,\ell}\), \(c_{1,\ell}\), and \(c_{2,\ell}\) be the audited
incremental costs of one value, first-direction, and matrix-Hessian scalar
CP3 seed at seam \(\ell\), after literal backbone reuse.  Let
\(\widehat p_{1,\ell},\widehat p_{2,\ell}\) be the exact joint seeded ranks
of (103.21) at that seam.  The literal Cap49 gate is

\[
 \boxed{
 C_{\rm backbone}+C_{\le2}+C_{\rm coll}
 +\sum_{\ell=1}^{D}
 \left[
 c_{0,\ell}R_{0,\ell}
 +c_{1,\ell}\widehat p_{1,\ell}
 +c_{2,\ell}\widehat p_{2,\ell}
 \right]\le49.}
 \tag{103.22}
\]

For an HVP-only circuit replace
\(c_{2,\ell}\widehat p_{2,\ell}\) by the exact signed/Jordan directional
count and cost.  \(R_{0,\ell}\) is the number of distinct ordinary value
seeds after leaf reuse.  Baur--Strassen/Pearlmutter constant-factor
statements justify finite \(c_1,c_2\) for one scalar seed; they do not alter
the ranks in (103.13)--(103.16).

No symbolic generic parameters make (103.22) pass: the current circuit's
literal per-seed costs and the deployed common-dictionary ranks are needed.
Equation (103.16) is already a fail-fast warning that a generic
quadratic-AQ bank cannot be rescued by calling the operation a VJP.

### 103.5 Ownership and value disposition

Complete-old projection must precede the ranks: first apply the literal
old-interface adjoint residualizer to the endpoint, then form and rank the
coefficient families (103.21) of that residualized endpoint.  Projecting a
factor bank after taking its rank is not equivalent.  If the full Section
49 AQ q211
surface is already in the comparator, (103.12) is an alternative
implementation and receives zero additive credit.  If only CP3 values are
present, the residual first/second jets can be novel but still need
(103.17), (103.22), and the independent fidelity certificate of the
underlying CP3 evaluator.

This route evaluates quadratic AQ only.  It supplies neither the dependent
true-minus-AQ q211 source nor a positive raw-MSE gain.  Endpoint value is
the usual complete-old Schur gain after the jet feature is fixed.

Graph-ready split:

- AQ_Q211_CP3_SEEDED_VJP -- **solved exact**, (103.4)--(103.12);
- AQ_Q211_CP3_FIRST_JET_RANK -- **necessary and sufficient**,
  (103.13);
- AQ_Q211_CP3_SECOND_JET_RANK -- **necessary and sufficient for a
  matrix-seed primitive**, (103.14);
- AQ_Q211_CP3_HVP_RANK -- **exact primitive-dependent rank**, (103.15);
- AQ_Q211_ONE_VJP_GENERIC -- **NO-GO**, (103.16);
- AQ_Q211_CP3_LITERAL_JET_OWNER -- **open literal row receipt**, (103.17);
- AQ_Q211_CP3_COMMON_DICTIONARY -- **conditional GO**,
  (103.18)--(103.20);
- AQ_Q211_CP3_MULTIATOM_SEED_RANK -- **open deployed factor receipt**,
  (103.21);
- AQ_Q211_CP3_CAP49 -- **exact conditional gate**, (103.22);
- AQ_Q211_CP3_WITH_FULL_SURFACE -- **replacement/zero double credit**;
- AQ_Q211_CP3_TRUE_MINUS_AQ -- **not supplied**.

## 102. Exact Dyson--Volterra resummation of a q211 transition defect:
the target cyclic-module theorem

Section 101 shows why the baseline Bellman scalar is generally only a first
variation of a finite transition repair.  The special escape is not
"one scalar output" by itself.  It is exact closure of every future hybrid
pullback of that output inside a small, locally evaluable observable module.
This section gives the necessary-and-sufficient condition and the exact
resummation.  The generic state derivatives remain those of Section 101;
here they are reorganized inside the closed module.

### 102.1 Koopman spelling makes the nonlinear Dyson identity exact

Let the baseline and repaired state maps be

\[
 F_l:X_l\to X_{l+1},\qquad
 T_l(x)=F_l(x)+G_l(x),\qquad 0\le l<D, \tag{102.1}
\]

where \(G_l\) is the complete-old-residualized q211 transition defect, not a
same-site total q211 source.  For scalar observables on \(X_{l+1}\), define
the pullbacks

\[
 \mathsf F_l f=f\circ F_l,qquad
 \mathsf T_l f=f\circ T_l,qquad
 \mathsf D_l=\mathsf T_l-\mathsf F_l. \tag{102.2}
\]

The additive spelling in (102.1) assumes that each target state space is a
vector space, or an affine chart in which both maps remain lawful.  On a
general manifold the theorem starts directly from two maps \(F_l,T_l\); no
globally defined difference \(G_l=T_l-F_l\) is implied.  The Koopman
identity needs only the two maps.

The state maps may be fully nonlinear; the pullbacks in (102.2) are linear
operators on observables.  For terminal observable \(h\),

\[
 J_T(x_0)=\bigl(\mathsf T_0\cdots\mathsf T_{D-1}h\bigr)(x_0),
 \qquad
 J_F(x_0)=\bigl(\mathsf F_0\cdots\mathsf F_{D-1}h\bigr)(x_0). \tag{102.3}
\]

Expanding \(\mathsf T_l=\mathsf F_l+\mathsf D_l\) gives the exact finite
Dyson identity

\[
\boxed{
 J_T-J_F=
 \sum_{k=1}^{D}
 \sum_{0\le i_1<\cdots<i_k<D}
 \operatorname{ev}_{x_0}
 \bigl[
 \mathsf F_0\cdots\mathsf D_{i_1}\cdots
 \mathsf D_{i_k}\cdots\mathsf F_{D-1}h
 \bigr].}
\tag{102.4}
\]

At every unlisted layer the operator is \(\mathsf F_l\).  Equation (102.4)
is an identity between the two nonlinear networks, not a small-defect Taylor
series.  Every term has at least one \(\mathsf D\), so the baseline/zero-
length response is absent.  Positive-depth words already present in the old
arena must still be projected out once.

That means applying the complete-old quotient to the total word span (or to
the total scalar in (102.4)).  Deleting apparently old summands one by one
is equivalent only when the old projector intertwines every declared
pullback; otherwise it can destroy cancellations and change the cyclic
rank.

### 102.2 Minimal target-conditioned cyclic module

To avoid the tautological but useless choice "store the exact value
function," fix the lawful local scalar primitive class, the family of
network states/covariances on which deployment must work, and all hybrid
choices \(F_l\) versus \(T_l\) (or all declared defect strengths).  Here a
\(q\)-channel realization means a **linear observable realization**: it
stores the values of \(q\) fixed scalar observables and advances their
coefficient vector by source-independent linear maps.  Arbitrary nonlinear
real-number encodings, or a scalar oracle allowed to reread the full state,
are not lawful channels.  Two
observables at layer \(l\) are equivalent when their difference annihilates
every lawful reachable state/source and every complete-old-residualized
endpoint test.  Work in this quotient.  The reachable/test family must be
closed under every declared \(F_l,T_l\).  Equivalently, the equivalence
relation must be a pullback congruence: equality of two classes at layer
\(l+1\) must imply equality after both \(\mathsf F_l\) and \(\mathsf T_l\)
at layer \(l\).  Without this condition (102.6) is not well defined on
equivalence classes.

Starting from

\[
 \mathcal M_D=\operatorname{span}\{[h]\}, \tag{102.5}
\]

define backwards

\[
\boxed{
 \mathcal M_l=\operatorname{span}
 \{\mathsf F_l\mathcal M_{l+1},
   \mathsf D_l\mathcal M_{l+1}\}.}
\tag{102.6}
\]

Equivalently, \(\mathcal M_l\) is the span of every legal future word in
the pullbacks \(\mathsf F,\mathsf D\) applied to the terminal target, modulo directions
which no lawful q211 birth/state can distinguish.  Put

\[
 \boxed{q_*:=\max_l\dim\mathcal M_l.} \tag{102.7}
\]

Then:

\[
\boxed{
 \begin{array}{c}
 \text{The complete nonlinear endpoint repair has an exact}\\
 \text{target-conditioned linear-observable realization with at most }q
 \text{ scalar channels}\\
 \text{for every lawful hybrid sequence}
 \end{array}
 \iff q_*\le q.}
\tag{102.8}
\]

provided the basis observables and their local q211 defect actions are
literally evaluable by the declared primitives.

The quantifier over every lawful hybrid sequence and the fixed linear-
observable channel model are essential.  For only one repaired sequence, or
for a nonlinear encoder allowed to depend on the queried state, \(q_*\) is
an upper bound but need not be a necessary state dimension.

Consequently, a family of widths, depths, and admissible nets has an
\(O(1)\)-channel direct realization precisely when

\[
 \boxed{\sup_{n,D,\theta}q_*(n,D,\theta)<\infty} \tag{102.8a}
\]

and the corresponding basis actions have uniformly bounded primitive
acquisition cost.  Finite depth alone is not such a result: it gives only
finitely many words (at worst exponentially many), not a depth-independent
module.

For sufficiency, choose basis rows
\(\Psi_l=(\psi_{l1},\ldots,\psi_{lq_l})\).  There are exact small matrices

\[
 \mathsf F_l\Psi_{l+1}=\Psi_l A_l,
 \qquad
 \mathsf D_l\Psi_{l+1}=\Psi_l B_l. \tag{102.9}
\]

Thus the full and baseline coefficient recurrences are respectively

\[
 c_l=(A_l+B_l)c_{l+1},
 \qquad c_l^{(0)}=A_lc_{l+1}^{(0)}, \tag{102.10}
\]

and the exact endpoint difference is obtained from the two \(q_l\)-channel
recurrences and local evaluations of \(\Psi_0(x_0)\).  No forward q211 tensor
is stored.  Necessity follows because the coordinate functions of any exact
\(q\)-channel realization form an invariant observable space containing all
hybrid pullbacks in (102.6), under precisely the fixed-observable/all-hybrid
definition above.

The equivalent realization-theory statement is useful in practice.  Let
\(H_l\) be the layered Hankel matrix whose rows are legal future hybrid words,
whose columns are lawful reachable states/local q211 sources, and whose entry
is their terminal scalar pairing.  Then

\[
\boxed{\operatorname{rank}H_l=\dim\mathcal M_l,\qquad
 q_*=\max_l\operatorname{rank}H_l.} \tag{102.11}
\]

The column family in this statement must separate the declared observable
quotient.  A sampled or otherwise nonseparating column panel has Hankel rank
only as a lower bound on \(\dim\mathcal M_l\), not as an equal quantity.

One direct scalar is exact iff every future-visible defect word is collinear
in this quotient.  Low rank of one \(G_l\), one source tensor, or one endpoint
ray is not sufficient: interposed dense resets can enlarge the word/Hankel
rank at the next layer.

### 102.3 Exact Volterra resolvent on the closed module

The same result has a causal return-kernel form.  Let \(b_j\in\mathbb R^{q_j}\)
be the direct reduced q211 birth/action at layer \(j\), and let
\(K_{ji}:\mathbb R^{q_i}\to\mathbb R^{q_j}\), \(i<j\), be its exact reduced
baseline transport followed by the next defect action.  The accumulated
reduced amplitudes obey

\[
 a_j=b_j+\sum_{i<j}K_{ji}a_i. \tag{102.12}
\]

On \(\bigoplus_j\mathbb R^{q_j}\), the block operator \(K\) is strictly
lower triangular.  Hence \(K^D=0\), and

\[
\boxed{
 a=(I-K)^{-1}b
 =\sum_{m=0}^{D-1}K^mb,qquad
 \Delta J=\ell^Ta.}
\tag{102.13}
\]

This is the exact discrete Volterra/Dyson resummation of every separated
return; finite depth needs no convergence assumption.  It is an alternative
coordinate system for the same paths as (102.4), never an additive source.

If same-layer feedback, an infinite horizon, or an analytically eliminated
sector makes \(K\) non-nilpotent, the algebraic resolvent exists iff

\[
 1\notin\operatorname{spec}K. \tag{102.14}
\]

For an infinite-dimensional lifted horizon this statement assumes that
\(K\) is a bounded operator on the declared Banach sequence space.  The
spectrum and convergence mode must be taken in that same space; pointwise
or strong convergence on one birth is weaker than operator-norm convergence
of the Neumann series.

The infinite Neumann/Dyson series converges on the minimal reachable-
observable quotient iff

\[
 \boxed{\rho(K)<1.} \tag{102.15}
\]

An unstable eigenmode which is unreachable from \(b\) or annihilated by
\(\ell\) is removed before applying (102.15).  If \(q_R\) is the dimension
of this minimal reachable-observable **resolvent** module, Cayley--Hamilton
reduces every analytic function of \(K\) to at most \(q_R\) powers.  For a shared
stationary, Markovian module with no unresolved delay state, \(q_R\le q_*\);
stationarity of a general Volterra kernel alone does not imply that bound.
For the layer-lifted nonstationary block
system only the safe bound \(q_R\le\sum_l q_l\) is automatic.

### 102.4 First and second insertions inside the module

For comparison with Section 101, take a Koopman homotopy

\[
 \mathsf T_l(\epsilon)
 =\mathsf F_l+\epsilon\mathsf D_l^{(1)}
 +{\epsilon^2\over2}\mathsf D_l^{(2)}+O(\epsilon^3). \tag{102.16}
\]

The expansion is in an operator topology strong enough for the displayed
terminal observables (for example, uniform (C^2) control on the reachable
set).  Pointwise differentiability of the state maps alone does not justify
the operator remainder in (102.16).

For the literal state-map homotopy (T_l(\epsilon)=F_l+\epsilon G_l), the
normalization in (102.16) is

\[
 (\mathsf D_l^{(1)}f)(x)
 =Df(F_l(x))[G_l(x)],\qquad
 (\mathsf D_l^{(2)}f)(x)
 =D^2f(F_l(x))[G_l(x),G_l(x)]. \tag{102.16a}
\]

If the state-map homotopy itself has nonzero second derivative, the second
formula acquires the additional term
(Df(F_l(x))[\partial_\epsilon^2T_l(0,x)]).  This fixes both the factor of
two and the scope of the phrase "same-layer nonlinear Koopman jet."

Let the baseline prefix evaluation functional and suffix observable be

\[
 \nu_i=\operatorname{ev}_{x_0}\mathsf F_0\cdots\mathsf F_{i-1},
 \qquad
 \lambda_{i+1}=\mathsf F_{i+1}\cdots\mathsf F_{D-1}h. \tag{102.17}
\]

Then

\[
\boxed{
 J'(0)=\sum_i\nu_i(\mathsf D_i^{(1)}\lambda_{i+1}),}
\tag{102.18}
\]

and

\[
\boxed{\begin{aligned}
 J''(0)={}&\sum_i\nu_i(\mathsf D_i^{(2)}\lambda_{i+1})\\
 &+2\sum_{i<j}\nu_i\!\left(
 \mathsf D_i^{(1)}
 \mathsf F_{i+1}\cdots\mathsf F_{j-1}
 \mathsf D_j^{(1)}\lambda_{j+1}
 \right).
\end{aligned}}
\tag{102.19}
\]

The first line of (102.19) is the same-layer nonlinear Koopman jet; the
second is the ordered two-defect/co-adaptation return.  The state-Hessian and
\(DG_l\) pieces of Section 101.7 are exactly the coordinate expansion of
these two terms.  When the additive *Koopman* defect is affine in
\(\epsilon\), \(\mathsf D^{(2)}=0\), but a state-map homotopy
\(F_l+\epsilon G_l\) generally has nonzero higher Koopman jets.

Equations (102.18)--(102.19) require only the small matrices representing
the displayed operators when the cyclic module is also closed under
\(\mathsf D_l^{(1)}\) and \(\mathsf D_l^{(2)}\).  Closure under the single
finite operator \(\mathsf D_l=\mathsf T_l-\mathsf F_l\) at one defect
strength does not imply this derivative closure.  It follows from (102.8)
only when the declared hybrid family includes a neighborhood of homotopy
strengths and the finite-dimensional module is closed in the chosen
operator topology.  Under that added hypothesis they do not require an H4/q211
state, and they expose why a single baseline scalar fails when the module has
more than one visible insertion word.

### 102.5 Sharp truncation and conditioning gates

For the reduced Volterra resolvent, truncating after \(p\) returns has the
exact target remainder

\[
\boxed{
 R_{>p}=\ell^TK^{p+1}(I-K)^{-1}b.}
\tag{102.20}
\]

Therefore

\[
 |R_{>p}|
 \le\|\ell^TK^{p+1}\|\,
       \|(I-K)^{-1}b\|
 \le\|\ell\|\|K^{p+1}\|
       \|(I-K)^{-1}\|\|b\|. \tag{102.21}
\]

The first expression is the sharp target-conditioned gate; the second is
only a safe norm fallback.  If \(\|K\|\le r_K<1\), it reduces to

\[
 |R_{>p}|\le
 \|\ell\|\|b\|{r_K^{p+1}\over1-r_K}. \tag{102.22}
\]

Nonnormality makes an eigenvalue gap alone insufficient.  The binding
absolute resolvent amplification (not, by itself, the relative condition
number of \(I-K\)) is

\[
 \kappa_R=\|(I-K)^{-1}\| \tag{102.23}
\]

on the minimal reachable-observable module.  If the implemented kernel and
birth satisfy

\[
 \|\widehat K-K\|\le\eta_K,qquad
 \|\widehat b-b\|\le\eta_b,qquad
 \kappa_R\eta_K<1, \tag{102.24}
\]

the resolvent identity gives

\[
\boxed{
 \|\widehat a-a\|
 \le {\kappa_R\over1-\kappa_R\eta_K}
 \{\eta_b+\kappa_R\eta_K\|b\|\}.}
\tag{102.25}
\]

This comparison requires \(K,\widehat K\) and \(b,\widehat b\) to be
identified on the same normed reduced space.  If the approximate module has
a different basis or dimension, an explicit lift/intertwiner and its error
must be included before (102.24) can be invoked.  Likewise an approximate
terminal row contributes the separate bound
\(\|\widehat\ell-\ell\|\,\|\widehat a\|\); it is absent from (102.25)
only when the endpoint evaluation is literal.

For an approximate cyclic module, let \(\widehat{\mathsf T}_l\) be its lifted
reduced transition.  The exact product telescope is

\[
 \mathsf T_0\cdots\mathsf T_{D-1}
 -\widehat{\mathsf T}_0\cdots\widehat{\mathsf T}_{D-1}
 =\sum_j
 \mathsf T_0\cdots\mathsf T_{j-1}
 (\mathsf T_j-\widehat{\mathsf T}_j)
 \widehat{\mathsf T}_{j+1}\cdots\widehat{\mathsf T}_{D-1}. \tag{102.26}
\]

Applying the actual endpoint evaluation to (102.26) is the sharp signed
module-tail feature.  The submultiplicative bound follows immediately but
must not be squared and RSS-summed without an orthogonality theorem.  After
complete-old corpus projection, the total of (102.20), (102.25), and
(102.26) must satisfy the Section 101 fidelity/value gate.  In particular,
a physics-repair claim capable of raw \(10^{-8}\) needs at most the optimistic
10.35% relative endpoint error from (101.18); a refitted-feature claim instead
needs held-out old-residualized correlation at least 0.8124.

That total is lawful only after the truncation, kernel perturbation, and
module-lift errors have been defined as nonoverlapping increments.  If
(102.25) and (102.26) bound the same transition discrepancy, adding them
double counts it; use the single exact telescope (102.26), or an explicitly
disjoint error decomposition, instead.

### 102.6 Timing and Cap49 implications

Small cyclic rank is an information theorem, not a free-product theorem.
Promotion additionally requires:

1. every local scalar
   \(\{\mathsf D_l\psi_{l+1,a}\}(x_l)\) to be obtainable from resident AQ,
   CP3, collision, and old-Star data without forming the q211 tensor;
2. the reduced matrices \(A_l,B_l\) to be available before their actions are
   needed, either from a compiled future context, a lawful two-pass schedule,
   or a resident Bellman view; and
3. the complete-old quotient to be applied before measuring \(q_*\), so
   neither zero-length nor already-arena positive-depth paths are credited.

If a live reduced channel is an \(n\)-vector column transported through a
dense width-\(n\) map, or an equivalent \(O(n^2)\) local reduction, let
\(r_l\) be the number of such channels actually transported at active seam
\(l\).  The optimistic necessary bill is

\[
 \boxed{C_{vec}\gtrsim {1\over n}
       \sum_{l\in\mathcal S}r_l.} \tag{102.27}
\]

Only under the additional uniform-live assumption \(r_l=q_*\) at all
\(D_s\) seams does this become \(D_sq_*/n\).  Thus, with \(n=1024\), ten
seams, and one remaining equivalent, \(q_*\le102\) is a valid uniform-live
screen, not a consequence of the maximum in (102.7) alone.  A module that
attains \(q_*\) at only one seam can have a much smaller bill.  Truly global
scalar reductions can be cheaper.

Conversely, one generic \(n\times n\) conjugation per live plane channel
costs

\[
 \boxed{C_{plane}\gtrsim2\sum_{l\in\mathcal S}r_l,} \tag{102.28}
\]

reducing to \(2D_sq_*\) only in the same uniform-live case.  Since even one
fresh generic conjugation costs about two equivalents, any such nonresident
plane channel already exceeds one-equivalent additive headroom.  Calling
the final state "one scalar" does not erase the reverse or source product
used to acquire it.

The exact route is therefore:

- **GO:** a literal primitive-evaluable module satisfying (102.8), stable
  reduced resummation satisfying (102.20)--(102.26), lawful future timing,
  total cost within the residual cap, and the Section 101 raw-value gate;
- **HOLD:** a low module is suggested but its local q211 actions, old quotient,
  timing, or conditioning are not literal;
- **NO-GO:** the layered Hankel/cyclic rank grows with width/depth beyond the
  cost screen, or even one required channel needs a fresh generic plane
  conjugation at every seam.

Graph-ready split:

- `Q211_KOOPMAN_DYSON_IDENTITY` -- **solved exact nonlinear identity**,
  (102.1)--(102.4);
- `Q211_TARGET_CYCLIC_MODULE_IFF` -- **necessary and sufficient**,
  (102.5)--(102.11), for the pullback-congruent, separating,
  fixed-linear-observable/all-hybrid model only;
- `Q211_DIRECT_SCALAR_RANK1` -- **conditional only** when the complete
  hybrid Hankel rank is one, not merely the local source rank;
- `Q211_LOW_RANK_VOLTERRA_RESOLVENT` -- **solved exact**, (102.12)--(102.15);
- `Q211_DYSON_FIRST_SECOND_ACTIONS` -- **solved module specialization**,
  (102.16)--(102.19), conditional on first/second Koopman-jet closure, with
  generic state derivatives owned by Section 101;
- `Q211_VOLTERRA_TARGET_REMAINDER` -- **solved exact/conditional gate**,
  (102.20)--(102.22);
- `Q211_VOLTERRA_CONDITIONING` -- **solved perturbation gate**,
  (102.23)--(102.26);
- `Q211_DIRECT_CHANNEL_TIMING` -- **open literal implementation receipt**;
- `Q211_DIRECT_CHANNEL_CAP49` -- **conditional** under (102.27), generic
  per-channel plane conjugation closed by (102.28); (q_*\le102) is only
  the uniform-live ten-seam specialization, not a max-rank theorem;
- `Q211_DIRECT_RESUMMED_VALUE` -- **open** until the complete-old feature
  clears the Section 101 held-out correlation or physics-fidelity gate.

## 104. Zero-bias Gaussian q211 co-adaptation: exact covariance Hessian,
the second sandwich, and the raw-\(10^{-8}\) gate

Section 101 is architecture-independent.  This section evaluates its two
local second-action terms for the zero-bias standardized Gaussian closure
of Sections 88 and 96.  Work off the correlation boundary; all formulas
extend by one-sided limits when the stated surface moments are finite.

Let

\[
 x=R_{ij},\qquad y=R_{ik},\qquad z=R_{jk},
\]

and write a covariance/cumulant tangent as

\[
 h=\xi_R,\qquad u=\xi_{K_3},\qquad
 h_x=h_{ij},\ h_y=h_{ik},\ h_z=h_{jk}. \tag{104.1}
\]

Diagonal variances are held standardized.  If they are allowed to move,
compose the formulas below with the exact Jacobian/Hessian of
\(R_{ab}=C_{ab}/\sqrt{C_{aa}C_{bb}}\); that scale channel is not zero by
convention.

### 104.1 Exact \(DG_\ell\,\xi_\ell\)

Use the decomposition

\[
\boxed{
 G^\sharp_{i;jk}(R,T)
 =-2\mu\{T_{ijk}-T^{\rm old}_{ijk}(R)\}
   +E_{\rm mv}(x,y,z)+xyp^\sharp(z),} \tag{104.2}
\]

where \(E_{\rm mv}\) contains the declared matvec/edge and quartic terms,
and

\[
 p^\sharp(z)=p(z)-2\mu^4z,\qquad
 p(z)={1\over4}+{\arcsin z\over2\pi}. \tag{104.3}
\]

Equation (104.2) covers both readings of Sections 88 and 96.  With exact
CP3 coordinates, \(T=K_3\).  With the deployed tree owner,

\[
 T^{\rm tree}(x,y,z)={\mu\over4}(xy+xz+yz), \tag{104.4}
\]

so

\[
\begin{aligned}
 DT^{\rm tree}[h]={\mu\over4}\{&
 (y+z)h_x+(x+z)h_y+(x+y)h_z\},\\
 D^2T^{\rm tree}[h,h]={\mu\over2}(&
 h_xh_y+h_xh_z+h_yh_z).
\end{aligned} \tag{104.5}
\]

The broad scalar has the exact directional derivative

\[
\boxed{\begin{aligned}
 D[xyp^\sharp(z)][h]
 ={}&(yh_x+xh_y)p^\sharp(z)
 +xy\,p_1^\sharp(z)h_z,\\
 p_1^\sharp(z)
 ={}&{1\over2\pi\sqrt{1-z^2}}-2\mu^4 .
\end{aligned}} \tag{104.6}
\]

Consequently

\[
\boxed{\begin{aligned}
 DG^\sharp[h,u]
 ={}&-2\mu\{u-DT^{\rm old}[h]\}
 +DE_{\rm mv}[h]\\
 &+(yh_x+xh_y)p^\sharp(z)
 +xy\,p_1^\sharp(z)h_z .
\end{aligned}} \tag{104.7}
\]

If the leading CP3 triangle has already been folded into \(p^\sharp\), its
coordinate must be removed from \(u-DT^{\rm old}[h]\); otherwise (104.7)
double counts precisely the fusion in (96.9).

### 104.2 Exact \(D^2F_\ell[\xi_\ell,\xi_\ell]\) in Gaussian coordinates

Let

\[
 v_+=\operatorname{Var}(G_+)
 ={1\over2}-\mu^2={\pi-1\over2\pi},
\qquad
 \psi(r)={q(r)-\mu^2\over v_+}. \tag{104.8}
\]

The standardized zero-bias covariance reset is entrywise:

\[
\boxed{\begin{aligned}
 DF_R[h]&={p[R]\over v_+}\odot h,\\
 D^2F_R[h,h]
 &= {1\over2\pi v_+}
    {h^{\odot2}\over\sqrt{1-R^{\odot2}}}
\end{aligned}} \tag{104.9}
\]

off the diagonal, with the standardized diagonal fixed.  Thus the
covariance Hessian itself is Hadamard and opens no dense word.

The CP3 reset is \(T^+_{ijk}=K_3(x,y,z)\).  Price differentiation gives

\[
\begin{aligned}
 K_x&=E[I_1I_2Z_3],&
 K_y&=E[I_1Z_2I_3],&
 K_z&=E[Z_1I_2I_3], \tag{104.10}\\
 K_{xx}&=E[\delta_1\delta_2Z_3],&
 K_{yy}&=E[\delta_1Z_2\delta_3],&
 K_{zz}&=E[Z_1\delta_2\delta_3],\\
 K_{xy}&=E[\delta_1I_2I_3],&
 K_{xz}&=E[I_1\delta_2I_3],&
 K_{yz}&=E[I_1I_2\delta_3].
 \tag{104.11}
\end{aligned}
\]

Here \(Z_a=(G_a)_+-\mu\), \(I_a=\mathbf1_{G_a>0}\), and
\(\delta_a=\delta(G_a)\).  Therefore

\[
\boxed{\begin{aligned}
 D^2F_T[h,h]={}&K_{xx}h_x^2+K_{yy}h_y^2+K_{zz}h_z^2\\
 &+2K_{xy}h_xh_y+2K_{xz}h_xh_z+2K_{yz}h_yh_z .
\end{aligned}} \tag{104.12}
\]

These boundary moments are elementary functions of \((x,y,z)\) through
the Section 86 conditional correlations.  A determinant margin is required
for a uniform Hessian bound.

If q211 is kept as an independent \((R,T)\) coordinate through

\[
 F_Q(R,T)=-2\mu T+F_{\rm edge}(R), \tag{104.13}
\]

then its \(T\)-dependence is linear and

\[
 D^2F_Q[(h,u),(h,u)]=D^2F_{\rm edge}[h,h]. \tag{104.14}
\]

If instead the Gaussian constraint \(T=K_3(R)\) is substituted before
differentiation, the correct Hessian is

\[
\boxed{
 D^2F_Q[h,h]
 =-2\mu D^2K_3[h,h]+D^2F_{\rm edge}[h,h].} \tag{104.15}
\]

Equations (104.14) and (104.15) are two coordinate charts of the same
closure; mixing them is a double count.

For the broad term specifically,

\[
\boxed{\begin{aligned}
 D^2[xyp^\sharp(z)][h,h]
 ={}&2h_xh_y p^\sharp(z)
 +2(yh_x+xh_y)p_1^\sharp(z)h_z\\
 &+xy\,p_2^\sharp(z)h_z^2,\\
 p_2^\sharp(z)
 ={}&{z\over2\pi(1-z^2)^{3/2}} .
\end{aligned}} \tag{104.16}
\]

All remaining edge terms are differentiated entrywise or by their existing
matvec words.

Combining (104.7), (104.9), (104.12), and (104.14)/(104.15) gives the
literal local vector in (101.7):

\[
\boxed{
 {\cal C}_{2,\ell}
 =D^2F_\ell[\xi_\ell,\xi_\ell]
 +2DG_\ell\,\xi_\ell .} \tag{104.17}
\]

No fitted beta occurs in (104.17).

### 104.3 What the paid \(P^\sharp\) sandwich does and does not reuse

For a leaf-symmetric CP atom define

\[
 {\cal W}_{\beta\gamma}(P)
 ={1\over2}\{D_\beta P D_\gamma+D_\gamma P D_\beta\},
\qquad
 C_\alpha(R)=R D_\alpha R. \tag{104.18}
\]

The one broad product of (96.11) is equivalently

\[
\boxed{
 B^\sharp_{\alpha\beta\gamma}(R)
 =\langle{\cal W}_{\beta\gamma}(P^\sharp),C_\alpha(R)\rangle_F.}
 \tag{104.19}
\]

Let

\[
\begin{aligned}
 \dot P^\sharp&=p_1^\sharp[R]\odot h,&
 \ddot P^\sharp&=p_2^\sharp[R]\odot h^{\odot2},\\
 C_{\alpha,1}&=hD_\alpha R+RD_\alpha h
             =2\operatorname{Sym}(hD_\alpha R),&
 C_{\alpha,2}&=2hD_\alpha h .
\end{aligned} \tag{104.20}
\]

Then

\[
\boxed{\begin{aligned}
 DB^\sharp[h]
 ={}&\langle{\cal W}(P^\sharp),C_{\alpha,1}\rangle_F
 +\langle{\cal W}(\dot P^\sharp),C_\alpha\rangle_F,\\
 D^2B^\sharp[h,h]
 ={}&\langle{\cal W}(P^\sharp),C_{\alpha,2}\rangle_F\\
 &+2\langle{\cal W}(\dot P^\sharp),C_{\alpha,1}\rangle_F
 +\langle{\cal W}(\ddot P^\sharp),C_\alpha\rangle_F .
\end{aligned}} \tag{104.21}
\]

Thus the already paid \(C_\alpha=RD_\alpha R\) owns every term on the
right involving \(C_\alpha\), including the \(p'\) and \(p''\) changes.
It does **not** own the cross word

\[
 C_{\alpha,1}=2\operatorname{Sym}(hD_\alpha R), \tag{104.22}
\]

and, if the old q211 Hessian is active, it also does not own

\[
 C_{\alpha,2}=2hD_\alpha h. \tag{104.23}
\]

Each needs one new dense multiplication per independent anchor direction;
the second orientation in (104.22) is a transpose view because the leaf
weight is symmetric.

For an aligned atom, collect the dense part of the complete second action
as

\[
 S_{2,\rm dense}
 =\langle A,C_{\alpha,1}\rangle_F
  +\langle B,C_{\alpha,2}\rangle_F, \tag{104.24}
\]

where, before old projection,

\[
 A=2\{{\cal W}_G(P^\sharp)+{\cal W}_F(\dot P^\sharp)\},
\qquad B={\cal W}_F(P^\sharp).
\]

The subscripts record whether the term came from \(2DG\) or \(D^2F\);
different Bellman atoms are stacked before taking the rank.

Let \(E_{\rm paid}\) contain the literal primal sandwich and every resident
reverse row at that seam.  The complete second action uses no new sandwich
if and only if

\[
\boxed{
 \ker E_{\rm paid}\subseteq
 \ker\{(C_{\alpha,1},C_{\alpha,2})
       \mapsto\langle A,C_{\alpha,1}\rangle
                    +\langle B,C_{\alpha,2}\rangle\}.} \tag{104.25}
\]

This is the exact “same sandwich” criterion.  It fails generically because
\(h\) can change (104.22) while \(RD_\alpha R\) is fixed.

After quotienting by \(E_{\rm paid}\), one **additional** sandwich suffices
if and only if the two dense query rows in (104.24) have rank at most one:

\[
\boxed{
 \operatorname{rank}\{
 \Pi_{\rm paid}^{\perp}A,\Pi_{\rm paid}^{\perp}B\}\le1.} \tag{104.26}
\]

When both rows are nonzero, (104.26) is equivalently the existence of a
scalar \(\tau\) such that

\[
 2B=\tau A\quad\text{in the paid quotient}. \tag{104.27}
\]

Indeed one shifted product

\[
 (R+\tau h)D_\alpha(R+\tau h)
 =C_\alpha+\tau C_{\alpha,1}
   +{\tau^2\over2}C_{\alpha,2} \tag{104.28}
\]

then generates the required combination after rescaling and subtracting
the resident \(C_\alpha\).  Conversely, one symmetric shifted sandwich
has only this fixed cross/square ratio, proving necessity.  If the baseline
\(F\) has no broad q211 Hessian, \(B=0\): the second action needs exactly
the cross word (104.22), unless (104.25) passes.  If both \(A,B\ne0\),
their entrywise \(p^\sharp,p_1^\sharp\) weights are generically not
proportional, so two new rows are generic.

### 104.4 Depth and Cap-49 count

Let \(q_{2,\ell}\in\{0,1,2,\ldots\}\) be the quotient rank of the dense
rows (104.24) at layer \(\ell\), after literal reuse and the possible
one-shift fusion (104.26).  A full \(n\times n\) sandwich costs one
width-GEMM equivalent, hence the additive gate is

\[
\boxed{
 C_{\rm coad}=\sum_{\ell\in{\cal A}}q_{2,\ell}
 +C_{\rm surface}+C_{\rm hook}\le1.} \tag{104.29}
\]

Across ten active layers:

1. only \(DG\) broad and no reuse gives \(q_{2,\ell}=1\), hence cost \(10\);
2. \(DG\) plus an independent \(D^2F\) broad row gives cost \(20\);
3. the proportionality (104.26) reduces case 2 to \(10\), not to one;
4. same-sandwich ownership (104.25) is the only generic-depth zero-word
   escape.

Even the \(D_{\rm TOT}=2\) instrument costs two full equivalents for one
new row at both seams, above the one-equivalent additive slack.  A
rectangular/low-rank implementation can replace (104.29) by

\[
 \boxed{\sum_\ell r_{2,\ell}/1024
       +C_{\rm surface}+C_{\rm hook}\le1,} \tag{104.30}
\]

but then \(\sum_\ell r_{2,\ell}\le1024\) is a binding factor-rank receipt,
not a consequence of CP naming.  A replace-D10 package must separately
debit the removed D10 Schur value.

### 104.5 The \(10.35\%\) redemption-scale fidelity gate

Let \(t\) be the exact finite-repair corpus correction, \(f_1=J'(0)\) the
baseline scalar provider, and

\[
 f_2=J'(0)+{1\over2}\widehat J''(0)
\]

an implemented second-action provider.  Let \(P_\perp\) be the complete-old
response projector and let \(\eta_{\rm surf}\) include the Section 96
Hermite tail, determinant/surface error, and sandwich approximation.  From
(101.12),

\[
\boxed{\begin{aligned}
 {\|P_\perp(t-f_1)\|\over\|P_\perp t\|}
 &\le
 { {1\over2}\|P_\perp J''(0)\|+B_3/6+\eta_{\rm surf}
  \over\|P_\perp t\|},\\
 {\|P_\perp(t-f_2)\|\over\|P_\perp t\|}
 &\le
 { {1\over2}\|P_\perp(J''-\widehat J'')\|
       +B_3/6+\eta_{\rm surf}
  \over\|P_\perp t\|}.
\end{aligned}} \tag{104.31}
\]

For a true finite correction of target correlation \(\rho_t\), raw
\(10^{-8}\) from \(2.9409\times10^{-8}\) requires
\(\rho_*=0.8124\), hence

\[
\boxed{
 \varepsilon_{\rm total}
 \le{\rho_t-0.8124\over1+0.8124}.} \tag{104.32}
\]

Even at the optimistic \(\rho_t=1\),

\[
\boxed{\varepsilon_{\rm total}\le0.1035.} \tag{104.33}
\]

The relative error is measured after complete-old projection.  A
10.35-percent bound on an unprojected source, a single seam, or the first
action alone is not the gate.

Graph-ready split:

- GAUSSIAN_Q211_DG -- **solved exact**, (104.2)--(104.7);
- GAUSSIAN_COVARIANCE_HESSIAN -- **solved exact**, (104.8)--(104.9);
- GAUSSIAN_CP3_HESSIAN -- **solved exact**, (104.10)--(104.12);
- GAUSSIAN_Q211_HESSIAN_COORDINATES -- **solved exact**,
  (104.13)--(104.17);
- GAUSSIAN_Q211_SECOND_SANDWICH -- **generic new row**,
  (104.18)--(104.24);
- GAUSSIAN_Q211_SAME_SANDWICH_IFF -- **solved exact**, (104.25);
- GAUSSIAN_Q211_ONE_EXTRA_SANDWICH_IFF -- **solved exact**,
  (104.26)--(104.28);
- GAUSSIAN_Q211_COADAPT_CAP49 -- **generic ten-/twenty-equivalent no-go**,
  conditional escapes (104.29)--(104.30);
- GAUSSIAN_Q211_RAW1E8_FIDELITY -- **at most 10.35% optimistic error after
  old projection**, (104.31)--(104.33);
- GAUSSIAN_Q211_COADAPT_VALUE -- **open** until the exact finite correction
  has complete-old corpus alignment.

## 99. Arbitrary-bias Hermite/Price (q211) quotient through correlation
degree four: exact-jet duplication, the live quadratic defect, and the
one-sandwich criterion

Sections 88, 90, and 94 leave two different quotients easy to conflate.
If the comparator literally owns the exact Gaussian CP3 value and its
anchor-colored first threshold jet, the remaining ReLU (q211) row has no
three-edge cycle at all.  The current (P_{\le2})/finite-D11 branch does not
have that owner.  Its intrinsic Hermite defect retains a broad triangle and,
at heterogeneous bias, three degree-four edge upgrades.  This section gives
both statements with the old-span and collision rows kept outside the
algebra.

Let

\[
 Y_a=\sigma_a(X_a-\tau_a)_+,
 \qquad \mu_a=E Y_a,
 \qquad F_a=Y_a-\mu_a,
 \tag{99.1}
\]

and use the unnormalized Hermite coefficients

\[
 b_{a1}=\sigma_a\overline\Phi(\tau_a),\qquad
 b_{aq}=\sigma_a\phi(\tau_a)
               \operatorname{He}_{q-2}(\tau_a),\quad q\ge2.
 \tag{99.2}
\]

Thus

\[
 \partial_{\tau_a}b_{aq}=-b_{a,q+1},\qquad
 \mu_a=b_{a2}-\tau_ab_{a1},\qquad
 b_{a,q+1}=\tau_ab_{aq}-(q-2)b_{a,q-1}\quad(q\ge2).
 \tag{99.3}
\]

For distinct (i,j,k), put

\[
 x=R_{ij},\qquad y=R_{ik},\qquad z=R_{jk},
 \tag{99.4}
\]

and write (K^H_{i;jk}=\kappa_4(H_i,H_i,H_j,H_k)).  The square-response
coefficients needed through degree four are

\[
\begin{aligned}
 h^F_{i1}&=2\mu_i(\sigma_i-b_{i1}),\\
 h^F_{iq}&=2\sigma_i b_{i,q-1}-2\mu_i b_{iq},\qquad q\ge2.
\end{aligned}
 \tag{99.5}
\]

Indeed (Y_i^2{}'=2\sigma_iY_i), while
(Y_i^2{}^{(q)}=2\sigma_i^2 I_i{}^{(q-2)}) for (q\ge2).
For the literal quadratic Hermite owner (denoted (Q_i), to distinguish it
from the standardized Gaussian coordinate (X_i)),

\[
 Q_i=P_{\le2}F_i
 =b_{i1}\operatorname{He}_1(X_i)
  +{b_{i2}\over2}\operatorname{He}_2(X_i),
 \tag{99.6}
\]

the corresponding coefficients are

\[
 h^Q_{i1}=2b_{i1}b_{i2},\quad
 h^Q_{i2}=2(b_{i1}^2+b_{i2}^2),\quad
 h^Q_{i3}=6b_{i1}b_{i2},\quad
 h^Q_{i4}=6b_{i2}^2.
 \tag{99.7}
\]

Set (d_{iq}=h^F_{iq}-h^Q_{iq}).

### 99.1 Exact CP3-plus-colored-JVP ownership kills every broad cycle

Let (T_{ijk}=\kappa_3(F_i,F_j,F_k)), and define the quotient left after
the *exact* Section 90 value and colored threshold-JVP row by

\[
 \mathcal E_{i;jk}:=K^F_{i;jk}
 +\sigma_i\left(\tau_i+{2\mu_i\over\sigma_i}
 +\partial_{\tau_i}+x\partial_{\tau_j}
 +y\partial_{\tau_k}\right)T_{ijk}.
 \tag{99.8}
\]

The collapsed-slot Price series (78.6), (99.3), and (99.5) give the
all-orders identity

\[
\boxed{\begin{aligned}
 \mathcal E_{i;jk}={}&
 \sum_{a,b\ge1}{c_{i;ab}b_{ja}b_{kb}\over a!b!}x^ay^b\\
 &+\sigma_i\mu_i\sum_{c\ge1}{
 b_{j,c+1}b_{kc}\,xz^c+b_{jc}b_{k,c+1}\,yz^c\over c!},\\
 c_{i;ab}:={}&\sigma_i
   (\mathbf1_{a=1}+\mathbf1_{b=1})b_{i,a+b-1}
   -2b_{ia}b_{ib}.
\end{aligned}} \tag{99.9}
\]

Here is the cancellation.  In a term with (c\ge1), let (n=a+b).
Before simplification its anchor coefficient is

\[
 h^F_{in}+\sigma_i\{(\tau_i+2\mu_i/\sigma_i)b_{in}
 -b_{i,n+1}-n b_{i,n-1}\}.
 \tag{99.10}
\]

It equals zero for every (n\ge2), by (99.3)--(99.5), and equals
(\sigma_i\mu_i) for (n=1).  Thus (99.9) has either the two rooted
edges (x,y), or one anchor edge and the leaf edge (x,z) or (y,z),
but never all three (x,y,z).  Consequently an exact CP3 value plus its
literal colored first threshold jet leaves **zero broad matrix
sandwiches**, not one, for a CP endpoint.  Every term in (99.9) is a finite
list of Schur-power matvecs after a correlation-degree cut.

Through total degree four this action can be written explicitly.  For one
ordered partially symmetric CP atom

\[
 L_{i;jk}=\alpha_i{
  \beta_j\gamma_k+\gamma_j\beta_k\over2},
 \tag{99.11}
\]

put

\[
 p_q^\beta=R^{\odot q}(\beta\odot b_q),\qquad
 p_q^\gamma=R^{\odot q}(\gamma\odot b_q).
 \tag{99.12}
\]

The rooted part is

\[
\begin{aligned}
 S_{\rm root}^{\le4}={}&
 \langle\alpha\odot2b_1(\sigma-b_1),
          p_1^\beta\odot p_1^\gamma\rangle\\
 &+{1\over2}\langle\alpha\odot(\sigma-2b_1)b_2,
       p_1^\beta\odot p_2^\gamma
       +p_2^\beta\odot p_1^\gamma\rangle\\
 &+{1\over6}\langle\alpha\odot(\sigma-2b_1)b_3,
       p_1^\beta\odot p_3^\gamma
       +p_3^\beta\odot p_1^\gamma\rangle
 -{1\over2}\langle\alpha\odot b_2^{\odot2},
       p_2^\beta\odot p_2^\gamma\rangle,
\end{aligned}
 \tag{99.13}
\]

and the open-path part is

\[
\boxed{\begin{aligned}
 S_{\rm path}^{\le4}
 =\sum_{c=1}^3{1\over c!}(\alpha\odot\sigma\odot\mu)^TR
 \big[&(\beta\odot b_{c+1})\odot
          R^{\odot c}(\gamma\odot b_c)\\
      +&(\gamma\odot b_{c+1})\odot
          R^{\odot c}(\beta\odot b_c)\big].
\end{aligned}} \tag{99.14}
\]

Equations (99.13)--(99.14) are before strict-collision subtraction.  Their
(i=j,i=k,j=k) pieces are two-label rows and must be compiled against the
declared K4/K31/K22/pair owner.  They are not a hidden broad product.

### 99.2 The current quadratic/D11 branch retains a broad degree-four jet

The strong cancellation above may not be credited to the current lineage:
its finite CP3/D11 rows are not the exact (T,J_\tau T) in (99.8).
The intrinsic quadratic Hermite defect is

\[
 \Delta_{i;jk}:=K^F_{i;jk}-K^Q_{i;jk}.
 \tag{99.15}
\]

The complete Taylor polynomial through total correlation degree four is
the following exact coefficient table.  Each displayed coefficient
multiplies the monomial in the left column.

\[
\begin{array}{c|l}
\text{monomial}&\text{coefficient}\\ \hline
xy&d_{i2}b_{j1}b_{k1}\\
xz&d_{i1}b_{j2}b_{k1}\\
yz&d_{i1}b_{j1}b_{k2}\\[1mm]
x^2y&\tfrac12d_{i3}b_{j2}b_{k1}\\
xy^2&\tfrac12d_{i3}b_{j1}b_{k2}\\
x^2z&\tfrac12h^F_{i2}b_{j3}b_{k1}\\
y^2z&\tfrac12h^F_{i2}b_{j1}b_{k3}\\
xz^2&\tfrac12h^F_{i1}b_{j3}b_{k2}\\
yz^2&\tfrac12h^F_{i1}b_{j2}b_{k3}\\
xyz&d_{i2}b_{j2}b_{k2}\\[1mm]
x^3y&\tfrac16(h^F_{i4}-2b_{i1}b_{i3})b_{j3}b_{k1}\\
xy^3&\tfrac16(h^F_{i4}-2b_{i1}b_{i3})b_{j1}b_{k3}\\
x^2y^2&\tfrac14d_{i4}b_{j2}b_{k2}\\
x^3z&\tfrac16h^F_{i3}b_{j4}b_{k1}\\
y^3z&\tfrac16h^F_{i3}b_{j1}b_{k4}\\
x^2z^2&\tfrac14h^F_{i2}b_{j4}b_{k2}\\
y^2z^2&\tfrac14h^F_{i2}b_{j2}b_{k4}\\
xz^3&\tfrac16h^F_{i1}b_{j4}b_{k3}\\
yz^3&\tfrac16h^F_{i1}b_{j3}b_{k4}\\
x^2yz&\tfrac12h^F_{i3}b_{j3}b_{k2}\\
xy^2z&\tfrac12h^F_{i3}b_{j2}b_{k3}\\
xyz^2&\tfrac12h^F_{i2}b_{j3}b_{k3}.
\end{array}
 \tag{99.16}
\]

Formula (99.16) follows directly from the two lines of (78.6).  It is
also a useful normalization audit: at zero bias (b_3=0), the only broad
term through degree four is
(d_{i2}b_{j2}b_{k2}xyz), which reproduces the coefficient
(-(4-\pi)/(4\pi^2)) in (91.7) after standardization.

The broad, three-edge part of (99.16) is exactly

\[
\boxed{\begin{aligned}
 \Delta^{\rm broad,\le4}_{i;jk}={}&
 d_{i2}b_{j2}b_{k2}xyz\\
 &+{h^F_{i3}\over2}
   \{b_{j3}b_{k2}x^2yz+b_{j2}b_{k3}xy^2z\}\\
 &+{h^F_{i2}\over2}b_{j3}b_{k3}xyz^2.
\end{aligned}} \tag{99.17}
\]

All other entries in (99.16) are rooted or open paths and are matvec-native
for (99.11).  Crucially, (99.17) is a *pre-old* identity.  Let
(P_{\rm old}) be the literal synchronized projector containing the
actually deployed finite CP3 value, colored JVP, (P_{\le2}), pair, and
collision rows.  The deployable coefficient is

\[
 \overline\Delta^{\rm broad,\le4}
 =(I-P_{\rm old})\Delta^{\rm broad,\le4}.
 \tag{99.18}
\]

No term may be deleted merely because its unprojected formula resembles a
CP3 or threshold row.  Conversely, a row literally in (P_{\rm old})
must not be billed again.  If (99.18) no longer preserves the separated
diagonal form in (99.17), its actual cycle separation rank, not the raw
formula below, is binding.

### 99.3 Literal sandwich count and the sharp ungraded one-product iff

For one atom (99.11), singleton symmetry fuses the two side orientations,
but does not fuse them with the base or leaf-edge-square channel.  Before
strict two-label corrections, the three literal products are

\[
\begin{aligned}
 B_0={}&\operatorname{tr}
  (D_{\alpha\odot d_2}R
   D_{\beta\odot b_2}R
   D_{\gamma\odot b_2}R),\\
 B_z={}&{1\over2}\operatorname{tr}
  (D_{\alpha\odot h^F_2}R
   D_{\beta\odot b_3}R^{\odot2}
   D_{\gamma\odot b_3}R),\\
 B_s={}&{1\over2}\operatorname{tr}\!\left[
  D_{\alpha\odot h^F_3}R^{\odot2}
  \{D_{\beta\odot b_3}R D_{\gamma\odot b_2}
   +D_{\gamma\odot b_3}R D_{\beta\odot b_2}\}R\right].
\end{aligned}
 \tag{99.19a}
\]

Their sum is (99.17).  Each line needs one ordinary query-specific dense
matrix multiplication when no identical product is resident.  For a
generic dense correlation matrix, (R) and (R^{\odot2}) are
algebraically independent, so the **literal ungraded minimum is three
sandwiches generically**.

The base and (z^2) channels fuse when

\[
 \boxed{h^F_{i2}=\lambda d_{i2}
 \quad\hbox{on every live anchor }i.} \tag{99.19b}
\]

In that case

\[
 B_0+B_z=\operatorname{tr}\!\left[
 D_{\alpha\odot d_2}R
 \{D_{\beta\odot b_2}R D_{\gamma\odot b_2}
  +{\lambda\over2}D_{\beta\odot b_3}R^{\odot2}
                    D_{\gamma\odot b_3}\}R\right].
 \tag{99.19c}
\]

Thus a common standardized threshold, for which (h^F_2) and (d_2)
both scale as (\sigma^2), reduces the ordinary count from three to two.
Generic heterogeneous thresholds do not satisfy (99.19b).

For an ordinary ungraded circuit, one sandwich is possible if and only if
the residual three-channel product has product-channel rank at most one:

\[
 \boxed{\operatorname{prank}_{\rm sand}
 \big((I-P_{\rm old})\Delta^{\rm broad,\le4}\big)\le1.}
 \tag{99.19d}
\]

The rank is taken in the tensor product of the cheaply formed left and
middle factor spaces, modulo resident products and the two-label collision
subspace.  On a generic dense (R), with nonzero base channel and the
separated diagonals of (99.17), this invariant iff reduces sharply to

\[
\boxed{\begin{gathered}
 h^F_{i2}=\lambda d_{i2}
 \quad\hbox{on the live anchor support},\\
 (I-P_{\rm old})\!\left[
 h^F_{i3}\{b_{j3}b_{k2}x^2yz
            +b_{j2}b_{k3}xy^2z\}\right]=0.
\end{gathered}} \tag{99.19e}
\]

If (d_{i2}=0), every surviving (h^F_{i2}) or side channel at that
anchor must itself be old or zero.  Sufficiency is (99.19c); necessity is
polynomial independence of the base, side-powered, and leaf-powered edge
words.  Zero bias satisfies (99.19e) because (b_3=0), leaving exactly
the ordinary (B_0) triangle.  Degenerate supports or resident products
are covered by the invariant test (99.19d), not by the generic
specialization (99.19e).

### 99.4 Formal graded compression: two jets generically and one under a
common diagonal ratio

First ignore old rows that mix the three coefficient channels, and apply
(99.17) to one atom (99.11).  On the live support where
(d_{i2}b_{j2}b_{k2}\ne0), define

\[
 r_i={h^F_{i3}\over2d_{i2}},\qquad
 s_a={b_{a3}\over b_{a2}},\qquad
 \lambda_i={h^F_{i2}\over d_{i2}}.
 \tag{99.19}
\]

The finite-threshold coefficients (b_{a2}=\sigma_a\phi(\tau_a)) are
positive, so only zeros of (d_{i2}) require a support split.  Put

\[
 A_i=\alpha_i d_{i2},\qquad
 B_j=\beta_jb_{j2},\qquad
 C_k=\gamma_kb_{k2},
 \tag{99.20}
\]

and define the two decorated anchor edges

\[
 U_{ij}=R_{ij}+r_is_jR_{ij}^2,qquad
 V_{ik}=R_{ik}+r_is_kR_{ik}^2.
 \tag{99.21}
\]

Then the degree-three base and the two degree-four anchor-edge upgrades are
the degree-at-most-four part of the single sandwich

\[
 \sum_{i,j,k}^{\rm strict}A_iB_jC_kU_{ij}V_{ik}R_{jk}.
 \tag{99.22}
\]

The remaining (xyz^2) channel is one further graded sandwich.  Hence the
generic formal minimum is **two** broad degree-jet sandwiches; singleton
symmetry fuses the (x^2yz) and (xy^2z) orientations.

There is an exact one-sandwich criterion in graded jet algebra.  The
polynomial jet (99.17) is
the degree-at-most-four part of one edge-local decorated triangle if and
only if, after the literal old/collision projection,

\[
 \boxed{\lambda_i=\lambda\quad
 \text{for every live anchor }i,} \tag{99.23}
\]

with the support condition that (d_{i2}=0) forces the surviving
(h^F_{i2}) and (h^F_{i3}b_{j3}b_{k2}) channels at that anchor to vanish
or be old.  Under (99.23), set

\[
 W_{jk}=R_{jk}+{\lambda\over2}s_js_kR_{jk}^2.
 \tag{99.24}
\]

Then

\[
\boxed{
 \langle L,\Delta^{\rm broad,\le4}\rangle
 =\left[\sum_{i,j,k}^{\rm strict}
 A_iB_jC_kU_{ij}V_{ik}W_{jk}\right]_{\deg\le4}.}
 \tag{99.25}
\]

Sufficiency follows by expanding one upgraded edge at a time.  For
necessity, compare the algebraically independent (xyz) and (xyz^2)
coefficients.  Their ratio at a fixed ((j,k)) must be an edge-local
quantity and therefore cannot depend on the anchor (i); that ratio is
((h^F_{i2}/d_{i2})(b_{j3}b_{k3}/(2b_{j2}b_{k2}))), proving (99.23).
The (x^2yz) and (xy^2z) ratios then force exactly the edge factors in
(99.21), but impose no additional cross-anchor condition.

For a projector that mixes the separated channels, the invariant version
of (99.23) is

\[
 \boxed{\operatorname{srank}_{\rm cycle}
 \big((I-P_{\rm old})\Delta^{\rm broad,\le4}\big)\le1,}
 \tag{99.26}
\]

where cycle separation rank is the minimum number of edge-local decorated
triangle products modulo the two-label collision subspace.  Equation
(99.26) is necessary and sufficient by definition; (99.23) is its sharp
diagonal-weight specialization for (99.17).  Generic heterogeneous
thresholds make (lambda_i) nonconstant and require two sandwiches.
A common standardized threshold makes every (h^F_{i2},d_{i2}) scale as
(\sigma_i^2), so (99.23) holds.  Zero bias is the stronger special case
(b_3=0): all degree-four upgrades vanish and the ordinary one triangle
sandwich remains.

The brackets in (99.25) are important.  A literal multiplication of the
decorated matrices also produces terms of degree at least five.  It is an
exact one-product *degree-four jet* only if the implementation uses graded
truncation, or else those higher terms define a deliberate resummation and
need a separate tail certificate.  Dropping the brackets while claiming
exact degree-four evaluation is false.

Finally, strict masking never changes the broad-product count.  For

\[
 \mathfrak S(a,b,c;A,B,C)=
 \sum_{i,j,k}a_ib_jc_kA_{ij}B_{jk}C_{ki},
 \tag{99.27}
\]

the strict version is

\[
 \mathfrak S_{\rm strict}
 =\mathfrak S-\mathfrak S_{i=j}-\mathfrak S_{j=k}
  -\mathfrak S_{k=i}+2\mathfrak S_{i=j=k}.
 \tag{99.28}
\]

Every subtraction in (99.28) is a one- or two-label contraction.  It must
be residualized against the collision owner, but it creates no extra dense
sandwich.

Graph-ready split:

- `EXACT_CP3_COLORED_JVP_Q211_DUPLICATE` -- **solved exact all orders**,
  (99.8)--(99.10): zero broad residual sandwich;
- `BIASED_Q211_DEGREE4_COEFFICIENTS` -- **solved exact**, (99.5)--(99.17);
- `CURRENT_D11_OLD_PROJECTION` -- **binding literal receipt**, (99.18);
- `BIASED_Q211_LITERAL_SANDWICH_COUNT` -- **three generically**, with the
  sharp ordinary one-product iff (99.19a)--(99.19e);
- `BIASED_Q211_GRADED_SANDWICH_COUNT` -- **two formal jets generically**,
  (99.19)--(99.22);
- `BIASED_Q211_GRADED_ONE_SANDWICH_IFF` -- **solved exact jet criterion**,
  (99.23)--(99.26);
- `BIASED_Q211_GRADED_PRODUCT` -- **required for literal degree-four
  exactness**; an ungraded product needs a degree-five tail receipt;
- `BIASED_Q211_COLLISION_OWNER` -- **separate two-label rows**, (99.28);
- `BIASED_Q211_TRUE_MINUS_GAUSSIAN` -- **not supplied**.

## 105. The physical variance adjoint turns the one-atom q211 sandwich
into a tetrahedron: exact rank, batching, and frozen-sketch gates

Section 88 proves that one *CP endpoint atom* needs one broad sandwich.
The literal variance/K4 Bellman row is generally not one atom.  Its shared
row structure gives an exact four-index contraction whose cost must be
audited before calling the proposed feature one product per layer.

### 105.1 Exact physical action

Let (W\in\mathbb R^{n\times n}) be the dense transition, let (w_a^T)
be row (a), let (q_a) be the synchronized scalar variance/K4 costate,
and let (R,P\in\mathbb R^{n\times n}) be symmetric.  For Section 88,
(P=p[R]); for Section 96, (P=P^\sharp).  The physical partially
symmetric endpoint row is

\[
 L_{i;jk}=\sum_a q_a W_{ai}^2W_{aj}W_{ak}. \tag{105.1}
\]

Its broad action is

\[
 \boxed{
 S=\sum_{a,i,j,k}q_aW_{ai}^2W_{aj}W_{ak}
 R_{ij}R_{ik}P_{jk}.}
 \tag{105.2}
\]

With (D_a=\operatorname{Diag}(w_a)) and
(M_a=D_aRD_a), cyclicity gives the useful exact normal form

\[
 \boxed{S=\sum_aq_a\operatorname{tr}(M_aPM_a).} \tag{105.3}
\]

Thus the “one sandwich” of (88.16) is one summand in (105.3), not the
whole physical row unless only one (q_a) is live or an exact common
factorization is supplied.

The pairwise factor graph of (105.2) has vertices (a,i,j,k) and all six
edges

\[
 ai,aj,ak,ij,ik,jk. \tag{105.4}
\]

It is the complete graph (K_4), of treewidth three.  In the ordinary
dense pairwise-contraction model, fixing a separable endpoint atom removes
the (a) vertex and leaves one triangular/GEMM contraction.  Keeping a
generic physical (a) vertex gives an (n^4) contraction, equivalently
(n) triangular products.  This is a circuit-model statement, not an
unconditional lower bound against arbitrary fast algebraic algorithms;
the exact rank obstructions below are model-independent.

### 105.2 Generic endpoint rank is broad even after strict masking

Before collision masking, the anchor-versus-pair flattening of (105.1) is

\[
 L_{i\mid(jk)}
 =(W^{\odot2})^TD_q\,\mathcal V_2(W),
 \qquad
 \mathcal V_2(W)_{a,(jk)}=W_{aj}W_{ak}. \tag{105.5}
\]

Both (W^{\odot2}) and the row-Veronese matrix
(\mathcal V_2(W)) have rank (n) on a nonempty Zariski-open set.
Therefore, if all (q_a\ne0),

\[
 \boxed{\operatorname{rank}L_{i\mid(jk)}=n} \tag{105.6}
\]

generically.  In particular every partially symmetric CP realization has
at least (n) atoms.

Strict q211 masking does not give a bounded-rank theorem.  Partition the
labels into disjoint sets (I,J,K) of size
(m=\lfloor n/3\rfloor) and restrict (105.5) to
(i\in I,j\in J,k\in K).  No collision occurs on this block.  A direct
matrix-unit choice of rows gives one nonzero (m\)-minor, and hence by
polynomial openness

\[
 \boxed{
 \operatorname{rank}L^{\rm strict}_{I\mid(JK)}\ge m}
 \tag{105.7}
\]

generically.  At (n=1024), this already forces at least (341) exact
endpoint atoms.  Complete-old projection can reduce this rank, but only its
literal projected minor/rank is a receipt; diagonal K4 nomenclature is not.

### 105.3 Exact factor-(P) batching

There is a second exact contraction order.  Let

\[
 P=\sum_{t=1}^{\rho}\lambda_tb_tb_t^T,
 \qquad \rho=\operatorname{rank}P, \tag{105.8}
\]

and form

\[
 H_t=WD_{b_t}R^T. \tag{105.9}
\]

Then

\[
 \boxed{
 S=\sum_{t=1}^{\rho}\lambda_t
 \sum_{a,i}q_a\{W_{ai}(H_t)_{ai}\}^2.}
 \tag{105.10}
\]

Proof follows by expanding \(b_{tj}b_{tk}\) in (105.2).  Hence a rank-
\(\rho\) gate kernel gives an exact \(\rho\)-product implementation, shared
over every physical row (a).  The honest deterministic product count is
the smaller of a certified old-projected endpoint factor count and a
certified (P)-factor count, plus masks, hooks, and lower-order rows.

For the zero-bias gate kernel

\[
 p(r)={1\over4}+{1\over2\pi}\arcsin r, \tag{105.11}
\]

there is no generic low-rank theorem.  At (R=I),

\[
 p[R]={1\over4}(I+\mathbf1\mathbf1^T), \tag{105.12}
\]

which is positive definite.  The strict off-diagonal version
((\mathbf1\mathbf1^T-I)/4) is also full rank for (n>1).  The modified
(P^\sharp=p[R]-R/(2\pi^2)) remains full rank at the same witness.  Thus

\[
 \boxed{\operatorname{rank}p[R]
 =\operatorname{rank}P^\sharp=n} \tag{105.13}
\]

on an open set.  The exact factor-(P) circuit then costs (n) products
per active seam, not one.

### 105.4 Deterministic spectral truncation

Define the source-weighted coefficient matrix

\[
 C=\sum_aq_aM_a^2. \tag{105.14}
\]

Equation (105.3) is simply (S=\langle C,P\rangle_F).  If (P_k) retains
any (k) spectral factors, then

\[
 \boxed{
 |S(P)-S(P_k)|
 \le\|C\|_F\|P-P_k\|_F
 =\|C\|_F\left(\sum_{t>k}\lambda_t^2\right)^{1/2}.}
 \tag{105.15}
\]

The operator/nuclear alternative is

\[
 |S(P)-S(P_k)|\le\|C\|_*\|P-P_k\|_{\rm op}. \tag{105.16}
\]

These are endpoint-weighted gates.  Source energy in (P), or a low
effective rank before multiplication by (C), is not sufficient.  Across
a corpus and layers, sum the signed errors first and then apply the
complete-old projector and the robust correlation gate (89.19).

### 105.5 Two exact frozen stochastic contractions

Because (R) is a correlation Gram, choose unit vectors (u_j) with
(R_{jk}=u_j^Tu_k), draw (z\sim N(0,I)), and set

\[
 b_j(z)=\mathbf1\{u_j^Tz>0\}. \tag{105.17}
\]

Then

\[
 p[R]=E_z[b(z)b(z)^T]. \tag{105.18}
\]

Substituting one frozen draw into (105.10) gives

\[
 Z(z)=\sum_{a,i}q_a
 \{W_{ai}(WD_{b(z)}R^T)_{ai}\}^2,
 \qquad E Z=S. \tag{105.19}
\]

The (k)-draw mean is unbiased and has the exact variance

\[
 \boxed{\operatorname{Var}\widehat S_k
 ={1\over k}\operatorname{Var}Z.} \tag{105.20}
\]

For \(P^\sharp=P-cR\), subtracting the exact linear-\(R\) action is free
only if that physical action is literally resident.  Otherwise
\(\operatorname{rank}R=r\) costs up to \(r\) additional factors.  A
one-product-per-draw unbiased alternative is a signed mixture: with
probability \(1/2\) use \(b(z)\) with scalar weight \(+2\), and with
probability \(1/2\) use \(v=Uz\), \(R=UU^T\), with weight \(-2c\).
Then

\[
 E[\omega\,\xi\xi^T]=P-cR=P^\sharp. \tag{105.20a}
\]

It preserves the one-factor arithmetic but has the variance of the signed
mixture, not the smaller variance in (105.20).  A common frozen seed must
be used across all panel nets/layers if the feature is to have a stable
corpus direction.

Alternatively sample physical endpoint rows.  Put
(c_a=q_a\operatorname{tr}(M_aPM_a)), choose preregistered probabilities
(pi_a>0), and draw (A_t\sim\pi).  Then

\[
 \widehat S_k={1\over k}\sum_{t=1}^k{c_{A_t}\over\pi_{A_t}},
 \qquad
 \operatorname{Var}\widehat S_k
 ={1\over k}\left\{\sum_a{c_a^2\over\pi_a}-S^2\right\}. \tag{105.21}
\]

The formal variance-minimizer is
(pi_a=|c_a|/\sum_b|c_b|), giving relative RMS

\[
 \boxed{
 {\sqrt{\operatorname{Var}\widehat S_k}\over|S|}
 ={1\over\sqrt k}\sqrt{\kappa^2-1},
 \qquad
 \kappa={\sum_a|c_a|\over|\sum_ac_a|}.}
 \tag{105.22}
\]

Since the optimal probabilities themselves depend on the expensive
actions, a lawful implementation must freeze a cheap envelope or pilot-
independent distribution.  Equation (105.21), not the formal optimum, is
the deployed certificate.  There is no uniform small-(k) guarantee:
signed (c_a) can make (kappa) arbitrarily large, and gate-feature laws
can put all variance in a direction missed by (k) fixed samples.

### 105.6 Conditional low-rank-\(R\) skinny-product escape

The full-product counts above can be sharpened when the *literal \(R\) in
the two anchor edges* has a certified factor

\[
 R=UU^T,\qquad U\in\mathbb R^{n\times r_R}. \tag{105.22a}
\]

For a sampled physical row \(a\), put

\[
 V_a=D_aU,\qquad G_a=V_a^TV_a,\qquad H_a=V_a^TPV_a. \tag{105.22b}
\]

Then its exact contribution is

\[
 \boxed{c_a=q_a\operatorname{tr}(G_aH_a).} \tag{105.22c}
\]

Only \(PV_a\) is a dense skinny product; all other work is
\(O(nr_R^2+r_R^3)\).  Thus \(k\) endpoint-row samples over \(D\) seams have
dominant count

\[
 \boxed{C_{\rm row,skinny}={Dkr_R\over n}} \tag{105.22d}
\]

before factor formation, masks, hooks, and reductions.

For one rank-one gate factor \(P=bb^T\), define

\[
 A=WD_bU,\qquad H=AU^T. \tag{105.22e}
\]

Substitution into (105.10) is exact.  It needs two skinny products,
\(WD_bU\) and \(AU^T\), unless one is a literal resident view.  Therefore

\[
 \boxed{C_{\rm gate,skinny}={2Dkr_R\over n},} \tag{105.22f}
\]

or half this count with one paid product reused.  Stacking all \(k\)
factors changes scheduling but not the classical FMA count.

This escape is typed.  An affine Gaussian comparator with latent width
\(r=55\) has \(r_R\le55\).  By contrast, the covariance of a generic
quadratic-AQ projection has feature dimension

\[
 r+{r(r+1)\over2}=55+1540=1595, \tag{105.22g}
\]

and therefore has generic physical rank \(1024\).  Adding a generic
independent Gaussian-rest covariance also restores full physical rank.
Consequently the phrase “AQ r55” is not a rank-55 receipt for the q211
correlation matrix.  The skinny escape is lawful only for an explicitly
affine rank-55 surface, or for a proved low-rank component with the
remainder charged separately.

If \(R=R_0+E\), write
\(M_a^0=D_aR_0D_a\) and \(N_a=D_aED_a\).  Holding \(P\) fixed, the exact
low-rank-edge truncation obeys

\[
\boxed{\begin{aligned}
 |S(R,P)-S(R_0,P)|
 \le\|P\|_{\rm op}\sum_a|q_a|
 \{2\|M_a^0\|_F\|N_a\|_F+\|N_a\|_F^2\}.
\end{aligned}} \tag{105.22h}
\]

If \(P=p[R]\) is also replaced by \(p[R_0]\), its additional kernel error
must be added through (105.15); the derivative of \(\arcsin\) is not
uniform near \(|R_{jk}|=1\).

### 105.7 Utility and decision gate

With ten active seams, (k) factor samples cost approximately (10k)
new f32 products before hooks.  At the audited (0.161\%\) utilization per
f32 equivalent, co-firing with the held (u=0.5629) pack gives the screen

\[
 u(k)\simeq0.5629(1+0.0161k). \tag{105.23}
\]

Thus (k\le4) is the largest integer that remains at or below the current
(u\simeq0.6002), before omitted costs.  If raw (10^{-8}) were actually
achieved, beating adjusted (6.5\times10^{-9}) requires (u(k)<0.65),
which gives (k\le9) under the same optimistic bill.  These are economics
ceilings, not fidelity guarantees.

For a literal affine \(r_R=55\) factor, (105.22d) costs
\(550k/1024\simeq0.5371k\) equivalents over ten seams, while (105.22f)
costs \(1.0742k\).  The corresponding optimistic utility-parity ceilings
are about \(k\le76\) endpoint rows or \(k\le38\) gate factors; the
raw-\(10^{-8}\) leader ceilings are about \(178\) and \(89\).
These larger counts are invalid for the generic quadratic-AQ/rest
covariance and still require the stochastic fidelity certificate.

The exact candidate gate is therefore:

1. obtain the literal complete-old physical costate (q,W);
2. use exact endpoint rank, exact (P)-rank, or a frozen (k\le4) (at
   most (9) for a goal-scale exception) estimator;
3. certify its corpus error through (105.15), (105.20), or (105.21);
4. require old-residualized held-out correlation at least (0.8124) for
   raw (10^{-8}); and
5. bill every seam, mask, hook, and control variate end to end.

Graph-ready split:

- Q211_PHYSICAL_VARIANCE_ACTION -- **solved exact tetrahedron**,
  (105.1)--(105.4);
- Q211_PHYSICAL_ENDPOINT_RANK -- **generic broad**, rank (n) unmasked
  and at least (\lfloor n/3\rfloor) strict, (105.5)--(105.7);
- Q211_GATE_FACTOR_BATCH -- **solved exact**, (105.8)--(105.10);
- Q211_GATE_FACTOR_GENERIC_RANK -- **generic full-rank NO-GO**,
  (105.11)--(105.13);
- Q211_GATE_SPECTRAL_TAIL -- **conditional deterministic**, (105.14)--
  (105.16);
- Q211_GATE_RANDOM_FEATURE -- **unbiased conditional**, (105.17)--
  (105.20);
- Q211_ENDPOINT_ROW_SKETCH -- **unbiased with exact variance**, (105.21)--
  (105.22), no uniform small-(k) guarantee;
- Q211_LOW_RANK_R_ROW_ESCAPE -- **conditional exact skinny circuit**,
  (105.22a)--(105.22d);
- Q211_LOW_RANK_R_GATE_ESCAPE -- **conditional exact two-skinny circuit**,
  (105.22e)--(105.22f);
- Q211_AQ_R55_RANK_PREMISE -- **generic NO for quadratic AQ/rest**,
  feature dimension 1595 and physical rank 1024, (105.22g);
- Q211_LOW_RANK_R_TAIL -- **conditional deterministic**, (105.22h);
- Q211_PHYSICAL_ACTION_UTILITY -- **(k\le4) utility-parity / (k\le9)
  optimistic raw-goal ceiling**, (105.23);
- Q211_ONE_SANDWICH_PER_LAYER -- **closed absent a literal rank-one
  endpoint or gate-factor receipt**;
- Q211_PHYSICAL_ACTION_VALUE -- **open** until old-residualized held-out
  correlation clears (0.8124).

## 106. The physical q211 complement is not ``matvec-only'': exact
reverse-root batching, the fused edge ledger, and the D10 gate

Section 88 counted the action against one partially symmetric CP atom.
Section 105 then restored the literal physical variance/K4 Bellman row and
showed why the broad term is generically a tetrahedron rather than one
sandwich.  There is one important exact batching order between those two
descriptions.  It gives the best resident-factor exception, but it also
exposes that every purported per-atom matvec in the remaining
\(F_{\rm edge}\) becomes a rectangular dense product when all physical
rows are consumed.

Throughout this section, absorb the synchronized variance factors into the
physical rays as in Section 105 and write

\[
 B=[v_1,\ldots,v_m]\in\mathbb R^{n\times m},\qquad
 V=B^{\odot2},\qquad \lambda_a=q_a.                 \tag{106.1}
\]

First merge exactly proportional rays: if \(v_b=cv_a\), replace their
weights by \(q_a+c^4q_b\), and delete any resulting zero arm.  Let
\(m=m_{\rm eff}\) after this canonicalization.  Scheduling or streaming
does not change this number.

### 106.1 Exact endpoint rank after ray canonicalization

The physical q211 endpoint tensor is

\[
 H_{i;jk}=\sum_{a=1}^m q_a v_{a,i}^2v_{a,j}v_{a,k}. \tag{106.2}
\]

Its anchor-versus-symmetric-pair flattening factors as

\[
 H_{i\mid(jk)}=X D_qY^T,qquad
 X_{ia}=v_{a,i}^2,qquad Y_{(jk),a}=v_{a,j}v_{a,k}. \tag{106.3}
\]

For generic nonproportional rays and nonzero weights, both factors have
column rank \(m\) when \(m\le n\).  Hence

\[
 \boxed{\operatorname{rank}H_{i\mid(jk)}=m,}        \tag{106.4}
\]

and the displayed \(m\)-arm realization attains the minimum partially
symmetric CP count.  Section 105.2 supplies the independent strict-mask
lower bound when all \(n\) physical rows are live.  Thus the only free
endpoint compression is exact zero/proportional-ray cancellation; signed
weights do not by themselves lower the generic rank.

### 106.2 Exact reverse-root batch for the broad term

Let \(P\) denote \(p[R]\) in Section 88 or \(P^\sharp\) in Section 96.
Only \(P_s=(P+P^T)/2\) contributes because
\(A_a:=D_{v_a}RD_{v_a}\) is symmetric.  Suppose a literal same-seam
signed congruence factor is resident:

\[
 P_s=FJF^T,qquad F\in\mathbb R^{n\times k},qquad
 J=\operatorname{diag}(\pm1),                       \tag{106.5}
\]

with magnitudes absorbed into \(F\).  Form the block of modulated roots

\[
 X=[D_{v_1}F\mid\cdots\mid D_{v_m}F],qquad
 Y=RX,                                               \tag{106.6}
\]

split \(Y=[Y_1|\cdots|Y_m]\), and put
\(Z_a=D_{v_a}Y_a=A_aF\).  Then the complete physical broad action is

\[
 \boxed{
 S_{\rm broad}=\sum_{a=1}^m q_a\operatorname{tr}(P_sA_a^2)
 =\sum_{a=1}^m\sum_{c=1}^k q_aJ_{cc}\|Z_{a:c}\|_2^2.} \tag{106.7}
\]

This is one *rectangular call*, not one width-\(n\) GEMM.  Its dense
arithmetic is

\[
 n^2mk+O(nmk),\qquad
 \boxed{C_{\rm broad}={mk\over n}\ \text{GEMM-equivalents}.} \tag{106.8}
\]

No cross-block \((mk)\times(mk)\) Gram is needed; signed column norms are
enough.  The route is at most one equivalent precisely when the factor is
already resident and \(mk\le n\).  Factoring a dense \(P\) costs
\(O(n^3)\) and invalidates that resident-factor claim.  If D10 exposes only
a bilinear factor \(P=UV^T\) of width \(k\), the exact identity

\[
 \operatorname{tr}(PA_a^2)=\langle A_aV,A_aU\rangle_F \tag{106.9}
\]

requires both modulated blocks and costs \(2mk/n\).  Symmetrizing a
generic \(UV^T\) does not remove this factor two.  A resident small middle
metric \(P=FHF^T\) may be signed-diagonalized in
\(O(k^3+nk^2)\); a dense middle metric may not be called free.

The width in (106.8) is generically irreducible.  The input block has
columns

\[
 X_{i,(a,c)}=v_{a,i}F_{ic},qquad
 \operatorname{rank}X=\min(n,mk)                    \tag{106.10}
\]

on a Zariski-open set.  A witness is
\(v_{a,i}=t_i^{a-1}\), \(F_{ic}=t_i^{m(c-1)}\), for distinct \(t_i\):
the \(mk\) columns are consecutive Vandermonde powers.  When \(mk\le n\),
every query column is therefore necessary.  When \(mk>n\), multiplying a
basis of \(n\) columns and reconstructing \(mk\) generic outputs requires
the same dense \(n\times mk\) reconstruction; rank saturation is not a
free billing escape.

For the strict \(j\ne k\) action, the diagonal of \(P\) is immaterial and
the sharp factor invariant is the symmetric off-diagonal completion rank

\[
 \rho_{\rm off}(P):=min_{d\in\mathbb R^n}
 \operatorname{rank}\{P+\operatorname{Diag}d\}.      \tag{106.11}
\]

Any chosen diagonal is sewn through the literal q22 collision row.  Thus
\(k=\rho_{\rm off}(P)\) is the minimum factor width in this modulated-root
circuit, provided that completion and its factor are actually available.
For generic off-diagonal data, the symmetric rank-\(r\) variety has
dimension \(r(2n-r+1)/2\), whereas the off-diagonal data have dimension
\(n(n-1)/2\).  Consequently

\[
 r\ge n-k_*,\qquad
 k_*:=\left\lfloor{\sqrt{1+8n}-1\over2}\right\rfloor. \tag{106.12}
\]

At \(n=1024\), this lower bound is \(r\ge980\).  Entrywise
\(p'(r)>0\) on \((-1,1)\), so \(p[\cdot]\) maps an open interior set of
correlation off-diagonals to another open set and the generic completion
bound applies.  Subtracting the explicit constant component
\(p(0)\mathbf1\mathbf1^T\) changes the dimension by at most one and does
not change the conclusion.

There is nevertheless one exact free direction.  Since

\[
 p[R]={1\over4}\mathbf1\mathbf1^T+\widetilde P,       \tag{106.13}
\]

the constant factor uses the already needed \(RB\): its contribution is
\(\tfrac14\operatorname{diag}\{V^T(RB)^{\odot2}\}\).  Only the
nonconstant completion must be paid.  A generic residual still has rank
\(n-O(\sqrt n)\).  More generally, an already computed broad input family
is reusable only when the exact columns \(D_{v_a}f_c\) lie in the old input
span at the same seam; a factor vector \(f_c\) by itself is not a product.

### 106.3 All non-broad edge terms: six products generically, five when
the broad intermediates are paid

Let

\[
 A_0={1\over4\pi}\sqrt{1-R^{\odot2}},\quad
 Q=q[R],\quad H=h[R],\quad S=s[R],\quad T=t[R],       \tag{106.14}
\]

and define \(d(X,Y)=\operatorname{diag}(X^TY)\),
\(s_B=B^T\mathbf1\), and \(r_B=V^T\mathbf1\).  The scalar kernels obey
the exact identities

\[
 q(r)=2A_0(r)+r p(r),\qquad h(r)=4q(r)-r,\qquad
 s(r)=4p(r)-1+8rA_0(r).                              \tag{106.14a}
\]

Consequently the following six batched dense products are sufficient:

\[
\begin{array}{lll}
 U_0=A_0B,& Z_0=R^{\odot2}(B\odot U_0),&U_R=RB,\\
 U_Q=QB,&U_H:=4U_Q-U_R,&Z_S=S(B\odot U_R),\\
 U_T=TB.&&
\end{array}                                           \tag{106.15}
\]

After using the \(j\leftrightarrow k\) symmetry, the complete all-index
non-broad edge action is the vector

\[
\boxed{\begin{aligned}
 f_{\rm thin}={}&2d(V,Z_0)
 +{1\over2}d(V,U_R\odot U_H)
 +{1\over4}d(V,Z_S)\\
 &-2d(V,U_Q\odot U_Q)
 -2\mu\{s_B\odot d(V,U_T)\}\\
 &+4\mu^2\{s_B\odot d(V,U_Q)\}
 +(\mu^2-2\mu^4)\{r_B\odot s_B\odot s_B\},
\end{aligned}}                                        \tag{106.16}
\]

and the physical scalar is \(q^Tf_{\rm thin}\).  Equation (106.16) is
just (88.9)--(88.10) with
\(\alpha=v_a^{\odot2}\), \(\beta=\gamma=v_a\); it contains no
statistical approximation.

There are useful exact alternative contraction orders.  The \(A_0\) term
may use \(A_0B\) and \(R^{\odot2}V\) by cyclicity, while

\[
 T B={\mu\over2}{\mathbf1s_B^T+2RB+R^{\odot2}B\}.    \tag{106.17}
\]

Thus a literal old \(R^{\odot2}V\) or \(R^{\odot2}B\) buffer can replace
the corresponding stage in (106.15).  The identity \(H=4Q-R\) means that
the H arm adds no product after \(QB\) and \(RB\) are known.  The generic
count is therefore two products for the \(A_0\) arm, one \(RB\), one
\(QB\), one additional S product, and one t product:

\[
 \boxed{C_{\rm thin}={6m\over n}.}                   \tag{106.18}
\]

Concatenating the six right hand sides into fewer library calls does not
alter their total width.  This count is attained and is generically
minimal before broad-factor reuse in the entrywise analytic-kernel model:
after applying (106.14a), the two-stage \(A_0\to R^{\odot2}\) and
\(R\to S\) paths contain a ray-dependent diagonal between their maps, and
the remaining \(Q\) and polynomial-t actions are linearly independent on
an open correlation set.

There is a stronger exact joint fusion.  If the broad factorization is
\(P=\sum_c\theta_cz_cz_c^T\) and its paid intermediates are
\(Y_c=R(D_{z_c}B)\), then

\[
 (R\odot P)B=\sum_c\theta_cD_{z_c}Y_c.               \tag{106.18a}
\]

Together with (106.14a), this supplies
\(U_Q=2U_0+(R\odot P)B\) and \(U_H=4U_Q-U_R\) without another dense
product.  An altered diagonal used for strict completion changes only a
row scaling and is corrected in the q22 sewing.  Therefore, whenever the
broad modulated products themselves are actually computed,

\[
 \boxed{C_{\rm thin\mid broad}={5m\over n}.}          \tag{106.18b}
\]

Merely owning the vectors \(z_c\), without the synchronized \(Y_c\), does
not trigger this reduction.

The nonconstant broad kernel cannot algebraically cancel into these five
thin terms.  Its mixed derivative in the two anchor correlations contains
the unique \(z\)-dependent term \(p(z)-p(0)\).  The \(A_0,S\) arms depend
on only one anchor correlation at a time, and \(q(x)q(y)\) has no \(z\)
dependence.  The broad completion-rank obstruction is therefore an
obstruction for the full edge complement, not merely for an inconvenient
decomposition of it.

### 106.4 Strict collisions add two more product families

For any symmetric edge kernel \({\cal K}(x,y,z)\), define

\[
 C_{31}(r)={\cal K}(1,r,r),\qquad
 C_{22}(r)={\cal K}(r,r,1),\qquad
 k_4={\cal K}(1,1,1),                                \tag{106.19}
\]

and apply these scalar functions entrywise to \(R\).  Inclusion--exclusion
gives, for every physical ray,

\[
\boxed{\begin{aligned}
 f_{\rm strict}(v_a)={}&f_{\rm all}(v_a)
 -2(v_a^{\odot3})^TC_{31}v_a\\
 &-(v_a^{\odot2})^TC_{22}v_a^{\odot2}
 +2k_4\mathbf1^Tv_a^{\odot4}.
\end{aligned}}                                        \tag{106.20}
\]

Thus strictification needs

\[
 U_{31}=C_{31}B,qquad U_{22}=C_{22}V,                \tag{106.21}
\]

or another \(2m/n\) equivalents generically.  The diagonal K4 term is
only a reduction.  These two products cost zero exactly when synchronized
q31/q22 rows own the same cut, collision convention, operands, and
contracted scalar.  The names “K31”, “K22”, or “Q24” are not ownership
receipts.

Combining (106.18) and (106.21), the supposedly matvec-only complement is

\[
 \boxed{C_{\rm thin+strict}={8m\over n},\qquad
 C_{\rm thin+strict\mid broad}={7m\over n}.}          \tag{106.22}
\]

before any broad factor, CP3 action, hooks, or root acquisition.

### 106.5 Literal ownership ledger and D10 exceptions

For seam \(\ell\), let \(b_\ell\) be the minimum number of unpaid stages
among

\[
\begin{gathered}
 A_0B,\quad R^{\odot2}(B\odot A_0B),\quad RB,
 \quad S(B\odot RB),\\
 QB,\quad TB,\quad C_{31}B,\quad C_{22}B^{\odot2},    \tag{106.23}
\end{gathered}
\]

after choosing between the exact alternatives in (106.17).  A stage may
be deleted only if its exact output or its final synchronized contraction
is already live at the seam.  Storing the kernel matrix, a pair scalar, a
plane family, a Mehler factor, or a CP3 value does not own the matrix-times-
physical-ray product.  In particular:

1. \(RB\) is shared by the H/S arms and the constant broad factor;
2. \(QB\) is shared by the H, quadratic-q, and linear-q arms;
3. the paid broad intermediates remove \(QB\) through (106.18a), but a
   resident Mehler vector \(z_c\) alone still leaves the paid product
   \(R(D_{z_c}B)\); and
4. Q24 or D10 removes an indicator only if the synchronized buffer survives
   with the same cut, mask, ray ordering, dtype, and lifetime.

There are two stronger D10 exceptions.  First, if D10 literally carries
the narrow signed root (106.5), use (106.6)--(106.8); a target-specific
factor reconstructed from a dense plane is not resident.  Second, if it
already carries the accumulated reverse plane

\[
 K_q=\sum_aq_aA_a^2,                                 \tag{106.24}
\]

then the broad scalar is just \(\langle P,K_q\rangle_F\) and no new
modulated product is required.  A resident factor
\(K_q=GJG^T\) of width \(h\) gives the intermediate exception
\(\operatorname{tr}(JG^TPG)\), costing \(h/n\) equivalents for \(PG\).
Acquiring either (106.24) or its factor is the expensive operation and
cannot be omitted from the bill.

If both \(P_s=FJF^T\) and a narrow factor
\(R=GKG^T\) are literal residents, define

\[
 C_a=G^TD_{v_a}F,qquad H_a=G^TD_{v_a^2}G.            \tag{106.25}
\]

Then

\[
 S_{\rm broad}=\sum_aq_a
 \operatorname{tr}(J C_a^TKH_aKC_a),                 \tag{106.26}
\]

with lower-order cost
\(O(mn r_Rk+mn r_R^2)\).  This is an exact skinny escape only when the
factor of the *literal transported* \(R\) is resident.  Section 105.6
explains why quadratic AQ plus a generic Gaussian rest does not supply an
\(r=55\) physical-rank receipt.

### 106.6 Complete cost gate and the only controlled Mehler escape

Let \(k_\ell\) be the unpaid signed-root width after exact constant/buffer
reuse; use \(2k_\ell\) for a bilinear D10 factor.  With \(D_s\) active
seams, the exact dense-product bill is

\[
\boxed{
 C_{106}=\sum_{\ell=1}^{D_s}{m_\ell(b_\ell+k_\ell)\over n}
 +C_{\rm roots}+C_{\rm CP3}+C_{\rm hooks}.}          \tag{106.27}
\]

The broad term is deleted from (106.27) only under the full resident-plane
exception (106.24).  The Cap49 gate is \(C_{106}\le49\); a base-48
package needs the stronger incremental gate \(C_{106}\le1\).

For the generic physical case \(m=n\), no old product ownership, and a
generic strict completion, one seam costs at least

\[
 7+\rho_{\rm off}(\widetilde P),                     \tag{106.28}
\]

which is at least \(987\) at \(n=1024\) by (106.12).  If the broad plane
were miraculously old but did not leave its factor intermediates live, ten
seams of the eight generic edge/collision products cost \(80\)
equivalents.  Conversely, if every thin/collision product is old but the
physical rays remain broad, ten seams require
\(10k\le49\), so only a literally resident \(k\le4\) root fits Cap49.
This recovers the Section 105 economic ceiling from an attained batching
circuit rather than from atomwise counting.

At the audited \(0.161\%\) relative utilization per f32 equivalent, the
held pack obeys approximately

\[
 u(N)=0.5629(1+0.00161N).                            \tag{106.29}
\]

The thin-plus-strict ten-seam bill \(N=80\) already gives
\(u\approx0.6354\) before broad work and hooks.  Therefore even a raw
\(10^{-8}\) result would have essentially no leader margin unless most of
the eight products are literal old views.

A finite Mehler expansion is an approximation, not exact ownership.  Since

\[
 p(r)-{1\over4}={1\over2\pi}
 \sum_{j\ge0}{(2j)!\over4^j(j!)^2(2j+1)}r^{2j+1},    \tag{106.30}
\]

the first nonconstant shell is proportional to \(R\) and is already full
rank for a generic transported covariance.  If all live off-diagonal
correlations satisfy \(|R_{jk}|\le\rho_0<1\), truncating through degree
\(2K+1\) has the deterministic entrywise tail

\[
 \epsilon_K\le{\rho_0^{2K+3}\over
 2\pi(2K+3)(1-\rho_0^2)}.                            \tag{106.31}
\]

Its physical broad-action error obeys

\[
 \boxed{
 |\Delta S|\le\epsilon_K\sum_a|q_a|
 \sum_i v_{a,i}^2\{(|R||v_a|)_i\}^2.}               \tag{106.32}
\]

Only this endpoint-weighted error, followed by complete-old projection and
the held-out value gate, can justify a truncated Mehler arm.  A fitted low-
rank factor or an inherited CP3 score is not a substitute.

Graph-ready split:

- Q211_PHYSICAL_RAY_CANONICAL_RANK -- **solved exact/generic**, merge
  proportional rays then rank \(m\), (106.1)--(106.4);
- Q211_BROAD_REVERSE_ROOT_BATCH -- **solved exact**, (106.5)--(106.8);
- Q211_BROAD_BILINEAR_D10_FACTOR -- **conditional, factor-two bill**,
  (106.9);
- Q211_BROAD_QUERY_WIDTH -- **generic irreducible**, (106.10);
- Q211_BROAD_OFFDIAG_COMPLETION -- **generic near-full-rank NO-GO**,
  (106.11)--(106.13);
- Q211_FEDGE_PHYSICAL_BATCH -- **solved exact six-product circuit, five
  marginal to paid broad factors**,
  (106.14)--(106.18);
- Q211_FEDGE_STRICT_BATCH -- **solved exact two-product correction**,
  (106.19)--(106.22);
- Q211_FEDGE_LITERAL_REUSE -- **blocking ownership census**, (106.23);
- Q211_D10_REVERSE_ROOT_EXCEPTION -- **conditional on literal resident
  factor**, (106.5)--(106.9);
- Q211_D10_ACCUMULATED_PLANE_EXCEPTION -- **conditional on literal
  resident \(K_q\)**, (106.24);
- Q211_BROAD_LOW_RANK_R_CORE -- **conditional exact**, (106.25)--(106.26);
- Q211_PHYSICAL_FULL_CAP -- **generic decisive NO-GO**, (106.27)--(106.29);
- Q211_MEHLER_ENDPOINT_TAIL -- **conditional approximation certificate**,
  (106.30)--(106.32);
- Q211_PHYSICAL_COMPLEMENT_VALUE -- **open** until the literal D10/Q24
  buffer census, complete-old projection, and held-out raw alignment are
  supplied.

## 107. The r55 regularity premise: exact adjoint test, sparse-frame defect,
and a two-minor Jordan reduction

Section 2 intentionally left one premise conditional: that the
(C,K21)-map has a point of full row rank at latent rank 55.  The dimension
count alone cannot prove it.  This section reduces the premise to explicit
finite minors, kills the tempting matrix-unit witness, and gives a sharp
43+12 block degeneration whose unexplained surplus is exactly 2,099.  It
does **not** assert that either remaining Jordan minor is nonzero.

### 107.1 Fixed-Gram tangent and the exact cokernel equations

Put

\[
 J=\operatorname{Sym}_{55},\qquad d=\dim J=1540,
 \qquad n=1024,qquad m=d-n=516.
\tag{107.1}
\]

It is enough to search among orthonormal independent tuples
(M_1,\ldots,M_n); this is a convenient witness class, not a claim that
Gram normalization preserves the rank of an arbitrary tuple.  If
(S=\operatorname{span}\{M_i\}) and (W=S^\perp), every fixed-Gram tangent
has the unique form

\[
 H_i=\sum_{p=1}^n\Omega_{pi}M_p+Z_i,qquad
 \Omega^T=-\Omega,qquad Z_i\in W.
\tag{107.2}
\]

Its dimension is

\[
 {n(n-1)\over2}+nm=523,776+528,384=1,052,160.
\tag{107.3}
\]

After dropping the irrelevant factor 8, the fixed-Gram cubic derivative is

\[
 \mathscr K_M(H)_{ij}
 =2\langle H_i,M_i\circ M_j\rangle
  +\langle M_i^2,H_j\rangle.
\tag{107.4}
\]

For a putative cokernel matrix (B\in\mathbb R^{n\times n}), define

\[
 \boxed{
 g_p(B)=2\sum_jB_{pj}(M_p\circ M_j)
       +\sum_iB_{ip}M_i^2.}
\tag{107.5}
\]

Pairing (107.4) with (B) shows that (B) annihilates every fixed-Gram
tangent if and only if

\[
 \boxed{
 g_p(B)\in S\quad\hbox{for every }p,qquad
 \Gamma_{pq}:=\langle g_p(B),M_q\rangle
 \quad\hbox{is symmetric}.}
\tag{107.6}
\]

Indeed, the normal space to the Stiefel manifold consists exactly of
collections (g_p=\sum_q\Gamma_{pq}M_q) with (\Gamma=\Gamma^T).  Thus
(D(C,K21)) has full row rank precisely when (107.6) has only (B=0).
The equation count is

\[
 nm+{n(n-1)\over2}=1,052,160>n^2=1,048,576
\tag{107.7}
\]

by 3,584.  This is compatibility with injectivity of the adjoint, not a
proof of it.

At every independent tuple, (DC) is onto.  Consequently

\[
 \boxed{D(C,K21)\hbox{ is onto}
 \iff DK21|_{\ker DC}\hbox{ is onto}.}
\tag{107.8}
\]

Equivalently, by the characteristic-zero Jacobian criterion, a regular
tuple exists iff the 1,573,376 trace polynomials

\[
 \{\operatorname{tr}(M_iM_j):i\le j\}
 \cup
 \{\operatorname{tr}(M_i^2M_j):i,j\}
\tag{107.9}
\]

are algebraically independent.  Simultaneous orthogonal conjugation gives
1,485 universal kernel directions.  If (107.8) holds, the total kernel is
(nd-q=3,584), leaving exactly (3,584-1,485=2,099) after gauge.  Without a
nonzero Jacobian minor, only the lower bound survives.

There is also a useful finite Schur certificate.  Suppose the (n) squares
(Q_i=M_i^2) are independent, let (\mathcal Q=\operatorname{span}\{Q_i\}),
and decompose

\[
 H_j=\sum_iU_{ij}Q^i+Z_j,qquad Z_j\in\mathcal Q^\perp,
\tag{107.10}
\]

where ((Q^i)) is the dual square basis.  Then

\[
 DK[H]=\mathcal L_M(U)+\mathcal R_M(Z),
\tag{107.11}
\]

where the identity term in (\mathcal L_M) is ((\mathcal L_MU)_{ij}=U_{ij}
+\cdots).  If (\det\mathcal L_M\ne0), solve (U) exactly as a function of
(Z) on (\ker DK).  Full regularity is then equivalent to surjectivity of
the finite Schur map

\[
 \boxed{
 \mathcal S_M:(\mathcal Q^\perp)^n\longrightarrow\operatorname{Sym}_n,
 \qquad Z\longmapsto DC[H(-\mathcal L_M^{-1}\mathcal R_MZ,Z)].}
\tag{107.12}
\]

Thus one determinant of (\mathcal L_M) and one 524,800-minor of
(\mathcal S_M) are a complete sufficient symbolic receipt.  Square
independence itself is easy to arrange analytically: choose 1,024
independent traceless (E_i) whose span generates the full matrix algebra and
put

\[
 M_i(t)=(I+tE_i)^{1/2}
\tag{107.13}
\]

for sufficiently small positive (t).  Then (M_i(t)^2=I+tE_i) are
independent and the tuple has scalar commutant.  What remains unproved for
this concrete family is exactly the two nonzero minors in (107.12).

### 107.2 The matrix-unit frame has an explicit cokernel

Choosing 1,024 normalized off-diagonal matrix units does not supply the
missing witness.  Let (G) be any 1,024-edge subset of the complete graph on
55 vertices and put

\[
 M_{uv}={E_{uv}+E_{vu}\over\sqrt2},\qquad uv\in G.
\tag{107.14}
\]

Fix one selected edge (p).  At most (2\cdot54-1=107) graph edges meet
(p), so at least (1,024-107=917) selected edges are disjoint from it.  For
such an edge (i),

\[
 M_i\circ M_p=0,
 \qquad
 M_i^2={D_{u(i)}+D_{v(i)}\over2}.
\tag{107.15}
\]

The latter squares lie in the 55-dimensional diagonal space.  Choose a
nonzero dependence (\sum_i b_iM_i^2=0) supported on disjoint edges, and
define the cokernel matrix by

\[
 B_{ip}=b_i,qquad B_{ab}=0\quad\hbox{otherwise}.
\tag{107.16}
\]

Equation (107.5) gives

\[
 g_i=2b_iM_i\circ M_p=0,qquad
 g_p=\sum_i b_iM_i^2=0,
\tag{107.17}
\]

and every other (g_q) is zero.  Hence (B\ne0) satisfies (107.6).  This is
an analytic rank defect, not bad conditioning.  Sparse edge bases and any
argument which treats their squares as generic are closed as r55 regularity
witnesses; a successful tuple must mix the matrix units densely.

### 107.3 Cubic symmetry and the 516-normal count do not force a defect

The obstruction in (107.17) uses the special Jordan multiplication.  There
is no obstruction from symmetric trilinearity or from the bare dimensions.
The following explicit comparison theorem is useful because its dimensions
are exactly ours.

Let (E=\mathbb R^n), (N=\mathbb R^m), (n=2k), and (m\ge k+1).  There is a
symmetric cubic tensor (T) on (E\oplus N) for which the derivative of

\[
 K_{ij}=T(e_i,e_i,e_j)
\tag{107.18}
\]

on the fixed-Gram Stiefel tangent is onto (\mathbb R^{n\times n}).  To
construct it, first set

\[
 T(e_i,e_i,e_i)=1,qquad
 T(e_i,e_j,e_l)=0
 \quad\hbox{for every other all-}(E)\hbox{ triple}.
\tag{107.19}
\]

Then a frame rotation (H_i=\sum_p\Omega_{pi}e_p) gives

\[
 \delta K_{ij}=\Omega_{ij}\quad(i\ne j),qquad
 \delta K_{ii}=0,
\tag{107.20}
\]

so rotations cover (\operatorname{Skew}_n) isomorphically.

It remains to cover (\operatorname{Sym}_n) with normal directions.  A
Walecki decomposition writes (K_{2k}) as (k-1) Hamilton cycles and one
perfect matching.  From each cycle remove a different edge incident to a
fixed vertex (v), and add a distinct loop.  Each resulting color class is a
spanning path plus an anchoring loop, whose unsigned incidence matrix is
nonsingular.  Put (k) further loops with the perfect matching, one at an
endpoint of each matched edge; its incidence matrix is blockwise
nonsingular.  The removed (k-1) star edges together with the final loop at
(v) form the last class, whose (k\times n) unsigned incidence matrix has
row rank (k).  These (k+1) classes partition every ordinary edge and every
loop, and every class has full row-rank unsigned incidence.

Let (c(i,j)) be the edge color and (\ell(i)) the loop color.  With normal
basis ((f_c)), set

\[
 T(e_i,e_j,f_c)=\mathbf1_{c=c(i,j)}\quad(i\ne j),
 \qquad
 T(e_i,e_i,f_c)=\epsilon\mathbf1_{c=\ell(i)}.
\tag{107.21}
\]

For (H_i=z_i\in N),

\[
 \delta K_{ii}=3\epsilon z_{i,\ell(i)},
\tag{107.22}
\]

\[
 \delta K_{ij}+\delta K_{ji}
 =2\{z_{i,c(i,j)}+z_{j,c(i,j)}\}
 +\epsilon\{z_{j,\ell(i)}+z_{i,\ell(j)}\}.
\tag{107.23}
\]

Choose a full row minor of the incidence matrix in each color, using
disjoint variables across colors.  Every diagonal row forces one factor of
(\epsilon); using an (\epsilon)-term from (107.23) costs an additional
factor.  Therefore the lowest-degree coefficient of the resulting
(\dim\operatorname{Sym}_n)-minor is

\[
 3^n2^{n(n-1)/2}\epsilon^n
 \prod_c\det I_c\ne0,
\tag{107.24}
\]

where the last class uses its nonzero (k)-minor.  Equations
(107.20)--(107.24) prove surjectivity.  At (n=1024), only (k+1=513)
normal colors are needed, below (m=516).  Thus a no-go for the r55 tuple
would have to be a special trace/Jordan identity; dimension and cubic
symmetry alone cannot supply one.

### 107.4 The exact 43+12 Jordan graph reduction

There is a decomposition with exactly the required tangent dimensions:

\[
 J=\mathcal A\oplus V,qquad
 \mathcal A=\operatorname{Sym}_{43}\oplus\operatorname{Sym}_{12},
 \qquad V=\mathbb R^{43\times12},
\tag{107.25}
\]

\[
 \dim\mathcal A=946+78=1,024=n,qquad
 \dim V=43\cdot12=516=m.
\tag{107.26}
\]

The grading satisfies
(\mathcal A\circ\mathcal A\subset\mathcal A),
(\mathcal A\circ V\subset V), and (V\circ V\subset\mathcal A).  For
(a=(P,Q)\in\mathcal A), write

\[
 a\cdot Y={PY+YQ\over2}.
\tag{107.27}
\]

Choose an orthonormal basis ((B_i)) of (\mathcal A) and a linear map
(X:\mathcal A\to V).  An exact analytic orthonormal graph frame is obtained
from

\[
 \widetilde M_i(\epsilon)=B_i+\epsilon X(B_i),qquad
 M(\epsilon)=\widetilde M(\epsilon)
 (I+\epsilon^2X^*X)^{-1/2}.
\tag{107.28}
\]

The first normal obstruction to the graph being a Jordan subalgebra is the
coboundary

\[
 \boxed{
 II_X(a,b)=a\cdot X(b)+b\cdot X(a)-X(a\circ b).}
\tag{107.29}
\]

Its kernel consists exactly of the mixed conjugation derivations

\[
 X_Z(P,Q)=ZQ-PZ,qquad Z\in\mathbb R^{43\times12},
\tag{107.30}
\]

of dimension 516.  This follows either from the Jordan Whitehead lemma or
directly by applying the derivation identity to the two central idempotents
and then to rank-one symmetric matrix units.

There are now exactly two finite minors to prove.  The base rotation map is

\[
 (\mathcal R_B\Omega)_{ij}
 =2\langle\Omega B_i,B_i\circ B_j\rangle
  +\langle B_i^2,\Omega B_j\rangle,
 \qquad\Omega\in\mathfrak{so}(\mathcal A).
\tag{107.31}
\]

Its kernel always contains

\[
 \operatorname{Der}(\mathcal A)
 =\mathfrak{so}_{43}\oplus\mathfrak{so}_{12},qquad
 \dim\operatorname{Der}(\mathcal A)=903+66=969.
\tag{107.32}
\]

The required first receipt is

\[
 \boxed{\operatorname{rank}\mathcal R_B
 ={n(n-1)\over2}-969=522,807.}
\tag{107.33}
\]

Let (P_{coker}) project away that base image.  Define the raw leading normal
matrix, after the tangential orthonormalization in (107.28), by

\[
\boxed{\begin{aligned}
 (\widetilde{\mathcal N}_{B,X}Z)_{ij}
 ={}&2\langle II_X(B_i,B_j),Z(B_i)\rangle\\
    &+\langle II_X(B_i,B_i),Z(B_j)\rangle,\\
 \mathcal N_{B,X}={}&P_{coker}\widetilde{\mathcal N}_{B,X}.
\end{aligned}}
\tag{107.34}
\]

Since (n^2-522,807=525,769), the second receipt is

\[
 \boxed{\operatorname{rank}\mathcal N_{B,X}=525,769.}
\tag{107.35}
\]

The normal domain modulo (107.30) has dimension

\[
 n\cdot516-516=527,868,
\tag{107.36}
\]

so (107.35) leaves exactly

\[
 527,868-525,769=2,099
\tag{107.37}
\]

nongauge directions.  The internal and mixed gauges total
(969+516=1,485), exactly the full conjugation gauge.

If one 522,807-minor in (107.33) and one 525,769-minor in (107.35) are
nonzero, a corresponding (n^2)-minor of the fixed-Gram derivative at
(M(\epsilon)) has a nonzero leading power of (\epsilon).  Hence every
sufficiently small generic nonzero (\epsilon) is the requested explicit
r55 regular tuple, and the local quotient fiber is exactly 2,099.  A useful
finite candidate takes (X) surjective, maps 516 basis vectors of
(\mathcal A) once onto a basis of (V), and maps another 508 onto repeated
copies.  Then (\ker X) has dimension 508.  A dense orthogonal basis (B),
not the coordinate matrix units, is essential.

### 107.5 The base minor is nonzero: an exact projector-basis theorem

The first of the two minors is in fact provable without a genericity
assumption.  For (J_p=\operatorname{Sym}_p), put

\[
 D_a=e_ae_a^T,\qquad
 P_{ab}={1\over2}(e_a+e_b)(e_a+e_b)^T\quad(a<b),
\tag{107.38}
\]

and (S_{ab}=E_{ab}+E_{ba}).  These are (p(p+1)/2) rank-one
idempotents and form a basis, since

\[
 S_{ab}=2P_{ab}-D_a-D_b.
\tag{107.39}
\]

Suppose (\Omega\in\mathfrak {so}(J_p)) is in the kernel of the base map
(107.31).  For any one of the basis idempotents (B), the equations over
all second indices, together with Frobenius self-adjointness of Jordan
multiplication and skew-adjointness of (\Omega), are equivalent to the
vector identity

\[
 0=2B\circ\Omega B-\Omega(B^2),\qquad
 \boxed{\Omega B=2B\circ\Omega B.}
\tag{107.40}
\]

Applying (107.40) first to (D_a) gives

\[
 \Omega D_a=\sum_{c\ne a}k_{ca}S_{ac}.
\tag{107.41}
\]

For (P_{ab}=vv^T), (107.40) says that (\Omega P_{ab}) is tangent to the
rank-one projector manifold: it has the form (vw^T+wv^T), with
(w\perp v).  In particular its ((a,a)) and ((b,b)) entries sum to zero.
Skew-adjointness against (D_a,D_b) identifies those entries as
(-k_{ba}) and (-k_{ab}), respectively.  Hence (k_{ab}=-k_{ba}).  If
(K=(k_{ca})) and (\delta_K A=KA-AK), then (K\in\mathfrak {so}_p) and
(\delta_K) agrees with (\Omega) on every (D_a).

Set (\Psi=\Omega-\delta_K).  It is still skew-adjoint, still obeys
(107.40), and annihilates all (D_a).  Skew-adjointness makes every
diagonal entry of (\Psi S_{ab}) zero.  The tangent equation at (P_{ab})
therefore forces

\[
 \Psi S_{ab}=\sum_{c\notin\{a,b\}}
 \lambda_{ab,c}(S_{ac}+S_{bc}).
\tag{107.42}
\]

For three distinct indices, skew-adjointness on the three pairs of edge
columns gives

\[
 \lambda_{ab,c}=-\lambda_{ac,b},\qquad
 \lambda_{ab,c}=-\lambda_{bc,a},\qquad
 \lambda_{ac,b}=-\lambda_{bc,a}.
\tag{107.43}
\]

The three signs are inconsistent unless every coefficient is zero.
Thus (\Psi=0), proving the exact kernel theorem

\[
 \boxed{\ker\mathcal R_{\{D,P\}}
 =\operatorname{Der}(\operatorname{Sym}_p)
 =\{A\mapsto KA-AK:K\in\mathfrak {so}_p\}.}
\tag{107.44}
\]

For the direct sum in (107.25), use the union of the two projector bases.
An off-diagonal skew block (L:J_{43}\to J_{12}) would have to satisfy,
for every basis idempotent (B\in J_{43}),

\[
 L(B)=2B\circ L(B)=0,
\tag{107.45}
\]

because the two simple ideals have zero cross-product.  Since the
idempotents span, (L=0); the reverse block vanishes identically as well.
Consequently

\[
 \boxed{\ker\mathcal R_B=\mathfrak {so}_{43}\oplus\mathfrak {so}_{12},
 \qquad \operatorname{rank}\mathcal R_B
 =523,776-969=522,807.}
\tag{107.46}
\]

This proves (107.33).  Orthonormality of the coordinate list is not needed:
at any independent frame, every fixed-Gram tangent splits uniquely into an
ambient skew motion plus a normal motion.  If an exactly constant Gram is
desired for the graph family, let (G_0=B^*B),
(G_\epsilon=G_0+\epsilon^2(XB)^*(XB)), and right-multiply the graph frame
by

\[
 R_\epsilon=G_\epsilon^{-1/2}
 (G_\epsilon^{1/2}G_0G_\epsilon^{1/2})^{1/2}G_\epsilon^{-1/2}.
\tag{107.47}
\]

Then (R_\epsilon^TG_\epsilon R_\epsilon=G_0) and
(R_\epsilon=I+O(\epsilon^2)), so this correction does not alter the
leading normal operator after quotienting by the base image.

There is also an exact conditioning receipt for this basis.  If (C_p) is
the unsigned vertex--edge incidence matrix of (K_p), its Gram matrix is

\[
 G_p=
 \begin{pmatrix}
 I&C_p/2\\ C_p^T/2&I/2+C_p^TC_p/4
 \end{pmatrix}
 =
 \begin{pmatrix}I&0\\C_p^T/2&I\end{pmatrix}
 \begin{pmatrix}I&0\\0&I/2\end{pmatrix}
 \begin{pmatrix}I&C_p/2\\0&I\end{pmatrix}.
\tag{107.48}
\]

Thus (\det G_p=2^{-\binom p2}).  Since
(C_pC_p^T=(p-2)I+\mathbf1\mathbf1^T), the nontrivial eigenvalue pairs are

\[
 \lambda_\pm(\sigma)={1\over2}
 \left[T_\sigma\pm\sqrt{T_\sigma^2-2}\right],\qquad
 T_\sigma={3\over2}+{\sigma^2\over4},
\tag{107.49}
\]

for (\sigma^2=2p-2) once and (\sigma^2=p-2) with multiplicity (p-1);
the remaining edge-null eigenvalues are (1/2).  In particular the frame
conditioning is explicit and only polynomial in (p), not a hidden
near-singularity.

### 107.6 A deterministic Vandermonde--Kronecker reduction of the one open minor

The remaining issue is now only (107.35).  It can be put in a much smaller
analytic family, with no unspecified ``generic dense basis.''  Start with
the projector basis (107.38), label its unit generating vectors by
(v_i^0), choose distinct positive rational numbers (\xi_i), and set

\[
 v_i(\tau)={v_i^0+\tau(1,\xi_i,\ldots,\xi_i^{p-1})^T
 \over\|v_i^0+\tau(1,\xi_i,\ldots,\xi_i^{p-1})^T\|},qquad
 B_i(\tau)=v_i(\tau)v_i(\tau)^T.
\tag{107.50}
\]

Use this construction separately for (p=43) and (p=12).  For every
sufficiently small positive (\tau), all matrix entries are dense, the
frame remains a basis, and the fixed nonzero base minor from (107.46)
remains nonzero.  This follows directly by continuity; (107.48)--(107.49)
give an explicit Gram margin.  Equivalently, one may use pure moment-curve
projectors with exponents from a Sidon set; their synthesis determinant is
a generalized Vandermonde determinant

\[
 \det[t_i^{e_j}]=s_\lambda(t_1,\ldots,t_s)
 \prod_{i<j}(t_j-t_i)\ne0,
\tag{107.51}
\]

for (0<t_1<\cdots<t_s), because all unordered exponent sums are distinct.

Let (\mathcal V(\tau)) be the block-diagonal symmetric-vectorization
matrix of the two bases, and choose the fixed ``twice-minus-eight''
surjection (F:\mathbb R^{1024}\to\mathbb R^{516}): its first 516 columns
are the coordinate vectors of (V), and its last 508 columns repeat the
first 508.  Define (X_F(B_i)=\operatorname{mat}(F_{:i})).  All ingredients
of the second minor then reduce to the following exact blocks:

\[
 c_{ij}=\mathcal V(\tau)^{-1}\operatorname{svec}(B_i\circ B_j),
\qquad
 h_{ij}=L_iF_{:j}+L_jF_{:i}-Fc_{ij},
\tag{107.52}
\]

where (h_{ij}=\operatorname{vec}II_{X_F}(B_i,B_j)) and

\[
 L_i={1\over2}(I_{12}\otimes B_i)\quad(B_i\in\operatorname{Sym}_{43}),
 \qquad
 L_i={1\over2}(B_i\otimes I_{43})\quad(B_i\in\operatorname{Sym}_{12}).
\tag{107.53}
\]

For a normal variation with columns (z_i), the entire raw matrix is simply

\[
 (\widetilde{\mathcal N}Z)_{ij}
 =2h_{ij}^Tz_i+h_{ii}^Tz_j.
\tag{107.54}
\]

Choose any 522,807-row pivot block (A(\tau)) of the now-certified base
image.  Writing a basis of that image as ((A^T,C^T)^T), an exact quotient
map is

\[
 Q(\tau)=(-C(\tau)A(\tau)^{-1},\ I),
\qquad Q(\tau)\mathcal R_{B(\tau)}=0.
\tag{107.55}
\]

Equations (107.50)--(107.55) turn (107.35) into nonvanishing of a maximal
minor of one explicit matrix made only of generalized-Vandermonde inverses,
rank-one Kronecker projectors, and the fixed zero--one matrix (F).  After
clearing (\det\mathcal V) and (\det A), it is a polynomial in the 1,024
scalar nodes and (\tau).  If a selected square minor is (H), its fully
analytic robustness receipt is

\[
 \sigma_{\min}(\mathcal N)
 \ge { |\det H|\over \|H\|_2^{525,768}},
\tag{107.56}
\]

with every norm on the right bounded explicitly by (107.49),
(\|L_i\|\le1/2), and the Vandermonde separation.  This is not yet a proof
that the polynomial is nonzero: the remaining exact task is to exhibit one
leading monomial (equivalently, one perfect matching after a superincreasing
specialization of the columns of (F)).  It does eliminate both the sparse
matrix-unit pathology and the previously unspecified dense-frame premise.

### 107.7 Exact symmetric/stabilizer splitting of the last quotient

The projector basis gives one further exact reduction.  Let
(r_{ij}=(\mathcal R_B\Omega)_{ij}) and project (r) to its antisymmetric
part.  Then

\[
 \boxed{\ker(P_-\mathcal R_B)=\operatorname{Der}(\mathcal A).}
\tag{107.57}
\]

Here is a direct proof.  Work first in (\operatorname{Sym}_p).  If
(r_{ij}=r_{ji}), the pair ((D_a,D_b)) shows that every diagonal entry of
(\Omega D_a) is zero.  Pairing (D_a) with a disjoint (P_{bc}) kills every
edge component of (\Omega D_a) not incident on (a).  Thus

\[
 \Omega D_a=\sum_{c\ne a}k_{ca}S_{ac}.
\tag{107.58}
\]

Substitution in the incident pair ((D_a,P_{ab})) gives
(k_{ab}=-k_{ba}).  Subtract the corresponding inner derivation and call
the residual (\Psi).  Then (\Psi D_a=0), and skew-adjointness permits the
notation

\[
 \Psi S_e=\sum_f t_{fe}S_f,qquad t_{fe}=-t_{ef}.
\tag{107.59}
\]

For disjoint edges (e,f), (P_e\circ P_f=0), and symmetry of (r) gives
(t_{ef}=0).  For the three edges (e,f,h) of a triangle, put
(x=t_{ef},y=t_{eh},z=t_{fh}).  Since for adjacent edges

\[
 P_e\circ P_f={1\over4}D_{e\cap f}
 +{1\over8}(S_e+S_f+S_h),
\tag{107.60}
\]

the three symmetry equations are

\[
 2x-y+z=0,qquad -x+2y-z=0,qquad x-y+2z=0.
\tag{107.61}
\]

Their coefficient determinant is (4), so (x=y=z=0).  Every pair of
distinct edges is either disjoint or lies in such a triangle; hence
(\Psi=0).  For the direct sum, a cross-ideal pair has zero Jordan product,
and (r_{ij}=r_{ji}) plus skew-adjointness kills the off-diagonal block.
This proves (107.57).

Consequently (P_-\mathcal R_B) has rank 522,807 inside the
523,776-dimensional antisymmetric target.  If

\[
 U=P_-\operatorname{im}\mathcal R_B\subset\operatorname{Skew}_n,
 \qquad \dim U=522,807,
\tag{107.62}
\]

then (P_-) is injective on the base image.  That image is the graph of a
linear map (S:U\to\operatorname{Sym}_n).  After choosing any complement
(W) of (U) in (\operatorname{Skew}_n), the exact target quotient is

\[
 \boxed{
 \operatorname{Mat}_n/\operatorname{im}\mathcal R_B
 \simeq \operatorname{Sym}_n\oplus W,qquad
 \dim W=523,776-522,807=969.}
\tag{107.63}
\]

Explicitly, if (a=u+w) is the (U\oplus W) decomposition of (P_-Y), the
quotient representative is ((P_+Y-Su,w)).  Thus the one normal receipt
(107.35) splits into two sharply sized tests:

\[
 \operatorname{rank}\mathcal N_X^{\rm sym}=524,800,
\qquad
 \operatorname{rank}\left(
 \mathcal N_X^{\rm stab}\big|_{\ker\mathcal N_X^{\rm sym}}
 \right)=969.
\tag{107.64}
\]

The effective normal domain has dimension 527,868.  Therefore, after the
first test in (107.64), its kernel has dimension only

\[
 527,868-524,800=3,068.
\tag{107.65}
\]

The entire residual Jordan-specific question is consequently a 969-row
map on 3,068 variables; a 969-by-969 nonzero minor leaves the predicted
(3,068-969=2,099) fiber.  The dense perturbation (107.50) preserves
(107.57) for all sufficiently small (\tau), since derivations remain in
the kernel and the certified rank is open.  This replaces the opaque
525,769-row determinant by a Walecki-sized symmetric coverage problem plus
one small stabilizer determinant.

### 107.8 A sharp near-regular-tournament certificate for symmetric coverage

There is a useful data-free sufficient condition for the first test in
(107.64).  Before the shear in (107.63), every diagonal row of the raw
symmetric normal map is supported on one copy (V_i), and every unordered
edge row (e=\{i,j\}) is supported on (V_i\oplus V_j).  Denote the
corresponding covectors by (\ell_i) and ((a_{e,i},a_{e,j})).  Choose a
near-regular tournament orientation of (K_{1024}); every outdegree is 511
or 512.  At vertex (i), require the at-most-513 covectors

\[
 \{\ell_i\}\cup\{a_{e,i}:e\text{ is oriented out of }i\}
\tag{107.66}
\]

to be independent in the 516-dimensional space (V_i^*).  Choose their
dual primal directions.  Across all vertices this selects exactly

\[
 1024+\binom{1024}{2}=524,800
\tag{107.67}
\]

columns.  With rows ordered by their owning tail, the owned coefficients
form the identity.  Put every response at the head of an edge, together
with the shear correction (S\pi_U P_-) from (107.63), into the leakage
matrix (E_T).  The selected square symmetric block is exactly

\[
 I+E_T.
\tag{107.68}
\]

Therefore a sharp spectral certificate and a convenient norm certificate
are

\[
 -1\notin\operatorname{spec}(E_T),
 \qquad\text{or, sufficiently,}\qquad
 \boxed{\|E_T\|_2<1.}
\tag{107.69}
\]

Under the latter condition,
(\sigma_{\min}(\mathcal N_X^{\rm sym})\ge1-\|E_T\|_2).
All covectors in (107.66), and the shear, are the explicit
Vandermonde--Kronecker expressions (107.52)--(107.55).  Thus (107.69) is a
finite analytic condition bound for the deterministic family, not a
numerical-rank premise.

The independence tests (107.66) alone are not enough.  On a directed
three-cycle, head leakage can produce a block with

\[
 \det(I+E_{\triangle})=1+\alpha\beta\gamma;
\tag{107.70}
\]

choosing (\alpha\beta\gamma=-1) makes it singular although every owned
covector is nonzero and independent.  This exact counterexample explains
why a bare full-spark or dimension argument cannot finish (107.64).  The
remaining constructive route is correspondingly precise: seek a
superincreasing Vandermonde specialization for which (E_T) is contractive,
then prove the separate 969-by-969 stabilizer leading monomial.

### 107.9 The 969 residual rows are an exact commutator moment map

The stabilizer block in (107.64) also has a closed coordinate-free form.
For (E\in\mathfrak h=\operatorname{Der}(\mathcal A)), define a target
stress (W^E) by expanding each derivation image in the chosen basis:

\[
 \sum_j W^E_{ij}B_j=E B_i.
\tag{107.71}
\]

These 969 stresses are linearly independent and lie in the cokernel of the
base map.  Indeed, with (J_i=2L_{B_i}-I),

\[
 \langle W^E,\mathcal R_B\Omega\rangle
 =\sum_i\langle J_i\Omega B_i,E B_i\rangle
 =\sum_i\langle\Omega B_i,J_iE B_i\rangle=0,
\tag{107.72}
\]

because a derivation is tangent to every projector:
(E(B_i^2)=2B_i\circ E B_i=E B_i).

Let (X_i=X(B_i)) and let (T) be the trace cubic.  Pairing the quadratic
graph term with (W^E) gives exactly

\[
\begin{aligned}
 q_E(X)
 ={}&\sum_i\{T(E B_i,X_i,X_i)
       +2T(B_i,X_i,X(E B_i))\}\\
 ={}&-2\sum_iT(B_i,X_i,\rho_E X_i),
 \qquad
 \rho_E X=E_VX-XE_{\mathcal A}.
\end{aligned}
\tag{107.73}
\]

The second equality is just infinitesimal invariance of (T) under the full
block-diagonal derivation (E).  On
(\mathcal X=\operatorname{Hom}(\mathcal A,V)), define the self-adjoint
block operator

\[
 (\mathcal L X)_i=L_{B_i}X_i.
\tag{107.74}
\]

Because the projector list is not orthonormal, (\rho_E) is Gram-skew but
need not be skew-adjoint for the raw coordinate inner product.  Let
(\rho_E^*) denote its adjoint for that coordinate inner product and put

\[
 \mathcal H_E=\mathcal L\rho_E+\rho_E^*\mathcal L.
\tag{107.75}
\]

Then

\[
 \boxed{
 q_E(X)=-\langle X,\mathcal H_EX\rangle,\qquad
 dq_E(X)[Z]=-2\langle Z,\mathcal H_EX\rangle.}
\tag{107.76}
\]

Thus the formerly opaque stabilizer determinant is the separating-vector
test for the 969-dimensional self-adjoint operator pencil

\[
 \mathscr G_X:\mathfrak h\longrightarrow\mathcal X,qquad
 E\longmapsto\mathcal H_EX.
\tag{107.77}
\]

After the symmetric rows have been eliminated, the exact remaining gate is

\[
 \boxed{\operatorname{rank}
 \left(P_{\ker\mathcal N_X^{\rm sym}}\mathscr G_X\right)=969.}
\tag{107.78}
\]

Expanding (107.76) on rectangular matrix units produces weighted versions
of the familiar moment-map commutators

\[
 XY^T-YX^T\in\mathfrak {so}_{43},qquad
 X^TY-Y^TX\in\mathfrak {so}_{12},
\tag{107.79}
\]

together with the explicit input-action terms (XE_{\mathcal A}).  In
particular, pairs ((E_{a1},E_{b1})) generate every (43)-skew matrix unit,
and pairs ((E_{1c},E_{1d})) generate every (12)-skew matrix unit.  A
block-triangular proof of (107.78) may therefore reserve 903 such row pairs
and 66 such column pairs before carrying out the tournament pivots.  The
only remaining obligation is to show that symmetric elimination does not
consume those reserved leading terms; no generic 969-dimensional
determinant need be guessed.

### 107.10 The raw 969 symmetrized-commutator minor is diagonal

The separating-vector part of (107.78) has an explicit witness.  Retain the
projector notation of (107.38), writing (\widehat D_c,\widehat P_{cd}) for
the (12)-block projectors, and define

\[
 X(D_a)=e_af_1^T,\quad X(P_{ab})=0,\qquad
 X(\widehat D_c)=e_1f_c^T,\quad X(\widehat P_{cd})=0.
\tag{107.80}
\]

Although (X(S_{ab})=-X(D_a)-X(D_b)) is nonzero, the input-action and
coordinate-adjoint terms combine with the output action exactly.  Put
(s_a=\mathbf1^TK e_a) and (t_c=\mathbf1^TL f_c).  Direct substitution in
(107.77) gives

\[
 (\mathcal H_KX)_{D_a}=-K e_a f_1^T+s_ae_af_1^T,\qquad
 (\mathcal H_LX)_{\widehat D_c}
 =-e_1(Lf_c)^T+t_ce_1f_c^T,
\tag{107.81}
\]

for (K\in\mathfrak {so}_{43}) and (L\in\mathfrak {so}_{12}).  The
(K)-action vanishes on the selected (12)-block components and the (L)-action
vanishes on the selected (43)-block components, because left and right
module actions commute.

For (a<b), pair the (K_{ab}=e_ae_b^T-e_be_a^T) row with the variation

\[
 Z^{ab}(D_a)=e_bf_1^T,qquad Z^{ab}(B_i)=0\ (B_i\ne D_a).
\tag{107.82}
\]

The correction in (107.81) is parallel to the anchor and is orthogonal to
the off-anchor variation in (107.82).  Therefore (107.81) gives value
(+1) on its own row and zero on every other
(K_{cd}) row.  The analogous variations
(\widehat Z^{cd}(\widehat D_c)=e_1f_d^T), (c<d), diagonalize the 66
(L_{cd}) rows.  Hence

\[
 \boxed{\operatorname{rank}\mathscr G_X=903+66=969,}
\tag{107.83}
\]

with a literal diagonal (+1) maximal minor.  This proves that the raw
stabilizer polynomial is nonzero and that there is no representation-
theoretic rank defect in the final 969 channels.

It does **not** by itself prove the projected condition (107.78).  Separate
nonvanishing of a symmetric minor and of (107.83) does not imply that their
row spaces are transverse.  The final Schur-survival obligation is exactly
to preserve these 969 columns after eliminating the 524,800 symmetric
rows--for example by a block-triangular tournament specialization with the
columns (107.82) reserved.  Irreducibility/Zariski openness alone cannot
replace that argument.

Therefore the binding status is now

\[
 \boxed{\dim\ker D(C,K21)/\text{gauge}\ge2,099,}
\tag{107.84}
\]

with equality conditional only on the single projected normal minor
(107.35), not on the base-rotation minor.

Graph-ready split:

- `NCQC_R55_ADJOINT_REGULARITY_TEST` -- **solved exact** by
  (107.5)--(107.8);
- `NCQC_R55_SQUARE_SCHUR_CERTIFICATE` -- **solved reduction**, deployed
  minors open in (107.12);
- `NCQC_R55_MATRIX_UNIT_WITNESS` -- **closed rank-defective** by
  (107.14)--(107.17);
- `NCQC_R55_DIMENSION_ONLY_DEFECT` -- **closed no-go as an argument**:
  the explicit comparison cubic (107.18)--(107.24) is surjective with only
  513 normal colors;
- `NCQC_R55_43_12_BASE_MINOR` -- **solved exact** by the projector-basis
  kernel theorem (107.38)--(107.46): rank 522,807;
- `NCQC_R55_43_12_GRAPH_WITNESS` -- **reduced to one exact open minor**,
  (107.35), with the deterministic Vandermonde--Kronecker form
  (107.50)--(107.55) and the exact 524,800 plus 969 split
  (107.63)--(107.65); the symmetric half has the sharp tournament gate
  (107.66)--(107.70), and the stabilizer half is the commutator gate
  (107.71)--(107.79), with the raw diagonal 969-minor certified by
  (107.80)--(107.83) and only Schur survival open;
- `NCQC_R55_FIBER_EXACT_2099` -- **still conditional**; only the lower
  bound (107.84) is currently certified.

## 108. Frozen stochastic tetrahedron contraction: the Rao--Blackwell
frontier, exact control variates, and sharp \(k=4/9\) limits

Section 105 identifies two one-product stochastic factorizations of the
physical action.  This section compares them at equal product count.  The
distinction between an ensemble-unbiased estimator and one acceptable
frozen feature is essential: unbiasedness alone is not a correlation
receipt for the one seed shipped in a card.

Throughout, \(M_a=M_a^T\), \(P=P^T\), and

\[
 \begin{aligned}
 t_a&=\operatorname{tr}(M_aPM_a)
     =\langle P,M_a^2\rangle_F,\\
 c_a&=q_at_a,\qquad
 C=\sum_aq_aM_a^2,\qquad
 S=\sum_ac_a=\operatorname{tr}(PC).
 \end{aligned} \tag{108.1}
\]

When \(P\succeq0\), every \(t_a\ge0\).  The aggregate \(C\) is nevertheless
indefinite unless the physical row \(q\), or the aggregate itself, has a
sign/PSD receipt.

### 108.1 Exact endpoint-row importance sampling

For frozen probabilities \(\pi_a>0\), draw \(A_s\sim\pi\) independently
and spend one exact triangular product on each selected endpoint row:

\[
 \widehat S_{\rm row}
 ={1\over k}\sum_{s=1}^k{c_{A_s}\over\pi_{A_s}}. \tag{108.2}
\]

Its first two moments are exactly

\[
 E\widehat S_{\rm row}=S,\qquad
 \boxed{
 \operatorname{Var}\widehat S_{\rm row}
 ={1\over k}\left(\sum_a{c_a^2\over\pi_a}-S^2\right).} \tag{108.3}
\]

Writing \(L=\sum_a|c_a|\) and \(p_a^*=|c_a|/L\),

\[
 \operatorname{Var}\widehat S_{\rm row}
 ={L^2\over k}
 \left\{1+\chi^2(p^*\Vert\pi)-\left({S\over L}\right)^2\right\}.
 \tag{108.4}
\]

Thus the formal optimum is

\[
 \boxed{\pi_a^*={|c_a|\over L},\qquad
 V_{\rm row}^*={L^2-S^2\over k}.} \tag{108.5}
\]

If all \(c_a\) have one sign, (108.5) has zero variance: every selected
row returns \(S\).  This is an algebraic optimum, not a deployable free
lunch, because computing its probabilities requires the very traces being
estimated.  Equation (108.4) quantifies exactly how a frozen proxy loses
that property.

There is a lawful minimax proxy.  If cheap receipts give
\(|c_a|\le w_a\), put \(W=\sum_aw_a\) and \(\pi_a=w_a/W\).  Then

\[
 \boxed{
 \operatorname{Var}\widehat S_{\rm row}
 \le {W^2-S^2\over k}.} \tag{108.6}
\]

Among iid row samplers using only the separate envelopes \(w_a\), this
choice minimizes the worst-case second moment
\(\sum_aw_a^2/\pi_a\).  A frozen certainty stratum \(D\) may be evaluated
exactly first.  Sampling the tail with the remaining \(k-|D|\) products
replaces \(S,c_a,L\) in (108.3)--(108.5) by their tail values; this is the
sharp way to protect a few visibly dominant rows.

The corpus objective changes the optimal probabilities.  For network
\(\nu\), let \(c_a^{(\nu)}\) be the row contribution and define the
complete-old projected contribution vector

\[
 v_a=\Pi_{\rm old}^{\perp}
       (c_a^{(1)},\ldots,c_a^{(N)})^T,\qquad v=\sum_av_a. \tag{108.7}
\]

Using the same frozen row draws across the corpus gives

\[
 \boxed{
 E\|\widehat v-v\|_2^2
 ={1\over k}\left(\sum_a{\|v_a\|_2^2\over\pi_a}
                       -\|v\|_2^2\right),\qquad
 \pi_a^{*,{\rm corpus}}
 ={\|v_a\|_2\over\sum_b\|v_b\|_2}.} \tag{108.8}
\]

Scalar-magnitude probabilities are therefore not optimal for the actual
board feature unless all network contribution vectors are collinear.
Probabilities fitted on the grading corpus are also not frozen; they need
an analytic envelope or an independent preregistration corpus.

### 108.2 Gate/random-feature estimators and Rao--Blackwell dominance

Let \(b\) be any exactly samplable factor with

\[
 E[bb^T]=P.
\]

The factor-batched contraction of (105.10) is simply

\[
 Z(b)=b^TCb=\sum_aq_a\|M_ab\|_2^2,\qquad EZ=S. \tag{108.9}
\]

For \(k\) independent one-product factors,

\[
 \boxed{
 \operatorname{Var}\widehat S_{\rm feat}
 ={1\over k}
 \left\langle C\otimes C,\,
 E[(bb^T-P)\otimes(bb^T-P)]\right\rangle.} \tag{108.10}
\]

Thus \(P=Ebb^T\) proves unbiasedness but says nothing about variance
without a fourth-moment receipt.

One may instead sample both a row and a factor.  Put
\(Y_a(b)=\|M_ab\|_2^2\), \(m_{2a}=E Y_a^2\), and

\[
 X(A,b)={q_A\over\pi_A}Y_A(b). \tag{108.11}
\]

Then

\[
 \boxed{\begin{aligned}
 EX&=S,\\
 \operatorname{Var}X
 &=\sum_a{q_a^2m_{2a}\over\pi_a}-S^2,\\
 \pi_a^{*,{\rm joint}}
 &={|q_a|\sqrt{m_{2a}}\over
          \sum_b|q_b|\sqrt{m_{2b}}}.
 \end{aligned}} \tag{108.12}
\]

Conditioning on the sampled row proves the exact Rao--Blackwell split

\[
 \boxed{
 \operatorname{Var}X
 =\operatorname{Var}\!\left({c_A\over\pi_A}\right)
  +\sum_a{q_a^2\over\pi_a}\operatorname{Var}Y_a.} \tag{108.13}
\]

Consequently, if an exact selected-row trace and a noisy selected-row
factor cost the same billed product, the exact endpoint-row estimator
strictly dominates.  The aggregate estimator (108.9) can still win:
one factor evaluates all rows together and retains the potentially
favorable cross-covariances
\(\operatorname{Cov}(Y_a,Y_b)\).  It must be compared through (108.10),
not through separate per-row variances.

For \(P^\sharp=P-cR\), let the \(P\)-factor and \(R\)-factor actions have
second moments \(m_{2P}\) and \(m_{2R}\).  Sampling component
\(\tau\in\{P,R\}\) with probabilities \(\theta_\tau\) and weights
\(1/\theta_P,-c/\theta_R\) is unbiased, with

\[
 E X^2={m_{2P}\over\theta_P}
       +{c^2m_{2R}\over\theta_R}. \tag{108.14}
\]

The sharp mixture is

\[
 \boxed{
 \theta_P^*={\sqrt{m_{2P}}\over
                  \sqrt{m_{2P}}+|c|\sqrt{m_{2R}}},
 \qquad
 \theta_R^*={|c|\sqrt{m_{2R}}\over
                  \sqrt{m_{2P}}+|c|\sqrt{m_{2R}}}.} \tag{108.15}
\]

The equal-probability signed mixture in (105.20a) is optimal only when
\(\sqrt{m_{2P}}=|c|\sqrt{m_{2R}}\).  If the linear-\(R\) action is
resident, evaluating it deterministically and randomizing only the
\(P\) part is its Rao--Blackwell improvement.

### 108.3 Gaussian/Rademacher Hutchinson is a special factor law

If a square-root action is available, take
\(b=P^{1/2}z\) and set

\[
 B=P^{1/2}CP^{1/2}. \tag{108.16}
\]

For standard Gaussian \(z\),

\[
 \boxed{
 \operatorname{Var}\widehat S_{\rm G}
 ={2\over k}\|B\|_F^2.} \tag{108.17}
\]

For iid Rademacher \(z\), in the chosen square-root basis,

\[
 \boxed{
 \operatorname{Var}\widehat S_{\rm R}
 ={2\over k}\|\operatorname{offdiag}B\|_F^2.} \tag{108.18}
\]

These are Hutchinson estimators of \(\operatorname{tr}B\).  A standard
trace probe applied separately inside a sampled row adds the nonnegative
conditional-variance term in (108.13); it is never preferable at equal
row-product cost.

If \(B\succeq0\), define

\[
 r_{\rm eff}(B)={(\operatorname{tr}B)^2\over\|B\|_F^2}\ge1.
 \tag{108.19}
\]

Both Gaussian and worst-basis Rademacher probes obey the sharp
dimension-free bound

\[
 \boxed{
 {\operatorname{Var}\widehat S\over S^2}
 \le {2\over k\,r_{\rm eff}(B)}
 \le {2\over k}.} \tag{108.20}
\]

The last constant is attained by a rank-one Gaussian quadratic form and
approached by a diffusely rotated rank-one Rademacher form.  It is the one
important universal exception to the no-guarantee results below.  It
requires a literal PSD aggregate after every subtraction; \(P\succeq0\)
alone is insufficient when \(q\) is signed.

For a frozen Gaussian seed, the PSD tail bound is also explicit.  With
probability at least \(1-2e^{-x}\),

\[
 |\widehat S-S|
 \le 2\|B\|_F\sqrt{x/k}
      +2\|B\|_{\rm op}\,x/k. \tag{108.21}
\]

For the native sign-gate factor, a corresponding claim needs a bound on
\(|Z-EZ|\) and its fourth law; Gaussian-Hutchinson concentration cannot
be imported merely because the latent gate is Gaussian.

### 108.4 Antithetic gates and resident old-row controls

For the zero-bias sign gate, \(b^-=\mathbf1-b\).  Put

\[
 m={1\over2}\mathbf1,\qquad c=b-m.
\]

The law of \(c\) is invariant under \(c\mapsto-c\).  Hence

\[
 \begin{aligned}
 Z(b)&=c^TCc+2m^TCc+m^TCm,\\
 {Z(b)+Z(b^-)\over2}&=c^TCc+m^TCm.
 \end{aligned} \tag{108.22}
\]

Odd/even symmetry makes the two random terms orthogonal, and therefore

\[
 \boxed{\begin{aligned}
 \operatorname{Var}Z
 &=\operatorname{Var}(c^TCc)
   +4m^TC\,\Sigma_c\,C m,\\
 \operatorname{Var}Z_{\rm anti}
 &=\operatorname{Var}(c^TCc),\qquad
 \Sigma_c=E[cc^T].
 \end{aligned}} \tag{108.23}
\]

This is an exact nonnegative variance reduction.  It need not cost two
products: evaluate the centered factor \(c\) once and add the resident
constant \(m^TCm\), if that constant action is already owned.  By
contrast, Gaussian or Rademacher antithetics \(b,-b\) give the identical
quadratic value and no reduction.

Let \(U=(U_1,\ldots,U_r)^T\) be same-seed old CP3/pair sample actions with
known mean \(u=EU\).  For any coefficient vector \(\theta\) frozen
independently of the evaluated samples,

\[
 Z_\theta=Z-\theta^T(U-u) \tag{108.24}
\]

is unbiased.  With

\[
 \Gamma=\operatorname{Cov}(U,U),\qquad
 g=\operatorname{Cov}(U,Z),
\]

the exact optimum and variance are

\[
 \boxed{
 \theta^*=\Gamma^\dagger g,\qquad
 \operatorname{Var}Z_{\theta^*}
 =\operatorname{Var}Z-g^T\Gamma^\dagger g.} \tag{108.25}
\]

Thus a resident row helps only through its measured or analytic
same-seed covariance.  For a Gaussian square-root probe, \(Z-EZ\) lies
in second Wiener chaos: a pure CP3/odd-chaos control has \(g=0\), whereas
a pair/even control can help.  After the sign-gate antithetic (108.22),
every odd control is again orthogonal by parity.  CP3 can help the
*unpaired* gate estimator only through its odd gate-chaos content.

The same theorem has a deterministic matrix form.  If resident matrices
\(B_j\) have known traces, estimate the trace of

\[
 B_{\rm res}=B-\sum_j\theta_jB_j
\]

and add back \(\sum_j\theta_j\operatorname{tr}B_j\).  Gaussian variance
is \(2\|B_{\rm res}\|_F^2/k\); Rademacher variance is
\(2\|\operatorname{offdiag}B_{\rm res}\|_F^2/k\).  Hence \(\theta\)
is the orthogonal projection in the corresponding Frobenius Gram
metric.  “Old CP3/pair is resident” is not itself a variance bound; the
residual Gram norm is the receipt.

There is also an exact row-sampling control.  If old row contributions
\(d_{ja}\) have known totals \(D_j=\sum_ad_{ja}\), then

\[
 \widehat S_{\theta,\pi}
 =\theta^TD+{1\over k}\sum_{s=1}^k
 {c_{A_s}-\theta^Td_{A_s}\over\pi_{A_s}} \tag{108.26}
\]

is unbiased, and for fixed \(\theta\) its optimal distribution is

\[
 \boxed{
 \pi_a\ \propto\
 |c_a-\theta^Td_a|.} \tag{108.27}
\]

Equations (108.25)--(108.27) are the sharp control-variate frontier.
Coefficients learned from the same probes without leave-one-block-out
independence need not preserve unbiasedness.

### 108.5 Correlation and the \(k=4/9\) product gates

Let \(s\) be the exact corpus action after complete-old projection,
\(\widehat s=s+e\), and

\[
 \nu={E\|e\|_2^2\over\|s\|_2^2}. \tag{108.28}
\]

Unbiasedness gives \(E\langle s,e\rangle=0\).  The exact ensemble
second-moment alignment with \(s\) is therefore

\[
 \rho_{\rm ens}={1\over\sqrt{1+\nu}}. \tag{108.29}
\]

If the exact action has target correlation \(\rho_t\), the corresponding
ensemble target alignment is

\[
 \boxed{
 \rho_{\rm target,ens}
 ={\rho_t\over\sqrt{1+\nu}},\qquad
 \nu\le\left({\rho_t\over0.8124}\right)^2-1} \tag{108.30}
\]

for the raw-\(10^{-8}\) screen.  This is an ensemble/RMS statement, not
a guarantee for one frozen seed.

In the ideal PSD Gaussian-Hutchinson case, (108.20) gives

\[
 \rho_{\rm ens}\ge
 \sqrt{{kr_{\rm eff}\over kr_{\rm eff}+2}}. \tag{108.31}
\]

At the worst \(r_{\rm eff}=1\),

\[
 k=4:\quad\rho_{\rm ens}\ge\sqrt{2/3}=0.8165\ldots,
 \qquad
 k=9:\quad\rho_{\rm ens}\ge3/\sqrt{11}=0.9045\ldots . \tag{108.32}
\]

Thus the utility-parity \(k=4\) ceiling clears \(0.8124\) only if the
*exact* physical action is essentially perfectly target-aligned:
\(\rho_t\ge0.8124\sqrt{3/2}\simeq0.995\) in the worst-rank case.
The \(k=9\) ceiling requires
\(\rho_t\ge0.8124\sqrt{11/9}\simeq0.898\).  More generally the exact
rank gate is

\[
 \boxed{
 r_{\rm eff}\ge
 {2\over k\{(\rho_t/0.8124)^2-1\}}.} \tag{108.33}
\]

All of (108.31)--(108.33) must be recomputed after complete-old
projection.  Projection can collapse \(\|s\|\) without proportionally
collapsing stochastic error, so a local PSD bound is not a corpus receipt.

For a literal frozen-seed statement, Markov already gives

\[
 \Pr\{\|e\|\le\varepsilon\|s\|\}
 \ge1-{\nu\over\varepsilon^2},\qquad
 \operatorname{corr}(s,s+e)\ge\sqrt{1-\varepsilon^2}. \tag{108.34}
\]

The \(10.35\%\) finite-repair fidelity gate from Section 104 corresponds
to \(\varepsilon=0.1035\), far stronger than the ensemble correlations
in (108.32).  Stochastic action noise, finite-repair error, and target
misalignment are three separate debits.

### 108.6 Sharp no-universal-guarantee constructions

First consider endpoint products.  After quotienting \(r\) independent
old controls, suppose \(d\) endpoint directions remain indistinguishable
to the frozen proxy.  Any iid probabilities have some
\(\pi_j\le1/d\).  If only row \(j\) contributes, (108.3) gives

\[
 \boxed{
 {\operatorname{Var}\widehat S\over S^2}
 \ge {d-1\over k}.} \tag{108.35}
\]

For any adaptive design observing at most \(k\) distinct product oracles,
some row has inclusion probability \(\iota_j\le k/d\).  Unbiased
estimation of a contribution hidden there has relative variance at least

\[
 \boxed{{1\over\iota_j}-1\ge {d\over k}-1.} \tag{108.36}
\]

The construction is realized by \(P=I\) and mutually orthogonal rank-one
\(M_a\), with the active coefficient chosen in the unobserved
control-orthogonal direction.  It is an oracle-model lower bound: a
separate algebraic shortcut that reveals the traces would change the
primitive and must be billed.

Second, signed Hutchinson has no relative guarantee.  For

\[
 B_\epsilon=
 \begin{pmatrix}\epsilon/2&1\\1&\epsilon/2\end{pmatrix},
 \qquad \operatorname{tr}B_\epsilon=\epsilon, \tag{108.37}
\]

a Rademacher probe returns
\(\epsilon+2z_1z_2\), so

\[
 {\operatorname{Var}\widehat S\over S^2}
 ={4\over k\epsilon^2}\longrightarrow\infty. \tag{108.38}
\]

The Gaussian variance has the same divergence.  Spectral positive/negative
parts realize \(B_\epsilon\) as a signed sum of squared physical slices,
so this is not outside (108.1).

Third, even the native zero-bias gate factor has no PSD dimension-free
bound near a correlation boundary.  Let two latent Gaussian coordinates
have correlation \(r\uparrow1\), let
\(b_i=\mathbf1_{\{G_i>0\}}\), \(v=(1,-1)^T\), and \(C=vv^T\succeq0\).
Then

\[
 Z=(b_1-b_2)^2,\qquad
 \delta:=EZ={\arccos r\over\pi},\qquad
 {\operatorname{Var}Z\over(EZ)^2}
 ={1-\delta\over\delta}\longrightarrow\infty. \tag{108.39}
\]

The complement antithetic has the identical value in this example.  A
determinant/orthant-probability margin or a control that literally owns
this disagreement mode is therefore necessary.

More abstractly, \(Ebb^T=P\) alone permits the rank-one rare factor
\(b=\delta^{-1/2}u\) with probability \(\delta\) and \(b=0\) otherwise.
Its relative quadratic variance is \(\delta^{-1}-1\).  A second-moment
factorization can never substitute for the fourth-moment gate.

### 108.7 Deployment theorem

At equal one-product samples, the sharp ordering is conditional:

1. residualize analytically against resident pair/even controls using
   (108.25) or (108.26);
2. if selected-row traces are exact, use their Rao--Blackwell endpoint
   sampler with corpus probabilities (108.8), not a nested trace probe;
3. if one gate factor evaluates all rows, use the aggregate estimator
   (108.9), complement-antithetic it by (108.22), and compare its exact
   residual fourth moment with (108.8);
4. use Gaussian/Rademacher Hutchinson only when the required square-root
   factor action is resident and the aggregate PSD/effective-rank receipt
   survives all subtractions;
5. accept \(k\le4\) for utility parity or \(k\le9\) for the optimistic
   leader bill only when the *post-old-projection* version of (108.30)
   and a frozen-seed tail gate such as (108.21)/(108.34) both pass.

There is no estimator family in this list with a universal \(k=4\) or
\(k=9\) guarantee for the signed physical tetrahedron.  The only
dimension-free exception is the literal PSD Gaussian/Rademacher
Hutchinson cone (108.20), and even its \(k=4\) margin is only
\(0.8165-0.8124\) before finite-repair, old-projection, and billing
debits.

Graph-ready split:

- Q211_TETRA_ROW_PPS -- **exact variance-optimal conditional**,
  (108.2)--(108.8);
- Q211_TETRA_GATE_FOURTH_MOMENT -- **exact**, (108.9)--(108.10);
- Q211_TETRA_RAO_BLACKWELL -- **exact dominance at equal selected-row
  cost**, (108.11)--(108.13);
- Q211_TETRA_SIGNED_MIXTURE -- **optimized**, (108.14)--(108.15);
- Q211_TETRA_HUTCHINSON -- **exact variance**, PSD exception
  (108.16)--(108.21);
- Q211_TETRA_ANTITHETIC -- **exact gate variance reduction**,
  (108.22)--(108.23);
- Q211_TETRA_OLD_CONTROL -- **exact projection frontier**,
  (108.24)--(108.27);
- Q211_TETRA_K4_K9_GATE -- **conditional only**, (108.28)--(108.34);
- Q211_TETRA_MINIMAX_NOGO -- **sharp oracle/signed/gate-boundary
  counterexamples**, (108.35)--(108.39);
- Q211_TETRA_VALUE -- **open** until the complete-old physical action
  and frozen corpus variance are known.

## 110. Deterministic physical-row quadrature: exact minimax radius,
state-subspace coresets, and the common \(k=76/4\) gate

Section 108 optimizes an unbiased random row sampler.  A deterministic
card needs a different theorem.  This section treats one exact physical
row action as an oracle query and permits the node rule to use only the
lawful state \((q,W,R,P)\), resident old rows, and previously queried
physical actions.  Target MSE residuals and truth moments are not
available to the design.

Let

\[
 c_a=q_a\operatorname{tr}(M_aPM_a),\qquad
 M_a=D_aRD_a,\qquad D_a=\operatorname{Diag}(w_a),\qquad
 S=\mathbf1^Tc. \tag{110.1}
\]

### 110.1 The exact envelope minimax theorem

A state-only absolute envelope is

\[
 \boxed{
 |c_a|\le U_a:=
 |q_a|\,\|P\|_{\rm op}\,\|M_a\|_F^2,\qquad
 \|M_a\|_F^2
 =(w_a^{\odot2})^T(R^{\odot2})w_a^{\odot2}.} \tag{110.2}
\]

If even the last quadratic form is too expensive to form for every row,
the cheaper but looser receipt

\[
 U_a^{\rm cheap}
 =|q_a|\,\|P\|_{\rm op}\,
   \|w_a\|_\infty^4\|R\|_F^2 \tag{110.3}
\]

is valid.  Proxy formation is not free merely because it is label-free;
its arithmetic and state must be billed.

More generally suppose the lawful state gives intervals

\[
 c_a\in[L_a,H_a],\qquad \Delta_a=H_a-L_a. \tag{110.4}
\]

After any queried set \(I\), the unobserved part of \(S\) lies in an
interval of width \(\sum_{a\notin I}\Delta_a\).  Its Chebyshev-center
estimate has worst-case absolute error one half that width.  Therefore
the exact deterministic minimax radius with at most \(k\) row queries is

\[
 \boxed{
 {\cal R}_k^{\rm env}
 ={1\over2}\sum_{j>k}\Delta_{(j)},} \tag{110.5}
\]

where the widths are sorted decreasingly and the \(k\) widest rows are
queried.  This is also an adaptive lower bound.  An adversary returns
interval midpoints on every queried row and chooses all remaining
endpoints only after the last query; observed values cannot reveal which
unqueried interval contains the error.

For symmetric envelopes \([-U_a,U_a]\),

\[
 {\cal R}_k^{\rm env}=\sum_{j>k}U_{(j)}. \tag{110.6}
\]

If \(P\succeq0\), then \(t_a=\operatorname{tr}(PM_a^2)\ge0\), so the
sign of \(c_a\) is the known sign of \(q_a\).  The intervals are
\([0,U_a]\) or \([-U_a,0]\), and the minimax midpoint radius is
\(\frac12\sum_{j>k}U_{(j)}\).  Without a positive lower bound on
\(|S|\), none of these absolute gates implies any relative-error or
correlation guarantee: lawful signed rows can cancel to arbitrary
precision.

For a panel, let \(U_a^{(\nu)}\) be state envelopes and define

\[
 \overline U_a=
 \left\|\bigl(U_a^{(1)},\ldots,U_a^{(N)}\bigr)\right\|_2. \tag{110.7}
\]

The common top-\(k\) rule minimizes the triangle-inequality certificate
\(\sum_{a\notin I}\overline U_a\).  It is not generally the exact
minimax vector design, but it is the strongest design available from
separate Euclidean envelopes alone.

### 110.2 Why affine \(r55\) makes queries cheap but not low-dimensional

Expanding a row action gives

\[
 t_a=\sum_{i,j,k}
 w_{ai}^2w_{aj}w_{ak}R_{ij}R_{ik}P_{jk}. \tag{110.8}
\]

It is a homogeneous quartic polynomial in the physical row \(w_a\).
If those rows are parameterized by an affine \(r\)-dimensional coordinate,
\(w_a=b+Ax_a\), its ambient feature space contains monomials of degree at
most four:

\[
 d_{\rm quartic}\le {r+4\choose4}. \tag{110.9}
\]

At \(r=55\),

\[
 {59\choose4}=455{,}126
\quad\text{(affine)},\qquad
 {58\choose4}=424{,}270
\quad\text{(homogeneous)}. \tag{110.10}
\]

This count alone is only an ambient bound, but full row rank has a simple
unmasked witness.  Take \(R=I\), vary diagonal \(P\), and choose
\(w_{ai}=\ell_i(x_a)\) for generic linear forms and generic
\(x_a\in\mathbb R^{55}\).  The action matrix is the degree-four polynomial
kernel
\[
 [\,\ell_i(x_a)^4\,]_{a,i},
\]
whose generic rank is
\(\min\{m,{58\choose4}\}=1024\).  Multiplication by nonzero known
\(q_a\) preserves it.  Strict collision sewing can reduce this witness,
but only its literal sewn rank is a receipt; affine rank alone supplies
no \(76\)-dimensional theorem.  Thus (105.22c)--(105.22d) gives a cheap
exact row oracle, not a small quadrature space.

### 110.3 Necessary and sufficient low-subspace escape

Let a label-free feature matrix

\[
 \Phi=(\phi_1^T,\ldots,\phi_m^T)^T\in\mathbb R^{m\times d},
 \qquad \operatorname{rank}\Phi=d, \tag{110.11}
\]

be computable from lawful state without querying all \(c_a\).  Assume
the admissible row-action family is

\[
 c=\Phi\theta. \tag{110.12}
\]

For a node set \(I\) and weights \(\omega\), the quadrature
\(\widehat S=\omega^Tc_I\) is exact for every \(\theta\) if and only if

\[
 \boxed{
 \Phi_I^T\omega=\Phi^T\mathbf1.} \tag{110.13}
\]

This moment equation is both the construction and the lower bound.  The
minimum number of signed-weight nodes is

\[
 k_{\min}
 =\min\{|I|:\Phi^T\mathbf1
                \in\operatorname{span}\{\phi_a:a\in I\}\}, \tag{110.14}
\]

which is generically \(d\).  If a constant column is included, the average
feature lies in the convex hull of the rows.  Caratheodory/Tchakaloff then
gives nonnegative weights summing to \(m\) on at most \(d\) nodes.

Conversely, if (110.13) fails, there is a
\(\theta_0\) with \((\Phi_I\theta_0)=0\) but
\(\mathbf1^T\Phi\theta_0\ne0\).  The zero response and any multiple of
\(\Phi\theta_0\) are indistinguishable on every queried node.  Hence no
adaptive algorithm with that support can have a finite uniform error
bound.  An adversary can return zeros while the algorithm pivots, so
adaptivity does not lower the generic \(d\)-query threshold.

A physical way for (110.12) to hold is a certified endpoint-matrix
subspace

\[
 P=\sum_{h=1}^d\theta_hQ_h,\qquad
 \Phi_{ah}=q_a\operatorname{tr}(M_aQ_hM_a). \tag{110.15}
\]

The \(Q_h\) must be resident or admit cheap factors; forming all columns
of \(\Phi\) by \(d\) undeclared dense contractions merely moves the cost.
If instead

\[
 P=\sum_h\theta_hQ_h+E,
\]

then \(c=\Phi\theta+r\) with the state-only row bound

\[
 |r_a|
 \le e_a:=|q_a|\,\|E\|_{\rm op}\,\|M_a\|_F^2. \tag{110.16}
\]

For any exact moment rule (110.13), extend its weights by zero and put

\[
 g=\mathbf1-E_I^T\omega,\qquad \Phi^Tg=0.
\]

The exact residual gates are

\[
 \boxed{
 |S-\widehat S|
 =|g^Tr|
 \le\sum_a|g_a|e_a
 \le\|g\|_2\|r\|_2.} \tag{110.17}
\]

This separates subspace algebra from approximation value.

### 110.4 Volume, leverage, and a deterministic robust rule

Put

\[
 \mu=\Phi^T\mathbf1,\qquad G=\Phi^T\Phi.
\]

For any \(k\)-node set with \(\Phi_I\) full column rank, the minimum-norm
exact weights are

\[
 \boxed{
 \omega_I=\Phi_I(\Phi_I^T\Phi_I)^{-1}\mu.} \tag{110.18}
\]

Their conditioning, rather than exactness, controls the residual
amplification in (110.17).  Size-\(k\) volume sampling assigns mass
proportional to \(\det(\Phi_I^T\Phi_I)\).  Its exact inverse-Gram identity
is

\[
 E_{\rm vol}[(\Phi_I^T\Phi_I)^{-1}]
 ={m-d+1\over k-d+1}\,G^{-1}. \tag{110.19}
\]

If the constant vector is in \(\operatorname{col}\Phi\), then
\(\mu^TG^{-1}\mu=m\) and
\(\|g\|_2^2=\|\omega_I\|_2^2-m\).  Consequently there exists a
deterministic \(k\)-node volume design with

\[
 \boxed{
 \|g\|_2^2
 \le {m(m-k)\over k-d+1}.} \tag{110.20}
\]

Conditional-expectation volume selection derandomizes this existence
bound using only \(\Phi\).  Strong rank-revealing QR or a max-volume
exchange rule is the practical deterministic pivot; it must report
\(\sigma_{\min}(\Phi_I)\), \(\|\omega_I\|\), and the literal bound
(110.17), not just a leverage plot.

The row leverage scores

\[
 \ell_a=\phi_a^TG^\dagger\phi_a,\qquad \sum_a\ell_a=d \tag{110.21}
\]

are the correct label-free pivot sensitivities.  Greedy residual fitting
to already observed \(c_a\) is lawful, but without an additional
smoothness/residual model it cannot improve (110.5) or (110.17);
unqueried orthogonal residuals remain adversarial.

### 110.5 Positive quadratic coresets and the signed cancellation debit

There is a second low-dimensional escape.  Suppose the admissible actions
have the positive quadratic form

\[
 c_a(\vartheta)=\|F_a\vartheta\|_2^2,\qquad
 G_F=\sum_aF_a^TF_a,\qquad \vartheta\in\mathbb R^d. \tag{110.22}
\]

A nonnegative weighted subset satisfying

\[
 (1-\varepsilon)G_F
 \preceq\sum_{a\in I}s_aF_a^TF_a
 \preceq(1+\varepsilon)G_F \tag{110.23}
\]

is a uniform relative coreset:

\[
 |\widehat S(\vartheta)-S(\vartheta)|
 \le\varepsilon S(\vartheta)
 \quad\text{for every }\vartheta. \tag{110.24}
\]

Whitened block leverage
\(\tau_a=\operatorname{tr}(F_aG_F^\dagger F_a^T)\) sums to \(d\).
Leverage sampling gives \(k=O(d\log(d/\delta)/\varepsilon^2)\);
conditional expectation or a BSS-type sparsifier gives a deterministic
support \(k=O(d/\varepsilon^2)\).  In the scalar-row BSS theorem, a
\(k>d\) support may be chosen with the explicit conservative distortion

\[
 \varepsilon_k
 \le2\sqrt{d/k}+d/k. \tag{110.25}
\]

For the physical action, (110.22) applies when \(q_a\ge0\) and the
allowed \(P=BB^T\) factors are restricted to a certified
\(d\)-dimensional operator domain, since

\[
 S(B)=\sum_aq_a\|M_aB\|_F^2. \tag{110.26}
\]

Generic \(B\in\mathbb R^{1024\times\rho}\) is not a small-\(d\) domain.
For signed \(q\), split \(S=S_+-S_-\) and sparsify both PSD Grams.  If
each side has relative error at most \(\varepsilon\), then

\[
 \boxed{
 {|\widehat S-S|\over|S|}
 \le\varepsilon\,
 {S_++S_-\over|S|}
 =\varepsilon\kappa_{\rm sign}.} \tag{110.27}
\]

Thus leverage does not cure physical cancellation; it merely moves the
same condition number into an explicit coreset gate.

### 110.6 A common frozen panel design

Let \(j\) index every panel network and active layer, with lawful feature
matrix \(\Phi^{(j)}\).  If one common weight vector is required, concatenate
the features by row:

\[
 \Psi=[\,\Phi^{(1)}\ \cdots\ \Phi^{(J)}\,],
 \qquad d_{\rm union}=\operatorname{rank}\Psi. \tag{110.28}
\]

A common quadrature is exact for the whole panel if and only if

\[
 \boxed{
 \Psi_I^T\omega=\Psi^T\mathbf1.} \tag{110.29}
\]

Generically this needs \(k\ge d_{\rm union}\).  If the common node set
may use state-specific weights \(\omega^{(j)}\), the weaker simultaneous
conditions are

\[
 \Phi_I^{(j)T}\omega^{(j)}
 =\Phi^{(j)T}\mathbf1\quad\text{for every }j. \tag{110.30}
\]

There is no theorem reducing their common support to
\(\max_jd_j\): different panel members can require disjoint pivot rows,
forcing as many as \(\sum_jd_j\).  This is the panel-common lower-bound
counterexample.

A lawful deterministic common design uses the group leverage

\[
 \ell_a^{\rm grp}
 =\sum_j\phi_a^{(j)T}
   (\Phi^{(j)T}\Phi^{(j)})^\dagger\phi_a^{(j)} \tag{110.31}
\]

and a fixed tie-broken group max-volume/log-determinant pivot.  Acceptance
requires, for every \(j\),

\[
 \operatorname{rank}\Phi_I^{(j)}=d_j,\qquad
 \sum_a|g_a^{(j)}|e_a^{(j)}
 \le\varepsilon_j\|s^{(j)}\|, \tag{110.32}
\]

with the corpus errors summed and complete-old projected before the
final correlation calculation.  A basis, pivot set, or stopping rule
chosen to improve panel MSE is target leakage, even if every matrix used
to evaluate the chosen rows is white-box.

### 110.7 The \(76\)-skinny/\(4\)-full decision gate

For ten active seams, a full-rank endpoint row costs ten full products
and the utility-parity ceiling is

\[
 \boxed{k_{\rm full}\le4.} \tag{110.33}
\]

On the literal affine factor \(R=UU^T\), \(r_R=55\), (105.22d) gives

\[
 C_{\rm skinny}={10\cdot55\over1024}k,
 \qquad
 \boxed{k_{\rm skinny}\le76} \tag{110.34}
\]

at the same optimistic utility ceiling.  These \(76\) queries are lawful
only for the certified affine-\(r55\) surface; a quadratic-AQ or
Gaussian-rest covariance restores full rank and the four-row gate.

The deterministic carrier passes only through one of the following:

1. the interval tail (110.5) itself meets the complete-old endpoint error
   gate;
2. a common exact state subspace satisfies (110.29)/(110.30) with
   \(d_{\rm union}\le76\) in the skinny circuit or \(\le4\) in the full
   circuit; or
3. an approximate volume/coreset rule reports (110.17) or (110.27) after
   all signed cancellation and panel-common conditioning.

If the exact physical feature has target correlation \(\rho_t\), a
deterministic relative quadrature error \(\varepsilon_{\rm quad}\) is
sufficient for the raw-\(10^{-8}\) screen only when

\[
 \boxed{
 \varepsilon_{\rm quad}
 \le{\rho_t-0.8124\over1+0.8124}.} \tag{110.35}
\]

Even at \(\rho_t=1\), this is the \(10.35\%\) post-old-projection gate.
At \(k=d\), exact moment matching has zero model-space error but no
redundancy; if the action is only approximate, (110.20) can be enormous.

The lawful/forbidden split is:

- lawful design state: \(q,W,R,P\), their certified affine factor,
  resident old CP3/pair matrices, analytic norms, and already billed
  queried \(c_a\);
- forbidden design signal: truth MSE residuals, truth planes, per-network
  gain, or any choice of basis/nodes/weights made after viewing the board
  target;
- grading labels may be used only after the node rule, all weights, and
  every stopping gate are frozen.

Graph-ready split:

- Q211_DET_ROW_ENVELOPE -- **sharp minimax**, (110.2)--(110.7);
- Q211_R55_QUARTIC_RANK -- **generic 1024-dimensional NO-GO**,
  (110.8)--(110.10);
- Q211_DET_SUBSPACE_QUADRATURE -- **necessary and sufficient**,
  (110.11)--(110.17);
- Q211_DET_VOLUME_RULE -- **exact conditional robust bound**,
  (110.18)--(110.21);
- Q211_DET_POSITIVE_CORESET -- **conditional PSD/operator-subspace
  escape**, (110.22)--(110.27);
- Q211_DET_PANEL_COMMON -- **exact union-rank/support gate**,
  (110.28)--(110.32);
- Q211_DET_K76_K4 -- **conditional utility gate**, (110.33)--(110.35);
- Q211_DET_VALUE -- **open** until a lawful common proxy has a certified
  post-old residual below (110.35).

## 111. Exact fourth-cumulant-to-variance response: why the transported
K22 face is the weak-correlation leading channel

The certified correction chain assigns most of its value to the variance
channel after plane errors are injected.  The local Gaussian Edgeworth
calculus identifies which transported H4 faces can create that response
and gives an analytic hierarchy independent of any fitted source amplitude.

### 111.1 Distributional derivative kernels

Let \(f(x)=x_+\), let \((X,Y)\) be standardized centered Gaussian with
correlation \(\rho\), and put

\[
 \mu=\phi(0)={1\over\sqrt{2\pi}},\qquad
 s=\sqrt{1-\rho^2}. \tag{111.1}
\]

For a small connected fourth-cumulant perturbation, Gaussian integration by
parts gives

\[
 \delta E[h(X,Y)]
 =\sum_{p+q=4}{\kappa_{pq}\over p!\,q!}
 E[\partial_X^p\partial_Y^q h(X,Y)]. \tag{111.2}
\]

The three required boundary kernels are exactly

\[
 \boxed{\begin{aligned}
 E[\delta(X)\delta(Y)]&={\mu^2\over s},\\
 E[\delta'(X)\mathbf1_{\{Y>0\}}]&=-{\mu^2\rho\over s},\\
 E[\delta''(X)Y_+]&={\mu^2(2\rho^2-1)\over s}.
 \end{aligned}} \tag{111.3}
\]

For example,
\[
 \int_0^\infty\phi_\rho(x,y)\,dy
 =\phi(x)\Phi\!\left({\rho x\over s}\right),
\]
so the second line is minus its derivative at zero.  For the third line,
\[
 E[Y_+\mid X=x]
 =s\phi(\rho x/s)+\rho x\Phi(\rho x/s),
\]
and differentiating the product with \(\phi(x)\) twice gives the displayed
coefficient.  Thus (111.3) is valid without a small-\(\rho\) expansion.

### 111.2 Exact centered covariance response

Since a mixed cumulant does not alter either marginal law, only the 40 and
04 terms require mean subtraction.  The marginal mean response is

\[
 \delta E[X_+]={\kappa_{40}\over24}E[\delta''(X)]
 =-{\mu\kappa_{40}\over24}. \tag{111.4}
\]

Combining (111.2)--(111.4), the first-order centered covariance response is

\[
\boxed{\begin{aligned}
 \delta\operatorname{Cov}(X_+,Y_+)
={}&{\mu^2\over24}
 \left\{{2\rho^2-1\over s}+1\right\}
 (\kappa_{40}+\kappa_{04})\\
 &-{\mu^2\rho\over6s}(\kappa_{31}+\kappa_{13})
 +{\mu^2\over4s}\kappa_{22}.
\end{aligned}} \tag{111.5}
\]

The factorial weights \(1/24,1/6,1/4\) are precisely the K4, K31, K22
consumer weights.  Separately, because

\[
 (x_+^2)^{(4)}=2\delta'(x),\qquad E[\delta'(X)]=0,
\]

the raw second moment has no first-order marginal K4 response at zero
bias, but the mean subtraction does.  Hence

\[
 \boxed{\delta\operatorname{Var}(X_+)
 ={\mu^2\over12}\kappa_{40}
 ={1\over24\pi}\kappa_{40}.} \tag{111.6}
\]

### 111.3 The weak-correlation hierarchy

At \(\rho=0\), (111.5) reduces exactly to

\[
 \boxed{
 \delta\operatorname{Cov}(X_+,Y_+)
 ={1\over8\pi}\kappa_{22};}
 \tag{111.7}
\]

the K31 and cross-K4 coefficients vanish.  Relative to the K22 coefficient,
the absolute kernel ratios at general \(\rho\) are

\[
 {K31\over K22}={2|\rho|\over3},\qquad
 {K40_{\rm cross}\over K22}
 ={1\over6}|\,2\rho^2-1+s\,|. \tag{111.8}
\]

The second ratio is \(\rho^2/4+O(\rho^4)\).  The diagonal K4-to-variance
coefficient in (111.6) is one third of the zero-correlation K22
coefficient.  These are coefficient statements, not value statements:
the actual cumulant amplitudes and Bellman costates may reverse the order.

### 111.4 Consequence for the q211 provider

For a strict q211 birth tensor \(T\) followed by a dense row map \(W\), the
visible faces are

\[
\begin{aligned}
 \delta k_{4,a}&=T(w_a,w_a,w_a,w_a),\\
 \delta K_{31,ab}&=T(w_a,w_a,w_a,w_b),\\
 \delta K_{22,ab}&=T(w_a,w_a,w_b,w_b).
\end{aligned} \tag{111.9}
\]

Equations (111.5)--(111.7) prove that a q211 source can feed the observed
variance/covariance correction after only one dense transport.  They also
show that a diagonal-K4-only audit is incomplete for the full covariance
Bellman action: in a weakly correlated post-transport frame the K22 face is
the leading local coefficient.

Therefore Section 105's scalar \(q_a\) route is exact for the declared
diagonal K4/variance endpoint row, but it does not establish ownership,
cost, or value of a dense K22/K31 endpoint.  A full candidate must either

1. prove that the complete-old Bellman row projects the new source onto the
   diagonal K4 branch;
2. include the synchronized K22/K31 actions and their target ranks; or
3. bound the omitted response using (111.5) and the actual source/costate
   norms.

This is the exact physics link to the certified \(dv\) attribution.  It is
also a steering correction: optimize the transported face action, not the
untransported q211 tensor norm or Waring rank.

Graph-ready split:

- H4_TO_RELU_COVARIANCE_KERNEL -- **solved exact**, (111.2)--(111.5);
- K4_TO_RELU_VARIANCE_KERNEL -- **solved exact**, (111.6);
- H4_WEAK_CORRELATION_K22_LEAD -- **solved coefficient hierarchy**,
  (111.7)--(111.8);
- Q211_ONE_TRANSITION_VARIANCE_MECHANISM -- **solved exact typing**,
  (111.9);
- Q211_DIAGONAL_K4_ONLY_COMPLETENESS -- **NO without a Bellman projection
  receipt**;
- Q211_K22_K31_ENDPOINT_RANK -- **open and separately billed**;
- Q211_DV_VALUE -- **open** until old-residualized transported face
  alignment is established.

## 112. Pulling a q211 birth into the leading K22 Bellman row: exact
spectral circuit, generic rank, and weak-correlation escape

Section 111 identifies K22 as the leading local coefficient at weak
post-transport correlation.  This section contracts that face with its
physical symmetric Bellman costate and shows exactly when it is cheaper
than a full q211 carrier.

### 112.1 Raw symmetric pullback

Let \(K_{i;jk}=K_{i;kj}\) be a strict q211 source.  Fix the raw ordered
normalization by

\[
\begin{aligned}
 T_K(x,x,y,y)=\sum_{i,j,k}K_{i;jk}\{&
 2x_i^2y_jy_k+2y_i^2x_jx_k\\
 &+4x_iy_i(x_jy_k+x_ky_j)\}.
\end{aligned} \tag{112.1}
\]

A normalized \(\operatorname{Sym}_4\) or deployed \(1/4!\) convention
rescales every formula below by the same known constant.  Let
\(\Lambda=\Lambda^T\) be the synchronized K22 costate after complete-old
projection, and let \(w_a^T\) be row \(a\) of the dense transition \(W\).
Using symmetry in \(a,b\) and \(j,k\),

\[
\boxed{\begin{aligned}
 S_{22}
={}&4\sum_{a,b,i,j,k}\Lambda_{ab}K_{i;jk}
 W_{ai}^2W_{bj}W_{bk}\\
 &+8\sum_{a,b,i,j,k}\Lambda_{ab}K_{i;jk}
 W_{ai}W_{bi}W_{aj}W_{bk}.
\end{aligned}} \tag{112.2}
\]

This is the exact pullback of the K22 row.  It is distinct from the
diagonal K4 row (105.1), even when both ultimately feed the same scalar
endpoint.

### 112.2 Exact three-product circuit per costate eigenmode

Specialize to the broad Price/CP3 kernel

\[
 K_{i;jk}=R_{ij}R_{ik}P_{jk}, \tag{112.3}
\]

and take a signed spectral factorization

\[
 \Lambda=\sum_{t=1}^{r_\Lambda}\lambda_ts_ts_t^T. \tag{112.4}
\]

For each factor define

\[
\begin{aligned}
 A_t&=(W^{\odot2})^Ts_t,\\
 B_t&=W^TD_{s_t}W,\\
 G_t&=RD_{A_t}R,\\
 C_t&=B_t\odot R.
\end{aligned} \tag{112.5}
\]

The two lines of (112.2) collapse exactly to

\[
\boxed{
 S_{22}
 =4\sum_t\lambda_t\langle B_t\odot P,G_t\rangle_F
 +8\sum_t\lambda_t\langle P,C_t^TC_t\rangle_F.}
 \tag{112.6}
\]

Indeed, the first factorized anchor/leaf coefficient is
\[
 \sum_{a,b}s_{ta}s_{tb}W_{ai}^2W_{bj}W_{bk}
 =(A_t)_i(B_t)_{jk},
\]
while the crossed coefficient is
\[
 \sum_{a,b}s_{ta}s_{tb}W_{ai}W_{bi}W_{aj}W_{bk}
 =(B_t)_{ij}(B_t)_{ik}.
\]

With no resident products, (112.6) needs exactly the three generic dense
products

\[
 W^TD_{s_t}W,\qquad RD_{A_t}R,\qquad C_t^TC_t \tag{112.7}
\]

per live spectral factor, plus elementwise contractions.  This is an
attained circuit, not a claim that three is an unconditional algebraic
lower bound.  A literal resident \(B_t\), \(G_t\), or \(C_t^TC_t\) removes
only its matching product at the same cut and mask.

### 112.3 Why the local K22 kernel is generically full rank

Let \(\Lambda_C\) be the downstream covariance costate before the local
zero-bias Edgeworth coefficient.  From (111.5), its K22 row is

\[
 \Lambda_{\rm eff}
 ={\mu^2\over4}\,
 \Lambda_C\odot D(R),\qquad
 D(R)_{ab}={1\over\sqrt{1-R_{ab}^2}}
 \quad(a\ne b). \tag{112.8}
\]

Even if \(\Lambda_C=uu^T\) is rank one, the strict matrix is generically
full rank.  At the witness \(R=I\), its off-diagonal kernel is
\(\mathbf1\mathbf1^T-I\); therefore

\[
 \operatorname{Diag}(u)(\mathbf1\mathbf1^T-I)
 \operatorname{Diag}(u) \tag{112.9}
\]

has rank \(n\) when every \(u_a\ne0\).  The nonzero determinant persists
on an open set.  Hence a low-rank covariance costate, a scalar endpoint,
or weak source rank alone does not prove low \(r_\Lambda\).

Strict masking is important here: deleting a diagonal can turn a rank-one
matrix into full rank.  A cheap low-rank representative is lawful only if
the removed diagonal is sewn through the literal K4/old collision row.

### 112.4 Weak-correlation quotient approximation

Suppose all live off-diagonal correlations obey \(|R_{ab}|\le\rho_*<1\).
Then

\[
 D(R)=\mathbf1\mathbf1^T+E,\qquad
 |E_{ab}|\le
 \delta_D:=(1-\rho_*^2)^{-1/2}-1. \tag{112.10}
\]

Before the strict diagonal mask,

\[
 \boxed{
 \|\Lambda_C\odot E\|_F
 \le\delta_D\|\Lambda_C\|_F.}
 \tag{112.11}
\]

Thus a rank-\(r_C\) factorization of the *full* covariance costate gives a
rank-\(r_C\) leading representative for (112.8), with the collision
diagonal handled separately and the nonlinear tail charged by (112.11).
For small \(\rho_*\),
\[
 \delta_D={\rho_*^2\over2}+O(\rho_*^4). \tag{112.12}
\]

This is the only analytic low-rank escape supplied by weak correlation.
It remains endpoint-weighted: after applying the source action, the
resulting corpus error must be projected against the complete old span and
meet the raw-value margin.  The K31 row has a separate
\(\rho/\sqrt{1-\rho^2}\) coefficient and cannot be hidden in (112.11).

### 112.5 Cost and decision gate

Across \(D_s\) active seams, a rank-\(r_\Lambda\) exact circuit costs

\[
 \boxed{
 C_{22}=3D_sr_\Lambda-C_{\rm literal\ reuse}
 +C_{\rm elem}+C_{\rm hook}.}
 \tag{112.13}
\]

For ten seams with no reuse, one rank already costs 30 full
matmul-equivalents.  At \(0.161\%\) relative utilization per f32
equivalent, that is about \(4.83\%\) per rank before hooks.  Co-firing with
the \(u=0.5629\) pack permits only \(r_\Lambda=1\) at utility parity, or at
most \(r_\Lambda=3\) under the optimistic raw-\(10^{-8}\) leader bill.
One or two literally resident products per factor change these ceilings;
shared shapes or algebraic reconstructibility do not.

The lawful hierarchy is:

1. pull back the complete-old K22 costate and measure/prove its actual
   spectral/Schur rank;
2. use (112.6) exactly if (112.13) passes;
3. otherwise use the weak-correlation representative only when the
   collision sewing and (112.11) endpoint error pass; and
4. jointly include or bound K31 and diagonal K4 through (111.5).

Graph-ready split:

- Q211_K22_PHYSICAL_PULLBACK -- **solved exact**, (112.1)--(112.2);
- Q211_K22_THREE_PRODUCT_CIRCUIT -- **solved attained**, (112.3)--(112.7);
- Q211_K22_LOCAL_KERNEL_RANK -- **generic full-rank NO-GO**, (112.8)--
  (112.9);
- Q211_K22_STRICT_MASK_RANK -- **collision sewing required**;
- Q211_K22_WEAK_CORRELATION_ESCAPE -- **conditional**, (112.10)--
  (112.12);
- Q211_K22_CAP -- **rank one only at utility parity without reuse**,
  (112.13);
- Q211_K22_VALUE -- **open** until transported old-residualized alignment
  and K31/K4 completion are certified.

## 113. CP4-seeded covariant AQ63 cut: exact joint tangent, K21
surjectivity, and the under-49 circuit boundary

The live CP4 frame removes the cold-orientation objection, but three
logically different questions must not be conflated:

1. can it define a lawful AQ covariance factor;
2. is the covariance-preserving AQ tangent onto the oriented K21 target
   (and, if required, the independent K22 future row); and
3. can the factor and section be acquired within the suffix bill?

This section gives exact answers to the first two and a sharp circuit
boundary for the third.  Throughout \(n=1024,\ r=63\), and

\[
 e=\dim\operatorname{Sym}_r={r(r+1)\over2}=2016,\qquad
 d=r+e=2079. \tag{113.1}
\]

### 113.1 A lawful joint H1--H2 anchor

Write the selected nonzero CP4 rays in scale-canonical form

\[
 g_a=|\lambda_a|^{1/4}f_a
\]

and select 63 rays by a neuron-permutation- and CP-gauge-invariant score with
a strict selection gap.  Let \(F\in\mathbb R^{n\times r}\) contain these
rays, put

\[
 P_{i,:}=\operatorname{svec}(f_if_i^T),\qquad
 G=FF^T,
\]

and, for nonzero invariant scalars \(\alpha,\beta\), define the joint live
feature frame

\[
 H_{\alpha,\beta}=[\,\alpha F,\ \beta P\,]\in\mathbb R^{n\times d}.
 \tag{113.2}
\]

The Frobenius-isometric svec convention gives

\[
 \boxed{
 K_H:=H_{\alpha,\beta}H_{\alpha,\beta}^T
 =\alpha^2G+\beta^2G^{\odot2}.}
 \tag{113.3}
\]

Assume \(K_H\succ0\) and let \(C\succ0\) be the synchronized cut covariance.
The row-polar anchor

\[
 \boxed{
 S_0=C^{1/2}K_H^{-1/2}H_{\alpha,\beta}
 =[\,U_0,\ X_0/\sqrt2\,]}
 \tag{113.4}
\]

satisfies

\[
 U_0U_0^T+\frac12X_0X_0^T=C. \tag{113.5}
\]

After CP scale canonicalization the remaining ray gauge is a signed
permutation \(R\), with orthogonal feature action
\(\rho(R)=R\oplus\operatorname{Sym}^2(R)\).  Under a neuron permutation
\(\Pi\),

\[
 H'=\Pi H\rho(R),\quad K_H'=\Pi K_H\Pi^T,\quad C'=\Pi C\Pi^T.
\]

Uniqueness of the PSD roots in (113.4) therefore gives

\[
 S_0'=\Pi S_0\rho(R). \tag{113.6}
\]

Thus (113.4) is a globally equivariant covariance anchor on its strict
rank/selection domain.  It is only the initial point: exact K21 matching
requires a joint covariance-constrained move of \(U\) and \(X\), not merely
a different scalar \(\alpha/\beta\).

There is a cheaper but conditional factorization.  If the neurons admit a
unique permutation-equivariant total order from invariant row keys, apply
that order to both \(C\) and \(K_H\), take their Cholesky factors
\(C=L_CL_C^T,\ K_H=L_KL_K^T\), and undo the order.  Then

\[
 \boxed{S_0=L_CL_K^{-1}H_{\alpha,\beta}} \tag{113.7}
\]

also obeys (113.5) and (113.6).  Without the shared canonical order,
Cholesky is not neuron-permutation covariant.  Ties are therefore a literal
failure gate, not a label-based tie-break opportunity.

### 113.2 Exact support dimension includes the support stabilizer

Let \(E\subseteq\operatorname{Sym}_r\) be an allowed H2 support of dimension
\(p\).  At a coefficient bank \(\mathbf A=(A_1,\ldots,A_n)\), define the
pointwise infinitesimal gauge intersection

\[
 \mathfrak g_{E,\mathbf A}=
 \{\Omega\in\mathfrak{so}_r:[A_i,\Omega]\in E\ \forall i\},\qquad
 g_{E,\mathbf A}=\dim\mathfrak g_{E,\mathbf A}. \tag{113.8}
\]

Here \(U\) is assumed to have full column rank, so this is also the gauge
orbit dimension.  Otherwise \(g_{E,\mathbf A}\) in every count below means
the dimension of the image of
\(\Omega\mapsto\{(-\Omega u_i,A_i\Omega-\Omega A_i)\}_i\), not merely the
dimension of the admissible Lie algebra.

More generally, if
\[
 Z=[\,U,\ X/\sqrt2\,]\in\mathbb R^{n\times(r+p)}
\quad\hbox{has rank }k,
\]
then the exact fixed-covariance tangent dimension is

\[
 \boxed{\dim\ker DC=n(r+p-k)+{k(k-1)\over2}.} \tag{113.9}
\]

Indeed \(DC[\dot Z]=\dot ZZ^T+Z\dot Z^T\) has rank
\(nk-k(k-1)/2\).  At a regular full-row-rank covariance point, \(k=n\),
so (113.9) becomes

\[
 n(r+p)-{n(n+1)\over2}. \tag{113.9a}
\]

The gauge directions

\[
 \delta u_i=-\Omega u_i,\qquad
 \delta A_i=A_i\Omega-\Omega A_i \tag{113.10}
\]

lie in both the covariance and every AQ-moment kernel whenever they remain
inside the support.  Hence the regular post-K21 physical fiber is

\[
\boxed{
 f_E=n(r+p)-{n(n+1)\over2}-n^2-g_{E,\mathbf A}.}
\tag{113.11}
\]

This stabilizer term matters.  For the full support
\(E=\operatorname{Sym}_r\),

\[
 p=2016,\quad g_{E,\mathbf A}={63\cdot62\over2}=1953,\quad
 f_E=553{,}567. \tag{113.12}
\]

For the off-diagonal coordinate support \(E_{\rm off}\), \(p=1953\) but
\(g_{E,\mathbf A}=0\) for a generic bank.  Indeed

\[
 [A,\Omega]_{ii}=-2\sum_{j\ne i}A_{ij}\Omega_{ij};
\]

requiring this to vanish for a generic sufficiently rich collection of
off-diagonal \(A_i\) forces \(\Omega=0\); failure of that finite linear
rank test is an explicit exceptional-bank gate.  Therefore, generically,

\[
 \boxed{f_{E_{\rm off}}=491{,}008.} \tag{113.13}
\]

For a generic gauge-pinned support (\(g_{E,\mathbf A}=0\)), weak K21 matching requires

\[
 p\ge
 \left\lceil {3n+1\over2}-r\right\rceil
 =1474. \tag{113.14}
\]

If the AQ state must also replace the generic independent off-diagonal K22
future row of dimension \(\binom n2=523{,}776\), (113.11) instead requires

\[
 p\ge 2n-r=1985\qquad(g_{E,\mathbf A}=0). \tag{113.15}
\]

The fixed supports invariant under permutation of the 63 selected CP rays
still have only the sizes

\[
 0,\quad63,\quad1953,\quad2016. \tag{113.16}
\]

Thus the clean fixed-support strong replacement must use all 2016 H2
coordinates: the off-diagonal orbit misses the generic K22 target by exactly

\[
 523{,}776-491{,}008=32{,}768, \tag{113.17}
\]

whereas the full support has the exact 29,791-dimensional margin

\[
 553{,}567-523{,}776=29{,}791. \tag{113.18}
\]

Equations (113.14)--(113.15) refine a count which subtracts the full
\(O(63)\) gauge even after imposing a non-\(O(63)\)-invariant support.  A
generic data-dependent support has zero continuous stabilizer; the strong
integer floor is 1985, not 1987.  The fixed-orbit conclusion (full Sym2)
does not change.

The generic K22 rank must itself be target-visible.  If \(\mathcal B_{22}\)
is the complete-old K22-to-endpoint Bellman map from Section 112, define

\[
 q_{22}^{\rm eff}
 =\operatorname{rank}\!\left(
 \mathcal B_{22}\,DQ_{22}\big|_{\ker D(C,K21)}
 \right). \tag{113.19}
\]

The exact target-conditioned requirement is \(f_E\ge q_{22}^{\rm eff}\),
not automatically \(f_E\ge\binom n2\).  Retaining CP4/Q24 separately sets
this AQ replacement burden to zero but incurs its own suffix transport/read
and may not claim the same value twice.

### 113.3 Complete joint K21 differential and dual iff gate

For variations \(v_i=\delta u_i\) and \(H_i=\delta A_i\),

\[
\boxed{\begin{aligned}
 \delta C_{ij}={}&
 v_i^Tu_j+u_i^Tv_j
+{1\over2}\operatorname{tr}(H_iA_j+A_iH_j),\\
 \delta K^{21}_{ij}={}&
2v_i^T(A_ju_i+A_iu_j)+2v_j^TA_iu_i\\
&+\left\langle H_i,
 A_iA_j+A_jA_i+u_iu_j^T+u_ju_i^T\right\rangle_F\\
&+\left\langle H_j,A_i^2+u_iu_i^T\right\rangle_F .
\end{aligned}}
\tag{113.20}
\]

For an oriented K21 dual plane \(\Lambda\), its joint adjoint rows are

\[
\boxed{\begin{aligned}
 g^u_k(\Lambda)={}&
2\sum_j\Lambda_{kj}(A_ju_k+A_ku_j)
+2\sum_i\Lambda_{ik}A_iu_i,\\
 g^A_k(\Lambda)={}&
\sum_j\Lambda_{kj}
\{A_kA_j+A_jA_k+u_ku_j^T+u_ju_k^T\}\\
&+\sum_i\Lambda_{ik}(A_i^2+u_iu_i^T).
\end{aligned}}
\tag{113.21}
\]

For a symmetric covariance dual \(Z\),

\[
 D C^*(Z)=
 \left(2ZU,\ 
 \left\{\sum_jZ_{kj}A_j\right\}_{k=1}^n\right). \tag{113.22}
\]

Consequently the covariance-restricted K21 differential is onto a declared
dual target space \(\mathcal W\) iff

\[
\boxed{
 \begin{array}{c}
 \Lambda\in\mathcal W,\quad Z=Z^T,\\
 g^u_k(\Lambda)=2(ZU)_k,\quad
 P_Eg^A_k(\Lambda)=P_E\sum_jZ_{kj}A_j\quad\forall k
 \end{array}
 \Longrightarrow\Lambda=0.}
\tag{113.23}
\]

For the full oriented plane, \(\mathcal W=\mathbb R^{n\times n}\); for the
strict K21 plane, \(\Lambda_{ii}=0\).  Equivalently,

\[
 DK\,P_{\ker DC}\,DK^*\succ0\quad\hbox{on }\mathcal W. \tag{113.24}
\]

This is the exact joint minor/conditioning receipt.  Testing only symmetric
\(\Lambda\), or testing \(A\) while freezing \(U\), is not the full joint
gate.

### 113.4 The affine CP seed is generically K21-onto

There is no intrinsic K21 tangent annihilator at \(r=63\).  At the affine
seed

\[
 U=F,\qquad A_i=0,
\]

with \(F\) of full column rank, \(\delta C=0\) implies

\[
 \delta U=U\Omega,\quad\Omega^T=-\Omega,
\]

while every \(\delta A_i\) is covariance-tangent.  The \(U\) variation is
pure latent gauge and does not enter \(\delta K21\).

Let \(g_i\) be row \(i\) of \(F\), let
\(\Pi_j=I-g_jg_j^T/\|g_j\|^2\), and form the projected Veronese family

\[
 \mathcal P_j=
 \left\{
 (\Pi_jg_i)(\Pi_jg_i)^T:i\ne j
 \right\}
 \subseteq\operatorname{Sym}(g_j^\perp). \tag{113.25}
\]

Then

\[
\boxed{
 \operatorname{rank}\mathcal P_j=n-1\quad\forall j
 \Longrightarrow
 DK21\big|_{\ker DC}\text{ is onto }\mathbb R^{n\times n}.}
\tag{113.26}
\]

To prove it, apply the adjoint at \(A=0\):

\[
 (DK21^*\Lambda)_j
 =\sum_i\Lambda_{ij}g_ig_i^T
 +2\sum_k\Lambda_{jk}\operatorname{sym}(g_jg_k^T). \tag{113.27}
\]

Projecting (113.27) on both sides by \(\Pi_j\) kills the second sum.
Independence in (113.25) forces every off-diagonal entry in column \(j\) of
\(\Lambda\) to vanish.  The remaining diagonal equation is
\(3\Lambda_{jj}g_jg_j^T=0\), so \(\Lambda=0\).

For strict K21 there is also a direct right inverse: impose
\(\delta A_jg_j=0\); then the row-pollution term vanishes and the independent
quadratic evaluations \(g_i^T\delta A_jg_i,\ i\ne j\), prescribe column \(j\).

Generically, (113.25) has rank

\[
 \min\left(n-1,{r(r-1)\over2}\right). \tag{113.28}
\]

At \(r=63\), the second dimension is 1953, so the certificate covers every
\(n\le1954\), including \(n=1024\).  The active quotient at this rank-63
affine seed has dimension

\[
 ne=2{,}064{,}384,
\]

leaving 1,016,832 directions after strict K21 and 1,015,808 after full K21.
This proves generic local K21 surjectivity at the affine CP seed.  It does
**not** by itself solve the actual full-rank target covariance: continuing
from \(C=FF^T\) to the D14 covariance must retain the joint certificate
(113.23) and a positive singular floor.

For a hard \(p\)-dimensional H2 support at the affine seed, the unavoidable
counts are only \(p\ge n-1\) (strict) and \(p\ge n\) (full).  A generic
constructive version of (113.25), using
\(E\cap\{B:Bg_j=0\}\), is available when

\[
 p-r\ge n-1,
\quad\hbox{i.e.}\quad p\ge1086 \text{ here}. \tag{113.29}
\]

The larger full-covariance floors (113.14)--(113.15), not this affine
certificate, bind the deployed seam.

### 113.5 Exact K21 circuit and why an iterative refit cannot fit

The K21 plane itself has a one-broad-product normal form.  Let

\[
\begin{aligned}
 X_{i,:}&=\operatorname{svec}(A_i)^T,\\
 R_{i,:}&=\operatorname{svec}\!\left(
 P_E(A_i^2+u_iu_i^T)\right)^T,\\
 V_{i,:}&=(A_iu_i)^T .
\end{aligned}
\tag{113.30}
\]

Then

\[
\boxed{K21=RX^T+2VU^T.} \tag{113.31}
\]

Thus one generic full-support K21 evaluation needs at least

\[
 {p+r\over n}
 ={2016+63\over1024}
 =2.0302734375 \tag{113.32}
\]

f32 width-matmul equivalents, before forming \(A_i^2\) and \(A_iu_i\).

The adjoint exposes an even sharper iteration floor.  Define

\[
 B=\Lambda X,\qquad D=\Lambda^TR,\qquad
 a=\Lambda U,\qquad b=\Lambda^TV. \tag{113.33}
\]

If \(B_k,D_k\) denote the symmetric matrices represented by their rows, then

\[
\boxed{\begin{aligned}
 g^u_k&=2(B_ku_k+A_ka_k+b_k),\\
 g^A_k&=P_E\!\left(
 A_kB_k+B_kA_k+D_k
       +u_ka_k^T+a_ku_k^T\right).
\end{aligned}}
\tag{113.34}
\]

For a generic dense residual \(\Lambda\), acquiring (113.33) requires the two
block products

\[
 \Lambda[\,X,U\,],\qquad
 \Lambda^T[\,R,V\,],
\]

whose exact rectangular-product census is

\[
\boxed{
 C_{\rm one\ K21\ adjoint}\ge {2(p+r)\over n}
 =4.060546875}
\tag{113.35}
\]

before the local Jordan products, covariance-tangent projection, retraction,
or line search.  The four-reset r63 shell has only
\(49-45.68054=3.31946\) equivalents of headroom; the five-reset shell has
only \(49-48.07040=0.92960\).  Therefore even one literal dense
gradient/Newton/refit iteration is arithmetically closed for both shells.
An equivariant flow is a mathematical section definition, not a deployable
algorithm under this cap.

### 113.6 Minimum anchor bill and the surviving direct-section branch

For the polar construction, write

\[
 S_0=BH,\qquad B=C^{1/2}K_H^{-1/2}.
\]

The first suffix coefficient push is the three-factor contraction

\[
 W^TS_0=W^TBH. \tag{113.36}
\]

In the declared binary dense-product census, with \(d>n\), the two possible
parenthesizations cost

\[
\begin{array}{c|c}
(W^TB)H&1+d/n\\
W^T(BH)&2d/n
\end{array}
\]

equivalents.  The shell already charges the final \(d/n\) coefficient push,
so a generic nonresident dense \(B\) has the exact incremental floor

\[
\boxed{\Delta C_B\ge\min(1,d/n)=1.} \tag{113.37}
\]

The selected-ray Gram \(G=FF^T\) adds \(r/n=0.0615234375\) unless literally
resident.  Hence the five-reset shell is already

\[
48.07040+1=49.07040>49 \tag{113.38}
\]

even granting a free Gram, free matrix roots, free K21 solve, and zero
workspace cost.  With the Gram it is 49.1319234375.  This is a decisive
no-go for the generic CP4-polar five-reset construction.

The literal spectral root route is still more expensive: one dense
width-1024 f32 symmetric eigendecomposition is billed at about \(9n^3\)
flops, or \(4.5\) f32 matmul-equivalents (twice that at the f64 billing
rate), before root reconstruction.  It closes even the four-reset shell.

The conditional canonical-order Cholesky construction (113.7) is the cheapest
identified exact escape from the spectral root.  Fusing its first push as

\[
 (W^TL_C)L_K^{-1}H
\]

costs one extra equivalent for the triangular multiply and solve.  Two f32
Cholesky factorizations cost \(1/3\) equivalent in the same leading census,
and \(FF^T\) costs \(63/1024\).  Its optimistic acquisition subtotal is
therefore

\[
\boxed{
 C_{\rm canon\text{-}chol,extra}
 \ge1+{1\over3}+{63\over1024}
 =1.3948567708\ldots .}
\tag{113.39}
\]

It still closes the five-reset shell.  It leaves the four-reset anchor at
about \(47.07540\), but adding even one generic K21 evaluation (113.32)
raises it above 49, before a solve.  At f64 rates the margin is smaller.

Therefore the sole under-49 survivor in this construction class is:

1. the four-reset suffix;
2. a strictly gated lawful structured/canonical congruence;
3. a **literal direct K21 section identity** requiring no dense K21
   evaluation or adjoint iteration;
4. the joint minor (113.23), target-conditioned K22 burden (113.19), and
   endpoint value gate; and
5. every factor/root/solve operation charged in the final dtype.

No such direct section identity is presently known.  A special resident
congruence \(C=DK_HD\) with diagonal \(D\), or another already-paid
low-complexity multiplier algebra, can remove (113.37); this is an exact
reopen condition, not a generic property.

The static full AQ63 bank contains

\[
 nd=2{,}128{,}896
\]

scalars (8.1211 MiB at f32, 16.2422 MiB at f64).  A literal dense gradient
step additionally needs a same-size adjoint/update bank and an
\(n^2=1{,}048{,}576\)-scalar K21 residual plane; roots/congruences require
at least width-square workspaces unless block-recomputed.  Streaming can
reduce peak temporaries but not the product lower bounds (113.35)--(113.39).

Graph-ready split:

- CP4_AQ63_JOINT_EQUIVARIANT_ANCHOR -- **solved exact**, (113.2)--(113.7);
- AQ63_SUPPORTED_GAUGE_DIMENSION -- **solved/refined**, (113.8)--(113.18);
- AQ63_TARGET_K22_FIBER_GATE -- **exact conditional**, (113.19);
- AQ63_JOINT_K21_DUAL_IFF -- **solved exact**, (113.20)--(113.24);
- AQ63_AFFINE_SEED_K21_SURJECTIVE -- **proved generic certificate**,
  (113.25)--(113.29), continuation to the full cut open;
- AQ63_K21_EVAL_CIRCUIT -- **solved exact**, (113.30)--(113.32);
- AQ63_K21_ITERATIVE_REFIT_CAP -- **NO-GO**, (113.33)--(113.35);
- CP4_AQ63_POLAR_L5_CAP -- **NO-GO even with free roots/solve**,
  (113.36)--(113.38);
- CP4_AQ63_CANON_CHOL_L4 -- **conditional arithmetic survivor only**,
  (113.39), direct K21 section/value still open;
- CP4_AQ63_STATIC_STORAGE -- **exact coefficient count**;
- CP4_AQ63_DIRECT_SECTION -- **open unique redemption path for this seam**.

## 114. The effective K22 costate: Schur-feature rank, dual source
factorization, and panel-common cap

Section 112 gives an exact three-product circuit for every spectral factor
of the local K22 costate.  This section characterizes when those factors
are actually few.  It also gives a dual circuit controlled by
\(\operatorname{rank}P\), and a source-atom circuit that is cheaper than
either rank route when a genuine small native decomposition exists.

Suppress the harmless coefficient \(\mu^2/4\) until the action bound.  On
the strict off-diagonal support put

\[
 f(r)=(1-r^2)^{-1/2},\qquad
 L=\mathcal M_{\rm off}\odot\Lambda\odot f[R]. \tag{114.1}
\]

All statements involving \(f[R]\) require a live support margin

\[
 \rho=\max_{a\ne b:\Lambda_{ab}\ne0}|R_{ab}|<1,\qquad
 \gamma=\sqrt{1-\rho^2}. \tag{114.2}
\]

### 114.1 Exact Schur-feature rank and its limitations

The even power series is

\[
 f(r)=\sum_{m=0}^{\infty}a_mr^{2m},\qquad
 a_m={\binom{2m}{m}\over4^m}. \tag{114.3}
\]

If \(R=UU^T\) has rank \(r_R\), its \(2m\)-th Schur power is the Gram
matrix of the symmetric tensor features \(u_a^{\otimes2m}\), hence

\[
 \operatorname{rank}(R^{\odot2m})
 \le d_m(r_R):={r_R+2m-1\choose2m}. \tag{114.4}
\]

Let \(\Lambda\) have signed spectral rank \(r_\Lambda\) and truncate
(114.3) at \(Q\).  Before strict diagonal deletion,

\[
\boxed{
 \operatorname{rank}\!\left[
 \Lambda\odot\sum_{m=0}^Qa_mR^{\odot2m}\right]
 \le
 \min\!\left\{n,\,
 r_\Lambda\sum_{m=0}^Qd_m(r_R)\right\}.} \tag{114.5}
\]

This follows from
\[
 (xx^T)\odot(yy^T)=(x\odot y)(x\odot y)^T.
\]
It is the exact Schur-feature rank bound.  It is useful only when both
factors are small: already
\[
 d_1(55)={56\choose2}=1540>1024. \tag{114.6}
\]

Thus an affine-\(r55\) correlation factor gives no nonconstant exact
rank saving at width \(1024\).  Commutativity or a common *spectral*
eigenbasis of \(R\) and \(\Lambda\) is also insufficient, because the
Hadamard product is tied to the physical coordinate basis.

Exact low-rank cases do exist:

1. \(R_{ab}^2\) is constant on the live support, so \(L\) is a scalar
   multiple of the supported \(\Lambda\);
2. \(f[R]\) has a proved \(s\)-term separable representation, giving
   \(\operatorname{rank}L\le r_\Lambda s\) before masking;
3. the live costate is supported inside an \(h\times h\) coordinate
   block, giving rank at most \(h\); or
4. \(\Lambda\) and \(R\) share a literal block-constant partition whose
   quotient Schur matrix has small rank.

Names such as “collective,” “AQ,” or “low latent rank” imply none of these
coordinate statements.

Strict masking can reverse the conclusion.  At \(R=I\) and
\(\Lambda=\mathbf1\mathbf1^T\),

\[
 L=\mathbf1\mathbf1^T-I,\qquad \operatorname{rank}L=n. \tag{114.7}
\]

If the diagonal K4 action is separately owned, one may instead choose a
diagonal completion and define the relevant collision-sewn rank

\[
 r_{\rm comp}(L)
 =\min_{d\in\mathbb R^n}\operatorname{rank}\{L+\operatorname{Diag}d\}.
 \tag{114.8}
\]

The witness (114.7) then has the exceptional rank-one completion
\(\mathbf1\mathbf1^T\).  This is not generic.  The symmetric rank-\(r\)
variety has dimension

\[
 nr-{r(r-1)\over2}.
\]

Since the off-diagonal data have dimension \(n(n-1)/2\), a generic
completion can have rank \(r\) only if

\[
 nr-{r(r-1)\over2}\ge{n(n-1)\over2}. \tag{114.9}
\]

Therefore

\[
\boxed{
 r_{\rm comp}^{\rm generic}
 \ge
 \left\lceil{2n+1-\sqrt{8n+1}\over2}\right\rceil.} \tag{114.10}
\]

At \(n=1024\) the lower bound is \(980\).  It applies even to a generic
rank-one \(\Lambda=uu^T\): diagonal congruence removes \(u\), while
entrywise \(f\) maps an open set of nonzero weak correlations to an open
set of off-diagonal values.  Hence collision completion is an exact
escape only with additional structure, not generically.

### 114.2 Sharp weak-correlation and conditioning bounds

Define

\[
 f_Q(r)=\sum_{m=0}^Qa_mr^{2m},\qquad
 T_Q(\rho)=f(\rho)-f_Q(\rho). \tag{114.11}
\]

Because \(a_m\) decreases,

\[
 0\le T_Q(\rho)
 \le {a_{Q+1}\rho^{2Q+2}\over1-\rho^2}. \tag{114.12}
\]

Consequently the exact Schur-tail certificate is

\[
\boxed{
 \|\Lambda\odot(f[R]-f_Q[R])\|_F
 \le T_Q(\rho)\|\Lambda\|_F.} \tag{114.13}
\]

For \(Q=0\),

\[
 T_0(\rho)=\gamma^{-1}-1
 \le{\rho^2\over2\gamma^3}. \tag{114.14}
\]

Thus, **after literal diagonal sewing**, the base covariance costate
\(\Lambda\) is a rank-\(r_\Lambda\) control variate and its omitted
nonconstant boundary row is \(O(\rho^2)\) in Frobenius norm.  If the
base-\(\Lambda\) K22 action is resident, only that tail needs new work.
Without diagonal sewing, the constant term itself becomes
\(\mathcal M_{\rm off}\odot\Lambda\) and may be full rank by (114.7).

There is nevertheless a sharp *approximate* no-sewing bound.  The
off-diagonal kernel tail and the deleted diagonal have disjoint support,
so using the full low-rank \(\Lambda\) as the representative gives

\[
\boxed{
 \|L-\Lambda\|_F^2
 =\|\operatorname{diag}\Lambda\|_2^2
  +\|\mathcal M_{\rm off}\odot\Lambda\odot(f[R]-1)\|_F^2
 \le
 \|\operatorname{diag}\Lambda\|_2^2
  +T_0(\rho)^2\|\mathcal M_{\rm off}\odot\Lambda\|_F^2.} \tag{114.14a}
\]

For the witness \(\Lambda=\mathbf1\mathbf1^T,R=I\), the strict matrix is
exact-full-rank but (114.14a) has relative Frobenius error \(1/\sqrt n\);
at width \(1024\), this is \(3.125\%\).  More generally the exact
diagonal-energy ratio
\(\|\operatorname{diag}\Lambda\|_2/\|\Lambda\|_F\), not rank alone,
decides whether collision sewing can be replaced by a charged
approximation.  Endpoint leverage through (114.15) still applies.

Let \(H_{22}\) denote the transported source K22 face at the same seam.
Any approximation \(L_r\) obeys

\[
\boxed{
 |\delta S_{22}|
 \le{\mu^2\over4}\|L-L_r\|_F\|H_{22}\|_F.} \tag{114.15}
\]

For a panel and several seams, sum the signed scalar errors first, apply
the complete-old corpus projector, and only then divide by the surviving
feature norm.  A small matrix tail is not a value receipt when the
retained action is endpoint-dark.

The boundary kernel is ill-conditioned near \(|R|=1\):

\[
 f'(r)={r\over(1-r^2)^{3/2}},\qquad
 \left|{rf'(r)\over f(r)}\right|
 ={r^2\over1-r^2}. \tag{114.16}
\]

To first order, perturbations satisfy

\[
\boxed{
 \|\delta(\Lambda\odot f[R])\|_F
 \le\gamma^{-1}\|\delta\Lambda\|_F
{\rho\over\gamma^3}\|\Lambda\|_F\|\delta R\|_{\max}.} \tag{114.17}
\]

A frozen spectral truncation also needs an absolute singular-value gap
\(\operatorname{gap}_r=\sigma_r-\sigma_{r+1}\) larger than the operator
perturbation; Wedin/Davis--Kahan then gives subspace rotation at most
\(\|\delta L\|_{\rm op}/\operatorname{gap}_r\).  For a signed eigenfactor
the intrinsic summation condition is

\[
 \kappa_{\rm fac}
 ={\,\|L_r\|_*\over\|L_r\|_F}\le\sqrt r. \tag{114.18}
\]

An arbitrary Schur expansion can have a much worse absolute-weight sum
and must report it separately.

### 114.3 A dual three-product circuit controlled by
\(\operatorname{rank}P\)

Section 112 factors the endpoint costate.  The same action admits an
exact source-side factorization.  Let

\[
 P=\sum_{t=1}^{r_P}\eta_tp_tp_t^T,\qquad
 H_t=WD_{p_t}R^T,\qquad
 Y=W^{\odot2},\quad X_t=H_t^{\odot2},\quad
 Z_t=W\odot H_t. \tag{114.19}
\]

Direct expansion of (112.2)--(112.3) gives

\[
\boxed{
 S_{22}
 =4\sum_t\eta_t\langle\Lambda,YX_t^T\rangle_F
 8\sum_t\eta_t\langle\Lambda,Z_tZ_t^T\rangle_F.} \tag{114.20}
\]

Per \(P\)-factor, the products are exactly

\[
 WD_{p_t}R^T,\qquad YX_t^T,\qquad Z_tZ_t^T. \tag{114.21}
\]

Hence the attained exact rank cost is

\[
\boxed{
 r_{\rm circuit}
 =\min\{r_{\rm eff}(\Lambda),\,r_P\},} \tag{114.22}
\]

where \(r_{\rm eff}(\Lambda)\) includes the strict-mask/diagonal-sewing
rule and any literal resident control.  The last two products in
(114.21) have generic bilinear rank two: the independent outer-product
terms \(YX^T\) and \(ZZ^T\) cannot be fused into one ordinary \(n^3\)
product without a dependency or a doubled inner dimension, which has the
same classical FMA count.

This dual circuit is useful if \(P\) has a literal deterministic factor
rank below the costate rank.  The gate kernel \(p[R]\) and the generic
\(P^\sharp\) are full rank by (105.13), so their random-feature names are
not deterministic rank receipts.

### 114.4 A cheaper direct scalar for a genuine native-atom source

There is a stronger conditional escape when the source itself is already
a small portfolio of typed atoms.  For the normalized tensor
\(\operatorname{Sym}_4(A,A,B,C)\), let \(a,b,c\) be its three transported
generator vectors and put

\[
 u=a^{\odot2},\quad v=b\odot c,\quad
 x=a\odot b,\quad y=a\odot c.
\]

Its K22 face is

\[
 H_{22}
 ={1\over6}(uv^T+vu^T)
 +{1\over3}(xy^T+yx^T). \tag{114.23}
\]

For symmetric strict \(L\),

\[
\boxed{
 \langle L,H_{22}\rangle_F
 ={1\over3}u^TLv+{2\over3}x^TLy.} \tag{114.24}
\]

Thus one atom needs only the two dense matrix-vector actions \(Lv,Ly\),
not a spectral factorization of \(L\).  For \(A\) atoms, batch their
\([v,y]\) columns; the exact cost is controlled by the query rank

\[
 r_Q=\operatorname{rank}[\,v_1,y_1,\ldots,v_A,y_A\,]\le2A. \tag{114.25}
\]

Across \(D_s\) seams its dominant matrix-by-skinny count is

\[
 C_{\rm atom}={D_sr_Q\over n}. \tag{114.26}
\]

At \(D_s=10,n=1024\), \(A\le51\) fits inside one full-equivalent before
hooks when the columns are generic.  This computes one declared endpoint
scalar only; it does not materialize a reusable pulled-back K22 row.
Moreover, the broad kernel (112.3) has no proved small native-atom
decomposition.  Equation (114.24) is therefore a real circuit escape,
not permission to rename a high-rank broad source as one atom.

### 114.5 Panel-common rank and cap

For panel/seam effective rows \(L_j\), a single common factor space
\(\mathcal U\) exists exactly when

\[
\boxed{
 r_{\rm union}
 =\operatorname{rank}[\,L_1\ L_2\ \cdots\ L_J\,]
 \le\dim\mathcal U.} \tag{114.27}
\]

Low rank of every \(L_j\) separately does not control \(r_{\rm union}\);
their ranges can be disjoint.  If per-instance state-derived eigenvectors
are allowed, the arithmetic rank is \(\max_j\operatorname{rank}L_j\),
but the truncation rule and tail threshold, not the eigenvectors, must be
frozen before target grading.

The best common one-sided rank-\(r\) range has residual

\[
 \boxed{
 \min_{U^TU=I_r}\sum_j\|(I-UU^T)L_j\|_F^2
 =\sum_{s>r}\lambda_s\!\left(\sum_jL_j^2\right).} \tag{114.28}
\]

For symmetric double-sided reconstructions \(UU^TL_jUU^T\), the total
residual lies between this quantity and twice it.  Equation (114.28) is
label-free because it uses only the physical costates, but computing and
storing it must be billed.  Weak-correlation control gives the common
upper bound

\[
 \sum_j\|L_j-\Lambda_j\|_F^2
 \le T_0(\rho)^2\sum_j\|\Lambda_j\|_F^2 \tag{114.29}
\]

only if all collision diagonals are owned and the retained
\(\Lambda_j\) spaces themselves have acceptable union rank.

With \(h\in\{0,1,2,3\}\) literally resident products per factor, the
three-product family costs

\[
\boxed{
 C_{22}=(3-h)D_sr_{\rm circuit}
       +C_{\rm elem}+C_{\rm hook}.} \tag{114.30}
\]

For \(h\le2\), under the per-seam full-product ceilings of Sections 105
and 112,

\[
\boxed{
 r_{\rm circuit}\le
 \left\lfloor{4\over3-h}\right\rfloor
 \quad\text{(utility parity)},\qquad
 r_{\rm circuit}\le
 \left\lfloor{9\over3-h}\right\rfloor
 \quad\text{(optimistic raw leader)}.} \tag{114.31}
\]

With no reuse these are ranks one and three.  If all three products are
literally resident (\(h=3\)), products impose no rank cap, but factor
state, elementwise work, and hooks still do.  Any spectral, Schur, common-
panel, or weak-correlation approximation must additionally satisfy
(114.15), the conditioning margin (114.17), and the deterministic
post-old fidelity gate

\[
 {\|e_{\rm corpus}\|\over\|s_{\rm corpus}\|}
 \le{\rho_t-0.8124\over1+0.8124}\le0.1035. \tag{114.32}
\]

Graph-ready split:

- K22_EFFECTIVE_SCHUR_RANK -- **solved conditional**, (114.3)--(114.6);
- K22_STRICT_AND_COMPLETION_RANK -- **generic full / at least 980 after
  generic diagonal completion**, (114.7)--(114.10);
- K22_WEAK_CORRELATION_TAIL -- **sharp deterministic**, (114.11)--
  (114.15);
- K22_BOUNDARY_CONDITIONING -- **exact local gate**, (114.16)--(114.18);
- K22_DUAL_P_FACTOR_CIRCUIT -- **exact three products per source factor**,
  (114.19)--(114.22);
- K22_NATIVE_ATOM_SCALAR -- **exact two-matvec conditional escape**,
  (114.23)--(114.26);
- K22_PANEL_COMMON_RANK -- **exact union/tail gate**, (114.27)--(114.29);
- K22_RANK_CAP -- **rank one utility / rank three optimistic leader
  without reuse**, (114.30)--(114.32);
- K22_VALUE -- **open** until the complete-old physical costate, collision
  sewing, and transported source action are available.

## 115. Can the covariance Bellman row stay low rank? Exact weak-
correlation reverse recurrence and the one-transition verdict

Section 112's cheap branch needs a low-rank covariance/K22 costate, not
merely a scalar endpoint.  Zero-bias Gaussian covariance has a distinguished
rank-preserving reverse mode, but its nonconstant gate kernel injects rank at
every reset.  The following recurrence quantifies that injection.

### 115.1 Reset Jacobian and the constant mode

For standardized zero-bias Gaussian ReLU covariance, Price's theorem gives

\[
 {d\over d\rho}\operatorname{Cov}(X_+,Y_+)
 =p(\rho),\qquad
 p(\rho)={1\over4}+{1\over2\pi}\arcsin\rho. \tag{115.1}
\]

Holding the marginal variances fixed, the off-diagonal reverse covariance
row therefore updates as

\[
 \Lambda^-= \Lambda^+\odot p[R]. \tag{115.2}
\]

If all live off-diagonal entries satisfy \(|R_{ab}|\le\rho_*<1\), write

\[
 p[R]={1\over4}\mathbf1\mathbf1^T+E_p,\qquad
 \|E_p\|_{\max}\le a_p:={\arcsin\rho_*\over2\pi}. \tag{115.3}
\]

The constant term maps any rank-\(r\) matrix \(L\) to \(L/4\), preserving
rank exactly.  The nonconstant term is the sole off-diagonal rank-injection
source in this typed covariance-only reset.

Exact rank is nevertheless generic-full after even one reset.  If
\(\Lambda^+=uu^T\) and \(R=I\), then
\[
 \Lambda^-=\operatorname{Diag}(u)\,
 {(\mathbf1\mathbf1^T+I)\over4}\,
 \operatorname{Diag}(u), \tag{115.3a}
\]
which has rank \(n\) for nonzero \(u_a\).  On the strict off-diagonal
quotient the kernel is \((\mathbf1\mathbf1^T-I)/4\), also full rank.
Using the rank-one constant representative therefore requires literal
diagonal/collision sewing; it is an approximation/quotient theorem, not an
exact matrix-rank theorem.

### 115.2 Exact relative-tail recurrence

Decompose a reverse row as

\[
 \Lambda=L+E,\qquad
 x={\|E\|_F\over\|L\|_F}, \tag{115.4}
\]

where \(L\) is the retained rank-\(r\) mode.  Use \(L^-=L/4\) as the next
retained mode.  From (115.2)--(115.3),

\[
\boxed{
 x^-\le(1+\eta_p)x+\eta_p,\qquad
 \eta_p={2\over\pi}\arcsin\rho_* .}
 \tag{115.5}
\]

Indeed,
\[
 E^-={E\over4}+L\odot E_p+E\odot E_p
\]
and \(\|A\odot E_p\|_F\le a_p\|A\|_F\).

A dense covariance transport has reverse map

\[
 \Lambda\longmapsto W^T\Lambda W. \tag{115.6}
\]

It preserves the rank of \(L\), but can amplify its relative tail.  Define
the exact leading-mode conditioning factor

\[
 \chi_W(L):=
 {\|W\|_{\rm op}^2\|L\|_F\over\|W^TLW\|_F}. \tag{115.7}
\]

Combining one reset and one transport gives

\[
\boxed{
 x_{\ell-1}\le
 \chi_{W_\ell}(L_\ell)
 \{(1+\eta_{p,\ell})x_\ell+\eta_{p,\ell}\}.}
 \tag{115.8}
\]

This is an upper certificate; it does not assume random weights or
favorable cancellation.  A near-null leading mode makes \(\chi_W\) large
even when \(\|W\|_{\rm op}\) is benign.

### 115.3 Adding the K22 boundary kernel

At the K22 consumer, Section 112 multiplies the covariance row by

\[
 D(R)=(1-R^{\odot2})^{-1/2}
 =\mathbf1\mathbf1^T+E_D. \tag{115.9}
\]

With
\[
 \delta_D=(1-\rho_*^2)^{-1/2}-1,
\]
the additional relative tail obeys

\[
\boxed{
 x_{22}\le(1+\delta_D)x+\delta_D.}
 \tag{115.10}
\]

The strict diagonal deletion is not included in (115.10); it must be sewn
through the collision/K4 row as required by Section 112.  The K31 boundary
kernel injects an independent \(O(\rho_*)\) row.

### 115.4 Ten-layer versus one-transition arithmetic

In the best-conditioned idealization
\(\chi_{W_\ell}=1\), with one uniform \(\eta_p\) and \(x_D=0\), (115.8)
solves exactly to

\[
 x_{D-d}\le(1+\eta_p)^d-1. \tag{115.11}
\]

Requiring this source-independent norm certificate to stay below the
optimistic \(10.35\%\) physics-fidelity margin of Section 101 gives, for
ten resets,

\[
 \eta_p\le(1.1035)^{1/10}-1\simeq0.00990,
\qquad
 \boxed{\rho_*\lesssim0.0156.} \tag{115.12}
\]

This is before the K22 tail (115.10), source-action leverage, old
projection, and any \(\chi_W>1\).  Therefore “weak correlation” does not
justify a rank-one row through ten generic resets unless the correlations
are at the percent level or a literal resident projection removes the
injected modes.

For a one-transition consume-and-retire source, the same ideal bound permits
\[
 \eta_p\le0.1035,\qquad \rho_*\lesssim0.162. \tag{115.13}
\]

That regime is materially less restrictive.  It explains why a local
K22 action may be viable even when a persistent low-rank covariance
costate is not.

The word “one-transition” refers only to the *forward source lifetime*.
It does not reset the age of the reverse Bellman row.  If the consuming
seam is \(h\) layers from the endpoint, its costate has already undergone
\(h\) instances of (115.8), regardless of whether the new q211 source is
retired immediately.  Therefore (115.13) applies only to a terminal/local
consumer or to a seam where a literal resident compressed reverse row is
supplied.  An L5 birth in a depth-16 network does not inherit the
one-step bound merely because its H4 tensor is consumed at L6.

### 115.5 Endpoint leverage and disposition

A Frobenius tail is not yet a scalar-action tail.  If
\(\mathcal A_T\) denotes the old-residualized q211-to-K22 action, define

\[
 \chi_T(L):=
 {\|\mathcal A_T\|\,\|L\|_F\over|\mathcal A_T(L)|}. \tag{115.14}
\]

Then a sufficient relative scalar-error gate is

\[
 \chi_T(L)x_{22}\le\varepsilon_{\rm value}. \tag{115.15}
\]

Small source norm, small \(\rho_*\), or low covariance rank gives no
positive lower bound on \(|\mathcal A_T(L)|\); the leverage receipt is
independent.  For a statistically refitted feature, replace the
coefficient-one margin by the direct held-out correlation gate, as in
Section 101.5.

Graph-ready split:

- RELU_COV_REVERSE_CONSTANT_MODE -- **solved exact**, (115.1)--(115.3);
- RELU_COV_REVERSE_EXACT_RANK -- **generic full after one reset; low-rank
  use is quotient/approximate only**, (115.3a);
- RELU_COV_REVERSE_RANK_TAIL -- **solved conditional recurrence**,
  (115.4)--(115.8);
- RELU_K22_BOUNDARY_RANK_TAIL -- **solved conditional**, (115.9)--
  (115.10);
- RELU_COV_RANK1_TEN_LAYER -- **effectively closed without
  \(\rho_*\lesssim0.0156\), conditioning, and collision receipts**,
  (115.11)--(115.12);
- RELU_COV_RANK1_TERMINAL_ONE_TRANSITION -- **conditional reopen up to
  \(\rho_*\simeq0.162\) only for a fresh/local reverse row**, (115.13);
- RELU_COV_RANK_TAIL_VALUE -- **open on endpoint leverage**, (115.14)--
  (115.15).

## 116. Concrete candidate: rank-one constant-Price K22 Bellman features
with layerwise refit

The exact covariance reverse is generically full rank, but its independent-
gate component is rank preserving and can be propagated using vectors.
Using that component as a *statistical feature query*, rather than claiming
it is the full physical reverse row, gives a concrete q211/K22 candidate
whose complete dominant bill fits inside the held f32 utility margin.

### 116.1 Lawful rank-one query

Suppose the scalar endpoint contains the covariance read

\[
 V_D=c^TC_Dc. \tag{116.1}
\]

Its covariance costate is \(cc^T\).  Set \(s_D=c\).  For zero-bias
standardized ReLU, retain only the constant Price mode \(p(0)=1/4\), and
propagate

\[
 \boxed{
 s_\ell={1\over2}W_{\ell+1}^Ts_{\ell+1},\qquad
 \Lambda_\ell^{(0)}=s_\ell s_\ell^T.}
 \tag{116.2}
\]

This is exact for the constant-mode reverse because
\[
 {1\over4}ss^T=(s/2)(s/2)^T
\]
and covariance conjugation preserves rank one.

For arbitrary standardized thresholds, let
\[
 g_{\ell,a}=P(G_{\ell,a}>\tau_{\ell,a}).
\]
The independent-gate Price matrix is \(gg^T\), so the exact constant-mode
recursion becomes

\[
 \boxed{
 s_\ell=W_{\ell+1}^T(g_{\ell+1}\odot s_{\ell+1}).}
 \tag{116.3}
\]

All quantities in (116.2)--(116.3) are label-free functions of the frozen
weights, biases, and declared endpoint.  The recursion costs one vector
matvec per layer, negligible on the width-GEMM census.  If the actual
endpoint covariance row is not rank one, each admitted endpoint factor is
a separate query and must be billed.

### 116.2 Layer feature

At layer \(\ell\), let \(K_\ell^{\rm new}\) be the declared q211 source
after literal current-D11/CP3/pair/collision subtraction.  It is zero if a
full Section 49 q211 surface is already the comparator.  For the broad
kernel

\[
 (K_\ell^{\rm new})_{i;jk}
 =R_{\ell,ij}R_{\ell,ik}(P_{\ell}^{\rm new})_{jk}, \tag{116.4}
\]

there are two rank-one source choices.  At zero bias,
\(P^{\rm new}=P^\sharp\) is the Section 96 CP3/q211-fused channel.  At
arbitrary bias, one may retain only the first product channel
\(d_{i2}b_{j2}b_{k2}xyz\) in (99.17), absorbing its diagonal coefficients
into the three legs.  The latter is an exact component of the current
residual but not the complete degree-four source; its omitted side-powered
channels remain an explicit feature-tail error.

prepare the single costate factor for Section 112 as follows.

The K22 Edgeworth consumer contributes its own boundary-density and
\(1/4\) factorial weight.  For standardized thresholds define

\[
 h_{\ell,a}=\phi(\tau_{\ell,a}),\qquad
 t_\ell={1\over2}(h_\ell\odot s_\ell). \tag{116.4a}
\]

At zero bias \(t_\ell=(\mu/2)s_\ell\), and
\(t_\ell t_\ell^T=(\mu^2/4)s_\ell s_\ell^T\), exactly matching (111.5).
Non-unit variance factors are applied in the declared standardized
coordinate map.  It is \(t_\ell\), not the bare covariance vector
\(s_\ell\), that enters the K22 action:

For nonzero correlation the exact boundary matrix is
\(\phi_2(\tau_i,\tau_j;\rho_{ij})/4\), not
\(\phi(\tau_i)\phi(\tau_j)/4\).  Equation (116.4a) retains only its
independent rank-one mode; the difference is another explicit query-tail,
or a separate source channel if carried exactly.

This tail has a closed entrywise certificate.  The exact ratio is
\[
 {\phi_2(\tau_i,\tau_j;\rho)\over\phi(\tau_i)\phi(\tau_j)}
 ={1\over\sqrt{1-\rho^2}}
 \exp\!\left\{
 {\rho\tau_i\tau_j-\frac12\rho^2(\tau_i^2+\tau_j^2)
  \over1-\rho^2}\right\}. \tag{116.4b}
\]
Hence, if \(|\rho|\le\rho_*\) and \(|\tau_i|,|\tau_j|\le T\),
\[
 \left|{\phi_2\over\phi_i\phi_j}-1\right|
 \le e^{L_*}-1,\qquad
 L_*=-{1\over2}\log(1-\rho_*^2)
 +{(\rho_*+\rho_*^2)T^2\over1-\rho_*^2}. \tag{116.4c}
\]
The resulting endpoint error is bounded by this factor times the
independent-mode source action norm; cancellation/leverage is still a
separate receipt.

\[
\begin{aligned}
 A_\ell&=(W_\ell^{\odot2})^Tt_\ell,\\
 B_\ell&=W_\ell^TD_{t_\ell}W_\ell,\\
 G_\ell&=R_\ell D_{A_\ell}R_\ell,\\
 C_\ell&=B_\ell\odot R_\ell,
\end{aligned} \tag{116.5}
\]

and define

\[
\boxed{
 f_\ell
 =4\langle B_\ell\odot P_\ell^{\rm new},G_\ell\rangle_F
 +8\langle P_\ell^{\rm new},C_\ell^TC_\ell\rangle_F.}
 \tag{116.6}
\]

The known raw-symmetrization/deployed-\(1/24\) scale is applied once at the
consumer.  Means, K31, diagonal K4, and finite-edge terms are not silently
included in (116.6); they are either literal old rows or separate candidate
features.

For the current L5--L14 source window, the proposed feature bank is

\[
 F_{\rm constK22}=(f_5,\ldots,f_{14}). \tag{116.7}
\]

It is jointly residualized against the complete old feature span and fitted
once under a preregistered fold.  No coefficient-one or finite-repair claim
is made.

### 116.3 Honest product and adjusted-score bill

Section 112 and the literal ownership audit show that none of

\[
 W^TD_sW,\qquad RD_AR,\qquad (B\odot R)^T(B\odot R) \tag{116.8}
\]

is resident in the current forward-only strict48.197/abstract D10 tape.
Thus (116.6) costs three fresh dense products per active layer *and per
independent broad source channel*.  Let \(r_{\rm src}\) be the
complete-old-projected product-channel rank.  Ten layers cost

\[
 \boxed{C_{\rm dom}=30r_{\rm src}\ {\rm f32\ equivalents}} \tag{116.9}
\]

before elementwise work, hooks, source formation, and old-row subtraction.
The zero-bias \(P^\sharp\) candidate has \(r_{\rm src}=1\).  At arbitrary
bias, Section 99 proves that the current finite-D11 quotient has
\(r_{\rm src}=3\) generically (two only in the graded formal jet, with an
additional tail), and rank one only under its proportional-channel
criterion.  Keeping just one biased channel is a statistical truncation,
not the exact degree-four source.

At the audited \(0.161\%\) relative utilization per equivalent, co-firing
the zero-bias/rank-one screen with the held \(u=0.5629\) pack gives

\[
 \boxed{
 u_{\rm screen}\simeq0.5629(1+30\cdot0.00161)
 \simeq0.5901.}
 \tag{116.10}
\]

This remains below the current \(u\simeq0.6002\), leaving only about
\(1.7\%\) relative utility margin for every omitted operation.  At raw
\(10^{-8}\), the screened adjusted score is about
\(5.90\times10^{-9}\), below the \(6.5\times10^{-9}\) leader.  These are
screen economics, not a fire receipt.

The generic biased \(r_{\rm src}=3\) source costs 90 equivalents and screens
at \(u\simeq0.6445\).  It would leave only about \(0.85\%\) relative
headroom to the \(0.65\) leader utility threshold at raw \(10^{-8}\), before
all omitted costs; it is therefore cost-HOLD unless products fuse or the
projected channel rank drops.

### 116.4 Fidelity versus value

As a physical approximation to the full covariance Bellman row,
\(\Lambda^{(0)}\) is controlled by Section 115 and is generally poor over
ten resets unless the nonconstant Price modes are endpoint-dark.  That
does not invalidate (116.7) as a refitted feature.  Its exact one-feature
or joint-feature gain is determined only by the held-out Schur projection:

\[
 \Delta_{\rm raw}
 =b^TG^\dagger b,\qquad
 b={1\over N}F_\perp^Tr,\quad
 G={1\over N}F_\perp^TF_\perp. \tag{116.11}
\]

Reaching raw \(10^{-8}\) from \(2.9409\times10^{-8}\) requires

\[
 \boxed{
 b^TG^\dagger b\ge1.9409\times10^{-8},}
 \tag{116.12}
\]

equivalently joint old-residualized correlation at least \(0.8124\).
The 20% and 30% fire bars require correlations \(0.4472\) and \(0.5477\).

The candidate is therefore a sharp fail-fast screen:

1. source is the current-lineage residual, never total Gaussian q211;
2. query is the label-free constant Price mode, never called the exact
   D10 Bellman row;
3. all ten scalars are jointly refitted with no retuning after the sentinel;
4. total f32 package must stay below the remaining utility margin; and
5. failure of the held-out Schur gain closes this rank-one query, not the
   full K22 physics.

Graph-ready split:

- Q211_K22_CONSTANT_PRICE_QUERY -- **solved exact rank-one construction**,
  (116.1)--(116.3);
- Q211_K22_CONSTANT_PRICE_FEATURE -- **solved exact declared action**,
  (116.4)--(116.7);
- Q211_K22_CONSTANT_PRICE_OLD_OWNERSHIP -- **new only for literal residual
  q211 source; full Section 49 closes it**;
- Q211_K22_CONSTANT_PRICE_COST -- **30 f32-equivalent screen plus omitted
  work per projected source channel**, (116.8)--(116.10);
- Q211_K22_CONSTANT_PRICE_BIASED_RANK -- **generic three-channel /
  90-equivalent HOLD; rank-one only under Section 99 criterion**;
- Q211_K22_CONSTANT_PRICE_PHYSICS -- **not the exact deep Bellman row**;
- Q211_K22_CONSTANT_PRICE_RAW1E8 -- **open; requires joint correlation
  \(0.8124\)**, (116.11)--(116.12).

## 117. Audit of the constant-Price rank-one feature: normalization,
general gates, old ownership, and fusion limits

Section 116 is a lawful statistical query, but three normalizations must
not be conflated:

1. the terminal scalar-to-covariance adjoint;
2. the independent-gate factor used in the reverse covariance recursion;
3. the local K22 boundary-density factor at the source seam.

The first two coincide with a vector recursion.  The third is a second
rank-one weighting at zero correlation and was only an absorbable scalar
in the zero-bias version of (116.6).

### 117.1 Exact endpoint and symmetric-matrix normalization

If the declared scalar read is

\[
 J_D=\alpha\,c^TC_Dc, \tag{117.1}
\]

then, under the full symmetric Frobenius pairing,

\[
 \Lambda_D=\alpha cc^T
 =\epsilon_\alpha s_Ds_D^T,\qquad
 s_D=\sqrt{|\alpha|}\,c,\qquad
 \epsilon_\alpha=\operatorname{sign}\alpha. \tag{117.2}
\]

The signed factor \(\epsilon_\alpha\) multiplies the Section 112 action.
For an upper-triangular storage convention the off-diagonal derivative is
\(2\alpha c_ac_b\); conversion to the full Frobenius convention returns
exactly (117.2), not \(2cc^T\).  If the endpoint is a standard deviation,
normalization, or a sum of readouts, its analytic derivative replaces
\(\alpha\); a sum of \(r_D\) independent signed factors creates \(r_D\)
queries.

The vector must come from the deployed readout.  Choosing \(s_D\) to fit
MSE labels defines a supervised probe, not a Bellman adjoint.

Let the forward ordering at one reverse step be covariance reset followed
by the dense map \(W_{\ell+1}\).  If

\[
 g_{\ell,a}=P(G_{\ell,a}>\tau_{\ell,a}),
\]

the independent-gate Price kernel is \(g_\ell g_\ell^T\), and the exact
constant-mode recursion is

\[
\boxed{
 s_\ell=W_{\ell+1}^TD_{g_{\ell+1}}s_{\ell+1}.} \tag{117.3}
\]

At zero bias \(g=\frac12\mathbf1\), giving (116.2).  For a nonconstant
gate, \(D_g\) cannot be commuted through \(W^T\); a formula
\(D_gW^Ts\) refers to the opposite forward ordering.

For a general Gaussian activation \(\varphi\), replace \(g_a\) by
\(m_a=E[\varphi'(G_a)]\).  The constant reverse mode remains
\(W^TD_ms\), while the omitted kernel is
\(\operatorname{Cov}(\varphi'(G_a),\varphi'(G_b))\).  This is not an
exact AQ/non-Gaussian Price theorem without an independent closure
derivation.

### 117.2 The missing local K22 density normalization

For a standardized biased ReLU with threshold \(\tau_a\), put

\[
 d_a=\phi(\tau_a),\qquad
 \phi_{2,ab}
 =\phi_2(\tau_a,\tau_b;R_{ab}). \tag{117.4}
\]

The mixed fourth-cumulant coefficient is

\[
 {1\over2!2!}E[\delta(G_a-\tau_a)\delta(G_b-\tau_b)]
 ={1\over4}\phi_{2,ab}. \tag{117.5}
\]

Factor the joint density as

\[
 \phi_{2,ab}=d_ad_bD^\tau_{ab}, \tag{117.6}
\]

where, for \(r=R_{ab}\),

\[
\boxed{
 D^\tau_{ab}
 ={1\over\sqrt{1-r^2}}
 \exp\!\left\{
 {2r\tau_a\tau_b-r^2(\tau_a^2+\tau_b^2)
  \over2(1-r^2)}\right\}.} \tag{117.7}
\]

Thus the exact independent-density K22 factor associated with a covariance
mode \(ss^T\) is

\[
\boxed{
 \widetilde s={1\over2}(d\odot s),\qquad
 \Lambda_{22}^{(0)}
 =\widetilde s\,\widetilde s^T.} \tag{117.8}
\]

The residual local boundary kernel is \(D^\tau-\mathbf1\mathbf1^T\).
At zero bias, \(d=\mu\mathbf1\) and
\[
 \widetilde s={\mu\over2}s,\qquad
 D^\tau_{ab}=(1-R_{ab}^2)^{-1/2}. \tag{117.9}
\]

Therefore the physically normalized zero-bias layer feature has either of
the following two, and only the following two, equivalent conventions.  Let
\({\cal F}_\ell(v)\) denote (116.5)--(116.6) evaluated with factor \(v\):

\[
\boxed{
 f_\ell^{\rm phys}
 =c_{\rm sym}{\cal F}_\ell(\widetilde s_ell)
 =c_{\rm sym}{\mu^2\over4}{\cal F}_\ell(s_\ell),
 \qquad \widetilde s_\ell={\mu\over2}s_\ell.}       \tag{117.10}
\]

Here \(c_{\rm sym}\) is the single declared raw/deployed tensor
normalization and the equality uses the degree-two homogeneity of the
rank-one costate action.  If \(\widetilde s\) is inserted into (116.5),
the factor \(\mu^2/4\) is already present and must not be multiplied a
second time.  A free coefficient for every layer absorbs the common scale
in feature *span*, but not in a coefficient-one physical claim, a shared-
coefficient bank, or arbitrary bias.  At bias one must insert
\(\widetilde s_\ell\) into (116.5); no scalar rescaling reproduces the
coordinate-dependent \(d_\ell\).

For unstandardized preactivations,
\(d_a=\phi(\tau_a)/\sigma_a\) in raw cumulant coordinates.  If the
cumulants were standardized first, the corresponding powers of
\(\sigma_a\) are already in the source.  Mixing these conventions gives
a false layer-dependent amplitude.

If \(|R_{ab}|\le\rho<1\) and \(|\tau_a|\le T\), (117.7) gives the uniform
tail bound

\[
\boxed{
 \|D^\tau-\mathbf1\mathbf1^T\|_{\max}
 \le\delta_\tau:=
 {1\over\sqrt{1-\rho^2}}
 \exp\!\left({T^2\rho\over1-\rho}\right)-1.} \tag{117.11}
\]

At \(T=0\), this is the Section 112/114
\((1-\rho^2)^{-1/2}-1\) tail.

### 117.3 Exact approximation recurrence versus the true covariance row

Write the exact joint-gate Price matrix as

\[
 \Pi=gg^T+E_g,\qquad e_g=\|E_g\|_{\max}. \tag{117.12}
\]

At some reverse step decompose the true covariance row as
\(\Lambda=L+E\), with \(L=ss^T\) and
\[
 x={\|E\|_F\over\|L\|_F}.
\]

Before transport, the retained row is
\[
 L_0=D_gLD_g.
\]

Define the exact rank-one gate conditioning quantities

\[
\begin{aligned}
 \alpha_g&=
 {g_{\max}^2\|L\|_F\over\|D_gLD_g\|_F},\\
 \eta_g&=
 {e_g\|L\|_F\over\|D_gLD_g\|_F},\\
 \chi_W&=
 {\|W\|_{\rm op}^2\|D_gLD_g\|_F
  \over\|W^TD_gLD_gW\|_F}.
\end{aligned} \tag{117.13}
\]

Then the literal one-step tail obeys

\[
\boxed{
 x^-\le
 \chi_W\{(\alpha_g+\eta_g)x+\eta_g\}.} \tag{117.14}
\]

For zero bias,
\[
 \alpha_g=1,\qquad
 \eta_g={2\over\pi}\arcsin\rho,
\]
and (117.14) is exactly (115.8).

At the local K22 consumer, let
\[
 L_d=D_dLD_d,\qquad
 \kappa_d={d_{\max}^2\|L\|_F\over\|D_dLD_d\|_F}.
\]

Combining the reverse-row tail with (117.11) gives

\[
\boxed{
 x_{22}
 \le(1+\delta_\tau)\kappa_dx+\delta_\tau.} \tag{117.15}
\]

For zero bias \(\kappa_d=1\).  With ideal conditioning and \(h\) zero-bias
reverse resets from the endpoint,

\[
\boxed{
 x_{22,h}
 \le(1+\delta_D)(1+\eta_g)^h-1,\qquad
 \delta_D=(1-\rho^2)^{-1/2}-1.} \tag{117.16}
\]

Each layer feature in (116.7) has its own reverse age \(h\).  A source
consumed after one forward transition does not make an L5 costate one
step old.  Diagonal normalization, mean coupling, K31/K4 rows, and
non-Gaussian/AQ reverse channels are additional tails, not included in
(117.14)--(117.16).

Equations (117.14)--(117.16) are required only if the feature is claimed
as an approximation to the physical Bellman correction.  As a
preregistered refitted feature it may be physically incomplete, but it
must then pass the held-out Schur gain rather than inherit coefficient-one
credit.

### 117.4 What is and is not new relative to D11

The constant Price mode is already a component of the Gaussian covariance
Jacobian used by the old D10/D11 forward closure.  It receives no novelty
credit by itself.  The candidate can be new only through the source on
which that old response is queried.

Let \(\mathcal O_{\rm src,\ell}\) be the literal same-cut D11/CP3/pair/
CP4/Star/Q24 source interface.  The lawful source is

\[
 K_\ell^{\rm new}
 =(I-\Pi_{\mathcal O_{\rm src,\ell}})K_\ell^{\rm candidate}, \tag{117.17}
\]

with the deployed collision convention applied before subtraction.
Then:

1. if a full Section 49 q211 surface is the comparator,
   \(K_\ell^{\rm new}=0\);
2. if current D11 owns only the pairing-plane K22 projection, the strict
   residual remains a conditional candidate;
3. CP3 tree, pairing, CP4, Star4, and diagonal K4 pieces must not be
   reintroduced through \(P_\ell^{\rm new}\); and
4. even nonzero source residual does not prove output novelty.

The final bank must still be projected against the complete old *corpus*
span:

\[
 F_\perp=(I-\Pi_{\mathcal O_{\rm corpus}})F. \tag{117.18}
\]

Source residualization and output Schur residualization are different
operations; neither substitutes for the other.  The current literal audit
therefore says “conditional residual candidate,” not “D11-independent
theorem.”

### 117.5 Product fusion and the exact ten-layer bill

For one costate/source factor, recall

\[
 B=W^TD_sW,\qquad A=(W^{\odot2})^Ts.
\]

There is one exact small fusion:

\[
\boxed{A=\operatorname{diag}B.} \tag{117.19}
\]

Moreover, if the reverse recursion uses the same cut,

\[
 W^TD_s[\,W,\ g\,]
 =[\,B,\ W^T(g\odot s)\,], \tag{117.20}
\]

so the next vector adjoint can be appended to the \(B\) product for only
one extra column.  This removes its separate matrix-vector pass but no
full \(n^3\) equivalent.

The remaining generic products

\[
 RD_{\operatorname{diag}B}R,\qquad
 (B\odot R)^T(B\odot R) \tag{117.21}
\]

have different middle factors and outputs.  No literal current-D11
product matches them, and no ordinary one-product fusion follows without
additional \(R,B,P\) structure.  Across layers, block-diagonal batching
changes scheduling but not classical FMA billing.  The dual-\(P\) circuit
of Section 114 also costs three products per live \(P\)-factor.  Only a
proved small native-atom source activates the two-matvec scalar circuit
(114.23)--(114.26).

Let \(r_{\rm src}\) be the complete-old projected broad source-channel
rank and \(h_{\rm reuse}\) the number of the three products literally
resident per layer/channel.  The dominant ten-layer bill is

\[
\boxed{
 C_{\rm dom}=10(3-h_{\rm reuse})r_{\rm src}.} \tag{117.22}
\]

With no reuse, zero-bias \(r_{\rm src}=1\) costs \(30\) equivalents.  The
held \(u_0=0.5629\) pack and audited \(0.161\%\) relative charge give

\[
 u(C)=0.5629(1+0.00161C). \tag{117.23}
\]

Utility parity with \(u=0.6002\) permits total
\[
 C\le
 {0.6002/0.5629-1\over0.00161}
 \simeq41.2, \tag{117.24}
\]
so the rank-one screen has only about \(11.2\) equivalents left for every
elementwise kernel, hook, subtraction, and package operation.  At raw
\(10^{-8}\), beating adjusted \(6.5\times10^{-9}\) requires \(u<0.65\),
or

\[
 C<
 {0.65/0.5629-1\over0.00161}
 \simeq96.1. \tag{117.25}
\]

The generic biased \(r_{\rm src}=3\) bank consumes \(90\) equivalents and
has only about \(6.1\) equivalents of optimistic leader headroom; it
cannot meet utility parity.

### 117.6 Value gates

For the ten normalized layer features, after (117.17)--(117.18), the exact
frozen refit gain is still

\[
 \Delta_{\rm raw}=b^TG^\dagger b. \tag{117.26}
\]

The value thresholds are

\[
\boxed{\begin{array}{c|c}
 \text{raw objective}&
 \text{required old-residualized joint correlation}\\ \hline
 20\%\ \text{gain}&\sqrt{0.20}=0.4472\ldots\\
 30\%\ \text{gain}&\sqrt{0.30}=0.5477\ldots\\
 2.9409\times10^{-8}\to10^{-8}&0.8124\ldots
 \end{array}} \tag{117.27}
\]

If the feature is instead asserted to approximate an exact physical
correction of target correlation \(\rho_t\), its full post-old relative
error, including (117.15), source-channel truncation, and finite-repair
co-adaptation, must obey

\[
\boxed{
 \varepsilon_{\rm total}
 \le{\rho_t-0.8124\over1+0.8124}
 \le0.1035.} \tag{117.28}
\]

At \(C=30\) and raw \(10^{-8}\), the optimistic adjusted score is
\(5.90\times10^{-9}\).  This number is meaningful only after the omitted
bill fits (117.24)/(117.25) and the held-out gain passes (117.27).

Graph-ready split:

- CONST_PRICE_ENDPOINT_NORMALIZATION -- **solved exact**, (117.1)--
  (117.3);
- CONST_PRICE_LOCAL_K22_DENSITY -- **corrected exact normalization**,
  (117.4)--(117.11);
- CONST_PRICE_TRUE_REVERSE_TAIL -- **solved conditional**, (117.12)--
  (117.16);
- CONST_PRICE_D11_OWNERSHIP -- **old response mode; novelty only through
  literal source and corpus residuals**, (117.17)--(117.18);
- CONST_PRICE_PRODUCT_FUSION -- **only diagonal/augmented-column reuse
  generic**, (117.19)--(117.21);
- CONST_PRICE_TEN_LAYER_BILL -- **30 equivalents rank one / 90 generic
  biased source**, (117.22)--(117.25);
- CONST_PRICE_REFIT_VALUE -- **open**, exact gates (117.26)--(117.28);
- CONST_PRICE_PHYSICAL_VALUE -- **open and generally poor at deep reverse
  age without the tail/leverage receipt**.

## 118. Same-cut old quotient of the constant-Price K22 query: the
covariance factor is old, the trace lift need not be

Sections 116--117 identify a lawful rank-one covariance response, while
Section 121 isolates the leading intrinsic source on which to query it.
The remaining ownership question has three distinct levels:

1. D11 knows the covariance and its constant Price factor;
2. its typed source interface may or may not own the strict q211 lift; and
3. its emitted feature interface may or may not own the *K22 action* of
   that lift.

Only the third level makes a scalar subtraction free.  This section gives
the exact quotient and the synchronized Jacobian criterion separating the
three.

### 118.1 The leading intrinsic source is pairing minus trace

Let

\[
 \kappa_2={4-\pi\over8\pi}>0,
\]

and work on ordered strict q211 slots.  The leading intrinsic
\(Y-P_{\le2}Y\) source is

\[
 K_2{}_{i;jk}=-\kappa_2R_{ij}R_{ik}.                 \tag{118.1}
\]

Define the synchronized pairing and identity--covariance trace words by

\[
 ({\cal P}_R)_{iijk}=R_{jk}+2R_{ij}R_{ik},\qquad
 ({\cal Z}_R)_{iijk}=R_{jk}.                         \tag{118.2}
\]

Therefore

\[
 K_2=-{\kappa_2\over2}{\cal P}_R
     +{\kappa_2\over2}{\cal Z}_R                    \tag{118.3}
\]

on the strict q211 slice.  If the complete same-cut old source space
\({\cal O}_{\rm src}\) contains the literal covariance pairing lift, the
exact source quotient is

\[
 \boxed{
 [K_2]_{{\cal Q}_{211}/{\cal O}_{\rm src}}
 ={\kappa_2\over2}[{\cal Z}_R]_{{\cal Q}_{211}/{\cal O}_{\rm src}}.}
                                                               \tag{118.4}
\]

This does **not** say that the source is new information: \({\cal Z}_R\)
is a deterministic lift of resident covariance.  It says that the typed
readout can be absent.

At a coordinate-birth D11 seam the literal ownership census is decisive:

- the pairing state owns \({\cal P}_R\);
- coordinate Star4 and coordinate CP4 have zero strict three-label slice;
- the bivariate Q24 collision rows contain \(R_{jk}\) as pair data but do
  not emit the map \(R\mapsto\Pi_{211}\operatorname{Sym}(I\otimes R)\);
  and
- diagonal K4/K31/K22 rows sew collisions but do not supply the strict
  trace lift.

Hence only the covariance *factor* is old there; \({\cal Z}_R\) is not in
the literal strict source row span.  At a later seam, inherited CP4 rays
and every actual Star/Q24 row must be included in the projector.  A full
Section 49 q211 surface owns the whole response and makes (118.4) zero.

### 118.2 The constant term of the fused \(P^\sharp\) source is not wholly
pairing-old

The Section 96 broad kernel has the exact decomposition

\[
\begin{aligned}
 p^\sharp(r)&={1\over4}+br+\psi(r),\\
 b&={\pi-1\over2\pi^2},\qquad
 \psi(r)={\arcsin r-r\over2\pi}.                    \tag{118.5}
\end{aligned}
\]

Using \(R_{ij}R_{ik}=({\cal P}_R-{\cal Z}_R)_{iijk}/2\), its exact
coordinate-birth quotient, before later inherited-row projection, is

\[
\boxed{
 [\,R_{ij}R_{ik}p^\sharp(R_{jk})\,]_{{\cal Q}_{211}/{\cal O}_{\rm src}}
 =-{1\over8}[{\cal Z}_R]
  +b[\,R_{ij}R_{ik}R_{jk}\,]
  +[\,R_{ij}R_{ik}\psi(R_{jk})\,].}              \tag{118.6}
\]

Thus the tempting rule “drop \(J/4\) because pairing is old” is false: its
pairing part is old, but it leaves the exact trace residual
\(-{\cal Z}_R/8\).  The triangle and analytic tail are separate channels;
they must be projected against the literal inherited CP3/CP4/Q24 rows and
must not be stacked again with an already included Section 96 provider.

For \(|r|\le\rho<1\), the analytic tail has the deterministic bound

\[
 |\psi(r)|\le\kappa_\psi(\rho)|r|,qquad
 \kappa_\psi(\rho)
 ={\arcsin(\rho)/\rho-1\over2\pi}.                  \tag{118.7}
\]

Relative to the triangle coefficient \(b\), its endpoint action is
controlled by \(\kappa_\psi(\rho)/b\) times the corresponding absolute-
over-signed action condition number.  Small \(\rho\) alone is not a
relative receipt when the triangle action cancels.

### 118.3 Synchronized-site Jacobian and Schur theorem

Let \(\Theta\) be the local primitive tangent space at the actual cut,
let

\[
 G:\Theta\to{\cal Q}_{211}
\]

be the candidate source Jacobian, let
\(J:\Theta\to\mathbb R^p\) be the Jacobian of every literal old emitted
row at that same cut, and let
\(a\in{\cal Q}_{211}^*\) be the fixed K22 consumer after its mask and
collision convention.  The candidate scalar Jacobian is

\[
 g=aG.                                                \tag{118.8}
\]

Then the following are equivalent:

\[
\boxed{
 \ker J\subseteq\ker g
 \iff g\in\operatorname{row}J
 \iff g(I-J^+J)=0.}                                  \tag{118.9}
\]

This is the necessary-and-sufficient *same-site emitted ownership*
theorem.  With a positive tangent metric \(\Sigma\), its exact Schur
novelty is

\[
\boxed{\begin{aligned}
 S_{\rm site}={}&g\Sigma g^T
 -g\Sigma J^T(J\Sigma J^T)^+J\Sigma g^T,\\
 g_\perp={}&g-g\Sigma J^T(J\Sigma J^T)^+J.
\end{aligned}}                                       \tag{118.10}
\]

Now let \(\Pi_{\rm src}\) be any declared old-source projector.  Source
subtraction commutes lawfully with the emitted quotient if and only if

\[
 \boxed{
 a\Pi_{\rm src}G=CJ\quad\text{for some row }C.}      \tag{118.11}
\]

Equivalently,
\(\ker J\subseteq\ker(a\Pi_{\rm src}G)\).  Under (118.11),

\[
 a(I-\Pi_{\rm src})G=g-CJ,                           \tag{118.12}
\]

so evaluating the candidate and subtracting the resident old scalar is
exactly the physical source quotient.  If (118.11) fails, storage of the
pairing plane does not make its new K22 contraction old.  This is the
precise distinction missed by saying merely “D11 owns covariance.”

For the leading source, (118.11) requires the synchronized scalar
\(a{\cal P}_R\), not just \({\cal P}_R\) or \(R\).  The current
116.8/117.21 census supplies no such resident K22 product.  Thus the
mathematical source quotient (118.4) is valid, while a zero-cost emitted
subtraction remains unproved.  The final corpus projector is a third
operator and must still be applied after the layer scalars are formed.

If the consumer itself varies with the primitive state, replace (118.8)
by the full product rule

\[
 D(a(\theta)K(\theta))[h]
 =Da(\theta)[h]K(\theta)+a(\theta)DK(\theta)[h].      \tag{118.13}
\]

Calling only the second summand a same-site Jacobian is lawful precisely
on tangents that freeze the consumer or when the first summand is included
in \(J\).

### 118.4 Exact K22 scale and the corrected normalization

Let \({\mathfrak A}_{\ell}(K;s)\) denote the raw Section 112 pullback of
source \(K\) with rank-one covariance costate \(ss^T\).  It is linear in
\(K\) and degree two in \(s\).  At zero bias the local K22 density factor
is

\[
 t={\mu\over2}s,qquad \mu={1\over\sqrt{2\pi}},       \tag{118.14}
\]

so

\[
 {\mathfrak A}_{\ell}(K;t)
 ={\mu^2\over4}{\mathfrak A}_{\ell}(K;s).            \tag{118.15}
\]

After exact pairing subtraction, the physically normalized leading trace
feature is therefore

\[
\boxed{\begin{aligned}
 f_{\ell,\rm tr}
 &=c_{\rm sym}{\kappa_2\over2}
   {\mathfrak A}_{\ell}({\cal Z}_R;t)\\
 &=c_{\rm sym}{\kappa_2\mu^2\over8}
   {\mathfrak A}_{\ell}({\cal Z}_R;s)\\
 &=c_{\rm sym}{4-\pi\over128\pi^2}
   {\mathfrak A}_{\ell}({\cal Z}_R;s).
\end{aligned}}                                       \tag{118.16}
\]

The single \(c_{\rm sym}\) contains the declared raw/deployed tensor
normalization.  Equation (118.16) is equivalent to evaluating the original
source as

\[
 -c_{\rm sym}\kappa_2
 {\mathfrak A}_{\ell}(R_{ij}R_{ik};t)
 =-{c_{\rm sym}\kappa_2\over2}{\mathfrak A}_{\ell}({\cal P}_R;t)
  +f_{\ell,\rm tr}.                                  \tag{118.17}
\]

Only when the first scalar on the right is resident does output
subtraction produce (118.16) for free.  Equations (117.10) and (118.15)
also resolve the normalization ambiguity: either insert \(t\) and multiply
only by \(c_{\rm sym}\), or insert bare \(s\) and multiply by
\(c_{\rm sym}\mu^2/4\), never both.

### 118.5 A nonzero lawful physical witness

The residual is not merely a dimension count.  Take three labels with

\[
 R_{12}=R_{13}=r,\qquad R_{23}=-2r^2,\qquad
 0<r<{1\over2}.                                      \tag{118.18}
\]

The correlation determinant is \(1-2r^2-8r^4>0\).  Across the three q211
orientations \((1;23),(2;13),(3;12)\), the first pairing and trace
coordinates are

\[
 p_1=R_{23}+2R_{12}R_{13}=0,\qquad
 z_1=R_{23}=-2r^2.                                   \tag{118.19}
\]

Choose two distinct physical output rows of \(W\) to be

\[
 w^{(1)}=(1,0,0),\qquad w^{(2)}=(0,1,1),             \tag{118.20}
\]

and use the strict off-diagonal part of the rank-one-completion K22
costate \(ss^T\), \(s=(1,1)\).  Its diagonal output collisions are sewn
through the K4 row; its two cross orientations are retained.  Under the
raw convention (112.1)--(112.2), the resulting strict q211 pullback on
this triple is

\[
 \ell=(4,0,0).                                       \tag{118.21}
\]

Indeed, only
\(4(w^{(1)}_1)^2w^{(2)}_2w^{(2)}_3=4\) survives; every crossed \(8\)-term
contains \(w^{(1)}_iw^{(2)}_i=0\).  Consequently

\[
 \ell^Tp=0,\qquad \ell^Tz=-8r^2\ne0.                \tag{118.21a}
\]

Coordinate CP4 and Star rows vanish on the strict triple, and the typed
Q24 pair rows have no strict trace lift.  Thus (118.21a) is a lawful
rank-one K22 witness that annihilates the old pairing direction but sees
the trace residual.  At a later seam, replace \(\ell\) by its exact
projection against every inherited strict row; nonzero survives if and
only if the Schur quantity (118.10) is positive.

This is a readout-novelty witness, not an information-novelty tangent.
Holding \(R\) and the marginal reset data fixed also fixes the trace word,
so no covariance-fixing primitive tangent can change it.  The possible
credit is a missing typed consumer of resident information.

The higher intrinsic-source tail remains the exact Section 121 bound

\[
 |K_{\rm rem}+\kappa_2R_{ij}R_{ik}|
 \le{\rho^3\over\pi^2}+{5\rho^4\over12\pi}
     +\varepsilon_5(\rho).                           \tag{118.22}
\]

It must be propagated through the same old-residualized endpoint action;
it is not reduced by the nonzero witness.

Graph-ready split:

- CONST_PRICE_SAMECUT_PAIR_MINUS_TRACE -- **solved exact**, (118.1)--
  (118.4);
- CONST_PRICE_D11_FACTOR_VS_LIFT -- **pairing/covariance factor old but
  strict trace lift absent at coordinate birth; later conditional**;
- CONST_PRICE_PSHARP_SOURCE_QUOTIENT -- **solved exact**, constant term
  leaves \(-{\cal Z}_R/8\), (118.5)--(118.7);
- CONST_PRICE_SYNCHRONIZED_JACOBIAN -- **necessary and sufficient**,
  (118.8)--(118.9), (118.13);
- CONST_PRICE_SAMECUT_SCHUR -- **solved exact metric test**,
  (118.10);
- CONST_PRICE_SOURCE_OUTPUT_COMMUTATION -- **conditional iff**,
  (118.11)--(118.12);
- CONST_PRICE_K22_EXACT_SCALE -- **corrected; no double density factor**,
  (118.14)--(118.17);
- CONST_PRICE_TRACE_NONZERO_WITNESS -- **proved for a physical rank-one-
  completion K22 endpoint**, (118.18)--(118.21a);
- CONST_PRICE_TRACE_INFORMATION -- **closed as new information; open as a
  typed readout**;
- CONST_PRICE_TRACE_FULL_SURFACE -- **closed duplicate when Section 49 is
  in the billed comparator**;
- CONST_PRICE_TRACE_VALUE -- **open** after complete inherited-row and
  corpus Schur projection; tail (118.22) remains binding.

## 119. Closing the canonical-Cholesky/direct-K21 survivor: the cubic
fixed point, covariant right inverse, and exact bill

Section 113 leaves only a four-reset branch with a cheap lawful covariance
anchor and a literal K21 section which does not evaluate or iteratively refit
the predicted K21 plane.  The most favorable place to seek such a section is
the affine CP seed, where K21 is linear in the quadratic coefficients to
first order and the projected-Veronese minor is generically onto.  Even there,
the exact map contains a cubic self-interaction.  This section isolates it
and closes the generic canonical right-inverse implementation.

### 119.1 The exact affine-seed equation is linear plus one cubic

Fix \(U=F\), write \(g_i\) for row \(i\) of \(F\), and let
\(\mathbf A=(A_1,\ldots,A_n)\).  Define

\[
\boxed{
 (\mathcal L_F\mathbf A)_{ij}
 =g_i^TA_jg_i+2g_i^TA_ig_j,}
\tag{119.1}
\]

and

\[
\boxed{
 \mathcal Q(\mathbf A)_{ij}
 =\operatorname{tr}(A_i^2A_j).}
\tag{119.2}
\]

The exact AQ repeated-cubic plane is

\[
\boxed{K21(\mathbf A)=\mathcal L_F\mathbf A+\mathcal Q(\mathbf A).}
\tag{119.3}
\]

For the strict plane there is a decoupled linear right inverse.  Put

\[
 E_j=\{A\in\operatorname{Sym}_r:Ag_j=0\},
\qquad
 \Phi_j(A)=\{g_i^TAg_i\}_{i\ne j}. \tag{119.4}
\]

When the projected-Veronese condition (113.25) holds, every \(\Phi_j\) is
onto.  Its canonical Frobenius minimum-norm inverse is

\[
\boxed{
 \mathcal R_j
 =\Phi_j^*(\Phi_j\Phi_j^*)^{-1}.}
\tag{119.5}
\]

For a strict target \(K\), setting

\[
 A_j=\mathcal R_jK_{-j,j} \tag{119.6}
\]

gives \(A_jg_j=0\), kills the row-pollution term in (119.1), and obeys

\[
 \mathcal L_F\mathbf A=K. \tag{119.7}
\]

It does **not** generally obey \(K21(\mathbf A)=K\), because
\(\mathcal Q(\mathbf A)\ne0\).  If
\(\mathcal R_F\) denotes (119.5)--(119.6) columnwise, an exact section in
its range must solve the cubic fixed-point equation

\[
\boxed{
 \mathbf A
 =\mathcal R_F\{K-\mathcal M_{\rm off}\mathcal Q(\mathbf A)\}.}
\tag{119.8}
\]

The full plane, including \(K_{iii}\), has the same form with the coupled
full right inverse from (113.23); the strict spelling is the cheaper and
therefore more favorable cap case.

### 119.2 The Moore--Penrose section is exactly gauge covariant

Let a neuron permutation be \(\Pi\), and let \(R\in O(r)\) be the inherited
signed-permutation or declared latent orthogonal action.  Under

\[
 F'=\Pi FR,\qquad K'=\Pi K\Pi^T,\qquad
 A_i'=R^TA_{\pi^{-1}(i)}R, \tag{119.9}
\]

the maps \(\Phi_j\) are conjugated by orthogonal domain and codomain maps.
The Moore--Penrose inverse is functorial under such conjugations.  Therefore

\[
\boxed{
 \mathcal R_{F'}(\Pi K\Pi^T)
 =\Pi\{R^T\mathcal R_F(K)R\}.}
\tag{119.10}
\]

The braces in (119.10) act bankwise.  Since \(\mathcal Q\) has the same
equivariance, every uniquely selected solution of (119.8) is lawful.
This proves that no lexicographic latent basis is needed.

Lawfulness is not global regularity.  It also requires:

1. the CP ray-selection gap;
2. every projected-Veronese singular floor
   \(\sigma_j=\sigma_{\min}(\Phi_j)>0\);
3. the common neuron-order gap if the Cholesky anchor (113.7) is used; and
4. a unique fixed-point/branch domain which does not cross a discriminant.

At a tie or a vanishing singular value the canonical map is discontinuous or
undefined; label-based tie breaking is not a repair.

### 119.3 Exact local conditioning and the infinite odd section

For the strict linear right inverse,

\[
 \|\mathcal R_F\|=\max_j\sigma_j^{-1}. \tag{119.11}
\]

No right inverse has a smaller operator norm than the reciprocal restricted
singular floor.  Perturbing the CP frame changes its pseudoinverse at scale
\(O(\sigma_j^{-2}\|\delta\Phi_j\|)\), before the CP selection-gap
conditioning.

Let

\[
 M=\max_i\|A_i\|_F,\qquad
 \|\mathbf H\|^2=\sum_i\|H_i\|_F^2.
\]

The derivative of (119.2) is

\[
 D\mathcal Q_{\mathbf A}[\mathbf H]_{ij}
 =\operatorname{tr}\{(H_iA_i+A_iH_i)A_j+A_i^2H_j\}. \tag{119.12}
\]

The Frobenius bound

\[
 |D\mathcal Q_{\mathbf A}[\mathbf H]_{ij}|
 \le M^2(2\|H_i\|_F+\|H_j\|_F)
\]

gives

\[
\boxed{
 \|D\mathcal Q_{\mathbf A}\|
 \le3\sqrt n\,M^2.}
\tag{119.13}
\]

Consequently (119.8) is locally contractive on a declared ball whenever

\[
\boxed{
 \eta:=3\sqrt n\,\|\mathcal R_F\|M^2<1.}
\tag{119.14}
\]

On that branch,

\[
 D_K\mathbf A
 =(I+\mathcal R_F\mathcal M_{\rm off}
 D\mathcal Q_{\mathbf A})^{-1}\mathcal R_F,
\qquad
 \|D_K\mathbf A\|
 \le{\|\mathcal R_F\|\over1-\eta}. \tag{119.15}
\]

At a general full-covariance anchor, replace the denominator in (119.15) by
the smallest singular value of the horizontal joint differential (113.23).
Any local section has Lipschitz constant at least its reciprocal.  The polar
anchor adds the spectral conditions

\[
 \|D(C^{1/2})\|\le{1\over2\sqrt{\lambda_{\min}(C)}},
\qquad
 \|D(K_H^{-1/2})\|
 \le{1\over2\lambda_{\min}(K_H)^{3/2}}. \tag{119.16}
\]

The canonical-order Cholesky branch avoids the spectral root bill but not
the same covariance conditioning; its ordering is locally constant only
inside the strict key-gap cell.

The direct section is generically not a finite polynomial or a finite
Taylor jet.  This is already forced by the one-unit restriction

\[
 k=3u^2a+a^3. \tag{119.17}
\]

With \(L=3u^2>0\), its analytic inverse begins

\[
 a(k)={k\over L}-{k^3\over L^4}
      +{3k^5\over L^7}-\cdots, \tag{119.18}
\]

with nonzero coefficients at arbitrarily high odd orders.  Hence a universal
finite polynomial identity for (119.8) is impossible unless its cubic
interaction is killed on the selected range.  A scalar cubic-root primitive
solves (119.17); it does not supply a simultaneous cheap solution of the
coupled \(n^2\)-equation system.

### 119.4 Resident K21 removes acquisition, not the section bill

The D14 cut may expose the exact target K21 plane.  Reusing it is lawful and
eliminates any charge for *forming the target*.  It does not make
(119.5) free.

For \(r=63\),

\[
 \dim E_j=e-r=1953,\qquad \dim\operatorname{codom}\Phi_j=n-1=1023.
 \tag{119.19}
\]

There are \(n\) different projected-Veronese systems.  Caching only their
\((n-1)\times(n-1)\) normal matrices requires

\[
\boxed{
 n(n-1)^2
 =1{,}071{,}645{,}696\ \text{scalars},}
\tag{119.20}
\]

about 3.99 GiB at f32, before factors or the AQ bank.  Forming/factoring them
on card is far beyond the suffix bill.

Even granting every inverse and factor for free, a generic dense map from
each strict K21 column to the literal full \(e\)-coordinate \(A_j\) costs

\[
\boxed{
 C_{\rm strict\ direct\ apply}
 \ge {e(n-1)\over n^2}
 ={2016\cdot1023\over1024^2}
 =1.966827392578125}
\tag{119.21}
\]

f32 matmul-equivalents in the declared dense-linear-map census.  The
optimistic canonical-Cholesky L4 anchor from (113.39) therefore satisfies

\[
\boxed{
 45.68054+1.3948567708+1.9668273926
 =49.0422241634>49,}
\tag{119.22}
\]

before the cubic correction, covariance continuation, K22 selection,
conditioning hooks, or storage traffic.  For the full K21 plane the dense
application floor is \(e/n=1.96875\), slightly larger.

Equation (119.21) is a closure of the canonical generic projected-Veronese
right inverse, not an unconditional arithmetic-circuit lower bound.  A
shared sparse transform, common diagonalization, or literal fusion with the
first AQ push could beat it, but must exhibit that structure at the actual
CP frame.  Merely calling K21 resident does not.

Every nonlinear fixed-point step is more expensive still.  Evaluating
\(\mathcal Q(\mathbf A)=YX^T\), with
\(Y_i=\operatorname{svec}(A_i^2)\), costs \(e/n=1.96875\) equivalents
before local squares; applying \(\mathcal R_F\) then repeats (119.21).
Thus neither Picard nor Newton realization of (119.8) fits.

### 119.5 The exact nil-cubic escape

There is one algebraic way for the affine right inverse to become a literal
one-shot section.  Its range must satisfy

\[
\boxed{
 \operatorname{tr}(A_i^2A_j)=0\quad\forall i,j,}
\tag{119.23}
\]

or a stronger universal support identity

\[
 \operatorname{tr}(A^2B)=0\quad\forall A,B\in E. \tag{119.24}
\]

Then \(\mathcal Q=0\) and (119.6) is exact.  A concrete cubic-null support is
the bipartite off-block space

\[
 E_{a,b}=
 \left\{
 \begin{pmatrix}0&B\\B^T&0\end{pmatrix}:B\in\mathbb R^{a\times b}
 \right\},\qquad a+b=r. \tag{119.25}
\]

Its square is block diagonal, so (119.24) holds, and

\[
 \dim E_{a,b}=ab\le\left\lfloor{r^2\over4}\right\rfloor. \tag{119.26}
\]

At \(r=63\), the best split \(31+32\) gives only 992 directions.  The
strict projected-evaluation target needs at least \(n-1=1023\) directions
per column and the full target needs 1024.  Thus the canonical bipartite
nil-cubic escape is short by 31 and 32 directions respectively.

There is a useful general obstruction.  Polarizing (119.24) gives

\[
 \langle AB+BA,C\rangle_F=0
 \quad\forall A,B,C\in E. \tag{119.27}
\]

Let \(N=\dim\operatorname{Sym}_{63}=2016\) and
\(\mathcal L_A(B)=AB+BA\).  Equation (119.27) says
\(\mathcal L_A(E)\subseteq E^\perp\), so rank--nullity gives, for every
\(A\in E\),

\[
\boxed{
 2\dim E
 \le N+\dim(E\cap\ker\mathcal L_A).}
\tag{119.28}
\]

If \(E\) contains even one Jordan-regular \(A\), meaning no two eigenvalues
of \(A\) sum to zero, then \(\mathcal L_A\) is invertible and
\[
 \dim E\le N/2=1008.
\]
That is already below the strict 1023-direction gate.  Therefore any
cubic-null escape large enough for strict/full K21 must be spectrally
exceptional at **every** point:

\[
\boxed{
\begin{array}{c|c}
\dim E\ge1023&
\dim(E\cap\ker\mathcal L_A)\ge30\quad\forall A\in E\\
\dim E\ge1024&
\dim(E\cap\ker\mathcal L_A)\ge32\quad\forall A\in E .
\end{array}}
\tag{119.29}
\]

If the eigenvalue multiplicities of \(A\) are \(m_\lambda\), the ambient
Jordan nullity is exactly

\[
 \dim\ker\mathcal L_A
 ={m_0(m_0+1)\over2}
 +\sum_{\lambda>0}m_\lambda m_{-\lambda}. \tag{119.30}
\]

Thus a surviving range needs a uniform large zero/opposite-eigenvalue
anticommutant, not merely zero average cubic trace.  This is a sharp
symbolic gate for any proposed larger construction.

For the **canonical decoupled strict section**, there is a stronger
columnwise form of this obstruction.  Let
\(\mathcal R_j:\mathbb R^{n-1}\to\operatorname{Sym}_r\) be its column
right inverse and put \(E_j=\operatorname{im}\mathcal R_j\).  The identity
\(\Phi_j\mathcal R_j=I_{n-1}\) makes \(\mathcal R_j\) injective, while the
decoupling constraint gives

\[
 \dim E_j=n-1=1023,
 \qquad
 E_j\subseteq\{A:Ag_j=0\}\simeq\operatorname{Sym}_{62}. \tag{119.31}
\]

If the proposed one-shot section uses the universal nil identity
(119.23), including its diagonal specialization, then varying one column
shows \(\operatorname{tr}(A^3)=0\) for every \(A\in E_j\).  Polarization
therefore gives (119.24) on each \(E_j\).  Applying (119.28) in
\(\operatorname{Sym}_{62}\), whose dimension is 1953 rather than 2016,
forces the much larger pointwise defect

\[
 \boxed{
 \dim(E_j\cap\ker\mathcal L_A)\ge
 2(1023)-1953=93
 \quad\text{for every }A\in E_j .}
 \tag{119.32}
\]

There is also a topology-free-of-conditioning check on the equality
case.  Suppose, contrary to the check, that
\(\mathcal L_A:E_j\to E_j^\perp\) were onto for every nonzero \(A\).
Over \(\mathbb {RP}^{1022}\) these maps would form the exact bundle
sequence

\[
 0\longrightarrow\mathcal K_{93}
 \longrightarrow 1023\,\gamma
 \longrightarrow \underline{\mathbb R}^{930}
 \longrightarrow0, \tag{119.33}
\]

where \(\gamma\) is the tautological real line bundle.  Hence
\(w(\mathcal K)=(1+x)^{1023}\).  Since 1023 is \(2^{10}-1\), Lucas's
parity rule makes every \(\binom{1023}{s}\) odd; in particular
\(w_{94}(\mathcal K)=x^{94}\ne0\), impossible for a rank-93 bundle.
Consequently every such \(E_j\) contains a nonzero point with

\[
 \dim(E_j\cap\ker\mathcal L_A)\ge94. \tag{119.34}
\]

Equations (119.31)--(119.34) do **not** prove the conjectural sharp
classification bound \(\dim E\le992\) for cubic-null
\(E\subset\operatorname{Sym}_{62}\).  They do prove that any construction
closing the 31-direction gap above (119.26) cannot be a regular
constant-minimal-kernel deformation of the bipartite model: it must cross
a higher Jordan-rank degeneracy.  They also apply only when the universal
diagonal nil identity (119.23) is claimed.  A strict-off-diagonal-only
section may instead claim typed cross-null identities between different
column images; that weaker escape must be audited separately and may not
be rejected by (119.31) alone.

A larger cubic-null support, or a target-specific right-inverse range
satisfying (119.23) without a common support, is an exact mathematical
reopen.  It must additionally satisfy the projected-Veronese onto gate,
the full-covariance/gaussian-rest ownership, the target-conditioned K22
gate, and a sub-49 structured application circuit.  Dimension alone does
not supply it.

### 119.6 Sufficiency and final decision tree

The resident triple \((C,K21,F)\) is a sufficient statistic for a **weak**
AQ seam only after specifying an exact covariant section such as (119.8)
and all of its branch/singular-value gates.  It is not sufficient for a
future-equivalent seam: the 553,567-dimensional full weak fiber contains
states with different K22/q4 responses.  The CP4/Q24 endpoint statistic or
the effective Bellman quotient (113.19) remains necessary.

The exact verdict is:

1. **generic polar L5:** closed by (113.37)--(113.38);
2. **iterative K21 refit, L4 or L5:** closed by (113.35);
3. **canonical Cholesky plus generic projected-Veronese direct section,
   L4:** closed by (119.20)--(119.22), even with the target K21 resident;
4. **finite polynomial/jet section:** generically false by
   (119.17)--(119.18);
5. **remaining reopen:** an explicit common sparse/fused right inverse or
   nil-cubic range satisfying (119.23), together with covariance, K22,
   conditioning, and value receipts.

Graph-ready split:

- AQ63_AFFINE_K21_CUBIC_FIXED_POINT -- **solved exact**, (119.1)--(119.8);
- AQ63_K21_RIGHT_INVERSE_EQUIVARIANCE -- **solved exact**, (119.9)--(119.10);
- AQ63_DIRECT_SECTION_CONDITIONING -- **solved exact gate**,
  (119.11)--(119.16);
- AQ63_FINITE_JET_DIRECT_SECTION_IN_AFFINE_CHART -- **generic NO-GO**,
  (119.17)--(119.18);
- AQ63_RESIDENT_K21_SUFFICIENCY -- **weak seam only**;
- AQ63_GENERIC_DIRECT_SECTION_STORAGE -- **NO-GO**, (119.19)--(119.20);
- AQ63_CANON_CHOL_DIRECT_APPLY_CAP -- **NO-GO**, (119.21)--(119.22);
- AQ63_NIL_CUBIC_SECTION -- **conditional exceptional reopen**,
  (119.23)--(119.30);
- AQ63_CP4_DIRECT_SECTION_FINAL -- **closed generically; exceptional
  sparse/fused or nil-cubic certificate only**.

## 120. Rank-one K22 pullback: one-product leaf factors, the exact
two-product correlation plane, and strict collision accounting

The three-product circuit (112.6) is sharp for an algebraically generic
dense leaf plane \(P\), but it is not sharp when the q211 source itself has
rank one or when \(P\) is the transported correlation plane.  This section
gives the exact reductions for the physically normalized rank-one costate
of Section 117.  All one-product statements are in the complete-old
quotient: source-label collisions must already be synchronized with their
literal K4/K31/K22 owner.  Section 120.2 shows what reopens when that premise
is removed.

### 120.1 One dense product for a rank-one leaf plane

Let the terminal covariance row be

\[
 \Lambda_C=\alpha cc^T
 =\epsilon_s ss^T,\qquad
 s=\sqrt{|\alpha|}\,c,qquad
 \epsilon_s=\operatorname{sign}\alpha,
 \tag{120.1}
\]

and insert the local independent-density factor exactly as in (117.8):

\[
 t={1\over2}\{\phi(\tau)\odot s\}.
 \tag{120.2}
\]

Thus the K22 costate used below is \(\epsilon_s tt^T\), not the bare
\(\alpha cc^T\).  Non-unit marginal scales and the single deployed
symmetrization constant remain outside the following raw convention.

Consider the signed rank-one leaf source

\[
 K^{(u)}_{i;jk}
 =\eta q_iR_{ij}R_{ik}u_ju_k,
 \tag{120.3}
\]

where the arbitrary anchor weight \(q\) is part of the source, not part of
the costate.  Define, using rows \(w_a^T\) of \(W\),

\[
 A_i=\sum_a t_aW_{ai}^2,\qquad
 H=RD_uW^T,qquad
 z_i=\sum_a t_aW_{ai}H_{ia}.
 \tag{120.4}
\]

Substitution into the two lines of (112.2) gives the exact identity

\[
\boxed{
 {\cal S}(q,u)
 =\epsilon_s\eta\left\{
 4\sum_{a,i}t_aq_iA_iH_{ia}^2
 +8\sum_iq_i z_i^2
 \right\}.}
 \tag{120.5}
\]

The first line follows because the two leaf sums at output row \(a\) are
both \(H_{ia}\); the crossed line is the square of \(z_i\).  Hence (120.5)
needs exactly one ordinary dense product, \(H=RD_uW^T\), followed by
\(O(n^2)\) contractions.  It does **not** need
\(B=W^TD_tW\), \(RD_AR\), or a Gram product.  If this exact \(H\), with the
same cut, mask, dtype, and diagonal \(u\), is resident, its incremental
dense-product count is zero.

More generally, for a signed leaf decomposition

\[
 P=\sum_{r=1}^{r_P}\eta_ru_ru_r^T,
 \tag{120.6}
\]

linearity gives one product \(H_r=RD_{u_r}W^T\) per independent leaf
factor.  Therefore the attained count is

\[
 \boxed{C_{\rm leaf}=r_P-r_{\rm literal\ reuse},}
 \tag{120.7}
\]

or the smaller generic dense-plane circuit when \(r_P>3\).  A symmetric
bilinear atom \((uv^T+vu^T)/2\) uses the two products \(H_u,H_v\) and
replaces squares in (120.5) by the corresponding cross products.  A
spectral factor may be rescaled \(u_r\mapsto\gamma_ru_r\),
\(\eta_r\mapsto\eta_r/\gamma_r^2\); balancing these factors avoids an
artificial range loss but does not change the exact action.

This theorem concerns the leaf-plane rank of \(P\).  The zero-bias
\(P^\sharp=p[R]-2\mu^4R\) in Section 116 is generically full spectral
rank.  Calling it one *product channel* does not activate (120.5).

### 120.2 Exact source and endpoint collision formulas

Formula (120.5) first sums all labels.  The output K22 diagonal
\(a=b\) is removed at no new dense-product cost:

\[
 \boxed{
 D_0=12\epsilon_s\eta
 \sum_{a,i}t_a^2q_iW_{ai}^2H_{ia}^2.}
 \tag{120.8}
\]

Thus the output-strict action is \({\cal S}-D_0\), using the same \(H\).
Source strictness is different.  Put

\[
 B=W^TD_tW,qquad A=\operatorname{diag}B.
 \tag{120.9}
\]

The three source collision faces of (120.3), with the full output row, are

\[
\begin{aligned}
 E_{i=j}=E_{i=k}
 &=12\epsilon_s\eta\sum_{i,k}
 q_iu_iu_kR_{ik}A_iB_{ik},\\
 E_{j=k}
 &=\epsilon_s\eta\left\{
 4\sum_{i,j}q_iR_{ij}^2u_j^2A_iA_j
 +8\sum_{i,j}q_iR_{ij}^2u_j^2B_{ij}^2\right\},\\
 E_{\rm all}
 &=12\epsilon_s\eta\sum_iq_iu_i^2A_i^2.
\end{aligned}
 \tag{120.10}
\]

In the ordered-\((j,k)\) normalization of Section 112, inclusion--exclusion
is therefore

\[
\boxed{
 {\cal S}_{\rm src\,strict}
 ={\cal S}-2E_{i=j}-E_{j=k}+2E_{\rm all}.}
 \tag{120.11}
\]

Equations (120.10)--(120.11) are the precise caveat to the one-product
claim.  Standalone source strictness generally reopens \(B\), a second
dense product.  It is zero-new-product only when these lower-label rows
are literally synchronized and sewn through the declared collision
interface.

For completeness, the endpoint-diagonal parts of the three source faces
are

\[
\begin{aligned}
 D_{i=j}&=12\epsilon_s\eta\sum_{a,i}
 t_a^2q_iu_iW_{ai}^3H_{ia},\\
 D_{j=k}&=12\epsilon_s\eta\sum_{a,i,j}
 t_a^2q_iR_{ij}^2u_j^2W_{ai}^2W_{aj}^2,\\
 D_{\rm all}&=12\epsilon_s\eta\sum_{a,i}
 t_a^2q_iu_i^2W_{ai}^4.
\end{aligned}
 \tag{120.12}
\]

The exact action strict in both source and output labels is

\[
\boxed{
 ({\cal S}-D_0)-2(E_{i=j}-D_{i=j})
 -(E_{j=k}-D_{j=k})+2(E_{\rm all}-D_{\rm all}).}
 \tag{120.13}
\]

The middle quantity in (120.12) contains
\((W^{\odot2})^TD_{t^{\odot2}}W^{\odot2}\) if evaluated as a standalone
dense contraction.  Consequently source and output masks must be applied
to the same collision interface; subtracting the unmasked \(D_0\) after a
source-strict shortcut is not exact.  The deployable one-product theorem
is (120.5), (120.8) **modulo** these complete-old collision rows, not an
unconditional strict-source circuit.

### 120.3 Two products when the leaf plane is the correlation plane

There is a second exact collapse which does not require low spectral rank.
For the diagonally decorated correlation plane

\[
 P=D_uRD_u,qquad
 C=B\odot R,qquad
 Z=CD_uR,
 \tag{120.14}
\]

the anchor-
\(q\) generalization of (112.6) is

\[
\boxed{\begin{aligned}
 {\cal S}_{P=D_uRD_u}
 =\epsilon_s\{&4\operatorname{tr}
   (Z D_{q\odot A}R D_u)\\
 &+8\operatorname{tr}(Z D_u C D_q)\}.
\end{aligned}}
 \tag{120.15}
\]

Only \(B=W^TD_tW\) and \(Z=(B\odot R)D_uR\) are dense products.  All
remaining operations are diagonal or \(O(n^2)\).  In the undecorated case
\(u=\mathbf1\), let \(Y=CR\); then

\[
 \boxed{
 {\cal S}_{P=R}
 =4\epsilon_s\operatorname{tr}(YD_{q\odot A}R)
 +8\epsilon_s\sum_iq_i(YC)_{ii}.}
 \tag{120.16}
\]

This is an exact two-product circuit for the zero-bias \(xyz\) jet of
Section 99.  It is not an exact replacement for the all-orders
\(P^\sharp\) surface.

On strict off-diagonal leaf labels,

\[
 P^\sharp={1\over4}J+bR+E_3,qquad
 b={\pi-1\over2\pi^2},qquad
 (E_3)_{jk}={\arcsin R_{jk}-R_{jk}\over2\pi}.
 \tag{120.17}
\]

The \(J\) term uses (120.5) with \(u=\mathbf1\); the \(R\) term uses
(120.16).  Thus the \(J+bR\) approximation costs three products if neither
piece is resident, two if the constant plane or \(RW^T\) is literally
old/resident, and only the two-product \(R\) circuit if complete-old
projection kills \(J\).  For \(|R_{jk}|\le\rho<1\),

\[
 \|E_3\|_F
 \le\sqrt{n(n-1)}\,{\arcsin\rho-\rho\over2\pi}.
 \tag{120.18}
\]

Writing the generic action as \({\cal A}(P)=\langle P,Q_P\rangle_F\),

\[
 Q_P=4(B\odot RD_{q\odot A}R)
 +8(B\odot R)^TD_q(B\odot R),
 \tag{120.19}
\]

gives the sharp dual tail certificate

\[
 \boxed{
 |{\cal A}(E_3)|
 \le\|Q_P\|_F\sqrt{n(n-1)}
 {\arcsin\rho-\rho\over2\pi}.}
 \tag{120.20}
\]

The quotient-weighted version of (120.20), not the entrywise
\(O(\rho^3)\) slogan, is the binding fidelity gate.

### 120.4 General decorated edges and biased-channel sharing

The exact sharing rule follows by allowing distinct symmetric anchor-edge
matrices (X,Y) and a symmetric leaf plane (P).
For

\[
 K_{i;jk}=q_iP_{jk}
 {X_{ij}Y_{ik}+Y_{ij}X_{ik}\over2},
 \tag{120.21}
\]

put \(C_X=B\odot X\), \(C_Y=B\odot Y\).  The rank-one-costate action is

\[
\boxed{\begin{aligned}
 {\cal S}_{X,Y,P}={}&
 4\epsilon_s\langle B\odot P,
     X^TD_{q\odot A}Y\rangle_F\\
 &+8\epsilon_s\langle P,
     C_X^TD_qC_Y\rangle_F.
\end{aligned}}
 \tag{120.22}
\]

The displayed nonsymmetric products are harmless because their symmetric
transpose has the same pairing with symmetric \(B\odot P\) or \(P\).
One universal \(B\) is shared by every channel with the same
\((W,t)\).  Each distinct signature

\[
 (X,Y,q,W,t,\hbox{cut},\hbox{mask},\hbox{dtype})
 \tag{120.23}
\]

then needs two post-\(B\) products.  Changing only \(P\) changes the final
Frobenius read and is free at dense-product order.  Proportional \(q\)'s
also share, after moving the scalar into the final read.  Different anchor
edge powers or nonproportional anchor diagonals do not share.

Consequently \(g\) generic signatures cost

\[
 \boxed{C_{\rm generic\,bank}=1+2g-r_{\rm literal\ reuse}}
 \tag{120.24}
\]

per seam, rather than \(3g\).  For the three broad biased degree-four
channels in (99.17), the attained counts are sharper still:

* the base \(xyz\) channel uses the special two-product circuit (120.15);
* the side-powered channel needs its two post-\(B\) products;
* the leaf-powered channel needs its two post-\(B\) products.

Thus the generic three-channel upper bound is six products per seam, not
nine.  If \(h^F_2=\lambda d_2\), the base and leaf channels share the two
generic post-\(B\) products and the count is five.  If the side channel is
old/zero as well, it is three; at zero bias \(b_3=0\), only the base
\(xyz\) jet remains and its exact count is two.  These are degree-four
jet counts.  An all-orders biased surface needs its own separation-rank
certificate.

For an algebraically generic dense \(P\), (112.6) remains the attained
three-product circuit.  In the conventional separated \(n\times n\)
GEMM model, after forming generic \(B\), the two independent bilinear
channels

\[
 RD_{q\odot A}R,qquad
 (B\odot R)^TD_q(B\odot R)
 \tag{120.25}
\]

have bilinear coefficient rank two on a two-coordinate restriction; one
ordinary product supplies rank one and necessarily introduces a cross
term.  Hence two post-\(B\) products, and three total, are necessary in
that model for generic data.  This is not an unconditional arithmetic-
circuit lower bound: (120.5) and (120.15) are explicit structured
counterexamples, and a doubled-inner-dimension block multiplication is
billed as two ordinary equivalents.

### 120.5 Conditioning, state, and the Cap49/value gates

No inverse occurs in (120.5).  A useful signed-cancellation condition
number is

\[
 \kappa_{\rm act}=
 {4\sum_{a,i}|t_aq_iA_i|H_{ia}^2
 +8\sum_i|q_i|
    (\sum_a|t_aW_{ai}H_{ia}|)^2
 \over
 |4\sum_{a,i}t_aq_iA_iH_{ia}^2+8\sum_iq_iz_i^2|}.
 \tag{120.26}
\]

Forward error is \(O(nu_{\rm mach}\kappa_{\rm act})\) under standard
GEMM accumulation, plus the factorization condition

\[
 \kappa_P={\sum_r|\eta_r|\|u_r\|_2^2\over\|P\|_F}
 \tag{120.27}
\]

for a signed decomposition.  Strict collision subtraction has its own
absolute-sum-over-net-action condition number and must be included in this
gate.  The boundary-density tail (117.11) is independent of arithmetic
conditioning.

One rank-one source needs one transient \(n\times n\) buffer \(H\), plus
\(A,q,t,u,z\); sources can be streamed so this state does not multiply by
the number of seams.  At \(n=1024\), that transient is 4,194,304 bytes at
f32.  A resident \(H\) is reusable only for the exact same source factor
and cut.  For the two-product \(P=D_uRD_u\) circuit, save
\(A=\operatorname{diag}B\), overwrite \(B\) by \(C=B\odot R\), and allocate
\(Z\).  It therefore needs two recyclable \(n\times n\) transients (8 MiB
at f32), not three; both trace reads in (120.15) are \(O(n^2)\).

For ten active seams, the new dominant counts are

\[
\begin{array}{c|c}
 \text{source class}&\text{f32 product equivalents}\\ \hline
 P=uu^T&10\\
 P=D_uRD_u&20\\
 \text{generic dense }P&30\\
 \text{biased degree-four, common-threshold three-channel}&50\\
 \text{biased degree-four, generic three-channel}&60.
\end{array}
 \tag{120.28}
\]

These counts are modulo synchronized collisions and before hooks.  Under
the literal strict-48.197 package, Cap49 leaves less than \(0.803\) of one
full equivalent.  Therefore even the one-product-per-seam circuit fails
the literal package cap unless its products are resident or skinny; the
fact that its *standalone* count is below 49 is irrelevant.  Under the
separate held-f32 adjusted-score economics, (117.24) permits about 41.2
incremental equivalents at utility parity: the 10-, 20-, and 30-product
banks fit before hooks, while the 50- and 60-product biased banks do not.
The optimistic raw-\(10^{-8}\) leader allowance (117.25) is 96.1, so all
rows of (120.28) fit that looser bill before hooks.

The physical-fidelity gate remains (117.28).  In particular, a
\(J+bR\) replacement for \(P^\sharp\) is admissible toward raw
\(10^{-8}\) only if its complete-old endpoint action error from (120.20),
together with reverse-row, collision, source-truncation, and co-adaptation
errors, is at most the available \(10.35\%\) relative budget.  Product
savings alone carry no raw-value credit.

Graph-ready split:

- K22_RANK1_LEAF_ONE_PRODUCT -- **solved exact modulo synchronized
  collisions**, (120.3)--(120.7);
- K22_RANK1_OUTPUT_COLLISION -- **one-product exact**, (120.8);
- K22_RANK1_SOURCE_COLLISIONS -- **exact; standalone strictness reopens
  B/lower-label work**, (120.9)--(120.13);
- K22_CORRELATION_PLANE_TWO_PRODUCT -- **solved exact**, (120.14)--
  (120.16);
- K22_PSHARP_WEAK_SPLIT -- **conditional approximation with exact tail
  gate**, (120.17)--(120.20);
- K22_BIASED_CHANNEL_SHARING -- **six generic / five common-threshold / two
  zero-bias-jet products per seam**, (120.21)--(120.28);
- K22_RANK1_CAP49 -- **closed without resident/skinny products on the
  strict-48.197 lineage; open under the held-f32 utility bill**;
- K22_RANK1_VALUE -- **open; must pass complete-old held-out gain and the
  physical fidelity gate**.

## 121. Leading trace-\(q_{211}\) consume-and-retire feature

This section isolates the cheapest exact component of the intrinsic
Gaussian surface defect.  It is a typed nonlinear readout of the resident
covariance, not an independent source of information.  The distinction is
essential: the construction can add a feature missing from the literal old
interface, but it cannot satisfy a theorem requiring a statistically
independent cubic or quartic source.

### 121.1 Exact pairing-minus-trace lineage

In the notation of Section 91, put

\[
 \kappa_2={4-\pi\over8\pi}>0.
\]

For three distinct labels, the exact intrinsic
\(Y-P_{\le2}Y\) Gaussian surface defect begins with

\[
 K_{\rm rem}(x,y,z)=-\kappa_2xy+T_{\ge3}(x,y,z),
 \qquad x=R_{ij},\quad y=R_{ik},\quad z=R_{jk}. \tag{121.1}
\]

Thus

\[
 (T_2)_{i; jk}
 =-\kappa_2\,\sigma_i^2\sigma_j\sigma_k
 R_{ij}R_{ik},\qquad i,j,k\ {\rm distinct}, \tag{121.2}
\]

is one exact declared source component.  Keeping \(T_2\) is a truncation
of the full defect, but it is not an approximate evaluation of this
component.

Let the raw Gaussian pairing and identity-trace lifts be

\[
\begin{aligned}
 ({\cal P}_R)_{iijk}&=R_{jk}+2R_{ij}R_{ik},\\
 ({\cal Z}_R)_{iijk}&=R_{jk}.
\end{aligned} \tag{121.3}
\]

Then

\[
 2R_{ij}R_{ik}=({\cal P}_R-{\cal Z}_R)_{iijk}, \tag{121.4}
\]

and consequently, at a synchronized cut where the complete covariance
pairing lift is old,

\[
 \boxed{
 [\,T_2\,]_{\rm old\ quotient}
 ={\kappa_2\over2}[\,{\cal Z}_R\,]_{\rm old\ quotient}.}
 \tag{121.5}
\]

Equation (121.5) is a lineage identity, not an assertion that every
implementation owns \({\cal P}_R\).  If the literal old interface does
not expose the complete pairing representative at the same cut, subtract
its actual Jacobian instead.  Conversely, if an included full Section 49
surface oracle or another old row already exposes the projected
\({\cal Z}_R\) action, (121.5) has zero additive novelty.

The source is entirely determined by \(R\).  Its only possible novelty is
that current coordinate CP4/Star/pair reads do not automatically implement
the identity-trace lift.  Complete-old source projection and final corpus
projection remain separate tests.

### 121.2 Exact one-product rank-one endpoint action

The following slightly more general identity makes the contraction cost
transparent.  Consider the all-pair completion

\[
 K_{i;jk}=q_iR_{ij}R_{ik}u_ju_k \tag{121.6}
\]

and a signed rank-one K22 endpoint
\(\Lambda=\epsilon\,tt^T\), \(\epsilon\in\{-1,+1\}\).
For the dense transition \(W\), define

\[
\begin{aligned}
 A_i&=\sum_a t_aW_{ai}^2,\\
 H_{ia}&=\sum_jR_{ij}u_jW_{aj}
       =(R D_uW^T)_{ia},\\
 v_i&=\sum_a t_aW_{ai}H_{ia}.
\end{aligned} \tag{121.7}
\]

Then the complete transported K22 action is exactly

\[
\boxed{
 {\cal F}(q,u,t)
 =\epsilon\left[
 4\sum_{i,a}q_iA_it_aH_{ia}^2
 +8\sum_iq_iv_i^2\right].}
 \tag{121.8}
\]

To verify (121.8), let \(B=W^TD_tW\), so
\(A=\operatorname{diag}B\), and put \(X=WD_u\).  The first Section 112
term is

\[
\begin{aligned}
 \langle D_uBD_u,\;R D_{q\odot A}R\rangle_F
 &=\operatorname{tr}
   (X^TD_tX\,R D_{q\odot A}R)\\
 &=\sum_{i,a}q_iA_it_aH_{ia}^2,
\end{aligned} \tag{121.9}
\]

while

\[
 ((B\odot R)u)_i
 =\sum_at_aW_{ai}H_{ia}=v_i. \tag{121.10}
\]

These are precisely the \(4\)- and \(8\)-weighted terms in (121.8).
Only

\[
 H=R D_uW^T \tag{121.11}
\]

is a dense matrix product; every other operation in (121.7)--(121.10)
is \(O(n^2)\).

For the standardized leading component, \(q=u=\mathbf1\).  If the
covariance endpoint is propagated by the constant zero-bias Price mode
\(s_\ell=\frac12W_{\ell+1}^Ts_{\ell+1}\), the local K22 factor is

\[
 t_\ell={\mu\over2}s_\ell,\qquad
 \mu={1\over\sqrt{2\pi}}. \tag{121.12}
\]

The physically normalized feature is therefore

\[
 f_{\ell,2}
 =-c_{\rm sym}\kappa_2\,{\cal F}
   (\mathbf1,\mathbf1,t_\ell), \tag{121.13}
\]

where \(c_{\rm sym}\) is the single raw/deployed tensor convention.
The factor \(\mu^2/4\) is already present quadratically through
\(t_\ell\); multiplying (121.13) by it again would double count the
boundary density.

If the output K22 row is strict in \(a\ne b\), subtract

\[
 {\cal F}_{a=b}
 =12\epsilon\sum_{a,i}q_it_a^2W_{ai}^2H_{ia}^2, \tag{121.14}
\]

which is \(O(n^2)\) after \(H\).  Source-side \(i=j\), \(i=k\), and
\(j=k\) exclusions are lower-label K31/K22/K4 representatives.  They
are zero-new-product only when that collision interface is literally
available at the same cut.  Without that receipt, the one-product theorem
applies to the completed tensor, not automatically to the strict source.

### 121.3 Product bill and the conditional \(WR\) reuse

With no resident buffer, (121.11) costs one dense product per active seam.
Across L5--L14,

\[
 C_{\rm dom}=10\quad\hbox{f32 GEMM-equivalents}, \tag{121.15}
\]

before hooks, collision reads, source formation, and package work.  At the
audited \(0.161\%\) relative charge per equivalent, co-firing with the
held \(u_0=0.5629\) pack screens at

\[
 u_{\rm screen}\simeq0.5629(1+10\cdot0.00161)
 \simeq0.5720. \tag{121.16}
\]

This leaves about \(31.2\) equivalents to the \(u=0.6002\) utility-parity
line.  At raw \(10^{-8}\), the optimistic adjusted score is about
\(5.72\times10^{-9}\), before omitted costs and before any value claim.
Under the separate strict48.197 Base48-to-49 package cap, ten fresh
products are closed; that branch reopens only through literal reuse,
skinny structure, or a different billed package.  The held-f32 economics
in (121.16) do not override that literal cap.

The current on-disk D11 tape does not prove reuse: its documented
covariance lineage and buffer lifetime are different.  The abstract
strict48.197 schedule, however, already pays for
\(S=WRW^T\).  A redesigned schedule may retain the first intermediate
\(J=WR\), in which case, for the special \(u=\mathbf1\) source only,

\[
 H=RW^T=J^T \tag{121.17}
\]

is a zero-new-GEMM view.  This requires the identical separated \(R\),
the identical \(W\) orientation and reset seam, and reductions before
\(J\) is recycled.  A nonconstant \(u\), merged total covariance, or a
different cut reopens (121.11).

### 121.4 Exact higher-source tail and value gates

Let
\[
 \rho=\max(|x|,|y|,|z|)
\]
on the declared strict source support, and let
\(\varepsilon_5(\rho)\) be the determinant-margin fifth-order remainder
from Section 91.  The explicit degree-three and degree-four coefficients
in (91.7) give the exact simplifications

\[
 2c_3+c_t={1\over\pi^2},\qquad
 c_{22}+2c_{31}+2c_{z2}={5\over12\pi}.
\]

Therefore

\[
\boxed{
 |K_{\rm rem}(x,y,z)+\kappa_2xy|
 \le E_2(\rho):=
 {\rho^3\over\pi^2}
 +{5\rho^4\over12\pi}
 +\varepsilon_5(\rho).}
 \tag{121.18}
\]

For a synchronized endpoint \(L\), one lawful absolute action certificate
is

\[
 |\langle L,T_{\ge3}\rangle|
 \le E_2(\rho)
 \sum_{i,j,k\ {\rm distinct}}
 |L_{i;jk}|\,\sigma_i^2\sigma_j\sigma_k. \tag{121.19}
\]

The corresponding Frobenius certificate follows by Cauchy--Schwarz.
There is no uniform relative bound when \(xy\) is small or the signed
endpoint cancels.  On the nonnegative local correlation cone all displayed
terms in (91.7) reinforce the leading sign, but a signed Bellman/corpus
action still needs (121.19), a cone receipt, or a directly measured
post-old error.  A coefficient-one physical claim must include all omitted
reverse, source, and finite-repair errors and obey the Section 117
\(10.35\%\) ideal-alignment ceiling.

For statistical credit define, at every layer,

\[
 T_{2,\perp,\ell}
 =(I-\Pi_{{\cal O}_{\rm src,\ell}})\,
   \Pi_{211}T_{2,\ell}. \tag{121.20}
\]

After evaluating the resulting layer bank \(F_2\), project once more:

\[
 F_{2,\perp}
 =(I-\Pi_{{\cal O}_{\rm corpus}})F_2. \tag{121.21}
\]

With

\[
 b={1\over N}F_{2,\perp}^Tr,\qquad
 G={1\over N}F_{2,\perp}^TF_{2,\perp},
\]

the exact frozen-refit gain is \(b^TG^\dagger b\).  Reaching raw
\(10^{-8}\) from \(2.9409\times10^{-8}\) requires

\[
 \boxed{
 b^TG^\dagger b\ge1.9409\times10^{-8},
 \qquad \rho_{\rm old\ quotient}\ge0.8124.}
 \tag{121.22}
\]

Source nonzero, typed-readout absence, and held-out value are three
different gates.

Graph-ready split:

- Q211_LEADING_TRACE_PAIRING_IDENTITY -- **solved exact**,
  (121.3)--(121.5);
- Q211_LEADING_TRACE_SOURCE_INFORMATION -- **zero new information;
  covariance-determined typed readout only**;
- Q211_LEADING_TRACE_OLD_SOURCE_QUOTIENT -- **conditional** on the
  synchronized complete pairing/trace Jacobian, (121.5), (121.20);
- Q211_LEADING_TRACE_ONE_PRODUCT_ACTION -- **solved exact**,
  (121.6)--(121.11);
- Q211_LEADING_TRACE_COLLISION_SEWING -- **output solved \(O(n^2)\);
  source lower-label ownership required**, (121.14);
- Q211_LEADING_TRACE_WR_REUSE -- **conditional redesign, not current
  on-disk ownership**, (121.17);
- Q211_LEADING_TRACE_TEN_SEAM_COST -- **10-equivalent default before
  omitted work; closed under the strict Base48-to-49 cap**, (121.15)--
  (121.16);
- Q211_LEADING_TRACE_HIGHER_SOURCE_TAIL -- **solved absolute gate**,
  (121.18)--(121.19);
- Q211_LEADING_TRACE_FULL_PHYSICS -- **HOLD** until all tails are below
  the post-old value margin;
- Q211_LEADING_TRACE_FULL_SURFACE_DUPLICATE -- **NO-GO** when the
  billed comparator already includes the exact q211 surface;
- Q211_LEADING_TRACE_CORPUS_SCHUR -- **open**, (121.20)--(121.22);
- Q211_LEADING_TRACE_RAW1E8 -- **open; requires held-out old-quotient
  correlation \(0.8124\)**.

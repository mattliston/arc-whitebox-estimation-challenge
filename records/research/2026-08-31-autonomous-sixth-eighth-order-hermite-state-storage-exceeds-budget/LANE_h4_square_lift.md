# H4 square lift

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## 2026-08-30 — square-lift normalization and reset theorem

### Gaussian-chaos conventions

Let `z ~ N(0,I_r)`, let `M,N` be real symmetric `r x r` matrices, and put

\[
Q_M(z)=\frac12\{z^T Mz-\operatorname{tr}M\},\qquad
s_M^2=\mathbb E Q_M^2=\frac12\operatorname{tr}(M^2).
\]

The polarized pure fourth-chaos square lift is

\[
U_{M,N}
=Q_MQ_N-\frac12\operatorname{tr}(MN)-Q_{MN+NM}
=\frac14 I_4(M\widetilde\otimes N),
\]

and in particular

\[
U_M=U_{M,M}=Q_M^2-\frac12\operatorname{tr}(M^2)-2Q_{M^2}.
\]

This fixes the normalization: `U` is exactly the fourth-chaos projection of the
quadratic-core product, not the whole centered square.  Polarization gives
`U_{M,N}=(U_{M+N}-U_{M-N})/4`.

The exact cross covariance is

\[
\boxed{
\mathbb E[U_MU_N]
=\frac12\operatorname{tr}(MN)^2
+\frac23\operatorname{tr}(M^2N^2)
+\frac13\operatorname{tr}(MNMN)
}
\]

or, equivalently,

\[
\mathbb E[U_MU_N]
=\frac12\operatorname{tr}(MN)^2
+\operatorname{tr}(M^2N^2)
-\frac16\|[M,N]\|_F^2.
\]

Thus the square Gram is positive semidefinite, but noncommutativity subtracts a
precisely normalized commutator energy.  This is the clean bridge from the
square lift to the even-ring / `k4` channel.

For one centered square, the fraction of nonconstant square energy already in
the second chaos is

\[
F_{H2\mid Q^2-\mathbb EQ^2}
=\frac{4\tau}{1+6\tau},\qquad
\tau=\frac{\operatorname{tr}M^4}{(\operatorname{tr}M^2)^2}.
\]

It vanishes for a diffuse core and is large only for spectrally concentrated
cores; the missing diffuse energy is therefore genuinely the pure `H4` square,
not a retuning of the old `H2` shadow.

### Exact ReLU projection at arbitrary bias

For `X ~ N(0,1)`, `a=b/s`, and `g(X)=sigma (X+a)_+`, the orthonormal Hermite
coefficients are

\[
b_0=\sigma\{\phi(a)+a\Phi(a)\},\qquad
b_1=\sigma\Phi(a),
\]

\[
\boxed{
b_m=\frac{\sigma\phi(a)}{\sqrt{m!}}\,He_{m-2}(-a),\quad m\ge2.
}
\]

The exact centered output energy is

\[
V(a)=\sigma^2\left[
(1+a^2)\Phi(a)+a\phi(a)
-\{\phi(a)+a\Phi(a)\}^2
\right].
\]

In the diffuse quadratic-core limit, scalar Hermite order one is the carried
underlying `H2` core, while scalar order two is exactly the new underlying
`H2+H4` square sector (the additional `H2` share from `M^2` vanishes with the
spectral concentration).  Therefore the exact general-bias retained fractions
are

\[
\boxed{
F_{H2}(a)
=\frac{\sigma^2\Phi(a)^2}{V(a)},
\qquad
F_{H2+H4}(a)
=\frac{\sigma^2\{\Phi(a)^2+\phi(a)^2/2\}}{V(a)}.
}
\]

The scale `sigma` cancels.  For a sequence of destructive diffuse resets with
biases `a_1,...,a_L`, the best correlation available from retaining only this
sector is bounded by

\[
\boxed{
q_{1:L}\le
\prod_{\ell=1}^L\sqrt{F_{H2+H4}(a_\ell)},
\qquad
\sum_{\ell=1}^L\log F_{H2+H4}(a_\ell)
\ge2\log q_\star
}
\]

as the exact necessary multiplicative-energy gate for a desired correlation
`q_star`.  This reduces to the zero-bias reset count below when every `a_l=0`.
Positive biases approach an affine gate and `F->1`; strongly negative
tail-selecting biases have no uniform square-fidelity guarantee.  Thus the
zero-bias `0.98334` cannot be imported layerwise without a certified bias
envelope.

In the diffuse quadratic-core limit this gives the reset law

\[
M^+=\frac{\sigma\Phi(a)}sM+
\frac{\sigma\phi(a)}{s^2}M^2,
\qquad
U_M^+=\delta U_M,
\qquad
\delta=\frac{\sigma\phi(a)}{2s^2},
\]

with the understanding that the `M^2` term is projected/centered in the chosen
quadratic state.  The fourth-chaos coefficient is not free: it is tied to the
same quadratic core that supplies the second-chaos reset.

For persistent H4 transport, fix the normalization explicitly.  The physical
preactivation is `b+Q_M+H`, with

\[
s^2=\mathbb E Q_M^2=\tfrac12\operatorname{tr}M^2,
\qquad X=Q_M/s,\qquad a=b/s,
\]

and write the output as `sigma (X+a+H/s)_+`.  Thus `sigma=s` for an ordinary
unscaled ReLU in physical units.  The Jacobian with respect to physical `H` is
`(sigma/s) 1_{X>-a}H`; with respect to standardized `Htilde=H/s` it is
`sigma 1_{X>-a}Htilde`.  The indicator's orthonormal scalar-Hermite
coefficients are

\[
d_0=\Phi(a),\qquad
d_m=\frac{\phi(a)}{\sqrt{m!}}He_{m-1}(-a),\quad m\ge1.
\]

The first correlated gate term has an exact noncommutative product law.  The
Wiener product formula gives

\[
\boxed{
\Pi_4\{U_{R,S}Q_M\}
=2\left(U_{M\circ R,S}+U_{R,M\circ S}\right),
\qquad M\circ R=\frac12(MR+RM).
}
\]

Indeed `I4(f) I2(M)` contributes `8 I4(f contraction_1 M)` at order four,
and symmetrizing the contracted leg produces the two Jordan multiplications.
Consequently the constant-plus-linear part of the correlated ReLU Jacobian on
a standardized H4 perturbation is

\[
\boxed{
\mathcal J_{a,M}^{(0:1)}U_{R,S}
=\sigma\Phi(a)U_{R,S}
+\frac{2\sigma\phi(a)}s
\left(U_{M\circ R,S}+U_{R,M\circ S}\right).
}
\]

For a square generator this is

\[
\mathcal J_{a,M}^{(0:1)}U_R
=\sigma\Phi(a)U_R
+\frac{4\sigma\phi(a)}sU_{M\circ R,R}.
\]

For a physical H4 perturbation, divide both displayed Jacobian formulas by
`s`: the coefficients are `sigma Phi(a)/s` and
`2 sigma phi(a)/s^2` (or `4 sigma phi(a)/s^2` for a square).

Equivalently, let

\[
\mu_4:\operatorname{Sym}^2(\mathcal J)\to K_4(\mathcal J),
\qquad \mu_4(R\odot S)=U_{R,S},
\]

and let `L_M R=M circ R`.  On standardized perturbations the exact
constant-plus-underlying-H2 intertwiner is

\[
\boxed{
\mathcal J_{a,M}^{H0+H2}\mu_4
=\mu_4\left[
\sigma\Phi(a)I
+\frac{2\sigma\phi(a)}s
(L_M\odot I+I\odot L_M)
\right].
}
\]

If `M in J` and `J` is Jordan closed, `L_M(J) subset J`; hence this operator
preserves `Sym^2(J)` and descends through any syzygies in `ker(mu4)`.  This is
the exact H0+H2 module-invariance theorem.  It says nothing yet about the
higher gate sectors or chaos-order excursions.

For a finite non-Gaussian quadratic core the same intertwiner is exact after
replacing

\[
\Phi(a)\to\theta_0=\mathbb E1_{Q_M>-b},\qquad
\phi(a)\to\theta_1=\mathbb E[X1_{Q_M>-b}].
\]

Only the two scalar coefficients change; the noncommutative Jordan action and
all normalization factors are unchanged.

This formula makes the rank-growth mechanism precise.  If all cores lie in a
common Jordan algebra `A`, the first correlated correction preserves
`span{U_AB:A,B in A}`.  Without Jordan closure, every pair `(M_i,R_s)` can
create a new independent mixed generator.

This is not by itself an exact truncation of the gate.  The omitted scalar
Hermite tail has the exact energy

\[
\boxed{
\sum_{m\ge2}d_m^2
=\Phi(a)\{1-\Phi(a)\}-\phi(a)^2.
}
\]

At zero bias it equals `1/4-1/(2pi)=0.090845...`, or about `36.3%` of the
indicator variance.  Therefore a constant-plus-linear gate approximation
cannot be promoted from Hermite energy alone.  One must either prove that the
higher gate sectors act inside the same Jordan/cyclic module, or bound their
target-conditioned contraction.  This is the exact boundary between the
constructive Jordan lane and a correlated-gate truncation error.

### Adjoint-weighted direct gate-tail leakage

Let

\[
R_a(X)=1_{X>-a}-\Phi(a)-\phi(a)X,
\qquad
\eta_a^2=\|R_a\|_2^2
=\Phi(a)\{1-\Phi(a)\}-\phi(a)^2.
\]

These `Phi,phi` formulas are exact for the diffuse standard-normal core.  For
a finite quadratic core, replace `Phi(a),phi(a)X` by the exact orthogonal
projection of the indicator onto `span{1,X}` under the generalized-chi-square
law; every operator identity below then holds with that distribution-specific
residual.

Let `K` be the retained H4 square/Jordan module and `D` the omitted or
Bellman-dark H4 subspace.  The exact one-reset leakage operator for a
standardized perturbation is

\[
\boxed{
\mathcal L_{a,M}^{D\leftarrow K}
=P_D\Pi_4\,M_{R_a(Q_M/s)}P_K,
}
\]

where `M_f` denotes multiplication by `f`.  If `J` is the observation/retention
map in the reverse notation, take `D=ker J`; if `J` is a terminal Bellman row,
the scalar leakage is obtained by pairing with its pulled-back adjoint.  The
binding candidate-specific constant is

\[
\boxed{
\lambda_{a,M}(D,K)
=\|\mathcal L_{a,M}^{D\leftarrow K}\|_{op}.
}
\]

In the forward-module notation requested for deployment, let `P_mod` project
onto `K4(J)`.  The **spatial module leakage** is specifically

\[
\boxed{
\mathcal L_{\rm mod}
=(I-P_{\rm mod})\Pi_4M_{R_a(Q_M/s)}P_{\rm mod}.
}
\]

This is the object that vanishes under exact Jordan-module invariance.

Endpoint novelty is a different projection.  Let `Phi_{L<-l}` be the complete
future suffix from the reset to the endpoint, let `O` be the complete-old
terminal observation, and put `N=ker O`.  The missed novel endpoint operator is

\[
\boxed{
\mathcal L_{\rm end,l}
=P_N\Phi_{L\leftarrow l}\Pi_4M_{R_{a_l}}P_{\rm mod},
}
\]

with adjoint form

\[
\boxed{
\mathcal L_{\rm end,l}^*
=P_{\rm mod}M_{R_{a_l}}\Pi_4\Phi_{L\leftarrow l}^*P_N.
}
\]

Thus `L_mod=0` proves no spatial escape at that reset, while the Bellman
novelty gate is the norm/action of `L_end`; neither statement implies the
other once future chaos-order excursions are allowed.

For a physical perturbation, multiply this operator by `sigma/s`; for a
standardized perturbation, by `sigma`.

For normalized `U in K` and `Lambda in D`, self-adjointness of multiplication
and orthogonal chaos projection gives the exact adjoint action

\[
\boxed{
\langle\Lambda,\mathcal L U\rangle
=\mathbb E\left[\Lambda\,R_a(Q_M/s)\,U\right].
}
\]

Thus a complete late-layer tail receipt is

\[
\left|\rho_{\rm tail}^{\rm direct}\right|
\le\sum_l\frac{\sigma_l}{s_l}
\lambda_{a_l,M_l}(D_l,K_l)\,\|U_l\|_2\,\|\Lambda_l\|_2.
\]

The scalar tail energy alone supplies only a coarse universal bound.  Gaussian
hypercontractivity on fourth chaos, `||U||_4<=9||U||_2`, and Holder give

\[
\boxed{
\lambda_{a,M}(D,K)\le81\eta_a.
}
\]

At zero bias this upper bound is about `24.4`, so the fact that the indicator
tail has only `0.090845` energy does **not** imply a small H4 action.  Promotion
requires the candidate-specific contraction or a structural zero.

The direct `H4 -> H4` tail is nevertheless finite-chaos.  Write the even Wiener
expansion of the (finite-core) gate residual as

\[
R=\sum_{m\ge0}I_{2m}(r_{2m}),\qquad U=I_4(f).
\]

In the product formula, an output of order four requires contraction order
`k=m`; since the H4 input has only four legs, necessarily `m<=4`.  Therefore

\[
\boxed{
\begin{aligned}
\Pi_4(RU)=I_4\big[&r_0f
+8\,\operatorname{Sym}(r_2\otimes_1f)
+72\,\operatorname{Sym}(r_4\otimes_2f)\\
&+480\,\operatorname{Sym}(r_6\otimes_3f)
+1680\,\operatorname{Sym}(r_8\otimes_4f)
\big].
\end{aligned}
}
\]

No gate chaos above H8 acts directly on H4 in one reset.  This gives a sharper
route than the `81 eta` norm envelope: certify exactly the four nonconstant
Wiener kernels `r2,r4,r6,r8` (with `r0=0` for a residual) and their restricted
contractions.  It does **not** truncate the temporal problem.  The same product
also emits H6/H8, and a later gate can return those sectors to H4; those are
the folded/Volterra terms recorded below.  Thus the direct-tail node is a
finite five-block operator, whereas the future cyclic-module node remains a
separate closure obligation.

There is also a closed-form nonzero witness.  In the diffuse scalar surrogate,
the normalized square mode is `u=h_2(X)=(X^2-1)/sqrt(2)`.  Then

\[
\boxed{
\langle u,\Pi_2\{R_a(X)u\}\rangle
=\mathbb E[R_a(X)h_2(X)^2]
=-\frac12a(1+a^2)\phi(a).
}
\]

Hence the tail multiplication operator has norm at least
`|a|(1+a^2)phi(a)/2` on the square sector whenever `a!=0`.  This witness is
inside the square module; it proves that deleting the tail changes even the
retained H4 coefficient.  Spatial leakage into `D` is zero only if the
candidate-specific projected contraction vanishes.

At zero bias there is an exact finite-core witness.  Take a common rank-one
core, so

\[
X=\frac{z^2-1}{\sqrt2},\qquad
u=\frac{He_4(z)}{\sqrt{24}},
\]

and the gate is `1_{|z|>1}`.  Its exact projection onto `span{1,X}` is

\[
d_0=2\bar\Phi(1),\qquad d_1=\sqrt2\phi(1).
\]

With `R_rank1=1_{|z|>1}-d0-d1 X`, the normalized H4 action is

\[
\boxed{
\mathbb E[R_{\rm rank1}u^2]
=-\frac{17}{3}\phi(1)\ne0.
}
\]

To verify it, use
`He4^2=z^8-12z^6+42z^4-36z^2+9`, the Gaussian tail recurrence at one, and
`E[Xu^2]=4sqrt(2)`.  This lower witness is large and exact.  A common rank-one
algebra makes `(I-P_mod)` leakage zero because its H4 module is one-dimensional,
but it emphatically does not make the within-module tail coefficient or later
chaos excursions vanish.

The first explicit noncommutative tail word appears already in scalar Hermite
order two.  Since

\[
h_2(Q_M/s)=\frac{U_M+2Q_{M^2}}{\sqrt2s^2},
\qquad d_2=-\frac{a\phi(a)}{\sqrt2},
\]

its H2-mediated contribution is

\[
\boxed{
d_2\Pi_4\{U_{R,S}h_2(Q_M/s)\}_{H2}
=-\frac{2a\phi(a)}{s^2}
\left(U_{M^2\circ R,S}+U_{R,M^2\circ S}\right).
}
\]

Any adjoint in `D` pairing nontrivially with this displayed word, after
subtracting the H4-mediated part and higher orders, is an algebraic lower-bound
witness for leakage.  Generically such a pairing is nonzero; a quantitative
lower bound is the absolute displayed pairing minus certified bounds for the
remaining tail terms.

The direct spatial leakage vanishes under any of the following exact
conditions:

1. `K=K4(A)=span{U_AB:A,B in A}` for a common special Jordan algebra `A`
   containing every gate and carried core, **and** `A` is certified closed
   under every two-open-leg Wick word generated by the gate;
2. all cores are commuting and `A` is enlarged to their full simultaneously
   diagonal polynomial algebra (mere pairwise commutation without this closure
   is insufficient);
3. all cores lie in one common rank-one Jordan algebra, or in a direct sum of
   mutually orthogonal commuting rank-one algebras, with the full corresponding
   square module retained;
4. the pulled-back Bellman adjoint annihilates the tail image, equivalently
   `P_D Pi4 M_R P_K=0` on the declared task tangent.

These are spatial-module statements.  They do not by themselves justify
discarding other chaos orders born by the same multiplication.

The Wick-word clause in item 1 is substantive.  Binary Jordan closure proves
the H0+H2 intertwiner above, but an arbitrary special Jordan subalgebra need not
contain every reversal-symmetrized word in its associative envelope.  Define

\[
\operatorname{Rev}(\mathcal A)
=\operatorname{span}\{A_1\cdots A_m+A_m\cdots A_1:m\ge1\}.
\]

A simple sufficient full-tail condition is

\[
\boxed{\operatorname{Rev}(\mathcal A)\subseteq\mathcal A.}
\]

Commuting matrix algebras and the common/orthogonal rank-one cases satisfy it.
For a non-reversible Jordan algebra, the correct retained core is its needed
Wick/reversible envelope, whose dimension must be rebilled.  Multi-layer H6/H8
excursions are especially capable of generating symmetrized words involving
different gate cores, even when each one-step H0+H2 map was Jordan closed.

There is an exact blockwise way to certify the one-gate operator.  Diagonalize
`M` and use the orthonormal multi-index Hermite basis

\[
H_\alpha(z)=\prod_i\frac{He_{\alpha_i}(z_i)}{\sqrt{\alpha_i!}},
\qquad |\alpha|=4.
\]

Since `R_a(Q_M/s)` is even in every eigen-coordinate,

\[
\boxed{
\langle H_\alpha,M_{R_a}H_\beta\rangle=0
\quad\text{unless}\quad
\alpha_i\equiv\beta_i\pmod2\ \text{for every }i.
}
\]

Thus the exact H4 tail matrix is block diagonal by the set of odd coordinates.
The block dimensions are

\[
\boxed{
d_{0\text{-odd}}=\binom{r+1}{2},\qquad
d_{\{i,j\}\text{-odd}}=r\ \text{for each }\{i,j\},\qquad
d_{\{i,j,k,l\}\text{-odd}}=1.
}
\]

They sum to `binom(r+3,4)`.  The zero-odd block contains the `(4)` and `(22)`
Hermite sectors; a two-odd block contains the corresponding `(31)` and `(211)`
sectors; a four-odd block is `(1111)`.  Every block entry is the exact truncated
generalized-chi-square moment

\[
\mathbb E[R_a(Q_M/s)H_\alpha H_\beta],
\]

so `lambda_a,M` can be certified by finite block singular values rather than a
full fourth-order tensor.

If all gate and carried cores commute in one eigenbasis, every square lift lies
in the zero-odd block and every gate preserves that block.  This proves the
commuting spatial-closure claim directly; its module has at most
`binom(r+1,2)` generators.  Dense changes of eigenbasis destroy the shared
parity labels, which is why per-layer diagonalization alone is not a free
multi-layer closure proof.

The commuting branch has an immediate inverse-moment gate:

\[
\boxed{\text{simultaneous commutation implies }\operatorname{rank}G\le r.}
\]

It has a sharper fourth-moment gate.  With
`A_ij=tr(M_i^2M_j^2)` and the present normalization
`K22_ij=2A_ij+tr(M_iM_jM_iM_j)`,

\[
\boxed{
\|[M_i,M_j]\|_F^2
=2\{3A_{ij}-K22_{ij}\}.
}
\]

Hence simultaneous commutation forces `A=K22/3`, and therefore

\[
\boxed{
\begin{pmatrix}G&K21^T\\K21&K22/3\end{pmatrix}\succeq0,
\qquad
\operatorname{rank}
\begin{pmatrix}G&K21^T\\K21&K22/3\end{pmatrix}\le r.
}
\]

In a diagonal realization `A=K22/3` is an ordinary completely-positive Gram
of the nonnegative eigenvalue-square vectors, with cp-rank at most `r`.  These
are cheap necessary falsifiers.  Exact sufficiency is the signed diagonal
moment factorization: find vectors `m_i in R^r` such that

\[
G_{ij}=\langle m_i,m_j\rangle,
\quad K21_{ij}=\langle m_i^{\odot2},m_j\rangle,
\quad K22_{ij}/3=\langle m_i^{\odot2},m_j^{\odot2}\rangle.
\]

When `rank G=r`, this factorization has a particularly sharp inverse form.
Assume the already-required support identity `K21=K21 G^+G`.
Choose any full-column Gram factor `X in R^{n by r}` with `G=XX^T` and define

\[
\boxed{
S_X=K21\,X(X^TX)^{-1}.
}
\]

A full-rank simultaneous-diagonal realization of `(G,K21)` exists exactly when
there is an orthogonal `O in O(r)` such that

\[
\boxed{S_XO=(XO)^{\odot2}.}
\]

It realizes the observed fourth plane exactly when the no-longer-free identity
holds:

\[
\boxed{
K22/3=S_XS_X^T=K21\,G^+K21^T.
}
\]

Proof: if the actual eigenvalue-row matrix is `D=XO`, then
`K21=D^{odot2}D^T`, so multiplication on the right by
`X(X^TX)^{-1}` gives `S_X=D^{odot2}O^T`; the displayed equations follow, and
the converse is immediate.

The remaining gauge is exactly the nonlinear stabilizer

\[
\boxed{
\mathcal G_D
=\{P\in O(r):(d_iP)^{\odot2}=(d_i^{\odot2})P\ \text{for every observed }i\}.
}
\]

If `G_D` contains only coordinate permutations, then every all-distinct
diagonal trace

\[
\operatorname{tr}(M_iM_jM_kM_l)
=\sum_aD_{ia}D_{ja}D_{ka}D_{la}
\]

is uniquely determined despite the remaining gauge.  A simple sufficient
certificate for permutation-only stabilizer is that the evaluation set is
quadratically determining,

\[
\operatorname{span}\{d_id_i^T\}_i=\operatorname{Sym}_r.
\]

Then each quadratic identity defining `P in G_D` holds for every `x`, so
polarization makes `P` an orthogonal automorphism of coordinatewise
multiplication, hence a permutation.  This sufficient condition needs at least
`r(r+1)/2` independent rows and cannot hold in one `n=1024,r=55` frame.
Therefore permutation-only gauge is **not** automatic here: it needs a direct
stabilizer/Jacobian proof or additional lawful trajectory rows.  Conditional
on that proof, `G,K21` reconstruct all H4 trace words and `K22` is an exact
held-out certificate; without it, the residual stabilizer is another possible
dark fiber.

The exact local stabilizer test is inexpensive algebraically.  At an actual
diagonal solution, write `P=I+epsilon Omega+o(epsilon)` with `Omega^T=-Omega`.
Then

\[
\boxed{
D F_i[\Omega]
=2d_i\odot(d_i\Omega)-(d_i^{\odot2})\Omega.
}
\]

The commuting core is locally unique up to its discrete permutation gauge iff

\[
\boxed{
DF_i[\Omega]=0\ \forall i,\quad \Omega^T=-\Omega
\Longrightarrow\Omega=0.
}
\]

This can hold with fewer than `r(r+1)/2` rows; it proves local isolation but
not absence of remote orthogonal solutions.  A global quartic-action claim
must either enumerate/separate those solutions or prove the full stabilizer is
permutational.

The two held-out fourth planes play different roles here.  `K22` cannot
separate solutions of the commuting `(G,K21)` equation because it is already
forced to `3S_XS_X^T`.  By contrast,

\[
\boxed{
K31_{ij}(O)=3\left\langle (XO)_i^{\odot3},(XO)_j\right\rangle
}
\]

generally varies across nonpermutation elements of `G_D`.  Its infinitesimal
variation at an actual diagonal solution is

\[
\boxed{
\frac13DK31_{ij}[\Omega]
=\left\langle3d_i^{\odot2}\odot(d_i\Omega),d_j\right\rangle
+\left\langle d_i^{\odot3},d_j\Omega\right\rangle.
}
\]

Therefore held-out `K31` is the first lawful same-frame discriminator of the
remaining commuting stabilizer; held-out `K22` is only a feasibility/checksum
receipt.  Promotion requires the joint kernel of the stabilizer Jacobian and
`DK31` to annihilate the desired all-distinct H4 action.  If remote candidates
still match `K31`, one mixed quartic trajectory frame remains necessary.

If the observed core Gram has rank greater than `r`, the common-eigenbasis
escape is impossible before any carrier construction.  If it passes, then at
`r=55` the complete commuting H4 module has only
`binom(56,2)=1540` generators.  Its dense width transport costs
`1540/1024=1.504` GEMM equivalents per active layer; ten late layers plus the
quoted 9-equivalent core total about `24.04` before births/hooks, inside the
49 cap.  Thus commutation is a genuinely constructive low-cost branch, but
only after the severe `rank G<=55` and shared-eigenbasis trajectory gates.

The corresponding forward state is explicit.  Identify the parity-zero H4
module with `Sym_r` through

\[
B_i\longmapsto
\mu_4(B_i)
=\sum_a(B_i)_{aa}U_{E_{aa}}
+2\sum_{a<b}(B_i)_{ab}U_{E_{aa},E_{bb}}.
\]

Dense width transport is simply

\[
B_i\leftarrow\sum_jW_{ji}B_j.
\]

For the H0+H2 gate, if `D_i=diag(d_i)`, the intertwiner becomes the Lyapunov
update

\[
\boxed{
B_i^+
=\frac{\sigma_i\Phi(a_i)}{s_i}B_i
+\frac{2\sigma_i\phi(a_i)}{s_i^2}
(D_iB_i+B_iD_i)
+\frac{\sigma_i\phi(a_i)}{2s_i^2}d_id_i^T,
}
\]

for a physical H4 coefficient state, where the last term is the new square
birth under the same coefficient convention.  A deliberately standardized
state removes one factor of `s_i` from the first two terms, exactly as in the
intertwiner above.  The Lyapunov and rank-one birth work is only `O(nr^2)`;
width transport dominates.

To retain the entire direct gate tail rather than the invalid H0+H2
truncation, replace the Lyapunov map by the exact parity-zero block
`T_i^(0-odd)` with entries
`E[gate_i H_alpha H_beta]`.  A dense application to all units costs at most

\[
\frac{nR^2}{n^3}=\left(\frac{1540}{1024}\right)^2=2.262
\]

GEMM equivalents per layer.  Ten layers then cost about `15.04` for width
transport plus `22.62` for local direct-tail maps; adding the 9-equivalent core
gives `46.66`, leaving only about `2.34` for kernel construction, births,
hooks, and provenance.  This is a conditional cost pass, not a packaged
receipt: the per-unit truncated-moment block must have a lawful analytic
implementation inside that margin, and the H6/H8 return node below must still
pass.

The exact analytic implementation is a Fourier--Mehler operator.  In the gate
eigenbasis,

\[
1_{Q_M+b>0}
=\frac12+\frac1\pi\int_0^\infty
\frac{\sin\{t(Q_M+b)\}}t\,dt,
\]

so on H4

\[
\boxed{
T_{a,M}^{44}
=\frac12I+\frac1\pi\int_0^\infty
\frac{\operatorname{Im}\{e^{itb}K_M(t)\}}t\,dt,
\qquad
K_M(t)=\Pi_4e^{itQ_M}\Pi_4.
}
\]

The individual Wiener kernels inside this operator also have an exact
resolvent form.  Put

\[
\chi_M(t)=e^{-it\operatorname{tr}M/2}\det(I-itM)^{-1/2},
\qquad
A_M(t)=itM(I-itM)^{-1}.
\]

The Wick-exponential identity

\[
\mathbb E\!left[e^{itQ_M(z)}:e^{h^Tz}:\right]
=\chi_M(t)\exp\!\left\{\frac12h^TA_M(t)h\right\}
\]

shows that the `2m`-th Wiener kernel of `e^{itQ_M}` is

\[
\frac{\chi_M(t)}{2^m m!}
\operatorname{Sym}\{A_M(t)^{\otimes m}\}.
\]

Hence for `m>=1` the exact indicator kernel is

\[
\boxed{
r_{2m}
=\frac1\pi\int_0^\infty
\frac{\operatorname{Im}\!left[
e^{itb}\chi_M(t)
\operatorname{Sym}\{A_M(t)^{\otimes m}\}
\right]}{2^m m!\,t}\,dt .
}
\]

Together with the finite direct-action theorem, only `m=1,2,3,4` are needed
for exact `H4 -> H4`.  This is valid without commutativity among the carried
cores: each frequency uses a rational function of the single current gate
matrix `M`.  Commutativity becomes relevant only when asking whether its
contractions preserve a small common module and whether those per-unit
resolvents can share a cheap basis.

Every matrix element of `K_M(t)` factorizes over latent coordinates.  If
`M=diag(lambda_c)`, then

\[
\langle H_\alpha,K_M(t)H_\beta\rangle
=e^{-it\operatorname{tr}M/2}
\prod_c\mathcal M_{\alpha_c\beta_c}(it\lambda_c),
\]

where the one-dimensional factors are generated exactly by

\[
\mathcal M_{mn}(u)
=\frac{\mathbb E[He_m(z)He_n(z)e^{uz^2/2}]}{\sqrt{m!n!}},
\]

and

\[
\boxed{
\sum_{m,n\ge0}
\frac{\mathbb E[He_m(z)He_n(z)e^{uz^2/2}]}
{m!n!}x^my^n
=(1-u)^{-1/2}
\exp\left[
\frac{u(x^2+y^2)}{2(1-u)}+\frac{xy}{1-u}
\right].
}
\]

Only `m,n<=4` are needed for the H4 parity-zero block.  This is an exact
closed-form kernel; the only approximation is truncating the one-dimensional
Fourier quadrature.

The oscillatory tail has a deterministic spectral bound.  For `u=it lambda`,

\[
|(1-u)^{-1/2}|=(1+t^2\lambda^2)^{-1/4},
\qquad
\left|\frac{u}{1-u}\right|\le1,
\qquad
\left|\frac1{1-u}\right|\le1.
\]

Because only even derivatives through order four occur in the parity-zero
block, the displayed generating polynomial gives the explicit universal
constant

\[
\boxed{C_4=35/8.}
\]

Indeed, writing `A=u/{2(1-u)}` and `B=1/(1-u)`, the largest one-coordinate
normalized coefficient is
`M_44/M_00=6A^4+12A^2B^2+B^4`, whose modulus is at most
`6/16+12/4+1=35/8`; all other degree partitions give a smaller product.  Thus
every normalized parity-zero matrix element obeys

\[
\boxed{
|\langle H_\alpha,K_M(t)H_\beta\rangle|
\le C_4\prod_{c:\lambda_c\ne0}
(1+t^2\lambda_c^2)^{-1/4}.
}
\]

If at least `rho` eigenvalues satisfy `|lambda_c|>=lambda_min` and
`T lambda_min>=1`, the parity-block operator tail has the safe bound

\[
\boxed{
\left\|\frac1\pi\int_T^\infty
\frac{\operatorname{Im}(e^{itb}K_M(t))}{t}\,dt\right\|_{op}
\le
\frac{2R C_4}{\pi\rho}
(T\lambda_{\min})^{-\rho/2},
\qquad R=\binom{r+1}{2}.
}
\]

Thus a full-rank `r=55` gate has algebraic tail exponent `27.5`; the hard cases
are spectrally low-rank or have very small nonzero eigenvalues.  This tail bound
does not control discretization on `[0,T]`, but it turns quadrature depth into
a finite-interval approximation problem with a certified remainder.

If one frequency action is organized in `O(nr^3)` work across all units, its
ten-layer cost at `r=55,n=1024` is

\[
10\frac{r^3}{n^2}=1.587
\]

GEMM equivalents.  After the 15.04 transports and 9-equivalent core, the
49-cap allows at most about `J=15` such frequencies before hooks.  An
`O(nr^2)` factorized action would cost only `0.02885 J` across ten layers, so
the arithmetic ceiling is about `J=865`; the actual factorization and memory
traffic decide which count is lawful.

For a preregistered quadrature `Q_J`, the binding error is not a pointwise
kernel norm but the endpoint action remainder

\[
\boxed{
\eta_{\rm Mehler,J}
=\left\|
P_N\Phi_{L\leftarrow l}
\left(T_{a,M}^{44}-Q_J[T_{a,M}^{44}]\right)P_{\rm mod}
\right\|.
}
\]

Equivalently, use the adjoint norm of
`P_mod (T-Q_J)^* Phi_{L<-l}^* P_N`.

It must satisfy the same
`min{3.80e-6,eta_star(g)}` gate derived below.  Thus a commuting direct-tail
carrier is constructive precisely if a `J<=15` cubic-cost quadrature (or a
lawful cheaper factorization) has a certified endpoint-weighted tail bound.

**Graph-ready child `P2R0G19B_H4_COMMUTING_MEHLER_KERNEL`:** promote on the
exact generating formula, a preregistered quadrature within the relevant cost
ceiling, and an endpoint remainder below the action/Schur radius.  Close if the
certified `J` exceeds the package allowance or the oscillatory-tail bound is
noncoercive.  Interpret the Heaviside integral by Abel damping/principal-value
limit when proving the quadrature remainder.

**Graph-ready node `P2R0G19B_H4_COMMUTING_PARITY_CARRIER`:** promote only if
the diagonal moment/stabilizer gates, shared-basis trajectory, exact parity-zero
gate kernel, total work `<=49`, and chaos-return Bellman gate all pass.  Close
on `rank G>r`, failed diagonal factorization, a nonpermutation target-distinct
stabilizer, kernel/hook cost above the `2.34` margin, or visible H6/H8 memory.

### H6/H8 excursion and return to H4

Write the gate in underlying Gaussian chaoses and let
`T_l^{q<-p}=Pi_q M_{gate_l} Pi_p`.  The exact block evolution is

\[
\begin{pmatrix}h_4^+\\h_R^+\end{pmatrix}
=\begin{pmatrix}T_{44}&T_{4R}\\T_{R4}&T_{RR}\end{pmatrix}
\begin{pmatrix}h_4\\h_R\end{pmatrix},
\qquad R=\{6,8,10,\ldots\}.
\]

An H4 input times gate H2 births H6 as well as H4/H2; gate H4 births H8 and
H6 as well as H4.  These excursions can return: `H6 times H2 -> H4` through a
two-index contraction, and `H8 times H4 -> H4` through a four-index
contraction.  In one scalar coordinate the exact product identities

\[
He_4He_2=He_6+8He_4+12He_2,
\qquad
He_6He_2=He_8+12He_6+30He_4,
\]

\[
He_4He_4=He_8+16He_6+72He_4+96He_2+24,
\qquad
[He_8He_4]_{H4}=1680He_4
\]

give nonzero two-reset excursion-return coefficients `30` through H6 and
`1680` through H8 in the unnormalized basis.  In the orthonormal Hermite basis
the corresponding path multipliers are `15` and `70`.  Thus even a common
rank-one or commuting Jordan algebra does not make chaos-order feedback zero;
it only keeps the returned H4 direction inside a bounded spatial module.

Autonomously carrying those excursion sectors is not a deployment escape.
In a common diagonal basis, the zero-odd H_{2m} sector has dimension

\[
\dim H_{2m}^{0\text{-odd}}=\binom{r+m-1}{m}.
\]

At `r=55`,

\[
\boxed{
R_{H6}=\binom{57}{3}=29260,
\qquad
R_{H8}=\binom{58}{4}=424270.
}
\]

One dense H6 transport costs `29260/1024=28.574` GEMM equivalents; H8 costs
`414.33`.  H8 therefore exceeds the 49 cap with the core on a single crossing,
and H6 cannot persist across the ten-layer suffix.  The only viable treatment
is a target-conditioned transient contraction or an analytic folding of the
Volterra return kernel; a forward autonomous H6/H8 bank is closed.

There is an exact cross-lane identification.  In the commuting eigenbasis, an
even H6 multi-index is `alpha=2 beta` with `|beta|=3`; therefore

\[
\boxed{
H_6^{0\text{-odd}}\cong\operatorname{Sym}^3(\mathbb R^r),
\qquad \dim=\binom{r+2}{3}=29260.
}
\]

This is exactly the coefficient dimension of the H3 CountSketch carrier below.
Consequently a lawfully normalized sketch of the H6 excursion/source and its
return adjoint is algebraically the same kind of `Sym^3` cross-Gram sketch.  It
may be reused only if the source and query tensors, owner subtraction, Hermite
multiplicity weights, and interposed width transport agree exactly; equal
dimension alone is not provenance.

If they do agree, the H6-memory sketch inherits the later CountSketch gates:
`q3>0.9736`, top-`k` energy above `94.79%`, and the exact controlled collision
threshold `k>36.39 C_off,V`.  Thus the additive norm-envelope route is closed
at `k<=36`; only a frozen exact-collision/target-conditioned receipt can make
the compressed H6 return cheaper than the analytic all-pairs fold.

More generally, the common diagonal even-chaos hierarchy is

\[
\boxed{
H_{2m}^{0\text{-odd}}
\cong\operatorname{Sym}^m(\mathbb R^r),
\qquad
\dim=\binom{r+m-1}{m}.
}
\]

Multiplication by an underlying H2 gate is block tridiagonal in `m`.  In one
coordinate the exact recurrence is

\[
\boxed{
He_{2m}He_2
=He_{2m+2}+4m\,He_{2m}
+2m(2m-1)He_{2m-2}.
}
\]

The multivariate operator is the symmetric-Fock creation/preservation/
annihilation analogue with core contractions.  Thus:

- quadratic cores are level `m=1`;
- the H4 square matrix `B` is level `m=2`;
- the 29260-coordinate H3/CountSketch bank is the level-`m=3` H6 bridge;
- the 424270-coordinate H8 sector is level `m=4`.

The `2/16/12` and `1/16/36/16/1` folded returns are precisely finite Schur
complements of excursions from level two into levels three and four.  The
**one-gate** Fourier--Mehler kernel derived above resums all gate-Hermite
contributions to the direct diagonal block `T44`; it does not resum an H6/H8
state that survives into a later layer.  Full cross-layer self-energy requires
the Volterra/folded-return series, or a new suffix-level unprojected Mehler
operator with no intermediate `Pi4` projections.

This gives one coherent constructive architecture: carry the value-visible H4
level, compress or analytically contract the H6 bridge, and resum/bound the H8+
cross-layer tail by the endpoint Volterra (or genuinely suffix-level Mehler)
operator.  It is valid only after the
commuting or target-cyclic provenance gates and the independent cost/error
receipts above.

**Graph-ready synthesis `P2R0G19B_EVEN_FOCK_JACOBI_CLOSURE`:** promote only if
levels `m=2,3` and the resummed `m>=4` self-energy factor causally through the
declared carrier/sketch, jointly fit cost, and meet the endpoint Schur radius.
Close if the H6 sketch lacks provenance/conditioning, the cross-layer
Volterra/suffix-level self-energy is not bounded, or a target-visible Jacobi
excursion lies in the carrier kernel.

The joint `m=2+m=3` compute ledger is narrowly feasible before `m>=4` memory.
Using the dense exact direct-H4 count `46.66`, the ten-layer r55 CountSketch
increment is

\[
C_3(k)=10\left(\frac{2k}{1024}+0.0279045+g_{src}\right).
\]

Thus:

- `k=36`: total `47.64` with exposed source coordinates, or `47.92` with one
  comparable structured source projection;
- `k=22`: total `47.36` or `47.64`, respectively.

This leaves only `1.08--1.64` equivalents for hooks and the entire
cross-layer `m>=4` self-energy.  Therefore the arithmetic does not close the
full hierarchy, but it identifies the only plausible package: exact H4 plus a
frozen-collision H6 sketch, with H8+ required to be target-dark or essentially
free through suffix-level fusion.  Recall that `k=36/22` already fail the
uncontrolled norm envelope; their exact `C_off,V` and singular-energy gates are
binding.

The first H6 excursion does admit an exact analytic fold.  Work first with
unnormalized Wiener kernels: start from `I4(f)` and let two successive gate-H2
kernels be `I2(M)` and `I2(N)`.  The H6 birth is
`I6(sym(f tensor M))`; the next product returns

\[
30I_4\left(\operatorname{Sym}(f\otimes M)\otimes_2N\right).
\]

Among the `binom(6,2)=15` placements of the two `M` legs, the return contraction
hits two, one, or zero of them in counts `1,8,6`.  Therefore

\[
\boxed{
\mathcal R_{N\leftarrow M}^{(6)}f
=2\langle M,N\rangle f
+16\mathcal A_{M,N}f
+12\operatorname{Sym}\left\{
M\otimes(f\otimes_2N)
\right},
}
\]

where

\[
\mathcal A_{M,N}f
=\frac14\sum_{q=1}^4
(M\circ N)\text{ acting on leg }q\text{ of }f.
\]

For one scalar coordinate all three operations are the identity and
`2+16+12=30`, recovering the Hermite witness.  If the physical indicator-H2
coefficient at layer `l` is

\[
c_l=\frac{\sigma_l\theta_{1,l}}{2s_l^2}
\quad\text{in front of }I_2(M_l),
\]

the folded two-reset action is multiplied by `c_l c_{l+1}` (plus the actual
dense width transport between the two gates).

The reverse contraction is immediate.  `A_{M,N}` is self-adjoint because
`M circ N` is symmetric, while

\[
\left[\operatorname{Sym}\{M\otimes(f\otimes_2N)\}\right]^*
\Lambda
=\operatorname{Sym}\{N\otimes(\Lambda\otimes_2M)\}.
\]

Hence

\[
\boxed{
(\mathcal R_{N\leftarrow M}^{(6)})^*
=\mathcal R_{M\leftarrow N}^{(6)}.
}
\]

The interposed dense width map is likewise transposed in the actual suffix.

In the commuting parity-zero representation every term in `R^(6)` reduces to
diagonal/Lyapunov contractions of the `r by r` coefficient matrix `B`, so the
H6 tensor need never be materialized.  Explicitly, for gate eigenvalue vectors
`m,n`, put

\[
g_n(B)=\frac13\left[Bn+2\operatorname{diag}(B)\odot n\right],
\qquad D_{mn}=\operatorname{Diag}(m\odot n).
\]

Then the exact folded return on coefficient matrices is

\[
\boxed{
\mathcal R_{n\leftarrow m}^{(6)}(B)
=2\langle m,n\rangle B
+8(D_{mn}B+BD_{mn})
+6\{m g_n(B)^T+g_n(B)m^T\}.
}
\]

For `r=1` this gives `2+16+12=30`.  Each source-target pair needs only
`O(r^2)` work: a scalar dot, diagonal scalings, one matrix-vector product, and
one symmetric outer product.

A direct all-pairs implementation over a dense width transition is
`O(n^2r^2)`.  One unit of `n^2r^2` work is already
`r^2/n=2.954` GEMM equivalents per transition at `r=55`; the displayed formula
has several such primitives.  Thus `29.54` across ten sites is an optimistic
floor, and it does not fit on top of the exact H4 carrier.  Promotion therefore
needs fusion/reuse, fewer active return sites, or a target-conditioned
contraction below this all-pairs bound.

**Graph-ready child `P2R0G19B_H4_H6_FOLDED_RETURN`:** promote on the exact
`2/16/12` contraction, lawful placement of the interposed width map, total
fused cost within the remaining package, and endpoint action error below the
robust radius.  Close if the all-pairs cost is unavoidable across ten sites or
if omitted multiple/H8 excursions remain value-visible.

The H8 excursion has an analogous exact fold.  Let the first and second
gate-H4 kernels be `g` and `h`.  Starting from `I4(f)`, the H8 birth is
`I8(Sym(f tensor g))`; contraction with `I4(h)` returns with Wiener coefficient
`4! binom(8,4)=1680`.  If `k` of the four contracted legs came from `g`, the
placement counts are

\[
\binom4k\binom4{4-k}=1,16,36,16,1.
\]

Let `C_k(f,g;h)` be the normalized symmetric contraction averaged over that
placement class.  Then

\[
\boxed{
\mathcal R_{h\leftarrow g}^{(8)}f
=24\sum_{k=0}^4\binom4k^2\mathcal C_k(f,g;h).
}
\]

The endpoints are
`C_0=<f,h>g` and `C_4=<g,h>f`, with
`C_{4-k}(f,g;h)=C_k(g,f;h)`; the three middle terms are the mixed one-, two-,
and three-leg contractions.  In one scalar coordinate every `C_k` is the
identity and the weights sum to `1680`, as required.

By the same multiplication/projection adjoint identity,

\[
\boxed{
(\mathcal R_{h\leftarrow g}^{(8)})^*
=\mathcal R_{g\leftarrow h}^{(8)}.
}
\]

This formula proves H8 can be folded algebraically without a 424270-mode
state.  It does **not** prove deployment feasibility: the actual gate-H4
kernels include all relevant reset components, the dense width map must be
inserted between birth and return, and the mixed contraction cost must be
billed.  Its target-conditioned endpoint action and interference with the H6
fold are the binding quantities.

Zero bias does not make the finite-core gate-H4 kernel vanish.  The scalar
Hermite `m=2` and `m=3` pieces give, exactly,

\[
\Pi_4 h_2(Q_M/s)=\frac{U_M}{\sqrt2s^2},
\qquad
\Pi_4 h_3(Q_M/s)=\frac{\sqrt6}{s^3}U_{M,M^2},
\]

because `Pi4(Q_M^3)=6U_{M,M^2}`.  With

\[
d_2=-\frac{a\phi(a)}{\sqrt2},
\qquad
d_3=\frac{(a^2-1)\phi(a)}{\sqrt6},
\]

the corresponding underlying gate-H4 kernel is

\[
\boxed{
\Gamma_4^{(2:3)}
=-\frac{a\phi(a)}{2s^2}U_M
+\frac{(a^2-1)\phi(a)}{s^3}U_{M,M^2}.
}
\]

At `a=0`, the first term vanishes but the second is
`-phi(0)U_{M,M^2}/s^3`, generally nonzero.  The tensor normalization gives the
safe bounds

\[
\|U_{A,B}\|_2
\le\sqrt{\frac32}\|A\|_F\|B\|_F,
\qquad
\tau=\frac{\operatorname{tr}M^4}{(\operatorname{tr}M^2)^2},
\]

and hence

\[
\boxed{
\|\Gamma_4^{(2)}\|_2
\le\frac{\sqrt6}{2}|a|\phi(a),
\qquad
\|\Gamma_4^{(3)}\|_2
\le2\sqrt3\,|1-a^2|\phi(a)\sqrt\tau.
}
\]

Higher scalar Hermite orders can also contract down to underlying H4 and need
a remainder bound.  Thus H8 memory is suppressed at near-zero bias only when
the core is spectrally diffuse **and** the higher contraction tail is
controlled; scalar parity alone is not a zero certificate.

Even maximal isotropic diffuseness is not a smallness proof.  For
`M=lambda I_r`, `M^2=lambda M` and the zero-bias `m=3` term has the exact norm

\[
\boxed{
\|\Gamma_4^{(3)}\|_2
=\phi(0)\sqrt{\frac4r+\frac8{r^2}}.
}
\]

At `r=55` this is about `0.1095`.  Since the normalized H8 excursion-return
multiplier can be `70`, a pair of such gate amplitudes is not automatically
negligible.  Only the actual tensor contractions and endpoint adjoint can
certify cancellation.

**Graph-ready child `P2R0G19B_H4_H8_FOLDED_RETURN`:** promote only with the
exact `1/16/36/16/1` contraction provenance, a fused cost below the remaining
cap, and a joint H6/H8/multiple-excursion Bellman error bound.  Close if the
mixed contraction requires an autonomous H8-equivalent state or exceeds cost.

Eliminating `R` yields the exact H4 memory expansion

\[
\boxed{
T_{44}^{\rm eff}(L:0)
=T_{44}^{(L:0)}
+\sum_{i<j}T_{4R}^{(L:j+1)}T_{RR}^{(j:i+1)}T_{R4}^{(i)}
+\text{multiple excursions}.
}
\]

More explicitly, when the discarded sector is initialized at zero,

\[
h_{R,l}
=\sum_{i<l}
T_{RR}^{(l-1:i+1)}T_{R4}^{(i)}h_{4,i},
\]

so the retained component satisfies the exact Volterra recurrence

\[
\boxed{
h_{4,l+1}
=T_{44}^{(l)}h_{4,l}
+\sum_{i<l}K_{l,i}h_{4,i},
\qquad
K_{l,i}=T_{4R}^{(l)}T_{RR}^{(l-1:i+1)}T_{R4}^{(i)}.
}
\]

This formula already includes arbitrarily long residence in H6/H8; repeated
excursions are generated when the Volterra recurrence is iterated.  For a
terminal adjoint `Lambda`, a safe first omitted-memory action bound is

\[
\boxed{
|\rho_{\rm mem}|
\le\sum_{i<l}
\|\Lambda_{l+1}\|\,
\|T_{4R}^{(l)}\|
\left\|T_{RR}^{(l-1:i+1)}\right\|
\|T_{R4}^{(i)}\|\,
\|h_{4,i}\|,
}
\]

followed by the ordinary discrete resolvent/Gronwall factor for feedback of
this error into later `h4`.  The sharper and binding gate is candidate-specific
annihilation of every `P_N Phi^* K_{l,i}` or a direct bound on their fused
operator norm.

The Markov H4-only recurrence retains only the first term.  It is exact only
if every pulled-back terminal adjoint kills all excursion-return words, or if
those words are analytically folded into a finite invariant cyclic state.
Otherwise H6/H8 must be carried or their full adjoint-weighted memory remainder
must meet the action-error gate.

**Graph-ready splits:**

- `P2R0G19B_H4_GATE_TAIL_DIRECT`: promote on a structural-zero proof or a
  candidate-specific adjoint leakage bound within `3.80e-6` and the robust
  Schur gate; close on a certified lower witness above tolerance.
- `P2R0G19B_H4_CHAOS_RETURN_MEMORY`: promote only if the backward cyclic module
  includes every H6/H8 excursion-return word within budget, or their summed
  adjoint action is below tolerance; close if a two-step return witness is
  value-visible and cannot be represented by the retained state.
- `P2R0G19B_H4_JORDAN_INVARIANCE`: promote spatial closure only for the full
  common Jordan module with a reversible/Wick-word closure receipt (automatic
  in the stated commuting/rank-one cases); do not use binary Jordan closure as
  evidence that higher-tail or chaos-order memory vanishes.  The commuting
  child additionally requires `rank G<=r` and one shared eigenbasis across the
  trajectory; close it immediately on `rank G>r`.

For a finite, non-diffuse core define

\[
\gamma=\frac{\operatorname{tr}M^3}{s^3},\qquad
\kappa=\frac{3\operatorname{tr}M^4}{s^4},\qquad
D=\sqrt{2+\kappa-\gamma^2},
\]

and the orthogonalized quadratic polynomial

\[
p_2(X)=\frac{X^2-1-\gamma X}{D}.
\]

Writing `T_j(a)=E[X^j 1_{X>-a}]`, the first two nonconstant reset
coefficients are exactly

\[
c_1=\sigma\{aT_1+T_2\},
\]

\[
c_2=\frac{\sigma}{D}
\left[
a(T_2-T_0-\gamma T_1)+(T_3-T_1-\gamma T_2)
\right].
\]

The corresponding finite-core retained-energy fraction is also exact.  With

\[
\mu_g=\sigma(aT_0+T_1),\qquad
V_g=\sigma^2(a^2T_0+2aT_1+T_2)-\mu_g^2,
\]

orthonormality of `X,p2(X)` gives

\[
\boxed{
F_{\{X,p_2\}}(M,a)=\frac{c_1^2+c_2^2}{V_g}.
}
\]

This is the exact finite-spectrum replacement for the diffuse
`F_H2+H4(a)` formula.  It supplies a purely analytic reset gate once the tail
moments are evaluated from the core spectrum; no Monte Carlo fidelity estimate
is needed.

All required tail moments can in principle be recovered without sampling from
the exact characteristic function

\[
\varphi_{Q_M}(t)
=e^{-it\operatorname{tr}M/2}\det(I-itM)^{-1/2}.
\]

### Diffuse zero-bias reset value and repeated-reset obstruction

At zero bias, the exact centered ReLU energy fractions are

\[
F_{H2}=\frac{\pi}{2(\pi-1)}=0.73347\ldots,
\qquad q_{H2}=0.85643\ldots,
\]

and

\[
\boxed{
F_{H2+H4}=\frac{\pi+1}{2(\pi-1)}=0.96695\ldots,
\qquad q_{H2+H4}=0.98334\ldots .
}
\]

This is the constructive motivation: one square reset retains nearly all the
task direction.  It is not, however, a license to discard the state after every
layer.  Independent reset losses multiply.  In particular `q^5 < 0.925`, so
five destructive square resets already miss the task-correlation gate.  The
`H4` component must be carried persistently, or the discarded complement must
be proved irrelevant by a target-conditioned adjoint.

The minimal shared-bank construction is

\[
H_i=\sum_{s=1}^{R} C_{is}U_{R_s},\qquad R=kn.
\]

It uses `O(k n^2)` state and a dense coefficient transport `C <- W^T C` costs
`O(k n^3)`.  One new square generator per neuron per layer keeps `k` bounded
only if all reset births project into the existing bank.  A correlated gate
generically produces mixed generators `U_{R_s,M_i}`; then `R -> nR` per reset
and the nominal square lift explodes.  Exact bounded closure therefore requires
one of:

1. all quadratic cores lie in a common low-dimensional Jordan algebra closed
   under reset products;
2. a target-conditioned adjoint proves the newly generated mixed sector dark;
3. a certified low-rank projection of each birth back to the shared bank.

For a common Jordan algebra `A` of dimension `q`, the fourth-chaos square module
has dimension at most `q(q+1)/2`.  A commuting `q ~ sqrt(n)` core family can
therefore fit `O(n^2)` state and `O(n^3)` dense work.  Generic cores generate
`Sym_r`, making the fourth module quadratic in `n` and restoring the
`Omega(n^3)` state / `Omega(n^4)` work obstruction.

The exact rank gate is the degree-four Hilbert function of the observed
quadratic subspace.  Let

\[
V=\operatorname{span}\{Q_{M_i}:1\le i\le n\},\qquad
m=\dim V=\operatorname{rank}G,
\]

and define the multiplication map followed by fourth-chaos projection

\[
\mu_4:\operatorname{Sym}^2(V)\longrightarrow H_4(\mathbb R^r),
\qquad Q_M\odot Q_N\longmapsto U_{M,N}.
\]

Because fourth-chaos projection preserves the homogeneous degree-four leading
term, `ker(mu4)` is exactly the space of quadratic-product syzygies in `V`.
Thus the minimal autonomous square-bank size is

\[
\boxed{
R_{H4}=\operatorname{rank}\mu_4,
\qquad
R_{H4}\le
\min\left\{\binom{m+1}{2},\binom{r+3}{4}\right\}.
}
\]

An `R=kn` carried bank can close only if

\[
\boxed{\operatorname{rank}\mu_4\le kn.}
\]

This condition is exact and basis/gauge invariant.  A low-dimensional Jordan
algebra is one sufficient way to create many syzygies, but it is not the
definition of closure.

There is a strong unconditional lower bound when the provider claims closure
under arbitrary dense mixtures of the entire `m`-dimensional core space.  Such
a provider contains `U_R` for every `R in V`; polarization then forces the
whole image of `mu4`.

Choose a row-echelon basis of `V` in the quadratic monomial basis and let `A`
be its `m` distinct leading exponent vectors, each of total degree two.  For
every distinct sum in `A+A`, a product of basis quadrics has that distinct
leading quartic monomial.  Hence

\[
\boxed{\operatorname{rank}\mu_4\ge|A+A|.}
\]

View `A` as edges (loops allowed) of a graph on the `r` variables.  A set of
degree-two exponent vectors contained in an affine subspace of dimension `d`
has at most `binom(d+2,2)` points: a nonbipartite component on `d+1` vertices
with all loops/edges attains the maximum; bipartite or disconnected components
have fewer edges at the same affine dimension.  Thus `m=1024` forces
`dim aff(A)>=44`, because `binom(45,2)=990<1024<=binom(46,2)`.

For completeness, the incidence-rank proof is: a connected looped/nonbipartite
component on `v` vertices has affine dimension `v-1` and at most
`binom(v+1,2)` loop-edges; a loopless bipartite component has dimension `v-2`
and at most `floor(v^2/4)` edges, which is smaller.  For multiple components,
the affine dimensions add with one extra joining dimension per component, and
convexity of the binomial bound makes a single component extremal.

Freiman's dimension lemma in the torsion-free exponent lattice gives

\[
|A+A|\ge(d+1)|A|-\frac{d(d+1)}2.
\]

In general, if

\[
d_m=\min\left\{d:\binom{d+2}{2}\ge m\right\}
=\left\lceil\frac{\sqrt{8m+1}-3}{2}\right\rceil,
\]

then every uniform `m`-dimensional quadratic square module obeys

\[
\boxed{
R_{H4}\ge(d_m+1)m-\frac{d_m(d_m+1)}2.
}
\]

At `m=1024,d>=44`,

\[
\boxed{
\operatorname{rank}\mu_4\ge45(1024)-\frac{44\cdot45}{2}
=45090.
}
\]

Therefore a uniform autonomous H4 bank needs

\[
k\ge\left\lceil\frac{45090}{1024}\right\rceil=45,
\qquad
\frac{R_{H4}}n\ge44.033
\]

GEMM equivalents for **each** dense width transport.  Adding the quoted
9-equivalent core already gives at least `53.03>49` with a single such
transport, before births/hooks.  This is a rigorous deployment no-go, not a
generic-rank heuristic.

The premise matters.  A single frozen trajectory that visits only a strict
subset of square births, or a target-conditioned quotient, can evade the full
`A+A` module.  It must then certify the rank of its actual birth/cyclic module;
it cannot claim the uniform autonomous closure proved impossible here.

The 49-cap gives that exception an exact size limit.  Let `R_vis` be the
dimension of the full backward-cyclic, endpoint-visible H4 module across ten
late dense transports.  With the 9-equivalent core and negligible local
updates,

\[
\boxed{9+10R_{\rm vis}/n\le49
\quad\Longrightarrow\quad R_{\rm vis}\le4096.}
\]

If each local exact H4 update is a dense `R_vis by R_vis` action, the cost is

\[
9+10\left(x+x^2\right)\le49,
\qquad x=R_{\rm vis}/n,
\]

so

\[
\boxed{
x\le\frac{\sqrt{17}-1}{2},
\qquad R_{\rm vis}\le1599.
}
\]

Let `sigma_j` be the endpoint/value-whitened singular values of the exact
birth-to-Bellman map.  Any rank-`R_vis` carrier has the necessary capacity gate

\[
\boxed{
\frac{\sum_{j>R_{\rm vis}}\sigma_j^2}{\sum_j\sigma_j^2}
\le1-q_\star^2.
}
\]

For the H4 local target `q_star=0.98346`, the visible top modes must explain at
least `96.719%` of endpoint energy; even the weaker global `0.9253` gate needs
at least `85.618%`.  These are necessary before source, hook, H6/H8-memory, or
conditioning costs, so the actual allowed rank is smaller.

At `r=55`, the entire fourth-chaos coefficient space has

\[
\binom{58}{4}=424270
\]

dimensions.  If `rank G=1024` and the candidate-specific multiplication map
has maximal rank, then `R_H4=424270`, requiring `k>=415`.  The coefficient
state alone is `n R_H4=434452480` scalars, and each dense width transport costs
`R_H4/n=414.33` width-GEMM equivalents per layer.  This is far outside the
49-equivalent total deployment budget.

Maximal rank is a generic determinantal claim, not an observed receipt: it is
proved for the candidate family only by a nonzero maximal minor (or an exact
syzygy computation).  Conversely, any constructive H4 lane must exhibit
enough lawful syzygies to reduce `rank(mu4)` from the ambient value to at most
`kn`; merely describing each reset birth as a square does not reduce this
rank.  A target-conditioned adjoint can evade the autonomous bound by replacing
`rank(mu4)` with the rank of its value-visible quotient.

The local `0.98334` square fidelity clears the global `0.9253` target but is
slightly below the equal-fidelity `V+k3+k4` target `0.98346`.  Applying its ideal
gain mechanically to the currently observed approximately `0.78` carrier
correlation gives only about `0.896`; the square lift is therefore a carrier
repair/complement, not by itself a proof of the required raw-MSE reduction.

The general-bias formula shows that the tiny local gap is not structurally
fixed.  At zero bias,

\[
\left.\frac{d}{da}F_{H2+H4}(a)\right|_{a=0}
=\sqrt{\frac\pi2}\,\frac{\pi-3}{(\pi-1)^2}>0.
\]

Since `0.98346^2=0.96719357...`, the first-order crossing from the zero-bias
fraction occurs at standardized bias only about `a=6.5e-3`.  This is a useful
sensitivity fact, not a promotion receipt: the exact bias formula (or the
finite-core `F_{X,p2}` formula) must be used at each active layer, and repeated
losses still multiply.

**Disposition:** retain as a conditional, persistent, target-conditioned square
lift.  Generic autonomous `H4` propagation is a complexity no-go.

**Graph-ready node `P2R0G19B_H4_HILBERT_RANK`:** autonomous square-bank
closure is promoted only by an exact `rank(mu4)<=kn` syzygy receipt together
with the corresponding transport cost.  The uniform `m=1024` branch is now
**closed** by the unconditional `rank(mu4)>=45090` sumset bound, which makes one
transport plus the core exceed 49.  A target-conditioned/actual-trajectory
quotient is a separate open child and must certify the rank and singular-energy
capture of the full backward cyclic module (`R<=4096` for cheap local updates,
`<=1599` for dense local maps), not merely one endpoint row.

**Graph-ready child `P2R0G19B_H4_TARGET_CYCLIC_QUOTIENT`:** open.  Promote
only with an exact causal factorization of every direct-tail and H6/H8-return
Bellman row through one cyclic module, rank within the appropriate 4096/1599
cost ceiling, top-rank energy above the declared correlation gate, and endpoint
error below `min(3.80e-6,eta_star)`.  Close if any target-visible birth/return
secant lies in the carrier kernel or the certified cyclic rank/cost exceeds the
cap.

## 2026-08-30 — inverse quadratic-core moment problem

Let `M_1,...,M_n in Sym_r`, and normalize the observed planes by

\[
G=2C,\quad G_{ij}=\operatorname{tr}(M_iM_j),\qquad
K=K21,\quad K_{ij}=\operatorname{tr}(M_i^2M_j).
\]

### Rank and parameter-count gates

Necessarily

\[
G\succeq0,
\qquad
\operatorname{rank}G\le d_r:=r(r+1)/2.
\]

Hence

\[
r\ge\left\lceil\frac{\sqrt{8\operatorname{rank}C+1}-1}{2}\right\rceil.
\]

For full rank at `n=1024`, `r_min=45` because `d_44=990` and `d_45=1035`.
After quotienting the common orthogonal-conjugacy gauge, the parameter count is

\[
p(r)=n d_r-r(r-1)/2.
\]

The `C,K21` observations have at most
`n(n+1)/2+n^2` scalar coordinates.  Thus generic local finite identification
can only be expected when

\[
p(r)\le n(n+1)/2+n^2.
\]

At `n=1024` this count permits `r=45,...,54`.  At `r=55` it leaves at least
`2099` local ambiguity dimensions; at `r=64`, at least `554528`.  These are
dimension gates, not proofs of Jacobian rank.

### Exact feasibility inequalities

Every row of `K` must lie in the row space of `G`:

\[
\boxed{K=KG^+G.}
\]

Let `P=KG^+K^T`.  Since this is the Gram matrix of the projections of `M_i^2`
onto `span{M_j}`, and put

\[
q_i=\operatorname{tr}M_i^4=K31_{ii}/3=K22_{ii}/3.
\]

Then

\[
P_{ii}\le q_i,\qquad
\frac{G_{ii}^2}{r}\le q_i\le G_{ii}^2,
\qquad
K_{ii}^2\le G_{ii}q_i.
\]

Introduce

\[
A_{ij}=\operatorname{tr}(M_i^2M_j^2).
\]

The augmented moment matrix obeys

\[
\boxed{
\begin{pmatrix}G&K^T\\K&A\end{pmatrix}\succeq0,
\qquad
\operatorname{rank}\begin{pmatrix}G&K^T\\K&A\end{pmatrix}\le d_r.
}
\]

Moreover `A` is completely-positive-semidefinite with cpsd-rank at most `r`,
`diag(A)=q`, and the fourth planes force the elementwise interval

\[
\boxed{K22/3\le A\le K22.}
\]

Consequently

\[
K22=K22^T,\quad K22_{ij}\ge0,
\quad \frac{G_{ij}^2}{r}\le K22_{ij}
\le3\sqrt{q_iq_j}.
\]

The pure-square fourth Gram itself gives the especially cheap falsification
gate

\[
\boxed{H^{(4)}=\frac12(G\odot G)+K22/3\succeq0.}
\]

For `L=K31/3`, a further support and norm gate is

\[
L=LG^+G,
\qquad
l_iG^+l_i^T\le q_i^{3/2},
\qquad
K31_{ii}=K22_{ii}.
\]

These conditions are necessary; the cpsd/Jordan realization is the remaining
nonconvex sufficiency problem.

### Identifiability, gauge, and exact Jacobian gate

For perturbations `dot M_i`,

\[
\dot G_{ij}=\operatorname{tr}(\dot M_iM_j+M_i\dot M_j),
\]

\[
\dot K_{ij}
=2\operatorname{tr}\{\dot M_i(M_i\circ M_j)\}
+\operatorname{tr}(M_i^2\dot M_j).
\]

The unavoidable gauge tangent is

\[
\dot M_i=[\Omega,M_i],\qquad \Omega^T=-\Omega.
\]

The core is locally identifiable modulo gauge exactly when the kernel of the
displayed `(G,K)` Jacobian equals this gauge space.  For the task, the weaker and
correct gate is

\[
\boxed{\ker D\mathcal O\subseteq\ker D\Xi,}
\]

where `Xi` is the transported target quantity (including signed `H3` and
external `q4(211)` if those feed value).  Full core uniqueness is unnecessary.

### Global sign/cycle counterexamples

There are exact dark fibers.  In `r=3`, take three edge-supported symmetric
matrices on a triangle and flip one edge sign.  `C,K21,K31,K22` in the original
basis are unchanged, while `tr(M_1M_2M_3)` flips sign.  In `r=4`, the analogous
four-edge cycle has all original-basis pair planes unchanged while the
all-distinct quartic cycle trace flips.  Therefore held-out `K31/K22` in the
same coordinate frame can falsify a proposed core but cannot certify its dense
transport behavior.

If the entire symmetric cubic tensor

\[
T_{ijk}=\operatorname{tr}(M_iM_jM_k)
\]

is known and `V=span{M_i}` is Jordan closed, then the structure constants are

\[
c_{ij}^{\ k}=T_{ij,:}G^{-1},
\]

and the higher planes are forced:

\[
K31_{ij}=3\langle M_i^2,M_i\circ M_j\rangle,
\]

\[
K22_{ij}=\langle M_i^2,M_j^2\rangle
+2\|M_i\circ M_j\|_F^2.
\]

Thus the inverse problem is exactly a **Jordan completion** problem: `K21`
reveals only `T_{iij}`, not the full product table.

Useful exact polarizations are

\[
T_{ijk}=\frac14\{T(e_i+e_j,e_i+e_j,e_k)
-T(e_i-e_j,e_i-e_j,e_k)\},
\]

and an all-distinct symmetric quartic coefficient can be recovered from square
planes by

\[
T_{ijkl}=\frac1{64}\sum_{\epsilon,\eta\in\{\pm1\}^2}
\epsilon_1\epsilon_2\eta_1\eta_2
T(x_\epsilon,x_\epsilon,y_\eta,y_\eta),
\]

with the obvious signed two-coordinate combinations `x_epsilon,y_eta`.

### Audit of the explicit diagonal-plus-star q211 fiber

The cubic lane supplies a concrete lower-moment fiber:

\[
A=\operatorname{diag}(5,6,7,9),\qquad B=dI+S,
\]

where `S` is the star centered at coordinate one and its three squared edge
weights are

\[
s(t)=(1+2t,\ 1-3t,\ 1+t).
\]

All declared trace words through degree three are constant.  The fourth-plane
audit is exact.  Put

\[
X=\operatorname{tr}(A^2B^2),\qquad
Y=\operatorname{tr}(ABAB).
\]

Since only the star squares vary,

\[
\dot X
=(5^2+6^2)2+(5^2+7^2)(-3)+(5^2+9^2)1
=6,
\]

whereas

\[
\dot Y
=2\cdot5\{6(2)+7(-3)+9(1)\}=0.
\]

With the standing normalization `K22(A,B)=2X+Y`,

\[
\boxed{
\frac d{dt}K22(A,B)=12,
\qquad
\frac d{dt}\|[A,B]\|_F^2=12.
}
\]

Because `G_AB` is constant, the pure-square Gram entry
`H4_AB=.5 G_AB^2+K22_AB/3` has derivative `4`.  Therefore the one-dimensional
fiber tangent has

\[
\boxed{
\operatorname{rank}D\mathcal O_{C,K21,K22}=1;
}
\]

it is not dark once the repeated-pair fourth plane is observed.  At `t=0`, the
edge derivative is `(1,-3/2,1/2)` and `||dot B||_F=sqrt(7)`, so the normalized
action magnitudes are `12/sqrt(7)` for raw K22 or `4/sqrt(7)` for the CP4/H4
normalized plane.

This closes the witness against the claim of **complete K22 darkness**: K22
does observe this tangent.  It remains a valid counterexample to `(C,K21)`
identifiability.  Since the old
CP4 interface already owns the exact K22 plane, observing this tangent there is
an identifiability/checksum receipt, not incremental board value; any new H4
feature must survive old-CP4 residualization.

### Literal D11/CP4 audit of the off-diagonal Vandermonde-star tangent

The stronger six-leaf star construction from the cubic lane fixes every
marginal and pairwise H4 face while moving a genuinely off-diagonal coordinate:

\[
\dot\kappa_4(q_A,q_A,q_B,q_C)=32,
\qquad B\ne C.
\]

Its barycentric tangent annihilates the five moments with powers `0,...,4`
and not the power-five word.  Unlike the preceding three-leaf witness, all
`K4`, `K31`, and `K22` faces on one or two named coordinates have zero
derivative.

The literal deployed CP4 tape represents

\[
T_{\rm cp4}=\sum_s\lambda_su_s^{\otimes4},
\]

but `_cp4_read_planes` exposes only

\[
d_i=T_{iiii},\qquad A_{ij}=T_{iiij},\qquad B_{ij}=T_{iijj}.
\]

At reset, `_cp4_reset` gates the already-owned factors and appends identity
columns whose coefficients are determined only by the marginal `k4_target`.
It has no acquisition map from an external all-distinct `AABC` coordinate into
`(lambda,u)`.  On the Vandermonde tangent the inherited tape, gate, and
marginal K4 are fixed, so

\[
\boxed{D(\lambda_{\rm cp4},u_{\rm cp4})[h]=0.}
\]

The other reset state is also fixed.  `_reset_stars` receives the K21 plane,
the diagonal-plus-K31 quartic star, and the scalar pairing fit from symmetric
K22.  Every one of these inputs has zero derivative on the construction;
hence

\[
\boxed{
D(\text{cubic star},\text{quartic star},\alpha_{22},\text{CP4 tape})[h]=0,
\qquad
Dq_{211,\rm offdiag}[h]=32.
}
\]

All later Q24/D11 state is a deterministic function of this reset state, so
its derivative is zero as well.  Although an already-owned CP4 factorization
has a mathematically evaluable all-distinct entry
`sum_s lambda_s u_is u_js u_ks u_ls`, that entry is unchanged here: implicit
evaluability is not an acquisition port.

The seam qualification is essential.  A mixed quartic direction encoded in
the CP4 factors *before* a dense push can rotate into later repeated faces and
be observed.  The theorem concerns a new off-diagonal source born at the
audited reset, where the literal interface records only marginal/pairwise
faces.  At that seam,

\[
\boxed{
\ker D\mathcal O_{\rm D11,reset}
\not\subseteq
\ker Dq_{211,\rm offdiag}.
}
\]

**Graph-ready node `P2R0G19B_H4_CP4_OFFDIAG_ACQUISITION_GAP`:** solved source
novelty theorem.  Evidence is the Vandermonde-star tangent plus the literal
CP4/reset Jacobian above.  This proves a missing provider coordinate, not
positive board value.  Promotion still requires its old-span Schur gain,
future action factorization, cost, and numerical-error gates; close only if a
different literal acquisition port is exhibited that is nonzero on this exact
tangent.

### Minimal CP4 repair at that seam: two mixed rays modulo Star4+pairing

The missing `AABC` coordinate has a rank-two CP representative after the
literal marginal/K31 star and covariance-pairing gauge is used correctly.
Assume the covariance identities C_AB=C_AC=0, as in the exact Vandermonde
witness, and define the three pairing coordinates

\[
s_1=C_{AA}C_{BB},\qquad
s_2=C_{AA}C_{CC},\qquad
s_3=C_{BB}C_{CC}+2C_{BC}^2,
\]

\[
\rho_{BC}=\frac{C_{BC}}{\sqrt{C_{BB}C_{CC}}},
\qquad
p_0=\frac{s_3}{\sqrt{s_1s_2}}.
\]

Choose the covariance-aligned singleton coefficients

\[
\beta=\sqrt{p_0}(s_1/s_2)^{1/4},\qquad
\gamma=\sqrt{p_0}(s_2/s_1)^{1/4},
\]

\[
u_+=A+\beta B+\gamma C,\qquad
u_-=-A+\beta B+\gamma C
\]

with the same weight `lambda`.  For a desired pre-pair q211 amplitude
`qprime`, set

\[
\boxed{\lambda=\frac{q'}{2p_0}.}
\]

The pair has q211 quotient vector (qprime,0,0) on
(AABC,ABBC,ABCC), and its complete K22 leakage is

\[
\eta(s_1,s_2,s_3),\qquad
\eta=\frac{q'}{\sqrt{s_1s_2}}.
\]

This is precisely the resident covariance-pairing direction.  Subtracting it
removes `rho_BC qprime` from the desired coordinate and leaves

\[
\boxed{
T_{AABC}^{\rm residual}=q'(1-\rho_{BC}).
}
\]

Therefore a desired raw coordinate `T_AABC=t` is supplied by

\[
\boxed{
q'=\frac{t}{1-\rho_{BC}},
\qquad
\lambda=\frac{t}{2p_0(1-\rho_{BC})}.
}
\]

For the Vandermonde tangent `t=32`, this is the two-ray weight
`16/[p0(1-rho_BC)]`.  Strict Cauchy--Schwarz gives
`|rho_BC|<1` because the star vectors `b=(1,...,1)` and `c=(x_j^3)` are not
proportional.

For the concrete lawful choice `x_j=j`, `j=1,...,6`, this conditioning is
available in closed form:

\[
b\cdot c=\sum_{j=1}^6j^3=441,\qquad
\|c\|^2=\sum_{j=1}^6j^6=67171,
\]

\[
\boxed{
\rho_{BC}=\frac{441}{\sqrt{403026}},\qquad
\kappa_{\rm pair}
=\frac{\sqrt{403026}}{\sqrt{403026}-441}
\approx3.28.
}
\]

Thus the explicit algebraic witness itself is not near the collinear
singularity.  This does not bound the correlation of the actual frozen-net
source factors.

These coefficients use the raw-cumulant convention.  A CP tape normalized by
`1/4!` scales both the source and every ray weight uniformly by `1/24`; it
does not change the rank, cancellation factor, or ownership statement.

All remaining marginal and odd K31 faces are in the
Star4/marginal gauge and must be canceled there.  Rank one cannot isolate
`AABC`: one triple-supported ray contributes a q211 vector proportional to

\[
\alpha\beta\gamma(\alpha,\beta,\gamma)
\]

on `(AABC,ABBC,ABCC)`, so a nonzero first coordinate necessarily leaks into
at least one of the other two.  The displayed pair cancels both leaks and is
minimal modulo the declared gauge.

If the pairwise cancellation receipt fails, the unconditional no-leak Waring
representation is the six-ray polarization

\[
\begin{aligned}
48T_{AABC}={}&p(A+B+C)-p(A+B-C)\\
&-p(A-B+C)+p(A-B-C)\\
&-2p(B+C)+2p(B-C),
\end{aligned}
\]

which requires no old pairwise compensation.

This gives a literal acquisition port absent from the current reset: append
the two moving factors `u_+,u_-` with the signed source weight, then let the
already-billed CP4 push rotate them so the hidden source appears in ordinary
future K4/K31/K22 reads.  Two columns add negligible dense arithmetic if they
replace retained capacity inside the existing CP4 tape; their source
formation, cap displacement, and amplitude must still be billed/audited.

If the CP4 cap is already full, two columns are not free merely because their
dense transport is already billed.  Let `O` be the common retained response
bank, `Q` the responses of the displaced CP columns, and `F` the two-ray
responses.  With value metric `W`, target `y`, and
`R_O=I-O(O^TWO)^+O^TW`, define

\[
S_X=X^TWR_OX,\qquad d_X=X^TWR_Oy.
\]

The exact capacity-exchange gate is

\[
\boxed{
\Delta_{\rm exchange}
=d_F^TS_F^+d_F-d_Q^TS_Q^+d_Q
\ge2.86\times10^{-10}.
}
\]

A source-energy top-cap score does not imply this response-space inequality.
The factor birth/drop rule must be frozen and target-lawful, followed by the
complete-arena exchange receipt.

The numerical conditioning is explicit:

\[
\boxed{\kappa_{\rm pair}=|1-\rho_{BC}|^{-1}.}
\]

The two large CP and pairing actions cancel to the desired residual, so their
amplitude-weighted rounding/model errors add and must remain below
`min(3.80e-6,eta_star)`.  Near-collinear `B,C` close the rank-two branch even
though the algebraic identity remains true.

**Graph-ready node `P2R0G19B_H4_CP4_Q211_TWO_RAY_BIRTH`:** constructive exact
source repair modulo the literal Star4+pairing gauge.  Promote on: exact
same-seam cancellation of every leakage term; the covariance-orthogonality
`C_AB=C_AC=0` used by the two-ray theorem (or an explicit generalized
orthogonalization/cancellation proof); lawful frozen `A,B,C` and source
amplitude; `2R` retained CP columns for residual source rank `R`; acceptable
`kappa_pair` and f32 endpoint error; positive complete-arena Schur gain; and no
unbilled transport.  Use the six-ray sibling if pairwise cancellation is not
literal.  Close on high residual rank/cap displacement, failed gauge
membership, cancellation conditioning, or task-dead response.

There is also a sharp uniform limitation.  The external 211 quotient has

\[
D_{211}=n\binom{n-1}{2}
\]

independent coordinates, while one projective CP ray plus its scale has at
most `n` continuous parameters.  Therefore a generic exact 211 tensor needs

\[
\boxed{
R_{\rm CP,generic}
\ge\left\lceil\frac{D_{211}}n\right\rceil
=\binom{n-1}{2}.
}
\]

At `n=1024` this is `522753` rays, versus the literal CP4 cap `2048`.  Hence
the two-ray theorem is a candidate-specific low-rank repair, not a uniform
closure of the missing sector.  Any board claim must prove concentration of
the actual old-residualized source/costate in the retained ray family; generic
dimension counts provide no such value receipt.

### Constructive bypass: low-rank reverse AABC action from the NCQC core

The acquisition gap does not force a transported fourth-order state.  Let the
resident quadratic-core map be

\[
\mathcal A(a)=\sum_i a_iA_i,
\]

and abbreviate `A=A(a), B=A(b), C=A(c)`.  The exact missing mixed action is

\[
\boxed{
q_{211}(a,a,b,c)
=32\operatorname{tr}(A^2BC)+16\operatorname{tr}(ABAC).
}
\]

It needs only two latent matrix products.  Put `P=AB` and `Q=AC`.  Since all
three cores are symmetric,

\[
\operatorname{tr}(A^2BC)=\langle P,Q\rangle_F,
\qquad
\operatorname{tr}(ABAC)=\operatorname{tr}(PQ),
\]

so

\[
\boxed{q_{211}(a,a,b,c)=32\langle P,Q\rangle_F+16\operatorname{tr}(PQ).}
\]

This is an exact target-conditioned source action, not a reconstruction of the
full quartic tensor.  Equivalently, if `K(u,v)=q4(u,u,u,v)`, polarization gives

\[
\boxed{
q4(a,a,b,c)
=\frac{K(a+b,c)-K(a-b,c)-2K(b,c)}6.
}
\]

If only diagonal quartic queries `p(u)=q4(u,u,u,u)` are exposed, an exact
six-query identity is

\[
\boxed{
\begin{aligned}
48q4(a,a,b,c)={}&p(a+b+c)-p(a+b-c)\\
&-p(a-b+c)+p(a-b-c)\\
&-2p(b+c)+2p(b-c).
\end{aligned}
}
\]

The first two-product formula is the preferred implementation; the
polarizations prove that the action is obtainable from lawful mixed marginal
queries and fix all combinatorial normalizations.

Let the old-residualized Bellman covector have an anchor-square decomposition

\[
\Gamma_{211}^{\rm res}
=\sum_{s=1}^k\gamma_s
\operatorname{Sym}(a_s^{\otimes2}\otimes b_s\otimes c_s).
\]

Then its complete declared scalar source action is the sum of the displayed
`k` core contractions.  The relevant rank is this target-conditioned
anchor-square rank, not the dimension of `Sym^4(R^n)` and not the CP4 tape
rank.

With a packed symmetric core of width
`d=r(r+1)/2`, three core mixtures and two `r by r` products per component cost

\[
\boxed{
C_{211}(k,L)
=kL\left(\frac{3d}{n^2}+\frac{2r^3}{n^3}
+c_{\rm pull}+c_{\rm hook}\right).
}
\]

At `r=55,n=1024,L=10`, the resident-core arithmetic before pullbacks/hooks is
about

\[
0.0472k
\]

f32 GEMM equivalents.  If three coefficient vectors must each be pulled
through one dense width map at every layer, add `3L/n`, giving about `0.0765k`.
Thus even a one-equivalent residual budget permits roughly `21` resident
contexts, or `13` with all three vector pullbacks, before non-GEMM charges.
These are ceilings, not billing receipts; shared anchors/directions lower the
actual count, while reconstructing the NCQC core must be charged if it is not
already resident.

The exact multilinear perturbation envelope is also simple.  In Frobenius
norm,

\[
|q4(A,A,B,C)|\le48\|A\|^2\|B\|\|C\|,
\]

and source errors bounded by `epsilon_A,epsilon_B,epsilon_C` obey

\[
\boxed{
|\delta q|
\le48\left[
(\|A\|+\epsilon_A)^2
(\|B\|+\epsilon_B)(\|C\|+\epsilon_C)
-\|A\|^2\|B\|\|C\|
\right]
}
\]

after reading the first product as multiplication of all three displayed norm
factors.  Across components/layers, amplitude bounds add before comparison
with `min(3.80e-6,eta_star)`.

**Graph-ready node `P2R0G19B_H4_Q211_REVERSE_AABC_ACTION`:** constructive and
cost-conditional.  Promote when the literal old-residualized reverse covector
has a frozen anchor-square decomposition within the audited `k` ceiling, its
directions and resident NCQC core have lawful provenance, the exact source sum
has robust Schur gain at least `2.86e-10`, and total action/model error passes
the endpoint bound.  The Vandermonde theorem supplies source novelty but does
not by itself prove overlap with the actual reverse covector.  Close on high
endpoint anchor rank, nonresident-core reconstruction cost, zero post-old-span
action, or failed Schur/error gates.

#### Actual `U Theta_i U^T` reverse form and the anchor-rank obstruction

For the literal reverse factorization

\[
\Lambda_i=U\Theta_iU^T,
\qquad
N_a=\sum_jU_{ja}A_j,
\qquad
P_{ia}=A_iN_a,
\]

the exact complete action is

\[
\boxed{
S_{211}
=\sum_{i,a,b}(\Theta_i)_{ab}
\left[
32\langle P_{ia},P_{ib}\rangle_F
+16\langle P_{ia}^T,P_{ib}\rangle_F
\right].
}
\]

Thus a resident ordered-product bank `P_ia` turns the missing provider into a
pure reduction; no new latent matrix multiplication is needed.  Define

\[
(F_i)_{ab}=\langle P_{ia},P_{ib}\rangle_F,
\qquad
(R_i)_{ab}=\langle P_{ia}^T,P_{ib}\rangle_F.
\]

Then `S211=sum_i <Theta_i,32F_i+16R_i>`.  A rank factorization of each
`Theta_i` evaluates the same expression through norms and traces of mixtures
`sum_a l_aP_ia`, avoiding explicit `k by k` Grams.  The generic dense
reduction price is `O(n k^2 r^2)` per layer; diagonal or rank-`s` costates
reduce it to `O(n k s r^2)`.  Creation, storage, and transport of `P` are not
free unless the literal old reverse lane already owns them.

Counting both symmetric contractions only for `a<=b`, the dense ten-layer
reduction has the explicit leading bill

\[
\boxed{
C_{P,\rm dense}
=\frac{Lk(k+1)r^2}{n^2}.
}
\]

At `r=55,n=1024,L=10` this is `0.02885 k(k+1)`: a one-equivalent residual
budget permits `k<=5`, while a hypothetical 21-equivalent replacement margin
permits `k<=26`, before hooks and memory traffic.  If every `Theta_i` is
diagonal in one common frozen basis, the bill is

\[
C_{P,\rm diag}=\frac{2Lkr^2}{n^2}=0.05770k,
\]

so one equivalent permits `k<=17`.  With a rank-`s` factorization per anchor,
forming each mixture and its two quadratic reductions gives the conservative

\[
C_{P,\rm rank-s}\lesssim
\frac{Ls(k+2)r^2}{n^2}
=0.02885s(k+2).
\]

These favorable counts apply only after the `P` bank is resident.  Building
all `P_ia=A_iN_a` from scratch costs

\[
\boxed{C_{P,\rm build}=\frac{Lknr^3}{n^3}
=1.5867k}
\]

at the same dimensions and immediately fails a one-equivalent additive margin
even for `k=1`.  Exact fusion with the already-required signed-H3 product bank
is therefore the principal economic loophole.

Crucially, factor width `k` does **not** bound the anchor-square rank used in
the preceding cheap-AABC ceiling.  For each `(a,b)`, the repeated-anchor
coefficient is

\[
D_{ab}=\operatorname{diag}_i(\Theta_i)_{ab}.
\]

Its representation as a sum of anchor squares has rank
`rank(D_ab)`, generically the number of live anchors (`n=1024`).  Replacing
the coordinate anchors by arbitrary mixed vectors creates unwanted
`ii'bc/(1111)` coordinates that must be canceled.  Therefore

\[
\boxed{
R_{\rm anchor}
=\operatorname{rank}_{\rm Veronese(2,1,1)}
(\Gamma_{211}^{\rm res}\bmod\text{ old/1111})
}
\]

must be certified independently.  The `0.0472 R_anchor` action route is lawful
only when this rank is at most its billed ceiling; generic `Theta_i` does not
pass merely because `U` is narrow.  Otherwise the resident-`P` contraction is
the constructive branch, conditional on its actual costate rank and reuse.

**Graph split:** keep `H4_Q211_REVERSE_AABC_ACTION` for genuinely low
`R_anchor`.  Add sibling `H4_Q211_RESIDENT_P_ACTION`, promoted by literal
resident-`P` ownership, a reduction/fusion cost within budget, exact
old/1111 subtraction, and the same Schur/error gates.  Close the low-anchor
branch on generic full-rank `D_ab`; do not close the resident-product branch
from that obstruction.

There is one exact quotient loophole to the full anchor-rank obstruction.  For
any signed anchor weights `h_i`, define

\[
(a_+)_i=\sqrt{(h_i)_+},\qquad
(a_-)_i=\sqrt{(-h_i)_+}.
\]

Modulo tensors whose two anchor legs are distinct,

\[
\boxed{
\sum_i h_i e_i^{\otimes2}
\equiv a_+^{\otimes2}-a_-^{\otimes2}.
}
\]

Therefore, modulo the all-distinct/1111 anchor-cross sector,

\[
\sum_i h_i e_i^{\otimes2}\odot b\odot c
\equiv
a_+^{\otimes2}\odot b\odot c
-a_-^{\otimes2}\odot b\odot c.
\]

Combining this signed-square identity with the two-ray Star4+pairing quotient
uses at most four CP rays per singleton pair; the unconditional six-ray
polarization uses at most twelve.  For a common singleton factor width `k`,
there are `k(k+1)/2` symmetric pairs, giving the cap bounds

\[
\boxed{
R_{\rm CP}\le2k(k+1)\quad\text{(pairing quotient)},\qquad
R_{\rm CP}\le6k(k+1)\quad\text{(six-ray form)}.
}
\]

At cap 2048 these allow respectively `k<=31` or `k<=17`; if every
anchor mask is nonnegative, the counts halve and allow `k<=44` or
`k<=25`.

The quotient condition is binding.  The discarded off-diagonal entries
`a_i a_j` generate genuine 1111 (and collision-degenerate) quartic sources
after transport.  Unless the complete future costate annihilates that sector
or the old engine owns it exactly, this construction is not an exact provider;
canceling those entries restores the ordinary `rank(diag h)` obstruction.
Graph sibling `H4_Q211_ANCHOR_SQRT_QUOTIENT` is conditional on a literal
old/dark 1111 receipt and the same CP-cap exchange, conditioning, and Schur
gates.

The same identity has an unbiased finite-cap relaxation when the 1111 quotient
is not exactly dark.  For `h=h_+-h_-` and independent preregistered
Rademacher signs `epsilon_s^+,epsilon_s^-`, define

\[
a_s^\pm=\epsilon_s^\pm\odot\sqrt{h_\pm},
\qquad
\widehat D_m=\frac1m\sum_{s=1}^m
\left(a_s^+(a_s^+)^T-a_s^-(a_s^-)^T\right).
\]

For every frozen seed, `diag(Dhat_m)=h` exactly, so the desired repeated-anchor
211 source has no sampling error.  Only the cross-anchor/1111 sector is noisy,
and

\[
\mathbb E\widehat D_m=\operatorname{diag}h.
\]

For any fixed symmetric endpoint mask `H`, the exact scalar variance is

\[
\boxed{
\operatorname{Var}\langle H,\widehat D_m-\operatorname{diag}h\rangle
=\frac2m\left[
\|\operatorname{off}(D_{\sqrt{h_+}}HD_{\sqrt{h_+}})\|_F^2
+\|\operatorname{off}(D_{\sqrt{h_-}}HD_{\sqrt{h_-}})\|_F^2
\right].
}
\]

The corpus/value-metric covariance is obtained by replacing the squared norms
with the corresponding cross-Frobenius Gram of the endpoint masks.  This is an
expectation over seed, not a frozen receipt; promotion requires the actual
endpoint action error below `min(3.80e-6,eta_star)` or its deterministic
restricted cross-Gram certificate.

If `A_{1111,V}` denotes the corpus/value-weighted sum of the two
off-diagonal mask energies in the variance formula, then

\[
\mathbb E\|e\|_V^2=\frac{2A_{1111,V}}m.
\]

Markov plus the exact Schur radius gives the explicit preregistration gate

\[
\boxed{
\frac{2A_{1111,V}}{\delta\,\eta_\star^2}
\le m\le\frac{512}{R}.
}
\]

It is only a seed-selection existence/sizing certificate; the chosen frozen
seed still needs the deterministic endpoint receipt.

Using the exact two-ray Star4+pairing birth for each signed square costs `4m`
CP columns per signed anchor mask (`2m` when `h>=0`).  Thus one mask fits
at most `m=512` under cap 2048 before displacement, while `R` simultaneous
masks require `4mR<=2048`.  The gauge-independent six-ray form costs
`12mR`.  Graph sibling `H4_Q211_SIGN_AVERAGED_BIRTH` is an approximate
candidate promoted only by frozen-seed 1111 error, CP-cap exchange,
pair-conditioning, and complete-arena Schur gates.  Unbiasedness alone is not
evidence of raw-MSE value.

**Later correction (Checkpoint 11):** the preceding counts describe the
restricted two-ray quotient and the naive per-square six-ray fallback.  The
globally lawful aggregate four-ray bundle cancels its Star leakage between
the positive and negative square averages, or with one two-ray control for a
one-sign mask.  Its exact grouped costs are `4(m_++m_-)` or `4m+2`, and those
later counts supersede this paragraph for deployment.

#### Conditional skew-Gram quotient after exact parallel ownership

The resident-product action has an even cheaper noncommutative quotient, but
only if the old engine literally owns the full parallel word.  Put

\[
K_{ia}=\operatorname{skew}P_{ia}
=\frac12[A_i,N_a].
\]

The two contractions satisfy

\[
F_{i,ab}-R_{i,ab}=2\langle K_{ia},K_{ib}\rangle_F,
\]

and therefore

\[
\boxed{
q_{211}=48F-8\langle[A_i,N_a],[A_i,N_b]\rangle_F
=48F-32\langle K_{ia},K_{ib}\rangle_F.
}
\]

If, and only if, the `48F` action has been acquired and subtracted at this
same seam, the novel scalar is the skew Gram

\[
\boxed{
S_{211}^{\rm skew}
=-32\sum_{i,a,b}(\Theta_i)_{ab}
\langle K_{ia},K_{ib}\rangle_F.
}
\]

This has a useful sign receipt: it is nonpositive for every
`Theta_i>=0`; signed costates retain no global sign guarantee.  With a resident
skew bank of dimension `r(r-1)/2`, its dense symmetric reduction costs

\[
\boxed{
C_{K,\rm dense}
=\frac{Lk(k+1)r(r-1)}{4n^2}
=0.007081\,k(k+1)
}
\]

at `r=55,n=1024,L=10`.  One equivalent therefore permits `k<=11`.  A common
diagonal costate basis costs

\[
C_{K,\rm diag}
=\frac{Lkr(r-1)}{2n^2}
=0.01416k,
\]

permitting `k<=70` before hooks.  Rank-`s` costates have the conservative
mixture bound `0.01416 s(k+1)`.

This quotient must not be invoked merely because a formula calls `48F`
"parallel."  The Vandermonde tangent proves that the literal CP4 reset has no
all-distinct acquisition port; if no other old path observes `48F`, both terms
are novel and the full resident-`P` action above is binding.  Graph sibling
`H4_Q211_SKEW_GRAM_ACTION` is therefore conditional on an explicit old-parallel
Jacobian equality and resident skew-bank reuse.

## 2026-08-30 — mixed-frame observability and the three distinct q4 objects

A lawful dense frame `W=(w_1,...,w_n)` exposes

\[
(D_W^3T)_{ab}=T(w_a,w_a,w_b),
\qquad
(D_W^{22}S)_{ab}=S(w_a,w_a,w_b,w_b).
\]

If the `C,K21` inverse fiber is finite, one additional generic dense frame that
records both cubic and quartic planes separates all target-distinct candidates
with probability one: each bad equality is a proper algebraic variety, and
there are finitely many pairs.  Zero extra frames fails by the cycle examples.
A cubic-only frame cannot see a pure four-cycle ambiguity whose entire cubic
tensor is identical; at least one lawful quartic mixed frame is necessary.

This finite-fiber statement does **not** extend to a generic unrestricted
tensor.  An arbitrary symmetric cubic needs at least `172` total `K21` frames
at `n=1024` by dimension.  The `211` quartic sector has

\[
n\binom{n-1}{2}=535299072
\]

coordinates, requiring at least `511` full `K31` frames or `1021` symmetric
`K22` frames by dimension.  A constant-frame result is possible only because
of the NCQC/core provenance manifold.

Trajectory frames count as repeated observations of one core only while the
core is transported linearly.  A nonlinear reset changes the latent object and
cannot be invoked as free polarization evidence without jointly including its
birth law.

There are three q4 notions which must not be conflated:

1. `q4_z(211)`: the `(211)` equality partition of `U_M` in latent Gaussian
   coordinates;
2. `q4_ext(211)=kappa_4(Q_i,Q_i,Q_j,Q_k)`: the externally collided fourth
   cumulant after dense mixing;
3. `q4_graph`: collision sectors internal to the graph/traffic expansion.

The full `U_M` contains `(4),(31),(22),(211),(1111)` sectors.  Its latent
`(211)` coefficient is, for a polarized lift,

\[
u^{211}_{M,N;i;jk}
=\frac12(M_{ii}N_{jk}+N_{ii}M_{jk})
+M_{ij}N_{ik}+M_{ik}N_{ij},
\]

and for a square it is `M_ii M_jk+2 M_ij M_ik`.  A latent partition is not by
itself the deploy-side external cumulant.

### Canonical NCQC residual

Let `V=span{A_i}`, let `P` be the trace-inner-product projector onto `V`, and
write

\[
Y_{ij}=A_i\circ A_j,\qquad
R_{ij}=(I-P)Y_{ij}.
\]

The old-span component `P Y_ij` is `q_parallel`; the genuinely new source is
the residual Jordan map

\[
\mathcal R:\operatorname{Sym}^2(V)\longrightarrow V^\perp,
\qquad e_i\odot e_j\mapsto R_{ij}.
\]

Its canonical symmetric fourth tensor is

\[
\boxed{
Q_{\perp,ijkl}=16\big[
\langle R_{ij},R_{kl}\rangle+
\langle R_{ik},R_{jl}\rangle+
\langle R_{il},R_{jk}\rangle
\big].
}
\]

In the external `211` collision this becomes exactly

\[
\boxed{
q_{\perp,iijk}=16\big[
\langle R_{ii},R_{jk}\rangle+
2\langle R_{ij},R_{ik}\rangle
\big].
}
\]

It is a symmetrized Gram tensor:

\[
Q_\perp(x,x,x,x)=48\|R(x,x)\|_F^2\ge0,
\]

although individual `211` entries can have either sign.  Under an invertible
external frame, `V` itself is unchanged and

\[
R'_{ab}=\sum_{ij}W_{ia}W_{jb}R_{ij};
\]

therefore `Q_perp` transforms as a genuine symmetric fourth tensor.

After residualizing against an old provider bank `B`, its exact conditional
gain is governed by the Schur complement

\[
G_{\perp\mid B}=G_{\perp\perp}-G_{\perp B}G_B^+G_{B\perp},
\]

\[
c_{\perp\mid B}=c_\perp-G_{\perp B}G_B^+c_B,
\qquad
\Delta\mathrm{MSE}=c_{\perp\mid B}^T
G_{\perp\mid B}^+c_{\perp\mid B}.
\]

This external residual, after graph collision and old-span subtraction, is the
only square-lift part whose incremental value should be counted.

## 2026-08-30 — finite-fiber conditioning theorem

Let the target-distinct candidate tensors be `Q_1,...,Q_N`, normalized in the
task norm `||.||_V`.  Define their minimum projective separation

\[
\delta_{\rm proj}
=\min_{s\ne t}\sqrt{1-\langle u_s,u_t\rangle_V^2},
\qquad u_s=Q_s/\|Q_s\|_V,
\]

and for a frame observation `E_W Q=(Q(w_a,w_a,w_b,w_b))_{ab}` define

\[
\boxed{
\sigma_{\rm proj}(W)=
\min_{s\ne t}
\frac{\|(I-\Pi_{E_WQ_t})E_WQ_s\|_Y}
{\|(I-\Pi_{Q_t})Q_s\|_V}.
}
\]

If `q_min=min_s ||Q_s||_V`, the measured projective separation is at least

\[
\sigma_{\rm proj}(W)q_{\min}\delta_{\rm proj}.
\]

With observation noise bounded by `eta`, finite-candidate recovery is certified
whenever

\[
\boxed{2\eta<\sigma_{\rm proj}q_{\min}\delta_{\rm proj}.}
\]

For a continuous candidate set, a restricted singular value `sigma` gives

\[
\|\widehat Q-Q\|_V\le\frac{2\eta}{\sigma},
\qquad
\|\Delta\Xi\|\le\frac{2L_V\eta}{\sigma}.
\]

The strict equal-fidelity `V+k3+k4` gate is

\[
\frac{2L_V\eta}{\sigma\|z\|}\le0.18113,
\]

whereas the global `0.925` correlation gate permits approximately `0.379` in
the same relative-error normalization.  In raw-MSE units the currently relevant
slack condition is

\[
\left(\frac{2L_V\eta}{\sigma}\right)^2<7.453\times10^{-10}.
\]

### Why generic uniqueness supplies no usable lower bound

No positive lower bound on `sigma_proj` follows only from projective candidate
separation, density of `W`, or `cond(W)`.  Take a cycle secant `Delta` blind in
one frame and `W_epsilon=exp(epsilon H)W_0`.  The frames can remain dense and
orthogonal, hence condition number one, while

\[
\|E_{W_\epsilon}\Delta\|=O(\epsilon^m)\to0
\]

for the first nonzero derivative order `m`.  Generic almost-sure uniqueness is
therefore a zero-versus-nonzero statement, not a conditioning certificate.

The mathematically sharp candidate-specific design is

\[
W_*\in\arg\max_{W\in\mathcal W_\kappa}\sigma_{\rm proj}(W).
\]

Compactness gives a maximizer; a finite target-distinct fiber gives a strictly
positive optimum because the bad frames form a finite union of proper
algebraic sets.  The value of that optimum must still be certified on the
actual secants.

A dimension-only probabilistic estimate can be no stronger than roughly

\[
\sigma\gtrsim c_{FV}n^{-2}(2N)^{-4/n}
\]

under an additional value/Frobenius norm comparability assumption.  At this
width it is far too weak for promotion and must not be used as evidence.

## 2026-08-30 — candidate-specific max–min frame theorem

### Exact plane-projection normalization

For a `G`-orthonormal frame `W=(w_a)`, define

\[
\mathcal S_W=\operatorname{span}\left\{
\operatorname{sym}(w_a^{\otimes2}\otimes w_b^{\otimes2}):a\le b
\right\}\subset\operatorname{Sym}^4(V).
\]

For `a<b`, the normalized basis vector is
`sqrt(6) sym(w_a^2 tensor w_b^2)`.  Hence

\[
\|\Pi_{\mathcal S_W}D\|_F^2
=\sum_aD_{aaaa}^2+6\sum_{a<b}D_{aabb}^2,
\]

whereas the full symmetric matrix observation has

\[
\|E_WD\|_F^2
=\sum_aD_{aaaa}^2+2\sum_{a<b}D_{aabb}^2.
\]

Therefore the exact norm sandwich is

\[
\boxed{
\frac13\|\Pi_{\mathcal S_W}D\|_F^2
\le\|E_WD\|_F^2
\le\|\Pi_{\mathcal S_W}D\|_F^2.
}
\]

The factor `1/3` is structural, not a heuristic.

Let `Sec_perp` be the normalized target-distinct secants of the NCQC
`Q_perp` fiber.  Define

\[
\mu_W=\inf_{D\in\mathrm{Sec}_\perp}
\frac{\|\Pi_{\mathcal S_W}D\|_{F,G}^2}{\|D\|_V^2}.
\]

Then

\[
\boxed{\sigma(W)\ge\sqrt{\mu_W/3}.}
\]

For `F` frames, with optional linearized transport `T_f` and noise metric
`N_f`, the exact fusion/observability Gram is

\[
\boxed{
\Gamma_F=\sum_{f=1}^F
T_f^*E_{W_f}^*N_f^{-1}E_{W_f}T_f.
}
\]

Its restricted secant eigenvalue

\[
\mu_F=\inf_{D\in\mathrm{Sec}_\perp}
\frac{\langle D,\Gamma_FD\rangle}{\|D\|_V^2}
\]

is the sharp algebraic promotion certificate.  With the unweighted projection
Gram `sum_f Pi_{S_Wf}`, the preceding normalization yields
`sigma_stack >= sqrt(mu_F/3)`.  No surrogate such as frame density or ordinary
matrix condition number can replace this restricted eigenvalue.

For a nonlinear finite candidate trajectory the exact, assumption-free version
is pairwise:

\[
\sigma_{10}=min_{\theta\ne\theta'}
\frac{\left[\sum_{t=1}^{10}
\|N_t^{-1/2}{E_tQ_t(\theta)-E_tQ_t(\theta')\}\|^2\right]^{1/2}}
{\|\Xi(\theta)-\Xi(\theta')\|_V}.
\]

This formulation correctly includes reset births; replacing `Q_t` by a linear
transport is allowed only after proving that closure.

### Why the actual NCQC fiber is the loophole

One symmetric `K22` frame supplies

\[
m=n(n+1)/2=524800
\]

coordinates.  Under maximal Jacobian rank, the `r=45,...,54` inverse fibers are
zero-dimensional.  At `r=55`, the parameter-count excess is only `2099`, so a
single generic genuine rotation has far more than the `2d+1` coordinates needed
to embed a `d<=2099` smooth compact fiber under the usual transversality
hypothesis.  Thus one frame is dimensionally sufficient for the actual
`r45–55` NCQC candidate family even though it is hopeless for an unrestricted
quartic tensor.  This conclusion is conditional on the Jacobian/transversality
gate; it is not inferred from count alone.

For a linear secant subspace of dimension `p_perp`, a necessary stacked-rank
gate is

\[
\boxed{p_\perp\le F n(n+1)/2.}
\]

Ten frames allow `5248000` measured dimensions, comfortably above the roughly
million-parameter raw core scale.  More importantly, ten distinct late suffix
frames provide a candidate-specific fusion Gram whose restricted minimum
eigenvalue can be positive even when individual frames have blind directions.

### Structured-frame construction and its exact limitation

Whiten coordinates first.  At `n=1024`, a tempting deterministic frame is

\[
W_H=G^{-1/2}H_{1024}/\sqrt n.
\]

It is lawful and `G`-orthonormal.  **A family obtained only by left row-sign
modulations `G^{-1/2}D_fH/sqrt(n)` is not universal.**  Row signs cancel on the
`(4)` and `(22)` equality sectors, so a `(22)` direction blind to the base
Hadamard plane remains blind for every sign modulation.  On the all-distinct
sector, Sylvester characters also alias monomials with equal XOR labels.  This
corrects the earlier tentative signed-Hadamard proposal.

A valid deterministic family must use genuinely different rotations, for
example

\[
W_f=G^{-1/2}R_fH/\sqrt n,
\]

where the `R_f` are not merely signed diagonal symmetries.  Permutations not in
the Hadamard automorphism group, sparse Givens networks, or already-present
dense trajectory suffixes are lawful choices.  Their acceptance criterion is
still the exact `mu_F` gate above.

There is a constructive algebraic selection theorem for a finite fiber.  Use
the Cayley family

\[
R(S)=(I-S)(I+S)^{-1},\qquad S^T=-S.
\]

For every nonzero target secant `D`, the equality `E_{R(S)W_0}D=0` is a proper
algebraic condition in `S` provided the full orthogonal orbit is not blind.
The complement of the finite union of these varieties is open and dense and
contains rational points.  Therefore the lexicographically first rational
skew matrix (in any fixed height enumeration) outside that union is a fully
deterministic separating frame.  This proves constructibility without claiming
a universal closed-form condition number.

### Ten-frame Krylov sufficient condition

For a more quantitative structured certificate, choose a skew generator `S`,
`W(t)=G^{-1/2}e^{tS}W_0`, and let `L_S` be its induced Lie derivative on
`Sym^4(V)`.  Put `B=E_{W_0}` and define the order-nine Krylov map

\[
\mathcal K_{S,9}D=
\left(BD,\frac{BL_SD}{1!},\ldots,
\frac{BL_S^9D}{9!}\right).
\]

If

\[
\beta=inf_{D\in\mathrm{Sec}_\perp}
\frac{\|\mathcal K_{S,9}D\|}{\|D\|_V}>0,
\]

then ten distinct nodes `t_f=rho xi_f` give a deterministic ten-frame
certificate.  If `V_xi=(xi_f^j)_{f,j=0}^9`, `ell=||L_S||`, and `rho<=1`, Taylor's
theorem gives the explicit lower bound

\[
\boxed{
\sigma_{10}\ge
\sigma_{\min}(V_\xi)\rho^9\beta
-\sqrt{10}\,e^{\rho\ell}
\frac{(\rho\ell)^{10}}{10!}\,c_{FV}.
}
\]

Any positive right-hand side is a purely algebraic conditioning certificate.
Chebyshev-like distinct nodes improve `sigma_min(V_xi)`, but the bound remains
candidate-specific.  This is the sharp replacement for the impossible claim
that ten arbitrary dense frames must work.

### Existing ten-transition second stream

The preferred construction is to reuse the ten actual late-layer suffixes.  If
`P_t` is the proven linear pullback of the shared core, define

\[
\Gamma_{10}=\sum_{t=1}^{10}
\mathcal E_{P_t}^*\mathcal N_t^{-1}\mathcal E_{P_t}.
\]

The square/q4 lane is identifiable and stably valuable exactly when

\[
\lambda_{\min}(\Gamma_{10}\mid\mathrm{Sec}_\perp)>0
\]

and the resulting value-error bound passes the `0.18113` fidelity inequality.
If the suffixes include nonlinear resets, use the pairwise trajectory definition
of `sigma_10` instead.

No universal lower bound is possible for a fixed collection of fewer than
enough planes: an adversarial secant subspace can be placed in their common
kernel whenever its codimension permits.  The restricted fusion eigenvalue is
therefore both the sufficient condition and, up to the chosen task norm, the
necessary one.

### Deployment-work gate

The persistent forward `H4` bank needs approximately `45` basis transports plus
the existing `9`-GEMM core, totaling at least `54` f32-GEMM equivalents; it is
over the absolute `49` cap before births and is closed as a direct forward
provider.

For a fused residual second stream across ten late transitions, the exact
budget inequality is

\[
\boxed{9+10g_\perp+F g_{\rm frame}\le49.}
\]

If `g_perp=4`, there is no room for a synthetic frame and the ten existing
suffix observations must supply conditioning for free.  If `g_perp=3`, up to
ten one-GEMM genuine frame transforms fit exactly: `9+30+10=49`.  Sign-only
Hadamard modulations do not qualify; genuine rotations or already billed suffix
maps do.

## Current graph-ready candidate and terminal gates

**Candidate:** a shared NCQC quadratic core with an old-span `q_parallel`
stream plus one fused late `q_perp` second stream.  Reconstruct only the
target-relevant residual Jordan Gram, not an unrestricted `H4` tensor.  Observe
it through the ten actual suffix frames, or through at most ten genuine
one-GEMM rotations if `g_perp=3`.

**Promote only if all of the following hold algebraically:**

1. `G,K21` pass PSD, support, rank, and moment-block feasibility.
2. The inverse Jacobian satisfies `ker DO subset ker D Xi`, or the finite fiber
   is explicitly separated by the mixed quartic observation.
3. The residual tensor is exactly the external `Q_perp` above after old-span
   subtraction; latent and graph collision partitions are not substituted.
4. The ten-frame restricted fusion eigenvalue is positive and its certified
   value error obeys `2 L_V eta/(sigma ||z||) <= 0.18113`.
5. Total incremental deployment work obeys `9+10 g_perp+F g_frame <=49`.
6. Any nonlinear reset birth is included in the trajectory secant; no linear
   transport is assumed across an unproved reset.

**Close immediately if:** the relevant q-perp secant lies in the common suffix
kernel; `g_perp=4` and the existing suffixes fail conditioning; the residual
Jordan module grows beyond the bounded cyclic/Jordan state; or the certified
value error misses the fidelity gate.

## Current next question

Derive the actual ten late-layer pullback operators on the shared NCQC residual
module symbolically and decide whether their fusion Gram is coercive on the
`q_perp` target secants.  The highest-value theorem is a provenance-aware proof
that the suffix family generates a cyclic module of dimension at most the
`r45–55` inverse fiber and has a positive restricted minimum eigenvalue.  If
that cannot be shown, derive a three- or ten-Givens deterministic frame family
whose order-nine Krylov constant is positive while retaining the `49`-GEMM cap.

## Cross-lane rank refinement: full plane versus deploy-lawful context

The `524800` coordinates in one complete offline `K22` frame must not be
confused with the rank of one deploy-lawful target-conditioned context.  At
`r=55`, the regular quotient parameter count is

\[
p=1575475,
\qquad q_{C,K21}=1573376,
\qquad f=p-q=2099.
\]

Let `F` be this regular inverse-fiber tangent, let
`f_v=rank(D(H3,q4_{211})|_F)` be its value-relevant rank, and let the endpoint
adjoint retained by one lawful context have dimension `k`.  Because the
centered external direction has only `1023` independent coordinates, one
context has rank at most

\[
b=1023k.
\]

Therefore a necessary context-count bound is

\[
\boxed{m\ge\left\lceil\frac{f_v}{1023k}\right\rceil.}
\]

In the worst case `f_v=2099`, this gives `m>=3,2,1` for `k=1,2,>=3`.  Under a
flat-energy model, the ranks needed for the two correlation thresholds are

\[
r_{0.9253}=1798,
\qquad
r_{0.98346}=2031.
\]

Thus a scalar (`k=1`) endpoint needs at least two contexts even for the weaker
gate; `k>=2` can be dimensionally sufficient with one.  If a context costs one
GEMM per late layer, the `49` cap permits at most four such contexts after the
nine-GEMM base; if it costs two, at most two.  These are necessary rank counts,
not sufficiency claims.

The exact target-conditioned conditioning constant is

\[
\tau_\Xi
=\inf_{h\in F,\ \|D\Xi h\|=1}\|\mathcal M_W h\|.
\]

Pointwise identifiability requires `tau_Xi>0`, equivalently a positive
restricted minor of `M_W^*M_W` on the value-relevant fiber.  A generic rank
claim requires an explicit core `A_*`, explicit contexts, and one nonzero
polynomial minor.  Counts alone do not supply this witness.  Moreover the
regular core locus is open/noncompact and accumulates on its discriminant,
where the minor vanishes; consequently the uniform infimum of `tau_Xi` over all
regular cores is zero.  There is no uniform `0.9253` or `0.98346` guarantee from
`C,K21` plus generic trajectory observations.  The exact promotion inequalities
are

\[
\frac{\text{observation/model mismatch}}
{\tau_\Xi\,\|\text{task signal}\|}\le0.3792
\]

for the global threshold and the same ratio at most `0.1811` for the strict
equal-fidelity threshold.

## Exact target-context contraction of q-perp

The full mixed frame is unnecessary when the downstream target arrives through
a small number of rank-one contexts.  For `u in V`, define

\[
R(u,u)=\sum_{jk}u_ju_kR_{jk},
\qquad
R(e_i,u)=\sum_j u_jR_{ij}.
\]

Contracting the external `211` tensor in its two free downstream slots gives

\[
\boxed{
[\mathcal M_uQ_\perp]_i
=Q_\perp(e_i,e_i,u,u)
=16\left[
\langle R_{ii},R(u,u)\rangle
+2\|R(e_i,u)\|_F^2
\right].
}
\]

The lawful deployed observation is its centered output
`P_0 M_u Q_perp`, with `P_0=I-11^T/n`, explaining the `n-1=1023` rank per
scalar context.

For a target secant `D`, define its symmetric `211` slices

\[
(D_i)_{jk}=D_{iijk},
\qquad
\bar D=\frac1n\sum_iD_i.
\]

For contexts `U={u_f}_{f=1}^m`, the observation energy is exactly

\[
\boxed{
\|\mathcal M_UD\|^2
=\sum_{i=1}^n\sum_{f=1}^m
\langle D_i-\bar D,u_fu_f^T\rangle_F^2.
}
\]

Equivalently, the context fusion frame on `Sym^2(V)` is

\[
H_U=\sum_{f=1}^m
\operatorname{vec}_s(u_fu_f^T)
\operatorname{vec}_s(u_fu_f^T)^T,
\]

and

\[
\|\mathcal M_UD\|^2
=\sum_i\langle D_i-\bar D,H_U(D_i-\bar D)\rangle.
\]

This is the exact candidate-specific max-min problem for a scalar endpoint.  It
also makes the limitation transparent: `rank(H_U)<=m`, so a constant number of
contexts cannot recover arbitrary slices.  It can recover the low-dimensional
NCQC inverse fiber if the restriction of this block frame to its target secants
is coercive.

For a finite target-distinct fiber, one generic context suffices algebraically.
Indeed, for every nonzero `211` secant there is an `i` with `D_i-bar D !=0`, and
`u^T(D_i-bar D)u` is a nonzero quadratic polynomial.  Its zero set is proper;
a finite union of such quadrics has measure zero.  This theorem applies to the
finite `r45–54` regular fiber.  It does not override the `r55` tangent rank
bound: a `2099`-dimensional continuous fiber requires at least three scalar
contexts for full identification, while two can cover the flat-energy strict
target rank `2031` because `2*1023=2046`.

For `u ~ N(0,I)`, the population second moment is exact:

\[
\boxed{
\mathbb E_u\|P_0\mathcal M_uD\|^2
=2\sum_i\|D_i-\bar D\|_F^2
+\sum_i\operatorname{tr}(D_i-\bar D)^2.
}
\]

Thus the isotropic population is coercive on every non-common-slice direction,
with lower constant `sqrt(2)` in the centered-slice Frobenius norm.  Its null is
exactly the common-slice sector `D_i=bar D`, which is already dark to a centered
endpoint.  This identity supplies a principled random-frame prior, but ten or
fewer empirical contexts still require the restricted `H_U` eigenvalue; an
expectation is not a promotion certificate.

A deterministic algebraic context can be generated by a Sidon moment curve.
For example `u_j(t)=t^{3^j}` gives a distinct exponent for every unordered pair
`(j,k)`, so `u(t)^TAu(t)` is not identically zero for any nonzero symmetric
`A`.  Avoiding the finitely many roots separates a finite fiber.  The powers are
catastrophically scaled at width 1024, so this is an existence proof only, not
a conditioned deployment design.  Genuine suffix contexts with a certified
restricted fusion Gram remain the useful construction.

### Cost-model correction

There are two alternative architectures and they must not be double-counted:

1. **Explicit q-perp state stream:**
   \[
   9+10g_\perp+F_{\rm one\mbox{-}shot}g_{\rm frame}\le49.
   \]
   A synthetic frame charged here is a one-time transform.  If it must be
   propagated through every late transition, it is a context stream instead.
2. **Target-conditioned context streams, no explicit q-perp state:**
   \[
   \boxed{9+10m c_{\rm ctx}\le49.}
   \]
   At one GEMM per context per transition, `m<=4`; at two, `m<=2`.

The second route exactly matches the rank theorem above.  At `r55`, three
one-GEMM scalar contexts are within the cap (`39` total including the base) and
can be dimensionally sufficient for the full `2099` fiber; two contexts fit the
strict flat-energy rank `2031`.  They are promoted only after a positive
target-restricted context Gram is exhibited.

### Pointwise generic sufficiency of three scalar contexts

There is a sharper pointwise theorem for a **fixed** contracted target tangent.
Let `T subset Sym^2_0(R^n)` have dimension `f`, and let one lawful orthogonal
basis `W` observe the centered diagonal

\[
\mathcal D_W(H)=\operatorname{diag}_0(W^THW).
\]

For every nonzero `H`, the blind-basis variety

\[
Z_H=\{W:\mathcal D_W(H)=0\}
\]

has codimension `n-1`.  Applying the incidence-dimension theorem to `m`
independent bases shows that generic bases are jointly injective on `T` whenever

\[
\boxed{m(n-1)\ge f.}
\]

Consequently, for the fixed `r55` regular tangent with `f=2099`, three scalar
contexts are not only necessary by count but generically pointwise sufficient
to produce a nonzero restricted minor.  Two are pointwise sufficient for the
flat-energy ranks `1798` and `2031` because `2*1023=2046`.

The quantifiers matter.  The tangent `T` changes along the unidentified core
fiber, so this does not select one uniformly conditioned basis family from
`C,K21`, and it supplies no positive lower bound on `tau_Xi`.  It proves only
that a witness exists at each fixed regular core.  Likewise, replacing the
capacity `1023` by `1023k` for endpoint dimension `k` is valid only when every
nonzero target direction has endpoint Schmidt rank `k` (or an equivalent
codimension statement has been proved).  Without that gate, the worst blind
direction may occupy a single endpoint mode and the guaranteed codimension
remains `1023`.

## 2026-08-30 — direct K31 action checkpoint

### Exact NCQC action formula

For the residual fourth tensor and two external directions `U,V`, symmetry and
the three pairings give

\[
\boxed{
\mathcal A_{U,V}(Q_\perp)
:=Q_\perp(U,U,U,V)
=48\langle R(U,U),R(U,V)\rangle.
}
\]

Equivalently, if `p(U)=Q_perp(U,U,U,U)=48||R(U,U)||^2`, then

\[
D p(U)[V]=4\mathcal A_{U,V}=192\langle R(U,U),R(U,V)\rangle.
\]

Thus one K31 action can be evaluated from only the two residual objects
`R(U,U)` and `R(U,V)`; materializing the full `211` tensor is unnecessary when
this scalar is exactly the Bellman quantity.

For a core-fiber tangent `h`, with residual perturbation `dot R`, and fixed
external masks, its differential is

\[
\boxed{
D\mathcal A_{U,V}[h]
=48\left[
\langle\dot R(U,U),R(U,V)\rangle
+\langle R(U,U),\dot R(U,V)\rangle
\right].
}
\]

When `U` and/or `V` themselves depend on the uncertain trajectory, the total
derivative must instead include

\[
(DQ[h])(U,U,U,V)
+3Q(\dot U,U,U,V)+Q(U,U,U,\dot V).
\]

Dropping these mask-motion terms is valid only when the masks are externally
fixed or their dependence has already been included in the reverse recursion.

The Frobenius norm of the normalized symmetric test tensor is exact.  With
`rho=<U,V>/(||U||||V||)`,

\[
\left\|\operatorname{sym}(U^{\otimes3}\otimes V)\right\|_F
=\frac{\|U\|^3\|V\|}{2}\sqrt{1+3\rho^2}.
\]

This controls raw scaling, but not its angle to the target-relevant fiber.

### Kernel on the r55 inverse fiber

Let

\[
F_{55}=\ker D(C,K21)/(\text{orthogonal gauge}),
\qquad \dim F_{55}=2099
\]

at a regular core.  For fixed masks define the action covector

\[
a_{U,V}=D\mathcal A_{U,V}|_{F_{55}}\in F_{55}^*.
\]

Its exact kernel is

\[
\boxed{
\ker a_{U,V}
=\left\{h\in F_{55}:
\langle\dot R_{UU},R_{UV}\rangle
+\langle R_{UU},\dot R_{UV}\rangle=0
\right\}.
}
\]

Unless this covector vanishes, one scalar action has rank one and a
`2098`-dimensional kernel.  For action pairs `(U_s,V_s)`, `s=1,...,m`, the
kernel is the intersection of these hyperplanes and has dimension at least
`2099-m`; generically it is exactly `2099-m` until saturation.  Therefore a
scalar K31 action does **not** identify the q-perp fiber and does not replace the
three earlier basis contexts for that purpose.  Full generic fiber recovery
would need at least `2099` independent scalar actions, versus three contexts
that each return `1023` centered coordinates.

The large kernel is harmless only if it is Bellman-dark.  If `Xi` denotes the
full desired correction and `A` stacks the observed actions, the exact
sufficiency condition is

\[
\boxed{
\ker(D A|_{F_{55}})\subseteq\ker(D\Xi|_{F_{55}}).
}
\]

Equivalently, there is a linear map `B` on the action range such that

\[
D\Xi=B\,D A.
\]

If `Xi` is literally the same scalar action, this factorization is the identity:
all `2098` dark directions are task-irrelevant and no core identification is
needed.  If the Bellman correction contains any independent quartic
contraction, the shortcut fails unless its covector lies in the span of the
chosen K31 action covectors on `F55`.

### Direct K31 action is not the generic 211 Bellman contraction

The previously derived diagonal-211 scalar for a forward direction `u` and
reverse mask `v` is

\[
\mathcal B_{u,v}(Q)
=\sum_i v_iQ(e_i,e_i,u,u)
=\left\langle Q,
\operatorname{sym}(D_v\otimes u^{\otimes2})\right\rangle,
\]

where `D_v=sum_i v_i e_i e_i^T`.  In contrast, one K31 action tests
`sym(U^3 tensor V)`.  Equality for every symmetric fourth tensor would require
the quartic-polynomial identity

\[
(x^TD_vx)(u^Tx)^2=(U^Tx)^3(V^Tx).
\]

By unique factorization this is exceptional: the diagonal quadratic must split
into the required linear factors and share the necessary factor with `u`.  For
a generic mask with `rank(D_v)>=3`, it is impossible.  Hence a direct K31 scalar
does not automatically equal the diagonal `211` Bellman correction.

On the restricted NCQC fiber the weaker condition is possible, but it must be
proved as the row-space gate

\[
D\mathcal B_{u,v}|_{F_{55}}
\in\operatorname{span}\left\{
D\mathcal A_{U_s,V_s}|_{F_{55}}:1\le s\le m
\right\}.
\]

The allowed projective one-action family has dimension at most
`2n-2=2046`, while `P(F55^*)` has dimension `2098`.  Thus a generic scalar
functional on the r55 fiber is not representable by one K31 action, although a
Bellman functional derived structurally from one `(U,V)` can of course be.

### Number of reverse masks

For fixed `U`, the K31 action is linear in `V`.  Define the action gradient

\[
g_j(U)=48\langle R(U,U),R(U,e_j)\rangle,
\qquad
\mathcal A_{U,V}=V^Tg(U).
\]

Therefore the number of required reverse streams is not three by default; it is
the dimension `k_eff` of the Bellman reverse-mask subspace after restriction to
the core fiber:

\[
k_{\rm eff}
=\operatorname{rank}\operatorname{span}\left\{
D\mathcal A_{U,V_a}|_{F_{55}}
\right\}.
\]

If a scalar Bellman adjoint supplies one known `V_l` at every layer, one reverse
mask stream yields the ten layer-local scalar actions and the three-context
identification burden disappears.  If there are `k_eff` independent endpoint
channels, exactly `k_eff` action streams are needed unless their restricted
covectors are dependent.  Under a one-GEMM-per-mask-per-layer implementation,
the work gate is

\[
9+10k_{\rm eff}\le49,
\]

so at most four independent masks fit.  At two GEMMs per mask per layer, at
most two fit.

This conclusion is conditional: if the full Bellman correction needs the
diagonal `211` vector, a K22 contraction, or another action outside this mask
span, one scalar reverse stream is insufficient and the context-identification
lane remains necessary.

### Exact restricted-minor and value-error gate

Let `J_A=D A|F55`, `J_B=D Xi|F55`, let `N` be the action-noise covariance or
chosen whitening metric, and choose columns of `Z` spanning
`F55/ker(J_B)`.  Define

\[
M_A=Z^TJ_A^TN^{-1}J_AZ,
\qquad
M_B=Z^TJ_B^TJ_BZ.
\]

The action route is locally sufficient and conditioned exactly when

\[
\boxed{
\tau_{A\to B}^2
=\lambda_{\min}(M_A,M_B)>0.
}
\]

Positivity is equivalent to the kernel inclusion above and to a nonzero
restricted minor of the whitened action Jacobian.  It also equals

\[
\tau_{A\to B}
=\inf_{h\in F55,\ \|J_Bh\|=1}
\|N^{-1/2}J_Ah\|.
\]

For finite, nonlinear candidates use the secant version

\[
\tau_{A\to B}^{\rm sec}
=\inf_{\theta\ne\theta'}
\frac{\|N^{-1/2}(A(\theta)-A(\theta'))\|}
{\|\Xi(\theta)-\Xi(\theta')\|}.
\]

With whitened action error at most `eta` and Bellman-factorization/model error
`epsilon_model`, candidate recovery gives

\[
\|\widehat\Xi-\Xi\|
\le\frac{2\eta}{\tau_{A\to B}}+\epsilon_{\rm model}.
\]

The strict promotion gate is therefore

\[
\boxed{
\frac{2\eta/\tau_{A\to B}+\epsilon_{\rm model}}
{\|\text{Bellman task signal}\|}\le0.18113,
}
\]

or `0.3792` for the weaker global-correlation gate.  When the correction is
directly the measured action in the same norm, `J_B=J_A`, `tau=1`, and no
inverse-fiber conditioning penalty remains; only action-evaluation error and
Bellman truncation error remain.

### Checkpoint disposition

**Promote the scalar-action pivot** only after symbolically proving that the
late Bellman correction factors through the exact actions
`Q_perp(U_l,U_l,U_l,V_l)` and counting the independent reverse-mask rank.  In
the scalar-adjoint case it genuinely replaces three q-perp identification
contexts with one reverse mask stream.

**Do not promote** merely because K31 is available.  Original-coordinate K31
planes do not bilinearly transform into `kappa4(U,U,U,V)`, and a generic
diagonal-211 correction is not one K31 action.  Failure of the row-space/kernel
gate returns the lane to the two-/three-context q-perp construction.

## Graph-node split required for independent falsification

The following claims must remain separate graph nodes; success of one is not
evidence for another.

### `P2R0G19B_H4_CONTEXT_INJECTIVITY`

- **Status:** hold.
- **Claim:** for a fixed target tangent of dimension `f`, generic centered
  diagonal contexts are pointwise injective when `m*1023>=f`; hence three for
  the full `r55` tangent and two for the two flat-energy target ranks.
- **Promotion criterion:** exhibit an explicit nonzero restricted minor at the
  actual core/context tuple.
- **Terminal close:** a target-relevant tangent direction lies in the common
  context kernel.

### `P2R0G19B_H4_CONTEXT_CONDITIONING`

- **Status:** hold.
- **Claim:** the restricted generalized eigenvalue `tau^2` controls stable
  Bellman recovery.
- **Promotion criterion:** `tau>0` and
  `(2 eta/tau+epsilon_model)/||signal|| <= 0.18113`.
- **Terminal close:** the minor is zero or the certified mismatch misses the
  value gate.  Counts, genericity, and ordinary frame condition numbers do not
  promote this node.

### `P2R0G19B_H4_CONTEXT_COST`

- **Status:** conditional.
- **Claim:** the required contexts fit the absolute deployment cap without
  double counting.
- **Promotion criterion:** audited lineage proves either
  `9+10m c_ctx<=49` for target-context streams or
  `9+10g_perp+F_one-shot g_frame<=49` for an explicit residual stream.
- **Terminal close:** the conditioning-required `m` exceeds four at one GEMM
  per layer or two at two GEMMs per layer, or the alleged one-shot frame must in
  fact be propagated.

### `P2R0G19B_H4_K31_ACTION_COVERAGE`

- **Status:** hold / pivot.
- **Claim:** direct scalar actions
  `A(U,V)=48<R(U,U),R(U,V)>` span the full Bellman correction on the r55 fiber.
- **Promotion criterion:** prove
  `ker DA subset ker D Xi`, equivalently the Bellman row space is contained in
  the chosen restricted K31-action row space, then pass the action-specific
  `tau` gate.
- **Terminal close:** the diagonal-211 or other Bellman covector lies outside
  that row span.  Availability of original-coordinate K31 planes is not
  sufficient.

## Cross-lane reset-mixture injectivity refinement

For a quadratic reset `R(X)=alpha X+beta X^2`, the transformed `K21` contains a
quartic pair of coefficient tensors of the form

\[
\alpha^3 U+\alpha^2\beta V
=\alpha^3(U+tV),\qquad t=\beta/\alpha.
\]

Let a fixed unidentified target tangent
`S subset Sym^2_0 direct-sum Sym^2_0` have dimension at most `2099`.  A reset
context `(W_s,t_s)` observes

\[
\operatorname{diag}_0\{W_s^T(U+t_sV)W_s\}.
\]

For a nonzero pair `(U,V)`, the matrix `U+tV` can vanish for at most one
distinct value of `t`.  Consequently four distinct reset ratios leave at least
three nonzero centered-diagonal constraints for every nonzero pair.  Their
generic incidence codimension is at least

\[
3(n-1)=3069>2098,
\]

so four generic bases give a uniform **fixed-S injectivity existence theorem**.
If the tangent excludes proportional coefficient pairs that can cancel at a
sampled ratio, three distinct ratios generically suffice.  This still supplies
no quantitative conditioning: promotion requires separation among the `t_s`
and the actual restricted generalized eigenvalue.  Four one-GEMM-per-layer
contexts consume `40` GEMM equivalents and exactly meet the `49` total after
the nine-GEMM base; the construction is infeasible at two GEMMs per context.

## 2026-08-31 — actual D10 K31 Bellman factor-through audit

### The actual frozen D10 reverse mask

The D10 covariance tangent and its reverse adjoint are

\[
E_{\ell+1}=W_{\ell+1}^T(D_\ell\odot E_\ell+J_\ell)W_{\ell+1},
\qquad Y=\langle H_L,E_L\rangle,
\]

\[
\Lambda_L=H_L,\qquad
B_\ell=W_{\ell+1}\Lambda_{\ell+1}W_{\ell+1}^T,
\qquad
\Lambda_\ell=D_\ell\odot B_\ell.
\]

For the K31 injection

\[
J_\ell^{31}=\frac16\left(
G_\ell^{31}\odot dK31_\ell
+G_\ell^{13}\odot dK31_\ell^T
\right),
\]

the exact source-plane reverse mask is

\[
\boxed{
Q_\ell^{31}
=\frac16\left[
B_\ell\odot G_\ell^{31}
+(B_\ell\odot G_\ell^{13})^T
\right].
}
\]

Consequently the layer contribution is

\[
a_\ell^{31}=\langle Q_\ell^{31},dK31_\ell\rangle,
\qquad
Y_{31}=\sum_{\ell\in\mathrm{D10}}a_\ell^{31}.
\]

This is an adjoint identity, not an approximation.  The masks are frozen at the
trajectory about which the D10 tangent is taken.  Coadapting them across a
finite alternative trajectory belongs to the frame/model residual, not to the
frozen first derivative.

Let

\[
P_\ell=W_\ell(Q_\ell^{31})^T,
\qquad
U_{\ell a}=\sum_iW_{\ell,ia}X_{\ell i},
\qquad
V_{\ell a}=\sum_iP_{\ell,ia}X_{\ell i}.
\]

Cumulant multilinearity proves, for any centered finite-fourth-moment law,

\[
\boxed{
a_\ell^{31}
=\sum_a\kappa_4(U_{\ell a},U_{\ell a},U_{\ell a},V_{\ell a}).
}
\]

Thus the actual D10 K31 mask is exactly a sum of the scalar actions studied in
the preceding checkpoint.

### Exact action Jacobian on the AQ core fiber

Use the unhalved quadratic convention

\[
X_i=h_i^Tz+z^TA_i z-\operatorname{tr}A_i.
\]

At one layer define

\[
N_a=\sum_iW_{ia}A_i,\qquad
D_a=\sum_iP_{ia}A_i,
\]

\[
u_a=\sum_iW_{ia}h_i,\qquad
r_a=\sum_iP_{ia}h_i.
\]

The exact local action is

\[
\mathcal F(N,D,u,r)
=48\operatorname{tr}(N^3D)
+24\left[u^TNDu+u^TN^2r\right].
\]

Its four gradients are

\[
G_D=48N^3+24\operatorname{Sym}(Nuu^T),
\qquad g_r=24N^2u,
\]

\[
g_u=24[(ND+DN)u+N^2r],
\]

\[
\begin{aligned}
G_N={}&48(N^2D+NDN+DN^2)\\
&+24\left\{
\operatorname{Sym}(Duu^T)
+\operatorname{Sym}[u(Nr)^T]
+\operatorname{Sym}[(Nu)r^T]
\right\}.
\end{aligned}
\]

The apparent dense intermediate `P=WQ31^T` fuses away.  If the forward mixed
cores are already available, then

\[
\boxed{
D_a=\sum_bQ^{31}_{ab}N_b,
\qquad
r_a=\sum_bQ^{31}_{ab}u_b.
}
\]

For the frozen reverse mask, a core-fiber perturbation therefore satisfies

\[
\dot D_a=\sum_bQ^{31}_{ab}\dot N_b,
\qquad
\dot r_a=\sum_bQ^{31}_{ab}\dot u_b.
\]

Define the already-pulled local gradients

\[
\widetilde G_{N,b}=G_{N,b}+\sum_aQ^{31}_{ab}G_{D,a},
\qquad
\widetilde g_{u,b}=g_{u,b}+\sum_aQ^{31}_{ab}g_{r,a}.
\]

Then the exact action Jacobian at layer `ell` is

\[
\boxed{
J_{A,\ell}[\dot\theta]
=\sum_b\left[
\langle\widetilde G_{N,b},\dot N_b\rangle_F
+\widetilde g_{u,b}^T\dot u_b
\right],
}
\]

where

\[
\dot N_b=\sum_iW_{ib}\dot A_i,
\qquad
\dot u_b=\sum_iW_{ib}\dot h_i.
\]

Equivalently its covectors on the original core bank are

\[
\Gamma_{A,i}=\sum_bW_{ib}\widetilde G_{N,b},
\qquad
\gamma_{h,i}=\sum_bW_{ib}\widetilde g_{u,b}.
\]

Composing these rows with the symbolic core-reset tangents through L5--L14 and
summing gives `J_A=sum_l J_A,l` on the original r55 inverse fiber.  The pure
quadratic specialization is obtained by deleting the vector terms:

\[
J_{A,\ell}
=48\sum_a\operatorname{tr}\left[
(N_a^2D_a+N_aD_aN_a+D_aN_a^2)\dot N_a
+N_a^3\dot D_a
\right].
\]

For direct moments rather than AQ inference, the exact single-action derivative
is also explicit.  With centered perturbations `dot U,dot V`,

\[
\begin{aligned}
D\kappa_4(U,U,U,V)={}&
\mathbb E[3U^2\dot U\,V+U^3\dot V]\\
&-3\left\{
2\mathbb E[U\dot U]\mathbb E[UV]
+\mathbb E[U^2]\mathbb E[\dot U V+U\dot V]
\right\}.
\end{aligned}
\]

This derivative is assumption-free if those projected action moments are
actually observable.

### Exact factorization theorem, with its boundary

Let `B_31` denote **only** the D10 K31 contribution to the frozen tangent
endpoint.  The adjoint derivation gives

\[
B_{31}(\theta)=A_{31}(\theta)
=\sum_{\ell,a}\kappa_4(U_{\ell a},U_{\ell a},U_{\ell a},V_{\ell a}).
\]

Therefore, on the regular AQ r55 core fiber,

\[
\boxed{
J_{B31}=J_A,
\qquad
\ker J_A=\ker J_{B31}.
}
\]

This proves the Bellman factor-through gate exactly for the K31 channel.  It is
stronger than generic context injectivity: no core representative must be
identified if the true scalar action is observed directly.

It does **not** prove factorization of the complete network correction.  The
exact Bellman decomposition is

\[
B_{\rm full}=A_{31}+\rho_B,
\]

where `rho_B` includes initialization, non-K31/K22 sources, beyond-AQ chaos,
frame/state coadaptation, collision/lineage error, numerical truncation, omitted
layers, and terminal error.  Hence

\[
J_{B,\rm full}=J_A+J_\rho,
\]

and

\[
\boxed{
\ker J_A\subseteq\ker J_{B,\rm full}
\iff J_\rho|_{\ker J_A}=0.
}
\]

That final condition is false for the unrestricted hierarchy.  A lawful
beyond-AQ perturbation in an `H3(U)H1(V)` crossed-action direction changes
`E[U^3V]` while preserving the lower AQ/core summaries; generic K22 and
frame/state defects give further directions with `J_A=0` but `J_B,full !=0`.
Thus:

- **proved:** exact K31-component factorization under the actual D10 mask;
- **refuted without extra premises:** factorization of the full Bellman
  correction through the AQ K31 scalar alone.

Direct acquisition of the universal scalar cumulant removes the AQ discrepancy
but does not remove independent K22, frame/state, omitted-layer, or terminal
terms.

### Conditioning without a generic-context argument

Compile the D10 accumulator itself as the observed scalar

\[
A_{31}=\sum_\ell a_\ell^{31}.
\]

For the K31 Bellman component in the same normalization, `J_B31=J_A`, so the
exact restricted conditioning is

\[
\boxed{\tau_{31,\rm direct}=1.}
\]

This is a normalization identity, not a probabilistic estimate.  If layer
actions are retained separately and the Bellman sum uses weights `w`, their
noise accumulates as `sqrt(w^T N w)`; compiling the weighted scalar does not
create an inverse-fiber penalty.

If the action is not observed directly and must instead be inferred from an
acquisition map `M:F55 -> Y`, let `ell=J_A`.  Exact scalar identifiability is

\[
\ker M\subseteq\ker\ell
\iff \ell\in\operatorname{row}M.
\]

With acquisition-noise metric `N`, the exact non-generic condition number is

\[
\boxed{
\tau_{31,M}
=\left[
\ell^T(M^TN^{-1}M)^+\ell
\right]^{-1/2},
}
\]

provided the row-space gate holds.  It is the reciprocal norm of the
minimum-noise-amplification coefficient vector representing `ell` from `M`.
This scalar-specific `tau`, not full fiber injectivity, is the relevant fallback
certificate.

### Model error and value gate

Write the deployed scalar residual as

\[
e_{31}=e_{\rm AQ}+e_{\rm mom}+e_{\rm mask}+e_{\rm layer}
+e_{\rm coef}+e_{\rm other},
\]

where `other` includes independent K22, frame/state, collision, numerical, and
terminal Bellman defects.  These terms are not presumed orthogonal.  Therefore

\[
\|e_{31}\|_V
\le\sum_r\|e_r\|_V,
\qquad
E_{31}=\|e_{31}\|_V^2
\le\left(\sum_r\|e_r\|_V\right)^2.
\]

With direct scalar action error `eta`, the exact K31 component has no `tau`
penalty; with inferred action it contributes `eta/tau_31,M` (or the conservative
`2 eta/tau` under candidate selection).  The strict normalized gate is

\[
\boxed{
\frac{2\eta/\tau_{31}+\sum_r\|e_r\|_V}
{\|\text{Bellman task signal}\|}\le0.18113.
}
\]

For the current raw-MSE accounting the stronger end-to-end receipt is

\[
\boxed{
L_{FH}+E_{31}\le7.453\times10^{-10}.
}
\]

After old-stream residualization, if `Z31_perp` is the novel scalar response,
`s31=||Z31perp||^2`, and `h31=<R,Z31perp>`, its exact Schur gain is

\[
\Delta_{31}=h_{31}^2/s_{31}.
\]

At the nominal exact-primary margin the scalar lane must supply at least
`2.86e-10`, equivalently at least `27.732%` of exact q4 value under exact
coefficient calibration.  Factorization and `tau=1` do not establish this
novelty/value inequality.

### Corrected action cost

At `n=1024,r=55`, `d=r(r+1)/2=1540`,

\[
d/n=1.50390625,
\qquad r^3/n^2\simeq0.15867.
\]

Because `D=Q31 N` fuses away `P=WQ31^T`, ten late core transitions plus the
local K31 endpoint cost, under the stated factorable schedule,

\[
\boxed{
11\left(d/n+r^3/n^2\right)\simeq18.288
}
\]

of the `21` equivalents remaining beside D14.  The margin is about `2.712`.
Three new rectangular context acquisitions cost
`3d/n=4.512` and therefore fail.  Any fallback context evidence must ride
already-computed natural layers; the direct scalar action or direct projected
moments are the only standalone route within this corrected budget.  The cost
statement remains conditional on exact factor reuse and a reset-product
constant no larger than approximately `2.71`.

## Graph-node split for checkpoint 8

### `P2R0G19B_K31_D10_EXACT_FACTORIZATION`

- **Status:** solved exact for the K31 component.
- **Claim:** the actual frozen D10 adjoint satisfies
  `B31=A31=sum_l,a kappa4(U,U,U,V)`, hence `J_B31=J_A` and the kernel inclusion
  is equality.
- **Promotion criterion:** verify the deployed orientation, centering,
  `1/6` convention, active L5--L14 mask, and collision/old-owner subtraction.
- **Non-claim:** no full-Bellman closure or value gain follows.

### `P2R0G19B_K31_D10_CONDITIONING`

- **Status:** direct-action theorem solved; acquisition fallback open.
- **Claim:** direct accumulated action has `tau31=1`; an inferred action has
  `tau31=[ell^T(M^TN^-1M)^+ell]^-1/2` after the row-space gate.
- **Promotion criterion:** prove direct moment/core-action observability, or
  exhibit the actual acquisition row-space representation and a positive
  scalar-specific `tau` meeting the `0.18113` gate.
- **Terminal close:** `ell` is outside the acquisition row space or the
  conditioned error exceeds the task margin.

### `P2R0G19B_K31_D10_MODEL_ERROR`

- **Status:** open / blocking for full correction.
- **Claim:** AQ action plus the D10 mask approximates the complete desired q4
  Bellman correction within the residual budget.
- **Promotion criterion:** certify
  `L_FH+||e_AQ+e_mom+e_mask+e_layer+e_coef+e_other||_V^2`
  at most `7.453e-10`, without assuming residual orthogonality.
- **Terminal close:** exhibit a target-valued beyond-AQ, K22, frame/state, or
  terminal direction in the K31-action kernel whose norm exhausts the margin.

## 2026-08-31 — checkpoint 9: residual K31 action, not total action

### Ownership correction

The previous identity

\[
\sum_{\ell,a}\kappa_4(U_{\ell a},U_{\ell a},U_{\ell a},V_{\ell a})
=\sum_\ell\langle\Lambda_\ell,K31_\ell\rangle
\]

is exact, but the deployed D11/Q24/CP3/DTOT2 lineage already owns its **formal
same-site assembled K31 action**.  Hence the solved total-action `tau=1` theorem
is an implementation equivalence/duplicate theorem, not a novelty theorem.

Let

\[
K_\ell^T=K31_\ell^{\rm true},
\qquad
K_\ell^A=K31_\ell^{\rm assembled},
\qquad
R_\ell^{31}=K_\ell^T-K_\ell^A.
\]

Here `K^A` is the single already-owned connected plane assembled from CP4,
quartic star, and the fitted connected covariance-pairing provider.  CP3 is
already folded into its reset moments, while DTOT2 contributes to the local
costate `Lambda`; none is a second provider.  The only same-site candidate is

\[
\boxed{
\rho_{31}^{\rm local}
=\sum_{\ell\in\mathrm{D10}}
\langle\Lambda_\ell,R_\ell^{31}\rangle_F.
}
\]

The diagonal collision belongs to K4 and the lower-moment disconnected
counterterm has already been removed exactly once.  Both `K^T` and `K^A` must
use the same centering, off-diagonal mask, normalization, birth site, and
costate before subtraction.

### Exact residual Jacobian

For the frozen D10 costate, the local residual action Jacobian on any admissible
tangent `h` is

\[
\boxed{
J_{\rho,\rm fr}[h]
=\sum_\ell\left\langle
\Lambda_\ell,
(DK_\ell^T-DK_\ell^A)[h]
\right\rangle_F.
}
\]

For a coadapted finite-trajectory claim, the total derivative is instead

\[
\boxed{
J_{\rho,\rm co}[h]
=J_{\rho,\rm fr}[h]
+\sum_\ell\langle D\Lambda_\ell[h],R_\ell^{31}\rangle_F.
}
\]

The second term is the consumer/mask-motion secant.  It vanishes only for the
compiled frozen tangent or at a point where the residual plane itself is zero.

For the AQ/core action derived in checkpoint 8,

\[
K^{AQ}=K^A
\]

at the formal Q24/DTOT2 ownership level.  Therefore the earlier AQ Jacobian is
the derivative of the assembled duplicate sibling.  The genuinely novel row is

\[
\ell_\rho=J_T-J_A,
\]

where `J_T` is the derivative of the actual true fourth-action moment and `J_A`
is the assembled/AQ derivative.  Reusing `J_A` alone makes `ell_rho=0` by
construction and cannot discover a residual.

### What the current contexts observe

The exact same-site Q24 receipt is

\[
\sum_i\kappa_4(U_i^A,U_i^A,U_i^A,V_i^A)
=\langle\Lambda,K^{A}\rangle_F.
\]

Consequently every context built solely from the current Q24 Edgeworth moments,
CP4/star/pairing pieces, or the AQ trace formula obeys

\[
\boxed{
\mathcal M_{\rm current}^{\rm residual}
=\mathcal M_{\rm current}^{T}
-\mathcal M_{\rm current}^{A}=0
}
\]

as a symbolic identity after the lawful duplicate subtraction.  The current
contexts therefore see **no nonduplicate residual at all**.  This does not prove
`R31=0` physically; it proves that the existing assembled state is blind to it.

More formally, let `S_old` be the deployed information sigma-field.  If two
lawful white-box completions agree on `S_old` but have different true projected
`m31`, then

\[
\inf_f\mathbb E[(\rho_{31}-f(S_{\rm old}))^2]
=\mathbb E\operatorname{Var}(\rho_{31}\mid S_{\rm old})>0.
\]

A calculation using only `S_old` returns the assembled action and hence zero
after subtraction.  A nonzero residual requires a new exact true-law moment
interface, a proven higher-chaos closure, or a future fresh-law observation.

### Direct residual factorization and its tautological tau

Suppose a new interface directly exposes the matched true scalar

\[
T_\ell=\langle\Lambda_\ell,K_\ell^T\rangle
=\sum_i\left\{
\mathbb E_T[U_i^3V_i]-3\mathbb E_T[U_i^2]\mathbb E_T[U_iV_i]
\right\},
\]

while the assembled scalar

\[
A_\ell=\langle\Lambda_\ell,K_\ell^A\rangle
\]

is already resident.  Then the observed difference `T_l-A_l` is literally the
same-site Bellman residual.  Hence

\[
J_{\rm obs}=J_{\rho,\rm fr},
\qquad
\ker J_{\rm obs}=\ker J_{\rho,\rm fr},
\qquad
\boxed{\tau_{\rho,\rm direct}=1}
\]

in the same accumulated-scalar norm.  This survival of `tau=1` is tautological
and conditional on acquiring new truth-side information.  It gives no evidence
that the current contexts contain that information or that the residual is
nonzero.

For an indirect residual acquisition `M`, with residual target row
`ell_rho`, the exact scalar condition remains

\[
\ker M\subseteq\ker\ell_\rho
\iff\ell_\rho\in\operatorname{row}M,
\]

\[
\boxed{
\tau_{\rho,M}
=\left[
\ell_\rho^T(M^TN^{-1}M)^+\ell_\rho
\right]^{-1/2}.
}
\]

For the current assembled contexts, `M=0` on the residual quotient.  Therefore

\[
\boxed{
\tau_{\rho,\rm current}=0
}
\]

whenever `ell_rho` is nonzero.

### Delayed collision-free primitive completion

Immediate same-frame K31 is zero on a collision-free primitive fourth source.
Let `q_t` be such a birth, `U_{l<-t}^{(4)}` its physical fourth-order transport
to a later fresh-law boundary, and `M_l` the repeated-index K31 sampler there.
Its future-visible action is

\[
\boxed{
\rho_{31}^{\rm prim}
=\sum_t\sum_{l>t}
\left\langle
\Lambda_l,
M_lU_{l\leftarrow t}^{(4)}q_t
\right\rangle.
}
\]

Define the block-lower-triangular causal observability map

\[
(\mathcal O_{31}q)_l
=\sum_{t\le l}M_lU_{l\leftarrow t}^{(4)}q_t.
\]

If `G_prim` is the future Bellman row in the primitive birth basis, the exact
coverage theorem is

\[
\boxed{
\ker(\mathcal O_{\rm obs}|_{\mathcal R})
\subseteq\ker(G_{\rm prim}|_{\mathcal R})
\iff
G_{\rm prim}|_{\mathcal R}
\in\operatorname{range}
((\mathcal O_{\rm obs}|_{\mathcal R})^*).
}
\]

The local residual and delayed-primitive ledgers must be disjoint: if a fresh
`K^T_l-K^A_l` measurement already contains the transported primitive at layer
`l`, it is not added again through the second display.

The current formal Q24 contexts have `O_obs=0` on this primitive residual
quotient.  They cover a delayed primitive only when its future Bellman row is
also zero.  An exact fresh post-reset true-law interface can reopen the lane;
predicting those future measurements from the lower-order state recreates the
old H4 carrier problem.

### Difference conditioning and cancellation error

Although direct residual observation has structural `tau=1`, subtracting two
large actions can be absolutely ill-conditioned.  Let

\[
\widehat\rho
=\langle\Lambda+e_\Lambda,
(K^T+e_T)-(K^A+e_A)\rangle.
\]

Then

\[
\boxed{
|\widehat\rho-\rho|
\le
\|\Lambda\|_* (\|e_T\|+\|e_A\|)
+\|e_\Lambda\|_*\|R^{31}\|
+\|e_\Lambda\|_*(\|e_T\|+\|e_A\|).
}
\]

Thus `tau=1` removes inverse-fiber amplification but not truth-minus-closure
cancellation.  A useful cancellation ratio is

\[
\gamma_\rho
=\frac{\|R^{31}\|}
{\max(\|K^T\|,\|K^A\|)}.
\]

No positive lower bound on `gamma_rho` follows from the total-action theorem.
The true moment, assembled action, and costate must be accurate on the residual
scale.

### Residual model-error kernel and value gate

Let the complete desired residual be

\[
\rho_{31}^{\rm full}
=\rho_{31}^{\rm local}+\rho_{31}^{\rm prim}
+e_{\rm mask}+e_{\rm suffix}+e_{\rm collision}
+e_{\rm moment}+e_{\rm other}.
\]

The remaining model-error kernel contains:

1. true higher-chaos `m31` directions invisible to the assembled Q24 state;
2. collision-free births with zero immediate K31 but nonzero future action;
3. costate/mask-motion directions omitted by the frozen D10 adjoint;
4. off-diagonal/K4 ownership or normalization mismatches;
5. independent K22, frame/state, and terminal Bellman directions if the target
   is broader than the K31 residual.

For an observation Jacobian `J_obs` and full residual row `J_full`, coverage is

\[
\ker J_{\rm obs}\subseteq\ker J_{\rm full}.
\]

The current-context map violates this condition for every target-visible
nonzero residual direction because `J_obs=0` there.  With a new true-law oracle
and a future-complete causal interface, the remaining normalized error must
satisfy

\[
\boxed{
\frac{\eta_{\rm diff}/\tau_\rho
+\eta_{\rm prim}+\eta_{\rm mask}+\eta_{\rm suffix}
+\eta_{\rm collision}+\eta_{\rm other}}
{\|\text{novel residual Bellman signal}\|}
\le0.18113.
}
\]

All amplitude errors add before squaring unless value-orthogonality is proved.
The campaign-level receipt remains

\[
L_{FH}+\|e_{31}^{\rm residual}\|_V^2
\le7.453\times10^{-10}.
\]

Finally, algebraic owner subtraction does not replace statistical novelty.
For residual scalar feature `z_rho` and old design `E`, require

\[
g_{\rho|E}=g_{\rho\rho}-g_{\rho E}G_E^+g_{E\rho}>0,
\]

\[
c_{\rho|E}=c_\rho-g_{\rho E}G_E^+c_E\ne0,
\qquad
\Delta_{\rm raw}=c_{\rho|E}^2/g_{\rho|E}.
\]

The residual lane closes as a duplicate if `g_rho|E=0` and as task-dead if
`c_rho|E=0`.

### Cost correction after the novelty audit

The `18.288/21` AQ/core calculation from checkpoint 8 evaluates the assembled
quadratic-chaos/DTOT2 action and is therefore a **duplicate cost**, not a costed
novel provider.  It cannot be credited toward raw reduction.  The incremental
cost that matters is acquisition of the adjoint-weighted true-minus-assembled
`m31` residual or a lawful fresh post-reset action.  Its cost remains open; any
method using only the existing Q24 Edgeworth moments has zero residual output.

## Graph-node split for checkpoint 9

### `P2R0G19B_K31_TOTAL_ACTION_DUPLICATE`

- **Status:** solved exact / duplicate sibling.
- **Claim:** total same-site K31 has exact D10 factorization and `tau=1`, but is
  already owned by the assembled D11/Q24/CP3/DTOT2 action.
- **Terminal criterion:** enforce mutual exclusivity with the old forward
  K31/Q24 implementation; assign zero new value after owner subtraction.

### `P2R0G19B_K31_RESIDUAL_FACTORIZATION`

- **Status:** conditional theorem.
- **Claim:** a new matched true-law action oracle minus the assembled scalar is
  exactly `rho31_local`, with Jacobian equality and `tau=1`.
- **Promotion criterion:** exhibit the truth-side `E[U^3V]` interface at the
  exact birth/consumption site and matched lower moments/owners.
- **Terminal close:** the purported oracle is a function only of the assembled
  Q24/AQ state, in which case its residual is identically zero.

### `P2R0G19B_K31_RESIDUAL_CONDITIONING`

- **Status:** current contexts closed no; new oracle open.
- **Claim:** current residual map is zero and has `tau=0` for a nonzero target;
  direct true-minus-assembled observation has structural `tau=1` but must pass
  the cancellation-error bound.
- **Promotion criterion:** positive residual-scale signal-to-error and the
  `0.18113` normalized gate, or the exact indirect pseudoinverse `tau_rho,M`.

### `P2R0G19B_K31_DELAYED_PRIMITIVE_COVERAGE`

- **Status:** open.
- **Claim:** future fresh K31 measurements cover collision-free primitive
  births through the causal map `O31`.
- **Promotion criterion:** prove
  `ker O_obs subset ker G_prim` and a positive restricted singular value
  without predicting the fresh law via an H4 carrier.
- **Terminal close:** an explicit future-valued primitive lies in the observed
  causal kernel.

### `P2R0G19B_K31_RESIDUAL_MODEL_ERROR`

- **Status:** open / blocking.
- **Claim:** truth-side moment, delayed primitive, mask, suffix, ownership, and
  other Bellman residuals fit the raw-MSE budget.
- **Promotion criterion:** the amplitude-summed residual passes both the
  `0.18113` gate and `L_FH+||e31_residual||^2<=7.453e-10`, followed by positive
  old-span Schur novelty.
- **Terminal close:** the current zero observation map is the only lawful
  interface, or any target-visible kernel/error consumes the margin.

## 2026-08-31 — checkpoint 10: Z22 transient witness versus CountSketch H3

The two mechanisms below are unrelated implementations and must remain
separate.  `Z22` contracts one transient fourth-order residual against a known
future covector before discard.  CountSketch compresses and transports cubic
coefficient vectors and approximates later source-query inner products.

## A. Transient Z22 one-step witness

### Exact source and fixed-context action

At the D11 reset, after subtracting the active CP4 `(22)` plane, define

\[
K_{22}^{\rm sym}=\frac12(K22+K22^T),
\qquad
P_{22}=\operatorname{offdiag}(vv^T+2C\odot C),
\]

\[
\alpha=\frac{\langle K_{22}^{\rm sym},P_{22}\rangle}
{\langle P_{22},P_{22}\rangle},
\qquad
\boxed{Z_{22}=K_{22}^{\rm sym}-\alpha P_{22}.}
\]

All objects use the same off-diagonal convention.  This is the transient plane
that `_reset_stars` determines and then discards.  For a pulled-back future
3--1 context `(u,v)`, put

\[
x=u\odot u,
\qquad y=u\odot v.
\]

The first dense K31 action of the pure-22 completion is

\[
\boxed{a_{u,v}(Z)=3x^TZy.}
\]

For symmetric `Z`, its source mask is

\[
B_{u,v}=\frac32(xy^T+yx^T).
\]

If `Pi_Z` is the projector onto the lawful residual source space after CP4 and
pairing removal, the exact zero gate is

\[
\boxed{b_{u,v}:=\Pi_Z^*B_{u,v}=0.}
\]

One fixed context has task rank one whenever `b_uv !=0`; the ordinary matrix
rank of `B_uv` is generically two.  With `m` contexts the ordered application
rank is at most `m`, and the symmetric-mask rank at most `2m`.

### Fused dense-mask form and the exact Bellman factor-through gate

The preceding vector formula is the rank-one specialization of a useful exact
dense identity.  Let `W` be the next dense map, set

\[
X=W^{\odot2},
\]

and let `Lambda` be an arbitrary reverse mask on the next-layer `K31` plane.
The pure-22 contribution is

\[
R_{ab}=3\sum_{ij}W_{ai}^2W_{aj}W_{bj}Z_{ij}.
\]

Consequently, with

\[
H=(XZ)\odot W,\qquad Y=W\odot(\Lambda W),
\]

one has the three exactly equivalent factorizations

\[
\boxed{
R=3HW^T,
\qquad
\rho_\Lambda:=\langle\Lambda,R\rangle
=3\langle\Lambda W,H\rangle
=3\langle Z,X^TY\rangle .
}
\]

Because `Z` is symmetric, its single source row is therefore

\[
\boxed{
B_\Lambda=\frac32(X^TY+Y^TX).
}
\]

This proves two distinct cost statements which must not be conflated.  If only
one pulled-back pair `(u,v)` is needed, `x^TZy` is one matrix-vector action,
`O(n^2)`.  If a completely dense `Lambda` is needed, all of its row contexts
can nevertheless be fused through one `XZ` matrix multiplication, after which
the remaining work is elementwise, provided `Lambda W` is already exposed by
the reverse pass.  Multiple already-exposed terminal masks reuse the same
`XZ`; a new computation of each `Lambda W` must be billed separately.

Let `M` stack whichever scalar actions are actually acquired on an admissible
tangent space and let `g` be the declared scalar Bellman row.  There exists an
exact action decoder iff

\[
\boxed{
\ker M\subseteq\ker g
\quad\Longleftrightarrow\quad
g=\beta^TM\ \text{for some }\beta.
}
\]

If observation errors are measured by `||z||_{N^{-1}}`, the sharp scalar
conditioning is

\[
\boxed{
\tau_g
=\left[min_{M^T\beta=g^T}\beta^TN\beta\right]^{-1/2}.
}
\]

In particular, when the deployed word is literally the same residual action
`rho_Lambda` that defines the Bellman correction, choose `beta=1`: structural
coverage is tautological and `tau_g=1` in matched units.  This does **not**
certify a complete fourth-order provider.  All delayed primitives, other
`K22/K4` rows, omitted layers, and mismatch between the declared reverse mask
and the true suffix remain a separate model-error row

\[
g_{\rm model}=g_{\rm true}-\beta^TM.
\]

Thus the direct residual-action node may pass exact factorization even when the
complete-provider node has `tau_Z=0`; the latter cannot be used to veto the
former, nor can the former be advertised as full closure.

### Physical Jacobian of the fitted residual plane

The fixed-mask row above is not yet the full physical Jacobian because both
the fitted CP4 direction `P` and the pulled-back context can move with the
underlying trajectory.  Let `p_2=<P,P>` and differentiate

\[
\alpha=\frac{\langle K,P\rangle}{p_2},\qquad Z=K-\alpha P,
\qquad \langle Z,P\rangle=0.
\]

The exact coefficient differential is

\[
\boxed{
d\alpha
=\frac{\langle dK,P\rangle}{p_2}
+\frac{\langle Z-\alpha P,dP\rangle}{p_2}.
}
\]

Writing `Pi_Pperp H=H-P<P,H>/p_2`, the residual-plane differential is

\[
\boxed{
dZ
=\Pi_{P^\perp}dK
-\alpha\Pi_{P^\perp}dP
-P\frac{\langle Z,dP\rangle}{p_2}.
}
\]

The last term is required by the moving-orthogonality identity
`<dZ,P>=-<Z,dP>` and cannot be discarded as an old-span component before the
future context is applied.  A local norm certificate is

\[
\boxed{
\|dZ\|
\le \|dK\|
+\left(|\alpha|+\frac{\|Z\|}{\|P\|}\right)\|dP\|.
}
\]

For `a=3x^TZy`, the complete trajectory Jacobian is

\[
\boxed{
da
=3\left[(dx)^TZy+x^T(dZ)y+x^TZ(dy)\right].
}
\]

Thus the fixed-context action row `b_uv` is a lawful identifiability receipt
only on tangents satisfying `dP=dx=dy=0`, or after the other three terms have
been included in the observation map.  Define the physical row

\[
J_{Z,\rm phys}h
=3\left[(D x[h])^TZy+x^T(DZ[h])y+x^TZ(Dy[h])\right].
\]

All kernel inclusions and generalized singular values for an end-to-end Z22
claim must use `J_Z,phys`, not merely `H -> <B_uv,H>`.  Conversely, if the
deployment computes the scalar action from the realized `K,P,x,y` rather than
linearizing a reconstructed core, these co-adaptation terms are included
automatically and the formula is an audit identity rather than an extra cost.

### Endpoint-weighted restricted singular values

Let `H_Z` be the source covariance/energy metric on the residual Z22 plane.
For a single scalar witness the only nonzero source-to-action singular value is

\[
\boxed{
\sigma_{Z,\rm local}
=\|H_Z^{+1/2}b_{u,v}\|,
}
\]

where the quotient is by the action kernel.  Equivalently, it is the norm of
the action row in the dual source metric.  A nonzero value proves only that the
chosen context sees some Z22 direction.

For the actual task, let `M_31` stack the observed K31 context rows, and let
`G_full` contain every value-relevant consequence of the physical completion,
including K31, K4, and K22.  On a basis `R` for the admissible residual tangent,
define

\[
\boxed{
\tau_Z^2
=\lambda_{\min}\left(
R^TM_{31}^TN^{-1}M_{31}R,
R^TG_{\rm full}^TVG_{\rm full}R
\right).
}
\]

Coverage is exact iff

\[
\ker(M_{31}R)\subseteq\ker(G_{\rm full}R).
\]

If the declared target is only the same K31 scalar witness, `G_full=M31` and
`tau_Z=1` in matched norms.  For a complete pure-22 provider this equality is
not established: generic K4 and crossed-K22 rows are independent of one K31
context, so `tau_Z` can be zero.

For a scalar full Bellman row `g` and context-row span `B`, the exact omitted
row is

\[
g_{\rm omit}=g-\Pi_B^{H_Z^{-1}}g.
\]

The K31 witness is a complete value decoder precisely when `g_omit=0`.
Otherwise `||H_Z^{+1/2}g_omit||` is the sharp source-weighted model-error norm.

### Z22 cancellation and arithmetic-error gate

The action is a difference of two potentially large contractions:

\[
a_Z=3x^TK_{22}^{\rm sym}y-3\alpha x^TP_{22}y.
\]

Define the action cancellation factor

\[
\boxed{
\kappa_{Z,\rm cancel}
=\frac{|3x^TK_{22}^{\rm sym}y|
+|3\alpha x^TP_{22}y|}{|a_Z|},
}
\]

with value infinity when the denominator is zero.  Algebraic orthogonality
`<Z22,P22>=0` does not bound this context-specific factor.

If source-plane errors obey `||e_K||<=eta_K`, `||e_P||<=eta_P` and the fitted
coefficient error is `|delta alpha|<=eta_alpha`, then

\[
\|\widehat Z-Z\|
\le\eta_K+|\alpha|\eta_P
+\eta_\alpha(\|P\|+\eta_P).
\]

Allowing context errors `eta_x,eta_y`, a safe one-action bound is

\[
\begin{aligned}
|\widehat a_Z-a_Z|
\le3\big[&
(\|x\|+\eta_x)(\|y\|+\eta_y)
(\|Z\|+\eta_Z)\\
&-\|x\|\|y\|\|Z\|
\big],
\end{aligned}
\]

where the first product in brackets is understood as
`(||x||+eta_x)(||y||+eta_y)(||Z||+eta_Z)`.
Across layers and contexts these amplitude bounds add before squaring.  The
endpoint receipt is

\[
\boxed{\|e_{Z,\rm endpoint}\|_V\le3.80\times10^{-6}.}
\]

### Schur-gain lower bound under action error

The following bound applies both to Z22 and to the H3 sketch below.  Let `E` be
the complete old response design, `P_E` its value-metric projector, `r` the old
target residual, `f=(I-P_E)f_raw` the exact novel response, and
`fhat=f+e`, with `||e||_V<=eta`.  Put

\[
c=\langle r,f\rangle_V,
\qquad s=\|f\|_V^2.
\]

Then the deployed one-feature Schur gain has the rigorous lower bound

\[
\boxed{
\Delta(\widehat f\mid E)
\ge
\frac{(|c|-\|r\|_V\eta)_+^2}
{(\sqrt{s}+\eta)^2}.
}
\]

Promotion requires this lower bound, not the ideal exact-feature gain, to be at
least

\[
\boxed{g_0=2.86\times10^{-10}.}
\]

The corresponding signal amplitude is `sqrt(g0)=1.691e-5`; the endpoint-error
allowance `3.80e-6` is already about `22.5%` of that minimum amplitude.

### Z22 cost and replacement gates

If the future contexts are already present when `Z22` exists, `m_l` contexts at
layer `l` cost one `n by n` times `n by m_l` product.  The exact normalized
increment is

\[
\boxed{
C_Z=\sum_l\frac{m_l}{n}+C_{\rm context}+C_{\rm hook}.
}
\]

For the base-48 stack, additive deployment requires `C_Z<=1`.  Ten layers with
one reused context have leading cost `10/1024=0.009765625`; this is a genuine
cost pass conditional on timing and buffer reuse.  A new full-width context
construction consumes the slack, and storing Z22 for a later reverse pass
forfeits the transient exception.

Replacing D10 does not automatically help: D10 is one source of the future
context that makes the rank-one contraction possible.  A replace-D10 package
must price a compiled analytic context or a replacement reverse pass.  In
value space, with `q` the removed D10 response and `E` the retained design, the
exact replacement gate is

\[
\boxed{
\Delta_Z(\widehat f_Z\mid E)-\Delta(q\mid E)\ge g_0.
}
\]

The additive gate instead uses `Delta_Z(fhat|E,q)>=g0`.  These gains cannot be
interchanged because the conditioning sets differ.

**Z22 disposition:** it can meet the one-GEMM cost gate as a one-step scalar
witness with a reused context.  Mathematics alone supplies no lower bound on
its old-span Schur gain or cancellation scale, so neither `2.86e-10` nor
`3.80e-6` is presently certified.  It is not a complete fourth-order provider
without the K4/K22 coverage gate.

## B. CountSketch H3 carrier

### Orthonormal coefficient convention and sketch

Vectorize each symmetric cubic coefficient tensor in the orthonormal
`Sym^3(R^r)` basis: entries with multiplicities `1,3,6` receive weights
`1,sqrt(3),sqrt(6)`.  At the active rank here, `r=55`,

\[
p=\binom{r+2}{3}=29260,
\qquad \frac{p}{n^2}=0.0279045\quad(n=1024).
\]

The earlier `p=5984` / `p/n^2=0.005707` arithmetic belonged to an `r=32`
checkpoint and is not a cost receipt for the current campaign.

Fix before any value observation a bucket map
`h:{1,...,p}->{1,...,k}` and signs `s_j`.  CountSketch is

\[
(Sx)_b=\sum_{j:h(j)=b}s_jx_j.
\]

The carried state for unit `i` is `C_i=SZ_i`.  Dense width mixing commutes with
the coefficient sketch, so the width transport is applied directly to the
`n by k` bank.  At the endpoint, if `G` is the exact H3 Bellman query tensor,
the decoded action is

\[
\widehat f_S=\sum_e\langle SZ_e,SG_e\rangle,
\]

where `e` ranges over the required net/layer/unit rows.  The same preregistered
sketch must be used on sources and queries.  No tensor reconstruction is
performed.

The lawful source is not the absolute H3/H31 action.  The deployed old lineage
already owns its assembled H31 source.  Let

\[
\Delta Z_e=Z_e^{\rm boundary}-Z_e^{\rm old}.
\]

The only creditable CountSketch response is

\[
\boxed{
\widehat f_{\Delta,S}
=\sum_e\langle S\Delta Z_e,SG_e\rangle.
}
\]

Linearity permits source-side subtraction either before or after hashing, but
the same seed and normalization must be used.  Appending
`sum <SZ_boundary,SG>` without subtracting the assembled H31 action is a
duplicate.  Every source span, collision factor, restricted singular value,
and Schur gain below is therefore defined on `Delta Z`, followed by final
response-space residualization against the complete old design.

The complete AQ-derived post-D11 linear H3 K31 adjoints used by this proposal
are

\[
\begin{aligned}
G_Z={}&18\operatorname{Sym}(u^2r)
+72\{J_{Ru}(N)+J_{Nu}(R)+2J_u(N\circ R)\\
&\hspace{35mm}+J_{Nr}(N)+J_r(N^2)\},\\
G_Y={}&6u^3+72\{J_{Nu}(N)+J_u(N^2)\}.
\end{aligned}
\]

Their provenance, reset-tail error, and old-owner subtraction remain separate
premises; the sketch only approximates their contractions.

### Exact CountSketch collision law

For fixed vectors `x,y`,

\[
\mathbb E_S\langle Sx,Sy\rangle=\langle x,y\rangle,
\]

and with four-wise independent signs and pairwise independent buckets,

\[
\operatorname{Var}_S\langle Sx,Sy\rangle
\le\frac{2}{k}\|x\|^2\|y\|^2.
\]

For an aggregate action let

\[
A=\sum_e Z_eG_e^T.
\]

The exact scalar collision variance for a shared seed is

\[
\boxed{
\operatorname{Var}_S(\widehat f_S-f)
=\frac1k\sum_{i<j}(A_{ij}+A_{ji})^2
=\frac{2}{k}\|\operatorname{offdiag}\operatorname{sym}A\|_F^2.
}
\]

Cheap exactly owned actions can be used as control variates before hashing.
Let `A_j` be aggregate matrices whose traces are already known, and set

\[
A_\alpha=A-\sum_j\alpha_jA_j.
\]

Estimate only `tr(A_alpha)` by CountSketch and add the exact old traces back.
With

\[
G_{ij}=\langle\operatorname{offsym}A_i,
\operatorname{offsym}A_j\rangle_F,
\qquad
b_i=\langle\operatorname{offsym}A_i,
\operatorname{offsym}A\rangle_F,
\]

the target-blind minimum-variance coefficients solve `G alpha=b`, and the exact
remaining collision energy is the off-diagonal Schur complement

\[
\boxed{
\|\operatorname{offsym}A\|_F^2-b^TG^+b.
}
\]

This controlled residual, not the raw action matrix, belongs in the sharp noise
gate.  The coefficient rule must be fixed without using target outcomes, and
the old traces are added exactly once.

For a corpus response vector, the analogous cross-net covariance uses the
Frobenius Gram of these off-diagonal symmetric matrices.  More explicitly,
let `H_a=offsym(A_a)` for corpus row `a`, after any lawful control-variate
subtraction, and put

\[
\Gamma_{ab}=\langle H_a,H_b\rangle_F.
\]

Then the exact weighted seed-average collision energy is

\[
\boxed{
\mathbb E_S\|\widehat f_S-f\|_V^2
=\frac2k\operatorname{tr}(V\Gamma).
}
\]

For shared control coefficients, the corpus-optimal target-blind subtraction
uses the product inner product

\[
\langle H,H'\rangle_{V,F}
=\sum_{a,b}V_{ab}\langle H_a,H'_b\rangle_F;
\]

its coefficients again solve `G alpha=b` in this metric.  Define the sharp
controlled collision factor

\[
\boxed{
C_{\rm off,V}
=\frac{\operatorname{tr}(V\Gamma_{\rm controlled})}
{\|(I-P_E)f\|_V^2}.
}
\]

Unlike the norm envelope below, `C_off,V` can be less than one or even zero.
Under the same seed-average signal/noise orthogonality assumption, its exact
correlation gate is

\[
\boxed{
q_S^2=\frac1{1+2C_{\rm off,V}/k},
\qquad k>36.39\,C_{\rm off,V}\quad(q_3>0.9736).
}
\]

Thus the present `k=36` and `k=22` additive ceilings require respectively
`C_off,V<0.9893` and `<0.6046`.  This is the precise control-variate/collision
loophole left open after the conservative norm envelope fails.

A safe stacked bound is

\[
\mathbb E_S\|\widehat f_S-f\|_V^2
\le\frac{2}{k}
\sum_e\|Z_e\|^2\|G_e\|^2.
\]

Consequently the seed-randomized tail bound at the absolute endpoint threshold
`tau0=3.80e-6` is only

\[
\Pr_S\{\|\widehat f_S-f\|_V>\tau_0\}
\le
\frac{2\sum_e\|Z_e\|^2\|G_e\|^2}
{k\tau_0^2}.
\]

This Markov/Chebyshev receipt can justify a preregistered random-seed success
probability when its right side is small.  It cannot certify a particular
already-frozen seed; that requires the deterministic restricted singular-value
test below.

Define the endpoint cancellation factor

\[
\boxed{
C_{\rm CS,V}
=\frac{\sum_e\|Z_e\|^2\|G_e\|^2}
{\|(I-P_E)f\|_V^2}.
}
\]

Here `Z_e` means the owner-subtracted `Delta Z_e`.  If that difference is
formed exactly before hashing, CountSketch collision noise scales with
`||Delta Z||`, which is favorable.  If the new and old actions are formed by
separate approximate paths, there is an additional owner-cancellation factor

\[
\boxed{
\kappa_{H3,\rm owner}
=\frac{\|f_{\rm boundary}\|_V+\|f_{\rm old}\|_V}
{\|f_{\rm boundary}-f_{\rm old}\|_V}.
}
\]

It is unbounded near a duplicate.  The source paths must therefore be
synchronized and subtracted before sketch decoding whenever possible; otherwise
their absolute errors are amplified on the novel residual scale and are charged
inside `eta_source`.

Under the ideal seed-average model with collision noise orthogonal to the
signal,

\[
q_S^2\ge\frac1{1+2C_{\rm CS,V}/k}.
\]

The updated raw-value decomposition requires the H3 carrier correlation
`q_3>0.9736` even under the best available `v+q4` complement.  The norm-only
seed-average certificate therefore requires

\[
\boxed{k>36.39\,C_{\rm CS,V}.}
\]

Moreover `C_CS,V>=1`: rowwise Cauchy gives
`|<Delta Z_e,G_e>|^2<=||Delta Z_e||^2||G_e||^2`, and old-span
residualization can only decrease the denominator.  Hence the current `r=55`,
ten-layer additive ceilings `k<=36` (source coordinates exposed) and `k<=22`
(one comparable source projection charged) **cannot pass this conservative
certificate even in the best possible `C=1` case**.  A hypothetical `k=51`
requires `C<=1.401`.  These are seed-average sufficient bounds under the
stated orthogonal-noise model, not frozen-seed necessities: the exact
off-diagonal collision law or a deterministic structured seed can still do
better than the norm envelope.

Independently of collision probability, an endpoint-whitened rank-`k` carrier
has a capacity gate.  If `sigma_j` are the singular values of the exact
old-residualized H3 response, the best rank-`k` energy correlation is

\[
q_{3,k}^2
=\frac{\sum_{j\le k}\sigma_j^2}{\sum_j\sigma_j^2}.
\]

Therefore promotion requires

\[
\boxed{
\frac{\sum_{j>k}\sigma_j^2}{\sum_j\sigma_j^2}<1-0.9736^2
=0.052103\ldots .
}
\]

The top `k` endpoint-weighted modes must explain more than `94.79%` of the H3
response energy.  This capacity gate applies even to the optimal tuned linear
sketch and is separate from the frozen-seed distortion gate.

### Deterministic endpoint-weighted restricted singular value

Let `F` be the exact old-span-residualized H3 response operator on an admissible
source tangent, `F_S` the response after the fixed sketch, and let columns of
`R` span the quotient by `ker(F)`.  The exact task-weighted singular value and
relative distortion are

\[
\boxed{
\tau_{H3,S}^2
=\lambda_{\min}\left(
R^TF_S^TVF_SR,
R^TF^TVFR
\right),
}
\]

\[
\boxed{
\delta_{H3,S}^2
=\lambda_{\max}\left(
R^T(F_S-F)^TV(F_S-F)R,
R^TF^TVFR
\right).
}
\]

The triangle inequality gives

\[
\tau_{H3,S}\ge(1-\delta_{H3,S})_+.
\]

This generalized eigenvalue, after the seed is frozen and after old-span
residualization, is the binding conditioning certificate.  Unbiasedness over
unseen seeds is not a deployment certificate.

There is a sharper coefficient-space cross-pair test.  Let `U_Z,U_G` be
orthonormal bases for the actual endpoint-weighted source and query spans.
Define

\[
\boxed{
\epsilon_{ZG}(S)
=\|U_Z^T(S^TS-I)U_G\|_{\rm op}.
}
\]

If

\[
\mathcal A_V
=\left\|\left(
\sum_e\|Z_e\|\|G_e\|
\right)_{\rm corpus}\right\|_V,
\]

then

\[
\|F_S-F\|_V\le\epsilon_{ZG}\mathcal A_V.
\]

The absolute endpoint gate is therefore

\[
\boxed{
\epsilon_{ZG}\mathcal A_V
+\eta_{\rm source}+\eta_{\rm query}+\eta_{\rm transport}
\le3.80\times10^{-6}.
}
\]

Embedding the union span is a simpler sufficient condition:

\[
\|U^T(S^TS-I)U\|_{op}\le\epsilon
\iff
\sigma(SU)\subset
[\sqrt{1-\epsilon},\sqrt{1+\epsilon}].
\]

It requires `rank(U)<=k`.  The cross-pair condition has the exact rank
obstruction

\[
\boxed{
\epsilon_{ZG}(S)
\ge\sigma_{k+1}(U_Z^TU_G),
}
\]

because `(SU_Z)^T(SU_G)` has rank at most `k`.  Hence a one-word sketch requires
the second endpoint-weighted cross singular value to be below the error
allowance; any proposed replacement of width `k` requires the `(k+1)`st to be
negligible.  If the
source and query spans coincide in dimension greater than `k`, the lower bound
is one and uniform recovery is impossible.

### Optimal exact linear sketch theorem

The rank obstruction is tight if one is allowed an arbitrary, candidate-tuned
linear sketch.  Let `X,Y` be the endpoint-weighted source and query coefficient
subspaces with orthonormal bases `U_X,U_Y`, and define

\[
r_\times=\operatorname{rank}(U_X^TU_Y).
\]

Then

\[
\boxed{
\min\left\{k:\exists S\in\mathbb R^{k\times p},\ 
\langle Sx,Sy\rangle=\langle x,y\rangle\ 
\forall x\in X,y\in Y\right\}=r_\times.
}
\]

The lower bound follows because the sketched cross Gram has rank at most `k`.
For achievability, take principal-vector bases `x_i,y_i` with
`<x_i,y_j>=sigma_i delta_ij`.  For each positive principal cosine define

\[
Sx_i=\sqrt{\sigma_i}e_i,\qquad
Sy_i=\sqrt{\sigma_i}e_i,
\]

map the zero-cosine directions to zero, and extend linearly from `X+Y` to the
coefficient space.  The definition is consistent when `sigma_i=1` because
then `x_i=y_i`.  This realizes the exact cross bilinear form with precisely
`r_cross` rows.

This theorem separates algebraic width from implementation cost.  The
principal-vector sketch is generically dense and depends on the actual
endpoint-conditioned subspaces.  Applying one dense `k by p` projection to
all `n` units costs `kp/n^2` GEMM equivalents per layer, rather than the
CountSketch cost `p/n^2` independent of `k`.  At current `p=29260,n=1024,L=10`,
one dense projection plus the `n by k` transports consumes approximately

\[
10k\left(\frac p{n^2}+\frac2n\right)=0.2986k
\]

equivalents before context/hook cost; two dense projections consume about
`0.5776k`.  Hence the one-equivalent additive package can use an exact tuned
sketch only if `r_cross<=3` with one reusable projection, or `r_cross<=1` when
both source and query projections are charged.  A D10 replacement with 21
equivalents permits at most `r_cross<=70` or `<=36`, respectively, before AQ,
context, hook, and memory charges.

For a one-sparse CountSketch, exactness is much less generic.  Its defect is

\[
E_S=S^TS-I,\qquad
(E_S)_{ij}=s_is_j\,1_{h(i)=h(j)}\quad(i\ne j),
\]

and exactness is exactly `U_X^T E_S U_Y=0`.  For any fixed collision pattern
with `E_S!=0`, this equality is a proper algebraic condition on `(X,Y)` and
therefore fails for generic endpoint subspaces.  Since `k<p` forces a
collision, a compressed CountSketch should not be expected to be exact; it
must earn promotion through the quantitative cross-Gram/noise gate rather
than an exactness claim.

### CountSketch cost gates

The exact incremental cost over `L` active layers is

\[
\boxed{
C_{\rm CS}(L,k)
=L\left[
\frac{2k}{n}+\frac{p}{n^2}+g_{\rm src}
\right]+C_{\rm context}+C_{\rm hook}.
}
\]

The terms are the `W/Q` sketch-bank mixes, hashing the transient adjoint, and
source formation.  `g_src=1` for a generic dense source such as
`diag(CMC^T)`.  A lawful `J`-node Fourier/local birth may replace it by roughly
`0.0279045 J` per layer plus a certified quadrature/resolvent tail.  Recomputing
the pre-W AQ state adds `1.15625` per active layer.  No current `r=55` receipt
establishes the old checkpoint's `k<=464` memory allowance; deployment memory
must be recounted independently from the compute inequalities here.

With only one equivalent of additive slack and local source formation,

\[
k\le\frac n2\left[
\frac1L-\frac{p}{n^2}-g_{\rm src}
\right].
\]

At `L=10,g_src=0`, this gives `k<=36` after the query-coordinate/hash term.
Charging one comparable structured source-coordinate projection,
`g_src=p/n^2`, gives `k<=22`.  A literal one-word `k=1` costs about `0.2986`
over ten layers when its source coordinates are already exposed, or `0.5776`
when one comparable source projection is charged.  Any `g_src=1` generic
source makes the additive Base48 route impossible before context cost.

If D10 is removed and its `21`-equivalent budget is genuinely available, its
compute-only width ceiling is

\[
\boxed{
k\le \frac n2\left[
\frac{21-C_{\rm context}-C_{\rm hook}}{10}
-\frac{p}{n^2}-g_{\rm src}-g_{\rm AQ}
\right],
}
\]

with `g_AQ=0` for resident AQ/H/Q and `1.15625` if AQ must be recomputed.  With
zero context/hook overhead, the resulting ceilings are approximately `1060`
for exposed local source/resident AQ, `548` for generic source/resident AQ, and
`468` for exposed local source plus AQ recomputation; generic source plus AQ
recomputation is a compute no-go.  These are compute ceilings only.  The
actual `k` is the minimum of this ceiling, coefficient dimension, and a newly
audited `r=55` memory ceiling.

These counts are conditional on the H/Q context remaining lawfully available
after D10 removal.  If D10 owns that context, its replacement cost must be
added.

The exact value exchange gate is

\[
\boxed{
\Delta_{\rm CS}(\widehat f_S\mid E)
-\Delta_{\rm D10}(q\mid E)
\ge2.86\times10^{-10},
}
\]

where `E` excludes D10.  For an additive sketch, use
`Delta_CS(fhat|E,D10)>=2.86e-10`.  Seed-average correlation, coefficient-space
energy, and compute feasibility do not imply either Schur inequality.

**CountSketch disposition:**

- a universal one-word (`k=1`) claim is closed; even before its cross-rank
  obstruction, the norm-only `q3>0.9736` certificate would require
  `C_CS,V<0.0275`, while `C_CS,V>=1`;
- an additive exposed-source sketch can use at most `k<=36` over ten layers
  (`k<=22` with one comparable structured source projection).  Both fail the
  norm-only correlation certificate because `k<36.39 C` and `C>=1`.  They can
  reopen only through a frozen exact-collision/cross-Gram receipt plus the
  `94.79%` top-`k` response-energy gate;
- a replace-D10 carrier has the symbolic compute ceiling above; no current
  `r=55` memory receipt supports importing the old `k=464` choice.  Generic
  source plus AQ recomputation is closed on compute, while the other cases
  remain value-, conditioning-, and memory-open.

Neither sketch size currently has a symbolic receipt for the `2.86e-10` gain
or `3.80e-6` absolute action error.

### Exact noisy-feature Schur gate

The value and approximation gates can be combined without an asymptotic or
orthogonal-noise assumption.  Work after exact residualization against the old
feature span `E`, and write

\[
t=(I-P_E)y,\qquad f=(I-P_E)f_{\rm exact},\qquad
e=(I-P_E)(\widehat f-f_{\rm exact}).
\]

Set `T=||t||_V`, `s=||f||_V^2`, `c=<t,f>_V`, and suppose a deterministic
certificate gives `||e||_V<=eta`.  Then the approximate conditional Schur gain
obeys the safe norm-only lower bound

\[
\boxed{
\Delta(\widehat f\mid E)
\ge
\frac{(|c|-T\eta)_+^2}{(\sqrt{s}+\eta)^2}.
}
\]

Consequently, for any required gain `g` a closed-form sufficient admissible
error radius is

\[
\boxed{
\eta\le \eta_{\max}(g)
:=\frac{|c|-\sqrt{g}\sqrt{s}}{T+\sqrt{g}},
\qquad c^2/s>g.
}
\]

This is both the Z22 one-word gate and the CountSketch gate after their
candidate-specific error bounds are substituted.  For additive promotion use
`g=2.86e-10`.  For a D10 replacement use

\[
g=\Delta(D10\mid E)+2.86\times10^{-10};
\]

thus a replacement with no exact value margin has zero approximation budget,
even if its action correlation is high.

In relative form, let

\[
\rho=\frac{|c|}{T\sqrt{s}},\qquad x=\frac{\eta}{\sqrt{s}}.
\]

To retain a fraction `theta` of the exact conditional gain it is sufficient
that

\[
\boxed{
x\le\frac{\rho(1-\sqrt\theta)}{1+\rho\sqrt\theta}.
}
\]

This exposes the cancellation issue: a weakly target-aligned residual
(`rho` small) has a much tighter permissible carrier error than its raw feature
norm suggests.

The preceding rational radius is convenient but conservative because it makes
the worst numerator and denominator perturbations independently.  The exact
minimax over the same error ball is available.  Whiten the `V` inner product
and retain the definitions of `rho` and `x` above.  Decomposing
`f/sqrt(s)` into its component `rho` along `t/T` and perpendicular component
`sqrt(1-rho^2)`, the error ball subtends angular radius `arcsin(x)` at the
origin.  Therefore

\[
\boxed{
\inf_{\|e\|_V\le\eta}
\Delta(f+e\mid E)
=T^2\left[
\rho\sqrt{1-x^2}-\sqrt{1-\rho^2}\,x
\right]_+^2,
\qquad 0\le x\le\rho,
}
\]

and the infimum is zero for `x>=rho`, because the error ball then reaches a
feature orthogonal to the target.  This is the exact deterministic robustness
law, not merely a triangle bound.

Let `q=sqrt(g)/T`.  Provided `0<=q<rho`, the largest exact error radius that
guarantees gain at least `g` is

\[
\boxed{
\eta_\star(g)
=\sqrt{s}\left[
\rho\sqrt{1-q^2}-q\sqrt{1-\rho^2}
\right]
=\sqrt{s}\sin(\arccos q-\arccos\rho).
}
\]

For retaining a fraction `theta` of the exact gain, this becomes

\[
\boxed{
\frac{\eta_\star}{\sqrt{s}}
=\rho\left[
\sqrt{1-\theta\rho^2}
-\sqrt\theta\sqrt{1-\rho^2}
\right].
}
\]

Thus the earlier rational radius may be used as an easy sufficient screen, but
`eta_star` is the correct promotion boundary when `T,s,c` are certified.

For CountSketch, let

\[
A_V=\sum_e\|\Delta Z_e\|^2\|G_e\|^2
\]

with the corpus endpoint weights absorbed into the rows.  The seed-average
bound is `E_S||e||_V^2<=2 A_V/k`.  Markov's inequality and the exact minimax
Schur radius give the explicit probability certificate

\[
\boxed{
k\ge\frac{2A_V}{\delta\,\eta_\star(g)^2}
\quad\Longrightarrow\quad
\Pr_S\{\Delta(\widehat f_S\mid E)\ge g\}\ge1-\delta.
}
\]

This is a preregistration sizing rule, not a receipt for the subsequently
frozen seed.  The frozen package must still pass the deterministic cross-Gram
or generalized-singular-value certificate.  Replacing `eta` above by

\[
\epsilon_{ZG}\mathcal A_V+\eta_{\rm source}
+\eta_{\rm query}+\eta_{\rm transport}
\]

turns that deterministic certificate directly into the exact Schur promotion
gate.  Replacing it by the Z22 source/action error bound does the same for the
transient witness.  The standing absolute `3.80e-6` tolerance remains an
additional cap: the exact binding radius is

\[
\boxed{\eta\le\min\{3.80\times10^{-6},\eta_\star(g)\}.}
\]

CountSketch is a fallback/cheaper branch, not the sole H3 reopen.  A separate
exact `CM` contraction path has a quoted conditional cost around
`22.83+10=32.83` when its reverse `H` bank is reused.  If its provenance and
reuse receipts hold, the exact path dominates a sketch with unresolved
collision error.  This sibling cost does not relax any CountSketch gate and
must be audited in its own node.

## Graph nodes for checkpoint 10

### `P2R0G19B_Z22_ENDPOINT_CONDITIONING`

- **Status:** conditional/open.
- **Claim:** the transient K31 context covers the target-visible Z22 quotient.
- **Promotion:** positive generalized `tau_Z`, or exact row-space coverage of
  the complete declared scalar target; endpoint action error at most `3.80e-6`.
- **Close:** a K4/K22-valued residual lies in the K31-context kernel.

### `P2R0G19B_Z22_VALUE_AND_COST`

- **Status:** cost conditional pass, value open.
- **Promotion:** reused-context `C_Z<=1` and error-robust old-span Schur gain at
  least `2.86e-10`; for replacement, subtract the lost D10 Schur gain.
- **Close:** context construction/storage destroys the transient cost exception
  or the response residualizes to zero.

### `P2R0G19B_H3_COUNTSKETCH_CONDITIONING`

- **Status:** open; universal low-rank claim closed.
- **Claim:** the frozen preregistered sketch has a positive endpoint-weighted
  restricted singular value and absolute action error below `3.80e-6`.
- **Promotion:** deterministic generalized `tau`, cross-pair embedding, and
  collision/source/query/transport error gates all pass.  An exact tuned
  linear sibling may replace these only with a certified cross rank
  `r_cross=rank(U_Z^TU_G)` and lawful frozen construction.  In every case the
  top `k` endpoint-whitened modes must explain more than `94.79%` of response
  energy (`q3>0.9736`).
- **Close:** `sigma_{k+1}(U_Z^TU_G)` alone exceeds the normalized error budget.

### `P2R0G19B_H3_COUNTSKETCH_VALUE_AND_COST`

- **Status:** conditional cost / open value.
- **Promotion:** exact cost formula fits either additive or replace-D10 package,
  and the appropriate error-robust Schur/exchange gain is at least `2.86e-10`.
- **Exact-sketch cost gate:** additive requires `r_cross<=3` with one dense
  projection reused or `<=1` with two charged; replace-D10 requires `<=70` or
  `<=36` before AQ/context/memory charges.
- **Norm-envelope verdict:** current additive `k<=36`/`k<=22` fails
  `k>36.39 C_CS,V` because `C_CS,V>=1`; only an exact collision/control-variate
  receipt can reopen it.  The sharp seed-average reopen thresholds are
  `C_off,V<0.9893` at `k=36` and `<0.6046` at `k=22`, followed by the frozen
  deterministic receipt.
- **Close:** generic-source/AQ/context costs exceed the selected package, or
  the frozen response is duplicate/task-dead after old-span residualization.

## Cross-lane exact audit: centered flat-spectrum Clifford carrier

Let symmetric involutions `A,B` obey

\[
A^2=B^2=I,\qquad AB=-BA,
\]

and let the centered gate core be `M=rho A`.  Put
`V=span{I,A,B}`.  The proposed six-state direct H4 module
`K=mu4(Sym^2 V)` is indeed exact for one complete gate, not only for its
H0+H2 truncation.

The resolvent in the exact Wiener kernels reduces to

\[
\boxed{
itA(I-itA)^{-1}
=\frac{-t^2I+itA}{1+t^2}\in\operatorname{span}\{I,A\}.
}
\]

By the finite direct-action theorem only its symmetric powers through order
four occur.  Moreover `V` is the three-dimensional spin-factor Jordan algebra
and is reversible: every reversal-symmetrized word in `A,B` reduces to
`I,A`, or `B`; the sole fourth Clifford basis element `AB` is skew and cancels
under reversal.  Therefore every `r2/r4/r6/r8` contraction maps
`Sym^2 V` back to `Sym^2 V`.  No seventh direct H4 state is hidden in the gate
tail.

There is, however, a sector correction.  Under conjugation by `(A,B)`, the
six lifts have parities

\[
\begin{array}{c|cccccc}
&II&IA&IB&AA&AB&BB\\ \hline
(\Gamma_A,\Gamma_B)&++&+-&-+&++&--&++.
\end{array}
\]

For

\[
D=\Pi_4M_{1_{Q_A>0}-1/2}\Pi_4
\]

one has `[D,Gamma_A]=0` and `{D,Gamma_B}=0`.  Hence, with

\[
N=\operatorname{span}\{U_{AA},U_{AB},U_{BB}\},
\]

\[
\boxed{
D\{U_{AA},U_{BB}\}\subseteq\operatorname{span}\{U_{IA}\},
\qquad
DU_{AB}\subseteq\operatorname{span}\{U_{IB}\},
}
\]

and consequently

\[
\boxed{P_N\Pi_4M_{1_{Q_A>0}}\Pi_4P_N=\tfrac12I_N.}
\]

This proves the advertised same-reset novel block.  The stronger statement
that the whole trace sector swaps with `N` is false:

\[
D U_{II}\subseteq\operatorname{span}\{U_{IA}\},
\qquad
D U_{IA}\subseteq\operatorname{span}\{U_{II},U_{AA},U_{BB}\},
\qquad
D U_{IB}\subseteq\operatorname{span}\{U_{AB}\}.
\]

Thus old-to-novel return is generically present.  A scalar recurrence that
keeps only the factor `1/2` is exact across time only if

\[
\boxed{
P_{N_t}\Phi_{t\leftarrow s}^{\rm future}
P_{O_s}D_sP_{N_s}=0
\quad\text{for every omitted old excursion }s<t,
}
\]

or if the old engine carries the complete coupled `O` block and its cross
actions.  A nonzero bias also destroys the gate-complement anticommutation and
must be treated as model error or with a larger module.

**Graph split:** `H4_CLIFFORD_DIRECT_SIX_LIFT` is solved exact under the
flat-spectrum/centered identities; `H4_CLIFFORD_HALF_QUOTIENT` is solved for
one reset; `H4_CLIFFORD_TEMPORAL_RETURN` remains conditional on complete old
cross-block ownership or the displayed endpoint-darkness zero.

## Checkpoint 11: CP4 four/six-ray provenance, residual q211 ownership, and the exact target-rank/cap exchange

### 11.1 The source is true minus the q4 law literally owned at this seam

Ownership depends on the actual reset tape.  Put `X=P_<=2(Y)`.  A carried
quadratic-chaos state by itself owns `kappa4(X)`; it does **not** own the
connected fourth cumulant of the ReLU output merely because it owns covariance
or K21.  Let `Y_AQ` be the output of the affine-quadratic reset law and define

\[
 \boxed{
 Y_{\rm owned}=
 \begin{cases}
 X,&\text{without an explicit fused H211 reset-surface oracle},\\
 Y_{\rm AQ},&\text{with that oracle literally present and retained}.
 \end{cases}}
\]

The lawful target is therefore

\[
 \boxed{
 \Gamma^{\rm res}_{211,l}
 =R_{\rm old}P_{211}
 \left[\kappa_4(Y_l^{\rm true})-
       \kappa_4(Y_{l}^{\rm owned})\right].}
\]

If the fused oracle is present, appending total AQ q211 is a duplicate and the
source is true-minus-AQ.  Without it, the lawful affine contribution includes
`kappa4(Y_AQ)-kappa4(X)`, which can be nonzero even at the affine model base.
An alternative CP implementation may instead replace the literal owner, in
which case its lost value enters the cap exchange below.  D10 covariance is
never credited with connected-q4 ownership by implication.

Write a residual anchored-pair decomposition at a birth seam as

\[
 \Gamma^{\rm res}_{211,l}
 =\sum_{r=1}^{R_l}{\cal Q}_{l,r}(D_{h_{lr}}),
 \qquad
 {\cal Q}_{l,r}(D)
 =\operatorname{Sym}_4(D\otimes b_{lr}\otimes c_{lr}),
 \qquad D_h=\operatorname{diag}h.
\]

Here `R_l` is an actual residual target rank, not the width of the AQ common
factor space.  Dense rotations of the pair legs can change the lawful CP4
acquisition class even when they leave the abstract tensor rank unchanged.

### 11.2 Exact four-ray provenance test versus the six-ray fallback

For one oriented atom, the four-ray identity is

\[
\begin{aligned}
G_4(A,B,C)={1\over4}\{&
 (A+B+C)^{\otimes4}-(A+B-C)^{\otimes4}\\
 &-(A-B+C)^{\otimes4}+(A-B-C)^{\otimes4}\}.
\end{aligned}
\]

In polynomial notation it equals

\[
 12A^2BC+4(B^3C+BC^3).
\]

Thus four CP columns acquire the desired q211 atom only after the oriented
leakage

\[
 \ell_\star(B,C)=B^{\odot3}\odot C+B\odot C^{\odot3}
\]

is canceled by the literal Star4 provider.  If `S_l` is that provider's
injection map, the exact same-seam condition for a collection of four-ray
atoms is the aggregate row-space test

\[
 \boxed{
 (I-P_{\operatorname{im}S_l})
 \sum_r \alpha_r\ell_\star(b_{lr},c_{lr})=0.}
 \tag{11.1}
\]

Testing atoms separately is sufficient but not necessary; cancellation can
occur across a grouped collection.  Literal coordinate/Star-frame pair legs
pass (11.1).  A generic rotated reverse mode does not pass merely because it
has low abstract rank.

If the group crosses further resets before being completely consumed, let
`R_CP,t`, `R_S,t`, and `R_t^(4)` be the actual CP, compensator, and represented
fourth-tensor reset maps.  Four-ray provenance then additionally requires

\[
 \boxed{
 R_{CP,t}G_t+R_{S,t}S_t
 =R_t^{(4)}(G_t+S_t)}
 \tag{11.2}
\]

at every crossed seam, with the oriented compensator itself retained.  A
same-site or one-transition relay may use four rays if the represented total
is consumed before the next Star reconstruction.  A multi-reset moving mode
may be charged at four rays only after (11.2) is proved.  Dense transport of a
coordinate `B^3C+BC^3` is generically outside the next coordinate-Star image,
so this is a real gate rather than bookkeeping.

If either (11.1) or the required instances of (11.2) fail, the exact no-leak
polarization

\[
\begin{aligned}
48T_{AABC}={}&p(A+B+C)-p(A+B-C)\\
 &-p(A-B+C)+p(A-B-C)-2p(B+C)+2p(B-C)
\end{aligned}
\]

uses six CP columns and carries autonomously.  The six-ray branch makes no
Star4 assumption.  It also does not declare sign-sketch q1111 error dark: that
error is charged by the complete endpoint map below.  Strict coordinate
collision sewing for a dense structured atom remains a separate gate.  Four
versus six is therefore decided by literal provider provenance and lifetime,
not by endpoint fit or by choosing a convenient rotated factorization.

For later rank accounting, define `R_star(Gamma)` as the smallest number of
anchored-pair atoms in a decomposition of `Gamma` whose **aggregate** leakage
passes (11.1) and whose lifetime passes (11.2).  Define `R_6(Gamma)` as the
smallest number of unrestricted anchored-pair atoms in a no-leak six-ray
decomposition.  A hybrid decomposition may use both.  These are acquisition
ranks relative to the literal card maps; they need not equal an SVD, tensor,
or common-factor rank.

For an approximate four-ray use, define the complete endpoint-whitened gauge
defect

\[
 g_r^{(4)}={\cal W}_r
 \left(\widehat T_r^{(4)}-T_r\right),
\]

where the difference includes every failed Star projection and every reset
intertwining defect.  Exact four-ray accounting requires `g_r^(4)=0`.
Otherwise this vector is deterministic model error in the cap/error theorem
below; it cannot be declared old or dark without applying the complete suffix
map.

### 11.3 Correct CP-column costs for the sign-averaged anchor square

For a signed anchor mask `h=h_+-h_-`, use independent Rademacher signs and

\[
 a_{rs}^{\pm}=\epsilon_{rs}^{\pm}\odot\sqrt{h_{r\pm}},
 \qquad
 \widehat D_r=
 {1\over m_{r+}}\sum_{s=1}^{m_{r+}}a_{rs}^+(a_{rs}^+)^T
 -{1\over m_{r-}}\sum_{s=1}^{m_{r-}}a_{rs}^-(a_{rs}^-)^T,
\]

with an absent sign side omitted.

Every frozen seed has

\[
 \operatorname{diag}\widehat D_r=h_r,
\]

so all error is the genuine cross-anchor/1111 component.  Let

\[
 s_r={\bf1}_{h_{r+}\ne0}+{\bf1}_{h_{r-}\ne0}\in\{1,2\}.
\]

There is a stronger aggregate identity than pricing every sampled square as
an independent four- or six-ray atom.  Recall

\[
 G_4(A;B,C)=12A^2BC+4L(B,C),
 \qquad L(B,C)=B^3C+BC^3.
\]

When both sign sides are present, they have total coefficients `+1` and `-1`,
so the leakage cancels **inside the CP bundle**:

\[
\boxed{
 {1\over m_+}\sum_{s=1}^{m_+}G_4(a_s^+;B,C)
 -{1\over m_-}\sum_{s=1}^{m_-}G_4(a_s^-;B,C)
 =12\widehat D_r\,BC.}
\]

This uses `4(m_++m_-)` columns, needs no Star provider, and is a pure grouped
CP tensor, so it transports autonomously.  For a one-sign mask, use the
single aggregate control

\[
 G_4(0;B,C)={1\over2}\{(B+C)^{\otimes4}-(B-C)^{\otimes4}\}=4L(B,C).
\]

Then `m^{-1}sum_s G_4(a_s)-G_4(0)` is exactly the sampled structured q211
word and costs only `4m+2` columns, rather than `6m`.  The two control rays are
retained in the same indivisible group.  Signed CP weights are required.

Thus the exact Star-free grouped cost for residual mode `r` is

\[
 \boxed{
 K_r(m_{r+},m_{r-})
 =4\left(m_{r+}{\bf1}_{h_{r+}\ne0}
        +m_{r-}{\bf1}_{h_{r-}\ne0}\right)
  +2{\bf1}_{s_r=1}.}
 \tag{11.3}
\]

If the one-sign mode separately passes the literal Star provenance and
lifetime gate, its final `+2` control columns may be omitted.  The ordinary
six-ray fallback costs `6(m_+1_++m_-1_-)`; it remains useful for a standalone
unbalanced atom but is dominated by the grouped identity for `m>1`.

Without a q1111 quotient, an exact anchor-square representation of one fixed
pair mode as real squares needs at least `rank(D_h)` square terms.  The
controlled construction above uses at most

\[
 4\operatorname{rank}(D_h)+2
\]

columns (the `+2` disappears when its square coefficients sum to zero or Star
provenance holds).  A full-support width-1,024 diagonal mask therefore does
not fit cap 2,048 through this exact square route.  Sign averaging is an
explicit approximation of the q1111/cross-anchor cancellation, not an exact
low-rank identity.

At a completely free cap of 2,048 columns, `R` simultaneously live equally
sampled signed modes cost `8mR` and satisfy

\[
 \boxed{
 m\le\left\lfloor{256\over R}\right\rfloor
 \quad(\text{balanced grouped four-ray}).}
 \tag{11.4}
\]

For one-sign masks, `R(4m+2)<=2048`.  If ten layer births remain live
simultaneously, replace `R` by the peak live rank, `10R` for a uniform
ten-layer persistent stream.  A certified one-transition consume-and-retire
relay instead pays only its simultaneous live rank.

More generally, with retained incumbent rank `K_keep`, birth lifetime sets
`I_r`, and cap `K_CP=2048`, the exact liveness constraint is

\[
 \boxed{
 K_{\rm keep}+\max_t\sum_{r:t\in I_r}
 K_r(m_{r+},m_{r-})\le K_{CP}.}
 \tag{11.5}
\]

Equation (11.5) is exact for the declared per-mode grouping and is a safe
upper bound when controls can be shared.  The globally sharp control count is
also explicit.  Partition sampled squares into indivisible groups `G` having
the same lifetime, write their four-ray coefficients as `alpha_q`, and define

\[
 L_G=\sum_{q\in G}\alpha_q
 \left(B_q^3C_q+B_qC_q^3\right),
 \qquad
 r_{\rm ctrl}(G)=\operatorname{rank}_{\rm Waring,\mathbb R}(L_G).
\]

Then the exact autonomous-CP count of this fixed four-ray-plus-control
construction is

\[
 \boxed{K_G=4|G|+r_{\rm ctrl}(G).}
 \tag{11.5a}
\]

Indeed the fixed four-ray blocks leave `4L_G`, and `r_ctrl(G)` signed fourth
powers are necessary and sufficient to cancel that aggregate leakage within
CP.  This does not claim that an unrelated joint Waring decomposition of the
entire target cannot use fewer rays; such a decomposition would be a separate
candidate with its own lawful source and grouping proof.
For paired positive/negative averages `L_G=0`; for one independent pair mode
`r_ctrl=2`, witnessed by the `(B+C),(B-C)` control.  Across modes with the same
lifetime, `r_ctrl` can be strictly below twice their count.  A lawful retained
Star compensator can instead make the CP control count zero, but then (11.1)
and (11.2) bind.  Controls cannot be shared across incompatible lifetimes
without a separate state-update receipt.

The CP-column receipt is only one part of deployment economics.  Source
oracle, Star compensation, hooks, consumption, and reset work must still pass
the separate 49-GEMM total-work gate.

### 11.4 Endpoint-whitened q1111 error without a darkness assumption

Let `E` denote the complete old arena after any incumbent groups selected for
displacement have been removed.  Let `Pi_E^V` be its projector in the corpus
value metric `V`, and let

\[
 {\cal W}_r=V^{1/2}\Pi_E^V{\cal L}_r:
 \operatorname{Sym}_n\longrightarrow {\cal H}_V
\]

be the complete suffix map from the anchor-square matrix of mode `r` to the
endpoint-whitened residual response.  This map includes transport, all
collisions, the q1111 sector, the old consumer, and the endpoint reverse
mask.  For an orthonormal basis `(e_a)` of its output support, set

\[
 H_{ra}={\cal W}_r^*e_a,
\]

and define the exact off-anchor sensitivity of each sign side

\[
\boxed{
A_{r\nu}=\sum_a
 \left\|\operatorname{off}\left(
 D_{\sqrt{h_{r\nu}}}H_{ra}D_{\sqrt{h_{r\nu}}}
 \right)\right\|_F^2,
 \qquad A_r=A_{r+}+A_{r-}.}
\tag{11.6}
\]

The absent sign side contributes zero.  Direct Rademacher contraction gives

\[
 \boxed{
 \mathbb E\left\|
 {\cal W}_r(\widehat D_r-D_{h_r})
 \right\|_2^2
 =2\sum_{\nu:h_{r\nu}\ne0}{A_{r\nu}\over m_{r\nu}}.}
 \tag{11.7}
\]

Equal counts `m_r+=m_r-=m_r` reduce (11.7) to `2A_r/m_r`.

The Jordan split `h=h_+-h_-` is not merely convenient; it minimizes this
endpoint variance over every coordinatewise nonnegative difference-of-squares
split.  Let `p,m>=0`, `p-m=h`, and even give the two sides the most favorable
common Rademacher signs.  On off-diagonal coordinate `(i,j)` the random
coefficient is

\[
 c_{ij}(p,m)=\sqrt{p_ip_j}-\sqrt{m_im_j}.
\]

If `h_i,h_j` have the same sign, Cauchy/AM--GM gives

\[
 |c_{ij}(p,m)|\ge\sqrt{|h_ih_j|};
\]

for opposite signs the universal lower bound is zero.  The Jordan choice
`p=h_+,m=h_-` attains both bounds simultaneously.  Since the complete
endpoint energy is a nonnegative weighted sum

\[
 2\sum_{a,i\ne j}c_{ij}(p,m)^2H_{ra,ij}^2,
\]

no added common diagonal baseline or antithetic coupling can reduce `A_r`
below (11.6).  Such a control can only tie the Jordan split on the endpoint's
supported pairs or increase variance.  Cross-mode correlated sketches remain
a separate joint-Gram question.

That cross-mode question has an exact constructive form.  Index sign sides by
`g=(r,nu)`, let `sigma_g` be the sign of that side in the source, and put

\[
 K_{g,a}=D_{\sqrt{h_g}}H_{r,a}D_{\sqrt{h_g}}.
\]

Within a block of sides having the same sample count, use one common
Rademacher vector but a frozen diagonal sign conjugation `S_g` for each side.
The exact block variance is

\[
\boxed{
 {2\over m}\sum_a
 \left\|\operatorname{off}\left(
 \sum_{g\in G}\sigma_gS_gK_{g,a}S_g
 \right)\right\|_F^2.}
\tag{11.7b}
\]

This changes neither the diagonal target nor the balanced CP-column count.
If the `S_g` are averaged independently, every cross term in (11.7b)
vanishes, so

\[
 \mathbb E_{S}A_G(S)=\sum_{g\in G}A_g.
\]

Consequently a deterministic sign design with

\[
 \boxed{\min_{S_g}A_G(S)\le\sum_gA_g}
\]

always exists, and strict improvement is possible when endpoint masks cancel
after sign conjugation.  Exact zero sketch variance occurs iff all the summed
off-diagonal masks in (11.7b) vanish.  This is a finite frozen max-cancellation
problem, not a claim from unbiasedness; its selected signs and actual
endpoint error must be preregistered.  It is the only within-family variance
loophole left by the Jordan optimality theorem.

This is already endpoint-whitened and therefore charges every future-visible
q1111 component.  No claim that q1111 is dark appears in (11.6)--(11.7).  If
independent sign families are used for different residual modes, their
cross-covariances vanish.  With shared signs, the right-hand side must instead
use the full cross-Frobenius Gram of the summed adjoint masks; replacing it by
the diagonal sum is not valid.

Let `z=sum_r z_r` be the exact endpoint-whitened residual target, with
`z_r={\cal W}_r(D_{h_r})`.  For a retained set `S`, define the deterministic omitted and
provenance/model defect

\[
 b_S=\sum_{r\in S}\left(g_r^{(4)}+g_r^{\rm coll}\right)+e_{\rm model}
     -\left(z-\sum_{r\in S}z_r\right).
\]

Here `g_r^coll` is the endpoint defect left by any incomplete strict-q211
collision sewing.  Both the balanced/control bundle in (11.3) and a six-ray
mode have `g_r^(4)=0`; neither has `g_r^coll=0` merely from polarization.  Let
`mathfrak D` range over lawful hybrid residual decompositions and lifetime
groupings, and let `K_mathfrakD(t)` be the sum of (11.5a) over groups live at
`t`.  The **exact seed-average target-rank/cap exchange within this declared
four/six acquisition family** is the integer program

\[
\boxed{
 {\cal E}_{K}^2=
 \inf_{{\mathfrak D}}\ \min_{S,\,m_{r\nu}\in\mathbb N}
 \left\{
 \|b_S\|_2^2
 +2\sum_{r\in S}\sum_{\nu:h_{r\nu}\ne0}
 {A_{r\nu}\over m_{r\nu}}:
 K_{\rm keep}+\max_t K_{\mathfrak D,S}(t)\le2048
 \right\}.}
 \tag{11.8}
\]

This is the requested exact rank/cap statement.  It jointly prices dropping
weak target modes, their actual endpoint Gram, balanced/control versus
standalone acquisition, lifetime, and finite-sample q1111 leakage.  It does
not equate abstract tensor rank with value-relevant rank.

At one common peak, let `K_0=sum_G r_ctrl(G)` be the frozen aggregate control
rank (the per-mode bound is `K_0<=2N_1(S)`), and put `K'=K-K_0`.  Fixing `S`
and the grouping while relaxing integer sample counts gives the closed-form
controlled-four-ray optimum

\[
 \boxed{
 m_{r\nu}^*={K'\sqrt{A_{r\nu}}
 \over4\sum_{j,\mu}\sqrt{A_{j\mu}}},
 \qquad
 \min 2\sum_{r,\nu}{A_{r\nu}\over m_{r\nu}}
 ={8\over K'}\left(\sum_{r,\nu}\sqrt{A_{r\nu}}\right)^2.}
 \tag{11.9}
\]

The sums range over present sign sides.  The package must round counts to
positive integers and recheck (11.5).  A standard six-ray-per-side allocation
has the same formula with `8/K'` replaced by `12/K`; it has `3/2` the sketch
MSE when the two-ray fixed overhead is negligible.  For one equally sampled
signed mode, the balanced bundle has MSE `16A/K`; for one one-sign mode it has
MSE `8A/(K-2)`.  For `R` equal-sensitivity, equally sampled signed modes the
MSE is `16R^2A/K`.

For endpoint-orthogonal balanced modes, (11.9) also gives an exact
target-rank selection rule in the continuous relaxation.  Put

\[
 a_r=\sum_{\nu}\sqrt{A_{r\nu}},
 \qquad \|z_r\|_2=\sigma_r,
 \qquad P_S=\sum_{r\in S}a_r.
\]

With fixed available `K'` and no mode-dependent control overhead,

\[
 \boxed{
 {\cal E}^2(S)=\sum_{r\notin S}\sigma_r^2
 +{8P_S^2\over K'}.}
 \tag{11.9b}
\]

At an optimum, every omitted mode obeys the no-add inequality

\[
 \sigma_r^2\le {8\over K'}(2P_Sa_r+a_r^2),
\]

and every retained mode obeys the no-drop inequality

\[
 \sigma_r^2\ge {8\over K'}(2P_Sa_r-a_r^2).
\]

For equal sensitivities `a_r=a` and ordered response energies, the optimum is
the best prefix `R`, with

\[
 {\cal E}_R^2=\sum_{j>R}\sigma_j^2+{8a^2R^2\over K'},
\]

and mode `R+1` is worth adding exactly when

\[
 \boxed{\sigma_{R+1}^2>{8a^2\over K'}(2R+1).}
 \tag{11.9c}
\]

This is the analytic target-rank/cap trade: adding weak modes eventually
hurts because it steals samples from every retained mode.  Nonorthogonal
responses must use their full endpoint Gram in (11.8), not this prefix rule.

If the lawful modes are endpoint-whitened orthogonal with norms `sigma_r`,
then the omission term in (11.8) is `sum_(r notin S) sigma_r^2`.  In the
general case it is the exact Gram norm of the omitted sum.  An SVD tail formula
may be used only when its rotated modes still satisfy the four-ray provenance
test **or** are retained in the balanced/control bundle (11.3)--(11.5a).
Otherwise standalone rotated atoms must be charged at six rays or decomposed
back into literal coordinate/Star atoms.  Collision sewing remains mandatory
for every dense option.

For preregistered random signs, a sufficient probability gate at the
controlled-four-ray continuous optimum is

\[
 \boxed{
 \|b_S\|_2+
 \sqrt{{8\over\delta K'}}
 \sum_{r,\nu}\sqrt{A_{r\nu}}
 \le\eta_{\rm req}.}
 \tag{11.10}
\]

Markov is deliberately assumption-light but can be very loose in `delta`.
There is a sharper bounded-increment certificate.  For each present sign side
define

\[
 B_{r\nu}=\sup_{\epsilon}
 \left\|{\cal W}_r
 \left(a_{r\nu}(\epsilon)a_{r\nu}(\epsilon)^T-D_{h_{r\nu}}\right)
 \right\|_2.
\]

A deterministic envelope is

\[
 B_{r\nu}\le
 \|{\cal W}_r\|_{F\to2}
 \sqrt{\|h_{r\nu}\|_1^2-\|h_{r\nu}\|_2^2}.
\]

The input norm here is exact and seed-independent because
`||aa^T-D_h||_F^2=||h||_1^2-||h||_2^2`.

Put

\[
 v=2\sum_{r,\nu}{A_{r\nu}\over m_{r\nu}},
 \qquad
 R_*=\max_{r,\nu}{B_{r\nu}\over m_{r\nu}},
 \qquad t=\log(2/\delta).
\]

The Hilbert-space Bernstein inequality gives the sufficient seed-sizing gate

\[
\boxed{
 \|b_S\|_2+\sqrt{2vt}+{2\over3}R_*t
 \le\eta_{\rm req}.}
\tag{11.10b}
\]

This charges the same complete q1111/collision endpoint map as (11.6), but has
logarithmic rather than `1/delta` tail dependence.  Allocation (11.9)
optimizes only the quadratic term; if the `R_*` term binds, sample floors must
also control `B_{rnu}/m_{rnu}`.

There is also a seed-independent fail-fast lower bound.  Every frozen side
error has zero diagonal and

\[
 \operatorname{rank}\widehat D_r\le q_r:=m_{r+}+m_{r-}.
\]

Let `|h_r|_(1)>=...>=|h_r|_(n)` be the ordered magnitudes.  Eckart--Young,
even before imposing the exact-diagonal constraint, gives

\[
 \boxed{
 \|\widehat D_r-D_{h_r}\|_F
 \ge\left(\sum_{j>q_r}|h_r|_{(j)}^2\right)^{1/2}.}
 \tag{11.10c}
\]

Let `E_r,q` be the candidate-specific set of nonzero errors generated by
admissible frozen sign averages with total rank at most `q`, and define

\[
 \tau_{r,q}^{\rm sec}=
 \inf_{X\in{\cal E}_{r,q}}
 {\|{\cal W}_rX\|_2\over\|X\|_F}.
\]

Every frozen single-mode sketch then obeys

\[
 \boxed{
 \|{\cal W}_r(\widehat D_r-D_{h_r})\|_2
 \ge\tau_{r,q_r}^{\rm sec}
 \left(\sum_{j>q_r}|h_r|_{(j)}^2\right)^{1/2}.}
 \tag{11.10d}
\]

For several modes, replace the separate constants by the corresponding
secant minimum of the joint block map `(X_r)_r -> sum_r W_rX_r`; otherwise
cross-mode cancellation can invalidate a sum of individual lower bounds.
Since the controlled CP bundle has `4q_r+r_ctrl` columns, (11.10d) is an
immediate cap-to-error gate before choosing a random seed whenever this
candidate-specific secant is certified.

The coarser linear-space constant obtained by infimizing over **all**
zero-diagonal symmetric matrices is valid but will usually be zero because
the endpoint has far smaller dimension.  Full off-diagonal injectivity is not
required; only the admissible rank-`q` error secants must avoid the endpoint
kernel.

The expected sensitivity has the compatible operator sandwich.  With

\[
 r_{h_{r\nu}}^2=\|h_{r\nu}\|_1^2-\|h_{r\nu}\|_2^2,
\]

let `tau_r,off` be the smallest singular value of `W_r` on the linear span of
the Rademacher side errors (zero is allowed).  Then

 \[
 {1\over2}\tau_{r,\rm off}^2r_{h_{r\nu}}^2
 \le A_{r\nu}\le
 {1\over2}\|{\cal W}_r\|_{F\to2}^2r_{h_{r\nu}}^2.
\]

There is one exact sparse-support escape from every stochastic gate.  If a
sign side has support size `s`, collect its sign vectors as the rows of an
`m by s` matrix `E`.  Then

\[
 \widehat D_{h}=D_{\sqrt h}{E^TE\over m}D_{\sqrt h},
\]

so the cross-anchor error vanishes identically iff

\[
 \boxed{E^TE=mI_s.}
 \tag{11.10e}
\]

Necessarily `m>=s` by rank.  Conversely, if `m=2^k>=s`, any `s` columns of the
Walsh--Hadamard matrix of order `m` give (11.10e).  Thus a signed two-side mode
has an explicit exact no-q1111 construction whenever

\[
 \boxed{
 4\left(2^{\lceil\log_2s_+\rceil}
       +2^{\lceil\log_2s_-\rceil}\right)\le K,}
 \tag{11.10f}
\]

with absent sides omitted and the one-sign two-ray control added when Star
provenance is unavailable.  Under a free cap `K=2048`, this is useful only for
sparse anchor masks: a one-sign Star-free Walsh construction is guaranteed up
to support 256 (`4*256+2` columns), while a lawful Star cancellation permits
support 512 exactly at cap.  It does not rescue a diffuse support-1,024 mask.

Exact **endpoint action** can require far fewer patterns than exact tensor
reconstruction.  For one sign side define the centered endpoint vector

\[
 v_{r\nu}(\epsilon)={\cal W}_r\left[
 D_{\sqrt{h_{r\nu}}}(\epsilon\epsilon^T-I)
 D_{\sqrt{h_{r\nu}}}\right]
\]

and let

\[
 d_{r\nu}=\dim\operatorname{span}\{v_{r\nu}(\epsilon):
 \epsilon\in\{\pm1\}^n\}.
\]

The uniform sign average is zero, so zero lies in the convex hull of these
endpoint vectors.  Caratheodory's theorem therefore gives signs and
nonnegative weights

\[
 m\le d_{r\nu}+1,
 \qquad w_s\ge0,
 \qquad\sum_sw_s=1,
 \qquad\sum_sw_sv_{r\nu}(\epsilon_s)=0.
\]

Because every sign outer product has the same diagonal, the weighted design
simultaneously preserves `diag Dhat=h_rnu`.  Hence

\[
\boxed{
 m\le d_{r\nu}+1
 \quad\Longrightarrow\quad
 \text{an exact finite-pattern endpoint-action design exists}.}
\tag{11.10g}
\]

The weights are carried in the signed CP amplitudes.  Their sums are still
`+1` and `-1`, so the balanced four-ray Star leakage cancels exactly.

There is a sharper simultaneous theorem than using joint tuples.  Let a block
contain `g` sign sides, write `sigma_j` for their source signs, and let `d_G`
be the dimension of their combined endpoint span.  Augment each **individual
side-pattern** by a side-mass coordinate,

\[
 \widetilde v_{j,\epsilon}
 =\left(\sigma_jv_j(\epsilon),e_j\right)
 \in{\cal H}_V\oplus\mathbb R^g.
\]

The point `(0,g^{-1}1)` lies in the convex hull of these augmented atoms.
They lie in an affine space of dimension at most `d_G+g-1`, so Caratheodory
uses at most `d_G+g` atoms total.  Rescaling its convex weights by `g` makes
the weights on every side sum to one while their signed endpoint errors sum to
zero.  Therefore

\[
\boxed{
 M_{\rm patterns}\le d_G+g,
 \qquad
 K_{\rm exact\ action}\le4(d_G+g)+r_{\rm ctrl}.}
\tag{11.10h}
\]

This is much smaller than `g(d_G+1)` joint tuples.  One signed mode has
`g=2,r_ctrl=0`, so exact endpoint action fits a free cap 2,048 whenever

\[
 \boxed{d_G\le510.}
\]

For `R` signed modes in one block (`g=2R`) the sufficient threshold is

\[
 \boxed{d_G\le512-2R.}
\]

A one-sign mode with its two-ray control gives the same `d<=510` ceiling.
These are pre-incumbent bounds; retained CP columns reduce them exactly
through (11.5).

This theorem annihilates q1111 only through the declared complete endpoint
map; it does not reconstruct the tensor.  It is therefore lawful only for a
frozen target/portfolio whose endpoint map is itself a valid deployment
object.  A corpus-fitted design with free per-net amplitudes is not a transfer
receipt.  Promotion requires explicit signs/weights, finite-precision error,
the post-incumbent cap, and the same held-out/shared-law discipline as every
target-conditioned carrier.

This only sizes the sketch.  The frozen seed must pass the deterministic
receipt

\[
 \boxed{\|\widehat z-z\|_2\le\eta_{\rm req}.}
 \tag{11.11}
\]

### 11.5 Cap displacement and value are one exchange, not two independent gates

Let `Q` be the incumbent CP/AQ groups displaced to make the column budget in
(11.5), and keep `E` as the arena with `Q` removed.  Their lost conditional
value is

\[
 \Delta_Q=d_Q^TS_Q^+d_Q.
\]

For the exact lawful residual target `f_res`, the required candidate gain is

\[
 g_{\rm req}=\Delta_Q+2.86\times10^{-10}.
\]

If `Delta(f_res|E)<=g_req`, no target-rank or sketch allocation can promote.
Otherwise the exact Schur robustness theorem above gives the binding error
radius

\[
 \boxed{
 \eta_{\rm req}=
 \min\{3.80\times10^{-6},\eta_\star(g_{\rm req})\}.}
 \tag{11.12}
\]

Equations (11.5), (11.10)--(11.12), followed by the frozen deterministic
receipt and the literal total-work ledger, are the complete promotion gate.
For a purely additive true-minus-owned residual with no displaced groups,
`Delta_Q=0`.  For a CP implementation replacing either the carried-X q4
provider or the fused AQ surface oracle, that literal owner's lost conditional
value belongs in `Delta_Q`; comparing the new absolute action to zero would
overstate value.

## Graph-ready split for checkpoint 11

### `H4_Q211_RESIDUAL_OWNERSHIP`

- **Status:** seam-dependent ownership solved; residual or replacement branch
  open.
- **Promotion:** declare `Y_owned=X` without the fused H211 oracle and
  `Y_owned=Y_AQ` only with it; source is
  `R_old[kappa4(Y_true)-kappa4(Y_owned)]`, or the package removes the literal
  provider whose action it replaces.
- **Close:** the purported feature factors through the q4 law already owned at
  that seam; D10 covariance alone is not an ownership receipt.

### `H4_Q211_FOUR_RAY_STAR_PROVENANCE`

- **Status:** exact same-site/one-transition algebra; multi-reset conditional.
- **Promotion:** aggregate row-space test (11.1), grouped four-column
  retention, and every required reset naturality identity (11.2); otherwise a
  measured endpoint gauge defect is charged.
- **Close:** actual dense reverse pair modes have nonzero Star-image residual
  or cross an unretained compensator reset.

### `H4_Q211_SIX_RAY_PERSISTENT_ACQUISITION`

- **Status:** algebraically exact / cap and total-work conditional.
- **Promotion:** six-column groups remain intact, their source amplitude has
  lawful residual provenance, dense modes pass strict collision sewing
  (11.15), and the liveness/cost ledgers pass.
- **Close:** CP displacement plus source/hook work exceeds value.

### `H4_Q211_BALANCED_FOUR_RAY_SKETCH`

- **Status:** solved exact grouped-CP algebra / endpoint value open.
- **Promotion:** positive and negative square averages share the same pair
  legs and are retained as one signed CP group, or a one-sign average retains
  its two-ray `G4(0)` control; strict collision sewing, liveness, and frozen
  endpoint error all pass.
- **Exact count:** `4(m_++m_-)` for two sign sides; `4m+2` for one sign side.
- **Close:** the implementation cannot retain signed group weights/controls,
  pair legs differ across the allegedly canceling sides, or cap/value fails.

### `H4_Q211_TARGET_RANK_CAP_EXCHANGE`

- **Status:** exact theorem (11.8); candidate-specific inputs open.
- **Promotion:** report the literal residual decomposition, each mode's
  grouped-balanced/control or standalone provenance label and lifetime,
  endpoint Gram, sign-side `A_rnu`, integer allocation, retained/dropped CP
  groups, and peak cap.
- **Close:** even the relaxed bound (11.9) exceeds the error radius or no
  positive integer allocation fits (11.5).

### `H4_Q211_SIGN_SKETCH_ENDPOINT_ERROR`

- **Status:** exact seed-average law / frozen conditioning open.
- **Promotion:** q1111 is propagated through the complete map in (11.6), the
  preregistered sizing gate (11.10) passes, and the frozen seed passes (11.11).
- **Close:** endpoint off-anchor sensitivity forces error above (11.12) at
  every cap-feasible allocation.

### `H4_Q211_CROSS_MODE_SIGN_DESIGN`

- **Status:** solved variance identity / frozen design open.
- **Promotion:** a preregistered common-sign block and diagonal conjugations
  reduce the complete endpoint energy (11.7b), preserve the diagonal target
  and CP grouping, and pass the frozen error/value gate.
- **Close:** the minimum signed joint Gram equals the independent-mode sum or
  its attainable reduction is below the required error margin.

### `H4_Q211_SPARSE_HADAMARD_EXACTNESS`

- **Status:** solved constructive support gate.
- **Promotion:** the frozen anchor support admits an `m by s` sign frame with
  `E^TE=mI`, and its balanced/control CP count plus incumbents fits cap.
- **Close:** `m<s` by rank, or the smallest certified sign-frame order exceeds
  the post-incumbent column budget.

### `H4_Q211_ENDPOINT_CARATHEODORY_DESIGN`

- **Status:** solved finite-target existence theorem / lawful construction
  open.
- **Promotion:** the complete deployment-valid endpoint error span has
  dimension `d_G` small enough that an explicit augmented Caratheodory design
  with at most `d_G+g` total side-patterns fits cap, transfers under a frozen
  shared law, and has finite-precision error below (11.12).
- **Close:** only a corpus-overfit/free-per-net endpoint map gives the small
  dimension, or `4(d_G+g)+r_ctrl` exceeds the post-incumbent cap.

### `H4_Q211_CP_DISPLACEMENT_VALUE`

- **Status:** open.
- **Promotion:** approximate candidate gain after dropping `Q` exceeds its
  lost Schur value by at least `2.86e-10`, with error below (11.12), and total
  incremental work stays below the 49-GEMM ceiling.
- **Close:** exact residual gain does not exceed `Delta_Q+2.86e-10`.

## 11.6 Actual reverse-mode verdict: the economical dense frame is six-ray,
## unless a candidate-specific Star polynomial vanishes

The target-conditioned native reverse slices use a common frame

\[
 \Lambda_i=U\Theta_iU^T.
\]

Their economical pair modes are columns `u_a,u_b` of `U`, not literal
coordinate axes.  Let

\[
 {\cal S}_{31}=\operatorname{span}
 \{x_i^3x_j:1\le i,j\le n\}
\]

be the polynomial image of the coordinate Star4 provider.  For one aggregate
reverse slice, the exact four-ray provenance test is

\[
\boxed{
 \Pi_{{\cal S}_{31}^{\perp}}
 \sum_{a,b}(\Theta_i)_{ab}
 \left[(u_a^Tx)^3(u_b^Tx)+(u_a^Tx)(u_b^Tx)^3\right]=0.}
\tag{11.13}
\]

Equation (11.13) includes the possible cancellations across modes that the
per-atom test omits.  It is a finite set of polynomial equalities: every
coefficient in the `22`, `211`, and `1111` coordinate partitions of the
displayed leakage must vanish.

There is a sharp frame-uniform theorem.  If (11.13) must hold for an open set
of diagonal `Theta_i`, then selecting one diagonal coefficient at a time
forces

\[
 (u_a^Tx)^4\in{\cal S}_{31}
 \quad\hbox{for every }a.
\]

If `u_a` has two nonzero coordinates `p,q`, this fourth power has the forbidden
coefficient `6u_{pa}^2u_{qa}^2 x_p^2x_q^2`.  Therefore every column is
one-sparse; for orthonormal `U`, it is a partial signed coordinate
permutation.  Conversely such a frame passes the same-seam Star test.  Hence

\[
\boxed{
 \text{four-ray provenance uniformly over reverse amplitudes}
 \iff U\text{ is a partial signed coordinate frame}.}
\tag{11.14}
\]

Generic dense reverse transport violates (11.14).  A single frozen dense
`Theta_i` could still pass by exceptional aggregate cancellation, but only a
literal coefficient/projector receipt for (11.13) can claim it.  No
common-support, SVD-tail, commuting-Theta, or low-`k` statement implies that
receipt.  Therefore the current conservative classification of the actual
economical target-conditioned reverse modes is **six-ray**, while four rays
remain available for literal coordinate/Star modes and certified
one-transition relays.

Finally, six-ray polarization represents the full structured word
`Sym(A,A,B,C)`.  For dense `A,B,C` it is not by itself the strict external
q211 collision projection.  Exact forward use additionally requires

\[
\boxed{
 P_{\rm complete}^{\perp}{\cal L}
 (I-\Pi_{211}^{\rm coord})\operatorname{Sym}(A,A,B,C)=0,}
\tag{11.15}
\]

or the synchronized K4/K31/K22 collision subtraction that makes (11.15)
true.  If not, its endpoint vector is `g_r^coll` in (11.8).  Literal distinct
coordinate triples make this defect zero.  For the sign-averaged mixed anchor,
the remaining cross-anchor sectors are stochastic and are already included in
`A_r`; they are not discarded under (11.15).

**Verdict:** absent a candidate-specific (11.13) receipt, a standalone dense
reverse atom is six-ray.  The sign-averaged carrier is the important
exception: its positive/negative four-ray groups self-cancel the Star leakage,
and a one-sign average needs only one two-ray aggregate control.  Price that
group by (11.3), not at six rays per sample.  In every case retain the
collision-sewing term.  A literal coordinate/Star one-transition source is
the other lawful four-ray path.

## Checkpoint 12: a middle-catalecticant lower bound makes exact full-rank anchor transport cap-tight before incumbents

The preceding `4q+r_ctrl` formula is a constructive count for a rank-`q`
anchor-square approximation.  A representation-independent lower bound shows
why an exact full-rank anchor mask is not a plausible cap loophole.

Let

\[
 p(x)=q_D(x)\,l(x)\,m(x),
 \qquad q_D(x)=x^TDx,
\]

with `D` nonsingular and `l,m` independent.  Every signed symmetric CP
decomposition `p=sum_s lambda_s(v_s^Tx)^4` has at least the rank of the middle
catalecticant

\[
 C_p:\operatorname{Sym}^2V\longrightarrow\operatorname{Sym}^2V^*,
 \qquad C_p(u\odot v)=\partial_u\partial_vp.
\]

The product rule gives

\[
\begin{aligned}
\partial_u\partial_vp={}&
2D(u,v)lm\\
&+2(Du\cdot x)\{l(v)m+m(v)l\}
 +2(Dv\cdot x)\{l(u)m+m(u)l\}\\
&+q_D\{l(u)m(v)+m(u)l(v)\}.
\end{aligned}
\tag{12.1}
\]

Hence

\[
 \operatorname{im}C_p
 \subseteq lV^*+mV^*+\operatorname{span}\{q_D\},
\]

whose dimension is at most `2n`.  This upper bound is attained.  In
coordinates `l=x_1,m=x_2`, take

\[
 q_D=2x_1x_2+\sum_{a=3}^n x_a^2.
\]

Then the second derivatives span independently

\[
 x_1x_a, x_2x_a (a=3,\ldots,n),quad
 x_1^2, x_2^2, x_1x_2,quad
 \sum_{a=3}^n x_a^2,
\]

exactly `2n` quadratics.  A nonzero `2n` minor is a polynomial in `D,l,m`, so
the same rank holds on a Zariski-open, hence almost-everywhere, set.  Therefore

\[
\boxed{
 \operatorname{rank}_{\rm Waring,\mathbb R}(q_Dlm)
 \ge\operatorname{rank}C_p=2n
 \quad\text{generically}.}
\tag{12.2}
\]

At width `n=1024`, an exact generic full-rank anchor-square word needs at least
2,048 signed CP rays: the entire incumbent cap, before retaining one old ray,
another live layer, or any control.  Equality of the lower bound with an
actual real decomposition is not guaranteed, so 2,048 is optimistic.  This
does not apply to the low-rank randomized `Dhat`; it proves that finite-rank
approximation plus the endpoint error gate is essential, rather than a
convenience.

The rank-`q` refinement is also sharp.  From (12.1), for `U=range(D)`,

\[
 \operatorname{im}C_p
 \subseteq lU+mU+\operatorname{span}\{lm,q_D\},
\]

so its dimension is at most `min(2q+2,2n)`.  For `q<=n-2`, the witness

\[
 q_D=\sum_{a=3}^{q+2}x_a^2,qquad l=x_1,quad m=x_2
\]

has second-derivative basis

\[
 \{x_1x_a,x_2x_a\}_{a=3}^{q+2}
 \cup\{x_1x_2,q_D\},
\]

of size `2q+2`.  Rank `n-1` is witnessed by
`q_D=(x_1+x_2)^2+sum_(a>=3)x_a^2`, and rank `n` by the preceding example.
Therefore, generically on every fixed-rank stratum,

\[
\boxed{
 \operatorname{rank}C_{q_Dlm}=\min\{2q+2,2n\}.}
\tag{12.3}
\]

For a frozen sign sketch of matrix rank `q`, (12.3) is a necessary CP-column
count.  The controlled four-ray construction uses `4q+r_ctrl`, giving a
factor-of-two gap between the generic catalecticant lower bound and the
current constructive upper bound.  A claimed improved compression must land
inside that gap with an explicit real Waring decomposition; energy or matrix
rank alone is not such a construction.

### `H4_Q211_FULL_ANCHOR_CATALECTICANT_CAP`

- **Status:** solved generic lower bound.
- **Claim:** exact generic full-rank `D_h` structured transport has real CP
  rank at least `2n=2048`.
- **Promotion of an exact exception:** exhibit a non-generic catalecticant rank
  below the free post-incumbent cap and an explicit real CP decomposition,
  then still pass collision sewing and value exchange.
- **Close exact branch:** the actual anchor word has rank-2,048 middle
  catalecticant while any incumbent/live-factor rank must remain.

## Checkpoint 13: fail-fast order for the residual H4/q211 CP branch

The mathematical gates above imply a strict decision order; no long carrier
construction is justified before these receipts.

1. **Synchronize ownership.**  Declare `Y_owned=X` or `Y_AQ` from the literal
   fused-oracle tape and form only
   `R_old[kappa4(Y_true)-kappa4(Y_owned)]`.
2. **Value before representation.**  Compute the exact residual target's
   complete-old Schur ceiling.  If it does not exceed
   `Delta_Q+2.86e-10`, close the branch before choosing CP rays.
3. **Sew strict collisions.**  Literal coordinate triples pass directly;
   dense modes need (11.15).  An unproved dense collision projection is not a
   source.
4. **Try exact endpoint action first.**  If the deployment-valid augmented
   endpoint dimension satisfies `4(d_G+g)+r_ctrl<=K_free`, use the explicit
   Caratheodory design and remove sketch variance entirely.  Sparse
   Walsh--Hadamard exactness is the stronger tensor-level special case.
5. **Apply deterministic lower bounds before random signs.**  Use the CP
   catalecticant count, anchor spectral tail (11.10c), and joint endpoint
   secant (11.10d).  If their error exceeds (11.12), close immediately.
6. **Only then allocate a stochastic bundle.**  First test the signed-matrix
   generator (14.1)--(14.6) when sampled squares share pair legs; otherwise
   use balanced/control CP with target-rank rule (11.9b)--(11.9c).  In either
   state use Bernstein (11.10b) or a preregistered cross-mode sign design
   (11.7b).
7. **Freeze and exchange.**  The literal frozen error must pass (11.11), CP
   liveness must pass (11.5), total work must pass 49 GEMMs, and approximate
   Schur value after displacement must clear (11.12).

This order separates algebraic impossibility, exact finite-action design, and
stochastic approximation.  It is the fail-fast replacement for treating a
low abstract q211 rank as evidence that a deployable carrier exists.

## Checkpoint 14: factor the anchor matrix directly instead of expanding every square into CP4 rays

The balanced four-ray construction is not the smallest structured state when
many sampled anchor squares share the same singleton pair `(B,C)`.  Let

\[
 D=F\Lambda F^T
 =\sum_{s=1}^q\lambda_sA_sA_s^T,
 \qquad d=\operatorname{diag}D,
\]

and carry the single structured polynomial

\[
 p(x)=12(x^TDx)(B^Tx)(C^Tx).
\]

Summing the exact rank-one three-generator faces gives

\[
\boxed{K4=12\,d\odot B\odot C,}
\tag{14.1}
\]

\[
\boxed{
K31=3\left[
(d\odot B)C^T+(d\odot C)B^T
+2D_{B\odot C}D
\right],}
\tag{14.2}
\]

and

\[
\boxed{
\begin{aligned}
K22=2\big[&d(B\odot C)^T+(B\odot C)d^T\\
&+2D_BDD_C+2D_CDD_B\big].
\end{aligned}}
\tag{14.3}
\]

Here `D_v=Diag(v)`.  These identities include every collision coefficient of
the full structured tensor.  Strict external q211 still requires the same
collision sewing (11.15); the formulas do not declare those faces novel.

For every common linear deployed factor map `T`, the state is exactly closed:

\[
 \boxed{
 F^+=TF,qquad B^+=TB,qquad C^+=TC,qquad
 D^+=TDT^T.}
\tag{14.4}
\]

Thus the sign sketch can be stored directly as one signed low-rank matrix,
with `q=m_++m_-`, rather than as `4q+r_ctrl` CP rays.  No Star leakage or
Waring control exists in this representation; the only approximation remains
`Dhat-D_h`, already measured by the endpoint gates above.

A conservative streamed rank-update census per live group is:

1. `q+2` column pushes for `(F,B,C)`;
2. `q+2` rank-one terms for K31; and
3. `2q+2` rank-one terms for K22.

Hence ten consecutive births, `55` live group-layers, cost

\[
\boxed{
 G_{D\text{-gen}}(q)
 ={55(4q+6)\over1024}
 =0.21484375q+0.322265625}
\tag{14.5}
\]

width-GEMM equivalents before source acquisition, collision hooks, reset
model error, and literal metering.  At `q=1` this reproduces the existing
three-generator count `550/1024=0.537109375`.  Shared `(B,C)` across several
anchor matrices can reuse their two column pushes and visible outer terms.

For a declared ten-layer incremental work allowance `G_slack`, the
rank-update census alone imposes

\[
 \boxed{
 q\le\left\lfloor{1024G_{\rm slack}/55-6\over4}\right\rfloor.}
\tag{14.5a}
\]

Thus a one-equivalent additive margin permits only `q<=3` before **any**
source/hook/meter charge (`q=3` already costs about `0.9668`), while a
21-equivalent replacement allowance permits `q<=96`.  The branch is therefore
a low-rank additive candidate or a broader replacement candidate, not a free
escape from the Base48 ledger.

If the transported structured word is completely consumed into synchronized
K4/K31/K22 after **one** dense transition and its factors are then retired,
there are only ten live group-layers, not 55.  The corresponding relay cost is

\[
 \boxed{
 G_{D\text{-relay}}(q)={10(4q+6)\over1024}
 =0.0390625q+0.05859375.}
\tag{14.5b}
\]

One additive equivalent then allows `q<=24` before extras.  This cheaper count
is lawful only with a literal next-seam complete consumer and no later
old-to-novel return; otherwise the persistent 55-layer count (14.5) binds.

Precisely, let `U_l` be the declared structured-source subspace, `A_l` its
one-transition map, `C_(l+1)` the synchronized old face consumer, and
`F_(l+1)` the complete future endpoint operator.  Consume-and-retire is exact
iff

\[
\boxed{
 \ker\left(C_{l+1}|_{A_lU_l}\right)
 \subseteq
 \ker\left(F_{l+1}|_{A_lU_l}\right),}
\tag{14.7}
\]

equivalently iff `F_(l+1) A_l` factors through `C_(l+1) A_l` on `U_l`.
Merely observing nonzero K4/K31/K22 after the dense transition does not prove
(14.7); collision faces can leave a future-visible all-distinct kernel.

For the matrix-generator family, however, a **full literal K31 plane** has an
exact left inverse after a generic dense transition.  Put

\[
 g_i=B_iC_i,qquad K=K31.
\]

From (14.2),

\[
 K_{ii}=12g_id_i,
\]

and hence, whenever every `g_i` is nonzero,

\[
\boxed{
 d_i={K_{ii}\over12B_iC_i},
 \qquad
 D_{ij}={K_{ij}/3-d_i(B_iC_j+C_iB_j)\over2B_iC_i}.}
\tag{14.8}
\]

Thus `D -> K31(D;B,C)` is injective for fixed known dense `(B,C)`, and the
complete structured tensor can be recovered before another transition.  In
this literal full-plane setting, the K31 consumer makes (14.7) hold on the
matrix-generator source subspace.  A Star4 projection, diagonal-only read, or
other compressed K31 interface does not inherit this theorem; it must prove
injectivity on the declared `D` subspace separately.

The inverse can be ill-conditioned.  With

\[
 \gamma=\min_i|B_iC_i|,
\]

a direct Frobenius bound is

\[
\boxed{
 \|\delta D\|_F
 \le\left[
 {1\over6\gamma}
 +{\|B\|_\infty\|C\|_2+\|C\|_\infty\|B\|_2
 \over24\gamma^2}
 \right]\|\delta K\|_F.}
\tag{14.9}
\]

Therefore exact factor-through and numerically useful factor-through are
separate nodes.  Diffuse dense vectors have no algebraic zeros but can have a
small `gamma`; the endpoint error budget must charge this amplification.

Peak factor memory for `L_live` equal-rank births is

\[
 4nL_{\rm live}(q+2)\ \text{bytes}.
\]

Under one `n^2`-float state allowance this requires

\[
 \boxed{L_{\rm live}(q+2)\le n.}
\tag{14.6}
\]

At `n=1024,L_live=10`, it permits `q<=100` before all other state, versus the
free CP cap condition `4qL_live<=2048`, namely `q<=51`, for the balanced CP
expansion.  Which branch wins after the existing AQ/D11 memory and work ledger
is a literal deployment question, but the matrix-generator branch strictly
removes the CP-column displacement term.

### `H4_Q211_SIGNED_MATRIX_GENERATOR`

- **Status:** solved exact structured transport/face algebra; source, strict
  collision, memory, and value open.
- **Promotion:** sign-sketch factors share frozen `(B,C)`, (14.1)--(14.4) are
  consumed exactly once, collision sewing passes, actual memory/work fits,
  relay pricing additionally passes (14.7), and the same frozen endpoint
  error/Schur gates pass.
- **Close:** pair legs are not shared, factor rank/lifetime exceeds (14.6), or
  hooks/source work removes its advantage over CP expansion.

### `H4_Q211_K31_RELAY_LEFT_INVERSE`

- **Status:** solved for a literal full K31 plane / compressed interface open.
- **Promotion:** known transported `(B,C)` have nonzero componentwise product,
  the consumer exposes the full synchronized K31 plane, and the conditioning
  bound (14.9) fits the endpoint tolerance.
- **Close cheap relay:** the actual consumer is a noninjective Star/diagonal
  projection on the declared `D` subspace, or inverse amplification exhausts
  the error budget.

## 15. Literal D11 interface audit: K31 is transiently visible, but the
## D-generator is not currently retained

This is a read-only provenance audit of
`p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py`; it is not a
proposal to edit the deployed estimator.

The literal carrier has the following sequence.

1. `_connected_planes` (beginning at line 624) materializes a full `n x n`
   connected `k31` plane in `fusion_context[17][9]`.
2. `_cp4_read_planes` (line 998) reads the active symmetric CP4 provider into
   synchronized `(K4,K31,K22)` faces.
3. `_assemble_quartic_provider` (line 1204; invoked near line 1767) adds those
   provider faces exactly once to the host faces before `_full_relu_reset`.
4. `_full_relu_reset` (line 1398) passes all of `pre_k4,pre_k31,pre_k22` into
   the Q24/Edgeworth consumer.
5. `_cp4_reset` (line 1030) gates the retained CP4 provider and returns its
   post-reset faces.
6. `_reset_stars` (line 1226; invoked near line 1848) calls
   `_quartic_residual` and subtracts the active post-reset provider exactly
   once before fitting the scalar pairing coefficient and generic residual
   `quartic` matrix.
7. On the next transition `_push_quartic` (line 1138) propagates that residual
   matrix with the generic Star4 formulas.  It does **not** apply the
   D-generator inverse (14.8), retain `(D,B,C)`, or use D-generator transport
   (14.4).

Consequently, the full K31 plane needed by (14.8) exists as a transient, but
the left inverse is only an implementable hook, not an already-owned/free
relay.  Decoding `D` and then dropping it into the current residual `quartic`
matrix would be invalid: generic Star4 propagation does not preserve the
structured polynomial `12(x^TDx)(B^Tx)(C^Tx)`.

There is, however, an exact interface-compatible implementation theorem.  Let
`G=Diag(gate)` and retain a provider state `(F,Lambda,B,C)` with
`D=F Lambda F^T`.  Across a dense preactivation map `T`, use (14.4).  Across
the first-order ReLU reset, use

\[
 \boxed{F^+=GF,\qquad B^+=GB,\qquad C^+=GC,
 \qquad D^+=GDG.}
\tag{15.1}
\]

Equations (14.1)--(14.3) then give, entrywise,

\[
 K4_i^+=g_i^4K4_i,\qquad
 K31_{ij}^+=g_i^3g_jK31_{ij},\qquad
 K22_{ij}^+=g_i^2g_j^2K22_{ij}.
\tag{15.2}
\]

These are exactly the inherited-provider gate powers implemented for CP4 in
`_cp4_reset`.  Therefore a D-generator can use the existing **semantic**
provider seam without converting to Star4:

\[
 \begin{array}{c}
 \text{read (14.1)--(14.3)}\to
 \text{add once before Q24}\to
 \text{gate by (15.1)}\to
 \text{subtract once before residual-Star fit}.
 \end{array}
\tag{15.3}
\]

If several providers are active, the seam is linear: add the sum of all
provider faces before Q24 and subtract the same synchronized post-reset sum
before the residual fit.  This proves exact no-double-counting for the
transported provider **conditional on a lawful birth**.

The audit does not provide that birth.  The current CP4 reset appends only an
exact coordinate-diagonal K4 correction, while the D-generator requires a
declared true-minus-owned q211 residual source.  Thus its remaining necessary
source gate is

\[
 \boxed{
 \Gamma_{D,l}
 =R_{\rm old,l}\Pi_{211}
 [\kappa_4(Y_{\rm true,l})-\kappa_4(Y_{\rm owned,l})]
 \in \{12(x^TDx)(B^Tx)(C^Tx)},}
\tag{15.4}
\]

with the ownership convention of (11.5).  Reading the already assembled
total K31 and calling it a new source would duplicate owned Q24/CP4/Star
content.  A residual-surface oracle, a replacement of an existing owner, or a
certified factor-through acquisition is still required.

### `H4_Q211_D11_PROVIDER_SEAM`

- **Status:** exact add/gate/subtract seam proved; not presently implemented
  for the D-generator; birth remains open.
- **Promotion:** a frozen lawful residual source satisfies (15.4), its factors
  are retained separately from generic Star4, (14.1)--(14.4) and (15.1) are
  used, and literal metering plus endpoint error gates pass.
- **Close as free/currently owned:** no such source/hook exists; the transient
  K31 plane alone does not mean the D-generator is retained.

### `H4_Q211_K31_RELAY_LITERAL_INTERFACE`

- **Status:** current D11 exposes full K31 transiently but does not decode or
  retain the structured generator.
- **Promotion:** implement the left inverse (14.8) at the synchronized seam,
  retain known `(B,C)` and `D` under (14.4)/(15.1), and prove no collision or
  source-ownership duplication.
- **Close cheap decode-into-Star route:** unconditional; `_push_quartic` has
  the wrong propagation law for decoded D-generator content.

## 16. Exact AQ pair-feature square lift: q4 is a three-pairing Gram, not an
## independent tensor once the full quadratic core is known

Use the half-scaled convention throughout this section:

\[
 H_i=u_i^Tz+Q_{A_i}(z),\qquad
 Q_A={1\over2}(z^TAz-\operatorname{tr}A),qquad A_i=A_i^T.
\tag{16.1}
\]

Define the symmetric bilinear pair feature

\[
 \boxed{
 \Psi_{ij}=left(A_i\circ A_j,\ A_i u_j+A_j u_i\right)
 \in\operatorname{Sym}_r\oplus\mathbb R^r,
 \qquad A\circ B={AB+BA\over2}.}
\tag{16.2}
\]

Its ambient dimension is

\[
 \boxed{D={r(r+1)\over2}+r={r(r+3)\over2}.}
\tag{16.3}
\]

Then the fully polarized fourth cumulant is exactly

\[
\boxed{
\begin{aligned}
 \kappa_4(H_i,H_j,H_k,H_l)
 ={}&\langle\Psi_{ij},\Psi_{kl}\rangle
   +\langle\Psi_{ik},\Psi_{jl}\rangle\\
  &+\langle\Psi_{il},\Psi_{jk}\rangle .
\end{aligned}}
\tag{16.4}
\]

A short proof fixes all normalizations.  For
`A(t)=sum_i t_i A_i` and `u(t)=sum_i t_i u_i`, the centered Gaussian
log-mgf is

\[
 -{1\over2}\operatorname{tr}A
 -{1\over2}\log\det(I-A)
 +{1\over2}u^T(I-A)^{-1}u.
\tag{16.5}
\]

Its homogeneous degree-four part is

\[
 {1\over8}\operatorname{tr}(A^4)+{1\over2}u^TA^2u.
\tag{16.6}
\]

Polarizing (16.6) gives (16.4).  In particular,

\[
 \boxed{
 \kappa_4(H)=3\|\Psi(H,H)\|^2
 =3\operatorname{tr}(A^4)+12u^TA^2u.}
\tag{16.7}
\]

For the unscaled convention `I2(A)=z^TAz-tr(A)`, put

\[
 \Phi_{ij}=left(A_i\circ A_j,
 {A_i u_j+A_j u_i\over2}\right).
\]

The right side of (16.4) is then multiplied by `16`.  This is the convention
behind the earlier pure-NCQC formula

\[
 q_{\perp}=16\left[
 \langle(I-P)Y_{ii},(I-P)Y_{jk}\rangle
 +2\langle(I-P)Y_{ij},(I-P)Y_{ik}\rangle\right].
\tag{16.8}
\]

### Repeated q211 and the exact residual split

The external repeated-index face is the immediate specialization

\[
\boxed{
 \kappa_4(H_i,H_i,H_j,H_k)
 =\langle\Psi_{ii},\Psi_{jk}\rangle
 +2\langle\Psi_{ij},\Psi_{ik}\rangle .}
\tag{16.9}
\]

Thus the AQ quartic polynomial is a sum of `D` squares of quadratic pair
features: if `P_(ij)=Psi_ij`, one pairing is the PSD Gram
`G_(ij),(kl)=<P_(ij),P_(kl)>`, of rank at most `D`, and q4 is the sum of its
three index reshufflings.  For `r=55`, `D=1595`.

If an owned source is literally the orthogonal feature subspace `P`, then
replacing every `Psi` in (16.9) by `(I-P)Psi` gives its exact perpendicular
Gram residual.  This is the linear-core extension of (16.8).  For a generic
old consumer, however, residualization must use its actual Schur projector;
one may not replace that projector by a convenient latent feature projector.

This identity sharpens, but does not remove, the inverse-fiber obstruction.
The lower moments are

\[
 C_{ij}=u_i^Tu_j+{1\over2}\operatorname{tr}(A_iA_j),
\tag{16.10}
\]

\[
 \kappa_3(H_i,H_i,H_j)
 =\operatorname{tr}(A_i^2A_j)
  +u_i^TA_j u_i+2u_i^TA_i u_j.
\tag{16.11}
\]

They need not identify the Gram of (16.2).  Two `(C,K21)`-equivalent cores
can therefore have different (16.9); the previously derived mixed-frame and
restricted-minor gates are exactly tests of the missing pair-feature Gram,
not evidence for an independent arbitrary fourth-order tensor.

There is also an exact H4 bridge and a more precise interface caveat.  In the
pure quadratic case, `Psi_ij=A_i circ A_j` is the connected-cycle contraction
of the H4 square lift.  The full pure-H4 object

\[
 U_{A,B}=Q_AQ_B-{1\over2}\operatorname{tr}(AB)-2Q_{A\circ B}
\]

has the exact Gram identity

\[
\boxed{
\begin{aligned}
 \mathbb E[U_{A,B}U_{C,D}]
 ={}&{1\over3}\kappa_4(Q_A,Q_B,Q_C,Q_D)\\
 &+C_Q(A,C)C_Q(B,D)+C_Q(A,D)C_Q(B,C),
\end{aligned}}
\tag{16.11a}
\]

where `C_Q(A,C)=E[Q_AQ_C]=tr(AC)/2` and the cumulant on the right is the
**pure-matrix** part of (16.4).  For squares this reduces to

\[
 \kappa_4(Q_M,Q_M,Q_N,Q_N)
 =3\mathbb E[U_MU_N]-{3\over2}\operatorname{tr}(MN)^2.
\tag{16.11b}
\]

Equation (16.11a) follows immediately from the square-square Gram formula at
the start of this lane and polarization in `(A,B)` and `(C,D)`.  Therefore
the **complete four-index pure q4 tensor plus quadratic covariance** determines
the entire Gram of the H4 pair module.  There is no extra information-theoretic
H4-square norm once both complete objects are known.

The deployed-interface caveat remains: D11 materializes only K4/K31/K22
collision faces, and evaluating only (16.9) materializes only a repeated
q211 action.  Neither is the complete four-index object in (16.11a), so a
future-visible q1111 kernel can remain.  Conversely, a full resident AQ core
can evaluate the missing all-distinct entries analytically through (16.4),
but their action and transport must be charged.  The vector leg `b_ij` belongs
to the mixed linear-quadratic cumulant and is not part of the pure H4 Gram in
(16.11a).  Thus any restricted-interface miss remains a separate
`g_coll`/model-error term, while a claimed information-theoretic miss must be
tested against the complete right side of (16.11a).

### Exact local reverse action without physical CP rays

Let `U in R^(n x k)` be a target-conditioned singleton frame and set

\[
 M_a=\sum_jU_{ja}A_j,\qquad v_a=\sum_jU_{ja}u_j.
\tag{16.12}
\]

For symmetric reverse slices `Theta_i in Sym_k`, (16.9) gives the exact local
AQ q211 action

\[
\boxed{
 \rho_{AQ}=\sum_{i,a,b}(\Theta_i)_{ab}
 \left[
 \langle\Psi_{ii},\Psi_{ab}\rangle
 +2\langle\Psi_{ia},\Psi_{ib}\rangle
 \right],}
\tag{16.13}
\]

where `Psi_ab=Psi((M_a,v_a),(M_b,v_b))` and similarly for `Psi_ia`.
This evaluates the Bellman contraction in latent matrices and vectors; it
does not require a six-ray physical CP expansion.

For the pure quadratic part, put `P_ia=A_iM_a`.  Its bracket reduces to

\[
\boxed{
 2\langle P_{ia},P_{ib}\rangle_F
 +\langle P_{ia}^T,P_{ib}\rangle_F.}
\tag{16.14}
\]

This is exactly `1/16` of the unscaled `I2` normalization
`32<P_ia,P_ib>+16<P_ia^T,P_ib>`.  The linear-core addition is

\[
\begin{aligned}
 &2\langle A_i u_i,M_a v_b+M_b v_a\rangle\\
 &\quad+2\langle A_i v_a+M_a u_i,
                    A_i v_b+M_b u_i\rangle .
\end{aligned}
\tag{16.15}
\]

If one reverse slice is diagonalized
`Theta_i=sum_s lambda_is t_is t_is^T` and
`(N_is,w_is)=sum_a(t_is)_a(M_a,v_a)`, its contribution is

\[
\boxed{
 \sum_s\lambda_{is}\left[
 \langle\Psi_{ii},\Psi_{N_{is}N_{is}}\rangle
 +2\|\Psi_{i,N_{is}}\|^2\right].}
\tag{16.16}
\]

This is the exact target-rank bridge: six/four rays price a physical forward
carrier, while (16.13)--(16.16) price a resident-core reverse contraction.
Low singleton width `k` alone is not low anchor-square rank: generic
`Theta_i` varies with `i`, so a physical repeated-anchor CP decomposition can
still have rank `n`.  The resident-core formula avoids that Waring-rank
obstruction, but must pay its matrix-product census.

Ignoring constant-factor symmetric savings, a transparent leading-work
census per active layer is

\[
 \boxed{
 G_{AQ\text{-}act}(k)
 \gtrsim {k r^3\over n^2}+{k^2r^2\over n^2},}
\tag{16.17}
\]

where the first term forms `P_ia` and the second contracts dense `Theta_i`
against their Grams.  If the exact same `P_ia` products are already resident
for another lawful NCQC/H3 consumer, only the second term is incremental.
At `n=1024,r=55,L=10`, the unit-coefficient arithmetic normalization is

\[
 L G_{AQ\text{-}act}(k)
 \gtrsim 1.5867k+0.02885k^2.
\tag{16.18}
\]

Therefore `k=21` is about `46.0` equivalents before source/reset/hooks if
products are recomputed, but about `12.7` if the full product bank is truly
reused.  Under a 21-equivalent replacement allowance, (16.18) permits only
about `k<=11` without reuse, while the contraction-only term permits about
`k<=26`.  These are arithmetic gates, not billing receipts; exact resident
provenance and literal metering bind.

There is an exact slice-rank refinement.  Suppose every reverse slice has
rank at most `s` and use (16.16) directly rather than first forming all
`P_ia`.  For each anchor, form its `s` combinations `(N_is,w_is)` and only
their `s` products `A_iN_is`.  The leading census becomes

\[
\boxed{
 G_{AQ\text{-}act}(k,s)
 \gtrsim {s r^3\over n^2}+{sk r^2\over n^2},
 \qquad s=\max_i\operatorname{rank}\Theta_i.}
\tag{16.18a}
\]

It recovers (16.17) at `s=k`, while a genuinely rank-one slice at
`n=1024,r=55,k=21,L=10` has the unit-coefficient arithmetic count about
`2.19`, before aggregation, hooks, and reset.  This is a real loophole only
if the **actual anchor-indexed reverse slices** have low rank.  A common
singleton frame of width `k` does not imply `s<<k`, and independently low
slice rank still does not give a low physical anchor-square Waring rank.

The exact reset recurrence shows why low `s` is not the default.  Let a reset
pair kernel have a PSD/Kraus decomposition

\[
 D_i=\sum_t v_{it}v_{it}^T,
 \qquad \Lambda_i=U\Theta_iU^T.
\]

Then its Schur action is

\[
\boxed{
 D_i\circ\Lambda_i
 =\sum_t[D_{v_{it}}U]\Theta_i[D_{v_{it}}U]^T.}
\tag{16.18f}
\]

Even preservation of the common frame `U` requires the invariant-frame
identities

\[
 \boxed{D_{v_{it}}U=UR_{it}\quad\text{for every live }(i,t).}
\tag{16.18g}
\]

Under (16.18g), the small slice recurrence is the completely positive map

\[
 \boxed{\Theta_i^+=\sum_tR_{it}\Theta_iR_{it}^T.}
\tag{16.18h}
\]

Starting from a rank-one terminal slice `Theta=theta ww^T`, its rank after
one reset is generically

\[
 \operatorname{rank}\Theta^+
 =\dim\operatorname{span}\{R_{it}w:t\},
\tag{16.18i}
\]

and it stays rank one iff all nonzero `R_it w` are collinear.  Dense anchor
mixing then takes linear sums of the resulting `Theta_j`; those sums are
generically full on the retained common frame unless every slice remains in
one matrix ray (or in a separately proved `s`-dimensional terminal cyclic
module).  Therefore one terminal word, one Chow atom, or one initially
rank-one mask is not a low-`s` receipt.  In the absence of (16.18g)--(16.18i)
plus dense-mixing closure, the conservative cost gate uses `s=k`.

If slices are truncated, the correct error is endpoint weighted, not the
discarded eigenvalue norm alone.  Let `e` index the complete endpoint family,
let `K_i^(e) in Sym_k` be the bracket matrix in (16.13), and define

\[
 ({\cal B}\Delta\Theta)_e
 =\sum_i\langle\Delta\Theta_i,K_i^{(e)}\rangle_F.
\tag{16.18b}
\]

For the endpoint weight `V`, the exact connected-action truncation error is

\[
\boxed{
 E_{conn}=\|V^{1/2}{\cal B}(\Theta-\widehat\Theta)\|_2.}
\tag{16.18c}
\]

A source-independent sufficient bound is

\[
 E_{conn}\le
 \|V^{1/2}{\cal B}\|_{F\to2}
 \left(\sum_i\|\Theta_i-\widehat\Theta_i\|_F^2\right)^{1/2},
\tag{16.18d}
\]

but promotion should use (16.18c) on the frozen masks.  Without a q1111
darkness theorem, the total error is instead

\[
\boxed{
 E_{tot}=\|V^{1/2}[{\cal B}(\Theta-\widehat\Theta)+g_{coll}+e_{model}]\|_2
 \le\eta_{req}.}
\tag{16.18e}
\]

This is the direct target-rank/work analogue of the CP cap program (11.8):
the local AQ route replaces CP columns by slice-rank matrix products, but it
does not erase collision or model bias.

### The linear-core pair feature is not automatically H3/K21-owned

The vector leg

\[
 b_{ij}:=A_i u_j+A_j u_i
\tag{16.18j}
\]

in (16.2) is absent from the pure-NCQC formula (16.14).  It is not determined
by a scalar/repeated K21 tape, and even a symmetric H3 tape does not generally
disentangle it.

There is an exact oriented-provider factorization.  Let
`L in R^(n x r)` have rows `u_i^T`, put

\[
 C_L=LL^T,\qquad B_i=LA_iL^T,
 \qquad (B_i)_{jk}=u_j^TA_i u_k.
\tag{16.18k}
\]

If `rank L=r`, then

\[
\boxed{LA_iA_kL^T=B_iC_L^+B_k.}
\tag{16.18l}
\]

Consequently the missing vector-feature Gram has the explicit four-term
readout

\[
\boxed{
\begin{aligned}
 \langle b_{ij},b_{kl}\rangle={}&
 (B_iC_L^+B_k)_{jl}
 +(B_iC_L^+B_l)_{jk}\\
 &+(B_jC_L^+B_k)_{il}
 +(B_jC_L^+B_l)_{ik}.
\end{aligned}}
\tag{16.18m}
\]

The proof is `L^T(LL^T)^+L=I_r` followed by expansion of (16.18j).
By contrast, the symmetric cubic cumulant exposes only

\[
 \kappa_3(H_i,H_j,H_k)
 =\operatorname{tr}(A_iA_jA_k)
 +(B_i)_{jk}+(B_j)_{ik}+(B_k)_{ij},
\tag{16.18n}
\]

and repeated K21 exposes still fewer oriented combinations, as in (16.11).
Thus an H3 value without separately identified oriented `B_i` planes does not
own (16.18m).

If `rank L<r`, (16.18l) inserts the projector
`P_L=L^T(LL^T)^+L`.  The components

\[
 (I-P_L)(A_i u_j+A_j u_i)
\tag{16.18o}
\]

are invisible to every `B_i` factorization but still contribute positively
and by cross terms to the vector-feature Gram.  They need a direct resident
`(A,u)` evaluation or a separate acquisition.

The direct resident-core action is already the linear addition (16.15).
Its leading arithmetic is matrix-vector rather than matrix-matrix work.  For
dense `Theta_i`, an optimistic streamed census has order

\[
\boxed{
 G_{vec}(k,L)
 =O\!\left(L{2kr^2+2k(k+1)r\over n^2}\right),}
\tag{16.18p}
\]

before oriented-source formation, hooks, reset, and memory traffic.  At
`n=1024,r=55,k=21,L=10`, the displayed unit-coefficient normalization is about
`1.70` equivalents.  The number is small enough to audit as a fused
replacement/resident-core read, but too large for a strict one-equivalent
additive margin before extras.  It is not free merely because pure products
`P_ia` are resident.

This creates a clean three-way ownership gate:

1. literal resident `(A_i,u_i)` permits direct use of (16.15);
2. literal oriented `B_i` plus full-rank `L` permits (16.18m); or
3. symmetric H3/K21 alone leaves the vector Gram unidentified.

For the separately derived six-reset `r=55` paid-Gram suffix, one can combine
this vector-leg census with the pure H211 surface count analytically.  This is
conditional on that schedule/ledger still being the active replacement
candidate.  Its dense pure count was

\[
 {6\over1024^2}(87k^2+4785k+1760).
\]

Adding the displayed unit-coefficient dense vector count in (16.18p) gives

\[
\boxed{
 C_{H211+vec}^{dense}(k)
 \simeq {6\over1024^2}
 (197k^2+10945k+1760).}
\tag{16.18q}
\]

The old strict `0.803` margin then permits `k<=10`; the terminal-total
`2.360` margin permits `k<=25`, before hooks and memory.  If every `Theta_i`
is literally diagonal in one frozen common frame, the corresponding
optimistic combined count is

\[
\boxed{
 C_{H211+vec}^{diag}(k)
 \simeq {6\over1024^2}(11032k+4785),}
\tag{16.18r}
\]

giving `k<=12` under `0.803` and `k<=36` under `2.360`.  These screens show
that the vector leg is not automatically fatal, but it materially lowers the
old pure-H211 rank ceilings and fits most naturally when the terminal-total
switch is lawful.  Literal operation reuse can lower the count; absent such
a receipt, neither (16.18q) nor (16.18r) may be called free.

Finally, (16.13) is total AQ q211.  It is additive only as

\[
 \boxed{\rho_{res}=\rho_{AQ,true}-\rho_{q4,owned}.}
\tag{16.19}
\]

If a full-AQ replacement already evaluates the same action, an added CP/native
q211 carrier is a duplicate.  Conversely, D10 covariance alone does not own
the connected Gram (16.4); ownership still follows (11.5).  If the endpoint
reverse covector queries a complete H4 sector beyond (16.4), the promotion
gate is the factor-through condition

\[
 \ker J_{\Psi\text{-Gram}}\subseteq\ker J_{endpoint},
\tag{16.20}
\]

not an assumption that q1111 is dark.

### `H4_AQ_PAIR_GRAM_IDENTITY`

- **Status:** solved exact algebra and normalization.
- **Promotion:** use (16.4) only for a core with lawful provenance and charge
  full formation/action cost; residualize via (16.19).
- **Close as independent arbitrary q4 parameter:** unconditional inside the
  exact AQ model; q4 is fixed by its pair-feature Gram once `(A,u)` is fixed.

### `H4_PURE_Q4_TO_H4_GRAM_BRIDGE`

- **Status:** solved exact identity (16.11a).
- **Promotion:** a claimed complete H4-square norm/action supplies the full
  pure q4 and quadratic covariance, or proves its endpoint factorization from
  the actually retained collision faces.
- **Close information-theoretic H4 miss:** when complete pure q4 and `C_Q` are
  both known; they already determine the H4 pair-module Gram.  This does not
  close a typed-interface/all-distinct or temporal transport miss.

### `H4_AQ_Q211_REVERSE_ACTION`

- **Status:** exact formula solved; resident-product reuse and endpoint
  coverage open.
- **Promotion:** the actual reverse masks have the form in (16.13), every
  needed `P_ia` is either charged or literally reused, total cost fits, and
  (16.20) passes for the claimed Bellman correction.
- **Close as cheap:** dense anchor-varying `Theta_i` forces recomputation whose
  literal census exceeds the replacement allowance, or the complete endpoint
  sees a q1111/collision kernel absent from (16.4).

### `H4_AQ_Q211_SLICE_RANK_CYCLIC_GATE`

- **Status:** generic low-rank claim closed / candidate-specific cyclic module
  open.
- **Promotion of `s<k`:** verify (16.18g) at every reset, bound the cyclic span
  (16.18i) and its dense-anchor mixing by the same `s`, and price all retained
  matrices.
- **Default:** `s=k`; an initially rank-one terminal mask is insufficient.

### `H4_AQ_PAIR_GRAM_INVERSE_FIBER`

- **Status:** exact location of ambiguity solved / observability open.
- **Promotion:** mixed frames make the relevant Schur-residual pair Gram
  identifiable and pass the previously stated restricted singular-value
  error gate.
- **Close from `(C,K21)` alone:** unconditional; (16.10)--(16.11) do not in
  general determine (16.4).

### `H4_AQ_VECTOR_PAIR_FEATURE_OWNERSHIP`

- **Status:** exact factorization solved / literal ownership open.
- **Promotion:** either evaluate (16.15) from a resident full core, or expose
  oriented `B_i` and verify (16.18l), including the rank/projector condition;
  residualize its Schur value against old H3/K21.
- **Close as already H3-owned:** unless the old tape identifies the oriented
  `B_i` planes rather than only their symmetric sums.

### `H4_AQ_FULL_PAIR_FEATURE_SIX_RESET_CAP`

- **Status:** analytic conditional screen solved / schedule and billing
  provenance open.
- **Promotion:** the six-reset ledger remains active, full vector-leg Schur
  novelty is nonzero, the actual slice structure satisfies the appropriate
  bound in (16.18q) or (16.18r), and literal hooks/memory retain total work
  below 49.
- **Close:** complete-old projection owns the vector leg, or audited extras
  erase the stated margin.

## 17. The pair square lift is exactly a typed Stein-discrepancy carrier

The pair feature (16.2) has an independent probabilistic characterization
that explains why it can affect a Bellman mean rather than merely fit q4.
Let `mathcal L` be the Ornstein--Uhlenbeck generator and define the symmetric
Malliavin--Stein matrix

\[
 \Gamma_{ij}^{s}
 ={1\over2}\left(
 \langle DH_j,-D\mathcal L^{-1}H_i\rangle
 +\langle DH_i,-D\mathcal L^{-1}H_j\rangle\right).
\tag{17.1}
\]

Since

\[
 DH_i=u_i+A_i z,
 \qquad -D\mathcal L^{-1}H_i=u_i+{1\over2}A_i z,
\]

its centered discrepancy from covariance is exactly

\[
\boxed{
 T_{ij}:=\Gamma_{ij}^{s}-C_{ij}
 ={3\over4}b_{ij}^Tz+{1\over2}I_2(S_{ij}),
 \quad S_{ij}=A_i\circ A_j,
 \quad b_{ij}=A_i u_j+A_j u_i.}
\tag{17.2}
\]

Therefore

\[
\boxed{
 \operatorname{Cov}(T_{ij},T_{kl})
 ={9\over16}\langle b_{ij},b_{kl}\rangle
 +{1\over2}\langle S_{ij},S_{kl}\rangle.}
\tag{17.3}
\]

The two chaos channels are orthogonal and separately typed.  Whiten them by

\[
 \widehat T_{ij}
 ={4\over3}\Pi_1T_{ij}+\sqrt2\,\Pi_2T_{ij}
 =b_{ij}^Tz+\sqrt2 Q_{S_{ij}}.
\tag{17.4}
\]

Then

\[
 \boxed{
 \operatorname{Cov}(\widehat T_{ij},\widehat T_{kl})
 =\langle\Psi_{ij},\Psi_{kl}\rangle.}
\tag{17.5}
\]

Combining (17.5) with (16.4) gives the exact Stein square-root identity

\[
\boxed{
\begin{aligned}
 \kappa_4(H_i,H_j,H_k,H_l)
 ={}&\operatorname{Cov}(\widehat T_{ij},\widehat T_{kl})\\
 &+\operatorname{Cov}(\widehat T_{ik},\widehat T_{jl})
 +\operatorname{Cov}(\widehat T_{il},\widehat T_{jk}).
\end{aligned}}
\tag{17.6}
\]

Raw covariance (17.3) alone is not enough to infer (17.6), because its H1
and H2 channels have different weights.  A lawful carrier must preserve their
types or apply the explicit channel rescaling (17.4).

### Exact lower-chaos subtraction

Define the oriented cross moment

\[
 J_{ij,k}=\mathbb E[T_{ij}H_k].
\]

Equation (17.2) gives

\[
\boxed{
 J_{ij,k}={3\over4}\langle b_{ij},u_k\rangle
 +{1\over2}\operatorname{tr}(A_iA_jA_k).}
\tag{17.7}
\]

The symmetric cubic cumulant is only its cyclic symmetrization:

\[
\boxed{
 \kappa_3(H_i,H_j,H_k)
 ={2\over3}(J_{ij,k}+J_{ik,j}+J_{jk,i}).}
\tag{17.8}
\]

Thus symmetric H3/K21 does not in general own the oriented `J` coordinates.
Let `K_T=Cov(T,T)` on pair indices and let `C=Cov(H,H)`.  The orthogonal
projection of `T` onto the original AQ variables has coefficient `JC^+`, and
the exact residual provider Gram is the PSD Schur complement

\[
\boxed{
 K_{T\mid H}=K_T-JC^+J^T\succeq0.}
\tag{17.9}
\]

This is the correct no-double-count square before any endpoint projection.
Replacing `J` by the cyclic sums (17.8) is invalid unless the old typed
interface actually identifies every oriented coordinate required by (17.7).

### Target fusion and the Bellman mechanism

For one symmetric pair mask

\[
 \Lambda=\sum_{a=1}^k\sigma_a v_av_a^T,
 \quad M_a=\sum_i(v_a)_iA_i,
 \quad \bar u_a=\sum_i(v_a)_iu_i,
\]

the complete contracted centered Stein provider fuses exactly to

\[
\boxed{
 \sum_{ij}\Lambda_{ij}T_{ij}
 =\sum_{a=1}^k\sigma_a\left[
 {3\over2}(M_a\bar u_a)^Tz+{1\over2}I_2(M_a^2)
 \right].}
\tag{17.10}
\]

Hence a deterministic rank-`k` reverse mask needs `k` mixed cores and `k`
matrix squares, not an `n^2` pair bank.  This is the finite
noncommutative square-lift mechanism behind the cheap target-conditioned
branch.

It couples to a mean through the exact multivariate Stein identity.  If
`f_h` solves the Gaussian Stein equation

\[
 x\cdot\nabla f_h(x)-C:\nabla^2 f_h(x)
 =h(x)-\mathbb Eh(Z_C),
\]

then

\[
\boxed{
 \mathbb Eh(H)-\mathbb Eh(Z_C)
 =\mathbb E[T:\nabla^2 f_h(H)].}
\tag{17.11}
\]

Thus `T` is a direct Bellman/mean correction provider.  The H2+H4 square
lift is not merely a fourth-moment fit: it is the finite AQ Stein discrepancy
paired against a pulled-back Hessian.

The caveat is equally exact.  For ReLU/network suffixes,
`nabla^2 f_h(H)` is generally state dependent.  The relevant deployable rank
is the functional surface-mask Schmidt rank in a decomposition

\[
 \nabla^2 f_h(H)=\sum_{s=1}^R\chi_s(H)\Lambda_s,
\tag{17.12}
\]

not the matrix rank of one same-site `Lambda`.  Formula (17.10) is cheap only
after (17.12) has a frozen finite-rank receipt and the correlations with every
`chi_s` are evaluated.  Gate multiplication can grow this cyclic module by
the Kraus mechanism (16.18f)--(16.18i).

Ownership is also explicit.  A Gaussian/covariance baseline owns `C`, not
`T`.  Symmetric H3 owns only the cyclic image (17.8).  A full AQ/Stein
replacement may own all of (17.11), in which case an added q4 carrier is a
duplicate.  Any additive claim must apply the typed Schur subtraction (17.9)
and then the complete-old endpoint projector.

### `H4_AQ_STEIN_SQUARE_ROOT`

- **Status:** exact algebra and Bellman identity solved.
- **Promotion:** retain/separately type both channels in (17.2), subtract
  (17.9), and show nonzero endpoint Schur gain through (17.11).
- **Close as moment-only heuristic:** unconditional; the exact mechanism is a
  Stein-Hessian action, not q4 correlation by itself.

### `H4_AQ_STEIN_TARGET_FUSION`

- **Status:** constant-mask fusion solved / functional mask rank open.
- **Promotion:** the actual pulled-back Hessian admits a frozen low-`R`
  decomposition (17.12), each contracted provider uses (17.10), and its
  total cyclic-module work/error fits the 49-equivalent gate.
- **Close from same-site rank alone:** unconditional; state-dependent surface
  masks can have large operator-Schmidt rank despite low pointwise rank.

### `H4_AQ_STEIN_ORIENTED_OLD_SUBTRACTION`

- **Status:** exact Schur theorem solved / old row-span open.
- **Promotion:** expose literal oriented `J`, compute (17.9), and retain only
  its positive endpoint-relevant residual.
- **Close as H3-owned:** only if the actual old typed rows recover `J`, not
  merely the cyclic combination (17.8).

## 18. Exact ReLU surface-rank obstruction to a globally low-rank Stein mask

The functional-rank caveat after (17.12) is generically active already for
one ReLU layer.  For

\[
 h(x)=\sum_{i=1}^m c_i(w_i^Tx+b_i)_+,
\]

its distributional Hessian is

\[
\boxed{
 \nabla^2h(x)=\sum_{i=1}^m
 c_i\,\delta(w_i^Tx+b_i)\,w_iw_i^T.}
\tag{18.1}
\]

Assume the affine hyperplanes
`Sigma_i={x:w_i^Tx+b_i=0}` are distinct and each has a point outside every
other `Sigma_j`.  Then the distributions `delta_(Sigma_i)` are linearly
independent: choose a smooth test function supported in a small neighborhood
of such a private point on `Sigma_i`.  Pairing a putative linear relation
with that test function forces its `i`th coefficient to vanish.

It follows that the minimal functional Schmidt rank of (18.1) is exactly the
rank of the matrix-valued coefficient family after grouping identical
surfaces.  In particular, if every `c_i` is nonzero and the rank-one matrices
`w_iw_i^T` are linearly independent,

\[
\boxed{
 \operatorname{Schmidt,rank}_{\mathcal D'\otimes\operatorname{Sym}}
 (\nabla^2h)=m.}
\tag{18.2}
\]

For generic dense rows with `m<=d(d+1)/2`, the matrices are independent, so
the exact rank is `m`.  At width 1,024, a one-word or one-matrix-rank terminal
description therefore does not imply a one-word Stein surface mask; the
generic exact birth has width rank before any temporal return.

A nonlinear suffix replaces each constant `c_i` by a pulled-back coefficient
on the same activation surface and creates further piecewise surfaces.  It
cannot lower (18.2) without an exact cancellation/old quotient.  Likewise, a
shared scalar Hermite or Fourier approximation

\[
 \delta(x)\approx\sum_{q=1}^J a_q\varphi_q(x)
\]

does not by itself make the physical functional rank `J`: the functions
`varphi_q(w_i^Tx+b_i)` remain generically independent across `i`.  It can
still be useful when each surface term is integrated and retired locally,
which is the streamed surface-action branch rather than a persistent
low-rank mask claim.

There are only four exact escape types:

1. many units have identical/proportional activation surfaces and can be
   grouped;
2. the complete-old/future quotient annihilates all but a low-dimensional
   measure-valued matrix span;
3. a symmetry/Jordan law makes the pulled-back coefficient measures
   identical in the relevant quotient; or
4. every surface is contracted locally into a scalar before transport, with
   all local work explicitly charged.

For an approximation, let `mathcal H_surf` be the endpoint-whitened operator
from the measure-valued Hessian birth to the Bellman residual.  A rank-`R`
surface carrier is admissible only when its frozen approximation satisfies

\[
\boxed{
 \|\mathcal H_{surf}(H_{surf}-\widehat H_R)|_2
 \le\eta_{req},}
\tag{18.3}
\]

or, for an orthogonal singular truncation, when the tail singular energy
falls below the corresponding correlation/error gate.  Pointwise matrix
rank, same coefficients `a_q`, and an unweighted surface-energy statement do
not imply (18.3).

### `H4_AQ_STEIN_RELU_SURFACE_RANK_NOGO`

- **Status:** generic exact no-go solved by (18.1)--(18.2).
- **Promotion of a low functional rank:** exhibit one of the four exact
  escapes and a literal quotient/factorization receipt, or pass the frozen
  endpoint error gate (18.3).
- **Default:** width-many surface directions; use the streamed local-action
  census, not the constant-mask fusion price.

## 19. Exact Bellman telescope: contract the square lift locally and retire it

There is an exact way around destructive repeated H4 resets that does not
claim an autonomous forward H4 bank.  Let `P_l` be the exact layer/Bellman
operator on test functions and `G_l` its declared Gaussian/AQ closure
operator.  For a terminal test `h`, the operator identity

\[
\boxed{
 P_0P_1\cdots P_{L-1}-G_0G_1\cdots G_{L-1}
 =\sum_{l=0}^{L-1}
 G_0\cdots G_{l-1}(P_l-G_l)P_{l+1}\cdots P_{L-1}}
\tag{19.1}
\]

is an exact hybrid/telescoping expansion (empty products are identities).
Thus every missing contribution can be contracted at its birth against the
**exact future value function** and retired; no H4 tensor need survive another
reset.

Suppose at one hybrid cut the exact local preactivation has the AQ form
`H_l=u_l z+Q_(A_l)` and `G_l` replaces it by a Gaussian `Z_l` with the same
covariance.  Put

\[
 V_{l+1}=P_{l+1}\cdots P_{L-1}h.
\]

If `f_l` solves the covariance-`C_l` Gaussian Stein equation for
`V_(l+1)`, then (17.11) gives the exact local defect

\[
\boxed{
 (P_l-G_l)V_{l+1}
 =\mathbb E[T_l:\nabla^2 f_l(H_l)].}
\tag{19.2}
\]

Equations (19.1)--(19.2) are the constructive sequence/mechanism suggested by
the square-reset result:

1. carry/refit the lawful AQ core `(A_l,u_l)`;
2. pull back the exact future Stein Hessian (or its certified quotient);
3. evaluate the local typed provider through (17.10), including the oriented
   Schur subtraction (17.9);
4. inject the scalar Bellman correction; and
5. retire the pair/H4 provider immediately.

The `0.98334^L` destructive-forward fidelity bound does not apply to an exact
local contraction, because the omitted component is never asked to survive
`L` resets.  Its replacement is the functional surface-rank/work/error gate
of Section 18.  This is the mathematically clean distinction between a dead
autonomous H4 carrier and a potentially viable reverse transient.

There are three nonnegotiable qualifications.

First, if the old `G_l` is D11/Q24 rather than a covariance-only Gaussian,
then (19.2) must use the residual local operator

\[
 P_l-G_l^{old},
\tag{19.3}
\]

not the total AQ-vs-Gaussian `T_l`; equations (17.7)--(17.9) and the complete
typed row-span implement that subtraction.

Second, replacing the exact future value `V_(l+1)` in (19.2) by the approximate
future value changes the telescope and creates a model remainder.  This
remainder is not controlled by the local q4 correlation.  It needs either the
exact hybrid ordering (19.1), a separate contraction/Lipschitz bound, or an
explicit joint refit.

Third, a finite-rank approximation `T_l:H_l^R` has propagated endpoint error.
Let `A_(0:l-1)=G_0...G_(l-1)` be the old-prefix value map and let `e_l` be the
local scalar/vector action error, including collision and AQ-model bias.  A
sufficient complete bound is

\[
\boxed{
 \left\|\sum_l A_{0:l-1}e_l\right\|_V
 \le\sum_l\|V^{1/2}A_{0:l-1}\|_{op}\,\|e_l\|
 \le\eta_{req}.}
\tag{19.4}
\]

The first norm in (19.4), with signed cross-layer cancellation retained, is
the sharp frozen gate; the triangle bound is only a sufficient envelope.
The same corrected feature must then clear the Schur gain requirement after
dropping/replacing its old owner.

### `H4_AQ_STEIN_BELLMAN_TELESCOPE`

- **Status:** exact operator mechanism solved / deployable Hessian quotient
  and old subtraction open.
- **Promotion:** every local defect uses the exact hybrid future or a bounded
  substitute, the D11/Q24 row-span is subtracted, streamed surface work fits,
  and the joint propagated error (19.4) plus Schur gain gates pass.
- **Close as autonomous H4 requirement:** inapplicable; (19.1) proves local
  contract-and-retire is a distinct exact architecture.  Close the transient
  branch only if its functional surface rank/work or residual value fails.

## 20. Critical ownership correction: the dominant diffuse H4-square energy
## is the incumbent Gaussian pairing, not novel q4 physics

The high one-reset correlation `0.98334` is a representation-fidelity number,
not an additive-value prior against D10.  Equation (16.11a) makes the reason
quantitative.  For one pure quadratic core, write

\[
 t_2=\operatorname{tr}M^2,qquad
 t_4=\operatorname{tr}M^4,qquad
 \tau={t_4\over t_2^2}.
\]

The exact H4-square norm is

\[
\boxed{
 \|U_M\|_2^2={1\over2}t_2^2+t_4
 =2C_Q(M,M)^2+{1\over3}\kappa_4(Q_M).}
\tag{20.1}
\]

The first term is precisely the disconnected Gaussian covariance pairing;
the second is the connected non-Gaussian correction.  On a single square ray,
their fractions are

\[
\boxed{
 F_{pair}^{U}={1\over1+2\tau},
 \qquad F_{conn}^{U}={2\tau\over1+2\tau}.}
\tag{20.2}
\]

For an equal-spectrum rank-`r` diffuse core, `tau=1/r`, so only

\[
 \boxed{F_{conn}^{U}={2\over r+2}}
\tag{20.3}
\]

of the H4-square norm is connected.  At `r=55`, this is `2/57`, about `3.5%`;
roughly `96.5%` of the apparent H4-square energy is the Gaussian pairing that
an exact covariance/Mehler baseline is designed to own.

The same conclusion holds for the whole centered scalar square.  Since

\[
 Q_M^2-\mathbb EQ_M^2=2Q_{M^2}+U_M,
\]

orthogonality gives

\[
\boxed{
 \|Q_M^2-\mathbb EQ_M^2\|_2^2
 ={1\over2}t_2^2+3t_4.}
\tag{20.4}
\]

A matched scalar Gaussian has square variance `t_2^2/2`; the excess
non-Gaussian square energy is only `3t_4`.  Its fraction is

\[
\boxed{
 F_{excess}^{square}={6\tau\over1+6\tau},}
\tag{20.5}
\]

which is `6/(r+6)` for an equal-spectrum rank-`r` core (about `9.8%` at
`r=55`).  The older formula
`4tau/(1+6tau)` is the share in the explicit underlying H2 shadow
`2Q_(M^2)`; (20.5) is instead the entire excess over the matched Gaussian
square.

It is useful to state the fire-bar implication in terms of the participation
rank

\[
 r_{eff}={1\over\tau}
 ={(\operatorname{tr}M^2)^2\over\operatorname{tr}M^4}.
\]

Then

\[
 F_{conn}^{U}={2\over r_{eff}+2},
 \qquad
 F_{excess}^{square}={6\over r_{eff}+6}.
\tag{20.5a}
\]

For even a `20%` (`30%`) total-energy novelty prior, the H4-only connected
piece requires

\[
 r_{eff}\le8\quad (r_{eff}\le14/3),
\tag{20.5b}
\]

while the entire centered-square excess requires

\[
 r_{eff}\le24\quad (r_{eff}\le14).
\tag{20.5c}
\]

These are not endpoint-value bounds--a small source can align unusually well
with the residual--but they are exact spectral fail-fast thresholds against
using total square energy as a 20--30% move prior.

The ownership fractions do not improve by nudging the scalar ReLU bias.  The
order-two scalar-Hermite contribution at any bias is

\[
 b_2(a){Q_M^2-s^2\over\sqrt2,s^2},
 \qquad b_2(a)={\sigma\phi(a)\over\sqrt2}.
\]

The same coefficient multiplies its Gaussian pairing and connected/excess
pieces, so (20.2) and (20.5) are unchanged **within the square sector**.
Bias changes the square sector's share of total ReLU variance and can cross a
total-fidelity threshold, but it cannot turn the mostly-old diffuse square
into mostly-novel physics.  The earlier `a about 6.5e-3` fidelity crossing is
therefore not a no-double-count remedy.

### Affine-quadratic extension: the vector leg can rescue only through
### spectral alignment

For `H=u^Tz+Q_A`, put

\[
 v=\operatorname{Var}H=\|u\|^2+{1\over2}\operatorname{tr}A^2.
\]

Equation (16.7) gives the exact excess kurtosis

\[
\boxed{
 \kappa_4(H)=3\operatorname{tr}A^4+12u^TA^2u\ge0.}
\tag{20.7}
\]

Therefore the fraction of centered-square energy beyond a matched scalar
Gaussian is

\[
\boxed{
 F_{excess}^{AQ}
 ={3\operatorname{tr}A^4+12u^TA^2u
 \over
 2v^2+3\operatorname{tr}A^4+12u^TA^2u}.}
\tag{20.8}
\]

To make this fraction at least `p`, a necessary and sufficient scalar gate is

\[
\boxed{
 3\operatorname{tr}A^4+12u^TA^2u
 \ge {2p\over1-p}v^2.}
\tag{20.9}
\]

Thus `20%` needs `kappa4(H)>=v^2/2`, and `30%` needs
`kappa4(H)>=6v^2/7`.

Define the two effective ranks

\[
 r_A={ (\operatorname{tr}A^2)^2\over\operatorname{tr}A^4},
 \qquad
 r_u={\operatorname{tr}A^2\,\|u\|^2\over u^TA^2u}.
\tag{20.10}
\]

Writing `q=tr(A^2)/2` and `ell=||u||^2`, (20.7) becomes

\[
\boxed{
 \kappa_4(H)={12q^2\over r_A}+{24q\ell\over r_u}.}
\tag{20.11}
\]

The vector pair feature is therefore a genuine escape from the pure-H4
diffuse bound only when `u` aligns with a small spectral subspace of `A`
(`r_u` is small), or when `A` itself is concentrated (`r_A` small).  If both
effective ranks are at least `r`, then

\[
 \kappa_4(H)le {12\over r}(q^2+2q\ell)
 \le {12\over r}v^2,
\]

and hence

\[
\boxed{F_{excess}^{AQ}\le {6\over r+6}.}
\tag{20.12}
\]

So an isotropically diffuse rank-55 affine-quadratic core is still capped by
the same `6/61` total-square novelty prior; the newly identified vector leg
matters only with a concrete alignment receipt.  This makes `(r_A,r_u)` a
strict analytic pre-screen before paying for the oriented vector provider.

Do not substitute nominal latent rank for `r_A`: a semicircle/GOE-like
spectrum has heuristically `r_A about r/2`, not `r`.  At nominal `r=55` that
would put the pure square excess near `12/(55+12)`, about `17.9%`--still below
a 20% prior, but close enough that only the actual spectral receipt can close
the lane.  This heuristic is interpretation only; (20.8)--(20.12) are the
binding deterministic statements.

### `H4_AQ_VECTOR_ALIGNMENT_FIRE_GATE`

- **Status:** solved exact scalar/effective-rank screen.
- **Promotion of vector-leg economics:** (20.9) has enough headroom at the
  actual core spectra/alignment and the endpoint Schur action is nonzero.
- **Close the diffuse vector rescue:** when both effective ranks obey a bound
  whose (20.12) ceiling is below the required move before old subtraction.

The family-wide version is a generalized quartic eigenvalue, not the average
of the per-unit ranks.  For `A(w)=sum_i w_iA_i`, `u(w)=sum_iw_i u_i`, set

\[
\boxed{
 g_*=sup_{w:\,w^TCw=1}
 {3\operatorname{tr}A(w)^4+12u(w)^TA(w)^2u(w)\over2}.}
\tag{20.13}
\]

Every normalized scalar ridge in the declared physical span has
matched-Gaussian excess-square fraction at most

\[
\boxed{F_*={g_*\over1+g_*}.}
\tag{20.14}
\]

Thus `g_*<1/4` rules out a `20%` square-novelty fraction on **every** scalar
ridge, and `g_*<3/7` rules out `30%`.  Whitening `C` and defining the linear
pair-feature operator

\[
 \mathcal P:\operatorname{Sym}_n\to
 \operatorname{Sym}_r\oplus\mathbb R^r,
 \qquad \mathcal P(ww^T)=\Psi(w,w),
\]

gives

\[
 g_*={3\over2}sup_{\|w\|=1}\|\mathcal P(ww^T)\|^2
 \le {3\over2}\|\mathcal P\|_{F\to2}^2.
\tag{20.15}
\]

The right side is a deterministic spectral upper certificate; the left side
is the sharp Veronese-restricted gate.  This does not by itself bound an
arbitrary signed multiridge Bellman combination, which still uses the full
endpoint Gram, but it is the exact fail-fast screen for every rank-one ReLU
surface term in (18.1).

### `H4_AQ_UNIFORM_RIDGE_KURTOSIS_GATE`

- **Status:** solved deterministic formulation / candidate spectrum open.
- **Promotion of a broad ridge mechanism:** `g_*` clears the desired novelty
  scale and the corresponding endpoint Schur modes survive old subtraction.
- **Close all individual ReLU-ridge square claims:** a certified upper bound
  in (20.15) lies below the chosen fire fraction.

There is a complementary first-order scale check for one scalar ReLU reset.
Let `X` have unit variance and small standardized fourth cumulant
`gamma_4`.  The order-four Edgeworth response of
`s(X+a)_+` is

\[
\boxed{
 \Delta\mu_4^{(1)}
 ={\gamma_4\over24}s\phi(a)(a^2-1).}
\tag{20.16}
\]

At zero bias the matched-Gaussian mean is `s phi(0)`, so

\[
\boxed{{\Delta\mu_4^{(1)}\over\mu_G}=-{\gamma_4\over24}.}
\tag{20.17}
\]

For a pure quadratic core,
`gamma_4=12/r_eff`, hence the per-reset relative mean scale is exactly

\[
\boxed{-{1\over2r_{eff}}}
\tag{20.18}
\]

at first order.  Equal-spectrum `r_eff=55` gives about `0.9%` per reset;
a semicircle-like `r_eff about27.5` gives about `1.8%`.  Ten perfectly coherent
unowned responses would therefore reach a 20% scale only around
`r_eff<=25`, consistent with the exact energy threshold (20.5c).

Here "first order" means the q4 derivative **holding lower cumulants fixed**.
At zero bias the leading combined skew/kurtosis mean term is instead

\[
\boxed{
 {\Delta\mu_{3^2+4}\over\mu_G}
 ={\gamma_3^2-\gamma_4\over24}.}
\tag{20.18a}
\]

The `gamma_3^2` term comes from the `He_6` coefficient
`gamma_3^2/[2(3!)^2]` and `E[X_+He_6]/mu_G=He_4(0)=3`.
Thus a same-sign equal-eigenvalue chi-square core has
`gamma_3^2=8/r,gamma_4=12/r` and an actual leading mean response
`-1/(6r)`, only one third of the q4-only `-1/(2r)`.  Indeed

\[
 {\mathbb E[(\chi_r^2-r)/\sqrt{2r}]_+\over\phi(0)}
 =1-{1\over6r}+{1\over72r^2}+\cdots.
\tag{20.18b}
\]

A sign-symmetric/semicircle core has `gamma_3=0`, so the q4-only scale
survives.  Actual spectral skewness and the old Q24 `k3^2` row are therefore
binding, not optional refinements.

For a pure quadratic core the cancellation nevertheless has a universal
factor-three bound.  Cauchy--Schwarz gives

\[
 (\operatorname{tr}A^3)^2
 \le\operatorname{tr}A^2\operatorname{tr}A^4,
\]

and hence

\[
\boxed{
 0\le{\gamma_3^2\over\gamma_4}
 ={2\over3}{(\operatorname{tr}A^3)^2
 \over\operatorname{tr}A^2\operatorname{tr}A^4}
 \le{2\over3}.}
\tag{20.18e}
\]

With `r_4=(tr A^2)^2/tr A^4`, the leading combined zero-bias scales obey

\[
\boxed{
 {1\over6r_4}
 \le {\gamma_4-\gamma_3^2\over24}
 \le {1\over2r_4},}
\tag{20.18f}
\]

for the magnitude of the relative mean decrease, and

\[
\boxed{
 {1\over3(\pi-1)r_4}
 \le {\Delta V_{3^2+4}/V_G}
 \le {1\over(\pi-1)r_4}.}
\tag{20.18g}
\]

Thus skew can reduce the pure-core q4 effect by at most a factor of three,
but the 17--18% ten-layer upper-scale hypothesis requires the low-skew end of
this interval rather than following from `r_4` alone.

For the full affine-quadratic core the two standardized quantities are

\[
\boxed{
 \gamma_3={\operatorname{tr}A^3+3u^TAu\over v^{3/2}},
 \qquad
 \gamma_4={3\operatorname{tr}A^4+12u^TA^2u\over v^2},
 \quad v=\|u\|^2+{1\over2}\operatorname{tr}A^2.}
\tag{20.18c}
\]

Thus the leading old-subtracted zero-bias scale is governed by the signed
combination

\[
 \boxed{\chi_{34}:=\gamma_4-\gamma_3^2,}
\tag{20.18d}
\]

not by `gamma_4` alone.  The family-wide ridge screen (20.13)--(20.15) should
therefore be paired with the exact `chi_34(w)` spectrum whenever Q24/H3
already carries the lower-cumulant square.

The zero-bias **variance** response is even more directly tied to the oracle
`dvhat` channel.  For every symmetric unit-variance input,
`E[X_+^2]=1/2` exactly, so the first-order kurtosis perturbation changes only
the squared mean.  Since

\[
 \mu_G=\phi(0),\qquad
 V_G={\pi-1\over2\pi},
\]

(20.17) gives

\[
\boxed{
 \Delta V_4^{(1)}={\gamma_4\over24\pi},
 \qquad
 {\Delta V_4^{(1)}\over V_G}
 ={\gamma_4\over12(\pi-1)}.}
\tag{20.19}
\]

For a pure quadratic core, `gamma_4=12/r_eff`, hence

\[
\boxed{
 {\Delta V_4^{(1)}\over V_G}
 ={1\over(\pi-1)r_{eff}}.}
\tag{20.20}
\]

This is about `0.85%` per reset at `r_eff=55` and `1.70%` at
`r_eff about27.5`; ten coherent semicircle-like resets have a `17%` first-order
variance scale.  That makes the `18.7%` broad `dvhat` coincidence sharper,
while retaining every ownership/remainder caveat below.

After the lower linear-q3 variance response is separated, the combined
`q3^2+q4` zero-bias residual is correspondingly

\[
\boxed{
 {\Delta V_{3^2+4}\over V_G}
 ={\gamma_4-\gamma_3^2\over12(\pi-1)}.}
\tag{20.20a}
\]

Equation (20.19) is the q4 partial derivative; (20.20a) is the correct leading
combined residual before the literal Q24/H3 row-span subtraction.  Crediting
both the q4-only response and an existing `k3^2/72` correction would double
count.

The exact Gaussian fourth-cumulant derivative is also available at arbitrary
bias.  Put

\[
 m_1(a)=\phi(a)+a\Phi(a),
 \quad
 v_0(a)=(1+a^2)\Phi(a)+a\phi(a)-m_1(a)^2.
\]

Gaussian integration by parts gives

\[
 \mathbb E[(X+a)_+^2He_4(X)]=-2a\phi(a).
\]

Therefore, holding mean/variance and lower cumulants fixed, the first q4
derivatives are

\[
\boxed{
 \partial_{\gamma_4}\mu\big|_0
 ={s\phi(a)(a^2-1)\over24},
}
\tag{20.21}
\]

\[
\boxed{
 \partial_{\gamma_4}V\big|_0
 =-{s^2\phi(a)\over12}
 \left[a+m_1(a)(a^2-1)\right].}
\tag{20.22}
\]

The relative local variance coefficient is (20.22) divided by
`s^2 v_0(a)`.  Thus coherence across biased layers is a deterministic sign
and amplitude question in their standardized biases; the zero-bias estimate
must not be copied to every seam.  Equations (20.21)--(20.22) are partial
derivatives at the matched Gaussian point, not a bound on finite simultaneous
H3/q4 perturbations.

The bias-general lower-cumulant-square completion is also closed.  With
`H_m` denoting probabilists' Hermite polynomials evaluated at `a`,

\[
\boxed{
 {\Delta\mu_{3^2+4}\over s\phi(a)}
 ={\gamma_4\over24}H_2(a)
 +{\gamma_3^2\over72}H_4(a),}
\tag{20.22a}
\]

and, using
`E[(X+a)_+^2He_6(X)]=-2H_3(a)phi(a)`,

\[
\boxed{
\begin{aligned}
 \Delta V_{3^2+4}
 =-{s^2\phi(a)\over12}\bigg\{&
 \gamma_4[,a+m_1(a)H_2(a),]\\
 &+{\gamma_3^2\over3}
 [,H_3(a)+m_1(a)H_4(a),]\bigg\}.
\end{aligned}}
\tag{20.22b}
\]

At `a=0`, (20.22a)--(20.22b) reduce to
(20.18a)/(20.20a).  Away from zero, `H_2,H_3,H_4` can change sign, so even a
fixed positive `chi_34` does not prove cross-layer coherence.  The exact bias
sequence or the scalar characteristic-function oracle in Section 21 must be
used.

For a spectrally sign-symmetric pure quadratic core, the next formal
Gram--Charlier terms can be written entirely in trace ratios.  The exact
cumulants are

\[
\boxed{
 \kappa_p(Q_M)={(p-1)!\over2}\operatorname{tr}M^p,
 \qquad p\ge2.}
\tag{20.23}
\]

Put

\[
 \tau_4={\operatorname{tr}M^4\over(\operatorname{tr}M^2)^2},
 \qquad
 \tau_6={\operatorname{tr}M^6\over(\operatorname{tr}M^2)^3}.
\]

Using `He_2(0)=-1,He_4(0)=3,He_6(0)=-15`, the zero-bias mean expansion through
second cumulant order is

\[
\boxed{
 {\mu\over\mu_G}
 =1-{\tau_4\over2}+2\tau_6-{15\over8}\tau_4^2
 +O_3.}
\tag{20.24}
\]

Because a symmetric unit-variance law still has `E[X_+^2]=1/2`, this implies

\[
\boxed{
 {V-V_G\over V_G}
 ={\tau_4-4\tau_6+{7\over2}\tau_4^2\over\pi-1}
 +O_3.}
\tag{20.25}
\]

For an equal-magnitude spectrum, the second-order numerator is
`1/r-1/(2r^2)`; for a semicircle moment model
`tau_4 about2/r,tau_6 about5/r^2`, it is `2/r-6/r^2`.  Thus the second formal
correction reduces the semicircle first-order variance scale by about `3/r`
relatively (roughly `5.5%` at `r=55`), not by an order-one factor.

Equations (20.24)--(20.25) are algebraically exact coefficients of the formal
cumulant expansion, not a convergence/remainder certificate for a finite
generalized-chi-square law.  Promotion still requires a characteristic-function
or hypercontractive remainder bound and must include odd cumulants when the
spectrum is not sign symmetric.

One sign-symmetric family has an exact closed validation.  Take `r=2m` equal
eigenvalues split evenly between `+lambda` and `-lambda`.  After variance
standardization,

\[
 X={\chi_m^2-\widetilde\chi_m^2\over2\sqrt m},
 \qquad p={r\over4},
\]

with independent chi-squares, and

\[
\boxed{
 \mathbb EX_+
 ={\Gamma(p+1/2)\over\sqrt\pi\,\Gamma(p)\sqrt m}
 =\phi(0)\left[
 1-{1\over2r}+{1\over8r^2}+{5\over16r^3}+O(r^{-4})
 \right].}
\tag{20.26}
\]

The law is exactly symmetric, so `E[X_+^2]=1/2` and

\[
\boxed{
 {V-V_G\over V_G}
 ={1/r-1/(2r^2)+O(r^{-3})\over\pi-1}.}
\tag{20.27}
\]

This matches (20.24)--(20.25) coefficient by coefficient and proves that the
q4 scale is real when `gamma_3=0`; it is not an artifact of the formal
Edgeworth derivative.  It does not prove the semicircle remainder or any
typed endpoint gain, but it makes the ratio `gamma_3^2/gamma_4`--rather than
nominal rank--the decisive local cancellation diagnostic.

Equations (20.16)--(20.18) are a scale diagnostic, not an exact finite-kurtosis
or end-to-end bound.  More importantly, the incumbent univariate K4/Q24 row
may already own this same-site response.  The external q211 candidate earns
credit only for the complete-old residual cross-unit/Bellman action, never by
summing (20.18) as though every local response were new and independent.

There is nevertheless a concrete cross-lane hypothesis worth preserving.  A
semicircle-like rank-55 core has `r_eff about27.5`, so (20.18) gives an
approximately `1.8%` mean scale and (20.20) gives a `1.7%` variance scale.
Ten coherent late resets give a scale near `17--18%`, strikingly close to the
separately certified `18.7%` broad variance
chain.  This is **not** a proof or a new numerical result.  It is an analytic
mechanism hypothesis: the oracle `dvhat` chain may be the accumulated
non-Gaussian AQ/Stein q4 residual, which should be evaluated by the local
Bellman telescope (19.1)--(19.2), not by transporting a nominal H4 tensor.
The decisive receipts are the actual `r_eff/r_u`, layer signs, complete-old
projection, and endpoint action.  Even a full `18.7%` recovery is only one
component of the raw-below-`1e-8` stack.

### `H4_Q4_EDGEWORTH_COHERENCE_SCALE`

- **Status:** exact first derivative solved / finite correction and ownership
  conditional.
- **Promotion:** higher-order remainder is bounded, local signs survive the
  exact Bellman telescope coherently, and univariate Q24 plus all old cross
  rows are subtracted.
- **Close as a direct additive estimate:** absent those receipts; (20.18) is
  only the local derivative scale.

### `H4_AQ_K3SQUARED_Q4_SUBTRACTION`

- **Status:** solved leading zero-bias combination / full typed row-span and
  finite remainder open.
- **Promotion:** use `chi_34` from (20.18c)--(20.18d), subtract the literal
  Q24 `k3^2` row exactly once, and show the remaining biased/endpoint action
  is coherent.
- **Close q4-only scale:** whenever `gamma_3^2` cancels it materially or the
  old row already spans the combined response.

## Current lane verdict and shortest next-proof path

1. **Autonomous H4 transport remains closed generically.**  Its latent
   Hilbert rank and correlated-reset module growth exceed the state/work cap.
   A narrow same-site frame or initially rank-one reverse word does not change
   this.
2. **The exact constructive architecture is a reverse local transient.**  The
   AQ pair feature is a typed Stein discrepancy, its constant-mask contraction
   fuses to latent squares (17.10), and the Bellman telescope (19.1) permits
   contract-and-retire at every seam.
3. **Exact ReLU surface rank prevents calling that transient globally cheap.**
   Generic surface masks have width functional rank.  Viability requires the
   already identified streamed/local surface action, a complete-old quotient,
   or a candidate-specific low cyclic module.
4. **The original `0.98334` prior was the wrong value prior.**  Most diffuse
   square energy is the Gaussian covariance/Mehler pairing already targeted
   by D10.  The connected/excess part is governed by `r_eff,r_u,g_*`, not by
   total H2+H4 fidelity.
5. **There is a plausible but unproved 17--18% mechanism.**  A
   semicircle-like r55 spectrum gives a 1.7% first-order zero-bias variance
   response per reset; ten coherent local residuals match the scale of the
   certified 18.7% `dvhat` chain.  This supports the AQ/Stein telescope as the
   correct carrier of that component, but does not by itself reach raw
   `1e-8` and cannot be credited before old-row subtraction.
6. **The linear-core vector leg is a separate conditional opportunity.**  It
   is not automatically H3/K21-owned; it is cheap enough only with resident
   `(A,u)` or oriented provider reuse and matters at fire scale only with
   small `r_u` / endpoint alignment.

The next proofs/receipts, in fail-fast order, are:

1. certify the actual core spectral gates `(r_A,r_u,g_*)`, the skew-corrected
   `chi_34=gamma_4-gamma_3^2`, and the signs of the exact biased response
   (20.22a)--(20.22b) at the live seams (or use the exact Section 21 oracle);
2. form the **typed** complete-old row-span for `T,J`, inherited CP4,
   Gaussian/Mehler, and Q24, then compute the oriented Schur residual
   (17.9)--not total q4;
3. factor the literal pulled-back ReLU surface Hessian at the synchronized
   cut, using full `s=k` unless the Kraus cyclic-module receipt passes;
4. use the exact Bellman telescope and joint endpoint error (19.4), with a
   characteristic-function/odd-cumulant remainder for (20.24)--(20.25); and
5. compare the surviving Schur gain with its replacement cost.  If it recovers
   only the predicted ~18% component, stack it with the other independent raw
   lanes rather than presenting it as the raw-below-`1e-8` solution alone.

## 21. Exact zero-bias scalar AQ oracle: remove every cumulant remainder with
## one characteristic-function integral

The local spectral/sign screen need not stop at Edgeworth order.  For

\[
 H=u^Tz+{1\over2}(z^TAz-\operatorname{tr}A),
\]

the exact characteristic function is

\[
\boxed{
 \varphi_H(t)
 =e^{-it\operatorname{tr}A/2}
 \det(I-itA)^{-1/2}
 \exp\left[-{t^2\over2}
 u^T(I-itA)^{-1}u\right].}
\tag{21.1}
\]

For any centered real variable with finite first moment,

\[
 \mathbb EH_+={1\over2}\mathbb E|H|
 ={1\over\pi}\int_0^\infty
 {1-\operatorname{Re}\varphi_H(t)\over t^2}\,dt.
\tag{21.2}
\]

The positive half second moment is the half-line mass of the weighted measure
`H^2 dP`.  Since its Fourier transform is `-varphi_H''(t)`, Gil--Pelaez
inversion gives (when there is no atom at zero)

\[
\boxed{
 \mathbb E[H^2 1_{H>0}]
 ={\mathbb EH^2\over2}
 -{1\over\pi}\int_0^\infty
 {\operatorname{Im}\varphi_H''(t)\over t}\,dt.}
\tag{21.3}
\]

Therefore the exact zero-bias ReLU variance is

\[
\boxed{
 \operatorname{Var}(H_+)
 =\mathbb E[H^2 1_{H>0}]-(\mathbb EH_+)^2,}
\tag{21.4}
\]

with no truncation in `gamma_3,gamma_4,gamma_6,...`.  For a sign-symmetric
law, the imaginary integral in (21.3) vanishes and the raw second moment is
exactly half the variance, recovering Sections 20.19--20.27.

The arbitrary-bias version is equally explicit.  For threshold `c=-b`, let

\[
 \psi_q(t)=i^{-q}\varphi_H^{(q)}(t)
 =\mathbb E[H^qe^{itH}],
 \qquad m_q=\mathbb EH^q,
\]

and define the weighted upper-tail moments

\[
\boxed{
 M_q(c)={m_q\over2}+{1\over\pi}\int_0^\infty
 {\operatorname{Im}[e^{-itc}\psi_q(t)]\over t}\,dt,
 \qquad q=0,1,2.}
\tag{21.5}
\]

Here `m_0=1,m_1=0`, `psi_0=varphi`,
`psi_1=-i varphi'`, and `psi_2=-varphi''`.  Then

\[
\boxed{
 \mathbb E(H+b)_+=M_1(-b)+bM_0(-b),}
\tag{21.6}
\]

\[
\boxed{
 \mathbb E(H+b)_+^2
 =M_2(-b)+2bM_1(-b)+b^2M_0(-b).}
\tag{21.7}
\]

Equations (21.5)--(21.7) replace the sign-sensitive truncated cumulant
formula (20.22b) by an exact one-dimensional spectral integral at every live
bias.  They also expose mean, raw second moment, and variance from the same
quadrature nodes and resolvents.

No numerical differentiation of (21.1) is needed.  Put

\[
 R(t)=(I-itA)^{-1},\qquad L(t)=\log\varphi_H(t).
\]

Since `R'=iRAR`, direct differentiation gives

\[
\boxed{
\begin{aligned}
 L'(t)={}&-{i\over2}\operatorname{tr}A
 +{i\over2}\operatorname{tr}(RA)
 -t,u^TRu-{it^2\over2}u^TRARu,
\end{aligned}}
\tag{21.8}
\]

and

\[
\boxed{
\begin{aligned}
 L''(t)={}&-{1\over2}\operatorname{tr}(RARA)-u^TRu\\
 &-2it,u^TRARu+t^2u^TRARARu.
\end{aligned}}
\tag{21.9}
\]

Thus

\[
 \boxed{\varphi_H'=\varphi_HL',\qquad
 \varphi_H''=\varphi_H[(L')^2+L''].}
\tag{21.10}
\]

In an eigensystem of `A`, every trace and quadratic form in
(21.8)--(21.10) is an `O(r)` spectral sum per quadrature node.  The exact
scalar oracle therefore adds no latent `r x r` products after an eigensystem
already paid by the AQ surface calculation; node count, special-function
policy, and error-controlled tail truncation remain to be billed.

This supplies a fail-fast **analytic** oracle for every target-fused scalar
core `(M_a,ubar_a)` in (17.10).  It needs eigenvalues/resolvent contractions
and a one-dimensional deterministic quadrature, not sampling or a fourth
tensor.  The same characteristic function can be shifted to a nonzero bias,
with half-line weighted inversion at threshold `-b`.

The ownership boundary remains unchanged.  Equations (21.1)--(21.4) compute
the total scalar AQ response.  If Q38/Q24 already owns its same-site marginal,
only the complete-old residual cross/Bellman action is new.  Nor does a scalar
oracle materialize K31/K22/q211 planes; it is most naturally used inside the
target-conditioned Bellman telescope or as a reference for a declared
replacement.

### `H4_AQ_EXACT_SCALAR_CF_ORACLE`

- **Status:** exact formulas solved / quadrature remainder, metering, and old
  subtraction open.
- **Promotion:** use a deterministic error-controlled quadrature of
  (21.1)--(21.4) on the frozen target-fused cores, charge it, and retain only
  the D11/Q24 Schur residual.
- **Close as a plane carrier:** unconditional; a scalar half-line oracle does
  not supply unsummed external q211 or future-visible q1111 state.

Therefore the diffuse square theorem does **not** imply a 96.7%-energy novel
correction after D10.  It says that an H2+H4 representation can reproduce a
square direction which is itself mostly the incumbent Gaussian/Mehler
direction.  The lawful candidate is only the complete-old residual:

\[
 \boxed{
 U_M^{nov}=P_{old}^{\perp}U_M,
 \qquad
 \text{or equivalently the typed Stein/q4 Schur residual of Sections 17--19}.}
\tag{20.6}
\]

For multiple noncommuting cores, (20.2) is not a componentwise PSD
decomposition under an arbitrary signed mixture; use the full Gram/Schur
projector (16.11a), not a sum of single-ray fractions.  But (20.2)--(20.5)
are exact fail-fast diagnostics for the diffuse-square prior.  A broad raw
gain remains possible only if the endpoint error is unusually concentrated
in the small connected residual, the spectra are much less diffuse, or the
old typed interface fails to own a nominally Gaussian pairing action.

### `H4_SQUARE_GAUSSIAN_PAIRING_OWNERSHIP`

- **Status:** solved exact single-ray decomposition and diffuse no-double-count
  correction.
- **Promotion:** report the complete-old Schur residual rather than total
  square fidelity, including the covariance/Mehler row and inherited CP4.
- **Close the `0.98334 => large novelty` inference:** unconditional.  At
  diffuse rank 55, the connected fractions are only (20.3)/(20.5) before
  endpoint alignment.

## 22. Exact equality-partition sewing of the Price trace word

This section records a normalization audit needed by the forward H211
consumer.  For a symmetric matrix `R` with unit diagonal define

\[
 (\mathcal Z_{I,R})_{abcd}
 =\delta_{ab}R_{cd}+\delta_{ac}R_{bd}+\delta_{ad}R_{bc}
  +\delta_{bc}R_{ad}+\delta_{bd}R_{ac}+\delta_{cd}R_{ab},
\tag{22.1}
\]

and let `\Pi_{211}` mean the strict equality-support projector, not a
polynomial-coefficient extractor.  Let `\mathcal C_{31}(\bar R)` be the
strict-31 tensor whose symmetrized entries are

\[
 \mathcal C_{31}(\bar R)_{iiij}=3R_{ij},\qquad i\ne j,
 \quad \bar R=R-I,
\tag{22.2}
\]

and let

\[
 (\mathcal P_I)_{abcd}
 =\delta_{ab}\delta_{cd}+\delta_{ac}\delta_{bd}
  +\delta_{ad}\delta_{bc}.
\tag{22.3}
\]

Direct evaluation on the four equality types gives

\[
\begin{array}{c|ccccc}
\text{type}&1111&211&31&22&4\\ \hline
\mathcal Z_{I,R}&0&R_{jk}&3R_{ij}&2&6\\
\mathcal C_{31}(\bar R)&0&0&3R_{ij}&0&0\\
2\mathcal P_I&0&0&0&2&6.
\end{array}
\tag{22.4}
\]

Consequently the exact tensor-entry identity is

\[
 \boxed{
 \mathcal Z_{I,R}
 =\Pi_{211}\mathcal Z_{I,R}
  +\mathcal C_{31}(\bar R)+2\mathcal P_I.}
\tag{22.5}
\]

There is no missing multinomial factor in this convention.  If a symmetric
tensor is instead stored as coefficients of its quartic polynomial, its
entries on the `211,31,22` orbits are multiplied respectively by
`12,4,6`; those orbit factors must not be mixed with (22.5).

For an affine map `W`, put `G=WW^T` and `S=WRW^T`.  Since
`W^{\otimes4}\mathcal Z_{I,R}=\mathcal Z_{G,S}` and
`W^{\otimes4}\mathcal P_I=\mathcal P_G`, linearity gives

\[
 \boxed{
 W^{\otimes4}\Pi_{211}\mathcal Z_{I,R}
 =\mathcal Z_{G,S}
  -W^{\otimes4}\mathcal C_{31}(\bar R)-2\mathcal P_G.}
\tag{22.6}
\]

The middle term may be called `Star31_W` only when it denotes the **full
affine image** `W^{\otimes4}\mathcal C_{31}(\bar R)`.  A post-affine K31
plane, or a separately fitted Star ansatz agreeing only on that plane, is
not enough: after generic dense `W` the strict coordinate-31 source has
support in every equality sector.  Also, using `R` rather than `\bar R` in
(22.2) double counts the `iiii` entries already supplied by
`2\mathcal P_I`.

### `H4_PRICE_TRACE_EQUALITY_SEWING`

- **Status:** exact normalization and affine sewing identity solved.
- **Promotion:** prove that the literal `Star31_W` state is exactly
  `W^{\otimes4}\mathcal C_{31}(R-I)` at the synchronized cut and that
  `\mathcal P_G` is the same affine pairing lineage.
- **Close nominal-plane reuse:** if only the K31 face matches, because that
  does not certify subtraction of the dense affine image in (22.6).

## 23. Explicit affine collision faces of the strict-31 subtraction

Equation (22.6) can be audited without ever constructing a quartic tensor.
Write `\bar R=R-I`, let `w_p^T` be row `p` of `W`, and set

\[
 A=W\bar R,qquad H=W^{\odot3},qquad V=W^{\odot2},qquad
 B=V\odot A,qquad C=W\odot A.
\tag{23.1}
\]

The source tensor `\mathcal C_{31}(\bar R)` has polynomial

\[
 \langle\mathcal C_{31}(\bar R),x^{\otimes4}\rangle
 =12\sum_{i\ne j}\bar R_{ij}x_i^3x_j.
\tag{23.2}
\]

Direct polarization therefore gives the three collision faces of its full
affine image `\mathcal C'=W^{\otimes4}\mathcal C_{31}(\bar R)`:

\[
 \boxed{
 \begin{aligned}
 K_4^{31}[p]
  &=12\,(w_p^{\odot3})^T\bar Rw_p,\\
 K_{31}^{31}[p,q]
  &=3\,(w_p^{\odot3})^T\bar Rw_q
    +9\,(w_p^{\odot2}\odot w_q)^T\bar Rw_p,\\
 K_{22}^{31}[p,q]
  &=6\{(w_p\odot w_q^{\odot2})^T\bar Rw_p
       +(w_p^{\odot2}\odot w_q)^T\bar Rw_q\}.
 \end{aligned}}
\tag{23.3}
\]

In matrix form these are

\[
 \boxed{
 \begin{aligned}
 K_4^{31}&=12\,\operatorname{diag}(HA^T),\\
 K_{31}^{31}&=3HA^T+9BW^T,\\
 K_{22}^{31}&=6(CV^T+VC^T).
 \end{aligned}}
\tag{23.4}
\]

The pairing subtraction in (22.6) has faces

\[
 \boxed{
 \begin{aligned}
 (2P_G)_4[p]&=6G_{pp}^2,\\
 (2P_G)_{31}[p,q]&=6G_{pp}G_{pq},\\
 (2P_G)_{22}[p,q]&=2G_{pp}G_{qq}+4G_{pq}^2.
 \end{aligned}}
\tag{23.5}
\]

Subtracting (23.3)/(23.5) from the already derived faces of
`\mathcal Z_{G,S}` gives the exact post-affine collision planes of the
transported strict-211 trace source.

This exposes the deployment issue hidden by the name `Star31_W`.  Without a
literal resident affine image, forming all of (23.4) needs the intermediate
`A=W\bar R` and three additional dense row-Gram-type products
`HA^T`, `BW^T`, and `CV^T` (the last supplies its transpose for free).
Elementwise powers are lower order, but they do not remove those products.
Even if the covariance push exposes `A`, this is approximately three dense
products per active seam, not a zero-cost plane subtraction.  A typed
already-paid Star31 image, or the target-conditioned covariance-matvec
action, is therefore the only credible cheap route under sub-GEMM slack.

### `H4_PRICE_TRACE_STRICT31_FACE_COST`

- **Status:** exact plane formulas and generic contraction count solved.
- **Promotion as free CSE:** identify the literal resident matrices whose
  products equal all three terms in (23.4), with matching normalization and
  lifetime.
- **Otherwise:** charge `W\bar R` if absent and three dense products per
  seam, or use the reverse target-conditioned action instead.

## 24. Pair-state insufficiency is a gauge theorem, not only a billing claim

The matrices `G=WW^T` and `S=WRW^T` do not determine the strict-partition
subtraction in (22.6).  The reason is that `(G,S)` is invariant under the
latent orthogonal gauge

\[
 W\longmapsto WO,qquad R\longmapsto O^TRO,qquad O\in O(n),
\tag{24.1}
\]

whereas coordinate equality projectors such as `\Pi_{31}` and `\Pi_{211}`
are invariant only under the monomial subgroup of signed permutations.
Indeed,

\[
 (WO)(WO)^T=G,qquad
 (WO)(O^TRO)(WO)^T=S,
\tag{24.2}
\]

but generically

\[
 (WO)^{\otimes4}\mathcal C_{31}
       (\operatorname{offdiag}(O^TRO))
 \ne
 W^{\otimes4}\mathcal C_{31}(\operatorname{offdiag}R).
\tag{24.3}
\]

The unit-diagonal restriction does not remove this obstruction.  For a
generic correlation matrix, the equations

\[
 \operatorname{diag}(O^TRO)=\mathbf1
\tag{24.4}
\]

impose only `n-1` independent constraints on `O(n)` (their sum is fixed by
the trace).  At a regular point their solution manifold has dimension

\[
 {n(n-1)\over2}-(n-1)={(n-1)(n-2)\over2},
\tag{24.5}
\]

which is positive for `n>=3` and vastly larger than the discrete monomial
stabilizer.  Along a generic tangent in this manifold, (24.3) changes.
Equivalently, the Hadamard-coordinate contractions `HA^T`, `BW^T`, and
`CV^T` in (23.4) are not orthogonal-gauge invariants and cannot be functions
of `(G,S)` alone.

Therefore a covariance recursion which retains only `G,S` has an exact dark
fiber for the pushed strict-31/strict-211 source.  Access to the literal
weights and `R` makes the source computable via (23.4), but does not make it
already owned; a typed partition buffer or extra contractions are necessary.

### `H4_PRICE_TRACE_PAIR_STATE_GAUGE_NOGO`

- **Status:** exact structural insufficiency solved on the regular
  unit-diagonal gauge fiber.
- **Promotion of covariance reuse:** requires an additional coordinate-typed
  state breaking (24.1), not merely equality of `G,S`.
- **Consequence:** pair-state ownership and strict-partition ownership must
  remain separate graph nodes even if both originate from the same `R,W`.

## 25. Full-gate spatial closure uses the associative envelope, not merely
## the first two Hermite coefficients

The gate-tail question has an exact algebraic answer.  Let `\mathcal A` be
the unital real star-algebra generated by every symmetric quadratic core
which can occur along the suffix, and put

\[
 \mathcal V=\mathcal A\cap\operatorname{Sym}_r,qquad
 q=\dim\mathcal V.
\tag{25.1}
\]

Define the order-four path module

\[
 \mathcal K_4(\mathcal V)
 =\operatorname{span}\{U_{R,S}:R,S\in\mathcal V\}.
\tag{25.2}
\]

For any `M in mathcal V` and any bounded scalar Borel function `f` for which
the product is square integrable,

\[
 \boxed{
 \Pi_4 M_{f(Q_M)}\,\mathcal K_4(\mathcal V)
 \subseteq\mathcal K_4(\mathcal V).}
\tag{25.3}
\]

In particular, (25.3) holds for the complete biased ReLU Jacobian
`f(Q_M)=1_{Q_M>-b}`.  It is not a truncation to the constant and linear
scalar-Hermite terms.

### Proof by degree-two diagrams

First take `f` polynomial.  Every copy of `Q_M` is a two-legged matrix
vertex.  The Wiener product formula expresses the fourth-chaos projection
of `f(Q_M)U_{R,S}` as a sum of contraction diagrams whose vertices all have
degree two.  A connected component with external legs is therefore a path,
never a branched graph.  Four surviving external legs consequently form
exactly two paths.  Reading matrices along a path gives an associative word
in `M,R,S`; adding the reverse orientation gives a self-adjoint word in
`mathcal A`, hence an element of `mathcal V`.  Closed components contribute
only trace scalars.  Every summand is thus a symmetrized product of two
members of `mathcal V`, which is (25.2).  Approximation under the Gaussian
measure with the polynomial weight supplied by `U_{R,S}` extends the result
to the bounded indicator.

The same diagram argument gives, for every `d`,

\[
 \mathcal K_{2d}(\mathcal V)
 =\operatorname{span}\left\{
 I_{2d}\operatorname{Sym}(A_1\otimes\cdots\otimes A_d):
 A_j\in\mathcal V\right\},
\tag{25.4}
\]

and

\[
 \Pi_{2d}M_{f(Q_M)}\mathcal K_{2e}(\mathcal V)
 \subseteq\mathcal K_{2d}(\mathcal V).
\tag{25.5}
\]

Thus the complete even-chaos hierarchy is spatially closed in the same
associative envelope.  Dense channel weights only take linear combinations
of the latent cores and therefore do not enlarge `mathcal V`.

### Exact rank consequence

The fourth-order module dimension obeys

\[
 k_4:=\dim\mathcal K_4(\mathcal V)
 \le {q(q+1)\over2},
\tag{25.6}
\]

with equality only when the square-lift map has no syzygies.  A forward
channel state costs `O(n k_4)` memory and a dense channel transport costs
`O(n^2 k_4)` arithmetic.

This separates two sharply different regimes.

1. If all cores commute, they are simultaneously block-diagonalizable and
   one may take `q<=r`.  At `r=55`, `k_4<=1540`; transport over ten layers is
   about `10 k_4/n=15.04` dense `n^3` equivalents before births and
   consumers.  Exact full-gate **spatial** H4 closure is therefore compatible
   with an `O(n^3)`-scale design.
2. Generic noncommuting symmetric cores generate the full matrix algebra
   (up to the usual real irreducibility qualification).  Then
   `q=r(r+1)/2`; at `r=55`, `q=1540` and the naive cap is
   `k_4=1,186,570`.  The state and transport are prohibitive.  A small
   Jordan span, or a Clifford name, is not sufficient if its associative
   envelope is full; one needs a proved target cyclic quotient or additional
   symmetry.

This strengthens the earlier constant-plus-H2 intertwiner: higher gate
Hermites do not leak outside (25.2), but they can expose the much larger
associative envelope hidden by a small first-order Jordan module.

### Temporal chaos-order return remains separate

Equation (25.3) is a spatial theorem, not permission to discard H6/H8.
Let `P` retain the chosen H2+H4 modules, `Q=I-P` retain the other even chaos
orders, and let `T_l` be the exact layer operator.  Put

\[
 a_l=\|PT_lP\|,quad b_l=\|PT_lQ\|,quad
 c_l=\|QT_lP\|,quad d_l=\|QT_lQ\|,
\qquad
 B_l=\begin{pmatrix}a_l&b_l\\c_l&d_l\end{pmatrix}.
\tag{25.7}
\]

Then the exact retained endpoint and the reset-by-reset truncation satisfy

\[
 \boxed{
 \left\|PT_L\cdots T_1P
 -PT_LP\cdots PT_1P\right\|
 \le
 \left(B_L\cdots B_1\right)_{11}-\prod_{l=1}^La_l.}
\tag{25.8}
\]

The right side is precisely the nonnegative sum over paths which leave the
retained chaos orders and later return, including multiple excursions.  The
first one-excursion terms contain products `b_j d_{j-1}...d_{i+1}c_i`.
Hence repeated H2+H4 resets have no `q>0.925` guarantee from the one-reset
`0.98334` figure: one must certify (25.8) in the endpoint norm, retain the
needed higher-order cyclic module, or use the local Bellman telescope of
Section 19 so the excursion is contracted before it is discarded.

### `H4_FULL_GATE_ASSOCIATIVE_ENVELOPE`

- **Status:** exact spatial closure and rank dichotomy solved.
- **Constructive gate:** commuting/block-reversible cores with
  `k_4/n` plus source/consumer cost inside the deployment cap.
- **Generic no-go:** full associative envelope gives `k_4` of order `r^4`;
  a low first-order Jordan dimension alone is not a receipt.

### `H4_HIGHER_CHAOS_RETURN_GATE`

- **Status:** exact block-norm majorant (25.8) solved.
- **Promotion of repeated H2+H4 resets:** endpoint-weighted bound from
  (25.8) must meet the correlation tolerance; otherwise use the Bellman
  contraction or retain the relevant H6/H8 cyclic modes.

## 26. Exact cyclic projector: what symmetric CP3 subtracts from the typed
## Stein provider

Recall the oriented lower moment

\[
 J_{ij,k}=E[T_{ij}H_k],\qquad J_{ij,k}=J_{ji,k},
\tag{26.1}
\]

and the convention

\[
 \kappa_{3,ijk}={2\over3}
 (J_{ij,k}+J_{ik,j}+J_{jk,i}).
\tag{26.2}
\]

Define `C:Sym^2(V) tensor V -> Sym^3(V)` by the right side of
(26.2), and let `E` be the literal embedding of a fully symmetric tensor
into the pair--singleton space.  With the ordinary Frobenius inner products,

\[
 \boxed{C^*=2E,\qquad CC^*=4I.}
\tag{26.3}
\]

Indeed, the unscaled three-term cyclic sum has adjoint `3E`, while the
factor `2/3` in (26.2) leaves `2E`.  Hence the exact orthogonal decomposition
is

\[
 \boxed{
 J={1\over2}E\kappa_3+J_\perp,qquad CJ_\perp=0.}
\tag{26.4}
\]

The scalar check is `J_{11,1}=tr(A^3)/2` and
`kappa_3(Q_A)=tr(A^3)`.  Therefore a fully symmetric CP3 lineage owns
exactly the first term of (26.4), not the oriented provider.  The residual
`J_perp` is invisible to every consumer which symmetrizes the pair and
singleton slots before contraction, but it remains in the typed Stein
Schur complement

\[
 K_T-JC^+J^T
\tag{26.5}
\]

and can be seen by a consumer retaining a distinguished pre-symmetrization
slot.  This is the precise sense in which it is CP3/H3-dark but potentially
H4/Stein-relevant.

### `H4_STEIN_ORIENTED_CYCLIC_QUOTIENT`

- **Status:** exact projector and normalization solved.
- **Old subtraction:** remove `E kappa_3/2`, not `E kappa_3` and not the
  whole oriented `J`.
- **Promotion of `J_perp`:** identify a literal distinguished-slot consumer
  and show nonzero endpoint Schur gain; a symmetric CP3 threshold JVP cannot
  see it.

## 27. Free-semicircular master closure: the whole AQ q4 tensor collapses to
## a `C,Q` pairing while retaining noncommutativity

There is one noncommutative closure in which the pair-feature square lift is
both exact and low-state.  Put

\[
 Q_{ij}={1\over2}\operatorname{tr}(A_iA_j),qquad
 q_i=Q_{ii},qquad L_{ij}=u_i^Tu_j,qquad C=L+Q,
\tag{27.1}
\]

and `c_ij=2Q_ij/r`.  Assume the quadratic cores obey the joint
semicircular fourth-word law

\[
 \operatorname{tr}(A_iA_jA_kA_l)
 =r(c_{ij}c_{kl}+c_{il}c_{jk}),
\tag{27.2}
\]

and the vector legs obey the **ordered** isotropy law

\[
 u_a^TA_iA_ju_b=L_{ab}c_{ij}.
\tag{27.3}
\]

Equation (27.3) is stronger than an identity only for the Jordan product;
every orientation used below is required.

For `S_ij=A_i circ A_j` and `b_ij=A_iu_j+A_ju_i`, free Wick contraction
gives

\[
 \boxed{
 \begin{aligned}
 \langle S_{ij},S_{kl}\rangle
 &= {4\over r}Q_{ij}Q_{kl}
   +{2\over r}(Q_{ik}Q_{jl}+Q_{il}Q_{jk}),\\
 \langle b_{ij},b_{kl}\rangle
 &= {2\over r}(L_{jl}Q_{ik}+L_{jk}Q_{il}
                   +L_{il}Q_{jk}+L_{ik}Q_{jl}).
 \end{aligned}}
\tag{27.4}
\]

Substitution in the exact three-pairing square-lift identity of Section 16
collapses the **complete external fourth cumulant** to

\[
 \boxed{
 \kappa_4(i,j,k,l)
 ={4\over r}\sum_{ab\mid cd\in\{ij\mid kl,ik\mid jl,il\mid jk\}}
 \{C_{ab}Q_{cd}+Q_{ab}C_{cd}\}.}
\tag{27.5}
\]

Thus the collision planes are

\[
 \boxed{
 \begin{aligned}
 K_4[i]&={24\over r}q_iC_{ii},\\
 K_{31}[i,j]&={12\over r}(q_iC_{ij}+C_{ii}Q_{ij}),\\
 K_{22}[i,j]&={4\over r}
 (q_iC_{jj}+C_{ii}q_j+4Q_{ij}C_{ij}).
 \end{aligned}}
\tag{27.6}
\]

The pure-core specialization is

\[
 K_4={24q_i^2\over r},\qquad
 K_{31}={24q_iQ_{ij}\over r},\qquad
 K_{22}={8\over r}(q_iq_j+2Q_{ij}^2).
\tag{27.7}
\]

This law is not a commuting approximation.  In normalized-trace notation,

\[
 \tau([A_i,A_j]^2)=2(c_{ij}^2-c_{ii}c_{jj}),
\qquad
 {1\over r}\|[A_i,A_j]\|_F^2
 =2(c_{ii}c_{jj}-c_{ij}^2),
\tag{27.8}
\]

so generic noncommutative `q4_perp` energy remains.  What collapses is its
external description: under (27.2)--(27.3), it is determined by the two
pair matrices `C,Q`.  A geometrically nonzero `q4_perp` is therefore not
automatically new information if those matrices and the law are owned.

### Exact defect tensor

The approximation can be falsified without an H4 tensor.  Define the ordered
defects

\[
 E_{ijkl}=\operatorname{tr}(A_iA_jA_kA_l)
 -r(c_{ij}c_{kl}+c_{il}c_{jk}),
\tag{27.9}
\]

\[
 D_{ab;ij}=u_a^TA_iA_ju_b-L_{ab}c_{ij}.
\tag{27.10}
\]

For a pair split `ij|kl`, put

\[
 \begin{aligned}
 E^S_{ij,kl}&={1\over4}
 (E_{ijkl}+E_{ijlk}+E_{jikl}+E_{jilk}),\\
 D^b_{ij,kl}&=D_{jl;ik}+D_{jk;il}+D_{il;jk}+D_{ik;jl}.
 \end{aligned}
\tag{27.11}
\]

Then the identity is exact:

\[
 \boxed{
 \kappa_4-\kappa_4^{free(C,Q)}
 =\sum_{ij|kl,ik|jl,il|jk}(E^S_{ab,cd}+D^b_{ab,cd}).}
\tag{27.12}
\]

This residual, contracted with the frozen endpoint adjoint after complete-old
subtraction, is the only model-error term.  Equation (27.12), rather than a
generic appeal to asymptotic freeness, is the promotion gate.

### Deployment interpretation

If `C` and the separated quadratic covariance `Q` are literal residents,
(27.6) is `O(n^2)` elementwise work and can inject the complete K4/K31/K22
faces into an existing plane consumer.  Equation (27.5) also gives any
target-contracted q211 action from pair-matrix operations, without a
`D=r(r+3)/2` provider bank.  It is an **alternative implementation** of
the full-AQ fourth cumulant, not additive capacity when full AQ already owns
that surface.  If only total `C` is resident and `Q` is not separately
typed, recovering `Q` remains a provider cost.

### `H4_FREE_SEMICIRCULAR_CQ_CLOSURE`

- **Status:** exact conditional master identity and collision faces solved.
- **Promotion:** certify the endpoint-weighted action of the explicit defect
  (27.12), separated `Q` residency, and exact old subtraction.
- **Potential value:** a noncommutative full-q4 replacement using only pair
  state; not a new feature beside full AQ.

### `H4_FREE_SEMICIRCULAR_DEFECT_GATE`

- **Status:** exact model-error tensor solved.
- **Pass:** the pulled-back action of (27.12), including bias coefficients
  and ten-layer accumulation, is below the declared endpoint tolerance.
- **Fail:** any large ordered free-Wick or vector-isotropy defect aligned
  with the endpoint; matching only K4 diagonals is insufficient.

## 28. Semicircular `C,Q` q4 implementation and the binding residual boundary

The master law (27.5) supplies a concrete explanation for the otherwise
disconnected scale observations.  Its repeated-index q211 face is

\[
 \boxed{
 \kappa_4(i,i,j,k)={4\over r}
 \{C_{ii}Q_{jk}+q_iC_{jk}
   +2(C_{ij}Q_{ik}+Q_{ij}C_{ik})\}.}
\tag{28.1}
\]

This includes the noncommutative commutator contribution but needs only the
separated pair states.  Because joint semicircularity and the tensor
factorization (27.3) are preserved under simultaneous linear mixtures of
all external indices, (27.5)/(28.1) are exactly closed under dense affine
transport:

\[
 C\mapsto WCW^T,\qquad Q\mapsto WQW^T.
\tag{28.2}
\]

The nonlinear reset is the only place where a new defect (27.12) can be
born; it can be charged locally before the next affine leg.

### Why the scale matches the variance-channel deficit but is not itself new capacity

For scalar direction `i`, (27.6) gives standardized kurtosis

\[
 \gamma_{4,i}={K_4[i]\over C_{ii}^2}
 ={24\over r}\eta_i,\qquad
 \eta_i={q_i\over C_{ii}}\in[0,1].
\tag{28.3}
\]

At zero bias and after separately accounting for the signed cubic and its
square, the first q4 variance correction is

\[
 {\Delta V_i\over V_{G,i}}
 ={\gamma_{4,i}-\gamma_{3,i}^2\over12(\pi-1)}
 ={2\eta_i\over r(\pi-1)}
  -{\gamma_{3,i}^2\over12(\pi-1)}.
\tag{28.4}
\]

In the low-skew, quadratic-dominant limit, rank `r=49` gives
`2/[49(pi-1)]`, about `1.9%` per reset.  Ten late resets therefore have a
first-order scale near `19%`, strikingly close to the independently located
`17.5--18.7` percentage-point variance correction.  This is a mechanism
and a scale explanation, not a new-value receipt: biases, `eta_i`, signed
propagation, `gamma_3^2`, and the explicit defect can reduce or reverse it.
More importantly, when `C,Q` describe the already-carried affine-quadratic
core `X=P_{<=2}Y`, this is the size of `kappa_4(X)`, which is already owned.
Numerical agreement with the missing variance value cannot override that
ownership fact.

At general bias define

\[
 m_1(a)=\phi(a)+a\Phi(a),\qquad
 V_0(a)=(1+a^2)\Phi(a)+a\phi(a)-m_1(a)^2.
\tag{28.5}
\]

The exact first q4 relative-variance coefficient is

\[
 B_4(a)=-{\phi(a)\{a+m_1(a)(a^2-1)\}\over12V_0(a)},
\tag{28.6}
\]

so the local free-law term is `24 B4(a) eta_i/r`, followed by the exact
q3-squared term of Section 20.  This bias telescope, rather than ten copies
of the zero-bias coefficient, is binding.

### Lawful ownership and implementation

Let `Y_owned` denote the comparator's literal non-Gaussian preactivation:
`X=P_{<=2}Y` when only the carried AQ core is owned, or `Y_AQ` when a fused
AQ fourth-surface oracle is already present.  For a genuinely enlarged
candidate `Y_cand`, the lawful new source is

\[
 \boxed{
 \Delta\kappa_{4,l}^{new}
 =R_{old,l}\left[
   \kappa_4(Y_{cand,l})-\kappa_4(Y_{owned,l})
  \right].}
\tag{28.7}
\]

Here `R_old,l` is the complete-old residual projector.  Formula (27.5) may
replace the first term only if `C,Q` are pair states of that enlarged
`Y_cand` and its defect (27.12) is certified.  If they are merely the states
of `X`, then (27.5) estimates an already-owned `kappa_4(X)` and (28.7) has no
new component.  If the comparator already owns the same free/full-AQ
surface, the residual is likewise zero.  Total q4 must never be appended
beside Q24/full AQ.

The valid implementation uses are:

1. **replacement:** retain separated `Q`, form (27.6)/(28.1), and replace
   a more expensive implementation of the already-owned AQ q4 surface;
2. **cost unlock:** use the saved work for an independently new residual
   provider, without claiming raw credit for (27.5);
3. **enlarged model:** only if a new `Y_cand-X` provider supplies its own
   `C,Q`, apply (27.5), subtract the literal owned q4 and q3-squared rows,
   and feed that residual into the existing variance tangent;
4. in every case contract (27.12) with the frozen endpoint adjoint and use
   the exact bias coefficient (28.6).

Once `C,Q` and the old planes are resident, the new formulas are `O(n^2)`
and lower order than one dense product.  Acquiring or separating `Q`, and
any absent reset consumer, remain billed.  This branch is best viewed as a
cheap q4 surface for the strict full-AQ schedule, or as a replacement for a
more expensive q4 provider.  With only the incumbent affine-quadratic states,
it is not additional physics.

### `H4_SEMICIRCULAR_CQ_VARIANCE_IMPLEMENTATION`

- **Status:** constructive analytic replacement/cost-unlock; no additive raw
  claim for incumbent `X`.
- **Why important:** it connects the rank-49 free-kurtosis scale to the
  independently observed ten-layer variance deficit and supplies its full
  external q4 shape at pair-state cost.
- **Promotion as replacement:** literal separated `Q`, lower total cost than
  the owned q4 implementation, and endpoint action of (27.12) below the
  replacement tolerance.
- **Promotion as raw feature:** additionally requires a genuinely enlarged
  `Y_cand-X` pair state and a nonzero complete-old Schur residual in (28.7).
- **Close additive interpretation for incumbent `X`:** unconditional.

## 29. Wedderburn/envelope cap: the exact algebraic dimension threshold

The associative-envelope result (25.3) has a direct cost screen.  A finite
real star-algebra has an orthogonal Wedderburn representation with simple
factors over `R,C,H`.  If the active irreducible sizes are `d_alpha`, the
real dimension of its self-adjoint part is

\[
 q=\sum_{\alpha:\mathbb R}{d_\alpha(d_\alpha+1)\over2}
   +\sum_{\alpha:\mathbb C}d_\alpha^2
   +\sum_{\alpha:\mathbb H}d_\alpha(2d_\alpha-1).
\tag{29.1}
\]

Representation multiplicities do not increase `q`; additional inequivalent
blocks do.  The universal exact H4 cap is still

\[
 k_4\le {q(q+1)\over2}.
\tag{29.2}
\]

If a width-`n` state transports `k_4` module coordinates through `L` dense
affine layers, the leading census in `n^3` equivalents is

\[
 G_{transport}={Lk_4\over n}.
\tag{29.3}
\]

Consequently an available budget `G_slack` imposes the exact preliminary
dimension gate

\[
 \boxed{
 k_4\le {nG_{slack}\over L},\qquad
 q\le\left\lfloor{sqrt{1+8nG_{slack}/L}-1\over2}\right\rfloor,}
\tag{29.4}
\]

before reset births, consumers, hooks, and memory.

At `n=1024,L=10`:

- a whole-card allowance of `49` permits `k_4<=5017` and hence `q<=99`;
- one additive GEMM permits `k_4<=102` and hence `q<=13`;
- the strict late-AQ remainder `0.803` permits `k_4<=82` and hence `q<=12`.

Thus a commuting rank-49 family (`q<=49,k_4<=1225`) is compatible with a
standalone exact H4 transport at about `11.96` equivalents, but not as an
addition to a `48.197` schedule.  A generic real rank-49 family has
`q=1225` and the naive `k_4` cap `750925`; a generic rank-55 family has
`q=1540,k_4=1186570`.  These are closed by many orders of magnitude.

The only valid relaxation is target cyclicity.  Let `Lambda_L` be the frozen
endpoint H4 covector and let `mathcal M_cyc` be the span generated by applying
all suffix adjoint gate/affine maps to it inside `mathcal K_4(mathcal V)`.
Replace `k_4` in (29.3)--(29.4) by

\[
 k_{cyc}=\dim\mathcal M_{cyc}.
\tag{29.5}
\]

This is exact for the declared endpoint, but not universal.  A small Jordan
list, CP name, or initial rank does not establish (29.5); one must prove
closure of the full terminal cyclic module under every live Kraus/path map.

### `H4_ASSOCIATIVE_ENVELOPE_CAP`

- **Status:** exact dimension/cost threshold solved.
- **Additive strict-AQ gate:** `q<=12` universally, or
  `k_cyc<=82` target-conditionally, before extras.
- **Standalone gate:** `q<=99` before extras under the full 49-equivalent
  allowance.
- **Close generic r49/r55 transport:** unconditional absent a certified
  cyclic quotient; the dimensions exceed the cap by orders of magnitude.

## 30. The CP3-dark Stein coordinate has an exact oriented `B_i` formula

The cyclic residual (26.4) simplifies much further for an AQ core.  Define
the oriented mixed cubic planes

\[
 B_i(j,k)=u_j^TA_i u_k,qquad B_i=B_i^T.
\tag{30.1}
\]

Then

\[
 J_{ij,k}={3\over4}\{B_i(j,k)+B_j(i,k)\}
          +{1\over2}\operatorname{tr}(A_iA_jA_k).
\tag{30.2}
\]

For symmetric matrices, the trace of a three-letter word is invariant under
every permutation: cyclicity and reversal give
`tr(A_iA_jA_k)=tr(A_{pi(i)}A_{pi(j)}A_{pi(k)})`.  Hence the entire trace term
lies in the symmetric cyclic image.  Subtracting (26.4) leaves

\[
 \boxed{
 J^\perp_{ij,k}={1\over4}
 \{B_i(j,k)+B_j(i,k)-2B_k(i,j)\}.}
\tag{30.3}
\]

The three-term cyclic sum of (30.3) vanishes identically.  It is zero against
every fully symmetric CP3 query, but is generally nonzero before the
pair--singleton slots are symmetrized.

### Exact low-CP target contraction

For a rank-one distinguished-slot reverse mask

\[
 \Lambda_{ij,k}=a_i b_j c_k,
\tag{30.4}
\]

put

\[
 \bar a=U^Ta,quad\bar b=U^Tb,quad\bar c=U^Tc,qquad
 M_a=\sum_i a_iA_i,quad M_b=\sum_i b_iA_i,quad M_c=\sum_i c_iA_i.
\tag{30.5}
\]

Substitution into (30.3) gives the exact scalar action

\[
 \boxed{
 \langle\Lambda,J^\perp\rangle
 ={1\over4}\{
   \bar b^TM_a\bar c+
   \bar a^TM_b\bar c-
   2\bar a^TM_c\bar b\}.}
\tag{30.6}
\]

If the complete mask is symmetric in `i,j`, the first two actions combine;
one symmetric CP atom needs at most two distinct fused matrix actions.
For CP rank `R`, (30.6) therefore avoids every `B_i` plane and uses two or
three latent core mixtures per atom.

Without reused mixtures, the leading cost over `L` seams is bounded by

\[
 G_{Jperp}^{generic}\simeq {3LRr^2\over n^2},
 \qquad
 G_{Jperp}^{sym}\simeq {2LRr^2\over n^2},
\tag{30.7}
\]

plus lower-order `U^T` matvecs and endpoint hooks.  At `n=1024,r=55,L=10`,
the coefficients are about `0.0865R` and `0.0577R`; the strict `0.803`
remainder permits at most roughly `R<=9` generic or `R<=13` symmetric atoms
before extras.  This is a target-conditioned route only; a generic
distinguished-slot surface has large CP rank.

### Ownership and value boundary

The exact typed Stein residual uses the full oriented lower projection
`JC^+H`, not only `E kappa_3 C^+H/2`.  Therefore a comparator which owns only
fully symmetric CP3 has not, by that fact alone, owned (30.3).  Conversely,
if its literal AQ/Stein tape already exposes the unsymmetrized `B_i` action,
(30.6) is duplicate.  Adding `Jperp` to an H4 Gram without updating the
Schur subtraction is also invalid: its role is the distinguished-slot
lower projection and associated residual action, not a free PSD q4 term.

### `H4_STEIN_JPERP_CP_ACTION`

- **Status:** exact source formula, CP contraction, and preliminary cap
  solved.
- **Promotion:** literal reverse mask with distinguished pair/singleton
  provenance, CP rank `R<=9/13` under the strict slack, complete-old audit,
  and nonzero endpoint Schur gain after the full `J` subtraction.
- **Close for symmetric CP3:** unconditional; a fully symmetrized threshold
  query annihilates (30.3).

## 31. The free `C,Q` cumulant is exactly tangent to the Gaussian Wick
## pairing manifold

Let

\[
 \mathcal P(C)_{ijkl}
 =C_{ij}C_{kl}+C_{ik}C_{jl}+C_{il}C_{jk}
\tag{31.1}
\]

be the Gaussian fourth-moment pairing tensor.  The master law (27.5) is
equivalently

\[
 \boxed{
 \kappa_4^{free(C,Q)}={4\over r}D\mathcal P_C[Q].}
\tag{31.2}
\]

This identity immediately reproduces every plane in (27.6) by
differentiating `3C_ii^2`, `3C_ii C_ij`, and
`C_ii C_jj+2C_ij^2`.  It also proves dense affine closure because
`mathcal P(WCW^T)=W^{tensor4}mathcal P(C)` and differentiation
intertwines the congruence.

For a smooth test function `f` and `Z_C~N(0,C)`, write
`mathcal D_M=M:D^2`.  The first Edgeworth action of (31.2) is exactly

\[
 \boxed{
 {1\over24}\kappa_4^{free}:E[D^4f(Z_C)]
 ={1\over r}E[(\mathcal D_C\mathcal D_Q)f(Z_C)].}
\tag{31.3}
\]

Indeed the six symmetric `C,Q` terms contract equally with `D^4f`; their
factor `(4/r)/24` leaves `1/r`.  Thus the free q4 surface is a nested Price
direction, which explains why its natural consumer is the variance tangent.

## 32. Zero-bias homogeneous annihilation: the free law cannot explain the
## measured noncollective variance value

Equation (31.3) has a sharp fail-fast consequence.  Suppose `f` is
homogeneous of degree `h`: `f(tx)=t^h f(x)`.  Put
`F(C)=E f(Z_C)`.  Gaussian Price differentiation and Euler homogeneity give

\[
 D_QF={1\over2}E[\mathcal D_Qf],
 \qquad E[\mathcal D_Cf]=hF.
\tag{32.1}
\]

Differentiate the second identity in direction `Q`.  Accounting both for
the explicit `C` and for the Gaussian law yields

\[
 E[\mathcal D_Qf]+{1\over2}E[\mathcal D_Q\mathcal D_Cf]
 ={h\over2}E[\mathcal D_Qf].
\tag{32.2}
\]

Therefore

\[
 \boxed{
 E[\mathcal D_C\mathcal D_Qf]
 =(h-2)E[\mathcal D_Qf],
 \qquad
 \Delta_f^{free}={h-2\over r}E[\mathcal D_Qf].}
\tag{32.3}
\]

At zero bias, every raw pair product
`f_ij(x)=(x_i)_+(x_j)_+` is homogeneous of degree two.  Hence

\[
 \boxed{\Delta E[(Z_i)_+(Z_j)_+]^{free}=0}
\tag{32.4}
\]

for all `i,j`, not only on the diagonal.  The mean is degree one, so

\[
 \Delta\mu_i^{free}
 =-{1\over r}E[\mathcal D_Q(x_i)_+]
 =-{phi(0)q_i\over r\sqrt{C_{ii}}}.
\tag{32.5}
\]

Consequently the complete first-order centered-covariance correction is

\[
 \boxed{
 \Delta\operatorname{Cov}_{ij}^{free}
 =-{\mu_i\Delta\mu_j+\Delta\mu_i\mu_j}
 ={\phi(0)^2\over r}\left{
 \sqrt{C_{ii}}{q_j\over\sqrt{C_{jj}}}
 +{q_i\over\sqrt{C_{ii}}}\sqrt{C_{jj}}
 \right}.}
\tag{32.6}
\]

This matrix has rank at most two.  Its diagonal reproduces exactly
`2 eta_i/[r(pi-1)]` relative centered-variance correction from (28.4), but
all of that zero-bias correction is a collective mean-counterterm outer
product.  It is not the broad-spectrum q211/variance plane which the value
map found across late layers.

Therefore the numerical `r=49`, ten-layer `~19%` scale match in Section 28
is structurally misleading as an explanation of the measured noncollective
value.  Given the independent result that the collective projection is
about `170x` too small and essentially uncorrelated with `dvhat`, the missing
valuable source cannot be the exact zero-bias free closure (31.2).  It must
come from at least one of:

1. the ordered free-Wick/isotropy defect (27.12);
2. nonzero-bias terms, which break degree-two homogeneity;
3. signed cubic/cubic-square interactions not absorbed by the free q4 law;
4. a genuinely new noncollective provider outside incumbent AQ.

### `H4_FREE_CQ_HOMOGENEOUS_NOGO`

- **Status:** exact zero-bias annihilation and rank-two centered correction
  solved.
- **Close as explanation of noncollective `dvhat`:** exact free law at zero
  bias, unconditionally.
- **Reopen:** quantify the endpoint action of explicit defect (27.12) or the
  exact bias telescope; the scalar diagonal scale alone is no evidence.

## 33. Freely independent non-semicircular extension: the fourth-word defect
## is exactly CP4 Star

The free-Wick defect (27.9) is itself not a generic H4 tensor when the cores
come from freely independent sources.  Let centered source cores `B_a` be
freely independent, allow arbitrary individual third/fourth free cumulants,
and let

\[
 A_i=\sum_a W_{ia}B_a.
\tag{33.1}
\]

With normalized trace `tau`, multilinearity and vanishing mixed free
cumulants give

\[
 \boxed{
 \begin{aligned}
 \tau(A_iA_jA_k)
 &=\sum_aW_{ia}W_{ja}W_{ka}\,\kappa_3^\tau(B_a),\\
 E_{ijkl}
 &=r\sum_aW_{ia}W_{ja}W_{ka}W_{la}\,
       \kappa_4^\tau(B_a),
 \end{aligned}}
\tag{33.2}
\]

where `E` uses the unnormalized trace convention of (27.9).  Thus the
degree-three trace word is a coordinate CP3 Star and the entire deviation
from the free `C,Q` fourth-word pairing is a coordinate CP4 Star.  No mixed
four-index provider remains in the matrix-core sector.

Combined with Section 32, a zero-bias freely independent matrix-core family
decomposes as follows:

1. the semicircular `C,Q` term has zero raw degree-two ReLU-pair action and
   only a rank-two mean-counterterm covariance correction;
2. all noncollective degree-three/four value is in literal CP3/CP4 Star
   sources (33.2);
3. the affine vector-leg defect `D` in (27.10) is separate and is not removed
   by matrix freeness alone.

Therefore, if the synchronized comparator really retains the CP3/CP4 Star
sources through the same affine/reset cuts, the complete freely independent
matrix-core q4 law is old-owned.  Failure to retain the source buffers can
make (33.2) a cheaper reconstruction seam, but not a new physical feature.
Only the vector-isotropy defect, nonfree mixed cumulants, bias-induced
nonhomogeneity, or true-minus-AQ residual can remain novel.

### `H4_FREE_SOURCE_STAR_DECOMPOSITION`

- **Status:** exact free-cumulant/CP3/CP4 decomposition solved.
- **Ownership gate:** audit literal source amplitudes, affine lineage, and
  collision complement; family names do not suffice.
- **Close as additive capacity:** if both Star sources and the `C,Q` pairing
  surface are synchronized old-owned.

## 34. Exact rank-`r` carrier for the affine vector-leg q4 residual

The vector-isotropy defect left open by Sections 32--33 is not a generic
four-tensor.  For each latent coordinate `alpha`, define the symmetric
external pair matrix

\[
 B^\alpha_{ij}=(b_{ij})_\alpha,qquad
 b_{ij}=A_iu_j+A_ju_i.
\tag{34.1}
\]

The complete vector part of the AQ fourth cumulant is exactly

\[
 \boxed{
 \kappa_4^{vec}(i,j,k,l)
 =\sum_{\alpha=1}^r\{
 B^\alpha_{ij}B^\alpha_{kl}
 +B^\alpha_{ik}B^\alpha_{jl}
 +B^\alpha_{il}B^\alpha_{jk}\}.}
\tag{34.2}
\]

Thus its pair flattening is the Gram of `r`-dimensional features and has
rank at most `r`, regardless of the `n(n+1)/2` pair count.  The visible
collision planes are

\[
 \boxed{
 \begin{aligned}
 K_4^{vec}[i]&=3\|b_{ii}\|^2,\\
 K_{31}^{vec}[i,j]&=3\langle b_{ii},b_{ij}\rangle,\\
 K_{22}^{vec}[i,j]&=\langle b_{ii},b_{jj}\rangle+2\|b_{ij}\|^2.
 \end{aligned}}
\tag{34.3}
\]

Dense affine transport is exact at the pair-matrix level:

\[
 B^\alpha\longmapsto WB^\alpha W^T,
\tag{34.4}
\]

provided the latent basis is held fixed.  This route retains broad external
structure even though its provider rank is only `r`; unlike the free `C,Q`
term, it need not collapse to a collective outer product.

### Formation and sketch costs

Forming all `b_ij` directly from `A_i,u_j` costs leading order
`n^2r^2`, or

\[
 G_{b,full}\simeq {r^2\over n}
\tag{34.5}
\]

dense equivalents per reset.  This is about `2.35` at `r=49` and `2.95` at
`r=55`; six full resets cost about `14.1` or `17.7`, before consumers.
It does not fit as an addition to the strict `48.197` schedule.

For an orthogonal latent projection `P=S^TS` of rank `k`, form
`\widetilde b_{ij}=Sb_{ij}` directly.  The leading batched cost becomes

\[
 G_{b,k}\simeq {kr\over n}
\tag{34.6}

per reset, plus the existing pair consumer.  At `r=49,k=21`, six resets are
about `6.0` equivalents.  A frozen target-conditioned contraction such as
(30.6) can be much cheaper still because it never materializes pair planes.

### Exact endpoint projection gate

Let `mathcal B` be the pair-by-latent matrix with rows `b_ij^T`.  For a fully
symmetric endpoint quartic covector `Lambda`, let `L_Lambda` be its
pair-flattening.  Then

\[
 \boxed{
 \langle\Lambda,\kappa_4^{vec}\rangle
 =3\operatorname{tr}(\mathcal B^T L_\Lambda\mathcal B).}
\tag{34.7}
\]

After latent projection `P`, the exact signed action error is

\[
 \boxed{
 e_P=3\operatorname{tr}
 \{(I-P)\mathcal B^TL_\Lambda\mathcal B\},}
\tag{34.8}
\]

and a universal sufficient bound is

\[
 |e_P|\le3\|L_\Lambda\|_{op}
          \|\mathcal B(I-P)\|_F^2.
\tag{34.9}
\]

For multiple nets/cuts, sum the signed matrices
`mathcal B_e^T L_{Lambda_e} mathcal B_e` with the exact endpoint weights;
the frozen top-`k` subspace or generalized eigenspace is the lawful
candidate.  Raw feature energy or an unweighted SVD of `b` is not the gate.
To guarantee task correlation above `q_star` by orthogonal retained-energy
projection, the endpoint-weighted tail must be below `1-q_star^2`; for
`q_star=.925` this is `0.144375`.

### `H4_VECTOR_LEG_RANKR_CARRIER`

- **Status:** exact rank, affine closure, collision planes, and projection
  error solved.
- **Promotion:** prove the vector leg is not already in full-AQ/Q24, identify
  the endpoint-weighted rank `k` meeting (34.8), and fit total cost by
  replacement or local reverse contraction.
- **Close full forward addition:** under the strict schedule, because even
  six rank-21 projected resets exceed the remaining `0.803` allowance.

## 35. Exact inverse bridge: oriented `J_perp` completes the vector q4 Gram

Let `U` be the `n x r` matrix whose row `k` is `u_k^T`, let
`L=UU^T`, and arrange the vectors `b_ij^T` as the rows of the pair-by-latent
matrix `mathcal B`.  Define the oriented observation

\[
 Y_{ij,k}=\langle b_{ij},u_k\rangle.
\tag{35.1}
\]

Then

\[
 Y=\mathcal B U^T,
\qquad
 Y_{ij,k}={4\over3}J_{ij,k}
           -{2\over3}\operatorname{tr}(A_iA_jA_k).
\tag{35.2}
\]

Let

\[
 P_U=U^T(UU^T)^+U
\tag{35.3}
\]

be the orthogonal projector onto the row span of `U`.  The exact vector-pair
Gram decomposes as

\[
 \boxed{
 \mathcal B\mathcal B^T
 =YL^+Y^T
  +\mathcal B(I-P_U)\mathcal B^T.}
\tag{35.4}
\]

The second term is PSD and invisible to every contraction of `b_ij` against
the existing linear cores.  It vanishes iff the rows `u_k` span the full
latent space.  In particular, if `rank U=r`, the complete vector q4 carrier
is algebraically determined by the oriented `J` and the cubic trace word.

Now split `J=E kappa_3/2+J_perp` as in (26.4).  Equation (35.2) becomes

\[
 Y=Y_\parallel+Y_\perp,
 \qquad
 Y_\perp={4\over3}J_\perp,
\tag{35.5}
\]

where `Y_parallel` is fixed by symmetric CP3 plus the trace Star.  Therefore

\[
 \boxed{
 \begin{aligned}
 YL^+Y^T={}&Y_\parallel L^+Y_\parallel^T\\
 &+Y_\parallel L^+Y_\perp^T
  +Y_\perp L^+Y_\parallel^T
  +Y_\perp L^+Y_\perp^T.
 \end{aligned}}
\tag{35.6}
\]

This is the exact no-double-count decomposition of the affine vector square
lift.  Symmetric CP3 owns only the first term.  The novel oriented completion
contains two signed cross terms and a PSD square; it is not lawful to append
only `J_perp L^+ J_perp^T` or to treat `Jperp` as an independent linear q4
feature.  If `rank U<r`, the final dark PSD term in (35.4) must also be
supplied or bounded.

### `H4_VECTOR_GRAM_ORIENTED_J_COMPLETION`

- **Status:** exact inverse, rank-deficiency remainder, and cyclic
  no-double-count split solved.
- **Promotion:** certify `rank U=r` or bound the endpoint action of the dark
  PSD remainder; acquire `Jperp` with distinguished-slot provenance; include
  every cross term in (35.6); residualize against literal CP3/Star ownership.
- **Consequence:** the rank-`r` vector carrier and the oriented Stein lane are
  the same information when `U` has full latent rank, not stackable gains.

## 36. Exact arbitrary-bias extension of the free-law homogeneous theorem

The possible bias reopening in Section 32 can be characterized without an
H4 tensor.  Let `f_b(x)` be a degree-`h` homogeneous ridge product after
adjoining its biases, so that

\[
 f_{\sqrt t b}(\sqrt t x)=t^{h/2}f_b(x).
\tag{36.1}
\]

For `F(C,b)=E f_b(Z_C)`, this implies

\[
 D_CF[C]+{1\over2}b\cdot\partial_bF={h\over2}F.
\tag{36.2}
\]

Repeat the Price differentiation in Section 32 at fixed bias.  With
`mathcal D_Q=Q:D^2`, one obtains

\[
 \boxed{
 E[\mathcal D_C\mathcal D_Q f_b]
 =(h-2)E[\mathcal D_Qf_b]
  -b\cdot\partial_bE[\mathcal D_Qf_b].}
\tag{36.3}
\]

For a biased ReLU pair
`f=(x_i+b_i)_+(x_j+b_j)_+`, `h=2`, so the entire free q4
raw-pair correction is

\[
 \boxed{
 \Delta F_{ij}^{free}
 =-{1\over r}(b_i\partial_{b_i}+b_j\partial_{b_j})
 E[\mathcal D_Q f].}
\tag{36.4}
\]

It vanishes continuously at zero bias and is determined entirely by the
ordinary Gaussian covariance JVP and its threshold derivative.  Explicitly,

\[
 \begin{aligned}
 E[\mathcal D_Qf]={}&q_iE[\delta(X_i+b_i)(X_j+b_j)_+]\\
 &+2Q_{ij}P(X_i>-b_i,X_j>-b_j)\\
 &+q_jE[(X_i+b_i)_+\delta(X_j+b_j)],
 \end{aligned}
\tag{36.5}

for the bivariate Gaussian marginal with covariance entries from `C`.
All terms and bias derivatives reduce to univariate/bivariate Gaussian
densities and CDFs.

Equation (36.4) is the exact bias gate missing from the scalar scale
argument.  It can produce a noncollective matrix when biases vary by neuron,
but it remains part of the already-owned free `C,Q` AQ surface.  It is useful
as a cheap replacement and as a quantitative bound on how far the zero-bias
rank-two no-go can be reopened; it is not additive capacity without a new
`Y_cand-X` provider.

### `H4_FREE_CQ_BIAS_JVP_IDENTITY`

- **Status:** exact arbitrary-bias reduction to Gaussian JVP derivatives
  solved.
- **Promotion as implementation:** evaluate (36.5) and its analytic bias
  derivative on the literal `C,Q,b` state and beat the existing q4 surface
  cost within replacement tolerance.
- **Raw novelty:** closed for incumbent `X`; only a genuinely enlarged
  residual or the explicit nonfree/vector defects can receive credit.

## 37. Literal reverse-tape audit closes the cheap `J_perp` action on the
## current D10/r49 design

The direct-diagram audit supplies the missing provenance result for Section
30.  Current D10 exposes forward covariance/response planes and same-site
equality adjoints; the proposed r49 lineage is a forward `(m,u,A)` state.
Neither contains a resident Bellman reverse tape nor an unsymmetrized
pair--singleton `J_{ij,k}` consumer.  Every symmetric H3 query factors
through the cyclic map and annihilates `Jperp` exactly.

Moreover, a hypothetical rank-one endpoint query does not remain low rank
under a generic ReLU covariance pullback.  Its pair slice acquires the form

\[
 \operatorname{Diag}(v)\,\Gamma\,\operatorname{Diag}(v),
\tag{37.1}
\]

with `Gamma` generically full rank.  Thus a low-CP endpoint name is not a
low-CP strict-depth distinguished mask.  The `R<=9/13` census in Section 30
is only a conditional promotion gate after a literal complete-old-projected
pullback factorization; current evidence does not meet it.

### `H4_STEIN_JPERP_CURRENT_TAPE_NOGO`

- **Status:** current-lineage provenance closed; conditional future route
  retained.
- **Close now:** D10/r49 has no distinguished-slot reverse consumer, and
  symmetric CP3 gives zero action.
- **Reopen:** build and bill an explicitly typed Bellman reverse tape, then
  prove its strict-depth CP rank stays at most `9/13` after old projection.

## 38. Exact collision-face factorization isolates the hard vector triangle

The vector carrier (34.2) can be decomposed into resident-sized matrix
contractions.  Put

\[
 p_i=A_i u_i,qquad h_i=A_i^2u_i,qquad
 Z_i=\operatorname{sym}(u_ip_i^T),qquad D_i=A_i^2.
\tag{38.1}
\]

Let `P,H` stack `p_i^T,h_i^T`; let `mathsf A,mathsf Z,mathsf D,mathsf R`
stack orthonormal symmetric vectorizations of
`A_i,Z_i,D_i,u_i u_i^T`; and define the symmetric orientation triangle

\[
 T_{ij}=u_j^TA_iA_ju_i=T_{ji}.
\tag{38.2}
\]

Then the exact planes are

\[
 \boxed{
 \begin{aligned}
 K_4^{vec}[i]&=12\|p_i\|^2,\\
 K_{31}^{vec}&=6\{HU^T+\mathsf Z\mathsf A^T\},\\
 K_{22}^{vec}&=4PP^T
   +2\{\mathsf D\mathsf R^T+\mathsf R\mathsf D^T\}+4T.
 \end{aligned}}
\tag{38.3}
\]

All terms except `T` are ordinary rectangular Grams in dimensions `r` or
`d=r(r+1)/2`.  The triangle can be written

\[
 T_{ij}=\langle A_iu_j,A_ju_i\rangle;
\tag{38.4}
\]

forming it generically requires the oriented edge bank
`V_{ij}=A_i u_j`, whose leading construction cost is the `n^2r^2` term in
(34.5).  It is exactly the identity-specific, noncollective part erased by
the ordered-isotropy/free closure.  A K31-only consumer can avoid the edge
bank using the second line of (38.3), but full K22 cannot claim that saving
without a separate low-rank/typed receipt for `T`.

At `r=49`, the symmetric-vector dimension is `d=1225`; one flattened product
costs about `d/n=1.20` dense equivalents per reset.  Thus (38.3) is useful
mainly for literal CSE or a K31-only seam.  If all faces are needed, forming
the common edge bank once (about `r^2/n=2.35` per reset) and reducing it is
cheaper than separately paying several `d`-dimensional products.

### `H4_VECTOR_TRIANGLE_BOTTLENECK`

- **Status:** exact plane split and hard contraction identified.
- **Promotion of a partial K31 seam:** prove its endpoint value and reuse of
  `mathsf A`/`mathsf Z`; do not infer K22 ownership.
- **Full vector q4:** requires resident `V_ij`, an exact low-rank triangle
  factorization, or the target-conditioned reverse route already closed on
  the current tape.

## 39. Literal strict-FULL-AQ audit: the vector edge bank is not resident

The declared strict full-AQ forward state retains the per-unit coefficient
bank `(u_i,vecsym(A_i))` of width `r(r+3)/2` and its split Grams

\[
 L_{ij}=u_i^Tu_j,qquad Q_{ij}={1\over2}\operatorname{tr}(A_iA_j).
\tag{39.1}
\]

It does **not** retain `A_iU^T`, the cross vectors `A_iu_j`, or the pair
features `b_ij`.  Per-unit eigensystems/quadrature may form `A_i u_i`
locally, but that supplies only the diagonal edge and cannot form the
orientation triangle (38.2).

Therefore (34.5) is genuinely incremental: about `2.35` dense equivalents
per reset at r49 or `2.95` at r55.  Six generic vector-q4 resets add about
`14.1/17.7`, which is fatal on top of the strict schedule.  The only current
algebraic escapes are:

1. ordered-isotropy collapse to the old `C,Q` surface, which Section 32
   shows cannot supply the valuable noncollective zero-bias mode;
2. a proved low-`k` target projection, for which the current forward/reverse
   tapes have no receipt;
3. a new same-destination fusion which constructs the required edge action
   while an already-paid contraction is live.

### `H4_VECTOR_EDGE_BANK_LITERAL_NOGO`

- **Status:** current strict-state CSE closed.
- **Do not claim:** split `L,Q` Grams or local `A_i u_i` as ownership of
  `A_i u_j`.
- **Reopen:** exact fused contraction with incremental census, or a literal
  endpoint-conditioned low-rank mask/subspace.

## 40. Exact constructive exception: a commuting core makes the hard vector
## triangle a cheap Gram

Assume all live symmetric cores share an orthogonal eigenbasis:

\[
 A_i=O\operatorname{Diag}(a_i)O^T,qquad
 \widetilde u_i=O^Tu_i,qquad
 p_i=a_i\odot\widetilde u_i.
\tag{40.1}
\]

Then the orientation triangle of Section 38 collapses exactly:

\[
 \boxed{
 T_{ij}=\widetilde u_j^T\operatorname{Diag}(a_i a_j)
          \widetilde u_i=p_i^Tp_j,\qquad T=PP^T.}
\tag{40.2}
\]

Even more strongly, the latent pair matrices in (34.1) are

\[
 B^\alpha
 =a^{(\alpha)}(\widetilde u^{(\alpha)})^T
  +\widetilde u^{(\alpha)}(a^{(\alpha)})^T,
\tag{40.3}
\]

so every `B^alpha` has external rank at most two.  All vector-q4 collision
faces can be formed in `O(n^2r)` work from the two `n x r` coefficient
matrices; no cross edge bank is needed.

The property is stable under dense channel mixing because `a,u` mix only in
their external row index, and it is stable under polynomial quadratic-core
resets because diagonal matrices remain diagonal.  Transporting both
coefficient banks costs about `2Lr/n` dense equivalents; at `r=55,L=10`
this is about `1.07`, before the lower-order plane reductions.  Thus a
commuting full H2+H4 square/vector carrier is a genuine `O(n^3)` constructive
architecture.

The condition is also exact and falsifiable.  The matrix-core q4 identity
already contains

\[
 \|[A_i,A_j]\|_F^2,
\tag{40.4}
\]

and the independently valuable `q4_perp`/commutator channel is precisely
what simultaneous diagonalization removes.  Approximate use therefore
requires a common-basis residual bound on both the matrix square Gram and
the vector triangle, pulled through the endpoint; small average pair
commutator energy alone is insufficient without a joint approximate
diagonalizer and conditioning bound.

### `H4_COMMUTING_VECTOR_CARRIER`

- **Status:** exact constructive closure and cost solved.
- **Promotion:** literal common eigenbasis (or certified endpoint-small
  residual), total cost including plane consumer, and nonzero value not
  already in `C,Q`/Stars.
- **Current caution:** the known noncommutative q4 value makes the exact
  commuting hypothesis unlikely for the desired residual; do not use it as
  a generic approximation.

## 41. Exact CP4-ray exception to the generic distinguished-mask no-go

The completed q211 normal-form audit confirms that generic complete-old
coordinate-anchor blocks have singleton rank up to `n-2`, and dense resets
can generate the full cyclic module.  There is no universal low-CP
distinguished query.  One literal CP4 endpoint ray is an exact exception.

For a ray `v^4`, its strict-q211 birth weight has the partially symmetric
form

\[
 \Lambda_{ij,k}=v_i^2v_jv_k
\tag{41.1}
\]

(with factor `6` for the ordered strict-q211 tensor contraction, plus the
typed collision complement).  In (30.6), take

\[
 a=v^{\odot2},\qquad b=c=v,
\tag{41.2}
\]

and define

\[
 \bar v=U^Tv,quad \bar v_2=U^T(v^{\odot2}),qquad
 M_v=\sum_i v_iA_i,quad M_2=\sum_i v_i^2A_i.
\tag{41.3}
\]

The exact oriented action collapses to

\[
 \boxed{
 \langle\Lambda,J^\perp\rangle
 ={1\over4}\{
 \bar v^TM_2\bar v-\bar v_2^TM_v\bar v\}.}
\tag{41.4}
\]

Thus one proven endpoint ray needs only two latent core mixtures/actions and
no `B_i` bank.  A sum of `R` literal rays costs the symmetric figure in
(30.7), subject to collision hooks; the strict slack still limits the
constant live rank to roughly `R<=13` before extras.

This exception is narrow.  Complete-old projection can turn one ray into a
sum, and a generic future ReLU pullback creates full pair slices.  Equation
(41.4) is lawful only when all of the following are literal:

1. the endpoint/source provenance is the declared CP4 ray through the whole
   relevant suffix;
2. the factor-6 ordered strict-q211 convention is used;
3. K4/K31/K22 and old-Star collision leakage are sewn exactly;
4. the affine `b_ij` source survives the old projector rather than being
   replaced by a generic q211 residual.

### `H4_JPERP_LITERAL_CP4_RAY_EXCEPTION`

- **Status:** exact two-action formula solved; current provenance open.
- **Promotion:** at most 13 fully traced rays under strict slack, exact
  collision sewing, and nonzero complete-old endpoint value.
- **Generic extension:** closed by the q211 anchor/singleton rank lower bound
  and dense-reset cyclic growth.

## 42. Updated lane verdict: exact elimination tree and the only live H4
## exceptions

The square-lift investigation now has a complete structural split.

1. **Generic persistent H4:** closed.  The complete ReLU gate is spatially
   closed only in the self-adjoint associative envelope, whose generic r49/
   r55 module is orders of magnitude above the deployment cap.  H6/H8 return
   also prevents multiplying the one-reset `0.98334` figure across depth.
2. **Semicircular matrix core:** exact cheap replacement, no additive value.
   Its q4 tensor is `(4/r)` times the derivative of the Wick-pairing map at
   `C` in direction `Q`; at zero bias it annihilates raw
   ReLU pair moments and changes centered covariance only through a rank-two
   mean counterterm.  It cannot explain the measured noncollective `dvhat`.
3. **Freely independent non-semicircular matrix core:** decomposes into the
   same cheap `C,Q` term plus literal CP3/CP4 Stars.  It is old if those
   source lineages are synchronized.
4. **Affine vector leg:** exact rank-`r` pair carrier and the only compact
   broad/noncollective AQ component.  It is equivalent to the oriented
   `Jperp` completion when `U` has full latent rank.  The current full-AQ
   state does not retain its cross edge bank; six forward resets cost about
   `14.1` extra at r49.  The current reverse tape has no distinguished
   low-rank query, so neither forward nor reverse generic acquisition fits.
5. **Two exact exceptions remain:**
   - a common commuting eigenbasis, which reduces the vector triangle to a
     cheap Gram but removes the known valuable commutator channel; or
   - at most roughly 13 literally traced CP4 endpoint rays, using (41.4)
     with exact collision/old sewing.

Consequently H4 square fidelity by itself is not a path from current raw
MSE to `1e-8`.  A live proposal must now exhibit one of three receipts:

\[
 \boxed{
 \begin{array}{l}
 \text{(a) a genuinely new true-minus-AQ vector/nonfree residual,}\cr
 \text{(b) a literal small CP4-ray or commuting cyclic exception,}\cr
 \text{(c) a cheaper replacement which frees budget for another proven lane.}
 \end{array}}
\tag{42.1}
\]

Anything else reintroduces an old q4 surface, a collective mode already
measured dead, or a carrier above the 49-equivalent cap.  The shortest next
proof is not another generic H4 approximation: it is the literal endpoint
provenance/rank audit for (41.4), or a same-destination fusion of the vector
triangle with a paid contraction.  If both fail, this lane should remain a
replacement theorem and no-go certificate rather than a raw-gain claim.

### `H4_SQUARE_LIFT_FINAL_STRUCTURAL_VERDICT`

- **Status:** generic additive lane closed; two narrow exceptions open.
- **Do not stack:** free `C,Q`, CP Stars, vector/Jperp, and full AQ without
  exact residual ownership; these are overlapping decompositions.
- **Next gate:** literal CP4-ray provenance with rank at most 13, or exact
  same-destination vector-triangle fusion under the cap.

## 43. Independent audit of pair-kernel rank, Hadamard CP4 faces, strict
## q211 trace collapse, and the three-node Jacobi closure

### 43.1 Biased ReLU pair Jacobian

For standardized thresholds, the covariance Jacobian

\[
 p_{ij}=P(Z_i>-a_i,Z_j>-a_j)=\Phi_2(a_i,a_j;R_{ij})
\tag{43.1}
\]

and its Mehler expansion in reverse-lane (77.2) are normalized correctly.
At zero bias,

\[
 p_{ij}={1\over4}+{\arcsin R_{ij}\over2\pi},
 \qquad
 \epsilon_1(\chi)={\arcsin\chi-\chi\over2\pi}
\tag{43.2}
\]

are exact, as is the entrywise Frobenius remainder bound.

There is one notation requirement.  In
`R^{odot q}=S^{odot q}(S^{odot q})^T`, the factor on the right must mean the
rowwise **symmetric Khatri--Rao tensor power**.  In a unique-monomial basis
its columns carry square-root multinomial weights.  Literal entrywise powers
of the columns of `S` do not give `R_ij^q` and generally have the wrong
dimension.  With the tensor-power meaning, the binomial rank bound is exact.

The low-rank completion is also explicitly a quotient statement.  Replacing
the physical pair diagonal by the diagonal of the low-rank completion is
lawful only when `j=k` collisions are read and subtracted separately.  In a
strict q211 action, the anchor collisions `i=j` and `i=k` must additionally
be sewn through the synchronized K31/K4 faces.  A diagonal completion alone
does not perform that anchor-dependent exclusion.

### 43.2 Hadamard CP4 K31: raw flat spectrum versus strict quotient

For normalized Hadamard columns and raw tensor-entry planes, reverse-lane
(78.2)--(78.4) are exact:

\[
 K_{31}={1\over n}U\operatorname{Diag}(\theta)U^T,
 \qquad \sigma_s(K_{31})={|\theta_s|\over n},
\tag{43.3}
\]

and the natural `K22` completion is rank one.  The equal-amplitude tail
`1-k/m` is therefore correct for this **full natural K31 completion**.

It is not automatically a lower bound for the strict diagonal quotient.
For the clearest counterexample, take `m=n` and all amplitudes equal with the
same sign.  Orthogonality gives

\[
 K_{31}={\theta\over n}I,
\tag{43.4}
\]

whose full singular spectrum is perfectly flat while its strict off-diagonal
face is identically zero.  Balanced signed amplitudes make the natural
diagonal zero and recover a genuine flat strict matrix, but a lawful
arbitrary diagonal completion can still lower its rank unless a diagonal-
rigidity theorem is supplied.  Hence the `k>=991` statement is valid only
for a fixed full-plane/post-old spectrum whose diagonal is binding, or after
proving the same tail for the minimum-rank diagonal completion or actual
endpoint action.  It cannot be inferred from Hadamard diffuseness alone.

### 43.3 Constant-modulus trace collapse is exact modulo collisions

Let `M_{i;(jk)}=1_{i,j,k distinct}`.  The literal zero-filled strict
flattening is

\[
 \boxed{
 F^{strict}=M\odot
  [D\operatorname{Diag}(\theta)B^T].}
\tag{43.5}
\]

The unmasked product in reverse-lane (79.2) is a collision-completed
representative.  Its ordinary matrix rank is therefore not, without this
qualification, the rank of the literal zero-filled strict flattening.

When every factor has constant squared coordinates, the completed
representative becomes `1 r^T` with

\[
 R=\sum_s\theta_sc_su_su_s^T,
\tag{43.6}
\]

and its action on a strict costate is exactly the Section-70 trace action
after summing only `i notin {j,k}`.  Thus the rank-one claim is exact in the
collision quotient/consume-and-retire sense.  The constant-modulus defect
must likewise be written with the mask in an exact strict formula.  The
unmasked Frobenius bound remains a valid robust upper bound because the
mask is contractive.

Finally, distinguish two trace normalizations.  The strict insertion `ZR`
has entries `R_jk` on `iijk` and no declared collision faces.  The full
six-term tensor `mathcal Z_R` completing it has K31 face `3R`.  The raw CP4
Hadamard plane in (43.3) has K31 `R`, not `3R`; exact collision sewing must
restore the difference rather than identifying the full tensors.

### 43.4 Three-node spectral/Jacobi K31 closure

The cubic residual theorem is normalized correctly:

\[
 R=B^3-\mu_4B-\mu_3I,qquad
 \|R\|_\tau^2=\mu_6-\mu_4^2-\mu_3^2.
\tag{43.7}
\]

The r55 semicircle fidelity near `.926`, the sixth-moment gates, the unique
symmetric law

\[
 {1\over4}\delta_{-\sqrt2}+{1\over2}\delta_0
 +{1\over4}\delta_{\sqrt2},
 \qquad B^3=2B,
\tag{43.8}
\]

and the independent-Haar alternating-word multiplier all check.

One logical statement needs correction.  Equality in (43.7) means

\[
 B^3\in\operatorname{span}\{I,B\},
\tag{43.9}
\]

not merely `span{I,B,B^2}`.  A general three-node Jacobi law obeys

\[
 B^3=aB^2+bB+cI.
\tag{43.10}
\]

When `a` is nonzero, it does not make the Q-only cubic residual vanish; it
moves that residual exactly into the signed H3 plane `tr(B_i^2B_j)`.  This
is a valid K31 cost reduction only if that H3 plane is literally resident
with matching normalization.  In half-gauge physical K31, the raw
normalized-trace identity must also be multiplied by the outer repeated-
cumulant factor (`3`, and `r` when converting `tau` to `tr`).

Per-core Gaussian quadratures generally have per-core coefficients/nodes;
a claimed common closure requires a shared spectral law or must retain the
row-dependent coefficients.  Dense mixing destroys (43.10), so retraction
must occur at every declared seam and its Bellman error must be charged.

This Jacobi result does not change the H4 square-lift disposition.  It can
remove or replace the **matrix K31 collision read** when H3 is owned, but it
does not supply crossed K22/q4perp, all-distinct q4, or the affine vector
pair leg of Sections 34--39.  It is a cost unlock, not an additive H4 raw
feature.

### Audit graph nodes

- `H4_RELU_PAIR_MEHLER_NORMALIZATION_AUDIT` — exact with Khatri--Rao and
  collision-sewing qualifications.
- `H4_HADAMARD_K31_DIAGONAL_QUOTIENT_CAVEAT` — raw flat spectrum solved;
  strict `k>=991` requires a diagonal-rigidity/endpoint-tail receipt.
- `H4_CONSTANT_MODULUS_TRACE_QUOTIENT_AUDIT` — rank-one collapse exact only
  as a collision-completed strict quotient; literal flattening needs mask.
- `H4_THREE_NODE_JACOBI_K31_AUDIT` — exact H3+Q reduction; general Jacobi
  moves rather than eliminates the cubic residual and leaves H4 verdict
  unchanged.

## 44. Sharp approximate-flat tail statement and the missing diagonal-rigidity
## observable

For the natural full Hadamard K31 completion (43.3), suppose

\[
 0<\theta_{min}\le|\theta_s|\le\theta_{max}.
\tag{44.1}
\]

Then every rank-`k` full-matrix approximation obeys the exact spectral lower
bound

\[
 \boxed{
 {\|K_{31}-K_{31}^{(k)}\|_F^2\over\|K_{31}\|_F^2}
 \ge {m-k\over m}
       \left({\theta_{min}\over\theta_{max}}\right)^2.}
\tag{44.2}
\]

For relative flatness `|theta_s| in [theta_0(1-eps),theta_0(1+eps)]`, the
factor is `((1-eps)/(1+eps))^2`.  This is the lawful robust version of
`1-k/m` when the **natural full diagonal is binding**.

For the strict collision quotient, define instead

\[
 \boxed{
 \delta_k^{diag}(K)
 =\inf_{\operatorname{rank}X\le k,\ d\in\mathbb R^n}
   \|K-X-\operatorname{Diag}(d)\|_F.}
\tag{44.3}
\]

Equivalently, this is the best rank-`k` error on the fixed off-diagonal
entries.  The counterexample (43.4) has `delta_0^diag=0` despite a flat full
spectrum.  Therefore no lower bound on (44.3) follows from the amplitude
flatness in (44.1) alone.

If `K=K_0+Delta` and a structured witness supplies a certified
`delta_k^diag(K_0)`, the robust perturbation gate is simply

\[
 \boxed{
 \delta_k^{diag}(K)
 \ge\delta_k^{diag}(K_0)-\|P_{off}\Delta\|_F.}
\tag{44.4}
\]

After complete-old projection and endpoint whitening, even (44.3) is only a
source-side proxy.  With the actual linear endpoint map `mathcal W` the
binding observable is

\[
 \delta_{k,end}^{diag}
 =\inf_{\operatorname{rank}X\le k,d}
 \|\mathcal W P_{old}^{\perp}
   (K-X-\operatorname{Diag}d)\|,
\tag{44.5}
\]

or, for one frozen endpoint covector, the exact signed discarded action.
The coherent sum must meet `3.80e-6`.  This formulation cleanly separates:

1. natural CP-factor flatness, controlled by (44.2);
2. collision-quotient diagonal rigidity, controlled by (44.3);
3. old/endpoint darkness, controlled by (44.5).

### `H4_HADAMARD_APPROXFLAT_ENDPOINT_GATE`

- **Status:** natural full-plane tail bound solved; strict/endpoint quantity
  identified.
- **Do not promote `k>=991`:** until either the physical diagonal is binding
  or `delta_k^diag`/`delta_k,end^diag` is certified for the actual residual.

## 45. Literal strict trace flattening is generically full anchor rank

The mask caveat in (43.5) has an exact rank consequence.  For a trace source
with dense off-diagonal `R`, define the literal strict matrix

\[
 F_{i;(jk)}=1_{i\notin\{j,k\}}R_{jk},
 \qquad j\ne k.
\tag{45.1}
\]

Assume `R_jk` is nonzero for every distinct pair and `n>=3`.  If a linear
combination of rows vanishes, then for every pair `j,k`,

\[
 0=\sum_i a_iF_{i;(jk)}
 =R_{jk}\left(S-a_j-a_k\right),
 \qquad S=\sum_i a_i.
\tag{45.2}
\]

Hence `a_j+a_k=S` for every pair.  Comparing pairs shows all `a_i=t`; then
`S=nt=2t`, so `t=0` for `n>=3`.  Therefore

\[
 \boxed{\operatorname{rank}F=n.}
\tag{45.3}
\]

The same source has the rank-one completed representative `1 r^T` before
the anchor-collision mask.  Thus:

- **literal zero-filled Schmidt rank:** generically `n`;
- **collision-quotient completion rank:** exactly one;
- **target-conditioned trace action:** one pair contraction after summing
  only allowed anchors.

All three statements are simultaneously true.  Deployment may use the last
two only if the excluded K31/K4 collision action is synchronized exactly.
This theorem should replace any unqualified statement that constant modulus
makes the literal strict-q211 flattening rank one.

### `H4_TRACE_LITERAL_VS_QUOTIENT_RANK`

- **Status:** exact rank dichotomy solved.
- **Use:** rank one for consume-and-retire quotient action; rank `n` for a
  literal persistently zero-masked carrier absent collision completion.

## 46. The r63/L5 first-order seam: exact defect-angle criterion and a sharp conditional existence theorem

This section answers the remaining mathematical question behind the r63/L5
shell.  The inequality

\[
 {r(r+3)\over2}\ge 2n
\]

only says that the universal Veronese-null cokernel has disappeared.  It does
**not** by itself prove a first-order section.  At `n=1024,r=63`, put

\[
 m=n-r=961,\qquad E=\operatorname{Sym}^2\mathbb R^{63},\quad e=2016,
\]

\[
 \mathcal Q_U(G)=(u_i^TGu_i)_{i=1}^n,\qquad
 E_0=\ker\mathcal Q_U.
\tag{46.1}
\]

When `Q_U` is onto, `q:=dim E0=e-n=992=m+31`.  Assume also that the affine
repeated-cubic differential

\[
 (F_UH)_{ij}=2u_i^TH_i u_j+u_i^TH_j u_i
\tag{46.2}
\]

is onto.  This is a separate premise, certified by the full Veronese minor
and all 1,023-row projected-Veronese minors; `rank Q_U=n` alone is not enough.

Let `L=col U`, `K=L^perp`, and

\[
 \mathcal A=\operatorname{range}F_U^*,\quad
 \mathcal B=\mathcal A\cap(K\otimes E),\quad
 \mathcal C=\mathcal B^\perp\subset K\otimes E_0.
\tag{46.3}
\]

For a quadratic prefix `N`, write `N0` for its `K tensor E0` block.  The
first normal Schur adjoint has the right-inverse-free representation

\[
 \boxed{
 \mathscr S_{1,U,N}^*Z=[ZN],\qquad
 \|\mathscr S_{1,U,N}^*Z\|_{quot}
   =\|P_{\mathcal C}(ZN_0)\|_F,
 \quad Z\in\operatorname{Sym}K.}
\tag{46.4}
\]

Equivalently, with

\[
 \mathcal D_U:=\mathcal C^\perp\cap(K\otimes E_0),
\tag{46.5}
\]

the exact first-order seam criterion is

\[
 \boxed{
 \{Z\in\operatorname{Sym}K:ZN_0\in\mathcal D_U\}=\{0\}.}
\tag{46.6}
\]

In the original neuron coordinates this is the finite polynomial condition
that the simultaneous equations

\[
 \sum_j Z_{pj}N_j
 =\sum_i\Lambda_{ip}u_iu_i^T
  +2\operatorname{sym}\!\left[
      u_p\left(\sum_j\Lambda_{pj}u_j\right)^T\right],
 \quad Z=Z^T,\quad ZU=0,
\tag{46.7}
\]

have no solution with `Z != 0`.  Thus (46.7), or positive definiteness of

\[
 \boxed{
 G_{63}(U,N):
 Z\longmapsto
 \langle P_{\mathcal C}(ZN_0),P_{\mathcal C}(\,\cdot\,N_0)\rangle,}
\tag{46.8}
\]

is the atomic nonzero-Schur-minor receipt.  It contains no chosen right
inverse and no generic-context assumption.

### 46.1 What the extra defect space is

There is a useful exact description of `D_U`.  For a bank in `L tensor E`,
write `H_i=sum_a u_{ia}G_a` and set

\[
 p_G(x)=\sum_a x_a\,x^TG_ax.
\]

Then the right-`L` coefficient of `F_UH` is `grad p_G(u_i)`.  Hence

\[
 F_U(L\otimes E)
 =\{(\nabla p(u_i))_{i=1}^n:p\in\operatorname{Sym}^3L\}\,U^T.
\tag{46.9}
\]

If the cubic-gradient evaluation is injective, its rank is

\[
 h=\dim\operatorname{Sym}^3\mathbb R^{63}
 ={63\cdot64\cdot65\over6}=43,680.
\]

For `H in K tensor E0`, the other K21 term vanishes and

\[
 F_UH=2(H_i u_i)_iU^T.
\]

Since `F_U` is onto, the induced map from `K tensor E0` onto the quotient of
`P tensor L` by (46.9) is onto.  Therefore

\[
 \boxed{
 \dim\mathcal D_U=nr-h=64,512-43,680=20,832,}
\tag{46.10}
\]

and

\[
 \dim\mathcal C=mq-20,832
 =953,312-20,832=932,480.
\tag{46.11}
\]

The defect also has a direct cubic-syzygy parameterization.  Let
`Gamma_U p=(grad p(u_i))_i` and, for `Y=(y_i) in ker Gamma_U^*`, form the
matrix bank

\[
 \mathcal R_U(Y)_i=2\operatorname{sym}(u_i y_i^T).
\]

Then

\[
 \boxed{
 \mathcal D_U=
 \{P_K\mathcal R_U(Y)P_{E_0}:Y\in\ker\Gamma_U^*\}.}
\tag{46.11a}
\]

Indeed, pairing with `H in K tensor E0` gives
`2 sum_i y_i^T H_i u_i`, exactly the adjoint of the right-`L` quotient map.
Under affine surjectivity this map from `ker Gamma_U^*` to `D_U` is
injective.  Thus (46.11a) is both a constructive defect basis and the bridge
to the symmetry calculation below; no `n^2` K21 right inverse is required.

This 20,832-dimensional gradient-compatibility space is precisely what was
missing from the earlier argument `N0 full row rank implies a seam`.  Full row
rank only excludes `ZN0=0`; it does not exclude a nonzero `ZN0 in D_U`.

### 46.2 Sharp conditional algebraic existence from rank-176 avoidance

Define the minimum row rank of the compatibility defect after complexifying,

\[
 \delta(U)=\min\{\operatorname{rank}T:
                  0\ne T\in\mathcal D_U\otimes\mathbb C\}.
\tag{46.12}
\]

There is a useful finite sufficient condition:

\[
 \boxed{\delta(U)\ge176.}
\tag{46.13}
\]

Under (46.13), a Zariski-open nonempty set of `961 by 992` prefix blocks
`N0` satisfies (46.6).  The proof is an incidence count, not a random-matrix
heuristic.  If `rank Z=s`, the projective symmetric rank-`s` locus has
dimension

\[
 d_s-1={s(2m-s+1)\over2}-1.
\]

For fixed `Z`, the map `N0 -> ZN0` has rank `sq`; requiring its value to lie
in the 20,832-dimensional `D_U` has codimension at least `sq-20,832`.
Because a full-row-rank `N0` makes `ZN0=0` impossible for `Z != 0`, (46.13)
restricts the incidence to `s>=176`.  At the first such rank,

\[
 sq-d_s+1
 ={s(s+61)\over2}+1
 ={176\cdot237\over2}+1
 =20,857>20,832,
\tag{46.14}
\]

and the margin increases with `s`.  The bad incidence therefore has smaller
dimension than the space of `N0`, proving the claim.  Rank 176 is sharp for
this dimension argument: at `s=175`, the right side is only 20,651.

Condition (46.13) is itself a finite algebraic receipt: the projective
20,831-plane `P(D_U)` must avoid the determinantal variety `rank<=175`.
Equivalently, its determinantal resultant is nonzero.  It is not implied by
parameter counting and is not yet proved for the physical L5 cut.  The exact
actual-cut alternative is simply the positive Gram gate (46.8).

There is a particularly concrete **constructive subfamily**.  In fixed
orthonormal bases of `K` and `E0`, choose 961 null-feature columns as an
identity block and set

\[
 \boxed{N_0=[I_{961}\mid X],\qquad X\in\mathbb R^{961\times31}.}
\tag{46.14a}
\]

Splitting `T in K tensor E0` into the same `961|31` columns, (46.6) becomes

\[
 \boxed{
 \mathcal D_U\cap
 \{[Z\mid ZX]:Z=Z^T\}=\{0\}.}
\tag{46.14b}
\]

Thus a finite structured witness is exactly one matrix `X` for which
(46.14b) holds; no nonlinear solve or right inverse appears.  The number of
free entries is

\[
 961\cdot31=29,791,
\]

which is exactly the previously derived r63 complete-pair quotient surplus.
Also `N0N0^T=I+XX^T`, so the row singular floor is at least one in this chart
before the overall physical scale is absorbed into `t`.  This chart does not
prove such an `X` exists for an arbitrary structured `D_U`; (46.14b), the
incidence conditions below, or the defect determinant remain the receipt.
Nor is it a claimed normal form for every physical prefix: reducing a general
invertible column block by a nonorthogonal left change also changes the metric
notion of `Z=Z^T`.

The rank-176 gate can be sharpened without changing the exact actual-prefix
test.  For `1<=s<=m`, define the worst compression-section dimension

\[
 \mu_s(U)=
 \max_{R\in\operatorname{Gr}(s,K)}
 \dim\bigl(\mathcal D_U\cap(R\otimes E_0)\bigr).
\tag{46.15}
\]

Repeating the incidence proof while retaining the dimension of the defect
inside each possible range of `Z` gives the sharper sufficient conditions

\[
 \boxed{
 \mu_s(U)<sq-{s(2m-s+1)\over2}+1
 ={s(s+61)\over2}+1
 \quad\text{for every }s=1,\ldots,961.}
\tag{46.16}
\]

If (46.16) holds, generic full-row `N0` satisfies (46.6).  Condition
`delta(U)>=176` implies (46.16) by using `mu_s=0` below 176 and
`mu_s<=20,832` thereafter, but (46.16) can still pass when `D_U` contains a
few low-rank matrices.  This is the sharpest dimension-only existence gate
available here; failure of one inequality is not by itself a no-go because
an incidence of the ambient dimension need not dominate the prefix space.

There is, however, an exact structural universal obstruction.  If a nonzero
subspace `R subset K` obeys

\[
 \boxed{R\otimes E_0\subseteq\mathcal D_U,}
\tag{46.17}
\]

then **every** prefix is singular: for any `N0`, every symmetric `Z` with
range in `R` has `ZN0 in R tensor E0 subset D_U`.  Thus `Sym R` is a dark
normal block independent of the prefix.  Here the arithmetic is exact:

\[
 \dim\mathcal D_U=20,832=21\cdot992.
\]

So the defect can be an entire 21-row compression, forcing at most the
`dim Sym^2(R^21)=231` modes of that full-compression form for every prefix.
This is smaller than r62's 561 forced modes, but raw dimension does not
exclude it.  Conversely, ruling out compression
sections by (46.16), or directly passing (46.8), is what turns the r63
dimension threshold into an existence statement.

There is also a physically relevant reason the convenient rank-176
corollary may fail even when the sharper criterion survives.  Let

\[
 M_3(U)=\sum_i u_i^{\otimes3}
\]

and let `Omega` be a skew latent generator.  Put

\[
 Y_{\Omega,i}=\Omega u_i.
\]

For the cubic-gradient evaluation `Gamma_U p=(grad p(u_i))_i`, direct
polarization gives

\[
 \boxed{
 \Gamma_U^*Y_\Omega
 =\Omega\cdot M_3(U)
 =3\sum_i\operatorname{sym}
       (u_i\otimes u_i\otimes\Omega u_i),}
\tag{46.18}
\]

up to the common tensor-inner-product convention.  Hence every infinitesimal
stabilizer of `M3` supplies a compatibility-defect covector.  In particular,
if `M3(U)=0`, every skew `Omega` does.  Under affine surjectivity the induced
element of `D_U` is nonzero, while for `rank Omega=s` its row span is contained
in

\[
 \operatorname{sym}(L\otimes\operatorname{im}\Omega),
 \qquad
 \dim=s r-{s(s-1)\over2}.
\tag{46.19}
\]

A rank-two rotation therefore gives a nonzero defect of row rank at most

\[
 2\cdot63-1=125.
\tag{46.20}
\]

There is one compatibility correction.  Under the standing
`rank Q_U=n` premise, `M3(U)=0` is itself impossible: contracting `M3=0`
with an arbitrary `v` gives

\[
 \sum_i(u_i^Tv)u_iu_i^T=0,
\]

and independence of the 1,024 outer products forces `u_i^Tv=0` for every
`i,v`.  Thus the relevant exact case is a **nonzero infinitesimal stabilizer**
of `M3`, not a fully zero third moment.  Such a rotationally symmetric
anchor cannot pass the coarse `delta>=176` theorem.  This is **not** a
universal seam no-go: a
125-rank defect ray need not contain a row-compression space and can still
obey the stratified inequalities (46.16) or miss the actual orbit
`{ZN0}`.  It does prove that treating `D_U` as an unconstrained generic
20,832-plane is unsafe for isotropic physical frames; use (46.8) or (46.16).

With orthonormal `svec` conventions, the `E by L` flattening of the third
moment is exactly

\[
 \operatorname{Flat}_{2|1}M_3=\mathcal Q_U^TU.
\]

Consequently

\[
 \|M_3\|_F\ge\sigma_{\min}(\mathcal Q_U)\|U\|_F.
\]

The finite symmetry receipt is therefore not the raw size of `M3` but the
rotation gap

\[
 \gamma_{rot}(U)=
 \inf_{\Omega^T=-\Omega,\ \|\Omega\|_F=1}
 \|\Omega\cdot M_3(U)\|_F.
\]

Positive `gamma_rot` removes this particular low-rank defect family; it does
not certify the full determinantal or principal-angle gate.

### 46.3 A finite unisolvent skeleton, and why it is not a seam witness

There is an explicit `U` that proves the quadratic and cubic-unisolvence
requirements are jointly feasible.  Partition the 63 latent coordinates
into sets `A,B` of sizes 31 and 32.  Take the 1,024 rows

\[
 \{e_a:a\in A\cup B\}
 \ \cup\
 \{e_a+e_b:a<b,\ a,b\in A\text{ or }a,b\in B\}.
\tag{46.21}
\]

There are `63+C(31,2)+C(32,2)=1,024` rows.  Their quadratic outer products
are independent: within-part off-diagonal entries first kill every edge
coefficient, then diagonal entries kill the vertex coefficients.  Thus
`rank Q_U=1,024` and `E0` is exactly the 992-dimensional cross-block matrix
space `A tensor B`.

The cubic-gradient frame is also injective.  Vanishing at every `e_a` kills
all cubic monomials with a repeated coordinate.  For every all-distinct
triple, two indices lie in the same one of the two parts; differentiating at
their row `e_a+e_b` in the third direction kills that square-free
coefficient.  Hence all 43,680 cubic coefficients vanish.

But (46.21) is deliberately also a warning.  For its cross-block `E0`, every
`H_i u_i` points into the opposite part, and every such pattern is already a
gradient of a mixed `AAB` or `ABB` cubic.  Consequently the induced quotient
map in (46.10) is zero and `F_U` is not onto.  So even simultaneous full
quadratic/cubic unisolvence and a perfectly conditioned choice
`N0=[I_961\ 0]` do **not** prove the seam.  A lawful witness must pass all
three independent gates: the projected-Veronese affine minor, the defect
transversality (46.6), and its singular-value bound.

### 46.4 Exact conditioning factorization

The Gram in (46.8) has a smaller exact normal form.  Define

\[
 \mathcal A_N:\operatorname{Sym}K\to K\otimes E_0,
 \qquad \mathcal A_NZ=ZN_0,
\]

and put `G0=N0 N0^T`.  Its adjoint and unprojected Gram are

\[
 \mathcal A_N^*T=\operatorname{sym}(TN_0^T),
 \qquad
 \mathcal L_{G_0}Z:=\mathcal A_N^*\mathcal A_NZ
 ={ZG_0+G_0Z\over2}.
\tag{46.22}
\]

For full-row `N0`, `L_G0` is positive definite.  Let

\[
 \mathcal T_{U,N}
 =P_{\mathcal D_U}\mathcal A_N\mathcal L_{G_0}^{-1/2}:
 \operatorname{Sym}K\to\mathcal D_U.
\tag{46.23}
\]

Then

\[
 \boxed{
 G_{63}
 =\mathcal L_{G_0}^{1/2}
  (I-\mathcal T_{U,N}^*\mathcal T_{U,N})
  \mathcal L_{G_0}^{1/2}.}
\tag{46.24}
\]

The operator `T` is a contraction.  Hence the exact defect-sized minor is

\[
 \boxed{
 \det G_{63}\ne0
 \iff
 \det_{\mathcal D_U}
 \left[I-
 P_{\mathcal D_U}\mathcal A_N\mathcal L_{G_0}^{-1}
 \mathcal A_N^*P_{\mathcal D_U}\right]>0.}
\tag{46.25}
\]

This is Sylvester's determinant identity; the determinant on the right is
only 20,832 dimensional rather than 462,241 dimensional.  Equivalently,

\[
 \boxed{
 \text{first seam}\iff \|\mathcal T_{U,N}\|_{op}<1.}
\tag{46.26}
\]

If `N0N0^T=I` (possible because `q=m+31`), `A_N` is an isometry and the
Schur singular values are exactly the sines of the principal angles between
the 462,241-dimensional orbit `A_N(Sym K)` and `D_U`.  In particular, the
compatibility subtraction has rank at most 20,832.  Thus for every full-row
prefix

\[
 \boxed{
 \dim\ker\mathscr S_1^*\le20,832.}
\tag{46.27}
\]

Moreover, at most 20,832 squared Schur singular values can fall below
`lambda_min(G0)`; at least `462,241-20,832=441,409` modes retain that floor.
The r63 threshold therefore does not guarantee zero first-order-dark modes,
but it confines every additional compatibility loss to a 20,832-dimensional
block.

For a full-row-rank `N0`, define the defect angle

\[
 \alpha(U,N)=
 \inf_{0\ne Z\in\operatorname{Sym}K}
 {\operatorname{dist}(ZN_0,\mathcal D_U)\over\|ZN_0\|_F}.
\tag{46.28}
\]

Then the first Schur singular value obeys the sharp product lower bound

\[
 \boxed{
 \sigma_{1,63}:=\sigma_{\min}(\mathscr S_{1,U,N}^*)
 \ge \sigma_{\min}(N_0)\,\alpha(U,N).}
\tag{46.29}
\]

Thus row conditioning of the prefix and transversality to the cubic-gradient
defect are separate.  Neither can compensate for a zero in the other.  For
the reduced odd expansion

\[
 \mathscr J_{63}(t)=t\mathscr S_1+t^3\mathscr S_3+O(t^5),
\]

assume the higher part is bounded by `|t|^3 M_3`.  Then

\[
 \sigma_{\min}(\mathscr J_{63}(t))
 \ge |t|\sigma_{1,63}-|t|^3M_3,
\tag{46.30}
\]

so `t^2 M3 <= sigma1,63/2` gives a right-inverse norm at most

\[
 {2\over |t|\sigma_{1,63}}.
\tag{46.31}
\]

The normal determinant opens at order `t^462241`; there is no forced excess
power.  By contrast r62 has `m=962`, a forced `R` of dimension 33 and
`Sym R` of dimension 561.  Even after a successful cubic minor its dark
inverse is

\[
 {1\over |t|^3\sigma_{3,62}}
\]

and its normal determinant has the excess `2*561=1,122` powers derived in
the r62 lane.  Therefore the asymptotic stability ratio is

\[
 \boxed{
 {\kappa_{62,dark}\over\kappa_{63}}
 \asymp {\sigma_{1,63}\over\sigma_{3,62}}|t|^{-2}.}
\tag{46.32}
\]

The five-transition terminal-factored costs are 47.43295 at r62 and
48.07040 at r63.  The r63 removal of the cubic seam costs only 0.63745
product-equivalents, about 0.182 percent adjusted score at utilization
0.5629, but leaves only 0.92960 absolute product-equivalents of shell
headroom instead of 1.56705.

### 46.5 The 561 modes matter to value only if the L5 suffix sees them

Let `V5` be the complete-old-projected map from the local covariance-normal
plane to the retained L5 endpoint action, and whiten its nonzero rows:

\[
 \widehat V_5=(V_5V_5^*)^{-1/2}V_5.
\]

The exact endpoint-weighted r63 first-order gate is

\[
 \boxed{
 \tau_{63,V}
 =\sigma_{\min}(\mathscr S_{1,63}^*\widehat V_5^*)>0.}
\tag{46.33}
\]

Full (46.8) implies (46.33), but (46.33) can pass even when irrelevant local
normal directions remain dark.  At r62 define

\[
 W_{62,dark}
 =\ker(\mathscr S_{1,62}^*\widehat V_5^*).
\tag{46.34}
\]

Only `q_V=dim W62,dark`, not automatically all 561 local cokernel modes,
must be recovered cubically for Bellman value.  If `q_V=0`, the physical L5
value seam at r62 is already first order and r63 has no `t^-2` value advantage;
its benefit is only full-interface regularity.  If `q_V>0`, the exact cubic
value gate is injectivity of

\[
 \mathscr S_{3,62}^*\widehat V_5^*
 \quad\text{on }W_{62,dark},
\tag{46.35}
\]

with smallest singular value `tau_3,62,V`.  For a trusted parameter radius
`Rtheta`, the leading guaranteed endpoint radii are then

\[
 R_{V,63}\ge R_\theta
 (|t|\tau_{63,V}-|t|^3M_{63,V}),
\]

\[
 R_{V,62,dark}\ge R_\theta
 (|t|^3\tau_{3,62,V}-|t|^5M_{62,V}).
\tag{46.36}
\]

For one frozen scalar endpoint action, `q_V` is only zero or one.  For a
rank-`k` panel/corpus endpoint it is at most `min(k,561)`.  Therefore the
full 561-mode determinant penalty is the correct complete-interface receipt,
but it is not automatically the score-relevant penalty.  A first-order state
section may still be required for nonlinear continuation; that is a separate
claim from a linearized scalar Bellman action.

This is the lawful value comparison.  Charging r62 all 561 cubic modes
without checking (46.34) overstates the r63 advantage; ignoring a nonzero
`W62,dark` understates it by two powers of the physical prefix scale.

### Atomic graph gates

- **`R63_L5_AFFINE_K21_MINOR`:** all projected-Veronese minors nonzero;
  `rank Q_U=n` alone is insufficient.
- **`R63_L5_FIRST_SCHUR_TRANSVERSALITY`:** (46.6), equivalently
  `G63(U,N)>0`; exact full-interface existence gate.
- **`R63_L5_DEFECT_RANK176_SUFFICIENT`:** cubic-gradient evaluation full,
  `delta(U)>=176`; then a nonempty Zariski-open family of prefix blocks opens
  the first seam.
- **`R63_L5_FIRST_SCHUR_CONDITIONING`:** certify both
  `sigma_min(N0)` and the defect angle `alpha`, then (46.29)--(46.30).
- **`R62_R63_L5_ENDPOINT_VALUE_SPLIT`:** compute (46.33)--(46.35); the r62
  cubic penalty applies only to the suffix-visible dark dimension `q_V`.

**Disposition:** r63/L5 is no longer blocked by a universal dimension theorem,
and (46.13)--(46.14) give a rigorous conditional existence route.  It is not
yet promoted: no actual-cut projected-Veronese minor, defect resultant/Gram,
or endpoint-weighted singular value has been supplied.  The next pure-math
question is whether the physical L5 frame has a structural reason forcing
`D_U` to contain rank-at-most-175 matrices; absent such a theorem, the exact
finite gate is (46.8), not a genericity slogan.

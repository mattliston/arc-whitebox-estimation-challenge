# An exact full-rank 561 construction for the anchored r62 tilted-wedge map

## Verdict

There is no universal rank loss in the locally anchored tilted-wedge map at

\[
 n=1024,\qquad r=62,\qquad \dim E_0=929,\qquad \dim R=33.
\]

An explicit algebraic schema below gives a regular pair `(U,N)` for which

\[
 \overline\Phi_N:\operatorname{Sym}R
 \longrightarrow
 \frac{\bigwedge^2E_0}{\operatorname{range}\mathcal T_{\rm anc}^*}
\]

has the maximum possible rank

\[
 \boxed{\operatorname{rank}\overline\Phi_N
        ={33\cdot34\over2}=561.}
\]

The mechanism is deliberately different from the failed disjoint affine
triples.  It uses 34 *overlapping* three-row groups to encode the 33 residual
coordinates.  The complete detector space is an automatically anchor-silent
copy of `so(34)`.  The 561 unordered pairs of the 34 groups simultaneously
form

1. a basis of `Sym(33)^*`, and
2. a basis of `wedge^2 R^34`.

Thus every diagonal residual mode is recovered through a cross-group
commutator; no residual basis vector is confined to one three-row group.

## 1. Latent split and the automatic anchor-silent detector

Split the latent space as

\[
 W=V\oplus C\oplus Z,
 \qquad \dim V=34,\quad\dim C=1,\quad\dim Z=27.
\tag{1.1}
\]

Fix a unit vector `c` spanning `C`.  Identify `V` isometrically with the
cross-block matrix space

\[
 X=V\odot C\subset\operatorname{Sym}W,
 \qquad
 S(x)={1\over\sqrt2}
 \begin{pmatrix}
 0&x&0\\
 x^T&0&0\\
 0&0&0
 \end{pmatrix},
 \qquad
 \langle S(x),S(y)\rangle_F=x^Ty.
\tag{1.2}
\]

All anchor rays used below lie in

\[
 H_0=V\oplus Z=\{c=0\}
 \quad\hbox{or}\quad
 H_1=C\oplus Z=\{v=0\}.
\tag{1.3}
\]

Consequently every such ray is isotropic for every matrix in `X`.  Once the
Veronese frame is completed, this gives

\[
 X\subset E_0:=\ker Q_U.
\tag{1.4}
\]

Extend any `A in so(X)` by zero on `X^perp cap E0`.  If at an active row

\[
 u=(\alpha v,0,z),\qquad n_X=S(\beta v),
\tag{1.5}
\]

then `(nA)^sharp=S(h)` with `h` proportional to `A v`, and hence

\[
 S(h)u=0
 \quad\Longleftrightarrow\quad
 h^Tv\ \propto\ v^TAv=0.
\tag{1.6}
\]

At inactive rows take `n_X=0`.  Therefore the entire 561-dimensional space

\[
 \boxed{\mathfrak{so}(X)\subset\ker\mathcal T_{\rm anc}.}
\tag{1.7}
\]

This turns the `wedge^2 X` projection of `Phi_N` into an exact lower bound on
the anchored quotient rank.

## 2. The 34 overlapping residual labels

Let the abstract residual coordinate space be `Y=R^33`, with basis
`e_1,...,e_33`.  Define 34 row labels

\[
 p_0=\sum_{k=1}^{33}e_k,
 \qquad
 p_a=e_a\quad(1\le a\le33).
\tag{2.1}
\]

The 561 functionals

\[
 \ell_{ab}(\Theta)=p_a^T\Theta p_b,
 \qquad 0\le a<b\le33,
 \qquad \Theta\in\operatorname{Sym}Y,
\tag{2.2}
\]

form a basis of `Sym(Y)^*`.  Indeed, the pairs `1<=a<b<=33` give all 528
off-diagonal entries.  Once those vanish, the 33 equations

\[
 p_0^T\Theta p_a
 =\Theta_{aa}+\sum_{k\ne a}\Theta_{ka}=0
\tag{2.3}
\]

give all diagonal entries.

Choose a basis `f_0,...,f_33` of `V`, put

\[
 \mathbf1=\sum_{a=0}^{33}f_a,
 \qquad
 v_a=f_a+\mathbf1.
\tag{2.4}
\]

The `v_a` are a basis because their column matrix is `I+11^T`, whose
eigenvalues are 35 and 1.  Moreover

\[
 v_a^Tv_b=36\ne0\qquad(a\ne b).
\tag{2.5}
\]

Hence the 561 bivectors `v_a wedge v_b`, `a<b`, form a basis of
`wedge^2 V` and none of the Gram multipliers in the construction below
vanishes.

## 3. Three active rows per group

Choose nonzero vectors `w_0,...,w_33 in Z` whose rank-one squares are
independent.  One fully explicit choice, for a basis `g_1,...,g_27` of `Z`,
is

\[
 w_a=g_{a+1}\quad(0\le a\le26),
 \qquad
 w_{26+j}=g_1+g_{j+1}\quad(1\le j\le7).
\tag{3.1}
\]

The first 27 squares give the diagonal coordinates and each of the last
seven has a unique `g_1 odot g_{j+1}` coordinate.

For each group `a=0,...,33` take three physical rows `t=0,1,2`.  Set

\[
 \alpha=(1,-2,1),
 \qquad
 z_{a,0}=0,\quad z_{a,1}=w_a,\quad z_{a,2}=-w_a,
\tag{3.2}
\]

and

\[
 u_{a,t}=\alpha_t v_a+z_{a,t}\in H_0.
\tag{3.3}
\]

For the `X` component of the coefficient row, put

\[
 \begin{aligned}
 \beta_{a,0}&=a,\\
 \beta_{a,1}&=-2(a+1),\\
 \beta_{a,2}&=a+2,
 \end{aligned}
 \qquad
 (n_{a,t})_X=S(\beta_{a,t}v_a).
\tag{3.4}
\]

Both row sums vanish:

\[
 \sum_tu_{a,t}=0,
 \qquad
 \sum_t\beta_{a,t}=0.
\tag{3.5}
\]

Define the physical residual embedding `P:R^33 -> R^1024` by putting the
same row `p_a^T` on all three rows in group `a`, and zero on the remaining
922 rows.  Equation (3.5) immediately gives

\[
 U^TP=0,
 \qquad
 N_X^TP=0.
\tag{3.6}
\]

The overlap is essential.  A coordinate residual `e_k` has support in group
0 and group `k`; it is never the isolated equal-weight vector of one group.

### 3.1 The 102 active Veronese rows are independent

Suppose

\[
 \sum_{a,t}\lambda_{a,t}u_{a,t}u_{a,t}^T=0.
\tag{3.7}
\]

The `V-V` block, using independence of the 34 matrices `v_av_a^T`, gives

\[
 \lambda_{a,0}+4\lambda_{a,1}+\lambda_{a,2}=0.
\tag{3.8}
\]

The `V-Z` block, using that the `v_a` are a basis, gives

\[
 -2\lambda_{a,1}-\lambda_{a,2}=0.
\tag{3.9}
\]

Thus

\[
 (\lambda_{a,0},\lambda_{a,1},\lambda_{a,2})
 =\kappa_a(-2,1,-2).
\tag{3.10}
\]

The `Z-Z` block is then

\[
 -\sum_a\kappa_aw_aw_a^T=0.
\tag{3.11}
\]

Independence in (3.1) forces every `kappa_a=0`.  Thus the 102 displayed
rank-one matrices are independent.

## 4. Exact calculation of the tilted-wedge minor

For `Theta in Sym(Y)`, set

\[
 Z_{\rm phys}=P\Theta P^T,
 \qquad
 z_{ab}=p_a^T\Theta p_b.
\tag{4.1}
\]

For a row `i=(a,t)`, its residual moment is

\[
 M_i=Q_U^*(Z_{\rm phys}e_i).
\tag{4.2}
\]

Only the `V-V` block is relevant to the projection onto `X`.  Since
`sum_t alpha_t^2=6`, it is

\[
 (M_i)_{VV}
 =6\sum_{b=0}^{33}z_{ab}v_bv_b^T,
\tag{4.3}
\]

independently of `t`.  The normalized cross-block identity is

\[
 P_X\{S(x),M_i\}=S((M_i)_{VV}x).
\tag{4.4}
\]

Define

\[
 D_a=\sum_{t=0}^2\beta_{a,t}^2
 =a^2+4(a+1)^2+(a+2)^2
 =6a^2+12a+8.
\tag{4.5}
\]

These numbers are strictly increasing for `a>=0`.  Substituting
(3.4)--(4.4) into the definition of `Phi_N` and combining the two ordered
orientations of each unordered pair gives the exact `wedge^2 X` projection

\[
 \boxed{
 P_{\wedge^2X}\Phi_N(P\Theta P^T)
 =6\sum_{0\le a<b\le33}
 (D_a-D_b)(v_a^Tv_b)z_{ab}\,
 S(v_a)\wedge S(v_b).}
\tag{4.6}
\]

Here `x wedge y=skew(x tensor y)` uses the same skew convention as `Phi_N`.
Every scalar multiplier in (4.6) is nonzero by (2.5) and (4.5).  Equations
(2.2) and (2.4) therefore prove that (4.6) is an isomorphism

\[
 \operatorname{Sym}(\mathbb R^{33})
 \overset{\sim}{\longrightarrow}
 \bigwedge^2X.
\tag{4.7}
\]

This is the promised triangular detector: the 528 pairs among groups 1--33
read the off-diagonal entries, and the 33 pairs `(0,a)` then read the
diagonals.

There is also a literal closed determinant receipt.  In the unnormalized
coordinate bases used above, the measurement change
`Theta -> (z_ab)_{a<b}` is triangular with determinant one, the induced
exterior change `f_a wedge f_b -> v_a wedge v_b` has determinant
`det(I+11^T)^33=35^33`, and

\[
 6(D_a-D_b)(v_a^Tv_b)
 =1296(a-b)(a+b+2).
\tag{4.8}
\]

Consequently the displayed 561-square minor has determinant, up to the one
fixed sign/skew-normalization convention,

\[
 \boxed{
 35^{33}\prod_{0\le a<b\le33}
 1296(a-b)(a+b+2)\ne0.}
\tag{4.9}
\]

## 5. Exact regular completion to n=1024

It remains to complete both the Veronese frame and the coefficient bank
without changing (4.6).

### 5.1 Complete U while preserving X subset E0

The rank-one matrices of rays in `H0 union H1` span

\[
 \operatorname{Sym}H_0+\operatorname{Sym}H_1,
\]

of dimension

\[
 {61\cdot62\over2}+{28\cdot29\over2}
 -{27\cdot28\over2}
 =1891+406-378=1919.
\tag{5.1}
\]

Its orthogonal complement is exactly the 34-dimensional cross block `X`.
By Section 3.1, the 102 active rays are independent elements of this
rank-one spanning family.  The basis-extension theorem therefore lets us add
922 further rays from `H0 union H1` so that all 1,024 rank-one matrices are
independent.  During the extension choose 61 generic `H0` rays spanning
`H0`, and one nonzero `C` ray.  These choices can be made simultaneously:
failure of either rank is a proper algebraic condition, and the permitted
rank-one rays span (5.1).

Thus

\[
 \operatorname{rank}U=62,
 \qquad
 Q_U:\operatorname{Sym}W\twoheadrightarrow\mathbb R^{1024},
 \qquad
 \dim E_0=1953-1024=929,
\tag{5.2}
\]

and `X subset E0` as required.  The extra physical rows have zero residual
labels.

### 5.2 One safe active completion column

The 34 specified `N_X` columns and the 62 columns of `U` are independent.
Indeed, the extension rows were chosen so that their `U` restriction has
rank 62, whereas `N_X` vanishes there.

On the 102 active rows, the span of `[U N_X]` is exactly the 68-dimensional
space of vectors whose three entries sum to zero separately in each group.
To see the rank directly, put

\[
 \delta=(0,1,-1),
 \qquad
 \beta_a=a\alpha-2\delta.
\tag{5.3}
\]

The active evaluation of latent coefficients `(y,z,h) in V+Z+X` is the
pair of 34-vectors

\[
 \left(
   (v_a^Ty+a\,v_a^Th)_a,
   (w_a^Tz-2v_a^Th)_a
 \right).
\tag{5.4}
\]

This map is onto `R^34+R^34`: the first component is solved with `y`, and
the second with `h`.  Hence its rank is 68.

The orthogonal complement of the 33 residual columns on the active rows has
dimension `102-33=69`.  Its missing one-dimensional direction is the
group-constant vector

\[
 y_{a,t}=\eta_a,
 \qquad
 \eta_0=1,
 \qquad
 \eta_a=-1\quad(1\le a\le33),
\tag{5.5}
\]

because

\[
 \sum_{a=0}^{33}\eta_ap_a
 =p_0-\sum_{a=1}^{33}p_a=0.
\tag{5.6}
\]

Choose any nonzero

\[
 H_{\rm safe}\in E_0\cap\operatorname{Sym}H_0.
\tag{5.7}
\]

Such matrices form a space of dimension at least `1891-1024=867`.  Assign
the physical coefficient column (5.5) to `H_safe` and set it to zero on the
922 extension rows.

This column changes neither the detector kernel nor (4.6): `A in so(X)` is
zero on `H_safe`, while every residual moment `M_i` is supported on `H0`, so

\[
 P_X\{H_{\rm safe},M_i\}=0.
\tag{5.8}
\]

### 5.3 Complete N and make the residual exactly 33-dimensional

Let `Z_0` be the 922-dimensional physical space supported only on the
extension rows.  Put

\[
 K_1=\operatorname{span}(U,N_X,y),
 \qquad \dim K_1=62+34+1=97.
\tag{5.9}
\]

Its active restriction has rank 69 by (5.4)--(5.5), so

\[
 \dim(K_1\cap Z_0)=97-69=28.
\tag{5.10}
\]

Consequently

\[
 \dim(K_1+Z_0)=97+922-28=991.
\tag{5.11}
\]

This sum lies in `P^perp`, which also has dimension `1024-33=991`.
Therefore

\[
 K_1+Z_0=P^\perp.
\tag{5.12}
\]

Choose 894 vectors in `Z0` which, together with `K1`, form a basis of
`P^perp`, and assign them to the remaining 894 basis matrices of `E0`.
The resulting 929-column coefficient bank satisfies

\[
 \boxed{
 \operatorname{rank}[U\ N]=991,
 \qquad
 \ker U^T\cap\ker N^T=\operatorname{range}P,
 \qquad \dim R=33.}
\tag{5.13}
\]

The 894 final coefficient columns vanish on the active rows.  On the
extension rows the residual moment is zero.  Thus they also leave (4.6)
unchanged.

## 6. Survival through the exact anchor-adjoint quotient

Take nonzero `Theta in Sym(Y)`.  By (4.7),

\[
 B_\Theta:=P_{\wedge^2X}\Phi_N(P\Theta P^T)\ne0.
\tag{6.1}
\]

Regard `B_Theta` as the corresponding skew endomorphism of `X`, extended by
zero on the rest of `E0`.  Equation (1.7) gives

\[
 B_\Theta\in\ker\mathcal T_{\rm anc}.
\tag{6.2}
\]

Orthogonality of the block projection then gives

\[
 \left\langle
  \Phi_N(P\Theta P^T),B_\Theta
 \right\rangle
 =\|B_\Theta\|_F^2>0.
\tag{6.3}
\]

Since

\[
 \operatorname{range}\mathcal T_{\rm anc}^*
 =(\ker\mathcal T_{\rm anc})^\perp,
\tag{6.4}
\]

no nonzero domain element maps into the anchor adjoint.  Therefore the
anchored quotient map is injective and has rank 561.

## 7. Why the disjoint-triple no-go does not apply

Each three-row group separately has equal residual weights and zero first
moments.  If the physical residual space contained a vector supported on one
group, its diagonal wedge would again be vulnerable.  But the row-label
matrix `(p_a)` has one relation

\[
 p_0=\sum_{a=1}^{33}p_a,
\tag{7.1}
\]

and no residual coordinate is supported on a single group.  The 33 diagonal
coordinates of `Theta` are read by the 33 cross-group pairs `(0,a)` after the
off-diagonal coordinates are removed.  Algebraically, the missing diagonal
information is converted into the commutators

\[
 (D_0-D_a)(v_0^Tv_a)\,v_0\wedge v_a.
\tag{7.2}
\]

Thus Checkpoint 110 remains correct for disjoint affine second differences,
while the present overlapping construction proves that its rank-33 loss is
not universal.

## Graph-ready node

`R62_TILTED_WEDGE_FULL561_OVERLAPPING_CONSTRUCTION`

- status: exact symbolic full-rank theorem for the local anchor quotient;
- dimensions: `n=1024`, `r=62`, `dim E0=929`, `dim R=33`;
- detector: the complete `so(34)` on the latent `V34-C1` cross block, all of
  which lies in `ker T_anc`;
- residual design: 34 three-row groups labeled by
  `p0=sum e_a`, `p_a=e_a`, producing 33 overlapping residual directions;
- exact minor: diagonal in the 561 bases
  `{p_a odot p_b:a<b}` and `{v_a wedge v_b:a<b}`, with nonzero entries
  `6(D_a-D_b)(v_a^Tv_b)`;
- quotient result: every nonzero residual symmetric mode pairs with an exact
  locally anchor-silent skew detector, hence rank 561 after quotient;
- supersedes only the proposed universal extension of the disjoint-triple
  NOGO; the disjoint-triple theorem itself remains valid;
- remaining campaign gate: identify whether the actual deployed prefix can
  realize or approach this overlapping regular schema with an acceptable
  cubic singular value and arithmetic cost.

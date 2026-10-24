# Full-AQ r54 seam: exact differential, affine obstruction, and Schur-section gate

Date: 2026-08-30

Scope: pure local algebra only.  This note makes no value, adjusted-score, or
deployment-cost claim.

Let \(n=1024\), \(r=54\), and use the unscaled quadratic-chaos convention

\[
 X_i=u_i^{\mathsf T}z+z^{\mathsf T}M_i z-\operatorname{tr}M_i,
 \qquad M_i\in\operatorname{Sym}_r.
\]

The observation map is

\[
 \Phi_{54}:\{(u_i,M_i)\}_{i=1}^n/O(r)\longrightarrow(C,K21),
\]

where

\[
 \boxed{C_{ij}=u_i^{\mathsf T}u_j+2\operatorname{tr}(M_iM_j),}
\]

\[
 \boxed{
 K_{ij}=8\operatorname{tr}(M_i^2M_j)
 +2u_i^{\mathsf T}M_ju_i
 +4u_i^{\mathsf T}M_iu_j.}
\]

The second formula is \(\kappa_3(X_i,X_i,X_j)\).

## 1. Exact trace-cubic embedding

Associate to each full-AQ coefficient the symmetric \(55\times55\) matrix

\[
 A_i=
 \begin{pmatrix}
 0&u_i^{\mathsf T}\\
 u_i&2M_i
 \end{pmatrix}.
\]

Then the full-AQ coefficient space is exactly the hyperplane

\[
 \mathcal H=\{A\in\operatorname{Sym}_{55}:A_{00}=0\},
 \qquad\dim\mathcal H=1539,
\]

and

\[
 \boxed{C_{ij}=\frac12\operatorname{tr}(A_iA_j),\qquad
 K_{ij}=\operatorname{tr}(A_i^2A_j).}
\]

Thus r54 full AQ is not an unrelated new cubic model: it is the ordinary trace
cubic on a codimension-one hyperplane of \(\operatorname{Sym}_{55}\).  The gauge
is the subgroup \(\operatorname{diag}(1,O(54))\), of dimension

\[
 \dim O(54)=\frac{54\cdot53}{2}=1431.
\]

The exact dimensions are

\[
 \begin{aligned}
 d_{\rm raw}&=1024(54+1485)=1,575,936,\\
 d_{\rm quotient}&=1,575,936-1,431=1,574,505,\\
 d_{\rm obs}&=\frac{1024\cdot1025}{2}+1024^2
 =524,800+1,048,576=1,573,376,\\
 d_{\rm quotient}-d_{\rm obs}&=1,129.
 \end{aligned}
\]

The positive count permits a submersion; it does not prove one.

## 2. Exact differential

For a tangent \((h_i,H_i)\),

\[
 \boxed{
 DC_{ij}=h_i^{\mathsf T}u_j+u_i^{\mathsf T}h_j
 +2\operatorname{tr}(H_iM_j+M_iH_j).}
\]

The repeated-cubic differential is

\[
 \boxed{
 \begin{aligned}
 DK_{ij}={}&8\operatorname{tr}[(M_iH_i+H_iM_i)M_j+M_i^2H_j]\\
 &+4h_i^{\mathsf T}M_ju_i+2u_i^{\mathsf T}H_ju_i\\
 &+4h_i^{\mathsf T}M_iu_j+4u_i^{\mathsf T}H_iu_j
 +4u_i^{\mathsf T}M_ih_j.
 \end{aligned}}
\]

Equivalently, with

\[
 \dot A_i=\begin{pmatrix}0&h_i^{\mathsf T}\\h_i&2H_i\end{pmatrix},
\]

\[
 DC_{ij}=\frac12\operatorname{tr}(\dot A_iA_j+A_i\dot A_j),
\]

\[
 DK_{ij}=\operatorname{tr}[(A_i\dot A_i+\dot A_iA_i)A_j+A_i^2\dot A_j].
\]

The infinitesimal gauge is

\[
 h_i=\Omega u_i,\qquad H_i=[\Omega,M_i],\qquad \Omega^{\mathsf T}=-\Omega.
\]

For the AQ metric \(\sum_i(\|h_i\|^2+2\|H_i\|_F^2)\), an exact horizontal
slice is

\[
 \boxed{
 \operatorname{skew}\!\left(
 \sum_i u_i h_i^{\mathsf T}+2\sum_i[M_i,H_i]
 \right)=0.}
\]

At a regular point, a local section is conditioned by the smallest singular value
of \(D\Phi_{54}\) restricted to this slice.

## 3. Affine structured seam \(M_i=0\): exact rank obstruction

Put \(M_i=0\), and let \(U\in\mathbb R^{n\times r}\) have rows
\(u_i^{\mathsf T}\) and full column rank.  Then

\[
 DC= hU^{\mathsf T}+Uh^{\mathsf T},
\]

so

\[
 \boxed{
 \operatorname{rank}DC=nr-\frac{r(r-1)}2
 =55,296-1,431=53,865.}
\]

Its range is the tangent to the rank-54 positive Gram variety.  Its orthogonal
normal is

\[
 \mathcal N_U=\{Z=Z^{\mathsf T}:ZU=0\},
\]

with exact dimension

\[
 \boxed{
 \dim\mathcal N_U=\frac{(n-r)(n-r+1)}2
 =\frac{970\cdot971}{2}=470,935.}
\]

At the same point, \(DK\) receives no \(h\)-variation and is

\[
 \boxed{
 (F_UH)_{ij}=2u_i^{\mathsf T}H_ju_i+4u_i^{\mathsf T}H_iu_j.}
\]

Even if \(F_U\) is surjective onto all \(n^2=1,048,576\) K21 coordinates,
the complete differential satisfies

\[
 \boxed{
 \operatorname{rank}D\Phi_{54}
 \le1,048,576+53,865=1,102,441,}
\]

and therefore has observation cokernel at least

\[
 \boxed{1,573,376-1,102,441=470,935.}
\]

Hence the affine/raw-Gaussian seam is **not** a local section of the joint
\((C,K21)\) interface.  The extra full-AQ parameter count cannot be borrowed at
\(M=0\); the quadratic prefix must already be nonzero.

## 4. Exact finite-minor certificate that the affine K21 block can be full

The preceding obstruction comes entirely from covariance.  The K21 block itself
is generically a submersion, and this can be certified without a numerical
Jacobian.

Let

\[
 \mathcal V_U:\mathbb R^n\to\operatorname{Sym}_r,
 \qquad \mathcal V_Uc=\sum_i c_i u_iu_i^{\mathsf T}.
\]

For each \(p\), let \(P_p\) be orthogonal projection onto \(u_p^\perp\) and

\[
 \mathcal V_{U,p}^{\perp}c
 =\sum_{i\ne p}c_i(P_pu_i)(P_pu_i)^{\mathsf T}.
\]

The checkable minor conditions are

\[
 \boxed{
 \operatorname{rank}\mathcal V_U=1024,
 \qquad
 \operatorname{rank}\mathcal V_{U,p}^{\perp}=1023
 \quad(p=1,\ldots,1024).}
\]

They fit dimensionally because

\[
 1024\le\dim\operatorname{Sym}_{54}=1485,
 \qquad
 1023\le\dim\operatorname{Sym}_{53}=1431.
\]

Equivalently: one \(1024\times1024\) minor of the symmetric-Veronese matrix is
nonzero, and for each \(p\), one \(1023\times1023\) minor of the projected
Veronese matrix is nonzero.  Such tuples exist generically: conditional on any
nonzero \(u_p\), continuously generic projected vectors in \(\mathbb R^{53}\)
have independent outer products up to the available dimension; the failure sets
are proper algebraic sets, and there are only finitely many \(p\).

To prove surjectivity, let \(\Lambda\in\mathbb R^{n\times n}\) annihilate
\(F_U\).  The coefficient of every \(H_p\) is

\[
 \sum_i\Lambda_{ip}u_iu_i^{\mathsf T}
 +2\operatorname{sym}\!\left(
 u_p\Big(\sum_j\Lambda_{pj}u_j\Big)^{\mathsf T}
 \right)=0.
\]

The projected-minor condition says

\[
 \operatorname{span}\{u_iu_i^{\mathsf T}\}_i
 \cap\{\operatorname{sym}(u_pa^{\mathsf T}):a\in\mathbb R^r\}
 =\operatorname{span}\{u_pu_p^{\mathsf T}\}.
\]

Thus column \(p\) of \(\Lambda\) is supported only at \(p\).  This holds for
every \(p\), so \(\Lambda\) is diagonal.  Substitution leaves
\(3\Lambda_{pp}u_pu_p^{\mathsf T}=0\), hence \(\Lambda=0\).  Therefore

\[
 \boxed{\operatorname{rank}F_U=n^2=1,048,576.}
\]

At a tuple satisfying these minors the affine total rank is exactly 1,102,441,
not merely bounded by it.  The raw kernel has dimension

\[
 1,575,936-1,102,441=473,495;
\]

after removing the 1,431-dimensional gauge, its dimension is 472,064.

## 5. The only near-affine full-row-rank escape

Set

\[
 M_i=tN_i
\]

and retain an affine tuple \(U\) satisfying the K21 minors above.  Since \(F_U\)
is onto, the K21 equation can be eliminated analytically for small \(t\).  The
leading new covariance normal is generated by

\[
 \mathcal L_N(H)_{ij}
 =2\operatorname{tr}(H_iN_j+N_iH_j).
\]

Let \(\Pi_U\) be orthogonal projection from \(\operatorname{Sym}_n\) onto
\(\mathcal N_U\), equivalently \(\Pi_U(Y)=P_UYP_U\), where

\[
 P_U=I-U(U^{\mathsf T}U)^{-1}U^{\mathsf T}.
\]

The exact Schur map is

\[
 \boxed{
 \mathscr S_{U,N}
 =\Pi_U\mathcal L_N\big|_{\ker F_U}:
 \ker F_U\longrightarrow\mathcal N_U.}
\]

Its dimensions are the knife edge

\[
 \dim\ker F_U
 =1024\cdot1485-1024^2=472,064,
\]

\[
 \dim\mathcal N_U=470,935,
 \qquad
 472,064-470,935=1,129.
\]

Therefore the near-affine r54 seam has a full-row-rank local section **if**

\[
 \boxed{\operatorname{rank}\mathscr S_{U,N}=470,935.}
\]

This is an exact \(470,935\times470,935\) minor certificate.  Choose:

1. \(1,048,576\) \(H\)-columns giving an invertible \(F_U\) minor;
2. \(53,865\) horizontal \(h\)-columns spanning the affine covariance tangent;
3. \(470,935\) columns in \(\ker F_U\) giving an invertible Schur minor.

For the corresponding square Jacobian minor,

\[
 \boxed{
 \det J(t)=c\,t^{470,935}+O(t^{470,936}),
 \qquad c\ne0
 }
\]

exactly when the three displayed minors are nonzero.  The analytic implicit
function theorem then supplies a local section for every sufficiently small
nonzero \(t\) away from the finite roots of this minor.

The conditioning warning is equally exact: the newly opened normal singular
values scale as

\[
 \sigma_{\rm new}(D\Phi_{54})
 =|t|\,\sigma(\mathscr S_{U,N})+O(t^2).
\]

Thus a section obtained arbitrarily close to the affine seam is necessarily
ill-conditioned like

\[
 \|(D\Phi_{54}|_{\rm horizontal})^+\|
 \gtrsim\frac1{|t|\,\sigma_{\min}(\mathscr S_{U,N})}.
\]

### Adjoint form of the Schur certificate

Let \(Z\in\mathcal N_U\), so \(Z=Z^{\mathsf T}\) and \(ZU=0\).  Then

\[
 (\mathcal L_N^*Z)_p=4\sum_jZ_{pj}N_j.
\]

The Schur map is onto iff the only \(Z\in\mathcal N_U\) for which there is a
\(\Lambda\in\mathbb R^{n\times n}\) satisfying

\[
 \boxed{
 4\sum_jZ_{pj}N_j
 =2\sum_i\Lambda_{ip}u_iu_i^{\mathsf T}
 +4\operatorname{sym}\!\left(
 u_p\Big(\sum_j\Lambda_{pj}u_j\Big)^{\mathsf T}
 \right)
 \quad\forall p
 }
\]

is \(Z=0\).  This is often a more tractable exact symbolic certificate than
forming the primal half-million-dimensional minor.

## 6. Verdict

The tractable affine/full-Gaussian r54 seam has a rigorous rank obstruction:
even with a fully surjective K21 differential it misses exactly 470,935 covariance
normal directions.  Therefore no local section exists at \(M=0\).

A nonzero quadratic prefix can reopen the map, but dimension counting is not the
certificate.  The binding object is the exact Schur minor
\(\mathscr S_{U,N}\).  This note derives the complete test and conditioning law;
it does not claim that a particular deployed r54 prefix passes it.  Until an exact
nonzero primal minor or the equivalent adjoint injectivity receipt is supplied,
the r54 full-AQ replacement seam is algebraically **conditional**, not certified.

## Atomic graph-ready nodes

### `R54_FULL_AQ_TRACE_HYPERPLANE`

- status: `exact_model_equivalence`;
- statement: \((u,M)\mapsto\begin{psmallmatrix}0&u^T\\u&2M\end{psmallmatrix}\in\operatorname{Sym}_{55}\) with \(C=\tfrac12\operatorname{tr}(A_iA_j)\), \(K21=\operatorname{tr}(A_i^2A_j)\);
- dimensions: raw 1,575,936; O54 gauge 1,431; quotient 1,574,505; observations 1,573,376; nominal surplus 1,129.

### `R54_AFFINE_SEAM_RANK_OBSTRUCTION`

- status: `proved_no_local_section_at_M_zero`;
- exact ranks under the Veronese minors: \(\operatorname{rank}DC=53,865\), \(\operatorname{rank}DK=1,048,576\), total 1,102,441;
- cokernel: 470,935 covariance-normal directions \(Z=Z^T,ZU=0\);
- implication: raw-Gaussian/affine initialization cannot use the r54 full-AQ capacity count.

### `R54_AFFINE_K21_VERONESE_MINOR`

- status: `exact_finite_minor_certificate`;
- minors: rank 1,024 outer-product bank and rank 1,023 projected outer-product bank for every unit;
- consequence: affine \(DK21\) is onto all 1,048,576 oriented entries.

### `R54_NEAR_AFFINE_SCHUR_SECTION_GATE`

- status: `exact_iff_gate_open_for_specific_prefix`;
- map: \(\mathscr S_{U,N}=\Pi_U\mathcal L_N|_{\ker F_U}\), dimensions 472,064 to 470,935;
- pass: one nonzero 470,935 minor, equivalently the displayed adjoint has no nonzero \(Z\);
- residual fiber after gauge: 1,129 dimensions;
- conditioning: new singular values scale as \(|t|\sigma(\mathscr S_{U,N})\);
- current conclusion: affine point closed; nonzero-prefix r54 remains conditional until the exact minor is supplied.

## 7. Conditional compression theorem (hypothesis later proved impossible)

> **Superseded gate.**  Sections 7--9 record a correct conditional incidence
> argument, but Section 10 proves that its opening projection-surjectivity
> hypothesis can never hold for the affine AQ tangent.  They must not be cited
> as evidence that a passing prefix exists.  They are retained to make the
> failed route and its exact assumption auditable.

The Schur gate can be pushed beyond dimension counting.  This section gives a
rigorous scalable sufficient theorem for existence of a prefix \(N\).

Let

\[
 P=\mathbb R^{1024}=L\oplus K,
 \qquad L=\operatorname{col}U,\quad \dim L=54,
 \quad \dim K=m=970,
\]

and let

\[
 E=\operatorname{Sym}_{54},\qquad d=\dim E=1485.
\]

Identify an \(H\)-bank with \(P\otimes E\).  Let

\[
 \mathcal A=\operatorname{range}F_U^*\subset P\otimes E,
 \qquad
 \mathcal B=\mathcal A\cap(K\otimes E).
\]

Assume the projection of \(\mathcal A\) onto \(L\otimes E\) is onto.  This is one
exact rank-\(rd=54\cdot1485=80,190\) minor.  Since \(F_U^*\) is injective,

\[
 \boxed{
 \dim\mathcal B=n^2-rd
 =1,048,576-80,190=968,386.}
\]

The quotient in which the Schur image must be independent has dimension

\[
 \boxed{
 q_0=\dim(K\otimes E)-\dim\mathcal B
 =970\cdot1485-968,386=472,064.}
\]

This equals \(\dim\ker F_U\), as it must.

For \(1\le s\le m\), define the exact row-compression profile

\[
 \boxed{
 \beta_s(\mathcal B)
 =\max_{R\in\operatorname{Gr}(s,K)}
 \dim\{\mathcal B\cap(R\otimes E)\}.}
\]

Let

\[
 v_s=\frac{s(2m-s+1)}2
\]

be the dimension of the variety of symmetric \(m\times m\) matrices of rank at
most \(s\).  Then:

### Compression-transversality theorem

If

\[
 \boxed{
 sd-\beta_s(\mathcal B)>v_s
 \qquad\text{for every }s=1,\ldots,m,}
\]

then a Zariski-open dense set of full-row-rank prefix banks
\(N_K\in K\otimes E\) satisfies

\[
 \mathscr S_{U,N}:\ker F_U\twoheadrightarrow\mathcal N_U.
\]

In particular, such an \(N\) exists and the near-affine r54 map is a submersion
for sufficiently small nonzero \(t\).

### Proof

Write a prefix as an injection \(J:K\to E\); in bases, its rows are the matrices
\(N_a\).  For \(Z\in\operatorname{Sym}K\),

\[
 \mathcal L_N^*Z=4(I_K\otimes J)Z.
\]

Fix a rank-\(s\) matrix \(Z\) with image \(R\).  As \(J\) varies, the map

\[
 J\longmapsto(I_K\otimes J)Z\pmod{\mathcal B}
\]

has rank at least

\[
 sd-\dim[\mathcal B\cap(R\otimes E)]
 \ge sd-\beta_s(\mathcal B).
\]

Therefore the set of \(J\)'s for which this fixed \(Z\) lands in \(\mathcal B\)
has codimension at least \(sd-\beta_s\).  The rank-at-most-\(s\) symmetric
variety has dimension \(v_s\).  If \(sd-\beta_s>v_s\), the incidence union over
all nonzero rank-\(s\) matrices has dimension strictly below \(\dim\operatorname{Hom}(K,E)\).
Taking the finite union over \(s\) leaves a proper algebraic bad set.  Outside it,

\[
 (I_K\otimes J)(\operatorname{Sym}K)\cap\mathcal B=\{0\},
\]

which is exactly adjoint injectivity of the Schur map.  This proves the theorem.

### Exact r54 margins under the minimal compression profile

The smallest possible intersection profile for a \(b\)-plane in an
\(md\)-dimensional ambient space is

\[
 \beta_s^{\min}=\max(0,b+sd-md)
 =\max(0,sd-q_0).
\]

If \(\mathcal B\) has this no-excess-compression profile, then:

- when \(sd\le q_0\),

\[
 sd-v_s
 =\frac{s(1029+s)}2>0;
\]

- when \(sd>q_0\), the incidence codimension is \(q_0=472,064\), while

\[
 v_s\le v_m=\frac{970\cdot971}{2}=470,935.
\]

The worst-case strict margin is therefore

\[
 \boxed{q_0-v_m=472,064-470,935=1,129.}
\]

Thus the nominal 1,129-dimensional surplus is precisely the algebraic incidence
margin, not just a parameter count.  What remains to certify for a concrete
\(U\) is absence of **excess row compression** in \(\mathcal B\).

An equivalent integer check is

\[
 \boxed{
 \beta_s(\mathcal B)\le sd-v_s-1
 \quad(s=1,\ldots,970).}
\]

These are deterministic ranks of tensor-compression maps.  A violation exhibits
a concrete obstruction subspace \(R\); satisfaction proves that generic exact
prefixes \(N\) pass the Schur gate.

## 8. Structured combinatorial minor for an explicit prefix

There is also a finite coordinate certificate substantially more structured than
an arbitrary half-million-dimensional determinant.

Choose bases \(k_1,\ldots,k_m\) of \(K\) and
\(e_1,\ldots,e_d\) of \(E\).  Let

\[
 \pi:K\otimes E\to(K\otimes E)/\mathcal B.
\]

If there is an injection

\[
 \varphi:\{1,\ldots,m\}\hookrightarrow\{1,\ldots,d\}
\]

such that the following \(m(m+1)/2=470,935\) quotient vectors are independent,

\[
 \boxed{
 \begin{aligned}
 &\pi(k_a\otimes e_{\varphi(a)}), &&1\le a\le m,\\
 &\pi(k_a\otimes e_{\varphi(b)}
      +k_b\otimes e_{\varphi(a)}), &&1\le a<b\le m,
 \end{aligned}}
\]

then the explicit prefix

\[
 \boxed{N_a=e_{\varphi(a)}}
\]

passes the Schur gate.  The displayed vectors are exactly the images of the
standard symmetric matrix basis under \(Z\mapsto ZN\).  Their independence is a
single exact \(470,935\)-minor, and the quotient has 1,129 unused dimensions.

This converts prefix construction into a symmetric matroid-matching problem.  A
block version permits a direct sum of injections \(\varphi_b\) on blocks of \(K\);
cross-block symmetric pairs must still be included, so merely checking diagonal
blocks is insufficient.

## 9. Current sharpened status

The r54 question is now separated into three exact levels:

1. **Affine seam:** impossible, by the 470,935-dimensional covariance-normal
   obstruction.
2. **Nonzero prefix with compression inequalities:** existence of a passing
   generic \(N\) is proved by the incidence theorem.
3. **Concrete deployed prefix:** must supply either the compression-profile bounds
   or the structured quotient minor.  Dimension alone remains insufficient.

### Additional graph-ready node: `R54_SCHUR_COMPRESSION_EXISTENCE_THEOREM`

- status: `exact_generic_existence_under_checkable_compression_profile`;
- spaces: \(K=970\), \(E=1485\), \(\dim\mathcal B=968,386\), quotient 472,064;
- hypothesis: \(sd-\beta_s> s(1941-s)/2\) for all \(s\);
- minimal-profile margin: exactly 1,129 at full rank;
- consequence: a Zariski-open dense set of nonzero prefixes gives full Schur rank 470,935;
- explicit alternative: symmetric quotient-matching minor built from 970 selected feature coordinates;
- unresolved concrete receipt: prove the affine-adjoint range \(\mathcal B\) has the required compression profile for the chosen \(U\), or exhibit the matching minor.

## 10. Decisive hook-symmetry obstruction: no prefix can make the Schur map onto

The missing receipt above is not merely difficult: its first hypothesis is
algebraically impossible.  The obstruction is the mixed-symmetry, or
\([2,1]\)-hook, component of a shared affine tangent.

Let

\[
 L=\operatorname{col}U,\qquad \dim L=r,
 \qquad E=\operatorname{Sym}^2L,\qquad d=\frac{r(r+1)}2.
\]

An element of the shared index-linear sector \(L\otimes E\) has the form

\[
 H_i=H(u_i)=\sum_{b=1}^r (u_i)_bG_b,
 \qquad G_b\in\operatorname{Sym}_r.
\]

Encode it by the trilinear form

\[
 T(x,y,z)=x^TH(y)z,
 \qquad T(x,y,z)=T(z,y,x).
\]

For \(x=u_i\), \(y=u_j\), the affine \(K_{21}\) differential is

\[
 \begin{aligned}
 (F_UH)_{ij}
 &=2x^TH(y)x+4x^TH(x)y\\
 &=2T(x,y,x)+4T(x,x,y)\\
 &=6(\operatorname{Sym}T)(x,x,y).
 \end{aligned}
\]

The last identity follows because symmetry of the first and third slots gives

\[
 (\operatorname{Sym}T)(x,x,y)
 =\frac{T(x,y,x)+2T(x,x,y)}3.
\]

Hence the restriction of \(F_U\) to \(L\otimes E\) factors exactly through
full symmetrization:

\[
 \boxed{
 L\otimes\operatorname{Sym}^2L
 \longrightarrow \operatorname{Sym}^3L
 \longrightarrow \mathbb R^{n\times n}.}
\]

This is independent of the choice of rows \(u_i\).  Therefore

\[
 \operatorname{rank}\bigl(F_U|_{L\otimes E}\bigr)
 \le \dim\operatorname{Sym}^3L
 =\binom{r+2}{3}.
\]

At \(r=54\),

\[
 rd=54\cdot1485=80,190,
 \qquad \binom{56}{3}=27,720,
\]

so the universal hook kernel has dimension at least

\[
 \boxed{80,190-27,720=52,470.}
\]

Equivalently, the projection of
\(\mathcal A=\operatorname{range}F_U^*\) onto \(L\otimes E\) has rank at most
27,720, never 80,190.  Thus the assumption opening Section 7 is impossible.

Now retain the already-certified surjectivity of \(F_U\), so
\(\dim\mathcal A=n^2=1,048,576\), and put

\[
 \mathcal B=\mathcal A\cap(K\otimes E),
 \qquad K=L^\perp,\quad \dim K=m=970.
\]

Because the kernel of the projection
\(\mathcal A\to L\otimes E\) is exactly \(\mathcal B\),

\[
 \dim\mathcal B
 \ge 1,048,576-27,720
 =1,020,856.
\]

Consequently

\[
 \boxed{
 \dim\frac{K\otimes E}{\mathcal B}
 \le 970\cdot1485-1,020,856
 =419,594.}
\]

For every prefix bank \(N\), the adjoint of the first-order normal Schur map is

\[
 \mathscr S_{U,N}^*: \operatorname{Sym}K
 \longrightarrow (K\otimes E)/\mathcal B,
 \qquad
 Z\longmapsto[4ZN].
\]

It therefore has rank at most 419,594, whereas

\[
 \dim\operatorname{Sym}K=\frac{970\cdot971}{2}=470,935.
\]

Thus, for **every** \(N\),

\[
 \boxed{
 \operatorname{rank}\mathscr S_{U,N}\le419,594,
 \qquad
 \operatorname{cork}\mathscr S_{U,N}\ge51,341.}
\]

This proves the requested impossibility: no explicit, generic, finite-field, or
otherwise chosen prefix \(N\) can have
\(\operatorname{rank}\mathscr S_{U,N}=470,935\).

### Near-affine conditioning consequence

The crude analyticity consequence is that the missing modes open no earlier
than second order.  Section 11 performs the block elimination exactly and
strengthens this: after quotienting the persistent orthogonal gauge, the normal
Schur series has only orders \(t,t^3,\ldots\).  Thus the 51,341 missing modes
open no earlier than order \(t^3\), and any eventual local section is at best
\(\Omega(t^{-3})\)-conditioned along them.

### Replacement graph-ready node: `R54_SCHUR_HOOK_NOGO`

- status: `proved_universal_first_order_nogo`;
- factorization: shared index-linear tangents pass only through
  \(\operatorname{Sym}^3\mathbb R^{54}\);
- universal hook kernel: 52,470 dimensions;
- normal quotient capacity: at most 419,594;
- required normal rank: 470,935;
- exact deficit: at least 51,341 for every prefix \(N\);
- conditioning: Section 11 strengthens the raw perturbative bound: those
  51,341 modes can open no earlier than order \(t^3\), so a finite-prefix
  section, if one exists, is at best \(\Omega(t^{-3})\)-conditioned
  near the affine seam;
- supersedes: `R54_SCHUR_COMPRESSION_EXISTENCE_THEOREM` as an actionable GO;
  Sections 7--9 survive only as a conditional theorem with an impossible AQ
  premise.

## 11. Exact block elimination: the next escape is cubic, not quadratic

The full differential at \((u_i,M_i)=(u_i,tN_i)\) has the exact block form

\[
 \boxed{
 J(t)=
 \begin{pmatrix}
 D & tL_N\\
 tG_N & F_U+t^2Q_N
 \end{pmatrix}.}
\]

Here rows are ordered as \((C,K_{21})\), columns as \((h,H)\), and

\[
 \begin{aligned}
 (Dh)_{ij}
 &=h_i^Tu_j+u_i^Th_j,\\
 (L_NH)_{ij}
 &=2\operatorname{tr}(H_iN_j+N_iH_j),\\
 (G_Nh)_{ij}
 &=4h_i^TN_ju_i+4h_i^TN_iu_j+4u_i^TN_ih_j,\\
 (Q_NH)_{ij}
 &=8\operatorname{tr}\!\left((N_iH_i+H_iN_i)N_j+N_i^2H_j\right).
 \end{aligned}
\]

The powers of \(t\) are exact, not asymptotic bookkeeping: covariance is
quadratic and \(K_{21}\) is linear plus cubic in the AQ parameters.

Choose:

- a complement to the infinitesimal \(O(r)\) gauge on which \(D\) is an
  isomorphism onto the affine covariance tangent space;
- a right inverse \(R\) of the surjection \(F_U\);
- a right inverse \(P\) of \(D\) on that tangent space;
- projections \(P_T\) and \(\Pi_U\) onto the covariance tangent and normal
  spaces.

For \(H_0\in\ker F_U\), cancel its order-\(t\) tangent covariance by taking

\[
 h=t h_1,
 \qquad
 h_1=-P P_TL_NH_0.
\]

The resulting order-\(t^2\) error in \(K_{21}\) is

\[
 t^2\left(Q_NH_0+G_Nh_1\right).
\]

Cancel it with

\[
 H=H_0+t^2H_2,
 \qquad
 H_2=-R\left(Q_NH_0-G_NP P_TL_NH_0\right).
\]

Substitution into covariance gives the reduced normal map

\[
 \boxed{
 J_{\mathrm{normal}}(t)H_0
 =t\mathscr S_1H_0+t^3\mathscr S_3H_0+O(t^5),}
\]

with

\[
 \boxed{
 \begin{aligned}
 \mathscr S_1
 &=\Pi_UL_N|_{\ker F_U},\\
 \mathscr S_3
 &=-\Pi_UL_NR
   \left(Q_N-G_NP P_TL_N\right)|_{\ker F_U}.
 \end{aligned}}
\]

Using the analytic inverse of \(F_U+t^2Q_N\) and the analytic inverse of the
tangent block shows that the omitted terms continue in odd powers
\(t^5,t^7,\ldots\).  Infinitesimal gauge columns do not supply an order-\(t^2\)
escape: at nonzero \(t\) they pair with
\(H_i=t[N_i,\Omega]\) to form the exact persistent \(O(r)\) kernel, and were
removed by the chosen gauge complement.

Therefore a finite-prefix r54 submersion requires the induced cubic map

\[
 \boxed{
 \overline{\mathscr S}_3:
 \ker\mathscr S_1\longrightarrow
 \operatorname{coker}\mathscr S_1}
\]

to be onto.  At the maximal possible first-order rank 419,594, its exact
dimensions are

\[
 \dim\ker\mathscr S_1
 =472,064-419,594=52,470,
\]

\[
 \dim\operatorname{coker}\mathscr S_1
 =470,935-419,594=51,341.
\]

The same 1,129-dimensional quotient surplus reappears:

\[
 52,470-51,341=1,129.
\]

So the exact fallback seam is now sharply stated.  A first-order prefix is
universally impossible; a cubic hook-to-normal conversion is dimensionally
possible but needs a genuine rank certificate for
\(\overline{\mathscr S}_3\).  If it succeeds, 419,594 modes scale as \(t\) and
at least 51,341 scale as \(t^3\), forcing
\(\kappa_{\mathrm{right}}=\Omega(t^{-3})\).  This is a conditioning statement
only and remains separate from any endpoint-value or deployment-cost claim.

### Additional graph-ready node: `R54_CUBIC_HOOK_ESCAPE_GATE`

- status: `exact_required_gate_not_certified`;
- exact reduced series: \(t\mathscr S_1+t^3\mathscr S_3+O(t^5)\);
- first-order maximum: 419,594;
- cubic source: \(\ker\mathscr S_1\), dimension at least 52,470;
- cubic target: \(\operatorname{coker}\mathscr S_1\), dimension at least 51,341;
- best-case surplus: 1,129;
- required receipt: full row rank of the induced cubic map
  \(\overline{\mathscr S}_3\);
- unavoidable conditioning if passed: at least 51,341 singular modes are
  \(O(t^3)\), hence local right-inverse norm \(\Omega(t^{-3})\).

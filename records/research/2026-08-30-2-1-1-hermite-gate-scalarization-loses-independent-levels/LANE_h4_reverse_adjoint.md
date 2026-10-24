# H4 reverse-adjoint / NCQC quotient lane

Date: 2026-08-30/31  
Lane: `h4_reverse_adjoint`  
Method: pure derivation only; no scientific numerical runs  
Current status: **conditional theory pass for the frozen one-insertion H4 branch; self-consistent pass only under the Jordan-quotient intertwining and leakage gates below.**

## 0. Executive result

There are three distinct conclusions.

1. A scalar endpoint can accumulate every bare H4 square birth with one `r^4 = O(n^2)` object and `O(L n^3)` work. This is exact for the frozen, centered, zero-bias, second-order tangent because the even-H4-to-even-H4 ReLU derivative is a common scalar.
2. A generic correlated/self-consistent ReLU reset breaks this collapse. The exact reverse H4 message then has neuron-versus-latent separation rank `n`, requiring `O(n^3)` state and `O(n^4)` dense work.
3. There is a genuine loophole for the NCQC core. If q4(211), the cubic square master, reset, feedback, and endpoint all factor through the one-matrix Jordan contraction, the terminal module has dimension at most `dim J = O(n)`, even for a fully noncommutative signed-H3 core. That gives `O(n^2)` state and `O(n^3)` work. The claim is invalid if any value-significant operation sees `ker J`.

The correlated-ReLU leakage is now quantitatively gated. In the worst coherent 10/14-layer interpretation it must be below about `1.39% / 0.994%` per layer in the endpoint-weighted norm. A fresh-cavity spectral bound shows that a diffuse `r=45..55` gate perturbation can meet this only when its quadratic RMS is about `3–4%` or less of the independent cavity RMS; concentrated/rank-one gates require sub-percent ratios.

---

## 1. Normalizations and pure H4 square

Let

\[
Q_M(z)=\frac12\left(z^\top Mz-\operatorname{tr}M\right),
\qquad M\in\operatorname{Sym}_r,
\qquad z\sim N(0,I_r).
\]

Then

\[
E[Q_MQ_N]=\frac12\operatorname{tr}(MN).
\]

The pure H4 square is

\[
U_M
=Q_M^2-\frac12\operatorname{tr}(M^2)-2Q_{M^2}
=\frac14 I_4(\operatorname{sym}M\otimes M).
\]

Its polarization is

\[
U_{M,N}
=Q_MQ_N-\frac12\operatorname{tr}(MN)-Q_{MN+NM}
=\frac14 I_4(M\widetilde\otimes N).
\]

These subtractions are binding. Using `Q_M^2` as a new H4 source would count the H0 scalar and H2 matrix `2Q_{M^2}` again, even though those paths already live in the deployed mean/covariance closure.

Let

\[
\mathcal E=\operatorname{Sym}_r,
\qquad d=\dim\mathcal E=\frac{r(r+1)}2,
\qquad \mathcal H_4=\operatorname{Sym}^4(\mathbb R^r).
\]

At `r=45`, `d=1035`, and a square-lift matrix has

\[
d^2=1{,}071{,}225=O(n^2)
\]

entries. The fully symmetric H4 dimension is

\[
\binom{48}{4}=194{,}580.
\]

---

## 2. Exact frozen one-insertion reverse recurrence

Let `B_{ell i} in H4` be the inherited H4 coefficient at neuron `i`; let `A_{ell i} in E` be the already-available H2 root. At frozen second order, write

\[
\widetilde B_\ell=W_\ell^\top B_{\ell-1},
\]

\[
B_{\ell i}
=d_{\ell i}\widetilde B_{\ell i}
+\kappa_{\ell i}U_{A_{\ell i}}.
\]

For terminal scalar readout `h^T B_L`, define

\[
\alpha_L=h,
\qquad
\alpha_{\ell-1}=W_\ell(d_\ell\odot\alpha_\ell).
\]

Repeated substitution gives the exact coefficient-valued Duhamel identity

\[
\boxed{
T_4
=\sum_i\alpha_{0i}B_{0i}
+\sum_{\ell=1}^L\sum_i
\alpha_{\ell i}\kappa_{\ell i}U_{A_{\ell i}}.
}
\]

Every dense `W_ell` moves onto the scalar physical adjoint `alpha_ell`; it never acts on `n` quartic tensors.

If the common reset is a structured operator,

\[
\mathcal R_{\ell i}=d_{\ell i}\mathcal R_\ell,
\]

then, after the backward `alpha` pass, one H4 accumulator obeys

\[
Z_0=\sum_i\alpha_{0i}B_{0i},
\qquad
Z_\ell=\mathcal R_\ell Z_{\ell-1}
+\sum_i\alpha_{\ell i}b_{\ell i}.
\]

### Weighted-square construction

Let `V_ell in R^{n x d}` contain `svec(A_{ell i})` as rows. Each entire layer birth is the signed Gram

\[
K_\ell
=V_\ell^\top
\operatorname{diag}(\alpha_\ell\odot\kappa_\ell)
V_\ell,
\]

followed by the fixed H4 Wick/symmetrization map. Signed weights must remain signed; there is no square-root factorization when weights are negative.

Consequently:

- state: `O(d^2)=O(n^2)` plus the existing `n x d` H2 roots;
- birth work: `O(n d^2)=O(n^3)` per layer;
- reverse dense transport: `O(n^2)` per layer;
- avoided forward H4 state: `n d^2=O(n^3)`;
- avoided forward H4 dense mixing: `O(n^4)` per layer.

---

## 3. Why centered ReLU makes the tangent recurrence exact

Let `g(z)=a^T z` be centered and let `F,G in H4`. Both are even under `z -> -z`. Hence

\[
E[FG1_{\{g>0\}}]
=E[FG1_{\{g<0\}}]
=\frac12E[FG].
\]

Therefore, for unnormalized ReLU,

\[
\boxed{
\Pi_4 1_{\{g>0\}}\big|_{\mathcal H_4}
=\frac12 I_{\mathcal H_4}.
}
\]

No independence between `g` and the H4 polynomial is needed for this pure-even statement. A normalized ReLU only changes the common scalar.

For

\[
\phi(g+\varepsilon Q+\varepsilon^2H)
=\phi(g)
+\varepsilon1_{\{g>0\}}Q
+\varepsilon^2\left[
1_{\{g>0\}}H+\frac12\delta(g)Q^2
\right]
+O(\varepsilon^3),
\]

the inherited H4 state propagates through a scalar at order `epsilon^2`, and the new pure H4 birth is the Wick-normal-ordered square. H4 feedback into the gate begins at a higher perturbative order. Thus the one-stream reverse formula is exact for the frozen second-order/one-insertion tangent, not for a nonperturbative ReLU closure.

---

## 4. Generic self-consistent rank obstruction

For general local H4 reset operators,

\[
B_{\ell i}
=\mathcal R_{\ell i}
\left(\sum_jW_{\ell,ji}B_{\ell-1,j}\right)
+b_{\ell i},
\]

the scalar terminal adjoint satisfies

\[
\Lambda_{Li}=h_iG_L,
\qquad
\Lambda_{\ell-1,j}
=\sum_iW_{\ell,ji}\mathcal R_{\ell i}^*\Lambda_{\ell i}.
\]

A single H4 tensor direction exists at layer `ell-1` iff

\[
\boxed{
\dim\operatorname{span}
\{\mathcal R_{\ell i}^*G_\ell:\alpha_{\ell i}\ne0\}=1.
}
\]

Equivalently, there must be scalars `d_{ell i}` and one `G_{ell-1}` with

\[
\mathcal R_{\ell i}^*G_\ell=d_{\ell i}G_{\ell-1}
\]

for every live neuron. With square full-rank dense `W`, this is necessary as well as sufficient because multiplication by `W` preserves the neuron-versus-H4 matricization rank.

For generic correlated gates the images are linearly independent. The separation rank becomes `n` after one reset, forcing

\[
\Omega(n\dim\mathcal H_4)=\Omega(nr^4)=\Omega(n^3)
\]

state and

\[
\Omega(n^2\dim\mathcal H_4)=\Omega(n^4)
\]

dense work. In tensor-network language, dense `W` creates an `n`-wide separator; the one-H4-tensor construction assumes bond dimension one.

---

## 5. Cyclic-module theorem and series-parallel loophole

Suppose

\[
\mathcal R_{\ell i}
=\sum_{m=1}^{s_\ell}c_{\ell im}S_{\ell m}.
\]

Low one-layer operator-Schmidt rank `s_ell` is not enough: products of the `S` operators can grow exponentially with depth. Define the unital algebra

\[
\mathfrak A
=\operatorname{alg}\{S_{\ell m}^*: \ell,m\}
\]

and the terminal cyclic module

\[
\mathcal M_G
=\operatorname{span}\{AG:A\in\mathfrak A\}.
\]

Then

\[
\boxed{t=\dim\mathcal M_G}
\]

is the exact minimal number of target-conditioned H4 directions.

For a basis `G_1,...,G_t` satisfying

\[
\mathcal R_{\ell i}^*G_a
=\sum_b C_{\ell i}^{ba}G_b,
\]

write

\[
\Lambda_{\ell i}=\sum_aX_{\ell,ia}G_a.
\]

The exact recurrence is

\[
Y_{\ell,ib}=\sum_aC_{\ell i}^{ba}X_{\ell,ia},
\qquad
X_{\ell-1,:,b}=W_\ell Y_{\ell,:,b},
\]

and the layer birth contribution is

\[
\Delta_\ell
=\sum_{i,a}X_{\ell,ia}\langle G_a,b_{\ell i}\rangle.
\]

Series-parallel diagrams are generated by addition and composition. Both stay inside the same finite algebra. Even a convergent resolvent `(I-Sigma)^{-1}` remains inside it by Cayley-Hamilton. Therefore an infinite series-parallel resummation is cheap iff its terminal cyclic module is small.

---

## 6. Quadratic-core/Jordan closure theorem

Let `A` be a `q`-dimensional unital Jordan algebra inside `Sym_r`, closed under

\[
A\circ B=\frac12(AB+BA),
\]

trace contractions, quadratic representations such as `ABA`, and every functional calculus used by reset. Define

\[
\mathcal K_4(\mathcal A)
=\operatorname{span}\{U_{A,B}:A,B\in\mathcal A\}.
\]

Then

\[
k:=\dim\mathcal K_4(\mathcal A)
\le\frac{q(q+1)}2.
\]

An exact coupled H2/H4 series-parallel closure remains inside

\[
\mathcal A\oplus\mathcal K_4(\mathcal A)
\]

if:

1. every H2 core and oriented gate projector lies in the same `A`;
2. H4 births are polarized squares of elements of `A`;
3. H4-to-H2 contractions return to `A`;
4. reset/self-energy maps preserve `K4(A)`;
5. no diagram cut exposes three independent core labels;
6. dense `W` acts only on physical coefficients.

The exact state and dense cost are then

\[
O(n(q+k))=O(nq^2),
\qquad
O(Ln^2(q+k))=O(Ln^2q^2).
\]

For a common commuting algebra with `q=r~sqrt(n)`, `k~r^2~n`; this is exactly `O(n^2)` state and `O(n^3)` work. A treewidth-three skeleton instead needs `q^3`, producing `O(n^{5/2})` state and `O(n^{7/2})` work when `q~sqrt(n)`.

Two generic symmetric matrices generate the full symmetric/Jordan algebra: diagonalize one with simple spectrum, use its polynomial spectral projectors, and use the nonzero off-diagonal entries of the second to generate every symmetric matrix unit. Thus the small common algebra is exceptional unless a hidden block/commuting structure is proved.

---

## 7. NCQC specialization (`r=45..55`)

The NCQC variables are

\[
Q_i=\frac12(z^\top M_i z-\operatorname{tr}M_i),
\qquad M_i\in\operatorname{Sym}_r.
\]

The signed cubic channel is

\[
\kappa_3(Q_i,Q_j,Q_k)
=\operatorname{tr}(M_iM_jM_k)
=\langle M_i,M_j\circ M_k\rangle_F.
\]

It changes sign under the global parity `M_i -> -M_i`.

### Exact external q4(211)

In this normalization,

\[
\boxed{
\kappa_4(Q_i,Q_i,Q_j,Q_k)
=\langle M_i^2,M_j\circ M_k\rangle
+2\langle M_i\circ M_j,M_i\circ M_k\rangle.
}
\]

The collision faces are

\[
\kappa_{31}(i,j)=3\operatorname{tr}(M_i^3M_j),
\]

\[
\kappa_{22}(i,j)
=\langle M_i^2,M_j^2\rangle
+2\|M_i\circ M_j\|_F^2,
\]

\[
\kappa_4(i)=3\operatorname{tr}(M_i^4).
\]

External injective `211` is unrestricted `211`, minus the two `31` faces and one `22` face, plus twice the all-equal `4` corner.

### Square master

For represented cubic provider `T_0` and NCQC addition `Delta T`, the binding source is

\[
\boxed{
\Delta Q_4
=P_4\left[
2T_0\widetilde\otimes_1\Delta T
+\Delta T\widetilde\otimes_1\Delta T
\right]
=P_4[(T_0+\Delta T)^2-T_0^2].
}
\]

The mixed:square ratio `2:1` is physics, not a fit parameter. For a trace vertex, this square is a Gram of the pair features `M_i circle M_j` and factors through an operator on the one-matrix Jordan space.

### Module classification

Let `J` be the Jordan algebra generated by all NCQC cores and gate/reset matrices; let `q=dim J`.

- common diagonal or bounded-block core: `q=O(r)=O(sqrt(n))`;
- full irreducible `Sym_r` core: `q=r(r+1)/2=1035..1540=O(n)`;
- generic pair-H4 preimage: `dim Sym^2(J)~q^2/2`.

Therefore:

| structure | q4(211)+square module | generic pair-H4 module |
|---|---:|---:|
| commuting/bounded-block | `O(sqrt(n))` | `O(n)` |
| full noncommutative NCQC | `O(n)` | `O(n^2)` |

For the full `r=45..55` core, the pair dimensions are `536,130..1,186,570`. A fully noncommutative core can therefore carry signed H3 and q4(211)+square at `O(n^3)` only when every value path remains in the one-matrix Jordan quotient.

---

## 8. Exact quotient/intertwining gates

Let

\[
\mathcal P=\operatorname{Sym}^2(\mathcal J),
\qquad
J:\mathcal P\to\mathcal J,
\qquad
J(A\odot B)=A\circ B,
\qquad
\mathcal N=\ker J.
\]

Dropping the full pair-H4 bank is exact iff:

1. **algebra closure:** every core, square, Jordan product, reset matrix, gate projector, and feedback insertion remains in one fixed `J`;
2. **reset intertwining:** for every layer there is `Rbar_ell` with
   \[
   J\mathcal R_\ell=\bar{\mathcal R}_\ell J,
   \]
   equivalently `R_ell(N) subset N`;
3. **feedback gate:** every H4-to-H2 map and endpoint consumer annihilates `N`;
4. **square gate:** use the complete difference of squares with the fixed `2:1` ratio;
5. **collision gate:** handle `211/31/22/4` by inclusion-exclusion at read time; equality partitions are not invariant under dense `W`;
6. **provenance gate:** all layers genuinely share a fixed NCQC channel or a fixed-dimensional provenance algebra;
7. **primitive-cost gate:** local pullbacks have implicit matrix/Jordan schedules and do not materialize a `q x q` operator per neuron;
8. **chaos-reset gate:** affine NCQC transport is exact, but literal ReLU does not close on finite quadratic chaos without an additional remainder theorem.

---

## 9. Three different “211” partitions and their exact bridge

### Latent-Hermite `q4_z(211)`

In a fixed whitened coordinate basis, for distinct latent indices `a,b,c`,

\[
\boxed{
u^{z,211}_{M,N;a;bc}
=\frac12(M_{aa}N_{bc}+N_{aa}M_{bc})
+M_{ab}N_{ac}+M_{ac}N_{ab}.
}
\]

For a square,

\[
u^{z,211}_{M;a;bc}
=M_{aa}M_{bc}+2M_{ab}M_{ac}.
\]

This is a basis-dependent projection of one H4 tensor.

### External-label `q4_ext(211)`

For distinct unit labels `i,j,k`,

\[
q4_{\rm ext}(211)(i;j,k)
=2u_{i;jk}+v_{i;jk},
\]

where

\[
u_{i;jk}=\operatorname{tr}(M_i^2M_jM_k),
\qquad
v_{i;jk}=\operatorname{tr}(M_iM_jM_iM_k).
\]

### Connected graph `C4`

The connected Wick contribution is the three-cycle orbit

\[
\boxed{
\Gamma_{C_4}(A,B,C,D)
=\operatorname{tr}(ABCD)
+\operatorname{tr}(ACBD)
+\operatorname{tr}(ABDC).
}
\]

Identifying two external labels yields the multiset `{u,u,v}`; one collided graph representative is not the full external cumulant.

### Exact Wick bridge

The product identity

\[
Q_AQ_B=C_{AB}+Q_{AB+BA}+U_{A,B},
\qquad C_{AB}=\frac12\operatorname{tr}(AB),
\]

gives

\[
\boxed{
\begin{aligned}
\kappa_4(A,B,C,D)
={}&2\langle A\circ B,C\circ D\rangle
+\langle U_{A,B},U_{C,D}\rangle\\
&-C_{AC}C_{BD}-C_{AD}C_{BC}
=\Gamma_{C_4}(A,B,C,D).
\end{aligned}
}
\]

Thus latent `211` is only one input sector to the external cumulant; it cannot replace external `211`, and neither equals a single graph topology.

### Minimal conflation counterexample

Take one latent coordinate, `r=1`, but three distinct external labels, with scalar matrices `M_a=[m_a]`. Then

\[
q4_z(211)=0,
\]

one graph cycle is

\[
m_i^2m_jm_k,
\]

while the external cumulant is

\[
q4_{\rm ext}(211)=3m_i^2m_jm_k.
\]

The three conflated quantities are therefore `0`, `1 x`, and `3 x`.

### Dense transport warning

Full cumulants transform multilinearly under `W`, and the latent Jordan map commutes with physical mixing. But

\[
\Pi_{\rm ext}^{211}(W^\top)^{\otimes4}
\ne
(W^\top)^{\otimes4}\Pi_{\rm ext}^{211}.
\]

Transport the collision-complete NCQC master and reconstruct equality sectors afterward; never propagate isolated external `211` as an invariant state.

---

## 10. Minimal self-consistent ReLU leakage witness

Take

\[
A=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad
B=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
\]

Then

\[
A\circ B=0,
\]

so `K=A odot B` lies in `ker J`. Let `L_A(X)=A circle X` and consider the symmetric one-core insertion

\[
\mathcal S_A=L_A\otimes I+I\otimes L_A.
\]

It sends

\[
K\mapsto A^2\odot B=I\odot B,
\]

and hence

\[
J\mathcal S_AK=B\ne0.
\]

This is the minimal proof that a correlated non-scalar gate can rotate discarded pair-H4 information back into the one-matrix quotient. In normalized Frobenius coordinates the gain is order one; there is no automatic `1/r` protection.

---

## 11. Exact adjoint-weighted leakage identity

Choose the orthogonal decomposition

\[
\mathcal P=\mathcal Q\oplus\mathcal N,
\qquad
\mathcal Q=\mathcal N^\perp,
\qquad
\mathcal N=\ker J.
\]

Write the linearized H4 reset as

\[
\mathcal R_\ell=
\begin{pmatrix}
A_\ell&B_\ell\\
C_\ell&D_\ell
\end{pmatrix},
\qquad
B_\ell=P_\mathcal Q\mathcal R_\ell P_\mathcal N.
\]

The full and quotient recurrences are

\[
q_{\ell+1}=A_\ell q_\ell+B_\ell k_\ell+s_\ell,
\qquad
\widehat q_{\ell+1}=A_\ell\widehat q_\ell+s_\ell.
\]

For terminal `c`, define

\[
\lambda_L=c,
\qquad
\lambda_\ell=A_\ell^*\lambda_{\ell+1}.
\]

Then

\[
\boxed{
\Delta y
=\sum_\ell\langle\lambda_{\ell+1},B_\ell k_\ell\rangle
=\sum_\ell
\langle P_\mathcal N\mathcal R_\ell^*\lambda_{\ell+1},k_\ell\rangle.
}
\]

The exact directional leakage norm is

\[
\boxed{
\beta_\ell^{\rm adj}
=\frac{
\|P_\mathcal N\mathcal R_\ell^*\lambda_{\ell+1}\|
}{\|\lambda_{\ell+1}\|}.
}
\]

If the kernel energy at layer `ell` is `tau_ell V`, the rigorous coherent bound is

\[
\boxed{
\|\Delta y\|_{\rm RMS}
\le\sqrt V\sum_\ell
\beta_\ell^{\rm adj}\sqrt{\tau_\ell}.
}
\]

Only with proved endpoint orthogonality/martingale structure may one use

\[
\|\Delta y\|_{\rm RMS}^2
\le V\sum_\ell
(\beta_\ell^{\rm adj})^2\tau_\ell.
\]

---

## 12. Leakage budgets from the raw target

The ideal channel ladder gives

\[
R_*=9.2547\times10^{-9},
\qquad T=10^{-8}.
\]

The total oracle-removable correction energy is

\[
V
=31.972\times10^{-9}-9.2547\times10^{-9}
=22.7173\times10^{-9},
\]

so

\[
\sqrt V\approx1.5072\times10^{-4}.
\]

The one-stream value-tail gate is

\[
\tau_*=1-0.98346^2=0.0328064,
\qquad
\sqrt{\tau_*}=0.181126.
\]

It corresponds exactly to

\[
\tau_*V=0.7453\times10^{-9}
\]

and the orthogonal RMS slack

\[
b_{\rm orth}=\sqrt{0.7453\times10^{-9}}\approx2.73\times10^{-5}.
\]

Under adversarial alignment, however, the endpoint headroom is only

\[
b_{\rm adv}
=\sqrt{10^{-8}}-\sqrt{9.2547\times10^{-9}}
\approx3.80\times10^{-6}.
\]

Normalized by `sqrt(V)`,

\[
\eta_{\rm adv}=b_{\rm adv}/\sqrt V\approx0.02520.
\]

The binding coherent gate is therefore

\[
\boxed{
\sum_\ell\beta_\ell^{\rm adj}\sqrt{\tau_\ell}
\le0.02520.
}
\]

The orthogonal allowance is `7.19x` larger in amplitude and `51.6x` larger in energy. It is unavailable without a proof of orthogonality.

### Layer accumulations

- One aggregate tail exposure:
  \[
  \beta\le0.02520/0.181126=0.1392.
  \]
- Same full tail pessimistically available at every layer:
  \[
  \beta\le0.0139\quad(L=10),
  \qquad
  \beta\le0.00994\quad(L=14).
  \]
- If instead `sum tau_ell <= tau_*` is proved:
  \[
  \beta\le0.0440\quad(L=10),
  \qquad
  \beta\le0.0372\quad(L=14).
  \]
- Conditional near-critical downstream attenuation `g_ell <= (ell/L)^3` gives, under the pessimistic repeated-tail convention,
  \[
  \beta\le0.0460\quad(L=10),
  \qquad
  \beta\le0.0346\quad(L=14).
  \]

The `3.28064%` number is a residualized target-value tail, not automatically a bound on every layer's `ker J` state.

---

## 13. Analytic NCQC correlated-gate leakage bound

Use a genuine cavity split

\[
g=\xi+Q_A(z),
\qquad
\xi\sim N(0,\sigma^2),
\qquad
\xi\perp z.
\]

Conditional on `z`, the gate probability is

\[
p_A(z)=\Phi(Q_A(z)/\sigma).
\]

Define

\[
s^2=\|Q_A\|_2^2=\frac12\operatorname{tr}(A^2),
\qquad
\delta=s/\sigma,
\qquad
\chi=\|A\|_{\rm op}/s.
\]

The exact H4 product formula is

\[
I_2(A)I_4(f)
=I_6(A\widetilde\otimes f)
+8I_4(A\otimes_1f)
+12I_2(A\otimes_2f).
\]

Because `Q_A=I_2(A)/2`, its H4-to-H4 component is the four-leg second-quantization operator `dGamma(A)`. Hence

\[
\boxed{
\|\Pi_4[Q_A\cdot]\Pi_4\|_{2\to2}
\le4\|A\|_{\rm op}.
}
\]

Taylor's theorem gives

\[
|\Phi(x)-1/2-\phi(0)x|
\le\frac{\phi(0)}6|x|^3.
\]

The centered quadratic-form cumulants imply

\[
E[Q_A^6]
=60\operatorname{tr}A^6
+\frac{45}{2}\operatorname{tr}A^4\operatorname{tr}A^2
+10(\operatorname{tr}A^3)^2
+15s^6.
\]

Using `|tr A^m| <= ||A||op^{m-2} tr A^2`,

\[
\frac{E[Q_A^6]}{s^6}
\le15+130\chi^2+120\chi^4.
\]

Finally, degree-four Gaussian hypercontractivity gives `||F||_4 <= 9||F||_2`. Combining the exact linear term with the cubic Taylor remainder yields the analytic bound

\[
\boxed{
\beta^{\rm adj}
\le
1.596\,\chi\delta
+5.385\sqrt{15+130\chi^2+120\chi^4}\,\delta^3.
}
\]

This is an upper bound on the full H4 multiplication operator and therefore also on its kernel-to-Jordan block.

### Spectral thresholds

For a maximally diffuse spectrum, `chi=sqrt(2/r)`.

- `r=45`, `chi=0.2108`:
  - 10-layer coherent gate `beta<=0.0139`: sufficient `delta <= about 3.75%`;
  - 14-layer coherent gate `beta<=0.00994`: sufficient `delta <= about 2.79%`.
- `r=55`, `chi=0.1907`:
  - 10 layers: sufficient `delta <= about 4.05%`;
  - 14 layers: sufficient `delta <= about 3.04%`.

For concentrated/rank-one spectrum, `chi=sqrt(2)`, and the linear term alone forces roughly

- `delta < about 0.62%` for 10 layers;
- `delta < about 0.44%` for 14 layers.

Thus diffuseness helps by replacing Frobenius gate size with spectral gate size, but no generic NCQC spectrum theorem guarantees the needed ratio.

### Exact fresh-bundle theorem

If the H4 bundle is independent of the gate bundle, then

\[
E[1_{\{g>0\}}\mid z_{H4}]=1/2
\]

and the reset is exactly scalar on the H4 bundle:

\[
\beta^{\rm adj}=0.
\]

This is the strongest possible route. The next open question is whether the matrix-polynomial two-bundle schedule supplies genuine probabilistic independence/conditional independence, or merely two correlated deterministic summaries.

---

## 14. Binding 49-GEMM criterion

The deployment ledger counts:

- one dense symmetric stream through `D` active layers as `2D` f32 GEMM-equivalents (`K M K^T`);
- one local dense Jordan product per active layer as `D`;
- vector/GEMV/Hadamard work as less than a full GEMM.

For streams with individual lifetimes,

\[
\boxed{
G_{\rm total}
=2\sum_sD_s+\sum_mD_m+G_{\rm endpoint}
\le49.
}
\]

For `p` full-depth propagated streams and `j` full-depth Jordan products,

\[
G_{\rm total}=2Dp+Dj+G_{\rm endpoint}.
\]

### Fourteen layers

One fused quotient stream plus one local Jordan product costs

\[
2(14)+14=42,
\]

leaving at most seven endpoint equivalents. A separate reverse-H4 matrix adds 28 and gives 70, which fails.

### Ten layers

The fused base costs

\[
2(10)+10=30.
\]

A second full-lifetime reverse matrix adds 20, giving 50 before endpoint work; it also fails.

Therefore at most one independent full-lifetime dense matrix stream may carry the fused NCQC/q4 quotient. Sharing means a genuine algebraic common recurrence or fixed linear combination; batching two independent matrices in one library call does not reduce billed equivalents.

---

## 15. Raw-MSE sufficiency and channel context

For baseline residual `e` and the H4 scalar feature residualized against every deployed feature, `z_4^perp`, optimal calibration yields

\[
M_{\rm new}
=M_0-
\frac{\operatorname{Cov}(e,z_4^\perp)^2}
{\operatorname{Var}(z_4^\perp)}.
\]

Using `M0=3.2334958e-8` and oracle floor `M*=6.24925e-9`, raw below `1e-8` requires

\[
\rho^2>0.8562,
\qquad |\rho|>0.9253.
\]

The diffuse one-reset H2+H4 energy `0.96695` / correlation `0.98334` clears this local mathematical threshold, but it is not an end-to-end or incremental-H4 certificate. H2 is already partly represented, errors can align across depth, and the exact-state ladder indicates H4 is a crossing channel rather than the whole carrier:

\[
\text{exact variance+k3}:1.0286\times10^{-8},
\qquad
+k4:9.2547\times10^{-9}.
\]

The reverse H4 mechanism is therefore best interpreted as the final quartic crossing term paired with an accurate variance/H2 and signed-H3 carrier.

---

## 16. Graph-ready dispositions

### `H4_REVERSE_ONE_INSERTION`

- status: `theory_pass_conditional`;
- claim: exact one-stream accumulation for centered frozen second-order H4 births;
- state/cost: `O(n^2)` / `O(Ln^3)`;
- close if deployed reset is not scalar/common at the retained perturbative order.

### `H4_REVERSE_GENERIC_SELFCONSISTENT`

- status: `closed_proof_generic`;
- reason: target adjoint separation rank becomes `n`, causing `n^3/n^4` state/work.

### `NCQC_JORDAN_Q4_QUOTIENT`

- status: `conditional_open_high_priority`;
- claim: signed H3 plus q4(211)+2:1 square master in one `dim J=O(n)` module;
- required proof: every reset, feedback, collision correction, and endpoint factors through `J` and satisfies the leakage bound;
- cost gate: one fused dense matrix stream, `<=49` incremental f32 GEMM-equivalents.

### `NCQC_Q4_PAIR_KERNEL`

- status: `open_leakage_risk`;
- issue: any observable `ker J` component expands the terminal module to `O(n^2)` for the full core and restores `n^3/n^4` cost.

---

## 17. Current next questions

1. Does the matrix-polynomial two-bundle schedule provide exact conditional independence between the ReLU gate bundle and H4 bundle? If yes, `beta=0`; if not, compute its `delta=||Q_A||_2/sigma` and spectral diffuseness `chi` against the analytic bound.
2. Does the actual q4 endpoint `P4` factor through the Jordan map, or does it see a component of `ker J`?
3. Are all q4(211) collision subtractions performed at read time with the exact `31/22/4` inclusion-exclusion coefficients?
4. Can the quotient adjoint be algebraically fused into the single already-paid dense matrix stream, with no second full-lifetime conjugation?
5. Does the D11 cubic baseline admit the same fixed `O(n)` whitening/dictionary needed to evaluate the complete mixed square term without a new rank portfolio?
6. If fresh independence is only approximate, can a cavity/leave-one-out theorem prove `delta <= 2.8–4.1%` for diffuse cores (or the stricter sub-percent gate for concentrated spectra) uniformly over 10/14 late layers?

---

## 18. Checkpoint 4: fresh-cavity theorem, generic lower bound, and the D14+D10 two-bundle schedule

### Exact conditional-cavity theorem

Full probabilistic independence is stronger than necessary. Let `H_ell` be the sigma-algebra generated by the H4 bundle at a reset and let `G_ell` be the gate event. If

\[
P(G_\ell\mid\mathcal H_\ell)=p_{\ell i}
\]

is a scalar with respect to the latent H4 coordinates, then for every H4 polynomial `K` measurable with respect to `H_ell`,

\[
E[1_{G_\ell}K\mid\mathcal H_\ell]=p_{\ell i}K.
\]

Therefore the H4 reset is exactly scalar, descends through the Jordan quotient, and has

\[
\boxed{\beta_\ell^{\rm adj}=0.}
\]

A sufficient construction is a leave-one-out/fresh-weight cavity: the gate is measurable with respect to a fresh weight column while the H4 bundle is measurable with respect to the sigma-algebra excluding that column. This must hold at every reused layer; independence at initialization alone is not enough after coadapted propagation.

Frobenius orthogonality of two deterministic matrix words is not conditional independence. Likewise, the q4-perp commutator word and an H4 square are both even under global sign reversal, so parity does not kill their correlated-gate product.

### Explicit generic lower-bound witness

Embed the two-dimensional matrices

\[
A=\operatorname{diag}(1,-1),
\qquad
B=\begin{pmatrix}0&1\\1&0\end{pmatrix}
\]

in the `r=45..55` core. They satisfy `A circle B=0`, so the H4 tensor `U_{A,B}` lies in the discarded Jordan kernel. The exact product formula gives

\[
\Pi_4[Q_A\,\cdot]\Pi_4=d\Gamma(A).
\]

On the symmetric pair tensor,

\[
d\Gamma(A)(A\widetilde\otimes B)
=2I\widetilde\otimes B,
\]

because `A^2=I` and `AB+BA=0`. The input and output pair tensors have the same norm before the factor two. Thus, for the cavity gate `Phi(delta Q_A)` with `||Q_A||_2=1`,

\[
\boxed{
\beta^{\rm adj}
\ge2\phi(0)|\delta|+O(\delta^3)
=0.7979|\delta|+O(\delta^3).
}
\]

There is consequently no uniform small-leakage theorem over the full NCQC core without a spectral-diffuseness or cavity-correlation hypothesis. At the canonical finite-width scale `delta=1/sqrt(1024)=0.03125`, this witness already gives first-order leakage about `2.49%`, above the coherent D10 allowance `1.39%` and D14 allowance `0.994%`.

### Finite-width scale under diffuse spectra

The analytic upper bound from Section 13 is encouraging specifically for diffuse cores. At

\[
\delta=1/\sqrt n=0.03125,
\]

it gives approximately

- `r=45`, `chi=sqrt(2/45)`: `beta <= 1.13%`;
- `r=55`, `chi=sqrt(2/55)`: `beta <= 1.02%`.

Both pass the D10 coherent gate `1.39%`; neither has comfortable D14 margin against `0.994%`. Equivalently, writing `delta=c/sqrt(n)`, the sufficient D10 constants are approximately

- `c <= 1.20` at `r=45`;
- `c <= 1.30` at `r=55`.

The corresponding D14 constants are about `0.89` and `0.97`.

### Matrix-polynomial two-bundle schedule

The proposed schedule is:

\[
\mathcal U_\ell=\operatorname{span}\{a_\ell\}
\quad(\ell<5),
\]

\[
\mathcal U_\ell=\operatorname{span}\{a_\ell,b_\ell\}
\quad(\ell\ge5),
\]

where `a` is the D14 base commutator word and `b` is the D10 q4-perp suffix word. The exact reverse condition is

\[
T_\ell^*\mathcal U_{\ell+1}\subseteq\mathcal U_\ell
\]

together with every direct q4/endpoint seed lying in `U_ell`. At the L5-to-L4 boundary, both

\[
T_4^*a_5,
\qquad
T_4^*b_5
\]

must be collinear with `a_4`.

The lifted H4 support is

\[
\operatorname{Sym}^2\mathcal U_\ell,
\]

so it contains one feature `aa` early and exactly three features `aa,ab,bb` late. These lifted features must be consumed through the fixed pair/Jordan metric, not propagated as three new dense streams.

Let `P_ell` project onto `U_ell` and define the primitive word leakage

\[
\eta_\ell
=\|(I-P_\ell)T_\ell^*P_{\ell+1}\|.
\]

Expanding the symmetric square gives the exact bound

\[
\boxed{
\eta_\ell^{H4}
\le2\|T_\ell\|\eta_\ell+\eta_\ell^2.
}
\]

The total per-layer quotient leakage is therefore bounded by

\[
\boxed{
\beta_\ell^{\rm total}
\le
\beta_\ell^{\rm surface}
+2\|T_\ell\|\eta_\ell+\eta_\ell^2.
}
\]

For `||T_ell|| <= 1` and exact fresh-gate cancellation, the D10 gate permits

\[
\eta_\ell\le\sqrt{1+0.0139}-1\approx0.00693.
\]

With the canonical diffuse `delta=1/sqrt(n)` surface bounds already charged, the remaining primitive-word gates tighten to roughly

- `eta <= 0.00132` for `r=45`;
- `eta <= 0.00184` for `r=55`.

Thus the two-bundle Krylov closure must be essentially exact if it relies on the finite-width correlated-gate bound rather than a genuine cavity theorem.

### Two-bundle GEMM consequence

Under the deployment ledger:

- D14 base stream `a`: `2*14=28` GEMM-equivalents;
- D10 suffix stream `b`: `2*10=20` GEMM-equivalents.

The two propagated streams already cost

\[
28+20=48,
\]

leaving only one equivalent under the binding allowance 49. Therefore:

1. no independent dense H4 reverse stream is possible;
2. the three late lifted features `aa,ab,bb` must be locally consumed, not transported;
3. any per-layer q211/Jordan product must be fused into an already billed conjugation or implemented below one total GEMM-equivalent over the suffix;
4. a nominal two-stream schedule plus ten separate local dense products costs 58 and fails.

### Updated next question

The decisive unresolved premise is whether `a_ell,b_ell` are merely Frobenius-orthogonal deterministic words or arise from conditionally independent cavity bundles. In the former case, the exact closure/leakage certificate is

\[
\beta_\ell^{\rm surface}
+2\|T_\ell\|\eta_\ell+\eta_\ell^2
\le0.0139
\]

over the D10 suffix, plus the boundary collinearity condition. In the latter case, `beta_surface=0`, leaving a much more plausible `eta<=0.693%` algebraic gate.

---

## 19. Checkpoint 5: does dense white-box geometry imply the cavity/diffuseness premise?

### Answer

No label-free guarantee follows from “dense”, He-scaled, or frozen white-box geometry alone. Exact independence requires a common-kernel/fresh-sigma-algebra condition, and the best deterministic spectral bound available without inspecting the actual cores is too weak. Certification must therefore be either:

1. a symbolic common-kernel/intertwining proof; or
2. a per-layer, per-frozen-net endpoint-adjoint residual observable.

### Stable-rank form of the spectral bound

Define

\[
\operatorname{sr}(A)
=\frac{\|A\|_F^2}{\|A\|_{\rm op}^2}.
\]

Because `||Q_A||_2^2=||A||_F^2/2`,

\[
\chi=\frac{\|A\|_{\rm op}}{\|Q_A\|_2}
=\sqrt{\frac{2}{\operatorname{sr}(A)}}.
\]

The analytic leakage bound becomes

\[
\boxed{
\beta^{\rm adj}
\le
1.596\,\delta\sqrt{\frac{2}{s}}
+5.385\sqrt{
15+\frac{260}{s}+\frac{480}{s^2}
}\,\delta^3,
\qquad s=\operatorname{sr}(A).
}
\]

At the canonical scale `delta=1/sqrt(1024)=1/32`:

- the D10 coherent gate `beta<=0.0139` requires approximately
  \[
  \operatorname{sr}(A)\ge29;
  \]
- the D14 gate `beta<=0.00994` requires approximately
  \[
  \operatorname{sr}(A)\ge59,
  \]
  which is impossible for `r<=55` under this conservative theorem.

Even a generic GOE-like dense matrix has operator norm about twice its RMS spectral scale, suggesting stable rank of order `r/4`, not the nearly flat `29+` spectrum required here. This is only a heuristic comparison, but it shows why entrywise density is not the needed hypothesis.

### Dense does not imply diffuse: sharp construction

Let `u` have every entry nonzero and let

\[
A_\varepsilon=uu^\top+\varepsilon D
\]

with a generic dense symmetric `D`. After Frobenius normalization, `A_epsilon` is entrywise dense but

\[
\operatorname{sr}(A_\varepsilon)\to1,
\qquad
\chi\to\sqrt2
\]

as `epsilon -> 0`.

Equivalently, conjugate the explicit two-dimensional leakage witness by a generic dense orthogonal matrix `O`:

\[
A'=OAO^\top,
\qquad
B'=OBO^\top.
\]

Both matrices are dense, their spectra and anticommutation relation are unchanged, and the order-one kernel-to-Jordan leakage persists by orthogonal equivariance. Dense orthogonal external transports can preserve this witness through every layer with coherent endpoint alignment. Thus no deterministic dense-layer theorem can force the `<=1.39%` D10 gate.

### No label-free bound on `delta`

The cavity ratio is

\[
\delta_\ell^2
=\frac{\frac12\operatorname{tr}(A_\ell^2)}
{\sigma_{{\rm cav},\ell}^2}.
\]

He normalization controls total second moment, not the Schur/conditional cavity variance in the denominator. Dense contributions can align or cancel so that `sigma_cav` is arbitrarily small while all row/column norms remain correctly scaled. The expected `O(n^{-1/2})` finite-width law is distributional; it is not a uniform statement over frozen nets.

### Exact Gaussian conditional-independence theorem

Let

\[
L=h^\top Z,
\qquad
Q_A=\frac12(Z^\top AZ-\operatorname{tr}A),
\qquad Z\sim N(0,I).
\]

Their joint moment-generating function contains

\[
\exp\left[
\frac{t^2}{2}h^\top(I-sA)^{-1}h
\right].
\]

It factors into the product of the marginal MGFs for all small `s,t` iff

\[
h^\top(I-sA)^{-1}h=\|h\|^2
\]

for all small `s`. Expanding in `s` and using symmetry gives

\[
h^\top A^2h=\|Ah\|^2=0.
\]

Therefore

\[
\boxed{
h^\top Z\ \perp\ Q_A
\quad\Longleftrightarrow\quad
Ah=0.
}
\]

For a family of quadratic cores, independence of the gate from the entire generated quadratic sigma-algebra requires

\[
\boxed{
M_s h=0\quad\text{for every carried core }M_s.
}
\]

A full NCQC core has trivial common kernel generically. Hence an ordinary dense gate is not a fresh cavity. Across depth, the condition must be regenerated at every layer; it is destroyed by a generic coadapted ReLU reset unless the construction explicitly uses disjoint Gaussian bundles or leave-one-out weights.

### Minimal lawful observable

Spectra `chi`, cavity ratio `delta`, and Frobenius word angles are not sufficient because two operators with the same such scalars can be orthogonally rotated to align leakage with, or away from, the endpoint adjoint.

The minimal target-conditioned certificate is the single residual norm

\[
\boxed{
\zeta_\ell
=\frac{
\|P_{\ker J}\mathcal R_\ell^*\lambda_{\ell+1}\|_{H4}
}{\|\lambda_{\ell+1}\|_{H4}}.
}
\]

For the two-bundle schedule, the exact version is

\[
\boxed{
\zeta_\ell^{(2)}
=\frac{
\|(I-P_{\operatorname{Sym}^2\mathcal U_\ell})
\mathcal R_\ell^*
P_{\operatorname{Sym}^2\mathcal U_{\ell+1}}
\Lambda_{\ell+1}\|
}{\|\Lambda_{\ell+1}\|}.
}
\]

This uses only the frozen white-box weights, the proposed reset, and the known endpoint adjoint. It uses no truth residuals or target labels. If the reset is a sum of a fixed small primitive portfolio, the norm can be obtained from the Gram matrix of the primitive reverse images after quotient projection; no full H4 tensor or full operator SVD is required.

### NCQC q4-perp realization of the observable

Let `V=span{A_i}`, let `P` project onto the represented one-matrix/Jordan sector, and define

\[
R_{ij}=(I-P)(A_i\circ A_j).
\]

The exact quartic complement is

\[
Q^\perp_{ijkl}
=16\left[
\langle R_{ij},R_{kl}\rangle
+\langle R_{ik},R_{jl}\rangle
+\langle R_{il},R_{jk}\rangle
\right].
\]

For the external `211` slice,

\[
Q^\perp_{iijk}
=16\left[
\langle R_{ii},R_{jk}\rangle
+2\langle R_{ij},R_{ik}\rangle
\right].
\]

A rank-one downstream context `u` contracts it without constructing the full tensor:

\[
\boxed{
[M_uQ^\perp]_i
=16\left[
\langle R_{ii},R(u,u)\rangle
+2\|R(e_i,u)\|^2
\right],
}
\]

followed by centering over `i`. For a context family `U={u_f}`, the exact secant-slice norm is

\[
\boxed{
\|M_UD\|^2
=\sum_{i,f}
\langle D_i-\bar D,u_fu_f^\top\rangle^2.
}
\]

Using the actual endpoint-adjoint contexts in this formula gives a lawful implicit evaluation of the off-quotient residual norm `zeta_ell`.

### Binding frozen-net certification

No aggregate distributional statement can replace the following per-net/per-layer condition:

\[
\boxed{
\sum_{\ell\in\text{active}}
\zeta_{\ell,\nu}\sqrt{\tau_{\ell,\nu}}
\le0.02520
\quad\text{for every frozen net }\nu.
}
\]

Under the pessimistic repeated-tail D10 convention, the sufficient shorthand is

\[
\max_{\ell,\nu}\zeta_{\ell,\nu}\le0.0139.
\]

For D14 it is `<=0.00994`. Only an exact common-kernel/intertwining proof may replace this observable audit by `zeta=0` identically.

## 20. Checkpoint 6: tensor-free K31 Bellman action

### 20.1 Verdict and claim separation

A target-contracted K31 action removes the need to **store** a fourth-order tensor, but does not generally remove the fourth-order information requirement. It is exact from a finite matrix state when the actual layer law is affine-quadratic Markov, or when an independent oracle supplies the actual post-reset fourth-moment action and its needed pullback. Under a general correlated ReLU reset, the same `ker(J)` leakage reappears in reverse moment acquisition.

Three claims must remain distinct:

1. **Static scalar acquisition:** one K31 action at one fixed law is a scalar fourth-moment contraction and needs no materialized H4 tensor.
2. **Exact affine-quadratic recursion:** if the law remains exactly affine-quadratic in a common Gaussian latent, that scalar and its derivative close on vectors and symmetric matrices.
3. **General-law Bellman recursion:** outside that family, the K31 influence pullback contains directions outside the matrix/Jordan quotient. A tensor-free formula then needs an actual-moment oracle or an explicit residual certificate.

### 20.2 Normalize first: the two gauges

Let `Z~N(0,I_r)`. There are two quadratic-core conventions in the active lanes.

In the half-quadratic/Hermite gauge used in this lane,

\[
Q_A(Z)=\frac12(Z^\top A Z-\operatorname{tr}A),
\qquad
U=u^\top Z+Q_A(Z),
\qquad
V=v^\top Z+Q_B(Z).
\]

Then

\[
\boxed{
\kappa_4(U,U,U,V)
=3\operatorname{tr}(A^3B)
+3u^\top(AB+BA)u
+6u^\top A^2v.
}
\tag{20.1}
\]

Since `A,B` are symmetric, `u^T ABu=u^T BAu`, so the middle term is also `6u^T ABu`.

The direct/self lanes use the full-quadratic gauge

\[
q_A(Z)=Z^\top A Z-\operatorname{tr}A=2Q_A(Z),
\qquad
\widetilde U=u^\top Z+q_A(Z),
\qquad
\widetilde V=v^\top Z+q_B(Z).
\]

Multilinearity multiplies the four-quadratic term by `2^4=16` and each two-quadratic/two-linear term by `2^2=4`, hence

\[
\boxed{
\kappa_4(\widetilde U,\widetilde U,\widetilde U,\widetilde V)
=48\operatorname{tr}(A^3B)
+12u^\top(AB+BA)u
+24u^\top A^2v.
}
\tag{20.2}
\]

Equivalently, under the symmetric-`AB` scalar convention, this is

\[
48\operatorname{tr}(A^3B)
+24\bigl(u^\top ABu+u^\top A^2v\bigr).
\]

Thus `3 tr(A^3B)` and `48 tr(A^3B)` are the same K31 identity in different gauges.

### 20.3 Exact affine-quadratic action and reverse derivative

Equation (20.1) follows from the joint cumulant-generating function

\[
K(s,t)
=-\frac12\log\det(I-M(s,t))
-\frac12\operatorname{tr}M(s,t)
+\frac12 b(s,t)^\top(I-M(s,t))^{-1}b(s,t),
\]

with `M(s,t)=sA+tB` and `b(s,t)=su+tv`, by extracting the coefficient of `s^3t`. This uses the **actual affine-quadratic law**, not a Gaussian moment closure imposed afterward.

In the half-quadratic gauge set

\[
P=uu^\top,
\qquad
S=\operatorname{sym}(uv^\top)=\frac12(uv^\top+vu^\top).
\]

The exact symmetric-matrix reverse sensitivities are

\[
\boxed{
\begin{aligned}
\bar A={}&3(A^2B+ABA+BA^2)+3(BP+PB)+6(AS+SA),\\
\bar B={}&3A^3+3(AP+PA),\\
\bar u={}&6(AB+BA)u+6A^2v,\\
\bar v={}&6A^2u.
\end{aligned}}
\tag{20.3}
\]

For the full-quadratic gauge they are

\[
\boxed{
\begin{aligned}
\bar A={}&48(A^2B+ABA+BA^2)+12(BP+PB)+24(AS+SA),\\
\bar B={}&48A^3+12(AP+PA),\\
\bar u={}&24(AB+BA)u+24A^2v,\\
\bar v={}&24A^2u.
\end{aligned}}
\tag{20.4}
\]

For the pure-quadratic plane

\[
K^{31}_{ij}=c_{31}\operatorname{tr}(M_i^3M_j),
\qquad
c_{31}=3\ \text{(half gauge)},\quad c_{31}=48\ \text{(full gauge)},
\]

and target action `S_Lambda=sum_ij Lambda_ij K31_ij`, the core gradient at unit `p` is

\[
\boxed{
\begin{aligned}
\bar M_p={}&c_{31}\sum_j\Lambda_{pj}
(M_p^2M_j+M_pM_jM_p+M_jM_p^2)\\
&+c_{31}\sum_i\Lambda_{ip}M_i^3.
\end{aligned}}
\tag{20.5}
\]

Writing `N_p=sum_j Lambda_pj M_j` contracts the first row action before the three local products. The plane need not be materialized: target mixing and trace-Grams stay at the ordinary `O(n^3)` matrix-stream scale when `r^2=Theta(n)`; local `r x r` products cost `O(n r^3)=O(n^(5/2))`.

Across a dense affine layer

\[
u'_a=\sum_iW_{ia}u_i,
\qquad
A'_a=\sum_iW_{ia}A_i,
\]

the exact pullback is

\[
\boxed{
\bar u_i=\sum_aW_{ia}\bar u'_a,
\qquad
\bar A_i=\sum_aW_{ia}\bar A'_a.
}
\tag{20.6}
\]

If a reset truly is the quadratic matrix polynomial `F(A)=alpha A+beta A^2`, then

\[
DF_A[\dot A]=\alpha\dot A+\beta(A\dot A+\dot A A),
\qquad
\boxed{DF_A^*[G]=\alpha G+\beta(AG+GA).}
\tag{20.7}
\]

Equations (20.3)--(20.7) form a complete tensor-free Bellman recursion only if every relevant post-reset unit still has the stated affine-quadratic representation, or if some larger sufficient statistic gives the reset action exactly. A literal ReLU of a nondegenerate quadratic Gaussian is not generally affine-quadratic; that premise cannot be silently assigned to the deployed law.

### 20.4 Exact static acquisition does not require affine-quadratic closure

Affine-quadratic Markov exactness is stronger than needed to evaluate one scalar action at one already-formed layer. For any centered random vector `X` with finite fourth moments,

\[
\kappa_4(X_a,X_a,X_a,X_b)
=\mathbb E[X_a^3X_b]-3C_{aa}C_{ab}.
\]

Therefore

\[
\boxed{
S_\Lambda(X)
=\sum_{a,b}\Lambda_{ab}\kappa_4(X_a,X_a,X_a,X_b)
=\mathbb E\big[(X^{\odot3})^\top\Lambda X\big]
-3\sum_{a,b}\Lambda_{ab}C_{aa}C_{ab}.
}
\tag{20.8}
\]

This is an exact tensor-free acquisition formula for an arbitrary law. It needs only the requested scalar fourth-moment action, not all `n^4` entries. Two lawful bypasses therefore exist:

- directly observe or evaluate (20.8) under the actual post-reset law at every required layer; or
- possess an oracle for the exact scalar action and its exact directional pullback.

These are **new information sources**, not consequences of the lower-order matrix quotient. If the estimator has only that quotient and must infer the action recursively through reset, (20.8) does not make the missing information disappear.

A minimal static diagnostic is

\[
\boxed{
\rho^{31}_\Lambda
=S_\Lambda(X_{\rm actual})-S_\Lambda(X_{\rm AQ}).
}
\tag{20.9}
\]

It certifies the chosen scalar at that layer, but not how the scalar changes under upstream perturbations. Bellman use needs the directional residual below.

### 20.5 General-law reverse differentiation and directional residual

Let `U,V` be centered and put `a=E[U^2]`, `b=E[UV]`. For centered perturbations `H_U,H_V`, direct differentiation of

\[
\kappa_{31}(U,V)=\mathbb E[U^3V]-3\mathbb E[U^2]\mathbb E[UV]
\]

gives

\[
\boxed{
D\kappa_{31}[H_U,H_V]
=\mathbb E[H_U\psi_U(U,V)]
+\mathbb E[H_V\psi_V(U,V)],
}
\tag{20.10}
\]

where, up to additive constants irrelevant to centered perturbations,

\[
\boxed{
\begin{aligned}
\psi_U(U,V)&=3\bigl(U^2V-2bU-aV\bigr),\\
\psi_V(U,V)&=U^3-3aU.
\end{aligned}}
\tag{20.11}
\]

For plane action `S_Lambda`, the total influence assigned to unit `p` is

\[
\boxed{
\Psi_p
=\sum_j\Lambda_{pj}\psi_U(X_p,X_j)
+\sum_i\Lambda_{ip}\psi_V(X_i,X_p).
}
\tag{20.12}
\]

Let `P_AQ` be the orthogonal projector onto the actually represented affine-quadratic tangent space (`H1 direct-sum H2` in the common latent Gaussian description). Define

\[
\boxed{
r_p^{31}=(I-P_{AQ})\Psi_p,
\qquad
\zeta_{31}=\frac{\|r^{31}\|_2}{\|\Psi\|_2}.
}
\tag{20.13}
\]

For any omitted centered law perturbation `h_perp`, the missed first-order action is

\[
\boxed{
\delta S_\Lambda=\langle r^{31},h_\perp\rangle,
\qquad
|\delta S_\Lambda|\le\|r^{31}\|_2\|h_\perp\|_2.
}
\tag{20.14}
\]

Thus `rho31` is the minimal observable for a fixed scalar value, while `r31` or its norm `zeta31` is the minimal observable for the Bellman/directional claim.

There is a stronger degree warning here. Even when `U,V` themselves are pure H2 quadratics, `psi_U` contains chaos degrees through H6 (`U^2V` has polynomial degree six), while `psi_V` contains degrees through H6 as well. After the cumulant subtractions, the non-AQ residual generally contains both H4 and H6 components. The `ker(J)` H4 witness below is therefore a **minimal** obstruction, not an assertion that H4 is the only missing reverse sector.

### 20.6 A literal ReLU pullback exposes the same missing directions

For a centered reset

\[
X'_a=\phi(Y_a)-\mathbb E\phi(Y_a),
\qquad
Y_a=\sum_iW_{ia}X_i,
\]

and centered post-reset influence `Psi'_a`, the exact pathwise pullback is

\[
\boxed{
\Psi^{Y}_a=P_0\bigl[\phi'(Y_a)\Psi'_a\bigr],
\qquad
\Psi^{X}_i=\sum_aW_{ia}\Psi^{Y}_a,
}
\tag{20.15}
\]

where `P0 f=f-Ef`. For ReLU, `phi'(Y)=1_{Y>0}` almost surely. Even if `Psi'` initially lies in a low Hermite/Jordan quotient, multiplication by the **correlated** half-space gate generally creates components outside it. This is the surface-leakage operator of Sections 17--19, now applied to the K31 influence instead of to a forward H4 tensor.

Let `H4` be the fourth-chaos state, `J:H4->Q` the one-matrix Jordan contraction, and `N=ker J`. A terminal K31 action that factors through the quotient has adjoint `g31=J^* gbar31`. For reset operator `R`, its pre-reset pullback factors through the same quotient if and only if

\[
\boxed{
P_N R^*J^*\bar g_{31}=0.
}
\tag{20.16}
\]

If (20.16) fails, choose

\[
K=\frac{P_N R^*J^*\bar g_{31}}
        {\|P_N R^*J^*\bar g_{31}\|}\in\ker J.
\]

Two pre-reset laws differing only by `epsilon K` have identical quotient data, but their post-reset K31 actions differ at first order by

\[
\boxed{
\epsilon\langle J^*\bar g_{31},RK\rangle
=\epsilon\|P_N R^*J^*\bar g_{31}\|\ne0.
}
\tag{20.17}
\]

This is an information-theoretic obstruction: changing contraction order cannot recover a component annihilated by the available statistic.

A concrete `2 x 2` witness uses

\[
A=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad
B=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
\]

Here `A circle B=0`, so `A symtensor B` lies in the kernel of Jordan multiplication. The correlated-gate generator `dGamma(A)=L_A tensor I+I tensor L_A`, with `L_A X=A circle X`, sends this kernel vector to a nonzero multiple of `I symtensor B`, whose Jordan image is `B`. Since the generator is self-adjoint, a quotient adjoint in the `B` direction pulls back into the discarded `A symtensor B` direction. This is a lawful K31 adjoint: in (20.5), the derivative with respect to the single `V` core is proportional to `M_U^3`, and choosing `M_U=B` gives `B^3=B`.

Hence

\[
\boxed{
\text{tensor-free K31 changes storage and contraction order, not reset sufficiency.}
}
\tag{20.18}
\]

### 20.7 Exact leakage and endpoint-error budgets

For an exact affine-quadratic Markov law, `r31=0`, `zeta31=0`, and (20.3)--(20.7) have no H4 quotient leakage. Any remaining error is arithmetic or an explicitly introduced approximation error.

For a general law, define at active layer `ell`

\[
\beta^{31}_\ell
=\frac{\|P_N R_\ell^*\lambda^{31}_{\ell+1}\|}
       {\|\lambda^{31}_{\ell+1}\|},
\]

and let `tau31_ell V` be the omitted forward energy that can pair with this residual. The exact derivative-level Duhamel error is

\[
\Delta y_{31}
=\sum_\ell\left\langle
P_N R_\ell^*\lambda^{31}_{\ell+1},k_\ell
\right\rangle.
\]

This is also an exact finite error whenever the omitted H4 state enters the propagated action linearly. For a genuinely nonlinear finite law displacement `h`, it is the first variation and a rigorous finite bound must add the Taylor remainder

\[
\frac12\sup_{0\le t\le1}
\|D^2S(\mu+t h)\|_{\rm op}\,\|h\|_2^2.
\]

Thus the rigorous coherent bound is

\[
\boxed{
\|e_{31}\|_2
\le\sqrt V\sum_\ell\beta^{31}_\ell\sqrt{\tau^{31}_\ell}.
}
\tag{20.19}
\]

Using the strict endpoint-amplitude allowance

\[
\sqrt{10^{-8}}-\sqrt{9.2547\times10^{-9}}=3.80\times10^{-6},
\qquad
\sqrt V=1.5072\times10^{-4},
\]

the binding dimensionless condition is again

\[
\boxed{
\sum_\ell\beta^{31}_\ell\sqrt{\tau^{31}_\ell}\le0.02520.
}
\tag{20.20}
\]

Under the repeated-tail shorthand `tau31=.0328064`, the sufficient uniform gates remain

\[
\boxed{
\beta^{31}_\ell\le0.0139\quad(D10),
\qquad
\beta^{31}_\ell\le0.00994\quad(D14).
}
\tag{20.21}
\]

Only with proved endpoint orthogonality may one instead use

\[
\|e_{31}\|_2^2
\le V\sum_\ell(\beta^{31}_\ell)^2\tau^{31}_\ell
\]

and the looser energy slack `0.7453e-9` (RMS `2.73e-5`). Without orthogonality, that substitution is invalid.

One conditional marginal statement is useful. If variance and K3 are exact, all remaining error lies orthogonally inside the captured quartic marginal, and no other channel moves, then the move from `1.0286e-8` (variance+K3) toward `9.2547e-9` (variance+K3+K4 oracle) only needs

\[
\boxed{
\frac{1.0286\times10^{-8}-1.0\times10^{-8}}
     {1.0286\times10^{-8}-9.2547\times10^{-9}}
=0.2773.
}
\tag{20.22}
\]

So `27.7%` of the *entire orthogonal quartic marginal* would suffice in that idealized accounting. K31 is only one collision plane: (20.22) is not a claim that K31 captures 27.7%, and it cannot replace the adversarial leakage gate.

### 20.8 Decision table and next lawful question

| Mechanism | Full H4 stored? | Exact for general law? | Reset-leakage certificate? |
|---|---:|---:|---:|
| Half/full-gauge AQ trace formula | no | no; exact AQ law only | no if AQ premise is proved |
| Direct actual scalar moment (20.8) after reset | no | yes for that scalar | no for static value; oracle/residual for derivative |
| Quotient-recursive K31 Bellman action | no | only if (20.16) holds | yes |
| Full actual influence pullback (20.10)--(20.15) | not necessarily | yes | residual carried, not omitted |

The decisive question is not whether `tr(A^3B)` is tensor-free—it is. It is

\[
\boxed{
\text{Can deployed observables evaluate (20.8), or certify (20.16)/(20.20),
at every active layer and frozen net?}
}
\]

Absent one of those receipts, tensor-free K31 is an attractive implementation identity but not a proof of raw-MSE-valid fourth-order acquisition.

### 20.9 Graph-node split required by the result

The HTML graph must represent these as separate claims rather than one mixed node:

- `P2R0G19B_MATH_K31_GAUGE` — **solved**: half gauge `3` versus full gauge `48`, with exact conversion (20.1)--(20.2).
- `P2R0G19B_MATH_K31_AQ_ACTION` — **solved conditional identity**: exact trace action and reverse derivatives (20.3)--(20.7); premise is exact AQ Markov law.
- `P2R0G19B_MATH_K31_DIRECT_MOMENT` — **solved mathematical identity / deployment-open**: arbitrary-law scalar acquisition (20.8); open question is whether the observable is lawfully available at every active layer.
- `P2R0G19B_MATH_K31_RESET_LEAK` — **generic no-go, certification-open**: quotient-recursive action fails unless (20.16) holds; witness (20.17)--(20.18).
- `P2R0G19B_MATH_K31_ERROR_GATE` — **open frozen-net gate**: certify (20.20), shorthand `1.39%` D10 or `0.994%` D14 under the stated repeated-tail convention.

Edges: `K31_GAUGE -> K31_AQ_ACTION`; `K31_AQ_ACTION -> K31_DIRECT_MOMENT` as a special case; `K31_DIRECT_MOMENT -> K31_RESET_LEAK` only for recursive acquisition; `K31_RESET_LEAK -> K31_ERROR_GATE`; and both `K31_DIRECT_MOMENT` and `K31_ERROR_GATE` inform the parent `P2R0G19B_MATH_H4_REVERSE_RANK` decision.

## 21. Checkpoint 7: can diagonal q4 be differentiated into K31?

### Functional identity versus stored-value fallacy

Let the actual diagonal connected-fourth potential be the functional

\[
\Phi(U)=\sum_a\kappa_4(U_a,U_a,U_a,U_a).
\]

For a direction `V` on the same probability space, cumulant multilinearity gives the exact JVP

\[
\boxed{
D\Phi(U)[V]
=4\sum_a\kappa_4(U_a,U_a,U_a,V_a).
}
\tag{21.1}
\]

Taking `V=Lambda U`, namely `V_a=sum_b Lambda_ab U_b`, yields

\[
\boxed{
\frac14D\Phi(U)[\Lambda U]
=\sum_{a,b}\Lambda_{ab}\kappa_4(U_a,U_a,U_a,U_b),
}
\tag{21.2}
\]

which is exactly the target-contracted K31 plane. Thus a **lawfully evaluable diagonal-q4 functional with a JVP interface** is sufficient; it does not require the full plane.

However, carrying only the scalar/vector values `Phi_a=kappa4(U_a^4)` is not such an interface. Diagonal values contain marginal information only and do not determine their derivative in a cross-unit direction. A minimal counterexample takes two copies with identical non-Gaussian marginals. In one joint law set `U_2=U_1`; in another let `U_2` be an independent copy of `U_1`. Every diagonal fourth cumulant is identical, while

\[
\kappa_4(U_1,U_1,U_1,U_2)
=\kappa_4(U_1^4)
\quad\text{in the first law},
\qquad
=0
\quad\text{in the second}.
\]

Therefore the minimal missing observable is the directional cross-action

\[
\boxed{
g_a(V_a)=\kappa_4(U_a,U_a,U_a,V_a),
}
\tag{21.3}
\]

or equivalently a functional gradient that can be paired with the requested direction. Equations (20.10)--(20.15) describe its exact reverse influence.

### Exact black-box polarization alternative

There is a second lawful interface between a stored value and a full JVP. Suppose a diagonal quartic-form oracle can evaluate `Phi` on **arbitrary mixed signals**, not merely return the already stored value `Phi(U)`. Put

\[
d_1=\Phi(U+V)-\Phi(U-V),
\qquad
d_2=\Phi(U+2V)-\Phi(U-2V).
\]

For the symmetric fourth-cumulant form, expansion gives

\[
d_1=8\sum_a\left[\kappa_4(U_a^3,V_a)+\kappa_4(U_a,V_a^3)\right],
\]

\[
d_2=16\sum_a\kappa_4(U_a^3,V_a)
    +64\sum_a\kappa_4(U_a,V_a^3).
\]

Eliminating the unwanted reversed `13` action yields

\[
\boxed{
\sum_a\kappa_4(U_a,U_a,U_a,V_a)
=\frac{8d_1-d_2}{48},
\qquad
D\Phi(U)[V]=\frac{8d_1-d_2}{12}.
}
\tag{21.3a}
\]

Thus four real potential evaluations provide an exact arbitrary-law static K31 action without an explicit gradient. This still requires a callable actual-law potential oracle. If each arm were an independent D10 chain, four arms would be decisively more expensive than the one-bundle JVP; only strong sharing/fusion could make this deployment-relevant.

### AQ realization and D10 cost

In the pure full-quadratic gauge, if unit `a` has base core `B_a` and tangent core `D_a`,

\[
\boxed{
g_a(D_a)=48\operatorname{tr}(B_a^3D_a).
}
\tag{21.4}
\]

For `D=Lambda B`, this is the exact tensor-free action. No symmetrization factor is missing because the three repeated `B` slots make all four cyclic placements identical under trace. If `D` depends on `B`, the total reverse derivative is

\[
\boxed{
\bar B_p=48\left[
B_p^2D_p+B_pD_pB_p+D_pB_p^2
+\sum_a\Lambda_{ap}B_a^3
\right].
}
\tag{21.5}
\]

The last term is the pullback through `D=Lambda B` and must not be dropped.

The exact AQ state is one tangent bundle `(delta u_a,delta A_a)` beside the already carried base bundle, requiring `O(nr^2)=O(n^2)` memory. If born at layer 5 and active for ten transitions, an independent dense symmetric-matrix stream costs

\[
2D=20\quad\text{f32 GEMM-equivalents}.
\]

If every reset JVP also requires a new unfused local dense Jordan product, add `D=10`, for a total incremental cost of `30`. Under the current D14-base plus D10-suffix ledger (`28+20=48` out of `49`), the diagonal-potential route fits only if that local JVP is fused into an already paid product or is exposed natively by the reset/moment interface. Otherwise it exceeds the allowance. For a general ReLU law, the finite AQ tangent bundle is not exact and the residual/oracle gate of Section 20 remains binding.

### Finite-correction cross-word warning

Equation (21.4) is a directional derivative. If a finite core correction is written `B=R+Delta`, the full diagonal quartic potential contains every noncommuting cross class:

\[
\boxed{
\operatorname{tr}(R+\Delta)^4
=\operatorname{tr}R^4
+4\operatorname{tr}(R^3\Delta)
+4\operatorname{tr}(R^2\Delta^2)
+2\operatorname{tr}(R\Delta R\Delta)
+4\operatorname{tr}(R\Delta^3)
+\operatorname{tr}\Delta^4.
}
\tag{21.6}
\]

Retaining only the K31 linear word is lawful for a JVP/first-order Bellman action, not for an exact finite baseline-to-corrected potential unless the remaining cross words are included or rigorously bounded.

### Graph split

- `P2R0G19B_MATH_Q4_POTENTIAL_JVP` — **solved identity**: (21.1)--(21.2).
- `P2R0G19B_MATH_Q4_POTENTIAL_POLARIZE` — **solved identity / deployment-conditional**: four arbitrary-mixture potential calls recover the cross-action by (21.3a).
- `P2R0G19B_MATH_Q4_STORED_VALUE_NOGO` — **solved no-go**: diagonal scalar values do not identify (21.3).
- `P2R0G19B_MATH_Q4_JVP_DEPLOY` — **conditional/open**: D10 costs `20` only with fused/native local JVP, otherwise `30`; general-law residual remains uncertified.

Edges: `Q4_POTENTIAL_JVP -> K31_DIRECT_MOMENT`; `Q4_POTENTIAL_POLARIZE -> K31_DIRECT_MOMENT`; `Q4_STORED_VALUE_NOGO -> K31_RESET_LEAK`; all three inform `Q4_JVP_DEPLOY`, which informs `H4_REVERSE_RANK`.

## 22. Checkpoint 8: exact D11 duplicate/ownership audit

### Verdict

The D11 lineage already contains its entire closure-owned K31 action exactly once. Consequently, a new NCQC scalar

\[
48\sum_a\operatorname{tr}(B_a^3D_a)
\]

may replace, correct, or be differenced against the matched old K31 action; adding it wholesale is duplicate credit. This statement is exact for the closure's own plane and conditional only on synchronizing the proposed AQ cores to the same site, frame, gauge, centering, and off-diagonal convention.

### Exact site map in the deployed-lineage source

The relevant source is

`p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py`.

At each reset:

1. `_full_relu_reset` passes `pre_k31` into `_edgeworth_weights`.
2. `_edgeworth_weights` installs `pre_k31.T/6` and `pre_k31/6` as the `H13` and `H31` Edgeworth channels. These channels are **consumers** of K31, not additional K31 sources.
3. `_four_pair_edgeworth` produces the reset pair moments, and `_connected_planes` forms `k31_target`; it performs all mean-centering counterterms and subtracts the disconnected cumulant pairing `3 v_i C_ij` exactly once before zeroing the off-diagonal convention's diagonal.

After that reset, the complete closure plane is partitioned as

\[
\boxed{
K_{31}^{old}
=K_{31}^{CP4}
+K_{31}^{star}
+3\alpha\,\operatorname{offdiag}(v[:,None]C).
}
\tag{22.1}
\]

The code realizes (22.1) literally:

- `_quartic_residual` subtracts the active CP4 provider plane once;
- `_reset_stars` fits/subtracts the connected pairing ansatz and stores the remainder as the quartic coordinate star;
- `_push_quartic` reconstructs the transported star plus connected pairing;
- `_cp4_read_planes` reconstructs the transported CP4 contribution; and
- `_assemble_quartic_provider` adds those mutually exclusive pieces once at the next pre-reset site.

Thus `k31_target` is the total target, not a fourth provider beside CP4/star/pairing. The fitted `3 alpha v C` term is a connected provider, not the already removed disconnected cumulant counterterm. CP3 affects the reset moments through K21/H3 and is consequently already folded into `k31_target`; it is not another post-hoc K31 feature.

### Same-site novelty theorem

Let `Lambda_l` be the exact local adjoint on the K31 plane, including both H31 and H13 consumers. The old scalar is

\[
\chi_l^{old}=\langle\Lambda_l,K_{31,l}^{old}\rangle_F.
\]

Let the actual full centered contexts define

\[
\chi_l^{full}
=\sum_a\left\{
\mathbb E[U_{la}^3V_{la}]
-3\mathbb E[U_{la}^2]\mathbb E[U_{la}V_{la}]
\right\}.
\]

Then the only complete local K31 novelty is

\[
\boxed{
\Delta\chi_l=\chi_l^{full}-\chi_l^{old}.
}
\tag{22.2}
\]

The synchronized baseline identity required to call this a pure new-stream correction is

\[
\boxed{
\kappa_4(U_l^D,U_l^D,U_l^D,V_l^D)
=\langle\Lambda_l,K_{31,l}^{old}\rangle_F.
}
\tag{22.3}
\]

For the literal D11 mapping, the K31 mask in (22.3) must be off-diagonal:

\[
\Lambda_l\leftarrow\operatorname{offdiag}(\Lambda_l).
\]

The code zeroes the diagonal of `k31_target` and owns that collision in the separate K4 vector, whose Edgeworth denominator is `24` rather than the K31/H31 denominator `6`. Therefore a trace action `48 sum_a tr(B_a^3 D_a)` with `D=Lambda B` duplicates/misweights diagonal K4 unless `diag(Lambda)=0`, or unless the diagonal is explicitly split and contracted with the separate K4 consumer.

Time indexing is equally binding. Cores formed after the dense `W` push and before ReLU must be compared with the assembled `pre_k31` immediately before `_full_relu_reset`; post-reset cores must instead be compared with `k31_target` and the post-reset costate. Contracting pre-reset NCQC cores against post-reset `k31_target` is not a novelty subtraction—it is a site mismatch.

If (22.3) holds and no residual stream is added, then (22.2) is identically zero. After any old-feature Schur projection it remains zero. Thus the synchronized AQ trace action has exactly **zero novel value**, irrespective of how cheaply it can be acquired. Its acquisition cost is the cost of replaying an old action, not the cost of a candidate improvement.

If (22.3) is unproved, keep the two effects separate:

\[
\boxed{
\Delta\chi_l
=\bigl[\chi_l(D+R)-\chi_l(D)\bigr]
+\bigl[\chi_l(D)-\chi_l^{old}\bigr].
}
\tag{22.4}
\]

The first bracket is new mixed-stream physics. The second is a zeroth-order AQ/closure synchronization defect; it may be valuable, but it is not evidence of a new q4 sector.

For full-quadratic cores, the first bracket is

\[
\boxed{
48\sum_a\left\{
\operatorname{tr}\left[(B_{D,a}+B_{R,a})^3(D_{D,a}+D_{R,a})\right]
-\operatorname{tr}(B_{D,a}^3D_{D,a})
\right\}.
}
\tag{22.5}
\]

Equation (22.5) contains every nonempty assignment of residual `R` to the four cumulant slots. An `R`-only term keeps just one of the fifteen polarized mixed terms and is not the residual.

### What can still be genuinely new

After subtracting (22.2), only three mathematically distinct residual classes remain:

1. **Beyond-AQ action:**

   \[
   \rho_l^{31}
   =\mathbb E[(Y_l^{\odot3})^\top Q_{31,l}Y_l]
   -\left[\text{AQ trace/Gram action at the same site}\right].
   \]

   Lower moments and diagonal K4 do not identify this crossed moment.

   Equivalently, the only countable same-site K31 correction is

   \[
   \boxed{
   \rho_l^{31}
   =\langle Q_l^{31},
   D_{l,true}^{31}-D_{l,assembled}^{31}\rangle_F.
   }
   \tag{22.5a}
   \]

   A small norm of this residual does not imply a low separation rank; amplitude and acquisition complexity are independent gates.

2. **Synchronization defect:** the second bracket in (22.4), recorded explicitly as AQ-versus-old closure mismatch rather than q4 novelty.

3. **Future-visible old-interface kernel:** let `O_l` denote the complete existing D11 H4 interface at the site, including its CP4 carrier state and its K4/K31/K22 star/pair readouts. A direction `h_l` may satisfy `O_l h_l=0` now yet become visible after later resets. Its exact missing-information criterion is

   \[
   \boxed{
   P_{\ker O_l}R_{m:l}^*\lambda_m\ne0
   }
   \tag{22.6}
   \]

   for some later consumer `m`. The larger `ker(C31_l)` is a useful K31-specific screening space, but a direction visible to the old K4, K22, or CP4 interface is not new merely because current K31 misses it. This is the same reverse quotient-leakage mechanism: a zero same-site K31 residual is not a proof of zero future value, and full-interface residualization is required before credit.

Only the components of these residuals that survive the endpoint adjoint and the final old-feature Schur projection can receive value credit.

### Exact double-counting rules

- Do not add `k31_target`, CP4 K31, star K31, and pairing K31 as separate features; the last three partition the first.
- Do not add H31/H13 response as a K31 provider; it is the consumer already defining `Lambda_l`.
- Do not subtract `3 E[U^2]E[UV]` after contracting the already connected old K31 plane.
- Do not subtract CP3 separately after subtracting total `K31_old`; CP3's reset effect is already inside that target.
- Do not credit a same-site `q_parallel/q_perp` attribution on top of (22.2); it repartitions the same scalar.
- Do not infer absence of future-visible H4/H6 information from equality of the current K31 action; apply (22.6).

### Graph split

- `P2R0G19B_MATH_D11_K31_OWNERSHIP` — **solved**: exact partition (22.1), H31 consumer ownership, and no fourth provider.
- `P2R0G19B_MATH_D11_K31_SYNC` — **conditional/open**: prove same-site identity (22.3) in the proposed core gauge.
- `P2R0G19B_MATH_D11_K31_NOVELTY` — **synchronized AQ action closed as duplicate; residual value-open**: grade only the true-minus-assembled residual (22.5a), or mixed correction (22.5), after old-span projection.
- `P2R0G19B_MATH_D11_K31_FUTURE_KERNEL` — **open**: certify or carry directions satisfying (22.6).

Edges: `D11_K31_OWNERSHIP -> D11_K31_SYNC -> D11_K31_NOVELTY`; `K31_BEYOND_AQ_ACTION -> D11_K31_NOVELTY`; `D11_K31_OWNERSHIP -> D11_K31_FUTURE_KERNEL`; both novelty and future-kernel nodes inform `H4_REVERSE_RANK`.

## 23. Polarization call-minimality and error conditioning

Let

\[
f(t)=\Phi(U+tV)
=a_0+4xt+6zt^2+4yt^3+a_4t^4,
\]

where

\[
x=\sum_a\kappa_4(U_a,U_a,U_a,V_a),
\quad
z=\sum_a\kappa_4(U_a,U_a,V_a,V_a),
\quad
y=\sum_a\kappa_4(U_a,V_a,V_a,V_a).
\]

### Four calls are necessary and sufficient without derivative data

For any nonzero real scale `h`, define `d_h=f(h)-f(-h)` and `d_2h=f(2h)-f(-2h)`. Then

\[
\boxed{
x=\frac{8d_h-d_{2h}}{48h}.
}
\tag{23.1}
\]

This uses the four calls `f(+h),f(-h),f(+2h),f(-2h)` and cancels every even coefficient plus the unwanted cubic coefficient `y`.

For a normalized query interval `|t|<=1`, write `Delta_t=f(t)-f(-t)` and use an inner scale `0<t<1` plus the outer scale `1`. The equivalent formula is

\[
\boxed{
x=\frac{\Delta_t-t^3\Delta_1}{8t(1-t^2)}.
}
\tag{23.1a}
\]

At the sharp symmetric choice `t=1/2`,

\[
\boxed{
x=\frac{f(1/2)-f(-1/2)}{3}
-\frac{f(1)-f(-1)}{24}.
}
\tag{23.1b}
\]

Four is minimal among real point-evaluation schemes that must work for every quartic and have no derivative or coefficient side information. To see this, suppose three distinct nodes `t1,t2,t3` sufficed. Let

\[
\omega(t)=\prod_{j=1}^3(t-t_j).
\]

Every polynomial `omega(t)q(t)` with `deg q<=1` vanishes at all three queried points, so exact recovery of `f'(0)` would require

\[
\omega'(0)=0,
\qquad
(t\omega(t))'|_{t=0}=\omega(0)=0.
\]

The second condition makes zero one of the nodes; with distinct nodes the first derivative there is the nonzero product of the other two roots. Contradiction. Repeated point calls add no information. The symmetric four-node stencil (23.1) attains the lower bound.

### Side information can reduce four calls to three, but not two

If both `a0=Phi(U)` and `a4=Phi(V)` are already known exactly, three additional calls at `t=-1,1,2` suffice:

\[
\boxed{
x=\frac{6f(1)-2f(-1)-f(2)-3a_0+12a_4}{24}.
}
\tag{23.2}
\]

Two calls cannot recover `x` on the remaining span `{t,t^2,t^3}`. Indeed, weights on two nonzero nodes would have to isolate the linear coefficient while annihilating both quadratic and cubic coefficients:

\[
w_1t_1^2+w_2t_2^2=0,
\qquad
w_1t_1^3+w_2t_2^3=0.
\]

For nonzero distinct nodes these equations force incompatible weight ratios. Knowing only `a0` or only `a4` leaves four unknown coefficients and does not generically reduce the four-call requirement.

Keeping every new call inside `[-1,1]`, a three-call form using `f(t),f(1),f(-1)` is

\[
\boxed{
\begin{aligned}
x={}&\frac{f(t)}{4t(1-t^2)}
-\frac{t f(1)}{8(1-t)}
-\frac{t f(-1)}{8(1+t)}\\
&-\frac{\Phi(U)}{4t}
+\frac{t\Phi(V)}4.
\end{aligned}}
\tag{23.2a}
\]

This is useful only when both baseline values are truly available in the same oracle and normalization.

The globally optimal bounded three-new-call stencil is not restricted to the node set `{t,+1,-1}`. With outer scale `h` and both `Phi(U),Phi(V)` cached exactly,

\[
\boxed{
x=\frac{1}{4h}\left[
\frac{f(-h)}3-2f(-h/2)+\frac{2f(h/2)}3
+\Phi(U)-\frac{h^4\Phi(V)}4
\right].
}
\tag{23.2b}
\]

Its new-call adversarial amplification is the sharp `3/(4|h|)`, equal to the four-call optimum at normalized `h=1`. Cached `Phi(U)` alone does not reduce four calls. Cached `Phi(V)` alone permits three, with sharp amplification `(7 sqrt(7)+10)/(36|h|)=0.79225.../|h|` under the corresponding optimal stencil.

An exact AD/JVP interface is categorically different: it returns `f'(0)=4x` in one differentiated evaluation. AD of an approximate Q24/CP4 potential, however, differentiates that approximate potential; it does not turn it into the true-law functional.

### Sharp deterministic and independent-noise amplification

Let the four oracle evaluations in (23.1) have errors `e_h^+,e_h^-,e_2h^+,e_2h^-`. The induced error is exactly

\[
\boxed{
\delta x
=\frac{8(e_h^+-e_h^-)-(e_{2h}^+-e_{2h}^-)}{48h}.
}
\tag{23.3}
\]

If every call has absolute error at most `epsilon`,

\[
\boxed{
|\delta x|\le\frac{3}{8|h|}\epsilon.
}
\tag{23.4}
\]

This constant is the exact `l1` norm of the stencil. If the four errors are independent, centered, and each has variance at most `sigma_e^2`,

\[
\boxed{
\operatorname{sd}(\delta x)
\le\sqrt{\frac{65}{1152}}\,\frac{\sigma_e}{|h|}.
}
\tag{23.5}
\]

The apparent improvement of (23.4) as `h` grows assumes a uniform absolute error even for quartically larger query signals. Under the binding normalized domain `|t|<=1`, the adversarial amplification for (23.1a) is

\[
\boxed{
A_4(t)=\frac{1/t+t^2}{4(1-t^2)}.
}
\tag{23.5a}
\]

Its global minimum is the sharp value

\[
\boxed{
A_4(1/2)=\frac34,
}
\tag{23.5b}
\]

and a Chebyshev-`T3` extremal lower bound shows that no four real bounded-domain calls improve this worst-case constant. For independent equal-variance errors, (23.1b) has RMS amplification

\[
\boxed{
\sqrt{65/288}=0.475\ldots.
}
\tag{23.5c}
\]

For the **restricted-node** three-new-call formula (23.2a), the new-call adversarial factor is

\[
A_3(t)=\frac{1+t^2}{4t(1-t^2)},
\]

minimized at `t^2=sqrt(5)-2`, with `A3 approximately 0.832`. This is not the global three-node optimum; (23.2b) reaches `0.75`. Errors in the supposedly known baselines add separately as

\[
\frac{|e_U|}{4t}+\frac{t|e_V|}{4}.
\]

For the JVP `4x`, multiply (23.4) by four, giving `3 epsilon/(2|h|)`. Larger `h` reduces amplification under a uniform **absolute** error model, but the queried signals `U plus/minus 2hV` have quartically growing scale; under relative or norm-scaled oracle error, `h` must be optimized against that growth rather than sent to infinity.

Across layers, the strict coherent endpoint requirement is the weighted sum of (23.3), not a root-sum-square unless orthogonality is proved. If `x_l` is already expressed in endpoint-output units, a sufficient condition is

\[
\boxed{
\sum_l
\frac{8(|e_{h,l}^+|+|e_{h,l}^-|)
+|e_{2h,l}^+|+|e_{2h,l}^-|}
{48|h_l|}
\le3.80\times10^{-6}.
}
\tag{23.6}
\]

If normalized by `sqrt(V)=1.5072e-4`, the corresponding dimensionless right side is `0.02520`.

For one aggregate endpoint action under the optimized normalized stencils, the strict `3.80e-6` allowance implies the simple acquisition gates

\[
\boxed{
\epsilon_{4call}\le5.067\times10^{-6},
\qquad
\epsilon_{3call,both\ cached}\le5.067\times10^{-6},
\qquad
\epsilon_{3call,\Phi(V)\ cached}\lesssim4.80\times10^{-6},
\qquad
\epsilon_{JVP}\le1.52\times10^{-5}.
}
\tag{23.6a}
\]

The three-call numbers exclude cached-baseline errors, and `epsilon_JVP` is error on `D Phi`, whose contribution to `x` is divided by four. Under a pessimistic equal D10 coherent allocation, divide each right side by ten.

If the oracle evaluates the aggregate potential `F(W)=sum_a Phi(w_a)` for a whole mixed field, the `4/3/1` counts apply to the full target action. If it exposes only one scalar marginal per column, the call count multiplies by the number of requested columns; “four calls” is not automatically four full-layer scalar queries.

### What polarization does and does not bypass

- Equations (23.1)--(23.2) rely on `Phi` being a homogeneous connected-fourth cumulant functional. Applying the stencil to a generic nonlinear response `Phi+R` returns the desired quartic coefficient plus the corresponding finite-difference contamination from `R`; it is not an exact K31 extractor unless that remainder vanishes or is separately bounded.
- If `Phi` is an exact actual-law oracle on arbitrary mixed signals, (23.1) **measures** the complete static K31 action, including all H4/H6 and higher-law effects that contribute to that scalar. It does not make those sectors disappear, but there is no beyond-AQ residual for the measured scalar itself.
- If `Phi` is the D11/Q24/CP4 diagonal potential, (23.1) recovers only its existing closure-owned action and is duplicate after (22.2).
- Even a true-law result must be differenced against the assembled old K31 action before receiving novelty credit.
- Even an exact static oracle does not propagate directions in the full old-interface kernel that become visible only after later resets; the future-observability gate (22.6) remains separate.

### Graph split

- `P2R0G19B_MATH_Q4_POLARIZE_MINCALL` — **solved**: four calls are necessary/sufficient without side data; three with exact `Phi(U),Phi(V)`.
- `P2R0G19B_MATH_Q4_POLARIZE_ERROR` — **solved formula / certification-open**: exact stencil error (23.3), strict endpoint condition (23.6).
- `P2R0G19B_MATH_Q4_POLARIZE_ORACLE_SCOPE` — **conditional**: true-law oracle removes static beyond-AQ error; closure oracle only reproduces old K31.

Edges: `Q4_POLARIZE_MINCALL -> Q4_POLARIZE_ERROR`; `Q4_POLARIZE_ORACLE_SCOPE -> D11_K31_NOVELTY`; both inform `Q4_JVP_DEPLOY`.

## 24. Full-interface future-observability theorem

Let `H_l` be the full relevant law/chaos state at layer `l`, let

\[
O_l:H_l\to S_l^{old}
\]

be the **complete** old D11 interface (carried CP4 state plus every K4/K31/K22 star/pair readout), and let `R_l:H_l->H_{l+1}` be the exact linearized reset/transport. Let `lambda_m` denote any later scalar consumer.

### Necessary and sufficient exact-sufficiency conditions

The old state is future-sufficient if and only if there exist reduced operators `Rbar_l` and reduced consumers `lambdabar_l` such that

\[
\boxed{
O_{l+1}R_l=\bar R_l O_l,
\qquad
\lambda_l=O_l^*\bar\lambda_l
}
\tag{24.1}
\]

at every active site. The operator condition is equivalently

\[
\boxed{
R_l\ker O_l\subseteq\ker O_{l+1}.
}
\tag{24.2}
\]

Necessity follows because two states in the same fiber of `O_l` must remain in the same fiber after reset. Sufficiency follows by defining `Rbar_l` on `ran(O_l)` via `Rbar_l(O_lh)=O_{l+1}R_lh`; (24.2) makes this well-defined. Consumer factorization then makes every future scalar a function of the old state alone.

Equations (24.1)--(24.2) are exact for the linearized Bellman/tangent system. For a finite nonlinear reset, replace kernel invariance by the stronger fiber congruence `O_l(x)=O_l(y) => O_{l+1}(F_lx)=O_{l+1}(F_ly)`; differentiating that condition recovers (24.2).

If either condition fails, there is an exact two-law counterexample. Choose `h in ker O_l` with either `O_{l+1}R_lh !=0` or `lambda_m(R_{m:l}h)!=0`. Then states `x` and `x+epsilon h` have identical current old interface but distinct future old state or endpoint scalar.

The smallest abstract witness is two-dimensional:

\[
O=(1\ \ 0),
\qquad
h=e_2,
\qquad
Rh=e_1,
\qquad
\lambda=e_1.
\]

It has `Oh=0` but `<lambda,Rh>=1`. The earlier ReLU/Jordan witness `A symtensor B -> I symtensor B` is its concrete correlated-gate realization whenever the hidden input direction is outside the complete CP4/star/pair old span. Checking that old-span premise is essential; the K31 kernel alone is not enough.

### Minimal target-conditioned augmentation

For a fixed endpoint portfolio, define the reverse residual vectors

\[
g_{l,m}=P_{\ker O_l}R_{m:l}^*\lambda_m,
\qquad m\ge l.
\]

The minimal missing target-conditioned state dimension at layer `l` is

\[
\boxed{
t_l=\dim\operatorname{span}\{g_{l,m}:m\ge l\},
}
\tag{24.3}
\]

equivalently the rank of their Gram matrix. This is the residual cyclic/observability module. A scalar final output does not imply `t_l=1`: distinct birth times and reset words can give linearly independent reverse images.

If only one terminal scalar at one fixed horizon is queried, one reverse covector per site is algebraically sufficient **provided the local pairing with every hidden birth can be acquired exactly**. That proviso is precisely the actual-moment/JVP oracle requirement of Sections 20--23; without it, reverse contraction does not create the missing local datum.

### Deployment gate under the current ledger

After the D14 base stream plus D10 suffix stream use `28+20=48` of the `49` f32 GEMM-equivalent allowance, any new independent D10 symmetric-matrix stream costs approximately another `20`, not the remaining `1`. Hence a nonzero module (24.3) is deployable only if its local action is exposed by an exact native oracle, lies in the already paid two-bundle span, or fuses into an already billed contraction. Generic independent future-visible directions are cost-closed even when `t_l=1`.

### Graph split

- `P2R0G19B_MATH_D11_INTERFACE_INTERTWINER` — **theorem solved / premise-open**: certify (24.1)--(24.2) for actual D11 reset maps.
- `P2R0G19B_MATH_D11_RESIDUAL_MODULE` — **solved characterization / rank-open**: minimal dimension is (24.3).
- `P2R0G19B_MATH_D11_RESIDUAL_COST` — **conditional no-go**: any independent D10 matrix stream exceeds the one-GEMM slack; native/fused scalar acquisition is the only current loophole.

Edges: `D11_K31_OWNERSHIP -> D11_INTERFACE_INTERTWINER`; failure of the intertwiner opens `D11_RESIDUAL_MODULE -> D11_RESIDUAL_COST`; all inform `D11_K31_FUTURE_KERNEL` and `H4_REVERSE_RANK`.

### 24.1 Strong stacked-future form and exact augmentation criterion

Let `N_l=ker O_l` and `Phi_{m:l}=R_{m-1}...R_l`. Stack every future old-interface read and every terminal action into

\[
F_l h
=\left(
\{O_m\Phi_{m:l}h\}_{m>l},
\{\langle\lambda_q,\Phi_{L:l}h\rangle\}_q
\right).
\]

Then the exact missing primal quotient is

\[
\boxed{
N_l/(N_l\cap\ker F_l),
}
\tag{24.4}
\]

and its dual cyclic module is

\[
\boxed{
\mathcal C_l
=\operatorname{range}(P_{N_l}F_l^*)
=\operatorname{span}_{m>l}
P_{N_l}\Phi_{m:l}^*\operatorname{range}(O_m^*)
+\text{terminal reverse rows}.
}
\tag{24.5}
\]

In particular

\[
\dim\mathcal C_l=\operatorname{rank}(F_l|_{N_l}),
\]

which is the sharpened version of (24.3).

Suppose an added observable is `A_l`. It makes the augmented state exactly future-sufficient if and only if

\[
\boxed{
\ker\begin{bmatrix}O_l\\A_l\end{bmatrix}
\subseteq\ker F_l
\quad\Longleftrightarrow\quad
\mathcal C_l\subseteq
\operatorname{range}(P_{N_l}A_l^*).
}
\tag{24.6}
\]

This gives a precise reason the same-site K31 duplicate cannot help: its row already belongs to `range(O_l^*)`, so its projection onto `N_l` is zero. Re-adding that oracle does not shrink (24.4) or cover any vector in (24.5).

For a hidden-state radius `rho_l`, the sharp remaining worst-case future miss after adding `A_l` is

\[
\boxed{
\eta_l
=\rho_l\left\|
(I-P_{\operatorname{range}(P_{N_l}A_l^*)})
P_{N_l}F_l^*
\right\|_{op}.
}
\tag{24.7}
\]

This is the lawful operator-norm observable to compare with the `3.80e-6` coherent endpoint budget.

### 24.2 Smallest collision-sector witness

The abstract two-state witness has a concrete external-index analogue already at `n=3`. Consider the symmetric quartic represented by the homogeneous polynomial

\[
h(x)=x_1^2x_2x_3.
\]

Every coordinate collision read of types `(4)`, `(31)`, and `(22)` is zero: the monomial has multiplicity pattern `(211)`. But for the dense direction

\[
w=(1,1,1)/\sqrt3,
\qquad
h(w)=1/9\ne0.
\]

Thus a dense next-layer mix turns a birth-invisible `(211)` direction into a visible diagonal K4 direction. To be a full D11 counterexample rather than only a collision-read counterexample, the chosen component must additionally lie outside the active CP4 carried span. The CP4 condition cannot be omitted, but whenever such a kernel component exists the future-visibility claim is exact.

Rank-nullity guarantees such CP4-dark components at the campaign width even under a very conservative CP4 tangent allowance. The collision-dark space is

\[
\mathcal V_{dark}=H_{211}\oplus H_{1111},
\]

with dimension

\[
\boxed{
D=n\binom{n-1}{2}+\binom n4.
}
\tag{24.7a}
\]

All coordinate K4/K31/K22 and coordinate-star/pair reads vanish identically on this space. A rank-`r4` symmetric CP4 factorization has factor-tangent rank at most `n r4` (amplitude is included in the radial factor direction), so

\[
\boxed{
\dim(\mathcal V_{dark}\cap\ker O_{CP4})
\ge D-nr_4.
}
\tag{24.7b}
\]

For `n=1024,r4=2048`,

\[
\dim H_{211}=535,299,072,
\qquad
\dim H_{1111}=45,545,029,376,
\]

and therefore at least

\[
\boxed{46,078,231,296}
\]

collision-and-CP4-dark directions remain. The exact positive-dimension width gate is

\[
\boxed{
\frac{(n-1)(n-2)(n+9)}{24}>r_4.
}
\tag{24.7c}
\]

At `r4=2048` it first holds at `n=35`; the `H211` sector alone guarantees a kernel from `n=66`, and `H1111` alone from `n=39`.

For any nonzero tensor `h` in this intersection, `p_h(w)=h[w,w,w,w]` is a nonzero homogeneous polynomial. Its zero set is proper, so a generic dense next-layer column exposes it in future K4. The remaining physical/value premise is that the ensuing ReLU fourth-response coefficient and endpoint Bellman covector are nonzero. If a future lineage adds a true all-distinct interface of rank `r_extra`, replace `n r4` by `n r4+r_extra` in (24.7b); coordinate collision star/pair storage itself adds no such rank.

This is an **ambient external-tensor** existence theorem. It does not by dimension alone prove that the lawful r45--55 NCQC/AQ image, or its 2099-dimensional r55 inverse-fiber tangent, contains one of these dark directions. That provider-intersection question is the separate restricted-rank gate (25.7).

### 24.3 Sharpened cost cut

Let `C_acq` be the acquisition cost and let each cut charge at least one equivalent per independent residual mode crossing it. A generic lower bound is

\[
\boxed{
C_{new}\ge C_{acq}+\sum_{\text{active cuts }c}\dim\mathcal C_c.
}
\tag{24.8}
\]

With `48/49` already allocated, any new mode transported across two or more cuts is closed even under this optimistic one-per-cut bound; a symmetric matrix stream is still worse (the earlier D10 census is `20`). The only remaining loophole is a direct complete-future-costate residual oracle evaluated locally for at most the one-equivalent slack, followed by the independent value gate of at least `2.86e-10` Schur gain.

## 25. Checkpoint 9: specialize the future-completion quotient

Three proposed completions have different state types and ownership. They must not be merged under “future H4.”

### 25.1 Transient connected `Z22`

At a post-reset site define

\[
\boxed{
Z=\operatorname{offdiag}\operatorname{sym}
\left\{K22^{target}-K22^{CP4}
-\alpha\left[vv^\top+2C\odot C\right]\right\}.
}
\tag{25.1}
\]

`K22_target` is already connected, so the Gaussian Wick disconnection is not subtracted again. The `alpha` term is a fitted connected old provider. The minimal pure-`22` lift places `Z_pq` on all six permutations of `(p,p,q,q)` and zero elsewhere.

For one next dense map with columns `w_a`, put `W2=W odot W`. Exact enumeration gives

\[
\boxed{
\delta K31_{ab}=3(w_a^{\odot2})^TZ(w_a\odot w_b),
\qquad
\delta k4_a=3(w_a^{\odot2})^TZw_a^{\odot2},
}
\tag{25.2}
\]

and

\[
\boxed{
\delta K22_{ab}
=(w_a^{\odot2})^TZw_b^{\odot2}
+2(w_a\odot w_b)^TZ(w_a\odot w_b).
}
\tag{25.3}
\]

For an off-diagonal future K31 costate `Lambda31`, set `R=ZW2` and `B=W Lambda31^T`. Then

\[
\boxed{
\chi_{Z\to31}=3\langle R,W\odot B\rangle_F.
}
\tag{25.4}
\]

This costs one dense product if `B` is already available, two if not. The full K22 branch contains the crossed Khatri--Rao quadratic form in (25.3) and is generically cost-closed without a low-rank K22 costate or a proof that the branch is Bellman-dark.

After removing a rank-2048 CP4 face and the one pairing direction, the symmetric off-diagonal `22` birth space still has dimension at least

\[
\binom{1024}{2}-2048-1=521,727.
\]

That is a capacity statement, not a value/tail certificate. Novelty remains

\[
(I-P_{old\ arena})\chi_{Z\to31},
\]

and its endpoint `tau`/Schur gain is open. Under one remaining GEMM-equivalent, only the one-transition/reused-`B` witness fits.

### 25.2 AQ cubic reset jet

The first reset jet not already assigned to the old tangent and square action is

\[
\boxed{
b_3=\frac16(I-P_{old})P_{\le2}[\delta'(g)h^3].
}
\tag{25.5}
\]

It lands back in the affine-quadratic `H<=2` state and has **strictly positive-depth** value only; every zero-length birth/consumer pairing is duplicate. A compiled future costate can contract it locally as

\[
\frac16\mathbb E[((I-P_{old})\lambda)(z)\delta'(g(z))h(z)^3].
\]

At `r=55` the full AQ coefficient dimension is

\[
d=1+r+\frac{r(r+1)}2=1596.
\]

For generic `(N,hh^T)`, the generated matrix algebra is all of `M_r`, so a bounded one/two-word compression is not exact. The forward AQ state plus backward AQ costate requires packed physical mixes across depth and does not fit the one-equivalent slack unless already paid/fused. The flat-r55 `0.0069` per-reset estimate is only a source-relative ceiling; promotion requires its strict-depth adjoint-weighted, complete-arena-residual amplitude to exceed `1.691e-5`.

### 25.3 Independent signed H3 defect

The post-AQ H3 candidate is a true-minus-assembled Wiener component

\[
I_3(Z_{true}-Z_{assembled}),
\qquad
Z_i\in\operatorname{Sym}^3(\mathbb R^{55}).
\]

It is not an AQ cubic jet. Its per-unit ambient dimension is

\[
\boxed{\binom{57}{3}=29,260.}
\tag{25.6}
\]

It is invisible to H1/H2 covariance but must still be residualized against the incumbent CP3/H3 dictionary and old depth arena. Its first external K31 variation has the exact form

\[
\delta\kappa_{31}=\langle Z,G_Z\rangle+\langle Y,G_Y\rangle,
\]

with the typed H3-H1-H1-H1 and H3-H1-H2-H2 coefficients derived in the H3 lane. A shared dictionary/CP realization is tensor-free at consumption, but generic carriage is closed; the existing fail-fast tail condition requires at least `86.2%` endpoint-weighted value capture, equivalently at most `13.8%` tail.

### 25.4 The r55 intersection question remains an actual rank test

The ambient H4 interface kernel in Section 24 has enormous dimension, but this alone does **not** imply that the lawful `2099`-dimensional regular r55 AQ inverse-fiber tangent intersects it. Let `T55` be that tangent and `Ocomplete` the complete old interface. The exact gate is

\[
\boxed{
\ker(O_{complete}|_{T55})\ne\{0\}
\quad\Longleftrightarrow\quad
\operatorname{rank}(O_{complete}|_{T55})<2099.
}
\tag{25.7}
\]

A generic 2099-dimensional subspace can avoid a large ambient kernel. Promotion therefore needs an explicit nonzero lawful tangent witness or a certified deficient minor/rank. `Z22` is an external pure-22 completion and the signed H3 defect is outside AQ; neither proves (25.7). The cubic jet is AQ, but its `(I-P_old)` projection must be nonzero in the **complete depth metric**, not merely locally.

| candidate | state type | cheapest exact local witness | binding unknown |
|---|---|---:|---|
| transient Z22 | external connected H4 `(22)` | 1 GEMM with reverse reuse | complete-arena Schur/tau; full K22 branch |
| cubic reset jet | AQ `H<=2` future birth | local surface scalar after full AQ costate | full-depth cost and tiny amplitude |
| signed H3 defect | independent `H3` | dictionary-contracted K31 tangent | signed acquisition, dictionary tail/rank |

For the r55 tangent gate, let `J=Ocomplete|T55`, `N=ker J`, and let `g` be the endpoint covector in the lawful tangent metric. A source-isotropic retention number is

\[
\boxed{
\tau_{55}=\frac{\|P_Ng\|^2}{\|g\|^2},
}
\tag{25.8}
\]

while for tangent covariance `Sigma` the correct value-weighted form is

\[
\boxed{
\tau_{55,\Sigma}
=\frac{g^TP_N\Sigma P_Ng}{g^T\Sigma g}.
}
\tag{25.9}
\]

Nonzero `ker J` is necessary but not sufficient: value also requires `P_Ng !=0`. For several frozen endpoints, the required residual rank is the Gram rank of their projected covectors, not the dimension of the source fiber.

## 26. Centered U-statistic and Stein-control audit for a sampled K31 action

Suppose the target scalar can be written

\[
\theta=\mathbb E[x]-3\,\alpha\cdot\beta,
\qquad
\alpha=\mathbb E[A],\quad\beta=\mathbb E[B].
\]

For two independent draws `z,z'`, the symmetric unbiased order-two kernel

\[
\boxed{
h(z,z')=\frac{x+x'}2
-\frac32\left(A\cdot B'+A'\cdot B\right)
}
\tag{26.1}
\]

has expectation `theta`. Writing centered `a=A-alpha`, `b=B-beta`, its first Hoeffding projection is

\[
h_1(z)=\frac12\left\{
x-\mathbb E x
-3[a\cdot\beta+\alpha\cdot b]
\right\}.
\]

Therefore the asymptotic influence function is

\[
\boxed{
\psi=2h_1
=x-\mathbb E x
-3[(A-\alpha)\cdot\beta+\alpha\cdot(B-\beta)],
}
\tag{26.2}
\]

and the degenerate second-order kernel is exactly

\[
\boxed{
h_2(z,z')
=-\frac32\left[
(A-\alpha)\cdot(B'-\beta)
+(A'-\alpha)\cdot(B-\beta)
\right].
}
\tag{26.3}
\]

There is no missing factor. For the complete normalized U-statistic on `N` draws,

\[
\boxed{
\operatorname{Var}(U_N)
=\frac{\operatorname{Var}(\psi)}N
+\frac{2\operatorname{Var}(h_2)}{N(N-1)}.
}
\tag{26.4}
\]

If `alpha,beta` are lawfully known exactly from lower moments, their product should be subtracted deterministically; U-estimating it only adds variance, and the sampled random part reduces to `x`.

For a Gaussian latent, a formal zero-variance Stein control can be written by solving the Ornstein--Uhlenbeck Poisson equation for `psi`: in Hermite coordinates, divide each nonconstant chaos coefficient by its degree. This does not give a new cheap weights-only route. For an AQ polynomial law it is equivalent in information and work to the exact trace/Hermite integration already available. For a true ReLU law the exact solution depends on the full activation-region/higher-chaos geometry and is as hard as the missing action. A finite weights-only Stein/control span has zero variance if and only if `psi` belongs exactly to that span; otherwise it only reduces the leading `Var(psi)/N` term and remains subject to the already failed MC variance/billing gate.

## 27. Ordinary fourth derivative is not Wiener H4

The polarization identities in Section 23 are exact when the queried object is already the connected quartic functional. They do not turn an arbitrary nonlinear response into its Wiener-H4 coefficient.

Let a one-dimensional Gaussian-Hermite expansion, with tensor contractions suppressed, be

\[
F(x)=\sum_{n\ge0}\frac1{n!}\langle K_n,\operatorname{He}_n(x)\rangle.
\]

At the origin, the ordinary fourth derivative is the triangular trace mixture

\[
\boxed{
D^4F(0)
=K_4-\frac12\operatorname{Tr}K_6
+\frac18\operatorname{Tr}^2K_8
-\frac1{48}\operatorname{Tr}^3K_{10}+\cdots.
}
\tag{27.1}
\]

In general the coefficient of the `q`-fold trace is `(-1)^q/(2^q q!)`. Contracting every tensor in (27.1) with three copies of `U` and one copy of `V` gives the corresponding K31-action contamination formula.

If the chaos hierarchy is known to stop at H8 and the two independent counterterms

\[
C_6=(\operatorname{Tr}K_6)[U,U,U,V],
\qquad
C_8=(\operatorname{Tr}^2K_8)[U,U,U,V]
\]

are available, then

\[
\boxed{
K_4[U,U,U,V]
=D^4F(0)[U,U,U,V]
+\frac12C_6-\frac18C_8.
}
\tag{27.2}
\]

Equivalently, in terms of the hierarchy of ordinary even derivatives, triangular inversion gives

\[
\boxed{
K_4
=D_4+\frac12\operatorname{Tr}D_6
+\frac18\operatorname{Tr}^2D_8
+\frac1{48}\operatorname{Tr}^3D_{10}+\cdots.
}
\tag{27.3}
\]

Thus a general smoothed ReLU law needs the infinite series

\[
K_4=\sum_{q\ge0}\frac{1}{2^q q!}
\operatorname{Tr}^qD_{4+2q}.
\]

For an unsmoothed literal ReLU, ordinary fourth derivatives at the kink are distributions rather than a classical oracle, so even the premise of a direct `D4` call fails without an explicitly defined smoothing/surface interface.

### Finite-stencil impossibility

No fixed finite set of ordinary response values can universally isolate the Wiener-H4 coefficient of an unrestricted law. If nodes `x_i` and weights `lambda_i` did so for every Hermite sequence, the Hermite exponential generating function would imply an identity of entire functions

\[
\boxed{
\sum_{i=1}^m\lambda_i e^{x_i z}
=Cz^4e^{z^2/2}.
}
\tag{27.4}
\]

The left side is a finite exponential polynomial of entire order at most one, whereas the nonzero right side has order two. This is impossible. Hence finite polarization is exact for a genuine cumulant/quartic-form oracle, but cannot separate connected H4/K31 from a general ReLU response. One needs the cumulant oracle itself, the full trace-counterterm hierarchy, or a rigorously truncated hierarchy with a remainder bound.

### Graph split

- `P2R0G19B_MATH_ORDINARY_D4_TRIANGLE` — **solved**: trace relations (27.1)--(27.3).
- `P2R0G19B_MATH_H4_FINITE_STENCIL_NOGO` — **solved no-go**: entire-function obstruction (27.4).
- `P2R0G19B_MATH_H4_COUNTERTERM_INTERFACE` — **conditional/open**: supply cumulant JVP or enough traced higher derivatives plus certified tail.

## 28. Z22 portfolio/Schur gates

Let `O` be the complete old feature/depth arena, let

\[
z=(I-P_O)\chi_Z,
\]

and let `P` contain the other retained new candidates. With endpoint residual `r`, put `z_perp=(I-P_P)z`. The exact raw-MSE gain from adding Z22 last is

\[
\boxed{
\Delta_{Z|P}
=\frac{\langle r,z_\perp\rangle^2}{\|z_\perp\|^2}.
}
\tag{28.1}
\]

If `F_{Z|P}` is the certified signal amplitude and `eta_Z` bounds acquisition/model error in the same normalized metric, the robust fire gate is

\[
\boxed{
(F_{Z|P}-\eta_Z)_+^2\ge2.86\times10^{-10}.
}
\tag{28.2}
\]

For standardized residual Z22 and strict-depth K31 features with target alignments `a_z,a_k` and mutual correlation `gamma`, their exact joint gain is

\[
\boxed{
G_{Z,K}
=\frac{a_z^2-2\gamma a_za_k+a_k^2}{1-\gamma^2},
\qquad
G_{Z|K}=\frac{(a_z-\gamma a_k)^2}{1-\gamma^2}.
}
\tag{28.3}
\]

No source-type orthogonality may replace these measured/certified Gram terms; the signed-H3 block uses the analogous block Schur complement.

The one-GEMM Z22 census is valid if and only if its scalar acquisition itself is free, `W2` is reused, and `B=W Lambda31^T` is reused. Otherwise the route costs at least two. The omitted crossed-K22 action has certified effective rank at least `980`; it must be proved Bellman-dark, shared, or low-rank, or its worst-case effect must be included in `eta_Z`.

Under the current q4 capacity floor, Z22 cannot displace the retained high-capacity H/all-distinct lane. If that lane is absent, Z22 must supply the full quoted `4.656e-9` missing portfolio value rather than merely the `2.86e-10` final-closer increment.

For adjusted-score swaps, a candidate `j` with raw gain `Delta_j` and incremental bill `C_j` dominates Z only according to the exact inequality

\[
\boxed{
(R-\Delta_Z)(u+\kappa C_Z)
\le(R-\Delta_j)(u+\kappa C_j),
}
\tag{28.4}
\]

not raw gain alone. Under the one-equivalent hard cap, however, any failure of the reuse premises closes Z22 before this softer comparison.

A thin strict-depth K31 competitor can fit the same one-equivalent cap only if its **pre-cut** factor ranks `R_l` satisfy

\[
\boxed{
\sum_l\frac{2n_lR_l}{n^2}\le1.
}
\tag{28.5}
\]

For ten full-width layers this means average `R_l<=51`; if one of the two projections is provably reused, `R_l<=102`. These are cost gates, not value guarantees: the factorization must exist before the billed cut and its endpoint-weighted discarded tail must be charged. Z22 versus thin K31 remains the full joint-Gram plus adjusted-score comparison (28.3)--(28.4), not an automatic Z22 win.

## 29. Exact crossed-K22 scalar, dark criterion, and norm bound

Let `L=Lambda22` be the symmetric future K22 costate, `q_a=w_a^{odot2}`, and `c_ab=w_a odot w_b`. The complete Z22-to-K22 scalar is

\[
\boxed{
\chi_{22}
=\sum_{a,b}L_{ab}q_a^TZq_b
+2\sum_{a,b}L_{ab}c_{ab}^TZc_{ab}.
}
\tag{29.1}
\]

The uncrossed first term is

\[
\langle L,W2^TZW2\rangle_F
=\langle Z,W2LW2^T\rangle_F.
\]

For the crossed term define

\[
\boxed{
M_x=\sum_{a,b}L_{ab}c_{ab}c_{ab}^T,
\qquad
\chi_x=2\langle Z,M_x\rangle_F.
}
\tag{29.2}
\]

If `N_Z` is the complete old-interface-residual source subspace, the crossed branch is uniformly Bellman-dark if and only if

\[
\boxed{
P_{N_Z}M_x=0.
}
\tag{29.3}
\]

For a source-radius receipt `||Z||F<=rho_Z`, the exact support-function bound is

\[
\boxed{
\sup_{Z\in N_Z,\ \|Z\|_F\le\rho_Z}|\chi_x|
=2\rho_Z\|P_{N_Z}M_x\|_F.
}
\tag{29.4}
\]

This is both necessary and sufficient; a small observed scalar on one Z is not a dark-branch certificate.

### Factorization and cost

If

\[
L=\sum_{s=1}^k\lambda_s\ell_s\ell_s^T,
\qquad
M_s=W\operatorname{diag}(\ell_s)W^T,
\]

then

\[
\boxed{
M_x=\sum_{s=1}^k\lambda_s(M_s\odot M_s),
\qquad
\chi_x=2\sum_s\lambda_s\langle Z,M_s\odot M_s\rangle_F.
}
\tag{29.5}
\]

Each nonreused `M_s` costs one dense product after column scaling, so the exact-rank route costs at least `rank(L)` products absent further shared structure. Rank one is a one-product loophole; generic rank is not.

Equivalently, let `T` be the `n by n^2` matrix whose `(a,b)` column is `c_ab`. Then

\[
M_x=T\operatorname{diag}(\operatorname{vec}L)T^T,
\qquad
TT^T=(WW^T)\odot(WW^T).
\]

Consequently

\[
\boxed{
|\chi_x|
\le2\|Z\|_F\|T\|_2^2\|L\|_F
\le2\|Z\|_F\|W\|_2^4\|L\|_F.
}
\tag{29.6}
\]

The projected norm in (29.4) is sharper and is the minimal lawful `eta_Z` observable. If neither it nor a certified bound on `rho_Z,||T||2,||L||F` is available, the K31-only Z22 witness has no rigorous full-interface error bar.

## 30. Exact trace-tail certificate after finite H6/H8 counterterms

Section 27 shows that a direct ordinary-fourth-derivative interface is a
triangular mixture of the Wiener-Hermite even hierarchy.  This section makes
the omitted-tail gate explicit.  Let `Tr_Sigma` denote contraction of a tensor
pair with the Gaussian covariance.  After supplying every counterterm through
order `4+2Q`, the exact residual in the recovered fourth chaos is

\[
\boxed{
R_Q=\sum_{q>Q}{(-1)^q\over 2^q q!}
\operatorname{Tr}_{\Sigma}^{q}K_{4+2q}.}
\tag{30.1}
\]

Thus after exact `K6` and `K8` counterterms,

\[
R_2=-{1\over48}\operatorname{Tr}_{\Sigma}^{3}K_{10}
 +{1\over384}\operatorname{Tr}_{\Sigma}^{4}K_{12}
 -{1\over3840}\operatorname{Tr}_{\Sigma}^{5}K_{14}+\cdots.
\tag{30.2}
\]

For the injective and Hilbert--Schmidt tensor norms respectively,

\[
\|\operatorname{Tr}_{\Sigma}^{q}A\|_{\rm op}
 \le \|\Sigma\|_*^q\|A\|_{\rm op},
\qquad
\|\operatorname{Tr}_{\Sigma}^{q}A\|_{\rm HS}
 \le \|\Sigma\|_{\rm HS}^q\|A\|_{\rm HS}.
\tag{30.3}
\]

If `S(U,V)=Sym(U^{otimes3} otimes V)`, then

\[
\boxed{
\|S(U,V)\|_{\rm HS}
 ={\|U\|^3\|V\|\over2}\sqrt{1+3\rho_{UV}^2},
\quad
\rho_{UV}={\langle U,V\rangle\over\|U\|\|V\|}.}
\tag{30.4}
\]

Consequently the exact mixed K31 action obeys

\[
|R_Q[U,U,U,V]|
\le \|S(U,V)\|_{\rm HS}
\sum_{q>Q}{\|\Sigma\|_{\rm HS}^q
\|K_{4+2q}\|_{\rm HS}\over2^q q!}.
\tag{30.5}
\]

The triangle sum is sharp without a phase/orthogonality premise.  One simple
analytic sufficient condition is

\[
\|K_{4+2q}\|_{\rm op}\le M q!R^{-q},
\qquad \theta={\|\Sigma\|_*\over2R}<1,
\]

which gives

\[
\boxed{\|R_Q\|_{\rm op}\le
M{\theta^{Q+1}\over1-\theta}.}
\tag{30.6}
\]

This is a Gaussian heat-analytic/Gevrey receipt, not a consequence of local
ordinary analyticity.

### 30.1 Weighted-Fock alternative

Define the measurable tail norm

\[
S_{\rho,Q}^2=\sum_{q>Q}
{\rho^{4+2q}\|K_{4+2q}\|_{\rm HS}^2\over(4+2q)!}.
\tag{30.7}
\]

Cauchy--Schwarz gives the lawful certificate

\[
\boxed{
\|R_Q\|_{\rm HS}\le {S_{\rho,Q}\over\rho^2}
\sqrt{G_Q\!\left({\|\Sigma\|_{\rm HS}^2\over\rho^2}\right)},
\quad
G_Q(x)=\sum_{q>Q}{(2q+4)!\over4^q(q!)^2}x^q.}
\tag{30.8}
\]

The generating series has radius one, so this needs
`rho>||Sigma||HS`.  If `F=P_r G` is supplied with an actual
Ornstein--Uhlenbeck smoothing receipt, then `K_n(F)=r^nK_n(G)` and
`S_{rho,Q}<=||G||_2` whenever `rho r^2<=1`.  A usable simultaneous margin
therefore requires

\[
\boxed{r^2\|\Sigma\|_{\rm HS}<1.}
\tag{30.9}
\]

Merely calling a response "Gaussian smoothed" does not establish (30.9).

### 30.2 Exact rank-one smoothed-ReLU specialization

For one certified rank-one normalized-ReLU channel `P_r psi`, the even
coefficients have the aligned form

\[
K_{4+2q}={(-1)^{q+1}(2q+1)!!\over\sqrt\pi}
r^{4+2q}.
\tag{30.10}
\]

All direct-tail terms then have one sign.  After `K6,K8`, their exact absolute
sum is

\[
\boxed{
T_2^{\rm ReLU}(r)={r^4\over\sqrt\pi}
\left[(1-r^2)^{-3/2}-1-{3\over2}r^2-{15\over8}r^4\right]
\ge {35\over16\sqrt\pi}r^{10}.}
\tag{30.11}
\]

With ten unit-weight, coherently aligned D10 layers, the strict endpoint
allowance is `3.80e-7` per layer.  The first omitted term alone therefore
forces

\[
\boxed{
r\le\left({16\sqrt\pi\over35}\,3.80\times10^{-7}\right)^{1/10}
\simeq0.223.}
\tag{30.12}
\]

This favorable scalar formula is not transferable to a multi-route AQ/ReLU
response unless the rank-one reduction, every `r_l`, and every endpoint route
weight are certified.

### 30.3 D10 endpoint gate and impossibility from present summaries

For endpoint weights `beta_l` and mixed directions `U_l,V_l`, a sufficient
deterministic gate is

\[
\boxed{
\sum_{l=5}^{14}|\beta_l|\|S(U_l,V_l)\|_{\rm HS}
{S_{\rho_l,Q,l}\over\rho_l^2}
\sqrt{G_Q\!\left({\|\Sigma_l\|_{\rm HS}^2\over\rho_l^2}\right)}
\le3.80\times10^{-6}.}
\tag{30.13}
\]

If only `K10` is omitted under the equal coherent allocation, its contracted
triple trace must be at most `48(3.80e-7)=1.824e-5` per layer.  For one already
aggregated D10 oracle the allowance is instead `1.824e-4`; do not divide by
ten twice.

No such bound follows from `D4,K6,K8`, ordinary analyticity, or an unweighted
Gaussian `L2` tail.  In one dimension,

\[
F_M(x)={M/48\over4!}\operatorname{He}_4(x)
       +{M\over10!}\operatorname{He}_{10}(x)
\tag{30.14}
\]

is entire, has `K6=K8=0` and `D4F_M(0)=0`, yet has arbitrarily large
`K4=M/48`.  Even fixing the Gaussian `L2` tail does not help: concentrating
tail norm `S` at `n=2q+4` contributes asymptotically
`4 pi^{-1/4} S q^{7/4}` to the trace remainder, which is unbounded.

The minimal lawful observable is therefore either the endpoint-weighted
contracted trace spectrum

\[
\tau_{q,l}=\|\operatorname{Tr}_{\Sigma_l}^qK_{4+2q,l}\|_{\rm op/HS}
\]

with its certified sum, a weighted-Fock pair `(rho,S_rho)`, or a direct
contract-native Hermite-H4 action.  Low-order faces and a raw unweighted chaos
tail do not certify the acquisition.

### Graph split

- `P2R0G19B_MATH_H4_TRACE_REMAINDER_IDENTITY` — **solved**: (30.1)--(30.5).
- `P2R0G19B_MATH_H4_WEIGHTED_FOCK_CERT` — **conditional/open receipt**: (30.7)--(30.9), (30.13).
- `P2R0G19B_MATH_H4_RELU_RANK1_TAIL` — **solved conditional specialization**: (30.10)--(30.12).
- `P2R0G19B_MATH_H4_LOW_ORDER_TAIL_NOGO` — **solved no-go**: (30.14) and the fixed-`L2` counterexample.

## 31. Contract-native Hermite-H4 oracle for one AQ ReLU reset

There is one exact way to avoid the infinite trace inversion in Section 30:
query the Gaussian Hermite score itself.  Let

\[
Z\sim N(0,I_r),\qquad
g(Z)=c+b^TZ+\frac12(Z^TMZ-\operatorname{tr}M),
\qquad F(Z)=g(Z)_+,
\tag{31.1}
\]

and write `p(Z)=nabla g=b+MZ`.  For a symmetric fourth-order test tensor `H`,
define

\[
\mathcal A_H=\langle K_4(F),H\rangle
=E\langle F(Z)\operatorname{He}_4(Z),H\rangle.
\]

Gaussian integration by parts, interpreted distributionally or by a smooth
ReLU limit, gives the exact surface formula

\[
\boxed{
\mathcal A_H
=E\left[
\delta''(g)H[p,p,p,p]
+6\delta'(g)H[M,p,p]
+3\delta(g)H[M,M]
\right].}
\tag{31.2}
\]

For the mixed action `H=Sym(U^{otimes3} otimes V)`, put

\[
g_U=b^TU+Z^TMU,\quad
g_V=b^TV+Z^TMV,\quad
g_{UU}=U^TMU,\quad g_{UV}=U^TMV.
\]

Then (31.2) is

\[
\boxed{
K_4(F)[U,U,U,V]
=E\left[
\delta''(g)g_U^3g_V
+3\delta'(g)(g_{UU}g_Ug_V+g_{UV}g_U^2)
+3\delta(g)g_{UU}g_{UV}
\right].}
\tag{31.3}
\]

These constants follow from the exact chain rule

\[
D^4(g_+)=
\delta''(g)p^{\otimes4}
+6\delta'(g)\operatorname{Sym}(M\otimes p^{\otimes2})
+3\delta(g)\operatorname{Sym}(M\otimes M).
\]

This is the latent-Hermite action.  It must not be conflated with the
external-label same-site `K31`, which Section 22 proves is already assembled
by D11.

### 31.1 Exact one-dimensional Fourier reduction

Let

\[
S_t=(I-itM)^{-1},\qquad
\mu_t=itS_tb,\qquad
m_t=b+M\mu_t,\qquad
C_t=MS_tM,
\tag{31.4}
\]

and choose the determinant square-root branch continuous from `t=0`.  The
AQ characteristic function is

\[
\boxed{
\chi(t)=
e^{it(c-\frac12\operatorname{tr}M)}
\det(I-itM)^{-1/2}
\exp\!\left(-\frac{t^2}{2}b^TS_tb\right).}
\tag{31.5}
\]

Under the complex tilted Gaussian, `p` has mean `m_t` and covariance `C_t`.
Therefore (31.2) reduces exactly to

\[
\boxed{
\begin{aligned}
\mathcal A_H={1\over2\pi}\int_{\mathbb R}\chi(t)\{&(it)^2[
 H[m_t,m_t,m_t,m_t]+6H[C_t,m_t,m_t]+3H[C_t,C_t]]\\
&+6(it)[H[M,m_t,m_t]+H[M,C_t]]+3H[M,M]\}\,dt.
\end{aligned}}
\tag{31.6}
\]

The integral is a tempered/distributional Fourier integral; equivalently,
insert `exp(-epsilon t^2/2)` and take `epsilon downarrow0`.  An implementation
must provide a nondegeneracy and quadrature-tail receipt before replacing the
integral by finitely many nodes.

For the mixed action define

\[
m_u=U^Tm_t,\quad m_v=V^Tm_t,
\quad c_{uu}=U^TC_tU,\quad c_{uv}=U^TC_tV.
\]

Its scalar integrand is

\[
\boxed{
\begin{aligned}
{\chi(t)\over2\pi}\{&(it)^2[
m_u^3m_v+3c_{uu}m_um_v+3c_{uv}m_u^2+3c_{uu}c_{uv}]\\
&+3(it)[g_{UU}(m_um_v+c_{uv})
+g_{UV}(m_u^2+c_{uu})]+3g_{UU}g_{UV}\}.
\end{aligned}}
\tag{31.7}
\]

Thus one AQ reset needs no stored H6/H8/H10 hierarchy and incurs no trace-tail
ambiguity: (31.6) is the contract-native Wiener-H4 action itself.

### 31.2 Square/Jordan contractions and exact arithmetic census

For a square-lift costate let

\[
H_{A,B}=\operatorname{Sym}(A\otimes B),
\qquad H_{A,B}[x,x,x,x]=(x^TAx)(x^TBx).
\]

For symmetric `C,D`, every contraction in (31.6) reduces to matrix traces:

\[
\boxed{
\begin{aligned}
H_{A,B}[C,D]={1\over6}\{&
\operatorname{tr}(AC)\operatorname{tr}(BD)
+\operatorname{tr}(BC)\operatorname{tr}(AD)\\
&+2\operatorname{tr}(B C A D)
+2\operatorname{tr}(A C B D)\}.
\end{aligned}}
\tag{31.8}
\]

In particular,

\[
H_{A,B}[C,C]
={1\over3}\operatorname{tr}(AC)\operatorname{tr}(BC)
+{2\over3}\operatorname{tr}(ACBC),
\tag{31.9}
\]

and

\[
H_{A,B}[C,x,x]
={\operatorname{tr}(AC)x^TBx+
\operatorname{tr}(BC)x^TAx+4x^TACBx\over6}.
\tag{31.10}
\]

The `U_{A,B}` convention of Section 1 adds its fixed Wick/square factor; it
does not change the closure test.

After one eigendecomposition of each `M_i`, every `S_t,C_t` is diagonal in
that unit's eigenbasis.  For a rank-`k` mixed-direction costate and `K`
certified Fourier nodes, the arithmetic scales as

\[
\boxed{
O(Dnr^3)+O(Dnkr^2)+O(DK nkr),
\qquad \text{state }O(nr^2+kr).}
\tag{31.11}
\]

Measured against one `n^3` dense GEMM, the factorization plus direction
transforms cost approximately

\[
\boxed{
C_{\rm AQscore}\simeq
{D(r^3+kr^2)\over n^2},}
\tag{31.12}
\]

before quadrature and constant factors.  For `n=1024,D=10`, the factorization
term alone is approximately `0.87` equivalent at `r=45` and `1.59` at `r=55`.
If all eigensystems are lawfully reused, the rank gate from (31.12) is
`k<=n^2/(Dr^2)`, about `51` at `r=45` and `34` at `r=55`.  If they are not
reused, the nominal residual gates tighten to about `k<=6` at `r=45` and are
already impossible at `r=55`.  These are arithmetic screens, not billing
receipts; the deployed census must decide how eigensolves and certified
quadrature are charged.

A full unstructured `r^4` costate still costs `O(nr^4)=O(n^3)` per layer just
to contract local births.  Across D10 that exceeds the one-equivalent
incremental allowance.  Hence the direct oracle is deployable only with a
proved low mixed rank, common commuting/Jordan basis, or already-reused
factorization.

### 31.3 What the direct score does and does not solve

Equation (31.6) exactly solves a **local AQ birth acquisition**.  It does not
by itself prove that the reverse action through ten correlated ReLU resets
stays in one tensor/matrix stream.  Differentiating `A_H(c_i,b_i,M_i)` across
neurons produces the family of unit-specific pullbacks

\[
\{D_{(c_i,b_i,M_i)}\mathcal A_{H,i}\}_{i=1}^n.
\]

The one-stream reverse is exact if and only if their future-visible
projections lie in the same finite cyclic/Jordan module of Sections 5, 18,
and 24.  Generically their neuron-versus-latent separation rank is `n`, so the
surface oracle removes the **trace-tail** obstruction but not the
**self-consistent quotient-leakage** obstruction.

If the true preactivation is `g+e` rather than the AQ law (31.1), ReLU
Lipschitzness gives the lawful local model-error bound

\[
\boxed{
|\langle K_4((g+e)_+)-K_4(g_+),H\rangle|
\le \sqrt{4!}\,\|e\|_2\|H\|_{\rm HS}.}
\tag{31.13}
\]

Thus an exact D10 certificate must charge AQ residual, quadrature error, and
cyclic-module leakage together:

\[
\boxed{
\sum_{l=5}^{14}|\beta_l|
\left[
\sqrt{24}\,\|e_l\|_2\|H_l\|_{\rm HS}
+\epsilon_{{\rm quad},l}
+\epsilon_{{\rm module},l}
\right]
\le3.80\times10^{-6}.}
\tag{31.14}
\]

Finally, novelty is the complete-interface residual
`(I-P_{O_l})K_4(g_+)`, propagated into the future cyclic module.  If the
costate lies in the old CP4/star/pair range, (31.6) merely recomputes D11.
Only the latent/all-distinct component in the future-visible complement may
receive raw-MSE credit.

### Graph split

- `P2R0G19B_MATH_AQ_H4_SCORE_IDENTITY` — **solved**: (31.2)--(31.7).
- `P2R0G19B_MATH_AQ_H4_SCORE_COST` — **conditional**: reuse/rank/quadrature gates (31.11)--(31.12).
- `P2R0G19B_MATH_AQ_H4_SCORE_MODEL_ERROR` — **open frozen-net receipt**: (31.13)--(31.14).
- `P2R0G19B_MATH_AQ_H4_SCORE_SELFCONSISTENCY` — **open structural gate**: local acquisition does not imply reverse-module closure.

### 31.4 Minimal fused-score interface and exact OU pullback

For `A=<U,Z>`, `B=<V,Z>`, `a=||U||^2`, and `d=<U,V>`, the mixed score is

\[
\boxed{
H_{31}(Z;U,V)=:A^3B:
=A^3B-3aAB-3dA^2+3ad.}
\tag{31.15}
\]

One fused expectation is information-theoretically minimal:

\[
\boxed{E[F H_{31}]=K_4(F)[U,U,U,V].}
\tag{31.16}
\]

If an interface exposes only raw expectations, the four generally independent
quantities `E[FA^3B]`, `E[FAB]`, `E[FA^2]`, and `E[F]` are necessary and
sufficient.  Fewer do not recover (31.16) uniformly without an additional
centering or orthogonality identity.

There is also an exact propagation theorem for a genuine Gaussian
second-quantized/OU transition.  If

\[
Y=RX+(I-RR^*)^{1/2}G,
\qquad (P_RF)(x)=E[F(Y)\mid X=x],
\]

with independent standard `X,G`, then

\[
\boxed{
K_4(P_RF)=(R^*)^{\otimes4}K_4(F),
\qquad
\mathcal A_{31}(P_RF;U,V)
=K_4(F)[RU,RU,RU,RV].}
\tag{31.17}
\]

Writing `u=RU,v=RV`, its JVP and reverse pullback are

\[
\delta\mathcal A_{31}
=3K_4[u,u,\dot R U,v]+K_4[u,u,u,\dot R V],
\tag{31.18}
\]

\[
\boxed{
\nabla_R\mathcal A_{31}
=3K_4[u,u,\cdot,v]\otimes U^\flat
+K_4[u,u,u,\cdot]\otimes V^\flat.}
\tag{31.19}
\]

Thus a certified OU/fresh-cavity transition preserves the action with a
rank-two reverse formula.  This is a genuine zero-leakage sufficient
condition.  A deterministic dense layer followed by a correlated ReLU is not
an OU transition merely because its covariance matches (31.17).

For normalized scalar ReLU `psi(z)=sqrt(2)z_+`, `K4(psi)=-1/sqrt(pi)`.  If
its correlation vector with the core is `r`, then

\[
\mathcal A_{31}(P_r\psi;U,V)
=-{(r^TU)^3(r^TV)\over\sqrt\pi},
\tag{31.20}
\]

and the two-ReLU pure-H4 kernel slice is `rho^4/(24 pi)`.  The full ReLU
kernel pullback instead uses `(pi-arccos rho)/pi`; it cannot be replaced by
the H4 slice unless chaos-diagonal closure is proved.

### 31.5 Exact AQ-parameter JVP

For an AQ perturbation

\[
\dot g=\dot c+\dot b^TZ+Q_{\dot M}(Z),
\]

the derivative of the fused score is the simpler volume moment

\[
\boxed{
D\mathcal A_{31}[\dot c,\dot b,\dot M]
=E[1_{\{g>0\}}\dot g\,H_{31}(Z;U,V)].}
\tag{31.21}
\]

Hence, before differentiating the frame/directions themselves,

\[
\partial_c\mathcal A_{31}=E[1_{g>0}H_{31}],
\qquad
\nabla_b\mathcal A_{31}=E[1_{g>0}ZH_{31}],
\tag{31.22}
\]

\[
\boxed{
\nabla_M\mathcal A_{31}
={1\over2}E[1_{g>0}(ZZ^T-I)H_{31}].}
\tag{31.23}
\]

These give the lawful joint-refit derivative for one local AQ reset; they do
not require differentiating a numerical Fourier quadrature.  The pullbacks
are nevertheless unit-specific future observables, so their transport is
still governed by the cyclic-rank test.

The AQ family is closed under a deterministic linear substitution `z=Rx`:

\[
b'=R^Tb,\qquad M'=R^TMR,\qquad
c'=c+{1\over2}(\operatorname{tr}M'-\operatorname{tr}M).
\tag{31.24}
\]

It is not closed under a nondegenerate noisy conditional transition followed
by ReLU.  The conditional response `E[g(Y)_+|X=x]` is analytic across the old
surface when the conditional noise is nondegenerate, whereas a nontrivial
AQ-ReLU retains a kink.  Therefore equality with one new AQ-ReLU is impossible
except under a null-noise/unused-direction degeneracy or another explicitly
proved finite-module identity.

### 31.6 Sharp local-versus-global conclusion

A fixed mixed action in a rank-`k` AQ core needs only the nonzero eigenvalues,
the coordinates of `b,U,V`, and the relevant perpendicular Gram scalars:
`Theta(k)` spectral information after factorization.  One fused score keeps
`2r+3` local scalars; four raw moments keep `2r+6`.

This finite **local** rank does not bound the **future cyclic** rank.  Already
in one dimension, pullbacks through distinct ReLU thresholds generate
linearly independent delta derivatives.  Equivalently, repeated
multiplication by the nonpolynomial ReLU kernel derivative generates an
unbounded function algebra unless it satisfies a certified finite polynomial
relation.  Hence a finite family of surface moments is globally exact if and
only if

\[
\boxed{L_l^*\mathcal S_{l+1}\subseteq\mathcal S_l}
\tag{31.25}
\]

at every layer and all local source/reset descendants lie in `S_l`.  Its
minimal size is the complete future cyclic-module rank, not the AQ variable
rank.

Additional graph splits:

- `P2R0G19B_MATH_H4_FUSED_SCORE_INTERFACE` — **solved**: one fused/four raw score observations, (31.15)--(31.16).
- `P2R0G19B_MATH_H4_OU_INTERTWINER` — **solved conditional theorem**: exact zero-leakage propagation (31.17)--(31.20).
- `P2R0G19B_MATH_H4_AQ_JVP` — **solved local derivative**: (31.21)--(31.24).
- `P2R0G19B_MATH_H4_SURFACE_GLOBAL_CLOSURE` — **open/no-go generically**: finite local surface rank is insufficient without (31.25).

## 32. Sharp one-dimensional threshold-rank lower bound

The global obstruction in Section 31.6 is already exact after one reset in
one Gaussian dimension.  Let

\[
h(z)=\operatorname{He}_4(z)=z^4-6z^2+3,
\qquad a_1<a_2<\cdots<a_n,
\]

and consider the AQ special case `g_i(z)=z-a_i`.  The local score derivative
with respect to the offset of unit `i` has representing function

\[
f_i(z)=1_{\{z>a_i\}}h(z).
\tag{32.1}
\]

These `n` functions are linearly independent in `L2(gamma)`.  Indeed, if
`sum_i c_i f_i=0`, then on every interval `(a_k,a_{k+1})`,

\[
h(z)\sum_{i\le k}c_i=0
\]

almost everywhere.  Since a nonzero polynomial cannot vanish on an interval,
every partial sum is zero and hence every `c_i=0`.

Therefore

\[
\boxed{
\dim\operatorname{span}\{1_{z>a_i}\operatorname{He}_4(z):1\le i\le n\}=n.}
\tag{32.2}
\]

This is a concrete neuron-versus-score separation-rank `n` witness.  A
full-rank dense physical mixing after the reset preserves that rank.  It also
shows why finite AQ variable dimension does not imply finite future cyclic
dimension: the nonpolynomial gate locations are independent functional
labels.

The exact frozen-instance diagnostic is the positive-definite Gram matrix

\[
\boxed{
G_{ij}=\langle f_i,f_j\rangle
=\int_{\max(a_i,a_j)}^\infty
\operatorname{He}_4(z)^2\varphi(z)\,dz.}
\tag{32.3}
\]

If `F:R^n->L2(gamma)` has columns `f_i`, then `F^*F=G`.  The best rank-`k`
module has exact worst-direction error

\[
\boxed{
\inf_{\operatorname{rank}P=k}\|(I-P)F\|_{\rm op}
=\sqrt{\lambda_{k+1}(G)},}
\tag{32.4}
\]

and exact aggregate squared tail

\[
\boxed{
\inf_{\operatorname{rank}P=k}\|(I-P)F\|_{\rm HS}^2
=\sum_{j>k}\lambda_j(G).}
\tag{32.5}
\]

Thus the lawful small-stream receipt is the endpoint-weighted analogue of
the Gram spectrum (32.3), or the complete future cyclic-module singular
spectrum.  Counting AQ variables, matching covariances, or noting that every
unit is individually rank one cannot replace it.

Coincident thresholds are a real exceptional collapse: if all `a_i` are
equal, the family in (32.1) has rank one.  This is precisely the common-gate
factorization used by the frozen one-insertion theorem.  The threshold witness
does not assert that every frozen net has rank `n`; it proves that no
label-free/density-only theorem can certify a smaller rank.  The actual net
must establish repeated gate coincidence, OU intertwining, Jordan closure,
or a small measured future Gram tail.

### Graph split

- `P2R0G19B_MATH_H4_THRESHOLD_RANK_NOGO` — **solved counterexample**: exact rank `n`, (32.1)--(32.2).
- `P2R0G19B_MATH_H4_CYCLIC_GRAM_GATE` — **open frozen-net receipt**: certify (32.4)--(32.5) in the endpoint-weighted actual module.

## 33. Immediate-seam reuse versus hidden H3 persistence

A local source need not open a new reverse row when it enters a state whose
complete Bellman costate is already resident.  Let a reset-local signed-H3
birth `z_l` emit into the next-cut quartic product state by

\[
L_lz_l=(L_{4,l}z_l,L_{31,l}z_l,L_{22,l}z_l,L_{211,l}z_l),
\tag{33.1}
\]

and let the resident next-cut costate be

\[
b_{l+1}=(Q_{4,l+1},Q_{31,l+1},Q_{22,l+1},\Lambda_{211,l+1}).
\]

Then the exact immediate scalar is

\[
\boxed{
\langle b_{l+1},L_lz_l\rangle
=\langle G_l,z_l\rangle,
\quad
G_l=L_{4,l}^*Q_{4,l+1}
+L_{31,l}^*Q_{31,l+1}
+L_{22,l}^*Q_{22,l+1}
+L_{211,l}^*\Lambda_{211,l+1}.}
\tag{33.2}
\]

Computing (33.2) is one local adjoint application, not a second backward
bundle, if and only if all four coordinates are literal views of the
already billed resident costate at the identical seam/frame.

The D11 forward code has the necessary product-state seam: it constructs
`(pre_k4,pre_k31,pre_k22)`, passes them through `_full_relu_reset`, and feeds
`(k4_target,k31_target,k22_target)` through `_reset_stars`, `_cp4_reset`, and
the next `_push_quartic`.  Therefore the exact reverse of this **full** product
map necessarily has a literal `Q31` coordinate, together with `Q4,Q22`.
It does **not** expose an independent off-diagonal `Lambda211`, because D11
stores only the collision-plane product state.  A complete new D10 provider
must either add that coordinate or prove that its `L211^*Lambda211` action is
compiled into the resident local reduction.

This proves the seam-reuse theorem, but not current ownership.  The proposed
D10 word has so far been specified abstractly as a q4-perpendicular costate,
not as a concrete reverse tape of the full D11 product state.  Shared weights,
the same latent rank, or the existence of a mathematical `Q31` derivative do
not prove that the deployed resident array exposes it.

Using only the immediate K31 coordinate gives the omitted scalar

\[
\boxed{
\epsilon_{4/22,l}
=\langle L_{4,l}^*Q_{4,l+1}
+L_{22,l}^*Q_{22,l+1}
+L_{211,l}^*\Lambda_{211,l+1},z_l\rangle.}
\tag{33.3}
\]

It must be computed or bounded; K31 ownership does not make K4/K22/Q211 emissions
Bellman-dark.  Moreover, if `z_l` has an independently persistent H3 state
`P_lz_l` with future costate `Q_{3,l+1}`, the further term

\[
\epsilon_{\rm persist,l}=\langle P_l^*Q_{3,l+1},z_l\rangle
\tag{33.4}
\]

is outside (33.2) unless that H3 coordinate is also resident.  This is the
precise distinction between a free one-step implicit action and a new
strict-depth future row.

### Graph split

- `P2R0G19B_MATH_IMMEDIATE_FULL_SEAM_REUSE` — **solved theorem**: complete local emission reuses a resident full product-state adjoint, (33.1)--(33.2).
- `P2R0G19B_MATH_D10_Q31_LITERAL_OWNERSHIP` — **open implementation receipt**: prove the abstract D10 word exposes the exact D11 seam coordinates.
- `P2R0G19B_MATH_H3_K4K22Q211_EMISSION` — **open completeness gate**: compute/bound (33.3).
- `P2R0G19B_MATH_H3_PERSISTENCE_ROW` — **open structural gate**: certify (33.4) is zero/resident or charge a new future row.

## 34. Correct fused score for the off-diagonal 211 carrier

The `H31=:A^3B:` oracle in Section 31 is not the target sector of the
noncommutative q4-perpendicular carrier.  Let `M_r,M_s,M_t` be NCQC cores and
define the commutator Gram

\[
G^{(r)}_{st}
=\langle[M_r,M_s],[M_r,M_t]\rangle_F.
\]

In the current residualized normalization,

\[
\boxed{
q_{4,\perp}(r,r,s,t)=-8G^{(r)}_{st}.}
\tag{34.1}
\]

Its K31 restriction is identically zero because
`[M_r,M_r]=0`; K22 sees only the diagonal `s=t`.  The true carrier is the
off-diagonal external `211` block.  Therefore no universal K31 scalar fold,
including an exact `H31` score, can substitute for it.

For latent directions `U,S,T`, put

\[
A=\langle U,Z\rangle,\quad B=\langle S,Z\rangle,
\quad C=\langle T,Z\rangle,
\]

and abbreviate

\[
v_A=\|U\|^2,\quad c_{US}=\langle U,S\rangle,
\quad c_{UT}=\langle U,T\rangle,\quad c_{ST}=\langle S,T\rangle.
\]

The correct fused `211` Gaussian score is

\[
\boxed{
\begin{aligned}
H_{211}(Z;U,S,T)=:A^2BC:
={}&A^2BC-v_ABC-2c_{US}AC-2c_{UT}AB-c_{ST}A^2\\
&+v_Ac_{ST}+2c_{US}c_{UT}.
\end{aligned}}
\tag{34.2}
\]

It gives the exact action

\[
\boxed{E[F H_{211}]=K_4(F)[U,U,S,T].}
\tag{34.3}
\]

One fused expectation is minimal.  A raw-moment interface generally needs
six independent values: those multiplying `A^2BC,BC,AC,AB,A^2,1` in
(34.2), rather than the four needed by `H31`.

For the AQ ReLU of Section 31, the exact surface derivative is

\[
\boxed{
\begin{aligned}
K_4(g_+)[U,U,S,T]=E\{&
\delta''(g)g_U^2g_Sg_T\\
&+\delta'(g)[g_{UU}g_Sg_T+2g_{US}g_Ug_T
+2g_{UT}g_Ug_S+g_{ST}g_U^2]\\
&+\delta(g)[g_{UU}g_{ST}+2g_{US}g_{UT}]\}.
\end{aligned}}
\tag{34.4}
\]

Equivalently, use the general Fourier formula (31.6) with
`H=Sym(U^{otimes2} otimes S otimes T)`.  Its local vector-rank arithmetic is
of the same order as (31.11), but the needed mixed rank is now the rank of an
adjoint-weighted decomposition of the **off-diagonal 211 block**, not the
ordinary matrix rank of a K31 mask.  A dense external rank-one-looking mask
can still have mixed/Waring rank `n` in this partially symmetric tensor
format.

The correct novelty test is therefore

\[
(I-P_{O_{\rm complete}})
\sum_{r,s\ne t}\Lambda^{211}_{r;st}
K_4(g_{r,+})[U_r,U_r,S_s,T_t],
\tag{34.5}
\]

including the crossed K22 and CP4 reads in `O_complete`.  Measuring only
`H31`, or only diagonal `H211` with `S=T`, leaves (34.1) unobserved.

### Graph split

- `P2R0G19B_MATH_Q4PERP_211_IDENTITY` — **solved**: commutator Gram carrier and K31-zero restriction (34.1).
- `P2R0G19B_MATH_H211_FUSED_SCORE` — **solved local oracle**: (34.2)--(34.4).
- `P2R0G19B_MATH_H211_MIXED_RANK` — **open frozen-net cost gate**: decompose the residualized adjoint in (34.5) at cap-compatible rank.

## 35. Exact H2-gate Jordan intertwiner and the higher-gate caveat

There is an exact algebraic reason the first correlated-gate correction is
promising.  For the square lifts of Section 1,

\[
\boxed{
\Pi_4\bigl(U_{R,S}Q_M\bigr)
=2\left(U_{M\circ R,S}+U_{R,M\circ S}\right),
\qquad M\circ R={MR+RM\over2}.}
\tag{35.1}
\]

This is the one-contraction term in the Wiener product `I4 times I2`; the H0
and H2 pieces of the raw product have already been Wick-subtracted in
`U_{R,S}`.

Let a cavity-conditioned reset gate have low-chaos expansion

\[
G(z)=g_0+g_2Q_M(z)+G_{\ge4}(z).
\]

Then its inherited H4 action is

\[
\boxed{
\Pi_4(GU_{R,S})
=g_0U_{R,S}
+2g_2(U_{M\circ R,S}+U_{R,M\circ S})
+\Pi_4(G_{\ge4}U_{R,S}).}
\tag{35.2}
\]

For the standard cavity parameterization with standardized bias `a`, cavity
scale `s`, and output prefactor `sigma`, the first two coefficients are
`g0=sigma Phi(a)` and `g2=sigma phi(a)/s`.  For a square `R=S`, the correlated
piece is `4g2 U_{M circle R,R}`.

If `M,R,S` lie in a Jordan algebra `J`, the first two terms in (35.2) remain
inside `K4(J)`.  Hence the H0+H2 gate has an exact one-matrix/tangent
intertwiner, and its adjoint can use the same local Jordan product already
counted in the D10 schedule.  This is stronger than an approximate
spectral-diffuseness claim.

It is not a proof for the complete gate.  The H4 and higher even components
of `G` act through higher contractions and can enlarge the terminal cyclic
module even when their images stay somewhere inside `Sym^2(J)`.  Odd gate
sectors do not map an even H4 input directly back to H4 at the same centered
reset, but they create odd H3/H5/... descendants that a later correlated
reset can return to the even endpoint.

At zero bias the centered scalar halfspace gate has residual energy after its
H0+H1 projection

\[
\boxed{
\left\|1_{\{X>0\}}-{1\over2}-\varphi(0)X\right\|_2^2
={1\over4}-{1\over2\pi}
=0.090845\ldots,}
\tag{35.3}
\]

which is `36.3%` of its total gate variance.  This number is not itself an
H4 error bound, but it rules out declaring the omitted gate hierarchy small
from scalar energy alone.

Let `P_mod` project onto the retained Jordan H4 module.  The exact immediate
leakage operator is

\[
\boxed{
B_l=(I-P_{\rm mod})\Pi_4M_{G_l}P_{\rm mod}.}
\tag{35.4}
\]

For a complete endpoint observation `O_l`, the future-relevant version is
the restriction of all descendants of `B_l` to `N_l=ker O_l`, as in Section
24.  A frozen-net proof must bound that endpoint-weighted operator, not merely
`||G_tail||2`; multiplication weights the tail by the square of the carried
polynomial and can align it with an unobserved direction.

### Graph split

- `P2R0G19B_MATH_H2_GATE_JORDAN_INTERTWINER` — **solved**: exact identity and first-order closure (35.1)--(35.2).
- `P2R0G19B_MATH_HIGHER_GATE_RETURN` — **open structural gate**: cover H4+ even contractions and odd-to-even future return.
- `P2R0G19B_MATH_GATE_TAIL_ENDPOINT_NORM` — **open frozen-net receipt**: certify the future-weighted version of (35.4); scalar tail energy (35.3) is insufficient.

## 36. Existing provenance summaries do not certify the AQ-score error

Let

\[
U_l=K_{4,l}^{\rm truth}-K_{4,l}^{\rm AQ}.
\]

Its equality-pattern decomposition has the `40,31,22,211,1111` sectors with
their usual multiplicities `1,4,6,12,24`.  If `H` denotes the current host
closure, the exact split is

\[
\boxed{
K_4^{\rm truth}-K_4^{\rm AQ}
=(K_4^{\rm truth}-K_4^{H})
+(K_4^{H}-K_4^{\rm AQ}).}
\tag{36.1}
\]

The banked truth-minus-host `dK4,dK31,dK22` planes observe collision images of
the first bracket.  `dK21` is an H3 observable and does not enter the local
direct H4 score.  An `eta2111` endpoint number controls the first bracket's
all-distinct residual only when its masks, weights, frame, and layer timing
are identical to the proposed score action.  None of these quantities bounds
the second, host-minus-AQ bracket without an ownership/coercivity theorem.

This is an exact identifiability issue.  Let `A` collect every existing
plane/provenance/eta observation of the hidden residual and let `ell_H` be the
coherent D10 H4-score covector.  Then the desired score action is determined
by the summaries if and only if

\[
\boxed{\ell_H\in\operatorname{range}(A^*).}
\tag{36.2}
\]

If (36.2) fails, there is an `h in ker A` with `ell_H(h)!=0`; scaling `h`
leaves every existing summary unchanged while making the missing score action
arbitrarily large.  Coherent endpoint cancellations give the same failure:
small `eta` for a signed sum does not bound the sum of absolute layer actions.

The minimal additional coherent scalar is

\[
\boxed{
\left\langle
(I-P_{\operatorname{range}A^*})\ell_H,
K_4^{\rm truth}-K_4^{\rm AQ}
\right\rangle.}
\tag{36.3}
\]

For the stronger adversarial triangle certificate (31.14), the minimal
nonnegative receipt is

\[
\boxed{
R_{\rm AQ,H}
=\sum_{l=5}^{14}|\beta_l|\|H_l\|_{\rm HS}
\|e_l\|_2,}
\tag{36.4}
\]

which gives

\[
|E_{\rm AQ}^{D10}|\le\sqrt{24}\,R_{\rm AQ,H}.
\tag{36.5}
\]

Together with quadrature and cyclic-module errors, the binding gate is

\[
\boxed{
\sqrt{24}\,R_{\rm AQ,H}
+\sum_l|\beta_l|(\epsilon_{{\rm quad},l}
+\epsilon_{{\rm module},l})
\le3.80\times10^{-6}.}
\tag{36.6}
\]

No current provenance summary supplies either (36.3) or (36.4).  Therefore
the contract-native AQ score remains a mathematically exact local oracle but
an uncertified approximation to the true deep-network action.

### Graph split

- `P2R0G19B_MATH_AQ_SCORE_SUMMARY_IDENTIFIABILITY` — **solved no-go from current summaries**: criterion (36.2).
- `P2R0G19B_MATH_AQ_SCORE_RESIDUAL_OBSERVABLE` — **open acquisition receipt**: obtain (36.3) or the stronger (36.4)--(36.6).

## 37. Tree/H2 source is a score-coordinate descendant, not independent H4 value

Normal ordering itself relates the tied K31 score to the H2 tree score.  With

\[
:A^3B:=A^3B-3v_AAB-3c_{AB}A^2+3v_Ac_{AB},
\]

hold the raw variables and gate law fixed and differentiate only the Wick
covariance coordinates.  Then

\[
\boxed{
\partial_{v_A}:A^3B:=-3:AB:,
\qquad
\partial_{c_{AB}}:A^3B:=-3:A^2:.}
\tag{37.1}
\]

Equivalently, for four independently polarized slots,

\[
\partial_{\operatorname{Cov}(A_1,A_2)}
:A_1A_2A_3B:=-:A_3B:.
\tag{37.2}
\]

The same lowering appears as a mean-translation Hessian; tying three equal
slots accounts for the factors of three/six.  Thus the tree score
`E[F:AB:]` is the covariance-score JVP/translation-Hessian descendant of the
contract-native H4 score.

For lawful first-chaos leaves `A=sigma_j g_j Z_j` and
`B=sigma_k g_k Z_k`, the local ReLU identity gives

\[
E[(m_i+\sigma_iZ_i)_+:AB:]=\beta_i c_{ij}c_{ik},
\tag{37.3}
\]

and cyclic summation produces the signed all-distinct tree source.  Therefore
an implementation that already differentiates the H4 score with respect to
its covariance frame must not count (37.3) again as independent physics.

Let `L_{4to2}` be this lowering map, `Q_S` the tree/H2 costate, and `Q_4` the
H4 costate.  Exact reuse on the lawful provider subspace `P_law` is

\[
\boxed{
P_{\rm law}(Q_4-L_{4\to2}^*Q_S)=0.}
\tag{37.4}
\]

When (37.4) holds, the H4 score derivative replaces the separate tree action;
the two do not stack.  When it fails, only their complete-old-arena Schur
residual may be credited separately.

This statement has a strict scope.  A physical network-parameter derivative
also changes the gate/producer and the score directions:

\[
\boxed{
D_\theta E[F_\theta H_\theta]
=E[(D_\theta F_\theta)H_\theta]
+E[F_\theta D_{\theta,{\rm slots}}H_\theta]
+E[F_\theta D_{\theta,{\rm Wick}}H_\theta].}
\tag{37.5}
\]

Only the last term is the normal-score lowering (37.1).  The first two are
gate/producer coadaptation and must remain in the complete order-two provider
unless separately shown to vanish or be owned.

### Graph split

- `P2R0G19B_MATH_H4_TO_TREE_SCORE_DESCENT` — **solved identity**: (37.1)--(37.3).
- `P2R0G19B_MATH_H4_TREE_DOUBLECOUNT_GATE` — **open ownership equality**: certify (37.4) at the synchronized cut.
- `P2R0G19B_MATH_H4_PHYSICAL_JVP_REMAINDER` — **open completeness gate**: retain the first two terms of (37.5).

## 38. Quantitative higher-gate leakage and two-step chaos return

Use the standardized gate coordinate `X=Q_M/s`, with
`s^2=tr(M^2)/2`, standardized bias `a`, and define the gate residual after
the H0+H2/linear scalar coordinate retained by (35.2):

\[
R_a=1_{\{X+a>0\}}-\Phi(a)-\varphi(a)X.
\tag{38.1}
\]

In the Gaussianized scalar-gate convention its energy is

\[
\eta_a^2=\|R_a\|_2^2
=\Phi(a)(1-\Phi(a))-\varphi(a)^2.
\tag{38.2}
\]

Let `P_mod` project onto the retained spatial Jordan module and
`N_l=ker O_l` be the complete-old observation kernel.  The direct spatial and
future endpoint leakage operators are respectively

\[
\boxed{
L_{\rm mod}=(I-P_{\rm mod})\Pi_4M_{R_a}P_{\rm mod},}
\tag{38.3}
\]

\[
\boxed{
L_{{\rm end},l}
=P_{N_l}\Phi_{L\leftarrow l}\Pi_4M_{R_a}P_{\rm mod},
\quad
L_{{\rm end},l}^*
=P_{\rm mod}M_{R_a}\Pi_4\Phi_{L\leftarrow l}^*P_{N_l}.}
\tag{38.4}
\]

For normalized H4 inputs/costates, degree-four Gaussian hypercontractivity
gives the universal but loose pairing bound

\[
\boxed{
|E[\Lambda R_a U]|
\le81\eta_a\|\Lambda\|_2\|U\|_2.}
\tag{38.5}
\]

At zero bias, `eta_0=sqrt(0.090845...)=0.3014...`, so (38.5) is about
`24.4` and provides no useful certificate against a `1.39%` per-layer D10
gate.  A small unweighted gate tail is not enough; the actual endpoint-weighted
operator (38.4) is required.

The tail is not symbolically zero.  A diffuse-square witness in the H2 sector
obeys

\[
\langle h_2,\Pi_2(R_ah_2)\rangle
=-{a(1+a^2)\varphi(a)\over2},
\tag{38.6}
\]

and the resulting second-order H2 tail word contributes

\[
\boxed{
-{2a\varphi(a)\over s^2}
\left(U_{M^2\circ R,S}+U_{R,M^2\circ S}\right).}
\tag{38.7}
\]

Even when a same-layer parity makes an odd tail invisible to H4, chaos-order
memory returns later.  In one scalar coordinate,

\[
\operatorname{He}_4\operatorname{He}_2
=\operatorname{He}_6+8\operatorname{He}_4+12\operatorname{He}_2,
\]

and the subsequent product `He6 He2` returns to H4 with coefficient

\[
\boxed{\binom62\binom22 2!=30.}
\tag{38.8}
\]

Therefore spatial closure inside a Jordan algebra does not eliminate the
need to account for H6/H8/... memory.  The spatial leak (38.3) vanishes for a
fully certified invariant `K4(J)` (including the required higher contraction
operations), a full commuting polynomial algebra, a common orthogonal
rank-one algebra, or terminal Bellman annihilation.  None of these alone
forces the chaos-order return (38.8) to vanish.

### Graph split

- `P2R0G19B_MATH_GATE_TAIL_OPERATOR` — **solved identity / open receipt**: exact operators (38.3)--(38.4), loose bound (38.5).
- `P2R0G19B_MATH_GATE_TAIL_NONZERO_WITNESS` — **solved**: (38.6)--(38.7).
- `P2R0G19B_MATH_H6_TO_H4_TWO_STEP_RETURN` — **solved obstruction**: coefficient 30 in (38.8); spatial Jordan closure is insufficient.

## 39. Exact chaos decomposition of the AQ H4-score JVP

Let `J=1_{g>0}`, let `H` be a symmetric H4 test tensor, and perturb the AQ
preactivation by

\[
\dot g=\dot c+I_1(h)+I_2(A).
\]

The score JVP is `E[J dot g I4(H)]`.  The Wiener product formula gives the
complete exact decomposition

\[
\boxed{
\begin{aligned}
D\mathcal A_H[\dot g]={}&
\dot c\langle K_4(J),H\rangle\\
&+\langle K_5(J),\operatorname{Sym}(h\otimes H)\rangle
+4\langle K_3(J),H\otimes_1h\rangle\\
&+\langle K_6(J),\operatorname{Sym}(A\otimes H)\rangle
+8\langle K_4(J),H\otimes_1A\rangle
+12\langle K_2(J),H\otimes_2A\rangle.
\end{aligned}}
\tag{39.1}
\]

Here `K_m(J)=partial_c K_m(g_+)` at the synchronized bias coordinate.  Thus
the only signed-H3 ladder term in the AQ score JVP is

\[
4\langle\partial_cK_3(g_+),H\otimes_1h\rangle.
\tag{39.2}
\]

It is the bias derivative of the H3 coefficient, not the H3 coefficient
itself.  Constant and quadratic tangents contain no H3 term by parity; the
linear tangent also contains an independent H5 action.

For the tied score `H=Sym(U^{otimes3} otimes V)`, the H3 contraction has only
the two star directions

\[
\boxed{
4H\otimes_1h
=(h^TV)U^{\otimes3}
+3(h^TU)\operatorname{Sym}(U^{\otimes2}\otimes V).}
\tag{39.3}
\]

Therefore the H4-score JVP can overlap at most this two-dimensional star
slice of a generic signed-H3 provider.  The remaining Jordan/nonlocal H3
directions, and the H5 term in (39.1), require their own complete-future
ownership or error bound.

Equation (39.1) is also the exact no-double-counting interface: project every
term against the synchronized D11 CP3/star/pair arena before assigning novelty.
Owning the local H4 score does not imply ownership of all `K3(J)`, and owning
the old H31 star does not imply ownership of the H5/H6 descendants.

### Graph split

- `P2R0G19B_MATH_AQ_H4_JVP_CHAOS_SPLIT` — **solved**: complete product formula (39.1).
- `P2R0G19B_MATH_AQ_H4_H3_STAR_OVERLAP` — **solved structural limit**: only (39.3) overlaps signed H3.
- `P2R0G19B_MATH_AQ_H4_JVP_HIGHER_DESCENDANTS` — **open completeness gate**: residual H5/H6 and non-star H3 after the old arena.

## 40. Tensor-free commutator-Gram contraction of the 211 costate

Equation (34.1) can be contracted without forming an external fourth-order
tensor.  Let `Lambda_i` be the symmetric future costate slice on labels
`(j,k)` for repeated anchor `i`.  The complete q4-perpendicular scalar is

\[
\boxed{
S_{211}=-8\sum_{i,j,k}(\Lambda_i)_{jk}
\langle[M_i,M_j],[M_i,M_k]\rangle_F.}
\tag{40.1}
\]

Here "q4-perpendicular" is essential: the native H211 identity also has the
commuting old-parallel `48F` term.  Equation (40.1) is the residual only after
that term is acquired at the same all-distinct seam and projected out.  A
collision-only CP4 read is not, by itself, such an acquisition receipt.

### 40.1 Common-factor theorem

Define

\[
\boxed{k_*:=\operatorname{rank}[\Lambda_1\ \Lambda_2\ \cdots\ \Lambda_n].}
\tag{40.2}
\]

This is the minimum dimension of a single external factor space supporting
all anchor slices.  Equivalently, `k_*<=k` if and only if there is one
`U in R^{n by k}` and symmetric `Theta_i` with

\[
\boxed{\Lambda_i=U\Theta_iU^T\quad\text{for every }i.}
\tag{40.3}
\]

Let

\[
N_a=\sum_jU_{ja}M_j,
\qquad C_{ia}=[M_i,N_a].
\]

Then (40.1) becomes

\[
\boxed{
S_{211}=-8\sum_i\sum_{a,b=1}^{k_*}
(\Theta_i)_{ab}\langle C_{ia},C_{ib}\rangle_F.}
\tag{40.4}
\]

The diagonal CP special case `Theta_i=diag(lambda_ia)` is

\[
\boxed{S_{211}=-8\sum_{i,a}\lambda_{ia}
\|[M_i,N_a]\|_F^2.}
\tag{40.5}
\]

The proof is just bilinearity of the commutator; minimality follows because
every shared preaggregation `N_a=sum_j U_ja M_j` restricts every `Lambda_i`
to `range U`, while (40.2) is the smallest common range.  Separate low-rank
eigendecompositions `Lambda_i=U_iTheta_iU_i^T` do not help when the `U_i`
differ with `i`.

### 40.2 Exact reverse gradient

For the diagonal form, write `N_{ia}=sum_j u_{ia,j}M_j` if factors are allowed
to depend on the anchor.  Holding the costate factors fixed, the exact
gradient is

\[
\boxed{
\begin{aligned}
\nabla_{M_p}S_{211}=-16\{&
\sum_a\lambda_{pa}[[M_p,N_{pa}],N_{pa}]\\
&+\sum_{i,a}\lambda_{ia}u_{ia,p}
[M_i,[M_i,N_{ia}]]\}.
\end{aligned}}
\tag{40.6}
\]

For the common factors `u_{ia}=u_a`, define

\[
H_a=\sum_i\lambda_{ia}[M_i,[M_i,N_a]],
\]

so the second line fuses to `sum_a u_{a,p}H_a`.  Thus scalar acquisition and
joint-refit pullback use the same commutator bank; no H4 tensor is required.

### 40.3 Cost and recurrence gate

With a common `U`, forming every `N_a` costs `O(k n r^2)=O(kn^2)` at
`r^2 asymp n`; the commutators cost `O(nkr^3)` per layer.  In dense-`n` GEMM
equivalents, the leading local cost across `D` layers is approximately

\[
\boxed{C_{211,\rm local}\asymp {Dk r^3\over n^2},}
\tag{40.7}
\]

before constants, gradient work, and any reuse.  At `n=1024,D=10`, this is
about `0.87k` at `r=45` and `1.59k` at `r=55`; hence at r55 even one additive
unreused common factor exceeds the one-equivalent allowance.  It can remain
live only if these Jordan/commutator products are literal resident work in the
D10 replacement schedule or displace an existing billed provider.

If the factors depend on the anchor, forming `N_{ia}` requires `nk` separate
mixes of the `n` cores, `O(kn^2r^2)=O(kn^3)` per layer.  That generic branch is
economically dead before the multi-layer reverse pass.

A common factor is preserved by a true common-leg map:
`U -> K^T U`.  It is also preserved by a terminal CP word whose every external
leg undergoes that same map.  A neuron-specific correlated-ReLU pullback does
not generically intertwine this way and makes the slice spaces anchor
dependent.  Therefore the current abstract statement "one D10 word" is not
a common-factor receipt.  The exact per-layer certificate is

\[
\boxed{\Lambda_{i,l}=U_l\Theta_{i,l}U_l^T
\quad\forall i,\qquad k_{*,l}\text{ within the billed cap},}
\tag{40.8}
\]

together with an explicit common-leg recurrence for `U_l`.  Otherwise the
generic anchor-dependent cost applies.

### 40.4 Zero-new-product antisymmetric fusion

The leading local cost in (40.7) disappears when the existing D10 Jordan
update literally computes the same full products.  For symmetric `M,N`, let

\[
P=MN,\qquad J={P+P^T\over2}=M\circ N,
\qquad A={P-P^T\over2}={[M,N]\over2}.
\]

Symmetric/skew orthogonality gives the exact identity

\[
\boxed{
\langle[M,N_a],[M,N_b]\rangle_F
=4\langle A_a,A_b\rangle_F
=4(\langle P_a,P_b\rangle_F-\langle J_a,J_b\rangle_F).}
\tag{40.9}
\]

Hence once the billed local Jordan product has formed `P_a=M N_a`, retaining
its antisymmetric half and reducing the small Gram costs no additional dense
matrix product.  In the rank-one costate case,

\[
\boxed{q_{4,\perp}=-32\lambda\|A\|_F^2.}
\tag{40.10}
\]

This is the concrete deployment loophole for (40.7).  It applies if and only
if:

1. the resident D14 core is exactly `M_i` and the resident D10 suffix is the
   exact common-factor combination `N_a` at the same reset seam;
2. the implementation forms or can retain the full `P_a` before
   symmetrization, rather than exposing only a separately computed `J_a`;
3. the q211 costate slices satisfy the common-factor receipt (40.8); and
4. the antisymmetric reductions and their f32 error fit the non-GEMM bill.

It does not rescue anchor-dependent `N_{ia}`: their formation remains the
generic `O(kn^3)` obstruction.  But for a literal rank-one D10 common suffix,
(40.9) changes the commutator action from an additive-cost no-go into a
zero-new-GEMM candidate.

### Graph split

- `P2R0G19B_MATH_Q211_COMMON_FACTOR_THEOREM` — **solved**: (40.2)--(40.5).
- `P2R0G19B_MATH_Q211_COMMUTATOR_REVERSE` — **solved local gradient**: (40.6).
- `P2R0G19B_MATH_Q211_COMMON_FACTOR_RECEIPT` — **open D10 structural gate**: prove (40.8); one-word billing is insufficient.
- `P2R0G19B_MATH_Q211_R55_ADDITIVE_COST` — **solved current arithmetic no-go unless reused/replacement**: (40.7).
- `P2R0G19B_MATH_Q211_JORDAN_ANTISYM_FUSION` — **conditional high-priority loophole**: zero-new-product identity (40.9)--(40.10) under the four literal-reuse receipts.

## 41. Exact common-factor recurrence for the 211 reverse costate

Let the reverse update on the external `211` costate have the general
anchor-mixing/pair-leg form

\[
\boxed{
\Lambda'_{p}=\sum_i A_{pi}B_i^T\Lambda_iB_i.}
\tag{41.1}
\]

Here `A` mixes the repeated anchor label and `B_i` is the pair-leg pullback at
anchor `i`.  If every input slice has one common support

\[
\Lambda_i=U\Theta_iU^T,
\]

then every output column lies in

\[
\boxed{
\mathcal U'=\operatorname{span}_{i}\operatorname{range}(B_i^TU).}
\tag{41.2}
\]

Conversely, for generic live anchor weights in `A`, (41.2) is the smallest
common support capable of representing all output slices.  Hence

\[
\boxed{k'=\dim\operatorname{span}_{i}B_i^T\mathcal U}
\tag{41.3}
\]

is the exact next-cut common-factor width.

Consequences:

- anchor mixing `A` alone does not enlarge the common pair space;
- a common pair-leg map `B_i=B` preserves `k` (up to null directions);
- for a rank-one terminal factor `u`, rank one survives if and only if
  `B_i^Tu` are collinear for every live `i`;
- generic neuron-specific `B_i` make the span in (41.3) width-size after one
  reset.

Thus the frozen common-scalar/OU/common-leg recurrence can preserve the
rank-one antisymmetric-fusion route, while a self-consistent correlated ReLU
requires an explicit collinearity or small-span theorem.  A terminal CP word
and one billed reverse stream do not imply (41.3).

### Approximate common support and the exact supply-weighted gate

For the realized NCQC cores define the commutator-Gram supply slice

\[
(G_i)_{jk}=\langle[M_i,M_j],[M_i,M_k]\rangle_F.
\tag{41.4}
\]

If a common-support approximation uses `hat Lambda_i`, its exact scalar error
is

\[
\boxed{
E_{211}=-8\sum_i\langle\Lambda_i-\widehat\Lambda_i,G_i\rangle_F.}
\tag{41.5}
\]

The universal Frobenius certificate is

\[
\boxed{
|E_{211}|
\le8
\left(\sum_i\|\Lambda_i-\widehat\Lambda_i\|_F^2\right)^{1/2}
\left(\sum_i\|G_i\|_F^2\right)^{1/2}.}
\tag{41.6}
\]

The exact endpoint-weighted contraction (41.5) is preferable and can be much
smaller.  Any D10 truncation must accumulate its layerwise version with the
route weights and keep the total, together with other acquisition errors,
below `3.80e-6`.  A small unweighted singular tail of the `Lambda_i` is not a
value certificate if the discarded directions align with `G_i`.

### Graph split

- `P2R0G19B_MATH_Q211_COMMON_FACTOR_RECURRENCE` — **solved theorem**: exact width update (41.1)--(41.3).
- `P2R0G19B_MATH_Q211_RANK1_COLLINEARITY` — **open high-priority frozen-net algebra gate**: certify all `B_i^Tu` are collinear at every D10 cut.
- `P2R0G19B_MATH_Q211_SUPPLY_WEIGHTED_TAIL` — **open endpoint receipt**: certify (41.5), with (41.6) as the safe fallback.

## 42. Transient Z22 seam: free full action only with literal Q22 ownership

Let `Z_l` be the transient K22 residual at the exact post-reset,
pre-compression seam.  If the already billed Bellman state has a literal
complete coordinate

\[
Q_{22,l}={\partial V\over\partial K_{22,l}},
\]

then its complete immediate action is simply

\[
\boxed{\Delta V_{Z,l}=\langle Q_{22,l},Z_l\rangle_F.}
\tag{42.1}
\]

This is `O(n^2)` reduction work and needs no `Z(W odot W)` product.  A formula
such as `3<ZX,Y>` reconstructs only the K31-induced factorized component of
`Q22`; it is unnecessary when the full row in (42.1) is truly resident.

The current D11 lineage does not supply that novel row.  `_reset_stars` fits

\[
\alpha={\langle K_{22}^{\rm residual},P_{\rm pair}\rangle
\over\|P_{\rm pair}\|_F^2}
\]

and retains only the pairing direction `P_pair`.  By construction the
transient residual

\[
Z=K_{22}^{\rm residual}-\alpha P_{\rm pair}
\]

is orthogonal to the old retained Q22 direction.  Therefore

\[
\boxed{\langle Q_{22}^{\rm D11,old},Z\rangle_F=0.}
\tag{42.2}
\]

The proposed D10 word is still an abstract q4-perpendicular costate, not a
concrete full-state reverse tape at this seam.  Consequently there are two
lawful cases:

1. Redefine/certify D10 on the full product state and expose the literal
   complete `Q22`.  Then (42.1) is free and includes crossed K22 completion.
2. Expose only Q31 or factorized rows.  Then use their factor contraction and
   separately compute/bound every omitted Q22 component; no full-interface
   credit follows from the K31-induced piece alone.

### Graph split

- `P2R0G19B_MATH_Z22_TRANSIENT_SEAM_ACTION` — **solved theorem**: (42.1).
- `P2R0G19B_MATH_D11_OLD_Q22_Z_ORTHOGONAL` — **solved duplicate-zero identity**: (42.2).
- `P2R0G19B_MATH_D10_Q22_LITERAL_OWNERSHIP` — **open implementation receipt**: full row free only if exposed at the exact seam.

## 43. D11 CP4 already owns the universal affine-ReLU H4 atom

For normalized centered affine ReLU, the scalar H4 coefficient is
`-1/sqrt(pi)` and the two-unit pure-H4 kernel slice is

\[
\kappa_4(\rho)={\rho^4\over24\pi}.
\tag{43.1}
\]

This is not a new direct-score feature in the deployed D11 lineage.  At each
reset, `_cp4_reset`:

1. gates every inherited external rank-one factor;
2. computes `birth_lambda=k4_target-inherited_d`; and
3. appends the coordinate factor `e_i` for every new marginal birth.

`_cp4_read_planes` then reads each retained `lambda u^{otimes4}` into its
K4/K31/K22 collision planes, and `_quartic_residual` subtracts those provider
planes before the star/pair fit.  Therefore the coordinate affine-ReLU atom
(43.1) is represented exactly at birth, subject only to the explicit CP4
capacity truncation and f32 storage.

Consequently an additional local affine H4 score is a duplicate after
synchronized CP4/star/pair residualization.  Novel H4 can begin only in:

- CP4-dropped components not subsequently captured by star/pair;
- identity-specific quadratic/noncommutative structure outside the retained
  external rank-one span, especially the off-diagonal q4-perpendicular `211`
  commutator Gram; or
- a higher-chaos future-return path not already represented by the H4
  provider.

The scalar H6 energy term (for affine ReLU, `1/(80 pi)` in the referenced
normalization) is not itself an H4 CP4 atom, but it earns no value until its
future return is residualized against the complete old provider.

### Graph split

- `P2R0G19B_MATH_CP4_AFFINE_H4_OWNED` — **solved duplicate audit**: universal rank-one affine atom is already in CP4.
- `P2R0G19B_MATH_DIRECT_H4_NOVELTY_DOMAIN` — **open residual lane**: only CP4-drop, identity-specific 211, or complete-old-residual higher return may count.

## 44. Exact H4 gate compression and the off-diagonal 211 no-go

Let `Z` be standard Gaussian, let

\[
F(Z)={\bf1}_{\{S(Z)>0\}},\qquad
S=b+a^TZ+Z^TMZ-\operatorname{tr}M,
\qquad p=E F,
\]

and let `A_F` denote multiplication by `F`.  For a symmetric tensor `T`, use
the Wiener normalization

\[
E[I_q(T)I_q(U)]=q!\langle T,U\rangle.
\]

Define the Hermite moment tensor `C_q` by

\[
\langle C_q,V\rangle=E[(F-p)I_q(V)].
\]

For unit vectors `I_4(T),I_4(U)` in fourth chaos,
`||T||=||U||=1/sqrt(24)`.  The product formula is

\[
\boxed{
I_4(T)I_4(U)
=I_8(T\widetilde\otimes U)
+16I_6(T\widetilde\otimes_1U)
+72I_4(T\widetilde\otimes_2U)
+96I_2(T\widetilde\otimes_3U)
+24\langle T,U\rangle.}
\tag{44.1}
\]

Consequently the exact compressed reset error is

\[
\boxed{
\begin{aligned}
\langle I_4(T),(P_4A_FP_4-pI)I_4(U)\rangle
={}&96\langle C_2,T\widetilde\otimes_3U\rangle
+72\langle C_4,T\widetilde\otimes_2U\rangle\\
&+16\langle C_6,T\widetilde\otimes_1U\rangle
+\langle C_8,T\widetilde\otimes U\rangle.
\end{aligned}}
\tag{44.2}
\]

Thus fourth-chaos multiplication is not controlled by one common scalar
unless all four even gate tensors `C2,C4,C6,C8` vanish on the live contraction
images.  Tensor contraction and symmetrization are norm nonincreasing, giving

\[
\boxed{
\|P_4A_FP_4-pI\|_{op}
\le \min\left\{\max(p,1-p),
4\|C_2\|+3\|C_4\|+{2\over3}\|C_6\|+{1\over24}\|C_8\|\right\}.}
\tag{44.3}
\]

If

\[
W_q=\|P_qF\|_2^2={\|C_q\|^2\over q!},
\]

then

\[
\boxed{
\|P_4A_FP_4-pI\|_{op}
\le4\sqrt2\sqrt{W_2}+6\sqrt6\sqrt{W_4}
+8\sqrt5\sqrt{W_6}+\sqrt{70}\sqrt{W_8}.}
\tag{44.4}
\]

In particular this is at most
`sqrt(638(W2+W4+W6+W8))`.  For orthonormal multivariate Hermite
coefficients, define

\[
\operatorname{Inf}^{\rm even\le8}_i(F)
=\sum_{q\in\{2,4,6,8\}}\sum_{|\alpha|=q}
\alpha_i\widehat F_\alpha^2.
\]

Since

\[
\sum_i\operatorname{Inf}^{\rm even\le8}_i(F)
=\sum_{q\in\{2,4,6,8\}}qW_q=:I_{\rm even},
\]

Cauchy--Schwarz sharpens (44.4) to

\[
\boxed{
\|P_4A_FP_4-pI\|_{op}
\le \sqrt{1585/12}\,\sqrt{I_{\rm even}}.}
\tag{44.5}
\]

A maximum coordinate influence `delta` gives only
`sqrt((1585/12) r delta)`.  The natural diffuse scale `delta=O(1/r)`
therefore does not make the operator error vanish.

For a linear score `S=G-t`, the complete spectrum is also explicit.  Fourth
chaos decomposes by the number `k=0,...,4` of legs along `G`, and the
corresponding eigenvalue is

\[
\lambda_k(t)={1\over k!}\int_t^\infty H_k(x)^2\phi(x)\,dx.
\]

The five deviations from `p` are

\[
\boxed{
0,\quad t\phi(t),\quad {t^3+t\over2}\phi(t),\quad
{t^5-t^3+6t\over6}\phi(t),\quad
{t^7-5t^5+17t^3+15t\over24}\phi(t).}
\tag{44.5a}
\]

At `t=1` the full H4 norm error is `7 phi(1)/6`, while the leakage rank is
`binom(r+2,3)=Theta(r^3)`.  A single diffuse score direction therefore
induces a high-rank fourth-chaos reset error.

### 44.1 Exact quadratic-score cavity formula

Put

\[
g=\nabla S=a+2MZ,\qquad Q=\nabla^2S=2M.
\]

Gaussian integration by parts and the partition form of Faa di Bruno give,
in the distributional sense,

\[
\boxed{
C_q=\sum_{j=0}^{\lfloor q/2\rfloor}
{q!\over2^jj!(q-2j)!}\operatorname{Sym}E\!\left[
\delta^{(q-j-1)}(S)Q^{\otimes j}\otimes
g^{\otimes(q-2j)}\right].}
\tag{44.6}
\]

It is a classical identity when the corresponding weighted score densities
have the displayed derivatives, and otherwise follows by mollifying the
Heaviside function and taking a weak limit.  The new H4 endpoint term is

\[
\boxed{
\begin{aligned}
C_8=E\{&\delta^{(7)}(S)g^{\otimes8}
+28\delta^{(6)}(S)\operatorname{Sym}(Q\otimes g^{\otimes6})\\
&+210\delta^{(5)}(S)\operatorname{Sym}(Q^{\otimes2}\otimes g^{\otimes4})
+420\delta^{(4)}(S)\operatorname{Sym}(Q^{\otimes3}\otimes g^{\otimes2})\\
&+105\delta^{(3)}(S)\operatorname{Sym}(Q^{\otimes4})\}.
\end{aligned}}
\tag{44.7}
\]

There is also an exact one-coordinate cavity recursion.  Write `Z=xv+y` and

\[
q_y(x)=\beta x^2+\alpha(y)x+\gamma(y),\qquad
\beta=v^TMv,\quad \alpha(y)=a^Tv+2v^TMy.
\]

For probabilists' Hermite polynomials `H_k`, `0<=k,l<=4`, and simple real
roots `rho` of `q_y`, let `p_y=P_x(q_y(x)>0)`.  Then

\[
\boxed{
\begin{aligned}
&\int {\bf1}_{\{q_y(x)>0\}}H_k(x)H_l(x)\phi(x)\,dx
=k!\,\delta_{kl}p_y+R_{kl}(y),\\
&R_{kl}(y)=
\sum_{\substack{0\le s\le\min(k,l)\\m=k+l-2s>0}}
s!{k\choose s}{l\choose s}
\sum_{q_y(\rho)=0}\operatorname{sgn}(q_y'(\rho))
\phi(\rho)H_{m-1}(\rho).
\end{aligned}}
\tag{44.8}
\]

Thus a cavity step consists exactly of lower-dimensional multiplication by
`p_y-p` plus the boundary matrix `R(y)`.  Small entries of `a` or `M` do not
by themselves bound either term; the needed premise is an averaged boundary
leverage or, equivalently for (44.2), small total even Hermite mass.

### 44.2 The off-diagonal 211 quotient retains every even leakage level

In a fixed external basis define

\[
\mathcal H_{211}^{\rm off}
=\operatorname{span}\{H_2(Z_i)Z_jZ_k: i,j,k\ {\text{distinct}},\ j<k\},
\]

and let `Q211` be its orthogonal projection.  If a complete-old subtraction
has also been certified, replace this space throughout by its orthogonal
residual subspace.  For

\[
\mathcal D_q^{211}
=\operatorname{span}\{T\widetilde\otimes_{(8-q)/2}U:
T,U\in\mathcal H_{211}^{\rm off}\},
\]

(44.2) remains exact with `C_q` replaced by its projection onto
`D_q^211`.  Hence (44.3)--(44.5) remain valid, and can only be sharpened by
using those projected norms.

The quotient does not remove any of the levels `2,4,6,8`.  Indeed the unit
211 vector

\[
h={H_2(Z_0)\over\sqrt2}Z_1Z_2
\]

obeys

\[
\boxed{
h^2={1\over2}\bigl(H_4(Z_0)+4H_2(Z_0)+2\bigr)
\bigl(H_2(Z_1)+1\bigr)\bigl(H_2(Z_2)+1\bigr),}
\tag{44.9}
\]

whose projections at total degrees `0,2,4,6,8` are all nonzero.  Therefore
collision removal is not a chaos-diagonalization theorem for the reset.

The failure persists under genuine diffuseness.  Let `G` be the normalized
sum of `n` coordinates disjoint from `Z0,Z1`, set

\[
F={\bf1}_{\{G>t\}},\qquad
h={H_2(Z_0)\over\sqrt2}Z_1G.
\]

Then `h` is a unit vector in the off-diagonal 211 space and

\[
\boxed{
\langle h,(Q_{211}A_FQ_{211}-pI)h\rangle
=E[{\mathbf 1}_{\{G>t\}}G^2]-p=t\phi(t).}
\tag{44.10}
\]

Every score coordinate has weight `1/sqrt(n)`.  Nevertheless at `t=1` the
operator error is at least `phi(1)`, approximately `0.242`.  This is a
distribution-free no-go for sector projection or diffuseness alone; it does
not assert that this particular vector survives the deployed complete-old
quotient.  An old-span annihilation of the witness must be proved separately.

There is an exact quotient-dimension refinement.  On a block of `m`
coordinates disjoint from the support of `G`, let

\[
\mathcal U=\operatorname{span}\{H_2(X_i)X_j:i\ne j\},
\qquad \mathcal W=G\mathcal U.
\]

Then `dim W=m(m-1)`, `W` lies in the off-diagonal 211 sector, and the linear
half-space compression acts on all of `W` by the same eigenvalue
`p+t phi(t)`.  Hence for any certified old space `O`,

\[
\boxed{
\dim(\mathcal W\cap\mathcal O^\perp)>0
\quad\Longrightarrow\quad
\|P_{\mathcal O^\perp}Q_{211}A_FQ_{211}P_{\mathcal O^\perp}-pI\|_{op}
\ge |t\phi(t)|.}
\tag{44.10a}
\]

In particular `dim O<m(m-1)` is sufficient for a surviving witness.  Thus a
complete-old quotient can close the example only by an actual span/rank
receipt, not by diffuseness language.

There is a pure-quadratic version.  On a disjoint block put

\[
Q_n={1\over\sqrt{2n}}\sum_{j=1}^n(Z_j^2-1),\qquad
F={\bf1}_{\{Q_n>t\}},\qquad h=X Y Q_n,
\]

where `X,Y` are independent coordinates outside the block.  Expanding `Q_n`
shows `h` lies entirely in the off-diagonal 211 sector.  Its norm is one and

\[
\boxed{
\langle h,(Q_{211}A_FQ_{211}-p_nI)h\rangle
=E[{\mathbf 1}_{\{Q_n>t\}}Q_n^2]-p_n
\longrightarrow t\phi(t).}
\tag{44.11}
\]

Here the quadratic score has unit variance,
`||M||op/||M||F=1/sqrt(n)`, and maximum low-degree Hermite influence
`O(1/n)`.  Thus both spectral and Hermite diffuseness fail to certify a
common-scalar 211 reset.

### 44.3 Stable rank cannot certify rank-one common-factor recurrence

The rank-one condition in (41.3) is literal collinearity of `B_i^Tu`.  It is
not a consequence of stable rank, flat singular values, or entrywise
diffuseness.  For a Hadamard order `m`, let `H` be a normalized Hadamard
matrix, let `P_i` be row permutations, put

\[
B_i=P_iH,\qquad u=e_1,
\]

and choose the permutations so that `P_i^Tu=e_i`.  Then every `B_i` is
orthogonal, has stable rank `m`, and every entry has magnitude `1/sqrt(m)`,
but

\[
\boxed{B_i^Tu=H^Te_i,\qquad
\langle B_i^Tu,B_j^Tu\rangle=\delta_{ij}.}
\tag{44.12}
\]

Therefore one rank-one terminal factor becomes common-factor width `m` in a
single reset.  The best common rank-one approximation to the matrix with
these vectors as columns has relative Frobenius error
`sqrt(1-1/m)`.  This is a maximally diffuse, maximally stable-rank
counterexample, not a concentrated pathology.

Even symmetric positive diagonal gates exhibit exact rank explosion.  With
`u=1/sqrt(m)` and

\[
B_i=I+\eta e_ie_i^T,
\]

each `B_i` has stable rank of order `m`, while
`B_i^Tu=u+eta e_i/sqrt(m)` span all of `R^m` for every nonzero `eta`.
Approximate compression is governed by the endpoint-weighted tail, not by
this stable rank.

For the exact q211 action (41.5), the necessary endpoint receipt remains

\[
\boxed{
8\left|\sum_{l,i}w_l
\langle\Lambda_{i,l}-\widehat\Lambda_{i,l},G_{i,l}\rangle_F\right|
\le3.80\times10^{-6}.}
\tag{44.13}
\]

The Hadamard witness permits a supply aligned with the discarded common-factor
tail, so no stable-rank-only upper bound on (44.13) exists.  In normalized
Frobenius units, the safe Cauchy certificate requires the product of costate
tail norm and supply norm to be at most
`(3.80e-6)/8=4.75e-7`, before other error allocations.

Likewise (44.10) shows that a scalar-reset replacement with `t=1` can meet
the full endpoint allowance only if the endpoint-weighted incoming 211
amplitude is at most

\[
\boxed{{3.80\times10^{-6}\over\phi(1)}\simeq1.57\times10^{-5},}
\tag{44.14}
\]

again before summing coherent layers or other acquisition errors.  Neither
score diffuseness nor map stable rank supplies that amplitude certificate.

### Graph split

- `P2R0G19B_MATH_H4_GATE_COMPRESSION_LEVELS` — **solved theorem**: exact levels `2/4/6/8`, (44.1)--(44.8).
- `P2R0G19B_MATH_H211_GATE_SCALARIZATION_NOGO` — **solved no-go**: the off-diagonal quotient retains all levels and has diffuse linear and quadratic witnesses, (44.9)--(44.11).
- `P2R0G19B_MATH_Q211_STABLERANK_COLLINEARITY_NOGO` — **solved no-go**: maximal stable rank and entrywise diffuseness can still produce full common-factor width, (44.12).
- `P2R0G19B_MATH_Q211_ENDPOINT_WEIGHTED_RESET_GATE` — **open frozen-net receipt**: certify (44.13), and charge scalar-reset leakage against (44.14).

## 45. Generic commutator-Gram rank is essentially full width

The rank-one fusion in Section 40 is a target-conditioned scalar route, not a
complete representation of the q211 supply.  Let

\[
\mathcal V=\operatorname{span}\{M_1,\ldots,M_n\}
\subseteq\operatorname{Sym}_r
\]

and, for fixed anchor `i`, define

\[
C_i:\mathbb R^n\to\operatorname{Skew}_r,
\qquad C_i u=[M_i,\sum_j u_jM_j].
\tag{45.1}
\]

Then the commutator Gram slice is exactly

\[
\boxed{G_i=C_i^*C_i,}
\tag{45.2}
\]

so

\[
\boxed{
\operatorname{rank}G_i
=\dim\mathcal V-dim(\mathcal V\cap Z_{\rm sym}(M_i)),}
\tag{45.3}
\]

where `Z_sym(M_i)` is the symmetric centralizer of `M_i`.

For a simple-spectrum symmetric anchor, `dim Z_sym(M_i)=r`.  Condition on the
mandatory shared direction `M_i in V cap Z_sym(M_i)` and otherwise place the
`n`-dimensional core span generically in
`d=r(r+1)/2` dimensions.  The generic intersection dimension is

\[
1+\max(0,n+r-d-1),
\]

and hence

\[
\boxed{
\operatorname{rank}G_i
=\min(n-1,d-r)
=\min\!\left(n-1,{r(r-1)\over2}\right).}
\tag{45.4}
\]

At `n=1024`, this is

\[
\boxed{990\quad(r=45),\qquad 1023\quad(r=55).}
\tag{45.5}
\]

The forced null direction is `u=e_i`, since `[M_i,M_i]=0`; this is the same
algebraic reason the K31 restriction of q4-perpendicular vanishes.

Thus exact complete q211 coverage generically needs nearly width-many external
factor directions.  A rank-one D10 suffix computes one lawful scalar
`u^TG_i u` cheaply via (40.9), but earns full-provider credit only if the
endpoint costate/value is concentrated in that direction.

If the nonzero eigenvalues of `G_i` have condition ratio
`kappa_G=lambda_max/lambda_min`, the maximum fraction of trace energy any
rank-`k` subspace can capture is bounded by

\[
\boxed{
{k\kappa_G\over(q-k)+k\kappa_G},
\qquad q=\operatorname{rank}G_i,}
\tag{45.6}
\]

and the analogous Frobenius-energy fraction uses `kappa_G^2`.  In the diffuse
case these are approximately `k/q`; no small-`k` completeness is possible.
The actual campaign gate must use the task-weighted contraction (41.5), since
no label-free condition-number bound is currently available.

### Graph split

- `P2R0G19B_MATH_Q211_GENERIC_RANK` — **solved structural theorem**: (45.3)--(45.5).
- `P2R0G19B_MATH_Q211_RANK1_IS_SCALAR_ONLY` — **solved scope limit**: antisymmetric fusion does not imply complete provider coverage.
- `P2R0G19B_MATH_Q211_TASK_CONCENTRATION` — **open value gate**: show the future costate/supply is concentrated in cap-compatible common factors via (41.5).

## 46. Forward autonomous H4 closure violates the billing cap

The target-conditioned reverse quotient is not merely a convenient design.
A uniform autonomous forward H4 carrier over the width-size core space has a
rigorous rank obstruction.

Represent the `m=1024` independent core directions by their leading monomial
supports.  Exact square closure must distinguish their pairwise support sums.
The core support has affine dimension at least `44`; Freiman's dimension
bound therefore gives

\[
\boxed{
\operatorname{rank}\mu_4
\ge |A+A|
\ge(44+1)m-{44\cdot45\over2}
=45{,}090.}
\tag{46.1}
\]

Transporting `R=45,090` autonomous channels through one dense width-1024 map
costs at least

\[
\boxed{{R\over n}={45{,}090\over1{,}024}=44.033\ldots}
\tag{46.2}
\]

full GEMM equivalents.  Adding the at-least-nine core/reset operations puts
the route above the hard `49` limit before any complete K22/211 readout or
error control.

Thus a noncommuting complete H4 provider cannot be carried forward within the
current budget.  The only mathematical escape is a target-conditioned reverse
cyclic quotient whose **actual** future rank is small, such as the conditional
rank-one common-factor/antisymmetric fusion in Sections 40--41.  Generic
square closure or a nominal low local tensor rank is insufficient.

### Graph split

- `P2R0G19B_MATH_FORWARD_H4_FREIMAN_NOGO` — **solved arithmetic/structural no-go**: (46.1)--(46.2).
- `P2R0G19B_MATH_REVERSE_CYCLIC_ONLY_ESCAPE` — **conditional conclusion**: must certify actual task-conditioned cyclic rank and value tail.

## 47. Complete Q22+Q211 sewing and old-quotient common rank

The off-diagonal `211` slice cannot be compressed independently of its K22
collision boundary.  After applying the binding factorial/collision
normalization, sew the off-diagonal Q211 costate and the canonical Q22
diagonal into one symmetric slice `Lambda_i` for every repeated anchor.

The relevant common-factor width after old-provider subtraction is

\[
\boxed{
k_*^\perp=min_{X_i\in\mathcal C_{\rm old}}
\operatorname{rank}
[\Lambda_1-X_1\ \cdots\ \Lambda_n-X_n].}
\tag{47.1}
\]

This is the correct rank gate, rather than the raw rank before CP4/star/pair
residualization.

The full Q22+Q211 action shares one literal D10 suffix `u` if and only if

\[
\boxed{
\Lambda_i=\theta_i uu^T\quad\text{for every anchor }i,}
\tag{47.2}
\]

`u` is the actual resident suffix, and every reverse reset preserves its
collinearity as in (41.3).  Equation (47.2) includes the collision sewing
condition

\[
(\Lambda_i)_{jj}=\theta_i u_j^2.
\tag{47.3}
\]

An off-diagonal rank-one Q211 fit does **not** give a complete action when the
Q22 diagonal fails (47.3).

Under the paid product `P_i=M_iN`, `N=sum_j u_jM_j`, the complete sewn
**q4-perpendicular** action is

\[
\boxed{
S_{22+211}=-32\sum_i\theta_i\|\operatorname{skew}P_i\|_F^2.}
\tag{47.4}
\]

For `k` common factors, let `K_i^A` be the small Gram of the retained skew
products.  Then

\[
\boxed{S_{22+211}=-32\sum_i\operatorname{tr}(\Theta_iK_i^A).}
\tag{47.5}
\]

Both are zero-new-GEMM reductions under the literal product-reuse receipts.
They do not include the native commuting `48F` parallel.  Calling (47.4) or
(47.5) the full raw H211 action additionally requires a literal synchronized
old-provider acquisition/subtraction of that parallel; CP4 collision reads
alone do not establish it.

Let `s_k` denote the post-old, task-weighted endpoint signal amplitude of the
rank-`k` sewn action and `eta_k` its acquisition/model/truncation error.  The
minimum final-closer promotion gate is

\[
\boxed{s_k\ge\eta_k+\sqrt{2.86\times10^{-10}}
=\eta_k+1.691\times10^{-5},}
\tag{47.6}
\]

while the adversarial D10 implementation/model error must separately remain
at most `3.80e-6`.  These two thresholds answer different questions and must
not be interchanged.

### Graph split

- `P2R0G19B_MATH_Q22_Q211_SEWING` — **solved completeness theorem**: (47.2)--(47.5).
- `P2R0G19B_MATH_Q211_48F_OLD_PARALLEL` — **open ownership receipt**: acquire/subtract the native all-distinct commuting parallel before using the q4-perpendicular formulas.
- `P2R0G19B_MATH_Q22_Q211_OLD_QUOTIENT_RANK` — **open frozen-net algebra receipt**: compute/certify (47.1).
- `P2R0G19B_MATH_Q22_Q211_ROBUST_PROMOTION` — **open value receipt**: (47.6) plus the `3.80e-6` D10 error gate.

## 48. Full H3/q211 seam fusion needs a larger product closure

The paid full product

\[
P_{ia}=M_iN_a
\]

contains two useful halves simultaneously:

\[
J_{ia}=\operatorname{sym}P_{ia}=M_i\circ N_a,
\qquad
A_{ia}=\operatorname{skew}P_{ia}={[M_i,N_a]\over2}.
\tag{48.1}
\]

The symmetric half feeds the ordinary H3/Jordan pullback, while the skew half
feeds the q211 commutator action.  Thus both are zero-new-product local
reductions when they use the same resident `M_i,N_a` at the same seam.

This does not make the complete H3 seam free automatically.  The exact local
query is

\[
\boxed{
G_3=L_4^*Q_4+L_{31}^*Q_{31}+L_{22}^*Q_{22}
+L_{211}^*\Lambda_{211}+P_3^*q_3.}
\tag{48.2}
\]

The independent off-diagonal q211 term is generically outside the collision
plane span.  There is, however, an important ordered-product reuse identity
that removes one apparent obstruction.  Since

\[
N_a=\sum_i U_{ia}M_i,
\qquad P_{ib}=M_iN_b,
\]

the full (not merely symmetrized) resident products recover

\[
\boxed{
N_aN_b=\sum_i U_{ia}P_{ib}.}
\tag{48.3}
\]

No commutativity is used in (48.3); the reverse ordering is obtained by
transposition.  If the implementation retains only `sym(P_ib)`, it recovers
only `sym(N_aN_b)` and not the ordered word needed by a general pullback.

The genuine unresolved collision object is instead

\[
\boxed{
D_{ab}:=\sum_i U_{ia}U_{ib}M_i^2.}
\tag{48.4}
\]

For example, the anchored all-distinct ordered sum is exactly

\[
\begin{aligned}
&\sum_{\substack{j,k\ne i\\j\ne k}}
 U_{ja}U_{kb}M_jM_k \\
&\quad=N_aN_b-U_{ia}M_iN_b-U_{ib}N_aM_i
 +2U_{ia}U_{ib}M_i^2-D_{ab}.
\end{aligned}
\tag{48.5}
\]

Thus the factor-factor word itself is free once full `P_ib` is resident.
What is not free generically is deleting the original-label diagonal, or an
equivalent bank of the required `M_i^2`/`D_ab` collision corrections.
Masking that diagonal after factorization generally destroys the common
support `U`.

For the commutator q211 scalar, the `j=i` and `k=i` terms vanish because
`[M_i,M_i]=0`; the `j=k` term does not vanish and is precisely the diagonal
q22 collision.  Consequently the complete sewn Q22+q211 slice of Section 47
may retain that diagonal and use the full factor product, whereas a pure
all-distinct/q211 projection must explicitly subtract (48.4).  The same
distinction must be respected in the H3 slot pullback: repeated-factor words
use (48.3), while singleton-anchor/collision words still require the relevant
`M_i^2` or their aggregated `D_ab`.

Therefore full zero-GEMM seam fusion requires the literal product-closure
receipt

\[
\boxed{
\{P_{ia}=M_iN_a,\ U,\ M_i^2\ \text{or}\ D_{ab}\}_{\rm needed\ slots}
\subseteq\text{resident products/views at the synchronized seam}.}
\tag{48.6}
\]

If (48.6) fails, the antisymmetric q211 scalar (47.5) may still be free, but
the complete H3/q211 pullback in (48.2) opens additional products or needs a
certified endpoint omission bound.  In particular, `G_Z/G_Y` constructed from
K31 alone is only `L31^*Q31`, not the complete query.

### Graph split

- `P2R0G19B_MATH_H3_Q211_FULL_PRODUCT_REUSE` — **conditional theorem**: symmetric/skew halves share `M_iN_a`, (48.1).
- `P2R0G19B_MATH_H3_Q211_COMPLETE_QUERY` — **solved completeness identity**: (48.2).
- `P2R0G19B_MATH_H3_Q211_FACTOR_PRODUCT_REUSE` — **solved ordered identity**: full resident `P_ib` yields `N_aN_b`, (48.3).
- `P2R0G19B_MATH_H3_Q211_COLLISION_SQUARE_CLOSURE` — **open literal-reuse gate**: certify (48.4)--(48.6), retain the sewn q22 diagonal, or charge new products/omission.

## 49. Persistent H6/H8 is impossible; analytic two-reset fold is exact

Even in the commuting zero-odd sector, the H6 and H8 state dimensions at
`r=55` are

\[
\boxed{
\dim H_6^{\rm even}=\binom{57}{3}=29{,}260,
\qquad
\dim H_8^{\rm even}=\binom{58}{4}=424{,}270.}
\tag{49.1}
\]

One dense width-1024 crossing would therefore cost approximately `28.574`
and `414.33` GEMM equivalents respectively.  H8 cannot cross once inside the
budget; H6 cannot persist through D10.  A stored higher-chaos carrier is not a
lawful remedy for the return obstruction in Section 38.

The H4-to-H6-to-H4 contribution can instead be folded analytically.  For an
H4 tensor `f` and the two H2 gate matrices `M,N`, define

\[
\mathcal A_{M,N}f={1\over4}\sum_{a=1}^4
(M\circ N)_{(a)}f,
\]

where the subscript denotes action on tensor leg `a`.  The exact unnormalized
two-reset return operator is

\[
\boxed{
\mathcal R^{(6)}_{N\leftarrow M}f
=2\langle M,N\rangle_F f
+16\mathcal A_{M,N}f
+12\operatorname{Sym}\!\left[M\otimes(f\otimes_2N)\right].}
\tag{49.2}
\]

The scalar specialization recovers the unnormalized return coefficient `30`
in (38.8).  Physical reset coefficients and the interposed dense transport
must multiply (49.2) at their actual sites; they cannot be absorbed into a
universal scalar unless the corresponding reverse cyclic module closes.

Thus the higher-chaos route is viable only as a target-conditioned transient
Volterra contraction

\[
P_N\Phi^*_{L\leftarrow l+1}\mathcal R^{(6)}_{N_l\leftarrow M_l}
\]

or as a certified endpoint norm bound.  It must never be promoted to a
persistent H6/H8 forward bank under the current bill.

### Graph split

- `P2R0G19B_MATH_HIGH_CHAOS_STORAGE_NOGO` — **solved**: dimensions/costs (49.1).
- `P2R0G19B_MATH_H6_ANALYTIC_RETURN_FOLD` — **solved local identity / conditional route**: (49.2) plus actual transport/cyclic closure.

## 50. A graded anticommuting carrier: nonzero q211 with zero H2 rotation

There is a structural exception to the usual conflict between noncommutative
q211 value and rank-one reset closure.  Let `J` be a symmetric involution,

\[
J^T=J,\qquad J^2=I,
\]

and split the symmetric core space into its `Z2` grading

\[
\mathcal E=\{A:[A,J]=0\},
\qquad
\mathcal O=\{M:\{M,J\}=0\}.
\tag{50.1}
\]

In a basis with `J=diag(I_p,-I_q)`, `E` is block diagonal and `O` is symmetric
block off-diagonal.  The multiplication rules are

\[
\mathcal E\circ\mathcal E\subseteq\mathcal E,
\quad
\mathcal E\circ\mathcal O\subseteq\mathcal O,
\quad
\mathcal O\circ\mathcal O\subseteq\mathcal E.
\tag{50.2}
\]

For every odd core `M in O`,

\[
\boxed{M\circ J=0,\qquad [M,J]=2MJ.}
\tag{50.3}
\]

Therefore the exact H0+H2 reset intertwiner (35.2) acts on the rank-one square
`U_{J,J}` by its common scalar only:

\[
\boxed{
\Pi_4[(g_0+g_2Q_M)U_{J,J}]=g_0U_{J,J}.}
\tag{50.4}
\]

At the same time the q4-perpendicular carrier is nonzero and maximal in this
direction:

\[
\boxed{
q_{4,\perp}(i,i,J,J)
=-8\|[M_i,J]\|_F^2
=-32\|M_i\|_F^2.}
\tag{50.5}
\]

The paid full product `P_i=M_iJ` is already purely skew because
`P_i^T=JM_i=-M_iJ`.  Hence the sewn rank-one action (46.4) is simply a
weighted reduction of `||P_i||F^2`, with no symmetrization loss and no new
dense product.

Dense physical mixing preserves the odd subspace:

\[
M_i\in\mathcal O\ \forall i
\quad\Longrightarrow\quad
\sum_iW_{ij}M_i\in\mathcal O.
\tag{50.6}
\]

Thus a parity-graded D14 core plus fixed/recurrent D10 grading suffix `J`
gives an exact rank-one common-factor recurrence through dense transport and
the first correlated H2 gate correction, while retaining a large q211
signal.  This is the cleanest algebraic escape found in this lane.

There is a binding lawfulness caveat.  In the external-factor construction of
Section 40 the suffix must normally be a core combination
`J=sum_j u_jM_j`.  An all-odd bank cannot contain its even grading involution.
There are only two valid implementations:

1. `J` is a separately carried **virtual auxiliary suffix**.  Then (50.5) is
   an engineered tracker feature, not automatically an exact external q211
   plane; its supply provenance must be proved.
2. The physical bank has the preserved decomposition

   \[
   \operatorname{span}\{J\}\oplus\mathcal O,
   \tag{50.6a}
   \]

   and `u` selects the distinguished `J` direction.  Only the odd anchor
   components enjoy (50.4).  A general anchor `M_i=alpha_iJ+O_i` obeys

   \[
   M_i\circ J=\alpha_i I,
   \tag{50.6b}
   \]

   so the first reset enlarges the root module to `span{J,I}` unless
   `alpha_i=0` on every value-carrying anchor or the `I` tangent is already
   Bellman-dark/old-owned.  A subsequent action on `I` exposes `M_i` and can
   reopen the full odd module.

Consequently dense transport must preserve the distinguished coefficient
direction rather than mix it arbitrarily into physical anchors, or an exact
projection/old-ownership theorem must remove the resulting `I` path.

### 50.1 Higher-word and capacity gates

The theorem stops at the first correlated gate order.  Odd products square
into the even algebra:

\[
M_i^2,\ M_iM_j+M_jM_i\in\mathcal E.
\]

The next gate/H6/H8 words can therefore generate directions such as
`M_i^2J` and longer reversible even words.  Full exactness requires the
terminal cyclic space

\[
\boxed{
\mathcal C_J=\operatorname{span}\{EJ:
 E\in\operatorname{RevAlg}(\{M_iM_j\})\}}
\tag{50.7}
\]

to be old-owned or cap-compatible, together with the analytic transient
returns of Section 49.  Merely having the binary grading (50.2) does not bound
`dim C_J`.

For `p+q=r`, the odd carrier has dimension `pq`, maximized at about `r^2/4`
(`756` at `r=55`).  It is large enough to be nontrivial but smaller than the
full symmetric core dimension `r(r+1)/2=1540`; its task-value/provenance must
therefore be certified rather than inferred from the unrestricted NCQC
oracle ceiling.

A stronger Clifford/composition identity such as

\[
M_iM_j+M_jM_i=2g_{ij}I
\tag{50.8}
\]

collapses the even-word action, but a large linearly independent real family
of such generators is dimension-limited.  It is a useful structured refit
hypothesis, not a free width-1024 theorem.

### Graph split

- `P2R0G19B_MATH_GRADED_ANTICOMMUTING_CARRIER` — **solved structural theorem**: zero H2 rotation plus nonzero q211, (50.3)--(50.6).
- `P2R0G19B_MATH_GRADED_SUFFIX_LAWFULNESS` — **open representation gate**: certify virtual-source provenance or the preserved decomposition (50.6a)--(50.6b).
- `P2R0G19B_MATH_GRADED_EVEN_WORD_MODULE` — **open exactness/cost gate**: certify (50.7) is old-owned or cap-compatible.
- `P2R0G19B_MATH_GRADED_CORE_VALUE` — **open provenance/value gate**: determine whether the `pq`-dimensional odd core retains the required `phi>=0.95696` task value.

## 51. The grading does not close the complete H3 reset seam

The involution in Section 50 induces an exact grading on third Wiener chaos.
Let

\[
\Gamma=J^{\otimes3}\big|_{\operatorname{Sym}^3(V)},
\qquad \mathcal H_3=\mathcal H_3^+\oplus\mathcal H_3^- .
\tag{51.1}
\]

If the \(+1/-1\) eigenspaces of \(J\) have dimensions \(p,q\), then

\[
\boxed{
\begin{aligned}
d_+&=\binom{p+2}{3}+p\binom{q+1}{2},\\
d_-&=q\binom{p+1}{2}+\binom{q+2}{3}.
\end{aligned}}
\tag{51.2}
\]

For the balanced r55 split \(p=28,q=27\), these are \(14644\) and
\(14616\).  Thus the binary grading alone divides H3 into two almost-full
halves; it is not a small carrier.

### 51.1 Exact reset block law

For a gate \(F(z)\), write

\[
\mathcal R_F=P_3M_FP_3,\qquad
F_\pm(z)=\frac{F(z)\pm F(Jz)}2 .
\]

Orthogonal Gaussian invariance gives the exact intertwining identity

\[
\boxed{
\Gamma\mathcal R_F\Gamma=\mathcal R_{F\circ J}.}
\tag{51.3}
\]

Therefore \(\mathcal R_{F_+}\) is block diagonal on (51.1), while
\(\mathcal R_{F_-}\) is block off-diagonal.  For

\[
g=b+a^{\mathsf T}z+z^{\mathsf T}(\alpha J+O)z
-\operatorname{tr}(\alpha J+O),qquad \{O,J\}=0,
\]

the transformed gate has \(a\mapsto Ja\) and \(O\mapsto-O\).  Unless these
changes leave the physical gate invariant, (51.3) supplies only a bipartite
block pattern, not scalar reset closure.

There is a particularly clean all-odd case.  If \(b=a=\alpha=0\), then
\(g(Jz)=-g(z)\), so, away from the null boundary,

\[
\boxed{
\mathcal R_F=\frac12I+B,qquad \Gamma B\Gamma=-B.}
\tag{51.4}
\]

The non-scalar leakage is entirely off-diagonal between
\(\mathcal H_3^+\) and \(\mathcal H_3^-\), but (51.4) puts no rank or norm
bound on that off-diagonal block.

### 51.2 What radial/I ownership actually removes

For a vector \(v\) and symmetric matrix \(A\), put

\[
T(v,A)=\operatorname{Sym}(v\otimes A),qquad
\mathscr L_MZ=\operatorname{Sym}(M\otimes_1Z).
\]

The Wiener product formula and a direct three-leg expansion give

\[
\boxed{
P_3[I_2(M)I_3(T(v,A))]
=6I_3(\mathscr L_MT(v,A)),
}
\tag{51.5}
\]

and

\[
\boxed{
6\mathscr L_MT(v,A)
=2T(Mv,A)+4T(v,M\circ A).}
\tag{51.6}
\]

With \(M_i=\alpha_iJ+O_i\), Section 50's identity
\(M_i\circ J=\alpha_iI\) therefore lifts to

\[
\boxed{
6\mathscr L_{M_i}T(v,J)
=2T(\alpha_iJv+O_iv,J)+4\alpha_iT(v,I).}
\tag{51.7}
\]

If the complete old interface owns the whole trace/radial H3 space

\[
\mathcal T_I=\{T(v,I):v\in V\},
\]

then it does remove the final term of (51.7) **at that cut**.  It does not
remove the first term: in particular the coefficient \(\alpha_i\) still
appears through \(T(Jv,J)\), and the odd core appears through
\(T(O_iv,J)\).

Nor may a locally old-owned \(I\) tangent simply be discarded before future
resets.  Indeed

\[
\boxed{
6\mathscr L_{M_i}T(v,I)
=2T(M_iv,I)+4\alpha_iT(v,J)+4T(v,O_i).}
\tag{51.8}
\]

Thus \(\mathcal T_I\) is not reset-invariant.  Dynamic removal requires the
strong seam receipt

\[
P_\perp\mathscr L_{M_i}P_{\mathcal T_I}=0
\quad\text{for every value-carrying reset }i,
\tag{51.9}
\]

or complete-old ownership of the full cyclic hull generated by (51.8).
Static radial ownership alone proves neither.

### 51.3 Exact trace-quotiented counterexample

The failure persists even when \(\alpha=0\), so there is no \(I\) tangent to
blame.  Choose one \(+/-\) coordinate pair \(x,y\), and on that plane take

\[
J=\begin{pmatrix}1&0\\0&-1\end{pmatrix},qquad
O=\begin{pmatrix}0&1\\1&0\end{pmatrix},qquad
g=z^{\mathsf T}Oz=2xy.
\tag{51.10}
\]

Let \(F=1_{\{xy>0\}}\).  Two normalized traceless H3 functions are

\[
h_+=\frac{x^3-3xy^2}{2\sqrt6},
\qquad
h_-=\frac{3x^2y-y^3}{2\sqrt6}.
\tag{51.11}
\]

They obey \(\Gamma h_+=h_+\), \(\Gamma h_-=-h_-\), and are orthogonal to
every trace cubic \(T(v,I)\).  Since
\(F-1/2=\tfrac12\operatorname{sgn}(x)\operatorname{sgn}(y)\), the Gaussian
absolute moments give exactly

\[
\boxed{
\langle h_+,(\mathcal R_F-\tfrac12I)h_-\rangle
=\frac1{3\pi},qquad
\langle h_\pm,(\mathcal R_F-\tfrac12I)h_\pm\rangle=0.}
\tag{51.12}
\]

Hence, after quotienting the **entire** radial/I H3 sector, the surviving
two-dimensional harmonic block is

\[
\boxed{
P_{\rm harm}(\mathcal R_F-\tfrac12I)P_{\rm harm}
=\begin{pmatrix}0&(3\pi)^{-1}\\(3\pi)^{-1}&0\end{pmatrix}.}
\tag{51.13}
\]

This counterexample embeds unchanged in r55.  It proves that exact
anticommutation, zero H2 rotation of \(U_{J,J}\), \(\alpha=0\), and complete
radial/I ownership still leave nonzero complete-H3 reset leakage.

For reference, before the trace quotient the four-dimensional H3 block in
the orthonormal order

\[
\left(\frac{\operatorname{He}_3(x)}{\sqrt6},
\frac{x\operatorname{He}_2(y)}{\sqrt2}\right)
\oplus
\left(\frac{\operatorname{He}_2(x)y}{\sqrt2},
\frac{\operatorname{He}_3(y)}{\sqrt6}\right)
\]

has centered off-diagonal matrix

\[
C=\frac1{2\pi}
\begin{pmatrix}\sqrt3&1/3\\1&\sqrt3\end{pmatrix},
\qquad
\|C\|_{\rm op}=\frac{\sqrt{32+8\sqrt7}}{6\pi}.
\tag{51.14}
\]

So the grading obstruction is order one, not a perturbative boundary case.

### 51.4 Exact finite-closure criterion and generic no-go

For projected reset blocks \(R_i=P_\perp\mathcal R_{F_i}P_\perp\) and a
source/endpoint seed space \(S\), the minimal exact carrier is the cyclic
module

\[
\boxed{
\mathcal C_3(S)=\operatorname{span}\{
R_{i_k}\cdots R_{i_1}s:
k\ge0,\ s\in S\}.}
\tag{51.15}
\]

The grading gives an exact finite closure of dimensions \(k_++k_-\) if and
only if there are spaces \(C_\pm\subseteq P_\perp\mathcal H_3^\pm\),
\(\dim C_\pm=k_\pm\), containing the projected seeds, such that every even
reset block preserves each \(C_\pm\) and every odd reset block maps
\(C_+\leftrightarrow C_-\).  This is the required common-intertwiner
receipt; binary parity by itself supplies none of its rank bounds.

There is a sharp generic obstruction already in the H2 tangent (51.6).
Taking the cross-block basis

\[
O_{a\mu}=E_{a\mu}+E_{\mu a}
\]

over a connected \(+/-\) bipartite graph, \(J\) and the \(O_{a\mu}\)
generate the full matrix Lie algebra (their commutators recover both
oriented cross-block units and the within-block matrix units).  Its induced
action on \(\operatorname{Sym}^3(V)\) is irreducible.  Consequently any
nonzero H3 carrier which is invariant under all corresponding first reset
tangents is the full

\[
\dim\mathcal H_3=\binom{r+2}{3}
\]

module.  At r55 this is \(29260\), far outside the bill.  Thus a uniform
finite closure theorem needs a stronger Clifford/common-intertwiner
identity or the actual complete-old cyclic-rank receipt (51.15); the
anticommuting grading alone is a solved NO-GO for the complete H3 seam.

### Graph split

- `P2R0G19B_MATH_GRADED_H3_BLOCK_LAW` — **solved theorem**: exact parity blocks (51.1)--(51.4).
- `P2R0G19B_MATH_GRADED_I_TANGENT_OWNERSHIP` — **solved local / open dynamic receipt**: local removal (51.7), required invariance (51.8)--(51.9).
- `P2R0G19B_MATH_GRADED_H3_TRACE_QUOTIENT_NOGO` — **solved counterexample**: nonzero harmonic leakage (51.10)--(51.14).
- `P2R0G19B_MATH_GRADED_H3_CYCLIC_CLOSURE` — **open frozen-net rank gate**: certify the common-intertwiner module (51.15); generic grading alone gives full H3.

## 52. Exact collision quotient and the target-conditioned aggregation loophole

Section 48 identifies the original-label square correction as the remaining
word-level obstruction.  It admits an exact quotient theorem.  Let

\[
V=\mathbb R^N,
\qquad S=\operatorname{col}U\subseteq V,
\qquad Q=I-P_S,
\]

where `N` is the number of labelled core matrices.  In the free ordered
degree-two word space, identify

\[
C\in V\otimes V
\quad\longleftrightarrow\quad
\mathcal W(C)=\sum_{ij}C_{ij}M_iM_j.
\]

The resident products `P_ia=M_iN_a` span \(V\otimes S\); if their transposes
are lawful resident views, the two orderings span

\[
\mathcal L=(V\otimes S)+(S\otimes V)
          =\ker[C\mapsto QCQ].
\tag{52.1}
\]

Thus `QCQ` is the exact unresolved coefficient of a word after all resident
factor products have been used.  With ordered `P` only, replace (52.1) by
the one-sided condition `CQ=0`.

Define the diagonal embedding and square-summary word

\[
\Delta(w)=\operatorname{diag}w,
\qquad
S_{\rm sq}(w)=\mathcal W(\Delta(w))=\sum_iw_iM_i^2.
\]

If `W0` is the span of old resident square-summary weights, put

\[
K=\{w:Q\Delta(w)Q=0\},
\qquad E=W_0+K.
\tag{52.2}
\]

Then the following criterion is necessary and sufficient in the free word
algebra:

\[
\boxed{
\mathcal W(C)\text{ is resident modulo old square summaries}
\iff
\exists w_0\in W_0:\ Q(C-\Delta(w_0))Q=0.}
\tag{52.3}
\]

In particular, for the factor-pair collision

\[
w_{ab}=u_a\odot u_b,
\qquad D_{ab}=S_{\rm sq}(w_{ab}),
\]

`D_ab` is free exactly when \(w_{ab}\in E\).  For a requested pair set
\(\mathcal P\), let

\[
H_{\mathcal P}=\operatorname{span}\{u_a\odot u_b:(a,b)\in\mathcal P\}.
\]

The minimal number of additional full-matrix square summaries is exactly

\[
\boxed{d_D=\dim((H_{\mathcal P}+E)/E).}
\tag{52.4}
\]

A complement basis attains (52.4), while each new retained square matrix can
remove at most one quotient direction, proving optimality.

### 52.1 Why all-anchor masking is generically fatal

For one factor pair write \(w=u_a\odot u_b\).  The quotient residue of the
anchored all-distinct word (48.5) is the diagonal weight

\[
g_i=2w_i e_i-w.
\tag{52.5}
\]

The `g_i` are the columns of

\[
T_w=2\operatorname{diag}w-w\mathbf1^T,
\qquad
\det T_w=2^N\Bigl(\prod_iw_i\Bigr)(1-N/2).
\tag{52.6}
\]

Hence, if every `w_i` is nonzero and \(N\ne2\), the anchored residues
span all of `V`.  Materializing every anchor exactly therefore needs
\(N-\dim E\) independent square-summary directions generically.  A common
factorization by itself does not make an all-anchor all-distinct seam cheap.

### 52.2 Scalar reverse aggregation is the sharp escape

A target-conditioned reverse contraction need not materialize every anchor.
If its anchor costates have common factor rank `q`,

\[
R_i=\sum_{t=1}^q c_{it}R_t,
\]

then all collision residues aggregate before any matrix is formed:

\[
\sum_i\langle R_i,S_{\rm sq}(g_i)\rangle
=\sum_{t=1}^q\langle R_t,S_{\rm sq}(T_wc_t)\rangle.
\tag{52.7}
\]

Consequently the minimal number of new square-summary directions for a
costate space \(C=\operatorname{span}\{c_t\}\) is

\[
\boxed{d_{\rm agg}=\dim((T_wC+E)/E),}
\tag{52.8}
\]

which is at most `q`, rather than `N`.  This is the precise target-conditioned
loophole: one sewn scalar Q22 costate can require only one aggregate square
summary.  It does **not** apply to an anchor-specific q211 seam unless that
seam's matrix costates have a certified low common factor rank.

For a generic independent family `R_i`, `q=N` and (52.6) restores the full
lower bound.  Replacing matrices by already-targeted scalars is lawful only
after the complete future costate is fixed; for `q` independent future
functionals, the information requirement is \(q d_D\) scalar contractions or
`d_D` retained matrices.

### 52.3 Structured collision closures

If the requested common-factor action is spectrally diagonal, only
\(w_{aa}=u_a^{\odot2}\) occurs.  For flat orthogonal/Hadamard `U`, every such
weight equals \(\mathbf1/N\); one old total-square summary
\(\sum_iM_i^2\) therefore closes all diagonal factor collisions.  Cross
\(a\ne b\) modes generally destroy this collapse.

More generally, it suffices that the old square-summary space contain the
Schur product space \(S\odot S\).  A unital subspace of \(\mathbb R^N\) closed under
entrywise multiplication is exactly the space of block-constant functions
for a partition.  Thus a robust exact design retains one square-sum matrix
per block.  Requiring every single-anchor deletion mask to preserve a dense
`S`, however, forces the full coordinate space; this is another form of the
all-anchor no-go.

### Graph split

- `P2R0G19B_MATH_COLLISION_WORD_QUOTIENT` — **solved exact theorem**: resident-product quotient (52.1)--(52.3).
- `P2R0G19B_MATH_COLLISION_SUMMARY_RANK` — **solved rank formula / open receipt**: evaluate (52.4) on the frozen common factors and old summaries.
- `P2R0G19B_MATH_ALL_ANCHOR_COLLISION_NOGO` — **solved generic no-go**: (52.5)--(52.6).
- `P2R0G19B_MATH_TARGET_COSTATE_AGGREGATION` — **conditional exact escape**: certify low common costate rank and (52.7)--(52.8).
- `P2R0G19B_MATH_SCHUR_BLOCK_COLLISION_CLOSURE` — **conditional design theorem**: old square summaries contain the required `S odot S` block algebra.

## 53. Exact Clifford-pair carrier: complete centered-gate H4 closure

The anticommuting idea becomes exact for the **complete** one-reset H4
operator under a stronger flat-spectrum identity.  Let

\[
J^T=J,qquad J^2=I,
\]

and suppose a centered quadratic gate core has the decomposition

\[
M=\alpha J+O,qquad O^T=O,qquad \{O,J\}=0,qquad O^2=\omega^2I.
\tag{53.1}
\]

Equivalently,

\[
\{M,J\}=2\alpha I,
\qquad M^2=(\alpha^2+\omega^2)I.
\tag{53.2}
\]

Put `rho=sqrt(alpha^2+omega^2)` and define

\[
A={M\over\rho},
\qquad
B={\omega J-(\alpha/\omega)O\over\rho}.
\tag{53.3}
\]

Then `A` and `B` are symmetric involutions satisfying

\[
A^2=B^2=I,qquad \{A,B\}=0,qquad
J={\alpha A+\omega B\over\rho}.
\tag{53.4}
\]

Thus the complete direct H4 descendants lie in the six-dimensional square
lift

\[
\operatorname{Sym}^2\operatorname{span}\{I,A,B\}
=\operatorname{span}\{U_{II},U_{IA},U_{IB},U_{AA},U_{AB},U_{BB}\}.
\tag{53.5}
\]

This is closure under the whole centered gate, not merely under its H0+H2
truncation.

### 53.1 Two symmetries force an exact scalar novel quotient

Let

\[
F(z)=\mathbf1\{Q_A(z)>0\},
\qquad
D=P_4M_{F-1/2}P_4.
\]

Conjugation by `A` preserves the gate, whereas conjugation by `B` reverses
the sign of `Q_A` and hence complements the zero-bias gate.  If `Gamma_A`
and `Gamma_B` denote their actions on H4, then

\[
[D,\Gamma_A]=0,
\qquad
\{D,\Gamma_B\}=0.
\tag{53.6}
\]

The matrix parities under `(Gamma_A,Gamma_B)` are

\[
I:(+,+),\qquad A:(+,-),\qquad B:(-,+).
\]

Consequently, with

\[
\mathcal T=\operatorname{span}\{U_{II},U_{IA},U_{IB}\},
\qquad
\mathcal N=\operatorname{span}\{U_{AA},U_{AB},U_{BB}\},
\]

the centered part maps the novel sector into the trace sector:

\[
D\{U_{AA},U_{BB}\}\subseteq\operatorname{span}\{U_{IA}\},
\qquad
DU_{AB}\in\operatorname{span}\{U_{IB}\}.
\tag{53.7}
\]

It is **not** generally true that \(D\mathcal T\subseteq\mathcal N\): for
example \(DU_{II}\) may remain in \(\operatorname{span}\{U_{IA}\}\), and
\(DU_{IA}\) may contain both
trace and novel components.  Only the displayed novel-to-trace inclusion is
needed for the next identity.

Therefore the exact same-sector reset is

\[
\boxed{
P_{\mathcal N}P_4M_FP_4P_{\mathcal N}
=\frac12 I_{\mathcal N}.}
\tag{53.8}
\]

Since `J in span{A,B}`, the desired square carrier `U_JJ` lies in
`mathcal N`.  At every gate satisfying (53.1),

\[
P_{\mathcal N}P_4M_FP_4U_{JJ}=\frac12U_{JJ}.
\tag{53.9}
\]

At the same time its leading Jordan rotation is entirely radial while its
q211 signal remains nonzero:

\[
M\circ J=\alpha I,
\qquad
-8\|[M,J]\|_F^2=-32r\omega^2.
\tag{53.10}
\]

Equations (53.8)--(53.10) are the first exact mechanism found here that
simultaneously gives a common one-word H4 quotient and an order-one
noncommutative q211 action.

The word "quotient" is binding.  The odd block `D U_JJ` is not zero; it
lands in the trace/H31 sector `mathcal T`.  Formula (53.9) is sufficient for
the end-to-end scalar only if the synchronized old engine literally owns the
complete `mathcal T` birth and its future Bellman action.  Merely subtracting
terminal correlations with old features is insufficient, because an old
birth can return to `mathcal N` at a later reset.  The lawful recurrence is a
coupled old-plus-new block system, with the new-to-new block fixed to `1/2`.
Equivalently, a standalone repeated scalar recurrence also needs the suffix
annihilation receipt

\[
P_{\mathcal N}\Phi_{t\leftarrow s}P_{\mathcal T}D_sP_{\mathcal N}=0
\quad\text{for every later suffix }t>s;
\tag{53.11}
\]

otherwise the old-to-new return must remain in the coupled Bellman system.

### 53.2 Multiple gates and the actual frozen-net receipt

For one common carrier `J`, every live gate must satisfy

\[
M_i=\alpha_iJ+O_i,qquad
\{O_i,J\}=0,qquad O_i^2=\omega_i^2I.
\tag{53.12}
\]

The auxiliary pairs `(A_i,B_i)` may vary with `i`; nevertheless (53.9)
keeps the projected `U_JJ` line common.  The old interface must own all
trace siblings generated by these pairs, equivalently the required part of

\[
\operatorname{span}_i\{U_{II},U_{I J},U_{I O_i}\}
\tag{53.13}
\]

at the correct seam.  In addition, deployability still requires the
independent receipts already derived above:

1. `J` is a lawful frozen source/common factor, not a virtual matrix chosen
   only in the proof;
2. its reverse recurrence remains collinear across dense transport;
3. either the full `M_iN` product is retained for the general sewn query, or
   the special Clifford scalar is evaluated by the trace/norm identity below;
4. q22 diagonal sewing/collision ownership satisfies Sections 47 and 52;
5. the old trace interface in (53.13) is complete under future resets.

Without (53.12), binary anticommutation only proves the H0+H2 identity and
the generic H3/H4 cyclic-rank no-go of Sections 44 and 51 returns.  In
particular, a generic symmetric core has neither paired spectrum nor a
common chiral involution; (53.12) is a measure-zero algebraic certificate,
not something implied by diffuseness.

### 53.3 Odd latent dimension

If `J` has plus/minus eigenspace dimensions `p,q`, an invertible `O`
anticommuting with `J` maps the two spaces bijectively.  Hence `p=q` and the
full-rank identity (53.1) is impossible when `r` is odd.  At `r=55`, the
flattest exact alternative has a common rank-54 support `P`,

\[
J_P=PJ,qquad J_P^2=P,qquad
O_i^2=\omega_i^2P,qquad \{O_i,J_P\}=0,
\tag{53.14}
\]

with all gate cores and the carrier supported on `P`.  The theorem then
holds inside the 54-dimensional support with `I` replaced by `P`.  Any
kernel component, nonzero action outside `P`, or kernel-cross lift is a
separate state and needs an old/dark or endpoint-error receipt.  In
particular, adding an unpaired `alpha J` action on the kernel breaks the
global complement symmetry.

Thus the clean exact implementation choices are an even latent rank (for
example 54) or a common supported rank-54 carrier with explicit kernel
ownership.  Calling the unrestricted r55 construction exact without that
kernel gate would be wrong.

### 53.4 Arithmetic shortcut and the dense-mixing capacity no-go

Under (53.11), the residual commutator scalar itself needs no dense matrix
product.  Orthogonality of the even and odd pieces gives

\[
\alpha_i={\operatorname{tr}(M_iJ)\over r},
\qquad
\omega_i^2={\|M_i\|_F^2\over r}-\alpha_i^2,
\qquad
-8\|[M_i,J]\|_F^2=-32r\omega_i^2.
\tag{53.15}
\]

These are reductions if `M_i,J` are resident.  In the unresidualized q211
identity the commutator term accompanies the commuting `48F` term; (53.15)
is the q4-perpendicular action only after that old-parallel term is literally
acquired and subtracted.  CP4 collision reads do not by themselves certify
an all-distinct `48F` acquisition port.

There is a separate, sharp capacity obstruction.  Suppose the flat identity
must survive arbitrary dense mixtures in an active odd-core space

\[
\mathcal O=\operatorname{span}\{O_1,\ldots,O_k\}.
\]

Requiring every \(O\in\mathcal O\) to satisfy \(O^2\in\mathbb RI\)
polarizes to

\[
\boxed{
O_iO_j+O_jO_i=2g_{ij}I.}
\tag{53.16}
\]

After quotienting null directions and whitening `g`, the common `J` and the
`k` independent odd generators form a real Clifford representation.  Even
the complex representation lower bound gives

\[
r\ge 2^{\lfloor(k+1)/2\rfloor},
\qquad
k\le 2\log_2 r+O(1).
\tag{53.17}
\]

At `r<=55` this safe bound gives at most ten independent odd directions
besides `J` (real symmetry/divisibility can be stricter).  Moreover the core
pair covariance satisfies

\[
\operatorname{tr}(M_iM_j)
=r(\alpha_i\alpha_j+g_{ij}),
\qquad
\operatorname{rank}C\le k+1.
\tag{53.18}
\]

For a flat width-1024 target covariance, even the best rank-11 core has
correlation at most

\[
\sqrt{11/1024}=0.1037.
\tag{53.19}
\]

Thus the exact Clifford family cannot be the full broad-capacity D14 state:
dense-mixing closure collapses it to a low-rank CP/star-like core, far below
the required width-scale capacity.  Its only plausible role is a
target-conditioned auxiliary/common D10 word whose base covariance and
trace siblings are owned by a different full-capacity engine.  Even that
role still needs the source-provenance, suffix-collinearity, and old-return
receipts above.

### Graph split

- `P2R0G19B_MATH_CLIFFORD_H4_SIX_LIFT_CLOSURE` — **solved algebraic theorem**: (53.1)--(53.5).
- `P2R0G19B_MATH_CLIFFORD_CENTERED_RESET_QUOTIENT` — **solved conditional identity**: exact `1/2` novel block (53.6)--(53.10).
- `P2R0G19B_MATH_CLIFFORD_OLD_TRACE_OWNERSHIP` — **same-cut representable / future-open receipt**: old engine owns (53.13) and its future returns.
- `P2R0G19B_MATH_CLIFFORD_FROZEN_CORE_RECEIPT` — **open / generically false**: certify (53.12), common-source provenance, and collinear recurrence.
- `P2R0G19B_MATH_CLIFFORD_R55_KERNEL_GATE` — **open parity-width receipt**: use an even core or certify the supported construction (53.14).
- `P2R0G19B_MATH_CLIFFORD_REDUCTION_COST` — **conditional zero-product identity**: (53.15), after literal old-parallel `48F` subtraction.
- `P2R0G19B_MATH_CLIFFORD_DENSE_CAPACITY_NOGO` — **solved structural no-go**: dense closure forces (53.16)--(53.19), so the family cannot replace full D14.

## 54. Exact finite direct H4 adjoint and the one-tensor sufficiency theorem

The direct H4-to-H4 action of an arbitrary quadratic gate is finite even
though the gate has an infinite Hermite expansion.  Let

\[
F=\sum_{m\ge0}I_{2m}(r_{2m}),
\qquad U=I_4(f),qquad f\in\operatorname{Sym}^4(\mathbb R^r),
\]

in the multiple-Wiener-integral convention.  The product formula gives

\[
\boxed{
\begin{aligned}
P_4(FU)=I_4\bigl[{}
&r_0f
+8\operatorname{Sym}(r_2\otimes_1f)
+72\operatorname{Sym}(r_4\otimes_2f)\\
&+480\operatorname{Sym}(r_6\otimes_3f)
+1680\operatorname{Sym}(r_8\otimes_4f)
\bigr].
\end{aligned}}
\tag{54.1}
\]

Indeed, an order-`2m` gate component can return order four only by `m`
contractions, so `m<=4`; the coefficient is

\[
m!\binom4m\binom{2m}m=(1,8,72,480,1680).
\]

No gate chaos above H8 acts **directly** from H4 back to H4 in one reset.
Because `P_4 M_F P_4` is self-adjoint, (54.1) is also the exact local reverse
operator, with the same four contraction blocks.

For the quadratic convention

\[
Q_M={1\over2}(z^TMz-\operatorname{tr}M),
\qquad F=\mathbf1\{b+Q_M>0\},
\]

the four kernels have an exact one-dimensional Fourier representation.  Put

\[
\chi_M(t)=e^{-it\operatorname{tr}M/2}\det(I-itM)^{-1/2},
\qquad
A_M(t)=itM(I-itM)^{-1}.
\tag{54.2}
\]

The Wick exponential identity

\[
\mathbb E[e^{itQ_M}:e^{h^Tz}:]
=\chi_M(t)\exp\!\left({1\over2}h^TA_M(t)h\right)
\tag{54.3}
\]

yields, for `m=1,...,4`,

\[
\boxed{
r_{2m}={1\over\pi}\int_0^\infty
 {\operatorname{Im}\!\left[e^{itb}\chi_M(t)
 \operatorname{Sym}(A_M(t)^{\otimes m})\right]
  \over 2^m m!\,t}\,dt.}
\tag{54.4}
\]

Thus a direct restricted action can be acquired analytically without an
infinite chaos truncation, although materializing the dense H8 kernel is
still prohibitive.  Section 53 is cheap precisely because (54.2) remains in
`span{I,A}` for an involutive core, so all four blocks reduce inside the six
lifts.

### 54.1 Exact target-conditioned reverse recurrence

Let `mathscr H_4=Sym^4(R^r)`.  Write the frozen linearized H4 layer as

\[
h_{\ell+1,j}=T_{\ell j}\!\left(\sum_iW_{\ell,ji}h_{\ell i}\right)
+b_{\ell j},
\tag{54.5}
\]

where `b_lj` is the H4 birth and `T_lj` is the realized local reset action.
For a scalar endpoint with terminal costates `lambda_Lj`, the exact frozen
H4-only reverse recurrence is

\[
\widetilde\lambda_{\ell j}=T_{\ell j}^*\lambda_{\ell+1,j},
\qquad
\lambda_{\ell i}=\sum_jW_{\ell,ji}\widetilde\lambda_{\ell j},
\tag{54.6}
\]

and every birth is accumulated once:

\[
\boxed{
\delta y=\sum_{\ell,j}
\langle\lambda_{\ell+1,j},b_{\ell j}\rangle.}
\tag{54.7}
\]

Dense transport only mixes the external coefficient vectors.  The reset is
the source of H4 basis-rank growth.  If at a cut

\[
\lambda_{\ell+1,j}=\sum_{a=1}^k c_{ja}f_a,
\]

first combine the modes at each neuron and apply its realized reset:

\[
g_j:=T_{\ell j}^*\!\left(\sum_a c_{ja}f_a\right).
\]

Let `G` be the H4-valued matrix whose `j`th column is `g_j`.  The minimal
exact H4 basis after the complete local-reset-plus-dense step is

\[
\boxed{
k'=\operatorname{rank}_{\mathscr H_4}(GW_\ell).}
\tag{54.8}
\]

Before dense transport it is `dim span{g_j}`.  If the square dense matrix
`W_l` is invertible, (54.8) equals that dimension; a singular or deliberately
low-rank transport can reduce it and must be credited through the exact
unfolding rank, not through a generic argument.

For an exact `k`-basis factorization the irreducible storage and dense-mix
counts are

\[
\boxed{
S_k=k\binom{r+3}{4}+Nk,
\qquad
C_{\rm dense}=O(LN^2k).}
\tag{54.8a}
\]

Local reset acquisition is additional and depends on whether the four
restricted contractions in (54.1) are resident/structured.  With
\(r^4=\Theta(N^2)\), `k=1` has the desired \(\Theta(N^2)\) state, whereas
generic \(k=\Theta(N)\) already has \(\Theta(N^3)\) state and
\(\Theta(LN^3)\) dense work.

Therefore a single `r^4=O(N^2)` adjoint tensor is sufficient across a cut
if and only if the complete future cyclic module is one-dimensional.  At one
reset, under invertible dense transport and nonzero active coefficients, this
requires

\[
T_{\ell j}^*f=\gamma_j f_-
\quad\text{for every live }j
\tag{54.9}
\]

modulo a **complete** old Bellman interface.  This is the exact
intertwining/collinearity receipt, not a heuristic low-rank assumption.

The diffuse linear-gate and Hadamard counterexamples of Section 44 show that
the `g_j` in (54.8) are generically independent up to
\(\min(N,\dim\mathscr H_4)\) already after one reset.  Since

\[
\dim\mathscr H_4=\binom{r+3}{4}=\Theta(r^4)=\Theta(N^2)
\quad\text{when }r=\Theta(\sqrt N),
\]

generic \(k'=\Theta(N)\) needs \(\Theta(N^3)\) tensor storage, plus its local
actions.  One target scalar does not by itself remove this operator-Schmidt
rank.  It removes it only through a proved common cyclic quotient such as
(54.9); Section 53 supplies one conditional example.

### 54.2 First-order exactness is not self-consistent chaos closure

Equations (54.1) and (54.6) are exact for paths constrained to remain in H4.
They are also the exact first derivative of a frozen H4-truncated model.
They are **not** automatically the exact derivative of the full nonlinear
law.  A gate has off-diagonal blocks

\[
T_{q\leftarrow p}=P_qM_FP_p;
\]

an H4 perturbation may enter H6 or H8 at one reset and return to H4 later.
Those temporal words are not included in the direct finite action (54.1).
The exact full scalar needs one of:

1. a complete cyclic module containing every future-visible chaos order;
2. the analytic H6/H8 Volterra folds of Section 49, including their suffix
   costates; or
3. a proof that the endpoint Bellman functional annihilates every omitted
   return word within the `3.80e-6` coherent norm gate.

This distinction explains why an `r^4` endpoint adjoint can be exact for a
single frozen insertion but fail after self-consistent resets.

### 54.3 Double-counting rules

The scalar identity (54.7) is exact only under the following ownership
rules.

1. **Births once.**  A physical H4 birth appears in exactly one `b_lj`.  Do
   not both inject it into the forward state and add its reverse contraction.
2. **Frozen versus co-adapted gates.**  If the proposed feature changes the
   gate parameters or lower-order state, their JVP terms belong in `T` and
   the birth.  A frozen-gate tangent plus a separately refitted gate response
   double counts the shared derivative.
3. **Old/new quotient once.**  Split each source and costate with the same
   synchronized complete-old projector.  A trace/H31 component owned by D14
   cannot also be credited to D10 merely because it reappears in (54.1).
4. **Collision sewing once.**  The diagonal q22 term must be either retained
   in the sewn q22+q211 slice or subtracted through `D_ab`, never both.
5. **Temporal returns once.**  H6/H8/Mehler return paths owned by an old
   analytic kernel must be removed from any explicit Volterra fold; primitive
   transverse returns remain separate.

### Graph split

- `P2R0G19B_MATH_H4_DIRECT_FINITE_GATE_ACTION` — **solved exact theorem**: (54.1)--(54.4).
- `P2R0G19B_MATH_H4_TARGET_REVERSE_RECURRENCE` — **solved exact recurrence**: (54.5)--(54.7).
- `P2R0G19B_MATH_H4_ONE_TENSOR_IFF` — **solved criterion / open frozen receipt**: cyclic collinearity (54.8)--(54.9).
- `P2R0G19B_MATH_H4_GENERIC_OPERATOR_SCHMIDT_NOGO` — **solved generic no-go**: one reset gives width-many H4 directions and cubic storage.
- `P2R0G19B_MATH_H4_TEMPORAL_CHAOS_RETURN` — **open end-to-end receipt**: direct H4 action does not own H6/H8 return words.
- `P2R0G19B_MATH_H4_DOUBLE_COUNT_AUDIT` — **binding accounting rule**: apply the five ownership rules above.

## 55. Full native H211 action from the resident ordered-product bank

The open `48F` ownership gate applies only to the cheaper skew-only quotient.
If the synchronized lane literally retains the full ordered products

\[
P_{ia}=M_iN_a,
\qquad N_a=\sum_jU_{ja}M_j,
\]

then the **full native** repeated-anchor H211 action has the exact tensor-free
form

\[
\boxed{
S_{211}^{\rm native}
=\sum_{i,a,b}(\Theta_i)_{ab}
\left[
32\langle P_{ia},P_{ib}\rangle_F
+16\langle P_{ia}^T,P_{ib}\rangle_F
\right].}
\tag{55.1}
\]

Define the small Grams

\[
(F_i)_{ab}=\langle P_{ia},P_{ib}\rangle_F,
\qquad
(R_i)_{ab}=\langle P_{ia}^T,P_{ib}\rangle_F.
\]

Then (55.1) is `sum_i <Theta_i,32F_i+16R_i>`.  It includes both the
commuting parallel and noncommuting residual; no separate `48F` acquisition
is assumed.  With

\[
K_{ia}=\operatorname{skew}P_{ia},
\qquad
F_i-R_i=2\langle K_{ia},K_{ib}\rangle_F,
\]

the same identity becomes

\[
S_{211}^{\rm native}=48F-32\langle K,K\rangle.
\tag{55.2}
\]

Thus there are two mutually exclusive accounting branches:

1. evaluate (55.1) directly from the resident full `P` bank; or
2. use only the skew Gram after a literal old engine has acquired and
   subtracted the synchronized all-distinct `48F` parallel.

Crediting (55.1) and the skew quotient simultaneously double counts the
noncommutative residual.

### 55.1 Exact reduction bill

Once `P` is resident, (55.1) needs only scalar reductions.  For dense
`Theta_i`, symmetry-reduced pair counting gives

\[
\boxed{
C_{P,\rm dense}={Lk(k+1)r^2\over N^2}.}
\tag{55.3}
\]

At `r=55,N=1024,L=10`,

\[
C_{P,\rm dense}=0.02885,k(k+1),
\]

so a one-equivalent margin permits `k<=5` before hooks.  If every
`Theta_i` is diagonal in one frozen common basis,

\[
\boxed{
C_{P,\rm diag}={2Lkr^2\over N^2}=0.05770,k,}
\tag{55.4}
\]

which permits `k<=17`.  A rank-`s` factorization per anchor obeys the
conservative reduction count

\[
C_{P,\rm rank-s}\lesssim {Ls(k+2)r^2\over N^2}
=0.02885,s(k+2).
\tag{55.5}
\]

These are reduction costs only.  Building the products from scratch costs

\[
\boxed{
C_{P,\rm build}={LkNr^3\over N^3}=1.5867,k}
\tag{55.6}
\]

at the same dimensions and fails the one-equivalent additive margin even at
`k=1`.  The branch is deployable only when the signed-H3/Jordan replacement
already pays for and retains the exact ordered `P`, or when it displaces an
equivalent existing product.

### 55.2 Exact native-action pullback for joint refitting

Holding the symmetric costate slices `Theta_i` and the common factors `U`
fixed, define

\[
\boxed{
G_{ia}:={\partial S_{211}^{\rm native}\over\partial P_{ia}}
=64\sum_b(\Theta_i)_{ab}P_{ib}
+32\sum_b(\Theta_i)_{ab}P_{ib}^T.}
\tag{55.7}
\]

Since

\[
dP_{ia}=dM_iN_a+M_i\sum_pU_{pa}dM_p,
\]

the exact symmetric-matrix gradient is

\[
\boxed{
\nabla_{M_p}S_{211}^{\rm native}
=\operatorname{sym}\!\left[
\sum_aG_{pa}N_a
+\sum_{i,a}U_{pa}M_iG_{ia}
\right].}
\tag{55.8}
\]

The factor gradient is

\[
\boxed{
{\partial S_{211}^{\rm native}\over\partial U_{pa}}
=\sum_i\langle G_{ia},M_iM_p\rangle_F,}
\tag{55.9}
\]

and the `Theta_i` gradient is simply the small native Gram
`32F_i+16R_i`.  Equations (55.7)--(55.9) differentiate the full native
action, so a joint refit need not optimize the skew residual under an
unverified fixed `48F` parallel.  Training-time reverse products are not a
deployment credit; the deployed forward still must satisfy (55.3)--(55.6).

Equation (55.1) solves local native moment acquisition.  It does not weaken
the independent gates on the common slice rank (40.8), reset recurrence
(41.3)/(54.8), q22 collision sewing (47.3), old/1111 residualization, or
endpoint value/error.  In particular, a generic width-rank H211 costate is
still dead even though each fixed low-rank contraction is cheap.

When the canonical q22 diagonal is sewn into `Theta_i` as in (47.3), the
resident Gram in (55.1) retains that diagonal automatically and no separate
`D_ab` subtraction is needed.  The collision-square quotient of Section 52
reappears only if one insists on extracting a pure all-distinct sector or if
the q22 boundary fails to sew.  This is the strongest reason to keep the
native sewn slice rather than zeroing its diagonal before factorization.

### Graph split

- `P2R0G19B_MATH_H211_NATIVE_RESIDENT_P_ACTION` — **solved exact local identity**: (55.1)--(55.2), no old-`48F` premise.
- `P2R0G19B_MATH_H211_NATIVE_P_REDUCTION_COST` — **conditional pass for low common rank**: caps (55.3)--(55.5).
- `P2R0G19B_MATH_H211_NATIVE_P_BUILD_NOGO` — **solved additive no-go**: (55.6); full products must be literal replacement-state work.
- `P2R0G19B_MATH_H211_NATIVE_REFIT_PULLBACK` — **solved exact gradient**: (55.7)--(55.9), suitable for joint source/query refitting.
- `P2R0G19B_MATH_H211_NATIVE_VS_SKEW_OWNERSHIP` — **binding branch rule**: full native reduction or old-parallel-plus-skew, never both.

## 56. Minimal common-intertwiner and Clifford-module H3 closure

Section 51 proves that a binary grading does not by itself make the surviving
harmonic H3 block small.  This section gives the exact additional conditions
under which it does become a finite deployable carrier.

### 56.1 Minimal common-intertwiner theorem

At cut \(\ell\), let

\[
\mathscr H_\ell
=P_{\perp,\ell}\mathcal H_3
\]

be the complete-old-quotiented H3 space, and let

\[
R_{\ell j}
=P_{\perp,\ell}P_3M_{F_{\ell j}}P_3P_{\perp,\ell+1}
\tag{56.1}
\]

be every live local reset/transport branch.  A \(K_\ell\)-state exact carrier
is an injective map

\[
S_\ell:\mathbb R^{K_\ell}\longrightarrow\mathscr H_\ell
\]

whose range contains all new source seeds and satisfies

\[
\boxed{
(I-\Pi_\ell)R_{\ell j}S_{\ell+1}=0
\quad\text{for every live }(\ell,j),\qquad
\Pi_\ell=S_\ell S_\ell^\dagger .}
\tag{56.2}
\]

Then the complete reduced reset is

\[
\widehat R_{\ell j}=S_\ell^\dagger R_{\ell j}S_{\ell+1},
\tag{56.3}
\]

and every endpoint is exact when its costate is restricted by
\(S_L^*\).  Conversely, any exact persistent linear carrier induces
embeddings satisfying (56.2).  Thus (56.2), not Clifford terminology, is
the necessary-and-sufficient common-intertwiner receipt.

For a fixed source and endpoint family one may quotient directions which are
never both reachable and observable.  If \(\mathcal R_\ell\) is the span of
all source words reaching cut \(\ell\), and

\[
\mathcal N_\ell
=\bigcap_{\text{future words }w}\ker(E_w)
\]

is the future-unobservable subspace, the exact minimal realization dimension
is the Kalman/Hankel rank

\[
\boxed{
K_\ell^{\min}
=\dim\mathcal R_\ell
-\dim(\mathcal R_\ell\cap\mathcal N_\ell).}
\tag{56.4}
\]

Equation (56.4) is the only lawful way a target-conditioned carrier can be
smaller than the invariant cyclic module.  Per-net scalar refits do not prove
a shared intertwiner.

### 56.2 Exact Clifford certificate and its dimensions

Let \(P\) be a common reducing projector and put \(U=P\mathbb R^r\).  A
nondegenerate \(c\)-generator graded Clifford bank has

\[
\begin{gathered}
[P,J]=0,\qquad J_P=PJ,\qquad J_P^2=P,\\
PO_a=O_aP=O_a,\qquad \{O_a,J_P\}=0,\\
\boxed{\{O_a,O_b\}=2G_{ab}P},\qquad G\succ0 .
\end{gathered}
\tag{56.5}
\]

After whitening \(G\), write

\[
J_P=\begin{pmatrix}I_s&0\\0&-I_s\end{pmatrix},
\qquad
O_a=\begin{pmatrix}0&A_a\\A_a^{\mathsf T}&0\end{pmatrix}.
\]

The Clifford identity is equivalent to

\[
\boxed{
\begin{aligned}
A_aA_b^{\mathsf T}+A_bA_a^{\mathsf T}
&=2\delta_{ab}I_s,\\
A_a^{\mathsf T}A_b+A_b^{\mathsf T}A_a
&=2\delta_{ab}I_s .
\end{aligned}}
\tag{56.6}
\]

In particular \(U_+\) and \(U_-\) have the same dimension.  Choosing \(A_1\)
as the common orthogonal intertwiner identifies the two halves, and

\[
E_a=A_aA_1^{\mathsf T},\qquad
E_a^{\mathsf T}=-E_a,\qquad
E_aE_b+E_bE_a=-2\delta_{ab}I_s
\quad(a,b>1).
\tag{56.7}
\]

Hence \(U_+\) is a real \(Cl_{0,c-1}\)-module.  The minimal half-dimensions
are

\[
\begin{array}{c|ccccccccc}
c-1&0&1&2&3&4&5&6&7&8\\ \hline
s_{\min}&1&2&4&4&8&8&8&8&16
\end{array},
\qquad s_{\min}(m+8)=16s_{\min}(m).
\tag{56.8}
\]

These divisibility conditions are necessary, not merely constructions.
Every active Clifford support has even dimension \(D=2s\).  Its full
trace-quotiented cubic envelope has exact dimension

\[
\boxed{
h_3(D)=\binom{D+2}{3}-D
=\frac{D(D-1)(D+4)}6.}
\tag{56.9}
\]

The two \(J\)-parities have equal dimension

\[
h_3^\pm(s)=\frac{s(2s-1)(s+2)}3.
\tag{56.10}
\]

Within either parity, the exact decomposition is

\[
\mathcal H^3(U_+)
\oplus\bigl(U_+\otimes\mathcal H^2(U_-)\bigr)
\oplus U_+,
\tag{56.11}
\]

with the analogous formula after swapping signs.  The final \(U_+\) is the
relative-radial bridge which survives quotienting by the **total** trace.
If the old interface additionally owns both block-radial branches and their
future returns, the remaining nonradial dimension per parity is

\[
h_{3,\mathrm{nr}}^\pm(s)
=\frac{s(s-1)(2s+5)}3.
\tag{56.12}
\]

That extra ownership is stronger than ordinary \(I\)-trace ownership; the
two-dimensional witness in Section 51 is exactly the \(s=1\) relative-radial
bridge.

For irreducible supports, the universal dimensions are therefore

\[
\begin{array}{c|c|c|c}
\text{odd generators }c&D_{\min}&h_3(D_{\min})
&2h_{3,\mathrm{nr}}^\pm\\ \hline
1&2&2&0\\
2&4&16&12\\
3\text{--}4&8&112&104\\
5\text{--}8&16&800&784\\
9&32&5952&5920
\end{array}.
\tag{56.13}
\]

At r55 no full-support nondegenerate Clifford block exists.  The largest
possible active ranks are \(54\) for \(c=1\), \(52\) for \(c=2\),
\(48\) for \(3\le c\le8\), and \(32\) for \(c=9\); \(c=10\) already
requires \(D\ge64\).  If a reducing complement has dimension \(t\), the
mixed trace-quotiented cubic sector has exact dimension

\[
\boxed{
h_3(D+t)-h_3(D)-h_3(t)
=\frac{Dt(D+t+2)}2.}
\tag{56.14}
\]

Thus even the \(54+1\) split creates \(1539\) mixed directions.  They must
be identically absent, complete-old-owned, or included in the carrier.

### 56.3 Reset invariance and literal product inventory

Clifford multiplication alone is not the reset-invariance receipt.  For the
harmonic quotient to be well defined, every deployed reset must preserve the
trace ideal,

\[
R_{\ell j}(g_U\odot U)\subseteq g_U\odot U+\mathcal O_{\rm complete},
\tag{56.15}
\]

and the chosen Clifford cyclic space must independently pass (56.2).  A
linear pullback \(A^{\odot3}\) descends to the trace quotient exactly when
\(AA^{\mathsf T}\) is scalar on \(U\).  Individual normalized Clifford words
are orthogonal and pass this test; a generic linear combination need not.

Moreover the actual estimator contains Jordan, commutator, Schur/Hadamard,
and covariance-conjugation primitives.  Let \(\mathscr S_\ell\) be the
matrix operator system used to realize the H3 atoms.  Exact closure requires

\[
\boxed{
\begin{aligned}
&I,P,J,O_a\in\mathscr S_\ell,\qquad
\mathscr S_\ell^{\mathsf T}=\mathscr S_\ell,\qquad
\mathscr S_\ell\mathscr S_\ell\subseteq\mathscr S_\ell,\\
&\mathscr S_\ell\circ R_{\ell q}\subseteq\mathscr S_\ell,\qquad
R_{\ell q}\mathscr S_\ell R_{\ell q}^{\mathsf T}
\subseteq\mathscr S_\ell
\quad\text{for every deployed mask/covariance }R_{\ell q}.
\end{aligned}}
\tag{56.16}
\]

Transport must intertwine the same representation at adjacent cuts:

\[
T_\ell P_{\ell+1}=P_\ell T_\ell,\qquad
T_\ell J_{\ell+1}=J_\ell T_\ell,\qquad
T_\ell O_{a,\ell+1}
=\sum_bH_{\ell,ba}O_{b,\ell}T_\ell,
\tag{56.17}
\]

where \(H_\ell^{\mathsf T}G_\ell H_\ell=G_{\ell+1}\).  Generic multiplicity
mixing or one generic Hadamard mask destroys the Clifford commutant and
returns the full matrix algebra.

For one direct H3 reset, only gate chaoses \(0,2,4,6\) return H3 to H3, so
at most three gate factors appear in a contraction.  Across depth those
words concatenate.  The exact persistent inventory is therefore:

1. a basis of the represented Clifford-word image, not just the generators;
2. its product/transpose structure constants;
3. every independent Hadamard and covariance-conjugation image in (56.16);
4. the vector actions needed by
   \(6\mathscr L_MT(v,A)=2T(Mv,A)+4T(v,M\circ A)\);
5. the literal \(M_iJ\) or \(M_iN_a\) products used by the q211 seam;
6. the kernel and mixed-support summaries required by (56.14).

If \(a_\ell=\dim\mathscr S_\ell\), then \(a_\ell\) is the exact generic
matrix-inventory count; fewer matrices are possible only after a fixed
future-costate aggregation like Section 52.  An approximate Clifford
relation is not exact closure unless every anticommutator error lies in a
carried invariant ideal.

### 56.4 Billing cap and endpoint-tail gates

Let \(K\) be the actual common H3 state dimension after all old quotients,
and let \(L\) be the number of active transitions.  An \(n\times K\)
coefficient bank costs \(K/n\) full-GEMM equivalents per width mix.  With
\(s_{\rm mix}\) forward/reverse mixes and a dense \(K\times K\) local reset,
the optimistic arithmetic lower bound is

\[
\boxed{
C_{\rm H3}\ \ge\
L\left(\frac{s_{\rm mix}K}{n}+\frac{K^2}{n^2}\right)
+C_{\rm products}.}
\tag{56.18}
\]

If every reduced reset is a scalar or signed permutation, the \(K^2/n^2\)
term becomes \(O(K/n^2)\), but the width-mix term remains.  Under the current
one-equivalent slack with \(n=1024,L=10\), the resulting necessary ceilings
are \(K\le102\) for one mix and \(K\le51\) for two.  Consequently the
universal \(c\ge3\) Clifford envelope is already closed: even after the
strong block-radial ownership in (56.12), its first case has \(K=104>102\).
Only the \(c\le2\) universal envelopes fit; a larger generator bank needs a
strictly smaller target-conditioned Hankel module certified by (56.4).
Product formation and non-scalar local reset work only tighten this screen.

Finally, algebraic closure and task value are independent.  Let \(Y\) be the
complete-old-residualized, endpoint-whitened panel response of the exact H3
state, and let \(\widehat Y_C\) be the response produced by the proposed
common module, including every reset-invariance defect.  The binding
end-to-end tail is

\[
\boxed{
\tau_C
=\frac{\|Y-\widehat Y_C\|_F^2}{\|Y\|_F^2}
<1-0.9736^2=0.05210.}
\tag{56.19}
\]

If \(\sigma_j\) are the singular values of the unrestricted whitened response
operator, every rank-\(K\) carrier necessarily obeys the Eckart--Young gate

\[
\boxed{
\frac{\sum_{j>K}\sigma_j^2}{\sum_j\sigma_j^2}<0.05210.}
\tag{56.20}
\]

This is only a lower bound on the error of a structured Clifford carrier;
its actual projector in (56.19) is binding.  The coherent implementation and
omitted-return error must separately remain below \(3.80\times10^{-6}\), and
the jointly Schur-residualized feature must still clear the
\(2.86\times10^{-10}\) raw-gain gate.  No value may be booked for trace/H31,
CP3, collision, or q211 coordinates already owned elsewhere.

### Graph split

- P2R0G19B_MATH_CLIFFORD_H3_ALGEBRAIC_CLOSURE — **solved iff theorem / open frozen receipt**: common intertwiner (56.2)--(56.4), Clifford certificate (56.5)--(56.12), and primitive closure (56.15)--(56.17).
- P2R0G19B_MATH_CLIFFORD_H3_COST_CAP — **solved census/no-go**: exact envelope dimensions (56.9)--(56.14) and billing gate (56.18); universal \(c\ge3\) is additive-cap NO.
- P2R0G19B_MATH_CLIFFORD_H3_OWNERSHIP — **open complete-seam receipt**: prove dynamic trace/relative-radial ownership, reducing complement and mixed-sector ownership, literal products, and no duplicate H31/CP3/q211 credit.
- P2R0G19B_MATH_CLIFFORD_H3_VALUE — **open endpoint receipt**: actual structured tail (56.19), singular lower bound (56.20), coherent error, and joint Schur gain.

## 57. One ReLU covariance reset generically destroys rank-one pair costates

The common-factor problem appears already in the ordinary Gaussian ReLU
covariance Jacobian.  For centered unit-variance Gaussian preactivations with
correlation `rho_ij`, the off-diagonal covariance map has derivative

\[
D_{ij}=\kappa'(\rho_{ij})
=\mathbb P(g_i>0,g_j>0)
=\frac14+\frac{\arcsin\rho_{ij}}{2\pi}.
\tag{57.1}
\]

Thus a reverse pair costate updates by a Hadamard multiplier.  If the future
costate is rank one,

\[
\Lambda^+=uu^T,
\]

then, ignoring only the separately owned variance/threshold derivative
terms,

\[
\boxed{
\Lambda^-=D\odot\Lambda^+
=\operatorname{diag}(u)D\operatorname{diag}(u).}
\tag{57.2}
\]

Whenever every active `u_i` is nonzero, (57.2) has exactly the rank of `D`.

### 57.1 Generic full-rank theorem

Write the correlation matrix as `rho_ij=x_i^T x_j` with unit vectors `x_i`.
The absolutely positive feature expansion

\[
\arcsin t=\sum_{m=0}^{\infty}
{(2m)!\over4^m(m!)^2(2m+1)}t^{2m+1}
\tag{57.3}
\]

shows that `D` is a Gram matrix containing a constant feature and every odd
symmetric tensor power of `x_i`.  It is generically strictly positive
definite for any finite `N`, even with latent dimension two.  An explicit
proof takes

\[
x_i=(\cos\theta_i,\sin\theta_i)
\]

with no antipodal pair.  If `a^TDa=0`, positivity of every coefficient in
(57.3) forces all odd tensor moments to vanish.  In complex notation
\(z_i=e^{i\theta_i}\), this includes

\[
\sum_i a_i z_i^{2m+1}=0,
\qquad m=0,\ldots,N-1.
\]

Putting `b_i=a_i z_i` gives the Vandermonde system

\[
\sum_i b_i(z_i^2)^m=0.
\]

The nodes `z_i^2` are distinct, so every `b_i`, hence every `a_i`, is zero.
Therefore one configuration has `det D>0`; analyticity makes `det D` nonzero
for a generic configuration of latent directions.  Consequently

\[
\boxed{\operatorname{rank}D=N\quad\text{generically}.}
\tag{57.4}
\]

The low rank of the correlation matrix itself
\((\operatorname{rank}\rho\le r)\) does not bound
the rank of the entrywise arcsine kernel.  Biases replace (57.1) by a
bivariate Gaussian CDF kernel and are generically no better; (57.4) is
already a zero-bias obstruction.

### 57.2 Consequence for the sewn q211/Q22 reverse word

A scalar endpoint may start with a rank-one covariance costate, but after one
generic ReLU pair reset its exact common pair-factor width is `N`.  Hence the
low-rank native-`P` caps in Section 55 cannot be inferred from scalar endpoint
conditioning.  They require one of the following independent receipts:

1. the complete-old quotient removes all but the admitted `k<=5` dense or
   `k<=17` common-diagonal directions;
2. the realized pair reset has a special low-rank/separable kernel;
3. a target-conditioned observation factors the full future action before
   the Hadamard reset, with the kernel inclusion proved as in Section 28; or
4. only an approximate factor is claimed and its discarded singular-value
   tail passes the end-to-end value/error gates.

Exact full rank alone does not prove a large **energy** tail; a numerical
singular spectrum or an analytic lower frame is still needed to close an
approximate low-rank proposal.  It does prove that rank-one exactness, and
therefore a one-word recurrence without an old quotient, is generically
false after a single covariance reset.

### Graph split

- `P2R0G19B_MATH_RELU_PAIR_ARCSINE_PULLBACK` — **solved exact identity**: (57.1)--(57.2).
- `P2R0G19B_MATH_RELU_PAIR_GENERIC_FULL_RANK` — **solved generic theorem**: positive odd-feature/Vandermonde proof (57.3)--(57.4).
- `P2R0G19B_MATH_Q211_SCALAR_ENDPOINT_RANK_NOGO` — **solved exact-rank no-go / approximate-tail open**: scalar conditioning does not preserve a common rank-one pair word.

## 58. Conditional analytic tail bound for the residual arcsine pair kernel

Exact full rank does not by itself rule out an accurate low-rank
approximation.  A tight-frame receipt does.  Let

\[
\rho=XX^T,
\qquad \|x_i\|=1,
\qquad X^TX={N\over r}I_r,
\qquad \mathbf1^TX=0,
\tag{58.1}
\]

and define the off-diagonal coherence

\[
\mu=\max_{i\ne j}|\rho_{ij}|<1.
\]

Remove the old collective `1/4` kernel and apply the canonical diagonal q22
sewing which removes only the excess

\[
\delta={\pi/2-1\over2\pi}
\]

from the diagonal.  The resulting complete sewn noncollective kernel is

\[
\boxed{
A={1\over2\pi}\rho+E,}
\tag{58.2}
\]

where `E_ii=0` and

\[
E_{ij}={\arcsin\rho_{ij}-\rho_{ij}\over2\pi}
\quad(i\ne j).
\]

Since `arcsin(t)/t` is increasing on `(0,1)`, put

\[
\varepsilon(\mu)={\arcsin\mu\over\mu}-1.
\tag{58.3}
\]

Then

\[
\|E\|_F
\le\varepsilon(\mu)\left\|{\rho\over2\pi}\right\|_F.
\tag{58.4}
\]

The `r` nonzero singular values of `rho/(2pi)` are all
`N/(2pi r)`.  By the one-Lipschitz property of distance to the rank-`k`
variety,

\[
\begin{aligned}
\operatorname{dist}_F(A,\operatorname{rank}\le k)
&\ge \operatorname{dist}_F(\rho/(2\pi),\operatorname{rank}\le k)
-\|E\|_F\\
&\ge \left\|{\rho\over2\pi}\right\|_F
\left(\sqrt{1-k/r}-\varepsilon(\mu)\right)_+.
\end{aligned}
\tag{58.5}
\]

Also `||A||_F<=(1+epsilon)||rho/(2pi)||_F`, so every rank-`k`
factor obeys the relative energy-tail bound

\[
\boxed{
\tau_k(A)
\ge
\left[
{(\sqrt{1-k/r}-\varepsilon(\mu))_+
 \over1+\varepsilon(\mu)}
\right]^2.}
\tag{58.6}
\]

Equivalently, to meet a required tail `tau_star`, it is necessary that

\[
\boxed{
{k\over r}
\ge1-\left[
\varepsilon(\mu)+(1+\varepsilon(\mu))\sqrt{\tau_\star}
\right]^2.}
\tag{58.7}
\]

The tight-frame hypothesis is needed only to make the spectrum explicit.
For an arbitrary sewn Gram define

\[
q_k(\rho)=
{\left(\sum_{j>k}\sigma_j(\rho)^2\right)^{1/2}\over\|\rho\|_F}.
\]

The same proof gives the general certificate

\[
\boxed{
\tau_k(A)\ge
\left[{(q_k(\rho)-\varepsilon(\mu))_+
\over1+\varepsilon(\mu)}\right]^2.}
\tag{58.8}
\]

In the ideal incoherent limit `epsilon -> 0`, the `3.28064%` one-stream tail
gate requires `k/r>=0.96719`; at `r=55` this means `k>=54`.  Even the looser
`5.210%` H3 envelope gate requires `k/r>=0.94790`, hence `k>=53`.  Small
nonzero coherence weakens these integers only slightly through the explicit
factor (58.7).  Therefore the resident-`P` additive caps `k<=5` dense and
`k<=17` diagonal cannot preserve a tight diffuse noncollective pair kernel.

This is a conditional theorem, not a claim that the frozen nets satisfy
(58.1).  The minimal lawful certificate is the spectrum/tightness of the
post-old, collision-sewn pair feature frame plus its off-diagonal coherence.
If the actual response is task-weighted, (58.6) must be applied to the
whitened response Gram rather than the unweighted local kernel.  A special
endpoint observation can still win only by proving that its future-visible
Hankel spectrum is much thinner than this local frame.

### Graph split

- `P2R0G19B_MATH_RELU_PAIR_TIGHT_FRAME_TAIL` — **solved conditional lower bound**: (58.1)--(58.8).
- `P2R0G19B_MATH_RELU_PAIR_FROZEN_FRAME_RECEIPT` — **open lawful observable**: certify post-old/collision-sewn tightness, coherence, and task weighting.
- `P2R0G19B_MATH_Q211_LOW_K_DIFFUSE_NOGO` — **conditional no-go**: under the frame receipt, `k<=5/17` cannot meet the required endpoint-tail gate.

## 59. The exact target-weighted escape test for native resident-P H211

The generic arcsine rank theorem is not the final decision object: only
future-visible, post-old, task-weighted directions matter.  This section
defines that object exactly.

Let `mathcal X_nat` be the Hilbert direct sum of all synchronized native
H211/Q22 source slices, with a positive source metric `Gamma` spanning the
lawful source probes/consumers.  Let `mathcal Y` be the corpus endpoint space
with task weight `W`, and let

\[
F_{\rm nat}:\mathcal X_{\rm nat}\longrightarrow\mathcal Y
\]

be the **full native** response operator represented locally by (55.1),
including both `48F` and the commutator term.  If `A` is the complete retained
old/other-feature design, write `P_A^\perp` for its `W`-orthogonal residual
projector.  The binding whitened response operator is

\[
\boxed{
\mathscr R_{\rm nat}
=W^{1/2}P_A^\perp F_{\rm nat}\Gamma^{1/2}.}
\tag{59.1}
\]

For a common frame `U in R^{N by k}`, let `P_U=UU^T` and let `Pi_U` act on
every pair slice by

\[
\Pi_U(X)=P_UXP_U.
\tag{59.2}
\]

The exact best structured target-weighted tail is

\[
\boxed{
\tau_k^{\rm cf}
=\min_{U^TU=I_k}
{\|\mathscr R_{\rm nat}(I-\Pi_U)\|_{\rm HS}^2
 \over\|\mathscr R_{\rm nat}\|_{\rm HS}^2}.}
\tag{59.3}
\]

Layer-dependent frames use the corresponding block-diagonal
`Pi_{U_1,...,U_L}`; cross-layer endpoint coherence remains inside the single
Hilbert--Schmidt norm in (59.3) and must not be replaced by a sum of local
tails.

The sharp exact-completeness equivalences are

\[
\boxed{
\tau_k^{\rm cf}=0
\iff
\mathscr R_{\rm nat}=\mathscr R_{\rm nat}\Pi_U
\iff
\operatorname{range}\mathscr R_{\rm nat}^*\subseteq
\operatorname{range}\Pi_U
\iff
\ker\Pi_U\subseteq\ker\mathscr R_{\rm nat}.}
\tag{59.4}
\]

Thus target conditioning is a lawful escape precisely when the omitted pair
directions are future-dark for the full native operator, not merely because
one fitted endpoint scalar is reproducible.

For an actual whitened source `xi`, the coherent absolute omission is

\[
\eta_\xi(U)=
\|\mathscr R_{\rm nat}(I-\Pi_U)\xi\|_2.
\tag{59.5}
\]

Promotion requires the applicable value tail (currently `3.28064%` for the
one-stream closer) **and** `eta_xi<=3.80e-6`; the jointly Schur-residualized
gain must separately exceed `2.86e-10`.

### 59.1 Pair-mode SVD: the fail-fast lower bound

Unfold `mathscr R_nat^*` with the symmetric pair coordinate as rows and all
layer/anchor/corpus/endpoint coordinates as columns; call the resulting
matrix `Z_nat`.  Whitening in (59.1) makes

\[
\|Z_{\rm nat}\|_F=\|\mathscr R_{\rm nat}\|_{\rm HS}.
\]

The support `Sym^2(U)` has dimension

\[
d_k={k(k+1)\over2}.
\]

Therefore every common-frame proposal obeys the unrestricted Eckart--Young
lower bound

\[
\boxed{
\tau_k^{\rm cf}
\ge
{\sum_{j>d_k}\sigma_j(Z_{\rm nat})^2
 \over\sum_j\sigma_j(Z_{\rm nat})^2}.}
\tag{59.6}
\]

Passing (59.6) is necessary, not sufficient, because an arbitrary
`d_k`-dimensional pair subspace need not equal `Sym^2(U)` and the frame must
also survive every reverse reset.

The billing caps of Section 55 make this especially sharp:

- dense `Theta`, `k<=5`: `d_k<=15`;
- common-diagonal `Theta`, `k<=17`: the admissible pair space is only
  `span{u_a u_a^T}`, of dimension at most 17, not `d_17=153`.

Hence the first fail-fast receipts are respectively the singular tails after
15 and 17 target-whitened native pair modes.  If either exceeds `3.28064%`,
that billed branch is closed before any nonlinear frame optimization.

### 59.2 Strict common-support receipts

Let `Lambda_c` denote every symmetric pair-costate slice occurring as a
column of the unwhitened adjoint response, where `c` ranges over the complete
lawful source/endpoint contexts.  Let `mathcal N_c` be the sum of its exact
old-equivalent and native-source-annihilator subspaces.  The smallest exact
dense common support is

\[
\boxed{
k_{\rm strict}
=\min_{X_c\in\mathcal N_c}
\operatorname{rank}[\Lambda_1-X_1\ \cdots\ \Lambda_C-X_C].}
\tag{59.7}
\]

The dense resident-`P` branch is exact only if `k_strict<=5` and the same
support passes the reverse recurrence (41.3)/(54.8).  The common-diagonal
branch is exact only if `k_strict<=17` **and**, for one minimizing family,

\[
[\Lambda_c-X_c,\Lambda_d-X_d]=0
\quad\text{for every }c,d.
\tag{59.8}
\]

Real symmetric matrices satisfying (59.8) are simultaneously orthogonally
diagonalizable, making (59.8) the sharp extra diagonal-frame receipt.  Small
per-slice eigenspaces or per-net fitted scales do not imply (59.7)--(59.8).

If `Gamma` has rank one because it contains only one scalar response per net,
(59.3) can be spoofed by net-specific amplitudes.  A corpus-complete receipt
must span multiple independent native consumers of the same frozen slices,
or impose a shared frozen coefficient law; this is the tomography condition
of Section 50 in the counterexample lane.

### 59.3 Keep the three quartic operators distinct

Write the native operator as

\[
\mathscr R_{\rm nat}=\mathscr R_{48F}+\mathscr R_{\rm comm}.
\tag{59.9}
\]

The skew-only tail may replace (59.3) only after the operator equality

\[
P_A^\perp F_{48F}=0
\tag{59.10}
\]

is proved on `mathcal X_nat`.  A scalar correlation or the word "parallel"
does not prove (59.10).  CP4 rank-four/collision acquisition is a third
operator `F_CP4`; its local collision reads cannot be substituted for
`F_48F` without an explicit synchronized Jacobian equality.  Therefore:

1. use full-native `Z_nat` and (55.1) by default;
2. use the skew-only response only under (59.10); and
3. residualize CP4 separately in `P_A`, without crediting an all-distinct
   port it does not possess.

### Graph split

- `P2R0G19B_MATH_H211_TARGET_WEIGHTED_OPERATOR` — **solved decision definition**: full-native whitened response (59.1)--(59.5).
- `P2R0G19B_MATH_H211_PAIR_MODE_SVD_GATE` — **open observable / solved fail-fast bound**: compute the post-15/post-17 tails in (59.6).
- `P2R0G19B_MATH_H211_STRICT_COMMON_FRAME` — **solved iff receipt / open frozen check**: (59.7)--(59.8) plus reverse recurrence.
- `P2R0G19B_MATH_H211_CORPUS_COMPLETE_WEIGHTING` — **open anti-spoof receipt**: source metric spans shared lawful consumers, not one free scalar per net.
- `P2R0G19B_MATH_H211_NATIVE_SKEW_CP4_SPLIT` — **binding ownership split**: (59.9)--(59.10); native, skew-only, and CP4 are distinct operators.

## 60. Full-AQ replacement: exact q211 ownership boundary and residual slots

There are two different objects which must not be conflated.  At an AQ
reset let

\[
Y_i^A=(g_i^A)_+,
\qquad X_i=P_{\le2}Y_i^A.
\]

Carrying the affine--quadratic state `X_i` owns `kappa_4(X)`.  It does
**not** by itself own the connected q211 of the nonlinear output
`kappa_4(Y^A)`.  In particular

\[
R_{AQ,4}:=\kappa_4(Y^A)-\kappa_4(X)
\tag{60.0}
\]

is nonzero already for a centered affine ReLU.  An exact Gaussian/Mehler
covariance rest is a two-point observable and does not own (60.0).

There is a sharp one-dimensional witness.  For `Z~N(0,1)`, put
`Y=Z_+` and `a=(2pi)^(-1/2)`.  Its order-two Hermite projection is

\[
X=P_{\le2}Y=a+{1\over2}Z+{a\over2}(Z^2-1).
\]

Direct centered-moment algebra gives

\[
\begin{aligned}
\kappa_4(Y)
&={3\over4}-{1\over\pi}-{3\over2\pi^2},\\
\kappa_4(X)
&={3\over2\pi}+{3\over4\pi^2},\\
\boxed{\kappa_4(Y)-\kappa_4(X)
&={3\over4}-{5\over2\pi}-{9\over4\pi^2}
\simeq-0.27375.}
\end{aligned}
\tag{60.0b}
\]

Thus even the simplest affine reset gives a large exact counterexample to
the inference `low-chaos state + covariance rest => connected H4 ownership`.
Taking three external labels with the same latent Gaussian makes their
`iijk` cumulant equal to (60.0b); continuity then gives nonzero affine q211
on nearby positive-definite correlation matrices.  The witness is therefore
not confined to the one-label `(4)` collision sector.

Let `iota_surf` equal one only when an explicitly billed fused connected-H4
reset-surface oracle is part of the declared replacement.  The actual owned
quartic baseline is

\[
\boxed{
K_4^{\rm owned}
=\kappa_4(X)+\iota_{\rm surf}R_{AQ,4}.}
\tag{60.0a}
\]

Only under `iota_surf=1` does the replacement own the complete AQ q211,
including both the commuting `48F` and commutator pieces.  In that branch,
evaluating (55.1) on the same AQ matrices is an alternate implementation of
retained value.  Under `iota_surf=0`, the same native action can instead be
a candidate implementation of the missing surface term (60.0); it is not a
duplicate until its equality to the nonlinear AQ action, its subtraction of
`kappa_4(X)`, and its billing are established.

For the true-minus-AQ part, decompose the synchronized true slot as

\[
Y_i^T=Q_i+E_i,
\qquad Q_i:=Y_i^A,
\tag{60.1}
\]

where `E_i` is the true-minus-AQ residual in the identical latent/gate
normalization.  Let
`mathcal K(A,B,C,D)` denote the fully polarized connected H4 functional
before the external 211 collision projection.  Multilinearity gives the
exact repeated-anchor difference

\[
\boxed{
\begin{aligned}
\Delta\mathcal K_{iijk}
={}&2\mathcal K(E_i,Q_i,Q_j,Q_k)
+\mathcal K(Q_i,Q_i,E_j,Q_k)
+\mathcal K(Q_i,Q_i,Q_j,E_k)\\
&+\mathcal K(E_i,E_i,Q_j,Q_k)
+2\mathcal K(E_i,Q_i,E_j,Q_k)
+2\mathcal K(E_i,Q_i,Q_j,E_k)\\
&+\mathcal K(Q_i,Q_i,E_j,E_k)
+\mathcal K(E_i,E_i,E_j,Q_k)
+\mathcal K(E_i,E_i,Q_j,E_k)\\
&+2\mathcal K(E_i,Q_i,E_j,E_k)
+\mathcal K(E_i,E_i,E_j,E_k).
\end{aligned}}
\tag{60.2}
\]

The multiplicities in (60.2) sum to `15`, one for every nonempty residual
subset of the four slots.  Thus every term in the **true-minus-AQ bracket**
contains at least one `E`; (60.0) is a separate zero-residual omission when
the surface oracle is absent.

Let `P_211` impose the canonical q211/q22 collision sewing and let
`P_complete^perp` residualize the actually owned AQ baseline, exact
Gaussian/Mehler covariance rest, CP4/star/pair, and every other retained
provider.  The full lawful omitted q211 is

\[
\boxed{
R_{211}^{\rm omit}
=P_{\rm complete}^{\perp}P_{211}
\left[\Delta\mathcal K
+(1-\iota_{\rm surf})R_{AQ,4}\right].}
\tag{60.3}
\]

When `iota_surf=1`, (60.3) reduces to the true-minus-AQ residual and every
novel term has a residual slot.  When it is zero, the AQ H4 rest must also be
included.  Equation (60.3), not the raw native q211 and not the AQ
commutator Gram alone, is the response whose target-weighted unfolding and
tail must be inserted into Section 59.  The exact Gaussian/Mehler engine may
remove covariance-generated two-point parts; primitive connected-H4
residuals remain automatically through the single complete projector.

For alignment with the existing host/Hermite interface, insert the
synchronized intermediate reconstruction `K4^H`:

\[
K_4^T-K_4^{AQ}
=(K_4^T-K_4^H)+(K_4^H-K_4^{AQ}).
\tag{60.3a}
\]

After the same masks and costates are applied, the complete residual can be
named as

\[
\boxed{
R_{211}^{\rm omit}
=P_{\rm complete}^{\perp}
\left(P_H^{T-H}+R_{2111}^{T-H}+R_{H-AQ}
+(1-\iota_{\rm surf})R_{AQ,4}\right).}
\tag{60.3b}
\]

Here `P_H^{T-H}` is the part actually exposed by the synchronized
truth-minus-host collision planes, `R_2111^{T-H}` is their collision-free
211/1111 complement, and `R_H-AQ` is the host/Hermite-to-AQ modeling
residual.  The final term is present exactly when the nonlinear AQ surface
is not owned.  This is an ownership split, not independent gains: they sum
coherently before the endpoint norm.  An `eta_2111` summary bounds the middle
term only under identical masks/weights, and neither `dK21` (an H3 object)
nor endpoint-cancelled eta summaries identify the first or last term.

### 60.1 Which residual pieces a native-P bank can acquire

Decompose the residual further into a quadratic part and a higher-chaos
part,

\[
E_i=Q_{\Delta M_i}+E_i^{\ge3}.
\tag{60.4}
\]

Independently of the quadratic/higher-chaos split, group (60.2) by the
number of residual slots:

\[
\Delta\mathcal K
=\Delta^{(1)}+\Delta^{(2)}+\Delta^{(3)}+\Delta^{(4)},
\tag{60.4a}
\]

with multiplicities `4,6,4,1`.  An explicitly acquired tilted generator

\[
\mathcal H_{E_i}(t)=E\left[E_i
\exp\left(\sum_jt_jQ_j\right)\right]
\tag{60.4b}
\]

and its connected third jet determine `Delta^(1)` exactly, including the
tied-anchor factor two.  This is a separate billed one-residual surface
oracle; it is not implied by carrying `X` or by the Gaussian covariance
rest.

No collection of the univariate generators (60.4b) identifies
`Delta^(2)` for generic residual sources.  Exact acquisition of the
remaining hierarchy needs joint generators indexed by residual-slot
**multisets** (so the repeated anchor may occur twice),

\[
\mathcal H_{I}(t)=E\left[
\prod_{i\in I}E_i
\exp\left(\sum_jt_jQ_j\right)\right],
\qquad I\text{ a multiset},\quad |I|=2,3,4,
\tag{60.4c}
\]

or an equal resident residual Gram/tensor action.  The two-residual level is
already the missing non-Gaussian-rest Gram in the r49 architecture.  Thus,
when both the fused AQ H4 surface and the one-residual tilted generator are
explicitly owned, the first possible novel q211 is
`P_complete^perp P_211(Delta^(2)+Delta^(3)+Delta^(4))`; without either
oracle its corresponding earlier block remains in (60.3).

The resident-P identity can acquire exactly the portion of (60.2) in which
every slot is represented by a quadratic matrix (`M_i^AQ` or `Delta M_i`).
Any term containing `E_i^{>=3}` is not identified by AQ matrices or their
ordered products; its one-residual part needs (60.4b), while two or more
residual slots need (60.4c) or an equivalent source.  Applying native-P only
to `M_i^AQ` returns the AQ q211.  It gives zero novelty when
`iota_surf=1`; when `iota_surf=0`, it can count only as a billed
implementation of (60.0), after subtracting the already-owned
`kappa_4(X)` action.

For a fixed common costate frame, let

\[
P_{ia}^{A}=M_i^{A}N_a^{A},
\qquad
P_{ia}^{T}=M_i^{T}N_a^{T},
\qquad
\Delta P_{ia}=P_{ia}^{T}-P_{ia}^{A}.
\]

With the symmetric bilinear native contraction

\[
\mathcal G(P,Q)=32\langle P,Q\rangle_F
+16\langle P^T,Q\rangle_F,
\tag{60.5}
\]

the exact quadratic residual action is

\[
\boxed{
S_{211}^{T}-S_{211}^{A}
=\sum_{i,a,b}(\Theta_i)_{ab}
\left[
2\mathcal G(P_{ia}^{A},\Delta P_{ib})
+\mathcal G(\Delta P_{ia},\Delta P_{ib})
\right].}
\tag{60.6}
\]

Equation (60.6) contains the AQ--residual cross terms and residual square
exactly once.  Computing only `G(Delta P,Delta P)` misses the leading
one-residual physics.  Adding (55.1) for `P^A` double counts only when the
fused AQ H4 surface is explicitly retained; otherwise it is a candidate for
the missing AQ surface block and must be compared to (60.0), not silently
discarded.

### 60.2 What the AQ tape actually owns

When the explicitly billed fused AQ H4 surface tape is present, it stores
per-unit matrices/eigensystems, resolvents, surface moments, and their
symmetric parameter gradients.  Those objects determine the AQ q211 scalar
internally, but they do not automatically retain
the cross-unit ordered bank

\[
P_{ia}^{A}=M_i^{A}\left(\sum_jU_{ja}M_j^{A}\right)
\tag{60.7}
\]

at the synchronized pre-symmetrization seam.  Per-unit spectral data and a
symmetric gradient are not a literal ordered-product reuse receipt.  The AQ
replacement can claim zero build cost only if (60.7), or an exactly equal
fused directional bilinear action for (60.6), is exposed on its forward or
reverse tape.

The obstruction is visible directly from (60.5).  Writing
`P=S_P+A_P` into symmetric and skew parts gives

\[
\boxed{
\mathcal G(P,Q)
=48\langle S_P,S_Q\rangle_F
+16\langle A_P,A_Q\rangle_F.}
\tag{60.7a}
\]

A post-symmetrization surface tape therefore loses an independently needed
skew action.  Moreover a first parameter derivative can return the linear
cross term only after the exact direction is supplied; it does not identify
the finite square `G(Delta P,Delta P)`.  Literal zero-bank reuse requires
same-seam full ordered `P^A` and `Delta P`, or an exactly equal fused oracle
which returns both the bilinear cross and quadratic direction action.

If the full true quadratic matrices and the AQ matrices are resident but no
ordered bank is retained, one newly built full product bank suffices to form
`P^T`; `P^A` must still be resident or separately built to evaluate the
cross term in (60.6).  Expanding directly gives

\[
\Delta P_{ia}
=M_i^A\Delta N_a+\Delta M_iN_a^A+\Delta M_i\Delta N_a,
\tag{60.8}
\]

so absent either full-product reuse or special low-rank structure, three
ordered product families are present.

### 60.3 r49 / nine-equivalent cap

At `r=49,N=1024,L=10`, one fresh `n by k` ordered-product bank costs

\[
c_{\rm build}={Lr^3\over N^2}=1.12199
\quad\text{equivalents per common factor},
\tag{60.9}
\]

while the full-native reductions cost

\[
c_{\rm dense}=0.02290,k(k+1),
\qquad
c_{\rm diag}=0.04580,k.
\tag{60.10}
\]

Under a nine-equivalent headroom, before hooks and memory traffic:

| fresh ordered banks | dense common frame | common-diagonal frame |
|---:|---:|---:|
| 0 (literal AQ-tape reuse) | reduction-only; `k<=19` | reduction-only; arithmetic not binding below `k=196` |
| 1 (`P^A` resident; build `P^T` or conversely) | `k<=6` | `k<=7` |
| 2 (build both `P^A,P^T`) | `k<=3` | `k<=3` |
| 3 (three-term (60.8), but `P^A` resident) | `k<=2` | `k<=2` |
| 4 (three-term (60.8) and build `P^A`) | `k<=1` | `k<=1` |

The zero-bank row is lawful only under the literal tape identity above; the
fused AQ scalar by itself does not establish it.  These caps apply to the
quadratic residual in (60.4), not to terms with a genuine higher-chaos slot.
Whenever `P^T` is directly formable, the two-bank route dominates expanding
all three terms of (60.8); the latter rows matter only under a structured
operand-reuse claim.

### 60.4 Binding target-weighted residual gate

Define `R_res` exactly as in (59.1) but with the response (60.3), and unfold
`R_res^*` to `Z_res`.  This is the possible escape from the generic full-rank
native kernel: the AQ replacement may leave a thin true-minus-AQ response
even when raw native q211 is broad.  The decision sequence is:

1. declare whether the fused connected-H4 surface is actually billed, then
   use (60.3) with the correct `iota_surf`; never infer q211 ownership from
   the D10 covariance rest or subtract only reported scalar scores;
2. split quadratic versus higher-chaos sources by (60.4) and split one-
   versus multi-residual acquisition by (60.4a)--(60.4c);
3. compute or certify the post-`d_k` singular tail of `Z_res` for the
   appropriate row of the cost table;
4. prove the structured common-frame/reverse recurrence, not only the SVD
   lower bound; and
5. require tail below `3.28064%`, coherent error below `3.80e-6`, and joint
   Schur gain above `2.86e-10`.

### Graph split

- `P2R0G19B_MATH_AQ_Q211_COMPLETE_OWNERSHIP` — **solved conditional iff / open declared-oracle receipt**: carried `P<=2` owns only `kappa4(X)`; full AQ q211 is owned only with the explicitly billed connected-H4 surface (60.0)--(60.0a).
- `P2R0G19B_MATH_TRUE_MINUS_AQ_Q211_RESIDUAL` — **solved exact decomposition / open response**: zero-residual AQ surface plus the `4/6/4/1` residual-slot hierarchy (60.2)--(60.4c).
- `P2R0G19B_MATH_AQ_Q211_THREE_BLOCK_OWNERSHIP` — **solved ownership split / open observables**: coherent plane, 2111, H-to-AQ, and conditional AQ-surface terms (60.3a)--(60.3b).
- `P2R0G19B_MATH_AQ_RESIDUAL_NATIVE_P_ACTION` — **solved quadratic identity**: cross-plus-square action (60.5)--(60.6); higher residual slots excluded.
- `P2R0G19B_MATH_AQ_NATIVE_P_TAPE_REUSE` — **open literal seam receipt**: (60.7)--(60.8), fused AQ scalar alone is insufficient.
- `P2R0G19B_MATH_AQ_R49_RESIDUAL_P_CAP` — **solved conditional census**: (60.9)--(60.10) and the zero/one/two/three-bank table.
- `P2R0G19B_MATH_AQ_Q211_RESIDUAL_TAIL` — **open highest-value observable**: target-whitened `Z_res` tail and structured recurrence under the five gates above.

## 61. Audit of a native pure-q211 atom `Sym(A,A,B,C)`

Consider the generic symmetric atom

\[
H=\operatorname{Sym}(a\otimes a\otimes b\otimes c),
\qquad \dim\operatorname{span}\{a,b,c\}=3.
\tag{61.1}
\]

This is the tensor form of the ternary monomial `a^2 b c`.  Carrying its
three generators is genuinely more economical than first converting it to
quartic CP rays: its complex symmetric/Waring rank is six, whereas the
native generator support has dimension three.

The linear reverse/source interfaces nevertheless have rank four, not three.
With `S(x,y)=xy^T+yx^T`, the repeated-anchor slice is exactly

\[
\boxed{
H_{ii::}={1\over12}\left[
a_i^2S(b,c)
+2a_ib_iS(a,c)
+2a_ic_iS(a,b)
+2b_ic_i aa^T
\right].}
\tag{61.2}
\]

Therefore:

\[
\begin{aligned}
\text{common pair support}&=\operatorname{span}\{a,b,c\},
&&k_{\rm support}=3,\\
\text{pair-flattening span}&=
\operatorname{span}\{S(b,c),S(a,c),S(a,b),aa^T\},
&&r_{\rm pair}=4,\\
\text{anchor coefficient span}&=
\operatorname{span}\{a^{\odot2},a\odot b,a\odot c,b\odot c\},
&&r_{\rm anchor}=4
\end{aligned}
\tag{61.3}
\]

generically.  Thus a three-generator nonlinear representation is exact, but
any linear plane/read interface must accommodate four independent channels.
The proposed `3 K31 + 4 K22` outer-product read is consistent with this rank
and may be a sufficient static read; it is not a proof that three linear
costate modes suffice.

The advertised `3+4` count can in fact be made exact at a synchronized read
seam.  In the convention `H(x,x,x,x)=(a^Tx)^2(b^Tx)(c^Tx)`, the collision
planes are

\[
\boxed{
H_{iiij}={1\over4}\left[
 (a_i^2b_i)c_j+(a_i^2c_i)b_j
 +2(a_ib_ic_i)a_j\right],}
\tag{61.3a}
\]

and

\[
\boxed{
H_{iijj}={1\over6}\left[
 a_i^2(b_jc_j)+(b_ic_i)a_j^2
 +2(a_ib_i)(a_jc_j)+2(a_ic_i)(a_jb_j)
 \right].}
\tag{61.3b}
\]

Thus the full K31 matrix is the sum of three outer products, and the full
K22 matrix is the sum of four.  Their common diagonal is
`H_iiii=a_i^2b_ic_i`, so the canonical K4/K31/K22 collision sewing is
available by diagonal reductions.  Equations (61.3a)--(61.3b) certify the
static plane-read arithmetic, including its constants.

They do **not** turn the external pure-211 mask into a rank-three transported
object.  Removing `j=k`, `j=i`, and `k=i` is an anchor-dependent
inclusion--exclusion operation.  It is cheap if applied once at the same
terminal seam using all three collision planes, but it does not commute with
dense transport or a correlated reset.  Therefore the `3+4` count is a
terminal-read receipt only; persistent use still requires (61.9)--(61.10).

For `B` independent atoms, the generic union has support dimension `3B` and
pair/anchor unfolding rank `4B` until capped by the ambient spaces.  Ten
unrelated births therefore mean support 30 and unfolding rank 40, not a
single rank-three response.  Shared generators or target-dark linear
combinations must be proved if the estimate assumes less.

### 61.1 Exact transport versus reset closure

Every common-leg linear map preserves the atom variety exactly:

\[
L^{\odot4}H
=\operatorname{Sym}(La,La,Lb,Lc).
\tag{61.4}
\]

Consequently dense transport costs only the three generator transports per
live atom, and the claimed plane reads can be applied at a terminal cut.

A coordinatewise/correlated ReLU reset is a different operator.  Let
`T_l=P_complete^perp P4 M_{F_l}P4` be its complete-old-quotiented H4 action,
and let `C_211` be the Chow variety of atoms (61.1).  Exact one-group
persistence requires

\[
\boxed{T_lH\in C_{211}\quad\text{for every carried }H,}
\tag{61.5}
\]

or, for a fixed target family, that the component outside the chosen sum of
groups be future-dark.  A sufficient structural receipt is that `T_l` acts
as one common leg map on the three-generator support modulo complete old:

\[
T_l|_{\operatorname{Sym}^4\operatorname{span}\{a,b,c\}}
=\gamma_l L_l^{\odot4}\pmod{\mathcal O_{\rm complete}}.
\tag{61.6}
\]

Generic correlated reset operators do not satisfy (61.6): repeated-anchor
and singleton slots acquire different surface/collision multipliers, and
the finite H2/H4/H6/H8 contractions in (54.1) create pair slices outside
the four-dimensional span in (61.3).  The exact local failure is measured by

\[
\epsilon_l(H)=
\inf_{\widehat H\in C_{211}^{(R)}}
\|P_{\rm future,l}(T_lH-\widehat H)\|,
\tag{61.7}
\]

where `C_211^(R)` is the allowed sum of `R` groups and `P_future,l` is the
complete suffix-observable quotient.  End-to-end coherent accumulation of
(61.7) must remain below `3.80e-6`.  Static K31/K22 read identities do not
bound it.  H4-to-H6/H8-to-H4 temporal returns remain a separate closure gate
as in Sections 49 and 54.

There is a sharp algebraic witness already in the H2 tangent.  Let the
one-leg derivation `D_M` satisfy

\[
Ma=d\notin\operatorname{span}\{a,b,c\},
\qquad Mb=Mc=0.
\]

Then, in polynomial notation,

\[
(I+tD_M)(a^2bc)=a^2bc+2tadbc=a(a+2td)bc.
\tag{61.8}
\]

For generic nonzero `t`, unique factorization gives four distinct linear
factors, so (61.8) cannot equal `tilde a^2 tilde b tilde c`.  The missing
`t^2d^2bc` term would be present in a true common-leg map
`(I+tM)^{odot4}` but is absent from the isolated tangent.  Hence the H2
block alone splits one q211 group into a four-generator/all-distinct sibling;
the higher direct gate blocks preserve one group only if their coefficients
complete exactly the common-leg binomial.  If that sibling is claimed old,
its complete future ownership must be shown explicitly.

### 61.2 Source rank and native-P ownership

There are two lawful orientations:

1. **Forward source atom.**  A physical provider supplies an amplitude
   `alpha_l` and the three generators; the atom is transported by (61.4) and
   read once at the endpoint.
2. **Reverse native-P action.**  A complete future q211 costate is contracted
   directly with the quadratic source by (55.1), or with the true-minus-AQ
   quadratic residual by (60.6).

If both represent the same source/query pairing, adjoint duality makes their
scalar contributions identical.  Adding both double counts.  Native-P does
not determine the amplitude of an arbitrary residual atom: it supplies that
action only when the atom is the exact q211 of resident quadratic matrices.
Using AQ matrices supplies the already-owned AQ atom and vanishes after
(60.3); a true-minus-AQ atom needs an independently identified residual slot
or `Delta M` provider.

For one atom, a native-P contraction uses common support `k=3`; ten atoms
with unrelated generator spans require up to `k=30` if contracted
simultaneously.  That exceeds the r49 nine-equivalent one-bank caps in
Section 60 and the standalone caps in Section 55.  The quoted `~0.537`
transport/read estimate can therefore be credited only to the forward
three-generator/plane-read implementation, not simultaneously to resident-P
product acquisition.  It must additionally include the four-channel reads
in (61.3), reset-rank growth (61.7), and physical source amplitudes.

### 61.3 Exact promotion receipts

The native-atom branch is mathematically live only if all of the following
are supplied:

1. every atom is post-AQ/post-old novel, with amplitude tied to the physical
   true-minus-AQ residual rather than a per-net scalar fit;
2. the four pair/anchor channels (61.3) are read with the stated K31/K22
   normalization and q22 collision sewing;
3. common-leg transport (61.4) is the only unpaid crossing, while every
   reset passes (61.5)--(61.7) or is bypassed by birth/read placement;
4. the target-weighted sum-of-atoms unfolding passes the Section 59 tail
   gate, including shared-generator rank across all live births; and
5. the implementation chooses forward atom reads **or** reverse native-P
   action for each physical contribution, never both.

### 61.4 Full symmetric atom is not an external pure-211 mask

Equations (61.1)--(61.8) concern the full symmetric tensor
`H=Sym(a,a,b,c)`.  If "pure q211" instead means the coordinate equality
projection

\[
H_{211}=P_{211}^{\rm ext}H,
\]

then the common-leg transport identity (61.4) no longer applies.  For a
generic dense `W`,

\[
\boxed{
[P_{211}^{\rm ext},W^{\odot4}]\ne0.}
\tag{61.9}
\]

A one-line witness starts with `e_1^{odot4}`, which is entirely in the
external `(4)` collision sector.  Its dense image `(We_1)^{odot4}` has
nonzero `(211)` and `(1111)` coordinates for a generic dense column.  Hence

\[
P_{211}^{\rm ext}W^{\odot4}P_4^{\rm ext}e_1^{\odot4}\ne0,
\qquad
W^{\odot4}P_{211}^{\rm ext}P_4^{\rm ext}e_1^{\odot4}=0.
\]

The same caveat applies to a latent-coordinate Hermite `(211)` projection
under a generic latent rotation: the full H4 space is invariant, an
individual equality partition is not.

Therefore transporting `a,b,c` computes the dense image of the **full**
atom.  To transport a masked external q211 source, one must also carry or
exactly subtract the transported `(4)`, `(31)`, `(22)`, and `(1111)` faces.
K31/K22 plane identities can establish some collision compensators at a
read seam, but they do not by themselves establish the q1111 complement or
intertwine those compensators through every reset.  The exact lawful receipt
is a suffix operator identity

\[
O_{\rm end}\Phi P_{211}^{\rm ext}H
=O_{\rm read}(a,b,c,\text{all face compensators})
\tag{61.10}
\]

on the complete source family.  Without (61.10), the claimed pure-q211
transport is a partition conflation even if every terminal plane formula is
algebraically correct for the full atom.

### 61.5 Full Waring rank six, but conditional quotient rank two

There is a useful exact signed-ray identity in the three-generator alphabet

\[
A=a^Tx,\qquad B=b^Tx,\qquad C=c^Tx.
\]

Namely,

\[
\boxed{
A^2BC\equiv
{(A+B+C)^4+(A-B-C)^4\over24}
\pmod{\mathcal L_{4,31,22}},}
\tag{61.11}
\]

where the complete leakage is

\[
\begin{aligned}
\mathcal L_{4,31,22}={1\over24}\big(&2A^4+12A^2B^2+12A^2C^2
+2B^4+8B^3C\\
&+12B^2C^2+8BC^3+2C^4\big).
\end{aligned}
\tag{61.12}
\]

Thus the **full** tensor `A^2BC` has symmetric Waring rank six, while its
quotient by the complete generator-level `(4),(31),(22)` leakage has exact
signed-ray rank two.  Rank one is impossible: one ray's three oriented 211
coordinates are tied as `alpha beta gamma (alpha,beta,gamma)` and cannot
isolate the AABC orientation.

This does not contradict the three-generator grouped carrier.  They are
three different representations:

1. one Chow group carries the full nonlinear atom `A^2BC` with generators
   `A,B,C`;
2. six CP rays reproduce that full tensor without any quotient; and
3. two signed rays reproduce only its quotient after the **entire** leakage
   (61.12) is owned.

After dense expansion into external labels, the generator collisions in
(61.12) themselves contain external `(211)` and `(1111)` coordinates.  Old
label-collision marginals do not automatically own them.  Therefore the
two-ray identity is lawful only when literal old providers own the full
dense generator tensors in (61.12), or when their terminal adjoint actions
are exactly supplied.

This gives the sharp sufficiency condition for the `3 K31 + 4 K22` reads:
their span must equal the terminal adjoint restricted to every leakage term
in (61.12), including Wick/covariance-pairing completion, and either the
induced external q1111 action is zero or a synchronized old CP4 provider
owns it.  If the seven reads are only label-collision marginals, they do not
establish the quotient and the two-ray compression is invalid.

### Graph split

- `P2R0G19B_MATH_Q211_NATIVE_AABC_RANK` — **solved exact rank audit**: support 3, pair/anchor rank 4, full Waring rank 6, conditional old-quotient signed-ray rank 2, (61.1)--(61.3) and (61.11)--(61.12).
- `P2R0G19B_MATH_Q211_NATIVE_AABC_COMMON_LEG` — **solved transport identity**: (61.4).
- `P2R0G19B_MATH_Q211_NATIVE_AABC_RESET_CLOSURE` — **open decisive receipt**: (61.5)--(61.7), static plane reads are insufficient.
- `P2R0G19B_MATH_Q211_NATIVE_AABC_SOURCE_OWNERSHIP` — **open physical amplitude gate**: only true-minus-AQ/post-old atoms count.
- `P2R0G19B_MATH_Q211_NATIVE_AABC_DOUBLE_COUNT` — **binding branch rule**: forward atom transport/read versus reverse native-P contraction.
- `P2R0G19B_MATH_Q211_AABC_PARTITION_INTERTWINER` — **open typing/transport receipt**: distinguish full, latent-211, and external-211 objects; certify (61.9)--(61.10) with all faces.
- `P2R0G19B_MATH_Q211_AABC_TWO_RAY_QUOTIENT` — **solved conditional identity / open complete leakage ownership**: (61.11)--(61.12); seven terminal reads suffice only under the full generator-leakage/q1111 adjoint equality.

## 62. Exact Schur-module expansion of a common pair frame

Section 57 proves the rank explosion for one rank-one endpoint costate.  The
corresponding theorem for an arbitrary proposed common frame is stronger and
gives the sharp structural receipt for a low-rank native-`P` branch.

Let the reverse covariance reset at one synchronized cut be

\[
\Lambda^- = D\circ\Lambda^+,
\tag{62.1}
\]

where `D` is the ReLU gate-overlap matrix.  In the centered Gaussian case it
is (57.1); with arbitrary frozen thresholds it is still the Gram matrix of
the gate indicators.  Write a thin Gram factorization

\[
D=VV^T=\sum_{s=1}^m v_sv_s^T,
\qquad m=\operatorname{rank}D,
\tag{62.2}
\]

and suppose the future pair family has a common frame

\[
\Lambda^+=U\Theta U^T,
\qquad S=\operatorname{range}U.
\tag{62.3}
\]

Define its Schur module under the reset feature space by

\[
\mathcal S_D(S)
=\operatorname{span}\{v\circ u:v\in\operatorname{range}V,
u\in S\}.
\tag{62.4}
\]

If

\[
\widetilde U
=\big[\operatorname{diag}(v_1)U\ \cdots\
       \operatorname{diag}(v_m)U\big],
\]

then the Schur-product identity gives

\[
\boxed{
D\circ(U\Theta U^T)
=\widetilde U(I_m\otimes\Theta)\widetilde U^T.}
\tag{62.5}
\]

Consequently the smallest common column space containing (62.5) for every
symmetric `Theta` is exactly `mathcal S_D(S)`.  It is already exact for one
positive-definite `Theta`: in that case the middle matrix in (62.5) is
positive definite and

\[
\operatorname{range}\big(D\circ(U\Theta U^T)\big)
=\operatorname{range}\widetilde U
=\mathcal S_D(S).
\tag{62.6}
\]

The same conclusion holds for a rank-one positive endpoint ray
`Theta=cc^T` after replacing `S` by the one-dimensional span of `Uc`.
Thus this is not merely a worst-case statement over freely retuned
coefficients.

### 62.1 Sharp full-rank obstruction and the only quotient escape

If `D` is nonsingular and `S` contains any vector `u` with no zero
coordinate, then

\[
\operatorname{range}V=\mathbb R^N,
\qquad
\{v\circ u:v\in\mathbb R^N\}=\mathbb R^N,
\]

so

\[
\boxed{\dim\mathcal S_D(S)=N.}
\tag{62.7}
\]

The generic positive-definiteness theorem of Section 57 therefore implies
that a single ordinary ReLU pair reset expands a generic full-support
endpoint ray to the entire width.  A strict native-`P` frame of width
`k<=5` (standalone dense), `k<=17` (standalone common diagonal), or any of
the r49 rows in Section 60 cannot be justified by scalar endpoint
conditioning.

If the pair state also crosses an invertible dense width map `K`, its reverse
update is

\[
\Lambda^-=K^T\big[D\circ(U\Theta U^T)\big]K,
\]

and the minimal common frame is `K^T mathcal S_D(S)`; invertible transport
does not reduce its dimension.  Across several cuts the exact unquotiented
frame recursion is therefore

\[
S_\ell
=K_\ell^T\mathcal S_{D_\ell}(S_{\ell+1}).
\tag{62.8}
\]

The only lawful escape is to apply the **complete** old/future-observation
quotient before measuring dimension.  Let `mathscr F_l` be the family of
all matrices in (62.5) demanded by the frozen future costates, and let
`Q_l` be the complete-old residual observation at the same cut.  A proposed
`k`-frame `S_l^hat` is exact if and only if

\[
\boxed{
Q_\ell(\mathscr F_\ell)
\subseteq Q_\ell\operatorname{Sym}(S_\ell^{\rm hat})
\quad\Longleftrightarrow\quad
\mathscr F_\ell
\subseteq\operatorname{Sym}(S_\ell^{\rm hat})+\ker Q_\ell,}
\tag{62.9}
\]

equivalently every member has a representative in
`Sym(S_l^hat)` modulo `ker Q_l`.  For an approximate branch, (62.9) is
replaced by the target-whitened response tail (59.3)--(59.6).  No local
spectrum of `D`, no rank of the covariance itself, and no per-net scalar fit
can substitute for that quotient-Hankel observable.

### 62.2 Exact structured low-rank receipt

Before quotienting, a common frame survives the reset with dimension at most
`k_-` if and only if

\[
\boxed{
\dim\operatorname{span}
\{v\circ u:v\in\operatorname{range}D, u\in S_+\}
\le k_-.}
\tag{62.10}
\]

This exposes the only robust algebraic cases: a genuinely low-rank gate
kernel, a coordinate/block Schur algebra preserved by every live gate, or a
future frame supported on a common zero pattern.  A different fitted frame
at each net or layer does not satisfy the recurrence (62.8).  After complete
old residualization the exact replacement for (62.10) is (62.9), and the
approximate replacement is the whitened singular-tail gate of Section 59.

### 62.3 Exact target-ray tail under bounded dynamic range

For one positive rank-one future query `uu^T`, (62.1) becomes

\[
A=\operatorname{diag}(u)D\operatorname{diag}(u).
\tag{62.11}
\]

Let `S={i:u_i\ne0}`, and define

\[
m_u=\min_{i\in S}|u_i|,
\qquad M_u=\max_{i\in S}|u_i|,
\qquad \kappa_u=M_u/m_u.
\]

Because `D_S` is a positive-semidefinite gate Gram, the min--max theorem for
the congruence in (62.11) gives, in decreasing order,

\[
m_u^2\lambda_j(D_S)
\le\lambda_j(A_S)
\le M_u^2\lambda_j(D_S).
\tag{62.12}
\]

It follows that the exact rank and relative best-rank-`k` energy tail obey

\[
\boxed{
\operatorname{rank}A=\operatorname{rank}D_S,
\qquad
\tau_k(A)
\ge\kappa_u^{-4}\tau_k(D_S).}
\tag{62.13}
\]

Thus a full-support endpoint ray with bounded coordinate dynamic range
inherits the broad gate-kernel spectrum quantitatively; scalar conditioning
is not a compression mechanism.  Before applying the old quotient, the only
escapes are a small support, a highly spiked query, or an actually low-rank
restricted gate Gram.  Equation (62.13) can be combined with the tight-frame
arcsine bound in Section 58.

The qualifier “before applying the old quotient” is binding.  A complete
future observation can annihilate precisely the broad directions appearing
in (62.13).  Therefore this is an analytic fail-fast lower bound, while the
post-old whitened operator (59.1) remains the final decision object.

### Graph split

- `P2R0G19B_MATH_Q211_SCHUR_MODULE_IDENTITY` — **solved exact theorem**: (62.2)--(62.6), the minimal common frame after one pair reset.
- `P2R0G19B_MATH_Q211_SCHUR_MODULE_FULL_RANK` — **solved generic no-go**: (62.7)--(62.8), scalar endpoints generically expand to width `N`.
- `P2R0G19B_MATH_Q211_SCHUR_MODULE_QUOTIENT` — **solved iff formulation / open frozen receipt**: complete-old representative condition (62.9).
- `P2R0G19B_MATH_Q211_STRICT_FRAME_RECURRENCE` — **solved structural iff / open candidate check**: (62.10) at every cut, or the target-weighted approximate tail.
- `P2R0G19B_MATH_Q211_TARGET_RAY_TAIL` — **solved pre-old analytic bound**: bounded-dynamic-range endpoint rays inherit at least `kappa_u^-4` times the restricted gate-kernel tail, (62.11)--(62.13).

## 63. AQ tangent absorption: quadratic q211 residuals are refit directions first

The r49 replacement is a parameterized model, not a frozen list of unrelated
features.  Let

\[
\mathcal F_{AQ}:\Theta_{49}\longrightarrow\mathcal Y
\tag{63.1}
\]

denote its complete deployed response, including the connected-H4 surface
**only if that surface is explicitly billed**.  The response space
`mathcal Y` includes every downstream covariance/Mehler and endpoint output
which changes when the AQ state changes.  At a fitted state `theta_0`, let

\[
J_0=D\mathcal F_{AQ}(\theta_0).
\tag{63.2}
\]

Any lawful matrix perturbation `Delta M` induced by a parameter tangent
`delta theta` has q211 action

\[
\delta y_{211}=P_{211}J_0\delta\theta.
\tag{63.3}
\]

It is therefore an old-model refit direction, not an additive carrier.  The
first-order residual eligible for a new feature is obtained from the joint
old-plus-refit design.  If `O_complete` denotes the complete frozen retained
interface, put `A_0=[O_complete,J_0]`; then

\[
\boxed{
r_{\rm normal}
=P_{\operatorname{range}A_0}^{\perp}r_{211}^{\rm omit}.}
\tag{63.4}
\]

The joint projector is essential: sequential orthogonal projectors need not
commute and generally do not equal residualization against the concatenated
design.

The projector in (63.4) must use the **joint** refit Jacobian, including all
low-order, covariance-rest, H4-surface, and downstream changes.  Projecting
only against the frozen q211 scalar or the old collision planes can falsely
promote a direction which the r49 model could already absorb for zero
incremental deployed cost.

### 63.1 Exact finite secant, and what equation (60.6) means

Suppose the full true quadratic completion is itself an admissible r49 state
`theta_1`, with matrices `M^T`, while `theta_0` has matrices `M^A`.  For a
fixed q211 query, define its surface action `S(M)` by (55.1).  Then (60.6) is
exactly

\[
\boxed{S(M^T)-S(M^A),}
\tag{63.5}
\]

the finite secant of the same AQ H4 surface.  If that surface is already
part of the replacement, the lawful deployment is to replace `theta_0` by
`theta_1` and evaluate the existing surface once.  No ordered-`P` additive
bank is needed, and firing both the refit and (60.6) double counts.

For a complete end-to-end claim the query/costate in (63.5) is not held
fixed: covariance propagation, reset Jacobians, and all downstream costates
must be recomputed at `theta_1`.  The exact nonlinear ownership object is
the secant set

\[
\operatorname{Sec}_{\theta_0}(\mathcal F_{AQ})
=\{\mathcal F_{AQ}(\theta)-\mathcal F_{AQ}(\theta_0):
\theta\in\Theta_{49}\}.
\tag{63.6}
\]

A finite residual is additive-novel only after quotienting by (63.6), or
after an actual joint refit has reached its optimum.  Tangent projection
(63.4) is the local fail-fast test, not a proof about a large finite step.

### 63.2 Exact limits of the absorption claim

The zero-incremental-cost conclusion requires all of the following:

1. `Delta M` is produced by one shared admissible r49 parameter change,
   respecting its latent gauge, cross-unit coupling, and frozen-net law;
2. the connected-H4 reset surface is already in the deployed replacement;
3. the final package jointly recomputes the lower moments, Gaussian rest,
   and future Bellman action rather than adding a frozen correction; and
4. the physical residual does not require extra latent directions or a
   higher-chaos slot outside `Theta_49`.

If condition 2 fails, native-`P` may be the implementation of the missing AQ
surface (60.0), so it is not absorbed by a low-chaos refit.  If condition 1
or 4 fails, only the projection of the residual onto the admissible tangent
is old; the normal component remains a legitimate carrier target.  In that
case the Section 59 target-weighted unfolding must be formed **after**
(63.4), which can be substantially thinner than the raw native q211 but is
not identified by the local pair kernel.

Thus the correct order is:

1. joint-refit the billed r49 model or project onto its complete Jacobian;
2. form the normal true-minus-refit connected-H4 response;
3. only then apply the native-`P` common-frame/tail and nine-equivalent
   screens.

### 63.3 State-manifold tangent versus deployable-refit tangent at r49

The abstract r49 AQ coefficient manifold has

\[
\Theta_{\rm state}
=\left(\mathbb R^{49}\oplus\operatorname{Sym}_{49}\right)^{1024}
/O(49)
\]

and therefore

\[
\boxed{
\dim\Theta_{\rm state}
=1024(49+1225)-{49\cdot48\over2}
=1,303,400.}
\tag{63.7}
\]

Within a fixed lawful latent gauge, an arbitrary bank
`(Delta u_i,Delta M_i)` is a tangent of this **state** manifold.  Hence a
native-`P` proposal whose only new object is a same-shape r49 matrix bank is
mathematically a state refit, not evidence for an additional representational
degree of freedom.

The deployed estimator may expose a smaller manifold.  If a shared
initialization/recurrence with parameters `psi` produces the state through
`theta=s(psi;weights)`, its actual refit Jacobian is

\[
J_{\rm deploy}=J_{\rm state}\,D_\psi s,
\qquad
\operatorname{range}J_{\rm deploy}
\subseteq\operatorname{range}J_{\rm state}.
\tag{63.8}
\]

Thus an arbitrary per-net `Delta M` is not deployably old unless one shared,
label-free update law produces it.  This is exactly the unresolved joint
fiber/section gate; state capacity cannot be borrowed as though each test
net supplied free fitted matrices.

There is also a rigorous normal-dimension warning in the other direction.
The synchronized `(C,K21)` target space has dimension `1,573,376`, so even
the full abstract r49 state Jacobian has generic codimension at least

\[
\boxed{1,573,376-1,303,400=269,976.}
\tag{63.9}
\]

A deployable section can only increase that normal dimension.  Therefore a
generic truth cannot be wholly absorbed by r49 refitting.  But (63.9) does
not prove that the particular complete-old q211 residual has a component in
the normal space, nor that the endpoint Bellman costate sees it.  The exact
promotion scalar remains

\[
\left\langle
P_{\operatorname{range}[O_{\rm complete},J_{\rm deploy}]}^\perp r_{211},
P_{\operatorname{range}[O_{\rm complete},J_{\rm deploy}]}^\perp\lambda
\right\rangle,
\tag{63.10}
\]

with the corresponding panel Gram for a corpus claim.  This cleanly
separates a large algebraic normal space from actual raw-MSE value.

### 63.4 Universal r49 hook obstruction at the affine seam

The generic dimension warning can be sharpened to a representation-theoretic
first-order no-go.  Let

\[
L=\mathbb R^{49},\qquad
E=\operatorname{Sym}^2L,\qquad d=1225,
\]

and let `u_i in L` be the affine feature rows.  A shared index-linear matrix
tangent has

\[
H_i=H(u_i),
\qquad
T(x,y,z)=x^TH(y)z=T(z,y,x).
\]

Its affine K21 differential is

\[
\boxed{
(F_UH)_{ij}
=2T(u_i,u_j,u_i)+4T(u_i,u_i,u_j)
=6(\operatorname{Sym}T)(u_i,u_i,u_j).}
\tag{63.11}
\]

Hence the entire map factors as

\[
L\otimes\operatorname{Sym}^2L
\longrightarrow\operatorname{Sym}^3L
\longrightarrow\mathbb R^{n\times n}.
\tag{63.12}
\]

The invisible `[2,1]` hook therefore has dimension

\[
\boxed{
49\cdot1225-\binom{51}{3}
=60,025-20,825=39,200.}
\tag{63.13}
\]

Now put `K=L^perp` inside the width space, so `m=dim K=975`.  If the base
affine K21 interface `F_U` is not surjective, a first-order local section is
already impossible.  Give the refit its strongest case and assume
`rank F_U=n^2=1,048,576`.  The hook factorization bounds the available
prefix-normal quotient by

\[
\dim{K\otimes E\over\mathcal B}
\le975\cdot1225-(1,048,576-20,825)
=166,624,
\tag{63.14}
\]

where `mathcal B=range(F_U^*) cap (K tensor E)`.  But

\[
\dim\operatorname{Sym}K={975\cdot976\over2}=475,800.
\]

Thus for **every** quadratic prefix `N`, the first-order normal Schur map
satisfies

\[
\boxed{
\operatorname{rank}\mathscr S_{U,N}\le166,624,
\qquad
\operatorname{cork}\mathscr S_{U,N}\ge309,176.}
\tag{63.15}
\]

Near a centered affine seam with quadratic prefix size `t`, covariance is
even in `t` while K21 is linear plus cubic.  After eliminating the
well-conditioned covariance/K21 image blocks and the persistent orthogonal
gauge, the exact reduced normal series is

\[
t\mathscr S_1+t^3\mathscr S_3+O(t^5).
\tag{63.15a}
\]

Thus hook-cokernel directions missed by `mathscr S_1` have singular values
only `O(t^3)`, and any eventual local right inverse is at least
`Omega(t^-3)`-conditioned along them.  This parity statement assumes the
base image blocks are surjective and well-conditioned; otherwise restrict to
their actual surjective image and retain the additional first-order normal.

Equation (63.15) decisively prevents the claim that one shared r49 prefix
tangent generically absorbs the full true quadratic normal.  It still does
**not** prove q211 value: the physical residual must occupy these hook-normal
directions and the endpoint costate in (63.10) must see them.  It is a source
normality theorem, not a raw-MSE certificate.

Under the same explicitly flat response-space diagnostic used for the r49
capacity estimate, however, (63.15) materially tightens the arithmetic prior.
The first-order missing fraction is at least

\[
e_{\rm hook}
={309,176\over1,573,376}
\simeq0.196505,
\tag{63.16}
\]

not the parameter-count value `0.171590`.  Against the ideal-6e-9 allowance
`e_*=0.204019`, the remaining normalized budget becomes only

\[
e_*-e_{\rm hook}\simeq0.007514,
\qquad
\rho_{\rm required}\simeq\sqrt{1-0.007514}=0.99624.
\tag{63.17}
\]

Thus the hook obstruction removes about seventy-seven percent of the
advertised r49 error margin at an affine seam.  This is still a conditional
flat/isotropic diagnostic, not a task-weighted lower bound, and a finite
quadratic prefix can open cubic-order directions.  It does show that the
earlier `0.98365` tolerance cannot be used for a near-affine first-order
section.

### 63.5 The hook threshold moves the first feasible AQ rank to r55

> **Superseded by Section 68.**  Equations (63.18)--(63.20) are a valid
> hook-only dimension screen, but they omit the universal Veronese-null
> kernel.  The positive `2,979` does not permit an r55 first-order section;
> every r55 prefix still has a kernel of dimension at least `102,831`.

The hook calculation has a sharp closed-form rank threshold.  For general
latent rank `r` at width `n=1024`, put

\[
d_r={r(r+1)\over2},\qquad
m_r=n-r,\qquad
s_r=\binom{r+2}{3}.
\]

Under the strongest surjective affine interface, the hook-normal quotient
capacity minus the required symmetric normal dimension is

\[
\boxed{
f(r)=m_rd_r-(n^2-s_r)-{m_r(m_r+1)\over2}.}
\tag{63.18}
\]

A first-order local section is universally impossible when `f(r)<0`.
Pascal's identity gives the exact monotonic increment

\[
\boxed{f(r+1)-f(r)=(n-r)(r+2)>0.}
\tag{63.19}
\]

At the decisive adjacent ranks,

\[
\boxed{f(54)=-51,341,
\qquad f(55)=+2,979.}
\tag{63.20}
\]

Therefore r55 is the **first** rank not killed by the universal affine hook
obstruction.  Full-AQ r54, despite exceeding the naive parameter-dimension
count by 1,129, still cannot possess a first-order affine local section;
r49 fails much more strongly by (63.15).  The positive `2,979` at r55 is
only room for surjectivity, not a proof that a lawful well-conditioned
section exists.

This theorem changes the architecture prior.  A near-affine r49 replacement
must rely on cubic-order opening plus an exceptional target-dark normal,
whereas an r55 branch can in principle be first-order regular.  The separate
billing census then decides whether r55 can run only on a late suffix; rank
feasibility does not waive its cost.

### 63.6 An exact incidence criterion for proving the r55 section exists

> **Closed by Section 68.**  The incidence criterion is formally sufficient,
> but its premise can never hold at r55: the structured space
> `W_N` always contains a `102,831`-dimensional Veronese-null subspace.  The
> proposed Walsh minor therefore cannot be nonzero at the stated size.

The positive margin in (63.20) is not yet an existence proof, but it admits
a sharp algebraic route.  At r55 let

\[
m=969,\qquad d=1540,\qquad
\mathcal V=K\otimes E,
\qquad
\mathcal B=\operatorname{range}(F_U^*)\cap\mathcal V.
\]

For a symmetric normal query \(Z\in\operatorname{Sym}K\), define

\[
L_Z:\mathcal V\to\mathcal V,
\qquad L_Z(N)=ZN,
\]

and the number of independent quotient constraints

\[
c_{\mathcal B}(Z)
=\operatorname{rank}\big(P_{\mathcal B^\perp}L_Z\big).
\tag{63.21}
\]

The affine variety of symmetric \(m\times m\) matrices of rank at most \(t\) has
dimension

\[
d_t={t(2m-t+1)\over2}.
\tag{63.22}
\]

For fixed nonzero \(Z\), the bad prefixes satisfying \(ZN\in\mathcal B\) form a
linear set of codimension \(c_{\mathcal B}(Z)\).  Therefore the incidence
dimension theorem gives the exact sufficient condition

\[
\boxed{
\inf_{\operatorname{rank}Z=t}c_{\mathcal B}(Z)>d_t
\quad\text{for every }t=1,\ldots,m.}
\tag{63.23}
\]

Under (63.23), the union of all bad incidences cannot dominate prefix space,
so a Zariski-open set of prefixes `N` makes

\[
Z\longmapsto[ZN]\in\mathcal V/\mathcal B
\]

injective.  This is precisely the missing first-order r55 Schur-section
receipt.

There is an even cleaner fixed-prefix formulation.  Let

\[
\mathcal W_N
=\{ZN:Z\in\operatorname{Sym}K\}\subseteq\mathcal V.
\tag{63.24}
\]

If \(Z\mapsto ZN\) is injective before quotienting, then

\[
\boxed{
\text{the r55 section is injective}
\quad\Longleftrightarrow\quad
\mathcal B(U)\cap\mathcal W_N=\{0\}.}
\tag{63.25}
\]

At the maximum hook quotient rank,

\[
\dim\mathcal V=1,492,260,\qquad
\dim\mathcal B=1,019,316,\qquad
\dim\mathcal W_N=469,965,
\]

so the Schubert dimension margin is exactly

\[
\boxed{
\dim\mathcal V-\dim\mathcal B-\dim\mathcal W_N=2,979.}
\tag{63.26}
\]

For a fixed \(Z\), an unconstrained generic codimension-472,944 subspace
would give \(c_{\mathcal B}(Z)=\min(1540\,\operatorname{rank}Z,472,944)\).
It is **not** lawful to assert this uniformly over every varying row space
from genericity alone; the determinantal incidence can have excess
intersections.  The exact remaining theorem is (63.25), equivalently one
nonzero 469,965-by-469,965 minor of the structured gate map.

A concrete analytic construction route is available.  Take the width labels
as the group \((\mathbb Z_2)^{10}\), choose 55 character columns for \(U\),
and Fourier-transform the matrix tangent.  Then

\[
H_g=\sum_k\chi_k(g)\widehat H_k
\]

turns the two terms of \(F_U\) into sparse triple-frequency incidences
\((s\oplus t,k)\) and \((k\oplus s,t)\).  Selecting the 55-frequency set and
a coordinate injection \(J:K\to\operatorname{Sym}^2S\) reduces (63.25) to a
finite combinatorial matching/minor problem.  No such matching has yet been
proved, so this is a sharply specified construction route, not a receipt.

### Graph split

- `P2R0G19B_MATH_AQ_Q211_REFIT_TANGENT` — **solved ownership theorem / open realized Jacobian**: (63.1)--(63.4), admissible AQ tangents are old-model directions.
- `P2R0G19B_MATH_AQ_Q211_FINITE_SECANT` — **solved exact branch rule / open joint refit**: (60.6) is the frozen-query secant (63.5); deploy the replacement state, not an additive duplicate.
- `P2R0G19B_MATH_AQ_Q211_NORMAL_RESIDUAL` — **open highest-value response**: complete true-minus-best-refit normal component after (63.6), then Section 59 tail/cost gates.
- `P2R0G19B_MATH_AQ_Q211_ABSORPTION_LIMITS` — **binding scope**: surface billing, admissible shared parameter law, self-consistent downstream recomputation, and no extra latent/higher-chaos directions.
- `P2R0G19B_MATH_AQ_R49_STATE_VS_DEPLOY_TANGENT` — **solved dimension/inclusion theorem / open section**: (63.7)--(63.9), arbitrary state perturbations are not free per-net deployable refits.
- `P2R0G19B_MATH_AQ_R49_NORMAL_VALUE` — **open target-weighted gate**: the 269,976-dimensional generic normal does not imply q211 or endpoint value; certify (63.10).
- `P2R0G19B_MATH_AQ_R49_HOOK_NORMAL_NOGO` — **solved universal first-order no-go**: (63.11)--(63.15), every prefix leaves at least 309,176 normal directions under the strongest surjective affine interface.
- `P2R0G19B_MATH_AQ_R49_HOOK_VALUE` — **open physical overlap gate**: hook-normal source capacity does not imply the true q211 residual or endpoint costate occupies it.
- `P2R0G19B_MATH_AQ_R49_HOOK_FLAT_CAPACITY` — **solved conditional diagnostic**: flat first-order tail at least `0.196505`, leaving only `0.007514` ideal-target margin and requiring about `0.99624` retention, (63.16)--(63.17).
- `P2R0G19B_MATH_AQ_HOOK_R55_THRESHOLD` — **superseded hook-only screen**: (63.18)--(63.20) omit the Section 68 Veronese-null kernel; r55 is closed at first order.
- `P2R0G19B_MATH_AQ_R55_SECTION_INCIDENCE` — **closed no-go**: the formally sufficient incidence gate (63.23) cannot hold because every prefix has the Section 68 null subspace.
- `P2R0G19B_MATH_AQ_R55_WALSH_MINOR` — **closed at stated size**: no character-frequency choice can remove the universal Veronese-null kernel; only a different quotient/interface or higher-order opening can reopen it.

## 64. Deterministic second-order bound for the multi-residual q211 block

When the nonlinear AQ H4 surface and the one-residual tilted generator are
both owned, Section 60 shows that the first omitted q211 term has at least
two true-minus-AQ residual slots.  This gives a genuine quadratic smallness
theorem, but only in a norm strong enough to control fourth moments.

For centered scalar random variables `Z_1,...,Z_4`,

\[
\kappa_4(Z_1,Z_2,Z_3,Z_4)
=E\prod_{a=1}^4Z_a
-\sum_{\text{three pairings}}E[Z_iZ_j]E[Z_kZ_l].
\]

Holder and `||Z||_2<=||Z||_4` give the universal bound

\[
\boxed{
|\kappa_4(Z_1,Z_2,Z_3,Z_4)|
\le4\prod_{a=1}^4\|Z_a\|_4.}
\tag{64.1}
\]

At one synchronized cut suppose

\[
\max_i\|Q_i\|_4\le q,
\qquad
\max_i\|E_i\|_4\le e.
\tag{64.2}
\]

Applying (64.1) to the `4/6/4/1` slot expansion gives

\[
\boxed{
\begin{aligned}
|\Delta^{(1)}|&\le16eq^3,\\
|\Delta^{(\ge2)}|
&\le4\left(6e^2q^2+4e^3q+e^4\right)
\end{aligned}}
\tag{64.3}
\]

for each unit coefficient of the polarized q211 tensor.  If `E` is the
orthogonal `H>=3` residual while `Q` lies in `H<=2`, all `E--Q` covariance
pairings vanish, but (64.3) remains a safe bound without relying on that
extra cancellation.  Under that exact global-chaos orthogonality the sharper
count is

\[
\boxed{
|\Delta^{(1)}|\le4eq^3,
\qquad
|\Delta^{(\ge2)}|
\le12e^2q^2+4e^3q+4e^4.}
\tag{64.3a}
\]

This refinement must not be applied to a generic true-minus-AQ residual,
which need not be orthogonal to the nonlinear AQ output.

Let `lambda_{l,nu}` be the actual complete-old/refit-normal q211 costate for
layer `l` and frozen net `nu`, in the same coefficient normalization as
(60.2), and let

\[
w_{l,\nu}=\|P_{211}^*\lambda_{l,\nu}\|_1.
\tag{64.4}
\]

Then the coherent omitted endpoint amplitude on that net is bounded by

\[
B_\nu
=\sum_l4w_{l,\nu}
\left(6e_{l,\nu}^2q_{l,\nu}^2
+4e_{l,\nu}^3q_{l,\nu}+e_{l,\nu}^4\right).
\tag{64.5}
\]

For corpus weights `p_nu`, the exact sufficient adversarial gate is

\[
\boxed{
\left(\sum_\nu p_\nu B_\nu^2\right)^{1/2}
\le3.80\times10^{-6}.}
\tag{64.6}
\]

If the one-residual surface is absent, add
`sum_l 16 w_lnu e_lnu q_lnu^3` inside `B_nu`.  An endpoint projective tensor
norm can replace the entrywise `l1` norm in (64.4) for a sharper certified
decomposition; the inequality is otherwise unchanged.

For ten equal, coherently aligned layers with `q=1` and `w=1`, the small-`e`
form of (64.6) requires roughly

\[
e\le\sqrt{3.80\times10^{-7}/24}
\simeq1.26\times10^{-4}.
\tag{64.7}
\]

This illustrates that “second order” is not by itself enough under the
strict endpoint budget.

For a certified Wiener polynomial residual of maximum chaos degree `d`,
Nelson hypercontractivity supplies the lawful conversion

\[
\boxed{\|E\|_4\le3^{d/2}\|E\|_2.}
\tag{64.8}
\]

Under the equal-layer illustration (64.7), this requires approximately
`||E||_2<=1.40e-5` for `d=4`, and `||E||_2<=1.55e-6` for `d=8`.
An untruncated ReLU residual has infinite chaos degree, so (64.8) is usable
only together with a certified finite-degree split and an analytic/Fock
bound on the tail.

Most importantly, no universal version of (64.6) follows from `L2` residual
energy alone: variables with fixed arbitrarily small `L2` norm can have
arbitrarily large `L4/L2` ratio.  Explicitly, for an event of probability
`p`, the centered variable `E=c(1_A-p)` with
`c=e/sqrt(p(1-p))` has `||E||_2=e` but `||E||_4/e` grows like
`p^(-1/4)`.  The minimal lawful receipt is therefore an
actual residual fourth-moment/hypercontractive envelope in the synchronized
AQ gauge, together with the complete normal costate weights (64.4).  Existing
covariance-rest energies and endpoint-cancelled eta summaries do not supply
either side of this bound.

### Graph split

- `P2R0G19B_MATH_AQ_Q211_MULTIRESIDUAL_L4_BOUND` — **solved deterministic bound / open moments**: (64.1)--(64.6), the post-surface/post-one-residual omission is quadratic in residual `L4` size.
- `P2R0G19B_MATH_AQ_Q211_SECOND_ORDER_BUDGET` — **solved stringent scale gate**: ten equal unit-weight layers require about `e<=1.26e-4`, (64.7).
- `P2R0G19B_MATH_AQ_Q211_L2_INSUFFICIENCY` — **closed information no-go**: covariance/L2 residual energy alone cannot certify connected q211; require fourth-moment envelope plus normal costates.
- `P2R0G19B_MATH_AQ_Q211_HYPERCONTRACTIVE_GATE` — **solved conditional conversion / open tail**: finite chaos degree gives (64.8); full ReLU still needs the contracted analytic/Fock remainder certificate.

## 65. Exact H211 surface cap for the six-reset r55 suffix

The hook threshold makes r55 the first locally feasible AQ rank, while the
paid-Gram census permits only a six-reset late suffix.  The two facts meet at
an exact surface-cost gate.

Take the D14-through-L8 / AQ55-plus-D10 L9--14 schedule with `K=32`
quadrature nodes.  Its strict paid-Gram subtotal is `48.197`, leaving
`0.803`; the exact terminal-total covariance switch lowers the subtotal to
`46.640`, leaving `2.360`, provided no future consumer needs the separated
final rest.

At a common H211 slice rank `k`, reuse the already-paid per-unit AQ
eigensystems, abscissae, characteristic values, and resolvents.  The exact
optimistic contraction count specialized from the H211 Fourier action is

\[
\boxed{
C_{211}^{\rm dense}(k)
={6\over1024^2}
\left(87k^2+4785k+1760\right),}
\tag{65.1}
\]

where `87=55+32`, `4785=55(55+32)`, and `1760=55*32`.
If every `Theta_i` is diagonal in one lawful common factor frame, the count
reduces to

\[
\boxed{
C_{211}^{\rm diag}(k)
={6\over1024^2}\left(4872k+1760\right),}
\tag{65.2}
\]

because `4872=(55+1)(55+32)`.

The resulting exact integer screens are

\[
\begin{array}{c|cc}
\text{available slack}&\text{dense common slice}&\text{common diagonal}\\
\hline
0.803\ \text{(strict paid Gram)}&k\le20&k\le28\\
2.360\ \text{(terminal total)}&k\le46&k\le84.
\end{array}
\tag{65.3}
\]

The latent factor rank need not exceed 55, so the diagonal arithmetic can
cover a full latent frame in the terminal-total branch:

\[
C_{211}^{\rm diag}(55)\simeq1.543,
\]

leaving about `0.817` for every collision subtraction, seam hook,
stabilization, and memory effect.  A generic dense full-rank slice does not
fit:

\[
C_{211}^{\rm dense}(55)\simeq3.012>2.360.
\tag{65.4}
\]

Nor does the arithmetic cap by itself imply adequate approximation.  A flat
55-direction query truncated at the dense cap k=46 discards

\[
{55-46\over55}={9\over55}\simeq16.36\%
\]

of its energy, far above the 3.28064-percent target-tail gate.  The dense
route therefore needs a strongly top-heavy **post-old/refit-normal endpoint**
spectrum, not merely k<=46 storage.

Therefore the six-reset hybrid has only three mathematical surface routes:

1. prove the complete-old/refit-normal H211 query has dense common rank at
   most 46, with enough lower rank to leave realistic hook margin;
2. prove simultaneous common-frame diagonal structure across every suffix
   reset; or
3. exhibit a literal operation-by-operation CSE which absorbs (65.1) into
   the already-paid K32 local reset.

For real symmetric slices the exact diagonal receipt in route 2 is

\[
\boxed{
[\Theta_{\ell i},\Theta_{\ell j}]=0
\quad\text{for every }i,j,\ell,}
\tag{65.5}
\]

together with a reverse recurrence which transports the same joint
eigenframe between cuts.  Separate per-anchor diagonalizations do not earn
(65.2).  For an approximate frame, the complete endpoint-weighted
off-diagonal remainder must be included in the Section 59 tail and coherent
error gates.

Sharing nodes or eigensystems alone is already credited in (65.1) and is not
the third receipt.  Equations (65.1)--(65.4) also exclude acquisition of the
true-minus-AQ multi-residual source, reverse Bellman construction, and
non-unit implementation constants.  They are necessary arithmetic gates,
not a complete deployment census.

### Graph split

- `P2R0G19B_MATH_AQ55_LATE_SUFFIX_H211_COST` — **solved exact conditional census**: six-reset surface costs (65.1)--(65.3), strict caps `k<=20/28`, terminal caps `k<=46/84`.
- `P2R0G19B_MATH_AQ55_DENSE_H211_NOGO` — **closed for generic full dense rank**: dense `k=55` costs about `3.012>2.360`, (65.4).
- `P2R0G19B_MATH_AQ55_DIAG_H211_ROUTE` — **open structural/census receipt**: common-diagonal `k=55` costs about `1.543`, leaving only `0.817`; prove pairwise commutation and recurrent joint eigenframe (65.5).
- `P2R0G19B_MATH_AQ55_K32_SURFACE_CSE` — **open literal fusion receipt**: node/eigensystem reuse is already charged; any further saving requires identical operations and buffers.

## 66. Exact incremental connected-H4 surface census for full-AQ r49

The r49 replacement has almost nine equivalents of nominal headroom, but the
connected-H4 surface is not free merely because its AQ eigensystems and reset
nodes are resident.  There is an exact contracted-action census.

Let

- `q` be the number of distinct latent directions exposed to the surface;
- `h` be the number of required symmetric tilted-covariance direction pairs;
- `a` be the number of final H211 atom contractions;
- `K` be the number of shared quadrature nodes; and
- `D` be the number of active resets.

With literal same-node/eigensystem reuse, the optimistic fused cost is

\[
\boxed{
C_{H4}^{\rm fused}
={D\over n^2}\left[q r^2+Kr(q+h)+Ka\right].}
\tag{66.1}
\]

This computes a contracted surface action, not a resident full H4 plane.  It
is lawful only if the complete future query factors through exactly these
directions at the synchronized seam.

At `r=49,K=32,D=14,n=1024`,

\[
\boxed{
C_{H4}^{\rm fused}
={14\over1,048,576}
\left(3969q+1568h+32a\right).}
\tag{66.2}
\]

For unrelated AABC atoms, `q=3a` and `h=4a`, so

\[
C_{H4}^{\rm fused}\simeq0.24314a.
\tag{66.3}
\]

The `8.98` nominal headroom admits at most `a=36`; `a=37` already costs
about `8.996` before collision sewing, hooks, source acquisition, memory, or
error control.  A dense symmetric Hessian with
`h=q(q+1)/2` admits at most `q=26` under the fused atom-assembly convention;
`q=27` fails before extras.  Replacing `D=14` by a ten-reset suffix scales
these arithmetic counts by `10/14` but does not change their ownership
requirements.

### 66.1 One-residual tilted generator cost

The one-residual oracle (60.4b) has two sharply different costs.  A new
latent matrix action per jet family costs

\[
C_{1R}^{\rm unfused}={Dq_{jet}r^3\over n^2},
\]

while literal fusion into already-paid matrix-vector/node actions has the
optimistic floor

\[
\boxed{
C_{1R}^{\rm fused}\ge{Dq_{jet}r^2\over n^2}.}
\tag{66.4}
\]

At r49 the per-family unfused/fused figures are

\[
\begin{array}{c|cc}
&\text{unfused}&\text{fused floor}\\
\hline
D=14&1.57079&0.032057\\
D=10&1.12199&0.02290.
\end{array}
\tag{66.5}
\]

Even the fused floor at generic width-many jet rank is about `32.8` for D14
or `23.4` for D10, so the one-residual interface also needs a strict
target-conditioned direction rank.  In addition, the physical residual
source `E_i` must be deployably identified; a query without its source is
not an oracle.

Native ordered products do not substitute for (66.1): on the carried
quadratic projection they compute `kappa4(X)`, whereas (66.1) is the billed
nonlinear AQ reset surface.  Conversely, a surface scalar or symmetric
gradient does not imply resident full ordered `P`, by (60.7a).  The two
interfaces have separate ownership and billing gates.

### Graph split

- `P2R0G19B_MATH_AQ49_CONNECTED_H4_INCREMENTAL_BILL` — **solved exact conditional census**: fused contracted surface formula (66.1)--(66.3), at most 36 unrelated atoms or dense direction rank 26 before extras.
- `P2R0G19B_MATH_AQ49_ONE_RESIDUAL_JET_COST` — **solved rank-cost theorem / open source**: (66.4)--(66.5), generic width-rank tilted generator is closed even at the fused floor.
- `P2R0G19B_MATH_AQ49_SURFACE_VS_NATIVE_P` — **binding ownership split**: nonlinear AQ surface, low-chaos native products, and true-residual jets are distinct actions.

## 67. A native three-generator q211 atom is not a reverse carrier

The static Chow/group representation from Section 61 is cheap at one
synchronized cut.  It does **not** by itself give a cheap carrier through a
coordinatewise correlated reset.  This section gives the exact reverse
recurrence, a one-reset rank witness in the q211 sector itself, and the only
target-weighted escape from that witness.

Let

\[
\mathcal H_4=\operatorname{Sym}^4(\mathbb R^r)
\]

and let the fourth-order state at cut `l` be

\[
B_l=(B_{l1},\ldots,B_{ln})\in\mathbb R^n\otimes\mathcal H_4.
\]

Write `W_l` for the dense neuron transport and
`K_{li}:\mathcal H_4\to\mathcal H_4` for the realized fourth-order reset
operator of output coordinate `i`.  Up to the harmless choice of whether a
cut is named before or after `W_l`, the exact forward and reverse equations
are

\[
\begin{split}
B_{l+1,i}
  &=K_{li}\!\left(\sum_j W_{l,ij}B_{lj}\right)+S_{l+1,i},\\
Q^-_{l,i}&=K_{li}^{*}Q_{l+1,i},\\
Q_{l,j}&=\sum_iW_{l,ij}Q^-_{l,i},\\
\delta J_{l+1}^{\rm birth}&=\sum_i\langle Q_{l+1,i},S_{l+1,i}\rangle.
\end{split}
\tag{67.1}
\]

Thus one endpoint-conditioned reverse tensor is sufficient at cut `l` iff
the anchor flattening of `Q_l` has rank one.  More generally, the exact
number of common H4 reverse streams is

\[
\boxed{
k_l^{*}=\operatorname{rank}\operatorname{Flat}_{i\mid H_4}
       \bigl(P_{\mathcal B_l}P_{{\rm old},l}^{\perp}Q_l\bigr),}
\tag{67.2}
\]

where `P_{B_l}` restricts the query to the lawful birth subspace.  Equation
(67.2) is both necessary and sufficient: an SVD supplies `k_l^*` common
tensors, and any representation by `k` common tensors has anchor-flattening
rank at most `k`.  Dense `W_l` only mixes anchor coefficients and therefore
does not increase this Schmidt rank.  The coordinate-dependent family
`K_{li}^{*}` is the possible rank explosion.

### 67.1 A fixed-coordinate H211 rank-n witness after one reset

The explosion occurs even if the incoming object is one grouped atom.  Split
the latent space as

\[
\mathbb R^r=S\oplus\operatorname{span}\{b,c\},
\qquad \dim S=r-2,
\]

choose `a,g_i` in `S`, and start from

\[
T=\operatorname{Sym}(a,a,b,c).
\]

For a biased rank-one threshold gate (for example threshold `t=1`, where
the fourth Hermite coefficient is nonzero) whose fourth Hermite kernel is

\[
C_{4,i}=\gamma_i g_i^{\otimes4},\qquad \gamma_i\ne0,
\]

the exact H4 reset formula from Section 44 contains

\[
3\operatorname{Sym}(C_{4,i}\otimes_2T).
\tag{67.3}
\]

Because `g_i` is orthogonal to `b,c`, the AABC component of (67.3) is a
nonzero convention-dependent scalar times

\[
\boxed{
\gamma_i(a^{T}g_i)^2
\operatorname{Sym}(g_i,g_i,b,c).}
\tag{67.4}
\]

All coefficient conventions only rescale (67.4); they do not change its
rank.  There is also an explicit fixed-coordinate H211 injection, so no
H1111 credit is needed for the lower bound.  Partition the latent
coordinates into `U,V`, with `|U|,|V|>=4`, and take `b,c` to be the normalized
block indicators of `U,V`.  Then

\[
S=\{x:\mathbf1_U^Tx=\mathbf1_V^Tx=0\}.
\]

For `G in Sym^2(S)`, put

\[
L_{211}(G)=P_{211}\operatorname{Sym}(G\otimes b\otimes c).
\]

This map is injective.  Indeed, write
`B=sym(b tensor c)`.  Its repeated-index entries, for distinct `i,j,k`, are
proportional to

\[
G_{ii}B_{jk}+2G_{ij}B_{ik}+2G_{ik}B_{ij}+G_{jk}B_{ii}.
\tag{67.5a}
\]

If (67.5a) vanishes, choose `i in U` and distinct `j,k in V`.  It gives
`G_ij+G_ik=0`; because `|V|>=3`, every `U`--`V` entry is zero.  Next choose
distinct `i,j in U` and `k in V`.  It gives
`G_ii+2G_ij=0`.  Since `G b=0`,

\[
G_{ii}+\sum_{j\in U\setminus\{i\}}G_{ij}
={3-|U|\over2}G_{ii}=0,
\]

so the `U` block vanishes; the `V` block follows symmetrically.  Thus
`G=0`.

The `C_4` images therefore span a space of dimension

\[
D_S={ (r-2)(r-1)\over2}.
\]

The common `pT` term and the `C_2` term occupy at most `1+(r-2)` H211
directions.  The `C_6,C_8` contractions vanish in this construction because
only the two `a` slots pair with `g_i`.  Since rank-one matrices `g_i g_i^T`
span `Sym^2(S)`, select them after quotienting that at-most-`r-1` dimensional
lower block.  This proves the constructive bound

\[
\boxed{
\operatorname{rank}\{K_i^{*}T:i\le n\}
\ge \operatorname{rank}\{P_{211}K_i^{*}T:i\le n\}
\ge \min\!\left\{n,{(r-2)(r-1)\over2}-(r-1)\right\}.}
\tag{67.5b}
\]

This is a constructive ReLU-compatible witness, not merely a generic
dimension count.  At width 1024 the fixed-H211 lower bound is 902 for
`r=45` and the full 1024 for `r=49` or `r=55`.  Projection onto a *proved*
complete-old or future-dark space can lower (67.5b); collision-plane names,
H1111 cancellation, and the seven static AABC reads do not prove that
projection.

The witness also identifies exactly what the grouped-atom estimate omits.
The three generators are preserved by a common dense leg map, but the
`C_4` reset produces one new repeated generator `g_i` per live gate.  The
three-K31/four-K22 formulas read a grouped atom at one seam; they are not an
intertwiner for (67.1).  Consequently the quoted roughly `0.537`-equivalent
ten-birth transport/read census is complete only if every intervening
nonlinear reset has already been eliminated, or if the following cyclic
receipt is supplied.

### 67.2 Exact common-frame and target-tail receipt

An exact `k`-stream carrier exists iff there are subspaces
`V_l` of `H_4`, with `dim V_l<=k`, such that

\[
P_{\mathcal B_l}P_{{\rm old},l}^{\perp}Q_l\in
\mathbb R^n\otimes V_l
\quad\text{and}\quad
P_{\mathcal B_l}P_{{\rm old},l}^{\perp}K_{li}^{*}V_{l+1}
\subseteq V_l
\quad\text{for every live }i,l.
\tag{67.6}
\]

Equivalently, `V_l` is the complete-old-residualized terminal cyclic module
generated by the realized reset adjoints.  A common-leg/OU law is one
sufficient way to prove (67.6); a static three-generator factorization is
not.

There is a sharp approximate version.  Let

\[
\sigma_{l1}\ge\sigma_{l2}\ge\cdots
\]

be the singular values of the endpoint-whitened, complete-old-residualized
flattening in (67.2), after applying the lawful source weighting.  If the
total Hilbert norm of the layer-`l` birth coefficient is at most `C_l`, the
best rank-`k_l` common-frame approximation obeys the exact coherent bound

\[
\boxed{
|e_{\rm end}|
\le
\sum_l C_l
\left(\sum_{a>k_l}\sigma_{la}^2\right)^{1/2}.}
\tag{67.7}
\]

The SVD attains the Frobenius tail in (67.7), and Cauchy--Schwarz makes the
bound sharp without an orthogonality premise.  For promotion, its right
side must be at most `3.80e-6`; the looser orthogonal `2.73e-5` budget is
available only after endpoint orthogonality is proved.

For the six-reset r55 architecture, the arithmetic receipts of Section 65
therefore have a precise interpretation.  The post-old/refit-normal cyclic
rank must fit `k<=20` (strict dense) or `k<=46` (terminal-total dense), or
the stronger recurrent common-diagonal receipt must fit `k<=28` or the
latent ceiling 55.  A raw rank of 55 or 1024 is irrelevant if the weighted
tail (67.7) is tiny, but that tail is the required evidence; one scalar
endpoint and one-net refitting do not certify it.

### 67.3 Ownership and amplitude: no AQ double credit

The reverse-rank theorem does not change the ownership split.  At an AQ
replacement seam,

\[
\kappa_4(Y_{\rm true})
=\kappa_4(X_{\rm AQ})
+\underbrace{[\kappa_4(Y_{\rm AQ})-\kappa_4(X_{\rm AQ})]}_{
  \text{billed AQ connected-H4 surface}}
+\underbrace{[\kappa_4(Y_{\rm true})-\kappa_4(Y_{\rm AQ})]}_{
  \text{true-minus-AQ residual}}.
\tag{67.8}
\]

Resident AQ ordered products are an implementation of the middle term only
when the synchronized full `48F+commutator` identity is actually evaluated.
They are not an additive feature on top of a billed AQ surface.  A new
three-generator AABC bank may own only the complete-old residual of the last
term in (67.8), and only if its physical amplitudes are supplied by a lawful
true-minus-AQ observable.

In particular, the static three-K31/four-K22 reads determine the tensor
shape of a grouped atom but do not determine its residual amplitude.  AQ
products `P_A` cannot acquire that amplitude.  For a quadratic residual
they require the explicit `Delta P` cross and square action of (60.8); for a
nonquadratic residual they require the corresponding one- or multi-residual
surface generator.  A coefficient chosen from endpoint labels is not a
deployable source.

The full-AQ replacement is therefore stronger than a standalone CP carrier
only conditionally: it subtracts the AQ connected-H4 block and shares the
surface eigensystem/nodes, leaving the narrower last term of (67.8).  It is
not a solution unless that residual source has low cyclic rank under (67.6),
passes (67.7), and fits the incremental surface/product census of Section
66.

### Graph split

- `P2R0G19B_MATH_Q211_GROUP_REVERSE_RECURRENCE` — **solved exact**: reverse recurrence (67.1) and minimal anchor-Schmidt rank (67.2).
- `P2R0G19B_MATH_Q211_GROUP_ONE_RESET_RANK_NOGO` — **solved constructive generic no-go**: a single correlated reset yields the fixed-coordinate H211 bound (67.5b), full width at r49/r55 even without H1111 credit.
- `P2R0G19B_MATH_Q211_TARGET_WEIGHTED_CYCLIC_ESCAPE` — **open sharp receipt**: prove invariant common frame (67.6) or endpoint-weighted tail (67.7) within the r55 caps.
- `P2R0G19B_MATH_Q211_NATIVE_0537_SCOPE` — **conditional census**: the grouped-atom transport/read count excludes nonlinear reset rank unless the cyclic receipt holds.
- `P2R0G19B_MATH_AQ_Q211_RESIDUAL_NO_DOUBLE_CREDIT` — **binding ownership split**: decomposition (67.8); native AQ products and a billed AQ surface are the same AQ value, while new atoms need a lawful true-minus-AQ amplitude.

## 68. Veronese-null theorem: r55 is still first-order impossible

The hook-only count in Section 63 missed an additional structured kernel.
This closes the proposed r55 first-order Schur section and moves the first
rank at which this particular obstruction can disappear to r63.

Let the width space split as

\[
P=L\oplus K,
\qquad \dim L=r,
\qquad \dim K=m=n-r,
\]

and put `E=Sym^2 L`, `d=dim E=r(r+1)/2`.  For affine rows
`u_i in L`, define the Veronese evaluation

\[
\boxed{
(Q_U G)_i=u_i^TGu_i,
\qquad Q_U:E\to\mathbb R^n.}
\tag{68.1}
\]

In the conventions of (63.11), the full affine tangent interface is

\[
\boxed{
(F_UH)_{ij}=2u_i^TH_i u_j+u_i^TH_j u_i,
\qquad F_U:P\otimes E\to\mathbb R^{n\times n}.}
\tag{68.2}
\]

If `F_U` is not onto, a complete first-order local section is already
impossible.  Give the model its strongest case and assume it is onto; then
`Q_U` is onto as well.  Let

\[
A=\operatorname{range}F_U^*,
\quad
B=A\cap(K\otimes E),
\quad
C=B^{\perp_{K\otimes E}}.
\]

The decisive exact inclusion is

\[
\boxed{C\subseteq K\otimes\ker Q_U.}
\tag{68.3}
\]

To prove it, use

\[
(A\cap W)^{\perp_W}
=\{H\in W:F_UH\in F_U(W^\perp)\},
\qquad W=K\otimes E.
\]

Write `H=sum_a k_a tensor G_a` in an orthonormal basis of `K`.  Every
output in `F_U(L tensor E)` has right factor in `L`.  Projecting the right
output leg onto `K` gives

\[
0=(F_UH)P_K=\sum_a(Q_UG_a)k_a^T.
\]

Multiplication by each `k_b` yields `Q_U G_b=0`, proving (68.3).  Notice that
the inclusion can be strict, which only strengthens the no-go.

Now represent a quadratic prefix by a matrix

\[
N: E\to K
\]

and let `N_0=N P_{ker Q_U}` be its restriction to the only columns visible
in (68.3).  A symmetric normal `Z in Sym K` can affect the quotient only
through `ZN_0`.  Therefore

\[
R=\ker N_0^T\subseteq K,
\qquad
\dim R\ge m-q,
\qquad
q=\dim\ker Q_U=d-n,
\]

has the universal property

\[
\boxed{
\operatorname{Sym}R
\subseteq\ker\{Z\mapsto[ZN]\}.}
\tag{68.4}
\]

Indeed, the columns of `N_0` lie in `R^perp`, so any symmetric `Z` supported
on `R` satisfies `ZN_0=0`.  This holds for **every** prefix `N`; no generic
minor or Walsh choice can evade it.

At `n=1024,r=55`,

\[
d=1540,
\quad m=969,
\quad q=516,
\quad \dim R\ge453,
\]

and hence

\[
\boxed{
\dim\ker\mathscr S_{55}
\ge {453\cdot454\over2}=102,831,}
\tag{68.5}
\]

so

\[
\operatorname{rank}\mathscr S_{55}
\le469,965-102,831=367,134.
\tag{68.6}
\]

The earlier positive `2,979` Schubert margin was therefore spurious as an
existence criterion: it counted an unconstrained quotient and missed
(68.3).  The proposed 469,965-dimensional Walsh minor is identically zero.

For the Veronese-null kernel merely to be allowed to vanish, one needs
`q>=m`, namely

\[
{r(r+1)\over2}-n\ge n-r
\quad\Longleftrightarrow\quad
\boxed{{r(r+3)\over2}\ge2n.}
\tag{68.7}
\]

At width 1024, r62 fails and r63 is the first integer satisfying (68.7).
This is only a necessary threshold: r63 still needs the full hook,
conditioning, billing, ownership, and endpoint-value receipts.  For r55,
the centered parity theorem (63.15a) means any recovery of the null block
can first occur at cubic order, with an `Omega(t^{-3})` inverse on a
near-affine prefix unless an additional interface changes the quotient.

This result does not close the six-reset r55 architecture as a predictor.
It closes only the claim that a shared r55 AQ refit **generically absorbs**
the complete quadratic q211 residual at first order.  A target-dark normal,
a directly billed residual surface, or a controlled finite cubic secant may
still work; each remains subject to the Section 59/67 endpoint-weighted
tail and `3.80e-6` error gate.

More generally, define

\[
\boxed{
t_r=\left(2n-{r(r+3)\over2}\right)_+.}
\tag{68.8}
\]

Then (68.4) gives the prefix-independent bound

\[
\boxed{
\dim\ker\mathscr S_r\ge {t_r(t_r+1)\over2}.}
\tag{68.9}
\]

At width 1024 the frontier is

\[
\begin{array}{c|ccccc}
r&55&60&61&62&63\\ \hline
t_r&453&158&96&33&0\\
\binom{t_r+1}{2}&102831&12561&4656&561&0.
\end{array}
\tag{68.10}
\]

This makes r62 the sharp near-cap cubic compromise: only 561 directions are
universally first-order dark, versus 102,831 at r55.  The independent
terminal-L5 census costs about `47.43295`, leaving `1.56705`, but promotion
still requires the induced cubic normal map to cover those 561 directions
with usable conditioning, a lawful D14-to-AQ initialization, and enough
five-layer endpoint value.  Dimension 561 is a tractable target theorem,
not evidence that the cubic map is nonzero.

The exact cubic gate is available symbolically.  At a prefix
`(u_i,M_i)=(u_i,tN_i)`, order rows as `(C,K21)` and columns as affine and
quadratic tangents `(h,H)`.  Then

\[
J(t)=
\begin{pmatrix}
D&tL_N\\ tG_N&F_U+t^2Q_N
\end{pmatrix},
\tag{68.11}
\]

where

\[
\begin{aligned}
(L_NH)_{ij}&=2\operatorname{tr}(H_iN_j+N_iH_j),\\
(G_Nh)_{ij}&=4h_i^TN_ju_i+4h_i^TN_iu_j+4u_i^TN_ih_j,\\
(Q_NH)_{ij}&=8\operatorname{tr}
\{(N_iH_i+H_iN_i)N_j+N_i^2H_j\}.
\end{aligned}
\tag{68.12}
\]

Choose a covariance-tangent inverse `P`, a right inverse `R_F` of `F_U`,
the tangent projector `P_T`, and covariance-normal projector `Pi`.  The
canonical reduced maps are

\[
\boxed{
\begin{aligned}
\mathscr S_1&=\Pi L_N|_{\ker F_U},\\
\mathscr S_3&=-\Pi L_NR_F
 (Q_N-G_NPP_TL_N)|_{\ker F_U},\\
\overline{\mathscr S}_3&:
\ker\mathscr S_1\longrightarrow\operatorname{coker}\mathscr S_1.
\end{aligned}}
\tag{68.13}
\]

Changing the auxiliary inverses changes `S3` only by the range of `S1`, so
the barred map is intrinsic.  For r62 the exact missing receipt is a
nonzero 561-by-561 minor of (68.13), followed by a lower singular-value
bound at the realized prefix.  This is substantially sharper than asking
whether an unspecified finite AQ refit might work.

### Graph split

- `P2R0G19B_MATH_AQ_VERONESE_NULL_NOGO` — **solved exact**: quotient dual inclusion (68.3) and prefix-independent kernel (68.4).
- `P2R0G19B_MATH_AQ_R55_FIRST_ORDER_SECTION_NOGO` — **closed**: every r55 prefix has at least 102,831 first-order null directions, (68.5)--(68.6).
- `P2R0G19B_MATH_AQ_FIRST_ORDER_R63_THRESHOLD` — **solved necessary threshold / section open**: (68.7), r63 is the first width-1024 rank not killed by this kernel.
- `P2R0G19B_MATH_AQ_R55_CUBIC_OR_DIRECT_ESCAPE` — **open target-weighted branch**: only cubic secant opening, a changed interface, or a directly billed complete-old-residual surface can bypass the first-order no-go.
- `P2R0G19B_MATH_AQ_R62_CUBIC_FRONTIER` — **open high-priority theorem**: (68.8)--(68.10), terminal-L5 r62 leaves only 561 universal null directions and about 1.567 equivalents; prove cubic surjectivity, conditioning, initialization, and value.

## 69. The native H211 cost is controlled by reverse slice rank

There is an exact pair-Gram bridge between the AQ core and a synchronized
H211 query, but its cost is governed by the rank of the **reverse query
slice**, not by the Chow rank of a forward AABC atom.

For affine-quadratic core coordinates `(u_i,A_i)`, define the pair feature

\[
\boxed{
\Psi_{ij}
=\bigl(A_i\circ A_j,\ A_i u_j+A_j u_i\bigr).}
\tag{69.1}
\]

With the corresponding product inner product, the fourth cumulant is the
three-pair Gram sum.  In particular,

\[
\boxed{
\kappa_4(i,i,j,k)
=\langle\Psi_{ii},\Psi_{jk}\rangle
+2\langle\Psi_{ij},\Psi_{ik}\rangle.}
\tag{69.2}
\]

Suppose the synchronized reverse H211 slice has a common external factor

\[
\Lambda_i=U\Theta_iU^T,
\qquad \Theta_i\in\operatorname{Sym}^k.
\]

Then its exact AQ action is

\[
\boxed{
\sum_{i,a,b}(\Theta_i)_{ab}
\left[
\langle\Psi_{ii},\Psi_{ab}^{U}\rangle
+2\langle\Psi_{ia}^{U},\Psi_{ib}^{U}\rangle
\right],}
\tag{69.3}
\]

where a superscript `U` means the corresponding linear combination over
the external index.  Equation (69.3) is the tensor-free native action.  An
eigendecomposition of each `Theta_i` reduces it to

\[
s=\max_i\operatorname{rank}\Theta_i
\tag{69.4}
\]

pair modes.  Thus the optimistic low-mode cost uses `s`, whereas one
three-generator forward atom, one D10 scalar word, or one terminal endpoint
does not imply `s=1`.

### 69.1 Exact recurrence of the slice rank

Let a realized reset pair kernel have a Gram factorization

\[
D_i=V_iV_i^T=\sum_{h=1}^{t_i}v_{ih}v_{ih}^T.
\]

The Schur action on a query slice is exactly

\[
\boxed{
D_i\circ(U\Theta_iU^T)
=\sum_h
(\operatorname{diag}v_{ih}U)\Theta_i
(\operatorname{diag}v_{ih}U)^T.}
\tag{69.5}
\]

If the same common frame is invariant,

\[
\operatorname{diag}(v_{ih})U=UR_{ih},
\tag{69.6}
\]

then the reduced recurrence is

\[
\boxed{
\Theta_i^+=\sum_hR_{ih}\Theta_iR_{ih}^T.}
\tag{69.7}
\]

For an incoming rank-one slice `Theta_i=theta_i ww^T`, the rank of (69.7)
is generically

\[
\dim\operatorname{span}\{R_{ih}w:h\le t_i\}.
\tag{69.8}
\]

It stays rank one iff all nonzero `R_ih w` are collinear.  If (69.6) fails,
the correct common support is the span of every
`diag(v_ih)U`; Section 62 shows that a full-rank ReLU pair kernel and a
full-support factor generically expand it to full width.

Dense anchor transport imposes a second, independent gate.  Linear
combinations of arbitrary rank-`s` symmetric matrices are not generally
rank `s`.  A sufficient and, for a linear family of symmetric slices,
necessary common-support form is

\[
\boxed{
\Theta_i=E H_iE^T
\quad\text{for one common }E\in\mathbb R^{k\times s}
\quad\text{at every live anchor}.}
\tag{69.9}
\]

For `s=1`, this says `Theta_i=theta_i ww^T` with the **same** `w` for every
anchor.  Per-anchor rank-one eigendirections do not survive a generic dense
mix.  Across depth, (69.6)--(69.9) must hold recursively, after the complete
old and future-dark quotient, and every `R_ih E` must land in the next
common support.

### 69.2 Disposition of the current D10/Q24 claim

The present D10 schedule specifies an abstract target-conditioned word, not
the slice recurrence (69.7).  Q24/collision reads specify synchronized
marginals, not a common matrix ray (69.9).  Therefore there is presently no
receipt for `s=1` or another small fixed `s`.  Under the generic
coordinate-dependent ReLU reset, (69.8) and the Schur expansion (69.5)
instead drive `s` to the full retained frame rank.

Accordingly, the low-mode native census (about `2.19` equivalents in the
current r55/k21/D10 comparison) is conditional.  Without an explicit
common-support recurrence, the full-slice charge (about `46` in that same
comparison) is the lawful baseline and does not fit as an unnoticed add-on.
The only escapes are:

1. prove (69.6)--(69.9) for the realized D10/Q24 reverse tape;
2. prove that the complete-old/refit-normal endpoint projection leaves a
   small target-weighted slice rank and passes (67.7); or
3. fuse the exact scalar (69.3) into an already-paid surface at the same
   seam, with literal operand and buffer identity.

There is also a sharp value--cost squeeze.  If the post-old slice spectrum
is flat on `k` directions, a rank-`s` action discards fraction `1-s/k` of
its energy.  The 3.28064-percent tail gate therefore requires

\[
\boxed{s\ge\left\lceil0.9671936\,k\right\rceil.}
\tag{69.10}
\]

For `k=21`, (69.10) requires all 21 modes; for `k=55`, it requires at least
54.  A rank-one cost claim is compatible with the value gate only if the
actual complete-old endpoint spectrum is extremely concentrated (or is
exactly rank one), not merely because a rank-one algebraic factorization
can be written before the reset.

This is also the double-count boundary.  If (69.3) is evaluated on the AQ
core by the billed connected-H4 surface, a native pair-Gram evaluation of
the same `Lambda_i` is a CSE, not a second correction.  Only the lawful
true-minus-AQ residual slice may add value.

### Graph split

- `P2R0G19B_MATH_H211_AQ_PAIR_GRAM_BRIDGE` — **solved exact**: tensor-free action (69.1)--(69.3).
- `P2R0G19B_MATH_H211_REVERSE_SLICE_RANK` — **solved exact recurrence**: Schur/reset laws (69.5)--(69.8); forward atom rank does not set reverse slice rank.
- `P2R0G19B_MATH_H211_COMMON_SLICE_SUPPORT` — **open structural receipt**: prove common invariant support (69.6),(69.9) through dense mixing and all suffix resets.
- `P2R0G19B_MATH_H211_D10_LOW_MODE_COST` — **conditional / generic no-go**: low-mode cost is unavailable from the abstract D10/Q24 word alone; charge the full slice absent the common-support or target-tail receipt.
- `P2R0G19B_MATH_H211_SLICE_VALUE_COST_SQUEEZE` — **solved conditional theorem**: a flat k-mode slice needs (69.10), so k21 needs all 21 modes and k55 at least 54.

## 70. Trace-collapse of a q211 costate: exact locally, not closed in depth

A special q211 source has a much cheaper scalar action than a generic
three-generator bank.  Define the strict repeated-anchor insertion

\[
(ZR)_{i;jk}=R_{jk},
\qquad i,j,k\ \text{distinct}.
\]

For an arbitrary synchronized q211 Bellman costate `Lambda`,

\[
\boxed{
\langle\Lambda,ZR\rangle
=\langle\Gamma,R\rangle,
\qquad
\Gamma_{jk}=\sum_{i\ne j,k}\Lambda_{i;jk}=Z^*\Lambda.}
\tag{70.1}
\]

Thus if `Gamma` is already a resident pair plane, a full strict-q211 trace
source costs only one `n^2` dot.  CP/Waring rank is irrelevant for this
specific action.  The issue is ownership and recurrence of `Gamma`.

### 70.1 Exact dense-leg intertwiner

Before applying strict collision masks, introduce the polarized insertion

\[
Z_G(R)=\operatorname{Sym}(G\otimes R).
\]

For any dense leg map `W`, direct adjunction gives

\[
\boxed{
Z_G^*(W^T)^{\otimes4}\Lambda
=W^T Z_{WGW^T}^*(\Lambda)W.}
\tag{70.2}
\]

Indeed,

\[
W^{\otimes4}Z_G(R)
=Z_{WGW^T}(WRW^T),
\]

and pairing with `Lambda` proves (70.2).  Therefore the uniform trace
metric `G=I` becomes `WW^T` after one generic dense layer.  A coordinatewise
reset first replaces it by gate-dependent diagonal weights, which a later
dense layer again turns into a full metric.  Applying the strict 211 mask
does not restore the intertwiner; it adds collision/1111 boundary terms
which must be synchronized separately.

Consequently a single unweighted `Gamma=Z_I^*Lambda` is not a recursively
closed reverse state.  Exact depth closure requires one of:

1. scaled-orthogonal dense legs and a reset which preserves scalar `G`;
2. a resident low-dimensional family of weighted metrics `G_l`, closed
   under every reset and `G mapsto WGW^T`; or
3. a full Bellman tape which has already compiled the weighted contraction
   in (70.2) into a paid pair-state adjoint.

### 70.2 When is Gamma literally the Q22 costate?

Let `R_l` be an actual resident pair state and let its forward q211 birth
Jacobian at the synchronized seam be `J_l^{tr}`.  The q211 trace contribution
is resident in the Q22 adjoint exactly when

\[
\boxed{J_l^{tr}=Z_l}
\tag{70.3}
\]

with the same normalization, collision mask, pre/post-reset cut, and
transport convention, and when the reverse tape includes

\[
(J_l^{tr})^*\Lambda_l=Z_l^*\Lambda_l.
\tag{70.4}
\]

A Q22 plane bearing the same indices is not an ownership proof.  In
particular, a direct K22 collision costate and an abstract D10 q4perp word
need not include (70.4).  If the resident Q22 plane is a **total** adjoint
which also contains direct collision and other births, then either the
physical source must be contracted with that total plane or the other
components must be exactly subtracted before claiming (70.1).  Otherwise
the proposal double counts them.

The current D10 schedule does not specify the literal Jacobian/tape identity
(70.3)--(70.4).  Therefore the trace collapse is a high-value seam receipt,
not yet a free implementation.  If the receipt is proved, it bypasses the
generic slice-rank cost for precisely the Price-trace source.  It does not
acquire the remaining true-minus-AQ q211 normal or make arbitrary AABC
atoms cheap.

### 70.3 Universal reuse moves the feature into the pair-supply lane

There is a useful no-double-count theorem.  Let `F` be the complete future
map from the H211 cut to endpoint space and `O` the resident pair-state
consumer.  Universal reuse of the trace adjoint means that for some fixed
source conversion `C`,

\[
\boxed{
Z^*F^*=C^*O^*
\quad\Longleftrightarrow\quad
FZ=OC.}
\tag{70.5}
\]

Thus a universally free trace-H211 action is not independent H4 physics:
its endpoint response factors through the resident pair consumer.  If
`CR` is already supplied by the old state, adding `ZR` is a literal
duplicate.  If `R` is a new truth-free observable, it may still be valuable,
but its novelty and acquisition cost belong to the **pair-supply** lane;
one should inject `CR` directly rather than bill an H211 carrier.

For a single terminal covector `lambda`, the weaker condition is only

\[
Z^*F^*\lambda\in\operatorname{range}O^*.
\tag{70.6}
\]

That can provide a target-conditioned scalar shortcut without the universal
factorization (70.5), but it is not a provider or corpus receipt.  It must
be frozen and pass the complete panel/future-query rank test.  This
separates a genuinely useful new pair source from an H4 double credit.

### Graph split

- `P2R0G19B_MATH_Q211_TRACE_COLLAPSE` — **solved exact static identity**: (70.1), one pair dot if the trace adjoint is resident.
- `P2R0G19B_MATH_Q211_TRACE_DENSE_INTERTWINER` — **solved structural no-go**: (70.2), the unweighted trace is not closed under generic dense/reset transport.
- `P2R0G19B_MATH_Q211_TRACE_Q22_REUSE` — **open literal seam receipt**: prove (70.3)--(70.4) on the actual D10/full-Q22 tape with every other contribution separated.
- `P2R0G19B_MATH_Q211_TRACE_RESIDUAL_SCOPE` — **binding scope**: a proved collapse owns only the Price-trace source, not arbitrary true-minus-AQ H211.
- `P2R0G19B_MATH_Q211_TRACE_PAIR_SUPPLY_RECLASS` — **solved ownership theorem**: universal reuse is equivalent to (70.5); novelty is either zero or a new pair-supply observable, not an independent H4 carrier.

## 71. Concrete Price-trace escape: direct covariance matvecs

The abstract trace collapse of Section 70 has one physically important
specialization which avoids forming `Gamma`.  At zero bias, let `R` be the
resident standardized preactivation correlation and put

\[
c={\pi-4\over8\pi}.
\]

Define the pairing and identity--covariance trace words by

\[
(\mathcal P_R)_{abcd}
=R_{ab}R_{cd}+R_{ac}R_{bd}+R_{ad}R_{bc},
\]

\[
(\mathcal Z_R)_{abcd}
=\delta_{ab}R_{cd}+\delta_{ac}R_{bd}+\delta_{ad}R_{bc}
 +\delta_{bc}R_{ad}+\delta_{bd}R_{ac}+\delta_{cd}R_{ab}.
\]

The exact second Price jet is

\[
\boxed{
\Pi_{211}\Delta^{P,(2)}
={c\over2}\Pi_{211}(\mathcal P_R-\mathcal Z_R).}
\tag{71.1}
\]

The carried covariance-pairing state owns the first word.  At a coordinate
birth, Star4 and coordinate CP4 have zero strict-211 slice and Q24 has no
typed map `R mapsto Z_R`; inherited CP4 columns must be projected at later
cuts.  Therefore the complete-old zero-bias residual is exactly

\[
\boxed{
P_{old}^{\perp}\Delta^{P,(2)}
=-{c\over2}P_{old}^{\perp}\mathcal Z_R.}
\tag{71.2}
\]

This is source-information old--`R` is resident--but its typed readout may
be new.  It is also part of the full AQ connected-H4 surface: if that
surface is already billed in the comparator, (71.2) has zero additive
credit.

### 71.1 Exact CP-costate action

Let `Rbar=R-Diag(diag R)`.  For one symmetric CP costate atom

\[
L_i[j,k]=\alpha_i\operatorname{Sym}_{jk}(\beta_j\gamma_k),
\]

strict collision exclusion gives

\[
\boxed{
\begin{aligned}
\langle L,\Pi_{211}\mathcal Z_R\rangle
={}&(\mathbf1^T\alpha)\,\beta^T\bar R\gamma\\
&-(\alpha\odot\beta)^T\bar R\gamma
 -(\alpha\odot\gamma)^T\bar R\beta.
\end{aligned}}
\tag{71.3}
\]

Thus a generic atom needs the two covariance matvecs `Rbar beta` and
`Rbar gamma`; when `beta=gamma`, it needs one.  No H211 tensor, per-anchor
slice eigendecomposition, or recurrent trace plane is required.  For
Let `R_{c,l}` be the costate-atom count at seam `l`.  The exact leading
census is

\[
\boxed{
C_Z^{generic}={2\over n}\sum_lR_{c,l},
\qquad
C_Z^{sym}={1\over n}\sum_lR_{c,l}.}
\tag{71.4}
\]

For a constant `R_c` over `D` seams this is `2DR_c/n` or `DR_c/n`.  At
width 1024 and `D=10`, one equivalent permits `R_c<=51` generically or
`R_c<=102` for symmetric atoms.  In the r55 **six-reset** billing frame, the
strict `0.803` slack permits constant `R_c<=68/137`, and the terminal-total
`2.360` slack permits `R_c<=201/402`, before hooks and memory.

### 71.2 Specialization to a grouped AABC query

One CP ray `u^4` has strict-H211 slice

\[
L_i[j,k]=u_i^2u_ju_k,
\]

so it is a symmetric atom in (71.3), with
`alpha=u odot u` and `beta=gamma=u`.  Therefore:

\[
\boxed{
f_R(u):={1\over6}\langle u^{\otimes4},\Pi_{211}\mathcal Z_R\rangle
=\|u\|^2u^T\bar Ru-2(u^{\odot3})^T\bar Ru.}
\tag{71.5a}
\]

If the full dense collision leakage is old-owned, the quotient identity of
Section 61 gives the explicit grouped action

\[
\boxed{
\langle A^2BC,\Pi_{211}\mathcal Z_R\rangle_{quot}
={f_R(a+b+c)+f_R(a-b-c)\over4}.}
\tag{71.5b}
\]

Here `f_R` is one sixth of the literal diagonal polynomial:
`(Pi_211 Z_R)[u^4]=6f_R(u)`.  The quartic polarization denominator is `24`,
so the resulting tensor action has denominator `24/6=4`.  A denominator
`24` would apply only to the normalized tensor `(Pi_211 Z_R)/6` or if
`f_R` denoted the full diagonal polynomial.

Thus the two-ray route supplies both the tensor shape and its Price-trace
amplitude directly from resident `R`; no fitted scalar is hidden in the
formula.  Equations (71.5a)--(71.5b) are quotient identities: the full
unquotiented action must restore or separately own the collision polynomial
listed in Section 61.

Therefore:

- a grouped AABC atom costs two symmetric matvec actions if its exact
  quotient-rank-two representation is lawful; and
- it costs six if the full Waring-rank-six representation is required.

The two-ray figure is lawful only when the complete dense 4/31/22 leakage
and induced 1111 action are synchronized old-owned, exactly as in Section
61.  Otherwise the six-ray count is the safe one.

For illustration, if one grouped atom is born at each of ten cuts and every
atom remains live to the endpoint, there are `55` atom-seam incidences.
Equation (71.4) then costs

\[
{110\over1024}\simeq0.1074
\quad\text{(lawful two-ray quotient)},
\qquad
{330\over1024}\simeq0.3223
\quad\text{(safe six-ray form)}.
\tag{71.5}
\]

These are genuine sub-equivalent scalar-action counts.  They do **not**
waive the Section 67 reset/cyclic receipt: (71.5) assumes the stated ray
costates are available at every seam.  Generic correlated reset can expand
their reverse Schmidt rank before the trace action is taken.

In particular, the three choices of repeated anchor form a three-dimensional
orientation fiber.  The rigorous depth propagator is the moving
three-by-three reset block of the rooted-Price lane, not the scalar CP rank.
If all three orientations must be carried with no batch reuse, the two
figures in (71.5) become

\[
{330\over1024}\simeq0.3223,
\qquad
{990\over1024}\simeq0.9668.
\tag{71.6}
\]

Thus a fully oriented safe six-ray carrier already misses the strict r55
`0.803` slack, although it fits the terminal-total branch; the lawful
two-ray quotient still fits.  If reset changes a self-paired ray into a
generic `(beta,gamma)` query, the two-matvec action doubles these upper
counts.  Exact reuse must replace these worst-case multipliers, not an
assumed scalar orientation.

The true rank-one separable reopen therefore requires all three independent
receipts: the complete-old quotient source is one-dimensional in the moving
symmetric orientation, every seam has zero symmetric/contrast coupling, and
the endpoint Gram separates the selected birth shells.  Otherwise the
three-state transport error, broad triangle floor, and joint target Gram
remain binding even though (71.3) is locally cheap.

### 71.3 Direct rooted-star replacement

If the old pairing action cannot be isolated safely, the entire zero-bias
strict Price jet can be evaluated without subtracting two large endpoint
features.  Since (71.1) gives

\[
\Delta^{P,(2)}_{i;jk}=cR_{ij}R_{ik}
\qquad(i,j,k\ \text{distinct}),
\]

put

\[
p=R\beta,
\quad q=R\gamma,
\quad h=\beta\odot\gamma,
\quad t=(R\circ R)h.
\]

Exact exclusion of `j=i`, `k=i`, and `j=k` yields

\[
\boxed{
\begin{aligned}
&\sum_{i,j,k\ distinct}
\alpha_i\beta_j\gamma_kR_{ij}R_{ik}\\
&\qquad=
\alpha^T\{(p-\beta)\odot(q-\gamma)-t+\beta\odot\gamma\}.
\end{aligned}}
\tag{71.7}
\]

Multiplication by `c` gives the full rooted Price action.  It needs three
matrix-vector actions for a generic atom (`R beta`, `R gamma`, and
`(R circ R)(beta odot gamma)`) and two when `beta=gamma`.  The incremental
trace correction (71.3) is cheaper only because the old pairing action is
credited.  Equation (71.7) is the safer replacement when that ownership or
its numerical subtraction is uncertain; the two implementations must not
be stacked.

This is the first explicit target-conditioned escape from the generic
H211 reset-rank theorem which has both an exact source and a sub-GEMM scalar
consumer.  Its scope is narrow:

1. use the complete-old-projected costate rank `R_c`, not the nominal CP or
   two-ray birth rank;
2. subtract inherited CP4 and every old typed row at the same cut;
3. do not add it on top of a billed full-AQ H4 surface; and
4. do not credit it toward the true-minus-AQ multi-residual block.

For biased heterogeneous units, the residual also contains anchor-dependent
root amplitudes and an open-path Jordan word; the latter generally needs a
third matvec.  Equation (71.3) is the exact zero-bias trace branch, not a
blanket biased-reset closure.

### Graph split

- `P2R0G19B_MATH_AFFINE_Q211_PRICE_ZERO_BIAS_TRACE` — **solved exact**: pairing-minus-trace identity (71.1)--(71.2), with literal typed-old projection.
- `P2R0G19B_MATH_AFFINE_Q211_PRICE_TRACE_CP_ACTION` — **solved exact target-conditioned escape**: direct action (71.3) and cost (71.4), no H211 carrier.
- `P2R0G19B_MATH_AFFINE_Q211_PRICE_TRACE_R55_CAP` — **solved conditional census**: r55 D6 strict caps `68/137`, terminal caps `201/402`, before extras.
- `P2R0G19B_MATH_AFFINE_Q211_PRICE_AABC_ACTION` — **solved local census / cyclic receipt open**: (71.5), two rays only with full leakage ownership, otherwise six; reset availability remains Section 67.
- `P2R0G19B_MATH_AFFINE_Q211_PRICE_ORIENTATION_COST` — **solved conditional census / transport open**: full three-orientation costs (71.6); exact small carrier requires zero coupling and the rooted-Price three-state receipt.
- `P2R0G19B_MATH_AFFINE_Q211_PRICE_ROOTED_DIRECT` — **solved exact replacement**: strict rooted action (71.7), three matvecs generic/two symmetric; use instead of, not in addition to, pairing-minus-trace.

## 72. Forward plane closure of the Price trace word

The Price trace source also has an exact one-affine-step **forward** closure
into the collision planes consumed by D11.  This can avoid a future H211
costate if the required two pair matrices are already resident.

For symmetric pair matrices `G,S`, use the six-term normalization

\[
\begin{aligned}
\mathcal Z_{G,S;abcd}={}&
G_{ab}S_{cd}+G_{ac}S_{bd}+G_{ad}S_{bc}\\
&+G_{bc}S_{ad}+G_{bd}S_{ac}+G_{cd}S_{ab}.
\end{aligned}
\tag{72.1}
\]

Let `g=diag G` and `s=diag S`.  Direct substitution gives every visible
collision plane:

\[
\boxed{
\begin{aligned}
K_4[p]&=6g_ps_p,\\
K_{31}[p,q]&=3\{g_pS_{pq}+s_pG_{pq}\},\\
K_{22}[p,q]&=g_ps_q+s_pg_q+4G_{pq}S_{pq}.
\end{aligned}}
\tag{72.2}
\]

Equivalently,

\[
K_{31}=3\{D_gS+D_sG\},
\qquad
K_{22}=gs^T+sg^T+4(G\circ S).
\tag{72.3}
\]

If a source trace word is `Z_{I,R}` and the next affine leg is `W`, then

\[
\boxed{
W^{\otimes4}\mathcal Z_{I,R}
=\mathcal Z_{WW^T,WRW^T}.}
\tag{72.4}
\]

Thus (72.2)--(72.4) give exact next-cut K4/K31/K22 using only outer,
diagonal, and Hadamard operations once

\[
G'=WW^T,
\qquad S'=WRW^T
\tag{72.5}
\]

are resident.  This is the forward counterpart of the adjoint identity
(70.2).

### 72.1 Strict-211 collision sewing

The Price source is the strict projection `P211 Z_{I,R}`, not the full word.
For a standardized correlation with unit diagonal, the source collision
planes of the full word are

\[
K_4[p]=6,
\qquad K_{31}[p,q]=3R_{pq},
\qquad K_{22}[p,q]=2
\quad(p\ne q),
\tag{72.6}
\]

and its 1111 sector is zero.  Therefore the transported strict source is
obtained exactly by (72.2) minus the affine transports of (72.6).  The
subtraction is free only when the synchronized CP4/Star4/pair collision
lineages literally carry those source tensors.  Family names or matching
planes before `W` are not enough; the affine leg must be the same.

If this receipt holds, (72.2) supplies a local one-step plane injection to
the already-paid variance tangent chain.  It does not persist a primitive
H4 tensor and hence does not need the Section 67 reverse cyclic module for
that one-step use.  Any effect of the strict source beyond the consumed
planes remains outside this closure.

### 72.2 The row-Gram cost gate

The apparent `O(n^2)` readout in (72.2) must not hide construction of `G'`.
The actual covariance push may already own `S'=WRW^T`, but it does not imply
that the weight row Gram `G'=WW^T` is resident.  A generic new `G'` is one
full dense Gram product per active affine leg.  Six such products do not fit
the r55 suffix slack, and even one exceeds the strict `0.803` margin.

Therefore the forward trace route is deployably cheap iff at each selected
seam either

1. `G'`, `D_sG'`, and `G' circ S'` are literal views/reductions of an
   already-paid collision intermediate; or
2. the proposal uses the CP-costate matvec action of Section 71 instead and
   passes its rank/orientation gates.

Weights-only availability is not a billing receipt.  The full-AQ surface,
the forward plane formula, and the reverse CP action are alternative
implementations of the same Price-trace value; at most one receives credit.

The literal state audit resolves this fork for the current architectures.

- On-disk D11 has **no** separately typed low Gram.  It carries only total
  covariance, pushes that total through `W`, and its Q24/quartic routines
  consume the merged plane.  `G'` cannot be recovered from total `S'`, so
  (72.2) is not free in D11.
- The strict six-reset late-AQ schedule costing about `48.197` separately
  charges the AQ Gram `SS^T`.  When the chaos-metric scale is embedded in
  `S`, this is algebraically the required
  `W diag(scale) W^T`.  Retaining that buffer through (72.2), before it is
  merged into total covariance, gives a literal zero-new-GEMM CSE.
- The terminal-total schedule costing about `46.640` explicitly omits the
  final separated Gram.  It cannot claim forward trace reuse at that seam;
  it must omit an unneeded terminal trace, use the reverse CP action, or
  pay the Gram.

Thus `AQ55 strict48.197 + retained separated Gram + exact collision sewing`
is a concrete implementation candidate for the Price trace planes within
the remaining `0.803`.  It is still conditional on buffer lifetime,
normalization identity, complete-old subtraction, and endpoint value; the
ordinary D11 lineage does not inherit this conclusion.

### Graph split

- `P2R0G19B_MATH_PRICE_TRACE_FORWARD_PLANES` — **solved exact**: plane closure (72.1)--(72.4).
- `P2R0G19B_MATH_PRICE_TRACE_STRICT_SEWING` — **open literal-old receipt**: subtract the synchronized transports of (72.6), not nominal collision families.
- `P2R0G19B_MATH_PRICE_TRACE_ROW_GRAM_REUSE` — **open decisive cost gate**: prove `WW^T` products are resident/CSE; otherwise the multi-layer forward route is over budget.
- `P2R0G19B_MATH_PRICE_TRACE_IMPLEMENTATION_FORK` — **binding no-double-count**: choose full surface, forward resident-plane injection, or reverse CP action for the same trace value.
- `P2R0G19B_MATH_PRICE_TRACE_AQ55_STRICT_CSE` — **conditional concrete GO**: strict48.197 paid Gram is the needed `G` if scaled and retained; on-disk D11 and terminal-total final seam do not own it.
- `P2R0G19B_MATH_AFFINE_Q211_PRICE_TRACE_SCOPE` — **binding ownership gate**: old inherited rows must be projected; full AQ surface makes it duplicate; true-minus-AQ residual remains separate.

## 73. Exact consume-and-retire normal form for the Price trace

Section 72 deliberately left the three birth-collision compensators
separate.  They are not independent.  In the entrywise symmetric-tensor
convention, the `(4)` and `(22)` compensators combine into one Gaussian
pairing tensor, so a strict trace birth needs only one genuinely new
cross-pair word plus the already-typed Star31 path.

Let `diag R=1`, put `R^o=R-I`, and define

\[
({\cal P}_A)_{abcd}
=A_{ab}A_{cd}+A_{ac}A_{bd}+A_{ad}A_{bc}.
\tag{73.1}
\]

Let `E31(H)` be the strict `(31)` injection whose entry on the orbit of
`(i,i,i,j)`, `i ≠ j`, is `H_ij`, and which is zero on every other equality
partition.  Directly reading the four source partitions of the six-term
trace tensor gives

\[
\begin{array}{c|cccc}
 & (4)&(31)&(22)&(211)\\ \hline
 {\cal Z}_{I,R}&6&3R_{ij}&2&R_{jk}.
\end{array}
\tag{73.2}
\]

Because `2 P_I` has `(4)=6`, `(22)=2`, and zero `(31),(211),(1111)`, the
full tensor identity is

\[
\boxed{
 P_{211}{\cal Z}_{I,R}
 ={\cal Z}_{I,R}-2{\cal P}_I-E_{31}(3R^o).}
\tag{73.3}
\]

This is an entrywise identity.  A monomial-polynomial convention would
multiply the displayed partition entries by their orbit sizes and must not
be mixed with (73.3).

### 73.1 One dense leg leaves one cross-pair word

For one affine leg `W`, define

\[
G=WW^T,\qquad C=WRW^T,\qquad
\Delta=C-G=W(R-I)W^T,
\tag{73.4}
\]

and let `S_W(H)=W^⊗4 E31(H)` be the literal full affine Star31 push.
Naturality of pairing and trace tensors gives

\[
W^{\otimes4}{\cal Z}_{I,R}={\cal Z}_{G,C},
\qquad
W^{\otimes4}{\cal P}_I={\cal P}_G.
\tag{73.5}
\]

Since `{\cal Z}_{G,G}=2{\cal P}_G`, equations (73.3)--(73.5) collapse to

\[
\boxed{
 W^{\otimes4}P_{211}{\cal Z}_{I,R}
 ={\cal Z}_{G,\Delta}-{\cal S}_W(3R^o).}
\tag{73.6}
\]

Thus the K4 and constant-K22 source compensators must **not** be transported
separately.  Doing so would form the unnecessary products
`W^⊙3 W^T` and `W^⊙2 (W^⊙2)^T`; their contributions cancel
exactly against the diagonal K4 transport inside `2 P_G`.

Writing `g=diag G` and `delta=diag Delta`, the complete collision planes of
the new cross word are

\[
\boxed{
\begin{aligned}
k_4^{tr}&=6g\odot\delta,\\
K_{31}^{tr}&=3\{D_g\Delta+D_\delta G\},\\
K_{22}^{tr}&=g\delta^T+\delta g^T+4G\odot\Delta.
\end{aligned}}
\tag{73.7}
\]

Everything in (73.7) is diagonal, outer-product, or Hadamard work once the
two pair matrices `G,C` are resident.  The complete strict action also
contains the old Star31 action

\[
-3\langle {\cal S}_W^*(\lambda_4,\Lambda_{31},\Lambda_{22}),R^o\rangle.
\tag{73.8}
\]

Calling a post-`W` K31 plane "Star4" is not enough for (73.8): the resident
object must be the full affine push of the source tensor `E31(3R^o)` with
the same masks and normalization.

There is a useful exact quotient formulation.  If `F` is the complete
future map and the old row space contains the endpoint image of
`S_W(E31(3R^o))`, then

\[
P_{old}^\perp F W^{\otimes4}P_{211}{\cal Z}_{I,R}
=P_{old}^\perp F{\cal Z}_{G,\Delta}.
\tag{73.9}
\]

Hence a joint old-residualized full cross-word feature is exactly the
strict-trace residual under this literal Star ownership receipt.  If the
receipt is absent, using the full cross word is still a lawful estimator
feature, but it is not pure-q211 attribution and its collision novelty must
be graded as part of that feature.

### 73.2 Exact retirement and the target-weighted escape

Let `R_coll` read all three planes in (73.7) at the consuming seam and let
`F_+` be every later query.  The trace source may be consumed and discarded
before the next nonlinear reset if and only if

\[
\boxed{
 \ker R_{coll}\subseteq\ker F_+
 \quad\Longleftrightarrow\quad
 F_+=A R_{coll}\ \hbox{on the transported trace-source span}.}
\tag{73.10}
\]

For a fixed target portfolio, the minimum number of extra continuations is

\[
q_{extra}
=\operatorname{rank}\!\begin{bmatrix}R_{coll}\\F_+\end{bmatrix}
 -\operatorname{rank}R_{coll}.
\tag{73.11}
\]

The certified 18.7% replay injects truth-minus-host K21/K22/K31 planes
locally into the covariance tangent.  For that **fixed tangent instrument**,
(73.10) is the correct symbolic route to `q_extra=0`: the new source is
formed directly in the same collision coordinates.  This does not prove
retirement for a nonlinear estimator which later retains CP4 or reads a
q211/1111 component.  Its exact omitted endpoint error is

\[
e_{dark}
=\sum_l P_{old}^\perp F_{+,l}
 (I-R_{coll,l}^{+}R_{coll,l})
 W_l^{\otimes4}P_{211}{\cal Z}_{I,R_l},
\tag{73.12}
\]

and the adversarial promotion gate is

\[
\boxed{\|e_{dark}\|_{L^2}\le3.80\times10^{-6}.}
\tag{73.13}
\]

Generic correlated ReLU resets do not force (73.12) to vanish; Section 67
supplies an explicit broad-rank witness.  Therefore (73.10), or a direct
endpoint-weighted evaluation/bound of (73.12), is the decisive escape from
the generic H211 reset-rank theorem.

### 73.3 Typed AQ55 liveness contract

The strict late-AQ schedule is presently an algebraic census, not an
implemented tape.  A lawful implementation at each of the six L9--L14
suffix resets must order its objects as follows:

1. push the AQ coefficient bank and form the **already billed** low Gram
   `G_l=F_l F_l^T` in the declared chaos metric;
2. keep `G_l` separately typed and live;
3. form the already-billed two-sided total/rest covariance push `C_l`;
4. set `Delta_l=C_l-G_l`, form (73.7), and apply the synchronized Star31
   old subtraction while `G_l,C_l` are both live;
5. only then merge/reuse buffers for the reset and next state.

The `48.197` strict subtotal pays one such `G_l` at all six resets and leaves
about `0.803`.  The `46.640` terminal-total variant omits the separated Gram
at L14; no `G`-dependent trace consumer may be claimed there unless the
terminal observation annihilates it or it is evaluated by a separately
billed target action.  The CP4/Star4/pair arrays are not otherwise included
in the `48.197` census.  Equation (73.6) removes the CP4/pair requirement,
but the one literal Star31 path remains a hard typing/billing receipt.

The paid Gram must also have the correct weight.  More generally, if the
trace metric at a source is `D_a=Diag(a)` while the paid metrics are
`D_{rho_t}`, reuse requires

\[
W D_aW^T\in\operatorname{span}_t\{W D_{\rho_t}W^T\}.
\tag{73.14}
\]

For an invertible square dense `W`, the map
`a mapsto W Diag(a)W^T` is injective.  Hence a single paid bank is exact iff
`a` is proportional to its declared scale vector.  Homogeneous zero bias
passes after one scalar normalization; heterogeneous bias generally does
not and needs a metric-bank span or a new Gram.  This is why the zero-bias
trace result cannot be silently extended to arbitrary AQ gates.

The elementwise work in (73.7) is `O(n^2)` per seam and can use one recycled
`n by n` scratch buffer.  It does not change the leading GEMM census, but its
measured writes/hooks must fit the remaining `0.803`; the dense-products-only
screen is not final billing.

### 73.4 Value and ownership boundary

At zero bias the Price coefficient is

\[
c={\pi-4\over8\pi},\qquad
P_{old}^\perp\Delta_{211}^{P,(2)}
=-{c\over2}P_{old}^\perp P_{211}{\cal Z}_{I,R}.
\tag{73.15}
\]

Equations (73.6)--(73.15) therefore give a lawful label-free source for the
quadratic Price-trace residual.  They do **not** give:

- the higher analytic terms of the exact Gaussian q211 orthant surface;
- a true-minus-full-AQ residual with one or more genuinely non-Gaussian
  slots;
- additive value when a fused full-AQ H4 surface already owns (73.15); or
- a raw-MSE prediction from the local coefficient alone.

The feature must be complete-old residualized and its joint endpoint gain
must be measured/certified.  Reaching raw `1e-8` from `2.9409e-8` requires a
joint reduction of `1.9409e-8`; a cheap nonzero trace action is not by itself
goal scale.  Its importance is narrower but concrete: it is the first
strict-q211 source in this lane for which source information, one-transition
collision action, and the late-AQ dense-product census can all fit at once.

### Graph split

- `P2R0G19B_MATH_TRACE_211_IDENTITY_PAIR_FUSION` — **solved exact**: (73.3), K4 plus constant-K22 equals `2 P_I`; separate CP4/K22 products are forbidden double work.
- `P2R0G19B_MATH_TRACE_211_ONE_STEP_CROSS` — **solved exact**: consume-and-retire normal form and planes (73.6)--(73.8).
- `P2R0G19B_MATH_TRACE_211_STAR_OWNERSHIP` — **open literal receipt**: same-source `E31(3R^o)` must ride the actual Star31 affine push.
- `P2R0G19B_MATH_TRACE_211_RETIRE_GATE` — **open decisive target gate**: prove (73.10) on the certified/full consumer or bound (73.12) by (73.13).
- `P2R0G19B_MATH_TRACE_211_AQ55_LIVENESS` — **conditional concrete GO**: strict `48.197` owns all six `G` products only under the typed order in Section 73.3.
- `P2R0G19B_MATH_TRACE_211_WEIGHTED_METRIC` — **solved iff / biased open**: exact reuse is (73.14); one invertible-bank route requires proportional weights.
- `P2R0G19B_MATH_TRACE_211_VALUE` — **open blocking**: quadratic Price trace only; complete-old endpoint gain and raw-goal scale are unmeasured.

### 73.5 The Star31 term is independent information and is not in `48.197`

The remaining receipt in (73.8) is not bookkeeping.  It contains information
which the resident pair matrices `G,C` do not identify.

Put

\[
A=W(R-I),\quad H=W^{\odot3},\quad V=W^{\odot2},\quad
B=V\odot A,\quad D=W\odot A.
\tag{73.16}
\]

Direct polarization of the source polynomial
`12 sum_i x_i^3 (R^o x)_i` gives the complete pushed Star31 collision planes

\[
\boxed{
\begin{aligned}
k_4^{star}&=12\operatorname{diag}(HA^T),\\
K_{31}^{star}&=3HA^T+9BW^T,\\
K_{22}^{star}&=6(DV^T+VD^T).
\end{aligned}}
\tag{73.17}
\]

Even if `A` is retained from the first half of the paid covariance
conjugation, a forward reconstruction of (73.17) generically needs the three
independent dense products `HA^T`, `BW^T`, and `DV^T`.  Six such seams are
far outside the `0.803` strict slack.

This cost obstruction reflects an exact information obstruction.  Under

\[
(W,R)\longmapsto(WO,O^TRO),\qquad O\in O(n),
\tag{73.18}
\]

both `G=WW^T` and `C=WRW^T` are invariant.  The strict external equality
projectors `P31,P211`, however, are equivariant only under signed
permutations, not a generic orthogonal `O`.  Imposing
`diag(O^TRO)=1` still leaves, at a regular point, a family of dimension

\[
{n(n-1)\over2}-(n-1)={(n-1)(n-2)\over2},
\tag{73.19}
\]

and (73.17) varies generically along this family.  Therefore no function of
`G,C` alone can recover the Star31 subtraction.  The paid low Gram and total
covariance are structurally insufficient, not merely inconvenient buffers.

The on-disk D11 lineage has a literal Star4 family, but the abstract
strict-`48.197` suffix census contains only the D10 two-sided plane, AQ
coefficient push, independent low Gram, eigensolve, and K32 quadrature.  It
does **not** bill the D11 CP3/CP4/Star4 affine/reset/arena path.  Consequently
the Price-trace forward candidate is currently **cost-HOLD**, not a complete
conditional GO, unless one of the following is proved:

1. the exact `E31(3R^o)` affine push replaces an already-billed product;
2. the complete-old/future costate annihilates (73.17); or
3. a low-rank target action evaluates (73.8) within the remaining cap.

For the third escape, let the consuming K31 and symmetric K22 costates have
ranks `k31_l,k22_l`.  Contracting each of the two K31 products and the one
K22 product in (73.17) against rank factors costs, conservatively,

\[
\boxed{
C_{star}^{act}
\le {2\over n}\sum_l(2k_{31,l}+k_{22,l}),}
\tag{73.20}
\]

before factor formation, hooks, and writes.  Each factorized bilinear scalar
uses two `n by n by k` passes; literal resident transformed factors may lower
this bound but must be named.  With six seams, equal constant ranks, width
1024, and the entire `0.803` assigned to this action, (73.20) requires

\[
k_{31}=k_{22}=k\le22.
\tag{73.21}
\]

Thus the sharp target-weighted escape is no longer “paid `G` exists.”  It is
“paid `G` exists **and** the old-projected Star31 Bellman action is zero or
has a jointly recurrent rank at most the gate (73.20).”  A generic full-rank
plane costate fails.

### Additional graph split

- `P2R0G19B_MATH_TRACE_211_STAR_FORWARD_COST` — **closed generic NO**: (73.17) needs about three fresh dense products per seam when the literal Star path is absent.
- `P2R0G19B_MATH_TRACE_211_PAIR_INFORMATION_NOGO` — **solved exact**: gauge family (73.18)--(73.19) proves `G,C` cannot identify the strict Star subtraction.
- `P2R0G19B_MATH_TRACE_211_STAR_TARGET_RANK` — **open sharp escape**: old-projected Bellman factors must pass (73.20), with equal-rank strict-six ceiling `k<=22` before extras.
- `P2R0G19B_MATH_TRACE_211_AQ55_LIVENESS` — **status correction to cost-HOLD**: the paid Gram is resident under the declared liveness contract, but the suffix is not deployable until the Star31 ownership/rank gate also passes.

### 73.6 A generic future ReLU makes the Star costate full rank

The rank-`22` escape in (73.21) is a structural receipt, not a plausible
generic consequence of a scalar endpoint.  For the centered normalized
Gaussian ReLU covariance kernel, its correlation derivative is

\[
\Gamma_{ij}=p(R_{ij}),\qquad
p(r)={1\over4}+{\arcsin r\over2\pi}.
\tag{73.22}
\]

If the next-cut covariance costate is the rank-one matrix `uu^T`, one reset
pulls it back to

\[
\boxed{
\Lambda^-=\Gamma\odot(uu^T)
=D_u\,p[R]\,D_u.}
\tag{73.23}
\]

At `R=I`,

\[
p[I]={1\over4}(I+\mathbf1\mathbf1^T),
\tag{73.24}
\]

which has rank `n`.  Its determinant is a nonzero analytic function of the
correlation entries, so `p[R]` is full rank on an open dense set.  Whenever
all entries of `u` are nonzero, (73.23) is therefore full rank generically.
The same conclusion holds for the strictly off-diagonal convention: the
constant off-diagonal matrix has eigenvalues `n-1,-1,...,-1` and is full
rank.

Consequently even a scalar terminal variance query generally produces a
full-rank pair costate after one future ReLU reset.  Substituting
`k22=n=1024` in (73.20) costs at least two equivalents for **one** seam and
already exceeds the `0.803` strict slack.  Multiple resets only strengthen
the obstruction.  Q24 truncation does not remove it: the constant plus
first active edge terms already give a generically full-rank matrix.

The only lawful low-rank reopenings are:

- an immediate consumer before any such reset;
- support of `u` on at most the allowed rank;
- a proved block/common-ray kernel for which `p[R]` has low rank on the live
  support;
- complete-old projection annihilating the high-rank costate component; or
- a directly compiled endpoint scalar oracle which never forms this pair
  costate.

For an approximate low-rank route the relevant receipt is the
**source-weighted, complete-old-projected** singular tail, not algebraic
rank alone.  A rank-one surrogate may always be rescaled to match one
already-known scalar, so such scalar refitting is not a provider proof; the
factors must be generated label-free from the declared reverse recurrence
and frozen across the panel.

### Further graph split

- `P2R0G19B_MATH_TRACE_211_RELU_COSTATE_FULL_RANK` — **solved generic NO**: (73.22)--(73.24) turn a rank-one endpoint into a full-rank pair costate after one reset.
- `P2R0G19B_MATH_TRACE_211_STAR_LOW_RANK_EXCEPTION` — **open structural only**: immediate, sparse-support, block/common-ray, old-annihilated, or direct-scalar cases listed above.

## 74. Exact CP3-node fusion for q211, and why AQ55 K32 does not own it

There is a second target-conditioned escape which computes the full
repeated-slot action rather than transporting an H4 tensor.  It is exact
only when the resident CP3 evaluator is a **shared latent cubature**, not a
collection of equal-length per-unit quadratures.

Let `X_i=Y_i-EY_i` and suppose a literal cubature
`{(z_q,w_q):q=1,...,Q}` evaluates products under the common latent law.  For
one endpoint atom `L_{i;jk}=alpha_i beta_j gamma_k`, define at every node

\[
A_{2q}=\sum_i\alpha_iX_i(z_q)^2,\qquad
B_q=\sum_j\beta_jX_j(z_q),\qquad
C_q=\sum_k\gamma_kX_k(z_q).
\tag{74.1}
\]

Let `Sigma=E[XX^T]` and `v=diag Sigma`.  The complete, unmasked connected
q211 action is exactly

\[
\boxed{
S_{211}
=\sum_qw_qA_{2q}B_qC_q
-(\alpha^Tv)(\beta^T\Sigma\gamma)
-2\alpha^T\{(\Sigma\beta)\odot(\Sigma\gamma)\}.
\tag{74.2}
\]

The signs in (74.2) are simply the three fourth-cumulant pairings.  A CP3
atom on the identical cubature already forms `B_q,C_q` and the analogous
linear `A_{1q}`.  Replacing only the repeated slot by `A_{2q}` therefore
costs `Q n R_c` scalar reductions for `R_c` endpoint atoms, plus old pair
corrections.  It does not materialize the `1+r+r(r+1)/2` Cameron--Martin
jet.

Strict external-211 masking is also finite at node level.  For the raw node
product, inclusion--exclusion over `i=j`, `i=k`, and `j=k` gives

\[
\begin{aligned}
[A_2BC]_{distinct}={}&A_2BC-A_{3,\alpha\beta}C
-A_{3,\alpha\gamma}B-A_2D_{2,\beta\gamma}
+2D_{4,\alpha\beta\gamma},
\end{aligned}
\tag{74.3}
\]

where

\[
A_{3,\alpha\beta}=\sum_i\alpha_i\beta_iX_i^3,\quad
D_{2,\beta\gamma}=\sum_i\beta_i\gamma_iX_i^2,\quad
D_{4,\alpha\beta\gamma}=\sum_i\alpha_i\beta_i\gamma_iX_i^4.
\tag{74.4}
\]

The covariance-pairing terms in (74.2) receive the matching two-index
exclusions.  Thus strict masking changes only a fixed number of `O(Qn)`
reductions/matvecs; it does not reopen an H4 tensor.

### 74.1 Literal-node criterion

The fusion is lawful iff all of the following are literal:

1. every `X_i(z_q)` is evaluated at the **same latent point** `z_q` and
   with the same weight `w_q` for all units in the product;
2. the paid CP3 action uses the products of the three node sums, rather than
   separately integrated marginal coefficients;
3. centering, collision masks, endpoint factors, and the AQ/true comparator
   are identical; and
4. the cubature error for degree-four ReLU--AQ products is included in the
   `3.80e-6` endpoint budget.

The current strict AQ55 `K=32` receipt does not satisfy item 1.  Its nodes
are per-unit one-dimensional Fourier/resolvent nodes for a generalized
chi-square marginal.  Joint CP3/q211 moments instead depend on

\[
\chi_{ijk}(t_1,t_2,t_3)
=\det\!\left(I-i\sum_{a\in\{i,j,k\}}t_aA_a\right)^{-1/2}
\times(\hbox{the corresponding linear tilt}),
\tag{74.5}
\]

which is not determined by the three marginal characteristic functions.
Giving those marginal rules the same node labels does not turn their
product into (74.5).  Equivalently, joint laws with the same one-unit
marginals can have different CP3 and q211 actions.

Therefore the node fusion is an exact architecture theorem but **not** a
CSE receipt for the `48.197` suffix.  It reopens only if a literal joint
CP3 cubature/tape is introduced or proved already resident.  A full fused
AQ H211 surface and (74.2) are alternative implementations of the same AQ
law; the true-minus-AQ residual remains separate.

### Graph split

- `P2R0G19B_MATH_AQ_Q211_CP3_NODE_FUSION` — **solved exact conditional**: shared-node action (74.1)--(74.4), incremental `O(QnR_c)` reductions.
- `P2R0G19B_MATH_AQ_Q211_CP3_NODE_TYPING` — **binding iff**: common latent nodes/weights, centering, masks, factors, and comparator must be literal.
- `P2R0G19B_MATH_AQ55_K32_NOT_JOINT_CP3` — **closed current CSE claim**: per-unit Fourier nodes do not identify the joint characteristic function (74.5).
- `P2R0G19B_MATH_AQ_Q211_CP3_CUBATURE_ERROR` — **open fidelity/value**: a newly designed joint cubature must bound its compiled endpoint error by the global budget.

## 75. Free-semicircular full-H4 closure from the paid quadratic Gram

The strict Price word is not the only use of the trace-plane algebra.  There
is a stronger replacement architecture in which the **entire** local
affine-quadratic H4 surface is a trace word.  It avoids the Star31 subtraction
because it does not claim an isolated external-211 source.

Use the half-scaled quadratic convention

\[
Q_i(z)={1\over2}(z^TA_iz-\operatorname{tr}A_i),\qquad
Q_{ij}={1\over2}\operatorname{tr}(A_iA_j),\qquad q_i=Q_{ii}.
\tag{75.1}
\]

Put `tau=r^{-1}tr` and `c_ij=tau(A_iA_j)=2Q_ij/r`.  Suppose the core family
obeys the degree-four free-semicircular Wick law

\[
\boxed{
\tau(A_iA_jA_kA_l)
=c_{ij}c_{kl}+c_{il}c_{jk}.}
\tag{75.2}
\]

This is noncommutative: the alternating word is retained rather than set to
zero.  The exact Gaussian-quadratic cumulant generating function gives, in
the gauge (75.1),

\[
\begin{aligned}
\kappa_4(Q_i,Q_i,Q_i,Q_j)&=3\operatorname{tr}(A_i^3A_j),\\
\kappa_4(Q_i,Q_i,Q_j,Q_j)&=
2\operatorname{tr}(A_i^2A_j^2)+\operatorname{tr}(A_iA_jA_iA_j).
\end{aligned}
\tag{75.3}
\]

Substitution of (75.2) yields the complete pure-core collision planes

\[
\boxed{
\begin{aligned}
K4_i&={24\over r}q_i^2,\\
K31_{ij}&={24\over r}q_iQ_{ij},\\
K22_{ij}&={8\over r}(q_iq_j+2Q_{ij}^2).
\end{aligned}}
\tag{75.4}
\]

Equivalently, the full symmetric H4 tensor on this law is

\[
\boxed{{\cal K}_4^{free}={4\over r}{\cal Z}_{Q,Q}
={8\over r}{\cal P}_Q.}
\tag{75.5}
\]

The commutator sector is not killed.  Indeed

\[
\tau([A_i,A_j]^2)=2(c_{ij}^2-c_{ii}c_{jj}),
\tag{75.6}
\]

so its free-law value is fixed by the same quadratic Gram.  What remains
outside (75.5) is the **instance-specific deviation** from the free Wick
law, not noncommutativity itself.

### 75.1 Affine-isotropic extension is one polarized trace word

Let `L_ij=u_i^T u_j` and let `C=L+Q` be the total low-AQ covariance in the
same normalization.  Assume the exact second-word isotropy relation

\[
u_a^TA_iA_j u_b=L_{ab}c_{ij}.
\tag{75.7}
\]

Then the full affine-quadratic repeated planes are

\[
\boxed{
\begin{aligned}
K4_i&={24\over r}q_iC_{ii},\\
K31_{ij}&={12\over r}(q_iC_{ij}+C_{ii}Q_{ij}),\\
K22_{ij}&={4\over r}
(q_iC_{jj}+C_{ii}q_j+4Q_{ij}C_{ij}).
\end{aligned}}
\tag{75.8}
\]

In the six-term convention of Sections 72--73 this is simply

\[
\boxed{{\cal K}_{4}^{free,aff}={4\over r}{\cal Z}_{Q,C}.}
\tag{75.9}
\]

Thus one retained quadratic Gram `Q`, one already-carried total covariance
`C`, and `O(n^2)` diagonal/outer/Hadamard work supply K4, K31, and K22.
The generic `P3 P1^T` K31 product is absent because (75.2) reduces
`tr(A_i^3A_j)` to the covariance Gram.  Unlike Section 73, no strict
equality-mask compensator or Star31 path is part of this full-law source.

### 75.2 Exact off-law residual

For arbitrary cores define

\[
E_{ijkl}:=\tau(A_iA_jA_kA_l)
-c_{ij}c_{kl}-c_{il}c_{jk},
\tag{75.10}
\]

and the affine isotropy defect

\[
D_{ab;ij}:=u_a^TA_iA_j u_b-L_{ab}c_{ij}.
\tag{75.11}
\]

The pure-core collision errors are exactly

\[
\boxed{
\begin{aligned}
\Delta K4_i&=3rE_{iiii},\\
\Delta K31_{ij}&=3rE_{iiij},\\
\Delta K22_{ij}&=r(2E_{iijj}+E_{ijij}).
\end{aligned}}
\tag{75.12}
\]

The mixed vector-leg errors are fixed linear combinations of (75.11) with
the same slot multiplicities as (75.3).  Therefore the approximation has a
fully typed fail-fast observable.  If
`lambda4_l,Lambda31_l,Lambda22_l` are the complete-old-projected future
costates, the pure-core omitted scalar at layer `l` is

\[
\epsilon_l^{E}
=3r\langle\lambda_{4,l},E_{iiii,l}\rangle
+3r\langle\Lambda_{31,l},E_{iiij,l}\rangle
+r\langle\Lambda_{22,l},2E_{iijj,l}+E_{ijij,l}\rangle.
\tag{75.13}
\]

Add the analogous contracted `D` terms.  The deterministic promotion gate
is the corpus endpoint bound

\[
\boxed{
\left\|\sum_l\beta_l(\epsilon_l^E+\epsilon_l^D)
\right\|_{L^2}\le3.80\times10^{-6},}
\tag{75.14}
\]

or the corresponding robust Schur-gain inequality if the closure is fitted
as a feature rather than asserted as exact physics.  A statement that the
cores look random or diffuse is not a substitute for (75.14).

### 75.3 The paid-Gram bill can retain `Q` without a second product

The AQ coefficient bank has a linear block of width `r=55` and a symmetric
quadratic block of width `d=r(r+1)/2=1540`.  Its concatenated covariance Gram
is the sum of the two block Grams; there are no cross terms between feature
blocks.  Replacing one concatenated Gram call by two block calls and retaining
the quadratic block before summation has the same leading flop count.

Splitting the already-paid width-1595 Gram into its width-55 linear and
width-1540 quadratic block products has the same leading flop sum:

\[
55+1540=1595.
\tag{75.15}
\]

It therefore retains `Q` as a free **typed buffer/CSE**, but it does not by
itself lower the strict subtotal.  The honest default remains

\[
\boxed{48.197\quad\hbox{with about }0.803\hbox{ slack}.}
\tag{75.16}
\]

The earlier apparent `0.322` saving is lawful only if the linear Gram
`L=UU^T` is independently resident and already paid, so its width-55 block
can truly be omitted rather than recomputed as part of the split.  In either
case, reuse is lawful only if the feature metric makes the retained
quadratic block literally equal to (75.1) and `C` remains separately
resident through the plane formation.

### 75.4 Closure across depth and ownership

Joint semicircularity and (75.7) are preserved by linear combinations, so a
dense affine leg transports

\[
Q\mapsto WQW^T,\qquad C\mapsto WCW^T,
\tag{75.17}
\]

and (75.9) intertwines exactly.  The nonlinear reset/refit is the real
closure gate: its new cores must again satisfy (75.2),(75.7), or their new
`E,D` actions must remain within (75.14).  A generic unconstrained AQ refit
does not inherit this merely from matching covariance.

This branch supplies the full local AQ H4 approximation.  It must replace,
or be jointly residualized against, the deployed CP4/Star/pairing and any
fused H211 surface.  It receives no additive credit on top of those same
planes.  The true-minus-AQ dependent residual remains separate.  Reaching
the raw goal still requires the complete-old endpoint gain, not just small
local `E,D`.

### Graph split

- `P2R0G19B_MATH_AQ_FREE_WICK_H4_CLOSURE` — **solved exact law**: noncommutative full-H4 identity (75.2)--(75.9).
- `P2R0G19B_MATH_AQ_FREE_WICK_DEFECT` — **solved exact observable / instance open**: residual tensors and plane errors (75.10)--(75.13).
- `P2R0G19B_MATH_AQ_FREE_WICK_ENDPOINT_GATE` — **open blocking**: compiled corpus defect must pass (75.14), with no diffuse-law shortcut.
- `P2R0G19B_MATH_AQ_FREE_WICK_QGRAM_CSE` — **conditional typed CSE**: split and retain the quadratic block at unchanged strict subtotal `48.197`/slack `0.803`; `47.875` requires an independently paid resident linear Gram.
- `P2R0G19B_MATH_AQ_FREE_WICK_RESET_CLOSURE` — **open structural**: linear transport is exact; every nonlinear refit must preserve the law or pay its new `E,D` defect.
- `P2R0G19B_MATH_AQ_FREE_WICK_OWNERSHIP` — **binding replacement rule**: full AQ H4, old CP4/Star/pair, and fused H211 are residualized once; true-minus-AQ is separate.
- `P2R0G19B_MATH_AQ_FREE_WICK_VALUE` — **open blocking**: complete-old raw gain is unknown despite the favorable arithmetic.

### 75.5 Exact finite-r GOE mean correction

There is one useful finite-r calculation which sharpens, but does not prove,
the defect gate.  Let the cores be a jointly Gaussian real-symmetric family
with entry covariance

\[
E[(A_i)_{ab}(A_j)_{cd}]
={c_{ij}\over r+1}
(\delta_{ac}\delta_{bd}+\delta_{ad}\delta_{bc}).
\tag{75.18}
\]

This normalization gives `E tau(A_iA_j)=c_ij` exactly.  Wick contraction of
the four matrix entries gives

\[
\boxed{
E\tau(A_iA_jA_kA_l)
=c_{ij}c_{kl}+c_{il}c_{jk}
+\varepsilon_r c_{ik}c_{jl},
\qquad
\varepsilon_r={r+3\over(r+1)^2}.}
\tag{75.19}
\]

The first two pairings have index-loop factor `r(r+1)^2`; after the outer
`1/r` trace and the two covariance denominators they contribute exactly
one.  The crossing pairing has factor `r(r+3)`, which gives the last term.

After the symmetric fourth-cumulant slot sum, (75.19) simply rescales every
pure-core plane in (75.4) by

\[
\boxed{1+{\varepsilon_r\over2}.}
\tag{75.20}
\]

At `r=55`, `epsilon_r=58/3136=29/1568`, so the ensemble-mean coefficient
correction is `1+29/3136`.  It is still an `O(n^2)` scalar adjustment and
can be fitted or fixed without a new matrix product.

Equation (75.20) removes the systematic real-GOE crossing bias.  It does
not bound the realized second-order-free fluctuation of `E`, and it says
nothing about the vector-state defect `D`.  Those remain target-weighted
terms in (75.14).

### 75.6 Exact affine isotropy has a capacity obstruction

The affine extension (75.7)--(75.9) is useful as a typed decomposition, but
it cannot be assumed exact for a high-capacity finite core.  Let `U` be the
`n by r` matrix with rows `u_a^T`.  If (75.7) holds for all `a,b,i,j` and
`rank U=r`, then

\[
U(A_iA_j-c_{ij}I)U^T=0.
\tag{75.21}
\]

A left inverse of `U` on both sides of (75.21) gives

\[
\boxed{A_iA_j=c_{ij}I\quad\hbox{for every }i,j.}
\tag{75.22}
\]

Choose any nonzero core `A_1`.  Since `A_1^2=c_{11}I`, it is invertible, and

\[
A_j={c_{1j}\over c_{11}}A_1
\tag{75.23}
\]

(with `A_j=0` when `c_1j=0`).  Hence the quadratic covariance Gram has rank
at most one.  If only the symmetrized condition is imposed, (75.22) is
replaced by a Clifford anticommutator law; the finite-dimensional Clifford
capacity bounds from Sections 53/56 then apply and are still far below a
full-width carrier.

Therefore a viable high-capacity use of the free-Wick closure must choose
one of two honest architectures:

1. a pure or low-rank-linear quadratic core using (75.4)--(75.5), with the
   high-rank Gaussian component carried independently; or
2. an approximate affine core which explicitly pays the target-weighted
   `D` defect in (75.14).

In the first architecture, nonlinear interaction with the independent
Gaussian rest belongs to the exact Mehler/heat-response lane.  It must not
be smuggled back into (75.9) by replacing `Q` with total `C`; cumulants of
independent components add and have no mixed preactivation fourth cumulant.
This pure-core-plus-heat split is the structurally consistent version of the
late-AQ proposal.

### Additional graph split

- `P2R0G19B_MATH_AQ_FREE_WICK_GOE_MEAN` — **solved exact ensemble identity**: finite-r crossing correction (75.19)--(75.20); realized fluctuation remains open.
- `P2R0G19B_MATH_AQ_FREE_WICK_AFFINE_CAPACITY_NOGO` — **solved exact**: full-rank `U` plus zero isotropy defect forces quadratic Gram rank at most one, (75.21)--(75.23).
- `P2R0G19B_MATH_AQ_FREE_WICK_PURE_PLUS_HEAT` — **open coherent architecture**: high-capacity pure quadratic free core plus separately certified Gaussian Mehler response.

### 75.7 Mean closure does not control the finite-instance K31 carrier

The small ensemble-mean correction (75.20) is not a provider-correlation
certificate.  The off-diagonal K31 signal of a finite semicircular core is
itself a trace fluctuation, and the Q-only closure is exactly the degree-one
projection of the spectral word `x^3`.

For the unit semicircle law,

\[
m_2=1,\qquad m_4=2,\qquad m_6=5.
\tag{75.24}
\]

Therefore the best scalar projection is `x^3 mapsto 2x`, with

\[
\boxed{
\operatorname{corr}(x^3,x)={m_4\over\sqrt{m_2m_6}}
={2\over\sqrt5},\qquad
{\|2x\|_2^2\over\|x^3\|_2^2}={4\over5}.}
\tag{75.25}
\]

Equivalently, the orthogonal residual `x^3-2x` has squared norm one out of
the target norm five: a `20%` word-energy tail.  This fails both campaign
provider screens,

\[
{2\over\sqrt5}<0.9253<0.98346,
\tag{75.26}
\]

and is far above the `3.28064%` tight value-tail allowance.  Thus a generic
finite semicircular core does **not** make the full K31 carrier accurate
enough merely because its expectation obeys free Wick.

The branch remains open only if complete-old/endpoint weighting annihilates
the `A_i^3-2A_i` residual, or if the already-paid Gram is changed to a
jointly refitted polynomial core `p(A)` containing the needed cubic word.
The latter is the one-Gram coadapted polynomial architecture, not the
Q-only free closure and not additive credit.

### Final graph correction for this branch

- `P2R0G19B_MATH_AQ_FREE_WICK_K31_FINITE_INSTANCE` — **closed generic strict carrier**: semicircle correlation `2/sqrt(5)` and `20%` word-energy tail, (75.24)--(75.26).
- `P2R0G19B_MATH_AQ_FREE_WICK_ENDPOINT_GATE` — **reopen only after old quotient**: must show the cubic-word residual is target-dark; ensemble-mean GOE correction is insufficient.
- `P2R0G19B_MATH_AQ_POLYNOMIAL_ONE_GRAM_REOPEN` — **open replacement architecture**: include the target degree combination in `p(A)` and refit jointly, with PSD/rest and commutator gates.

### 75.8 The finite-instance residual is CP4 for a free independent-component core

There is an exact way for the old CP3/CP4 carriers to repair the Q-only
finite-instance residual.  Semicircularity is stronger than necessary.
Suppose

\[
A_i=\sum_{s=1}^m U_{is}B_s,
\tag{75.27}
\]

where the centered generators `B_s` are freely independent in the trace
state, with free cumulants

\[
v_s=\kappa_2^{free}(B_s,B_s),\qquad
a_s=\kappa_3^{free}(B_s,B_s,B_s),\qquad
b_s=\kappa_4^{free}(B_s,B_s,B_s,B_s).
\tag{75.28}
\]

The noncrossing moment--cumulant formula gives

\[
c_{ij}=\sum_s v_sU_{is}U_{js},
\tag{75.29}
\]

\[
\boxed{
\tau(A_iA_jA_k)
=\sum_sa_sU_{is}U_{js}U_{ks},}
\tag{75.30}
\]

and

\[
\boxed{
E_{ijkl}
=\sum_sb_sU_{is}U_{js}U_{ks}U_{ls}.}
\tag{75.31}
\]

Thus the exact pure-core degree-three and degree-four law in the half gauge
is

\[
\boxed{
{\cal K}_3=r\sum_sa_su_s^{\otimes3},\qquad
{\cal K}_4={4\over r}{\cal Z}_{Q,Q}
+3r\sum_sb_su_s^{\otimes4},}
\tag{75.32}
\]

where `u_s` is column `s` of `U`.  The free-Wick defect is therefore a
diagonal CP4 tensor, not an omitted broad word.  In particular, the
`A^3-2A` semicircle residual highlighted in Section 75.7 is exactly repaired
when its fourth free cumulant lies in the synchronized CP4 carrier.

Dense neuron mixing preserves the decomposition with

\[
U\mapsto WU.
\tag{75.33}
\]

This is the clean mathematical synthesis of the covariance Gram, CP3, and
CP4 lanes: the Gram owns the two noncrossing pairings, CP3 owns the third
free cumulant, and CP4 owns the fourth free cumulant.  None is additive
twice.

The actual ownership test is still literal.  If `J_CP4` is the deployed
same-cut CP4 row map, exact repair is equivalent to

\[
\boxed{
P_{old}^\perp\left(3rE-J_{CP4}\theta_4\right)=0}
\tag{75.34}
\]

for some frozen amplitudes/factors with the same transport and reset seam.
The robust approximate gate replaces the left side by its complete-future
endpoint norm and requires it, jointly with CP3 and heat errors, below
`3.80e-6`.

There are three unresolved implementation constraints.

1. A generic finite matrix basis is not freely independent and its `E`
   need not have the diagonal CP form (75.31); CP rank `2048` is capacity,
   not an ownership proof.
2. The strict `48.197` suffix does not bill the deployed CP4 plane
   transport/read.  Sharing the coefficient columns `U` makes their
   one-sided transport resident, but K31/K22 consumption of the CP4 tensor
   still needs a literal fused/old action or its honest bill.
3. A nonlinear ReLU reset generally creates products of the generators and
   need not return to the linear span (75.27).  The common free-component
   factorization must be re-established at every refit cut, or its leakage
   joins (75.34).

### Graph split for the repaired free branch

- `P2R0G19B_MATH_AQ_FREE_COMPONENT_CP34` — **solved exact law**: common-factor decomposition (75.27)--(75.32).
- `P2R0G19B_MATH_AQ_FREE_COMPONENT_CP4_OWNERSHIP` — **open literal receipt**: exact iff (75.34) with the deployed same-cut CP4 factors.
- `P2R0G19B_MATH_AQ_FREE_COMPONENT_CP4_COST` — **open blocking**: coefficient transport may be shared, but CP4 K31/K22 consumption is absent from strict `48.197`.
- `P2R0G19B_MATH_AQ_FREE_COMPONENT_RESET` — **open structural**: nonlinear refit must preserve the common generator span/diagonal free cumulants.
- `P2R0G19B_MATH_AQ_FREE_WICK_K31_FINITE_INSTANCE` — **status refinement**: Q-only is closed at correlation `2/sqrt5`; Q plus a literal common-factor CP4 can repair that residual exactly.

## 76. Exact target-action census for the common-factor CP4 repair

The free-component identity (75.32) is useful under the `48.197` cap only
if its CP4 part can be consumed without reviving a full H4 plane.  This has
an exact target-conditioned formulation.  Write the complete-old residual
CP4 source at one synchronized cut as

\[
T_4=\sum_{s=1}^{m}\theta_s u_s^{\otimes4},
\qquad U=[u_1\ \cdots\ u_m]\in\mathbb R^{n\times m},
\tag{76.1}
\]

where `theta_s=3r b_s` for the literal free-cumulant block, after subtracting
every same-cut CP4 contribution already owned by the deployed card.  Put

\[
V_p=U^{\odot p},\qquad (V_p)_{is}=U_{is}^{p}.
\tag{76.2}
\]

In raw collision-plane convention,

\[
K_{4}=V_4\theta,
\qquad
K_{31}=V_3\operatorname{Diag}(\theta)V_1^T,
\qquad
K_{22}=V_2\operatorname{Diag}(\theta)V_2^T.
\tag{76.3}
\]

Any orbit multiplicities can be absorbed into the synchronized Bellman
costates below; (76.3) itself uses one entry of the symmetric tensor on each
face.  For costates `lambda_4,Lambda_31,Lambda_22`, the exact scalar action is

\[
\boxed{
\begin{aligned}
\mathcal A_4(T_4) = \sum_s\theta_s\{&(V_4^T\lambda_4)_s
+(V_3^T\Lambda_{31}V_1)_{ss}\\
&+(V_2^T\Lambda_{22}V_2)_{ss}\}.
\end{aligned}}
\tag{76.4}
\]

Thus a full-plane implementation needs exactly two `n by m` outer-product
banks per seam: one for `K31` and one for `K22`.  Its leading bill is

\[
\boxed{C_{\rm CP4,plane}=\sum_l {2m_l\over n}.}
\tag{76.5}
\]

The diagonal `K4` read is reduction-scale.  At `n=1024`, six suffix seams,
and slack `0.803`, (76.5) fits iff a constant common-factor rank satisfies

\[
\boxed{m\le \left\lfloor{0.803\,n\over12}\right\rfloor=68.}
\tag{76.6}
\]

In particular, a *proved* rank-55 free-component factorization would cost
`660/1024=0.64453125` equivalents and leave about `0.15847` before hooks.
The latent AQ rank `r=55` does not prove `m=55`: `m` is the CP rank of the
complete-old residual fourth free-cumulant tensor, and may be width-sized.

There is a second, genuinely target-weighted escape.  Suppose the literal
reverse tape exposes factorizations

\[
\Lambda_{31}=P_{31}Q_{31}^T,
\qquad
\Lambda_{22}=P_{22}D_{22}P_{22}^T,
\tag{76.7}
\]

of ranks `k31,k22`; an indefinite diagonal `D22` is allowed.  Then (76.4)
needs two projections for the directed `31` action and one for the symmetric
`22` action.  No `n by n` source plane is formed, and the leading bill is

\[
\boxed{
C_{\rm CP4,act}
=\sum_l {m_l(2k_{31,l}+k_{22,l})\over n^2}.}
\tag{76.8}
\]

For each channel one should take the cheaper of (76.5) and (76.8):

\[
C_l={m_l\over n}
\left[
\min\!\left(1,{2k_{31,l}\over n}\right)
+\min\!\left(1,{k_{22,l}\over n}\right)
\right].
\tag{76.9}
\]

For a width-sized residual `m=n` over six seams, the strict slack requires

\[
\boxed{
\sum_{l=1}^{6}(2k_{31,l}+k_{22,l})\le0.803\,n.
}
\tag{76.10}
\]

With constant equal ranks this is `k<=45` before hooks.  This is the sharp
low-rank Bellman loophole: a small **forward** CP/Chow atom count does not
imply (76.10), and an SVD computed only for this purpose is not resident
reuse.  The factors in (76.7) must be literal outputs of the synchronized
reverse tape, or their acquisition bill must be added.

### 76.1 Exact target-tail and ownership gates

Let `DeltaLambda31,DeltaLambda22` be the discarded parts of a proposed
low-rank reverse action and define the exact source planes

\[
B_{31}=V_3\operatorname{Diag}(\theta)V_1^T,
\qquad
B_{22}=V_2\operatorname{Diag}(\theta)V_2^T.
\tag{76.11}
\]

The layer error is not a rank heuristic; it is exactly

\[
e_l=\langle\Delta\Lambda_{31,l},B_{31,l}\rangle
   +\langle\Delta\Lambda_{22,l},B_{22,l}\rangle,
\tag{76.12}
\]

and a label-free sufficient receipt is

\[
|e_l|\le
\|\Delta\Lambda_{31,l}\|_F\|B_{31,l}\|_F
+\|\Delta\Lambda_{22,l}\|_F\|B_{22,l}\|_F.
\tag{76.13}
\]

After endpoint whitening and complete-old residualization, the coherent
sum of (76.12), or the robust sum of (76.13), must be at most `3.80e-6`.
One scalar/net with freely refitted amplitudes cannot certify the rank;
the frozen factor law must be shared across the required panel or probed by
enough independent same-state consumers, as in the tomography theorem of
Section 50.

Most importantly, (76.4) is a **consumption** theorem, not an acquisition
theorem.  If the existing CP4 carrier already owns the diagonal
free-cumulant tensor in (75.31), then (76.4) is a CSE or a replacement
implementation and has zero additive raw value.  The lawful source is

\[
\boxed{
T_4^{new}=P_{old}^{\perp}
\{3rE_{true}-T_{4,old}-T_{4,AQ\ surface}\},}
\tag{76.14}
\]

with the comparator term omitted when no AQ connected-H4 surface is billed.
Neither the resident covariance Gram nor the AQ eigensystem identifies the
amplitudes in (76.14).  They require a true-minus-comparator fourth-moment
observable or an exact common-free-component law with frozen cumulants.

Equation (76.8) also bypasses forward reset closure only when the complete
future costates are actually available at the birth seam.  If instead the
CP4 source is carried forward, every nonlinear reset must preserve the
common factorization; Sections 44, 62, 67, and 69 prove that a generic
correlated ReLU reset expands it to full common-frame rank.

### 76.2 Companion CP3 cost and the strict-suffix boundary

For completeness, the same free-component law has

\[
K_{21}=V_2\operatorname{Diag}(\alpha)V_1^T,
\tag{76.15}
\]

so a full companion CP3 plane costs `m/n` per seam, while a resident
rank-`k21` directed costate costs `2mk21/n^2`.  A six-seam rank-55 CP4
full-plane action leaves enough of the strict `0.803` slack for CP3 iff

\[
{12\cdot55\over1024}
+{12\cdot55\,k_{21}\over1024^2}\le0.803,
\tag{76.16}
\]

namely `k21<=251` before hooks.  Three full CP3/CP4 banks would cost
`18*55/1024=0.96680` and do not fit.  Therefore a claim that the complete
free-component correction fits must identify whether CP3 is already owned,
target-factorized as in (76.16), or separately billed.  The larger
terminal-total slack can change this arithmetic but not the ownership or
rank receipts.

### 76.3 Current typed disposition

The declared strict `48.197` AQ suffix is a forward tape.  It retains its
AQ coefficient bank around the paid Gram/reset, but it does **not** expose
literal factorized `Lambda31/Lambda22` endpoint costates.  Consequently
(76.8) is not current CSE: building a new reverse family must be billed.
Moreover, for coordinate free components followed by a generic dense
mixing matrix, the natural factor count is `m=n`, not the latent AQ rank
`r`.  The direct bill (76.5) is then twelve equivalents over six seams,
far above `0.803`.

The current status is therefore **COST-HOLD**, with only three exact reopen
conditions:

1. the complete-old CP4 action is already supplied at the same seam;
2. one fixed terminal query is fused into the forward source while `U` and
   its Hadamard powers are still live; or
3. a literal target factorization satisfying (76.10) and the endpoint-tail
   receipt (76.12) is added to the declared tape.

Forward AQ factors alone, a named D10 word, and the equality `r=55` imply
none of these conditions.

### Graph split

- `P2R0G19B_MATH_FREE_CP4_TARGET_ACTION` — **solved exact**: collision-plane action (76.3)--(76.4) and direct bill (76.5).
- `P2R0G19B_MATH_FREE_CP4_STRICT_RANK68_GATE` — **solved conditional cap**: a literal six-seam common factor fits strict slack only at `m<=68`, (76.6).
- `P2R0G19B_MATH_FREE_CP4_BELLMAN_LOW_RANK` — **open sharp escape**: resident reverse factors must pass (76.8)--(76.10); equal width-sized ranks require `k<=45` before hooks.
- `P2R0G19B_MATH_FREE_CP4_CURRENT_TAPE` — **cost-hold**: strict `48.197` has no resident Q31/Q22 reverse factorization and generically has `m=n`, so neither (76.5) nor (76.8) is presently free.
- `P2R0G19B_MATH_FREE_CP4_ENDPOINT_TAIL` — **open required receipt**: exact/robust errors (76.12)--(76.13) must fit `3.80e-6` after the complete old quotient.
- `P2R0G19B_MATH_FREE_CP4_NO_DOUBLE_CREDIT` — **binding ownership**: only (76.14) is new; an old CP4 or billed AQ surface is not additive value.
- `P2R0G19B_MATH_FREE_CP34_JOINT_COST` — **conditional**: companion CP3 must be old, low-target-rank, or separately billed; three full banks fail strict slack at rank55.

## 77. Quotient-low-rank expansion of the ReLU pair-kernel adjoint

The generic full-rank statement in Section 73.6 can be sharpened in the
only way relevant to a target-conditioned strict-q211 action.  The diagonal
of a `22` costate is a collision face and may be assigned to the separately
synchronized `K4/K31` read.  Anchor collisions `i=j` or `i=k` in a strict
q211 slice must likewise be sewn through those synchronized K31/K4 rows.
What matters is therefore the minimum-rank
**diagonal completion** of the off-diagonal ReLU pair kernel, not the rank
of the literal matrix with its physical diagonal.

Let `a_i` be the standardized bias and let `R` be the preactivation
correlation matrix.  The covariance Jacobian is the joint gate probability

\[
p_{ij}=\Phi_2(a_i,a_j;R_{ij}).
\tag{77.1}
\]

Mehler's formula gives the exact off-diagonal expansion

\[
\boxed{
p_{ij}=\Phi(a_i)\Phi(a_j)
+\phi(a_i)\phi(a_j)
\sum_{q\ge1}{\operatorname{He}_{q-1}(a_i)
                  \operatorname{He}_{q-1}(a_j)\over q!}R_{ij}^{q}.}
\tag{77.2}
\]

If `R=SS^T` has rank `rho`, then

\[
R^{\odot q}=S^{\odot q}(S^{\odot q})^T,
\qquad
\operatorname{rank}R^{\odot q}
\le {\rho+q-1\choose q}.
\tag{77.3}
\]

Here `S^{\odot q}` denotes the rowwise symmetric Khatri--Rao tensor
power, with square-root multinomial weights, not the entrywise scalar power
of `S`.

Consequently, after a lawful diagonal completion, the constant-plus-linear
gate has rank at most

\[
\boxed{k_{gate}^{(1)}\le1+\rho,}
\tag{77.4}
\]

with factors

\[
H^{(1)}=
\left[
\Phi(a),\ \operatorname{Diag}(\phi(a))S
\right].
\tag{77.5}
\]

The order-`Q` completion has the exact algebraic bound

\[
k_{gate}^{(Q)}
\le1+\sum_{q=1}^{Q}{\rho+q-1\choose q},
\tag{77.6}
\]

before dependencies and old projection.  This is an actual low-rank
receipt only when `rho` is the rank of the **effective total gate
correlation at that seam**.  The latent AQ core size `r` does not imply
`rho=r`; the paid `1595`-feature Gram is generically full width.

At zero bias, (77.2) reduces to

\[
p[R]={1\over4}\mathbf1\mathbf1^T
+{1\over2\pi}\arcsin^{\odot}(R).
\tag{77.7}
\]

Modulo its diagonal, the rank-`1+rho` approximation is

\[
p^{(1)}[R]={1\over4}\mathbf1\mathbf1^T+{1\over2\pi}R.
\tag{77.8}
\]

If `chi=max_{i\ne j}|R_ij|`, the exact entrywise remainder obeys

\[
\boxed{
\epsilon_1(\chi)
={\arcsin\chi-\chi\over2\pi},
\qquad
\|\Lambda\odot(p-p^{(1)})\|_F
\le\epsilon_1(\chi)\|\Lambda_{off}\|_F.}
\tag{77.9}
\]

No probabilistic independence is used in (77.9).  For nonzero bias, the
same exact certificate is obtained by taking the maximum off-diagonal
remainder of the truncated series (77.2).  If `R_0` is only a low-rank
approximation to `R`, the additional first-order error is the explicit
Schur term

\[
\operatorname{Diag}(\phi(a))(R-R_0)
\operatorname{Diag}(\phi(a)).
\tag{77.10}
\]

### 77.1 Rank recurrence and the common-frame receipt

Suppose a symmetric incoming pair costate has factor support
`X in R^{n by k}`.  Schur multiplication by the order-one kernel has the
exact factor support

\[
\boxed{
\mathcal V^-=\operatorname{span}
\{x_c\odot h_b:c\le k,\ b\le1+\rho\},}
\tag{77.11}
\]

where the `h_b` are columns of (77.5).  Thus one reset has rank at most
`k(1+rho)`, but repeated resets generically multiply rather than preserve
the rank.  Including the dense reverse leg `D_l^*`, an exact `k_*`-frame
suffix exists iff there are spaces `V_l`, `dim V_l<=k_*`, such that

\[
\boxed{
D_l^*\operatorname{span}
\{v\odot h_{lb}:v\in V_{l+1},b\le1+\rho_l\}
\subseteq V_l}
\tag{77.12}
\]

after the complete-old quotient at every cut.  This is the strict
low-rank common-frame receipt requested by (76.8).

A useful sufficient structure is a transported Schur algebra: each `V_l`
is a block-constant function space, contains the gate factors, is closed
under entrywise products, and the dense leg intertwines `V_{l+1}` into
`V_l`.  Conversely, the diagonal algebra generated by the gate-factor row
signatures separates every pair of coordinates with distinct signatures.
For a vector of full support, its cyclic module then has dimension equal to
the number of distinct signatures; generic dense factors give `n`.
Therefore diffuse entries and low rank at one reset do not prove deep
closure.  A block/common-signature or endpoint-annihilation receipt is
necessary to keep (77.12) small across the suffix.

### 77.2 Exact endpoint leakage gate and current disposition

Let `Lambda_{l+1}` be the exact post-reset pair costate, `E_l` the omitted
kernel in (77.2), and `B_l` the complete-old residual source plane at the
pre-reset seam.  The local scalar error is exactly

\[
e_l=\langle\Lambda_{l+1}\odot E_l,B_l\rangle,
\tag{77.13}
\]

with the robust bound

\[
|e_l|\le
\|\Lambda_{l+1}\odot E_l\|_F\|B_l\|_F.
\tag{77.14}
\]

After inserting the intervening dense/suffix operator norms, the coherent
sum of (77.14) over all live resets must be at most `3.80e-6`.  The
`3.28064%` value-tail figure is not a substitute for this absolute endpoint
receipt.

This theorem reopens the generic full-rank pair-kernel no-go only
conditionally.  The current strict AQ suffix has a generically full-width
total gate Gram and no literal low-rank reverse pair tape.  Hence neither
(77.4) nor (77.12) presently lowers the bill.  The branch becomes real only
with one of:

1. a certified low-rank effective total correlation `R_0` plus (77.9)--
   (77.10);
2. a block/common-signature invariant frame satisfying (77.12); or
3. a measured complete-old endpoint spectrum whose discarded action passes
   (77.13)--(77.14).

### Graph split

- `P2R0G19B_MATH_RELU_PAIR_MEHLER_RANK` — **solved exact**: biased expansion (77.2), rank bounds (77.3)--(77.6), and zero-bias completion (77.7)--(77.9).
- `P2R0G19B_MATH_RELU_PAIR_STRICT_DIAGONAL_QUOTIENT` — **solved exact qualification**: physical full matrix rank is not binding when collision diagonals are separately sewn; minimum completion rank is the relevant object.
- `P2R0G19B_MATH_RELU_PAIR_COMMON_FRAME` — **open strict receipt**: deep low rank requires the Khatri--Rao/dense intertwining condition (77.12), not one-reset diffuseness.
- `P2R0G19B_MATH_RELU_PAIR_ENDPOINT_TAIL` — **open required observable**: coherent error (77.13)--(77.14) must fit `3.80e-6`.
- `P2R0G19B_MATH_RELU_PAIR_CURRENT_AQ_SUFFIX` — **cost-hold**: total gate correlation is generically full width and the declared tape has no resident low-rank reverse pair factors.

## 78. Diffuse common factors do not compress the CP4 K31 face

There is a sharp deterministic witness against using diffuseness to justify
the low-rank loophole (76.8).  Let `U in R^{n by m}` consist of `m` columns
of a normalized Hadamard matrix, so

\[
U^TU=I_m,
\qquad |U_{is}|={1\over\sqrt n}.
\tag{78.1}
\]

Then the Hadamard powers satisfy

\[
V_2={1\over n}\mathbf1\mathbf1_m^T,
\qquad
V_3={1\over n}U.
\tag{78.2}
\]

For the CP4 source (76.1), its collision planes are therefore

\[
\boxed{
K_{31}={1\over n}U\operatorname{Diag}(\theta)U^T,
\qquad
K_{22}={\mathbf1^T\theta\over n^2}\mathbf1\mathbf1^T.}
\tag{78.3}
\]

Thus maximum entrywise diffuseness collapses the `22` face to rank one but
leaves the `31` face with exactly the factor rank.  Its nonzero singular
values are

\[
\boxed{\sigma_s(K_{31})={|\theta_s|\over n}.}
\tag{78.4}
\]

If the residual amplitudes have equal magnitude, every best rank-`k`
approximation to the **full K31 plane** with `k<m` loses the exact
Frobenius-energy fraction

\[
\boxed{{\|K_{31}-K_{31}^{(k)}\|_F^2\over\|K_{31}\|_F^2}
=1-{k\over m}.}
\tag{78.5}
\]

For a width-sized block `m=n=1024`, any endpoint norm which retains this
full-plane Frobenius spectrum would require

\[
\boxed{k\ge\lceil0.9671936\,n\rceil=991,}
\tag{78.6}
\]

before any stronger `3.80e-6` coherent-error requirement.  This is **not**
automatically a lower bound after strict collision quotienting.  If
`m=n` and all `theta_s` have the same sign and magnitude, for example,
`K31` is a scalar identity: its off-diagonal strict face is zero and the
diagonal is sewn through K4.  More generally an arbitrary lawful diagonal
completion can lower the quotient rank.  Therefore (78.6) applies only
with a diagonal-rigidity or endpoint-action receipt proving that the
post-old query retains the spectrum in (78.4).

Under that receipt, the strict equal-rank cost cap `k<=45` from (76.10) is
incompatible with a flat diffuse residual.  Without it, the conclusion is
only that full-plane diffuseness does not imply compression; the
complete-old collision quotient must be analyzed explicitly.

The witness also separates the companion CP3 physics.  Its `K21` plane is
rank one in the same Hadamard model because `V2` has one column direction,
whereas CP4 `K31` remains rank `m`.  Consequently CP3 compressibility is not
evidence for CP4 compressibility, even with identical factor columns.

### Graph split

- `P2R0G19B_MATH_FREE_CP4_DIFFUSE_K31_NOGO` — **solved exact witness**: Hadamard-diffuse factors give rank-one K22 but flat rank-`m` K31, (78.3)--(78.5).
- `P2R0G19B_MATH_FREE_CP4_FLAT_VALUE_COST` — **conditional exact gate**: width1024 requires `k>=991` only when the post-old/diagonal-completed endpoint norm retains the flat K31 spectrum; raw full-plane SVD alone is not a strict-q211 lower bound.
- `P2R0G19B_MATH_FREE_CP4_POSTOLD_CONCENTRATION` — **only remaining low-rank escape**: prove that old projection or endpoint weighting removes/concentrates the K31 singular spectrum; diffuseness alone points the other way.

## 79. Constant-modulus CP4 factors collapse strict q211 to one trace source

The Hadamard witness has a second, favorable side which is invisible in
the K31 spectrum.  For the CP4 tensor (76.1), first form the
collision-completed repeated-anchor representative

\[
\boxed{
(F^c)_{i;(jk)}=\sum_s\theta_s
(u_s\odot u_s)_i\,(u_s)_j(u_s)_k,
\qquad j\ne k.}
\tag{79.1}
\]

Let `D=U^{\odot 2}`, and let `B` have column `s` equal to the strict
off-diagonal pair vector `((u_s)_j(u_s)_k)_{j\ne k}`.  Then

\[
\boxed{
F^c=D\operatorname{Diag}(\theta)B^T,
\qquad
F^{str}=\mathsf M\odot F^c.}
\tag{79.2}
\]

Here `M_{i;(jk)}=1` exactly when `i,j,k` are distinct.  The ordinary
matrix rank of the zero-filled strict flattening can exceed that of the
completed representative because this anchor-dependent mask is not
separable.  After synchronized collision sewing, the relevant completed
rank is

\[
\boxed{
k_{comp}=\operatorname{rank}
P_{old}^{\perp}D\operatorname{Diag}(\theta)B^T.}
\tag{79.3}
\]

This is an algebraic channel rank, not automatically the literal strict
tensor Schmidt rank.  If the future tape does not provide the collision
completion, one must use `rank(P_old^perp F^str)` instead.

In fact the distinction is sharp.  For dense `R` and `n>=3`, the literal
strict trace flattening

\[
(Z_R)_{i;(jk)}=\mathbf1_{\{i\notin\{j,k\}\}}R_{jk}
\tag{79.3a}
\]

has row rank `n`.  If `sum_i alpha_i(Z_R)_i=0`, then for every active pair
`j,k`, writing `S=sum_i alpha_i` gives `S-alpha_j-alpha_k=0`.  All
`alpha_i` are therefore equal to a common `t`, and `nt=2t`; hence `t=0`.
Thus constant modulus yields three different exact notions: literal masked
rank `n`, collision-completed channel rank one, and one target trace dot.

Now assume every factor has constant coordinate magnitude,

\[
(u_s)_i^2=c_s\quad\text{for every }i.
\tag{79.4}
\]

Then `D=1 c^T`; the completed representative and literal strict tensor are

\[
\boxed{
(F^c)_{i;(jk)}=R_{jk},
\qquad
(F^{str})_{i;(jk)}=\mathsf M_{i;(jk)}R_{jk}=(Z_R)_{i;(jk)},
\qquad
R=\sum_s\theta_sc_su_su_s^T.}
\tag{79.5}
\]

Thus a width-sized CP4 tensor can collapse to one **trace-source channel**:
it is exactly the strict insertion `Z_R` of Section 70.  Its ordinary
zero-filled strict flattening need not have rank one, but its local Bellman
action is one pair contraction if `Z^*Lambda` is resident.  This `Z_R`
must not be confused with the six-term full trace tensor `mathcal Z_R` of
Section 71, whose K31 normalization is `3R`.  For normalized
Hadamard columns, `c_s=1/n`, and the same matrix appears on the K31 face,

\[
K_{31}=R.
\tag{79.6}
\]

Equation (79.6) explains both sides of the result.  The full K31 plane can
remain flat rank `m` as in Section 78, while the strict-211 source has one
trace channel because the repeated coordinate has lost all factor
identity.  If the
complete future factors through the K31/pair consumer, the q211 action is
an old pair-supply duplicate by (70.5).  If it does not, the trace source
still needs the target-conditioned trace adjoint and is not automatically
owned.

### 79.1 Exact deviation from the trace collapse

For general factors set

\[
c_s={\|u_s\|_2^2\over n},
\qquad
e_s=u_s\odot u_s-c_s\mathbf1,
\qquad E=[e_1\ \cdots\ e_m].
\tag{79.7}
\]

Then the source splits exactly as

\[
\boxed{
(F^c)=\mathbf1 r^T+E\operatorname{Diag}(\theta)B^T,
\qquad
r=B\operatorname{Diag}(\theta)c,
\qquad
F^{str}=\mathsf M\odot F^c.}
\tag{79.8}
\]

The first term is the one-trace carrier; the second is the complete
constant-modulus defect.  Its exact post-old singular tail is the required
source-rank observable, and a label-free bound is

\[
\boxed{
\|P_{old}^{\perp}E\operatorname{Diag}(\theta)B^T\|_F
\le\|E\operatorname{Diag}(\theta)\|_F\,\|B\|_{op}.}
\tag{79.9}
\]

The same bound remains valid after the strict mask because entrywise
multiplication by `M` is a contraction in Frobenius norm.

For a Bellman query `Q`, the actual endpoint error is the signed contraction
with the defect in (79.8), bounded by `||Q||_F` times (79.9).  Coherent
accumulation over births must remain below `3.80e-6`; a small maximum entry
of `U` does not control `E`, because squared magnitudes may fluctuate at
order one relative to their mean.

### 79.2 Transport gate

The collapse is local, not generically recurrent.  Under a dense leg,
`U` becomes `WU`.  Condition (79.4) survives only if every live vector
`Wu_s` again has constant coordinate magnitude.  For a complete orthogonal
factor basis this means that `WU` is another real flat orthogonal frame;
a generic dense network matrix does not satisfy it.  A correlated ReLU
reset likewise produces factor-dependent squared profiles and reopens
`rank D`.

Therefore the exact cheap architecture is consume-and-retire at a seam
where (79.4) or the small defect (79.9) is certified.  A persistent
rank-one claim additionally needs the trace-intertwining/future-factor
condition (70.2)--(70.5) at every later cut.

### Graph split

- `P2R0G19B_MATH_CP4_Q211_ANCHOR_FLATTENING` — **solved qualification**: (79.2) splits the low-rank collision-completed representative from the possibly high-rank zero-filled strict mask; (79.3) binds only with synchronized sewing.
- `P2R0G19B_MATH_CP4_CONSTANT_MODULUS_TRACE` — **solved exact structured escape**: constant squared profiles collapse arbitrary CP count to one trace-source action, not necessarily Schmidt rank one, (79.4)--(79.6).
- `P2R0G19B_MATH_CP4_TRACE_DEFECT` — **open measurable gate**: certify the post-old spectrum/action of (79.8), with robust bound (79.9) and endpoint budget `3.80e-6`.
- `P2R0G19B_MATH_CP4_TRACE_TRANSPORT` — **generic no-go / structured open**: generic dense/reset maps destroy constant modulus; recurrent use needs a flat-frame intertwiner or consume-and-retire.

## 80. A three-node spectral core removes the finite-instance K31 defect

The Q-only free-Wick branch failed in Section 75.7 because a semicircular
matrix has a nonzero cubic spectral residual `A^3-2A`.  This defect can be
removed exactly by changing the core manifold rather than by carrying a
new K31 plane.

More generally, suppose one live symmetric core obeys a cubic Jacobi
identity

\[
\boxed{A_i^3=a_iA_i^2+b_iA_i+c_iI.}
\tag{80.1}
\]

For traceless `A_j`, the half-gauge K31 face reduces exactly to

\[
\boxed{
\kappa_4(Q_i,Q_i,Q_i,Q_j)
=3a_i\operatorname{tr}(A_i^2A_j)
+6b_iQ_{ij}.}
\tag{80.2}
\]

Thus a Jacobi core needs no cubic matrix word.  Its directed K31 row is a
fixed combination of the synchronized K21/H3 row and the resident
quadratic Gram.  This is zero additional product only when those two rows
are literally owned at the same cut and normalization.  The strict
`48.197` census does not include the deployed CP3/K21 tape, so the general
`a_i\ne0` case remains a typed reuse question.

There is a stronger symmetric specialization which avoids CP3 entirely.
Let

\[
D_i^3=2D_i,
\qquad
\operatorname{spec}(D_i)\subset\{-\sqrt2,0,+\sqrt2\},
\tag{80.3}
\]

with normalized trace weights `1/4,1/2,1/4`, and put `A_i=s_iD_i`.
Then

\[
\tau(D_i)=0,qquad \tau(D_i^2)=1,qquad
\tau(D_i^4)=2,qquad D_i^3=2D_i,
\tag{80.4}
\]

so

\[
\boxed{
\begin{aligned}
\kappa_4(Q_i,Q_i,Q_i,Q_j)&=12s_i^2Q_{ij},\\
\kappa_4(Q_i,Q_i,Q_i,Q_i)&={24q_i^2\over r},
\qquad q_i={1\over2}\operatorname{tr}(A_i^2).
\end{aligned}}
\tag{80.5}
\]

These are exactly the free-Wick Q formulas, but now they hold for the
realized finite matrix on every K4/K31 repeated-anchor face.  The spectral
moments are `mu4=2,mu6=4`, so the scalar K31 projection correlation is one,
not `2/sqrt(5)`.  The symmetric three-node law is the unique centered
three-point law with unit variance and fourth moment two: if its nonzero
atoms are `plus or minus c` with total weight `p`, then `pc^2=1` and
`pc^4=2`, hence `c^2=2,p=1/2`.

Exact finite multiplicities require `r` divisible by four.  This favors
ranks such as `52` or `56`, not `55`; billing and capacity must be redone
for the selected rank rather than rounded from the r55 census.

For the current six-reset formula, the exact r56 arithmetic is

\[
\boxed{
C_{56}^{strict}
=28+{12\cdot1652\over1024}
+{6\cdot56^3\over1024^2}
+{192\cdot56^2\over1024^2}
=48.93848,}
\tag{80.5a}
\]

leaving only `0.06152`, while the terminal-total variant costs

\[
\boxed{
C_{56}^{terminal}=47.32520,}
\tag{80.5b}
\]

leaving `1.67480`.  These figures assume that retraction to the three-node
spectrum is fused into the already-paid eigensystem/K32 local stage and
that no CP3/H4 arrays are appended.  Here `K32` is the quadrature-node count,
not a resident K21/signed-H3 provider; it does not make the general
`a_i\ne0` Jacobi row free.

### 80.1 What this closes, and what it does not

The three-node identity closes the K4/K31 trace defect only.  It does not
determine the mixed ordered word on K22.  Indeed,

\[
\operatorname{tr}(A_iA_jA_iA_j)
=\operatorname{tr}(A_i^2A_j^2)
+{1\over2}\operatorname{tr}[A_i,A_j]^2,
\tag{80.6}
\]

so the q211/q4-perpendicular commutator Gram remains genuine
noncommutative information.  Individual spectral quantization therefore
does not replace the native H211 provider; it removes one expensive and
previously inaccurate collision face, potentially leaving cap for the
target-conditioned commutator action.

Nor is (80.3) stable under generic dense mixing:

\[
A_j^+=\sum_iW_{ij}A_i
\tag{80.7}
\]

does not normally satisfy the same minimal polynomial.  A viable design
must project/refit every live core back to the scaled three-node orbit at
each cut.  Because the AQ reset already diagonalizes its core, eigenvalue
quantization may share that spectral stage, but literal reconstruction,
gradient, and raw-error costs still need a typed receipt.

For an approximate Jacobi core define

\[
R_i=A_i^3-a_iA_i^2-b_iA_i-c_iI.
\tag{80.8}
\]

The omitted K31 plane is exactly

\[
\Delta K_{31,ij}=3\operatorname{tr}(R_iA_j),
\tag{80.9}
\]

and for a synchronized costate `Lambda31` its endpoint action is the signed
contraction with (80.9), bounded by

\[
3\sum_{ij}|\Lambda_{31,ij}|\,\|R_i\|_F\|A_j\|_F.
\tag{80.10}
\]

After the complete-old quotient and suffix weighting, the coherent version
of (80.9), not a local spectral moment average, must be at most `3.80e-6`.

### Graph split

- `P2R0G19B_MATH_AQ_JACOBI_K31_CLOSURE` — **solved exact**: cubic minimal polynomial reduces K31 to K21 plus Q, (80.1)--(80.2).
- `P2R0G19B_MATH_AQ_THREENODE_K31_EXACT` — **solved exact structured design**: the scaled `{-sqrt2,0,+sqrt2}` orbit makes realized K4/K31 exactly Q-owned, (80.3)--(80.5).
- `P2R0G19B_MATH_AQ_THREENODE_RANK_GATE` — **solved conditional census**: r56 costs `48.93848` strict or `47.32520` terminal-total under fused spectral retraction; strict has essentially no hook margin.
- `P2R0G19B_MATH_AQ_THREENODE_RESET_REFIT` — **open implementation/value**: dense mixing leaves the orbit, so every reset needs a jointly trained spectral projection and the residual action (80.9) below `3.80e-6`.
- `P2R0G19B_MATH_AQ_THREENODE_Q211_SCOPE` — **binding limitation**: K22/commutator q211 remains new by (80.6); the design frees a collision word but is not a complete H4 carrier.

## 81. Connected q211 is a threshold jet of exact K3 plus one pair sandwich

There is a second tensor-free route to the *affine-Gaussian comparator*
q211.  It does not carry an H4 tensor; it differentiates the exact connected
third moment with respect to the repeated threshold.

Let

\[
\rho_a=(X_a-\tau_a)_+,
\qquad
x=R_{ij},\quad y=R_{ik},\quad z=R_{jk},
\tag{81.1}
\]

and let `K3(i,j,k)` be the exact connected third cumulant of the three
thresholded Gaussian coordinates.  Gaussian integration by parts gives

\[
\boxed{
\kappa_{211}(i;j,k)
=-\mathcal D_iK_3(i,j,k)+F_{\le2}(i;j,k),}
\tag{81.2}
\]

where

\[
\boxed{
\mathcal D_i=
\tau_i+2m_i+\partial_{\tau_i}
+x\partial_{\tau_j}+y\partial_{\tau_k},
\qquad m_i=E\rho_i,}
\tag{81.3}
\]

and `F_{<=2}` is an explicit sum of one- and two-edge Gaussian functions.
At zero bias this reduces to the simpler form

\[
\boxed{\kappa_{211}=-2\mu K_3+F_{edge}.}
\tag{81.4}
\]

For physical outputs `Y_a=\sigma_a\rho_a`, restore the prefactor
`sigma_i^2 sigma_j sigma_k`; equivalently the K3 term carries one factor
`-sigma_i D_i` after using physical K3.  Equations (81.2)--(81.4) are
static identities of the affine-Gaussian law.  They do not identify a
true-minus-AQ residual.

All terms in `F_edge` reduce to pair/Mehler matvecs and Hadamard reductions
except one genuinely crossed word.  For a CP costate atom

\[
L_{i;jk}=\alpha_i\operatorname{Sym}_{jk}(\beta_j\gamma_k),
\tag{81.5}
\]

that word is

\[
\boxed{
S(\alpha,\beta,\gamma)
=\operatorname{tr}
\{D_\alpha R D_\beta\,p[R]\,D_\gamma R\},}
\tag{81.6}
\]

where `p[R]` is the ReLU pair Jacobian (77.1).  With a dense `p[R]`, (81.6)
is one full sandwich product for each independent `(beta,gamma)` mode and
therefore one GEMM-equivalent per seam; it cannot hide in `0.803` over six
seams.

### 81.1 Exact low-rank pair-kernel fusion

If the synchronized pair kernel has a literal factorization

\[
p[R]=HH^T,
\qquad H\in\mathbb R^{n\times h},
\tag{81.7}
\]

then

\[
\boxed{
S=\sum_{t=1}^{h}\sum_i\alpha_i
\bigl(RD_\beta H\bigr)_{it}
\bigl(RD_\gamma H\bigr)_{it}.}
\tag{81.8}
\]

Thus a generic atom needs two skinny products and a symmetric atom
`beta=gamma` needs one.  For `k` independent modes over `D` seams, the
leading bill is

\[
\boxed{
C_{edge}={Dkh\over n}
\quad(\beta=\gamma),
\qquad
C_{edge}={2Dkh\over n}
\quad(\text{generic}).}
\tag{81.9}
\]

Under the conditional order-one gate rank `h=56` from Section 77 and six
seams, strict slack permits at most two symmetric modes (`0.65625`) or one
generic mode (`0.65625`) before hooks.  With the actual full-width gate
kernel, (81.9) supplies no reduction.

For an approximate factorization `p=p_h+Delta p`, the omitted crossed-word
action is exactly

\[
\Delta S
=\langle\Delta p,
D_\beta R D_\alpha R D_\gamma\rangle,
\tag{81.10}
\]

up to cyclic transpose convention, and hence

\[
|\Delta S|\le
\|\Delta p\|_F
\|D_\beta R D_\alpha R D_\gamma\|_F.
\tag{81.11}
\]

The coherent sum of (81.11), the K3-jet error, and the remaining edge
errors must be below `3.80e-6`.

### 81.2 Ownership and node identity

The identity is useful only if `K3` is exact and its threshold JVP is
available.  A CP3 value does not determine its threshold derivative.
At literal shared Gaussian cubature nodes, the q211 repeated slot replaces
one factor `sum alpha_i Y_i` by `sum alpha_i Y_i^2`; this is reduction-scale
and can fuse with an exact CP3 node evaluation.  Per-unit one-dimensional
AQ resolvent nodes are not a shared trivariate cubature and do not satisfy
this reuse condition.

The deployed D11 CP3 is a finite tree approximation, not the exact Gaussian
`K3`, and the strict `48.197` suffix does not bill that CP3 tape.  Therefore
the present architecture has neither the exact jet owner nor a free
sandwich.  A new row is lawful only under all four receipts:

1. exact same-law K3 value and threshold JVP at the same cut;
2. complete pair-edge complement with synchronized collision sewing;
3. low CP costate rank and pair-kernel factor rank satisfying (81.9); and
4. complete-old subtraction of any billed AQ connected-H4 surface.

If the AQ H4 surface is already billed, (81.2) is merely an alternative
implementation of its q211 block.  It is not additive raw value.  If it is
not billed, (81.2) can upgrade an exact K3+pair state to the missing AQ
comparator H4 action, but the true-minus-AQ residual-slot hierarchy of
Section 60 remains outside.

Finally, (81.2) is an acquisition identity, not a temporal closure theorem.
Avoiding an H211 carrier requires consuming the scalar against the complete
future costate at that seam.  Otherwise the reset/cyclic-rank no-go of
Sections 67 and 69 reappears.

### Graph split

- `P2R0G19B_MATH_AQ_Q211_K3_THRESHOLD_JET` — **solved exact comparator identity**: connected q211 is the exact K3 threshold jet plus pair edges, (81.2)--(81.4).
- `P2R0G19B_MATH_AQ_Q211_EDGE_SANDWICH` — **solved exact action**: the only GEMM-scale edge word is (81.6), with low-pair-rank fusion (81.8)--(81.9).
- `P2R0G19B_MATH_AQ_Q211_K3_NODE_REUSE` — **current no / structured open**: D11 finite-tree CP3 and per-unit AQ nodes are not the exact shared K3+JVP oracle.
- `P2R0G19B_MATH_AQ_Q211_EDGE_COSTATE_RANK` — **open sharp cap**: h56 permits only two symmetric or one generic CP modes in strict slack; actual pair kernel is presently full width.
- `P2R0G19B_MATH_AQ_Q211_JET_NO_DOUBLE_CREDIT` — **binding ownership**: the jet implements AQ comparator q211, not a new feature on top of a billed AQ H4 surface and not true-minus-AQ residual physics.

## 82. Exact local reset module of the three-node AQ core

The spectral design of Section 80 has an additional exact advantage: its
**single-unit affine-quadratic reset** can be evaluated in a finite local
module.  Let

\[
A=\sum_{\alpha=1}^{s}\lambda_\alpha P_\alpha,
\qquad
u_\alpha=P_\alpha u,
\tag{82.1}
\]

with mutually orthogonal spectral projectors.  The tilted Gaussian
resolvent is

\[
(I-itA)^{-1}=\sum_\alpha{P_\alpha\over1-it\lambda_\alpha},
\tag{82.2}
\]

and the exact characteristic function of
`g=c+u^Tz+(z^TAz-tr A)/2` is

\[
\boxed{
\chi_g(t)=e^{it(c-\operatorname{tr}A/2)}
\prod_\alpha(1-it\lambda_\alpha)^{-r_\alpha/2}
\exp\left[-{t^2\over2}
\sum_\alpha{\|u_\alpha\|^2\over1-it\lambda_\alpha}\right].}
\tag{82.3}
\]

Every tilted mean lies in `span{u_alpha}`, and every tilted second moment
lies in

\[
\boxed{
\mathcal M_A=\operatorname{span}
\{P_\alpha,\operatorname{Sym}(u_\alpha u_\beta^T):
\alpha\le\beta\}.}
\tag{82.4}
\]

Therefore the exact lower AQ reset has vector dimension at most `s` and
matrix dimension at most

\[
\boxed{s+{s(s+1)\over2}.}
\tag{82.5}
\]

For the three-node core this is three vector directions and nine matrix
directions.  The result is exact for the local reset; no small-eigenvalue or
diffuseness approximation is involved.

For the symmetric r56 ranks `14/28/14` with standardized pure quadratic
variance, the zero-affine characteristic function is

\[
\boxed{\chi_Q(t)=\left(1+{4t^2\over r}\right)^{-r/8}.}
\tag{82.6}
\]

It matches the desired fourth-order jet, while its first sixth-order mean
jet defect is only order `-2/r^2`.  If the paid eigensystem also owns the
projector reconstruction and the reset needs only the six rank-one outer
writes in (82.4), replacing the nominal `K=32` local term by an effective
`K_eff<=6` gives the conditional census

\[
\boxed{
C_{56,eff}^{strict}\simeq48.472,
\qquad
C_{56,eff}^{terminal}\simeq46.859.}
\tag{82.7}
\]

This is a promising replacement-state cost reduction, but it is conditional
on literal eigensystem/projector liveness and on charging the spectral
retraction inside the paid reset.

### 82.1 Why the nine-dimensional lower state does not yet own H211

A mixed H211 surface differentiates (82.3) in external quadratic
directions `B,C` which generally do not share the projectors of `A`.
Already at `u=0`, the second directional derivative contains both

\[
\boxed{
\operatorname{tr}(P_\alpha B)
\operatorname{tr}(P_\beta C),
\qquad
\operatorname{tr}(P_\alpha B P_\beta C),
\quad 1\le\alpha,\beta\le s.}
\tag{82.8}
\]

The first family is determined by `s` block traces per direction; the
second needs the actual cross-block matrices.  With nonzero `u`, additional
terms have the form

\[
u_\alpha^TB u_\beta,
\qquad
u_\alpha^TBP_\beta C u_\gamma.
\tag{82.9}
\]

These contractions are finite in number for `s=3`, so the local H211
surface is still tensor-free.  But they are not contained in the nine
lower-state coefficients (82.4).  Each target aggregate direction must be
rotated into the gate's eigenframe or supplied by a literal existing
surface tape.  Since different external anchors have different projectors,
single-unit closure does not establish a common D10 reverse frame.

Accordingly, (82.7) cannot yet be credited to a q211 provider.  It becomes
one only if the declared K32/surface work already exposes every block
contraction in (82.8)--(82.9) for the complete-old residual target modes,
with no additional dense direction rotation, and the resulting endpoint
tail fits `3.80e-6`.  Otherwise (82.7) is a lower-reset cost theorem and the
H211 action must be billed separately.

### Graph split

- `P2R0G19B_MATH_AQ_THREENODE_RESET_MODULE` — **solved exact**: local vector/matrix closure (82.1)--(82.5) and scalar characteristic (82.3).
- `P2R0G19B_MATH_AQ_THREENODE_KEFF6` — **conditional cost unlock**: r56 can reach about `48.472/46.859` only with literal projector reconstruction and six-write fusion.
- `P2R0G19B_MATH_AQ_THREENODE_H211_BLOCKS` — **solved required inventory / open reuse**: mixed surface needs block traces/products (82.8)--(82.9), not merely the nine lower-state coefficients.
- `P2R0G19B_MATH_AQ_THREENODE_H211_OWNERSHIP` — **open typed seam**: prove K32/surface tape owns all target block rotations and endpoint tail before crediting q211 at the reduced census.

## 83. Typed audit of the rank-55 physical-row skinny q211 action

The physical q211 comparator has an exact low-rank-edge contraction, but
its advertised product count excludes the Bellman row which selects the
physical outputs.  Let `W` be the dense transition, `w_a^T` its row,
`q_a` the synchronized scalar variance/K4 costate, and let `R,P` be the
source correlation and pair kernel.  The exact physical action is

\[
\boxed{
S=\sum_aq_a\operatorname{tr}(M_aPM_a),
\qquad M_a=D_{w_a}RD_{w_a}.}
\tag{83.1}
\]

If the **literal** source edge has rank `r_R`,

\[
R=UU^T,
\qquad V_a=D_{w_a}U,
\tag{83.2}
\]

then one selected physical row is

\[
\boxed{
c_a=q_a\operatorname{tr}(G_aH_a),
\qquad G_a=V_a^TV_a,
\qquad H_a=V_a^TPV_a.}
\tag{83.3}
\]

Only `PV_a` is a dense skinny product.  Selecting `k` rows over `D` seams
therefore costs

\[
\boxed{C_{row}={Dkr_R\over n}}
\tag{83.4}
\]

before the row costate, masks, sampling error, and hooks.  At `r_R=55` and
ten seams this is `0.5371k`, exactly as claimed by the cubic lane.

### 83.1 The missing owner is `q`, not `W`

The physical rows `w_a` are literal network weights.  The scalars `q_a`
are different: they are components of the complete-future Bellman
covector at the same post-reset/pre-dense seam.  A forward covariance/D10
state does not determine them.  Formally, if `O_l` is the forward state and
`F_{l+}` the future endpoint map, then

\[
q_l=\operatorname{diag}F_{l+}^*\lambda.
\tag{83.5}
\]

Changing the endpoint covector `lambda` leaves every forward state fixed
and changes `q_l`, proving that no forward-only state identity can recover
it.  Exact deep ownership requires either a literal compiled reverse row or
the finite target cyclic-module factorization of Section 67.  A generic
full covariance reverse stream costs the two-sided conjugations which the
row census (83.4) omitted; over ten layers this reopens the roughly
`20`-equivalent reverse word.

At the terminal seam, `q` may be a direct output coefficient and (83.4)
can be literal.  At an internal seam, calling it resident requires the
actual D10 Bellman tape, not a forward D10 plane with the same indices.  The
declared strict `48.197` suffix is forward-only and does not expose (83.5).

### 83.2 The rank-55 premise is also typed

An affine component can have `R=UU^T` with `r_R=55`.  The complete
quadratic-AQ feature Gram has nominal width `55+1540=1595` and generic
physical rank `n`; adding an independent Gaussian-rest covariance is also
full rank.  Since `p[R]` is nonlinear in the **total** correlation, one
cannot replace it by the affine rank-55 component without charging both
the two-edge cross terms and the kernel difference.  For `R=R_0+E`, the
deterministic source error is controlled by

\[
\boxed{
\|P\|_{op}\sum_a|q_a|
\left(2\|D_{w_a}R_0D_{w_a}\|_F\|D_{w_a}ED_{w_a}\|_F
+\|D_{w_a}ED_{w_a}\|_F^2\right),}
\tag{83.6}
\]

plus the separate `p[R]-p[R_0]` action.  It must join the `3.80e-6`
coherent endpoint budget.

Finally, using a few physical rows is a stochastic/frozen sketch unless
the omitted rows are proved dark.  Its unbiasedness is not a fixed-feature
correlation receipt; the exact corpus variance or deterministic endpoint
tail is required.  Thus the current disposition is:

- terminal one-row action: algebraically real if the exact terminal `q`
  and rank-55 affine source are the declared comparator;
- deep ten-seam action: **not resident** in the current tape because (83.5)
  is missing;
- generic quadratic-AQ/rest action: **rank premise false** absent the
  decomposition/error receipt (83.6).

### Graph split

- `P2R0G19B_MATH_Q211_ROW_SKINNY_IDENTITY` — **solved exact**: physical action (83.1)--(83.4).
- `P2R0G19B_MATH_Q211_ROW_BELLMAN_OWNER` — **current no**: internal `q` is a complete-future reverse covector (83.5), absent from forward strict `48.197`; adding it can reopen the 20-equivalent word.
- `P2R0G19B_MATH_Q211_ROW_R55_SCOPE` — **binding limitation**: only an explicitly affine rank55 source qualifies; total quadratic-AQ/rest correlation is generically full width.
- `P2R0G19B_MATH_Q211_ROW_TAIL` — **open required receipt**: low-edge error (83.6) and frozen row-sketch error must pass the complete-old `3.80e-6` gate.

## 84. The three-node orbit closes K31 but has a broad commutator q211 bank

The r56 three-node core removes the finite-instance K31 word, but Haar
orientation does not make its noncommutative H211 sector small.  This can
be quantified exactly at second order.

Let `D` have the symmetric three-node spectrum of Section 80, normalized by

\[
\operatorname{tr}D=0,
\qquad
\operatorname{tr}D^2=r,
\qquad
\operatorname{tr}D^4=2r,
\tag{84.1}
\]

and let `B=ODO^T` for Haar orthogonal `O`.  On the traceless symmetric
space of dimension

\[
d_0={ (r-1)(r+2)\over2},
\tag{84.2}
\]

the orbit covariance is isotropic:

\[
E[\operatorname{vec}B\operatorname{vec}B^T]
={r\over d_0}P_{\operatorname{Sym}_0}.
\tag{84.3}
\]

A direct contraction gives

\[
\boxed{
E\operatorname{tr}(D^2B^2)=r,
\qquad
E\operatorname{tr}(DBDB)=r\eta_r,
\qquad
\eta_r={r-2\over(r-1)(r+2)}.}
\tag{84.4}
\]

Therefore the mean normalized K22 collision is `2+eta_r` rather than the
free value `2`; the relative mean correction is only `eta_r/2` to first
order (`about 0.85%` at r56).  But `eta_r` is an amplitude mean, not a
realized tensor-energy bound.  Indeed,

\[
\boxed{
E\operatorname{tr}[D,B]^2=2r(\eta_r-1),}
\tag{84.5}
\]

so the commutator itself is large.  Equation (84.4) says nothing about the
all-distinct q211 Gram between two different orientations.

### 84.1 Exact commutator covariance spectrum

Fix the repeated anchor `D` and define the skew feature

\[
g(B)=[D,B].
\tag{84.6}
\]

In the eigenbasis of `D`, the covariance eigenvalue on skew coordinate
`(p,q)` is a common positive constant times
`(lambda_p-lambda_q)^2`.  For multiplicities
`r/4,r/2,r/4`, the nonzero spectrum therefore has

\[
\begin{array}{c|c|c}
\text{spectral pair}&\text{multiplicity}&
 (\lambda_p-\lambda_q)^2\\ \hline
(-,0)\text{ or }(0,+)&r^2/4&2\\
(-,+)&r^2/16&8.
\end{array}
\tag{84.7}
\]

Its exact effective rank is

\[
\boxed{
r_{eff}(g)
={ (\operatorname{tr}\Sigma_g)^2\over
    \operatorname{tr}\Sigma_g^2}
={r^2\over5}.}
\tag{84.8}
\]

Equivalently, for independent orientations `B,C`,

\[
{\sqrt{E\langle g(B),g(C)\rangle_F^2}
 \over E\|g(B)\|_F^2}
={\sqrt5\over r},
\tag{84.9}
\]

which is about `3.99%` at r56.  More importantly, the whole bank is broad:
at r56 the high eigenvalue block has dimension `196` and carries `80%` of
the covariance-squared energy, while the lower block has dimension `784`
and carries the remaining `20%`.  A fixed common frame retaining
`96.71936%` of this ensemble energy needs

\[
\boxed{k\ge196+656=852.}
\tag{84.10}
\]

This width is a **one-anchor slice** statement.  If
`A_i=O_iDO_i^T`, define

\[
 h_{i,j}=\operatorname{vec}_{\rm skew}
 \left([D,O_i^TA_jO_i]\right),
 \qquad H_i[j,:]=h_{i,j}^T.
\tag{84.11}
\]

Then the repeated-anchor q211 slice is exactly

\[
 \boxed{T_{i;jk}=-8(H_iH_i^T)_{jk}.}
\tag{84.12}
\]

Thus one fixed slice has an exact pre-collision signed root of width `980`,
and the sharp Haar-energy truncation has width `852`.  But the eigenbasis,
and therefore `H_i`, changes with the repeated anchor.  There is no single
global `H` for all `i` unless a separate common-intertwiner receipt is
proved.  Materializing (84.12) independently for every anchor would be
prohibitive; the lawful target-action escape is instead the common-costate
ordered-product contraction (40.4) or the literal resident-`P` action
(55.1).  Neither (84.7) nor equal spectra imply that receipt.

This is an exact expected fixed-frame lower bound under the Haar orbit, not
a deterministic statement about an adaptively fitted finite family.  It is
nevertheless enough to reject the argument that an isotropic three-node
orbit automatically supplies a `k<=20` or `k<=46` D10 commutator frame.
Only complete-old/endpoint annihilation, a non-Haar structured orbit/design,
or a measured per-instance target tail can reopen it.

Thus the three-node architecture has a clean division of labor:

- K4/K31 trace faces are exactly Q-owned;
- the K22 collision **mean** is close to free by (84.4);
- the identity-specific q211 commutator bank remains the broad carrier
  problem by (84.7)--(84.10).

### Graph split

- `P2R0G19B_MATH_THREENODE_HAAR_K22_MEAN` — **solved exact mean only**: (84.4), relative collision correction about `eta/2`; not a realized tail certificate.
- `P2R0G19B_MATH_THREENODE_COMMUTATOR_SPECTRUM` — **solved exact ensemble covariance**: two-level spectrum (84.7) and effective rank `r^2/5`.
- `P2R0G19B_MATH_THREENODE_COMMUTATOR_LOW_RANK_NOGO` — **closed for a fixed Haar common frame**: r56 needs `k>=852` for the 3.28064% energy tail, far above D10 caps.
- `P2R0G19B_MATH_THREENODE_COMMUTATOR_ENDPOINT_ESCAPE` — **open only after old quotient**: prove a deterministic structured orbit or measured target-weighted annihilation; Haar mean closure is insufficient.

## 85. Weak-correlation Edgeworth audit: K22 is the leading covariance row

The physical-row contraction of Section 83 targets a diagonal
variance/K4 costate.  It is complete only if the downstream Bellman map
annihilates the separate K22/K31 outputs.  At weak correlation, the local
Edgeworth coefficients show that this is a strong condition: K22, not K31
or K4, is the leading off-diagonal covariance correction.

For standardized independent Gaussian coordinates, use the raw connected
cumulant convention

\[
f(x,y)=\phi(x)\phi(y)
\left[1+{\kappa_{22}\over2!2!}H_2(x)H_2(y)
+{\kappa_{31}\over3!1!}H_3(x)H_1(y)+\cdots\right].
\tag{85.1}
\]

The needed half-line Hermite moments are

\[
E[X_+H_2(X)]={1\over\sqrt{2\pi}},
\qquad
E[X_+H_3(X)]=0.
\tag{85.2}
\]

Therefore at `rho=0`, after centering the ReLU outputs,

\[
\boxed{
\delta\operatorname{Cov}(X_+,Y_+)
={\kappa_{22}\over8\pi},
\qquad
\delta\operatorname{Cov}|_{\kappa_{31}}=0,
\qquad
\delta\operatorname{Cov}|_{\kappa_{40}}=0.}
\tag{85.3}
\]

The marginal variance has a different coefficient.  Since

\[
E[X_+H_4(X)]=-{1\over\sqrt{2\pi}},
\qquad
E[X_+^2H_4(X)]=0,
\tag{85.4}
\]

the mean subtraction gives

\[
\boxed{\delta\operatorname{Var}(X_+)
={\kappa_{40}\over24\pi}.}
\tag{85.5}
\]

These constants use raw standardized cumulants.  A deployed plane divided
by factorial or orbit multiplicity must be converted before comparing its
literal coefficient; the `/8pi` and `/24pi` ratio is not optional.

Equation (85.3) has a direct ownership consequence.  A q211 tensor pushed
through a dense `W` emits both diagonal K4 rows and crossed K22 pair rows.
The endpoint-row action (83.1) contracts only the part represented by
scalar diagonal costates `q_a`.  A complete action also needs the pair
costate unless

\[
\boxed{
P_{q211}^{birth}F_{+}^*\lambda
\in\operatorname{range}(J_{diag}^*)}
\tag{85.6}
\]

after the complete-old quotient, where `J_diag` is the diagonal
variance/K4 birth map.  Equivalently, every K22/K31 emission outside that
row must be old-owned or future-dark.  The vanishing K31 coefficient in
(85.3) does not help: the K22 coefficient is already nonzero at zero
correlation and is the leading pair response.

The declared D10/strict AQ schedule has no proof of (85.6).  Hence even a
terminally available `q_a` and an exact rank-55 row sandwich do not by
themselves provide the complete deep q211 correction.  They provide a
diagonal sub-action whose additive value must be measured only after the
missing K22 Bellman row is residualized.

### Graph split

- `P2R0G19B_MATH_EDGEWORTH_K22_LEADING` — **solved exact local constants**: raw standardized `/8pi` K22 covariance and `/24pi` marginal K4 variance, (85.3)--(85.5).
- `P2R0G19B_MATH_Q211_ROW_PAIR_COMPLETION` — **open required ownership**: row PPS is complete only under the diagonal-range condition (85.6); current D10 has no such receipt.
- `P2R0G19B_MATH_Q211_ROW_DIAGONAL_SCOPE` — **binding limitation**: terminal/deep scalar q actions are sub-actions until crossed K22/K31 emissions are old-owned or future-dark.

## 86. Typed ownership audit for the exact K22 three-product circuit

The exact K22 pullback in the cubic lane exposes an important distinction
between an algebraically available product and a product that is literally
resident at the synchronized forward/reverse cut.  For a strict q211 source

\[
 K_{i;jk}=R_{ij}R_{ik}P_{jk}
\tag{86.1}
\]

and a signed spectral factorization of the complete-old K22 costate

\[
 \Lambda=\sum_{t=1}^{r_\Lambda}\lambda_t s_ts_t^{\mathsf T},
\tag{86.2}
\]

the exact scalar circuit uses

\[
 A_t=(W^{\odot2})^{\mathsf T}s_t,
 \qquad B_t=W^{\mathsf T}D_{s_t}W,
 \qquad G_t=R D_{A_t}R,
 \qquad C_t=B_t\odot R
\tag{86.3}
\]

through

\[
 S_{22}=4\sum_t\lambda_t\langle B_t\odot P,G_t\rangle_F
 +8\sum_t\lambda_t\langle P,C_t^{\mathsf T}C_t\rangle_F.
\tag{86.4}
\]

The declared strict-48.197 late-AQ schedule owns none of the three dense
products in (86.3)--(86.4):

1. `B_t` is a **reverse target-diagonal conjugation**.  The schedule's
   forward two-sided D10/rest push has a different middle operand and does
   not become `B_t` merely because it uses the same `W`.
2. `G_t` is a **target-dependent source sandwich**.  A Q24 or covariance
   plane with matching external indices is not the same object unless its
   literal middle diagonal is exactly `D_{A_t}` in the same gauge and at
   the same cut.
3. `C_t^T C_t` is the Gram of a **target-weighted Hadamard bank**.  Neither
   the separated AQ low Gram nor the D10 rest plane exposes this bank or
   its Gram.

Thus the current literal-reuse indicators satisfy

\[
 (\iota_B,\iota_G,\iota_C)=(0,0,0),
 \qquad
 C_{22}=D_s\sum_{t=1}^{r_\Lambda}
 (3-\iota_{B,t}-\iota_{G,t}-\iota_{C,t})
 +C_{\rm elem}+C_{\rm hook}.
\tag{86.5}
\]

For `D_s=10`, one exact costate eigenmode therefore costs thirty full
matrix-product equivalents before hooks.  This is an ownership no-go for
the current forward-only tape, not a lower bound against a redesigned
fused circuit.  A reuse indicator may be set to one only when all five
receipts agree: exact operands, orientation, mask, normalization/gauge,
and lifetime at the synchronized cut.  Sharing dimensions, `W`, or an
external label is insufficient.

The weak-correlation approximation does not change that conclusion by
itself.  If

\[
 \Lambda_{\rm eff}={\mu^2\over4}\,
 \Lambda_C\odot D(R),
 \qquad
 D(R)_{ab}=(1-R_{ab}^2)^{-1/2},
\tag{86.6}
\]

then before strict diagonal masking

\[
 D(R)=\mathbf1\mathbf1^{\mathsf T}+E,
 \qquad
 \|\Lambda_C\odot E\|_F
 \le\left((1-\rho_*^2)^{-1/2}-1\right)\|\Lambda_C\|_F.
\tag{86.7}
\]

The first term inherits the rank of `Lambda_C` only after the collision
diagonal is sewn through the synchronized K4/old row.  Moreover, the
lawful error is the endpoint action, not the unweighted Frobenius tail:

\[
 e_{22}=P_{\rm old}^{\perp}{\cal A}_{22}
 \left[{\mu^2\over4}(\Lambda_C\odot E)\right],
 \qquad
 \|e_{22}\|_{L^2({\rm corpus})}\le3.80\times10^{-6}.
\tag{86.8}
\]

K31 and diagonal K4 emissions must still be old-owned or included.  Hence
the weak-correlation branch is a conditional low-rank screen, while the
present exact three-product branch remains cost-closed.

### Graph split

- `P2R0G19B_MATH_Q211_K22_TYPED_PRODUCTS` — **solved ownership audit**:
  the exact circuit has three target-typed products per live costate mode.
- `P2R0G19B_MATH_Q211_K22_CURRENT_REUSE` — **closed NO** for strict48.197:
  no literal `B_t`, `G_t`, or `C_t^T C_t` is resident.
- `P2R0G19B_MATH_Q211_K22_FUSED_REDESIGN` — **conditional**: reuse requires
  exact operand/orientation/mask/gauge/lifetime identity at one seam.
- `P2R0G19B_MATH_Q211_K22_WEAK_RHO_TAIL` — **open conditional**: sew the
  collision diagonal and certify the endpoint error (86.8), jointly with
  K31/K4 completion.

## 87. Target-action escape for the r56 three-node commutator bank

Section 84 rules out a small autonomous common frame, but it does not rule
out contracting a low-rank **terminal costate** directly.  The anchor must
be aggregated after, not before, its commutator map.

For one external CP costate atom, write the leaf factor as
`sym(beta gamma)` and define the aggregate quadratic cores

\[
 B_\beta=\sum_j\beta_jA_j,
 \qquad
 B_\gamma=\sum_j\gamma_jA_j.
\tag{87.1}
\]

Because `[A_i,A_i]=0`, anchor collisions vanish.  Removing the leaf
collision `j=k` gives the exact identity

\[
\begin{aligned}
 &\sum_{\substack{j,k:\ i,j,k\ {m distinct}}}
 \beta_j\gamma_k
 \langle[A_i,A_j],[A_i,A_k]\rangle_F\\
 &\quad=
 \langle[A_i,B_\beta],[A_i,B_\gamma]\rangle_F
 -\sum_{j\ne i}\beta_j\gamma_j\|[A_i,A_j]\|_F^2.
\end{aligned}
\tag{87.2}
\]

The second line is the synchronized Q22/leaf-collision row.  If that row
is literally old-owned, the novel action needs only the aggregate matrices
in (87.1).  Across all costate atoms at a seam let

\[
 t_l=\dim\operatorname{span}
 \{B_{\beta,c,l},B_{\gamma,c,l}:c\ {m live}\}.
\tag{87.3}
\]

A generic symmetric dyad uses two directions; `beta=gamma` uses one.
This is the target-conditioned rank.  It is unrelated to the `980`
one-anchor source rank in (84.12).

### 87.1 Exact three-block spectral microkernel

For each anchor, let

\[
 A_i=O_iDO_i^T,
 \qquad
 \dim E_- =14,\qquad \dim E_0=28,\qquad \dim E_+=14,
\tag{87.4}
\]

with eigenvalues `-sqrt(2),0,+sqrt(2)` before the unit scale.  For any
aggregate dense `B`, only the `(-,0)`, `(0,+)`, and `(-,+)` blocks of
`O_i^TBO_i` enter `[A_i,B]`.  All exact commutator coordinates can be
formed by

1. multiplying `B` by the `28` extreme eigenvectors `E_- plus E_+`; and
2. projecting the `196+784=980` live scalar block coordinates.

The arithmetic per anchor and aggregate direction is therefore

\[
 28r^2+980r={13\over16}r^3
 \qquad(r=56),
\tag{87.5}
\]

strictly below a generic full `r by r` product.  Aggregate formation costs
only `O(ntr^2)`.  Across width `n=1024` and active seams,

\[
\boxed{
 C_{\rm exact}
 ={13\over16}{r^3\over n^2}\sum_l t_l
 ={4459\over32768}\sum_l t_l
 \simeq0.136078\sum_l t_l.}
\tag{87.6}
\]

This is an action circuit; it never materializes `H_iH_i^T` or a common
980-column root.

### 87.2 Sharp 3.28064%-source-tail microkernel

The one-anchor Haar covariance has squared weights `64` on the `196` high
coordinates and `4` on the `784` low coordinates.  The sharp fixed-frame
tail retains all `196` high coordinates and `656` low coordinates.  The
high block forces `14` extreme right-hand-side columns; those columns also
supply `392` low coordinates.  The remaining `264` low coordinates require
`10` more extreme columns.  Hence the attained arithmetic is

\[
 24r^2+852r={549\over784}r^3,
\tag{87.7}
\]

and

\[
\boxed{
 C_{852}={3843\over32768}\sum_l t_l
 \simeq0.117279\sum_l t_l.}
\tag{87.8}
\]

The omitted source-pair fraction is exactly

\[
 {128\cdot4\over15680}={512\over15680}
 =3.265306\%<3.28064\%,
\tag{87.9}
\]

whereas omitting `129` low coordinates gives `3.290816%` and fails.  This
is sharp for the Haar source metric.  It is not yet an endpoint guarantee:
a non-isometric old quotient/Bellman map can reweight the two blocks, so a
lawful candidate must either use (87.6) or measure/prove the composed
endpoint-weighted tail and still meet the absolute `3.80e-6` budget.

### 87.3 Binding depth and ownership census

The one-seam inequality must not be quoted as a persistent-rank budget.
The bill depends on `sum_l t_l`.  For the terminal-total r56 census
`47.32520`, the nominal headroom `1.67480` permits at most

\[
 \sum_l t_l\le12\quad\hbox{exact},
 \qquad
 \sum_l t_l\le14\quad\hbox{at the 852 source-tail gate},
\tag{87.10}
\]

before hooks, f32 error, and the collision-row cost.  Thus one generic
two-direction dyad across six seams is arithmetically inside the exact
microkernel (`sum t_l=12`) but leaves essentially no unbilled margin.  The
strict r56 census `48.93848` has only `0.06152` headroom and admits no fresh
direction even once.

Finally, (87.2) is new value only if all of the following hold:

- the Q22 leaf-collision term is literally resident/old-owned at the same
  seam;
- the CP costate factors are the complete-future, complete-old-residualized
  factors, not a fitted source decomposition;
- the three-node eigenframes are already owned by the AQ reset and remain
  live for the block projections;
- K31 and diagonal K4 siblings are owned or included; and
- an explicitly billed AQ H4 surface is not also credited for the same
  comparator q211 action.

### Graph split

- `P2R0G19B_MATH_R56_Q211_CP_ACTION_COLLAPSE` — **solved exact**: aggregate
  identity (87.2), conditional on the synchronized collision row.
- `P2R0G19B_MATH_R56_Q211_EXACT_MICROKERNEL` — **solved arithmetic**:
  exact cost (87.6), no autonomous H4 root.
- `P2R0G19B_MATH_R56_Q211_852_MICROKERNEL` — **solved source metric /
  endpoint conditional**: sharp (87.7)--(87.9), requires composed-tail
  certification.
- `P2R0G19B_MATH_R56_Q211_DEPTH_CAP` — **conditional narrow GO**:
  terminal-total admits at most 12 exact direction-seams before hooks;
  strict r56 admits none.
- `P2R0G19B_MATH_R56_Q211_OWNERSHIP` — **open binding receipts**: complete
  CP reverse factors, collision ownership, eigenframe liveness, sibling
  completion, and no AQ-surface duplicate.

## 88. Exact six-ray fallback for one native three-generator H211 atom

The two-ray quotient implementation of a grouped
`Sym(a,a,b,c)` atom is cheap only when every induced collision tensor is
literally owned.  There is an exact collision-independent fallback which
uses six diagonal H4 surface actions and never constructs a quartic tensor.

Let `T` be any symmetric four-linear form and

\[
 p(x)=T(x,x,x,x).
\tag{88.1}
\]

Direct polarization gives

\[
\boxed{\begin{aligned}
48T(a,a,b,c)={}&p(a+b+c)-p(a+b-c)\\
&-p(a-b+c)+p(a-b-c)\\
&-2p(b+c)+2p(b-c).
\end{aligned}}
\tag{88.2}
\]

The first four calls isolate terms odd in both `b` and `c`; the final two
cancel the surviving `T(b,b,b,c)` and `T(b,c,c,c)` terms.  Equation (88.2)
is the full real Waring-rank-six action.  It needs no K4/K31/K22 sibling
read and remains valid after any old-space projection because it is a
linear identity in `T`.

For comparison, the cheaper two-ray identity is

\[
\begin{aligned}
{p(a+b+c)+p(a-b-c)\over24}
={}&T(a,a,b,c)+{p(a)+p(b)+p(c)\over12}\\
&+{1\over2}\{T(a,a,b,b)+T(a,a,c,c)+T(b,b,c,c)\}\\
&+{1\over3}\{T(b,b,b,c)+T(b,c,c,c)\}.
\end{aligned}
\tag{88.3}
\]

Thus a two-ray package is exact only if the complete dense-generator
collision action on the last two lines of (88.3) is resident at the same
cut.  Label-collision marginals, a fitted Star ansatz, or CP4 planes in a
different gauge are not sufficient.  Equation (88.2) is the lawful
fallback whenever this receipt fails.

### 88.1 Relation to a strict q211 CP costate

For a strict repeated-anchor source `K_{i;jk}`, a CP costate with repeated
weight `alpha` and leaf factors `beta,gamma` is one grouped query precisely
when

\[
 \alpha=a\odot a.
\tag{88.4}
\]

An arbitrary real signed `alpha` has the exact two-group decomposition

\[
 \alpha=a_+\odot a_+-a_-\odot a_-,
 \qquad
 (a_+)_i=\sqrt{(\alpha_i)_+},\qquad
 (a_-)_i=\sqrt{(-\alpha_i)_+}.
\tag{88.5}
\]

Hence one arbitrary signed CP anchor costs at most two native grouped
atoms; calling it one group without the signed-square receipt is a rank
error.

Dense transport preserves the group exactly:

\[
 (W^{\otimes4}T)(a,a,b,c)
 =T(W^Ta,W^Ta,W^Tb,W^Tc).
\tag{88.6}
\]

This does **not** prove persistence through a correlated ReLU reset.  The
reverse gate acts through its H2/H4/H6/H8 contraction blocks, and a generic
neuron-specific gate maps one grouped costate to an anchor-dependent span
of growing rank.  Sections 44 and 54 give explicit witnesses.  Therefore
a three-generator package is lawful as

1. a reset-local source consumed at the next synchronized surface;
2. a common-leg dense transport followed by immediate read; or
3. a deeper word only after an exact common-intertwiner/cyclic-rank receipt.

It is not an autonomous ten-layer H4 carrier merely because (88.6) holds.

Finally, a billed AQ H4 surface which supplies all six `p` calls already
owns the AQ comparator action in (88.2).  Native grouped rays may implement
that surface, but are not additive value.  True-minus-AQ novelty requires
the corresponding true/residual surface oracle with the same directions,
normalization, old projection, and seam.

### Graph split

- `P2R0G19B_MATH_H211_SIX_RAY_ACTION` — **solved exact**: collision-free
  polarization (88.2).
- `P2R0G19B_MATH_H211_TWO_RAY_COLLISION_GATE` — **conditional**: (88.3)
  requires full dense-generator sibling ownership at the same cut.
- `P2R0G19B_MATH_H211_SIGNED_ANCHOR_GROUP_RANK` — **solved**: one group for
  a signed-square anchor, at most two for arbitrary real `alpha`, (88.5).
- `P2R0G19B_MATH_H211_GROUP_TRANSPORT` — **solved dense / reset-local only**:
  (88.6) preserves dense transport; generic correlated reset does not.
- `P2R0G19B_MATH_H211_GROUP_SURFACE_OWNERSHIP` — **binding no-double-credit**:
  AQ rays implement a billed AQ surface; residual novelty needs a true or
  residual surface oracle.

## 89. Direct three-Hessian AQ surface for a grouped q211 query

The six-ray identity is universal, but an affine-quadratic source surface
admits a more direct mixed action.  This is a local acquisition theorem,
not a proof that the grouped Bellman costate survives depth.

Let the centered AQ consumer variables be

\[
\begin{aligned}
 U&=u^Tz+\frac12(z^TAz-\operatorname{tr}A),\\
 V&=v^Tz+\frac12(z^TBz-\operatorname{tr}B),\\
 W&=w^Tz+\frac12(z^TCz-\operatorname{tr}C),
\end{aligned}
\tag{89.1}
\]

and define the symmetric pair direction

\[
 \Psi(V,W)
 ={1\over2}(vw^T+wv^T+BC+CB).
\tag{89.2}
\]

Only the derivative partition `(2,1,1)` survives in the fourth Hermite
query of `UVW`: the `(2,2,0)` terms contain one centered undifferentiated
factor and average to zero.  Therefore

\[
\boxed{
 T_4(UVW)=12\operatorname{Sym}\!\left[
 A\otimes\Psi(V,W)+B\otimes\Psi(U,W)
 +C\otimes\Psi(U,V)\right].}
\tag{89.3}
\]

As a normalization check, setting `V=U` gives

\[
\begin{aligned}
T_4(U^2W)=12\operatorname{Sym}\{&
[uw^T+wu^T+AC+CA]\otimes A\\
&+[uu^T+A^2]\otimes C\},
\end{aligned}
\tag{89.4}
\]

which is the previously certified tied K31/H211 formula.

For the scalar AQ reset surface

\[
 {\cal F}(\Sigma)=E_{\Sigma}[f(z)],
\tag{89.5}
\]

Gaussian covariance differentiation gives

\[
 \langle T_4(f),\operatorname{Sym}(M\otimes N)\rangle
 =4D_\Sigma^2{\cal F}[M,N].
\tag{89.6}
\]

Combining (89.3) and (89.6),

\[
\boxed{\begin{aligned}
\langle T_4(f),T_4(UVW)\rangle
=48\{&
D_\Sigma^2{\cal F}[A,\Psi(V,W)]\\
&+D_\Sigma^2{\cal F}[B,\Psi(U,W)]\\
&+D_\Sigma^2{\cal F}[C,\Psi(U,V)]\}.
\end{aligned}}
\tag{89.7}
\]

Thus one grouped q211 query is acquired by three mixed covariance-Hessian
actions.  Equation (89.7) already contains the full dense-generator
collision completion; it is the direct AQ equivalent of the six calls in
(88.2), not an additional feature.

The literal-cost gate remains typed.  The three pair directions require
the exact `BC+CB`, `AC+CA`, and `AB+BA` actions.  They are free only if
the same-seam AQ surface tape accepts those mixed directions using its
resident eigensystem/resolvents or retains equal ordered products.  A
lower-reset eigensystem, a symmetric first gradient, or an unrelated
Jordan product does not by itself prove Hessian-direction ownership.
Per-unit rotations, quadrature nodes, and reverse hooks must be included
in the incremental census.

Under the already-audited same-eigensystem/same-node surface contract, the
conservative exact census for `a` grouped atoms is

\[
 C_{\rm fused}(D,K;q,h,a)
 ={D\over n^2}\{qr^2+Kr(q+h)+Ka\}.
\tag{89.8}
\]

One unshared grouped H211 atom uses three latent directions and the four
required pair bilinears in the resident implementation, so
`q=3,h=4,a=1`.  At `r=49,n=1024,K=32`,

\[
 \boxed{
 C_{\rm fused}=0.24314\quad(D=14),
 \qquad
 C_{\rm fused}=0.17367\quad(D=10)}
\tag{89.9}
\]

per atom before quadrature certification, hooks, memory, and old-source
subtraction.  These numbers require literal reuse of the reset
eigensystem, resolvents, and nodes.  Without reuse add at least

\[
 {Dr^3\over n^2}
 =1.57079\ (D=14),\qquad 1.12199\ (D=10)
\tag{89.10}
\]

per separately factorized family.  Thus local acquisition can fit the r49
headroom for a bounded number of atoms, but it does not solve future
query rank or source acquisition.

The source typing is equally important:

- with `f` equal to the AQ comparator reset, (89.7) owns the AQ q211
  comparator and duplicates any separately billed AQ H4 surface;
- with `f=true-AQ` and an exact tilted residual surface, it owns the
  one-residual true-minus-AQ block;
- it does not identify terms with two or more independent residual slots;
  those require the higher-arity source oracle of Section 60.

Finally, (89.7) consumes a supplied grouped future query.  It does not
construct that query or prove its common-factor recurrence through earlier
correlated resets.

### Graph split

- `P2R0G19B_MATH_H211_THREE_HESSIAN_IDENTITY` — **solved exact**:
  (89.3)--(89.7), collision-complete local action.
- `P2R0G19B_MATH_H211_THREE_HESSIAN_LITERAL_COST` — **conditional typed
  reuse**: arithmetic (89.8)--(89.10) is solved; literal mixed-direction
  product/rotation/resolvent reuse at the same seam remains open.
- `P2R0G19B_MATH_H211_THREE_HESSIAN_SOURCE_ARITY` — **solved scope**:
  AQ comparator or one-residual oracle only; multi-residual q4 excluded.
- `P2R0G19B_MATH_H211_THREE_HESSIAN_FUTURE_QUERY` — **open cyclic receipt**:
  local acquisition does not supply the deep Bellman costate.

## 90. Rank-one covariance costate pulls back to full-rank q211 slices

The weak-correlation K22 costate can be rank one after collision sewing,
but this does not make its pre-transition q211 query a single grouped atom.
The crossed dense pullback expands it generically to full width.

Let the synchronized K22 costate be

\[
 \Lambda=ss^T,
\tag{90.1}
\]

and define

\[
 a=(W^{\odot2})^Ts,
 \qquad
 B=W^TD_sW,
 \qquad
 b_i=B_{i,:}^T.
\tag{90.2}
\]

Specializing the exact physical pullback (112.2) to (90.1), the symmetric
leaf slice for repeated source anchor `i` is

\[
\boxed{
 L_i^{211}=4a_iB+8b_ib_i^T
}
\tag{90.3}
\]

before strict collision deletion and the known common normalization.
The first term comes from

\[
 \sum_{a,b}s_as_bW_{ai}^2W_{bj}W_{bk}
 =a_iB_{jk},
\tag{90.4}
\]

and the crossed term from

\[
 \sum_{a,b}s_as_bW_{ai}W_{bi}W_{aj}W_{bk}
 =B_{ij}B_{ik}.
\tag{90.5}
\]

Every slice range lies in `range(B)`, and the row vectors `b_i` span that
range.  Conversely, any nonzero `a_iB` already has the full range of `B`.
Hence the exact common leaf support is

\[
\boxed{
 \operatorname{rank}[L_1^{211}\ \cdots\ L_n^{211}]
 =\operatorname{rank}B
}
\tag{90.6}
\]

generically, and the equality still follows from the row-outer terms when
the `a_i` happen to cancel.  If `W` is invertible,

\[
 \operatorname{rank}B=\operatorname{rank}D_s
 =|\{a:s_a\ne0\}|.
\tag{90.7}
\]

A dense terminal vector has full support, so a rank-one K22 costate
generically yields common q211 width `n`, not one.  Strict diagonal deletion
can make the literal slice ranks still larger; only a synchronized
complete-old quotient can lower (90.6), and that reduction must be proved
on the actual slices.

### 90.1 Consequence for the commutator source

Let

\[
 G_i(j,k)=\langle[A_i,A_j],[A_i,A_k]\rangle_F,
\tag{90.8}
\]

and define the transported aggregate cores

\[
 C_a=\sum_jW_{aj}A_j,
 \qquad
 N_i=\sum_jB_{ij}A_j.
\tag{90.9}
\]

Then the two pieces of (90.3) have exact actions

\[
\boxed{\begin{aligned}
\langle B,G_i\rangle_F
 &=\sum_a s_a\|[A_i,C_a]\|_F^2,\\
b_i^TG_ib_i
 &=\|[A_i,N_i]\|_F^2.
\end{aligned}}
\tag{90.10}
\]

The first is a broad all-anchor/all-output collision bank.  The second
requires an anchor-dependent aggregate `N_i`; it is not the common
`B_beta` direction of Section 87.

Even if the forward transport has retained all `C_a`, forming the complete
`N` bank is the reverse rectangular pass

\[
 [N_i]_{i=1}^n
 =W^TD_s[C_a]_{a=1}^n,
\tag{90.11}
\]

on `r^2` matrix coordinates.  Its leading bill per seam is

\[
 {n^2r^2\over n^3}={r^2\over n},
\tag{90.12}
\]

followed by `n` local commutator products costing `r^3/n^2`.  At
`r=56,n=1024`,

\[
 {r^2\over n}+{r^3\over n^2}
 =3.0625+0.16748047
 =3.22998047
\tag{90.13}
\]

per seam before the first line of (90.10), hooks, or sibling completion.
Six seams already exceed `19.37` equivalents.  Thus the terminal-total
r56 microkernel of Section 87 is useful only for a genuinely supplied
low-rank **q211** costate.  It cannot be inferred from a rank-one downstream
covariance costate.

The only exact escapes from (90.6)--(90.13) are:

1. `B` has certified low rank/support and is used through the common-factor
   route;
2. the complete-old/future map annihilates the crossed `b_ib_i^T` family;
3. an exact source-specific fused circuit, such as the Price/CP3 circuit
   of Section 112, evaluates (90.3) without materializing the q211 slices;
   or
4. a callable full H4 surface consumes the grouped query before this K22
   pullback is formed.

### Graph split

- `P2R0G19B_MATH_K22_RANK1_TO_Q211_RANK` — **solved exact no-go**:
  common q211 width equals `rank(W^T D_s W)`, (90.3)--(90.7).
- `P2R0G19B_MATH_K22_RANK1_COMMUTATOR_ACTION` — **solved exact split**:
  broad and anchor-dependent actions (90.10).
- `P2R0G19B_MATH_K22_RANK1_RECTANGULAR_BILL` — **closed generic r56**:
  the anchor-dependent branch alone costs at least (90.13) per seam.
- `P2R0G19B_MATH_K22_RANK1_ESCAPE` — **open only with typed structure**:
  low-rank `B`, old/future annihilation, a source-specific fused circuit,
  or pre-pullback full-surface consumption.

## 91. Cross-audit of the leading trace-q211 consume-and-retire feature

The strict trace component is a genuine exception to the generic pulled-
costate rank theorem, but only because its **source tensor** has a fused
identity-pair form.  Algebraic survival, buffer reuse, and incremental value
are three independent gates.

### 91.1 What complete-old collision sewing removes

The exact lineage starts from the leading intrinsic Gaussian surface
defect

\[
 (T_2)_{i;jk}=-\kappa_2R_{ij}R_{ik},
 \qquad
 \kappa_2={4-\pi\over8\pi}>0.
\tag{91.1a}
\]

Since the complete Gaussian pairing and identity-trace lifts obey

\[
 ({\cal P}_R)_{iijk}=R_{jk}+2R_{ij}R_{ik},
 \qquad
 ({\cal Z}_R)_{iijk}=R_{jk},
\tag{91.1b}
\]

the synchronized old quotient satisfies

\[
 \boxed{
 [T_2]_{\rm old\ quotient}
 ={\kappa_2\over2}[{\cal Z}_R]_{\rm old\ quotient}}
\tag{91.1c}
\]

**only when the complete pairing lift is literally old-owned at that cut**.
If the deployed old interface exposes only partial collision planes, project
`T_2` against its actual Jacobian; replacing it by the trace lift is then
unauthorized.

For a standardized correlation matrix `R`, the strict trace representative is

\[
 T_{\rm tr}
 =\Pi_{211}{\cal Z}_I(R)
 ={\cal Z}_I(R)-2{\cal P}_I-E_{31}(3R^\circ),
 \qquad R^\circ=R-I.
\tag{91.1}
\]

The subtraction makes every birth K4/K31/K22 collision face vanish, but it
does not remove the strict q211 entries.  Therefore an old interface
containing only label-collision reads cannot own or annihilate
`T_tr`; the all-distinct trace component survives the birth quotient unless
an old full-H4/trace provider is explicitly present.  Equation (91.1c)
states when this surviving representative is exactly the leading physical
defect rather than merely an algebraic trace feature.

After one dense transition define

\[
 G=WW^T,\qquad C=WRW^T,\qquad\Delta=C-G.
\tag{91.2}
\]

Tensor naturality gives the exact consume-and-retire form

\[
\boxed{
 W^{\otimes4}T_{\rm tr}
 ={\cal Z}_G(\Delta)-{\cal S}_W(3R^\circ).
}
\tag{91.3}
\]

The second term is old only when the literal synchronized Star4 push accepts
the source matrix \(-3R^\circ\).  The first term is the cross-pair collision
word.  It remains new after complete-old projection unless the old pair
interface or a full AQ H4 surface already spans its exact seam action.

Retiring (91.3) after its K4/K31/K22 read is exact if and only if

\[
 \ker{\cal R}_{\rm coll}\subseteq\ker F_{\rm future},
\tag{91.4}
\]

equivalently every downstream q211/q1111 costate component is zero or old-
owned.  Collision equality at the immediate seam does not justify retirement
when a later full-H4 consumer remains.

### 91.2 Why this does not contradict the rank-one K22 no-go

Section 90 concerns representing the **generic q211 costate pulled back**
from `ss^T`; its slices contain

\[
 4a_i(W^TD_sW)+8b_ib_i^T
\tag{91.5}
\]

and have common width `rank(W^T D_s W)=n` generically.  The trace circuit
does not represent those slices.  It contracts one fixed structured source
through the tensor identity (91.3) before the broad costate is materialized.
It is exactly the “source-specific fused circuit” exception in Section 90.

Consequently:

- (91.3) may be cheap even though the generic K22 pullback is full rank;
- it does not certify a reusable low-rank q211 Bellman row; and
- it cannot be extended from the trace source to arbitrary H211 residuals.

### 91.3 Exact `u=1` transient reuse and its liveness contract

The rank-one trace action uses the product

\[
 H(u)=R D_u W^T.
\tag{91.6}
\]

For the leading zero-bias trace orientation `u=1`,

\[
\boxed{
 H(\mathbf1)=RW^T=(WR)^T
}
\tag{91.7}
\]

because `R` is symmetric.  Therefore a schedule which already pays

\[
 C=W R W^T
\tag{91.8}
\]

can expose (91.7) with zero new dense product **if** it forms the left-
associated intermediate `J=WR`, retains `J` until the trace reductions
finish, and permits the transpose view `H=J^T`.

This is a lawful implementation contract for the abstract strict/terminal
AQ schedule; it is not a receipt from current on-disk D11.  The following
typed conditions are all necessary:

1. the operand is the same separated `R`, not merged total covariance;
2. the dense map has the exact orientation in (91.8);
3. the cut, normalization, and pre/post-reset convention match;
4. `J` remains live before the second product/recycling; and
5. all reductions and scratch writes are billed.

For nonconstant `u`,

\[
 H(u)=RW^T+R D_{u-\mathbf1}W^T,
\tag{91.9}
\]

and the second term is a fresh product unless it has its own low-rank,
resident, or endpoint-error receipt.  Near-constant is not exact reuse.

Moreover, the `WR` view does not supply every operand in (91.3).  In
particular it does not by itself prove residency of `G=WW^T`, the Star4
adjoint, or the complete collision costates.  Their liveness/cost gates
remain separate.

The rank-one \(H\) circuit evaluates the all-pair completion.  Turning it
into the strict source additionally removes \(i=j\), \(i=k\), and \(j=k\)
terms through synchronized K31/K22/K4 representatives.  Those removals are
zero-new-product only when the literal lower-label interface is resident at
the same cut; otherwise (91.7) is a cheap incomplete action.

### 91.4 Mixed-Hessian ownership and value

The local three-Hessian identity (89.7) evaluates the full AQ H4 action,
including the trace q211 component.  Hence:

- if an explicit callable AQ H4 surface is billed, (91.3) is a cheaper
  implementation candidate or a consistency check, not additive value;
- without that surface, (91.3) can fill one missing AQ/Price trace action,
  after old Star/pair subtraction;
- it does not acquire true-minus-AQ q211 amplitude unless a separate true
  or residual source oracle supplies that amplitude; and
- a deterministic trace coefficient or zero-new-product receipt says
  nothing about residual raw-MSE gain.

The promotion statistic must therefore be the complete-old-residualized
endpoint action of (91.3), with the same corpus whitening and an acquisition
error no larger than `3.80e-6`.  Cost, liveness, and value must be reported
as separate receipts.

The trace term is also only the declared quadratic source component.  If
`rho=max(|R_ij|,|R_ik|,|R_jk|)` and `epsilon_5(rho)` is the certified
determinant-margin remainder, the omitted higher source obeys

\[
 E_2(\rho)
 ={\rho^3\over\pi^2}+{5\rho^4\over12\pi}
 +\varepsilon_5(\rho).
\tag{91.10}
\]

For a synchronized endpoint costate `L`,

\[
 |\langle L,T_{\ge3}\rangle|
 \le E_2(\rho)
 \sum_{i,j,k\ {\rm distinct}}
 |L_{i;jk}|\,\sigma_i^2\sigma_j\sigma_k.
\tag{91.11}
\]

There is no uniform relative bound when the leading signed action cancels.
The full-physics claim therefore remains on hold until (91.11), source
collisions, reverse error, and finite-repair error all fit the endpoint
margin.

Finally, reducing raw MSE from `2.9409e-8` to `1e-8` with this feature bank
alone requires the frozen old-residualized Schur gain

\[
 \boxed{
 b^TG^\dagger b\ge1.9409\times10^{-8},
 \qquad
 \rho_{\rm old\ quotient}\ge0.8124.}
\tag{91.12}
\]

This is a value gate, not a prediction that the covariance-determined trace
readout will attain it.

### Graph split

- `P2R0G19B_MATH_TRACE211_BIRTH_QUOTIENT` — **solved conditional lineage**:
  collision sewing leaves the strict trace component, and it equals the
  leading defect only under the complete-pairing receipt (91.1c).
- `P2R0G19B_MATH_TRACE211_ONE_STEP_RETIRE` — **conditional exact**:
  transported fusion (91.3) and future factorization gate (91.4).
- `P2R0G19B_MATH_TRACE211_RANK1_K22_RECONCILE` — **solved distinction**:
  source-specific fusion does not imply a low-rank generic pullback.
- `P2R0G19B_MATH_TRACE211_WR_TRANSIENT` — **conditional zero-product**:
  `u=1` gives the transpose view (91.7) only under the five liveness/type
  receipts.
- `P2R0G19B_MATH_TRACE211_SURFACE_DUPLICATE` — **binding ownership**:
  explicit AQ H4 surface already owns this comparator component.
- `P2R0G19B_MATH_TRACE211_VALUE` — **open**: require complete-old endpoint
  gain and the `3.80e-6` acquisition bound; zero incremental GEMM is not a
  value certificate.
- `P2R0G19B_MATH_TRACE211_HIGHER_TAIL` — **open bounded**: certify (91.10)--
  (91.11) on the synchronized source/costate support.
- `P2R0G19B_MATH_TRACE211_RAW1E8` — **open hard gate**: Schur gain and
  old-quotient correlation must meet (91.12).

## 92. Independent-information fallback: the least lawful residual-H4 observable

The covariance and AQ-source audits leave a precise information question.
This section answers it without assuming that a target/label-conditioned
scalar can be deployed.  Such a scalar is useful for diagnosis, but the
estimator must emit all `n=1024` final means from target-free inputs.

### 92.1 Exact information rank after the complete-old quotient

At all live cuts, collect the true-minus-AQ connected-H4 sources in a
finite-dimensional Hilbert space

\[
 {\cal S}=\bigoplus_l{\cal S}_l.
\]

Let

\[
 O:{\cal S}\longrightarrow{\mathbb R}^m
\tag{92.1}
\]

be the stack of every literally owned observation: covariance/Mehler
readouts, AQ connected-H4 surface when billed, synchronized
K4/K31/K22/CP4/Star rows, and any lawful refit tangent.  Let

\[
 C:{\cal S}\longrightarrow{\mathbb R}^n
\tag{92.2}
\]

be the frozen complete-future map from the omitted source to the vector of
final mean corrections.  Put `N=\ker O` and let `P_N` be its orthogonal
projector.  Then the old interface identifies the missing correction
exactly if and only if

\[
\boxed{\ker O\subseteq\ker C
\quad\Longleftrightarrow\quad
P_NC^*=0.}
\tag{92.3}
\]

More generally, the minimum dimension of any additional **target-free**
linear observation `A:{\cal S}\to{\mathbb R}^k` which makes `C`
identifiable is

\[
\boxed{
k_*=\operatorname{rank}(C|_N)
   =\dim\operatorname{range}(P_NC^*).}
\tag{92.4}
\]

Indeed, exact recovery requires
`\ker[O;A]\subseteq\ker C`, equivalently
`\operatorname{range}(P_NC^*)\subseteq
\operatorname{range}(P_NA^*)`.  This proves the lower bound in (92.4);
choosing a basis of `\operatorname{range}(C|_N)` and factoring

\[
 CP_N=H A_N,\qquad
 H\in{\mathbb R}^{n\times k_*},
\tag{92.5}
\]

attains it.

For one frozen scalar endpoint covector `e`, the missing row is merely

\[
 z_e=\langle P_NC^*e,S\rangle.
\tag{92.6}
\]

Thus one scalar is information-theoretically necessary and sufficient for
that **one** endpoint.  But if `e` is the realized residual/label, (92.6)
is label seeded and cannot be an estimator feature.  It also does not emit
the correction vector or its norm.  For a candidate target-free correction
`z=CP_NS`, offline grading still needs

\[
 c=\langle r,z\rangle,\qquad G=\|z\|_2^2,
\qquad \Delta_{\rm opt}={c^2\over G},
\tag{92.7}
\]

where `r` is used only by the frozen evaluation protocol.  A scalar reverse
action gives `c` but neither `z` nor `G`.

At a nondegenerate dense suffix, the 1024 terminal coordinate covectors
remain independent under reverse transport.  Therefore, absent an actual
old-quotient rank receipt,

\[
\boxed{k_*=\min\{n,\dim N\}=1024\quad\hbox{generically}.}
\tag{92.8}
\]

A lawful low-rank escape is a fixed, target-free head
`H\in{\mathbb R}^{n\times k}`.  It emits `H a(S)` and is exact only if
`CP_NS\in\operatorname{range}H` for every lawful source.  On a frozen
corpus the approximate receipt is the endpoint-whitened tail

\[
\boxed{
\eta_{\rm head}^2
=\sum_\nu p_\nu
 \|(I-HH^\dagger)C_\nu P_{N_\nu}S_\nu\|_2^2.}
\tag{92.9}
\]

Choosing `H` from per-instance labels is illegal; a globally trained and
then frozen head is target-free, but must pass (92.9) on untouched nets.

The information obstruction is physical, not peculiar to AQ algebra.
Let `S_G` be standard Gaussian and `S_R` be a standardized Rademacher
variable, and define three distinct labels

\[
 Y_i=S+\epsilon G_i
\]

with independent Gaussian `G_i`.  The two choices of `S` have identical
means and covariance matrices, hence identical covariance/Mehler
two-point inputs, but

\[
 \kappa_4(Y_i,Y_i,Y_j,Y_k)=\kappa_4(S)
 =0\quad\hbox{or}\quad-2
\tag{92.10}
\]

for distinct `i,j,k`.  For a richer old interface, (92.3) is the sharp
replacement for this witness: whenever `P_NC^*\ne0`, sources differing by
a vector in `N` have identical old readouts and different endpoint
corrections.

### 92.2 Exact finite-amplitude signed source from two residual VJPs

There is nevertheless a tensor-free exact acquisition formula once the
missing true residual is itself deployably evaluable.  At one cut couple
the true and AQ outputs on the same latent probability space:

\[
 Y_t=Y_A+tE,\qquad E=Y_T-Y_A,\qquad 0\le t\le1.
\tag{92.11}
\]

Fix the complete-old projector and a target-free head covector `\Lambda`,
and absorb both into the symmetric four-linear cumulant action.  Define

\[
 F(t)=
\left\langle\Lambda,\,
 P_{\rm complete}^{\perp}P_{211}\kappa_4(Y_t)
\right\rangle.
\tag{92.12}
\]

Because joint cumulants are multilinear in their slots, `F` is a polynomial
of degree at most four and

\[
 F'(t)=4\left\langle\Lambda,\,
 P_{\rm complete}^{\perp}P_{211}
 {\cal K}(E,Y_t,Y_t,Y_t)\right\rangle.
\tag{92.13}
\]

The two-node Gauss rule is therefore an **identity**:

\[
\boxed{
F(1)-F(0)
={1\over2}\{F'(t_-)+F'(t_+)\},\qquad
t_\pm={1\over2}\left(1\pm{1\over\sqrt3}\right).}
\tag{92.14}
\]

Equation (92.14) acquires the one-, two-, three-, and four-residual blocks
of (60.2) at finite amplitude.  It has no Taylor remainder.  Two
derivative queries are minimal for a general source: one weighted sample
cannot integrate every cubic `F'` (exactness on `1,t` fixes its node and
weight, after which it fails on `t^2`).

For a diagonal output direction `v`, put
`x_t=v^T(Y_t-\mathbb E Y_t)` and
`e=v^T(E-\mathbb E E)`.  The scalar residual-VJP needed in (92.14) has the explicit
three-accumulator form

\[
\boxed{
j_v(t)=4\left\{
\mathbb E[e\,x_t^3]
-3\mathbb E[e\,x_t]\mathbb E[x_t^2]\right\}.}
\tag{92.15}
\]

Thus a weights-only deterministic source quadrature can compute a signed
directional fourth jet without storing a fourth-order tensor.  A direct
endpoint source surface is still cheaper: if
`p_T(v)=\kappa_4(v^TY_T)` is callable and the AQ value is cached, the single
signed observation `p_T(v)-p_A(v)` already contains the finite secant.

For one grouped `T(a,a,b,c)` query, the collision-independent interface
uses the six rays of (88.2).  Hence:

- a direct signed point-surface needs six residual directional calls;
- the two-node residual-VJP needs twelve calls;
- with literal complete dense-generator sibling ownership, the two-ray
  quotient needs two point calls or four derivative calls; and
- an arbitrary signed repeated-anchor weight may double these counts by
  (88.5).

If each signed directional call has adversarial absolute error
`\epsilon_p`, the six-ray coefficient \(\ell_1\) norm is
`8/48=1/6`.  For grouped weights `\theta_g` the resulting action error is
bounded by

\[
\boxed{
\epsilon_{\rm group}
\le{\epsilon_p\over6}\sum_g|\theta_g|.}
\tag{92.16}
\]

This is a numerical-conditioning receipt, not a source-identification
receipt.

A single AQ-point jet is not finite-amplitude exact.  Its exact omitted
part is

\[
\boxed{
F(1)-F(0)-F'(0)
=6{\cal K}_\Lambda(E,E,Y_A,Y_A)
{}+4{\cal K}_\Lambda(E,E,E,Y_A)
{}+{\cal K}_\Lambda(E,E,E,E).}
\tag{92.17}
\]

Consequently the Section 64 bound is binding: with residual `L^4` size
`e`, comparator size `q`, and endpoint projective weight `w`, a safe
one-jet remainder is

\[
4w(6e^2q^2+4e^3q+e^4).
\tag{92.18}
\]

Covariance or residual `L^2` energy cannot certify (92.18).  If an
approximate residual \(\widetilde E\) is used, multilinear telescoping gives
the additional source-model bound

\[
\boxed{
|Z(E)-Z(\widetilde E)|
\le16w\,\|E-\widetilde E\|_4
\left(q+\max\{\|E\|_4,\|\widetilde E\|_4\}\right)^3.}
\tag{92.19}
\]

This is why `L_4` as a fourth-order Bellman covector and `L^4` as a
residual moment norm must not be conflated: the former supplies the signed
query, while the latter only certifies a remainder.

For the physically important preactivation perturbation

\[
g_a^\epsilon=g_a+\epsilon H_a,\qquad
Y_a^\epsilon=(g_a^\epsilon)_+,
\tag{92.19a}
\]

the first signed source can be written without an abstract residual slot.
Let `I_a=1_{\{g_a>0\}}`, `Z_a=Y_a-\mathbb EY_a`, and define

\[
{\cal W}_3(a,b,c)
=Z_aZ_bZ_c-C_{ab}Z_c-C_{ac}Z_b-C_{bc}Z_a-T_{abc}.
\tag{92.19b}
\]

With \(\bar H_a=\mathbb E[H_a\mid z]\), the exact repeated-anchor
variation is

\[
\boxed{\begin{aligned}
\dot K_{i;jk}={}&
2\mathbb E[\bar H_i I_i{\cal W}_3(i,j,k)]\\
&+\mathbb E[\bar H_j I_j{\cal W}_3(i,i,k)]
+\mathbb E[\bar H_k I_k{\cal W}_3(i,i,j)].
\end{aligned}}
\tag{92.19c}
\]

This is the concrete true-minus-AQ signed statistic derived independently
in the cubic lane.  For a pure third-chaos `H` it contracts a signed
`C^{(3)}` source against AQ gate, delta, delta-prime, and
delta-double-prime surface moments.  A general `H` also has the exact
higher-chaos first-order tail

\[
\sum_{a\in\{i,i,j,k\}}
\|P_{>3}\bar H_a\|_2\,
\|P_{>3}(I_a{\cal W}_{3,-a})\|_2
\tag{92.19d}
\]

with the repeated-anchor multiplicity understood.  Thus a signed H3 tensor
alone is not complete source information.

For finite amplitude, a density/coarea margin gives the independent
nonasymptotic receipt

\[
\boxed{
|K(\epsilon)-K(0)-\epsilon\dot K(0)|
\le{\epsilon^2\over2}B_2(\epsilon),}
\tag{92.19e}
\]

where `B_2` is the endpoint-weighted gate-surface/`L^4` envelope of cubic
Section 97.  The higher-chaos tail (92.19d), the finite secant
(92.19e), sketch/action error, and complete-old projection all add
coherently in (92.24).

Against a fixed CP query, (92.19c) can be fused to one scalar.  Against a
target-free portfolio, the exact required rank is the old-quotient rank of
the signed source/query cross operator, not the number of scalar outputs.
The conservative two-mix plus source/query CountSketch circuit fits only
`k\le22` under its one-equivalent incremental screen; a literal resident
mix can improve that census, but no current tape proves it.  This offers a
real low-query implementation only if the same fixed head passes (92.9);
it does not evade the generic rank-1024 conclusion.

### 92.3 Frozen-source exactness versus self-consistent endpoint response

Equation (92.14) is exact for a fixed complete-old projector and fixed
future covector, or for a declared linear tangent chain which consumes the
local source.  It is not automatically the finite secant of a jointly
recomputed nonlinear suffix.  For a physical parameter/state path
`x(t)`, the exact scalar endpoint response is

\[
\boxed{
\Phi(1)-\Phi(0)
=\int_0^1
\sum_l\left\langle\lambda_l(t),
 {d\over dt}F_l(x_l(t))\right\rangle dt,}
\tag{92.20}
\]

with the path-dependent future adjoint `\lambda_l(t)`.  A fixed
`\lambda_l(0)` omits downstream coadaptation.  Applying the two-node rule
to a general integrand `g(t)` has the deterministic remainder

\[
\left|\int_0^1g(t)dt-\frac{g(t_-)+g(t_+)}2\right|
\le{\sup_t|g^{(4)}(t)|\over4320}.
\tag{92.21}
\]

Thus a self-consistent claim needs either the fifth endpoint derivative
receipt implicit in (92.21), a certified higher-node quadrature, or the
direct target-free endpoint hybrid vector

\[
\boxed{z_{\rm hybrid}=\Phi_{\rm true}-\Phi_{\rm AQ}.}
\tag{92.22}
\]

The hybrid is the least exact target-free observable at finite amplitude.
It is useful only if its `true` branch is itself a lawful weights-only
computation; an exported truth/label measurement cannot be packaged.

### 92.4 Coherent endpoint error gate

For frozen net `\nu`, let `\epsilon_{l\nu,\pm}` bound the two signed VJP
errors after the actual head/costate contraction, and let
`\rho_{l\nu}` contain (92.19), (92.21), projector mismatch, and floating
point error.  A sufficient adversarial bound is

\[
\eta_\nu\le
\eta_{{\rm head},\nu}
+\sum_l\left\{
{\epsilon_{l\nu,-}+\epsilon_{l\nu,+}\over2}
+\rho_{l\nu}\right\}.
\tag{92.23}
\]

The binding corpus gate is

\[
\boxed{
\left(\sum_\nu p_\nu\eta_\nu^2\right)^{1/2}
\le3.80\times10^{-6}.}
\tag{92.24}
\]

For ten equally weighted, coherently aligned layers with exact quartic
paths and equal per-node errors, (92.24) requires at most
`3.80e-7` per contracted node.  For fourteen layers it requires
`2.714e-7`.  A single fused vector oracle which aggregates every layer
before rounding instead receives the full `3.80e-6` allowance.  The looser
`2.73e-5` energy slack is unavailable without endpoint orthogonality.

### 92.5 Acquisition cost and the target-free closure

Under the established census, one independent propagated H4 reverse head
costs at least `2D` dense-GEMM equivalents.  A rank-`k` target-free head
therefore costs

\[
\boxed{C_{\rm reverse}\ge2Dk.}
\tag{92.25}
\]

At width `n=1024=2^{10}`, one GEMM equivalent is `n^3=2^{30}` scalar
multiply-accumulates in the present normalization.  Even granting the
entire physical `2^{41}` allowance gives only 2048 equivalents, hence

\[
\begin{array}{c|cc}
&D=10&D=14\\
\hline
k_{\max}\text{ under }2^{41}&102&73\\
k_{\max}\text{ under the 49-eq screen}&2&1.
\end{array}
\tag{92.26}
\]

The generic `k=1024` family instead costs

\[
\boxed{
20{,}480\ {\rm eq}\simeq2^{44.322}\ (D=10),\qquad
28{,}672\ {\rm eq}\simeq2^{44.807}\ (D=14),}
\tag{92.27}
\]

respectively ten and fourteen times the whole `2^{41}` physical allowance.
The actual r49 and late-suffix residual headrooms are smaller still.

The two-node source acquisition is cheaper than the reverse family but is
not free.  If `q_{\rm src}` residual jet directions fuse literally into
resident node/eigensystem work, its optimistic floor is

\[
\boxed{
C_{\rm src,2J}^{\rm fused}
\ge {2Dq_{\rm src}r^2\over n^2};
\qquad
C_{\rm src,2J}^{\rm unfused}
\ge {2Dq_{\rm src}r^3\over n^2}.}
\tag{92.28}
\]

At r49 this is `0.064114q_src` / `3.14158q_src` for D14 and
`0.04580q_src` / `2.24398q_src` for D10.  If the existing primitives must
also synthesize one grouped future query at both nodes, the optimistic
combined counts are about

\[
0.55039\quad(D=14),\qquad0.39314\quad(D=10)
\tag{92.29}
\]

per group and one residual family, before collision sewing, reverse-head
construction, hooks, memory, and error certification.  A literal fused
residual-VJP may improve (92.29), but only an operation-and-buffer receipt
earns that credit.

For `q_src=1024`, the fused source floors alone are `65.65` equivalents
at D14 and `46.90` at D10.  They fit the coarse `2^{41}` physical
allowance but fail or nearly exhaust the 49-equivalent screen; the generic
reverse family (92.27), not the three scalar accumulators, is the decisive
physical obstruction.

There is one exact dual bypass.  If a complete, target-free source
direction `h` is known in advance, one forward JVP emits the whole vector
`Jh` in one pass; a finite endpoint hybrid similarly differences one base
and one perturbed suffix.  A mean-only tangent is small, while a
covariance/AQ tangent generically carries an `O(n^2)` state and costs about
two dense products per crossed transition.  Thus a **known** closed source
can cost only `2D` rather than `2Dk`.

That bypass does not identify `h`.  The true-minus-AQ H4 source is not in
the current covariance/AQ state, and the preceding rank/reset theorems show
that its generic forward carrier is not closed.  Folding a source into the
base AQ state is lawful only after it is selected from target-free
white-box observables; selecting it from the residual/label scalar (92.6)
is leakage.

Therefore the present interfaces have a decisive disposition:

1. a label-seeded scalar residual-VJP is an offline value diagnostic only;
2. the generic 1024-output reverse family fails even the `2^{41}` physical
   allowance;
3. a low-rank fixed head is viable only with a measured
   endpoint-whitened tail satisfying (92.9) and total error (92.24); and
4. absent such a head, the least additional lawful measurement is the
   full target-free endpoint correction vector (92.22), supplied by a
   weights-only endpoint hybrid or by a proved closed forward JVP.

Current D11 and the declared AQ suffix have neither a true-residual source
oracle nor a resident complete-future L4 reverse head.  The route is
therefore **closed for packaging as currently typed**, while remaining a
precise two-query diagnostic and a conditional reopen if (92.9),
(92.24), and the literal source/tape bills are supplied.

### Graph split

- `P2R0G19B_MATH_RESIDUAL_H4_INFORMATION_RANK` — **solved exact**:
  old-interface sufficiency (92.3) and minimum target-free observation
  rank (92.4).
- `P2R0G19B_MATH_RESIDUAL_H4_SCALAR_LABEL_LEAK` — **closed for deploy**:
  one frozen scalar is sufficient only for one label-seeded endpoint and
  does not emit the 1024-vector or its Gram.
- `P2R0G19B_MATH_RESIDUAL_H4_TWO_NODE_SECANT` — **solved exact frozen
  source**: two residual VJPs at (92.14) acquire every residual-slot order.
- `P2R0G19B_MATH_RESIDUAL_H4_ONE_JET_REMAINDER` — **solved bound / closed
  from covariance alone**: (92.17)--(92.19) require residual `L^4`
  information.
- `P2R0G19B_MATH_RESIDUAL_H4_SELFCONSISTENT_SECANT` — **conditional**:
  nonlinear downstream coadaptation requires (92.20)--(92.22), not the
  frozen two-node identity alone.
- `P2R0G19B_MATH_RESIDUAL_H4_TARGET_FREE_HEAD` — **open sharp receipt**:
  fixed-head rank/tail (92.8)--(92.9) and coherent error (92.24).
- `P2R0G19B_MATH_RESIDUAL_H4_PHYSICAL_RANK_NOGO` — **closed generic
  reverse route**: `k=1024` violates both the physical `2^{41}` allowance
  and the 49-equivalent screen, (92.25)--(92.27).
- `P2R0G19B_MATH_RESIDUAL_H4_MIN_VECTOR_ORACLE` — **least lawful reopen**:
  a weights-only full endpoint hybrid vector or proved closed forward JVP;
  neither is resident in the current package.

## 93. Primitive H5 / fifth-cumulant provider: exact AQ law, arbitrary-bias ReLU action, and the actual missing shell

### 93.1 Disposition first

The cheap part of the fifth-order problem is not new.  The complete
support-at-most-two shell

\[
[5]+[41]+[32]
\quad\longleftrightarrow\quad
(k5d,K41,K32)
\tag{93.1}
\]

is already implemented and parity-checked in `public_exact_ledger_v1`.  Its
affine action includes the exact multiplicities `1,5,10`, and its bivariate
reset consumes the same fifth-order Edgeworth channels as every other pair
moment.  The autonomous Level-D carry was unstable; relabelling it as a new
H5 provider would repeat a closed experiment.

The first genuinely unledgered shell is

\[
\boxed{[311]+[221].}
\tag{93.2}
\]

It is independent of the marginal K5 ladder, `K41/K32`, and K22.  The prior
`p2r0f3b1b_output_specific_joint_k5_audit_v1` gives an exact fixture on which
all those old fields vanish while a `[311]` action is nonzero.  It also proves
that the generic dense three-distinct contraction has treewidth three and
exceeds the complete physical budget.  Thus the only useful outcome of the
present derivation is either

1. an exact low-factor-rank, target-free fused `[311]+[221]` action, or
2. a directly supplied complete-old-residualized 1024-vector action.

Everything below separates the exact AQ **comparator**, the cheap old
bivariate consumer, and this true new source.

### 93.2 Exact fifth cumulants of an affine-quadratic Gaussian core

Use the half-quadratic convention

\[
g_i=u_i^Tz+\frac12(z^TM_i z-\operatorname{tr}M_i),
\qquad z\sim N(0,I),\quad M_i=M_i^T.
\tag{93.3}
\]

For sources `t_i`, put `A=sum_i t_i M_i` and `q=sum_i t_i u_i`.  The centered
joint cumulant generating function is exactly

\[
K(t)=-\frac12\log\det(I-A)+\frac12q^T(I-A)^{-1}q.
\tag{93.4}
\]

Consequently the complete polarized AQ fifth cumulant is

\[
\boxed{
\kappa_5^{AQ}(i_1,\ldots,i_5)
=\frac1{10}\sum_{\pi\in S_5}
  \operatorname{tr}(M_{i_{\pi_1}}\cdots M_{i_{\pi_5}})
+\frac12\sum_{\pi\in S_5}
  u_{i_{\pi_1}}^T M_{i_{\pi_2}}M_{i_{\pi_3}}M_{i_{\pi_4}}
  u_{i_{\pi_5}}.}
\tag{93.5}
\]

This formula includes every collision class, including (93.2).  Its marginal
specialization is

\[
\boxed{\kappa_5^{AQ}(g_i)=12\operatorname{tr}(M_i^5)
             +60u_i^TM_i^3u_i.}
\tag{93.6}
\]

Writing `M=M_i,N=M_j,u=u_i,v=u_j`, coefficient extraction from (93.4) gives
the two old bivariate planes exactly:

\[
\boxed{
\begin{aligned}
K41_{ij}^{AQ}={}&12\operatorname{tr}(M^4N)
 +24u^TM^2Nu+12u^TMNMu+24u^TM^3v,\\
K32_{ij}^{AQ}={}&6\operatorname{tr}(M^3N^2)
 +6\operatorname{tr}(M^2NMN)\\
&+12u^TMN^2u+6u^TNMNu+6v^TM^3v\\
&+12u^T(M^2N+MNM+NM^2)v.
\end{aligned}}
\tag{93.7}
\]

Setting `(N,v)=(M,u)` in either line recovers (93.6), fixing all constants.
Bias constants do not change cumulants of order at least two.

Equations (93.5)--(93.7) are an exact comparator/source formula only when the
actual `(u_i,M_i)` AQ state is present.  Literal D11 does not store those
matrices or their eigensystems.  Its mean, covariance, K3/K4, CP3/CP4, star,
and pair fields do not identify (93.5): two laws can agree through fourth
order and have arbitrary different fifth cumulants.  Therefore these formulas
do not turn covariance into a free H5 provider.

### 93.3 Normal-order ownership: what is old and what is primitive H5

For a symmetric cubic tensor `T`, let `C_T=I_3(T)/6` and
`Q_M=(z^TMz-tr M)/2`.  The exact mixed quadratic/cubic product is

\[
\boxed{
Q_M\diamond C_T
=Q_MC_T-3C_{\operatorname{sym}(M\cdot_1T)}
 -\frac12(M\cdot_2T)\cdot z
=\frac1{12}I_5(\operatorname{sym}M\otimes T).}
\tag{93.8}
\]

The subtracted H3 term belongs to CP3/Q24 and the H1 term belongs to the
mean/linear path.  Only the normal-ordered right side may receive new H5
credit.  Likewise, the pair-density terms

\[
\Lambda_3\Lambda_4+\frac16\Lambda_3^3
\tag{93.9}
\]

are deterministic products of the already carried K3/K4 fields.  They are
not a fifth-cumulant source.  The complete third tilt shell is

\[
\boxed{\Lambda_5+\Lambda_3\Lambda_4+\frac16\Lambda_3^3,}
\tag{93.10}
\]

and `public_trivariate_derivation_v1` already showed that deploying (93.9)
without `Lambda5` worsens the current host by 5--11% because the two pieces
nearly cancel.  Any H5 proposal must therefore subtract (93.9) and every
same-cut CP descendant exactly once.

The Gaussian-deformation identity gives a tensor-free **AQ comparator** action.
For

\[
U=u^Tz+\tfrac12(z^TAz-\operatorname{tr}A),\qquad
V=v^Tz+\tfrac12(z^TBz-\operatorname{tr}B),
\]

the exact query tensors are

\[
T_5(U^3)=90\operatorname{Sym}(u\otimes A\otimes A),
\tag{93.11}
\]

\[
T_5(U^2V)=30\operatorname{Sym}(v\otimes A\otimes A)
          +60\operatorname{Sym}(u\otimes A\otimes B).
\tag{93.12}
\]

If `F(mu,Sigma)` is the exact AQ ReLU mean surface, then

\[
\boxed{
\begin{aligned}
\langle T_5(F),T_5(U^3)/90\rangle
 &=4D_\mu D_\Sigma^2F[u;A,A],\\
\langle T_5(F),T_5(U^2V)\rangle
 &=120D_\mu D_\Sigma^2F[v;A,A]
  +240D_\mu D_\Sigma^2F[u;A,B].
\end{aligned}}
\tag{93.13}
\]

These derivatives reuse a resident AQ determinant/resolvent and cost `O(r^3)`
per fixed source/query family.  They are not resident in D11, and, even in a
full-AQ replacement, they own only the AQ comparator.  The genuine innovation
is

\[
\boxed{
R_{5,l}=P_{\rm complete\ old,l}^{\perp}
       \bigl(K_{5,l}^{true}-K_{5,l}^{AQ}\bigr),}
\tag{93.14}
\]

including its `[311]+[221]` part.  No fourth-order field determines (93.14).

### 93.4 Arbitrary-bias univariate ReLU action

Let `X=(mu+sigma Z)_+`, `alpha=mu/sigma`, and `w=-alpha`.  Define

\[
c_{p,m}=E[X^p\operatorname{He}_m(Z)],\qquad R_p=E[X^p].
\]

Repeated Gaussian integration by parts gives the exact all-bias table

\[
\boxed{
c_{p,m}=
\begin{cases}
\dfrac{p!}{(p-m)!}\sigma^mR_{p-m},&m\le p,\\[5pt]
p!\sigma^p\phi(w)\operatorname{He}_{m-p-1}(w),&m\ge p+1.
\end{cases}}
\tag{93.15}
\]

In particular,

\[
\begin{array}{c|c}
p&c_{p,5}\\ \hline
1&\sigma\phi(\alpha)\operatorname{He}_3(-\alpha)\\
2&2\sigma^2\phi(\alpha)\operatorname{He}_2(-\alpha)\\
3&6\sigma^3\phi(\alpha)\operatorname{He}_1(-\alpha)\\
4&24\sigma^4\phi(\alpha)\\
5&120\sigma^5\Phi(\alpha).
\end{array}
\tag{93.16}
\]

Hence an incoming marginal fifth cumulant changes the raw moment by

\[
\boxed{\delta R_p={\kappa_5\over120\sigma^5}c_{p,5}.}
\tag{93.17}
\]

The mean action, important for sign/parity checks, is

\[
\boxed{
\delta E[X]
={\kappa_5\over120\sigma^4}
 \phi(\alpha)\operatorname{He}_3(-\alpha)
={\kappa_5\over120\sigma^4}\phi(\alpha)\alpha(3-\alpha^2).}
\tag{93.18}
\]

It vanishes at zero bias.  This is why a marginal K5 carrier can be tiny even
when `kappa5` itself is not.

For completeness, the Gaussian raw fifth moment is

\[
R_5=\mu(\mu^4+10\mu^2\sigma^2+15\sigma^4)\Phi(\alpha)
 +\sigma(\mu^4+9\mu^2\sigma^2+8\sigma^4)\phi(\alpha),
\tag{93.19}
\]

and the corresponding post-ReLU fifth cumulant is

\[
\boxed{
\begin{aligned}
\kappa_5(X)={}&R_5-5R_4R_1-10R_3R_2+20R_3R_1^2\\
&+30R_2^2R_1-60R_2R_1^3+24R_1^5.
\end{aligned}}
\tag{93.20}
\]

This is the exact univariate source already used by the old marginal probe.

### 93.5 Arbitrary-bias bivariate `Lambda5` action

For a Gaussian comparator pair with correlation `rho`, let

\[
\gamma_{ab,ij}={\kappa_{ab,ij}\over\sigma_i^a\sigma_j^b},
\qquad a+b=5.
\]

The exact linear `Lambda5` contribution to any pair raw moment is

\[
\boxed{
\delta M^{pq}_{ij}
=\sum_{a+b=5}{\gamma_{ab,ij}\over a!b!}
 \sum_{t\ge0}{\rho_{ij}^t\over t!}
 c^{(i)}_{p,a+t}c^{(j)}_{q,b+t}.}
\tag{93.21}
\]

This is the ordinary cumulant differential operator followed by Mehler/Stein;
it is valid at arbitrary biases.  A finite D11 Mehler table can consume the
same formula through its declared truncation with only elementwise `O(n^2)`
work once the six fifth-order planes are supplied.  Calling that truncated
consumer physically exact additionally requires its usual Mehler-tail receipt.

At `rho=0`, the covariance action exposes the parity distinction directly:

\[
\delta M^{11}_{ij}[K41]
={K41_{ij}\over24\sigma_i^3}
 \phi(\alpha_i)(\alpha_i^2-1)\Phi(\alpha_j),
\tag{93.22}
\]

\[
\delta M^{11}_{ij}[K32]
=-{K32_{ij}\over12\sigma_i^2\sigma_j}
 \alpha_i\phi(\alpha_i)\phi(\alpha_j).
\tag{93.23}
\]

Thus K41 survives at a centered independent boundary whereas K32 vanishes
there but is generally nonzero at bias.  Off-diagonal K41/K32 do not alter
either marginal mean, so their covariance contribution equals their raw
`M11` contribution; marginal K50/K05 must use the ordinary centering
correction.

### 93.6 Cheapest literal full-output circuits

If a reset supplies only a diagonal primitive fifth source `h_i`, its next
affine support-at-most-two fields are

\[
\boxed{
\begin{aligned}
k5d'&=(W^{\odot5})^Th,\\
K41'&=(W^{\odot4}\odot h)^TW,\\
K32'&=(W^{\odot3}\odot h)^TW^{\odot2},
\end{aligned}}
\tag{93.24}
\]

with diagonals sewn into `k5d'`.  Powers are elementwise.  The marginal costs
`O(n^2)`; the two planes cost two dense products per source transition.  This
is exactly the old ledgered spelling, not a new provider.  Over a D10 suffix
its product floor is `20` GEMM equivalents, already beyond the `.803` strict
r55 headroom, the `2.360` terminal-total headroom, and the `8.98` r49
incremental surface allowance.  It fits the coarse physical cap only as a
replacement whose removed old work is explicitly rebilled.

There is a genuine low-factor-rank loophole for (93.2).  Suppose a strict
`[311]` atom has separated coefficient `a_i b_j c_k`.  For one output column
`x=W_:,o`, its all-distinct contraction is exactly

\[
\begin{aligned}
S_{311}(x)={}&A_3B_1C_1-(AB)_4C_1-(AC)_4B_1\\
&-A_3(BC)_2+2(ABC)_5,
\end{aligned}
\tag{93.25}
\]

where `A_p=sum_i a_i x_i^p`, `(AB)_p=sum_i a_i b_i x_i^p`, and similarly.
For `[221]`,

\[
S_{221}(x)=A_2B_2C_1-(AB)_4C_1-(AC)_3B_2
           -A_2(BC)_3+2(ABC)_5.
\tag{93.26}
\]

These are inclusion--exclusion identities, so they handle every source-index
collision exactly.  Batched over all 1024 outputs, one strict atom needs seven
skinny products by `W^{odot p}`.  If a literal complete-old collision interface
already supplies the subtractions, the collision-completed grouped atom needs
only the three main factor pushes.

For total grouped rank `q=q311+q221` across D active transitions, the optimistic
incremental floors are therefore

\[
\boxed{C_{grouped}\ge {3Dq\over n}\quad\text{(complete collision CSE)},
\qquad
C_{strict}\ge {7Dq\over n}\quad\text{(self-sewn)}.}
\tag{93.27}
\]

At `n=1024,D=10`, the `.803` strict-r55 slack permits at most `q=27` / `q=11`;
the `2.360` terminal-total slack permits at most `q=80` / `q=34`.  These are
arithmetic ceilings before source-factor acquisition, complete-state response,
hooks, memory, and error.  A factorization fitted to labels is illegal; the
factors and rank must be emitted target-free from the white-box state.

The generic source does not satisfy this receipt.  The prior audit proves a
generic action rank `n`, a dense treewidth-three contraction cost
`32,888,774,983,680` FLOPs over 15 layers (about `14.96` complete `2^41`
budgets), and no current trivariate covariance-resummed ReLU birth.  Therefore
(93.27) is a precise conditional reopen, not evidence that such factors exist.

### 93.7 Measured receipts and the rank-35 fail-fast sentinel

The rank-35 `K5_MEASURED_RESULT.json` already falsifies exact recovery from the
old closure `5 g3 g4-(40/9)g3^3`.  On two million samples the masked median
`measured/closure` ratios are

\[
0.716\pm0.0359\ (L8),\qquad
0.768\pm0.0273\ (L11),\qquad
0.7638\pm0.0241\ (L14).
\tag{93.28}
\]

The stability of a roughly three-quarter ratio is not an exact law and does
not authorize a fitted rescaling.  The independent marginal-Hermite audit also
found K5/K7/K9 marginal sources 25--50 times too small with absolute
correlation at most `.114`.  The support-at-most-two Level-D exact recursion
then became unstable.  Together these receipts close another marginal or
closure-only probe.

A lawful rank-35 sentinel for a genuinely new provider must instead freeze one
natural target-free object before any target read:

\[
\boxed{
b_{5,l}=\mathcal A_{311,l}(R_{311,l})
       +\mathcal A_{221,l}(R_{221,l}),}
\tag{93.29}
\]

or, still better, the already complete-state-propagated endpoint vector

\[
\boxed{z_5=\sum_l\Phi_{L\leftarrow l}E_{5,l}R_{5,l}\in\mathbb R^{1024}.}
\tag{93.30}
\]

It must pass: exact small-width dense parity including all multiplicities;
zero response on CP/old fixtures; nonzero `[311]` or `[221]` fixture with
`k5d=K41=K32=K22=0`; frozen rank/factor bill; finiteness; and a complete-old
projection in the same cut/gauge.  A scalar residual VJP is not a package
observable; (93.29) or (93.30) is target-free and emits all outputs.

For baseline raw `R0=2.9409e-8`, a 20% raw fire move is
`Delta20=5.8818e-9`.  Two distinct amplitude gates should not be conflated:

\[
\boxed{
\begin{aligned}
\text{necessary natural-vector move: }&
\|z_5\|_{RMS}\ge\sqrt{R_0}-\sqrt{R_0-\Delta_{20}}
\simeq1.81\times10^{-5},\\
\text{old-orthogonalized Schur capture: }&
{\langle e,z_5^\perp\rangle^2\over\|z_5^\perp\|^2}
\ge5.8818\times10^{-9},
\end{aligned}}
\tag{93.31}
\]

the latter corresponding to captured residual RMS `sqrt(Delta20)=7.67e-5`.
Raw below `1e-8` requires a natural-vector triangle move at least
`sqrt(2.9409e-8)-1e-4=7.15e-5`, or Schur capture at least `1.9409e-8`.
In either case the coherent implementation/model error remains

\[
\boxed{\eta_{endpoint}\le3.80\times10^{-6}.}
\tag{93.32}
\]

The earlier pair-L6 `6.616e-5` number is a planning heuristic tied to a
different D4/oracle comparison, not an H5 gate.

### 93.8 Final decision and least additional measurement

Against literal D11:

- **univariate and bivariate consumer:** algebraically available with a small
  derivative-shell extension and `O(n^2)` elementwise work after supplied
  planes;
- **support-at-most-two source/carry:** already implemented and empirically
  closed as an autonomous route;
- **AQ H5 action:** exact from a resident AQ resolvent, but D11 has no such
  state and the action is comparator ownership, not true residual novelty;
- **genuine `[311]+[221]` source:** neither observed nor generated;
- **generic complete action:** physically over budget by the prior exact audit.

Therefore the present H5 route is **CLOSED**.  The least additional lawful
measurement is not another scalar cumulant or residual correlation.  It is
one sealed, weights-only `FUSED-K5-3D-ACTION` observable: either the per-layer
full-output vector (93.29), with a proved complete-state propagation, or the
final vector (93.30) directly.  A lower-dimensional replacement is lawful only
with the explicit target-free factor rank, collision sewing, endpoint tail,
and error receipts in (93.27), (93.31), and (93.32).

### Graph split

- `P2R0G19B_MATH_H5_AQ_CGF` — **solved exact comparator**: (93.4)--(93.7).
- `P2R0G19B_MATH_H5_NORMAL_OWNERSHIP` — **solved**: CP/H1 descendants and
  disconnected `Lambda3 Lambda4+Lambda3^3/6` are old; (93.8)--(93.14).
- `P2R0G19B_MATH_H5_RELU_UNIVARIATE` — **solved exact all bias**:
  (93.15)--(93.20).
- `P2R0G19B_MATH_H5_RELU_PAIR` — **solved exact linear action**:
  (93.21)--(93.23), conditional on supplied planes and Mehler-tail receipt.
- `P2R0G19B_MATH_H5_SUPPORT2` — **closed/redundant**: exact old
  `k5d/K41/K32` ledger, unstable autonomous carry.
- `P2R0G19B_MATH_H5_THREE_DISTINCT_INFORMATION` — **closed current D11**:
  `[311]+[221]` is the first unobserved source.
- `P2R0G19B_MATH_H5_GROUPED_ACTION` — **conditional reopen**: target-free
  grouped-rank circuit (93.25)--(93.27) plus complete-state action.
- `P2R0G19B_MATH_H5_GENERIC_ACTION_NOGO` — **closed this form**: prior
  treewidth-three/physical-budget receipt.
- `P2R0G19B_MATH_H5_R35_SENTINEL` — **defined, not authorized here**:
  freeze (93.29) or (93.30), then apply (93.31)--(93.32) with no tuning.
- `P2R0G19B_MATH_H5_MIN_OBSERVABLE` — **least reopen**: sealed target-free
  complete `[311]+[221]` action vector, not a label-seeded scalar.

## 94. Independent audit of the causal marginal-H6 candidate

Candidate audited: `public_d11_pair_l6_causal_v1/estimator_pair_l6.py`, frozen
hash `b807dbdf68b50f45d7060c54c8c7938ffd446d339f6021545dc34b2554d22dd0`.
No value run is used here.

### 94.1 Exact PASS/HOLD answer

| claim | verdict | reason |
|---|---|---|
| K33/K42/K51 affine coefficients `10/15/6` | **PASS** | exact multinomial/orientation count |
| raw-to-standardized factor and `/720` | **PASS** | `6!`, with `sigma^-6` applied exactly once |
| insertion into pair slots `(0,6)` and `(6,0)` | **PASS** | exact marginal-L6 endpoints |
| diagonal replacement | **PASS, no double count** | pair diagonal is discarded and replaced by the univariate result |
| connected-L6 novelty versus old `L3^2/2` | **PASS algebraically** | primitive `Lambda6` adds to, rather than duplicates, the same derivative address |
| exact one-transition selected-pair-source to marginal kappa6 | **PASS** | retained vector is exactly the diagonal of the pushed selected K6 tensor |
| exact full bivariate reset | **HOLD / false if claimed** | five mixed target collision slots are omitted |
| exact autonomous H6 transport | **HOLD** | target support-three-and-higher sectors and finite-order completion are omitted |
| complete-old endpoint value | **HOLD** | source novelty does not prove endpoint Schur novelty or gain |

Thus the implementation is correctly typed as a **causal diagonal-projection
candidate**.  It is not a complete H6 closure.

### 94.2 Affine coefficient and orientation audit

Let `W[i,a]` map source coordinate `i` to target preactivation `a`.  At the
post-reset source cut use

\[
K33_{ij}=\operatorname{cum}(x_i^3,x_j^3),\quad
K42_{ij}=\operatorname{cum}(x_i^4,x_j^2),\quad
K51_{ij}=\operatorname{cum}(x_i^5,x_j),
\tag{94.1}
\]

with zero diagonals.  `K33` is symmetric; `K42` and `K51` are oriented, so
their transposes are K24 and K15.  The selected support-two contribution to
the target marginal is

\[
\boxed{
\begin{aligned}
r^{(6)}_a={}&10\sum_iW_{ia}^3(K33W^3)_{ia}\\
&+15\sum_iW_{ia}^4(K42W^2)_{ia}
 +6\sum_iW_{ia}^5(K51W)_{ia}.
\end{aligned}}
\tag{94.2}
\]

For K42 and K51, `15=6!/(4!2!)` and `6=6!/(5!1!)`; summing the full oriented
plane already includes the complementary orientation, so no extra factor two
is allowed.  For K33, the unordered multiplicity is `20=6!/(3!3!)`, while
the full symmetric off-diagonal quadratic form counts each unordered pair
twice, giving `20/2=10`.  Therefore all three coefficients pass.

Equation (94.2) is precisely

\[
r^{(6)}_a=\widetilde K_6(e_a,e_a,e_a,e_a,e_a,e_a)
\tag{94.3}
\]

for the part of the source tensor in the three named planes.  The affine
collapse to this one target-free vector is exact for a **one-coordinate test
function at the immediately following reset**.

### 94.3 Slot and scale audit

Let `sigma_a` be the target preactivation standard deviation.  The
standardized marginal cumulant is

\[
g_{6,a}={r^{(6)}_a\over\sigma_a^6}.
\tag{94.4}
\]

For a ReLU raw moment row `p`, its linear primitive-L6 response is

\[
\delta R_{p,a}={g_{6,a}\over6!}\,c_{p,6,a}
={r^{(6)}_a\over720\sigma_a^6}c_{p,6,a}.
\tag{94.5}
\]

The candidate uses `raw_k6*inv_sigma6/720` in the univariate map.  Its
manually formed fourth-moment coefficient is

\[
c_{4,6}=24\sigma^4\phi(-\alpha)\operatorname{He}_1(-\alpha)
       =-24\sigma^4\alpha\phi(\alpha),
\tag{94.6}
\]

which matches the code exactly.

In the pair kernel the derivative feature at slot `(a,b)` already contains
`sigma_i^{-a}sigma_j^{-b}`.  Hence the correct raw weights are

\[
\omega_{60,ij}\mathrel{+}={r^{(6)}_i\over720},\qquad
\omega_{06,ij}\mathrel{+}={r^{(6)}_j\over720},
\tag{94.7}
\]

with no second standardization.  The candidate writes these into indices
`16=(6,0)` and `10=(0,6)`, respectively.  Those addresses already contain
the disconnected `Lambda3^2/2` weights.  Addition is correct because the
order-six differential address has total coefficient

\[
\left[\Lambda_6+\frac12\Lambda_3^2\right]_{(6,0)/(0,6)};
\tag{94.8}
\]

the connected sixth cumulant is not the disconnected K3 square.

For off-diagonal pair moments, both marginal endpoint terms in (94.7) are
real contributions.  On the diagonal, evaluating a correlated-pair formula
would not be the lawful marginal calculation; the unchanged code overwrites
that diagonal with (94.5).  Therefore the pair writes and the univariate
write do not double count.

### 94.4 Why marginal collapse is not a full reset: exact counterexample

The complete linear sixth-order pair response at the target cut is

\[
\boxed{
\delta M^{pq}_{ab}
=\sum_{r=0}^{6}{\widetilde K_{r,6-r;ab}\over r!(6-r)!}
 G^{pq}_{r,6-r;ab}.}
\tag{94.9}
\]

The candidate retains only `r=0,6`.  It omits

\[
\boxed{
\widetilde K15, \widetilde K24, \widetilde K33,
\ \widetilde K42, \widetilde K51}
\tag{94.10}
\]

at the **target** pair cut.  Source support-two does not imply target
support-one after a dense affine map.

A two-coordinate formal tensor is already a decisive witness.  Take a target
sixth cumulant with

\[
\widetilde K33_{12}=1,\qquad
\widetilde\kappa_{6,1}=\widetilde\kappa_{6,2}=0,
\tag{94.11}
\]

and every other sixth face zero.  The marginal vector (94.3) is identically
zero, so the candidate emits no correction.  Nevertheless, at an independent
centered Gaussian pair the raw `M22` response is

\[
\delta M^{22}_{12}
={\widetilde K33_{12}\over3!3!\sigma_1^3\sigma_2^3}
 c^{(1)}_{2,3}c^{(2)}_{2,3}
={\widetilde K33_{12}\phi(0)^2\over9\sigma_1\sigma_2}\ne0,
\tag{94.12}
\]

because `c_{2,3}=2 sigma^2 phi(0)`.  This witness survives at zero bias and
does not rely on a small numerical effect.  It proves that causal recomputation
of the next source cannot repair the already omitted current-reset pair
response.

The candidate's association is still internally causal: it recomputes the
next K33/K42/K51 source after the changed reset rather than reusing a stale
base tape.  That correctly propagates its chosen diagonal projection.  It
does not make the projection exact.

### 94.5 Missing source and transport sectors

The source wrappers are complete only for the exactly-two-coordinate sixth
shell `[33]+[42]+[51]` (with reverse orientations).  They exclude the source
marginal `[6]` and every source support-three-or-more partition:

\[
[411], [321], [222], [3111], [2211], [21111], [111111].
\tag{94.13}
\]

Even if only those selected pair sources are injected, a dense `W` produces
the five mixed target pair planes (94.10), as well as target sectors in
(94.13).  The least addition needed for an exact reset of D11's retained
univariate **and bivariate** state is the five target planes in (94.10),
computed by their polarized affine actions and consumed in (94.9).  The least
addition needed for exact autonomous H6 transport is the complete reachable
sixth-order quotient, or an endpoint-Bellman proof that every omitted sector
is future-dark.  No such annihilation receipt exists.

### 94.6 Complete-old novelty and finite-order boundary

The connected K33/K42/K51 assemblies subtract all lower set partitions.
Therefore their primitive `Lambda6` source is algebraically distinct from the
old `Lambda3^2/2` channel, and the candidate's parent-minus-candidate output is
an honest physical difference of the implemented projected recurrence.  This
passes source-level dedupe.

It does not by itself establish portfolio novelty.  Before stacking or value
credit, the final vector must still be residualized against the complete old
endpoint span:

\[
d_6^\perp=(I-P_{\mathcal O})d_6,\qquad
\Delta_6={\langle e,d_6^\perp\rangle^2\over\|d_6^\perp\|^2}.
\tag{94.14}
\]

There is also a formal-order boundary.  Adding `Lambda6` alone is the exact
primitive grade-four term (`grade Lambda_r=r-2`) in the adopted Edgeworth
truncation.  A complete grade-four density additionally contains

\[
\Lambda_3\Lambda_5+\frac12\Lambda_4^2
+\frac12\Lambda_3^2\Lambda_4+\frac1{24}\Lambda_3^4,
\tag{94.15}
\]

and a finite-amplitude exact response contains still higher products involving
`Lambda6`.  Thus `F(z+Delta z)-F(z)` is exact for the implemented projected
map, not an exact physical all-orders ReLU law.

### 94.7 Final audit disposition

**PASS** the candidate's narrow advertised semantics:

> verified K33/K42/K51 source planes -> exact selected contribution to the
> next marginal raw kappa6 -> correctly normalized univariate L6 and
> bivariate marginal-endpoint response -> causal recomputation of the same
> projected state.

**HOLD** any stronger claim, including “complete pair H6,” “exact H6 causal
closure,” or package/value readiness.  The minimum mathematical repair for
the first stronger claim is (94.10); the minimum repair for the second is
(94.10)+(94.13), or a complete endpoint-dark certificate.  The rank-35 value
sentinel may test the narrow projected candidate, but it cannot certify the
omitted sectors.

### Graph split

- `P2R0G19B_MATH_PAIR_L6_ACTION_COEFFICIENTS` — **PASS exact**: 10/15/6 in
  (94.2).
- `P2R0G19B_MATH_PAIR_L6_SLOT_NORMALIZATION` — **PASS exact**: raw `/720`,
  one standardization, and no diagonal double count, (94.4)--(94.8).
- `P2R0G19B_MATH_PAIR_L6_MARGINAL_CAUSAL_MAP` — **PASS projected map**:
  exact source-to-marginal action with causal recomputation.
- `P2R0G19B_MATH_PAIR_L6_TARGET_MIXED_SLOTS` — **HOLD / required repair**:
  five planes (94.10), with counterexample (94.11)--(94.12).
- `P2R0G19B_MATH_PAIR_L6_COMPLETE_TRANSPORT` — **HOLD**: missing sectors
  (94.13) or endpoint-dark theorem.
- `P2R0G19B_MATH_PAIR_L6_SOURCE_DEDUPE` — **PASS algebraic / endpoint open**:
  connected `Lambda6` is not old `Lambda3^2/2`; Schur gate (94.14) remains.
- `P2R0G19B_MATH_PAIR_L6_GRADE4_COMPLETION` — **HOLD**: missing terms
  (94.15) and finite-amplitude remainder.

## 95. Target-blind raw20 graph messages: exact map, prior-art boundary, and information ceiling

### 95.1 Prior art changes the novelty claim

The signed raw20 recurrence

\[
 Z_l=\Phi_l\,W_l^\top Z_{l-1}+X_l
\tag{95.1}
\]

has already been measured in
`public_q24_nonlinear_transported_raw20_head_v1`: OOF moved from
\(3.066199675740282\times10^{-8}\) to
\(3.053845857548064\times10^{-8}\), a \(0.402903\%\) gain.  It is a
control, not new work.

There is also partial square-weight prior art.  The held
`public_innovation_mechanism_screen_v1` sent four raw channels through
centered \(W,W^2,W^3,W^4\) at the final seam and through one two-hop arm.
The one-hop messages alone explained \(1.146741\%\) of teacher-forced held
innovation.  Adding them to the local basis changed held residual MSE only
from \(8.23604753994647\times10^{-9}\) to
\(8.223753811499082\times10^{-9}\), a \(0.149271\%\) incremental gain.
Thus a broader \(W^2\) raw20 bank is a completion/control, not a new
mechanism.

A repository-wide source audit found no prior q-head/raw20 message through
\(|W|\).  The absolute-weight operator is the only genuinely unexecuted
operator in the frozen joint arm.  The complete machine-readable audit is
`public_q24_raw_abs_sq_messages_v1/PRIOR_ART_AUDIT.json`.

### 95.2 Frozen target-blind feature map

For target coordinate \(j\), source coordinate \(i\), raw channel \(q\),
and suffix layer \(l=10,\ldots,15\), set

\[
 d^{(1)}_{lj}=\max\!\left(\sum_i|W_{l,ij}|,2^{-30}\right),\qquad
 d^{(2)}_{lj}=\max\!\left(\sum_iW_{l,ij}^2,2^{-30}\right),
\tag{95.2}
\]

\[
 A_{l,jq}={\sum_i|W_{l,ij}|X_{l-1,iq}\over d^{(1)}_{lj}},\qquad
 E_{l,jq}={\sum_iW_{l,ij}^2X_{l-1,iq}\over d^{(2)}_{lj}}.
\tag{95.3}
\]

The 40 columns are ordered `abs_raw00..19,sq_raw00..19`.  They use only
literal native `raw20.npy` and sealed observer-fit `weights.npy`; no host,
truth, residual, fold assignment, or fitted statistic is read.  The cache is
therefore target-blind.  Fold centers/scales are fitted only by the consuming
OOF runner.

The source/target orientation is exact.  Literal D11 stores \(W\) with
source rows and target columns, and the existing state update is `state @ W`;
therefore (95.3) is `operator.T @ X`.

### 95.3 Exact permutation equivariance and normalization

For source and target permutation matrices \(P,Q\), let
\(X'=PX\) and \(W'=PWQ^\top\).  Entrywise absolute value and square obey

\[
 |W'|=P|W|Q^\top,\qquad (W')^{\odot2}=P(W^{\odot2})Q^\top.
\tag{95.4}
\]

Their column denominators permute by \(Q\), hence
\(A'=QA\) and \(E'=QE\).  Both normalized operators are column-stochastic,
so a constant source channel is preserved exactly.  The independent
synthetic audit gives maximum equivariance error
\(4.440892098500626\times10^{-16}\) and constant-preservation error zero.

### 95.4 Deployment cost and cache size

There are two skinny matrix products, six layers, and 20 columns:

\[
 C_{\rm dense}={2\cdot6\cdot20\over1024}=0.234375
\tag{95.5}
\]

width-by-width GEMM equivalents.  Counting multiplication and addition
separately, the matrix products use exactly

\[
 2\cdot6\cdot1024\cdot20\cdot(2\cdot1024-1)
 =503{,}070{,}720
\tag{95.6}
\]

scalar FLOPs per network.  Operator construction, column reductions, floors,
and divisions add \(25{,}411{,}584\) under the one-operation convention, for
\(528{,}482{,}304\) total.  This is \(0.02403\%\) of \(2^{41}\), before the
consumer head's separately audited input projection.  One emitted row is
\(6\cdot1024\cdot40\cdot4=983{,}040\) bytes; all 688 rows are
676,331,520 bytes excluding small receipts.

The target-blind cache completed for 688/688 rows.  Every row has shape
`[6,1024,40]`, dtype `float32`, and finite entries.  Binding receipts are:

- builder SHA-256 `783b89bc27a4707915a9e5df542ad1945969edca6e9ecf7390f528e0f646deea`;
- `CACHE_SUMMARY.json` SHA-256
  `180e2f3458f64552583877dec80c8558e61bf0bebda7676f0921df9db823e9e9`;
- ordered digest of the 688 row-output digests
  `9875706603c09210b5d375fa1ab6bba6c9289009ade5d605818b84e917ed1518`.

The observed all-corpus cache range is `[-0.8671926259994507,
1.9095311164855957]`; per-row RMS lies in `[0.4647474619014388,
0.5578607692608568]`.  These are feature-only diagnostics, not a label or
value screen.

### 95.5 Sharp information ceiling

These features are not a closure theorem.  At a fixed target column \(j\)
and raw channel \(q\), even after adding the old signed message, the source
vector \(x\in\mathbb R^{1024}\) is observed only through

\[
 \langle w_j,x\rangle,\quad
 \langle |w_j|,x\rangle,\quad
 \langle w_j^{\odot2},x\rangle.
\tag{95.7}
\]

For generic \(w_j\), the common kernel has dimension \(1021\).  Across 20
channels this leaves at least 20,420 source degrees of freedom at that target.
Any quadratic cross-channel or pair-plane response can change along this
kernel while all 82 head inputs remain fixed.  In particular, covariance,
K21/K31/K22, and true-minus-AQ response vectors are not identified by (95.7).
The arm is a cheap statistical screen for edge-magnitude correlations, not an
exact surrogate for the perfect-state oracle.

The least next target-free linear observable that defeats this particular
first-moment ambiguity is a weighted cross-channel Gram,

\[
 G_{l,j,qr}={\sum_iW_{l,ij}^2X_{l-1,iq}X_{l-1,ir}\over d^{(2)}_{lj}},
 \qquad q\le r,
\tag{95.8}
\]

or a lower-rank preregistered projection of it.  Equation (95.8) has 210
channels and is not authorized in the current run; it is recorded only as the
minimal information repair if the frozen arm fails.

### 95.6 Honest attribution rule for the single joint OOF arm

The authorized model has 82 inputs: local22, old signed20, absolute20, and
square20.  With no preregistered factorial, an observed joint gain cannot be
partitioned causally.  Syntactically, 20/82 input columns and half of the new
message products are the new \(|W|\) operator; signed \(W\) is fully old and
\(W^2\) is partial operator-level prior art.  Scientifically, the only lawful
claim is “the frozen joint arm improved/failed.”  Feature weights, saliency,
or post-hoc removal cannot establish a positive \(|W|\) contribution.

### Graph split

- `P2R0G19B_RAW20_SIGNED_MESSAGE_PRIOR` — **CLOSED measured**, (95.1).
- `P2R0G19B_RAW20_W2_MESSAGE_PRIOR` — **PARTIAL measured / weak**, final-seam
  four-channel receipt in §95.1.
- `P2R0G19B_RAW20_ABS_MESSAGE_MAP` — **SOLVED target-blind construction**,
  (95.2)--(95.4); value remains open in the single authorized joint run.
- `P2R0G19B_RAW20_GRAPH_MESSAGE_COST` — **SOLVED**, (95.5)--(95.6), consumer
  head cost pending the frozen implementation receipt.
- `P2R0G19B_RAW20_GRAPH_MESSAGE_INFORMATION` — **CLOSED as exact closure / OPEN
  statistical value**, kernel theorem (95.7).
- `P2R0G19B_RAW20_GRAPH_MESSAGE_ATTRIBUTION` — **CONDITIONAL joint-only**: no
  post-hoc component attribution without a new preregistered factorial.

## 96. Predeclared failure successor: the exact 210-channel raw20 cross-Gram

Section 95 proves that first-moment messages leave a 1021-dimensional source
kernel per target and channel.  The minimal universal degree-two repair is the
packed symmetric weighted Gram.  For fold-standardized/clipped source
features \(Z_{l-1,iq}\), define

\[
 G_{l,j,qr}={\sum_iW_{l,ij}^2Z_{l-1,iq}Z_{l-1,ir}\over
 \max(\sum_iW_{l,ij}^2,2^{-30})},\qquad 0\le q\le r<20.
\tag{96.1}
\]

There are exactly \(\dim\operatorname{Sym}_{20}=210\) ambient channels in
lexicographic triangular order.  Source and final-message normalization must
use only the OOF training fold; neither target values nor response-derived
statistics enter (96.1).

Under \(X'=PX,W'=PWQ^\top\), equation (96.1) obeys \(G'=QG\), proving exact
source/target permutation equivariance.  For an unconstrained 20-channel
source, 210 is minimal for all quadratic queries: every linear compression of
\(\operatorname{Sym}_{20}\) below 210 has a nonzero symmetric kernel
direction \(H\), and near a positive-definite baseline the two realizable
Grams \(I\pm\epsilon H\) collide under the compression while the query
\(\langle H,G\rangle\) separates them.

Literal raw20 channel zero is depth and is constant across neurons at a fixed
layer.  Hence \(G_{00}\) is fixed and the actual per-layer family has at most
209 varying coordinates.  Without source clipping, reuse of the current
first moments would also remove the 19 \(G_{0r}\) entries, leaving a sharp
190-channel tail on raw channels 1..19.  The predeclared successor retains all
210 because its clipped source law does not make that reuse exact and no
post-failure subset selection is authorized.  The minimality theorem is
therefore ambient, not a claim that literal D11 needs all 210.

The leading bill is

\[
 {6\cdot210\over1024}=1.23046875\ \text{GEMM equivalents},
\tag{96.2}
\]

or 2,641,121,280 multiply/add FLOPs.  Source outer products and output
division bring the incremental total to 2,643,701,760 if the current
\(W^2\) buffer/denominator is reused: \(0.12022\%\) of \(2^{41}\).
Standalone construction totals 2,656,284,672 operations.  A hypothetical
all-688 cache would be 3,550,740,480 bytes, so none is built before the
failure gate.

Prior-art scope is exact.  `public_full688_full_gram_pooled_context_cv_v1`
used the same \(W^2\) topology on four learned correction states (10 pair
channels) and regressed OOF from \(3.2334958229470984\times10^{-8}\) to
\(3.2444190155738505\times10^{-8}\).  Other arms carry only diagonal learned
state energies or raw first moments.  No executed arm uses all literal
\(X_qX_r\) sources.  Thus the source bank is new, the operator idea is old,
and the prior is low.

Even (96.1) is not a v/k3/k4 closure: equal first and second moments can have
different degree-three/higher connected structure.  It is a predeclared
successor only if the authorized 82-input arm fails, with no target access,
cache build, feature subset, or rank sweep before a separate frozen release.
Full details are in `public_q24_raw20_cross_gram_successor_v1/`.

### Graph split

- `P2R0G19B_RAW20_CROSS_GRAM_MAP` — **SOLVED construction / DORMANT** until
  the 82-input failure gate, (96.1).
- `P2R0G19B_RAW20_CROSS_GRAM_MINIMALITY` — **SOLVED ambient / QUALIFIED
  literal**: 210 is sharp unconstrained; literal depth fixes one coordinate
  and an unclipped-first-moment design could reduce the new tail to 190.
- `P2R0G19B_RAW20_CROSS_GRAM_COST` — **SOLVED feature bill**, (96.2); consumer
  head remains separately billable.
- `P2R0G19B_RAW20_CROSS_GRAM_PRIOR` — **LOW / PARTIAL prior art**: learned
  four-state Gram regressed, literal raw20 pair source unexecuted.
- `P2R0G19B_RAW20_CROSS_GRAM_HIGHER_CHAOS` — **CLOSED as exact closure**:
  degree-three/higher ambiguity remains.

## 97. Independent terminal audit of the frozen 82-input graph-message head

The one authorized OOF run is byte-complete and internally consistent.  I
independently rehashed the frozen program, preregistration, durable run claim,
model bundle, result, and both cache summaries.  The binding digests are:

- program `e5028cfe8cd088abc6f4997d9995f2c6b1ba65afd1ee280e763856578a6390ba`;
- preregistration `3e30b7f9c9ada81a7e15b2e6ac3792af5f47ddab7c176f808b162fafa665a4d1`;
- one-use claim `64bd82fd4961c63c8fe413ca842e450f4b35b703d7992886b80687d381b83a68`;
- result `87624c36e90e3db7e5592ff8eb56acf5696a37d9ad0f2dd7ed656394301f98b4`;
- OOF models `21572ccd20eb339057c9c32a127fd794c14355dc654df517a27091000ee807d2`.

The claim was durably created with exclusive-create and directory `fsync`
after all 1,376 cache rows had been rehashed and before the first semantic
target decode.  Hashing the sealed target file before the claim was byte
access, and is disclosed; no target value was decoded then.  The only target
role afterward was the declared observer-fit response.  The run opened no
observer-selection target, name, score, board result, or estimator.

Recomputing from the 688 unique per-network rows and the frozen saved-control
rows gives

\[
 R_{82}=2.9728904042473882\times10^{-8},\qquad
 R_{0}=2.985249910997227\times10^{-8},
\tag{97.1}
\]

\[
 R_0-R_{82}=1.2359506749839335\times10^{-10},\qquad
 {R_0-R_{82}\over R_0}=0.004140191648380495.
\tag{97.2}
\]

The network win fraction is (0.6162790697674418) and the paired-network
(z) statistic is (7.782984179268749).  All four folds improve their
matched saved control, but the frozen material threshold was
(2.9553974118872554\times10^{-8}).  Thus the aggregate 0.414019% gain fails
the binding 1% gate.  It is also (3.198322064738811\times10^{-10}) worse
than the current board raw reference.  The exact decision is therefore
**FAIL MATERIAL OOF / STOP NO REFIT**, despite the statistically stable sign.

The source semantics also pass static audit: (W) has source rows and target
columns, every message commutes with independent source/target permutations,
added-channel statistics are fit-fold-only, local22 uses the corresponding
fold-J statistics, and held scoring uses all 1024 nodes after fixed 256-node
fit sampling.  This does not change the information theorem in (95.7), and
it does not isolate the contribution of |W|.  The complete machine-readable
independent receipt is
`public_q24_raw_abs_sq_messages_v1/INDEPENDENT_82_HEAD_AUDIT.json`.

### Graph split

- `P2R0G19B_RAW20_82_OOF_INTEGRITY` — **PASS terminal**, frozen hashes and
  target firewall above.
- `P2R0G19B_RAW20_82_OOF_VALUE` — **CLOSED FAIL**, 0.414019% is below the
  preregistered 1% gate.
- `P2R0G19B_RAW20_82_REFIT` — **CLOSED NOT AUTHORIZED**, no refit, ablation,
  or deployment follows this result.
- `P2R0G19B_RAW20_CROSS_GRAM_MAP` — **ELIGIBLE PREDECLARED SUCCESSOR**, but
  still requires a separately sealed consumer and target authorization.

## 98. Smallest cache-free D11 realization, conditional only and closed by §97

Although §97 forbids packaging this failed arm, the exact implementation
receipt is useful if the same feature law is ever independently reopened.
It does not require a cache, NumPy, Torch, or a second copy of the old J path.

Let (x_l\in\mathbb R^{n\times20}) be the native raw20 row emitted by D11,
let (a_l\in\mathbb R^n) be the already-paid primary J state, and let
(S_l) be raw20 standardized with the signed-cache constants.  Maintain

\[
 Z_2=S_2,\qquad Z_l=\phi_l(W_l^\top Z_{l-1})+S_l,quad l=3,\ldots,15,
\tag{98.1}
\]

and, only at (l=10,\ldots,15), emit

\[
 A_l={|W_l|^\top x_{l-1}\over
             \max(|W_l|^\top\mathbf1,2^{-30})},\qquad
 E_l={(W_l^{\odot2})^\top x_{l-1}\over
             \max((W_l^{\odot2})^\top\mathbf1,2^{-30})}.
\tag{98.2}
\]

The local innovation required by the head is available without another J
network:

\[
 q_l^{J}=a_l-\phi_l(W_l^\top a_{l-1}).
\tag{98.3}
\]

After fold/full-fit constants have standardized the 60 added fields, the
literal 82-to-256-to-256-to-256-to-1 pointwise head emits (s_l).  Its own
endpoint state is

\[
 c_{10}=s_{10},\qquad
 c_l=\phi_l(W_l^\top c_{l-1})+s_l,quad l=11,\ldots,15.
\tag{98.4}
\]

D11 then returns its host unchanged before layer 10 and `max(host+c_l,0)`
on the six suffix rows.  The old J primary output is an input to (98.3), not
a second correction added to the host.  This is the no-double-count rule.

The lifetime schedule is constructive.  Run the old J cell once, online,
as each native raw20 row is produced; retain only (x_{l-1},x_l).  After the
host has finished all uses of the current weight, multiply the concatenated
signed/base-innovation/candidate banks by (W_l).  Then overwrite the owned
float64 weight work copy first by |W_l| and then by its square, consuming
(98.2) before recycling it.  Thus no new (n\times n) operator is required.
The current post-pass full native tape is removed rather than duplicated.

The OOF cache used float64 accumulation followed by a float32 cast at each
recurrence/message.  Preserving that numerical law, rather than silently
switching to a fused float32 approximation, gives this incremental census
above the already-paid D11 and base-J work:

| component | scalar operations |
|---|---:|
| 13 signed 20-column transports | 544,993,280 |
| six exact base-(q^J) transport columns | 12,576,768 |
| five candidate-state transport columns | 10,480,640 |
| six |W| and six (W^2) 20-column transports | 503,070,720 |
| |W|/(W^2) formation, denominators, floors, divisions | 25,411,584 |
| signed/base-(q)/candidate recurrence arithmetic | 555,008 |
| three-layer 82/256 head, including hidden biases | 1,871,708,160 |
| SiLU elementary arithmetic, excluding exponent calls | 14,155,776 |
| feature normalization and physical source scaling | 1,312,768 |
| six suffix host additions/floors | 12,288 |
| **total basic scalar operations** | **2,984,276,992** |

There are additionally exactly 4,718,592 scalar exponential calls and
1,329,152 comparison/clip operations.  The basic-operation total is
(0.13570920564234257\%) of (2^{41}).  It is not a literal FlopScope bill;
metering remains mandatory.

A node-streamed head needs no (1024\times82) packed feature matrix.  With
two raw20 rows, two signed-state banks, one 40-column abs/square bank, one
22-column float64 output scratch, two denominators, candidate/base-(q)
vectors, one-node head scratch, and all candidate parameters/constants, the
constructive incremental working set is 1,324,000 bytes if the mutable
float64 weight work copy is proved owned.  If aliasing forces a dedicated
8 MiB operator copy, the bound is 9,712,608 bytes.  The old stacked native
tape occupied 1,376,256 bytes, so the owned-work-copy implementation is
non-increasing relative to that post-pass design.  Actual process RSS is
still **HOLD pending a meter**.

Two gates prevent this from being a package now.  First, §97 forbids a refit.
Second, the four OOF heads are not one canonical estimator payload: their
local22 inputs depend on four different fold-J models.  A cheap single-model
deployment therefore requires a separately preregistered full-688 refit and
an f64-cache-to-FlopScope parity receipt.  A four-model ensemble would repeat
the fold-J consumers and is neither the measured arm nor the minimal bill.

### Graph split

- `P2R0G19B_RAW20_82_CACHE_FREE_RECURRENCE` — **SOLVED algebraically / CLOSED
  by value gate**, (98.1)--(98.4).
- `P2R0G19B_RAW20_82_NO_DUPLICATE_J` — **SOLVED typed ownership**: one J
  state supplies local22; its old output is not added separately.
- `P2R0G19B_RAW20_82_STATIC_BILL` — **SOLVED static / HOLD literal meter**,
  2,984,276,992 basic operations and the workspace bounds above.
- `P2R0G19B_RAW20_82_NUMERIC_PARITY` — **OPEN only under a future lawful
  reopen**: float64-cache semantics must not be replaced silently by f32.
- `P2R0G19B_RAW20_82_DEPLOYMENT_PAYLOAD` — **CLOSED absent refit**: four OOF
  models do not define one cheap target-free estimator.

## 99. Independent direct-transpose fold-0 release and claim audit

The next frozen arm is the predictor-only D11 cubic-action transpose head at

```text
public_full688_d11_cubic_action_transpose_q_sentinel_v1/
```

The final runner, release, and durable claim are bound respectively by

```text
run_fold0.py                    c47c36caf2877f1b31b76446d29817e844d1318f206d7b9040192d8dc0d18b37
FOLD0_EXECUTION_RELEASE.json    99122771701ee879b264d842559ad5b69927a579d700cfdbfa449c13c0eb54ce
FOLD0_RUN_CLAIM.json            e64d73daa5cae27b311822fdaf60bd28af1197c31fd4cb6886c6538ea56c5f47
```

The target-free preflight returned
`READY_FOR_EXACTLY_ONE_FOLD0_RUN`: Python 3.11.5, NumPy 1.26.4, Torch 2.7.0,
6,539 parameters, exact release validation, no claim/result/model/failure at
preflight, zero target values decoded, and zero derived caches.  The bound
`execute` path independently enforces those runtime versions, all four BLAS/
OMP thread variables equal to four, and niceness at least ten before it can
claim the run.

### 99.1 Exact fail-fast identity

With

\[
 e_l=y_l-h_l-A_l,\qquad q_l^*=e_l-T_l e_{l-1},\qquad
 d_4=0,\quad d_l=T_l d_{l-1}+q_l^*,
\]

induction gives the exact telescoping identity

\[
 \boxed{e_{15}-d_{15}=T_{15}T_{14}\cdots T_5e_4}. \tag{99.1}
\]

The amended runner therefore gates before training by returning only the 172
fold-0 held target rows, forming their layer-4 residuals, and applying eleven
frozen float64 matvec/float32-round tangent steps.  Fit targets are not
returned, materialized into NumPy, or used unless the mean square in (99.1) is
strictly below (10^{-8}).  If the gate fails, the runner writes a terminal
result with zero training and no retry.

There is one deliberately explicit storage caveat.  The sealed target Parquet
has one 688-row row group.  Its filtered read returns only 172 rows, so there
is no statistical access to fit values before the gate, but physical decoding
of pages containing fit-row bytes is not certified.  The receipt therefore
says `fit rows returned/materialized/used = 0`, not that storage-level page
decoding is proved selective.

After a gate pass, training remains outer-fold pure: fold-0 J, centers,
scales, RMS, q-RMS, and student updates use the 516 fit rows; the 172 held
rows are gate/score only.  The `source_on` and `source_zero` heads have the
same initialization, row order, node samples, optimizer, target, and
architecture.  Only the three appended predictor coordinates are zeroed.
The old-owned cubic action (c_l) is never added directly.

### 99.2 One-shot authority and stop law

The release permits only observer-fit columns `mlp_id,all_layer_means`, one
fold-0 sentinel, and no retry.  Observer-selection, names, scores, network,
cache, remaining folds, all-data fit, package, integration, meter, board, and
submission are all false.  The claim was created with `O_EXCL`, file fsync,
and directory fsync before any target value was returned to the program.  It
records zero target values before claim and 992,217,088 verified native bytes.

Any post-claim exception is terminal.  Oracle failure trains no model; fold-0
failure closes the spelling; fold-0 pass writes HOLD and requests a different
authority.  There is no folds-1--3 loop in this runner.  At the independent
claim audit, result, model, and failure artifacts were all absent, as expected
for a live run.

### 99.3 Exact resource distinctions

The static per-network feature generator costs 21,071,872 arithmetic and
78,848 comparison operations, with 233,472 bytes of NumPy-owned peak state.
A minimal one-output deployment above already-paid D11/J work has 172,300,288
non-SiLU basic operations, 1,441,792 scalar SiLU evaluations, and 563,200
comparisons.  Counting each SiLU as three elementary operations gives
176,625,664 basic operations.  The preregistered 265,524,224-operation number
is a conservative package reserve, not a claim about autograd or wall time.

For the sentinel itself, the held gate returns 2,818,048 target floats
(11,272,192 bytes), retains the same number of fold-J prediction bytes, and
performs exactly 3,967,811,584 tangent arithmetic operations excluding J
inference.  Only after a pass does it materialize the 45,088,768-byte full
target array and the 839,783,808-byte prepared bank.  Each arm then has 1,677
updates and 18,889,728 sampled node-layer examples.  Backpropagation and J
inference are intentionally not mislabeled as an exact static FLOP count; the
hard live RSS gate is eight GiB.

The independent receipts are

```text
INDEPENDENT_EXECUTION_RELEASE_AUDIT.json  1d2299754ef3b0581586179a50954594b40efeb6a665376bdd768c0aa40a94d6
INDEPENDENT_RUN_CLAIM_AUDIT.json          2bb67e2748eca296e179a01f8a83c71518f7f2fa9cbed38a7c749afcd239edaf
```

This audit opened no target values, native rows, or estimator, created no
feature cache, and did not execute the sentinel.  The 210-channel cross-Gram
successor remains frozen at design SHA
`26d6b3c21c46f639e8b20a5f45a4fec1a804599d106385f0366f07113e39b2c2`;
no cache or run was started.

### Graph split

- `P2R0G19B_CUBIC_TRANSPOSE_START5_GATE` — **PASS algebra and implementation**,
  exact identity (99.1), held-only semantic materialization before the gate.
- `P2R0G19B_CUBIC_TRANSPOSE_TARGET_PURITY` — **PASS statistical / storage caveat**,
  zero fit values returned or used pre-gate; physical Parquet page selectivity
  is not claimed.
- `P2R0G19B_CUBIC_TRANSPOSE_RELEASE` — **PASS exact-byte authority**, fold 0
  only and every broader authority false.
- `P2R0G19B_CUBIC_TRANSPOSE_CLAIM` — **PASS one-shot durable claim**, no retry
  and no remaining-fold authority.
- `P2R0G19B_CUBIC_TRANSPOSE_SENTINEL` — **ACTIVE**, terminal result pending;
  do not interfere with or duplicate the bound process.
- `P2R0G19B_RAW20_CROSS_GRAM_MAP` — **FROZEN ELIGIBLE, NOT RUN**, unchanged
  while the direct-transpose sentinel is active.

## 100. Direct-transpose terminal result and four-field successor interlock

The release-bound fold-0 process completed normally.  Its immutable roots are

```text
FOLD0_RESULT.json  e1ed27d0882dddb09698da78d167f7e34096ce44742b210b64792a3d20babd7f
FOLD0_MODELS.pt    868969ca197f19d1e2e70b7bbf51aa83ea7724b4571122eeab3826eb70437f20
```

There is no terminal-failure artifact.  The result status is
`FAIL_FOLD0_SENTINEL__TERMINAL_NO_RETRY`, and its binding hashes reproduce the
audited claim, release, runner, feature generator, native model, and target
bundle exactly.

### 100.1 Independent recomputation from the saved 172 rows

Recomputing every endpoint statistic from `per_network` gives

| metric | recomputed value |
|---|---:|
| fold-J base | 3.0923227278552953e-8 |
| source-on | 3.033603794829636e-8 |
| source-zero | 3.033159078239202e-8 |
| scored exact-q oracle | 4.698527945787826e-10 |
| direct start-5 oracle | 4.698527912579728e-10 |
| network win fraction vs fold J | 0.7441860465116279 |
| paired network z vs fold J | 7.57630483690023 |

All 172 role ranks are unique and all values are finite.  The direct and
scored oracle spells differ by only (-3.32081\,10^{-18}).  The source-on
head does improve fold J by (1.8988617\%), so generic head capacity and the
correction locus are both present.  The appended physical-source coordinates,
however, are incrementally harmful relative to the identical zero-source
control:

\[
 M_{\rm on}-M_{\rm zero}=4.44716590434\,10^{-12}>0,
\]

and their local-q loss is worse by (3.62121635933\,10^{-12}).  Source-on is
also worse than frozen independent-256 by (1.15720521169\,10^{-10}) and
misses its absolute endpoint gate by (2.38044073712\,10^{-9}).

Thus four substantive gates fail exactly as recorded:

1. endpoint MSE is not at most (2.79555972111794\,10^{-8});
2. source-on does not beat source-zero endpoint;
3. source-on does not beat source-zero local q; and
4. source-on does not beat frozen independent-256.

The oracle, 60% win, (z\ge3), and finiteness gates pass.  The conjunction is
false, so the direct spelling is terminal; folds 1--3, all-data refit,
packaging, and integration remain forbidden.

### 100.2 Model-byte replay

The two saved state dictionaries are finite float32 and contain 7,163 values
each, including 6,539 trainable parameters and 624 frozen normalization
buffers.  Recomputed canonical terminal hashes are

```text
source_on    2d370d45edfcd0f1bed7f3f743870c8f62936fa0ad01b2cfced35c0794fb2fb1
source_zero  7830b6d4d29ed95d6fa4ae8ec6d6a310d4aaee9e3432bbc8543c6e4255bb9732
```

Both arms replay the same initial-state hash
`907ada604ca3b31496f5dd71b4329d295e7b34aa678da59a83d3817f1162f94d`,
row-order hash
`763b453ac7c9d284d00ffccb38ff7ddd4938f98167dbf35806a444d82472eadb`,
and node-sample hash
`a47a9484564d28bfd2278972bb49ecab3a60b70a0aed46cf487bf5d73b0d0dfd`.
Each has exactly 13 epochs.  Maximum process RSS was 1,822,867,456 bytes,
well below the eight-GiB hard gate.

### 100.3 Conditional successor is now eligible, not authorized

The separately frozen directory

```text
public_q24_owned_l3_l3square_forward_fields_v1/
```

binds two conjunctive interlocks:

- direct start-5 exact-q oracle strictly below (10^{-8}); and
- direct source-on sentinel fails its frozen continuation gate.

The first is true at (4.69853\,10^{-10}) and the second is true by §100.1.
Therefore the four-field owned L3/L3² product-before-transport bundle is now
the sole **eligible** successor.  Its static roots remain

```text
IMPLEMENTATION_FREEZE.json  ac8be2bb247e505f37919823e988c8622058bf45385bd0faa5cdc223610f9666
feature_generator.py        5f4ed4a777f5ab80141bc2e3d1ced73b973f7a60b9f694dd330ca461b545c086
STATIC_VERIFICATION.json    905ccaf22a1444aede1df641155a3a36fadd8690958218a36566f64d095f0d94
```

Eligibility does not confer authority.  There is still no target, cache,
training, execution, remaining-fold, integration, meter, board, or submission
permission.  A lawful next transaction must use all four frozen fields jointly
against identical zeroed-feature bytes, with no post-result subset selection,
and must recover at least 8% of the preregistered parent-to-oracle gap with the
correct q sign.  Its resident-W² generator bill is 0.0234375 GEMM-equivalent.

The independent terminal receipt is

```text
INDEPENDENT_TERMINAL_RESULT_AUDIT.json  b527b9c24bc4a74c3da0ee6259bbfe100b7d1a2696c454d548cec876610bd0da
```

This audit read only the immutable result/model/control artifacts and static
successor files.  It opened no target values or native rows, launched no
training/scoring/estimator run, and created no derived cache.

### Graph split

- `P2R0G19B_CUBIC_TRANSPOSE_SENTINEL` — **CLOSED FAIL terminal**, four value/
  control gates fail; do not run folds 1--3.
- `P2R0G19B_CUBIC_TRANSPOSE_ORACLE` — **PASS strong**, start-5 exact-q raw
  (4.69853\,10^{-10}); correction-locus capacity is not the failure.
- `P2R0G19B_CUBIC_TRANSPOSE_INCREMENTAL_VALUE` — **CLOSED negative**, source-on
  loses to identical source-zero in endpoint and local-q loss.
- `P2R0G19B_OWNED_L3_L3SQUARE_FOUR_FIELD_INTERLOCK` — **SATISFIED**, both
  preregistered conditions are true.
- `P2R0G19B_OWNED_L3_L3SQUARE_FOUR_FIELD_EXECUTION` — **ELIGIBLE BUT NOT
  AUTHORIZED**, requires a new exact release; no cache/run has started.

## 101. Independent four-field fold-0 runner and release audit

The execution bundle at

```text
public_full688_d11_owned_l3_l3square_q_sentinel_v1/
```

is now frozen at

```text
run_fold0.py                    a6c891cd19b62e1d48280308d4607fefd2a9f8d6d2eaf4a0277dc230c5af6cc5
feature_generator.py            5c85616fdd102986cae4b97309d1ddedee5564d3c490b0ebf198fe7ac56fbda7
PREREGISTRATION.json            b3d7197b0410f93096fa32d8c018f0e4037bc710e1d5de5da43f84e220b9c624
FOLD0_EXECUTION_RELEASE.json    16715386a00d64347a20d1d4e85ecf5410d2c4d86abe79179c6dab78d3ec622b
FOLD0_RUN_CLAIM_CONTRACT.json   09cb93a6381b3ecc80196529f5fca055be0bf148c8ce6448226da1946f2de11a
STATIC_VERIFICATION.json        c0a3c18d81872e6c216bb5eaa4cc36deff716ddb5b1fdcf5c288d292059e779a
```

An earlier 26-input draft was rejected before release because it silently
dropped the frozen parent's signed/absolute/squared 60-coordinate bank.  The
final runner is the literal frozen successor: local22 + parent60 + new4,
therefore 86 inputs, three 256-wide linear/SiLU blocks, six scalar heads, and
155,398 trainable parameters.

### 101.1 Typed fields and paired control

At each target layer 10--15 the mean fields are

\[
 {\phi_l\,W_l^T d\mu_{3,l-1}\over \sigma_l},\qquad
 {\phi_l\,W_l^T d\mu_{33,l-1}\over \sigma_l},
\]

while the variance fields are the incoming preactivation providers

\[
 {(W_l^{\circ2})^Tdv_{3,l-1}\over\sigma_l^2},\qquad
 {(W_l^{\circ2})^Tdv_{33,l-1}\over\sigma_l^2}.
\]

No unproved variance `gate^2` is inserted.  These are old-owned local L3 and
L3-squared actions exposed through a new product-before-transport interface;
they are predictor inputs and never receive direct additive value credit.

Both arms have the same 86-input architecture and retain the same local22 and
parent60 coordinates.  They share initialization, fold-J payload, fit rows,
q target and q-RMS, row order, 256-node samples, optimizer, 13-epoch cosine
schedule, and endpoint scoring.  `source_zero` zeros only the four normalized
new coordinates; it does not zero the parent bank.  The frozen continuation
law is exactly

\[
 {M_{82}-M_{\rm on}\over M_{82}-M_{q^*}}\ge .08,
 \qquad M_{\rm on}<M_{\rm zero},
 \qquad L^q_{\rm on}<L^q_{\rm zero}. \tag{101.1}
\]

Here (M_{82}=3.0023717906766396\,10^{-8}) and
(M_{q^*}=6.1455295488600225\,10^{-9}), so the equivalent endpoint threshold
is (2.8113462838133887\,10^{-8}).  The isolated zero-to-oracle recovery is
reported only as a diagnostic and is not a silently added gate.

### 101.2 Target-free gate and transaction ordering

Before claim, after the exact release has been validated, the runner rehashes
all 688 signed and 688 absolute/squared cache rows and builds every new field
in RAM.  Its frozen Schur diagnostic conditions the normalized new4 on a
constant, the exact local22, and the normalized parent60 on 32 fixed fit ranks
and fixed epoch-zero nodes.  Thus it tests new4 given the full parent82, not
merely given the message bank.  A zero conditional rank or nonfinite field
writes an exclusive, file-and-directory-fsynced
`PRETARGET_TERMINAL_CLOSURE.json`; it cannot be retried.

Only after this gate does the process create the durable `O_EXCL` claim.  The
target bundle is byte-hashed before claim solely for integrity, with zero
semantic values decoded; semantic observer-fit values are returned only after
claim.  Any later exception writes a separate durable
`FOLD0_TERMINAL_FAILURE.json`.  Pass holds for separate folds-1--3 authority;
failure closes with no retry.  The release permits only role `observer_fit`,
columns `mlp_id,all_layer_means`, paired fold 0, and target-blind cache reads.
All ablation, remaining-fold, all-data-fit, name, score, network, estimator,
package, integration, meter, board, and submission authorities are false.

### 101.3 Static resources and independent receipt

The standalone W-squared fallback costs 56,850,432 arithmetic operations per
network, 0.0264859779555447 dense-GEMM equivalent, with 4,358,144 explicit
peak bytes.  The all-688 in-memory parent60 and new4 arrays occupy respectively
1,014,497,280 and 67,633,152 bytes.  Each arm has 1,677 updates and 10,303,488
sampled node-layer examples; the hard process RSS gate is eight GiB.

The independent receipt is

```text
INDEPENDENT_H4_RUNNER_AUDIT.json  a5010c9595885ff8a944464c0acb16ea82b2c8f2af8aebfc0c6ad34cc138391a
```

This audit opened zero semantic target values, zero cache rows, zero estimator
calls, and launched no training or scoring run.  It hashed two cache manifests
read-only.  The independent cubic runner audit subsequently passed at SHA-256
`341785c61a6746d17760d3fb5e1969552420fe3c4db0f2d2beec313643f7e739`,
and the target-free preflight is READY.  Both release interlocks are therefore
satisfied; this section does not itself launch the process.

### Graph split

- `P2R0G19B_OWNED_L3_L3SQUARE_RUNNER_ARCHITECTURE` — **PASS static**, exact
  parent82 + new4, 86/256x3/155398.
- `P2R0G19B_OWNED_L3_L3SQUARE_PAIRED_CONTROL` — **PASS static**, only new4 is
  zeroed and every training/supervision coordinate is paired.
- `P2R0G19B_OWNED_L3_L3SQUARE_FULL82_SCHUR` — **PASS design / MEASURED AT
  EXECUTION**, target-free full-parent conditioning before claim; zero rank is
  terminal.
- `P2R0G19B_OWNED_L3_L3SQUARE_RELEASE_SCOPE` — **PASS exact-byte authority**,
  cache/target chronology and every broader false flag bound.
- `P2R0G19B_OWNED_L3_L3SQUARE_H4_AUDIT` — **PASS**, receipt above.
- `P2R0G19B_OWNED_L3_L3SQUARE_CUBIC_AUDIT` — **PASS**, receipt
  `341785c6...`.
- `P2R0G19B_OWNED_L3_L3SQUARE_FOLD0_SENTINEL` — **READY UNDER NARROW
  RELEASE / NOT LAUNCHED BY THIS LANE**, durable claim pending.

### 101.4 Durable claim landed

The released process subsequently created

```text
FOLD0_RUN_CLAIM.json  09d6d1b0bc0b4e0271eb06b682d904998a6aab6fcfc2757b930f10b384902901
```

at `2026-08-31T17:29:49.869772Z` under PID 35716.  It binds the final runner,
release, claim contract, preregistration, both independent audits, frozen
generator, parent result, and target-bundle hash.  It records exactly 1,376
target-blind cache rows verified, 992,217,088 native bytes, zero derived-cache
writes, zero semantic target values decoded before claim, and minimum
full-parent82 conditional Schur rank four.  Neither result, model,
post-claim-failure, nor pretarget-closure artifact existed at the claim audit.

The independent claim receipt is

```text
INDEPENDENT_RUN_CLAIM_AUDIT.json  624e9293ed60a3d45223bbad4052949fd8454e98de7587b027c5d9c5971f6730
```

Therefore `P2R0G19B_OWNED_L3_L3SQUARE_FOLD0_CLAIM` is **PASS LIVE** and
`P2R0G19B_OWNED_L3_L3SQUARE_FOLD0_SENTINEL` is **ACTIVE, RESULT PENDING**.
This lane did not launch or interfere with the bound process.

## 102. Four-field terminal result: paired sign passes, material gate fails

The one-shot fold-0 transaction completed normally.  Its immutable roots are

```text
FOLD0_RESULT.json  d84c5831d3392961d04fcf23323794ca75858190924672aa21c4e496731b98f5
FOLD0_MODELS.pt    1d464e4fc445a65df20b889e0ce652c1eafce26288dd10d802dbc51a304862dc
FOLD0_RUN_CLAIM   09d6d1b0bc0b4e0271eb06b682d904998a6aab6fcfc2757b930f10b384902901
```

There is no post-claim failure or pretarget-closure artifact.  The result is
`FAIL_FOLD0_FOUR_FIELD_SENTINEL__TERMINAL_NO_RETRY`, with the exact decision
`CLOSE_FOUR_FIELD_BUNDLE__DO_NOT_RUN_FOLDS_1_TO_3`.

Independently averaging the 172 finite per-network endpoint rows gives

\[
 M_{82}=3.002371790676639\,10^{-8},\quad
 M_{\rm on}=3.000138043398760\,10^{-8},\quad
 M_{0}=3.0031374128960025\,10^{-8},
\]

\[
 M_{q^*}=6.145529548860023\,10^{-9}.
\]

Hence the frozen parent-to-oracle recovery is only

\[
 {M_{82}-M_{\rm on}\over M_{82}-M_{q^*}}
 =9.3547602707439\,10^{-4}=0.0935476\%,
\tag{102.1}
\]

against the binding 8% gate.  It misses by 7.9064524 percentage points and
lies (1.8879176\,10^{-9}) above the equivalent endpoint threshold
(2.8113463\,10^{-8}).

The paired sign is nevertheless real.  Source-on beats source-zero endpoint
by (2.9993695\,10^{-11}), wins 104/172 networks, and has paired
(z=3.6170943).  It also beats source-zero local-q loss by
(3.1276369\,10^{-12}).  These two strict-sign conjuncts do not rescue the
failed material conjunct.  There is no folds-1--3 run, retry, refit, ablation,
or board composition.

The independent receipt is

```text
INDEPENDENT_H4_TERMINAL_RESULT_AUDIT.json
46927c94a87ce76ceecf1158cb3d40edc7dc1c6a10351b49d5a9eda59c51c1cd
```

It opened no observer-fit/board targets or feature cache, deserialized no model
payload, and launched no estimator/training/scoring call.

### Graph split

- `P2R0G19B_OWNED_L3_L3SQUARE_RESULT_INTEGRITY` — **PASS terminal**, hashes,
  row means, paired signs, and controls reproduce.
- `P2R0G19B_OWNED_L3_L3SQUARE_INCREMENTAL_SIGN` — **PASS weak positive**,
  endpoint and local-q source-on both beat the matched zero arm.
- `P2R0G19B_OWNED_L3_L3SQUARE_MATERIAL_VALUE` — **CLOSED FAIL**, 0.09355%
  recovery versus 8%.
- `P2R0G19B_OWNED_L3_L3SQUARE_REMAINING_FOLDS` — **CLOSED**, no folds 1--3,
  retry, refit, or board test.

## 103. Current-lineage audit of the physical v/k3/k4 fallback

The archived physical-state design at
`public_v34_physical_state_response_sentinel_v1/` is not executable as
written.  Its frozen predicate says it is reachable only if the direct exact
start-5 q oracle is *not* below (10^{-8}); the measured oracle is instead
(4.69852795\,10^{-10}).  The freeze must not be silently rewritten.  The
smallest lawful repair is an append-only post-q-head interlock that treats the
strong oracle as a capacity control and binds the terminal no-retry failures
of the 82-input, direct-transpose, and four-field authorized q spellings.  It
also binds the fact that the 210-channel cross-Gram has only a design/prior-art
receipt: no builder, cache, preregistration, release, claim, result, or live
authority.

That new interlock would make one physical-state sentinel **eligible only**.
It would confer no target, capture, training, execution, remaining-network,
meter, package, board, or submission authority.

### 103.1 Literal target-free interface

For preactivation layers (l=1,\ldots,15), the exact 24-input pointwise
interface is

\[
 [\operatorname{native20}_l, k_{4,l}/\sigma_l^4, u^V_l, u^3_l, u^4_l],
\tag{103.1}
\]

where

\[
 e^2_t=\operatorname{raw2}^{\rm uni}_t-
       \operatorname{diag}(M^{\rm pair}_{11,t})_{\rm before\ overwrite},
 \qquad
 u^V_l={(W_l^{\circ2})^T e^2_{l-1}\over\sigma_l^2},
\tag{103.2}
\]

\[
 u^3_l={\operatorname{diag}(K^{\rm CP3}_{21,l})\over\sigma_l^3},
 \qquad
 u^4_l={K^{\rm CP4}_{4,l}\over\sigma_l^4}.
\tag{103.3}
\]

These objects are literal in source
`public_d11_util_halving_v1/estimator_cap.py` at SHA
`ace56bf60b267666e4f2a6d8ae772a089f234ebc725f34a4798b2f9e8284cbc2`:

- `raw2` and the unsewn pair `m11` coexist at lines 1458--1483;
- `cp_pre_k21` exists before its cubic-star addition at 1801--1812;
- `cp4_pre_provider_planes[0]` exists before host assembly at 1813--1837;
- total `pre_k4` is live there, but native20 exports total k3 and not total k4.

The final layer has existing `_cp_k3` and `_cp4_read_diagonal`
specializations.  Thus the new interface needs three provider splits plus one
old total-k4 export; it needs no new dense plane.  The provider coordinates are
old-owned decomposition/readout information.  They receive **zero direct
value credit**: only the learned dimensionless fields

\[
 ((v^*-v)/\sigma^2,(k_3^*-k_3)/\sigma^3,(k_4^*-k_4)/\sigma^4)
\tag{103.4}
\]

may emit a response source.  A zeroed learned source must reproduce the exact
candidate base.  Diagonal extraction, provider reads, the squared-weight
push, the shared pointwise head, and the signed-weight response recurrence all
commute with source/target neuron permutations.

Before label access, a capture adapter must prove byte-identical reported
means relative to the uninstrumented source, fixed cast/floor conventions,
finite `[15,1024]` arrays in one preactivation gauge, and correct seam
liveness.  In particular (103.2) must be captured before diagonal overwrite,
not reconstructed from the repaired covariance later.

### 103.2 Exact static bill and missing total receipt

The frozen conservative shell uses 15 squared-weight provider matvecs, 14
nonzero response matvecs, an 899-parameter 24--32--3 head, and 491,520 ReLU
comparisons.  Its basic-operation upper bound is 92,244,992, only
(4.19482\,10^{-5}) of (2^{41}), with 256 KiB incremental streaming workspace.
The first Gaussian reset has (e^2_0=0), so a separately frozen exact-zero
elision reduces the shell to 28 matvecs and 90,148,864 operations; this
optimization is not inherited automatically.

The incremental shell is plainly cheap, but the old util-halving census does
not bind the present source hash.  A release still needs an exact FlopScope
total (\le2^{41}) and an RSS receipt for the **modified complete estimator**.
Static incremental arithmetic alone is not a deployment meter.

### 103.3 Label, chronology, and freshness HOLDs

The ten existing P=8,000,000 MC files do contain the required
`mc_pre_v,mc_pre_k3,mc_pre_k4,mc_post,mc_se` arrays in `[16,1024]` float64
gauge.  Their preliminary fit-then-hold ordered hash digest is
`4294ccde300e9f599d65ea4e7cd682eb3b428bb0350403aa58916161190c5dcf`.
This is not yet a release label manifest.  Candidate-specific host-state
captures, exact layer alignment, input/loss normalizers, optimizer, learning
rate, epochs, sample order/RNG, dtype, terminal checkpoint, response floor
convention, and source-zero tolerance are all unspecified in the old freeze.

A future one-shot transaction must claim before semantic label decode, train
on the nine fixed fit ranks only, freeze the model, and only then decode the
held response.  The original cosine (.983459), endpoint-RMS
(3.79871\,10^{-6}), component-sign, finiteness, and noise-corrected
(<10^{-8}) gates remain conjunctive.

There is also a binding interpretation caveat: rank 35 and all ten MC truth
rows have already appeared throughout public oracle-development receipts.
Rank 35 can therefore be a deterministic development mechanism replay, not a
fresh confirmatory held test.  A package/value claim requires either a newly
hash-selected network with a separately authorized MC state label or an
explicit downgrade of the rank-35 result to descriptive evidence.

Machine-readable receipts are

```text
POST_QHEAD_PROVIDER_SPLIT_AUDIT.json
22f919ca62775faac26a7a84b6a730483e036c515da7505b6c1622b4a2421916

INDEPENDENT_CURRENT_LINEAGE_REOPEN_AUDIT.json
ec6808df8b7f9863c7dd7cba5ef381384919ce7fd7f910cc22b07140799448a4
```

The independent audit performed no estimator, training, scoring, or cache
construction.  It inspected one already-public MC archive's array schema and
computed no truth statistic.

### Graph split

- `P2R0G19B_V34_ORIGINAL_INTERLOCK` — **CLOSED FALSE**, exact-q oracle passed;
  old freeze is nonexecutable.
- `P2R0G19B_V34_POST_QHEAD_INTERLOCK` — **SOLVED REQUIREMENTS / NOT ISSUED**,
  append-only eligibility receipt binding the three q terminal results.
- `P2R0G19B_V34_PROVIDER_INTERFACE` — **PASS static literal**, three splits
  plus total-k4 export at synchronized D11 seams.
- `P2R0G19B_V34_PROVIDER_CAPTURE` — **HOLD**, adapter/parity/cast/gauge manifest
  absent.
- `P2R0G19B_V34_TRAINING_LABEL_MANIFEST` — **HOLD**, existing public truth is
  capacity evidence, not a candidate-specific transaction manifest.
- `P2R0G19B_V34_TRAINING_LAW` — **HOLD**, optimizer/schedule/normalization and
  chronology are not frozen.
- `P2R0G19B_V34_COST` — **PASS incremental static / HOLD total meter**, less
  than 0.00420% of (2^{41}) incrementally.
- `P2R0G19B_V34_RANK35_FRESHNESS` — **DESCRIPTIVE ONLY**, not a fresh held
  confirmation.
- `P2R0G19B_V34_EXECUTION` — **HOLD NO AUTHORITY**.

## 104. V34 V2R1 confirmatory firewall: noncircular four-row selector, P8M-times-two uncertainty, and target-last raw gate

The V2 provider algebra remains unchanged, but its rank-35 sentinel is not
confirmatory: all ten old P8M state rows, the rank-35 exact response, and its
LOO betas were already exposed.  V2R1 therefore uses the nine fixed public
state rows 100,150,200,250,300,400,500,600,650 for fitting only, excludes
rank35 entirely, and selects confirmation from the untouched
`observer_selection` role.  The full fit row/weight/truth hash inventory is in
`DATA_PROTOCOL_FREEZE_V2R1.json`.

### 104.1 Winner-free selection and frozen panel

To avoid selector circularity, a standalone winner-free specification was
written and hashed before any winner-bearing receipt.  Its exact preimage is

```text
v34-v2r1-confirmatory-held-v1|whest-phase2-network-split-v1|SPLIT_SHA|GLOBAL|MLP|ASSIGNMENT_SHA|WEIGHTS_SHA
```

in canonical ASCII with no terminal newline.  It binds the immutable split
SHA, V2 hashes, and all three terminal q-result hashes.  The candidate set is
all 88 `observer_selection` rows, no post-role exclusion; its global-rank
ordered commitment is
`3ee0298e1939f08601fae2c4b468ff51dfe1437884cd7fc77158a5b672bab6bc`.
The four smallest digest rows are

```text
panel 0  digest 005767ad...  global757  role45  mlp887
panel 1  digest 01576725...  global746  role34  mlp288
panel 2  digest 06443609...  global780  role68  mlp780
panel 3  digest 0bc088ff...  global759  role47  mlp449
```

Row757 is fail-fast.  Failure or insufficient MC resolution stops without
replacement; a pass unlocks only the next predeclared row.  One row is
selection evidence only.  Package evidence requires four individual passes
and the frozen aggregate gate.

### 104.2 Exact MC and finite-P gates

Each held row has two independent PCG64DXSM streams, exactly P=8,000,000 per
stream.  Each stream retains 16 consecutive 500,000-sample raw-moment blocks,
so the only uncertainty replicates are the 32 blocks, never the 1024 output
coordinates.  The eight exact seed digests and unsigned integer seeds are
frozen in the protocol receipt.

For deterministic endpoint output (y) and the 32 block endpoint means (T_b),
the diagnostic raw MSE is the off-diagonal U-statistic

\[
 M_U(y)={1\over32\,31\,1024}\sum_{a\ne b}
        \langle T_a-y,T_b-y\rangle .
\tag{104.1}
\]

Its pseudovalues are (P_b=32M_U-31M_{U,-b}) and
(\operatorname{SE}=s(P)/\sqrt{32}).  Source-on versus zero uses the linear
block difference

\[
 D_b={\|T_b-p\|^2-\|T_b-p_0\|^2\over1024},
 \qquad \bar D+4s(D)/\sqrt{32}<0.
\tag{104.2}
\]

EDG2 is nonlinear in the sampled moments, so no finite-P unbiased-response
claim is made.  Every response scalar is recomputed from both 16-member
delete-one families.  With (\theta_{A,-b}) and (\theta_{B,-b}),

\[
 V_J={15\over16}\sum_b(\theta_{A,-b}-\bar\theta_{A,-})^2
    +{15\over16}\sum_b(\theta_{B,-b}-\bar\theta_{B,-})^2,
\tag{104.3}
\]

\[
 \theta_J=\theta+15(\theta-\bar\theta_{A,-})
                    +15(\theta-\bar\theta_{B,-}).
\tag{104.4}
\]

Both raw (\theta) and bias-corrected (\theta_J), each with adverse
(4\sqrt{V_J}), must pass the same oracle, cosine, RMS, and component-sign
threshold.  A two-replica midpoint/radius additionally lower-bounds cosine
and component dots and upper-bounds response RMS.  Negative cross norms or
errors, mathematical-boundary crossing, disagreement, or a wide P8M-times-two
interval is `MC_RESOLUTION_INSUFFICIENT`: no clamp, sample extension, or
retry.

### 104.3 Official endpoint target is last

MC supplies physical state truth, oracle capacity, EDG2 bias/stability,
replication, cosine, RMS, component signs, and the raw diagnostic.  Only after
the model, normalizers, provider capture, both MC payloads, compiled response,
and every preceding decision hash freeze may an exclusive one-use reader open
the current row's official `all_layer_means`.  The authoritative raw gates are

\[
 {1\over1024}\|p-y_{\rm off}\|^2<10^{-8},\qquad
 \|p-y_{\rm off}\|^2-\|p_0-y_{\rm off}\|^2<0.
\tag{104.5}
\]

Official-target unavailability is terminal, not an MC fallback.  The
alternative 679 non-fit `observer_fit` endpoint score is not authorized here;
moreover rank35 inside that set is model-held but not globally state-unexposed
(the strict fresh count would be 678).  The chosen package path is the full
four-row target-last panel.

### 104.4 Receipts and independent Matrix audit

```text
FRESH_HELD_SELECTOR_SPEC_V2R1.json
589e3d8c4606cc4b57081458b9197370e77753f2a1fb9d1b247d1eac2de6c0ca

FRESH_HELD_SELECTION_RECEIPT_V2R1.json
22be0d94054cb026375709ea78d9f2b9586325405876ebed5126b681a7e2770f

CONFIRMATORY_DATA_PROTOCOL_V2R1.md
f978719aacfde1eeb94b94513cf020a806dd5f5afb1c3aed5df43569769b37bd

DATA_PROTOCOL_FREEZE_V2R1.json
207756fc08d535ffacb32ae5e3bd0b9b746e1253c096786e3ed4a6a90196d18a
```

The independent audit of Matrix's stable repair/selection/freeze bytes
(`c7e4aafa...`,`5f6f4c43...`,`f816c400...`) is

```text
INDEPENDENT_H4_V2R1_PROTOCOL_AUDIT.json
9d68732c115c9b7590bec1e95d8f5586c1201a4b42908231785c9bbb1f2c3983
```

It passes the static stage-1 data firewall and formulas, but holds every data
or execution action until append-only selector, exact seed/generator,
capture/parity, fit, held, official-target reader, claim, full meter, and RSS
bindings land.  This audit opened no target or array payload and performed no
generation, estimator, fit, or score.

### Graph split

- `P2R0G19B_V34_V2R1_SELECTOR_SPEC` — **PASS**, winner-free immutable selector.
- `P2R0G19B_V34_V2R1_SELECTION` — **PASS target-blind**, ordered panel frozen.
- `P2R0G19B_V34_V2R1_FIT_SOURCE` — **PASS descriptive inventory / HOLD release**.
- `P2R0G19B_V34_V2R1_MC_LAW` — **SOLVED protocol / HOLD implementation**.
- `P2R0G19B_V34_V2R1_EDG2_RESOLUTION` — **SOLVED 4SE gate / empirically unknown**.
- `P2R0G19B_V34_V2R1_OFFICIAL_TARGET_LAST` — **SOLVED chronology / SEALED**.
- `P2R0G19B_V34_V2R1_ROW757` — **HOLD NO AUTHORITY**, fail-fast stage 1.
- `P2R0G19B_V34_V2R1_PANEL` — **HOLD NO AUTHORITY**, four passes required.
- `P2R0G19B_V34_V2R1_EXECUTION` — **HOLD**, no release or claim exists.

## 105. V34 V2R1 release state machine and one-shot firewall requirements

The static protocol is not an execution grant.  The minimal lawful
implementation is a sequence of mutually isolated capabilities:

```text
FIT_RELEASE / FIT_CLAIM
        -> MODEL_HASH_FREEZE
        -> HELD_GENERATION_RELEASE_q / CLAIM_q
        -> MC_SEAL_q
        -> HELD_SEMANTIC_RELEASE_q / CLAIM_q
        -> RESPONSE_HASH_FREEZE_q
        -> OFFICIAL_TARGET_LAST_RELEASE_q / CLAIM_q
        -> ROW_TERMINAL_RESULT_q
        -> (only on PASS) q+1
        -> PANEL_AGGREGATE_RESULT.
```

Every claim must be created with `O_CREAT|O_EXCL`, written and file-fsynced,
then followed by a parent-directory fsync **before** newly authorized semantic
payload access.  Startup aborts if any claim, preclaim closure, owned failure,
result, or successor-stage artifact exists.  After release issuance even a
target-blind preflight failure writes a durable preclaim terminal closure;
after claim any exception writes one owned terminal failure.  No artifact is
overwritten.  Changed bytes require a new versioned release, never a retry.

The fit claim can see only the exact nine public physical-state labels and
their target-blind rows.  It freezes the current-D11 captures, parity receipt,
899-parameter model, normalizers, 64-epoch/2176-step transcript, complete
FlopScope total, and RSS.  The held generator sees only one row's weights,
generator, and two precommitted seeds; it may bind the model-freeze hash but
cannot read model bytes, predictions, labels, metrics, or official targets.
Its MC seal binds both P8M completions and all 32 raw blocks.

The semantic worker may read the frozen model/capture and sealed MC, but no
official target.  It freezes (p,p_0), both physical response replicas,
component responses, all raw and jackknife metrics, and a `PASS_REQUIRED_FOR_TARGET_LAST`
status.  The final reader sees only immutable (p,p_0) plus the single current
row's `all_layer_means`; it cannot see the model, MC, state labels, alternate
predictions, or any other target row.  It computes only the two raw MSEs and
their difference.  Panel aggregation reads only four immutable PASS receipts.

The machine-readable requirements are

```text
INDEPENDENT_RELEASE_FIREWALL_REQUIREMENTS_V2R1.json
57a6041808239723989b51b9014d5f46418ef4aed6da6b50dfb61f0bff0a98bc
```

One current binding defect blocks release construction:
`V2R1_AUDIT_BINDING.json` binds the earlier cubic-audit hash `bb13c7ed...`,
while the current cubic receipt hashes `2cdb2d26...`.  It needs an append-only
rebind; silently replacing the binding is forbidden.

### Graph split

- `P2R0G19B_V34_V2R1_FIT_CLAIM` — **SPECIFIED / HOLD bytes and release**.
- `P2R0G19B_V34_V2R1_MODEL_FREEZE` — **SPECIFIED / HOLD fit result**.
- `P2R0G19B_V34_V2R1_HELD_GENERATION` — **SPECIFIED isolated / HOLD**.
- `P2R0G19B_V34_V2R1_MC_SEAL` — **SPECIFIED exact P8Mx2 / HOLD**.
- `P2R0G19B_V34_V2R1_SEMANTIC_CLAIM` — **SPECIFIED target-free / HOLD**.
- `P2R0G19B_V34_V2R1_RESPONSE_FREEZE` — **SPECIFIED / HOLD**.
- `P2R0G19B_V34_V2R1_TARGET_LAST_CLAIM` — **SPECIFIED least privilege / HOLD**.
- `P2R0G19B_V34_V2R1_AUDIT_BINDING` — **HOLD stale hash**, append-only rebind required.

## 106. Exact-byte V34 V2R1 release implementation audit

The single declared stable implementation set

```text
IMPLEMENTATION_MANIFEST_V2R1.json
efb093d59000f76be614c9268754fc589c09e243ac891b1902c43a633f332a22

IMPLEMENTATION_FREEZE_V2R1.json
5a0d937711794ad7d53e1449faf54f67b49d21c110a57b2533fece39ae82f6eb

IMPLEMENTATION_STATIC_VERIFICATION_V2R1.json
c395adc76b149defc04b6569ea494b0f2ddf9b83757e5a986d9e1fdba3d0c870
```

is hash-closed, target-free, and statically coherent at the provider,
two-stream statistic, target-last, and fixed-panel levels.  It nevertheless
cannot be released.  The independent exact-byte audit found eleven blocking
implementation defects:

1. the official release keyset omits the deploy-resource digest that the
   official scorer requires;
2. the authenticated transaction then omits that digest from its returned
   mapping while result construction indexes it;
3. the held-semantic release forbids metering while the stage always meters;
4. the deployed head evaluates in float64 where frozen fit/held inference
   casts the normalized input to float32;
5. fit hashes independent audits but never parses or requires PASS;
6. arbitrary unknown permission keys are accepted;
7. the fit model freeze can bind a different release hash from the release
   consumed by the durable claim;
8. the still-bound old MC seed protocol contradicts the corrected R1 seed
   law and receipts;
9. the frozen static receipt records NumPy 2.4.6 rather than the required
   NumPy 1.26.4 test environment;
10. `MODEL_HASH_FREEZE` lacks the preregistered complete modified-estimator
    FlopScope/RSS receipt; and
11. the pre-target later-row guard omits future panel-continuation releases.

The full receipt is

```text
public_v34_physical_state_response_sentinel_v2r1/
  INDEPENDENT_H4_RELEASE_IMPLEMENTATION_AUDIT.json
05e969498fce33b06d60bdfb8234ab80d69da3056c8904f709926e9ed2f1d68b
```

No target, production array, MC payload, estimator, capture, fit, or score was
opened or run.  All fit, held, official, continuation, aggregate, package,
board, submission, and network authority remains **HOLD** pending one new
immutable R1 set and three independent PASS audits.

### Graph split

- `P2R0G19B_V34_V2R1_HASH_CLOSURE` — **PASS static**.
- `P2R0G19B_V34_V2R1_TARGET_PURITY` — **PASS static**.
- `P2R0G19B_V34_V2R1_STAT_SPELLING` — **PASS static**.
- `P2R0G19B_V34_V2R1_RELEASE_FSM` — **HOLD 11 defects**.
- `P2R0G19B_V34_V2R1_EXECUTION` — **HOLD, zero authority**.

## 107. Frozen H4 R1 PASS schema and adversarial transaction oracle

Before any replacement implementation was inspected, the H4 lane froze an
implementation-independent acceptance contract.  A future H4 PASS is lawful
only when one stable R1 byte set supplies all eleven §106 repairs plus exact
stage runtimes, same-PID claims, complete evidence roots, the full global
successor-absence matrix, and downstream rehashes of every fit row, weight,
truth, capture, label, dataset, model, normalizer, transcript, and complete
modified-estimator resource receipt.

The runtime contract is deliberately stage-specific:

```text
fit / held-generation / held-semantic / panel
  Anaconda Python 3.11.5, NumPy 1.26.4; Torch 2.7.0 where imported

official-target-last
  data_gate Python 3.11.5, NumPy 2.4.6, PyArrow 25.0.1
```

Both claims bind their actual executable path and executable SHA.  The two
environment paths currently resolve to the same CPython executable bytes, but
their installed package contracts differ; swapping the stage interpreter is a
hard failure.

The frozen receipts are

```text
H4_IMPLEMENTATION_PASS_AUDIT_SCHEMA_R1.json
d932c307a3e65aee6ea456837c3a2230af65a72ab6bfdd788e3cf13ee4a741fd

h4_r1_transaction_contract.py
1a363ba29ac0fe76345644f0996ac46d75c406bf024180de5b59286f178e4c35

verify_h4_r1_transaction_contract.py
169690be620e21dd3db14c279df0d02af22b1aaf43d69e4e5fd56f78cace47b4

H4_R1_TRANSACTION_CONTRACT_STATIC_VERIFICATION.json
53715147e56eaf367ae3e7ed4b93f6afdc933f5c670196deda6a533d01ed9a9d
```

The reference oracle accepts two canonical positive transactions and rejects
all 59 preregistered mutations under both stage-runtime paths.  Mutations cover
resource-digest deletion at release/actual/return/result, permission smuggling,
meter denial, float64 head drift, HOLD audit admission, release/claim splice,
old seed law, wrong runtime/interpreter/PID, missing resource evidence, every
cap boundary, early continuation/stage/aggregate authority, incomplete global
path enumeration, all fit-manifest rehash classes, missing state/component
evidence, panel reorder, and population overclaim.

This is a reference contract only.  It opened no repair draft, target, array,
estimator, capture, MC stream, fit, or score and grants no authority.  Once
Matrix declares one stable R1 set, the lane must extract the evidence from the
actual bytes, exercise the actual repaired transaction path, and publish a new
append-only PASS/HOLD receipt.

### Graph split

- `P2R0G19B_V34_R1_H4_PASS_SCHEMA` — **FROZEN**.
- `P2R0G19B_V34_R1_REFERENCE_POSITIVE` — **PASS 2/2 twice**.
- `P2R0G19B_V34_R1_REFERENCE_ADVERSARIAL` — **PASS reject 59/59 twice**.
- `P2R0G19B_V34_R1_MATRIX_BYTES` — **WAIT stable declaration**.
- `P2R0G19B_V34_R1_H4_FINAL_AUDIT` — **HOLD pending actual bytes**.

## 108. Acyclic H4 R1 acceptance transaction

The first §107 reference snapshot was not production-lawful: it required the
future H4 candidate-audit file digest inside the very transaction that the H4
file would contain.  That is an impossible self-hash.  Its four hashes remain
an immutable historical receipt but are **HOLD for production wiring**.

The append-only R1A correction is fully independent.  The implementation
freeze and H4 evidence snapshot bind only fixed Cubic, Direct, and H4 contract
artifacts (schema, validator, adversarial verifier, and static contract
receipt).  They contain no Cubic, Direct, or H4 candidate-audit digest.  The
final H4 candidate audit binds the canonical evidence-snapshot digest and the
fixed contract root, but has no own-digest field and does not bind the other
two candidate audits.  Only the later fit release hashes, parses, and requires
PASS from all three independently completed candidate-audit files.

The frozen acyclic receipts are

```text
H4_IMPLEMENTATION_PASS_AUDIT_SCHEMA_R1A_NONCIRCULAR.json
0cb28198163915937e375949dc0b1fb3d19ce45e39651d3b472d3f1a8cfbdace

h4_r1a_noncircular_transaction_contract.py
313bcc6d01c8bc89281ee85e4e2ae0e3ec7d710b11792c76833c6a71056e8da8

verify_h4_r1a_noncircular_transaction_contract.py
41d5da70276276303d604cace8541bb2d5c646bd1c08e9085a5ea7cd15476a6f

H4_R1A_NONCIRCULAR_TRANSACTION_CONTRACT_STATIC_VERIFICATION.json
683fcd6f557a903f0faf62ea27457280d9703beca586f6d71ebe34d468ea51e8
```

Both pinned stage runtimes accept two canonical evidence snapshots and two
canonical final receipts, and reject all 65 preregistered mutations.  The
injected H4 audit, injected audit digest in the full evidence root, injected
final self-digest, changed snapshot/root/contract digest, permission and PID
drift, fit splice, missing manifest rehash, resource-cap violation, early
successor artifact, wrong interpreter, and panel/population overclaim all
fail closed.  The fixed Cubic and Direct contract bytes and the imported base
H4 schema/validator/verifier/static receipt are independently rehashed, not
merely trusted from constants; changing the transitive base validator is a
separate rejected mutation.

The eventual candidate receipt filename and schema are exactly

```text
INDEPENDENT_H4_R1_RELEASE_IMPLEMENTATION_AUDIT.json
v34-v2r1-independent-h4-release-implementation-audit-r1
```

This work inspected no moving Matrix repair draft, target, array, cache row,
weight, feature, fit truth, MC block, estimator, capture, training, or scoring
payload and created no claim, release, or authority.  It is ready to audit one
explicit `STABLE FINAL` R1 declaration and nothing else.

### Graph split

- `P2R0G19B_V34_R1_H4_R1A_SCHEMA` — **FROZEN acyclic**.
- `P2R0G19B_V34_R1_H4_R1A_POSITIVE` — **PASS 2 snapshots + 2 receipts twice**.
- `P2R0G19B_V34_R1_H4_R1A_ADVERSARIAL` — **PASS reject 65/65 twice**.
- `P2R0G19B_V34_R1_H4_FINAL_AUDIT` — **WAIT explicit STABLE FINAL bytes**.
- `P2R0G19B_V34_R1_EXECUTION` — **HOLD, zero authority**.

## 109. Split-runtime capture, training, meter, and held FSM

Whest/FlopScope D11 capture cannot run inside the Anaconda training or
semantic workers.  The exact acyclic replacement is therefore

```text
DATA_GATE FIT_CAPTURE (nine target-free rows)
  -> ANACONDA FIT_TRAIN
  -> DATA_GATE FIT_RESOURCE_METER (nine full source-on paths)
  -> ANACONDA MODEL_HASH_FREEZE (hash-only)

per held row q:
DATA_GATE HELD_CAPTURE_METER
  -> ANACONDA HELD_GENERATION_A
  -> ANACONDA HELD_GENERATION_B
  -> ANACONDA HELD_SEMANTIC_SCORE
  -> DATA_GATE OFFICIAL_TARGET_LAST
  -> transaction-only continuation
  -> ...
  -> ANACONDA PANEL_AGGREGATE.
```

The frozen machine-readable schema is

```text
H4_SPLIT_RUNTIME_STAGE_MAP_ADDENDUM_R1.json
f7f5b5c47437617c23fb559efa91711dbfe4b2d29decae4ea3af8c341aff7bfd
```

It fixes the stage names and order, 31-key permission universe and exact true
sets, release/claim/result/preclaim/failure keysets, artifact templates, seal
payload keysets, O_EXCL plus file/directory fsync law, same-PID recheck,
complete global successor absence, terminal/no-retry behavior, and data
firewalls.  Every fit capture and meter has cardinality nine.  For each row,
the Anaconda uninstrumented, data-gate uninstrumented, and data-gate
FlopScope-instrumented complete `[16,1024]` float32 source-on outputs must be
C-order byte-identical.  Trainer/deploy/public head intermediates are likewise
bitwise equal.  The complete meter includes parent, descendants, and backend
work; endpoint-only, head-only, or inferred streaming receipts are invalid.

The append-only invocation-path reconciliation is

```text
H4_RUNTIME_ALIAS_RECONCILIATION_R1.json
408bcde33b94b1e273dbaa1ff2ba52f2ad73bd3787fb28346e5916af717949f4
```

Anaconda numeric claims must invoke
`/Users/matt/anaconda3/bin/python3.11`; `/python` and `/python3` are discovery
aliases only.  Data-gate claims must invoke the exact `.venv/bin/python`.
Although both resolve to binary SHA `6116d7bc...`, invocation path,
`sys.prefix`, package versions and locations, lock digest, PID, nice value,
and thread environment all remain independently binding.  This addendum also
supersedes the stale Cubic alias receipt `5ba0...` with `f676...` without
changing any f7f5 stage-map clause.

No moving repair byte, target, array, cache row, estimator, capture, meter,
training, or scoring payload was opened or executed, and no authority was
created.

### Graph split

- `P2R0G19B_V34_R1_SPLIT_STAGE_MAP` — **FROZEN**.
- `P2R0G19B_V34_R1_RUNTIME_ALIAS` — **FROZEN append-only**.
- `P2R0G19B_V34_R1_FIT_CAPTURE` — **SPECIFIED / HOLD release**.
- `P2R0G19B_V34_R1_FIT_TRAIN` — **SPECIFIED / HOLD predecessor**.
- `P2R0G19B_V34_R1_FIT_METER` — **SPECIFIED / HOLD predecessor**.
- `P2R0G19B_V34_R1_MODEL_FREEZE` — **SPECIFIED hash-only / HOLD**.
- `P2R0G19B_V34_R1_HELD_CHAIN` — **SPECIFIED / HOLD model freeze**.

## 110. Direct/H4 split-stage least-privilege reconciliation

The later frozen Direct 28-node transaction contract exposed two overly broad
permissions in the immutable H4 stage map.  First, `FIT_CAPTURE` is a
capture-only transaction: it may use `flopscope.numpy` as the accepted D11
arithmetic backend, but it may not open `BudgetContext`, invoke the resource
meter, or write a capture-resource receipt.  Only `FIT_RESOURCE_METER`, after
training, owns the nine complete modified-estimator resource receipts.
Second, `HELD_SEMANTIC_SCORE` is a statistics-only transaction over the
sealed held endpoints and the two sealed MC streams.  It may rehash the model
freeze as a predecessor, but it may not open model/normalizer arrays, import
the estimator/head, or recompute predictions.

The append-only exact-keyset supersession is

```text
H4_DIRECT_STAGE_SPLIT_CONFLICT_SUPERSESSION_R1.json
71eefb09d8a7115183e78efd659aa4e720cca1846ea1f8beebd2f970f68b6c44
```

It binds the immutable H4 stage map `f7f5...`, runtime reconciliation
`408b...`, final Cubic alias/parity ruling `f676...`, and Direct addendum,
verifier, static receipt, and reconciliation `8c70.../9599.../9f61.../81f6...`.
Its full effective permission map is authoritative over the two parent-map
conflicts:

- `FIT_CAPTURE` has standalone D11 capture authority and no meter authority;
- `FIT_RESOURCE_METER` has meter authority and no standalone capture/export
  authority;
- `HELD_CAPTURE_METER` remains the only held stage allowed to deserialize the
  frozen model and execute the full candidate;
- `HELD_SEMANTIC_SCORE` reads only the freeze receipt, sealed endpoint
  artifact, and sealed MC summaries.

Every implementation freeze, release, claim, seal, response freeze, and
downstream result must bind and rehash this supersession in addition to
`f7f5...` and `408b...`.  Missing it, binding the stale Cubic `5ba0...`, or
setting any extra permission is terminal HOLD.

The parent seal schema separately forced a now-illegal
`nine_capture_resource_sha256` field.  The exact-keyset correction is

```text
H4_FIT_CAPTURE_RESULT_KEYSET_SUPERSESSION_R1.json
cde44700aeb28d477ae4e5c36c38cb3a2e07c854586c4de353d35f84cfc91a97
```

It removes every resource field and even a null resource placeholder from
the `FIT_CAPTURE` seal/result.  The `FIT_RESOURCE_METER` result alone requires
the nine-receipt resource root and triple-parity root.  Exact key equality is
mandatory; subset validation is a hard failure.  Every transaction that
binds `71eefb09...` must therefore also bind `cde44700...`.

A subsequent full audit found that f7f5's generic result schema also polluted
all other stages with irrelevant row, stream, resource, parity, or seal
placeholders.  The complete 28-node replacement is

```text
H4_COMPLETE_STAGE_RESULT_KEYSET_SUPERSESSION_R1.json
fd6fe0021e1956bbfc738e7fe3b88f192d6588ea4694676dacf2811773bca30a

verify_h4_complete_stage_result_keysets_r1.py
c03ef7588455205641058b8e8972da374ab7f88dfa3a65499e092698db3cc994

H4_COMPLETE_STAGE_RESULT_KEYSET_STATIC_VERIFICATION_R1.json
3983104291e653202b04bcb6773f75623d830efd86b7bbc56bdc927e65244948
```

It assigns one exact result class to each of the 28 ordered transactions.
Row fields occur only on row transactions; stream fields only on the A/B
generators; resource and parity roots only on their literal owners; and each
seal/freeze field is confined to its producing stage.  Null placeholders,
unknown keys, subset validation, cross-stage fields, wrong rows/streams, and
wrong stage order all fail.  The standard-library verifier accepts 28/28
canonical results and rejects 851/851 adversarial mutations under both exact
Anaconda and data-gate interpreters.

No moving repair byte, target, data/cache array, weight/feature array, MC,
estimator, capture, meter, training, scoring, or network operation was opened
or executed, and no authority was created.

### Graph split

- `P2R0G19B_V34_R1_DIRECT_STAGE_SPLIT` — **FROZEN 28-node contract**.
- `P2R0G19B_V34_R1_H4_DIRECT_CONFLICT_SUPERSESSION` — **FROZEN append-only**.
- `P2R0G19B_V34_R1_FIT_CAPTURE_PERMISSION` — **PASS capture-only**.
- `P2R0G19B_V34_R1_FIT_METER_PERMISSION` — **PASS meter-only**.
- `P2R0G19B_V34_R1_HELD_SEMANTIC_PERMISSION` — **PASS statistics-only**.
- `P2R0G19B_V34_R1_EXECUTION` — **HOLD, zero authority**.

## 111. Minimal descriptive rank-35 V34 value sentinel

The fresh four-row/MC package route was removed from the critical path.  The
remaining question is only whether the fixed 899-parameter physical-state
head has descriptive value on the already-exposed rank-35 development row.
This cannot be called fresh, confirmatory, package, or population evidence.

The exact minimal firewall is frozen in

```text
H4_RANK35_DESCRIPTIVE_SENTINEL_MINIMAL_RELEASE_CHECKLIST_R1.json
d7b0524c383fbf978cd605a92b15dfcf61dcfad28ab299b75ab30092a82dea58
```

There are four transactions in three runtime processes:

```text
DATA_GATE FIT_CAPTURE_9
  -> ANACONDA FIT_TRAIN_MODEL_FREEZE
  -> DATA_GATE RANK35_CAPTURE_OUTPUT_FREEZE
  -> DATA_GATE RANK35_TARGET_SCORE.
```

The first transaction captures exactly the nine frozen fit rows and cannot
open fit labels or anything from rank 35.  The second opens only the nine
pre-state labels, performs the single fixed 899-parameter fit, and freezes the
model/normalizer/transcript before any rank-35 access.  The third then opens
only the target-blind rank-35 manifest and weights, runs the complete pinned
D11 source-zero/source-on candidate, meters it, and durably freezes both
`[16,1024]` outputs.  It cannot open rank-35 state labels or the mean target.
Only after that output freeze may a distinct release and O_EXCL+file-fsync+
directory-fsync claim authorize decoding the already-exposed rank-35 row-15
mean target.

The sole value gate is exact raw endpoint value:

```text
mse0 = mean((float64(source_zero[15]) - float64(target[15]))**2)
mse1 = mean((float64(source_on [15]) - float64(target[15]))**2)
PASS iff mse1 < 1e-8 and mse1 < mse0.
```

Source-zero must reproduce the pinned rank-35 parent bitwise, every output is
finite, and the full modified estimator must pass the `<2^41` FLOP, `<8 GiB`
RSS, and `<120 s` gates.  No MC, new target, alternate row, retry, replacement,
refit, feature/normalizer/weight selection, or package continuation exists.
A PASS is explicitly `DESCRIPTIVE_PASS`; a failure closes these exact bytes.

No moving implementation byte, data/cache array, weight/feature array, fit
truth, rank-35 target/state array, MC, estimator, capture, meter, training,
scoring, or network operation was opened or executed, and no authority was
created.

### Graph split

- `P2R0G19B_V34_R35_FIT_CAPTURE_9` — **SPECIFIED / HOLD release**.
- `P2R0G19B_V34_R35_FIXED899_FIT` — **SPECIFIED / HOLD capture predecessor**.
- `P2R0G19B_V34_R35_TARGET_BLIND_OUTPUT_FREEZE` — **SPECIFIED / HOLD model**.
- `P2R0G19B_V34_R35_EXPOSED_TARGET_SCORE` — **SPECIFIED descriptive-only / HOLD output freeze**.
- `P2R0G19B_V34_R35_MC_OR_FRESH_PANEL` — **CLOSED, absent**.
- `P2R0G19B_V34_R35_EXECUTION` — **HOLD, zero authority**.

## 112. Descriptive-only target and resource scope amendment

The frozen rank-35 row is already fully exposed.  The target-score process
may therefore materialize its full `[16,1024]` `all_layer_means` value, but
only after its durable score claim.  Exactly row 15 and 1024 values may enter
either MSE; the full-array hash and page/row/value disclosure must be
persisted.  Other networks and every preclaim target open remain forbidden.

The descriptive resource conjunct is the complete modified-estimator meter:
strictly below `2^41` FLOPs, 8 GiB RSS, and 120 seconds, with a literal
`float32[16,1024]` return, byte-identical parent rows 0 through 14, and an
exact source-zero parent.  The 256 KiB streaming-workspace proof is deferred
to packaging and conveys no package authority here.

The append-only ruling also freezes stage-specific result keysets.  The
transaction layer lawfully appends `stage`, `release_sha256`, `claim_sha256`,
`terminal`, and `no_retry`; arbitrary additional result fields remain
forbidden.

```text
H4_RANK35_DESCRIPTIVE_SCOPE_AMENDMENT_R1.json
5674c5f21ddd523c4def4d94ef86de6807538081bea006f49e04bbeaae542f62
```

No target, data/cache array, weight/feature array, estimator, capture, meter,
training, scoring, MC, or network operation was opened or executed, and no
execution authority was created.

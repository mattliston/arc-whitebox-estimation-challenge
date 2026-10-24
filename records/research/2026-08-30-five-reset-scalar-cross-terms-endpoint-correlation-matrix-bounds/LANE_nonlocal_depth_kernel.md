# Lane: nonlocal depth kernel, NCQC provenance, and finite-horizon carriers

Date: 2026-08-30  
Mode: mathematical derivation only; no numerical science run was used for this lane.  
Target: determine the smallest causal finite-width closure capable of contributing to raw MSE below `1e-8`, while respecting the dense-matmul budget.

## Executive result

A covariance-only layer-local self-energy is not correct through order `1/n`. Two cubic vertices at depths `m<l` generate a retarded Volterra kernel. At critical He/ReLU scaling the two-line response is marginal, so there is no analytic decay that localizes this kernel.

The contracted cubic-fish/H2 sector nevertheless has an exact causal realization with an auxiliary pair matrix: `O(n^2)` storage and `O(L n^3)` work. That pair self-energy is even under reversal of the cubic vertex and therefore cannot supply the signed all-distinct H3 information required by the raw target. An open H4 source is distinct from its contracted tadpole as well.

The strongest structured extension found in this lane is the shared NCQC-by-singleton module. One propagated NCQC matrix word simultaneously supplies:

- the signed H3 provenance;
- the later H2 cubic-pair return;
- the identity-specific q4(211) branch;
- a forward terminal H4 contraction without an independent reverse-H4 matrix.

This exact one-dense-stream construction costs two dense matmuls per active transition. Across 14 transitions it costs 28 GEMMs. A genuinely observable `q4_perp` word needs another matrix stream unless it is absorbed by an explicitly proved forward quotient. With 21 GEMMs left, a second word may persist for at most ten transitions.

Under the normal-ordered injective Gaussian reset used by the theorem, a truly perpendicular word cannot later be absorbed: chaos grade and matrix-word independence are preserved. The optimal second-stream support is therefore the final ten-transition suffix. Rolling/restarting does not improve it unless a new, explicit, forward-local rank-deficient quotient is proved.

## Raw-value gates

Board raw MSE, oracle raw MSE, and the campaign target are

\[
B=2.94090718356\times10^{-8},\qquad
O=6.24925261423\times10^{-9},\qquad
T=10^{-8}.
\]

The required captured oracle-gap fraction is

\[
\frac{B-T}{B-O}=0.838049367.
\]

The residualized exact-state ladder gives

- variance-only raw: `14.656e-9`;
- variance plus signed k3 raw: `10.286e-9`, a k3 gain of `4.370e-9`;
- variance plus k3 plus k4 raw: `9.2547e-9`, a further k4 gain of `1.0313e-9`.

Consequently, after perfect variance, retained gain fractions must obey

\[
4.370\,\alpha_3+1.0313\,\alpha_4\ge 4.656
\quad\text{(all entries in }10^{-9}\text{ units).}
\]

Perfect H3 still needs at least `27.7%` of the residualized k4 gain. Perfect H4 still needs at least `82.9%` of the signed k3 gain.

For an orthogonal best projection of the oracle source, correlation and retained value energy satisfy

\[
q_r^2=\frac{\text{retained value energy}}{\text{total value energy}}.
\]

Thus:

- `q >= 0.92545` requires at least `85.6458%` retained value energy and permits at most `14.3542%` tail energy;
- `q >= 0.98346` requires at least `96.7194%` retained value energy and permits at most `3.28064%` tail energy.

All ranks and tails in this lane are to be measured after residualizing against the current D11/F2J feature span and whitening by the endpoint MSE-value Gram. Raw tensor Frobenius energy is not the decision metric.

## Causal field theory through order `1/n`

Let

\[
\epsilon=n^{-1/2}.
\]

Write the exact discrete one-layer map as

\[
\mathcal U_l(\epsilon)
=\mathcal A_l+\epsilon\mathcal V_{3,l}
+\epsilon^2\mathcal V_{4,l}+O(\epsilon^3).
\]

The order classes are:

1. `O(epsilon)`: one open signed H3 insertion. It may vanish under a proved ensemble symmetry, but it does not vanish merely because the board problem is conditioned on a realized white-box network.
2. `O(epsilon^2)`: a separated cubic pair at arbitrary `m<l`; a same-layer cubic contact; one quartic contact/tadpole; and an open H4 source.
3. Disconnected Wick terms are removed by cumulant/Hermite normal ordering.

For the contracted cubic-pair sector, define split, pair transport, merge, and local quartic maps `J_l`, `T_l^(2)`, `R_l`, and `Q_l`. The retarded self-energy is

\[
\Sigma^{(33)}_{lm}
=\epsilon^2 R_lT^{(2)}_{l-1}\cdots T^{(2)}_{m+1}J_m,
\qquad m<l,
\]

and the local quartic contact is

\[
\Sigma^{(4)}_{lm}=\epsilon^2Q_l\,\delta_{lm}.
\]

At He/ReLU criticality,

\[
\chi=\sigma_w^2\,\mathbb E[\phi'(Z)^2]=2\cdot\frac12=1,
\]

so the two-line response is not known to decay. With `n=1024`, `L=16`, the natural ordered counts before topology coefficients are

\[
L\epsilon=0.5,
\qquad
\binom{16}{2}\epsilon^2=0.1171875,
\qquad
L\epsilon^2=0.015625,
\]

\[
\binom{16}{3}\epsilon^3=0.0170898,
\qquad
\binom{16}{4}\epsilon^4=0.0017357.
\]

The separated cubic-pair combinatorics are `7.5x` the same-layer contact count, or `15x` if the two orientations are counted separately. A local-only self-energy is therefore not a controlled replacement.

## Pair-field realization and its minimality

The retarded kernel has the causal state-space form

\[
P_{l+1}=T_l^{(2)}P_l+\epsilon J_lu_l,
\]

\[
u_{l+1}=A_lu_l+\epsilon R_lP_l+\epsilon^2Q_lu_l.
\]

Eliminating `P` recovers every separated time-ordered `3 x 3` self-energy insertion. When

\[
T_l^{(2)}(P)=D_l^{(2)}\odot(L_lP R_l^\top),
\]

the pair update uses two dense matmuls plus local masks. It therefore uses `Theta(n^2)` memory and `Theta(L n^3)` work.

For a homogeneous surrogate, eliminating the pair state produces the rational self-energy

\[
\Sigma(z)=\epsilon^2Q+\epsilon^2R(zI-T^{(2)})^{-1}J.
\]

A layer-local self-energy replaces this rational function by a constant and deletes its poles. At criticality the relevant poles need not be separated from the unit circle.

At a finite-horizon cut `k`, form the joint block Hankel operator

\[
\mathsf H_k=[K_{lm}]_{l>k,m\le k},
\]

where the output includes every H2 and H4 readout required from the eliminated H3 sector. The minimal arbitrary time-varying state dimension is

\[
D_*=\max_k\operatorname{rank}(\mathsf H_k),
\]

subject to the usual nested row/column shift compatibility. If one matrix stream has symmetric dimension `d2=n(n+1)/2`, `r_p` streams require

\[
\max_k\operatorname{rank}(\mathsf H_k)\le r_p d_2.
\]

This rank certificate is separate from the compute certificate. A matrix-state transition is `O(n^3)` only if its Van-Loan reshuffling has constant Kronecker rank `kappa`:

\[
T(P)=\sum_{q=1}^{\kappa}L_q P R_q^\top.
\]

Finite Hankel rank without constant Kronecker rank can still hide `O(n^4)` application cost.

The pair self-energy alone cannot recover the required signed cubic source: it is quadratic in the cubic vertex and invariant under `V3 -> -V3`. Likewise, a quartic tadpole only observes contractions of the H4 tensor. It cannot recover a generic traceless open H4 source.

## Normal-ordered H2/H3/H4 hierarchy

Let `u` be the order-one H2 state, `c` the coefficient of `epsilon` in a retained signed H3 subspace, `p` a realization of the complementary H3 kernel, `a` the coefficient of `epsilon^2` in H2, and `s` the coefficient of `epsilon^2` in H4. The coefficient recurrence is

\[
u'=A_2u,
\]

\[
c'=A_3c+J_cu,
\qquad
p'=Tp+J_pu,
\]

\[
a'=A_2a+R_{2c}c+R_{2p}p+Q_2u,
\]

\[
s'=A_4s+R_{4c}c+R_{4p}p+B_4^{\rm sq}u.
\]

Use an orthogonal/provenance partition `J_c=Pi_3 J`, `J_p=(I-Pi_3)J`. Every row reads the old H3/pair state before injecting the current birth. Thus the pair rows contain only `m<l` products. All same-layer second-order contacts belong to the discrete `V4` coefficient in `Q2` and `B4`, and are never injected into `p`.

The pair compression is exact for H2 and H4 simultaneously only if, for all `m<l`,

\[
\begin{bmatrix}
\Pi_2V_{3,l}\\
\Pi_4V_{3,l}
\end{bmatrix}
A_{3,l:m}(I-\Pi_3)V_{3,m}\Pi_2
=
\begin{bmatrix}
R_{2p,l}\\
R_{4p,l}
\end{bmatrix}
T_{l:m}J_{p,m}.
\]

The H4 readout may increase the joint Hankel rank even when the H2 fish alone has an `n^2` realization. That increase is an explicit open-H4 obstruction.

## Minimal target-adjoint block

For a fixed endpoint readout, forward `c`, `a`, and `s` can be eliminated by Duhamel adjoints. Retain the existing baseline `u`, the true memory states `P^alpha`, and backward structured adjoints `Lambda_2`, `Lambda_3`, `Lambda_4`:

\[
P_{l+1}^\alpha
=\sum_{\beta,q}t_{\alpha\beta q,l}
L_{q,l}P_l^\beta R_{q,l}^\top+J_{\alpha,l}u_l,
\]

\[
\Lambda_{r,l}=A_{r,l}^*\Lambda_{r,l+1},
\qquad r\in\{2,3,4\}.
\]

Add a local adjoint source if the endpoint functional is distributed over layers. The endpoint expansion is

\[
y=y_0
+\epsilon\sum_l\langle\Lambda_{3,l+1},B_{3,l}u_l\rangle
\]

\[
\quad+\epsilon^2\sum_l
\left[
\langle\Lambda_{2,l+1},Q_{2,l}u_l+R_{2,l}P_l\rangle
+\langle\Lambda_{4,l+1},B_{4,l}u_l+R_{4,l}P_l\rangle
\right].
\]

This is exact through `epsilon^2` on the retained subspaces. The remaining error consists of H3 projection leakage, H4 projection leakage, joint-kernel factorization error, open H5/H6/rest, and irreducible terms of order `epsilon^3` and higher.

## NCQC and q4 normal-order identities

For centered standard Gaussian `z`, define

\[
Q_M=\frac12(z^\top Mz-\operatorname{tr}M).
\]

The exact trace/Jordan product decomposition is

\[
\mathbb E[Q_MQ_N]=\frac12\operatorname{tr}(MN),
\]

\[
Q_MQ_N
=\frac12\operatorname{tr}(MN)
+2Q_{M\circ N}+U_{M,N},
\qquad
M\circ N=\frac12(MN+NM),
\]

where

\[
U_{M,N}
=Q_MQ_N-\frac12\operatorname{tr}(MN)-2Q_{M\circ N}
=\frac14I_4(\operatorname{sym}(M\otimes N)).
\]

In particular,

\[
U_M=Q_M^2-\frac12\operatorname{tr}(M^2)-2Q_{M^2}.
\]

Adding raw `Q_M^2` to a closure that already carries its scalar and H2 contractions double counts. Only `U_M` belongs to the open H4 lane.

The H4 square inner product is

\[
\langle U_M,U_N\rangle
=\operatorname{tr}(MNMN)
+\frac12[\operatorname{tr}(MN)]^2.
\]

Define a signed structured H3 atom

\[
C(v,M)=I_3(\operatorname{sym}(v\otimes M)).
\]

Its polynomial is

\[
C(v,M)
=(v\cdot z)(z^\top Mz)
-\operatorname{tr}(M)(v\cdot z)-2(Mv)\cdot z.
\]

The corrected exact H3 inner product is

\[
\langle C(v,M),C(w,N)\rangle
=2(v\cdot w)\operatorname{tr}(MN)+4v^\top NMw.
\]

Under a separable Gaussian leg map `K`,

\[
C(v,M)\mapsto C(Kv,KMK^\top),
\]

\[
U_M\mapsto U_{KMK^\top}.
\]

An NCQC family `M_i` has rank `r3` when the whitened unit-by-symmetric-pair matrix whose `i`th row is `svec(M_i)` has rank `r3`. Equivalently,

\[
M_i=\sum_{\alpha=1}^{r_3}v_{\alpha,i}M_\alpha.
\]

The matrix-word span must also be closed under every required conjugation and Jordan product. Rank-one source factorization alone is not autonomous closure: a one-word algebra is exact only if

\[
M^2\in\operatorname{span}\{I,M\}.
\]

Otherwise `M^2` must be routed into an existing H2 sink, added as a second word, or counted as leakage.

For q4(211), define

\[
D(M;v,w)=I_4(\operatorname{sym}(M\otimes v\otimes w)).
\]

The explicit collision-subtracted polynomial is

\[
D(M;v,w)=:(z^\top Mz)(v\cdot z)(w\cdot z):
\]

\[
=(z^\top Mz)(v\cdot z)(w\cdot z)
-\operatorname{tr}(M)(v\cdot z)(w\cdot z)
\]

\[
\quad-2(Mv\cdot z)(w\cdot z)
-2(Mw\cdot z)(v\cdot z)
-(v\cdot w)z^\top Mz
\]

\[
\quad+\operatorname{tr}(M)(v\cdot w)+2v^\top Mw.
\]

It obeys

\[
D(M;v,w)\mapsto D(KMK^\top;Kv,Kw).
\]

## Shared NCQC-by-singleton identity

The key identity in this lane is the Wiener product formula

\[
C(v,M)I_1(w)
=D(M;v,w)
+I_2\left(2v\odot(Mw)+(v\cdot w)M\right),
\]

where

\[
v\odot a=\frac12(va^\top+av^\top).
\]

This follows from

\[
I_3(f)I_1(w)=I_4(\operatorname{sym}(f\otimes w))+3I_2(f\otimes_1w).
\]

The later H2 pair return and q4(211) branch are therefore orthogonal chaos projections of the same signed NCQC provenance. They are not separate physical sources, but retaining both is not double counting.

Consequences:

1. A carried NCQC atom supplies its H2 and q4(211) readouts without a second forward matrix word.
2. The H2 readout uses one matrix-vector product, outer products, and a scaled copy of `M`; no dense `M^2` product is needed.
3. The q4 branch keeps the same `M` leg. Only its singleton vectors fork.
4. A local diagonal terminal H4 functional `T_4=sum_i t_i e_i^(tensor 4)` has contraction

\[
\mathcal L_4[D(M,v,w)]
=4!\sum_i t_iM_{ii}v_iw_i
\]

up to the adopted normalization. This is `O(n)`.

## Exact one-stream causal theorem

Define the shared q4 module

\[
\mathcal S_4(M)
=\{I_4(\operatorname{sym}(M\otimes S)):S\in\operatorname{Sym}^2(V)\}.
\]

The exact causal state is

\[
X_l=(M_l,v_l,\{\beta_b,x_b,y_b\}_{b\le l},A_l^{(2)}),
\]

where `M_l` is the one new dense matrix word, `v_l` is the aggregate signed H3 singleton, the vector-pair list represents q4(211) branches sharing `M_l`, and `A_l^(2)` is the already-carried H2/covariance sink.

The update is:

1. Propagate `M`, `v`, and all singleton vectors with the common leg map. Degree-specific Hermite gains are scalar branch weights.
2. Before adding the current H3 birth, multiply the old signed H3 provenance by the current singleton. Inject its H2 projection into `A^(2)` and append its q4(211) vector pair.
3. Add the new H3 birth only after this read. It is one-stream exact only if its matrix leg lies in the current one-dimensional matrix-word span.
4. A primitive V4 birth is free of another matrix stream only if its pair leg lies in `span(M_l)` and its singleton-pair rank remains cheap.

All live matrix lineages must be collinear at every causal cut. Otherwise aggregating them into one `M` creates cross-contamination between old q4 branches and later births.

The singleton-pair factor also matters. A primitive birth

\[
B_4^\parallel=I_4(\operatorname{sym}(M\otimes S)),
\qquad
S=\sum_{j=1}^k s_j(v_j\odot w_j),
\]

is cheap only when `k` is small. The sequential construction has at most one new pair per layer. A dense-rank `S` requires `Theta(n)` vector streams and is GEMM-scale despite sharing the pair leg.

The primary reset invariance condition is

\[
\mathcal R_{3,l}C(v,M)
=\rho_{3,l}C(K_lv,K_lMK_l^\top),
\]

\[
\mathcal R_{4,l}D(M,v,w)
=\rho_{4,l}D(K_lMK_l^\top,K_lv,K_lw).
\]

Different scalar degree gains are harmless. A different matrix-leg operator or a nonseparable collision term produces `q4_perp`.

Terminal H4 is evaluated forward:

\[
y_4=\epsilon^2\sum_b\beta_b
\mathcal L_4[D(M_L,x_{b,L},y_{b,L})].
\]

No reverse-H4 matrix is required when the terminal functional is local or constant structured rank.

## `q4_perp` obstruction

Use the typed, collision-subtracted pair-by-singleton-pair flattening. Every tensor in `S4(M)` has pair-leg column space contained in `span(svec(M))`. Decompose a reset leakage or primitive fourth-order birth as

\[
B_4=B_4^\parallel+B_4^\perp,
\qquad
B_4^\parallel\in\mathcal S_4(M).
\]

If `B4_perp` has an endpoint-observable pair leg `N` not proportional to `M`, the cut rank is at least two matrix words. Singleton vectors and scalar refits cannot remove this lower bound. A component outside q4(211) is an even stronger obstruction.

The sharp one-stream value-tail is

\[
\operatorname{tail}_1
=\frac{\|(I-P_1)S\|_V^2}{\|S\|_V^2},
\]

where `P1` projects onto the complete one-stream reachable endpoint module. The best possible one-stream correlation is exactly

\[
q_{\max,1}=\sqrt{1-\operatorname{tail}_1}.
\]

Kill the one-stream lane if `tail_1 > 0.0328064` for the `q=0.98346` path, or if it exceeds `0.1435423` even for the looser path.

## Resource theorem

One dense conjugation

\[
M\leftarrow KMK^\top
\]

costs two dense GEMMs per transition. If `D` transitions are active, a matrix word costs `2D` GEMMs.

For `r_p` pair streams, `r_W` shared NCQC/q4 words, transition Kronecker ranks `kappa_p`, `kappa_W`, and `g_J` unshared dense Jordan products per layer,

\[
G_{\rm new}
=2D(\kappa_pr_p+\kappa_Wr_W)+Dg_J-G_{\rm shared}.
\]

Important cases under a 49-GEMM envelope:

- `D=14`: one full word costs 28; two cost 56 and do not fit.
- `D=10`: two full words cost 40 and fit.
- one full 14-layer word plus one local `M^2` product per layer costs 42.
- with a 14-layer primary word, 21 GEMMs remain. A second word can traverse at most `floor(21/2)=10` transitions, leaving one dense product total.

Propagating `k_l` singleton vectors at a layer costs `k_l/n` GEMM equivalents per vector family. One new q4 branch per layer over 14 layers costs about

\[
\frac{2}{1024}\sum_{l=1}^{14}l
=\frac{210}{1024}\approx0.205
\]

GEMM equivalents. A dense-rank singleton factor carried for all layers becomes a second GEMM-scale stream.

The one-stream accounting assumes the H2 return rides an existing covariance recursion. Without that sink, the H2 accumulator itself is another full dense stream.

## Optimal late second stream

After a 14-transition primary word, the residual budget permits ten secondary transitions. For a genuinely open q4-perpendicular word, a birth at `t` needs the interval `[t,L]`. These intervals are nested.

Therefore the maximal feasible region is the endpoint suffix

\[
[L-10,L].
\]

Starting later cannot increase a nonnegative value objective: the earlier-capable stream can remain zero or skip injections until any later birth.

Let `F_suffix` be the residualized value-whitened operator whose columns encode the compatible perpendicular births in the suffix, in their common causal matrix-word coordinates. The best one-word captured energy is

\[
E_2^*=\sigma_1(F_{\rm suffix})^2.
\]

The sharp two-stream tail is

\[
E_{\rm tail,2}
=E_{\rm early}
+\sum_{j\ge2}\sigma_j(F_{\rm suffix})^2
+E_{\rm non211}.
\]

If only layerwise nonnegative weights `w_t` are known, but not the cross-layer alignment Gram, the exact answer is not identified. The sharp bounds are

\[
\max_{t\in\rm suffix}w_t
\le E_2^*\le
\sum_{t\in\rm suffix}w_t.
\]

The lower extreme occurs for mutually value-orthogonal independent words. The upper extreme occurs for collinear compatible words.

After the second stream, require

\[
\frac{E_{\rm tail,2}}{E_{\rm total}}\le0.0328064
\]

for the `q=0.98346` lane.

## Exact two-stream finite-horizon recurrence

Let

\[
X_l^\parallel=(M_l,v_l,B_l,A_l^{(2)})
\]

be the primary state and

\[
X_l^\perp=(N_l,C_l)
\]

the optional secondary matrix word and its singleton-pair list. While active,

\[
N_{l+1}=K_l^\perp N_lK_l^{\perp\top},
\qquad
C_{l+1}=K_l^\perp\cdot C_l.
\]

A later perpendicular birth may enter this state only if it is aligned with the same transported word. An independent word is a distinct scheduling job and cannot coexist on the one physical secondary stream.

The terminal endpoint is the primary terminal contraction plus

\[
\sum_{c\in C_L}\mathcal L_4[D(N_L,c_x,c_y)].
\]

For a schedule `pi`, the dense cost is

\[
G(\pi)
=2D_{\rm primary}
+2|\mathcal A_\pi|
+G_{\rm abs}+G_{\rm local}
+\frac{2}{n}\sum_lk_l,
\]

where `|A_pi|` is the union count of active secondary transitions.

## Legal absorption and Bellman scheduling

A forward-local absorption at layer `e` is legal only if it preserves every future endpoint action:

\[
\mathcal O_{e\to L}^\perp z
=\mathcal O_{e\to L}^\parallel\mathcal A_ez
\]

for every carried secondary state `z`, together with the corresponding shift/intertwining relation on its future orbit.

Two concrete possibilities are:

1. Shared-module absorption: `N_e=gamma M_e` in the typed pair-leg quotient. Transfer every secondary vector pair into the primary list with coefficient `gamma`.
2. H2 absorption: an explicitly available grade-changing local map sends the entire secondary state to the H2 sink with no remaining H4-observable component.

A cash-out using a future adjoint `Lambda_4,e` is not a forward-local absorption. It is a hidden reverse-H4 stream.

If legal absorptions exist, collapse each rank-one secondary lineage into a scheduling job `j` with interval

\[
I_j=[t_j,e_j],
\qquad
h_j=e_j-t_j,
\]

and residualized value `W_j`. Sort jobs by end time, let `p(j)` be the last nonoverlapping job, and use the Bellman recurrence

\[
V(j,b)
=\max\left\{
V(j-1,b),
W_j+V(p(j),b-h_j)
\right\},
\]

for transition budget

\[
b\le H
=\left\lfloor
\frac{49-2D_{\rm primary}-G_{\rm abs}-G_{\rm local}}2
\right\rfloor.
\]

This scalar Bellman rule is exact when job values are Gram-orthogonal. With correlated job directions, replace scalar addition by the residualized Gram projection value of the selected span. Layerwise weights alone are not sufficient in that case.

Persistent jobs all end at `L`, overlap, and reduce to the one-suffix/top-singular-word theorem.

## Absorption no-go under the normal reset

Under the normal-ordered baseline reset, chaos propagation is Gaussian second quantization. It preserves chaos degree. Hence H4 cannot enter H2 through the baseline map.

If `K` is injective,

\[
KMK^\top\propto KNK^\top
\quad\Longrightarrow\quad
M\propto N.
\]

Therefore a genuinely perpendicular word cannot later enter the shared q4 module. Its matrix-word independence persists to the endpoint.

An exact early absorption is possible only if one of the following is exhibited:

- the word was actually parallel already;
- its entire future endpoint value is zero;
- the reset has an explicit rank-deficient forward quotient/intertwiner;
- there is a genuine grade-changing local map whose dense cost is charged.

Absent such a receipt, the final ten-transition suffix bound is unbeatable.

Rolling/restarting does not create a loophole. Stopping at `e<L` requires the sufficient statistic

\[
\langle\Lambda_{4,e},N_e\rangle.
\]

Computing it uses a reverse H4 segment of cost `2(L-e)`. Forward carriage from `t` to `e` plus reverse cash-out costs

\[
2(e-t)+2(L-e)=2(L-t),
\]

the same as direct persistence, before restart overhead. Recomputing separate cash-outs at several stops is the hidden `L^2` route. One shared reverse pass is another full matrix stream.

If a real forward absorption exists, rolling can help without hidden `L^2` work: selected disjoint intervals execute only their total active length. That exceptional case is exactly the Bellman problem above.

## Sharp endpoint-error certificate

Let `S` be the full residualized oracle endpoint source and let the scheduled primary-plus-secondary reachable module be `U_pi`. The exact best structural tail is

\[
E_{\rm struct}^*
=\min_{\pi:\,G(\pi)\le49}
\|(I-P_{\mathcal U_\pi})S\|_V^2.
\]

Let

\[
\eta_{\rm struct}=\sqrt{E_{\rm struct}^*},
\]

`eta_abs` bound any approximate absorption residual, and `eta_>=3` bound higher-order irreducible terms. Without proved mutual orthogonality,

\[
\eta_{\rm total}
\le
\eta_{\rm struct}+\eta_{\rm abs}+\eta_{\ge3}.
\]

For ideal residual `e` and endpoint approximation error `r`,

\[
R_{\rm impl}
=R_{\rm ideal}+2\langle e,r\rangle+\|r\|^2
\le
(\sqrt{R_{\rm ideal}}+\eta_{\rm total})^2.
\]

At

\[
R_{\rm ideal}=9.2547\times10^{-9},
\]

the sharp norm-only safety condition for raw below `1e-8` is

\[
\eta_{\rm total}
\le
\sqrt{10^{-8}}-\sqrt{9.2547\times10^{-9}}
=3.80\times10^{-6}.
\]

If structural, absorption, and higher-order errors are proved orthogonal in the residualized endpoint Gram, they may be combined by root-sum-square. Otherwise that replacement is invalid.

The natural ordered `epsilon^3` count is `0.0170898`, about `0.396e-9` on the `23.1598e-9` oracle-gap scale before its topology coefficient. An order-`epsilon^2` closure therefore needs a total normalized third-order coefficient below about `1.88`, after allocating margin to all other errors, or it must explicitly include/bound the third-order sector.

## Graph-ready dispositions

- `covariance-only local self-energy is sufficient`: `closed_proof` (false).
- `retarded cubic fish has an n^2 pair-state / L n^3 realization`: `solved` mathematically.
- `pair self-energy alone supplies the signed k3 source`: `closed_proof` (false by sign invariance).
- `raw Q_M^2 may be added beside its trace/Jordan terms`: `closed_proof` (double counting).
- `shared NCQC x singleton gives H2 plus q4(211) from one word`: `solved` mathematically.
- `one-stream full raw closure`: conditional on rank-one matrix-word alignment, cheap singleton-pair rank, common reset leg map, existing H2 sink, structured terminal contraction, and value tail below `3.28064%`.
- `visible primitive/reset q4_perp requires no second stream`: `closed_proof` unless its value projection is zero.
- `early exact absorption under injective second quantization`: `closed_proof`.
- `late second stream`: conditional on its top suffix value mode reducing the total tail below the correlation/raw gates within ten transitions.

## Current next mathematical question

The next question is not another scheduling variant. It is the exact source/rank audit needed to decide the theorem's premises:

1. Express the actual normal-ordered finite-width cubic birth in NCQC form and prove or refute rank-one unit-by-pair factorization on the residualized endpoint-observable subspace.
2. Derive the actual degree-3 and degree-4 reset operators symbolically and determine whether both use the same injective matrix-leg map, up to scalar Hermite gains.
3. Decompose the primitive/discrete `V4` coefficient into the shared q4(211) module and typed `q4_perp`; compute its analytic matrix-word rank and singleton-pair rank, not a raw plane SVD.
4. Determine the value-observable cross-layer Gram of the suffix `q4_perp` births. With weights alone, the second-stream value lies only between `max w_l` and `sum w_l`; the Gram decides the exact leading singular energy.
5. Prove a conservative endpoint norm for the irreducible `epsilon^3` sector and combine it with the structural tail under the `3.80e-6` endpoint-error gate.

The fail-fast decision is then exact: if the shared first word plus the best ten-transition perpendicular word retains less than `96.7194%` of residualized v+k3+k4 value energy, or violates the channel gain inequality, the proposed `O(n^2)`, at-most-49-GEMM closure cannot reach raw `1e-8` in this form.

## Checkpoint 5: fused-pair effective Kraus rank

The fused-pair derivation supplies a new exact premise: after the external bridge, a generic `Sym_45` q4-perpendicular provider has effective Kraus rank 11 on the reachable and endpoint-observable quotient.

Let a bridged completely positive update be

\[
\Phi_l(X)=\sum_{a=1}^{r_l}A_{a,l}XA_{a,l}^\top.
\]

After composing with the external ingress and egress bridges and removing unreachable or endpoint-null directions, define its effective Choi matrix

\[
\mathsf C_l
=\sum_a
\operatorname{vec}(\widetilde A_{a,l})
\operatorname{vec}(\widetilde A_{a,l})^\top.
\]

The minimum exact number of effective provider directions is

\[
r_{\rm eff,l}=\operatorname{rank}(\mathsf C_l).
\]

This is invariant under Kraus rotations. Fusing the external bridge may change the Kraus basis and delete null directions, but it cannot represent a rank-11 effective Choi matrix with fewer than 11 rank-one provider terms.

Under the fused-provider billing convention, one effective Kraus direction carried through one transition costs one matmul-equivalent. Therefore the exact marginal provider cost over an active interval `I` is

\[
G_{\rm provider}^{\rm exact}(I)
=\sum_{l\in I}r_{\rm eff,l}.
\]

For constant effective rank `r` over `h` transitions,

\[
G_{\rm provider}^{\rm exact}=rh.
\]

This is already the favorable fused count. An unfused two-sided dense conjugation would cost more.

For the generic q4-perpendicular result,

\[
r=11,\qquad h=10,
\]

so

\[
G_{\rm provider}^{\rm exact}=110.
\]

The available provider budget is 10. Hence the generic exact ten-transition second-stream route is closed by an `11x` cost lower bound. Batching or rotating the Kraus basis does not alter this billed-flop/rank lower bound.

### Rank-lifetime tradeoff under the ten-unit provider budget

Let `B_ext=10`. For a suffix of `h` transitions, a retained effective rank `r` must satisfy

\[
rh\le B_{\rm ext}.
\]

Let `G_h` be the residualized endpoint-value Gram of the lawful effective Kraus lineages in the final `h`-transition suffix, after pulling them into common causal bridge coordinates. Write its eigenvalues as

\[
\lambda_{h,1}\ge\lambda_{h,2}\ge\cdots\ge0.
\]

The sharp value captured by the best lawful rank-`r` spectral subspace is at most

\[
E_{h,r}=\sum_{j=1}^r\lambda_{h,j},
\]

with equality when those eigenmodes are implementable by the lawful bridge class. The full budgeted suffix optimization is

\[
E_{\rm bridge}^*(10)
=\max_{1\le h\le10}
\max_{1\le r\le\min(11,\lfloor10/h\rfloor)}
\sum_{j=1}^r\lambda_{h,j}.
\]

Its structural tail must also include every q4-perpendicular birth earlier than the selected suffix and every component outside the fused Kraus module.

For the full ten-transition suffix, the budget forces

\[
r\le1.
\]

Thus the only full-support approximation is a rank-one external bridge.

### Sharp lawful rank-one criterion

Choose a whitened basis of the 11 effective Kraus directions. A unit coefficient vector `c` defines the one-Kraus candidate

\[
A(c)=\sum_{a=1}^{11}c_aA_a.
\]

Let `G_V` be the residualized endpoint-value covariance of those effective directions. For an unrestricted lawful linear combination, the maximum retained quartic value fraction is exactly

\[
p_1^*
=\frac{\lambda_{\max}(G_V)}{\operatorname{tr}(G_V)}.
\]

If the external bridge permits only a subspace `C` or uses a nonidentity normalization metric `H`, replace the numerator by the largest lawful generalized Rayleigh quotient

\[
\max_{c\in\mathcal C,\,c^\top Hc=1}c^\top G_Vc.
\]

The unrestricted `lambda_max / trace` ratio is then an optimistic upper bound.

The residualized quartic ladder requires at least

\[
p_{\rm req}=0.27732
\]

of the total k4 value when variance and signed k3 are perfect. Equivalently, a lawful full-suffix rank-one bridge is viable only if

\[
\frac{\lambda_{\max}(G_V)}{\operatorname{tr}(G_V)}
\ge0.27732.
\]

Define the value stable rank

\[
r_{\rm stable,V}
=\frac{\operatorname{tr}(G_V)}{\lambda_{\max}(G_V)}.
\]

The same gate is

\[
r_{\rm stable,V}\le\frac1{0.27732}=3.60594.
\]

If the 11 nonzero value eigenvalues have condition number

\[
\kappa=\frac{\lambda_1}{\lambda_{11}},
\]

then

\[
p_1^*
\le
\frac{\kappa}{\kappa+10}.
\]

Consequently, any analytic bound

\[
\kappa<3.8374
\]

proves that no rank-one bridge reaches the `27.732%` quartic gate. An isotropic eleven-direction value spectrum would retain only `1/11 = 9.09%`, but isotropy has not been proved.

Effective Kraus rank 11 by itself is not a value-concentration theorem. A full-rank spectrum may be highly anisotropic. Therefore the exact rank-11 route is closed by cost, while the approximate rank-one route remains conditional on the value spectrum.

If the primary shared q4 module already captures quartic value `E_parallel`, the precise combined gate is

\[
\frac{E_{\parallel}+\lambda_{\max}(G_V)}{E_{4,\rm total}}
\ge0.27732.
\]

At equality, the ideal raw lands only on the `1e-8` boundary and leaves no room for factorization, higher-order, or implementation error. With an additional raw-MSE safety allocation `delta`, the necessary fraction becomes

\[
p_1
\ge
\frac{0.286\times10^{-9}+\delta}
{1.0313\times10^{-9}},
\]

after subtracting any already captured parallel quartic value consistently.

### Checkpoint-5 graph disposition and next question

- `generic Sym_45 rank-11 q4_perp, exact D10 provider under ten-unit budget`: `closed_cost` (`110 > 10`).
- `one full-suffix effective direction`: the only budget-feasible full-support approximation.
- `rank-one bridge captures enough k4 value`: conditional on `lambda_max(G_V)/tr(G_V) >= 0.27732` and on the leading direction being lawful/frozen rather than truth-selected.
- `rank 11 alone proves rank-one value failure`: `closed_proof` (false; rank does not control spectral concentration).

The current decisive question is the analytic residualized value spectrum of the 11 bridged directions. Prove either

\[
r_{\rm stable,V}>3.60594
\]

to close the compressed route, or exhibit a lawful leading bridge direction with more than `27.732%` of total residualized quartic value plus enough additional margin for the endpoint-error certificate. If neither is proved, the generic exact route remains closed and the rank-one approximation remains explicitly unresolved rather than being mislabeled a solution.

## Checkpoint 6: dynamic lawfulness of a rank-one full-suffix bridge

The static ratio `lambda_max(G_V) / trace(G_V)` is only an optimistic value bound. A ten-transition provider must be one causal transported word from L5 through L14. It may not select an unrelated leading eigenvector independently at every layer.

### Observable quotient and Choi recurrence

At layer `l`, quotient the effective Kraus space by directions that either enter the already-carried primary module exactly or are null under every future endpoint readout. Denote the remaining q4-perpendicular value space by `Q_l`.

Represent the effective q4-perpendicular Choi state by a positive semidefinite matrix `C_l` on `Q_l`. A one-Kraus carrier has

\[
C_l=s_lc_lc_l^\top,
\qquad
s_l\ge0,
\qquad
\|c_l\|=1.
\]

Let the normal-ordered reset and external bridge act on this quotient by effective maps `F_{mu,l}`. Let `B_{l+1} >= 0` be the Choi matrix of every new boundary/layer injection that the carrier claims to retain. Then

\[
C_{l+1}^{-}
=\sum_\mu F_{\mu,l}C_lF_{\mu,l}^\top,
\]

\[
C_{l+1}=C_{l+1}^{-}+B_{l+1}.
\]

The L5 boundary is `C_5=B_5` unless an earlier state is explicitly supplied.

### Necessary and sufficient rank-one intertwining theorem

An exact lawful rank-one carrier exists over L5--L14 if and only if there are nonzero unit directions `c_5,...,c_14` such that:

1. Boundary alignment:

\[
\operatorname{range}(B_5)\subseteq\operatorname{span}(c_5).
\]

2. Reset alignment for every effective reset Kraus leg:

\[
F_{\mu,l}c_l=a_{\mu,l}c_{l+1}
\]

for some scalars `a_{mu,l}`.

3. Injection alignment:

\[
\operatorname{range}(B_{l+1})
\subseteq\operatorname{span}(c_{l+1}).
\]

Equivalently, at every transition,

\[
\operatorname{rank}
\begin{bmatrix}
F_{1,l}c_l&\cdots&F_{m,l}c_l&B_{l+1}^{1/2}
\end{bmatrix}
\le1.
\]

All statements are on the observable quotient; a component in the exact primary module or future-null kernel has already been removed.

Proof: `C_{l+1}` is a sum of positive semidefinite rank-one outer products generated by the columns

\[
\{\sqrt{s_l}F_{\mu,l}c_l\}_\mu
\cup
\{B_{l+1}^{1/2}e_j\}_j.
\]

A positive semidefinite sum has rank one if and only if all of its generating vectors are collinear. Positivity prevents cancellation of transverse Kraus directions. The displayed alignment conditions are therefore necessary. They are sufficient because the scalar state then obeys

\[
s_{l+1}
=s_l\sum_\mu|a_{\mu,l}|^2
+\operatorname{tr}(B_{l+1}),
\]

after expressing the aligned birth in the normalized `c_{l+1}` direction.

This is the exact bridge/reset intertwiner: the family of one-dimensional spaces

\[
\mathcal L_l=\operatorname{span}(c_l)
\]

must form an invariant line bundle through the time-varying effective Kraus system, and every retained boundary injection must enter that same line.

### Reset leakage

Before quotienting, write

\[
F_{\mu,l}c_l
=a_{\mu,l}c_{l+1}+r_{\mu,l}.
\]

The reset leakage `r_{mu,l}` is harmless only if it is transferred by a proved forward-local map into the primary module or is null under all future endpoint observations. Any endpoint-visible transverse remainder creates Choi rank at least two and is a structural approximation error.

Likewise, decompose a birth as

\[
B_{l+1}
=P_{l+1}B_{l+1}P_{l+1}
+B_{l+1}^{\perp},
\qquad
P_{l+1}=c_{l+1}c_{l+1}^\top,
\]

including the cross blocks in the residual definition. An endpoint-visible `B_perp` cannot be retained by the one-Kraus state. Its future value belongs in the structural tail.

### Boundary injection

At L5, exact rank one requires the observable boundary Choi matrix to have rank at most one. The generic rank-11 boundary fails this exact condition immediately. A rank-one approximation chooses one line in `range(B_5)` and discards the value-observable orthogonal boundary component.

Later injections impose the same condition. A line that is perfect for the L5 boundary may be poor for later births; the lawful direction must therefore be selected from the entire finite-horizon value problem, not from the L5 spectrum alone.

### Why independent per-layer eigenvectors are illegal

Let `u_l` be the leading eigenvector of a layer-local value Gram `G_l`. The sequence `{u_l}` is a lawful full-suffix rank-one bridge if and only if

\[
F_{\mu,l}u_l\in\operatorname{span}(u_{l+1})
\]

for every reset leg and every layer, and every retained birth at `l+1` is aligned with `u_{l+1}` on the observable quotient.

If this fails, switching from `u_l` to an unrelated `u_{l+1}` has only three interpretations:

1. Keep the old state and inject the new word, producing rank at least two and exceeding the one-direction provider budget.
2. Discard the old state, losing all earlier endpoint value.
3. Cash out the old state with a future H4 adjoint, which is the hidden reverse-H4 route already closed by cost.

Thus

\[
\sum_l\lambda_{\max}(G_l)
\]

is generally a noncausal optimistic bound. It is attainable only when the local leading eigendirections satisfy the invariant-line and boundary-alignment equations.

### Joint block-Hankel value gate

Let `H` denote the value-whitened finite-horizon block-Hankel operator mapping all L5--L14 q4-perpendicular boundary/layer births to every future endpoint-observable response. Let `C_1^causal` be the set of nonzero block-Hankel families that:

- have cut rank at most one at every causal cut;
- satisfy the shift/intertwining equations above;
- use one full-suffix provider direction;
- respect the external bridge and boundary-injection constraints.

Exact one-state realization is equivalent to

\[
\operatorname{rank}(\mathsf H_k)\le1
\]

for every cut `k`, together with shift compatibility. The generic effective rank-11 result rules out exact equality.

The sharp dynamic rank-one value concentration is

\[
p_{1,\rm dyn}^*
=\sup_{\widetilde H\in\mathcal C_1^{\rm causal}}
\frac{|\langle H,\widetilde H\rangle_V|^2}
{\|H\|_V^2\,\|\widetilde H\|_V^2}.
\]

This is the squared value correlation of the best lawful causal rank-one Hankel family. It automatically charges boundary misalignment, reset leakage, and cross-layer direction inconsistency. It obeys

\[
p_{1,\rm dyn}^*
\le
\frac{\lambda_{\max}(G_V)}{\operatorname{tr}(G_V)},
\]

where the right side is the prior unconstrained static spectral bound.

The quartic raw gate is now

\[
p_{1,\rm dyn}^*\ge0.27732
\]

after accounting consistently for any quartic value already captured by the primary shared module. If this inequality fails, no lawful rank-one ten-transition bridge reaches raw `1e-8` even when its independently optimized per-layer spectra appear favorable.

### Single-map reduction

If the observable quotient has one effective transition map `F_l` per layer, with no multi-Kraus splitting on the chosen line, then a lawful trajectory is determined by `c_5`:

\[
c_l\propto F_{l-1}\cdots F_5c_5.
\]

Pull the layerwise value Grams back to L5 coordinates:

\[
G_{\rm joint}
=\sum_{l=5}^{14}
F_{l:5}^*G_lF_{l:5},
\qquad
F_{l:5}=F_{l-1}\cdots F_5.
\]

With the matching pulled-back normalization Gram `H_joint`, the best lawful initial direction is the top generalized eigenvector of

\[
G_{\rm joint}c=\lambda H_{\rm joint}c,
\]

subject to all boundary-injection alignment constraints. For a multi-Kraus reset, additionally restrict `c_5` to the algebraic invariant-line set defined by

\[
(F_{\mu,l}c_l)\wedge(F_{\nu,l}c_l)=0
\]

for every pair of reset legs and every layer.

### Checkpoint-6 disposition and next question

- `choose the leading value-Kraus eigenvector independently per layer`: `closed_proof` unless the invariant-line equations hold.
- `static lambda_max / trace is the exact deployed rank-one value`: `closed_proof`; it is an optimistic upper bound.
- `lawful rank-one full-suffix carrier`: conditional on rank-one boundary injection, reset-leg collinearity, zero observable reset leakage, and joint causal-Hankel value at least `27.732%`.

The decisive next object is the actual L5--L14 effective reset/injection pencil

\[
\{F_{\mu,l},B_l\}_{l=5}^{14}.
\]

Symbolically determine whether it has any observable invariant line. If none exists, close the rank-one approximation by proof before estimating value. If one exists, evaluate its joint block-Hankel value concentration; only the joint value, not the sum of layerwise top eigenvalues, may be compared with `27.732%`.

## Checkpoint 7: universal per-layer K31 Bellman-action accumulator

The dynamic rank-one q4 suffix can be avoided if the endpoint derivative of the full q4 source factors through fresh, local K31 moments. This is a different architecture: contract each q4 birth with its future Bellman value at the birth layer, then carry only a scalar action sum.

### Exact order-`1/n` causal realization

Let `x_l` be the baseline deployed state and let the physical q4 correction enter through a K31-plane perturbation at order `epsilon^2=1/n`:

\[
x_{l+1}
=F_l(x_l)+\epsilon^2B_l\,\delta k_l
+O(\epsilon^3),
\qquad
\delta k_l=\operatorname{vec}(\delta K31_l).
\]

Let the scalar endpoint objective be

\[
y=g(x_L).
\]

Define the baseline Bellman costates

\[
\lambda_L=\nabla g(x_L),
\]

\[
\lambda_l
=(D F_l(x_l))^*\lambda_{l+1}.
\]

The local K31 action weight is

\[
A_l=B_l^*\lambda_{l+1}.
\]

First-order Duhamel expansion gives

\[
\delta y
=\epsilon^2\sum_{l=5}^{14}
\langle A_l,\delta k_l\rangle
+R_{\rm nonlinear}.
\]

Therefore the complete order-`1/n` causal state is the scalar recurrence

\[
S_5=0,
\]

\[
S_{l+1}=S_l+\langle A_l,\delta k_l\rangle,
\]

\[
\delta y=\epsilon^2S_{15}.
\]

No H4 matrix word is transported. Once the local scalar actions are known, even one scalar per layer is unnecessary: a single running scalar suffices.

Purely q4--q4 multiple-insertion terms begin at `epsilon^4=1/n^2`. Interactions with an order-`epsilon` signed H3 correction may begin at `epsilon^3`. Thus the accumulator is exact through order `1/n` when evaluated on the baseline trajectory.

The Bellman weights must be computable from the deployed white-box inputs and a frozen formula. A truth-dependent costate is not a lawful universal accumulator.

### Two-pass and storage qualification

The scalar state does not imply that all supporting computation is scalar. `A_l` is generally a K31-plane adjoint. It can be obtained by:

1. a backward Bellman pass followed by a forward/local moment pass;
2. an analytic value function that supplies `A_l` during the forward pass; or
3. storing the local K31 moments until the backward weights are available.

This still eliminates H4 memory, but any dense costate propagation and any stored `n x n` K31 plane must be charged separately. The accumulator theorem concerns the q4 causal state, not a free moment/costate oracle.

### K31 as a fourth-cumulant slice

For a centered random vector `z` with covariance `C`, the normal-ordered K31 plane is

\[
\delta K31_{ij}
=\mathbb E[z_i^3z_j]-3C_{ii}C_{ij}
=\kappa_{iiij}.
\]

Equivalently, with

\[
h_{3,i}(z)=z_i^3-3C_{ii}z_i,
\]

the per-layer Bellman action is

\[
a_l
=\langle A_l,\delta K31_l\rangle
=\mathbb E_\mu[h_3(z)^\top A_lz].
\]

This expression needs only one scalar contraction of the fourth cumulant, not the full `n^4` tensor and not even the full `n^2` K31 plane if the law interface can evaluate the displayed expectation directly.

### General-law moment acquisition cost

If the layer law is given by `N` weighted atoms `z_s`, then

\[
a_l
=\sum_{s=1}^Np_s h_3(z_s)^\top A_lz_s.
\]

This costs `O(N n^2)` arithmetic for a dense `A_l`. Relative to an `n^3` dense matmul it is about `N/n` matmul-equivalents per layer. The full K31 plane can be formed as the rectangular product

\[
K31=H_3(Z)^\top\operatorname{diag}(p)Z,
\]

with the same leading arithmetic order. If `N<=n`, direct action acquisition can fit within about one dense-matmul equivalent per layer. If `N` grows substantially beyond `n`, it does not.

If a law oracle supplies the exact K31 plane, the final action is only an `O(n^2)` inner product. But no finite universal fourth-moment oracle follows from mean, covariance, and third moments. A standardized Rademacher variable and a standard Gaussian have the same first three moments but fourth moments 1 and 3. Their normal-ordered K31 diagonal values differ. Hence lower-moment state alone cannot identify the required action for a general law.

### Provider action rank

A scalar output does not automatically have provider rank one. Let `S_l` be the local general-law/q4 source map, `M31_l` the K31 acquisition map, and `A_l` the Bellman row. Define

\[
r_{\rm action,l}
=\operatorname{rank}_{\rm provider}
\left(A_l\circ M31_l\circ S_l\right),
\]

where `rank_provider` is the minimal lawful separated/bilinear factorization rank after all external bridges, not ordinary scalar-output linear rank.

The fused-provider acquisition cost is

\[
G_{\rm action}
=\sum_{l=5}^{14}r_{\rm action,l}
+G_{\rm Bellman}+G_{\rm law}.
\]

With a ten-unit provider budget and ten nonzero layers, the local action must average at most one provider direction per layer, before any unshared Bellman/law cost. The architecture is a genuine cost bypass only if the Bellman row can be fused before the expensive nonlinear moment operation so that `r_action,l=1` (or the total rank sum remains at most ten).

If moment acquisition first materializes all 11 generic fused-pair directions and only then contracts them to a scalar, the cost remains

\[
11\times10=110.
\]

That moves the closed q4 carrier into the moment provider; it does not bypass it.

### Necessary and sufficient factor-through-K31 theorem

Let `Q` be the vector space of all reachable order-`1/n` q4 source histories. Let

\[
\mathcal V:Q\to Y
\]

be the endpoint Fr\'echet derivative in the residualized prediction-value space. Let

\[
\mathcal M_{31}:Q
\to\bigoplus_{l=5}^{14}\mathcal K_{31,l}
\]

map a q4 history to all of its fresh local K31 planes.

There exist Bellman rows `{A_l}` for which the complete order-`1/n` endpoint action is

\[
\mathcal V
=\sum_lA_l\mathcal M_{31,l}
\]

if and only if

\[
\ker(\mathcal M_{31})
\subseteq
\ker(\mathcal V).
\]

Equivalently,

\[
\mathcal V\in\operatorname{range}(\mathcal M_{31}^*).
\]

Necessity: two q4 histories with identical local K31 planes must have identical endpoint derivative if a K31-only accumulator is exact. Sufficiency: the kernel inclusion makes `V` a well-defined linear functional on the quotient `Q / ker(M31)`, so it factors through `M31`; its quotient coordinates are the Bellman rows.

This is the exact condition under which the dynamic rank-one/Hankel obstruction disappears. After the local K31 measurements, the contracted q4 endpoint map has no retarded state: its strictly delayed block-Hankel part is zero and only direct feedthrough scalars remain.

If the kernel inclusion fails, there exists a reachable q4 history with zero local K31 measurements and nonzero endpoint value. The scalar accumulator necessarily misses it. Recovering that history requires an additional transported state or another local moment family.

### When the obstruction is merely moved into moment computation

The mathematical factorization above is not by itself a deployable bypass. Fresh local acquisition must also be sufficient under the allowed law interface.

Let `D_l` denote all law information actually available to the estimator at layer `l`. Exact local acquisition requires

\[
\ker(D_l)
\subseteq
\ker(A_lM31_l)
\]

on the admissible law class. If two admissible laws have the same `D_l` but different Bellman-weighted K31 action, no universal local formula exists.

Therefore the K31 scalar route bypasses the q4 Hankel obstruction if and only if all three conditions hold:

1. Endpoint sufficiency: `ker(M31) subset ker(V)` on reachable q4 histories.
2. Local law sufficiency: each weighted K31 action is identifiable from the allowed fresh layer-law interface.
3. Arithmetic sufficiency: `sum_l r_action,l` plus Bellman/law costs fits the provider budget.

Failure of condition 1 leaves missing q4 endpoint value. Failure of condition 2 recreates the need to transport a fourth-order law state. Failure of condition 3 recreates the 11-direction cost behind a scalar output.

### Value gate

If exact factorization does not hold, let `P_31` be the residualized endpoint-value projection onto `range(M31^*)`. The maximum quartic fraction available to any K31-action accumulator is

\[
p_{31}
=\frac{\|P_{31}\mathcal V\|_V^2}
{\|\mathcal V\|_V^2}.
\]

After including any quartic value already supplied by the shared primary module, the K31 route must still satisfy the residualized ladder requirement of at least `27.732%` total k4 value. If `p31` is below the required remainder, no choice of scalar Bellman weights can win.

### Total endpoint-error certificate

Let `r_fact` be the endpoint derivative left outside `range(M31^*)`. Let `e_l` be K31 moment-acquisition error and `Delta A_l` Bellman-weight error. Then a conservative prediction-error bound is

\[
\eta_{31}
\le
\epsilon^2
\left[
\|r_{\rm fact}\|\,\|q_4\|
+\sum_l
\left(
\|A_l\|\,\|e_l\|
+\|\Delta A_l\|\,\|\delta k_l\|
+\|\Delta A_l\|\,\|e_l\|
\right)
\right]
+\eta_{\ge3}.
\]

If the endpoint value has Hessian norm at most `H_V` in the injected K31 directions, the purely q4 second-order Bellman remainder obeys

\[
|R_{\rm q4-q4}|
\le
\frac12\epsilon^4H_V
\left(\sum_l\|\delta k_l\|\right)^2.
\]

Add the order-`epsilon^3` H3--q4 interaction separately unless a parity/orthogonality proof removes it.

The raw-MSE certificate remains

\[
R_{\rm impl}
\le
(\sqrt{R_{\rm ideal}}+\eta_{31})^2.
\]

At `R_ideal=9.2547e-9`, require

\[
\eta_{31}\le3.80\times10^{-6}.
\]

### Checkpoint-7 disposition and next question

- `one scalar accumulator can sum exact local K31 Bellman actions`: `solved` through order `1/n`.
- `scalar endpoint implies one provider direction`: `closed_proof` (false in general).
- `general-law K31 follows from lower moments`: `closed_proof` (nonidentifiable).
- `K31 action bypasses q4 Hankel memory`: conditional on endpoint factorization through fresh local K31, local law sufficiency, and total action-provider rank at most ten.

The decisive next derivation is to write the actual q4-to-endpoint derivative and test

\[
\ker(\mathcal M_{31})\subseteq\ker(\mathcal V).
\]

If it passes, derive the fused polynomial for

\[
\mathbb E[h_3(z)^\top A_lz]
\]

and prove its provider factorization rank is one per layer without first constructing the 11 fused-pair directions. If it fails, identify the smallest additional local moment slice whose direct sum with K31 makes the kernel inclusion true.

## Checkpoint 8: graph-ready K31 theorem decomposition

The K31 proposal contains five logically independent claims.  They must not be
represented by one packed graph node: the first and fifth are mathematical
theorems already proved, while the middle three remain candidate-specific
scientific gates.

Throughout this section, let `Q_l` be the order-four source space born at layer
`l`, let `J_l:Q_l -> X_{l+1}` be its linearized state injection, let
`T_{l+1:L}` be the baseline tangent propagator to the endpoint, and let `c` be
the endpoint covector.  Define

\[
\lambda_{l+1}=T_{l+1:L}^*c,
\qquad
g_l=J_l^*\lambda_{l+1}\in Q_l^*.
\]

Thus `g_l` is the complete endpoint value row of a fresh quartic source at
layer `l`.  Let

\[
M_l:Q_l\longrightarrow \mathcal K_{31,l}
\]

be the fresh K31 observation map, and let `R` be the actually reachable
subspace of `direct_sum_l Q_l`.

### Node A -- scalar Bellman transport exactness

- **Suggested id:** `P2R0G19B_MATH_K31_SCALAR_BELLMAN`
- **Label:** `Target-adjoint scalar transport through order 1/n`
- **Kind:** `phase2_g19b_math_theorem`
- **Status:** `solved`
- **Priority:** `1`
- **Summary:** For additive order-`1/n` quartic births in a differentiable
  Markov baseline, the complete target-specific first variation is the sum of
  local costate actions.  A single running scalar carries that value; no H4
  state is needed after each local action has been supplied.

The exact recurrence for `epsilon=n^{-1/2}` is

\[
S_5=0,
\qquad
S_{l+1}=S_l+g_l(q_l),
\qquad
\delta y=\epsilon^2S_{15}+R_{\ge3}.
\]

This is exact in the coefficient of `epsilon^2=1/n`.  If an order-`epsilon`
H3 perturbation is present, the first omitted mixed term can be
`O(epsilon^3)`; with only q4 perturbations, the first multiple-insertion term
is `O(epsilon^4)`.

**Necessary and sufficient condition.**  On the permitted source history
space, a target-specific scalar recurrence of this form is exact through
order `1/n` if and only if the order-`1/n` endpoint coefficient is an additive
linear functional of the local births,

\[
\mathcal V(q_5,\ldots,q_{14})=\sum_l g_l(q_l).
\]

For a differentiable Markov baseline with additive local injections this
condition follows from Duhamel's formula, with the displayed `g_l`.  For a
non-Markov description it holds after augmenting the baseline state until its
tangent propagator has the semigroup property.  In target-conditioned Hankel
language, contraction with the exact local rows makes the strictly delayed
q4 input/output Hankel operator zero.

This is an algebraic state theorem, not a free-oracle theorem.  A literal
one-pass deployed recurrence additionally requires each `g_l` to be
measurable and computable from the allowed current information.  An allowed
reverse pass or a frozen analytic value function also suffices, but its state
and arithmetic must be billed.

- **Terminal criterion:** Proved by the costate/Duhamel identity through
  `1/n`; retain `solved`.  Reopen only if the proposed baseline omits a state
  needed for the tangent semigroup or if the estimator forbids every lawful
  way to obtain the Bellman rows.
- **Evidence:** This file, sections `Checkpoint 7: universal per-layer K31
  Bellman-action accumulator` and `Node A -- scalar Bellman transport
  exactness`.
- **Decomposition children:** none.
- **Goal-path summary:** The depth-memory obstruction is eliminated only
  after exact local action acquisition; Nodes B--D decide whether that premise
  is lawful and cheap.

### Node B -- factor-through-K31 condition

- **Suggested id:** `P2R0G19B_MATH_K31_FACTOR`
- **Label:** `Endpoint quartic value factors through fresh K31`
- **Kind:** `phase2_g19b_math_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** Fresh K31 actions contain the complete quartic endpoint value
  exactly if and only if every reachable quartic history invisible to all
  fresh K31 slices is also endpoint invisible.

Stack the maps as

\[
M=\operatorname{diag}(M_5,\ldots,M_{14}),
\qquad
G(q)=\sum_lg_l(q_l).
\]

The candidate is exact on the reachable history space `R` if and only if

\[
\ker(M|_R)\subseteq\ker(G|_R),
\tag{B.1}
\]

or, equivalently in finite dimension,

\[
G|_R\in\operatorname{range}\big((M|_R)^*\big).
\tag{B.2}
\]

If layer births are independently reachable, so that
`R=direct_sum_l R_l`, this global condition is equivalent to the ten local
conditions

\[
g_l|_{R_l}\in
\operatorname{range}\big((M_l|_{R_l})^*\big),
\qquad l=5,\ldots,14.
\tag{B.3}
\]

In Euclidean coordinates define

\[
P_l=M_l^*(M_lM_l^*)^\dagger M_l.
\]

Then the minimum-norm K31 Bellman row and the irreducible local residual are

\[
A_l=(M_l^*)^\dagger g_l,
\qquad
r_l=(I-P_l)g_l.
\tag{B.4}
\]

Exact local factorization is precisely `r_l=0`.  With non-Euclidean source
and value metrics, these are the corresponding metric adjoints and weighted
pseudoinverses.

There is also a state-space form that is often easier to audit.  Let

\[
\mathcal U_{l+1}=\ker(\lambda_{l+1}^*)
\]

be the target-unobservable state subspace at the injection boundary.  Then
(B.3) is equivalent to

\[
J_l\big(\ker M_l\cap R_l\big)\subseteq\mathcal U_{l+1}.
\tag{B.5}
\]

If the actual injection factors as `J_l=\bar J_l M_l` on `R_l`, the condition
holds automatically and

\[
A_l=\bar J_l^*\lambda_{l+1}.
\]

Conversely, one reachable source in `ker M_l` whose injected state has
nonzero pairing with `lambda_{l+1}` disproves the exact K31 bypass.

#### Dense-reset generic obstruction

Let

\[
\mathcal S_{31}
=\operatorname{span}\{\operatorname{sym}(e_i^{\otimes3}\otimes e_j):
1\le i,j\le n\}
\subset\operatorname{Sym}^4(\mathbb R^n)^*
\]

be the coordinate K31 row space.  It has dimension `n^2`.  A dense reset
`K` pulls a future K31 row back to

\[
\Gamma_4(K)^*\mathcal S_{31}.
\]

The whole K31 row space is invariant under this pullback only for exceptional
resets.  Indeed, the pullback of `e_i^{\otimes4}` is `(K^Te_i)^{\otimes4}`.
If row `i` of `K` has two nonzero entries, its polynomial contains an
`x_a^2x_b^2` term, which is outside `S_31`.  Hence invariance of the complete
space forces every row, and for invertible `K` every column, to have one
nonzero entry: `K` must be monomial up to scaling.  Generic dense resets do
not satisfy this.

For one particular future action the exact condition is weaker but explicit:

\[
(I-P_{31})\,
\Gamma_4(K)^*
\left(\sum_{ij}A_{ij}\,
\operatorname{sym}(e_i^{\otimes3}\otimes e_j)\right)=0.
\tag{B.6}
\]

The left side contains the pulled-back `2+2`, `2+1+1`, and all-distinct
coefficients.  Their simultaneous cancellation is a proper algebraic
condition on a dense `K` and `A`; absent a structural identity it is
measure-zero.  Therefore K31-only factorization is generically false on the
full quartic source space.  It can still be true on the campaign's restricted
reachable NCQC/q4(211) quotient, which is exactly why the candidate-specific
test (B.1)--(B.5) remains open rather than closed by dimension counting.

#### Sharp value projection when exact factorization fails

Let `q` have source covariance `Sigma` in the residualized value ensemble.
The best linear K31 action is

\[
a_*=(M\Sigma M^*)^\dagger M\Sigma g,
\]

and its exactly captured value fraction is

\[
p_{31}
=\frac{(M\Sigma g)^*(M\Sigma M^*)^\dagger(M\Sigma g)}
{g^*\Sigma g}.
\tag{B.7}
\]

This formula automatically handles cross-layer source correlations.  For an
orthogonal direct sum it reduces to the sum of the layerwise projected
energies.  A layer-by-layer top direction is not a substitute for (B.7): the
stacked weights must belong to the row space of one lawful observation map on
the reachable history quotient.

For a scalar target, one additional formal observable

\[
N_l(q_l)=r_l(q_l)
\]

repairs a failed local factorization exactly.  This is only an algebraic
minimality statement.  If evaluating `N_l` requires the primitive H4 law, it
is the forbidden reverse-H4 provider under another name.  For a vector of
targets, the minimal number of extra scalar moment rows is the rank of the
residual target operator `(I-P_l)G_l` on the reachable quotient.

- **Terminal criterion:** On the actual restricted q4 source quotient, derive
  every `g_l=J_l^*lambda_{l+1}` and either prove all residual rows in (B.4)
  vanish, or exhibit an endpoint-visible kernel witness.  If exactness fails,
  report (B.7) and require the remaining quartic value gate before handing a
  residual row to Node C.
- **Evidence:** This file, sections `Necessary and sufficient
  factor-through-K31 theorem`, `Node B -- factor-through-K31 condition`, and
  equations (B.1)--(B.7).
- **Decomposition children:** none.
- **Goal-path summary:** This is the decisive scientific gate.  Generic dense
  mixing works against K31 sufficiency; only a restricted-source or
  target-unobservability identity can save it.

### Node C -- fresh moment identifiability

- **Suggested id:** `P2R0G19B_MATH_K31_IDENTIFIABILITY`
- **Label:** `Fresh Bellman-weighted K31 is identifiable from the deployed law`
- **Kind:** `phase2_g19b_math_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** Even when endpoint value factors through K31, its fresh local
  scalar action must be determined by the law information actually available
  at deployment.  Mean/covariance/third-order state is insufficient for a
  general law.

Let `D_l(mu)` be the deployed observation of an admissible layer law `mu`, and
let

\[
\phi_l(\mu)=
\langle A_l,K31(\mu)\rangle
=\mathbb E_\mu[h_3(z)^TA_lz].
\]

**Necessary and sufficient condition.**  An exact deterministic fresh
provider `f_l` with `phi_l=f_l o D_l` exists if and only if `phi_l` is constant
on every fiber of `D_l`:

\[
D_l(\mu)=D_l(\nu)
\quad\Longrightarrow\quad
\phi_l(\mu)=\phi_l(\nu).
\tag{C.1}
\]

For a linear signed-measure model this is

\[
\ker D_l\subseteq\ker\phi_l.
\tag{C.2}
\]

The lower-moment-only subclaim is already closed false.  A standardized
Gaussian and standardized Rademacher scalar have identical moments through
order three but fourth moments `3` and `1`.  Embedding either law along a
direction `u` produces a weighted K31-action difference proportional to

\[
2\sum_{ij}A_{ij}u_i^3u_j.
\]

Unless the Bellman row annihilates every admissible repeated-index quartic,
one may choose `u` for which this is nonzero.  Thus no universal
mean/covariance/K3 formula supplies the action.

Two sufficient interfaces are immediate: an exact K31 oracle, or a complete
weighted atom law `{(p_s,z_s)}` from which the expectation is evaluated
directly.  Whether either is lawful and present in the deployed lineage is a
candidate-specific fact, not implied by Node A.

- **Terminal criterion:** Specify the exact deployed `D_l`.  Prove (C.1) for
  every active layer, or give two admissible same-`D_l` laws with different
  actions.  The lower-moment-only branch remains `closed_proof`; the parent
  node stays `open` until the actual fresh-law interface is resolved.
- **Evidence:** This file, sections `General-law moment acquisition cost`,
  `When the obstruction is merely moved into moment computation`, and `Node C
  -- fresh moment identifiability`.
- **Decomposition children:** none.
- **Goal-path summary:** Failure here moves the H4 memory obstruction into an
  unobserved fresh moment; it does not constitute a scalar-state bypass.

### Node D -- local provider rank and cost

- **Suggested id:** `P2R0G19B_MATH_K31_PROVIDER_COST`
- **Label:** `Fresh K31 action fits the ten-unit provider budget`
- **Kind:** `phase2_g19b_math_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** A scalar result is cheap only if the Bellman contraction can be
  fused before the charged external bridge.  Materializing eleven generic
  q4 directions and contracting afterward still costs eleven directions per
  layer.

Let `C_l` denote the charged cut between the available law representation and
the Bellman consumer.  The exact local provider rank is the operator Schmidt
rank of the action tensor across `C_l`, after quotienting all already shared
features:

\[
r_{\mathrm{action},l}
=\operatorname{rank}_{C_l}
\big(A_l\circ M_l\circ S_l\big).
\tag{D.1}
\]

This rank, not the one-dimensional output, is the minimum number of exact
directions that must cross that interface.  A one-direction bridge is lawful
if and only if the charged-cut matricization in (D.1) has rank at most one and
the corresponding factors are available on their respective sides.

There is an important alternative contraction order when Node C supplies a
weighted atom law.  With `Z` and `H_3(Z)` holding the atom rows,

\[
\phi_l
=\sum_sp_sh_3(z_s)^TA_lz_s
\]

can be evaluated by one rectangular dense multiply, for example
`H_3(Z) A_l`, followed by a weighted Frobenius contraction with `Z`.  Its
leading cost is `2N_ln^2`, or `N_l/n` square-GEMM equivalents under the same
flop convention.  Therefore `N_l<=n` gives at most one such equivalent per
active layer without ever forming the K31 plane.  This statement assumes the
atom table and dense `A_l` are already lawfully available on the same side of
the charged cut.

By contrast, if the interface first exposes the previously derived eleven
generic fused-pair Kraus directions, post-contraction does not reduce their
acquisition cost:

\[
G_{\rm q4}=\sum_{l=5}^{14}11=110.
\]

The complete exact cost law is

\[
G_{\rm action}
=G_{\rm Bellman}+G_{\rm law}
+\sum_{l=5}^{14}c_l r_{\rm action,l},
\tag{D.2}
\]

where `c_l` is the billed cost of one direction at layer `l`.  In the
normalized ten-unit residual budget, deployability is exactly

\[
G_{\rm action}\le10.
\tag{D.3}
\]

Ten nonzero atomic layers at one unit each saturate the budget, so the
Bellman rows and law access must then be free, shared with the primary module,
or cheaper than the conservative unit bound.  A hidden dense reverse pass is
not free merely because the final accumulator is scalar.

- **Terminal criterion:** Fix the actual charged cut and contraction order;
  prove the ten local ranks/costs and all Bellman/law overhead satisfy (D.3),
  or close the route by a lower bound above ten.  Do not assign rank one from
  scalar output dimension alone.
- **Evidence:** This file, sections `Provider action rank`, `General-law moment
  acquisition cost`, and `Node D -- local provider rank and cost`.
- **Decomposition children:** none.
- **Goal-path summary:** This distinguishes a real fused expectation bypass
  from the already cost-closed 11-direction q4 provider followed by a scalar
  dot product.

### Node E -- endpoint error certificate

- **Suggested id:** `P2R0G19B_MATH_K31_ENDPOINT_CERT`
- **Label:** `K31 Bellman route endpoint-error theorem`
- **Kind:** `phase2_g19b_math_theorem`
- **Status:** `solved`
- **Priority:** `1`
- **Summary:** Factorization, moment, Bellman, and higher-order errors compose
  into one endpoint prediction norm.  The ideal `9.2547e-9` ladder beats raw
  `1e-8` under the sharp worst-case norm gate `eta<=3.80e-6`.

Let `r_fact=G-M^*A` on the reachable source history, let `e_l` be fresh K31
acquisition error, and let `Delta A_l` be Bellman-row error.  Then

\[
\eta_{31}
\le \epsilon^2\left[
\|r_{\rm fact}(q)\|_2
+\sum_l\left(
\|A_l\|\,\|e_l\|
+\|\Delta A_l\|\,\|M_lq_l\|
+\|\Delta A_l\|\,\|e_l\|
\right)\right]
+\eta_{\ge3}.
\tag{E.1}
\]

One may replace the first term by the exact covariance-weighted projection
tail from (B.7).  The pure q4 multiple-insertion contribution satisfies

\[
|R_{q4-q4}|
\le\frac12\epsilon^4H_V
\left(\sum_l\|M_lq_l\|\right)^2,
\]

and any H3--q4 `O(epsilon^3)` term must be added unless removed by a proved
parity or orthogonality identity.

For ideal residual `e_ideal` and implementation deviation `d`, the exact raw
criterion is

\[
R_{\rm impl}
=R_{\rm ideal}+2\langle e_{\rm ideal},d\rangle+\|d\|_2^2
<10^{-8}.
\tag{E.2}
\]

If only `||d||_2<=eta_31` is known, the sharp worst-case certificate is

\[
R_{\rm impl}
\le(\sqrt{R_{\rm ideal}}+\eta_{31})^2.
\tag{E.3}
\]

At `R_ideal=9.2547e-9`, (E.3) requires

\[
\eta_{31}
<10^{-4}-\sqrt{9.2547\times10^{-9}}
\simeq3.80\times10^{-6}.
\tag{E.4}
\]

The independent channel-value gate remains

\[
4.370\,\alpha_3+1.0313\,\alpha_4\ge4.656
\qquad\text{(all coefficients in }10^{-9}\text{)}.
\tag{E.5}
\]

Thus with perfect H3, the K31/shared quartic mechanisms must retain at least
`alpha_4=0.27732`; with imperfect H3 the required quartic fraction is

\[
\alpha_4
\ge\frac{4.656-4.370\alpha_3}{1.0313}.
\]

- **Terminal criterion:** The composition theorem and sharp norm gate are
  proved, so this theorem node is `solved`.  The candidate may claim the gate
  only after Nodes B--D provide numerical/symbolic bounds for every term in
  (E.1) and the retained fractions in (E.5).
- **Evidence:** This file, sections `Value gate`, `Total endpoint-error
  certificate`, and `Node E -- endpoint error certificate`.
- **Decomposition children:** none.
- **Goal-path summary:** This is the final theorem used to translate the three
  open K31 feasibility gates into raw-MSE authority.

### Recommended graph edges

Add decomposition edges from `P2R0G19B_MATH_CAUSAL_PAIR_FIELD` to Nodes A--E.
The internal theorem/gate edges are:

1. A `informs` B: the local endpoint rows are defined by the scalar costate
   theorem.
2. B `informs` C and D: only the K31-projected Bellman rows need fresh-law and
   provider audits.
3. A, B, C, and D each `informs` E: all four supply terms or premises in the
   endpoint certificate.
4. E `informs` `P2R0G19B_MATH_PORTFOLIO_GRAM`: only a route passing (E.4) and
   (E.5) should receive portfolio value credit.

If the graph requires hard dependency rather than informational edges, use
`requires` from B, C, and D to the eventual deployable K31 candidate, not to
the already solved theorem Node E.

### Checkpoint-8 disposition and most actionable successor

- A: `solved` as a target-specific order-`1/n` state theorem.
- B: `open`; generic dense resets violate K31 row-space invariance, while the
  restricted reachable quotient has not yet been proved to do so.
- C: `open`; lower-moment-only identifiability is `closed_proof`, but an exact
  atom/oracle interface would pass.
- D: `open`; the eleven-direction post-contraction route is `closed_cost`,
  while pre-bridge atomic fusion remains conditional.
- E: `solved` as an error-composition theorem; candidate-specific satisfaction
  is conditional on B--D.

The highest-value next symbolic object is not another recurrence.  It is the
per-layer residual quartic covector

\[
r_l=(I-P_l)J_l^*T_{l+1:L}^*c
\tag{B.8}
\]

restricted to the actual NCQC/q4(211) source quotient.  Normal-order (B.8)
into K31, q4(211) shared-module, `2+2`, and primitive/all-distinct blocks.  A
zero residual proves the exact bypass.  A nonzero residual immediately gives
both the smallest target-specific extra moment row and, through (B.7), the
maximum K31-only value.  This single row-space derivation resolves Node B
without constructing or timing a long-running program.

## Checkpoint 8B: exact K31 rank on the structured q4(211) quotient

The dense-reset obstruction in Node B applies to the full quartic space.  It
does **not** imply failure on one shared q4(211) matrix word.  On that
structured subspace the K31 map has an explicit inverse under a mild support
condition.

Use the polynomial normalization

\[
D(M;v,w)[x,x,x,x]
=(x^TMx)(v^Tx)(w^Tx),
\]

where `M` is symmetric and `v,w` are known singleton factors.  Let

\[
K_{ij}=D(M;v,w)[e_i,e_i,e_i,e_j].
\]

Coefficient extraction gives

\[
K_{ii}=M_{ii}v_iw_i,
\tag{B.9}
\]

and, for `i!=j`,

\[
4K_{ij}
=M_{ii}(v_iw_j+w_iv_j)+2M_{ij}v_iw_i.
\tag{B.10}
\]

Other tensor normalizations multiply both equations by known nonzero
constants and do not affect rank.

Set

\[
a_i=v_iw_i,
\qquad
b_{ij}=v_iw_j+w_iv_j.
\]

If every `a_i` is nonzero, then K31 reconstructs the complete matrix word:

\[
M_{ii}=K_{ii}/a_i,
\]

\[
M_{ij}
=\frac{4K_{ij}-b_{ij}K_{ii}/a_i}{2a_i}.
\tag{B.11}
\]

Consequently

\[
\ker\big(M_{31}|_{\{D(M;v,w):M=M^T\}}\big)=\{0\}
\]

under full coordinate support.  Node B then passes automatically for every
endpoint covector on this one-word q4(211) module: injectivity is stronger
than the needed endpoint kernel inclusion.

Full support is sufficient, not necessary.  The exact necessary and
sufficient condition is that the only symmetric `M` satisfying

\[
a_iM_{ii}=0,
\]

\[
b_{ij}M_{ii}+2a_iM_{ij}=0
\qquad(i\ne j)
\tag{B.12}
\]

is `M=0`; the equation with `i,j` reversed is included separately.  This is a
pure symbolic row-rank test.  In particular, any off-diagonal block supported
on two or more indices with `a_i=0` is invisible unless the diagonal-coupling
equations remove it.

The same result covers several known singleton branches sharing the **same**
matrix word.  For

\[
D_{\rm eff}(M)
=\sum_b\beta_bD(M;v_b,w_b),
\]

replace

\[
a_i=\sum_b\beta_bv_{b,i}w_{b,i},
\qquad
b_{ij}=\sum_b\beta_b
(v_{b,i}w_{b,j}+w_{b,i}v_{b,j}).
\tag{B.13}
\]

Equations (B.9)--(B.12) remain valid.  Thus branch count does not increase the
K31 identification rank while all branches share one `M`.  Independent
matrix words are different: their direct-sum dimension grows, and their
combined K31 column map must be audited.

### Primitive residual rank certificate

Let `U_l` be the actual value-visible primitive/q4-perpendicular source
subspace at a layer, with a fixed basis `T_{l,1},...,T_{l,r_l}`.  Define its
K31 sampling matrix

\[
(C_l)_{(i,j),a}=T_{l,a}[e_i,e_i,e_i,e_j].
\tag{B.14}
\]

Then

\[
M_{31}|_{U_l}\text{ is injective}
\quad\Longleftrightarrow\quad
\operatorname{rank}C_l=r_l
\quad\Longleftrightarrow\quad
C_l^*C_l\succ0.
\tag{B.15}
\]

If (B.15) holds, K31 factors **every** endpoint row on `U_l`, irrespective of
the eleven-direction transport rank.  The eleven-direction cost no-go and the
K31 sufficiency question are therefore distinct: a high-rank state can still
be identified by one sufficiently rich fresh observation, provided that
observation itself is lawfully cheap.

If `U_l` is defined in the birth frame as a collision-free primitive V4 space
whose repeated-index entries all vanish, then `C_l=0`; immediate same-frame
K31 is provably useless.  After a dense reset `K`, the relevant matrix is

\[
(C_l(K))_{(i,j),a}
=\big(\Gamma_4(K)T_{l,a}\big)[e_i,e_i,e_i,e_j].
\tag{B.16}
\]

For `r_l<=n^2`, full column rank is generic once one reset orientation with a
nonzero `r_l`-minor exists, because that minor is a polynomial in the entries
of `K`.  However, exploiting (B.16) without transporting `U_l` requires a
genuinely fresh post-reset law interface.  Computing the post-reset K31 from
the missing primitive tensor is precisely the old carrier problem.

If `h_l` is the endpoint row in the `U_l` basis, the exact factor test is

\[
h_l\in\operatorname{row}C_l,
\tag{B.17}
\]

and the irreducible coefficient residual is

\[
h_l^\perp
=h_l\left(I-C_l^*(C_lC_l^*)^\dagger C_l\right),
\tag{B.18}
\]

with the evident transpose change under column-vector convention.  A
source-covariance-weighted version of (B.18) supplies the value tail in
(B.7).

### Checkpoint-8B implication

Node B can now be narrowed rather than treated as one undifferentiated open
claim:

1. **Shared q4(211) matrix word:** `solved conditional`; it factors through
   K31 whenever (B.12) has trivial kernel, with full singleton support as a
   simple sufficient certificate.
2. **Same-frame collision-free primitive V4:** `closed_proof`; its K31 map is
   zero by definition.
3. **Fresh post-reset primitive quotient:** `open`; evaluate the symbolic rank
   certificate (B.15)--(B.17) and separately prove that the fresh law
   interface supplies it without recreating H4 transport.

The most actionable candidate-specific proof is therefore a small exact
matrix-rank derivation, not an endpoint run: write the at-most-eleven primitive
basis tensors, form their symbolic `iiij` columns after the lawful fresh-law
boundary, and exhibit either one nonzero full minor or one explicit
endpoint-visible kernel vector.  This simultaneously resolves the remaining
part of Node B and hands the correct observable to Nodes C and D.

## Checkpoint 8C: causal K31 observability and fused covariance action

### Birth coordinates versus fresh-measurement coordinates

Equations (B.3)--(B.5) are the correct local test when `q_l` denotes the
complete fresh quartic input consumed at layer `l`.  If instead `q_t` denotes
a primitive quartic **birth** whose descendants are measured at later layers,
the K31 map is not block diagonal.  Let `U_{l<-t}^{(4)}` be its order-four
transport from birth boundary `t` to fresh-law boundary `l`, including all
known linearized resets.  Then

\[
m_l
=\sum_{t\le l}H_{lt}q_t,
\qquad
H_{lt}=M_lU_{l\leftarrow t}^{(4)}.
\tag{B.19}
\]

Stacking all `m_l` gives the block-lower-triangular causal observability map

\[
\mathcal O_{31}=(H_{lt})_{l\ge t}.
\]

The exact birth-history theorem is

\[
\ker(\mathcal O_{31}|_R)
\subseteq\ker(G|_R)
\quad\Longleftrightarrow\quad
G|_R\in\operatorname{range}
\big((\mathcal O_{31}|_R)^*\big).
\tag{B.20}
\]

Thus a collision-free V4 birth may have `H_tt=0` and still be recoverable from
later blocks `H_lt`.  The minimum-norm Bellman weights solve

\[
\mathcal O_{31}^*A=P_{\operatorname{range}\mathcal O_{31}^*}G.
\tag{B.21}
\]

This is the correct joint block-Hankel/observability version of Node B.  It
also clarifies the deployment issue.  If a lawful fresh-law interface directly
measures each physical `m_l`, the estimator need not transport the primitive
birth merely because (B.19) contains its physical propagation.  If the
estimator must *predict* `m_l` from its own lower-order state, evaluating the
off-diagonal blocks `H_lt` recreates the missing H4 carrier unless Node C
provides another exact fresh-law identity.

The value-projection formula (B.7) remains valid after replacing `M` by
`O_31`.  Independent per-layer K31 fits are generally suboptimal or illegal:
the weights must solve the one joint adjoint equation (B.21).

### Universal local covariance-action identity

There is a common and favorable consumer geometry for the K31 slice.  Let a
coordinatewise nonlinearity `phi` act on a Gaussian baseline `G` with the
current covariance.  The order-four cumulant expansion for
`F_ij(z)=phi(z_i)phi(z_j)` is

\[
\delta\mathbb E[F_{ij}]
=\frac1{4!}\sum_{a,b,c,d}
\kappa_{abcd}\,
\mathbb E_G[\partial_{abcd}F_{ij}(G)].
\]

The directed K31 terms are therefore

\[
\delta C^+_{ij}\big|_{31}
=W_{ij}K31_{ij}+W_{ji}K31_{ji},
\tag{D.4}
\]

where

\[
W_{ij}=\frac16
\mathbb E_G[\phi'''(G_i)\phi'(G_j)].
\tag{D.5}
\]

For nonsmooth activations the same statement is interpreted by Hermite
coefficients or weak derivatives.  No smoothness assumption is needed once
`W` is defined as the exact linear K31 response kernel.

Let `Lambda^+` be the symmetric covariance costate after this consumer.  Then

\[
\langle\Lambda^+,\delta C^+|_{31}\rangle
=\langle A,K31\rangle,
\]

with

\[
A=2\,\Lambda^+\odot W
\tag{D.6}
\]

under a full directed-matrix Frobenius convention; half-matrix conventions
only change the known factor of two.  Equation (D.6) proves three useful
facts:

1. the Bellman K31 row is an elementwise readout of the already needed H2
   costate, not an H4 reverse matrix;
2. constructing `A` costs `O(n^2)` once `Lambda^+` and `W` are available; and
3. injecting the K31 action into the covariance endpoint does not require a
   second transported H4 stream.

More generally, these conclusions hold whenever the local quartic consumer
is a known linear map `B_l:K31_l -> Sym_n` and

\[
A_l=B_l^*\Lambda_{l+1}.
\tag{D.7}
\]

The Hadamard formula is the coordinatewise-cumulant specialization.

### Conditional 38-GEMM realization

Suppose all of the following hold:

1. the covariance costate propagates through each dense reset as
   `Lambda_l=K_l^T Lambda_{l+1}K_l` and can occupy the primary shared matrix
   word;
2. there are fourteen charged covariance-adjoint transitions, at two square
   GEMMs each;
3. ten layers require fresh K31 actions;
4. each layer supplies an exact weighted atom table with `N_l<=n`; and
5. `W_l` and all elementwise operations are lower-order in the billing model.

Then an exact order-`1/n` realization uses at most

\[
G_{\rm adjoint}=2\times14=28,
\]

\[
G_{\rm fresh\ K31}
=\sum_{l=5}^{14}N_l/n\le10,
\]

and hence

\[
G_{\rm total}\le38<49.
\tag{D.8}
\]

It leaves eleven square-GEMM equivalents for nonshared law/kernel overhead or
another proved residual.  The scalar actions use

\[
\phi_l
=\sum_sp_s h_3(z_s)^TA_lz_s,
\]

evaluated by one `N_l by n` times `n by n` rectangular multiply plus an
`O(N_ln)` contraction.  Neither the full K31 plane nor the eleven primitive
Kraus directions is materialized.

Equation (D.8) is a conditional realization theorem, not yet a deployability
certificate.  It fails if the primary matrix word is already committed to an
incompatible forward state, if a separate backward costate is billed, if the
law has `N_l>n`, if atom access is unavailable, or if computing `W_l` incurs a
dense unshared chain.  These are exactly the remaining Node C/D facts to
resolve.

### Checkpoint-8C consequence for the five graph nodes

- Node A remains solved: (D.6) supplies its local action row without H4
  reverse state.
- Node B must use the global causal test (B.20) for primitive births; its
  diagonal specialization remains correct for independently supplied fresh
  layer laws.
- Node C is now the central physical question: are the later K31 measurements
  in (B.19) genuinely fresh observables, or predictions that require the
  missing primitive state?
- Node D has a concrete conditional `38<49` construction.  Its remaining
  proof obligation is interface/cost provenance, not quartic tensor algebra.
- Node E consumes the observability projection tail of (B.20) and any fresh
  moment errors exactly as already stated.

## Checkpoint 9: D11/Q24 novelty correction -- strict-depth residual K31 only

### Binding correction to Checkpoints 8B--8C

The deployed D11/Q24 lineage already owns the same-site total K31 consumer.
Therefore equations (B.9)--(B.13) remain valid identifiability facts, but their
same-layer reconstruction has **zero incremental novelty credit**.  Likewise,
the diagonal blocks of the observability map in Checkpoint 8C must not be
counted as new value.  The only potentially new K31 route is the
true-minus-assembled action generated by delayed collision-free births and
seen through strict-lower, genuinely fresh-law blocks.

This section supersedes any novelty or economic interpretation of a total
same-site K31 action in Checkpoints 8B--8C.  It does not alter the scalar
Bellman theorem, the structured rank identities, or the endpoint error
theorem.

### Exact strict-depth adjoint identity

Let an order-four action word obey

\[
z_{l+1}=T_lz_l+b_l,
\]

and let the K31 consumer at depth `k` be the linear map `C_k`, paired with the
endpoint/Bellman row `B_k`.  The full response of all births contains
zero-length and delayed terms.  Its potentially novel part is exactly

\[
\Delta_<
=\sum_{j<k}
\left\langle
B_k,
C_kT_{k-1}\cdots T_{j+1}b_j
\right\rangle.
\tag{N.1}
\]

The `j=k` term

\[
\langle B_j,C_jb_j\rangle
\]

is the already-owned same-site D11/Q24 action and must be subtracted.

Equivalently, define the full reverse costate by

\[
\lambda_j=T_j^*\lambda_{j+1}+C_j^*B_j.
\tag{N.2}
\]

The future-only novelty row at birth `j` is

\[
\mu_j
=\lambda_j-C_j^*B_j
=T_j^*\lambda_{j+1},
\tag{N.3}
\]

and

\[
\Delta_<=\sum_j\langle\mu_j,b_j\rangle.
\tag{N.4}
\]

Equations (N.1)--(N.4) are an exact duplicate audit.  Any derivation that uses
`lambda_j` rather than `mu_j` to assign candidate value gives the new route
credit for the incumbent zero-length term.

### True-minus-assembled strict-lower operator

Let

\[
H^{\rm true}_{kj}
=C_kT^{\rm true}_{k-1}\cdots T^{\rm true}_{j+1},
\]

and let `H^asm_kj` be the corresponding action already assembled by the
deployed lineage.  For delayed blocks define

\[
\Delta H_{kj}
=H^{\rm true}_{kj}-H^{\rm asm}_{kj},
\qquad j<k,
\tag{N.5}
\]

and set every diagonal block to zero.  The candidate observation is the
aggregate fresh residual

\[
e_k^<
=m_k^{\rm true}-m_k^{\rm asm}
=\sum_{j<k}\Delta H_{kj}b_j,
\tag{N.6}
\]

after removing any non-delayed component already owned by D11/Q24.  Stack the
strict-lower blocks into

\[
\Delta\mathcal O_<
=\big(\Delta H_{kj}\,\mathbf 1_{j<k}\big)_{k,j}.
\tag{N.7}
\]

If `G_delta` is the complete true-minus-assembled endpoint row on the birth
history, the exact delayed-K31 factorization condition is

\[
\ker\big((\Delta\mathcal O_<)|_R\big)
\subseteq
\ker(G_\Delta|_R).
\tag{N.8}
\]

For the narrower target consisting **only** of the delayed K31-mediated
response in (N.1), factorization is tautological:

\[
G_{<,j}
=\sum_{k>j}(\Delta H_{kj})^*B_k,
\qquad
G_<=\Delta\mathcal O_<^*B.
\tag{N.9}
\]

The scientific question is whether (N.9) spans enough of the complete oracle
residual `G_delta`, not whether a K31-defined target factors through itself.

### Conditional novelty relative to the incumbent feature span

Subtracting one assembled scalar is not by itself a complete duplicate audit.
Let

\[
F:R\longrightarrow Z_0
\]

be the full frozen D11/Q24 feature map already available to the estimator.
An added strict-depth observation is exact for the complete endpoint row `G`
if and only if

\[
\ker F\cap\ker\Delta\mathcal O_<
\subseteq\ker G,
\tag{N.10}
\]

or equivalently

\[
G\in
\operatorname{range}(F^*)
+\operatorname{range}(\Delta\mathcal O_<^*).
\tag{N.11}
\]

This is the correct row-space gate for incremental value.  Let `Sigma` be the
birth-history covariance in the residualized value ensemble and define its
conditional covariance after all incumbent features,

\[
\Sigma_{\perp F}
=\Sigma
-\Sigma F^*(F\Sigma F^*)^\dagger F\Sigma.
\tag{N.12}
\]

The sharp incrementally capturable fraction of the incumbent-residual target
is

\[
p_<
=\frac{
(\Delta\mathcal O_<\Sigma_{\perp F}g)^*
(\Delta\mathcal O_<\Sigma_{\perp F}
 \Delta\mathcal O_<^*)^\dagger
(\Delta\mathcal O_<\Sigma_{\perp F}g)
}{g^*\Sigma_{\perp F}g}.
\tag{N.13}
\]

If `D_=` denotes the diagonal/same-site K31 observation and its rows are in
`range(F^*)`, then

\[
D_=\Sigma_{\perp F}=0.
\tag{N.14}
\]

Consequently adding `D_=` to `Delta O_<` changes neither (N.10) nor (N.13).
It can consume arithmetic, but it cannot receive any incremental value credit.

### Updated no-hidden-L-squared cost gate

Let `I_<` be the set of consumer layers with a nonzero, incumbent-residual
strict-depth action.  If a lawful fresh layer law returns the **aggregate**
residual (N.6), one contraction at layer `k` sums all past births at once.  For
an atom law with `N_k` atoms, its cost remains `N_k/n` square-GEMM equivalents,
not one cost for every pair `j<k`:

\[
G_{\rm residual\ law}
\le\sum_{k\in I_<}\frac{N_k}{n}.
\tag{N.15}
\]

The already assembled scalar is reused and subtracted in `O(n^2)` or less.
But if the estimator constructs each `Delta H_kj b_j` separately in order to
manufacture (N.6), its cost is

\[
\sum_k\sum_{j<k}G_{kj},
\tag{N.16}
\]

which is generically `O(L^2)` and is the forbidden transported-H4 computation
under another schedule.

Across the charged fresh-law cut, let

\[
r_{<,k}
=\operatorname{rank}_{\rm cut}
\left(B_k\circ e_k^<\right)
\]

after quotienting the complete incumbent feature span.  The exact incremental
cost is

\[
G_<
=G_{\rm future\ costate,new}
+G_{\rm fresh\ law,new}
+\sum_{k\in I_<}c_kr_{<,k}.
\tag{N.17}
\]

A deployable residual route requires

\[
G_<\le G_{\rm residual\ budget}
\tag{N.18}
\]

and sufficient incremental value from (N.13).  The earlier conditional
`28+10=38<49` construction remains a valid upper bound only if the 28-GEMM
future covariance costate is genuinely shared/newly authorized and each
fresh contraction produces (N.6).  It cannot be cited as evidence that total
same-site K31 is a new feature.

Let `Delta R_<^oracle` be the raw-MSE reduction available from the complete
incumbent-residual delayed K31 target.  The value gate is an **absolute
incremental** inequality,

\[
p_<\,\Delta R_<^{\rm oracle}
\ge
4.656\times10^{-9}
-4.370\times10^{-9}\alpha_3
-\Delta R_{\rm other,new}.
\tag{N.19}
\]

The old `27.732%` quartic threshold follows only when
`Delta R_<^oracle=1.0313e-9`, `alpha_3=1`, and no other new quartic correction
is credited.  Because diagonal K31 is incumbent-owned, that denominator must
be verified on the residual true-minus-assembled target before reusing the
percentage.

## Corrected graph-ready novelty nodes

### Node N1 -- strict-depth novelty theorem

- **Suggested id:** `P2R0G19B_MATH_K31_STRICT_NOVELTY`
- **Label:** `K31 novelty is strict-depth only`
- **Kind:** `phase2_g19b_math_theorem`
- **Status:** `solved`
- **Priority:** `1`
- **Summary:** D11/Q24 owns every zero-length same-site K31 action.  The only
  new action is the `j<k` sum (N.1), equivalently the full reverse costate with
  `C_j^*B_j` subtracted at each birth.
- **Terminal criterion:** Retain solved while the candidate is evaluated on
  the exact deployed D11/Q24 lineage; any future lineage change must redo the
  incumbent-span audit.
- **Evidence:** This file, equations (N.1)--(N.4).
- **Decomposition children:** none.
- **Goal-path summary:** Enforces zero value credit for diagonal K31 and
  defines the delayed residual target consumed by N2--N4.

### Node N2 -- delayed residual row-space gate

- **Suggested id:** `P2R0G19B_MATH_K31_DELAYED_FACTOR`
- **Label:** `Complete residual value factors through delayed K31`
- **Kind:** `phase2_g19b_math_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** The strict-lower true-minus-assembled K31 observations add all
  missing endpoint value exactly iff the incumbent-conditional kernel
  inclusion (N.10) holds; their sharp partial value is (N.13).
- **Terminal criterion:** Derive the actual strict-lower blocks on the
  collision-free reachable quotient and prove (N.10), or exhibit one
  endpoint-visible joint kernel witness.  If partial, report (N.13) against
  the residual oracle denominator in (N.19).
- **Evidence:** This file, equations (N.5)--(N.14).
- **Decomposition children:** none.
- **Goal-path summary:** This replaces the unresidualized Node-B test as the
  only lawful K31 value gate.

### Node N3 -- delayed fresh-law identifiability

- **Suggested id:** `P2R0G19B_MATH_K31_DELAYED_FRESH_LAW`
- **Label:** `Delayed K31 residual is a genuine fresh observable`
- **Kind:** `phase2_g19b_math_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** The aggregate residual (N.6) must be identifiable from a lawful
  post-reset law interface without transporting the primitive birth or using
  truth unavailable to the estimator.
- **Necessary and sufficient condition:** If `D_k` is the allowed fresh-law
  data and `phi_k^<=<B_k,e_k^<>`, then
  `D_k(mu)=D_k(nu)` must imply `phi_k^<(mu)=phi_k^<(nu)` on the admissible law
  class.  In a linear signed-law model this is
  `ker D_k subset ker phi_k^<`.
- **Terminal criterion:** Prove fiber constancy for the exact deployed law
  interface, or close with a same-interface pair having different delayed
  residual actions.  A computation from the missing V4 birth does not count
  as fresh.
- **Evidence:** This file, equations (N.5)--(N.7), (N.15)--(N.16), and the
  Node-C fiber theorem.
- **Decomposition children:** none.
- **Goal-path summary:** Decides whether strict-lower observability is a real
  information bypass or merely H4 transport relabeled as moment acquisition.

### Node N4 -- incremental residual cost/value gate

- **Suggested id:** `P2R0G19B_MATH_K31_DELAYED_COST`
- **Label:** `Delayed K31 residual clears cost and raw-value gates`
- **Kind:** `phase2_g19b_math_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** Only incumbent-residual fresh contractions are billed and
  credited.  Aggregate fresh acquisition can remain `O(L)`; pairwise
  reconstruction is hidden `O(L^2)`.  The exact gates are (N.17)--(N.19).
- **Terminal criterion:** Supply a charged-cut factorization with (N.18), no
  diagonal value credit, no pairwise hidden work, and enough conditional
  capture from (N.13) to pass (N.19); otherwise close by cost or value.
- **Evidence:** This file, equations (N.13)--(N.19).
- **Decomposition children:** none.
- **Goal-path summary:** Converts N2/N3 into an incremental raw-MSE claim
  without paying for or double-counting the incumbent K31 path.

### Corrected edges and statuses

Use decomposition edges from `P2R0G19B_MATH_CAUSAL_PAIR_FIELD` to N1--N4.
N1 `informs` N2, N3, and N4.  N2 and N3 each `informs` N4.  N4 `informs` the
endpoint-certificate Node E and the portfolio Gram node.  The earlier general
Nodes B--D may remain as background theorems, but the deployable candidate
must depend on N2--N4, not on their unresidualized total-K31 versions.

Current statuses are N1 `solved`; N2 `open`; N3 `open`; N4 `open`.  The next
symbolic target is the strict-lower residual row

\[
G_{<,j}
=\sum_{k>j}
(H^{\rm true}_{kj}-H^{\rm asm}_{kj})^*B_k
\]

on the collision-free primitive basis, followed by incumbent conditional
projection (N.12).  No diagonal `k=j` coefficient belongs in that derivation.

## Checkpoint 9B: one-reset primitive row and the D14+D10 portfolio gate

### Explicit first delayed collision-free row

Let `V4` be the coordinate collision-free subspace spanned by symmetric basis
tensors with four distinct indices.  For one dense reset `K`, define the K31
covector associated with a directed Bellman matrix `A` by

\[
S_{31}(A)
=\sum_{i,j}A_{ij}
\operatorname{sym}(e_i^{\otimes3}\otimes e_j).
\]

The first delayed row on `V4` is

\[
g_{K,A}^{V4}
=P_{V4}\,\Gamma_4(K)^*S_{31}(A).
\tag{N.20}
\]

For four distinct source indices `a,b,c,d`, its coefficient is, up to the
fixed symmetrization normalization,

\[
\begin{aligned}
g_{abcd}
=\sum_{i,j}A_{ij}\big(&
K_{ia}K_{ib}K_{ic}K_{jd}
+K_{ia}K_{ib}K_{id}K_{jc}\\
&+K_{ia}K_{ic}K_{id}K_{jb}
+K_{ib}K_{ic}K_{id}K_{ja}
\big).
\end{aligned}
\tag{N.21}
\]

If `k_a` denotes column `a` of `K`, this is the compact four-term identity

\[
g_{abcd}
=\sum_{r\in\{a,b,c,d\}}
\left(
\mathop{\odot}_{s\in\{a,b,c,d\}\setminus\{r\}}k_s
\right)^TAk_r.
\tag{N.22}
\]

The direct same-site row is zero on `V4`; (N.20) is therefore automatically
strict-depth.  For generic dense `K` and nonzero `A`, (N.21) is nonzero.  It
also displays why delayed K31 can see a collision-free birth after one reset,
and why computing its action directly from the birth is a primitive-H4
contraction rather than a free K31 slice.

On an `r<=11` primitive basis, the exact first-delay observation matrix is
obtained by evaluating (N.21) on those basis tensors.  Full column rank of
that matrix proves first-delay observability; an endpoint row outside its row
space gives the explicit residual required by N2.

### Sharp raw-capacity ceiling after duplicate removal

Let

\[
V_{<}^{\rm oracle}
=g^*\Sigma_{\perp F}g
\]

be the incumbent-residual value energy available in the complete delayed
target, and let `p_<` be (N.13).  The strict-depth route's maximum raw-MSE
reduction is

\[
\Delta R_{<}^{\max}
=p_<\,\Delta R_{<}^{\rm oracle}
\le\Delta R_{<}^{\rm oracle}.
\tag{N.23}
\]

No diagonal K31 value is included in either side.  With perfect H3 and no
other new channel, raw `<1e-8` requires

\[
\Delta R_{<}^{\max}\ge0.2860\times10^{-9}.
\tag{N.24}
\]

More generally the right side is the one in (N.19).  If, and only if, the
verified residual oracle denominator is the full `1.0313e-9` quartic ladder
gain, (N.24) is the familiar `p_<>=0.27732` fraction.  The capacity claim is
otherwise `Delta R_<^oracle`, not the total quartic number.

### Exact one-unit incremental-cost condition

The current `D14+D10` schedule consumes

\[
2\times14+2\times10=48
\]

square-GEMM equivalents, leaving only one additive unit.  Under the explicit
atom-law contraction of (N.15), an additive strict-depth route fits if and
only if

\[
G_{\rm unshared}
+\sum_{k\in I_<}\frac{N_k}{n}
\le1.
\tag{N.25}
\]

Stacking the depths into one library call does not change (N.25): the dense
flop count remains `2n^2 sum_k N_k`.  In the ordinary `N_k=n` interface, two
informative consumer layers already cost two units, so every genuinely
multi-layer form is excluded from the one-unit additive slot.

If exactly one layer may be acquired at unit cost, its sharp conditional
capture is

\[
p_{1,\rm layer}
=\max_{k\in I_<}
\frac{
(O_k\Sigma_{\perp F}g)^*
(O_k\Sigma_{\perp F}O_k^*)^\dagger
(O_k\Sigma_{\perp F}g)
}{g^*\Sigma_{\perp F}g},
\tag{N.26}
\]

where `O_k` is row block `k` of `Delta O_<`.  This is the capacity ceiling for
a literal one-layer add-on.  Calling the sum of all Bellman actions one scalar
does not make (N.26) equal (N.13), because acquiring that scalar still crosses
all of the charged layer-law cuts.

There are only three mathematical exceptions to the multi-layer cost lower
bound:

1. the deployed interface already supplies every Bellman-contracted delayed
   residual scalar at `O(n^2)` total extra work;
2. a proved joint charged-cut rank-one identity evaluates all layers with
   total atom count `sum_k N_k<=n`; or
3. one consumer layer alone passes both (N.24) and (N.26).

These are information/factorization premises, not consequences of scalar
output dimension.  They are the weakest premises under which any strict-depth
form can fit additively at `C<=1`.

### Replacement economics

Replacing, rather than augmenting, the existing D10 second stream frees
twenty GEMM equivalents.  If the D14 primary word is the required future
covariance costate, a ten-layer atom-law realization has the conditional cost

\[
G_{\rm replace}
\le28+10=38,
\tag{N.27}
\]

leaving eleven units under the 49-unit cap.  Hence the standard multi-layer
strict-depth route is economically plausible only as a D10 replacement, not
as a neighbor of the complete `D14+D10=48` schedule.  It must then beat the
**value lost by removing D10** in addition to satisfying (N.19):

\[
\Delta R_<^{\max}
-\Delta R_{D10\,\rm lost}
\ge
\Delta R_{\rm required,new}.
\tag{N.28}
\]

### Additional graph split for the portfolio decision

#### Node N4A -- additive dense-law form

- **Suggested id:** `P2R0G19B_MATH_K31_DELAYED_ADD_ONEUNIT`
- **Label:** `Delayed K31 dense-law add-on within one GEMM`
- **Kind:** `phase2_g19b_math_cost_gate`
- **Status:** `closed_cost` for the ordinary multi-layer `N_k=n` interface;
  exceptional precontracted/rank-one interfaces remain delegated to N3.
- **Summary:** Two or more independent full-width fresh-law contractions
  violate the single unit left by D14+D10.  Depth batching does not reduce
  billed arithmetic.
- **Terminal criterion:** Reopen only with one of the three exact exceptions
  following (N.26), including a full provenance and charged-cut proof.
- **Evidence:** This file, equations (N.25)--(N.26).
- **Goal-path summary:** Closes the standard additive spelling without closing
  a genuinely free fresh-action oracle.

#### Node N4B -- replace the D10 stream

- **Suggested id:** `P2R0G19B_MATH_K31_DELAYED_REPLACE_D10`
- **Label:** `Delayed K31 residual as D10 replacement`
- **Kind:** `phase2_g19b_math_portfolio_gate`
- **Status:** `open`
- **Summary:** Replacing D10 provides enough arithmetic for the conditional
  38-GEMM realization, but the delayed residual must repay D10's removed value
  and then clear the remaining raw target.
- **Terminal criterion:** Prove N2/N3, verify all costs in (N.27), and show the
  net value inequality (N.28); otherwise close the replacement.
- **Evidence:** This file, equations (N.23)--(N.28).
- **Goal-path summary:** This is the only standard-interface portfolio slot in
  which a multi-layer delayed K31 completion can fit.

Add N4A and N4B as decomposition children of N4.  N2 and N3 inform both;
N4B, if solved, informs the endpoint certificate and portfolio Gram.  N4A is
terminal for the scoped dense-law additive form.

## Checkpoint 9C: positive-law identifiability no-go for incumbent-only freshness

The delayed collision-free residual cannot be inferred universally from the
same-site D11/Q24 moment set.  This admits an exact positive-probability
witness, stronger than the earlier Gaussian-versus-Rademacher fourth-moment
example.

For `sigma in {+1,-1}`, let `mu_sigma` be the uniform law on the eight sign
vectors

\[
x\in\{-1,+1\}^4,
\qquad
x_1x_2x_3x_4=\sigma.
\tag{N.29}
\]

Equivalently, choose `x_1,x_2,x_3` independently Rademacher and set

\[
x_4=\sigma x_1x_2x_3.
\]

Every marginal on at most three distinct coordinates is uniform.  Therefore
`mu_+` and `mu_-` have exactly the same:

- means and covariance (`C=I`);
- all third moments;
- marginal fourth moments;
- repeated `3+1` moments/K31; and
- repeated `2+2` moments/K22.

Indeed powers of a sign variable reduce modulo two, so every repeated-index
moment involves at most three distinct coordinates and is identical under the
two laws.  But

\[
\mathbb E_{\mu_\sigma}[x_1x_2x_3x_4]=\sigma,
\]

and, because all off-diagonal covariances vanish,

\[
\kappa_{1234}(\mu_+)-\kappa_{1234}(\mu_-)=2.
\tag{N.30}
\]

Thus the two laws differ by a pure collision-free V4 direction while agreeing
on the full standard repeated-index order-four interface.

Now apply a dense linear reset `y=Kx`.  Multilinearity of cumulants gives

\[
\begin{aligned}
&\kappa_{aaab}(y;\mu_+)
-\kappa_{aaab}(y;\mu_-)\\
&\quad=
12\sum_{u=1}^4K_{bu}
\prod_{r\in\{1,2,3,4\}\setminus\{u\}}K_{ar},
\end{aligned}
\tag{N.31}
\]

where the factor `12` is `2` from (N.30) times the six permutations assigning
the other three source indices to the repeated output index `a`.  The right
side is a nonzero polynomial for generic dense `K`.  Hence the two laws have
different delayed K31 after one reset despite identical incumbent moment
data at the birth boundary.

### Exact information consequence

Let `F_birth` contain any feature family determined by moments supported on at
most three distinct indices, including total same-site K31/K22/K4.  Let
`phi_<` be a delayed Bellman action for which (N.31) has nonzero pairing.  Then

\[
F_{\rm birth}(\mu_+)=F_{\rm birth}(\mu_-),
\qquad
\phi_< (\mu_+)\ne\phi_< (\mu_-).
\]

Therefore

\[
\ker F_{\rm birth}\not\subseteq\ker\phi_<,
\]

and the Node-N3 fiber condition fails.  No universal analytic refit of the
incumbent repeated-moment state can be called a fresh delayed-K31 provider.

The weakest lawful information premise is now precise: the new interface must
separate every endpoint-visible class in the collision-free quotient.  If
`U_<` is that quotient and `Phi_<:U_< -> Y` its delayed target operator, any
target-blind exact interface `D_new` must satisfy

\[
\ker(D_{\rm new}|_{U_<})
\subseteq\ker\Phi_<,
\tag{N.32}
\]

and hence needs at least

\[
\operatorname{rank}(\Phi_<)
\]

independent scalar coordinates on the quotient.  For one frozen scalar target
this algebraic lower bound can be one, but that one coordinate must be the
actual target-specific primitive action; producing it is exactly the charged
cut problem in N4, not a consequence of old moments.

### Node-N3 status refinement

- `incumbent-only repeated-moment fresh formula`: `closed_proof` by the two
  positive laws (N.29)--(N.31).
- `genuine post-reset law observation`: remains `open`; it can distinguish the
  witness because the physical post-reset K31 differs.
- `target-specific precontracted scalar oracle`: remains logically sufficient
  but must be justified as a lawful deployed input and billed under N4.

This witness also proves that the strict-lower route is genuinely new
information rather than an algebraic rewrite of same-site K31.  It does not
prove that the information is available or that its residual value clears
(N.19).

## Checkpoint 9D: eleven independent positive-law parity directions

The preceding witness tensorizes and gives a sharp information-rank result
matching the generic eleven-direction primitive quotient.

Let `r<=floor(n/4)` and choose disjoint coordinate quartets

\[
Q_s=\{4s-3,4s-2,4s-1,4s\},
\qquad s=1,\ldots,r.
\]

For each sign vector `sigma in {+1,-1}^r`, define a positive law by taking the
blocks independently and imposing

\[
\prod_{i\in Q_s}x_i=\sigma_s
\]

inside block `s`.  All `2^r` laws agree on every moment involving at most
three distinct coordinates and hence on all incumbent mean/C/K3/K4/K31/K22
features.  Their collision-free fourth cumulants differ independently along

\[
T_s
=\operatorname{sym}
(e_{4s-3}\otimes e_{4s-2}\otimes e_{4s-1}\otimes e_{4s}).
\tag{N.33}
\]

The hidden incumbent fiber therefore contains an `r`-dimensional affine cube,
not just one ambiguous direction.

Choose an invertible block-diagonal reset with one generic `4 by 4` block on
each `Q_s`.  In each block choose an output pair `(a_s,b_s)` for which

\[
q_s(K)
=\sum_{u\in Q_s}K_{b_su}
\prod_{v\in Q_s\setminus\{u\}}K_{a_sv}
\ne0.
\tag{N.34}
\]

Such a block exists by the explicit polynomial (N.31).  The delayed K31
coordinate `(a_s,a_s,a_s,b_s)` then responds to `T_s` and to no `T_t` with
`t!=s`.  Hence the `r by r` minor of the delayed K31 observation matrix is
diagonal with nonzero diagonal.  Its determinant is a nonzero polynomial in
the reset entries, so the same rank `r` holds for generic dense resets.

At width `n=1024`, set `r=11`.  Then the generic eleven-dimensional primitive
quotient has eleven independently identifiable delayed K31 directions while
remaining completely invisible to the incumbent repeated-moment interface.

### Target-blind versus target-adjoint consequence

For a target-blind provider required to preserve the delayed action for an
open set of downstream Bellman rows, the interface rank is at least eleven:

\[
\operatorname{rank}D_{\rm new}|_{U_<}\ge11.
\tag{N.35}
\]

Thus a universal rank-one fresh observable cannot replace the generic
primitive provider.  This recovers the fused-pair rank-11 obstruction using
only positive laws and causal delayed K31.

For one frozen scalar target, the composed row

\[
G_<|_{U_<}:U_<\to\mathbb R
\]

has algebraic rank at most one.  It may therefore cross the interface as one
coordinate **only if** the Bellman row is lawfully present on the law side
before the charged cut.  If the law side must export target-independent
features first, eleven coordinates are necessary.  This gives the exact
intertwining/fusion condition behind the apparent scalar loophole:

\[
\text{rank-one charged route}
\quad\Longleftrightarrow\quad
\text{the frozen Bellman contraction factors before }D_{\rm new}.
\tag{N.36}
\]

If the target row is supplied only after `D_new`, post-contraction cannot
retroactively reduce (N.35).

### Status refinement

- `target-blind one-coordinate delayed interface`: `closed_proof` generically
  on the eleven-dimensional primitive quotient.
- `target-specific pre-cut Bellman contraction`: `open`, governed by N3's
  lawful-information provenance and N4's per-layer arithmetic.
- `full delayed K31 plane as fresh observation`: information-sufficient
  generically, but its acquisition cost remains unresolved and it receives no
  same-site value credit.

This separates two ranks that must not be conflated: delayed K31 can have
measurement rank eleven while the final target action is scalar.  Which rank
is billed depends entirely on which side of the external bridge owns the
Bellman covector.

## Checkpoint 9E: exact one-product Z22 exception to the additive cost closure

The scoped N4A closure concerns generic multi-layer fresh-law acquisition.  A
separate lane derived one exact one-transition exception from a discarded
post-reset `(22)` residual plane `Z`.  With reset matrix `W` and
`W2=W odot W`, its delayed K31 plane is

\[
\delta K31
=3\{W\odot(ZW2)\}^TW.
\tag{N.37}
\]

For future K31 costate `Lambda31`, Frobenius cyclicity gives

\[
\begin{aligned}
\langle\Lambda31,\delta K31\rangle
&=3\left\langle
W\odot(ZW2),,W\Lambda31^T
\right\rangle_F.
\end{aligned}
\tag{N.38}
\]

Thus if `W Lambda31^T` is already exposed by the reverse consumer, the sole
new dense product is

\[
ZW2,
\]

and the witness fits the one GEMM left by D14+D10.  If that reverse product is
not reusable, the witness costs two and cannot be an additive neighbor of the
48-GEMM schedule.

Equation (N.38) is precisely the third exception following (N.26): one
transition and one structured residual plane, with the Bellman contraction on
the law/source side of the charged cut.  It does not reopen the generic
multi-layer atom-law form.

Three further gates remain binding before this scalar receives raw-value
credit:

1. residualize its endpoint action against the complete CP4/pair/star/depth
   arena, not merely the birth K22 plane;
2. count no same-site K31 action, in accordance with N1; and
3. either include the K4 and crossed-K22 consequences of the same physical
   pure-22 lift, or prove those omitted Bellman actions are dark/low rank.

The crossed K22 branch is generically not a one-product full provider.  Hence
the one-GEMM result is a strict-depth **value witness**, not yet a complete
fourth-order recurrence.  Its source formulas and complete duplicate audit
are recorded in `LANE_traffic_probability_closure.md`, equations
`(T9.8)--(T9.14)`.

Graphically, `Z22_K31_SCALAR_WITNESS_COST` should inform N4A as an exceptional
conditional child; `Z22_COMPLETE_ARENA_SCHUR_GATE` and
`Z22_FULL_PROVIDER_K22_COST` remain prerequisites for promotion beyond a
scalar witness.

## Checkpoint 9F: which complete Z22 actions factor through the one product

Let

\[
U=W\odot W,
\qquad
R=ZU.
\]

The pure-22 lift generates K4, K31, and K22 at the first dense reset.  Pair
each with its frozen future Bellman weight: a vector `d` for K4, a directed
matrix `Lambda31` for K31, and a symmetric matrix `Lambda22` for K22.

### K31, K4, and uncrossed K22

The K31 action is already

\[
\chi_{31}
=\left\langle
R,
3W\odot(W\Lambda31^T)
\right\rangle.
\tag{N.39}
\]

The K4 action obeys

\[
\chi_4
=3\sum_a d_a u_a^TZu_a
=\left\langle
R,
3U\operatorname{diag}(d)
\right\rangle,
\tag{N.40}
\]

where `u_a` is column `a` of `U`.  It therefore needs only a column scaling
after `R` is available.

The first, uncrossed K22 term is `U^T ZU`; hence

\[
\chi_{22,\rm uncrossed}
=\langle\Lambda22,U^TZU\rangle
=\langle R,U\Lambda22\rangle.
\tag{N.41}
\]

It is free after `R` only if `U Lambda22` is already exposed by the reverse
consumer; otherwise that adjoint product must receive one additional GEMM.

Thus K31, K4, and uncrossed K22 all share the one source-side statistic `R`.
Their combined readout is

\[
H_{\rm easy}
=3W\odot(W\Lambda31^T)
+3U\operatorname{diag}(d)
+U\Lambda22.
\tag{N.42}
\]

### Crossed K22 action

Let `v_ab=w_a odot w_b`.  The remaining action is

\[
\chi_{22,\rm cross}
=2\sum_{a,b}(\Lambda22)_{ab}\,v_{ab}^TZv_{ab}
=\langle Z,L_{\rm cross}\rangle,
\tag{N.43}
\]

where

\[
(L_{\rm cross})_{ij}
=2\,r_{ij}^T\Lambda22 r_{ij},
\qquad
r_{ij}=W_{i,:}\odot W_{j,:}.
\tag{N.44}
\]

For generic dense `Lambda22`, forming all entries in (N.44) is the high-cost
crossed Khatri--Rao quadratic-form problem.  If

\[
\Lambda22=\sum_{t=1}^r\lambda_tq_tq_t^T,
\]

then

\[
L_{\rm cross}
=2\sum_{t=1}^r\lambda_t
\left(W\operatorname{diag}(q_t)W^T\right)^{\odot2}.
\tag{N.45}
\]

This gives an `r`-GEMM construction for a symmetric rank-`r` costate, followed
by cheap Hadamard squares.  It does not fit the last one-unit slot together
with `R=ZU` unless the required conjugations are already shared or the crossed
costate is Bellman-dark.

### Information factorization versus arithmetic factorization

If `U` is invertible, then `R=ZU` is information-theoretically sufficient for
**every** scalar linear action of `Z`.  In particular,

\[
\chi_{22,\rm cross}
=\langle R,H_{\rm cross}\rangle,
\qquad
H_{\rm cross}=L_{\rm cross}U^{-T}.
\tag{N.46}
\]

Generic invertibility of `U=W odot W` follows once one nonzero determinant
example is exhibited; a diagonal invertible `W` gives diagonal invertible
`U`, so singularity is a proper algebraic condition.  Therefore the crossed
branch is not a missing-information obstruction after `R`.

But (N.46) does not make it cheap.  Constructing `L_cross` is generically the
cost in (N.44), and applying `U^{-T}` is a dense multiple-right-hand-side
solve.  The inverse formulation moves the arithmetic to the target-adjoint
side; it does not erase it.  Its conditioning gate is

\[
\|H_{\rm cross}\|
\le\frac{\|L_{\rm cross}\|}{\sigma_{\min}(U)},
\tag{N.47}
\]

so an ill-conditioned Hadamard-square reset also amplifies acquisition error
in the endpoint certificate.

### Exact completion criterion for the one-product witness

The Z22 first-reset scalar is complete within one new source product if and
only if all target-adjoint weights in

\[
H_{\rm total}
=H_{\rm easy}+H_{\rm cross}
\]

are already available from the shared reverse module at no additional dense
cost and satisfy the conditioning/error gate.  A sufficient cheap case is

\[
\Lambda22_{\rm cross}=0,
\]

with both `W Lambda31^T` and `U Lambda22` reused.  A rank-one nonzero crossed
costate still normally needs one conjugation in (N.45), so it is a two-product
candidate unless that conjugation is shared.

This sharpens the earlier statement: the omitted crossed K22 branch need not
be carried as a plane, but its **target-adjoint construction cost** must be
proved cheap.  The one-product K31 witness is not automatically the complete
pure-22 value action.

## Checkpoint 9G: conditioning closes the naive Hadamard-square inverse loophole

Generic invertibility in (N.46) is only an information theorem.  It gives a
sharp stability requirement that is unfavorable for dense mixing.

If the acquired statistic is perturbed to

\[
\widetilde R=ZU+E_R,
\]

then an inverse-defined action weight `H=L U^{-T}` incurs

\[
|\langle E_R,H\rangle|
\le
\|E_R\|_F\,\|L\|_F\,\|U^{-1}\|_2
=
\frac{\|E_R\|_F\,\|L\|_F}{\sigma_{\min}(U)}.
\tag{N.48}
\]

To allocate endpoint-error budget `eta_branch` to this action one needs

\[
\|E_R\|_F
\le
\eta_{\rm branch}
\frac{\sigma_{\min}(U)}{\|L\|_F}.
\tag{N.49}
\]

This includes approximation, roundoff, and any true-minus-assembled moment
error in `R`.

For a clean dense-reset model, take `W` orthogonal.  Then

\[
U=W\odot W
\]

is doubly stochastic (orthostochastic), so `U 1=1` and `U^T 1=1`.  Its top
singular value is one.  Moreover

\[
\|U\|_F^2=\sum_{i,j}W_{ij}^4.
\]

Therefore every realized dense orthogonal reset obeys

\[
\sigma_{\min}(U)^2
\le
\frac{\|U\|_F^2-1}{n-1}.
\tag{N.50}
\]

For Haar orthogonal `W`,

\[
\mathbb E\sum_{i,j}W_{ij}^4
=\frac{3n}{n+2},
\]

and hence

\[
\mathbb E\,\sigma_{\min}(U)^2
\le\frac{2}{n+2}.
\tag{N.51}
\]

At `n=1024`, the square-root scale in this rigorous average bound is about
`0.0442`, corresponding to at least about `22.6` amplification in a weakest
direction whenever the realized fourth-energy is near its Haar value.  This
is only an upper bound on the smallest singular value; the actual smallest
value can be much lower.  For entrywise-Gaussian dense resets, writing `U` as
a rank-one mean plus a centered square matrix likewise predicts a much
smaller hard-edge singular scale, but that heuristic is not needed for the
certificate.

Consequently a proposal using `H_cross=L_cross U^{-T}` must supply the actual
`sigma_min(U)` bound and insert (N.48) into Node E.  Algebraic genericity cannot
license a finite-precision implementation.  The preferred one-product cases
are precisely the direct structured weights (N.39)--(N.42), which avoid
`U^{-T}` altogether; the crossed K22 inverse is a fallback only with a proved
conditioning and construction-cost certificate.

## Checkpoint 9H: correlation allocation after strict-depth residualization

For an optimally rescaled scalar channel with task correlation `q`, the
retained value energy is `alpha=q^2`.  Insert this directly into the certified
ladder

\[
R_{\rm raw}
=14.656
-4.370\alpha_3
-1.0313\alpha_4
\qquad\text{in }10^{-9}.
\tag{N.52}
\]

This ladder is conditional on the variance correction being perfect.  Any
variance-channel residual must be added before using the numerical rows below,
so the table is an optimistic allocation bound rather than a complete
deployed prediction.

For the two available correlation levels,

\[
q_{.92545}^2=0.8564577,
\qquad
q_{.98346}^2=0.9671936.
\tag{N.53}
\]

The four idealized allocations are therefore approximately:

| H3 correlation | quartic correlation | implied raw MSE |
|---:|---:|---:|
| `.92545` | `.92545` | `1.00300e-8` |
| `.92545` | `.98346` | `9.9158e-9` |
| `.98346` | `.92545` | `9.5461e-9` |
| `.98346` | `.98346` | `9.4319e-9` |

Thus two `.92545` channels miss the target, narrowly but genuinely.  If only
one channel can receive the `.98346` fidelity, assigning it to H3 has much
larger raw value because the certified H3 opportunity is `4.370e-9` versus
`1.0313e-9` for quartic.

If H3 remains at `.92545`, the minimum total quartic retained fraction is

\[
\alpha_{4,\min}
=\frac{4.656-4.370(0.8564577)}{1.0313}
\simeq0.8856,
\tag{N.54}
\]

corresponding to a quartic task correlation of about

\[
q_{4,\min}\simeq0.9411.
\]

Now let `f_<` be the fraction of the **incumbent-residual quartic oracle
value** that actually belongs to the strict-depth target, and let `q_<` be the
correlation of its estimator.  Its contribution to (N.52) is

\[
\alpha_4=f_<q_<^2.
\tag{N.55}
\]

At `q_<=.98346`, (N.54)--(N.55) require

\[
f_<\gtrsim0.9156.
\tag{N.56}
\]

At `q_<=.92545`, even `f_<=1` is insufficient when H3 is also `.92545`.
Therefore the duplicate-corrected delayed route is viable in that allocation
only if it contains more than about `91.6%` of the verified residual quartic
oracle value and attains the higher correlation.

If H3 instead attains `.98346`, the required total quartic fraction falls to
about `0.4163`; a `.92545` quartic estimator then needs

\[
f_<\gtrsim0.486.
\]

These are capacity gates, not measured claims.  In particular, correlations
computed on total same-site K31 cannot be inserted into (N.55); both `f_<` and
`q_<` must be evaluated after D11/Q24 and arena residualization.

## Checkpoint 9I: generic crossed-K22 diagonal-completion rank is at least 980

The K22 costate is physically defined on distinct pairs.  Its diagonal may be
rerouted into the separately owned K4 action.  Therefore the relevant
separation rank for (N.45) is the minimum symmetric diagonal-completion rank

\[
r_{\rm dc}(\Lambda22)
=\min_{d\in\mathbb R^n}
\operatorname{rank}
(\Lambda22+\operatorname{diag}d).
\tag{N.57}
\]

Each rank-one term in such a completion produces one conjugation
`W diag(q_t) W^T` in the crossed pullback.  Thus `r_dc` is the canonical
conjugation count after all lawful K4 diagonal rerouting.

For a generic off-diagonal symmetric matrix this rank is almost full.  The
variety of symmetric `n by n` matrices with nullity at least `m` has codimension

\[
\frac{m(m+1)}2
\]

inside `Sym_n`.  The affine family of diagonal completions has dimension `n`.
A generic such family cannot intersect the nullity-`m` variety when

\[
\frac{m(m+1)}2>n.
\tag{N.58}
\]

Consequently, for generic off-diagonal `Lambda22`,

\[
r_{\rm dc}(\Lambda22)
\ge
n-
\left\lfloor
\frac{\sqrt{8n+1}-1}{2}
\right\rfloor.
\tag{N.59}
\]

At `n=1024`, the largest allowed generic nullity is `44`, because

\[
44\cdot45/2=990\le1024,
\qquad
45\cdot46/2=1035>1024.
\]

Hence

\[
\boxed{r_{\rm dc}(\Lambda22)\ge980}
\tag{N.60}
\]

for a generic width-1024 off-diagonal costate.

This closes two generic economic forms:

1. the one-GEMM additive Z22 completion cannot include crossed K22; after
   spending its unit on `R=ZU`, it has no room even for rank one; and
2. the D10-replacement construction leaves about eleven units after the
   38-GEMM K31 realization, still nowhere near 980 conjugations plus the dense
   solve in (N.46).

Therefore a complete Z22 target-adjoint route survives only under a
candidate-specific theorem that the incumbent-residual crossed-K22 costate is
Bellman-dark, already available from shared conjugations, or has exceptional
diagonal-completion rank within the actual remaining budget.  Low rank cannot
be inferred from scalar endpoint dimension or from K4 diagonal freedom.

The exact pure-22 pullback and diagonal-completion audit are independently
recorded in `LANE_traffic_probability_closure.md` Checkpoint 10.  Equations
(N.57)--(N.60) are the resulting cost theorem needed by Nodes N4A/N4B.

## Checkpoint 9J: sharp Bellman-dark gate for omitted crossed K22

Let `ell_cross` be the scalar crossed-K22 endpoint row on the actual Z22
source space, after the exact pullback (N.43)--(N.45).  Let `F_plus` stack the
complete incumbent arena features **and** the proposed cheap K31/K4/uncrossed
witness.  Define

\[
\Sigma_{\perp F_+}
=\Sigma_Z
-\Sigma_ZF_+^*
(F_+\Sigma_ZF_+^*)^\dagger
F_+\Sigma_Z.
\tag{N.61}
\]

The crossed branch is exactly Bellman-dark, including all duplicate overlap,
if and only if

\[
\ell_{\rm cross}^*
\Sigma_{\perp F_+}
\ell_{\rm cross}=0.
\tag{N.62}
\]

Equivalently, its pullback matrix lies in the old-plus-cheap row span on the
reachable Z22 quotient.  Merely having small Frobenius norm before Schur
residualization is neither necessary nor sufficient.

If (N.62) is not zero, the irreducible omitted endpoint RMS is

\[
\eta_{\rm cross}
=\sqrt{
\ell_{\rm cross}^*
\Sigma_{\perp F_+}
\ell_{\rm cross}}
\tag{N.63}
\]

under the normalized endpoint ensemble.  It must be inserted into Node E
together with all other errors.

The correlation allocation in Checkpoint 9H shows how tight this can be.  In
the `q_3=.92545`, `q_4=.98346` idealized allocation, the modeled raw MSE is
about

\[
R_{\rm modeled}=9.9158\times10^{-9}.
\]

If only an RMS error bound is known, the exact worst-case allowance before
crossing raw `1e-8` is

\[
\eta_{\rm cross}+\eta_{\rm other}
<10^{-4}-\sqrt{9.9158\times10^{-9}}
\simeq4.22\times10^{-7}.
\tag{N.64}
\]

For the more favorable `q_3=.98346`, `q_4=.92545` allocation with modeled raw
about `9.5461e-9`, the corresponding allowance is about

\[
2.30\times10^{-6}.
\tag{N.65}
\]

Thus the narrow allocation requires an exceptionally strong darkness or
overlap certificate.  A qualitative assertion that crossed K22 is secondary
cannot license the one-product witness.

### Graph node

- **Suggested id:** `P2R0G19B_MATH_Z22_CROSSED_K22_DARK`
- **Label:** `Crossed K22 is dark after arena plus cheap witness`
- **Kind:** `phase2_g19b_math_value_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** Generic crossed K22 is cost-closed at rank at least 980, so the
  one-product Z22 witness survives only if its incumbent-and-witness
  conditional endpoint norm is zero or below the remaining raw error margin.
- **Terminal criterion:** Prove (N.62), or bound (N.63) jointly with all other
  errors below the allocation-specific (N.64)/(N.65); otherwise close the
  complete one-product Z22 route.
- **Evidence:** This file, equations (N.43)--(N.65), plus
  `LANE_traffic_probability_closure.md` Checkpoint 10.
- **Goal-path summary:** This is a hard prerequisite for promoting the cheap
  strict-depth K31 scalar from a witness to a complete physical value action.

Add an `informs` edge from this node to N4A and a hard `requires` edge from it
to any complete one-product Z22 candidate.

## Checkpoint 9K: diagonal duplicate removal does not reduce delayed Hankel rank

Let the strict-depth input/output kernel be

\[
K_{kj}=\Delta H_{kj},
\qquad j<k,
\]

with all `j=k` feedthrough blocks deleted by N1.  At a depth cut `s`, define
the past-to-future block Hankel operator

\[
\mathfrak H_s
=\big(K_{kj}\big)_{k>s,\,j\le s}.
\tag{N.66}
\]

The minimal exact target-blind causal state dimension is at least

\[
r_H=\max_s\operatorname{rank}\mathfrak H_s,
\tag{N.67}
\]

with equality when the usual shift/intertwining consistency conditions hold.
This is the finite-horizon Hankel realization theorem already used earlier in
the lane.

Same-site diagonal blocks never cross a past/future cut.  Therefore deleting
or residualizing them changes the direct feedthrough operator but does not
change any `H_s` in (N.66).  In particular, D11/Q24 duplicate removal cannot
by itself reduce delayed state rank or turn an eleven-direction carrier into
one direction.

The tensorized parity construction of Checkpoint 9D gives a concrete generic
lower bound.  Place eleven independent collision-free parity births before a
cut and their eleven K31 observation coordinates after one dense reset.  The
corresponding `11 by 11` minor is diagonal and nonzero for the explicit block
reset, hence nonzero for generic dense resets.  Thus

\[
\boxed{r_H\ge11}
\tag{N.68}
\]

on the generic eleven-dimensional primitive quotient, using strict-lower
blocks alone.

Consequences:

1. a target-blind one-word D10 carrier cannot exactly realize the delayed
   kernel;
2. rolling or restarting one word cannot beat (N.68) without recomputing past
   births, which is the hidden `O(L^2)` schedule in (N.16); and
3. subtracting the incumbent diagonal prevents double credit but provides no
   state-capacity discount.

A frozen scalar endpoint can still collapse the **output** rank by contracting
the future rows with Bellman weights.  It evades (N.68) only if that target
covector is available on the birth/law side of the charged cut, producing the
source action (N.3)--(N.4).  If the provider must remain target blind until
after the cut, the rank-11 lower bound remains binding.

### Graph node

- **Suggested id:** `P2R0G19B_MATH_K31_STRICT_HANKEL_RANK`
- **Label:** `Strict-depth primitive K31 has Hankel rank at least eleven`
- **Kind:** `phase2_g19b_math_theorem`
- **Status:** `solved_generic`
- **Priority:** `1`
- **Summary:** Removing same-site D11/Q24 feedthrough leaves every causal
  past-to-future Hankel block unchanged.  Eleven positive-law parity births
  yield a generic rank-11 strict-lower minor.
- **Terminal criterion:** Retain for generic target-blind primitive transport;
  a candidate may evade it only by proving a smaller restricted reachable
  quotient or lawful pre-cut target-adjoint contraction.
- **Evidence:** This file, equations (N.29)--(N.36) and (N.66)--(N.68).
- **Goal-path summary:** Closes rank-one target-blind/rolling carriers after
  duplicate correction while preserving the target-specific scalar exception.

This node `informs` N2, N3, N4A, and N4B.

## Checkpoint 9L: low-rank Bellman action is a fourth one-unit exception

The dense atom-law cost in (N.25) assumes a generic dense action matrix.  A
target-adjoint action may admit a much cheaper separated evaluation even when
the underlying delayed H4 state has rank eleven.

From the covariance consumer (D.6),

\[
A_l=2\Lambda_l\odot W_l.
\]

Suppose exact or controlled factorizations are available,

\[
\Lambda_l=X_lY_l^T,
\qquad
W_l=U_lV_l^T,
\]

with ranks `r_Lambda,l` and `r_W,l`.  The Schur-product factorization gives

\[
A_l=P_lQ_l^T,
\qquad
R_l:=\operatorname{rank}A_l
\le r_{\Lambda,l}r_{W,l},
\tag{N.69}
\]

where the columns of `P_l` and `Q_l` are all pairwise Hadamard products of
columns of `(X_l,U_l)` and `(Y_l,V_l)`.

For an `N_l`-atom law, the scalar action is

\[
\phi_l
=\sum_sp_s
\big(P_l^Th_3(z_s)\big)^T
\big(Q_l^Tz_s\big).
\tag{N.70}
\]

It needs two thin projections and a lower-order contraction, not one dense
`N_l by n` times `n by n` product.  Under a conservative square-GEMM
normalization its cost is

\[
G_l^{\rm thin}
\le\frac{2N_lR_l}{n^2}.
\tag{N.71}
\]

If one of the two projections is already shared by the law or reverse module,
the factor two can be removed.

The exact additive one-unit gate becomes

\[
G_{\rm factors,new}
+\sum_{l\in I_<}\frac{2N_lR_l}{n^2}
\le1.
\tag{N.72}
\]

For ten layers with `N_l=n=1024`, no shared projection, and roughly constant
rank, (N.72) permits

\[
R_l\le\frac{1024}{20}\simeq51
\tag{N.73}
\]

on average.  With one projection shared, the threshold is about `102`.

This is a real exception to the generic dense-law N4A closure, but it requires
four independent facts:

1. the action is the incumbent-residual strict-depth row, not total same-site
   K31;
2. the factors are available before the charged law cut;
3. their construction/storage does not consume the saved arithmetic; and
4. truncation preserves enough **residual task value**, not merely Frobenius
   energy.

For approximate rank `R`, a deterministic endpoint-error bound is

\[
|\delta\phi_l|
\le
\|A_l-A_{l,R}\|_F\,\|K31_l^{\rm residual}\|_F,
\tag{N.74}
\]

and the sum of these errors enters Node E.  The optimal rank must be chosen in
the atom/action-induced value Gram when available; ordinary SVD energy is only
a sufficient norm bound.  In particular, a short Mehler expansion does not
automatically imply low matrix rank: Hadamard powers of a full-rank
correlation matrix can themselves have high rank.

### Graph refinement

Add a conditional child of N4A:

- **Suggested id:** `P2R0G19B_MATH_K31_DELAYED_THIN_ACTION`
- **Label:** `Strict-depth K31 thin Bellman action within one GEMM`
- **Status:** `open`
- **Terminal criterion:** Prove (N.72) from actual pre-cut factors and insert
  the value-weighted tail (N.74) into the raw certificate.  Require the
  residual correlations/fractions in Checkpoint 9H.

This child is distinct from a rank-one q4 carrier.  It compresses the frozen
consumer action after lawful fresh acquisition, not the target-blind delayed
state.

## Checkpoint 9M: Mehler/separable rank theorem for the thin-action exception

### Rank propagation of a covariance costate

For a covariance tangent with elementwise response kernel `J_l`, the reverse
matrix has the generic form

\[
\Lambda_l
=K_l^T(\Lambda_{l+1}\odot J_l)K_l.
\tag{N.75}
\]

If `Lambda_{l+1}` has separation rank `r_l` and `J_l` has rank `s_l`, Schur
factorization gives

\[
\operatorname{rank}\Lambda_l
\le r_ls_l.
\tag{N.76}
\]

Dense conjugation does not increase ordinary rank when `K_l` is invertible.
Therefore a rank-one response kernel preserves costate rank, while exact
multi-rank kernels can multiply it across depth.  An exact thin-action proof
must bound the resulting ranks jointly over the full suffix, not at one layer.

The relevant rank is after quotienting cheap/owned diagonal readouts.  Define

\[
r_{\rm sep/diag}(A)
=\min_d\operatorname{rank}(A+\operatorname{diag}d).
\tag{N.77}
\]

For example, an off-diagonal constant kernel is an outer product minus a
diagonal.  Its ordinary matrix rank is full, but its action separation rank
modulo the K4 diagonal is one.  Equations (N.69)--(N.73) should use this
quotient rank whenever the diagonal correction is explicitly rerouted and
billed cheaply.

### Exact Mehler expansion

For standardized jointly Gaussian coordinates with correlation `rho_ij`, the
Mehler/Price identity gives, with appropriate weak-derivative interpretation,

\[
W_{ij}
=\frac16\sum_{m=0}^{\infty}
\frac{\rho_{ij}^m}{m!}
a_{m,i}b_{m,j},
\tag{N.78}
\]

where

\[
a_{m,i}=\mathbb E[\phi^{(m+3)}(G_i)],
\qquad
b_{m,j}=\mathbb E[\phi^{(m+1)}(G_j)]
\]

after absorbing marginal-variance scalings into `a,b`.

The `m=0` term is separable:

\[
W^{(0)}=\frac16a_0b_0^T.
\]

Thus

\[
\Lambda\odot W^{(0)}
=\frac16\operatorname{diag}(a_0)
\Lambda
\operatorname{diag}(b_0),
\tag{N.79}
\]

and preserves the rank of `Lambda` on nonzero support.  This is the exact
algebra behind a correlation-independent thin Bellman action.

For `m>=1`, the term contains the Hadamard power `rho^{odot m}`.  If `rho` has
Gram rank `d`, then

\[
\operatorname{rank}(\rho^{\odot m})
\le {d+m-1\choose m},
\tag{N.80}
\]

and hence

\[
\operatorname{rank}
(\Lambda\odot W^{(m)})
\le
\operatorname{rank}(\Lambda)
{d+m-1\choose m}.
\tag{N.81}
\]

For full-rank dense correlations this bound is capped only by `n`; a short
Mehler series does not mean a short matrix factorization.

### Tail certificate

If `|rho_ij|<=rho_max<1`, truncating (N.78) after degree `M` gives the entrywise
bound

\[
|E^{(M)}_{ij}|
\le\frac16\sum_{m>M}
\frac{|a_{m,i}b_{m,j}|}{m!}
\rho_{\max}^m.
\tag{N.82}
\]

Let the right side be bounded uniformly by `delta_M`.  The omitted local K31
action then obeys

\[
|\delta\phi_l|
\le
\|\Lambda_l\odot E_l^{(M)}\|_F
\|K31_l^{\rm residual}\|_F
\le
\delta_{M,l}\|\Lambda_l\|_F
\|K31_l^{\rm residual}\|_F.
\tag{N.83}
\]

The order-`1/n` accumulated prediction error is at most

\[
\eta_{\rm Mehler}
\le\frac1n\sum_l
\delta_{M,l}\|\Lambda_l\|_F
\|K31_l^{\rm residual}\|_F,
\tag{N.84}
\]

plus any error from truncating the reverse costates themselves.  It must meet
the allocation-specific endpoint gate, potentially the `4.22e-7` narrow gate
in (N.64).

### Thin-action decision theorem

A Mehler-based multi-layer action fits the one-unit exception only if all of
the following are proved:

1. after diagonal/old-span quotienting, the retained terms satisfy the
   charged rank sum (N.72);
2. the factor recurrences (N.75)--(N.81) are available before the law cut at
   their stated cost;
3. the true-minus-assembled strict-depth moments are lawfully observable; and
4. (N.84), together with factor-roundoff error, preserves the residual task
   correlations required in Checkpoint 9H.

The leading separable term is a mathematically coherent candidate.  Exact
full-rank Mehler kernels are not.

## Checkpoint 9N: crossed-K22 darkness is algebraically exceptional

The darkness condition (N.62) is not enforced by the pure-22 partition.  It is
a proper algebraic constraint on the reset, costate, and residual source
subspace.

A two-coordinate witness suffices.  Take

\[
W=\frac1{\sqrt2}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix},
\qquad
Z=\begin{pmatrix}0&z\\z&0\end{pmatrix},
\qquad z\ne0.
\]

For output pair `(a,b)=(1,2)`,

\[
v_{12}=w_1\odot w_2
=\frac12(1,-1)^T,
\]

and hence

\[
v_{12}^TZv_{12}=-\frac z2\ne0.
\tag{N.85}
\]

Choosing a K22 Bellman row with a nonzero `(1,2)` coefficient gives a nonzero
crossed action.  Therefore the polynomial map

\[
(W,\Lambda22,Z)\longmapsto
\ell_{\rm cross}(Z)
\]

is not identically zero.  On any reachable residual quotient containing this
type of pure-22 direction, its exact zero set is a proper algebraic subset.
Generic dense resets and generic nonzero K22 costates are not Bellman-dark.

Complete arena Schur projection can still make the **residualized** action
zero, but only if the old-plus-cheap response span contains this polynomial
row.  That is the explicit row-space identity (N.62), not a consequence of
partition orthogonality at birth.

Status consequence: `Z22_CROSSED_K22_DARK` remains open only for a proved
candidate-specific structural/Schur identity.  Its generic branch is
`closed_proof` as a darkness claim, and the generic complete one-product Z22
route remains cost-closed by (N.60).

## Checkpoint 9O: H3-resummed Bellman baseline removes the n^-3/2 mixed remainder

The order-`1/n` scalar theorem was initially stated around an uncorrected
baseline, where an order-`n^{-1/2}` H3 perturbation creates omitted mixed
H3--q4 terms at order `n^{-3/2}`.  At `L/sqrt(n)=0.5`, this is not
automatically negligible.

Let

\[
\epsilon=n^{-1/2}.
\]

If per-layer normalized H3 and H4 insertion norms are of order one, a naive
depth-16 time-order count gives

\[
{16\choose3}\epsilon^3
=\frac{560}{32768}
\simeq0.0171
\]

for three H3 insertions and approximately

\[
16\cdot15\,\epsilon^3
\simeq0.00732
\]

for ordered H3--H4 pairs.  These coefficients are far too large to discard
merely on asymptotic order when the endpoint error margin can be `4.22e-7`.

### Correct co-adapted expansion

Instead, absorb the complete deployed H3 correction into the baseline map:

\[
x_{l+1}^{(3)}=F_l^{(3)}(x_l^{(3)}).
\]

Let

\[
T_l^{(3)}=D F_l^{(3)}(x_l^{(3)}),
\qquad
\lambda_l^{(3)}=(T_l^{(3)})^*\lambda_{l+1}^{(3)}.
\]

Now introduce the residual quartic correction with its true scale

\[
\delta=1/n:
\qquad
x_{l+1}=F_l^{(3)}(x_l)+\delta B_lq_l.
\]

The exact first variation is

\[
\delta y
=\delta\sum_l
\left\langle
B_l^*\lambda_{l+1}^{(3)},q_l
\right\rangle
+R_{q4^2}.
\tag{N.86}
\]

Every time-ordered term containing any number of baseline H3 interactions and
exactly one q4 insertion is already included in `T^(3)` and `lambda^(3)`.
There is no separate omitted `epsilon^3` H3--q4 term.  The first perturbative
remainder contains two q4 insertions:

\[
|R_{q4^2}|
\le
\frac12\delta^2H_{V,3}
\left(\sum_l\|B_lq_l\|\right)^2,
\tag{N.87}
\]

where `H_V,3` is the endpoint Hessian along the H3-corrected trajectory.

For ten active layers, the dimensionless combinatorial scale is

\[
\frac12(10/n)^2
=4.77\times10^{-5}
\qquad(n=1024),
\tag{N.88}
\]

versus an `O(10^2/n^{3/2})` mixed scale around `0.00305` before
co-adaptation.  The actual endpoint bound still needs `H_V,3` and source
norms; (N.88) is not by itself an error certificate.

### Binding implementation consequence

Every strict-depth K31/Z22/thin-action candidate must derive its Bellman rows
from the fully H3-corrected D14 trajectory and reverse costate.  A row frozen
on the Gaussian, covariance-only, or separately trained H3-free baseline
forfeits (N.86) and must explicitly pay/bound the `n^{-3/2}` mixed terms.

This is the mathematical form of the co-adaptation rule: joint refit is not
only an empirical preference; it changes the controlled remainder from mixed
H3--q4 order to q4-squared order.

### Graph node

- **Suggested id:** `P2R0G19B_MATH_K31_H3_RESUMMED_BASELINE`
- **Label:** `Quartic Bellman row uses the H3-resummed trajectory`
- **Kind:** `phase2_g19b_math_remainder_gate`
- **Status:** `solved_theorem`, candidate compliance `open`
- **Terminal criterion:** Prove the deployed row is exactly
  `B_l^*lambda_{l+1}^{(3)}` for the jointly corrected trajectory and certify
  (N.87) inside Node E; otherwise restore the full mixed-order error.
- **Evidence:** This file, equations (N.86)--(N.88).
- **Goal-path summary:** Makes the order-`1/n` K31 action quantitatively
  credible at finite depth and forbids separately frozen quartic injection.

This node is a hard prerequisite for N2--N4 and the endpoint certificate.

## Checkpoint 9P: residual H3 uncertainty restores a mixed-error term

Checkpoint 9O resums every interaction with the **implemented** H3 trajectory.
It does not make an unobserved error in the H3 carrier disappear.

Write the true H3 source as

\[
q_3=\widehat q_3+e_3,
\]

and build the baseline/costate from `q3_hat`.  Let `q4` be the residual
quartic source.  The joint endpoint expansion contains

\[
R_{e_3\times q_4}
=\epsilon\delta\,
D_{34}^2V[\,e_3,q_4\,]
+\text{higher orders},
\]

where `epsilon=n^{-1/2}` and `delta=n^{-1}`.  If the mixed derivative norm is
bounded by `H_34`, then

\[
|R_{e_3\times q_4}|
\le
n^{-3/2}H_{34}
\left(\sum_l\|e_{3,l}\|\right)
\left(\sum_l\|q_{4,l}\|\right).
\tag{N.89}
\]

For ten active layers the unnormalized combinatorial coefficient can again be
of order

\[
100/n^{3/2}\simeq0.00305.
\]

Thus (N.87) is the complete remainder only when the H3 baseline is exact for
the relevant source quotient, the mixed residual is explicitly modeled, or
(N.89) is separately below the endpoint budget.

Task correlation illustrates the relative risk but is not by itself a source
norm bound.  The residual RMS fractions are

\[
\sqrt{1-.92545^2}\simeq0.379,
\qquad
\sqrt{1-.98346^2}\simeq0.181.
\tag{N.90}
\]

Moving the higher correlation to H3 reduces this action-space residual scale
by about a factor of `2.09`, in addition to H3's larger direct raw value in
Checkpoint 9H.  To insert (N.90) into (N.89), however, one must prove the
operator relating endpoint action residual to the mixed source norm; ordinary
task correlation cannot silently substitute for it.

### Correct compliance gate

The H3-resummed-baseline node passes for an approximate carrier only if one of
the following holds:

1. the residual H3 quotient is Bellman-dark to every retained q4 direction;
2. the joint refit includes the mixed derivative/action and certifies its
   value; or
3. the source-conditioned bound (N.89), plus q4-squared (N.87), fits Node E.

This qualification should be attached to the
`K31_H3_RESUMMED_BASELINE` terminal criterion.  Co-adaptation removes the
large known-H3 mixed series; it does not justify independent channel addition
in the presence of an unbounded H3 estimation error.

## Checkpoint 9Q: complete three-channel correlation gate

The H3/H4 allocation in Checkpoint 9H was conditional on perfect variance.
The board raw value and oracle ladder identify the variance opportunity as

\[
G_v
=29.4090718-14.656
=14.7530718
\qquad\text{in }10^{-9}.
\]

With retained value fractions `alpha_v,alpha_3,alpha_4`, the complete linear
residualized ladder is

\[
R_{\rm raw}
=29.4090718
-14.7530718\alpha_v
-4.370\alpha_3
-1.0313\alpha_4
\qquad\text{in }10^{-9}.
\tag{N.91}
\]

Raw `<1e-8` therefore requires

\[
14.7530718\alpha_v
+4.370\alpha_3
+1.0313\alpha_4
>19.4090718.
\tag{N.92}
\]

This is the binding capacity plane before nonlinear interaction/error terms.

### Variance is the non-negotiable channel

Even with perfect H3 and H4, variance must satisfy

\[
\alpha_v
>\frac{19.4090718-4.370-1.0313}{14.7530718}
\simeq0.9495,
\]

or

\[
q_v>\sqrt{0.9495}\simeq0.9744.
\tag{N.93}
\]

Thus a variance carrier at `q=.92545` is mathematically incapable of reaching
raw `1e-8`, regardless of perfect cubic and quartic additions.  Its lost
variance value alone is about `2.12e-9`, exceeding the ideal margin.

If all three channels have a common task correlation `q`, then

\[
q^2
>\frac{19.4090718}{14.7530718+4.370+1.0313}
\simeq0.9630,
\]

so

\[
q\gtrsim0.9813.
\tag{N.94}
\]

At `q=.98346` for all three, (N.91) gives approximately

\[
R_{\rm raw}=9.916\times10^{-9},
\tag{N.95}
\]

a narrow pass.  At `.92545` for all three it fails by a wide margin.

The earlier allocation coincidence is now explained.  Perfect variance with
`.92545` H3 loses about the same total value as `.98346` variance plus
`.98346` H3; both cases require quartic retained fraction about `0.8856`.
Consequently `.98346` quartic is still needed for a narrow pass in either
allocation, while `.92545` quartic narrowly misses.

### Binding use

Every graph node quoting the H3/H4 thresholds must also state its variance
premise.  In particular, the delayed K31 route is not a redemption path for a
sub-.974 variance carrier.  Its value can matter only after the primary D14
variance lane has crossed (N.93), preferably near `.98346`, and all channel
fractions have been measured in the same incumbent-residual value Gram.

Useful two-fixed-channel slices of (N.92) are:

- with variance and quartic both at `.98346`, H3 needs
  `alpha_3 about 0.9480`, or `q_3 about 0.9736`; keeping H3 at `.92545`
  gives raw about `1.040e-8` and fails;
- with H3 and quartic both at `.98346`, variance needs
  `alpha_v about 0.9615`, or `q_v about 0.9806`; and
- with variance and H3 both at `.98346`, quartic still needs
  `alpha_4 about 0.8856`, or `q_4 about 0.9411`.

Therefore the rational fidelity order is variance first, H3 second, and the
strict-depth quartic residual third.  Spending the high-correlation machinery
on quartic while leaving either dominant channel near `.92545` cannot cross
the raw target.

## Checkpoint 9R: H3 singular-tail gate induced by the full capacity plane

For an incumbent-residualized rank-`k` H3 carrier, let `sigma_j` be the
singular values of the endpoint-whitened target/source action in decreasing
order.  The exact best rank-`k` correlation is

\[
q_{3,\max}(k)^2
=\frac{\sum_{j\le k}\sigma_j^2}
{\sum_j\sigma_j^2}.
\tag{N.96}
\]

This is a sharp Eckart--Young/value-Gram statement.  Combining it with the
two-fixed-channel slice of (N.92), variance and quartic at `.98346` require

\[
q_3\gtrsim0.9736,
\]

so the admissible H3 tail is only

\[
\frac{\sum_{j>k}\sigma_j^2}{\sum_j\sigma_j^2}
<1-.9736^2
\simeq0.0521.
\tag{N.97}
\]

For the stronger `.98346` H3 target the tail gate is

\[
1-.98346^2=0.0328064,
\tag{N.98}
\]

whereas `.92545` permits `0.1435423` but is insufficient in the stated
three-channel allocation.

The current H3 resource sketch permits at most approximately:

- `k=36` for an exposed/local signed source; or
- `k=22` when a separate source-coordinate pass is required.

Therefore the exact immediate kill gates are

\[
\frac{\sum_{j>36}\sigma_j^2}{\sum_j\sigma_j^2}
\ge0.0521
\quad\Longrightarrow\quad
\text{NO for the exposed rank-36 route},
\tag{N.99}
\]

and the analogous `j>22` tail for the source-pass route.  A `.98346` claim
uses the stricter `0.0328064` thresholds.

This spectral condition is necessary and sufficient for best linear rank-`k`
value capture, but it does not make the signed source lawful.  Covariance/H2
data alone cannot identify signed H3: the transformation of the cubic source
that reverses its sign preserves every even H2 observable while reversing the
target action.  Static AQ/H31 total action is also incumbent D11-owned and has
zero novelty.

The symbolic H3 result is therefore:

1. a new generic compressed carrier has no guarantee of reaching `.9736`;
2. under D14, the only plausible high-correlation spelling is literal reuse
   or replacement with the pulled-back signed-H31 costate already exposed on
   the source side; and
3. promotion requires the actual endpoint-value singular tail in (N.97), not
   ordinary tensor energy.

The independent H3 derivation is recorded in
`LANE_h3_reverse_adjoint.md`.  Its singular receipt is now a hard input to the
three-channel capacity node and the H3-resummed q4 baseline.

## Checkpoint 9S: H4 pre-residual ceiling cannot satisfy the novelty gate

The independent H4 lane derives a diffuse zero-bias, one-reset oracle energy
fraction

\[
\alpha_{4,\rm pre}=0.96695,
\qquad
q_{4,\rm pre}=0.98334.
\tag{N.100}
\]

This is close to the high-correlation target but is explicitly **before**
synchronized residualization against D11/Q24 and the full depth arena.
Same-site AQ K31 belongs exactly to the assembled incumbent interface, so its
post-residual novelty is zero.

Let `O_complete` be the full incumbent fourth-order response map and let `V4`
be the oracle quartic endpoint row.  The only lawful novel target is

\[
V_{4,\rm novel}
=V_4P_{\ker O_{\rm complete}},
\tag{N.101}
\]

restricted further to future-visible strict-depth births.  If `C_new` is a
candidate carrier/action span, its relevant correlation is

\[
q_{4,\rm residual}^2
=\frac{
\|P_{C_{\rm new}}V_{4,\rm novel}\|_V^2
}{
\|V_{4,\rm novel}\|_V^2
}.
\tag{N.102}
\]

Neither the numerator nor denominator in (N.102) is determined by (N.100).
In particular, a high correlation on total same-site K31 can coexist with
zero novel value after the incumbent projection.

In the allocation where variance and H3 are both near `.98346`, Checkpoint 9Q
requires

\[
q_{4,\rm residual}>0.9411
\tag{N.103}
\]

on the actual residual target.  The H4 lane has not supplied this receipt.
It instead gives the same endpoint acquisition requirement already derived
here: all strict-depth/repeated-tail error must fit the absolute
`3.80e-6` ideal gate, or the tighter allocation-specific gate.

The reported coherent D10 approximation scale (`1.39%` per layer) and
normalized repeated-tail scale (`0.02520`) are useful inputs to a future
operator-norm composition bound, but they cannot be converted into (N.102)
without specifying whether they are amplitude, energy, independent, or
coherent errors.  No multiplicative correlation claim is licensed from those
percentages alone.

### Graph node

- **Suggested id:** `P2R0G19B_MATH_H4_POSTRESID_CORRELATION`
- **Label:** `Novel strict-depth H4 clears post-D11 correlation gate`
- **Kind:** `phase2_g19b_math_value_gate`
- **Status:** `open`
- **Terminal criterion:** Compute (N.102) on
  `ker(O_complete)` and require (N.103) or the allocation-specific threshold;
  total/same-site correlation is inadmissible evidence.
- **Evidence:** This file, equations (N.100)--(N.103), and
  `LANE_h4_reverse_adjoint.md`.
- **Goal-path summary:** Prevents the pre-residual `.98334` oracle ceiling from
  being mistaken for a deployable quartic correlation.

This node is a hard input to N2/N4 and the three-channel capacity gate.

## Checkpoint 9T: compact corrected graph ledger

This ledger is the authoritative graph mapping after the D11/Q24 novelty
correction.  The unresidualized total-K31 Nodes B--D in Checkpoint 8 are
background factorization theorems only; they must not be used as deployable
candidate gates.

| Suggested node id suffix | Status | Scope |
|---|---|---|
| `K31_SCALAR_BELLMAN` | `solved` | One scalar exactly sums supplied local order-`1/n` target actions. |
| `K31_STRICT_NOVELTY` | `solved` | Same-site/zero-length K31 is D11/Q24-owned; only `j<k` is new. |
| `K31_STRICT_HANKEL_RANK` | `solved` | Strict-lower target-blind primitive kernel has generic Hankel rank at least 11. |
| `K31_DELAYED_FACTOR` | `open` | Complete incumbent-residual endpoint value must pass (N.10)/(N.13). |
| `K31_DELAYED_FRESH_LAW` | `open` | Genuine post-reset residual law interface; incumbent-only branch is `closed_proof`. |
| `K31_DELAYED_ADD_ONEUNIT` | `closed_cost` | Generic multi-layer dense atom-law add-on cannot fit beside D14+D10=48. |
| `K31_DELAYED_THIN_ACTION` | `open` | Exceptional pre-cut thin action must satisfy rank/cost (N.72) and tail (N.84). |
| `K31_DELAYED_REPLACE_D10` | `open` | Conditional 38-GEMM replacement; must repay removed D10 value. |
| `Z22_CROSSED_K22_DARK` | `open` | Generic darkness is false and generic rank is at least 980; only exact arena-dark/shared exception remains. |
| `K31_H3_RESUMMED_BASELINE` | `open` | Theorem solved, but actual row must use joint H3 costate and bound residual-H3 mixing. |
| `H3_VALUE_TAIL_CAPACITY` | `open` | Rank-36/22 signed carrier must meet the endpoint singular-tail gate (N.97)--(N.99). |
| `H4_POSTRESID_CORRELATION` | `open` | Novel future-visible quartic correlation must be measured after the full old span. |
| `RAW_THREE_CHANNEL_CAPACITY` | `solved` | Exact linear capacity plane (N.91)--(N.94); candidate fractions remain open inputs. |
| `K31_ENDPOINT_CERT` | `solved` | Error-composition theorem; candidate numerical/symbolic satisfaction remains open. |

### Two additional full-schema records

#### Raw three-channel capacity

- **Id:** `P2R0G19B_MATH_RAW_THREE_CHANNEL_CAPACITY`
- **Label:** `Variance/H3/H4 retained-value plane for raw below 1e-8`
- **Kind:** `phase2_g19b_math_theorem`
- **Status:** `solved`
- **Priority:** `1`
- **Summary:** The exact residualized linear gate is (N.92).  Variance needs
  correlation above `.9744` even with perfect other channels; common
  correlation must exceed about `.9813`.
- **Terminal criterion:** Retain solved as a capacity theorem.  Every candidate
  must supply post-incumbent fractions in the same value Gram and separately
  bound nonlinear interaction error.
- **Evidence:** This file, Checkpoints 9H, 9Q, and 9R.
- **Decomposition children:** none.
- **Goal-path summary:** Prevents a weak variance or H3 lane from being
  papered over by a small quartic gain.

#### H3 value-tail capacity

- **Id:** `P2R0G19B_MATH_H3_VALUE_TAIL_CAPACITY`
- **Label:** `Signed H3 carrier clears the rank-36/22 value tail`
- **Kind:** `phase2_g19b_math_value_gate`
- **Status:** `open`
- **Priority:** `1`
- **Summary:** With variance and quartic near `.98346`, H3 needs correlation
  above `.9736`; the best rank-`k` endpoint-whitened tail must be below
  `5.21%`, with `k<=36` or `k<=22` under the two current cost forms.
- **Terminal criterion:** Supply the signed-source provenance and singular
  receipt (N.96)--(N.99), or close the scoped rank cap.
- **Evidence:** This file, Checkpoint 9R, and
  `LANE_h3_reverse_adjoint.md`.
- **Decomposition children:** none.
- **Goal-path summary:** This is the dominant nonvariance capacity gate and a
  prerequisite for the H3-resummed quartic baseline.

### Corrected critical edges

1. `RAW_THREE_CHANNEL_CAPACITY` is informed by
   `H3_VALUE_TAIL_CAPACITY` and `H4_POSTRESID_CORRELATION` and requires the
   primary variance correlation receipt.
2. `K31_STRICT_NOVELTY` and `K31_STRICT_HANKEL_RANK` inform
   `K31_DELAYED_FACTOR`, `K31_DELAYED_FRESH_LAW`, and both cost forms.
3. `K31_DELAYED_FACTOR` plus `K31_DELAYED_FRESH_LAW` inform
   `K31_DELAYED_THIN_ACTION` and `K31_DELAYED_REPLACE_D10`.
4. `Z22_CROSSED_K22_DARK` is a hard prerequisite for any complete one-product
   Z22 exception.
5. `H3_VALUE_TAIL_CAPACITY` informs and
   `K31_H3_RESUMMED_BASELINE` gates every quartic action.
6. All surviving candidate paths inform `K31_ENDPOINT_CERT`; only then may
   they inform the portfolio Gram/board path.

The active mathematical frontier in this lane is therefore exactly four
receipts: primary variance `q_v`, signed-H3 singular tail, post-residual
strict-depth H4 correlation, and lawful/costed fresh delayed action.  Every
other generic shortcut above is already solved or closed in its stated scope.

## Checkpoint 9U: directional surface-oracle rank is not Bellman matrix rank

An exact mixed directional oracle of the form

\[
\mathfrak O_{31}(F;U,V)
=K_4(F)[U,U,U,V]
\]

is one scalar fourth-chaos query and can be implemented by an exact local
surface functional for suitable AQ/ReLU laws.  However, a general K31 Bellman
matrix is not one such query even when that matrix has ordinary rank one.

Let `ell_i=K^Te_i` be the source directions after pulling a K31 action through
a dense reset.  The quartic Bellman polynomial is

\[
P_A(z)
=\sum_{i,j}A_{ij}\,\ell_i(z)^3\ell_j(z).
\tag{N.104}
\]

Its exact directional-query count is the partially symmetric tensor rank

\[
r_{31}^{\rm dir}(A;K)
=\operatorname{rank}_{(3,1)}
\left(
\sum_{i,j}A_{ij}\,
\ell_i^{\otimes3}\otimes\ell_j
\right),
\tag{N.105}
\]

where one rank-one term is `U^{otimes3} otimes V`.  This is the charged-cut
rank for an interface exposing only `O31(U,V)` calls.

If `A=xy^T` has matrix rank one, then

\[
P_A(z)
=\left(\sum_i x_i\ell_i(z)^3\right)
\left(\sum_j y_j\ell_j(z)\right).
\tag{N.106}
\]

The left factor is a diagonal/Fermat cubic in the invertible basis `{ell_i}`.
For generic nonzero `x`, its symmetric Waring rank is the number of nonzero
coordinates of `x`; a dense vector therefore has rank `n`.  An explicit
decomposition uses the `n` cubes already displayed, and the cubic
catalecticant gives the matching generic lower bound.  Consequently a dense
rank-one Bellman matrix generically needs `n` directional mixed queries, not
one.

For a rank-`r` factorization

\[
A=\sum_{t=1}^rx_ty_t^T,
\]

one valid upper bound is

\[
r_{31}^{\rm dir}(A;K)
\le\sum_{t=1}^r
\operatorname{wrank}
\left(\sum_i(x_t)_i\ell_i^3\right),
\tag{N.107}
\]

which is generically of order `nr`, capped by the ambient partially symmetric
rank.  Joint decompositions can improve this upper bound, but ordinary
`rank(A)=r` supplies only a flattening lower bound and never proves `r`
surface queries.

This does not invalidate the atom-law thin action in Checkpoint 9L.  An atom
interface exposes the coordinatewise vector `h_3(z)`, so a matrix factor
`A=PQ^T` is evaluated by thin projections exactly as in (N.70).  A directional
surface interface exposes only Veronese cubics `(U dot z)^3`, so it pays the
additional Waring rank.

### Decision gate for the surface route

A local tail-free `O31` formula is a one-unit provider only when the actual
incumbent-residual Bellman tensor in (N.105) has charged rank within the budget
or the surface implementation accepts the **already fused full polynomial**
`P_A` at the same cost.  A rank-`k` AQ core or rank-one Bellman matrix alone is
not sufficient evidence.  Across depth, the future adjoint cyclic module must
also remain in that same low-rank surface family; otherwise the strict Hankel
rank obstruction reappears.

The exact local surface construction is recorded in the polarization/K31
audit under `LANE_h6_remainder_bound.md`; this section supplies its missing
provider-rank qualification.

## Checkpoint 9V: portfolio synthesis makes variance provenance the first kill gate

The independent portfolio and self-consistent lanes identify the minimum
candidate stack as

\[
A+B+(C\ \text{or}\ D),
\]

where:

- `A` is the self-consistent all-distinct order-two variance provider;
- `B` is the owner-correct post-AQ signed-H3 provider;
- `C` is the thin strict-depth K31 action; and
- `D` is the Z22 exception, admissible only after its complete omission/dark
  certificate.

The complete task-weighted provenance receipt for `A` must retain at least

\[
a_v\ge0.95696,
\tag{N.108}
\]

or correlation scale

\[
q_v\ge\sqrt{0.95696}\simeq0.9782.
\tag{N.109}
\]

This portfolio threshold is stronger than the bare variance-only necessary
bound `.9744` in (N.93), because it incorporates the minimum feasible value
left for the downstream H3/quartic stack.  If (N.108) fails, B--D cannot
compensate and the complete architecture is dead before any quartic carrier
audit.

The all-distinct lane supplies an exact local signed tree contraction with one
dense product and a conditional all-reuse cost around `32.83` GEMM
equivalents.  But it does **not** yet prove that this local source equals the
certified truth-minus-estimated plane supplies.  A pairwise/moment-only state
cannot certify that provenance; positive-law parity/Rademacher witnesses give
the same type of nonidentifiability established in Checkpoints 9C--9D.

The certified truth-fed `18.7%` replay is not the missing receipt.  It consumes
offline true-minus-host planes and is nondeployable.  Its raw improvement alone
would leave approximately

\[
2.39095\times10^{-8},
\]

still requiring about `1.39095e-8` further reduction.  A deploy-only
all-distinct source is therefore a new value-unverified feature until its
complete old-arena, task-weighted provenance error is certified.

### Updated fail-fast order

1. **Variance provenance A:** require (N.108) on the complete incumbent-
   residual target.  Failure closes the whole A+B+(C/D) portfolio.
2. **Signed H3 B:** require legal source/costate reuse and the rank-36/22 tail
   in (N.97)--(N.99).
3. **Quartic C or D:** require post-D11 strict-depth correlation (N.102)--
   (N.103), plus thin-action cost for C or crossed-K22 darkness for D.
4. **Joint remainder:** enforce H3-resummed compliance and residual mixed
   bound (N.87)--(N.89), then the endpoint certificate.

This ordering is mathematical rather than experimental convenience: each
later channel has insufficient maximum raw capacity to repair failure of an
earlier gate.

### Graph edge update

Add the portfolio's all-distinct provenance node as a hard prerequisite of
`RAW_THREE_CHANNEL_CAPACITY`.  `H3_VALUE_TAIL_CAPACITY` and
`H4_POSTRESID_CORRELATION` remain its downstream inputs.  The thin K31 and Z22
nodes must not remain active if the provenance node closes below (N.108).

The cross-lane derivations are recorded in
`LANE_all_distinct_source.md`,
`LANE_self_consistent_chaos_hierarchy.md`, and
`LANE_diagram_portfolio_orthogonality.md`.

## Checkpoint 9W: exact scalar fail-fast inequality for variance provenance

Let `phi` be the maximum fraction of the certified variance action that the
lawful deploy-only all-distinct source can expose after the complete old arena.
Let

\[
\eta_A
=\eta_{\rm prov}
+\eta_{2111}
+\eta_3
\]

be a conservative sum of normalized amplitude errors from source provenance,
the missing 2111/all-distinct completion, and the third-order recurrence
remainder.  The self-consistent value gate is

\[
\boxed{
\phi(1-\eta_A^2)\ge0.95696.
}
\tag{N.110}
\]

This gives the immediate terminals:

\[
\phi<0.95696
\quad\Longrightarrow\quad
\text{portfolio closed regardless of later channels},
\tag{N.111}
\]

and, for `phi>0.95696`,

\[
\eta_A
\le
\sqrt{1-0.95696/\phi}.
\tag{N.112}
\]

Even the optimistic `phi=1` case permits at most

\[
\eta_A\le\sqrt{0.04304}\simeq0.2075
\]

in normalized amplitude, corresponding to the `4.304%` value-energy loss.
As `phi` approaches the threshold, the permissible provenance error collapses
to zero.

Equation (N.110) is the correct first symbolic receipt to request.  A local
tree identity, source-plane fit, or truth-fed replay supplies neither `phi`
nor `eta_A` until it is synchronized against the full deployed old span and
the certified truth-minus-estimated variance action.

## Checkpoint 9X: smooth near-Gaussian all-distinct identifiability witness

The parity laws in Checkpoints 9C--9D are not a discrete-support pathology.
There is an everywhere-positive smooth witness arbitrarily close to a standard
Gaussian law.

Let `gamma_4(x)` be the standard four-dimensional Gaussian density and choose
`0<epsilon<1`.  Define

\[
p_\sigma(x)
=\gamma_4(x)
\left[
1+\sigma\epsilon
\prod_{i=1}^4\tanh(x_i)
\right],
\qquad
\sigma\in\{+1,-1\}.
\tag{N.113}
\]

Because the product of the four hyperbolic tangents lies in `[-1,1]`, both
densities are strictly positive.  They integrate to one because `tanh` is odd
under the centered Gaussian.

For any integrable statistic `f` depending on at most three of the four
coordinates, at least one factor `tanh(x_j)` is integrated against an even
Gaussian density without any other dependence on `x_j`.  That integral is
zero.  Hence

\[
\mathbb E_{p_+}f
=\mathbb E_{p_-}f
=\mathbb E_{\gamma_4}f.
\tag{N.114}
\]

The two laws therefore agree exactly on means, covariance, every cubic
moment, and all K4/K31/K22 repeated-index fourth moments.  But

\[
\mathbb E_{p_\sigma}[X_1X_2X_3X_4]
=\sigma\epsilon
\prod_{i=1}^4
\mathbb E_{G\sim N(0,1)}[G\tanh G],
\tag{N.115}
\]

which is nonzero and changes sign.  Since all lower cross-cumulant partitions
vanish, this is also the all-distinct fourth cumulant.

The total-variation distance from the Gaussian is `O(epsilon)`, so the
witness exists in every sufficiently rich neighborhood of the Gaussian law.
After a generic dense reset, multilinearity gives the same nonzero delayed
K31 polynomial as (N.31), scaled by the positive factor in (N.115).

Consequently the incumbent-only Node-N3 no-go applies not only to arbitrary
or discrete laws but to any admissible layer-law class containing these smooth
near-Gaussian perturbations.  Escaping it requires a genuinely restrictive
law theorem that excludes the all-distinct tangent direction, or a fresh
observable that detects it.

The construction tensorizes over eleven disjoint four-coordinate blocks:

\[
p_{\boldsymbol\sigma}(x)
=\gamma_{44}(x)
\prod_{s=1}^{11}
\left[
1+\sigma_s\epsilon
\prod_{i\in Q_s}\tanh(x_i)
\right].
\tag{N.116}
\]

For sufficiently small `epsilon` these `2^11` laws lie arbitrarily close to
the Gaussian while carrying eleven independent smooth all-distinct tangent
directions and identical repeated-index interfaces.  The generic rank-11
delayed-observability/Hankel lower bound in Checkpoints 9D and 9K therefore
also holds in a smooth local neighborhood, not only on parity atoms.

## Checkpoint 9Y: exact strict surface-oracle row/cost theorem

The independent polarization audit confirms the Waring-rank qualification and
specializes it to the incumbent-residual strict-depth quotient.

Let `P_<` project onto strict delayed collision-free births and let `P_perp`
denote the complete incumbent value/source residualization.  For a dense reset
`K` and K31 Bellman matrix `A`, the exact delayed covector is

\[
d_{K,A}
=P_<\Gamma_4(K)^*S_{31}(A).
\tag{N.117}
\]

On the eleven parity basis, a sufficiently rich **family** of K31 rows has
generic `11 by 11` rank, as proved in Checkpoint 9D.  One frozen Bellman matrix
provides only one `1 by 11` row.  It observes the target-specific combination;
it does not identify all eleven primitive coordinates.

### Exact N3 row gate

Let `h` be the desired H3-resummed, true-minus-assembled endpoint row and let
`D` be the surface/fresh-law observation map.  One fused scalar clears N3 if
and only if

\[
hP_\perp
\in\operatorname{row}(DP_\perp),
\tag{N.118}
\]

equivalently

\[
\ker(DP_\perp)
\subseteq\ker(hP_\perp).
\tag{N.119}
\]

For one scalar row this means

\[
hP_\perp=\alpha\,dP_\perp,
\tag{N.120}
\]

with `alpha=1` when the oracle claims to return the exact fused Bellman score.
The Bellman row must be lawfully present before the charged cut, and `D` must
return the aggregate strict `j<k` true-minus-assembled residual.  A raw
same-site `O31` query remains zero novelty.

If `d_tilde` approximates `d`, its exact residual RMS uses the full joint
cross-layer covariance:

\[
\eta_d^2
=(d-\widetilde d)
\Sigma_{\perp F}
(d-\widetilde d)^*.
\tag{N.121}
\]

Without that covariance, per-layer errors must be added coherently in
amplitude and compared with the applicable `3.80e-6` or narrower gate.

### Exact Segre--Veronese query rank

The directional surface-call count is

\[
\begin{aligned}
r_{SV}=\min\{r:\quad
&P_\perp\sum_{i,j}A_{ij}
\operatorname{sym}(w_i^{\otimes3}\otimes w_j)\\
=&P_\perp\sum_{a=1}^r c_a
\operatorname{sym}(u_a^{\otimes3}\otimes v_a)
\},
\end{aligned}
\tag{N.122}
\]

where `w_i=K^Te_i`.  This is the residual Segre--Veronese rank, the precise
version of (N.105).  If `A=xy^T` and the singleton factor is shared, (N.122)
reduces to the cubic Waring rank of

\[
C_x(z)=\sum_ix_i(w_i\cdot z)^3,
\]

which is `n` for invertible `K` and generic dense `x` by the first
catalecticant.  Allowing varying singleton factors can reduce the rank only
with an explicit Segre--Veronese decomposition; ordinary rank-one `A` is not
such a certificate.

### Charged cost

For an `N`-atom law and one dense frozen row,

\[
G_{\rm row}
=G_{A,\rm new}+N/n
\tag{N.123}
\]

GEMM equivalents via the direct `h_3(Z)A` contraction.  `G_A,new=0` only when
the exact row/kernel is already exposed before the cut.  A directional factor
implementation with `r_C` cubic and `r_L` singleton thin projections costs

\[
G_{\rm factors,new}
+\frac{N(r_C+r_L)}{n^2},
\tag{N.124}
\]

or the earlier conservative `2Nr/n^2` when neither side is shared.  For
rank-one dense `A`, generic `r_C=n,r_L=1`; at `N=n` it approaches one unit only
through batching/shared factors, not through one directional oracle call.

A newly constructed reverse row of depth `d` costs `2d` dense products under
the conjugation model.  Therefore the exact local surface functional can
clear N3 at one boundary, but it does not eliminate the future cyclic module;
that module must be shared with D14/D10 or charged separately.

The full surface derivation is recorded in `LANE_h6_remainder_bound.md`
section 20.  Equations (N.117)--(N.124) are the binding integration with this
lane's novelty, Waring, and endpoint-error gates.

## Checkpoint 9Z: exact one-unit and replacement cost frontiers

The surface-oracle theorem separates three costs which must not be conflated:

1. identifying the target row;
2. transporting its future Bellman adjoint;
3. evaluating the local row on each fresh layer law.

One target scalar can solve the first problem without solving the latter two.
This section gives the exact budget frontiers under the charged dense-linear
provider model used by the card.

### Local atom-action cost

At layer `l`, let `Z_l` contain `N_l` fresh atoms and let the required local
action be

\[
 \phi_l
 =\langle h_3(Z_l),Z_l A_l^T\rangle_{p_l}.
\tag{N.125}
\]

Residualize `A_l` against every action factor already exposed before the cut.
Let `R_l` be the least rank of the residual matrix factorization

\[
 A_l^\perp=P_lQ_l^T,
\tag{N.126}
\]

and let `s_l` be the number of its two projected sides already exposed:
`s_l=0,1,2`.  Dense evaluation and thin evaluation give the sharp available
upper envelope

\[
 c_l
 \le
 \min\left\{
 \frac{N_l}{n},
 (2-s_l)\frac{N_lR_l}{n^2}
 \right\}
\tag{N.127}
\]

GEMM equivalents, with the second term equal to zero for `s_l=2` up to
lower-order elementwise contractions.  Here "sharp" means within this
provider model: a dense `N_l by n` times `n by n` product costs `N_l/n`, and
each unexposed thin projection costs `N_lR_l/n^2`.  Vertically stacking
independent layers does not reduce the count, because the total row volume is
`sum_l N_l`.

For ten full atom laws, `N_l=n`, the generic full-rank case costs

\[
 C_{\rm atoms}=\sum_{l=5}^{14}c_l=10.
\tag{N.128}
\]

To fit all ten local actions in one unit through thin factors requires

\[
 \sum_{l=5}^{14}(2-s_l)R_l\le n.
\tag{N.129}
\]

Thus the equal-rank gates are

\[
 \bar R\le n/20\simeq51.2 \quad(s_l=0),
 \qquad
 \bar R\le n/10\simeq102.4 \quad(s_l=1).
\tag{N.130}
\]

These are factorization receipts, not consequences of having only one scalar
endpoint target.  A generic dense target row has `R_l=n` and fails them.

### Future-adjoint charge

Let `C_rev` be the incremental cost of constructing the exact H3-resummed,
strict-depth Bellman rows `A_l`.  A new ten-transition conjugation stream has

\[
 C_{\rm rev}=2\cdot10=20.
\tag{N.131}
\]

Only literal reuse of an already billed row/kernel, or a proved thinner
intertwining realization, can reduce (N.131).  Calling the endpoint target
one-dimensional does not reduce a dense reverse conjugation.

The exact incremental provider budget is therefore

\[
 C_{\rm new}=C_{\rm rev}+C_{\rm atoms}.
\tag{N.132}
\]

For generic full atom laws with no reuse, `C_new=20+10=30`.

### Current schedule frontiers

The current `D14+D10` schedule charges `28+20=48` against a `49`-unit cap.
Consequently:

\[
\begin{array}{c|c|c}
\text{portfolio operation}&\text{available units}&\text{exact cost gate}\\
\hline
\text{add to D14+D10}&1&C_{\rm rev}+C_{\rm atoms}\le1\\
\text{replace D10, keep D14}&21&C_{\rm rev}+C_{\rm atoms}\le21\\
\text{replace D14, keep D10}&29&C_{\rm rev}+C_{\rm atoms}\le29.
\end{array}
\tag{N.133}
\]

The generic cost `30` fails all three rows: it exceeds the additive slot by
`29`, the D10-replacement allowance by `9`, and the D14-replacement allowance
by `1`.  This yields the useful sharp consequences:

- additive completion is possible only if the reverse is already free and
  all local residual actions together cost at most one, or if equivalent work
  is literally fused into already billed modules;
- D10 replacement fits exactly at `21` if the full reverse costs `20` and the
  local actions satisfy the thin gate (N.129);
- with a free reused reverse, ten generic dense local actions cost `10` and
  fit a D10 replacement comfortably;
- a generic `20+10` provider cannot be rescued by replacing D14.  Moreover,
  D14 carries the indispensable variance lane, whose capacity gate already
  requires approximately `q_v>0.9744` even with perfect H3/H4, so deleting it
  is not a credible raw-target route unless the replacement also repays its
  variance value.

### Replacement value must be recomputed against the smaller incumbent

Cost feasibility alone is not a fire criterion.  Let `F_0` be the complete
incumbent feature space with D10 removed, `D` the D10 feature map, `K` the new
strict K31 map, and `g` the endpoint target.  For any candidate map `X`, define
its conditional captured value

\[
 {\cal V}(X\mid F_0)
 =g^*\Sigma_{\perp F_0}X^*
 (X\Sigma_{\perp F_0}X^*)^\dagger
 X\Sigma_{\perp F_0}g.
\tag{N.134}
\]

The exact D10-replacement gain is

\[
 \Delta_{K\leftarrow D10}
 ={\cal V}(K\mid F_0)-{\cal V}(D\mid F_0).
\tag{N.135}
\]

It is **not** the incremental value of `K` residualized against the full
`F_0+D`, because that quantity gives no credit for D10 value which is lost on
replacement.  A lawful replacement must satisfy both

\[
 C_{\rm rev}+C_{\rm atoms}\le21
\quad\text{and}\quad
\Delta_{K\leftarrow D10}>0,
\tag{N.136}
\]

with the stronger raw-capacity threshold applied to (N.135) before any fire.

### Weakest premise that permits a one-unit route

The weakest information premise is not "one scalar per layer."  It is the
joint computational factorization

\[
 \boxed{
 C_{\rm rev,new}
 +\sum_{l=5}^{14}
 \min\left\{
 \frac{N_l}{n},
 (2-s_l)\frac{N_lR_l}{n^2}
 \right\}
 \le1,
 }
\tag{N.137}
\]

together with the exact row-space condition (N.118).  Equation (N.118) is
the information gate; (N.137) is the implementation gate.  Neither implies
the other.  In particular, a pre-cut fused scalar oracle can make (N.118)
true while hiding an unfunded 20-unit cyclic adjoint or ten-unit atom provider.

Graph-ready consequences:

- `K31_DELAYED_ADD_ONEUNIT`: `closed_cost` for the generic ten-layer dense
  provider; reopen only on a receipt satisfying (N.118) and (N.137).
- `K31_DELAYED_REPLACE_D10`: `open`, but requires either a free reverse plus
  local cost at most 21, or a full reverse plus the one-unit thin-action gate
  (N.129), and must pass the replacement-value equation (N.135).
- `K31_DELAYED_REPLACE_D14`: `closed_capacity` absent a theorem that the new
  module simultaneously repays D14's variance-channel value.

## Checkpoint 10: ReLU Bellman-action rank theorem

Checkpoint 9Z leaves one mathematical escape from the one-unit cost closure:
the residual Bellman actions might have rank at most roughly `51` when neither
projection is exposed, or `102` when one projection is exposed.  For the
actual ReLU covariance consumer this is not an automatic consequence of a
scalar endpoint or a rank-one covariance costate.

### Exact off-diagonal ReLU response kernel

Let `(X_i,X_j)` be jointly Gaussian with means `mu_i,mu_j`, standard
deviations `sigma_i,sigma_j`, correlation `rho_ij`, and standardized means

\[
 a_i=\mu_i/\sigma_i,
 \qquad
 a_j=\mu_j/\sigma_j.
\]

For `i != j`, the ReLU derivatives are
`phi'(x)=1_{x>0}` and `phi'''(x)=delta'(x)` in the distributional sense.
Writing

\[
 t_{ij}
 =\frac{a_j-\rho_{ij}a_i}{\sqrt{1-\rho_{ij}^2}},
\tag{N.138}
\]

differentiating the conditional Gaussian density at `X_i=0` gives

\[
 \boxed{
 W_{ij}
 =-\frac{\varphi(a_i)}{6\sigma_i^2}
 \left[
 a_i\Phi(t_{ij})
 +\frac{\rho_{ij}}{\sqrt{1-\rho_{ij}^2}}\varphi(t_{ij})
 \right].
 }
\tag{N.139}
\]

The diagonal is a separate K4 collision convention and may be completed
optimally; it does not weaken the off-diagonal conclusion below.

At zero off-diagonal correlation, (N.139) has the expansion

\[
 W_{ij}
 =u_iv_j+d_ie_j\rho_{ij}+O(\rho_{ij}^2),
\tag{N.140}
\]

where

\[
 u_i=-\frac{\varphi(a_i)a_i}{6\sigma_i^2},
 \quad v_j=\Phi(a_j),
 \quad d_i=-\frac{\varphi(a_i)(1-a_i^2)}{6\sigma_i^2},
 \quad e_j=\varphi(a_j).
\tag{N.141}
\]

The first term is separable, but the first correlation-dependent term is a
diagonally scaled copy of the complete correlation matrix.  For centered,
homoscedastic ReLU the separable term vanishes and the exact formula reduces
to

\[
 W_{ij}
 =-\frac{1}{12\pi\sigma^2}
 \frac{\rho_{ij}}{\sqrt{1-\rho_{ij}^2}},
 \qquad i\ne j.
\tag{N.142}
\]

Thus the local map from sufficiently small symmetric off-diagonal
correlations to off-diagonal `W` has nonsingular coordinatewise derivative.
Its image contains an open set of symmetric off-diagonal matrices.

### A rank-one endpoint does not imply a thin action

The K31 Bellman matrix is

\[
 A=2\Lambda\odot W.
\tag{N.143}
\]

If the covariance costate is rank one, `Lambda=pp^T`, then

\[
 A=2\operatorname{diag}(p)W\operatorname{diag}(p).
\tag{N.144}
\]

When `p` has full support, diagonal scaling preserves rank, every singular
nullity statement, and generic diagonal-completion rank.  Hence even the most
favorable rank-one terminal covariance costate does **not** make the ReLU K31
action rank one; its rank is that of the response kernel `W`.

For a generic symmetric off-diagonal matrix, the smallest possible rank after
arbitrary diagonal completion is `n-m`, where a nullity `m` can occur only if

\[
 \frac{m(m+1)}2\le n.
\tag{N.145}
\]

This follows by comparing the dimension
`r(2n-r+1)/2` of the symmetric rank-at-most-`r` variety with the
`n(n-1)/2` off-diagonal data dimensions.  At `n=1024`, the largest generic
nullity is `44`, so

\[
 \boxed{r_{\rm dc}(A)\ge980\quad\text{generically}.}
\tag{N.146}
\]

Equation (N.142) makes this dimension theorem applicable to the homogeneous
ReLU kernel on an open correlation neighborhood.  Therefore the **exact**
rank-`51` and rank-`102` thin-action branches are closed generically.  They
can reopen only through a special deployed correlation/costate intertwining
identity, not through ReLU algebra or endpoint scalarity.

### Approximate thin action: the binding value spectrum

Approximation depends on the endpoint-whitened, incumbent-residualized action
operator, not on the unweighted Frobenius spectrum of `W` or `A`.  Let its
singular values be

\[
 \sigma_1\ge\sigma_2\ge\cdots.
\]

For a total retained factor rank `k`, the best possible captured value
fraction is

\[
 \alpha_k
 =\frac{\sum_{r\le k}\sigma_r^2}
 {\sum_r\sigma_r^2}.
\tag{N.147}
\]

With independent layer blocks, the optimal allocation chooses modes across
layers in decreasing endpoint-value per charged projection unit; with
cross-layer covariance, the singular values must instead come from the full
joint block operator.  Independent per-layer spectra are not a valid receipt.

A useful proof-only kill follows from stable rank

\[
 \operatorname{sr}({\cal A})
 =\frac{\sum_r\sigma_r^2}{\sigma_1^2}:
 \qquad
 \alpha_k\le\frac{k}{\operatorname{sr}({\cal A})}.
\tag{N.148}
\]

The absolute perfect-variance/perfect-H3 quartic floor requires
`alpha_k>=0.27732`.  Therefore a candidate is closed whenever

\[
 \operatorname{sr}({\cal A})
 >\frac{k}{0.27732}.
\tag{N.149}
\]

Under the less forgiving allocation with H3 correlation near `.925` **even
while variance is perfect**, the quartic requirement rises to approximately
`alpha_k>=0.8856`, replacing the denominator in (N.149) by `.8856`.

For intuition only, if the residual homogeneous kernel has a Wigner-like
flat spectrum, then `sr` is asymptotically about `n/4`.  The exact semicircle
benchmark for retaining a fraction `p=k/n` of the largest singular modes is

\[
 p=\frac{2\theta-\sin2\theta}{\pi},
 \qquad
 \alpha(p)=\frac{2\theta-\tfrac12\sin4\theta}{\pi}.
\tag{N.150}
\]

It gives approximately `17.4%` energy at `p=.05` and `31.4%` at `p=.10`.
Thus the no-side `k about 51` branch misses even the absolute `27.732%`
quartic floor in this diffuse benchmark.  The one-side `k about 102` branch
barely clears that idealized floor but fails the `88.56%` requirement by a
wide margin.  This benchmark is not evidence that the deployed operator is
Wigner; it shows exactly what spectral concentration the remaining receipt
must beat.

Graph-ready split:

- `K31_RELU_EXACT_THIN_RANK`: `closed_proof` generically by (N.139)--(N.146);
  exceptional structured deployed matrices remain logically possible.
- `K31_RELU_APPROX_VALUE_TAIL`: `open_receipt`; require the joint whitened
  spectrum (N.147), the one-unit rank allocation (N.129), and the applicable
  raw-ladder fraction.  Kill immediately by (N.149) when the stable-rank
  threshold is exceeded.

## Checkpoint 10A: D14/D10 do not expose a thin-action side

The one-projection-reused `R about 102` branch in (N.130) is only a
conditional cost formula.  The presently defined D14 and D10 words do not
supply that reuse.

D14 carries the unit-pair symmetric primal tangent

\[
 E_{l+1}
 =W_{l+1}^T(D_l\odot E_l+I_l^{FH})W_{l+1},
\tag{N.151}
\]

and D10 carries unit-pair reverse masks `B_l,H_l` and scalar accumulators.  By
contrast, a rank-`R_l` K31 atom action requires the two rectangular sides

\[
 X_l=h_3(Z_l)P_l\in\mathbb R^{N_l\times R_l},
 \qquad
 Y_l=Z_lQ_l\in\mathbb R^{N_l\times R_l}.
\tag{N.152}
\]

The D14/D10 paid products have unit-pair state semantics and right-hand sides
from the conjugation recurrences.  Neither equals either product in (N.152).
When `N_l=n`, equality of array dimensions is still not an arithmetic alias:
reuse requires literal equality of the operand, right-hand side, cut, and
normalization, or an explicit sub-GEMM intertwiner deriving (N.152) from the
paid result.

Therefore the current typed schedule has

\[
 s_l=0
 \quad\text{for every K31 thin-action layer},
\tag{N.153}
\]

unless a new deployed-interface receipt proves otherwise.  The additive
one-unit gate is consequently

\[
 \sum_{l=5}^{14}2N_lR_l/n^2\le1,
\tag{N.154}
\]

which at ten `N_l=n` layers means average rank at most approximately `51`,
not `102`.  The rank-`102` alternative may be used only after identifying an
exact pre-cut `X_l` or `Y_l` already computed by a billed path.

Combining (N.153) with the ReLU theorem gives the current logical status:

- exact additive thin K31 is closed twice, by generic diagonal-completion
  rank at least `980` versus the rank-`51` budget;
- approximate additive thin K31 remains conditional on the joint value
  spectrum, but the diffuse semicircle benchmark's top-5% fraction `17.4%`
  is below even the ideal `27.732%` quartic floor;
- D10 replacement is different: removing its 20-unit word allows D14 plus
  ten dense one-row actions at total cost `38`, provided D14 is lawfully the
  required H3-resummed Bellman/Green realization and equation (N.135) repays
  D10's unique value.

Graph-ready node:

- `K31_D14_PROJECTION_REUSE`: `closed_type` for the current D14/D10
  definitions.  Reopen only with a literal pre-cut identity for one side of
  (N.152); common weights, common storage, or equal shape are insufficient.

## Checkpoint 10B: random-reset benchmark makes the rank-51 route knife-edge

The Wigner comparison in (N.150) is deliberately not a deployed receipt.  A
square random reset acting on an approximately isotropic covariance produces
a different favorable benchmark: a Marchenko--Pastur correlation matrix at
aspect ratio one.

Let `lambda` have the MP(1) density

\[
 f(\lambda)
 =\frac1{2\pi}\sqrt{\frac{4-\lambda}{\lambda}},
 \qquad0<\lambda<4.
\tag{N.155}
\]

For centered ReLU, (N.142) is linear in the `O(n^{-1/2})` off-diagonal
correlations to leading Frobenius order.  After collision-diagonal removal,
the comparison operator is therefore `R-I`, with eigenvalue `lambda-1` and
unit total second moment:

\[
 \mathbb E(\lambda-1)^2=1.
\tag{N.156}
\]

Set `lambda=4cos^2(theta)`.  Until the singular threshold reaches the lower
MP edge, the top positive modes have rank fraction and captured residual
energy

\[
 p(\theta)
 =\frac{2\theta-\sin2\theta}{\pi},
 \qquad
 \alpha_{MP}(\theta)
 =\frac2\pi\left(\theta-\frac{\sin6\theta}{6}\right).
\tag{N.157}
\]

The identity for `alpha_MP` follows from the exact cancellation

\[
 (4\cos^2\theta-1)^2\sin^2\theta
 =\frac12(1-\cos6\theta).
\]

Equation (N.157) gives

\[
 \alpha_{MP}(.05)\simeq0.301,
 \qquad
 \alpha_{MP}(.10)\simeq0.469,
\tag{N.158}
\]

where the argument denotes retained rank fraction rather than `theta`.  The
absolute quartic floor `.27732` is reached at only about `p=.0447`.  Hence an
MP-like reset is much more top-heavy than the Wigner benchmark and prevents
a universal spectral no-go at rank `51`.

It also yields a sharp conditional capacity test.  Let `f_<` be the fraction
of the **full verified quartic oracle** which remains both strict-depth and
incumbent-residual, and let `pi` collect every additional provenance and
acquisition retention factor.  The no-side rank-51 route can contribute at
most

\[
 a_{4,C}
 \le f_<\,\pi\,\alpha_{MP}(.05)
 \simeq0.301 f_<\pi.
\tag{N.159}
\]

Even under perfect variance and perfect H3 it must satisfy

\[
 \boxed{f_<\pi\gtrsim0.27732/0.301\simeq0.921.}
\tag{N.160}
\]

Thus the favorable random-reset benchmark leaves only about `7.9%` total
loss for same-site D11 ownership, strict-depth invisibility, old-span
projection, action truncation beyond the benchmark, law error, and frozen-net
shift combined.  With H3 correlation around `.925`, the perfect-variance
quartic demand near `.8856` already exceeds the benchmark ceiling even when
`f_< = pi = 1`.

This is not a proof that the deployed action has an MP spectrum.  It is the
best analytic argument for keeping the rank-51 exception open, and at the
same time supplies its fail-fast receipt:

1. measure/prove `f_<` after complete D11/arena residualization;
2. measure/prove the joint whitened top-51 energy, not raw matrix energy;
3. multiply every provenance/acquisition factor once;
4. close immediately if their product is below the raw-ladder threshold.

Graph-ready node:

- `K31_R51_RANDOM_RESET_KNIFE_EDGE`: `open_receipt`; optimistic analytic
  ceiling (N.160), not achieved value.  It depends on
  `K31_STRICT_NOVELTY`, `K31_RELU_APPROX_VALUE_TAIL`, and the complete
  provenance/error ledger.

## Checkpoint 11: shared-reverse 48-unit replacement theorem

There is one stronger architecture than the additive rank-51 exception.  A
single reverse covariance word can contract **both** the complete
all-distinct supply and the strict K31 supply at their birth cuts.  It replaces
the forward D14 realization and the D10 word; it is not an additive third
stream.

### Frozen multi-source Green identity

Let the frozen deployed tangent be

\[
 e_{l+1}=J_le_l+s_l^A+s_l^K,
\tag{N.161}
\]

where `s_l^A` is the complete all-distinct/certified supply at the declared
cut and `s_l^K` is the strict `j<k`, true-minus-assembled K31 supply.  For a
scalar endpoint with reverse state

\[
 \lambda_L=Dy,
 \qquad
 \lambda_l=J_l^*\lambda_{l+1},
\tag{N.162}
\]

the exact frozen response is

\[
 \boxed{
 \delta y
 =\sum_l\langle\lambda_{l+1},s_l^A\rangle
 +\sum_l\langle\lambda_{l+1},s_l^K\rangle.
 }
\tag{N.163}
\]

The same reverse word supplies the vector/matrix query in the one-product
all-distinct contraction and the Bellman row in the one-product dense K31
atom action, provided their cut conventions are synchronized.

For fourteen reverse transitions and ten active L5--14 births of each type,
the exact census is

\[
 \boxed{
 C_{AK}^{\rm frozen}
 =2\cdot14+10+10=48.
 }
\tag{N.164}
\]

This fits the 49-unit cap.  It preserves D14's endpoint response by Green
duality while replacing D10 with the strict K31 action.  It does not require
the rank-51 compression from Checkpoint 10; each K31 layer pays its full one
unit.

Value sharing is purely computational.  The all-distinct and K31 responses
must still be scored by one joint conditional Gram after complete old-span
residualization.  Equation (N.163) does not license adding their standalone
gains, and same-site K31 remains zero-credit.

### The decisive provider-level premise

Equation (N.163) is useful only if `s_l^A` is the **complete supply object**
whose frozen replay earned the D14 response.  Task-weighted equality is
sufficient:

\[
 \langle\lambda_{l+1},
 s_{l,\rm certified}^A-s_{l,\rm deployed}^A\rangle=0
\tag{N.165}
\]

at every synchronized cut, or a joint error bound small enough for the raw
gate.  A truth-fed `dK` plane fails (N.165).  So does a local primitive cubic
birth which omits the time-ordered cubic-pair completion contained in the
certified plane.

This distinction is forced by perturbative order.  If primitive signed-H3
births `b_j` are `O(n^{-1/2})`, their time-ordered pair response is

\[
 \delta^{(2)}y
 =\sum_{j<k}
 \left\langle
 \lambda_{k+1},
 D^2F_k[J_{k-1:j+1}b_j,b_k]
 \right\rangle,
\tag{N.166}
\]

which is `O(L^2/n)`.  At `L/sqrt(n)=0.5`, this is an order-one finite-depth
coefficient, not a disposable local remainder.  A linear contraction of the
primitive births in (N.163) does not create (N.166).

If instead `s_l^A` is already the complete order-`1/n` supply plane after the
cubic-pair resummation, then (N.163) evaluates that plane exactly and no
separate pair carrier is needed merely for the scalar frozen endpoint.

### Nonlocal finite-width closure condition

Let `D_l` be the deployed Markov state and `e_l^3` the hidden H3 correction
needed to form the H3-resummed K31 Bellman row.  The information-theoretic
local-closure condition beyond the frozen identity is that the mixed future
action factor through the deployed state.  One equivalent formulation is

\[
 D_l(h)=D_l(h')
 \quad\Longrightarrow\quad
 \Phi_{3\times4,l}(h)=\Phi_{3\times4,l}(h')
\tag{N.167}
\]

for every pair of reachable histories.  Equivalently, on the reachable
history space,

\[
 \ker D_l\subseteq\ker\Phi_{3\times4,l},
\tag{N.168}
\]

or the strict mixed block-Hankel row space lies in the already carried
deployed observability span.

When (N.168) holds, there is a deployed-local representative for every
value-relevant mixed action.  It yields a one-word implementation only if
that representative and the corrected local Jacobian/K31 row can also be
formed below the remaining arithmetic budget and have the same typed matrix
word.  Under those implementation conditions, one reverse recurrence sums
all time-ordered insertions.  This is the precise mathematical content needed
from an integrated/jointly retrained closure; information factorization alone
does not certify its price.

When (N.168) fails, two histories have identical deployed state but different
future mixed value.  No scalar accumulator or local modification of the one
reverse word can distinguish them.  One must carry a forward hidden tangent
or a second reverse mixed-response word.  The cheapest generic suffix adds
about `20` products, raising (N.164) to at least `68`; a separate full forward
and reverse pair is still more expensive.  Both violate the 49-unit cap.

### If exact closure is absent, the mixed remainder must be killed

The only alternative to (N.168) is to prove the omitted mixed term harmless.
With normalized H3 error history `e_l^3`, quartic births `q_l^4`, and mixed
Hessian bound `H_34`, the conservative amplitude gate is

\[
 \boxed{
 |R_{3\times4}|
 \le n^{-3/2}H_{34}
 \left(\sum_l\|e_l^3\|\right)
 \left(\sum_l\|q_l^4\|\right)
 \le\eta_{\rm mix}.
 }
\tag{N.169}
\]

`eta_mix` must fit inside the endpoint allowance after provenance, source,
action-tail, and frozen-network errors have already been charged; the broad
ideal allowance was only `3.80e-6`, and realistic channel allocations are
narrower.

The shared-reverse route therefore has a clean fail-fast order:

1. prove the complete task-weighted supply equality (N.165);
2. prove the deployed-state factorization (N.168), or the remainder bound
   (N.169);
3. verify synchronized query cuts so the same reverse word supplies both
   local actions;
4. compute the joint conditional value against the D14+D10 incumbent with
   D10 removed;
5. only then use the exact 48-unit census (N.164).

Graph-ready nodes:

- `AK_SHARED_REVERSE_48`: `solved_theorem`, application `open_interface`;
  exact frozen cost (N.164), conditional on (N.165) and synchronized cuts.
- `AK_COMPLETE_SUPPLY_PROVENANCE`: `open_receipt`; primitive H3 and truth-fed
  planes do not pass.
- `AK_MIXED_MARKOV_FACTOR`: `open`; exact information N&S gate
  (N.167)--(N.168), followed by a typed one-word/local-cost receipt.
- `AK_MIXED_REMAINDER`: `open_bound`; required only when exact factorization
  is unavailable, with terminal inequality (N.169).

## Checkpoint 11A: complete-seam prerequisite for the 48-unit theorem

The shared reverse state is not by itself the complete H3/H4 query.  At each
reset, the endpoint pullback can contain K4, K31, K22, and lower-order seam
components.  A K31-only scalar receives full quartic credit only after the
factor-through theorem is checked on the strict incumbent residual.

Let the complete residual quartic row be

\[
 g_{4,l}^{\rm seam}
 =P_{\perp F}
 \left[
 L_l^*(Q_{4,l},Q_{31,l},Q_{22,l})+P_l^*q_{3,l}
 \right],
\tag{N.170}
\]

and let `D_31,l` be the actual fresh strict-K31 observation/action map.  The
one-product K31 action is complete if and only if

\[
 \ker(D_{31,l}|_{R_l})
 \subseteq
 \ker(g_{4,l}^{\rm seam}|_{R_l}),
\tag{N.171}
\]

equivalently there is a Bellman row `A_l` with

\[
 g_{4,l}^{\rm seam}|_{R_l}
 =D_{31,l}^*A_l|_{R_l}.
\tag{N.172}
\]

The same audit applies jointly across depth after replacing `D_31,l` by the
strict block observability map.  K4/K22 terms can be omitted only when their
complete residual pullback is zero, is already incumbent-owned, or is included
in the same charged local action.  The generic crossed-K22 pullback has
diagonal-completion rank near width and is not a free extension of the K31
row.

Likewise, let `g_3^seam` be the complete signed-H3 query after all K4/K31/K22
and lower-order pullbacks.  The one-product all-distinct action must reproduce
that synchronized task row on the claimed source subspace.  Reusing merely an
`H31` or K31 face is insufficient: a single score spans only its cubic-star
slice, while generic Jordan/seam terms remain independent.

Consequently (N.164) is an exact `48` only under the **full-seam reuse**
premise:

1. the one reverse word supplies the exact local complete-seam rows at the
   same cuts;
2. every omitted K4/K22/lower face passes (N.171) or a zero-value/error bound;
3. forming the seam rows is elementwise/lower-order or already billed; and
4. the two one-product local contractions evaluate the complete residual
   actions, not selected faces.

If a separate D10-depth row is required to construct the missing seam, it adds
about `20` products and takes the architecture from `48` to `68`.  If generic
crossed K22 must be built, its rank is vastly above the remaining one-unit
allowance.  Either event closes the cap-feasible branch.

Graph-ready node:

- `AK_SHARED_REVERSE_FULL_SEAM`: `open_interface`, required by
  `AK_SHARED_REVERSE_48`.  Terminal criterion is (N.171)--(N.172) for the
  joint strict history plus a literal cost receipt for all surviving seam
  faces; K31/H31 face equality alone does not pass.

### Terminal-boundary blind subspace

The strict-lower observation has an exact boundary kernel independent of any
matrix spectrum.  Let `K_obs` be the set of fresh K31 observation depths and
define

\[
 R_{\rm term}
 =\bigoplus_{j:\,\{k\in K_{\rm obs}:k>j\}=\varnothing}R_j.
\tag{N.173}
\]

Every birth in `R_term` occurs at or after the final future K31 read.  By
definition,

\[
 \Delta\mathcal O_<|_{R_{\rm term}}=0.
\tag{N.174}
\]

Therefore the complete strict factor-through condition immediately implies

\[
 \boxed{G_\Delta|_{R_{\rm term}}=0.}
\tag{N.175}
\]

This is necessary and sufficient on the terminal subspace.  If the endpoint
has a direct K4, K22, primitive-V4, or other complete-seam response to one of
these births, K31-only strict completion is false regardless of how well all
earlier layers align.  The route can survive only if the deployed lineage
already owns that terminal response, the endpoint annihilates it, or there is
an additional post-birth fresh K31 observation before the endpoint.

For the target restricted to this terminal component, the exact conditional
charge after every allowed old/fresh observation is

\[
 E_{\rm term}^2
 =(G_\Delta P_{R_{\rm term}})
 \Sigma_{\perp [F,\Delta\mathcal O_<]}
 (G_\Delta P_{R_{\rm term}})^*,
\tag{N.176}
\]

where the covariance is conditioned jointly on the incumbent and all strict
observations.  When the terminal and earlier target pieces are correlated,
their prediction errors can interfere, so (N.176) must be recombined in the
full joint target projection rather than added as an independent marginal
energy.

Graph-ready node:

- `K31_STRICT_TERMINAL_BOUNDARY`: `solved_theorem`, application
  `open_interface`.  Terminal criterion: document a later fresh read for
  every claimed birth or prove (N.175); otherwise close exact K31 completion
  and charge (N.176).

## Checkpoint 12: finite-depth Dyson order required by the endpoint gate

Formal correctness through `1/n` retains at most two primitive cubic
insertions.  At fixed `n=1024,L=16`, that is not by itself a controlled
finite-depth approximation.

Let

\[
 x=\epsilon\sum_{l=1}^L\|V_{3,l}\|,
 \qquad \epsilon=n^{-1/2}.
\tag{N.177}
\]

Submultiplicativity and time ordering bound the norm of all `r`-insertion
terms by `x^r/r!`.  If the Dyson series is truncated after order `p`, then

\[
 \boxed{
 \|R_{>p}\|
 \le e^x\frac{x^{p+1}}{(p+1)!}.
 }
\tag{N.178}
\]

At unit normalized vertex strength,

\[
 x=L/\sqrt n=16/32=0.5.
\]

The two relevant bounds are

\[
 e^{1/2}\frac{(1/2)^6}{6!}
 \simeq3.58\times10^{-5},
\tag{N.179}
\]

after retaining only five insertions, and

\[
 e^{1/2}\frac{(1/2)^7}{7!}
 \simeq2.56\times10^{-6},
\tag{N.180}
\]

after retaining six.  Thus the broad ideal endpoint allowance
`3.80e-6` already requires primitive cubic order at least six under the
unit-norm worst-case bound.  The formal `1/n` truncation at cubic order two
misses this gate by orders of magnitude.  If odd orders vanish by an actually
proved symmetry, the first omitted even term after order four is still
`(1/2)^6/6! about 2.17e-5`; retaining through order six remains necessary.

For the ten-transition suffix, `x=10/32=.3125`; the analogous worst-case
gate requires order five.  These are lower requirements only when the stated
vertex norm and symmetry premises are certified; larger norms or a narrower
shared error budget increase the order.

Local quartic insertions have cumulative strength

\[
 y=\epsilon^2\sum_l\|V_{4,l}\|.
\tag{N.181}
\]

If they are truncated rather than placed inside the local transition, their
repetitions obey the same combined bound with `x+y`; in particular a two-H4
term scales like `O(L^2/n^2)` and can exceed a few-parts-in-`10^6` endpoint
budget.  A local `V4` term should therefore be incorporated into the carried
transition and resummed, or its Hessian-weighted repetition must be bounded
explicitly.

This result favors a state-space resummation over a sixth-order diagram list.
The pair-field recurrence already sums arbitrarily many separated cubic-pair
insertions while retaining `O(n^2)` state and `O(Ln^3)` work, and a local
quartic self-energy can be included in the same transition.  What it cannot
supply is the signed open-H3 source or a primitive H4 sector outside its
observable Hankel quotient; those require the provenance/shared-module gates
derived above.

Graph-ready node:

- `FINITE_DEPTH_DYSON_ORDER`: `solved_bound`.  Under normalized cumulative
  strength `x=.5`, any primitive truncation below order six fails the broad
  endpoint allowance; prefer exact pair/local-transition resummation.

## Checkpoint 12A: provenance capacity dominates the 48-unit cost pass

The independent exact-rest decomposition supplies the binding external
capacity inequality for the shared all-distinct source.  With `phi` the
endpoint-old-residualized projection-energy fraction, `d` the normalized
source distance, `C3` the third-order response constant, `eta` the aggregate
provenance/missing-sector/numerical amplitude error, and `R_other` every other
normalized remainder, raw-target feasibility requires

\[
 \boxed{
 1.7316\left[
 1-\phi+\phi(C_3d^3+\eta)^2
 \right]+R_{\rm other}
 \le0.07453.
 }
\tag{N.182}
\]

Even with exact internal approximation and `R_other=0`, (N.182) implies

\[
 \boxed{\phi\ge1-0.07453/1.7316=0.95696.}
\tag{N.183}
\]

Thus no arithmetic realization, including `AK_SHARED_REVERSE_48`, can reach
the target unless at least `95.696%` of this source's endpoint value survives
the deployed provenance and complete old-span quotient.

At the illustrative source correlation `.78`,

\[
 d^3=(\sqrt{1-.78^2})^3\simeq0.24506.
\]

Taking `phi=1`, `C3=.8`, and `R_other=0` leaves only

\[
 \eta
 \le\sqrt{0.07453/1.7316}-.8(0.24506)
 \simeq0.0114.
\tag{N.184}
\]

for all provenance, missing-sector, and numerical amplitude error combined.
The frozen `18.7%` replay was truth-fed and therefore supplies no positive
lower bound on `phi`.  Likewise, the fact that a target-conditioned action is
one scalar supplies no `phi` lower bound without a query-rank/provenance
theorem.

The fail-fast ordering for the 48-unit route is therefore capacity before
implementation:

1. prove (N.183) with deployed, label-free source/query provenance;
2. prove the complete-seam and mixed-Markov gates;
3. only then exploit the cost identity (N.164).

Graph-ready edge:

- `AK_COMPLETE_SUPPLY_PROVENANCE` is a hard prerequisite of
  `AK_SHARED_REVERSE_48`, with terminal gate (N.182), not merely nonzero source
  correlation.

## Checkpoint 13: generic no-go for fusing the two local products

The 48-unit shared-reverse census would fall to 38 if the all-distinct action
and strict-K31 action at each active layer could share one dense product.  A
common reverse costate or common storage does not establish such arithmetic
fusion.

After polarizing all prepared elementwise factors, write the two scalar
actions in the generic bilinear form

\[
 f_A=\operatorname{tr}(H_A^TX_AY_A),
 \qquad
 f_K=\operatorname{tr}(H_K^TX_KY_K),
\tag{N.185}
\]

where `(X_A,Y_A)` represent the all-distinct `C,M` operands and `(X_K,Y_K)`
represent the K31 atom/action operands.  Holding the prepared nonlinear
weights fixed is a lawful polarization slice; impossibility on that slice is
an impossibility for the full circuit.

For one scalar matrix-product contraction

\[
 b_H(X,Y)=\operatorname{tr}(H^TXY),
\]

the coefficient matrix obtained by flattening against `vec(X)` and `vec(Y)`
has rank

\[
 \operatorname{rank}{\cal B}_H=n\,\operatorname{rank}(H).
\tag{N.186}
\]

The direct-sum action (N.185), when its two operand pairs remain independent
after the complete incumbent quotient, therefore has flattening rank

\[
 \operatorname{rank}{\cal B}_{A\oplus K}
 =n\left[\operatorname{rank}(H_A)+
          \operatorname{rank}(H_K)\right].
\tag{N.187}
\]

Any scheme which first compresses the two left operands to one `n by n`
matrix, compresses the two right operands to one `n by n` matrix, performs one
dense multiplication, and applies arbitrary lower-order linear readouts has
coefficient-flattening rank at most `n^2`.  Consequently a necessary fusion
condition is

\[
 \boxed{
 \operatorname{rank}(H_A)+\operatorname{rank}(H_K)\le n.
 }
\tag{N.188}
\]

For generic full-rank local readouts, the left side is `2n`, so one-product
fusion is impossible.  Multiplying `(X_A+X_K)(Y_A+Y_K)` does not evade the
bound: it creates the unrequested cross terms `X_AY_K` and `X_KY_A`, and one
evaluation cannot separate them generically.

Condition (N.188) is necessary, not sufficient.  Exact fusion additionally
requires the direct-sum bilinear tensor to factor through the matrix-product
tensor with all cross terms annihilated.  Simple sufficient exceptions are:

1. literal product equality `X_AY_A=X_KY_K` at the synchronized cut;
2. an exact shared factor with complementary, provably disjoint supports;
3. a residual low-rank decomposition whose complete formation cost is already
   included in the one-unit action budget; or
4. one action lying in the incumbent span and therefore receiving zero new
   value and zero required compute.

For full-rank operands, the literal same-product exception has a sharp gauge
form.  The two billed products

\[
 P=CM,
 \qquad Q=ZA^T
\]

are identical if and only if there is a zero-cost invertible common gauge `S`
such that

\[
 \boxed{Z=CS,\qquad A^T=S^{-1}M.}
\tag{N.189}
\]

Necessity follows by taking `S=C^{-1}Z`; sufficiency is immediate.  On
rank-deficient supports, replace inverses by the corresponding common support
factorization and require that both kernels be respected.  After complete
old-span residualization the deployable receipt is

\[
 P_{AD}(CM-ZA^T)=0
\tag{N.190}
\]

together with provenance showing that the gauge factors themselves are
available below the cut at zero dense cost.  Output correlation is not a
substitute for (N.189)--(N.190).

If the products are distinct, the obvious exact packing

\[
 [\,C\;Z\,]
 \begin{bmatrix}M\\A^T\end{bmatrix}
 =CM+ZA^T
\]

has inner width `2n` and costs two GEMM equivalents, not one.

The NCQC-by-singleton identity can produce such equality for selected
same-site q4(211) faces, but those faces are D11/Q24-owned.  It does not imply
that the **strict true-minus-assembled** K31 operand equals the deployed
all-distinct source operand after a dense delayed reset.

Thus the current generic cost remains 48, not 38.  Reopen fusion only with a
literal typed factorization at every active cut; shared weights, a common
costate ancestor, common scratch memory, or equal matrix dimensions do not
pass.

Graph-ready node:

- `AK_LOCAL_PRODUCT_FUSION`: `closed_proof` for independent generic full-rank
  residual operands by (N.186)--(N.188); structured exact quotient remains an
  explicit exception.

## Checkpoint 14: antisymmetric-Jordan q211 inside the shared-reverse stack

The sewn Q22/Q211 commutator action can be added to
`AK_SHARED_REVERSE_48` without a new dense product, but only under three
logically independent receipts:

1. the synchronized AK product/action gauge;
2. the Q22/Q211 common-support sewing; and
3. literal residency of the unsymmetrized NCQC product.

Neither of the first two implies the other, and neither implies the third.

### The q211 zero-product identity

At one layer let `M_i` be the symmetric NCQC core for repeated anchor `i`.
After complete collision normalization and old-span subtraction, let the sewn
Q22/Q211 Bellman slice be `Lambda_i`.  If

\[
 \Lambda_i=\theta_i uu^T
 \qquad\text{for every live }i,
\tag{N.191}
\]

define

\[
 N=\sum_j u_jM_j,
 \qquad
 P_i=M_iN,
 \qquad
 J_i=\operatorname{sym}P_i,
 \qquad
 A_i=\operatorname{skew}P_i.
\tag{N.192}
\]

Then the complete sewn diagonal-Q22 plus off-diagonal-Q211 commutator action
is

\[
 \boxed{
 S_{22+211}
 =-8\sum_i\theta_i\|[M_i,N]\|_F^2
 =-32\sum_i\theta_i\|A_i\|_F^2.
 }
\tag{N.193}
\]

If the paid Jordan path literally forms `P_i=M_iN` and retains it before
symmetrization, both `J_i` and `A_i` are available by transpose/addition and
the small skew Gram costs no further dense product.  Exposure of `J_i` alone
is insufficient when it was formed without retaining the full `P_i`.

### Independence from the synchronized AK gauge

Use calligraphic letters for the two products in Checkpoint 13:

\[
 P_A=C\mathcal M,
 \qquad
 P_K=Z\mathcal A^T.
\]

Their full-support synchronized gauge is

\[
 Z=CS,
 \qquad
 \mathcal A^T=S^{-1}\mathcal M.
\tag{N.194}
\]

Equation (N.194) acts on the physical/atom inner index of the dense local
product.  Equation (N.191) acts on the pair labels inside each anchor-specific
Q22/Q211 Bellman slice.  They live in different typed spaces unless a further
intertwiner is explicitly supplied.

The nonimplications have elementary counterexamples:

- Take `C=Z=I`, `mathcal M=mathcal A^T=I`, and `S=I`.  Equation (N.194)
  holds, but the two live slices
  `Lambda_1=e_1e_1^T` and `Lambda_2=e_2e_2^T` have no common one-dimensional
  range.  Thus the AK gauge does not imply q211 sewing.
- Take `Lambda_i=theta_i e_1e_1^T`, preserve the common leg under every
  reverse reset, and make every `M_iN` a resident product.  Choosing
  `C=Z=I`, `mathcal M=I`, and `mathcal A^T=2I` makes (N.194) impossible even
  though sewing and antisymmetric residency are exact.

Rank-one sewing also does not force a computation to retain `P_i`; an
implementation may form only a symmetric Jordan result.  Conversely, a bank
of resident `M_iN_a` products can support a common factor width greater than
one and therefore does not imply (N.191).

### Exact combined gauge

Let `P_base,l` denote the matrices actually produced by the two paid local
products of `AK_SHARED_REVERSE_48` at layer `l`, before destructive
symmetrization or compression.  Under full support, the simultaneous property
"the AK products obey their synchronized gauge **and** q211 is a one-suffix,
zero-new-product action" holds at every active layer if and only if there are

- a physical product gauge `S_l`;
- a nonzero pair factor `u_l` and scalars `theta_li`;
- old-span representatives `X_li` in the complete Q22/Q211 incumbent slice;
- zero-dense-cost typed extraction maps `E_li`; and
- reverse collinearity scalars `gamma_li`;

such that

\[
 \boxed{
 \begin{aligned}
 Z_l&=C_lS_l,
 &\mathcal A_l^T&=S_l^{-1}\mathcal M_l,\\
 \Lambda_{li}-X_{li}&=\theta_{li}u_lu_l^T,
 &u_l&\parallel\bar u_l,\\
 B_{li}^Tu_{l+1}&=\gamma_{li}u_l,
 &N_l&=\sum_j(u_l)_jM_{lj},\\
 \mathcal E_{li}(P_{\rm base,l})&=M_{li}N_l
 &&\text{before symmetrization.}
 \end{aligned}
 }
\tag{N.195}
\]

Here `bar u_l` is the literal factor carried by the resident suffix and
`B_li` is the anchor-dependent reverse pair leg.  The second line sews both
the off-diagonal 211 entries and the canonical Q22 collision diagonal; a
rank-one fit to off-diagonal entries alone does not pass.  The third line is
the exact recurrence condition: one factor remains one factor if and only if
all live `B_li^T u_(l+1)` are collinear.

For rank-deficient AK operands, replace the first line by the action-coset
intersection

\[
 (P_A+\ker\mathcal R_A)
 \cap(P_K+\ker\mathcal R_K)\ne\varnothing
\tag{N.196}
\]

and require `E_li` to act on a literal one-product representative of that
intersection.  Equations (N.195)--(N.196) are necessary and sufficient under
the declared zero-cost operation class for this simultaneous property.
Algebraic product equality without resident factors, or abstract rank one
without suffix alignment, is not an implementation.

For inserting q211 into the already costed 48-unit AK base **without** trying
to fuse the two AK products, delete the first line of (N.195) and omit
(N.196).  The remaining sewing, recurrence, and literal-extraction lines are
necessary and sufficient for zero incremental dense products.  This is why
q211 can pass while the generic AK product-fusion branch remains closed.

If the AK path keeps its two already budgeted local products, preserving the
antisymmetric half changes the ledger by zero:

\[
 \boxed{C_{AK+211}=28+10+10+0=48.}
\tag{N.197}
\]

The possible reduction from 48 to 38 by separately exploiting an AK
one-product representative is a different packaging/lifetime optimization;
q211 sewing neither proves nor is required for it.  The binding integrated
theorem conservatively retains the certified 48-unit census.

For the q211 scalar alone, the products `M_iN` suffice.  Claiming the complete
H3/q211 seam at zero cost is stronger and additionally requires every needed

\[
 \{M_iN_a,\ N_aN_b,\ M_i^2\}
\tag{N.198}
\]

to be a resident product/view at the synchronized cut.  Otherwise only the
antisymmetric scalar is free; omitted seam faces need new products or an
endpoint error bound.

The exact NCQC-by-singleton reuse is the special resident product

\[
 P_{ia}=M_iN_a:
 \qquad
 \operatorname{sym}P_{ia}=M_i\circ N_a
 \quad\text{feeds the compatible H3/Jordan slot,}
 \qquad
 \operatorname{skew}P_{ia}
 \quad\text{feeds q211-perp.}
\tag{N.198a}
\]

This literal symmetric/skew sharing is independent of the all-distinct/K31
operand gauge.  It does not supply the anchor-square `M_i^2` or factor-pair
`N_aN_b` slots in (N.198).

### Exact supply-weighted tail and promotion gate

Let `B_AK` be the complete retained old plus all-distinct plus strict-K31
response span.  For an approximate rank-one sewing define

\[
 R_{li}
 =\Lambda_{li}^{\perp}-\theta_{li}u_lu_l^T,
\]

the commutator supply slices

\[
 (G_{li})_{jk}
 =\langle[M_{li},M_{lj}],[M_{li},M_{lk}]\rangle_F,
\]

and the endpoint action process

\[
 \mathscr V(R)
 =-8\sum_l\beta_l\sum_i\langle R_{li},G_{li}\rangle_F.
\tag{N.199}
\]

The binding truncation error is the supply-weighted, jointly residualized
quantity

\[
 \boxed{
 \eta_1^{\rm cf}
 =\|P_{\mathcal B_{AK}}^\perp\mathscr V(R)\|_{L^2(\Omega)},
 }
\tag{N.200}
\]

not an unweighted singular tail of the `Lambda_i`.  Let `eta_rec` charge
reverse-collinearity leakage, `eta_seam` charge imperfect extraction/full-seam
closure, and `eta_fp` charge preservation of the skew block.  The coherent
carrier error gate is

\[
 \boxed{
 \eta_{211}^{\rm tot}
 =\eta_1^{\rm cf}+\eta_{\rm rec}+\eta_{\rm seam}+\eta_{\rm fp}
 \le3.80\times10^{-6}.
 }
\tag{N.201}
\]

The retained conditional signal is

\[
 F_{211}
 =P_{\mathcal B_{AK}}^\perp
 \left[-32\sum_{l,i}\beta_l\theta_{li}
 \|\operatorname{skew}(M_{li}N_l)\|_F^2\right],
 \qquad
 s_{211}=\|F_{211}\|_2.
\tag{N.202}
\]

To cover the isolated quartic shortfall `2.86e-10`, the exact robust
promotion gate is

\[
 \boxed{
 (s_{211}-\eta_{211}^{\rm tot})_+^2\ge2.86\times10^{-10}
 \quad\Longleftrightarrow\quad
 s_{211}\ge\eta_{211}^{\rm tot}+1.69115\times10^{-5}.
 }
\tag{N.203}
\]

For another portfolio allocation, replace `2.86e-10` by the actual remaining
joint raw-gain requirement.  The q211 signal must never be added to standalone
AK gains; (N.202) conditions it on the complete retained stack.

Graph-ready nodes:

- `AK_Q211_GAUGE_INDEPENDENCE`: `solved_proof`; (N.194) and (N.191) are
  independent typed conditions.
- `AK_Q211_COMBINED_GAUGE`: `solved_theorem`, application `open_interface`;
  exact conjunction (N.195)--(N.196).
- `AK_Q211_ANTISYM_RESIDENCY`: `open_implementation`; require literal
  pre-symmetrization `M_iN` extraction.
- `AK_Q211_SHARED_REVERSE_48`: `conditional_cost_pass`; zero incremental
  GEMMs under the combined gauge, total (N.197).
- `AK_Q211_SUPPLY_TAIL`: `open_receipt`; joint conditional gates
  (N.200)--(N.203).

## Checkpoint 15: five-reset r62 Bellman geometry and the finite Stein--Price endpoint gate

The terminal-factored r62 shell has the optimistic arithmetic subtotal

\[
 C_{62,5}=47.43295,
 \qquad
 s_{62,5}=49-C_{62,5}=1.56705.
 \tag{N.204}
\]

Its five reset errors cannot be combined by root-sum-square merely because
the declared rest is Gaussian or because a finite Price/Stein expansion is
orthogonal at one local reference law.  The exact answer has three distinct
forms:

1. five causal Bellman cross scalars for a reusable prefix certificate;
2. one aggregate cross scalar, or the total endpoint norm itself, for one
   fixed final output; and
3. a Pythagorean tail with no cross acquisition only when the finite and full
   providers are *fixed nested endpoint feature spaces* and both are jointly
   refit in the same corpus metric.

These alternatives are proved below.  They are certificates for a compiled
route.  They do not construct the still-missing r62 seam section, a Gaussian
rest source, or a future costate.

### Exact nonlinear five-reset telescope

Work in the weighted corpus/endpoint Hilbert space `H`, normalized so that
`||r||_H^2` is raw MSE.  Let `c_0` collect
the residual of the chosen ideal reference together with every prefix and
seam error which precedes the r62 suffix.  For a nonlinear suffix, define
hybrid endpoints `Y^(j)`, `j=0,...,5`, by replacing the first `j` suffix
resets by the proposed r62/rest reset and leaving the remaining resets on the
reference route.  Put

\[
 z_j=Y^{(j)}-Y^{(j-1)},\qquad 1\le j\le5.
 \tag{N.205}
\]

With the sign convention absorbed into the `z_j`, the proposed final
residual is exactly

\[
 \boxed{r_{62}=c_0+\sum_{j=1}^5z_j.}
 \tag{N.206}
\]

This is a finite telescoping identity, not a tangent approximation.  A local
formula `z_j=P_j epsilon_j` is optional: in a nonlinear self-consistent
reset, `P_j` must be the exact hybrid secant/future map, not the Jacobian at
only the approximate trajectory.

Use the current raw reference

\[
 R_0=2.9409\times10^{-8}
 \]

and normalize

\[
 e_0={\|c_0\|_H^2\over R_0},\qquad
 d_j={\|z_j\|_H^2\over R_0},\qquad
 Z_{j-1}=c_0+\sum_{k<j}z_k,
 \tag{N.207}
\]

\[
 \boxed{
 \chi_j={\langle Z_{j-1},z_j\rangle_H\over R_0}.}
 \tag{N.208}
\]

Then the normalized prefix energies obey the exact causal recursion

\[
 \boxed{
 E_0=e_0,\qquad
 E_j=E_{j-1}+d_j+2\chi_j,
 \qquad
 {\operatorname{raw}(r_{62})\over R_0}=E_5.}
 \tag{N.209}
\]

The goal thresholds are

\[
 \tau_{10}={10^{-8}\over R_0}\simeq0.340032,
 \qquad
 \tau_6={6\times10^{-9}\over R_0}\simeq0.204019.
 \tag{N.210}
\]

Thus raw below `1e-8` is exactly `E_5<tau_10`; the more aggressive `6e-9`
gate is `E_5<tau_6`.

The scalar in (N.208) is information-theoretically minimal at each causal
step.  Given only `E_(j-1)` and `d_j`, positive semidefiniteness permits the
entire interval

\[
 -\sqrt{E_{j-1}d_j}\le\chi_j
 \le\sqrt{E_{j-1}d_j},
 \tag{N.211}
\]

and every point is realized by a two-vector Gram matrix.  Hence no function
of the two diagonal energies alone determines `E_j`.  One aggregate
past-versus-new scalar per reset is necessary and sufficient for the exact
causal scalar certificate.  Adjacent-reset correlations are insufficient:
`z_j` may be orthogonal to `z_(j-1)` while correlated with `c_0` or any
earlier `z_k`.

If only an upper normalized coherence `0<=bar rho_j<=1` is known,

\[
 \chi_j\le\bar\rho_j\sqrt{E_{j-1}d_j},
 \]

the sharp scalar robust recursion is

\[
 \boxed{
 E_j^+=E_{j-1}^++d_j
       +2\bar\rho_j\sqrt{E_{j-1}^+d_j}.}
 \tag{N.212}
\]

No smaller universal one-step upper bound follows from those scalars.

### One final-only cross scalar and the sharp six-block elliptope

For the final endpoint alone define

\[
 \Chi_5:=\sum_{j=1}^5\chi_j
 =\sum_{0\le i<j\le5}{\langle z_i,z_j\rangle_H\over R_0},
 \qquad z_0:=c_0.
 \tag{N.213}
\]

Then

\[
 \boxed{
 E_5=e_0+\sum_{j=1}^5d_j+2\Chi_5.}
 \tag{N.214}
\]

Thus one aggregate cross scalar is necessary and sufficient once all six
diagonal energies are known.  It can be acquired by the endpoint
polarization identity

\[
 2R_0\Chi_5
 =\left\|c_0+\sum_jz_j\right\|_H^2
  -\|c_0\|_H^2-\sum_j\|z_j\|_H^2.
 \tag{N.215}
\]

If the aggregate suffix vector `delta=sum_j z_j` is directly available, it
is enough instead to know `||delta||^2` and the single cross
`<c_0,delta>`; all reset-to-reset crosses are already inside
`||delta||^2`.  Directly acquiring `||r_62||^2` makes even that decomposition
unnecessary.

The exact no-cross-information bounds are also closed.  Put

\[
 a_0=\sqrt{e_0},\qquad a_j=\sqrt{d_j},\qquad
 A=\sum_{j=0}^5a_j,\qquad a_{max}=\max_ja_j.
 \tag{N.216}
\]

Over every positive-semidefinite Gram matrix with these diagonal entries,

\[
 \boxed{
 E_{min}=\left(\max\{2a_{max}-A,0\}\right)^2,
 \qquad
 E_{max}=A^2.}
 \tag{N.217}
\]

Both endpoints are attained: the upper value aligns all six vectors; the
lower value is the polygon cancellation law.  Equivalently, the physically
admissible aggregate cross interval is

\[
 {E_{min}-e_0-\sum_jd_j\over2}
 \le\Chi_5\le
 {E_{max}-e_0-\sum_jd_j\over2}.
 \tag{N.218}
\]

Therefore, with only scalar energies:

- `E_max<tau` is the sharp robust pass;
- `E_min>=tau` is the sharp robust close; and
- the interval between them is genuinely unresolved until a cross or
  directional block is acquired.

In particular, an initial block above the target is not by itself an
absolute close: a later block can cancel it.  It only closes a robust
positive-budget argument.  The polygon lower value is the correct absolute
no-cross close test.

For five equal reset energies `d` and `e_0=0`, the exact extremes reduce to

\[
 E_{max}=25d,\qquad E_{RSS}=5d.
 \tag{N.219}
\]

Hence

\[
\begin{array}{c|cc}
\text{goal}&\text{unknown coherence}&\text{proved causal orthogonality}\\
\hline
10^{-8}&d<\tau_{10}/25\simeq0.0136013
       &d<\tau_{10}/5\simeq0.0680064\\
6\times10^{-9}&d<\tau_6/25\simeq0.00816076
       &d<\tau_6/5\simeq0.0408038.
\end{array}
 \tag{N.220}
\]

With a nonzero seam block, replace these by

\[
 \boxed{
 d<\left({\sqrt\tau-\sqrt{e_0}\over5}\right)^2
 \quad\text{for the coherent robust pass},
 \qquad
 d<{\tau-e_0\over5}
 \quad\text{under exact causal orthogonality},}
 \tag{N.221}
\]

when the right sides are positive.

If directional residual covariance blocks `C_j` are available at their
injection sites and `P_j` are the exact frozen future maps, the sharp
five-reset block certificate is the SDP

\[
 \boxed{
 \begin{aligned}
 U_5&=\max_{G\succeq0,\ G_{jj}=C_j}
 \sum_{i,j=0}^5
 \operatorname{tr}(P_i^*HP_jG_{ji}),\\
 L_5&=\min_{G\succeq0,\ G_{jj}=C_j}
 \sum_{i,j=0}^5
 \operatorname{tr}(P_i^*HP_jG_{ji}).
 \end{aligned}}
 \tag{N.222}
\]

After division by `R_0`, `U_5<tau` robustly passes and `L_5>=tau`
robustly closes.  Collapsing each transported block to only its trace gives
exactly (N.217); matrix geometry helps only when it is actually retained.

### When a frozen Gaussian rest really gives `chi_j=0`

Let `F_(j-1)` contain the accumulated past before reset `j`, and let
`epsilon_j` be a local error injection.  Write the Bellman test as

\[
 \chi_j=R_0^{-1}E\langle q_j,\epsilon_j\rangle,
 \qquad
 q_j=P_j^*HZ_{j-1}.
 \tag{N.223}
\]

If

\[
 q_j\ \text{is }\mathcal F_{j-1}\text{-measurable},
 \qquad
 E[\epsilon_j\mid\mathcal F_{j-1}]=0,
 \tag{N.224}
\]

then `chi_j=0` exactly.  Conversely, conditional centering is necessary if
zero cross is required against every bounded `F_(j-1)`-measurable Bellman
test: choose the test to expose the conditional mean.  Thus (N.224) is the
sharp reusable martingale-difference theorem, not a heuristic independence
claim.

A finite conditional Hermite projection or a same-row Stein control can
satisfy (N.224) when all of the following hold:

1. a genuinely fresh independent Gaussian variable is retained at the
   sample level;
2. the omitted component is explicitly conditionally centered; and
3. the future map is frozen/past-measurable, so it does not depend on that
   same omitted component.

This theorem does **not** apply to the analytic D10 Gaussian rest or to a
finite Price/heat expansion after the Gaussian variable has been integrated
out.  Its truncation remainder is a deterministic conditional bias

\[
 b_j(\mathcal F_{j-1})
 =S_j^{exact}(\mathcal F_{j-1})-S_j^{finite}(\mathcal F_{j-1}),
 \tag{N.225}
\]

so `E[b_j|F_(j-1)]=b_j`, not zero.  Literal independence of the *integrated*
Gaussian rest supplies its law, but it does not make biases at five frozen
nets/layers mutually orthogonal in the corpus endpoint metric.  Likewise, a
self-consistent future reset generally makes `q_j` depend on the current
innovation and breaks the measurability premise in (N.224).  Such branches
must use (N.209), (N.214), or (N.222).

Local chaos orthogonality is also insufficient.  It is in the local Gaussian
`L2` metric, whereas `q_j` contains the anisotropic future Bellman pullback
and earlier high-chaos exits and re-entries.  It implies the endpoint cross
only when the relevant chaos subspaces are invariant and orthogonal under
the complete frozen future map, which is exactly the missing intertwining
receipt.

### Finite Stein hierarchy: a lawful local amplitude, not a coherence law

For one scalar AQ core, the matrix-power Stein hierarchy obeys the existing
exact bound

\[
 \|\Gamma_k^c\|_2
 \le a^k\left[{k+2\over2}\|u\|
             +{1\over\sqrt2}\|A\|_F\right],
 \qquad a=\|A\|_{op}<1.
 \tag{N.226}
\]

If the endpoint/Bellman derivative acting on every omitted unit-weight
hierarchy term has norm at most `L_j`, summing the geometric tail after
degree `Q` gives the explicit local endpoint-amplitude bound

\[
 \boxed{
 b_{j,Q}^{Stein}\le
 L_j a^{Q+1}\left[
 {\|u\|\{(Q+3)-(Q+2)a\}\over2(1-a)^2}
 +{\|A\|_F\over\sqrt2(1-a)}
 \right].}
 \tag{N.227}
\]

For non-unit hierarchy coefficients their absolute majorant belongs inside
the sum before using (N.227).  Mixed cores additionally require a common
word-algebra or target-functional-rank receipt; `a<1` does not prevent the
generic Burnside word space from becoming the full matrix algebra.

Equation (N.227) supplies a diagonal amplitude `sqrt(d_j)` for the five-reset
gate.  It supplies no cross sign.  Across the five resets the safe bound is
the coherent sum in (N.217), unless (N.224), the block SDP, or an endpoint
cross is proved.

### Exact finite-observation condition for Price or Stein providers

Let `X_j` be the complete-old-residualized local source space at reset `j`,
let `O_j:X_j -> R^(q_j)` be the finite Price/Stein observations actually
computed, and let the frozen endpoint map be

\[
 \mathcal H(x_1,\ldots,x_5)=\sum_{j=1}^5P_jx_j,
 \qquad
 O=\operatorname{diag}(O_1,\ldots,O_5).
 \tag{N.228}
\]

The finite provider determines the complete endpoint contribution exactly
if and only if

\[
 \boxed{
 \ker O\subseteq\ker\mathcal H
 \quad\Longleftrightarrow\quad
 \mathcal H=A O\ \text{for some }A.}
 \tag{N.229}
\]

For one fixed scalar endpoint covector `h`, the weaker necessary-and-
sufficient condition is

\[
 \boxed{\mathcal H^*h\in\operatorname{ran}O^*.}
 \tag{N.230}
\]

These are the precise target-conditioned escapes from the generic infinite
Price/Stein hierarchy.  The centered Gaussian q211 rooted slice has an
infinite-rank coefficient Hankel matrix, so no fixed finite edge/Price/Stein
dictionary satisfies (N.229) uniformly on an open covariance set.  That
generic no-go does not close (N.230) for one frozen endpoint or finite panel;
it makes the residualized target tail the binding evidence.

Price root-tail, orientation-transition, broad-diagram, quadrature, and
true-minus-Gaussian-rest errors must first be summed into the actual `z_j`
at each reset.  If only their separate norms are known, the within-reset
bound is itself coherent,

\[
 \sqrt{d_j}\le
 a_{j,root}+a_{j,trans}+a_{j,broad}
 +a_{j,quad}+a_{j,nonG},
 \tag{N.231}
\]

before the across-reset elliptope is applied.  Increasing finite Price degree
can reduce the root tail but cannot silently reduce the independent broad or
non-Gaussian terms.

### Sharper final-only theorem for a jointly refit finite dictionary

There is one important way to avoid both the coherent amplitude gate and the
five causal crosses.  Freeze the r62 states, transitions, ownership projector,
and every candidate endpoint column.  Let `V_infty` be the closure of the old
columns plus the complete Price/Stein endpoint dictionary, and let
`V_Q subseteq V_infty` contain the old columns plus the chosen finite
dictionary.  If *both* models are jointly least-squares refit in the same
weighted corpus metric, write their orthogonal projectors as `P_infty,P_Q`.
Then

\[
 r_Q=(I-P_Q)y
 =\underbrace{(I-P_\infty)y}_{r_\infty}
 +\underbrace{(P_\infty-P_Q)y}_{t_Q},
 \qquad
 r_\infty\perp t_Q.
 \tag{N.232}
\]

Consequently

\[
 \boxed{
 m_Q=m_\infty+\|t_Q\|_H^2,
 \qquad
 m_Q<10^{-8}
 \Longleftrightarrow
 \|t_Q\|_H^2<10^{-8}-m_\infty.}
 \tag{N.233}
\]

This is the sharp final-only target-tail gate.  It is an energy allowance,
not the much smaller adversarial amplitude allowance.

For a finite omitted block `X_R`, residualize it against the retained design:

\[
 Z_R=(I-P_Q)X_R,
 \qquad
 G_R=Z_R^*WZ_R,
 \qquad
 b_R=Z_R^*Wr_Q.
 \tag{N.234}
\]

If `X_R` spans the remaining full dictionary modulo `V_Q`, the tail in
(N.233) is exactly

\[
 \boxed{
 \|t_Q\|_H^2
 =b_R^*G_R^\dagger b_R.}
 \tag{N.235}
\]

For an infinite dictionary use the monotone limit over finite omitted
blocks.  Source energy, unweighted Hermite tail, or kernel approximation
error is not a substitute for this complete-old-residualized target Gram.

The Pythagorean theorem fails if truncating the provider changes the r62
states, section, reset maps, ownership projection, or candidate columns; if
the physical coefficients are frozen rather than jointly refit; or if
`V_Q` is not literally nested in the full comparator.  In those cases the
exact final formula is instead

\[
 m_Q=m_\star+D_Q+2c_Q,
 \qquad
 D_Q=\|r_Q-r_\star\|_H^2,
 \qquad
 c_Q=\langle r_\star,r_Q-r_\star\rangle_H,
 \tag{N.236}
\]

and unknown orientation gives the sharp robust condition

\[
 \boxed{
 \sqrt{D_Q}<\sqrt{10^{-8}}-\sqrt{m_\star}.}
 \tag{N.237}
\]

For context, if a lawful ideal route had raw `m_star=9.2547e-9`, (N.237)
allows only about `3.80e-6` RMS endpoint error, or `1.44e-11` error energy,
whereas the nested-refit theorem (N.233) allows the full energy difference
`7.453e-10`.  For the quoted near-exact oracle interval
`m_star in [7.09,9.89]e-9`, the adversarial amplitude allowance ranges from
about `1.58e-5` down to `5.52e-7`, while the nested energy allowance ranges
from `2.91e-9` down to `1.10e-10`.  These figures are conditional
illustrations, not value credit for r62: its section and full-provider raw
have not been established.

### Acquisition economics under the r62 headroom

The `chi_j` accumulator is scalar only **after** the endpoint contribution
or reverse Bellman test has been acquired.  A new dense covariance reverse
over five transitions costs

\[
 C_{reverse}=2\times5=10>s_{62,5}=1.56705.
 \tag{N.238}
\]

It therefore cannot be added as a new deployed state to the r62 shell.
Causal certification must use a literal resident/low-rank Bellman view or be
performed off-card.  The hybrid final-endpoint/polarization route is also an
off-card validation certificate; it does not make an unavailable local
source deployable.

The remaining headroom gives useful optimistic finite-Price rank screens.
For complete-old-projected CP ranks `R_j`, the zero-bias trace action costs

\[
 C_{trace}^{sym}={\sum_jR_j\over1024},
 \qquad
 C_{trace}^{gen}={2\sum_jR_j\over1024},
 \tag{N.239}
\]

whereas the full biased root/path CP fallback costs

\[
 C_{biased}^{sym}={2\sum_jR_j\over1024},
 \qquad
 C_{biased}^{gen}={3\sum_jR_j\over1024}.
 \tag{N.240}
\]

If the rank is constant over five resets, (N.204) permits at most

\[
 \boxed{
 R\le320\ \text{(symmetric trace)},\quad
 R\le160\ \text{(generic trace or symmetric biased)},\quad
 R\le106\ \text{(generic biased)}.}
 \tag{N.241}
\]

These are necessary optimistic caps before seam acquisition, hooks, local
core arithmetic, writes, stabilization, and error control.  They compute a
Price source action once its CP query is resident; they do not pay for the
missing reverse Bellman query in (N.238).  A Price response already contained
in a billed full convolved AQ surface has zero additive value and must be
removed before applying either the rank or endpoint-tail gate.

### Depth-first fail-fast order

1. **Section first.**  Prove the r62 barred cubic map onto all 561 forced
   normal directions, its conditioning, and a literal oriented initializer.
   Otherwise `c_0` is uncontrolled and none of the rest gates promotes the
   arithmetic shell.
2. **Ownership second.**  Project every finite Price/Stein source against the
   complete old plus full-AQ surface.  Separate the non-Gaussian/dependent
   residual; covariance does not own it.
3. **Fixed-column final route whenever available.**  If the candidate is a
   fixed nested linear dictionary, use (N.233)--(N.235).  This is the sharpest
   and cheapest final-only theorem.
4. **Otherwise acquire geometry.**  Use one aggregate final cross for a fixed
   output, five `chi_j` for a reusable causal prefix certificate, or the block
   SDP.  Local scalar error bars alone receive the sharp coherent/polygon
   range (N.217), not RSS.
5. **Charge acquisition.**  A new dense reverse is cost-closed by (N.238).
   Any surviving finite Price/Stein branch must fit (N.241) or a stronger
   literal reuse theorem and pass the complete target tail.

### Atomic graph-ready nodes

#### `R62_FIVE_RESET_HYBRID_TELESCOPE`

- label: Exact nonlinear telescope for the r62 five-reset suffix
- kind: theorem
- status: solved
- priority: critical
- summary: Any fixed reference/proposed pair has the exact six-block endpoint
  decomposition (N.205)--(N.206); no tangent premise is needed.
- terminal_criterion: Closed mathematically; application must exhibit the
  actual reference and five hybrid endpoint differences.
- evidence: (N.205)--(N.210).
- decomposition_children: `P2R0G19B_MATH_AQ_R62_CUBIC_FRONTIER`,
  `R62_FIVE_RESET_MINIMAL_CAUSAL_CHI`,
  `R62_FIVE_RESET_SCALAR_ELLIPTOPE`.
- goal_path_summary: Supplies the only lawful error decomposition for testing
  the five-reset arithmetic shell against raw below `1e-8`.

#### `R62_FIVE_RESET_MINIMAL_CAUSAL_CHI`

- label: One Bellman cross per reset is the minimal causal scalar state
- kind: theorem
- status: solved_theorem_acquisition_open
- priority: critical
- summary: (N.209) is exact; the full Cauchy interval (N.211) proves one new
  scalar per injection is necessary and sufficient.
- terminal_criterion: Pass after all five `chi_j` are lawfully acquired or a
  theorem forces them to their certified values.
- evidence: (N.208)--(N.212).
- decomposition_children: `R62_GREST_MARTINGALE_ORTHOGONALITY`,
  `R62_BELLMAN_ACQUISITION_BILL`.
- goal_path_summary: Prevents unproved RSS from turning five small local rest
  errors into a false endpoint pass.

#### `R62_FIVE_RESET_FINAL_ONLY_CROSS`

- label: One aggregate cross is exact for one fixed final r62 output
- kind: theorem
- status: solved_theorem_acquisition_open
- priority: high
- summary: `Chi_5` in (N.213)--(N.215), or the final norm itself, eliminates
  the need for five separate causal crosses when no prefix guarantee is used.
- terminal_criterion: Acquire the exact aggregate endpoint vector/cross in the
  same corpus metric; do not use this node as a source-construction receipt.
- evidence: (N.213)--(N.215).
- decomposition_children: `R62_FIVE_RESET_HYBRID_TELESCOPE`.
- goal_path_summary: Gives the fastest exact validation route for a compiled
  deterministic five-reset estimator.

#### `R62_FIVE_RESET_SCALAR_ELLIPTOPE`

- label: Sharp six-block polygon and block-elliptope pass/close gate
- kind: bound
- status: solved
- priority: critical
- summary: Scalar energies give exactly (N.217); directional blocks give the
  exact SDP (N.222).  The upper is a robust pass and the lower a robust close.
- terminal_criterion: Compare the normalized upper/lower values to
  `tau_10=0.340032` or the declared stricter target.
- evidence: (N.216)--(N.222), including the equal-reset thresholds (N.220).
- decomposition_children: `R62_FIVE_RESET_FINAL_ONLY_CROSS`.
- goal_path_summary: Turns partial local error receipts into sharp conclusions
  without inventing cross-layer independence.

#### `R62_GREST_MARTINGALE_ORTHOGONALITY`

- label: Frozen Gaussian rest gives RSS only under a conditional-centering law
- kind: iff_theorem
- status: conditional_open
- priority: critical
- summary: Past-measurable future pullback plus
  `E[epsilon_j|F_(j-1)]=0` forces `chi_j=0`; analytic integrated Price/heat
  truncation is a bias and fails that premise generically.
- terminal_criterion: Exhibit sample-level fresh rest, conditional centering,
  and frozen pullback measurability at all five resets, or keep the crosses.
- evidence: (N.223)--(N.225).
- decomposition_children: `R62_FIVE_RESET_MINIMAL_CAUSAL_CHI`,
  `R62_FINITE_STEIN_LOCAL_TAIL`.
- goal_path_summary: Defines exactly when the fivefold RSS allowance in
  (N.220) is lawful.

#### `R62_FINITE_STEIN_LOCAL_TAIL`

- label: Explicit scalar-core finite Stein endpoint-amplitude tail
- kind: bound
- status: solved_conditional
- priority: medium
- summary: Under `||A||op<1` and a uniform Bellman derivative norm, the
  degree-Q tail is (N.227); mixed cores still need a common algebra/rank proof.
- terminal_criterion: Insert all five certified amplitudes into the coherent,
  chi, or elliptope gate; the bound alone does not imply RSS.
- evidence: (N.226)--(N.227).
- decomposition_children: `R62_GREST_MARTINGALE_ORTHOGONALITY`,
  `R62_FIVE_RESET_SCALAR_ELLIPTOPE`.
- goal_path_summary: Converts a finite Stein hierarchy into a lawful local
  error input for the raw-MSE gate.

#### `R62_FINITE_PRICE_STEIN_FACTOR_GATE`

- label: Exact factor-through-observations condition for finite providers
- kind: iff_theorem
- status: solved_theorem_target_receipt_open
- priority: high
- summary: Full endpoint exactness is `ker O subset ker H`; one target needs
  only `H^*h in ran O^*`.  Generic Gaussian q211 has no universal finite
  dictionary, but a frozen target-specific factor remains possible.
- terminal_criterion: Prove (N.229) or (N.230) after complete-old projection,
  or carry the target-weighted omitted tail through (N.231).
- evidence: (N.228)--(N.231) plus the infinite-Hankel Gaussian q211 witness.
- decomposition_children: `R62_FINITE_PRICE_TARGET_TAIL`,
  `R62_PRICE_HEADROOM_CAP`.
- goal_path_summary: Separates universal finite-word impossibility from the
  only useful target-conditioned reopen.

#### `R62_FINITE_PRICE_TARGET_TAIL`

- label: Pythagorean final-only tail for a fixed jointly refit dictionary
- kind: theorem
- status: solved_theorem_application_open
- priority: critical
- summary: For literal nested fixed endpoint spaces, finite raw equals full
  raw plus the Schur-complement target tail (N.232)--(N.235); otherwise use
  the cross/amplitude gate (N.236)--(N.237).
- terminal_criterion: Establish fixed nested columns and joint refit, then
  require the tail below `1e-8-m_infty`; if any outer state changes, acquire
  the cross instead.
- evidence: (N.232)--(N.237).
- decomposition_children: `R62_FINITE_PRICE_STEIN_FACTOR_GATE`,
  `R62_FIVE_RESET_FINAL_ONLY_CROSS`.
- goal_path_summary: This is the sharpest fail-fast value test for finite
  Price/Stein columns on one frozen corpus.

#### `R62_BELLMAN_ACQUISITION_BILL`

- label: A new five-transition dense Bellman reverse does not fit r62
- kind: cost_no_go
- status: closed_for_new_dense_reverse
- priority: critical
- summary: The shell leaves 1.56705 equivalents while a fresh reverse costs
  10, (N.238); scalar contraction is not scalar acquisition.
- terminal_criterion: Reopen only with a literal resident view, certified
  low-rank reverse inside the margin, or off-card-only validation.
- evidence: (N.204), (N.238).
- decomposition_children: `R62_FIVE_RESET_MINIMAL_CAUSAL_CHI`.
- goal_path_summary: Prevents a mathematically exact Bellman gate from hiding
  an impossible deployed reverse bill.

#### `R62_PRICE_HEADROOM_CAP`

- label: Optimistic finite-Price CP ranks under the 1.56705 margin
- kind: cost_bound
- status: solved_necessary_caps
- priority: high
- summary: Constant five-reset ranks are capped at 320 symmetric trace, 160
  generic trace/symmetric biased, or 106 generic biased before all extras.
- terminal_criterion: Close any candidate above (N.241); below it still needs
  query residency, source ownership, hooks, and the target-tail gate.
- evidence: (N.239)--(N.241).
- decomposition_children: `R62_FINITE_PRICE_STEIN_FACTOR_GATE`,
  `R62_FINITE_PRICE_TARGET_TAIL`.
- goal_path_summary: Couples finite-provider rank to the actual r62 arithmetic
  shell without crediting it as Bellman acquisition.

Current next question: can the 561-mode r62 cubic section be constructed with
a uniformly conditioned literal initializer at the five-reset seam, and, on
that fixed state, does the complete-old-residualized finite Price/Stein target
Gram satisfy (N.233) inside one of the necessary caps (N.241)?  Until the
section exists, no `e_0=0` or oracle raw may be assigned to this shell.

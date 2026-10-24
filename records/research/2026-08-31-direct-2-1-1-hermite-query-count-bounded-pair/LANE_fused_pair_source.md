# Fused pair source lane

Theory-only ledger. No numerical experiment is authorized by this file.

## 1. Wiener/Q24 completeness

For a local field `X_i = I1(a_i) + I3(f_i)`, the entire degree-at-most-four pair correction is

`3 I2(a_i tensor_1 f_j + a_j tensor_1 f_i) + I4(a_i tensor f_j + a_j tensor f_i) + 18 I2(f_i tensor_2 f_j) + 9 I4(f_i tensor_1 f_j) + 6 <f_i,f_j>`.

The signed terms are the existing CP3/star arm. The scalar, H2, and H4 cubic-pair contractions are the existing fish/G11/Q24 arm. At this order there is no additional O(H)-equivariant treewidth-at-most-two topology. The `I6` term requires Q6 and lies outside Q24. A connected C4 provider is distinct from the reducible `9 I4(f tensor_1 g)` term.

The maximal consumer for a genuinely new cubic provider `h` is

`Delta S_ij = L(a_i,h_j)+L(a_j,h_i)+B(f_i,h_j)+B(h_i,f_j)+B(h_i,h_j)`,

where

`L(a,h)=3<a tensor_1 h,Q2>+<a tensor h,Q4>`

and

`B(f,g)=6 q0<f,g>+18<f tensor_2 g,Q2>+9<f tensor_1 g,Q4>`.

The first two lines are the signed first-order gate; `B(h,h)` is second order and cannot rescue a zero tangent under fail-fast discipline.

## 2. Edge-colored completeness

With fixed Wick propagator `P_ab` and pair masks `A,B`, rank-one cubic atoms reduce to only

- `P_ab^3`;
- `P_ab^2 A_ab`;
- `P_ab (A_aa B_bb + B_aa A_bb + 4 A_ab B_ab)/6`.

The last item is the fixed dumbbell-plus-theta polarization of a genuine symmetric Q4 tensor. Its pieces are not independently tunable. Off-diagonal atom pairs are already included by a dense G11. New internal colors require typed components of a genuinely new provider, not a relabeling of Wick edges.

## 3. NCQC and canonical q4(211)

For `Q_M=(z^T M z-tr M)/2`, with `A=M_i`, `B=M_j`, and `C=M_k`,

`kappa4(Q_A,Q_A,Q_B,Q_C)=2 tr(A^2 B C)+tr(A B A C)`.

Writing `t=tr(A^2BC)` and `s=tr(ABAC)`,

`s-t = (1/2) tr([A,B][A,C])`,

so the physical cumulant is

`2t+s = 3t + (1/2)tr([A,B][A,C])`.

In an orthonormal skew basis, the genuinely noncommutative term is the negative Gram `-(1/2) sum_alpha C^alpha_AB C^alpha_AC`. If the old span contains the Jordan/path tangent `t`, canonical and commutator q211 become the same quotient task direction. They must not be counted as an additive portfolio.

## 4. Canonical latent trajectory quotient

Let `V=span{A_i}` in latent matrix space, let `P_V` be its Frobenius projector, and define

`Y_ab=A_a circle A_b=(A_a A_b+A_b A_a)/2`,

`Z_ab=(I-P_V)Y_ab`,

`Gperp_ab,cd=<Z_ab,Z_cd>`.

This Schur Gram is the single quartic provider object. Its two typed consumers are

`qperp_abcd=16(Gperp_ab,cd+Gperp_ac,bd+Gperp_ad,bc)`

and

`Kcommperp_ab,cd=4(Gperp_ac,bd-Gperp_ad,bc)`.

The identity behind the split is

`<[A_a,A_b],[A_c,A_d]>=4(<Y_ac,Y_bd>-<Y_ad,Y_bc>)`.

Orthogonality of `P_V` gives an exact parallel term obtained from `P_V Y` and the displayed residual term obtained from `Z`. The parallel term is already determined by the old H3/Jordan coordinates. The commutator residual is a contrast readout of the same `Gperp`, not a second provider.

For external `(i,i,j,k)`, set `u=Gperp_ii,jk` and `v=Gperp_ij,ik`. Then

`qperp_iijk=16(u+2v)`

and

`ncommperp_i;jk=2(v-u)`.

Latent trajectory projection, external collision projection, the dense bridge, and endpoint old-span residualization are four different linear maps and must be applied in that order.

## 5. Joint endpoint Schur and cost gate

After external projection, bridge, and endpoint old-span residualization, write the base endpoint vectors as `x` for `u` and `y` for `v`. Define unscaled consumers

`c=x+2y` and `n=y-x`.

Let `a=<x,x>`, `b=<x,y>`, `d=<y,y>`, `p=<r,x>`, and `q=<r,y>`. Then

`s_cc=a+4b+4d`,

`s_nn=a-2b+d`,

`s_cn=-a-b+2d`,

`h_c=p+2q`, and `h_n=q-p`.

The best joint gain is `h^T S^+ h`. When `ad-b^2>0`, it is exactly

`(d p^2-2 b p q+a q^2)/(ad-b^2)`.

The Gram determinant is `9(ad-b^2)`. One readout structurally subsumes the other exactly when `ad-b^2=0`; the second is merely target-redundant when its residualized score after the first is zero.

Choose an orthonormal basis of the residual provider span and write its external matrices as `Z_alpha`, `alpha=1,...,r_perp`, where `r_perp=rank(Gperp)` after any exact external restriction. For a dense suffix adjoint `B`,

`U_i(B)=sum_alpha (Z_alpha)_ii <Z_alpha,B>`

and

`V_i(B)=sum_alpha (Z_alpha B Z_alpha^T)_ii`.

The U contraction costs `O(r_perp n^2)`. The V contraction needs one dense matmul per Kraus component when only the diagonal is required, and two per component for a full plane unless one multiplication rides an already billed conjugation. The Choi matrix of this map is `Gperp`, so its minimal exact Kraus count is generically `r_perp`; a block GEMM does not reduce billed matmul equivalents.

For a ten-transition suffix with a ten-matmul marginal budget, the exact diagonal-consumer cost gate is

`sum_l r_eff,l <= 10`.

Generic residual rank greater than one per transition closes deployment cost. A rank-one residual at each of ten transitions is the boundary case. Low-rank suffix adjoints can replace `n` by their rank in the arithmetic count, but must be physics-fixed rather than fitted compression.

## Current next question

Determine the exact effective Kraus ranks after external `(211)` restriction and the fixed endpoint bridge. If their ten-transition sum exceeds ten, close the provider on cost without a numerical MSE run. If it passes, evaluate the already-banked endpoint Schur ceiling once and require more than 0.2773 of residualized quartic energy before any implementation experiment.

## Checkpoint 4 refinement

For the physically normalized consumers `c=16(x+2y)` and `n=2(y-x)`, the consumer transform from the base `(x,y)` coordinates is

`T=[[16,32],[-2,2]]`.

Thus the exact endpoint Schur matrix is `S=T [[a,b],[b,d]] T^T`, namely

`S_cc=256(a+4b+4d)`,

`S_nn=4(a-2b+d)`,

`S_cn=32(-a-b+2d)`.

The score vector is `(16(p+2q), 2(q-p))`. Since `det(T)=96`, `det(S)=9216(ad-b^2)`. The two consumers are structurally redundant exactly when `ad-b^2=0`; the second is only target-redundant when its Schur-residual score is zero.

For rank-one effective quotient, the cheapest exact diagonal contraction is one GEMM `M=ZB`, followed by `V=diag(MZ^T)` as rowwise dot products and `U=diag(Z)<Z,B>`. This is one matmul-equivalent per transition and produces both consumers. Collision exclusions are lower-order corrections. For effective quotient rank `r`, the corresponding completely positive map has Choi/Kraus rank `r`, so generic exact cost is `r` matmul-equivalents per transition for a diagonal endpoint, or `2r` for a full plane unless the second multiplication fuses into an existing conjugation.

Conditional dimension count: if the deployed latent cores are generic elements of `Sym_45`, then the ambient dimension is 1035 and 1024 trajectory cores leave an 11-dimensional orthogonal complement. If the Jordan products hit that complement generically, `r_perp=11` per layer and the ten-transition suffix costs 110 diagonal matmul-equivalents, which is closed by the budget. Only an exact bridge-induced reduction to `sum_l r_eff,l<=10` can reopen it.

## Checkpoint 5: sharp rank-versus-value gate

For layer `l`, choose an orthonormal quotient basis `Z_l,alpha` and let the external/bridge/endpoint task metric induce the weighted synthesis operator

`R_l c = H_l^(1/2) (sum_alpha c_alpha Z_l,alpha) B_l^(1/2)`.

Its Gram is

`Xi_l[alpha,beta]=tr(B_l Z_l,alpha^T H_l Z_l,beta)`.

The squared singular values of `R_l` are the eigenvalues `lambda_l,1>=...>=0` of `Xi_l`. The effective exact Kraus rank is `rank(Xi_l)`. By Eckart-Young/Ky Fan, the best rank-`k_l` bridge restriction retains exactly `sum_{j<=k_l} lambda_l,j` of weighted quartic response energy. Across layers, the best total rank budget `K` retains the `K` largest eigenvalues in the union of all layer spectra.

The ten-transition reopen therefore requires both

`sum_l rank(Xi_l)<=10`

and

`sum_l sum_j lambda_l,j >= 0.27732 E4_residual`

for an exact structural restriction. If approximate rank truncation is authorized, its absolute upper bound is the sum of the ten largest eigenvalues across all layers; if that is below `0.27732 E4_residual`, it is a proof-level NO-GO.

A rank-one collapse at layer `l` occurs iff all weighted quotient matrices `H_l^(1/2) Z_l,alpha B_l^(1/2)` are collinear, equivalently iff every `2x2` minor of `Xi_l` vanishes. Concentration rather than exact collapse requires `lambda_l,1/tr(Xi_l)` to be large; rank one retains at least 27.732 percent at a single layer iff this ratio is at least `0.27732` relative to the relevant residual-energy denominator.

K31 diagonal extraction does not force this condition. With a diagonal mask of broad nonzero support, restriction of a generic 11-dimensional quotient into an `n=1024` dimensional diagonal response remains injective, hence rank 11 generically. It collapses only if all restricted row-response vectors are collinear or the mask annihilates ten quotient directions.

A scalar backward value output also does not imply a rank-one Kraus computation. The scalar action is `sum_alpha tr(B Z_alpha^T H Z_alpha)` and generically needs every Kraus component. If `H` and `B` are full rank, their weighting preserves linear independence and `rank(Xi)=r_perp`. A true analytic collapse requires low-support metrics or a symmetry making all `H^(1/2) Z_alpha B^(1/2)` collinear. Rank-one `H=hh^T` and `B=bb^T` is sufficient and reduces the action to `sum_alpha (h^T Z_alpha b)^2`, computable in `O(r_perp n^2)`; the dense backward value adjoint does not have this property generically.

## Current next question after checkpoint 5

Prove or disprove, from the exact K31 bridge and backward-adjoint formulas, that their induced weighted Gram `Xi_l` has rank one on any L5--14 transition. Without such an identity, the generic rank-11 cost closure remains binding; task-value concentration alone is not an exact billing reopen.

## Checkpoint 6: cross-layer sharing does not beat the local rank gate

Let `A_l` be the linear suffix transition on matrix state, `S_l` the quotient-provider synthesis at layer `l`, and `C_t` the complete required endpoint/panel consumer suite. Across cut `m`, define the block Hankel operator

`H_m = [ C_t Phi_(t<-l) S_l ]_(t>m,l<=m)`.

Its rank `h_m` is the minimal number of observable state modes that any exact realization must carry through cut `m`. A compatible factorization uses state bases `U_m` with `A_m U_m = U_(m+1) F_m` modulo the full task-unobservable kernel and expresses each new source in `U_m` modulo that kernel. Under the one-marginal-GEMM convention, exact propagation cost is at least `sum_m h_m`; double it for an unfused two-sided conjugation.

A single shared mode exists iff every `H_m` has rank at most one with compatible factorizations. Equivalently, there are matrices `U_m` such that every observable quotient source at layer `m` is proportional to `U_m` and `A_m(U_m)` is proportional to `U_(m+1)`. A common fixed mode would further require `W_m U W_m^T` proportional to `U` at every layer. Generic dense, unrelated suffix weights and generic quotient subspaces do not satisfy these polynomial rank-one identities.

Most importantly, `h_m` is at least the effective rank `r_eff,m` of the current layer's observable source block, because that block is a column suboperator of `H_m`. Hence

`sum_m h_m >= sum_m r_eff,m`.

Cross-layer propagation can avoid additional persistence work but can never beat the already-derived local reverse-contraction rank gate. It meets the ten-GEMM budget only if the local ranks already sum to at most ten. Thus it cannot rescue generic `r_eff=11` layers.

If each new 11-dimensional layer quotient is generically independent of transported earlier quotients, then after `k` layers `h_k=11k` until state dimension saturation, and ten cuts cost `11(1+...+10)=605` marginal transitions. The local backward evaluation at 110 equivalents is strictly cheaper. Even exact alignment into one common 11-dimensional invariant bundle costs `11*10=110`. Only a one-dimensional invariant bundle reaches 10, and that is precisely the per-layer rank-one collapse already required by checkpoint 5.

The joint endpoint Gram must include cross-layer blocks `F_l^*F_k`; aligned layer values are combined coherently and cannot also be counted as separate gains. A scalar final adjoint gives a rank-one output but does not reduce block Kraus/Hankel ranks when the full frozen panel/task suite is used.

## Current next question after checkpoint 6

No cross-layer sharing loophole remains. The only reopen is still an explicit layerwise bridge identity giving `sum_l r_eff,l<=10` plus at least 27.732 percent residual quartic value. Otherwise close the canonical quartic quotient on deployment cost while retaining its algebraic source theorem.

## Checkpoint 7: tensor-free K31 scalar action

For centered affine-quadratic Gaussian laws

`U=l^T z + (z^T M z-tr M)/2`

and

`V=v^T z + (z^T N z-tr N)/2`,

the exact mixed fourth cumulant is

`kappa4(U,U,U,V)=3 tr(M^3 N)+3 l^T(MN+NM)l+6 l^T M^2 v`.

Since the scalar transpose identity gives `l^T MN l=l^T NM l`, this can be written

`3<M^3,N>+6<sym((Ml)l^T),N>+6<M^2l,v>`.

For `V=U`, it reduces to

`kappa4(U)=3||M^2||_F^2+12||Ml||^2`,

which requires only one dense latent squaring per unit. Thus a literal diagonal self-kurtosis endpoint genuinely bypasses materializing the 11-Kraus quartic plane; this does not contradict the Kraus lower bound because it evaluates one fixed linear functional, not the full operator.

For all pairs `(i,j)`, define `F_i=3M_i^3+6 sym((M_i l_i)l_i^T)` and `b_i=M_i^2l_i`. Then the full K31 matrix is

`K_ij=<F_i,M_j>+6<b_i,l_j>`.

After per-unit powers, this is two rectangular Gram products with arithmetic `O(n^2 r^2+n^2 r)` rather than 11 width conjugations. If the endpoint contracts the singleton `V_j` leg with a single weight vector `w_j`, multilinearity permits the preaggregation `Nbar=sum_j w_j N_j`, `vbar=sum_j w_j v_j`; each `U_i` then needs only the displayed scalar formula against `(Nbar,vbar)`.

The one irreducible crossed fourth-action term is `3 tr(M^3N)`. The affine terms are vector/matrix Grams. For a general carried law, the corresponding irreducible datum is `E[(U-EU)^3(V-EV)]`; the exact identity is

`kappa4(U,U,U,V)=E[Uc^3 Vc]-3 E[Uc^2]E[Uc Vc]`.

Means, covariance, marginal k3/k4, and lower action moments do not identify this crossed moment. A formal perturbation by `H3(U)H1(V)` changes it while leaving covariance and marginals fixed. Hence a general tensor-free formula merely renames the missing fourth-action moment unless the carried family is affine-quadratic or explicitly stores that action.

If singleton weights depend densely on `i`, preaggregation becomes `Nbar_i=sum_j W_ij N_j`; forming all `Nbar_i` costs `O(n^2r^2)`, and the crossed trace block remains the expensive term. More sharply, let `d_sym=r(r+1)/2` and pack symmetric matrices with the off-diagonal square-root-two weights. The exact dense factorization is

`K=Fpack Mpack^T+6 b l^T`,

so it needs two per-unit latent products to form `M_i^3`, followed by one `n by d_sym` Gram and one `n by r` Gram. Its minimal classical arithmetic is `O(2nr^3+n^2(d_sym+r))`, not an intrinsically quartic tensor contraction. In width-matmul equivalents this is

`2r^3/n^2+(d_sym+r)/n` per layer,

about `1.23` at `r=45,n=1024`, or about `12.3` over ten layers if every layer materializes the dense plane. The earlier unpacked `r^2/n` count is a conservative implementation count. A diagonal self-kurtosis instead needs one latent square, `r^3/n^2=0.087` per layer and `0.87` over ten.

For a single backward adjoint `w`, do not materialize `K`: compute `K w=Fpack(Mpack^T w)+6b(l^T w)`. Equivalently use `(Nbar,vbar)`. After the per-unit powers this costs only `O(n(d_sym+r))`; including the two latent products it costs about `2r^3/n^2=0.174` width equivalents per layer, or `1.74` over ten. For a rank-`q` row-dependent bridge `W=A B^T`, precompute the same `F,b` once and evaluate `q` aggregates at `O(qn(d_sym+r))`. Thus the exact bypass gate is endpoint bridge rank, not the superficial fact that the requested output is scalar. Rank one reopens; generic rank `n` recovers the dense cost.

The state census is also distinct: diagonal or one-adjoint evaluation streams with one `r by r` temporary plus the existing `(M_i,l_i)` cores and an `n`-scalar output, so it transports no quartic plane. Reusable arbitrary-adjoint evaluation stores `n(d_sym+r)` feature coordinates; materializing all pair responses additionally stores `n^2` scalars.

## Current next question after checkpoint 7

Keep four decisions separate in the graph: (1) the local 11-Kraus provider lower bound remains proved for an arbitrary quartic operator; (2) scalar-at-birth bypass is open only for an exact affine-quadratic law and literal diagonal or fixed singleton adjoint; (3) the endpoint rank-factorization gate is `rank(W)` (rank one is budget-safe, generic dense is not); and (4) novelty is decided only after bridge, endpoint, and old-span residualization. A cheap response entirely in `q_parallel` is still valueless.

Determine whether the actual K31 endpoint is literally diagonal or has a single factorable singleton-leg adjoint, and whether the suffix law stays exactly affine-quadratic. If yes, the scalar action is a genuine low-cost consumer and should be graded by its analytic 27.732-percent residual ceiling. If it requires an `i`-dependent dense singleton bridge or a general carried law, the crossed fourth-action moment restores the cost/information closure.

## Checkpoint 8: deployability theorem for the actual reverse K31 mask

The deployed reverse identity fixes the local mask. With symmetric reverse covariance plane `H=W_(l+1) B_(l+1) W_(l+1)^T`, and `G13=G31^T`,

`Q31=(H hadamard G31+(H hadamard G13)^T)/6=(H hadamard G31)/3`.

The subsequent coordinate pullback is `P=W_l Q31^T`, and the external fourth-tensor mask is

`Xi_ijkl=sum_a W_ia W_ja W_ka P_la`.

Equivalently, in the post-`W_l` coordinates `Y_a`,

`chi_l=<Q31,D31>=sum_a kappa4(Y_a,Y_a,Y_a,V_a)`,

where `V_a=sum_b Q31_ab Y_b`. This is one scalar endpoint but it contains one row-dependent singleton aggregate for every `a`.

### Exact endpoint-rank theorem

Let `q_l=rank(Q31_l)`. If `H=UV^T` and `G31=ST^T`, the Hadamard/Khatri-Rao identity gives

`Q31=(U odot_KR S)(V odot_KR T)^T/3`,

so `q_l<=min(n,rank(H)rank(G31))`. Rank one holds if and only if every two-by-two minor vanishes:

`H_ab G_ab H_cd G_cd=H_ad G_ad H_cb G_cb`

for all rows `a,c` and columns `b,d`. Neither the covariance adjoint recurrence nor the correlated-activation masks impose these identities. Full rank is generic: `det(H hadamard G31)` is a polynomial that is not identically zero, as witnessed for example by `H=G31=I`. Hence the Zariski-open/default rank is `q_l=n`; a scalar terminal and one reverse covariance stream do not imply a rank-one singleton bridge.

This also settles the residualization issue. Collision/Q24/CP3 subtraction is source-side:

`chi_l^new=<Q31_l,D31_l-D31_l^old>`.

It leaves `Q31_l` exactly unchanged. The later task old-span operation `(I-P_old) chi` is applied to the already accumulated scalar response, not to the `(a,b)` coordinates. If it is pulled back into a linear combination of task-specific masks, a generic linear combination of full-rank masks remains full rank; rank-one would again require all displayed minors to vanish. Therefore old-span residualization supplies no algebraic rank collapse.

### Exact affine-quadratic acquisition and cost

In the half-quadratic convention, let each post-mix coordinate be

`Y_a=u_a^T z+(z^T A_a z-tr A_a)/2`.

Pack symmetric matrices isometrically into `d_sym=r(r+1)/2` coordinates and define

`F_a=3A_a^3+6 sym((A_a u_a)u_a^T)`, `b_a=A_a^2u_a`.

Then

`D31_ab=<F_a,A_b>+6<b_a,u_b>`

and the complete reverse action is

`chi_l=tr(Fpack^T Q31 Apack)+6 tr(b^T Q31 u)`.

No K31 plane need be materialized. If `Q31=L R^T` has rank `q_l`, evaluate this with `q_l` singleton aggregates at arithmetic

`O(2nr^3+q_l n(d_sym+r))`.

In width-`n` matmul equivalents the per-layer marginal is

`c_l=2r^3/n^2+q_l(d_sym+r)/n^2`,

assuming the post-`W` affine-quadratic cores and `Q31` are already present and the reverse `H` conjugation is reused. For `r=45,n=1024`:

- analytic rank one gives `10 c_l about 1.75` over ten layers;
- the actual structurally generic `q_l=n` mask gives `10 c_l about 12.285`.

Thus the certified deployability census must use the approximately `12.3` full-dense-mask path, not the approximately `1.74` aggregate path. Existing `A_a^2`/Q24 reuse can reduce the latent-power term, but cannot remove the `Q31 Apack` rectangular multiply. If the post-`W` cores are not already carried, their forward mixing is an additional `(d_sym+r)/n` equivalent per layer.

This is the incremental scalar-consumer census. Algebraically, for pre-`W` cores `Mpack`, `Npack=W^T Mpack`, `P=W Q31^T`, and `Rpack=P^T Mpack=Q31 Npack`. Therefore forming `P` is unnecessary for the scalar once `Npack` is carried. A joint-refit/source-gradient implementation that explicitly needs `P` adds one full width GEMM per active layer; a lineage that does not already carry `Npack` adds its forward rectangular mix; and a lineage that does not reuse the reverse covariance conjugation `H` must charge that separately. Hence `12.3` is a proved lower incremental path under named reuse premises, not a complete unqualified package bill.

The streamed state is the existing `Q31` mask and affine-quadratic cores, transient `F,b,Q31 A,Q31 u`, and one scalar accumulator. It transports no quartic plane. The dense cost is an acquisition cost, not a q4-state cost.

### Beyond-affine-quadratic premise

For arbitrary centered layer variables the same mask has the universal action

`chi_l=sum_ab Q31_ab {E[Y_a^3Y_b]-3E[Y_a^2]E[Y_aY_b]}`.

The covariance subtraction is available at `O(n^2)`. The crossed action moment

`sum_ab Q31_ab E[Y_a^3Y_b]=E[(Y^odot3)^T Q31 Y]`

is not determined by means, covariance, or marginal third/fourth moments. Exact AQ closure in one synchronized Gaussian core proves the trace formula; a true ReLU/higher-chaos law requires either an engine-provided projected fourth-action oracle, an explicit higher-chaos state, or a certified beyond-AQ residual bound. Without one of these premises, `12.3` is only the cost of the AQ surrogate, not the cost of an exact K31 action.

## Separate graph nodes after checkpoint 8

### `P2R0G19B_K31_PROVIDER_LOWER_BOUND`

- **Status:** `closed_proof` for the generic transported quartic operator.
- **Claim:** an arbitrary observable `q_perp` provider has effective Kraus rank eleven per generic layer; a scalar endpoint does not change that forward-provider theorem.
- **Reopen only if:** the action is contracted at birth through a separately proved sufficient local statistic.

### `P2R0G19B_K31_SCALAR_AT_BIRTH`

- **Status:** `solved_algebra`, deployability conditional.
- **Claim:** `sum_a kappa4(Y_a,Y_a,Y_a,V_a)` exactly equals the complete reverse K31 action and needs no transported quartic tensor.
- **Terminal falsifier:** the desired endpoint derivative contains a K22/other quartic covector outside the K31 action row space.

### `P2R0G19B_K31_MASK_RANK`

- **Status:** `closed_proof` against the rank-one assumption; actual numerical rank remains a receipt.
- **Claim:** `Q31=(H hadamard G31)/3`; rank one is equivalent to vanishing of every two-by-two minor and is not implied by the deployed reverse recursion. Generic rank is `n`, and source/old-span subtraction does not lower it.
- **Deploy cost:** use `q_l=n`, approximately `12.285` equivalents over ten layers. Promote the `1.75` census only with an analytic rank-one identity, not a fitted low-rank observation.

### `P2R0G19B_K31_AQ_LAW_GATE`

- **Status:** `conditional_open`.
- **Claim:** the packed trace/Gram acquisition is exact if every active layer law is affine-quadratic in one synchronized Gaussian core and the primal/dual cores use the same gauge.
- **Terminal falsifier:** two admissible states with the same carried AQ summaries but distinct Bellman-weighted K31 action, or an uncertified beyond-H2 component larger than the endpoint error budget.

### `P2R0G19B_K31_BEYOND_AQ_ACTION`

- **Status:** `closed_proof` for lower-moment identifiability; oracle availability open.
- **Claim:** the sole missing general-law datum is the dense crossed moment `E[(Y^odot3)^T Q31 Y]`; lower moments do not determine it.
- **Promotion criterion:** a lawful engine action oracle or a certified residual bound for all non-AQ chaos.

### `P2R0G19B_K31_OLDSPAN_NOVELTY`

- **Status:** `open_value_gate`.
- **Claim:** after exact collision ownership and scalar task residualization, only `(I-P_old)chi` is novel. Cheap or exact acquisition is not evidence of new value.
- **Promotion criterion:** its residual Schur gain is at least `2.86e-10`, equivalently at least `27.732%` of the quoted quartic value under the stated projection metric.

## Critical ownership correction: only true-minus-assembled K31 is novel

The complete same-site K31 action is already owned by the D11 CP4/star/pair assembly. Let `Dold_l` be that assembled connected plane and `Dtrue_l` the actual same-site connected plane. With the exact reverse mask `Q_l`,

`chi_old_l=<Q_l,Dold_l>`,

`chi_true_l=<Q_l,Dtrue_l>`,

and the only candidate novelty is

`rho31_l=<Q_l,Dtrue_l-Dold_l>`.

The H31/H13 terms defining `Q_l` are consumers, not another source. The disconnected cumulant pairing is already removed inside `Dold_l`; it must not be subtracted again. The D11 off-diagonal K31 convention and separate diagonal K4 ownership must also be respected before differencing.

### Subtraction-rank theorem

Source subtraction leaves the physical mask exactly unchanged. If `E_l=Dtrue_l-Dold_l`, then acquisition is `rho31_l=<Q_l,E_l>` with the same

`Q_l=(H_l hadamard G31_l)/3`.

Consequently `rank(Q_l)` and its generic full-rank endpoint aggregation cost do not fall merely because `||E_l||` or the final scalar is small. More generally, let `S_true,S_old` be the provider-to-plane maps. The residual provider rank is the separation rank of `S_res=S_true-S_old` after the full bridge and old-interface quotient. It is smaller only if an exact common-factor cancellation lowers a matricization rank of `S_res`. There is no inequality from small value to small rank: for every `epsilon>0`, `S_res=epsilon C` can have full generic rank while arbitrarily small norm.

Thus subtracting an already available scalar `chi_old` is `O(1)`, but acquiring `chi_true` or `rho31` remains full-mask work unless the residual itself has a proved low-rank action oracle. The approximately `12.285` AQ census therefore describes a dense synchronized AQ action computation; it is not automatically the cost of the novel residual. The approximately `1.748` path remains invalid absent an analytic rank-one residual bridge.

### AQ ownership trichotomy

At a perfectly synchronized site/gauge, if the D11 assembled plane is exactly the affine-quadratic plane generated by the same carried cores, then

`chi_AQ_l=chi_old_l`

and the AQ trace/Gram action has identically zero novel Schur gain. If an AQ decoder disagrees with `chi_old`, decompose

`rho31_l=[chi_true_l-chi_AQ_l]+[chi_AQ_l-chi_old_l]`.

The first bracket is the beyond-AQ crossed action. The second is an AQ-versus-D11 synchronization/closure defect; it may be tested as a replacement or refit correction but is not evidence for a new quartic sector. Only the first bracket answers the missing-physics question.

For arbitrary centered variables the first bracket contains

`E[(Y^odot3)^T Q_l Y]-AQ_trace_action`

after the common covariance counterterm cancels. Lower moments do not identify it. Therefore recomputing the AQ scalar cannot acquire the genuinely novel residual; a lawful projected fourth-action oracle, a richer exact law, or a certified tail bound is required.

## Full-interface future-observability theorem after D11 subtraction

Let `H_l` now denote the Hilbert space of residual fourth/higher-chaos perturbations at site `l` after the assembled D11 source is subtracted. Let

`O_l:H_l -> Y_l`

be the complete existing interface, including CP4 carried coordinates and K4/K31/K22 star/pair readouts, and put `N_l=ker(O_l)`. Let `R_l:H_l->H_(l+1)` be the linearized dense-push plus ReLU/reset map and `Phi_(m:l)=R_(m-1)...R_l`. Define the stacked future observability map on the currently dark space by

`F_l h=(O_(l+1)Phi_(l+1:l)h,...,O_m Phi_(m:l)h,...,terminal actions)` for `h in N_l`.

Then the exact minimal residual quotient is

`Qobs_l=N_l/(N_l intersect ker(F_l))`,

and its dual/minimal cyclic module is

`C_l=range(P_(N_l) F_l^*)`

`=span_{m>l}{P_(N_l) Phi_(m:l)^* range(O_m^*)}`

with the terminal endpoint covectors included. Its dimension satisfies

`dim(C_l)=rank(F_l restricted to N_l)`

and is the minimum number of independent linear residual coordinates any exact realization must retain at cut `l` to reproduce every future old-interface output. This follows from the factorization theorem: an added observation `A_l` is sufficient if and only if

`ker([O_l;A_l]) subseteq ker(F_l)`,

equivalently

`C_l subseteq range(P_(N_l) A_l^*)`.

For one fixed future scalar covector `lambda_m`, the entire condition reduces to

`c_(l,m)=P_(N_l) Phi_(m:l)^* lambda_m`.

There is no miss exactly when `c_(l,m)=0`; otherwise the information-minimal added observable for that one scalar is `<c_(l,m),h_l>`. For a family of layers, nets, or endpoint heads, the required count is the dimension of their span, not one merely because each output is scalar.

Equivalently, the existing interface is future closed exactly when its dark bundle is output-nulling invariant:

`Phi_(m:l) N_l subseteq ker(O_m)` for every future `m`.

A local sufficient recurrence is `R_l N_l subseteq N_(l+1)` at every cut together with terminal darkness. Generic dense mixing plus ReLU reset does not preserve this bundle.

### Why same-site scalar K31 cannot certify the future kernel

The D11 same-site K31 row already lies in `range(O_l^*)`, so its projection onto `N_l` is zero. Adding it again leaves `C_l` completely untouched. A genuinely residual scalar-at-birth oracle `a_l` captures the future value only if every required `c_(l,m)` is collinear with `P_(N_l)a_l^*`; one oracle cannot cover a higher-dimensional cyclic module.

The smallest abstract counterexample has two state coordinates:

`O_0=O_1=[1,0]`, `N_0=span(e2)`, and `R=[[1,1],[0,1]]`.

The perturbation `h=e2` has `O_0h=0`, and the duplicate same-site scalar also returns zero, but `O_1Rh=1`. With future covector `lambda=e1`,

`P_(N_0)R^*lambda=e2 !=0`.

Dimension two is minimal because a nonzero one-dimensional interface has trivial kernel.

The smallest collision-sector realization uses three external coordinates. The symmetric quartic with polynomial `x1^2 x2 x3` is a `(211)` direction and has zero K4 `(4)`, K31 `(31)`, and K22 `(22)` coordinate readouts at birth. Whenever its component surviving the CP4 carrier lies in `N_l`, a dense next-layer column `w=(1,1,1)/sqrt(3)` gives `h(w,w,w,w)!=0`, so the next diagonal K4 readout sees it. A ReLU reset with nonzero fourth-Hermite response preserves this first-order visibility. For two external coordinates, `(4),(31),(22)` exhaust `Sym^4`, so three is the smallest collision example.

### Sharp certificate and cost gate

For added residual observations `A_l`, the exact worst-case future miss on a radius-`r_l` residual ball is

`eta_l=r_l ||(I-P_range(P_N A_l^*)) P_N F_l^*||_op`.

The bound is sharp: equality is attained by a residual aligned with the leading missed singular vector. For one endpoint covector replace `F_l^*` by `Phi^*lambda`. In a source covariance metric, use the corresponding whitened projector; the same kernel/range criterion is unchanged.

Let `k_t=dim(C_t)` be the exact residual state count at each suffix cut. Under one dense transition per carried mode, every exact realization costs at least

`G_future >= C_acquire + sum_t k_t`

matmul equivalents; double the transition term for an unfused two-sided conjugation. Under the current D14+D10 base cost `48` and cap `49`, the binding gate is

`C_acquire+sum_t k_t<=1`.

Thus any nonzero residual mode needing transport through two or more cuts is a deployment NO-GO for that stack. The sole possible reopen is a direct, lawful birth contraction against the complete future costate, fused for total incremental cost at most one, with no unobserved cyclic remainder. It must additionally produce an old-span-residualized Schur gain of at least `2.86e-10`; the already-owned AQ K31 scalar has gain exactly zero by construction.

## Corrected graph-node split

### `P2R0G19B_K31_OLD_ACTION_DUPLICATE`

- **Status:** `closed_proof`.
- **Claim:** the same-site complete K31 scalar is owned exactly once by the D11 CP4/star/pair assembly; H31/H13 are consumers.
- **Consequence:** the synchronized AQ trace action has zero new Schur gain.

### `P2R0G19B_K31_RESIDUAL_ACQUISITION`

- **Status:** `conditional_open`.
- **Claim:** only `<Q,Dtrue-Dold>` can count. Source subtraction leaves the generic full-rank reverse mask and acquisition cost intact unless the residual map itself has a proved lower separation rank.
- **Promotion:** lawful direct residual-action oracle with the same site/off-diagonal convention and total incremental cost within the remaining cap.

### `P2R0G19B_K31_SYNC_DEFECT`

- **Status:** `open_replacement_only`.
- **Claim:** `chi_AQ-chi_old` is a gauge/site/closure synchronization defect, not new q4 physics; grade only as a matched replacement/refit correction.

### `P2R0G19B_K31_RESIDUAL_AQ_GATE`

- **Status:** `closed_proof` for insufficiency of the synchronized AQ decoder.
- **Claim:** synchronized AQ reconstructs the old action, while the novel residual is beyond-AQ and not identified by lower moments.
- **Reopen:** exact richer-law decoder or projected fourth-action oracle with certified tail error.

### `P2R0G19B_K31_FUTURE_OBSERVABILITY`

- **Status:** `solved_theorem`, frozen-interface certificate open.
- **Claim:** the minimal missing module is `C_l=range(P_ker(O_l)F_l^*)`; a same-site duplicate K31 scalar has zero projection into it.
- **Promotion:** kernel/range inclusion plus `C_acquire+sum k_t<=1` and residual Schur gain at least `2.86e-10`.

## Checkpoint 9: local-transient Z22 future completion

The generic future-module no-go has one exact exception: contract a residual source against its complete pulled-back Bellman context while that source already exists transiently. The D11 reset constructs precisely such a 22 residual before discarding it.

After `_quartic_residual` has subtracted the active CP4 22 plane, `_reset_stars` forms

`K22sym=(k22+k22^T)/2`,

`P22=offdiag(v outer v+2 C hadamard C)`,

`alpha=<K22sym,P22>/<P22,P22>`.

Both planes use the D11 off-diagonal convention. Therefore the exact residual outside CP4 plus the fitted pairing coordinate is

`Z22=K22sym-alpha P22`.

The function does not return `Z22`; it next returns the K31/K4 coordinate star and `alpha`, so this 22 residual is otherwise discarded.

### Fixed-context rank theorem

For one pulled-back Bellman 3--1 context `(u,v)`, put

`x=u hadamard u`, `y=u hadamard v`.

The exact first future K31 action of `Z22` is

`a_(u,v)(Z22)=3 x^T Z22 y`.

As a map from the residual source space to task value, this has exact rank one whenever its residual mask is nonzero. As an ordered left/right bilinear test it also has separation rank one. Because `Z22` is symmetric, the equivalent symmetric mask is

`B_(u,v)=(3/2)(x y^T+y x^T)`,

whose ordinary matrix rank is generically two, rank one exactly when nonzero `x,y` are collinear, and zero when its projection onto the lawful residual plane space vanishes. Equivalently it is a signed two-Kraus polarization, but it still needs only one application `Z22 y`.

For `m` fixed contexts, with `X=[x_f]` and `Y=[y_f]`,

`a(Z22)=3 tr(X^T Z22 Y)`.

The ordered separation rank is `rank(XY^T)<=m`, while the symmetric mask rank is at most `2m`. The task output remains one accumulated scalar. Thus the enormous generic dimension of the 22 kernel is irrelevant after the exact endpoint context is fixed: the task-visible quotient is one scalar, not a transported 22 plane.

For the **actual compiled D10 Bellman context**, however, `m=n`, one context for every width column. Writing

`X=W hadamard W`, `Y=W hadamard P`, `P=W Q31^T`,

the exact action is

`a_l(Z22)=3<Z22 X,Y>_F=3<Z22,Y X^T>_F`.

Its effective ordered provider rank is

`q_l=rank(Y X^T)`

before quotienting source-invisible masks. The exact lawful quotient rank is

`qeff_l=min_{S in U22+span(P22), D diagonal} rank(sym(Y X^T)-S-D)`.

This is the minimal signed symmetric-Kraus count of the fixed mask on the zero-diagonal residual `Z22` space. For generic dense `W,P`, `sym(YX^T)` is a generic symmetric matrix: choose any dense `W` for which `X=W hadamard W` is invertible, then `P=(B X^(-T)) divided_elementwise W` realizes arbitrary `Y X^T=B`, and an appropriate dense `Q31` realizes `P=W Q31^T`.

The equivalence space has dimension at most `r4+1+n`. The symmetric rank-at-most-`q` variety has codimension

`(n-q)(n-q+1)/2`.

Therefore, under generic transversality, no quotient representative of rank at most `q` exists whenever this codimension exceeds `r4+1+n`. At `n=1024,r4=2048`, the equivalence dimension is at most `3073`; since `78*79/2=3081>3073`,

`qeff_l>=947`

generically. Even the strongest generic use of CP4, pairing, and diagonal invisibility can remove at most about 77 mask ranks, nowhere near the average rank `<=102` required by the cap. A scalar final output therefore does not make the actual acquisition rank one. The preceding rank-one statement applies to one individual `(u,v)` pair, not to the full deployed Bellman sum.

The symbolic zero gate is

`Pi_Z sym(XY^T)=0`,

where `Pi_Z` projects away the CP4-22 and pairing spans. Nonzero `Z22` alone does not prove a particular Bellman context sees it.

### Contract-before-discard theorem

If `(u_f,v_f)` are available at `_reset_stars`, the exact action can be inserted immediately after `alpha` is computed:

`3 sum_f x_f^T (K22sym-alpha P22) y_f`.

No persistent plane and no future H4 mode are required. It may be evaluated without forming a new array by separately contracting `K22sym` and `P22`, or by overwriting a dead scratch plane with `Z22`. CP4 ownership is already correct because `_quartic_residual` ran first; pairing ownership is removed by the displayed `alpha` term; diagonal K4 is untouched because all operands are off diagonal.

This is exactly the direct-complete-costate exception to the generic cyclic-module bound. It is conditional on context timing: a forward-only pass does not know a future Bellman context. Deployment needs either a compiled analytic context, a baseline reverse pass followed by reset recomputation, or an already carried D10 context. Storing `Z22` until a later reverse pass forfeits the transient exception.

### Exact marginal cost gate

Stacking `m_l` contexts at layer `l` evaluates the action by one `n by n` times `n by m_l` rectangular product plus `O(n^2+m_ln)` reductions. With an exact rank-`q_l` factorization of the fixed mask, the same action costs `q_l/n` width-GEMM equivalents. Relative to a full width GEMM, the leading incremental cost is

`C_Z=sum_l q_l/n+C_context+C_hook`,

Here `C_context` is any cost not already owned by the D10 reverse context and `C_hook=O(L/n)` accounts for forming/subtracting the transient residual and scalar reductions. Under base `48` and cap `49`, the exact resource gate is

`sum_l q_l/n+C_context+C_hook<=1`.

For the actual D10 context, generic `q_l=n`, so `Z22 X` costs one full GEMM per layer and ten active layers cost `10` equivalents even when `P` is already available. If `P=W Q31^T` is not already owned by D10, its construction adds another full GEMM per layer. Under the base-48 stack this is a deployment NO-GO. The exact reopen is

`sum_l q_l<=n`

after all lower-order/context costs, i.e. average rank at most about `102` over ten layers, or an exact reuse of `Z22 X`/the dense mask from an already billed intermediate. Merely observing that the final action is scalar or that `Z22` is transient does not supply this factorization.

### Schur/value gate

Let

`z_Z=sum_l 3 sum_f x_lf^T Z22_l y_lf`

over the corpus, and let `P_old` be the final D11/native-feature response projector. Only

`z_Zperp=(I-P_old)z_Z`

is new. Its exact one-feature Schur gain is

`Delta_Z=<R,z_Zperp>^2/<z_Zperp,z_Zperp>`.

Promotion requires all of:

1. the fixed-context mask projection is nonzero at the source level;
2. the full `n`-context Bellman mask is lawful and available before the transient is discarded;
3. `C_Z<=1` under the base-48 stack, requiring `sum q_l<=n` or an exact billed-intermediate reuse;
4. `Delta_Z>=2.86e-10`, or the correspondingly reduced remainder when stacked with another independently residualized quartic source.

The analytic proof that generic `Z22` is nonzero is not a Schur certificate. Frobenius orthogonality to CP4/pairing at each net also does not imply task-response orthogonality to the old estimator; the final response-space projection is binding.

### Generic module versus transient exception

- A residual direction that must persist through two or more cuts remains closed by `C_acquire+sum dim(C_t)>1`.
- `Z22` is different in **state**, because D11 already materializes the exact residual ingredients locally and the complete future covector reduces them to one scalar before the cut. For the actual full-rank D10 context it is not different enough in arithmetic: the local contraction still costs one GEMM per layer.
- The exception covers only the 22 completion. Generic 211 and 1111 sources remain outside this plane and retain their acquisition/module obstruction unless they obtain their own direct endpoint-action identities.

## Removing the CP4 caveat from the future-visible 211/1111 witness

Let

`Vdark=H211 direct_sum H1111`.

Every current coordinate K4 `(4)`, K31 `(31)`, and K22 `(22)` readout vanishes identically on `Vdark`. Its dimension is

`Ddark=n choose-repeated-211 + choose-all-distinct`

`=n*C(n-1,2)+C(n,4)`.

Even under the conservative assumption that the complete symmetric CP4 factor interface of rank cap `r4` can impose `n r4` independent linear constraints through all factor coordinates, rank-nullity gives

`dim(Vdark intersect ker(CP4))>=Ddark-n r4`.

At `n=1024,r4=2048`, `dim(H211)=535299072`, `dim(H1111)=45545029376`, and `n r4=2097152`, leaving at least `46078231296` nonzero dark directions. Hence a CP4-kernel direction exists without any generic-position assumption.

The exact sufficient width condition under this conservative tangent-rank count is

`n*C(n-1,2)+C(n,4)>n r4`,

equivalently

`(n-1)(n-2)(n+9)/24>r4`.

For `r4=2048`, the smallest guaranteed width is `n=35` (`2057>2048`; at `n=34` the left side is `1892`). The 211 sector alone suffices from `n=66`, and the 1111 sector alone from `n=39`. If the CP4 interface observes only its `r4` represented tensor amplitudes rather than the full factor tangent, the weaker condition `Ddark>r4` suffices; the `n r4` bound above is deliberately safer.

Choose any nonzero `h` in this intersection. The homogeneous polynomial

`p_h(w)=h[w,w,w,w]`

is not identically zero, because diagonal evaluation determines a symmetric tensor by polarization. Its zero locus is therefore a proper algebraic set. A generic dense next-layer column `w` satisfies `p_h(w)!=0`, and the future diagonal K4 coordinate of `W^(tensor4)h` exposes the direction in one step. A post-mix ReLU interface preserves this witness whenever its fourth-Hermite/Edgeworth response coefficient and Bellman K4 covector are nonzero.

The D11 coordinate star and pairing state do not change this count: by the ownership audit they factor entirely through the `(4),(31),(22)` collision restrictions, which already vanish on `Vdark`. If any future lineage gives those states an additional all-distinct linear readout of rank `r_extra`, replace the right-hand side by `n r4+r_extra`; that is the exact missing-premise adjustment.

## Checkpoint-9 graph-node split

### `P2R0G19B_Z22_TRANSIENT_IDENTITY`

- **Status:** `closed_proof`.
- **Claim:** `_reset_stars` transiently determines `Z22=K22sym-alpha P22` after CP4 subtraction; a fixed context reads it as `3(u^2)^T Z22(u v)` before discard.

### `P2R0G19B_Z22_CONTEXT_RANK`

- **Status:** `closed_proof`.
- **Claim:** one individual pair has ordered rank one, but the actual D10 bank has `m=n`. Its lawful quotient rank is the minimum rank of `sym((W hadamard P)(W hadamard W)^T)` modulo CP4-22, pairing, and diagonal masks; generically it is at least `947` and ordinarily full rank at width1024/rank2048.

### `P2R0G19B_Z22_CONTEXT_TIMING`

- **Status:** `conditional_open`.
- **Claim:** the exception is deployable only if the exact future Bellman contexts are compiled or available during reset; storing Z22 closes the cost advantage.

### `P2R0G19B_Z22_COST_GATE`

- **Status:** `closed_cost_generic`, conditional reopen.
- **Gate:** `sum_l qeff_l/n+C_context+C_hook<=1`. The generic quotient lower bound alone costs at least `9470/1024 about 9.25` over ten layers and fails; reopen only with the non-generic certificate `sum qeff_l<=n` or exact reuse of the required dense product from an already billed intermediate.

### `P2R0G19B_Z22_SCHUR_GATE`

- **Status:** `open_value`.
- **Gate:** final old-span-residualized gain at least `2.86e-10`; analytic nonzero and source Frobenius orthogonality are insufficient.

### `P2R0G19B_FUTURE_DARK_DIMENSION`

- **Status:** `closed_proof` under the documented collision-only star/pair interface.
- **Claim:** at width1024/rank2048, at least `46078231296` 211/1111 directions evade current CP4 plus all collision reads, and a generic dense column exposes any fixed nonzero one in future K4.

## Checkpoint 10: reset-local post-AQ H3 CountSketch audit

The candidate source is not the affine-quadratic cubic jet and not the incumbent CP3/H31 source. At site `l` it is the signed residual

`Z_l=Ztrue_l-Zassembled_l`,

with each unit row in `H3=Sym^3(R^r)`. The current `r=55` ambient coordinate count is

`p=C(r+2,3)=C(57,3)=29260`.

Use orthonormal symmetric coordinates: entries of multiplicity `(3),(21),(111)` receive weights `sqrt(1),sqrt(3),sqrt(6)`, so Euclidean products equal the Wiener/Frobenius kernel product.

### Exact reset-local bilinear form

Let `GZ_l,GY_l` be the two typed H3 adjoints in the post-mix K31 tangent. Pulling both dense maps back to the reset-local source gives

`Gamma_l=W_l(GZ_l+Q_l^T GY_l)`.

The exact first-order residual action is

`f_l=<Z_l,Gamma_l>_F`.

Thus, after conditioning on one frozen complete Bellman adjoint, the source-to-task map is one scalar covector. This does **not** mean a one-coordinate oblivious sketch is exact for a family of units, nets, and endpoint masks.

Let `X` and `Y` be the old-span-residualized source and query subspaces in `H3`, with orthonormal bases `U_X,U_Y`. Their exact conditional interaction/provider rank is

`r_cross=rank(U_X^T U_Y)`.

For a common linear sketch `S:H3->R^k`, uniform exactness is

`U_X^T(S^T S-I)U_Y=0`.

Since the sketched cross Gram has rank at most `k`, necessarily `k>=r_cross`. For approximate uniform action the exact certificate is

`||U_X^T(S^T S-I)U_Y||_op<=epsilon`.

Moreover every rank-`k` sketch obeys the deterministic lower bound

`||U_X^T S^T S U_Y-U_X^T U_Y||_op>=sigma_(k+1)(U_X^T U_Y)`.

This is the binding conditional-rank gate. Embedding `span(X union Y)` is sufficient but stronger than necessary. A single matched pair may use a pair-specific variance certificate, but that is not a uniform replacement theorem.

### Exact CountSketch collision error

Let `h:[p]->[k]` be two-wise independent and the signs four-wise independent. With

`(Sx)_b=sum_(j:h(j)=b) sign_j x_j`,

`E <Sx,Sy>=<x,y>`

and the exact variance is

`Var(<Sx,Sy>)=(1/k) sum_(a!=b) [x_a^2 y_b^2+x_a y_b x_b y_a]`

`=(1/k)[||x||^2||y||^2+<x,y>^2-2 sum_a x_a^2 y_a^2]`

`<=2||x||^2||y||^2/k`.

For the full per-example matrix action, define

`A_e=sum_(l,i) z_(eli) gamma_(eli)^T`.

Then `f_e=tr(A_e)` and the common-sketch estimator is `fhat_e=tr(S^T S A_e)`, with

`Var(fhat_e)=(1/k) sum_(a!=b)[A_(e,ab)^2+A_(e,ab)A_(e,ba)]`

`<=2||offdiag(A_e)||_F^2/k`.

The sum over layers occurs inside `A_e` when one seed is reused, so coherent cross-layer errors must not be replaced by a sum of independent layer variances. Corpus squared error does sum over examples. Define

`C_cross=sum_e ||offdiag(A_e)||_F^2/||f||^2`.

Then the pair-specific mean-square relative error is at most `2 C_cross/k`. Under the earlier `0.1601` sketch-noise fraction this requires

`k>=2 C_cross/0.1601 about 12.49 C_cross`.

At the base-48 arithmetic cap below, even the transport-only maximum `k=51` would require `C_cross<=4.08`. Unbiasedness over random seeds is not a certificate for the one preregistered deployed seed; promotion needs either the deterministic cross-embedding test or a stated high-probability repetition/error allocation.

### Algebraic sufficiency of the reset-local frame

The CountSketch action is lawful only if all of the following hold:

1. `Ztrue-Zassembled` is generated from white-box state at the exact reset site, with CP3/H31 and D11 ownership removed once;
2. `Gamma_l` is the complete baseline Bellman pullback to that same site, including both `GZ` and `GY` paths;
3. the sketch cross error, source projection/quadrature error, and finite-`Z` nonlinear remainder are charged;
4. the correction is accumulated as a first-order replacement action and does not silently alter the frozen reset masks; and
5. final response-space old-span residualization is applied.

Conditioned on these premises, no H3 tensor need persist across a reset: form `SZ_l` and `S Gamma_l`, take their rowwise Frobenius product, and discard both.

### Generic nonlinear-reset no-go

CountSketch commutes exactly with dense linear unit mixing because it acts on latent tensor coordinates. It does not close a subsequent nonlinear ReLU reset. If `R_g` is the linearized H3 reset operator, an exact recurrence from `SZ` exists iff there is `A_g` with

`S R_g=A_g S`,

equivalently

`R_g ker(S) subseteq ker(S)`.

For `k<p`, a generic CountSketch kernel is not invariant under even one generic dense reset, much less every unit-specific `R_g`. Hence a forward multi-reset sketched H3 carrier is generically invalid. Only generate-and-contract at the reset with an already-backpropagated complete adjoint avoids this no-go; otherwise the quotient leakage must be explicitly carried and bounded.

### Source and query arithmetic at `r=55`

No current D14/D10 field supplies the post-AQ source. The proposed exact local source is the Fourier/boundary reset kernel `T(a,M)/6`, so it is new provider work.

For `n=1024,p=29260`, one explicit source or query coordinate/hash pass costs

`p/n^2=29260/1048576 about 0.02790`

width-GEMM equivalents per active layer. A finite `J`-node ordinary CountSketch projection of the Fourier formula costs at least `Jp/n^2` for coordinate generation/hashing, plus resolvent work. With dense per-unit eigensystems/solves, a transparent census is

`g_F >= r^3/n^2+J(r^3+p)/n^2`

unless a separately proved TensorSketch/Krylov contraction avoids the dense reconstructions. A generic cross-unit source such as `diag(C M C^T)` instead costs at least one full GEMM per layer and is immediately incompatible with the additive base-48 stack.

The complete pullback `Gamma=W(GZ+Q^T GY)` requires two `n by n` times `n by k` products, so its incremental dense cost is `2k/n` per active layer. Neither product is currently materialized by D14/D10.

For `L_s` active sites, the exact additive gate is

`G_CS=L_s[2k/n+c_query p/n^2+g_source]+G_hook<=1`.

At `L_s=10`:

- sketch transport alone gives `k<=floor(1024/20)=51`;
- one transient query coordinate/hash pass leaves at most `k=36`;
- one comparable explicit source pass as well leaves at most `k=22`;
- any new one-GEMM source at each layer closes the additive lane before sketch error is considered.

The older `r=32,k=464,L=14` memory/cost numbers belong to a different hypothetical architecture and do not transfer to the current `r=55`, base-48 package.

### Paid-buffer sharing audit

The candidate can reuse the numerical **inputs** `W,Q/H` and synchronized AQ cores, and an `n by k` transient may potentially alias a dead `n by n` D14/D10 scratch buffer after an exact lifetime audit. These are memory reuses only.

There is no proved arithmetic reuse:

- paid D14/D10 GEMMs have different right-hand sides from `SGZ,SGY`;
- appending `k` sketch columns increases billed work by exactly `k/n` for each dense mix;
- D11 does not generate `T(a,M)/6` or its CountSketch;
- incumbent CP3/H31 subtraction may reuse its already assembled scalar/source convention, but it does not produce the true residual sketch; and
- local resolvent, coordinate hashing, and sketch collision reductions are new operations.

An arithmetic-reuse claim requires literal equality with an existing billed RHS/intermediate, not common weights or common storage. Replacing D10 or D14 is a separate portfolio: it frees cost only after charging the value lost and proving the sketch subsumes the removed response.

### Endpoint/value gate

Let `f_CS` be the accumulated sketched residual action and

`z_CS=(I-P_old)f_CS`.

Its exact marginal gain is

`Delta_CS=<R,z_CS>^2/<z_CS,z_CS>`.

Unbiasedness, nonzero source norm, and cross-embedding are capacity/fidelity statements, not evidence of target alignment. Promotion requires simultaneously:

1. the current portfolio-specific `Delta_CS` crossing requirement (at least `2.86e-10` if credited as the missing quartic/K31 marginal, or the stronger joint H3 residual requirement if it replaces D14 value);
2. total implementation amplitude error, including sketch, Fourier/quadrature, higher-order finite-`Z`, and mask drift, inside the raw certificate (`<=3.80e-6` for the quoted `9.2547e-9` ideal); and
3. the additive cost gate above, or a fully costed replacement inequality including lost D14/D10 value.

No numerical screen is justified until the cross-rank/error bound and source-cost/reuse premise leave a nonempty `k` interval.

## Checkpoint-10 graph-node split

### `P2R0G19B_H3_CS_ALGEBRAIC_SUFFICIENCY`

- **Status:** `conditional_theorem`.
- **Claim:** a reset-local true-minus-assembled H3 source contracts exactly with the complete pulled-back adjoint; a common sketch is sufficient only under the typed site/ownership and cross-embedding conditions.

### `P2R0G19B_H3_CS_CONDITIONAL_RANK`

- **Status:** `open_instance_rank`.
- **Claim:** exact uniform sketch dimension is constrained by `r_cross=rank(U_X^T U_Y)` and approximate error by its singular tail; current additive cost permits at most `k=51` even before source/query work.

### `P2R0G19B_H3_CS_COLLISION_ERROR`

- **Status:** `solved_formula`, frozen-seed certificate open.
- **Claim:** CountSketch is unbiased with the exact variance above; same-seed cross-layer errors combine coherently, and the deterministic cross-embedding test is binding for replacement guarantees.

### `P2R0G19B_H3_CS_RESET_NOGO`

- **Status:** `closed_proof` for generic forward carriage.
- **Claim:** `R_g ker(S) subset ker(S)` fails generically for `k<p`; only reset-local contraction avoids nonlinear leakage.

### `P2R0G19B_H3_CS_SOURCE_COST`

- **Status:** `closed_cost_generic`, exceptional reuse/structured projection open.
- **Claim:** no source is currently exposed; Fourier/boundary projection and both sketch-adjoint mixes are incremental. Under base48, feasible `k` falls from `51` to `36` with one query pass and `22` with one source pass, before quadrature/resolvent.

### `P2R0G19B_H3_CS_BUFFER_REUSE`

- **Status:** `memory_reuse_possible`, arithmetic reuse unproved.
- **Claim:** scratch may alias, but no paid D14/D10 RHS equals the sketch products; common weights do not reduce billing.

### `P2R0G19B_H3_CS_ENDPOINT_VALUE`

- **Status:** `open_value_gate`.
- **Claim:** only final old-span Schur gain counts, with all sketch/source/reset errors inside the raw-MSE certificate and any replacement charged for removed value.

## Checkpoint 10B: exact D14+D10 reuse theorem for reset-local H3

This section distinguishes three different claims which must not be merged:

1. the billed streams contain enough information to determine the H3 action;
2. the required factors are literal zero-GEMM combinations/views of resident arrays; and
3. a compressed CountSketch approximates the action accurately.

The first is a quotient theorem, the second is a schedule theorem, and the
third is an approximation theorem.  Neither common weights nor common scratch
storage implies either of the first two.

### Canonical source--adjoint cross operator

Let

`Htot=direct_sum_(l in D10) [R^n tensor Sym^3(R^55)]`

with the corpus/endpoint weights and final old-feature residualization already
absorbed into its inner product.  Let

`Z:X->Htot`, `G:Y->Htot`

be the linearized true-minus-assembled source and complete frozen-Bellman
query maps.  On their nonnull supports put

`GX=Z^* Z`, `GY=G^* G`,

`UZ=Z GX^(dagger/2)`, `UG=G GY^(dagger/2)`.

Thus `UZ` and `UG` are partial isometries from whitened source and query
coefficient spaces.  The exact endpoint-conditioned cross operator is

`C_x=UZ^* UG=GX^(dagger/2) Z^* G GY^(dagger/2)`.

Its singular values are the source--query principal cosines and

`k_x=rank(C_x)`

is the exact bilinear separation rank.  Indeed, if

`C_x=L Sigma R^*`,

then the action is the inner product of the two `k_x`-channel factors

`s=Sigma^(1/2)L^* xhat`,

`q=Sigma^(1/2)R^* yhat`.

Conversely every representation as an inner product of `k` source/query
channels factors `C_x` through a rank-`k` middle space, so `k>=k_x`.  This
proves that `k_x`, not the scalar dimension of one frozen endpoint, is the
family closure rank.  A common latent sketch `S` is exact only if

`UZ^*(I tensor S^*S)UG=C_x`;

its dimension is necessarily at least `k_x`.  The SVD factors show that this
rank obstruction is tight for arbitrary asymmetric, endpoint-tuned linear
factors, but not automatically for a one-sparse common CountSketch.

### Necessary and sufficient information-reuse theorem

Work in the whitened source/query coordinates above.  Let

`OF:X->F`, `OR:Y->R`

record **only arrays literally materialized at the required site and lifetime**
by the paid D14 forward bundle and D10 reverse bundle.  In particular, an
input weight, an object that could be recomputed, and an aspirational D14
state do not belong to `OF` or `OR`.  Put

`PF=OF^dagger OF`, `PR=OR^dagger OR`,

the orthogonal projectors onto the two billed row spaces.

There is an exact implementation of the complete cross action using only the
two billed observations and fixed linear maps if and only if

`C_x=PF C_x PR`.                                                    `(10B.1)`

Equivalently,

`ker(OF) subset ker(C_x^*)`,

`ker(OR) subset ker(C_x)`.                                         `(10B.2)`

Necessity is immediate: a source perturbation invisible to D14, or a query
perturbation invisible to D10, must then have zero action against every
admissible perturbation on the other side.  For sufficiency define

`D0=(OF^dagger)^* C_x OR^dagger`.

Equations (10B.1)--(10B.2) give

`C_x=OF^* D0 OR`.

Moreover `rank(D0)=rank(C_x)=k_x` after deleting null observed directions.
An SVD of `D0` therefore produces exactly `k_x` paired factors, each a fixed
linear combination of the already billed observations.  This is the weakest
action-level reuse theorem: it does not require reconstruction of the full
H3 source or query.

For the stronger claim that the canonical SVD factors themselves are
recoverable, let

`FS=Sigma^(1/2)L^*`, `FQ=Sigma^(1/2)R^*`.

Then fixed factor maps `AS,AQ` with

`FS=AS OF`, `FQ=AQ OR`

exist if and only if

`ker(OF) subset ker(FS)`, `ker(OR) subset ker(FQ)`;

equivalently `ran(L) subset ran(OF^*)` and
`ran(R) subset ran(OR^*)`.  This is just the finite-dimensional
factor-through/quotient theorem and is an exact symbolic test.

### Exact fresh-source and fresh-query ranks

If (10B.1) fails, define the two dark ranks

`k_src=rank((I-PF)C_x)`,

`k_qry=rank(C_x(I-PR))`.                                           `(10B.3)`

These are the exact minimum numbers of new source-side and query-side linear
observations required to augment the billed catalogs.  For example, every
augmentation `AF` must make `ran(C_x)` a subspace of
`ran(OF^*)+ran(AF^*)`; projecting this inclusion onto `ker(OF)` proves
`dim(AF)>=k_src`.  Choosing `AF^*` to span
`ran((I-PF)C_x)` attains the bound.  The adjoint argument proves and attains
the query bound.  Hence

- `k_src=0,k_qry=0`: information-exact two-bundle reuse;
- `k_src>0,k_qry=0`: a genuinely new source observation is unavoidable;
- `k_src=0,k_qry>0`: a genuinely new query observation is unavoidable; and
- both positive: both sides need new information, including the hidden--hidden
  block `(I-PF)C_x(I-PR)`.

Failure of a kernel condition is stronger than a cost statement: two lawful
states have identical paid streams and different desired action, so no
recombination of those streams can work.  It still does not prove a universal
one-GEMM lower bound, because a native precontracted scalar oracle could add
the missing observation in `O(n^2)` work.

### Information reuse versus literal zero-GEMM reuse

Equation (10B.1) allows an arbitrary fixed dense recombination of observed
entries.  Zero **transition-GEMM** reuse is stricter.  Let `L0` contain scalar
linear combinations of co-resident arrays, fixed masks/permutations, column
views, and reductions already charged in the hook; it excludes multiplication
by a new dense `n by n` matrix or a dense `k by p` H3 projector.  The action is
zero-GEMM reusable if and only if the factor maps above can be selected from
`L0`.  If they exist only as dense maps, the information premise passes but a
fresh arithmetic pass remains mandatory.

For the actual D10 recurrence,

`B_l=W_(l+1) Lambda_(l+1) W_(l+1)^T`,

`Lambda_l=D_l hadamard B_l`,

the paid conjugation may expose its one-sided intermediate
`U_l=W_(l+1)Lambda_(l+1)` as well as `B_l`.  The CountSketch query is

`Gamma_l S^*=W_l[GZ_l S^*+Q_l^T GY_l S^*]`.                       `(10B.4)`

A concrete zero-GEMM sufficient identity is that the bracketed RHS in
(10B.4) equals a fixed column view/masked scalar combination of the exact
D10 RHS whose paid one-sided product is resident.  Then `Gamma_l S^*` is the
same view/combination of `U_l`.  Merely sharing `W_l` is not enough.  If only
the inner `Q_l^T` action is resident, the final `W_l` multiplication is one
new rectangular pass; if neither action is resident, (10B.4) is two passes.
An arbitrary dense postmultiplication of `U_l` is also a new rectangular pass,
not a free column view.

On the source side, `Z_l S^*` is zero-GEMM only if it is a fixed `L0`
combination of arrays materialized by D14 at the same reset site.  Any catalog
containing only AQ/H2/Jordan matrices fails this condition symbolically:
`Z->-Z` leaves that catalog unchanged and reverses every nonzero signed-H3
factor.  A genuinely signed D14 tangent could evade this witness, but the
repository audit states that the actual D14 signed-H3/reset coefficients are
not yet defined.  It therefore cannot currently be counted as a resident
source.  Likewise no current identity places `GZ S^*`, `GY S^*`, or
`Gamma S^*` in the literal D10 resident span.

### Cap-49 rank gates

Let `a` be the total number of fresh rectangular width-mix passes per retained
channel and let `b` be the number of full one-sparse H3 coordinate/hash passes
per active layer.  For a common width `k`, the optimistic additive cost over
ten active layers is

`Cadd=10[a k/1024+b(29260/1024^2)+g_local]+C_hook<=1`.              `(10B.5)`

With `g_local=C_hook=0`, (10B.5) gives the following exact integer ceilings:

| fresh rectangular passes `a` | hash passes `b=0` | `b=1` | `b=2` |
|---:|---:|---:|---:|
| `0` | no compute-width bound | no compute-width bound | no compute-width bound |
| `1` | `k<=102` | `k<=73` | `k<=45` |
| `2` | `k<=51` | `k<=36` | `k<=22` |

The `a=0` row is not an infinite-rank claim: `k` is still bounded by the
rank of the literal free stream span, memory, hook work, and the cross-rank
condition.  It says only that no new dense transition is charged.  The known
unreused complete query (10B.4) has `a=2`; one transient query hash gives the
`36` ceiling, and a comparable source hash gives `22`.  Partial reuse of one
of the two query mixes gives the `a=1` row.

For an **exact endpoint-tuned dense** factor rather than a one-sparse
CountSketch, each new `k by p` H3 projection costs `kp/n^2` per layer.  Under
the optimistic convention of one such projection per fresh side and two
rectangular query mixes, the cap gives

- fresh source factor, query literally free:
  `10 k p/n^2<=1`, hence `k<=3`;
- fresh query factor, source literally free:
  `10 k[p/n^2+2/n]<=1`, hence `k<=3`;
- both factors fresh with common rank `k`:
  `10 k[2p/n^2+2/n]<=1`, hence `k<=1`.

These are optimistic ceilings: two separately projected query banks or any
source resolvent only lower them.  They expose why algebraic tightness of the
rank-`k_x` SVD does not make it deployable.  Exact additive closure through
dense new factors requires `k_x<=3` when one side is literally resident and
`k_x=1` when both sides are new.  Approximate CountSketch has the wider
`51/36/22` ceilings but must separately pass its frozen-seed cross-error and
endpoint Schur gates.

### Current disposition

The exact theorem is solved, but the favorable premise is not.  The present
audited catalog supplies neither a proved signed post-AQ source factor from
D14 nor a fixed-linear D10 identity for the complete query.  Thus the
zero-GEMM row cannot be credited.  Under the current ordinary CountSketch
census the conservative additive ceiling remains `k<=22` before
quadrature/resolvent/hook work; under exact dense principal factors the
optimistic both-fresh ceiling is rank one.  Promotion requires an explicit
resident-array equality or kernel/range receipt, not a claim of shared inputs.

## Checkpoint-10B graph-node split

### `P2R0G19B_H3_CROSS_OPERATOR_RANK`

- **Status:** `solved_theorem`, actual singular spectrum open.
- **Claim:** `C_x=UZ^*UG` is the exact endpoint-conditioned source--adjoint
  cross operator; `rank(C_x)` is the minimum number of arbitrary paired linear
  channels, and its singular tail controls every lower-rank approximation.

### `P2R0G19B_H3_TWO_BUNDLE_INFO_REUSE`

- **Status:** `solved_iff`, current premise fails/unproved.
- **Claim:** D14+D10 observations determine the action exactly iff
  `C_x=PF C_x PR`, equivalently both kernel containments (10B.2).  A sign-flip
  witness closes AQ/H2-only D14 source reuse.

### `P2R0G19B_H3_TWO_BUNDLE_ZERO_GEMM`

- **Status:** `open_identity`, no current receipt.
- **Claim:** zero transition-GEMM reuse additionally requires the SVD factors
  to be literal masks/views/scalar combinations of resident arrays.  Shared
  weights, scratch, or a dense postmultiplication do not qualify.

### `P2R0G19B_H3_FRESH_PASS_RANKS`

- **Status:** `solved_characterization`, actual ranks open.
- **Claim:** the minimum new source and query observation dimensions are
  `rank((I-PF)C_x)` and `rank(C_x(I-PR))`; a nonzero value proves a fresh
  observation unavoidable independently of implementation.

### `P2R0G19B_H3_D10_QUERY_REUSE`

- **Status:** `open_identity`, conservatively two fresh passes.
- **Claim:** query reuse needs literal equality of (10B.4) with the paid D10
  RHS/intermediate span.  No current identity supplies `SGZ`, `SGY`, or
  `Gamma`; sharing `W/Q` is insufficient.

### `P2R0G19B_H3_D14_SOURCE_REUSE`

- **Status:** `architecture_premise_open`.
- **Claim:** AQ/H2 streams cannot expose a signed H3 residual.  Reopen only
  after the actual D14 signed-H3 reset is defined and `Z S^*` is proved a
  literal resident factor at the contraction site.

### `P2R0G19B_H3_REUSE_CAP49`

- **Status:** `solved_cost_cases`, favorable case premise open.
- **Claim:** the common-CountSketch ceilings are `102/73/45` with one fresh
  width mix and `51/36/22` with two; exact dense factors permit rank at most
  three with one side free and rank one with both sides fresh.  The zero-pass
  case is bounded by resident-span rank, memory, and hook cost rather than by
  transition GEMMs.

## Checkpoint 10C: specialization to the contract-native AQ H4 score

The local AQ reset is

`g(z)=c+b^T z+(z^T M z-tr(M))/2`, `F=g_+`,

and a mixed fourth-chaos query is

`A_31(theta;U,V)=K4(F)[U,U,U,V]`, `theta=(c,b,M)`.

With `A=U^Tz`, `B=V^Tz`, `a=||U||^2`, and `d=<U,V>`, its exact fused score is

`H31(z;U,V)=A^3 B-3aAB-3dA^2+3ad`,

`A_31=E[F H31]`.

For a frozen mixed costate the AQ-parameter pullback is

`g_c(H)=E[1_(g>0) H31]`,

`g_b(H)=E[1_(g>0) z H31]`,

`G_M(H)=(1/2)E[1_(g>0)(zz^T-I)H31]`.                       `(10C.1)`

All formulas below apply after the same-site D11/Q24 action and complete old
q4 range have been removed.  Otherwise the oracle merely recomputes old
ownership.

### Exact AQ/D10 cross operator

Let `T_theta` be the stacked AQ-parameter tangent over all active reset sites,
and let `Y_10` be the admissible D10 `q4_perp` costate space.  Define

`C_AQ:Y_10->T_theta^*`

by

`<dot(theta),C_AQ H>
 =D_theta <K4(g_theta,+),H>[dot(theta)]`.                  `(10C.2)`

For a directional costate, `C_AQ H` is exactly the triple (10C.1); extend by
linearity to an arbitrary costate.  Thus `C_AQ` is the contract-native
specialization of the source--adjoint cross operator in Checkpoint 10B.

Let `J14:T_theta->E14` be the **actual** parameter-to-resident-D14-word
Jacobian and let `O10:Y_10->E10` retain only the literal resident D10 costate,
RHS, and paid intermediates.  Put

`P14=J14^dagger J14` on `T_theta`,

`P10=O10^dagger O10` on `Y_10`.

The local score and its joint-refit pullback use only the paid two-bundle
information if and only if

`C_AQ=P14 C_AQ P10`.                                      `(10C.3)`

Equivalently,

`ker(J14) subset ker(C_AQ^*)`,

`ker(O10) subset ker(C_AQ)`.                               `(10C.4)`

The first inclusion says that every AQ parameter perturbation invisible to
the D14 base word has zero score derivative against every admitted q4
costate.  The second says that every D10-invisible costate has zero AQ score
against every admitted parameter tangent.  These are necessary by the
two-indistinguishable-states argument and sufficient by the pseudoinverse
factorization from Checkpoint 10B.

The exact missing observation ranks are

`k_14=rank((I-P14)C_AQ)`,

`k_10=rank(C_AQ(I-P10))`.                                 `(10C.5)`

They are respectively the minimum numbers of new AQ-pullback and costate
observations needed to repair the two resident catalogs.  A positive `k_14`
cannot be cured by the resident D10 word; a positive `k_10` cannot be cured
by the D14 base word.  In the physical schedule a new independent D14 word
costs about `28` and a new D10 word about `20`, so either missing rank closes
the additive Base48 lane unless it is supplied by a native reset-local scalar
or another already-paid literal array.

At factor level, exact literal reuse requires a resident row `a_14(H)` with

`C_AQ H=J14^* a_14(H)`                                    `(10C.6)`

for every retained costate, where `a_14(H)` is a mask/view/scalar combination
of resident D14/D10 arrays.  An arbitrary dense solution of (10C.6) proves
information reuse but not zero-GEMM reuse.  Equation (10C.6), with the exact
same parameter convention, reset site, frame, and lifetime, is the binding
AQ-parameter-pullback receipt.

### Mixed-direction rank is not Bellman matrix rank

The directional score interface can consume a q4 costate only after an exact
partially symmetric decomposition

`H_10^perp=sum_(alpha=1)^k omega_alpha
 Sym(U_alpha^tensor3 tensor V_alpha)`.                     `(10C.7)`

The minimum such `k` is the `(3,1)` mixed/Waring rank

`k_mix=rank_(3,1)(H_10^perp)`.

If some terms of (10C.7) are already literal resident factors, the actual new
rank is the minimum rank of the residual after subtracting their span.  The
ordinary matrix rank of an external K31 mask is not this number.  After a
dense leg map, even `A=xy^T` gives

`sum_i x_i ell_i^tensor3 tensor (sum_j y_j ell_j)`;

the Fermat cubic on the left has generic Waring rank equal to the number of
nonzero `x_i`, namely `n` for a dense vector.  Hence a generic matrix-rank-one
costate needs `k_mix=n=1024` directional oracle calls and fails the local
`34`-direction screen below.

Literal direction reuse is therefore the identity (10C.7) together with

`(omega,U,V)=L10^0(resident D10 q4_perp arrays)`,            `(10C.8)`

where `L10^0` uses no new eigendecomposition, Waring/SVD factorization, or
dense width multiplication.  If only `H_10^perp` is resident, information is
present but the low-rank arithmetic receipt is absent.  The flattening rank is
a cheap necessary lower bound on `k_mix`; it is not generally sufficient for
the partially symmetric decomposition.

### Exact spectral-state and pullback reuse identities

For each unit write

`M=R Lambda R^T`, `bhat=R^T b`,

`Uhat_alpha=R^T U_alpha`, `Vhat_alpha=R^T V_alpha`.          `(10C.9)`

The one-dimensional Fourier formula for the AQ score depends only on `c`,
the eigenvalues, these direction coordinates, the full coordinates of `b`
(including its null-space part), and the fixed normalization/centering.  Thus
a reused forward eigensystem is lawful if and only if the tuple (10C.9)
belongs to the literal D14/D10 resident catalog at the same site and gauge.
Eigenvalues of a different matrix, an eigensystem recomputable from `M`, or
directions available only in the physical frame are not literal reuse.

Forward spectral reuse still does not imply reverse AQ-pullback reuse.  Let
`Omega=R^T dot(R)` be skew, put hats on all physical score gradients, and let
`G_Ualpha,G_Valpha` denote direction gradients when those directions are
coadapted.  Since

`R^T dot(M)R=dot(Lambda)+[Omega,Lambda]`,

`R^T dot(x)=dot(xhat)+Omega xhat`,

the exact eigenframe torque is

`T(Omega)=<Ghat_M,[Omega,Lambda]>
 +ghat_b^T Omega bhat
 +sum_alpha(ghat_Ualpha^T Omega Uhat_alpha
            +ghat_Valpha^T Omega Vhat_alpha)`.              `(10C.10)`

Terms for frozen physical directions are included through the induced
variation of their hatted coordinates; terms for genuinely fixed spectral
coordinates are omitted only under that explicit convention.  A
fixed-eigenbasis reverse that carries only eigenvalue/coordinate derivatives
is exact if and only if

`T(Omega)=0 for every Omega^T=-Omega`.                       `(10C.11)`

Otherwise it misses eigenvector motion.  The alternatives are to carry the
full orientation tangent in D14 or to apply the exact eigendecomposition
adjoint/back-rotation to recover `(g_b,G_M)`.  The missing orientation rank is

`k_Omega=rank[Omega -> {T_alpha(Omega)}_alpha]`,              `(10C.12)`

bounded by `r(r-1)/2` per unit and generically nonzero for mixed directions.
The simpler commutator condition `[M,G_M]=0` is sufficient only when all
vector/direction torque terms vanish separately.

Consequently the complete literal-reuse receipt has three independent parts:

1. the AQ score-sufficient spectral tuple (10C.9) is resident;
2. the D10 costate has a resident mixed factorization (10C.7)--(10C.8); and
3. the physical AQ pullback satisfies (10C.6), including either the full
   spectral adjoint or the zero-torque identity (10C.11).

Failure of any one part invalidates a zero-cost claim even when the other two
hold.

### Cap-49 directional-rank ceiling

Let `D=10`, `n=1024`, `r=55`, let `K` be the certified Fourier-node count,
and write

`c_spec` for the number of new per-unit `r^3` spectral operations
(eigendecomposition or dense matrix-gradient back-rotation), and
`c_dir` for the number of separate `r^2` direction transforms per mixed term.
The optimistic local census is

`C_AQscore=
 D/n^2 [c_spec r^3+c_dir k_mix r^2+K k_mix r]
 +C_factor+C_hook <=1`.                                    `(10C.13)`

Here `C_factor` charges a nonresident decomposition (10C.7); it is zero only
under (10C.8).  Equation (10C.13) is subordinate to (10C.3): a missing D14 or
D10 stream already costs `28` or `20` and cannot be repaired by this local
census.

At current rank,

`10 r^3/n^2=1.5867...>1`.

Therefore:

- **no eigensystem reuse:** `c_spec>=1`, so the additive lane is closed for
  every `k_mix`, even before quadrature;
- **forward eigensystem reused but AQ matrix pullback not reused:** the new
  back-rotation again gives `c_spec>=1`, so the lane is still closed;
- **eigensystem and full AQ pullback both literal:** `c_spec=0`, and

  `k_mix<=floor[(n^2/D-C')/(c_dir r^2+K r)]`,               `(10C.14)`

  where `C'=(C_factor+C_hook)n^2/D` in the same arithmetic units.

Ignoring quadrature/factor/hook work, the campaign's optimistic fused
`c_dir=1` convention yields

`k_mix<=floor(1024^2/(10*55^2))=34`.

If `U` and `V` require two literal separate eigenbasis transforms,
`c_dir=2` and the ceiling is `17`.  The `34` claim therefore needs an actual
fused/shared transform receipt, not big-O notation.  If all `Uhat,Vhat` are
already resident (`c_dir=0`), only quadrature gives the width gate

`k_mix<=floor(1024^2/(10*K*55))`,

and `K` must come from a certified tail bound.  With one fused transform and
quadrature, the exact optimistic formula is

`k_mix<=floor[104857.6/(3025+55K)]`;                       `(10C.15)`

with two transforms replace `3025` by `6050`.

For comparison, at `r=45` one unreused eigensystem costs about `0.869`,
leaving the nominal fused ceiling `k<=6`; with reuse it gives `k<=51` before
quadrature.  At the actual `r=55`, the corresponding no-reuse branch is
already empty.

The generic dense-rank-one K31 example has `k_mix=1024`, so it is closed even
under the most favorable documented `34` screen.  The only reopen is a
literal D10 mixed decomposition of rank at most the applicable bound, a
resident same-gauge AQ eigensystem and pullback, certified quadrature, and a
nonzero final old-span Schur gain.

## Checkpoint-10C graph-node split

### `P2R0G19B_AQH4_TWO_BUNDLE_CROSS_IFF`

- **Status:** `solved_iff`, actual resident-range receipt open.
- **Claim:** the contract-native parameter/source cross operator is (10C.2)
  and D14+D10 information reuse holds exactly iff (10C.3)--(10C.4).

### `P2R0G19B_AQH4_MISSING_BUNDLE_RANKS`

- **Status:** `solved_characterization`, actual ranks open.
- **Claim:** `k_14` and `k_10` in (10C.5) are the minimum missing AQ-pullback
  and q4-costate observation dimensions.  Any genuinely new full D14/D10 word
  violates Base48.

### `P2R0G19B_AQH4_MIXED_DIRECTION_RANK`

- **Status:** `conditional/open`; generic dense mask closed.
- **Claim:** oracle-call count is the `(3,1)` mixed rank in (10C.7), not
  Bellman matrix rank.  A dense ordinary-rank-one mask generically has
  `k_mix=n=1024>34`.

### `P2R0G19B_AQH4_LITERAL_FACTOR_REUSE`

- **Status:** `open_identity`.
- **Claim:** zero-GEMM reuse requires (10C.6), (10C.8), and same-site/gauge
  spectral tuple (10C.9); algebraic reconstructibility or common weights are
  insufficient.

### `P2R0G19B_AQH4_SPECTRAL_PULLBACK_TORQUE`

- **Status:** `solved_theorem`, zero-torque premise open.
- **Claim:** forward eigensystem reuse is reverse-exact only with the full
  spectral adjoint or the all-skew torque cancellation (10C.11).  Otherwise a
  new `r^3` back-rotation closes the r55 additive lane.

### `P2R0G19B_AQH4_CAP49_RANK`

- **Status:** `solved_cost_cases`, favorable literal-reuse premises open.
- **Claim:** without reused eigensystems, or without reused AQ matrix
  pullback, no rank fits at r55.  With both reused the nominal fused ceiling is
  `34`, the literal two-direction-transform ceiling is `17`, and quadrature
  tightens them by (10C.14)--(10C.15).

## Checkpoint 10D: signed q211 surface-query factor rank and product audit

This checkpoint concerns the **contract-native post-reset** `211` source, not
the polynomial quartic of an already projected AQ/NCQC state.  Reverse-compile
the complete-old map and every collision map first:

`Ltilde=A_old^* L`,

with `Ltilde_(i;jk)=Ltilde_(i;kj)` and all forbidden collisions either sewn to
their typed Q22 boundary or set to zero.  Every rank and cost below is computed
after this operation.  A raw costate rank before old/collision compilation is
not a source rank receipt.

### The one genuinely new scalar

For one latent mixed factor, the exact score is

`H211(z;U,S,T)=:(U^Tz)^2(S^Tz)(T^Tz):`,

and

`E[F H211]=K4(F)[U,U,S,T]`.                                `(10D.1)`

Let `Ksurf` denote the true or AQ-ReLU contract-native H4 surface and let
`Kpoly` denote the complete low-chaos/AQ/NCQC quartic already obtainable from
resident matrix products.  The only possible new fixed-endpoint scalar is

`s_surf=<Ltilde, Pi_211,off (Ksurf-Kpoly-Kother_old)>`.       `(10D.2)`

Every term on the right must use the same centering, normalization, reset
site, and collision convention.  If a full-AQ H211 surface has already been
credited, (10D.2) is zero by ownership and cannot be added again.  If only
post-reset AQ coefficients or NCQC products are resident, (10D.2) is not
owned.

Let `Ores` be the literal scalar/vector outputs of all resident D10/D14
product and score providers at that lifetime.  One zero-new-product scalar
exists if and only if there is a fixed zero-cost row `a` such that

`s_surf=a^T Ores`.                                          `(10D.3)`

For a family of lawful local laws, (10D.3) implies the fiber condition

`Ores(x)=Ores(y) => s_surf(x)=s_surf(y)`,                   `(10D.4)`

and infinitesimally `ker(D Ores) subset ker(D s_surf)`.
Conversely, the fiber condition gives an information-level factor on the
resident quotient, but a literal free saving still requires that factor to
be a resident mask/view/scalar reduction rather than a new dense or
trivariate evaluator.  Equations (10D.3)--(10D.4) are the exact provenance
gate for the requested “one additional scalar.”

### Pairwise Gaussian rest is information-insufficient

There is an explicit two-law witness.  Let `B,C,E,D0` be independent
Rademacher signs, let `D=BC`, and for `0<epsilon<1` put

`A=E sqrt(1+epsilon D)`,

`A0=E sqrt(1+epsilon D0)`.

The laws `(A,B)`, `(A,C)`, and `(B,C)` are identical to the corresponding
pairwise laws with `A0`: conditional on either `B` or `C`, `D` is an
independent sign.  Nevertheless all pair covariances vanish and

`kappa(A,A,B,C)=E[A^2BC]=epsilon`,

`kappa(A0,A0,B,C)=0`.                                      `(10D.5)`

Adding sufficiently large constants makes all four variables nonnegative
without changing a fourth cumulant.  Thus even the complete collection of
bivariate output laws cannot identify signed q211 on the unrestricted
post-reset law class.  The incumbent pairwise Gaussian/Mehler rest is
therefore not a q211 source.  Under an **additional exact joint-Gaussian
premise**, means and covariance determine the trivariate law, but evaluating
the resulting arcsine/orthant surface is still a new typed trivariate
operation unless that exact evaluator is literally resident.

### Strongest exact factor-rank invariant

Let `V` be the latent direction space and collect the reverse-compiled mixed
surface test into

`Hres in Sym^2(V) tensor Sym^2(V)`.

Use the repeated pair `(UU)` as the left factor and the singleton pair `(ST)`
as the right factor.  Define the canonical flattening

`F_22(Hres):Sym^2(V)->Sym^2(V)`.

One direct H211 query contributes

`U^tensor2 tensor (S odot T)`,

which has flattening rank one.  Hence every exact `k`-query realization obeys

`boxed: k >= rho_22:=rank F_22(Hres)`.                       `(10D.6)`

This lower bound is basis invariant and survives every reparameterization of
the individual factors.  It is also an exact rank-one characterization after
the following algebraic type checks: `rho_22=1`, the left singular vector is
on the quadratic Veronese cone `{U^tensor2}`, and the right symmetric matrix
has rank at most two so it equals `S odot T`.  For general `k`, exact
realizability is membership in the `k`-th secant of this typed product
variety; ordinary matrix rank is necessary but not sufficient.

For the common external form

`Hres=sum_(i=1)^n u_i^tensor2 tensor B_i`,                  `(10D.7)`

if the `u_i^tensor2` and residual singleton slices `B_i` are each linearly
independent, then `rho_22=n`.  This is generic whenever
`dim Sym^2(V)>=n`.  At `n=1024`, both `r=49`
(`dim Sym^2=1225`) and `r=55` (`1540`) satisfy that inequality, so

`boxed: rho_22=1024 generically`.                           `(10D.8)`

Old projection can lower (10D.8), which is why the actual post-old flattening
must be formed symbolically.  But neither an ordinary rank-one Bellman matrix
nor a rank-one CP costate proves a low `rho_22`: the source kernel and dense
leg map can generate independent `B_i` slices.

There is a second, independent generic no-go.  For a standardized
zero-mean Gaussian triple, the exact ReLU q211 kernel

`K(x,y,z)=kappa4(X_+,X_+,Y_+,Z_+)`

contains square roots and arcsines.  On the slice `z=0`, its bivariate
coefficient Hankel matrix has infinite rank.  Any finite collection of
resident Price/trace/pair/Jordan products has finite separable rank on that
slice.  Therefore even a CP endpoint costate does not turn the full surface
into a finite combination of pair products on an open covariance set.  The
only exact choices are a trivariate surface evaluator, a controlled infinite
Hermite expansion, or a structured common-ray/block exception.

### Sharp positive exception

If the live Gaussian coordinates obey one sign-coherent ray

`m=mu v`, `C(t)=q(t)vv^T`,

then

`K_(i;jk)(t)=kappa4((mu+sqrt(q(t))Z)_+)
               v_i^2 v_j v_k`.                             `(10D.9)`

The source flattening is rank one for the entire path.  After old/collision
projection, one scalar amplitude and one resident endpoint contraction

`A_L=sum_(i,j,k)Ltilde_(i;jk)v_i^2v_jv_k`

are necessary and sufficient.  Independent sign-coherent one-ray blocks give
the corresponding block rank, provided the old/collision map preserves the
blocks.  Equation (10D.9), or a literal target-specific common support
`G_L(t)=U Theta(t)U^T`, is the strongest exact low-rank reopen.  Low covariance
rank two is not enough: the arc-cosine kernel on generic planar directions
already has growing coordinate rank.

### Audit of the resident-product and score-provider routes

**Resident full product `P=M_iN_a`.**  At the synchronized seam it gives the
polynomial NCQC action

`sum_(i,a,b)Theta_(i,ab)
 {32<P_(ia),P_(ib)>+16<P_(ia)^T,P_(ib)>}`,                 `(10D.10)`

and, by retaining symmetric and skew halves, may make this low-chaos
subtraction zero-new-GEMM.  It does **not** evaluate the gate/surface residual
(10D.2).  Thus its true saving is avoidance of a duplicate `D k r^3/n^2`
matrix-product charge for `Kpoly`; it provides zero product saving for the
new trivariate expectation itself.  A literal `P` buffer is useful provenance
for the subtraction, not evidence that `s_surf` is resident.

**Existing H31 score.**  A universal H31 oracle can synthesize one H211 atom
by the exact polarization

`K[U,U,S,T]=(1/6){K[U+S,U+S,U+S,T]
                    -K[U-S,U-S,U-S,T]
                    -2K[S,S,S,T]}`.                        `(10D.11)`

Hence it costs three new H31 direction calls per H211 factor.  One fixed
resident H31 scalar does not satisfy (10D.11), and K31 ownership does not
identify off-diagonal 211.  A direct resident H211 provider is three times
more direction-efficient.

**Full-AQ r49 plus Gaussian rest.**  The r49 eigensystems and Fourier nodes
may be shared only if they use the identical pre-reset `(c,b,M)`, node set,
normalization, and lifetime.  Such sharing removes the eigensolve/resolvent
setup, but every new H211 factor still needs its `U,S,T` contractions and
node reductions.  The pairwise Gaussian rest supplies no term in (10D.2).
If the rest is made self-consistent by forming fresh AQ Grams, the documented
extra Gram bill already pushes the package above 49; only the frozen-rest
frame retains the nominal surface margin, with its frame mismatch charged as
model error.

### Precise cap-49 exchange

For `D` active surface sites, latent rank `r`, `Q` certified Fourier nodes,
and direct H211 factor rank `k`, the optimistic incremental census after a
literal eigensystem/resolvent reuse is

`Csurf(k)=D k(3r^2+Qr)/n^2+C_compile+C_hook`.                `(10D.12)`

The coefficient `3` is the literal transform count for `U,S,T`; it may be
reduced only by a resident same-gauge direction-coordinate identity.  If the
eigensystem is not resident, add `D r^3/n^2`.  If the alleged provider is
H31-only, replace `k` by `3k` using (10D.11).

For the native Base48 route at `r=55,D=10`:

- without literal eigensystem reuse, `10*55^3/1024^2=1.5867>1`, so no rank
  fits;
- with reuse and before quadrature, direct H211 permits at most `k=11`;
- with `Q=32`, (10D.12) permits at most `k=9`;
- an H31-only implementation then permits at most three H211 factors.

For the nominal full-AQ r49 subtotal `40.02`, let `B=49-40.02=8.98`:

- with the paid eigensystems/nodes literally alive, `r=49,Q=32,D=10` gives
  `Csurf(k)=0.08365... k`, hence `k<=107`;
- if all fourteen sites consume the surface, the ceiling is `k<=76`;
- if the D10-site eigensystems must be recomputed, their `1.122...` charge
  lowers the ten-site ceiling to `k<=93`;
- H31 polarization lowers the corresponding direct-factor ceilings to
  `35`, `25`, and `31`.

These are arithmetic ceilings, not rank receipts.  The generic exact lower
bound `rho_22=1024` violates every row.  Only an endpoint-specific residual
flattening of cap-compatible rank, or the common-ray/block exception, keeps
the exact surface lane open.

### Approximation and value gates

Let `H_k` be any cap-compatible typed factor approximation and define the
coherent endpoint action error

`eta_k=|sum_l <Hres_l-H_(k,l),K4_l>|+eta_quad+eta_model`.    `(10D.13)`

Eckart--Young gives the necessary coefficient-space tail

`||F_22(Hres)-F_22(H_k)||_F^2
 >=sum_(j>k)sigma_j(F_22(Hres))^2`,                         `(10D.14)`

but only the task-weighted action (10D.13) may be charged to the final error.
A safe sufficient conversion is the corresponding weighted Cauchy product.
Promotion requires simultaneously

`eta_k<=3.80e-6`,                                           `(10D.15)`

and, after complete incumbent residualization,

`Delta_surf=<t,f_k>_V^2/||f_k||_V^2 >=2.86e-10`.            `(10D.16)`

In the campaign's normalized amplitude convention, the robust closer form is

`s_k>=eta_k+sqrt(2.86e-10)
     =eta_k+1.691e-5`.                                      `(10D.17)`

For a replacement package, (10D.16) must additionally repay the Schur value
of every removed D14/D10 feature.  Coefficient rank, nonzero surface norm,
and Gaussian information sufficiency imply none of (10D.15)--(10D.17).

## Checkpoint-10D atomic graph nodes

### `P2R0G19B_Q211_SURFACE_PROVENANCE`

- **Status:** `solved_iff`, literal equality absent.
- **Claim:** the only new scalar is (10D.2); zero-new-product ownership is
  exactly (10D.3), after complete-old/collision compilation.  Resident low
  chaos and an already credited full surface are mutually exclusive ownership
  cases.

### `P2R0G19B_Q211_PAIRWISE_REST_NOGO`

- **Status:** `closed_proof` for unrestricted post-reset laws.
- **Claim:** the pairwise-identical witness (10D.5) has different signed q211;
  bivariate Gaussian/Mehler rest is not a q211 measurement.

### `P2R0G19B_Q211_SURFACE_FLATTEN_RANK`

- **Status:** `solved lower bound`, frozen residual rank open.
- **Claim:** every direct H211 factor has `(UU)|(ST)` flattening rank one, so
  `k>=rho_22`; generic r49/r55 width-1024 residuals have `rho_22=1024`.

### `P2R0G19B_Q211_SURFACE_TYPED_SECANT`

- **Status:** `solved exact characterization`, instance membership open.
- **Claim:** matrix rank is only necessary; exact rank `k` requires membership
  in the typed H211 secant variety, including Veronese repeated factors and
  symmetric-rank-at-most-two singleton factors.

### `P2R0G19B_Q211_SURFACE_PAIRWORD_NOGO`

- **Status:** `closed generic`.
- **Claim:** the Gaussian q211 kernel has infinite separable rank on an open
  covariance slice, so no finite resident Price/trace/pair/Jordan dictionary
  equals the full surface.

### `P2R0G19B_Q211_SURFACE_COMMON_RAY`

- **Status:** `exact conditional GO`.
- **Claim:** a sign-coherent common ray gives the rank-one amplitude identity
  (10D.9); blockwise reuse additionally requires old/collision preservation.

### `P2R0G19B_Q211_RESIDENT_P_AUDIT`

- **Status:** `low-chaos reuse solved`, surface reuse closed.
- **Claim:** a literal `P=M_iN_a` saves the polynomial product bill through
  (10D.10), but cannot form the new trivariate residual (10D.2).

### `P2R0G19B_Q211_H31_POLARIZATION_COST`

- **Status:** `solved identity/cost`.
- **Claim:** one H211 atom needs three arbitrary-direction H31 calls by
  (10D.11); one fixed H31 scalar has no off-diagonal-211 coverage.

### `P2R0G19B_Q211_SURFACE_CAP49`

- **Status:** `solved cost cases`, favorable rank/lifetime receipts open.
- **Claim:** native Base48 permits no factor without eigensystem reuse and at
  most `9` direct factors with 32 nodes; full-AQ r49 nominally permits `107`
  over D10 (`76` over all fourteen).  Generic exact rank `1024` is closed.

### `P2R0G19B_Q211_SURFACE_VALUE_GATE`

- **Status:** `open endpoint receipt`.
- **Claim:** factor, quadrature, and model error must pass (10D.15), while the
  old-residual Schur gain and robust amplitude must pass (10D.16)--(10D.17).

## Checkpoint 10E: literal D11 CP3 audit of the Gaussian q211 jet

This checkpoint corrects the scope of the generic statement in 10D.  An
**exact, same-cut** CP3 value and threshold jet remove the trivariate Gaussian
part of q211.  The literal D11 carrier is neither of those objects.  It is a
finite full-birth CP approximation whose only deployed dense reads are K21 and
the marginal K3 diagonal.  Therefore the raw-surface rank obstruction in 10D
must be applied to the true-minus-D11-assembled residual, not discarded merely
because the package contains arrays named CP3.

### Exact identity and the two regimes

For standardized Gaussian preactivations with thresholds `tau`, write

`K3_ijk=kappa3((X_i-tau_i)_+,(X_j-tau_j)_+,(X_k-tau_k)_+)`.

The exact connected identity is

`K211_(i;jk)=-(tau_i+2m_i+d_i+R_ij d_j+R_ik d_k)K3_ijk`
`             +F_pair_(i;jk)`,                             `(10E.1)`

where `d_a=partial/partial tau_a` and `F_pair` is a finite composition of
means, bivariate truncated moments, and their threshold derivatives.  Thus an
arbitrary-bias site needs the **slot-typed first threshold jet**, not merely a
CP3 value.

At the exactly zero-threshold site parity eliminates the independent jet and
(10E.1) reduces to

`K211_(i;jk)=-2 mu K3_ijk+F_edge_(i;jk)`.                  `(10E.2)`

For one endpoint atom `L_(i;jk)=alpha_i beta_j gamma_k`, every term in
`F_edge` is a bounded list of matvec/Hadamard reductions except

`B=tr(D_alpha R D_beta Pplus D_gamma R)`,
`Pplus=p[R]`.                                               `(10E.3)`

The arbitrary-bias JVP route and the zero-bias one-sandwich route are two
specializations of the same identity.  They must not be charged as two
independent providers: with a literal JVP, its boundary/triangle content is
inside the JVP; after zero-threshold parity removes that JVP, (10E.3) is the
remaining broad edge word.  The L5--14 reset sites are generically biased
because the literal code uses `pre_mean=W^T mean`; (10E.2) is exact only at a
site where `pre_mean/sigma` is identically zero, not approximately small.

### What the D11 CP3 tensor actually is

The literal functions `_cp_read` and `_cp_k3` prove the normalization.  The
carried tensor is

`That3=sum_(a=1)^r lambda_a S(p_a,p_a,q_a)`,               `(10E.4)`

where

`S(p,p,q)_ijk=p_i p_j q_k+p_i q_j p_k+q_i p_j p_k`.

Indeed the only full-plane CP read is

`That3_(iij)=sum_a lambda_a(p_ia^2 q_ja+2p_ia q_ia p_ja)`, `(10E.5)`

and the final marginal read is

`That3_(iii)=3 sum_a lambda_a p_ia^2 q_ia`.                `(10E.6)`

For a generic ordered endpoint atom its value action is exactly

`<alpha tensor beta tensor gamma,That3>`
`=sum_a lambda_a[(p_a.alpha)(p_a.beta)(q_a.gamma)`
` + (p_a.alpha)(q_a.beta)(p_a.gamma)`
` + (q_a.alpha)(p_a.beta)(p_a.gamma)]`.                    `(10E.7)`

Thus a new generic scalar read needs the six factor projections
`p^T alpha,p^T beta,p^T gamma,q^T alpha,q^T beta,q^T gamma`; it needs four
when `beta=gamma`.  `_cp_read` materializes instead the two products
`(p*p)(lambda*q)^T` and `(p*q)(lambda*p)^T`; `_cp_k3` materializes only the
diagonal.  Neither buffer is a fixed reduction of (10E.7).  The tensor witness
`e_1 odot e_2 odot e_3` has zero K21 and diagonal reads but a nonzero
all-distinct endpoint action, so no universal recovery from the deployed
reads is possible.

The local grow is also explicit.  With `g=Phi(-tau)`,
`b=phi(tau)/sigma`, transported columns `pbar,qbar`, and covariance `C`, it
uses

`p_inh=g*pbar, q_inh=g*qbar`,
`p_birth^a=g*C_(:a), q_birth^a=e_a, lambda_birth^a=b_a`.   `(10E.8)`

For one common standardized threshold direction `v`, the corresponding local
tangent can be formed algebraically from

`gdot=-phi(tau)*v`, `bdot=-tau*b*v`,
`pdot_inh=gdot*pbar`, `qdot_inh=gdot*qbar`,
`pdot_birth^a=gdot*C_(:a)`.                                `(10E.9)`

This is a derivative formula, not a residency receipt.  D11 stores no
`pdot,qdot,bdot` bank and computes no endpoint contraction of it.  The live
`gate`, `pdf`, and covariance save only the elementwise coefficient setup in
(10E.9).  The CP banks alternate and are overwritten during the single
forward sweep; no future q211 costate has a documented same-lifetime overlap.

### D11 is not the exact synchronized CP3 owner

"Full-birth" means that every birth in the chosen closure is retained; it
does not make that closure the exact Gaussian trivariate cumulant.  The first
reset gives a symbolic counterexample.  For distinct `i,j,k`, (10E.8) gives

`That3_ijk=b_k g_i g_j C_ik C_jk`
`          +b_j g_i g_k C_ij C_kj`
`          +b_i g_j g_k C_ji C_ki`,                        `(10E.10)`

while `_reset_stars` sets a coordinate-star residual only from
`K21_target-cp_post_k21`; that star has no all-distinct entry at its birth
site.  At zero threshold the exact ReLU CP3 cumulant has, among its higher
Hermite diagrams, the nonzero `H2-H2-H2` triangle proportional to
`R_ij R_ik R_jk`.  Equation (10E.10) is only quadratic in the off-diagonal
correlations and cannot contain that term, let alone the remaining infinite
tail.  Consequently

`K3_exact=That3_D11+delta3`, with `delta3 != 0` generically. `(10E.11)`

The exact q211 decomposition against the literal package is therefore

`K211=-D_i That3_D11+F_pair-D_i delta3`                    `(10E.12)`

in the biased case, and

`K211=-2mu That3_D11+F_edge-2mu delta3`                    `(10E.13)`

at zero threshold.  The exact CP3 fusion removes the raw infinite-rank
obstruction only when `delta3=0` at the synchronized cut.  The literal D11
code proves the opposite generically.  Thus the `rho_22=1024` generic screen
from 10D remains a lawful screen for the residual in (10E.12)--(10E.13),
although its actual endpoint-weighted rank still must be computed after old
and collision projection.

### The threshold-JVP target rank and arithmetic

Even if one elects to differentiate only the assembled D11 tensor, a single
rank-one external endpoint does not produce one common threshold direction.
For example the `R_ij d_j` part has coefficient matrix

`C_(alpha,beta)=D_alpha R D_beta`.                          `(10E.14)`

Any representation by `k` common directional JVPs has coefficient-matrix
rank at most `k`.  Hence the exact direction lower bound is

`k >= rank(D_alpha R D_beta)`,                             `(10E.15)`

and similarly for the gamma leaf.  With dense nonzero endpoint factors and a
generic correlation matrix, (10E.15) is `n=1024`.  A scalar final answer does
not lower this acquisition rank.

In the CP factorization, the same obstruction appears operationally.  For
each factor column, the leaf terms require the four query-specific blocks

`R(D_beta p_tau), R(D_beta q_tau),`
`R(D_gamma p_tau), R(D_gamma q_tau)`,                      `(10E.16)`

with two blocks only in the special `beta=gamma` atom.  No block in (10E.16)
is one of the literal `_cp_read`, covariance-conjugation, or star-push RHSs.
Let `r_l` be the total post-grow rank.  The current `n` birth columns have
`q_birth=e_a`, so their `q_tau` row tangent is zero; the derivative of their
`lambda_birth` is a diagonal/pair reduction rather than a dense factor
product.  Consequently (10E.16) has the sharp direct cost

`2 r_l/n+2(r_l-n)/n=4r_l/n-2`                              `(10E.17)`

per site, or `2r_l/n-1` for `beta=gamma`, plus lower-order factor projections
and birth-weight reductions.  On the ten L5--14 sites, the full-birth ranks
are `5n,6n,...,14n`, so

`sum_l r_l/n=95`,
`C_leaf=360` generic, or `180` for beta=gamma.              `(10E.18)`

This is far outside either the Base48 one-equivalent slack or the nominal
full-AQ r49 margin.  If `R_l` has a resident exact rank-`s_l` factorization,
the sharp associative replacement for each RHS is

`min(r_l/n, 2 s_l r_l/n^2)`,                               `(10E.19)`

before factor-projection, coefficient, lifetime, and hook costs.  Generic
dense D11 correlations give `s_l=n`, so direct multiplication wins.  A low
rank claim must be a literal same-cut factorization; numerical near-rank is an
approximation with its own coherent endpoint error gate.

### The one broad sandwich is exact but not resident

Let `A=D_beta Pplus D_gamma`.  The exact one-product evaluation of (10E.3) is

`Z=R A`,
`B=<D_alpha Z,R>_F`,                                      `(10E.20)`

using symmetry of `R`.  Diagonal scalings, kernel formation, collision
subtractions, and the final Frobenius reduction do not add another dense
product.  Conversely, for generic full-rank `Pplus` and dense supported
`beta,gamma`, `A` has rank `n`; a fixed exact factorization cannot beat the
one-product count.

The literal D11 pair routines form pair kernels elementwise.  Its paid dense
products are covariance conjugations by `W`, the two CP products in
`_cp_read`, and star products with `W`.  None has left factor `R` and literal
RHS `D_beta Pplus D_gamma` at the q211 seam.  Therefore (10E.20) has **zero
documented product fusion**.  Algebraic column-span reconstruction from the
wide CP bank is not reuse: it would require a new dense query-specific
coefficient solve/product and has the wrong RHS and lifetime.

The marginal bill is one dense equivalent per active zero-threshold site.
Across ten sites it is `10`, so Base48+1 is closed and the optimistic full-AQ
subtotal is `40.02+10=50.02>49` before CP3 reads, masks, or hooks.  The full-AQ
margin `8.98` can cover at most eight such sites in isolation.  In the actual
biased L5--14 lane, however, (10E.20) is not a substitute for the missing JVP
and that eight-site arithmetic is not a deployment GO.

### Ownership statement

There are three distinct meanings of "CP3 is old":

1. D11 stores the approximate value factors (10E.4) -- true;
2. the deployed reads own an arbitrary endpoint action (10E.7) -- false as a
   literal-buffer claim;
3. D11 owns the exact same-cut value and slot-typed threshold JVP in (10E.1)
   -- false generically by (10E.10)--(10E.11), and independently absent as a
   JVP read by (10E.9), (10E.14)--(10E.16).

Therefore neither the CP3-named state nor pairwise Q24 kernels may be removed
from a new feature by name alone.  Complete old ownership is the response-row
condition

`G_l[D_i K3_exact,F_pair] in row(J_D11,J_Q24,J_collision)` `(10E.21)`

at the same cut.  No literal D11 receipt establishes (10E.21).  A new scalar
read of the approximate CP bank must still be Schur-residualized against the
incumbent response; the exact residual `D_i delta3` must remain in the source
definition.

## Checkpoint-10E atomic graph nodes

### `P2R0G19B_Q211_CP3_JET_IDENTITY`

- **Status:** `solved_exact`.
- **Claim:** arbitrary-bias Gaussian q211 is the slot-typed CP3 threshold jet
  plus a finite pair composition, (10E.1); the zero-threshold value-plus-one-
  sandwich formula is its parity specialization, not a second provider.

### `P2R0G19B_D11_CP3_LITERAL_REPRESENTATION`

- **Status:** `solved_code_identity`.
- **Claim:** D11 stores (10E.4), reads only (10E.5)--(10E.6), and a generic
  endpoint action needs (10E.7).  The all-distinct kernel witness proves the
  deployed reads are insufficient.

### `P2R0G19B_D11_CP3_EXACT_OWNER`

- **Status:** `closed_negative_generic`.
- **Claim:** the first-reset formula (10E.10) misses the exact Gaussian
  `H2-H2-H2` triangle and higher tail.  Full-birth D11 CP3 is not the exact
  synchronized Gaussian K3 owner; retain `delta3` in (10E.12)--(10E.13).

### `P2R0G19B_D11_CP3_THRESHOLD_JVP_RESIDENCY`

- **Status:** `closed_literal_reuse`; a new derivative evaluator is possible
  but not resident.
- **Claim:** D11 has value factors and the elementwise primitives for
  (10E.9), but no tangent bank, slot-typed JVP output, or future-costate
  lifetime overlap.

### `P2R0G19B_Q211_CP3_JVP_TARGET_RANK`

- **Status:** `solved_generic_no_go`; structured-rank exception open.
- **Claim:** the leaf coefficient has minimum common-direction rank
  `rank(D_alpha R D_beta)`, generically 1024, (10E.14)--(10E.15).

### `P2R0G19B_Q211_CP3_JVP_COST`

- **Status:** `closed_current_cap`.
- **Claim:** the four literal product families (10E.16), after removing the
  identity-q current births, cost 360 equivalents over L5--14 (180 in the
  `beta=gamma` special atom); (10E.19) is the only exact
  low-rank-R reopen.

### `P2R0G19B_Q211_BROAD_SANDWICH_FACTOR`

- **Status:** `solved_one_product`.
- **Claim:** (10E.20) is an exact one-GEMM realization; generic pair kernel
  rank prevents an exact lower-rank shortcut.

### `P2R0G19B_Q211_BROAD_SANDWICH_D11_FUSION`

- **Status:** `closed_literal_reuse`.
- **Claim:** no D11 paid product has both the same `R` and the query-specific
  RHS `D_beta Pplus D_gamma`.  Pair-kernel residence saves kernel formation,
  not the product.

### `P2R0G19B_Q211_BROAD_SANDWICH_CAP49`

- **Status:** `closed_D10`; conditional short-site arithmetic only.
- **Claim:** ten sites cost 10, so Base48+1 and full-AQ `40.02+10` both fail;
  the nominal r49 margin covers at most eight zero-threshold sites before all
  other costs, and does not solve biased L5--14.

### `P2R0G19B_Q211_D11_OLDSPAN_OWNERSHIP`

- **Status:** `open_response_schur`, with state-name ownership rejected.
- **Claim:** only the same-cut response-row test (10E.21) establishes old
  ownership.  D11's CP3 name, K21 read, and pairwise Q24 kernels do not.

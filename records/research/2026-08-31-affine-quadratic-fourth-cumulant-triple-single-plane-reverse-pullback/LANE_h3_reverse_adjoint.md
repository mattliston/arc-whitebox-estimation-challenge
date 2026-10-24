# H3 reverse adjoint

Durable pure-mathematics lane record. No numerical science runs are authorized during this derivation window.

Checkpoint derivations, corrections, terminal gates, and successor questions are appended below by the assigned agent.

## 1. Generic reverse-H3 contraction and the structural lower bound

Let `Sym` denote normalized orthogonal symmetrization.  For a symmetric cubic
coefficient `T` and unit vectors/rows `w_i`, the repeated-index hook plane is

\[
Q_W(T)_{ij}=T(w_i,w_i,w_j),\qquad
Q_W^*H=\operatorname{Sym}\sum_{ij}H_{ij}
 w_i\otimes w_i\otimes w_j .
\]

For an exact cubic tangent recurrence

\[
T_{l+1}=R_l(W_l^{\otimes3}T_l)+S_{l+1},
\]

the exact reverse recurrence is

\[
\Lambda_l=C_l+(W_l^\top)^{\otimes3}R_l^*\Lambda_{l+1}.
\]

This is an exact consumer identity, not a source construction.  A generic
three-leg response of a nonlinear reset is nonseparable, so writing the
contraction as a lazy arithmetic DAG does not by itself give an
`O(n^2)`-memory, `O(L n^3)` algorithm.  Such an algorithm requires a fixed
`O(n^2)` Markov state closed under the reset and primitives whose contractions
are each explicitly `O(n^3)`.  A CP birth of rank `n` at every layer grows to
rank `Theta(Ln)` and costs `Theta(L^2 n^3)` over depth unless births have bounded
lifetime, are consumed locally, or are compressed by a proved invariant
dictionary.  A noninvertible reset also means that an exact reverse pass needs
an `O(Ln^2)` tape; demanding memory independent of `L` forces checkpointing or
recomputation and can reintroduce the hidden `L^2` cost.

For a structured CP cubic

\[
T=\operatorname{Sym}\sum_{q=1}^R x_q\otimes y_q\otimes z_q,
\]

the hook plane is, up to the chosen normalized-Sym constant,

\[
(X\odot Y)Z^\top+(X\odot Z)Y^\top+(Y\odot Z)X^\top,
\]

and costs `O(n^2R)`.  The exact local reverse for a product-plane term
`P=(F_1 odot ... odot F_{k-1})F_k^T` is

\[
\bar F_s=(HF_k)\odot\!\prod_{t\ne s,k}F_t,
\qquad
\bar F_k=H^\top(F_1\odot\cdots\odot F_{k-1}).
\]

Thus `R=O(n)` is an `O(n^3)` structured GO, but it is not a generic-H3 result.

## 2. NCQC/Jordan core and the sibling product identity

For symmetric latent matrices `M_i in Sym_r`, the NCQC/Jordan cubic is

\[
T_{ijk}=\operatorname{tr}\{M_i(M_j\circ M_k)\},\qquad
M\circ N=(MN+NM)/2,
\]

and its hook is `T_iij=tr(M_i^2 M_j)`.  With `r^2=Theta(n)` it is directly
contractible in `O(n^3)` and stored in `O(n^2)`.  This fish/Jordan family is
already a structured CP3/G11 family; the consumer identity alone is not a new
source.

Define

\[
J_v(M)=\operatorname{Sym}(v\otimes M).
\]

The exact Wiener product identity is

\[
I_3(J_v(M))I_1(w)
=I_4(\operatorname{Sym}(v\otimes M\otimes w))
+3I_2(J_v(M)\otimes_1w),
\]

with the contraction

\[
3(J_v(M)\otimes_1w)
=(v^\top w)M+2\operatorname{Sym}(v\otimes Mw).
\]

For `B=vw^T`, write the linked siblings

\[
K_B(M)=\operatorname{Sym}(B\otimes M),\qquad
L_B(M)=\operatorname{tr}(B)M+BM+MB^\top.
\]

For a signed symmetric global parameter
`S=sum_q lambda_q v_qv_q^T`, the minimal one-stream family is

\[
C_i^{(3)}=J_h(M_i),\qquad
Q_i^{(4)}=K_S(M_i),\qquad
Q_i^{(2)}=L_S(M_i),
\]

where

\[
L_S(M)=\operatorname{tr}(S)M+SM+MS.
\]

It uses one propagated matrix stream plus `h in R^r` and
`S in Sym_r`; at `n=1024,r=32` this is only `32+528=560` global latent
directions.  Affine transport is exact because
`J_h(sum_i W_ai M_i)=sum_i W_ai J_h(M_i)` and likewise for `K_S,L_S`.
If `h` or `S` varies by unit, a generic dense mix is not closed: exact closure
requires a separately transported coefficient-weighted matrix stream.  Also,
multiple global vectors against the same `M` collapse linearly, e.g.
`sum_q J_vq(M)=J_sum(vq)(M)`; vector count alone does not create rank.

The family is reset-closed only if the nonlinear reset preserves the displayed
`J/K/L` subspace.  Otherwise, with reset leakages `eta_3,eta_4`, the omitted
endpoint value is bounded by the exact dual estimate

\[
|b_{reset}|\le\sum_l
(\|\Lambda_l^{(3)}\|_*\|\eta_l^{(3)}\|
 +\|\Lambda_l^{(4)}\|_*\|\eta_l^{(4)}\|).
\]

## 3. H2-shadow identifiability and latent-rank facts

For fixed `M`, let `mathscr L_M(S)={L_S(M_i)}_i`.  Its exact adjoint is

\[
L_M^*(Y)=\langle Y,M\rangle_F I+YM+MY
\]

for symmetric `M,Y`.  If `I` lies in `span{M_i}`, the full latent H2 shadow is
injective.  Indeed, for coefficients with `sum_i c_iM_i=I`, set
`Y=sum_i c_iL_S(M_i)=L_S(I)`.  Then

\[
S={1\over2}\left(Y-{\operatorname{tr}Y\over r+2}I\right).
\]

This identifies the latent `S`; it does not prove that `K_S(M_i)` equals any
external unit-index cumulant.  External covariance/H2 Gram data alone are
strictly weaker: `Cov(q_i,q_j)=2tr(M_iM_j)` does not determine quartic trace
words.  For example, `diag(1,0)` and `diag(1/sqrt(2),1/sqrt(2))` have the same
Frobenius norm and hence the same one-unit quadratic variance, but fourth trace
`1` versus `1/2`.

For the normalized symmetric cubic map,

\[
\|J_h(M)\|^2={1\over3}\|h\|^2\|M\|_F^2
+{2\over3}h^\top M^2h.
\]

Consequently its Gram operator over units is

\[
\mathcal J_M^*\mathcal J_M
=\sum_i\left({\|M_i\|_F^2\over3}I+{2M_i^2\over3}\right),
\]

so `h` is stably identifiable whenever the stream is nonzero.  Nevertheless,
the signed-H3 representable subspace has dimension at most `r`.

For any symmetric fourth-order consumer `R`, normalized symmetrization gives

\[
\langle R,K_S(M)\rangle
=\langle R\otimes_2M,S\rangle_F.
\]

For structured fourth-order factors, an exact materialization-free identity is

\[
\langle K_S(M),K_T(N)\rangle
={1\over6}\{\langle S,T\rangle\langle M,N\rangle
+\langle S,N\rangle\langle M,T\rangle
+4\operatorname{tr}(NSTM)\}.
\]

The generic coordinate dimensions at `n=1024` are
`dim H3_distinct=178,433,024` and `dim q4(211)=535,299,072`; hence a 560-direction
global family has no generic exactness claim.  This is a worst-case rank fact,
not a claim that the campaign residual is isotropic.

## 4. Complete external K31 value: exact one-stream consumer

Use the external endpoint convention supplied by the direct-diagram lane.  At
layer `l`, with reverse variance plane `B`, action planes `G31,G13`, and the
appropriate weights,

\[
H=W_{l+1}B W_{l+1}^\top,
\quad Q^{31}={1\over6}\{H\odot G31+(H\odot G13)^\top\},
\quad P=W_l(Q^{31})^\top.
\]

For an external symmetric fourth cumulant `T`, define

\[
\Xi_{ijkl}=\sum_aW_{ia}W_{ja}W_{ka}P_{la},\qquad
\mathcal A_{B,l}[T]=\langle\operatorname{Sym}\Xi,T\rangle.
\]

For centered quadratic-chaos units
`q_i=z^T M_i z-tr(M_i)`, the complete external cumulant is

\[
T^{ext}_{ijkl}=16\{\langle M_i\circ M_j,M_k\circ M_l\rangle
+\langle M_i\circ M_k,M_j\circ M_l\rangle
+\langle M_i\circ M_l,M_j\circ M_k\rangle\}.
\]

Set

\[
N_a=\sum_iW_{ia}M_i,\qquad R_a=\sum_iP_{ia}M_i.
\]

The three symmetric pairings contract equally, giving the exact complete
functional

\[
\boxed{\mathcal A_{B,l}[T^{ext}]
=48\sum_a\operatorname{tr}(N_a^3R_a).}
\]

No `Sym^4(n)` object, `q4(211)` plane, or pair-of-unit Jordan bank is formed.
The exact local reverse is

\[
\bar N_a=48(N_a^2R_a+N_aR_aN_a+R_aN_a^2),
\qquad \bar R_a=48N_a^3,
\]

and hence the source-stream injection is

\[
\bar M_i=48\sum_a\left[
W_{ia}(N_a^2R_a+N_aR_aN_a+R_aN_a^2)+P_{ia}N_a^3
\right].
\]

This is the exact two-pass adjoint for a joint refit of the NCQC matrices.

Let `s=r(r+1)/2`.  Do not form the dense matrix `P`.  Associativity gives the
exact factorization

\[
R=P^\top M=Q^{31}W^\top M=Q^{31}N.
\]

Packed mixing first by `W^T` and then by `Q31` costs two rectangular matrix
multiplies, each `s/n` full-`n^3` GEMM equivalents.  Two local `r x r`
products (`N^2` and `NR`) evaluate the trace word and cost
`2r^3/n^2` equivalents.  Thus, from the pre-weight stream, the complete
consumer costs

\[
g_{31}=2s/n+2r^3/n^2=1.09375
\]

per active layer at `n=1024,r=32`, or `15.3125` over 14 layers.  If post-weight
`N` is already the carried primal state, the marginal scalar consumer is only
`s/n+2r^3/n^2=0.578125` per layer, or `8.09375` over 14.  Reusing the existing
variance conjugation remains preferable, but even recomputing its two full
GEMMs per layer gives `43.3125` total from the pre-weight state, still below
the 49-GEMM cap.  Forming `P` was an avoidable one-full-GEMM-per-layer
overestimate.

This result is a **consumer GO**.  It supplies a correction only if the carried
`M_i` (or a jointly refitted change in them) supplies the missing external
cumulant.

## 5. External/latent type separation and a sharp common-S no-go

`K_S(M_i)` lives in
`R^n tensor Sym^4(R^r)`, whereas `T_ext` lives in `Sym^4(R^n)`.  There is no
canonical projection between them.  A generative decoder `E` must first be
derived (for example from explicit H1 coefficients and the Wiener product
law).  Once such a decoder is fixed, define

\[
\mathcal U_M=\operatorname{im}\{S\mapsto
\mathcal E((K_S(M_i))_i)\},\quad
T_\perp=(I-\Pi_{\mathcal U_M})T^{ext},
\]

and the exact missed endpoint scalar is

\[
\eta_B=\langle\operatorname{Sym}\Xi,T_\perp\rangle.
\]

Without `E`, equating a latent `211` Wick sector with external unit-index
`q4(211)` is ill-typed.  Matching one scalar mask by rescaling `S` is also not
transport: exactness must hold over the family of layer/net reverse masks.

There is a directly testable necessary H2-shadow condition for trying to use a
single global sibling `S` to replace the complete external cubic response.
Since the true functional is

\[
48\sum_a\langle R_a,N_a^3\rangle_F,
\]

the one-global-S shadow can match all independently variable reverse matrices
only if

\[
\boxed{N_a^3=L_S(N_a)=\operatorname{tr}(S)N_a+SN_a+N_aS
\quad\hbox{for every }a.}
\]

The exact least-squares residual and normal equation are

\[
\rho_{31}^2=\min_S\sum_a\|N_a^3-L_S(N_a)\|_F^2,
\]

\[
\sum_aL_{N_a}^*\{L_S(N_a)-N_a^3\}=0.
\]

Moreover,

\[
\sup_{\sum_a\|R_a\|_F^2\le1}
\left|48\sum_a\langle R_a,N_a^3-L_S(N_a)\rangle\right|
=48\rho_{31}.
\]

This is a sharp external/latent projection residual for the H2-shadow
surrogate.  Exact zero residual is exceptional.  Commuting the displayed
identity with `N_a` gives `[S,N_a^2]=0`.  If the `N_a` have generic simple
absolute spectra, `S` must share their eigenbasis.  If two generic `N_a` do
not commute, their common commutant is scalar, forcing every
`N_a^2` to be the same scalar multiple of `I`.  In the merely commuting case,
with eigenvalues `lambda_ap`, exactness requires

\[
\operatorname{tr}(S)+2s_p=\lambda_{ap}^2
\]

for every active `a,p` with nonzero eigenvalue: squared eigenvalues in each
latent channel must be independent of `a`.  Failure of this compatibility is
a symbolic NO-GO before any run.  The linked H4 sibling cannot be tuned
independently to repair a failed H2-shadow test because both are fixed by the
same `S` in the exact product identity.

## 6. Raw-improvement and deployment gates

The campaign's quoted full `q4(211)` ceiling is `1.0313e-9` MSE and the minimum
useful quartic gain is `2.86e-10`.  Therefore a projection-only provider must
capture at least

\[
2.86/10.313=0.27732
\]

of the q211 oracle MSE gain, equivalently correlation at least `0.5266` when
ordinary orthogonal projection assumptions apply.  With exact variance and
the quoted combined H3+quartic ceiling `5.4013e-9`, raw below `1e-8` requires
combined unexplained value below `0.7453e-9`, i.e. explained fraction above
`0.8620` or joint correlation above `0.92845` under the same projection metric.

For signed H3, let
`C_M={ (J_h(M_i))_i : h in R^r }`.  The family can carry both signs, but its
dimension is only `r=32`; its exact best error is the target's projection
residual onto `C_M`.  For quartic supply, use either the explicitly typed
decoder image `U_M` above or the tangent image of a joint NCQC refit
`im(D T_ext[M])`.  The conditional GO theorem is:

1. the reset/provenance leakage satisfies the end-to-end `0.7453e-9` residual
   budget;
2. the signed-H3 target projection plus the typed external-quartic projection
   has joint correlation at least `0.92845`;
3. the quartic component alone gains at least `2.86e-10`;
4. the contraction is associated as `R=Q(W^TM)` rather than by materializing
   `P=WQ^T`; reuse of `H` then preserves substantial headroom below 49 GEMMs.

Absent an external decoder, or if the common-S compatibility residual above
already exceeds the value budget, the global sibling family is a provider
NO-GO even though the direct NCQC K31 contraction is a consumer GO.

## Current next question

Derive the actual generative decoder from the deployed H1/H2/H3 coefficient
law, if it exists, and evaluate symbolically whether its pullback image is the
NCQC tangent image or only the 528-dimensional common-S shadow.  The decisive
analytic certificate is a bound on the endpoint-weighted projection residual,
not another single-net scalar fit.

## 7. Pivot: the exact affine--quadratic external decoder

The decoder exists for a precisely typed affine--quadratic generative law.  Let

\[
X_i=a_i^\top z+z^\top M_i z-\operatorname{tr}M_i,
\qquad z\sim N(0,I_r),
\]

and let `A(theta)=sum_i theta_iM_i` and
`h(theta)=sum_i theta_i a_i`.  The exact joint cumulant-generating function is

\[
K(\theta)=-\operatorname{tr}A
-{1\over2}\log\det(I-2A)
+{1\over2}h^\top(I-2A)^{-1}h.
\]

Its homogeneous fourth-order term is

\[
K_4(\theta)=2\operatorname{tr}(A^4)+2h^\top A^2h.
\]

For projected variables

\[
U_a=w_a^\top X=(u_a,N_a),\qquad
V_a=p_a^\top X=(r_a,R_a),
\]

where

\[
u_a=\sum_iW_{ia}a_i,\quad N_a=\sum_iW_{ia}M_i,
\quad r_a=\sum_iP_{ia}a_i,\quad R_a=\sum_iP_{ia}M_i,
\]

coefficient extraction gives the exact complete action

\[
\boxed{\kappa_4(U_a,U_a,U_a,V_a)
=48\operatorname{tr}(N_a^3R_a)
+24\{u_a^\top N_aR_au_a+u_a^\top N_a^2r_a\}.}
\]

Thus the latent/external type obstruction disappears for this law: no H4
tensor or arbitrary decoder is needed.  It does **not** disappear for a naked
global `h`.  The exact state is one per-unit H1 vector bank `a_i` propagated in
the same latent frame as the one per-unit H2/NCQC matrix bank `M_i`.  A global
`h` is only the special case in which all `a_i` are collinear (with any required
unit scalar coefficients carried separately).

The link to the sibling `L/K` identity is exact and reveals why one fixed `S`
was too small.  At each endpoint the effective sibling parameters are
`S_a=u_au_a^T` and `T_a=Sym(u_ar_a^T)`, which vary with `a`.  Specifically,

\[
24\{u^\top NRu+u^\top N^2r\}
=12\big[\langle R,L_{uu^\top}(N)\rangle
-\|u\|^2\operatorname{tr}(RN)
+\langle N,L_{\operatorname{Sym}(ur^\top)}(N)\rangle
-(u^\top r)\|N\|_F^2\big].
\]

The subtracted trace products are precisely the disconnected H2/covariance
shadows.  Hence endpoint-conditioned rank-one siblings close the AQ decoder,
while the earlier common-global-S compatibility theorem remains a valid
NO-GO for a fixed `S` and for any chaos beyond the affine--quadratic law.

The covariance/H2 shadow is

\[
\operatorname{Var}U=\|u\|^2+2\operatorname{tr}(N^2),\qquad
\operatorname{Cov}(U,V)=u^\top r+2\operatorname{tr}(NR),
\]

and the raw fourth moment obeys

\[
E[U^3V]=\kappa_4(U,U,U,V)
+3\operatorname{Var}(U)\operatorname{Cov}(U,V).
\]

Therefore H2 data identify the disconnected subtraction exactly.  They do not
identify the connected AQ term unless the relative latent orientation of
`a_i` and `M_i` is retained.  For example, with fixed
`N=R=diag(1,0)`, vectors `u=e_1` and `u=e_2` have the same norm and therefore
the same external H2 contribution, but `u^TNRu` is respectively `1` and `0`.

## 8. Exact AQ reverse and in-place resource census

For one endpoint write

\[
F=48\operatorname{tr}(N^3R)+24\{u^\top NRu+u^\top N^2r\}.
\]

With `Sym(X)=(X+X^T)/2`, the exact symmetric-state derivatives are

\[
\bar R=48N^3+24\operatorname{Sym}(Nuu^\top),
\qquad \bar r=24N^2u,
\]

\[
\bar u=24\{(NR+RN)u+N^2r\},
\]

\[
\bar N=48(N^2R+NRN+RN^2)
+24\big\{\operatorname{Sym}(Ruu^\top)
+\operatorname{Sym}[u(Nr)^\top]
+\operatorname{Sym}[(Nu)r^\top]\big\}.
\]

The exact stream pullback is

\[
\bar M_i=\sum_a(W_{ia}\bar N_a+P_{ia}\bar R_a),\qquad
\bar a_i=\sum_a(W_{ia}\bar u_a+P_{ia}\bar r_a).
\]

Concatenate the packed matrix and vector states into a width
`d=s+r=560` bank.  The exact scalar path is one rectangular `W^T` mix and one
rectangular `Q31` mix.  The two local latent products used above are shared by
the AQ cross terms.  Thus, from the pre-weight state,

\[
g_{AQ31}=2d/n+2r^3/n^2=1.15625
\]

per layer, or `16.1875` over 14 layers.  If post-weight `N,u` are already
carried, the marginal scalar consumption is
`d/n+2r^3/n^2=0.609375` per layer, or `8.53125` over 14.  Even recomputing the
two-GEMM variance conjugation gives `44.1875` total from the pre-weight state,
below 49 with `4.8125` equivalents of headroom.  Reuse is still the safer
deployment premise because other uncounted setup must fit in that headroom.

The exact state adjoint also stays factored: reverse `R=QN,r=Qu` with
rectangular `Q^T` multiplies and then reverse `N=W^TM,u=W^Ta` with rectangular
`W` multiplies.  No dense `P` is required in either direction.

For a particularly transparent fused scalar form, define

\[
F_a=48N_a^3+24\operatorname{Sym}(N_au_au_a^\top),
\qquad b_a=24N_a^2u_a.
\]

Then

\[
F_{AQ}=\sum_a\{\langle F_a,(QN)_a\rangle_F
+b_a^\top(Qu)_a\}
=\operatorname{tr}(F_{pack}^\top QN_{pack})+b^\top Qu.
\]

This proves directly that the endpoint plane `Q` remains generically dense;
the saving is rectangular right-hand width, not a rank-one assumption.
Residualizing a new source against an old source changes `D` in
`<Q,D-D_old>` and does not reduce the rank of `Q`.

The same state produces a signed external cubic action.  The homogeneous cubic
CGF term is `(4/3)tr(A^3)+h^TAh`, hence

\[
\boxed{\kappa_3(U,U,V)
=8\operatorname{tr}(N^2R)+2u^\top Ru+4u^\top Nr.}
\]

No positivity constraint prevents either sign.  State factors are shared with
K31, but an H3 reverse mask is not automatically the K31 mask.  Even for a
distinct `Q3`, however, the scalar contraction applies it directly to the
width-`d` state; it does not form `P3=WQ3^T`.  With post-weight state carried,
the marginal mask application is at most `d/n=0.546875` per layer, and local
`N^2` can be reused from K31.  An independent full-GEMM cost is incurred only
if constructing the H3 mask itself genuinely requires one; it is not implied
by the external adjoint factorization.

Typing caveat: this is a signed **external third cumulant** supplied by the
affine--quadratic law.  The variable `X_i=a_i^Tz+q_Mi(z)` has no independent
Wiener-chaos degree-3 coefficient of its own.  If the campaign target called
"H3" is an actual degree-3 chaos innovation rather than the external K3
functional, it remains part of the beyond-AQ residual and needs a separately
typed H3 provider even though its external scalar consumer may share state.

There is an exact intermediate interpretation that matters for a closure whose
state is built from pair products.  Although `U` itself has only Wiener H1 and
H2 under the AQ law, its product has a signed H3 component.  The complete
product coefficients are

\[
[U^2]_0=\|u\|^2+2\operatorname{tr}N^2,\quad [U^2]_1=4Nu,
\]
\[
[U^2]_2=u\otimes u+4N^2,\quad [U^2]_3=2J_u(N),
\quad [U^2]_4=K_N(N),
\]

and, for `V=I1(r)+I2(R)`,

\[
[UV]_0=u^\top r+2\operatorname{tr}(NR),\quad
[UV]_1=2(Ru+Nr),
\]
\[
[UV]_2=\operatorname{Sym}(u\otimes r)+4\operatorname{Sym}(NR),
\]
\[
[UV]_3=J_u(R)+J_r(N),\qquad [UV]_4=K_N(R).
\]

These follow directly from the Wiener product formula and independently
recover the sibling structure.  Therefore a target H3 **of the pair-product
closure** is supplied locally by the shared vector/matrix state; an independent
H3 coefficient of `X` is not.

The same expansion gives an exact channel attribution.  Define
`T=tr(N^3R)`, `A=u^TNRu`, `B=u^TN^2r`,
`C1=||u||^2 tr(NR)`, `C2=(u^Tr)tr(N^2)`, and
`D=tr(N^2)tr(NR)`.  In

\[
\kappa_4(U,U,U,V)=-2[U^2]_0[UV]_0
+\sum_{m=1}^4m!\langle[U^2]_m,[UV]_m\rangle,
\]

the degree contributions are

\[
m=1:8(A+B),
\]
\[
m=2:2\|u\|^2(u^\top r)+8(A+B)+32T,
\]
\[
m=3:4(C1+C2)+8(A+B),
\]
\[
m=4:8D+16T,
\]
\[
-2m_0:-2\|u\|^2(u^\top r)-4C1-4C2-8D.
\]

All disconnected terms cancel, leaving `24(A+B)+48T`.  Hence the AQ
cross-quartic action is divided equally among the product H1, H2, and H3
channels, while the pure NCQC trace action is divided H2:H4 as `32:16`.  A
patch that carries only the H3 sibling can capture at most one third of the AQ
cross term even with exact coefficients; an H4-only patch carries only one
third of the pure-NCQC term.  The full linked closure is structurally required.

## 9. Exact beyond-AQ residual and a norm certificate

For arbitrary centered deployed layer variables the convention-free complete
action is

\[
F_{true}=\sum_a\kappa_4(U_a,U_a,U_a,V_a)
=\sum_a\{E[U_a^3V_a]-3E[U_a^2]E[U_aV_a]\}.
\]

The exact external/AQ residual is therefore

\[
\eta_{>AQ}=F_{true}-\sum_aF_{AQ,a}.
\]

This definition requires neither an external fourth-order tensor nor a decoder
convention.  A useful analytic bound follows by writing
`U=U_0+epsilon`, `V=V_0+delta`, where `U_0,V_0` are the jointly framed AQ
parts.  Multilinearity of the fourth cumulant and Holder's inequality give,
for centered arguments,

\[
|\kappa_4(A,B,C,D)|\le4\|A\|_4\|B\|_4\|C\|_4\|D\|_4.
\]

Consequently, with
`u0=||U_0||_4`, `e=||epsilon||_4`, `v0=||V_0||_4`, and
`d=||delta||_4`,

\[
|\eta_{>AQ,a}|\le4\big(
[(u0+e)^3-u0^3]v0+(u0+e)^3d\big).
\]

Summing this over endpoint/layer actions gives a certified high-chaos/reset
leakage budget.  It is conservative but exact as an upper bound.  Promotion
still requires the endpoint-weighted AQ projection to deliver at least
`2.86e-10` quartic MSE gain and the combined H3+quartic unexplained value to
remain below `0.7453e-9`; algebraic representability and billing alone do not
establish those empirical alignment inequalities.

There is no replacement certificate based only on H2/L2 residual energy.  For
`Z` standard Gaussian, let `p=P(Z>t)` and

\[
e={\sigma\over\sqrt{p(1-p)}}(1_{\{Z>t\}}-p).
\]

Then `e` is centered and `||e||_2=sigma`, while
`E[e^4]` grows as `sigma^4/p` as `p` tends to zero.  Thus even within functions
of a Gaussian, a fixed arbitrarily small covariance residual can have
arbitrarily large fourth action.  A deployed beyond-AQ certificate needs L4,
sub-Gaussian, or degree-controlled hypercontractive information; H2 shadow
alone cannot certify the `2.86e-10` quartic gate.

## 10. Novelty closure against the existing D11 action

Cross-lane reconciliation identifies the exact complete AQ same-site scalar
above with the total action already represented by the old D11 lineage.  This
changes its promotion status but not the algebra: it is an exact cheaper
consumer identity, not a new raw direction.

Let `S_old` be the banked D11 feature span, let `f_AQ` be the vector of fused AQ
actions over the grading corpus, and let

\[
f_\perp=(I-\Pi_{S_{old}})f_{AQ}.
\]

For a target residual `d`, the best incremental one-feature MSE gain is

\[
\Delta_{new}={\langle d,f_\perp\rangle^2\over\|f_\perp\|^2}
\]

when `f_perp` is nonzero, and exactly zero when it is zero.  Therefore the
proved identity `f_AQ in S_old` implies

\[
\boxed{\Delta_{new}=0<2.86\times10^{-10}.}
\]

This is a raw-provider NO-GO for statically adding the one-stream AQ scalar,
irrespective of its low GEMM count.  Novel value can arise only by changing the
provider matrices/vectors through a joint refit so that the resulting action
leaves the old span, or by supplying the beyond-AQ defect
`chi_true-chi_AQ`.  The latter is, by definition, not supplied by the AQ
family and needs separately typed higher-chaos/tail state.

## 11. Signed-H3 cannot be identified from an external H2 shadow

Write a post-AQ defect as

\[
X_i=X_i^{AQ}+I_3(Z_i),\qquad Z_i\in\operatorname{Sym}^3(\mathbb R^r).
\]

Wiener orthogonality gives its covariance/H2 contribution

\[
\Delta C_{ij}=3!\langle Z_i,Z_j\rangle.
\]

This Gram is invariant under the global sign flip `Z_i -> -Z_i` (and under a
common orthogonal rotation in cubic-feature space).  In contrast, its leading
external fourth-cumulant action with three H1 legs is

\[
\kappa_4(I_3(Z_i),I_1(a_j),I_1(a_k),I_1(a_l))
=3!\langle Z_i,\operatorname{Sym}(a_j\otimes a_k\otimes a_l)\rangle,
\]

which flips sign.  Therefore no external H2/covariance-only carrier can
identify the signed H3 correction, even with a perfect covariance plane.

An internal linked shadow `Z_i tensor_1 a` can break the ambiguity if known H1
probe vectors span the latent space, but generic tomography then requires the
independent cubic state.  At `r=32`,
`dim Sym^3(R^r)=binom(34,3)=5984` per unit, rather than the 528 matrix
coordinates of the AQ stream.  The old one-stream AQ/D11 state cannot encode a
generic signed post-AQ H3 defect.

## 12. Fixed-state dictionary rank gate for the novel H3 defect (r=32 checkpoint)

At `n=1024,r=32`, let `p=dim Sym^3(R^r)=5984`.  The base one-stream AQ state
uses

\[
n\{r(r+1)/2+r\}=1024(528+32)=573,440
\]

scalars.  Under a literal `n^2=1,048,576` fixed-state ceiling, 475,136 scalars
remain.  A generic shared dictionary

\[
Z=CD,\qquad C\in\mathbb R^{n\times k},\quad
D\in\mathbb R^{k\times p},
\]

therefore permits at most

\[
k\le\left\lfloor{475136\over1024+5984}\right\rfloor=67.
\]

A much more structured symmetric-CP dictionary
`Z_i=sum_q C_iq v_q^tensor3` stores `k(n+r)` scalars and permits at most

\[
k\le\left\lfloor{475136\over1024+32}\right\rfloor=449.
\]

Let `sigma_j` be the singular values of the target cubic-defect matrix after
whitening by the endpoint/campaign value metric.  Eckart--Young gives the exact
generic-dictionary gate

\[
\sum_{j>k}\sigma_j^2\le0.1380\sum_j\sigma_j^2,
\]

because the combined raw target requires at least 86.2% explained value.  Thus
the generic dictionary is a NO-GO unless the top 67 modes pass this gate; the
CP dictionary requires an analogous best-449 rank-one-atom certificate.  Its
leading H3/H1/H1/H1 consumer is materialization-free:

\[
6\sum_q z_{aq}(v_q^\top u_a)^3,
\]

and its linked H2 shadow is

\[
\sum_q z_{aq}(v_q^\top u_a)v_qv_q^\top.
\]

Only the coefficient bank `C` must be transported, but reset closure is exact
only if the same atoms are invariant; otherwise the discarded dictionary tail
must be charged to the `0.7453e-9` residual budget.

## 13. Exact novel K31 tangent supplied by an independent H3 defect

The previous section gives a dictionary, but novelty requires its external
action after the D11/AQ span is removed.  Let

\[
U=U_0+I_3(Z),\qquad V=V_0+I_3(Y),
\]

where `U_0=I1(u)+I2(N)` and `V_0=I1(r)+I2(R)`.  The exact first variation is

\[
\delta\kappa_{31}
=3\kappa_4(I_3(Z),U_0,U_0,V_0)
+\kappa_4(U_0,U_0,U_0,I_3(Y)).
\]

Because H3 is orthogonal to AQ H1/H2, every covariance-subtraction term has
zero first variation.  Parity leaves only H1/H1/H1 and H1/H2/H2 choices.  The
needed exact product identity is

\[
E[I_3(Z)I_1(a)I_2(M)I_2(R)]
=12\langle Z,
J_{Ra}(M)+J_{Ma}(R)+2J_a(M\circ R)\rangle.
\]

It follows that the coefficient paired with the `U`-side cubic perturbation is

\[
\begin{aligned}
G_Z={}&18\operatorname{Sym}(u\otimes u\otimes r)\\
&+72\{J_{Ru}(N)+J_{Nu}(R)+2J_u(N\circ R)
+J_{Nr}(N)+J_r(N^2)\},
\end{aligned}
\]

while the coefficient paired with the `V`-side perturbation is

\[
G_Y=6u^{\otimes3}+72\{J_{Nu}(N)+J_u(N^2)\}.
\]

Thus

\[
\boxed{\delta\kappa_{31}=\langle Z,G_Z\rangle
+\langle Y,G_Y\rangle.}
\]

For a shared symmetric-CP dictionary
`Z_i=sum_q c_iq v_q^tensor3`, set `z=W^Tc` and `y=Qz`.  For one atom define

\[
t=v^\top u,\ s=v^\top r,\ \alpha=v^\top Nv,\ \beta=v^\top Rv,
\quad n_v=Nv,\ r_v=Rv.
\]

Then the exact scalar consumers are

\[
\begin{aligned}
A_q={}&18t^2s+72\{(r_v^\top u)\alpha
+(n_v^\top u)\beta+2t(n_v^\top r_v)\\
&\hspace{38mm}+(n_v^\top r)\alpha+s\|n_v\|^2\},
\end{aligned}
\]

\[
B_q=6t^3+72\{(n_v^\top u)\alpha+t\|n_v\|^2\},
\]

and

\[
\delta F=\sum_{a,q}(z_{aq}A_{aq}+y_{aq}B_{aq}).
\]

Only `Nv` and `Rv` are needed locally.  The coefficient-bank `W/Q` mixes cost
`2k/n` full-GEMM equivalents per layer and the two latent matrix/dictionary
applications cost another `2k/n`, for total `4k/n`.  At the memory-bound
`k=449`, this is `1.75390625` per layer.  Added to the pre-weight AQ scalar
path, the 14-layer census is about `40.74`, below 49 if the existing endpoint
`H/Q` action is reused.

This is the first typed direction in this lane that can be both a signed H3
source and a novel external quartic source after the complete D11/AQ action is
removed.  It is a conditional GO, not a promotion: its Schur-residualized
quartic feature must gain at least `2.86e-10`, and the dictionary/reset tail
must satisfy the combined `0.7453e-9` unexplained-value gate.

The displayed feature is the exact first derivative, not the full finite-`Z`
cumulant.  Let
`u4=||U_0||_4`, `v4=||V_0||_4`, `e=||I3(Z)||_4`, and
`d=||I3(Y)||_4`.  After removing both AQ and the exact linear tangent,
multilinearity and the fourth-cumulant Holder bound give

\[
|R_{\ge2}|\le4\big[
(3u4\,e^2+e^3)v4
+(3u4^2e+3u4\,e^2+e^3)d\big].
\]

Unlike an arbitrary beyond-AQ remainder, a pure third-chaos perturbation has a
hypercontractive L4 certificate:

\[
\|I_3(Z)\|_4\le3^{3/2}\|I_3(Z)\|_2
=3^{3/2}\sqrt{3!}\,\|Z\|_F,
\]

and likewise for `Y`.  Thus its covariance energy controls the magnitude of
the omitted nonlinear diagrams once the signed orientation is separately
supplied.  The bound must fit the `0.7453e-9` budget, or the quadratic and
higher `Z` diagrams must also be consumed.

Normalization warning: `Z` here is the standard Wiener kernel in `I3(Z)`.
If the closure stores the derivative/Hermite tensor `T=E[D^3f]`, then its
Wiener kernel is `Z=T/3!=T/6`; every displayed linear `Z/Y` action must be
multiplied by `1/6`.  In a half-quadratic convention `A=2N`, the matrix factors
must likewise be rescaled before comparing constants.

## 14. Portfolio stacking and the exact capacity gate

Let `P0` project onto the old D11/Q24 span and let `Phi_C,Phi_H,Phi_K` denote,
respectively, the independent cubic dictionary, a separate structured
all-distinct-H3 lane, and a strict-depth K31-residual lane.  Define the
residualized subspaces

\[
U_C=(I-P_0)\operatorname{im}\Phi_C,
\quad U_H=(I-P_0)\operatorname{im}\Phi_H,
\quad U_K=(I-P_0)\operatorname{im}\Phi_K.
\]

The only non-double-counted joint gain is

\[
\Delta_{joint}=d^\top P_{[U_C,U_H,U_K]}d.
\]

For example, the marginal cubic-dictionary value after the other two lanes is

\[
\Delta_{C\mid H,K}
=d^\top\{P_{[U_H,U_K,U_C]}-P_{[U_H,U_K]}\}d.
\]

Standalone gains must not be added.  The cubic dictionary dominates the
structured-H3 lane only if `U_H subset U_C` under the same endpoint action and
its deployed cost is no larger.  A useful conditional containment follows
from polarization:

\[
\operatorname{Sym}(v\otimes e\otimes e)
={(v+e)^{\otimes3}+(v-e)^{\otimes3}-2v^{\otimes3}\over6}.
\]

Thus a Jordan atom `J_v(M)` with a rank-`r` matrix in a common eigenbasis uses
at most `2r+1` symmetric-CP cubes.  This containment generally fails when the
matrix eigenbases vary by unit/layer.  A strict-depth K31 residual is not, in
general, contained in the linear H3 tangent merely because it uses the same
endpoint `Q`; source overlap must be decided by the joint Schur projection.

For extra deployed per-layer costs `g_H,g_K`, the CP dictionary's exact cost
cap from the pre-weight AQ state is

\[
k\le\left\lfloor{n\over4}
(3.5-1.15625-g_H-g_K)\right\rfloor.
\]

At `k=449`, only `0.589844` full-GEMM equivalents per layer remain under the
49/14 ceiling; with post-weight AQ state already carried, the remainder is
`1.136719` per layer.  These presume the endpoint `H/Q` action is already
exposed.  The generic singular-value tail is an exact capacity test for an
unconstrained rank-`k` dictionary and only a necessary lower bound for a
symmetric-CP dictionary; CP promotion needs its actual best `k`-atom error in
the endpoint-whitened metric.

## 15. Exact reset provenance for the independent cubic kernel

The dictionary coefficients need not be free or truth-fed.  For a local
affine--quadratic preactivation

\[
g(z)=b+a^\top z+z^\top Mz-\operatorname{tr}M,
\qquad f(z)=\operatorname{ReLU}(g(z)),
\]

Gaussian integration by parts and distributional differentiation give

\[
T:=E[D_z^3f]
=E\left[\delta'(g)(a+2Mz)^{\otimes3}
+6\delta(g)J_{a+2Mz}(M)\right],
\qquad Z=T/3!.
\]

This is an exact signed reset-local cubic source.  It is collective rank one
when `M=0`, but a nonzero identity-specific `M` generates additional boundary
directions.

Equivalently, with `v=a+2Mz`, define
`mu_3(x)=p_g(x)E[v^tensor3|g=x]` and
`mu_J(x)=p_g(x)E[J_v(M)|g=x]`.  Distributional integration gives the geometric
boundary law

\[
T=-\mu_3'(0)+6\mu_J(0).
\]

The signed noncollective term is therefore the derivative of the conditional
gradient-cube density at the ReLU boundary, while its Jordan sibling is the
conditional boundary-curvature term.

There is a closed one-dimensional Fourier/resolvent representation.  Put

\[
K_t=I-2itM,\quad m_t=K_t^{-1}a,
\quad C_t=4MK_t^{-1}M,
\]

\[
\phi(t)=e^{it(b-\operatorname{tr}M)}\det(K_t)^{-1/2}
\exp\{-\tfrac12t^2a^\top K_t^{-1}a\}.
\]

Under the usual nondegeneracy needed for the density at zero (otherwise in the
tempered-distribution sense),

\[
T={1\over2\pi}\int_{\mathbb R}\phi(t)
\left[it\{m_t^{\otimes3}+3J_{m_t}(C_t)\}
+6J_{m_t}(M)\right]dt.
\]

Projection onto a fixed CP atom is scalar and does not materialize `T`:

\[
\begin{aligned}
b_{iq}:=\langle T_i,v_q^{\otimes3}\rangle
={1\over2\pi}\int\phi_i(t)\big[&it\{(v_q^\top m_t)^3
+3(v_q^\top m_t)(v_q^\top C_tv_q)\}\\
&+6(v_q^\top m_t)(v_q^\top Mv_q)\big]dt.
\end{aligned}
\]

For dictionary Gram
`G_pq=<v_p^tensor3,v_q^tensor3>=(v_p^Tv_q)^3`, the coefficients
`c_i=G^dagger b_i` are the exact orthogonal projection of the reset kernel.
This supplies the sign and provenance information that covariance cannot.
Any finite quadrature or finite dictionary remains an approximation, so its
analytic quadrature error and the orthogonal tensor tail must be included in
the raw residual gate.

The complete reset-local Fourier action against the post-AQ K31 adjoint is
also tensor-free.  Every query term is Jordan because
`Sym(u tensor u tensor r)=J_r(uu^T)` and `u^tensor3=J_u(uu^T)`.  The universal
pairing is

\[
\langle J_v(A),J_\mu(B)\rangle
={1\over3}\left[(v^\top\mu)\operatorname{tr}(AB)
+v^\top(AB+BA)\mu\right].
\]

Hence, once the tilted `mu_t,Sigma_t` and the local query matrices are
available, each Fourier-node/query contraction is `O(r^2)` and no H3 tensor is
formed.  This does not make generic source acquisition free: producing
`mu_t,Sigma_t` from a dense `M` needs a solve/eigendecomposition unless its
spectral basis is reused, and pulling unit-varying Jordan query nodes through a
dense `W/Q` still incurs the generic source/costate contraction unless a
bounded shared rank is proved.

The first perturbative term explains why a cheap sibling can already be owned
without exhausting the exact boundary source.  Let
`sigma=||a||`, `e=a/sigma`, `alpha=b/sigma`, and `m=e^TMe`.  Expansion around
`M=0` gives

\[
T_0=-\sigma\alpha\phi(\alpha)e^{\otimes3},
\]

\[
\delta T^{(1)}=\phi(\alpha)\big[
mH_4(\alpha)e^{\otimes3}
+6H_2(\alpha)\operatorname{Sym}(e^{\otimes2}\otimes Me)
+6J_e(M)\big].
\]

Here the scalar identities are

\[
E[(x^2-1)\delta''(b+\sigma x)]
={H_4(\alpha)\phi(\alpha)\over\sigma^3},
\quad
E[x\delta'(b+\sigma x)]
={H_2(\alpha)\phi(\alpha)\over\sigma^2}.
\]

Thus the first-order Wiener correction `deltaZ=deltaT/6` lies entirely in the
collective/Jordan span generated by `e^3`, `Sym(e^2 tensor Me)`, and `J_e(M)`.
If D11/H31 already owns that span, the genuinely novel identity-specific
boundary source begins at order `M^2`, namely the higher Krylov/resolvent
terms.  This gives a symbolic explanation for a dead first-order sibling patch
without implying that the full boundary residual is zero.

For the ownership diff, the exact first-order probe scalar is

\[
\langle T^{(1)},c^{\otimes3}\rangle
=\phi(\alpha)\left[
mH_4(\alpha)x^3+6H_2(\alpha)x^2y+6x a_c\right],
\]

where `x=c^Te`, `y=c^TMe`, and `a_c=c^TMc`.  Thus any old-H31 mismatch at
order one can be graded with the same matrix--vector primitives and should be
tested before the smaller order-two source.

The complete second-order term is still a small deterministic
Krylov/Jordan basis.  Let `P=I-ee^T`, `mu=Me`, `w=Pmu`,
`C=MPM`, `B=PMP`, write `H_j=H_j(alpha)`, and let `Phi=phi(alpha)`.  Direct
second-order expansion of the same delta/delta-prime identity gives

\[
\begin{aligned}
T^{(2)}={}&-{12\alpha\Phi\over\sigma}
\{(\alpha^2-2)J_e(\mu\mu^\top)+J_e(C)\}\\
&-{6\Phi\over\sigma}\operatorname{Sym}\left(
e^{\otimes2}\otimes
[m(H_5+2H_3)\mu+2H_3Mw]\right)\\
&-{\Phi\over2\sigma}\left[
m^2(H_7+4H_5+2H_3)
+4\|w\|^2(H_5+H_3)+2\operatorname{tr}(B^2)H_3
\right]e^{\otimes3}\\
&-{12\alpha\Phi\over\sigma}J_\mu(M)
-{6mH_3\Phi\over\sigma}J_e(M).
\end{aligned}
\]

The scalar distribution identities used here are all instances of

\[
E[h(x)\delta^{(j)}(b+\sigma x)]
={(-1)^j\over\sigma^{j+1}}
\left.{d^j\over dx^j}\{h(x)\phi(x)\}\right|_{x=-\alpha}.
\]

Thus the first post-sibling boundary correction can be carried as a handful of
tagged matrix/vector births rather than as a full generic cubic tensor (5,984
coordinates at `r=32`, or 29,260 at `r=55`), provided
the deployed D11 ownership audit confirms that the order-zero and order-one
terms are already present.  As with every nonlinear birth, these tagged terms
must be contracted reset-locally or transported as their own streams.

Every scalar coefficient in `T^(2)` is odd in the effective threshold
`alpha=b/sigma` (it contains `alpha` or an odd Hermite
`H_3,H_5,H_7`).  Hence

\[
T^{(2)}=0\quad\text{at }\alpha=0,
\]

and the branch is only `O(|alpha|)` near a centered zero-threshold reset.  A
zero/small effective-bias receipt is therefore an analytic kill before any
rank or billing work; in that regime the first post-owned term moves to order
`M^3`.

The identity was independently audited against both the scalar `r=1` reduction
and the purely transverse `Me=0` reduction.  It is exact with `T^(2)` denoting
the coefficient of `epsilon^2` in `T(epsilon M)`, not one half of a second
derivative.

For a bounded-rank cubic adjoint
`Gamma_i=sum_q gamma_iq c_iq^tensor3`, the term can be consumed without forming
`M^2`.  For one probe `c`, put
`x=c^Te`, `y=c^Tmu`, `a_c=c^TMc`,
`d_c=||P Mc||^2`, and `h_c=(Mc)^Tw`, and define

\[
K=m^2(H_7+4H_5+2H_3)+4\|w\|^2(H_5+H_3)
+2\operatorname{tr}(B^2)H_3.
\]

Then

\[
\begin{aligned}
\langle T^{(2)},c^{\otimes3}\rangle={}&
-{12\alpha\Phi\over\sigma}[(\alpha^2-2)xy^2+xd_c]\\
&-{6\Phi\over\sigma}x^2[m(H_5+2H_3)y+2H_3h_c]
-{\Phi\over2\sigma}Kx^3\\
&-{12\alpha\Phi\over\sigma}ya_c
-{6mH_3\Phi\over\sigma}xa_c.
\end{aligned}
\]

Also
`tr(B^2)=||M||_F^2-m^2-2||w||^2`, so every probe needs only `Mc` plus
matrix--vector/scalar work.  If the old H31 adjoint atoms are already exposed,
the incremental local cost is approximately `(q+1)r^2/n^2` per site.  In the
current `r=55,L=10` Base48 regime, one total GEMM of slack requires `q<=33`
before overhead.  A generic unexposed adjoint/source costs at least one full
GEMM per site and is additive-NO.

Novelty is defined, not presumed, by

\[
R_2=(I-\Pi_{old\ H31})T^{(2)}/6,
\qquad \chi_2=\mathcal A_{H31}[R_2].
\]

The exact higher-order reset tail is

\[
R_{\ge3}=T_{Fourier}-T_0-T^{(1)}-T^{(2)},
\]

with endpoint bound

\[
|E_{tail}|\le{1\over6}\sum_l\|\Gamma_l\|_F\|R_{\ge3,l}\|_F.
\]

This bound, or an exact higher-order consumer, must fit the unexplained-value
budget.  The M2 theorem is therefore an exact identity and a conditional
bounded-rank source path, not a claim of full reset closure.

Under the later portfolio gate `q_3>0.9736`, the stronger terminal condition is
that the endpoint-whitened squared norm of the higher-order tail plus any old
H31 ownership mismatch is below `0.05210` of the true signed-H3 defect energy
after optimal scaling.  Merely fitting the earlier absolute residual budget is
not sufficient.

Nor does a short list of nonlinear per-unit tags imply a low-dimensional
forward carrier.  The family `J_e(S)` already spans all of `Sym^3`: each basis
tensor `Sym(e_i tensor e_j tensor e_k)` is
`J_ei(Sym(e_j tensor e_k))`.  Even PSD families such as `M^2` and `MPM` span
the full symmetric-matrix space by linear combinations of rank-one matrices.
With unit-varying eigenframes, a dense mix `sum_i W_ai T_i^(2)` is therefore
generically full rank.  Exact cheap deployment requires reset-local
contraction with an already-pulled-back costate, a proved shared invariant
eigen/Krylov dictionary, or an explicit low-rank tail receipt; the tag formulas
do not commute with dense mixing.

In the current one-slack regime, the bounded-rank loophole is itself sharply
restricted.  A generic full-rank Jordan tensor `J_v(M)` has mode-1 flattening
rank `r`, hence symmetric CP/Waring rank at least `r`.  At `r=55` this lower
bound is 55, already above the local-cost gate `q<=33`.  Since the exact H31
costate contains such Jordan terms, newly constructing a generic costate is an
additive cost-NO.  The M2 route survives only if that H31 action is literally
already exposed for reuse, the relevant matrices have a proved rank/shared
basis below the gate, or a replacement removes enough old billing.

Numerically, a full-rank `q=55` local probe evaluation over ten sites costs
about

\[
10(56){55^2\over1024^2}=1.6155
\]

full-GEMM equivalents, so with one unit of slack it needs at least `0.6155` of
old billing removed if the H31 costate is already exposed.  If two width-55
costate pullbacks must also be formed, add `10(2*55/1024)=1.0742`; total cost
is `2.6897` and required freed billing is at least `1.6897`, before overhead.

The resolvent structure also states the closure risk sharply.  The vector
`m_t=(I-2itM)^-1a` lies in the unit-specific Krylov space
`span{a,Ma,M^2a,...}`, while `C_t` is a rational function of `M` in that unit's
eigenbasis.  This creates the desired identity-specific, noncollective
directions, but the reset map `T(a,M)` is nonlinear.  After a dense unit mix,
`sum_i W_ai T(a_i,M_i)` cannot in general be reconstructed from only
`sum_i W_ai a_i` and `sum_i W_ai M_i`.  Exact provenance still requires the
cubic state or proved projections of it.  Varying unit eigenbases may therefore
make the shared-CP projection tail large, consistent with the earlier carrier
compression no-gos.

## 16. Oblivious value sketch (r=32 theorem; current r=55 correction below)

The carrier need not reconstruct `Z`.  Fix, before fitting, a CountSketch or
TensorSketch

\[
S:\operatorname{Sym}^3(\mathbb R^{32})\longrightarrow\mathbb R^k
\]

and carry only `SZ_i`.  Because `S` is linear,

\[
S\left(\sum_iW_{ia}Z_i\right)=\sum_iW_{ia}SZ_i,
\qquad S(Y)=Q\,S(Z).
\]

At consumption, form the transient cubic adjoints `G_Z,G_Y` above and use
`<SZ,SG>` instead of reconstructing a tensor.  This targets the actual
source--adjoint inner product and is oblivious to unit-specific eigenbases, so
the adaptive low-rank reconstruction no-go does not automatically apply.

The sketch must use orthonormal symmetric coordinates: unique entries with
index multiplicities `3`, `21`, and `111` are weighted by
`sqrt(1),sqrt(3),sqrt(6)` respectively, so Euclidean dot product equals the
Frobenius/Wiener-kernel inner product.  With 4-wise-independent signs and
2-wise-independent buckets,

\[
E\langle Sx,Sy\rangle=\langle x,y\rangle,
\qquad
\operatorname{Var}(\langle Sx,Sy\rangle)
\le {2\over k}\|x\|^2\|y\|^2.
\]

For a corpus, stack endpoint tensors by rows.  The same fixed sketch obeys

\[
E\|\widehat f-f\|^2
\le {2\over k}\sum_e\|Z_e\|_F^2\|G_e\|_F^2.
\]

Let

\[
\mathcal C={\sum_e\|Z_e\|_F^2\|G_e\|_F^2\over\|f\|^2}.
\]

Requiring sketch noise/signal at most
`1/0.8620-1=0.1601` gives the sufficient joint-correlation gate

\[
k\ge {2\mathcal C\over0.1601}\simeq12.49\mathcal C.
\]

After the AQ state, an in-place sketch stores `nk` values and therefore permits
`k<=floor(475136/1024)=464` under the literal `n^2` ceiling.  This passes the
bound when `mathcal C<=37.1`, roughly an effective squared source--adjoint
cosine of at least `0.027`.  Its unit-transport and endpoint action cost only
`2k/n=0.90625` full-GEMM equivalents per layer at `k=464`; forming and hashing
the transient `p=5984`-coordinate adjoints is `O(np)`, far below `n^3`.

This endpoint census does not erase a generic source-evaluation kernel.  If the
cubic source is exposed only through a dense cross-unit contraction of the
form `diag(C M C^T)`, charge an additional one full GEMM per layer (or its
proved bounded-rank fraction).  Then the `k=464` pre-weight sketch path costs
approximately

\[
1.15625+0.90625+1+0.006\simeq3.0685
\]

per layer, about `42.96` over 14, still below 49.  The CP path at `k=449`
would rise above 49; with a generic one-GEMM source its pre-weight cap becomes
`k<=344`.  The Fourier reset projection bypasses this dense source cost only
when each `Z_i=T(a_i,M_i)/6` is generated and projected locally before dense
unit mixing.  It does not bypass a genuinely cross-unit cubic source merely
because the endpoint consumer is tensor-free.

This is a mathematical conditional GO, not a certified campaign win.  The
sketch seed must be preregistered, the reset projection/quadrature error and
finite-`Z` remainder must be charged, and actual promotion still uses the
D11-residualized Schur gain rather than the unbiasedness statement alone.

Because the old D11/Q24 total is already decomposed as an existing
`CP4+3 alpha vC+H31` action, the post-AQ cubic lane should be owned as a source
replacement inside `H31`, not as an appended absolute consumer.  The novel
quantity is

\[
\chi_\Delta=\mathcal A_{H31}[Z_{boundary}-Z_{old}].
\]

If the H31 reverse query factors are already exposed, their construction can
be reused; only the replacement-source projection/transport and any generic
source kernel are incremental.  Promotion must grade this difference against
the existing span and replacement tolerance.

The sketch should therefore be applied directly to the source residual:
`S deltaZ=S(Z_boundary-Z_old)`.  Its collision variance is bounded by

\[
{2\over k}\|\delta Z\|_F^2\|G_{H31}\|_F^2,
\]

which can be much smaller than sketching two absolute actions.  At `k=464`,
the corresponding corpus ratio
`C_delta=sum||deltaZ||^2||G||^2/||A_H31[deltaZ]||^2` must be at most about
`37.1` for the sufficient joint-correlation certificate.

For a deterministic all-query guarantee after freezing the seed, let `U` be
an orthonormal basis for the old-span-residualized source/query subspace.  The
restricted singular-value condition is

\[
\|U^\top(S^\top S-I)U\|_{op}\le\epsilon,
\]

equivalently every singular value of `SU` lies between
`sqrt(1-epsilon)` and `sqrt(1+epsilon)`.  Necessarily `rank(U)<=k`; if the
relevant endpoint subspace has rank above 464, no 464-coordinate linear sketch
can provide a uniform direction-wise certificate, even though the weaker
pair-specific variance gate may still hold.

For separate source and query spaces with orthonormal bases `U_X,U_Y`, the
sharp uniform bilinear condition is the weaker cross-embedding test

\[
\|U_X^\top(S^\top S-I)U_Y\|_{op}\le\epsilon.
\]

Embedding their union is sufficient but not necessary.  Since the sketched
cross Gram has rank at most `k`, its uniform error is necessarily at least the
`(k+1)`-st singular value of `U_X^TU_Y`.  Matched endpoint pairs can use the
weaker variance certificate, but a replace-action tolerance `tau` requires the
weighted cross error times the relevant source/query norms to be at most
`tau`.

The exact incremental cost over a card that already exposes post-weight AQ and
`H/Q` is, for `L_s` active source sites,

\[
G_{sketch}=L_s\left({2k\over n}+{p\over n^2}+g_{src}\right),
\]

where `p=5984` and `g_src` is zero for a genuinely local Fourier/atom birth and
one for a generic dense `diag(CMC^T)` source.  At `L_s=14,k=464` this is
`12.768` or `26.768`, respectively.  Recomputing pre-weight AQ state adds
`14(1.15625)`, giving `28.955` or `42.955`.  Thus an additive card total is
`G_card+G_sketch`, while replacing D10 is
`G_card-G_D10+G_sketch`; the literal `k<=464` was a memory gate, not an
assertion that either package has that much billing slack.

For an owner-correct replacement, if `S_rest` is the card span with D10
removed, the net raw change is

\[
d^\top\{P_{[S_{rest},F_{new}]}-P_{[S_{rest},F_{D10}]}\}d.
\]

Thus the new conditional marginal must exceed the lost D10 conditional
marginal by at least the required `2.86e-10`; freed billing cannot be credited
without debiting the removed raw value.

Here `g_src=0` means only that the dense source GEMM is absent.  A finite
`J`-node direct Fourier projection still costs approximately
`g_F=Jp/n^2=0.005707J` equivalents per layer (or about `Jkr/n^2` for direct
structured-atom projections), plus local resolvent work.  The actual census
must substitute this cost and include a certified quadrature-tail error;
"local" is not synonymous with free.

Finally, the sketch is exact through affine mixing but not through a generic
subsequent ReLU reset.  Linearizing that reset at `g_0` defines

\[
6\langle\mathcal R_{g_0}Z,G\rangle
=E[1_{\{g_0>0\}}I_3(Z)I_3(G)].
\]

An exact update from `SZ` exists only if some `A_g` satisfies
`S R_g=A_g S`, equivalently `ker S` is invariant under every unit-specific
reset operator.  This fails generically for `k<p`.  The valid deployment frame
is therefore a reset-local source contracted with an already-backpropagated
complete `Q/B` adjoint at that site; a forward multi-reset H3 carrier needs an
additional closure proof or leakage charge.

### Current-package regime correction

All `k=449/464`, `p=5984`, and 14-layer figures above are explicitly for the
checkpoint architecture `r=32,n=1024`.  A later cross-lane audit reports that
the current candidate package may instead use `r=55`, so

\[
p=\binom{57}{3}=29,260,
\]

with ten active sites and only one full-GEMM equivalent of slack above a
Base48 card.  No current receipt proves that `SG_Z`, `SG_Y`, or
`Gamma=W(G_Z+Q^TG_Y)` is already materialized.  Counting both rectangular
sketch actions gives `2k/n` per site and hence the transport-only bound

\[
k\le\left\lfloor{1024\over2\cdot10}\right\rfloor=51.
\]

One transient full-coordinate query/hash costs
`p/n^2=0.02790` per site and reduces the bound to about `k<=36`; a comparable
source-coordinate pass reduces it to about `k<=22`, before quadrature or
resolvent work.  A generic one-full-GEMM source kernel per site cannot fit the
one-GEMM total slack at all.  Therefore CountSketch is an additive
current-package NO-GO unless a reuse receipt exists.  It remains conditional
as a D10 replacement that frees enough billing, or with a bounded structured
source/query formula that avoids both full-coordinate passes.

The sufficient correlation certificate also tightens sharply: the bound
`C_delta<=k/12.49` becomes `4.08`, `2.88`, and `1.76` at `k=51,36,22`
respectively, corresponding roughly to effective squared source--query cosine
at least `0.245`, `0.347`, and `0.568`.  Under one-GEMM slack this is not a
generic broad-spectrum carrier; it needs unusually strong endpoint alignment
or replacement billing.

A later full-portfolio capacity decomposition tightens the earlier coarse
`0.92845` joint-correlation heuristic.  Its exact plane is

\[
14.7531q_v+4.370q_3+1.0313q_4>19.4091
\quad\text{(all coefficients in }10^{-9}\text{)},
\]

so even perfect H3/H4 need `q_v>0.9744`, and at the quoted
`q_{v+q4}=0.98346` the H3 lane needs `q_3>0.9736`.  For an endpoint-whitened
rank-`k` H3 carrier this requires

\[
{\sum_{j>k}\sigma_j^2\over\sum_j\sigma_j^2}
<1-0.9736^2=0.05210.
\]

Thus a `k=36` or `k=22` current-package carrier is capacity-NO whenever the
tail beyond mode 36 or 22 exceeds 5.21%.  A target correlation of `0.983`
would require tail below 3.371%.  These portfolio gates supersede the weaker
coarse correlation criterion wherever their assumptions apply.

For CountSketch, `q_3>0.9736` requires noise/signal below
`1/0.9736^2-1=0.05497`, so the corresponding sufficient variance condition is

\[
k\ge36.39\,\mathcal C_\Delta.
\]

Since `C_delta>=1` by Cauchy--Schwarz, `k=36` does not meet this conservative
certificate even at ideal aggregate alignment; `k=51` requires
`C_delta<=1.401`, and `k=22` cannot.  This is a sufficient stochastic bound,
not a deterministic impossibility for a particular frozen collision pattern;
the cross-singular and target-tail tests remain the sharp deterministic kills.

## Revised current next question

The fail-fast order is now fixed.  First, algebraically diff the deployed old
H31 reset source against `T0+T1`; any mismatch is cheaper than the M2 route.
Second, require a literal receipt that the pulled-back H31 costate/action is
already exposed; absent that receipt, generic Jordan rank 55 exceeds the
one-slack rank/cost gate 33 and kills additive M2.  Third, if reuse exists,
contract the exact M2 probe formula reset-locally and require the
old-residualized higher-order H3 tail below 5.21%.  Only after those gates may a
D10 replacement be considered, with both its removed billing and its lost
conditional raw value explicitly debited.  The generic CountSketch branch is
already additive cost/correlation NO in the current r55/Base48 regime unless
replacement or a new reuse receipt changes its rank ceiling.

## 18. Checkpoint 12: exact overlap with the latent Hermite-H4 score/JVP

The direct AQ Hermite-H4 oracle and the signed-H3 reset source are linked by
Wiener raising/lowering identities, but they are not the same provider.  This
section separates four claims which must not be conflated:

1. an H4 scalar can be a translation derivative of an H3 scalar;
2. an AQ-parameter JVP of that H4 scalar contains a boundary-H3 **ladder
   term**;
3. the old D11 interface can own either scalar at a synchronized seam; and
4. the two scalars can sometimes reuse one future costate.

Only the fourth statement is a deployment-cost statement, and none of the
four implies that the absolute post-AQ H3 defect is zero.

### 18.1 Exact Wiener product decomposition of the AQ score JVP

Use the half-quadratic convention of the H4 lane,

\[
g(z)=c+b^\top z+\frac12(z^\top Mz-\operatorname{tr}M),
\qquad F=g_+,
\qquad J=1_{\{g>0\}}.
\]

Let `K_m(X)=E[X He_m(Z)]` be the order-`m` Hermite tensor and let
`I_m(A)` denote the corresponding multiple Wiener polynomial.  For
`H in Sym^4`, define

\[
\mathcal A_H(F)=\langle K_4(F),H\rangle
=E[F I_4(H)].
\tag{18.1}
\]

Write an AQ tangent as

\[
\dot g=\dot c+I_1(h)+I_2(A),
\qquad A=\dot M/2.
\]

The exact JVP is `E[J dot g I4(H)]`.  Applying

\[
I_p(f)I_q(k)=\sum_{s=0}^{\min(p,q)}
s!{p\choose s}{q\choose s}
I_{p+q-2s}(f\widetilde\otimes_s k)
\]

gives the degree-resolved identity

\[
\boxed{
\begin{aligned}
D\mathcal A_H[\dot g]={}&
\dot c\,\langle K_4(J),H\rangle\\
&+\langle K_5(J),h\widetilde\otimes H\rangle
+4\langle K_3(J),H\otimes_1h\rangle\\
&+\langle K_6(J),A\widetilde\otimes H\rangle
+8\langle K_4(J),H\otimes_1A\rangle
+12\langle K_2(J),H\otimes_2A\rangle .
\end{aligned}}
\tag{18.2}
\]

Here every contraction is symmetrized in its free indices.  Because

\[
K_m(J)=\partial_c K_m(F),
\tag{18.3}
\]

the **only order-three ladder term** in an AQ score JVP is

\[
\boxed{
4\langle\partial_cK_3(F),H\otimes_1h\rangle .}
\tag{18.4}
\]

A constant tangent has no H3 term.  A quadratic tangent has only the even
`6/4/2` sectors.  Most importantly, (18.4) contains the bias derivative of
the signed H3 tensor, not the absolute tensor `K3(F)`.  The same linear JVP
also contains the inseparable H5 term in (18.2).  Therefore one fixed H4
score/JVP observation is not an H3 source oracle.

For the mixed rank-one score

\[
H_{31}=\operatorname{Sym}(U^{\otimes3}\otimes V),
\]

its entire H3 ladder range for a linear AQ tangent is

\[
\boxed{
4(H_{31}\otimes_1h)
=(h^\top V)U^{\otimes3}
+3(h^\top U)\operatorname{Sym}(U^{\otimes2}\otimes V).}
\tag{18.5}
\]

Thus one fixed `(U,V)` action sees at most the two-dimensional cubic star
span in (18.5).  It does not generically contain the matrix-Jordan terms
`J_{Ru}(N)`, `J_{Nu}(R)`, `J_u(N\circ R)`, `J_{Nr}(N)`, and `J_r(N^2)` in
the exact `G_Z/G_Y` action of Section 13.

For a more general square/Jordan H4 costate

\[
H_{A,B}=\operatorname{Sym}(A\otimes B),
\qquad H_{A,B}[x^4]=(x^\top Ax)(x^\top Bx),
\]

the exact lowering formula is

\[
\boxed{
4(H_{A,B}\otimes_1h)
=2\{J_{Ah}(B)+J_{Bh}(A)\}.}
\tag{18.6}
\]

Hence a resident lifted matrix-word module can overlap some Jordan pieces of
`G_Z/G_Y`, but only when the literal matrices and vector images in (18.6)
match.  Sharing rank, weights, or a latent basis is not that equality.

### 18.2 Raising identity and the missing integration constant

Let `tau_h F(z)=F(z+epsilon h)`.  Gaussian integration by parts cancels the
lower-order product term and yields

\[
\boxed{
{d\over d\epsilon}\bigg|_0
\langle K_3(\tau_hF),G\rangle
=\langle K_4(F),\operatorname{Sym}(h\otimes G)\rangle .}
\tag{18.7}
\]

So an H4 score with a factored test tensor is exactly a **translation
derivative** of an H3 action.  Conversely, differentiating the H4 score once
more under the same translation raises rather than lowers the chaos degree:

\[
{d\over d\epsilon}\bigg|_0
\mathcal A_H(\tau_hF)
=\langle K_5(F),\operatorname{Sym}(h\otimes H)\rangle .
\tag{18.8}
\]

Equations (18.4), (18.7), and (18.8) resolve the apparent paradox.  The H3
term visible inside the **parameter** JVP is a gate-boundary derivative; the
H4 score itself is a translation derivative of the H3 coefficient; and the
translation JVP of H4 is H5.  None supplies the absolute H3 integration
constant at the frozen reset.

Bias integration can recover it in principle:

\[
K_3(F(c))=\int_{-\infty}^{c}K_3(1_{\{g_s>0\}})\,ds,
\tag{18.9}
\]

provided the source vanishes at the lower endpoint.  But (18.9) requires a
continuum, or a certified quadrature and the separation of the H5 term in
(18.2).  A single frozen score/JVP does not implement (18.9).  For the
post-owned perturbative source of Section 15, the score JVP can therefore
overlap `partial_c T^(2)`; it does not by itself own `T^(2)`.

### 18.3 Exact D11 ownership split

Let `T_3=K_3(F)` and let `T_3^old` be the synchronized old H31 reset source,
including its convention and site.  Define

\[
\Delta T_3=T_3-T_3^{old}.
\tag{18.10}
\]

The H3 ladder term in (18.2) splits exactly as

\[
4\langle\partial_cT_3,H\otimes_1h\rangle
=4\langle\partial_cT_3^{old},H\otimes_1h\rangle
+4\langle\partial_c\Delta T_3,H\otimes_1h\rangle .
\tag{18.11}
\]

The first term is D11-owned only if the deployed H31 provider is
differentiated in the same frame and parameterization.  The second is an
unowned **derivative** of the source residual.  The absolute novel H3 action
remains

\[
\boxed{
\chi_{3,\Delta}={1\over3!}\langle\Delta T_3,G_3\rangle ,}
\tag{18.12}
\]

where `G3` is the complete future H3 query.  There is no algebraic identity
between (18.12) and the second term of (18.11) at one bias.

Applied to the boundary expansion of Section 15, the classification is
literal: `T0+T^(1)` and `partial_c(T0+T^(1))` are old only if the pending
deployed-H31 equality check passes; `T^(2)` is the first candidate absolute
post-owned birth under that premise; and the H4-score JVP overlaps only
`partial_c T^(2)`.  The exact centered/transverse fidelity theorem from the
direct-adjoint lane is consistent with this split--noncollective local
fidelity has zero first variation--but it does not replace the old-source
equality check or identify the M2 integration constant.

Likewise, decompose the H4 costate/test into the complete old D11 interface
dual and its complement.  A synchronized `A_H` with `H` in the old
CP4/star/pair/K31 range is the same-site scalar already audited in Sections
10 and 15 and receives zero additive credit.  Its AQ JVP is the tangent of
that old scalar under a joint refit; it is not a second feature.  Only

\[
\chi_{4,\perp}
=\langle(I-\Pi_{O_4})K_4(F),H\rangle
\tag{18.13}
\]

and its response-space residual can count as a new H4 direction.  The H5/H6
terms in (18.2) are required pieces of the exact JVP, but they must not be
relabelled as signed-H3 gain.  They are either internal traced terms of the
same old tangent or separately typed higher-chaos/model residuals.

The binding no-double-count rule is response-space Schur projection.  Let
`S_old` be the complete banked D11/F2J feature space, let `J4` be the columns
of any proposed residual H4-score/JVP family, and let `F3` be the columns of
the absolute H3 residual (18.12).  The only lawful joint new space is

\[
\boxed{
\mathcal N
=(I-P_{S_{old}})\operatorname{span}[J_4,F_3].}
\tag{18.14}
\]

Equivalently one may book

\[
F_{3\mid4}=(I-P_{[S_{old},J_4]})F_3,
\tag{18.15}
\]

or reverse the order; the projector onto the combined span is invariant.
Diagram labels such as “H3 ladder” and “H4 score” do not imply statistical
orthogonality.  Promotion still requires at least `2.86e-10` joint Schur raw
gain, and the H3 carrier still has to meet `q3>0.9736`, equivalently less than
`5.210%` endpoint-whitened squared tail under the current portfolio premise.

### 18.4 Costate reuse theorem

There are two different reuse questions.

First, compare the **parameter JVPs** of the unnormalized H3 action
`tilde chi3=<K3(F),G>` and an H4 score at fixed queries `G in Sym^3` and
`H in Sym^4`:

\[
D\widetilde\chi_3[\dot g]=E[J\dot g I_3(G)],
\qquad
D\mathcal A_H[\dot g]=E[J\dot g I_4(H)].
\]

Let `P_AQ` be orthogonal projection onto the local AQ tangent space
`H0 direct-sum H1 direct-sum H2`.  These two JVP costates are interchangeable
for **every** AQ tangent if and only if

\[
\boxed{
P_{AQ}\{J[I_3(G)-I_4(H)]\}=0.}
\tag{18.16}
\]

This is the sharp local score-costate reuse test.  The lowering identities
(18.5)--(18.6) provide useful necessary structure, but do not replace (18.16)
because multiplication by the nonpolynomial gate mixes all Hermite degrees.
Even when (18.16) holds, it identifies tangent functionals only; it does not
make the absolute values (18.12) and (18.13) equal.

Second, compare the **future Bellman rows**.  Let a local H3 birth `z_l` emit
into the already carried next-cut product state and a hidden persistent H3
state as

\[
z_l\longmapsto(L_lz_l,P_lz_l),
\qquad
L_lz_l=(L_{4,l}z_l,L_{31,l}z_l,L_{22,l}z_l).
\]

With resident product-state costate
`b_{l+1}=(Q4,Q31,Q22)` and hidden H3 costate `q3`, the exact query is

\[
\boxed{
G_{3,l}=L_l^*b_{l+1}+P_l^*q_{3,l+1}.}
\tag{18.17}
\]

The `G_Z/G_Y` formulas of Section 13 are exactly the `L31^*Q31` component of
(18.17), in the synchronized AQ convention.  They are not literally the
resident matrix word `b_l`, and they omit

\[
L_{4,l}^*Q4+L_{22,l}^*Q22.
\tag{18.18}
\]

Therefore the paid D10 row consumes the complete local signed-H3 birth with
no new reverse stream if and only if all of the following hold:

1. `Q4,Q31,Q22` are literal coordinates/views of the already billed D10
   reverse tape at the identical seam, frame, collision convention, and time;
2. the local adjoint application `L_l^*b_{l+1}` is fused/native or fits the
   remaining local arithmetic allowance;
3. `P_l^*q3=0`, or the persistent H3 coordinate is itself already resident;
4. the K4/K22 term (18.18) is included or certified Bellman-dark.

Under these conditions, reuse is an exact theorem: the H4 score, its lawful
JVP, and the immediate H3 emission can share the same resident future row,
while their source acquisitions remain separately charged.  If any condition
fails, shared `W/Q`, rank, or latent factors do not save the cost: the missing
future component is a new cyclic row.  A new independent D10 matrix bundle
costs about `20` GEMM-equivalents, so it is closed against the current
Base48/49 ledger.

### 18.5 Sharp disposition

- **Old H4 score/JVP:** duplicate or old-model tangent when its test costate is
  in the synchronized complete D11 interface.
- **H3 ladder overlap:** exactly (18.4); it is `partial_c K3(F)` and for one
  H31 score is restricted to (18.5).
- **Absolute signed-H3 residual:** (18.12), not supplied by one score/JVP.
- **Immediate H3-to-product emission:** free in reverse only under all four
  conditions following (18.18).
- **Persistent signed-H3 future value:** independent and cost-closed unless
  its row is zero, already resident, or a measured low cyclic-rank quotient
  fits the cap.

Graph-ready nodes:

- `P2R0G19B_MATH_H3_H4_JVP_LADDER` — **solved identity**: (18.2)--(18.6).
- `P2R0G19B_MATH_H3_H4_ABSOLUTE_OBSTRUCTION` — **solved no-go**: one frozen
  H4 score/JVP does not determine `K3(F)`, (18.7)--(18.12).
- `P2R0G19B_MATH_H3_H4_D11_OWNERSHIP` — **solved decomposition / premise-open
  synchronization**: (18.10)--(18.15).
- `P2R0G19B_MATH_H3_H4_LOCAL_COSTATE_REUSE` — **solved iff theorem**:
  (18.16).
- `P2R0G19B_MATH_H3_H4_FUTURE_ROW_REUSE` — **solved iff theorem / open
  resident receipt**: (18.17)--(18.18) and the four conditions.
- `P2R0G19B_MATH_H3_H4_JOINT_RAW_GATE` — **open value receipt**: combined
  response-space projection (18.14), `>=2.86e-10`, with H3 tail `<5.210%`.

Edges: `AQ_H4_SCORE_IDENTITY -> H3_H4_JVP_LADDER`;
`H3_H4_JVP_LADDER -> H3_H4_ABSOLUTE_OBSTRUCTION`;
`D11_K31_OWNERSHIP -> H3_H4_D11_OWNERSHIP`;
`H3_H4_LOCAL_COSTATE_REUSE -> H3_H4_FUTURE_ROW_REUSE`;
`H3_H4_D11_OWNERSHIP`, `H3_H4_ABSOLUTE_OBSTRUCTION`, and
`H3_H4_FUTURE_ROW_REUSE -> H3_H4_JOINT_RAW_GATE`.

## 19. Checkpoint 13: full-seam H3 adjoint under q211 Jordan/skew fusion

The q211 antisymmetric loophole retains a full matrix product before its
symmetric and skew reductions.  That product can also serve the immediate H3
source adjoint, but only for the matrix-pair words it literally contains.
The exact statement is most transparent through one universal symmetric
trilinear query.

### 19.1 Universal one-H3/three-AQ contraction

For an affine--quadratic chaos coordinate write

\[
\theta=(x,M),\qquad X_\theta=I_1(x)+I_2(M),
\]

where `M` is symmetric and uses the Wiener-kernel convention of Section 13.
Define

\[
\boxed{
\mathcal C_x(M,N)
=J_{Nx}(M)+J_{Mx}(N)+2J_x(M\circ N),}
\tag{19.1}
\]

and, for `theta_a=(x_a,M_a)`,

\[
\boxed{
\begin{aligned}
\mathcal B(\theta_1,\theta_2,\theta_3)
={}&6\operatorname{Sym}(x_1\otimes x_2\otimes x_3)\\
&+12\{\mathcal C_{x_1}(M_2,M_3)
+\mathcal C_{x_2}(M_1,M_3)
+\mathcal C_{x_3}(M_1,M_2)\}.
\end{aligned}}
\tag{19.2}
\]

The Wiener product formula gives the exact identity

\[
\boxed{
E[I_3(Z)X_{\theta_1}X_{\theta_2}X_{\theta_3}]
=\langle Z,\mathcal B(\theta_1,\theta_2,\theta_3)\rangle.}
\tag{19.3}
\]

There is no first-order covariance-counterterm contribution because `I3(Z)`
is orthogonal to every AQ `H1/H2` coordinate.  Thus (19.3) is also the exact
first variation of the connected fourth cumulant when one slot acquires an
H3 birth.

Useful repeated-slot reductions are

\[
\boxed{
\mathcal B(\theta,\theta,\theta)
=6x^{\otimes3}+72\{J_{Mx}(M)+J_x(M^2)\},}
\tag{19.4}
\]

\[
\boxed{
\begin{aligned}
\mathcal B(\theta,\theta,\eta)
={}&6\operatorname{Sym}(x^{\otimes2}\otimes y)\\
&+24\{\mathcal C_x(M,N)+J_{My}(M)+J_y(M^2)\},
\end{aligned}}
\tag{19.5}
\]

for `eta=(y,N)`.  Equation (19.5), multiplied by three, is exactly the
`G_Z` tensor of Section 13; (19.4) is exactly `G_Y`.  This independently fixes
all slot multiplicities.

Normalization is binding.  Equations (19.1)--(19.5) use the Wiener kernel
`I2(M)=z^T M z-tr M`.  In the half-quadratic convention

\[
Q_A={z^\top A z-\operatorname{tr}A\over2}=I_2(A/2),
\]

substitute `M=A/2,N=B/2`.  Every matrix-Jordan term in (19.5) is bilinear in
the two quadratic kernels and is therefore divided by four, while the pure
H1 star is unchanged.  In particular, for a structured source functional
`S(x,M)=<S,J_x(M)>`,

\[
\boxed{
\begin{aligned}
\langle S,\mathcal B((u,A/2),(u,A/2),(v,B/2))\rangle
={}&6S(v,uu^\top)+6S(Bu,A)+6S(Au,B)\\
&+12S(u,A\circ B)+6S(Av,A)+6S(v,A^2).
\end{aligned}}
\tag{19.5a}
\]

Using the unscaled `24,24,48,24,24` coefficients with half-quadratic matrices
would overcount the H3 surface action by four.

### 19.2 Complete product-seam pullback

Let the complete fourth-order seam use the diagonal K4, repeated/singleton
K31, and paired K22 coordinates.  For one action written once in its displayed
slot convention, the cubic pullbacks are

\[
\boxed{
\begin{array}{c|c}
\text{seam action}&\text{H3 source adjoint}\\
\hline
q_4(\theta,\theta,\theta,\theta)&4\mathcal B(\theta,\theta,\theta)\\
q_{31}(\theta,\theta,\theta,\eta)&
3\mathcal B(\theta,\theta,\eta)\ \text{on }Z_\theta,
\quad \mathcal B(\theta,\theta,\theta)\ \text{on }Z_\eta\\
q_{22}(\theta,\theta,\eta,\eta)&
2\mathcal B(\theta,\eta,\eta)\ \text{on }Z_\theta,
\quad2\mathcal B(\eta,\theta,\theta)\ \text{on }Z_\eta .
\end{array}}
\tag{19.6}
\]

The actual stored-plane Frobenius convention may include both orientations;
then the corresponding transpose costate and multiplicity must be included
before applying (19.6).  A convention-free version is obtained by assembling
the fully symmetric external costate `Lambda_abcd`:

\[
\boxed{
G^{\rm product}_{3,i}
=4\sum_{j,k,l}\Lambda_{ijkl}
\mathcal B(\theta_j,\theta_k,\theta_l).}
\tag{19.7}
\]

Equations (19.6)--(19.7) are the explicit realization of

\[
L_4^*Q_4+L_{31}^*Q_{31}+L_{22}^*Q_{22}.
\tag{19.8}
\]

If the stored source is the derivative tensor `T3=K3(F)` rather than its
Wiener kernel, the local scalar is
\(\langle T_3,G_3\rangle/3!\), as in Section 18.

### 19.3 The independent off-diagonal 211 coordinate

An external off-diagonal `211` action is not algebraically one of the three
collision-plane reads in (19.8).  Write its costate action as

\[
S_{211}=\sum_{i,j,k}\Lambda_{i;jk}
\kappa_4(X_i,X_i,X_j,X_k),
\qquad \Lambda_{i;jk}=\Lambda_{i;kj},
\tag{19.9}
\]

with collision restrictions and normalization absorbed into `Lambda`.  Its
exact H3 pullback is

\[
\boxed{
\begin{aligned}
G^{211}_{3,p}={}&
2\sum_{j,k}\Lambda_{p;jk}\mathcal B(\theta_p,\theta_j,\theta_k)\\
&+2\sum_{i,k}\Lambda_{i;pk}\mathcal B(\theta_i,\theta_i,\theta_k).
\end{aligned}}
\tag{19.10}
\]

The first line differentiates either repeated-anchor slot and the second
combines the two symmetric singleton slots.

If the q211 costate has the common-factor receipt

\[
\Lambda_i=U\Theta_iU^\top,
\qquad
\theta_a=\sum_jU_{ja}\theta_j=(u_a,N_a),
\tag{19.11}
\]

trilinearity of `B` gives the tensor-free external reduction

\[
\boxed{
\begin{aligned}
G^{211}_{3,p}={}&
2\sum_{a,b}(\Theta_p)_{ab}
\mathcal B(\theta_p,\theta_a,\theta_b)\\
&+2\sum_{i,a,b}U_{pa}(\Theta_i)_{ab}
\mathcal B(\theta_i,\theta_i,\theta_b).
\end{aligned}}
\tag{19.12}
\]

This uses the same common external factors as the commutator-Gram q211
consumer.  It does not form an external `n^3` 211 tensor.

Consequently the literally complete seam query is

\[
\boxed{
G_{3,l}=
L_4^*Q_4+L_{31}^*Q_{31}+L_{22}^*Q_{22}
+L_{211}^*\Lambda_{211}+\mathcal P_l^*q_{3,l+1}.}
\tag{19.13}
\]

The four-term formula without `L211^*Lambda211` is exact only if the
off-diagonal 211 action has been **literally compiled** into the resident local
reduction and (19.10) is included there, or if (19.10) vanishes on the lawful
source manifold.  This is not automatic: the commutator q211 has zero K31
restriction and K22 sees only its diagonal.  Antisymmetric arithmetic fusion
shares products; it does not turn an independent 211 coordinate into a
collision-plane coordinate.

For the q4-perpendicular provider, (19.10)--(19.12) must be passed through the
same path/collision/old-span residual projector that defines q4-perp.  The
baseline AQ identity `q4_perp=-8` times a commutator Gram does not by itself
define or own the beyond-AQ H3 tangent.

More explicitly, in the raw quadratic-core convention the exact same-site AQ
identity is

\[
\boxed{
\kappa_{211}^{AQ}(i;j,k)
=48\operatorname{tr}(M_i^2M_jM_k)
-8\langle[M_i,M_j],[M_i,M_k]\rangle_F.}
\tag{19.13a}
\]

Thus `-8` times the full `Theta_ab` commutator Gram is the complete AQ
**q4-perpendicular** scalar only when the first path/parallel term and all
collision sectors have been removed by the defining residual projector.  A
diagonal reduction `-8 sum lambda_ia ||[M_i,N_a]||^2` is complete only when
`Theta_i` is literally diagonal; otherwise its `a != b` skew cross-Grams are
missing.  For a finite post-AQ cubic perturbation the completeness equation is

\[
\boxed{
S_{211}(X^{AQ}+I_3Z)
=S_{211,\parallel}^{AQ}+S_{211,\perp}^{AQ}
+\sum_p\langle Z_p,G^{211}_{3,p}\rangle+R_{\ge2}(Z).}
\tag{19.13b}
\]

The antisymmetric AQ scalar owns only the second baseline term.  Equation
(19.12) supplies the exact linear H3 term; quadratic and higher cubic-defect
diagrams remain in `R_{>=2}`.

### 19.4 Exact Jordan/skew split of the required products

For every symmetric matrix pair required by (19.1), retain the full ordered
product

\[
\mathsf P_{MN}=MN,
\qquad
\mathsf J_{MN}={\mathsf P_{MN}+\mathsf P_{MN}^\top\over2}=M\circ N,
\qquad
\mathsf A_{MN}={\mathsf P_{MN}-\mathsf P_{MN}^\top\over2}={[M,N]\over2}.
\tag{19.14}
\]

Then

\[
\boxed{
\mathcal C_x(M,N)
=J_{Nx}(M)+J_{Mx}(N)+2J_x(\mathsf J_{MN}),}
\tag{19.15}
\]

while the antisymmetric q211 reduction is

\[
\boxed{
\langle[M,N_a],[M,N_b]\rangle_F
=4\langle\mathsf A_{MN_a},\mathsf A_{MN_b}\rangle_F.}
\tag{19.16}
\]

Thus H3 and q211 use complementary orthogonal halves of the **same full
product**:

- the immediate H3 emission uses only `J_MN` plus the cheap matrix-vector
  images `Mx,Nx`;
- the noncommutative q211 scalar uses only `A_MN`;
- `J` alone cannot recover `A`, and a reduced skew Gram cannot recover `J`;
- retaining the full `P_MN` before symmetrization supplies both without a new
  dense latent product.

The absence of `A_MN` from (19.15) is exact, not an approximation.  A skew
two-tensor has zero quadratic polynomial and its symmetrization inside a
cubic Wiener kernel vanishes.  Therefore the H3 and commutator scalars share
arithmetic but are not duplicate physical features.

### 19.5 Which emissions actually fuse

The product inventory follows directly from (19.2): every unordered pair of
quadratic legs among the three surviving AQ slots requires its full ordered
product.  In particular,

\[
\boxed{
\begin{array}{c|c}
\text{cubic query atom}&\text{required dense latent products}\\
\hline
\mathcal B(\theta,\theta,\theta)&M^2\\
\mathcal B(\theta,\theta,\eta)&M^2,\ MN\\
\mathcal B(\theta,\eta,\eta)&MN,\ N^2\\
\mathcal B(\theta_i,\theta_a,\theta_b)&M_iN_a,\ M_iN_b,\ N_aN_b\\
\mathcal B(\theta_i,\theta_i,\theta_b)&M_i^2,\ M_iN_b.
\end{array}}
\tag{19.17}
\]

The q211 antisymmetric fusion of Section 40 in the H4 lane forms only the
anchor--factor products `M_i N_a`.  Therefore, with no further resident
products:

1. every pure-vector star in (19.2) fuses trivially;
2. every cross Jordan term whose pair is exactly `(M_i,N_a)` uses
   `Sym(P_ia)` and fuses;
3. the commutator q211 action uses `Skew(P_ia)` and fuses;
4. K4 and both K31 slot pullbacks fuse only if the needed anchor square
   `M_i^2` is also resident;
5. the K22 side with two suffix legs additionally needs `N_a^2`;
6. the q211 repeated-anchor H3 term in the first line of (19.12) additionally
   needs every live factor--factor product `N_aN_b`; even diagonal `Theta`
   needs `N_a^2`;
7. the q211 singleton H3 term in the second line of (19.12) needs only
   `M_i^2` and the already formed `M_iN_b`.

This is the sharp correction to a blanket “q211 fusion makes the full H3 seam
free” claim.  In the common rank-one case the missing product inventory is
only `M_i^2` and `N^2`, but those squares still require literal receipts.

If all anchor--factor products `M_iN_a` are already resident, the additional
latent-product census for **explicitly constructing the full cubic query** is:

\[
\boxed{
\begin{array}{c|c|c}
\text{costate support}&\text{additional factor products per layer}
&\text{D-layer width-GEMM equivalent}\\
\hline
\text{full common }k&k(k+1)/2&Dk(k+1)r^3/(2n^3)\\
\text{diagonal common }k&k& Dkr^3/n^3\\
\text{rank one}&1&Dr^3/n^3 .
\end{array}}
\tag{19.18}
\]

In every row, add `n` anchor squares per layer, costing

\[
\boxed{D r^3/n^2,}
\tag{19.19}
\]

unless the exact `M_i^2` arrays are already resident.  Equations
(19.18)--(19.19) count explicit query construction and are also the generic
full-rank-source census.  They are not a lower bound for every factorized
source.  For example, against a CP atom `c^tensor3`,

\[
\boxed{
\langle c^{\otimes3},J_x(M\circ N)\rangle
=(c^\top x)(Mc)^\top(Nc),}
\tag{19.20}
\]

so the missing `MN` is consumed by two matrix-vector products without being
formed.  Source evaluation is therefore a separate exact-factorization
receipt.

Constructing `G3` is also distinct from contracting it with the reset source.
For a CP/rank-one source, every pairing with (19.2) is matrix-vector/scalar
work.  For the exact Fourier AQ-reset source of Section 15, the Jordan pairing
is tensor-free, but its query matrices must be rotated into the reset
eigenbasis or matched to already resident products.  A new source-side dense
matrix product is not excused merely because `P_ia` exists on the costate
side.

### 19.6 Necessary and sufficient literal-ownership theorem

At one seam let `E_l` be the set of matrix pairs appearing with nonzero
coefficient in (19.7), (19.12), and the persistent query, and let `W_l^res`
be the set/span of full ordered products literally available before
symmetrization.  The complete **source-independent cubic adjoint** is a
zero-new-dense-product resident object **if and only if** all of the following
hold:

1. **Complete costate ownership.** `Q4,Q31,Q22` and, when live,
   `Lambda211` are literal coordinates of the paid reverse tape at the same
   seam, frame, normalization, collision convention, and time.
2. **Word closure.** For every `(M,N) in E_l`, the exact ordered word `MN` is
   a literal resident product or an exact frozen linear view of
   `W_l^res`; in particular the square and factor--factor requirements in
   (19.17) are not omitted.
3. **Pre-symmetry retention.** The implementation exposes the full `MN`, or
   both its symmetric and skew halves, before reducing the q211 Gram.
4. **Common-factor recurrence.** The q211 slices obey the common-support
   recurrence and every `N_a` in (19.11) is the same resident D10 operand,
   rather than an anchor-dependent or reconstructed lookalike.
5. **Future completion.** \(\mathcal P_l^*q_3\) is zero, Bellman-dark, or itself
   a literal member of the same resident cubic-query module.  Otherwise it is
   a new positive-depth row.

Sufficiency is constructive from (19.2), (19.12), and (19.14)--(19.17): form
all vector images, take `Sym(P)` for the H3 Jordan atoms, take `Skew(P)` for
q211, and reduce the resident coefficients.  Necessity is exact for generic
symmetric operands.  Degree-two noncommutative words are linearly independent
in the free matrix algebra; a missing `MN` cannot be reconstructed from
unrelated products by rowwise/Hadamard reductions.  Likewise two products
with the same symmetric half and opposite skew halves give the same Jordan
state but different q211 values, while equal skew Gram reductions can have
different symmetric halves and hence different H3 actions.  Therefore a
failure of any literal word/coordinate condition admits two lawful local
states indistinguishable to the claimed resident interface but with different
full-seam H3 adjoints.

There is a strictly weaker scalar-consumer branch.  If word closure fails,
the **action on one specified source** is still zero-new-product exactly when
the restricted functional

\[
Z\longmapsto\langle Z,G_3^{\rm full}\rangle
\tag{19.21}
\]

has a native/factorized evaluator from resident operands--for example
(19.20), a shared eigenbasis, or the certified Fourier/Jordan evaluator of
Section 15--and that evaluator includes every live costate and persistent
term.  This is equality on the lawful source manifold; it does not make
`G3_full` resident or reusable for another source.  A source-evaluator receipt
and a future-costate receipt are therefore independent.

If the source-independent theorem passes, q211 and the immediate H3 source share product
arithmetic with zero additional dense latent multiplication; only the
non-GEMM reductions and source-evaluation error remain.  If it fails, the
missing product can be bypassed only by the source-restricted branch (19.21),
while a missing persistent row must still be charged.  A generic new D10 reverse
bundle remains about `20` GEMM-equivalents and is closed against Base48/49.

### Graph-ready split

- `P2R0G19B_MATH_H3_FULL_SEAM_TRILINEAR` — **solved identity**:
  (19.1)--(19.8).
- `P2R0G19B_MATH_H3_Q211_PULLBACK` — **solved identity**: direct and
  common-factor forms (19.10)--(19.12).
- `P2R0G19B_MATH_H3_Q211_COORDINATE_OBSTRUCTION` — **solved no-go**:
  off-diagonal 211 must appear in (19.13) unless literally compiled/dark.
- `P2R0G19B_MATH_H3_JORDAN_SKEW_COMPLEMENT` — **solved zero-product
  identity**: (19.14)--(19.16).
- `P2R0G19B_MATH_H3_Q211_PRODUCT_INVENTORY` — **solved exact gate**:
  (19.17); anchor--factor fusion alone omits squares/factor--factor words.
- `P2R0G19B_MATH_H3_FULL_SEAM_LITERAL_OWNERSHIP` — **solved iff theorem /
  open deployed receipt**: the five source-independent conditions and the
  source-restricted alternative (19.21).

Edges: `Q211_JORDAN_ANTISYM_FUSION -> H3_JORDAN_SKEW_COMPLEMENT`;
`H3_FULL_SEAM_TRILINEAR -> H3_Q211_PULLBACK`;
`H3_Q211_PULLBACK -> H3_Q211_PRODUCT_INVENTORY`;
`H3_Q211_COORDINATE_OBSTRUCTION`, `H3_JORDAN_SKEW_COMPLEMENT`, and
`H3_Q211_PRODUCT_INVENTORY -> H3_FULL_SEAM_LITERAL_OWNERSHIP`;
`H3_FULL_SEAM_LITERAL_OWNERSHIP -> H3_H4_JOINT_RAW_GATE`.

## 20. Checkpoint 14: exact tied-rank-one source factorization

Section 19.6 left a source-restricted loophole: the full cubic adjoint may
need matrix products which disappear after pairing with a special lawful
source.  For a tied symmetric-CP atom this collapse is exact for the complete
K4/K31/K22/q211 seam.

### 20.1 Universal scalar reduction

Let the H3 source at physical coordinate `p` be

\[
Z_p=\gamma_p c^{\otimes3},
\tag{20.1}
\]

with one latent direction `c` shared across every source coordinate involved
in the seam.  For `theta_i=(x_i,M_i)` define

\[
s_i=c^\top x_i,qquad v_i=M_ic,qquad
a_i=c^\top v_i,qquad
e_{ij}=x_i^\top v_j,qquad d_{ij}=v_i^\top v_j.
\tag{20.2}
\]

The basic Jordan contraction is

\[
\boxed{
\begin{aligned}
\langle c^{\otimes3},\mathcal C_x(M,N)\rangle
={}&(x^\top Nc)(c^\top Mc)
+(x^\top Mc)(c^\top Nc)\\
&+2(c^\top x)(Mc)^\top(Nc).
\end{aligned}}
\tag{20.3}
\]

Substitution into the universal tensor (19.2) gives

\[
\boxed{
\begin{aligned}
b_c(i,j,k)
:={}&\langle c^{\otimes3},
\mathcal B(\theta_i,\theta_j,\theta_k)\rangle\\
={}&6s_is_js_k\\
&+12\{e_{ik}a_j+e_{ij}a_k+2s_i d_{jk}\\
&\hspace{9mm}+e_{jk}a_i+e_{ji}a_k+2s_jd_{ik}\\
&\hspace{9mm}+e_{kj}a_i+e_{ki}a_j+2s_kd_{ij}\}.
\end{aligned}}
\tag{20.4}
\]

This is the complete scalar source adjoint.  In particular,

\[
\boxed{
\begin{aligned}
c^\top M^2c&=\|Mc\|^2,\\
c^\top(M\circ N)c&=(Mc)^\top(Nc),\\
c^\top(N_a\circ N_b)c&=(N_ac)^\top(N_bc).
\end{aligned}}
\tag{20.5}
\]

Thus every `M^2`, `N^2`, and `N_aN_b` in the product inventory (19.17)
reduces to a vector norm or bilinear.  No latent matrix product is required.
The reductions include every non-product term as well: `J_{Nx}(M)` becomes
`(x^T Nc)(c^T Mc)`, and the pure H1 star becomes `6s_i s_j s_k`.

Combining (20.4) with the slot table (19.6) evaluates the complete
K4/K31/K22 source action.  Combining it with (19.10) evaluates q211.  Under
the common-factor costate (19.11), the latter is

\[
\boxed{
\begin{aligned}
\chi^{211}_{c}
=2\sum_p\gamma_p\{&
\sum_{a,b}(\Theta_p)_{ab}b_c(p,a,b)\\
&+\sum_{i,a,b}U_{pa}(\Theta_i)_{ab}b_c(i,i,b)\}.
\end{aligned}}
\tag{20.6}
\]

Here the aggregate coordinates are lawful vector reductions:

\[
u_a=\sum_iU_{ia}x_i,qquad
N_ac=\sum_iU_{ia}(M_ic),qquad
c^\top u_a=\sum_iU_{ia}s_i.
\tag{20.7}
\]

No `N_a` matrix has to be squared or multiplied by another factor matrix.

### 20.2 Exact scope of the tied-source premise

The collapse is source-specific.  It holds for a shared dictionary

\[
Z_p=\sum_{q=1}^{k_s}\gamma_{pq}c_q^{\otimes3}
\tag{20.8}
\]

by summing (20.4) independently over `q`.  Linearity means that no
`q,q'` cross-products arise in a first-order H3 action.

The saving fails for generic unit-specific atoms.  If
\(Z_p=\gamma_p c_p^{\otimes3}\), the q211 term for source `p` requires
`M_j c_p` for all live `(p,j)`.  Without a shared `k_s`-atom dictionary this
is an `n by n by r` vector bank and `O(n^2 r^2)` matrix-vector work per seam,
rather than the tied `O(n r^2)` work.  A small CP rank for each source
separately is not sufficient; the **union** of source directions must have
small shared rank.

Nor does (20.5) make the source tensor itself resident.  It evaluates one
fixed linear functional.  Changing the source, differentiating with respect
to its atom, or exposing `G3` to another consumer requires the corresponding
new vector bank or the source-independent word closure of Section 19.6.

### 20.3 Minimal state and arithmetic

Let `k_s` be the number of tied source atoms and `k_f` the q211 common-factor
width.  If the AQ cores `(x_i,M_i)` and costate factors are already resident,
the only dense latent preparation is

\[
V_{iq}=M_i c_q.
\tag{20.9}
\]

Its exact leading census per layer is

\[
\boxed{
\text{work }O(nk_s r^2),qquad
\text{state }nk_s r,qquad
\text{width-GEMM equivalent }{k_s r^2\over n^2}.}
\tag{20.10}
\]

Across `D` active layers the last quantity is `D k_s r^2/n^2`.  If the
`M_i c_q` vectors are literal resident arrays, the incremental preparation
and persistent state are zero; scalar projections can be streamed with
`O(k_s r+k_f k_s r)` scratch.

The full common-factor q211 reduction (20.6) costs

\[
\boxed{O(nk_s k_f^2 r)}
\tag{20.11}
\]

rowwise/vector arithmetic after the resident `Theta_i` coefficients are
read.  A literally diagonal `Theta_i` reduces this to `O(n k_s k_f r)`.
Normalized by a width GEMM these are respectively
`k_s k_f^2 r/n^2` and `k_s k_f r/n^2` per layer.  No `r^3` term remains.

The minimal additional live state for one already-resident tied atom is one
`n by r` vector bank if (20.9) is not resident, plus `O(k_f r)` aggregate
vectors and scalar scratch.  Materializing pair Grams `d_ij` is unnecessary;
each required dot product is reduced directly under the resident costate.
External mask/common-factor reduction remains a separate cost: the rank-one
latent collapse does not lower an unfactorized external q211 costate.

### 20.4 Exact D10 vector-recurrence gate

Local source contraction does not imply that the vector bank closes through
depth.  Let `C` be the matrix whose columns are the tied directions and define

\[
\mathcal V_C(M)=MC.
\tag{20.12}
\]

For a fixed symmetric multiplier `N`, the Jordan update is

\[
T_N(M)=M\circ N,qquad
T_N(M)C={M(NC)+N(MC)\over2}.
\tag{20.13}
\]

Knowing only `MC` determines (20.13) for every symmetric `M` if and only if

\[
\boxed{N\operatorname{range}C\subseteq\operatorname{range}C.}
\tag{20.14}
\]

Sufficiency is constructive: if `NC=CR`, then

\[
T_N(M)C={ (MC)R+N(MC)\over2}.
\tag{20.15}
\]

For necessity, if `Nc` has a component outside `range C`, choose two symmetric
matrices agreeing on `range C` but acting differently on that component.
They have identical `MC` banks but different `M(Nc)`, hence different
Jordan-update banks.  Equivalently,

\[
\ker\mathcal V_C\subseteq\ker(\mathcal V_C\circ T_N)
\quad\Longleftrightarrow\quad(20.14).
\tag{20.16}
\]

For one tied atom, (20.14) says exactly that `c` is an eigenvector of `N`.
Across D10, a one-atom vector-only recurrence therefore requires `c` to be a
common eigenvector of every live Jordan multiplier.  A `k_s`-atom bank
requires one common invariant subspace of dimension at most `k_s`.  A generic
family of symmetric, noncommuting reset/core matrices has no proper common
invariant subspace, so its exact tied bank expands to the full latent space.

If the full matrices remain resident, one may recompute (20.9) independently
at every seam for the cost (20.10); condition (20.14) is then unnecessary.
This is a reset-local evaluation strategy, not a free persistent carrier.

### 20.5 Transport and nonlinear-reset closure

Let \(\mathcal D_l=\operatorname{span}\{c_{q,l}^{\otimes3}\}\) be the tied
cubic dictionary at cut `l` and
let `R_l` be the latent linear pullback.  Exact transport closure requires

\[
\boxed{R_l^{\otimes3}\mathcal D_{l+1}\subseteq\mathcal D_l.}
\tag{20.17}
\]

For one atom this means `R_l c_{l+1}` is proportional to `c_l`.  A shared
multi-atom dictionary must contain every transported cube; invariance of the
vector span alone is not always sufficient because the cube of a nontrivial
linear combination contains mixed cubic atoms absent from a pure-cube list.

The nonlinear AQ ReLU reset is still more restrictive.  Its exact H3 source
contains the Jordan family

\[
E[\delta'(g)p^{\otimes3}+3\delta(g)J_p(M)]
\tag{20.18}
\]

in the half-quadratic convention.  It lies in a one-atom cube span only in a
degenerate affine/common-direction case, or after an explicit residual
projection proving every Jordan component old or Bellman-dark.  Generic
identity-specific `M` creates `J_c(M)` and higher Krylov terms, so tied pure
cubes are not closed through correlated ReLU resets.

The disposition is therefore:

- **GO as a reset-local scalar factorization** when the lawful residual source
  is tied/shared-CP and the complete future costate is resident;
- **conditional GO as a recomputed D10 evaluator** at cost (20.10) per seam
  when full AQ matrices remain resident;
- **NO as a free persistent one-vector carrier** without the common invariant
  subspace (20.14), transport closure (20.17), and reset-source closure
  (20.18).

### 20.6 Double-counting and value ownership

The identities above are computational and earn no raw credit by themselves.
The lawful source is

\[
\Delta Z_p=(I-\Pi_{old\ H31})\gamma_p c^{\otimes3},
\tag{20.19}
\]

with the affine `T0` atom retained in the synchronization residual until the
pending CP3/H31 seam-and-normalization audit passes.  The complete scalar must
be formed once from the full seam costate.  Its K4/K31/K22 collision pieces
and its off-diagonal q211 piece cannot also be booked as separate gains when
they are coordinates of the same external quartic action.

The AQ commutator baseline and the linear H3 term are different perturbation
orders, but their response vectors can correlate.  Let `F_tied` be the
corpus feature produced by (20.4)--(20.6).  Its only new contribution is

\[
\boxed{(I-P_{[S_{old},F_{H4},F_{q211}^{AQ}]})F_{tied}.}
\tag{20.20}
\]

It must be included in the same joint Gram as the H4-score and q211 baseline,
not added by diagram labels.  The existing promotion gates remain binding:
joint Schur raw gain at least `2.86e-10`, and any H3 carrier credited toward
the goal portfolio must retain correlation above `0.9736` or squared tail
below `5.210%` under the stated capacity decomposition.

### Graph-ready split

- `P2R0G19B_MATH_H3_TIED_SOURCE_SCALAR` — **solved identity**:
  (20.3)--(20.7).
- `P2R0G19B_MATH_H3_TIED_SOURCE_SCOPE` — **solved rank theorem**: shared
  dictionary (20.8); per-unit low rank is insufficient.
- `P2R0G19B_MATH_H3_TIED_SOURCE_COST` — **solved census**:
  (20.9)--(20.11), conditional on resident complete costates.
- `P2R0G19B_MATH_H3_TIED_JORDAN_RECURRENCE` — **solved iff theorem**:
  common invariant-space gate (20.14)--(20.16).
- `P2R0G19B_MATH_H3_TIED_RESET_CLOSURE` — **conditional no-go**:
  (20.17)--(20.18); generic correlated reset leaves the pure-cube span.
- `P2R0G19B_MATH_H3_TIED_NO_DOUBLE_COUNT` — **solved ownership rule / open
  value receipt**: (20.19)--(20.20).

Edges: `H3_FULL_SEAM_TRILINEAR -> H3_TIED_SOURCE_SCALAR`;
`H3_Q211_PULLBACK -> H3_TIED_SOURCE_SCALAR`;
`H3_TIED_SOURCE_SCOPE -> H3_TIED_SOURCE_COST`;
`H3_TIED_JORDAN_RECURRENCE` and `H3_TIED_RESET_CLOSURE` ->
`H3_TIED_SOURCE_COST`;
`H3_TIED_SOURCE_SCALAR`, `H3_H4_D11_OWNERSHIP`, and
`Q211_JORDAN_ANTISYM_FUSION -> H3_TIED_NO_DOUBLE_COUNT`.

## 21. Owner-correct signed-H3 endpoint tail for the r62 AQ/Price/Stein shell

The current late-cut shell combines a rank-62 affine--quadratic trajectory
with Price/Stein quartic readouts.  None of the following three statements is
equivalent to either of the other two:

1. the rank-62 state can match the weak `(C,K21)` cut interface;
2. its AQ law determines one modeled symmetric cubic tensor; and
3. the deployed endpoint retains enough of the **true old-residual signed-H3
   value** to cross the raw target.

This section gives the exact implication boundary.  In particular, the 561
Veronese-null directions at r62 are a weak-interface section obstruction, not
an H3 endpoint-tail estimate.

### 21.1 Tight capacity constant

The three-channel capacity plane is

\[
14.7530718\alpha_v+4.370\alpha_3+1.0313\alpha_4
>19.4090718,
\tag{21.1}
\]

with each `alpha` a retained value-energy fraction in the same
incumbent-residual endpoint metric.  Under the currently quoted complement

\[
q_v=q_4=0.98346,
\qquad
\alpha_v=\alpha_4=0.98346^2=0.9671935716,
\]

the exact H3 threshold induced by (21.1) is

\[
\boxed{
\alpha_{3,*}
=\frac{19.4090718-(14.7530718+1.0313)(0.98346)^2}{4.370}
\simeq0.9479471.}
\tag{21.2}
\]

Equivalently,

\[
\boxed{
q_3>\sqrt{\alpha_{3,*}}\simeq0.973626,
\qquad
\delta_{3,*}:=1-\alpha_{3,*}\simeq0.0520529.}
\tag{21.3}
\]

Thus the exact conditional wording is **more than about 94.7947% of the
old-residual H3 response energy**, or less than about 5.2053% tail.  The old
`q3>0.9736`, `tail<0.05210` numbers are harmless rounded summaries, but
(21.2)--(21.3) are the binding constants when the `.98346` complement is
used.  If either complement changes, (21.1), rather than the rounded number,
must be reevaluated.

### 21.2 Exact response-space ownership projector

Let `Y` be the corpus/endpoint response space with task metric `V`, let `O`
contain every **literally verified** old response column, and let

\[
R_O=I-P_O^V
\]

be the `V`-orthogonal complete-old residualizer.  In particular, an
unverified D14 signed-H3 spelling or the still-open CP3 affine-H3
normalization equality is not included in `O` by name alone.

For each H3 birth cut `l`, let

\[
F_l:\mathcal Z_{3,l}\longrightarrow\mathcal Y
\]

be the complete future response, including the K4, K31, K22, persistent-H3,
and lower-order seam emissions.  Stack the active cuts as

\[
F=[F_l]_l,
\qquad z=(z_l)_l,
\qquad h_3=R_OFz.
\tag{21.4}
\]

Let `A_62,3` denote only the response columns generated by a lawful r62 H3
source/trajectory and let `A_J` denote a separately supplied oriented Stein
provider, if one exists.  The typed H3 response space is

\[
\boxed{
\mathcal U_{62,PS}^{(3)}
=R_O\operatorname{span}[A_{62,3},A_J].}
\tag{21.5}
\]

The owner-correct retained fraction is exactly

\[
\boxed{
\alpha_3^{62,PS}
=\frac{\|P_{\mathcal U_{62,PS}^{(3)}}^Vh_3\|_V^2}
       {\|h_3\|_V^2},
\qquad
1-\alpha_3^{62,PS}
=\frac{\|(I-P_{\mathcal U_{62,PS}^{(3)}}^V)h_3\|_V^2}
       {\|h_3\|_V^2}.}
\tag{21.6}
\]

For a nonlinear AQ trajectory, (21.6) is an optimistic local/refit envelope
when `A_62,3` is taken to be the full tangent span.  The actual nonlinear
model set can only do worse unless a local section and its finite radius are
proved.  For a fixed single compiled correction, replace the projector by
the best scalar-calibrated response; the same squared-correlation gate
results.

The exact promotion condition is

\[
\boxed{\alpha_3^{62,PS}>\alpha_{3,*}.}
\tag{21.7}
\]

No source Frobenius fit, local moment fit, parameter count, or marginal
feature correlation substitutes for (21.6).

### 21.3 Price/Stein typing: what can and cannot enter (21.5)

For the half-quadratic AQ variables

\[
G_i=u_i^Tz+\frac12(z^TA_i z-\operatorname{tr}A_i),
\]

the centered symmetric Stein matrix has the exact linear projection

\[
J_{ij,k}
=E[(\Gamma^s_{ij}-C_{ij})G_k]
=\frac34\langle A_iu_j+A_ju_i,u_k\rangle
\frac12\operatorname{tr}(A_iA_jA_k),
\tag{21.8}
\]

and

\[
\boxed{
\kappa_3(G_i,G_j,G_k)
=\frac23(J_{ij,k}+J_{ik,j}+J_{jk,i}).}
\tag{21.9}
\]

Consequently the ownership table is exact:

- the AQ `(u,A)` bank determines its **modeled** `B3/kappa3` law; this law can
  enter (21.5) only through the literal AQ recurrence or an explicitly billed
  H3 consumer, and it does not identify true-minus-AQ H3;
- symmetric `K21` exposes only repeated/cyclic cubic faces and does not
  reconstruct the oriented `J` coordinates;
- the Price q211 partial trace `Gamma_L` is a quartic future-costate collapse
  and contributes zero typed H3 source dimension;
- the Stein-square covariance `K^Gamma` is an even H4 Gram.  A Gram does not
  recover the sign/orientation of `J`, so it also contributes zero typed H3
  dimension;
- an oriented `J` computed from the same `(u,A)` is a useful readout but does
  not enlarge the AQ source model class; an **independently acquired true
  residual** `J` would enlarge (21.5), but no such resident provider is in the
  current D10/D11/r62 declaration.

The literal audit reports no unsymmetrized `J_{ij,k}` carrier in the current
lineage.  D11 exposes symmetric K21/K31/K22, CP4/star/Q24 faces; the proposed
Price object is the even q211 partial trace; and the Stein-square is even.
Therefore the current Price/Stein additions supply **no independent signed-H3
credit** in (21.6).  They may earn separately residualized q4 value, and their
responses belong in the joint Schur Gram, but they may not be used to increase
`alpha_3` in the channel capacity plane.

This is also an information theorem.  Opposite all-distinct odd perturbations
can share covariance, repeated cubic faces, and every even Price/Stein
observable while reversing the signed-H3 endpoint response.  Hence an
even-only provider has zero uniform signed-H3 identifiability, regardless of
its quartic accuracy.

### 21.4 Why r62's 561 modes do not bound the signed-H3 tail

Let

\[
D_{21}:\operatorname{Sym}^3\mathbb R^n\longrightarrow\mathbb R^{n\times n},
\qquad
(D_{21}T)_{ij}=T_{iij},
\tag{21.10}
\]

be the repeated-slot observation used by the weak seam.  Every symmetric
tensor supported on strict triples lies in its kernel.  At width 1024 this
subspace has dimension

\[
\boxed{
\dim\mathcal Z_{3,\mathrm{dist}}
=\binom{1024}{3}=178,433,024.}
\tag{21.11}
\]

Thus exact matching of `(C,K21)` places no norm bound at all on the strict
all-distinct cubic defect

\[
\Delta Z_3
=K_3^{true}-K_3^{AQ62}
\quad\text{inside }\ker D_{21}.
\tag{21.12}
\]

For r62, the AQ quotient has

\[
d_{62}=\frac{62(62+3)}2=2015,
\]

\[
p_{62}^{AQ}=1024(2015)-\frac{62\cdot61}{2}=2,061,469,
\]

and hence weak-interface fiber dimension

\[
\boxed{
g_{62}=p_{62}^{AQ}-1,573,376=488,093.}
\tag{21.13}
\]

Even granting a regular exact weak section, the infinitesimal AQ fiber can
move through at most 488,093 strict-H3 directions.  Against the unrestricted
strict sector this leaves at least

\[
\boxed{178,433,024-488,093=177,944,931}
\tag{21.14}
\]

source directions outside a generic fiber image.  This is a universal-source
dimension obstruction, not a claim that the realized D14 source is generic;
the actual endpoint quotient can be much smaller and must be measured or
proved.

By contrast, the number 561 is

\[
\binom{33+1}{2}=561,
\]

the prefix-independent first-order cokernel lower bound for the **weak
`(C,K21)` section Schur map** at r62.  A nonzero cubic minor can repair those
561 weak-interface normals.  Even perfect repair only establishes weak seam
existence; it says nothing about (21.12)--(21.14) and therefore gives no
positive lower bound on (21.6).

The exact fixed-suffix escape is a future quotient.  Put

\[
H_3=R_OF|_{\mathcal Z_{3,\mathrm{dist}}}
\]

and let `A_fib` be the strict-H3 differential generated by the lawful r62
trajectory fiber.  Uniform local endpoint completion is equivalent to

\[
\boxed{
\operatorname{range}(H_3A_{fib})=\operatorname{range}(H_3),}
\tag{21.15}
\]

or, dually,

\[
\boxed{
\ker(A_{fib}^*H_3^*)=\ker(H_3^*).}
\tag{21.16}
\]

For exact completion from the repeated weak interface alone, the sharper
factor-through condition is

\[
\boxed{
R_OF(\ker D_{21})\subseteq\mathcal U_{62,PS}^{(3)}.}
\tag{21.17}
\]

One dense future mix followed by a repeated read has zero suffix-uniform dark
kernel by polarization, so (21.17) is not an architecture-free identity.  It
can hold only for the supplied fixed suffix and must be proved in its actual
endpoint metric.

### 21.5 Complete-depth block Gram and coherent error budget

Define the endpoint block Gram

\[
\boxed{
K_{lm}=F_l^*R_O^*VR_OF_m.}
\tag{21.18}
\]

If `e_l` are the source-equivalent residuals after the best lawful r62 H3
fit, then the exact tail is

\[
\boxed{
1-\alpha_3^{62,PS}
=\frac{\sum_{l,m}\langle e_l,K_{lm}e_m\rangle}
       {\sum_{l,m}\langle z_l,K_{lm}z_m\rangle}.}
\tag{21.19}
\]

This formula automatically includes the late-cut seam defect, every suffix
reset birth, persistent-H3 transport, Price/Stein evaluation truncation, and
their cross-layer cancellation or reinforcement.  Per-layer Frobenius tails
or an average of local correlations do not imply (21.19).

For a decomposition of the endpoint error into seam, reset/model, reverse,
and synchronization pieces, define their normalized endpoint amplitudes
`epsilon_a`.  Without cross information the sharp coherent sufficient gate
is

\[
\boxed{
\sum_a\epsilon_a<\sqrt{\delta_{3,*}}\simeq0.22815.}
\tag{21.20}
\]

The exact block Gram (21.18), not an arbitrary equal split of (21.20), is the
preferred certificate.

For a uniform guarantee over an admissible source tangent with basis `Q`
modulo `ker H3`, the binding restricted singular receipt is

\[
\boxed{
\tau_{62,PS}^2
=\lambda_{min}\left(
Q^*H_3^*VP_{\mathcal U_{62,PS}^{(3)}}^VH_3Q,
Q^*H_3^*VH_3Q
\right)>\alpha_{3,*}.}
\tag{21.21}
\]

For the one realized source, replace the minimum generalized eigenvalue by
its Rayleigh quotient.  For a rank-`k` typed H3 dictionary, (21.21) reduces
to the endpoint-whitened singular-tail condition

\[
\boxed{
\frac{\sum_{j>k}\sigma_j^2}{\sum_j\sigma_j^2}
<\delta_{3,*}\simeq0.0520529.}
\tag{21.22}
\]

The dimension or CP rank of a q211/Stein-square dictionary is irrelevant to
(21.22), because those are not typed H3 columns.

### 21.6 GO/NO-GO theorem for the current shell

The r62 AQ/Price/Stein shell is a signed-H3 **GO** only if all four conditions
hold:

1. the weak r62 seam exists with its 561-mode cubic-section and conditioning
   receipt;
2. the literal AQ recurrence or an explicitly billed oriented `J` consumer
   exposes the modeled AQ H3 response at every claimed suffix seam;
3. complete-old residualization includes only verified D14/CP3/H31 ownership,
   and q211/Stein-square value is booked outside the H3 numerator; and
4. the actual endpoint projector (21.6), or the stronger restricted receipt
   (21.21), exceeds (21.2).

Failure of condition 1 closes the weak AQ seam.  Passing condition 1 does not
advance condition 4.  Failure of condition 2 means the shell has no new
literal H3 consumer.  Failure of condition 4 closes raw-`<1e-8` under the
quoted `.98346` variance/quartic complement, regardless of how accurately the
Price/q211 quartic surface is evaluated.

The present state is therefore:

\[
\boxed{
\text{r62 weak seam: conditional/open; Price/Stein H3 ownership: absent;
H3 endpoint retention: unproved.}}
\tag{21.23}
\]

There is no lawful positive uniform H3-retention bound from the current state
description alone.  The worst-case bound is zero because the weak interface
and even Price/Stein observables admit opposite future-visible all-distinct
odd sources.  This does not prove the realized network tail is large; it
proves that the missing receipt must be endpoint/source-specific.

### 21.7 Atomic graph split and next question

- `P2R0G19B_MATH_H3_R62_TIGHT_VALUE_GATE` -- **solved exact conditional
  constant**: (21.1)--(21.3), `alpha3>0.9479471`, tail `<0.0520529` under the
  `.98346` complement.
- `P2R0G19B_MATH_H3_OWNER_RESPONSE_PROJECTOR` -- **solved definition/iff
  gate**: (21.4)--(21.7); only literal H3-bearing columns enter the numerator.
- `P2R0G19B_MATH_H3_PRICE_STEIN_TYPE_SPLIT` -- **solved ownership no-go for
  the current declaration**: Price q211 trace and Stein square are even;
  oriented `J` is absent, (21.8)--(21.9).
- `P2R0G19B_MATH_H3_R62_WEAK_SEAM_NOT_TAIL` -- **solved obstruction**:
  strict-H3 dimension (21.11), AQ fiber bound (21.13)--(21.14), and the 561
  weak-seam modes are different spaces.
- `P2R0G19B_MATH_H3_R62_ENDPOINT_SECTION` -- **open high-priority theorem**:
  prove (21.15)--(21.17) or certify the realized tail.
- `P2R0G19B_MATH_H3_R62_DEPTH_GRAM` -- **open receipt with solved formula**:
  (21.18)--(21.22), including every seam/reset/reverse cross term.
- `P2R0G19B_MATH_H3_R62_PRICE_STEIN_DISPOSITION` -- **HOLD**: (21.23); no
  raw promotion until a literal H3 consumer and `>94.7947%` owner-correct
  endpoint retention are supplied.

The single highest-value next mathematical question is now finite and
owner-correct: for the supplied late suffix, what is the range or restricted
singular spectrum of `R_O F` on the strict all-distinct H3 defect after
quotienting the actual r62 trajectory fiber?  A positive answer is (21.21);
a nonzero endpoint row in the remaining quotient is an immediate exact
counterexample to complete H3 ownership.  The 561-by-561 weak-seam cubic
minor should be pursued in parallel, not used as a surrogate for this test.

## 22. The r62/CP3-threshold-JVP oriented-Stein audit

The proposed reopening is to use the incumbent CP3 path and a threshold JVP
to expose the unsymmetrized Stein projection `J_ij,k`.  There is an exact
algebraic split:

- the cyclic part of `J` is precisely the ordinary symmetric H3 tensor and is
  already the CP3 ownership question;
- the genuinely unsymmetrized part is invisible to every symmetric H3
  consumer and can earn value only through a distinguished-slot Stein/H4
  action.

Thus an owner-correct `J` provider cannot be credited as a second signed-H3
carrier.  The only possible deployable novelty is a separately typed
pair-versus-singleton provider.

### 22.1 Orthogonal cyclic decomposition

Let

\[
\mathcal J=\operatorname{Sym}^2(\mathbb R^n)\otimes\mathbb R^n
\]

with the ordered-slot Frobenius inner product, and define

\[
(\mathcal C J)_{ijk}
=\frac23\left(J_{ij,k}+J_{ik,j}+J_{jk,i}\right).
\tag{22.1}
\]

For the AQ Stein projection of (21.8),

\[
\mathcal CJ=\kappa_3.
\tag{22.2}
\]

If a fully symmetric tensor is embedded in `J` by regarding its first two
slots as the pair, direct summation gives

\[
\boxed{
\mathcal C^*G=2G,
\qquad
\mathcal C\mathcal C^*=4I.}
\tag{22.3}
\]

Therefore the exact orthogonal projector onto the cyclic/H3-owned subspace is

\[
\boxed{
P_{cyc}=\mathcal C^*(\mathcal C\mathcal C^*)^{-1}\mathcal C
=\frac14\mathcal C^*\mathcal C,}
\tag{22.4}
\]

and every oriented provider has the unique decomposition

\[
\boxed{
J=\frac12\kappa_3+J_\perp,
\qquad
J^\perp_{ij,k}+J^\perp_{ik,j}+J^\perp_{jk,i}=0.}
\tag{22.5}
\]

For any fully symmetric future H3 query `G`,

\[
\boxed{
\langle G,\kappa_3\rangle
=2\langle G,J\rangle,
\qquad
\langle G,J_\perp\rangle=0.}
\tag{22.6}
\]

This is the decisive ownership theorem.  On a strict-all-distinct or
collision-weighted D11 coordinate system, `C*` and the numerical constants in
(22.3)--(22.5) must be formed in that induced measure.  The invariant
conclusion survives unchanged: `ker C` is invisible to every fully symmetric
H3 query.

For an additive r62 provider beside incumbent CP3, the owner-correct source
split is consequently

\[
\boxed{
J^{AQ62}-J^{old}
=\frac12(\kappa_3^{AQ62}-\kappa_3^{CP3,old})
+J_\perp^{AQ62}-J_\perp^{old}.}
\tag{22.7}
\]

The current old tape has no oriented `J_perp` field, so its algebraic
oriented projection is not identified by name.  Response-space subtraction
against every old column remains mandatory.  The first term in (22.7) is
exactly the ordinary symmetric H3 replacement and may be booked only once.
The second term has zero H3-channel value by (22.6).

### 22.2 A true CP3 threshold JVP cannot expose `J_perp`

Write a symmetric CP3 bank as

\[
K_3(b)=\sum_{s=1}^R\lambda_s(b)a_s(b)^{\otimes3}.
\]

Its exact threshold JVP is

\[
\boxed{
\dot K_3
=\sum_s\left[
\dot\lambda_s a_s^{\otimes3}
+3\lambda_s\operatorname{Sym}
(a_s^{\otimes2}\otimes\dot a_s)
\right]\in\operatorname{Sym}^3.}
\tag{22.8}
\]

Hence its canonical oriented lift is `dot K3/2`, lies entirely in
`range C*`, and has zero `J_perp` component.  Differentiating the already
assembled symmetric CP3 tensor does not undo its lost slot orientation.

The same conclusion holds for an arbitrary number of scalar threshold
directions as long as the interface exports only the assembled symmetric JVP.
A model-specific reconstruction of `J_perp` from a family of such JVPs would
need a separate injective decoder theorem from resident `(A,u)` data; in that
case `(A,u)`, not CP3 symmetry, is the information source.

There is a mathematically different object.  Before symmetrization, retain
one distinguished singleton slot of one atom:

\[
L^{slot}_{ij,k}=a_i a_j\dot a_k.
\tag{22.9}
\]

Its owner-correct cyclic-null component is exactly

\[
\boxed{
(L^{slot}_\perp)_{ij,k}
=\frac23a_i a_j\dot a_k
-\frac13(a_i\dot a_j+\dot a_i a_j)a_k.}
\tag{22.10}
\]

It has pair-versus-singleton Schmidt rank at most two and total pair spectral
rank at most three: one rank-one pair `aa^T` with singleton `dot a`, and one
rank-at-most-two pair `a dot a^T+dot a a^T` with singleton `a`.

But (22.9) is **not** the JVP (22.8).  It is a new typed pre-symmetrization
interface.  If the three slots subsequently undergo the same transport and
are mapped to an actual symmetric H3 tensor, (22.10) is annihilated at birth
and can never reappear.  It matters only when the pair and singleton slots
meet different future operators, exactly as in the Stein-square/Price
mechanism.

### 22.3 Literal D11 residency verdict

The current D11 spelling has the following exact ownership:

1. production CP3 factors plus the cubic star form the complete **assembled
   modeled symmetric H3** interface used by Q24;
2. the same-site H31/K31 action of that assembled tensor is an old duplicate;
3. the exact true-minus-assembled directional D3 reset interface is absent;
4. no `J_ij,k`, no pair-versus-singleton `J_perp`, and no slot-labelled CP3
   threshold derivative is retained; and
5. no endpoint reverse costate for such an oriented field is resident at the
   source seam.

The CP3 affine rank-one H3 source has the known formula
`-sigma alpha phi(alpha)e^tensor3`, but its exact equality to the stored CP3
source at the same seam, normalization, collision convention, and positive
depth remains the separate affine-H3 synchronization audit.  It cannot be
silently removed from the true residual.

Thus a symmetric CP3 threshold JVP is either an old-model tangent/duplicate or
a new symmetric refit column.  It is not an oriented provider.  A
distinguished-slot implementation would be a new state, source callback, and
reverse row; none is literal in the current lineage.

### 22.4 Tensor-free contraction of a lawful AQ `J`

If the r62 `(A_i,u_i)` cores are resident and a lawful oriented future mask
is independently supplied, `J` can be contracted without materializing its
`n^3` coordinates.  Let the pair-versus-singleton factorization be

\[
L_{ij,k}=\sum_{a=1}^q S^{(a)}_{ij}v^{(a)}_k,
\qquad
S^{(a)}=\sum_{p=1}^{s_a}\lambda_{ap}x_{ap}x_{ap}^T,
\tag{22.11}
\]

and put

\[
K=\sum_{a=1}^qs_a,
\]

\[
A_x=\sum_i x_iA_i,quad u_x=\sum_i x_iu_i,quad
A_v=\sum_kv_kA_k,quad u_v=\sum_kv_ku_k.
\]

Substitution into (21.8) gives the exact scalar

\[
\boxed{
\langle L,J\rangle
=\sum_{a,p}\lambda_{ap}left[
\frac32\langle A_{x_{ap}}u_{x_{ap}},u_{v^{(a)}}\rangle
+\frac12\operatorname{tr}
(A_{x_{ap}}^2A_{v^{(a)}})
\right].}
\tag{22.12}
\]

The owner-correct action uses `L_perp=(I-P_cyc)L` against `J`, or,
equivalently, `L` against `J_perp`, followed by the complete response-space
old projector.  For a fully symmetric H3 costate, `L_perp=0`; (22.12) then
reduces to the CP3-owned cyclic action and has no oriented novelty.

The resource parameter is not ordinary matrix rank of one endpoint mask.  It
is the pair-versus-singleton Schmidt rank `q` together with the sum `K` of the
pair-slice spectral ranks.  A generic oriented costate has `q=n` and each
slice has rank `n`, hence `K` of order `n^2`.  Even a singleton-Schmidt-rank-one
query has `K=n` when its pair matrix is dense.

### 22.5 r62 five-reset cost gate

The conditional r62 terminal-factored five-reset shell costs `47.43295`, so
its optimistic pre-hook headroom is

\[
H=49-47.43295=1.56705.
\tag{22.13}
\]

Let `d62=2015`.  If none of the oriented external factors or aggregated AQ
cores is resident, a direct factor implementation of (22.12) has the leading
five-reset bill

\[
\boxed{
C_J^{fresh}
=5\left[
(K+q)\left(\frac1{1024}+\frac{2015}{1024^2}\right)
+K\frac{62^3}{1024^3}
\right].}
\tag{22.14}
\]

The three terms are external factor pullback, aggregation of the resident
`n by d62` AQ core bank, and one latent cubic contraction per pair spectral
atom.  Hooks, slot sewing, source formation, stabilization, and memory are
not included.  Numerically,

\[
C_J^{fresh}\simeq0.0156009K+0.0144911q.
\tag{22.15}
\]

Therefore the optimistic ceilings are

\[
\boxed{
q=1:\ K\le99,
\qquad
q=K:\ K\le52,}
\tag{22.16}
\]

before all omitted overhead.  A dense rank-one pair mask has `K=1024` and is
already closed.

If all required external factors are literal resident views but the mixed AQ
cores are not, the corresponding optimistic bill is

\[
\boxed{
C_J^{factor-resident}
=5\left[(K+q)\frac{2015}{1024^2}
+K\frac{62^3}{1024^3}\right]
\simeq0.0107181K+0.00960827q,}
\tag{22.17}
\]

giving `K<=145` at `q=1` or `K<=77` at `q=K`.  The current D11 audit does not
satisfy this residency premise for an oriented reverse row.

For `R` generic distinguished-slot CP atoms, (22.10) gives

\[
q\le2R,qquad K\le3R.
\tag{22.18}
\]

Equations (22.14)--(22.18) then give the optimistic screens

\[
\boxed{
R\le20\quad\text{fresh},
\qquad
R\le30\quad\text{with literal factor residency},}
\tag{22.19}
\]

again before every hook and acquisition cost.  A nonlinear reset may append
new CP births and increase these ranks.  Promotion requires the maximum
oriented cyclic-module ranks at every one of the five seams, not merely the
terminal atom count.

### 22.6 Endpoint-tail and channel attribution

The symmetric owner-correct H3 candidate is

\[
\Delta\kappa_3
=R_O(\kappa_3^{true}-\kappa_3^{AQ62/CP3}),
\tag{22.20}
\]

with the exact choice of replacement versus additive old subtraction stated
once.  Its endpoint response must satisfy Section 21's binding gate

\[
\boxed{
\alpha_3(\Delta\kappa_3)>0.9479471,
\qquad
q_3>0.973626.}
\tag{22.21}
\]

If only `R` symmetric CP/JVP response modes are retained, the exact
endpoint-whitened condition is

\[
\boxed{
\frac{\sum_{j>R}\sigma_j^2}{\sum_j\sigma_j^2}
<0.0520529.}
\tag{22.22}
\]

The oriented residual `J_perp` does not change the left side: by (22.6),

\[
\boxed{
\alpha_3(\operatorname{span}[\Delta\kappa_3,J_\perp])
=\alpha_3(\Delta\kappa_3)}
\tag{22.23}
\]

for the typed symmetric-H3 channel.  Any raw gain from `J_perp` belongs to the
Stein/H4 or distinguished-slot mixed channel and must be placed in the joint
Schur Gram without relabelling it as H3 retention.

There is no present receipt for (22.21)--(22.22).  Keeping all assembled model
CP3 births proves modeled-state completeness, not correlation with the
true-minus-assembled signed-H3 endpoint residual.  The exact residual D3
source is not exposed by the current white-box state.

### 22.7 Disposition and atomic graph split

The `r62 + CP3 threshold JVP` idea has a sharp disposition:

- as an **additional H3 provider**, it is closed: the true CP3 JVP is
  symmetric/cyclic, while the owner-correct unsymmetrized remainder is H3-dark;
- as a **replacement/refit H3 model**, it reduces to the ordinary r62/CP3
  symmetric source and remains open only through (22.21)--(22.22);
- as a **Stein/H4 distinguished-slot provider**, it is conditional on a new
  pre-sym interface, a true source, a low-rank future row satisfying
  (22.14)--(22.19), and nonzero complete-old Schur value.

Atomic nodes:

- `P2R0G19B_MATH_H3_J_CYCLIC_PROJECTOR` -- **solved exact**:
  (22.1)--(22.6); `J_perp` is invisible to symmetric H3.
- `P2R0G19B_MATH_H3_J_OWNER_SPLIT` -- **solved no-double-count rule**:
  (22.7); cyclic AQ-minus-CP3 value is ordinary H3, oriented value is a
  different channel.
- `P2R0G19B_MATH_H3_CP3_THRESHOLD_JVP_CYCLIC_NOGO` -- **closed as an
  unsymmetrized provider**: the literal JVP (22.8) remains symmetric.
- `P2R0G19B_MATH_H3_CP3_DISTINGUISHED_SLOT` -- **conditional new interface**:
  (22.9)--(22.10); absent from D11 and useful only before common-slot
  symmetrization.
- `P2R0G19B_MATH_H3_J_D11_RESIDENCY` -- **closed current receipt**: modeled
  symmetric CP3 exists; oriented source, slot derivative, and reverse row do
  not.
- `P2R0G19B_MATH_H3_J_TENSOR_FREE_ACTION` -- **solved conditional identity**:
  (22.11)--(22.12), governed by `(q,K)` rather than one mask rank.
- `P2R0G19B_MATH_H3_J_R62_COST` -- **conditional screen / current
  residency-NO**: (22.13)--(22.19); generic dense slices fail the headroom.
- `P2R0G19B_MATH_H3_CP3_R62_TAIL` -- **open binding receipt**:
  (22.20)--(22.22); `J_perp` cannot improve it by (22.23).
- `P2R0G19B_MATH_H3_J_AS_H3_PROVIDER` -- **closed owner-correctly**:
  additive cyclic part is CP3/H3 replacement value, and the genuinely new
  unsymmetrized part has zero symmetric-H3 value.

The next lawful question is not whether the assembled CP3 threshold JVP can
be called `J`.  It cannot.  It is whether the proposed architecture is willing
to add a distinguished-slot Stein interface for at most about twenty fresh
atoms, and whether that non-H3 feature has enough joint Schur value to justify
its cost.  Independently, the symmetric r62/CP3 H3 replacement must still
clear (22.21) on its own.

### 22.8 Literal Gaussian-CP3 audit strengthens the current-cap NO

The literal D11 CP3 representation is not an exact Gaussian `K3` oracle.  In
its deployed normalization it carries

\[
\widehat T_3
=\sum_{a=1}^{r_{CP}}\lambda_a
\operatorname{Sym}(p_a\otimes p_a\otimes q_a),
\tag{22.24}
\]

and its dense/marginal reads expose only the repeated plane
`That3_iij` and diagonal `That3_iii`.  A generic ordered endpoint action
requires all six factor projections against its three external legs; those
projections are neither a fixed view of the repeated-plane buffer nor of the
diagonal buffer.

Already at the first zero-threshold reset, (22.24) is only quadratic in the
off-diagonal correlations.  The exact Gaussian ReLU `K3` contains the nonzero
`H2-H2-H2` triangle proportional to

\[
R_{ij}R_{ik}R_{jk}
\]

and an infinite higher-Hermite tail.  Hence, generically,

\[
\boxed{
K_3^{exact}=\widehat T_3^{D11}+\delta_3,
\qquad\delta_3\ne0.}
\tag{22.25}
\]

`Full birth` means that all births in the selected CP closure are retained;
it does not make that closure the exact synchronized Gaussian cubic tensor.
The owner-correct source and the q3 tail gate must therefore retain
`delta3`.  At arbitrary bias the exact Gaussian q211 identity contains the
slot-labelled first threshold derivatives of `K3`; it also contains the
corresponding derivative of `delta3`.  Differentiating only (22.24) cannot
remove that residual.

The literal tangent setup confirms the residency result of Section 22.3.
D11 can form the elementwise derivatives of the CP birth/gate coefficients,
but stores no `pdot`, `qdot`, or `lambdadot` bank, exports no slot-typed JVP,
and has no future-costate lifetime overlap with such a bank.  Thus even the
approximate threshold JVP is a new evaluator, not a resident view.

There is also a stronger target-rank obstruction than the abstract
`(q,K)` screen whenever the actual Gaussian threshold leaf is used.  For a
rank-one external q211 atom with dense factors `alpha,beta`, one leaf has
coefficient matrix

\[
\boxed{
C_{\alpha,\beta}=D_\alpha R D_\beta.}
\tag{22.26}
\]

Any exact representation by `k` common threshold directions has rank at most
`k`, so

\[
\boxed{
k\ge\operatorname{rank}(D_\alpha R D_\beta).}
\tag{22.27}
\]

For dense nonzero endpoint factors and a generic correlation matrix, the
right side is `n=1024`.  A scalar endpoint does not make this source
acquisition rank one.

Operationally each CP factor column needs four query-specific correlation
products for a generic three-leg atom, or two in the repeated
`beta=gamma` case.  On the literal ten-site L5--14 D11 schedule, the retained
full-birth ranks sum to `95n`, giving

\[
\boxed{
C_{leaf}=380\quad\text{generic},
\qquad190\quad(\beta=\gamma).}
\tag{22.28}

These are dense-GEMM equivalents before lower-order projections and hooks.
Any five-site suffix cut remains far outside the `1.56705` r62 headroom for
the literal full-birth ranks.  A low-rank correlation factorization can lower
the products only after its exact same-cut rank and coherent endpoint error
are supplied; generic dense `R` gives no saving.

At exactly zero threshold the slot JVP collapses by parity, but one broad
edge sandwich remains.  It has an exact one-product realization

\[
Z=R(D_\beta P_+D_\gamma),
\qquad
B=\langle D_\alpha Z,R\rangle_F.
\tag{22.29}
\]

No paid D11 product has both this left factor and this query-specific RHS at
the same seam.  Thus it costs one dense equivalent per active site; even five
sites cost `5>1.56705`.  Moreover the actual late suffix is generically biased,
so (22.29) does not replace the missing slot JVP there.

Equations (22.14)--(22.19) remain a valid conditional construction only when
a new low-`(q,K)` distinguished-slot factor interface is supplied directly.
They must not be applied to the literal Gaussian threshold leaf by treating
its final scalar as rank one: (22.27)--(22.28) are the binding current-lineage
rank/cost audit.

Additional atomic nodes:

- `P2R0G19B_MATH_H3_D11_CP3_EXACTNESS` -- **closed generic exact-owner
  claim**: (22.24)--(22.25), with `delta3` retained in the H3 source/tail.
- `P2R0G19B_MATH_H3_D11_SLOT_JVP_RESIDENCY` -- **closed literal reuse**:
  elementwise primitives exist, but no tangent bank, oriented output, or
  reverse lifetime overlap exists.
- `P2R0G19B_MATH_H3_CP3_THRESHOLD_TARGET_RANK` -- **solved generic no-go**:
  (22.26)--(22.27), generic minimum direction rank 1024.
- `P2R0G19B_MATH_H3_CP3_THRESHOLD_LITERAL_COST` -- **closed current cap**:
  (22.28); the low-rank factor exception requires a new exact same-cut
  receipt.
- `P2R0G19B_MATH_H3_CP3_ZERO_BIAS_SANDWICH` -- **solved one-product identity
  / closed five-site additive cost**: (22.29), and not applicable to generic
  biased resets.

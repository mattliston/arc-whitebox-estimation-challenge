# Full-AQ r49 plus Gaussian rest: resident-adjoint and Bellman-cross audit

This is a pure mathematical and source-read audit.  No numerical experiment is
used.  Its purpose is to connect the exact AQ tilted/residual generator to the
proposed `FULL_AQ_R49_PLUS_GAUSSIAN_REST` construction without conflating three
different objects:

1. a local source contraction identity;
2. a future endpoint costate which chooses the contraction;
3. a forward state capable of coadapting later resets.

## Verdict

The tilted generator is an exact **local one-residual query oracle** for the
full-AQ r49 state.  It does not supply the omitted two-residual covariance
source, and it does not construct the reverse Bellman query selected by future
layers.  Neither the proposed r49 forward state nor the literal D10/D11 lineage
contains a resident reverse operator or the concrete rank-capacity residual.

For one fixed scalar endpoint error certificate, the minimum causal missing
observable is exactly one Bellman cross scalar per residual injection,

\[
\boxed{
\chi_\ell
=\mathbb E\langle x_\ell^-,\varepsilon_\ell\rangle_{H_\ell}.}
\]

It is sufficient for an exact prefix energy recursion, but not for construction
of the complete forward covariance state.  Without it (or full block geometry),
the coherent bound is sharp.  At the stated r49 diagnostic capacity tail, this
shrinks the allowable aggregate rest energy from `0.032429` under exact
orthogonality to approximately `1.40e-3` under unknown coherent alignment.

## 1. Type the proposed source correctly

At one reset let

\[
g_i=m_i+u_i^{\mathsf T}z+Q_{M_i}(z),
\qquad
R_i^{AQ}=P_{\le2}(g_i)_+,
\qquad
r_i=(I-P_{\le2})(g_i)_+.
\]

The proposed covariance source is

\[
C_{ij}^{AQ+Grest}
=\langle R_i^{AQ},R_j^{AQ}\rangle+K_{ij}^{G,>2},
\]

and the genuinely missing reset plane is

\[
\boxed{
\Delta K_{ij}^{rest}
=\langle r_i,r_j\rangle-K_{ij}^{G,>2},}
\]

after complete incumbent H31/CP4/Q24 ownership subtraction.  Thus the missing
source is a **two-residual Gram**, not a one-residual cumulant.  It vanishes at
the affine base and has zero first variation in centered quadratic directions,
but it need not vanish at a finite, biased AQ reset.

The number

\[
e_{cap}=0.171590
\]

is a flat response-dimension diagnostic.  By itself it is not a concrete
endpoint vector, covariance block, or resident state.  A cross term is
well-typed only after freezing an actual endpoint metric and a projection or
joint-fit convention.  In that convention define

\[
c=r_{cap},
\qquad
\frac{\|c\|_H^2}{\|y\|_H^2}=e_{cap}.
\]

If no such concrete `c` is produced, only scalar coherent/elliptope bounds are
available; one cannot measure or assert its orientation from the dimension
ratio.

## 2. What the exact tilted generator contributes

For a residual source `r` and AQ consumers

\[
U=u^{\mathsf T}z+\frac12(z^{\mathsf T}Az-\operatorname{tr}A),
\quad
V=v^{\mathsf T}z+\frac12(z^{\mathsf T}Bz-\operatorname{tr}B),
\]

the exact local generator is

\[
H_r(\alpha,\beta)
=\mathbb E\!left[r\exp(\alpha U+\beta V)\right].
\]

For example, the one-residual K31 tangent is exactly

\[
\delta\kappa_{31}^{(1)}
=3\,\partial_\alpha^2\partial_\beta H_r(0,0)
+\partial_\alpha^3H_s(0,0).
\]

This scalar jet automatically sums the H3--H6 bands with their Wiener
factorials.  Consequently it is a lawful local reset oracle for every
decomposable AQ cubic query exposed by the r49 recurrence.

It does **not** directly acquire

\[
\langle r_i,r_j\rangle.
\]

That is a product of two distinct residual laws.  Generically it requires the
bivariate AQ positive-part oracle or a separate simultaneous/fused structure.
The one-residual generator can evaluate a residual against an AQ polynomial; it
cannot turn the other non-AQ residual into such a polynomial.  The six-outer-
product sign-flip envelope remains a valid certificate, but is not a correction
provider.

That envelope does give a clean handoff once a future plane costate has been
constructed.  With the one-unit quantities from the centered finite-
deformation bound, put

\[
B
=aa^{\mathsf T}+cb^{\mathsf T}+bc^{\mathsf T}
+ab^{\mathsf T}+ba^{\mathsf T}+bb^{\mathsf T},
\qquad |\Delta K^{rest}_{ij}|\le B_{ij}.
\]

For any frozen symmetric reset-plane costate `Lambda`, therefore,

\[
\boxed{
|\langle\Lambda,\Delta K^{rest}\rangle_F|
\le\langle |\Lambda|_{entry},B\rangle_F.}
\]

Every term on the right is a reduction such as
`a^T |Lambda|_entry b`.  This is a rigorous direct-contraction **bound** using
only univariate source quantities after the reverse query is known.  It neither
constructs `Delta K^rest` for the forward recurrence nor supplies `Lambda`, and
an endpoint squared-error claim must insert it through the applicable Bellman
dual/elliptope rather than treating a linear bound as an error energy.

There is a second, independent limitation.  Let `O_l` be the finite local
observation map whose coordinates are the tilted jets actually acquired at
layer `l`.  A future scalar query `lambda_l` against the source is determined
from those observations exactly iff

\[
\boxed{
\lambda_\ell\in\operatorname{ran}O_\ell^*
\quad\Longleftrightarrow\quad
\ker O_\ell\subseteq\ker\lambda_\ell.}
\]

This is the sharp information criterion.  Cheapness further requires a bounded
jet/polarization decomposition inside the billed census.  The generic dense
singleton K31 costate has full source-jet rank, so local exactness alone does
not establish that condition.

## 3. Literal resident-state audit

### Proposed full-AQ r49 state

The proposed forward state consists of `m_i,u_i,M_i` and their one-sided
transport bank of width

\[
r+d_r=49+\frac{49\cdot50}{2}=1274.
\]

It defines the forward AQ law and local reset potentials.  It does not include:

- the concrete endpoint capacity residual `c`;
- a reverse costate `lambda_l` or Bellman matrix `H_l`;
- the missing two-residual plane `Delta K_l^rest`;
- the accumulated true-minus-approximate error `x_l^-`.

Therefore the proposed state is not, as written, a resident Bellman-cross
provider.

### Literal D10/D11 lineage

The on-disk D10 and D11 estimator docstrings explicitly enumerate the live
state after a hidden layer: mean/covariance, CP3, a cubic coordinate-star
residual, CP4, a quartic coordinate-star, and one covariance-pairing
coefficient.  Their `_predict` functions perform a single forward layer loop.
The D11 correction carries forward learned `states` and scalar `energies`; its
estimator is explicitly source-only and says that no target-dependent state
exists.

No listed buffer is an endpoint covector, reverse covariance matrix, omitted
rest source, or capacity residual.  The affine Gaussian/Mehler computation
supplies the forward control variate `K^{G,>2}`.  It does not supply its error
against the true AQ rest.

Between resets, covariance transport

\[
C\longmapsto W^{\mathsf T}CW
\]

has Frobenius adjoint

\[
\Lambda\longmapsto W\Lambda W^{\mathsf T}.
\]

Thus the weights make a reverse sweep algebraically possible.  This does not
make `Lambda` resident: a separate matrix costate, reverse applications, reset
Jacobians, and either a stored forward tape or recomputation are still needed.
At a nonlinear AQ reset the adjoint must also pass through the reset potential
and ownership projection.  None of these reverse objects is present in the
declared roughly-40-equivalent r49 census or in the literal D10/D11 state.

Nor can a cheap rank-one reverse state be presumed to survive the incumbent
Gaussian reset.  Holding the marginal variances fixed, Price's theorem gives
for the off-diagonal Gaussian ReLU covariance map

\[
\frac{\partial K_{ij}^{G,+}}{\partial C_{ij}}
=\Pr(X_i>0,X_j>0)=:\Gamma_{ij}.
\]

Hence the corresponding part of the reset adjoint is the Schur multiplier

\[
\boxed{\Lambda^- = \Gamma\odot\Lambda^+.}
\]

If the outgoing scalar query is rank one, `Lambda^+=vv^T`, then

\[
\Lambda^-=\operatorname{Diag}(v)\,\Gamma\,\operatorname{Diag}(v).
\]

For nonzero `v` and a generic correlation frame, `Gamma` has full rank, so the
reverse costate becomes full rank in one reset.  Mean/variance derivative terms
add further components and do not restore a generic rank bound.  A bounded-rank
reverse census therefore needs a separate structural theorem; scalar endpoint
rank alone is not such a theorem.

At zero bias this generic-rank statement is visible directly from

\[
\Gamma_{ij}=\frac14+\frac1{2\pi}\arcsin\rho_{ij}.
\]

The positive power series of `arcsin` contains odd tensor features of
unbounded degree.  For any fixed finite number of generic, distinct latent
directions, their combined feature Gram is full rank once enough degrees are
included, even when the underlying linear covariance has small rank.

A literal dense covariance reverse chain uses two `n x n` matrix products per
transition, or 28 square-product equivalents over fourteen transitions, before
reset-adjoint work.  That operation may be performed off-card for a certificate
without affecting deployment billing.  If it is claimed as a deployed resident
provider, however, it is not contained in the roughly-40-equivalent forward
census.  Forward insertion of a known correction into an already-paid
covariance conjugation is a different operation from constructing the reverse
query which says what correction is needed.

An important no-free-lunch statement follows.  Reverse AD of the approximate
recurrence can construct the sensitivity to a **hypothetical**
`Delta K_l^rest`; it cannot reconstruct that unobserved plane.  The source
contraction and reverse-query construction are two separate acquisition gates.

## 4. Exact minimum causal Bellman observable

Freeze the endpoint metric `H` and the linearized transport from a reset site
to the endpoint.  In the Bellman recurrence write

\[
x_{\ell+1}=A_\ell x_\ell+\varepsilon_{\ell+1},
\qquad
H_\ell=P_{\ell\to L}^{\mathsf T}HP_{\ell\to L}.
\]

If the concrete capacity error has a state-space representative, fold it into
the initial accumulated error (otherwise append it as a separate identity-
transport endpoint block), so

\[
\frac{\mathbb E\|x_0\|_{H_0}^2}{\|y\|_H^2}=e_{cap}.
\]

Immediately before a new omitted-rest injection, let `x_l^-` denote all
previous capacity/reset error expressed at the same local site.  Define

\[
d_\ell
=\frac{\mathbb E\|\varepsilon_\ell\|_{H_\ell}^2}{\|y\|_H^2},
\qquad
\chi_\ell
=\frac{\mathbb E\langle x_\ell^-,\varepsilon_\ell\rangle_{H_\ell}}
{\|y\|_H^2}.
\]

Then the normalized endpoint-equivalent prefix energy obeys the exact identity

\[
\boxed{
E_\ell^+=E_\ell^-+d_\ell+2\chi_\ell.}
\]

Propagation without an injection preserves this value because
`H_l=A_l^*H_{l+1}A_l`.  Hence one `chi_l` per injection is sufficient to update
the exact scalar endpoint certificate, including both capacity--rest and
past-rest--new-rest coherence.

For a nonlinear finite recurrence, the endpoint formulation below remains
exact after hybrid telescoping one reset at a time.  A local `H_l` expression
then uses the corresponding path-integrated secant pullback; it must not be
silently identified with the tangent at only the approximate trajectory.

It is also minimal.  Given only `E_l^-` and `d_l`, positive semidefiniteness
allows

\[
-\sqrt{E_\ell^-d_\ell}
\le\chi_\ell\le
+\sqrt{E_\ell^-d_\ell}.
\]

Every value in this interval is realized already by a two-vector Gram matrix,
and produces a different next energy.  Unless a structural orthogonality law is
proved, one additional real scalar is therefore information-theoretically
necessary for exact causal scalar recursion.

Equivalently, with endpoint contributions

\[
z_\ell=P_{\ell\to L}\varepsilon_\ell,
\qquad
Z_{\ell-1}=c+\sum_{k<\ell}z_k,
\]

the observable is

\[
\chi_\ell
=\frac{\mathbb E\langle Z_{\ell-1},z_\ell\rangle_H}{\|y\|_H^2}.
\]

All `chi_l=0` make RSS exact at every prefix.  For only the final endpoint, one
noncausal total cross scalar `sum_l chi_l` is sufficient.  Adjacent-layer cross
terms are not sufficient because the new residual can correlate with any
earlier residual.

For the campaign's declared L5--14 residual window this means ten causal
scalars.  Any initialization, early-reset, quadrature, or capacity error not
already folded into the initial block is an additional injection and needs its
own cross acquisition (or inclusion in a jointly acquired block).

The associated reverse covector is

\[
\lambda_\ell
=P_{\ell\to L}^*H Z_{\ell-1},
\qquad
\chi_\ell
=\frac{\mathbb E\langle\lambda_\ell,\varepsilon_\ell\rangle}
{\|y\|_H^2}.
\]

A reverse operator is not the only **off-card certification** interface.  If
exact hybrid endpoint contributions can be constructed, polarization gives

\[
\boxed{
2\chi_\ell
=\frac{
\|Z_{\ell-1}+z_\ell\|_H^2
-\|Z_{\ell-1}\|_H^2
-\|z_\ell\|_H^2}{\|y\|_H^2}.}
\]

This acquires precisely the same one scalar without materializing `H_l` or
`lambda_l`.  It requires truthful hybrid endpoint paths (or their exact
analytic equivalents), so it is a diagnostic/certification route, not a
target-free deployed source and not a substitute for forward covariance
closure.

The tilted generator helps with the second pairing only when this particular
`lambda_l` lies in its acquired local query span.  It does not produce
`lambda_l`; that requires future reverse sewing.  Conversely, a reverse sweep
does not produce the omitted source `epsilon_l`.  This is the required strict
separation between local acquisition and future costate closure.

One scalar per injection certifies one fixed scalar endpoint value.  It does
not construct the matrix cross block needed for multiple future heads or for a
forward recurrence whose future gates coadapt to the corrected covariance.  A
deployed correction still needs the complete cross state, or proof that every
unobserved direction is future-dark.

## 5. Exact r49 pass and close gates

First aggregate all reset/Gaussian-rest approximation error at the endpoint:

\[
\delta=\sum_\ell z_\ell,
\qquad
e_{rest}=\frac{\|\delta\|_H^2}{\|y\|_H^2},
\qquad
\chi=\frac{\langle c,\delta\rangle_H}{\|y\|_H^2}.
\]

The total normalized error is exactly

\[
\boxed{e_{tot}=0.171590+e_{rest}+2\chi.}
\]

For the `6e-9` target,

\[
e_*=0.204019,
\qquad
e_*-e_{cap}=0.032429.
\]

Thus the exact pass/close boundary is

\[
\boxed{
\chi\le\frac{0.032429-e_{rest}}2
\quad\text{(pass)},}
\]

with strict reversal closing that frozen decomposition.

### Structural orthogonality

If a genuine endpoint/Bellman theorem gives `chi=0`, then

\[
\boxed{e_{rest}\le0.032429.}
\]

Same-site AQ projection orthogonality does not prove this.  Earlier omitted
high-chaos residuals can re-enter the future AQ observation, and the future
metric is not the local Euclidean metric.

### Known scalar coherence

Let

\[
\rho=\frac{\chi}{\sqrt{e_{cap}e_{rest}}}.
\]

For a certified upper coherence `rho_bar`, the sharp robust scalar gate is

\[
0.171590+e_{rest}
+2\bar\rho\sqrt{0.171590\,e_{rest}}
\le0.204019.
\]

Equivalently,

\[
\boxed{
e_{rest}\le
\left[
-\bar\rho\sqrt{0.171590}
+\sqrt{0.204019-(1-\bar\rho^2)0.171590}
\right]^2.}
\]

At `rho_bar=0` this is `0.032429`.  With completely unknown orientation,
`rho_bar=1`, the coherent bound is sharp and becomes

\[
\boxed{
e_{rest}
\le(\sqrt{0.204019}-\sqrt{0.171590})^2
\simeq1.40\times10^{-3}.}
\]

There is a conditional probabilistic interpretation of the original flat
dimension diagnostic, but it is not a certificate.  If the concrete capacity
normal is additionally modeled as uniform in its `D=269976`-dimensional normal
space and independent of the rest endpoint direction, then

\[
\mathbb E\rho=0,
\qquad
\mathbb E\rho^2=\frac1D,
\qquad
\Pr(|\rho|\ge t)
\le2\exp\!\left(-\frac{(D-2)t^2}{2}\right)
\]

up to the standard spherical-cap constant.  The typical coherence scale is
therefore about `D^{-1/2} ~= 1.9e-3` under that model.  This explains why the
`0.032429` orthogonal budget is the internally consistent *average* reading of
the flat-capacity heuristic.  It cannot be promoted to a frozen gate: the
capacity miss and non-Gaussian rest are generated by the same network/core, so
their independence is precisely what has not been proved.

For a lower coherence certificate `rho_under`, the corresponding robust close
condition is

\[
0.171590+e_{rest}
+2\underline\rho\sqrt{0.171590\,e_{rest}}
>0.204019.
\]

With no orientation information at all, a two-block scalar model can close for
every alignment only if

\[
(\sqrt{e_{rest}}-\sqrt{0.171590})^2>0.204019;
\]

for the small-rest regime this is intentionally weak.  The useful no-data gate
is therefore the robust-pass threshold, not a robust close.

### Sharp elliptope gates

If capacity and rest are random endpoint blocks with endpoint covariances
`C_cap,C_rest`, their two-block second-moment extrema are

\[
\boxed{
\mathcal U_2
=\operatorname{tr}C_{cap}+\operatorname{tr}C_{rest}
+2F(C_{cap},C_{rest}),}
\]

\[
\boxed{
\mathcal L_2
=\operatorname{tr}C_{cap}+\operatorname{tr}C_{rest}
-2F(C_{cap},C_{rest}),}
\]

where

\[
F(C,D)=\operatorname{tr}(C^{1/2}DC^{1/2})^{1/2}.
\]

After division by the target energy, `U_2<=0.204019` is a robust pass and
`L_2>0.204019` is a robust close.  Scalar coherence replaces `F` by its sharp
trace-only upper bound `sqrt(tr C tr D)`.

For capacity plus all layer residuals, let `G` be their joint block Gram, fix
the local diagonal blocks `E_l`, and let

\[
K_{lk}=P_l^*HP_k.
\]

Then

\[
\mathcal U_*
=\max_{G\succeq0,\ G_{ll}=E_l}\langle K,G\rangle,
\qquad
\mathcal L_*
=\min_{G\succeq0,\ G_{ll}=E_l}\langle K,G\rangle.
\]

The normalized conditions

\[
\boxed{\mathcal U_*/\|y\|_H^2\le0.204019}
\]

and

\[
\boxed{\mathcal L_*/\|y\|_H^2>0.204019}
\]

are respectively sharp second-moment pass and close certificates.  If all
matrix geometry is collapsed to endpoint scalar energies `d_l`, the upper
program reduces exactly to

\[
\boxed{
\left(\sqrt{0.171590}+\sum_\ell\sqrt{d_\ell}\right)^2,}
\]

so the triangle/coherent gate cannot be improved without cross or shape
information.

The trace-only lower elliptope also has a closed form.  Put

\[
a_0=\sqrt{0.171590},
\qquad a_\ell=\sqrt{d_\ell},
\qquad a_{max}=\max_j a_j.
\]

Then the sharp range allowed by scalar energies alone is

\[
\boxed{
\left[\max\left(2a_{max}-\sum_j a_j,0\right)\right]^2
\le e_{tot}\le
\left(\sum_j a_j\right)^2.}
\]

The lower value is the polygon-cancellation law.  Consequently its strict
exceedance of `0.204019` is the sharp trace-only robust close gate, just as the
upper value's nonexceedance is the sharp trace-only robust pass gate.

## 6. Promotion routes and exact dependencies

The architecture can clear this audit by any one of the following logically
complete routes:

1. **Exact causal receipt.**  Define a concrete capacity projection, build the
   reverse Bellman covector, acquire the omitted-rest contraction, and record
   one `chi_l` per injection.  This certifies a fixed endpoint scalar but does
   not by itself repair forward state closure.
2. **Block elliptope receipt.**  Acquire lawful local residual blocks (or
   rigorous envelopes), transport them in the fixed future metric, and pass the
   upper SDP.  Close the branch if the lower SDP exceeds the target.
3. **Future-dark theorem.**  Prove that every unobserved local residual lies in
   the kernel of the complete future pullback.  Local AQ orthogonality is not
   this theorem.
4. **Joint end-to-end refit.**  Fit the r49 and rest providers together and
   grade the resulting frozen endpoint residual directly.  In this route the
   heuristic split into `e_cap,e_rest,chi` is no longer independently binding;
   the actual joint residual is.  It still requires lawful deployed source
   construction and a complete cost census.

Projection orthogonality cannot be borrowed across these routes.  If `c` is
defined as the residual after projecting only onto an AQ-r49 endpoint space,
an omitted-rest column need not be orthogonal to it.  If that column is added
to the fitted space and the target is reprojected, the new final residual is
orthogonal, but the old `0.171590 + e_rest + 2chi` decomposition and its flat
capacity interpretation must be recomputed rather than setting `chi=0` by
declaration.

The present evidence supplies none of the first three complete receipts.  The
r49+Gaussian-rest architecture remains mathematically plausible because of the
centered parity cancellation and its `0.032429` orthogonal budget, but it is not
certified by the local tilted generator or by forward-state residency.

## Graph-ready nodes

### R49_GREST_LOCAL_GENERATOR_BOUNDARY

- status: solved exact interface / global acquisition open;
- result: tilted AQ jets acquire local one-residual q3/q4 actions, but
  `Delta K^rest` is a two-residual Gram and needs bivariate/fused acquisition;
- exact information gate: `lambda_l in ran O_l^*`;
- warning: local source acquisition and future costate construction are
  independent.

### R49_GREST_RESIDENT_REVERSE_NOGO

- status: closed for the declared state;
- evidence: r49 carries only forward AQ coefficients; literal D10/D11 carries
  forward mean/covariance and CP/star states, with no endpoint costate,
  capacity residual, omitted-rest plane, or accumulated error;
- consequence: covariance self-adjointness permits a new reverse sweep but is
  not a resident-operator receipt.

### R49_GREST_MINIMAL_CAUSAL_CHI

- status: solved information theorem / acquisition open;
- observable:
  `chi_l=E<x_l^-,epsilon_l>_{H_l}/||y||_H^2`;
- identity: `E_l^+=E_l^-+d_l+2chi_l`;
- minimality: diagonal energies leave the full Cauchy interval feasible;
- scope: one fixed endpoint scalar, not complete forward covariance closure.

### R49_GREST_COHERENCE_GATE

- status: solved sharp scalar gate;
- exact: `chi <= (0.032429-e_rest)/2`;
- orthogonal allowance: `e_rest<=0.032429`;
- unknown-coherence allowance: `e_rest<=1.40e-3` approximately;
- robust geometry: block-elliptope upper value passes at `<=0.204019`, lower
  value closes at `>0.204019`.

---

# D14 prefix through L4 plus full-AQ r55 Gaussian-rest suffix

This continuation audits the proposed hybrid:

1. retain the incumbent two-sided D14 state through the cut after L4;
2. initialize one full affine--quadratic rank-55 state at that cut;
3. use the one-sided AQ recurrence over L5--14;
4. retain an incumbent D10 affine Gaussian/Mehler rest plane over the same ten
   late transitions.

The quoted arithmetic is about `46.1` square-product equivalents at 32 reset
quadrature nodes.  The route is materially better posed than raw-Gaussian r49
initialization, but only after distinguishing a weak low-chaos section from a
strong future-equivalent seam.

## 7. Exact weak seam map

Use the half-quadratic convention

\[
s_i=(u_i,A_i)\in
V_r:=\mathbb R^r\oplus\operatorname{Sym}_r,
\qquad
X_i=m_i+u_i^{\mathsf T}z
+\frac12(z^{\mathsf T}A_i z-\operatorname{tr}A_i).
\]

The covariance inner product is

\[
B_2(s_i,s_j)
=u_i^{\mathsf T}u_j+\frac12\operatorname{tr}(A_iA_j),
\]

and the cubic polarization is

\[
\begin{aligned}
B_3(s_1,s_2,s_3)
={}&\frac12\operatorname{tr}
(A_1A_2A_3+A_1A_3A_2)\\
&+u_1^{\mathsf T}A_3u_2
+u_1^{\mathsf T}A_2u_3
+u_2^{\mathsf T}A_1u_3.
\end{aligned}
\]

Thus the exact directed repeated cubic plane of an AQ state is

\[
\boxed{
K^{21}_{ij}
=B_3(s_i,s_i,s_j)
=\operatorname{tr}(A_i^2A_j)
+u_i^{\mathsf T}A_ju_i
+2u_i^{\mathsf T}A_iu_j.}
\]

Let `C^D,K21^D,m^D` be the centered D14 cut observables.  The **weak exact
section problem** is precisely

\[
\boxed{
m_i=m_i^D,
\qquad
B_2(s_i,s_j)=C^D_{ij},
\qquad
B_3(s_i,s_i,s_j)=K^{21,D}_{ij}
\quad\forall i,j.}
\]

This is the exact identity needed to avoid an artificial jump in the shared
mean/covariance/K21 interface.  It is not yet equality of the complete D14
state.

Define the quotient map

\[
\Psi_r:(V_r^n/O(r))\longrightarrow
\operatorname{Sym}_n\oplus\mathbb R^{n\times n},
\qquad
\Psi_r(s)=(C,K^{21}).
\]

The weak seam exists iff

\[
\boxed{(C^D,K^{21,D})\in\operatorname{im}\Psi_{55}.}
\]

This image-membership statement, not a parameter count, is the necessary and
sufficient global condition.

### Dimension and local-section theorem

At `n=1024,r=55`,

\[
\dim V_{55}=55+\frac{55\cdot56}{2}=1595,
\]

so, modulo the common latent rotation,

\[
\boxed{
p_{55}^{AQ}
=1024(1595)-\frac{55\cdot54}{2}
=1,631,795.}
\]

The shared interface dimension is

\[
\boxed{
q=\frac{1024\cdot1025}{2}+1024^2
=1,573,376,}
\]

leaving a regular fiber dimension

\[
\boxed{p_{55}^{AQ}-q=58,419.}
\]

Even full-AQ r54 exceeds `q` by 1,129, so r55 has no positive flat
dimension-tail at this interface if its Jacobian is actually onto.  The large
surplus is freedom for conditioning and future-visible fiber selection, not a
proof of global surjectivity.

At a solution `s^0`, the exact local stable-section condition is

\[
\boxed{
D\Psi_{55}(s^0)
\text{ is onto after quotienting the }O(55)\text{ gauge}.}
\]

Equivalently, the restricted smallest singular value

\[
\sigma_{seam}
=\sigma_{min}
\left(D\Psi_{55}|_{\mathrm{gauge}^{\perp}}\right)
\]

must be positive.  The implicit-function theorem then gives a local smooth
section, with first-order sensitivity at least proportional to
`1/sigma_seam`.  A deployable corpus-wide section additionally needs one
target-free branch choice which avoids the discriminant and remains stable
under neuron permutations and latent gauge changes.  Solving a different
nonlinear inverse problem with truth labels on every network is not a lawful
section.

## 8. Exact moment-image necessary conditions

The D14 cut is not automatically in the AQ moment image.  Several exact,
source-only fail-fast conditions follow from the AQ algebra.

First,

\[
C^D\succeq0.
\]

For a fixed anchor `i`, define the AQ Riesz vector

\[
T_i
=\left(2A_iu_i,\ 2(A_i^2+u_iu_i^{\mathsf T})\right)\in V_r.
\]

It obeys

\[
K^{21}_{ij}=B_2(T_i,s_j).
\]

Therefore every AQ seam satisfies the exact row-space condition

\[
\boxed{
\ker C^D\subseteq\ker K^{21,D},}
\]

and the leverage inequality

\[
\boxed{
K^{21,D}_{i,:}(C^D)^+
K^{21,D\,\mathsf T}_{i,:}
\le8(C^D_{ii})^2
\qquad\forall i.}
\]

Indeed, the left side is `||P_span{s_j} T_i||^2`.  Direct calculation gives

\[
\|T_i\|_{B_2}^2
=2\operatorname{tr}A_i^4
+8u_i^{\mathsf T}A_i^2u_i
+2\|u_i\|^4
\le8\left(
\|u_i\|^2+\frac12\operatorname{tr}A_i^2
\right)^2.
\]

These conditions are necessary, not sufficient; they are useful because they
involve only the materialized D14 `C,K21` planes.

There is also a sharp scalar deformation constraint.  For any one AQ direction
put

\[
v=\|u\|^2+\frac12\operatorname{tr}A^2,
\qquad
x=\frac12\operatorname{tr}A^2,
\qquad
t=x/v\in[0,1].
\]

Its third cumulant obeys

\[
\boxed{
\frac{|\kappa_3(X)|}{v^{3/2}}
\le\sqrt{2t}\,(3-t)
\le2\sqrt2.}
\]

The first bound is sharp for a rank-one `A` with `u` in the same eigendirection.
Thus an observed standardized D14 skewness gives a nonzero lower bound on the
quadratic energy fraction `t`; a strongly skewed cut cannot simultaneously be
called a small deformation of the affine Gaussian base.

Applied to every two-unit linear combination, this gives a necessary condition
entirely in the weak seam observables.  For all `i,j` and all real `alpha,beta`,

\[
\boxed{
\begin{aligned}
&|\alpha^3K_{ii}^{21}
+3\alpha^2\beta K_{ij}^{21}
+3\alpha\beta^2K_{ji}^{21}
+\beta^3K_{jj}^{21}|\\
&\qquad\le2\sqrt2\,
(\alpha^2C_{ii}+2\alpha\beta C_{ij}+\beta^2C_{jj})^{3/2}.
\end{aligned}}
\]

Violation for one pair and one direction proves that no AQ section of any rank
matches the D14 `C,K21` cut.

If a **strong** seam also matches a fourth cumulant direction, every AQ scalar
direction necessarily satisfies

\[
\boxed{
0\le\kappa_4(X)
=3\operatorname{tr}A^4+12u^{\mathsf T}A^2u
\le12v^2.}
\]

A D14 future-visible quartic direction outside this cone cannot be reproduced
by any rank, not merely by r55.

## 9. Weak section versus strong future-equivalent seam

Matching `m,C,K21` does not match the D14 CP3/cubic state or its
K4/K31/K22/CP4 readouts.  Dense future weights can read all-distinct cubic and
quartic components which are not determined by the repeated K21 plane.  D10's
Gaussian rest repairs a covariance sector; it does not identify these signed
cumulant discrepancies.

Let `T_3^D,T_4^D` denote the complete D14 cut cubic and quartic states in the
typed old quotient, and let

\[
T_3^{AQ}(s)=B_3(s,s,s),
\qquad
T_4^{AQ}(s)=\operatorname{polarize}\Phi_4(s),
\]

with

\[
\Phi_4(u,A)=3\operatorname{tr}A^4+12u^{\mathsf T}A^2u.
\]

If `B_3^+,B_4^+` are the complete strict-future backward query spaces generated
by L5--14 and every reported endpoint head, the exact strong seam condition is

\[
\boxed{
\langle\Lambda_3,T_3^D-T_3^{AQ}(s)\rangle=0
\quad\forall\Lambda_3\in B_3^+,}
\]

\[
\boxed{
\langle\Lambda_4,T_4^D-T_4^{AQ}(s)\rangle=0
\quad\forall\Lambda_4\in B_4^+.}
\]

Equivalently, the discarded seam state must lie in the complete future-dark
kernel.  Full tensor equality is sufficient; annihilation of the actual future
query quotient is necessary and sufficient for the frozen future response.

The 58,419-dimensional regular fiber may be optimized to satisfy some of these
conditions.  Let `Theta` collect the required future-visible cubic/quartic
functionals.  Local exact fiber selection requires

\[
T_\Theta^D\in\Theta(\Psi_{55}^{-1}(C^D,K^{21,D})),
\]

and stable local selection requires the fiber derivative

\[
\boxed{
D\Theta|_{\ker D\Psi_{55}}
\text{ onto the required quotient with a positive restricted singular value}.}
\]

Without this receipt, `exact r55 initialization` means only an exact weak
low-chaos section.  The unmatched cubic/quartic seam is a new residual
injection and must enter the Bellman/elliptope gate.

## 10. What the prefix does and does not solve

The prefix does solve the specific r49 raw-Gaussian obstruction.  It never asks
for a 1024-dimensional cubic-null subspace of `V_49`: D14 owns the raw Gaussian
and early ReLU evolution, and the L4 target is allowed to have nonzero K3.
Moreover `dim V_55=1595>1024`, so covariance rank alone is not restrictive.

It does **not** prove the weak AQ moment-image condition, a stable section, or
the strong future quotient identities.  It replaces the cubic-null problem by
a cut-state inverse problem and a large fiber-selection problem.

Nor does exact matching to D14 make the cut exact relative to truth.  Let
`c_D` be the propagated true-minus-D14 prefix error.  A perfect section copies
the D14 state, so it inherits `c_D`; it does not erase it.

The earlier draft incorrectly inferred a frozen-row MSE floor.  The official
score is **final-layer raw MSE only**; all-layer MSE is a diagnostic.  Hence no
positive lower bound follows from the number or diagnostic error of prefix
rows.

The correct object is the cut error transported to the final layer.  In a
frozen linearized frame, let `P_cut` be the cut-to-final propagator, let `c_D`
be true-minus-D14 cut error, and let `z_suf` collect seam and suffix
innovations.  Then

\[
 \boxed{e_{\rm final}=\|P_{\rm cut}c_D+z_{\rm suf}\|_H^2.}
\]

There is no nonzero prefix-only floor without a restricted lower singular
value for `P_cut` **and** an angle or anti-cancellation certificate against
`z_suf`.  At most, if such independent bounds are proved,

\[
 e_{\rm final}\ge
 \left(\sigma_{\min}^{\rm res}(P_{\rm cut})\|c_D\|
       -\|z_{\rm suf}\|_H\right)_+^2.
\]

This can be zero.  The exact Bellman cross, not prefix row counting, is the
binding endpoint gate.

## 11. Gaussian-rest error at an r55 cut

For the selected r55 state, define as before

\[
\Delta K_{ij}^{rest}
=\langle r_i^{AQ},r_j^{AQ}\rangle-K_{ij}^{G,>2}.
\]

At an affine state `A_i=0` the D10 Gaussian rest is exact.  At zero bias its
first variation in `A` vanishes, so the missing source starts at quadratic
order.  Neither premise is automatic at L4.  In fact, matching a nonzero D14
K3 requires nonzero quadratic content, and the sharp skewness inequality above
can force a material minimum `t`.  Nonzero standardized bias also restores the
first-order threshold-band leakage.

Consequently the prefix does not remove the Gaussian-rest source gate; it
merely restricts it to ten late resets.  The first L5 source can already be a
finite-deformation error.  Its rigorous source-only routes remain:

- exact bivariate AQ residual acquisition;
- the six-outer-product sign-flip envelope;
- a future-dark theorem;
- or direct endpoint/elliptope certification.

## 12. Capacity and Bellman/coherence gates

At a regular exact weak r55 section the flat `(C,K21)` dimension tail is zero,
not `0.171590`.  Therefore the r49 `0.032429` leftover is not the correct r55
budget.  But zero dimension tail must not be confused with zero inherited,
seam-fiber, Gaussian-rest, or quadrature error.

Aggregate all fixed pre-suffix error into an endpoint contribution `c_0`:

\[
e_0=\frac{\|c_0\|_H^2}{\|y\|_H^2}.
\]

It includes the propagated D14 prefix error, any weak/strong seam mismatch, and
any frozen direct-output floor expressed in the common score normalization.
Let `delta` be the aggregate L5--14 Gaussian-rest/reset/quadrature error,

\[
e_{suf}=\frac{\|\delta\|_H^2}{\|y\|_H^2},
\qquad
\chi=\frac{\langle c_0,\delta\rangle_H}{\|y\|_H^2}.
\]

For the `6e-9` normalized target `e_*=0.204019`, the exact gate is

\[
\boxed{
e_0+e_{suf}+2\chi\le0.204019.}
\]

Hence:

\[
\boxed{
e_{suf}\le0.204019-e_0
\quad\text{under genuine Bellman orthogonality},}
\]

whereas with unknown positive coherence the sharp robust scalar gate is

\[
\boxed{
e_{suf}
\le\left(\sqrt{0.204019}-\sqrt{e_0}\right)^2.}
\]

If `e_0>=0.204019`, the frozen route is closed before any suffix error.  In the
idealized relative-to-D14 analysis with `e_0=0`, the entire `0.204019` is
available to the aggregate suffix error; this is a relative interface statement
and omits the transported cut-state error.

For layer contributions `z_l` over L5--14 with normalized diagonal energies
`d_l`, the exact causal recursion remains

\[
E_l^+=E_l^-+d_l+2\chi_l,
\qquad
\chi_l
=\frac{\mathbb E\langle Z_{l-1},z_l\rangle_H}{\|y\|_H^2},
\]

starting from `E_4=e_0`.  Ten `chi_l` values give the exact scalar prefix
certificate.  If only scalar energies are known, the sharp coherent upper gate
is

\[
\boxed{
\left(\sqrt{e_0}+\sum_{l=5}^{14}\sqrt{d_l}\right)^2
\le0.204019.}
\]

The corresponding scalar-elliptope lower value is the polygon law from the r49
audit.  Full block residual Grams give the same sharp upper/lower SDPs, now with
the D14/seam error as block zero.  Local AQ projection and centered parity do
not imply `chi_l=0` in the future metric.

## 13. Resident reverse verdict for the suffix

The suffix does not solve the resident-reverse objection.  Neither the D14 cut
state nor `(m,u,A)` plus the D10 rest plane contains the concrete truth error,
the accumulated rest error, or the pulled-back endpoint costate.  The Gaussian
reset adjoint still contains the full-rank Schur multiplier

\[
\Lambda^- =\Gamma\odot\Lambda^+.
\]

Cutting at L4 only shortens a dense covariance reverse chain from fourteen to
ten transitions.  A literal dense suffix reverse still costs about 20 square
products before reset adjoints.  It may be used off-card for certification, or
replaced there by exact hybrid endpoint polarization, but it is not resident in
the quoted `46.1` forward census.  Adding it on-card would exceed the hard cap.

Forward insertion of an already-known correction into the paid D10 covariance
conjugation can remain cheap.  That does not acquire the missing two-residual
source or construct the reverse query.

## 14. Exact arithmetic behind the `46.1 at K32` claim

With the cut convention of four paid D14 transitions and ten suffix
transitions, the declared census is

\[
\boxed{
\begin{aligned}
C_{D14\to AQ55+G}(K)
={}&2(4)+2(10)
+10\frac{1595}{1024}\\
&+10\frac{55^3}{1024^2}
+10K\frac{55^2}{1024^2}.
\end{aligned}}
\]

The terms are, respectively:

1. four two-sided incumbent D14 transports;
2. ten two-sided D10 Gaussian-rest transports;
3. ten one-sided full-AQ coefficient-bank transports;
4. ten local rank-55 eigensolves;
5. ten `K`-node rank-55 reset quadratures.

At `K=32`,

\[
\boxed{C_{D14\to AQ55+G}(32)\simeq46.086,}
\]

leaving approximately `2.914` square-product equivalents.

This is a conditional subtotal, not yet a complete deploy receipt.  Its exact
premises are:

1. **Boundary convention.**  The cut after L4 must really contain four paid
   D14 conjugations.  One additional two-sided boundary push raises the subtotal
   by 2 to about `48.086` before all other omissions.
2. **Literal prefix scope.**  The `8` term prices the abstract D14 two-sided
   word.  If retaining the literal on-disk D14 prefix also requires CP3/CP4,
   Edgeworth, source/reset, or arena operations not already shared and billed,
   they must be added.
3. **Seam construction.**  Materializing the D14 `C,K21`, solving the global
   nonlinear r55 section, choosing its 58,419-dimensional fiber, and stabilizing
   its gauge are absent.  A target-free closed/fused section is required; an
   off-card per-network inverse solve cannot be borrowed for deployment.
4. **Low-chaos covariance read.**  The AQ bank push `W^T S` and the all-pairs
   Gram read `SS^T` are distinct generic rectangular products.  The latter is
   needed whenever the Gaussian-rest source requires the matched full
   correlation plane.  The subtotal charges only one `1595/1024` operation per
   layer.  Avoiding a second such charge requires a literal identical resident
   covariance plane or source/transport fusion.  Charging it generically adds
   `10(1595/1024)=15.576`, closing the budget.
5. **Rest source.**  The `20` term transports a known plane; it does not form
   the non-Gaussian `Delta K^rest`, perform complete-old residualization, or
   supply a bivariate AQ correction.
6. **Cumulant reads.**  Complete K3/q211/q4 local actions and any seam
   future-quotient read must be shown to be scalar-potential reductions already
   covered by the eigensolve/quadrature subtotal.  A generic n-by-n or tensor
   read is not free.
7. **Quadrature accuracy.**  `K=32` is a cost choice, not an error theorem.  Its
   reset error joins the Bellman innovations unless a uniform analytic bound is
   supplied.
8. **Reverse/cross acquisition.**  Off-card `chi_l` certification is unbilled;
   an on-card dense reverse is not.  No resident-reverse credit appears in the
   subtotal.
9. **Precision and stabilization.**  PSD repair, eigengap handling, branch
   selection, memory traffic, and dtype conversions must match the billed
   implementation rather than the symbolic operation count.

The all-pairs Gram-read caveat is especially binding: if the D10 plane is only
the high-chaos rest, it does not by itself materialize the AQ low-chaos Gram.
If it is instead declared to be the total covariance, the exact reset identity
which combines the factor Gram and Gaussian rest into that plane must be shown
without silently paying `SS^T`.

## 15. Suffix-route verdict

The D14-prefix/r55-suffix route is a legitimate mathematical improvement over
raw-initialized r49:

- it removes the cubic-null Gaussian seed obstruction;
- r55 has generic dimension surplus at the shared `(C,K21)` interface;
- it limits new Gaussian-rest innovations to ten late resets;
- and its declared forward subtotal is below 49 at K32.

It is not yet a certified architecture.  The binding order is:

1. prove weak cut-state image membership and a stable target-free section;
2. prove strong future quotient identity or charge the cubic/quartic seam
   residual;
3. acquire or bound the transported cut-state error and its suffix cross;
4. acquire/bound the ten Gaussian-rest/reset errors and their Bellman crosses;
5. give literal CSE receipts for seam construction, covariance read, rest
   source, and every prefix operation inside the `46.086` subtotal.

Failure of the weak image or final transported-error gate closes this form.
Failure of the strong seam does not necessarily close the broader route, but
turns the mismatch into an explicit block in the exact elliptope rather than an
unpriced initialization.

## Graph-ready suffix nodes

### D14_L4_TO_FULL_AQ55_WEAK_SECTION

- status: open exact existence/stability gate;
- identity: exact match of `m,C,K21` through `Psi_55`;
- dimension: `p55AQ-q=58,419`, so flat capacity tail is zero only at a regular
  onto point;
- fail-fast necessary tests: `C>=0`, `ker C subset ker K21`, and
  `K21_i C+ K21_i^T<=8 C_ii^2`;
- promotion: image membership plus positive gauge-restricted `sigma_seam` and a
  target-free branch.

### D14_L4_TO_FULL_AQ55_STRONG_SEAM

- status: open future-identity gate;
- requirement: the D14-minus-AQ cubic/quartic cut state annihilates every L5--14
  future query, or is matched by fiber selection;
- warning: weak `C,K21` equality does not match all-distinct CP3 or q4 state;
- fiber gate: `DTheta|ker DPsi` must cover the required quotient stably.

### D14_PREFIX_ENDPOINT_TRANSPORT_GATE

- status: corrected exact endpoint formulation / transported value unknown;
- official score: final-layer raw MSE; all-layer prefix MSE gives no floor;
- endpoint: `||P_cut c_D + z_suf||_H^2`;
- lower bound: zero absent restricted `sigma_min` plus anti-cancellation/angle
  information;
- dependency: acquire the final Bellman action and cross at the actual cut.

### AQ55_SUFFIX_GREST_BELLMAN_GATE

- status: solved exact information gate / acquisition open;
- capacity: no positive flat `(C,K21)` tail at a regular r55 seam;
- absolute gate: `e0+e_suf+2chi<=0.204019`;
- causal receipt: ten `chi_l`, or full block elliptope;
- warning: the finite nonaffine L4 cut does not inherit the affine parity base,
  and the resident-reverse no-go remains.

### D14_AQ55_SUFFIX_COST_RECEIPT

- status: conditional subtotal, not complete;
- declared formula: `46.086` at K32 with about `2.914` slack;
- binding omitted premises: exact cut convention, literal prefix operations,
  seam solver, AQ Gram-read CSE, rest-source formation, q3/q4 reads, quadrature
  error, and dtype/stabilization;
- close trigger: a generic second `SS^T`-equivalent AQ Gram read adds `15.576`
  and exceeds the cap.

## 16. Native quadratic q211 cross: literal r49 tape audit

Let the synchronized native quadratic products in one common costate frame be

\[
 P^A_{ia}=M_i^A N_a^A,
 \qquad
 P^T_{ia}=M_i^T N_a^T,
 \qquad
 \Delta P_{ia}=P^T_{ia}-P^A_{ia},
\]

and let

\[
 \mathcal G(P,Q)=32\langle P,Q\rangle_F
                  +16\langle P^T,Q\rangle_F .
\]

For a fixed symmetric q211 costate `Theta`, polarization gives the exact
quadratic true-minus-AQ action

\[
 \Delta S_{211}^{(2)}
 =\sum_{i,a,b}(\Theta_i)_{ab}
 \left\{
 2\mathcal G(P^A_{ia},\Delta P_{ib})
 +\mathcal G(\Delta P_{ia},\Delta P_{ib})
 \right\}.
\tag{16.1}
\]

### 16.1 Zero-new-bank verdict

The declared r49 forward state does **not** expose (16.1) with zero fresh
ordered-product banks.  It carries `(m,u,A)` and the D10 Gaussian-rest
objects.  Those data determine the operands algebraically, but do not retain
the pre-symmetrization words `M_i N_a`.  Moreover, the declared construction
contains no resident true-minus-AQ quadratic product bank.

Adding an explicitly billed fused AQ connected-H4 surface does not change
this conclusion by itself.  Per-unit eigensystems, surface moments, an AQ
q211 scalar, and a symmetric parameter gradient are not a literal copy of
`P^A`, and they do not determine the finite quadratic term
`G(Delta P,Delta P)`.  A first derivative at the AQ point can at most supply
the linear cross after the exact direction has independently been acquired.
The skew information also matters:

\[
 \mathcal G(P,Q)
 =48\langle\operatorname{sym}P,\operatorname{sym}Q\rangle_F
  +16\langle\operatorname{skew}P,\operatorname{skew}Q\rangle_F,
\tag{16.2}
\]

so a post-symmetrization receipt is insufficient.

The exact zero-bank criterion is therefore literal: at the same seam, frame,
collision convention, and costate normalization, the tape must expose both
the full ordered `P^A` and `Delta P` (or an exactly equal fused bilinear plus
quadratic directional action).  Neither the forward r49 declaration nor a
generic reverse surface gradient satisfies this criterion.

### 16.2 Conditional product bill

At `r=49`, `N=1024`, and ten suffix layers, one fresh `n by k` ordered-product
family costs

\[
 c_{\rm bank}(k)=\frac{10kr^3}{N^2}=1.12199k
\]

width-matmul equivalents.  Once the products are resident, the exact native
reductions cost

\[
 c_{\rm red,dense}(k)=0.02290k(k+1),
 \qquad
 c_{\rm red,diag}(k)=0.04580k.
\]

Hence, under the optimistic premise that the synchronized true quadratic
matrices and the common factors are already acquired:

- if `P^A` is literally resident, build `P^T` once and subtract: the bill is
  `1.12199 k + c_red`, admitting `k<=6` for a dense common frame or `k<=7`
  for a common-diagonal frame under nine equivalents;
- if neither product is resident, build both `P^A` and `P^T`: the bill is
  `2.24398 k + c_red`, admitting only `k<=3` in either frame;
- if only increment operands are exposed and `Delta P` is expanded as
  `M^A Delta N + Delta M N^A + Delta M Delta N`, it contains three fresh
  product families before reductions, admitting only `k<=2` under the same
  cap (and a nonresident `P^A` must still be supplied separately).

These are product/reduction bills only.  They exclude the fused AQ H4 surface,
the acquisition of `Delta M` or an equivalent true quadratic source, hooks,
memory traffic, and the reverse q211 costate.  If `M^T,N^T` are not resident,
the problem is not merely an omitted arithmetic charge: (16.1) is
unidentified.  And (16.1) covers only the all-quadratic residual sector;
one-residual higher-chaos terms need a tilted generator, while two-or-more
higher-chaos residual slots need joint generators or an equal resident
residual action.

### Graph node: R49_AQ_NATIVE_CROSS_TAPE_RECEIPT

- status: zero-bank claim closed for the declared r49/D10 tape;
- exact observable: cross-plus-square identity (16.1), including symmetric
  and skew ordered-product information;
- zero-bank iff: literal `P^A` plus `Delta P`, or an exactly equal fused
  directional action, is exposed at the synchronized seam;
- optimistic cost: one bank if `P^A` is resident, two if neither `P^A` nor
  `P^T` is resident, three for direct increment expansion;
- identifiability warning: absent a true quadratic residual source, no bank
  census acquires the action.

## 17. Exact r55 affine and near-affine seam rank theorem

The r55 suffix has enough nominal parameters to match `(C,K21)`, but this
surplus is unavailable at the affine seam and becomes ill-conditioned near
it.  The statement can be made exact in the half-quadratic convention used in
Section 8.  Embed

\[
 s_i=(u_i,A_i)
 \quad\longmapsto\quad
 \widehat A_i=
 \begin{pmatrix}
 0&u_i^T\\
 u_i&A_i
 \end{pmatrix}
 \in\mathcal H_{56}:=
 \{H\in\operatorname{Sym}_{56}:H_{00}=0\}.
\]

Then, exactly,

\[
 C_{ij}=\frac12\operatorname{tr}(\widehat A_i\widehat A_j),
 \qquad
 K21_{ij}=\operatorname{tr}(\widehat A_i^2\widehat A_j).
\tag{17.1}
\]

The coefficient-space dimension is `1595`; quotienting the `O(55)` gauge
gives

\[
 1024(1595)-\frac{55\cdot54}{2}=1,631,795
\]

parameters, a surplus of `58,419` over the `1,573,376` observations.

### 17.1 Affine seam is rank-obstructed

Set every `A_i=0` and let the row matrix `U` of the `u_i` have rank 55.  The
covariance differential is the tangent to the rank-55 Gram variety, hence

\[
 \operatorname{rank}DC
 =1024\cdot55-\frac{55\cdot54}{2}=54,835.
\]

Its orthogonal missing space is

\[
 \mathcal N_U=\{Z=Z^T:ZU=0\},
 \qquad
 \dim\mathcal N_U
 =\frac{(1024-55)(1025-55)}2=469,965.
\tag{17.2}
\]

The quadratic variation `H_i=delta A_i` acts on K21 as

\[
 (F_UH)_{ij}
 =2u_i^T H_i u_j+u_i^T H_j u_i.
\tag{17.3}
\]

This map is generically onto all `1024^2` oriented K21 entries.  A finite
minor certificate is

\[
 \operatorname{rank}\{u_i u_i^T\}_{i=1}^{1024}=1024,
\]

and, for every anchor `p`, rank 1023 for the projected outer products
`(P_pu_i)(P_pu_i)^T`, `i != p`, in `Sym(u_p^perp)`.  These ranks fit because
`dim Sym_55=1540` and `dim Sym_54=1485`.  The standard annihilator argument
then forces every left-null matrix of (17.3) to vanish.

Even under these generic minors,

\[
 \operatorname{rank}D(C,K21)
 =54,835+1,048,576=1,103,411,
\]

so the cokernel is exactly the `469,965`-dimensional covariance normal
(17.2).  Thus neither raw-Gaussian initialization nor an exactly affine L4
cut has a local r55 `(C,K21)` section.

### 17.2 Near-affine Schur gate and exact 58,419 fiber

Let `A_i=tB_i` and retain an affine tuple satisfying the K21 minors.  Since
`F_U` is onto, its kernel has dimension

\[
 1024\,\dim\operatorname{Sym}_{55}-1024^2
 =1024(1540)-1,048,576=528,384.
\]

The leading quadratic contribution to the missing covariance normal is

\[
 (\mathcal L_BH)_{ij}
 =\frac12\operatorname{tr}(H_iB_j+B_iH_j).
\]

With `Pi_U(Y)=P_UYP_U` the projection onto `N_U`, define the exact leading
Schur map

\[
 \boxed{
 \mathscr S_{U,B}^{(55)}
 =\Pi_U\mathcal L_B\big|_{\ker F_U}:
 \ker F_U\longrightarrow\mathcal N_U .}
\tag{17.4}
\]

The dimensions satisfy

\[
 528,384-469,965=58,419,
\]

which is exactly the global quotient surplus.  Therefore the r55 seam opens
all missing covariance directions at **first order** exactly when

\[
 \operatorname{rank}\mathscr S_{U,B}^{(55)}=469,965.
\tag{17.5}
\]

Under a nonzero Schur minor, the corresponding full Jacobian minor obeys

\[
 \det J(t)=c\,t^{469,965}+O(t^{469,966}),
 \qquad c\ne0,
\]

and the newly opened singular values scale as

\[
 \sigma_{\rm new}=|t|\,\sigma(\mathscr S_{U,B}^{(55)})+O(t^2).
\tag{17.6}
\]

Thus the larger r55 fiber makes a submersion more plausible than r54, but it
does not remove the seam receipt: an almost-affine prefix is necessarily
ill-conditioned as `1/|t|`, and parameter counting alone cannot certify the
cut.  If (17.5) fails, this proves only that the first-order route is
singular; a section could still open through a higher power of `t`, but then
requires a separate higher-order Schur certificate and is even more poorly
conditioned.

The equivalent adjoint certificate avoids forming the half-million-row
primal matrix.  For `Z in N_U`,

\[
 (\mathcal L_B^*Z)_p=\sum_j Z_{pj}B_j,
\]

while

\[
 (F_U^*\Lambda)_p
 =\sum_i\Lambda_{ip}u_i u_i^T
 +2\operatorname{sym}\!\left[
 u_p\left(\sum_j\Lambda_{pj}u_j\right)^T
 \right].
\]

Consequently (17.5) holds iff the simultaneous system

\[
 \boxed{
 \sum_j Z_{pj}B_j
 =\sum_i\Lambda_{ip}u_i u_i^T
 +2\operatorname{sym}\!\left[
 u_p\left(\sum_j\Lambda_{pj}u_j\right)^T
 \right],\qquad p=1,\ldots,1024 }
\tag{17.7}
\]

with `Z=Z^T` and `ZU=0` has no solution with `Z != 0`.  This is the exact
symbolic seam receipt for a proposed quadratic prefix.

### Graph node: R55_AFFINE_SEAM_RANK_OBSTRUCTION

- status: exact no-local-section theorem at `A=0`;
- affine ranks: covariance `54,835`, generic K21 `1,048,576`;
- exact cokernel: `469,965` covariance-normal directions;
- implication: r55 nominal capacity cannot be borrowed at an affine cut.

### Graph node: R55_NEAR_AFFINE_SCHUR_SECTION_GATE

- status: exact iff first-order gate / deployed cut unresolved;
- Schur map: (17.4), from dimension `528,384` to `469,965`;
- surplus: exactly `58,419`, agreeing with quotient counting;
- pass: full row rank `469,965` plus nonzero finite minor;
- conditioning: newly opened singular values are linear in quadratic-prefix
  scale `|t|`.

## 18. Late-cut repair: pay the AQ Gram and shorten the suffix

The generic AQ low-Gram read is not a CSE with coefficient transport.  That
closes the proposed ten-transition L5--14 subtotal, but it does not close every
D14-prefix/AQ55-suffix schedule.  Moving the cut later trades frozen-prefix
error for an exactly paid Gram.

Let `s` be the number of AQ55 suffix resets among the fourteen costed
transitions and keep D14 for the other `14-s`.  Write

\[
 d_{55}=55+\frac{55\cdot56}{2}=1595.
\]

Under the same census assumptions as Section 14, but now charging one AQ
coefficient push **and** one independent low-Gram product at every suffix
reset, the exact declared subtotal is

\[
 \boxed{
 C_{\rm paid\ Gram}(s,K)
 =2(14-s)+2s
 +2s\frac{1595}{1024}
 +s\frac{55^3}{1024^2}
 +sK\frac{55^2}{1024^2}.}
\tag{18.1}
\]

The first two terms combine to 28 because D14 prefix transport and the D10
suffix plane each cost two equivalents per transition.  At `K=32`, every AQ
suffix reset therefore adds about `3.3662` beyond that base:

\[
 C_{\rm paid\ Gram}(s,32)\simeq28+3.3662s.
\]

Consequently `s=6` costs about `48.197`, leaving about `0.803`, while `s=7`
costs about `51.563` and is arithmetically closed.  In the present indexing,
the maximal fully Gram-paid schedule is therefore a D14 prefix through L8
and an AQ55 suffix over L9--14.

There is one exact terminal optimization.  If the last suffix reset stores
the D10 **total** covariance solely for the final linear/read transition, no
future reset needs its Gaussian-rest decomposition.  The final full AQ Gram
can then be omitted; only its diagonal row norms are needed for the final
univariate mean.  Conditional on that literal state switch,

\[
 \boxed{
 C_{\rm terminal\ total}(s,K)
 =28+(2s-1)\frac{1595}{1024}
 +s\frac{55^3}{1024^2}
 +sK\frac{55^2}{1024^2}.}
\tag{18.2}
\]

At `s=6,K=32` this is about `46.640`, leaving about `2.360`.  It is not a
generic omission: it is lawful only if the last stored plane is demonstrably
the total covariance, the AQ low component is not added again, and no later
consumer asks for the separated rest.

### 18.1 Why the late cut is value-plausible but not value-certified

The existing chain-value receipt makes the trade attractive.  In
`CHAIN_VALUE_MAP.json`, the variance-channel cells L9--L15 sum to `15.64`,
versus `17.46` across L1--L15, so roughly `89.6%` of that channel's linear
attribution lies at or after the proposed cut.  The corresponding k3 cells
sum to `1.57` versus `2.17`, about `72.4%`.  These are prioritization facts,
not a predicted raw MSE for a jointly refitted nonlinear replacement: LOO
cells and channel sums cannot simply be subtracted from raw MSE.

Because the official score is final-layer raw MSE, freezing nine diagnostic
rows creates no `9/16` loss floor.  The late cut is instead graded by the D14
cut-state error transported through L9--final together with all six suffix
innovations and their cross terms.  Without a restricted singular value and
anti-cancellation certificate this transported prefix contribution has no
positive standalone lower bound.  Subject to that corrected endpoint gate,
the later cut has three real advantages: it pays the previously omitted Gram,
starts from a more strongly nonaffine D14 state than L4, and reduces
Gaussian-rest/Bellman innovations from ten to six.

The remaining cost caveat is severe.  The `0.803` strict slack (or `2.360`
under the terminal-total switch) must cover every operation not literally
fused into the K32 local surface term: complete AQ connected-H4 ownership,
q3/q4 reads, seam construction, stabilization, hooks, and memory traffic.  A
fresh generic surface plane or reverse chain still closes the card.

### Graph node: D14_L8_AQ55_SUFFIX_GRAM_PAID

- status: exact conditional cost reopen / value and surface receipts open;
- schedule: D14 through L8, AQ55 plus D10 over six L9--14 resets;
- strict paid-Gram cost: `48.197` at K32;
- terminal-total variant: `46.640` iff the last separated rest is never
  consumed;
- value support: 89.6% of recorded variance-channel linear attribution lies
  in L9--L15, but this is not a raw-MSE prediction;
- hard gates: transported cut-to-final error/action, actual r55 seam
  minor/conditioning,
  complete H4 surface ownership, six-block Bellman coherence, and literal
  end-to-end census.

## 19. Incremental bill for an explicit r49 connected-H4 surface

The roughly `40.02`-equivalent r49+Gaussian-rest subtotal leaves approximately
`8.98` before hooks.  That subtotal carries `P<=2`; it does not by itself own
the nonlinear AQ connected-H4/q211 reset surface.  There are three materially
different implementation branches.

### 19.1 Same-eigensystem, same-node fused surface action

Assume the base AQ reset's per-unit eigensystem, resolvents, and `K` certified
Fourier nodes are exposed at the exact H4 seam.  For one residualized future
costate decomposition, let

- `q` be the number of distinct latent direction vectors whose coordinates
  must be transformed into each unit eigenbasis;
- `h` be the number of distinct symmetric direction pairs whose tilted
  covariance bilinears are required; and
- `a` be the number of final fused H211 atoms/scalar coefficient contractions.

At one unit and layer, the direction transforms cost `q r^2`.  At every node,
the tilted means cost `q r`, the symmetric Hessian/bilinears cost `h r`, and
the final atom assembly costs `a`.  Thus the leading exact direction/node
census is

\[
 \boxed{
 C_{\rm fused}(D,K;q,h,a)
 ={D\over n^2}\left[
 qr^2+Kr(q+h)+Ka
 \right].}
\tag{19.1}
\]

This counts arithmetic at already resident nodes/eigensystems; quadrature
certification, hooks, and memory are separate.  At `r=49,n=1024,K=32,D=14`,

\[
 \boxed{
 C_{\rm fused}
 ={14\over1,048,576}
 \left(3969q+1568h+32a\right).}
\tag{19.2}
\]

The corresponding D10-local expression replaces 14 by 10.  Therefore the
exact pre-hook headroom gate is

\[
 3969q+1568h+32a
 \le {8.98\cdot1,048,576\over14}
 \quad(D=14).
\tag{19.3}
\]

For unshared H211 atoms, each with three distinct directions and the four
bilinears `(UU,US,UT,ST)`, `q=3a` and `h=4a`.  Then

\[
 C_{\rm fused}=0.24314\,a\quad(D=14),
\]

so at most 36 atoms fit the nominal `8.98`; atom 37 is already just over the
cap before hooks.  If a complete symmetric Hessian is formed on one common
`q`-direction span, `h=q(q+1)/2`; with atom assembly fused rather than
enumerated, `q=26` fits and `q=27` fails.  These examples are arithmetic caps,
not a rank receipt: the old-projected future H211 costate must actually admit
the stated shared decomposition.

Formula (19.1) produces a **costate-contracted surface action**.  It is not a
full external H4 plane.  It owns the nonlinear AQ surface only when every
deployed output/future consumer factors through the declared directions and
the same-node tape is literal.  If eigensystems are not reused, add

\[
 {Dr^3\over n^2}
 =1.57079\ (D=14),\qquad 1.12199\ (D=10)
\]

for each separately factorized family.

### 19.2 Full native ordered-product action

For common-factor rank `k`, a freshly built ordered bank
`P_ia=M_i N_a` costs

\[
 C_{P,{\rm build}}={Dkr^3\over n^2}.
\]

The full native reductions cost

\[
 C_{P,{\rm dense}}={Dk(k+1)r^2\over n^2},
 \qquad
 C_{P,{\rm diag}}={2Dkr^2\over n^2}.
\tag{19.4}
\]

The declared r49 tape does not retain `P`, so the build is incremental.  For
all fourteen AQ resets, the one-bank totals are

\[
 1.57079k+0.032057k(k+1)
 \quad\hbox{or}\quad
 1.57079k+0.064114k,
\]

and either frame permits only `k<=5` under `8.98`.  If the action is needed
only over ten proved-active resets, the totals are

\[
 1.12199k+0.02290k(k+1)
 \quad\hbox{or}\quad
 1.12199k+0.04580k,
\]

giving `k<=6` dense or `k<=7` common-diagonal.  If `P` were literally
resident, reduction-only caps would be `k<=16` dense / about 140 diagonal for
D14, and `k<=19` / about 196 for D10.

This branch has a binding ownership limit: applied to the carried quadratic
projection, native `P` evaluates `kappa4(X)`.  It does **not** supply the
nonlinear AQ rest `kappa4(Y^A)-kappa4(X)`.  Therefore its attractive small-rank
bill is not, by itself, a bill for the missing connected-H4 surface; the fused
surface of Section 19.1 or an equal nonlinear oracle is still required.

### 19.3 One-residual tilted generator

Let `q_jet` be the average symmetric/decomposable rank of the complete-old,
strict-positive-depth cubic costate polynomial seen by each residual source.
One unfused exact third-jet family has the constructive upper bill

\[
 \boxed{
 C_{\rm jet}\le {Dq_{\rm jet}r^3\over n^2},}
\tag{19.5}
\]

and, with resident eigensystems plus a proved fused reverse evaluation, the
direction-read floor is `D q_jet r^2/n^2`.  At r49 this means

\[
\begin{array}{c|cc}
 &D=14&D=10\\ \hline
 r^3\text{ upper/family}&1.57079&1.12199\\
 r^2\text{ fused floor/family}&0.032057&0.02290.
\end{array}
\]

Hence the advertised unfused construction fits at most five D14 families or
eight D10 families arithmetically; the eighth D10 family leaves essentially
no hook margin.  For a generic dense singleton-residual q211/K31 costate,
`q_jet` is order `n`, so the exact construction costs about `1608.5`
equivalents for D14 or `1148.9` for D10 and is closed by orders of magnitude.

More fundamentally, AQ same-node data identify the AQ-induced ReLU residual,
not an arbitrary physical true-minus-AQ residual.  If that residual source is
not separately resident or supplied by a lawful oracle, (19.5) is not an
omitted arithmetic charge: the desired generator is unidentified.  Even when
the one-residual generator is owned, it covers only the four one-residual
slots; two-, three-, and four-residual joint actions remain separate.

### 19.4 Verdict

Only branch 19.1 has a plausible path to billing the missing nonlinear AQ H4
surface inside `8.98`, and only at certified small future mixed rank (for
example at most 36 completely unshared H211 atoms over D14).  The literal
ordered-P branch is cheaper only for `kappa4(X)`, which is already low-chaos
ownership.  The one-residual generator fits only bounded sewn polynomial rank
and does not acquire the zero-residual AQ surface or any multi-residual block.

### Graph node: R49_AQ_CONNECTED_H4_INCREMENTAL_BILL

- status: exact conditional census / mixed rank and tape reuse open;
- fused surface: (19.1)--(19.3), `a<=36` for unshared D14 H211 atoms under
  nominal headroom;
- native P: `k<=5` over D14 with a fresh bank, but owns only low-chaos
  `kappa4(X)`;
- one-residual generator: at most five unfused D14 jet families, generic
  rank-`n` action cost-closed and truth-source-unidentified;
- promotion: literal same-node/eigensystem reuse, complete-future costate rank,
  quadrature tail, old/refit residualization, and hooks must all pass.

## 20. Decisive r55 first-order section obstruction from the Veronese null

The positive `2,979` hook count is not sufficient.  A second universal
structure forces a much larger r55 first-order cokernel.

Let `P=R^n=L direct_sum K`, where `L=col(U)` has dimension `r`, `K` has
dimension `m=n-r`, and

\[
 E=\operatorname{Sym}^2L,
 \qquad d=\frac{r(r+1)}2.
\]

Define the quadratic-evaluation/Veronese map

\[
 \mathcal Q_U:E\longrightarrow P,
 \qquad
 (\mathcal Q_U G)_i=u_i^TGu_i.
\tag{20.1}
\]

### 20.1 Affine K21 surjectivity forces `Q_U` onto

For a quadratic tangent bank `H_i`, the affine K21 differential is

\[
 (F_UH)_{ij}=2u_i^TH_i u_j+u_i^TH_j u_i.
\]

After right projection onto `K`, the first term vanishes because it has right
factor `U`.  The second term factors through `Q_U`.  If
`q=rank Q_U`, then

\[
 \operatorname{rank}F_U
 \le nr+q(n-r).
\tag{20.2}
\]

Therefore `rank F_U=n^2` implies `q=n`.  The Veronese map is not an optional
genericity assumption in the strongest affine-section case; it is necessary.
At r55,

\[
 d=1540,qquad
 E_0:=\ker\mathcal Q_U,qquad
 \dim E_0=d-n=516.
\tag{20.3}
\]

### 20.2 The quotient dual lives entirely in `K tensor E_0`

Put

\[
 \mathcal A=\operatorname{range}F_U^*,
 \qquad
 \mathcal B=\mathcal A\cap(K\otimes E),
 \qquad
 \mathcal C=\mathcal B^\perp\subset K\otimes E.
\]

Equivalently,

\[
 \mathcal C
 =\{H\in K\otimes E:F_UH\in F_U(L\otimes E)\}.
\tag{20.4}
\]

Write `H=sum_a k_a tensor G_a` in a basis of K.  Every shared-L output
`F_U(L tensor E)` has right factor in L.  Right projection of (20.4) onto K
therefore gives

\[
 0=(F_UH)P_K
 =\sum_a(\mathcal Q_U G_a)k_a^T.
\]

The `k_a` are independent, so every `G_a in E_0`.  Hence the exact inclusion

\[
 \boxed{\mathcal C\subseteq K\otimes E_0.}
\tag{20.5}
\]

This is stronger than the hook dimension count; it identifies which quadratic
coordinates the Schur quotient can actually test.

### 20.3 Every r55 prefix leaves a 102,831-dimensional normal kernel

Let `N in K tensor E` be an arbitrary quadratic prefix bank.  The adjoint of
the first-order normal Schur map is

\[
 \mathscr S_{U,N}^*:\operatorname{Sym}K
 \longrightarrow(K\otimes E)/\mathcal B,
 \qquad Z\longmapsto[ZN].
\tag{20.6}
\]

Its dual action on `H in C` is, up to the fixed convention factor,

\[
 H\longmapsto\operatorname{sym}(HN^T).
\tag{20.7}
\]

By (20.5), only the `E_0` projection `N_0` of N is visible in (20.7).  Regard
`N_0` as an `m by 516` matrix.  Its transpose has a kernel

\[
 R=\ker N_0^T\subset K,
 \qquad
 \dim R\ge969-516=453.
\]

For every `Z in Sym R`, `ZN_0=0`, hence `ZN` annihilates all of C and belongs
to B.  Thus

\[
 \boxed{
 \operatorname{Sym}R\subseteq\ker\mathscr S_{U,N}^*,
 \qquad
 \dim\ker\mathscr S_{U,N}^*
 \ge\frac{453\cdot454}{2}=102,831.}
\tag{20.8}
\]

Since `dim Sym K=469,965`, for **every** r55 prefix,

\[
 \boxed{
 \operatorname{rank}\mathscr S_{U,N}
 \le367,134,
 \qquad
 \operatorname{cork}\mathscr S_{U,N}
 \ge102,831.}
\tag{20.9}
\]

This closes the hoped-for first-order r55 section despite the positive hook
margin.  If `F_U` is not onto, the affine block already fails; if it is onto,
(20.2)--(20.9) apply.

The exact block elimination still has odd orders

\[
 J_{\rm normal}(t)=t\mathscr S_1+t^3\mathscr S_3+O(t^5).
\]

At the maximal first-order rank in (20.9), the cubic fallback has source and
target dimensions

\[
 528,384-367,134=161,250,
 \qquad
 469,965-367,134=102,831,
\]

whose difference is again the global `58,419` surplus.  A finite nonaffine
D14 cut is therefore not universally impossible, but it needs an onto induced
cubic map and at least 102,831 near-affine singular modes scale no better than
`t^3`; the right inverse is `Omega(t^-3)` along them.

### 20.4 General threshold

The same argument leaves

\[
 \dim R\ge
 (n-r)-\left(\frac{r(r+1)}2-n\right)
 =2n-r-\frac{r(r+1)}2.
\tag{20.10}
\]

Thus first-order surjectivity is universally impossible until

\[
 \frac{r(r+3)}2\ge2n.
\tag{20.11}
\]

At `n=1024`, r62 still fails (`2015<2048`) and r63 is the first rank not
killed by this Veronese-null obstruction (`2079>=2048`).  This supersedes r55
as the first dimensionally possible first-order AQ seam rank.

### Graph node: R55_VERONESE_NULL_SCHUR_NOGO

- status: universal first-order no-go;
- necessary premise: affine K21 surjectivity forces the rank-1024 Veronese
  evaluation map;
- quotient support: `B^perp subset K tensor ker(Q_U)`, with quadratic null
  width 516;
- every prefix: at least a 453-dimensional invisible K-subspace and 102,831
  symmetric normal modes;
- rank bound: first-order Schur rank at most 367,134 versus 469,965 required;
- fallback: cubic map `161,250 -> 102,831`, with `Omega(t^-3)` near-affine
  conditioning;
- new threshold: r63 is the first rank not universally killed by (20.11).

## 21. General rank--window theorem under the exact-Gram bill

The Veronese-null obstruction and the late-cut bill can be combined without a
numerical experiment.  Put

\[
 e_r={r(r+1)\over2},\qquad
 d_r=r+e_r={r(r+3)\over2},
\]

and let `L` be the number of AQ/Gaussian-rest suffix resets.  At `n=1024`,
`K=32`, and with zero seam charge, define the exact per-reset increment

\[
 \boxed{
 \Delta_r={2d_r\over1024}
 +{r^3+32r^2\over1024^2}.}
\tag{21.1}
\]

If the last reset must materialize its full all-pairs low AQ Gram, the subtotal
is

\[
 \boxed{C_{\rm full}(r,L)=28+L\Delta_r.}
\tag{21.2}
\]

If the terminal consumer reads only the scalar/factored total variance and no
later operation consumes the separated AQ/rest plane, exactly one Gram can be
omitted:

\[
 \boxed{C_{\rm no\ terminal\ Gram}(r,L)
 =28+L\Delta_r-{d_r\over1024}.}
\tag{21.3}
\]

Both costs are strictly increasing in `r`.  Section 20 shows that the smallest
rank not universally killed at first order is `r=63`.  Here

\[
 d_{63}=2079,\qquad
 \Delta_{63}
 ={2079\over512}+{377055\over1048576}
 \simeq4.42013454.
\]

It follows immediately that

\[
\begin{array}{c|c|c|c}
\text{terminal convention}&\text{longest }L&\text{smallest admissible }r
 &\text{subtotal}\ \\ \hline
\text{full terminal Gram}&4&63&45.68054\\
\text{no terminal Gram}&5&63&48.07040.
\end{array}
\tag{21.4}
\]

The next reset is impossible arithmetically: full-Gram `r63,L=5` costs
`50.10067`, and no-terminal-Gram `r63,L=6` costs `52.49053`.  Monotonicity in
rank makes these universal cost closures for every `r>=63`.  Conversely, the
complete rank window at the two longest first-order-eligible schedules is

\[
 \boxed{
 \begin{array}{c|c}
 L=4,\ \text{full terminal Gram}&63\le r\le68,\\
 L=5,\ \text{no terminal Gram}&63\le r\le64.
 \end{array}}
\tag{21.5}
\]

Indeed `C_full(68,4)=48.62329` while `C_full(69,4)>49`, and
`C_no terminal Gram(64,5)=48.71875` while the r65 value is about `49.378`.
These are pre-hook arithmetic statements.  The r63 five-reset branch leaves
only `0.92960`; r64 leaves `0.28125`.  In particular neither margin can be
silently used for a generic connected-H4 surface or a seam solve.

### 21.1 The complete higher-order trade frontier

Let

\[
 p_r=1024d_r-{r(r-1)\over2},\qquad
 q={1024\cdot1025\over2}+1024^2=1,573,376,
\]

and `g_r=p_r-q`.  The number `g_r` is the exact global parameter surplus of
the weak `(C,K21)` map after quotienting the orthogonal gauge.  Let

\[
 \rho_r=(2048-d_r)_+,
 \qquad k_r={\rho_r(\rho_r+1)\over2}.
\tag{21.6}
\]

The Veronese-null theorem guarantees at least `k_r` first-order invisible
covariance-normal modes.  The cost and obstruction frontiers meet as follows:

\[
\begin{array}{c|r|r|r|r|c|c}
r&d_r&g_r&\rho_r&k_r&L_{\max}^{\rm full}&
L_{\max}^{\rm no\ terminal\ Gram}\\ \hline
54&1539&1,129&509&129,795&6&6\\
55&1595&58,419&453&102,831&6&6\\
56&1652&116,732&396&78,606&6&6\\
57&1710&176,068&338&57,291&5&6\\
58&1769&236,427&279&39,060&5&6\\
59&1829&297,809&219&24,090&5&5\\
60&1890&360,214&158&12,561&5&5\\
61&1952&423,642&96&4,656&5&5\\
62&2015&488,093&33&561&4&5\\
63&2079&553,567&0&0&4&5\\
64&2144&620,064&0&0&4&5.
\end{array}
\tag{21.7}
\]

Rank 53 and below are universally closed at **all** orders by the global
dimension count (`g_53=-55,138`).  Ranks 54--62 are different: they are
first-order obstructed, but there is no remaining universal dimension
obstruction to a higher-order seam.

More precisely, after eliminating the affine K21 block, write the normal
Schur series

\[
 \mathscr J_r(t)=t\mathscr S_1+t^3\mathscr S_3+O(t^5).
\]

If the first-order cokernel has dimension `c`, then

\[
 \dim\ker\mathscr S_1=g_r+c.
\]

The exact next gate is therefore

\[
 \boxed{
 \pi_{\operatorname{coker}\mathscr S_1}\mathscr S_3
 \big|_{\ker\mathscr S_1}:
 \mathbb R^{g_r+c}\twoheadrightarrow\mathbb R^c.}
\tag{21.8}
\]

At maximal first-order rank for r62, this is a map
`R^488654 -> R^561`.  Thus r62 with five terminal-no-Gram resets is a sharp
conditional compromise: its subtotal is `47.43295`, leaving `1.56705`, and
only 561 normal modes are universally forced to cubic order.  Passing (21.8)
would make those modes scale as `t^3`; failure sends the surviving cokernel to
the next odd-order gate.  Parameter surplus does not prove (21.8), but there
is no universal higher-order no-go from the Veronese argument once `r>=54`.

The flat response-dimension diagnostic does not choose among these seams.  In
the convention which produced the recorded r49 number it is

\[
 e_{{\rm cap},r}={(q-p_r)_+\over q}.
\]

Thus `e_cap,49=0.171590<e_star=0.204019`, while the same diagnostic is exactly
zero for every rank in (21.7), because `g_r=p_r-q>0` from r54 onward.  This is
only a necessary parameter-capacity statement; it is not an endpoint raw-MSE
certificate or a section-rank proof.

### 21.2 The r63 five-reset branch has no resident D14 initialization

The arithmetic survivor in (21.4) still needs an oriented AQ cut state

\[
 S_{63}=\{(u_i,A_i)\}_{i=1}^{1024},
 \qquad u_i\in\mathbb R^{63},\quad A_i\in\operatorname{Sym}_{63},
\]

with 2,079 coordinates per unit and the correct joint `(C,K21)` orientation.
The literal D14 cut does not store this object.  It stores moment/response
planes and separately typed CP3, CP4, star, and Q24 carriers.  A covariance
factor alone would not orient K21: covariance is quadratic in `(u,A)`, while
K21 is cubic and changes under the unresolved signed branch.

The tempting rank-2048 CP4 factor is not a near-complete AQ63 bank.  Its
columns are signed fourth-order lineage atoms; its read uses elementwise cubic
and quartic powers with separate lambdas.  Its Euclidean Gram is not the D14
covariance, its columns have no `Sym^2 R^63` labeling, and it has no common
orientation with the CP3 pair factors.  Zero-padding is lawful only if a
synchronized lower-rank AQ tuple bank is already resident, which the declared
D14 lineage does not contain.

Therefore the five-reset r63 branch still requires one of the same three
acquisition mechanisms as r55: a cold nonlinear `(C,K21)` factor/orientation
solve, warm AQ carry, or a new literal view theorem.  Warm carry through the
nine omitted prefix transitions costs at least

\[
 9{d_{63}\over1024}=18.27246
\]

coefficient-product equivalents before any prefix Gram or reset work, far
above the `0.92960` margin.  A five-transition dense future/reverse action
costs ten products before reset adjoints.  No cold solver or D14-to-AQ63 view
is present in the subtotal.  Hence r63,L5 is the unique cheapest
first-order-eligible **arithmetic shell**, not yet an executable late-cut
architecture.

### Graph node: AQ_RANK_WINDOW_VERONESE_COST_FRONTIER

- status: exact symbolic frontier / seam acquisition open;
- universal first-order threshold: r63;
- full terminal Gram: at most four resets, with `63<=r<=68`;
- terminal factored/scalar read: at most five resets, with `63<=r<=64`;
- cubic compromise: r62,L5 costs `47.43295` but must open at least 561
  cubic-order normal modes;
- all-order dimension threshold: r54; r53 and below are impossible for the
  complete weak interface;
- deployment blocker: literal D14 has no oriented AQ63 factor view, and known
  warm/reverse/cold acquisition routes do not fit the remaining margin.

## 22. Price strict-triple trace collapse: cheap contraction, absent query

There is an exact rank-saving identity for a Price source after its future
q211 costate has been acquired.  On strict triples define

\[
 Z(R)_{i;jk}=R_{jk},\qquad i\ne j,k,
\]

in the same ordered/symmetric convention as the q211 costate `L`.  Then

\[
 \boxed{
 \langle L,Z(R)\rangle
 =\langle\Gamma_L,R\rangle_F,
 \qquad
 (\Gamma_L)_{jk}=\sum_{i\ne j,k}L_{i;jk}.}
\tag{22.1}
\]

Thus once `Gamma_L` is resident, the source contraction is an `O(n^2)`
reduction.  It must not be charged as a generic CP decomposition of the
three-index Price tensor.

This identity does not make the query resident.  The D10 forward Price-gate
matrix is a local covariance derivative, not the future trace (22.1).  The
full-AQ forward state contains `(m,U,A)` and local potentials, not `L` or
`Gamma_L`; the declared D10/D11 state likewise has no reverse q211 costate.
Moreover the subspace `range Z` is not generically invariant under a dense
mix followed by neuron-specific reset sewing.  Equivalently, there is no
proved intertwiner

\[
 J_lZ=Z\widetilde J_l
\]

that would let `Gamma_L` close as a two-index reverse state.  Trace collapse
at the source seam therefore does not replace construction of the future
query.  One must either run the full reverse action and trace it, or prove a
new directly fused two-index adjoint recurrence.  Even a closed dense-plane
reverse conjugation costs two products per suffix transition before reset
adjoints; no such reverse row is included in the forward subtotal.

### Graph node: PRICE_Q211_TRACE_COLLAPSE_RESIDENCY_GATE

- status: exact scalar collapse / reverse acquisition open;
- solved: `<L,Z(R)>=<Gamma_L,R>` avoids a CP-rank source charge;
- not resident: D10's local Price gate is not `Gamma_L`, and full AQ stores no
  future q211 costate;
- closure gate: prove `J_l Z=Z Jtilde_l` through dense mix and reset sewing, or
  bill a new reverse action;
- implication: source evaluation is cheap after query acquisition, but does
  not repair the late-cut Bellman/resident-reverse objection.

## 23. Exact r62 cubic Schur operator and the sole compatibility syzygy

The 561 first-order-dark modes need not be studied through an unspecified
`O(t^3)` remainder.  The complete cubic operator, including the `u`-variation
and tangent-covariance correction, has a closed form.

At the near-affine prefix `(U,A=tB)`, write a parameter variation as `(X,H)`.
Use

\[
 (D_UX)_{ij}=x_i^Tu_j+u_i^Tx_j,
\]

\[
 (L_BH)_{ij}={1\over2}
 \operatorname{tr}(H_iB_j+B_iH_j),
\tag{23.1}
\]

and retain the affine K21 map `F_U` from Section 20.  The two remaining K21
blocks are

\[
 \boxed{
 (G_BX)_{ij}
 =2x_i^TB_ju_i+2x_i^TB_i u_j+2u_i^TB_i x_j,}
\tag{23.2}
\]

\[
 \boxed{
 (M_BH)_{ij}
 =\operatorname{tr}\!\left[
 H_i(B_iB_j+B_jB_i)+H_jB_i^2\right].}
\tag{23.3}
\]

Thus the exact differential block series is

\[
 \begin{pmatrix}
 D_U&tL_B\\
 tG_B&F_U+t^2M_B
 \end{pmatrix}.
\tag{23.4}
\]

Let `P_T` and `P_N` project covariance planes onto the affine Gram tangent
and normal spaces.  Choose a gauge-fixed right inverse `P` of `D_U` on the
tangent and a right inverse `R_F` of `F_U`.  For `H_0 in ker F_U`, the unique
chosen formal kernel lift begins

\[
 X(t)=-tP P_TL_BH_0+O(t^3),
\]

\[
 H(t)=H_0-t^2R_F
 \left(M_B-G_BP P_TL_B\right)H_0+O(t^4).
\]

Consequently

\[
 \boxed{
 \begin{aligned}
 \mathscr S_1&=P_NL_B|_{\ker F_U},\\
 \mathscr S_3&=-P_NL_BR_F
 (M_B-G_BP P_TL_B)|_{\ker F_U}.
 \end{aligned}}
\tag{23.5}
\]

The involution

\[
 (U,A)\longmapsto(U,-A),
 \qquad (C,K21)\longmapsto(C,-K21)
\]

makes `X(t)` odd and `H(t)` even.  Hence the normal Schur series contains
only `t,t^3,t^5,...`.  This forbids a quadratic escape; it does not kill the
explicit cubic word (23.3).

### 23.1 The right inverses disappear from the intrinsic cubic minor

Let `Z in coker S_1`.  Exactly because `Z` annihilates
`P_NL_B(ker F_U)`, there is a K21 multiplier `Lambda_Z` satisfying

\[
 \boxed{L_B^*Z=F_U^*\Lambda_Z.}
\tag{23.6}
\]

For `H in ker F_U intersect ker S_1`, (23.5) and
`F_UR_F=I` give

\[
 \boxed{
 \langle Z,\mathscr S_3H\rangle
 =-\left\langle\Lambda_Z,
 (M_B-G_BP P_TL_B)H\right\rangle.}
\tag{23.7}
\]

This identity is independent of `R_F`; changing `P` by a gauge section also
changes only a represented first-Schur row.  It isolates the exact
compatibility obstruction.  A cubic-dark covector is precisely a nonzero
`Z in coker S_1` for which

\[
 \boxed{
 (M_B-G_BP P_TL_B)^*\Lambda_Z
 \in\operatorname{range}F_U^*
   +L_B^*\mathcal N_U.}
\tag{23.8}
\]

At maximal r62 first-order rank, `coker S_1=Sym R`, `dim R=33`.
Therefore the induced cubic map is onto all 561 forced modes iff (23.8)
has no solution with `0 != Z in Sym R`.  This is the complete
representation/matching condition; parity and the Veronese null contribute
no additional automatic equation.

The adjoint of the direct cubic block, useful for a symbolic minor, is

\[
 \boxed{
 (M_B^*\Lambda)_k
 =\sum_j\Lambda_{kj}(B_kB_j+B_jB_k)
  +\sum_i\Lambda_{ik}B_i^2.}
\tag{23.9}
\]

### 23.2 A correction-free 426,496-dimensional witness space

There is a much smaller sufficient certificate than constructing the full
half-million-column Schur matrix.  Put

\[
 \mathcal K_0=\ker F_U\cap\ker L_B.
\]

Since the quadratic-bank ambient dimension at r62 is
`1024*1953=1,999,872`,

\[
 \boxed{
 \dim\mathcal K_0
 \ge1,999,872-1,048,576-524,800
 =426,496.}
\tag{23.10}
\]

Every `H in K_0` already has zero first Schur output and zero tangent
correction, so (23.7) reduces to

\[
 \boxed{
 \langle Z,\mathscr S_3H\rangle
 =-\langle M_B^*\Lambda_Z,H\rangle.}
\tag{23.11}
\]

Choose a basis `{Z_b}_{b=1}^{561}` of `Sym R`.  The existence of any 561
banks `{H_a} subset K_0` for which

\[
 \boxed{
 \det\left[
 \langle M_B^*\Lambda_{Z_b},H_a\rangle
 \right]_{a,b=1}^{561}\ne0}
\tag{23.12}
\]

is an explicit symbolic surjectivity minor for the **full** cubic Schur map,
not merely its Jordan core.  It simultaneously resolves the affine K21
right inverse, tangent correction, and normal projection.

There is an even more structured source space.  Let

\[
 E_0=\ker Q_U,
 \qquad
 \mathcal D_i=\{H\in E_0:H u_i=0\}.
\]

Then `dim D_i>=929-62=867`.  A bank with `H_i in D_i` lies in
`ker F_U` term by term: `Q_UH_i=0` kills the evaluation column and
`H_iu_i=0` kills the anchored row.  Hence

\[
 \mathcal K_{sil}
 :=\left(\bigoplus_i\mathcal D_i\right)\cap\ker L_B
 \subseteq\mathcal K_0,
\]

with the unconditional lower bound

\[
 \boxed{
 \dim\mathcal K_{sil}
 \ge1024\cdot867-524,800=363,008.}
\tag{23.13}
\]

Thus there is no quotient-dimension obstruction to placing all 561 pivots
inside banks that are already affine-K21-silent before any global
cancellation.  Equivalently, if `Pi_sil` is the orthogonal projection onto
`K_sil`, the finite Gram receipt

\[
 \boxed{
 \det\left[
 \left\langle
 \Pi_{sil}M_B^*\Lambda_{Z_a},
 \Pi_{sil}M_B^*\Lambda_{Z_b}
 \right\rangle
 \right]_{a,b=1}^{561}>0}
\tag{23.14}
\]

implies (23.12).

### 23.3 What the rank-one Jordan witness proves, and what remains

For the 33 invisible physical combinations, choose independent vectors
`v_a in R^62` with every `v_a^Tv_b != 0` and set

\[
 B_a=v_av_a^T.
\]

Then

\[
 B_aB_b+B_bB_a
 =(v_a^Tv_b)(v_av_b^T+v_bv_a^T),
\]

and the 561 matrices on the right are independent.  Hence the Jordan
catalecticant `Sym^2 R -> Sym_62` has exact rank 561.  Since `E_1=range Q_U^*`
has dimension 1,024, the 33 `B_a` can occupy the Veronese-visible block while
the 929-dimensional `E_0` block supplies the maximal first-order complement.
This proves that neither latent dimension, commutativity, parity, nor the
Veronese split forces any fixed Jordan combination to fifth order.

It does **not** by itself prove (23.12): the 561 Jordan rows could still meet
the coupled space `range F_U^*+L_B^*N_U`.  Equations (23.8) and (23.12)--
(23.14) identify that as the sole remaining compatibility syzygy.  No
universal quotient-dimension obstruction remains--the correction-free silent
space has dimension 363,008--but a literal D14 cut still needs one nonzero
minor or an analytic transversality construction.  Claiming generic
surjectivity from the Jordan core alone would skip exactly the seam issue
being audited.

### Graph node: R62_CUBIC_SCHUR_INTRINSIC_MINOR

- status: exact cubic operator and finite full-Schur minor / actual prefix
  receipt open;
- parity: even orders vanish, but no automatic fifth-order promotion;
- sole dark-mode condition: equation (23.8);
- correction-free test: a 561-row minor on `K_0`, dimension at least426,496;
- stronger typed test: `K_sil`, dimension at least363,008, whose members are
  individually Veronese-null and anchor-killing;
- Jordan result: a rank-561 core exists inside `Sym_62`, so only the coupled
  `F_U^*+L_B^*N_U` compatibility can still close the r62 cubic branch.

## 24. The r62 silent projection is a one-residual tilted wedge, not the raw Jordan core

The Gram test (23.14) can be reduced much further on an exact rank chart.  The
reduction also exposes an important correction: the rank-561 Jordan
catalecticant in Section 23.3 is **not** itself resident in the individually
anchored silent quotient.  Its pure visible part cancels identically there.

Write

\[
 q_i=u_i u_i^T=\mathcal Q_U^*e_i,
 \qquad E_1=\operatorname{range}\mathcal Q_U^*,
 \qquad E_0=\ker\mathcal Q_U,
\]

and use the explicit construction chart

\[
 B_i=q_i+N_i,
 \qquad N_i\in E_0.
\tag{24.1}
\]

Regard the rows `N_i` as an `n by e` matrix `N`, where
`e=dim E_0=929`.  Assume `N` has full column rank, its range is contained in
`K=ker U^T`, and

\[
 R=\ker N^T\cap K,
 \qquad \dim R=33.
\tag{24.2}
\]

For `Z in Sym R`, the multiplier in (23.6) is literal:

\[
 \boxed{\Lambda_Z=Z.}
\tag{24.3}
\]

Indeed `ZN=0`, while

\[
 (L_B^*Z)_p=\sum_j Z_{pj}q_j=\mathcal Q_U^*(Ze_p).
\]

The anchored term in `F_U^*Z` is zero because `ZU=0`, and its evaluation
term is exactly the same expression.

### 24.1 Exact form of the silent source

Let a bank `H_i in E_0` have coefficient matrix
`X in \mathbb R^{n\times e}`.
Since `E_0` is orthogonal to every `q_i`, the equation `L_BH=0` is

\[
 XN^T+NX^T=0.
\]

Full column rank of `N` implies, without a genericity assumption,

\[
 \boxed{X=NA,\qquad A^T=-A.}
\tag{24.4}
\]

The remaining individual-anchor equations are

\[
 \boxed{(NA)_i u_i=0\quad\text{for every }i.}
\tag{24.5}
\]

Thus `K_sil` is a subspace of the skew operators on `E_0`, cut by the
anchor map (24.5).  This description is exact; in particular no
`R tensor E_0` component survives `L_BH=0`.

Put

\[
 S_i(Z)=\mathcal Q_U^*(Ze_i)\in E_1.
\]

The direct cubic covector has the row form

\[
 (M_B^*Z)_i=\{B_i,S_i(Z)\}+\sum_k Z_{ki}B_k^2,
 \qquad \{X,Y\}=XY+YX.
\tag{24.6}
\]

Pair (24.6) with `H=NA`.  If `T_k=P_{E_0}(B_k^2)`, the second bank in
(24.6) is `ZT`; hence

\[
 N^T(ZT)=0.
\tag{24.7}
\]

It is therefore killed already by skew projection, before using an anchor.
The pure visible part of the first bank is also killed, now pointwise:

\[
 \langle H_i,\{q_i,S_i\}\rangle
 =2u_i^T H_iS_i u_i=0.
\tag{24.8}
\]

Only one residual occurrence remains.  In coefficient-free notation the
exact induced covector is

\[
 \boxed{
 \Phi_N(Z)
 =\operatorname{skew}\sum_i
 N_i\otimes P_{E_0}\{N_i,S_i(Z)\}
 \quad\bmod\operatorname{range}\mathcal T_{anc}^*,}
\tag{24.9}
\]

where

\[
 \mathcal T_{anc}(A)=\big((NA)_iu_i\big)_{i=1}^n.
\tag{24.10}
\]

Equivalently, for every skew `A` satisfying (24.5),

\[
 \langle M_B^*Z,NA\rangle
 =\langle\Phi_N(Z),A\rangle.
\tag{24.11}
\]

Equations (24.7)--(24.9) are universal on this chart.  They show why the
raw Jordan witness is insufficient: all `B_k^2` columns disappear by
`N^TZ=0`, and the zero-residual Jordan action disappears by the local
anchor.  The r62 cubic seam is governed by the **one-residual tilted
generator** (24.9).

### 24.2 Affine-second-difference triples give a 33-bank finite model

There is a lawful finite construction which realizes the desired 33
rank-one effective matrices without violating injectivity of
`Q_U^*`.  On three disjoint physical rows for each `a`, set

\[
 u_{a,+}=a_a+v_a,\qquad
 u_{a,-}=a_a-v_a,\qquad
 u_{a,0}=a_a,
\]

and

\[
 r_a=(1/2,1/2,-1)
\tag{24.12}
\]

on that triple.  Then

\[
 r_a^TU=0,
 \qquad
 \mathcal Q_U^*r_a=v_av_a^T=:M_a.
\tag{24.13}
\]

Unlike an antipodal pair, the three Veronese rows are generically
independent.  Choosing the `99` rank-one forms in (24.12) independently and
extending them by `925` further rank-one forms gives a rank-1024 evaluation
map.

Impose the matching residual rows

\[
 N_{a,+}=c_a+d_a,\qquad
 N_{a,-}=c_a-d_a,\qquad
 N_{a,0}=c_a,
 \qquad c_a,d_a\in E_0.
\tag{24.14}
\]

Then `r_a^TN=0`; the remaining rows may complete `N` to rank 929 in
`K intersect R^perp`.  Write `Z=(z_ab)` in the `r_a` coordinates and put

\[
 M(z_a)=\sum_bz_{ab}M_b.
\]

Substitution into (24.9) gives the exact cancellation

\[
\begin{aligned}
 &\frac12(c_a+d_a)\wedge
 P_0\{c_a+d_a,M\}
 +\frac12(c_a-d_a)\wedge
 P_0\{c_a-d_a,M\}\\
 &\hspace{35mm}-c_a\wedge P_0\{c_a,M\}
 =d_a\wedge P_0\{d_a,M\}.
\end{aligned}
\]

Consequently the whole 561-column question reduces to

\[
 \boxed{
 \Phi(Z)=\sum_{a=1}^{33}
 d_a\wedge P_{E_0}\{d_a,M(z_a)\}
 \quad\bmod\operatorname{range}\mathcal T_{anc}^*.}
\tag{24.15}
\]

No half-million-column Schur matrix is needed.  Before the anchor quotient,
the explicit columns are

\[
\begin{aligned}
 \phi_{aa}&=d_a\wedge P_0\{d_a,M_a\},\\
 \phi_{ab}&=d_a\wedge P_0\{d_a,M_b\}
             +d_b\wedge P_0\{d_b,M_a\},\qquad a<b.
\end{aligned}
\tag{24.16}
\]

The silent Gram is positive definite iff these 561 wedges are independent
in the quotient by `range T_anc^*`.

### 24.3 A concrete cross-block construction and the remaining minor

The dimensions in (24.15) admit a much more structured exact witness than a
generic bank.  Split the latent space as

\[
 \mathbb R^{62}=V\oplus T,
 \qquad \dim V=33,\quad\dim T=29.
\]

Choose

\[
 E_1=\operatorname{Sym}V\oplus\operatorname{Sym}T\oplus C_{28},
 \qquad C_{28}\subset V\odot T,
\tag{24.17}
\]

so

\[
 \dim E_1=561+435+28=1024,
 \qquad
 E_0=(V\odot T)\ominus C_{28},\quad\dim E_0=929.
\tag{24.18}
\]

This is compatible with a Veronese evaluation bank: rank-one forms with
vectors in `V` span `Sym V`, those in `T` span `Sym T`, and 28 mixed
rank-one forms add any chosen independent 28 cross directions modulo the
two block-diagonal spaces.  The triples (24.12) can be chosen inside `V`
and extended to such a bank.

Identify `d_a` with a cross-block map `D_a:V\to T`.  For
`M_b=v_bv_b^T`,

\[
 P_0\{d_a,M_b\}
 =P_0\big((D_av_b)v_b^T\big)
\tag{24.19}
\]

in the cross-block identification.  Thus (24.16) is an explicit finite
matching problem on 33 maps `D_a`, not a dimension heuristic.  One fully
symbolic sufficient certificate is:

1. choose 561 distinct wedge coordinates in `\Lambda^2 E_0` and order the
   pairs `(a,b)`;
2. require the coefficient of the assigned coordinate in `phi_ab` to be
   nonzero and all later assigned coefficients to vanish, giving a
   triangular 561 by 561 wedge minor;
3. require the span of those 561 wedge columns to meet
   `range T_anc^*` only at zero.

The first two requirements are polynomial conditions in the entries of the
`D_a` and the third is one ordinary stacked-rank minor.  This is the precise
Jordan/pseudoforest matching receipt which remains to be exhibited.  It is
strictly stronger than the raw fact that the matrices
`{M_aM_b+M_bM_a}` are independent.

There is a useful fail-fast warning.  If the `v_a` are chosen orthogonal and
span `V`, then `sum_a M_a=I_V`; taking every row of `Z` constant makes
`D_a wedge D_a I_V=0`.  Hence that convenient diagonal frame has an exact
one-dimensional wedge kernel.  Pairwise-nonorthogonal/generic `v_a` are not
cosmetic: they remove this particular scalar-right-action syzygy.

### 24.4 Exact Zariski-open statement, and the honest current verdict

Fix a nonzero-minor chart for `Q_U`, `N`, the maximal first Schur rank, and a
basis of `R`.  On this chart the basis changes, quotient projection, and
Gram entries are rational functions of `(U,N)`.  After multiplying by the
nonzero chart denominators,

\[
 \Delta_{sil}(U,N)
 =\det\operatorname{Gram}
 \left(\Pi_{K_{sil}}M_B^*\Lambda_{Z_{ab}}\right)
\tag{24.20}
\]

is a polynomial.  Therefore:

\[
 \boxed{
 \text{one exact configuration passing (24.15)--(24.16)}
 \Longrightarrow
 \Delta_{sil}\ne0\text{ on a Zariski-open dense subset of its component}.}
\tag{24.21}
\]

This proves that a single symbolic matching witness is sufficient for the
generic theorem.  It does **not** supply that witness by dimension alone.
The current rank-one Jordan construction proves no parity or latent-Jordan
no-go, but (24.7)--(24.9) show that its zero-residual interpretation was
overstated.  Generic positive-definiteness of the full anchored silent Gram
is still conditional on the finite wedge/anchor minor in (24.16); there is
not yet a lawful basis for calling it proved.

### Graph node: R62_ONE_RESIDUAL_TILTED_WEDGE_GATE

- status: exact quotient reduction / finite symbolic matching minor open;
- universal cancellation: all square-column terms vanish by `N^TZ=0`, and
  the pure visible Jordan term is killed by the individual anchor;
- exact survivor: the one-residual tilted wedge (24.9);
- 33-triple reduction: 561 columns reduce to (24.16) on 33 matrices `d_a`;
- structured witness space: the `33+29` cross-block model (24.17)--(24.19);
- correction to prior evidence: rank 561 of the raw Jordan catalecticant does
  not prove the silent Gram;
- genericity theorem: one nonzero wedge-plus-anchor minor implies a
  Zariski-open dense pass set, but that nonzero minor is the remaining gate.

## 25. An explicit common-action witness survives the complete anchor quotient

The remaining minor in Section 24 can in fact be opened symbolically.  The
construction needs two ingredients which are easy to miss: a rank-one
cross-block action, so all 33 tilted words have one common action, and
physical compensator rows, so the 66 residual directions are literally in
`ker U^T` without creating new anchor equations.

### 25.1 A 1024-row Veronese chart containing the required null forms

Split the latent space as

\[
 \mathbb R^{62}=V\oplus C_0\oplus T,
 \qquad \dim V=33,\quad\dim C_0=9,\quad\dim T=20.
\tag{25.1}
\]

Choose a basis `v_1,...,v_33` of `V` with dense positive-definite Gram
matrix

\[
 H_{ab}=v_a^Tv_b,
\]

and a linear functional `\ell\in V^*` for which

\[
 \beta_a=\ell(v_a)\ne0
\tag{25.2}
\]

are not all equal in absolute value.  Fix `t_0 in C_0` and define the
rank-one cross block

\[
 D_0=t_0\odot\ell.
\tag{25.3}
\]

Choose 33 independent off-diagonal forms
`E_a in Sym C_0`; this is possible because the off-diagonal space has
dimension 36.  Put

\[
 d_a=D_0+E_a,\qquad
 M_a=v_av_a^T,\qquad
 c_a=\{D_0,M_a\}.
\tag{25.4}
\]

Every `E_a` kills `V`, so

\[
 \boxed{\{d_a,M_b\}=c_b\quad\text{for every }a,b.}
\tag{25.5}
\]

Up to a nonzero convention scalar, `c_b=t_0\odot(\beta_bv_b)`; hence the
33 `c_b` are independent.  The 66 forms `{d_a,c_b}` are independent:
the `C_0`-off-diagonal component first kills every coefficient of the
`d_a`, after which independence of the `c_b` kills the rest.

There are enough real Veronese points annihilating all these forms.  Take
1015 independent rank-one forms with vectors in the 53-dimensional space
`V\oplus T`, and nine more at the coordinate axes of `C_0`.  Their span
has dimension 1024, while their vectors span all 62 latent directions.
Every such vector is isotropic for `D_0`, every `E_a`, and every `c_b`.
Thus

\[
 \operatorname{rank}\mathcal Q_U=1024,\qquad
 d_a,c_b\in E_0=\ker\mathcal Q_U,\qquad\dim E_0=929.
\tag{25.6}
\]

The 99 triple rows (24.12) lie in `V` and may be included among the 1015
rows.  Their rank-one forms can be chosen independent and extended inside
`\operatorname{Sym}(V\oplus T)`, so (24.13) gives the desired `M_a\in E_1` without an
antipodal duplicate.

The rows can simultaneously satisfy the affine-K21 minors from Section 17.1.
For a pure `C_0` axis, the other 1023 forms have zero intersection with
`e_k\odot\mathbb R^{62}` because the missing cross and off-diagonal
`C_0` blocks are not in `E_1`.  For a row in `V\oplus T`, the condition is
that its 1014 companions in `E_{1,S}` avoid the 53-dimensional space
`u_i\odot(V\oplus T)`.  Since `1014+53<1431`, this is a nonempty finite
Veronese-minor open condition.  Choose the 1015 points in that open set.
Then `F_U` is onto as well as `\mathcal Q_U`.

### 25.2 Literal physical completion with no hidden anchor rows

For each `a`, choose auxiliary vectors `t_a,s_a\in T` so that all prescribed
rank-one rows below remain independent.  In addition to the three rows in
(24.12), include

\[
 a_a+t_a,\quad t_a,\qquad v_a+s_a,\quad s_a.
\tag{25.7}
\]

Assign the physical coefficient vector for `c_a` the values

\[
 (1,1,1;-3,3;0,0)
\]

on the triple, the first auxiliary pair, and the second pair.  Assign the
coefficient vector for `d_a` the values

\[
 (1,-1,0;0,0;-2,2).
\]

Both vectors are in `ker U^T`, because

\[
 3a_a-3(a_a+t_a)+3t_a=0,\qquad
 2v_a-2(v_a+s_a)+2s_a=0.
\tag{25.8}
\]

They are also orthogonal to every `r_b`; on their own triple the dot
products are

\[
 (1/2,1/2,-1)\cdot(1,1,1)=0,\qquad
 (1/2,1/2,-1)\cdot(1,-1,0)=0.
\]

The 66 physical vectors are independent by their disjoint triple pivots.
Map the 66-dimensional span of `{c_a,d_a}` to these vectors and complete
arbitrarily to an isomorphism

\[
 N:E_0\longrightarrow K\cap R^\perp.
\tag{25.9}
\]

The dimensions match exactly: the completion has `929-66=863` directions.
Choose it block diagonally with respect to the displayed 66-dimensional
subspace.  Then a skew source supported on `span{c_a,d_a}` is zero on every
unlisted physical row.  Moreover all `c_a,d_a` kill `T`, so the auxiliary
anchors in (25.7) repeat, rather than enlarge, the anchors at `a_a` and
`v_a`.

### 25.3 The tilted map contains the full coefficient matrix

Substituting (25.5) into (24.15) gives

\[
 \boxed{\Phi(Z)=\sum_{a,b}z_{ab}\,d_a\wedge c_b.}
\tag{25.10}
\]

Because the two 33-dimensional spans are disjoint, (25.10) is injective
even on all `33 by 33` coefficient matrices before anchors.  It remains to
show that no nonzero symmetric coefficient matrix belongs to the anchor
adjoint.

Work in the biorthogonal coefficient bases of
`D=\operatorname{span}\{d_a\}` and `C=\operatorname{span}\{c_b\}`.  On a vector `x\in V`, every form in
these bases acts in the single latent direction `t_0`:

\[
 d_p x=\ell(x)t_0,\qquad
 c_bx=\beta_b(v_b^Tx)t_0.
\tag{25.11}
\]

Define the two `33 by 33` matrices

\[
 G_{ab}=\beta_bv_b^Tv_a=H_{ab}\beta_b,\qquad
 A_{ab}=\beta_bv_b^Ta_a.
\tag{25.12}
\]

For one triple, if `X_a=c_aA` and `Y_a=d_aA`, the three exact anchor
equations are

\[
 X_aa_a=0,\qquad Y_av_a=0,\qquad X_av_a+Y_aa_a=0.
\tag{25.13}
\]

The auxiliary rows add only scalar multiples of the first two equations.
The projections of the 99 anchor-adjoint generators to
`\Lambda^2(D\oplus C)` are therefore

\[
\begin{aligned}
 F^1_a&=c_a\wedge s(a_a),\\
 F^2_a&=d_a\wedge s(v_a),\\
 F^3_a&=c_a\wedge s(v_a)+d_a\wedge s(a_a),
\end{aligned}
\tag{25.14}
\]

where the `D` coordinates of `s(x)` are the constant vector
`\ell(x)\mathbf 1` and its `C` coordinates are
`(beta_b v_b^T x)_b`.

Suppose (25.10) is a combination of (25.14), with multipliers
`\lambda,\mu,\nu\in\mathbb R^{33}`.  Its `\Lambda^2 C` block first requires

\[
 \boxed{
 \operatorname{skew}\big(\operatorname{diag}(\lambda)A
 +\operatorname{diag}(\nu)G\big)=0.}
\tag{25.15}
\]

Choose the still-free vectors `a_a` so the map in (25.15) has rank 65 and
kernel

\[
 \ker(25.15)=\operatorname{span}\{(0,\beta)\}.
\tag{25.16}
\]

This has a finite symbolic pseudoforest certificate.  On a 32-edge
spanning tree, set the corresponding two orientations of `A` to zero.
The `G` equations then have rank 32 and force `\nu_a/\beta_a` to be constant.
On an edge-disjoint connected unicyclic graph with 33 edges, choose the
oriented `A` weights so their weighted incidence determinant is nonzero.
Those rows give an invertible 33-column `lambda` block.  Since the map

\[
 x\longmapsto(\beta_bv_b^Tx)_b
\]

is invertible on `V`, these prescribed rows of `A` are realized by actual
vectors `a_a`; unused entries remain available for the Veronese-independence
minor.

Under (25.16), `\lambda=0` and `\nu=c\beta`.  Vanishing of the
`\Lambda^2 D` block then gives

\[
 \mu_a={s\over\beta_a}-c\ell(a_a)
\tag{25.17}
\]

for two scalars `s,c`.  Finally the `D\wedge C` coefficient matrix is

\[
\begin{aligned}
 Z(s,c)
 &=s\,\operatorname{diag}(\beta^{-1})G\\
 &\quad+c\left[-\operatorname{diag}(\ell(a))G
 +\operatorname{diag}(\beta)A
 -\mathbf1(\beta\odot\beta)^T\right].
\end{aligned}
\tag{25.18}
\]

Choose the nonzero `beta_a` and the unused entries of `A` so the skew
parts of the two bracketed matrices in (25.18) are linearly independent.
This is one explicit `2 by 2` polynomial minor.  The first skew part is
already nonzero whenever `H` is connected through nonzero entries and the
`|\beta_a|` are not all equal.  Hence `Z(s,c)` is symmetric only for
`s=c=0`.

Equations (25.15)--(25.18) prove

\[
 \boxed{
 \operatorname{range}\Phi|_{\operatorname{Sym}_{33}}
 \cap\operatorname{range}\mathcal T_{anc}^*=\{0\}.}
\tag{25.19}
\]

Completion rows in (25.9) have first wedge factor outside `D\oplus C`, so
they cannot change the `\Lambda^2(D\oplus C)` comparison.  Thus (25.19) is
the complete anchor quotient, not merely the 99-row restriction.

### 25.4 Consequence and scope

The common-action construction supplies the missing nonzero determinant in
(24.20).  Therefore the 561 projected silent vectors are independent at
one exact symbolic r62 configuration and, by (24.21), on a Zariski-open
dense subset of the corresponding maximal-rank component.

This is an **existence/genericity theorem**, not yet a D14 seam receipt.
A proposed prefix must still pass the maximal first-Schur minor, remain on
the same real component with acceptable conditioning, and acquire the
oriented AQ state within the cost cap.  What is now ruled out is a universal
representation-theoretic promotion of any of the 561 modes to fifth order.

### Graph node: R62_TILTED_WEDGE_GENERIC_OPEN

- status: symbolic existence theorem / actual D14 cut and conditioning open;
- Veronese chart: `V33+C0_9+T20`, with 1024 independent legal rank-one rows;
- residual bank: 33 common-action pairs `d_a=D0+E_a`,
  `c_b={D0,M_b}` plus literal physical compensators;
- cubic map: `Phi(Z)=sum z_ab d_a wedge c_b`, injective before anchors;
- anchor receipt: rank-65 tree/unicyclic incidence minor plus the two-skew
  independence minor (25.18);
- theorem: the 561-dimensional silent Gram is positive definite on a
  nonempty Zariski-open set;
- remaining deployment gate: maximal first Schur rank, real conditioning,
  lawful seam acquisition, and the 47.43295 cost premise.

## 26. Supersession: affine rank-one triples have 33 exact cubic-dark loop modes

Section 25's claimed anchor pass is false.  The error is instructive and
must be retained: the biorthogonal coefficient calculation introduced an
artificial `Lambda^2 D` constraint, whereas the actual Euclidean anchor
Riesz vector for a rank-one effective matrix lies in the same cross block
as the tilted word.

For the affine-second-difference triple, the diagonal target
`Z=r_ar_a^T` has

\[
 M_a=v_av_a^T
\]

and its tilted column from (24.16) is

\[
 \phi_{aa}=d_a\wedge P_0\{d_a,v_av_a^T\}.
\tag{26.1}
\]

But

\[
 \{d_a,v_av_a^T\}
 =(d_av_a)v_a^T+v_a(d_av_a)^T
\tag{26.2}
\]

is exactly the Riesz form of the local anchor equation `Y_av_a=0` from
(25.13).  Therefore

\[
 \boxed{\phi_{aa}\in\operatorname{range}\mathcal T_{anc}^*,
 \qquad a=1,\ldots,33.}
\tag{26.3}
\]

Within the matched affine-second-difference realization, this conclusion is independent of the choice of `d_a`, the common-action
ansatz, and the auxiliary physical completion.  The 33 diagonal basis
vectors of `Sym R` are killed exactly, so every such affine rank-one-triple silent Gram
satisfies

\[
\boxed{\operatorname{rank}\operatorname{Gram}_{sil}\le561-33=528.}
\tag{26.4}
\]

This is not a theorem about every non-affine realization of a rank-one
`M_a`.  For a general coefficient vector `r` with
`Q_U^*r=vv^T`, the diagonal pairing is

\[
 2\sum_i r_i\,v^T(N_iA)N_iv.
\tag{26.5a}
\]

The local anchors give `(N_iA)u_i=0`, not `(N_iA)v=0`; hence (26.5a)
need not vanish.  The one-line affine-triple proof works only because its
three anchor equations imply `(d_aA)v_a=0`.  A non-affine coefficient
second-moment construction whose residual directions are not aligned with
that same `v_a` is a genuine separate escape and must be judged by (24.9),
not by (26.3).

Thus the graph node `R62_TILTED_WEDGE_GENERIC_OPEN` in Section 25 is
**invalidated for its declared affine-triple construction**.  The tree/unicyclic argument may still organize the 528
off-diagonal modes, but it cannot certify that full r62 cubic seam.  The
Zariski-open implication (24.21) remains correct; Section 25 did not supply
its required nonzero determinant.

### 26.1 Minimal rank-two replacement inside the finite-difference family

The loop obstruction is specific to a rank-one effective `M_a`.  A lawful
rank-two effective direction is generated by four rows

\[
 u_{a,00}=a_a,\quad
 u_{a,10}=a_a+v_a,\quad
 u_{a,01}=a_a+w_a,\quad
 u_{a,11}=a_a+v_a+w_a
\]

with physical coefficient vector

\[
 r_a=(1,-1,-1,1).
\tag{26.5}
\]

Then

\[
 r_a^TU=0,\qquad
 \mathcal Q_U^*r_a=v_aw_a^T+w_av_a^T=:M_a,
\tag{26.6}
\]

which is generically rank two.  Use residual rows

\[
 N_{00}=c_a,\quad N_{10}=c_a+d_a,\quad
 N_{01}=c_a+e_a,\quad N_{11}=c_a+d_a+e_a.
\tag{26.7}
\]

They obey `r_a^TN=0`.  Since the map
`n mapsto n wedge P_0{n,M}` is quadratic, its mixed finite difference is
exact:

\[
\boxed{
 \Phi_2(Z)=\sum_a\left[
 d_a\wedge P_0\{e_a,M(z_a)\}
 +e_a\wedge P_0\{d_a,M(z_a)\}\right],
 \qquad M(z_a)=\sum_bz_{ab}M_b.}
\tag{26.8}
\]

For a loop, each anticommutator now splits into two distinct Riesz
directions:

\[
 \{d_a,M_a\}
 =v_a\odot(d_aw_a)+w_a\odot(d_av_a),
\tag{26.9}
\]

and similarly for `e_a`.  The feature in (26.8) also crosses the first
residual factors `d_a,e_a`.  Thus it is no longer the single local-anchor
covector in (26.2); whether a combination of all four corner anchors
removes it is precisely the new finite minor, not a universal identity.
Hence (26.3) no longer gives a universal loop identity.  A common-action
choice for `d_a,e_a` reduces the next exact question to a `66 by 66`
bipartite wedge/anchor matching problem.  This is the smallest explicit
finite-difference route currently known which can lift all 33 loop modes at cubic order.

### Graph node: R62_AFFINE_RANKONE_LOOP_NOGO_NONAFFINE_OPEN

- status: exact no-go for the affine rank-one witness / non-affine and rank-two minors open;
- exact loss under (24.12)--(24.14): all 33 diagonal `Sym R` modes are local-anchor covectors, so
  the rank-one silent Gram has rank at most 528;
- invalidated claim: `R62_TILTED_WEDGE_GENERIC_OPEN` from Section 25;
- finite-difference escape: four-point mixed difference (26.5)--(26.8), producing
  rank-two `M_a` and two residual directions per invisible mode;
- next gate: prove the 561-column rank of (26.8) modulo all four-point
  anchors and a maximal first-Schur chart.

## 27. Audit of the overlapping 34-group construction: the local 561 minor is exact, but the declared normal chart is not yet typed

The construction in `LANE_r62_full561_overlapping_anchor_design.md` does
solve the representation-theoretic part of the problem.  Its two 561-element
families are genuine bases:

\[
 \{p_a^T\Theta p_b:0\le a<b\le33\}
 \quad\hbox{is a basis of }\operatorname{Sym}(\mathbb R^{33})^*,
\]

and `\{S(v_a)\wedge S(v_b):a<b\}` is a basis of `\bigwedge^2X`.  The
calculation

\[
 P_{\wedge^2X}\Phi_N(P\Theta P^T)
 =6\sum_{a<b}(D_a-D_b)(v_a^Tv_b)
 (p_a^T\Theta p_b)S(v_a)\wedge S(v_b)
\tag{27.1}
\]

is consequently an isomorphism.  The detector typing is also sound after
identifying `X\simeq V`: a detector is `A\in\mathfrak{so}(V)`, acting on the
coefficient block `X=V\odot C`; at an active row it sends
`S(\beta v)` to `S(\beta Av)`, whose action on
`u=(\alpha v,0,z)` vanishes because `v^TAv=0`.  Thus the displayed local
anchor quotient really has rank 561.  This is an exact counterexample to a
*universal* cubic anchor or Jordan syzygy, and it leaves the disjoint-affine
loop NOGO of Section 26 intact.

There is, however, a binding promotion gap relative to the explicit normal
chart assumed in (24.1)--(24.2).  That chart requires

\[
 \boxed{\operatorname{range}N\subseteq K=\ker U^T,
 \quad\text{i.e. }U^TN=0.}
\tag{27.2}
\]

The overlapping completion proves instead that `[U\ N]` has rank 991 and
that `\ker U^T\cap\ker N^T=\operatorname{range}P`; these facts do not imply
(27.2).  In fact its specified active `X` bank violates (27.2) literally.
With

\[
 \alpha=(1,-2,1),\qquad
 \beta_a=(a,-2(a+1),a+2),\qquad
 z_a=(0,w_a,-w_a),
\]

one has

\[
 \sum_t\alpha_t\beta_{a,t}=6(a+1),
 \qquad
 \sum_t\beta_{a,t}z_{a,t}=-(3a+4)w_a.
\tag{27.3}
\]

Hence, for an `X\simeq V` coefficient `h`, the active contribution is

\[
 (U^TN_X)h
 =\sum_{a=0}^{33}
 \bigl(6(a+1)v_a-(3a+4)w_a\bigr)(v_a^Th),
\tag{27.4}
\]

up to the fixed harmless normalization of `S`.  Its `V` block
`\sum_a6(a+1)v_av_a^T` is positive definite because the `v_a` form a basis,
so (27.4) is certainly nonzero.

Orthogonally shearing the bank to

\[
 \widetilde N=P_KN
 =N-U(U^TU)^{-1}U^TN
\tag{27.5}
\]

does preserve full column rank and the residual space `R=\operatorname{range}P`
(because `[U\ N]` is full rank and `P\subset K` is orthogonal to `N`).  It
does **not** visibly preserve the rowwise alignment
`(n_{a,t})_X=S(\beta_{a,t}v_a)` which made all of
`\mathfrak{so}(X)` anchor-silent, nor therefore the diagonal minor (27.1).
Calling (27.5) a free gauge fix would require a proved invariance of the
intrinsic cubic Schur covector under that shear; no such invariance appears
in the current derivation.

Two further promotion premises remain separate.  The construction proves
`Q_U` onto, but not the stronger `F_U`-onto minor; and it does not prove that
the first Schur map has maximal rank with cokernel exactly `\operatorname{Sym}R`.
Those are open algebraic conditions, not consequences of the local
determinant (27.1).

The precise final verdict is therefore:

\[
 \boxed{
 \begin{array}{l}
 \text{full rank 561 for the displayed local anchored tilted map: PROVED;}\\
 \text{universal cubic/anchor rank-loss identity: REFUTED;}\\
 \text{lawful full r62 cubic Schur witness on the normal chart: OPEN.}
 \end{array}}
\tag{27.6}
\]

The smallest exact repair certificate is either (i) a proof that the
`U^TN=0` premise can be removed by a Schur-gauge invariance which transports
the detector, or (ii) a modified 34-group completion satisfying (27.2)
while retaining an anchor-silent 561-dimensional detector and the minor
(27.1).  Only after that repair, plus explicit `F_U` and maximal-`S_1`
minors, does the Zariski-open argument (24.21) promote the local theorem to
the complete cubic seam.

### Graph node: R62_OVERLAPPING_FULL561_LOCAL_PASS_NORMAL_TYPING_OPEN

- status: exact local anchor-quotient pass / full normal-Schur promotion open;
- exact pass: the 34 overlapping groups give the nonzero 561 determinant
  (27.1), so no universal cubic Jordan/anchor syzygy exists;
- exact typing failure: the displayed active bank has `U^TN_X\ne0`, with
  receipt (27.3)--(27.4), while the Section 24 normal chart assumes
  `range N\subset\ker U^T`;
- unsafe shortcut: `N\mapsto P_KN` preserves rank and `R` but has no proved
  preservation of the automatic `so(34)` anchor detector or tilted minor;
- additional gates: explicit `F_U` surjectivity and maximal first-Schur rank;
- supersession boundary: Section 26 remains exact only for disjoint affine
  triples; the overlapping construction refutes its universal extension but
  does not yet certify a lawful deployed r62 seam.

# H6/H8, latent/external quartic bridge, and fail-fast bounds

Theory-only lane.  No numerical experiment was used in deriving these results.
All Hermites below are probabilists' Hermites and the He-normalized activation is
\(\psi(z)=\sqrt2\,z_+\).

## 1. Scalar ReLU Hermite constants

For \(k\ge1\),

\[
a_{2k}=\frac{(-1)^{k-1}(2k-3)!!}{\sqrt\pi(2k)!},
\qquad
e_{2k}=(2k)!a_{2k}^2
=\frac{((2k-3)!!)^2}{\pi(2k)!}.
\]

In particular,

\[
e_6=\frac1{80\pi},\qquad e_8=\frac5{896\pi}.
\]

The exact normalized ReLU kernel is

\[
\kappa(\rho)
=\frac{\sqrt{1-\rho^2}+(\pi-\arccos\rho)\rho}{\pi}.
\]

The native even tail after the scalar H4 term is

\[
T_{\ge6}(\rho)
=\frac1\pi\left[
\sqrt{1-\rho^2}+\rho\arcsin\rho
-1-\frac{\rho^2}{2}-\frac{\rho^4}{24}
\right].
\]

For \(|\rho|\le r<1\),

\[
\frac{r^6}{80\pi}
\le T_{\ge6}(r)
\le
\frac{r^6}{80\pi}
\left(1+\frac{25r^2}{56(1-r^2)}\right).
\]

The native scalar-kernel H6 is normally already inside any exact Gaussian ReLU
kernel.  It must not be confused with the covariance-plane H6 lift below.

## 2. Correct covariance H4/H6 normal ordering

Define

\[
Q_M=\frac12(z^TMz-\operatorname{tr}M)=\frac12I_2(M),
\]

and

\[
U_{A,B}
=Q_AQ_B-Q_{AB+BA}-\frac12\operatorname{tr}(AB)
=\frac14I_4(\operatorname{sym}A\otimes B).
\]

Thus \(U_M=U_{M,M}=:Q_M^2:\).  The pure covariance-generated H6 is

\[
V_M=\frac18I_6(\operatorname{sym}M^{\otimes3}).
\]

The **correct general-matrix identity** is

\[
\boxed{
V_M
=Q_M^3-6Q_MQ_{M^2}+6Q_{M^3}
-\frac32\operatorname{tr}(M^2)Q_M
+2\operatorname{tr}(M^3).
}
\]

Equivalently,

\[
Q_M^3
=V_M+6U_{M^2,M}+6Q_{M^3}
+\frac32\operatorname{tr}(M^2)Q_M
+\operatorname{tr}(M^3).
\]

An earlier scratch derivation used coefficients that are valid only after the
rank-one collapse \(Q_{M^3}=\operatorname{tr}(M^2)Q_M\).  The boxed identity
above supersedes that scratch formula everywhere.

Its exact energy is

\[
\mathbb E V_M^2
=\frac34\left[
(\operatorname{tr}M^2)^3
+6\operatorname{tr}(M^2)\operatorname{tr}(M^4)
+8\operatorname{tr}(M^6)
\right].
\]

For the Gaussian covariance tilt \(I+\epsilon M\), the density expansion is

\[
L_\epsilon
=1+\epsilon Q_M+\frac{\epsilon^2}{2}U_M
+\frac{\epsilon^3}{6}V_M+\cdots.
\]

Therefore the H6 density energy is

\[
\frac{\epsilon^6}{48}
\left[t_2^3+6t_2t_4+8t_6\right].
\]

The exact entire covariance-density tail is controlled by

\[
\mathbb E L_\epsilon^2
=\det(I-\epsilon^2M^2)^{-1/2}.
\]

After H4, subtract

\[
1+\frac{\epsilon^2}{2}t_2
+\frac{\epsilon^4}{8}(t_2^2+2t_4)
\]

to obtain the exact H6-and-higher source energy.

## 3. H5 mixed quadratic/cubic source

For a symmetric cubic tensor \(T\), let

\[
C_T=\frac16I_3(T).
\]

The exact normal-ordered mixed H5 source is

\[
\boxed{
W_{M,T}=Q_M\diamond C_T
=Q_MC_T-3C_{\operatorname{sym}(M\cdot_1T)}
-\frac12(M\cdot_2T)\cdot z
=\frac1{12}I_5(\operatorname{sym}M\otimes T).
}
\]

The subtracted H3 term overlaps CP3/Q24; the H1 term overlaps the linear/mean
path.  Only the normal-ordered H5 remainder can receive new source credit.

At an independent whitened pair boundary, put

\[
B_{ij}=u_i^TMu_j,\quad d_i=B_{ii},\quad
c_i=T[u_i,u_i,u_i],\quad P_{ij}=T[u_i,u_j,u_j].
\]

Then the exact pair action is

\[
\boxed{
K^{(5)}
=-\frac1{6\sqrt{2\pi}}B\odot(c\mathbf1^T+\mathbf1c^T)
-\frac1{4\sqrt{2\pi}}
\left[P\odot(\mathbf1d^T)+P^T\odot(d\mathbf1^T)\right].
}
\]

This term is already present iff the CP3/Q24 response is evaluated at the
corrected covariance; it is the covariance derivative of that cubic response.

## 4. Pair-boundary H6 and its sharp zero gate

For the actual density coefficient \(V_M/6\), the raw pair second-moment term is

\[
\frac1{16\pi}\left[
d^{\odot3}\mathbf1^T+\mathbf1(d^{\odot3})^T
-d(d^{\odot2})^T-(d^{\odot2})d^T
-4B^{\odot2}\odot(d\mathbf1^T+\mathbf1d^T)
\right].
\]

After consistently subtracting the perturbed means, the centered term for
off-diagonal pairs collapses to

\[
\boxed{
K_{\rm cov}^{(6)}
=-\frac1{4\pi}B^{\odot2}\odot(d\mathbf1^T+\mathbf1d^T).
}
\]

Hence, at a zero-correlation/independent base,

\[
\boxed{\operatorname{diag}B=0\Longrightarrow K_{\rm cov}^{(6)}=0.}
\]

This statement is local to expansion about a zero pair correlation.  Around a
nonzero base correlation, the exact derivative
\(D^3F_C[M,M,M]/6\) should be used and need not vanish.

After residualizing against an existing span \(X\), the H6 Schur energy obeys

\[
g_{6\mid X}
=g_{66}-g_{6X}G_{XX}^{+}g_{X6}
\le
\left[
\frac1{4\pi}\sum_l\beta_l
\left(\sum_{ij}B_{l,ij}^4(d_{l,i}+d_{l,j})^2\right)^{1/2}
\right]^2.
\]

It is exactly zero when every \(d_l\) is zero.

## 5. Fish overlap

The normal-ordered cubic fish is

\[
F_T=\frac12C_T\diamond C_T
=\frac1{72}I_6(\operatorname{sym}T\otimes T).
\]

Although diagrammatically distinct, \(F_T\) and \(V_M\) are both H6 and need
not be orthogonal.  With

\[
v_a=\sum_{bc}M_{bc}T_{bca},
\]

their exact source overlap is

\[
\boxed{
\mathbb E[V_MF_T]
=\frac12\langle T,M^{\otimes3}T\rangle
+\frac34v^TMv.
}
\]

For \(T\propto a^{\otimes3}\) and \(M\propto aa^T\), they are exactly
collinear.  Every covariance H6 claim must first be residualized against the
existing fish/Q24 span.

## 6. Correct H8 identity and correlation-only tail

Define

\[
Y_M=\frac1{16}I_8(\operatorname{sym}M^{\otimes4}).
\]

The exact normal-order identity is

\[
\boxed{
\begin{aligned}
Y_M={}&Q_M^4-12V_{M^2,M,M}-24U_{M^3,M}-12U_{M^2,M^2}
-3t_2U_M\\
&-24Q_{M^4}-6t_2Q_{M^2}-4t_3Q_M
-3t_4-\frac34t_2^2.
\end{aligned}
}
\]

Equivalently,

\[
Y_M=Q_MV_M-6V_{M^2,M,M}-6U_{M^3,M}-\frac32t_2U_M.
\]

The density coefficient is \(Y_M/24\), and

\[
\mathbb E Y_M^2
=\frac32\left[t_2^4+12t_2^2t_4+12t_4^2+32t_2t_6+48t_8\right].
\]

For a correlation-only perturbation at an independent base,

\[
K_{\rm cov}^{(8)}=\frac1{24\pi}B^{\odot4}.
\]

The exact entire post-H4 centered tail is

\[
\boxed{
T_{8+}(b)
=\frac1\pi\left[
\sqrt{1-b^2}+b\arcsin b-1-\frac{b^2}{2}
\right].
}
\]

It equals

\[
\frac{b^4}{24\pi}+\frac{b^6}{80\pi}
+\frac{5b^8}{896\pi}+\cdots.
\]

There is no total-chaos H10 term in this cross-only setting: five cross edges
would produce H5 on both endpoints and ReLU's H5 coefficient is zero.

Relative to the carried centered H4 term \(b^2/(2\pi)\), define

\[
A(r)=\frac{2[\sqrt{1-r^2}+r\arcsin r-1-r^2/2]}{r^2}.
\]

Then \(A\) is increasing and

\[
A(1)=\pi-3,\qquad
E_{8+}\le(\pi-3)^2E_4\approx0.0200485E_4.
\]

Using the optimistic ideal-k4 task energy \(1.0313\times10^{-9}\), the full
tail can contribute at most \(2.07\times10^{-11}\) under the same transport.
The exact radius for it to cover the razor-thin \(5.51\times10^{-12}\) deficit
is about \(0.821\).  No radius can cover \(0.7453\times10^{-9}\).  H8+ is
therefore closed as a standalone lane and retained only as a zero-GEMM exact
resummation completeness term when a corrected covariance is already present.

## 7. Unequal component-fidelity simplex

The exact conditional innovation energies, in units of \(10^{-9}\), are

\[
I_v=17.316,\quad I_3=4.370,\quad I_4=1.0313,\quad I_R=3.005447.
\]

For conditional endpoint fidelities \(q_v,q_3,q_4\), define

\[
L_{\rm low}
=17.316(1-q_v^2)+4.370(1-q_3^2)+1.0313(1-q_4^2).
\]

Higher-order capture is unnecessary for raw \(<10^{-8}\) iff

\[
L_{\rm low}\le0.7453.
\]

If \(q_v^2=0.96695\) and q3/q4 are exact, raw is
\(9.8270\times10^{-9}\), so H6+ is not mandatory.  If all three squared
fidelities are \(0.96695\), the missing gain is only
\(5.51\times10^{-12}\).

## 8. Exact latent-Hermite H4 partition energies

For \(U_M=:Q_M^2:\), the five equality partitions are orthogonal and have
energies

\[
E_4=\frac32\sum_iM_{ii}^4,
\]

\[
E_{31}=6\sum_{i\ne j}M_{ii}^2M_{ij}^2,
\]

\[
E_{22}=4\sum_{i<j}\left(\frac12M_{ii}M_{jj}+M_{ij}^2\right)^2,
\]

\[
E_{211}=2\sum_i\sum_{\substack{j<k\\j,k\ne i}}
\left(M_{ii}M_{jk}+2M_{ij}M_{ik}\right)^2,
\]

\[
E_{1111}=4\sum_{i<j<k<l}
\left(M_{ij}M_{kl}+M_{ik}M_{jl}+M_{il}M_{jk}\right)^2.
\]

They sum to

\[
\frac12[(\operatorname{tr}M^2)^2+2\operatorname{tr}M^4].
\]

The complete external NCQC K31/211 witness mixes all five latent partitions.
The endpoint old-span projector does not commute with the latent partition
projectors, so external-211 must never be equated with latent-211.

## 9. Label-free latent/external bridge certificate

Let

\[
z=R_0q_{4,z},\qquad f=R_0q_{\rm ext,full},
\qquad q_{\rm ext,full}=48\sum_a\operatorname{tr}(N_a^3R_a).
\]

Put

\[
a=\|z\|^2,\qquad b=\|f\|^2,\qquad c=\langle z,f\rangle.
\]

The exact label-free energy of the latent witness outside the external
direction is

\[
\boxed{B_z=a-c^2/b}\qquad (b>0).
\]

The external route needs \(27.732\%\) of the ideal quartic energy, equivalently

\[
|\operatorname{corr}(y_4,f)|>\sqrt{0.27732}\approx0.52661.
\]

If \(y_4=z\), this is exactly

\[
B_z<0.72268\,a.
\]

For a bridge mismatch \(h=y_4-z\) with
\(\|h\|^2\le B_{\rm bridge}\),

\[
\boxed{
\|P_{f^\perp}y_4\|^2
\le(\sqrt{B_z}+\sqrt{B_{\rm bridge}})^2.
}
\]

Since \(I_4=1.0313\times10^{-9}\), a sufficient certificate is

\[
\boxed{
(\sqrt{B_z}+\sqrt{B_{\rm bridge}})^2
<0.7453\times10^{-9}.
}
\]

Without a bridge assumption, one external direction has no positive
label-free worst-case gain over a residual quartic space of dimension greater
than one: choose \(y_4\perp f\).

## 10. Current bridge question and strongest analytic form

Let \(\mathcal O_4\) be the complete carried degree-4 observation map
(D11/Q24/CP3, q4_z, complete external K31, and old-span quotient data), and let
\(\mathcal V_4\) be the exact degree-4 source-to-endpoint value map.  Functional
completeness gives the source split

\[
C_4=C_{4,z}+C_{4,\perp},
\qquad C_{4,\perp}\in\ker\mathcal O_4.
\]

At a single scalar ReLU response, the H4 coefficient energy is
\(e_4=1/(24\pi)\), hence

\[
|\Delta\psi_4|^2\le\frac1{24\pi}\|C_{4,\perp}\|_{L_2}^2.
\]

For ten layers, the strongest generic analytic bridge bound has the form

\[
\boxed{
B_{\rm bridge}
\le
\left[
\sum_{l=5}^{14}
\beta_l\sqrt{E_{4,\perp,l}/(24\pi)}
+\sum_{l=5}^{14}\beta_l\sqrt{E_{>4,l}^{\rm response}}
\right]^2.
}
\]

The second sum can be bounded by the exact ReLU Hermite tails above.  The first
sum cannot currently be bounded from the carried state: no bound on the hidden
primitive-quartic energies \(E_{4,\perp,l}\) or on their target pullbacks has
been established.  The endpoint `rest' capacity \(3.005447\times10^{-9}\) is
already larger than the allowed \(0.7453\times10^{-9}\), so the current coarse
rest bound cannot prove the desired bridge inequality.

The precise identifiability condition is

\[
\boxed{
\ker\mathcal O_4\subseteq\ker\mathcal V_4.
}
\]

If it fails, choose \(H\in\ker\mathcal O_4\) with
\(\mathcal V_4H\ne0\).  Two sufficiently small admissible degree-4
perturbations with opposite hidden component have identical carried
observations but different endpoint quartic values.  Therefore no deterministic
bound below the goal threshold can be proved from the current state alone.

The least additional information is still **order four**, not H6/H8:

* target-specific: carry the scalar contractions
  \(\langle\Lambda_l^{(4)},C_{4,\perp,l}\rangle\) with the reverse H4 adjoint;
* structured deployable: add one residual Jordan/NCQC matrix word for the
  value-visible quotient of \(\ker\mathcal O_4\), if the endpoint separation
  rank is bounded;
* label-free universal: carry a basis of
  \(\ker\mathcal O_4/\ker\mathcal V_4\), generically as large as the residual
  symmetric fourth-tensor quotient.

Only after this primitive H4 quotient is supplied does the next missing order
become H5 (the mixed \(A_2\diamond A_3\) source) for a general pair observable;
under an even/unary symmetry that kills H5, H6 is next.

### Current next question

Can the exact one-stream NCQC/K31 construction prove
\(\ker\mathcal O_4\subseteq\ker\mathcal V_4\) for the actual L5--14 endpoint
adjoint, or does the value-visible primitive/reset \(q_{4,\perp}\) require a
second matrix word?  This is now the decisive bridge question.

## 11. Bridge attack: exact response constants and impossibility result

Let \(H_l=P_{\ker\mathcal O_{4,l}}C_{4,l}\) be the hidden primitive-quartic
birth at layer \(l\), and let \(\Lambda_l^{(4)}\) be the exact endpoint H4
pullback.  Within the degree-4 tangent dynamics the bridge is exactly

\[
y_4-q_{4,z}
=\sum_{l=5}^{14}\langle\Lambda_l^{(4)},H_l\rangle.
\]

Consequently the sharp bound using only layerwise source norms is

\[
\boxed{
B_{\rm bridge}
\le
\left[
\sum_{l=5}^{14}
\|P_{\ker\mathcal O_{4,l}}\Lambda_l^{(4)}\|
\,\|H_l\|
\right]^2.
}
\]

This bound is attained when the hidden births align with their pullbacks and
their endpoint signs cohere.  It may be replaced by a sum of squares only after
proving endpoint or martingale orthogonality.

For one normalized scalar ReLU, the H4 response coefficient has squared norm

\[
e_4=\frac1{24\pi}.
\]

For an independent centered pair \((\psi_i-E\psi_i)(\psi_j-E\psi_j)\), the
H4 representer is the H2-by-H2 component and has squared norm

\[
\frac1{4\pi^2}.
\]

These constants bound the response operator, but not the hidden source norms
\(\|H_l\|\).  Fixed covariance and all currently carried contractions do not
upper-bound a primitive fourth cumulant; in the observation kernel it can vary
independently at tangent order.

The strongest universal endpoint statement available from the oracle ledger is
only the full quartic capacity

\[
B_{\rm bridge}\le I_4=1.0313\times10^{-9}
\]

when the bridge is defined as the unresolved part of that quartic innovation.
This is already larger than \(0.7453\times10^{-9}\).  If the higher functional
rest is admitted into the bridge uncertainty, its capacity
\(3.005447\times10^{-9}\) makes the bound still weaker.  Thus the desired
bridge certificate does not follow from current data.

The nonidentifiability theorem can be stated as follows.

> A value functional on degree-four sources is determined by the carried state
> iff \(\ker\mathcal O_4\subseteq\ker\mathcal V_4\).  If this inclusion fails,
> choose \(H\in\ker\mathcal O_4\) with \(\mathcal V_4H\ne0\).  The two tangent
> states \(C_4+\epsilon H\) and \(C_4-\epsilon H\) have identical carried
> degree-4 observations but endpoint values separated by
> \(2\epsilon\mathcal V_4H\).  Hence no deterministic label-free bound below
> the allowed quartic capacity can be inferred from those observations.

For a fixed endpoint the minimal extra statistic is the one-dimensional Riesz
functional

\[
s_l=\langle P_{\ker\mathcal O_{4,l}}\Lambda_l^{(4)},H_l\rangle
\]

for every independent layer/adjoint mode.  Dynamically this is the
value-visible primitive/reset \(q_{4,\perp}\) second word.  For all possible
endpoints, the minimal universal state is the quotient

\[
\ker\mathcal O_4/(\ker\mathcal O_4\cap\ker\mathcal V_4),
\]

which is generically a residual symmetric fourth-tensor space.  This is an
order-four identifiability failure; increasing to H5/H6 without first carrying
this quotient cannot fix it.

## 12. Fixed-endpoint primitive-H4 quotient rank and the D10 word

The quotient above is the mathematically correct value-visible hidden quotient;
the shorthand \(\ker\mathcal O_4/\ker\mathcal V_4\) is not generally
well-defined.  For one fixed scalar endpoint,
\(\operatorname{rank}(\mathcal V_4|_{\ker\mathcal O_4})\le1\).  Thus the
information-theoretic minimum is one scalar Riesz statistic.  This does not
imply that one causal matrix word can represent that statistic.

For the D10 construction, the late-layer word space is

\[
U_l=\operatorname{span}\{a_l,b_l\},\qquad l=5,\ldots,14,
\]

while the early space is \(U_l=\operatorname{span}\{a_l\}\).  Its lifted H4
support is

\[
\operatorname{Sym}^2U_l
=\operatorname{span}\{a_la_l,a_lb_l,b_lb_l\},
\]

so the existing word plus one new D10 word have at most three late-layer lifted
adjoint modes.  Exact closure requires

\[
T_l^*U_{l+1}\subseteq U_l
\]

and all primitive reset actions on the endpoint adjoint to lie in
\(\operatorname{Sym}^2U_l\).  At the L5-to-L4 boundary, both
\(T_4^*a_5\) and \(T_4^*b_5\) must collapse to the one-dimensional early
space \(\operatorname{span}\{a_4\}\).

For approximate word closure define

\[
\eta_l=\|(I-P_l)T_l^*P_{l+1}\|.
\]

The induced lifted leak obeys

\[
\boxed{
\eta_l^{(4)}\le2\|T_l\|\eta_l+\eta_l^2.
}
\]

The safe endpoint error is the coherently transported sum of these leaks
contracted with the hidden source norms; it must be tested against the endpoint
amplitude budget, not merely against a local plane RMS.

Suppose the incoming state and ten late primitive resets give candidate reverse
maps \(R_0^*,\ldots,R_{10}^*\).  For terminal adjoint \(G\), define

\[
\chi_G
=\dim\operatorname{span}
\{P_{\mathcal Q_4}R_a^*G:a=0,\ldots,10\}.
\]

The count eleven is not itself a rank theorem: the exact obstruction is
\(\chi_G\).  Under a generic nonvanishing-minor condition and sufficient
ambient dimension, \(\chi_G=11\); algebraic closure, commutation, or endpoint
degeneracy may reduce it.  A single D10 word is exact only when the action lies
inside its three-dimensional lifted support and passes the boundary-collapse
gate.  Generic rank eleven cannot fit.

With \(p\) matrix words,

\[
\dim\operatorname{Sym}^2U\le\frac{p(p+1)}2.
\]

A necessary dimension condition for eleven generic lifted modes is

\[
\frac{p(p+1)}2\ge11,
\]

so \(p\ge5\).  Since the existing word supplies one, at least four additional
D10 words are necessary even before closure constraints.  A D10 word costs
about twenty to twenty-one GEMM equivalents; the current 49-GEMM envelope has
room for only one additional late word.  Exact generic rank-eleven closure is
therefore outside the envelope.

For an approximate Schmidt/Kraus decomposition with singular values
\(\sigma_1\ge\cdots\ge\sigma_{\chi_G}\), a three-mode D10 support needs

\[
\boxed{
\|H_{\rm hidden}\|
\left(\sum_{j>3}\sigma_j^2\right)^{1/2}
\le3.80\times10^{-6}
}
\]

for the Hilbert-norm certificate, or the corresponding operator-norm condition
with \(\sigma_4\).  Dimension alone cannot rule out rapid spectral decay, so
approximate D10 success requires this explicit spectral certificate.

The amplitude \(3.80\times10^{-6}\) is

\[
\sqrt{10^{-8}}-\sqrt{9.2547\times10^{-9}}.
\]

Its squared budget is about \(1.44\times10^{-11}\), only \(1.40\%\) of the
ideal quartic energy \(1.0313\times10^{-9}\).  Under an orthogonal quartic
energy interpretation, D10 must therefore retain at least \(98.60\%\) of the
primitive-H4 endpoint energy.  The separate 3.28064-percent one-stream tail is
the looser orthogonal-energy allowance relative to the full low-order
correction; it must not replace this adversarial endpoint-amplitude gate unless
orthogonality is proved.
### 12.1 Sharp fail-fast and cost gates for the primitive-H4 quotient

There are two different ranks and they must not be conflated.  For one fixed
scalar endpoint, the unrestricted information quotient

\[
 {\cal Q}_4:=\ker O_4/(\ker O_4\cap\ker V_4)
 \simeq V_4(\ker O_4)\subseteq\mathbb R
\]

has dimension at most one.  (The literal quotient
`ker(O4)/ker(V4)` is defined only under the additional inclusion
\(\ker V_4\subseteq\ker O_4\).)  This rank-one statement says that an
unrestricted Riesz representer exists; it does **not** say that the
representer factors through one causal/local D10 word.

For the latter question, let \(G\) be the fixed terminal adjoint and let
\(R_0,\ldots,R_{10}\) denote the incoming primitive-H4 route plus the ten
late-layer reset routes, all pulled back to one common boundary.  Define

\[
 g_a=P_{{\cal Q}_4}R_a^*G,\qquad
 \Gamma_{ab}=\langle g_a,g_b\rangle,\qquad
 \chi_G=\operatorname{rank}\Gamma
       =\dim\operatorname{span}\{g_0,\ldots,g_{10}\}.
\]

This \(\chi_G\), not the scalar quotient dimension and not the bare count
eleven, is the required deployable adjoint-mode rank.  Eleven routes imply
\(\chi_G=11\) only under the explicit generic-independence/nonzero-minor
hypothesis.  A single D10 `q4_perp` word gives a two-word state
\(U=\operatorname{span}\{a,b\}\), hence only
\(\dim\operatorname{Sym}^2U=3\) primitive-H4 directions.  Therefore:

* exact one-word success requires every \(g_a\in\operatorname{Sym}^2U\),
  reverse invariance at every late boundary, and the two-dimensional
  \(L5\to L4\) state to collapse to the one-dimensional early state;
* any nonzero \(4\times4\) principal (or mixed) Gram minor of \(\Gamma\)
  is already a complete exact no-go certificate for one D10 word;
* if \(\chi_G=\chi\), any construction from \(p\) matrix words obeys the
  necessary capacity bound
  \[
  \frac{p(p+1)}2\ge\chi,
  \qquad
  p\ge\left\lceil\frac{\sqrt{1+8\chi}-1}{2}\right\rceil .
  \]
  Thus generic \(\chi=11\) needs \(p\ge5\), i.e. at least four additional
  D10-class words beyond the existing word.  At roughly 20--21 GEMM
  equivalents per added word, this is at least 80 extra GEMM equivalents;
  with the 28-GEMM core it is at least 108, outside the 49-GEMM envelope.

Approximation is governed by spectrum, not dimension.  If
\(\sigma_1\ge\cdots\ge\sigma_{\chi_G}\) are the singular values of the
weighted reset-to-endpoint action, the best three-mode carrier has
worst-case omitted gain \(\sigma_4\), and isotropic/RMS omitted energy
\(\sum_{j>3}\sigma_j^2\).  With hidden-source norm \(H_4\), the sharp gates
are

\[
 H_4\sigma_4\le3.80\times10^{-6}\quad\text{(adversarial)},
 \qquad
 H_4^2\sum_{j>3}\sigma_j^2\le(3.80\times10^{-6})^2
 \quad\text{(orthogonal/RMS)}.
\]

The amplitude budget squares to \(1.444\times10^{-11}\).  Relative to the
ideal quartic energy \(1.0313\times10^{-9}\), a one-word D10 carrier must
therefore retain at least \(98.60\%\) of the value-visible primitive-H4
energy.  The previously quoted 3.28064% one-stream tail is a different,
looser orthogonal-energy allowance and cannot replace this adversarial
amplitude gate without a proved orthogonality.

If \(P_l\) projects onto the proposed local word space and
\(\eta_l=\|(I-P_l)T_l^*P_{l+1}\|\), the lifted one-step leakage is bounded by
\(2\|T_l\|\eta_l+\eta_l^2\).  A complete ten-layer acceptance certificate
must add the propagated sum of these leakages to the best-rank-three
spectral residual and keep the total below \(3.80\times10^{-6}\); satisfying
the spectral gate alone is not enough.

**Graph-ready decision:** prove one nonzero four-action Gram minor to close
exact single-word D10 immediately.  If exact closure survives, prove the
weighted spectral-tail-plus-propagated-leak bound below \(3.80\times10^{-6}\).
Without either certificate, the single-word claim is nonidentifiable from
the endpoint scalar alone.

## 13. Universal projected K31 action kills the fixed-endpoint H4 quotient

> **Binding ownership correction (Checkpoint 8).**  Sections 13--14 establish
> an information identity, not additive novelty.  The total same-site
> assembled K31/Q24 action is already owned by D11.  Every statement below
> about a "new" or "minimal" total action is superseded by Section 15: only
> the consistently residualized true-minus-assembled defect can be new.

Let \({\cal H}_4\) be the primitive fourth-cumulant tangent space, let
\(N=\ker O_4\), and use the well-defined value-visible quotient

\[
Q(N,V)=N/(N\cap\ker V),\qquad \dim Q(N,V)=\operatorname{rank}(V|_N).
\]

At one layer define the endpoint-projected variables

\[
U_a=\sum_iW_{ia}X_i,\qquad V_a=\sum_iP_{ia}X_i.
\]

For centered variables the universal tensor-free observation is

\[
K_{31}=\sum_a\kappa_4(U_a,U_a,U_a,V_a)
=\sum_a\{\mathbb E[U_a^3V_a]
-3\mathbb E[U_a^2]\mathbb E[U_aV_a]\}.
\]

By cumulant multilinearity this equals the complete connected K31 endpoint
action \(\langle Q^{31},D^{31}\rangle\), with no latent-partition
assumption.  Therefore, after applying the same collision and old-span
subtractions on both sides,

\[
\boxed{K_{31}|_N=V_{31}|_N.}
\]

Consequently, for a fixed nonzero scalar K31 endpoint, augmenting the
observation map by the **full** projected cumulant kills exactly
\(\operatorname{rank}(V_{31}|_N)=1\) quotient dimension, i.e. 100 percent of
the K31 value-visible quotient:

\[
N\cap\ker K_{31}\subseteq\ker V_{31},\qquad
Q(N\cap\ker K_{31},V_{31})=0.
\]

If the requested value also has a non-K31 component \(V_{\rm other}\), only
the K31 row direction is killed; on \(N\cap\ker K_{31}\) the residual is
\(V_{\rm other}\).  Thus this theorem must not be promoted into completeness
for a generic K22 or other primitive-H4 endpoint.

### 13.1 Exact residual quotient and action-Gram formula

For several endpoint/reset actions let \(g_j\in N\) be their Riesz vectors,
and let \(k_t\in N\) be the added observable vectors.  Put
\(S_K=\operatorname{span}\{k_t\}\).  The residual action vectors are
\(g_j^{\rm res}=P_{S_K^\perp}g_j\), and their Gram is the exact Schur
complement

\[
\boxed{
\Gamma_{\rm res}
=\Gamma_{gg}-\Gamma_{gk}\Gamma_{kk}^{+}\Gamma_{kg}.
}
\]

Hence

\[
\operatorname{rank}_{\rm res}
=\dim P_{S_K^\perp}\operatorname{span}\{g_j\},
\]

and the number of quotient directions killed is
\(\dim(\operatorname{span}\{g_j\}\cap S_K)\).  If each of the ten late-layer
projected actions is observed, then \(g_j=k_j\) and
\(\Gamma_{\rm res}=0\): the generic eleven-Kraus obstruction disappears for
this target-adjoint consumer.  If only the one summed scalar is observed,
then it kills the complete fixed aggregate endpoint, but it removes only one
direction from an independently queryable route/panel action Gram; generically
its rank falls from \(\chi_G\) to \(\chi_G-1\), not to zero.  This separates
fixed-endpoint information rank from universal causal-state rank.

### 13.2 Exact affine--quadratic decoder

Suppose the retained synchronized source law is

\[
X_i=h_i^{\mathsf T}z+z^{\mathsf T}A_i z-\operatorname{tr}A_i,
\qquad z\sim N(0,I).
\]

For one projected pair put

\[
N_a=\sum_iW_{ia}A_i,\quad R_a=\sum_iP_{ia}A_i,
\qquad h_a=\sum_iW_{ia}h_i,\quad r_a=\sum_iP_{ia}h_i.
\]

The cumulant-generating function

\[
-\operatorname{tr}A-\tfrac12\log\det(I-2A)
+\tfrac12h^{\mathsf T}(I-2A)^{-1}h
\]

has homogeneous degree-four part
\(2\operatorname{tr}(A^4)+2h^{\mathsf T}A^2h\).  Polarizing its \(x^3y\)
coefficient gives the exact closed form

\[
\boxed{
\kappa_4(U_a,U_a,U_a,V_a)
=48\operatorname{tr}(N_a^3R_a)
+24\{h_a^{\mathsf T}N_aR_ah_a+h_a^{\mathsf T}N_a^2r_a\}.
}
\]

The first term is the pure quadratic NCQC trace.  The second is the entire
H1--H1--H2--H2 contribution.  Thus the trace alone does not kill the full
affine--quadratic quotient unless the linear core vanishes; trace plus the two
matrix-vector terms does.

### 13.3 Rigorous beyond-quadratic remainder and mismatch gate

Write the exact projected variables as

\[
U=E+\epsilon,\qquad V=F+\zeta,
\]

where \((E,F)\) are the synchronized affine--quadratic projections and
\((\epsilon,\zeta)\) contain all remaining chaos and frame error.  Cumulant
multilinearity gives the exact remainder

\[
\begin{aligned}
d={}&3\kappa(\epsilon,E,E,F)+3\kappa(\epsilon,\epsilon,E,F)
+\kappa(\epsilon,\epsilon,\epsilon,F)+\kappa(E,E,E,\zeta)\\
&+3\kappa(\epsilon,E,E,\zeta)
+3\kappa(\epsilon,\epsilon,E,\zeta)
+\kappa(\epsilon,\epsilon,\epsilon,\zeta).
\end{aligned}
\]

For centered arguments,
\(|\kappa(A,B,C,D)|\le4\|A\|_4\|B\|_4\|C\|_4\|D\|_4\).
With \(e=\|E\|_4,f=\|F\|_4,r=\|\epsilon\|_4,s=\|\zeta\|_4\),

\[
\boxed{
|d|\le b
=4\{3e^2rf+3er^2f+r^3f+e^3s+3e^2rs+3er^2s+r^3s\}.
}
\]

Equivalently, if \(r\le\alpha e\) and \(s\le\beta f\),

\[
|d|\le4e^3f\{(1+\alpha)^3(1+\beta)-1\}.
\]

For a finite Gaussian-chaos remainder, Nelson hypercontractivity supplies the
fully L2-based sufficient estimate

\[
r\le\sum_{m\ge3}3^{m/2}\|\epsilon_m\|_2,
\qquad
s\le\sum_{m\ge3}3^{m/2}\|\zeta_m\|_2,
\]

whenever the displayed weighted sums converge.  Otherwise the direct L4
bound above remains the rigorous gate.

Let \(\widehat g_j\) denote the affine--quadratic action vector and
\(d_j=g_j-\widehat g_j\) the exact beyond-quadratic remainder.  Since the
new observable span contains every \(\widehat g_j\),

\[
\boxed{
\Gamma_{\rm res}
=D^*P_{S_{\widehat K}^{\perp}}D
\preceq D^*D,
}
\]

so its rank is at most \(\operatorname{rank}D\), its operator norm at most
\(\|D\|_{\rm op}^2\), and its trace at most
\(\sum_j\|d_j\|^2\).  For the fixed summed endpoint, Minkowski and the bound
above give

\[
B_{\rm residual}^{1/2}
\le\left\|\sum_{l,a}d_{l,a}\right\|_2
\le\left\|\sum_{l,a}b_{l,a}\right\|_2
=:\eta_{>2}.
\]

Therefore the exact universal action gives \(B_{\rm residual}=0\) for the
K31 primitive-H4 branch and passes both budgets automatically.  The
affine--quadratic surrogate is certified only if

\[
\eta_{>2}<\sqrt{0.7453\times10^{-9}}
\simeq2.730\times10^{-5}
\]

for the minimum quartic gain, and it satisfies the stricter endpoint margin
only if

\[
\boxed{\eta_{>2}\le3.80\times10^{-6}.}
\]

The latter leaves residual energy at most \(1.444\times10^{-11}\), or 1.40
percent of the ideal \(1.0313\times10^{-9}\) quartic energy.  No such bound
follows merely from H1/H2 synchronization: the required L4 (or weighted
chaos-tail) norms of the actual H3/H4-plus remainder must be exposed.  Thus
the full action moment closes primitive-H4 identifiability exactly; the
affine--quadratic formula alone remains conditional on this explicit tail
certificate.

**Graph-ready gates.**  `K31_ACTION_EXACT`: verify the full moment and
covariance subtraction use the identical endpoint mask and old-span lineage;
then the fixed-endpoint residual rank is zero.  `K31_ACTION_AQ`: include both
linear--quadratic terms, not just \(48\operatorname{tr}(N^3R)\), and require
\(\eta_{>2}\le3.80\times10^{-6}\).  `K31_ACTION_PANEL`: an aggregate scalar
does not close an independent panel/route Gram; expose one action per required
adjoint direction or retain the Schur-complement rank.

## 14. Checkpoint 7: best current-state AQ-tail certificate

> **Ownership scope.**  The bounds in this section apply to the
> true-minus-AQ action.  They apply to the novel D11 residual only if the AQ
> action is proved identical to the assembled D11/Q24 action under the same
> Site-A/Site-B, collision, and old-span conventions.  See Section 15.

The AQ residual must be split before assigning a radius.  At every endpoint
projection write

\[
U=E+T+F,\qquad V=G+S+J,
\]

where `(E,G)` are the synchronized affine--quadratic variables, `(T,S)` are
intrinsic source chaos of degree at least three in that synchronized frame,
and `(F,J)` are core/frame/state or endpoint-mask errors.  The frame terms are
not themselves intrinsic higher chaos and must not be charged to the same
budget.

### 14.1 Sharp projected L4 bound

For a collection indexed by layer and projected column, define

\[
\|A\|_{4,4}:=
\left(\sum_{l,a}\|A_{l,a}\|_{L^4}^4\right)^{1/4}.
\]

For the four-linear cumulant action

\[
{\cal C}(A,B,C,D)=\sum_{l,a}\kappa_4(A_{l,a},B_{l,a},C_{l,a},D_{l,a}),
\]

Hölder first in probability and then in `(l,a)` gives

\[
\boxed{
|{\cal C}(A,B,C,D)|
\le4\|A\|_{4,4}\|B\|_{4,4}\|C\|_{4,4}\|D\|_{4,4}.
}
\]

This is the natural projective/dual L4 bound for the already endpoint-mixed
variables; it is never worse than summing four separate worst-case scalar
maxima.  More abstractly, for a fourth-tensor source norm `||.||_T`, the exact
best ball certificate is

\[
|\langle\Lambda_l,\Delta C_{4,l}\rangle|
\le\|P_{\ker O_4}\Lambda_l\|_{T^*}\,\|\Delta C_{4,l}\|_T.
\]

No smaller distribution-free constant is possible for that norm because the
dual norm is a supremum.  The problem is not the dual factor--the known
weights and reverse mask determine it--but the absence of a radius for the
hidden source ball.

For one perturbation `(R,Z)` around `(E,G)`, exact telescoping yields

\[
\begin{aligned}
&|{\cal C}(E+R,E+R,E+R,G+Z)-{\cal C}(E,E,E,G)|\\
&\quad\le4\{r(e+r)^2(g+z)+er(e+r)(g+z)
+e^2r(g+z)+e^3z\}\\
&\quad=4\{[(e+r)^3-e^3]g+(e+r)^3z\},
\end{aligned}
\]

where lower-case letters denote the corresponding `||.||_(4,4)` norms.
Applying this bound first with `(R,Z)=(T,S)` and then from `(E+T,G+S)` to the
actual `(U,V)` with `(F,J)` gives separate coherent radii
`eta_chaos` and `eta_frame`; mixed chaos--frame terms occur only in the second
telescoping factor and are not omitted or double counted.

### 14.2 What the current chaos summaries can and cannot bound

Decompose the synchronized tail by Wiener order,

\[
T=T_3+T_4+T_{\ge5},\qquad S=S_3+S_4+S_{\ge5}.
\]

For a pure order-`m` Gaussian chaos, Nelson hypercontractivity gives

\[
\|T_m\|_4\le3^{m/2}\|T_m\|_2.
\]

Hence the lawful finite-order receipts are

\[
\|T_3\|_4\le3^{3/2}\|T_3\|_2,qquad
\|T_4\|_4\le9\|T_4\|_2,
\]

and, only when it converges,

\[
\|T_{\ge5}\|_4
\le\sum_{m\ge5}3^{m/2}\|T_m\|_2.
\]

The same formulas hold for `S`.  The current variance/CP3/Q24 ledger supplies
at most selected low-order contractions and an **unweighted** residual L2
energy.  There is no finite constant `C` such that
`||T_{>=3}||_4 <= C ||T_{>=3}||_2` over unbounded Gaussian chaos: a normalized
order-`m` component has hypercontractive scale growing with `m`.  ReLU has an
infinite Hermite expansion, so a degree-four hypercontractive constant cannot
be applied to its full tail.  Thus the unweighted residual variance is not a
valid substitute for the required L4 radius.

Some coarse physical L4 quantities are available.  For a centered unit,

\[
\|X_i\|_4^4=\kappa_{4,iiii}+3C_{ii}^2,
\]

and therefore

\[
\left\|\sum_iw_iX_i\right\|_4
\le\sum_i|w_i|\{\kappa_{4,iiii}+3C_{ii}^2\}^{1/4}.
\]

This uses only marginals and is lawful, but it is an L1-in-coordinate worst
case and does not isolate the AQ remainder.  The missing all-distinct/mixed
fourth moments are exactly what would sharpen it.  In particular, marginal
K4, K31 collision faces, K22 faces, and selected Q24 planes do not determine
the fourth moment of a generic dense projection.

### 14.3 Same-frame core error is computable exactly

For

\[
Y=h^Tz+z^TAz-\operatorname{tr}A
\]

in the no-half quadratic convention,

\[
\boxed{
\|Y\|_4^4
=3(\|h\|^2+2\operatorname{tr}A^2)^2
+48\operatorname{tr}A^4+48h^TA^2h.
}
\]

Therefore, if the true and tracked cores are expressed in one synchronized
Gaussian gauge, the L4 norm of their AQ difference is obtained exactly by
inserting `(delta h,delta A)` into this formula after the `W` or `P` mixing.
This is the best usable analytic frame certificate.

It does not presently close the gate.  The retained core is known only up to
a nontrivial inverse fiber, and the state has no certified target-secanted
radius for `(delta h,delta A)`.  Without a synchronized coupling, even the
difference formula is unavailable; only the triangle bound between the two
individual L4 norms remains.

### 14.4 Endpoint-mask and nonlinear secant error

If `U` is exact but the endpoint projection changes from `P_hat` to `P`, let

\[
\Delta V_a=\sum_i(P-\widehat P)_{ia}X_i.
\]

Linearity in the fourth slot gives the sharp projected bound

\[
\boxed{
|\Delta K_{31}^{\rm mask}|
\le4\|U\|_{4,4}^3\|\Delta V\|_{4,4}.
}
\]

Marginal fourth moments give the safe but coarse receipt

\[
\|\Delta V_a\|_4
\le\sum_i|\Delta P_{ia}|\|X_i\|_4.
\]

For the normalized ReLU covariance kernel

\[
\Psi(c)=\{\sqrt{1-c^2}+(\pi-\arccos c)c\}/\pi,
\]

if both exact and tracked correlations lie in `|c|<=1-gamma`, the frozen
tangent remainder obeys

\[
|R_{ij}|
\le c_\gamma|\Delta C_{ij}|^2,
\qquad
c_\gamma={1\over2\pi\sqrt{2\gamma-\gamma^2}}.
\]

Thus a reverse matrix `H` has the rigorous dual receipts

\[
|\langle H,R\rangle|
\le c_\gamma\min\left\{
\|H\|_F\|\Delta C\odot\Delta C\|_F,
\|H\|_1\|\Delta C\|_\infty^2
\right\}.
\]

An exact path-integrated secant mask makes this term zero.  The current state
has neither a certified exact-minus-tracked `Delta C` norm nor a uniform
correlation margin; near `|c|=1` the Hessian constant diverges.  Hence this
term also has no current `3.80e-6` receipt.

### 14.5 Normalization reconciliation

The universal action uses actual centered physical variables and the ordinary
joint cumulant; there is no Hermite factorial outside
`kappa4(U,U,U,V)`.  With

\[
q_A=z^TAz-\operatorname{tr}A,
\]

the AQ K31 coefficients are `48` and `24`, as in Section 13.  In the square
lift convention

\[
Q_M=\tfrac12(z^TMz-\operatorname{tr}M),
\]

set `A=M/2`; the identical formula becomes

\[
\boxed{
\kappa_4(U,U,U,V)
=3\operatorname{tr}(M^3R)
+6\{h^TMRh+h^TM^2r\}.
}
\]

Likewise

\[
\|h^Tz+Q_M\|_4^4
=3(\|h\|^2+\tfrac12\operatorname{tr}M^2)^2
+3\operatorname{tr}M^4+12h^TM^2h.
\]

The endpoint mask `P=W(Q31)^T` already contains the deployed diagram
normalization; multiplying the cumulant by another `3!`, `4!`, or square-lift
factor would be double counting.

Finally, `3.80e-6` is an L2 **amplitude** allowance:

\[
3.80\times10^{-6}
=\sqrt{10^{-8}}-\sqrt{9.2547\times10^{-9}},
\]

whereas `0.7453e-9` is an MSE/energy allowance.  A scalar residual radius
`eta` compares directly with the former and only `eta^2` compares with the
latter.

### 14.6 Decision: a new projected action is information-necessary

Let `S_current` be the carried means, covariance, marginal/equality K4 faces,
CP3/Q24 contractions, and AQ core.  The minimax uncertainty of the projected
action at a fixed state value is half its fiber width,

\[
r_*(s)=\frac12\left[
\sup_{P:S_{\rm current}(P)=s}K_{31}(P)
-\inf_{P:S_{\rm current}(P)=s}K_{31}(P)
\right].
\]

The current observation kernel contains primitive fourth-cumulant directions
on all-distinct/mixed sectors that preserve every listed face but have
nonzero contraction with the generic dense K31 endpoint tensor.  It also
contains higher-chaos/frame completions with the same unweighted low-order
state and different next-layer secant action.  Therefore the current summaries
do not imply `r_*(s)<=3.80e-6`; indeed no threshold-small distribution-free
radius follows from them.

The information-minimal repair for one fixed endpoint is one new scalar,

\[
K_{31}^{\rm exact}
=\sum_{l,a}\{E[U_{l,a}^3V_{l,a}]
-3E[U_{l,a}^2]E[U_{l,a}V_{l,a}]\},
\]

or equivalently its residual relative to the AQ formula.  This sets the K31
fiber width to zero.  Per-layer scalars are needed only for separate route
queries; a panel needs one scalar per independent endpoint-adjoint row.  An
assumption proving a convergent weighted chaos-tail bound, synchronized core
radius, exact secant action, and collision/state receipts could replace the
observable, but no such theorem is present.  Thus the projected fourth-action
observable is strictly necessary for a current-state, distribution-free
`3.80e-6` certificate.

### Graph-node split required by hygiene

1. `K31_ACTION_EXACT_QUOTIENT`: full projected cumulant equals the fixed K31
   value functional; terminal criterion is identical mask/lineage and zero
   quotient residual.
2. `K31_ACTION_AQ_DECODER`: the `48+24` (or normalized `3+6`) closed form;
   terminal criterion includes the H1--H2 cross term.
3. `K31_ACTION_HGE3_TAIL`: intrinsic H3/H4/H>=5 plus core/frame/secant radii;
   terminal criterion is a lawful combined L2 radius, not unweighted variance.
4. `K31_ACTION_RANK_SCOPE`: aggregate fixed endpoint versus independent
   route/panel Schur-complement rank.
5. `K31_ACTION_ENDPOINT_GATES`: keep `0.7453e-9` energy and `3.80e-6`
   amplitude gates dimensionally separate.

## 15. Binding novelty correction: bound the true-minus-assembled defect

The total same-site K31 action is already an owned D11/Q24 action.  Define,
with identical Site-A/Site-B, endpoint mask, collision subtraction, and
old-span residualization,

\[
\chi_{\rm true}=\langle\Lambda,K_{31}^{\rm true}\rangle,
\quad
\chi_{\rm AQ}=\langle\Lambda,K_{31}^{\rm AQ}\rangle,
\quad
\chi_{\rm asm}=\langle\Lambda,K_{31}^{\rm assembled}\rangle.
\]

The only candidate new source is

\[
\boxed{
\rho_{31}=\chi_{\rm true}-\chi_{\rm asm}
=(\chi_{\rm true}-\chi_{\rm AQ})
+(\chi_{\rm AQ}-\chi_{\rm asm}).
}
\]

The seven-term cumulant expansion and every L4 bound in Sections 13--14 apply
**exactly** to the first bracket, `chi_true-chi_AQ`.  They bound the novel
defect itself only after proving the ownership/assembly identity

\[
\chi_{\rm AQ}=\chi_{\rm asm}.
\]

If that identity fails, the second bracket is a separate deterministic
assembly, gauge, collision, site, or lineage discrepancy.  The only lawful AQ
feature is then `chi_AQ-chi_asm`; using `chi_AQ` alone double-counts the old
D11/Q24 action.  Conversely, if exact synchronized AQ physics proves
`chi_true=chi_AQ=chi_asm`, it proves `rho31=0` and closes novelty rather than
promoting it.

For a collision-free primitive birth `H_t` whose same-frame repeated-index
K31 slice is zero, the immediate coordinate K31 action is zero.  Its first
possible contribution is at a fresh dense-law boundary,

\[
\rho_{\rm cf}
=\sum_{l>t}\left\langle\Lambda_l,
U_{l\leftarrow t}^{(4)}H_t\right\rangle,
\]

where `U^(4)` includes the lawful order-four transport and resets.  This term
is contained in `chi_true-chi_AQ` only when the AQ state omits the birth and
the true projected action is evaluated at that future boundary.  A same-frame
K31 face or same-site AQ recomputation cannot reveal it.

The corrected radius is therefore

\[
\eta_{31}
\le \eta_{\rm true-AQ}
+\|\chi_{\rm AQ}-\chi_{\rm asm}\|_2
+\eta_{\rm site/coll/lineage},
\]

with `eta_true-AQ` supplied conditionally by Section 14.  None of these terms
currently has a certified combined radius below `3.80e-6`.  Exact access to
the **true** projected action, followed by subtraction of the assembled old
action, makes the information residual exact (up to acquisition error); exact
access only to the AQ or assembled action has zero new information.

This supersedes the earlier graph-node names.  The lawful split is:

1. `K31_D11_OWNERSHIP`: total assembled same-site action, solved duplicate;
2. `K31_DIRECT_DEFECT_IDENTITY`: true-minus-assembled scalar identity;
3. `K31_AQ_DEFECT_PROXY`: only `chi_AQ-chi_asm`, conditional on a tail bound;
4. `K31_FUTURE_COLLISION_FREE`: delayed fresh-boundary observability;
5. `K31_DEFECT_ENDPOINT_GATE`: residualized defect amplitude and acquisition
   error below the endpoint budgets.

## 16. Exact diagonal-quartic polarization and JVP audit

Let `T` be an arbitrary real symmetric four-linear form and

\[
\Phi(x)=T(x,x,x,x).
\]

For fixed real directions `U,V`, put

\[
p(t)=\Phi(U+tV)
=a+4bt+6ct^2+4dt^3+et^4,
\]

where

\[
b=T(U,U,U,V)=\frac14D\Phi(U)[V].
\]

The formulas below apply to the true fourth-cumulant potential.  To recover an
entire endpoint sum in one call, define the aggregate potential

\[
{\cal F}(W)=\sum_a\Phi(w_a),
\]

and use the matrix direction `P=(p_a)`; then
`D F(W)[P]/4=sum_a T(w_a,w_a,w_a,p_a)`.  If the API exposes only scalar
single-direction calls, multiply every call count below by the number of
columns.

### 16.1 Point values only: four calls are necessary and sufficient

For any `0<t<1`, let

\[
\Delta_t=\Phi(U+tV)-\Phi(U-tV),
\qquad
\Delta_1=\Phi(U+V)-\Phi(U-V).
\]

Then

\[
\boxed{
b=\frac{\Delta_t-t^3\Delta_1}{8t(1-t^2)}.
}
\]

At `t=1/2` this is the particularly stable exact formula

\[
\boxed{
b={\Phi(U+V/2)-\Phi(U-V/2)\over3}
-{\Phi(U+V)-\Phi(U-V)\over24}.
}
\]

Four real calls are minimal.  Indeed, if at most three projective point
evaluations determined `b` for every binary quartic, there could be no degree
at-most-four polynomial vanishing at all queried ratios with nonzero
derivative at zero.  For distinct finite ratios, let `g` be their vanishing
polynomial.  Either `g'(0) != 0`, or `g(0) != 0` and `t g(t)` has nonzero
derivative at zero; both have degree at most four.  A query at infinity merely
fixes the leading coefficient and leaves the same construction at degree at
most three.  Thus three calls cannot identify `b`.

If the four returned values have deterministic errors `eps_(+t),...`, the
exact worst-case amplification is

\[
|\widehat b-b|
\le {\epsilon_{+t}+\epsilon_{-t}
+t^3(\epsilon_{+1}+\epsilon_{-1})
\over8t(1-t^2)}.
\]

For equal absolute errors `epsilon`,

\[
A_4(t)={t^{-1}+t^2\over4(1-t^2)}.
\]

On normalized queries `U+rV`, `|r|<=1`, this is minimized at `t=1/2`, giving
the sharp adversarial factor

\[
\boxed{|\widehat b-b|\le\tfrac34\epsilon.}
\]

Sharpness follows from the Chebyshev polynomial
`T_3(r)=4r^3-3r`: it is bounded by one on `[-1,1]` and its target coefficient
has magnitude `3/4`, so no point-evaluation representation can have total
absolute weight below `3/4`.  For independent equal-variance call errors, the
same four-call formula has RMS factor

\[
\sqrt{2(1/3)^2+2(1/24)^2}=\sqrt{65/288}\simeq0.475.
\]

An absolute-error oracle with unrestricted rescaling is ill posed: homogeneity
would allow arbitrarily large query vectors and arbitrarily small divided
error.  The sharp factor above assumes normalized queries or, equivalently, a
scale-covariant/relative oracle error.

### 16.2 Known `Phi(U)` and `Phi(V)`: three additional calls are minimal

If `a=Phi(U)` and `e=Phi(V)` are already known, then for any `0<t<1`, three
new values `p(t),p(1),p(-1)` give

\[
\boxed{
\begin{aligned}
b={}&{p(t)\over4t(1-t^2)}
-{t\,p(1)\over8(1-t)}
-{t\,p(-1)\over8(1+t)}\\
&-{a\over4t}+{t e\over4}.
\end{aligned}}
\]

Two new finite nonzero ratios cannot suffice: cancelling the unknown `c,d`
would require weights satisfying both
`lambda r^2+mu s^2=0` and `lambda r^3+mu s^3=0`, whose determinant is nonzero
for distinct `r,s`.  A zero or infinite query contains only an already-known
coefficient.  Hence three is minimal.

At `t=1/2`,

\[
b={2\over3}p(1/2)-{1\over8}p(1)-{1\over24}p(-1)
-{1\over2}a+{1\over8}e.
\]

For exact known baselines and equal errors on the three new calls, this
particular three-node family has adversarial factor

\[
A_3(t)={1+t^2\over4t(1-t^2)}.
\]

Within this restricted `{t,1,-1}` family it is minimized at
`t^2=sqrt(5)-2`, with

\[
A_{3,\min}
={1+\sqrt5\over8\sqrt{\sqrt5-2}}\simeq0.832.
\]

This is **not** the globally optimal three-call design.  With both pure
baselines cached, three calls attain the same sharp `3/4` factor as the
four-call stencil.  For any `h>0`, put `f(t)=Phi(U+tV)`; then

\[
\boxed{
b={\frac13f(-h)-2f(-h/2)+\frac23f(h/2)
+\Phi(U)-\frac14h^4\Phi(V)\over4h}.
}
\]

The three noisy-call weights have total absolute mass

\[
{1/3+2+2/3\over4h}={3\over4h},
\]

which is globally sharp by the same Chebyshev witness after rescaling to
`[-h,h]`.  Thus for normalized `h=1`, both the minimal four-call no-baseline
formula and the minimal three-call two-baseline formula amplify equal
adversarial value error by exactly `3/4`.

There is a baseline asymmetry.  Cached `Phi(U)` alone does not reduce the four
fresh calls, while cached `Phi(V)` (the leading coefficient) reduces them to
three; the latter's optimal equal-error factor is

\[
{7\sqrt7+10\over36h}\simeq{0.79225\over h}.
\]

If the known baselines in either displayed formula have errors, their own
absolute interpolation weights must be added.  If they require fresh noisy
calls, the baseline-free four-call stencil dominates.

### 16.3 AD/JVP: one call

One exact forward JVP gives

\[
\boxed{b=\tfrac14\operatorname{JVP}_{\Phi}(U;V).}
\]

Equivalently, one JVP of the aggregate `F(W)` in direction `P` gives the full
endpoint sum.  One derivative call is minimal unless that derivative is
already stored.  If the JVP has absolute error `epsilon_D`, then

\[
\boxed{|\widehat b-b|\le\epsilon_D/4.}
\]

For a reverse gradient error `Delta grad Phi`, the corresponding bound is
`|error|<=||Delta grad Phi||_* ||V||/4` in any dual norm pair.

### 16.4 Endpoint budget and what is actually measured

For one aggregate normalized oracle, the four-point formula meets the strict
`3.80e-6` acquisition allowance if its per-call absolute error obeys

\[
\epsilon\le {3.80\times10^{-6}\over3/4}
=5.067\times10^{-6}.
\]

The optimal three-new-call formula with both exact baselines has the same
`5.067e-6` allowance; the restricted `{t,1,-1}` formula would require about
`4.57e-6`.  A JVP requires `epsilon_D<=1.52e-5`.  For ten separately
acquired layer actions with adversarial coherent errors, divide these
per-layer allowances by ten (or use the actual Bellman coefficient L1 norm).
Independent-error RMS aggregation is not a lawful replacement for the strict
adversarial gate without an independence receipt.

If `Phi` is the **exact true-law** fourth-cumulant potential, polarization/JVP
measures all higher-chaos, latent-partition, and future-visible
collision-free contributions to that projected fourth action.  It does not
make them disappear; it observes them, so no AQ tail bound is needed for this
scalar.  Subtracting `chi_asm` then exposes `rho31` exactly up to acquisition
and lineage error.  If `Phi` is constructed only from the AQ core or D11/Q24
assembly, the same formulas recover only the old owned action and leave the
higher-chaos/collision-free defect completely unresolved.

There is no established contract-native exact-potential/JVP oracle in the
current deployed state.  If the black box instead returns a generic nonlinear
response `Phi+R`, the JVP contains `D R(U)[V]`, and the finite stencil retains
the linear and higher odd secant contamination of `R`.  A fourth Taylor
derivative at zero isolates ordinary degree four, but not automatically a
Wiener/Hermite-H4 source, because higher chaoses have lower-order contractions.
If evaluating the exact cumulant potential already requires the missing H4
action, polarization is circular rather than a carrier.  The query theorem is
therefore an oracle-interface gate, not proof that the current engine supplies
the oracle.

### 16.5 Explicit optimum when only `Phi(V)` is cached

The leading-coefficient-only result above has an exact three-node
construction.  Write `f(t)=Phi(U+tV)`, `e=Phi(V)`, and set

\[
p={\sqrt7-1\over3},\qquad
r={p\over1+p}={\sqrt7-1\over\sqrt7+2}.
\]

Use the three normalized nodes

\[
x_1=-r,\qquad x_2=p,\qquad x_3=1
\]

and weights

\[
\begin{aligned}
w_1&=-{(p+1)^3\over p(p+2)(2p+1)},\\
w_2&={1\over p(p+2)(1-p)},\\
w_3&=-{p^2\over(2p+1)(1-p)}.
\end{aligned}
\]

They obey

\[
\sum_iw_i=0,\quad \sum_iw_ix_i=1,\quad
\sum_iw_ix_i^2=\sum_iw_ix_i^3=0,
\quad \sum_iw_ix_i^4=x_1x_2x_3=-rp.
\]

Consequently the exact formula, for `Phi(X)=c T(X,X,X,X)`, is

\[
\boxed{
T(U,U,U,V)=
{w_1f(-rh)+w_2f(ph)+w_3f(h)+rp\,h^4\Phi(V)
 \over4ch}.}
\]

Here `c=1` under the convention used elsewhere in this lane.  The proof of
optimality is elementary.  Three nodes can represent the derivative of every
cubic exactly only if their node polynomial has zero linear coefficient:

\[
x_1x_2+x_1x_3+x_2x_3=0.
\]

The signs must therefore split two against one.  Up to reflection and scaling
every admissible triple is `(-p/(1+p),p,1)`, `0<p<1`.  The weights have signs
`(-,+,-)`, so their absolute mass is

\[
\sum_i|w_i|=2w_2={2\over p(p+2)(1-p)}.
\]

Maximizing the denominator gives `3p^2+2p-2=0`, hence the displayed `p`, and

\[
{1\over4h}\sum_i|w_i|
={7\sqrt7+10\over36h}\simeq{0.79225\over h}.
\]

For comparison, the two-baseline formula in Section 16.2 is verified by the
three weights `(1/3,-2,2/3)` at `(-1,-1/2,1/2)`: their moments of orders
`0,1,2,3,4` are respectively `-1,1,0,0,1/4`.  Adding `Phi(U)` and subtracting
`h^4 Phi(V)/4` therefore leaves exactly `4 c h T(U,U,U,V)`.

## 17. Ordinary fourth Taylor degree is not Wiener/Hermite H4

Let `Z` be standard Gaussian and expand a sufficiently regular scalar response
in factorial-normalized multivariate probabilists' Hermites,

\[
F(x)=\sum_{n\ge0}{1\over n!}
\langle K_n,\operatorname{He}_n(x)\rangle,
\]

with symmetric coefficient tensors `K_n`.  Let `Tr` contract two tensor slots
with the Gaussian covariance (replace it by `Tr_Sigma` off the standardized
frame).  Normal ordering gives the exact triangular identity

\[
\boxed{
D^4F(0)=K_4-{1\over2}\operatorname{Tr}K_6
+{1\over8}\operatorname{Tr}^2K_8
-{1\over48}\operatorname{Tr}^3K_{10}+\cdots .}
\]

Thus, in the required mixed direction,

\[
D^4F(0)[U,U,U,V]
=K_4[U,U,U,V]
-\frac12(\operatorname{Tr}K_6)[U,U,U,V]
+\frac18(\operatorname{Tr}^2K_8)[U,U,U,V]-\cdots.
\]

The coefficient of the `q`th term is exactly
`(-1)^q/(2^q q!)`.  It follows directly from

\[
\operatorname{He}_{4+2q}(x)
\supset {(-1)^q(4+2q)!\over2^q q!4!}\,x^{\otimes4}
\]

after the outside `1/(4+2q)!` normalization.  In the unnormalized scalar
convention this is the familiar

\[
24a_4-360a_6+5040a_8-\cdots
=24(a_4-15a_6+210a_8-\cdots).
\]

For a response certified to stop at Hermite order eight, two independent
counterterm scalars are generically necessary and sufficient:

\[
C_6=(\operatorname{Tr}K_6)[U,U,U,V],\qquad
C_8=(\operatorname{Tr}^2K_8)[U,U,U,V],
\]

and

\[
K_4[U,U,U,V]=D^4F(0)[U,U,U,V]+\tfrac12C_6-\tfrac18C_8.
\]

Equivalently, if ordinary derivative contractions rather than Hermite
counterterms are exposed, triangular inversion gives

\[
\boxed{
K_4[U,U,U,V]
=D^4F(0)[U,U,U,V]
+\tfrac12(\operatorname{Tr}D^6F(0))[U,U,U,V]
+\tfrac18(\operatorname{Tr}^2D^8F(0))[U,U,U,V].}
\]

The two extra scalars are minimal without a relation between `K_6` and `K_8`,
because their contracted actions can vary independently while preserving the
ordinary fourth derivative.  For an unrestricted ReLU Hermite law the inverse
is instead the infinite series

\[
K_4=\sum_{q\ge0}{1\over2^q q!}
\operatorname{Tr}^qD^{4+2q}F(0),
\]

whenever the Gaussian-smoothed response makes this series meaningful.  An
unsmoothed ReLU is not four-times differentiable at its kink, so it supplies no
ordinary fourth-derivative oracle there.

No finite real point-polarization stencil can be a universal Hermite-H4
projector for a general ReLU law.  If nodes `x_i` and weights `lambda_i` did so
in one dimension, then

\[
\sum_i\lambda_i\operatorname{He}_n(x_i)=0\quad(n\ne4)
\]

with a nonzero value at `n=4`.  The Hermite exponential generating function
would require

\[
e^{-z^2/2}\sum_i\lambda_i e^{x_i z}=Cz^4,
\qquad\text{or}\qquad
\sum_i\lambda_i e^{x_i z}=Cz^4e^{z^2/2}.
\]

The left side is a finite exponential polynomial of entire-function order at
most one; the nonzero right side has order two.  This is impossible.  Hence a
finite stencil either assumes an exactly quartic potential, measures a
contaminated ordinary derivative/secant, or requires a genuine chaos-projection
oracle.  It cannot manufacture the connected K31/H4 action from a general
ReLU response by polarization alone.

## 18. Remainder certificate for the Hermite-to-Taylor trace inversion

Continue with factorial-normalized Hermite tensors and let `Tr_Sigma` denote
pair contraction with the Gaussian covariance.  From Section 17,

\[
D^4F(0)=\sum_{q\ge0}{(-1)^q\over2^q q!}
\operatorname{Tr}_{\Sigma}^{q}K_{4+2q}.
\]

If the direct Hermite counterterms through order `4+2Q` are supplied, define

\[
\widehat K_{4,Q}=D^4F(0)-
\sum_{q=1}^{Q}{(-1)^q\over2^q q!}
\operatorname{Tr}_{\Sigma}^{q}K_{4+2q}.
\]

The exact remainder is

\[
\boxed{
\widehat K_{4,Q}-K_4=
\sum_{q>Q}{(-1)^q\over2^q q!}
\operatorname{Tr}_{\Sigma}^{q}K_{4+2q}.}
\]

In particular, after the `K6,K8` counterterms,

\[
R_2=-{1\over48}\operatorname{Tr}_{\Sigma}^3K_{10}
+{1\over384}\operatorname{Tr}_{\Sigma}^4K_{12}
-{1\over3840}\operatorname{Tr}_{\Sigma}^5K_{14}+\cdots.
\]

The corresponding formula for ordinary-derivative inversion has the same norm
bound: truncate
`K4=sum_q Tr_Sigma^q D^(4+2q)F(0)/(2^q q!)` after `q=Q` and bound its positive
tail.

### 18.1 Exact operator- and Hilbert--Schmidt-norm bounds

For the injective tensor norm,

\[
\|\operatorname{Tr}_{\Sigma}^{q}A\|_{\rm op}
\le \|\Sigma\|_*^q\|A\|_{\rm op},
\]

where `||Sigma||_*` is the nuclear norm.  For Hilbert--Schmidt tensors,

\[
\|\operatorname{Tr}_{\Sigma}^{q}A\|_{\rm HS}
\le \|\Sigma\|_{\rm HS}^q\|A\|_{\rm HS}.
\]

Therefore

\[
\boxed{
\|R_Q\|_{\rm op}
\le\sum_{q>Q}{\|\Sigma\|_*^q\|K_{4+2q}\|_{\rm op}\over2^q q!},
\qquad
\|R_Q\|_{\rm HS}
\le\sum_{q>Q}{\|\Sigma\|_{\rm HS}^q\|K_{4+2q}\|_{\rm HS}\over2^q q!}.}
\]

These triangle bounds are sharp without phase or orthogonality information.
For the mixed endpoint action, the HS dual tensor can be symmetrized.  If
`rho_UV=<U,V>/(||U||||V||)`, then

\[
s(U,V):=\|\operatorname{Sym}(U^{\otimes3}\otimes V)\|_{\rm HS}
={\|U\|^3\|V\|\over2}\sqrt{1+3\rho_{UV}^2},
\]

so `|R_Q[U,U,U,V]|<=s(U,V)||R_Q||_HS`.  The operator version uses the simpler
factor `||U||^3||V||`.

One convenient sufficient operator envelope is

\[
\|K_{4+2q}\|_{\rm op}\le Mq!R^{-q}.
\]

If `theta=||Sigma||_*/(2R)<1`, it gives

\[
\boxed{\|R_Q\|_{\rm op}\le
M{\theta^{Q+1}\over1-\theta}.}
\]

This is not a consequence of ordinary local analyticity; it is an additional
Gaussian heat-analytic/Gevrey envelope.

### 18.2 A lawful weighted-Fock certificate

Define the measured analytic-tail norm

\[
S_{\rho,Q}^2=
\sum_{q>Q}{\rho^{4+2q}\|K_{4+2q}\|_{\rm HS}^2\over(4+2q)!}.
\]

Cauchy--Schwarz gives

\[
\boxed{
\|R_Q\|_{\rm HS}\le {S_{\rho,Q}\over\rho^2}
\sqrt{G_Q\!\left({\|\Sigma\|_{\rm HS}^2\over\rho^2}\right)},}
\]

where

\[
G_Q(x)=\sum_{q>Q}{(2q+4)!\over4^q(q!)^2}x^q.
\]

This series converges exactly for `x<1`; thus a finite certificate requires
`rho>||Sigma||_HS`.  Its full generating function can be written as

\[
(2x\partial_x+1)(2x\partial_x+2)
(2x\partial_x+3)(2x\partial_x+4)(1-x)^{-1/2},
\]

with the first `Q+1` terms removed.

An Ornstein--Uhlenbeck smoothing receipt supplies such a norm.  If
`F=P_rG`, then `K_n(F)=r^nK_n(G)` and
`S_{rho,Q}<=||G||_(L2(gamma))` whenever `rho r^2<=1`.  Combining this with
the convergence condition requires the quantitative smoothing margin

\[
r^2\|\Sigma\|_{\rm HS}<1.
\]

Merely saying "Gaussian smoothed" without certifying this margin and the
weighted norm does not establish the bound.

#### Exact scalar OU-smoothed ReLU special case

There is a sharper closed form when the response is certified to be one
rank-one scalar normalized-ReLU channel `P_r psi`, rather than a general AQ
mixture.  Section 1 gives

\[
K_{4+2q}=(4+2q)!a_{4+2q}r^{4+2q}
={(-1)^{q+1}(2q+1)!!\over\sqrt\pi},r^{4+2q}.
\]

After multiplication by the alternating trace coefficient, every omitted
term has the same sign.  Therefore the direct-counterterm remainder after
`q=Q` has the exact absolute value

\[
\boxed{
T_Q^{\rm ReLU}(r)={r^4\over\sqrt\pi}
\sum_{q>Q}(2q+1){\binom{2q}{q}\over4^q}r^{2q},
\qquad 0\le r<1.}
\]

Since

\[
\sum_{q\ge0}(2q+1){\binom{2q}{q}\over4^q}x^q=(1-x)^{-3/2},
\]

the `K6,K8` remainder is exactly

\[
\boxed{
T_2^{\rm ReLU}(r)={r^4\over\sqrt\pi}
\left[(1-r^2)^{-3/2}-1-{3\over2}r^2-{15\over8}r^4\right].}
\]

In particular

\[
T_2^{\rm ReLU}(r)\ge {35\over16\sqrt\pi}r^{10}.
\]

Thus even this most favorable exact scalar receipt needs
`sum_l |beta_l| T_2^ReLU(r_l)<=delta`.  With ten unit-weight adversarial
layers, the necessary first-omitted-term condition is

\[
r_l\le\left({16\sqrt\pi\over35},3.80\times10^{-7}\right)^{1/10}
\simeq0.223.
\]

This scalar formula cannot be transferred to a multi-route AQ response unless
the rank-one reduction, each OU correlation `r_l`, and the endpoint route
weights are certified.  Otherwise the contracted trace spectrum remains the
minimal lawful observable.

### 18.3 D10 endpoint gate

For late layers `l=5,...,14`, endpoint coefficients `beta_l`, and mixed
directions `U_l,V_l`, a lawful deterministic gate is

\[
\boxed{
\sum_{l=5}^{14}|\beta_l|s(U_l,V_l)
{S_{\rho_l,Q,l}\over\rho_l^2}
\sqrt{G_Q\!\left({\|\Sigma_l\|_{\rm HS}^2\over\rho_l^2}\right)}
\le3.80\times10^{-6}.}
\]

The direct trace-spectrum bound in Section 18.1 may be substituted layer by
layer and is sharper.  Under the deliberately crude normalization
`|beta_l|=||U_l||=||V_l||=1` with ten coherently aligned layer errors, equal
allocation requires at most `3.80e-7` per layer.  If `K10` is the only omitted
term, this means

\[
\|(\operatorname{Tr}_{\Sigma_l}^3K_{10,l})[U_l,U_l,U_l,V_l]\|
\le48(3.80\times10^{-7})=1.824\times10^{-5}
\]

per layer.  For an already aggregated D10 oracle, the corresponding total
`K10` action allowance is `48 delta=1.824e-4`; one must not divide by ten
twice.

### 18.4 Sharp no-go from the present premises

No finite bound follows from knowledge through `K8`, ordinary analyticity, or
an unweighted Gaussian `L2` tail.  In one dimension consider

\[
F_M(x)={M/48\over4!}\operatorname{He}_4(x)
+{M\over10!}\operatorname{He}_{10}(x).
\]

It is an entire polynomial with `K6=K8=0` and `D^4F_M(0)=0`, yet its true
`K4=M/48` is arbitrarily large.  Thus even `D4,K6,K8` together do not identify
or bound the omitted trace.

Even a fixed `L2(gamma)` norm is insufficient.  Put all tail energy at
`n=4+2q` with `K_n=S sqrt(n!)`.  Its Gaussian `L2` norm is `S`, while its
contribution to the trace remainder in one dimension is

\[
{S\sqrt{(2q+4)!}\over2^q q!}
\sim4\pi^{-1/4}S q^{7/4},
\]

which is unbounded.  Hence raw Hermite-tail variance, H6/H8 faces, and local
analyticity cannot certify the strict endpoint gate.

The minimal missing observable is the endpoint-weighted contracted trace
spectrum

\[
\tau_{q,l}=\|\operatorname{Tr}_{\Sigma_l}^qK_{4+2q,l}\|_{\rm op/HS}
\]

together with a certified tail sum

\[
\sum_l|\beta_l|s_l\sum_{q>Q}{\tau_{q,l}\over2^q q!}
\le3.80\times10^{-6}.
\]

A certified pair `(rho,S_rho)` satisfying the weighted-Fock condition is a
stronger but sufficient substitute.  Without one of these observables, or a
direct contract-native Hermite-H4 projection, finite K6/K8 correction remains
uncertified.

## 19. Minimal tail-free Gaussian-score / OU oracle

Let `Z` be an isonormal standard Gaussian on a real Hilbert space and retain
the factorial Hermite convention

\[
F(Z)=\sum_{n\ge0}{1\over n!}\langle K_n,\operatorname{He}_n(Z)\rangle.
\]

For fixed Cameron--Martin directions `U,V`, put

\[
A=\langle U,Z\rangle,\quad B=\langle V,Z\rangle,
\quad a=\|U\|^2,\quad c=\langle U,V\rangle.
\]

The mixed fourth Gaussian score is the Wick polynomial

\[
H_{31}(Z;U,V)=:A^3B:
=A^3B-3aAB-3cA^2+3ac.
\]

Orthogonality of Wiener chaoses gives the exact one-scalar oracle

\[
\boxed{
\mathfrak O_{31}(F;U,V)
:=\mathbb E[F(Z)H_{31}(Z;U,V)]
=K_4[U,U,U,V].}
\]

It annihilates every chaos except H4, including all H6/H8 contractions, so it
has no trace-inversion remainder.  Equivalently, by Gaussian integration by
parts,

\[
\mathfrak O_{31}(F;U,V)
=\mathbb E[D_U^3D_VF(Z)]
=\left.\partial_s^3\partial_t
\mathbb E F(Z+sU+tV)\right|_{s=t=0},
\]

whenever the derivatives exist; the score definition remains valid for ReLU
without classical derivatives.  For covariance `Sigma`, use its isonormal
inner product, or whiten first.  In physical shift coordinates the score uses
`Sigma^{-1}U,Sigma^{-1}V`.

Information-theoretically one fused score expectation is minimal.  If an API
exposes only raw moments, exactly four generally nonzero surface moments
suffice:

\[
m_{31}=E[FA^3B],\quad m_{11}=E[FAB],\quad
m_{20}=E[FA^2],\quad m_0=E[F],
\]

with `O31=m31-3a m11-3c m20+3ac m0`.  These four raw monomials are linearly
independent, so fewer cannot work uniformly without centering, orthogonality,
or a fused-score primitive.

### 19.1 OU/correlated-Gaussian pullback and exact JVP

Let

\[
Y=RX+(I-RR^*)^{1/2}G,
\qquad (P_RF)(x)=E[F(Y)\mid X=x],
\]

with independent standard `X,G`.  Second quantization acts diagonally on
chaos, hence

\[
K_4(P_RF)=(R^*)^{\otimes4}K_4(F)
\]

and therefore

\[
\boxed{
\mathfrak O_{31}(P_RF;U,V)
=K_4(F)[RU,RU,RU,RV].}
\]

Writing `u=RU`, `v=RV`, its JVP in direction `dot R` is

\[
\boxed{
\delta\mathfrak O_{31}
=3K_4[u,u,(\dot R)U,v]+K_4[u,u,u,(\dot R)V].}
\]

If `g_u=K_4[u,u,\cdot,v]` and `g_v=K_4[u,u,u,\cdot]`, the exact reverse
pullback is

\[
\boxed{\nabla_R\mathfrak O_{31}=3g_u\otimes U^\flat+g_v\otimes V^\flat.}
\]

For a scalar correlation `rho`, this reduces to multiplication by `rho^4`
and JVP `4 rho^3 dot rho`.

For the normalized ReLU `psi(z)=sqrt(2) z_+`, `K_4(psi)=-1/sqrt(pi)`.  If
`Y` is standard and has correlation vector `r` with the core Gaussian, then

\[
\boxed{
\mathfrak O_{31}(P_r\psi;U,V)
=-{(r\cdot U)^3(r\cdot V)\over\sqrt\pi}.}
\]

Its JVP and reverse gradient with respect to `r` are

\[
-{3(r\cdot U)^2(\dot r\cdot U)(r\cdot V)
+(r\cdot U)^3(\dot r\cdot V)\over\sqrt\pi},
\]

and

\[
-{3(r\cdot U)^2(r\cdot V)U+(r\cdot U)^3V\over\sqrt\pi},
\]

respectively.  For two scalar correlated ReLUs, the H4 kernel slice and its
JVP are

\[
\kappa_4(\rho)={\rho^4\over24\pi},
\qquad D\kappa_4(\rho)[\dot\rho]
={\rho^3\dot\rho\over6\pi}.
\]

These are H4-specific.  Pulling a generic future loss through the full ReLU
kernel uses

\[
\kappa'(\rho)={\pi-\arccos\rho\over\pi},
\]

and is not replaced by the scalar H4 multiplier unless chaos-diagonal closure
has been certified.

### 19.2 Certified finite-rank AQ core and ReLU surface moments

Suppose the complete future response is measurable with respect to an
`r`-dimensional whitened core `Y`; write it as `g(Y)`.  Gaussian conditional
expectation is second quantization of the orthogonal projection, so only the
projected directions `alpha=P U`, `beta=P V` survive:

\[
\mathfrak O_{31}(F;U,V)
=E[g(Y):\langle\alpha,Y\rangle^3
\langle\beta,Y\rangle:].
\]

Thus one fused scalar, or the same four raw moments, is sufficient for a fixed
mixed action once **complete future measurability in the core** is certified.
Finite core rank by itself is not that certificate.

For an explicit affine--quadratic preactivation

\[
q(Y)=q_0+b^TY+\tfrac12Y^TCY,
\qquad F(Y)=q(Y)_+,
\]

Gaussian integration by parts gives a finite exact surface representation.
Put

\[
q_U=D_Uq,\quad q_V=D_Vq,\quad
q_{UU}=U^TCU,\quad q_{UV}=U^TCV.
\]

Then, in the distributional sense,

\[
\boxed{
\begin{aligned}
\mathfrak O_{31}(q_+;U,V)
={}&E[\delta''(q)q_U^3q_V]\\
&+3q_{UU}E[\delta'(q)q_Uq_V]
+3q_{UV}E[\delta'(q)q_U^2]
+3q_{UU}q_{UV}E[\delta(q)].
\end{aligned}}
\]

Multiply by `sqrt(2)` for the normalized ReLU.  Hence four raw surface
moments, or one fused surface functional, are generically necessary and
sufficient for one AQ-ReLU action.  If `C=0`, only the `delta''` term remains.

The exact law is determined by the finite AQ parameters.  For example,

\[
E[e^{itq(Y)}]
=e^{itq_0}\det(I-itC)^{-1/2}
\exp\left[-{t^2\over2}b^T(I-itC)^{-1}b\right],
\]

and the four surface moments are Fourier integrals obtained by source
differentiation.  This is finite-dimensional information, but exact surface
evaluation is not a fixed finite arithmetic operation and is not free merely
because `r` is finite.

Under the lane's centered AQ convention

\[
g(z)=c+b^Tz+Q_M(z),\qquad
Q_M(z)=\tfrac12(z^TMz-\operatorname{tr}M),
\]

the preceding formula uses `q0=c-tr(M)/2`.  Define

\[
S_t=(I-itM)^{-1},\quad
\mu_t=itS_tb,\quad m_t=b+M\mu_t,\quad C_t=MS_tM,
\]

and

\[
\chi(t)=e^{it(c-\operatorname{tr}M/2)}
\det(I-itM)^{-1/2}
\exp[-\tfrac12t^2b^TS_tb].
\]

For any symmetric fourth tensor `H`, the audited general surface identity is

\[
\boxed{
K_4:H=E[\delta''(g)H[p,p,p,p]
+6\delta'(g)H[M,p,p]+3\delta(g)H[M,M]],
\qquad p=b+Mz.}
\]

Here `H[M,p,p]` contracts one pair against `M`, and `H[M,M]` contracts two
pairs.  For `H=Sym(U^3 tensor V)`, the factors `6,3` reduce exactly to the
mixed coefficients `3,3,3` displayed above.

The Fourier reduction is a single real-line integral:

\[
\boxed{
\begin{aligned}
K_4:H={1\over2\pi}\int_{\mathbb R}\chi(t)\{&(it)^2[
H[m_t^4]+6H[C_t,m_t,m_t]+3H[C_t,C_t]]\\
&+6it[H[M,m_t,m_t]+H[M,C_t]]+3H[M,M]\}\,dt .
\end{aligned}}
\]

The identity follows because the complex Gaussian tilt has mean `mu_t` and
covariance `S_t`, while
`delta^(j)(g)=(2pi)^(-1) integral (it)^j exp(itg) dt`.  A Gaussian regulator
may be inserted and removed distributionally when the surface density is not
classically smooth.

For the mixed action alone, let

\[
d_U(t)=U^Tm_t,\quad d_V(t)=V^Tm_t,\quad
s_{UU}(t)=U^TC_tU,\quad s_{UV}(t)=U^TC_tV.
\]

Then the only tilted moment of order four is

\[
E_t[g_U^3g_V]=d_U^3d_V+3s_{UU}d_Ud_V
+3s_{UV}d_U^2+3s_{UU}s_{UV},
\]

with `E_t[g_Ug_V]=d_Ud_V+s_UV` and
`E_t[g_U^2]=d_U^2+s_UU`.  This explicitly audits every coefficient in the
four mixed surface moments.

For fixed `U,V`, the exact JVP with respect to the AQ parameters is simpler in
score form.  If

\[
\dot g=\dot c+\dot b^Tz+Q_{\dot M}(z),
\]

then

\[
\boxed{D\mathfrak O_{31}[\dot c,\dot b,\dot M]
=E[1_{\{g>0\}}\dot g\,H_{31}(Z;U,V)].}
\]

Thus the exact parameter pullbacks are

\[
\partial_c\mathfrak O_{31}=E[1_{g>0}H_{31}],\quad
\nabla_b\mathfrak O_{31}=E[1_{g>0}ZH_{31}],
\]

\[
\nabla_M\mathfrak O_{31}
=\tfrac12E[1_{g>0}(ZZ^T-I)H_{31}],
\]

before any dependence of `U,V` or the covariance frame is added.  These are
finite local volume moments, but propagating their adjoints through future
layers is exactly the cyclic-module problem below.

For one fused core-score oracle, the new local state is two projected
directions (`2r` scalars), the two Gram scalars `a,c`, and one accumulator:
`2r+3` scalars.  Four exposed raw moments use `2r+6`.  For AQ surface synthesis
one additionally needs `CU,CV` and `q_UU,q_UV`; dense formation costs two
`r`-dimensional matvecs, `Theta(r^2)` arithmetic, before surface integration.
An eigenspectral implementation retains `Theta(r)` eigenvalues and direction
coordinates after a `Theta(r^3)` generic factorization.  These are local
algebraic costs, not a receipt that the current contract exposes the oracle.

If `rank(M)=k`, a fixed mixed action needs only the `k` nonzero eigenvalues
and the coordinates of `b,U,V` in their eigenbasis, plus
`||b_perp||^2` and the required perpendicular inner products.  Hence its local
spectral information is `Theta(k)` and each Fourier-integrand evaluation is
`Theta(k)` after factorization.  A generic fourth tensor `H`, in contrast, can
require `Theta(k^4)` eigenbasis contractions; the cheap count is special to
the rank-one mixed tensor `Sym(U^3 tensor V)`.

The AQ family is closed under a deterministic linear pullback `z=Rx`:

\[
b'=R^Tb,\quad M'=R^TMR,\quad
c'=c+\tfrac12(\operatorname{tr}M'-\operatorname{tr}M).
\]

It is not generically closed under a noisy correlated transition followed by
conditional expectation.  If
`Y=RX+(I-RR^*)^(1/2)G`, then
`E[g(Y)_+|X=x]` is a Gaussian-smoothed threshold response, not one ReLU of a
fixed AQ form, unless the noise is null in every direction used by `b,M` (or
another explicit degeneracy holds).  With nondegenerate smoothing it is
analytic across the old surface, whereas a nontrivial AQ-ReLU retains a kink;
therefore equality is impossible.  The H4 tensor itself still obeys the exact
OU pullback in Section 19.1, but the complete future response needed for its
endpoint value leaves the finite AQ family.

### 19.3 Future cyclic-module obstruction

Let `L_l` be the forward conditional/ReLU operator from core layer `l` to
`l+1`, and let `G_L` be the terminal score functional.  A finite family of
surface moments `S_l` is an exact deployable state if and only if

\[
L_l^*S_{l+1}\subseteq S_l
\]

at every layer and every source/reset functional lies in the corresponding
`S_l`.  The minimal exact state dimension is the rank of the future cyclic
module

\[
\boxed{
\mathcal C(G_L)=\operatorname{span}
\{L_l^*L_{l+1}^*\cdots L_{L-1}^*G_L
\text{ and all reset descendants}\}.}
\]

Proof is linear-algebraic: the stored observations define a quotient of the
future-response space; exact recovery is possible precisely when every needed
adjoint functional annihilates its kernel.  Otherwise two future responses
agree on all stored moments but have different endpoint K31 actions.

Finite variable rank does not imply finite cyclic rank.  Already in one
dimension, ReLU pullbacks generate delta derivatives at future thresholds;
distinct thresholds and derivative orders are linearly independent
distributions.  Equivalently, repeated multiplication by the nonpolynomial
`kappa'(rho)` produces an unbounded function algebra unless a special
polynomial relation is certified.  Thus a local four-surface-moment AQ oracle
does not avoid carrying the full future cyclic module.

Operationally, a reverse pass with all cores retained can use `O(r)` working
state per step plus the stored future cores.  A one-pass forward contract must
carry at least `dim C(G_L)` eligibility coordinates; if that module is the
full matrix/function space, the Gaussian-score identity has not reduced the
carrier cost.  The tail-free oracle is contract-native only if the score
expectation itself is exposed or the adjoint module closes in the billed
finite state.

## 20. Strict-depth true-minus-assembled specialization

Let `V_<` be the collision-free primitive quotient, let `F` be the complete
incumbent D11/Q24 feature map, and retain only strict blocks `j<k`.  For one
dense reset `K` and frozen directed Bellman row `A`, the score row is

\[
d_{K,A}=P_{V_<}\Gamma_4(K)^*S_{31}(A),
\qquad
S_{31}(A)=\sum_{ij}A_{ij}\operatorname{Sym}(e_i^{\otimes3}\otimes e_j).
\]

For parity/all-distinct basis directions `T_s`, its observable coordinates are
`d_s=<d_(K,A),T_s>`.  A family of post-reset K31 rows can give an `11 by 11`
matrix `(d_(a,s))` of generic rank eleven.  One frozen Bellman row gives only
a `1 by 11` row and therefore observes one target-specific linear combination,
not eleven coordinates injectively.  This is sufficient for a frozen scalar
endpoint only if the row is present on the law side before the charged cut.

The lawful score is the incumbent-residual strict-depth functional

\[
D_k^<(\mu)=
\left\langle A_k,
\sum_{j<k}(H^{\rm true}_{kj}-H^{\rm asm}_{kj})b_j
\right\rangle,
\]

not total same-site K31.  The `j=k` action is projected into the incumbent
span and receives zero novelty credit.  Acquiring each `j<k` term separately
is still the forbidden quadratic-in-depth reconstruction; the fresh oracle
must expose the aggregate consumer-layer residual.

### 20.1 Exact row-space and error gates

Let `P_perp` residualize against the incumbent feature span on the reachable
birth quotient, and let `h_k` be the frozen strict-depth endpoint row.  For a
score interface with row matrix `D_k`, exact target recovery is

\[
\boxed{
h_kP_\perp\in\operatorname{row}(D_kP_\perp)
\quad\Longleftrightarrow\quad
\ker(D_kP_\perp)\subseteq\ker(h_kP_\perp).}
\]

For one fused scalar `d_k`, this reduces to
`h_kP_perp=alpha d_kP_perp`.  If the exact Bellman row is fused into the true
minus assembled score, `alpha=1` and the Node-N3 fiber condition holds.  If
the row arrives only after the charged cut, a target-blind interface
generically needs eleven independent coordinates instead.

Let `Sigma_(perp F)` be the incumbent-conditional birth covariance.  For an
approximate row `d_tilde`, the exact per-layer RMS action error is

\[
\epsilon_k^2=(d_k-\widetilde d_k)
\Sigma_{\perp F,k}(d_k-\widetilde d_k)^*.
\]

For the joint birth history, use the full joint conditional covariance, which
retains every cross-layer sign.  Without that Gram, the lawful deterministic
bound is the coherent sum `eta<=sum_k epsilon_k`.  It must fit the applicable
endpoint allocation, never more than `3.80e-6` and possibly the narrower
delayed-K31 gate.

### 20.2 Bellman matrix rank is not directional-score rank

Let `w_i` be the reset rows.  The frozen Bellman action polynomial is

\[
P_{K,A}(z)=\sum_{ij}A_{ij}(w_i\cdot z)^3(w_j\cdot z),
\]

with Wick lower contractions understood.  One directional
`O31(U,U,U,V)` call represents
`Sym(U^(tensor 3) tensor V)`.  The exact number of unrestricted directional
calls is therefore the residual Segre--Veronese rank

\[
\boxed{
r_{31}^{\rm dir}(K,A)=\min\left\{r:
P_\perp\sum_{ij}A_{ij}\operatorname{Sym}(w_i^{\otimes3}\otimes w_j)
=P_\perp\sum_{a=1}^r c_a
\operatorname{Sym}(u_a^{\otimes3}\otimes v_a)
\right\}.}
\]

It is unrelated in general to `rank(A)`.  If `A=xy^T`, then

\[
P_{K,A}(z)=C_x(z)L_y(z),\qquad
C_x(z)=\sum_i x_i(w_i\cdot z)^3,quad
L_y(z)=\sum_jy_j(w_j\cdot z).
\]

Keeping the common singleton `L_y`, the exact call count is the cubic Waring
rank of `C_x`.  For invertible `K` and generic nonzero `x`, `C_x` is a linear
change of a diagonal/Fermat cubic and has Waring rank `n`: its first
catalecticant has rank `n`, and the displayed `n` cubes attain the bound.
Thus a rank-one Bellman matrix is generically not one O31 direction.  Allowing
different `v_a` can be credited only after proving a smaller residual
Segre--Veronese rank in the boxed identity.

If `A=sum_(b=1)^R x_b y_b^T`, the common-singleton construction uses

\[
r_C=\sum_{b=1}^R\operatorname{wrank}(C_{x_b})
\]

cubic projections and at most `r_L=R` singleton projections, before sharing
or algebraic cancellation.

### 20.3 Exact charged cost for one frozen Bellman row

If the fresh law exposes `N_k` weighted atoms and accepts the general dense
row `A_k`, the fused scalar is

\[
\phi_k=\sum_sp_s h_3(z_s)^TA_kz_s.
\]

One `N_k by n` times `n by n` rectangular multiply costs `N_k/n`
square-GEMM equivalents under the lane billing convention; the score
contraction and assembled-scalar subtraction are lower order.  Hence

\[
\boxed{G_k^{\rm row}=G_{A_k,\rm new}+{N_k\over n}.}
\]

Here `G_(A_k,new)=0` only when the fully H3-resummed future Bellman row and its
law kernel are already lawfully available pre-cut.  A new dense reverse
segment of depth `d` costs `2d` conjugation GEMMs unless shared; the shared
fourteen-transition covariance costate costs 28 across the chain.

If the contract exposes only directional score calls, shared projection
billing gives

\[
\boxed{G_k^{\rm dir}=G_{\rm factors,new}
+{N_k(r_C+r_L)\over n^2},}
\]

or the conservative `2N_k r/n^2` when no factors are shared.  For generic
rank-one `A=xy^T`, `r_C=n,r_L=1`; at `N_k=n` the cost is approximately one
GEMM only because all `n` cubic projections are batched and the singleton is
shared, not because there is one directional oracle call.

Therefore the Gaussian score solves Node N3 locally exactly when the frozen
strict-depth Bellman row is already pre-cut and the aggregate true-minus-
assembled law is exposed.  It does not transport or manufacture that future
row.  Without a shared reverse costate or a proved finite adjoint module, the
full future cyclic-module cost remains binding.

## 21. Endpoint-compatible multi-residual q211 certificate

This section sharpens the uniform bound
`4(6 A^2 E^2+4 A E^3+E^4)` from cubic Sections 68--69.  It compiles the
complete-old projection first, separates the affine-Gaussian higher-chaos
rest from the true-minus-AQ residual, and uses only directional fourth norms
that a factorized AQ/D10 interface could expose.

### 21.1 Once-only ownership and synchronized decomposition

Move collision removal and complete-old residualization to the reverse side:

\[
\widetilde\Lambda
=\Pi_{\rm off}^{*}{\cal R}_{\rm old}^{*}\Lambda .
\]

All bounds below use this `tilde Lambda`; applying an old projector again on
the source would double subtract.  Assume the true and AQ worlds have a
literal synchronized degree-at-most-two projection

\[
X=P_{\le2}Y^{\rm true}=P_{\le2}Y^{\rm AQ},
\qquad
r=Y^{\rm AQ}-X,
\qquad
d=Y^{\rm true}-Y^{\rm AQ}.
\]

Then, in the common Gaussian frame,

\[
E[X_i r_j]=E[X_i d_j]=0\qquad\text{for every }i,j.
\tag{21.1}
\]

Here `r` is the exactly parameterized affine-Gaussian ReLU rest and `d` is
the genuinely true-minus-AQ deformation.  If the full AQ surface is actually
evaluated and charged, every pure-`r` q211 term is owned and subtracted.  The
new source contains at least one `d`; the present section bounds its terms
with at least **two** `d` slots.  If synchronization (21.1) is not proved,
the sharpened constants below are invalid and every cumulant must use the
generic factor four.

Let an exact complete-old query factorization be

\[
\widetilde\Lambda
=\sum_{q=1}^{R}c_q\,
\operatorname{Sym}(u_q\otimes u_q\otimes v_q\otimes w_q),
\tag{21.2}
\]

with every diagram normalization absorbed in `c_q`.  This is an information
formula, not a free rank receipt: if `R` is large, the directional moment
interface is correspondingly expensive.  A coordinatewise version follows
by taking `(u,v,w)=(e_i,e_j,e_k)` and summing
`|tilde Lambda_(i;jk)|`.

For one ray suppress `q` and set

\[
x_s=\|s^TX\|_4,qquad
a_s=\|s^Tr\|_4,qquad
e_s=\|s^Td\|_4,qquad s\in\{u,v,w\}.
\]

### 21.2 Exact arity polynomials and improved constants

Define the two-deformation polynomials according to how many of the two
remaining comparator slots use `X` or `r`:

\[
\begin{aligned}
P_{2,XX}={}&e_u^2x_vx_w
+2e_ue_vx_ux_w+2e_ue_wx_ux_v+x_u^2e_ve_w,\\
P_{2,rr}={}&e_u^2a_va_w
+2e_ue_va_ua_w+2e_ue_wa_ua_v+a_u^2e_ve_w,\\
P_{2,Xr}={}&e_u^2(x_va_w+a_vx_w)\\
&+2e_ue_v(x_ua_w+a_ux_w)
+2e_ue_w(x_ua_v+a_ux_v)
+2x_ua_u e_ve_w.
\end{aligned}
\tag{21.3}
\]

The three- and four-deformation polynomials are

\[
\begin{aligned}
P_{3,X}&=e_u^2(e_vx_w+x_ve_w)+2e_ux_ue_ve_w,\\
P_{3,r}&=e_u^2(e_va_w+a_ve_w)+2e_ua_ue_ve_w,\\
P_4&=e_u^2e_ve_w.
\end{aligned}
\tag{21.4}
\]

The ordinary centered fourth cumulant is a raw fourth moment minus three
pairings.  Under (21.1):

* a two-`d`, two-`X` term has only the `dd` times `XX` pairing, hence constant
  two rather than four;
* a two-`d`, one-`X`, one-`r` term has every pairing zero, hence constant one;
* a two-`d`, two-`r` term is generic, hence constant four;
* a three-`d`, one-`X` term is a raw moment (constant one), whereas a
  three-`d`, one-`r` term is generic (constant four);
* the four-`d` term retains all residual pairings (constant four).

Hölder therefore gives the per-ray certificate

\[
\boxed{
B_q^{(\ge2d)}
=2P_{2,XX}+P_{2,Xr}+4P_{2,rr}
+P_{3,X}+4P_{3,r}+4P_4.
}
\tag{21.5}
\]

This is sharp from marginal L4 information and the orthogonality statements:
each surviving raw moment/pairing can align adversarially.  Without (21.1),
the safe replacement is

\[
B_{q,\rm generic}^{(\ge2d)}
=4(P_{2,XX}+P_{2,Xr}+P_{2,rr}+P_{3,X}+P_{3,r}+P_4).
\tag{21.6}
\]

For the complete endpoint, the lawful coherent bound is

\[
\boxed{
\eta_{\ge2d}
\le
\left\|\sum_{q=1}^{R}|c_q|B_q^{(\ge2d)}\right\|_{L^2({\rm nets})}.
}
\tag{21.7}
\]

Absolute coefficients are necessary for an adversarial label-free radius;
crediting cancellation between signed costate factors requires the actual
joint signed surface, not marginal norms.  In the uniform specialization
`x_s<=X`, `a_s=0`, `e_s<=E`, (21.6) reduces to the older
`4(6 X^2E^2+4XE^3+E^4)` bound.  Equation (21.5) is strictly tighter whenever
the synchronized projection and AQ-rest split are available.

### 21.3 Quantities the AQ and D10 lanes can lawfully expose

The low AQ projection is closed.  A direction `s` mixes the packed AQ bank
into

\[
s^TX=h_s^Tz+Q_{M_s}(z),
\qquad Q_M=\tfrac12(z^TMz-\operatorname{tr}M),
\]

and hence

\[
\boxed{
x_s^4
=3(\|h_s\|^2+\tfrac12\operatorname{tr}M_s^2)^2
+3\operatorname{tr}M_s^4+12h_s^TM_s^2h_s.
}
\tag{21.8}
\]

For one AQ ReLU source, the tilted residual generator in the direct lane gives
the exact scalar `E[r_i^4]`.  For a dense physical direction, a joint
multi-source tilted generator is needed for equality.  The safe currently
factorable fallback is Minkowski,

\[
a_s\le\sum_i|s_i|\,\|r_i\|_4.
\tag{21.9}
\]

Thus the affine-Gaussian rest has a lawful, if potentially loose, white-box
radius.  D10 covariance alone does not own its connected q211 sign; a paid
full AQ surface owns and subtracts it.

For the genuinely non-Gaussian deformation, suppose D10 exposes a directional
residual covariance `Sigma_d`.  A sufficient additional directional kurtosis
receipt is

\[
\gamma_s
:={E[(s^Td)^4]\over(s^T\Sigma_d s)^2},
\qquad
\boxed{e_s=\gamma_s^{1/4}(s^T\Sigma_d s)^{1/2}.}
\tag{21.10}
\]

No finite `gamma_s` bound follows from covariance alone.  If the residual is
resolved in a common Gaussian Wiener frame, the finite Stein/Price bands give
only

\[
e_s\le
\sum_{m=3}^{6}3^{m/2}{s^T\Sigma_{d,m}s\}^{1/2}
+\tau_{s,>6}^{(4)},
\qquad
\tau_{s,>6}^{(4)}=\|s^Td_{>6}\|_4.
\tag{21.11}
\]

The first term uses quantities a band-resolved D10 interface could expose;
the last term is not present in the current finite branch.  A marginal-only
fallback is

\[
e_s\le\sum_i|s_i|\left(
\|Y_i^{\rm true}-EY_i^{\rm true}\|_4
+\|Y_i^{\rm AQ}-EY_i^{\rm AQ}\|_4\right),
\tag{21.12}
\]

with each physical fourth norm obtained from marginal K4 and covariance.
It is lawful but normally far too coarse to certify a microunit endpoint
allowance.

If an actual OU/Mehler suffix damps order `m` by at most `rho^m`, the omitted
tail in (21.11) has the useful L2-energy certificate

\[
\boxed{
\sum_{m\ge M}3^{m/2}\rho^m\sigma_{m,s}
\le{(\sqrt3\rho)^M\over\sqrt{1-3\rho^2}}
\left(\sum_{m\ge M}\sigma_{m,s}^2\right)^{1/2},
\quad \rho<1/\sqrt3.
}
\tag{21.13}
\]

This is lawful only with a literal chaos-diagonal Mehler contraction and the
band-energy receipt.  Ordinary covariance recursion, or a heuristic
``Gaussian rest'', does not supply either premise.

### 21.4 Why H3--H6 first-order completeness does not close this tail

The one-residual K31/q211 influence polynomial has degree at most six, so its
first variation sees only H3--H6.  With two residual slots, Wiener products
return high orders to the low AQ query.  Against an `H<=2` base, bands `m,n`
can contribute whenever

\[
|m-n|\le4
\quad\text{and the required parity matches}.
\tag{21.14}
\]

Thus H4-by-H6 and H4-by-H8 returns are real, as are infinitely many high--high
diagonal bands.  A cutoff at H6 misses, among others, `(3,7)`, `(4,8)`, and
the continuing `m` versus `m,m+2,m+4` families.  The exact AQ tilted L4
potential resums the pure affine-Gaussian versions of these returns.  It does
not bound a genuine non-Gaussian `d` tail.

The cubic H9 sign witness makes the obstruction distribution-free: two laws
can agree on AQ state, D10 H3--H6, every univariate and bivariate residual
observable, and the complete old faces while the q211 action flips sign.
Consequently neither the finite Stein branches nor a residual Gram imply a
threshold-small (21.7).

### 21.5 Endpoint and gain decision

Let `w_1` be the complete-old-projected signed one-residual response supplied
by the finite H3--H6 tilted/Price oracle, and let `eta_other` collect future
darkness, provenance, and arithmetic error.  Ignoring the multi-residual
source is safe for the strict raw endpoint only if

\[
\boxed{
\eta_{\ge2d}+\eta_{\rm other}\le3.80\times10^{-6}.
}
\tag{21.15}
\]

To certify the required quartic gain,

\[
\boxed{
(\|w_1\|_2-\eta_{\ge2d}-\eta_{\rm other})_+^2
\ge2.86\times10^{-10},
}
\tag{21.16}
\]

equivalently the bracketed amplitude must be at least
`1.69115e-5`.  A magnitude bound on the multi-residual term cannot itself
prove gain, because its signed alignment is unidentified.

The current state cannot evaluate (21.15): it has no directional
`gamma_s`/`tau_(s,>6)`, no literal joint signed surface, and no certified
small complete-old query rank `R`.  The H9 witness proves that this is an
information gap, not merely a loose inequality.  Therefore the present
finite Stein/Price stack neither proves the `3.80e-6` error gate nor the
`2.86e-10` gain gate.  Promotion requires one of:

1. the direct complete-old signed surface scalar for the frozen endpoint;
2. directional residual fourth-moment/kurtosis receipts plus a certified
   low-rank factorization (21.2) that makes (21.7) pass;
3. a literal Mehler contraction and band-energy tail satisfying (21.13),
   together with the same query-rank and synchronization receipts.

### Atomic graph nodes

* `Q211_MULTI_D_ENDPOINT_L4_BOUND`: equation (21.5), solved conditional on
  synchronization and directional L4 radii;
* `Q211_COMPLETE_OLD_QUERY_FACTOR`: equation (21.2), open rank/cost receipt;
* `AQ_HIGH_CHAOS_L4_OWNERSHIP`: equations (21.8)--(21.9), conditional on the
  tilted AQ source/interface; pure AQ surface is once-only old ownership;
* `TRUE_MINUS_AQ_DIRECTIONAL_KURTOSIS`: equation (21.10), open and absent in
  current D10;
* `Q211_H3_H6_TO_MULTI_TAIL`: first-order finite, nonlinear tail open by
  (21.14) and the H9 no-go;
* `Q211_MEHLER_MULTI_TAIL_GATE`: equation (21.13), conditional on literal
  `rho<1/sqrt(3)` and band energies;
* `Q211_MULTI_RESIDUAL_ENDPOINT_GATE`: equations (21.15)--(21.16), currently
  unproved; direct signed surface remains the information-minimal escape.

## 22. Directional true-minus-AQ fourth-moment receipt: exact jets, a finite-CF tail certificate, and the information boundary

This section attacks the missing quantity in (21.10) directly.  Fix one of
the distinct compiled directions

\[
 s\in {\cal S}:=\{u_q,v_q,w_q:1\le q\le R\},\qquad |{\cal S}|\le3R,
\]

and write

\[
 F_s=s^Td= s^T(Y^{\rm true}-Y^{\rm AQ}),\qquad EF_s=0,
 \qquad e_s=\|F_s\|_4.
\tag{22.1}
\]

All statements below are about the *same-lineage coupling* in (22.1).
Separate true and AQ marginals do not determine the law of their difference.

### 22.1 The information-minimal exact receipt is one signed fourth jet

Let

\[
 \Phi_s(t,u)=E\exp\{i[t s^TY^{\rm true}+u s^TY^{\rm AQ}]\},
 \qquad \phi_s(t)=\Phi_s(t,-t)=Ee^{itF_s}.
\tag{22.2}
\]

Then, whenever the fourth moment exists,

\[
 \boxed{e_s^4=\phi_s^{(4)}(0).}
\tag{22.3}
\]

Equivalently, for a joint cumulant generator `K_s` along the signed ray,

\[
 \boxed{e_s^4=K_s^{(4)}(0)+3\{K_s^{(2)}(0)\}^2.}
\tag{22.4}
\]

Thus a generator that accepts the signed true-minus-AQ tilt can return a
four-jet in one truncated-power-series evaluation; only one scalar receipt is
needed for each distinct `s`.  A true-only jet and an AQ-only jet do **not**
suffice, because (22.4) contains the mixed cumulants

\[
 \kappa_4(Y^{\rm true}-Y^{\rm AQ})
 =\kappa_{40}-4\kappa_{31}+6\kappa_{22}-4\kappa_{13}+\kappa_{04}.
\tag{22.5}
\]

There is one important factorable special case.  Conditional on an exposed
state `F`, suppose the coordinates `d_i` are genuinely independent, with
conditional cumulants `m_i,v_i,c_(3,i),c_(4,i)`.  Define

\[
 M_s=\sum_i s_im_i,\quad V_s=\sum_i s_i^2v_i,\quad
 C_{3,s}=\sum_i s_i^3c_{3,i},\quad C_{4,s}=\sum_i s_i^4c_{4,i}.
\]

Then per-site four-jets give the exact cheap contraction

\[
 \boxed{
 e_s^4=E\left[M_s^4+6M_s^2V_s+4M_sC_{3,s}+3V_s^2+C_{4,s}\right].
 }
\tag{22.6}
\]

Conditional centering sets `M_s=0`.  Formula (22.6) is lawful only with an
exact conditional-product lineage.  Conditional Gaussian covariance,
pairwise independence, or dense-mixing intuition does not imply it.

### 22.2 Two characteristic-function values plus a tail receipt give a rigorous upper bound

For real `x`, define the positive fourth-difference kernel

\[
 k_h(x)={6-8\cos(hx)+2\cos(2hx)\over h^4}
 ={16\sin^4(hx/2)\over h^4}.
\]

Since `|sin y|<=|y|` and
`1-(sin(y)/y)^4 <= (2/3)y^2`, one has the pointwise sharp-order bounds

\[
 0\le k_h(x)\le x^4,
 \qquad 0\le x^4-k_h(x)\le {h^2x^6\over6}.
\tag{22.7}
\]

Consequently

\[
 \boxed{
 K_s(h):={6-8\Re\phi_s(h)+2\Re\phi_s(2h)\over h^4}
 \le e_s^4
 \le K_s(h)+{h^2\over6}\|F_s\|_6^6.
 }
\tag{22.8}
\]

This corrects a tempting but invalid use of finite CF values: the two values
alone provide a lower bound, not an upper bound.  A sixth-moment, support, or
gradient tail receipt is essential.  If `|F_s|<=A_s` almost surely and
`hA_s<sqrt(6)`, (22.8) closes without a sixth moment:

\[
 \boxed{e_s^4\le {K_s(h)\over1-h^2A_s^2/6}.}
\tag{22.9}
\]

This bounded-support shortcut is not available for an untruncated Gaussian
ReLU lineage.

If the two measured real CF values have certified absolute errors
`epsilon_h,epsilon_(2h)`, replace `K_s(h)` on the upper side by

\[
 \widehat K_s(h)+{8\epsilon_h+2\epsilon_{2h}\over h^4}.
\tag{22.10}
\]

The `h^-4` term is the unavoidable cancellation/noise cost of a value-only
fourth derivative.

### 22.3 A Gaussian directional-Jacobian receipt supplies the missing tail

Assume now that `F_s` is a differentiable functional of the common standard
Gaussian/Wiener input `Z`; ReLU is covered almost everywhere.  Put

\[
 G_{p,s}=\left\|\,\|D F_s\|_2\,\right\|_p
 =\left\|\,\|J_d(Z)^Ts\|_2\,\right\|_p.
\tag{22.11}
\]

Gaussian integration by parts gives, for every `p>=2`, the dimension-free
inequality

\[
 \boxed{\|F_s\|_p\le\sqrt{p-1}\,G_{p,s}.}
\tag{22.12}
\]

A short proof useful for implementation audit is as follows.  With `L` the
OU generator, let

\[
 U_s=-DL^{-1}F_s=\int_0^1P_\rho(DF_s)\,d\rho,
 \qquad \Gamma_s=\langle DF_s,U_s\rangle.
\tag{22.13}
\]

For `psi(x)=x|x|^(p-2)`, Stein's identity gives

\[
 E|F_s|^p=(p-1)E[|F_s|^{p-2}\Gamma_s].
\]

OU contraction implies `||U_s||_p<=G_(p,s)`, hence
`||Gamma_s||_(p/2)<=G_(p,s)^2`; Holder proves (22.12).

Two immediately deployable receipts follow:

\[
 \boxed{e_s\le\sqrt3\,G_{4,s},}
\tag{22.14}
\]

and, combining `||F_s||_6^6<=125G_(6,s)^6` with (22.8)--(22.10),

\[
 \boxed{
 e_s^4\le
 \widehat K_s(h)+{8\epsilon_h+2\epsilon_{2h}\over h^4}
 +{125\over6}h^2G_{6,s}^6.
 }
\tag{22.15}
\]

Writing `A=8 epsilon_h+2 epsilon_(2h)` and
`C=125G_(6,s)^6/6`, the worst-case noise-plus-tail addition
`A/h^4+Ch^2` is minimized at `h=(2A/C)^(1/6)` and equals

\[
 3\,2^{-2/3}A^{1/3}C^{2/3}.
\tag{22.16}
\]

The direct VJP bound (22.14) is cheaper; (22.15) is useful only when the
observed signed CF slice is much smaller than the global Jacobian radius.
An almost-sure directional Lipschitz receipt `||J_d^Ts||<=L_s` may replace
both `G_(4,s)` and `G_(6,s)` by `L_s`, although separately bounding true and
AQ Jacobians discards the cancellation that makes `d` small.

### 22.4 Exact Stein-resolvent receipt and a finite-Mehler remainder

The same construction gives a sharper exact identity, not merely a
Poincare bound:

\[
 \boxed{e_s^4=3E[F_s^2\Gamma_s],\qquad
 e_s^2\le3\|\Gamma_s\|_2.}
\tag{22.17}
\]

It aligns naturally with a genuine Mehler interface.  Let an implementation
return an approximation `Uhat_s` to

\[
 U_{s,rho0}=\int_{\rho_0}^1P_\rho(DF_s)\,d\rho
\]

with `||Uhat_s-U_(s,rho0)||_4<=delta_(U,s)`.  The omitted low-correlation
interval obeys `||U_s-U_(s,rho0)||_4<=rho_0G_(4,s)`.  Therefore

\[
 \boxed{
 e_s\le
 \left\{3\left[
 \|\langle DF_s,\widehat U_s\rangle\|_2
 +G_{4,s}(\rho_0G_{4,s}+\delta_{U,s})
 \right]\right\}^{1/2}.
 }
\tag{22.18}
\]

This is a literal moment/tail certificate for finite Mehler branches.  A
covariance recursion called ``Mehler-like'' does not satisfy it: the receipt
must act on the signed residual Jacobian and must expose the L2 norm of the
contracted scalar.

For clarity about the present H3--H6 state, split `F_s=A_s+B_s`, where `A_s`
is the exactly exposed finite branch and `B_s` is the unexposed tail.  If
`Gamma_(A,s)` is exact and
`G_A=||DA_s||_4`, `G_B=||DB_s||_4`, then

\[
 \boxed{
 e_s\le
 \left\{3\left[
 \|\Gamma_{A,s}\|_2+2G_AG_B+G_B^2
 \right]\right\}^{1/2}.
 }
\tag{22.19}
\]

There is no generic `1/7` improvement for an H7-plus tail in L4.  The
`1/m` OU resolvent weight is an L2 spectral fact; transferring it through the
L2 norm of the product in (22.17) requires a hypercontractive/band-energy or
pointwise Jacobian-tail receipt.  Thus the current finite H3--H6 branches do
not by themselves evaluate (22.19).

### 22.5 Information-theoretic closure of the current interface

Three obstructions are exact.

1. **Marginal coupling.**  Even complete separate characteristic functions
   of true and AQ do not determine (22.2): equal Rademacher marginals can be
   coupled identically, giving `F=0`, or oppositely, giving `|F|=2`.
2. **Low moments.**  Mean and variance do not upper-bound a fourth moment.
   The centered law taking values `+M,-M,0` with probabilities
   `1/(2M^2),1/(2M^2),1-1/M^2` has variance one and fourth moment `M^2`.
3. **Finite CF values.**  A finite collection of bounded Fourier observables,
   even together with moments through order two, gives no distribution-free
   fourth-moment upper bound in a nondegenerate moment class.  In the dual
   moment problem such a bound would require a pointwise domination of
   `x^4` by a finite linear combination of `1,x,x^2,cos(t_jx),sin(t_jx)`,
   impossible as `|x|` tends to infinity.

Even all one-, two-, and three-coordinate residual laws do not determine a
dense directional fourth moment.  Let three independent Rademachers be
`eps_1,eps_2,eps_3` and set
`eps_4=+eps_1eps_2eps_3` or `-eps_1eps_2eps_3`.  The two four-vectors have
identical proper-subset laws and covariance, while for `s=(1,1,1,1)`

\[
 E(s^T\epsilon)^4=64\quad\hbox{or}\quad16.
\tag{22.20}
\]

This proves that per-site tilted generators and finite pair/triple Price
branches cannot supply `e_s` unless the conditional-product premise of
(22.6) is separately certified.

Therefore the present interface is analytically closed: it has neither the
signed joint four-jet (22.3), the conditional-product premise (22.6), the
directional Jacobian L4/L6 receipt (22.11), nor the genuine Mehler-resolvent
receipt (22.18).  The least additional state is one of:

* one exact or one-sided signed fourth-jet scalar for every distinct compiled
  direction `s`;
* one batched residual VJP moment `G_(4,s)` per direction, using (22.14);
* two signed CF values plus `G_(6,s)`, using (22.15);
* the direct complete-old signed q211 surface, which is preferable if `R`
  and hence `|S|` is not small.

### 22.6 Recompiled endpoint gate

For each `s`, let `ebar_s` be a certified upper radius from (22.3), (22.6),
(22.14), (22.15), or (22.18), and substitute `ebar_s` monotonically into the
polynomials in (21.3)--(21.5).  Denote the result by `Bbar_q`.  The coherent
endpoint condition remains exactly

\[
 \boxed{
 \left\|\sum_{q=1}^R|c_q|\bar B_q\right\|_{L^2({\rm nets})}
 +\eta_{\rm other}\le3.80\times10^{-6}.
 }
\tag{22.21}
\]

The gain condition remains

\[
 \boxed{
 \left(\|w_1\|_2-
 \left\|\sum_q|c_q|\bar B_q\right\|_2-
 \eta_{\rm other}\right)_+^2\ge2.86\times10^{-10}.
 }
\tag{22.22}
\]

No unsigned fourth-moment receipt establishes favorable alignment; it only
certifies that omission cannot move the endpoint by more than the allowed
radius.

### Atomic graph nodes

* `TRUE_MINUS_AQ_DIRECTIONAL_FOURTH_JET`: (22.2)--(22.5), exact and minimal;
  open because no same-lineage signed joint generator is currently exposed;
* `TRUE_MINUS_AQ_CONDITIONAL_PRODUCT_GATE`: (22.6), exact cheap factorization
  conditional on a lineage-level independence proof;
* `TRUE_MINUS_AQ_CF_TAIL_CERTIFICATE`: (22.7)--(22.16), two signed CF values
  plus a sixth-moment/Jacobian tail, solved theorem but interface open;
* `TRUE_MINUS_AQ_GRADIENT_L4_GATE`: (22.11)--(22.14), one batched residual VJP
  moment per compiled direction; cheapest sufficient new observable;
* `TRUE_MINUS_AQ_STEIN_RESOLVENT_L4`: (22.17)--(22.19), exact OU/Mehler
  contraction and finite-tail certificate; current H3--H6 branch is
  insufficient without `G_B`;
* `TRUE_MINUS_AQ_FINITE_CF_NOGO`: the three information obstructions and
  parity witness (22.20), closing marginal/finite-value routes;
* `Q211_DIRECTIONAL_L4_ENDPOINT_RECOMPILE`: (22.21)--(22.22), open until one
  of the signed-fourth, VJP, CF-plus-tail, or direct-surface receipts exists.

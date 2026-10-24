# Angular-energy pair statistic: exact Gaussian initialization, not a deep carrier

September 7, 2026, Phase 2 only. Use filesystem mtime for completion.
Source/algebra work by root and the two existing agents; no numerical
experiment, model arrays, targets, fit, candidate or score measurement.
The preceding turn clarified the conditional-source/gain boundary and
refreshed the board; it did not improve a score. Preserve all closeouts.

## Decision relative to the actual objective

Confirmed official #329926 remains adjusted4.982231460183304e-9; qualified
independent14 local0b84 remains4.788089831892398e-9. Its NCraw is
2.245315959186444e-8 and complete meanF467915916017.2143. The strict
below1e-9 goal and below4e-9/3e-9/2e-9 milestone policy remain unchanged.
The response-row exception is completed, not open to another fit or sweep.

The two concrete source checks considered in this round were:

1. **Exact removal of near-linear carrier coordinates.** Expected raw
   change would be zero in exact arithmetic, but the current frozen heads
   still consume intermediate scalar and directed-plane information.
   No material complete-cost saving was established. The only newly
   itemized literal duplicate gate expression saves a source-priced
   1,671,168F/card, approximately0.00036%, before any qualification. It is
   not a milestone route. The cheapest dependency check is complete below.
2. **Angular-energy joint source.** Homogeneity alone does not supply the
   required energy-pair covariance. A new exact initializer for the first
   zero-mean Gaussian layer is derived below. Given G, its FP64 dense
   product/matvec subtotal is8,589,932,544F plus120n^2+O(n) pointwise work
   at n1024, excluding guards/copies and all later transport. There is no
   supported final raw-gain magnitude or complete estimator cost. The
   unresolved assumption is affordable, accurate nonlinear propagation
   and actual consumer value, not initialization. The cheapest algebraic
   constructor/completeness check was selected and completed; it does not
   admit a substantial run.

The old isolated full-trace oracle's approximately.61 next-response error
ratio is not a final raw forecast for this initializer. Neither it nor
mathematical exactness justifies a larger trace/radial/conditional fit.
Current all-P0b84 would still require over16.459% uniform complete-cost
savings at unchanged per-net raw to cross4e-9, or37.345% to cross3e-9.
These are planning ratios, not changed-card measurements. At the.1 floor,
current raw would still give2.2453e-9, so a raw route remains necessary.

## 1. Exact radial projection and its material remainder

Let X~N(0,I_d), H(X) be a degree-one homogeneous bias-free ReLU prefix,
m=E H, z=H-m, C=Cov(H), and T its centered third tensor. Put

    E_hidden=||H||^2, e=E[E_hidden], R2=||X||^2,
    eta=e/d, epsilon_E=E_hidden-eta*R2.

Radial separation (or Gaussian integration by parts) gives

    Cov(R2,H_i)=m_i,
    Cov(R2,H_i H_j)=2 E[H_i H_j],
    Cov(R2,z_i z_j)=2C_ij.

Since Cov(E_hidden,R2)=2e and Var(R2)=2d, epsilon_E has mean zero and is
orthogonal to R2. This is the exact least-squares radial projection.
For Q the centered FOURTH CUMULANT, not a raw moment,

    sum_k Q_kkij = 2eta*C_ij-2(C^2)_ij-2T(m)_ij
                  +Cov(epsilon_E,z_i z_j),
    T(m)_ij = sum_k m_k*T_ijk.

Dropping the last term is a concrete approximation, not a consequence of
homogeneity or concentration. Take the literal first layer H_i=(X_i)+,
W=I, width d=n. With c0=1/sqrt(2pi),

    eta=1/2, epsilon_E=(1/2)sum_i sign(X_i)*X_i^2,
    Var(epsilon_E)=3n/4, Var(E_hidden)=5n/4,
    Cov(epsilon_E,z_i^2)=3/4-1/pi.

Thus60% of norm variance remains after radial projection. Omitting the
angular term predicts marginal fourth cumulant -3/(2pi^2), whereas the
exact value is3/4-1/pi-3/(2pi^2), even a sign difference. The centered and
angular OFFDIAGONAL entries are zero for this independent example.
Root corrected an initially misstated offdiagonal comment to the agent
before this record; no numerical result was involved.

Relative norm concentration as n grows does not force these individual
mixed-cumulant remainders to vanish. This is an architectural example,
not a sample from the random Phase2 panel or a rejection of every late-
layer angular approximation. Input-radius/sphere/chi receipts stay closed.

## 2. Constructive exception: zero-mean Gaussian first-layer source

Let Z~N(0,G), with symmetric PSD G and v_i=G_ii>0. Define H=ReLU(Z),
s_i=sqrt(v_i), rho_ij=G_ij/(s_i*s_j), and u_ij=sqrt(1-rho_ij^2).
Use the following pair kernels (matrix B is generally not symmetric):

    D_ij = E[|Z_i||Z_j|]
         = (2/pi)*s_i*s_j*(u_ij+rho_ij*asin(rho_ij)),
    S_ij = (2/pi)*asin(rho_ij),
    A_ij = (2/pi)*(s_j/s_i)*u_ij,
    B_ki = E[Z_k|Z_k|sign(Z_i)]
         = (2v_k/pi)*(asin(rho_ki)+rho_ki*u_ki),
    b_i = sum_k B_ki,
    J = G@G, U=G@D,
    m_i=s_i/sqrt(2pi), Cpost=(G+D)/4-m*m.T.

The exact RAW-product energy covariance is

    L_ij = Cov(||H||^2,H_i H_j)
         = [2J_ij + J_ii*A_ij + 2J_ij*S_ij + J_jj*A_ji
            +2(U_ij+U_ji)+G_ij*(b_i+b_j)]/8.

Its required vector counterpart is

    c_i = Cov(||H||^2,H_i)
        = (G@m)_i+(1/2)*m_i*J_ii/v_i.

Consequently the CENTERED-product statistic and angular remainder are

    K_ij = Cov(||H||^2,(H_i-m_i)(H_j-m_j))
         = L_ij-m_i*c_j-m_j*c_i,
    Aenergy = K-2eta*Cpost.

Here eta=trace(G)/(2*d_input) if Z=W0.T X. The denominator is the original
standard-normal input dimension, NOT rank(G); it is1024 in Phase2. The
diagonal factors in L are J_ii*A_ij and J_jj*A_ji; row-scaling both terms
would be wrong. DG=U.T relies on symmetric G and D.

Zero-variance coordinates have H_i=0 and can be handled directly. The
rho=+/-1 cases, including diagonal entries, follow by continuous Gaussian
limits. This is a real-arithmetic identity, not a certificate that an
unqualified floating covariance/rho or a clipped implementation is valid.

## 3. Derivation and independent checks

Expand H_k^2=(Z_k^2+Z_k|Z_k|)/2 and H_i=(Z_i+|Z_i|)/2.
Global zero-mean Gaussian parity leaves precisely four terms in
8E[H_k^2 H_i H_j]:

    E[Z_k^2 Z_i Z_j], E[Z_k^2 |Z_i||Z_j|],
    E[Z_k|Z_k| Z_i|Z_j|], E[Z_k|Z_k| |Z_i|Z_j].

Wick/Stein identities evaluate them as

    v_k*G_ij+2G_ki*G_kj,
    v_k*D_ij+G_ki^2*A_ij+2G_ki*G_kj*S_ij+G_kj^2*A_ji,
    2G_ki*D_kj+G_ij*B_kj,
    2G_kj*D_ki+G_ij*B_ki.

Summing k and subtracting E||H||^2 E[H_iH_j] cancels the trace(G)
terms and yields L without subtracting those large terms numerically.
For the vector identity,

    E[H_k^2 H_i]
      = [v_k*E|Z_i|+2G_ki^2/(sqrt(2pi)*s_i)
                         +2G_ki*E|Z_k|]/4.

Summing and centering gives c. In the nonsingular case, distributional
derivatives give A=E[2delta(Z_i)|Z_j|] and S=E[sign(Z_i)sign(Z_j)].
For B, differentiating with respect to rho gives
4v_k*sqrt(1-rho^2)/pi and B(0)=0. Degenerate cases use continuity, not
division by a zero conditional variance.

Root checked the Wick/Stein expansion and the independent G=I diagonal
and offdiagonal moments. Both agents independently derived the formulas
and checked the endpoint/centering ownership. No sampled or numerical
matrix certificate was run. The scoped archive search found no equivalent
exact energy-pair initializer in the inspected lineage; it is not a
claim of global novelty. The elementary bivariate absolute kernel is
established prior art: Haas2018, Eq3,
https://accessecon.com/Pubs/EB/2018/Volume38/EB-18-V38-I1-P63.pdf
and Gaussian covariance differentiation has the distributional formulation
https://link.springer.com/article/10.1007/s10959-020-01017-w .
The matrix contraction and application here are our derivation; neither
paper establishes a Phase2 estimator gain.

## 4. Paid source arithmetic, not a complete candidate bill

Given symmetric FP64 G, dense J and U cost4n^2(2n-1) charged FLOPs in
installed FNP.12. G@m costs2n(2n-1), for combined8n^3-2n:
8,589,932,544F at1024. A fresh FP64 W0.T@W0 would add4,292,870,144F at
square1024; the given-G constructor does not acquire this for free.

A shared-intermediate literal FP64 pointwise spelling has the following
leading coefficients of n^2, ALREADY including dtype rate2:

| Work | Coefficient |
| --- | ---: |
| outer(s,s), rho, sqrt(1-rho^2), asin(rho), S,D,A,B, column-sum B |68|
| exact Cpost |8|
| L, including its final division by8 |28|
| subtraction of m*c.T+c*m.T |8|
| subtraction of2eta*Cpost |4|
| final symmetrization |4|
| Total |120|

120n^2=125,829,120F at1024; vectors addO(n). Source inspection confirms
arcsin weight16, basic arithmetic/sqrt/outer/sum weight1, FP64 rate2 and
matmul mn(2k-1) before dtype in data_gate/.venv/lib/python3.11/site-packages/
flopscope/{data/default_weights.json,_registry.py}. This subtotal excludes
explicit copies/casts, source symmetrization if needed, domain/finite
guards, allocations not covered by the respective primitive, acquisitions,
all future nonlinear work, heads, RSS and runtime. One retained full FP64
matrix is8MiB, not a peak-memory claim.

The active0b84 first map at1120--1170 uses the order6 Mehler pair map.
Its covariance is NOT literally the exact arcsine Cpost here. Input Gram
rounding and later learned/masked covariances also have their own owners.
No existing head/prefix parity or native whole-card bill follows from
this source identity. No frozen candidate was edited.

## 5. What remains missing

Even exact K above is NOT the full centered fourth partial trace:

    R4=K-2T(m)-2Cpost@Cpost.

The additional Cpost-square product and the full-third contraction must be
supplied. A generic existing CP evaluation of T(m) needs a weighted Gram
with leading2n^2R conventional FP32 work:42,681,237,504 at archived
R=20352, plus vector, star/trace and preparation operations. That is a
subtotal on an old shape, not current acquisition, a lower bound or a
new native bill. Using the approximate current CP tensor does not turn
this into the true full fourth trace. Nor does a trace determine full Q.

A fixed energy label transports its centered weighted pair statistic
through an affine map by W.T K W. The next ReLU needs higher label-weighted
moments/conditional dependence; the affine transport alone is not a closed
recurrence. Refreshing the hidden-energy label also requires a new source.
For nonzero Gaussian means, the parity cancellation used above fails and
additional trivariate absolute-value terms survive. For actual interior
non-Gaussian inputs, G alone does not determine them at all.

Thus there is now an explicit inexpensive exact INITIAL source, rather
than only a missing-statistic label. There is still no economical paid
deep propagation/completion, overlap subtraction, supported final raw
gain or complete adjusted-score configuration. No source/rank/order/bin
sweep, old-family replay, training or substantial numerical run follows.
Any future admission must freeze the actual raw-times-complete-multiplier
criterion, with all current-head, numerical, resource and release checks.
Raw increases remain acceptable when the complete cost reduction pays.

## 6. Exact near-linear pruning check

Current derivatives518--530 use the Gaussian pdf independently of a
rounded near-one gate; finite near-linearity does not make curvature
identically zero. Existing response-row omission and its radial repair
already own the declared approximation. Further row omission changes it.
Even exact physical linearity would not erase the intermediate CP push:
all16 frozen P heads consume scalar cumulants and directed K21 statistics.
Cold covariance diagonals survive masking; youngest-pair common pushes
are already shared. These confirm the existing exact-reuse and last-two-
layer audits, not a new universal lower bound.

No significant product was removed. No estimator process, MPS job,
sampling, candidate, submission, discussion post, commit, promotion,
deletion or frozen edit occurred. Both existing agents only performed
read-only reviews. The goal remains active and unachieved; scientific
uncertainty is not an external blocked condition.

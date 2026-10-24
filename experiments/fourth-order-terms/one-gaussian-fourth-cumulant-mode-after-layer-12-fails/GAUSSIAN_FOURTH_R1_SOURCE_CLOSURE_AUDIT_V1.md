# Gaussian fourth birth: source closure and odd-descendant ownership

September6,2026 PDT. Phase2 only. Source/algebra audit; no network arrays,
targets, estimator run, numerical certificate, fit, source modification,
rank sweep, or reopening of a closed arm. This is not a candidate design
or an authorization to test a new family.

## Verdict

An arbitrary full symmetric modulation matrix is more expressive than the
closed covariance-tangent Sym(C,D) update. Its affine action and its
leading fourth-to-third descendants can be written explicitly. Neither
fact supplies an accurate fixed-rank nonlinear source.

In particular, ONE independently born matrix plus the existing radial
matrix is not an exact closure even for the Gaussian two-quadratic /
two-linear birth on three independent coordinates. A selected-column,
projection, or residual-discard approximation is unavoidable. No such
fixed supplier with a useful error certificate has been provided. The
source-class obstruction below does not prove that a rank-one
approximation would fail on the actual MLPs.

## 1. Conventions and the Gaussian source

Here use the NORMALIZED convention

    Sbar(V)_ijkl = (Vij Vkl + Vik Vjl + Vil Vjk)/3,
    T4 = lambda Sbar(V),
    T4[a,a,a,a] = lambda (a.T V a)^2.

FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md instead calls the sum
without /3 S(V); its lambda must therefore be multiplied by3 when translated
to this note. This is a normalization change only.

Let C be the preactivation covariance, g and d the Gaussian first and
second ReLU mean derivatives, and G=diag(g). The Gaussian Hermite birth
with two quadratic and two linear vertices is

    t(a) = (diag(d) a) * (C G a),
    F(a) = 12 t(a).T C t(a).

If a real factor C=L L.T exists, then exactly

    Vr = sym[diag(d * L[:,r]) C G],
    F(a) = 12 sum_r (a.T Vr a)^2.

The sum over ALL columns is invariant under L -> L O for orthogonal O.
One selected column is not. A fixed-rank supplier must specify which
factorization, which column/subspace, and the charge/error of that choice.
"One mode" is not itself a source-selection rule.

For a proposed Ctilde with E=C-Ctilde positive semidefinite, replacing
ONLY the middle covariance gives the exact query-specific residual

    F(a)-Ftilde(a) = 12 t(a).T E t(a)
                  <= 12 ||E||op ||t(a)||^2.

This is neither a final-mean error bound nor a certificate for replacing
the two C-containing linear legs as well. For an indefinite E, the
absolute-value bound uses ||E||op but loses the nonnegative-residual claim.
Acquiring a useful spectral bound and controlling downstream consumers
remain obligations, not free consequences of carrying Vr.

The bank does not guarantee PSD of every covariance state. The distinct
trace repair recorded minimum eigenvalue -1.3257340074301217e-6 for its
symmetrized uncorrected post12 C on rank35. That is NOT a measurement of
the proposed pre12 birth covariance, but it prohibits assuming general
PSD ownership from the word "covariance." A real Cholesky source must
qualify its actual input; a signed spectral source needs its own complete
factorization bill. No clipping, jitter, or replacement covariance is
licensed by this audit.

## 2. Even radial plus one arbitrary matrix is not closed

Restrict to three independent, identically distributed Gaussian inputs,
so C and the postactivation covariance are scalar multiples of identity,
and g,d are equal nonzero constants. Then the specified birth is

    F(a) = k (a1^4 + a2^4 + a3^4),  k>0.

Suppose, after absorbing covariance units into alpha, that it were exactly

    alpha (a1^2+a2^2+a3^2)^2 + lambda (a.T V a)^2

for real symmetric V and real alpha,lambda. Lambda=0 is immediately
impossible because a radial square has nonzero cross-pair coefficients.

If k-alpha !=0, comparing ai^4 gives lambda Vii^2=k-alpha !=0, so all
Vii are nonzero. The absent ai^3 aj coefficients give Vij=0. Comparing
ai^2 aj^2 then gives alpha+lambda Vii Vjj=0 for every pair. In three
coordinates the three nonzero diagonal entries must be equal. Consequently
alpha+lambda Vii^2=0, contradicting k>0.

If k-alpha=0, alpha=k>0 and Vii=0. Cancellation of each radial ai^2 aj^2
coefficient requires 2k+4lambda Vij^2=0, hence every off-diagonal Vij is
nonzero. But the ai^2 aj ak coefficient is 8lambda Vij Vik !=0, another
contradiction.

Thus even the radial background cannot make R1 an exact source closure.
Retaining the missing coordinate-fourth tensor separately would be a
different mixed state whose subsequent transport/readout must be charged.
This is not a universal minimum-rank theorem, an approximation lower
bound, or a statement about measured error on ranks35/100.

## 3. Leading fourth-to-third descendants have explicit owners

For lambda Sbar(V), the quadratic-linear-linear contribution is

    Delta T3_ijk = (lambda/6) sum_cyclic
                  d_i g_j g_k [Vii Vjk + 2 Vij Vik].

This is exact for the stated Hermite/leading-cumulant term, not for every
higher Gaussian-correlation correction to the complete ReLU map.

Write Sym3(b,B)_ijk = b_i B_jk+b_j B_ik+b_k B_ij. The first term can
attach a vector b to the SAME retained matrix V:

    pre:  V <- W.T V W,   b <- W.T b,
    post: V <- G V G,     b <- g*b + (lambda/6)*d*diag(V_pre).

It shares V's affine push. Its repeated plane is

    K21_ij = Vii*b_j + 2*Vij*b_i.

The second term is

    (lambda/3) sum_i d_i Sym3(e_i, c_i c_i.T),
    c_i = G V_pre[:,i].

For any fixed positive unit scale sigma_i, let q_i=c_i/sigma_i and
v_i=sigma_i e_i. The exact polarization is

    (lambda*d_i*sigma_i/6) [(q_i+v_i)^3-(q_i-v_i)^3]
      - (lambda*d_i*sigma_i/3) v_i^3.

Thus two signed CP cubes per coordinate plus a coordinate-diagonal
third correction suffice. After an affine push they contain all-distinct
third entries; they cannot be replaced by a one-step repeated-plane star
and then discarded. The sigma choice affects floating-point conditioning,
not this real-arithmetic identity, and would need a fixed numerical policy.

If these sources are added to the current carrier, the post-plane star
residual must subtract their ACTUAL repeated planes, including the CP
diagonal polarization correction and the independent b,V trace term.
Otherwise the already-computed post-K21 target and the newly carried
sources are counted twice. Masking/cold-diagonal policies also need an
explicit common convention; the current active-compaction routine cannot
silently transport a new unmasked full-support factor bank exactly.

An algebraically low-rank V=U diag(s) U.T reduces the coordinate sum to
q(q+1)/2 mixed triples using d*U[:,a]*U[:,b] and g*U[:,a],g*U[:,b]. But
the Gaussian Vr above is generically full algebraic rank even when only
ONE L column is retained. For diagonal-plus-low-rank C it is initially
diagonal-plus-low-rank, but generic W.T diag(v) W is full, not coordinate
diagonal. No fixed-q invariant is supplied. Replacing additional C legs
by a low-rank approximation is a stronger source approximation, not a
free implementation of the general full-matrix mechanism.

## 4. Radial overlap must be subtracted

For symmetric V, sum_ij K22(V)_ij = lambda Q(V)/3, where

    Q(V) = trace(V)^2 + 2*sum_ij Vij^2.

Adding a Gaussian-born mode to the current radial reset without a
counterterm includes its scalar trace projection twice. One EXPLICIT
bookkeeping convention is to preserve the current represented K22 sum:

    alpha_new = alpha_bank - lambda Q(V_new)/Q(C_carried).

The new state is alpha_new Sbar(C_carried)+lambda Sbar(V_new). This
removes only overlap in that one scalar functional. It is not a tensor
orthogonal projection, does not certify correct all-distinct entries, and
does not remove all overlap in fourth-to-third descendants. In an actual
port the counterterm's changed radial b/birth contribution and the
independent V descendants must BOTH enter the same star rebasing.

Use the declared actual carried metric and the bank's own trace/Frobenius
denominator. The map's pre-head covariance, mean-corrected carried
covariance, active-masked covariance, and any symmetrized replacement are
different numerical objects; choosing among them after a result is not
permitted. At lambda=0 the chosen law must reduce to the unchanged bank.

## 5. What can and cannot yet be priced

For illustration ONLY, suppose one mode is already supplied at post12,
gets no new fourth births, undergoes the leading-linear gate update at
13/14, and emits the exact leading third sources just described. Dense
full-support classical FP32 product subtotals at width n are:

| Work | Leading arithmetic |
| --- | ---: |
| Full V congruences at13 and14 | 8 n^3 |
| Final15 diagonal-only V read | 2 n^3 |
| New post13 CP pair: first shared push and pre14 plane | 6 n^3 |
| Old post13 plus new post14 CP pairs: final scalar pushes | 6 n^3 |
| Subtotal, with separately implemented shared pair pushes | 22 n^3 |

At1024 this subtotal is23,622,320,128 classical FP32 FLOPs, utilization
.0107421875. Without the two shared-first-push implementations it becomes
26 n^3. The source/readout/trace/birth elementwise work, casts, copies,
per-unit unit scales, native kernel dispatch, all initialization, and
necessary residual/mask handling are not included. A successful FP64
Cholesky alone would add at most(2/3)n^3 metered FLOPs in the installed
registry, but it neither selects an accurate column nor resolves a
non-PSD input. These are conditional primitive subtotals, not a complete
candidate bill or an attainable raw/adjusted result.

There is one additional exact REAL-ARITHMETIC sharing opportunity, not
used in that conservative subtotal. Store the post-mode as (A=G V_pre,G),
so V_post=A G. At the next affine layer form P=W.T A once. Then

    V_next = P (G W),
    W.T q_i = P[:,i]/sigma_i,
    W.T v_i = sigma_i W[i,:].T.

The two newly emitted CP atoms therefore share the mode's first product,
with no division by a potentially tiny gate. This can remove the two
separate first-push products above, reducing that same leading primitive
subtotal to18 n^3 =19,327,352,832 at1024. It requires a frozen factorized
state/rounding convention and numerical qualification; it is not a claim
of byte parity with computing a rounded V_post first. It does not remove
the later push of the older cohort, its pre14 plane, the source-selection
problem, or the unpriced nonlinear/residual work.

Keeping all n Gaussian-source modes is not a cheap escape: an outgoing
full congruence layer alone costs4 n^4 classical FLOPs; full mode storage
is4GiB at1024 in FP32 before workspaces or any other state. The alternative
three-matrix implicit source representation has a generically n^4 K22
readout; its cheaper marginal/K31 contractions do not remove that cost.

Still missing are a fixed causal mode supplier and error policy, a
fixed-rank nonlinear reset/merge or an explicit decision to omit later
births, complete treatment of incoming-third-to-fourth sources and other
Gaussian vertices, common carrier/mask/PSD conventions, and a frozen
head-compatibility strategy. The full actual card bill therefore cannot
yet be estimated honestly. No new numerical test is preregisterable from
the supplied equations alone.

## Sources and closed-family boundary

- FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md: exact overlap with
  radial, SSC and transported Sym(C,D), and scoped prior failures.
- radial_mehler.py: Gaussian6/cubic2 repeated-plane map, marginal and
  K31 assumptions, and scalar radial reset.
- radial_anisotropic_fourth_source_v1.py and
  ANISOTROPIC_FOURTH_SOURCE_ROUND_V1_CLOSED.md: explicit old trace
  projection and first-order column shift; raw+1.09%/+6.01% on the
  then-current unlearned host. Neither is the exact general V recurrence.
- FOURTH_TRACE_DISTINCT_NECESSITY_V2_CLOSED.md and
  FOURTH_TRACE_DISTINCT_INVERSE_REPAIR_V2.md: omission of one actual
  all-distinct incoming-third source was material; actual signed inverse
  domain and the separately identified non-PSD post covariance.

All referenced negative receipts remain closed. No source/column/rank,
threshold or late-window variant has been run by this audit.

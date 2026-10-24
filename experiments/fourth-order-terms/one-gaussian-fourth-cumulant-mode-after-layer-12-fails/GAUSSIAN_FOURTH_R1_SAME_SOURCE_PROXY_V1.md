# One Gaussian fourth mode: same-source information gap

Phase2 only. Written September6,2026 PDT (clock checked September7,
03:23UTC). Source/algebra follow-up, not an acquisition protocol or run.
No network/target arrays, estimator, numerical experiment, fit or submission
were used. Frozen predecessors and their closed arms remain unchanged.

## Decision

One causal source law and a cheap, target-free falsifier can now be stated.
The inspected receipts do not contain this particular middle-covariance
Gaussian-source comparison. They also provide no measured effect size or
source-concentration evidence supporting a full-card implementation.

This qualifies the earlier source-closure audit: failure of exact R1
closure is not a reason to reject an explicitly approximate hypothesis.
The next missing fact is whether ONE fixed mode improves even its own
Gaussian source's relevant output-ray shape over a same-source radial
control. Passing that question would not establish useful true fourth
information, downstream covariance accuracy, or final raw improvement.

## One fixed causal supplier, with explicit support

Use only rank35, birth after layer12, on the unchanged current-bank
trajectory. Let C0 be the actual incoming pre12 covariance and define the
auxiliary source matrix C=(C0+C0.T)/2. This is an explicitly symmetrized
source reference, NOT a repair or replacement of the bank covariance.
Let g and d=pdf/sigma be the bank's Gaussian responses before the layer12
learned mean hook. Let A be the existing layer12 alive mask after that
hook and its normal feedback; no new threshold is introduced. Set
dA=A*d, gA=A*g.

Starting with u=ones/sqrt(1024), perform exactly16 iterations

    u <- C u / ||C u||_2,
    theta = u.T C u,
    V = sym[diag(dA*u) C diag(gA)],
    lambda = 12 theta.

All norms and fields must be finite, and every normalization norm must be
nonzero. No fallback, jitter, clipping, sign selection, extra iterations
or alternate initialization is implicit. This is fixed power iteration,
not a claim of convergence to the largest positive eigenvalue. A signed
theta is allowed for a FORMAL cumulant source; no square root of theta or
PSD interpretation is needed. V is generically a full-rank matrix.

Equivalently V=A sym[diag(d*u) C diag(g)] A. This fixed support matches the
bank's existing active-coordinate transport convention. It is an explicit
approximation to the unmasked Gaussian birth, not free exact full-support
transport. Any test below compares the SAME masked source throughout.

Use the normalized tensor convention

    Sbar(V)_ijkl=(Vij*Vkl+Vik*Vjl+Vil*Vjk)/3.

The source being approximated is precisely

    t(a)=(dA*a) * (C @ (gA*a)),
    F(a)=12 t(a).T C t(a),
    F1(a)=lambda*(a.T V a)^2=12 theta*(u.T t(a))^2.

Only the MIDDLE C in this Gaussian two-quadratic/two-linear birth is
replaced by theta*u*u.T. Both C-containing external legs remain full.
It is not a projection of every tensor leg, an incoming-third source,
or a model of the complete fourth cumulant. For indefinite C or an
unconverged mode, even F-F1 need not be nonnegative.

## Exact trace of this same source

Let v=diag(C), h=dA*gA, and T_F=sum_ij K22(F)_ij. Wick contraction gives

    T_F = 4 [ (v*h).T C (v*h)
              + sum_i v_i*dA_i^2 * ((C*C) @ (gA*gA))_i
              + sum_ij C_ij^3*h_i*h_j ].

This costs O(n^2), not a tensor construction. To verify the normalization,
write r_i=C[i,:]*gA and let a be an auxiliary standard-normal vector.
Then

    E[t_i*t_j] = dA_i*dA_j *
        [delta_ij*(r_i dot r_j) + r_i,i*r_j,j + r_i,j*r_j,i],
    E[F(a)] = 3 T_F.

The three Wick pairings give exactly the three displayed terms. With
diagonal C, T_F=12 sum_i dA_i^2*gA_i^2*v_i^3, as required. This is an
algebraic expectation identity; no random draws were made.

For symmetric V define Q(V)=trace(V)^2+2 sum_ij V_ij^2. The selected
mode's K22 sum is lambda*Q(V)/3.

## Same-source radial control and actual-bank overlap

Let M be the ACTUAL carried post12 covariance after the learned hook,
mean feedback and existing active masking, including its retained cold
diagonals. The bank computes its radial reset before some of these
operations; its stored alpha_bank must not silently be recomputed on M.
Use its declared trace/Frobenius convention

    D_M=trace(M)^2+2 sum_ij M_ij^2,
    alpha_F=3*T_F/D_M,
    alpha_1=lambda*Q(V)/D_M.

Require finite nonzero D_M. With q(a)=a.T M a the two same-source
approximations are

    R0(a)=alpha_F*q(a)^2,
    R1(a)=(alpha_F-alpha_1)*q(a)^2 + lambda*(a.T V a)^2.

This is a natural-strength, fixed scalar-overlap convention, not a fitted
coefficient. Comparing F against alpha_bank*q^2 would be the WRONG
comparator: alpha_bank includes other Gaussian and incoming-moment
sources. A later actual intervention would instead subtract alpha_1 from
the unchanged alpha_bank and add the mode, owning the changed radial
descendants and all new fourth-to-third descendants together.

Numerical asymmetry matters. If M is not symmetric, D_M is the literal
bank-style K22 bookkeeping denominator, whereas q(a) sees sym(M). Thus
this convention is NOT an exact symmetric-tensor trace projection unless
M is symmetric. Its explicit discrepancy is

    D_M-Q(sym(M)) = 2*||skew(M)||_F^2
                 = ||M-M.T||_F^2/2.

A future proxy should report this scalar, not silently symmetrize M or
claim a physical covariance law. The source C and the carried metric M
are deliberately different, fully specified objects.

## Cheapest missing-information test; not authorized here

One coherent input-only capture would supply pre12 C0/g/d, actual alive12,
actual carried post12 M, W13, and actual pre13 alpha/sigma/original
variance. It must bind the unchanged host, P/Q and weights and qualify
its bank parity. Existing old covariance receipts are not a substitute
for a coherent current capture. No true fourth, Monte Carlo truth, or
current per-unit error labels are needed.

For all1024 actual W13 columns simultaneously, compute

    B = C @ (gA[:,None]*W13),
    T = (dA[:,None]*W13)*B,
    Y = C @ T,
    F_j = 12*sum_i T_ij*Y_ij,
    F1_j = 12*theta*(u.T @ T)_j^2.

Use the fixed fourth-to-Gaussian-mean consumer

    omega_j=(alpha13_j^2-1)*pdf13_j /
            (24*original_pre_variance13_j*sigma13_j).

Keep the original variance denominator; do not substitute a rounded or
floored sigma-squared object. This is the isolated local Gaussian mean
response, not the learned head response or final-layer error. Using the
unchanged q_j=diag(W13.T M W13), report

    E0=mean([omega*(F-R0)]^2),
    E1=mean([omega*(F-R1)]^2),
    E1/E0, correction/residual cosine and norms,
    theta, the power residual norm, and the metric-asymmetry discrepancy.

No amplitude, column, layer, mode-count, mask or fit selection is allowed
within this hypothesis. E0=0 needs explicit degeneracy handling, not a
ratio. E1>=E0 would reject this fixed supplier even on its own source.
E1<E0 establishes only a source-shape improvement; no numerical gain
threshold or downstream/full-card experiment has yet been authorized.

The exact F contraction is TWO classical dense products:4*n^3 =
4.294967296B FLOPs at1024. Sixteen power steps add about32*n^2;
Rayleigh, mode, trace and scalar reads are O(n^2). If q is not available
as a correctly owned saved field, M@W13 adds one product,2*n^3. No full
tensor, Cholesky or eigendecomposition is required. These are proxy
arithmetic subtotals, not a candidate meter or wall-time certificate.

A deployable late source also needs its two-product V congruences,
vector-matrix and CP third descendants, exact actual star rebasing,
common masks/units, all pair consumers, and frozen head compatibility.
The preceding audit's conditional18*n^3 or22*n^3 late-carrier product
subtotals are NOT a complete bill. This note does not bypass that work.

## What the saved evidence says

The archived covariance audit explicitly reads aux['pre_cov']; at pre12,
the leading-eigenvalue covariance trace shares are .06620534778928751
on rank35 and .0722222936630815 on rank100. The top-four shares are
.14580724559380473 and .15341390191087312. These do not suggest ordinary
rank-one covariance dominance. They are neither measurements nor bounds
on F concentration: the source's t(a) also contains d, g and full C legs.
Those old states are not claimed byte-identical to the proposed capture.

The corrected SSC results (.61-.69 held-out R2 on privileged total
marginal-fourth labels) concern nine fixed local basis matrices and their
products. The old Sym(C,D) recurrences and q4 leg projections likewise
do not execute the selected middle-C source comparison above. Their
failures stay closed; they are not a numerical result for this supplier.

More directly discouraging, FOURTH_EMISSION_ORACLE_V2_CLOSED records
natural-strength true-fourth-minus-actual-radial distinct emission at
post12 versus accumulated third error: weighted residual ratio
.992040038, cosine .103912887, correction/error norm .157185073. Its
fixed gates failed. This argues against expecting one late fourth source
to repair the accumulated odd error. It is scoped to that one-cell
consumer, not a universal fourth bound; direct covariance/next-mean paths
remain different. Neither the isolated roughly39% trace-oracle gain nor
the proposed Gaussian proxy establishes actual raw benefit.

Consequently the evidence supports, at most, asking this precise cheap
source-information question before any full card. It supports no invented
raw effect size or claim that this route reaches a score milestone.

## Inspected source and scalar receipt bindings

Paths are relative to this note's directory; hashes are SHA256.

- GAUSSIAN_FOURTH_R1_SOURCE_CLOSURE_AUDIT_V1.md:
  cf5e7ce122e018728673bb40b69e0ce119e630170926f23131d1720fb124af18
- FULL_MATRIX_FOURTH_MODULATION_PRIOR_SCOPE_V1.md:
  5baec5e2a27759f01953bb22fea4e78b1ba6b484814ecc77fbc6d160a375edf6
- audit_current_covariance_error_v1.py:
  45510555ca8cf277f0e89391896781988a17f11f1fd55cac48c19e63dba7ca06
- current_covariance_error_r35_v1.json:
  90a72bb2a7c4412637b2e985adbd16327c00773f9abce11d35bb075727984868
- current_covariance_error_r100_v1.json:
  c7fdd2c1392c02cf08a3ac080fb517f3a6b93a5be4ce7cb57a4a1ee600305ed8
- FOURTH_EMISSION_ORACLE_V2_CLOSED.md:
  bbee772f84fd4610d095298278e05d3f48e3f740c3c98e203b974bb5c40dcae9

The prior-scope note supplies the exact SSC and older carrier source paths.
Only existing source and saved JSON scalar text were inspected here.

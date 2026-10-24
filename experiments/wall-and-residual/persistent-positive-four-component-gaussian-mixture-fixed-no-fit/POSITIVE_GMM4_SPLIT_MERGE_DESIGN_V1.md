# Positive GMM4 split/merge: one fixed source-only proposal

Phase 2 only. Prepared September 6, 2026. **No network execution or target read
has occurred under this proposal.** Root must review the inert
`positive_gmm4_split_merge_prototype_v1.py` before separately authorizing the
one smoke below. The failed pair-memory experiment supplies no state or positive
evidence to this proposal. No training, tuning, new MC, or candidate is included.

## Novelty audit and its serious limitation

The Phase2 search found no executed endogenous persistent split/merge carrier
with independently retained full conditional covariances. The WGMprop read-only
audit (file mtime September6 02:39 PDT) rejects a full-network sigma-point
transplant, not this analytic layerwise recursion. Gamma/trace mean mixtures are
local formal consumers of an existing CP card; conditional quadratic/skew
probes are features; the four angular-energy bins are an oracle conditional
marginal experiment, not an endogenous Gaussian state. None is relabeled here.

However, `ONE_HIGH_THIRD_V1_CLOSED.md`, `CP_ONLY_ONE_HIGH_V1_CLOSED.md` and
`ONE_HIGH_ACTUAL_VALUE_V3_CLOSED.md` remain strong negative evidence. For any
K-component Gaussian mixture, with d_b=m_b-m_bar,

    T3 = sum_b pi_b [d_b^3 + Sym3(d_b,C_b)].

Thus T3 vanishes on three directions orthogonal to span{d_b}, whose dimension
is at most K-1. Untied full covariances do NOT eliminate this obstruction.
The possible new physics is nonperturbative, history-dependent mean/covariance
branching through gates, including structured all-distinct d_i*C_jk terms and
the mixture's implied higher moments. It is not generic third recovery.
Every Gaussianization discards within-component non-Gaussian cumulants and ReLU
face atoms; every merge also changes higher moments. This is a low-prior raw
experiment, not a rationale for another expensive fit around a failed tensor
projection. Broad convergence claims in mixture literature do not establish
width1024/depth16 or 1e-9 final-mean accuracy.

## Complete endogenous state

Retain at most four positive masses pi_b, means m_b (float64), and full dense
factors L_b (float32). The covariance owned by the next state is L_b L_b.T,
interpreting stored factors as real coefficients, **not** the previous rounded
pair-map input. There is no CP tensor, star, radial parameter, learned bank P/Q,
hidden teacher or extra residual carrier. All mixture cumulants are implied
by these masses/means/covariances; within each component they are Gaussian.

Start with one N(0,I). On layer0 form W0.T W0 in float32 (one full product).
Thereafter form B=L.T W and the pair-map input Cpre=B.T B with two float32
products; symmetrize the rounded Cpre in float64. Mean transport W.T m is
float64. A rounded Gram matrix is not asserted mathematically PSD merely
because it was formed as a Gram. The retained Cholesky factor below, and its
explicit no-retry qualification, own the next valid mixture state.

## One fixed split before each nonfinal ReLU

For every current pre component start q=ones/sqrt(n), perform exactly FOUR
normalized q <- Cpre*q power steps, then one additional Cpre*q for its
Rayleigh quotient. No QR, eigensolver, alternate seed, carried direction or
iteration convergence test exists. If this seed has exactly zero covariance
action, that component has no eligible split; a negative Rayleigh quotient or
nonfinite arithmetic fails the test. Define

    a = Cpre*q / sqrt(2*q.T*Cpre*q)
    score = pi * sum_i [phi(mu_i/sigma_i)/sigma_i * a_i^2].

Exactly-zero-variance coordinates contribute zero curvature. Choose the
largest strictly positive score; ties use the smallest deterministic component
ID. If all scores are zero, split none. For the chosen parent use the fixed
nodes (-sqrt(3),0,sqrt(3)) and weights (1/6,2/3,1/6):

    pi_child = pi_parent * weight
    mu_child = mu_parent + node*a
    C_child = Cpre - a*a.T.

In exact arithmetic C_child >= Cpre/2 for PSD Cpre. This positive three-node
Gaussian rule matches the original Gaussian moments through total order5;
it is NOT an exact decomposition into finitely many Gaussians. Its first
omitted terms start at degree6. It resolves one half of the selected Gaussian
linear-coordinate variance rather than discarding the transverse covariance.

## Fixed Gaussian nonlinear map and numerical conventions

Each child receives the exact scalar Gaussian ReLU mean and variance and a
fixed order12 Gaussian Hermite covariance. With sigma=sqrt(diag C), alpha=mu/sigma,
A1=Phi(alpha), and A_r=(-1)^(r-2)He_(r-2)(alpha)phi(alpha), set

    c_r = sigma*A_r/sqrt(r!),
    Cpost = sum_(r=1..12) outer(c_r,c_r) * rho^r + diagonal_tail.

The tail restores each exact scalar variance. For PSD rho, every summand is
PSD and the true tail is nonnegative. This is a declared covariance
approximation, not an exact Gaussian pair integral. Gaussian scalar functions,
Hermite recurrences, powers built by repeated multiplication, covariance maps,
split scores and merge scans are float64. No current third/fourth Edgeworth
input or bank head is evaluated.

Scalar moments use a fixed cancellation-resistant spelling. For a=abs(alpha),
q=Phi(-a), r=phi(a)-a*q, the positive-alpha mean is mu+sigma*r and its variance
is v*[1+(a^2-1)q-a*phi(a)-r^2]; the negative-alpha mean is sigma*r and variance
is v*[(a^2+1)q-a*phi(a)-r^2]. This is the same Gaussian moment law, not a new
approximation or an adaptive covariance repair.

Zero variance means an exactly deterministic coordinate: its output is
max(mu,0), its covariance row must be exactly zero, and its Hermite coefficients
are zero. There is no positive variance floor. Negative input variance fails.
Only scalar mean/variance/tail negative roundoff within 64 float64 eps times
the relevant input sigma/variance may be lifted to zero; larger deficits fail.
All lifts are recorded. Correlations are not clipped; absolute correlation
above 1+1e-6 fails. These entry checks are not advertised as PSD certificates.

After merging, symmetrize Cpost, separate exact zero-variance rows, normalize
the positive-variance principal covariance to a correlation, and perform ONE
float64 Cholesky. A failure ends the fixed test: **no jitter, retry, eigensolver,
threshold sweep or fallback**. Rescale its rows by the standard deviations and
cast the factor to float32. That rounded full factor is the next covariance
state; its relative factor-rounding error must be <=1e-6. Exact zero factors
and any variance coordinates lost to float32 underflow are explicitly recorded.
No extra product is used to pretend the old rounded covariance equals L L.T.

## Fixed positive merge and tie break

Once the child ReLU maps finish, merge until at most four components remain.
For each pair a,b with w=pi_a+pi_b, alpha=pi_a/w, beta=pi_b/w and d=m_a-m_b,

    m_merge = alpha*m_a + beta*m_b
    C_merge = alpha*C_a + beta*C_b + alpha*beta*d*d.T.

This exactly preserves first and second moments in exact arithmetic. The
global third tensor lost in this merge is

    DeltaT = w*alpha*beta [Sym3(d,C_a-C_b) + (beta-alpha)*d^3].

Use the stable positive norm from root's independent tiny certificate:

    H = C_a-C_b + (beta-alpha)*d*d.T/3
    frobenius_squared = (w*alpha*beta)^2 * [3*||d||^2*||H||_F^2 + 6*||H*d||^2]
    trace_vector = w*alpha*beta * [d*trace(H) + 2*H*d]
    loss = 6*frobenius_squared + 9*||trace_vector||^2.

Choose minimum loss; ties use sorted component IDs lexicographically. Recompute
all pair scores after the first merge. This is O(n^2) per candidate pair,
not a hidden tensor materialization or eigendecomposition. The single fixed
metric is E[DeltaT(g,g,g)^2] for g~N(0,I), including the trace channel; it is NOT
known-W response or final-mean risk. The same cached H*d supplies both terms.
It is NOT a bound on fourth cumulants or Wasserstein distance; pairs with equal
means can have zero lost-third score and still lose conditional fourth
structure. Positive masses receive only a roundoff-scale normalization at each
boundary; mass drift above1e-12 fails. New split/merge IDs increase monotonically.

Root's separate eight n5-fixture certificate passes at worst1.554e-14 for
splits through degree5, full lost-third tensors, positive norms, and PSD laws.
Files `certify_positive_gmm_merge_algebra_v1.py` and
`positive_gmm_merge_algebra_certificate_v1.json` are hashed, not modified.
That is algebra only, not a network qualification.
The independently certified Wick identity (seven n5 fixtures, exact degree6
Gaussian quadrature, worst2.489e-15 including the rank-one sixth moment) is in
`certify_gmm_gaussian_query_merge_v1.py` and
`gmm_gaussian_query_merge_certificate_v1.json`; both are also hashed and frozen.

## Final layer and complete cost ownership

Layer15 is scalar-only: for each of the four components calculate B=L.T W15,
its pre-variance by column sums of B^2 in float64, and exact Gaussian ReLU
means. No final full covariance, split, merge or Cholesky is computed. Report
all layers as the positive mixture's mean. This final path does not run the
old final Q or any separate estimator.

With all splits eligible, component counts are 1 -> 3 -> 4, then four carried
components with six temporary children on layers2..14. Consequently:

- 115 full float32 GEMMs: layer0 one, layer1 six, layers2..14 104, final four.
  Conservative ordinary charge 230*n^3 =246.961B.
- 59 float64 Choleskys, each charged up to (2/3)*n^3 including dtype rate2:
  42.234B. Positive-variance submatrices can only reduce this ceiling.
- 86 Gaussian Hermite12 pair maps; 335 merge-pair comparisons; at most280
  direction matrix-vector actions; all full conditional means and covariances.
  Gaussian-map work is approximately10B, stable merge scans approximately5B,
  direction scans approximately1.2B. Reserve a TOTAL35B for these and all scalar
  functions, normalization, factor scaling/casting, means, masks, ingress,
  initialization, copies, component assembly and guards. This part requires a
  final exact-backend census and is not called a metered bill.

Complete provisional ceiling: **324.194B, utilization about .14743**. There is
only about5.66B margin below .15; no new neural head or hidden carrier fits for
free. State is four full dense factors plus temporary full child covariances,
not four diagonal approximations. The intended <=2GiB process cap is unmeasured
until the separately authorized smoke. Standard dense BLAS/Cholesky may have
useful wall headroom, but no official persistent runtime claim follows from
this census. ReLU-map and component Python dispatch could still be slow.

## ONE fixed cheapest final-raw falsifier

Run only ranks35/100, unchanged weights from
`rows_current_transport_shadow_v2/row_NNNN/weights.npy`. Compare GMM4 against
a matched SINGLE-Gaussian recursion: same Hermite12 map, factor ownership,
float32/float64 policy, final scalar handling, no splits or merges, and no
bank heads. Neither arm is trained. Freeze both predictions on both networks
and all source/weight hashes before opening the existing1e9 truths
`public_g22_operator_features_v1/truthpost/truthpost_rankNNNN.npz` (`mc_post`).
This uses existing research truth, not a fresh validation panel.

Require on **BOTH** networks: final served-float32 raw MSE <=6e-8 AND
GMM4/single-Gaussian raw ratio <=.25. Also require numerical qualification,
<=2GiB process RSS, <=120s per offline model run, and the complete provisional
cost envelope <=.15. A zero control denominator is inconclusive.

Failure closes this exact no-fit component count/source/direction/power/map/
merge round: no components, seeds, directions, steps, order, split strength,
heads or training sweep follows automatically. It does not rule out all learned
mixtures. A pass earns ONLY a separate small coadaptation design review, not
permission to fit, port or submit. The gate is explicitly weaker than bank
raw parity: a materially better untrained raw source may justify that review.
Any eventual card still needs complete native metering, numerical certificates,
both14-net panels and persistent runtime. At the .14743 ceiling, raw below
about6.78e-9 is needed for adjusted below1e-9; the6e-8 gate is nowhere near the
goal and is not advertised as a competitive standalone score.

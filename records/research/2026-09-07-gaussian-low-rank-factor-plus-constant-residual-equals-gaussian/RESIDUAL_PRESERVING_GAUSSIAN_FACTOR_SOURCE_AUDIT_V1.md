# Low-rank Gaussian factors with a full residual: exact scope audit

Phase2 only, September6,2026 PDT. Source and algebra only. No network or
target arrays, numeric experiment, fit, rank/basis/temperature sweep or
candidate. Existing files are unmodified.

## Conclusion

**The old7x projection failure does not close residual-preserving conditional
propagation.** Its code never constructs conditional Gaussian responses.
But the literal finite state `h=m+Uz+epsilon`, with independent Gaussian z
and epsilon and a full constant residual covariance, is still one Gaussian.
Re-fitting that same state after each ReLU is algebraically the ordinary
Gaussian recursion, regardless of how its covariance is divided into U and R.

The genuinely different information is a retained nonlinear conditional mean
`M(z)` together with its full, z-dependent conditional covariance `R(z)`.
Their joint dependence supplies third cumulants absent from a constant
residual. A finite-node implementation is a full-covariance Gaussian-mixture
recursion. The already-executed GMM4 did retain that information, although
its precise split/merge rule does not close every continuous/fixed-latent
conditional model. No new affordable finite state preserving the missing
transverse information is identified here; no further run is proposed.

## 1. What the prior sources actually did

All relative paths start at the Phase2 experiment directory.

| Prior source | Executed object and actual boundary |
|---|---|
| `public_g22_operator_features_v1/subspace_skew_test.py` | Projects centered samples as `(hc@V)@V.T` before computing next-layer skew. Its comment explicitly says it does not run the diagonal-residual variant. This is an outright sample projection, not conditional integration with a full residual. |
| `public_g22_operator_features_v1/run_kprop_rank.py`; trackerT-129 | Retains the full `K[2]` covariance while replacing every all-distinct third factor A by `V@(V.T@A)` after each nonlinearity. Repeated slices remain in the organizer's separate state. Rank1024/512/256/128 raw was3.45e-8/2.29e-7/7.31e-7/1.41e-6. This is third-factor projection, not a conditional residual-response model. |
| `public_g22_operator_features_v1/ssc_k4_test.py`; trackerT-134a | Fits products of nine quadratic-form features built from covariance/plane response modes to next marginal k4. It does not evolve latent conditional means or covariances. Held-out-unit R2 .02/.26/.43/.53 was also based on selecting among four ridge penalties, and the later noise audit qualifies interpretation. It is not a general conditional-Gaussian closure theorem. |
| `astra_phase2_reexamination_v1/{CONDITIONAL_SKEW_ENERGY_PROBE_V1.md,generate_conditional_skew_energy_probe_v1.py}` | Four top-eight-mode regression-energy invariants, appended to a same-capacity teacher residual head. No distributional latent state or full covariance field was propagated. Its20/4,40-epoch information test failed. |
| `astra_phase2_reexamination_v1/conditional_quadratic_tensor_v1.py` and its CLOSED receipt | Contracts the complete carried third tensor onto two top-eight whitened directions, then forms eight per-unit rotation-invariant quadratic statistics. The fixed20/4 teacher-feature fit failed, not an endogenous conditional quadrature carrier. |
| `astra_phase2_reexamination_v1/{ONE_HIGH_THIRD_V1_CLOSED.md,CP_ONLY_ONE_HIGH_V1_CLOSED.md,ONE_HIGH_ACTUAL_VALUE_V3_CLOSED.md}` | Actual one-high cubic geometry/value tests. Strongly relevant to a *linear* latent mean with small-dimensional span, but not a proof that a nonlinear function of one latent coordinate has one-dimensional physical span. |
| `astra_phase2_reexamination_v1/positive_gmm4_split_merge_prototype_v1.py` and `POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md` | A genuine persistent positive mixture, each component retaining a full covariance factor. A split replaces C by `C-aa.T`, not zero or a diagonal, then computes a full Hermite12 Gaussian ReLU covariance per child. Independent conditional covariances survive until the specified moment-preserving merge. Raw35/100 was2.61823e-6/4.25983e-6 versus matched Gaussian2.89618e-6/4.63005e-6; the fixed raw/value gates failed. |

The old mathematical `LANE_h3_leakage_conditional_gaussian.md` studies a
quadratic/linear Gaussian cavity's gated H3 operator. It explicitly uses no
numerical experiment and does not supply this full conditional carrier.
The WGMprop audit concerns full-network sigma-point evaluation, not an
analytic residual-preserving layerwise state. No Phase1 result is used here.

## 2. A constant full Gaussian residual is not new raw information

Suppose

```
z ~ N(0,I_r), epsilon ~ N(0,R), independent,
h = m + U z + epsilon,        R >= 0.
```

Then `h~N(m,C)` exactly, with `C=UU.T+R`. Let `Y=ReLU(h)`. Conditional on z,

```
M_i(z) = E_epsilon ReLU(m_i+(Uz)_i+epsilon_i),
C_ij(z) = Cov_epsilon(ReLU(h_i),ReLU(h_j)).
```

Each M is the scalar Gaussian ReLU function and C is the full bivariate
Gaussian response using R. Exact outer integration gives

```
m_plus = E_z M(z),
C_plus = E_z C(z) + Cov_z M(z).
```

These are precisely the usual full-Gaussian ReLU mean and covariance based
on `(m,C)`. They cannot depend on the chosen factor split.

If the next state is projected back onto a linear Gaussian factor plus an
independent constant Gaussian residual, the exact regression coefficient is

```
U_plus = E[(Y-m_plus) z.T]
       = diag(Phi(m_i/sqrt(C_ii))) U,
R_plus = C_plus - U_plus U_plus.T.
```

The last matrix is PSD as a regression-residual covariance. It contains
both `E C(z)` and the unexplained nonlinear conditional-mean covariance;
using only `E C(z)` would not even preserve the total covariance. Declaring
the residual independent Gaussian makes the next h Gaussian again. Carrying
the latent cross-covariance U does not change any subsequent marginal
Gaussian recursion result. An approximate outer quadrature can introduce
dependence on U, but that is numerical approximation, not additional state
information.

If z is instead non-Gaussian while epsilon remains independent Gaussian,
all third cumulants lie in `U tensor U tensor U`. A full transverse R still
does not create transverse third information. If the mean stays linear but
R depends on z, at least one third-tensor leg remains in span(U). These two
different geometries must not be conflated with the original three-high
projection result.

## 3. The exact information that must survive a nonlinear response

For any conditional state define `d(z)=M(z)-E M(z)`. The law of total third
cumulance is

```
T_ijk = E[T_ijk | z] + E[d_i d_j d_k]
        + E[d_i R_jk(z) + d_j R_ik(z) + d_k R_ij(z)].
```

The new mixed term is the covariance of conditional means and conditional
covariances. Tying R across z removes it exactly. Gaussianizing each
conditional component removes the first term, even when every conditional
ReLU mean/covariance was evaluated exactly before that reset.

For conditionally Gaussian components, the fourth cumulant additionally
contains the covariance of covariance entries and the covariance of
`d_i d_j` with R. Thus SSC-like covariance-response tensors are a restricted
piece of a possible conditional state, not its whole nonlinear recurrence.

A crucial non-equivalence: nonlinear `M_i(z)` can be linearly independent
as functions of a SCALAR z. Its physical mean span need not have dimension r.
Consequently the continuum nonlinear conditional state need not have a
three-transverse-leg zero tensor. A Q-node positive approximation does have
mean span at mostQ-1; its Gaussian-mixture third vanishes on three vectors
orthogonal to that span even with untied full R. That finite-node fact does
not transfer automatically to the continuum state.

Full residual covariance is also not the full residual LAW. After a ReLU,
conditional face atoms, conditional skew and higher cumulants exist. Exact
Gaussian pair integration handles only the assumed conditional input law;
it does not make a subsequent Gaussian conditional reset exact. Retaining
the missing `E[T|z]` would require a new conditional third carrier, and
ultimately its higher-order response ownership, not just the matrix R.

## 4. The closed implementable recurrence and its full cost obligations

For fixed positive quadrature nodes/weights `(z_b,p_b)`, retain
`(M_b,R_b)` for each b. One complete *approximate* recurrence is

```
pre_M_b = W.T M_b
pre_R_b = W.T R_b W
M_b_plus = GaussianReLUMean(pre_M_b, diag(pre_R_b))
R_b_plus = GaussianReLUCov(pre_M_b, pre_R_b).
```

Read out `sum_b p_b M_b`. This is a Q-component Gaussian-mixture moment
recursion, not a smaller hidden state merely because nodes originated from
a low-dimensional Gaussian factor. Fixed nodes versus the executed adaptive
GMM4 split/merge is a real algorithmic difference, so the GMM4 result does
not numerically close that exact unexecuted rule. No positive source-capacity
evidence or prescribed new rule is supplied here, however.

For a persistent input latent with `R0=I-U0 U0.T`, all initial conditional
pre-covariances are shared. Form `W0.T W0` once and subtract
`(W0.T U0)(W0.T U0).T`. After the first ReLU the conditional covariances are
different. Through depth16 the full-factor implementation needs, before
any maps or latent setup,

```
FP32 full products: 1 + 28Q + Q = 1+29Q
  (one common first Gram, two per node on layers1..14,
   one final factor product per node for variances only)
FP64 Choleskys: 15Q
```

Using the same PSD factor-ownership policy and dtype charges as the executed
GMM prototype, these alone cost `(2+68Q)n^3`. At n=1024:

| Q | GEMM/Cholesky subtotal | Utilization subtotal | Raw needed for adj<1e-9, before remaining cost |
|---:|---:|---:|---:|
|4|294.205B|.133789|<7.474e-9|
|8|586.263B|.266602|<3.751e-9|

These are not metered whole-card bills. Add15Q full Gaussian pair maps,
16Q mean transports, final reductions, the initial `O(n^2 r)` products,
basis acquisition, node construction, factor scaling/casts, PSD/domain
guards, all scalar functions, buffers and ingress/return. A J-term Hermite
pair approximation gives total asymptotic work
`O(16Q n^3 + 15Q J n^2 + n^2 r)` plus basis/node construction, and memory
`O(Q n^2 + Q n + Q r)`. It is not exact pair integration: a fixed J needs its
own error/domain certificate. Exact bivariate Gaussian responses do not
come with a free finite operation count.

The executed adaptive GMM4's complete PROVISIONAL envelope was324.194B
(.147427), including its35B auxiliary reserve, with115 products,59
Choleskys,86 pair maps and split/merge work. It was never a native metered
candidate. This anchors, rather than replaces, the above new recurrence's
missing complete-bill qualification.

Tying all residual covariances allows one shared affine covariance push and
roughly `O(n^3+Q J n^2)` work per layer, but removes the mixed third term in
section3. Expanding R(z) in s full matrix-valued basis coefficients requires
transporting those s matrices. A generic diagonal gate modulation does not
commute with W; `W.T D(z) R D(z) W` is not one common congruence for all z.
A finite basis also needs an explicit projection after each Gaussian ReLU
response, including its PSD and moment error. No such cheap closed basis is
defined by saying "keep a full residual covariance."

## Endpoint

The source audit preserves the distinction requested: outright sample
projection, projected third factors, scalar conditional features, a Gaussian
factor reparameterization and nonlinear conditional covariance evolution are
different objects. Only the last supplies the proposed new mixed information.
The actual GMM4 already paid for one finite realization and failed far from
raw parity; the constant-residual simplification loses its key mixed term.
No complete affordable alternative retaining the omitted within-conditional
odd information is supplied. The continuum conditional-state idea remains
logically open, but is not promoted into an experiment or a parameter sweep.

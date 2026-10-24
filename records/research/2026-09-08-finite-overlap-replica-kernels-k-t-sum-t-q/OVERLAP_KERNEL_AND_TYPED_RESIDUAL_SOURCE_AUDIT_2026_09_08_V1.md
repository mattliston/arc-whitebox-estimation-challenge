# Replica kernels and typed residual memory: source-only decisions

September 8, 2026 PDT. Phase 2 only. Read-only algebra/source review by
root and the existing accuracy agent; no arrays, numerical experiment,
fit, new download, estimator change or submission. Old receipts remain
unchanged. This is not a new raw or adjusted score.

The confirmed official bank remains 329926 at
4.982231460183304e-9 adjusted. The best fully qualified local14 candidate
is unused_materializations_v1/estimator.py, SHA
409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f,
at 4.770235434670105e-9 adjusted, NC raw2.2458484596872894e-8 and
complete mean466058731667.9286 FLOPs. It is unchanged and unuploaded.
The strict official1e-9 goal and milestone-only4/3/2e-9 banking remain.

## 1. Finite-overlap kernels do not supply the connected remainder

For a fixed actual network prefix h and Gaussian replicas X,Y_t with
coordinate cross-covariance tI, define K(t)=E[h(X)h(Y_t).T]. Using
orthonormal Gaussian Hermite coefficients c_alpha,

    K(t) = sum_{q>=0} t^q B_q,
    B_q = sum_{|alpha|=q} c_alpha c_alpha.T >= 0.

Consequently K(0)=m m.T, K(1)-K(0)=C and
K'(1-)=E[J_h J_h.T]. ReLU prefixes are Lipschitz and have the first-order
Gaussian Sobolev regularity needed here. Gaussian Hermite degrees are NOT
spherical-harmonic/Legendre degrees. Positive homogeneity fixes radial
scaling, not the unknown angular expansion.

For 0<t<1 there is a valid Loewner bound

    (K(t)-t*K(1))/(1-t) <= m m.T <= K(t).

Proof: every B_q is PSD; t^q-t<=0 for q>=1, while the q=0 coefficient
is1-t. The upper bound follows directly from the nonnegative coefficients
of K(t)-B_0. This requires the TRUE kernels. Finite-node polynomial
cancellation cannot remove an arbitrary infinite Hermite tail without
a separately controlled remainder. No current numerical mean certificate
is supplied by the bound.

The exact across-depth derivative dependency is already recorded in
TWO_LAYER_CONNECTED_GATE_SOURCE_AUDIT_V1.md. For the sample Jacobian
cross-product P_l(X,Y_t),

    E[P_l]_ij = Q_l,ij(t)*(W.T E[P_(l-1)] W)_ij + Xi_l,ij(t),

    Xi_l,ij(t) = sum_ab W_ai W_bj *
        Cov(gamma_i(X)*gamma_j(Y_t), P_(l-1),ab(X,Y_t)).

Even exact pair-gate probabilities Q do not acquire Xi. Gaussian pair
probabilities introduce an additional approximation. Setting the connected
remainder to zero recovers the known Gaussian/Dirichlet factorization;
in a consistent Gaussian shadow the zero-overlap boundary returns that
shadow's own mean outer product. More nodes do not add the missing source.

With a shared first Gram, q full covariance-like trajectories have the
classical transport subtotal30q square products, or64.42450944q billion
FP32 multiply/add operations at n1024. Derivative states add comparable
transport. Gates, Xi acquisition, integration, consumers, ingress, copies
and initialization are additional. This is NOT complete FNP billing or
a universal lower bound. Actual paired-forward acquisition instead has
sampling cost and unmeasured correction variance.

Archive checks:

- measure_curve_v2.py sampled actual overlaps but retained aggregate
  centered traces after a2048-path mean calibration. These are not
  per-output true kernels or a signed mean-error supplier.
- NNGP_CENTERING_AUDIT_V1_CORRECTION.md corrects annealed theory; it does
  not establish a fixed-network kernel provider.
- The executed fixed Dirichlet observer's source/parent raw ratio was
  1.000934916. Its closeout remains scoped to that observer.

Decision: no overlap census, node sweep or fit. Missing is an affordable
causal Xi supplier, or controlled true-kernel extrapolation error, together
with useful signed current-head action. An exact identity is not evidence
of an attainable final-score magnitude.

## 2. Nonlinear pair memory is already an executed mechanism

The following are actual, narrower results, not generic impossibility
theorems. Root inspected the closed joint-edge/graph designs; the accuracy
agent additionally checked memory/observer implementations and receipts.

| Executed mechanism | Supervision and result |
| --- | --- |
| matrix_error_observer_v1.py | Full covariance-error matrix with Gaussian-tangent transport and fixed14-feature learned sources. Final raw improves2.99%, insufficient for its added matrix cost; closed. |
| train_joint_edge_covariance_v1.py | Stateless nonlinear endpoint/R/S/S-transpose pair decoder, true covariance innovations. Held-out covariance ratios.8945/.8911, next-mean.9948/.9522; closed. |
| Learned endpoint-history decoder | Frozen64-dimensional node histories, pair bilinear decoder, true covariance labels. Covariance.7414/.7444, next-mean.9933/.9855; closed. |
| nonlinear_memory_capacity_current_v1/memory_model.py | Full n-by-n H, V.T H V, nonlinear ordered-pair updates and Kstar/U transpose inputs. Bank-action teacher, not true joint residuals. Memory/control plane1.01099 and both current-head consumers worse; closed. |
| train_learned_joint_graph_probe_v1.py | R, elementwise R-squared, S and S-transpose messages into node outputs. Mean-residual/parent1.000888; closed. |

The exact combination of a nonlinear recurrent TRUE joint-residual sidecar
with coherent current-bank writes was not found executed in this bounded
inspection. That distinction does not itself establish new information:
the memory already has ordered pair history, and the covariance studies
already have true joint supervision. Replacing labels or increasing
capacity is not a substantiated supplier advantage.

### Type and centering obligations

Covariance error legitimately transports as W.T delta_C W. A repeated
third plane does NOT: its actual affine action is

    T'_aab = sum_ijk W_ia W_ja W_kb T_ijk.

A congruence-transported 'K21 channel' is therefore latent unless a
three-leg decoder/completion is specified. In addition, delta_K21 itself
changes the pair-map covariance; a separate delta_C must target the
REMAINING error after that response under fixed means/head conventions.
Otherwise the same correction is counted twice.

Current409f computes response-selected CP rows plus star/trace and a full
diagonal. It does not supply a full physical K21 plane for free. A full
plane input needs its extra CP read, or the sidecar must explicitly use
the already partially read input. Heads, masks, star/radial ownership,
covariance recentering and altered future births all remain paid.

### Available signal and its limitations

The older qualified rank35 local oracle ratios are.3275 for true
off-diagonal K3 and.0209 for joint K3/K4 next-mean error. They establish
privileged value, not a target-free predictor or409f final-score forecast.
The independent covariance-label noise audit found6.09% covariance and
9.23% next-mean noise; noise alone does not explain the weak predictors.

Eight older8M covariance trajectories use previous true mean/C. Public
100M MINI moment metadata instead supplies pre M11/M21/M31/M22 and post
M11. Its fixed train split is57,68,45,59,3,12 and holdout14,71, disjoint
from FULL1000 and virgin6. ONLY MINI57 was downloaded and qualified.
Its K21 storage-rounding bound is.00028057 of missing-plane RMS; that is
not a late sampling/shared-bias certificate. Its saved bank frame is old
P0 (166b...), NOT current409f/all-P. The seven other files remain unopened.
The old endpoint-span closeout does not invalidate the dataset, but does
close that dictionary/objective.

For k full latent matrices across15 transitions, ordinary congruence
transport alone costs64.42450944k billion FP32 operations. A two-channel
additive sidecar therefore adds128.84901888B to466.058731668B before all
other work. This classical circuit subtotal is neither a measured bill
nor a lower bound against fast products. Under a UNIFORM-cost yardstick,
transport alone demands approximately22% raw improvement to break even,
and approximately51% for3e-9. No such supplier result is available.

Decision: do not launch another fit. A current-reference MINI57 capture
could qualify inputs/readers, but a perfect substitution or in-sample fit
would be another oracle diagnostic, not cross-net prediction evidence.
The prerequisite is a specified causal nonlinear innovation with signed
current-head value outside the already-tested endpoint/history source
span. Neither novelty nor the old oracle percentage supplies it.

No frozen family, response-row exception, package, root estimator, forum,
release or promotion was changed. No numerical worker was started by this
audit. The active goal is unmet, not externally blocked.

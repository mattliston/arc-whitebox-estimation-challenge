# Shared-vertex covariance innovation: fixed information round

2026-09-05, Phase2. Current goal<1e-9; no claim this reaches it.

Prior covariance_innovation_probe_v2 used 315 coefficients on PAIR-LOCAL
F_i Theta F_j times rho_ij^r, r0..2. It did not sum over a third vertex.
The late rank8 congruence fit optimized a state write through mean loss,
and also failed. Neither is retrained or enlarged here. The old K3-squared
bivariate Edgeworth density round is closed. This new36-coefficient
dictionary uses full shared-intermediate matrix products, not those
pair-local or low-rank endpoint kernels alone.

Inputs reuse the frozen coherent8M oracle-state source rows from the prior
covariance pilot:6 train nets35,100,150,200,250,300; held out400,500;
layers8..14. The prior mean/covariance truth feeds have correlated sampling
noise; no independent-replica ceiling is claimed. Current features were
captured BEFORE current truth feedback. No new MC, private targets or
candidate rollout. This diagnoses a source on an oracle input distribution.

Let R be pre-correlation with diagonal zero, F=(g,phi,alpha*phi), and
ms=host_mean/pre_sigma. Three fixed middle weights are 1,
h2=2g-2ms*phi, and h3=2phi+2ms*alpha*phi. Build

    M_b = R diag(h_b) R.
    Ehat_ij = sigma_i sigma_j sum_(b=0..2,p=0..1)
              M_b,ij R_ij^p [F_i^T Theta_(b,p) F_j].

Each Theta is symmetric3x3:36 total coefficients shared across layers.
No rank, basis, depth, feature or seed sweep. Marginal diagonals of M are
kept; R^1 diagonal is zero. The desired covariance update need not be PSD
by itself; deployable moment feasibility remains a separate future gate.

Use an EXACT36-dimensional regularized normal-equation solve, not an
optimizer/epoch sweep. Objective is covariance MSE divided by TRAIN target
energy plus next-layer mean-response MSE divided by TRAIN response energy.
Each column is rescaled by its combined TRAIN Gram diagonal; ridge1e-3
on those normalized columns. Fit only train summaries. Evaluate held-out
400/500 once. Require BOTH error ratios<=.75 on EACH held-out net. Failure
closes this fixed dictionary with no more features/epochs/ridge choices.

Price actual deployed work before numeric labels: three full f32 GEMMs,
six thin endpoint-matrix products and pointwise assembly per used layer,
no teacher contractions in deployment. Require<=10% extra bank FLOPs.
Small/f64 scalar operations priced explicitly. First feature-only numeric
cell must have f32-vs-f64 M_b relative error<=1e-5. Full estimator precision,
source-state exposure, mean-head matching,14-net panel, native metering and
wall/residual/memory gates remain mandatory if this information test passes.

Training response Jacobians and Gram summaries are offline diagnosis only.
Store summaries/provenance, never overwrite previous pilot artifacts.
One nice10 three-thread CPU job; no MPS, forum writes or submissions.

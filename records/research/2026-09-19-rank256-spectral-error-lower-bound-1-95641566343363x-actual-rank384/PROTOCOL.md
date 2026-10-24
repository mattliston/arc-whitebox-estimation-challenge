# Rank-256 next-tensor lower bound

Phase 2 only. This is an additive capacity diagnostic after the failed
full_tensor_exact_gram_information_v1 experiment. It constructs no alternate
basis or candidate, does not relax that experiment's gates, and uses no
mean-reference labels. Official Lazy #331599 remains 3.8164492657795035e-9
adjusted; the approximately 1e-9 objective is active.

Use exactly the previous sealed rank35, entry-layer8 join packet, SHA256
adc532fcbd06c9c9f0de8636067a168ee42b0385089958e47bd0f79ec7aa63e6.
Let R=W.T [Q F,J], in FP64 from saved rounded arrays, define
T=sum lambda_c R_c^tensor3, and let T0 be the tensor from the actual saved
rounded parent factors. No new estimator call or packet copy is needed.

For a mode-one unfolding A of T, G=A A.T. For ANY tensor X whose mode-one
rank is at most256, with an orthonormal basis U for its column space,

    ||T-X||_F^2 >= ||(I-U U.T) A||_F^2
                 = trace(G)-trace(U.T G U)
                 >= sum of all but the largest256 eigenvalues of G.

This includes arbitrary rank256 three-leg subspaces and arbitrary cores;
it is broader than choosing a particular basis for the existing CP columns.
Independent standard-normal a,b also satisfy
E||T(I,a,b)-X(I,a,b)||^2=||T-X||_F^2. Compare the bound with the EXACT
Frobenius error of T0, rather than a finite random estimate.

This tests the corresponding expected isotropic-query criterion. It is
not an impossibility result for a selected finite probe set, task-weighted
queries, repeated-index planes, estimator means, different states, or all
possible cheap representations. It supplies no board-score forecast.

Compute cross-mode Grams directly from signed CP atoms:

    cross(P,a; S,b) = (P diag(a)) ((P.T S)^2) (S diag(b)).T.

Then G=cross(R,lambda;R,lambda) and
Gerr=G+cross(P0,lambda;P0,lambda)-cross(R,lambda;P0,lambda)
     -cross(R,lambda;P0,lambda).T.
Use symmetrization only after measuring relative asymmetry. The parent
error is trace(Gerr). No eigenvalue clipping, regularization, parameter
fit, rank sweep, new layer/network, or alternative basis is permitted.

First independently certify both cross Grams, the residual Gram, the
spectral tail, and its attained matrix-rank optimum against dense small
tensors/SVD. Include ordinary signed, a perturbed parent, and exact-zero
residual fixtures; seed2026092011. Scaled errors <=1e-11, null exact.

Real-packet numerical qualification: finite arrays; Gram asymmetry <=1e-10;
reference minimum eigenvalue >=-1e-10 times its maximum; residual minimum
eigenvalue >=-1e-8 times its maximum; full eigen residual <=1e-10. Check
eight fixed Gaussian v, seed2026092012: v.T Gerr v must agree to relative
1e-8 with ||T(v,I,I)-T0(v,I,I)||_F^2, computed by separate contractions.
Also compare trace(Gerr) to independent signed-cubic Gram scalar sums to
relative1e-8. These checks detect substantial cancellation/order errors;
they are not a formal interval-arithmetic rounding proof.

Predeclared decision: if the spectral tail exceeds 0.75 times exact parent
error by a relative margin at least1e-4, report numerically supported
infeasibility of the expected-query criterion for rank256 on this modeled
state. Otherwise report that this bound is inconclusive. No other rank or
candidate experiment follows within this protocol. The same512 prior probes
may be compared only as descriptive historical estimates, not a new gate.

One nice10/BLAS1 group. Hard8GB process-tree RSS; zero added swap required.
Unit30s, real certificate180s, new files<=40MB. Retain only the two1024-square
Grams and eigenvalues; never save the8128-square intermediates. No MPS,
sampling of network paths, training, download, accuracy panel, or upload.

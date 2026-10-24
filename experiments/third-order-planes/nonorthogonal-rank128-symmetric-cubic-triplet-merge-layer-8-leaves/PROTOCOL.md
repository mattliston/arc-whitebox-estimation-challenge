# Fixed nonorthogonal triplet compression information screen

Phase2 only. Prospective 2026-09-20 UTC, before this study decodes arrays.
Official Lazy#331599 remains3.8164492657795035e-9 adjusted; approximately1e-9
goal active. No estimator, archive, approval request or upload is part of this
screen. All previously closed constructors remain closed.

New representation:128 general real triples in the retained384-dimensional
shared space, T_hat=sum_r Sym(a_r outer b_r outer c_r), with Sym the average
of all six permutations. This uses384 stored factor columns and retains mixed
cubic terms; no orthogonality or equality between the three factors is imposed.
It is different from the failed384 orthogonal cubes and the rank256 physical
mode projection. It is not a fitted rescue/checkpoint of those constructors.
No general CP-rank or capacity conclusion follows from this fixed algorithm.

Source: Kolda, Numerical Optimization for Symmetric Tensor Decomposition,
arXiv1410.4536v2, sections3.1 and4.7. The paper discusses fitting unrestricted
factor matrices to symmetric data. Here the constructed result is explicitly
symmetrized; convergence to identical factors is neither required nor assumed.
https://arxiv.org/pdf/1410.4536

Use exactly the same sealed Lazy rank35 ENTRY8 OLD-prefix packet
adc532fcbd06c9c9f0de8636067a168ee42b0385089958e47bd0f79ec7aa63e6:
Q896x384, F384x6144, lambda[:6144], recorded W896x1024. Exclude the entering
1984 atoms and all star/covariance-trace state. Reduced QR Q=U R and A=R F
put the original stored tensor into physical orthonormal coordinates.

Fit the UNSYMMETRIZED rank128 CP model to that symmetric tensor with exactly
two A/B/C Gauss-Seidel ALS sweeps in FP64. Seed PCG64DXSM2026092005.
Initialize B and C independently from reduced QR of384x128 standard-normal
draws. Initialize A=0; the first A update is determined by B,C and the data.
For an A update, M=F diag(lambda)[(F.T B)*(F.T C)] and
G=(B.T B)*(C.T C). Solve A(G+rho I)=M, rho=1e-8*trace(G)/128.
Use analogous cyclic updates. Every trace/rho must be positive and finite;
require relative normal-equation residual<=1e-9. No pseudoinverse, rank repair,
precision change or adaptive regularization. Cache each F.T factor product
until that factor changes, and charge every actual construction product.
After each full sweep, balance each nonzero triple to equal column norms
by geometric-mean scaling (product of scales1); update cross-product caches.
Exactly the second endpoint is used; no loss-based early stop or checkpoint
choice. Final representation is the symmetrized endpoint, never the raw
unsymmetric tensor. No target/reference/probe data informs initialization.

Construction scope and cost premise: at6144 old atoms, the fixed two-sweep
core requires fourteen6144x384x128 conventional products including initialization,
about8.456GF FP32-leading arithmetic or16.911GF FP64-weighted, before QR,
Gram matrices, solves, balancing, buffers and physical reconstruction. This
can only plausibly amortize over later reads; it is NOT a complete estimator
saving, and fresh refits at every layer are not selected. A positive fidelity
result would require separately pricing a single middle-layer compression,
maintaining the parent's weighted-Gram frame state, future cohorts and all
physical owners. No prospective deployment cost or accuracy gain is promised.

Tiny independent tests precede the packet: factored MTTKRP/normal equations
against materialized signed symmetric tensors; full two-sweep endpoint against
a dense implementation; explicit six-permutation symmetry and repeated-plane,
diagonal/two-leg query contractions; balancing invariance; zero and mixed-sign
factors. Scaled error<=1e-10. No convergence or rank-recovery test is asserted.

On the packet, verify QR reconstruction/orthogonality<=1e-10, finite matrices,
all solves and nonzero column norms. Save/seal all three physical factor
matrices BEFORE making verification probes. Reuse exactly512 PCG64(2026092004)
Gaussian pairs used by the previous tensor diagnostics, and compare the complete
1024x1024 repeated plane after the recorded W push. These probes are reused
verification data, not independent generalization evidence.

Advance only if residual squared energy / original OLD signal<=.01 for BOTH
the two-leg queries and complete repeated plane. Report diagonals and full
symmetrized/unsymmetrized tensor error separately using signed contractions.
No means/physical truth/learned head/target labels. No rank, seed, sweep count,
ridge, symmetry constraint, initialization or checkpoint rescue on failure.
Passing would license a separate native-cost/ownership design, not a bank.

Record all conventional FP64 GEMM-leading subtotals, including cached-product
maintenance and qualification/diagnostics separately. These omit QR/solve/
elementwise/buffer charges and are not complete FNP deployment bills. Avoid
materializing the384-cube tensor or6144-square Gram. Store at most12MB of new
artifacts. One nice10 BLAS1 guarded process at a time,8GB RSS,zero added swap.
Unit30s; actual screen180s. No model calls, new samples, downloads or mean fits.

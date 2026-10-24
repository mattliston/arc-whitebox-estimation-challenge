# Fixed orthogonal cubic merge inside the retained shared space

Phase2 only. Written before new array decoding or numerical work. Previous
coherent-variance endpoint is closed negative. Official Lazy#331599 remains
adjusted3.8164492657795035e-9; the approximately1e-9 goal remains active.

Question: can the accumulated OLD tensor be merged into384 orthogonal
cubic atoms without shrinking its384-dimensional physical span? This is
different from rank256 mode projection, complete-pair coefficient selection,
random hashing and independent thinning. Those adverse results remain.

Use the existing exact-Lazy rank35 ENTRY8 packet, SHA256
adc532fcbd06c9c9f0de8636067a168ee42b0385089958e47bd0f79ec7aa63e6.
Only its OLD prefix is tested: Q shape896x384, F shape384x6144,
lambda[:6144], and the recorded W shape896x1024. The entering1984 columns,
coordinate star and covariance trace are not merged. This is the exact
formal old tensor represented by the stored rounded Q/F BEFORE the entry8
join, not physical truth or the original uncompressed trajectory.

In FP64, reduced QR gives Q=U R, then A=R F. Fix the single direction
a=U.T W[:,0], normalized to unit length. Construct the symmetric slice

    S = A diag(lambda * (A.T a)) A.T.

Check roundoff asymmetry before symmetrizing. Use ALL384 eigenvectors V
of S, with no truncation, whitening, eigenvalue clipping or second slice.
Let B=U V and c_i=sum_j lambda_j*(V.T A)_ij^3. The merged tensor is

    T_hat = sum_i c_i B_i^tensor3.

For a truly orthogonally decomposable tensor with a separating slice this
recovers its components. For a general tensor it is only the orthogonal
projection onto the384 selected cubes; no general decomposition claim.
The independently derived identity is ||T-T_hat||_F^2=||T||_F^2-sum c_i^2.
Physical linear span is retained, but its mixed cubic coefficients may be
lost. This is the unresolved, testable premise.

Primary reference: Kolda, *Symmetric Orthogonal Tensor Decomposition is
Trivial*, arXiv1503.01375v1, Section3. The paper's exact recovery assumption
is not assumed for these factors. No general tensor-rank optimality claim.
https://arxiv.org/pdf/1503.01375

Tiny independent dense tests at widths3/5/7: exact orthogonal recovery;
general signed and near-cancelling tensors; exact cancellation; slice,
diagonal coefficients, Frobenius projection identity, two-leg queries and
repeated plane against materialized tensors. Scaled errors <=1e-11.

On the actual packet require finite inputs/outputs; QR reconstruction,
orthogonality, slice symmetry and eigensystem residual <=1e-10. Require
finite nonzero direction, full-rank QR, and nonnegative resolved squared
norms. No tolerance, slice or precision repair after failure. Save and seal
B/c before constructing any verification queries.

Push both old P=QF and merged B through the same recorded W. Compare
T(I,A_probe,B_probe) on the512 PCG64(2026092004) Gaussian probe pairs already
used by the older diagnostics, and the COMPLETE1024x1024 repeated plane.
These reused verification probes never construct or select the merge.
Primary gate: residual squared energy / original OLD tensor signal <=.01
on BOTH consumers. Report cubic diagonals and full tensor norm separately.
This fidelity screen is not a raw mean score or an independent holdout.
Passing only justifies separately pricing and qualifying a full recursive
representation; it cannot authorize submission on tensor fidelity alone.

Failure closes this prescribed one-slice384-atom orthogonal constructor.
No slice, seed, network, rank, cut, whitening, nonlinear fit, trace add-on,
or source rescue; no production candidate, panel, archive or upload.
It is not a bound on unrestricted nonorthogonal CP decompositions.

Cost: record all conventional FP64 GEMM leading subtotals, separating
construction from diagnostics; QR/eigh/rank-SVD/GEMV/elementwise/buffer/copy charges are
not included in that subtotal. It is NOT a complete FNP deployment bill.
No reduction in current-card cost is claimed without an implemented meter.

One nice10 BLAS1 guarded process at a time;8GB RSS maximum and zero added
swap required. Unit30s, information180s. No model calls, new samples, mean
reference labels, training, dataset downloads, or large persistent arrays.
Do not save the6144-square diagnostic Gram. New artifact cap12MB.

# Exact full-tensor Gram: fixed information screen

Phase 2 only. The accepted Lazy bank is #331599, adjusted
3.8164492657795035e-9. The approximately 1e-9 objective remains active and
qualified board submissions remain preauthorized.

The previous full signed-tensor random sketch at rank 256 failed. This
separate experiment removes its random approximation, using the same fixed
rank 256 and the same saved rank-35, layer-8 join. It does not reopen or
reinterpret the failed sketch, pair-metric, or Krylov experiments.

For the captured modeled tensor T=sum_c lambda_c p_c^tensor3, form its
complete mode-one Gram, including every signed cross-atom term:

    H = P.T P
    G = (P diag(lambda)) (H * H) (P diag(lambda)).T.

Use the leading 256 eigenvectors of the symmetrized FP64 G. Symmetrization
only removes roundoff asymmetry after it has been checked. No eigenvalue
clipping, ridge, response weighting, fitting, power iteration, rank search,
or alternate seed is permitted. The top eigenspace maximizes retained
one-mode Frobenius energy; it is not claimed to minimize the error after
three-leg projection, the next repeated-index plane, or estimator mean MSE.

Reuse the sealed K/capture/join.npz packet from
full_tensor_sketch_information_v1, SHA-256
adc532fcbd06c9c9f0de8636067a168ee42b0385089958e47bd0f79ec7aa63e6.
That observer reproduced all 16 original Lazy output arrays and the full
398080917697-FLOP cold bill exactly. Here no estimator is executed and no
mean-reference labels or weights dataset are opened. Reconstruct P=[Q F,J]
from the saved rounded factors in FP64. This is modeled state, not physical
truth or the original uncompressed trajectory. The comparison is to the
actual rounded parent rank-384 factors in the same packet.

First certify the Gram against independently materialized dense small
tensors, its eigenspace energy against the unfolded dense tensor, and the
three-leg contraction and repeated-index plane. Include ordinary signed,
near-canceling, and exact cross-pair cancellation cases; require scaled
errors <=1e-11. Fixed unit RNG seed 2026092009. Exact cancellation must be
zero while the sum of the two complete-pair Grams is diag(2,2,0).

On the real packet, require all finite values; relative Gram asymmetry
<=1e-10; minimum eigenvalue >=-1e-10 times maximum eigenvalue; normalized
eigendecomposition residual and rank-256 orthogonality <=1e-10. Independently
check eight fixed Gaussian vector quadratic forms against explicit two-leg
contractions T(v,I,I), seed 2026092010. Require each relative discrepancy
<=1e-8. These check numerical cancellation, and do not select the basis.
Failure of a numerical check closes the run without an information verdict.

Seal the chosen basis before reading verification probes. Project P on all
three legs, then push through the saved affine W. Compare the candidate and
the saved parent with the same uncompressed modeled tensor. Use exactly
512 verification probes, PCG64 seed 2026092004, and the complete next-layer
T(i,i,j) plane. These probes were already used in the old sketch diagnostic;
they are not a fresh holdout, and they do not enter basis construction.
Require candidate squared residual energy <=0.75 of parent on BOTH queries
and the complete plane. Report the diagonal separately, without a gate.
Reproduce the previous diagnostic's parent residual energies to relative
1e-10 as an independent packet/comparison check.

This is an optimistic FP64 information test, not a production candidate.
Dense products for this construction are expensive and eigendecomposition
is unpriced. Record a conventional GEMM arithmetic subtotal, explicitly not
a flopscope bill. A pass still requires a separately specified affordable
legal implementation and native endpoint validation. A failure closes this
fixed exact rank-256 construction before another rank, join, network,
precision, basis criterion, implementation, accuracy panel, or submission.

One serial nice-10 BLAS-1 process group at a time. Hard 8 GB process-tree RSS
guard; qualification requires zero added swap. Unit limit 30 seconds;
information limit 180 seconds. New artifacts <=40 MB; do not copy the packet
or save the 8128-square intermediate Gram. Preserve all prior artifacts.

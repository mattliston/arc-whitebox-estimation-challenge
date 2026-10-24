# One-slice orthogonal cubic merge: fixed information gate fails

Phase 2 only. The fixed constructor retains the old 384-dimensional physical
span but replaces its 6,144 signed cubic atoms with 384 orthogonal cubes.
The eigensystem comes from one recorded weight-column slice. It is sealed
before all reused verification probes. This tests the OLD prefix of the
saved current-Lazy rank35 entry8 state, before its next join.

Both prescribed squared residual/signal gates of <=0.01 FAIL:

| Diagnostic | Squared residual / original signal |
|---|---:|
| 512 Gaussian two-leg queries after the recorded weight push | 0.9540261570726674 |
| Complete 1024 by 1024 repeated-coordinate plane after that push | 0.8740800687347923 |
| Cubic diagonal, descriptive only | 0.8504987027021403 |
| Full old tensor before that push, all signed cross terms | 0.964405347741994 |

The full old tensor norm squared is 21.143643621508645; the retained
orthogonal-cube energy is 0.7526006421748065. Thus keeping the linear span
does not preserve this tensor's mixed cubic coordinates. The signed norm
calculation includes every cross-atom term; absolute/signed sum is
31.20866772543698. These are modeled-state fidelity results, not final mean
MSEs, deployment costs, or estimates of the archive's official score.

Twelve independent small dense fixtures pass, including signed/general,
near-canceling, exactly zero and orthogonally decomposable tensors. The
largest scaled dense identity error is 4.541183937712605e-12. On the saved
packet the largest QR/orthogonality/symmetry/eigensystem residual is
3.2277073571993677e-15. No numerical or resource failure masks the result.

Both single-thread guarded runs completed and retired, totaling
2.6365506649017334 seconds. Peak including worker self-report is
600621056 bytes and added swap is zero. No new estimator calls, mean
references, samples, training or dataset downloads were used. The fixed
merge is about 2.76 MB. No full 6144-square diagnostic Gram was saved.

Conventional FP64 GEMM leading billed subtotals: construction 11,400,118,272;
qualification 1,283,457,024; diagnostics 158,846,681,088. QR, eigendecomposition,
rank-SVD, GEMV, elementwise, buffers and copies are excluded. These are
explicitly incomplete research subtotals, not complete FNP deployment bills.

Close only this one-slice, 384-atom orthogonal constructor. No different
slice, seed, rank, whitening, nonlinear fit, trace repair or source rescue
is selected. This does not bound unrestricted nonorthogonal CP approximations
or final mean error. No production candidate, panel, client, archive or upload.

Official best remains graded Lazy #331599: adjusted 3.8164492657795035e-9,
raw 2.169264529783277e-8. The 4e-9 bank is achieved; 3e-9, 2e-9 and the
approximately 1e-9 goal remain unmet and active. Qualified Phase 2 submissions
remain preauthorized.

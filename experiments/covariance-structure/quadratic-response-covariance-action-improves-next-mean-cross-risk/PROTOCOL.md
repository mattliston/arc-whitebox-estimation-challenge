# Fixed current-P covariance-consumer witness

September 8, 2026 PDT. PHASE 2 only. One prospective local no-fit test,
rank35/post12, amplitude1. No new model, carrier, training, sampling,
strength/layer sweep, candidate, upload, or changes to frozen work.
Prior current-head numerical qualification52938606..., gate-information
b549ad49..., and fixed synthetic covariance algebra01802bd2... all passed.
Their full hashes are pinned in check.py. Gate information was13.34%/12.06%
lower cross-MSE than the host on rank35/100, but the margins beyond the
10% threshold are only about1.04/.52 paired jackknife standard errors.
No strong-confidence10% claim or final raw gain follows from that proxy.

## Fixed prediction, before any target array is opened

Use the sealed current409f rank35/layer12 and13 capture, the authenticated
canonical rank35 weights, and the saved qualified response vectors.
a=delta_derivative[post12], b=.5*delta_second_derivative[post12]; these are
P-correction derivatives only. Do not include the host derivative, a
constant mean correction in the new observable, or any current-target fit.

The unchanged frozen FP64 covariance reference constructs DeltaC. All raw
incoming variances must be positive and >=1e-10; reject, never repair.
The fixed current P mean correction d is retained, and the baseline is

    C0=M(Chost-outer(mhost,d)-outer(d,mhost)-outer(d,d)),
    M(X)=A X A+diag((1-alive)*diag(X)), A=diag(captured_alive),
    D=M(DeltaC).

Preserve the current subtraction order and ALL diagonals, including cold
ones. No new mask, head/features recomputation, target centering, PSD
projection or diagonal repair. No K21/K22/radial/star/CP/birth changes.
This is a covariance-only fixed local action, not a complete joint law.

Freeze W=canonical_weights[13] in FP64 and
omega=capture13.aux_pdf/(2*capture13.aux_sigma). The sole primary consumer is

    J(X)=omega*sum(W*(X@W),axis=0).

This is the declared Gaussian variance-tangent consumer. It is NOT the
rounded native covariance push, the full next learned head, or final MSE.
Seal C0,D,a,b,omega,J(D) and corrected current mean in an exclusive
prediction NPZ before opening either covariance target. Confirm target-free
next-mean frame consistency from captured next mu and W.T*(mhost+d).

## Existing targets and provenance qualification

Use unmasked mc_post_cov[12] from the two already-existing8M sources:

- A: public_g19_attribution_v1/mps_cov2_rank0035.npz,
  SHA7e6b97166885de0cdd870a835ae536830fb6da41ca345d8a66e2eb8d0d6380e6.
- B: covariance_replica_truth_v1/rank_0035_8M_v1.npz,
  SHAe6a7787d974c6b56f39a4444bd54c793ff3efa6e62acb55054774ddb2a2e3358.

Each source stores covariance as mean(xx.T)-outer(samplemean,samplemean).
Convert the stored FP32 matrix to FP64, then multiply by8000000/7999999.
This removes ordinary finite-N covariance bias in ideal arithmetic, not
MPS-FP32 forward/accumulation, storage rounding or shared numerical bias.
Never recenter the truth around current means, mask it, substitute a
different layer or use old trajectory-dependent innovation arrays.

The replica's preserved source/JSON records xor0x4153545241434f56 and
PCG64DXSM CPU-F32 draws, batch131072, Torch2.7.0. Archived original-source
law has xor0x47414D454F4E5F31 and the same canonical rank35 entry, but a
contemporaneous original-generation source/seed/weight seal was NOT found.
Its current generator source mtime is later than the original NPZ. Record
this limitation explicitly; hash/rank checks cannot manufacture that seal.
The old executed replica audit asserted P==8M for both preserved archives.
No new data acquisition or historical execution claim is made here.

On authorized loading, check pinned bytes, scalar rank35/P8000000,
FP32(16,1024,1024) post covariance, finite means/covariances and shape.
As a fixed additional weights-consistency guard, on BOTH sources require
each stored pre mean at layers1..15 to agree with
W[layer].T@stored_post_mean[layer-1] with Euclidean error
<=1e-8+1e-5*norm(stored_pre_mean). This checks the affine identity on the
actual saved streams; it is not claimed to recover a historical hash seal.
No tolerance adjustment or array repair if this guard fails.

## Fixed primary decision

Let eA=J(CA-C0), eB=J(CB-C0), s=J(D). Require finite, strictly positive
reference denominators and BOTH of the following:

    mean((eA-s)*(eB-s))/mean(eA*eB) <= .75,
    mean((es-s)^2)/mean(es^2) <= .85, separately for A and B.

Do not clip signed cross-products. Report individual stream errors,
signal/inner products, cross covariance risk as secondary information,
and the two-stream disagreement/noise scale. These reused development
truths have no saved independent covariance batches for a new jackknife;
do not invent confidence intervals or transfer old noise fractions.
Failure closes this fixed action: no scale, curvature, layer or threshold
rescue. A pass only justifies further design, not a final-score claim.

One nice10 OFFPY worker, BLAS envs1 and detected-pool checks, exact pinned
versions, no concurrent numerical/MPS/server process,60s watchdog,
peakRSS<1GB, exclusive outputs and all source/fixture/result hashes
before/after. Root seal and independent source review precede --run.

## Score route and release boundary

This is the strongest remaining raw route because current response
information passed a fixed two-network test and its local mathematical
implementation now passes numerical checks. Expected FINAL improvement
is unknown. Source arithmetic suggests roughly0.9-1.5B extra FLOPs per
affected head; a complete rollout bill and feedback ownership are still
missing. This local test is the cheapest decisive signed-consumer check.
No material new exact cost-saving circuit has a credible full-cost and
causal endpoint-error witness, so no substantial cost-only run is selected.

Before full integration, own K21/K22/radial/star/birth effects, null parity,
complete cost and runtime. Substantial advancement requires actual final
raw times EACH net's own complete scoring multiplier <=.90 parent on BOTH
public8 and virgin6. The1.08 cost ceiling is a rejection guard, not an
estimate. Preserve small validated components. Bank qualified below4/3/2e-9
milestones immediately, with sub3e-9 ASAP the immediate priority; official
final goal remains strictly below1e-9. No forum posts or commits.

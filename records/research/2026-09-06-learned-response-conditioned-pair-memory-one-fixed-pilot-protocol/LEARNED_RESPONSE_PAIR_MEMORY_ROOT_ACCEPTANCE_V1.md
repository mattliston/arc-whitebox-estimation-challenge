# Root acceptance — one learned response-conditioned pair-memory pilot

September6,2026, Phase2 only. Root has read the complete draft protocol,
SHA1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c,
and accepts its ONE10-parameter history/RESET law,100 epochs per arm,
Adam1e-3, seed930074, shared TRAIN permutations, declared penalty/gradient
cap, exact directed covariance/next-mean consumers and .50/.50/.80/3-of-4
held-out gates. All sources must be reviewed and frozen before execution.
This is not a reopening of the old fixed-H or reflected-copula arm.

## Numerical and resource prerequisites

1. One synthetic n7/history13 fixture, seed930174, fixed nonzero10-vector
   [0.17,-0.23,0.31,0.11,-0.13,0.19,0.07,-0.29,0.37,-0.41]. Both arms;
   NumPy64/TorchCPU64 outputs and fixed central differences(step1e-5)
   versus autograd <=1e-6 relative. MPS32 outputs versus NumPy32 and
   gradients versus FD-qualified CPU64 <=1e-4. Check response-unit
   conversion identity <=1e-12, latent symmetry/bounds and query sign parity.
   This is a numerical implementation certificate, not network information.
2. Prepare the existing authenticated rank35 causal data only, preserving
   source/receipt/weight/P/Q ownership. Separate causal input files from
   loss-only covariance coefficients, pre13 teacher and pre14 response.
   Freeze each causal file before decoding its teacher. No bank rerun or
   new MC sampling is needed. Rank35 is a TRAIN network.
3. One rank35 wide output/gradient numerical comparison of the same fixed
   nonzero parameter vector and both arms, without opening its loss file;
   NumPy64/TorchCPU64 and MPS32 within1e-4 relative. CPU64 autograd gradient
   can be reused as the synthetic-FD-qualified reference. No step/parameter/
   seed sweep. Failure is a numerical prerequisite, not a predictive result.
4. Only after those pass, prepare the remaining fixed23 rows and run the
   single paired20/4 fit. Freeze BOTH final models before any VALID loss
   is decoded by the trainer. No validation curves or checkpoint selection.

The stable real-equivalent spelling of psi is declared before real-state
access: t=1/(1+abs(x)); y=x*t; psi=y/sqrt(y*y+t*t). This avoids squaring a
large normalized skew product in FP32, without clipping or changing the
mathematical map. Its input must remain finite. Dataset qualification
must reject a d range whose pair products are nonfinite in FP32; it may
not repair d or adjust the fixed variance-unit floor. Cold variance counts
and all finite-unit checks are recorded, not hidden.

One nice10 NumPy preparation worker with two requested CPU threads and
<=6GiB RSS; one nice10 AP/MPS worker at a time, MPSfallback0 and no import
of threadpoolctl in AP. Training must remain <=6GiB process RSS, with
MPS allocated/driver memory reported separately. The paired fit's600s cap
starts after immutable data transfer and ends with both100th epochs;
check at every per-net update. Preserve an incomplete/resource failure
instead of silently resuming, increasing a cap or choosing an earlier model.
Data may be cached on MPS if the memory bound holds; computation is one
network at a time. No concurrent bank replay is permitted.

## Source interface and causal boundary

The pure model accepts only V[13,n,n], b[13,n,n], d[13,n], g[13,n],
post_scale[13,n], query_V[n,n], K[n,n], mask[n,n]. V contains PRE-unit
transports; post_scale=g*alive*p_pre/s_post converts after the shared
congruence. V0 is zero. b is the fixed psi(B) source, not a fitted target.
All these arise from post0..12/lower-order pre history; none is a pre13
third target or post13/current-head corrected field. RESET may execute
only the identical layer12 zero-history update, since all earlier results
are discarded. Coefficients on u and u*b are then inactive and reported.

The predictor never receives the loss dictionary. Loss coefficients retain
the exact directed rho/original-variance convention. The pre14 Gaussian
response is LOSS ONLY. This test predicts the bank's missing action, not
true final means; all lower-state/D history is teacher-forced. A pass does
not qualify an autonomous CP-free card, native costs or a raw-score gain.

No candidate submission, further ranks, widths, normalization/source/gain,
optimizer or epoch variants are authorized here. The prospective290B
whole-card ceiling is a budget, not a measured utilization. The official
bank is unchanged; sub1e-9 remains an unachieved active objective. The
separate response-row exception still awaits user approval.
